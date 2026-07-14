"""Concrete benchmark-v0 data workspace phases used by the public CLI."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from . import BENCHMARK_VERSION, HEADLINE_HORIZON, HEADLINE_MIN_LENGTH
from .corruption import CORRUPTION_GRID, LANE_OF_SCENARIO, replicates_for
from .datasets import DATASET_MANIFEST, dataset_manifest_document
from .materialize import (
    ParsedSeries,
    RawAsset,
    materialize_clean_base,
    parse_gefcom2012_load_zip,
    parse_metr_la_hdf,
    parse_metr_la_sensor_locations,
    parse_monash_parquet,
    parse_noaa_global_hourly,
    parse_uci_electricity_zip,
    read_clean_base,
    verify_raw_asset,
    write_text_lf,
)
from .metrics import UndefinedSeasonalScale, seasonal_scale
from .probe import probe_registry
from .registry import (
    Admission,
    SeriesRecord,
    admit_series,
    import_legacy_inventory,
    read_registry_jsonl,
    write_registry_jsonl,
)
from .sources import METR_LA_SPATIAL_BLOCKS, SOURCE_SPECS
from .programs import pool_manifest
from .spatial import block_diagnostics, build_spatial_blocks
from .split import SplitManifest, build_split_manifest, build_support_a_subsplit

MAX_CLEAN_LENGTH = 1024
SPLIT_SALT = "benchmark-v0-split-salt-v1"
MAX_NATURAL_MISSING_RATE = 0.30
MAX_IRREGULAR_SAMPLING_RATE = 0.05
REGIME_THRESHOLDS = {"high": 0.60, "moderate": 0.35}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_raw(path: Path) -> RawAsset:
    sidecar = path.with_name(path.name + ".asset.json")
    payload = json.loads(sidecar.read_text("utf-8"))
    return verify_raw_asset(
        RawAsset(path, payload["sha256"], payload["source_revision"], payload["size"])
    )


def _probe_consumed_traffic(project_root: Path) -> set[str]:
    path = project_root / "results" / "Stage2" / "P6Probes" / "u_admission_v2_traffic_hourly.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text("utf-8"))
    values = payload.get("all_probe_consumed_item_ids", [])
    return {str(value) for value in values}


def metr_la_blocking(root: Path):
    """Build the frozen METR-LA spatial blocking from the pinned coordinate asset."""
    locations = root / "raw" / "metr_la" / "graph_sensor_locations.csv"
    if not locations.is_file():
        return None, None
    _bound_raw(locations)
    coordinates = parse_metr_la_sensor_locations(locations)
    blocking = build_spatial_blocks(coordinates, n_blocks=METR_LA_SPATIAL_BLOCKS)
    return coordinates, blocking


def _automatic_parsed(
    root: Path,
    *,
    min_length: int,
    horizon: int,
    max_length: int,
    metr_blocking: object | None = None,
) -> Iterable[tuple[str, str, ParsedSeries, RawAsset, str]]:
    """Yield (source_id, dataset_id, parsed, raw_asset, overlap_group).

    The overlap group is the atomic outer-split unit.  It is the series itself for
    every source whose entities are independent, but for METR-LA it is the *spatial
    block*: co-located freeway sensors are not independent series and must never
    straddle Support and Query.
    """
    monash = root / "raw" / "monash_hf"
    for path in sorted(monash.glob("*/test/0000.parquet")):
        config = path.parent.parent.name
        asset = _bound_raw(path)
        for row in parse_monash_parquet(
            path,
            config=config,
            min_length=min_length,
            horizon=horizon,
            max_length=max_length,
        ):
            yield "monash_hf", f"monash:{config}", row, asset, f"{config}:{row.entity_id}"

    metr = root / "raw" / "metr_la" / "metr-la.h5"
    if metr.is_file():
        if metr_blocking is None:
            raise RuntimeError(
                "METR-LA is materialized but its pinned sensor coordinates are missing; "
                "without them the split cannot block co-located sensors and would leak"
            )
        asset = _bound_raw(metr)
        for row in parse_metr_la_hdf(
            metr,
            min_length=min_length,
            horizon=horizon,
            max_length=max_length,
        ):
            block = metr_blocking.get(row.entity_id)
            if block is None:
                raise RuntimeError(
                    f"METR-LA sensor {row.entity_id!r} has no pinned coordinate; "
                    "refusing to fall back to a per-sensor split"
                )
            yield "metr_la", "metr_la", row, asset, f"metr_la:block_{int(block):02d}"

    uci = root / "raw" / "uci_electricity_load_diagrams" / "electricityloaddiagrams20112014.zip"
    if uci.is_file():
        asset = _bound_raw(uci)
        for row in parse_uci_electricity_zip(
            uci,
            min_length=min_length,
            horizon=horizon,
            max_length=max_length,
        ):
            yield (
                "uci_electricity_load_diagrams",
                "uci_electricity_load_diagrams",
                row,
                asset,
                f"uci_eld:{row.entity_id}",
            )

    noaa_root = root / "raw" / "noaa_global_hourly"
    for path in sorted(noaa_root.glob("[0-9][0-9][0-9][0-9]/*.csv")):
        asset = _bound_raw(path)
        try:
            row = parse_noaa_global_hourly(
                path,
                min_length=min_length,
                horizon=horizon,
                max_length=max_length,
            )
        except ValueError:
            continue
        yield "noaa_global_hourly", "noaa_global_hourly", row, asset, f"noaa_isd:{row.entity_id}"

    gefcom_root = root / "incoming" / "gefcom2012"
    gefcom_archives = sorted(
        path for path in gefcom_root.glob("*.zip")
        if not path.name.endswith(".asset.json")
    )
    if len(gefcom_archives) > 1:
        raise RuntimeError("GEFCom2012 incoming directory contains multiple ZIP assets")
    if gefcom_archives:
        path = gefcom_archives[0]
        asset = _bound_raw(path)
        for row in parse_gefcom2012_load_zip(
            path,
            min_length=min_length,
            horizon=horizon,
            max_length=max_length,
        ):
            yield (
                "gefcom2012",
                "gefcom2012_load",
                row,
                asset,
                f"gefcom2012_load:{row.entity_id}",
            )


def _regime(features: dict[str, float | int]) -> str:
    seasonal = float(features["seasonal_strength"])
    trend = float(features["trend_strength"])
    if seasonal >= REGIME_THRESHOLDS["high"]:
        return "seasonal_high"
    if trend >= REGIME_THRESHOLDS["high"]:
        return "trend_high"
    if max(seasonal, trend) >= REGIME_THRESHOLDS["moderate"]:
        return "structured_mixed"
    return "low_structure"


def _admission_with_scale(
    record: SeriesRecord,
    values: np.ndarray,
    *,
    period: int,
    min_length: int,
    min_scale_pairs: int,
) -> Admission:
    base = admit_series(
        record,
        min_len=min_length,
        allowed_frequencies={"hourly", "daily", "monthly"},
        max_natural_missing_rate=MAX_NATURAL_MISSING_RATE,
        max_irregular_sampling_rate=MAX_IRREGULAR_SAMPLING_RATE,
    )
    reasons = list(base.reasons)
    train = values[: len(values) - 2 * HEADLINE_HORIZON]
    try:
        seasonal_scale(
            train,
            np.isfinite(train),
            period=period,
            min_pairs=min_scale_pairs,
        )
    except (UndefinedSeasonalScale, ValueError):
        reasons.append("undefined_seasonal_scale")
    reasons = list(dict.fromkeys(reasons))
    return Admission(
        eligible=not reasons,
        reasons=tuple(reasons),
        natural_missing_rate=record.natural_missing_rate,
        irregular_sampling_rate=record.irregular_sampling_rate,
    )


def _probe_with_cache(
    data_root: Path,
    records: list[SeriesRecord],
    values_by_uid: dict[str, np.ndarray],
    timestamps_by_uid: dict[str, np.ndarray],
    period_by_uid: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    cache_root = data_root / "probe_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    output: dict[str, dict[str, float | int]] = {}
    for row in sorted(records, key=lambda item: item.series_uid):
        path = cache_root / f"{row.series_uid}.json"
        if path.is_file():
            payload = json.loads(path.read_text("utf-8"))
            if (
                payload.get("schema_version") not in {
                    "benchmark-probe-cache/1",
                    "benchmark-probe-cache/2",
                }
                or payload.get("content_sha") != row.content_sha
                or not isinstance(payload.get("features"), dict)
            ):
                raise RuntimeError(f"probe cache binding is invalid: {path}")
            cached = dict(payload["features"])
            cached.update(
                {
                    "natural_missing_count": row.natural_missing_count,
                    "natural_missing_rate": row.natural_missing_rate,
                    "irregular_interval_count": row.irregular_interval_count,
                    "irregular_sampling_rate": row.irregular_sampling_rate,
                }
            )
            output[row.series_uid] = cached
            continue
        timestamps = (
            {row.series_uid: timestamps_by_uid[row.series_uid]}
            if row.timestamps_sha is not None
            else {}
        )
        result = probe_registry(
            [row],
            {row.series_uid: values_by_uid[row.series_uid]},
            timestamps_by_uid=timestamps,
            period_by_uid={row.series_uid: period_by_uid[row.series_uid]},
        )[row.series_uid]
        payload = {
            "schema_version": "benchmark-probe-cache/2",
            "series_uid": row.series_uid,
            "content_sha": row.content_sha,
            "features": result,
        }
        temporary = path.with_suffix(".json.partial")
        write_text_lf(
            temporary,
            json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n",
        )
        temporary.replace(path)
        output[row.series_uid] = result
    return output


def probe_workspace(
    root: Path | str,
    out: Path | str,
    *,
    include_legacy: bool = True,
    min_length: int = HEADLINE_MIN_LENGTH,
    horizon: int = HEADLINE_HORIZON,
    max_length: int = MAX_CLEAN_LENGTH,
    min_scale_pairs: int = 32,
    work_root: Path | str | None = None,
) -> dict[str, object]:
    """Materialize immutable clean bases, run the loss-free probe, and write registry.

    `root` holds the immutable source layer (`raw/`, `incoming/`), which is shared by
    every benchmark version.  `work_root` holds the *derived* clean base, and defaults to
    `root`.

    They are separable because `clean_base/record.json` carries protocol decisions -- most
    importantly `overlap_group`, the atomic split unit -- and a protocol decision can
    legitimately change between benchmark versions while the underlying bytes do not.  A
    version that redefines an overlap group (v0.1 makes METR-LA's atomic unit the spatial
    block rather than the individual sensor) therefore needs its own derived layer; writing
    it over v0's would either corrupt a sealed artifact or trip the immutability guard.
    The probe cache stays under `root`, because probe features are a function of the values
    alone and are version-independent.
    """

    if horizon != HEADLINE_HORIZON and include_legacy:
        raise ValueError("production probe must use the frozen headline horizon")
    data_root, output = Path(root), Path(out)
    derived_root = Path(work_root) if work_root is not None else data_root
    output.mkdir(parents=True, exist_ok=True)
    clean_root = derived_root / "clean_base"
    project_root = Path(__file__).resolve().parents[1]
    values_by_uid: dict[str, np.ndarray] = {}
    timestamps_by_uid: dict[str, np.ndarray] = {}
    records: list[SeriesRecord] = []
    legacy_keys: set[tuple[str, str]] = set()

    if include_legacy:
        metadata_path = project_root / "data" / "_artifacts" / "monash_clean.meta.jsonl"
        legacy_records = import_legacy_inventory(metadata_path)
        values_path = metadata_path.with_name("monash_clean.npz")
        with np.load(values_path, allow_pickle=True) as archive:
            legacy_values = [np.asarray(value, dtype=np.float64) for value in archive["clean"]]
        for record, values in zip(legacy_records, legacy_values):
            asset = materialize_clean_base(
                clean_root,
                dataset_id=record.dataset_id,
                entity_id=record.entity_id,
                values=values,
                source_id=record.source_id,
                source_asset_sha256=record.source_asset_sha256,
                source_revision=record.source_revision,
                license_id=record.license_id,
                overlap_family=record.overlap_family,
                exposure_class=record.exposure_class,
                frequency=record.frequency,
                overlap_group=record.overlap_group or record.series_uid,
                overlap_status="resolved",
                overlap_evidence_sha256=record.overlap_evidence_sha256 or record.source_asset_sha256,
            )
            values_loaded, _, _ = read_clean_base(asset)
            records.append(asset.record)
            values_by_uid[asset.record.series_uid] = values_loaded
            legacy_keys.add((record.dataset_id.split(":", 1)[-1], record.entity_id))

    coordinates, blocking = metr_la_blocking(data_root)
    consumed_traffic = _probe_consumed_traffic(project_root)
    for source_id, dataset_id, parsed, raw_asset, overlap_group in _automatic_parsed(
        data_root,
        min_length=min_length,
        horizon=horizon,
        max_length=max_length,
        metr_blocking=blocking,
    ):
        config = dataset_id.split(":", 1)[-1] if dataset_id.startswith("monash:") else ""
        if config and (config, parsed.entity_id) in legacy_keys:
            continue
        exposure = (
            "probe_consumed"
            if source_id == "monash_hf" and config == "traffic_hourly" and parsed.entity_id in consumed_traffic
            else "certified_virgin"
        )
        spec = SOURCE_SPECS[source_id]
        asset = materialize_clean_base(
            clean_root,
            dataset_id=dataset_id,
            entity_id=parsed.entity_id,
            values=parsed.values,
            timestamps=parsed.timestamps,
            source_id=source_id,
            source_asset_sha256=raw_asset.sha256,
            source_revision=spec.source_revision,
            license_id=spec.license_id,
            overlap_family=spec.overlap_family,
            exposure_class=exposure,
            frequency=parsed.frequency,
            overlap_group=overlap_group,
            overlap_status="resolved",
            overlap_evidence_sha256=raw_asset.sha256,
        )
        values_loaded, times_loaded, _ = read_clean_base(asset)
        records.append(asset.record)
        values_by_uid[asset.record.series_uid] = values_loaded
        assert times_loaded is not None
        timestamps_by_uid[asset.record.series_uid] = times_loaded

    if not records:
        raise RuntimeError("probe found no materialized benchmark series")
    period_by_uid = {
        row.series_uid: {"hourly": 24, "daily": 7, "monthly": 12}[row.frequency]
        for row in records
    }
    features = _probe_with_cache(
        data_root,
        records,
        values_by_uid,
        timestamps_by_uid,
        period_by_uid,
    )
    finalized: list[SeriesRecord] = []
    for row in records:
        admission = _admission_with_scale(
            row,
            values_by_uid[row.series_uid],
            period=period_by_uid[row.series_uid],
            min_length=min_length,
            min_scale_pairs=min_scale_pairs,
        )
        finalized.append(
            row.with_probe_result(
                probe_features=features[row.series_uid],
                regime_tag=_regime(features[row.series_uid]),
                admission=admission,
            )
        )
    write_registry_jsonl(output / "series_registry.jsonl", finalized)

    if coordinates is not None and blocking is not None:
        diagnostics = block_diagnostics(coordinates, blocking)
        spatial_payload = {
            "schema_version": "benchmark-spatial-blocks/1",
            "benchmark_version": BENCHMARK_VERSION,
            "dataset_id": "metr_la",
            "source_revision": SOURCE_SPECS["metr_la"].source_revision,
            "sensor_locations_sha256": _sha256_file(
                data_root / "raw" / "metr_la" / "graph_sensor_locations.csv"
            ),
            "n_blocks": blocking.n_blocks,
            "site_merge_radius_km": blocking.site_merge_radius_km,
            "diagnostics": diagnostics,
            "overlap_group_of_entity": {
                entity: f"metr_la:block_{int(block):02d}"
                for entity, block in sorted(blocking.items())
            },
        }
        write_text_lf(
            output / "metr_la_spatial_blocks.json",
            json.dumps(spatial_payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n",
        )

    counts = Counter("eligible" if not row.admission_reasons else "rejected" for row in finalized)
    summary: dict[str, object] = {
        "schema_version": "benchmark-probe-summary/1",
        "benchmark_version": BENCHMARK_VERSION,
        "n_registry": len(finalized),
        "n_eligible": counts["eligible"],
        "n_rejected": counts["rejected"],
        "datasets": dict(sorted(Counter(row.dataset_id for row in finalized).items())),
        "regimes": dict(sorted(Counter(row.regime_tag for row in finalized).items())),
    }
    write_text_lf(
        output / "probe_summary.json",
        json.dumps(summary, sort_keys=True, ensure_ascii=True, indent=2) + "\n",
    )
    return summary


def _write_json(path: Path, payload: object) -> None:
    # Compared and written as BYTES: this artifact's digest is pinned into the benchmark
    # manifest, so a platform-dependent newline would make the freeze irreproducible.
    encoded = (
        json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    ).encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"frozen artifact differs on rerun: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def _data_card(
    records: list[SeriesRecord],
    eligible: list[SeriesRecord],
    role_counts: Counter,
    dataset_counts: Counter,
    output: Path,
) -> str:
    spatial_path = output / "metr_la_spatial_blocks.json"
    spatial = (
        json.loads(spatial_path.read_text("utf-8")) if spatial_path.is_file() else None
    )
    lines = [
        f"# Benchmark {BENCHMARK_VERSION} data card",
        "",
        f"Registry rows: {len(records)}; eligible: {len(eligible)}.",
        "",
        f"Frozen role counts: `{dict(sorted(role_counts.items()))}`.",
        "",
        f"Series per dataset: `{dict(sorted(dataset_counts.items()))}`.",
        "",
        "## What a number on this benchmark is allowed to claim",
        "",
        "Claim tiers and domains are declared per dataset in `dataset_manifest.json`, "
        "before any method was run. `supplementary` datasets (GEFCom2012, 20 zones) "
        "join the pooled roster and the per-cell tables, but a result on a "
        "supplementary dataset alone is not a reportable finding -- its Query sample "
        "is too small to carry one.",
        "",
        "### Traffic: two networks, and only one of them is spatially clean",
        "",
        "Traffic series are sensors on a road graph, not independent entities, so a "
        "within-dataset traffic result can only ever claim *unseen sensor in this "
        "network* -- never cross-network generalization. Cross-network evidence has to "
        "come from METR-LA (Los Angeles) and monash:traffic_hourly (San Francisco Bay "
        "Area) testing each other, because they are genuinely different road graphs.",
        "",
    ]
    if spatial is not None:
        diagnostics = spatial["diagnostics"]
        lines += [
            f"**METR-LA is spatially blocked.** Its {diagnostics['n_sensors']} sensors "
            f"merge into {diagnostics['n_sites']} co-location sites at a "
            f"{spatial['site_merge_radius_km']} km radius (largest site: "
            f"{diagnostics['site_size_max']} sensors -- these are detectors metres apart, "
            "e.g. opposite directions at one milepost), and those sites are grouped into "
            f"{spatial['n_blocks']} compact blocks by deterministic median bisection. The "
            "block, not the sensor, is the atomic split unit, so an entire block always "
            "travels to the same role. Minimum distance between sensors in *different* "
            f"blocks: {diagnostics['min_cross_block_distance_km']} km. That is the "
            "residual: blocking bounds adjacent-sensor leakage, it does not eliminate it.",
            "",
            "**monash:traffic_hourly is NOT spatially blocked, and cannot be.** The pinned "
            "Monash release ships no sensor coordinates, so its 862 Bay Area sensors are "
            "split per series. Adjacent sensors can therefore land on opposite sides of "
            "Support/Query and its within-dataset numbers should be read as an optimistic "
            "bound. METR-LA is the spatially clean traffic read; this one is not.",
            "",
        ]
    lines += [
        "## Corruption",
        "",
        "The grid in `corruption_grid.json` was pre-registered in full before any method "
        "was run against it. It has a **Natural lane** (`natural`, dose 0: no synthetic "
        "damage at all, only the missingness the source actually shipped with -- "
        "deterministic, so one replicate), the three inherited **Controlled v0** "
        "missingness cells, and five new **Controlled v0.1** cells covering outliers, "
        "structural break, additive noise, and timestamp disorder. A pipeline that only "
        "imputes will score identically to Raw on the last four; that is the point.",
        "",
        "## Support-A is two pools, not one",
        "",
        "`support_a_subsplit.json` partitions Support-A at the overlap-group level into "
        "`support_a_discovery` (search and fit freely) and `support_a_validation` (the "
        "development promotion gate). A candidate is never judged on the series used to "
        "find it. This is a partition of the Support-A *population* and has nothing to do "
        "with the chronological train/validation/test boundaries inside each series.",
        "",
        "## U",
        "",
        "NOAA Global Hourly is the sealed weather U pool. Natural NaNs are preserved; "
        "hourly bins with partial or absent observations remain missing.",
        "",
    ]
    return "\n".join(lines)


def freeze_workspace(root: Path | str, out: Path | str) -> SplitManifest:
    """Freeze registry membership and protocol artifacts without reading Final utility."""

    data_root, output = Path(root), Path(out)
    records = read_registry_jsonl(output / "series_registry.jsonl")
    eligible = [row for row in records if row.admission_reasons == ()]
    candidates = [row.to_split_candidate() for row in eligible]
    u_selected = {row.series_uid for row in eligible if row.dataset_id == "noaa_global_hourly"}
    manifest = build_split_manifest(candidates, BENCHMARK_VERSION, SPLIT_SALT, u_selected)
    _write_json(output / "split_manifest.json", manifest.to_dict())

    dataset_counts = Counter(row.dataset_id for row in records)
    _write_json(
        output / "dataset_manifest.json",
        dataset_manifest_document(dict(dataset_counts)),
    )
    _write_json(output / "support_a_subsplit.json", build_support_a_subsplit(manifest))

    corruption_document = {
        "schema_version": "benchmark-corruption-grid/1",
        "benchmark_version": BENCHMARK_VERSION,
        "preregistered_before_any_method_ran": True,
        "cells": [
            {
                "scenario": scenario,
                "dose": dose,
                "lane": LANE_OF_SCENARIO[scenario],
                "replicates": list(replicates_for(scenario)),
            }
            for scenario, dose in CORRUPTION_GRID
        ],
        "seed_rule": (
            "sha256([benchmark_version, clean_content_sha, scenario, dose, replicate_idx]) "
            "-- invariant to source order and subset selection; every method consumes the "
            "identical realization (CRN)"
        ),
    }
    _write_json(output / "corruption_grid.json", corruption_document)
    # The pool is frozen here, at freeze time, before any v0.2 number exists. A pool chosen
    # after the fact -- "these are the operators that count" -- is the shortest path from a
    # null result to a discovery.
    _write_json(output / "program_pool.json", pool_manifest())

    acquisition = data_root / "acquisition_manifest.json"
    acquisition_sha = _sha256_file(acquisition) if acquisition.is_file() else None
    registry_sha = _sha256_file(output / "series_registry.jsonl")
    split_sha = _sha256_file(output / "split_manifest.json")
    spatial_path = output / "metr_la_spatial_blocks.json"
    benchmark_manifest = {
        "schema_version": "benchmark-manifest/1",
        "benchmark_version": BENCHMARK_VERSION,
        "acquisition_manifest_sha256": acquisition_sha,
        "registry_sha256": registry_sha,
        "split_manifest_sha256": split_sha,
        "dataset_manifest_sha256": _sha256_file(output / "dataset_manifest.json"),
        "support_a_subsplit_sha256": _sha256_file(output / "support_a_subsplit.json"),
        "corruption_grid_sha256": _sha256_file(output / "corruption_grid.json"),
        "program_pool_sha256": _sha256_file(output / "program_pool.json"),
        "metr_la_spatial_blocks_sha256": (
            _sha256_file(spatial_path) if spatial_path.is_file() else None
        ),
        "split_salt": SPLIT_SALT,
        "headline": {"lookback": 48, "horizon": 48, "stride": 4, "min_length": 207},
        "corruption_seeds": [0, 1],
        "model_seeds": [0, 1, 2],
        "bootstrap": {"b": 2000, "seed": 20260713},
        "harm_threshold": 0.05,
        "harm_threshold_kind": "conventional",
        "normalization_owner": "benchmark",
        "regime_thresholds": REGIME_THRESHOLDS,
        "final_query_state": "sealed",
    }
    _write_json(output / "benchmark_manifest_v0.yaml", benchmark_manifest)
    role_counts = Counter(row.role.value for row in manifest.assignments)
    write_text_lf(
        output / "data_card.md",
        _data_card(records, eligible, role_counts, dataset_counts, output),
    )
    write_text_lf(
        output / "training_evaluation_protocol.md",
        f"# Benchmark {BENCHMARK_VERSION} training and evaluation protocol\n\n"
        "Normalization is benchmark-owned. Raw means No-op + canonical ingestion. "
        "Closed-form, Adam-DLinear, and LSTM share eligibility, windows, ingestion, and normalization.\n\n"
        "One model is trained per (program, scenario, dose, corruption replicate, dataset) on "
        "that dataset's inner-train with series-equal weighting. The frozen spec fixes trainer "
        "internals but never fixed the training pool's scope; v0.2 fixes it at dataset. The pool "
        "is never sliced by regime_tag, which is a benchmark-private diagnostic label, and "
        "dataset_id is public metadata, so slicing by it leaks nothing.\n\n"
        "Oracles are RETRAINED: once a policy picks a program per (cell, scenario, dose), the "
        "corpus those picks produce is assembled and a model is trained on it, through the same "
        "path a Method takes. An oracle read off single-program models describes a corpus no "
        "model was ever fitted to and is reported as descriptive only.\n\n"
        "Aggregation order: model seed -> corruption replicate -> scenario and dose -> one row per uid "
        "-> cell series mean -> dataset macro mean within regime.\n\n"
        "Final-Query is sealed until one frozen evaluation campaign records durable unseal/access events.\n",
    )
    freeze_event = {
        "event": "benchmark_freeze",
        "benchmark_version": BENCHMARK_VERSION,
        "registry_sha256": registry_sha,
        "split_manifest_sha256": split_sha,
        "final_query_state": "sealed",
    }
    ledger_bytes = (
        json.dumps(freeze_event, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    ledger_path = output / "virgin_ledger.jsonl"
    if ledger_path.exists() and ledger_path.read_bytes() != ledger_bytes:
        raise RuntimeError("frozen virgin ledger differs on rerun")
    ledger_path.write_bytes(ledger_bytes)
    return manifest


__all__ = ["freeze_workspace", "probe_workspace"]
