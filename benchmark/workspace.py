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
from .materialize import (
    ParsedSeries,
    RawAsset,
    materialize_clean_base,
    parse_metr_la_hdf,
    parse_monash_parquet,
    parse_noaa_global_hourly,
    parse_uci_electricity_zip,
    read_clean_base,
    verify_raw_asset,
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
from .sources import SOURCE_SPECS
from .split import SplitManifest, build_split_manifest

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


def _automatic_parsed(
    root: Path,
    *,
    min_length: int,
    horizon: int,
    max_length: int,
) -> Iterable[tuple[str, str, ParsedSeries, RawAsset, str]]:
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
            yield "monash_hf", f"monash:{config}", row, asset, config

    metr = root / "raw" / "metr_la" / "metr-la.h5"
    if metr.is_file():
        asset = _bound_raw(metr)
        for row in parse_metr_la_hdf(
            metr,
            min_length=min_length,
            horizon=horizon,
            max_length=max_length,
        ):
            yield "metr_la", "metr_la", row, asset, "metr_la"

    uci = root / "raw" / "uci_electricity_load_diagrams" / "electricityloaddiagrams20112014.zip"
    if uci.is_file():
        asset = _bound_raw(uci)
        for row in parse_uci_electricity_zip(
            uci,
            min_length=min_length,
            horizon=horizon,
            max_length=max_length,
        ):
            yield "uci_electricity_load_diagrams", "uci_electricity_load_diagrams", row, asset, "uci_eld"

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
        yield "noaa_global_hourly", "noaa_global_hourly", row, asset, "noaa_isd"


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
        temporary.write_text(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
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
) -> dict[str, object]:
    """Materialize immutable clean bases, run the loss-free probe, and write registry."""

    if horizon != HEADLINE_HORIZON and include_legacy:
        raise ValueError("production probe must use the frozen headline horizon")
    data_root, output = Path(root), Path(out)
    output.mkdir(parents=True, exist_ok=True)
    clean_root = data_root / "clean_base"
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

    consumed_traffic = _probe_consumed_traffic(project_root)
    for source_id, dataset_id, parsed, raw_asset, overlap_label in _automatic_parsed(
        data_root,
        min_length=min_length,
        horizon=horizon,
        max_length=max_length,
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
            overlap_group=f"{overlap_label}:{parsed.entity_id}",
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
    (output / "probe_summary.json").write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    if path.exists() and path.read_text("utf-8") != encoded:
        raise RuntimeError(f"frozen artifact differs on rerun: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def freeze_workspace(root: Path | str, out: Path | str) -> SplitManifest:
    """Freeze registry membership and protocol artifacts without reading Final utility."""

    data_root, output = Path(root), Path(out)
    records = read_registry_jsonl(output / "series_registry.jsonl")
    eligible = [row for row in records if row.admission_reasons == ()]
    candidates = [row.to_split_candidate() for row in eligible]
    u_selected = {row.series_uid for row in eligible if row.dataset_id == "noaa_global_hourly"}
    manifest = build_split_manifest(candidates, BENCHMARK_VERSION, SPLIT_SALT, u_selected)
    _write_json(output / "split_manifest.json", manifest.to_dict())
    acquisition = data_root / "acquisition_manifest.json"
    acquisition_sha = _sha256_file(acquisition) if acquisition.is_file() else None
    registry_sha = _sha256_file(output / "series_registry.jsonl")
    split_sha = _sha256_file(output / "split_manifest.json")
    benchmark_manifest = {
        "schema_version": "benchmark-manifest/1",
        "benchmark_version": BENCHMARK_VERSION,
        "acquisition_manifest_sha256": acquisition_sha,
        "registry_sha256": registry_sha,
        "split_manifest_sha256": split_sha,
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
    (output / "data_card.md").write_text(
        "# Benchmark v0 data card\n\n"
        f"Registry rows: {len(records)}; eligible: {len(eligible)}.\n\n"
        f"Frozen role counts: `{dict(sorted(role_counts.items()))}`.\n\n"
        "NOAA Global Hourly is the frozen weather U pool. Natural NaNs are preserved; "
        "hourly bins with partial or absent observations remain missing.\n",
        encoding="utf-8",
    )
    (output / "training_evaluation_protocol.md").write_text(
        "# Benchmark v0 training and evaluation protocol\n\n"
        "Normalization is benchmark-owned. Raw means No-op + canonical ingestion. "
        "Closed-form, Adam-DLinear, and LSTM share eligibility, windows, ingestion, and normalization.\n\n"
        "Aggregation order: model seed -> corruption replicate -> scenario and dose -> one row per uid "
        "-> cell series mean -> dataset macro mean within regime.\n\n"
        "Final-Query is sealed until one frozen evaluation campaign records durable unseal/access events.\n",
        encoding="utf-8",
    )
    freeze_event = {
        "event": "benchmark_freeze",
        "benchmark_version": BENCHMARK_VERSION,
        "registry_sha256": registry_sha,
        "split_manifest_sha256": split_sha,
        "final_query_state": "sealed",
    }
    ledger_bytes = json.dumps(
        freeze_event, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ) + "\n"
    ledger_path = output / "virgin_ledger.jsonl"
    if ledger_path.exists() and ledger_path.read_text("utf-8") != ledger_bytes:
        raise RuntimeError("frozen virgin ledger differs on rerun")
    ledger_path.write_text(ledger_bytes, encoding="utf-8")
    return manifest


__all__ = ["freeze_workspace", "probe_workspace"]
