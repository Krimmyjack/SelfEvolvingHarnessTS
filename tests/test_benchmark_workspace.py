from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from SelfEvolvingHarnessTS.benchmark.materialize import write_raw_once
from SelfEvolvingHarnessTS.benchmark.registry import read_registry_jsonl
from SelfEvolvingHarnessTS.benchmark.sources import SOURCE_SPECS
from SelfEvolvingHarnessTS.benchmark.workspace import freeze_workspace, probe_workspace


def _write_small_uci(root, n_meters: int = 24):
    """A roster big enough to populate every outer role.

    A freeze whose Support-A is empty is not a valid benchmark, and freeze_workspace now
    says so, so the fixture can no longer be a single series.
    """
    header = ";".join(['""'] + [f'"MT_{meter:03d}"' for meter in range(1, n_meters + 1)])
    rows = [header]
    for index in range(880):
        hour, minute = divmod(index * 15, 60)
        stamp = f"2014-01-{1 + hour // 24:02d} {hour % 24:02d}:{minute:02d}:00"
        values = ";".join(f"{index + meter * 7},0" for meter in range(n_meters))
        rows.append(f"{stamp};{values}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("LD2011_2014.txt", "\n".join(rows) + "\n")
    spec = SOURCE_SPECS["uci_electricity_load_diagrams"]
    write_raw_once(
        root / "raw" / spec.source_id / "electricityloaddiagrams20112014.zip",
        buffer.getvalue(),
        source_revision=spec.source_revision,
    )


def _write_small_gefcom2012(root):
    hours = [f"h{hour}" for hour in range(1, 25)]
    rows = []
    for day in pd.date_range("2004-01-01", periods=10, freq="D"):
        rows.append(
            dict(
                zone_id=1,
                year=day.year,
                month=day.month,
                day=day.day,
                **{name: hour + len(rows) * 24 for hour, name in enumerate(hours, 1)},
            )
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "GEFCOM2012_Data/Load/Load_history.csv",
            pd.DataFrame(rows).to_csv(index=False),
        )
        archive.writestr(
            "GEFCOM2012_Data/Load/Load_solution.csv",
            pd.DataFrame(rows).assign(id=range(1, 11), weight=1).to_csv(index=False),
        )
    spec = SOURCE_SPECS["gefcom2012"]
    write_raw_once(
        root / "incoming" / spec.incoming_subdir / "GEFCom2012.zip",
        buffer.getvalue(),
        source_revision=spec.source_revision,
    )


def _metr_sensor_id(site: int, twin: int) -> str:
    return str(700000 + site * 2 + twin)


def _write_small_metr(root, tmp_path, n_sites: int = 22, with_locations: bool = True):
    """A miniature road network: `n_sites` locations, each with a co-located twin sensor.

    The twins sit ~20 m apart (as real METR-LA detector pairs do) and the sites ~1 km
    apart, so the fixture exercises the real invariant: twins must never be separable.
    """
    source = tmp_path / "metr-source.h5"
    index = pd.date_range("2012-03-01", periods=240 * 12, freq="5min")
    columns = {}
    for site in range(n_sites):
        for twin in range(2):
            columns[_metr_sensor_id(site, twin)] = np.sin(
                np.arange(len(index)) / 12 + site + 0.01 * twin
            )
    pd.DataFrame(columns, index=index).to_hdf(source, key="df")

    spec = SOURCE_SPECS["metr_la"]
    write_raw_once(
        root / "raw" / spec.source_id / "metr-la.h5",
        source.read_bytes(),
        source_revision=spec.source_revision,
    )
    if not with_locations:
        return
    lines = ["index,sensor_id,latitude,longitude"]
    row = 0
    for site in range(n_sites):
        latitude = 34.0 + site * 0.009  # ~1 km between sites
        for twin in range(2):
            twin_latitude = latitude + twin * 0.0002  # ~22 m between twins
            lines.append(
                f"{row},{_metr_sensor_id(site, twin)},{twin_latitude:.6f},-118.000000"
            )
            row += 1
    write_raw_once(
        root / "raw" / spec.source_id / "graph_sensor_locations.csv",
        ("\n".join(lines) + "\n").encode("utf-8"),
        source_revision=spec.source_revision,
    )


def test_probe_and_freeze_workspace_produce_finalized_registry_without_unseal(tmp_path):
    root = tmp_path / "data"
    out = tmp_path / "results"
    _write_small_uci(root)
    summary = probe_workspace(
        root,
        out,
        include_legacy=False,
        min_length=207,
        horizon=2,
        max_length=220,
        min_scale_pairs=2,
    )
    assert summary["n_registry"] == 24
    registry = read_registry_jsonl(out / "series_registry.jsonl")
    assert all(row.probe_features is not None for row in registry)
    assert all(row.admission_reasons == () for row in registry)
    cache_rows = list((root / "probe_cache").glob("*.json"))
    assert len(cache_rows) == 24

    manifest = freeze_workspace(root, out)
    assert len(manifest.assignments) == 24
    assert (out / "benchmark_manifest_v0.yaml").is_file()
    assert (out / "split_manifest.json").is_file()

    # The freeze now also pins the sidecars the arena's honesty depends on.
    assert (out / "dataset_manifest.json").is_file()
    assert (out / "corruption_grid.json").is_file()
    subsplit = json.loads((out / "support_a_subsplit.json").read_text("utf-8"))
    assert subsplit["unit"] == "overlap_group"
    assert subsplit["counts"]["support_a_discovery"] > 0
    assert subsplit["counts"]["support_a_validation"] > 0

    events = [json.loads(line) for line in (out / "virgin_ledger.jsonl").read_text("utf-8").splitlines()]
    assert [event["event"] for event in events] == ["benchmark_freeze"]
    assert not any(event["event"] == "unseal" for event in events)


def test_probe_workspace_consumes_registered_gefcom_and_metr_assets(tmp_path):
    root = tmp_path / "data"
    out = tmp_path / "results"
    _write_small_gefcom2012(root)
    _write_small_metr(root, tmp_path)

    summary = probe_workspace(
        root,
        out,
        include_legacy=False,
        min_length=207,
        horizon=2,
        max_length=220,
        min_scale_pairs=2,
    )

    assert summary["datasets"] == {"gefcom2012_load": 1, "metr_la": 44}
    registry = read_registry_jsonl(out / "series_registry.jsonl")
    assert {row.source_id for row in registry} == {"gefcom2012", "metr_la"}

    # METR-LA's atomic split unit is the spatial block, never the individual sensor.
    metr_rows = [row for row in registry if row.source_id == "metr_la"]
    group_of_sensor = {row.entity_id: row.overlap_group for row in metr_rows}
    assert all(group.startswith("metr_la:block_") for group in group_of_sensor.values())
    assert len(set(group_of_sensor.values())) < len(metr_rows)

    # The invariant that matters: a co-located twin pair is inseparable, so it can never
    # end up with one sensor in Support and its near-duplicate in Query.
    for site in range(22):
        assert (
            group_of_sensor[_metr_sensor_id(site, 0)]
            == group_of_sensor[_metr_sensor_id(site, 1)]
        )
    assert (out / "metr_la_spatial_blocks.json").is_file()


def test_frozen_artifacts_are_byte_exact_lf_and_hash_to_their_pinned_digests(tmp_path):
    """The manifest's digests must hold for the bytes actually on disk, on any platform.

    Path.write_text emits CRLF on Windows, so a manifest frozen there would pin CRLF
    digests that a Linux re-freeze could never reproduce -- and git's `text=auto` would
    rewrite the artifact on checkout so it no longer hashes to its own recorded digest.
    """
    root = tmp_path / "data"
    out = tmp_path / "results"
    _write_small_uci(root)
    probe_workspace(
        root, out, include_legacy=False, min_length=207, horizon=2, max_length=220,
        min_scale_pairs=2,
    )
    freeze_workspace(root, out)

    manifest = json.loads((out / "benchmark_manifest_v0.yaml").read_bytes())
    pinned = {
        "series_registry.jsonl": "registry_sha256",
        "split_manifest.json": "split_manifest_sha256",
        "dataset_manifest.json": "dataset_manifest_sha256",
        "support_a_subsplit.json": "support_a_subsplit_sha256",
        "corruption_grid.json": "corruption_grid_sha256",
    }
    for name, key in pinned.items():
        payload = (out / name).read_bytes()
        assert b"\r\n" not in payload, f"{name} was written with CRLF"
        assert hashlib.sha256(payload).hexdigest() == manifest[key], (
            f"{name} does not hash to the digest pinned in the benchmark manifest"
        )

    for name in ("probe_summary.json", "data_card.md", "virgin_ledger.jsonl"):
        assert b"\r\n" not in (out / name).read_bytes(), f"{name} was written with CRLF"


def test_metr_la_without_coordinates_is_refused_rather_than_split_per_sensor(tmp_path):
    root = tmp_path / "data"
    out = tmp_path / "results"
    _write_small_metr(root, tmp_path, with_locations=False)

    # Silently falling back to a per-sensor split would leak co-located sensors across
    # roles, so the absence of coordinates must be loud.
    with pytest.raises(RuntimeError, match="would leak"):
        probe_workspace(
            root,
            out,
            include_legacy=False,
            min_length=207,
            horizon=2,
            max_length=220,
            min_scale_pairs=2,
        )
