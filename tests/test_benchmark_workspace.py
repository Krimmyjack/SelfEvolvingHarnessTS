from __future__ import annotations

import io
import json
import zipfile

from SelfEvolvingHarnessTS.benchmark.materialize import write_raw_once
from SelfEvolvingHarnessTS.benchmark.registry import read_registry_jsonl
from SelfEvolvingHarnessTS.benchmark.sources import SOURCE_SPECS
from SelfEvolvingHarnessTS.benchmark.workspace import freeze_workspace, probe_workspace


def _write_small_uci(root):
    rows = ['"";"MT_001"']
    for index in range(880):
        hour, minute = divmod(index * 15, 60)
        rows.append(f"2014-01-{1 + hour // 24:02d} {hour % 24:02d}:{minute:02d}:00;{index},0")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("LD2011_2014.txt", "\n".join(rows) + "\n")
    spec = SOURCE_SPECS["uci_electricity_load_diagrams"]
    write_raw_once(
        root / "raw" / spec.source_id / "electricityloaddiagrams20112014.zip",
        buffer.getvalue(),
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
    assert summary["n_registry"] == 1
    registry = read_registry_jsonl(out / "series_registry.jsonl")
    assert registry[0].probe_features is not None
    assert registry[0].admission_reasons == ()
    cache_rows = list((root / "probe_cache").glob("*.json"))
    assert len(cache_rows) == 1

    manifest = freeze_workspace(root, out)
    assert len(manifest.assignments) == 1
    assert (out / "benchmark_manifest_v0.yaml").is_file()
    assert (out / "split_manifest.json").is_file()
    events = [json.loads(line) for line in (out / "virgin_ledger.jsonl").read_text("utf-8").splitlines()]
    assert [event["event"] for event in events] == ["benchmark_freeze"]
    assert not any(event["event"] == "unseal" for event in events)
