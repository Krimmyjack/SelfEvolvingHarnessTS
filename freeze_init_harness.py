"""freeze_init_harness.py — freezes the Init Harness membership manifest (2026-07-13).

Init Harness = every Support-A-only series that carries real prior exposure
from before benchmark-v0 existed:

  1) legacy_core (mandatory, 80 series): the admitted subset of the 83-series
     legacy Monash bundle (`legacy_internal_monash_clean`, exposure_class=
     confirmed_exposed — registry.import_legacy_inventory).
  2) probe_consumed_extension (56 series, included by default): traffic_hourly
     entity_ids that P6's own U-admission probe recorded as fully consumed
     while inspecting candidates for the pre-benchmark harness's U pool
     (results/Stage2/P6Probes/u_admission_v2_traffic_hourly.json:
     all_probe_consumed_item_ids). This is real prior exposure to the
     pre-benchmark harness's own formation, not a hypothetical taint, so it is
     routed to Init rather than left as Fresh Support-A. A consumer that wants
     a more conservative Init Harness can filter members to group=legacy_core.

Both groups are Support-A-only by registry construction (`_SUPPORT_A_ONLY` in
benchmark/registry.py) and therefore can never serve Dev-Query, Support-B,
Final-Query, or U. The complement of Init Harness within Support-A
(exposure_class=certified_virgin) is the Fresh Support-A harness-update/
selection pool; it is not named here because it is already fully derivable
from split_manifest.json (role=support_a and exposure_class=certified_virgin)
without a second frozen artifact.

This script only names and hash-pins an existing subset of the already-frozen
registry/split; it never writes to registry.jsonl, split_manifest.json, or
benchmark_manifest_v0.yaml.

Run: PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.freeze_init_harness
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .benchmark import BENCHMARK_VERSION
from .benchmark.materialize import write_text_lf
from .benchmark.registry import SeriesRecord, read_registry_jsonl

PKG = Path(__file__).resolve().parent
RESULTS = PKG / "results" / "Benchmark_v0_1"
DATA = PKG / "data" / "benchmark_v0_1"
REGISTRY_PATH = RESULTS / "series_registry.jsonl"
SPLIT_PATH = RESULTS / "split_manifest.json"
BENCHMARK_MANIFEST_PATH = RESULTS / "benchmark_manifest_v0.yaml"
U_ADMISSION_PROBE_PATH = PKG / "results" / "Stage2" / "P6Probes" / "u_admission_v2_traffic_hourly.json"
INIT_HARNESS_PATH = RESULTS / "init_harness_manifest.json"
CLEAN_BASE_ROOT = DATA / "clean_base"

LEGACY_SOURCE_ID = "legacy_internal_monash_clean"
EXPECTED_LEGACY_TOTAL = 83
EXPECTED_LEGACY_QUALIFIED = 80


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean_base_slot_key(source_id: str, dataset_id: str, entity_id: str) -> str:
    # Mirrors benchmark.materialize.materialize_clean_base's slot_key exactly
    # (materialize.py: slot_key = sha256(json([source_id, dataset_id, entity_id]))).
    payload = json.dumps(
        [source_id, dataset_id, entity_id], ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _verify_frozen_bytes(benchmark_manifest: dict) -> None:
    registry_sha = _sha256_bytes(REGISTRY_PATH.read_bytes())
    split_sha = _sha256_bytes(SPLIT_PATH.read_bytes())
    if registry_sha != benchmark_manifest["registry_sha256"]:
        raise SystemExit(
            "series_registry.jsonl no longer matches the frozen benchmark_manifest_v0.yaml "
            f"registry_sha256 (expected {benchmark_manifest['registry_sha256']}, got {registry_sha})"
        )
    if split_sha != benchmark_manifest["split_manifest_sha256"]:
        raise SystemExit(
            "split_manifest.json no longer matches the frozen benchmark_manifest_v0.yaml "
            f"split_manifest_sha256 (expected {benchmark_manifest['split_manifest_sha256']}, got {split_sha})"
        )


def _row_entry(
    row: SeriesRecord, group: str, split_assignments: dict[str, dict]
) -> dict:
    assignment = split_assignments.get(row.series_uid)
    if assignment is None:
        raise SystemExit(f"series_uid {row.series_uid} has no split assignment")
    if assignment["role"] != "support_a":
        raise SystemExit(
            f"series_uid {row.series_uid} is role={assignment['role']!r}, expected support_a"
        )
    slot_key = _clean_base_slot_key(row.source_id, row.dataset_id, row.entity_id)
    if not (CLEAN_BASE_ROOT / slot_key).is_dir():
        raise SystemExit(f"clean_base slot missing for series_uid {row.series_uid}: {slot_key}")
    return dict(
        group=group,
        series_uid=row.series_uid,
        dataset_id=row.dataset_id,
        entity_id=row.entity_id,
        source_id=row.source_id,
        exposure_class=row.exposure_class,
        content_sha=row.content_sha,
        role=assignment["role"],
        clean_base_slot=slot_key,
    )


def build_init_harness() -> dict:
    if INIT_HARNESS_PATH.exists():
        raise SystemExit(
            f"init_harness_manifest.json already exists ({INIT_HARNESS_PATH}); "
            "frozen artifacts are not overwritten — move it aside first if you intend to redo it."
        )

    benchmark_manifest = json.loads(BENCHMARK_MANIFEST_PATH.read_text("utf-8"))
    _verify_frozen_bytes(benchmark_manifest)

    registry_rows = read_registry_jsonl(REGISTRY_PATH)
    split_assignments = {
        a["series_uid"]: a for a in json.loads(SPLIT_PATH.read_text("utf-8"))["assignments"]
    }

    legacy_rows = [r for r in registry_rows if r.source_id == LEGACY_SOURCE_ID]
    if len(legacy_rows) != EXPECTED_LEGACY_TOTAL:
        raise SystemExit(
            f"expected {EXPECTED_LEGACY_TOTAL} legacy rows in the registry, found {len(legacy_rows)}"
        )
    legacy_qualified = [r for r in legacy_rows if not r.admission_reasons]
    legacy_excluded = [r for r in legacy_rows if r.admission_reasons]
    if len(legacy_qualified) != EXPECTED_LEGACY_QUALIFIED:
        raise SystemExit(
            f"expected {EXPECTED_LEGACY_QUALIFIED} qualified legacy rows, found {len(legacy_qualified)}"
        )

    probe_consumed_rows = [r for r in registry_rows if r.exposure_class == "probe_consumed"]
    if any(r.admission_reasons for r in probe_consumed_rows):
        raise SystemExit("a probe_consumed row carries a disqualifying admission reason")

    # Cross-check the probe_consumed label against P6's own U-admission exclusion
    # ledger rather than trusting the registry tag on its own.
    u_admission = json.loads(U_ADMISSION_PROBE_PATH.read_text("utf-8"))
    ledger_ids = set(u_admission["all_probe_consumed_item_ids"])
    registry_ids = {r.entity_id for r in probe_consumed_rows}
    if ledger_ids != registry_ids:
        raise SystemExit(
            "probe_consumed registry entity_ids disagree with the P6 U-admission probe ledger:\n"
            f"  ledger-only: {sorted(ledger_ids - registry_ids)}\n"
            f"  registry-only: {sorted(registry_ids - ledger_ids)}"
        )

    legacy_entries = [_row_entry(r, "legacy_core", split_assignments) for r in legacy_qualified]
    probe_entries = [
        _row_entry(r, "probe_consumed_extension", split_assignments) for r in probe_consumed_rows
    ]
    members = sorted(legacy_entries + probe_entries, key=lambda e: e["series_uid"])

    legacy_by_dataset = Counter(r.dataset_id for r in legacy_qualified)
    legacy_excluded_detail = [
        dict(dataset_id=r.dataset_id, entity_id=r.entity_id, reasons=list(r.admission_reasons))
        for r in legacy_excluded
    ]

    doc = dict(
        schema_version="benchmark-init-harness-manifest/1",
        benchmark_version=BENCHMARK_VERSION,
        registry_sha256=benchmark_manifest["registry_sha256"],
        split_manifest_sha256=benchmark_manifest["split_manifest_sha256"],
        definition=dict(
            statement=(
                "Init Harness is the frozen pre-benchmark exposure record: every "
                "Support-A-only series whose values were touched before benchmark-v0 "
                "existed. It is not a Fresh Support-A development pool and can never "
                "serve Dev-Query, Support-B, Final-Query, or U."
            ),
            member_exposure_classes=["confirmed_exposed", "probe_consumed"],
            excludes=(
                "certified_virgin support_a series remain the Fresh Support-A "
                "harness-update/selection pool; they are not part of Init Harness "
                "and are not listed here (derivable from split_manifest.json as "
                "role=support_a and exposure_class=certified_virgin)."
            ),
        ),
        groups=dict(
            legacy_core=dict(
                mandatory=True,
                source_id=LEGACY_SOURCE_ID,
                rationale=(
                    "83 exposed legacy series bundled at project start "
                    "(20 nn5_daily, 20 fred_md, 20 tourism_monthly, 20 covid_deaths, "
                    "1 each us_births/saugeenday/sunspot); exposure_class=confirmed_exposed "
                    "by construction (registry.import_legacy_inventory)."
                ),
                total_count=len(legacy_rows),
                qualified_count=len(legacy_qualified),
                excluded_count=len(legacy_excluded),
                excluded=legacy_excluded_detail,
                qualified_by_dataset={k: v for k, v in sorted(legacy_by_dataset.items())},
            ),
            probe_consumed_extension=dict(
                mandatory=False,
                default_include=True,
                source_id="monash_hf",
                dataset_id="monash:traffic_hourly",
                rationale=(
                    "56 traffic_hourly series P6's own U-admission probe "
                    "(results/Stage2/P6Probes/u_admission_v2_traffic_hourly.json: "
                    "all_probe_consumed_item_ids) recorded as fully probe-consumed "
                    "while inspecting candidates for the pre-benchmark harness's U pool "
                    "(P6 prereg: traffic_hourly universe 862 - 56 consumed = 806 sampling "
                    "universe). Real prior exposure to the pre-benchmark harness's own "
                    "formation, not a hypothetical taint, so it is included in Init Harness "
                    "by default rather than left in the Fresh Support-A pool. Filter "
                    "members to group=legacy_core for a more conservative legacy-only "
                    "Init Harness."
                ),
                count=len(probe_entries),
                u_admission_ledger_sha256=_sha256_bytes(U_ADMISSION_PROBE_PATH.read_bytes()),
            ),
        ),
        total_count=len(members),
        core_only_count=len(legacy_entries),
        members=members,
    )
    doc["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in doc.items() if k != "manifest_sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return doc


def main() -> None:
    doc = build_init_harness()
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Byte-exact LF: this manifest carries its own SHA256, and Path.write_text would emit
    # CRLF on Windows, making the digest depend on which OS froze it.
    write_text_lf(
        INIT_HARNESS_PATH,
        json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
    )
    extension_count = doc["total_count"] - doc["core_only_count"]
    print(
        f"init_harness_manifest.json written: {doc['total_count']} series "
        f"({doc['core_only_count']} legacy_core + {extension_count} probe_consumed_extension), "
        f"manifest_sha256={doc['manifest_sha256'][:16]}..."
    )


if __name__ == "__main__":
    main()
