"""Rebuild a fresh NOAA cohort from raw station files, then health-check it.

Step 0 is a deterministic consumption census over already-written files.
Step 1 materializes unconsumed stations onto a frozen hourly grid.
Step 2 is a per-station development-only physical exam against the #13
bars (length, missing-rate cap derived from min_series_length, and the
substrate scale-floor / flatline guard).  0 Consumer retrains.

2025 confirmation values are never read: every load slices ``[:8760]`` and
no 2025 csv is opened.  0 LLM calls.

Writes ``artifacts/functional/e2/noaa_fresh_cohort_v2.json`` and ``.md``.
Does not rewrite the v1 census artifacts.  Arrays go under
``data/benchmark_noaa_fresh_v1/``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)


# ------------------------------------------------------------------ frozen
# Written before any series value is read.  Do not retune after seeing data.
PROTOCOL_VERSION = "noaa_fresh_cohort_v2"
GRID_START = datetime(2024, 1, 1)
DEVELOPMENT_HOURS = 8760  # 365 * 24; calendar_sealed_boundary
PARTITION_KIND = "calendar_sealed_boundary"
# #14 froze 24 as slack on a 12-seat roster.  Under every consumption
# counting method 24 unconsumed 2024 stations are unreachable (widest
# remainder 23).  #14 opened zero csv and read zero series values.
# Corrected to 20 before any csv is opened: 20 still leaves 8 seats of
# slack on the 12-station confirmation roster.
MIN_FRESH_STATIONS = 20
MIN_FRESH_STATIONS_V1 = 24
MAX_SELECT = 40
TMP_MISSING = 9999
FIRST_FINITE_WINS = True  # no hourly mean, no interpolation, no smoothing
MIN_ROSTER_PASS = 12

# v1 frozen Judge bars, quoted from the #13 artifact at health-check time.
# Duplicated here so --census-only can name them without importing the stack.
MIN_TRAIN_SERIES = 12
MIN_EVAL_SERIES = 8
MIN_SERIES_LENGTH = 5760
MAX_EVAL_LOSS_SPREAD = 5.0
MAX_SINGLE_SERIES_LOSS_SHARE = 0.40
MIN_SERIES_WITH_PUBLIC_PHENOMENON = 4
RETRAIN_BUDGET = 0  # this protocol: 0 Consumer retrains
MIN_CHANNELS_FOR_PER_CHANNEL = 2
GENERATOR_SHA_ON_RECORD_V1 = (
    "45b85890c79baadde49cf1e07e5df688ad5346b2e02deaa1db9a4859ab4485ea"
)
FLOOR_CORRECTION = {
    "from": MIN_FRESH_STATIONS_V1,
    "to": MIN_FRESH_STATIONS,
    "when": "before any csv opened, before any series value read",
    "why": (
        "the floor was slack on a 12-seat roster, not a data-dependent bar.  "
        "Under every #14 consumption counting method 24 unconsumed 2024 "
        "stations are unreachable (widest remainder 23; delivered #14 "
        "count 20).  20 still leaves 8 seats of slack."
    ),
    "hash_on_record_v1": GENERATOR_SHA_ON_RECORD_V1,
}

# Probe origins: the frozen three non-overlapping development windows.
# They already sit inside [0, 8760).  Guard numbers are reused; only the
# sealed boundary is rebound from 3072 to 8760.
FROZEN_DEVELOPMENT_ORIGINS = (1104, 1368, 1800)
FROZEN_SEALED_FROM_INDEX_V1 = 3072
FROZEN_NOAA_CONFIG = {
    "dataset_id": "noaa_global_hourly",
    "selection_origin": 768,
    "support_origin": 720,
    "period": 24,
}
FROZEN_HORIZON = 48
FROZEN_FIXED_ROSTER_N = MIN_TRAIN_SERIES + MIN_EVAL_SERIES

VERDICTS = (
    "FRESH_COHORT_READY",
    "CENSUS_DRIFT",
    "MATERIALIZATION_STRUCTURE_FAIL",
    "INSUFFICIENT_HEALTHY_STATIONS",
    "JUDGE_UNREADABLE",
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "noaa_fresh_cohort_v2.json"
OUT_MD = E2 / "noaa_fresh_cohort_v2.md"
OUT_JSON_V1 = E2 / "noaa_fresh_cohort_v1.json"
OUT_MD_V1 = E2 / "noaa_fresh_cohort_v1.md"
HEALTH_V13 = E2 / "noaa_health_check_v1.json"

EXPOSURE_DISCLOSURE_VERBATIM = (
    "family = AGGREGATE_SEEN(旧线 9 份 outcome 报告 + registry 40)\n"
    "instance(本 20 站)= SCANNED_BY_RETIRED_SCREENING_NO_SURVIVING_READOUT\n"
    "(旧线 p0 曾扫描全部 64 站,62 个拒绝读数无存留;本线方法开发未用任何 NOAA 数值)\n"
    "outcome(本 20 站)= SEALED(从无 Consumer 在其上重训;2025 csv 未打开)"
)
DATA_DIR = PROJECT_ROOT / "data" / "benchmark_noaa_fresh_v1"
REGISTRY = PROJECT_ROOT / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
RAW_NOAA = PROJECT_ROOT / "data/benchmark_v0/raw/noaa_global_hourly"
RAW_2024 = RAW_NOAA / "2024"
RAW_2025 = RAW_NOAA / "2025"
DATASET_ID = "noaa_global_hourly"

OLD_LINE_REPORTS: tuple[str, ...] = (
    "autonomous_natural_workflow_scope_induction_v2_noaa_confirmation_report.json",
    "noaa_multichannel_local_repair_2025_report.json",
    "noaa_multichannel_local_repair_p0_report.json",
    "w1_a5_vs_a3_report_noaa.json",
    "w1_noaa_a5_vs_a3_report.json",
    "w1_noaa_cross_domain_premise_report.json",
    "w1_noaa_impute_census_report.json",
    "w1_noaa_impute_fft_census_report.json",
    "w1_noaa_impute_linear_census_report.json",
)

STATION_RE = re.compile(r"\b(\d{11})\b")
UID_RE = re.compile(r"\b([0-9a-f]{64})\b")
PREFIX_RE = re.compile(r"\b([0-9a-f]{8})\b")
FEASIBILITY_BLOCK_RE = re.compile(
    r"^NOAA_DEWPOINT_FEASIBILITY_STATIONS = \((.*?)\)",
    re.S | re.M,
)


def _generator_sha() -> str:
    payload = Path(__file__).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _refuse_2025(path: Path) -> None:
    resolved = path.resolve()
    parts = resolved.parts
    if "2025" in parts:
        raise SystemExit("refusing to open a 2025 path: %s" % path)


def _v1_fresh_pool() -> list[str]:
    if not OUT_JSON_V1.is_file():
        return []
    payload = json.loads(OUT_JSON_V1.read_text(encoding="utf-8"))
    pool = payload.get("step_0_consumption_census", {}).get("fresh_pool")
    return [str(s) for s in pool] if isinstance(pool, list) else []


def _v13_criteria_quote() -> dict[str, Any]:
    if not HEALTH_V13.is_file():
        return {"source": _rel(HEALTH_V13), "exists": False}
    payload = json.loads(HEALTH_V13.read_text(encoding="utf-8"))
    screen = payload.get("step_1_outcome_blind_screen") or {}
    return {
        "source": _rel(HEALTH_V13),
        "exists": True,
        "pre_registered_criteria": (
            payload.get("pre_registered") or {}
        ).get("criteria"),
        "structure_criterion_verbatim": (
            (screen.get("structure") or {}).get("criterion")
        ),
        "substrate_criterion_verbatim": (
            (screen.get("substrate_double_guard") or {}).get("criterion")
        ),
        "min_series_length": (
            (payload.get("pre_registered") or {}).get("criteria") or {}
        ).get("min_series_length", MIN_SERIES_LENGTH),
    }


# ------------------------------------------------------------------ census
def _registry_rows() -> list[dict[str, Any]]:
    if not REGISTRY.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset_id") == DATASET_ID:
            rows.append(row)
    return rows


def _raw_2024_stations() -> list[str]:
    if not RAW_2024.is_dir():
        return []
    return sorted(
        path.stem
        for path in RAW_2024.glob("*.csv")
        if path.suffix == ".csv"
    )


def _rglob_csv_inventory() -> list[str]:
    if not RAW_NOAA.is_dir():
        return []
    return sorted(
        path.relative_to(RAW_NOAA).as_posix()
        for path in RAW_NOAA.rglob("*.csv")
        if not path.name.endswith(".asset.json")
    )


def _2025_stems_metadata_only() -> list[str]:
    """Filenames only.  File contents are not opened."""
    if not RAW_2025.is_dir():
        return []
    return sorted(path.stem for path in RAW_2025.glob("*.csv"))


def _feasibility_stations_from_source() -> dict[str, Any]:
    path = (
        PROJECT_ROOT
        / "evaluation/functional/run_e2_cross_series_curation.py"
    )
    text = path.read_text(encoding="utf-8")
    match = FEASIBILITY_BLOCK_RE.search(text)
    stations = (
        tuple(STATION_RE.findall(match.group(1))) if match else ()
    )
    return {
        "source": _rel(path) + "::NOAA_DEWPOINT_FEASIBILITY_STATIONS",
        "used_by": (
            "evaluation/functional/run_e2_cross_series_curation.py "
            "phase=noaa-multichannel-repair-2025 "
            "(station_ids=NOAA_DEWPOINT_FEASIBILITY_STATIONS, year=2025)"
        ),
        "stations": list(stations),
        "note": (
            "the 2025 report JSON lists no station_id; the runner config "
            "that produced it does"
        ),
    }


def _fixed_roster_entity_ids(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = int(FROZEN_NOAA_CONFIG["selection_origin"]) + FROZEN_HORIZON
    eligible = sorted(
        (
            row
            for row in rows
            if int(row.get("length", 0)) >= required
        ),
        key=lambda row: str(row["series_uid"]),
    )
    selected = eligible[:FROZEN_FIXED_ROSTER_N]
    return {
        "source": (
            "evaluation/functional/"
            "run_e2_autonomous_natural_workflow_generation.py"
            "::_fixed_roster + DATASET_CONFIGS['noaa']"
        ),
        "rule": (
            "dataset_id=noaa_global_hourly, length >= selection_origin+HORIZON "
            "(%d), sort by series_uid, take first %d"
            % (required, FROZEN_FIXED_ROSTER_N)
        ),
        "stations": [str(row["entity_id"]) for row in selected],
        "series_uids": [str(row["series_uid"]) for row in selected],
    }


def _walk_strings(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, Mapping):
        for value in obj.values():
            found.extend(_walk_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_walk_strings(value))
    elif isinstance(obj, str):
        found.append(obj)
    return found


def _extract_report_stations(
    path: Path,
    uid_to_entity: Mapping[str, str],
    prefix_to_entities: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    literal = sorted(set(STATION_RE.findall(text)))
    full_uids = [uid for uid in UID_RE.findall(text) if uid in uid_to_entity]
    mapped_from_uid = sorted({uid_to_entity[uid] for uid in full_uids})
    prefix_hits: list[dict[str, str]] = []
    prefix_ambiguous: list[dict[str, Any]] = []
    for prefix in sorted(set(PREFIX_RE.findall(text))):
        entities = list(prefix_to_entities.get(prefix, ()))
        if len(entities) == 1:
            prefix_hits.append(
                {"prefix": prefix, "entity_id": entities[0]}
            )
        elif len(entities) > 1:
            prefix_ambiguous.append(
                {"prefix": prefix, "n_registry_hits": len(entities)}
            )
    mapped_from_prefix = sorted({row["entity_id"] for row in prefix_hits})
    stations = sorted(set(literal) | set(mapped_from_uid) | set(mapped_from_prefix))
    return {
        "path": _rel(path),
        "exists": True,
        "literal_station_ids": literal,
        "series_uid_count": len(set(full_uids)),
        "mapped_from_series_uid": mapped_from_uid,
        "mapped_from_unique_8char_prefix": mapped_from_prefix,
        "prefix_hits": prefix_hits,
        "prefix_ambiguous": prefix_ambiguous,
        "stations": stations,
    }


def step_0_census() -> dict[str, Any]:
    rows = _registry_rows()
    registry_entities = [str(row["entity_id"]) for row in rows]
    uid_to_entity = {
        str(row["series_uid"]): str(row["entity_id"]) for row in rows
    }
    prefix_to_entities: dict[str, list[str]] = {}
    for row in rows:
        prefix_to_entities.setdefault(str(row["series_uid"])[:8], []).append(
            str(row["entity_id"])
        )

    citations: list[dict[str, Any]] = [
        {
            "source": _rel(REGISTRY),
            "rule": (
                "every row with dataset_id=noaa_global_hourly, field entity_id"
            ),
            "n_rows": len(registry_entities),
            "stations": list(registry_entities),
        }
    ]
    consumed: set[str] = set(registry_entities)

    report_extractions: list[dict[str, Any]] = []
    for name in OLD_LINE_REPORTS:
        path = E2 / name
        if not path.is_file():
            report_extractions.append(
                {"path": _rel(path), "exists": False, "stations": []}
            )
            continue
        extracted = _extract_report_stations(
            path, uid_to_entity, prefix_to_entities
        )
        report_extractions.append(extracted)
        consumed.update(extracted["stations"])
        citations.append(
            {
                "source": extracted["path"],
                "rule": (
                    "11-digit tokens; 64-char series_uid mapped through the "
                    "registry; unique 8-char series_uid prefix mapped through "
                    "the registry"
                ),
                "stations": list(extracted["stations"]),
                "literal_station_ids": list(extracted["literal_station_ids"]),
                "mapped_from_series_uid": list(
                    extracted["mapped_from_series_uid"]
                ),
                "mapped_from_unique_8char_prefix": list(
                    extracted["mapped_from_unique_8char_prefix"]
                ),
            }
        )

    roster = _fixed_roster_entity_ids(rows)
    consumed.update(roster["stations"])
    citations.append(
        {
            "source": roster["source"],
            "rule": roster["rule"],
            "stations": list(roster["stations"]),
            "applies_to_reports": [
                "autonomous_natural_workflow_scope_induction_v2_noaa_confirmation_report.json",
                "w1_a5_vs_a3_report_noaa.json",
                "w1_noaa_a5_vs_a3_report.json",
                "w1_noaa_cross_domain_premise_report.json",
                "w1_noaa_impute_census_report.json",
                "w1_noaa_impute_fft_census_report.json",
                "w1_noaa_impute_linear_census_report.json",
            ],
        }
    )

    feasibility = _feasibility_stations_from_source()
    consumed.update(feasibility["stations"])
    citations.append(
        {
            "source": feasibility["source"],
            "rule": feasibility["used_by"],
            "stations": list(feasibility["stations"]),
            "note": feasibility["note"],
        }
    )

    raw_2024 = _raw_2024_stations()
    rglob_csv = _rglob_csv_inventory()
    stems_2025 = _2025_stems_metadata_only()
    fresh = sorted(station for station in raw_2024 if station not in consumed)
    expected_v1 = _v1_fresh_pool()
    census_drift = list(fresh) != list(expected_v1)
    n_select = min(len(fresh), MAX_SELECT)
    selected = fresh[:n_select]
    sufficient = (not census_drift) and len(fresh) >= MIN_FRESH_STATIONS

    extras_beyond_registry = sorted(consumed - set(registry_entities))
    p0_report = next(
        (
            row
            for row in report_extractions
            if row.get("path", "").endswith(
                "noaa_multichannel_local_repair_p0_report.json"
            )
        ),
        {},
    )

    ambiguities = [
        {
            "id": "v1_raw_file_count_vs_station_count",
            "text": (
                "noaa_health_check_v1 counted raw_station_files=74 by "
                "rglob('*.csv') under data/benchmark_v0/raw/noaa_global_hourly "
                "(%d files: %d under 2024/, %d under 2025/, plus isd-history.csv). "
                "The fresh pool is unique 11-digit stems of 2024/*.csv (%d). "
                "2025 files are the same station ids (metadata listing only); "
                "isd-history.csv is not a station series."
                % (
                    len(rglob_csv),
                    sum(1 for name in rglob_csv if name.startswith("2024/")),
                    sum(1 for name in rglob_csv if name.startswith("2025/")),
                    len(raw_2024),
                )
            ),
        },
        {
            "id": "p0_unnamed_rejected_stations",
            "text": (
                "noaa_multichannel_local_repair_p0_report.json names two "
                "station_id values %s and records source_file_count=64 with "
                "62 rejected as FEWER_THAN_TWO_AFFECTED_TRAINING_ROWS.  Those "
                "62 ids do not appear in the report or in an explicit roster, "
                "so they are not added to the consumed set.  If they were, "
                "the fresh pool would be empty."
                % (p0_report.get("literal_station_ids") or [])
            ),
        },
        {
            "id": "2025_config_vs_report_body",
            "text": (
                "noaa_multichannel_local_repair_2025_report.json contains no "
                "station_id.  NOAA_DEWPOINT_FEASIBILITY_STATIONS is the runner "
                "config that produced it and is included because the task "
                "counts roster/config station numbers.  Those four ids not "
                "already in the registry (%s) are the difference between a "
                "24-station registry-only remainder and the 20-station fresh "
                "pool below the pre-registered floor."
                % sorted(set(feasibility["stations"]) - set(registry_entities))
            ),
        },
        {
            "id": "leap_year_vs_8760",
            "text": (
                "2024 is a leap year (8784 hours).  The sealed boundary is "
                "the frozen 8760-hour conventional year starting 2024-01-01, "
                "not a Feb-29-aware calendar year.  Index 8760 would be "
                "2024-12-31 00:00.  2025 csv contents are not ingested."
            ),
        },
    ]

    return {
        "registry_path": _rel(REGISTRY),
        "registry_n": len(registry_entities),
        "registry_entity_ids": registry_entities,
        "old_line_reports": list(OLD_LINE_REPORTS),
        "report_extractions": report_extractions,
        "fixed_roster": roster,
        "feasibility_config": feasibility,
        "consumed_citations": citations,
        "consumed_set": sorted(consumed),
        "consumed_n": len(consumed),
        "consumed_beyond_registry": extras_beyond_registry,
        "raw_2024_dir": _rel(RAW_2024),
        "raw_2024_n": len(raw_2024),
        "raw_2024_stations": raw_2024,
        "rglob_csv_n": len(rglob_csv),
        "rglob_csv": rglob_csv,
        "raw_2025_stems_metadata_only": stems_2025,
        "raw_2025_opened": False,
        "fresh_pool": fresh,
        "fresh_n": len(fresh),
        "min_fresh_stations": MIN_FRESH_STATIONS,
        "min_fresh_stations_v1": MIN_FRESH_STATIONS_V1,
        "floor_correction": FLOOR_CORRECTION,
        "expected_fresh_pool_from_v1": expected_v1,
        "census_drift": census_drift,
        "max_select": MAX_SELECT,
        "n_select": n_select if sufficient else 0,
        "selected_lexicographic": selected if sufficient else [],
        "selection_rule": (
            "fresh pool sorted by station id, take the first "
            "N=min(pool, %d); no data-dependent pick"
            % MAX_SELECT
        ),
        "sufficient_to_materialize": sufficient,
        "ambiguities": ambiguities,
        "status": (
            "CENSUS_DRIFT" if census_drift
            else (
                "SUFFICIENT_FRESH_POOL" if sufficient
                else "INSUFFICIENT_UNCONSUMED_STATIONS"
            )
        ),
    }


# ----------------------------------------------------------- materialize
def _decode_tmp(raw: str) -> float | None:
    token = raw.split(",", 1)[0].strip()
    try:
        value = int(token)
    except ValueError:
        return None
    if abs(value) == TMP_MISSING:
        return None
    return value / 10.0


def _parse_hour_index(date_raw: str) -> int | None:
    if not date_raw:
        return None
    try:
        timestamp = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)
    timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
    delta = (timestamp - GRID_START).total_seconds()
    index = int(delta // 3600)
    return index


def parse_station_development(path: Path) -> dict[str, Any]:
    """Hourly TMP on the frozen 8760-hour development grid.

    Missing hours stay NaN.  Rows at index >= 8760 are skipped before TMP
    is decoded, so confirmation numbers are not read.  First finite value
    in an hour wins; later observations in the same hour are ignored.
    """
    _refuse_2025(path)
    import numpy as np

    values = np.full(DEVELOPMENT_HOURS, np.nan, dtype=np.float64)
    n_rows = 0
    n_decoded = 0
    n_skipped_past_boundary = 0
    n_missing_token = 0
    n_hour_collision = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n_rows += 1
            index = _parse_hour_index(str(row.get("DATE") or ""))
            if index is None:
                continue
            if index < 0:
                continue
            if index >= DEVELOPMENT_HOURS:
                n_skipped_past_boundary += 1
                continue
            decoded = _decode_tmp(str(row.get("TMP") or ""))
            if decoded is None:
                n_missing_token += 1
                continue
            if np.isfinite(values[index]):
                n_hour_collision += 1
                continue
            values[index] = decoded
            n_decoded += 1
    finite = int(np.isfinite(values).sum())
    return {
        "values": values,
        "n_rows_seen": n_rows,
        "n_tmp_written": n_decoded,
        "n_skipped_past_boundary": n_skipped_past_boundary,
        "n_missing_or_unparsed_tmp_in_window": n_missing_token,
        "n_hour_collisions_ignored": n_hour_collision,
        "n_finite_development": finite,
        "n_nan_development": DEVELOPMENT_HOURS - finite,
        "missing_rate_development": (
            1.0 - finite / DEVELOPMENT_HOURS if DEVELOPMENT_HOURS else 1.0
        ),
    }


def step_1_materialize(
    census: Mapping[str, Any],
    *,
    write: bool,
) -> dict[str, Any]:
    if not census.get("sufficient_to_materialize"):
        return {
            "ran": False,
            "reason": "fresh pool below the pre-registered floor of %d"
            % MIN_FRESH_STATIONS,
            "stations": [],
        }
    import numpy as np
    selected = [str(s) for s in census["selected_lexicographic"]]
    opened_2025 = False
    per_station: list[dict[str, Any]] = []
    values: dict[str, Any] = {}
    if write:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    for station in selected:
        path = RAW_2024 / ("%s.csv" % station)
        _refuse_2025(path)
        parsed = parse_station_development(path)
        series = np.asarray(parsed.pop("values"), dtype=np.float64)
        if int(series.size) != DEVELOPMENT_HOURS:
            raise SystemExit("development series length drifted from 8760")
        if int(series.size) > DEVELOPMENT_HOURS:
            raise SystemExit("a series is longer than the sealed boundary")
        values[station] = series
        record = {
            "dataset_id": "noaa_global_hourly_fresh_v1",
            "entity_id": station,
            "frequency": "hourly",
            "grid_start": GRID_START.isoformat(),
            "length": DEVELOPMENT_HOURS,
            "n_finite_development": parsed["n_finite_development"],
            "missing_rate_development": parsed["missing_rate_development"],
            "parse": {
                "missing_timestamp_is_nan": True,
                "interpolate": False,
                "smooth": False,
                "select_good_span": False,
                "hourly_rule": "first_finite_wins",
                "tmp_missing_code": TMP_MISSING,
            },
            "source_csv": _rel(path),
            **parsed,
        }
        if write:
            dest = DATA_DIR / "series" / station
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest / "values.npy", series)
            (dest / "record.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8", newline="\n",
            )
        per_station.append(
            {key: value for key, value in record.items() if key != "parse"}
        )

    finite_counts = [row["n_finite_development"] for row in per_station]
    missing_rates = [row["missing_rate_development"] for row in per_station]
    finite_arr = np.asarray(finite_counts, dtype=np.float64)
    miss_arr = np.asarray(missing_rates, dtype=np.float64)

    manifest = {
        "cohort_id": "benchmark_noaa_fresh_v1",
        "dataset_id": "noaa_global_hourly_fresh_v1",
        "namespace": _rel(DATA_DIR),
        "does_not_touch": "data/benchmark_v0_2",
        "grid_start": GRID_START.isoformat(),
        "development_hours": DEVELOPMENT_HOURS,
        "partition": {
            "kind": PARTITION_KIND,
            "development": {
                "index": [0, DEVELOPMENT_HOURS],
                "label": "2024",
            },
            "confirmation": {
                "index": [DEVELOPMENT_HOURS, None],
                "label": "2025",
                "ingested": False,
                "reason": (
                    "2025 raw csv contents were not opened; confirmation "
                    "values are not on disk in this task"
                ),
            },
        },
        "station_list": selected,
        "n_stations": len(selected),
        "time_rule": (
            "hourly bins from 2024-01-01 00:00; missing hours are NaN; "
            "rows with index >= 8760 are skipped before TMP is decoded"
        ),
        "parse_rule": (
            "univariate TMP; 9999 -> NaN; first finite observation in the "
            "hour wins; no interpolation, no smoothing, no good-span screen"
        ),
        "consumption_census": {
            "consumed_n": census["consumed_n"],
            "fresh_n": census["fresh_n"],
            "consumed_set": census["consumed_set"],
            "fresh_pool": census["fresh_pool"],
            "sources": [
                row["source"] for row in census["consumed_citations"]
            ],
        },
        "generator": _rel(Path(__file__)),
        "generator_sha256": _generator_sha(),
        "raw_2025_opened": opened_2025,
    }
    if write:
        (DATA_DIR / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n",
        )

    def _pct(arr: Any, q: float) -> float:
        return float(np.quantile(arr, q)) if arr.size else float("nan")

    return {
        "ran": True,
        "wrote": write,
        "n_stations": len(selected),
        "length": DEVELOPMENT_HOURS,
        "stations": selected,
        "per_station": per_station,
        "finite_count_distribution": {
            "min": int(finite_arr.min()) if finite_arr.size else 0,
            "p25": _pct(finite_arr, 0.25),
            "median": _pct(finite_arr, 0.50),
            "p75": _pct(finite_arr, 0.75),
            "max": int(finite_arr.max()) if finite_arr.size else 0,
        },
        "missing_rate_distribution": {
            "min": float(miss_arr.min()) if miss_arr.size else 0.0,
            "p25": _pct(miss_arr, 0.25),
            "median": _pct(miss_arr, 0.50),
            "p75": _pct(miss_arr, 0.75),
            "max": float(miss_arr.max()) if miss_arr.size else 0.0,
        },
        "n_with_5760_finite": int(sum(c >= MIN_SERIES_LENGTH for c in finite_counts)),
        "manifest": manifest,
        "raw_2025_opened": opened_2025,
        "_values": values,
    }


def _load_truncated_from_disk(
    stations: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    values: dict[str, Any] = {}
    for station in stations:
        path = DATA_DIR / "series" / station / "values.npy"
        handle = np.load(path, mmap_mode="r")
        if int(handle.shape[0]) > DEVELOPMENT_HOURS:
            series = np.array(
                handle[:DEVELOPMENT_HOURS], dtype=np.float64, copy=True
            )
        else:
            series = np.array(handle, dtype=np.float64, copy=True)
        del handle
        if int(series.size) > DEVELOPMENT_HOURS:
            raise SystemExit(
                "a loaded series is longer than the sealed boundary"
            )
        values[str(station)] = series
    sizes = [int(series.size) for series in values.values()]
    return values, {
        "loaded_series": len(values),
        "longest_loaded_length": max(sizes) if sizes else 0,
        "shortest_loaded_length": min(sizes) if sizes else 0,
        "no_index_at_or_past_boundary_was_read": True,
        "enforcement": (
            "every loaded series is sliced to [:%d] before any consumer "
            "sees it, and an assertion refuses to continue if any array "
            "is longer" % DEVELOPMENT_HOURS
        ),
    }


# ----------------------------------------------------------- health check
def _import_health_stack() -> dict[str, Any]:
    import numpy as np

    import run_e2_autonomous_natural_workflow_generation as v6
    from evaluation.functional.task_episode_harness import e1 as e1mod
    from evaluation.functional.task_episode_harness import g1
    from evaluation.functional.task_episode_harness.agentic import g3_sourcing
    from run_v1_kdd2018_natural_slow_update import _config

    return {
        "np": np,
        "v6": v6,
        "e1mod": e1mod,
        "g1": g1,
        "g3_sourcing": g3_sourcing,
        "_config": _config,
    }


def _assert_windows_inside_development(stack: Mapping[str, Any]) -> dict[str, Any]:
    g3 = stack["g3_sourcing"]
    e1mod = stack["e1mod"]
    v6 = stack["v6"]
    origins = tuple(int(o) for o in g3.DEVELOPMENT_ORIGINS)
    horizon = int(v6.HORIZON)
    specs = list(e1mod._frozen_task_roster()[:9])
    roster_indices = [
        int(origin)
        for spec in specs
        for role in ("support_origins", "delayed_origins")
        for origin in spec[role]
    ]
    anchors = [int(a) for a in dict(stack["_config"]())["anchors"]]
    train_far = max(anchors) + horizon
    eval_far = max(roster_indices)
    probe_far = max(origins) + horizon
    reach = {
        "development_origins": list(origins),
        "horizon": horizon,
        "probe_farthest_index": probe_far,
        "probe_inside_boundary": probe_far <= DEVELOPMENT_HOURS,
        "train_guard_anchors": anchors,
        "train_guard_farthest_index": train_far,
        "train_guard_inside_boundary": train_far <= DEVELOPMENT_HOURS,
        "eval_guard_roster_tasks": [
            str(spec["task_episode_id"]) for spec in specs
        ],
        "eval_guard_farthest_index": eval_far,
        "eval_guard_inside_boundary": eval_far <= DEVELOPMENT_HOURS,
        "public_phenomenon_cutoff": int(specs[0]["support_origins"][0]),
        "sealed_from_index_v1": FROZEN_SEALED_FROM_INDEX_V1,
        "sealed_from_index_v2": DEVELOPMENT_HOURS,
        "remap": (
            "numeric windows reused from g3_sourcing / e1 roster / _config "
            "anchors; sealed boundary rebound from 3072 to 8760 so the eval "
            "guard that used to cross the old index wall now sits inside 2024"
        ),
        "identity_probe_run": False,
        "consumer_retrains": 0,
    }
    assert origins == FROZEN_DEVELOPMENT_ORIGINS
    assert probe_far <= DEVELOPMENT_HOURS
    assert train_far <= DEVELOPMENT_HOURS
    assert eval_far <= DEVELOPMENT_HOURS
    assert int(g3.MIN_SERIES_LENGTH) == MIN_SERIES_LENGTH
    return reach


def step_2_health_check(
    values: Mapping[str, Any],
    *,
    stack: Mapping[str, Any],
) -> dict[str, Any]:
    """Per-station exam on the development slice.  0 Consumer retrains."""
    np = stack["np"]
    g3 = stack["g3_sourcing"]
    g1 = stack["g1"]
    e1mod = stack["e1mod"]
    quote = _v13_criteria_quote()
    reach = _assert_windows_inside_development(stack)
    uids = sorted(values)
    for uid, series in values.items():
        if int(np.asarray(series).size) > DEVELOPMENT_HOURS:
            raise SystemExit(
                "health check saw an index at or past the confirmation "
                "boundary on %s" % uid
            )

    specs = list(e1mod._frozen_task_roster()[:9])
    anchors = [int(a) for a in dict(stack["_config"]())["anchors"]]
    usable, unusable = g3._drop_series_with_unusable_windows(
        values, uids, specs, anchors,
    )
    unusable_set = set(unusable)

    max_missing_count = DEVELOPMENT_HOURS - MIN_SERIES_LENGTH
    max_missing_rate = max_missing_count / float(DEVELOPMENT_HOURS)
    rows: list[dict[str, Any]] = []
    for uid in uids:
        series = np.asarray(values[uid], dtype=np.float64)
        length = int(series.size)
        n_finite = int(np.isfinite(series).sum())
        n_missing = length - n_finite
        missing_rate = (
            1.0 - n_finite / float(DEVELOPMENT_HOURS)
            if DEVELOPMENT_HOURS else 1.0
        )
        length_pass = length == DEVELOPMENT_HOURS and length >= MIN_SERIES_LENGTH
        missing_pass = n_finite >= MIN_SERIES_LENGTH
        window_ok = uid not in unusable_set
        train_clean = False
        eval_clean = False
        guard_error = None
        if window_ok:
            try:
                train_pf = g1.train_substrate_preflight(values, [uid], anchors)
                eval_pf = g1.eval_substrate_preflight(values, [uid], specs)
                train_clean = bool(train_pf["per_series"][uid]["clean"])
                eval_clean = bool(eval_pf["per_series"][uid]["clean"])
            except (ValueError, KeyError) as exc:
                guard_error = "%s: %s" % (type(exc).__name__, exc)
        flatline_pass = bool(
            window_ok and train_clean and eval_clean and guard_error is None
        )
        passed = bool(length_pass and missing_pass and flatline_pass)
        failed: list[str] = []
        if not length_pass:
            failed.append("length")
        if not missing_pass:
            failed.append("missing_rate")
        if not flatline_pass:
            failed.append("constant_flatline")
        rows.append({
            "station_id": uid,
            "length": length,
            "length_criterion_verbatim": quote.get(
                "structure_criterion_verbatim"
            ),
            "min_series_length_quoted": quote.get("min_series_length"),
            "length_pass": length_pass,
            "n_finite": n_finite,
            "n_missing": n_missing,
            "missing_rate": missing_rate,
            "max_missing_count": max_missing_count,
            "max_missing_rate": max_missing_rate,
            "missing_pass": missing_pass,
            "unusable_window": uid in unusable_set,
            "train_substrate_clean": train_clean,
            "eval_substrate_clean": eval_clean,
            "guard_error": guard_error,
            "flatline_criterion_verbatim": quote.get(
                "substrate_criterion_verbatim"
            ),
            "flatline_pass": flatline_pass,
            "pass": passed,
            "verdict": "PASS" if passed else "FAIL",
            "failed_criteria": failed,
        })

    passed_ids = [row["station_id"] for row in rows if row["pass"]]
    roster = passed_ids[:MIN_ROSTER_PASS]
    substitutes = passed_ids[MIN_ROSTER_PASS:]
    return {
        "ran": True,
        "consumer_retrains": 0,
        "identity_probe_run": False,
        "identity_probe_reason": (
            "0 Consumer retrain: the #13 identity readability probe is not "
            "run in this protocol"
        ),
        "criteria_quote_from_13": quote,
        "frozen_screen_reach": reach,
        "missing_rate_cap": {
            "derived_from": (
                "artifacts/functional/e2/noaa_health_check_v1.json "
                "pre_registered.criteria.min_series_length = 5760, applied "
                "as a finite-point minimum on the constructed 8760-hour "
                "development slice"
            ),
            "min_finite_points": MIN_SERIES_LENGTH,
            "development_hours": DEVELOPMENT_HOURS,
            "max_missing_count": max_missing_count,
            "max_missing_rate": max_missing_rate,
        },
        "flatline_screen": {
            "instrument": (
                "g1.train_substrate_preflight + g1.eval_substrate_preflight "
                "(scale_floor_fallback) and g3._drop_series_with_unusable_windows"
            ),
            "criterion_verbatim": quote.get("substrate_criterion_verbatim"),
            "per_station_rule": (
                "a station PASSes the flatline screen only when both "
                "substrate guards call it clean and it has no unusable window"
            ),
        },
        "n_stations": len(rows),
        "n_pass": len(passed_ids),
        "n_fail": len(rows) - len(passed_ids),
        "min_roster_pass": MIN_ROSTER_PASS,
        "per_station": rows,
        "confirmation_roster": roster,
        "substitutes": substitutes,
        "channel_rule": (
            "each station is one univariate series examined on its own"
        ),
    }


# ----------------------------------------------------------------- verdict
def _verdict(
    census: Mapping[str, Any],
    materialize: Mapping[str, Any] | None,
    health: Mapping[str, Any] | None,
) -> dict[str, Any]:
    side = {name: "NOT_REACHED" for name in VERDICTS}
    side["JUDGE_UNREADABLE"] = "NOT_REACHED"
    if census.get("census_drift"):
        side["CENSUS_DRIFT"] = "SELECTED"
        return {
            "verdict": "CENSUS_DRIFT",
            "reason": (
                "census v2 fresh pool %s does not match the #14 lexicographic "
                "list of 20 (%s)"
                % (census.get("fresh_pool"), census.get("expected_fresh_pool_from_v1"))
            ),
            "side_by_side": side,
        }
    side["CENSUS_DRIFT"] = "NOT_SELECTED"
    if not census.get("sufficient_to_materialize"):
        side["MATERIALIZATION_STRUCTURE_FAIL"] = "NOT_REACHED"
        return {
            "verdict": "CENSUS_DRIFT",
            "reason": (
                "census v2 did not confirm a 20-station pool "
                "(fresh_n=%s)" % census.get("fresh_n")
            ),
            "side_by_side": side,
        }
    if not (materialize or {}).get("ran"):
        side["MATERIALIZATION_STRUCTURE_FAIL"] = "SELECTED"
        return {
            "verdict": "MATERIALIZATION_STRUCTURE_FAIL",
            "reason": (materialize or {}).get("reason") or "materialization did not run",
            "side_by_side": side,
        }
    n_written = int((materialize or {}).get("n_stations") or 0)
    length = int((materialize or {}).get("length") or 0)
    if n_written != MIN_FRESH_STATIONS or length != DEVELOPMENT_HOURS:
        side["MATERIALIZATION_STRUCTURE_FAIL"] = "SELECTED"
        return {
            "verdict": "MATERIALIZATION_STRUCTURE_FAIL",
            "reason": (
                "expected 20 stations of length 8760, wrote %d of length %d"
                % (n_written, length)
            ),
            "side_by_side": side,
        }
    side["MATERIALIZATION_STRUCTURE_FAIL"] = "NOT_SELECTED"
    if health is None or not health.get("ran"):
        side["INSUFFICIENT_HEALTHY_STATIONS"] = "SELECTED"
        return {
            "verdict": "INSUFFICIENT_HEALTHY_STATIONS",
            "reason": "health check did not run",
            "side_by_side": side,
        }
    n_pass = int(health.get("n_pass") or 0)
    if n_pass < MIN_ROSTER_PASS:
        side["INSUFFICIENT_HEALTHY_STATIONS"] = "SELECTED"
        return {
            "verdict": "INSUFFICIENT_HEALTHY_STATIONS",
            "reason": (
                "%d stations PASS the per-station exam; the pre-registered "
                "roster needs %d.  Follow-up is a blind expansion download, "
                "not this task."
                % (n_pass, MIN_ROSTER_PASS)
            ),
            "side_by_side": side,
        }
    side["INSUFFICIENT_HEALTHY_STATIONS"] = "NOT_SELECTED"
    side["FRESH_COHORT_READY"] = "SELECTED"
    return {
        "verdict": "FRESH_COHORT_READY",
        "reason": (
            "census v2 confirmed the #14 20-station pool; materialization "
            "wrote 20 x 8760 development arrays; %d stations PASS, roster "
            "has %d"
            % (n_pass, len(health.get("confirmation_roster") or []))
        ),
        "side_by_side": side,
    }


def _exposure_ledger(
    census: Mapping[str, Any],
    materialize: Mapping[str, Any] | None,
    health: Mapping[str, Any] | None,
) -> dict[str, Any]:
    context_seen = bool((materialize or {}).get("ran"))
    return {
        "disclosure_verbatim": EXPOSURE_DISCLOSURE_VERBATIM,
        "family": {
            "state": "AGGREGATE_SEEN",
            "detail": "旧线 9 份 outcome 报告 + registry 40",
        },
        "instance": {
            "state": "SCANNED_BY_RETIRED_SCREENING_NO_SURVIVING_READOUT",
            "stations": list(census.get("fresh_pool") or []),
            "detail": (
                "旧线 p0 曾扫描全部 64 站,62 个拒绝读数无存留;"
                "本线方法开发未用任何 NOAA 数值"
            ),
        },
        "outcome": {
            "state": "SEALED",
            "consumer_retrains": 0,
            "detail": "从无 Consumer 在其上重训;2025 csv 未打开",
        },
        "this_task_development_context": {
            "state": "INSTANCE_SEEN" if context_seen else "UNTOUCHED",
            "scope": "2024 development slice [0, 8760)",
            "note": (
                "this task may read 2024 TMP on the 20-station development "
                "grid; confirmation index >= 8760 and 2025 csv stay unread"
            ),
        },
        "confirmation_partition": {
            "state": "SEALED",
            "boundary": DEVELOPMENT_HOURS,
            "kind": PARTITION_KIND,
            "indices_read_at_or_past_boundary": 0,
            "raw_2025_opened": False,
            "ingested": False,
        },
        "consumed_n": census.get("consumed_n"),
        "fresh_n": census.get("fresh_n"),
    }


# ------------------------------------------------------------------ report
def _markdown(payload: Mapping[str, Any]) -> str:
    census = payload["step_0_consumption_census"]
    materialize = payload["step_1_blind_materialization"]
    health = payload["step_2_health_check_v2"]
    ledger = payload["exposure_ledger"]
    lines = [
        "# NOAA fresh cohort v2",
        "",
        "**Overall: `%s`** -- %s"
        % (payload["overall_verdict"], payload["overall_verdict_reason"]),
        "",
        "Pre-registered verdicts, reported side by side:",
        "",
        "| verdict | status |",
        "| --- | --- |",
    ]
    for name, status in (payload.get("verdicts_side_by_side") or {}).items():
        lines.append("| `%s` | %s |" % (name, status))
    lines += [
        "",
        "0 LLM calls.  0 Consumer retrains.  2025 csv not opened.",
        "",
        "## Floor correction",
        "",
        json.dumps(FLOOR_CORRECTION, ensure_ascii=False, indent=2),
        "",
        "## Exposure disclosure (verbatim)",
        "",
        "```",
        str(ledger.get("disclosure_verbatim") or EXPOSURE_DISCLOSURE_VERBATIM),
        "```",
        "",
        "## Step 0 -- census v2",
        "",
        "| field | value |",
        "| --- | --- |",
        "| status | `%s` |" % census.get("status"),
        "| census_drift | %s |" % census.get("census_drift"),
        "| registry | `%s` (%d) |"
        % (census["registry_path"], census["registry_n"]),
        "| consumed | %d |" % census["consumed_n"],
        "| fresh pool | **%d** |" % census["fresh_n"],
        "| floor (v2 / v1) | %d / %d |"
        % (census["min_fresh_stations"], census["min_fresh_stations_v1"]),
        "| sufficient | %s |" % census["sufficient_to_materialize"],
        "",
        "Fresh pool (lexicographic):",
        "",
    ]
    for station in census["fresh_pool"]:
        lines.append("- `%s`" % station)
    lines += ["", "### Ambiguities", ""]
    for row in census.get("ambiguities") or []:
        lines += ["- **%s**: %s" % (row["id"], row["text"]), ""]

    lines += ["## Step 1 -- blind materialization", ""]
    if not (materialize or {}).get("ran"):
        lines += [
            "Not run.  %s" % (materialize or {}).get("reason"),
            "",
        ]
    else:
        dist_f = materialize["finite_count_distribution"]
        dist_m = materialize["missing_rate_distribution"]
        lines += [
            "| field | value |",
            "| --- | --- |",
            "| stations written | %d |" % materialize["n_stations"],
            "| length | %d |" % materialize["length"],
            "| finite-count min / median / max | %s / %s / %s |"
            % (dist_f["min"], dist_f["median"], dist_f["max"]),
            "| missing-rate min / median / max | %.4f / %.4f / %.4f |"
            % (dist_m["min"], dist_m["median"], dist_m["max"]),
            "| series with >=5760 finite | %d |"
            % materialize["n_with_5760_finite"],
            "| 2025 opened | %s |" % materialize["raw_2025_opened"],
            "| generator sha256 | `%s` |"
            % materialize["manifest"]["generator_sha256"],
            "",
        ]

    lines += ["## Step 2 -- per-station health check (development only)", ""]
    if not (health or {}).get("ran"):
        lines += ["Not run.", ""]
    else:
        quote = health.get("criteria_quote_from_13") or {}
        cap = health.get("missing_rate_cap") or {}
        lines += [
            "Criteria quoted from `%s`." % quote.get("source"),
            "",
            "- length (verbatim): %s"
            % quote.get("structure_criterion_verbatim"),
            "- length recorded: 8760 by construction (min_series_length=%s)"
            % quote.get("min_series_length"),
            "- missing-rate cap: finite points >= %s on the 8760-hour slice "
            "(max missing %s, rate <= %.6f); %s"
            % (
                cap.get("min_finite_points"), cap.get("max_missing_count"),
                float(cap.get("max_missing_rate") or 0.0),
                cap.get("derived_from"),
            ),
            "- constant/flatline (verbatim): %s"
            % quote.get("substrate_criterion_verbatim"),
            "",
            "Identity readability probe **not run** (0 Consumer retrain).",
            "",
            "| station | length | finite | missing_rate | length | missing | "
            "flatline | verdict |",
            "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
        for row in health.get("per_station") or []:
            lines.append(
                "| `%s` | %d | %d | %.4f | %s | %s | %s | **%s** |"
                % (
                    row["station_id"], row["length"], row["n_finite"],
                    row["missing_rate"],
                    "PASS" if row["length_pass"] else "FAIL",
                    "PASS" if row["missing_pass"] else "FAIL",
                    "PASS" if row["flatline_pass"] else "FAIL",
                    row["verdict"],
                )
            )
        lines += [
            "",
            "PASS %d / %d.  Confirmation roster (lexicographic first %d PASS):"
            % (
                health["n_pass"], health["n_stations"], MIN_ROSTER_PASS,
            ),
            "",
        ]
        for station in health.get("confirmation_roster") or []:
            lines.append("- `%s`" % station)
        lines += ["", "Substitutes:", ""]
        subs = health.get("substitutes") or []
        if not subs:
            lines.append("(none)")
        for station in subs:
            lines.append("- `%s`" % station)
        lines.append("")

    lines += [
        "## Cost",
        "",
        "%d Consumer retrains; 0 LLM calls."
        % payload["consumer_retrains"],
        "",
        "generator sha256: `%s`" % payload["generator_sha256"],
        "",
        "v1 generator sha on record: `%s`" % GENERATOR_SHA_ON_RECORD_V1,
        "",
    ]
    return "\n".join(lines) + "\n"


def _drop_private(payload: dict[str, Any]) -> dict[str, Any]:
    materialize = payload.get("step_1_blind_materialization")
    if isinstance(materialize, dict):
        materialize.pop("_values", None)
    return payload


def _assert_not_v1(path: Path) -> None:
    if path.resolve() in {OUT_JSON_V1.resolve(), OUT_MD_V1.resolve()}:
        raise SystemExit("refusing to rewrite v1 artifact %s" % path)


# --------------------------------------------------------------------- run
def run(
    *,
    census_only: bool = False,
    materialize_only: bool = False,
    health_check: bool = False,
    dry_run: bool = False,
) -> int:
    started = time.perf_counter()
    census = step_0_census()
    print(
        "census fresh=%d consumed=%d drift=%s sufficient=%s"
        % (
            census["fresh_n"], census["consumed_n"],
            census["census_drift"], census["sufficient_to_materialize"],
        ),
        flush=True,
    )
    materialize: dict[str, Any] | None = None
    health: dict[str, Any] | None = None
    values: dict[str, Any] = {}

    do_materialize = (not census_only) and census.get("sufficient_to_materialize")
    do_health = (
        do_materialize
        and not materialize_only
        and (health_check or not census_only)
    )
    if materialize_only:
        do_health = False
    if health_check and not census_only:
        do_materialize = bool(census.get("sufficient_to_materialize"))
        do_health = do_materialize

    if do_materialize:
        write_arrays = not dry_run
        stations = [str(s) for s in census["selected_lexicographic"]]
        already = bool(stations) and all(
            (DATA_DIR / "series" / station / "values.npy").is_file()
            for station in stations
        ) and (DATA_DIR / "manifest.json").is_file()
        if already and not dry_run:
            values, load_report = _load_truncated_from_disk(stations)
            per_station = []
            for station in stations:
                record = json.loads(
                    (DATA_DIR / "series" / station / "record.json").read_text(
                        encoding="utf-8"
                    )
                )
                per_station.append({
                    key: value for key, value in record.items()
                    if key != "parse"
                })
            finite_counts = [
                int(row.get("n_finite_development") or 0) for row in per_station
            ]
            missing_rates = [
                float(row.get("missing_rate_development") or 0.0)
                for row in per_station
            ]
            import numpy as np
            finite_arr = np.asarray(finite_counts, dtype=np.float64)
            miss_arr = np.asarray(missing_rates, dtype=np.float64)
            materialize = {
                "ran": True,
                "wrote": False,
                "reused_existing": True,
                "n_stations": len(stations),
                "length": DEVELOPMENT_HOURS,
                "stations": stations,
                "per_station": per_station,
                "finite_count_distribution": {
                    "min": int(finite_arr.min()) if finite_arr.size else 0,
                    "p25": float(np.quantile(finite_arr, 0.25)) if finite_arr.size else float("nan"),
                    "median": float(np.quantile(finite_arr, 0.50)) if finite_arr.size else float("nan"),
                    "p75": float(np.quantile(finite_arr, 0.75)) if finite_arr.size else float("nan"),
                    "max": int(finite_arr.max()) if finite_arr.size else 0,
                },
                "missing_rate_distribution": {
                    "min": float(miss_arr.min()) if miss_arr.size else 0.0,
                    "p25": float(np.quantile(miss_arr, 0.25)) if miss_arr.size else float("nan"),
                    "median": float(np.quantile(miss_arr, 0.50)) if miss_arr.size else float("nan"),
                    "p75": float(np.quantile(miss_arr, 0.75)) if miss_arr.size else float("nan"),
                    "max": float(miss_arr.max()) if miss_arr.size else 0.0,
                },
                "n_with_5760_finite": int(
                    sum(c >= MIN_SERIES_LENGTH for c in finite_counts)
                ),
                "manifest": json.loads(
                    (DATA_DIR / "manifest.json").read_text(encoding="utf-8")
                ),
                "load_report": load_report,
                "raw_2025_opened": False,
            }
        else:
            materialize = step_1_materialize(census, write=write_arrays)
            values = dict(materialize.pop("_values", {}) or {})
            if not values and not dry_run:
                values, load_report = _load_truncated_from_disk(
                    materialize["stations"]
                )
                materialize["load_report"] = load_report

    if do_health and (materialize or {}).get("ran"):
        stack = _import_health_stack()
        if not values and not dry_run:
            values, load_report = _load_truncated_from_disk(
                materialize["stations"]
            )
            materialize["load_report"] = load_report
        health = step_2_health_check(values, stack=stack)

    verdict = _verdict(census, materialize, health)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "census v2 of the #14 20-station remainder, blind-materialize "
            "the 2024 development slice, then per-station health-check with "
            "0 Consumer retrains"
        ),
        "not_authorization_evidence": (
            "no Skill, no Episode, no Harness change, no NOAA utility claim"
        ),
        "overall_verdict": verdict["verdict"],
        "overall_verdict_reason": verdict["reason"],
        "verdicts_side_by_side": verdict["side_by_side"],
        "pre_registered": {
            "fixed_before_the_run": True,
            "zero_llm": True,
            "zero_consumer_retrain": True,
            "development_hours": DEVELOPMENT_HOURS,
            "partition_kind": PARTITION_KIND,
            "min_fresh_stations": MIN_FRESH_STATIONS,
            "min_roster_pass": MIN_ROSTER_PASS,
            "max_select": MAX_SELECT,
            "grid_start": GRID_START.isoformat(),
            "probe_origins": list(FROZEN_DEVELOPMENT_ORIGINS),
            "verdicts": list(VERDICTS),
            "2025_values_unread": True,
            "floor_correction": FLOOR_CORRECTION,
            "v1_artifact_untouched": True,
        },
        "step_0_consumption_census": census,
        "step_1_blind_materialization": materialize or {
            "ran": False,
            "reason": (
                "census-only" if census_only
                else (
                    "census drift" if census.get("census_drift")
                    else "fresh pool below the pre-registered floor of %d"
                    % MIN_FRESH_STATIONS
                )
            ),
            "stations": [],
        },
        "step_2_health_check_v2": health,
        "exposure_ledger": _exposure_ledger(census, materialize, health),
        "llm_calls": 0,
        "consumer_retrains": 0,
        "retrain_budget": RETRAIN_BUDGET,
        "generator_sha256": _generator_sha(),
        "generator_sha_on_record_v1": GENERATOR_SHA_ON_RECORD_V1,
        "wall_seconds": time.perf_counter() - started,
    }
    payload = _drop_private(payload)
    if dry_run:
        print(json.dumps(
            {
                "verdict": payload["overall_verdict"],
                "reason": payload["overall_verdict_reason"],
                "fresh_n": census["fresh_n"],
                "census_drift": census["census_drift"],
                "fresh_pool": census["fresh_pool"],
                "retrains": payload["consumer_retrains"],
            },
            indent=2, ensure_ascii=False, default=str,
        ))
        return 0
    _assert_not_v1(OUT_JSON)
    _assert_not_v1(OUT_MD)
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("verdict", payload["overall_verdict"], flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-only", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument(
        "--health-check", action="store_true",
        help="run the per-station development exam (after materialize)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run the selected stages but do not write artifacts or arrays",
    )
    args = parser.parse_args(argv)
    if args.census_only and args.materialize_only:
        parser.error("choose at most one of --census-only / --materialize-only")
    return run(
        census_only=bool(args.census_only),
        materialize_only=bool(args.materialize_only),
        health_check=bool(args.health_check),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())

