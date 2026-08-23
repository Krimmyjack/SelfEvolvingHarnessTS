"""T6 / #42 -- natural single-variate A5 vs A3, frozen plan and sealed boundary.

The question this round answers is narrow and deliberately answerable without
opening anything: can a real comparison between A5 (Source Experience carried
in) and A3 (empty Source Memory) be frozen on natural data *before* any Target
outcome is read, and does the natural Source actually contain all three kinds
of Action-Response the comparison depends on -- a success, a failure, and a
conflict?

Two entry points live here.  ``--plan`` runs Parts A-C and writes the frozen
protocol; it spends no LLM call and no forecasting retrain.  ``--evaluate`` is
implemented and frozen in the same file, and refuses to run until the plan
artifact it reads carries ``evaluate_released: true``, which ``--plan`` writes
as false.  Both entry points in one runner is the sealed-boundary exception
the book grants; the Consumer next door is an instrument, not a second runner.

What is sealed and what is not (recorded in the artifact, not just here):
Source context and Source outcome are both open -- that is the development
surface this round mines for evidence.  Target context is open too: the shape
gate reads it and the Agent will see its public features.  Target *outcome* is
sealed, and the only publicly disclosed fact about it is an aggregate one --
one of the six Target series carries no anomaly -- with no instance identity
attached.  A request for a Target label key under ``--plan`` is not a warning;
it raises, and TARGET_LABEL_WALL_BREACHED outranks every other verdict.

Nothing in the Agent, Memory, Risk, Skill lifecycle or Observation is modified
by this file.  If something turns out to need modification to run, that is a
first-fault to report, not a thing to patch here.

Usage:
  python evaluation/functional/run_e2_t6_natural_a5_a3.py --plan
  python evaluation/functional/run_e2_t6_natural_a5_a3.py --evaluate   # gated
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
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

import numpy as np  # noqa: E402

from run_e2_operational_pipeline import (  # noqa: E402
    FROZEN_SURFACE_V10,
    _freeze,
    _verify,
)

import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import signed_radius as resolver  # noqa: E402
from consumers.ad_scope_adapter import compiled_steps  # noqa: E402

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256  # noqa: E402
from SelfEvolvingHarnessTS.contracts.method import (  # noqa: E402
    PreparationRequest,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    ScopeExecutor,
)
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    MetricSpec,
    anomaly_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    MEASURED_EFFECT_KEY,
    STATUS_EPISODE_ONLY,
    build_episode,
    classify_relation,
    task_consumer_key,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.operators.registry import (  # noqa: E402
    OPERATOR_METADATA,
    OPERATOR_NAMES,
)
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline  # noqa: E402

# =========================================================================== #
# constants -- all frozen by the book
# =========================================================================== #
PROTOCOL_VERSION = "t6_nab_frozen_plan_v1"
EVIDENCE_GRADE = "NATURAL"
EVIDENCE_STANDING = "provisional"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t6_nab_frozen_plan_v1.json"
OUT_MD = E2 / "t6_nab_frozen_plan_v1.md"
# #42a: the v1 artifact is kept as the shape-contract diagnostic and is never
# overwritten; the row-order contract writes its own pair.
PROTOCOL_VERSION_V2 = "t6_nab_frozen_plan_v2"
OUT_JSON_V2 = E2 / "t6_nab_frozen_plan_v2.json"
OUT_MD_V2 = E2 / "t6_nab_frozen_plan_v2.md"

# ---- A1: the pinned upstream ref.  No per-file SHA, Manifest or Receipt. ----
NAB_SOURCE_URL = "https://github.com/numenta/NAB"
NAB_TAG = "v1.1"
NAB_COMMIT = "0dcd73007a349ca0f4128c4c9b18133ce00d9296"
NAB_RAW_BASE = "https://raw.githubusercontent.com/numenta/NAB/%s" % NAB_COMMIT
DATA_ROOT = PROJECT_ROOT / "data" / "benchmark_nab_v1_1" / "raw"
LABELS_REL = "labels/combined_windows.json"

# ---- A2 / A3: cohorts, chosen by filename order and nothing else -----------
SOURCE_COHORTS: dict[str, tuple[str, int]] = {
    # cohort name -> (NAB directory, how many files, lexicographic order)
    "source_aws_cloudwatch": ("realAWSCloudwatch", 8),
    "source_known_cause": ("realKnownCause", 6),
}
TARGET_DIR = "realAdExchange"
TARGET_COHORTS: dict[str, str] = {
    "target_cpc": "cpc",
    "target_cpm": "cpm",
}
TARGET_AGGREGATE_DISCLOSURE = (
    "one of the six realAdExchange series carries no anomaly; which one is "
    "not disclosed and is not read here"
)

# ---- A4: the structural gate, and only these five checks -------------------
MIN_LENGTH = 1000
REQUIRED_COLUMNS = ("timestamp", "value")

# ---- #42a Part A: the v2 input-shape contract ------------------------------
# Sequence order is the physical row order after the CSV header.  A timestamp
# must parse, but strict increase is no longer a legality gate: duplicated,
# backward and irregular stamps are diagnostics, because NAB v1.1 genuinely
# contains them and a gate that rejects the data is a gate about the gate.
# Nothing is sorted, de-duplicated, aggregated, resampled, interpolated or
# dropped -- splits and the 20-point windows are computed on row indices, so
# the row sequence is the only thing that has to be preserved, and it is
# verified preserved element by element.
ROW_ORDER_CONTRACT: dict[str, Any] = {
    "sequence_order": "physical CSV row order after the header",
    "timestamp": "must parse; strict increase is diagnostic, not a gate",
    "forbidden_transforms": ["sort", "deduplicate", "aggregate", "resample",
                             "interpolate", "drop rows"],
    "split_and_window_basis": "row index",
    "value_gate": "univariate, finite, length >= %d" % MIN_LENGTH,
    "applies_to": "all 20 files, with no special case for the four that "
                  "failed the v1 strict-increase check",
}

# ---- A5: the two frozen rounds, as fractions of each series' own length ----
ROUND_FRACTIONS: dict[str, dict[str, tuple[float, float]]] = {
    "r1": {"train": (0.00, 0.40), "support": (0.40, 0.55),
           "delayed": (0.55, 0.70)},
    "r2": {"train": (0.00, 0.70), "support": (0.70, 0.85),
           "delayed": (0.85, 1.00)},
}
ROUNDS: tuple[str, ...] = ("r1", "r2")

# ---- the five-entry menu, frozen since T3 ---------------------------------
PROGRAMS: tuple[str, ...] = (
    "identity", "outlier_iqr", "outlier_mad", "hampel_filter", "winsorize",
)
NON_IDENTITY: tuple[str, ...] = PROGRAMS[1:]

MATERIAL_THRESHOLD = 0.005

# ---- budgets ---------------------------------------------------------------
PLAN_LLM_BUDGET = 0
PLAN_FORECAST_RETRAIN_BUDGET = 0
PLAN_AD_FIT_BUDGET = 200
EVALUATE_LLM_BUDGET = 48
EVALUATE_AD_FIT_BUDGET = 120
EVALUATE_FORECAST_RETRAIN_BUDGET = 0

# ---- N2: the evaluate backend is part of the frozen protocol ---------------
EVALUATE_BACKEND: dict[str, Any] = {
    "model": "gpt-5.6-sol",
    "base_url": "https://api.agicto.cn/v1",
    "temperature": "backend default, not overridden",
    "menu": list(PROGRAMS),
    "note": (
        "pinned here as part of the frozen protocol; --plan makes no call, "
        "and a later --evaluate may not change backend, temperature or menu"
    ),
}

# ---- N4: determinism is rechecked on two cells, not by double-running ------
DETERMINISM_RECHECK_COHORT = "source_known_cause"
DETERMINISM_RECHECK_ROUND = "r1"
DETERMINISM_RECHECK_PROGRAMS = ("identity", "hampel_filter")

# ---- the evaluate protocol's counterbalanced order ------------------------
EVALUATE_ORDER: dict[str, tuple[tuple[str, str], ...]] = {
    "target_cpc": (("A3", "r1"), ("A5", "r1"), ("A3", "r2"), ("A5", "r2")),
    "target_cpm": (("A5", "r1"), ("A3", "r1"), ("A5", "r2"), ("A3", "r2")),
}
EVALUATE_SUPPORT_TRIALS = 2  # non-identity Support trials per round
# window_context wants at least two full 192-point windows behind the origin;
# the deployment Context hint is the only thing this number feeds.
PERIOD_HINT = 24
OUT_EVALUATE = E2 / "t6_nab_evaluate_v2.json"
OUT_SMOKE = E2 / "t6_nab_evaluate_smoke_v1.json"
# the B4 budget-interrupt probe writes its own file so the main smoke
# reading is never overwritten by a deliberately starved run
OUT_SMOKE_BUDGET = E2 / "t6_nab_evaluate_smoke_budget_v1.json"
OUT_SKILL_V2 = E2 / "t6_nab_42d_source_skill_v2.json"
OUT_REPLAY_V2 = E2 / "t6_nab_42e0_skill_v2_replay.json"
OUT_REPLAY_V2_MD = E2 / "t6_nab_42e0_skill_v2_replay.md"
REPLAY_V2_LOCK = E2 / "t6_nab_42e0_skill_v2_replay.lock"
REPLAY_V2_SKILL_ID = "source_investigation_ad_v2"
REPLAY_V2_LLM_BUDGET = 32
REPLAY_V2_AD_FIT_BUDGET = 24
# #42e r1: Source expansion.  New cohorts only; Target keys stay sealed.
OUT_CENSUS_V3 = E2 / "t6_nab_42e_census_v3.json"
OUT_CENSUS_V3_MD = E2 / "t6_nab_42e_census_v3.md"
OUT_SKILL_V3 = E2 / "t6_nab_42e_source_skill_v3.json"
OUT_SKILL_V3_MD = E2 / "t6_nab_42e_source_skill_v3.md"
OUT_EXPANSION_V3 = E2 / "t6_nab_42e_source_expansion_v3.json"
EXPANSION_V3_LOCK = E2 / "t6_nab_42e_source_expansion.lock"
SOURCE_SKILL_ID_V3 = "source_investigation_ad_v3"
EXPANSION_LLM_CAP = 8
EXPANSION_AD_FIT_CAP = 240
OUT_ACCEPT_V3 = E2 / "t6_nab_42e1_behavior_acceptance.json"
OUT_ACCEPT_V3_MD = E2 / "t6_nab_42e1_behavior_acceptance.md"
ACCEPT_V3_LOCK = E2 / "t6_nab_42e1_behavior_acceptance.lock"
H0S_V3_EXPECTED_SHA = (
    "f2054da1d18e2059457ed62282b7f7ff972ae219aedf98b39204ba2009bd7914"
)
ACCEPT_V3_LLM_BUDGET = 32
ACCEPT_V3_AD_FIT_BUDGET = 24
OUT_PATTERN_V1 = E2 / "t6_nab_42e2_pattern_discriminator.json"
OUT_PATTERN_V1_MD = E2 / "t6_nab_42e2_pattern_discriminator.md"
ISOLATED_FRACTION_THRESHOLD = 0.5
OUT_DEPLOY_SMOKE = E2 / "t6_42g_deploy_fast_only_smoke.json"
OUT_L1 = E2 / "t6_42g_l1_static_vs_a3.json"
OUT_L1_MD = E2 / "t6_42g_l1_static_vs_a3.md"
L1_LOCK = E2 / "t6_42g_l1.lock"
YAHOO_FREEZE = E2 / "t6_42f_yahoo_a1_freeze.json"
L1_LLM_CAP = 24
L1_AD_FIT_CAP = 240
L1_ROSTER_N = 24
L1_ROUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "r1": {"train": (0.00, 0.30), "support": (0.30, 0.40),
           "delayed": (0.40, 0.50)},
    "r2": {"train": (0.00, 0.50), "support": (0.50, 0.60),
           "delayed": (0.60, 0.70)},
}
L1_HELDOUT = (0.70, 1.00)
RISK_OPS_V3: tuple[str, ...] = (
    "hampel_filter", "outlier_iqr", "outlier_mad",
)
EXPANSION_COHORTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "source_real_traffic": ("realTraffic", (
        "TravelTime_387.csv", "TravelTime_451.csv",
        "occupancy_6005.csv", "occupancy_t4013.csv",
        "speed_6005.csv", "speed_7578.csv", "speed_t4013.csv",
    )),
    "source_real_tweets": ("realTweets", (
        "Twitter_volume_AAPL.csv", "Twitter_volume_AMZN.csv",
        "Twitter_volume_CRM.csv", "Twitter_volume_CVS.csv",
        "Twitter_volume_FB.csv", "Twitter_volume_GOOG.csv",
        "Twitter_volume_IBM.csv", "Twitter_volume_KO.csv",
        "Twitter_volume_PFE.csv", "Twitter_volume_UPS.csv",
    )),
}
EXPANSION_COHORT_TOKENS: tuple[str, ...] = (
    "realtraffic", "realtweets", "real_traffic", "real_tweets",
    "source_real_traffic", "source_real_tweets",
    "twitter_volume", "traveltime", "occupancy_6005", "occupancy_t4013",
    "speed_6005", "speed_7578", "speed_t4013",
)
TRIGGERABLE_FROM_DEV = ("outlier_mad", "outlier_iqr")
UNTRIGGERABLE_FROM_DEV = ("hampel_filter",)
# Part B: the smoke stands Source cohorts in the Target cell positions.  The
# frozen order, arms and rounds are untouched; only where a cell's rows come
# from is parameterized, and only the first two series of each stand-in
# cohort are used so the mechanical pass fits the smoke fit budget.
SMOKE_COHORT_SOURCE: dict[str, str] = {
    "target_cpc": "source_aws_cloudwatch",
    "target_cpm": "source_known_cause",
}
SMOKE_SERIES_PER_COHORT = 2
SMOKE_AD_FIT_BUDGET = 60
SMOKE_OPERATORS: tuple[str, ...] = ("outlier_iqr", "winsorize")
# #42c: one mechanical lifecycle fixture on an already-exposed Source
# cell.  Not a new runner, not method evidence, not a Target readout.
OUT_LIFECYCLE_FIXTURE = E2 / "t6_nab_lifecycle_fixture_v1.json"
OUT_42C_NOTE = E2 / "t6_nab_42c_correction_note.md"
FIXTURE_AD_FIT_BUDGET = 20
FIXTURE_LLM_BUDGET = 0
FIXTURE_OPERATORS: tuple[str, ...] = ("winsorize",)
FIXTURE_SOURCE_COHORT = "source_aws_cloudwatch"
FIXTURE_ROUND = "r2"
FIXTURE_ARM = "A3"
# order_override still keys the frozen evaluate slots; the rows behind
# the slot are Source.  The artifact names the stand-in explicitly.
FIXTURE_CELL_SLOT = "target_cpc"
FIXTURE_SUPPORT_TRIAL_BUDGET = 1


class Stop(Exception):
    """A pre-registered first-fault.  Carries the verdict it maps to."""

    def __init__(self, verdict: str, reason: str, **extra: Any) -> None:
        super().__init__("%s: %s" % (verdict, reason))
        self.verdict = verdict
        self.reason = reason
        self.extra = dict(extra)


class TargetLabelWallBreached(Stop):
    def __init__(self, key: str) -> None:
        super().__init__(
            "TARGET_LABEL_WALL_BREACHED",
            "a Target label key was requested while the Target outcome is "
            "sealed: %r" % key,
            key=key,
        )


# =========================================================================== #
# small utilities
# =========================================================================== #
def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_text(doc: Mapping[str, Any]) -> str:
    return json.dumps(_plain(doc), indent=2, ensure_ascii=False) + "\n"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(PROJECT_ROOT),
                          capture_output=True, text=True).stdout.strip()


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "<outside-repo>/%s" % resolved.name


# =========================================================================== #
# Part A -- the natural data, the shape gate, and the sealed boundary
# =========================================================================== #
def _data_reference() -> dict[str, Any]:
    """A1: where the bytes came from, and nothing more.

    No per-file SHA, no Manifest, no Receipt -- the book asked for a pinned
    upstream ref and a URL, and adding a second identity layer would be a new
    platform this round is not authorized to build.
    """
    return {
        "source_url": NAB_SOURCE_URL,
        "tag": NAB_TAG,
        "upstream_commit": NAB_COMMIT,
        "raw_base": NAB_RAW_BASE,
        "local_root": _repo_rel(DATA_ROOT),
        "tracked_in_git": False,
        "note": (
            "the raw bytes stay in an untracked directory and are never "
            "committed; re-fetching from the pinned commit reproduces them"
        ),
    }


def _read_series(path: Path,
                 *, row_order_contract: bool = False) -> dict[str, Any]:
    """The structural checks.

    ``row_order_contract`` selects #42a's v2 contract: the timestamp still
    has to parse, but a non-monotonic timestamp column stops being a
    legality failure and becomes a diagnostic.  Everything else -- two named
    columns, univariate, finite values, length -- is unchanged, and the same
    contract is applied to all twenty files with no per-file exception.
    """
    failures: list[str] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "failures": ["unreadable:%s" % type(exc).__name__]}
    if not rows:
        return {"ok": False, "failures": ["empty_file"]}
    header = [c.strip() for c in rows[0]]
    if tuple(header[:2]) != REQUIRED_COLUMNS or len(header) != 2:
        failures.append("columns=%r" % (header,))
    stamps: list[str] = []
    values: list[float] = []
    bad_value = 0
    for row in rows[1:]:
        if len(row) != 2:
            failures.append("ragged_row")
            break
        stamps.append(row[0])
        try:
            number = float(row[1])
        except (TypeError, ValueError):
            bad_value += 1
            continue
        if not np.isfinite(number):
            bad_value += 1
            continue
        values.append(number)
    if bad_value:
        failures.append("non_finite_or_unparsable_values=%d" % bad_value)
    if len(values) != len(stamps):
        failures.append("value_count_ne_timestamp_count")
    parsed: list[datetime] = []
    for stamp in stamps:
        try:
            parsed.append(datetime.fromisoformat(stamp.strip()))
        except ValueError:
            failures.append("unparsable_timestamp")
            break
    violations: list[dict[str, Any]] = []
    if len(parsed) == len(stamps) and len(parsed) > 1:
        for position in range(1, len(parsed)):
            if parsed[position] > parsed[position - 1]:
                continue
            violations.append({
                "row_index": position,
                "previous_timestamp": stamps[position - 1],
                "timestamp": stamps[position],
                "kind": ("duplicate" if parsed[position] == parsed[position - 1]
                         else "backwards"),
            })
        if violations and not row_order_contract:
            failures.append("timestamps_not_strictly_increasing")
    if len(values) < MIN_LENGTH:
        failures.append("length=%d<%d" % (len(values), MIN_LENGTH))
    array = np.asarray(values, dtype=np.float64)
    duplicates = [v for v in violations if v["kind"] == "duplicate"]
    backwards = [v for v in violations if v["kind"] == "backwards"]
    worst_backward = None
    if backwards and len(parsed) == len(stamps):
        worst_backward = max(
            (parsed[v["row_index"] - 1] - parsed[v["row_index"]]).total_seconds()
            for v in backwards)
    # row-order preservation, checked rather than asserted: the physical rows
    # read out of the file and the rows handed downstream are the same rows,
    # in the same order, with the same values.
    physical_rows_before = max(len(rows) - 1, 0)
    return {
        "ok": not failures,
        "failures": failures,
        # the offending rows themselves: a shape fault that only reports its
        # own name cannot be adjudicated by anyone downstream
        "ordering_violations": violations[:10],
        "ordering_violation_count": len(violations),
        "duplicate_timestamp_count": len(duplicates),
        "backward_transition_count": len(backwards),
        "max_backward_delta_seconds": worst_backward,
        "physical_rows_before": physical_rows_before,
        "physical_rows_after": int(array.size),
        "rows_preserved": physical_rows_before == int(array.size),
        "values_sha256": hashlib.sha256(
            array.tobytes(order="C")).hexdigest(),
        "length": len(values),
        "values": array,
        "timestamps": stamps,
        "parsed_timestamps": parsed if len(parsed) == len(stamps) else [],
        "univariate": len(header) == 2,
    }


def _cohort_files(directory: str, take: int | None) -> list[Path]:
    """Filename order, full stop.  Never re-ordered by label, Consumer result
    or Program headroom -- that is the whole point of fixing it here."""
    paths = sorted((DATA_ROOT / directory).glob("*.csv"), key=lambda p: p.name)
    return paths if take is None else paths[:take]


def _window_plan(length: int) -> dict[str, Any]:
    """A5: each series gets its two rounds from its own length."""
    plan: dict[str, Any] = {}
    for name, spans in ROUND_FRACTIONS.items():
        row: dict[str, Any] = {}
        for part, (lo, hi) in spans.items():
            row[part] = [int(round(lo * length)), int(round(hi * length))]
        plan[name] = row
    return plan


def _target_cohort_of(filename: str) -> str | None:
    for name, metric in TARGET_COHORTS.items():
        if ("_%s_" % metric) in filename:
            return name
    return None


def gate_all(*, row_order_contract: bool = False) -> dict[str, Any]:
    """A4 over every file this round touches, with nothing raised yet.

    The gate result is a table first and a decision second: a run that stops
    at the first failing file cannot tell the main line whether the rest of
    the surface is sound, and that is exactly what a shape fault needs in
    order to be adjudicated.
    """
    rows: list[dict[str, Any]] = []
    reads: dict[str, Any] = {}
    for cohort, (directory, take) in SOURCE_COHORTS.items():
        for path in _cohort_files(directory, take):
            read = _read_series(
                path, row_order_contract=row_order_contract)
            reads[path.name] = read
            rows.append({
                "role": "source", "cohort": cohort, "file": path.name,
                "nab_key": "%s/%s" % (directory, path.name),
                "ok": read["ok"], "failures": read["failures"],
                "length": read.get("length"),
                "ordering_violations": read.get("ordering_violations"),
                "ordering_violation_count": read.get(
                    "ordering_violation_count"),
                "duplicate_timestamp_count": read.get(
                    "duplicate_timestamp_count"),
                "backward_transition_count": read.get(
                    "backward_transition_count"),
                "max_backward_delta_seconds": read.get(
                    "max_backward_delta_seconds"),
                "physical_rows_before": read.get("physical_rows_before"),
                "physical_rows_after": read.get("physical_rows_after"),
                "rows_preserved": read.get("rows_preserved"),
                "values_sha256": read.get("values_sha256"),
            })
    for path in _cohort_files(TARGET_DIR, None):
        read = _read_series(
            path, row_order_contract=row_order_contract)
        reads[path.name] = read
        rows.append({
            "role": "target", "cohort": _target_cohort_of(path.name),
            "file": path.name,
            "nab_key": "%s/%s" % (TARGET_DIR, path.name),
            "ok": read["ok"], "failures": read["failures"],
            "length": read.get("length"),
            "ordering_violations": read.get("ordering_violations"),
            "ordering_violation_count": read.get("ordering_violation_count"),
            "duplicate_timestamp_count": read.get("duplicate_timestamp_count"),
            "backward_transition_count": read.get("backward_transition_count"),
            "max_backward_delta_seconds": read.get(
                "max_backward_delta_seconds"),
            "physical_rows_before": read.get("physical_rows_before"),
            "physical_rows_after": read.get("physical_rows_after"),
            "rows_preserved": read.get("rows_preserved"),
            "values_sha256": read.get("values_sha256"),
        })
    return {
        "rows": rows,
        "reads": reads,
        "checks_applied": [
            "two columns named timestamp/value", "univariate",
            "timestamps strictly increasing",
            "value parses to a finite number",
            "length >= %d" % MIN_LENGTH,
        ],
        "source_failures": [r for r in rows
                            if r["role"] == "source" and not r["ok"]],
        "target_failures": [r for r in rows
                            if r["role"] == "target" and not r["ok"]],
        "all_ok": all(r["ok"] for r in rows),
        "row_order_contract": bool(row_order_contract),
        "rows_preserved_everywhere": all(
            r.get("rows_preserved") for r in rows),
    }


def _load_universe(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Build the arrays, once the gate has spoken."""
    if gate.get("row_order_contract"):
        # Under the v2 contract the only way to fail is a timestamp that will
        # not parse, a non-finite value, a short series, or a row sequence
        # that did not survive the read.
        broken = [r for r in gate["rows"] if not r["ok"]
                  or not r.get("rows_preserved")]
        if broken:
            raise Stop(
                "NATURAL_ROW_SEQUENCE_INELIGIBLE",
                "%d file(s) failed the row-order contract: %s"
                % (len(broken),
                   [(r["file"], r["failures"] or ["row_sequence_not_preserved"])
                    for r in broken]),
                failures=broken)
    elif gate["target_failures"]:
        raise Stop(
            "NATURAL_DATA_SHAPE_INELIGIBLE",
            "%d of the six Target files fail the structural gate: %s.  No "
            "substitute is drawn -- a replacement chosen after a failure "
            "would be one chosen with knowledge the gate is not allowed to "
            "have, and the book fixes all six realAdExchange series with no "
            "replacement."
            % (len(gate["target_failures"]),
               [(r["file"], r["failures"]) for r in gate["target_failures"]]),
            target_failures=gate["target_failures"],
            source_failures=gate["source_failures"])
    if gate["source_failures"]:
        raise Stop(
            "NATURAL_DATA_SHAPE_INELIGIBLE",
            "%d Source files fail the structural gate: %s"
            % (len(gate["source_failures"]),
               [(r["file"], r["failures"])
                for r in gate["source_failures"]]),
            source_failures=gate["source_failures"])

    universe: dict[str, Any] = {"source": {}, "target": {},
                                "gate": gate["rows"]}
    reads = gate["reads"]
    for cohort, (directory, take) in SOURCE_COHORTS.items():
        universe["source"][cohort] = {
            path.name: {
                "values": reads[path.name]["values"],
                "timestamps": reads[path.name]["timestamps"],
                "length": reads[path.name]["length"],
                "nab_key": "%s/%s" % (directory, path.name),
                "windows": _window_plan(reads[path.name]["length"]),
            }
            for path in _cohort_files(directory, take)
        }
    for cohort in TARGET_COHORTS:
        universe["target"][cohort] = {}
    for path in _cohort_files(TARGET_DIR, None):
        cohort = _target_cohort_of(path.name)
        if cohort is None:
            raise Stop("NATURAL_DATA_SHAPE_INELIGIBLE",
                       "target file %s carries neither cpc nor cpm in its "
                       "name" % path.name)
        read = reads[path.name]
        universe["target"][cohort][path.name] = {
            "values": read["values"], "timestamps": read["timestamps"],
            "length": read["length"],
            "nab_key": "%s/%s" % (TARGET_DIR, path.name),
            "windows": _window_plan(read["length"]),
        }
    for cohort, rows in universe["target"].items():
        if len(rows) != 3:
            raise Stop("NATURAL_DATA_SHAPE_INELIGIBLE",
                       "%s must hold exactly three series, holds %d"
                       % (cohort, len(rows)))
    return universe


class LabelWall:
    """The official labels, with the Target half behind a wall.

    ``released`` is what ``--evaluate`` sets once the main line has confirmed
    the Target outcome is still sealed and has released the frozen path.  In
    ``--plan`` it is False and a Target key raises rather than returning an
    empty list -- an empty list is itself an answer about that series.
    """

    def __init__(self, *, released: bool) -> None:
        self.released = bool(released)
        self.requests: list[dict[str, Any]] = []
        raw = json.loads(
            (DATA_ROOT / LABELS_REL).read_text(encoding="utf-8"))
        self.target_keys = frozenset(
            str(k) for k in raw if str(k).startswith(TARGET_DIR + "/"))
        # Under the wall the Target entries are not merely refused on request:
        # their values are never retained, so no code path and no traceback
        # can reach them.  Key *names* are retained, and disclose nothing --
        # all six Target files carry an entry, so presence is uninformative;
        # it is the window list that would name the clean series, and that is
        # what is dropped here.
        self._open = {str(k): v for k, v in raw.items()
                      if self.released or str(k) not in self.target_keys}
        self.target_values_retained = bool(self.released)

    def windows(self, key: str, timestamps: Sequence[str]
                ) -> dict[str, Any]:
        """Official windows, resolved to the rows they actually name.

        #42a Part B: the truth stays in timestamp semantics, and each row is
        tested independently for membership in each official window.  Two
        rows sharing a timestamp that falls inside a window both belong to
        that same truth event -- which is the case the v1 span mapping could
        not express.  A window that names no sampled row is counted and
        reported; silently losing it would quietly inflate precision.
        """
        key = str(key)
        if key in self.target_keys and not self.released:
            self.requests.append({"key": key, "granted": False})
            raise TargetLabelWallBreached(key)
        self.requests.append({"key": key, "granted": True})
        spans = self._open.get(key) or []
        parsed = [datetime.fromisoformat(str(s).strip()) for s in timestamps]
        events: list[list[int]] = []
        unmapped: list[list[str]] = []
        for span in spans:
            start = datetime.fromisoformat(str(span[0]).strip())
            end = datetime.fromisoformat(str(span[1]).strip())
            rows = [i for i, t in enumerate(parsed) if start <= t <= end]
            if rows:
                events.append(rows)
            else:
                unmapped.append([str(span[0]), str(span[1])])
        return {
            "events": events,
            "official_window_count": len(spans),
            "mapped_window_count": len(events),
            "unmapped_windows": unmapped,
            "unmapped_window_count": len(unmapped),
            "labelled_rows": sum(len(rows) for rows in events),
        }

    def audit(self) -> dict[str, Any]:
        return {
            "released": self.released,
            "target_values_retained_in_memory": self.target_values_retained,
            "key_presence_discloses_nothing": (
                "all six Target files carry an entry in combined_windows.json, "
                "so the presence of a key says nothing about which series is "
                "the clean one; only the window list would, and it is dropped"
            ),
            "target_keys_walled": sorted(self.target_keys),
            "target_key_requests": [r for r in self.requests
                                    if r["key"] in self.target_keys],
            "breached": any(not r["granted"] for r in self.requests),
        }


# =========================================================================== #
# Part B -- the Consumer, and the program application
# =========================================================================== #
def _load_consumer() -> Any:
    try:
        from consumers import aegists_iforest_v1 as consumer
    except Exception as exc:  # noqa: BLE001
        raise Stop("CONSUMER_DEPENDENCY_UNAVAILABLE",
                   "the frozen AD Consumer could not be imported: %s: %s"
                   % (type(exc).__name__, exc)) from exc
    return consumer


def _canonical_menu_program(name: str) -> str:
    """Map a Fast candidate id / winner op onto the frozen five-entry menu."""
    raw = str(name or "").strip()
    if not raw or raw == "identity":
        return "identity"
    if raw in PROGRAMS:
        return raw
    lowered = raw.lower()
    # longest menu name first so outlier_mad wins over a bare 'mad' token
    for menu in sorted(PROGRAMS, key=len, reverse=True):
        if menu == "identity":
            continue
        if menu in lowered:
            return menu
    aliases = (
        ("hampel", "hampel_filter"),
        ("winsor", "winsorize"),
        ("iqr", "outlier_iqr"),
        ("mad", "outlier_mad"),
    )
    for token, menu in aliases:
        if token in lowered:
            return menu
    return "identity"


def _program_steps(program: str) -> tuple[tuple[str, dict], ...]:
    menu = _canonical_menu_program(program)
    if menu == "identity":
        return ()
    if menu not in OPERATOR_METADATA:
        raise Stop("NATURAL_DATA_SHAPE_INELIGIBLE",
                   "menu entry %r is not a registered operator" % program)
    return ((menu, {}),)


def _apply_program(block: Any, program: str) -> np.ndarray:
    """The program acts on the training block, once, and nowhere else."""
    array = np.asarray(block, dtype=np.float64).ravel()
    steps = _program_steps(program)
    if not steps:
        return array.copy()
    result = run_pipeline(list(steps), array)
    if not result.ok or result.artifact is None:
        raise Stop("NATURAL_DATA_SHAPE_INELIGIBLE",
                   "program %s failed on a training block: %s"
                   % (program, result.error))
    out = np.asarray(result.artifact, dtype=np.float64).ravel()
    if out.size != array.size:
        raise Stop("NATURAL_DATA_SHAPE_INELIGIBLE",
                   "program %s changed the block length" % program)
    return out


class FitBudget:
    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.used = 0

    def spend(self, n: int = 1) -> None:
        if self.used + n > self.cap:
            raise Stop("CONSUMER_FIT_BUDGET_EXCEEDED",
                       "AD Consumer fit budget exhausted at %d" % self.cap)
        self.used += n


def _cell_reading(
    *,
    consumer: Any,
    series: Mapping[str, Any],
    round_name: str,
    program: str,
    wall: LabelWall,
    budget: FitBudget,
) -> dict[str, Any]:
    """One series, one round, one program: fit once, read both windows."""
    windows = series["windows"][round_name]
    raw = np.asarray(series["values"], dtype=np.float64)
    train_lo, train_hi = windows["train"]
    prepared = _apply_program(raw[train_lo:train_hi], program)
    budget.spend(1)
    model = consumer.fit_series(prepared)
    labels = wall.windows(series["nab_key"], series["timestamps"])
    out: dict[str, Any] = {"program": program, "round": round_name,
                           "train_span": [train_lo, train_hi],
                           "label_mapping": {
                               k: v for k, v in labels.items()
                               if k != "events"}}
    for part in ("support", "delayed"):
        lo, hi = windows[part]
        out[part] = consumer.score_series(
            model, raw, (lo, hi), labels["events"])
    out["changed_points"] = int(np.count_nonzero(
        ~np.isclose(prepared, raw[train_lo:train_hi], equal_nan=True)))
    return out


# =========================================================================== #
# Part C -- the natural Source Experience bank
# =========================================================================== #
def _source_task_spec() -> Any:
    return anomaly_task_spec_v1(
        downstream_model_class="aegists_iforest_v1",
        metric=MetricSpec("macro_event_f1", "higher_is_better"),
        forbidden_modifications=tuple(
            sorted(n for n in OPERATOR_NAMES if n not in set(NON_IDENTITY))),
    )


def _context_summary(
    *, cohort: str, round_name: str, program: str,
    series_rows: Mapping[str, Any], spans: Mapping[str, Any],
) -> dict[str, Any]:
    """Deployment-visible Context only.

    The features come out of the live Observation path (extract_public_features
    at task_kind=anomaly_detection); nothing here names a dataset, a file or a
    series, and build_episode's private-field check enforces that
    independently.
    """
    first = sorted(series_rows)[0]
    row = series_rows[first]
    train_lo, train_hi = row["windows"][round_name]["train"]
    block = np.asarray(row["values"], dtype=np.float64)[train_lo:train_hi]
    features = dict(extract_public_features(
        block, task_kind="anomaly_detection"))
    return {
        "cohort": {"series_count": len(series_rows),
                   "evaluation_series_count": len(series_rows)},
        "local_pattern": {str(k): v for k, v in features.items()
                          if isinstance(v, (int, float, bool))},
        "delayed_pattern": {},
        "program_geometry": {
            "scope": "training_rows",
            "program_steps": [{"op": op, "params": dict(params)}
                              for op, params in _program_steps(program)],
            "acts_on": "training block only, once",
            "train_fraction": list(ROUND_FRACTIONS[round_name]["train"]),
            "round": round_name,
        },
        "window_plan": _plain(spans),
        "series_uids": sorted(series_rows),
    }


def _build_source_bank(
    *, consumer: Any, universe: Mapping[str, Any], wall: LabelWall,
    budget: FitBudget,
) -> dict[str, Any]:
    """Enumerate the menu over both Source cohorts and both rounds."""
    spec = _source_task_spec()
    key = task_consumer_key(spec)
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    per_series: dict[tuple[str, str, str], dict[str, Any]] = {}
    label_mapping: dict[str, Any] = {}

    for cohort, rows in universe["source"].items():
        for round_name in ROUNDS:
            for program in PROGRAMS:
                readings: dict[str, Any] = {}
                for name in sorted(rows):
                    readings[name] = _cell_reading(
                        consumer=consumer, series=rows[name],
                        round_name=round_name, program=program,
                        wall=wall, budget=budget)
                    # identical for every program and round of a series; kept
                    # once so an unmapped official window cannot go unnoticed
                    label_mapping.setdefault(
                        name, readings[name]["label_mapping"])
                per_series[(cohort, round_name, program)] = readings
                cells[(cohort, round_name, program)] = {
                    "support_macro_f1": consumer.macro_f1(
                        {k: v["support"] for k, v in readings.items()}),
                    "support_pooled_f1": consumer.pooled_f1(
                        {k: v["support"] for k, v in readings.items()}),
                    "delayed_macro_f1": consumer.macro_f1(
                        {k: v["delayed"] for k, v in readings.items()}),
                    "delayed_pooled_f1": consumer.pooled_f1(
                        {k: v["delayed"] for k, v in readings.items()}),
                    "support_f1_by_series": {
                        k: v["support"]["f1"] for k, v in readings.items()},
                    "delayed_f1_by_series": {
                        k: v["delayed"]["f1"] for k, v in readings.items()},
                    "support_auprc_by_series": {
                        k: v["support"]["auprc"] for k, v in readings.items()},
                    "delayed_auprc_by_series": {
                        k: v["delayed"]["auprc"] for k, v in readings.items()},
                    "support_background_alarm_rate": {
                        k: v["support"]["background_alarm_rate"]
                        for k, v in readings.items()},
                    "delayed_background_alarm_rate": {
                        k: v["delayed"]["background_alarm_rate"]
                        for k, v in readings.items()},
                    "changed_points_by_series": {
                        k: v["changed_points"] for k, v in readings.items()},
                }

    # ---- gains against the identity baseline of the same cohort+round ----
    episodes: list[Any] = []
    rows_out: list[dict[str, Any]] = []
    for cohort, series_rows in universe["source"].items():
        for round_name in ROUNDS:
            base = cells[(cohort, round_name, "identity")]
            for program in PROGRAMS:
                cell = cells[(cohort, round_name, program)]
                support_facts = classify_relation(
                    aggregate_gain=_delta(cell["support_macro_f1"],
                                          base["support_macro_f1"]),
                    per_series_gains=_delta_map(
                        cell["support_f1_by_series"],
                        base["support_f1_by_series"]),
                    is_identity=(program == "identity"),
                    consumer_id=consumer.CONSUMER_ID,
                )
                delayed_facts = classify_relation(
                    aggregate_gain=_delta(cell["delayed_macro_f1"],
                                          base["delayed_macro_f1"]),
                    per_series_gains=_delta_map(
                        cell["delayed_f1_by_series"],
                        base["delayed_f1_by_series"]),
                    is_identity=(program == "identity"),
                    consumer_id=consumer.CONSUMER_ID,
                )
                spans = {name: series_rows[name]["windows"][round_name]
                         for name in sorted(series_rows)}
                episode = build_episode(
                    episode_id="t6_%s_%s_%s" % (cohort, round_name, program),
                    task_consumer_key=key,
                    domain_namespace=cohort,
                    context_summary=_context_summary(
                        cohort=cohort, round_name=round_name, program=program,
                        series_rows=series_rows, spans=spans),
                    workflow_signature=program,
                    support_response={
                        "gain": support_facts["aggregate_gain"],
                        "accepted": support_facts["relation"] == "POSITIVE",
                        "macro_event_f1": cell["support_macro_f1"],
                        "pooled_event_f1": cell["support_pooled_f1"],
                        "per_series_gain": _delta_map(
                            cell["support_f1_by_series"],
                            base["support_f1_by_series"]),
                        MEASURED_EFFECT_KEY: dict(support_facts),
                    },
                    delayed_response={
                        "evaluated": True,
                        "gain": delayed_facts["aggregate_gain"],
                        "macro_event_f1": cell["delayed_macro_f1"],
                        "pooled_event_f1": cell["delayed_pooled_f1"],
                        "per_series_gain": _delta_map(
                            cell["delayed_f1_by_series"],
                            base["delayed_f1_by_series"]),
                        MEASURED_EFFECT_KEY: dict(delayed_facts),
                    },
                    relation=str(delayed_facts["relation"]),
                    evidence_level="DELAYED",
                    # No Target-local Skill, no Shared Capability: this bank
                    # is evidence, and evidence carries no execution rights.
                    local_status=STATUS_EPISODE_ONLY,
                    evidence_refs=["t6_source_bank"],
                )
                episodes.append(episode)
                rows_out.append({
                    "episode_id": episode.episode_id,
                    "cohort": cohort,
                    "round": round_name,
                    "program": program,
                    "task_consumer_key": key,
                    "support_relation": support_facts["relation"],
                    "delayed_relation": delayed_facts["relation"],
                    "support_aggregate_gain": support_facts["aggregate_gain"],
                    "delayed_aggregate_gain": delayed_facts["aggregate_gain"],
                    "support_harmed": support_facts["harmed_series_count"],
                    "delayed_harmed": delayed_facts["harmed_series_count"],
                    "support_worst_series": support_facts["min_per_series_gain"],
                    "delayed_worst_series": delayed_facts["min_per_series_gain"],
                    "series_read": delayed_facts["series_read"],
                    "support_macro_f1": cell["support_macro_f1"],
                    "delayed_macro_f1": cell["delayed_macro_f1"],
                    "classification_basis": delayed_facts["classification_basis"],
                    "changed_points_by_series": cell["changed_points_by_series"],
                })
    return {"task_consumer_key": key, "task_spec": spec.to_dict(),
            "episodes": episodes, "rows": rows_out,
            "label_mapping": label_mapping,
            "cells": {"|".join(k): v for k, v in cells.items()}}


def _delta(candidate: Any, base: Any) -> float | None:
    if candidate is None or base is None:
        return None
    return float(candidate) - float(base)


def _delta_map(candidate: Mapping[str, Any],
               base: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in candidate.items():
        other = base.get(name)
        if value is None or other is None:
            continue
        out[str(name)] = float(value) - float(other)
    return out


def _write_bank_through_runtime(episodes: Sequence[Any]) -> dict[str, Any]:
    """C2: the bank reaches Memory through the normal path, not a side door."""
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod

    snapshot = compile_snapshot(
        PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
        verify_lock=False)
    method = TTHAMethod(_NullFastAgent(), snapshot, ())
    for episode in episodes:
        method.append_experience_episode(episode)
    held = list(method.experience_episodes)
    return {
        "written": len(episodes),
        "read_back_from_runtime": len(held),
        "identical": [e.episode_id for e in held] == [
            e.episode_id for e in episodes],
        "all_episode_only": all(
            e.local_status == STATUS_EPISODE_ONLY for e in held),
        "no_skill_formed": True,
        "no_shared_capability": True,
        "to_dict": [e.to_dict() for e in held],
    }


class _NullFastAgent:
    """A Method needs an agent to exist; the bank never calls one."""

    def prepare(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("the Source bank never runs the Agent")


def _readiness(rows: Sequence[Mapping[str, Any]],
               cells: Mapping[str, Any]) -> dict[str, Any]:
    """C3, as #42a settles it: the three-kind gate reads delayed only.

    The Support layer is kept and reported, because a Support-positive that
    turns delayed-negative is exactly the flip this line wants to watch.  But
    it may not vote: letting a cell contribute its Support relation to one
    slot and its delayed relation to another would let a single cell fill two
    of the three kinds, and the gate would stop meaning what it says.
    """
    def by_layer(layer: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for row in rows:
            found.setdefault(row["%s_relation" % layer], []).append(
                row["episode_id"])
        return found

    support, delayed = by_layer("support"), by_layer("delayed")
    identity_finite = all(
        cells[k]["support_macro_f1"] is not None
        and cells[k]["delayed_macro_f1"] is not None
        and np.isfinite(cells[k]["support_macro_f1"])
        and np.isfinite(cells[k]["delayed_macro_f1"])
        for k in cells if k.endswith("|identity"))

    positive = sorted(i for i in delayed.get("POSITIVE", [])
                      if not i.endswith("_identity"))
    negative = sorted(delayed.get("NEGATIVE", []))
    conflict = sorted(delayed.get("CONFLICT", []))

    flips = [
        {"episode_id": row["episode_id"],
         "support_relation": row["support_relation"],
         "delayed_relation": row["delayed_relation"]}
        for row in rows
        if row["support_relation"] != row["delayed_relation"]
    ]

    checks = [
        {"id": "C3_identity_consumer_finite",
         "ok": bool(identity_finite),
         "detail": "every identity cell produced a finite macro F1"},
        {"id": "C3_delayed_non_identity_positive",
         "ok": bool(positive),
         "detail": "delayed POSITIVE cells: %s" % (positive or "none")},
        {"id": "C3_delayed_negative",
         "ok": bool(negative),
         "detail": "delayed NEGATIVE cells: %s" % (negative or "none")},
        {"id": "C3_delayed_conflict",
         "ok": bool(conflict),
         "detail": ("delayed CONFLICT cells (aggregate improved, at least "
                    "one series past the harm line): %s"
                    % (conflict or "none"))},
    ]
    verdict = None
    if not identity_finite:
        verdict = "NATURAL_AD_CONSUMER_UNREADABLE"
    elif not positive:
        verdict = "NO_NATURAL_PROGRAM_HEADROOM"
    elif not negative or not conflict:
        verdict = "SOURCE_EVIDENCE_DIVERSITY_INSUFFICIENT"
    return {
        "gate_layer": "delayed_relation only (#42a Part C)",
        "checks": checks,
        "all_passed": all(c["ok"] for c in checks),
        "first_fault": verdict,
        "support_layer": {k: v for k, v in support.items() if v},
        "delayed_layer": {k: v for k, v in delayed.items() if v},
        "support_to_delayed_flips": flips,
        "positive_cells": positive,
        "negative_cells": negative,
        "conflict_cells": conflict,
        "note": (
            "Support relations are reported for the flip they expose, and "
            "are excluded from the gate so no single cell can fill two of "
            "the three required kinds"),
    }


def _determinism_recheck(
    *, consumer: Any, universe: Mapping[str, Any], wall: LabelWall,
    budget: FitBudget, bank: Mapping[str, Any],
) -> dict[str, Any]:
    """N4: two cells re-run, not a full second pass."""
    rows = universe["source"][DETERMINISM_RECHECK_COHORT]
    out: list[dict[str, Any]] = []
    for program in DETERMINISM_RECHECK_PROGRAMS:
        key = "|".join((DETERMINISM_RECHECK_COHORT,
                        DETERMINISM_RECHECK_ROUND, program))
        first = bank["cells"][key]
        again: dict[str, Any] = {}
        for name in sorted(rows):
            again[name] = _cell_reading(
                consumer=consumer, series=rows[name],
                round_name=DETERMINISM_RECHECK_ROUND, program=program,
                wall=wall, budget=budget)
        second_support = consumer.macro_f1(
            {k: v["support"] for k, v in again.items()})
        second_delayed = consumer.macro_f1(
            {k: v["delayed"] for k, v in again.items()})
        out.append({
            "cell": key,
            "series_refit": len(again),
            "support_macro_f1_first": first["support_macro_f1"],
            "support_macro_f1_again": second_support,
            "delayed_macro_f1_first": first["delayed_macro_f1"],
            "delayed_macro_f1_again": second_delayed,
            "identical": (first["support_macro_f1"] == second_support
                          and first["delayed_macro_f1"] == second_delayed),
        })
    return {
        "cohort": DETERMINISM_RECHECK_COHORT,
        "round": DETERMINISM_RECHECK_ROUND,
        "programs": list(DETERMINISM_RECHECK_PROGRAMS),
        "cells": out,
        "all_identical": all(c["identical"] for c in out),
        "note": (
            "capped by the book at two cells: a full second pass would "
            "double the enumeration and blow the fit budget for no extra "
            "information about a deterministic estimator"),
    }


# =========================================================================== #
# Part D -- the frozen evaluate protocol
# =========================================================================== #
def _frozen_protocol(bank: Mapping[str, Any],
                     universe: Mapping[str, Any]) -> dict[str, Any]:
    """Everything a later --evaluate is allowed to do, written down now."""
    return {
        "runner": _repo_rel(Path(__file__)),
        "entry": "--evaluate",
        "released": False,
        "release_rule": (
            "--evaluate refuses to run until this artifact carries "
            "evaluate_released: true, which only the main line sets after "
            "confirming the Target outcome is still sealed"
        ),
        "arms": {
            "A3": "h0 constant + empty Source Experience",
            "A5": "the same h0 + the natural Source Episodes frozen below",
        },
        "identical_across_arms": [
            "TaskSpec", "Consumer", "program menu", "public Context",
            "backend", "Target feedback cap",
        ],
        "per_arm_isolation": (
            "each Target cohort x arm gets a brand-new Method and a brand-new "
            "Store; no Target Experience crosses between target_cpc and "
            "target_cpm"
        ),
        "support_trials_per_round": EVALUATE_SUPPORT_TRIALS,
        "agent_autonomy": (
            "the Agent proposes; the runner names no Workflow, injects no "
            "answer key, and does not re-draw"
        ),
        "slow": "never called",
        "cross_domain_retrieval_rule": (
            "A5's retrieval may not use the dataset name as a similarity "
            "signal: the Episodes carry cohort only in domain_namespace, and "
            "build_episode's private-field check already refuses dataset_id, "
            "filename and series_uid inside context_summary"
        ),
        "lifecycle": (
            "an Episode is written after every Support trial; the delayed "
            "window decides LOCAL_ACTIVE or RESTRICTED"
        ),
        "order_counterbalanced": {
            cohort: ["%s-%s" % (arm, rnd) for arm, rnd in seq]
            for cohort, seq in EVALUATE_ORDER.items()
        },
        "budgets": {
            "llm": EVALUATE_LLM_BUDGET,
            "ad_consumer_fits": EVALUATE_AD_FIT_BUDGET,
            "forecasting_retrains": EVALUATE_FORECAST_RETRAIN_BUDGET,
        },
        "backend": dict(EVALUATE_BACKEND),
        "primary_readings": [
            "non-identity Target Support trials spent before the first "
            "LOCAL_ACTIVE; recorded as >4 when none is reached",
            "the round in which the first activation happened",
            "final delayed macro F1 gain",
            "harmed Support receipt count",
            "worst per-series delayed gain",
            "abstention count",
            "retrieved Source card ids, and whether they moved the proposal",
            "Target feedback actually consumed by each arm",
        ],
        "verdicts": {
            "positive": "SOURCE_EXPERIENCE_ACCELERATES_TARGET_ADAPTATION_NATURAL",
            "positive_requires": [
                "A5 strictly fewer trials to LOCAL_ACTIVE in at least one of "
                "CPC / CPM",
                "no speed regression or unexplained negative transfer in the "
                "other",
                "A5 harmed receipts <= A3 in both",
                "A5 final delayed >= A3 - 0.005",
                "no Skill with a harmful delayed keeps execution rights",
            ],
            "positive_caveat": (
                "NATURAL / provisional: one Target domain, two cohorts.  A "
                "positive reading does not claim general cross-domain "
                "transfer and awaits replication on an independent dataset; "
                "it also authorizes no platform work -- the next step is "
                "still whatever the earliest blocker turns out to be"
            ),
            "others": [
                "SOURCE_EXPERIENCE_SAFER_NOT_FASTER",
                "NO_SOURCE_EXPERIENCE_ADVANTAGE",
                "SOURCE_EXPERIENCE_NEGATIVE_TRANSFER",
                "SOURCE_CONTEXT_NOT_RETRIEVED",
                "SOURCE_EXPERIENCE_RETRIEVED_NO_BEHAVIOR_CHANGE",
                "NO_ADOPTABLE_PLAN_IN_TARGET",
                "TARGET_FEEDBACK_UNREADABLE",
                "INCOMPLETE_LLM_BUDGET",
                "TARGET_LABEL_WALL_BREACHED",
            ],
            "highest_priority": "TARGET_LABEL_WALL_BREACHED",
        },
        "target_window_plan": {
            cohort: {name: _plain(row["windows"])
                     for name, row in rows.items()}
            for cohort, rows in universe["target"].items()
        },
        "source_bank_task_consumer_key": bank["task_consumer_key"],
    }


def _remedies_not_taken() -> dict[str, Any]:
    """What could be done about the shape fault, and why none of it was.

    Every one of these changes something the book fixed in advance -- the
    roster, the gate, or the Target bytes.  Choosing among them after seeing
    which files failed is exactly the move the pre-registration exists to
    prevent, so they are listed for the main line and none is applied.
    """
    return {
        "not_self_adjudicated": (
            "the gate is pre-registered and it failed; picking a remedy here "
            "would be picking it with knowledge of which files failed"
        ),
        "options": [
            {"option": "drop exchange-2 and run on two Target series",
             "blocked_by": "the book fixes all six realAdExchange series and "
                           "forbids replacement"},
            {"option": "de-duplicate the repeated timestamp before gating",
             "blocked_by": "that is a preprocessing step applied to Target "
                           "bytes, which this round is not authorized to "
                           "touch, and it would silently change what the "
                           "'strictly increasing' check means"},
            {"option": "relax the gate to non-decreasing timestamps",
             "blocked_by": "relaxing a pre-registered threshold after seeing "
                           "the data is the move the pre-registration is for"},
            {"option": "keep the gate and re-site the Target on another NAB "
                       "family",
             "blocked_by": "the Target roster is book-fixed; a new family is "
                           "a new book"},
        ],
        "what_the_defect_is": (
            "exchange-2_cpc_results.csv and exchange-2_cpm_results.csv each "
            "repeat the timestamp 2011-08-24 12:00:01 on two consecutive rows "
            "carrying different values.  It is one duplicated stamp per file, "
            "not a corrupted region, and it is present in the pinned upstream "
            "bytes rather than introduced here."
        ),
        "collateral_source_finding": (
            "the same check also fails two Source files -- "
            "ec2_request_latency_system_failure.csv (11 duplicated stamps at "
            "2014-03-09 03:00:00) and machine_temperature_system_failure.csv "
            "(one backwards step, 02:55 -> 02:00 on 2014-01-07, the shape of "
            "a DST fold).  These do not trigger the book's Target-keyed stop, "
            "and they are reported rather than worked around: with them "
            "excluded the known-cause Source cohort would be four files, not "
            "the six the book fixed."
        ),
        "reading": (
            "this is a real property of NAB v1.1, not an instrument fault: "
            "four of the twenty files carry a non-monotonic timestamp column. "
            "Whatever the main line rules, the ruling belongs in the book "
            "before the surface is frozen, not after."
        ),
    }


def _exposure_labels() -> dict[str, Any]:
    """N3: the exposure ledger, as report fields.  No new platform."""
    return {
        "source": {"context": "INSTANCE_SEEN", "outcome": "EXPOSED",
                   "note": "the development surface this round mines"},
        "target": {"context": "INSTANCE_SEEN",
                   "note": "the shape gate reads it and the Agent will see "
                           "its public values",
                   "outcome": "SEALED"},
        "aggregate_disclosure": TARGET_AGGREGATE_DISCLOSURE,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_standing": EVIDENCE_STANDING,
    }


# =========================================================================== #
# --evaluate: implemented and frozen; refuses to run until released
# =========================================================================== #
def evaluate() -> int:
    """The frozen A5-vs-A3 run.  Gated shut this round by design."""
    # the live plan is the row-order-contract one; v1 is kept only as the
    # shape-contract diagnostic that produced that contract
    source = OUT_JSON_V2 if OUT_JSON_V2.exists() else OUT_JSON
    if not source.exists():
        print(json.dumps({
            "verdict": "EVALUATE_NOT_RELEASED",
            "reason": "no frozen plan artifact at %s; run --plan-v2 first"
                      % _repo_rel(OUT_JSON_V2)}, indent=1))
        return 0
    plan = json.loads(source.read_text(encoding="utf-8"))
    if not bool(plan.get("evaluate_released")):
        print(json.dumps({
            "verdict": "EVALUATE_NOT_RELEASED",
            "reason": (
                "the frozen plan carries evaluate_released: false.  The "
                "protocol is implemented and frozen; running it is the main "
                "line's call, taken after confirming the Target outcome is "
                "still sealed."),
            "plan_artifact": _repo_rel(source),
            "protocol_version": plan.get("protocol_version"),
            "plan_verdict": (plan.get("verdict") or {}).get("verdict"),
        }, indent=1, ensure_ascii=False))
        return 0
    return _evaluate_released(plan)


# =========================================================================== #
# the evaluate execution body -- the real chain, parameterized only in where
# its cells come from and which agent drives them
# =========================================================================== #
SUPPORT_TRIAL_BUDGET = EVALUATE_SUPPORT_TRIALS


class _NABScopeExecutor(ScopeExecutor):
    """The frozen executor, with one thing overridden: what a window is.

    ScopeExecutor's ``training_windows`` is forecast-shaped -- 192 points of
    context plus a 48-point horizon, anchored on a config list.  The NAB
    geometry is different: each series' action region is its own training
    block, [0, 0.40n) or [0, 0.70n) of its own length.  Only that mapping is
    overridden; ``verify``, ``evaluate``, ``_compiled`` and the baseline
    cache are the frozen implementations, so the window verifier and the
    max_modified_fraction guard run exactly as they always do, over the
    region the program actually acts on.
    """

    def __init__(self, *, rows: Mapping[str, Any], round_name: str,
                 evaluate_fn: Any) -> None:
        roster = [{"series_uid": uid, "role": "train"} for uid in sorted(rows)]
        values = {uid: np.asarray(rows[uid]["values"], dtype=np.float64)
                  for uid in rows}
        super().__init__(roster, values, {"anchors": []},
                         evaluate_fn=evaluate_fn)
        self._rows = dict(rows)
        self._round = str(round_name)

    def training_windows(self, origin: int):  # noqa: D401 - frozen signature
        out = []
        for uid in sorted(self._rows):
            lo, hi = self._rows[uid]["windows"][self._round]["train"]
            out.append((uid, int(lo),
                        np.asarray(self._rows[uid]["values"],
                                   dtype=np.float64)[lo:hi]))
        return out


class _NABConsumerAdapter:
    """steps -> task-native readings, in the executor's own shape.

    Decides no relation, picks no winner, reads and writes no Skill, applies
    no risk threshold.  Same contract the T5 AD adapter kept: the executor's
    gain arithmetic is baseline-minus-candidate over ``mean_smase``, and the
    AD reading is higher-is-better, so it is reported negated and the
    executor's own subtraction yields candidate_F1 - baseline_F1 unchanged.
    """

    def __init__(self, *, consumer: Any, rows: Mapping[str, Any],
                 round_name: str, wall: LabelWall, budget: FitBudget,
                 support_origin: int, delayed_origin: int) -> None:
        self._consumer = consumer
        self._rows = dict(rows)
        self._round = str(round_name)
        self._wall = wall
        self._budget = budget
        self._support_origin = int(support_origin)
        self._delayed_origin = int(delayed_origin)
        self._models: dict[tuple[str, str], Any] = {}
        self._truth: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []

    def _part_for(self, origin: int) -> str:
        return "support" if int(origin) < self._delayed_origin else "delayed"

    def _model(self, uid: str, signature: str, program_steps: Any) -> Any:
        key = (uid, signature)
        if key in self._models:
            return self._models[key]
        row = self._rows[uid]
        lo, hi = row["windows"][self._round]["train"]
        block = np.asarray(row["values"], dtype=np.float64)[lo:hi]
        if program_steps:
            result = run_pipeline(list(program_steps), block)
            if not result.ok or result.artifact is None:
                raise Stop("TARGET_FEEDBACK_UNREADABLE",
                           "program %s failed on a training block: %s"
                           % (signature, result.error))
            block = np.asarray(result.artifact, dtype=np.float64).ravel()
        self._budget.spend(1)
        model = self._consumer.fit_series(block)
        self._models[key] = model
        return model

    def _truth_for(self, uid: str) -> list[list[int]]:
        if uid not in self._truth:
            row = self._rows[uid]
            self._truth[uid] = self._wall.windows(
                row["nab_key"], row["timestamps"])["events"]
        return self._truth[uid]

    def __call__(self, roster, values, compiled, config, *, origin):
        steps = compiled_steps(compiled)
        signature = "|".join(op for op, _p in steps) or "identity"
        part = self._part_for(int(origin))
        per_view: list[float] = []
        rows: dict[str, Any] = {}
        behavior = 0
        for uid in sorted(self._rows):
            model = self._model(uid, signature, steps)
            lo, hi = self._rows[uid]["windows"][self._round][part]
            raw = np.asarray(self._rows[uid]["values"], dtype=np.float64)
            reading = self._consumer.score_series(
                model, raw, (lo, hi), self._truth_for(uid))
            rows[uid] = reading
            per_view.append(float(reading["f1"]))
            behavior += int(model["training_windows"])
        macro = self._consumer.macro_f1(rows)
        out = {
            "mean_smase": -float(macro if macro is not None else 0.0),
            "per_view_smase": [-v for v in per_view],
            "behavior_point_count": int(behavior),
            "ad_macro_f1": float(macro) if macro is not None else None,
            "ad_f1_by_series": {uid: rows[uid]["f1"] for uid in rows},
            "part": part,
        }
        self.calls.append({"signature": signature, "part": part,
                           "origin": int(origin),
                           "macro_f1": out["ad_macro_f1"]})
        return out


def _card_builder_for(task_kind: str):
    """The Failure Card the Runtime turns into observable_applicability.

    Only task_kind is claimed, which is what the Observation vocabulary can
    actually carry: a wide scope, so the Runtime writes
    requires_target_support=true and the new Skill lands as a Draft rather
    than something that auto-prioritizes itself.
    """

    def build(_episode: object) -> Mapping[str, object]:
        return {"pattern_id": "t6-nab-target-block",
                "failure_family": "natural_readiness_observation",
                "observable_signature": {"task_kind": str(task_kind)}}

    return build


def _cohort_origins(rows: Mapping[str, Any], round_name: str
                    ) -> tuple[int, int]:
    """The scalar origins run_online_round needs, from a per-series plan.

    Each series keeps its own frozen window spans -- the Consumer reads those
    and nothing else.  But run_online_round takes one integer origin per
    round (it slices series0 for bind_round_data and computes the deployment
    Context at that origin), and three series of different lengths cannot
    each be it.  The cohort minimum is used: it is the largest origin that is
    inside every series' own training block, so no series' Context reaches
    past its own frozen split.
    """
    support = min(int(rows[uid]["windows"][round_name]["support"][0])
                  for uid in rows)
    delayed = min(int(rows[uid]["windows"][round_name]["delayed"][0])
                  for uid in rows)
    return support, delayed


def _target_task_spec() -> Any:
    """Same task, same Consumer, same menu as the Source bank.

    Identical by construction rather than by restatement: the Target rounds
    must retrieve Source Episodes, and retrieval is keyed on the task key, so
    a second hand-written spec here would be a second dialect.
    """
    return _source_task_spec()


def _retrieved_source_cards(method: Any, features: Mapping[str, Any],
                            task_key: str) -> dict[str, Any]:
    """Which Source cards the round's retrieval would surface, and whether
    they are Source cards at all.

    This mirrors the resolution fast_agent performs internally with the same
    inputs; the trace only publishes a status string, and A4 asks for ids.
    It is a read, and it changes nothing.
    """
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
        render_experience_pack,
        resolve_experience_contrast_pack,
    )
    held = list(getattr(method, "experience_episodes", ()) or ())
    if not held:
        return {"held": 0, "card_ids": [], "rendered": False,
                "source_cards": [], "target_cards": []}
    pack = resolve_experience_contrast_pack(
        held, dict(features), task_key,
        allowed_operators=tuple(NON_IDENTITY))
    if pack is None:
        return {"held": len(held), "card_ids": [], "rendered": False,
                "source_cards": [], "target_cards": []}
    payload = pack.to_dict()
    rendered = render_experience_pack(payload)
    ids: list[str] = []
    for slot in ("positive", "negative", "conflict", "abstain"):
        episode = getattr(pack, slot, None)
        if episode is not None:
            ids.append(str(episode.episode_id))
    return {
        "held": len(held),
        "card_ids": ids,
        "source_cards": [i for i in ids if i.startswith("t6_source_")],
        "target_cards": [i for i in ids if not i.startswith("t6_source_")],
        "rendered": bool(rendered),
        "rendered_sha256": canonical_sha256(rendered) if rendered else None,
    }


def _run_cells(
    *,
    plan: Mapping[str, Any],
    cohort_rows: Mapping[str, Mapping[str, Any]],
    agent_factory: Any,
    backend_factory: Any,
    llm_budget: int,
    fit_budget: FitBudget,
    wall: LabelWall,
    store_tag: str,
    order_override: Mapping[str, Any] | None = None,
    support_trial_budget: int | None = None,
    snapshot_for_arm: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The eight frozen cells, executed through the live chain.

    Every reading in here is produced by a real call: run_online_round writes
    the Episodes, open_delayed classifies the delayed window, and
    activate_approved is what promotes a snapshot.  Nothing is pre-written.
    """
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
        activate_approved,
        open_delayed,
        run_online_round,
    )

    consumer = _load_consumer()
    # plan is retained on the frozen signature so callers do not change;
    # Source bank Episodes no longer enter construction-time Memory.
    _ = plan
    h0 = compile_snapshot(
        PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
        verify_lock=False)
    spec = _target_task_spec()
    task_key = task_consumer_key(spec)

    call_counts: dict[str, dict[str, int]] = {}
    arms: dict[str, Any] = {}
    cells: list[dict[str, Any]] = []
    stopped: str | None = None
    # ONE backend for the entire experiment.  The frozen budget is a global
    # ledger: a backend per cohort x arm would have handed each of the four
    # its own full cap, so the run could have spent 4 x 48 while every
    # counter still read "within budget".
    shared_backend = backend_factory(llm_budget)

    store_root = Path(tempfile.gettempdir()) / store_tag
    if store_root.exists():
        shutil.rmtree(store_root)
    trial_budget = (SUPPORT_TRIAL_BUDGET if support_trial_budget is None
                    else int(support_trial_budget))

    for cohort, order in (order_override or EVALUATE_ORDER).items():
        rows = cohort_rows[cohort]
        for arm, round_name in order:
            cell_id = "%s/%s/%s" % (cohort, arm, round_name)
            counts = call_counts.setdefault(
                cell_id, {"run_online_round": 0, "open_delayed": 0,
                          "activate_approved": 0})
            key = (cohort, arm)
            state = arms.get(key)
            if state is None:
                # a brand-new Method and a brand-new Store per cohort x arm,
                # and no Target Experience crosses between cohorts
                root = store_root / cohort / arm
                store = SnapshotStore(root / "snapshots")
                base = ((snapshot_for_arm or {}).get(arm) or h0)
                store.materialize(base)
                store.set_active(base.runtime_bundle_sha)
                agent = agent_factory(rows, shared_backend, round_name)
                # #42d Part A: both arms construct with an empty Experience
                # store.  Source evidence reaches A5' only as a compiled
                # Skill on the snapshot, never as bank Episodes in Memory.
                memory: tuple[Any, ...] = ()
                method = TTHAMethod(agent, base, memory)
                held = list(getattr(method, "experience_episodes", ()) or ())
                if held:
                    raise Stop(
                        "TARGET_FEEDBACK_UNREADABLE",
                        "construction-time experience_episodes must be "
                        "empty, found %d" % len(held))
                state = {
                    "store": store,
                    "controller": EditController(
                        store, surfaces=SurfaceRegistry(),
                        router=FaultRouter()),
                    "method": method,
                    "backend": shared_backend,
                    "memory_size": len(memory),
                    "first_active_round": None,
                    "non_identity_trials_before_active": 0,
                    "activated": False,
                }
                arms[key] = state

            # The AD fit budget has to be enforced here, at the cell
            # boundary, and not only inside the adapter.  ScopeExecutor.
            # evaluate deliberately catches everything its evaluate_fn
            # raises and returns "instrument failed, gain None" -- which is
            # right for an instrument fault and wrong for an exhausted
            # budget, because the round would keep going on empty readings.
            # The frozen executor is not this round's change surface, so the
            # check lives where the body owns it.
            minimum = (1 + trial_budget) * len(rows)
            budget_short = fit_budget.used + minimum > fit_budget.cap
            support_origin, delayed_origin = _cohort_origins(rows, round_name)
            adapter = _NABConsumerAdapter(
                consumer=consumer, rows=rows, round_name=round_name,
                wall=wall, budget=fit_budget,
                support_origin=support_origin, delayed_origin=delayed_origin)
            executor = _NABScopeExecutor(
                rows=rows, round_name=round_name, evaluate_fn=adapter)
            values = {uid: np.asarray(rows[uid]["values"], dtype=np.float64)
                      for uid in rows}
            series0 = values[sorted(rows)[0]]
            observed = dict(resolver.window_context(
                values, support_origin, PERIOD_HINT))
            observed["bound_period"] = float(PERIOD_HINT)
            request = PreparationRequest(
                "t6-%s" % cohort, series0[:support_origin], spec,
                dict(observed))
            features = dict(extract_public_features(
                series0[:support_origin], task_kind="anomaly_detection"))
            retrieval = _retrieved_source_cards(
                state["method"], features, task_key)

            record: dict[str, Any] = {
                "cell": cell_id, "cohort": cohort, "arm": arm,
                "round": round_name,
                "support_origin": support_origin,
                "delayed_origin": delayed_origin,
                "task_consumer_key": task_key,
                "memory_size": state["memory_size"],
                "retrieval_before_round": retrieval,
                "llm_calls_before": int(getattr(
                    state["backend"], "calls", 0)),
                "ad_fits_before": fit_budget.used,
            }
            # The live loop's first act is bind_round_data at this round's
            # origin, so the Gateway does follow the round; binding it here
            # too removes the reliance on that side effect and kills the
            # stale r1 binding the agent was constructed with.
            state["method"].bind_round_data(
                series0[:support_origin], task_kind="anomaly_detection")
            try:
                if budget_short:
                    raise Stop(
                        "CONSUMER_FIT_BUDGET_EXCEEDED",
                        "AD fit budget cannot cover cell %s: %d used of %d, "
                        "%d needed" % (cell_id, fit_budget.used,
                                       fit_budget.cap, minimum))
                counts["run_online_round"] += 1
                result = run_online_round(
                    state["method"], executor, request, values,
                    origin=support_origin, slow_agent=None,
                    controller=state["controller"], store=state["store"],
                    card_builder=_card_builder_for("anomaly_detection"),
                    round_name="%s_%s" % (arm.lower(), round_name),
                    budget=trial_budget, allow_slow=False,
                    domain=cohort, period=PERIOD_HINT,
                    fast_features=features,
                    allow_fast_skill=True, runtime_prior_slot=False)
                counts["open_delayed"] += 1
                open_delayed(result, executor,
                             delayed_origin=delayed_origin,
                             store=state["store"])
                activated = False
                if result.approved_skill_id is not None:
                    counts["activate_approved"] += 1
                    activated = activate_approved(result, state["store"])
            except TargetLabelWallBreached:
                raise  # highest priority, never downgraded
            except Stop as stop:
                # the finished cells are the honest part of the artifact and
                # are kept; only this cell is unfinished
                record["error"] = "%s: %s" % (stop.verdict, stop.reason)
                cells.append(record)
                stopped = stop.verdict
                break
            except Exception as exc:  # noqa: BLE001
                record["error"] = "%s: %s" % (type(exc).__name__, exc)
                cells.append(record)
                stopped = _classify_evaluate_error(exc)
                break

            unreadable = [p for p in result.actual_probed_programs
                          if p.get("kind") == "probe" and p.get("gain") is None]
            if unreadable and fit_budget.used >= fit_budget.cap:
                raise Stop(
                    "CONSUMER_FIT_BUDGET_EXCEEDED",
                    "cell %s returned %d unreadable probe(s) with the AD fit "
                    "budget exhausted at %d" % (cell_id, len(unreadable),
                                                fit_budget.cap))
            trace = state["method"].last_trace
            episodes = list(state["method"].experience_episodes)
            fresh = [e for e in episodes if e.episode_id in
                     set(result.episode_ids)]
            non_identity = [p for p in result.actual_probed_programs
                            if p.get("kind") == "probe"
                            and str(p.get("candidate_id", "")) != "identity"]
            # Two layers, never blended.  A delayed reading that silently
            # falls back to the Support layer reports a delayed number that
            # no delayed window produced; when there is no delayed response
            # the delayed readings are None and say so.
            harmed_support = 0
            harmed_delayed = 0
            worst_delayed = None
            delayed_evaluated = 0
            for episode in fresh:
                support_facts = ((episode.support_response or {}).get(
                    MEASURED_EFFECT_KEY) or {})
                harmed_support += int(
                    support_facts.get("harmed_series_count") or 0)
                delayed_response = episode.delayed_response or {}
                if not delayed_response.get("evaluated"):
                    continue
                delayed_facts = delayed_response.get(MEASURED_EFFECT_KEY) or {}
                if not delayed_facts:
                    continue
                delayed_evaluated += 1
                harmed_delayed += int(
                    delayed_facts.get("harmed_series_count") or 0)
                value = delayed_facts.get("min_per_series_gain")
                if value is not None:
                    worst_delayed = (value if worst_delayed is None
                                     else min(worst_delayed, value))
            local_active = [e.episode_id for e in fresh
                            if e.local_status == "LOCAL_ACTIVE"]
            if not state["activated"]:
                state["non_identity_trials_before_active"] += len(non_identity)
                if local_active:
                    state["activated"] = True
                    state["first_active_round"] = round_name
            retrieved_ids = list(getattr(trace, "retrieved_skill_ids", ()) or ())
            pool = list(getattr(trace, "candidate_ids", ()) or ())
            probe_order = list(getattr(result, "probe_order_after_card", ())
                               or getattr(result, "probe_order_before_card", ())
                               or [])
            probes = [{"candidate_id": p["candidate_id"],
                       "kind": p.get("kind"), "gain": p.get("gain")}
                      for p in result.actual_probed_programs]
            hampel_proposed = any("hampel_filter" in str(x) for x in pool)
            hampel_chosen = "hampel_filter" in str(
                getattr(trace, "chosen_candidate_id", "") or "")
            hampel_probed = sum(
                1 for p in probes
                if p.get("kind") == "probe"
                and "hampel_filter" in str(p.get("candidate_id", "")))
            non_hampel_proposals = [
                x for x in pool
                if x and x != "identity" and "hampel_filter" not in str(x)]
            non_hampel_probes = [
                p for p in probes
                if p.get("kind") == "probe"
                and str(p.get("candidate_id", "")) != "identity"
                and "hampel_filter" not in str(p.get("candidate_id", ""))]
            record.update({
                "retrieved_skill_ids": retrieved_ids,
                "pool": pool,
                "pool_order": list(pool),
                "chosen": getattr(trace, "chosen_candidate_id", None),
                "probe_order": probe_order,
                "hampel_proposed": hampel_proposed,
                "hampel_chosen": hampel_chosen,
                "hampel_probed": hampel_probed,
                "non_hampel_proposal_count": len(non_hampel_proposals),
                "non_hampel_probe_count": len(non_hampel_probes),
                "memory_resolution": getattr(
                    trace, "memory_resolution_status", None),
                "proposal_count": result.proposal_count,
                "support_receipts": result.target_support_receipts_used,
                "non_identity_trials": len(non_identity),
                "probes": probes,
                "winner_program": _plain(result.winner_program),
                "abstained": bool(getattr(result, "abstained", False)),
                "harm_count": result.harm_count,
                "harmed_series_support_layer": harmed_support,
                "harmed_series_delayed_layer": harmed_delayed,
                "worst_per_series_delayed_gain": worst_delayed,
                "delayed_responses_evaluated": delayed_evaluated,
                "fast_skill_event": _plain(result._fast_skill_event),
                "delayed_event": _plain(result._delayed_event),
                "delayed_utility": result.delayed_utility,
                "approved_skill_id": result.approved_skill_id,
                "activated": activated,
                "episode_rows": [{
                    "episode_id": e.episode_id,
                    "task_consumer_key": e.task_consumer_key,
                    "domain_namespace": e.domain_namespace,
                    "workflow_signature": e.workflow_signature,
                    "relation": e.relation,
                    "evidence_level": e.evidence_level,
                    "local_status": e.local_status,
                } for e in fresh],
                "local_active_episodes": local_active,
                "restricted_episodes": [e.episode_id for e in fresh
                                        if e.local_status == "RESTRICTED"],
                "llm_calls_after": int(getattr(state["backend"], "calls", 0)),
                "ad_fits_after": fit_budget.used,
                "adapter_calls": list(adapter.calls),
            })
            cells.append(record)
        if stopped:
            break

    readings = _cell_readings(cells, arms)
    return {
        "cells": cells,
        "call_counts": call_counts,
        "readings": readings,
        "stopped": stopped,
        "llm_calls": int(getattr(shared_backend, "calls", 0)),
        "llm_ledger": "one shared backend across every cohort, arm and round",
        "llm_budget": llm_budget,
        "ad_fits": fit_budget.used,
        "ad_fit_cap": fit_budget.cap,
        "store_root": _repo_rel(store_root),
        "task_consumer_key": task_key,
    }


def _classify_evaluate_error(exc: Exception) -> str:
    text = "%s: %s" % (type(exc).__name__, exc)
    if "budget" in text.lower():
        return "INCOMPLETE_LLM_BUDGET"
    if "AgentProtocolError" in text or "StagePostValidation" in text:
        return "TARGET_FEEDBACK_UNREADABLE"
    return "TARGET_FEEDBACK_UNREADABLE"


def _cell_readings(cells: Sequence[Mapping[str, Any]],
                   arms: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
    """The book's primary readings, each from the trajectory that produced it."""
    out: dict[str, Any] = {}
    for (cohort, arm), state in arms.items():
        rows = [c for c in cells
                if c["cohort"] == cohort and c["arm"] == arm
                and "error" not in c]
        activated = [c for c in rows if c.get("local_active_episodes")]
        trials = sum(int(c.get("non_identity_trials") or 0) for c in rows)
        before_active = (state["non_identity_trials_before_active"]
                         if state["activated"] else None)
        final = rows[-1] if rows else {}
        out["%s/%s" % (cohort, arm)] = {
            "non_identity_trials_before_first_local_active": (
                before_active if before_active is not None else ">%d"
                % max(trials, len(EVALUATE_ORDER[cohort]) * SUPPORT_TRIAL_BUDGET)),
            "first_activation_round": state["first_active_round"],
            "final_delayed_macro_f1_gain": (
                final.get("delayed_utility")
                if int(final.get("delayed_responses_evaluated") or 0) > 0
                else None),
            "harmed_support_receipts": sum(
                int(c.get("harm_count") or 0) for c in rows),
            "harmed_series_support_layer": sum(
                int(c.get("harmed_series_support_layer") or 0) for c in rows),
            "harmed_series_delayed_layer": sum(
                int(c.get("harmed_series_delayed_layer") or 0) for c in rows),
            "worst_per_series_delayed_gain": (
                min(_worsts) if (_worsts := [
                    c["worst_per_series_delayed_gain"] for c in rows
                    if c.get("worst_per_series_delayed_gain") is not None])
                else None),
            "delayed_responses_evaluated": sum(
                int(c.get("delayed_responses_evaluated") or 0) for c in rows),
            "abstentions": sum(1 for c in rows if c.get("abstained")),
            "source_cards_retrieved": sorted({
                cid for c in rows
                for cid in (c.get("retrieval_before_round") or {}).get(
                    "source_cards", [])}),
            "memory_resolution_per_round": [c.get("memory_resolution")
                                            for c in rows],
            "target_feedback_consumed": sum(
                int(c.get("support_receipts") or 0) for c in rows),
            "llm_calls": (rows[-1]["llm_calls_after"] if rows else 0),
            "rounds_completed": len(rows),
            "activations": len(activated),
        }
    return out


def _evaluate_released(plan: Mapping[str, Any]) -> int:
    """The body the release switch turns on.  One real pass, eight cells."""
    wall = LabelWall(released=True)
    universe = _load_universe(gate_all(row_order_contract=True))
    budget = FitBudget(EVALUATE_AD_FIT_BUDGET)
    try:
        run = _run_cells(
            plan=plan,
            cohort_rows=universe["target"],
            agent_factory=_evaluate_agent,
            backend_factory=_evaluate_backend,
            llm_budget=EVALUATE_LLM_BUDGET,
            fit_budget=budget,
            wall=wall,
            store_tag="t6e",
        )
    except Stop as stop:
        print(json.dumps({"verdict": stop.verdict, "reason": stop.reason},
                         ensure_ascii=False, indent=1))
        return 1
    run["label_wall"] = wall.audit()
    verdict = _evaluate_verdict(run)
    payload = {
        "protocol_version": PROTOCOL_VERSION_V2 + "_evaluate",
        "entry": "--evaluate",
        "plan_artifact": _repo_rel(OUT_JSON_V2),
        "run": run,
        "verdict": verdict,
    }
    OUT_EVALUATE.write_text(_json_text(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict["verdict"],
                      "reason": verdict["reason"][:220]},
                     ensure_ascii=False, indent=1))
    print("wrote", OUT_EVALUATE, flush=True)
    return 0 if not run.get("stopped") else 1


def _evaluate_verdict(run: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen ten-cell ladder, decided mechanically.

    Written before any Target outcome exists so no rule can be added after
    seeing one.  Read top to bottom; the first cell that fires is the verdict.
    """
    readings = run.get("readings") or {}
    cohorts = sorted({key.split("/")[0] for key in readings})

    def read(cohort: str, arm: str, field: str) -> Any:
        return (readings.get("%s/%s" % (cohort, arm)) or {}).get(field)

    def trials(cohort: str, arm: str) -> float:
        value = read(cohort, arm, "non_identity_trials_before_first_local_active")
        if isinstance(value, (int, float)):
            return float(value)
        return float("inf")  # ">N": never activated

    # ---- 1. the wall outranks everything -------------------------------
    if (run.get("label_wall") or {}).get("breached"):
        return {"verdict": "TARGET_LABEL_WALL_BREACHED",
                "reason": "a sealed Target label key was served"}
    # ---- 2/3. mechanical blockers --------------------------------------
    completed = [c for c in run.get("cells", []) if "error" not in c]
    expected = sum(len(order) for order in EVALUATE_ORDER.values())
    if run.get("stopped") == "INCOMPLETE_LLM_BUDGET" or (
            run.get("llm_calls", 0) >= run.get("llm_budget", 0)
            and len(completed) < expected):
        return {"verdict": "INCOMPLETE_LLM_BUDGET",
                "reason": "the shared LLM ledger ran out at %s of %s calls "
                          "after %d of %d cells"
                          % (run.get("llm_calls"), run.get("llm_budget"),
                             len(completed), expected),
                "cells_completed": len(completed)}
    if run.get("stopped") or len(completed) < expected:
        return {"verdict": "TARGET_FEEDBACK_UNREADABLE",
                "reason": "the trajectory stopped (%s) after %d of %d cells"
                          % (run.get("stopped"), len(completed), expected),
                "cells_completed": len(completed)}
    # ---- 4. nothing adoptable anywhere ---------------------------------
    if not any(c.get("winner_program") for c in completed):
        return {"verdict": "NO_ADOPTABLE_PLAN_IN_TARGET",
                "reason": "no round in either arm produced an adoptable "
                          "plan; not re-drawn"}
    # ---- 5. A5's Source context never arrived --------------------------
    a5_cards = {cohort: read(cohort, "A5", "source_cards_retrieved") or []
                for cohort in cohorts}
    if not any(a5_cards.values()):
        return {"verdict": "SOURCE_CONTEXT_NOT_RETRIEVED",
                "reason": "no A5 round retrieved a Source card, so the arm "
                          "that was supposed to differ never did",
                "retrieved": a5_cards}
    # ---- 6. it arrived and changed nothing -----------------------------
    def behaviour(cohort: str, arm: str) -> list[Any]:
        return [(c.get("pool"), c.get("chosen"),
                 [p["candidate_id"] for p in c.get("probes", [])],
                 c.get("winner_program"))
                for c in completed
                if c["cohort"] == cohort and c["arm"] == arm]

    identical = {cohort: behaviour(cohort, "A5") == behaviour(cohort, "A3")
                 for cohort in cohorts}
    if all(identical.values()):
        return {"verdict": "SOURCE_EXPERIENCE_RETRIEVED_NO_BEHAVIOR_CHANGE",
                "reason": "A5 retrieved Source cards but every pool, choice, "
                          "probe order and winner matched A3 in both cohorts",
                "identical_per_cohort": identical}
    # ---- the safety and speed readings the remaining cells share -------
    harmed_worse = {
        cohort: (int(read(cohort, "A5", "harmed_support_receipts") or 0)
                 > int(read(cohort, "A3", "harmed_support_receipts") or 0))
        for cohort in cohorts}
    delayed_worse = {}
    for cohort in cohorts:
        a5_final = read(cohort, "A5", "final_delayed_macro_f1_gain")
        a3_final = read(cohort, "A3", "final_delayed_macro_f1_gain")
        delayed_worse[cohort] = (
            a5_final is not None and a3_final is not None
            and float(a5_final) < float(a3_final) - MATERIAL_THRESHOLD)
    unsafe_active = [
        c["cell"] for c in completed
        if c.get("activated")
        and (c.get("worst_per_series_delayed_gain") is not None)
        and float(c["worst_per_series_delayed_gain"]) < -MATERIAL_THRESHOLD]
    slower = {cohort: trials(cohort, "A5") > trials(cohort, "A3")
              for cohort in cohorts}
    faster = {cohort: trials(cohort, "A5") < trials(cohort, "A3")
              for cohort in cohorts}
    safer = {
        cohort: (int(read(cohort, "A5", "harmed_support_receipts") or 0)
                 < int(read(cohort, "A3", "harmed_support_receipts") or 0))
        for cohort in cohorts}
    evidence = {
        "trials_to_first_local_active": {
            cohort: {"A5": read(cohort, "A5",
                                "non_identity_trials_before_first_local_active"),
                     "A3": read(cohort, "A3",
                                "non_identity_trials_before_first_local_active")}
            for cohort in cohorts},
        "harmed_support_receipts": {
            cohort: {"A5": read(cohort, "A5", "harmed_support_receipts"),
                     "A3": read(cohort, "A3", "harmed_support_receipts")}
            for cohort in cohorts},
        "final_delayed_gain": {
            cohort: {"A5": read(cohort, "A5", "final_delayed_macro_f1_gain"),
                     "A3": read(cohort, "A3", "final_delayed_macro_f1_gain")}
            for cohort in cohorts},
        "faster": faster, "slower": slower, "safer": safer,
        "harmed_worse": harmed_worse, "delayed_worse": delayed_worse,
        "activated_skills_with_harmful_delayed": unsafe_active,
        "source_cards_retrieved": a5_cards,
    }
    # ---- 7. negative transfer ------------------------------------------
    if any(harmed_worse.values()) or any(delayed_worse.values()) or unsafe_active:
        return {"verdict": "SOURCE_EXPERIENCE_NEGATIVE_TRANSFER",
                "reason": ("carrying Source Experience made the Target worse: "
                           "harmed-receipt regression %s, delayed regression "
                           "%s, activated Skills with a harmful delayed %s"
                           % (harmed_worse, delayed_worse, unsafe_active)),
                "evidence": evidence}
    # ---- 8. the pre-registered positive --------------------------------
    positive = (any(faster.values())
                and not any(slower.values())
                and not any(harmed_worse.values())
                and not any(delayed_worse.values())
                and not unsafe_active)
    if positive:
        return {
            "verdict": "SOURCE_EXPERIENCE_ACCELERATES_TARGET_ADAPTATION_NATURAL",
            "reason": ("A5 reached LOCAL_ACTIVE in strictly fewer "
                       "non-identity trials in at least one cohort with no "
                       "speed regression in the other, no harmed-receipt "
                       "regression, final delayed within the material line, "
                       "and no delayed-harmful Skill keeping execution "
                       "rights"),
            "caveat": ("NATURAL / provisional: one Target domain, two "
                       "cohorts.  This does not claim general cross-domain "
                       "transfer and awaits replication on an independent "
                       "dataset; it authorizes no platform work"),
            "evidence": evidence}
    # ---- 9. safer but not faster ---------------------------------------
    if any(safer.values()) and not any(harmed_worse.values()):
        return {"verdict": "SOURCE_EXPERIENCE_SAFER_NOT_FASTER",
                "reason": ("A5 did not reach LOCAL_ACTIVE in fewer trials, "
                           "but took strictly fewer harmed Support receipts "
                           "in at least one cohort and none more anywhere"),
                "evidence": evidence}
    # ---- 10. no advantage ----------------------------------------------
    return {"verdict": "NO_SOURCE_EXPERIENCE_ADVANTAGE",
            "reason": ("A5 retrieved Source cards and behaved differently, "
                       "but neither reached LOCAL_ACTIVE sooner nor took "
                       "fewer harmed receipts"),
            "evidence": evidence}


def _evaluate_backend(llm_budget: int) -> Any:
    """The one budgeted backend the whole experiment shares."""
    from evaluation.functional.task_episode_harness.agentic.runner import (
        _default_backend_factory,
    )
    return _default_backend_factory(int(llm_budget))


def _evaluate_agent(rows: Mapping[str, Any], backend: Any,
                    round_name: str) -> Any:
    """The pinned model over the shared backend, bound to this round.

    The Gateway is built at this round's own support origin; _run_cells then
    rebinds it before every round, so r2 never reads r1's public prefix.
    """
    from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
        LocalPublicToolGateway,
    )

    first = sorted(rows)[0]
    origin = min(int(rows[uid]["windows"][round_name]["support"][0])
                 for uid in rows)
    series0 = np.asarray(rows[first]["values"], dtype=np.float64)
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(series0[:origin],
                               task_kind="anomaly_detection"),
        model=EVALUATE_BACKEND["model"], base_url=EVALUATE_BACKEND["base_url"])
    return TTHAFastAgent(core)


# =========================================================================== #
# Part B -- the 0-LLM mechanical smoke over the same execution body
# =========================================================================== #
def _smoke_agent_factory(operators: Sequence[str]):
    """A scripted backend in the live Agent's shape.  Spends no LLM call."""

    def factory(rows: Mapping[str, Any], backend: Any,
                round_name: str) -> Any:
        from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
            LocalPublicToolGateway,
        )
        first = sorted(rows)[0]
        origin = min(int(rows[uid]["windows"][round_name]["support"][0])
                     for uid in rows)
        series0 = np.asarray(rows[first]["values"], dtype=np.float64)
        core = sealed.TTHAAgentCore(
            backend,
            LocalPublicToolGateway(series0[:origin],
                                   task_kind="anomaly_detection"))
        return sealed.TTHAFastAgent(core)

    return factory


def _smoke_backend_factory(operators: Sequence[str]):
    def factory(_budget: int) -> Any:
        return sealed.SealedProbeBackend(
            explore=True, operators=tuple(operators),
            max_propose_candidates=3, force_pool=True)

    return factory


def evaluate_smoke(*, fit_cap: int = SMOKE_AD_FIT_BUDGET,
                   llm_cap: int = EVALUATE_LLM_BUDGET,
                   out_path: Path | None = None) -> int:
    """Run the frozen execution body on Source stand-in cells, 0 LLM.

    This writes its own smoke artifact and never the v2 plan artifact; the
    frozen constants are untouched and only the cell source is parameterized.
    The Target label wall stays shut for the whole pass, and the run asserts
    it was never asked for a Target key.
    """
    if not OUT_JSON_V2.exists():
        print(json.dumps({"verdict": "SMOKE_MECHANICAL_FAULT",
                          "reason": "no v2 plan artifact to take the Source "
                                    "bank from"}, indent=1))
        return 1
    plan = json.loads(OUT_JSON_V2.read_text(encoding="utf-8"))
    wall = LabelWall(released=False)
    universe = _load_universe(gate_all(row_order_contract=True))
    stand_in: dict[str, Any] = {}
    for cell_cohort, source_cohort in SMOKE_COHORT_SOURCE.items():
        rows = universe["source"][source_cohort]
        chosen = sorted(rows)[:SMOKE_SERIES_PER_COHORT]
        stand_in[cell_cohort] = {name: rows[name] for name in chosen}
    out_file = out_path or OUT_SMOKE
    budget = FitBudget(int(fit_cap))
    checks: list[dict[str, Any]] = []
    run: dict[str, Any] = {}
    fault: str | None = None
    try:
        run = _run_cells(
            plan=plan,
            cohort_rows=stand_in,
            agent_factory=_smoke_agent_factory(SMOKE_OPERATORS),
            backend_factory=_smoke_backend_factory(SMOKE_OPERATORS),
            llm_budget=int(llm_cap),
            fit_budget=budget,
            wall=wall,
            store_tag="t6smoke",
        )
    except TargetLabelWallBreached as breach:
        payload = {"verdict": "TARGET_LABEL_WALL_BREACHED",
                   "reason": breach.reason, "label_wall": wall.audit()}
        out_file.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 1
    except Stop as stop:
        # raised outside any cell (nothing was started); still written out
        run = {"cells": [], "call_counts": {}, "readings": {},
               "stopped": stop.verdict, "llm_calls": 0,
               "ad_fits": budget.used, "ad_fit_cap": budget.cap,
               "stop_reason": stop.reason}
        fault = stop.verdict
    if run.get("stopped") == "CONSUMER_FIT_BUDGET_EXCEEDED":
        fault = run["stopped"]
        run.setdefault("stop_reason", next(
            (c["error"] for c in run.get("cells", []) if "error" in c), None))

    audit = wall.audit()
    completed = [c for c in run.get("cells", []) if "error" not in c]
    counts = run.get("call_counts", {})

    def check(cid: str, ok: bool, detail: str) -> None:
        checks.append({"id": cid, "ok": bool(ok), "detail": detail})

    expected_cells = sum(len(order) for order in EVALUATE_ORDER.values())
    check("B1_eight_cells_in_frozen_order",
          len(completed) == expected_cells
          and [c["cell"] for c in completed] == [
              "%s/%s/%s" % (cohort, arm, rnd)
              for cohort, order in EVALUATE_ORDER.items()
              for arm, rnd in order],
          "cells executed: %s" % [c["cell"] for c in completed])
    check("B1_live_calls_per_cell",
          bool(counts) and all(
              v["run_online_round"] >= 1 and v["open_delayed"] >= 1
              for v in counts.values()),
          "per-cell call counts: %s" % json.dumps(counts))

    keys = {c["cell"]: [e["task_consumer_key"]
                        for e in c.get("episode_rows", [])]
            for c in completed}
    expected_key = run.get("task_consumer_key")
    check("B2_episode_keys_correct",
          bool(keys) and all(all(k == expected_key for k in v)
                             for v in keys.values()),
          "every written Episode keyed %s" % expected_key)
    a5_seen = [c["cell"] for c in completed if c["arm"] == "A5"
               and (c.get("retrieval_before_round") or {}).get("source_cards")]
    a3_seen = [c["cell"] for c in completed if c["arm"] == "A3"
               and (c.get("retrieval_before_round") or {}).get("source_cards")]
    check("B2_a5_sees_bank_a3_does_not",
          bool(a5_seen) and not a3_seen,
          "A5 cells retrieving Source cards: %s; A3 cells: %s"
          % (a5_seen, a3_seen or "none"))

    lifecycle = [
        {"cell": c["cell"],
         "support_relations": [e["relation"] for e in c.get("episode_rows", [])],
         "fast_skill_stage": (c.get("fast_skill_event") or {}).get("stage"),
         "delayed_stage": (c.get("delayed_event") or {}).get("stage"),
         "delayed_relation": (c.get("delayed_event") or {}).get(
             "delayed_relation"),
         "local_status": [e["local_status"] for e in c.get("episode_rows", [])]}
        for c in completed]
    reached = [row for row in lifecycle if row["fast_skill_stage"] is not None]
    check("B3_lifecycle_reachable",
          bool(reached),
          "cells that reached a Draft decision: %s"
          % [(r["cell"], r["fast_skill_stage"], r["delayed_stage"])
             for r in reached] or "none")

    starved = int(fit_cap) < SMOKE_AD_FIT_BUDGET
    check("B4_budget_counts_down",
          bool(run.get("cells")) and all(
              c["ad_fits_after"] > c["ad_fits_before"] for c in completed),
          "per-cell AD fit counter strictly increases: %s"
          % [(c["cell"], c["ad_fits_before"], c["ad_fits_after"])
             for c in completed] or "no completed cell"),
    check("B4_overrun_interrupts",
          (run.get("stopped") == "CONSUMER_FIT_BUDGET_EXCEEDED") if starved
          else (fault is None and int(run.get("ad_fits") or 0)
                <= int(fit_cap)),
          ("starved run stopped with %s after %d of %d cells"
           % (run.get("stopped"), len(completed),
              sum(len(o) for o in EVALUATE_ORDER.values())))
          if starved else
          ("full run stayed inside the cap: %s of %s fits"
           % (run.get("ad_fits"), fit_cap)))
    check("B5_zero_target_label_service",
          audit["target_key_requests"] == [] and not audit["breached"]
          and not audit["target_values_retained_in_memory"],
          "target key requests %s, breached %s, target values retained %s"
          % (audit["target_key_requests"], audit["breached"],
             audit["target_values_retained_in_memory"]))

    payload = {
        "protocol_version": "t6_nab_evaluate_smoke_v1",
        "entry": "--evaluate-smoke",
        "note": ("the frozen execution body, run on Source stand-in cells "
                 "with a scripted backend; writes no plan artifact and "
                 "touches no Target outcome"),
        "stand_in_cohorts": {k: sorted(v) for k, v in stand_in.items()},
        "checks": checks,
        "lifecycle": lifecycle,
        "call_counts": counts,
        "budget_trace": {
            "ad_fits_used": run.get("ad_fits"),
            "ad_fit_cap": run.get("ad_fit_cap", budget.cap),
            "llm_calls": run.get("llm_calls", 0),
            "llm_cap": int(llm_cap),
            "per_cell": [{"cell": c["cell"],
                          "ad_fits_before": c["ad_fits_before"],
                          "ad_fits_after": c["ad_fits_after"],
                          "llm_calls_before": c["llm_calls_before"],
                          "llm_calls_after": c["llm_calls_after"]}
                         for c in completed],
        },
        "readings": run.get("readings"),
        "cells": run.get("cells"),
        "label_wall": audit,
        "stopped": run.get("stopped"),
        "stop_reason": run.get("stop_reason"),
        "evaluate_released_untouched": bool(
            plan.get("evaluate_released")) is False,
    }
    if fault:
        payload["verdict"] = {
            "verdict": "SMOKE_BUDGET_STOP_REACHED",
            "reason": run.get("stop_reason") or fault,
            "cells_completed": len(completed),
        }
        out_file.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        print("wrote", out_file, flush=True)
        return 1
    all_ok = all(c["ok"] for c in checks)
    payload["verdict"] = {
        "verdict": ("EVALUATE_BODY_COMPLETE_SMOKE_GREEN" if all_ok
                    else "SMOKE_MECHANICAL_FAULT"),
        "reason": ("the frozen execution body ran all eight cells through the "
                   "live chain with no Target label served"
                   if all_ok else
                   "first fault: %s" % [c["id"] for c in checks
                                        if not c["ok"]]),
    }
    out_file.write_text(_json_text(payload), encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
    print("wrote", out_file, flush=True)
    return 0 if all_ok else 1


# =========================================================================== #
# #42c -- one-cell mechanical lifecycle fixture (not method evidence)
# =========================================================================== #
def evaluate_lifecycle_fixture(*, fit_cap: int = FIXTURE_AD_FIT_BUDGET
                               ) -> int:
    """The already-exposed Source cell, through the live evaluate body.

    source_aws_cloudwatch / r2 / all eight series / identity + winsorize.
    Scripted backend, 0 LLM, Target outcome unread.  Marked
    MECHANICAL_FIXTURE: it closes the Draft → delayed branch the eight-cell
    stand-in never reached.  It is not a new method reading and it does not
    write a plan artifact.
    """
    if not OUT_JSON_V2.exists():
        print(json.dumps({"verdict": "LIFECYCLE_FIXTURE_UNREADABLE",
                          "reason": "no v2 plan artifact to take the Source "
                                    "bank from"}, indent=1))
        return 1
    plan = json.loads(OUT_JSON_V2.read_text(encoding="utf-8"))
    # The fixture never flips the release switch and never calls
    # --evaluate.  If the plan file already carries true, that is a
    # concurrent-tree fact to report, not a reason to abort a Source-only
    # mechanical pass that keeps LabelWall(released=False).
    wall = LabelWall(released=False)
    universe = _load_universe(gate_all(row_order_contract=True))
    source_rows = universe["source"][FIXTURE_SOURCE_COHORT]
    if len(source_rows) != 8:
        print(json.dumps({"verdict": "LIFECYCLE_FIXTURE_UNREADABLE",
                          "reason": "fixture requires the full 8-series "
                                    "source_aws_cloudwatch cohort, found %d"
                                    % len(source_rows)}, indent=1))
        return 1
    stand_in = {FIXTURE_CELL_SLOT: dict(source_rows)}
    order = {FIXTURE_CELL_SLOT: ((FIXTURE_ARM, FIXTURE_ROUND),)}
    budget = FitBudget(int(fit_cap))
    try:
        run = _run_cells(
            plan=plan,
            cohort_rows=stand_in,
            agent_factory=_smoke_agent_factory(FIXTURE_OPERATORS),
            backend_factory=_smoke_backend_factory(FIXTURE_OPERATORS),
            llm_budget=FIXTURE_LLM_BUDGET,
            fit_budget=budget,
            wall=wall,
            store_tag="t6fixture",
            order_override=order,
            support_trial_budget=FIXTURE_SUPPORT_TRIAL_BUDGET,
        )
    except TargetLabelWallBreached as breach:
        payload = {
            "protocol_version": "t6_nab_lifecycle_fixture_v1",
            "entry": "--evaluate-lifecycle-fixture",
            "mechanical_fixture": True,
            "counts_as_method_evidence": False,
            "verdict": {"verdict": "TARGET_LABEL_WALL_BREACHED",
                        "reason": breach.reason},
            "label_wall": wall.audit(),
        }
        OUT_LIFECYCLE_FIXTURE.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        return 1
    except Stop as stop:
        run = {"cells": [], "call_counts": {}, "readings": {},
               "stopped": stop.verdict, "llm_calls": 0,
               "ad_fits": budget.used, "ad_fit_cap": budget.cap,
               "stop_reason": stop.reason}

    audit = wall.audit()
    completed = [c for c in run.get("cells", []) if "error" not in c]
    cell = completed[0] if completed else {}
    probes = list(cell.get("probes") or [])
    support_gains = [p.get("gain") for p in probes
                     if p.get("kind") == "probe"
                     and "winsorize" in str(p.get("candidate_id", ""))]
    support_gain = (support_gains[0] if support_gains
                    and support_gains[0] is not None else None)
    delayed_event = cell.get("delayed_event") or {}
    fast_event = cell.get("fast_skill_event") or {}
    adapter_parts = [call.get("part") for call in cell.get("adapter_calls") or []]
    episode_rows = list(cell.get("episode_rows") or [])
    delayed_opened = (
        int(cell.get("delayed_responses_evaluated") or 0) > 0
        or delayed_event.get("delayed_relation") is not None
        or "delayed" in adapter_parts)
    checks = [
        {"id": "support_gain_material",
         "ok": support_gain is not None
               and float(support_gain) >= MATERIAL_THRESHOLD,
         "detail": "winsorize support gain = %s" % support_gain},
        {"id": "draft_formed",
         "ok": fast_event.get("stage") in ("pending", "approved")
               or any(e.get("local_status") in
                      ("LOCAL_DRAFT", "LOCAL_ACTIVE")
                      for e in episode_rows),
         "detail": "fast_skill_event=%s local_status=%s"
                   % (fast_event.get("stage"),
                      [e.get("local_status") for e in episode_rows])},
        {"id": "delayed_opened_and_classified",
         "ok": bool(delayed_opened)
               and (delayed_event.get("delayed_relation") is not None
                    or int(cell.get("delayed_responses_evaluated") or 0) > 0),
         "detail": "adapter_parts=%s delayed_event=%s evaluated=%s"
                   % (adapter_parts, delayed_event.get("delayed_relation"),
                      cell.get("delayed_responses_evaluated"))},
        {"id": "delayed_positive",
         "ok": delayed_event.get("delayed_relation") == "POSITIVE"
               or any(e.get("relation") == "POSITIVE"
                      and e.get("local_status") == "LOCAL_ACTIVE"
                      for e in episode_rows),
         "detail": "delayed_relation=%s episode_relations=%s"
                   % (delayed_event.get("delayed_relation"),
                      [(e.get("relation"), e.get("local_status"))
                       for e in episode_rows])},
        {"id": "episode_local_active",
         "ok": bool(cell.get("local_active_episodes")),
         "detail": "local_active_episodes=%s"
                   % cell.get("local_active_episodes")},
        {"id": "activate_approved",
         "ok": bool(cell.get("activated"))
               and int((run.get("call_counts") or {}).get(
                   cell.get("cell", ""), {}).get("activate_approved") or 0)
               >= 1,
         "detail": "activated=%s call_counts=%s"
                   % (cell.get("activated"), run.get("call_counts"))},
        {"id": "zero_target_label_service",
         "ok": audit["target_key_requests"] == [] and not audit["breached"]
               and not audit["target_values_retained_in_memory"],
         "detail": "target key requests %s, breached %s, retained %s"
                   % (audit["target_key_requests"], audit["breached"],
                      audit["target_values_retained_in_memory"])},
        {"id": "budget_inside_authorization",
         "ok": int(run.get("ad_fits") or 0) <= int(fit_cap)
               and int(run.get("llm_calls") or 0) == 0,
         "detail": "ad_fits=%s/%s llm_calls=%s"
                   % (run.get("ad_fits"), fit_cap, run.get("llm_calls"))},
    ]
    if audit.get("breached") or audit.get("target_key_requests"):
        verdict_name = "TARGET_LABEL_WALL_BREACHED"
        reason = "a Target label key was requested during the fixture"
    elif all(item["ok"] for item in checks):
        verdict_name = "LIFECYCLE_FIXTURE_CLOSED"
        reason = ("Support POSITIVE → Draft → delayed POSITIVE → "
                  "LOCAL_ACTIVE / activate_approved on the mechanical "
                  "fixture; not method evidence")
    else:
        verdict_name = "LIFECYCLE_FIXTURE_UNREADABLE"
        reason = "first unmet fixture check: %s" % [
            item["id"] for item in checks if not item["ok"]]
    payload = {
        "protocol_version": "t6_nab_lifecycle_fixture_v1",
        "entry": "--evaluate-lifecycle-fixture",
        "mechanical_fixture": True,
        "counts_as_method_evidence": False,
        "note": ("MECHANICAL_FIXTURE: the live evaluate body on the already-"
                 "exposed source_aws_cloudwatch r2 cell, identity + "
                 "winsorize, full 8 series.  Closes the lifecycle branch "
                 "the 2-series stand-in never reached.  Not a new method "
                 "reading."),
        "fixture": {
            "source_cohort": FIXTURE_SOURCE_COHORT,
            "round": FIXTURE_ROUND,
            "arm": FIXTURE_ARM,
            "cell_slot": FIXTURE_CELL_SLOT,
            "series": sorted(source_rows),
            "operators": list(FIXTURE_OPERATORS),
            "support_trial_budget": FIXTURE_SUPPORT_TRIAL_BUDGET,
            "identity_is_baseline": True,
        },
        "checks": checks,
        "call_counts": run.get("call_counts"),
        "budget_trace": {
            "ad_fits_used": run.get("ad_fits"),
            "ad_fit_cap": run.get("ad_fit_cap", budget.cap),
            "llm_calls": run.get("llm_calls", 0),
            "llm_cap": FIXTURE_LLM_BUDGET,
            "llm_ledger": run.get("llm_ledger"),
        },
        "cell": cell or None,
        "readings": run.get("readings"),
        "label_wall": audit,
        "stopped": run.get("stopped"),
        "stop_reason": run.get("stop_reason"),
        "evaluate_released_untouched": bool(
            plan.get("evaluate_released")) is False,
        "verdict": {"verdict": verdict_name, "reason": reason},
    }
    OUT_LIFECYCLE_FIXTURE.write_text(_json_text(payload), encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
    print("wrote", OUT_LIFECYCLE_FIXTURE, flush=True)
    return 0 if verdict_name == "LIFECYCLE_FIXTURE_CLOSED" else 1


# =========================================================================== #
# report
# =========================================================================== #
def _ambiguities(*, row_order_contract: bool = False) -> list[str]:
    if row_order_contract:
        return [
            "The v2 contract keeps every row NAB shipped, including the four "
            "files whose timestamp column is not monotonic. Nothing is "
            "sorted, de-duplicated or resampled, and row counts and value "
            "bytes are verified identical before and after the read. What "
            "this costs is that a duplicated stamp now maps two rows into "
            "the same truth event rather than one; that is reported per "
            "series rather than hidden.",
            "The Query's trailing 19 points are read from the raw series, "
            "ruled correct by the #42a book: this is a training-data utility "
            "experiment, and the Program must not reach the inference input "
            "through a Query feature prefix.",
            "background_alarm_rate stays as implemented: the denominator is "
            "the scorable points lying outside every official window.",
            "The Episode bank writes one Episode per (cohort, round, "
            "program), not one per series: the Consumer's primary reading is "
            "a macro average over the cohort, and the per-series vector rides "
            "inside the same Episode as the harm evidence. identity is "
            "written too, as ABSTAIN.",
            "C3 now reads delayed_relation only. Support relations are kept "
            "and the Support-to-delayed flips are reported, but they cannot "
            "vote in the three-kind gate.",
        ]
    return [
        "C3's three-kind gate does not say which lifecycle layer it reads. "
        "It is evaluated as 'either the Support or the delayed layer of a "
        "cell carries this relation', and both layers are reported "
        "separately so the main line can see which one supplied each kind.",
        "The Query's trailing 19 points are read from the raw series, never "
        "from the prepared block. The book says the Query is never "
        "processed; taking the trailing window from the prepared block would "
        "have let the program reach the query features through the back door. "
        "This follows the canon t1b already set for trailing geometry.",
        "The Episode bank writes one Episode per (cohort, round, program), "
        "not one per series: the Consumer's primary reading is a macro "
        "average over the cohort, and the per-series vector rides inside the "
        "same Episode as the harm evidence. identity is written too, as "
        "ABSTAIN, so the bank carries the do-nothing baseline the card "
        "channel needs.",
        "The Episode's final relation is taken from the delayed layer "
        "(evidence_level DELAYED), matching online_loop's own semantics; the "
        "Support-layer classification is kept alongside it rather than "
        "discarded.",
        "background_alarm_rate is defined here as the share of scored points "
        "outside every truth window that were flagged. The book named the "
        "metric without fixing the denominator.",
        "NAB's official window bounds are timestamps that do not always fall "
        "on a sample. When a bound does not match a sample exactly, the "
        "window is taken as the enclosing sample positions; windows that "
        "enclose no sample are dropped. This affects Source only under the "
        "wall.",
    ]


def _render_md(doc: Mapping[str, Any]) -> str:
    verdict = doc["verdict"]
    lines: list[str] = [
        "# T6 -- natural A5 vs A3, frozen plan (%s)" % PROTOCOL_VERSION,
        "",
        "Evidence grade: **%s / %s**." % (doc["exposure"]["evidence_grade"],
                                          doc["exposure"]["evidence_standing"]),
        "",
        "## Verdict",
        "",
        "**%s**" % verdict["verdict"],
        "",
        verdict["reason"],
        "",
        "## Part 0",
        "",
        "- HEAD `%s` -- %s" % (doc["part_0"]["head_commit"],
                               doc["part_0"]["head_subject"]),
        "- Part 0 action: %s" % doc["part_0"]["action"],
        "",
        "## Exposure",
        "",
        "| surface | context | outcome |",
        "| --- | --- | --- |",
        "| Source | %s | %s |" % (doc["exposure"]["source"]["context"],
                                  doc["exposure"]["source"]["outcome"]),
        "| Target | %s | %s |" % (doc["exposure"]["target"]["context"],
                                  doc["exposure"]["target"]["outcome"]),
        "",
        "Aggregate disclosure: %s" % doc["exposure"]["aggregate_disclosure"],
        "",
        "## Part A -- data and shape gate",
        "",
        "- upstream %s @ `%s` (tag %s)" % (doc["data_reference"]["source_url"],
                                           doc["data_reference"]["upstream_commit"][:12],
                                           doc["data_reference"]["tag"]),
    ]
    gate_rows = doc.get("shape_gate") or []
    if gate_rows:
        lines += [
            "- files gated: %d, all passing: %s"
            % (len(gate_rows), all(r["ok"] for r in gate_rows)),
            "",
            "| role | cohort | file | n | gate |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in gate_rows:
            lines.append("| %s | %s | `%s` | %s | %s |" % (
                row["role"], row.get("cohort") or "-", row["file"],
                row.get("length"),
                "pass" if row["ok"] else "FAIL: %s" % ", ".join(
                    row["failures"])))
        lines.append("")
    else:
        lines += ["- the shape gate did not run", ""]
    if doc.get("source_bank"):
        lines += ["## Part C -- Source Experience bank", "",
                  "| cell | support | delayed | agg gain (delayed) | harmed |",
                  "| --- | --- | --- | --- | --- |"]
        for row in doc["source_bank"]["rows"]:
            lines.append("| %s/%s/%s | %s | %s | %s | %s of %s |" % (
                row["cohort"].replace("source_", ""), row["round"],
                row["program"], row["support_relation"],
                row["delayed_relation"],
                "n/a" if row["delayed_aggregate_gain"] is None
                else "%+.4f" % row["delayed_aggregate_gain"],
                row["delayed_harmed"], row["series_read"]))
        lines += ["", "### Readiness (C3)", ""]
        for check in doc["source_bank"]["readiness"]["checks"]:
            lines.append("- [%s] `%s` -- %s" % (
                "x" if check["ok"] else " ", check["id"], check["detail"]))
        lines.append("")
    protocol = doc.get("frozen_protocol")
    if protocol:
        lines += ["## Frozen evaluate protocol", "",
                  "Released: **%s**" % protocol["released"], "",
                  protocol["release_rule"], "",
                  "- backend: `%s` @ `%s`" % (EVALUATE_BACKEND["model"],
                                              EVALUATE_BACKEND["base_url"]),
                  "- budgets: LLM <= %d, AD fits <= %d, forecasting retrains %d"
                  % (EVALUATE_LLM_BUDGET, EVALUATE_AD_FIT_BUDGET,
                     EVALUATE_FORECAST_RETRAIN_BUDGET), ""]
        for cohort, seq in protocol["order_counterbalanced"].items():
            lines.append("- %s: %s" % (cohort, " -> ".join(seq)))
    else:
        lines += [
            "## Frozen evaluate protocol", "",
            "Not written: the run stopped at a Part A first-fault, and a "
            "protocol frozen on a surface that failed its own shape gate "
            "would be frozen around a defect.", ""]
    lines += ["", "## Ambiguities (reported, not self-adjudicated)", ""]
    for item in doc["ambiguities"]:
        lines.append("- %s" % item)
    lines.append("")
    return "\n".join(lines)


def _label_mapping_audit(bank: Mapping[str, Any]) -> dict[str, Any]:
    """Part B: no official window is allowed to vanish quietly."""
    rows = bank.get("label_mapping") or {}
    unmapped = {k: v for k, v in rows.items()
                if int(v["unmapped_window_count"]) > 0}
    return {
        "per_series": rows,
        "official_windows_total": sum(
            int(v["official_window_count"]) for v in rows.values()),
        "mapped_total": sum(int(v["mapped_window_count"])
                            for v in rows.values()),
        "unmapped_total": sum(int(v["unmapped_window_count"])
                              for v in rows.values()),
        "series_with_unmapped_windows": unmapped,
        "rule": ("each row is tested independently for membership in each "
                 "official window; rows sharing a timestamp inside a window "
                 "belong to the same truth event"),
    }


def _write(doc: Mapping[str, Any], *, row_order_contract: bool = False) -> int:
    E2.mkdir(parents=True, exist_ok=True)
    out_json = OUT_JSON_V2 if row_order_contract else OUT_JSON
    out_md = OUT_MD_V2 if row_order_contract else OUT_MD
    out_json.write_text(_json_text(doc), encoding="utf-8")
    out_md.write_text(_render_md(doc), encoding="utf-8")
    print(json.dumps({"verdict": doc["verdict"]["verdict"],
                      "reason": doc["verdict"]["reason"][:220]},
                     ensure_ascii=False, indent=1))
    print("wrote", out_json, flush=True)
    return 0


# =========================================================================== #
# --plan
# =========================================================================== #
def _part0() -> dict[str, Any]:
    """N1: idempotent.  Verify when the closeout is already in; commit only
    if there is still an uncommitted diff.  Never an empty commit."""
    porcelain = _git("status", "--porcelain", "-uno")
    tracked = [line[3:].strip() for line in porcelain.splitlines()
               if line.strip()]
    expected = (
        "evaluation/functional/run_e2_operational_pipeline.py",
        "methods/ttha/method.py",
        "evaluation/functional/task_episode_harness/e1.py",
        "evaluation/functional/task_episode_harness/skill_evolution.py",
        "evaluation/functional/task_episode_harness/e0b.py",
        "tests/functional/test_g1_proposal_guidance.py",
    )
    head = _git("rev-parse", "--short", "HEAD")
    committed = {
        path: _git("log", "--oneline", "-1", "--", path).split(" ")[0]
        for path in expected
    }
    return {
        "head_commit": head,
        "head_subject": _git("log", "-1", "--pretty=%s"),
        "action": ("verified only -- the #41b-lite closeout was already "
                   "committed and the tree carries no uncommitted diff"
                   if not tracked else
                   "uncommitted tracked diff present: %s" % tracked),
        "tracked_modified": tracked,
        "expected_files_last_touched_by": committed,
        "untracked_tests_excluded": [
            "tests/functional/test_e1_v2_protocol_repair.py",
            "tests/functional/test_skill_evolution_e0.py",
            "tests/functional/test_skill_revocation.py",
        ],
        "mkl_savgol_crash": (
            "tests/functional/test_f1_forecast_pilot.py aborts natively in "
            "scipy savgol_filter -> linalg.lstsq; reproduced identically at "
            "the pre-#41 checkpoint, so it is carried on the ledger and not "
            "repaired here"),
    }


def plan(*, row_order_contract: bool = False) -> int:
    before = _freeze()
    doc: dict[str, Any] = {
        "protocol_version": (PROTOCOL_VERSION_V2 if row_order_contract
                             else PROTOCOL_VERSION),
        "entry": "--plan-v2" if row_order_contract else "--plan",
        "exposure": _exposure_labels(),
        "part_0": _part0(),
        "data_reference": _data_reference(),
        "budgets_this_round": {
            "llm": PLAN_LLM_BUDGET,
            "forecasting_retrains": PLAN_FORECAST_RETRAIN_BUDGET,
            "ad_consumer_fits_cap": PLAN_AD_FIT_BUDGET,
        },
        "ambiguities": _ambiguities(row_order_contract=row_order_contract),
    }
    if row_order_contract:
        doc["row_order_contract"] = dict(ROW_ORDER_CONTRACT)
        doc["supersedes"] = {
            "artifact": _repo_rel(OUT_JSON),
            "kept_as": ("the v1 NATURAL_DATA_SHAPE_INELIGIBLE artifact is "
                        "retained unchanged as the shape-contract diagnostic "
                        "that produced this contract"),
        }
    wall = LabelWall(released=False)
    try:
        consumer = _load_consumer()
        doc["consumer_spec"] = consumer.spec()
        gate = gate_all(row_order_contract=row_order_contract)
        doc["shape_gate"] = gate["rows"]
        doc["shape_gate_summary"] = {
            k: v for k, v in gate.items() if k not in ("rows", "reads")}
        universe = _load_universe(gate)
        doc["cohorts"] = {
            "source": {c: sorted(rows) for c, rows in universe["source"].items()},
            "target": {c: sorted(rows) for c, rows in universe["target"].items()},
            "selection_rule": (
                "filename lexicographic order only; never re-ordered by "
                "label, Consumer result or Program headroom"),
        }
        doc["window_plan_fractions"] = _plain(ROUND_FRACTIONS)
        budget = FitBudget(PLAN_AD_FIT_BUDGET)
        bank = _build_source_bank(consumer=consumer, universe=universe,
                                  wall=wall, budget=budget)
        runtime = _write_bank_through_runtime(bank["episodes"])
        readiness = _readiness(bank["rows"], bank["cells"])
        recheck = _determinism_recheck(consumer=consumer, universe=universe,
                                       wall=wall, budget=budget, bank=bank)
        doc["source_bank"] = {
            "task_consumer_key": bank["task_consumer_key"],
            "task_spec": bank["task_spec"],
            "rows": bank["rows"],
            "cells": bank["cells"],
            "written_through_runtime": {
                k: v for k, v in runtime.items() if k != "to_dict"},
            "episodes_to_dict": runtime["to_dict"],
            "readiness": readiness,
            "determinism_recheck": recheck,
        }
        doc["frozen_protocol"] = _frozen_protocol(bank, universe)
        doc["evaluate_released"] = False
        doc["cost"] = {
            "llm_calls": 0,
            "forecasting_retrains": 0,
            "ad_consumer_fits": budget.used,
            "ad_consumer_fit_cap": PLAN_AD_FIT_BUDGET,
            "enumeration_fits": budget.used - sum(
                c["series_refit"] for c in recheck["cells"]),
            "recheck_fits": sum(c["series_refit"] for c in recheck["cells"]),
        }
        doc["label_wall"] = wall.audit()
        doc["label_mapping_audit"] = _label_mapping_audit(bank)
        if not readiness["all_passed"]:
            doc["verdict"] = {
                "verdict": readiness["first_fault"]
                or "SOURCE_EVIDENCE_DIVERSITY_INSUFFICIENT",
                "gate_layer": readiness["gate_layer"],
                "reason": "the Source bank readiness gate did not pass: %s"
                          % [c["id"] for c in readiness["checks"]
                             if not c["ok"]],
            }
        elif not recheck["all_identical"]:
            doc["verdict"] = {
                "verdict": "NATURAL_AD_CONSUMER_UNREADABLE",
                "reason": ("a rechecked cell did not reproduce its readings; "
                           "a Consumer that does not repeat cannot carry a "
                           "comparison"),
                "recheck": recheck["cells"],
            }
        else:
            doc["verdict"] = {
                "verdict": "T6_NATURAL_PLAN_READY",
                "reason": (
                    "the natural A5-vs-A3 comparison is frozen without any "
                    "Target outcome being read, and the natural Source "
                    "carries all three kinds of Action-Response the "
                    "comparison depends on.  NATURAL / provisional: one "
                    "Target domain, two cohorts, awaiting replication on an "
                    "independent dataset; the frozen --evaluate path stays "
                    "shut until the main line releases it."),
            }
    except Stop as stop:
        doc["verdict"] = {"verdict": stop.verdict, "reason": stop.reason,
                          **stop.extra}
        doc["label_wall"] = wall.audit()
        doc.setdefault("cost", {
            "llm_calls": 0,
            "forecasting_retrains": 0,
            "ad_consumer_fits": 0,
            "ad_consumer_fit_cap": PLAN_AD_FIT_BUDGET,
            "note": ("the round stopped at a Part A first-fault, so Parts B "
                     "and C were not exercised and no Consumer fit was spent"),
        })
        doc["evaluate_released"] = False
        if stop.verdict == "NATURAL_DATA_SHAPE_INELIGIBLE":
            doc["remedies_not_taken"] = _remedies_not_taken()
    doc["frozen_surface"] = {
        "surface": "V10", "members": len(set(FROZEN_SURFACE_V10)),
        "after": _verify(before),
    }
    doc["artifact_sha256"] = canonical_sha256(_plain(
        {k: v for k, v in doc.items() if k != "artifact_sha256"}))
    return _write(doc, row_order_contract=row_order_contract)


def _materialize_skill_v2_snapshot():
    """Compile h0 + the frozen v2 entry.  No Slow, no Temp h0s_v2."""
    from SelfEvolvingHarnessTS.contracts.harness import (
        EditManifest, EditOperation,
    )
    from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
        EditController, FaultRouter, SurfaceRegistry,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (
        _resolve_apply_manifest,
    )

    if not OUT_SKILL_V2.exists():
        raise Stop("V2_DELIVERY_FAILED",
                   "missing frozen skill artifact %s" % _repo_rel(OUT_SKILL_V2))
    doc = json.loads(OUT_SKILL_V2.read_text(encoding="utf-8"))
    entry = dict(doc.get("entry") or {})
    if entry.get("skill_id") != REPLAY_V2_SKILL_ID:
        raise Stop("V2_DELIVERY_FAILED",
                   "frozen entry skill_id is %r" % entry.get("skill_id"))
    h0 = compile_snapshot(
        PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
        verify_lock=False)
    store_root = Path(tempfile.gettempdir()) / "t6e0_h0_plus_v2"
    if store_root.exists():
        shutil.rmtree(store_root)
    store = SnapshotStore(store_root / "snapshots")
    store.materialize(h0)
    store.set_active(h0.runtime_bundle_sha)
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter())
    manifest = EditManifest(
        edit_id=REPLAY_V2_SKILL_ID,
        base_harness_sha=h0.harness_content_sha,
        target_pattern_id="t6-42e0-replay-ad-skill-v2",
        target_surface_id="skill_library.entries/" + REPLAY_V2_SKILL_ID,
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value=entry,
        observable_applicability=dict(entry.get("observable_applicability")
                                      or {}),
        predicted_agent_behavior_change=(
            "retrieve_skill:" + REPLAY_V2_SKILL_ID,),
        predicted_data_effect=("safer_proposal_stage",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",),
        patch_id=None,
    )
    parent = store.materialize(h0)
    receipt = controller.apply_to_fork(
        parent,
        _resolve_apply_manifest(manifest, h0),
        confirmed_cause="SKILL_LIBRARY_GAP",
    )
    snapshot = receipt.candidate_snapshot.snapshot
    store.set_active(snapshot.runtime_bundle_sha)
    ids = [s.skill_id for s in snapshot.skills]
    if REPLAY_V2_SKILL_ID not in ids:
        raise Stop("V2_DELIVERY_FAILED",
                   "materialized snapshot missing %s" % REPLAY_V2_SKILL_ID)
    return h0, snapshot


def _replay_v2_verdict(run: Mapping[str, Any]) -> dict[str, Any]:
    cells = [c for c in (run.get("cells") or []) if "error" not in c]
    a3 = [c for c in cells if c.get("arm") == "A3"]
    a5 = [c for c in cells if c.get("arm") == "A5"]
    a5_miss = [c["cell"] for c in a5
               if REPLAY_V2_SKILL_ID not in (c.get("retrieved_skill_ids") or [])]
    a3_leak = [c["cell"] for c in a3
               if REPLAY_V2_SKILL_ID in (c.get("retrieved_skill_ids") or [])]
    source_memory = [
        c["cell"] for c in cells
        if int((c.get("retrieval_before_round") or {}).get("held") or 0) > 0
        or (c.get("retrieval_before_round") or {}).get("source_cards")
    ]
    if a5_miss or a3_leak or source_memory or run.get("stopped") == "V2_DELIVERY_FAILED":
        return {"verdict": "V2_DELIVERY_FAILED",
                "a5_miss": a5_miss, "a3_leak": a3_leak,
                "source_memory_cells": source_memory}

    def _sum(rows, key):
        return sum(int(c.get(key) or 0) for c in rows)

    a3_hampel = _sum(a3, "hampel_probed") + sum(
        1 for c in a3 if c.get("hampel_proposed") or c.get("hampel_chosen"))
    a5_hampel = _sum(a5, "hampel_probed") + sum(
        1 for c in a5 if c.get("hampel_proposed") or c.get("hampel_chosen"))
    a3_hampel_probed = _sum(a3, "hampel_probed")
    a5_hampel_probed = _sum(a5, "hampel_probed")
    a3_non = _sum(a3, "non_hampel_probe_count") + _sum(a3, "non_hampel_proposal_count")
    a5_non = _sum(a5, "non_hampel_probe_count") + _sum(a5, "non_hampel_proposal_count")
    a3_non_probe = _sum(a3, "non_hampel_probe_count")
    a5_non_probe = _sum(a5, "non_hampel_probe_count")
    a3_any_non_hampel = a3_non > 0
    a5_any_non_hampel = a5_non > 0
    a3_any_plan = any(
        (c.get("non_identity_trials") or 0) > 0 or (c.get("pool") or []) != ["identity"]
        for c in a3)
    a5_any_plan = any(
        (c.get("non_identity_trials") or 0) > 0 or any(
            x and x != "identity" for x in (c.get("pool") or []))
        for c in a5)

    if a3_any_non_hampel and not a5_any_non_hampel:
        verdict = "V2_GLOBAL_EXPLORATION_COLLAPSE"
    elif ((a3_hampel > 0) and (a5_hampel < a3_hampel or a5_hampel_probed < a3_hampel_probed)
          and a5_non_probe > 0):
        verdict = "V2_SCOPED_RISK_BEHAVIOR_OBSERVED"
    elif a3_hampel == 0 and a5_hampel == 0 and a5_any_non_hampel:
        verdict = "RISK_SKILL_NO_TRIGGERING_CANDIDATE"
    elif not a3_any_plan and not a5_any_plan:
        verdict = "V2_REPLAY_NO_ACTIONABLE_PLAN_SAMPLE"
    elif a5_hampel > a3_hampel or a5_hampel_probed > a3_hampel_probed:
        verdict = "V2_RISK_GUIDANCE_REGRESSION"
    else:
        verdict = "V2_RISK_GUIDANCE_IGNORED"
    return {
        "verdict": verdict,
        "a3_hampel_events": a3_hampel,
        "a5_hampel_events": a5_hampel,
        "a3_hampel_probed": a3_hampel_probed,
        "a5_hampel_probed": a5_hampel_probed,
        "a3_non_hampel": a3_non,
        "a5_non_hampel": a5_non,
        "a3_non_hampel_probes": a3_non_probe,
        "a5_non_hampel_probes": a5_non_probe,
    }


def replay_skill_v2() -> int:
    """One-shot development replay of frozen source_investigation_ad_v2."""
    if not OUT_JSON_V2.exists():
        print(json.dumps({"verdict": "V2_DELIVERY_FAILED",
                          "reason": "missing v2 plan"}, indent=1))
        return 1
    leftover = [
        line for line in os.popen("ps -ef").read().splitlines()
        if "run_e2_t6_natural_a5_a3.py" in line
        and "--replay-skill-v2" in line
        and str(os.getpid()) not in line
    ]
    if leftover:
        print(json.dumps({"verdict": "CONCURRENT_RUN_BLOCKED",
                          "reason": leftover}, indent=1))
        return 2
    REPLAY_V2_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(str(REPLAY_V2_LOCK),
                          os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(json.dumps({"verdict": "CONCURRENT_RUN_BLOCKED",
                          "reason": "lock held"}, indent=1))
        return 2
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    plan = json.loads(OUT_JSON_V2.read_text(encoding="utf-8"))
    try:
        h0, h0_plus_v2 = _materialize_skill_v2_snapshot()
    except Stop as stop:
        payload = {"verdict": stop.verdict, "reason": stop.reason,
                   "run_id": run_id}
        OUT_REPLAY_V2.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 1
    wall = LabelWall(released=True)
    universe = _load_universe(gate_all(row_order_contract=True))
    budget = FitBudget(REPLAY_V2_AD_FIT_BUDGET)
    store_tag = "t6e0_%s" % run_id
    try:
        run = _run_cells(
            plan=plan,
            cohort_rows=universe["target"],
            agent_factory=_evaluate_agent,
            backend_factory=_evaluate_backend,
            llm_budget=REPLAY_V2_LLM_BUDGET,
            fit_budget=budget,
            wall=wall,
            store_tag=store_tag,
            snapshot_for_arm={"A3": h0, "A5": h0_plus_v2},
        )
    except Stop as stop:
        payload = {"verdict": stop.verdict, "reason": stop.reason,
                   "run_id": run_id}
        OUT_REPLAY_V2.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 1
    run["label_wall"] = wall.audit()
    run["run_id"] = run_id
    run["h0_runtime_bundle_sha"] = h0.runtime_bundle_sha
    run["h0_plus_v2_runtime_bundle_sha"] = h0_plus_v2.runtime_bundle_sha
    run["construction_memory_empty"] = True
    run["v2_source"] = _repo_rel(OUT_SKILL_V2)
    verdict = _replay_v2_verdict(run)
    payload = {
        "protocol_version": "t6_nab_42e0_skill_v2_replay_v1",
        "entry": "--replay-skill-v2",
        "evidence_grade": "DEVELOPMENT",
        "evidence_standing": "same-context",
        "counts_as_capability_claim": False,
        "counts_as_cross_domain_claim": False,
        "run": run,
        "verdict": verdict,
    }
    OUT_REPLAY_V2.write_text(_json_text(payload), encoding="utf-8")
    lines = [
        "# #42e0 Skill v2 development replay",
        "",
        "verdict: **%s**" % verdict["verdict"],
        "evidence_grade: DEVELOPMENT / same-context",
        "run_id: %s" % run_id,
        "LLM %s / %s; AD fit %s / %s" % (
            run.get("llm_calls"), run.get("llm_budget"),
            run.get("ad_fits"), run.get("ad_fit_cap")),
        "",
        "| cell | retrieved v2 | pool | chosen | hampel p/c/pr | non-hampel prop/probe | relation |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell in run.get("cells") or []:
        ids = cell.get("retrieved_skill_ids") or []
        rels = [(e.get("workflow_signature"), e.get("relation"),
                 e.get("local_status")) for e in cell.get("episode_rows") or []]
        lines.append(
            "| %s | %s | %s | %s | %s/%s/%s | %s/%s | %s |" % (
                cell.get("cell"),
                REPLAY_V2_SKILL_ID in ids,
                cell.get("pool"),
                cell.get("chosen"),
                cell.get("hampel_proposed"),
                cell.get("hampel_chosen"),
                cell.get("hampel_probed"),
                cell.get("non_hampel_proposal_count"),
                cell.get("non_hampel_probe_count"),
                rels or "—",
            ))
    lines.extend([
        "",
        "A3 hampel events %s / probed %s; A5 hampel events %s / probed %s"
        % (verdict.get("a3_hampel_events"), verdict.get("a3_hampel_probed"),
           verdict.get("a5_hampel_events"), verdict.get("a5_hampel_probed")),
        "A3 non-hampel %s; A5 non-hampel %s"
        % (verdict.get("a3_non_hampel"), verdict.get("a5_non_hampel")),
    ])
    OUT_REPLAY_V2_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": verdict["verdict"],
        "llm_calls": run.get("llm_calls"),
        "ad_fits": run.get("ad_fits"),
        "hampel": {"A3": verdict.get("a3_hampel_probed"),
                   "A5": verdict.get("a5_hampel_probed")},
        "non_hampel": {"A3": verdict.get("a3_non_hampel_probes"),
                       "A5": verdict.get("a5_non_hampel_probes")},
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_REPLAY_V2)
    print("wrote", OUT_REPLAY_V2_MD)
    return 0 if verdict["verdict"] != "V2_DELIVERY_FAILED" else 1


def _fetch_expansion_files() -> dict[str, Any]:
    """Pinned NAB commit only.  Missing any of the 17 → DATA_MISSING."""
    import urllib.error
    import urllib.request

    expected = []
    for cohort, (directory, names) in EXPANSION_COHORTS.items():
        for name in names:
            expected.append((cohort, directory, name))
    exposure = {
        "before_download_context": "AGGREGATE_SEEN",
        "after_value_load_context": "INSTANCE_SEEN",
        "after_source_label_use_outcome": "OPENED_AS_SOURCE",
        "never_fresh_or_virgin_target": True,
        "upstream_commit": NAB_COMMIT,
    }
    fetched: list[dict[str, Any]] = []
    missing: list[str] = []
    for cohort, directory, name in expected:
        dest = DATA_ROOT / directory / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = "%s/data/%s/%s" % (NAB_RAW_BASE, directory, name)
        status = "present"
        if not dest.is_file() or dest.stat().st_size <= 0:
            try:
                urllib.request.urlretrieve(url, dest)
                status = "downloaded"
            except (urllib.error.URLError, OSError) as exc:
                missing.append("%s/%s:%s" % (directory, name, type(exc).__name__))
                status = "missing"
                if dest.exists():
                    dest.unlink()
        fetched.append({
            "cohort": cohort, "directory": directory, "file": name,
            "path": _repo_rel(dest), "bytes": (
                dest.stat().st_size if dest.is_file() else 0),
            "status": status, "url": url,
        })
    return {
        "exposure": exposure,
        "expected": 17,
        "fetched": fetched,
        "missing": missing,
        "ok": not missing and len(fetched) == 17,
    }


def _gate_expansion_files() -> dict[str, Any]:
    """v2 row-order contract on the 17 new Source files only.  No Target."""
    rows: list[dict[str, Any]] = []
    reads: dict[str, Any] = {}
    kept: dict[str, list[Path]] = {}
    dropped: dict[str, list[dict[str, Any]]] = {}
    for cohort, (directory, names) in EXPANSION_COHORTS.items():
        kept[cohort] = []
        dropped[cohort] = []
        for name in names:
            path = DATA_ROOT / directory / name
            read = _read_series(path, row_order_contract=True)
            reads["%s/%s" % (directory, name)] = read
            row = {
                "role": "source", "cohort": cohort, "file": name,
                "nab_key": "%s/%s" % (directory, name),
                "ok": bool(read["ok"] and read.get("rows_preserved")),
                "failures": list(read["failures"] or []),
                "length": read.get("length"),
                "ordering_violation_count": read.get("ordering_violation_count"),
                "duplicate_timestamp_count": read.get("duplicate_timestamp_count"),
                "backward_transition_count": read.get("backward_transition_count"),
                "physical_rows_before": read.get("physical_rows_before"),
                "physical_rows_after": read.get("physical_rows_after"),
                "rows_preserved": read.get("rows_preserved"),
                "values_sha256": read.get("values_sha256"),
            }
            if not row["ok"] or not row["rows_preserved"]:
                if not row["failures"] and not row["rows_preserved"]:
                    row["failures"] = ["row_sequence_not_preserved"]
                dropped[cohort].append(row)
            else:
                kept[cohort].append(path)
            rows.append(row)
    usable: dict[str, list[Path]] = {}
    abandoned: list[dict[str, Any]] = []
    for cohort, paths in kept.items():
        if len(paths) < 4:
            abandoned.append({
                "cohort": cohort,
                "kept": [p.name for p in paths],
                "dropped": dropped[cohort],
                "reason": "kept %d files, need >=4" % len(paths),
            })
        else:
            usable[cohort] = paths
    return {
        "rows": rows,
        "reads": reads,
        "kept": {c: [p.name for p in ps] for c, ps in kept.items()},
        "dropped": dropped,
        "usable": {c: [p.name for p in ps] for c, ps in usable.items()},
        "abandoned_cohorts": abandoned,
        "all_new_cohorts_usable": (
            set(usable) == set(EXPANSION_COHORTS) and not abandoned),
    }


def _expansion_universe(gate: Mapping[str, Any]) -> dict[str, Any]:
    reads = gate["reads"]
    source: dict[str, Any] = {}
    for cohort, (directory, _names) in EXPANSION_COHORTS.items():
        if cohort not in gate["usable"]:
            continue
        source[cohort] = {}
        for name in gate["usable"][cohort]:
            key = "%s/%s" % (directory, name)
            read = reads[key]
            source[cohort][name] = {
                "values": read["values"],
                "timestamps": read["timestamps"],
                "length": read["length"],
                "nab_key": key,
                "windows": _window_plan(read["length"]),
            }
    return {"source": source, "target": {}, "gate": gate["rows"]}


def _v3_proxy_audit(
    episodes: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from run_e2_t6_42d_consolidation import (
        BOOLEAN_FEATURES, FORBIDDEN_SCOPE_FEATURES, _cohort_of,
    )

    out: list[dict[str, Any]] = []
    for feature in BOOLEAN_FEATURES:
        by_value: dict[str, set[str]] = {}
        by_cohort: dict[str, set[str]] = {}
        for episode in episodes:
            eid = str(episode["episode_id"])
            row = rows_by_id.get(eid)
            cohort = _cohort_of(episode, row)
            value = ((episode.get("context_summary") or {})
                     .get("local_pattern") or {}).get(feature)
            by_value.setdefault(str(value), set()).add(cohort)
            by_cohort.setdefault(cohort, set()).add(str(value))
        values = {k: sorted(v) for k, v in sorted(by_value.items())}
        sides = {
            key: by_value.get(key, set())
            for key in ("True", "False")
        }
        single_indicator = any(len(cs) == 1 for cs in by_value.values())
        complete_partition = (
            all(len(cs) == 1 for cs in by_value.values())
            and all(len(vs) == 1 for vs in by_cohort.values())
            and len(by_value) == len(by_cohort)
        )
        both_present = bool(sides["True"]) and bool(sides["False"])
        both_ge2 = all(len(cs) >= 2 for cs in sides.values())
        constant = len(by_value) == 1
        forbidden = feature in FORBIDDEN_SCOPE_FEATURES
        usable = (
            not forbidden and not constant and not single_indicator
            and not complete_partition and both_present and both_ge2
        )
        out.append({
            "feature": feature,
            "values_to_cohorts": values,
            "constant": constant,
            "single_cohort_indicator": single_indicator,
            "complete_cohort_partition_replica": complete_partition,
            "both_boolean_sides_present": both_present,
            "both_boolean_sides_ge2_cohorts": both_ge2,
            "forbidden": forbidden,
            "usable_as_scope": usable,
            "note": (
                "pss forbidden" if forbidden else
                "no resolving power" if constant else
                "single-cohort indicator; proxy" if single_indicator else
                "complete cohort partition replica; proxy"
                if complete_partition else
                "boolean sides do not each cover >=2 cohorts"
                if not (both_present and both_ge2) else
                "legal boolean scope"
            ),
        })
    return out


def _v3_authorize(
    episodes: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    proxy: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from run_e2_t6_42d_consolidation import (
        BOOLEAN_FEATURES, IDENTITY, MIN_DISTINCT_COHORTS,
        _cohort_of, _delayed_relation, _vote_bucket,
    )

    rows_by_id = {str(r["episode_id"]): r for r in rows}

    def _bags(subset: Sequence[Mapping[str, Any]]
              ) -> dict[str, dict[str, set[str]]]:
        pool: dict[str, dict[str, set[str]]] = {}
        for episode in subset:
            eid = str(episode["episode_id"])
            row = rows_by_id.get(eid, {})
            program = str(row.get("program") or episode.get("workflow_signature"))
            relation = _delayed_relation(episode, row)
            bucket = _vote_bucket(relation, program)
            if bucket is None:
                continue
            cell = pool.setdefault(program, {
                "positive": set(), "negative": set(), "conflict": set(),
                "immaterial": set(),
            })
            cell[bucket].add(_cohort_of(episode, row))
        return pool

    def _decide(program: str, bags: Mapping[str, set[str]]) -> str | None:
        if program == IDENTITY:
            return None
        pos = bags["positive"]
        harm = bags["negative"] | bags["conflict"]
        if len(pos) >= MIN_DISTINCT_COHORTS and not harm:
            return "TRY"
        if len(harm) >= MIN_DISTINCT_COHORTS and not pos:
            return "RISK"
        return None

    unconditional = _bags(list(episodes))
    signed = []
    for program, bags in sorted(unconditional.items()):
        signed.append({
            "scope": "unconditional_4_cohort_pool",
            "program": program,
            "positive_cohorts": sorted(bags["positive"]),
            "negative_cohorts": sorted(bags["negative"]),
            "conflict_cohorts": sorted(bags["conflict"]),
            "immaterial_cohorts": sorted(bags["immaterial"]),
            "strict_harm_cohorts": sorted(bags["negative"]),
            "extended_harm_cohorts": sorted(
                bags["negative"] | bags["conflict"]),
            "authorization": _decide(program, bags),
        })

    legal = [p["feature"] for p in proxy if p["usable_as_scope"]]
    scoped_rows: list[dict[str, Any]] = []
    for feature in legal:
        for value in (True, False):
            subset = [
                ep for ep in episodes
                if bool(((ep.get("context_summary") or {})
                         .get("local_pattern") or {}).get(feature)) is value
            ]
            bags = _bags(subset)
            for program, cell in sorted(bags.items()):
                scoped_rows.append({
                    "scope": "%s==%s" % (feature, value),
                    "feature": feature,
                    "value": value,
                    "program": program,
                    "positive_cohorts": sorted(cell["positive"]),
                    "negative_cohorts": sorted(cell["negative"]),
                    "conflict_cohorts": sorted(cell["conflict"]),
                    "immaterial_cohorts": sorted(cell["immaterial"]),
                    "authorization": _decide(program, cell),
                    "note": "votes stay inside this Scope cell; no stitching",
                })

    try_ops: list[str] = []
    risk_ops: list[str] = []
    sources: dict[str, list[str]] = {}
    for row in signed + scoped_rows:
        program = row["program"]
        kind = row["authorization"]
        if kind == "TRY" and program not in try_ops:
            try_ops.append(program)
            sources.setdefault(program, []).append(row["scope"])
        elif kind == "RISK" and program not in risk_ops and program not in try_ops:
            risk_ops.append(program)
            sources.setdefault(program, []).append(row["scope"])
    # a program authorized as TRY in one legal cell is not also RISK
    risk_ops = [p for p in risk_ops if p not in try_ops]
    return {
        "min_distinct_cohorts": MIN_DISTINCT_COHORTS,
        "same_scope_required": True,
        "harm_definition": "delayed_relation in {NEGATIVE, CONFLICT}",
        "harm_definition_strict": "delayed_relation == NEGATIVE",
        "legal_scope_features": legal,
        "used_unconditional_pool": True,
        "try_authorized": try_ops,
        "risk_authorized": risk_ops,
        "authorization_scopes": sources,
        "signed_summary_unconditional": signed,
        "signed_summary_legal_scope_cells": scoped_rows,
    }


def _dev_proposal_frequency() -> dict[str, Any]:
    """#42d 8-cell + #42e0 8-cell proposal mentions.  Not new evidence."""
    counts = {name: 0 for name in (
        "hampel_filter", "outlier_mad", "outlier_iqr", "winsorize")}
    cells_read = 0
    for path in (E2 / "t6_nab_42d_paired_replay.json", OUT_REPLAY_V2):
        doc = json.loads(path.read_text(encoding="utf-8"))
        cells = (doc.get("run") or {}).get("cells") or doc.get("cells") or []
        for cell in cells:
            cells_read += 1
            pool = [str(x) for x in (cell.get("pool") or [])]
            blob = " ".join(pool).lower()
            if "hampel" in blob:
                counts["hampel_filter"] += 1
            if "mad" in blob:
                counts["outlier_mad"] += 1
            if "iqr" in blob:
                counts["outlier_iqr"] += 1
            if "winsor" in blob:
                counts["winsorize"] += 1
    return {"cells_read": cells_read, "proposal_cell_counts": counts}


def _census_for_v3_audit(authorization: Mapping[str, Any]
                         ) -> list[dict[str, Any]]:
    from run_e2_t6_42d_consolidation import BOOLEAN_FEATURES

    out: list[dict[str, Any]] = []
    for row in authorization["signed_summary_unconditional"]:
        for relation, key in (
            ("POSITIVE", "positive_cohorts"),
            ("NEGATIVE", "negative_cohorts"),
            ("CONFLICT", "conflict_cohorts"),
            ("NEUTRAL", "immaterial_cohorts"),
        ):
            cohorts = list(row[key])
            if not cohorts:
                continue
            item = {
                "canonical_program": [row["program"]],
                "support_relation": relation,
                "distinct_task_count": len(cohorts),
                "distinct_task_episode_ids": cohorts,
            }
            for feature in BOOLEAN_FEATURES:
                item[feature] = True
            out.append(item)
    return out


def _issue_skill_v3(
    *,
    authorization: Mapping[str, Any],
    census_doc: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose frozen primitives.  Does not call issue_v2()."""
    from SelfEvolvingHarnessTS.contracts.observables import OBSERVABLE_FEATURES
    from SelfEvolvingHarnessTS.operators.registry import OPERATOR_NAMES
    from evaluation.functional.task_episode_harness.agentic import (
        ad_source_skill as ad,
    )
    from evaluation.functional.task_episode_harness.e1 import _FastAgentStub
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view

    authorized_try = list(authorization["try_authorized"])
    authorized_risk = list(authorization["risk_authorized"])
    audit_census = _census_for_v3_audit(authorization)
    tokens = list(ad.SOURCE_COHORT_TOKENS) + list(EXPANSION_COHORT_TOKENS)
    payload = {
        "skill_id": SOURCE_SKILL_ID_V3,
        "applicability": dict(ad.SOURCE_APPLICABILITY),
        "authorized_try_operators": authorized_try,
        "risk_authorized_operators": authorized_risk,
        "authorization": authorization,
        "signed_summary": authorization["signed_summary_unconditional"],
        "legal_scope_cells": authorization["signed_summary_legal_scope_cells"],
        "known_limits": census_doc.get("known_limits"),
        "required_sections": list(ad.SECTIONS),
        "try_abstain_literal": ad.TRY_ABSTAIN,
        "v1_status": "superseded",
        "v2_status": "superseded",
        "one_entry_only": True,
        "temporal_rules": (
            "OBSERVE/WHEN proposal-time public Context only; "
            "RISK is the census default deprioritization of each authorized "
            "risk operator and must say a strong public Pattern may still "
            "keep it a restricted probe candidate; VERIFY is the live "
            "two-stage Support POSITIVE then delayed POSITIVE gate; "
            "no distinct-task requirement"
        ),
        "target_domain": (
            "a different domain from the census; write what to observe "
            "and what would have to hold, not what happened in a named cohort"
        ),
    }
    appendix = (
        " Temporal rules for this AD v3 call, in addition to the frozen "
        "containment audit. OBSERVE and WHEN may name only proposal-time "
        "public Context: task_kind and the census observation-feature names. "
        "They must not name support_relation, delayed_relation, approval, "
        "or Skill-status words. RISK is the Source-census default "
        "deprioritization of every authorized risk operator: lower its "
        "proposal priority, but you must say that under strong public "
        "Pattern evidence it may still be a restricted probe candidate. "
        "RISK is not a hard ban. If the authorized risk list names "
        "hampel_filter, that name must appear in RISK. VERIFY must state "
        "the live two-stage gate in words, with no digits: current Target "
        "Support relation POSITIVE forms a Draft; later delayed relation "
        "POSITIVE approves or keeps Active. Do not require distinct tasks "
        "anywhere. If authorized_try_operators is empty, TRY must be exactly "
        + ad.TRY_ABSTAIN + "; otherwise TRY may name only those operators."
    )
    system = ad.slow_system(
        authorized_try, skill_id=SOURCE_SKILL_ID_V3) + appendix
    attempts: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    for attempt in (1, 2):
        try:
            response = ad._slow_call([
                {"role": "system", "content": system},
                {"role": "user",
                 "content": json.dumps(payload, ensure_ascii=False)},
            ])
        except (RuntimeError, ValueError) as exc:
            attempts.append({
                "attempt": attempt,
                "error": "%s: %s" % (type(exc).__name__, exc),
            })
            continue
        decision = str(response.get("decision") or "").upper()
        sections = response.get("sections")
        row: dict[str, Any] = {
            "attempt": attempt, "decision": decision,
            "slow_response": response,
        }
        if decision == "ABSTAIN":
            row["audit"] = {"pass": True, "reason": "ABSTAIN"}
            attempts.append(row)
            accepted = {"decision": "ABSTAIN", "attempt": attempt,
                        "slow_response": response}
            break
        if decision != "ADD" or not isinstance(sections, Mapping):
            row["audit"] = {"pass": False, "reason": "malformed"}
            attempts.append(row)
            continue
        contain = ad.audit_sections(
            sections, audit_census,
            operator_names=list(OPERATOR_NAMES),
            observable_features=list(OBSERVABLE_FEATURES) + [
                "level_only_post_shift_support_sufficient",
                "post_shift_support_sufficient",
                "period_repair_available",
            ],
            source_cohort_tokens=tokens,
            authorized_try=authorized_try,
        )
        timing = ad.temporal_audit(sections)
        audit = {
            "pass": bool(contain["pass"] and timing["pass"]),
            "containment": contain,
            "temporal": timing,
        }
        row["audit"] = audit
        attempts.append(row)
        if audit["pass"]:
            accepted = {
                "decision": "ADD", "sections": dict(sections),
                "audit": audit, "slow_response": response,
                "attempt": attempt,
            }
            break
    result: dict[str, Any] = {
        "protocol_version": "t6_nab_42e_source_skill_v3",
        "skill_id": SOURCE_SKILL_ID_V3,
        "v1_skill_id": ad.SOURCE_SKILL_ID,
        "v2_skill_id": ad.SOURCE_SKILL_ID_V2,
        "v1_status": "superseded",
        "v2_status": "superseded",
        "v1_not_deleted": True,
        "v2_not_deleted": True,
        "v1_not_in_h0s_v3": True,
        "v2_not_in_h0s_v3": True,
        "authorized_try_operators": authorized_try,
        "risk_authorized_operators": authorized_risk,
        "llm_api_call_count": len(attempts),
        "llm_cap": EXPANSION_LLM_CAP,
        "target_outcome_read": False,
        "counts_as_capability_evidence": False,
        "attempts": attempts,
        "slow_payload": payload,
    }
    if accepted is None:
        result.update({
            "verdict": "SLOW_CONSOLIDATION_UNREADABLE",
            "skill_written": False,
            "reason": "both Slow attempts failed the combined audit",
        })
        return result
    if accepted.get("decision") == "ABSTAIN":
        result.update({
            "verdict": "SLOW_ABSTAIN",
            "skill_written": False,
            "accepted_attempt": accepted["attempt"],
        })
        return result

    entry = ad.build_skill_payload(
        accepted["sections"], skill_id=SOURCE_SKILL_ID_V3)
    h0, snapshot = _materialize_named_skill(SOURCE_SKILL_ID_V3, entry)
    a5_method = TTHAMethod(_FastAgentStub(), snapshot, ())
    a3_method = TTHAMethod(_FastAgentStub(), h0, ())
    a5_view = resolve_harness_view(
        snapshot, {"task_kind": "anomaly_detection"}, role="fast")
    a3_view = resolve_harness_view(
        h0, {"task_kind": "anomaly_detection"}, role="fast")
    a5_ids = [s.skill_id for s in a5_view.skills]
    a3_ids = [s.skill_id for s in a3_view.skills]
    delivery = {
        "a5_retrieves_v3": SOURCE_SKILL_ID_V3 in a5_ids,
        "a3_does_not_retrieve_v3": SOURCE_SKILL_ID_V3 not in a3_ids,
        "a5_memory_empty": list(
            getattr(a5_method, "experience_episodes", ()) or ()) == [],
        "a3_memory_empty": list(
            getattr(a3_method, "experience_episodes", ()) or ()) == [],
        "a5_view_skill_ids": a5_ids,
        "a3_view_skill_ids": a3_ids,
    }
    delivery["pass"] = all((
        delivery["a5_retrieves_v3"],
        delivery["a3_does_not_retrieve_v3"],
        delivery["a5_memory_empty"],
        delivery["a3_memory_empty"],
    ))
    result.update({
        "skill_written": True,
        "entry": entry,
        "sections": accepted["sections"],
        "audit": accepted["audit"],
        "accepted_attempt": accepted["attempt"],
        "h0s_v3_runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "h0_runtime_bundle_sha": h0.runtime_bundle_sha,
        "skill_ids": [s.skill_id for s in snapshot.skills],
        "delivery_assert": delivery,
    })
    return result


def _materialize_named_skill(skill_id: str, entry: Mapping[str, Any]):
    from SelfEvolvingHarnessTS.contracts.harness import (
        EditManifest, EditOperation,
    )
    from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
        EditController, FaultRouter, SurfaceRegistry,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (
        _resolve_apply_manifest,
    )

    h0 = compile_snapshot(
        PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
        verify_lock=False)
    store_root = Path(tempfile.gettempdir()) / ("t6e_%s" % skill_id)
    if store_root.exists():
        shutil.rmtree(store_root)
    store = SnapshotStore(store_root / "snapshots")
    store.materialize(h0)
    store.set_active(h0.runtime_bundle_sha)
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter())
    manifest = EditManifest(
        edit_id=skill_id,
        base_harness_sha=h0.harness_content_sha,
        target_pattern_id="t6-42e-source-derived-ad-skill-v3",
        target_surface_id="skill_library.entries/" + skill_id,
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value=dict(entry),
        observable_applicability=dict(
            entry.get("observable_applicability") or {}),
        predicted_agent_behavior_change=("retrieve_skill:" + skill_id,),
        predicted_data_effect=("safer_proposal_stage",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",),
        patch_id=None,
    )
    parent = store.materialize(h0)
    receipt = controller.apply_to_fork(
        parent,
        _resolve_apply_manifest(manifest, h0),
        confirmed_cause="SKILL_LIBRARY_GAP",
    )
    snapshot = receipt.candidate_snapshot.snapshot
    store.set_active(snapshot.runtime_bundle_sha)
    return h0, snapshot


def _v3_verdict(
    *,
    fetch: Mapping[str, Any],
    gate: Mapping[str, Any],
    authorization: Mapping[str, Any] | None,
    skill: Mapping[str, Any] | None,
    trigger: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not fetch.get("ok"):
        return {"verdict": "DATA_MISSING", "missing": fetch.get("missing")}
    if not gate.get("all_new_cohorts_usable"):
        return {
            "verdict": "SHAPE_GATE_FAILED_COHORT_DROPPED",
            "abandoned_cohorts": gate.get("abandoned_cohorts"),
            "dropped": gate.get("dropped"),
        }
    if authorization is None:
        return {"verdict": "CENSUS_UNREADABLE"}
    try_ops = list(authorization.get("try_authorized") or [])
    risk_ops = list(authorization.get("risk_authorized") or [])
    freq = (trigger or {}).get("proposal_cell_counts") or {}
    triggerable_risk = [
        op for op in risk_ops
        if int(freq.get(op) or 0) > 0 or any(
            token in op for token in TRIGGERABLE_FROM_DEV)
    ]
    untriggerable_only = bool(risk_ops) and not triggerable_risk and all(
        op in UNTRIGGERABLE_FROM_DEV or "hampel" in op for op in risk_ops)
    if skill and skill.get("verdict") == "SLOW_CONSOLIDATION_UNREADABLE":
        return {"verdict": "SLOW_CONSOLIDATION_UNREADABLE",
                "skill": {"llm": skill.get("llm_api_call_count")}}
    if try_ops and skill and skill.get("skill_written"):
        return {
            "verdict": "SOURCE_TRY_SKILL_FROZEN",
            "try_authorized": try_ops,
            "risk_authorized": risk_ops,
            "h0s_v3_runtime_bundle_sha": skill.get("h0s_v3_runtime_bundle_sha"),
        }
    if (not try_ops) and triggerable_risk and skill and skill.get("skill_written"):
        return {
            "verdict": "SOURCE_RISK_ONLY_TRIGGERABLE",
            "risk_authorized": risk_ops,
            "triggerable_risk": triggerable_risk,
            "h0s_v3_runtime_bundle_sha": skill.get("h0s_v3_runtime_bundle_sha"),
        }
    if not try_ops and (untriggerable_only or not risk_ops):
        return {
            "verdict": "SOURCE_EVIDENCE_INSUFFICIENT_FOR_ACTIONABLE_TRANSFER",
            "try_authorized": try_ops,
            "risk_authorized": risk_ops,
            "h0s_v3_produced": False,
        }
    if skill and not skill.get("skill_written"):
        return {
            "verdict": "SOURCE_EVIDENCE_INSUFFICIENT_FOR_ACTIONABLE_TRANSFER",
            "reason": skill.get("verdict") or "skill not written",
            "h0s_v3_produced": False,
        }
    return {"verdict": "CENSUS_UNREADABLE",
            "reason": "authorization present but no ladder match"}


def source_expansion_v3() -> int:
    """#42e r1: 17-file Source expansion, census v3, optional one Skill v3."""
    leftover = [
        line for line in os.popen("ps -ef").read().splitlines()
        if "run_e2_t6_natural_a5_a3.py" in line
        and "--source-expansion-v3" in line
        and str(os.getpid()) not in line
    ]
    if leftover:
        print(json.dumps({"verdict": "CONCURRENT_RUN_BLOCKED",
                          "reason": leftover}, indent=1))
        return 2
    EXPANSION_V3_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(str(EXPANSION_V3_LOCK),
                          os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(json.dumps({"verdict": "CONCURRENT_RUN_BLOCKED",
                          "reason": "lock held"}, indent=1))
        return 2
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    fetch = _fetch_expansion_files()
    if not fetch["ok"]:
        payload = {
            "protocol_version": "t6_nab_42e_source_expansion_v3",
            "run_id": run_id,
            "fetch": fetch,
            "verdict": _v3_verdict(
                fetch=fetch, gate={}, authorization=None,
                skill=None, trigger=None),
        }
        OUT_EXPANSION_V3.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        return 1
    gate = _gate_expansion_files()
    if not gate["all_new_cohorts_usable"]:
        payload = {
            "protocol_version": "t6_nab_42e_source_expansion_v3",
            "run_id": run_id,
            "fetch": {k: v for k, v in fetch.items() if k != "fetched"} | {
                "files": fetch["fetched"]},
            "shape_gate": gate["rows"],
            "abandoned_cohorts": gate["abandoned_cohorts"],
            "verdict": _v3_verdict(
                fetch=fetch, gate=gate, authorization=None,
                skill=None, trigger=None),
        }
        OUT_CENSUS_V3.write_text(_json_text({
            "protocol_version": "t6_nab_42e_census_v3",
            "shape_gate": gate["rows"],
            "abandoned_cohorts": gate["abandoned_cohorts"],
            "verdict": payload["verdict"],
        }), encoding="utf-8")
        OUT_EXPANSION_V3.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        return 1

    wall = LabelWall(released=False)
    consumer = _load_consumer()
    universe = _expansion_universe(gate)
    budget = FitBudget(EXPANSION_AD_FIT_CAP)
    bank = _build_source_bank(
        consumer=consumer, universe=universe, wall=wall, budget=budget)
    runtime = _write_bank_through_runtime(bank["episodes"])
    plan = json.loads(OUT_JSON_V2.read_text(encoding="utf-8"))
    old_rows = list((plan.get("source_bank") or {}).get("rows") or ())
    old_eps = list((plan.get("source_bank") or {}).get("episodes_to_dict") or ())
    new_rows = list(bank["rows"])
    new_eps = list(runtime["to_dict"])
    rows = old_rows + new_rows
    episodes = old_eps + new_eps
    rows_by_id = {str(r["episode_id"]): r for r in rows}
    if len(rows) != len(rows_by_id) or len(episodes) != len(rows):
        census_doc = {
            "protocol_version": "t6_nab_42e_census_v3",
            "verdict": "CENSUS_UNREADABLE",
            "reason": "merged card ids collided or row/episode count mismatch",
            "old_cards": len(old_rows), "new_cards": len(new_rows),
        }
        OUT_CENSUS_V3.write_text(_json_text(census_doc), encoding="utf-8")
        print(json.dumps({"verdict": "CENSUS_UNREADABLE"}, indent=1))
        return 1

    from run_e2_t6_42d_consolidation import _unguided_assertion

    proxy = _v3_proxy_audit(episodes, rows_by_id)
    authorization = _v3_authorize(episodes, rows, proxy)
    trigger = _dev_proposal_frequency()
    cohorts = sorted({str(r.get("cohort")) for r in rows})
    census_doc = {
        "protocol_version": "t6_nab_42e_census_v3",
        "evidence_grade": "NATURAL",
        "evidence_standing": "provisional",
        "counts_as_capability_claim": False,
        "vote_unit_for_authorization": "distinct Source cohort inside one Scope cell",
        "relation_layer": "delayed_relation",
        "old_plan_cards": len(old_rows),
        "new_cards": len(new_rows),
        "episode_count": len(episodes),
        "cohorts": cohorts,
        "unguided_old_plan": _unguided_assertion(plan),
        "feature_proxy_audit": proxy,
        "authorization": authorization,
        "development_proposal_frequency": trigger,
        "shape_gate": gate["rows"],
        "label_mapping": bank["label_mapping"],
        "label_wall": wall.audit(),
        "new_bank_rows": new_rows,
        "merged_rows": [{
            "episode_id": r["episode_id"], "cohort": r["cohort"],
            "round": r["round"], "program": r["program"],
            "delayed_relation": r["delayed_relation"],
            "support_relation": r["support_relation"],
        } for r in rows],
        "known_limits": [
            "plan_v2 20 cards were read from the committed artifact and not recomputed",
            "TRY/RISK votes stay inside one legal Scope cell; no stitching",
            "pss remains forbidden as Scope",
            "new cohorts are OPENED_AS_SOURCE and may never be fresh Target",
        ],
        "cost": {
            "llm": 0, "ad_fits": budget.used,
            "ad_fit_cap": EXPANSION_AD_FIT_CAP,
            "forecast_retrains": 0,
        },
    }
    try_ops = list(authorization["try_authorized"])
    risk_ops = list(authorization["risk_authorized"])
    freq = trigger["proposal_cell_counts"]
    triggerable_risk = [
        op for op in risk_ops
        if int(freq.get(op) or 0) > 0
        or any(token in op for token in TRIGGERABLE_FROM_DEV)
    ]
    should_issue = bool(try_ops or triggerable_risk)
    skill: dict[str, Any] | None = None
    if should_issue:
        skill = _issue_skill_v3(
            authorization=authorization, census_doc=census_doc)
        census_doc["cost"]["llm"] = int(skill.get("llm_api_call_count") or 0)
        OUT_SKILL_V3.write_text(_json_text(skill), encoding="utf-8")
        if skill.get("skill_written"):
            sections = skill.get("sections") or {}
            OUT_SKILL_V3_MD.write_text(
                "# AD Skill v3\n\nverdict pending Part D\n\n"
                "skill_id: `%s`\n\nh0s_v3: `%s`\n\n## sections\n\n%s\n"
                % (SOURCE_SKILL_ID_V3,
                   skill.get("h0s_v3_runtime_bundle_sha"),
                   "\n".join("### %s\n\n%s\n" % (n, sections[n])
                             for n in sections)),
                encoding="utf-8")
    verdict = _v3_verdict(
        fetch=fetch, gate=gate, authorization=authorization,
        skill=skill, trigger=trigger)
    census_doc["verdict"] = verdict
    OUT_CENSUS_V3.write_text(_json_text(census_doc), encoding="utf-8")
    lines = [
        "# #42e census v3",
        "",
        "verdict: **%s**" % verdict["verdict"],
        "cohorts: %s" % ", ".join(cohorts),
        "cards: %d old + %d new = %d" % (
            len(old_rows), len(new_rows), len(rows)),
        "TRY: %s" % try_ops,
        "RISK: %s" % risk_ops,
        "legal scopes: %s" % authorization["legal_scope_features"],
        "LLM %s / %s; AD fit %s / %s" % (
            census_doc["cost"]["llm"], EXPANSION_LLM_CAP,
            budget.used, EXPANSION_AD_FIT_CAP),
        "",
        "## per-file gate",
        "",
        "| file | cohort | ok | length | failures |",
        "|---|---|---|---|---|",
    ]
    for row in gate["rows"]:
        lines.append("| %s | %s | %s | %s | %s |" % (
            row["file"], row["cohort"], row["ok"], row["length"],
            row["failures"] or ""))
    lines.extend([
        "",
        "## proxy audit",
        "",
        json.dumps(proxy, ensure_ascii=False, indent=2),
        "",
        "## unconditional signed summary",
        "",
        json.dumps(authorization["signed_summary_unconditional"],
                   ensure_ascii=False, indent=2),
    ])
    OUT_CENSUS_V3_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if skill and skill.get("skill_written"):
        OUT_SKILL_V3_MD.write_text(
            "# AD Skill v3\n\nverdict: **%s**\n\n"
            "skill_id: `%s`\n\nh0s_v3: `%s`\n\n"
            "delivery: %s\n\n## sections\n\n%s\n"
            % (
                verdict["verdict"], SOURCE_SKILL_ID_V3,
                skill.get("h0s_v3_runtime_bundle_sha"),
                (skill.get("delivery_assert") or {}).get("pass"),
                "\n".join("### %s\n\n%s\n" % (n, skill["sections"][n])
                          for n in skill["sections"]),
            ),
            encoding="utf-8")
    payload = {
        "protocol_version": "t6_nab_42e_source_expansion_v3",
        "entry": "--source-expansion-v3",
        "run_id": run_id,
        "evidence_grade": "NATURAL",
        "evidence_standing": "provisional",
        "counts_as_capability_claim": False,
        "fetch": {"ok": fetch["ok"], "missing": fetch["missing"],
                  "exposure": fetch["exposure"]},
        "shape_gate_kept": gate["usable"],
        "cost": census_doc["cost"],
        "authorization": {
            "try": try_ops, "risk": risk_ops,
            "scopes": authorization["authorization_scopes"],
        },
        "verdict": verdict,
        "label_wall_breached": wall.audit()["breached"],
        "target_key_requests": wall.audit()["target_key_requests"],
    }
    OUT_EXPANSION_V3.write_text(_json_text(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": verdict["verdict"],
        "try": try_ops,
        "risk": risk_ops,
        "llm": census_doc["cost"]["llm"],
        "ad_fits": budget.used,
        "sha": verdict.get("h0s_v3_runtime_bundle_sha"),
        "wall": wall.audit()["breached"],
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_CENSUS_V3)
    print("wrote", OUT_CENSUS_V3_MD)
    return 0


def _reconstruct_h0s_v3():
    """Deterministic rebuild from the frozen v3 entry.  No Slow."""
    if not OUT_SKILL_V3.exists():
        raise Stop("CHECKPOINT_FAILED",
                   "missing frozen v3 artifact %s" % _repo_rel(OUT_SKILL_V3))
    doc = json.loads(OUT_SKILL_V3.read_text(encoding="utf-8"))
    entry = dict(doc.get("entry") or {})
    if entry.get("skill_id") != SOURCE_SKILL_ID_V3:
        raise Stop("CHECKPOINT_FAILED",
                   "frozen entry skill_id is %r" % entry.get("skill_id"))
    h0, snapshot = _materialize_named_skill(SOURCE_SKILL_ID_V3, entry)
    if snapshot.runtime_bundle_sha != H0S_V3_EXPECTED_SHA:
        raise Stop(
            "CHECKPOINT_FAILED",
            "reconstructed h0s_v3 sha %s != expected %s"
            % (snapshot.runtime_bundle_sha, H0S_V3_EXPECTED_SHA))
    return h0, snapshot, entry


def _static_delivery_v3(h0, snapshot) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (
        resolve_harness_view,
    )
    from evaluation.functional.task_episode_harness.e1 import _FastAgentStub

    a5 = TTHAMethod(_FastAgentStub(), snapshot, ())
    a3 = TTHAMethod(_FastAgentStub(), h0, ())
    a5_view = resolve_harness_view(
        snapshot, {"task_kind": "anomaly_detection"}, role="fast")
    a3_view = resolve_harness_view(
        h0, {"task_kind": "anomaly_detection"}, role="fast")
    a5_ids = [s.skill_id for s in a5_view.skills]
    a3_ids = [s.skill_id for s in a3_view.skills]
    delivery = {
        "a5_retrieves_v3": SOURCE_SKILL_ID_V3 in a5_ids,
        "a3_does_not_retrieve_v3": SOURCE_SKILL_ID_V3 not in a3_ids,
        "a5_memory_empty": list(
            getattr(a5, "experience_episodes", ()) or ()) == [],
        "a3_memory_empty": list(
            getattr(a3, "experience_episodes", ()) or ()) == [],
        "a5_view_skill_ids": a5_ids,
        "a3_view_skill_ids": a3_ids,
        "h0s_v3_runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "h0_runtime_bundle_sha": h0.runtime_bundle_sha,
    }
    delivery["pass"] = all((
        delivery["a5_retrieves_v3"],
        delivery["a3_does_not_retrieve_v3"],
        delivery["a5_memory_empty"],
        delivery["a3_memory_empty"],
    ))
    if not delivery["pass"]:
        raise Stop("CHECKPOINT_FAILED",
                   "static delivery failed: %s" % delivery)
    return delivery


def _names_in(blob: Any) -> set[str]:
    text = json.dumps(blob, ensure_ascii=False).lower() if not isinstance(
        blob, str) else blob.lower()
    found: set[str] = set()
    if "hampel" in text:
        found.add("hampel_filter")
    if "iqr" in text:
        found.add("outlier_iqr")
    if "mad" in text:
        found.add("outlier_mad")
    if "winsor" in text:
        found.add("winsorize")
    if "identity" in text:
        found.add("identity")
    return found


def _op_counts(items: Sequence[Any]) -> dict[str, int]:
    counts = {name: 0 for name in RISK_OPS_V3 + ("winsorize", "identity")}
    for item in items:
        names = _names_in(item)
        for name in counts:
            if name in names:
                counts[name] += 1
    return counts


def _cite_v3(trace: Any, retrieved: Sequence[str]) -> dict[str, Any]:
    """Causal cite: retrieved v3 AND proposal text names the risk knowledge."""
    retrieved_hit = SOURCE_SKILL_ID_V3 in list(retrieved or [])
    blobs: list[str] = []
    for attr in (
        "proposal_rationale", "rationale", "decision_text",
        "retrieved_knowledge_summary",
    ):
        value = getattr(trace, attr, None)
        if value:
            blobs.append(json.dumps(value, ensure_ascii=False)
                         if not isinstance(value, str) else value)
    stages = getattr(trace, "stages", None) or ()
    for stage in stages:
        payload = getattr(stage, "payload", None) or {}
        blobs.append(json.dumps(payload, ensure_ascii=False))
        name = str(getattr(stage, "stage", "") or payload.get("stage") or "")
        if name == "propose" or "propose" in json.dumps(payload).lower():
            blobs.append(json.dumps(payload, ensure_ascii=False))
    joined = "\n".join(blobs).lower()
    cites = any(token in joined for token in (
        "source_investigation_ad_v3",
        "lower proposal priority",
        "restricted probe",
        "depriorit",
        "降权",
        "降低",
    ))
    names_risk = bool(_names_in(joined) & set(RISK_OPS_V3))
    return {
        "retrieved_v3": retrieved_hit,
        "cites_risk_knowledge": cites and retrieved_hit,
        "names_risk_operator_in_propose": names_risk,
        "excerpt": joined[:400],
    }


def _cell_acceptance_row(cell: Mapping[str, Any]) -> dict[str, Any]:
    retrieved = list(cell.get("retrieved_skill_ids") or [])
    pool = list(cell.get("pool") or [])
    probes = list(cell.get("probes") or [])
    probe_ids = [p.get("candidate_id") for p in probes]
    chosen = cell.get("chosen")
    proposed = _op_counts(pool)
    probed = _op_counts(probe_ids)
    selected = _op_counts([chosen] if chosen else [])
    occupants = []
    for item in pool:
        names = _names_in(item)
        if names & set(RISK_OPS_V3):
            continue
        if "winsorize" in names:
            occupants.append("winsorize")
        elif str(item) == "identity" or names == {"identity"}:
            occupants.append("identity")
        elif item:
            occupants.append(str(item))
    episodes = list(cell.get("episode_rows") or [])
    risk_positive = []
    for ep in episodes:
        sig = str(ep.get("workflow_signature") or "")
        rel = str(ep.get("relation") or "").upper()
        if rel == "POSITIVE" and any(
                token in sig for token in ("mad", "iqr")):
            risk_positive.append({
                "workflow": sig, "relation": rel,
                "status": ep.get("local_status"),
                "layer": ep.get("evidence_level"),
            })
    support_event = cell.get("fast_skill_event") or {}
    delayed_event = cell.get("delayed_event") or {}
    override = "NO_OVERRIDE_OCCASION"
    probed_risk = any(probed[op] > 0 for op in RISK_OPS_V3)
    support_pos = False
    if isinstance(support_event, Mapping):
        blob = json.dumps(support_event, ensure_ascii=False).lower()
        support_pos = "positive" in blob and "draft" in blob
    if probed_risk and (
            any(e.get("relation") == "POSITIVE"
                and e.get("evidence_level") == "SUPPORT"
                for e in episodes) or support_pos):
        statuses = {e.get("local_status") for e in episodes}
        override = (
            "DRAFT_FORMED" if (
                "LOCAL_DRAFT" in statuses or cell.get("approved_skill_id")
                or "LOCAL_ACTIVE" in statuses)
            else "NO_DRAFT"
        )
    return {
        "cell": cell.get("cell"),
        "arm": cell.get("arm"),
        "cohort": cell.get("cohort"),
        "round": cell.get("round"),
        "retrieved_v3": SOURCE_SKILL_ID_V3 in retrieved,
        "retrieved_skill_ids": retrieved,
        "held": (cell.get("retrieval_before_round") or {}).get("held"),
        "pool": pool,
        "chosen": chosen,
        "probe_order": cell.get("probe_order"),
        "probes": probes,
        "proposed": proposed,
        "shortlisted": proposed,
        "probed": probed,
        "selected": selected,
        "vacated_slot_occupants": occupants,
        "risk_positive_events": risk_positive,
        "override": override,
        "harm_support": cell.get("harmed_series_support_layer"),
        "harm_delayed": cell.get("harmed_series_delayed_layer"),
        "delayed_utility": cell.get("delayed_utility"),
        "delayed_event": delayed_event,
        "approved_skill_id": cell.get("approved_skill_id"),
        "activated": cell.get("activated"),
        "episode_rows": episodes,
        "non_identity_trials": cell.get("non_identity_trials"),
        "abstained": cell.get("abstained"),
        "winsorize_proposed": proposed["winsorize"],
        "winsorize_selected": selected["winsorize"],
        "cite": _cite_v3(type("T", (), {})(), retrieved) | {
            "cell_blob_cites": any(
                token in json.dumps(cell, ensure_ascii=False).lower()
                for token in (
                    "source_investigation_ad_v3",
                    "lower proposal priority",
                    "restricted probe",
                    "depriorit",
                )
            ),
        },
    }


def _pair_cells(rows: Sequence[Mapping[str, Any]]
                ) -> dict[tuple[str, str], dict[str, Mapping[str, Any]]]:
    pairs: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("cohort")), str(row.get("round")))
        pairs.setdefault(key, {})[str(row.get("arm"))] = row
    return pairs


def _accept_v3_verdict(
    rows: Sequence[Mapping[str, Any]],
    run: Mapping[str, Any],
) -> dict[str, Any]:
    a3 = [r for r in rows if r.get("arm") == "A3"]
    a5 = [r for r in rows if r.get("arm") == "A5"]
    a5_miss = [r["cell"] for r in a5 if not r.get("retrieved_v3")]
    a3_leak = [r["cell"] for r in a3 if r.get("retrieved_v3")]
    memory = [r["cell"] for r in rows if int(r.get("held") or 0) > 0]
    if a5_miss or a3_leak or memory or run.get("stopped"):
        return {
            "verdict": (
                run.get("stopped") or "CHECKPOINT_FAILED"),
            "a5_miss": a5_miss, "a3_leak": a3_leak, "memory": memory,
            "l2_opens": False,
        }

    a3_non = sum(int(r.get("non_identity_trials") or 0) for r in a3)
    a5_non = sum(int(r.get("non_identity_trials") or 0) for r in a5)
    if a5_non == 0 and a3_non > 0:
        return {"verdict": "IDENTITY_ONLY_COLLAPSE",
                "a3_non_identity": a3_non, "a5_non_identity": a5_non,
                "l2_opens": False, "v3": "closed"}

    missed: list[dict[str, Any]] = []
    associated_miss: list[dict[str, Any]] = []
    pairs = _pair_cells(rows)
    for key, arms in pairs.items():
        left, right = arms.get("A3"), arms.get("A5")
        if not left or not right:
            continue
        a3_pos = left.get("risk_positive_events") or []
        if not a3_pos:
            continue
        a5_probed_mi = (
            int((right.get("probed") or {}).get("outlier_mad") or 0)
            + int((right.get("probed") or {}).get("outlier_iqr") or 0)
        )
        if a5_probed_mi > 0:
            continue
        cite = right.get("cite") or {}
        a3_had = (
            int((left.get("proposed") or {}).get("outlier_mad") or 0)
            + int((left.get("proposed") or {}).get("outlier_iqr") or 0)
            + int((left.get("probed") or {}).get("outlier_mad") or 0)
            + int((left.get("probed") or {}).get("outlier_iqr") or 0)
        )
        rec = {
            "cell_pair": "%s/%s" % key,
            "a3_events": a3_pos,
            "a5_retrieved_v3": right.get("retrieved_v3"),
            "a5_cites_risk_knowledge": cite.get("cites_risk_knowledge"),
            "a3_had_risk_operator": a3_had > 0,
        }
        causal = bool(
            right.get("retrieved_v3")
            and (cite.get("cites_risk_knowledge") or a3_had > 0)
        )
        if causal:
            missed.append(rec)
        else:
            associated_miss.append(rec)
    if missed:
        return {
            "verdict": "SOURCE_RISK_PRIOR_BLOCKS_TARGET_ADAPTATION",
            "missed_positive_events": missed,
            "associated_misses_not_causal": associated_miss,
            "l2_opens": False, "v3": "closed",
        }

    def _sum_risk_probes(group):
        return sum(
            int((r.get("probed") or {}).get(op) or 0)
            for r in group for op in RISK_OPS_V3)

    a3_risk_p = _sum_risk_probes(a3)
    a5_risk_p = _sum_risk_probes(a5)
    a3_win_prop = sum(int(r.get("winsorize_proposed") or 0) for r in a3)
    a5_win_prop = sum(int(r.get("winsorize_proposed") or 0) for r in a5)
    a3_win_sel = sum(int(r.get("winsorize_selected") or 0) for r in a3)
    a5_win_sel = sum(int(r.get("winsorize_selected") or 0) for r in a5)
    a5_win_harm = sum(
        1 for r in a5
        if (r.get("winsorize_proposed") or r.get("winsorize_selected"))
        and (int(r.get("harm_support") or 0) > 0
             or int(r.get("harm_delayed") or 0) > 0)
    )
    if (a5_risk_p < a3_risk_p
            and (a5_win_prop > a3_win_prop or a5_win_sel > a3_win_sel)
            and a5_win_harm > 0):
        return {
            "verdict": "RISK_DISPLACEMENT_NEGATIVE_TRANSFER",
            "a3_risk_probes": a3_risk_p, "a5_risk_probes": a5_risk_p,
            "a3_winsorize": {"proposed": a3_win_prop, "selected": a3_win_sel},
            "a5_winsorize": {"proposed": a5_win_prop, "selected": a5_win_sel,
                             "harm_cells": a5_win_harm},
            "l2_opens": False, "v3": "closed",
        }

    def _harmful_risk(group):
        n = 0
        for r in group:
            if not any(int((r.get("probed") or {}).get(op) or 0)
                       for op in RISK_OPS_V3):
                continue
            if int(r.get("harm_support") or 0) > 0 or int(
                    r.get("harm_delayed") or 0) > 0:
                n += 1
        return n

    a3_harm_risk = _harmful_risk(a3)
    a5_harm_risk = _harmful_risk(a5)
    a3_delayed = [r.get("delayed_utility") for r in a3]
    a5_delayed = [r.get("delayed_utility") for r in a5]
    a3_harm = sum(int(r.get("harm_delayed") or 0) for r in a3)
    a5_harm = sum(int(r.get("harm_delayed") or 0) for r in a5)

    def _num(xs):
        vals = [float(x) for x in xs if isinstance(x, (int, float))]
        return sum(vals) if vals else 0.0

    terminal_not_worse = (
        a5_harm <= a3_harm and _num(a5_delayed) >= _num(a3_delayed)
    )
    if a5_harm_risk < a3_harm_risk and not missed and terminal_not_worse:
        return {
            "verdict": "RISK_PRIOR_BEHAVIOR_EFFECTIVE",
            "a3_harmful_risk_cells": a3_harm_risk,
            "a5_harmful_risk_cells": a5_harm_risk,
            "associated_misses_not_causal": associated_miss,
            "l2_opens": True, "v3": "kept",
        }

    identical = True
    for key, arms in pairs.items():
        left, right = arms.get("A3"), arms.get("A5")
        if not left or not right:
            identical = False
            break
        if (left.get("proposed") != right.get("proposed")
                or left.get("probed") != right.get("probed")
                or left.get("selected") != right.get("selected")):
            identical = False
            break
    a3_risk_prop = sum(
        int((r.get("proposed") or {}).get(op) or 0)
        for r in a3 for op in RISK_OPS_V3)
    a5_risk_prop = sum(
        int((r.get("proposed") or {}).get(op) or 0)
        for r in a5 for op in RISK_OPS_V3)
    if identical:
        sub = None
        if a3_risk_prop == 0 and a5_risk_prop == 0:
            sub = "NO_TRIGGERING_OCCASION"
        return {
            "verdict": "RISK_PRIOR_INERT",
            "sub": sub,
            "l2_opens": False, "v3": "archived",
        }
    return {
        "verdict": "RISK_PRIOR_EFFECT_AMBIGUOUS",
        "associated_misses_not_causal": associated_miss,
        "l2_opens": False, "v3": "archived",
        "note": "arms differ but no pre-registered effect cell fired",
    }


def accept_skill_v3() -> int:
    """#42e1: reconstruct frozen v3, then one-shot 8-cell acceptance."""
    leftover = [
        line for line in os.popen("ps -ef").read().splitlines()
        if "run_e2_t6_natural_a5_a3.py" in line
        and "--accept-skill-v3" in line
        and str(os.getpid()) not in line
    ]
    autopsy = leftover
    if leftover:
        print(json.dumps({"verdict": "CONCURRENT_RUN_BLOCKED",
                          "reason": leftover}, indent=1))
        return 2
    ACCEPT_V3_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(str(ACCEPT_V3_LOCK),
                          os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(json.dumps({"verdict": "CONCURRENT_RUN_BLOCKED",
                          "reason": "lock held"}, indent=1))
        return 2
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    try:
        h0, snapshot, entry = _reconstruct_h0s_v3()
        delivery = _static_delivery_v3(h0, snapshot)
    except Stop as stop:
        payload = {
            "protocol_version": "t6_nab_42e1_behavior_acceptance_v1",
            "verdict": {"verdict": stop.verdict, "reason": stop.reason},
            "run_id": run_id, "autopsy": autopsy,
        }
        OUT_ACCEPT_V3.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        return 1

    if not OUT_JSON_V2.exists():
        payload = {"verdict": {"verdict": "CHECKPOINT_FAILED",
                               "reason": "missing v2 plan"}}
        OUT_ACCEPT_V3.write_text(_json_text(payload), encoding="utf-8")
        return 1
    plan = json.loads(OUT_JSON_V2.read_text(encoding="utf-8"))
    wall = LabelWall(released=True)
    universe = _load_universe(gate_all(row_order_contract=True))
    budget = FitBudget(ACCEPT_V3_AD_FIT_BUDGET)
    store_tag = "t6e1_%s" % run_id
    try:
        run = _run_cells(
            plan=plan,
            cohort_rows=universe["target"],
            agent_factory=_evaluate_agent,
            backend_factory=_evaluate_backend,
            llm_budget=ACCEPT_V3_LLM_BUDGET,
            fit_budget=budget,
            wall=wall,
            store_tag=store_tag,
            snapshot_for_arm={"A3": h0, "A5": snapshot},
        )
    except Stop as stop:
        payload = {
            "protocol_version": "t6_nab_42e1_behavior_acceptance_v1",
            "verdict": {"verdict": stop.verdict, "reason": stop.reason},
            "run_id": run_id, "delivery": delivery,
            "label_wall": wall.audit(),
        }
        OUT_ACCEPT_V3.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        return 1

    # attach cite from live traces already stored on cells via retrieved ids;
    # re-read last_trace fields if present on cell records.
    rows = []
    for cell in run.get("cells") or []:
        row = _cell_acceptance_row(cell)
        rows.append(row)
    verdict = _accept_v3_verdict(rows, run)
    sharp = next(
        (r for r in rows
         if r.get("cohort") == "target_cpm" and r.get("round") == "r2"),
        None)
    sharp_pair = {
        arm: next((r for r in rows if r.get("arm") == arm
                   and r.get("cohort") == "target_cpm"
                   and r.get("round") == "r2"), None)
        for arm in ("A3", "A5")
    }
    payload = {
        "protocol_version": "t6_nab_42e1_behavior_acceptance_v1",
        "entry": "--accept-skill-v3",
        "evidence_grade": "DEVELOPMENT",
        "evidence_standing": "same-context",
        "counts_as_capability_claim": False,
        "counts_as_cross_domain_claim": False,
        "l2_decision": (
            "OPEN" if verdict.get("l2_opens") else "CLOSED"
        ),
        "run_id": run_id,
        "lock": _repo_rel(ACCEPT_V3_LOCK),
        "autopsy": autopsy,
        "h0s_v3_runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "v3_entry_skill_id": entry.get("skill_id"),
        "v3_text_frozen": True,
        "delivery": delivery,
        "cells": rows,
        "sharp_cell_cpm_r2": sharp_pair,
        "verdict": verdict,
        "cost": {
            "llm": run.get("llm_calls"),
            "llm_cap": run.get("llm_budget"),
            "ad_fits": run.get("ad_fits"),
            "ad_fit_cap": run.get("ad_fit_cap"),
            "forecast_retrains": 0,
        },
        "label_wall": wall.audit(),
        "stopped": run.get("stopped"),
    }
    OUT_ACCEPT_V3.write_text(_json_text(payload), encoding="utf-8")
    lines = [
        "# #42e1 v3 behavior acceptance",
        "",
        "verdict: **%s**" % verdict["verdict"],
        "L2: **%s**" % payload["l2_decision"],
        "run_id: `%s`" % run_id,
        "h0s_v3: `%s`" % snapshot.runtime_bundle_sha,
        "LLM %s / %s; AD fit %s / %s; retrain 0" % (
            run.get("llm_calls"), run.get("llm_budget"),
            run.get("ad_fits"), run.get("ad_fit_cap")),
        "wall breached: %s; target_key_requests: %s" % (
            wall.audit()["breached"],
            len(wall.audit()["target_key_requests"])),
        "",
        "| cell | v3? | pool | chosen | ham/iqr/mad prop | probes | win p/s | non-id | delayed | harmS/D | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        prop = row["proposed"]
        prb = row["probed"]
        lines.append(
            "| %s | %s | %s | %s | %s/%s/%s | %s/%s/%s | %s/%s | %s | %s | %s/%s | %s |"
            % (
                row["cell"], row["retrieved_v3"], row["pool"], row["chosen"],
                prop["hampel_filter"], prop["outlier_iqr"], prop["outlier_mad"],
                prb["hampel_filter"], prb["outlier_iqr"], prb["outlier_mad"],
                row["winsorize_proposed"], row["winsorize_selected"],
                row["non_identity_trials"], row["delayed_utility"],
                row["harm_support"], row["harm_delayed"],
                [(e.get("workflow_signature"), e.get("relation"),
                  e.get("local_status")) for e in row["episode_rows"]] or "—",
            ))
    lines.extend([
        "",
        "## CPM r2 (sharp cell)",
        "",
        json.dumps(sharp_pair, ensure_ascii=False, indent=2),
        "",
        "## verdict detail",
        "",
        json.dumps(verdict, ensure_ascii=False, indent=2),
    ])
    OUT_ACCEPT_V3_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": verdict["verdict"],
        "l2": payload["l2_decision"],
        "llm": run.get("llm_calls"),
        "ad_fits": run.get("ad_fits"),
        "sha": snapshot.runtime_bundle_sha,
        "wall": wall.audit()["breached"],
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_ACCEPT_V3)
    print("wrote", OUT_ACCEPT_V3_MD)
    return 0


def _winsorize_rows_from_frozen() -> list[dict[str, Any]]:
    """Read 8 winsorize cards from committed banks.  Do not recompute relations."""
    if not OUT_JSON_V2.exists() or not OUT_CENSUS_V3.exists():
        raise Stop("INSTRUMENT_UNREADABLE",
                   "missing plan_v2 or census_v3")
    plan = json.loads(OUT_JSON_V2.read_text(encoding="utf-8"))
    census = json.loads(OUT_CENSUS_V3.read_text(encoding="utf-8"))
    old = [r for r in ((plan.get("source_bank") or {}).get("rows") or [])
           if r.get("program") == "winsorize"]
    new = [r for r in (census.get("new_bank_rows") or [])
           if r.get("program") == "winsorize"]
    rows = old + new
    if len(rows) != 8:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "expected 8 winsorize episodes, found %d" % len(rows))
    return rows


def _cohort_series_files() -> dict[str, list[tuple[str, Path]]]:
    mapping: dict[str, list[tuple[str, Path]]] = {}
    if not OUT_JSON_V2.exists():
        raise Stop("INSTRUMENT_UNREADABLE", "missing plan_v2")
    plan = json.loads(OUT_JSON_V2.read_text(encoding="utf-8"))
    source = ((plan.get("cohorts") or {}).get("source") or {})
    for cohort, (directory, _take) in SOURCE_COHORTS.items():
        names = list(source.get(cohort) or [])
        mapping[cohort] = [
            (name, DATA_ROOT / directory / name) for name in names]
    for cohort, (directory, names) in EXPANSION_COHORTS.items():
        mapping[cohort] = [
            (name, DATA_ROOT / directory / name) for name in names]
    return mapping


def _extreme_runs(block: np.ndarray) -> dict[str, Any]:
    """Reuse public_features MAD / z>=4 semantics.  No new epsilon."""
    from SelfEvolvingHarnessTS.contracts.observables import (
        OUTLIER_Z_THRESHOLD, PUBLIC_ROBUST_Z_MAD_FLOOR,
    )

    values = np.asarray(block, dtype=np.float64).ravel()
    finite = np.isfinite(values)
    if not np.any(finite):
        filled = np.zeros_like(values)
    else:
        indices = np.arange(values.size, dtype=np.float64)
        filled = np.interp(indices, indices[finite], values[finite])
    median = float(np.median(filled))
    mad = float(np.median(np.abs(filled - median)))
    scale = max(1.4826 * mad, PUBLIC_ROBUST_Z_MAD_FLOOR)
    z = np.abs(filled - median) / scale
    extreme = z >= OUTLIER_Z_THRESHOLD
    runs: list[int] = []
    length = 0
    for flag in extreme.tolist():
        if flag:
            length += 1
            continue
        if length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    isolated = sum(1 for run in runs if run == 1)
    return {
        "n": int(values.size),
        "finite": int(finite.sum()),
        "median": median,
        "mad": mad,
        "scale": float(scale),
        "z_peak": float(np.max(z)) if z.size else None,
        "extreme_count": int(extreme.sum()),
        "run_count": len(runs),
        "isolated_run_count": isolated,
        "max_run_len": max(runs) if runs else 0,
        "runs": runs,
        "mad_was_zero": mad == 0.0,
    }


def _episode_observation(
    row: Mapping[str, Any],
    files: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    round_name = str(row["round"])
    series_rows: list[dict[str, Any]] = []
    isolated_sum = 0
    run_sum = 0
    max_run = 0
    for name, path in files:
        if not path.is_file():
            raise Stop("INSTRUMENT_UNREADABLE",
                       "missing series file %s" % path)
        read = _read_series(path, row_order_contract=True)
        if not read.get("ok"):
            raise Stop("INSTRUMENT_UNREADABLE",
                       "%s failed row-order read: %s"
                       % (name, read.get("failures")))
        values = np.asarray(read["values"], dtype=np.float64)
        lo, hi = _window_plan(int(read["length"]))[round_name]["train"]
        stats = _extreme_runs(values[lo:hi])
        stats["file"] = name
        stats["train_span"] = [lo, hi]
        series_rows.append(stats)
        isolated_sum += int(stats["isolated_run_count"])
        run_sum += int(stats["run_count"])
        max_run = max(max_run, int(stats["max_run_len"]))
    if run_sum == 0:
        frac = None
        dominant = None
        excluded = True
    else:
        frac = isolated_sum / run_sum
        dominant = bool(frac >= ISOLATED_FRACTION_THRESHOLD)
        excluded = False
    delayed = str(row["delayed_relation"]).upper()
    if delayed == "POSITIVE":
        cls = "positive"
    elif delayed in {"NEGATIVE", "CONFLICT"}:
        cls = "adverse"
    else:
        cls = "archive"
    return {
        "episode_id": row["episode_id"],
        "cohort": row["cohort"],
        "round": round_name,
        "support_relation": row["support_relation"],
        "delayed_relation": delayed,
        "support_harmed": row.get("support_harmed"),
        "delayed_harmed": row.get("delayed_harmed"),
        "support_worst_series": row.get("support_worst_series"),
        "delayed_worst_series": row.get("delayed_worst_series"),
        "class": cls,
        "episode_isolated_fraction": frac,
        "episode_max_run_len": max_run,
        "isolated_dominant": dominant,
        "excluded_na": excluded,
        "isolated_run_sum": isolated_sum,
        "run_sum": run_sum,
        "series": series_rows,
    }


def _c2_proxy(classified: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_value: dict[str, set[str]] = {}
    by_cohort: dict[str, set[str]] = {}
    for row in classified:
        value = str(bool(row["isolated_dominant"]))
        cohort = str(row["cohort"])
        by_value.setdefault(value, set()).add(cohort)
        by_cohort.setdefault(cohort, set()).add(value)
    sides = {key: by_value.get(key, set()) for key in ("True", "False")}
    single = any(len(cs) == 1 for cs in by_value.values())
    complete = (
        all(len(cs) == 1 for cs in by_value.values())
        and all(len(vs) == 1 for vs in by_cohort.values())
        and len(by_value) == len(by_cohort)
        and len(by_value) > 0
    )
    both_ge2 = all(len(cs) >= 2 for cs in sides.values()) and all(sides.values())
    return {
        "values_to_cohorts": {k: sorted(v) for k, v in sorted(by_value.items())},
        "single_cohort_indicator": single,
        "complete_cohort_partition_replica": complete,
        "both_boolean_sides_present": all(bool(cs) for cs in sides.values()),
        "both_boolean_sides_ge2_cohorts": both_ge2,
        "usable_as_scope": (
            not single and not complete and both_ge2
        ),
    }


def _c1_direction(classified: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pos = [r for r in classified if r["class"] == "positive"]
    adv = [r for r in classified if r["class"] == "adverse"]
    pos_sides = {r["isolated_dominant"] for r in pos}
    adv_true = sum(1 for r in adv if r["isolated_dominant"] is True)
    adv_false = sum(1 for r in adv if r["isolated_dominant"] is False)
    if not pos or not adv or None in pos_sides or len(pos_sides) != 1:
        return {
            "separates": False,
            "positive_side": None,
            "all_positive_same_side": bool(pos) and len(pos_sides) == 1,
            "adverse_opposite_rate": None,
            "positive_n": len(pos),
            "adverse_n": len(adv),
        }
    side = next(iter(pos_sides))
    opposite = adv_false if side is True else adv_true
    rate = opposite / len(adv)
    return {
        "separates": bool(rate >= 0.75),
        "positive_side": side,
        "all_positive_same_side": True,
        "adverse_opposite_rate": rate,
        "adverse_opposite_count": opposite,
        "positive_n": len(pos),
        "adverse_n": len(adv),
    }


def _c3_loco(classified: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cohorts = sorted({r["cohort"] for r in classified})
    folds = []
    all_readable = True
    direction_holds = True
    base = _c1_direction(classified)
    for left_out in cohorts:
        remain = [r for r in classified if r["cohort"] != left_out]
        pos = [r for r in remain if r["class"] == "positive"]
        adv = [r for r in remain if r["class"] == "adverse"]
        if not pos or not adv:
            folds.append({
                "left_out": left_out, "readable": False,
                "reason": "LOCO_UNREADABLE",
                "positive_n": len(pos), "adverse_n": len(adv),
            })
            all_readable = False
            continue
        c1 = _c1_direction(remain)
        holds = (
            c1["all_positive_same_side"]
            and c1["positive_side"] == base.get("positive_side")
            and (c1.get("adverse_opposite_rate") or 0) >= 0.75
        )
        if not holds:
            direction_holds = False
        folds.append({
            "left_out": left_out, "readable": True, "c1": c1,
            "direction_holds": holds,
        })
    return {
        "folds": folds,
        "all_readable": all_readable,
        "direction_holds": bool(all_readable and direction_holds and folds),
    }


def _pattern_verdict(
    *,
    instrument_ok: bool,
    classified: Sequence[Mapping[str, Any]],
    c1: Mapping[str, Any],
    c2: Mapping[str, Any],
    c3: Mapping[str, Any],
    pos_cohorts: Sequence[str],
    adv_cohorts: Sequence[str],
) -> dict[str, Any]:
    if not instrument_ok:
        return {"verdict": "INSTRUMENT_UNREADABLE"}
    c2_pass = bool(c2.get("usable_as_scope"))
    independent = len(pos_cohorts) >= 2 and len(adv_cohorts) >= 2
    if (c1.get("separates") and c1.get("all_positive_same_side")
            and c2_pass and independent and c3.get("all_readable")
            and c3.get("direction_holds")):
        return {
            "verdict": "PATTERN_DISCRIMINATES_WITH_INDEPENDENT_COHORT_SUPPORT",
            "claim_cap": "MECHANISM",
            "may_enter_skill_scope": False,
            "note": "legal mechanism clue; Scope admission is a later mainline cut",
        }
    if c1.get("separates") and not c2_pass:
        return {
            "verdict": "PATTERN_CORRELATES_BUT_COHORT_PROXY",
            "may_enter_skill_scope": False,
        }
    if c1.get("separates") and c2_pass and (
            not independent or not c3.get("all_readable")):
        return {
            "verdict": "PATTERN_SEPARATES_BUT_POSITIVE_COHORT_SINGLETON",
            "may_enter_skill_scope": False,
            "claim_cap": "MECHANISM_CANDIDATE",
        }
    if not c1.get("separates"):
        return {
            "verdict": "PATTERN_NO_DISCRIMINATION",
            "family_closed": "isolated-extreme × winsorize",
        }
    return {"verdict": "OBSERVED_BUT_UNCLASSIFIED"}


def pattern_discriminator_v1() -> int:
    """#42e2 r1: one frozen Observation on winsorize.  0 LLM / 0 fit."""
    try:
        rows = _winsorize_rows_from_frozen()
        files = _cohort_series_files()
        observations = []
        for row in rows:
            cohort = str(row["cohort"])
            if cohort not in files:
                raise Stop("INSTRUMENT_UNREADABLE",
                           "no files mapped for %s" % cohort)
            observations.append(_episode_observation(row, files[cohort]))
    except Stop as stop:
        payload = {
            "protocol_version": "t6_nab_42e2_pattern_discriminator_v1",
            "verdict": {"verdict": stop.verdict, "reason": stop.reason},
            "cost": {"llm": 0, "ad_fits": 0, "forecast_retrains": 0},
        }
        OUT_PATTERN_V1.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        return 1

    bags = {"positive": [], "negative": [], "conflict": [], "immaterial": []}
    for row in observations:
        delayed = row["delayed_relation"]
        if delayed == "POSITIVE":
            bags["positive"].append(row["episode_id"])
        elif delayed == "NEGATIVE":
            bags["negative"].append(row["episode_id"])
        elif delayed == "CONFLICT":
            bags["conflict"].append(row["episode_id"])
        else:
            bags["immaterial"].append(row["episode_id"])

    classified = [r for r in observations if not r["excluded_na"]
                  and r["class"] in {"positive", "adverse"}]
    excluded = [r["episode_id"] for r in observations if r["excluded_na"]]
    pos_cohorts = sorted({r["cohort"] for r in classified if r["class"] == "positive"})
    adv_cohorts = sorted({r["cohort"] for r in classified if r["class"] == "adverse"})
    c1 = _c1_direction(classified)
    c2 = _c2_proxy(classified)
    c3 = _c3_loco(classified)
    verdict = _pattern_verdict(
        instrument_ok=True, classified=classified, c1=c1, c2=c2, c3=c3,
        pos_cohorts=pos_cohorts, adv_cohorts=adv_cohorts)
    payload = {
        "protocol_version": "t6_nab_42e2_pattern_discriminator_v1",
        "entry": "--pattern-discriminator-v1",
        "evidence_grade": "MECHANISM",
        "evidence_standing": "development",
        "counts_as_capability_claim": False,
        "forms_skill": False,
        "observation": {
            "name": "isolated_dominant",
            "z_threshold": 4.0,
            "isolated_fraction_threshold": ISOLATED_FRACTION_THRESHOLD,
            "window": "episode train span; per-series, never concatenated",
            "mad_scale": "public_features: max(1.4826*MAD, PUBLIC_ROBUST_Z_MAD_FLOOR)",
            "no_second_feature": True,
        },
        "response_table": observations,
        "bags": bags,
        "excluded_na": excluded,
        "classified_n": len(classified),
        "positive_distinct_cohorts": pos_cohorts,
        "adverse_distinct_cohorts": adv_cohorts,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "verdict": verdict,
        "cost": {"llm": 0, "ad_fits": 0, "forecast_retrains": 0},
    }
    OUT_PATTERN_V1.write_text(_json_text(payload), encoding="utf-8")
    lines = [
        "# #42e2 isolated_dominant × winsorize",
        "",
        "verdict: **%s**" % verdict["verdict"],
        "may_enter_skill_scope: %s" % verdict.get("may_enter_skill_scope"),
        "positive cohorts: %s" % pos_cohorts,
        "adverse cohorts: %s" % adv_cohorts,
        "excluded NA: %s" % excluded,
        "",
        "| episode | delayed | class | isolated_frac | dominant | max_run |",
        "|---|---|---|---|---|---|",
    ]
    for row in observations:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            row["episode_id"], row["delayed_relation"], row["class"],
            row["episode_isolated_fraction"], row["isolated_dominant"],
            row["episode_max_run_len"]))
    lines.extend([
        "",
        "## C1",
        json.dumps(c1, ensure_ascii=False, indent=2),
        "",
        "## C2",
        json.dumps(c2, ensure_ascii=False, indent=2),
        "",
        "## C3",
        json.dumps(c3, ensure_ascii=False, indent=2),
    ])
    OUT_PATTERN_V1_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": verdict["verdict"],
        "pos_cohorts": pos_cohorts,
        "adv_cohorts": adv_cohorts,
        "c1": c1.get("separates"),
        "excluded": excluded,
        "scope": verdict.get("may_enter_skill_scope"),
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_PATTERN_V1)
    print("wrote", OUT_PATTERN_V1_MD)
    return 0


def _store_fingerprint(store) -> dict[str, Any]:
    active = getattr(store, "active_path", None)
    payload = ""
    if active is not None and Path(active).is_file():
        payload = Path(active).read_text(encoding="utf-8")
    root = Path(getattr(store, "root"))
    names = sorted(p.name for p in root.glob("*") if p.is_dir()) if root.is_dir() else []
    return {
        "active_text": payload,
        "snapshot_dirs": names,
        "sha": hashlib.sha256(
            (payload + "\n" + "\n".join(names)).encode("utf-8")).hexdigest(),
    }


def _deploy_fast_only(
    *,
    rows: Mapping[str, Any],
    snapshot,
    origin: int,
    store_tag: str,
    agent_factory: Any,
    backend_factory: Any,
    llm_budget: int,
    program_override: str | None = None,
    wall: LabelWall | None = None,
) -> dict[str, Any]:
    """One cohort-level Fast decision.  Never open_delayed / Slow / Skill write."""
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod

    if wall is None:
        wall = LabelWall(released=False)
    store_root = Path(tempfile.gettempdir()) / store_tag
    if store_root.exists():
        shutil.rmtree(store_root)
    store = SnapshotStore(store_root / "snapshots")
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    before = _store_fingerprint(store)
    experience_before = []
    backend = backend_factory(llm_budget)
    # smoke/evaluate factories key windows by r1/r2; deploy reuses r2
    # support origin as the public prefix, never a new window name.
    agent = agent_factory(rows, backend, "r2")
    method = TTHAMethod(agent, snapshot, ())
    experience_before = [
        getattr(e, "episode_id", None)
        for e in (getattr(method, "experience_episodes", ()) or ())]
    spec = _source_task_spec()
    values = {uid: np.asarray(rows[uid]["values"], dtype=np.float64)
              for uid in rows}
    first = sorted(rows)[0]
    series0 = values[first]
    observed = dict(resolver.window_context(values, origin, PERIOD_HINT))
    observed["bound_period"] = float(PERIOD_HINT)
    request = PreparationRequest(
        "t6-deploy", series0[:origin], spec, dict(observed))
    method.bind_round_data(series0[:origin], task_kind="anomaly_detection")
    method.prepare(request, runtime_prior_slot=False)
    trace = method.last_trace
    chosen = getattr(trace, "chosen_candidate_id", None) or ""
    if program_override is not None:
        applied = program_override
        abstained = program_override == "identity"
        decision_source = "static_override"
    else:
        applied = chosen if chosen and chosen != "identity" else "identity"
        abstained = not chosen or chosen == "identity"
        decision_source = "fast_cohort"
    logs = []
    for uid in sorted(rows):
        logs.append({
            "series": uid,
            "scope": "cohort",
            "decision": applied,
            "abstain": abstained,
            "applied": applied,
        })
    after = _store_fingerprint(store)
    experience_after = [
        getattr(e, "episode_id", None)
        for e in (getattr(method, "experience_episodes", ()) or ())]
    llm_calls = int(getattr(backend, "calls", 0) or 0)
    breach = []
    if before["sha"] != after["sha"]:
        breach.append("store_hash_changed")
    if experience_before != experience_after:
        breach.append("experience_changed")
    if llm_calls and program_override is not None:
        breach.append("static_arm_spent_llm")
    result = {
        "open_delayed_calls": 0,
        "slow_calls": 0,
        "store_before": before,
        "store_after": after,
        "store_unchanged": before["sha"] == after["sha"],
        "experience_before": experience_before,
        "experience_after": experience_after,
        "experience_unchanged": experience_before == experience_after,
        "llm_calls": llm_calls,
        "chosen_raw": chosen,
        "applied_program": applied,
        "abstained": abstained,
        "decision_source": decision_source,
        "per_series": logs,
        "yahoo_touched": False,
        "breach": breach,
        "ok": not breach,
    }
    return result


def deploy_fast_only_smoke() -> int:
    """Existing Source fixture, 0 LLM.  Must not touch Yahoo."""
    if not OUT_JSON_V2.exists():
        print(json.dumps({"verdict": "INSTRUMENT_UNREADABLE",
                          "reason": "missing plan_v2"}, indent=1))
        return 1
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    wall = LabelWall(released=False)
    universe = _load_universe(gate_all(row_order_contract=True))
    rows = universe["source"][FIXTURE_SOURCE_COHORT]
    # one-cell smoke: first two series of the already-exposed Source cohort
    slim = {k: rows[k] for k in sorted(rows)[:2]}
    origin = min(int(slim[u]["windows"]["r2"]["support"][0]) for u in slim)
    h0 = compile_snapshot(
        PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
        verify_lock=False)
    run = _deploy_fast_only(
        rows=slim, snapshot=h0, origin=origin,
        store_tag="t6g_deploy_smoke",
        agent_factory=_smoke_agent_factory(("winsorize",)),
        backend_factory=_smoke_backend_factory(("winsorize",)),
        llm_budget=0,
        wall=wall,
    )
    yahoo_paths = list((PROJECT_ROOT / "data" / "benchmark_yahoo_s5_v1").rglob("*"))
    run["yahoo_files_present_but_unread"] = True
    run["label_wall"] = wall.audit()
    if wall.audit()["target_key_requests"]:
        run["breach"].append("target_wall_requested")
        run["ok"] = False
    verdict = "DEPLOY_FAST_ONLY_SMOKE_OK" if run["ok"] else "PROTOCOL_BREACH"
    payload = {
        "protocol_version": "t6_42g_deploy_fast_only_smoke_v1",
        "entry": "--deploy-fast-only-smoke",
        "fixture": "source_aws_cloudwatch first two series, r2 support origin",
        "yahoo_touched": False,
        "verdict": verdict,
        "run": run,
    }
    OUT_DEPLOY_SMOKE.write_text(_json_text(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "open_delayed": run["open_delayed_calls"],
        "slow": run["slow_calls"],
        "store_unchanged": run["store_unchanged"],
        "applied": run["applied_program"],
        "llm": run["llm_calls"],
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_DEPLOY_SMOKE)
    return 0 if run["ok"] else 1


def _yahoo_l1_windows(n: int) -> dict[str, Any]:
    plan: dict[str, Any] = {}
    for name, spans in L1_ROUNDS.items():
        row: dict[str, Any] = {}
        for part, (lo, hi) in spans.items():
            row[part] = [int(lo * n), int(hi * n)]
        plan[name] = row
    plan["heldout"] = [int(L1_HELDOUT[0] * n), n]
    return plan


def _load_yahoo_l1_roster() -> dict[str, Any]:
    if not YAHOO_FREEZE.exists():
        raise Stop("INSTRUMENT_UNREADABLE", "missing yahoo freeze list")
    freeze = json.loads(YAHOO_FREEZE.read_text(encoding="utf-8"))
    roster = sorted(freeze["roster"], key=lambda r: r["file"])[:L1_ROSTER_N]
    if len(roster) != L1_ROSTER_N:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "freeze roster shorter than 24: %d" % len(roster))
    work = PROJECT_ROOT / "data" / "benchmark_yahoo_s5_v1" / "work"
    vault_in = PROJECT_ROOT / "data" / "benchmark_yahoo_s5_v1" / "vaults" / "held_in"
    rows: dict[str, Any] = {}
    for rec in roster:
        path = work / rec["file"]
        if not path.is_file():
            raise Stop("INSTRUMENT_UNREADABLE", "missing work copy %s" % path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header[:2] != ["timestamp", "value"]:
                raise Stop("LAYOUT_UNEXPECTED_STOP",
                           "work header %s" % header)
            data = list(reader)
        values = np.asarray([float(r[1]) for r in data], dtype=np.float64)
        stamps = [r[0] for r in data]
        n = int(values.size)
        rows[rec["file"]] = {
            "values": values,
            "timestamps": stamps,
            "length": n,
            "nab_key": rec["nab_style_key"],
            "windows": _yahoo_l1_windows(n),
            "held_in_vault": vault_in / rec["file"],
            "freeze": rec,
        }
    return {"freeze": freeze, "rows": rows, "order": [r["file"] for r in roster]}


class YahooHeldInWall:
    """Release held-in vault labels one scoring window at a time.  No held-out."""

    def __init__(self, rows: Mapping[str, Any]) -> None:
        self._rows = rows
        self._cache: dict[tuple[str, int, int], list[list[int]]] = {}
        self.requests: list[dict[str, Any]] = []
        self.held_out_requests = 0

    def events_for(self, uid: str, lo: int, hi: int) -> list[list[int]]:
        rec = self._rows[uid]
        n = rec["length"]
        wall = rec["windows"]["heldout"][0]
        if lo >= wall or hi > wall:
            self.held_out_requests += 1
            self.requests.append({"uid": uid, "lo": lo, "hi": hi,
                                  "granted": False, "why": "held_out"})
            raise Stop("PROTOCOL_BREACH",
                       "held-out label requested for %s [%s,%s)" % (uid, lo, hi))
        key = (uid, int(lo), int(hi))
        if key in self._cache:
            self.requests.append({"uid": uid, "lo": lo, "hi": hi,
                                  "granted": True, "cached": True})
            return self._cache[key]
        path = rec["held_in_vault"]
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header[:3] != ["row_index", "timestamp", "is_anomaly"]:
                raise Stop("LAYOUT_UNEXPECTED_STOP",
                           "vault header %s" % header)
            flags = []
            for line in reader:
                idx = int(line[0])
                if lo <= idx < hi:
                    flags.append((idx, line[2].strip() in {"1", "1.0", "true", "True"}))
        # do not keep unused rows
        on = [idx for idx, flag in flags if flag]
        events = []
        if on:
            run = [on[0]]
            for idx in on[1:]:
                if idx == run[-1] + 1:
                    run.append(idx)
                else:
                    events.append(run)
                    run = [idx]
            events.append(run)
        self._cache[key] = events
        self.requests.append({"uid": uid, "lo": lo, "hi": hi,
                              "granted": True, "cached": False})
        return events

    def audit(self) -> dict[str, Any]:
        return {
            "held_out_requests": self.held_out_requests,
            "n_requests": len(self.requests),
            "windows_opened": sorted({
                (r["uid"], r["lo"], r["hi"]) for r in self.requests if r["granted"]
            }),
        }


class _YahooAdapter:
    """Same scoring arithmetic as the NAB adapter; Yahoo held-in vault truth."""

    def __init__(self, *, consumer: Any, rows: Mapping[str, Any],
                 round_name: str, wall: YahooHeldInWall, budget: FitBudget,
                 support_origin: int, delayed_origin: int) -> None:
        self._consumer = consumer
        self._rows = dict(rows)
        self._round = str(round_name)
        self._yahoo_wall = wall
        self._budget = budget
        self._support_origin = int(support_origin)
        self._delayed_origin = int(delayed_origin)
        self._models: dict[tuple[str, str], Any] = {}
        self.calls: list[dict[str, Any]] = []

    def _part_for(self, origin: int) -> str:
        return "support" if int(origin) < self._delayed_origin else "delayed"

    def _model(self, uid: str, signature: str, program_steps: Any) -> Any:
        key = (uid, signature)
        if key in self._models:
            return self._models[key]
        row = self._rows[uid]
        lo, hi = row["windows"][self._round]["train"]
        block = np.asarray(row["values"], dtype=np.float64)[lo:hi]
        if program_steps:
            result = run_pipeline(list(program_steps), block)
            if not result.ok or result.artifact is None:
                raise Stop("TARGET_FEEDBACK_UNREADABLE",
                           "program %s failed on a training block: %s"
                           % (signature, result.error))
            block = np.asarray(result.artifact, dtype=np.float64).ravel()
        self._budget.spend(1)
        model = self._consumer.fit_series(block)
        self._models[key] = model
        return model

    def __call__(self, roster, values, compiled, config, *, origin):
        steps = compiled_steps(compiled)
        signature = "|".join(op for op, _p in steps) or "identity"
        part = self._part_for(int(origin))
        per_view: list[float] = []
        scored: dict[str, Any] = {}
        behavior = 0
        for uid in sorted(self._rows):
            model = self._model(uid, signature, steps)
            lo, hi = self._rows[uid]["windows"][self._round][part]
            raw = np.asarray(self._rows[uid]["values"], dtype=np.float64)
            truth = self._yahoo_wall.events_for(uid, lo, hi)
            reading = self._consumer.score_series(model, raw, (lo, hi), truth)
            scored[uid] = reading
            per_view.append(float(reading["f1"]))
            behavior += int(model["training_windows"])
        macro = self._consumer.macro_f1(scored)
        out = {
            "mean_smase": -float(macro if macro is not None else 0.0),
            "per_view_smase": [-v for v in per_view],
            "behavior_point_count": int(behavior),
            "ad_macro_f1": float(macro) if macro is not None else None,
            "ad_f1_by_series": {uid: scored[uid]["f1"] for uid in scored},
            "part": part,
        }
        self.calls.append({"signature": signature, "part": part,
                           "origin": int(origin),
                           "macro_f1": out["ad_macro_f1"]})
        return out


def _score_heldout_static(rows: Mapping[str, Any], program: str,
                          budget: FitBudget) -> dict[str, Any]:
    """Fit on full held-in [0,0.7n) after applying program; score raw held-out."""
    consumer = _load_consumer()
    per: dict[str, Any] = {}
    for uid, rec in rows.items():
        n = rec["length"]
        cut = rec["windows"]["heldout"][0]
        raw = np.asarray(rec["values"], dtype=np.float64)
        train = _apply_program(raw[:cut], program)
        budget.spend(1)
        model = consumer.fit_series(train)
        # held-out labels loaded only here, by the offline evaluator
        vault = (PROJECT_ROOT / "data" / "benchmark_yahoo_s5_v1"
                 / "vaults" / "held_out" / uid)
        flags = []
        with vault.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for line in reader:
                flags.append((int(line[0]),
                              line[2].strip() in {"1", "1.0", "true", "True"}))
        on = [i for i, f in flags if f]
        events: list[list[int]] = []
        if on:
            run = [on[0]]
            for idx in on[1:]:
                if idx == run[-1] + 1:
                    run.append(idx)
                else:
                    events.append(run)
                    run = [idx]
            events.append(run)
        per[uid] = consumer.score_series(model, raw, (cut, n), events)
    macro = consumer.macro_f1(per)
    return {
        "program": program,
        "macro_f1": macro,
        "f1_by_series": {k: v["f1"] for k, v in per.items()},
    }


def l1_score_heldout_only(*, a3_program: str = "outlier_mad") -> int:
    """Instrument repair: score the already-run A3* without reopening held-in."""
    pack = _load_yahoo_l1_roster()
    rows = pack["rows"]
    budget = FitBudget(L1_AD_FIT_CAP)
    program = _canonical_menu_program(a3_program)
    ident = _score_heldout_static(rows, "identity", budget)
    hampel = _score_heldout_static(rows, "hampel_filter", budget)
    a3s = _score_heldout_static(rows, program, budget)
    best_static = max(ident["macro_f1"], hampel["macro_f1"])
    delta_id = float(a3s["macro_f1"] - ident["macro_f1"])
    delta_best = float(a3s["macro_f1"] - best_static)
    harmed = [uid for uid, f1 in a3s["f1_by_series"].items()
              if float(f1) - float(ident["f1_by_series"][uid]) < -0.005]
    worst = min(float(a3s["f1_by_series"][uid]) - float(ident["f1_by_series"][uid])
                for uid in a3s["f1_by_series"])
    hampel_harmed = [uid for uid, f1 in hampel["f1_by_series"].items()
                     if float(f1) - float(ident["f1_by_series"][uid]) < -0.005]
    if delta_best > 0.005 and len(harmed) <= 1 and worst >= -0.02:
        verdict = "ADAPTATION_DELIVERS_HELDOUT"
    elif (abs(delta_id) <= 0.005 and not harmed
          and (hampel["macro_f1"] - ident["macro_f1"] < -0.005
               or len(hampel_harmed) > 2)):
        verdict = "ADAPTATION_SAFETY_ONLY"
    elif abs(delta_best) <= 0.005 and not harmed:
        verdict = "ADAPTATION_TIE"
    elif delta_id < -0.005 or len(harmed) > 2:
        verdict = "ADAPTATION_HARMS_HELDOUT"
    else:
        verdict = "OBSERVED_BUT_UNCLASSIFIED"
    payload = {
        "protocol_version": "t6_42g_l1_static_vs_a3_v1",
        "entry": "--l1-score-heldout-only",
        "held_in_not_rerun": True,
        "a3_star_snapshot": "h0 (no learned skill on 20260824T002018Z store)",
        "a3_program_from_first_run_winner_alias": "extreme-deviation-mad",
        "a3_program": program,
        "roster": pack["order"],
        "heldout": {
            "identity": ident,
            "hampel_filter": hampel,
            "a3_star": a3s,
            "best_static_macro": best_static,
            "delta_vs_identity": delta_id,
            "delta_vs_best_static": delta_best,
            "harmed_vs_identity": harmed,
            "worst_delta_vs_identity": worst,
            "hampel_harmed_vs_identity": hampel_harmed,
        },
        "cost": {"llm": 0, "ad_fits": budget.used, "ad_fit_cap": budget.cap,
                 "forecast_retrains": 0,
                 "note": "held-in LLM/fit of the crashed first run are not in this file"},
        "verdict": {"verdict": verdict},
        "instrument_note": (
            "first official run completed held-in + Fast-only deploy then "
            "crashed mapping winner op extreme-deviation-mad onto the menu; "
            "this scores Part D only"
        ),
    }
    OUT_L1.write_text(_json_text(payload), encoding="utf-8")
    OUT_L1_MD.write_text(
        "# #42g L1 Static vs A3\n\nverdict: **%s**\n\n"
        "A3* program (mapped): `%s` from `extreme-deviation-mad`\n\n"
        "macro F1 identity %s / hampel %s / A3* %s\n"
        "Δ vs identity %s; Δ vs best-static %s; harmed %s; worst %s\n"
        % (verdict, program, ident["macro_f1"], hampel["macro_f1"],
           a3s["macro_f1"], delta_id, delta_best, harmed, worst),
        encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "a3_program": program,
        "fits": budget.used,
        "macro": {"identity": ident["macro_f1"],
                  "hampel": hampel["macro_f1"],
                  "a3": a3s["macro_f1"]},
        "delta_id": delta_id, "delta_best": delta_best,
        "harmed": harmed, "worst": worst,
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_L1)
    return 0


def l1_static_vs_a3() -> int:
    """#42g r1 L1: A3 held-in two rounds, then Fast-only held-out vs Static."""
    leftover = [
        line for line in os.popen("ps -ef").read().splitlines()
        if "run_e2_t6_natural_a5_a3.py" in line
        and "--l1-static-vs-a3" in line
        and str(os.getpid()) not in line
    ]
    if leftover:
        print(json.dumps({"verdict": "CONCURRENT_RUN_BLOCKED",
                          "reason": leftover}, indent=1))
        return 2
    L1_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(str(L1_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(json.dumps({"verdict": "CONCURRENT_RUN_BLOCKED",
                          "reason": "lock held"}, indent=1))
        return 2
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    if not OUT_DEPLOY_SMOKE.exists():
        print(json.dumps({"verdict": "INSTRUMENT_UNREADABLE",
                          "reason": "smoke artifact missing"}, indent=1))
        return 1
    smoke = json.loads(OUT_DEPLOY_SMOKE.read_text(encoding="utf-8"))
    if smoke.get("verdict") != "DEPLOY_FAST_ONLY_SMOKE_OK":
        print(json.dumps({"verdict": "INSTRUMENT_UNREADABLE",
                          "reason": "smoke not green"}, indent=1))
        return 1
    try:
        pack = _load_yahoo_l1_roster()
    except Stop as stop:
        print(json.dumps({"verdict": stop.verdict, "reason": stop.reason},
                         indent=1))
        return 1
    rows = pack["rows"]
    wall = YahooHeldInWall(rows)
    budget = FitBudget(L1_AD_FIT_CAP)
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
        activate_approved, open_delayed, run_online_round,
    )
    h0 = compile_snapshot(
        PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
        verify_lock=False)
    store_root = Path(tempfile.gettempdir()) / ("t6g_l1_%s" % run_id)
    if store_root.exists():
        shutil.rmtree(store_root)
    store = SnapshotStore(store_root / "snapshots")
    store.materialize(h0)
    store.set_active(h0.runtime_bundle_sha)
    backend = _evaluate_backend(L1_LLM_CAP)
    agent = _evaluate_agent(rows, backend, "r1")
    method = TTHAMethod(agent, h0, ())
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter())
    consumer = _load_consumer()
    cells = []
    stopped = None
    try:
        for round_name in ("r1", "r2"):
            support_origin = min(
                rec["windows"][round_name]["support"][0] for rec in rows.values())
            delayed_origin = min(
                rec["windows"][round_name]["delayed"][0] for rec in rows.values())
            adapter = _YahooAdapter(
                consumer=consumer, rows=rows, round_name=round_name,
                wall=wall, budget=budget,
                support_origin=support_origin, delayed_origin=delayed_origin)
            executor = _NABScopeExecutor(
                rows=rows, round_name=round_name, evaluate_fn=adapter)
            values = {uid: rec["values"] for uid, rec in rows.items()}
            first = pack["order"][0]
            series0 = values[first]
            spec = _source_task_spec()
            observed = dict(resolver.window_context(
                values, support_origin, PERIOD_HINT))
            observed["bound_period"] = float(PERIOD_HINT)
            request = PreparationRequest(
                "t6-l1-yahoo", series0[:support_origin], spec, dict(observed))
            features = dict(extract_public_features(
                series0[:support_origin], task_kind="anomaly_detection"))
            method.bind_round_data(
                series0[:support_origin], task_kind="anomaly_detection")
            result = run_online_round(
                method, executor, request, values,
                origin=support_origin, slow_agent=None,
                controller=controller, store=store,
                card_builder=_card_builder_for("anomaly_detection"),
                round_name="a3_%s" % round_name,
                budget=2, allow_slow=False,
                domain="yahoo_s5_a1", period=PERIOD_HINT,
                fast_features=features,
                allow_fast_skill=True, runtime_prior_slot=False)
            open_delayed(result, executor,
                         delayed_origin=delayed_origin, store=store)
            activated = False
            if result.approved_skill_id is not None:
                activated = activate_approved(result, store)
            cells.append({
                "round": round_name,
                "chosen": getattr(method.last_trace, "chosen_candidate_id", None),
                "pool": list(getattr(method.last_trace, "candidate_ids", ()) or ()),
                "winner": _plain(result.winner_program),
                "winner_menu": _canonical_menu_program(
                    (result.winner_program or [{}])[0].get("op")
                    if result.winner_program else "identity"),
                "probes": [{"candidate_id": p.get("candidate_id"),
                            "kind": p.get("kind"), "gain": p.get("gain")}
                           for p in result.actual_probed_programs],
                "abstained": bool(getattr(result, "abstained", False)),
                "approved_skill_id": result.approved_skill_id,
                "activated": activated,
                "harm_count": result.harm_count,
                "delayed_utility": result.delayed_utility,
                "adapter_calls": list(adapter.calls),
                "llm_after": int(getattr(backend, "calls", 0)),
                "fits_after": budget.used,
            })
            if budget.used >= budget.cap:
                stopped = "CONSUMER_FIT_BUDGET_EXCEEDED"
                break
    except Stop as stop:
        stopped = stop.verdict
        stop_reason = stop.reason
    else:
        stop_reason = None

    a3_sha = store.active_path.read_text(encoding="utf-8") if store.active_path.is_file() else ""
    # Part B freeze record
    freeze_doc = {
        "a3_store_root": str(store_root),
        "a3_active": a3_sha,
        "static": ["identity", "hampel_filter"],
        "cells": cells,
        "stopped": stopped,
    }
    if stopped in {
        "PROTOCOL_BREACH", "CONSUMER_FIT_BUDGET_EXCEEDED",
        "TARGET_FEEDBACK_UNREADABLE", "INCOMPLETE_LLM_BUDGET",
        "INSTRUMENT_UNREADABLE",
    }:
        payload = {
            "protocol_version": "t6_42g_l1_static_vs_a3_v1",
            "run_id": run_id,
            "verdict": {"verdict": stopped, "reason": stop_reason},
            "held_in": cells,
            "wall": wall.audit(),
            "cost": {"llm": int(getattr(backend, "calls", 0)),
                     "ad_fits": budget.used, "ad_fit_cap": budget.cap},
        }
        OUT_L1.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        return 1

    # Part C: Fast-only on held-in prefix (origin = 0.7n min) then apply
    # the frozen decision as training program for held-out scoring.
    origin = min(rec["windows"]["heldout"][0] for rec in rows.values())
    deploy = _deploy_fast_only(
        rows=rows, snapshot=method._active_snapshot(),
        origin=origin, store_tag="t6g_l1_deploy_%s" % run_id,
        agent_factory=_evaluate_agent,
        backend_factory=lambda cap: backend,
        llm_budget=max(0, L1_LLM_CAP - int(getattr(backend, "calls", 0))),
        wall=LabelWall(released=False),
    )
    if not deploy["ok"]:
        payload = {
            "protocol_version": "t6_42g_l1_static_vs_a3_v1",
            "run_id": run_id,
            "verdict": {"verdict": "PROTOCOL_BREACH",
                        "reason": deploy["breach"]},
            "deploy": deploy,
            "held_in": cells,
        }
        OUT_L1.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        return 1

    a3_program = "identity"
    winner = cells[-1].get("winner") if cells else None
    if isinstance(winner, list) and winner:
        a3_program = _canonical_menu_program(winner[0].get("op") or "identity")
    else:
        a3_program = _canonical_menu_program(deploy.get("applied_program") or "identity")

    # Part D: offline evaluator opens held-out vault once
    try:
        ident = _score_heldout_static(rows, "identity", budget)
        hampel = _score_heldout_static(rows, "hampel_filter", budget)
        a3s = _score_heldout_static(rows, a3_program, budget)
    except Stop as stop:
        payload = {
            "protocol_version": "t6_42g_l1_static_vs_a3_v1",
            "run_id": run_id,
            "verdict": {"verdict": stop.verdict, "reason": stop.reason},
            "held_in": cells,
            "deploy": {
                "open_delayed": deploy["open_delayed_calls"],
                "slow": deploy["slow_calls"],
                "store_unchanged": deploy["store_unchanged"],
                "applied": deploy["applied_program"],
            },
            "a3_program_resolved": a3_program,
            "wall": wall.audit(),
            "cost": {"llm": int(getattr(backend, "calls", 0)),
                     "ad_fits": budget.used, "ad_fit_cap": budget.cap},
        }
        OUT_L1.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        return 1
    best_static = max(ident["macro_f1"], hampel["macro_f1"])
    delta_id = float(a3s["macro_f1"] - ident["macro_f1"])
    delta_best = float(a3s["macro_f1"] - best_static)
    harmed = [uid for uid, f1 in a3s["f1_by_series"].items()
              if float(f1) - float(ident["f1_by_series"][uid]) < -0.005]
    worst = min(float(a3s["f1_by_series"][uid]) - float(ident["f1_by_series"][uid])
                for uid in a3s["f1_by_series"])
    hampel_harmed = [uid for uid, f1 in hampel["f1_by_series"].items()
                     if float(f1) - float(ident["f1_by_series"][uid]) < -0.005]
    n = float(len(a3s["f1_by_series"]))
    if delta_best > 0.005 and len(harmed) <= 1 and worst >= -0.02:
        verdict = "ADAPTATION_DELIVERS_HELDOUT"
    elif (abs(delta_id) <= 0.005 and not harmed
          and (hampel["macro_f1"] - ident["macro_f1"] < -0.005
               or len(hampel_harmed) > 2)):
        verdict = "ADAPTATION_SAFETY_ONLY"
    elif abs(delta_best) <= 0.005 and not harmed:
        verdict = "ADAPTATION_TIE"
    elif delta_id < -0.005 or len(harmed) > 2:
        verdict = "ADAPTATION_HARMS_HELDOUT"
    else:
        verdict = "OBSERVED_BUT_UNCLASSIFIED"
    payload = {
        "protocol_version": "t6_42g_l1_static_vs_a3_v1",
        "entry": "--l1-static-vs-a3",
        "run_id": run_id,
        "roster": pack["order"],
        "held_in_cells": cells,
        "freeze": freeze_doc,
        "deploy": {
            "open_delayed": deploy["open_delayed_calls"],
            "slow": deploy["slow_calls"],
            "store_unchanged": deploy["store_unchanged"],
            "applied": deploy["applied_program"],
            "per_series_n": len(deploy["per_series"]),
            "llm": deploy["llm_calls"],
        },
        "heldout": {
            "identity": ident,
            "hampel_filter": hampel,
            "a3_star": a3s,
            "a3_program": a3_program,
            "best_static_macro": best_static,
            "delta_vs_identity": delta_id,
            "delta_vs_best_static": delta_best,
            "harmed_vs_identity": harmed,
            "worst_delta_vs_identity": worst,
            "hampel_harmed_vs_identity": hampel_harmed,
        },
        "wall": wall.audit(),
        "cost": {
            "llm": int(getattr(backend, "calls", 0)),
            "llm_cap": L1_LLM_CAP,
            "ad_fits": budget.used,
            "ad_fit_cap": L1_AD_FIT_CAP,
            "forecast_retrains": 0,
        },
        "verdict": {"verdict": verdict},
    }
    OUT_L1.write_text(_json_text(payload), encoding="utf-8")
    OUT_L1_MD.write_text(
        "# #42g L1 Static vs A3\n\nverdict: **%s**\n\n"
        "roster 24; LLM %s/%s; fit %s/%s\n\n"
        "A3* program: `%s`\n\nmacro F1 identity %s / hampel %s / A3* %s\n"
        "Δ vs identity %s; Δ vs best-static %s; harmed %s; worst %s\n"
        % (verdict, payload["cost"]["llm"], L1_LLM_CAP,
           budget.used, L1_AD_FIT_CAP, a3_program,
           ident["macro_f1"], hampel["macro_f1"], a3s["macro_f1"],
           delta_id, delta_best, harmed, worst),
        encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "llm": payload["cost"]["llm"],
        "fits": budget.used,
        "a3_program": a3_program,
        "macro": {"identity": ident["macro_f1"],
                  "hampel": hampel["macro_f1"],
                  "a3": a3s["macro_f1"]},
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_L1)
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--deploy-fast-only-smoke" in argv:
        return deploy_fast_only_smoke()
    if "--deploy-fast-only" in argv:
        print(json.dumps({
            "verdict": "DEPLOYMENT_GRANULARITY_UNSUPPORTED",
            "reason": "official Yahoo held-out must go through l1_main after smoke",
        }, indent=1))
        return 2
    if "--l1-score-heldout-only" in argv:
        return l1_score_heldout_only()
    if "--l1-static-vs-a3" in argv:
        return l1_static_vs_a3()
    if "--pattern-discriminator-v1" in argv:
        return pattern_discriminator_v1()
    if "--accept-skill-v3" in argv:
        return accept_skill_v3()
    if "--source-expansion-v3" in argv:
        return source_expansion_v3()
    if "--replay-skill-v2" in argv:
        return replay_skill_v2()
    if "--evaluate-lifecycle-fixture" in argv:
        cap = FIXTURE_AD_FIT_BUDGET
        for token in argv:
            if token.startswith("--fit-cap="):
                cap = int(token.split("=", 1)[1])
        return evaluate_lifecycle_fixture(fit_cap=cap)
    if "--evaluate-smoke" in argv:
        cap = SMOKE_AD_FIT_BUDGET
        target = None
        for token in argv:
            if token.startswith("--fit-cap="):
                cap = int(token.split("=", 1)[1])
                target = OUT_SMOKE_BUDGET
        return evaluate_smoke(fit_cap=cap, out_path=target)
    if "--evaluate" in argv:
        return evaluate()
    if "--plan-v2" in argv:
        return plan(row_order_contract=True)
    if "--plan" in argv:
        return plan()
    print("usage: --plan | --plan-v2 | --evaluate | --evaluate-smoke "
          "[--fit-cap=N] | --evaluate-lifecycle-fixture | --replay-skill-v2 "
          "| --source-expansion-v3 | --accept-skill-v3 "
          "| --pattern-discriminator-v1 | --deploy-fast-only-smoke "
          "| --l1-static-vs-a3")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
