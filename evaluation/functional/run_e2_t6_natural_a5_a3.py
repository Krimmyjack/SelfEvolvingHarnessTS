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


def _program_steps(program: str) -> tuple[tuple[str, dict], ...]:
    if program == "identity":
        return ()
    if program not in OPERATOR_METADATA:
        raise Stop("NATURAL_DATA_SHAPE_INELIGIBLE",
                   "menu entry %r is not a registered operator" % program)
    return ((program, {}),)


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
            record.update({
                "pool": list(getattr(trace, "candidate_ids", ()) or ()),
                "chosen": getattr(trace, "chosen_candidate_id", None),
                "memory_resolution": getattr(
                    trace, "memory_resolution_status", None),
                "proposal_count": result.proposal_count,
                "support_receipts": result.target_support_receipts_used,
                "non_identity_trials": len(non_identity),
                "probes": [{"candidate_id": p["candidate_id"],
                            "kind": p.get("kind"), "gain": p.get("gain")}
                           for p in result.actual_probed_programs],
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


def main() -> int:
    argv = sys.argv[1:]
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
          "[--fit-cap=N] | --evaluate-lifecycle-fixture")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
