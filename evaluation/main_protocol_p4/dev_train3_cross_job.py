"""DEV-TRAIN-3: cross-job knowledge reuse and Target adaptation (one runner).

Course::

    KDD formation material (TRAIN2 10/10) -> Slow: Source-derived guidance
    -> Beijing job J1 (b=3600): held-in adaptation q1..q4 -> freeze
    -> Slow: integrate J1 held-in experience (A5 only)
    -> Beijing job J2 (b=10800): held-in adaptation q1..q4 -> freeze
    -> external E scoring after every E output of every arm is frozen

Inside every arm and every query the organisation is the TRAIN2 one: one W
per training series -> pooled training material -> one shared Ridge -> one V
per prediction series -> predict with that shared model.

Reused, not re-implemented:
  dev_train1_evaluator   numeric kernel (design / fit / predict / score)
  dev_train1_runner      Fast decision, batch checkpoints, Budget, Slow
                         boundary, legality, model freeze
  dev_train2_workflow    seven-workflow menu, cap=1.0 wiring
  dev_train1b_observations.execution_observations   window packets

This module does not edit canonical h0, operators, the Consumer, scoring, or
any historical result.  Fast never receives future masks, truth, scores,
or another arm's decisions.  All indices below are job-local; absolute
index = block_start + local index.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import multiprocessing as _mp
import os
import random
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import seasonal_scale
from SelfEvolvingHarnessTS.methods.ttha.agent_core import AgentRole
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view
from SelfEvolvingHarnessTS.runtime.agent_backend import AgentCallBudgetExceeded
from SelfEvolvingHarnessTS.runtime.candidate_verification import verify_candidate
from evaluation.main_protocol_p4 import dev_seq2_knowledge as know
from evaluation.main_protocol_p4 import dev_train1_evaluator as numeric
from evaluation.main_protocol_p4 import dev_train1_runner as R
from evaluation.main_protocol_p4 import dev_train2_workflow as W
from evaluation.main_protocol_p4 import per_sequence as ps
from evaluation.main_protocol_p4.dev_train1b_observations import execution_observations

ROOT = R.ROOT
PACKAGE = "DEV-TRAIN-3"
SEED = 20260911
WORK_ROOT = ROOT / "_scratch" / "dev_train3_cross_job"
SOURCE_RUN = R.RUNS_ROOT / "dev_train2_uncapped_workflow_10x10_20260910"
DATA_DIR = ROOT / "_scratch" / "train3_preflight" / "data" / "prsa" / "PRSA_Data_20130301-20170228"
CSV_NAME = "PRSA_Data_%s_20130301-20170228.csv"
COLUMN = "PM2.5"
CSV_START = _dt.datetime(2013, 3, 1, 0)

STATIONS = ("Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng")
# Fast/Slow see neutral series ids; dataset / site names never enter a prompt.
UIDS = tuple("s%02d" % (index + 1) for index in range(len(STATIONS)))
STATION_BY_UID = dict(zip(UIDS, STATIONS))

CONTEXT_LENGTH = int(numeric.CONTEXT_LENGTH)
HORIZON = int(numeric.HORIZON)
PERIOD = 24
TRAIN_PREFIX_END = 1800
ANCHORS = tuple(range(1212, 1753, 60))
BLOCKS = {"C_A": (2040, 2280), "C_B": (2520, 2760), "E": (3240, 3480)}
LOCAL_LEN = BLOCKS["E"][-1] + HORIZON              # 3528
JOBS = {"J1": 3600, "J2": 10800}
MAX_ABS_ROWS = max(JOBS.values()) + LOCAL_LEN       # 14328 (exclusive), before 2016
TRUTH_MIN_FINITE = 32                               # task-specific availability rule
SCALE_MIN_PAIRS = 32                                # scorer's historical seasonal pairs
MATERIAL = float(R.MATERIAL)

MENU_LABELS = tuple(W.FIXED_WORKFLOW_LABELS)
SOURCE_FIXED_CANDIDATES = tuple(label for label in MENU_LABELS if label != "identity")

LLM_ARMS = ("F", "A3", "A5")
SEARCH_ARMS = ("source_order", "random")
FIXED_ARMS = ("static", "source_fixed")
QUERIES = ("q1", "q2", "q3", "q4")

UNIT_MAX_API = 650
UNIT_MAX_TOKENS = 4_500_000
UNIT_MAX_ACTIVE_SECONDS = 2 * 3600
PKG_MAX_API = 4400
PKG_MAX_TOKENS = 30_000_000
PKG_MAX_FITS = 100
PKG_MAX_WALL_SECONDS = 12 * 3600
GLOBAL_FAST_SESSIONS = 4
PER_DECISION_CAP = int(R.PER_SEQUENCE_LLM_CAP)
VERIFY_TIME_LIMIT_SECONDS = 60
SLOW_ATTEMPTS = int(R.SLOW_ATTEMPTS)

_EFFECT_WORDS = (
    "smase", "score", "loss", "gain", "utility", "better", "worse", "improv",
    "harm", "degrad", "outperform", "best", "winner", "static", "baseline",
    "reduc", "increas", "lower", "higher", "beat", "error", "accura",
)

# ---------------------------------------------------------------------------
# capability statement: instrument fix shared by every LLM arm, not learning
# ---------------------------------------------------------------------------

RUNTIME_CAPABILITIES = {
    "phase_semantics": {
        "adaptation": (
            "during held-in adaptation the Runtime executes a small number of "
            "budgeted whole-job contrasts (fit one shared model, score one "
            "block); a Slow/research role reads that real feedback at batch "
            "boundaries.  The Fast role deciding one sequence never receives "
            "the downstream effect of the object it is currently deciding."
        ),
        "deployment": (
            "after freeze, prediction inputs are prepared sequence by sequence "
            "with no feedback, no refit and no knowledge change."
        ),
    },
    "probe_fields": {
        "fixed_probe_panel": "absent",
        "imputation_probe_direction": "unavailable/unknown",
        "clipping_probe_direction": "unavailable/unknown",
        "denoising_probe_direction": "unavailable/unknown",
        "level_probe_direction": "unavailable/unknown",
        "unknown_means": "not measured; it is neither positive nor negative evidence",
        "no_fast_tool_returns_a_probe": True,
    },
    "roles": {
        "train_workflow": (
            "observe the whole legal training prefix; the chosen W is replayed on "
            "every 240-point historic window [anchor-192, anchor+48); both the 192 "
            "context points and the 48 historic target points may be prepared; "
            "one shared linear model is then fitted from all sequences' prepared "
            "windows."
        ),
        "predict_input": (
            "observe and prepare only the last 192 points before the origin; the "
            "shared model is frozen and cannot be refitted; evaluation truth is "
            "raw and never modified."
        ),
    },
    "available_fast_tools": ["summarize_series", "localize_regions"],
    "verify_means": "deterministic legality / trial execution, not utility",
    "identity_is_legal": True,
    "any_legal_1_to_4_step_workflow_is_allowed": True,
    "no_mandated_operator_threshold_program_count_nonidentity_rate_or_recommendation": True,
    "effects_never_guaranteed": True,
    "modified_fraction_cap": 1.0,
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def steps_label(steps: Any) -> str:
    if steps is None:
        return "identity"
    return ps.program_label_with_params(ps.normalise_steps(tuple(steps)))


def menu_steps(label: str) -> Any:
    value = W.workflow_steps(label, PERIOD)
    if value == "UNAVAILABLE":
        raise RuntimeError("menu workflow unavailable: " + label)
    return value


def known_candidate_menu() -> list[dict[str, Any]]:
    return [
        {"label": label, "typed_steps": R.steps_to_json(menu_steps(label))}
        for label in MENU_LABELS
    ]


def fmean_or_unknown(values: Sequence[Any]) -> Any:
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numbers or len(numbers) != len(list(values)):
        return "UNKNOWN"
    return float(statistics.fmean(numbers))


def filter_for_fast(text: Any) -> str:
    """Only investigation purpose reaches Fast; numbers / effect words are withheld."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if re.search(r"\d", raw) or any(word in lowered for word in _EFFECT_WORDS):
        return "[withheld by runtime: the text contained numbers or effect language]"
    return raw[:300]


def random_order(seed: int) -> list[str]:
    rng = random.Random(int(seed))
    order = list(SOURCE_FIXED_CANDIDATES)
    rng.shuffle(order)
    return order


# ---------------------------------------------------------------------------
# time-restricted reader and job views
# ---------------------------------------------------------------------------

def read_station_prefix(path: Path, n_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse only the first ``n_rows`` rows.  Later rows are never parsed."""
    values: list[float] = []
    continuity_ok = True
    first_bad: Any = None
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= int(n_rows):
                break
            expected = CSV_START + _dt.timedelta(hours=index)
            try:
                stamp = _dt.datetime(int(row["year"]), int(row["month"]), int(row["day"]), int(row["hour"]))
            except Exception:  # noqa: BLE001
                stamp = None
            if stamp != expected and continuity_ok:
                continuity_ok = False
                first_bad = {"row": index, "expected": expected.isoformat(), "got": str(stamp)}
            raw = (row.get(COLUMN) or "").strip()
            try:
                value = float(raw)
            except ValueError:
                value = float("nan")
            values.append(value)
    array = np.asarray(values, dtype=np.float64)
    receipt = {
        "rows_parsed": int(array.size),
        "rows_requested": int(n_rows),
        "continuity_ok": bool(continuity_ok and array.size == int(n_rows)),
        "first_discontinuity": first_bad,
        "last_parsed_timestamp": (CSV_START + _dt.timedelta(hours=int(array.size) - 1)).isoformat(),
        "tail_rows_not_parsed": True,
        "missing_fraction_in_parsed_prefix": float(np.mean(~np.isfinite(array))) if array.size else None,
    }
    return array, receipt


def load_job_views(data_dir: Path = DATA_DIR) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Read the shared prefix once; return per-job local views and a receipt."""
    full: dict[str, np.ndarray] = {}
    receipts: dict[str, Any] = {}
    for uid in UIDS:
        station = STATION_BY_UID[uid]
        array, receipt = read_station_prefix(Path(data_dir) / (CSV_NAME % station), MAX_ABS_ROWS)
        if not receipt["continuity_ok"]:
            raise RuntimeError("time column discontinuity in %s: %s" % (station, receipt["first_discontinuity"]))
        full[uid] = array
        receipts[uid] = {"station": station, **receipt}
    views: dict[str, dict[str, np.ndarray]] = {}
    for job, start in JOBS.items():
        views[job] = {uid: full[uid][start:start + LOCAL_LEN].copy() for uid in UIDS}
        for uid in UIDS:
            if views[job][uid].size != LOCAL_LEN:
                raise RuntimeError("job %s uid %s local view is not %d points" % (job, uid, LOCAL_LEN))
    data_receipt = {
        "csv_hour_index_origin": CSV_START.isoformat(),
        "rows_parsed_per_station": int(MAX_ABS_ROWS),
        "max_absolute_index_exclusive": int(MAX_ABS_ROWS),
        "before_2016": True,
        "stations": receipts,
        "jobs": {job: {"block_start": int(start), "local_length": int(LOCAL_LEN)} for job, start in JOBS.items()},
        "uid_map": dict(STATION_BY_UID),
    }
    return views, data_receipt


def origin_eligibility(values: Mapping[str, np.ndarray], origin: int) -> dict[str, Any]:
    """Availability only: future finite count and a computable historical scale."""
    origin = int(origin)
    per_uid: dict[str, Any] = {}
    ok = True
    for uid in UIDS:
        raw = np.asarray(values[uid], dtype=np.float64)
        future = raw[origin:origin + HORIZON]
        finite = int(np.isfinite(future).sum()) if future.size == HORIZON else 0
        scale_ok = True
        scale_why = None
        try:
            seasonal_scale(raw[:origin], np.isfinite(raw[:origin]), period=PERIOD, min_pairs=SCALE_MIN_PAIRS)
        except Exception as exc:  # noqa: BLE001
            scale_ok = False
            scale_why = type(exc).__name__
        row = {"future_finite": finite, "future_ok": finite >= TRUTH_MIN_FINITE,
               "history_scale_ok": scale_ok, "history_scale_why": scale_why}
        per_uid[uid] = row
        if not (row["future_ok"] and scale_ok):
            ok = False
    return {"origin": origin, "eligible": bool(ok), "per_uid": per_uid,
            "rule": "all six series: >= %d finite future points AND historical seasonal scale with min_pairs=%d"
                    % (TRUTH_MIN_FINITE, SCALE_MIN_PAIRS)}


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    block_start: int
    eligible: dict[str, list[int]]
    eligibility: dict[str, Any]
    scoreable: bool

    def origins(self, block: str) -> list[int]:
        return list(self.eligible.get(block) or [])


def build_job_spec(job_id: str, values: Mapping[str, np.ndarray]) -> JobSpec:
    eligible: dict[str, list[int]] = {}
    eligibility: dict[str, Any] = {}
    scoreable = True
    for block, origins in BLOCKS.items():
        rows = [origin_eligibility(values, origin) for origin in origins]
        eligibility[block] = rows
        eligible[block] = [int(row["origin"]) for row in rows if row["eligible"]]
        if not eligible[block]:
            scoreable = False
    return JobSpec(job_id=str(job_id), block_start=int(JOBS[job_id]), eligible=eligible,
                   eligibility=eligibility, scoreable=scoreable)


def run_spec_for(job: JobSpec, *, live: bool) -> R.RunSpec:
    formation = tuple((index + 1, origin) for index, origin in enumerate(job.origins("C_A") + job.origins("C_B")))
    evaluation = tuple((index + 1, origin) for index, origin in enumerate(job.origins("E")))
    return R.RunSpec(
        train_uids=list(UIDS), eval_uids=list(UIDS), anchors=ANCHORS, period=PERIOD,
        formation_units=formation or ((0, 0),), eval_units=evaluation or ((0, 0),),
        training_prefix_end=TRAIN_PREFIX_END, concurrency=GLOBAL_FAST_SESSIONS,
        max_llm=PKG_MAX_API, max_fits=PKG_MAX_FITS, max_wall_seconds=PKG_MAX_WALL_SECONDS,
        live_api=bool(live), maximum_modified_fraction=1.0,
    )


# ---------------------------------------------------------------------------
# package state: budget, token ledger, orchestration
# ---------------------------------------------------------------------------

class UnitCeiling(R.PackageCeiling):
    """One LLM arm x job crossed its API / token / active-time ceiling."""


@dataclass
class TokenLedger:
    path: Path
    lock: threading.Lock = field(default_factory=threading.Lock)
    prompt_by_arm: dict[str, int] = field(default_factory=dict)
    completion_by_arm: dict[str, int] = field(default_factory=dict)
    calls_by_arm: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "TokenLedger":
        ledger = cls(path=Path(path))
        if ledger.path.is_file():
            state = R.read_json(ledger.path)
            ledger.prompt_by_arm = {str(k): int(v) for k, v in (state.get("prompt_by_arm") or {}).items()}
            ledger.completion_by_arm = {str(k): int(v) for k, v in (state.get("completion_by_arm") or {}).items()}
            ledger.calls_by_arm = {str(k): int(v) for k, v in (state.get("calls_by_arm") or {}).items()}
        return ledger

    def add(self, arm: str, prompt: int, completion: int) -> None:
        with self.lock:
            self.prompt_by_arm[arm] = self.prompt_by_arm.get(arm, 0) + int(prompt or 0)
            self.completion_by_arm[arm] = self.completion_by_arm.get(arm, 0) + int(completion or 0)
            self.calls_by_arm[arm] = self.calls_by_arm.get(arm, 0) + 1
            R.atomic_write_json(self.path, self.to_state())

    def tokens(self, prefix: str = "") -> int:
        total = 0
        for arm in set(self.prompt_by_arm) | set(self.completion_by_arm):
            if arm.startswith(prefix):
                total += self.prompt_by_arm.get(arm, 0) + self.completion_by_arm.get(arm, 0)
        return int(total)

    def to_state(self) -> dict[str, Any]:
        return {
            "prompt_by_arm": dict(self.prompt_by_arm),
            "completion_by_arm": dict(self.completion_by_arm),
            "calls_by_arm": dict(self.calls_by_arm),
            "total_tokens": self.tokens(),
        }


class CourseState:
    """Process-local shared state for one course run."""

    def __init__(self, course_dir: Path, *, live: bool) -> None:
        self.course_dir = Path(course_dir)
        self.live = bool(live)
        self.budget = R.Budget.load(
            self.course_dir / "budget.json",
            max_llm=PKG_MAX_API, max_fits=PKG_MAX_FITS, max_wall_seconds=PKG_MAX_WALL_SECONDS,
        )
        self.budget.path = self.course_dir / "budget.json"
        self.budget.concurrency_used = GLOBAL_FAST_SESSIONS
        self.budget.concurrency_note = "one global %d-active-Fast semaphore across arms and jobs; one fit writer" % GLOBAL_FAST_SESSIONS
        self.budget.persist()
        self.tokens = TokenLedger.load(self.course_dir / "tokens.json")
        self.fast_gate = threading.Semaphore(GLOBAL_FAST_SESSIONS)
        self.fit_lock = threading.RLock()
        self.state_lock = threading.Lock()
        self.live_fast = 0
        self.peak_fast = 0
        self.batch_context: dict[str, dict[str, Any]] = {}
        self.unit_active: dict[str, float] = {}
        self.account_fault: BaseException | None = None
        self.llm_log_dir = self.course_dir / "llm_log"
        self.llm_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_lock = threading.Lock()
        self.machinery: dict[str, Any] | None = None

    def unit_prefix(self, job: str, arm: str) -> str:
        return "%s/%s/" % (job, arm)

    def unit_api(self, job: str, arm: str) -> int:
        prefix = self.unit_prefix(job, arm)
        return int(sum(v for k, v in self.budget.llm_by_arm.items() if k.startswith(prefix)))

    def unit_tokens(self, job: str, arm: str) -> int:
        return self.tokens.tokens(self.unit_prefix(job, arm))

    def unit_active_seconds(self, job: str, arm: str) -> float:
        prefix = self.unit_prefix(job, arm)
        return float(sum(v for k, v in self.unit_active.items() if k.startswith(prefix)))

    def check_unit(self, job: str, arm: str, *, extra_api: int = 0) -> None:
        if self.unit_api(job, arm) + int(extra_api) > UNIT_MAX_API:
            raise UnitCeiling("unit %s/%s API ceiling %d" % (job, arm, UNIT_MAX_API))
        if self.unit_tokens(job, arm) > UNIT_MAX_TOKENS:
            raise UnitCeiling("unit %s/%s token ceiling %d" % (job, arm, UNIT_MAX_TOKENS))
        if self.unit_active_seconds(job, arm) / GLOBAL_FAST_SESSIONS > UNIT_MAX_ACTIVE_SECONDS:
            raise UnitCeiling("unit %s/%s active-time ceiling %ds (slot-normalised)" % (job, arm, UNIT_MAX_ACTIVE_SECONDS))
        if self.tokens.tokens() > PKG_MAX_TOKENS:
            raise R.PackageCeiling("package token ceiling %d" % PKG_MAX_TOKENS)

    def log_llm(self, arm: str, record: Mapping[str, Any]) -> None:
        name = arm.split("/")[0] + "_" + (arm.split("/")[1] if "/" in arm else "misc") + ".jsonl"
        with self.log_lock:
            with (self.llm_log_dir / name).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=R._json_default) + "\n")

    def orchestration(self) -> dict[str, Any]:
        return {
            "peak_fast_sessions": self.peak_fast,
            "global_fast_sessions": GLOBAL_FAST_SESSIONS,
            "fit_writers": 1,
            "unit_active_seconds": dict(self.unit_active),
            "active_seconds_rule": "sum of Fast decision wall seconds per arm key; unit cap compares sum/%d against %ds" % (GLOBAL_FAST_SESSIONS, UNIT_MAX_ACTIVE_SECONDS),
            "isolated_verification": (dict(getattr(self, "verifier").stats) if getattr(self, "verifier", None) is not None else None),
            "verification_time_limit_seconds": VERIFY_TIME_LIMIT_SECONDS,
        }


class LedgerBackend:
    """Sending backend wrapper: model identity, token ledger, raw response log, unit caps."""

    def __init__(self, inner: Any, *, state: CourseState, arm: str) -> None:
        self.inner = inner
        self.state = state
        self.arm = str(arm)
        self.calls = 0
        self.returned_models: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def complete(self, request: Any) -> Any:
        parts = self.arm.split("/")
        if len(parts) >= 3 and parts[1] in LLM_ARMS:
            self.state.check_unit(parts[0], parts[1])
        if self.state.tokens.tokens() > PKG_MAX_TOKENS:
            raise R.PackageCeiling("package token ceiling %d" % PKG_MAX_TOKENS)
        started = time.time()
        try:
            response = self.inner.complete(request)
        except Exception as exc:
            self.calls += 1
            self.state.log_llm(self.arm, {
                "ts": utc_now(), "arm": self.arm, "case_id": getattr(request, "case_id", None),
                "stage": getattr(request, "stage", None), "call_index": getattr(request, "call_index", None),
                "transport_ok": False, "error": R._sanitize("%s: %s" % (type(exc).__name__, exc)),
                "seconds": round(time.time() - started, 2),
            })
            raise
        self.calls += 1
        meta = getattr(response, "provider_metadata", None) or {}
        returned = str(meta.get("returned_model") or "") if isinstance(meta, Mapping) else ""
        usage = dict(meta.get("usage") or {}) if isinstance(meta, Mapping) else {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        self.state.tokens.add(self.arm, prompt, completion)
        if returned:
            self.returned_models.add(returned)
        self.state.log_llm(self.arm, {
            "ts": utc_now(), "arm": self.arm, "case_id": getattr(request, "case_id", None),
            "stage": getattr(request, "stage", None), "call_index": getattr(request, "call_index", None),
            "transport_ok": True, "returned_model": returned, "prompt_tokens": prompt,
            "completion_tokens": completion, "parse_status": getattr(response, "parse_status", None),
            "finish_reason": getattr(response, "finish_reason", None),
            "assistant_text": str(getattr(response, "assistant_text", "") or ""),
            "seconds": round(time.time() - started, 2),
        })
        if self.state.live and returned != R.REQUIRED_RETURNED_MODEL:
            raise R.AccountFault("MODEL_IDENTITY_MISMATCH: returned %r, required %r" % (returned, R.REQUIRED_RETURNED_MODEL))
        return response


def make_sender_factory(state: CourseState, raw_factory: Callable[[], Any]) -> Callable[[str], Callable[[], Any]]:
    """``factory_for(arm)()`` returns the sending backend for that arm key."""
    def factory_for(arm: str) -> Callable[[], Any]:
        def make() -> Any:
            return LedgerBackend(raw_factory(), state=state, arm=arm)
        return make
    return factory_for


def bind_metered_factory(factory: Callable[[], Any], budget: R.Budget, arm: str) -> Callable[[], Any]:
    """Meter innermost with the per-decision cap, retry outermost (one charge per HTTP attempt)."""
    def bound() -> Any:
        sender = factory()
        if isinstance(sender, R.BoundedTransportRetry):
            raise RuntimeError("factory returned a retry wrapper; meter must sit inside retry")
        metered = R.MeteredBudgetBackend(sender, budget=budget, arm=arm, per_sequence_cap=PER_DECISION_CAP)
        return R.BoundedTransportRetry(metered, attempts=R.TRANSPORT_ATTEMPTS)
    return bound


_current_extra: ContextVar[dict[str, Any] | None] = ContextVar("dev_train3_extra", default=None)


# ---------------------------------------------------------------------------
# isolated, cached, hard-time-limited candidate verification
# ---------------------------------------------------------------------------
#
# Fast's actionability menu (fast_agent._actionable_operators) and its propose-
# stage verification both call verify_candidate, which really executes the
# operator on the whole observation array (1800 points for the training role).
# impute_ssm fits a statsmodels UnobservedComponents model there; on the J2
# Changping prefix its default seasonal guess is 360 and the Kalman filter
# holds the GIL for hours, starving every other Fast thread (plan §6.8).
# The executor has no time limit and a thread cannot interrupt a C extension,
# so verification is moved into one child process with a hard wall limit.
# Same function, same arguments, same receipt when it completes in time; a
# call that exceeds the limit is killed, recorded, and returned as a rejected
# EXECUTION_FAILED receipt (closed vocabulary), so the operator is simply not
# selectable for that decision -- it is not rewritten to identity or linear.
# Identical (program, array, limits) calls are answered from a process-local
# cache, which removes the repeated menu probes across queries and arms.


def _verify_worker(conn: Any) -> None:
    import numpy as _np
    from SelfEvolvingHarnessTS.contracts.candidate import Candidate as _Candidate
    from SelfEvolvingHarnessTS.contracts.program import Program as _Program
    from SelfEvolvingHarnessTS.runtime.candidate_verification import verify_candidate as _verify
    while True:
        try:
            job = conn.recv()
        except EOFError:
            return
        if job is None:
            return
        try:
            steps = [(str(op), dict(params)) for op, params in job["steps"]]
            program = _Program.from_steps(steps, source=str(job["source"]))
            candidate = _Candidate.program_candidate(str(job["candidate_id"]), program, source=str(job["source"]))
            artifact = _verify(candidate, _np.asarray(job["values"], dtype=_np.float64), **job["kwargs"])
            receipt = artifact.receipt
            conn.send({
                "ok": True,
                "receipt": {
                    "status": receipt.status, "program_sha": receipt.program_sha,
                    "operator_legality_ok": receipt.operator_legality_ok, "compilation_ok": receipt.compilation_ok,
                    "execution_ok": receipt.execution_ok, "shape_preserved": receipt.shape_preserved,
                    "finite_output": receipt.finite_output,
                    "effect_equivalent_to_identity": receipt.effect_equivalent_to_identity,
                    "modified_fraction": float(receipt.modified_fraction),
                    "modified_region_fractions": [list(r) for r in receipt.modified_region_fractions],
                    "outside_inspected_region_modified": receipt.outside_inspected_region_modified,
                    "warnings": list(receipt.warnings), "rejection_code": receipt.rejection_code,
                },
                "prepared_values": None if artifact.prepared_values is None else _np.asarray(artifact.prepared_values).tolist(),
                "execution_trace": [dict(row) for row in artifact.execution_trace],
                "modified_indices": [int(i) for i in artifact.modified_indices],
            })
        except Exception as exc:  # noqa: BLE001
            conn.send({"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:300])})


class IsolatedVerifier:
    """One child process, hard wall limit per call, exact-argument cache."""

    def __init__(self, ledger_path: Path, *, limit: float = VERIFY_TIME_LIMIT_SECONDS) -> None:
        self.limit = float(limit)
        self.ledger_path = Path(ledger_path)
        self.lock = threading.Lock()
        self.cache: dict[Any, dict[str, Any]] = {}
        self.stats = {"calls": 0, "cache_hits": 0, "worker_calls": 0, "timeouts": 0, "worker_restarts": 0, "worker_seconds": 0.0}
        self._ctx = _mp.get_context("spawn")
        self._proc: Any = None
        self._conn: Any = None

    def _start(self) -> None:
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(target=_verify_worker, args=(child,), daemon=True)
        proc.start()
        child.close()
        self._proc, self._conn = proc, parent

    def _stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.join(10)
            except Exception:  # noqa: BLE001
                pass
        self._proc, self._conn = None, None

    def close(self) -> None:
        with self.lock:
            self._stop()

    @staticmethod
    def _key(steps: Any, values: np.ndarray, kwargs: Mapping[str, Any]) -> Any:
        return (json.dumps(steps, sort_keys=True, default=str), values.tobytes(), values.shape,
                json.dumps({k: (list(map(list, v)) if k == "inspected_regions" else v) for k, v in kwargs.items()}, sort_keys=True, default=str))

    def verify(self, candidate: Any, raw_values: Any, **kwargs: Any) -> Any:
        from SelfEvolvingHarnessTS.contracts.candidate import CandidateKind
        from SelfEvolvingHarnessTS.runtime import candidate_verification as cv

        values = np.asarray(raw_values, dtype=np.float64).ravel()
        if candidate.kind is CandidateKind.IDENTITY or candidate.program is None:
            return cv.verify_candidate(candidate, values, **kwargs)
        steps = [[str(op), dict(params)] for op, params in candidate.program.execution_steps()]
        plain_kwargs = {
            "allowed_operators": tuple(str(o) for o in kwargs.get("allowed_operators", ())),
            "inspected_regions": tuple((int(a), int(b)) for a, b in (kwargs.get("inspected_regions") or ())),
            "maximum_modified_fraction": float(kwargs.get("maximum_modified_fraction", 1.0)),
            "preserve_outside_inspected_region": bool(kwargs.get("preserve_outside_inspected_region", False)),
            "require_finite_output": bool(kwargs.get("require_finite_output", True)),
        }
        key = self._key(steps, values, plain_kwargs)
        with self.lock:
            self.stats["calls"] += 1
            hit = self.cache.get(key)
            if hit is None:
                self.stats["worker_calls"] += 1
                hit = self._run(candidate, steps, values, plain_kwargs)
                self.cache[key] = hit
            else:
                self.stats["cache_hits"] += 1
        return self._artifact(candidate, hit)

    def _run(self, candidate: Any, steps: Any, values: np.ndarray, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        if self._proc is None or not self._proc.is_alive():
            self._start()
        started = time.time()
        try:
            self._conn.send({"steps": steps, "values": values.tolist(), "kwargs": dict(kwargs),
                             "candidate_id": str(candidate.candidate_id), "source": str(candidate.source)})
            if self._conn.poll(self.limit):
                reply = self._conn.recv()
                self.stats["worker_seconds"] += time.time() - started
                if reply.get("ok"):
                    return reply
                return {"ok": False, "error": reply.get("error"), "timed_out": False}
        except Exception as exc:  # noqa: BLE001
            self._stop()
            self.stats["worker_restarts"] += 1
            return {"ok": False, "error": "worker fault: %s" % type(exc).__name__, "timed_out": False}
        # hard limit exceeded: kill the child, restart lazily, record
        self.stats["timeouts"] += 1
        self.stats["worker_restarts"] += 1
        self.stats["worker_seconds"] += time.time() - started
        self._stop()
        row = {"ts": utc_now(), "steps": steps, "n_points": int(values.size), "missing": int(np.isnan(values).sum()),
               "limit_seconds": self.limit, "outcome": "verification_time_limit_exceeded"}
        try:
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "error": "verification_time_limit_exceeded:%ds" % int(self.limit), "timed_out": True}

    @staticmethod
    def _artifact(candidate: Any, reply: Mapping[str, Any]) -> Any:
        from SelfEvolvingHarnessTS.runtime import candidate_verification as cv
        if reply.get("ok"):
            r = reply["receipt"]
            receipt = cv.CandidateVerificationReceipt(
                candidate_id=candidate.candidate_id, candidate_kind=candidate.kind.value, status=r["status"],
                program_sha=r["program_sha"], operator_legality_ok=r["operator_legality_ok"],
                compilation_ok=r["compilation_ok"], execution_ok=r["execution_ok"], shape_preserved=r["shape_preserved"],
                finite_output=r["finite_output"], effect_equivalent_to_identity=r["effect_equivalent_to_identity"],
                modified_fraction=float(r["modified_fraction"]),
                modified_region_fractions=tuple((float(a), float(b)) for a, b in r["modified_region_fractions"]),
                outside_inspected_region_modified=r["outside_inspected_region_modified"],
                warnings=tuple(r["warnings"]), rejection_code=r["rejection_code"],
            )
            prepared = None if reply.get("prepared_values") is None else np.asarray(reply["prepared_values"], dtype=np.float64)
            return cv.CandidateExecutionArtifact(candidate, receipt, prepared, tuple(reply.get("execution_trace") or ()),
                                                 tuple(int(i) for i in reply.get("modified_indices") or ()))
        receipt = cv.CandidateVerificationReceipt(
            candidate_id=candidate.candidate_id, candidate_kind=candidate.kind.value, status="rejected",
            program_sha=candidate.program.sha(), operator_legality_ok=True, compilation_ok=True, execution_ok=False,
            shape_preserved=False, finite_output=False, effect_equivalent_to_identity=False, modified_fraction=0.0,
            modified_region_fractions=(), outside_inspected_region_modified=False, rejection_code="EXECUTION_FAILED",
        )
        trace = ({"op": steps_label([(op, params) for op, params in candidate.program.execution_steps()]),
                  "ok": False, "error": str(reply.get("error") or "verification failed in isolated worker"),
                  "isolated_verification": True, "timed_out": bool(reply.get("timed_out"))},)
        return cv.CandidateExecutionArtifact(candidate, receipt, None, trace)


@contextmanager
def install_course(state: CourseState):
    """Global Fast gate, single fit writer, Fast public_input extras, per-decision cap."""
    original_fast = R.run_fast_decision
    original_fit = R.fit_arm
    original_freeze = R.freeze_model
    original_extra = R.public_extra
    original_bind = R.bind_backend_factory
    original_classify = R.classify_fault
    menu = known_candidate_menu()
    from SelfEvolvingHarnessTS.methods.ttha import fast_agent as _fast_agent
    original_fast_verify = _fast_agent.verify_candidate
    verifier = IsolatedVerifier(state.course_dir / "verification_timeouts.jsonl")
    state.verifier = verifier

    def classify_fault(detail: str) -> str:
        # The relay's own transient class (AgentTransportError wraps 5xx /
        # InternalServerError / connection faults) was not in the marker list,
        # so a single relay 500 ended a decision without the bounded retry.
        kind = original_classify(detail)
        lowered = str(detail).lower()
        if kind == "ACCOUNT_OR_PERMISSION_FAULT":
            return kind
        if "agenttransporterror" in lowered or "relay transport failed" in lowered or "internalservererror" in lowered:
            return "TRANSPORT_TRANSIENT_FAULT"
        return kind

    def gated_fast(**kwargs: Any) -> Any:
        if state.account_fault is not None:
            raise state.account_fault
        arm = str(kwargs.get("arm"))
        context = dict(state.batch_context.get(arm) or {})
        spec = kwargs["spec"]
        bundle = kwargs["bundle"]
        uid = str(kwargs["uid"])
        role = str(kwargs["role"])
        raw = np.asarray(bundle.values[uid], dtype=np.float64)
        if role == "train_workflow":
            observed = raw[:int(spec.training_prefix_end)]
        else:
            origin_i = int(kwargs["origin"])
            observed = raw[origin_i - CONTEXT_LENGTH:origin_i]
        context["execution_observations"] = execution_observations(observed, role, spec)
        state.fast_gate.acquire()
        with state.state_lock:
            state.live_fast += 1
            state.peak_fast = max(state.peak_fast, state.live_fast)
        token = _current_extra.set(context)
        started = time.time()
        try:
            record = original_fast(**kwargs)
            record["execution_observations"] = context["execution_observations"]
            record["task_research_summary_delivered"] = context.get("task_research_summary")
            record["job_phase"] = context.get("job_phase")
            return record
        except R.AccountFault as exc:
            state.account_fault = exc
            raise
        finally:
            _current_extra.reset(token)
            with state.state_lock:
                state.live_fast -= 1
                state.unit_active[arm] = state.unit_active.get(arm, 0.0) + (time.time() - started)
            state.fast_gate.release()

    def gated_fit(**kwargs: Any) -> Any:
        with state.fit_lock:
            return original_fit(**kwargs)

    def gated_freeze(**kwargs: Any) -> Any:
        with state.fit_lock:
            return original_freeze(**kwargs)

    def extra(role: str, *, observed_length: int, spec: R.RunSpec | None = None) -> dict[str, Any]:
        payload = dict(original_extra(role, observed_length=observed_length, spec=spec))
        payload["cap_facts"] = W.public_cap_facts(spec) if spec is not None else {"maximum_modified_fraction": 1.0}
        payload["runtime_capabilities"] = RUNTIME_CAPABILITIES
        payload["known_candidate_menu"] = {
            "candidates": menu,
            "meaning": ("known legal candidates listed neutrally for reference; they carry no "
                        "effect labels, no ranking and no recommendation; any legal 1-4 step "
                        "workflow or identity remains allowed"),
        }
        frame = payload.get("observation_frame")
        if isinstance(frame, dict) and role == "predict_input":
            frame = dict(frame)
            frame.pop("no_current_downstream_feedback", None)
            payload["observation_frame"] = frame
        exec_frame = payload.get("execution_frame")
        if isinstance(exec_frame, dict) and role == "predict_input":
            exec_frame = dict(exec_frame)
            exec_frame.pop("no_current_downstream_feedback", None)
            exec_frame["feedback"] = (
                "the effect of this prediction input is not visible to this role; during "
                "adaptation the Runtime scores whole-job outputs afterwards, during "
                "deployment nothing is scored back"
            )
            payload["execution_frame"] = exec_frame
        current = _current_extra.get(None)
        if current is not None and role in ("train_workflow", "predict_input"):
            for key in ("job_phase", "query", "task_research_summary", "execution_observations"):
                if key in current:
                    payload[key] = current[key]
        return payload

    R.run_fast_decision = gated_fast
    R.fit_arm = gated_fit
    R.freeze_model = gated_freeze
    R.public_extra = extra
    R.bind_backend_factory = bind_metered_factory
    R.classify_fault = classify_fault
    _fast_agent.verify_candidate = verifier.verify
    try:
        yield state
    finally:
        _fast_agent.verify_candidate = original_fast_verify
        verifier.close()
        R.run_fast_decision = original_fast
        R.fit_arm = original_fit
        R.freeze_model = original_freeze
        R.public_extra = original_extra
        R.bind_backend_factory = original_bind
        R.classify_fault = original_classify
        R.atomic_write_json(state.course_dir / "orchestration.json", state.orchestration())


# ---------------------------------------------------------------------------
# job context and numeric layer (one shared model per assignment, cached per job)
# ---------------------------------------------------------------------------

class JobContext:
    def __init__(self, *, job: JobSpec, values: Mapping[str, np.ndarray], state: CourseState,
                 machinery: Mapping[str, Any], live: bool) -> None:
        self.job = job
        self.state = state
        self.machinery = machinery
        self.spec = run_spec_for(job, live=live)
        self.bundle = R.DataBundle(values={uid: np.asarray(values[uid], dtype=np.float64) for uid in UIDS},
                                   train_uids=list(UIDS), eval_uids=list(UIDS), period=PERIOD)
        self.task_ctx = R.task_context_for(self.spec)
        self.dir = state.course_dir / job.job_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_lock = threading.RLock()
        self._models: dict[str, tuple[numeric.FrozenRidge, str]] = {}
        self.static_ref: dict[str, dict[int, dict[str, Any]]] = {}
        index_path = self.models_dir / "index.json"
        if index_path.is_file():
            for fingerprint, name in R.read_json(index_path).items():
                path = self.models_dir / name
                if path.is_file():
                    self._models[fingerprint] = (numeric.FrozenRidge.from_dict(R.read_json(path)), name)

    def fit(self, assignment: Mapping[str, Any], *, arm: str) -> tuple[numeric.FrozenRidge, dict[str, Any]]:
        """One pooled Ridge for this exact train assignment; exact reuse is a cache hit."""
        fingerprint = R._assignment_fingerprint(assignment)
        with self.model_lock:
            hit = self._models.get(fingerprint)
            if hit is not None:
                return hit[0], {"physical_fit": 0, "cache_hit": True, "model_file": hit[1]}
        with self.state.fit_lock:
            with self.model_lock:
                hit = self._models.get(fingerprint)
                if hit is not None:
                    return hit[0], {"physical_fit": 0, "cache_hit": True, "model_file": hit[1]}
            model = R.fit_arm(spec=self.spec, bundle=self.bundle, train_assignment=assignment,
                              budget=self.state.budget, arm=arm, origin=TRAIN_PREFIX_END)
            with self.model_lock:
                name = "model_%03d.json" % len(self._models)
                payload = model.to_dict()
                payload["dev_train3_context"] = {
                    "job": self.job.job_id, "fitted_by_arm": arm, "assignment_fingerprint": fingerprint,
                    "training_prefix_end": TRAIN_PREFIX_END, "anchors": list(ANCHORS),
                }
                R.atomic_write_json(self.models_dir / name, payload)
                self._models[fingerprint] = (model, name)
                R.atomic_write_json(self.models_dir / "index.json",
                                    {fp: nm for fp, (_m, nm) in self._models.items()})
            return model, {"physical_fit": 1, "cache_hit": False, "model_file": name}

    def model_by_file(self, name: str) -> numeric.FrozenRidge:
        return numeric.FrozenRidge.from_dict(R.read_json(self.models_dir / name))

    def predict(self, model: numeric.FrozenRidge, v_assignment: Mapping[str, Any], origin: int) -> dict[str, Any]:
        return R.predict_arm(spec=self.spec, bundle=self.bundle, model=model,
                             eval_assignment=v_assignment, origin=int(origin))

    def score(self, pred: Mapping[str, Any], origin: int, block: str) -> dict[str, Any]:
        scored = R.score_arm(spec=self.spec, bundle=self.bundle, predictions=pred, origin=int(origin))
        pop = R.population_from_scores(list(UIDS), scored)
        pop["metric_scales"] = scored.get("metric_scales")
        raw_mae = []
        truth_rows = []
        for index, uid in enumerate(UIDS):
            raw = self.bundle.values[uid]
            truth = raw[int(origin):int(origin) + HORIZON]
            mask = np.isfinite(truth)
            if mask.any():
                raw_mae.append(float(np.mean(np.abs(truth[mask] - np.asarray(pred["predictions"])[index][mask]))))
            else:
                raw_mae.append(None)
            truth_rows.append(int(mask.sum()))
        pop["raw_mae_per_uid"] = dict(zip(UIDS, raw_mae))
        pop["truth_finite_points_per_uid"] = dict(zip(UIDS, truth_rows))
        static = (self.static_ref.get(block) or {}).get(int(origin))
        if static is not None:
            utilities: dict[str, Any] = {}
            for uid in UIDS:
                s_v = static["per_uid_smase"].get(uid)
                a_v = pop["per_uid_smase"].get(uid)
                utilities[uid] = (float(s_v) - float(a_v)) if isinstance(s_v, (int, float)) and isinstance(a_v, (int, float)) else "UNKNOWN"
            pop["per_uid_utility_vs_static"] = utilities
            pop["risk"] = R.harmed_stats([v for v in utilities.values() if isinstance(v, (int, float))])
        return pop

    def predict_and_score_block(self, model: numeric.FrozenRidge, v_by_origin: Mapping[int, Mapping[str, Any] | None],
                                block: str, *, save_dir: Path, tag: str) -> dict[str, Any]:
        out: dict[str, Any] = {"block": block, "origins": {}, "mean_smase": "UNKNOWN"}
        means: list[Any] = []
        for origin in self.job.origins(block):
            path = Path(save_dir) / ("%s_scores_%s_%d.json" % (tag, block, origin))
            cached = R.read_json(path) if path.is_file() else None
            if cached is not None and isinstance(cached.get("mean_smase"), (int, float)):
                pop = cached
            else:
                assignment = v_by_origin.get(int(origin))
                if assignment is None:
                    pop = R.unknown_population(list(UIDS), "incomplete prediction assignment")
                else:
                    try:
                        pred = self.predict(model, assignment, origin)
                        pop = self.score(pred, origin, block)
                        pop["v_assignment"] = {uid: R.steps_to_json(assignment[uid]) for uid in UIDS}
                    except (R.PackageCeiling, R.AccountFault):
                        raise
                    except Exception as exc:  # noqa: BLE001
                        pop = R.unknown_population(list(UIDS), "%s: %s" % (type(exc).__name__, R._sanitize(exc)))
                if isinstance(pop.get("mean_smase"), (int, float)):
                    R.atomic_write_json(path, pop)
            out["origins"][str(origin)] = pop
            means.append(pop.get("mean_smase"))
        out["mean_smase"] = fmean_or_unknown(means)
        utilities = [v for pop in out["origins"].values()
                     for v in (pop.get("per_uid_utility_vs_static") or {}).values() if isinstance(v, (int, float))]
        out["risk"] = R.harmed_stats(utilities)
        out["mean_utility_vs_static"] = fmean_or_unknown(utilities) if len(utilities) == len(UIDS) * len(self.job.origins(block)) else "UNKNOWN"
        return out

    def freeze_predictions(self, model: numeric.FrozenRidge, v_by_origin: Mapping[int, Mapping[str, Any] | None],
                           *, save_dir: Path, tag: str) -> dict[str, Any]:
        """E outputs: predictions only.  Truth is not read here."""
        out: dict[str, Any] = {}
        for origin in self.job.origins("E"):
            path = Path(save_dir) / ("%s_predictions_E_%d.json" % (tag, origin))
            if path.is_file() and R.read_json(path).get("predictions_frozen"):
                out[str(origin)] = "FROZEN"
                continue
            assignment = v_by_origin.get(int(origin))
            if assignment is None:
                R.atomic_write_json(path, {"origin": int(origin), "predictions_frozen": False,
                                           "why": "incomplete prediction assignment"})
                out[str(origin)] = "UNKNOWN"
                continue
            try:
                pred = self.predict(model, assignment, origin)
            except (R.PackageCeiling, R.AccountFault):
                raise
            except Exception as exc:  # noqa: BLE001
                R.atomic_write_json(path, {"origin": int(origin), "predictions_frozen": False,
                                           "why": "%s: %s" % (type(exc).__name__, R._sanitize(exc))})
                out[str(origin)] = "UNKNOWN"
                continue
            R.atomic_write_json(path, {
                "origin": int(origin), "eval_uids": list(pred["eval_uids"]),
                "predictions": np.asarray(pred["predictions"]).tolist(),
                "behavior_point_count": pred.get("behavior_point_count"),
                "v_assignment": {uid: R.steps_to_json(assignment[uid]) for uid in UIDS},
                "truth_not_included": True, "predictions_frozen": True,
            })
            out[str(origin)] = "FROZEN"
        return out

    def score_frozen(self, path: Path) -> dict[str, Any]:
        """External E scoring of one frozen prediction file."""
        data = R.read_json(path)
        if not data.get("predictions_frozen"):
            return R.unknown_population(list(UIDS), str(data.get("why") or "not frozen"))
        pred = {"eval_uids": data["eval_uids"], "predictions": np.asarray(data["predictions"], dtype=np.float64), "consumer_fits": 0}
        return self.score(pred, int(data["origin"]), "E")


def uniform(steps: Any) -> dict[str, Any]:
    return {uid: steps for uid in UIDS}


def by_origin(job: JobSpec, block: str, assignment: Mapping[str, Any] | None) -> dict[int, Any]:
    return {int(origin): assignment for origin in job.origins(block)}


def assignment_json(assignment: Mapping[str, Any] | None) -> Any:
    if assignment is None:
        return None
    return {uid: R.steps_to_json(assignment[uid]) for uid in assignment}


def assignment_from_json(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {str(uid): R.steps_from_json(steps) for uid, steps in value.items()}


def legal_steps(steps: Any) -> tuple[bool, str]:
    if steps is None:
        return True, "identity"
    try:
        tup = tuple((str(op), dict(params or {})) for op, params in steps)
    except Exception as exc:  # noqa: BLE001
        return False, "malformed steps: %s" % type(exc).__name__
    if not 1 <= len(tup) <= 4:
        return False, "workflow must have 1-4 steps"
    allowed = set(R.legal_forecast_ops())
    for op, _params in tup:
        if op not in allowed:
            return False, "operator %s is not a legal forecast operator" % op
    try:
        numeric.compile_steps(tup)
    except Exception as exc:  # noqa: BLE001
        return False, "compile failed: %s" % type(exc).__name__
    return True, steps_label(tup)


# ---------------------------------------------------------------------------
# Fast batches
# ---------------------------------------------------------------------------

class FastRunner:
    def __init__(self, ctx: JobContext, factory_for: Callable[[str], Callable[[], Any]]) -> None:
        self.ctx = ctx
        self.factory_for = factory_for

    def _batch(self, *, arm_key: str, snapshot: Any, role: str, origin: int, position: int,
               checkpoint: Path, context: Mapping[str, Any]) -> list[dict[str, Any]]:
        state = self.ctx.state
        state.batch_context[arm_key] = dict(context)
        try:
            rows = None
            for repass in range(2):
                # run_batch_fast re-asks only rows whose checkpoint status is a
                # fault (the runner's own resume rule); valid or completed-invalid
                # decisions are never re-sampled.  One bounded re-pass per batch.
                rows = R.run_batch_fast(
                    spec=self.ctx.spec, bundle=self.ctx.bundle, snapshot=snapshot, machinery=self.ctx.machinery,
                    backend_factory=self.factory_for(arm_key), uids=list(UIDS), role=role, arm=arm_key,
                    position=int(position), origin=int(origin), task_ctx=self.ctx.task_ctx,
                    budget=state.budget, checkpoint_path=checkpoint,
                )
                if not any(not R.checkpoint_is_final(row) for row in rows):
                    break
                if repass == 0:
                    for row in rows:
                        if not R.checkpoint_is_final(row):
                            row["fault_repass_attempted"] = True
            return rows
        finally:
            state.batch_context.pop(arm_key, None)

    def train_w(self, *, arm_key: str, snapshot: Any, save_dir: Path, context: Mapping[str, Any]):
        rows = self._batch(arm_key=arm_key, snapshot=snapshot, role="train_workflow", origin=TRAIN_PREFIX_END,
                           position=0, checkpoint=Path(save_dir) / "train_w.json", context=context)
        assignment, missing = R.records_to_assignment(list(UIDS), rows)
        return rows, assignment, missing

    def predict_v(self, *, arm_key: str, snapshot: Any, origin: int, save_dir: Path, context: Mapping[str, Any]):
        rows = self._batch(arm_key=arm_key, snapshot=snapshot, role="predict_input", origin=int(origin),
                           position=int(origin), checkpoint=Path(save_dir) / ("predict_v_%d.json" % int(origin)),
                           context=context)
        assignment, missing = R.records_to_assignment(list(UIDS), rows)
        return rows, assignment, missing


def decision_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append({
            "series_uid": row.get("series_uid"), "deploy_status": row.get("deploy_status"),
            "deployed_program": row.get("deployed_program"), "typed_steps": row.get("typed_steps"),
            "valid": bool(row.get("valid_mechanism_decision")), "llm_requests_sent": row.get("llm_requests_sent"),
            "wall_seconds": row.get("wall_seconds"), "why_unknown": row.get("why_unknown"),
            "returned_models": row.get("returned_models"),
        })
    return out


# ---------------------------------------------------------------------------
# LLM agent cores for research / Slow calls
# ---------------------------------------------------------------------------

def make_core(ctx: JobContext, factory_for: Callable[[str], Callable[[], Any]], arm_key: str,
              extra: Mapping[str, Any]) -> Any:
    metered = R.MeteredBudgetBackend(factory_for(arm_key)(), budget=ctx.state.budget, arm=arm_key,
                                     per_sequence_cap=PER_DECISION_CAP)
    inner = R.BoundedTransportRetry(metered, attempts=R.TRANSPORT_ATTEMPTS)
    core = R.RoleInjectingCore(
        ctx.machinery["TTHAAgentCore"](inner, R.UnavailableToolGateway(), model=ctx.spec.requested_model,
                                       base_url=R._core_base_url(ctx.spec)),
        extra,
    )
    core.metered = metered  # type: ignore[attr-defined]
    return core


RESEARCH_PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["decision", "changed_factor", "execution", "uncertain_decision", "observations_vs_interpretations",
                 "result_that_changes_next_step", "investigation_purpose_for_fast", "expected_fits"],
    "properties": {
        "decision": {"enum": ["run_contrast", "stop_and_reuse"]},
        "changed_factor": {"enum": ["train_side", "predict_side", "both", "none"]},
        "execution": {
            "type": "object", "additionalProperties": False, "required": ["mode"],
            "properties": {
                "mode": {"enum": ["pinned_program", "fast_regenerate", "none"]},
                "pinned_train_program": {"type": ["array", "null"], "items": {
                    "type": "object", "additionalProperties": False, "required": ["op"],
                    "properties": {"op": {"type": "string"}, "params": {"type": "object"}}}},
                "pinned_predict_program": {"type": ["array", "null"], "items": {
                    "type": "object", "additionalProperties": False, "required": ["op"],
                    "properties": {"op": {"type": "string"}, "params": {"type": "object"}}}},
            },
        },
        "uncertain_decision": {"type": "string", "maxLength": 600},
        "observations_vs_interpretations": {"type": "string", "maxLength": 900},
        "result_that_changes_next_step": {"type": "string", "maxLength": 600},
        "investigation_purpose_for_fast": {"type": "string", "maxLength": 300},
        "expected_fits": {"type": "integer"},
    },
}

REFLECTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["updated_investigation_purpose_for_fast", "temporary_state_note"],
    "properties": {
        "updated_investigation_purpose_for_fast": {"type": "string", "maxLength": 300},
        "temporary_state_note": {"type": "string", "maxLength": 900},
    },
}


def run_llm_stage(ctx: JobContext, factory_for: Callable[[str], Callable[[], Any]], *, arm_key: str,
                  snapshot: Any, stage: str, case_id: str, public_input: Mapping[str, Any],
                  schema_name: str, schema: Mapping[str, Any], checkpoint: Path,
                  extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One Slow-role stage with a local schema; checkpointed; raw text saved."""
    if checkpoint.is_file():
        saved = R.read_json(checkpoint)
        saved["resumed_from_checkpoint"] = True
        return saved
    core = make_core(ctx, factory_for, arm_key, extra or {})
    view = resolve_harness_view(snapshot, {}, role="slow")
    out: dict[str, Any] = {"stage": stage, "arm_key": arm_key, "case_id": case_id, "started_utc": utc_now()}
    try:
        result = core.run_stage(
            role=AgentRole.SLOW, stage=stage, case_id=case_id, public_input=public_input, harness_view=view,
            output_schema_name=schema_name, output_schema=schema, source_snapshot_sha=snapshot.runtime_bundle_sha,
            task_context_sha=ctx.task_ctx.sha(), validation_retries=1,
        )
        out["status"] = "OK"
        out["payload"] = json.loads(json.dumps(result.payload, default=R._json_default))
        out["assistant_text"] = str(getattr(result.response, "assistant_text", "") or "")
        out["validation_retry_count"] = int(result.validation_retry_count)
    except R.AccountFault:
        raise
    except Exception as exc:  # noqa: BLE001
        detail = "%s: %s" % (type(exc).__name__, exc)
        R.raise_if_account(detail, cause=exc)
        out["status"] = "FAULT"
        out["fault_kind"] = R.classify_fault(detail)
        out["detail"] = R._sanitize(detail, 400)
        out["last_assistant_text"] = str(getattr(exc, "last_assistant_text", "") or "")[:500]
        out["payload"] = None
    out["llm_requests_sent"] = int(core.metered.calls)
    out["returned_models"] = sorted(core.metered.returned_models)
    out["ended_utc"] = utc_now()
    R.atomic_write_json(checkpoint, out)
    return out


# ---------------------------------------------------------------------------
# cards
# ---------------------------------------------------------------------------

def job_frame(ctx: JobContext) -> dict[str, Any]:
    return {
        "series": list(UIDS),
        "training_prefix_points": TRAIN_PREFIX_END,
        "training_windows": "10 anchors %s; each window [anchor-192, anchor+48)" % list(ANCHORS),
        "feedback_blocks": {"C_A": ctx.job.origins("C_A"), "C_B": ctx.job.origins("C_B")},
        "later_deployment_block_origins_not_scored_during_adaptation": len(ctx.job.origins("E")),
        "period": PERIOD, "context_length": CONTEXT_LENGTH, "horizon": HORIZON,
        "indices_are_local_to_this_job": True,
        "series_identity_is_not_an_applicability_reason": True,
    }


def query_public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """The Slow-visible view of one query: assignments, scores, static reference, utilities."""
    return {
        "query": record.get("query"), "block": record.get("block"), "status": record.get("status"),
        "w_source": record.get("w_source"), "v_source": record.get("v_source"),
        "w_by_uid": {uid: steps_label(R.steps_from_json(v)) for uid, v in (record.get("w_assignment") or {}).items()} if record.get("w_assignment") else None,
        "w_typed_steps": record.get("w_assignment"),
        "v_by_origin": {o: {uid: steps_label(R.steps_from_json(v)) for uid, v in a.items()} for o, a in (record.get("v_assignment_by_origin") or {}).items() if a} if record.get("v_assignment_by_origin") else None,
        "scores": {
            o: {"mean_smase": pop.get("mean_smase"), "per_uid_smase": pop.get("per_uid_smase"),
                "per_uid_utility_vs_static": pop.get("per_uid_utility_vs_static"), "risk": pop.get("risk")}
            for o, pop in ((record.get("scores") or {}).get("origins") or {}).items()
        },
        "mean_smase": (record.get("scores") or {}).get("mean_smase"),
        "mean_utility_vs_static": (record.get("scores") or {}).get("mean_utility_vs_static"),
        "physical_fit": record.get("physical_fit"), "cache_hit": record.get("cache_hit"),
        "decision_validity": record.get("decision_validity"),
    }


def static_public(ctx: JobContext, block: str) -> dict[str, Any]:
    return {str(o): {"mean_smase": pop.get("mean_smase"), "per_uid_smase": pop.get("per_uid_smase")}
            for o, pop in (ctx.static_ref.get(block) or {}).items()}


def features_by_uid(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    for row in rows:
        feats = dict(row.get("public_features") or {})
        keep = {}
        for key, value in feats.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)):
                keep[str(key)] = float(value)
        out[str(row.get("series_uid"))] = keep
    return out


def research_card(ctx: JobContext, unit: "UnitState", *, purpose: str) -> dict[str, Any]:
    card: dict[str, Any] = {
        "pattern_id": "devtrain3-%s-%s-%s" % (ctx.job.job_id, unit.arm, purpose),
        "observable_signature": {"task_kind": "forecast", "decision_role": "slow",
                                 "scope": "job-level research planning at a batch boundary"},
        "role_of_this_call": purpose,
        "job_frame": job_frame(ctx),
        "consumer_structure": dict(R.CONSUMER_STRUCTURE),
        "runtime_capabilities": RUNTIME_CAPABILITIES,
        "known_candidate_menu": known_candidate_menu(),
        "legal_operators": R.legal_forecast_ops(),
        "feedback_semantics": {
            "metric": "sMASE on raw truth, lower is better; the seasonal scale is fixed per origin by the raw history",
            "utility_vs_static": "static sMASE minus this scheme's sMASE for the same series and origin; positive is better",
            "shared_model_caveat": ("one shared model is fitted from all six series' prepared windows, so a per-series "
                                    "loss is not the independent causal effect of that series' own W"),
            "material_difference": MATERIAL,
        },
        "static_reference": {block: static_public(ctx, block) for block in ("C_A",)},
        "queries": [query_public_record(q) for q in unit.query_records()],
        "train_prefix_public_features_by_uid": unit.train_features,
        "observations_vs_hypotheses": "numbers here are observations; mechanism talk is hypothesis and may be wrong",
    }
    if purpose == "q2_plan":
        card["budget"] = {
            "planned_feedback_queries_total": 4, "used": 1,
            "this_call_decides": "q2: one more whole-job scheme scored on block C_A",
            "after_q2": ("q3/q4 re-solve block C_B under old and new knowledge; that is not planned here.  "
                         "Only what you declare below is executed."),
            "one_query_costs": "at most one new shared-model fit plus scoring on all eligible C_A origins",
        }
        card["what_you_decide"] = {
            "decision": "run_contrast or stop_and_reuse (no distinguishing hypothesis is a legal answer)",
            "changed_factor": ("train_side: only the training W changes, q1's exact V is reused; "
                               "predict_side: q1's exact W and model are reused, only V changes; "
                               "both: candidate search, no single-factor attribution will be claimed"),
            "execution.mode": ("pinned_program: the Runtime applies the typed program you give uniformly on the "
                               "changed side(s) ([] means identity); fast_regenerate: the Fast role re-solves the "
                               "changed side(s) sequence by sequence under the same knowledge"),
            "investigation_purpose_for_fast": ("<=300 chars; the only text of yours that reaches the Fast role; "
                                               "numbers and effect words are withheld by the runtime"),
        }
    return card


def boundary_card(ctx: JobContext, unit: "UnitState") -> dict[str, Any]:
    card = research_card(ctx, unit, purpose="skill_boundary")
    card["static_reference"] = {block: static_public(ctx, block) for block in ("C_A",)}
    card["research_plan_q2"] = unit.state.get("q2_plan")
    card["frozen_task_summary_given_to_fast"] = unit.state.get("summary")
    decisions: dict[str, Any] = {}
    for q in ("q1", "q2"):
        decisions[q] = {
            "train_w": decision_summary(unit.rows.get((q, "W")) or []),
            "predict_v": {str(o): decision_summary(rows) for (qq, role, o), rows in unit.rows_by_origin.items()
                          if qq == q and role == "V"},
        }
    card["decisions"] = decisions
    card["what_you_may_write"] = (
        "Exactly one edit on one surface from writable_surface_catalog, or no_proposal.  Guidance prose only: "
        "applicability conditions must be observable features, never a series id or a dataset name; "
        "no frozen program card; keeping the current knowledge is legal."
    )
    card["what_happens_next"] = (
        "block C_B is re-solved twice: once under the current knowledge, once under your candidate; "
        "the candidate is adopted only if its C_B mean sMASE improves by at least %s; after that, "
        "prediction inputs on a later block are prepared with no feedback and no refit." % MATERIAL
    )
    card["select_cannot_veto_deployment"] = "Fast's chosen program is delivered after legality; no Support admission."
    return card


# ---------------------------------------------------------------------------
# unit state (one LLM arm x one job)
# ---------------------------------------------------------------------------

def load_saved_rows(unit: "UnitState", queries: Sequence[str]) -> None:
    for q in queries:
        qdir = unit.dir / q
        path = qdir / "train_w.json"
        if path.is_file() and (q, "W") not in unit.rows:
            unit.rows[(q, "W")] = list(R.read_json(path).get("sequences") or [])
        for block in ("C_A", "C_B"):
            for origin in unit.ctx.job.origins(block):
                path = qdir / ("predict_v_%d.json" % origin)
                if path.is_file() and (q, "V", origin) not in unit.rows_by_origin:
                    unit.rows_by_origin[(q, "V", origin)] = list(R.read_json(path).get("sequences") or [])


class UnitState:
    def __init__(self, ctx: JobContext, arm: str) -> None:
        self.ctx = ctx
        self.arm = str(arm)
        self.dir = ctx.dir / arm
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "unit_state.json"
        self.state: dict[str, Any] = R.read_json(self.path) if self.path.is_file() else {}
        self.rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.rows_by_origin: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        self.train_features: dict[str, Any] = {}

    def save(self) -> None:
        R.atomic_write_json(self.path, self.state)

    def key(self, tag: str) -> str:
        return "%s/%s/%s" % (self.ctx.job.job_id, self.arm, tag)

    def query_records(self) -> list[dict[str, Any]]:
        return [self.state[q] for q in ("q1", "q2", "q3", "q4") if isinstance(self.state.get(q), dict)]


# ---------------------------------------------------------------------------
# one query = one whole-job scheme scored on one block
# ---------------------------------------------------------------------------

def run_scheme(ctx: JobContext, *, tag: str, arm_key: str, w_assignment: Mapping[str, Any] | None,
               v_by_origin: Mapping[int, Mapping[str, Any] | None], block: str, save_dir: Path,
               w_source: str, v_source: str, missing_w: Sequence[str] = (), missing_v: Mapping[int, Sequence[str]] | None = None,
               logical_query: bool = True) -> dict[str, Any]:
    """Fit (cached) + score one block.  Physical vs logical accounting is explicit."""
    record: dict[str, Any] = {
        "query": tag, "block": block, "arm_key": arm_key, "w_source": w_source, "v_source": v_source,
        "w_assignment": assignment_json(w_assignment), "logical_query": bool(logical_query),
        "v_assignment_by_origin": {str(o): assignment_json(a) for o, a in v_by_origin.items()},
        "missing_w": list(missing_w), "missing_v": {str(o): list(m) for o, m in (missing_v or {}).items()},
    }
    if w_assignment is None:
        record["status"] = "W_INCOMPLETE"
        record["scores"] = {"block": block, "mean_smase": "UNKNOWN", "origins": {}}
        record["physical_fit"] = 0
        record["cache_hit"] = False
        return record
    try:
        model, fit_info = ctx.fit(w_assignment, arm=arm_key)
    except (R.PackageCeiling, R.AccountFault):
        raise
    except Exception as exc:  # noqa: BLE001
        record["status"] = "FIT_FAILED"
        record["why"] = "%s: %s" % (type(exc).__name__, R._sanitize(exc))
        record["scores"] = {"block": block, "mean_smase": "UNKNOWN", "origins": {}}
        record["physical_fit"] = 0
        record["cache_hit"] = False
        return record
    record.update(fit_info)
    record["scores"] = ctx.predict_and_score_block(model, v_by_origin, block, save_dir=save_dir, tag=tag)
    complete = all(isinstance(p.get("mean_smase"), (int, float)) for p in record["scores"]["origins"].values()) and bool(record["scores"]["origins"])
    record["status"] = "READ" if complete else "INCOMPLETE"
    return record


def run_llm_unit(ctx: JobContext, fast: FastRunner, factory_for: Callable[[str], Callable[[], Any]],
                 *, arm: str, snapshot: Any, store: Any, controller: Any, h_label: str) -> dict[str, Any]:
    """q1 .. q4, candidate H, adoption, E outputs for one LLM arm on one job."""
    unit = UnitState(ctx, arm)
    st = unit.state
    st.setdefault("arm", arm)
    st.setdefault("job", ctx.job.job_id)
    st["initial_h"] = {"label": h_label, "runtime_bundle_sha": snapshot.runtime_bundle_sha}
    st.setdefault("started_utc", utc_now())
    unit.save()
    job_id = ctx.job.job_id
    state = ctx.state

    def ctx_for(query: str, phase: str, summary: Any = None) -> dict[str, Any]:
        out = {"job_phase": phase, "query": query}
        if summary is not None:
            out["task_research_summary"] = summary
        return out

    # ---- q1: initial solve on C_A ------------------------------------------------
    if not isinstance(st.get("q1"), dict) or st["q1"].get("status") != "READ":
        state.check_unit(job_id, arm)
        q1_dir = unit.dir / "q1"
        rows_w, w_asg, w_missing = fast.train_w(arm_key=unit.key("q1"), snapshot=snapshot, save_dir=q1_dir,
                                                context=ctx_for("q1", "adaptation"))
        unit.rows[("q1", "W")] = rows_w
        v_asg: dict[int, Any] = {}
        v_missing: dict[int, list[str]] = {}
        for origin in ctx.job.origins("C_A"):
            rows_v, asg, missing = fast.predict_v(arm_key=unit.key("q1"), snapshot=snapshot, origin=origin,
                                                  save_dir=q1_dir, context=ctx_for("q1", "adaptation"))
            unit.rows_by_origin[("q1", "V", origin)] = rows_v
            v_asg[origin] = asg
            v_missing[origin] = missing
        st["q1"] = run_scheme(ctx, tag="q1", arm_key=unit.key("q1"), w_assignment=w_asg, v_by_origin=v_asg,
                              block="C_A", save_dir=q1_dir, w_source="fast", v_source="fast",
                              missing_w=w_missing, missing_v=v_missing)
        st["q1"]["decision_validity"] = {
            "train_w": {r["series_uid"]: r.get("deploy_status") for r in rows_w},
            "predict_v": {str(o): {r["series_uid"]: r.get("deploy_status") for r in unit.rows_by_origin[("q1", "V", o)]} for o in v_asg},
        }
        unit.save()
    load_saved_rows(unit, ("q1", "q2"))
    unit.train_features = features_by_uid(unit.rows.get(("q1", "W")) or [])

    # ---- research call: plan q2 --------------------------------------------------
    plan_out = run_llm_stage(
        ctx, factory_for, arm_key=unit.key("research_q2"), snapshot=snapshot, stage="research_plan",
        case_id="%s-%s-q2-plan" % (job_id.lower(), arm.lower()), public_input=research_card(ctx, unit, purpose="q2_plan"),
        schema_name="research_plan_v1", schema=RESEARCH_PLAN_SCHEMA, checkpoint=unit.dir / "research_q2.json",
        extra={"runtime_capabilities": RUNTIME_CAPABILITIES},
    )
    plan = validate_plan(ctx, plan_out.get("payload") if plan_out.get("status") == "OK" else None)
    plan["call_status"] = plan_out.get("status")
    plan["call_detail"] = plan_out.get("detail")
    st["q2_plan"] = plan
    unit.save()

    # ---- q2: execute the declared contrast ---------------------------------------
    if not isinstance(st.get("q2"), dict) or (st["q2"].get("status") not in ("READ", "NOT_EXECUTED", "W_INCOMPLETE", "FIT_FAILED")
                                              and "fast" in (str(st["q2"].get("w_source")), str(st["q2"].get("v_source")))):
        q2_dir = unit.dir / "q2"
        q2_dir.mkdir(parents=True, exist_ok=True)
        q1 = st["q1"]
        q1_w = assignment_from_json(q1.get("w_assignment"))
        q1_v = {int(o): assignment_from_json(a) for o, a in (q1.get("v_assignment_by_origin") or {}).items()}
        purpose = filter_for_fast(plan.get("investigation_purpose_for_fast"))
        if plan["plan_status"] != "VALID" or plan["decision"] == "stop_and_reuse":
            st["q2"] = {
                "query": "q2", "block": "C_A", "status": "NOT_EXECUTED",
                "why": plan["plan_status"] if plan["plan_status"] != "VALID" else "stop_and_reuse",
                "logical_query": plan["plan_status"] == "VALID",
                "reuses": "q1", "w_assignment": q1.get("w_assignment"), "v_assignment_by_origin": q1.get("v_assignment_by_origin"),
                "scores": q1.get("scores"), "w_source": "reused:q1", "v_source": "reused:q1",
                "physical_fit": 0, "cache_hit": True,
            }
        else:
            factor = plan["changed_factor"]
            mode = plan["execution"]["mode"]
            w_asg, w_missing, w_source = q1_w, [], "reused:q1"
            v_asg, v_missing, v_source = dict(q1_v), {}, "reused:q1"
            if factor in ("train_side", "both"):
                if mode == "pinned_program":
                    w_asg, w_source = uniform(plan["pinned_train_steps"]), "pinned:" + plan["pinned_train_label"]
                else:
                    state.check_unit(job_id, arm)
                    rows_w, w_asg, w_missing = fast.train_w(arm_key=unit.key("q2"), snapshot=snapshot, save_dir=q2_dir,
                                                            context=ctx_for("q2", "adaptation", {
                                                                "kind": "contrast_query", "changed_factor": factor,
                                                                "investigation_purpose": purpose,
                                                                "note": "this side is being re-solved as a contrast to the previous solve; scores are not visible to this role"}))
                    unit.rows[("q2", "W")] = rows_w
                    w_source = "fast"
            if factor in ("predict_side", "both"):
                if mode == "pinned_program":
                    v_asg = by_origin(ctx.job, "C_A", uniform(plan["pinned_predict_steps"]))
                    v_source = "pinned:" + plan["pinned_predict_label"]
                else:
                    v_asg = {}
                    for origin in ctx.job.origins("C_A"):
                        state.check_unit(job_id, arm)
                        rows_v, asg, missing = fast.predict_v(arm_key=unit.key("q2"), snapshot=snapshot, origin=origin,
                                                              save_dir=q2_dir, context=ctx_for("q2", "adaptation", {
                                                                  "kind": "contrast_query", "changed_factor": factor,
                                                                  "investigation_purpose": purpose,
                                                                  "note": "this side is being re-solved as a contrast to the previous solve; scores are not visible to this role"}))
                        unit.rows_by_origin[("q2", "V", origin)] = rows_v
                        v_asg[origin] = asg
                        v_missing[origin] = missing
                    v_source = "fast"
            st["q2"] = run_scheme(ctx, tag="q2", arm_key=unit.key("q2"), w_assignment=w_asg, v_by_origin=v_asg,
                                  block="C_A", save_dir=q2_dir, w_source=w_source, v_source=v_source,
                                  missing_w=w_missing, missing_v=v_missing)
            st["q2"]["contrast"] = {"changed_factor": factor, "mode": mode,
                                    "single_factor_attribution": factor in ("train_side", "predict_side"),
                                    "label": "controlled_contrast" if factor in ("train_side", "predict_side") else "candidate_search"}
        unit.save()

    # ---- q2 checkpoint: better of q1/q2 on C_A -----------------------------------
    if "q2_checkpoint" not in st:
        q1m = (st["q1"].get("scores") or {}).get("mean_smase")
        q2m = (st["q2"].get("scores") or {}).get("mean_smase")
        chosen = "q1"
        if st["q2"].get("status") == "READ" and isinstance(q1m, (int, float)) and isinstance(q2m, (int, float)) and (q1m - q2m) >= MATERIAL:
            chosen = "q2"
        elif st["q2"].get("status") == "READ" and not isinstance(q1m, (int, float)) and isinstance(q2m, (int, float)):
            chosen = "q2"
        st["q2_checkpoint"] = {"chosen": chosen, "q1_mean": q1m, "q2_mean": q2m, "rule": "q2 replaces q1 only if C_A mean sMASE improves by >= %s" % MATERIAL}
        unit.save()

    # ---- frozen task-internal summary (structural; before any candidate H) -------
    if "summary" not in st:
        q2 = st["q2"]
        purpose = filter_for_fast(plan.get("investigation_purpose_for_fast"))
        contrast = q2.get("contrast") or {}
        if q2.get("status") in ("READ", "INCOMPLETE"):
            q2_line = "q2: contrast on %s via %s" % (contrast.get("changed_factor"), contrast.get("mode"))
            if str(q2.get("w_source", "")).startswith("pinned"):
                q2_line += " (train side pinned: %s)" % plan.get("pinned_train_label")
            if str(q2.get("v_source", "")).startswith("pinned"):
                q2_line += " (predict side pinned: %s)" % plan.get("pinned_predict_label")
        else:
            q2_line = "q2: not executed (%s)" % q2.get("why")
        st["summary"] = {
            "kind": "task_internal_research_summary",
            "frozen_before_candidate_knowledge": True,
            "queries_so_far": [
                "q1: initial per-sequence solve (train W and predict V) under the current knowledge on block C_A",
                q2_line,
            ],
            "investigation_purpose": purpose,
            "contrast_constraint": "the other side kept the previous exact assignment" if contrast.get("single_factor_attribution") else "none",
            "legal_observations": "the training prefix and the serve contexts you are given; no scores reach this role",
            "next": "re-solve on block C_B with the same permissions; later deployment inputs are prepared with no feedback",
        }
        st["summary_filter"] = {"original_purpose": plan.get("investigation_purpose_for_fast"), "delivered": purpose}
        unit.save()
    summary = st["summary"]

    # ---- candidate H (A3/A5) or same-H second branch (F / no proposal) -----------
    candidate = None
    if arm in ("A3", "A5"):
        bdir = unit.dir / "boundary"
        bdir.mkdir(parents=True, exist_ok=True)
        boundary = run_skill_boundary(ctx, factory_for, arm_key=unit.key("slow_boundary"), parent=snapshot,
                                      card=boundary_card(ctx, unit), store=store, controller=controller,
                                      checkpoint=bdir / "boundary.json", exposure_contexts=exposure_contexts(ctx, blocks=("C_B", "E")))
        st["boundary"] = {k: v for k, v in boundary.items() if k != "candidate_snapshot"}
        candidate = boundary.get("candidate_snapshot") if boundary.get("treatment") == "UPDATE_TREATMENT" else None
        unit.save()
    else:
        reflection = run_llm_stage(
            ctx, factory_for, arm_key=unit.key("research_reflect"), snapshot=snapshot, stage="research_reflection",
            case_id="%s-%s-boundary-reflection" % (job_id.lower(), arm.lower()),
            public_input={**research_card(ctx, unit, purpose="fixed_h_reflection"),
                          "what_you_may_change": "only this job's temporary research state: the investigation purpose text given to Fast (filtered) and a private note; knowledge is not editable in this arm"},
            schema_name="research_reflection_v1", schema=REFLECTION_SCHEMA, checkpoint=unit.dir / "research_reflect.json",
            extra={"runtime_capabilities": RUNTIME_CAPABILITIES},
        )
        st["boundary"] = {"outcome": "FIXED_H_REFLECTION", "treatment": "NO_UPDATE_TREATMENT", "call_status": reflection.get("status")}
        if reflection.get("status") == "OK" and "summary_after_reflection" not in st:
            updated = filter_for_fast((reflection.get("payload") or {}).get("updated_investigation_purpose_for_fast"))
            st["temporary_research_state"] = reflection.get("payload")
            st["summary_after_reflection"] = {**summary, "investigation_purpose": updated,
                                              "reflection_applied": True}
            st["summary_filter"]["reflection_original"] = (reflection.get("payload") or {}).get("updated_investigation_purpose_for_fast")
            st["summary_filter"]["reflection_delivered"] = updated
        unit.save()
        if isinstance(st.get("summary_after_reflection"), dict):
            summary = st["summary_after_reflection"]
    same_h_second_branch = candidate is None
    st["q4_knowledge"] = ("candidate_h" if candidate is not None else "same_h_second_branch")

    # ---- q3 / q4: re-solve C_B under old / new H ----------------------------------
    def branch(tag: str, snap: Any) -> dict[str, Any]:
        if isinstance(st.get(tag), dict) and st[tag].get("status") == "READ":
            return st[tag]
        state.check_unit(job_id, arm)
        bdir = unit.dir / tag
        rows_w, w_asg, w_missing = fast.train_w(arm_key=unit.key(tag), snapshot=snap, save_dir=bdir,
                                                context=ctx_for(tag, "adaptation", summary))
        unit.rows[(tag, "W")] = rows_w
        v_asg: dict[int, Any] = {}
        v_missing: dict[int, list[str]] = {}
        for origin in ctx.job.origins("C_B"):
            rows_v, asg, missing = fast.predict_v(arm_key=unit.key(tag), snapshot=snap, origin=origin, save_dir=bdir,
                                                  context=ctx_for(tag, "adaptation", summary))
            unit.rows_by_origin[(tag, "V", origin)] = rows_v
            v_asg[origin] = asg
            v_missing[origin] = missing
        rec = run_scheme(ctx, tag=tag, arm_key=unit.key(tag), w_assignment=w_asg, v_by_origin=v_asg, block="C_B",
                         save_dir=bdir, w_source="fast", v_source="fast", missing_w=w_missing, missing_v=v_missing)
        rec["knowledge_sha"] = snap.runtime_bundle_sha
        rec["decision_validity"] = {
            "train_w": {r["series_uid"]: r.get("deploy_status") for r in rows_w},
            "predict_v": {str(o): {r["series_uid"]: r.get("deploy_status") for r in unit.rows_by_origin[(tag, "V", o)]} for o in v_asg},
        }
        return rec

    errors: list[BaseException] = []
    results: dict[str, dict[str, Any]] = {}

    def run_branch(tag: str, snap: Any) -> None:
        try:
            results[tag] = branch(tag, snap)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    q4_snapshot = candidate if candidate is not None else snapshot
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_branch, "q3", snapshot), pool.submit(run_branch, "q4", q4_snapshot)]
        for fut in as_completed(futures):
            fut.result()
    if errors:
        raise errors[0]
    st["q3"] = results["q3"]
    st["q4"] = results["q4"]
    unit.save()

    # ---- adoption on C_B ---------------------------------------------------------
    if "adoption" not in st:
        q3m = (st["q3"].get("scores") or {}).get("mean_smase")
        q4m = (st["q4"].get("scores") or {}).get("mean_smase")
        adopted = "q3"
        why = "q3 kept"
        if st["q4"].get("status") == "READ" and isinstance(q3m, (int, float)) and isinstance(q4m, (int, float)) and (q3m - q4m) >= MATERIAL:
            adopted, why = "q4", "q4 improves C_B mean sMASE by >= MATERIAL"
        elif st["q4"].get("status") == "READ" and not isinstance(q3m, (int, float)) and isinstance(q4m, (int, float)):
            adopted, why = "q4", "q3 unreadable, q4 readable"
        st["adoption"] = {
            "adopted": adopted, "why": why, "q3_mean": q3m, "q4_mean": q4m,
            "q4_knowledge": st["q4_knowledge"],
            "adopted_h_sha": (candidate.runtime_bundle_sha if (adopted == "q4" and candidate is not None) else snapshot.runtime_bundle_sha),
            "this_is_a_development_adoption_mark_not_production_safety_approval": True,
        }
        unit.save()

    # ---- E outputs: three frozen branches ---------------------------------------
    edir = unit.dir / "E"
    edir.mkdir(parents=True, exist_ok=True)
    st.setdefault("E", {})
    ckpt = st["q2_checkpoint"]["chosen"]
    ckpt_rec = st[ckpt]
    branches = {
        "q2ckpt": {"h": snapshot, "record": ckpt_rec, "summary": summary, "h_label": "old"},
        "q3": {"h": snapshot, "record": st["q3"], "summary": summary, "h_label": "old"},
        "q4": {"h": q4_snapshot, "record": st["q4"], "summary": summary, "h_label": ("candidate" if candidate is not None else "old(second branch)")},
    }
    for name, info in branches.items():
        if isinstance(st["E"].get(name), dict) and st["E"][name].get("frozen"):
            continue
        rec = info["record"]
        w_json = rec.get("w_assignment")
        if not w_json:
            st["E"][name] = {"frozen": False, "why": "no complete W / model for this branch", "h_sha": info["h"].runtime_bundle_sha}
            unit.save()
            continue
        w_asg = assignment_from_json(w_json)
        model, fit_info = ctx.fit(w_asg, arm=unit.key("E-" + name))
        v_source = str(rec.get("v_source") or "fast")
        v_by: dict[int, Any] = {}
        if v_source.startswith("pinned:"):
            steps = None
            for uid_steps in (rec.get("v_assignment_by_origin") or {}).values():
                if uid_steps:
                    steps = R.steps_from_json(next(iter(uid_steps.values())))
                    break
            v_by = by_origin(ctx.job, "E", uniform(steps))
            v_mode = v_source
        else:
            v_mode = "fast"
            for origin in ctx.job.origins("E"):
                state.check_unit(job_id, arm)
                rows_v, asg, _missing = fast.predict_v(arm_key=unit.key("E-" + name), snapshot=info["h"], origin=origin,
                                                       save_dir=edir / name, context=ctx_for("deployment", "deployment", info["summary"]))
                v_by[origin] = asg
        frozen = ctx.freeze_predictions(model, v_by, save_dir=edir / name, tag=name)
        st["E"][name] = {"frozen": all(v == "FROZEN" for v in frozen.values()), "origins": frozen,
                         "h_sha": info["h"].runtime_bundle_sha, "h_label": info["h_label"], "model_file": fit_info.get("model_file"),
                         "v_mode": v_mode, "e_fit_was_cache_hit": fit_info.get("cache_hit")}
        unit.save()
    st["main_deployment_branch"] = st["adoption"]["adopted"]
    st["ended_utc"] = utc_now()
    st["unit_cost"] = {"api_attempts": state.unit_api(job_id, arm), "tokens": state.unit_tokens(job_id, arm),
                       "fast_session_seconds": round(state.unit_active_seconds(job_id, arm), 1)}
    st["status"] = "COMPLETED"
    unit.save()
    return st


def validate_plan(ctx: JobContext, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"plan_status": "NO_PLAN", "decision": None, "changed_factor": None, "execution": {"mode": "none"}}
    plan = json.loads(json.dumps(payload, default=R._json_default))
    plan["plan_status"] = "VALID"
    decision = plan.get("decision")
    factor = plan.get("changed_factor")
    execution = dict(plan.get("execution") or {})
    mode = execution.get("mode")
    if decision == "stop_and_reuse":
        return plan
    if factor not in ("train_side", "predict_side", "both") or mode not in ("pinned_program", "fast_regenerate"):
        plan["plan_status"] = "INVALID_PLAN"
        plan["invalid_why"] = "run_contrast requires changed_factor in train_side/predict_side/both and mode pinned_program/fast_regenerate"
        return plan
    if mode == "pinned_program":
        for side, key in (("train_side", "pinned_train_program"), ("predict_side", "pinned_predict_program")):
            if factor in (side, "both"):
                raw = execution.get(key)
                if raw is None:
                    plan["plan_status"] = "INVALID_PLAN"
                    plan["invalid_why"] = "%s must be given ([] = identity) for %s" % (key, factor)
                    return plan
                steps = None if len(raw) == 0 else tuple((str(s["op"]), dict(s.get("params") or {})) for s in raw)
                ok, label = legal_steps(steps)
                if not ok:
                    plan["plan_status"] = "INVALID_PLAN"
                    plan["invalid_why"] = "%s: %s" % (key, label)
                    return plan
                if steps is not None:
                    try:
                        if side == "train_side":
                            R.assert_train_windows_legal(ctx.spec, ctx.bundle, uniform(steps), training_design="whole_window")
                        else:
                            for origin in ctx.job.origins("C_A"):
                                for uid in UIDS:
                                    raw_ctx = np.asarray(ctx.bundle.values[uid], dtype=np.float64)[origin - CONTEXT_LENGTH:origin]
                                    program = Program.from_steps(list(steps), source="dev_train3")
                                    artifact = verify_candidate(Candidate.program_candidate("plan-legality", program, source="dev_train3"),
                                                                raw_ctx, allowed_operators=tuple(R.legal_forecast_ops()),
                                                                maximum_modified_fraction=1.0, require_finite_output=False)
                                    if artifact.receipt.status == "rejected":
                                        raise numeric.PreparationFailed("rejected on %s@%d: %s" % (uid, origin, artifact.receipt.rejection_code))
                    except Exception as exc:  # noqa: BLE001
                        plan["plan_status"] = "INVALID_PLAN"
                        plan["invalid_why"] = "%s illegal on this job: %s" % (key, R._sanitize("%s: %s" % (type(exc).__name__, exc)))
                        return plan
                prefix = "pinned_train" if side == "train_side" else "pinned_predict"
                plan[prefix + "_steps"] = R.steps_to_json(steps) if steps is not None else None
                plan[prefix + "_label"] = label
                plan[prefix + "_in_menu"] = label in MENU_LABELS or steps is None
    # steps kept as tuples for execution
    for prefix in ("pinned_train", "pinned_predict"):
        if prefix + "_steps" in plan:
            plan[prefix + "_steps"] = R.steps_from_json(plan[prefix + "_steps"])
    return plan


# ---------------------------------------------------------------------------
# Slow boundary (candidate H) with exposure check
# ---------------------------------------------------------------------------

def exposure_contexts(ctx: JobContext, *, blocks: Sequence[str]) -> list[tuple[str, str, np.ndarray]]:
    out: list[tuple[str, str, np.ndarray]] = []
    for uid in UIDS:
        raw = np.asarray(ctx.bundle.values[uid], dtype=np.float64)
        out.append(("train_workflow", uid, raw[:TRAIN_PREFIX_END]))
        for block in blocks:
            for origin in ctx.job.origins(block):
                out.append(("predict_input@%d" % origin, uid, raw[origin - CONTEXT_LENGTH:origin]))
    return out


def exposure_report(parent: Any, candidate: Any, contexts: Sequence[tuple[str, str, np.ndarray]]) -> dict[str, Any]:
    rows = []
    for role, uid, observed in contexts:
        features = dict(R.extract_public_features(np.asarray(observed, dtype=np.float64), task_kind="forecast"))
        before = know.fast_view_bytes(parent, features)
        after = know.fast_view_bytes(candidate, features)
        rows.append({"decision_role": role, "series_uid": uid, "knowledge_view_differs": before != after})
    exposed = sum(1 for r in rows if r["knowledge_view_differs"])
    return {"decisions_checked": len(rows), "decisions_the_edit_reaches": exposed,
            "outcome": "EXPOSED" if exposed else "COMPILED_BUT_NEVER_EXPOSED", "per_decision": rows,
            "cost": "zero fits, zero LLM"}


def run_skill_boundary(ctx: JobContext, factory_for: Callable[[str], Callable[[], Any]], *, arm_key: str, parent: Any,
                       card: Mapping[str, Any], store: Any, controller: Any, checkpoint: Path,
                       exposure_contexts: Sequence[tuple[str, str, np.ndarray]]) -> dict[str, Any]:
    """Slow proposes at most one edit; compile into an isolated fork; check delivery."""
    machinery = ctx.machinery
    if checkpoint.is_file():
        saved = R.read_json(checkpoint)
        saved["resumed_from_checkpoint"] = True
        sha = (saved.get("applied") or {}).get("runtime_bundle_sha")
        if saved.get("treatment") == "UPDATE_TREATMENT" and sha:
            fork = Path(store.root) / str(sha)
            recompiled = machinery["compile_snapshot"](fork, verify_lock=False)
            if recompiled.runtime_bundle_sha != sha:
                raise RuntimeError("recompiled Slow fork SHA mismatch")
            saved["candidate_snapshot"] = recompiled
        return saved
    R.atomic_write_json(checkpoint.with_name("slow_input.json"), card)
    parent_mat = store.materialize(parent)
    catalog = know.build_catalog(controller=controller, parent=parent_mat)
    out: dict[str, Any] = {"arm_key": arm_key, "parent_runtime_bundle_sha": parent.runtime_bundle_sha,
                           "surfaces_offered": [row["surface_id"] for row in catalog], "started_utc": utc_now()}
    if not catalog:
        out.update(outcome="NO_AUTHORISED_SURFACE", treatment="NO_UPDATE_TREATMENT")
        R.atomic_write_json(checkpoint, out)
        return out
    agents: list[Any] = []
    extra = {"runtime_capabilities": RUNTIME_CAPABILITIES, "consumer_structure": dict(R.CONSUMER_STRUCTURE),
             "cap_facts": W.public_cap_facts(ctx.spec)}
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent

    def slow_factory() -> Any:
        agent = TTHASlowAgent(make_core(ctx, factory_for, arm_key, extra))
        agents.append(agent)
        return agent

    proposal = know.propose_update(slow_factory=slow_factory, card=card, catalog=catalog, snapshot=parent,
                                   task_context=ctx.task_ctx, attempts=SLOW_ATTEMPTS)
    why = str(proposal.get("why") or "")
    if why == "ACCOUNT_OR_PERMISSION_FAULT":
        raise R.AccountFault(R._sanitize(proposal.get("detail") or why))
    out["proposal"] = {k: v for k, v in proposal.items() if k != "manifest"}
    out["raw_responses"] = [
        {"assistant_text": str(getattr(getattr(a.last_stage_result, "response", None), "assistant_text", "") or ""),
         "no_proposal_reason": a.last_no_proposal_reason} for a in agents]
    out["llm_requests_sent"] = int(sum(a.core.metered.calls for a in agents))
    if not proposal.get("proposed"):
        out["outcome"] = {"SLOW_ABSTAINED": "NO_UPDATE", "TRANSPORT_TRANSIENT_FAULT": "BOUNDARY_TRANSPORT_FAULT"}.get(why, "NO_PROPOSAL")
        out["treatment"] = "NO_UPDATE_TREATMENT"
        R.atomic_write_json(checkpoint, out)
        return out
    applied = know.apply_update(controller=controller, store=store, snapshot=parent, manifest=proposal["manifest"], catalog=catalog)
    out["applied"] = {k: v for k, v in applied.items() if k != "snapshot"}
    if not applied.get("applied"):
        out.update(outcome="INVALID_EDIT", treatment="NO_UPDATE_TREATMENT")
        R.atomic_write_json(checkpoint, out)
        return out
    candidate = applied["snapshot"]
    exposure = exposure_report(parent, candidate, exposure_contexts)
    out["exposure"] = {k: v for k, v in exposure.items() if k != "per_decision"}
    out["exposure"]["per_decision_file"] = "exposure.json"
    R.atomic_write_json(checkpoint.with_name("exposure.json"), exposure)
    if exposure["outcome"] != "EXPOSED":
        out.update(outcome="COMPILED_BUT_NEVER_EXPOSED", treatment="NO_UPDATE_TREATMENT")
        out["compiled_candidate_receipt_preserved"] = True
        R.atomic_write_json(checkpoint, out)
        return out
    out.update(outcome="CANDIDATE_COMPILED", treatment="UPDATE_TREATMENT")
    out["ended_utc"] = utc_now()
    R.atomic_write_json(checkpoint, out)
    out["candidate_snapshot"] = candidate
    return out


# ---------------------------------------------------------------------------
# Source guidance formation (§6.1) and J1 -> J2 integration (§6.5)
# ---------------------------------------------------------------------------

def compress_source_material() -> dict[str, Any]:
    """Only the §4.1 material: TRAIN2 formation records, compressed, failures kept."""
    contract = R.read_json(SOURCE_RUN / "run_contract.json")
    material = R.read_json(SOURCE_RUN / "fixed_workflow_material.json")
    slow_input = R.read_json(SOURCE_RUN / "slow_input.json")
    formation_origins = [int(o) for _p, o in contract["formation_units"]]

    workflows = []
    for reading in material.get("readings") or []:
        row = {"label": reading.get("label"), "typed_steps": reading.get("typed_steps"), "status": reading.get("status"),
               "fit": reading.get("fit"), "roles": {}}
        for role_name, role in (reading.get("roles") or {}).items():
            origins = {}
            for item in role.get("origins") or []:
                pop = item.get("population") or {}
                origins[str(item.get("origin"))] = {
                    "status": item.get("status", "READ"),
                    "mean_smase": pop.get("mean_smase"), "per_uid_smase": pop.get("per_uid_smase"),
                    "per_uid_utility_vs_static": pop.get("per_uid_utility_vs_static"), "risk": pop.get("risk"),
                    "n_unknown": pop.get("n_unknown"),
                }
            row["roles"][role_name] = {"W": role.get("W"), "V": role.get("V"), "status": role.get("status"),
                                      "mean_formation_smase": role.get("mean_formation_smase"), "origins": origins}
        mod = reading.get("train_modification") or {}
        row["train_modification"] = {"behavior_point_count": mod.get("behavior_point_count"), "n_windows": mod.get("n_windows")}
        workflows.append(row)

    members = []
    for member in (slow_input.get("training_batch") or {}).get("members") or []:
        members.append({"series_uid": member.get("series_uid"), "deployed_program": member.get("deployed_program"),
                        "typed_steps": member.get("typed_steps"), "valid": member.get("valid"),
                        "public_features": member.get("public_features")})
    decisions = []
    for key in ("failing_eval_decisions", "matched_successes"):
        for item in slow_input.get(key) or []:
            decisions.append({"series_uid": item.get("series_uid"), "origin": item.get("origin"),
                              "deployed_program": item.get("deployed_program"), "typed_steps": item.get("typed_steps"),
                              "utility_vs_static": item.get("utility"), "sign": item.get("sign"), "valid": item.get("valid"),
                              "public_features": item.get("public_features")})
    aggregate = (slow_input.get("training_batch") or {}).get("aggregate_by_origin")
    return {
        "source_job_frame": {
            "n_train": len(contract["train_uids"]), "n_eval": len(contract["eval_uids"]),
            "train_uids": contract["train_uids"], "eval_uids": contract["eval_uids"],
            "training_prefix_end": contract["training_prefix_end"], "anchors": contract["anchors"],
            "formation_origins": formation_origins, "period": contract["period"],
            "consumer": contract["consumer"], "maximum_modified_fraction": contract["maximum_modified_fraction"],
            "source_material_is_a_different_data_source_than_the_next_job": True,
        },
        "seven_uniform_workflows_all_roles_full_population": workflows,
        "identity_reference": material.get("identity_reference"),
        "agent_formation_under_h0": {
            "training_workflows": members,
            "prediction_decisions_with_utility": decisions,
            "aggregate_by_origin": aggregate,
            "note": "origin with UNKNOWN mean had two failed decisions; not filled",
        },
        "case_groups": slow_input.get("case_groups"),
        "excluded_by_protocol": ["later evaluation origins of the source run", "the source run's final results",
                                 "the source run's later rejected card", "any result from the next jobs"],
    }


def source_formation_card(compressed: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pattern_id": "devtrain3-source-formation",
        "observable_signature": {"task_kind": "forecast", "decision_role": "slow",
                                 "scope": "cross-source consolidation of formation evidence into soft guidance"},
        "consumer_structure": dict(R.CONSUMER_STRUCTURE),
        "runtime_capabilities": RUNTIME_CAPABILITIES,
        "known_candidate_menu": known_candidate_menu(),
        "what_this_material_is": (
            "formation evidence from a previous job on a different data source: seven uniform workflows scored on "
            "the full population in three roles (train_and_serve / train_only / serve_only), plus the agent's own "
            "per-sequence decisions under the current knowledge with their utilities.  Numbers are observations; "
            "a per-series utility under a shared model is not that series' own causal effect."
        ),
        "next_job": (
            "a new job on another data source with six series, the same consumer, the same operators, cap 1.0, "
            "a small feedback budget during adaptation and no feedback after freeze.  Guidance written here "
            "is soft: it is calibrated by that job's own feedback and may be overridden."
        ),
        "source_material": compressed,
        "what_you_may_write": (
            "Exactly one edit on one surface from writable_surface_catalog, or no_proposal.  Allowed surfaces are the "
            "bootstrap procedure bodies, the candidate-policy guidance strings, or one new guidance capability entry.  "
            "Do not write a frozen program card; do not use series ids or a data-source name as an applicability reason; "
            "applicability must be observable features of the sequence being decided."
        ),
        "observations_vs_hypotheses": "Numbers on this card are observations.  Mechanism talk in a proposal is a hypothesis and may be wrong.",
        "select_cannot_veto_deployment": "Fast's chosen program is delivered after legality; no Support admission.",
    }


def integration_card(ctx: JobContext, unit_state: Mapping[str, Any]) -> dict[str, Any]:
    st = dict(unit_state)
    return {
        "pattern_id": "devtrain3-j1-to-j2-integration",
        "observable_signature": {"task_kind": "forecast", "decision_role": "slow",
                                 "scope": "boundary integration of one job's legal held-in experience for the next job"},
        "consumer_structure": dict(R.CONSUMER_STRUCTURE),
        "runtime_capabilities": RUNTIME_CAPABILITIES,
        "known_candidate_menu": known_candidate_menu(),
        "job_frame": job_frame(ctx),
        "static_reference": {block: static_public(ctx, block) for block in ("C_A", "C_B")},
        "held_in_queries": [query_public_record(st[q]) for q in ("q1", "q2", "q3", "q4") if isinstance(st.get(q), dict)],
        "research_plan_q2": st.get("q2_plan"),
        "boundary_in_that_job": st.get("boundary"),
        "adoption_in_that_job": st.get("adoption"),
        "train_prefix_public_features_by_uid": st.get("train_features"),
        "next_job": ("a later job on the same data source with the same six series but a different training block and "
                     "different feedback blocks; the same consumer, operators and budget; later-block results of this job "
                     "are not included and did not participate"),
        "what_you_may_write": (
            "Exactly one edit on one surface from writable_surface_catalog, or no_proposal.  Guidance prose only; "
            "no frozen program card; applicability by observable features only; keeping the current knowledge is legal."
        ),
        "observations_vs_hypotheses": "Numbers on this card are observations; mechanism talk is hypothesis.",
        "select_cannot_veto_deployment": "Fast's chosen program is delivered after legality; no Support admission.",
    }


# ---------------------------------------------------------------------------
# numeric arms: static, source-fixed, source-order search, random search
# ---------------------------------------------------------------------------

def run_static(ctx: JobContext) -> dict[str, Any]:
    sdir = ctx.dir / "static"
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / "static.json"
    model, fit_info = ctx.fit(uniform(None), arm="%s/static" % ctx.job.job_id)
    out: dict[str, Any] = {"arm": "static", "model_file": fit_info["model_file"], "blocks": {}}
    for block in ("C_A", "C_B"):
        block_out = ctx.predict_and_score_block(model, by_origin(ctx.job, block, uniform(None)), block, save_dir=sdir, tag="static")
        ctx.static_ref[block] = {int(o): pop for o, pop in block_out["origins"].items()}
        out["blocks"][block] = block_out
    out["E"] = ctx.freeze_predictions(model, by_origin(ctx.job, "E", uniform(None)), save_dir=sdir, tag="static")
    out["reference_readings_registered_separately"] = True
    R.atomic_write_json(path, out)
    return out


def run_fixed(ctx: JobContext, *, arm: str, label: str) -> dict[str, Any]:
    fdir = ctx.dir / arm
    fdir.mkdir(parents=True, exist_ok=True)
    steps = menu_steps(label)
    model, fit_info = ctx.fit(uniform(steps), arm="%s/%s" % (ctx.job.job_id, arm))
    out: dict[str, Any] = {"arm": arm, "label": label, "typed_steps": R.steps_to_json(steps), "model_file": fit_info["model_file"],
                           "physical_fit": fit_info["physical_fit"], "blocks": {}}
    for block in ("C_A", "C_B"):
        out["blocks"][block] = ctx.predict_and_score_block(model, by_origin(ctx.job, block, uniform(steps)), block, save_dir=fdir, tag=arm)
    out["held_in_readings_are_diagnostic_not_consumed"] = True
    out["E"] = ctx.freeze_predictions(model, by_origin(ctx.job, "E", uniform(steps)), save_dir=fdir, tag=arm)
    R.atomic_write_json(fdir / (arm + ".json"), out)
    return out


def source_gain_ranking(compressed: Mapping[str, Any]) -> list[dict[str, Any]]:
    identity_mean = None
    rows = []
    for wf in compressed["seven_uniform_workflows_all_roles_full_population"]:
        role = (wf.get("roles") or {}).get("train_and_serve") or {}
        mean = role.get("mean_formation_smase")
        if wf["label"] == "identity":
            identity_mean = mean
        rows.append((wf["label"], mean, role.get("status")))
    out = []
    for label, mean, status in rows:
        if label == "identity":
            continue
        gain = (float(identity_mean) - float(mean)) if isinstance(mean, (int, float)) and isinstance(identity_mean, (int, float)) else "UNKNOWN"
        out.append({"label": label, "source_mean_smase": mean, "source_mean_gain": gain, "status": status})
    readable = [r for r in out if isinstance(r["source_mean_gain"], float)]
    unread = [r for r in out if not isinstance(r["source_mean_gain"], float)]
    readable.sort(key=lambda r: -r["source_mean_gain"])
    return readable + unread


def _simpler(label_a: str, label_b: str) -> str:
    def n(label: str) -> int:
        return 0 if label == "identity" else len(label.split(">"))
    return label_a if n(label_a) <= n(label_b) else label_b


def pick(readings: Mapping[str, Any], candidates: Sequence[str]) -> str:
    """Lowest mean sMASE; within MATERIAL of the best, the simpler program wins."""
    best = None
    for label in candidates:
        mean = readings.get(label)
        if not isinstance(mean, (int, float)):
            continue
        if best is None or mean < readings[best] - MATERIAL:
            best = label
        elif abs(mean - readings[best]) <= MATERIAL:
            best = _simpler(best, label)
    return best if best is not None else "identity"


def run_search(ctx: JobContext, *, arm: str, order: Sequence[str], order_source: Mapping[str, Any]) -> dict[str, Any]:
    """Two cheap searches: q1/q2 on C_A, q3 re-check + q4 next on C_B, pick on C_B."""
    sdir = ctx.dir / arm
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / (arm + ".json")
    if path.is_file():
        saved = R.read_json(path)
        if saved.get("status") == "COMPLETED":
            return saved
    order = list(order)
    out: dict[str, Any] = {"arm": arm, "order": order, "order_source": dict(order_source), "queries": {}, "seed": SEED}

    def query(tag: str, label: str, block: str) -> dict[str, Any]:
        steps = menu_steps(label)
        rec = run_scheme(ctx, tag=tag, arm_key="%s/%s/%s" % (ctx.job.job_id, arm, tag), w_assignment=uniform(steps),
                         v_by_origin=by_origin(ctx.job, block, uniform(steps)), block=block, save_dir=sdir,
                         w_source="uniform:" + label, v_source="uniform:" + label)
        rec["label"] = label
        out["queries"][tag] = rec
        return rec

    q1 = query("q1", order[0], "C_A")
    q2 = query("q2", order[1], "C_A")
    ca_static = fmean_or_unknown([pop.get("mean_smase") for pop in ctx.static_ref["C_A"].values()])
    ca = {order[0]: (q1.get("scores") or {}).get("mean_smase"), order[1]: (q2.get("scores") or {}).get("mean_smase"), "identity": ca_static}
    ckpt = pick(ca, ["identity", order[0], order[1]])
    out["q2_checkpoint"] = {"chosen": ckpt, "C_A_readings": ca}
    q3 = query("q3", ckpt, "C_B")
    tested = {order[0], order[1], ckpt}
    next_label = next((label for label in order if label not in tested), None)
    if next_label is None:
        out["queries"]["q4"] = {"query": "q4", "status": "NOT_EXECUTED", "why": "no untested candidate left"}
        q4_label = None
    else:
        query("q4", next_label, "C_B")
        q4_label = next_label
    cb_static = fmean_or_unknown([pop.get("mean_smase") for pop in ctx.static_ref["C_B"].values()])
    cb = {ckpt: (q3.get("scores") or {}).get("mean_smase"), "identity": cb_static}
    if q4_label:
        cb[q4_label] = (out["queries"]["q4"].get("scores") or {}).get("mean_smase")
    final = pick(cb, [label for label in ["identity", ckpt] + ([q4_label] if q4_label else []) if label is not None])
    out["final"] = {"chosen": final, "C_B_readings": cb, "rule": "lowest C_B mean sMASE; identity is a floor; within %s the simpler wins" % MATERIAL}
    for name, label in (("q2ckpt", ckpt), ("final", final)):
        steps = menu_steps(label)
        model, fit_info = ctx.fit(uniform(steps), arm="%s/%s/E-%s" % (ctx.job.job_id, arm, name))
        out.setdefault("E", {})[name] = {"label": label, "model_file": fit_info["model_file"],
                                        "origins": ctx.freeze_predictions(model, by_origin(ctx.job, "E", uniform(steps)), save_dir=sdir / "E", tag=name)}
    out["logical_queries"] = sum(1 for q in out["queries"].values() if q.get("logical_query"))
    out["physical_fits_this_arm"] = sum(int(q.get("physical_fit") or 0) for q in out["queries"].values())
    out["j1_readings_for_reorder"] = {
        label: {block: rec["scores"]["mean_smase"] for tag, rec in out["queries"].items() for block in [rec.get("block")]
                if rec.get("label") == label and rec.get("status") == "READ"}
        for label in set(rec.get("label") for rec in out["queries"].values() if rec.get("label"))
    }
    out["status"] = "COMPLETED"
    R.atomic_write_json(path, out)
    return out


def reorder_from_j1(source_order: Sequence[str], j1: Mapping[str, Any], j1_static: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Source-order J2: tested candidates first by equal-weight mean gain over distinct tested blocks."""
    readings = j1.get("j1_readings_for_reorder") or {}
    gains: dict[str, Any] = {}
    for label, blocks in readings.items():
        if label == "identity":
            continue
        per_block = []
        for block, mean in blocks.items():
            static_mean = j1_static.get(block)
            if isinstance(mean, (int, float)) and isinstance(static_mean, (int, float)):
                per_block.append(float(static_mean) - float(mean))
        if per_block:
            gains[label] = float(statistics.fmean(per_block))
    tested = sorted(gains, key=lambda label: -gains[label])
    untested = [label for label in source_order if label not in gains]
    return tested + untested, {"j1_mean_gain_by_tested_label": gains, "rule": "tested by equal-weight mean gain over distinct tested blocks, then untested in source order; exact replays add no weight; UNKNOWN not zero-filled"}


# ---------------------------------------------------------------------------
# course
# ---------------------------------------------------------------------------

def course_config(job_specs: Mapping[str, JobSpec], *, source_order: Sequence[str], rnd_order: Sequence[str],
                  source_fixed_label: str, h0_sha: str, run_sha: str) -> dict[str, Any]:
    return {
        "package": PACKAGE, "seed": SEED, "stations": list(STATIONS), "uid_map": dict(STATION_BY_UID),
        "jobs": {job: {"block_start": spec.block_start, "training_prefix": [0, TRAIN_PREFIX_END], "anchors": list(ANCHORS),
                       "blocks": {b: list(o) for b, o in BLOCKS.items()}, "eligible": spec.eligible, "scoreable": spec.scoreable,
                       "eligibility": spec.eligibility} for job, spec in job_specs.items()},
        "local_length": LOCAL_LEN, "max_absolute_index_exclusive": MAX_ABS_ROWS,
        "consumer": dict(R.CONSUMER_STRUCTURE), "period": PERIOD, "context_length": CONTEXT_LENGTH, "horizon": HORIZON,
        "availability_rule": {"truth_min_finite": TRUTH_MIN_FINITE, "scale_min_pairs": SCALE_MIN_PAIRS,
                              "note": "the 32-finite-future rule is this task's availability rule, not the scorer's truth threshold"},
        "menu": list(MENU_LABELS), "source_order": list(source_order), "random_order": list(rnd_order),
        "source_fixed_label": source_fixed_label,
        "budgets": {"unit_api": UNIT_MAX_API, "unit_tokens": UNIT_MAX_TOKENS, "unit_active_seconds": UNIT_MAX_ACTIVE_SECONDS,
                    "package_api": PKG_MAX_API, "package_tokens": PKG_MAX_TOKENS, "package_fits": PKG_MAX_FITS,
                    "package_wall_seconds": PKG_MAX_WALL_SECONDS, "global_fast_sessions": GLOBAL_FAST_SESSIONS,
                    "per_decision_cap": PER_DECISION_CAP, "planned_queries_per_arm_job": 4},
        "model": {"requested": R.REQUESTED_MODEL, "required_returned": R.REQUIRED_RETURNED_MODEL, "base_url": R.LOOPBACK_URL},
        "h0": {"canonical_runtime_bundle_sha": h0_sha, "run_permission_snapshot_sha": run_sha, "cap": 1.0,
               "capability_statement_is_instrument_not_learning": True},
        "material": MATERIAL,
        "frozen_utc": utc_now(),
    }


def run_course(*, course_id: str, live: bool, raw_backend_factory: Callable[[], Any], data_views: Mapping[str, Mapping[str, np.ndarray]] | None = None,
               data_receipt: Mapping[str, Any] | None = None, skip_llm: bool = False, machinery: Mapping[str, Any] | None = None,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    course_dir = WORK_ROOT / str(course_id)
    course_dir.mkdir(parents=True, exist_ok=True)
    state = CourseState(course_dir, live=live)
    machinery = machinery or R.load_machinery()
    state.machinery = dict(machinery)
    if data_views is None:
        data_views, data_receipt = load_job_views()
    R.atomic_write_json(course_dir / "data_receipt.json", data_receipt or {})
    job_specs = {job: build_job_spec(job, data_views[job]) for job in JOBS}

    h0 = R.load_h0_snapshot(machinery)
    any_spec = run_spec_for(job_specs["J1"], live=live)
    run_snapshot = R.configure_snapshot_for_spec(any_spec, h0, machinery)
    store = machinery["SnapshotStore"](course_dir / "store")
    controller = machinery["EditController"](store, surfaces=machinery["SurfaceRegistry"](), router=machinery["FaultRouter"]())
    store.materialize(run_snapshot)

    compressed = compress_source_material()
    ranking = source_gain_ranking(compressed)
    source_order = [row["label"] for row in ranking]
    rnd_order = random_order(SEED)
    source_fixed_label = source_order[0]
    config = course_config(job_specs, source_order=source_order, rnd_order=rnd_order, source_fixed_label=source_fixed_label,
                           h0_sha=h0.runtime_bundle_sha, run_sha=run_snapshot.runtime_bundle_sha)
    config["source_ranking"] = ranking
    config_path = course_dir / "course.json"
    if config_path.is_file():
        previous = R.read_json(config_path)
        for key in ("jobs", "menu", "source_order", "random_order", "source_fixed_label", "h0", "stations"):
            if previous.get(key) != config.get(key):
                raise RuntimeError("frozen course config differs on %s; use a new course id" % key)
        config = previous
    else:
        R.atomic_write_json(config_path, config)
    factory_for = make_sender_factory(state, raw_backend_factory)
    result: dict[str, Any] = {"package": PACKAGE, "course_id": course_id, "course_dir": str(course_dir), "status": "RUNNING",
                              "exit_code": 0, "jobs": {}, "config_frozen": True}
    result_path = course_dir / "result.json"

    def persist(status: str | None = None) -> None:
        if status:
            result["status"] = status
        result["budget"] = state.budget.to_state()
        result["tokens"] = state.tokens.to_state()
        result["orchestration"] = state.orchestration()
        R.atomic_write_json(result_path, json.loads(json.dumps(result, default=R._json_default)))

    contexts: dict[str, JobContext] = {}
    try:
        with install_course(state):
            # ---- Source guidance formation ------------------------------------
            j1 = JobContext(job=job_specs["J1"], values=data_views["J1"], state=state, machinery=machinery, live=live)
            contexts["J1"] = j1
            source_dir = course_dir / "source"
            source_dir.mkdir(parents=True, exist_ok=True)
            R.atomic_write_json(source_dir / "compressed_material.json", compressed)
            if skip_llm:
                source_boundary: dict[str, Any] = {"outcome": "SKIPPED", "treatment": "NO_UPDATE_TREATMENT"}
            else:
                source_boundary = run_skill_boundary(
                    j1, factory_for, arm_key="source/formation", parent=run_snapshot,
                    card=source_formation_card(compressed), store=store, controller=controller,
                    checkpoint=source_dir / "boundary.json",
                    exposure_contexts=exposure_contexts(j1, blocks=("C_A", "C_B", "E")))
            result["source_formation"] = {k: v for k, v in source_boundary.items() if k != "candidate_snapshot"}
            h_source = source_boundary.get("candidate_snapshot") if source_boundary.get("treatment") == "UPDATE_TREATMENT" else None
            result["source_formation"]["a5_j1_treatment"] = "SOURCE_DERIVED_H" if h_source is not None else "EMPTY_(H0)"
            persist()

            a5_h: Any = h_source if h_source is not None else run_snapshot
            a5_label = "source_derived" if h_source is not None else "h0_empty_source_treatment"
            for job_id in ("J1", "J2"):
                ctx = contexts.get(job_id) or JobContext(job=job_specs[job_id], values=data_views[job_id], state=state, machinery=machinery, live=live)
                contexts[job_id] = ctx
                job_out: dict[str, Any] = {"scoreable": ctx.job.scoreable, "eligible": ctx.job.eligible}
                result["jobs"][job_id] = job_out
                if not ctx.job.scoreable:
                    job_out["status"] = "NOT_SCOREABLE"
                    job_out["why"] = "a required block has no eligible origin; paid runs for this job are not started"
                    persist()
                    continue
                log("%s static + fixed + search" % job_id)
                job_out["static"] = run_static(ctx)
                job_out["source_fixed"] = run_fixed(ctx, arm="source_fixed", label=source_fixed_label)
                if job_id == "J1":
                    so_order, so_source = source_order, {"kind": "source_mean_gain", "ranking": ranking}
                else:
                    j1_search = result["jobs"]["J1"]["source_order"]
                    j1_static = {block: result["jobs"]["J1"]["static"]["blocks"][block]["mean_smase"] for block in ("C_A", "C_B")}
                    so_order, so_source = reorder_from_j1(source_order, j1_search, j1_static)
                    so_source["kind"] = "source_order_updated_by_own_j1_queries"
                job_out["source_order"] = run_search(ctx, arm="source_order", order=so_order, order_source=so_source)
                job_out["random"] = run_search(ctx, arm="random", order=rnd_order, order_source={"kind": "seeded_random", "seed": SEED})
                persist()
                if skip_llm:
                    job_out["llm_arms"] = "SKIPPED"
                    continue
                fast = FastRunner(ctx, factory_for)
                arms = {"F": (run_snapshot, "h0"), "A3": (run_snapshot, "h0"), "A5": (a5_h, a5_label)}
                unit_errors: list[BaseException] = []
                unit_out: dict[str, Any] = {}

                def one_unit(arm: str, snap: Any, label: str) -> None:
                    try:
                        log("%s %s start (H=%s)" % (job_id, arm, label))
                        unit_out[arm] = run_llm_unit(ctx, fast, factory_for, arm=arm, snapshot=snap, store=store,
                                                     controller=controller, h_label=label)
                        log("%s %s done" % (job_id, arm))
                    except UnitCeiling as exc:
                        unit_out[arm] = {"status": "UNIT_CEILING", "detail": str(exc)}
                        log("%s %s unit ceiling: %s" % (job_id, arm, exc))
                    except BaseException as exc:  # noqa: BLE001
                        unit_errors.append(exc)
                        log("%s %s error: %s: %s" % (job_id, arm, type(exc).__name__, exc))

                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(one_unit, arm, snap, label) for arm, (snap, label) in arms.items()]
                    for fut in as_completed(futures):
                        fut.result()
                job_out["llm_arms"] = {arm: {k: v for k, v in (unit_out.get(arm) or {}).items() if k not in ("train_features",)} for arm in arms}
                persist()
                if unit_errors:
                    raise unit_errors[0]
                # ---- J1 -> J2 integration for A5 (before E scoring) -------------
                if job_id == "J1":
                    a5_state = unit_out.get("A5") or {}
                    idir = course_dir / "integration"
                    idir.mkdir(parents=True, exist_ok=True)
                    if a5_state.get("status") == "COMPLETED":
                        adopted_sha = (a5_state.get("adoption") or {}).get("adopted_h_sha")
                        parent = a5_h
                        if adopted_sha and adopted_sha != a5_h.runtime_bundle_sha:
                            parent = machinery["compile_snapshot"](Path(store.root) / adopted_sha, verify_lock=False)
                        j2 = JobContext(job=job_specs["J2"], values=data_views["J2"], state=state, machinery=machinery, live=live)
                        contexts["J2"] = j2
                        card = integration_card(ctx, {**a5_state, "train_features": features_by_uid(
                            R.read_json(ctx.dir / "A5" / "q1" / "train_w.json")["sequences"] if (ctx.dir / "A5" / "q1" / "train_w.json").is_file() else [])})
                        integration = run_skill_boundary(
                            ctx, factory_for, arm_key="integration/a5_j1_to_j2", parent=parent, card=card, store=store,
                            controller=controller, checkpoint=idir / "boundary.json",
                            exposure_contexts=exposure_contexts(j2, blocks=("C_A", "C_B", "E")))
                        result["integration"] = {k: v for k, v in integration.items() if k != "candidate_snapshot"}
                        result["integration"]["parent_sha"] = parent.runtime_bundle_sha
                        if integration.get("treatment") == "UPDATE_TREATMENT":
                            a5_h, a5_label = integration["candidate_snapshot"], "integrated_from_j1"
                        else:
                            a5_h, a5_label = parent, "j1_adopted_h_no_integration"
                        result["integration"]["a5_j2_h"] = {"label": a5_label, "sha": a5_h.runtime_bundle_sha}
                    else:
                        result["integration"] = {"outcome": "A5_J1_INCOMPLETE", "treatment": "NO_UPDATE_TREATMENT"}
                        a5_label = a5_label + "_j1_incomplete"
                    persist()
            # ---- all E outputs frozen -> external scoring -------------------------
            result["E"] = score_all_E(contexts, result)
            result["main_table"] = main_table(result)
            persist("COMPLETED")
    except R.AccountFault as exc:
        result["detail"] = str(exc)
        result["exit_code"] = 2
        persist("ACCOUNT_OR_PERMISSION_FAULT")
    except R.PackageCeiling as exc:
        result["detail"] = str(exc)
        result["exit_code"] = 3
        persist("PACKAGE_CEILING")
    except Exception as exc:  # noqa: BLE001
        import traceback
        result["detail"] = "%s: %s" % (type(exc).__name__, R._sanitize(exc, 600))
        result["traceback"] = traceback.format_exc()[-4000:]
        result["exit_code"] = 1
        persist("RUNNER_FAULT")
    return result


# ---------------------------------------------------------------------------
# external E scoring and aggregation
# ---------------------------------------------------------------------------

def score_all_E(contexts: Mapping[str, JobContext], result: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for job_id, ctx in contexts.items():
        job_res = (result.get("jobs") or {}).get(job_id) or {}
        if job_res.get("status") == "NOT_SCOREABLE":
            out[job_id] = {"status": "NOT_SCOREABLE"}
            continue
        # freeze receipt: every branch's E files must exist before any truth is read
        pending = []
        branches: dict[str, Path] = {}
        sdir = ctx.dir / "static"
        branches["static"] = sdir
        for arm in ("source_fixed",):
            branches[arm] = ctx.dir / arm
        for arm in SEARCH_ARMS:
            for name in ("q2ckpt", "final"):
                branches["%s/%s" % (arm, name)] = ctx.dir / arm / "E"
        llm = job_res.get("llm_arms") if isinstance(job_res.get("llm_arms"), dict) else {}
        for arm in LLM_ARMS:
            for name in ("q2ckpt", "q3", "q4"):
                branches["%s/%s" % (arm, name)] = ctx.dir / arm / "E" / name
        scores: dict[str, Any] = {}
        static_e: dict[int, dict[str, Any]] = {}
        # static first (reference)
        for origin in ctx.job.origins("E"):
            path = sdir / ("static_predictions_E_%d.json" % origin)
            pop = ctx.score_frozen(path) if path.is_file() else R.unknown_population(list(UIDS), "no frozen static E file")
            static_e[origin] = pop
        ctx.static_ref["E"] = static_e
        for key, folder in branches.items():
            tag = key.split("/")[-1] if "/" in key else key
            per_origin: dict[str, Any] = {}
            for origin in ctx.job.origins("E"):
                path = Path(folder) / ("%s_predictions_E_%d.json" % (tag, origin))
                if not path.is_file():
                    per_origin[str(origin)] = R.unknown_population(list(UIDS), "no frozen E output")
                    pending.append("%s@%d" % (key, origin))
                    continue
                per_origin[str(origin)] = ctx.score_frozen(path)
            means = [pop.get("mean_smase") for pop in per_origin.values()]
            utilities = [v for pop in per_origin.values() for v in (pop.get("per_uid_utility_vs_static") or {}).values() if isinstance(v, (int, float))]
            scores[key] = {
                "mean_smase": fmean_or_unknown(means),
                "mean_utility_vs_static": fmean_or_unknown(utilities) if len(utilities) == len(UIDS) * len(ctx.job.origins("E")) else "UNKNOWN",
                "risk": R.harmed_stats(utilities), "origins": per_origin,
            }
        out[job_id] = {"status": "SCORED", "scores": scores, "missing_frozen_outputs": pending, "scored_utc": utc_now()}
        R.atomic_write_json(ctx.dir / "E_scores.json", out[job_id])
    return out


def _diff(a: Any, b: Any) -> Any:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) - float(b)
    return "UNKNOWN"


def main_table(result: Mapping[str, Any]) -> dict[str, Any]:
    table: dict[str, Any] = {"per_job": {}, "equal_weight_two_jobs": {}}
    keys: dict[str, str] = {}
    for job_id, job_res in (result.get("jobs") or {}).items():
        e = ((result.get("E") or {}).get(job_id) or {}).get("scores") or {}
        llm = job_res.get("llm_arms") if isinstance(job_res.get("llm_arms"), dict) else {}
        row: dict[str, Any] = {}
        static_mean = (e.get("static") or {}).get("mean_smase")

        def final_key(arm: str) -> str:
            adopted = ((llm.get(arm) or {}).get("adoption") or {}).get("adopted") or "q3"
            return "%s/%s" % (arm, adopted)

        arms = {
            "static": "static", "source_fixed": "source_fixed",
            "source_order": "source_order/final", "random": "random/final",
            "F": final_key("F"), "A3": final_key("A3"), "A5": final_key("A5"),
        }
        keys[job_id] = json.dumps(arms)
        for arm, key in arms.items():
            block = e.get(key) or {}
            row[arm] = {"E_key": key, "mean_smase": block.get("mean_smase"), "gain_vs_static": _diff(static_mean, block.get("mean_smase")),
                        "risk": block.get("risk")}
        for arm in LLM_ARMS:
            for name in ("q2ckpt", "q3", "q4"):
                block = e.get("%s/%s" % (arm, name)) or {}
                row["%s/%s" % (arm, name)] = {"mean_smase": block.get("mean_smase"), "gain_vs_static": _diff(static_mean, block.get("mean_smase")), "risk": block.get("risk")}
        for arm in SEARCH_ARMS:
            block = e.get("%s/q2ckpt" % arm) or {}
            row["%s/q2ckpt" % arm] = {"mean_smase": block.get("mean_smase"), "gain_vs_static": _diff(static_mean, block.get("mean_smase")), "risk": block.get("risk")}
        m = {arm: row[arm]["mean_smase"] for arm in arms}
        row["contrasts_smase_difference_lower_is_better_for_first"] = {
            "A5_minus_A3": _diff(m["A5"], m["A3"]), "A3_minus_static": _diff(m["A3"], m["static"]),
            "A5_minus_source_fixed": _diff(m["A5"], m["source_fixed"]), "A5_minus_source_order": _diff(m["A5"], m["source_order"]),
            "A5_minus_random": _diff(m["A5"], m["random"]), "A3_minus_F": _diff(m["A3"], m["F"]), "A5_minus_F": _diff(m["A5"], m["F"]),
            "A5_q2ckpt_minus_A3_final": _diff(row["A5/q2ckpt"]["mean_smase"], m["A3"]),
        }
        for arm in LLM_ARMS:
            row["%s_new_minus_old_H_on_E" % arm] = _diff(row["%s/q4" % arm]["mean_smase"], row["%s/q3" % arm]["mean_smase"])
        table["per_job"][job_id] = row
    jobs = [j for j in table["per_job"] if isinstance(table["per_job"][j].get("static", {}).get("mean_smase"), (int, float))]
    if jobs:
        agg: dict[str, Any] = {}
        for arm in ("static", "source_fixed", "source_order", "random", "F", "A3", "A5"):
            vals = [table["per_job"][j][arm]["mean_smase"] for j in jobs]
            gains = [table["per_job"][j][arm]["gain_vs_static"] for j in jobs]
            agg[arm] = {"mean_smase": fmean_or_unknown(vals), "gain_vs_static": fmean_or_unknown(gains)}
        agg["A5_minus_A3"] = fmean_or_unknown([table["per_job"][j]["contrasts_smase_difference_lower_is_better_for_first"]["A5_minus_A3"] for j in jobs])
        agg["A5_q2ckpt_minus_A3_final"] = fmean_or_unknown([table["per_job"][j]["contrasts_smase_difference_lower_is_better_for_first"]["A5_q2ckpt_minus_A3_final"] for j in jobs])
        table["equal_weight_two_jobs"] = agg
    table["note"] = "two jobs, six series, a few origins, one LLM course; not independent samples; no significance claimed"
    return table


__all__ = [name for name in dir() if not name.startswith("_")]
