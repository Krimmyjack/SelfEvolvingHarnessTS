"""DEV-TRAIN-1 Fast/Slow runner: per-train W → one shared Ridge → per-eval V.

Fast is the real ``TTHAMethod`` / ``TTHAFastAgent`` / ``TTHAAgentCore`` path.
Slow is the real ``dev_seq2_knowledge`` catalog / propose / apply path.
Fit, freeze, predict and score live in ``dev_train1_evaluator``.

Initial knowledge is ``methods/ttha/harness/h0`` (General / bootstrap;
``learned/`` empty).  Historical K0 frozen-program cards are not loaded.
"""
from __future__ import annotations

import json
import os
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np

from SelfEvolvingHarnessTS.contracts.method import (
    PreparationRequest,
    PreparationStatus,
)
from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.contracts.task import (
    MetricSpec,
    deployment_constraints_v1,
    forecast_task_context_v1,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features
from SelfEvolvingHarnessTS.operators.registry import (
    OPERATOR_METADATA,
    OPERATOR_NAMES,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import AgentCallBudgetExceeded
from SelfEvolvingHarnessTS.runtime.candidate_verification import verify_candidate
from evaluation.main_protocol_p4 import dev_seq2_knowledge as know
from evaluation.main_protocol_p4 import dev_train1_evaluator as numeric
from evaluation.main_protocol_p4 import per_sequence as ps
from evaluation.main_protocol_p4.dev_seq1_knowledge import FROZEN_PROGRAM_MARKER

ROOT = Path(__file__).resolve().parents[2]
H0_ROOT = ROOT / "methods" / "ttha" / "harness" / "h0"
RUNS_ROOT = ROOT / "_scratch" / "dev_train1" / "runs"

PACKAGE = "DEV-TRAIN-1"
DOMAIN = "devtrain1"
REQUESTED_MODEL = "cpa-grok-4.6"
LOOPBACK_URL = "http://127.0.0.1:8318/v1"
HTTPS_PLACEHOLDER = "https://api.agicto.cn/v1"

CONTEXT_LENGTH = int(numeric.CONTEXT_LENGTH)
HORIZON = int(numeric.HORIZON)
ANCHORS = tuple(range(312, 853, 60))
TRAINING_PREFIX_END = int(max(ANCHORS) + HORIZON)
PERIOD = 24
MAX_MODIFIED_FRACTION = 0.35
MATERIAL = 0.005

# SEQ-2/SEQ-3 source: u9/u10/u11 origins 1656/2136/2376.  Those units use
# span=[40, 80], not [0, 40].  u12/u13 are the next origin_grid entries
# after 2376 (2616, 2856).  1896 is skipped because SEQ numbering skips it.
FORMATION_UNITS = ((9, 1656), (10, 2136), (11, 2376))
EVAL_UNITS = ((12, 2616), (13, 2856))

ARMS = ("static", "fixed_onestep", "old_skill", "new_skill")
MAX_LLM = 2000
MAX_FITS = 500
MAX_WALL_SECONDS = 6 * 3600
CONCURRENCY = 4
TRANSPORT_ATTEMPTS = 3
PER_SEQUENCE_LLM_CAP = 24
SLOW_ATTEMPTS = 2
REQUIRED_RETURNED_MODEL = "grok-4.6-build"
PRIOR_PACKAGE = {
    "llm": 2,
    "fits": 43,
    "source": (
        "root live preflight 2 LLM (requested cpa-grok-4.6, returned "
        "grok-4.6-build) + 10 kernel synthetic fits + 8 first fake-contract "
        "synthetic fits + 12 root contract fits + 13 final contract fits. "
        "This runner does not zero those costs."
    ),
}
COURSE_SPAN = [40, 80]
COURSE_SLICE = "[40:80]"

CONSUMER_CLASS_ID = "pooled_ridge_a1"
CONSUMER_STRUCTURE = {
    **dict(numeric.CONSUMER_STRUCTURE),
    "training_windows": "each train series × frozen anchors [312, 852] step 60",
    "input_length_points": CONTEXT_LENGTH,
    "horizon_points": HORIZON,
    "alpha": 1,
    "intercept": "unpenalized",
    "pooled_across_series": True,
    "scoring": "sMASE on raw evaluation truth; training loss is squared error",
}

_SENSITIVE = (
    "api_key", "apikey", "authorization", "secret", "password", "token",
    "credential", "bearer",
)
_EXTRA_ACCOUNT = (
    "insufficient_user_quota", "用户额度不足", "预扣费额度失败",
)
_ABSOLUTE_INDEX_KEYS = (
    "region_start", "region_end", "start_index", "end_index",
    "offset", "index0", "index1",
)
_OUTCOME_KEYS = {
    "evaluation_truth", "delayed_outcome", "held_out_outcome",
    "raw_truth", "future_truth", "open_delayed",
}


class PackageCeiling(RuntimeError):
    """The package LLM / fit / wall ceiling would be crossed."""


class AccountFault(RuntimeError):
    """The request never reached a model.  Stops the line."""


class LiveDataForbidden(RuntimeError):
    """Raised only when a live data path is invoked without --live."""


def _sanitize(text: Any, limit: int = 300) -> str:
    raw = str(text or "")
    lowered = raw.lower()
    if any(fragment in lowered for fragment in _SENSITIVE):
        return "[redacted error detail]"
    return raw[:limit]


def classify_fault(detail: str) -> str:
    text = str(detail)
    lowered = text.lower()
    if any(marker in text or marker in lowered for marker in _EXTRA_ACCOUNT):
        return "ACCOUNT_OR_PERMISSION_FAULT"
    return know._fault_kind(detail)


def raise_if_account(detail: str, *, cause: BaseException | None = None) -> None:
    if classify_fault(detail) == "ACCOUNT_OR_PERMISSION_FAULT":
        exc = AccountFault(_sanitize(detail))
        if cause is not None:
            raise exc from cause
        raise exc


def _json_default(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("not jsonable: %r" % type(value))


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=_json_default),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def steps_to_json(steps: Sequence[tuple[str, Mapping[str, Any]]] | None) -> Any:
    if steps is None:
        return None
    return [{"op": str(op), "params": dict(params or {})} for op, params in steps]


def steps_from_json(value: Any) -> tuple[tuple[str, dict], ...] | None:
    if value is None:
        return None
    return tuple((str(item["op"]), dict(item.get("params") or {})) for item in value)


def decision_key(role: str, arm: str, position: int, uid: str) -> str:
    return "%s|%s|%03d|%s" % (role, arm, int(position), uid)


@dataclass
class RunSpec:
    train_uids: list[str]
    eval_uids: list[str]
    anchors: tuple[int, ...] = ANCHORS
    period: int = PERIOD
    formation_units: tuple[tuple[int, int], ...] = FORMATION_UNITS
    eval_units: tuple[tuple[int, int], ...] = EVAL_UNITS
    training_prefix_end: int = TRAINING_PREFIX_END
    concurrency: int = CONCURRENCY
    max_llm: int = MAX_LLM
    max_fits: int = MAX_FITS
    max_wall_seconds: int = MAX_WALL_SECONDS
    requested_model: str = REQUESTED_MODEL
    base_url: str = LOOPBACK_URL
    live_api: bool = False
    onestep_ops: tuple[str, ...] | None = None
    training_design: str = "whole_window"
    recombination: str | None = None
    maximum_modified_fraction: float = 0.35


def effective_modified_cap(spec: RunSpec) -> float:
    value = float(spec.maximum_modified_fraction)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("maximum_modified_fraction must be finite and in [0,1]")
    return value


def _legacy_cap_context(context: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(context)
    normalized.setdefault("maximum_modified_fraction", 0.35)
    return normalized


def validate_run_cap(spec: RunSpec, run_dir: Path) -> None:
    """Reject cross-permission continuation before any old receipt is written."""
    requested = effective_modified_cap(spec)
    for name in ("run_cap.json", "preflight.json", "run_contract.json", "boundary.json"):
        path = Path(run_dir) / name
        if path.is_file():
            saved = read_json(path)
            previous = float(saved.get("maximum_modified_fraction", 0.35))
            if previous != requested:
                raise RuntimeError("modified-fraction cap mismatch in %s; use a new run ID" % name)


@dataclass
class DataBundle:
    """Caller-supplied arrays.  This runner does not load the KDD cache."""

    values: dict[str, np.ndarray]
    train_uids: list[str]
    eval_uids: list[str]
    period: int = PERIOD

    def roster(self) -> list[dict[str, str]]:
        rows = [{"series_uid": uid, "role": "train"} for uid in self.train_uids]
        rows.extend({"series_uid": uid, "role": "eval"} for uid in self.eval_uids)
        return rows


def default_spec(*, train_uids: Sequence[str], eval_uids: Sequence[str]) -> RunSpec:
    return RunSpec(train_uids=list(train_uids), eval_uids=list(eval_uids))


def planned_roster_contract() -> dict[str, Any]:
    return {
        "data_identity": "EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING",
        "n_train": 20,
        "n_eval": 20,
        "anchors": list(ANCHORS),
        "training_prefix_end": TRAINING_PREFIX_END,
        "formation_units": [list(row) for row in FORMATION_UNITS],
        "eval_units": [list(row) for row in EVAL_UNITS],
        "period": PERIOD,
        "data_bytes_read": 0,
        "live_load": "not invoked in this pass",
    }


def numeric_config(spec: RunSpec) -> dict[str, Any]:
    return {"anchors": list(spec.anchors), "period": int(spec.period)}


@dataclass
class Budget:
    started: float = field(default_factory=time.time)
    llm_by_arm: dict[str, int] = field(default_factory=dict)
    fits_by_arm: dict[str, int] = field(default_factory=dict)
    max_llm: int = MAX_LLM
    max_fits: int = MAX_FITS
    max_wall_seconds: int = MAX_WALL_SECONDS
    path: Path | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    concurrency_used: int = CONCURRENCY
    concurrency_note: str = ""
    prior_llm: int = 0
    prior_fits: int = 0
    completed_fits_by_arm: dict[str, int] = field(default_factory=dict)
    paused_seconds: float = 0.0

    def llm(self) -> int:
        return int(sum(self.llm_by_arm.values()))

    def fits(self) -> int:
        return int(sum(self.fits_by_arm.values()))

    def elapsed(self) -> float:
        return time.time() - self.started - self.paused_seconds

    def require(self, *, llm: int = 0, fits: int = 0) -> None:
        if self.prior_llm + self.llm() + int(llm) > self.max_llm:
            raise PackageCeiling("LLM ceiling %d+%d/%d" % (self.llm(), llm, self.max_llm))
        if self.prior_fits + self.fits() + int(fits) > self.max_fits:
            raise PackageCeiling("fit ceiling %d+%d/%d" % (self.fits(), fits, self.max_fits))
        if self.elapsed() > self.max_wall_seconds:
            raise PackageCeiling("wall ceiling %.0f/%d" % (self.elapsed(), self.max_wall_seconds))

    def spend_llm(self, arm: str, n: int) -> None:
        if n:
            self.llm_by_arm[arm] = self.llm_by_arm.get(arm, 0) + int(n)
            self.persist()

    def spend_fits(self, arm: str, n: int) -> None:
        if n:
            self.fits_by_arm[arm] = self.fits_by_arm.get(arm, 0) + int(n)
            self.persist()

    def to_state(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "paused_seconds": self.paused_seconds,
            "llm_by_arm": dict(self.llm_by_arm),
            "fits_by_arm": dict(self.fits_by_arm),
            "max_llm": self.max_llm,
            "max_fits": self.max_fits,
            "max_wall_seconds": self.max_wall_seconds,
            "llm_total": self.llm(),
            "fits_total": self.fits(),
            "wall_seconds": round(self.elapsed(), 2),
            "concurrency_used": self.concurrency_used,
            "concurrency_note": self.concurrency_note,
            "this_run_llm": self.llm(),
            "this_run_fits": self.fits(),
            "prior_package": {"llm": self.prior_llm, "fits": self.prior_fits},
            "package_cumulative_llm": self.llm() + self.prior_llm,
            "package_cumulative_fits": self.fits() + self.prior_fits,
            "completed_fits_by_arm": dict(self.completed_fits_by_arm),
            "fit_accounting": "reserved attempts; unmatched completions are uncertain, not certified finished fits",
            "prior_not_zeroed": True,
        }

    def persist(self) -> None:
        if self.path is None:
            return
        atomic_write_json(self.path, self.to_state())

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "Budget":
        path = Path(path)
        budget = cls(path=path, **kwargs)
        if path.is_file():
            state = read_json(path)
            budget.started = float(state.get("started") or budget.started)
            budget.paused_seconds = float(state.get("paused_seconds") or 0.0)
            prior = state.get("prior_package") or {}
            budget.prior_llm = int(prior.get("llm") or 0)
            budget.prior_fits = int(prior.get("fits") or 0)
            budget.completed_fits_by_arm = dict(state.get("completed_fits_by_arm") or {})
            budget.llm_by_arm = {
                str(k): int(v) for k, v in (state.get("llm_by_arm") or {}).items()
            }
            budget.fits_by_arm = {
                str(k): int(v) for k, v in (state.get("fits_by_arm") or {}).items()
            }
        return budget


def load_machinery() -> dict[str, Any]:
    from evaluation.main_protocol_p4 import run_source_line as v1runner
    return v1runner._machinery()


def allow_exact_loopback_http() -> None:
    """Process-local: allow only http://127.0.0.1:8318/v1.  File on disk is not edited."""
    from SelfEvolvingHarnessTS.runtime import agent_backend as backend

    if getattr(backend._validate_base_url, "_devtrain1_loopback_patch", False):
        return
    original = backend._validate_base_url

    def _patched(base_url: str) -> None:
        if base_url == LOOPBACK_URL:
            parsed = urlsplit(base_url)
            if (
                parsed.scheme == "http"
                and parsed.hostname == "127.0.0.1"
                and parsed.path == "/v1"
                and parsed.query == ""
                and parsed.fragment == ""
            ):
                return
            raise ValueError("loopback URL is not the exact authorised origin")
        original(base_url)

    _patched._devtrain1_loopback_patch = True  # type: ignore[attr-defined]
    backend._validate_base_url = _patched


class BoundedTransportRetry:
    """At most TRANSPORT_ATTEMPTS tries on transient transport.  Account faults stop."""

    def __init__(self, delegate: Any, *, attempts: int = TRANSPORT_ATTEMPTS) -> None:
        self.delegate = delegate
        self.attempts = int(attempts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def complete(self, request: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                return self.delegate.complete(request)
            except AccountFault:
                raise
            except Exception as exc:  # noqa: BLE001
                detail = "%s: %s" % (type(exc).__name__, exc)
                raise_if_account(detail, cause=exc)
                if classify_fault(detail) != "TRANSPORT_TRANSIENT_FAULT":
                    raise
                last = exc
        raise last  # type: ignore[misc]


class MeteredBudgetBackend:
    """Reserve and persist one LLM call before every complete() attempt.

    Failures still count.  The per-sequence cap is a safety limit: exceeding
    it is UNKNOWN, not identity.  Package ceiling is checked under the budget
    lock before the attempt.
    """

    def __init__(
        self,
        inner: Any,
        *,
        budget: Budget,
        arm: str,
        per_sequence_cap: int | None = None,
    ) -> None:
        self.inner = inner
        self.budget = budget
        self.arm = str(arm)
        self.per_sequence_cap = per_sequence_cap
        self.session_calls = 0
        self.calls = 0
        self.returned_models: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def complete(self, request: Any) -> Any:
        with self.budget.lock:
            if self.per_sequence_cap is not None and self.session_calls >= self.per_sequence_cap:
                raise AgentCallBudgetExceeded(
                    "per-sequence safety cap %d reached; UNKNOWN, not identity"
                    % self.per_sequence_cap
                )
            self.budget.require(llm=1)
            self.budget.spend_llm(self.arm, 1)
        self.session_calls += 1
        self.calls += 1
        try:
            response = self.inner.complete(request)
        except Exception as exc:
            detail = "%s: %s" % (type(exc).__name__, exc)
            raise_if_account(detail, cause=exc)
            models = getattr(self.inner, "returned_models", None)
            if models:
                self.returned_models |= set(models)
            raise
        models = getattr(self.inner, "returned_models", None)
        if models:
            self.returned_models |= set(models)
        meta = getattr(response, "provider_metadata", None) or {}
        if isinstance(meta, Mapping) and meta.get("returned_model"):
            self.returned_models.add(str(meta["returned_model"]))
        return response


class UnavailableToolGateway:
    """Slow gets no public tools, and no invented data standing in for them.

    The batch boundary is not a single series, so there is no legal array to
    bind a ``LocalPublicToolGateway`` to.  Binding one to ``np.zeros(8)``
    would answer Slow's tool calls with fabricated flat data that looks like
    a measurement.  Offering no schema is the honest state: the boundary
    evidence Slow may use is the card, and a tool call is refused in words.
    """

    context_sha = "devtrain1-slow-no-public-tools"

    def schemas_for(self, *, role: Any = None, stage: Any = None) -> tuple:
        del role, stage
        return ()

    def call(self, tool_name: str, arguments: Any) -> Any:
        del arguments
        raise RuntimeError(
            "public tools are unavailable at the DEV-TRAIN-1 batch boundary "
            "(no single bound series); %r was not answered with substitute "
            "data" % str(tool_name)
        )


def bind_backend_factory(
    factory: Callable[[], Any], budget: Budget, arm: str,
) -> Callable[[], Any]:
    """Meter innermost, retry outermost: one charge per real HTTP attempt.

    ``BoundedTransportRetry`` calls ``complete`` again on a transient
    transport fault, so the meter has to sit *under* it.  With the meter on
    the outside a fail/fail/success sequence spends three HTTP attempts and
    reports one.  ``factory()`` must therefore hand back the sending backend
    itself, never a pre-wrapped retry.
    """
    def bound() -> Any:
        sender = factory()
        if isinstance(sender, BoundedTransportRetry):
            raise RuntimeError(
                "backend_factory returned a retry wrapper; the meter must be "
                "inside the retry so every HTTP attempt is charged once"
            )
        metered = MeteredBudgetBackend(sender, budget=budget, arm=arm)
        return BoundedTransportRetry(metered, attempts=TRANSPORT_ATTEMPTS)

    return bound


def build_live_backend(*, maximum_calls: int) -> Any:
    """The sending backend only.  Retry/metering are added by the caller.

    ``max_retries=0`` on the injected client: the OpenAI SDK otherwise retries
    internally, which would spend HTTP attempts this package never sees and
    never charges.  Retry is the package's own bounded metered wrapper.
    Key stays in env and is never written to a receipt.
    """
    from SelfEvolvingHarnessTS.runtime.agent_backend import (
        AgictoChatCompletionsBackend,
    )

    allow_exact_loopback_http()
    key = next(
        (
            os.environ.get(name, "").strip()
            for name in ("CPA_API_KEY", "AGICTO_API_KEY", "OPENAI_API_KEY")
            if os.environ.get(name, "").strip()
        ),
        "",
    )
    if not key:
        raise RuntimeError("an existing API key env var is required for live transport")
    import importlib

    openai = importlib.import_module("openai")
    client = openai.OpenAI(
        api_key=key, base_url=LOOPBACK_URL, timeout=240, max_retries=0,
    )
    del maximum_calls
    return AgictoChatCompletionsBackend(
        client=client, base_url=LOOPBACK_URL, timeout_seconds=240,
    )


def load_h0_snapshot(machinery: Mapping[str, Any]) -> Any:
    snapshot = machinery["compile_snapshot"](H0_ROOT, verify_lock=False)
    frozen = []
    for skill in getattr(snapshot, "skills", ()) or ():
        body = str(getattr(skill, "body", "") or "")
        if FROZEN_PROGRAM_MARKER in body:
            frozen.append(str(skill.skill_id))
    if frozen:
        raise RuntimeError(
            "h0 snapshot carries Frozen program steps on %s; DEV-TRAIN-1 "
            "refuses historical program-card supply" % frozen
        )
    learned = list((H0_ROOT / "skills" / "learned").glob("*.json"))
    if learned:
        raise RuntimeError("h0 learned/ is not empty: %s" % [p.name for p in learned])
    return snapshot


def configure_snapshot_for_spec(spec: RunSpec, snapshot: Any,
                                machinery: Mapping[str, Any]) -> Any:
    """Recompile the common run permission; never edit canonical knowledge.

    This is protocol configuration, not a Slow update. Existing authored Skill
    guards and every other verification field remain unchanged.
    """
    cap = effective_modified_cap(spec)
    if float(snapshot.verification.get("max_modified_fraction", 0.35)) == cap:
        return snapshot
    with tempfile.TemporaryDirectory(prefix="train-run-permission-") as directory:
        store = machinery["SnapshotStore"](Path(directory) / "store")
        parent = store.materialize(snapshot)
        fork = store.fork(parent, "run-permission")
        path = fork / "verification.json"
        verification = read_json(path)
        verification["max_modified_fraction"] = cap
        atomic_write_json(path, verification)
        configured = machinery["compile_snapshot"](fork, verify_lock=False)
    return configured


class RoleInjectingCore:
    """Merge role / consumer / frame into Fast/Slow public_input.

    TaskContext.to_dict() cannot carry Ridge geometry.  Extra keys go through
    the existing ``run_stage(..., public_input=)`` hook.  Hashed core is unchanged.
    """

    def __init__(self, inner: Any, extra: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_extra", dict(extra))
        object.__setattr__(self, "last_public_inputs", [])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_inner", "_extra", "last_public_inputs"):
            object.__setattr__(self, name, value)
            return
        setattr(self._inner, name, value)

    def run_stage(self, *args: Any, public_input: Mapping[str, Any], **kwargs: Any) -> Any:
        merged = dict(public_input)
        for key, value in self._extra.items():
            merged.setdefault(key, value)
        self.last_public_inputs.append(merged)
        return self._inner.run_stage(*args, public_input=merged, **kwargs)


def task_context_for(spec: RunSpec) -> Any:
    task = forecast_task_spec_v1(
        horizon=HORIZON,
        downstream_model_class=CONSUMER_CLASS_ID,
        metric=MetricSpec("sMASE", "lower_is_better"),
    )
    constraints = deployment_constraints_v1(
        maximum_candidates=3,
        maximum_modified_fraction=effective_modified_cap(spec),
        fixed_downstream_model_id="pooled-ridge-a1",
        constraint_id="forecast-pooled-ridge-a1-v1",
    )
    return forecast_task_context_v1(task_spec=task, deployment_constraints=constraints)


def training_window_geometry(spec: RunSpec | None) -> dict[str, Any]:
    """The exact anchors and 240-point intervals this UID's one W will run on.

    All of it comes from the same legal prefix Fast is already given.  One W
    per training UID is the approved shape, so the honest statement is that
    whatever fractions the program carries are replayed on **each** of these
    windows -- a fraction is a position inside a window, never a position in
    the prefix.
    """
    if spec is None:
        return {"status": "not_supplied_to_this_role"}
    prefix_end = int(spec.training_prefix_end)
    anchors = [
        int(anchor) for anchor in spec.anchors
        if int(anchor) + HORIZON <= prefix_end
    ]
    windows = [
        {
            "anchor": int(anchor),
            "window_start_index": int(anchor) - CONTEXT_LENGTH,
            "window_end_index_exclusive": int(anchor) + HORIZON,
            "x_points": CONTEXT_LENGTH,
            "y_points": HORIZON,
        }
        for anchor in anchors
    ]
    return {
        "prefix_interval": [0, prefix_end],
        "anchors": anchors,
        "n_windows": len(windows),
        "windows": windows,
        "one_workflow_per_training_uid": True,
        "the_same_program_runs_on_every_window_above": True,
        "a_fraction_is_repeated_on_each_local_window": (
            "region_start_fraction / region_end_fraction are read against the "
            "240 points of each window in turn, so one pair of fractions "
            "selects a different absolute interval in every window.  They do "
            "not name a location in the prefix."
        ),
        "all_indices_are_inside_the_legal_prefix": True,
    }


def observation_and_execution_frames(
    role: str, *, observed_length: int, spec: RunSpec | None = None,
) -> dict[str, Any]:
    if role == "train_workflow":
        return {
            "decision_role": role,
            "observation_frame": {
                "kind": "fixed_training_prefix",
                "values_length": int(observed_length),
                "absolute_end_index": int(observed_length),
                "not_the_240_point_execution_window": True,
                "feature_window": (
                    "the whole prefix Fast is given; every public tool result "
                    "is computed over this PREFIX, not over one 240-point "
                    "training window"
                ),
                "public_tool_results_are_prefix_scoped": True,
            },
            "execution_frame": {
                "kind": "each_training_window",
                "window_points": CONTEXT_LENGTH + HORIZON,
                "layout": "[anchor-192, anchor+48) = 192 X + 48 training y",
                "y_is_legally_observed_training_target": True,
                "region_fractions_relative_to": "this 240-point window, not the prefix",
                "prefix_absolute_indices_have_no_legal_map": True,
                "window_geometry": training_window_geometry(spec),
            },
        }
    if role == "predict_input":
        return {
            "decision_role": role,
            "observation_frame": {
                "kind": "eval_serve_context",
                "values_length": int(observed_length),
                "window_points": CONTEXT_LENGTH,
                "layout": "[origin-192, origin)",
                "feature_window": "the 192-point serve context Fast is given",
                "not_a_summary_of_a_longer_prefix": True,
            },
            "execution_frame": {
                "kind": "eval_serve_context",
                "window_points": CONTEXT_LENGTH,
                "no_current_downstream_feedback": True,
                "region_fractions_relative_to": "this 192-point window",
            },
        }
    return {
        "decision_role": role,
        "observation_frame": {"kind": "batch_boundary_card"},
        "execution_frame": {"kind": "skill_text_only"},
    }


def public_extra(
    role: str, *, observed_length: int, spec: RunSpec | None = None,
) -> dict[str, Any]:
    frames = observation_and_execution_frames(
        role, observed_length=observed_length, spec=spec,
    )
    return {
        **frames,
        "consumer_structure": dict(CONSUMER_STRUCTURE),
        "region_geometry_rule": (
            "region_start_fraction and region_end_fraction, if emitted, are the "
            "existing DSL geometry and are interpreted relative to the execution "
            "window in execution_frame, not relative to the observation prefix. "
            "Do not emit prefix-absolute indices; there is no legal map from a "
            "prefix index onto a 240-point training window.  Intrinsic and "
            "global operators need no region map."
        ),
        "identity_is_not_a_global_zero": (
            "an identity prediction input is still scored by the shared fitted "
            "model; it is not a hardcoded gain of 0"
        ),
    }


def make_request(
    *,
    uid: str,
    observed: np.ndarray,
    spec: RunSpec,
    task_ctx: Any,
) -> PreparationRequest:
    observed = np.asarray(observed, dtype=np.float64).ravel()
    pattern = observed_pattern_spec(str(uid), observed, int(spec.period))
    return PreparationRequest(
        str(uid),
        observed,
        task_ctx.task_spec,
        pattern,
        task_context=task_ctx,
    )


def region_geometry_report(
    steps: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    observation_length: int,
    execution_length: int,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    fraction_keys: list[str] = []
    for op, params in steps:
        params = dict(params or {})
        for key, value in params.items():
            name = str(key)
            if name.endswith("_fraction"):
                fraction_keys.append("%s.%s" % (op, name))
                continue
            if name in _ABSOLUTE_INDEX_KEYS:
                issues.append({
                    "op": op,
                    "param": name,
                    "value": value,
                    "why": (
                        "absolute/prefix index is not a legal map onto an "
                        "execution window of %d points" % execution_length
                    ),
                })
        mode = str(OPERATOR_METADATA.get(str(op), {}).get("targeting_mode") or "")
        if mode == "external_region" and not any(
            key in params for key in ("region_start_fraction", "region_end_fraction")
        ):
            issues.append({
                "op": op,
                "param": "external_region",
                "value": None,
                "why": (
                    "external_region operator without fraction params; "
                    "not rewritten to whole_window"
                ),
            })
    if issues:
        return {
            "status": "UNMAPPED",
            "issues": issues,
            "observation_length": int(observation_length),
            "execution_length": int(execution_length),
            "did_not_replace_with_whole_window": True,
            "action": "not_executed",
        }
    return {
        "status": "DSL_FRACTION_OR_INTRINSIC",
        "observation_length": int(observation_length),
        "execution_length": int(execution_length),
        "fraction_keys": fraction_keys,
        "rule": (
            "fractions are the existing DSL geometry and are relative to the "
            "execution array; intrinsic/global operators need no map"
        ),
        "did_not_strip_params": True,
    }


def observed_pattern_spec(uid: str, observed: np.ndarray, period: int) -> dict[str, float]:
    """Build observed_pattern_spec from the window Fast actually sees.

    ``window_context`` / ``CohortHistoryPublicToolGateway`` need two complete
    192-point windows of *this* series.  Predict Fast sees 192 points; a
    short train prefix may also be shorter than 384.  Do not invent an
    earlier window, do not read past the observation, and do not borrow
    another series.  When this series' own observed prefix is long enough,
    recent/change features are added from that prefix only.
    """
    import math

    observed = np.asarray(observed, dtype=np.float64).ravel()
    features = extract_public_features(observed, task_kind="forecast")
    pattern: dict[str, float] = {}
    for key, value in dict(features).items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            pattern[str(key)] = float(value)
    pattern["bound_period"] = float(period)
    pattern["observation_length"] = float(observed.size)
    if observed.size >= 2 * CONTEXT_LENGTH:
        from SelfEvolvingHarnessTS.methods.ttha import signed_radius

        extra = signed_radius.window_context(
            {str(uid): observed}, int(observed.size), int(period),
        )
        for key, value in extra.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                pattern[str(key)] = float(value)
    return pattern


def _status_value(result: Any) -> str:
    if result is None:
        return ""
    status = getattr(result, "status", None)
    if status is None:
        return ""
    if isinstance(status, PreparationStatus):
        return str(status.value)
    return str(getattr(status, "value", status) or "").lower()


def _receipt_error(result: Any) -> str:
    receipt = getattr(result, "receipt", None)
    return str(getattr(receipt, "error", "") or "")


def _steps_map(trace: Any) -> dict[str, tuple]:
    if trace is None:
        return {}
    raw = dict(getattr(trace, "candidate_program_steps", None) or {})
    return {
        str(key): tuple((str(op), dict(params)) for op, params in value)
        for key, value in raw.items()
    }


def interpret_prepare(
    *, result: Any, trace: Any, thrown: BaseException | None = None,
) -> dict[str, Any]:
    if result is not None:
        error = _receipt_error(result)
        if error:
            raise_if_account(error)
    status_value = _status_value(result)
    steps_map = _steps_map(trace)
    chosen = str(
        (getattr(trace, "chosen_candidate_id", None) if trace is not None else None) or ""
    )

    def programs() -> dict[str, str]:
        return {
            key: ps.program_label_with_params(ps.normalise_steps(value))
            for key, value in steps_map.items()
        }

    def missing(status: str, why: str) -> dict[str, Any]:
        return {
            "deploy_status": status,
            "valid_mechanism_decision": False,
            "assignment_policy": "unknown",
            "deployed_steps": None,
            "chosen_candidate_id": chosen or None,
            "candidate_programs": programs(),
            "why_unknown": why,
        }

    if thrown is not None and result is None:
        name = type(thrown).__name__
        detail = "%s: %s" % (name, thrown)
        if "AgentCallBudgetExceeded" in name or "tool round limit" in str(thrown).lower():
            return missing(
                "BACKEND_SAFETY_LIMIT",
                "a backend/core safety limit was hit (%s); this is UNKNOWN, "
                "not identity" % _sanitize(detail, 200),
            )
        return missing(
            "FAULT_NO_DECISION",
            "the session raised %s before a consumed PreparationResult; "
            "this is not a zero" % name,
        )
    if status_value == PreparationStatus.FAILED.value:
        return missing(
            "PREPARE_FAILED",
            "PreparationResult.status is FAILED; a leftover trace is not a valid decision",
        )
    if result is None or trace is None:
        return missing(
            "FAULT_NO_DECISION",
            "the session faulted before a decision was reached; this is not a zero",
        )
    if (
        status_value not in (PreparationStatus.PREPARED.value, PreparationStatus.ABSTAINED.value)
        or not getattr(getattr(result, "receipt", None), "ok", False)
    ):
        return missing(
            "INVALID_PREPARATION_RESULT",
            "missing/invalid success status or unsuccessful execution receipt",
        )
    if not chosen:
        return missing(
            "EMPTY_SELECTION_NO_VALID_DECISION",
            "empty chosen_candidate_id is not a deliberate identity choice",
        )
    if chosen == "identity":
        return {
            "deploy_status": "IDENTITY_SELECTED",
            "valid_mechanism_decision": True,
            "assignment_policy": "explicit_identity",
            "deployed_steps": (),
            "chosen_candidate_id": "identity",
            "candidate_programs": programs(),
            "why_unknown": None,
        }
    if chosen not in steps_map:
        return missing(
            "UNKNOWN_CANDIDATE_NO_VALID_DECISION",
            "chosen_candidate_id %r is not in the trace candidate map" % chosen,
        )
    return {
        "deploy_status": "CANDIDATE_CHOSEN",
        "valid_mechanism_decision": True,
        "assignment_policy": "typed_steps",
        "deployed_steps": steps_map[chosen],
        "chosen_candidate_id": chosen,
        "candidate_programs": programs(),
        "why_unknown": None,
    }


def _core_base_url(spec: RunSpec) -> str:
    if spec.live_api:
        allow_exact_loopback_http()
        return spec.base_url
    return HTTPS_PLACEHOLDER


def forbidden_outcome_paths(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key)
            here = "%s.%s" % (path, name) if path else name
            if name.lower() in _OUTCOME_KEYS:
                found.append(here)
            found.extend(forbidden_outcome_paths(nested, here))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            found.extend(forbidden_outcome_paths(nested, "%s[%d]" % (path, index)))
    return found


def run_fast_decision(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    snapshot: Any,
    machinery: Mapping[str, Any],
    backend_factory: Callable[[], Any],
    uid: str,
    role: str,
    arm: str,
    position: int,
    origin: int | None,
    task_ctx: Any,
) -> dict[str, Any]:
    started = time.time()
    uid = str(uid)
    raw = np.asarray(bundle.values[uid], dtype=np.float64)
    if role == "train_workflow":
        cutoff = int(spec.training_prefix_end)
        observed = raw[:cutoff]
        execution_length = CONTEXT_LENGTH + HORIZON
        extra = public_extra(role, observed_length=int(observed.size), spec=spec)
    elif role == "predict_input":
        origin_i = int(origin)
        start = origin_i - CONTEXT_LENGTH
        if start < 0:
            raise ValueError("origin %d is shorter than 192" % origin_i)
        observed = raw[start:origin_i]
        execution_length = CONTEXT_LENGTH
        extra = public_extra(role, observed_length=int(observed.size))
    else:
        raise ValueError("unknown Fast role %r" % role)

    record: dict[str, Any] = {
        "key": decision_key(role, arm, position, uid),
        "role": role,
        "arm": arm,
        "position": int(position),
        "series_uid": uid,
        "origin": None if origin is None else int(origin),
        "observed_length": int(observed.size),
        "execution_length": int(execution_length),
        "faults": [],
        "valid_mechanism_decision": False,
    }
    request = make_request(uid=uid, observed=observed, spec=spec, task_ctx=task_ctx)
    record["bound_series_uid"] = str(request.series_uid)
    record["bound_values_length"] = int(np.asarray(request.values).size)
    record["public_features"] = dict(
        machinery["extract_public_features"](observed, task_kind="forecast")
    )
    record["feature_window"] = extra["observation_frame"]

    inner = backend_factory()
    core = RoleInjectingCore(
        machinery["TTHAAgentCore"](
            inner,
            machinery["LocalPublicToolGateway"](observed, task_kind="forecast"),
            model=spec.requested_model,
            base_url=_core_base_url(spec),
        ),
        extra,
    )
    method = machinery["TTHAMethod"](
        machinery["TTHAFastAgent"](core), snapshot, (),
    )
    result = None
    trace = None
    thrown: BaseException | None = None
    try:
        method.bind_round_data(observed, task_kind="forecast")
        result = method.prepare(request, runtime_prior_slot=False, pool_mode="actionable")
        trace = method.last_trace
        error = _receipt_error(result)
        if error:
            raise_if_account(error)
    except AccountFault:
        raise
    except Exception as exc:  # noqa: BLE001
        detail = "%s: %s" % (type(exc).__name__, exc)
        raise_if_account(detail, cause=exc)
        record["faults"].append({"kind": type(exc).__name__, "why": _sanitize(detail)})
        thrown = exc

    record["llm_requests_sent"] = int(getattr(inner, "calls", 0) or 0)
    record["returned_models"] = sorted(getattr(inner, "returned_models", set()) or set())
    record["requested_model"] = spec.requested_model
    record["required_returned_model"] = REQUIRED_RETURNED_MODEL
    if record["returned_models"] and record["returned_models"] != [REQUIRED_RETURNED_MODEL]:
        record["model_alias_vs_returned"] = {
            "requested": spec.requested_model,
            "required_returned": REQUIRED_RETURNED_MODEL,
            "returned": record["returned_models"],
            "frozen_correspondence": (
                "requested cpa-grok-4.6 corresponds to returned grok-4.6-build; "
                "literal equality is not required; arbitrary models are not allowed"
            ),
        }
    interpreted = interpret_prepare(result=result, trace=trace, thrown=thrown)
    for key in (
        "deploy_status", "valid_mechanism_decision", "assignment_policy",
        "chosen_candidate_id", "candidate_programs", "why_unknown",
    ):
        if key in interpreted:
            record[key] = interpreted[key]
    steps = interpreted.get("deployed_steps")
    if not interpreted.get("valid_mechanism_decision"):
        record["deployed_program"] = "UNKNOWN"
        record["typed_steps"] = steps_to_json(steps) if steps else None
    elif not steps:
        record["deployed_program"] = "identity"
        record["typed_steps"] = None
        record["region_geometry"] = {"status": "IDENTITY"}
    else:
        record["deployed_program"] = ps.program_label_with_params(ps.normalise_steps(steps))
        record["typed_steps"] = steps_to_json(steps)
        geo = region_geometry_report(
            steps,
            observation_length=int(observed.size),
            execution_length=execution_length,
        )
        record["region_geometry"] = geo
        if geo["status"] == "UNMAPPED":
            record["valid_mechanism_decision"] = False
            record["assignment_policy"] = "unknown"
            record["deploy_status"] = "REGION_GEOMETRY_UNMAPPED"
            record["why_unknown"] = (
                "chosen program carries prefix/absolute region indices that "
                "have no legal map onto the execution window; not rewritten "
                "to whole_window and not executed"
            )
    record["knowledge_version"] = getattr(snapshot, "runtime_bundle_sha", None)
    record["wall_seconds"] = round(time.time() - started, 2)
    record["public_input_keys_injected"] = sorted(extra)
    leaks = []
    for payload in getattr(core, "last_public_inputs", []):
        leaks.extend(forbidden_outcome_paths(payload))
    record["forbidden_outcome_paths_in_fast_public_input"] = leaks
    return record


def assignment_value(record: Mapping[str, Any]) -> Any:
    """Evaluator contract: None with the uid present is identity; missing is an error."""
    if not record.get("valid_mechanism_decision"):
        return "MISSING"
    policy = record.get("assignment_policy")
    if policy == "explicit_identity":
        return None
    if policy == "typed_steps":
        steps = steps_from_json(record.get("typed_steps"))
        if not steps:
            return None
        return steps
    return "MISSING"


def complete_assignment(
    uids: Sequence[str], records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    out: dict[str, Any] = {}
    missing: list[str] = []
    for uid in uids:
        record = records.get(str(uid))
        if record is None:
            missing.append(str(uid))
            continue
        value = assignment_value(record)
        if value == "MISSING":
            missing.append(str(uid))
            continue
        out[str(uid)] = value
    if missing:
        return None, missing
    return out, []


RETRYABLE_FAST_STATUSES = frozenset({
    "FAULT_NO_DECISION",
    "PREPARE_FAILED",
    "BACKEND_SAFETY_LIMIT",
    "INVALID_PREPARATION_RESULT",
})


def checkpoint_is_final(row: Mapping[str, Any] | None) -> bool:
    """Successful or completed-invalid rows are not re-asked.  Faults are retried."""
    if not row:
        return False
    status = str(row.get("deploy_status") or "")
    if status in RETRYABLE_FAST_STATUSES or status == "":
        return False
    return True


def run_batch_fast(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    snapshot: Any,
    machinery: Mapping[str, Any],
    backend_factory: Callable[[], Any],
    uids: Sequence[str],
    role: str,
    arm: str,
    position: int,
    origin: int | None,
    task_ctx: Any,
    budget: Budget,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    saved: dict[str, Any] = {}
    if checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        if float(checkpoint.get("maximum_modified_fraction", 0.35)) != effective_modified_cap(spec):
            raise RuntimeError("Fast checkpoint cap mismatch; completed state is not overwritten")
        saved = {
            str(row["series_uid"]): row
            for row in (checkpoint.get("sequences") or [])
        }
    rows_by_uid: dict[str, dict[str, Any]] = dict(saved)
    if arm == "old_skill_formation" and (checkpoint_path.parent.parent / "boundary.json").is_file():
        # The existing Slow patch used these exact facts, including UNKNOWN.
        # Recovery must not silently improve its already-consumed evidence.
        if set(rows_by_uid) != set(uids):
            raise RuntimeError("completed boundary has incomplete formation checkpoint")
        return [rows_by_uid[str(uid)] for uid in uids]
    pending = [
        str(uid) for uid in uids
        if not checkpoint_is_final(rows_by_uid.get(str(uid)))
    ]
    lock = budget.lock
    account: list[AccountFault] = []
    metered_factory = bind_backend_factory(backend_factory, budget, arm)

    def one(uid: str) -> tuple[str, dict[str, Any] | None]:
        if account:
            return uid, None
        row = run_fast_decision(
            spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
            backend_factory=metered_factory, uid=uid, role=role, arm=arm,
            position=position, origin=origin, task_ctx=task_ctx,
        )
        with lock:
            rows_by_uid[uid] = row
            atomic_write_json(checkpoint_path, {
                "maximum_modified_fraction": effective_modified_cap(spec),
                "role": role, "arm": arm, "position": position, "origin": origin,
                "sequences": [rows_by_uid[item] for item in uids if item in rows_by_uid],
            })
        return uid, row

    workers = max(1, int(spec.concurrency))
    budget.concurrency_used = workers
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(one, uid): uid for uid in pending}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except AccountFault as exc:
                    account.append(exc)
                    for other in futures:
                        other.cancel()
                    break
                except PackageCeiling:
                    raise
    if account:
        raise account[0]
    ordered = []
    for uid in uids:
        row = rows_by_uid.get(str(uid))
        if row is None:
            row = {
                "key": decision_key(role, arm, position, str(uid)),
                "series_uid": str(uid),
                "valid_mechanism_decision": False,
                "deploy_status": "FAULT_NO_DECISION",
                "why_unknown": "no checkpoint row after the batch",
                "typed_steps": None,
            }
        ordered.append(row)
    atomic_write_json(checkpoint_path, {
        "maximum_modified_fraction": effective_modified_cap(spec),
        "role": role, "arm": arm, "position": position, "origin": origin,
        "sequences": ordered,
        "order_is_roster_not_completion": True,
    })
    return ordered


def window_legality_ok(moved_points: int, window_points: int,
                       maximum_modified_fraction: float = 0.35) -> bool:
    if window_points <= 0:
        return False
    return (float(moved_points) / float(window_points)) <= maximum_modified_fraction + 1e-12


def assert_train_windows_legal(
    spec: RunSpec,
    bundle: DataBundle,
    assignment: Mapping[str, Any],
    *,
    training_design: str,
) -> None:
    """verify_candidate on the execution window.  Not numeric.moved_points.

    input_only is verified on the 192-point X.  whole_window / recombination
    are verified on the 240-point [X, y] window.  A rejected window fails the
    UID; windows are not dropped and the UID is not silently rewritten to
    identity.  Prefix-absolute indices never reach this path.
    """
    allowed = tuple(legal_forecast_ops())
    origin = int(spec.training_prefix_end)
    anchors = [int(anchor) for anchor in spec.anchors if int(anchor) + HORIZON <= origin]
    for uid, value in assignment.items():
        if value is None:
            continue
        raw = np.asarray(bundle.values[str(uid)], dtype=np.float64)
        program = Program.from_steps(list(value), source="dev_train1")
        candidate = Candidate.program_candidate("legality", program, source="dev_train1")
        if not anchors:
            raise numeric.PreparationFailed(
                "train uid %s has no legal anchors; not filling with identity" % uid
            )
        for anchor in anchors:
            if training_design == "input_only":
                start = int(anchor) - CONTEXT_LENGTH
                window = raw[start:int(anchor)]
                if window.size != CONTEXT_LENGTH:
                    raise numeric.PreparationFailed(
                        "train uid %s input_only window at %s is not 192" % (uid, anchor)
                    )
            else:
                start = int(anchor) - CONTEXT_LENGTH
                stop = int(anchor) + HORIZON
                window = raw[start:stop]
                if window.size != CONTEXT_LENGTH + HORIZON:
                    raise numeric.PreparationFailed(
                        "train uid %s whole window at %s is not 240" % (uid, anchor)
                    )
            # require_finite_output=False matches the pipeline that actually
            # runs: numeric._apply is run_pipeline THEN _linear_integrity, and
            # it raises PreparationFailed if anything is still non-finite
            # afterwards.  Demanding finiteness *before* integrity here would
            # newly reject gap-preserving operators that the approved
            # execution path fills legally. The run's explicit permission
            # applies equally to every program and role.
            artifact = verify_candidate(
                candidate,
                window,
                allowed_operators=allowed,
                maximum_modified_fraction=effective_modified_cap(spec),
                require_finite_output=False,
            )
            receipt = artifact.receipt
            if str(getattr(receipt, "status", "")) == "rejected":
                raise numeric.PreparationFailed(
                    "verify_candidate rejected train uid %s anchor=%s code=%s; "
                    "windows are not dropped and the UID is not rewritten to identity"
                    % (uid, anchor, getattr(receipt, "rejection_code", "") or "")
                )


def legal_forecast_ops() -> list[str]:
    names = []
    for name in OPERATOR_NAMES:
        meta = OPERATOR_METADATA[name]
        if "forecast" not in (meta.get("allowed_tasks") or []):
            continue
        if meta.get("shape_changing") or meta.get("is_alias") or meta.get("changes_target_space"):
            continue
        names.append(name)
    return sorted(names)


def onestep_candidate(op: str, period: int) -> tuple[str, Any]:
    if op == "identity":
        return "identity", None
    schema = OPERATOR_METADATA[op].get("public_parameter_schema") or {}
    required = schema.get("required") or []
    params: dict[str, Any] = {}
    for key in required:
        if key == "period":
            params[key] = int(period)
        else:
            return op, "UNAVAILABLE"
    return op, ((op, params),)


def onestep_menu(spec: RunSpec) -> list[tuple[str, Any]]:
    ops = spec.onestep_ops if spec.onestep_ops is not None else ("identity", *legal_forecast_ops())
    out = []
    for op in ops:
        label, value = onestep_candidate(str(op), spec.period)
        if value == "UNAVAILABLE":
            continue
        out.append((label, value))
    return out


def harmed_stats(utilities: Sequence[float]) -> dict[str, Any]:
    if not utilities:
        return {"harmed_fraction": None, "max_single_series_harm": None, "n": 0}
    harmed = [item for item in utilities if item < -MATERIAL]
    worst = min(utilities)
    return {
        "harmed_fraction": round(len(harmed) / len(utilities), 6),
        "max_single_series_harm": round(abs(min(0.0, worst)), 6),
        "n": len(utilities),
        "n_harmed": len(harmed),
    }


def fit_arm(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    train_assignment: Mapping[str, Any],
    budget: Budget,
    arm: str,
    origin: int,
    training_design: str | None = None,
    recombination: str | None = None,
) -> numeric.FrozenRidge:
    design_name = spec.training_design if training_design is None else str(training_design)
    reco = spec.recombination if recombination is None else recombination
    with budget.lock:
        budget.require(fits=1)
    assert_train_windows_legal(
        spec, bundle, train_assignment, training_design=design_name,
    )
    design = numeric.build_training_design(
        bundle.roster(), bundle.values, train_assignment, numeric_config(spec),
        origin=int(origin),
        training_design=design_name,
        recombination=reco,
    )
    with budget.lock:
        budget.require(fits=1)
        budget.spend_fits(arm, 1)
    model = numeric.fit_assignment(design)
    if int(model.consumer_fits) != 1:
        raise RuntimeError("shared Ridge fit did not report exactly one fit")
    with budget.lock:
        budget.completed_fits_by_arm[arm] = budget.completed_fits_by_arm.get(arm, 0) + 1
        budget.persist()
    return model


def predict_arm(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    model: numeric.FrozenRidge,
    eval_assignment: Mapping[str, Any],
    origin: int,
) -> dict[str, Any]:
    for uid, steps in eval_assignment.items():
        if steps is None:
            continue
        raw = np.asarray(bundle.values[uid], dtype=np.float64)[int(origin)-CONTEXT_LENGTH:int(origin)]
        program = Program.from_steps(list(steps), source="dev_train1")
        artifact = verify_candidate(
            Candidate.program_candidate("predict-legality", program, source="dev_train1"),
            raw, allowed_operators=tuple(legal_forecast_ops()),
            maximum_modified_fraction=effective_modified_cap(spec), require_finite_output=False,
        )
        if artifact.receipt.status == "rejected":
            raise numeric.PreparationFailed("prediction uid %s rejected: %s" % (uid, artifact.receipt.rejection_code))
    pred = numeric.predict_assignment(
        model, bundle.roster(), bundle.values, eval_assignment,
        numeric_config(spec), origin=int(origin),
    )
    if int(pred.get("consumer_fits") or 0) != 0:
        raise RuntimeError("predict_assignment billed a fit; eval-period fits must be 0")
    return pred


def score_arm(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    predictions: Mapping[str, Any],
    origin: int,
) -> dict[str, Any]:
    scored = numeric.score_predictions(
        predictions, bundle.roster(), bundle.values, numeric_config(spec),
        origin=int(origin),
    )
    mean = scored.get("mean_smase")
    return {
        **scored,
        "mean_smase": ("UNKNOWN" if mean is None else float(mean)),
        "denominator_was_not_shrunk": True,
        "complete_population": bool(scored.get("complete")),
    }


def population_from_scores(
    eval_uids: Sequence[str], scored: Mapping[str, Any],
) -> dict[str, Any]:
    losses = list(scored.get("per_view_smase") or [])
    uids = list(scored.get("eval_uids") or eval_uids)
    per_uid: dict[str, Any] = {}
    unknown: list[str] = []
    readable: list[float] = []
    for uid, loss in zip(uids, losses):
        if loss is None:
            per_uid[uid] = "UNKNOWN"
            unknown.append(uid)
        else:
            per_uid[uid] = float(loss)
            readable.append(float(loss))
    for uid in eval_uids:
        if uid not in per_uid:
            per_uid[str(uid)] = "UNKNOWN"
            unknown.append(str(uid))
    mean: Any = "UNKNOWN"
    if not unknown and readable and len(readable) == len(list(eval_uids)):
        mean = float(statistics.fmean(readable))
    diagnostic: Any = float(statistics.fmean(readable)) if readable else "UNKNOWN"
    return {
        "mean_smase": mean,
        "diagnostic_readable_mean_smase": diagnostic,
        "per_uid_smase": per_uid,
        "n_eval": len(list(eval_uids)),
        "n_readable": len(readable),
        "n_unknown": len(unknown),
        "unknown_uids": unknown,
        "denominator_was_not_shrunk": True,
    }


def unknown_population(eval_uids: Sequence[str], why: str) -> dict[str, Any]:
    return {
        "mean_smase": "UNKNOWN",
        "diagnostic_readable_mean_smase": "UNKNOWN",
        "per_uid_smase": {str(uid): "UNKNOWN" for uid in eval_uids},
        "n_eval": len(list(eval_uids)),
        "n_readable": 0,
        "n_unknown": len(list(eval_uids)),
        "unknown_uids": list(eval_uids),
        "denominator_was_not_shrunk": True,
        "why": why,
    }


def _sign(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "UNKNOWN"
    if value > MATERIAL:
        return "POSITIVE"
    if value < -MATERIAL:
        return "NEGATIVE"
    return "NEUTRAL"


def build_slow_card(
    *,
    spec: RunSpec,
    train_records: Sequence[Mapping[str, Any]],
    eval_records_by_origin: Mapping[int, Sequence[Mapping[str, Any]]],
    formation_scores: Mapping[int, Mapping[str, Any]],
    xy_decomposition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    failures = []
    successes = []
    for origin, rows in eval_records_by_origin.items():
        scored = formation_scores.get(int(origin)) or {}
        per_uid = dict(scored.get("per_uid_utility") or {})
        for row in rows:
            uid = str(row.get("series_uid"))
            util = per_uid.get(uid, "UNKNOWN")
            item = {
                "series_uid": uid,
                "origin": int(origin),
                "role": "eval_predict",
                "deployed_program": row.get("deployed_program"),
                "typed_steps": row.get("typed_steps"),
                "utility": util,
                "sign": _sign(util),
                "valid": bool(row.get("valid_mechanism_decision")),
                "public_features": row.get("public_features"),
            }
            if item["sign"] == "NEGATIVE":
                failures.append(item)
            elif item["sign"] == "POSITIVE":
                successes.append(item)
    train_rows = []
    for row in train_records:
        train_rows.append({
            "series_uid": row.get("series_uid"),
            "role": "train_workflow",
            "deployed_program": row.get("deployed_program"),
            "typed_steps": row.get("typed_steps"),
            "valid": bool(row.get("valid_mechanism_decision")),
            "public_features": row.get("public_features"),
            "observed_length": row.get("observed_length"),
            "execution_length": row.get("execution_length"),
            "region_geometry": row.get("region_geometry"),
            "action_frame": {
                "observation": row.get("feature_window"),
                "execution_length": row.get("execution_length"),
            },
            "loo_credit": "UNKNOWN",
            "loo_why": "leave-one-train-out was not measured",
            "this_uid_is_not_a_harmed_eval_uid": True,
        })
    return {
        "pattern_id": "devtrain1-formation-boundary",
        "observable_signature": {
            "task_kind": "forecast",
            "decision_role": "slow",
            "scope": "formation batch boundary, not one series",
            "consumer_family": CONSUMER_STRUCTURE.get("family"),
            "consumer_shared_across_series": CONSUMER_STRUCTURE.get(
                "shared_across_series"),
            "input_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "series_features_are_per_member_not_aggregated": True,
            "why_no_representative_vector": (
                "averaging 20 train and 20 eval feature vectors into one "
                "'representative' series would be an invented observation, "
                "and an eval UID's features are not the training UID's "
                "features.  Per-member vectors are under "
                "training_batch.members[].public_features and "
                "failing_eval_decisions[].public_features."
            ),
            "how_slow_reads_this": (
                "slow_agent._public_features_from_card keeps only registered "
                "observable feature keys, so this signature resolves the "
                "boundary view without asserting a per-series predicate.  "
                "That is the honest state of a batch decision, not a "
                "missing field."
            ),
        },
        "consumer_structure": dict(CONSUMER_STRUCTURE),
        "what_happened": (
            "Old Skill generated one Workflow per train series on the frozen "
            "training prefix, one shared Ridge was fitted, and each formation "
            "eval series received its own predict-side Workflow."
        ),
        "training_batch": {
            "n_train": len(train_rows),
            "members": train_rows,
            "aggregate_by_origin": {
                str(origin): {
                    "mean_smase": (formation_scores.get(int(origin)) or {}).get("mean_smase"),
                    "complete": (formation_scores.get(int(origin)) or {}).get("complete_population"),
                }
                for origin, _pos in ((item[1], item[0]) for item in spec.formation_units)
            },
        },
        "failing_eval_decisions": failures,
        "matched_successes": successes,
        "eval_uid_is_not_a_train_uid": (
            "A harmed eval series is not a training series that was changed "
            "badly.  Shared-model loss on an eval UID is not the independent "
            "causal effect of that UID's own training Workflow."
        ),
        "loo_credits": {row["series_uid"]: "UNKNOWN" for row in train_rows},
        "observations_vs_hypotheses": (
            "Numbers on this card are observations.  Mechanism talk in a "
            "proposal is a hypothesis and may be wrong."
        ),
        "xy_decomposition": xy_decomposition or {
            "status": "not_supplied",
            "why": "not required before Slow; formation-only if present; u12/u13 never enter",
        },
        "does_not_include_u12_u13": True,
        "formation_units": [list(row) for row in spec.formation_units],
        "what_you_may_write": (
            "Exactly one edit on one surface from writable_surface_catalog. "
            "ADD, PATCH, or no_proposal are all legal.  Keeping a surface "
            "unchanged is legal."
        ),
        "select_cannot_veto_deployment": (
            "This package does not run Support admission.  Fast's chosen "
            "program is the delivered program after legality."
        ),
    }


def run_slow_boundary(
    *,
    spec: RunSpec,
    machinery: Mapping[str, Any],
    parent: Any,
    task_ctx: Any,
    card: Mapping[str, Any],
    budget: Budget,
    slow_factory: Callable[[], Any],
    store: Any,
    controller: Any,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if checkpoint_path.is_file():
        saved = read_json(checkpoint_path)
        if float(saved.get("maximum_modified_fraction", 0.35)) != effective_modified_cap(spec):
            raise RuntimeError("Slow boundary cap mismatch; old patch is not re-used")
        saved["resumed_from_checkpoint"] = True
        saved["slow_was_not_reasked"] = True
        sha = (saved.get("applied") or {}).get("runtime_bundle_sha")
        if saved.get("outcome") == "CANDIDATE_COMPILED" and sha:
            fork = Path(store.root) / str(sha)
            if fork.is_dir():
                recompiled = machinery["compile_snapshot"](fork, verify_lock=False)
                if recompiled.runtime_bundle_sha != sha:
                    raise RuntimeError("recompiled Slow fork SHA mismatch")
                saved["candidate_snapshot"] = recompiled
        return saved

    parent_mat = store.materialize(parent)
    catalog = know.build_catalog(controller=controller, parent=parent_mat)
    out: dict[str, Any] = {
        "maximum_modified_fraction": effective_modified_cap(spec),
        "surfaces_offered": [row["surface_id"] for row in catalog],
        "card_pattern_id": card.get("pattern_id"),
        "task_context_passed": True,
    }
    if not catalog:
        out["outcome"] = "NO_AUTHORISED_SURFACE"
        out["treatment"] = "NO_UPDATE_TREATMENT"
        atomic_write_json(checkpoint_path, out)
        return out

    proposal = know.propose_update(
        slow_factory=slow_factory, card=card, catalog=catalog,
        snapshot=parent, task_context=task_ctx, attempts=SLOW_ATTEMPTS,
    )
    del budget
    why = str(proposal.get("why") or "")
    if why == "ACCOUNT_OR_PERMISSION_FAULT":
        raise AccountFault(_sanitize(proposal.get("detail") or why))
    out["proposal"] = {k: v for k, v in proposal.items() if k != "manifest"}
    if not proposal.get("proposed"):
        out["outcome"] = {
            "SLOW_ABSTAINED": "NO_UPDATE",
            "TRANSPORT_TRANSIENT_FAULT": "BOUNDARY_TRANSPORT_FAULT",
        }.get(why, "NO_PROPOSAL")
        out["treatment"] = "NO_UPDATE_TREATMENT"
        atomic_write_json(checkpoint_path, out)
        return out

    applied = know.apply_update(
        controller=controller, store=store, snapshot=parent,
        manifest=proposal["manifest"], catalog=catalog,
    )
    out["applied"] = {k: v for k, v in applied.items() if k != "snapshot"}
    if not applied.get("applied"):
        out["outcome"] = "INVALID_EDIT"
        out["treatment"] = "NO_UPDATE_TREATMENT"
        atomic_write_json(checkpoint_path, out)
        return out
    out["candidate_snapshot"] = applied["snapshot"]
    out["outcome"] = "CANDIDATE_COMPILED"
    out["treatment"] = "UPDATE_TREATMENT"
    serialisable = {k: v for k, v in out.items() if k != "candidate_snapshot"}
    atomic_write_json(checkpoint_path, serialisable)
    return out


def make_slow_factory(
    *,
    spec: RunSpec,
    machinery: Mapping[str, Any],
    backend_factory: Callable[[], Any],
    extra: Mapping[str, Any],
    budget: Budget,
) -> Callable[[], Any]:
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent

    metered = MeteredBudgetBackend(backend_factory(), budget=budget, arm="slow")
    inner = BoundedTransportRetry(metered, attempts=TRANSPORT_ATTEMPTS)
    core = RoleInjectingCore(
        machinery["TTHAAgentCore"](
            inner,
            UnavailableToolGateway(),
            model=spec.requested_model,
            base_url=_core_base_url(spec),
        ),
        extra,
    )

    def factory() -> Any:
        return TTHASlowAgent(core)

    factory.backend = metered  # type: ignore[attr-defined]
    return factory


def _identity_assignment(uids: Sequence[str]) -> dict[str, None]:
    return {str(uid): None for uid in uids}


def _records_by_uid(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["series_uid"]): row for row in rows}


def records_to_assignment(
    uids: Sequence[str], rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    return complete_assignment(uids, _records_by_uid(rows))


def calibrate_fixed(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    budget: Budget,
    static_by_origin: Mapping[int, Mapping[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    menu = onestep_menu(spec)
    calibration_path = run_dir / "fixed_onestep_calibration.json"
    context = _frozen_context(spec, bundle)
    context["menu"] = [[label, steps_to_json(value)] for label, value in menu]
    if calibration_path.is_file():
        saved = read_json(calibration_path)
        if _legacy_cap_context(saved.get("context") or {}) != context:
            raise RuntimeError("fixed calibration context mismatch; use a distinct run ID")
        return saved
    readings = []
    train_ids = spec.train_uids
    eval_ids = spec.eval_uids
    prefix = int(spec.training_prefix_end)
    winner = None
    for label, value in menu:
        train_asg = {uid: value for uid in train_ids}
        eval_asg = {uid: value for uid in eval_ids}
        try:
            model = freeze_model(
                spec=spec, bundle=bundle, assignment=train_asg,
                budget=budget, arm="fixed_menu/" + label, run_dir=run_dir,
            )
        except (PackageCeiling, AccountFault):
            # A ceiling or an account fault is a package stop, not a verdict
            # on this candidate.  Reporting it as FIT_FAILED and continuing
            # would keep spending and would name a menu winner chosen from a
            # truncated menu.
            raise
        except Exception as exc:  # noqa: BLE001
            readings.append({
                "op": label, "status": "FIT_FAILED",
                "why": _sanitize("%s: %s" % (type(exc).__name__, exc)),
            })
            continue
        origin_rows = []
        complete = True
        utilities: list[float] = []
        smases: list[float] = []
        for _position, origin in spec.formation_units:
            try:
                pred = predict_arm(
                    spec=spec, bundle=bundle, model=model,
                    eval_assignment=eval_asg, origin=origin,
                )
                scored = score_arm(
                    spec=spec, bundle=bundle, predictions=pred, origin=origin,
                )
            except (PackageCeiling, AccountFault):
                raise
            except Exception as exc:  # noqa: BLE001
                origin_rows.append({
                    "origin": origin, "status": "SCORE_FAILED",
                    "why": _sanitize(type(exc).__name__),
                })
                complete = False
                continue
            pop = population_from_scores(eval_ids, scored)
            origin_rows.append({"origin": origin, "population": pop})
            if pop["mean_smase"] == "UNKNOWN":
                complete = False
            else:
                smases.append(float(pop["mean_smase"]))
            static_pop = static_by_origin.get(int(origin)) or {}
            static_per = dict(static_pop.get("per_uid_smase") or {})
            arm_per = dict(pop.get("per_uid_smase") or {})
            for uid in eval_ids:
                static_v = static_per.get(uid)
                arm_v = arm_per.get(uid)
                if isinstance(static_v, (int, float)) and isinstance(arm_v, (int, float)):
                    utilities.append(float(static_v) - float(arm_v))
        risk = harmed_stats(utilities)
        mean_smase: Any = "UNKNOWN"
        if complete and smases:
            mean_smase = float(statistics.fmean(smases))
        row = {
            "op": label,
            "status": "READ" if complete else "INCOMPLETE",
            "mean_formation_smase": mean_smase,
            "risk": risk,
            "typed_steps": steps_to_json(value) if value is not None else None,
            "origins": origin_rows,
            "what_this_is_not": (
                "not the strongest of all legal 1-4 step combinations; "
                "it is the calibrated fixed single-step baseline"
            ),
        }
        readings.append(row)
        if complete and isinstance(mean_smase, float):
            if winner is None or float(mean_smase) < float(winner["mean_formation_smase"]):
                winner = row
    chosen = winner or {
        "op": "identity", "typed_steps": None,
        "mean_formation_smase": "UNKNOWN",
        "why": "no complete one-step candidate; identity frozen",
        "what_this_is_not": "not the strongest of all legal combinations",
    }
    out = {
        "context": context,
        "menu": [label for label, _value in menu],
        "readings": readings,
        "chosen": chosen,
        "label": "fixed_onestep_baseline",
    }
    atomic_write_json(run_dir / "fixed_onestep_calibration.json", out)
    return out


def generate_train_w(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    snapshot: Any,
    machinery: Mapping[str, Any],
    backend_factory: Callable[[], Any],
    arm: str,
    task_ctx: Any,
    budget: Budget,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    rows = run_batch_fast(
        spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
        backend_factory=backend_factory, uids=spec.train_uids,
        role="train_workflow", arm=arm, position=0,
        origin=spec.training_prefix_end, task_ctx=task_ctx, budget=budget,
        checkpoint_path=run_dir / arm / "train_w.json",
    )
    assignment, missing = records_to_assignment(spec.train_uids, rows)
    return rows, assignment, missing


def generate_eval_v(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    snapshot: Any,
    machinery: Mapping[str, Any],
    backend_factory: Callable[[], Any],
    arm: str,
    position: int,
    origin: int,
    task_ctx: Any,
    budget: Budget,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    rows = run_batch_fast(
        spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
        backend_factory=backend_factory, uids=spec.eval_uids,
        role="predict_input", arm=arm, position=position, origin=origin,
        task_ctx=task_ctx, budget=budget,
        checkpoint_path=run_dir / arm / ("eval_v_u%03d.json" % position),
    )
    assignment, missing = records_to_assignment(spec.eval_uids, rows)
    return rows, assignment, missing


def freeze_model(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    assignment: Mapping[str, Any],
    budget: Budget,
    arm: str,
    run_dir: Path,
) -> numeric.FrozenRidge:
    path = run_dir / arm / "frozen_ridge.json"
    receipt_path = run_dir / arm / "frozen_assignment.json"
    fingerprint = _assignment_fingerprint(assignment)
    expected = {"arm": arm, "assignment_fingerprint": fingerprint,
                "context": _frozen_context(spec, bundle)}
    if path.is_file():
        saved = read_json(path)
        previous = dict(saved.get("dev_train1_context") or {})
        previous["context"] = _legacy_cap_context(previous.get("context") or {})
        if previous != expected:
            raise RuntimeError("frozen model context mismatch; completed state is not overwritten")
        return numeric.FrozenRidge.from_dict(saved)
    if receipt_path.is_file():
        raise RuntimeError("frozen model payload missing; do not silently refit a completed receipt")
    model = fit_arm(
        spec=spec, bundle=bundle, train_assignment=assignment,
        budget=budget, arm=arm, origin=int(spec.training_prefix_end),
    )
    payload = model.to_dict()
    payload["dev_train1_context"] = expected
    atomic_write_json(path, payload)
    atomic_write_json(receipt_path, expected)
    return model


def _frozen_context(spec: RunSpec, bundle: DataBundle) -> dict[str, Any]:
    return {
        "maximum_modified_fraction": effective_modified_cap(spec),
        "data_identity": "KDD2018_WITH_MISSING_DEVELOPMENT" if spec.live_api else "SYNTHETIC_FIXTURE",
        "train_uids": list(spec.train_uids), "eval_uids": list(spec.eval_uids),
        "anchors": list(spec.anchors), "period": int(spec.period),
        "training_prefix_end": int(spec.training_prefix_end),
        "training_design": spec.training_design, "recombination": spec.recombination,
        "formation_units": [list(x) for x in spec.formation_units],
        "consumer": dict(CONSUMER_STRUCTURE),
    }


def _assignment_fingerprint(assignment: Mapping[str, Any]) -> str:
    payload = {}
    for uid in assignment:
        value = assignment[uid]
        payload[str(uid)] = None if value is None else steps_to_json(value)
    return json.dumps(payload, sort_keys=True, default=_json_default)


def score_units(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    model: numeric.FrozenRidge,
    assignment_by_origin: Mapping[int, Mapping[str, Any] | None],
    missing_by_origin: Mapping[int, Sequence[str]],
    units: Sequence[tuple[int, int]],
    arm: str,
    run_dir: Path,
    static_by_origin: Mapping[int, Mapping[str, Any]] | None = None,
    write_scores: bool = True,
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for position, origin in units:
        origin_i = int(origin)
        missing = list(missing_by_origin.get(origin_i) or ())
        assignment = assignment_by_origin.get(origin_i)
        path = run_dir / arm / ("predictions_u%03d.json" % position)
        score_path = run_dir / arm / ("scores_u%03d.json" % position)
        if path.is_file():
            existing = read_json(path)
            if existing.get("predictions_frozen") and int(existing.get("origin") or -1) == origin_i:
                if write_scores:
                    if score_path.is_file():
                        out[origin_i] = read_json(score_path)
                        continue
                    pred = {
                        "eval_uids": existing["eval_uids"],
                        "predictions": np.asarray(existing["predictions"], dtype=np.float64),
                        "consumer_fits": 0,
                    }
                    pop = _score_prediction_payload(
                        spec, bundle, pred, origin_i, static_by_origin,
                    )
                    atomic_write_json(score_path, pop)
                    out[origin_i] = pop
                else:
                    out[origin_i] = existing
                continue
        if assignment is None or missing:
            pop = unknown_population(
                spec.eval_uids,
                "incomplete eval assignment missing=%s" % missing,
            )
            atomic_write_json(path, {
                "origin": origin_i, "population": pop, "predictions_frozen": False,
            })
            out[origin_i] = pop
            continue
        pred = predict_arm(
            spec=spec, bundle=bundle, model=model,
            eval_assignment=assignment, origin=origin_i,
        )
        frozen = {
            "origin": origin_i,
            "eval_uids": list(pred["eval_uids"]),
            "predictions": np.asarray(pred["predictions"]).tolist(),
            "consumer_fits": 0,
            "behavior_point_count": pred.get("behavior_point_count"),
            "truth_not_included": True,
            "predictions_frozen": True,
        }
        atomic_write_json(path, frozen)
        if not write_scores:
            out[origin_i] = frozen
            continue
        pop = _score_prediction_payload(spec, bundle, pred, origin_i, static_by_origin)
        atomic_write_json(score_path, pop)
        out[origin_i] = pop
    return out


def _score_prediction_payload(
    spec: RunSpec,
    bundle: DataBundle,
    pred: Mapping[str, Any],
    origin_i: int,
    static_by_origin: Mapping[int, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    scored = score_arm(spec=spec, bundle=bundle, predictions=pred, origin=origin_i)
    pop = population_from_scores(spec.eval_uids, scored)
    if static_by_origin and origin_i in static_by_origin:
        static_per = dict(static_by_origin[origin_i].get("per_uid_smase") or {})
        utilities = {}
        for uid in spec.eval_uids:
            static_v = static_per.get(uid)
            arm_v = pop["per_uid_smase"].get(uid)
            if isinstance(static_v, (int, float)) and isinstance(arm_v, (int, float)):
                utilities[uid] = float(static_v) - float(arm_v)
            else:
                utilities[uid] = "UNKNOWN"
        pop["per_uid_utility"] = utilities
        pop["risk"] = harmed_stats(
            [item for item in utilities.values() if isinstance(item, (int, float))]
        )
    return pop


def score_frozen_eval_units(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    arm: str,
    run_dir: Path,
    static_by_origin: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    dummy = numeric.FrozenRidge(
        coefficients=np.zeros((CONTEXT_LENGTH + 1, HORIZON), dtype=np.float64),
        n_train_rows=1, n_features=CONTEXT_LENGTH, n_targets=HORIZON,
        ridge_alpha=1.0, training_design="whole_window", recombination=None,
        origin=int(spec.training_prefix_end), consumer_fits=0,
    )
    return score_units(
        spec=spec, bundle=bundle, model=dummy,
        assignment_by_origin={}, missing_by_origin={},
        units=spec.eval_units, arm=arm, run_dir=run_dir,
        static_by_origin=static_by_origin, write_scores=True,
    )


def preflight_record(spec: RunSpec, snapshot: Any) -> dict[str, Any]:
    return {
        "maximum_modified_fraction": effective_modified_cap(spec),
        "effective_snapshot_modified_fraction": float(snapshot.verification.get("max_modified_fraction", 0.35)),
        "package": PACKAGE,
        "requested_model": spec.requested_model,
        "base_url": spec.base_url,
        "live_api": bool(spec.live_api),
        "api_called": False,
        "returned_models": [],
        "requested_returned_correspondence": {
            "requested": REQUESTED_MODEL,
            "required_returned": REQUIRED_RETURNED_MODEL,
            "literal_equality_not_required": True,
            "arbitrary_model_not_allowed": True,
        },
        "no_flash_or_sol_fallback": True,
        "h0": str(H0_ROOT),
        "h0_runtime_bundle_sha": getattr(snapshot, "runtime_bundle_sha", None),
        "learned_skill_files": [],
        "frozen_program_supply": False,
        "roster": planned_roster_contract(),
        "consumer_structure": dict(CONSUMER_STRUCTURE),
        "task_context_hook_gap": (
            "TaskContext.to_dict() carries task_spec.downstream_model_class="
            "pooled_ridge_a1, metric=sMASE, horizon=48, and "
            "fixed_downstream_model_id=pooled-ridge-a1.  Alpha, unpenalized "
            "intercept, per-window center/scale, squared training loss vs "
            "sMASE scoring, and train/predict role frames cannot enter that "
            "hashed object.  They are merged through TTHAAgentCore.run_stage "
            "public_input via RoleInjectingCore, and onto the Slow card."
        ),
        "ceilings": {
            "llm": spec.max_llm, "fits": spec.max_fits,
            "wall_seconds": spec.max_wall_seconds,
            "concurrency": spec.concurrency,
            "transport_attempts": TRANSPORT_ATTEMPTS,
        },
        "prior_package": dict(PRIOR_PACKAGE),
        "course_span": list(COURSE_SPAN),
    }


def development_roster_from_metadata() -> dict[str, Any]:
    """SEQ-3 u9/u10/u11 identity: span=[40, 80].  Do not swap in [0, 40]."""
    path = ROOT / "artifacts" / "main_protocol" / "p4ac_hec1_course_supply.json"
    doc = read_json(path)
    block = None
    for item in doc.get("blocks") or ():
        if list(item.get("span") or []) == COURSE_SPAN:
            block = item
            break
    if block is None:
        raise RuntimeError("p4ac supply has no span=[40, 80] block")
    return {
        "train_uids": [str(uid) for uid in block["face_b"]],
        "eval_uids": [str(uid) for uid in block["face_a"]],
        "span": list(COURSE_SPAN),
        "slice": str(block.get("slice") or COURSE_SLICE),
        "data_version": doc.get("data_version"),
        "origin_grid": list(doc.get("origin_grid") or []),
        "why": (
            "SEQ-3 u9/u10/u11 metadata origin=1656/2136/2376 with span=[40,80]; "
            "ForecastCell support_a evaluates face_a and trains face_b"
        ),
    }


def load_kdd_missing_development_bundle() -> DataBundle:
    """Read the with-missing archive.  Call only from --live."""
    from evaluation.main_protocol_p4.preflight_natural_gap_variant import load_variant

    roster = development_roster_from_metadata()
    values = load_variant()
    needed = list(roster["train_uids"]) + list(roster["eval_uids"])
    missing = [uid for uid in needed if uid not in values]
    if missing:
        raise RuntimeError("with-missing variant missing uids %s" % missing[:8])
    subset = {uid: np.asarray(values[uid], dtype=np.float64) for uid in needed}
    return DataBundle(
        values=subset,
        train_uids=list(roster["train_uids"]),
        eval_uids=list(roster["eval_uids"]),
        period=PERIOD,
    )


def formation_xy_decomposition(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    train_assignment: Mapping[str, Any] | None,
    eval_assignment_by_origin: Mapping[int, Mapping[str, Any] | None],
    static_model: numeric.FrozenRidge,
    form_model: numeric.FrozenRidge | None,
    static_by_origin: Mapping[int, Mapping[str, Any]],
    budget: Budget,
    run_dir: Path,
) -> dict[str, Any]:
    """Formation-only X/y designs.  Predict V is frozen.  No new LLM.

    whole_window and Xp_yp share a fit.  X0_y0 reuses Static.  Identity W
    makes Xp_y0 and X0_yp the same as X0_y0.  input_only is fitted once even
    for identity W because B(X) may differ from B([X,y])[:192].
    """
    path = run_dir / "xy_decomposition.json"
    if path.is_file():
        return read_json(path)
    if train_assignment is None:
        payload = {
            "status": "UNKNOWN",
            "why": "formation train assignment incomplete; not invented",
        }
        atomic_write_json(path, payload)
        return payload

    identity = all(value is None for value in train_assignment.values())
    cache: dict[str, numeric.FrozenRidge] = {"identity_whole": static_model}
    if form_model is not None and spec.training_design == "whole_window":
        cache["w_whole"] = form_model

    def obtain(key: str, design: str, reco: str | None, assignment: Mapping[str, Any]) -> tuple[Any, str]:
        if key in cache:
            return cache[key], "reused"
        try:
            model = freeze_model(
                spec=replace(spec, training_design=design, recombination=reco),
                bundle=bundle, assignment=assignment,
                budget=budget, arm="xy_%s" % key, run_dir=run_dir,
            )
        except (
            numeric.PreparationFailed,
            numeric.MissingAssignmentError,
            numeric.DegenerateContext,
        ) as exc:
            # Numerical reasons stay UNKNOWN.  PackageCeiling / AccountFault
            # are package stops and are deliberately not caught here.
            return None, "%s: %s" % (type(exc).__name__, _sanitize(exc, 160))
        cache[key] = model
        return model, "fitted"

    menu = [
        ("whole_window", "whole_window", None, "w_whole", train_assignment),
        ("input_only", "input_only", None, "w_input_only", train_assignment),
        ("X0_y0", "recombination", "X0_y0", "identity_whole", _identity_assignment(spec.train_uids)),
        ("Xp_y0", "recombination", "Xp_y0", "w_xp_y0", train_assignment),
        ("X0_yp", "recombination", "X0_yp", "w_x0_yp", train_assignment),
        ("Xp_yp", "recombination", "Xp_yp", "w_whole", train_assignment),
    ]
    rows = []
    for label, design, reco, key, assignment in menu:
        how = "reused"
        if label in ("Xp_y0", "X0_yp") and identity:
            key = "identity_whole"
            how = "identity_W_equals_X0_y0"
        if label == "Xp_yp" and "w_whole" in cache:
            how = "same_as_whole_window"
        model, obtained = obtain(key, design, reco, assignment)
        if how == "reused" and obtained == "fitted":
            how = obtained
        elif obtained != "reused" and obtained != "fitted":
            rows.append({
                "label": label, "training_design": design, "recombination": reco,
                "status": "UNKNOWN", "why": obtained, "fit": obtained,
            })
            continue
        origin_scores = {}
        complete = True
        smases: list[float] = []
        for position, origin in spec.formation_units:
            eval_asg = eval_assignment_by_origin.get(int(origin))
            if eval_asg is None or model is None:
                origin_scores[str(origin)] = "UNKNOWN"
                complete = False
                continue
            try:
                pred = predict_arm(
                    spec=spec, bundle=bundle, model=model,
                    eval_assignment=eval_asg, origin=int(origin),
                )
                scored = score_arm(
                    spec=spec, bundle=bundle, predictions=pred, origin=int(origin),
                )
            except (PackageCeiling, AccountFault):
                raise
            except Exception as exc:  # noqa: BLE001
                origin_scores[str(origin)] = {
                    "status": "UNKNOWN",
                    "why": "%s: %s" % (type(exc).__name__, _sanitize(exc, 120)),
                }
                complete = False
                continue
            pop = population_from_scores(spec.eval_uids, scored)
            origin_scores[str(origin)] = pop["mean_smase"]
            if pop["mean_smase"] == "UNKNOWN":
                complete = False
            else:
                smases.append(float(pop["mean_smase"]))
            del position
        rows.append({
            "label": label,
            "training_design": design,
            "recombination": reco,
            "fit": how if key in cache else obtained,
            "mean_formation_smase": (
                float(statistics.fmean(smases)) if complete and smases else "UNKNOWN"
            ),
            "per_origin": origin_scores,
            "recommendation": None,
            "this_is_observation_not_a_lesson": True,
        })
    payload = {
        "status": "COMPUTED",
        "predict_v_fixed": True,
        "no_new_llm": True,
        "u12_u13_not_used": True,
        "designs": rows,
    }
    atomic_write_json(path, payload)
    return payload


def menu_xy_decomposition(*, spec: RunSpec, bundle: DataBundle,
                          static_model: numeric.FrozenRidge,
                          static_by_origin: Mapping[int, Mapping[str, Any]],
                          budget: Budget, run_dir: Path) -> dict[str, Any]:
    """Frozen single-step training menu; identical identity V within every contrast."""
    if (run_dir / "menu_xy_decomposition.json").is_file():
        return read_json(run_dir / "menu_xy_decomposition.json")
    result: dict[str, Any] = {}
    for label, steps in onestep_menu(spec):
        assignment = {uid: steps for uid in spec.train_uids}
        try:
            model = freeze_model(spec=spec, bundle=bundle, assignment=assignment,
                                 budget=budget, arm="fixed_menu/" + label, run_dir=run_dir)
        except (numeric.PreparationFailed, numeric.MissingAssignmentError, numeric.DegenerateContext):
            model = None
        result[label] = formation_xy_decomposition(
            spec=spec, bundle=bundle, train_assignment=assignment,
            eval_assignment_by_origin={origin: _identity_assignment(spec.eval_uids)
                                      for _, origin in spec.formation_units},
            static_model=static_model, form_model=model,
            static_by_origin=static_by_origin, budget=budget,
            run_dir=run_dir / "menu_decomposition" / label,
        )
    payload = {"menu": result, "evaluation_units": "FORMATION_ONLY",
               "predict_v": "identity held equal across all training variants",
               "not_additive_causal_shares": True, "project_llm_calls": 0}
    atomic_write_json(run_dir / "menu_xy_decomposition.json", payload)
    return payload


def dual_role_exposure(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    parent: Any,
    candidate: Any,
    run_dir: Path,
) -> dict[str, Any]:
    """Zero fits, zero LLM: does the compiled edit change what Fast reads?

    ``fast_agent`` resolves its view with
    ``extract_public_features(request.values)`` and ``role="fast"``
    (``fast_agent.py:776-780``), so this renders the same view over the same
    arrays the two **final** roles are about to be given -- the training
    prefix for each train UID, and the 192-point serve context for each eval
    UID at u12 and u13 -- and compares the rendered bytes, not a substring of
    a body.  A byte difference is what an ADD, a ``.body`` PATCH, an
    ``.observable_applicability`` PATCH and a ``candidate_policy`` control
    PATCH all have in common; a substring probe only catches the second.

    Training-role exposure alone is a real treatment: one shared Ridge is
    fitted from every train UID's prepared windows, so a training-side change
    moves every eval prediction even when no eval decision reads the edit.
    Exposure on neither role means the paid new-vs-old course would compare
    two identical knowledge renderings.
    """
    rows: list[dict[str, Any]] = []

    def one(role: str, uid: str, observed: np.ndarray, origin: int | None) -> None:
        features = dict(
            extract_public_features(
                np.asarray(observed, dtype=np.float64), task_kind="forecast",
            )
        )
        before = know.fast_view_bytes(parent, features)
        after = know.fast_view_bytes(candidate, features)
        rows.append({
            "decision_role": role,
            "series_uid": str(uid),
            "origin": None if origin is None else int(origin),
            "knowledge_view_differs": bool(before != after),
            "parent_view_chars": len(before),
            "candidate_view_chars": len(after),
        })

    prefix_end = int(spec.training_prefix_end)
    for uid in spec.train_uids:
        raw = np.asarray(bundle.values[str(uid)], dtype=np.float64)
        one("train_workflow", str(uid), raw[:prefix_end], None)
    for _position, origin in spec.eval_units:
        origin_i = int(origin)
        for uid in spec.eval_uids:
            raw = np.asarray(bundle.values[str(uid)], dtype=np.float64)
            one(
                "predict_input", str(uid),
                raw[origin_i - CONTEXT_LENGTH:origin_i], origin_i,
            )

    def count(role: str) -> tuple[int, int]:
        subset = [row for row in rows if row["decision_role"] == role]
        return len(subset), len([r for r in subset if r["knowledge_view_differs"]])

    train_n, train_exposed = count("train_workflow")
    predict_n, predict_exposed = count("predict_input")
    total_exposed = train_exposed + predict_exposed
    payload = {
        "cost": "zero Consumer fits, zero LLM calls",
        "view_role": "fast",
        "feature_source": "extract_public_features on the exact array each role receives",
        "compared": "full rendered harness view bytes, not a body substring",
        "train_decisions_checked": train_n,
        "train_decisions_the_edit_reaches": train_exposed,
        "predict_decisions_checked": predict_n,
        "predict_decisions_the_edit_reaches": predict_exposed,
        "per_decision": rows,
        "training_only_exposure_is_still_a_treatment": (
            "one shared Ridge is fitted from all prepared training windows, "
            "so a training-side knowledge change reaches every eval "
            "prediction through the model even with zero predict-side "
            "exposure.  Eval decisions that do not read the edit are not "
            "therefore 'pure sampling noise'."
        ),
        "what_this_is_not": (
            "not a coverage gate; it does not ask the condition to explain "
            "failures, only whether the thing about to be paid for is on "
            "screen at all"
        ),
    }
    if total_exposed == 0:
        payload["outcome"] = "COMPILED_BUT_NEVER_EXPOSED"
        payload["treatment"] = "COMPILED_BUT_NEVER_EXPOSED__UTILITY_UNTESTED"
        payload["compiled_candidate_receipt_preserved"] = True
    else:
        payload["outcome"] = "EXPOSED"
        payload["treatment"] = "UPDATE_TREATMENT"
    atomic_write_json(run_dir / "exposure_check.json", payload)
    return payload


def _safe_freeze(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    assignment: Mapping[str, Any] | None,
    missing: Sequence[str],
    budget: Budget,
    arm: str,
    run_dir: Path,
) -> tuple[numeric.FrozenRidge | None, str | None]:
    if assignment is None:
        return None, "incomplete train assignment missing=%s" % list(missing)
    try:
        return freeze_model(
            spec=spec, bundle=bundle, assignment=assignment,
            budget=budget, arm=arm, run_dir=run_dir,
        ), None
    except (
        numeric.PreparationFailed,
        numeric.MissingAssignmentError,
        numeric.DegenerateContext,
    ) as exc:
        return None, "%s: %s" % (type(exc).__name__, _sanitize(exc))


def run_package(
    *,
    spec: RunSpec,
    bundle: DataBundle,
    run_id: str,
    backend_factory: Callable[[], Any] | None = None,
    machinery: Mapping[str, Any] | None = None,
    snapshot: Any | None = None,
    skip_slow: bool = False,
) -> dict[str, Any]:
    if backend_factory is None:
        raise RuntimeError("backend_factory is required")

    run_dir = RUNS_ROOT / str(run_id)
    validate_run_cap(spec, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    machinery = machinery or load_machinery()
    snapshot = snapshot or load_h0_snapshot(machinery)
    snapshot = configure_snapshot_for_spec(spec, snapshot, machinery)
    atomic_write_json(run_dir / "run_cap.json", {
        "maximum_modified_fraction": effective_modified_cap(spec),
    })
    task_ctx = task_context_for(spec)
    budget = Budget.load(
        run_dir / "budget.json",
        max_llm=spec.max_llm, max_fits=spec.max_fits,
        max_wall_seconds=spec.max_wall_seconds,
    )
    budget.path = run_dir / "budget.json"
    if spec.live_api and budget.llm() == 0 and budget.fits() == 0:
        budget.prior_llm = int(PRIOR_PACKAGE["llm"])
        budget.prior_fits = int(PRIOR_PACKAGE["fits"])
    budget.concurrency_used = spec.concurrency
    budget.persist()

    preflight = preflight_record(spec, snapshot)
    atomic_write_json(run_dir / "preflight.json", preflight)

    store = machinery["SnapshotStore"](run_dir / "store")
    controller = machinery["EditController"](
        store, surfaces=machinery["SurfaceRegistry"](),
        router=machinery["FaultRouter"](),
    )
    store.materialize(snapshot)

    result: dict[str, Any] = {
        "package": PACKAGE,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": "RUNNING",
        "exit_code": 0,
        "h0_runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "arms": {},
    }

    try:
        static_train = _identity_assignment(spec.train_uids)
        static_eval = _identity_assignment(spec.eval_uids)
        static_model = freeze_model(
            spec=spec, bundle=bundle, assignment=static_train,
            budget=budget, arm="static", run_dir=run_dir,
        )
        static_formation = score_units(
            spec=spec, bundle=bundle, model=static_model,
            assignment_by_origin={
                origin: static_eval
                for _position, origin in spec.formation_units
            },
            missing_by_origin={},
            units=spec.formation_units, arm="static", run_dir=run_dir,
            write_scores=True,
        )
        score_units(
            spec=spec, bundle=bundle, model=static_model,
            assignment_by_origin={
                origin: static_eval for _position, origin in spec.eval_units
            },
            missing_by_origin={},
            units=spec.eval_units, arm="static", run_dir=run_dir,
            write_scores=False,
        )
        result["arms"]["static"] = {
            "train": "identity",
            "model": static_model.to_dict(),
            "formation_scores": {
                str(key): value for key, value in static_formation.items()
            },
        }

        fixed_cal = calibrate_fixed(
            spec=spec, bundle=bundle, budget=budget,
            static_by_origin=static_formation, run_dir=run_dir,
        )
        menu_xy = menu_xy_decomposition(
            spec=spec, bundle=bundle, static_model=static_model,
            static_by_origin=static_formation, budget=budget, run_dir=run_dir,
        )
        chosen_steps = steps_from_json(fixed_cal["chosen"].get("typed_steps"))
        fixed_train = {uid: chosen_steps for uid in spec.train_uids}
        fixed_eval = {uid: chosen_steps for uid in spec.eval_uids}
        fixed_model, fixed_why = _safe_freeze(
            spec=spec, bundle=bundle, assignment=fixed_train, missing=(),
            budget=budget, arm="fixed_onestep", run_dir=run_dir,
        )
        if fixed_model is not None:
            score_units(
                spec=spec, bundle=bundle, model=fixed_model,
                assignment_by_origin={
                    origin: fixed_eval for _p, origin in spec.eval_units
                },
                missing_by_origin={},
                units=spec.eval_units, arm="fixed_onestep", run_dir=run_dir,
                write_scores=False,
            )
            fixed_payload = fixed_model.to_dict()
        else:
            fixed_payload = None
        result["arms"]["fixed_onestep"] = {
            "calibration": {"chosen": fixed_cal["chosen"], "menu": fixed_cal["menu"]},
            "model": fixed_payload,
            "eval_frozen_unscored": True,
            "why_unscored": fixed_why,
        }

        form_train_rows, form_train_asg, form_train_missing = generate_train_w(
            spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
            backend_factory=backend_factory, arm="old_skill_formation",
            task_ctx=task_ctx, budget=budget, run_dir=run_dir,
        )
        form_model, form_why = _safe_freeze(
            spec=spec, bundle=bundle, assignment=form_train_asg,
            missing=form_train_missing, budget=budget,
            arm="old_skill_formation", run_dir=run_dir,
        )
        formation_eval_rows: dict[int, list[dict[str, Any]]] = {}
        formation_eval_asg: dict[int, dict[str, Any] | None] = {}
        formation_eval_missing: dict[int, list[str]] = {}
        for position, origin in spec.formation_units:
            rows, asg, missing = generate_eval_v(
                spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
                backend_factory=backend_factory, arm="old_skill_formation",
                position=position, origin=origin, task_ctx=task_ctx,
                budget=budget, run_dir=run_dir,
            )
            formation_eval_rows[int(origin)] = rows
            formation_eval_asg[int(origin)] = asg
            formation_eval_missing[int(origin)] = missing

        if form_model is None:
            formation_scores = {
                int(origin): unknown_population(spec.eval_uids, form_why or "formation fit failed")
                for _position, origin in spec.formation_units
            }
        else:
            formation_scores = score_units(
                spec=spec, bundle=bundle, model=form_model,
                assignment_by_origin=formation_eval_asg,
                missing_by_origin=formation_eval_missing,
                units=spec.formation_units, arm="old_skill_formation",
                run_dir=run_dir, static_by_origin=static_formation,
                write_scores=True,
            )

        xy = formation_xy_decomposition(
            spec=spec, bundle=bundle, train_assignment=form_train_asg,
            eval_assignment_by_origin=formation_eval_asg,
            static_model=static_model, form_model=form_model,
            static_by_origin=static_formation, budget=budget, run_dir=run_dir,
        )
        xy = {"agent_assignment": xy, "frozen_menu": menu_xy}
        if skip_slow:
            boundary = {"outcome": "SKIPPED", "treatment": "NO_UPDATE_TREATMENT"}
        else:
            extra_slow = public_extra("skill_revision", observed_length=0)
            extra_slow["consumer_structure"] = dict(CONSUMER_STRUCTURE)
            slow_factory = make_slow_factory(
                spec=spec, machinery=machinery, backend_factory=backend_factory,
                extra=extra_slow, budget=budget,
            )
            card = build_slow_card(
                spec=spec, train_records=form_train_rows,
                eval_records_by_origin=formation_eval_rows,
                formation_scores=formation_scores, xy_decomposition=xy,
            )
            boundary = run_slow_boundary(
                spec=spec, machinery=machinery, parent=snapshot, task_ctx=task_ctx,
                card=card, budget=budget, slow_factory=slow_factory,
                store=store, controller=controller,
                checkpoint_path=run_dir / "boundary.json",
            )
        result["boundary"] = {
            k: v for k, v in boundary.items() if k != "candidate_snapshot"
        }
        result["xy_decomposition"] = xy

        old_rows, old_asg, old_missing = generate_train_w(
            spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
            backend_factory=backend_factory, arm="old_skill",
            task_ctx=task_ctx, budget=budget, run_dir=run_dir,
        )
        old_model, old_why = _safe_freeze(
            spec=spec, bundle=bundle, assignment=old_asg, missing=old_missing,
            budget=budget, arm="old_skill", run_dir=run_dir,
        )
        old_eval_asg: dict[int, dict[str, Any] | None] = {}
        old_eval_missing: dict[int, list[str]] = {}
        for position, origin in spec.eval_units:
            _rows, asg, missing = generate_eval_v(
                spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
                backend_factory=backend_factory, arm="old_skill",
                position=position, origin=origin, task_ctx=task_ctx,
                budget=budget, run_dir=run_dir,
            )
            old_eval_asg[int(origin)] = asg
            old_eval_missing[int(origin)] = missing
        if old_model is not None:
            score_units(
                spec=spec, bundle=bundle, model=old_model,
                assignment_by_origin=old_eval_asg,
                missing_by_origin=old_eval_missing,
                units=spec.eval_units, arm="old_skill", run_dir=run_dir,
                write_scores=False,
            )
            old_payload = old_model.to_dict()
        else:
            old_payload = None
        result["arms"]["old_skill"] = {
            "train_missing": old_missing,
            "model": old_payload,
            "n_train_decisions": len(old_rows),
            "eval_frozen_unscored": True,
            "why_unscored": old_why,
        }

        exposure: dict[str, Any] | None = None
        if (
            boundary.get("treatment") == "UPDATE_TREATMENT"
            and boundary.get("candidate_snapshot") is not None
        ):
            exposure = dual_role_exposure(
                spec=spec, bundle=bundle, parent=snapshot,
                candidate=boundary["candidate_snapshot"], run_dir=run_dir,
            )
            result["exposure_check"] = exposure
        if (
            boundary.get("treatment") == "UPDATE_TREATMENT"
            and boundary.get("candidate_snapshot") is not None
            and (exposure or {}).get("outcome") == "EXPOSED"
        ):
            cand = boundary["candidate_snapshot"]
            new_rows, new_asg, new_missing = generate_train_w(
                spec=spec, bundle=bundle, snapshot=cand, machinery=machinery,
                backend_factory=backend_factory, arm="new_skill",
                task_ctx=task_ctx, budget=budget, run_dir=run_dir,
            )
            new_model, new_why = _safe_freeze(
                spec=spec, bundle=bundle, assignment=new_asg, missing=new_missing,
                budget=budget, arm="new_skill", run_dir=run_dir,
            )
            new_eval_asg: dict[int, dict[str, Any] | None] = {}
            new_eval_missing: dict[int, list[str]] = {}
            for position, origin in spec.eval_units:
                _rows, asg, missing = generate_eval_v(
                    spec=spec, bundle=bundle, snapshot=cand, machinery=machinery,
                    backend_factory=backend_factory, arm="new_skill",
                    position=position, origin=origin, task_ctx=task_ctx,
                    budget=budget, run_dir=run_dir,
                )
                new_eval_asg[int(origin)] = asg
                new_eval_missing[int(origin)] = missing
            if new_model is not None:
                score_units(
                    spec=spec, bundle=bundle, model=new_model,
                    assignment_by_origin=new_eval_asg,
                    missing_by_origin=new_eval_missing,
                    units=spec.eval_units, arm="new_skill", run_dir=run_dir,
                    write_scores=False,
                )
                new_payload = new_model.to_dict()
            else:
                new_payload = None
            result["arms"]["new_skill"] = {
                "train_missing": new_missing,
                "model": new_payload,
                "n_train_decisions": len(new_rows),
                "treatment": "UPDATE_TREATMENT",
                "eval_frozen_unscored": True,
                "why_unscored": new_why,
            }
        elif (exposure or {}).get("outcome") == "COMPILED_BUT_NEVER_EXPOSED":
            result["arms"]["new_skill"] = {
                "treatment": "COMPILED_BUT_NEVER_EXPOSED__UTILITY_UNTESTED",
                "why": (
                    "the compiled candidate changes no rendered Fast view on "
                    "any final training or prediction decision, so the paid "
                    "new-vs-old course would compare identical knowledge; the "
                    "compiled candidate receipt is kept and the arm is not run"
                ),
                "exposure_check": "exposure_check.json",
                "scores": None,
            }
        else:
            result["arms"]["new_skill"] = {
                "treatment": "NO_UPDATE_TREATMENT",
                "why": (
                    "Slow produced no compiled update; new_skill is not a "
                    "second copy of old_skill and is not scored as a treatment"
                ),
                "scores": None,
            }

        static_eval_scores = score_frozen_eval_units(
            spec=spec, bundle=bundle, arm="static", run_dir=run_dir,
        )
        result["arms"]["static"]["scores"] = {
            str(key): value for key, value in static_eval_scores.items()
        }
        if fixed_model is not None:
            fixed_scores = score_frozen_eval_units(
                spec=spec, bundle=bundle, arm="fixed_onestep", run_dir=run_dir,
                static_by_origin=static_eval_scores,
            )
            result["arms"]["fixed_onestep"]["scores"] = {
                str(key): value for key, value in fixed_scores.items()
            }
            result["arms"]["fixed_onestep"]["eval_frozen_unscored"] = False
        if old_model is not None:
            old_scores = score_frozen_eval_units(
                spec=spec, bundle=bundle, arm="old_skill", run_dir=run_dir,
                static_by_origin=static_eval_scores,
            )
            result["arms"]["old_skill"]["scores"] = {
                str(key): value for key, value in old_scores.items()
            }
            result["arms"]["old_skill"]["eval_frozen_unscored"] = False
        if result["arms"]["new_skill"].get("treatment") == "UPDATE_TREATMENT" and result["arms"]["new_skill"].get("model"):
            new_scores = score_frozen_eval_units(
                spec=spec, bundle=bundle, arm="new_skill", run_dir=run_dir,
                static_by_origin=static_eval_scores,
            )
            result["arms"]["new_skill"]["scores"] = {
                str(key): value for key, value in new_scores.items()
            }
            result["arms"]["new_skill"]["eval_frozen_unscored"] = False

        result["status"] = "COMPLETED"
        result["budget"] = budget.to_state()
    except AccountFault as exc:
        result["status"] = "ACCOUNT_OR_PERMISSION_FAULT"
        result["detail"] = str(exc)
        result["exit_code"] = 2
        result["budget"] = budget.to_state()
        budget.persist()
    except PackageCeiling as exc:
        result["status"] = "PACKAGE_CEILING"
        result["detail"] = str(exc)
        result["exit_code"] = 3
        result["budget"] = budget.to_state()
    except Exception as exc:  # noqa: BLE001
        result["status"] = "RUNNER_FAULT"
        result["detail"] = "%s: %s" % (type(exc).__name__, _sanitize(exc))
        result["exit_code"] = 1
        result["budget"] = budget.to_state()

    serialisable = json.loads(json.dumps(result, default=_json_default))
    atomic_write_json(run_dir / "result.json", serialisable)
    return result
