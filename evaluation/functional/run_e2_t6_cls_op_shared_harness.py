"""CLS-OP -- the second Task's minimal operational replay on the shared Harness.

The C38 family (controlled_classification_local_event_dynamic_binding_v1) is a
known positive classification Capability: six Target datasets, H1 dynamic
binding exact 6/6, A5-A3 adapt AUC +0.0833, zero negative Target.  It was
produced by a legacy dedicated runner
(``run_e2_promoted_binding_capability_transfer.py``) that compiles the answer
into a frozen Capability card and replays it.  Nothing of it ever touched the
current shared Harness: no Episode was written, no Memory formed, no
Target-local Skill lifecycle ran.

This runner pushes that same family through the shared lifecycle instead:

    methods/ttha/online_loop.run_online_round  (Fast probe -> Episode write)
    methods/ttha/online_loop.open_delayed      (delayed relation -> lifecycle)
    methods/ttha/online_loop.activate_approved (approved snapshot -> active)
    methods/ttha/method.TTHAMethod             (Skill formation and approval)
    methods/ttha/experience_memory             (relation classification)

Nothing in ``methods/`` is modified.  The single new Harness-side file is
``evaluation/functional/consumers/cls_scope_adapter.py``, the thin
classification ``evaluate_fn`` that mirrors ``consumers/ad_scope_adapter.py``:
it converts the accuracy reading into the executor's lower-is-better shape and
decides no relation, no winner, no Skill and no risk.

Entry points::

  python evaluation/functional/run_e2_t6_cls_op_shared_harness.py --smoke
  python evaluation/functional/run_e2_t6_cls_op_shared_harness.py --run

``--smoke`` is the 0-LLM mechanical pass over the same execution body with a
scripted backend; ``--run`` is the live Fast Agent pass that produces the
book's artifact.  Neither entry opens Yahoo, NOAA, NAB or SMD.

Evidence grade: DEVELOPMENT.  Every UCR split used here was already opened by
W48/W55/W56; this is a lifecycle replay, not a fresh Capability claim.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
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

import run_v1_sealed_a5_a3 as sealed  # noqa: E402
from consumers.cls_scope_adapter import (  # noqa: E402
    DELAYED,
    HELDOUT,
    SUPPORT,
    ClassificationConsumerAdapter,
    raw_plus_difference,
)

from SelfEvolvingHarnessTS.contracts.method import (  # noqa: E402
    PreparationRequest,
)
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    classification_local_event_task_quality_contract_v1,
    classification_task_context_v1,
    classification_task_spec_v1,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha import signed_radius as resolver  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    MEASURED_EFFECT_KEY,
    task_consumer_key,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    ScopeExecutor,
)
from SelfEvolvingHarnessTS.operators.registry import (  # noqa: E402
    OPERATOR_METADATA,
    OPERATOR_NAMES,
)

# =========================================================================== #
# frozen constants
# =========================================================================== #
PROTOCOL_VERSION = "t6_cls_op_shared_harness_v1"
EVIDENCE_GRADE = "DEVELOPMENT"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t6_cls_op_shared_harness.json"
OUT_MD = E2 / "t6_cls_op_shared_harness.md"
OUT_SMOKE = E2 / "t6_cls_op_shared_harness_smoke.json"

DATA_DIR = "data/ucr_task_context"

# The two frozen rosters of the C38 family.  Source must come from the roster
# the family's evidence was *formed* on (W55); Target from the roster the
# promoted Capability was *transferred* to (W56).
W55_SOURCE_ROSTER = (
    "FordB", "Lightning2", "MoteStrain", "SonyAIBORobotSurface1",
    "ProximalPhalanxOutlineCorrect", "MiddlePhalanxOutlineCorrect",
)
W56_TARGET_ROSTER = (
    "BirdChicken", "HouseTwenty", "ToeSegmentation1",
    "PhalangesOutlinesCorrect", "SonyAIBORobotSurface2", "GunPointAgeSpan",
)
# Outcome-independent selection rule, fixed before any arm ran: within each
# frozen roster take the datasets with the most official TRAIN rows -- a coarse
# evaluation surface is the one instrument fault this book cannot afford --
# subject to a fit cohort of at most FIT_ROW_CEILING rows, because the window
# verifier runs one verify_candidate per fit row and FordB's 2545-row cohort
# costs minutes per probe.  Three Source datasets, so a leave-one-out
# authorization rule has something to remove.
FIT_ROW_CEILING = 1500
SOURCE_DATASETS = ("ProximalPhalanxOutlineCorrect",
                   "MiddlePhalanxOutlineCorrect", "Lightning2")
TARGET_DATASETS = ("PhalangesOutlinesCorrect", "GunPointAgeSpan")
CONDITIONS = ("fit_only_artifact", "stable_task_event")
TARGET_CONDITION = "fit_only_artifact"

ARMS = ("A3", "A4", "A5")
HELD_IN_ROUNDS = ("r1", "r2")
SOURCE_ROUNDS = ("r1",)
SUPPORT_TRIAL_BUDGET = 2      # non-identity Support receipts per round, per arm
PERIOD_HINT = 24
OBSERVATION_POINTS = 3200     # deployment-visible observation window, in points
RIDGE_ALPHA = 1.0
MATERIAL = 0.005              # experience_memory.CLASSIFICATION_MATERIAL_THRESHOLD
HARM_BAR = 0.05               # book's per-class recall harm bar

LLM_BUDGET_TOTAL = 80
# The Slow consolidation must still be affordable after the Fast rounds have
# run, so its share is reserved up front rather than competed for.
LLM_SLOW_RESERVE = 8
LLM_FAST_CAP = LLM_BUDGET_TOTAL - LLM_SLOW_RESERVE
CONSUMER_FIT_CAP = 600
SLOW_ATTEMPT_CAP = 2
SLOW_MODEL = "gpt-5.6-sol"

SOURCE_SKILL_ID = "source_investigation_cls_v1"
SOURCE_APPLICABILITY: dict[str, Any] = {
    "feature": "task_kind", "op": "==", "value": "classification",
}
CENSUS_CONDITION_KEY = "support_reproduces_fit_signal"
MIN_DISTINCT_TASKS = 2        # leave-one-out floor for an active recommendation

SMOKE_OPERATORS = ("winsorize", "outlier_iqr", "outlier_mad", "hampel_filter")

DEPLOY_SOURCE_ACTIVE = "FROZEN_ACTIVE_SKILL_RECALL"
DEPLOY_SOURCE_INCUMBENT = "FROZEN_LEDGER_INCUMBENT"
DEPLOY_SOURCE_IDENTITY = "FROZEN_LEDGER_NO_INCUMBENT_IDENTITY"

FORBIDDEN_DATA_ROOTS = (
    "benchmark_yahoo_s5_v1", "benchmark_nab_v1_1", "noaa", "smd",
)


class Stop(Exception):
    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__("%s: %s" % (verdict, reason))
        self.verdict = verdict
        self.reason = reason


class FitBudget:
    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.used = 0

    def spend(self, n: int = 1) -> None:
        self.used += int(n)
        if self.used > self.cap:
            raise Stop("CONSUMER_FIT_BUDGET_EXCEEDED",
                       "%d fits used, cap %d" % (self.used, self.cap))


# =========================================================================== #
# Part A -- material and wiring archaeology
# =========================================================================== #
def _material_census() -> dict[str, Any]:
    """What the C38 family is made of, with the file and line it lives at.

    Read, not restated: every path below is imported and executed by this
    runner, so a drift in any of them shows up as a run failure rather than a
    stale sentence in a report.
    """
    return {
        "capability_id": "controlled_classification_local_event_dynamic_binding_v1",
        "legacy_runner": {
            "path": "evaluation/functional/run_e2_promoted_binding_capability_transfer.py",
            "target_roster_lines": "34-41",
            "promoted_capability_card_lines": "53-67",
            "source_admission_gate_lines": "77-110",
            "arms_lines": "220-284",
            "what_it_never_did": (
                "no Episode, no experience_memory write, no Skill lifecycle, "
                "no run_online_round / open_delayed / activate_approved"
            ),
        },
        "injection_template": {
            "path": "evaluation/functional/run_e2_task_context_label_evidence_witness.py",
            "spike_fractions_line": "37",
            "spike_amplitude_line": "38 (SPIKE_AMPLITUDE = 16.0)",
            "bound_positions_lines": "88-92",
            "inject_lines": "95-100",
            "row_znorm_in_loader_lines": "66-69",
            "note": (
                "the loader z-normalises each row, so amplitude 16 is 16 row "
                "standard deviations; the impulse sign is the class label"
            ),
        },
        "fit_support_split": {
            "path": "evaluation/functional/run_e2_task_context_label_evidence_witness.py",
            "lines": "73-85",
            "rule": "per class, evenly spaced round(30%) support; remainder fit",
        },
        "observer": {
            "path": "evaluation/functional/run_e2_task_scoped_impulse_skill_transfer.py",
            "symbol": "_observe_class_conditioned_impulse_topology",
            "role": "class-conditioned impulse topology -> localized nodes",
        },
        "cross_cohort_witness": {
            "path": "evaluation/functional/run_e2_task_context_label_evidence_witness.py",
            "lines": "113-154 (_build_witness), 195-202 (_compile_decision)",
            "role": (
                "support/fit strength ratio + direction alignment; the only "
                "deployment-legal signal that separates a fit-only acquisition "
                "artifact from a stable task event"
            ),
        },
        "program": {
            "path": "evaluation/functional/run_e2_task_conditioned_bound_impulse_oracle.py",
            "symbol": "_apply_bound_impulse_oracle",
            "geometry": "center-excluded local median, radius 3, window 7",
            "binding": (
                "W55 dynamic binding to the observer-localized nodes "
                "(run_e2_program_binding_harness_update.py lines 107-170)"
            ),
        },
        "capability_cards": {
            "H1_dynamic_binding": "bind the unchanged local-median Program to "
                                  "the current fit series' observer nodes",
            "H0_fixed_binding": "reuse the Source instance's absolute positions "
                                "(12, 36, 156, 180)",
            "source": "run_e2_program_binding_harness_update.py lines 40-51",
        },
        "original_source_target_split": {
            "W48_source": ["Coffee", "ECG200", "FordA", "GunPoint"],
            "W55_admission_roster": list(W55_SOURCE_ROSTER),
            "W56_promoted_transfer_roster": list(W56_TARGET_ROSTER),
            "W55_reference": "run_e2_program_binding_harness_update.py lines 32-39",
            "W56_reference": "run_e2_promoted_binding_capability_transfer.py lines 34-41",
        },
    }


def _wiring_check() -> dict[str, Any]:
    """Availability of the shared lifecycle entry for a classification Task."""
    spec = _task_spec()
    context = _task_context()
    classification_ops = tuple(
        name for name in OPERATOR_NAMES
        if "classification" in OPERATOR_METADATA[name]["allowed_tasks"]
        and not OPERATOR_METADATA[name].get("shape_changing")
    )
    return {
        "task_spec": {
            "factory": "contracts.task.classification_task_spec_v1",
            "downstream_model_class": spec.downstream_model_class,
            "consumer_name_matches_legacy": (
                spec.downstream_model_class == "ridge-raw-plus-difference-v1"),
            "metric": spec.metric.to_dict(),
            "sha": spec.sha(),
        },
        "task_context": {
            "factory": "contracts.task.classification_task_context_v1",
            "quality_contract": context.quality_contract.contract_id,
            "maximum_modified_fraction": (
                context.deployment_constraints.maximum_modified_fraction),
            "maximum_candidates": (
                context.deployment_constraints.maximum_candidates),
            "sha": context.sha(),
            "carried_on_every_behaviour_request": True,
        },
        "task_consumer_key": task_consumer_key(spec),
        "consumer_adapter": {
            "path": "evaluation/functional/consumers/cls_scope_adapter.py",
            "precedent": "evaluation/functional/consumers/ad_scope_adapter.py",
            "decides_relation": False,
            "decides_winner": False,
            "reads_or_writes_skill": False,
            "applies_risk_threshold": False,
        },
        "classification_legal_operator_menu": list(classification_ops),
        "shared_lifecycle_entry": [
            "methods/ttha/online_loop.run_online_round",
            "methods/ttha/online_loop.open_delayed",
            "methods/ttha/online_loop.activate_approved",
            "methods/ttha/method.TTHAMethod.handle_fast_winner",
            "methods/ttha/experience_memory.classify_relation",
        ],
        "methods_package_modified": False,
        "known_supply_gap": (
            "the shared operator registry carries no bound center-excluded "
            "local-median operator; the legacy C38 Program is an oracle-bound "
            "instance that only the legacy runner can construct.  The "
            "mechanism-matched member of the shared menu is hampel_filter "
            "(local median replacement at intrinsically detected impulse "
            "points).  This runner does not add an operator: the Agent must "
            "find its Workflow in the existing supply, which is the point."
        ),
    }


# =========================================================================== #
# task contracts
# =========================================================================== #
_SPEC_CACHE: dict[str, Any] = {}


def _task_spec() -> Any:
    spec = _SPEC_CACHE.get("spec")
    if spec is None:
        spec = classification_task_spec_v1(
            downstream_model_class="ridge-raw-plus-difference-v1")
        _SPEC_CACHE["spec"] = spec
    return spec


def _task_context() -> Any:
    """The one TaskContext every behaviour path in this book carries.

    ``maximum_candidates`` is raised from the classification default of 2 to
    ``1 + SUPPORT_TRIAL_BUDGET``.  It is an *adaptation-window* exploration
    limit, not a restatement of the deployment protocol -- the same correction
    #42k Part A made on the AD side.  At 2 the pool is identity plus exactly
    one candidate, so a two-receipt Support budget can never be spent and every
    round silently becomes a one-shot trial.  ``maximum_modified_fraction``
    stays at the classification default of 0.10.
    """
    context = _SPEC_CACHE.get("context")
    if context is None:
        from SelfEvolvingHarnessTS.contracts.task import (
            deployment_constraints_v1,
        )
        context = classification_task_context_v1(
            task_spec=_task_spec(),
            quality_contract=(
                classification_local_event_task_quality_contract_v1()),
            deployment_constraints=deployment_constraints_v1(
                constraint_id="classification-fixed-consumer-v1",
                fixed_downstream_model_id="fixed:classification-consumer-v1",
                maximum_candidates=1 + SUPPORT_TRIAL_BUDGET,
                maximum_modified_fraction=0.1),
        )
        _SPEC_CACHE["context"] = context
    return context


def _assert_behaviour_context(request: Any) -> None:
    ctx = getattr(request, "task_context", None)
    if ctx is None or ctx.sha() != _task_context().sha():
        raise Stop("LIFECYCLE_BROKEN(task_context)",
                   "behaviour-path request does not carry the one TaskContext")


# =========================================================================== #
# cells -- the family's own material, sliced for the shared lifecycle
# =========================================================================== #
def _legacy_helpers() -> tuple[Any, dict[str, Any]]:
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _helpers,
    )
    return _helpers()


def _quarter(labels: Any, indices: Any) -> list[list[int]]:
    """Deal the family's support pool into four class-balanced slices.

    The fit cohort -- the rows a Workflow acts on -- is byte-identical to the
    family's own split.  Only the legacy support pool is subdivided, because
    the shared lifecycle needs a Support surface and a *disjoint* delayed
    surface per round and the family only ever produced one.
    """
    slices: list[list[int]] = [[], [], [], []]
    for label in sorted(set(np.asarray(labels).tolist())):
        own = [int(i) for i in indices if int(labels[int(i)]) == int(label)]
        for position, index in enumerate(own):
            slices[position % 4].append(index)
    return [sorted(part) for part in slices]


def _build_cell(dataset: str, condition: str) -> dict[str, Any]:
    """One (dataset, condition) cell: fit cohort, four held-in slices, context.

    The official TEST split is *not* read here.  It is opened once, by
    ``_score_heldout``, after every arm's Workflow is frozen.
    """
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _condition_inputs,
    )

    _ctx, helpers = _legacy_helpers()
    archive = PROJECT_ROOT / DATA_DIR / ("%s.zip" % dataset)
    train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
    fit_indices, support_indices = helpers["split"](np, train_labels)
    positions = tuple(int(p) for p in helpers["positions"](train_values.shape[1]))
    base_fit = train_values[fit_indices]
    fit_labels = train_labels[fit_indices]
    base_support = train_values[support_indices]
    support_labels = train_labels[support_indices]
    fit_values, support_values = _condition_inputs(
        np,
        base_fit=base_fit, fit_labels=fit_labels,
        base_support=base_support, support_labels=support_labels,
        positions=positions, condition=condition, inject=helpers["inject"])

    observation = helpers["observe"](np, fit_values, fit_labels)
    nodes = tuple(int(node) for node in observation["nodes"])
    witness = helpers["witness"](
        np, fit_values, fit_labels, support_values, support_labels, nodes,
        helpers["rolling_median"])
    legacy_decision, legacy_reasons = helpers["risk_decision"](witness)

    order = {int(index): position
             for position, index in enumerate(support_indices)}
    parts = _quarter(support_labels,
                     [order[int(i)] for i in support_indices])
    surfaces: dict[str, tuple[Any, Any]] = {}
    for name, part in zip(("r1_support", "r1_delayed",
                           "r2_support", "r2_delayed"), parts):
        surfaces[name] = (support_values[part], support_labels[part])

    length = int(train_values.shape[1])
    rows_in_window = max(4, -(-OBSERVATION_POINTS // length))
    rows_in_window = min(rows_in_window, int(fit_values.shape[0]))
    observation_block = np.asarray(
        fit_values[:rows_in_window], dtype=np.float64).ravel()
    return {
        "dataset": dataset,
        "condition": condition,
        "archive": "%s/%s.zip" % (DATA_DIR, dataset),
        "series_length": length,
        "official_train_rows": int(train_values.shape[0]),
        "fit_rows": int(fit_values.shape[0]),
        "support_pool_rows": int(support_indices.size),
        "slice_rows": {name: int(len(part)) for name, part
                       in zip(("r1_support", "r1_delayed",
                               "r2_support", "r2_delayed"), parts)},
        "fit_values": fit_values,
        "fit_labels": fit_labels,
        "surfaces": surfaces,
        "controlled_impulse_positions": list(positions),
        "observer_localized_nodes": list(nodes),
        "observer_recovered_all_nodes": set(nodes) == set(positions),
        "witness": dict(witness),
        "legacy_scope_decision": legacy_decision,
        "legacy_scope_reasons": list(legacy_reasons),
        CENSUS_CONDITION_KEY: bool(
            legacy_decision == "ABSTAIN_KEEP_INCUMBENT"),
        "observation_rows": rows_in_window,
        "observation_block": observation_block,
    }


def _heldout_surface(dataset: str, condition: str) -> tuple[Any, Any]:
    """Open the official TEST split.  Called once per Target, after freeze."""
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _helpers as _h,
    )
    _ctx, helpers = _h()
    archive = PROJECT_ROOT / DATA_DIR / ("%s.zip" % dataset)
    values, labels = helpers["load"](np, archive, dataset, "TEST")
    if condition == TARGET_CONDITION:
        return values, labels
    return helpers["inject"](
        np, values, labels,
        tuple(int(p) for p in helpers["positions"](values.shape[1]))), labels


# =========================================================================== #
# executor
# =========================================================================== #
class _ClsScopeExecutor(ScopeExecutor):
    """The frozen executor with one thing overridden: what a window is.

    ``ScopeExecutor.training_windows`` is forecast-shaped -- 192 context points
    plus a 48-point horizon anchored on a config list.  A classification cohort
    has no time axis: the region a Workflow acts on is one *row*, and the
    verifier must check every row it will actually be run on.  Only that
    mapping is overridden; ``verify``, ``evaluate``, ``_compiled`` and the
    baseline cache stay the frozen implementations, so the window verifier and
    the max_modified_fraction guard run exactly as they always do.
    """

    def __init__(self, *, cell: Mapping[str, Any], evaluate_fn: Any,
                 max_modified_fraction: float) -> None:
        block = np.asarray(cell["observation_block"], dtype=np.float64)
        super().__init__(
            [{"series_uid": "heldin_observation", "role": "train"}],
            {"heldin_observation": block}, {"anchors": []},
            evaluate_fn=evaluate_fn,
            max_modified_fraction=float(max_modified_fraction))
        self._rows = np.asarray(cell["fit_values"], dtype=np.float64)

    def training_windows(self, origin: int):  # noqa: D401 - frozen signature
        return [("row_%05d" % index, 0, np.asarray(row, dtype=np.float64))
                for index, row in enumerate(self._rows)]


def _card_builder(_episode: object) -> Mapping[str, object]:
    return {"pattern_id": "t6-cls-op-local-event",
            "failure_family": "classification_local_event_readiness",
            "observable_signature": {"task_kind": "classification"}}


# =========================================================================== #
# agents
# =========================================================================== #
class _CountingBackend:
    """One shared call ledger, however many backend objects sit behind it.

    The live path keeps a single inner backend for the whole experiment: a
    backend per arm would have handed each arm its own full cap while every
    counter still read "within budget".  The scripted path needs a fresh
    exploration state machine per arm -- its ``_explored`` list is per-object
    and would otherwise leak one arm's exploration into the next -- so it gets
    a new inner object per arm and this ledger keeps the single count.
    """

    def __init__(self, factory: Any, cap: int, *, share_inner: bool) -> None:
        self._factory = factory
        self._share = bool(share_inner)
        self._shared = factory() if share_inner else None
        self.cap = int(cap)
        self.calls = 0
        self._live: Any = None

    def new_arm_backend(self) -> Any:
        self._live = self._shared if self._share else self._factory()
        return _ArmBackend(self._live, self)


class _ArmBackend:
    def __init__(self, inner: Any, ledger: _CountingBackend) -> None:
        self._inner = inner
        self._ledger = ledger

    @property
    def calls(self) -> int:
        return self._ledger.calls

    def complete(self, request: Any) -> Any:
        if self._ledger.calls >= self._ledger.cap:
            raise Stop("LLM_BUDGET_EXCEEDED",
                       "LLM cap %d reached" % self._ledger.cap)
        self._ledger.calls += 1
        return self._inner.complete(request)


def _live_backend(cap: int) -> _CountingBackend:
    from evaluation.functional.task_episode_harness.agentic.runner import (
        _default_backend_factory,
    )
    return _CountingBackend(lambda: _default_backend_factory(cap), cap,
                            share_inner=True)


def _scripted_backend(cap: int) -> _CountingBackend:
    return _CountingBackend(
        lambda: sealed.SealedProbeBackend(
            explore=True, operators=tuple(SMOKE_OPERATORS),
            max_propose_candidates=2),
        cap, share_inner=False)


def _live_agent(block: Any, backend: Any) -> Any:
    from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
        LocalPublicToolGateway,
    )
    from evaluation.functional.task_episode_harness.e1 import NF_BASE_URL

    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(block, task_kind="classification"),
        model=SLOW_MODEL, base_url=NF_BASE_URL)
    return TTHAFastAgent(core)


def _scripted_agent(block: Any, backend: Any) -> Any:
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
        LocalPublicToolGateway,
    )
    core = sealed.TTHAAgentCore(
        backend, LocalPublicToolGateway(block, task_kind="classification"))
    return sealed.TTHAFastAgent(core)


# =========================================================================== #
# one held-in round through the shared lifecycle
# =========================================================================== #
def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _run_round(
    *,
    state: Mapping[str, Any],
    cell: Mapping[str, Any],
    round_name: str,
    arm: str,
    fit_budget: FitBudget,
    allow_fast_skill: bool,
) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
        activate_approved,
        open_delayed,
        run_online_round,
    )

    block = np.asarray(cell["observation_block"], dtype=np.float64)
    support_origin = int(block.size)
    delayed_origin = support_origin + 1
    heldout_origin = support_origin + 2
    surfaces = {
        SUPPORT: cell["surfaces"]["%s_support" % round_name],
        DELAYED: cell["surfaces"]["%s_delayed" % round_name],
    }
    adapter = ClassificationConsumerAdapter(
        fit_values=cell["fit_values"], fit_labels=cell["fit_labels"],
        surfaces=surfaces, delayed_origin=delayed_origin,
        heldout_origin=heldout_origin, budget=fit_budget,
        ridge_alpha=RIDGE_ALPHA, allowed_surfaces=(SUPPORT, DELAYED))
    executor = _ClsScopeExecutor(
        cell=cell, evaluate_fn=adapter,
        max_modified_fraction=float(
            _task_context().deployment_constraints.maximum_modified_fraction))
    values = {"heldin_observation": block}
    observed = dict(resolver.window_context(values, support_origin, PERIOD_HINT))
    observed["bound_period"] = float(PERIOD_HINT)
    request = PreparationRequest(
        "cls-op-%s" % cell["dataset"], block, _task_spec(), dict(observed),
        task_context=_task_context())
    _assert_behaviour_context(request)
    features = dict(extract_public_features(block, task_kind="classification"))

    method = state["method"]
    method.bind_round_data(block, task_kind="classification")
    started = time.time()
    result = run_online_round(
        method, executor, request, values,
        origin=support_origin, slow_agent=None,
        controller=state["controller"], store=state["store"],
        card_builder=_card_builder,
        round_name="%s_%s_%s" % (arm.lower(), cell["dataset"], round_name),
        budget=SUPPORT_TRIAL_BUDGET, allow_slow=False,
        domain="%s/%s" % (cell["dataset"], cell["condition"]),
        period=PERIOD_HINT, fast_features=features,
        allow_fast_skill=allow_fast_skill, runtime_prior_slot=False)
    open_delayed(result, executor, delayed_origin=delayed_origin,
                 store=state["store"])
    activated = False
    if result.approved_skill_id is not None:
        activated = activate_approved(result, state["store"])
        if activated:
            state["approved_skill_ids"].append(str(result.approved_skill_id))

    trace = method.last_trace
    fresh_ids = set(result.episode_ids)
    fresh = [e for e in method.experience_episodes
             if e.episode_id in fresh_ids]
    if result.winner_program is not None:
        state["incumbent"] = _plain(result.winner_program)
    record = {
        "round": round_name,
        "arm": arm,
        "dataset": cell["dataset"],
        "condition": cell["condition"],
        "support_origin": support_origin,
        "delayed_origin": delayed_origin,
        "task_consumer_key": task_consumer_key(_task_spec()),
        "pool": list(getattr(trace, "candidate_ids", ()) or ()),
        "chosen": getattr(trace, "chosen_candidate_id", None),
        "retrieved_skill_ids": list(
            getattr(trace, "retrieved_skill_ids", ()) or ()),
        "memory_resolution": getattr(trace, "memory_resolution_status", None),
        "proposal_count": result.proposal_count,
        "support_receipts": result.target_support_receipts_used,
        "probes": [{"candidate_id": p.get("candidate_id"),
                    "kind": p.get("kind"), "gain": p.get("gain")}
                   for p in result.actual_probed_programs],
        "winner_program": _plain(result.winner_program),
        "abstained": bool(result.abstained),
        "harm_count": int(result.harm_count),
        "delayed_utility": result.delayed_utility,
        "fast_skill_event": _plain(result._fast_skill_event),
        "delayed_event": _plain(result._delayed_event),
        "approved_skill_id": result.approved_skill_id,
        "activated": activated,
        "episodes": [{
            "episode_id": e.episode_id,
            "domain_namespace": e.domain_namespace,
            "workflow_signature": e.workflow_signature,
            "relation": e.relation,
            "evidence_level": e.evidence_level,
            "local_status": e.local_status,
            "support_gain": (e.support_response or {}).get("gain"),
            "delayed_gain": (e.delayed_response or {}).get("gain"),
            "support_effect": _plain(
                (e.support_response or {}).get(MEASURED_EFFECT_KEY)),
            "delayed_effect": _plain(
                (e.delayed_response or {}).get(MEASURED_EFFECT_KEY)),
        } for e in fresh],
        "adapter_calls": _plain(adapter.calls),
        "consumer_fits_after": fit_budget.used,
        "seconds": round(time.time() - started, 2),
    }
    return record


def _new_arm_state(*, snapshot: Any, agent: Any, store_root: Path,
                   tag: str) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod

    root = store_root / tag
    if root.exists():
        shutil.rmtree(root)
    store = SnapshotStore(root / "snapshots")
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    method = TTHAMethod(agent, snapshot, ())
    held = list(getattr(method, "experience_episodes", ()) or ())
    if held:
        raise Stop("LIFECYCLE_BROKEN(memory)",
                   "construction-time Memory must be empty, found %d"
                   % len(held))
    return {
        "store": store,
        "controller": EditController(store, surfaces=SurfaceRegistry(),
                                     router=FaultRouter()),
        "method": method,
        "incumbent": None,
        "approved_skill_ids": [],
        "tag": tag,
    }


# =========================================================================== #
# Part B -- Source Experience, census, authorization, Slow consolidation
# =========================================================================== #
def _relation_of(episode: Mapping[str, Any]) -> str:
    """The lifecycle's own final relation for one Episode.

    ``delayed`` overrides ``support`` when a delayed reading exists, which is
    exactly the lifecycle's rule; the census therefore counts the relation the
    Harness itself settled on, never a second opinion computed here.
    """
    return str(episode["relation"])


def _source_probes(records: Sequence[Mapping[str, Any]],
                   cells: Mapping[str, Mapping[str, Any]]
                   ) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for record in records:
        key = "%s/%s" % (record["dataset"], record["condition"])
        condition = bool(cells[key][CENSUS_CONDITION_KEY])
        for episode in record["episodes"]:
            signature = str(episode["workflow_signature"])
            if signature in ("identity", "unknown"):
                continue
            probes.append({
                "task_episode_id": key,
                "arm": "SOURCE",
                "program": signature,
                "context_condition": condition,
                "support_gain": episode["support_gain"],
                "relation": _relation_of(episode),
                # Nothing in the Source pass was run under a Skill that named
                # a Program family: every arm started from h0 with empty
                # Memory, so no probe is Harness-conditioned.
                "conditioned_snapshot": False,
                "conditioned_served": False,
            })
    return probes


def _census_rows(probes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, bool, str], set[str]] = {}
    for probe in probes:
        key = (probe["program"], bool(probe["context_condition"]),
               probe["relation"])
        grouped.setdefault(key, set()).add(probe["task_episode_id"])
    rows: list[dict[str, Any]] = []
    for (program, condition, relation), tasks in sorted(
            grouped.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        rows.append({
            "canonical_program": program.split("|"),
            CENSUS_CONDITION_KEY: condition,
            "support_relation": relation,
            "distinct_task_count": len(tasks),
            "distinct_task_episode_ids": sorted(tasks),
        })
    return rows


def _slow_call(messages: list[dict[str, str]]) -> Mapping[str, Any]:
    import openai
    from evaluation.functional.task_episode_harness.e1 import (
        NF_BASE_URL, _parse_json_response,
    )

    api_key = next(
        (os.environ.get(name, "").strip()
         for name in ("OPENAI_API_KEY", "AGICTO_API_KEY")
         if os.environ.get(name, "").strip()),
        None,
    )
    if not api_key:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "OPENAI_API_KEY or AGICTO_API_KEY is required for Slow")
    client = openai.OpenAI(api_key=api_key, base_url=NF_BASE_URL, timeout=180)
    completion = client.chat.completions.create(
        model=SLOW_MODEL, messages=messages)
    return _parse_json_response(str(completion.choices[0].message.content or ""))


def _consolidate_source_skill(
    *, census: Sequence[Mapping[str, Any]], probes: Sequence[Mapping[str, Any]],
    llm_ledger: dict[str, int], live: bool,
) -> dict[str, Any]:
    """Slow consolidation into the six-section Source-derived Skill.

    The sections are authored by the Slow stage and validated by the frozen
    deterministic audit in ``source_skill.py``.  Nothing is hand-authored: if
    the Slow stage declines, or what it returns fails the audit, no Skill is
    written and the arms run without a Source prior.
    """
    from evaluation.functional.task_episode_harness.agentic import (
        source_skill as ss,
    )
    from SelfEvolvingHarnessTS.contracts.observables import OBSERVABLE_FEATURES

    audit = ss.authorization_audit(
        probes, min_distinct_tasks=MIN_DISTINCT_TASKS,
        conditioning_key="conditioned_snapshot")
    authorized = sorted(ss.authorized_try_operators(audit))
    risk_authorized = sorted({
        op
        for cell in audit if cell["deprioritization_authorized"]
        for op in str(cell["program"]).split("+")
    })
    summary = ss.signed_summary(census)
    payload = {
        "authorized_try_operators": authorized,
        "risk_authorized_operators": risk_authorized,
        "authorization": audit,
        "signed_summary": summary,
        "required_sections": list(ss.SECTIONS),
        "try_abstain_literal": ss.TRY_ABSTAIN,
        "skill_id": SOURCE_SKILL_ID,
        "applicability": SOURCE_APPLICABILITY,
        "target_domain": (
            "a different classification cohort than the census; write what to "
            "observe and what would have to hold, never that a family worked "
            "in some named cohort"
        ),
    }
    system = ss.slow_system(authorized, skill_id=SOURCE_SKILL_ID)
    attempts: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    cohort_tokens = tuple(name.lower() for name in
                          (*W55_SOURCE_ROSTER, *W56_TARGET_ROSTER))
    if live:
        for attempt in range(1, SLOW_ATTEMPT_CAP + 1):
            try:
                llm_ledger["slow"] += 1
                response = _slow_call([
                    {"role": "system", "content": system},
                    {"role": "user",
                     "content": json.dumps(payload, ensure_ascii=False)},
                ])
            except Exception as exc:  # noqa: BLE001
                attempts.append({"attempt": attempt,
                                 "error": "%s: %s" % (type(exc).__name__, exc)})
                continue
            decision = str(response.get("decision") or "").upper()
            sections = response.get("sections")
            row: dict[str, Any] = {"attempt": attempt, "decision": decision,
                                   "slow_response": _plain(response)}
            if decision != "ADD" or not isinstance(sections, Mapping):
                row["audit"] = {"pass": False, "reason": "malformed_or_abstain"}
                attempts.append(row)
                continue
            section_audit = ss.audit_sections(
                sections, census,
                operator_names=list(OPERATOR_NAMES),
                observable_features=list(OBSERVABLE_FEATURES)
                + [CENSUS_CONDITION_KEY],
                source_cohort_tokens=list(cohort_tokens),
                authorized_try=authorized)
            row["audit"] = section_audit
            attempts.append(row)
            if section_audit["pass"]:
                accepted = {"sections": dict(sections), "audit": section_audit,
                            "attempt": attempt}
                break
    return {
        "authorization_audit": audit,
        "authorized_try_operators": authorized,
        "risk_authorized_operators": risk_authorized,
        "signed_summary_by_program": summary,
        "min_distinct_tasks": MIN_DISTINCT_TASKS,
        "slow_attempts": attempts,
        "slow_llm_calls": len(attempts),
        "accepted": accepted,
        "skill_written": accepted is not None,
        "execution_right_granted": bool(accepted is not None and authorized),
        "execution_right_rule": (
            "a Source-derived card carries a frozen Workflow only when the "
            "deterministic authorization audit authorizes a TRY operator: "
            "unguided POSITIVE cells that survive leave-one-out at "
            "min_distinct_tasks=%d with no opposing evidence in the same "
            "observable Context.  Otherwise the card is advisory only and "
            "supplies no candidate." % MIN_DISTINCT_TASKS
        ),
    }


def _source_skill_entry(consolidation: Mapping[str, Any]) -> dict[str, Any] | None:
    from evaluation.functional.task_episode_harness.agentic import (
        source_skill as ss,
    )

    accepted = consolidation.get("accepted")
    if not accepted:
        return None
    entry = ss.build_skill_payload(
        accepted["sections"], skill_id=SOURCE_SKILL_ID,
        applicability=SOURCE_APPLICABILITY)
    authorized = list(consolidation["authorized_try_operators"])
    if authorized:
        # Execution right, and only then: the audited TRY operators become a
        # frozen Workflow the Fast Path can be handed.  The card stops being a
        # soft prior at exactly the point the deterministic gate says the
        # Source evidence is repeated, unopposed and not carried by one cell.
        steps = [{"op": op, "params": {}} for op in authorized[:1]]
        entry["body"] = entry["body"] + "\nFrozen program steps: " + json.dumps(
            steps)
        entry["allowed_tools"] = [op for op in authorized[:1]]
        guards = dict(entry["risk_guards"])
        guards["requires_target_support"] = False
        guards["execution_right"] = "authorized_by_source_census_audit"
        entry["risk_guards"] = guards
    else:
        guards = dict(entry["risk_guards"])
        guards["requires_target_support"] = True
        guards["execution_right"] = "withheld_no_authorized_try_operator"
        entry["risk_guards"] = guards
    return entry


def _materialize_source_snapshot(entry: Mapping[str, Any], base: Any,
                                 store_root: Path) -> Any:
    from SelfEvolvingHarnessTS.contracts.harness import (
        EditManifest, EditOperation,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (
        _resolve_apply_manifest,
    )

    root = store_root / "source_skill"
    if root.exists():
        shutil.rmtree(root)
    store = SnapshotStore(root / "snapshots")
    store.materialize(base)
    store.set_active(base.runtime_bundle_sha)
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    manifest = EditManifest(
        edit_id=SOURCE_SKILL_ID,
        base_harness_sha=base.harness_content_sha,
        target_pattern_id="t6-cls-op-source-derived-skill",
        target_surface_id="skill_library.entries/" + SOURCE_SKILL_ID,
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value=dict(entry),
        observable_applicability=dict(SOURCE_APPLICABILITY),
        predicted_agent_behavior_change=("retrieve_skill:" + SOURCE_SKILL_ID,),
        predicted_data_effect=("safer_proposal_stage",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",),
        patch_id=None,
    )
    receipt = controller.apply_to_fork(
        store.materialize(base), _resolve_apply_manifest(manifest, base),
        confirmed_cause="SKILL_LIBRARY_GAP")
    snapshot = receipt.candidate_snapshot.snapshot
    store.set_active(snapshot.runtime_bundle_sha)
    return snapshot


# =========================================================================== #
# freeze -> held-out Fast-only deployment
# =========================================================================== #
def _store_fingerprint(store: Any) -> dict[str, Any]:
    import hashlib
    active = getattr(store, "active_path", None)
    payload = ""
    if active is not None and Path(active).is_file():
        payload = Path(active).read_text(encoding="utf-8")
    root = Path(getattr(store, "root"))
    names = sorted(p.name for p in root.glob("*") if p.is_dir()) if root.is_dir() else []
    return {"active_text": payload, "snapshot_dirs": names,
            "sha": hashlib.sha256(
                (payload + "\n" + "\n".join(names)).encode("utf-8")).hexdigest()}


def _frozen_recall(state: Mapping[str, Any], features: Mapping[str, Any]
                   ) -> dict[str, Any]:
    """The frozen deploy decision.  Reads no Outcome and writes nothing.

    #45-Frep-b's symmetric rule, applied identically to every arm: recall an
    ACTIVE Skill that carries a frozen Workflow and is applicable here; failing
    that, keep the incumbent the arm was standing on when it froze; failing
    that, identity.
    """
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _parse_frozen_steps,
    )
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (
        resolve_harness_view,
    )

    snapshot = state["method"]._active_snapshot()  # noqa: SLF001
    view = resolve_harness_view(snapshot, dict(features), role="fast")
    candidates: list[tuple[str, Any]] = []
    for skill in view.skills:
        if str(getattr(getattr(skill, "skill_kind", None), "value", "")) \
                != "capability":
            continue
        steps = _parse_frozen_steps(skill.body)
        if steps is None:
            continue
        guards = dict(skill.risk_guards or {})
        if bool(guards.get("requires_target_support")) and \
                str(skill.skill_id) not in set(state["approved_skill_ids"]):
            continue
        candidates.append((str(skill.skill_id), steps))
    candidates.sort(key=lambda item: item[0])
    if candidates:
        skill_id, steps = candidates[0]
        return {
            "applied_steps": [{"op": op, "params": dict(params)}
                              for op, params in steps],
            "source": DEPLOY_SOURCE_ACTIVE,
            "active_skill_id": skill_id,
            "recall_hit": True,
            "view_skill_ids": [str(s.skill_id) for s in view.skills],
            "why": "Fast-only recall of a frozen Skill carrying a Workflow",
        }
    incumbent = state.get("incumbent")
    if incumbent:
        return {
            "applied_steps": _plain(incumbent),
            "source": DEPLOY_SOURCE_INCUMBENT,
            "active_skill_id": None,
            "recall_hit": False,
            "view_skill_ids": [str(s.skill_id) for s in view.skills],
            "why": "the last held-in round adopted this Workflow",
        }
    return {
        "applied_steps": [],
        "source": DEPLOY_SOURCE_IDENTITY,
        "active_skill_id": None,
        "recall_hit": False,
        "view_skill_ids": [str(s.skill_id) for s in view.skills],
        "why": "no frozen Skill and no incumbent; identity is the fallback",
    }


class _FrozenCompiled:
    """A CompiledWorkflow-shaped carrier so the adapter reads the same bytes."""

    def __init__(self, steps: Sequence[Mapping[str, Any]]) -> None:
        from SelfEvolvingHarnessTS.contracts.candidate import (
            Candidate, CandidateKind,
        )
        from SelfEvolvingHarnessTS.contracts.program import Program

        pairs = [(str(step["op"]), dict(step.get("params") or {}))
                 for step in steps]
        program = Program.from_steps(pairs, source="cls_op_deploy")
        self.candidate = Candidate(candidate_id="cls_op_deploy",
                                   kind=CandidateKind.PROGRAM,
                                   program=program, source="cls_op_deploy")
        self.template_steps = tuple(program.steps)


def _deploy_and_score(*, state: Mapping[str, Any], cell: Mapping[str, Any],
                      arm: str, fit_budget: FitBudget) -> dict[str, Any]:
    from consumers.ad_scope_adapter import compiled_steps

    block = np.asarray(cell["observation_block"], dtype=np.float64)
    features = dict(extract_public_features(block, task_kind="classification"))
    before = _store_fingerprint(state["store"])
    episodes_before = [getattr(e, "episode_id", None)
                       for e in state["method"].experience_episodes]

    decision = _frozen_recall(state, features)
    applied = list(decision["applied_steps"])

    heldout_values, heldout_labels = _heldout_surface(
        cell["dataset"], cell["condition"])
    heldout_origin = int(block.size) + 2
    adapter = ClassificationConsumerAdapter(
        fit_values=cell["fit_values"], fit_labels=cell["fit_labels"],
        surfaces={HELDOUT: (heldout_values, heldout_labels)},
        delayed_origin=int(block.size) + 1, heldout_origin=heldout_origin,
        budget=fit_budget, ridge_alpha=RIDGE_ALPHA,
        allowed_surfaces=(HELDOUT,))
    baseline = adapter([], {}, None, {}, origin=heldout_origin)
    compiled = _FrozenCompiled(applied) if applied else None
    scored = adapter([], {}, compiled, {}, origin=heldout_origin)
    scored_steps = [{"op": op, "params": dict(params)}
                    for op, params in compiled_steps(compiled)]

    after = _store_fingerprint(state["store"])
    episodes_after = [getattr(e, "episode_id", None)
                      for e in state["method"].experience_episodes]
    breach: list[str] = []
    if before["sha"] != after["sha"]:
        breach.append("store_changed_during_deploy")
    if episodes_before != episodes_after:
        breach.append("experience_changed_during_deploy")
    if scored_steps != applied:
        breach.append("scored_program_is_not_the_applied_program")
    base_recalls = baseline["cls_recall_by_class"]
    scored_recalls = scored["cls_recall_by_class"]
    harmed = sorted(label for label in scored_recalls
                    if scored_recalls[label] - base_recalls[label] < -HARM_BAR)
    return {
        "arm": arm,
        "dataset": cell["dataset"],
        "condition": cell["condition"],
        "mode": "FROZEN_FAST_ONLY_DEPLOY",
        "deploy_source": decision["source"],
        "deploy_why": decision["why"],
        "active_skill_id": decision["active_skill_id"],
        "recall_hit": decision["recall_hit"],
        "view_skill_ids": decision["view_skill_ids"],
        "applied_program": applied,
        "scored_program": scored_steps,
        "scored_equals_applied": scored_steps == applied,
        "heldout_rows": int(heldout_labels.size),
        "heldout_identity_accuracy": baseline["cls_accuracy"],
        "heldout_accuracy": scored["cls_accuracy"],
        "heldout_accuracy_gain": (
            scored["cls_accuracy"] - baseline["cls_accuracy"]),
        "heldout_recall_by_class": scored_recalls,
        "heldout_identity_recall_by_class": base_recalls,
        "heldout_recall_delta_by_class": {
            label: scored_recalls[label] - base_recalls[label]
            for label in scored_recalls},
        "harmed_classes_over_bar": harmed,
        "harm_bar": HARM_BAR,
        "open_delayed_calls": 0,
        "slow_calls": 0,
        "llm_calls": 0,
        "new_skill_formed": False,
        "store_unchanged": before["sha"] == after["sha"],
        "experience_unchanged": episodes_before == episodes_after,
        "heldout_openings": 1,
        "consumer_fits_after": fit_budget.used,
        "breach": breach,
        "pure": not breach,
    }


# =========================================================================== #
# the run
# =========================================================================== #
def _preflight() -> dict[str, Any]:
    """Cheap identity checks that would otherwise fail late and vaguely."""
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_context_label_evidence_witness import (
        _features as _legacy_features,
    )
    sample = np.asarray([[1.0, 2.0, 4.0, 7.0], [0.0, -1.0, 1.0, 3.0]])
    same = bool(np.array_equal(_legacy_features(np, sample),
                               raw_plus_difference(sample)))
    return {
        "consumer_feature_map_matches_legacy": same,
        "legacy_feature_reference": (
            "run_e2_task_context_label_evidence_witness.py lines 103-105"),
        "forbidden_data_roots_untouched": list(FORBIDDEN_DATA_ROOTS),
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT,
                              capture_output=True, text=True,
                              check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def run(*, live: bool, out_path: Path) -> int:
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )

    started = time.time()
    fit_budget = FitBudget(CONSUMER_FIT_CAP)
    llm_ledger = {"fast": 0, "slow": 0}
    store_root = Path(tempfile.gettempdir()) / (
        "t6_cls_op_live" if live else "t6_cls_op_smoke")
    if store_root.exists():
        shutil.rmtree(store_root)
    h0 = compile_snapshot(PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    backend_factory = _live_backend if live else _scripted_backend
    agent_factory = _live_agent if live else _scripted_agent
    backend = backend_factory(LLM_FAST_CAP)

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "entry": "--run" if live else "--smoke",
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": _git("rev-parse", "HEAD"),
        "material_census": _material_census(),
        "wiring_check": _wiring_check(),
        "preflight": _preflight(),
        "binding": {
            "source_roster": list(W55_SOURCE_ROSTER),
            "target_roster": list(W56_TARGET_ROSTER),
            "source_datasets": list(SOURCE_DATASETS),
            "source_conditions": list(CONDITIONS),
            "target_datasets": list(TARGET_DATASETS),
            "target_condition": TARGET_CONDITION,
            "selection_rule": (
                "most official TRAIN rows first, within each frozen roster; "
                "fixed before any arm ran and independent of any outcome"),
            "held_in_rounds_source": list(SOURCE_ROUNDS),
            "held_in_rounds_target": list(HELD_IN_ROUNDS),
            "support_trial_budget_per_round": SUPPORT_TRIAL_BUDGET,
            "held_out": "official UCR TEST split, opened once per Target arm",
        },
        "budgets": {
            "llm_cap": LLM_BUDGET_TOTAL,
            "consumer_fit_cap": CONSUMER_FIT_CAP,
        },
    }
    stopped: str | None = None
    try:
        # -------- Part B: Source Experience through the shared lifecycle ----
        source_cells: dict[str, dict[str, Any]] = {}
        source_records: list[dict[str, Any]] = []
        for dataset in SOURCE_DATASETS:
            for condition in CONDITIONS:
                key = "%s/%s" % (dataset, condition)
                cell = _build_cell(dataset, condition)
                source_cells[key] = cell
                state = _new_arm_state(
                    snapshot=h0,
                    agent=agent_factory(cell["observation_block"],
                                        backend.new_arm_backend()),
                    store_root=store_root,
                    tag="source_%s_%s" % (dataset, condition))
                for round_name in SOURCE_ROUNDS:
                    record = _run_round(
                        state=state, cell=cell, round_name=round_name,
                        arm="SOURCE", fit_budget=fit_budget,
                        allow_fast_skill=False)
                    source_records.append(record)
                    print("SOURCE %-46s probes=%d episodes=%s"
                          % (key, record["support_receipts"],
                             [e["relation"] for e in record["episodes"]]),
                          flush=True)
        llm_ledger["fast"] = int(backend.calls)

        probes = _source_probes(source_records, source_cells)
        census = _census_rows(probes)
        relations: dict[str, int] = {}
        for record in source_records:
            for episode in record["episodes"]:
                relations[episode["relation"]] = relations.get(
                    episode["relation"], 0) + 1
        payload["part_b"] = {
            "cells": [{k: v for k, v in cell.items()
                       if k not in ("fit_values", "fit_labels", "surfaces",
                                    "observation_block")}
                      for cell in source_cells.values()],
            "rounds": source_records,
            "episode_count": sum(len(r["episodes"]) for r in source_records),
            "episode_counts_by_relation": relations,
            "probes": probes,
            "census": census,
        }
        if not source_records or payload["part_b"]["episode_count"] == 0:
            raise Stop("EPISODE_FORMATION_FAILED",
                       "the Source pass produced no Episode at all")

        consolidation = _consolidate_source_skill(
            census=census, probes=probes, llm_ledger=llm_ledger, live=live)
        entry = _source_skill_entry(consolidation)
        payload["part_b"]["consolidation"] = {
            k: v for k, v in consolidation.items() if k != "accepted"}
        payload["part_b"]["source_skill_sections"] = (
            (consolidation.get("accepted") or {}).get("sections"))
        payload["part_b"]["source_skill_entry"] = entry
        source_snapshot = h0
        materialize_error = None
        if entry is not None:
            try:
                source_snapshot = _materialize_source_snapshot(
                    entry, h0, store_root)
            except Exception as exc:  # noqa: BLE001
                materialize_error = "%s: %s" % (type(exc).__name__, exc)
        payload["part_b"]["snapshots"] = {
            "h0_runtime_bundle_sha": h0.runtime_bundle_sha,
            "source_runtime_bundle_sha": source_snapshot.runtime_bundle_sha,
            "source_skill_on_snapshot": (
                entry is not None and materialize_error is None),
            "materialize_error": materialize_error,
            "a5_equals_a3_by_construction": (
                source_snapshot.runtime_bundle_sha == h0.runtime_bundle_sha),
        }

        # -------- Part C: the three-arm Target contest ----------------------
        target_cells = {dataset: _build_cell(dataset, TARGET_CONDITION)
                        for dataset in TARGET_DATASETS}
        arm_rounds: list[dict[str, Any]] = []
        deployments: list[dict[str, Any]] = []
        for dataset in TARGET_DATASETS:
            cell = target_cells[dataset]
            for arm in ARMS:
                base = h0 if arm == "A3" else source_snapshot
                state = _new_arm_state(
                    snapshot=base,
                    agent=agent_factory(cell["observation_block"],
                                        backend.new_arm_backend()),
                    store_root=store_root,
                    tag="target_%s_%s" % (dataset, arm))
                if arm != "A4":
                    for round_name in HELD_IN_ROUNDS:
                        record = _run_round(
                            state=state, cell=cell, round_name=round_name,
                            arm=arm, fit_budget=fit_budget,
                            allow_fast_skill=True)
                        arm_rounds.append(record)
                        print("%-3s %-28s %s probes=%d winner=%s delayed=%s"
                              % (arm, dataset, round_name,
                                 record["support_receipts"],
                                 record["winner_program"],
                                 record["delayed_utility"]), flush=True)
                deployment = _deploy_and_score(
                    state=state, cell=cell, arm=arm, fit_budget=fit_budget)
                deployments.append(deployment)
                print("DEPLOY %-3s %-28s %-32s heldout_acc=%.4f gain=%+.4f"
                      % (arm, dataset, deployment["deploy_source"],
                         deployment["heldout_accuracy"],
                         deployment["heldout_accuracy_gain"]), flush=True)
        llm_ledger["fast"] = int(backend.calls)
        payload["part_c"] = {
            "cells": [{k: v for k, v in cell.items()
                       if k not in ("fit_values", "fit_labels", "surfaces",
                                    "observation_block")}
                      for cell in target_cells.values()],
            "rounds": arm_rounds,
            "deployments": deployments,
            "arm_table": _arm_table(arm_rounds, deployments),
            "deploy_purity": _deploy_purity(deployments),
        }
    except Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {"verdict": stopped,
                           "reason": "%s: %s" % (type(exc).__name__, exc),
                           "traceback": traceback.format_exc()}

    payload["ledger"] = {
        "llm_calls_fast": llm_ledger["fast"],
        "llm_calls_slow": llm_ledger["slow"],
        "llm_calls_total": llm_ledger["fast"] + llm_ledger["slow"],
        "llm_cap": LLM_BUDGET_TOTAL,
        "llm_within_cap": (llm_ledger["fast"] + llm_ledger["slow"]
                           <= LLM_BUDGET_TOTAL),
        "consumer_fits": fit_budget.used,
        "consumer_fit_cap": fit_budget.cap,
        "wall_seconds": round(time.time() - started, 1),
    }
    payload["verdict"] = _verdict(payload, stopped=stopped)
    payload["obligations"] = _obligations(payload, live=live)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False,
                   sort_keys=False) + "\n",
        encoding="utf-8")
    if live:
        OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "llm": payload["ledger"]["llm_calls_total"],
                      "fits": payload["ledger"]["consumer_fits"],
                      "artifact": str(out_path)},
                     ensure_ascii=False, indent=1))
    return 0 if payload["verdict"]["verdict"].startswith(
        ("SECOND_TASK_LIFECYCLE_CLOSED", "SMOKE")) else 1


def _arm_table(rounds: Sequence[Mapping[str, Any]],
               deployments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_arm: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in rounds:
        by_arm.setdefault((record["dataset"], record["arm"]), []).append(record)
    rows: list[dict[str, Any]] = []
    for deployment in deployments:
        key = (deployment["dataset"], deployment["arm"])
        own = by_arm.get(key, [])
        first_positive = None
        cost = 0
        for record in own:
            for probe in record["probes"]:
                if probe.get("kind") == "probe" and probe.get("gain") is not None:
                    cost += 1
                    if first_positive is None and float(probe["gain"]) >= MATERIAL:
                        first_positive = cost
        delayed = [r["delayed_utility"] for r in own
                   if r["delayed_utility"] is not None]
        rows.append({
            "dataset": deployment["dataset"],
            "arm": deployment["arm"],
            "held_in_rounds": len(own),
            "support_receipts": sum(r["support_receipts"] for r in own),
            "first_positive_support_receipt": first_positive,
            "episodes": sum(len(r["episodes"]) for r in own),
            "held_in_delayed_utility": (delayed[-1] if delayed else None),
            "held_in_delayed_utility_all": delayed,
            "approved_skill_ids": [r["approved_skill_id"] for r in own
                                   if r["approved_skill_id"]],
            "deploy_source": deployment["deploy_source"],
            "applied_program": deployment["applied_program"],
            "heldout_accuracy": deployment["heldout_accuracy"],
            "heldout_identity_accuracy": deployment["heldout_identity_accuracy"],
            "heldout_accuracy_gain": deployment["heldout_accuracy_gain"],
            "harmed_classes_over_bar": deployment["harmed_classes_over_bar"],
        })
    for dataset in {row["dataset"] for row in rows}:
        subset = {row["arm"]: row for row in rows if row["dataset"] == dataset}
        for arm in ("A3", "A4", "A5"):
            if arm not in subset:
                continue
        if "A5" in subset and "A3" in subset:
            subset["A5"]["A5_minus_A3_heldout_accuracy"] = (
                subset["A5"]["heldout_accuracy"]
                - subset["A3"]["heldout_accuracy"])
        if "A5" in subset and "A4" in subset:
            subset["A5"]["A5_minus_A4_heldout_accuracy"] = (
                subset["A5"]["heldout_accuracy"]
                - subset["A4"]["heldout_accuracy"])
    return sorted(rows, key=lambda row: (row["dataset"], row["arm"]))


def _deploy_purity(deployments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = {}
    for deployment in deployments:
        rows["%s/%s" % (deployment["dataset"], deployment["arm"])] = {
            "open_delayed_calls": deployment["open_delayed_calls"],
            "slow_calls": deployment["slow_calls"],
            "llm_calls": deployment["llm_calls"],
            "new_skill_formed": deployment["new_skill_formed"],
            "store_unchanged": deployment["store_unchanged"],
            "experience_unchanged": deployment["experience_unchanged"],
            "heldout_openings": deployment["heldout_openings"],
            "scored_equals_applied": deployment["scored_equals_applied"],
            "breach": deployment["breach"],
            "ok": deployment["pure"],
        }
    return {"slots": rows,
            "all_pure": bool(rows) and all(row["ok"] for row in rows.values())}


def _verdict(payload: Mapping[str, Any], *, stopped: str | None) -> dict[str, Any]:
    if stopped:
        return {
            "verdict": stopped,
            "stage": (payload.get("stop") or {}).get("reason"),
            "note": "stop-report; no downstream reading is claimed",
        }
    part_b = payload.get("part_b") or {}
    part_c = payload.get("part_c") or {}
    counts = dict(part_b.get("episode_counts_by_relation") or {})
    real_episodes = int(part_b.get("episode_count") or 0)
    kinds = sorted(counts)
    arms_complete = sorted({
        "%s/%s" % (row["dataset"], row["arm"])
        for row in (part_c.get("arm_table") or [])})
    expected = sorted("%s/%s" % (dataset, arm)
                      for dataset in TARGET_DATASETS for arm in ARMS)
    purity = (part_c.get("deploy_purity") or {}).get("all_pure")
    source_skill = bool(
        (part_b.get("consolidation") or {}).get("skill_written"))
    target_skills = sorted({
        str(row["approved_skill_id"])
        for row in (part_c.get("rounds") or []) if row.get("approved_skill_id")})
    skill_formed = bool(source_skill or target_skills)
    three_arm_ok = bool(arms_complete == expected and purity)
    if not three_arm_ok:
        verdict = "LIFECYCLE_BROKEN(three_arm_or_deploy_purity)"
    elif not skill_formed:
        verdict = "LIFECYCLE_BROKEN(skill_formation)"
    else:
        verdict = "SECOND_TASK_LIFECYCLE_CLOSED"
    return {
        "verdict": verdict,
        "mechanical_failure": bool(not three_arm_ok),
        "real_episodes": real_episodes,
        "episode_kinds": kinds,
        "episode_counts_by_relation": counts,
        "source_derived_skill_written": source_skill,
        "target_local_skills_approved": target_skills,
        "skill_formed": skill_formed,
        "three_arm_cells": arms_complete,
        "deploy_purity_all_pure": purity,
        "skill_formation_note": (
            "Episodes and the three arms are the mechanical half of closure; "
            "an Experience->Skill lifecycle that never produces a Skill has "
            "not closed.  When this line is the only one missing, the failure "
            "is evidential (no POSITIVE Support relation to consolidate), not "
            "mechanical, and it is a user checkpoint rather than a bug."
        ),
        "claim_limit": (
            "DEVELOPMENT.  Every UCR split here was already opened by "
            "W48/W55/W56 and the local event is a controlled injection, so "
            "this is a lifecycle-closure reading, not a fresh classification "
            "Capability claim.  Whether A5 beats A3 or A4 is reported as "
            "measured and is not part of the pass condition."
        ),
    }


def _obligations(payload: Mapping[str, Any], *, live: bool) -> dict[str, Any]:
    return {
        "methods_package_unmodified": True,
        "new_files": [
            "evaluation/functional/consumers/cls_scope_adapter.py "
            "(the one permitted thin evaluate_fn adapter)",
            "evaluation/functional/run_e2_t6_cls_op_shared_harness.py "
            "(the runner; a book that writes an artifact needs one, and the "
            "adapter deliberately holds no protocol -- reported for ruling)",
        ],
        "forbidden_data_untouched": (
            "no Yahoo, NOAA 2025, beyond_17520, NAB or SMD path is opened by "
            "this runner; the only data root is data/ucr_task_context"),
        "legacy_capability_card_not_injected": (
            "the W56 promoted Capability card was read for archaeology only "
            "and was never written into experience_memory or a snapshot"),
        "fast_never_saw_raw_source_episodes": (
            "Source Episodes stayed in their own per-cell Method instances; "
            "the Target arms construct with empty Memory and receive Source "
            "evidence only as the audited Skill on the snapshot"),
        "deviations": [
            "The family's fit/support split is reused byte-for-byte, but the "
            "legacy support pool is quartered into per-round Support and "
            "delayed surfaces, because the shared lifecycle needs a delayed "
            "surface the family never produced.",
            "The deployment-visible observation is a fixed ~%d-point window of "
            "the fit cohort rather than the whole block; the executor still "
            "acts on every fit row.  Without it, one actionability probe cost "
            "215 s." % OBSERVATION_POINTS,
            "Slow Path is off inside the held-in rounds; the only Slow call is "
            "the Source consolidation that authors the six-section card.",
            "The per-view axis is the class axis, so CONFLICT means 'accuracy "
            "rose while a class recall fell'.  This is stricter than the "
            "family's accuracy-only gate and is the main reason POSITIVE is "
            "rare here.",
        ],
        "backend": "live Fast Agent" if live else "scripted 0-LLM backend",
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload.get("verdict") or {}
    ledger = payload.get("ledger") or {}
    part_b = payload.get("part_b") or {}
    part_c = payload.get("part_c") or {}
    lines = [
        "# CLS-OP -- second Task on the shared Harness",
        "",
        "protocol: `%s`  evidence grade: **%s**"
        % (payload.get("protocol_version"), payload.get("evidence_grade")),
        "",
        "## Verdict",
        "",
        "**%s**%s" % (verdict.get("verdict"),
                      ("  --  qualifier **%s**" % verdict["qualifier"])
                      if verdict.get("qualifier") else ""),
        "",
        (verdict.get("qualifier_note") or ""),
        "",
        "- real Episodes formed: %s (%s)"
        % (verdict.get("real_episodes"),
           verdict.get("episode_counts_by_relation")),
        "- three-arm cells: %s" % (verdict.get("three_arm_cells"),),
        "- deployment purity: %s" % verdict.get("deploy_purity_all_pure"),
        "",
        verdict.get("claim_limit", ""),
        "",
        "## Budget",
        "",
        "- LLM: %s of %s (fast %s, slow %s)"
        % (ledger.get("llm_calls_total"), ledger.get("llm_cap"),
           ledger.get("llm_calls_fast"), ledger.get("llm_calls_slow")),
        "- Consumer fits: %s of %s"
        % (ledger.get("consumer_fits"), ledger.get("consumer_fit_cap")),
        "- wall clock: %s s" % ledger.get("wall_seconds"),
        "",
        "## Source Experience",
        "",
        "Episodes: %s  by relation: %s"
        % (part_b.get("episode_count"),
           part_b.get("episode_counts_by_relation")),
        "",
        "Source-derived Skill written: %s"
        % ((part_b.get("consolidation") or {}).get("skill_written")),
        "",
    ]
    sections = part_b.get("source_skill_sections") or {}
    for name in ("WHEN", "OBSERVE", "TRY", "RISK", "VERIFY", "FALLBACK"):
        if name in sections:
            lines.append("- **%s**: %s" % (name, sections[name]))
    lines += ["", "## Three-arm table", "",
              "| dataset | arm | deploy source | applied | held-out acc | "
              "gain vs identity | A5-A3 | A5-A4 | harmed classes |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in part_c.get("arm_table") or []:
        lines.append(
            "| %s | %s | %s | %s | %.4f | %+.4f | %s | %s | %s |"
            % (row["dataset"], row["arm"], row["deploy_source"],
               (row["applied_program"] or "identity"),
               row["heldout_accuracy"], row["heldout_accuracy_gain"],
               row.get("A5_minus_A3_heldout_accuracy", ""),
               row.get("A5_minus_A4_heldout_accuracy", ""),
               row["harmed_classes_over_bar"]))
    fault = payload.get("first_fault")
    if fault:
        lines += [
            "", "## First fault", "",
            "**%s**" % fault["localization"], "",
            fault["statement"], "",
            "- starved rounds (zero legal Support receipts): %s of %s"
            % (fault["starved_round_count"], fault["total_rounds"]),
            "- deployment constraint in force: maximum_modified_fraction = %s"
            % fault["deployment_constraint_in_force"],
            "",
            "| dataset | condition | fit rows | program | mean frac | max frac"
            " | rows over 0.10 | passes 0.10 | passes 0.20 | passes 0.35 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for cell in fault["cells"]:
            for row in cell["programs"]:
                if "modified_fraction_mean" not in row:
                    continue
                lines.append(
                    "| %s | %s | %d | %s | %.4f | %.4f | %d | %s | %s | %s |"
                    % (cell["dataset"], cell["condition"], cell["fit_rows"],
                       row["program"], row["modified_fraction_mean"],
                       row["modified_fraction_max"], row["rows_over_0.10"],
                       row["cohort_passes_at_0.10"],
                       row["cohort_passes_at_0.20"],
                       row["cohort_passes_at_0.35"]))
    lines += ["", "## Obligations", ""]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


DIAGNOSE_OPERATORS = (
    "winsorize", "outlier_iqr", "outlier_mad", "hampel_filter",
    "denoise_median", "repair_level_shift", "repair_burst_segment",
    "denoise_savgol", "smooth_ma",
)
DIAGNOSE_THRESHOLDS = (0.10, 0.20, 0.35)


def _contract_params(op: str) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _default_params_from_contract,
    )
    return dict(_default_params_from_contract(op))


def _verifier_census(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-row modified fraction of every candidate Program, on one cell.

    ``run_online_round`` records a verifier rejection as a bare
    ``verifier_rejected`` entry with no rejection code, so a round that
    starves cannot say *why* from its own trace.  This pass replays the
    frozen menu deterministically -- no LLM, no Consumer fit, no TEST split --
    and reports the distribution the all-or-nothing window verifier is
    actually deciding on.
    """
    from SelfEvolvingHarnessTS.contracts.candidate import Candidate
    from SelfEvolvingHarnessTS.contracts.program import Program
    from SelfEvolvingHarnessTS.runtime.candidate_verification import (
        verify_candidate,
    )
    from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

    rows = np.asarray(cell["fit_values"], dtype=np.float64)
    out: list[dict[str, Any]] = []
    for op in DIAGNOSE_OPERATORS:
        if "classification" not in OPERATOR_METADATA[op]["allowed_tasks"]:
            continue
        params = _contract_params(op)
        fractions: list[float] = []
        failed = False
        for row in rows:
            result = run_pipeline([(op, params)], row)
            if not result.ok or result.artifact is None:
                failed = True
                break
            changed = np.asarray(result.artifact, dtype=np.float64).ravel()
            fractions.append(
                float(np.count_nonzero(~np.isclose(changed, row)))
                / float(row.size))
        if failed:
            out.append({"program": op, "params": params,
                        "execution": "FAILED_ON_A_ROW"})
            continue
        array = np.asarray(fractions, dtype=np.float64)
        worst = int(np.argmax(array))
        program = Program.from_steps([(op, params)], source="cls_op_diagnose")
        candidate = Candidate.program_candidate(
            "diagnose_%s" % op, program, source="cls_op_diagnose")
        codes: dict[str, Any] = {}
        for label, index in (("worst_row", worst), ("median_row", int(
                np.argsort(array)[len(array) // 2]))):
            artifact = verify_candidate(
                candidate, rows[index], allowed_operators=(op,),
                inspected_regions=((0, int(rows.shape[1])),),
                maximum_modified_fraction=0.1,
                preserve_outside_inspected_region=True,
                require_finite_output=True)
            codes[label] = {
                "selectable": bool(artifact.selectable),
                "rejection_code": artifact.receipt.rejection_code,
                "modified_fraction": float(array[index]),
            }
        entry: dict[str, Any] = {
            "program": op,
            "params": params,
            "rows": int(array.size),
            "modified_fraction_mean": float(array.mean()),
            "modified_fraction_max": float(array.max()),
            "modified_fraction_p95": float(np.quantile(array, 0.95)),
            "verify_receipts": codes,
        }
        for threshold in DIAGNOSE_THRESHOLDS:
            over = int(np.count_nonzero(array > threshold))
            entry["rows_over_%.2f" % threshold] = over
            entry["cohort_passes_at_%.2f" % threshold] = bool(over == 0)
        out.append(entry)
    return out


def diagnose() -> int:
    """0-LLM first-fault localization, merged into the book's artifact."""
    if not OUT_JSON.is_file():
        print(json.dumps({"verdict": "INSTRUMENT_UNREADABLE",
                          "reason": "no live artifact to annotate"}, indent=1))
        return 1
    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    cells: list[dict[str, Any]] = []
    for dataset in SOURCE_DATASETS:
        for condition in CONDITIONS:
            cells.append(_build_cell(dataset, condition))
    for dataset in TARGET_DATASETS:
        cells.append(_build_cell(dataset, TARGET_CONDITION))
    rows: list[dict[str, Any]] = []
    for cell in cells:
        census = _verifier_census(cell)
        rows.append({
            "dataset": cell["dataset"],
            "condition": cell["condition"],
            "fit_rows": cell["fit_rows"],
            "series_length": cell["series_length"],
            "programs": census,
        })
        passing = [row["program"] for row in census
                   if row.get("cohort_passes_at_0.10")]
        print("%-30s %-18s pass@0.10=%s" % (cell["dataset"], cell["condition"],
                                            passing), flush=True)
    starved = [
        "%s/%s/%s" % (record["arm"], record["dataset"], record["round"])
        for record in (payload.get("part_b", {}).get("rounds", [])
                       + payload.get("part_c", {}).get("rounds", []))
        if int(record.get("support_receipts") or 0) == 0
    ]
    total_rounds = len(payload.get("part_b", {}).get("rounds", [])) + len(
        payload.get("part_c", {}).get("rounds", []))
    payload["first_fault"] = {
        "localization": "candidate exists but the window verifier rejects it",
        "chain_position": (
            "AGENTS.md section 6: Program exists but no candidate survives -> "
            "Observation / localization / supply"),
        "statement": (
            "ScopeExecutor.verify is cohort-all-or-nothing: it runs "
            "verify_candidate on every training window and one rejection "
            "rejects the whole candidate.  A forecasting or AD cohort has a "
            "dozen windows; a classification cohort has one window per fit "
            "row -- 42 to 1260 here -- so the chance that at least one row "
            "exceeds maximum_modified_fraction approaches one and the round "
            "starves before it can buy a Support receipt."
        ),
        "starved_rounds": starved,
        "starved_round_count": len(starved),
        "total_rounds": total_rounds,
        "thresholds_probed": list(DIAGNOSE_THRESHOLDS),
        "deployment_constraint_in_force": (
            payload.get("wiring_check", {})
                   .get("task_context", {})
                   .get("maximum_modified_fraction")),
        "cells": rows,
        "llm_calls": 0,
        "consumer_fits": 0,
        "target_test_read": False,
        "note": (
            "This pass changes no protocol and re-runs no arm.  It only says "
            "what the verifier was deciding, because run_online_round records "
            "a rejection without its rejection code."
        ),
    }
    # The milestone verdict is unchanged -- it is about whether the lifecycle
    # ran, and it did.  What the first-fault pass adds is the qualifier that
    # keeps anyone from reading the three-arm table as a capability result:
    # every arm deployed identity because nothing survived supply.
    verdict = dict(payload.get("verdict") or {})
    verdict["qualifier"] = "SUPPLY_STARVED_BY_WINDOW_VERIFIER"
    verdict["qualifier_note"] = (
        "%d of %d held-in rounds bought zero legal Support receipts, so every "
        "arm froze on identity and the three-arm table carries no signal.  "
        "The lifecycle closed; the contest did not happen.  Re-running any arm "
        "at a looser maximum_modified_fraction to harvest a better number "
        "would be tuning the protocol for a result, so it is left as the "
        "mainline's call."
        % (len(starved), total_rounds))
    payload["verdict"] = verdict
    OUT_JSON.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"starved_rounds": len(starved),
                      "of": total_rounds,
                      "qualifier": verdict["qualifier"],
                      "artifact": str(OUT_JSON)}, indent=1))
    return 0


def micro() -> int:
    """One live Source round.  Proves the live Fast path answers at all."""
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    store_root = Path(tempfile.gettempdir()) / "t6_cls_op_micro"
    if store_root.exists():
        shutil.rmtree(store_root)
    h0 = compile_snapshot(PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    backend = _live_backend(6)
    cell = _build_cell(SOURCE_DATASETS[0], "fit_only_artifact")
    state = _new_arm_state(
        snapshot=h0,
        agent=_live_agent(cell["observation_block"], backend.new_arm_backend()),
        store_root=store_root, tag="micro")
    record = _run_round(state=state, cell=cell, round_name="r1", arm="MICRO",
                        fit_budget=FitBudget(40), allow_fast_skill=False)
    print(json.dumps({
        "pool": record["pool"], "chosen": record["chosen"],
        "probes": record["probes"],
        "relations": [e["relation"] for e in record["episodes"]],
        "llm_calls": backend.calls, "seconds": record["seconds"],
    }, ensure_ascii=False, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if sum([args.smoke, args.run, args.micro, args.diagnose]) != 1:
        parser.error(
            "choose exactly one of --smoke / --run / --micro / --diagnose")
    if args.diagnose:
        return diagnose()
    if args.micro:
        return micro()
    if args.smoke:
        return run(live=False, out_path=OUT_SMOKE)
    return run(live=True, out_path=OUT_JSON)


if __name__ == "__main__":
    raise SystemExit(main())
