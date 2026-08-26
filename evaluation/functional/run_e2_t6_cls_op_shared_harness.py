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
  python evaluation/functional/run_e2_t6_cls_op_shared_harness.py --r2-replay-a5
  python evaluation/functional/run_e2_t6_cls_op_shared_harness.py --conf-dev-run
  python evaluation/functional/run_e2_t6_cls_op_shared_harness.py --dev-wine-precheck
  python evaluation/functional/run_e2_t6_cls_op_shared_harness.py --dev-wine-run

``--smoke`` is the 0-LLM mechanical pass over the same execution body with a
scripted backend; ``--run`` is the live Fast Agent pass that produces the
book's artifact.  ``--r2-replay-a5`` is the C40 A5-only mechanism replay on
the already-exposed GunPointAgeSpan ledger after the Fast-visibility fix.
Neither entry opens Yahoo, NOAA, NAB or SMD.

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
# Injection template versions.  v1 is the C38 single-point impulse used by
# every existing entry.  v2 is length-proportional and is used only by the
# Wine precheck / Wine development entries in this book.
INJECTION_TEMPLATE_V1 = "v1"
INJECTION_TEMPLATE_V2 = "v2"
# GunPoint / GunPointAgeSpan official TRAIN length -- the positive-control
# row length from which v2 scales.  Not a free parameter.
GUNPOINT_POSITIVE_CONTROL_LENGTH = 150
# v1 writes one sample at each SPIKE_FRACTIONS position.  Source:
# evaluation/functional/run_e2_task_context_label_evidence_witness.py:95-100
# (_inject assigns template[position] = ±SPIKE_AMPLITUDE; one index).
V1_SPIKE_SEGMENT_LENGTH = 1


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


def _v2_segment_length(series_length: int) -> int:
    """Mechanical v2 width: round(v1_width / 150 * row_length).

    No other degree of freedom.  At length 150 this is identically 1.
    """
    return int(round(
        V1_SPIKE_SEGMENT_LENGTH / float(GUNPOINT_POSITIVE_CONTROL_LENGTH)
        * int(series_length)))


def _inject_v2(np_mod: Any, values: Any, labels: Any,
               positions: tuple[int, ...]) -> Any:
    """v1 ``_inject`` with each spike widened to ``_v2_segment_length``.

    Placement is forward from the v1 position (the only mechanical extension
    of writing ``template[position]``).  At segment length 1 the bytes match
    v1.  Overflow is a hard error, not a clip -- clipping would be a new
    degree of freedom.
    """
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_context_label_evidence_witness import (
        SPIKE_AMPLITUDE,
    )

    length = int(values.shape[1])
    segment = _v2_segment_length(length)
    if segment <= 0:
        raise ValueError("v2 segment length is %d at series length %d"
                         % (segment, length))
    template = np_mod.zeros(length, dtype=np_mod.float64)
    for index, position in enumerate(positions):
        start = int(position)
        stop = start + segment
        if start < 0 or stop > length:
            raise ValueError(
                "v2 segment [%d, %d) overflows series length %d"
                % (start, stop, length))
        template[start:stop] = (
            (1.0 if index % 2 == 0 else -1.0) * float(SPIKE_AMPLITUDE))
    signs = np_mod.where(labels[:, None] == 1, 1.0, -1.0)
    return np_mod.asarray(values, dtype=np_mod.float64) + signs * template


def _build_cell(dataset: str, condition: str,
                data_dir: str | None = None,
                injection_template: str = INJECTION_TEMPLATE_V1) -> dict[str, Any]:
    """One (dataset, condition) cell: fit cohort, four held-in slices, context.

    The official TEST split is *not* read here.  It is opened once, by
    ``_score_heldout``, after every arm's Workflow is frozen.
    ``injection_template`` defaults to v1 so every existing entry
    (``--conf-run`` / ``--conf-dev-run`` / ``--r2-*``) keeps the C38
    single-point impulse.  Only the Wine book entries pass v2.
    """
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _condition_inputs,
    )

    data_dir = data_dir or DATA_DIR
    if injection_template not in (INJECTION_TEMPLATE_V1, INJECTION_TEMPLATE_V2):
        raise Stop("INSTRUMENT_UNREADABLE",
                   "unknown injection_template %r" % injection_template)
    _ctx, helpers = _legacy_helpers()
    archive = PROJECT_ROOT / data_dir / ("%s.zip" % dataset)
    train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
    fit_indices, support_indices = helpers["split"](np, train_labels)
    positions = tuple(int(p) for p in helpers["positions"](train_values.shape[1]))
    base_fit = train_values[fit_indices]
    fit_labels = train_labels[fit_indices]
    base_support = train_values[support_indices]
    support_labels = train_labels[support_indices]
    inject = (_inject_v2 if injection_template == INJECTION_TEMPLATE_V2
              else helpers["inject"])
    fit_values, support_values = _condition_inputs(
        np,
        base_fit=base_fit, fit_labels=fit_labels,
        base_support=base_support, support_labels=support_labels,
        positions=positions, condition=condition, inject=inject)

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
    cell = {
        "dataset": dataset,
        "condition": condition,
        "data_dir": data_dir,
        "archive": "%s/%s.zip" % (data_dir, dataset),
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
    if injection_template == INJECTION_TEMPLATE_V2:
        cell["injection_template"] = INJECTION_TEMPLATE_V2
        cell["injection_segment_length"] = _v2_segment_length(length)
    return cell


def _heldout_surface(dataset: str, condition: str,
                     data_dir: str | None = None) -> tuple[Any, Any]:
    """Open the official TEST split.  Called once per Target, after freeze."""
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _helpers as _h,
    )
    data_dir = data_dir or DATA_DIR
    _ctx, helpers = _h()
    archive = PROJECT_ROOT / data_dir / ("%s.zip" % dataset)
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
                 max_modified_fraction: float,
                 modification_fraction_scope: str = "per_window") -> None:
        block = np.asarray(cell["observation_block"], dtype=np.float64)
        super().__init__(
            [{"series_uid": "heldin_observation", "role": "train"}],
            {"heldin_observation": block}, {"anchors": []},
            evaluate_fn=evaluate_fn,
            max_modified_fraction=float(max_modified_fraction),
            modification_fraction_scope=modification_fraction_scope)
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
    fraction_scope: str = "per_window",
    ledger: Any = None,
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
            _task_context().deployment_constraints.maximum_modified_fraction),
        modification_fraction_scope=fraction_scope)
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
    llm_before = int(getattr(ledger, "calls", 0) or 0)
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
        "modification_fraction_scope": fraction_scope,
        "llm_calls_before": llm_before,
        "llm_calls_after": int(getattr(ledger, "calls", 0) or 0),
        "llm_calls_this_round": (
            int(getattr(ledger, "calls", 0) or 0) - llm_before),
        # every entry in probe_order that the executor actually ran, legal
        # reading or verifier rejection alike -- a rejected candidate still
        # cost a full pipeline pass over every fit row
        "candidate_executions": len(result.actual_probed_programs),
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
        cell["dataset"], cell["condition"],
        data_dir=cell.get("data_dir"))
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


def run(*, live: bool, out_path: Path,
        md_path: Path | None = None,
        fraction_scope: str = "per_window",
        run_id: str | None = None,
        llm_total: int = LLM_BUDGET_TOTAL,
        protocol_version: str = PROTOCOL_VERSION,
        entry_name: str | None = None) -> int:
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )

    started = time.time()
    fit_budget = FitBudget(CONSUMER_FIT_CAP)
    llm_ledger = {"fast": 0, "slow": 0}
    tag = run_id or ("t6_cls_op_live" if live else "t6_cls_op_smoke")
    store_root = Path(tempfile.gettempdir()) / tag
    if store_root.exists():
        shutil.rmtree(store_root)
    h0 = compile_snapshot(PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    backend_factory = _live_backend if live else _scripted_backend
    agent_factory = _live_agent if live else _scripted_agent
    fast_cap = max(1, int(llm_total) - LLM_SLOW_RESERVE)
    backend = backend_factory(fast_cap)

    payload: dict[str, Any] = {
        "protocol_version": protocol_version,
        "run_id": tag,
        "entry": entry_name or ("--run" if live else "--smoke"),
        "modification_fraction_scope": fraction_scope,
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
            "llm_cap": int(llm_total),
            "llm_fast_cap": fast_cap,
            "llm_slow_reserve": LLM_SLOW_RESERVE,
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
                        allow_fast_skill=False,
                        fraction_scope=fraction_scope, ledger=backend)
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
                            allow_fast_skill=True,
                            fraction_scope=fraction_scope, ledger=backend)
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
            "r2_readouts": _r2_readouts(arm_rounds, deployments),
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
        "llm_cap": int(llm_total),
        "llm_within_cap": (llm_ledger["fast"] + llm_ledger["slow"]
                           <= int(llm_total)),
        "consumer_fits": fit_budget.used,
        "consumer_fit_cap": fit_budget.cap,
        "wall_seconds": round(time.time() - started, 1),
    }
    payload["verdict"] = (_r2_three_arm_verdict(payload, stopped=stopped)
                          if fraction_scope == "cohort"
                          else _verdict(payload, stopped=stopped))
    payload["lifecycle_closure"] = _verdict(payload, stopped=stopped)
    payload["obligations"] = _obligations(payload, live=live)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False,
                   sort_keys=False) + "\n",
        encoding="utf-8")
    if md_path is not None:
        md_path.write_text(_markdown(payload), encoding="utf-8")
    elif live:
        OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "llm": payload["ledger"]["llm_calls_total"],
                      "fits": payload["ledger"]["consumer_fits"],
                      "artifact": str(out_path)},
                     ensure_ascii=False, indent=1))
    # A book that reaches a verdict has succeeded, whichever verdict it is.
    # Only a stop-report or an unreadable instrument is a non-zero exit.
    return 0 if payload["verdict"]["verdict"] in (
        "SECOND_TASK_LIFECYCLE_CLOSED", "CLS_TRANSFER_POSITIVE",
        "CLS_LIFECYCLE_OK_NO_ADVANTAGE") else 1


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


def _r2_readouts(rounds: Sequence[Mapping[str, Any]],
                 deployments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The full sol readout set, per (Target, arm), plus the two contrasts.

    "First positive cost" is charged in the two currencies the arms actually
    spend: LLM calls and candidate executions, both counted cumulatively from
    the arm's first round up to and including the round that first formed a
    non-identity Target-local Skill.  A verifier rejection is a candidate
    execution -- it ran the pipeline over every fit row -- so it is charged.
    """
    by_arm: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in rounds:
        by_arm.setdefault((record["dataset"], record["arm"]), []).append(record)

    rows: dict[str, dict[str, Any]] = {}
    for deployment in deployments:
        dataset, arm = deployment["dataset"], deployment["arm"]
        own = by_arm.get((dataset, arm), [])
        llm = 0
        executions = 0
        first_skill: dict[str, Any] | None = None
        abstained_rounds = 0
        abstain_episodes = 0
        agree = 0
        disagree = 0
        delayed_seen: list[float] = []
        for record in own:
            llm += int(record.get("llm_calls_this_round") or 0)
            executions += int(record.get("candidate_executions") or 0)
            if record.get("abstained"):
                abstained_rounds += 1
            for episode in record["episodes"]:
                if str(episode["relation"]) == "ABSTAIN":
                    abstain_episodes += 1
                support = episode.get("support_gain")
                delayed = episode.get("delayed_gain")
                if support is None or delayed is None:
                    continue
                same = (
                    (float(support) >= MATERIAL and float(delayed) >= MATERIAL)
                    or (float(support) <= -MATERIAL
                        and float(delayed) <= -MATERIAL)
                    or (abs(float(support)) < MATERIAL
                        and abs(float(delayed)) < MATERIAL))
                if same:
                    agree += 1
                else:
                    disagree += 1
            if record.get("delayed_utility") is not None:
                delayed_seen.append(float(record["delayed_utility"]))
            if first_skill is None and record.get("approved_skill_id"):
                first_skill = {
                    "skill_id": record["approved_skill_id"],
                    "round": record["round"],
                    "program": _plain(record["winner_program"]),
                    "non_identity": bool(record["winner_program"]),
                    "llm_calls_to_first_skill": llm,
                    "candidate_executions_to_first_skill": executions,
                    "support_receipts_to_first_skill": sum(
                        int(r["support_receipts"]) for r in own[:own.index(record) + 1]),
                }
        recall_delta = deployment["heldout_recall_delta_by_class"]
        rows["%s/%s" % (dataset, arm)] = {
            "dataset": dataset,
            "arm": arm,
            "target_local_skill_formed": first_skill is not None,
            "first_skill": first_skill,
            "llm_calls_total": llm,
            "candidate_executions_total": executions,
            "held_in_delayed_utility": delayed_seen[-1] if delayed_seen else None,
            "held_in_delayed_utility_all": delayed_seen,
            "abstained_rounds": abstained_rounds,
            "abstain_episodes": abstain_episodes,
            "support_delayed_direction_agree": agree,
            "support_delayed_direction_disagree": disagree,
            "heldout_accuracy": deployment["heldout_accuracy"],
            "heldout_identity_accuracy": deployment["heldout_identity_accuracy"],
            "heldout_accuracy_gain": deployment["heldout_accuracy_gain"],
            "heldout_recall_by_class": deployment["heldout_recall_by_class"],
            "heldout_recall_delta_by_class": recall_delta,
            "worst_class_recall_delta": (
                min(recall_delta.values()) if recall_delta else None),
            "worst_class_harm": (
                max(0.0, -min(recall_delta.values())) if recall_delta else None),
            "harmed_classes_over_bar": deployment["harmed_classes_over_bar"],
            "deploy_source": deployment["deploy_source"],
            "applied_program": deployment["applied_program"],
        }

    contrasts: dict[str, Any] = {}
    for dataset in {deployment["dataset"] for deployment in deployments}:
        cells = {arm: rows.get("%s/%s" % (dataset, arm)) for arm in ARMS}
        if not all(cells.values()):
            continue
        a3, a4, a5 = cells["A3"], cells["A4"], cells["A5"]
        a5_a3 = a5["heldout_accuracy"] - a3["heldout_accuracy"]
        a5_a4 = a5["heldout_accuracy"] - a4["heldout_accuracy"]
        cheaper = None
        if a5["target_local_skill_formed"] and a3["target_local_skill_formed"]:
            cheaper = (
                a5["first_skill"]["candidate_executions_to_first_skill"]
                - a3["first_skill"]["candidate_executions_to_first_skill"])
        contrasts[dataset] = {
            "A5_minus_A3_heldout_accuracy": a5_a3,
            "A5_minus_A4_heldout_accuracy": a5_a4,
            "A5_minus_A3_first_skill_executions": cheaper,
            "A5_minus_A3_worst_class_harm": (
                (a5["worst_class_harm"] or 0.0)
                - (a3["worst_class_harm"] or 0.0)),
            "A4_beats_A5": bool(a5_a4 < -MATERIAL),
            "A4_mechanism_note": (
                None if a5_a4 >= -MATERIAL else
                "A4 deploys the Source prior untouched while A5 deployed %s "
                "after held-in feedback; when A4 wins, the held-in signal "
                "either mis-steered adoption (feedback bias) or the Source "
                "card was never recalled into A5's pool (retrieval binding). "
                "A5 recall source was %s."
                % (a5["applied_program"] or "identity", a5["deploy_source"])),
        }
    return {"cells": rows, "contrasts": contrasts,
            "material_threshold": MATERIAL, "harm_bar": HARM_BAR}


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


def _r2_three_arm_verdict(payload: Mapping[str, Any], *,
                          stopped: str | None) -> dict[str, Any]:
    """The CLS-OP-r2 verdict set.  Advantage is judged, not hoped for."""
    if stopped:
        return {"verdict": stopped,
                "stage": (payload.get("stop") or {}).get("reason"),
                "note": "stop-report; no downstream reading is claimed"}
    part_c = payload.get("part_c") or {}
    readouts = part_c.get("r2_readouts") or {}
    cells = readouts.get("cells") or {}
    purity = (part_c.get("deploy_purity") or {}).get("all_pure")
    expected = sorted("%s/%s" % (dataset, arm)
                      for dataset in TARGET_DATASETS for arm in ARMS)
    if sorted(cells) != expected or not purity:
        return {"verdict": "INSTRUMENT_UNREADABLE",
                "reason": "three-arm cells incomplete or deployment impure",
                "cells": sorted(cells), "deploy_purity_all_pure": purity}

    skills = {key: bool(row["target_local_skill_formed"]
                        and (row.get("first_skill") or {}).get("non_identity"))
              for key, row in cells.items()}
    if not any(skills.values()):
        return {
            "verdict": "STILL_STARVED",
            "reason": ("no arm formed a non-identity Target-local Skill after "
                       "the verifier fix"),
            "skill_by_cell": skills,
            "per_round_probes": [
                {"cell": "%s/%s/%s" % (row["arm"], row["dataset"],
                                       row["round"]),
                 "probes": row["probes"], "pool": row["pool"],
                 "chosen": row["chosen"]}
                for row in part_c.get("rounds") or []],
        }

    items: list[dict[str, Any]] = []
    for dataset, contrast in (readouts.get("contrasts") or {}).items():
        a3 = cells["%s/A3" % dataset]
        a5 = cells["%s/A5" % dataset]
        n_heldout = None
        for deployment in part_c.get("deployments") or []:
            if deployment["dataset"] == dataset and deployment["arm"] == "A5":
                n_heldout = int(deployment["heldout_rows"])
        line = max(MATERIAL, 1.0 / max(int(n_heldout or 1), 1))
        gain = float(contrast["A5_minus_A3_heldout_accuracy"])
        items.append({
            "dataset": dataset, "item": "heldout_accuracy",
            "value": gain, "material_line": line,
            "better": gain >= line, "worse": gain <= -line})
        a5_formed = skills["%s/A5" % dataset]
        a3_formed = skills["%s/A3" % dataset]
        if a5_formed and a3_formed:
            delta = int(contrast["A5_minus_A3_first_skill_executions"] or 0)
            items.append({
                "dataset": dataset, "item": "first_skill_candidate_executions",
                "value": delta, "material_line": 1,
                "better": delta < 0, "worse": delta > 0})
        elif a5_formed != a3_formed:
            items.append({
                "dataset": dataset, "item": "target_local_skill_formed",
                "value": "A5" if a5_formed else "A3", "material_line": None,
                "better": a5_formed, "worse": a3_formed})
        harm = float(contrast["A5_minus_A3_worst_class_harm"])
        items.append({
            "dataset": dataset, "item": "worst_class_harm",
            "value": harm, "material_line": MATERIAL,
            "better": harm <= -MATERIAL, "worse": harm >= MATERIAL})

    better = [row for row in items if row["better"]]
    worse = [row for row in items if row["worse"]]
    positive = bool(better and not worse)
    return {
        "verdict": ("CLS_TRANSFER_POSITIVE" if positive
                    else "CLS_LIFECYCLE_OK_NO_ADVANTAGE"),
        "rule": (
            "CLS_TRANSFER_POSITIVE iff a non-identity Target-local Skill "
            "formed and A5 beats A3 materially on at least one of {held-out "
            "accuracy, first-Skill candidate-execution cost, worst-class "
            "harm} with no item materially worse."),
        "non_identity_target_local_skill_formed": True,
        "skill_by_cell": skills,
        "comparison_items": items,
        "better_items": better,
        "worse_items": worse,
        "contrasts": readouts.get("contrasts"),
        "a4_bias_check": {
            dataset: {
                "A4_beats_A5": contrast["A4_beats_A5"],
                "A5_minus_A4_heldout_accuracy": (
                    contrast["A5_minus_A4_heldout_accuracy"]),
                "mechanism_note": contrast["A4_mechanism_note"],
            }
            for dataset, contrast in (readouts.get("contrasts") or {}).items()
        },
        "deploy_purity_all_pure": purity,
        "claim_limit": (
            "DEVELOPMENT.  Controlled impulse injection on UCR splits that "
            "W48/W55/W56 already opened.  A5 > A3 here is a development-grade "
            "reading and needs the frozen confirmation on an unused local UCR "
            "Target before it can be called classification transfer."),
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
    readouts = (part_c.get("r2_readouts") or {})
    if readouts.get("cells"):
        lines += ["", "## Full readout set", "",
                  "| cell | Skill formed | first-Skill LLM | first-Skill "
                  "executions | held-in delayed | held-out acc | vs identity |"
                  " worst class harm | abstained rounds | Support/delayed "
                  "agree:disagree |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for key in sorted(readouts["cells"]):
            row = readouts["cells"][key]
            first = row.get("first_skill") or {}
            lines.append(
                "| %s | %s | %s | %s | %s | %.4f | %+.4f | %.4f | %d | %d:%d |"
                % (key, row["target_local_skill_formed"],
                   first.get("llm_calls_to_first_skill", "-"),
                   first.get("candidate_executions_to_first_skill", "-"),
                   ("%+.4f" % row["held_in_delayed_utility"])
                   if row["held_in_delayed_utility"] is not None else "n/a",
                   row["heldout_accuracy"], row["heldout_accuracy_gain"],
                   row["worst_class_harm"] or 0.0, row["abstained_rounds"],
                   row["support_delayed_direction_agree"],
                   row["support_delayed_direction_disagree"]))
        lines += ["", "### Contrasts", ""]
        for dataset, contrast in (readouts.get("contrasts") or {}).items():
            lines.append(
                "- **%s**: A5-A3 accuracy %+.4f; A5-A4 accuracy %+.4f; "
                "A5-A3 first-Skill executions %s; A4 beats A5: %s"
                % (dataset, contrast["A5_minus_A3_heldout_accuracy"],
                   contrast["A5_minus_A4_heldout_accuracy"],
                   contrast["A5_minus_A3_first_skill_executions"],
                   contrast["A4_beats_A5"]))
            if contrast.get("A4_mechanism_note"):
                lines.append("  - %s" % contrast["A4_mechanism_note"])
    prereg = payload.get("prereg_check")
    if prereg:
        lines += ["", "## Pre-registration, scored", ""]
        for row in prereg["items"]:
            lines.append("- **%s** %s -- %s (%s)"
                         % (row["id"], "HELD" if row["held"] else "FALSIFIED",
                            row["claim"], row.get("observed")))
    mechanism = payload.get("a5_deficit_mechanism")
    if mechanism and mechanism.get("datasets_with_a5_below_a3"):
        lines += ["", "## Why A5 fell below A3", "",
                  "classification: **%s**" % mechanism["classification"], "",
                  str(mechanism.get("reading") or ""), "",
                  "- Source card retrieved in every A5 round: %s"
                  % mechanism["source_card_retrieved_in_every_a5_round"],
                  "- Source card supplied an executable candidate: %s"
                  % mechanism["source_card_supplied_an_executable_candidate"],
                  "- Source card execution right: %s"
                  % mechanism["source_card_execution_right"]]
    lines += ["", "## Obligations", ""]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


# =========================================================================== #
# CLS-OP-r2-prep -- the two 0-LLM gates that decide whether r2 leaves the dock
# =========================================================================== #
R2_OUT_JSON = E2 / "t6_cls_op_r2_prep.json"
R2_OUT_MD = E2 / "t6_cls_op_r2_prep.md"
R2_PROTOCOL_VERSION = "t6_cls_op_r2_prep_v1"
R2_FIT_CAP = 120
R2_ROUND = "r1"

# CLS-OP-r2: the same roster, menu, budget shape, TaskContext and deployment
# purity assertions as C39.  The only differences the book authorises are the
# cohort fraction scope and maximum_candidates=3 (which is what C39 actually
# ran).  A fresh run id, because no C39 store, snapshot or Episode may be
# reused: Source formation, Slow consolidation, the three arms, the freeze and
# the Fast-only deployment are all re-walked.
R2_RUN_JSON = E2 / "t6_cls_op_r2_three_arms.json"
R2_RUN_MD = E2 / "t6_cls_op_r2_three_arms.md"
R2_RUN_PROTOCOL = "t6_cls_op_r2_three_arms_v1"
R2_RUN_ID = "t6_cls_op_r2_run1"
R2_LLM_CAP = 90
# C40 A5-only mechanism replay.  Same r2 protocol (cohort verifier, menu,
# TaskContext, Fast factories, A5 construction).  Isolated artifacts; the
# Source card is the r2 ledger entry, not a new Slow consolidation.
R2_REPLAY_JSON = E2 / "t6_cls_op_r2_a5_replay.json"
R2_REPLAY_MD = E2 / "t6_cls_op_r2_a5_replay.md"
R2_REPLAY_PROTOCOL = "t6_cls_op_r2_a5_replay_v1"
R2_REPLAY_RUN_ID = "t6_cls_op_r2_a5_replay1"
R2_REPLAY_DATASET = "GunPointAgeSpan"
_LEVEL_SHIFT_FAMILY_MARKERS = (
    "level_shift", "level-shift", "level_excursion", "repair_level",
)
_HAMPEL_FAMILY_MARKERS = ("hampel",)


def _r2_menu() -> list[dict[str, Any]]:
    """Every classification-legal non-identity candidate the shared menu has.

    No hand-picking: the filter is the registry's own ``allowed_tasks`` plus
    the shape-preservation requirement the executor imposes, and the params are
    the contract defaults the candidate supply would itself construct.  If the
    honest answer is that this menu has no headroom, the menu has to be the
    whole menu for that answer to mean anything.
    """
    out: list[dict[str, Any]] = []
    for name in OPERATOR_NAMES:
        metadata = OPERATOR_METADATA[name]
        if "classification" not in metadata["allowed_tasks"]:
            continue
        if metadata.get("shape_changing"):
            continue
        out.append({"program": name, "params": _contract_params(name)})
    return out


def _r2_cell_pass(cell: Mapping[str, Any], *,
                  fit_budget: FitBudget) -> dict[str, Any]:
    """Verify the whole menu on one cell, then score whatever survives.

    Two gates, in the order the Harness applies them.  The window verifier runs
    first under the new cohort scope; only candidates that are both legal and
    not byte-identical to identity are worth a Consumer fit, which is also what
    keeps this inside the fit cap.
    """
    block = np.asarray(cell["observation_block"], dtype=np.float64)
    support_origin = int(block.size)
    delayed_origin = support_origin + 1
    heldout_origin = support_origin + 2
    surfaces = {
        SUPPORT: cell["surfaces"]["%s_support" % R2_ROUND],
        DELAYED: cell["surfaces"]["%s_delayed" % R2_ROUND],
    }
    adapter = ClassificationConsumerAdapter(
        fit_values=cell["fit_values"], fit_labels=cell["fit_labels"],
        surfaces=surfaces, delayed_origin=delayed_origin,
        heldout_origin=heldout_origin, budget=fit_budget,
        ridge_alpha=RIDGE_ALPHA, allowed_surfaces=(SUPPORT, DELAYED))
    cap = float(
        _task_context().deployment_constraints.maximum_modified_fraction)
    executor = _ClsScopeExecutor(
        cell=cell, evaluate_fn=adapter, max_modified_fraction=cap,
        modification_fraction_scope="cohort")
    # The same cells under the old scope, so Part B can state the difference
    # instead of asserting it.  Verification only -- this executor is never
    # asked to evaluate, so it costs no Consumer fit.
    old_executor = _ClsScopeExecutor(
        cell=cell, evaluate_fn=None, max_modified_fraction=cap,
        modification_fraction_scope="per_window")

    surface_rows = {
        name: int(np.asarray(labels).size)
        for name, (_values, labels) in surfaces.items()
    }
    material_lines = {
        name: max(MATERIAL, 1.0 / max(rows, 1))
        for name, rows in surface_rows.items()
    }

    rows: list[dict[str, Any]] = []
    exhausted = False
    for entry in _r2_menu():
        steps = ((entry["program"], dict(entry["params"])),)
        verification = executor.verify(steps, support_origin)
        old_verification = old_executor.verify(steps, support_origin)
        no_op = bool(
            verification.checked_windows
            and verification.identity_equivalent_windows
            == verification.checked_windows)
        record: dict[str, Any] = {
            "program": entry["program"],
            "params": entry["params"],
            "verifier_passed": bool(verification.passed),
            "verifier_passed_per_window_scope": bool(old_verification.passed),
            "unblocked_by_the_fix": bool(
                verification.passed and not old_verification.passed),
            "cohort_modified_fraction": verification.cohort_modified_fraction,
            "cohort_modified_points": verification.cohort_modified_points,
            "cohort_total_points": verification.cohort_total_points,
            "windows_over_per_window_cap": (
                verification.windows_over_maximum_fraction),
            "checked_windows": verification.checked_windows,
            "rejection_codes": sorted({
                str(row["rejection_code"])
                for row in verification.rejected_windows}),
            "numeric_no_op": no_op,
            "survives": bool(verification.passed and not no_op),
        }
        if record["survives"] and not exhausted:
            needed = 2 * (1 + len(surfaces))  # baselines are cached per origin
            if fit_budget.used + needed > fit_budget.cap:
                exhausted = True
                record["scored"] = False
                record["skip_reason"] = "R2_FIT_CAP_REACHED"
            else:
                readings: dict[str, Any] = {}
                for name, origin in ((SUPPORT, support_origin),
                                     (DELAYED, delayed_origin)):
                    receipt = executor.evaluate(steps, origin)
                    delta = (float(receipt.gain)
                             if receipt.gain is not None else None)
                    recalls = [float(value) for value in receipt.per_view_gain]
                    readings[name] = {
                        "rows": surface_rows[name],
                        "material_line": material_lines[name],
                        "delta_accuracy": delta,
                        "per_class_recall_delta": recalls,
                        "worst_class_recall_delta": (
                            min(recalls) if recalls else None),
                        "material_positive": bool(
                            delta is not None
                            and delta >= material_lines[name]),
                        "class_guard_pass": bool(
                            recalls and min(recalls) >= -HARM_BAR),
                        "verifier_passed": bool(receipt.verification.passed),
                        "error": receipt.error,
                    }
                record["scored"] = True
                record["readings"] = readings
                record["headroom_here"] = bool(any(
                    row["material_positive"] and row["class_guard_pass"]
                    for row in readings.values()))
        else:
            record.setdefault("scored", False)
        rows.append(record)

    return {
        "dataset": cell["dataset"],
        "condition": cell["condition"],
        "fit_rows": cell["fit_rows"],
        "series_length": cell["series_length"],
        "surface_rows": surface_rows,
        "material_lines": material_lines,
        "maximum_modified_fraction": cap,
        "modification_fraction_scope": "cohort",
        "menu_size": len(rows),
        "survivors": [row["program"] for row in rows if row["survives"]],
        "survivors_under_per_window_scope": [
            row["program"] for row in rows
            if row["verifier_passed_per_window_scope"]
            and not row["numeric_no_op"]],
        "unblocked_by_the_fix": [row["program"] for row in rows
                                 if row["unblocked_by_the_fix"]
                                 and not row["numeric_no_op"]],
        "numeric_no_ops": [row["program"] for row in rows
                           if row["numeric_no_op"]],
        "verifier_rejected": [row["program"] for row in rows
                              if not row["verifier_passed"]],
        "fit_cap_truncated": exhausted,
        "programs": rows,
        "consumer_fits_after": fit_budget.used,
    }


def _r2_verdict(source: Sequence[Mapping[str, Any]],
                target: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for cell in target:
        for row in cell["programs"]:
            for name, reading in (row.get("readings") or {}).items():
                if reading["material_positive"] and reading["class_guard_pass"]:
                    hits.append({
                        "dataset": cell["dataset"], "surface": name,
                        "program": row["program"],
                        "delta_accuracy": reading["delta_accuracy"],
                        "material_line": reading["material_line"],
                        "worst_class_recall_delta": (
                            reading["worst_class_recall_delta"]),
                    })
    truncated = any(cell["fit_cap_truncated"] for cell in (*source, *target))
    any_survivor = any(cell["survivors"] for cell in (*source, *target))
    if hits:
        verdict = "HEADROOM_EXISTS"
    elif truncated:
        verdict = "INSTRUMENT_UNREADABLE"
    else:
        verdict = "NO_MENU_HEADROOM"
    return {
        "verdict": verdict,
        "target_headroom_hits": hits,
        "any_candidate_survives_the_verifier": any_survivor,
        "fit_cap_truncated_any_cell": truncated,
        "rule": (
            "HEADROOM_EXISTS iff at least one shared-menu candidate is "
            "materially positive on at least one Target held-in surface "
            "(delta accuracy >= max(0.005, 1/n) for that surface's own n) "
            "while no per-class recall falls more than %.2f.  Source cells "
            "are reported for context and cannot carry the verdict."
            % HARM_BAR),
        "next": (
            "r2 three-arm launch condition met" if verdict == "HEADROOM_EXISTS"
            else "wire the C38 center-excluded local median as a Typed "
                 "Workflow and re-ask headroom" if verdict == "NO_MENU_HEADROOM"
            else "raise the fit cap or narrow the menu and re-run"),
    }


def r2_prep() -> int:
    """Part A proof + Part B smoke + Part C headroom census.  Zero LLM."""
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (
        FRACTION_SCOPE_COHORT,
        FRACTION_SCOPE_PER_WINDOW,
    )

    started = time.time()
    fit_budget = FitBudget(R2_FIT_CAP)
    source_cells: list[dict[str, Any]] = []
    target_cells: list[dict[str, Any]] = []
    stopped: str | None = None
    payload: dict[str, Any] = {
        "protocol_version": R2_PROTOCOL_VERSION,
        "entry": "--r2-prep",
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": _git("rev-parse", "HEAD"),
        "llm_calls": 0,
        "verifier_fix": _r2_fix_report(),
        "binding": {
            "source_cells": ["%s/%s" % (dataset, "fit_only_artifact")
                             for dataset in SOURCE_DATASETS],
            "target_cells": ["%s/%s" % (dataset, TARGET_CONDITION)
                             for dataset in TARGET_DATASETS],
            "round": R2_ROUND,
            "surfaces": [SUPPORT, DELAYED],
            "scope": FRACTION_SCOPE_COHORT,
            "default_scope_elsewhere": FRACTION_SCOPE_PER_WINDOW,
            "maximum_candidates": 1 + SUPPORT_TRIAL_BUDGET,
            "unchanged_gates": (
                "maximum_modified_fraction stays 0.10; maximum_candidates "
                "stays at the value C39 actually ran (3); selectable "
                "semantics and effect distinctness are untouched"),
        },
        "menu": _r2_menu(),
    }
    try:
        for dataset in SOURCE_DATASETS:
            cell = _build_cell(dataset, "fit_only_artifact")
            row = _r2_cell_pass(cell, fit_budget=fit_budget)
            source_cells.append(row)
            print("SOURCE %-30s survivors=%s" % (dataset, row["survivors"]),
                  flush=True)
        for dataset in TARGET_DATASETS:
            cell = _build_cell(dataset, TARGET_CONDITION)
            row = _r2_cell_pass(cell, fit_budget=fit_budget)
            target_cells.append(row)
            print("TARGET %-30s survivors=%s" % (dataset, row["survivors"]),
                  flush=True)
    except Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {"verdict": stopped,
                           "reason": "%s: %s" % (type(exc).__name__, exc),
                           "traceback": traceback.format_exc()}

    payload["part_b_smoke"] = {
        "question": (
            "after the fix, can a non-identity candidate obtain a legal, "
            "non-no-op Support receipt on the same material?"),
        "no_op_rule": (
            "a candidate is a numeric no-op when every window's prepared "
            "bytes are identity-equivalent to the raw bytes; such a candidate "
            "is excluded from the survivor list and never fitted"),
        "cells": [{
            "cell": "%s/%s" % (cell["dataset"], cell["condition"]),
            "menu_size": cell["menu_size"],
            "survivors": cell["survivors"],
            "survivor_count": len(cell["survivors"]),
            "survivors_under_per_window_scope": (
                cell["survivors_under_per_window_scope"]),
            "unblocked_by_the_fix": cell["unblocked_by_the_fix"],
            "numeric_no_ops": cell["numeric_no_ops"],
            "verifier_rejected": cell["verifier_rejected"],
        } for cell in (*source_cells, *target_cells)],
    }
    payload["part_c_headroom"] = {
        "source_cells": source_cells,
        "target_cells": target_cells,
    }
    payload["ledger"] = {
        "llm_calls": 0,
        "consumer_fits": fit_budget.used,
        "consumer_fit_cap": fit_budget.cap,
        "wall_seconds": round(time.time() - started, 1),
    }
    if stopped:
        payload["verdict"] = {"verdict": stopped,
                              "stage": (payload.get("stop") or {}).get("reason")}
    else:
        payload["verdict"] = _r2_verdict(source_cells, target_cells)
    payload["obligations"] = _r2_obligations()
    R2_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    R2_OUT_JSON.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    R2_OUT_MD.write_text(_r2_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "fits": fit_budget.used,
                      "artifact": str(R2_OUT_JSON)},
                     ensure_ascii=False, indent=1))
    return 0


# =========================================================================== #
# CLS-CONF -- the frozen confirmation on a UCR Target no classification
# experiment in this repository has ever touched
# =========================================================================== #
CONF_OUT_JSON = E2 / "t6_cls_conf_unused_target.json"
CONF_OUT_MD = E2 / "t6_cls_conf_unused_target.md"
CONF_PROTOCOL = "t6_cls_conf_unused_target_v1"
CONF_RUN_ID = "t6_cls_conf_run1"
CONF_LLM_CAP = 40
CONF_FIT_CAP = 200
CONF_ARMS = ("A3", "STATIC")
CONF_TRAIN_ROWS_MIN = 40
CONF_TRAIN_ROWS_MAX = 400
CONF_CONDITION = "fit_only_artifact"
# The census walks the repository itself rather than trusting a hand-kept list,
# but these are the rosters the reader will want named: anything here is
# excluded by construction and the name census must agree.
CONF_KNOWN_ROSTERS = {
    "W48_source": ["Coffee", "ECG200", "FordA", "GunPoint"],
    "W49_target": ["Wafer", "ECGFiveDays", "TwoLeadECG", "BeetleFly"],
    "W55_admission": list(W55_SOURCE_ROSTER),
    "W56_transfer": list(W56_TARGET_ROSTER),
    "cls1_cls2_cls3_cls4": ["GunPoint", "ECG200"],
}
# Directories whose contents are the data itself or this line's own scratch,
# and therefore say nothing about whether an experiment used a dataset.
CONF_CENSUS_SKIP_DIRS = frozenset({
    ".git", "data", "_scratch", "__pycache__", ".pytest_cache", ".ruff_cache",
    "node_modules",
})
CONF_CENSUS_SKIP_SUFFIXES = frozenset({
    ".zip", ".npy", ".npz", ".pyc", ".png", ".jpg", ".jpeg", ".pdf", ".gz",
    ".xz", ".so", ".dll", ".pyd", ".parquet",
})
CONF_CENSUS_MAX_BYTES = 64 * 1024 * 1024
CONF_CENSUS_CACHE = PROJECT_ROOT / "_scratch" / "_conf_name_census.json"
# A file that names almost the whole pool is a data inventory -- a download
# script, a roster dump, a census artifact -- and inventories say nothing about
# which datasets an *experiment* used.  Declared before the rule was applied.
CONF_INVENTORY_MIN_DISTINCT = 30

# CLS-CONF r2 -- same confirmation, narrowed "unused" definition.  Isolated
# artifact paths so a rerun cannot overwrite the r1 stall record.
CONF_R2_OUT_JSON = E2 / "t6_cls_conf_r2_unused_target.json"
CONF_R2_OUT_MD = E2 / "t6_cls_conf_r2_unused_target.md"
CONF_R2_PROTOCOL = "t6_cls_conf_unused_target_v2"
CONF_R2_RUN_ID = "t6_cls_conf_run2"
CONF_IMPULSE_TOKENS = ("fit_only_artifact", "stable_task_event")
# Pre-registered prediction for the r2 selection gate, frozen before this
# machine recount.  Not a roster and not an input to eligibility.
CONF_R2_PREDICTED_ELIGIBLE = (
    "Computers",
    "FreezerRegularTrain",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "PowerCons",
    "SemgHandGenderCh2",
    "WormsTwoClass",
    "Yoga",
)

# CLS-CONF-dl -- authorized download of <=3 new UCR binaries; isolated artifacts.
CONF_DL_OUT_JSON = E2 / "t6_cls_conf_dl.json"
CONF_DL_OUT_MD = E2 / "t6_cls_conf_dl.md"
CONF_DL_PROTOCOL = "t6_cls_conf_downloaded_target_v1"
CONF_DL_RUN_ID = "t6_cls_conf_dl_run1"
CONF_DL_DATA_DIR = "data/ucr_conf_downloaded/D1"
CONF_DL_ROSTER = PROJECT_ROOT / "data" / "ucr_conf_downloaded" / "ROSTER.md"
CONF_DL_META = PROJECT_ROOT / "_scratch" / "tsc_metadata.csv"

# CLS-DEV -- local already-used substrate, development-grade conf lifecycle.
# Isolated artifacts; same two-arm machinery as --conf-run; not a confirmation.
CONF_DEV_OUT_JSON = E2 / "t6_cls_conf_dev_ecg200.json"
CONF_DEV_OUT_MD = E2 / "t6_cls_conf_dev_ecg200.md"
CONF_DEV_PROTOCOL = "t6_cls_conf_dev_v1"
CONF_DEV_RUN_ID = "t6_cls_conf_dev_ecg200_run1"
CONF_DEV_DEFAULT_DATASET = "ECG200"
CONF_DEV_WALL_SECONDS = 90 * 60
CONF_DEV_EVIDENCE_GRADE = "development"
CONF_DEV_HONESTY = (
    "ECG200 was previously used by the W48 / W49 / curvature lines under the "
    "same impulse condition pair (audit: artifacts/functional/e2/"
    "t6_cls_conf_r3_selection.json). This run is therefore not an independent "
    "confirmation. Every judgement stays at evidence_grade=development. The "
    "label CLS_CHAIN_CONFIRMED must not be used."
)
CONF_DEV_POSITIVE = "DEV_CHAIN_POSITIVE"
CONF_DEV_NO_POSITIVE = "DEV_CHAIN_NO_POSITIVE"
GUNPOINT_A3_REFERENCE = {
    "dataset": "GunPointAgeSpan",
    "source": "artifacts/functional/e2/t6_cls_op_r2_three_arms.json",
    "skill": "hampel_filter",
    "first_skill_round": "r1",
    "support_gain": 0.5,
    "delayed_utility": 0.4,
    "heldout_accuracy": 0.8512658227848101,
    "heldout_identity_accuracy": 0.5822784810126582,
    "heldout_accuracy_gain": 0.2689873417721519,
    "heldout_recall_delta_by_class": {"0": 0.28125, "1": 0.2564102564102564},
    "support_delayed_direction_agree": 2,
    "support_delayed_direction_disagree": 0,
    "llm_calls_to_first_skill": 6,
    "candidate_executions_to_first_skill": 1,
    "retrieved_r1": [
        "build_contrastive_candidates",
        "inspect_and_localize",
        "select_or_identity_and_verify",
    ],
}


def _conf_dl_load_selection() -> dict[str, Any]:
    """Reload the pre-download filter trajectory.  Never invent a new pick."""
    if not CONF_DL_OUT_JSON.is_file():
        raise Stop("INSTRUMENT_UNREADABLE",
                   "filter-trajectory artifact missing: %s" % CONF_DL_OUT_JSON)
    payload = json.loads(CONF_DL_OUT_JSON.read_text(encoding="utf-8"))
    selection = payload.get("selection")
    if not isinstance(selection, dict) or not selection.get("eligible"):
        raise Stop("INSTRUMENT_UNREADABLE",
                   "filter-trajectory artifact has no eligible list")
    return selection


def _conf_dl_census() -> dict[str, Any]:
    """D1 is the lexicographic first eligible name, unless structurally replaced."""
    selection = dict(_conf_dl_load_selection())
    d1 = PROJECT_ROOT / CONF_DL_DATA_DIR
    live = sorted(
        path.stem for path in d1.glob("*.zip")
        if path.stem == selection.get("selected")
        or (not path.stem.endswith("_aeon_source")
            and path.stem != "BinaryHeartbeat_aeon_source"))
    promoted = selection.get("promoted_d1")
    if promoted:
        selection["selected"] = promoted
        selection["selection_basis"] = (
            "D3 promoted after structural failure of original D1; "
            "original D1=%s" % selection.get("original_d1"))
    elif selection.get("selected") not in {
            path.stem for path in d1.glob("*.zip")}:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "D1 zip %s.zip is not in %s (found %s)"
                   % (selection.get("selected"), CONF_DL_DATA_DIR, live))
    return selection


def _conf_dl_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload.get("verdict") or {}
    ledger = payload.get("ledger") or {}
    selection = payload.get("selection") or {}
    download = payload.get("download_record") or {}
    conversion = payload.get("format_conversion") or {}
    seal = payload.get("seal_table") or []
    lines = [
        "# CLS-CONF-dl -- A3 vs Static identity on a downloaded UCR Target",
        "",
        "protocol: `%s`  target: **%s**  evidence grade: **%s**"
        % (payload.get("protocol_version"), payload.get("target"),
           payload.get("evidence_grade")),
        "",
        "## Verdict",
        "",
        "**%s**" % verdict.get("verdict"),
        "",
        str(verdict.get("rule") or verdict.get("reason") or ""),
        "",
        "- non-identity Target-local Skill formed: %s"
        % verdict.get("non_identity_target_local_skill_formed"),
        "- A3 minus Static held-out accuracy: %s (material line %s)"
        % (verdict.get("a3_minus_static_heldout_accuracy"),
           verdict.get("material_line")),
        "- worst per-class recall delta: %s (zero class harm: %s)"
        % (verdict.get("worst_class_recall_delta"),
           verdict.get("zero_class_harm")),
        "- deployment purity: %s" % verdict.get("deploy_purity_all_pure"),
        "",
        str(verdict.get("claim_limit", "")),
        "",
        "## Filter trajectory",
        "",
        str(selection.get("rule", "")),
        "",
        "- official table: `%s` (%s rows, fetched %s)"
        % ((selection.get("metadata_source") or {}).get("url"),
           (selection.get("metadata_source") or {}).get("n_rows"),
           (selection.get("metadata_source") or {}).get("fetched_utc")),
        "- local zip stems: %d from `data/ucr_task_context`"
        % (selection.get("local_zip_enumeration") or {}).get("n_zips", 0),
        "- eligible (%d): %s"
        % (len(selection.get("eligible") or []),
           ", ".join(selection.get("eligible") or []) or "(none)"),
        "- D1/D2/D3: %s"
        % ", ".join("%s=%s" % (row.get("role"), row.get("dataset"))
                    for row in (selection.get("d1_d2_d3") or [])),
        "",
    ]
    for step in selection.get("filter_steps") or []:
        lines += [
            "### Step %s — %s" % (step.get("step"), step.get("predicate")),
            "",
            "- pass %s / fail %s" % (step.get("n_pass"), step.get("n_fail")),
            "",
        ]
    lines += [
        "## Downloads",
        "",
        "- count: %s (cap 3; no full archive)"
        % (download.get("n_dataset_zips") or ledger.get("downloads")),
        "",
    ]
    for row in download.get("items") or []:
        lines.append(
            "- **%s** `%s`: %s  bytes=%s  utc=%s  members=%s"
            % (row.get("role"), row.get("dataset"), row.get("source_url"),
               row.get("bytes"), row.get("downloaded_utc"),
               ", ".join(row.get("member_names") or []) or "(n/a)"))
    lines += [
        "",
        "## Format conversion (D1 only)",
        "",
        json.dumps(conversion, ensure_ascii=False, indent=1)
        if conversion else "(none)",
        "",
        "## Held-in trajectory",
        "",
        "| arm | round | proposal | Support receipts | winner | delayed | "
        "abstain | relation(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in payload.get("rounds") or []:
        relations = ",".join(str(ep.get("relation"))
                             for ep in record.get("episodes") or []) or "-"
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (record.get("arm"), record.get("round"),
               record.get("chosen"), record.get("support_receipts"),
               record.get("winner_program"), record.get("delayed_utility"),
               record.get("abstained"), relations))
    if not payload.get("rounds"):
        lines.append("| (none) | | | | | | | |")
    lines += [
        "",
        "## Skill and freeze",
        "",
    ]
    cells = (payload.get("readouts") or {}).get("cells") or {}
    for key in sorted(cells):
        row = cells[key]
        first = row.get("first_skill") or {}
        lines.append(
            "- **%s**: skill_formed=%s first=%s deploy see below"
            % (key, row.get("target_local_skill_formed"), first))
    lines += [
        "",
        "## Held-out accuracy and per-class recall",
        "",
        "| arm | held-out acc | vs identity | recall | recall delta | "
        "Support/delayed agree:disagree |",
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(cells):
        row = cells[key]
        lines.append(
            "| %s | %s | %s | %s | %s | %s:%s |"
            % (row.get("arm"), row.get("heldout_accuracy"),
               row.get("heldout_accuracy_gain"),
               row.get("heldout_recall_by_class"),
               row.get("heldout_recall_delta_by_class"),
               row.get("support_delayed_direction_agree"),
               row.get("support_delayed_direction_disagree")))
    lines += [
        "",
        "## Cost",
        "",
        "- LLM: %s / %s" % (ledger.get("llm_calls"), ledger.get("llm_cap")),
        "- Consumer fits: %s / %s"
        % (ledger.get("consumer_fits"), ledger.get("consumer_fit_cap")),
        "- wall clock: %s s" % ledger.get("wall_seconds"),
        "- dataset zips downloaded: %s"
        % (download.get("n_dataset_zips") or ledger.get("downloads")),
        "",
        "## Seal table",
        "",
        "| role | dataset | path | sealed | values loaded |",
        "|---|---|---|---|---|",
    ]
    for row in seal:
        lines.append("| %s | %s | %s | %s | %s |"
                     % (row.get("role"), row.get("dataset"), row.get("path"),
                        row.get("sealed"), row.get("values_loaded")))
    lines += ["", "## Obligations", ""]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


def _conf_dl_sidecar() -> dict[str, Any]:
    """Download / conversion / seal records written beside the runner."""
    log_path = PROJECT_ROOT / "_scratch" / "cls_conf_dl_download_log.json"
    d3_path = PROJECT_ROOT / "_scratch" / "cls_conf_dl_d3_retry.json"
    conv_path = PROJECT_ROOT / "_scratch" / "cls_conf_dl_conversion.json"
    download_items = []
    if log_path.is_file():
        download_items.extend(json.loads(log_path.read_text(encoding="utf-8")))
    d3 = json.loads(d3_path.read_text(encoding="utf-8")) if d3_path.is_file() else {}
    conversion = (json.loads(conv_path.read_text(encoding="utf-8"))
                  if conv_path.is_file() else {})
    return {
        "download_record": {
            "n_dataset_zips": 3,
            "items": download_items,
            "d3_equivalent_link": d3,
        },
        "format_conversion": conversion,
        "seal_table": [
            {"role": "D1", "dataset": "BinaryHeartbeat",
             "path": "data/ucr_conf_downloaded/D1/",
             "sealed": False, "values_loaded": True,
             "note": "CLS-CONF target; converted zip loaded"},
            {"role": "D2_sealed", "dataset": "CatsDogs",
             "path": "data/ucr_conf_downloaded/D2_sealed/",
             "sealed": True, "values_loaded": False,
             "note": "zip open + member names only; reserved for future A5 vs A3"},
            {"role": "D3_reserve", "dataset": "Epilepsy2",
             "path": "data/ucr_conf_downloaded/D3_reserve/",
             "sealed": True, "values_loaded": False,
             "note": (
                 "aeon-toolkit/Epilepsy2.zip 404; equivalent official zip "
                 "EpilepticSeizures.zip (same official TRAIN/TEST/length/"
                 "classes as metadata Epilepsy2) stored; members only")},
        ],
        "obligations": {
            "downloads": 3,
            "d2_d3_values_not_loaded": True,
            "methods_package_unmodified": True,
            "selection_rule_not_relaxed": True,
            "italy_power_demand_not_downloaded": True,
        },
    }


def conf_dl_select() -> int:
    """0-LLM dry run: recap the frozen pick and build the D1 cell."""
    census = _conf_dl_census()
    print("CLS-CONF-dl select  protocol=%s  selected=%s"
          % (CONF_DL_PROTOCOL, census.get("selected")), flush=True)
    print("eligible:", census.get("eligible"), flush=True)
    print("roles:", census.get("d1_d2_d3"), flush=True)
    cell = _build_cell(str(census["selected"]), CONF_CONDITION,
                       data_dir=CONF_DL_DATA_DIR)
    summary = {key: cell[key] for key in (
        "dataset", "condition", "archive", "series_length",
        "official_train_rows", "fit_rows", "support_pool_rows",
        "slice_rows", "controlled_impulse_positions",
        "observer_recovered_all_nodes")}
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


def conf_dl_run(*, run_id: str = CONF_DL_RUN_ID) -> int:
    """CLS-CONF-dl: same two-arm protocol on the downloaded D1 zip."""
    extra = _conf_dl_sidecar()
    extra["ledger_downloads"] = 3
    return conf_run(
        run_id=run_id, protocol=CONF_DL_PROTOCOL,
        out_json=CONF_DL_OUT_JSON, out_md=CONF_DL_OUT_MD,
        census_fn=_conf_dl_census, entry="--conf-dl-run",
        data_dir=CONF_DL_DATA_DIR, extra=extra,
        markdown_fn=_conf_dl_markdown)


def _conf_dev_census(dataset: str) -> dict[str, Any]:
    """Pick a local already-used zip.  Not an unused-Target census."""
    if "/" in dataset or "\\" in dataset or ".." in dataset:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "dataset name must be a bare UCR stem, got %r" % dataset)
    archive = PROJECT_ROOT / DATA_DIR / ("%s.zip" % dataset)
    if not archive.is_file():
        raise Stop("INSTRUMENT_UNREADABLE",
                   "local zip missing: %s/%s.zip" % (DATA_DIR, dataset))
    _ctx, helpers = _legacy_helpers()
    train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
    n_train = int(train_values.shape[0])
    length = int(train_values.shape[1])
    n_classes = int(len({int(label) for label in train_labels}))
    return {
        "selected": dataset,
        "rule": (
            "development reuse of a local UCR zip already used under the "
            "same impulse condition pair; the unused-Target census is not "
            "applied"),
        "pool_size": 1,
        "eligible": [dataset],
        "selection_basis": (
            "task-book CLS-DEV-ECG200; --dataset default %s; archive %s/%s.zip"
            % (CONF_DEV_DEFAULT_DATASET, DATA_DIR, dataset)),
        "archive": "%s/%s.zip" % (DATA_DIR, dataset),
        "bytes": int(archive.stat().st_size),
        "official_train_rows": n_train,
        "series_length": length,
        "class_count": n_classes,
        "approx_heldin_points": n_train * length,
        "independent_confirmation": False,
        "honesty_constraint": CONF_DEV_HONESTY,
        "prior_use_audit": "artifacts/functional/e2/t6_cls_conf_r3_selection.json",
    }


def _conf_dev_shape_note(payload: Mapping[str, Any]) -> str:
    """One paragraph: same-shape / different-shape vs GunPointAgeSpan A3."""
    difference = payload.get("difference_read") or {}
    hampel = difference.get("hampel_filter_on_this_target") or {}
    rounds = [row for row in (payload.get("rounds") or [])
              if row.get("arm") == "A3"]
    families = []
    for record in rounds:
        for probe in record.get("probes") or []:
            families.append("%s/%s:%s" % (
                record.get("round"), probe.get("candidate_id"),
                probe.get("kind")))
    rejected = hampel.get("rejection_codes") or []
    fraction = hampel.get("cohort_modified_fraction")
    return (
        "Same-shape: A3-only, bootstrap-three retrieval, cohort verifier, "
        "maximum_candidates=3, held-in r1/r2 → freeze → Fast-only held-out. "
        "Different-shape: GunPoint formed a hampel Target-local Skill in r1 "
        "with Support/delayed both positive; this run proposed %s and formed "
        "no Skill. Post-freeze menu diagnostic: hampel_filter verifier_passed="
        "%s rejection=%s cohort_fraction=%s. Scope-compiler development "
        "should treat this as same impulse family, different Program "
        "geometry / verifier fate, not as a second hampel positive."
        % (families or "(none)",
           hampel.get("verifier_passed"),
           ",".join(str(code) for code in rejected) or "-",
           fraction))


def _conf_dev_verdict(payload: Mapping[str, Any], *,
                      stopped: str | None) -> dict[str, Any]:
    """Same gate as _conf_verdict; DEV labels only; no CLS_CHAIN_CONFIRMED."""
    verdict = _conf_verdict(
        payload, stopped=stopped,
        positive_label=CONF_DEV_POSITIVE,
        negative_label=CONF_DEV_NO_POSITIVE,
        claim_limit=(
            "development only.  " + CONF_DEV_HONESTY
            + "  The impulse is a controlled injection; a positive here is a "
            "second development-grade Target-local Skill, not a fresh "
            "confirmation."))
    verdict["forbidden_label"] = "CLS_CHAIN_CONFIRMED"
    verdict["independent_confirmation"] = False
    verdict["evidence_grade"] = CONF_DEV_EVIDENCE_GRADE
    return verdict


def _conf_dev_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload.get("verdict") or {}
    ledger = payload.get("ledger") or {}
    selection = payload.get("selection") or {}
    cells = (payload.get("readouts") or {}).get("cells") or {}
    honesty = payload.get("honesty_constraint") or CONF_DEV_HONESTY
    lines = [
        "# CLS-DEV-ECG200 -- development-grade conf lifecycle on local ECG200",
        "",
        "protocol: `%s`  target: **%s**  evidence grade: **%s**"
        % (payload.get("protocol_version"), payload.get("target"),
           payload.get("evidence_grade")),
        "",
        "## Verdict",
        "",
        "**%s**" % verdict.get("verdict"),
        "",
        str(verdict.get("rule") or verdict.get("reason") or ""),
        "",
        "- non-identity Target-local Skill formed: %s"
        % verdict.get("non_identity_target_local_skill_formed"),
        "- A3 minus Static held-out accuracy: %s (material line %s)"
        % (verdict.get("a3_minus_static_heldout_accuracy"),
           verdict.get("material_line")),
        "- worst per-class recall delta: %s (zero class harm: %s)"
        % (verdict.get("worst_class_recall_delta"),
           verdict.get("zero_class_harm")),
        "- deployment purity: %s" % verdict.get("deploy_purity_all_pure"),
        "- forbidden label unused: CLS_CHAIN_CONFIRMED",
        "",
        str(verdict.get("claim_limit") or ""),
        "",
        "## Honesty constraint",
        "",
        honesty if isinstance(honesty, str) else json.dumps(
            honesty, ensure_ascii=False),
        "",
        "This artifact is **development** evidence.  It is not an independent "
        "confirmation and must not be cited as CLS_CHAIN_CONFIRMED.",
        "",
        "## Substrate",
        "",
        "- archive: `%s` (%s bytes)"
        % (selection.get("archive"), selection.get("bytes")),
        "- TRAIN rows × length: %s × %s; classes: %s"
        % (selection.get("official_train_rows"),
           selection.get("series_length"), selection.get("class_count")),
        "- selection: %s" % selection.get("selection_basis"),
        "- condition: `%s`" % payload.get("condition"),
        "",
        "## Held-in trajectory",
        "",
        "| arm | round | retrieved_skill_ids | chosen | probes | winner | "
        "Support receipts | delayed | abstain | relation(s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in payload.get("rounds") or []:
        relations = ",".join(str(ep.get("relation"))
                             for ep in record.get("episodes") or []) or "-"
        retrieved = ",".join(record.get("retrieved_skill_ids") or []) or "-"
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (record.get("arm"), record.get("round"), retrieved,
               record.get("chosen"), record.get("proposal_count"),
               record.get("winner_program"), record.get("support_receipts"),
               record.get("delayed_utility"), record.get("abstained"),
               relations))
    if not payload.get("rounds"):
        lines.append("| (none) | | | | | | | | | |")
    lines += [
        "",
        "## Two-arm readouts",
        "",
        "| arm | Skill formed | first-Skill LLM | first-Skill "
        "executions | held-in delayed | held-out acc | vs identity | "
        "recall by class | recall delta | worst class recall d | "
        "Support/delayed agree:disagree | deploy |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(cells):
        row = cells[key]
        first = row.get("first_skill") or {}
        acc = row.get("heldout_accuracy")
        gain = row.get("heldout_accuracy_gain")
        worst = row.get("worst_class_recall_delta")
        delayed = row.get("held_in_delayed_utility")
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s:%s | %s |"
            % (row.get("arm"), row.get("target_local_skill_formed"),
               first.get("llm_calls_to_first_skill", "-"),
               first.get("candidate_executions_to_first_skill", "-"),
               ("%+.4f" % delayed) if delayed is not None else "n/a",
               ("%.4f" % acc) if acc is not None else "n/a",
               ("%+.4f" % gain) if gain is not None else "n/a",
               row.get("heldout_recall_by_class"),
               row.get("heldout_recall_delta_by_class"),
               ("%+.4f" % worst) if worst is not None else "n/a",
               row.get("support_delayed_direction_agree"),
               row.get("support_delayed_direction_disagree"),
               row.get("deploy_source")))
    if not cells:
        lines.append("| (none) | | | | | | | | | | | |")
    ref = GUNPOINT_A3_REFERENCE
    a3 = cells.get("%s/A3" % payload.get("target")) or {}
    first = a3.get("first_skill") or {}
    lines += [
        "",
        "## Lifecycle shape vs GunPointAgeSpan A3 positive",
        "",
        "GunPointAgeSpan A3 (development positive, `%s`): r1 retrieved "
        "bootstrap-only `%s`; proposal `hampel_filter`; Support +%.2f → "
        "delayed +%.2f; Skill in r1; held-out %.4f vs identity %.4f "
        "(+%0.4f); per-class recall delta %s; Support-delayed %s:%s; "
        "first-Skill LLM %s / executions %s."
        % (ref["source"], ",".join(ref["retrieved_r1"]),
           ref["support_gain"], ref["delayed_utility"],
           ref["heldout_accuracy"], ref["heldout_identity_accuracy"],
           ref["heldout_accuracy_gain"],
           ref["heldout_recall_delta_by_class"],
           ref["support_delayed_direction_agree"],
           ref["support_delayed_direction_disagree"],
           ref["llm_calls_to_first_skill"],
           ref["candidate_executions_to_first_skill"]),
        "",
        "This run A3: retrieved bootstrap-only on both rounds; first "
        "Skill=%s round=%s; held-out acc=%s gain=%s; recall delta=%s; "
        "Support-delayed %s:%s."
        % (first.get("program"), first.get("round"),
           a3.get("heldout_accuracy"), a3.get("heldout_accuracy_gain"),
           a3.get("heldout_recall_delta_by_class"),
           a3.get("support_delayed_direction_agree"),
           a3.get("support_delayed_direction_disagree")),
        "",
        _conf_dev_shape_note(payload),
        "",
        "Same-shape / different-shape notes are for Scope-compiler "
        "development only.  Shared-capability induction is not authorized "
        "from this development pair.",
        "",
        "## Cost",
        "",
        "- LLM: %s / %s" % (ledger.get("llm_calls"), ledger.get("llm_cap")),
        "- Consumer fits: %s / %s"
        % (ledger.get("consumer_fits"), ledger.get("consumer_fit_cap")),
        "- wall clock: %s s (cap %s s)"
        % (ledger.get("wall_seconds"), CONF_DEV_WALL_SECONDS),
        "- downloads: %s" % ledger.get("downloads", 0),
        "",
        "## Obligations",
        "",
    ]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


def _conf_dev_paths(dataset: str) -> tuple[Path, Path, str]:
    slug = "".join(ch for ch in dataset.lower() if ch.isalnum()) or "target"
    if dataset == CONF_DEV_DEFAULT_DATASET:
        return CONF_DEV_OUT_JSON, CONF_DEV_OUT_MD, CONF_DEV_RUN_ID
    return (E2 / ("t6_cls_conf_dev_%s.json" % slug),
            E2 / ("t6_cls_conf_dev_%s.md" % slug),
            "t6_cls_conf_dev_%s_run1" % slug)


def _conf_dev_write_stall(*, out_json: Path, out_md: Path, run_id: str,
                          dataset: str, started: float,
                          reason: str) -> None:
    payload = {
        "protocol_version": CONF_DEV_PROTOCOL,
        "run_id": run_id,
        "entry": "--conf-dev-run",
        "evidence_grade": CONF_DEV_EVIDENCE_GRADE,
        "target": dataset,
        "honesty_constraint": CONF_DEV_HONESTY,
        "independent_confirmation": False,
        "verdict": {
            "verdict": "COMPUTE_BUDGET_EXCEEDED",
            "reason": reason,
            "note": (
                "stop-report; 90-minute wall-clock cap; not a scientific "
                "negative"),
            "forbidden_label": "CLS_CHAIN_CONFIRMED",
        },
        "ledger": {
            "llm_calls": None,
            "llm_cap": CONF_LLM_CAP,
            "consumer_fits": None,
            "consumer_fit_cap": CONF_FIT_CAP,
            "wall_seconds": round(time.time() - started, 1),
            "wall_cap_seconds": CONF_DEV_WALL_SECONDS,
            "downloads": 0,
        },
        "obligations": {
            "methods_package_unmodified": True,
            "downloads": 0,
            "ucr_conf_downloaded_not_opened": True,
            "not_an_independent_confirmation": True,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    out_md.write_text(_conf_dev_markdown(payload), encoding="utf-8")


def conf_dev_run(*, dataset: str = CONF_DEV_DEFAULT_DATASET,
                 run_id: str | None = None) -> int:
    """CLS-DEV: same two-arm conf lifecycle on a local light substrate."""
    import threading

    out_json, out_md, default_run_id = _conf_dev_paths(dataset)
    run_id = run_id or default_run_id
    started = time.time()
    print("CLS-DEV start  protocol=%s  dataset=%s  wall_cap=%ss"
          % (CONF_DEV_PROTOCOL, dataset, CONF_DEV_WALL_SECONDS), flush=True)
    armed = {"live": True}

    def _timeout() -> None:
        if not armed["live"]:
            return
        _conf_dev_write_stall(
            out_json=out_json, out_md=out_md, run_id=run_id,
            dataset=dataset, started=started,
            reason=("hard wall-clock cap of %ss reached; process terminated"
                    % CONF_DEV_WALL_SECONDS))
        print(json.dumps({
            "verdict": "COMPUTE_BUDGET_EXCEEDED",
            "target": dataset,
            "wall_seconds": round(time.time() - started, 1),
            "artifact": str(out_json),
        }, ensure_ascii=False, indent=1), flush=True)
        os._exit(2)

    timer = threading.Timer(CONF_DEV_WALL_SECONDS, _timeout)
    timer.daemon = True
    timer.start()
    extra = {
        "honesty_constraint": CONF_DEV_HONESTY,
        "independent_confirmation": False,
        "forbidden_label": "CLS_CHAIN_CONFIRMED",
        "wall_cap_seconds": CONF_DEV_WALL_SECONDS,
        "ledger_downloads": 0,
        "obligations": {
            "sealed_d2_d3_untouched": True,
            "ucr_conf_downloaded_not_opened": True,
            "methods_package_unmodified": True,
            "downloads": 0,
            "not_an_independent_confirmation": True,
            "cls_chain_confirmed_label_not_used": True,
            "target_never_used_before": (
                "FALSE.  ECG200 is a local already-used substrate "
                "(W48/W49/curvature under the same impulse pair); this is "
                "development reuse, not a virgin Target"),
        },
    }
    try:
        return conf_run(
            run_id=run_id, protocol=CONF_DEV_PROTOCOL,
            out_json=out_json, out_md=out_md,
            census_fn=lambda: _conf_dev_census(dataset),
            entry="--conf-dev-run", data_dir=DATA_DIR, extra=extra,
            markdown_fn=_conf_dev_markdown,
            verdict_fn=_conf_dev_verdict,
            evidence_grade=CONF_DEV_EVIDENCE_GRADE,
            accepted_verdicts=(CONF_DEV_POSITIVE, CONF_DEV_NO_POSITIVE))
    except Stop as stop:
        if stop.verdict == "INSTRUMENT_UNREADABLE":
            payload = {
                "protocol_version": CONF_DEV_PROTOCOL,
                "run_id": run_id,
                "entry": "--conf-dev-run",
                "evidence_grade": CONF_DEV_EVIDENCE_GRADE,
                "target": dataset,
                "honesty_constraint": CONF_DEV_HONESTY,
                "independent_confirmation": False,
                "verdict": {"verdict": stop.verdict, "reason": stop.reason,
                            "forbidden_label": "CLS_CHAIN_CONFIRMED"},
                "ledger": {"llm_calls": 0, "llm_cap": CONF_LLM_CAP,
                           "consumer_fits": 0,
                           "consumer_fit_cap": CONF_FIT_CAP,
                           "wall_seconds": round(time.time() - started, 1),
                           "downloads": 0},
                "obligations": extra["obligations"],
            }
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(
                json.dumps(_plain(payload), indent=1, ensure_ascii=False)
                + "\n", encoding="utf-8")
            out_md.write_text(_conf_dev_markdown(payload), encoding="utf-8")
            print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
            return 1
        raise
    finally:
        armed["live"] = False
        timer.cancel()


def _conf_name_census(names: Sequence[str]) -> dict[str, Any]:
    """One pass over the repository, counting every dataset name at once.

    ripgrep is not on PATH in this environment, and a per-name subprocess would
    be 40 passes anyway, so the scan is done here: walk once, and for each
    readable text file count every candidate name in it.  Directories that hold
    the data itself or this line's own scratch are skipped, because a zip
    filename says nothing about whether an experiment used the dataset.

    Substring collisions are deliberately left in.  Counting "GunPoint" also
    counts "GunPointAgeSpan" mentions, which over-excludes GunPoint --- and
    over-excluding is the safe direction when the whole point is a Target
    nothing has touched.
    """
    import re as _re

    if CONF_CENSUS_CACHE.is_file():
        cached = json.loads(CONF_CENSUS_CACHE.read_text(encoding="utf-8"))
        if sorted(cached.get("names") or ()) == sorted(names):
            return cached
    # longest name first, so a GunPointAgeSpan mention counts as itself rather
    # than also as a GunPoint mention
    pattern = _re.compile("|".join(
        _re.escape(name) for name in sorted(names, key=len, reverse=True)))
    per_file: list[dict[str, Any]] = []
    scanned = 0
    skipped = 0
    unreadable: list[str] = []
    seen_dirs: set[str] = set()
    seen_files: set[str] = set()
    root_text = str(PROJECT_ROOT)
    # os.walk with in-place pruning, not rglob: the data directory holds an
    # unreadable link (data/tsquality) that rglob stats before any filter can
    # skip it.  The repository also contains a directory link back to itself,
    # which os.walk follows on Windows because a junction is not an os.path
    # symlink -- so every real path is visited at most once by realpath.
    for directory, subdirectories, filenames in os.walk(PROJECT_ROOT):
        directory_real = os.path.realpath(directory)
        if directory_real in seen_dirs:
            subdirectories[:] = []
            continue
        seen_dirs.add(directory_real)
        keep: list[str] = []
        for name in subdirectories:
            if name in CONF_CENSUS_SKIP_DIRS:
                continue
            child = os.path.join(directory, name)
            child_real = os.path.realpath(child)
            if child_real in seen_dirs or not child_real.startswith(root_text):
                continue
            if os.path.islink(child):
                continue
            try:
                attributes = os.stat(child, follow_symlinks=False)
                if getattr(attributes, "st_file_attributes", 0) & 0x400:
                    continue  # FILE_ATTRIBUTE_REPARSE_POINT: junction
            except OSError:
                continue
            keep.append(name)
        subdirectories[:] = keep
        for filename in filenames:
            path = Path(directory) / filename
            relative = path.relative_to(PROJECT_ROOT)
            if path.suffix.lower() in CONF_CENSUS_SKIP_SUFFIXES:
                skipped += 1
                continue
            try:
                file_real = os.path.realpath(path)
                if file_real in seen_files:
                    skipped += 1
                    continue
                seen_files.add(file_real)
                if path.stat().st_size > CONF_CENSUS_MAX_BYTES:
                    skipped += 1
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                skipped += 1
                unreadable.append(relative.as_posix())
                continue
            scanned += 1
            counts: dict[str, int] = {}
            for match in pattern.finditer(text):
                token = match.group(0)
                counts[token] = counts.get(token, 0) + 1
            if counts:
                per_file.append({"path": relative.as_posix(),
                                 "counts": counts,
                                 "distinct": len(counts)})
    census = {"tool": "in-process single pass, longest-name-first alternation",
              "names": sorted(names),
              "files_scanned": scanned, "files_skipped": skipped,
              "unreadable": unreadable[:8], "per_file": per_file}
    CONF_CENSUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CONF_CENSUS_CACHE.write_text(
        json.dumps(census, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return census


def _conf_usage_from_census(census: Mapping[str, Any],
                            names: Sequence[str]) -> dict[str, Any]:
    """Split mentions into pool inventory and actual experimental usage."""
    usage: dict[str, int] = {name: 0 for name in names}
    inventory: dict[str, int] = {name: 0 for name in names}
    usage_files: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    inventories: list[dict[str, Any]] = []
    for row in census.get("per_file") or ():
        is_inventory = int(row["distinct"]) >= CONF_INVENTORY_MIN_DISTINCT
        if is_inventory:
            inventories.append({"path": row["path"],
                                "distinct_datasets": row["distinct"]})
        for name, count in row["counts"].items():
            if name not in usage:
                continue
            if is_inventory:
                inventory[name] += int(count)
            else:
                usage[name] += int(count)
                usage_files[name].append({"path": row["path"],
                                          "hits": int(count)})
    for name in names:
        usage_files[name].sort(key=lambda item: (-item["hits"], item["path"]))
    return {"usage_hits": usage, "inventory_hits": inventory,
            "usage_files": usage_files,
            "pool_inventory_files": sorted(
                inventories, key=lambda item: item["path"]),
            "inventory_min_distinct": CONF_INVENTORY_MIN_DISTINCT}


def _conf_candidate_census() -> dict[str, Any]:
    """The whole pool, every exclusion reason, and the mechanical pick."""
    _ctx, helpers = _legacy_helpers()
    paths = sorted((PROJECT_ROOT / DATA_DIR).glob("*.zip"))
    names = [path.stem for path in paths]
    census = _conf_name_census(names)
    usage = _conf_usage_from_census(census, names)
    rows: list[dict[str, Any]] = []
    for path in paths:
        name = path.stem
        row: dict[str, Any] = {
            "dataset": name,
            "repo_name_hits": usage["usage_hits"][name],
            "pool_inventory_hits": usage["inventory_hits"][name],
            "repo_name_hit_files": usage["usage_files"][name][:8],
            "repo_name_hit_file_count": len(usage["usage_files"][name]),
            "in_named_roster": sorted({
                key for key, members in CONF_KNOWN_ROSTERS.items()
                if name in members}),
        }
        try:
            values, labels = helpers["load"](np, path, name, "TRAIN")
            row["train_rows"] = int(values.shape[0])
            row["series_length"] = int(values.shape[1])
            row["class_count"] = int(len(set(np.asarray(labels).tolist())))
            row["loadable"] = True
        except Exception as exc:  # noqa: BLE001
            row["loadable"] = False
            row["load_error"] = "%s: %s" % (type(exc).__name__, exc)
        reasons: list[str] = []
        if not row.get("loadable"):
            reasons.append("not_loadable_as_binary_ucr")
        else:
            if row["class_count"] != 2:
                reasons.append("not_binary")
            if not (CONF_TRAIN_ROWS_MIN <= row["train_rows"]
                    <= CONF_TRAIN_ROWS_MAX):
                reasons.append("train_rows_outside_%d_%d"
                               % (CONF_TRAIN_ROWS_MIN, CONF_TRAIN_ROWS_MAX))
        if row["repo_name_hits"] is None:
            reasons.append("name_census_unavailable")
        elif row["repo_name_hits"] > 0:
            reasons.append("name_already_appears_in_the_repository")
        row["claiming_runners"] = sorted({
            hit["path"] for hit in usage["usage_files"][name]
            if hit["path"].startswith("evaluation/functional/run_")})
        row["excluded_because"] = reasons
        row["eligible"] = not reasons
        rows.append(row)
    eligible = sorted(row["dataset"] for row in rows if row["eligible"])
    return {
        "rule": (
            "candidate pool = every zip in %s whose name no non-inventory file "
            "in the repository mentions; keep the binary ones whose official "
            "TRAIN row count is in [%d, %d]; take the lexicographically first. "
            "A file naming >= %d of the pool is a data inventory, not an "
            "experiment roster, and its mentions are not usage.  Fixed before "
            "the rule was applied and independent of any outcome."
            % (DATA_DIR, CONF_TRAIN_ROWS_MIN, CONF_TRAIN_ROWS_MAX,
               CONF_INVENTORY_MIN_DISTINCT)),
        "named_rosters_excluded_by_construction": CONF_KNOWN_ROSTERS,
        "census_method": census["tool"],
        "census_files_scanned": census["files_scanned"],
        "census_files_skipped": census["files_skipped"],
        "census_skipped_directories": sorted(CONF_CENSUS_SKIP_DIRS),
        "pool_inventory_files": usage["pool_inventory_files"],
        "inventory_min_distinct": CONF_INVENTORY_MIN_DISTINCT,
        "pool_size": len(rows),
        "candidates": rows,
        "eligible": eligible,
        "selected": eligible[0] if eligible else None,
        "selection_basis": (
            "lexicographically first of %d eligible" % len(eligible)
            if eligible else "no eligible dataset"),
    }


def conf_select() -> int:
    """Part A on its own, so the pick can be inspected before any LLM spend."""
    census = _conf_candidate_census()
    print("pool inventory files (mentions not counted as usage):")
    for row in census["pool_inventory_files"]:
        print("   %-70s distinct=%d" % (row["path"], row["distinct_datasets"]))
    print()
    for row in census["candidates"]:
        print("%-32s usage=%-6s inv=%-6s rows=%-5s cls=%-3s %s"
              % (row["dataset"], row["repo_name_hits"],
                 row["pool_inventory_hits"],
                 row.get("train_rows"), row.get("class_count"),
                 ",".join(row["excluded_because"]) or "ELIGIBLE"),
              flush=True)
        if row["eligible"] or (row["repo_name_hits"] or 0) <= 4:
            for hit in row["repo_name_hit_files"][:4]:
                print("        %-70s %d" % (hit["path"], hit["hits"]))
    print(json.dumps({"eligible": census["eligible"],
                      "selected": census["selected"]},
                     ensure_ascii=False, indent=1))
    return 0


def _conf_impulse_hits(claiming_runners: Sequence[str]) -> list[dict[str, Any]]:
    """Read each claiming runner; record which impulse-condition tokens it holds.

    Over-exclusion is the safe direction: an unreadable claiming runner is
    treated as a hit, because we cannot prove it never used the pair.
    """
    hits: list[dict[str, Any]] = []
    for relative in claiming_runners:
        path = PROJECT_ROOT / relative
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            hits.append({"runner": relative, "tokens": list(CONF_IMPULSE_TOKENS),
                         "unreadable": "%s: %s" % (type(exc).__name__, exc)})
            continue
        tokens = [token for token in CONF_IMPULSE_TOKENS if token in text]
        if tokens:
            hits.append({"runner": relative, "tokens": tokens})
    return hits


def _conf_r2_candidate_census() -> dict[str, Any]:
    """r1 census plus the narrowed unused rule: impulse-condition token scan."""
    base = _conf_candidate_census()
    rows: list[dict[str, Any]] = []
    for row in base["candidates"]:
        reasons = [reason for reason in row["excluded_because"]
                   if reason != "name_already_appears_in_the_repository"]
        impulse_hits = _conf_impulse_hits(row["claiming_runners"])
        if impulse_hits:
            reasons.append("claiming_runner_used_impulse_condition_pair")
        new_row = dict(row)
        new_row["impulse_condition_hits"] = impulse_hits
        new_row["excluded_because"] = reasons
        new_row["eligible"] = not reasons
        rows.append(new_row)
    eligible = sorted(row["dataset"] for row in rows if row["eligible"])
    predicted = list(CONF_R2_PREDICTED_ELIGIBLE)
    selected = eligible[0] if eligible else None
    predicted_selected = predicted[0] if predicted else None
    gate = {
        "predicted_eligible": predicted,
        "predicted_selected": predicted_selected,
        "actual_eligible": eligible,
        "actual_selected": selected,
        "eligible_match": eligible == predicted,
        "selected_match": selected == predicted_selected,
        "passed": eligible == predicted and selected == predicted_selected,
        "rule": (
            "stop before any LLM if the machine recount of eligible or "
            "selected disagrees with the pre-registered prediction; do not "
            "relax or tighten the exclusion rule"),
    }
    return {
        "rule": (
            "r2 unused = never used under the impulse defect-repair condition "
            "pair: if any claiming runner file contains %s or %s, the dataset "
            "is out (over-exclude). Keep the binary ones whose official TRAIN "
            "row count is in [%d, %d]; take the lexicographically first. "
            "Name-appearance is no longer an exclusion.  Fixed before the "
            "rule was applied and independent of any outcome."
            % (CONF_IMPULSE_TOKENS[0], CONF_IMPULSE_TOKENS[1],
               CONF_TRAIN_ROWS_MIN, CONF_TRAIN_ROWS_MAX)),
        "protocol_version": CONF_R2_PROTOCOL,
        "named_rosters_excluded_by_construction": CONF_KNOWN_ROSTERS,
        "census_method": base["census_method"],
        "census_files_scanned": base["census_files_scanned"],
        "census_files_skipped": base["census_files_skipped"],
        "census_skipped_directories": base["census_skipped_directories"],
        "pool_inventory_files": base["pool_inventory_files"],
        "inventory_min_distinct": base["inventory_min_distinct"],
        "pool_size": len(rows),
        "impulse_condition_tokens": list(CONF_IMPULSE_TOKENS),
        "r1_name_census_reused": True,
        "candidates": rows,
        "eligible": eligible,
        "selected": selected,
        "selection_basis": (
            "lexicographically first of %d eligible" % len(eligible)
            if eligible else "no eligible dataset"),
        "prediction_gate": gate,
    }


def _conf_r2_write(payload: Mapping[str, Any]) -> None:
    CONF_R2_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONF_R2_OUT_JSON.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    CONF_R2_OUT_MD.write_text(_conf_markdown(payload), encoding="utf-8")


def conf_r2_select() -> int:
    """Part A r2: recount the unused Target under the narrowed rule.  0 LLM."""
    started = time.time()
    census = _conf_r2_candidate_census()
    gate = census["prediction_gate"]
    print("CLS-CONF r2 selection  protocol=%s  cache_reused=%s"
          % (CONF_R2_PROTOCOL, census["r1_name_census_reused"]), flush=True)
    print("impulse tokens: %s" % ", ".join(CONF_IMPULSE_TOKENS), flush=True)
    print(flush=True)
    print("%-32s rows=%-5s cls=%-3s runners  impulse_hits  status"
          % ("dataset", "n", "k"), flush=True)
    for row in census["candidates"]:
        hits = row.get("impulse_condition_hits") or ()
        hit_text = ";".join(
            "%s[%s]" % (Path(item["runner"]).name, ",".join(item["tokens"]))
            for item in hits) or "-"
        runners = ",".join(Path(path).name
                           for path in row.get("claiming_runners") or ()) or "-"
        print("%-32s rows=%-5s cls=%-3s %s | %s | %s"
              % (row["dataset"], row.get("train_rows"), row.get("class_count"),
                 runners, hit_text,
                 ",".join(row["excluded_because"]) or "ELIGIBLE"),
              flush=True)
    print(flush=True)
    print("eligible (%d):" % len(census["eligible"]), flush=True)
    for row in census["candidates"]:
        if not row["eligible"]:
            continue
        print("   %-32s rows=%s cls=%s claiming=%s"
              % (row["dataset"], row.get("train_rows"), row.get("class_count"),
                 ",".join(Path(path).name
                          for path in row.get("claiming_runners") or ())
                 or "-"),
              flush=True)
    print("selected: %s (%s)" % (census["selected"], census["selection_basis"]),
          flush=True)
    print(json.dumps({"prediction_gate": gate}, ensure_ascii=False, indent=1),
          flush=True)
    payload: dict[str, Any] = {
        "protocol_version": CONF_R2_PROTOCOL,
        "run_id": CONF_R2_RUN_ID,
        "entry": "--conf-r2-select",
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": _git("rev-parse", "HEAD"),
        "selection": census,
        "target": census["selected"],
        "condition": CONF_CONDITION,
        "arms": list(CONF_ARMS),
        "prediction_gate": gate,
        "verdict": (
            {"verdict": "PREDICTION_GATE_PASSED",
             "selected": census["selected"]}
            if gate["passed"] else
            {"verdict": "PREDICTION_GATE_FAILED",
             "reason": (
                 "machine recount of eligible/selected disagrees with the "
                 "pre-registered prediction; stopped before any LLM; the "
                 "exclusion rule was not relaxed or tightened"),
             "predicted_eligible": gate["predicted_eligible"],
             "actual_eligible": gate["actual_eligible"],
             "predicted_selected": gate["predicted_selected"],
             "actual_selected": gate["actual_selected"]}),
        "ledger": {"llm_calls": 0, "llm_cap": CONF_LLM_CAP,
                   "consumer_fits": 0, "consumer_fit_cap": CONF_FIT_CAP,
                   "wall_seconds": round(time.time() - started, 1)},
        "obligations": {
            "selection_rule_not_relaxed": True,
            "selection_rule_not_tightened": True,
            "llm_spent": 0,
            "downloads": 0,
            "methods_package_unmodified": True,
            "r1_artifacts_not_overwritten": True,
        },
    }
    _conf_r2_write(payload)
    print("wrote %s" % CONF_R2_OUT_JSON, flush=True)
    return 0 if gate["passed"] else 1


def _conf_verdict(payload: Mapping[str, Any], *,
                  stopped: str | None,
                  positive_label: str = "CLS_CHAIN_CONFIRMED",
                  negative_label: str = "CLS_CHAIN_NOT_REPLICATED",
                  claim_limit: str | None = None) -> dict[str, Any]:
    """Positive label needs a Skill *and* a clean held-out gain."""
    if stopped:
        return {"verdict": stopped,
                "stage": (payload.get("stop") or {}).get("reason"),
                "note": "stop-report; no downstream reading is claimed"}
    readouts = (payload.get("readouts") or {}).get("cells") or {}
    purity = (payload.get("deploy_purity") or {}).get("all_pure")
    dataset = payload.get("target")
    expected = sorted("%s/%s" % (dataset, arm) for arm in CONF_ARMS)
    if sorted(readouts) != expected or not purity:
        return {"verdict": "INSTRUMENT_UNREADABLE",
                "reason": "two-arm cells incomplete or deployment impure",
                "cells": sorted(readouts),
                "deploy_purity_all_pure": purity}
    a3 = readouts["%s/A3" % dataset]
    static = readouts["%s/STATIC" % dataset]
    first = a3.get("first_skill") or {}
    skill = bool(a3["target_local_skill_formed"] and first.get("non_identity"))
    heldout_rows = None
    for deployment in payload.get("deployments") or []:
        if deployment["arm"] == "A3":
            heldout_rows = int(deployment["heldout_rows"])
    line = max(MATERIAL, 1.0 / max(int(heldout_rows or 1), 1))
    gain = float(a3["heldout_accuracy"]) - float(static["heldout_accuracy"])
    recall_delta = a3["heldout_recall_delta_by_class"]
    worst = min(recall_delta.values()) if recall_delta else 0.0
    material_positive = gain >= line
    zero_class_harm = worst >= -MATERIAL
    confirmed = bool(skill and material_positive and zero_class_harm)
    default_claim = (
        "DEVELOPMENT.  The impulse is a controlled injection on a UCR "
        "background; this confirms the Harness chain reproduces on a "
        "Target it had never seen, not that natural classification data "
        "carries the same defect.")
    return {
        "verdict": (positive_label if confirmed else negative_label),
        "rule": (
            "%s iff a non-identity Target-local Skill formed "
            "and the frozen Fast-only deployment beats Static identity by at "
            "least max(0.005, 1/n) on held-out accuracy with no per-class "
            "recall falling more than 0.005." % positive_label),
        "non_identity_target_local_skill_formed": skill,
        "a3_minus_static_heldout_accuracy": gain,
        "material_line": line,
        "material_positive": material_positive,
        "worst_class_recall_delta": worst,
        "zero_class_harm": zero_class_harm,
        "harm_bar_view": {
            "harm_bar": HARM_BAR,
            "classes_over_bar": a3["harmed_classes_over_bar"]},
        "deploy_purity_all_pure": purity,
        "claim_limit": claim_limit if claim_limit is not None else default_claim,
    }


def _conf_difference_read(cell: Mapping[str, Any], rounds: Sequence[Mapping[str, Any]],
                          *, fit_budget: FitBudget) -> dict[str, Any]:
    """Why this Target behaved as it did, against GunPointAgeSpan's profile.

    Run after the arms are frozen and scored, so it can inform the reading
    without having been able to steer the protocol.  Zero LLM; it opens no
    TEST split and reuses the held-in surfaces the arms already saw.
    """
    census = _r2_cell_pass(cell, fit_budget=fit_budget)
    probed = sorted({str(episode["workflow_signature"])
                     for record in rounds for episode in record["episodes"]})
    relations = {}
    for record in rounds:
        for episode in record["episodes"]:
            relations.setdefault(str(episode["relation"]), 0)
            relations[str(episode["relation"])] += 1
    hampel = next((row for row in census["programs"]
                   if row["program"] == "hampel_filter"), None)
    return {
        "note": ("post-freeze diagnostic; the arms were already scored when "
                 "this ran, so it could not have selected anything"),
        "operators_that_reached_a_legal_receipt": probed,
        "episode_relations": relations,
        "menu_survivors": census["survivors"],
        "menu_numeric_no_ops": census["numeric_no_ops"],
        "menu_verifier_rejected": census["verifier_rejected"],
        "hampel_filter_on_this_target": hampel,
        "gunpointagespan_reference": {
            "survivors": ["outlier_iqr", "hampel_filter"],
            "hampel_support_delta_accuracy": 0.5,
            "hampel_delayed_delta_accuracy": 0.3,
            "hampel_worst_class_recall_delta": 0.4,
            "source": "t6_cls_op_r2_prep.json / t6_cls_op_r2_three_arms.json",
        },
        "consumer_fits_after": fit_budget.used,
    }


def conf_r2_run(*, run_id: str = CONF_R2_RUN_ID) -> int:
    """CLS-CONF r2: same two-arm run, r2 census and isolated artifacts."""
    census = _conf_r2_candidate_census()
    if not census["prediction_gate"]["passed"]:
        started = time.time()
        payload: dict[str, Any] = {
            "protocol_version": CONF_R2_PROTOCOL,
            "run_id": run_id,
            "entry": "--conf-r2-run",
            "evidence_grade": EVIDENCE_GRADE,
            "git_head": _git("rev-parse", "HEAD"),
            "selection": census,
            "target": census["selected"],
            "condition": CONF_CONDITION,
            "arms": list(CONF_ARMS),
            "prediction_gate": census["prediction_gate"],
            "verdict": {
                "verdict": "PREDICTION_GATE_FAILED",
                "reason": (
                    "machine recount of eligible/selected disagrees with the "
                    "pre-registered prediction; stopped before any LLM"),
                "predicted_eligible": census["prediction_gate"][
                    "predicted_eligible"],
                "actual_eligible": census["prediction_gate"]["actual_eligible"],
            },
            "ledger": {"llm_calls": 0, "llm_cap": CONF_LLM_CAP,
                       "consumer_fits": 0, "consumer_fit_cap": CONF_FIT_CAP,
                       "wall_seconds": round(time.time() - started, 1)},
            "obligations": {
                "selection_rule_not_relaxed": True,
                "selection_rule_not_tightened": True,
                "llm_spent": 0,
                "downloads": 0,
                "methods_package_unmodified": True,
                "r1_artifacts_not_overwritten": True,
            },
        }
        _conf_r2_write(payload)
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        return 1
    return conf_run(run_id=run_id, protocol=CONF_R2_PROTOCOL,
                    out_json=CONF_R2_OUT_JSON, out_md=CONF_R2_OUT_MD,
                    census_fn=_conf_r2_candidate_census,
                    entry="--conf-r2-run")


def conf_run(*, run_id: str = CONF_RUN_ID,
             protocol: str = CONF_PROTOCOL,
             out_json: Path | None = None,
             out_md: Path | None = None,
             census_fn: Any = None,
             entry: str = "--conf-run",
             data_dir: str | None = None,
             extra: Mapping[str, Any] | None = None,
             markdown_fn: Any = None,
             verdict_fn: Any = None,
             evidence_grade: str | None = None,
             accepted_verdicts: Sequence[str] | None = None,
             injection_template: str = INJECTION_TEMPLATE_V1) -> int:
    """CLS-CONF: A3 against Static identity on the mechanically chosen Target."""
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )

    out_json = out_json or CONF_OUT_JSON
    out_md = out_md or CONF_OUT_MD
    census_fn = census_fn or _conf_candidate_census
    data_dir = data_dir or DATA_DIR
    markdown_fn = markdown_fn or _conf_markdown
    verdict_fn = verdict_fn or _conf_verdict
    evidence_grade = evidence_grade or EVIDENCE_GRADE
    accepted_verdicts = tuple(accepted_verdicts or (
        "CLS_CHAIN_CONFIRMED", "CLS_CHAIN_NOT_REPLICATED"))
    started = time.time()
    fit_budget = FitBudget(CONF_FIT_CAP)
    store_root = Path(tempfile.gettempdir()) / run_id
    if store_root.exists():
        shutil.rmtree(store_root)
    h0 = compile_snapshot(PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    backend = _live_backend(CONF_LLM_CAP)

    census = census_fn()
    payload: dict[str, Any] = {
        "protocol_version": protocol,
        "run_id": run_id,
        "entry": entry,
        "evidence_grade": evidence_grade,
        "git_head": _git("rev-parse", "HEAD"),
        "selection": census,
        "target": census["selected"],
        "condition": CONF_CONDITION,
        "arms": list(CONF_ARMS),
        "protocol": {
            "modification_fraction_scope": "cohort",
            "maximum_modified_fraction": float(
                _task_context().deployment_constraints.maximum_modified_fraction),
            "maximum_candidates": (
                _task_context().deployment_constraints.maximum_candidates),
            "held_in_rounds": list(HELD_IN_ROUNDS),
            "support_trial_budget_per_round": SUPPORT_TRIAL_BUDGET,
            "task_context_sha": _task_context().sha(),
            "new_surfaces_opened": "none; every gate is r2's",
            "static_arm": (
                "same store, same snapshot, zero held-in rounds, deploy from "
                "frozen state -- which is identity, because a Harness that "
                "never adapted has no incumbent"),
        },
        "injection": {
            "template": "C38 class-conditioned impulse, unchanged",
            "amplitude": "SPIKE_AMPLITUDE = 16.0 row standard deviations",
            "fractions": [0.08, 0.20, 0.80, 0.92],
            "seed_ledger": (
                "the family's injection is seed-free: _inject writes a fixed "
                "signed template at positions derived from the series length, "
                "and the fit/support split is deterministic evenly-spaced "
                "selection.  There is no RNG to seed, so a fresh seed would "
                "be a fiction; the run is bit-reproducible instead."),
        },
        "prediction_gate": census.get("prediction_gate"),
    }
    if injection_template == INJECTION_TEMPLATE_V2:
        payload["injection"] = {
            "template": (
                "C38 class-conditioned impulse, v2 length-proportional "
                "segments"),
            "template_version": INJECTION_TEMPLATE_V2,
            "amplitude": "SPIKE_AMPLITUDE = 16.0 row standard deviations",
            "fractions": [0.08, 0.20, 0.80, 0.92],
            "v1_segment_length": V1_SPIKE_SEGMENT_LENGTH,
            "gunpoint_positive_control_length": (
                GUNPOINT_POSITIVE_CONTROL_LENGTH),
            "segment_formula": (
                "round(%d/%d * series_length)"
                % (V1_SPIKE_SEGMENT_LENGTH, GUNPOINT_POSITIVE_CONTROL_LENGTH)),
            "source_constants": (
                "evaluation/functional/"
                "run_e2_task_context_label_evidence_witness.py:37 "
                "SPIKE_FRACTIONS; :38 SPIKE_AMPLITUDE; :95-100 _inject "
                "writes one point per position (v1 segment length = 1)"),
            "seed_ledger": (
                "the family's injection is seed-free: v2 writes a fixed "
                "signed template of width round(1/150*L) at the same "
                "relative positions as v1.  There is no RNG to seed."),
        }
    if census["selected"] is None:
        claimed: dict[str, list[str]] = {}
        for row in census["candidates"]:
            for runner in row["claiming_runners"]:
                claimed.setdefault(runner, []).append(row["dataset"])
        unusable = [row["dataset"] for row in census["candidates"]
                    if not row.get("loadable")]
        payload["exhaustion_analysis"] = {
            "statement": (
                "the local UCR inventory is exhausted with respect to this "
                "book's own selection rule: every one of the %d datasets in "
                "%s is already a roster member of some prior classification "
                "experiment in this repository, and the only two whose sole "
                "mention is incidental cannot be loaded as binary UCR at all."
                % (census["pool_size"], DATA_DIR)),
            "datasets_claimed_by_prior_runners": {
                runner: sorted(names) for runner, names in sorted(
                    claimed.items())},
            "datasets_the_loader_rejects": unusable,
            "loader_rejection_detail": {
                "DodgerLoopWeekend": (
                    "binary labels but the table is not finite (the series "
                    "carry missing values), and 20 TRAIN rows is below the "
                    "row floor anyway"),
                "KeplerLightCurves": (
                    "the archive contains no <name>_TRAIN.txt member; it is "
                    "packaged differently from the rest of the pool"),
            },
            "not_done_here": (
                "no rule was relaxed to manufacture a Target.  Widening "
                "'unused' after seeing that the strict pool is empty would be "
                "choosing the confirmation set with the answer in view."),
            "options_for_the_mainline": [
                "narrow 'unused' to 'never used under the C38 impulse "
                "family', which would admit the six datasets whose only prior "
                "use was run_e2_integrated_context_harness_evolution and the "
                "three from run_e2_source_prior_evidence_fusion -- weaker "
                "than a virgin Target but still outside the impulse line",
                "confirm on a frozen split of an already-used dataset that "
                "the impulse line never scored, and say plainly that it is a "
                "split-level rather than dataset-level confirmation",
                "authorise one download, which the current discipline forbids",
            ],
        }
        payload["verdict"] = {
            "verdict": "INSTRUMENT_UNREADABLE",
            "reason": ("the pre-registered candidate pool is empty; no unused "
                       "local UCR Target exists to confirm on"),
            "pool_size": census["pool_size"],
            "eligible": [],
            "llm_spent": 0,
            "arms_run": 0,
        }
        payload["ledger"] = {"llm_calls": 0, "llm_cap": CONF_LLM_CAP,
                             "consumer_fits": 0,
                             "consumer_fit_cap": CONF_FIT_CAP,
                             "wall_seconds": round(time.time() - started, 1)}
        payload["obligations"] = {
            "selection_rule_not_relaxed": True,
            "llm_spent_before_stopping": 0,
            "downloads": 0,
            "methods_package_unmodified": True,
            "artifact_not_committed": True,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(_plain(payload), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
        out_md.write_text(_conf_markdown(payload), encoding="utf-8")
        print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
        return 1

    dataset = str(census["selected"])
    stopped: str | None = None
    rounds: list[dict[str, Any]] = []
    deployments: list[dict[str, Any]] = []
    cell: Mapping[str, Any] | None = None
    try:
        cell = _build_cell(dataset, CONF_CONDITION, data_dir=data_dir,
                           injection_template=injection_template)
        payload["cell"] = {key: value for key, value in cell.items()
                           if key not in ("fit_values", "fit_labels",
                                          "surfaces", "observation_block")}
        for arm in CONF_ARMS:
            state = _new_arm_state(
                snapshot=h0,
                agent=_live_agent(cell["observation_block"],
                                  backend.new_arm_backend()),
                store_root=store_root, tag="conf_%s_%s" % (dataset, arm))
            if arm == "A3":
                for round_name in HELD_IN_ROUNDS:
                    record = _run_round(
                        state=state, cell=cell, round_name=round_name,
                        arm=arm, fit_budget=fit_budget, allow_fast_skill=True,
                        fraction_scope="cohort", ledger=backend)
                    rounds.append(record)
                    print("%-6s %-28s %s probes=%d winner=%s delayed=%s "
                          "retrieved=%s chosen=%s"
                          % (arm, dataset, round_name,
                             record["support_receipts"],
                             record["winner_program"],
                             record["delayed_utility"],
                             ",".join(record.get("retrieved_skill_ids") or [])
                             or "-",
                             record.get("chosen") or "-"), flush=True)
                    wall_cap = (extra or {}).get("wall_cap_seconds")
                    if wall_cap is not None and (
                            time.time() - started) > float(wall_cap):
                        raise Stop(
                            "COMPUTE_BUDGET_EXCEEDED",
                            "wall-clock cap of %ss exceeded after %s %s"
                            % (wall_cap, arm, round_name))
            deployment = _deploy_and_score(
                state=state, cell=cell, arm=arm, fit_budget=fit_budget)
            deployments.append(deployment)
            print("DEPLOY %-6s %-28s %-34s heldout_acc=%.4f gain=%+.4f"
                  % (arm, dataset, deployment["deploy_source"],
                     deployment["heldout_accuracy"],
                     deployment["heldout_accuracy_gain"]), flush=True)
    except Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {"verdict": stopped,
                           "reason": "%s: %s" % (type(exc).__name__, exc),
                           "traceback": traceback.format_exc()}

    payload["rounds"] = rounds
    payload["deployments"] = deployments
    payload["readouts"] = _r2_readouts(rounds, deployments)
    payload["deploy_purity"] = _deploy_purity(deployments)
    payload["verdict"] = verdict_fn(payload, stopped=stopped)
    if cell is not None and not stopped:
        try:
            payload["difference_read"] = _conf_difference_read(
                cell, rounds, fit_budget=fit_budget)
        except Stop as stop:
            payload["difference_read"] = {"error": stop.reason}
    payload["ledger"] = {
        "llm_calls": int(backend.calls),
        "llm_cap": CONF_LLM_CAP,
        "llm_within_cap": int(backend.calls) <= CONF_LLM_CAP,
        "consumer_fits": fit_budget.used,
        "consumer_fit_cap": fit_budget.cap,
        "wall_seconds": round(time.time() - started, 1),
    }
    payload["obligations"] = {
        "methods_package_unmodified": True,
        "new_gates_opened": "none; cohort scope and maximum_candidates=3 are "
                            "exactly r2's",
        "target_never_used_before": (
            "selected by a repository-wide name census, not by a hand-kept "
            "list; the census table is in this artifact"),
        "downloads": 0,
        "forbidden_data_untouched": (
            "no Yahoo, NOAA, NAB or SMD path is opened; the only data root is "
            + DATA_DIR),
        "artifact_not_committed": True,
        "difference_read_ran_after_the_freeze": True,
    }
    if extra:
        for key, value in extra.items():
            if key == "obligations" and isinstance(value, Mapping):
                payload["obligations"].update(value)
            elif key not in payload:
                payload[key] = value
        payload["obligations"]["forbidden_data_untouched"] = (
            "no Yahoo, NOAA, NAB or SMD path is opened; the only data root "
            "opened for values is " + data_dir)
        if extra.get("ledger_downloads") is not None:
            payload["ledger"]["downloads"] = extra["ledger_downloads"]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    out_md.write_text(markdown_fn(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "target": dataset,
                      "llm": payload["ledger"]["llm_calls"],
                      "fits": payload["ledger"]["consumer_fits"],
                      "artifact": str(out_json)},
                     ensure_ascii=False, indent=1))
    return 0 if payload["verdict"]["verdict"] in accepted_verdicts else 1


def _conf_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload.get("verdict") or {}
    ledger = payload.get("ledger") or {}
    selection = payload.get("selection") or {}
    lines = [
        "# CLS-CONF -- frozen confirmation on an unused UCR Target",
        "",
        "protocol: `%s`  target: **%s**  evidence grade: **%s**"
        % (payload.get("protocol_version"), payload.get("target"),
           payload.get("evidence_grade")),
        "",
        "## Verdict",
        "",
        "**%s**" % verdict.get("verdict"),
        "",
        str(verdict.get("rule") or verdict.get("reason") or ""),
        "",
        "- non-identity Target-local Skill formed: %s"
        % verdict.get("non_identity_target_local_skill_formed"),
        "- A3 minus Static held-out accuracy: %s (material line %s)"
        % (verdict.get("a3_minus_static_heldout_accuracy"),
           verdict.get("material_line")),
        "- worst per-class recall delta: %s (zero class harm: %s)"
        % (verdict.get("worst_class_recall_delta"),
           verdict.get("zero_class_harm")),
        "- deployment purity: %s" % verdict.get("deploy_purity_all_pure"),
        "",
        str(verdict.get("claim_limit", "")),
        "",
        "## Part A -- how the Target was chosen",
        "",
        str(selection.get("rule", "")),
        "",
        "- pool: %s zips; eligible: %s; selected: **%s** (%s)"
        % (selection.get("pool_size"), selection.get("eligible"),
           selection.get("selected"), selection.get("selection_basis")),
        "",
        "| dataset | usage hits | claiming runner(s) | train rows | classes | "
        "excluded because |",
        "|---|---|---|---|---|---|",
    ]
    for row in selection.get("candidates") or []:
        runners = ", ".join(
            path.rsplit("/", 1)[-1] for path in row.get("claiming_runners") or ())
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (row["dataset"], row["repo_name_hits"],
                        runners or "(none)",
                        row.get("train_rows"), row.get("class_count"),
                        ", ".join(row["excluded_because"]) or "**ELIGIBLE**"))
    gate = payload.get("prediction_gate") or selection.get("prediction_gate")
    if gate:
        lines += ["", "### Prediction gate", "",
                  "- predicted eligible: %s" % gate.get("predicted_eligible"),
                  "- actual eligible: %s" % gate.get("actual_eligible"),
                  "- predicted selected: **%s**" % gate.get("predicted_selected"),
                  "- actual selected: **%s**" % gate.get("actual_selected"),
                  "- passed: **%s**" % gate.get("passed"),
                  ""]
    impulse_rows = [row for row in selection.get("candidates") or []
                    if row.get("impulse_condition_hits")]
    if impulse_rows:
        lines += ["### Impulse-condition token hits on claiming runners", ""]
        for row in impulse_rows:
            for hit in row["impulse_condition_hits"]:
                lines.append("- **%s**: `%s` tokens=%s%s"
                             % (row["dataset"],
                                Path(hit["runner"]).name,
                                ",".join(hit.get("tokens") or ()),
                                (" unreadable=%s" % hit["unreadable"]
                                 if hit.get("unreadable") else "")))
        lines.append("")
    exhaustion = payload.get("exhaustion_analysis")
    if exhaustion:
        lines += ["", "### Pool exhausted", "", exhaustion["statement"], "",
                  "Datasets claimed by prior runners:", ""]
        for runner, names in exhaustion[
                "datasets_claimed_by_prior_runners"].items():
            lines.append("- `%s`: %s" % (runner, ", ".join(names)))
        lines += ["", "Loader rejects: %s"
                  % ", ".join(exhaustion["datasets_the_loader_rejects"]), ""]
        for name, why in exhaustion["loader_rejection_detail"].items():
            lines.append("- **%s**: %s" % (name, why))
        lines += ["", exhaustion["not_done_here"], "",
                  "Options for the mainline:", ""]
        for option in exhaustion["options_for_the_mainline"]:
            lines.append("- %s" % option)
    cells = (payload.get("readouts") or {}).get("cells") or {}
    lines += ["", "## Part B -- two arms", "",
              "| arm | Skill formed | first-Skill LLM | first-Skill "
              "executions | held-in delayed | held-out acc | vs identity | "
              "worst class recall d | abstained rounds | Support/delayed "
              "agree:disagree |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        row = cells[key]
        first = row.get("first_skill") or {}
        lines.append(
            "| %s | %s | %s | %s | %s | %.4f | %+.4f | %+.4f | %d | %d:%d |"
            % (row["arm"], row["target_local_skill_formed"],
               first.get("llm_calls_to_first_skill", "-"),
               first.get("candidate_executions_to_first_skill", "-"),
               ("%+.4f" % row["held_in_delayed_utility"])
               if row["held_in_delayed_utility"] is not None else "n/a",
               row["heldout_accuracy"], row["heldout_accuracy_gain"],
               row["worst_class_recall_delta"] or 0.0,
               row["abstained_rounds"],
               row["support_delayed_direction_agree"],
               row["support_delayed_direction_disagree"]))
    difference = payload.get("difference_read")
    if difference and "menu_survivors" in difference:
        lines += ["", "## Difference read against GunPointAgeSpan", "",
                  "- operators that reached a legal receipt: %s"
                  % difference["operators_that_reached_a_legal_receipt"],
                  "- Episode relations: %s" % difference["episode_relations"],
                  "- menu survivors on this Target: %s"
                  % difference["menu_survivors"],
                  "- hampel_filter here: %s"
                  % json.dumps(difference["hampel_filter_on_this_target"],
                               ensure_ascii=False)]
    lines += ["", "## Budget", "",
              "- LLM: %s of %s" % (ledger.get("llm_calls"),
                                   ledger.get("llm_cap")),
              "- Consumer fits: %s of %s" % (ledger.get("consumer_fits"),
                                             ledger.get("consumer_fit_cap")),
              "- wall clock: %s s" % ledger.get("wall_seconds"),
              "", "## Obligations", ""]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


R2_PREREGISTRATION = (
    {"id": "P1",
     "claim": "after the fix, hampel_filter or repair_level_shift enters the "
              "Source probe order"},
    {"id": "P2",
     "claim": "and may form a POSITIVE Source Episode"},
    {"id": "P3",
     "claim": "the Slow audit may then authorize a non-empty TRY, giving A5 a "
              "real prior"},
    {"id": "P4",
     "claim": "on the Target, hampel_filter can become a non-identity "
              "candidate and compete for a Target-local Skill"},
)


def r2_annotate() -> int:
    """0-LLM pass: score the pre-registration and read the A5-vs-A3 mechanism.

    Everything below is derived from the artifact the run already wrote.  It
    adds no reading and re-runs no arm; it exists so the falsifiable claims the
    book registered before the run are scored in the same file as the result.
    """
    if not R2_RUN_JSON.is_file():
        print(json.dumps({"verdict": "INSTRUMENT_UNREADABLE",
                          "reason": "no r2 artifact to annotate"}, indent=1))
        return 1
    payload = json.loads(R2_RUN_JSON.read_text(encoding="utf-8"))
    part_b = payload.get("part_b") or {}
    part_c = payload.get("part_c") or {}
    source_rounds = part_b.get("rounds") or []
    target_rounds = part_c.get("rounds") or []
    watched = ("hampel_filter", "repair_level_shift")

    def probed_ops(rounds: Sequence[Mapping[str, Any]]) -> set[str]:
        out: set[str] = set()
        for record in rounds:
            for episode in record["episodes"]:
                out.add(str(episode["workflow_signature"]))
        return out

    source_ops = probed_ops(source_rounds)
    counts = dict(part_b.get("episode_counts_by_relation") or {})
    authorized = list(
        (part_b.get("consolidation") or {}).get("authorized_try_operators")
        or [])
    target_ops = probed_ops(target_rounds)
    skills = [record["approved_skill_id"] for record in target_rounds
              if record.get("approved_skill_id")]
    results = {
        "P1": {"held": bool(source_ops & set(watched)),
               "observed": sorted(source_ops & set(watched)),
               "all_source_operators_probed": sorted(source_ops)},
        "P2": {"held": int(counts.get("POSITIVE") or 0) > 0,
               "observed": counts},
        "P3": {"held": bool(authorized), "observed": authorized,
               "try_clause": (part_b.get("source_skill_sections") or {}).get(
                   "TRY")},
        "P4": {"held": "hampel_filter" in target_ops and bool(skills),
               "hampel_probed_on_target": "hampel_filter" in target_ops,
               "target_local_skills_approved": skills},
    }
    prereg = [{**dict(row), **results[row["id"]]} for row in R2_PREREGISTRATION]

    a5_rounds = [record for record in target_rounds if record["arm"] == "A5"]
    a3_rounds = [record for record in target_rounds if record["arm"] == "A3"]
    retrieved_everywhere = bool(a5_rounds) and all(
        SOURCE_SKILL_ID in (record.get("retrieved_skill_ids") or ())
        for record in a5_rounds)
    supplied_candidate = any(
        str(candidate).startswith("cand_skill_")
        for record in a5_rounds
        for candidate in (record.get("pool") or ()))
    a5_rejected = [probe for record in a5_rounds
                   for probe in record["probes"]
                   if probe.get("kind") == "verifier_rejected"]
    a5_receipts = [probe for record in a5_rounds
                   for probe in record["probes"]
                   if probe.get("kind") == "probe"]
    a3_winners = [record["winner_program"] for record in a3_rounds
                  if record.get("winner_program")]
    deficits = [dataset for dataset, contrast
                in ((part_c.get("r2_readouts") or {}).get("contrasts")
                    or {}).items()
                if float(contrast["A5_minus_A3_heldout_accuracy"]) < -MATERIAL]
    if not deficits:
        classification = "no_a5_deficit"
    elif not retrieved_everywhere:
        classification = "retrieval_binding_miss"
    elif supplied_candidate:
        classification = "prior_supplied_a_candidate_that_lost"
    else:
        classification = "prior_delivered_but_steered_the_proposal_elsewhere"
    payload["prereg_check"] = {
        "registered_before_the_run": True,
        "items": prereg,
        "held": [row["id"] for row in prereg if row["held"]],
        "falsified": [row["id"] for row in prereg if not row["held"]],
    }
    payload["a5_deficit_mechanism"] = {
        "datasets_with_a5_below_a3": deficits,
        "classification": classification,
        "source_card_retrieved_in_every_a5_round": retrieved_everywhere,
        "source_card_supplied_an_executable_candidate": supplied_candidate,
        "source_card_execution_right": (
            (part_b.get("source_skill_entry") or {}).get("risk_guards")
            or {}).get("execution_right"),
        "a5_verifier_rejected_probes": a5_rejected,
        "a5_legal_receipts": a5_receipts,
        "a3_winning_programs": _plain(a3_winners),
        "reading": (
            "the Source card was retrieved into every A5 round and carries no "
            "frozen Workflow, so it supplied no candidate and could only act "
            "on the proposal stage as text.  A5's proposals went to the "
            "level-shift family on both rounds and the deployment constraint "
            "rejected them, while A3 reached the local-median family in one "
            "round.  This is neither a retrieval miss nor feedback bias: the "
            "prior was delivered, read, and unhelpful."
            if classification
            == "prior_delivered_but_steered_the_proposal_elsewhere" else None),
    }
    R2_RUN_JSON.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    R2_RUN_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"held": payload["prereg_check"]["held"],
                      "falsified": payload["prereg_check"]["falsified"],
                      "a5_deficit": classification}, indent=1))
    return 0


def _r2_fix_report() -> dict[str, Any]:
    return {
        "file": "methods/ttha/scope_executor.py",
        "before": {
            "where": "ScopeExecutor.verify, HEAD 1ba923c lines 153-197",
            "semantics": (
                "verify_candidate is called per training window with "
                "maximum_modified_fraction; any window whose own "
                "modified_fraction exceeds the cap sets "
                "MODIFICATION_FRACTION_EXCEEDED, and passed = not rejected, so "
                "one window vetoes the candidate for the whole cohort"),
        },
        "after": {
            "where": "ScopeExecutor.verify + WindowVerification + __init__",
            "semantics": (
                "modification_fraction_scope='cohort' passes 1.0 as the "
                "per-window cap so the fraction gate cannot fire per window, "
                "accumulates len(modified_indices) and window.size over every "
                "window, and rejects once if the ratio exceeds the cap.  Every "
                "other gate still vetoes per window."),
            "diagnostics_kept": [
                "window_modified_fractions", "windows_over_maximum_fraction",
                "cohort_modified_points", "cohort_total_points",
                "cohort_modified_fraction"],
            "switch": (
                "modification_fraction_scope defaults to 'per_window', so "
                "forecasting, AD and minipipe callers are unchanged; only "
                "this book's executor opts in"),
            "number_unchanged": "maximum_modified_fraction is still 0.10 here",
        },
        "why_not_a_number_change": (
            "0.20 and 'share of rows over the line' were both suggested by "
            "the CLS-OP diagnostic table, so adopting either would import a "
            "result-derived constant.  Changing which quantity the existing "
            "constant measures imports nothing."),
        "geometry_argument": (
            "the contract limits how much of the data a preparation may "
            "rewrite.  With a dozen windows per cohort the per-window and "
            "cohort readings are close; with one window per fit row (42-1260 "
            "here) the per-window rule is dominated by the single worst row."),
        "regression_subset": {
            "selected": [
                "tests/runtime (verify_candidate itself, the shared gate whose "
                "argument this change re-aims)",
                "tests/methods (ScopeExecutor's own package; two files "
                "construct WindowVerification directly)",
                "tests/integration (p4 runner binding + minipipe cycles, the "
                "deployment-constraint consumers)",
                "tests/functional/test_p2_online_route_abstain.py and "
                "tests/functional/test_ordering_card.py (the functional tests "
                "that reach ScopeExecutor.verify on forecast geometry)",
            ],
            "excluded": [
                "the rest of tests/functional and tests/contracts: grep shows "
                "no path to ScopeExecutor.verify or verify_candidate",
                "tests/functional/test_skill_revocation.py: untracked and does "
                "not parse on this interpreter (f-string spanning lines, "
                "line 166) -- a pre-existing collection error, not this change",
            ],
            "before": "40 failed, 170 passed, 3 skipped, 1 xfailed",
            "after": "40 failed, 170 passed, 3 skipped, 1 xfailed",
            "failure_set_identical": True,
            "failure_list_sha256_16": "9947c9ed623279f4",
            "pre_existing_cause": (
                "38 of the 40 are ValueError 'snapshot lock mismatch; run "
                "compiler with --write-lock' from "
                "methods/ttha/harness/compiler.py; the h0 lock was last "
                "regenerated at 29bed7e (2026-08-24 16:27) and "
                "operators/registry.py changed at 5ef9726 (2026-08-25 11:55), "
                "after it.  The lock carries an operator_bundle_sha, so the "
                "CLS-4 operator commit is the likely origin.  Not repaired "
                "here: it is another line's surface and this book is "
                "single-face."),
            "method": (
                "the same subset was run twice with only "
                "methods/ttha/scope_executor.py swapped between its HEAD bytes "
                "and the fixed bytes; nothing was stashed, so the other line's "
                "in-flight files were never touched"),
        },
        "new_unit_tests": {
            "file": "tests/methods/test_scope_executor_cohort_fraction.py",
            "count": 8,
            "result": "8 passed",
            "covers": [
                "default scope is still per_window",
                "unknown scope raises at construction",
                "the one divergence case: per_window rejects what cohort admits",
                "cohort rejects when the aggregate is over",
                "the two scopes agree above every per-window fraction",
                "non-fraction gates still veto per window under cohort scope",
                "diagnostics are produced under both scopes",
                "windows_over_maximum_fraction has no veto power",
            ],
        },
        "commit": "1402b08",
    }


def _r2_obligations() -> dict[str, Any]:
    return {
        "llm_calls": 0,
        "downloads": 0,
        "forbidden_data_untouched": (
            "no Yahoo, NOAA 2025, beyond_17520, NAB or SMD path is opened; "
            "the only data root is data/ucr_task_context"),
        "single_face": (
            "maximum_modified_fraction stays 0.10, maximum_candidates stays "
            "3, selectable semantics and effect distinctness are untouched; "
            "the verifier is modified once and only in the fraction scope"),
        "no_new_runner": (
            "Part B and Part C are entries on the CLS-OP runner, which "
            "already owns the cell builder, the executor subclass and the "
            "Consumer adapter; a second runner would be a second dialect"),
        "artifact_not_committed": True,
        "known_debts_not_paid_here": [
            "verify_candidate.selectable still does not require effect "
            "distinctness, so a numeric no-op is still 'actionable' to the "
            "candidate supply; this book only excludes no-ops from its own "
            "survivor list",
            "run_online_round still records a verifier rejection without its "
            "rejection code",
        ],
    }


def _r2_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload.get("verdict") or {}
    ledger = payload.get("ledger") or {}
    fix = payload.get("verifier_fix") or {}
    lines = [
        "# CLS-OP-r2-prep -- verifier fix, smoke, Program headroom",
        "",
        "protocol: `%s`  evidence grade: **%s**  LLM: 0"
        % (payload.get("protocol_version"), payload.get("evidence_grade")),
        "",
        "## Verdict",
        "",
        "**%s**" % verdict.get("verdict"),
        "",
        str(verdict.get("rule", "")),
        "",
        "next: %s" % verdict.get("next", ""),
        "",
        "## Part A -- verifier fix",
        "",
        "- before: %s" % (fix.get("before") or {}).get("semantics"),
        "- after: %s" % (fix.get("after") or {}).get("semantics"),
        "- switch: %s" % (fix.get("after") or {}).get("switch"),
        "- why not a number change: %s" % fix.get("why_not_a_number_change"),
        "",
        "### Zero regression",
        "",
        "- before: %s" % (fix.get("regression_subset") or {}).get("before"),
        "- after: %s" % (fix.get("regression_subset") or {}).get("after"),
        "- identical failure set: %s (sha %s)"
        % ((fix.get("regression_subset") or {}).get("failure_set_identical"),
           (fix.get("regression_subset") or {}).get("failure_list_sha256_16")),
        "- pre-existing cause: %s"
        % (fix.get("regression_subset") or {}).get("pre_existing_cause"),
        "- new unit tests: %s"
        % (fix.get("new_unit_tests") or {}).get("result"),
        "",
        "## Part B -- smoke: what survives the fixed verifier",
        "",
        "| cell | menu | survivors after fix | survivors before fix | "
        "unblocked by the fix | numeric no-ops | verifier-rejected |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell in (payload.get("part_b_smoke") or {}).get("cells") or []:
        lines.append("| %s | %d | %s | %s | %s | %d | %d |"
                     % (cell["cell"], cell["menu_size"],
                        ", ".join(cell["survivors"]) or "(none)",
                        ", ".join(cell["survivors_under_per_window_scope"])
                        or "(none)",
                        ", ".join(cell["unblocked_by_the_fix"]) or "(none)",
                        len(cell["numeric_no_ops"]),
                        len(cell["verifier_rejected"])))
    lines += ["", "## Part C -- Program headroom census", "",
              "| cell | surface | n | material line | program | d accuracy | "
              "worst class recall d | material+ | guard |",
              "|---|---|---|---|---|---|---|---|---|"]
    census = payload.get("part_c_headroom") or {}
    for cell in (census.get("source_cells") or []) + (
            census.get("target_cells") or []):
        for row in cell["programs"]:
            for surface, reading in (row.get("readings") or {}).items():
                lines.append(
                    "| %s | %s | %d | %.4f | %s | %s | %s | %s | %s |"
                    % (cell["dataset"], surface, reading["rows"],
                       reading["material_line"], row["program"],
                       ("%+.4f" % reading["delta_accuracy"])
                       if reading["delta_accuracy"] is not None else "n/a",
                       ("%+.4f" % reading["worst_class_recall_delta"])
                       if reading["worst_class_recall_delta"] is not None
                       else "n/a",
                       reading["material_positive"],
                       reading["class_guard_pass"]))
    lines += ["", "## Budget", "",
              "- LLM: 0",
              "- Consumer fits: %s of %s"
              % (ledger.get("consumer_fits"), ledger.get("consumer_fit_cap")),
              "- wall clock: %s s" % ledger.get("wall_seconds"),
              "", "## Obligations", ""]
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


def _proposal_family(name: Any) -> str:
    text = str(name or "").strip().lower()
    if not text or text == "identity":
        return "identity"
    if any(marker in text for marker in _LEVEL_SHIFT_FAMILY_MARKERS):
        return "level-shift"
    if any(marker in text for marker in _HAMPEL_FAMILY_MARKERS):
        return "hampel/local-median"
    return "other"


def _non_identity_names(record: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for candidate in record.get("pool") or ():
        if str(candidate) != "identity":
            names.append(str(candidate))
    for probe in record.get("probes") or ():
        candidate = probe.get("candidate_id")
        if candidate and str(candidate) != "identity":
            names.append(str(candidate))
    winner = record.get("winner_program") or ()
    if isinstance(winner, Sequence) and not isinstance(winner, (str, bytes)):
        for step in winner:
            if isinstance(step, Mapping) and step.get("op"):
                names.append(str(step["op"]))
    for episode in record.get("episodes") or ():
        signature = episode.get("workflow_signature")
        if signature and str(signature) not in ("identity", "unknown"):
            names.append(str(signature))
    # Preserve order, drop repeats -- family identity is what matters.
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _families_of(record: Mapping[str, Any]) -> list[str]:
    families = [_proposal_family(name) for name in _non_identity_names(record)]
    return [family for family in families if family != "identity"]


def _round_is_level_shift_only(record: Mapping[str, Any]) -> bool:
    families = _families_of(record)
    return bool(families) and all(family == "level-shift" for family in families)


def _c40_a5_rounds(r2: Mapping[str, Any], *, dataset: str
                   ) -> list[dict[str, Any]]:
    rounds = ((r2.get("part_c") or {}).get("rounds") or [])
    return [dict(record) for record in rounds
            if record.get("arm") == "A5" and record.get("dataset") == dataset]


def _c40_a3_families(r2: Mapping[str, Any], *, dataset: str) -> list[str]:
    rounds = ((r2.get("part_c") or {}).get("rounds") or [])
    families: list[str] = []
    for record in rounds:
        if record.get("arm") == "A3" and record.get("dataset") == dataset:
            families.extend(_families_of(record))
    seen: set[str] = set()
    ordered: list[str] = []
    for family in families:
        if family not in seen:
            seen.add(family)
            ordered.append(family)
    return ordered


def _r2_backend_identity(r2: Mapping[str, Any]) -> dict[str, Any]:
    """Identity the r2 run actually used, reconstructed from its code path.

    The r2 artifact records only ``obligations.backend = "live Fast Agent"``.
    ``--r2-run`` builds Fast through ``_live_backend`` + ``_live_agent``:
    AgictoChatCompletionsBackend at ``NF_BASE_URL``, model ``SLOW_MODEL``
    (the name this runner passes to TTHAAgentCore for the Fast path).
    """
    from evaluation.functional.task_episode_harness.e1 import NF_BASE_URL

    return {
        "artifact_field": "obligations.backend",
        "artifact_value": (r2.get("obligations") or {}).get("backend"),
        "r2_git_head": r2.get("git_head"),
        "r2_entry": r2.get("entry"),
        "code_path": {
            "fast_factory": (
                "evaluation.functional.task_episode_harness.agentic.runner"
                "._default_backend_factory"),
            "transport": "AgictoChatCompletionsBackend",
            "model": SLOW_MODEL,
            "base_url": NF_BASE_URL,
            "agent_constructor": "_live_agent -> TTHAAgentCore(model=SLOW_MODEL)",
        },
        "note": (
            "r2 JSON has no model/base_url fields.  Replay locks the same "
            "unchanged --r2-run factories; no other relay or model is allowed."),
    }


def _probe_locked_backend(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Reachability check on the locked model/URL.  Not charged to the A5 cap."""
    import openai
    from evaluation.functional.task_episode_harness.e1 import NF_BASE_URL

    api_key = next(
        (os.environ.get(name, "").strip()
         for name in ("OPENAI_API_KEY", "AGICTO_API_KEY")
         if os.environ.get(name, "").strip()),
        None,
    )
    expected_model = str((identity.get("code_path") or {}).get("model") or "")
    expected_url = str((identity.get("code_path") or {}).get("base_url") or "")
    if expected_model != SLOW_MODEL or expected_url != NF_BASE_URL:
        return {"ok": False,
                "reason": "locked identity drifted from --r2-run factories"}
    if not api_key:
        return {"ok": False,
                "reason": "OPENAI_API_KEY and AGICTO_API_KEY are both empty"}
    try:
        client = openai.OpenAI(api_key=api_key, base_url=NF_BASE_URL, timeout=60)
        completion = client.chat.completions.create(
            model=SLOW_MODEL,
            messages=[{"role": "user",
                       "content": "Reply with the single word pong."}])
        relay_error = None
        try:
            from SelfEvolvingHarnessTS.runtime.agent_backend import (
                _relay_error_payload,
            )
            relay_error = _relay_error_payload(completion)
        except Exception:  # noqa: BLE001
            relay_error = None
        if relay_error:
            return {"ok": False,
                    "reason": "relay error payload: %s" % relay_error}
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return {"ok": False, "reason": "completion returned no choices"}
        returned = getattr(completion, "model", None)
        return {
            "ok": True,
            "returned_model": returned,
            "completion_id": getattr(completion, "id", None),
            "probe_charged_to_a5_cap": False,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "reason": "%s: %s" % (type(exc).__name__, exc)}


def _snapshot_skill_ids(snapshot: Any) -> list[str]:
    return sorted(str(skill.skill_id) for skill in getattr(snapshot, "skills", ()))


def _view_skill_ids(snapshot: Any, features: Mapping[str, Any], *,
                    role: str) -> list[str]:
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (
        resolve_harness_view,
    )
    view = resolve_harness_view(snapshot, dict(features), role=role)
    return [str(skill.skill_id) for skill in view.skills]


def _a5_arm_caps_from_r2(r2: Mapping[str, Any], *, dataset: str
                         ) -> dict[str, Any]:
    """What the r2 A5 arm actually spent, and the protocol caps it sat under."""
    budgets = r2.get("budgets") or {}
    rounds = _c40_a5_rounds(r2, dataset=dataset)
    all_a5 = [record for record in ((r2.get("part_c") or {}).get("rounds") or [])
              if record.get("arm") == "A5"]
    cell_llm = sum(int(record.get("llm_calls_this_round") or 0)
                   for record in rounds)
    arm_llm = sum(int(record.get("llm_calls_this_round") or 0)
                  for record in all_a5)
    return {
        "protocol_llm_cap": int(budgets.get("llm_cap") or R2_LLM_CAP),
        "protocol_llm_fast_cap": int(budgets.get("llm_fast_cap")
                                     or (R2_LLM_CAP - LLM_SLOW_RESERVE)),
        "protocol_llm_slow_reserve": int(budgets.get("llm_slow_reserve")
                                         or LLM_SLOW_RESERVE),
        "protocol_fit_cap": int(budgets.get("consumer_fit_cap")
                                or CONSUMER_FIT_CAP),
        "r2_a5_gunpoint_llm": cell_llm,
        "r2_a5_all_targets_llm": arm_llm,
        "replay_uses_protocol_caps": True,
        "note": (
            "Replay keeps the r2 protocol Fast/fit caps (not a newly invented "
            "tighter A5-only number).  A5-only will underspend them."),
    }


def _legacy_deficit_mislabel(rounds: Sequence[Mapping[str, Any]]
                             ) -> dict[str, Any]:
    """What r2_annotate's classifier would say.  Not invoked; not modified.

    That classifier (run_e2_t6_cls_op_shared_harness.py ~3675-3701) maps
    'card absent from retrieved_skill_ids' to retrieval_binding_miss.  After
    the visibility fix that absence is intentional withholding.
    """
    retrieved_everywhere = bool(rounds) and all(
        SOURCE_SKILL_ID in (record.get("retrieved_skill_ids") or ())
        for record in rounds)
    if retrieved_everywhere:
        would_say = "prior_delivered_but_steered_the_proposal_elsewhere"
        nature = "not_a_mislabel_on_this_trace"
    else:
        would_say = "retrieval_binding_miss"
        nature = (
            "MISLABEL: the card is withheld by the Fast-visibility predicate, "
            "not a retrieval-binding failure")
    return {
        "classifier_location": (
            "evaluation/functional/run_e2_t6_cls_op_shared_harness.py:"
            "r2_annotate a5_deficit_mechanism"),
        "would_classify_as": would_say,
        "nature": nature,
        "classifier_modified": False,
        "shared_runner_logic_modified": False,
    }


def _r2_replay_mechanism_verdict(
    *,
    rounds: Sequence[Mapping[str, Any]],
    deployment: Mapping[str, Any] | None,
    a3_families: Sequence[str],
    install: Mapping[str, Any],
) -> dict[str, Any]:
    fast_views = [list(record.get("retrieved_skill_ids") or ())
                  for record in rounds]
    if deployment is not None:
        fast_views.append(list(deployment.get("view_skill_ids") or ()))
    present = [SOURCE_SKILL_ID in ids for ids in fast_views]
    card_absent = bool(fast_views) and not any(present)
    c40_pattern = bool(rounds) and all(
        _round_is_level_shift_only(record) for record in rounds)
    legal_receipts = [
        {"round": record.get("round"), "probe": probe}
        for record in rounds
        for probe in (record.get("probes") or ())
        if probe.get("kind") == "probe"]
    replay_families: list[str] = []
    for record in rounds:
        replay_families.extend(_families_of(record))
    a3_type = bool(set(replay_families) & set(a3_families))
    exploration_restored = bool(legal_receipts) or a3_type
    if any(present):
        verdict = "VISIBILITY_INVARIANT_VIOLATED"
    elif card_absent and c40_pattern:
        verdict = "EXPLORATION_NOT_RESTORED"
    elif card_absent and (not c40_pattern) and exploration_restored:
        verdict = "VISIBILITY_INVARIANT_HOLDS"
    elif card_absent and (not c40_pattern):
        # Card withheld and the C40 lock is gone, but neither restoration
        # witness fired.  Named as HOLDS on the visibility half; the false
        # exploration_restored flag is the remainder, not a second leak path.
        verdict = "VISIBILITY_INVARIANT_HOLDS"
    else:
        verdict = "INSTRUMENT_UNREADABLE"
    return {
        "verdict": verdict,
        "claim_limit": (
            "MECHANISM.  Tests the inertness invariant on the exposed C40 "
            "ledger after the Fast-visibility fix.  Held-out accuracy is "
            "recorded, not judged.  No capability claim."),
        "card_absent_every_fast_view": card_absent,
        "card_present_in_any_fast_view": bool(any(present)),
        "fast_views_checked": len(fast_views),
        "c40_level_shift_only_pattern": c40_pattern,
        "legal_support_receipts": legal_receipts,
        "legal_support_receipt_count": len(legal_receipts),
        "replay_proposal_families": replay_families,
        "a3_cold_start_families": list(a3_families),
        "proposal_families_a3_same_type": a3_type,
        "exploration_restored": exploration_restored,
        "source_card_installed": bool(install.get("card_in_store")),
        "source_card_in_slow_view": bool(install.get("card_in_slow_view")),
        "source_card_in_preflight_fast_view": bool(
            install.get("card_in_preflight_fast_view")),
    }


def _r2_replay_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload.get("verdict") or {}
    ledger = payload.get("ledger") or {}
    lock = payload.get("backend_lock") or {}
    install = payload.get("source_install") or {}
    part_c = payload.get("part_c") or {}
    comparison = payload.get("proposal_family_comparison") or []
    mislabel = payload.get("retrieval_binding_miss_note") or {}
    lines = [
        "# CLS-OP-r2 A5 mechanism replay (inertness invariant)",
        "",
        "protocol: `%s`  evidence grade: **MECHANISM**"
        % payload.get("protocol_version"),
        "",
        "## Verdict",
        "",
        "**%s**" % verdict.get("verdict"),
        "",
        verdict.get("claim_limit", ""),
        "",
        "- card absent every Fast view: %s"
        % verdict.get("card_absent_every_fast_view"),
        "- C40 level-shift-only pattern: %s"
        % verdict.get("c40_level_shift_only_pattern"),
        "- legal Support receipts: %s"
        % verdict.get("legal_support_receipt_count"),
        "- proposal families A3-same-type: %s"
        % verdict.get("proposal_families_a3_same_type"),
        "- source card installed / Slow-visible: %s / %s"
        % (verdict.get("source_card_installed"),
           verdict.get("source_card_in_slow_view")),
        "",
        "## Backend lock",
        "",
        "- artifact field `%s` = %s"
        % (lock.get("artifact_field"), lock.get("artifact_value")),
        "- r2 git_head: %s" % lock.get("r2_git_head"),
        "- locked model: %s" % ((lock.get("code_path") or {}).get("model")),
        "- locked base_url: %s"
        % ((lock.get("code_path") or {}).get("base_url")),
        "- probe: %s" % ((payload.get("backend_probe") or {}).get("ok")),
        "",
        "## Source card install",
        "",
        "- skill_id: %s" % install.get("skill_id"),
        "- store skill ids: %s" % install.get("store_skill_ids"),
        "- preflight Fast view: %s" % install.get("preflight_fast_view"),
        "- preflight Slow view: %s" % install.get("preflight_slow_view"),
        "",
        "## Per-round comparison (replay vs C40 A5)",
        "",
        "| dataset | round | side | retrieved contains card | "
        "non-identity pool | family | probe kind | Support | delayed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in comparison:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (row.get("dataset"), row.get("round"), row.get("side"),
               row.get("card_in_retrieved"),
               row.get("non_identity_pool"),
               row.get("families"),
               row.get("probe_kinds"),
               row.get("support_receipts"),
               row.get("delayed_utility")))
    deployment = None
    for item in part_c.get("deployments") or []:
        deployment = item
    lines += [
        "",
        "## Held-out (information only)",
        "",
        "- accuracy: %s" % (
            None if deployment is None else deployment.get("heldout_accuracy")),
        "- vs identity: %s" % (
            None if deployment is None
            else deployment.get("heldout_accuracy_gain")),
        "- applied: %s" % (
            "identity" if deployment is None
            else (deployment.get("applied_program") or "identity")),
        "- deploy source: %s" % (
            None if deployment is None else deployment.get("deploy_source")),
        "",
        "## retrieval_binding_miss mislabel",
        "",
        "- the r2_annotate classifier would say: **%s**"
        % mislabel.get("would_classify_as"),
        "- nature: %s" % mislabel.get("nature"),
        "- classifier modified: %s" % mislabel.get("classifier_modified"),
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
        "## Obligations",
        "",
    ]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


def _write_replay_artifacts(payload: Mapping[str, Any]) -> None:
    R2_REPLAY_JSON.parent.mkdir(parents=True, exist_ok=True)
    R2_REPLAY_JSON.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False,
                   sort_keys=False) + "\n",
        encoding="utf-8")
    R2_REPLAY_MD.write_text(_r2_replay_markdown(payload), encoding="utf-8")


def r2_replay_a5(*, run_id: str = R2_REPLAY_RUN_ID) -> int:
    """Replay only the C40 A5 arm on the exposed GunPointAgeSpan ledger.

    The unauthorized Source card is installed into a fresh store exactly as
    r2 materialized it.  The visibility fix is the only intended change.
    """
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )

    started = time.time()
    if not R2_RUN_JSON.is_file():
        payload = {
            "protocol_version": R2_REPLAY_PROTOCOL,
            "run_id": run_id,
            "entry": "--r2-replay-a5",
            "evidence_grade": "MECHANISM",
            "verdict": {"verdict": "INSTRUMENT_UNREADABLE",
                        "reason": "r2 artifact missing"},
        }
        _write_replay_artifacts(payload)
        print(json.dumps(payload["verdict"], indent=1))
        return 1

    r2 = json.loads(R2_RUN_JSON.read_text(encoding="utf-8"))
    entry = (r2.get("part_b") or {}).get("source_skill_entry")
    identity = _r2_backend_identity(r2)
    caps = _a5_arm_caps_from_r2(r2, dataset=R2_REPLAY_DATASET)
    c40_rounds = _c40_a5_rounds(r2, dataset=R2_REPLAY_DATASET)
    a3_families = _c40_a3_families(r2, dataset=R2_REPLAY_DATASET)

    payload: dict[str, Any] = {
        "protocol_version": R2_REPLAY_PROTOCOL,
        "run_id": run_id,
        "entry": "--r2-replay-a5",
        "modification_fraction_scope": "cohort",
        "evidence_grade": "MECHANISM",
        "claim_limit": (
            "MECHANISM only.  Governance-fix replay; no capability reading."),
        "git_head": _git("rev-parse", "HEAD"),
        "r2_source": {
            "path": str(R2_RUN_JSON.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"),
            "run_id": r2.get("run_id"),
            "protocol_version": r2.get("protocol_version"),
            "git_head": r2.get("git_head"),
        },
        "backend_lock": identity,
        "a5_budget_from_r2": caps,
        "c40_a5_reference": [{
            "round": record.get("round"),
            "dataset": record.get("dataset"),
            "retrieved_skill_ids": record.get("retrieved_skill_ids"),
            "pool": record.get("pool"),
            "chosen": record.get("chosen"),
            "families": _families_of(record),
            "probes": record.get("probes"),
            "support_receipts": record.get("support_receipts"),
            "delayed_utility": record.get("delayed_utility"),
            "abstained": record.get("abstained"),
        } for record in c40_rounds],
        "wiring_check": _wiring_check(),
        "preflight": _preflight(),
        "binding": {
            "dataset": R2_REPLAY_DATASET,
            "target_condition": TARGET_CONDITION,
            "held_in_rounds": list(HELD_IN_ROUNDS),
            "support_trial_budget_per_round": SUPPORT_TRIAL_BUDGET,
            "maximum_candidates": 1 + SUPPORT_TRIAL_BUDGET,
            "modification_fraction_scope": "cohort",
            "source_skill_id": SOURCE_SKILL_ID,
            "source_card_origin": "r2 part_b.source_skill_entry (same ledger)",
        },
        "budgets": {
            "llm_cap": caps["protocol_llm_cap"],
            "llm_fast_cap": caps["protocol_llm_fast_cap"],
            "llm_slow_reserve": caps["protocol_llm_slow_reserve"],
            "consumer_fit_cap": caps["protocol_fit_cap"],
        },
    }

    def stop_write(verdict: str, reason: str, extra: Mapping[str, Any] | None = None
                   ) -> int:
        payload["stop"] = {"verdict": verdict, "reason": reason}
        payload["verdict"] = {"verdict": verdict, "reason": reason}
        if extra:
            payload.update(dict(extra))
        payload["ledger"] = {
            "llm_calls_fast": 0, "llm_calls_slow": 0, "llm_calls_total": 0,
            "llm_cap": caps["protocol_llm_cap"],
            "consumer_fits": 0,
            "consumer_fit_cap": caps["protocol_fit_cap"],
            "wall_seconds": round(time.time() - started, 1),
        }
        payload["obligations"] = {
            "methods_package_unmodified": True,
            "runtime_contracts_operators_unmodified": True,
            "other_line_files_untouched": True,
            "backend": "locked r2 Fast path; no substitute relay",
            "data": "only the r2 local GunPointAgeSpan substrate",
        }
        _write_replay_artifacts(payload)
        print(json.dumps({"verdict": verdict, "reason": reason}, indent=1))
        return 1

    if not isinstance(entry, Mapping) or entry.get("skill_id") != SOURCE_SKILL_ID:
        return stop_write("INSTRUMENT_UNREADABLE",
                          "r2 source_skill_entry missing or wrong skill_id")
    if identity.get("artifact_value") != "live Fast Agent":
        return stop_write(
            "BACKEND_UNAVAILABLE",
            "r2 artifact backend is %r, not 'live Fast Agent'"
            % identity.get("artifact_value"))

    probe = _probe_locked_backend(identity)
    payload["backend_probe"] = probe
    if not probe.get("ok"):
        return stop_write("BACKEND_UNAVAILABLE",
                          str(probe.get("reason") or "locked backend probe failed"),
                          extra={"backend_probe": probe})

    fit_budget = FitBudget(caps["protocol_fit_cap"])
    llm_ledger = {"fast": 0, "slow": 0}
    tag = run_id
    store_root = Path(tempfile.gettempdir()) / tag
    if store_root.exists():
        shutil.rmtree(store_root)
    stopped: str | None = None
    try:
        h0 = compile_snapshot(
            PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
            verify_lock=False)
        source_snapshot = _materialize_source_snapshot(entry, h0, store_root)
        cell = _build_cell(R2_REPLAY_DATASET, TARGET_CONDITION)
        block = np.asarray(cell["observation_block"], dtype=np.float64)
        features = dict(extract_public_features(block, task_kind="classification"))
        store_ids = _snapshot_skill_ids(source_snapshot)
        fast_ids = _view_skill_ids(source_snapshot, features, role="fast")
        slow_ids = _view_skill_ids(source_snapshot, features, role="slow")
        install = {
            "skill_id": SOURCE_SKILL_ID,
            "store_skill_ids": store_ids,
            "card_in_store": SOURCE_SKILL_ID in store_ids,
            "preflight_fast_view": fast_ids,
            "preflight_slow_view": slow_ids,
            "card_in_preflight_fast_view": SOURCE_SKILL_ID in fast_ids,
            "card_in_slow_view": SOURCE_SKILL_ID in slow_ids,
            "h0_runtime_bundle_sha": h0.runtime_bundle_sha,
            "source_runtime_bundle_sha": source_snapshot.runtime_bundle_sha,
            "r2_h0_runtime_bundle_sha": (
                (r2.get("part_b") or {}).get("snapshots") or {}
            ).get("h0_runtime_bundle_sha"),
            "sha_note": (
                "h0 runtime_bundle_sha may differ from r2 because T1 regenerated "
                "the lock after changing retrieval.py; harness_content_sha is "
                "unchanged.  The installed card bytes are the r2 ledger entry."),
        }
        payload["source_install"] = install
        payload["part_b"] = {
            "source_skill_entry": entry,
            "source_skill_sections": (
                (r2.get("part_b") or {}).get("source_skill_sections")),
            "reused_r2_card": True,
            "slow_not_re_run": True,
        }
        if not install["card_in_store"] or not install["card_in_slow_view"]:
            raise Stop("INSTRUMENT_UNREADABLE",
                       "Source card failed to install into the A5 store/Slow view")

        backend = _live_backend(caps["protocol_llm_fast_cap"])
        state = _new_arm_state(
            snapshot=source_snapshot,
            agent=_live_agent(cell["observation_block"],
                              backend.new_arm_backend()),
            store_root=store_root,
            tag="target_%s_A5" % R2_REPLAY_DATASET)
        arm_rounds: list[dict[str, Any]] = []
        for round_name in HELD_IN_ROUNDS:
            record = _run_round(
                state=state, cell=cell, round_name=round_name,
                arm="A5", fit_budget=fit_budget,
                allow_fast_skill=True,
                fraction_scope="cohort", ledger=backend)
            arm_rounds.append(record)
            print("A5 %-28s %s retrieved=%s probes=%s winner=%s delayed=%s"
                  % (R2_REPLAY_DATASET, round_name,
                     record["retrieved_skill_ids"],
                     record["probes"],
                     record["winner_program"],
                     record["delayed_utility"]),
                  flush=True)
        deployment = _deploy_and_score(
            state=state, cell=cell, arm="A5", fit_budget=fit_budget)
        print("DEPLOY A5 %-28s %-32s heldout_acc=%.4f gain=%+.4f"
              % (R2_REPLAY_DATASET, deployment["deploy_source"],
                 deployment["heldout_accuracy"],
                 deployment["heldout_accuracy_gain"]),
              flush=True)
        llm_ledger["fast"] = int(backend.calls)
        payload["part_c"] = {
            "cells": [{k: v for k, v in cell.items()
                       if k not in ("fit_values", "fit_labels", "surfaces",
                                    "observation_block")}],
            "rounds": arm_rounds,
            "deployments": [deployment],
            "arm_table": _arm_table(arm_rounds, [deployment]),
            "deploy_purity": _deploy_purity([deployment]),
            "r2_readouts": _r2_readouts(arm_rounds, [deployment]),
        }
        comparison = []
        by_round = {record["round"]: record for record in arm_rounds}
        for record in c40_rounds:
            comparison.append({
                "dataset": record.get("dataset"),
                "round": record.get("round"),
                "side": "c40_a5",
                "card_in_retrieved": SOURCE_SKILL_ID in (
                    record.get("retrieved_skill_ids") or ()),
                "non_identity_pool": _non_identity_names(record),
                "families": _families_of(record),
                "probe_kinds": [p.get("kind")
                                for p in (record.get("probes") or ())],
                "support_receipts": record.get("support_receipts"),
                "delayed_utility": record.get("delayed_utility"),
            })
            replay = by_round.get(record["round"])
            if replay is not None:
                comparison.append({
                    "dataset": replay.get("dataset"),
                    "round": replay.get("round"),
                    "side": "replay_a5",
                    "card_in_retrieved": SOURCE_SKILL_ID in (
                        replay.get("retrieved_skill_ids") or ()),
                    "non_identity_pool": _non_identity_names(replay),
                    "families": _families_of(replay),
                    "probe_kinds": [p.get("kind")
                                    for p in (replay.get("probes") or ())],
                    "support_receipts": replay.get("support_receipts"),
                    "delayed_utility": replay.get("delayed_utility"),
                })
        payload["proposal_family_comparison"] = comparison
        payload["retrieval_binding_miss_note"] = _legacy_deficit_mislabel(
            arm_rounds)
        payload["verdict"] = _r2_replay_mechanism_verdict(
            rounds=arm_rounds, deployment=deployment,
            a3_families=a3_families, install=install)
    except Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
        payload["verdict"] = {"verdict": stop.verdict, "reason": stop.reason}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {"verdict": stopped,
                           "reason": "%s: %s" % (type(exc).__name__, exc),
                           "traceback": traceback.format_exc()}
        payload["verdict"] = payload["stop"]

    payload["ledger"] = {
        "llm_calls_fast": llm_ledger["fast"],
        "llm_calls_slow": llm_ledger["slow"],
        "llm_calls_total": llm_ledger["fast"] + llm_ledger["slow"],
        "llm_cap": caps["protocol_llm_cap"],
        "llm_within_cap": (llm_ledger["fast"] + llm_ledger["slow"]
                           <= caps["protocol_llm_cap"]),
        "consumer_fits": fit_budget.used,
        "consumer_fit_cap": fit_budget.cap,
        "wall_seconds": round(time.time() - started, 1),
        "backend_probe_llm_calls": 1,
        "backend_probe_charged_to_a5_cap": False,
        "r2_a5_gunpoint_llm": caps["r2_a5_gunpoint_llm"],
        "r2_a5_all_targets_llm": caps["r2_a5_all_targets_llm"],
    }
    payload["obligations"] = {
        "methods_package_unmodified": True,
        "runtime_contracts_operators_unmodified": True,
        "other_line_files_untouched": True,
        "forbidden_data_untouched": (
            "no Yahoo, NOAA 2025, beyond_17520, NAB or SMD path is opened; "
            "the only data root is data/ucr_task_context"),
        "source_card_reused_from_r2_ledger": True,
        "slow_not_re_run": True,
        "deficit_classifier_unmodified": True,
        "backend": (
            "locked r2 Fast path %s @ %s"
            % (SLOW_MODEL, identity["code_path"]["base_url"])),
        "full_repo_pytest_not_run": True,
        "downloads": 0,
    }
    _write_replay_artifacts(payload)
    print(json.dumps({
        "verdict": (payload.get("verdict") or {}).get("verdict"),
        "llm": payload["ledger"]["llm_calls_total"],
        "fits": payload["ledger"]["consumer_fits"],
        "artifact": str(R2_REPLAY_JSON),
    }, ensure_ascii=False, indent=1))
    verdict_name = str((payload.get("verdict") or {}).get("verdict") or "")
    return 0 if verdict_name in (
        "VISIBILITY_INVARIANT_HOLDS",
        "VISIBILITY_INVARIANT_VIOLATED",
        "EXPLORATION_NOT_RESTORED",
    ) and stopped is None else 1


# =========================================================================== #
# CLS-DEV-WINE -- v2 injection precheck, then optional development lifecycle
# =========================================================================== #
WINE_DATASET = "Wine"
WINE_PRECHECK_JSON = E2 / "t6_cls_dev_wine_precheck.json"
WINE_PRECHECK_MD = E2 / "t6_cls_dev_wine_precheck.md"
WINE_OUT_JSON = E2 / "t6_cls_dev_wine.json"
WINE_OUT_MD = E2 / "t6_cls_dev_wine.md"
WINE_PROTOCOL = "t6_cls_dev_wine_v1"
WINE_RUN_ID = "t6_cls_dev_wine_run1"
WINE_WALL_SECONDS = 90 * 60
WINE_EVIDENCE_GRADE = "development"
ECG200_REFERENCE_LENGTH = 96
WINE_PRECHECK_REQUIRED = (
    "identity", "hampel_filter", "outlier_mad", "outlier_iqr",
)
WINE_HONESTY = (
    "Wine was previously used by the action_credit line "
    "(run_e2_action_credit_candidate_ordering.py) under the same impulse "
    "condition pair (audit: artifacts/functional/e2/"
    "t6_cls_conf_r3_selection.json). This run is therefore not an "
    "independent confirmation. Every judgement stays at "
    "evidence_grade=development. The label CLS_CHAIN_CONFIRMED must not "
    "be used."
)


def _v2_invariance_at_150() -> dict[str, Any]:
    """At GunPoint length 150, v2 constants and inject bytes equal v1."""
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_context_label_evidence_witness import (
        SPIKE_AMPLITUDE,
        SPIKE_FRACTIONS,
    )

    _ctx, helpers = _legacy_helpers()
    length = GUNPOINT_POSITIVE_CONTROL_LENGTH
    segment = _v2_segment_length(length)
    positions = tuple(int(p) for p in helpers["positions"](length))
    values = np.tile(
        np.linspace(-1.0, 1.0, length, dtype=np.float64), (6, 1))
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    v1 = helpers["inject"](np, values, labels, positions)
    v2 = _inject_v2(np, values, labels, positions)
    checks = {
        "segment_length_equals_v1": segment == V1_SPIKE_SEGMENT_LENGTH,
        "positions_equal": True,
        "amplitude_equals_v1": float(SPIKE_AMPLITUDE) == 16.0,
        "fractions_equal_v1": tuple(SPIKE_FRACTIONS) == (0.08, 0.20, 0.80, 0.92),
        "injected_values_equal": bool(np.array_equal(v1, v2)),
    }
    return {
        "length": length,
        "v1_segment_length": V1_SPIKE_SEGMENT_LENGTH,
        "v2_segment_length": segment,
        "positions": list(positions),
        "source_constants": (
            "evaluation/functional/"
            "run_e2_task_context_label_evidence_witness.py:37 "
            "SPIKE_FRACTIONS=(0.08, 0.20, 0.80, 0.92); :38 "
            "SPIKE_AMPLITUDE=16.0; :95-100 _inject writes one point "
            "per position so v1 segment length = 1"),
        "formula": "round(%d/%d * L)"
                   % (V1_SPIKE_SEGMENT_LENGTH,
                      GUNPOINT_POSITIVE_CONTROL_LENGTH),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _v2_geometry_derivation(series_length: int, *,
                            hampel_window: int,
                            label: str) -> dict[str, Any]:
    """Constant-only occupancy and theoretical hampel fraction.

    Injected points = n_spikes * v2_segment (no-overlap, proved from
    positions).  Hampel theoretical upper bound per spike is
    segment + window - 1 (the segment plus a window-3 halo).  ECG200
    v1 measured 0.128 against this bound of 12/96 = 0.125.
    """
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_context_label_evidence_witness import (
        _bound_positions,
        SPIKE_FRACTIONS,
    )

    length = int(series_length)
    segment = _v2_segment_length(length)
    n_spikes = len(SPIKE_FRACTIONS)
    positions = tuple(int(p) for p in _bound_positions(length))
    gaps = [positions[i + 1] - positions[i]
            for i in range(len(positions) - 1)]
    injected = n_spikes * segment
    injection_fraction = injected / float(length)
    halo = segment + int(hampel_window) - 1
    hampel_points = n_spikes * halo
    hampel_fraction = hampel_points / float(length)
    cap = float(
        _task_context().deployment_constraints.maximum_modified_fraction)
    return {
        "label": label,
        "series_length": length,
        "v2_segment_length": segment,
        "formula": "round(%d/%d * L)"
                   % (V1_SPIKE_SEGMENT_LENGTH,
                      GUNPOINT_POSITIVE_CONTROL_LENGTH),
        "n_spikes": n_spikes,
        "positions": list(positions),
        "min_gap": min(gaps) if gaps else None,
        "segments_overlap": bool(gaps) and min(gaps) < segment,
        "injected_points": injected,
        "injection_fraction": injection_fraction,
        "injection_fraction_exact": "%d/%d" % (injected, length),
        "hampel_window": int(hampel_window),
        "hampel_halo_per_spike": halo,
        "hampel_theoretical_modified_points_upper_bound": hampel_points,
        "hampel_theoretical_modified_fraction": hampel_fraction,
        "hampel_theoretical_fraction_exact": "%d/%d" % (hampel_points, length),
        "modification_cap": cap,
        "injection_below_cap": injection_fraction < cap,
        "hampel_theoretical_below_cap": hampel_fraction < cap,
    }


def _wine_heldin_pool(cell: Mapping[str, Any]) -> tuple[Any, Any]:
    """Concatenate the four held-in slices: the full support pool."""
    values: list[Any] = []
    labels: list[Any] = []
    for name in ("r1_support", "r1_delayed", "r2_support", "r2_delayed"):
        block, labs = cell["surfaces"][name]
        values.append(np.asarray(block, dtype=np.float64))
        labels.append(np.asarray(labs))
    return np.concatenate(values), np.concatenate(labels)


def _wine_precheck_pass(cell: Mapping[str, Any], *,
                        fit_budget: FitBudget) -> dict[str, Any]:
    """0-LLM legality + held-in headroom census on one v2 cell."""
    heldin_values, heldin_labels = _wine_heldin_pool(cell)
    n_heldin = int(heldin_labels.size)
    line = max(MATERIAL, 1.0 / max(n_heldin, 1))
    block = np.asarray(cell["observation_block"], dtype=np.float64)
    support_origin = int(block.size)
    delayed_origin = support_origin + 1
    heldout_origin = support_origin + 2
    adapter = ClassificationConsumerAdapter(
        fit_values=cell["fit_values"], fit_labels=cell["fit_labels"],
        surfaces={SUPPORT: (heldin_values, heldin_labels)},
        delayed_origin=delayed_origin, heldout_origin=heldout_origin,
        budget=fit_budget, ridge_alpha=RIDGE_ALPHA,
        allowed_surfaces=(SUPPORT,))
    cap = float(
        _task_context().deployment_constraints.maximum_modified_fraction)
    executor = _ClsScopeExecutor(
        cell=cell, evaluate_fn=adapter, max_modified_fraction=cap,
        modification_fraction_scope="cohort")

    rows: list[dict[str, Any]] = [{
        "program": "identity",
        "params": {},
        "legal": True,
        "verifier_passed": True,
        "cohort_modified_fraction": 0.0,
        "cohort_modified_points": 0,
        "cohort_total_points": int(np.asarray(cell["fit_values"]).size),
        "rejection_codes": [],
        "numeric_no_op": True,
        "headroom": 0.0,
        "delta_accuracy": 0.0,
        "per_class_recall_delta": [],
        "worst_class_recall_delta": 0.0,
        "scored": False,
        "skip_reason": "identity_headroom_is_zero_by_definition",
    }]
    for entry in _r2_menu():
        steps = ((entry["program"], dict(entry["params"])),)
        verification = executor.verify(steps, support_origin)
        no_op = bool(
            verification.checked_windows
            and verification.identity_equivalent_windows
            == verification.checked_windows)
        record: dict[str, Any] = {
            "program": entry["program"],
            "params": entry["params"],
            "legal": bool(verification.passed),
            "verifier_passed": bool(verification.passed),
            "cohort_modified_fraction": verification.cohort_modified_fraction,
            "cohort_modified_points": verification.cohort_modified_points,
            "cohort_total_points": verification.cohort_total_points,
            "windows_over_per_window_cap": (
                verification.windows_over_maximum_fraction),
            "checked_windows": verification.checked_windows,
            "rejection_codes": sorted({
                str(row["rejection_code"])
                for row in verification.rejected_windows}),
            "numeric_no_op": no_op,
            "headroom": None,
            "delta_accuracy": None,
            "per_class_recall_delta": None,
            "worst_class_recall_delta": None,
            "scored": False,
        }
        if not record["legal"]:
            record["skip_reason"] = (
                "verifier_rejected:" + ",".join(record["rejection_codes"]))
        elif no_op:
            record["headroom"] = 0.0
            record["delta_accuracy"] = 0.0
            record["per_class_recall_delta"] = []
            record["worst_class_recall_delta"] = 0.0
            record["skip_reason"] = "numeric_no_op"
        else:
            receipt = executor.evaluate(steps, support_origin)
            delta = (float(receipt.gain) if receipt.gain is not None
                     else None)
            recalls = [float(value) for value in receipt.per_view_gain]
            record["scored"] = True
            record["headroom"] = delta
            record["delta_accuracy"] = delta
            record["per_class_recall_delta"] = recalls
            record["worst_class_recall_delta"] = (
                min(recalls) if recalls else None)
            record["evaluate_error"] = receipt.error
        rows.append(record)

    missing = [name for name in WINE_PRECHECK_REQUIRED
               if not any(row["program"] == name for row in rows)]
    if missing:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "precheck menu missing required operators: %s" % missing)
    hampel = next(row for row in rows if row["program"] == "hampel_filter")
    hampel_headroom = hampel.get("headroom")
    gate_pass = bool(
        hampel["legal"]
        and hampel_headroom is not None
        and float(hampel_headroom) >= line)
    return {
        "dataset": cell["dataset"],
        "condition": cell["condition"],
        "injection_template": INJECTION_TEMPLATE_V2,
        "fit_rows": cell["fit_rows"],
        "series_length": cell["series_length"],
        "n_heldin": n_heldin,
        "material_line": line,
        "maximum_modified_fraction": cap,
        "modification_fraction_scope": "cohort",
        "consumer": "ridge-raw-plus-difference-v1",
        "metric": "accuracy",
        "heldin_surface": "full_support_pool",
        "programs": rows,
        "hampel_legal": bool(hampel["legal"]),
        "hampel_headroom": hampel_headroom,
        "hampel_modified_fraction": hampel.get("cohort_modified_fraction"),
        "hampel_rejection_codes": hampel.get("rejection_codes") or [],
        "gate": {
            "rule": (
                "PASS iff hampel_filter is legal under the cohort verifier "
                "(cap 0.10) and its held-in headroom vs identity is "
                ">= max(0.005, 1/n_heldin).  Otherwise "
                "FAMILY_CLOSURE_RECOMMENDED; do not spend LLM, do not "
                "change the substrate."),
            "n_heldin": n_heldin,
            "material_line": line,
            "passed": gate_pass,
            "verdict": (
                "PASS" if gate_pass else "FAMILY_CLOSURE_RECOMMENDED"),
        },
        "consumer_fits_after": fit_budget.used,
    }


def _dev_wine_census() -> dict[str, Any]:
    """Local already-used Wine zip.  Not an unused-Target census."""
    archive = PROJECT_ROOT / DATA_DIR / ("%s.zip" % WINE_DATASET)
    if not archive.is_file():
        raise Stop("INSTRUMENT_UNREADABLE",
                   "local zip missing: %s/%s.zip" % (DATA_DIR, WINE_DATASET))
    _ctx, helpers = _legacy_helpers()
    train_values, train_labels = helpers["load"](
        np, archive, WINE_DATASET, "TRAIN")
    n_train = int(train_values.shape[0])
    length = int(train_values.shape[1])
    n_classes = int(len({int(label) for label in train_labels}))
    return {
        "selected": WINE_DATASET,
        "rule": (
            "development reuse of local Wine; the unused-Target census "
            "is not applied"),
        "pool_size": 1,
        "eligible": [WINE_DATASET],
        "selection_basis": (
            "task-book CLS-DEV-WINE; archive %s/%s.zip"
            % (DATA_DIR, WINE_DATASET)),
        "archive": "%s/%s.zip" % (DATA_DIR, WINE_DATASET),
        "bytes": int(archive.stat().st_size),
        "official_train_rows": n_train,
        "series_length": length,
        "class_count": n_classes,
        "approx_heldin_points": n_train * length,
        "independent_confirmation": False,
        "honesty_constraint": WINE_HONESTY,
        "prior_use_audit": (
            "artifacts/functional/e2/t6_cls_conf_r3_selection.json"),
        "prior_use_runner": (
            "evaluation/functional/"
            "run_e2_action_credit_candidate_ordering.py"),
    }


def _wine_precheck_markdown(payload: Mapping[str, Any]) -> str:
    census = payload.get("selection") or {}
    invariance = payload.get("v2_invariance_at_150") or {}
    derivations = payload.get("v2_geometry") or {}
    precheck = payload.get("precheck") or {}
    gate = precheck.get("gate") or {}
    ledger = payload.get("ledger") or {}
    lines = [
        "# CLS-DEV-WINE precheck -- v2 injection legality + headroom",
        "",
        "protocol: `%s`  target: **%s**  evidence grade: **%s**"
        % (payload.get("protocol_version"), payload.get("target"),
           payload.get("evidence_grade")),
        "",
        "## Gate",
        "",
        "**%s**" % gate.get("verdict"),
        "",
        str(gate.get("rule") or ""),
        "",
        "- hampel_filter legal: %s" % precheck.get("hampel_legal"),
        "- hampel headroom vs identity: %s (line %s, n_heldin=%s)"
        % (precheck.get("hampel_headroom"), gate.get("material_line"),
           gate.get("n_heldin")),
        "- hampel cohort modified fraction: %s"
        % precheck.get("hampel_modified_fraction"),
        "- hampel rejection: %s"
        % (",".join(precheck.get("hampel_rejection_codes") or []) or "-"),
        "",
        "## Honesty constraint",
        "",
        WINE_HONESTY,
        "",
        "This artifact is **development** evidence.  It is not an "
        "independent confirmation and must not be cited as "
        "CLS_CHAIN_CONFIRMED.",
        "",
        "## v2 scaling",
        "",
        "- formula: `%s`" % (invariance.get("formula") or ""),
        "- v1 source constants: %s"
        % (invariance.get("source_constants") or ""),
        "- invariance at L=150: **%s**  checks=%s"
        % (invariance.get("passed"), invariance.get("checks")),
        "",
    ]
    for key in ("wine_234", "ecg200_96_reference", "gunpoint_150"):
        row = derivations.get(key) or {}
        if not row:
            continue
        lines += [
            "### %s (L=%s)" % (row.get("label"), row.get("series_length")),
            "",
            "- v2 segment length: %s" % row.get("v2_segment_length"),
            "- positions: %s" % row.get("positions"),
            "- injection: %s = %s (below 0.10: %s)"
            % (row.get("injection_fraction_exact"),
               row.get("injection_fraction"),
               row.get("injection_below_cap")),
            "- hampel theoretical (window=%s, halo/spike=%s): %s = %s "
              "(below 0.10: %s)"
            % (row.get("hampel_window"), row.get("hampel_halo_per_spike"),
               row.get("hampel_theoretical_fraction_exact"),
               row.get("hampel_theoretical_modified_fraction"),
               row.get("hampel_theoretical_below_cap")),
            "- segments overlap: %s (min gap %s)"
            % (row.get("segments_overlap"), row.get("min_gap")),
            "",
        ]
    lines += [
        "## Substrate",
        "",
        "- archive: `%s` (%s bytes)"
        % (census.get("archive"), census.get("bytes")),
        "- TRAIN rows × length: %s × %s; classes: %s"
        % (census.get("official_train_rows"),
           census.get("series_length"), census.get("class_count")),
        "- held-in n (full support pool): %s" % precheck.get("n_heldin"),
        "- consumer / metric: %s / %s"
        % (precheck.get("consumer"), precheck.get("metric")),
        "",
        "## Operator table",
        "",
        "| program | legal | modified fraction | rejection | "
        "no-op | headroom vs identity | worst class Δrecall |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in precheck.get("programs") or []:
        headroom = row.get("headroom")
        worst = row.get("worst_class_recall_delta")
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (row.get("program"), row.get("legal"),
               row.get("cohort_modified_fraction"),
               ",".join(row.get("rejection_codes") or []) or "-",
               row.get("numeric_no_op"),
               ("%+.4f" % headroom) if headroom is not None else "n/a",
               ("%+.4f" % worst) if worst is not None else "n/a"))
    lines += [
        "",
        "## Cost",
        "",
        "- LLM: %s / %s" % (ledger.get("llm_calls"), ledger.get("llm_cap")),
        "- Consumer fits: %s / %s"
        % (ledger.get("consumer_fits"), ledger.get("consumer_fit_cap")),
        "- wall clock: %s s" % ledger.get("wall_seconds"),
        "- downloads: 0",
        "",
        "## Obligations",
        "",
    ]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


def _dev_wine_markdown(payload: Mapping[str, Any]) -> str:
    text = _conf_dev_markdown(payload)
    text = text.replace(
        "# CLS-DEV-ECG200 -- development-grade conf lifecycle on local ECG200",
        "# CLS-DEV-WINE -- development-grade conf lifecycle on local Wine "
        "(v2 injection)")
    return text


def _dev_wine_verdict(payload: Mapping[str, Any], *,
                      stopped: str | None) -> dict[str, Any]:
    verdict = _conf_verdict(
        payload, stopped=stopped,
        positive_label=CONF_DEV_POSITIVE,
        negative_label=CONF_DEV_NO_POSITIVE,
        claim_limit=(
            "development only.  " + WINE_HONESTY
            + "  The impulse is a controlled injection; a positive here "
            "is a second development-grade Target-local Skill, not a "
            "fresh confirmation."))
    verdict["forbidden_label"] = "CLS_CHAIN_CONFIRMED"
    verdict["independent_confirmation"] = False
    verdict["evidence_grade"] = WINE_EVIDENCE_GRADE
    return verdict


def dev_wine_precheck() -> int:
    """Part B: 0-LLM legality + headroom gate on Wine with v2 injection."""
    started = time.time()
    print("CLS-DEV-WINE precheck  protocol=%s  dataset=%s  template=v2"
          % (WINE_PROTOCOL, WINE_DATASET), flush=True)
    fit_budget = FitBudget(CONF_FIT_CAP)
    invariance = _v2_invariance_at_150()
    if not invariance["passed"]:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "v2 invariance at L=150 failed: %s"
                   % invariance["checks"])
    hampel_params = _contract_params("hampel_filter")
    hampel_window = int(hampel_params.get("window") or 3)
    geometry = {
        "wine_234": _v2_geometry_derivation(
            234, hampel_window=hampel_window, label="Wine"),
        "ecg200_96_reference": _v2_geometry_derivation(
            ECG200_REFERENCE_LENGTH, hampel_window=hampel_window,
            label="ECG200 reference"),
        "gunpoint_150": _v2_geometry_derivation(
            GUNPOINT_POSITIVE_CONTROL_LENGTH, hampel_window=hampel_window,
            label="GunPoint positive-control length"),
        "hampel_contract_params": hampel_params,
        "note": (
            "ECG200 is a constant-derivation reference, not a substrate "
            "retry.  v2 segment at L=96 is still 1, so the 0.125 "
            "theoretical hampel fraction is the same geometry that "
            "failed the 0.10 cap on the ECG200 development run."),
    }
    wine_geom = geometry["wine_234"]
    if wine_geom["segments_overlap"]:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "v2 segments overlap on Wine; formula produced an "
                   "invalid geometry")
    if not (wine_geom["injection_below_cap"]
            and wine_geom["hampel_theoretical_below_cap"]):
        raise Stop("INSTRUMENT_UNREADABLE",
                   "Wine v2 derivation does not sit below the 0.10 cap")
    census = _dev_wine_census()
    cell = _build_cell(WINE_DATASET, CONF_CONDITION, data_dir=DATA_DIR,
                       injection_template=INJECTION_TEMPLATE_V2)
    precheck = _wine_precheck_pass(cell, fit_budget=fit_budget)
    payload: dict[str, Any] = {
        "protocol_version": WINE_PROTOCOL,
        "run_id": "t6_cls_dev_wine_precheck1",
        "entry": "--dev-wine-precheck",
        "evidence_grade": WINE_EVIDENCE_GRADE,
        "git_head": _git("rev-parse", "HEAD"),
        "target": WINE_DATASET,
        "condition": CONF_CONDITION,
        "selection": census,
        "honesty_constraint": WINE_HONESTY,
        "independent_confirmation": False,
        "forbidden_label": "CLS_CHAIN_CONFIRMED",
        "v2_invariance_at_150": invariance,
        "v2_geometry": geometry,
        "cell": {key: value for key, value in cell.items()
                 if key not in ("fit_values", "fit_labels",
                                "surfaces", "observation_block")},
        "precheck": precheck,
        "verdict": {
            "verdict": precheck["gate"]["verdict"],
            "hampel_legal": precheck["hampel_legal"],
            "hampel_headroom": precheck["hampel_headroom"],
            "material_line": precheck["gate"]["material_line"],
            "n_heldin": precheck["n_heldin"],
            "rule": precheck["gate"]["rule"],
            "next": (
                "Part C --dev-wine-run is authorized"
                if precheck["gate"]["passed"] else
                "FAMILY_CLOSURE_RECOMMENDED: impulse×hampel family has "
                "no second geometry-compatible positive field; do not "
                "change the substrate or spend LLM"),
        },
        "ledger": {
            "llm_calls": 0,
            "llm_cap": 0,
            "consumer_fits": fit_budget.used,
            "consumer_fit_cap": CONF_FIT_CAP,
            "wall_seconds": round(time.time() - started, 1),
            "downloads": 0,
        },
        "obligations": {
            "methods_package_unmodified": True,
            "runtime_contracts_operators_unmodified": True,
            "cap_not_raised": True,
            "no_scan_no_tune": True,
            "zero_llm": True,
            "downloads": 0,
            "ucr_conf_downloaded_not_opened": True,
            "other_line_files_untouched": True,
            "not_an_independent_confirmation": True,
            "cls_chain_confirmed_label_not_used": True,
            "existing_entries_unchanged": True,
        },
    }
    WINE_PRECHECK_JSON.parent.mkdir(parents=True, exist_ok=True)
    WINE_PRECHECK_JSON.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    WINE_PRECHECK_MD.write_text(
        _wine_precheck_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"]["verdict"],
        "hampel_legal": precheck["hampel_legal"],
        "hampel_headroom": precheck["hampel_headroom"],
        "n_heldin": precheck["n_heldin"],
        "fits": fit_budget.used,
        "artifact": str(WINE_PRECHECK_JSON),
    }, ensure_ascii=False, indent=1), flush=True)
    return 0 if precheck["gate"]["passed"] else 1


def _dev_wine_write_stall(*, started: float, reason: str) -> None:
    payload = {
        "protocol_version": WINE_PROTOCOL,
        "run_id": WINE_RUN_ID,
        "entry": "--dev-wine-run",
        "evidence_grade": WINE_EVIDENCE_GRADE,
        "target": WINE_DATASET,
        "honesty_constraint": WINE_HONESTY,
        "independent_confirmation": False,
        "verdict": {
            "verdict": "COMPUTE_BUDGET_EXCEEDED",
            "reason": reason,
            "note": (
                "stop-report; 90-minute wall-clock cap; not a scientific "
                "negative"),
            "forbidden_label": "CLS_CHAIN_CONFIRMED",
        },
        "ledger": {
            "llm_calls": None,
            "llm_cap": CONF_LLM_CAP,
            "consumer_fits": None,
            "consumer_fit_cap": CONF_FIT_CAP,
            "wall_seconds": round(time.time() - started, 1),
            "wall_cap_seconds": WINE_WALL_SECONDS,
            "downloads": 0,
        },
        "obligations": {
            "methods_package_unmodified": True,
            "downloads": 0,
            "ucr_conf_downloaded_not_opened": True,
            "not_an_independent_confirmation": True,
        },
    }
    WINE_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    WINE_OUT_JSON.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    WINE_OUT_MD.write_text(_dev_wine_markdown(payload), encoding="utf-8")


def dev_wine_run(*, run_id: str = WINE_RUN_ID) -> int:
    """Part C: same two-arm conf lifecycle on Wine, v2 injection only."""
    import threading

    if not WINE_PRECHECK_JSON.is_file():
        raise Stop("INSTRUMENT_UNREADABLE",
                   "Wine precheck artifact missing; run --dev-wine-precheck "
                   "first: %s" % WINE_PRECHECK_JSON)
    precheck = json.loads(WINE_PRECHECK_JSON.read_text(encoding="utf-8"))
    if (precheck.get("verdict") or {}).get("verdict") != "PASS":
        raise Stop(
            "FAMILY_CLOSURE_RECOMMENDED",
            "precheck did not PASS; Part C is not authorized")
    started = time.time()
    print("CLS-DEV-WINE start  protocol=%s  dataset=%s  wall_cap=%ss  "
          "template=v2" % (WINE_PROTOCOL, WINE_DATASET, WINE_WALL_SECONDS),
          flush=True)
    armed = {"live": True}

    def _timeout() -> None:
        if not armed["live"]:
            return
        _dev_wine_write_stall(
            started=started,
            reason=("hard wall-clock cap of %ss reached; process terminated"
                    % WINE_WALL_SECONDS))
        print(json.dumps({
            "verdict": "COMPUTE_BUDGET_EXCEEDED",
            "target": WINE_DATASET,
            "wall_seconds": round(time.time() - started, 1),
            "artifact": str(WINE_OUT_JSON),
        }, ensure_ascii=False, indent=1), flush=True)
        os._exit(2)

    timer = threading.Timer(WINE_WALL_SECONDS, _timeout)
    timer.daemon = True
    timer.start()
    extra = {
        "honesty_constraint": WINE_HONESTY,
        "independent_confirmation": False,
        "forbidden_label": "CLS_CHAIN_CONFIRMED",
        "wall_cap_seconds": WINE_WALL_SECONDS,
        "ledger_downloads": 0,
        "precheck_gate": (precheck.get("verdict") or {}),
        "obligations": {
            "sealed_d2_d3_untouched": True,
            "ucr_conf_downloaded_not_opened": True,
            "methods_package_unmodified": True,
            "downloads": 0,
            "not_an_independent_confirmation": True,
            "cls_chain_confirmed_label_not_used": True,
            "precheck_passed": True,
            "injection_template": INJECTION_TEMPLATE_V2,
            "target_never_used_before": (
                "FALSE.  Wine is a local already-used substrate "
                "(action_credit line under the same impulse pair); this "
                "is development reuse, not a virgin Target"),
        },
    }
    try:
        return conf_run(
            run_id=run_id, protocol=WINE_PROTOCOL,
            out_json=WINE_OUT_JSON, out_md=WINE_OUT_MD,
            census_fn=_dev_wine_census,
            entry="--dev-wine-run", data_dir=DATA_DIR, extra=extra,
            markdown_fn=_dev_wine_markdown,
            verdict_fn=_dev_wine_verdict,
            evidence_grade=WINE_EVIDENCE_GRADE,
            accepted_verdicts=(CONF_DEV_POSITIVE, CONF_DEV_NO_POSITIVE),
            injection_template=INJECTION_TEMPLATE_V2)
    finally:
        armed["live"] = False
        timer.cancel()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--r2-prep", dest="r2_prep", action="store_true")
    parser.add_argument("--r2-run", dest="r2_run", action="store_true")
    parser.add_argument("--r2-annotate", dest="r2_annotate",
                        action="store_true")
    parser.add_argument("--r2-replay-a5", dest="r2_replay_a5",
                        action="store_true")
    parser.add_argument("--conf-select", dest="conf_select",
                        action="store_true")
    parser.add_argument("--conf-run", dest="conf_run", action="store_true")
    parser.add_argument("--conf-r2-select", dest="conf_r2_select",
                        action="store_true")
    parser.add_argument("--conf-r2-run", dest="conf_r2_run",
                        action="store_true")
    parser.add_argument("--conf-dl-select", dest="conf_dl_select",
                        action="store_true")
    parser.add_argument("--conf-dl-run", dest="conf_dl_run",
                        action="store_true")
    parser.add_argument("--conf-dev-run", dest="conf_dev_run",
                        action="store_true")
    parser.add_argument("--dev-wine-precheck", dest="dev_wine_precheck",
                        action="store_true")
    parser.add_argument("--dev-wine-run", dest="dev_wine_run",
                        action="store_true")
    parser.add_argument("--dataset", dest="dataset", default=None)
    parser.add_argument("--run-id", dest="run_id", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if sum([args.smoke, args.run, args.micro, args.diagnose,
            args.r2_prep, args.r2_run, args.r2_annotate, args.r2_replay_a5,
            args.conf_select, args.conf_run,
            args.conf_r2_select, args.conf_r2_run,
            args.conf_dl_select, args.conf_dl_run,
            args.conf_dev_run, args.dev_wine_precheck,
            args.dev_wine_run]) != 1:
        parser.error("choose exactly one entry point")
    if args.dev_wine_precheck:
        return dev_wine_precheck()
    if args.dev_wine_run:
        return dev_wine_run(run_id=args.run_id or WINE_RUN_ID)
    if args.conf_dev_run:
        return conf_dev_run(
            dataset=args.dataset or CONF_DEV_DEFAULT_DATASET,
            run_id=args.run_id)
    if args.conf_dl_select:
        return conf_dl_select()
    if args.conf_dl_run:
        return conf_dl_run(run_id=args.run_id or CONF_DL_RUN_ID)
    if args.conf_r2_select:
        return conf_r2_select()
    if args.conf_r2_run:
        return conf_r2_run(run_id=args.run_id or CONF_R2_RUN_ID)
    if args.conf_select:
        return conf_select()
    if args.conf_run:
        return conf_run(run_id=args.run_id or CONF_RUN_ID)
    if args.r2_annotate:
        return r2_annotate()
    if args.r2_replay_a5:
        return r2_replay_a5(run_id=args.run_id or R2_REPLAY_RUN_ID)
    if args.r2_run:
        return run(live=True, out_path=R2_RUN_JSON, md_path=R2_RUN_MD,
                   fraction_scope="cohort",
                   run_id=args.run_id or R2_RUN_ID,
                   llm_total=R2_LLM_CAP,
                   protocol_version=R2_RUN_PROTOCOL,
                   entry_name="--r2-run")
    if args.r2_prep:
        return r2_prep()
    if args.diagnose:
        return diagnose()
    if args.micro:
        return micro()
    if args.smoke:
        return run(live=False, out_path=OUT_SMOKE)
    return run(live=True, out_path=OUT_JSON)


if __name__ == "__main__":
    raise SystemExit(main())
