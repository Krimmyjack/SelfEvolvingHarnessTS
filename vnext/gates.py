"""Pure, preregisterable decision functions for Task G, M3, and SA-V."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ._canonical import require_sha, sha256


@dataclass(frozen=True)
class JudgeLadderPreregV1:
    retrained_judges: tuple[str, ...] = (
        "closed_form_dlinear", "adam_dlinear", "lstm_scratch",
    )
    zero_shot_diagnostics: tuple[str, ...] = ("chronos_bolt",)
    estimand: str = "published_joint_v02"
    epsilon: float = 0.02
    harm_delta: float = 0.05
    ci_level: float = 0.90
    bootstrap_replicates: int = 2000
    cluster_unit: str = "overlap_group"

    def __post_init__(self) -> None:
        if self.retrained_judges != ("closed_form_dlinear", "adam_dlinear", "lstm_scratch"):
            raise ValueError("Task G requires the frozen three-judge retrained roster")
        if self.zero_shot_diagnostics != ("chronos_bolt",):
            raise ValueError("Chronos-Bolt is a separate zero-shot diagnostic")
        if self.estimand != "published_joint_v02" or self.cluster_unit != "overlap_group":
            raise ValueError("Task G prereg drifted from benchmark-v0.2")

    @property
    def sha256(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class JudgeGain:
    judge_id: str
    ex_covid_gain: float
    ci90_low: float
    ci90_high: float
    readable_non_covid_datasets: tuple[str, ...]
    natural_denominator_robust: bool = True


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    code: str
    details: tuple[str, ...]

    @property
    def sha256(self) -> str:
        return sha256(self)


def task_g_verdict(rows: Sequence[JudgeGain]) -> GateVerdict:
    if len(rows) != 3 or len({row.judge_id for row in rows}) != 3:
        return GateVerdict(False, "invalid_judge_roster", ())
    positive = [
        row for row in rows
        if row.ex_covid_gain > 0.02 and row.ci90_low > 0
        and len(set(row.readable_non_covid_datasets)) >= 2
        and row.natural_denominator_robust
    ]
    reversal = [row.judge_id for row in rows if row.ex_covid_gain < -0.02 and row.ci90_high < 0]
    passed = len(positive) >= 2 and not reversal
    return GateVerdict(
        passed, "unlock_ttha0" if passed else "capability_track",
        tuple([f"qualifying={','.join(sorted(row.judge_id for row in positive))}"]
              + ([f"reversal={','.join(sorted(reversal))}"] if reversal else [])),
    )


@dataclass(frozen=True)
class ArmComparison:
    arm_id: str
    delta_vs_deterministic: float
    ci90_low: float
    supply_ceiling_delta: float
    worst_readable_harm: float
    cost_gate_passed: bool
    replay_gate_passed: bool


def runtime_supplier_verdict(row: ArmComparison, *, delta: float = 0.05) -> GateVerdict:
    passed = (
        row.delta_vs_deterministic >= 0.02 and row.ci90_low > 0
        and row.supply_ceiling_delta >= 0
        and row.worst_readable_harm <= delta
        and row.cost_gate_passed and row.replay_gate_passed
    )
    return GateVerdict(passed, "runtime_eligible" if passed else "offline_only", (row.arm_id,))


@dataclass(frozen=True)
class PromotionSummary:
    ex_covid_gain: float
    ci90_low: float
    natural_regression: float
    controlled_regression: float
    worst_readable_harm: float
    prepared_valid_fraction: float
    unrecorded_fallbacks: int
    dependency_masquerades: int
    budget_exceeded: bool


def support_a_validation_verdict(row: PromotionSummary, *, delta: float = 0.05) -> GateVerdict:
    passed = (
        row.ex_covid_gain > 0.02 and row.ci90_low > 0
        and row.natural_regression <= delta and row.controlled_regression <= delta
        and row.worst_readable_harm <= delta
        and row.prepared_valid_fraction == 1.0
        and row.unrecorded_fallbacks == 0 and row.dependency_masquerades == 0
        and not row.budget_exceeded
    )
    return GateVerdict(passed, "promote" if passed else "retain_h0", ())


SIX_ARM_ROSTER: tuple[str, ...] = (
    "frozen_hbase", "deterministic_b3", "random_valid_b3", "llm_direct_b3",
    "llm_plan_compiler_b3", "hybrid_escalation_b3",
)


@dataclass(frozen=True)
class M3IdentityGatePreregV2:
    roster: tuple[str, ...] = SIX_ARM_ROSTER
    primary_llm_arm: str = "llm_plan_compiler_b3"
    secondary_llm_arms: tuple[str, ...] = ("llm_direct_b3", "hybrid_escalation_b3")
    comparator: str = "deterministic_b3"
    candidate_slots: int = 3
    random_distribution: str = "uniform_without_replacement_over_active_effect_classes"
    random_seed_rule: str = "sha256(global_seed,fold,slot,pattern_semantic_sha)"
    hybrid_slot_rule: str = "deterministic_top1_plus_two_plan_slots_on_frozen_trigger"
    epsilon: float = 0.02
    harm_delta: float = 0.05
    ci_level: float = 0.90
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 20260713
    schema_version: str = "vnext-m3a-prereg/2"

    def __post_init__(self) -> None:
        if self.roster != SIX_ARM_ROSTER:
            raise ValueError("M3a roster or order drifted")
        if self.primary_llm_arm != "llm_plan_compiler_b3":
            raise ValueError("only plan-compiler is the primary LLM test")
        if self.candidate_slots != 3:
            raise ValueError("M3a arms must spend exactly three slots")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class SupplierArmAggregateV2:
    arm_id: str
    delta_vs_deterministic: float
    ci90_low_vs_deterministic: float
    delta_vs_frozen_hbase: float
    ci90_low_vs_frozen_hbase: float
    supply_ceiling_delta_vs_deterministic: float
    worst_readable_loss_regression: float
    worst_readable_regression_ci90_low: float
    prepared_valid_fraction: float
    cost_gate_passed: bool
    replay_gate_passed: bool

    @property
    def material_harm(self) -> bool:
        return (
            self.worst_readable_loss_regression > 0.05
            and self.worst_readable_regression_ci90_low > 0
        )


@dataclass(frozen=True)
class H0SupplierSelectionV1:
    supplier_policy_id: str
    reason: str
    initial_runtime_efficacy: bool

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


def select_h0_supplier(
    rows: Sequence[SupplierArmAggregateV2],
    prereg: M3IdentityGatePreregV2 | None = None,
) -> H0SupplierSelectionV1:
    """Apply the frozen LLM→random→deterministic→incumbent hierarchy."""
    prereg = prereg or M3IdentityGatePreregV2()
    by_id = {row.arm_id: row for row in rows}
    if set(by_id) != set(prereg.roster):
        raise ValueError("M3a aggregate does not contain the frozen six-arm roster")

    def safe(row: SupplierArmAggregateV2) -> bool:
        return (
            not row.material_harm and row.prepared_valid_fraction == 1.0
            and row.cost_gate_passed and row.replay_gate_passed
        )

    primary = by_id[prereg.primary_llm_arm]
    if (
        safe(primary)
        and primary.delta_vs_deterministic >= prereg.epsilon
        and primary.ci90_low_vs_deterministic > 0
        and primary.supply_ceiling_delta_vs_deterministic >= 0
    ):
        return H0SupplierSelectionV1(
            primary.arm_id, "primary_llm_runtime_gate_passed", True,
        )
    random = by_id["random_valid_b3"]
    if (
        safe(random)
        and random.delta_vs_deterministic >= prereg.epsilon
        and random.ci90_low_vs_deterministic > 0
        and random.supply_ceiling_delta_vs_deterministic >= 0
    ):
        return H0SupplierSelectionV1(
            random.arm_id, "random_increment_over_deterministic", False,
        )
    deterministic = by_id["deterministic_b3"]
    if (
        safe(deterministic)
        and deterministic.ci90_low_vs_frozen_hbase >= -prereg.epsilon
    ):
        return H0SupplierSelectionV1(
            deterministic.arm_id, "deterministic_safe_noninferior", False,
        )
    return H0SupplierSelectionV1(
        "frozen_hbase", "all_generated_suppliers_failed_frozen_gate", False,
    )


# v3 semantics: H0 is already frozen from Init Corpus.  M3 chooses only a runtime
# supplier control and cannot participate in H0 construction.  V2 remains replayable.
SIX_ARM_RUNTIME_ROSTER: tuple[str, ...] = (
    "frozen_h0", "deterministic_b3", "random_valid_b3", "llm_direct_b3",
    "llm_plan_compiler_b3", "hybrid_escalation_b3",
)


@dataclass(frozen=True)
class M3RuntimeSupplierPreregV3:
    roster: tuple[str, ...] = SIX_ARM_RUNTIME_ROSTER
    primary_llm_arm: str = "llm_plan_compiler_b3"
    comparator: str = "deterministic_b3"
    incumbent: str = "frozen_h0"
    changes_h0: bool = False
    candidate_slots: int = 3
    epsilon: float = 0.02
    harm_delta: float = 0.05
    ci_level: float = 0.90
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 20260713
    schema_version: str = "vnext-m3-runtime-supplier-prereg/3"

    def __post_init__(self) -> None:
        if self.roster != SIX_ARM_RUNTIME_ROSTER or self.incumbent != "frozen_h0":
            raise ValueError("M3 runtime supplier roster drifted from formal H0")
        if self.changes_h0 or self.candidate_slots != 3:
            raise ValueError("M3 cannot redefine H0 or change the three-slot budget")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class RuntimeSupplierArmAggregateV1:
    arm_id: str
    delta_vs_deterministic: float
    ci90_low_vs_deterministic: float
    delta_vs_frozen_h0: float
    ci90_low_vs_frozen_h0: float
    supply_ceiling_delta_vs_deterministic: float
    worst_readable_loss_regression: float
    worst_readable_regression_ci90_low: float
    prepared_valid_fraction: float
    cost_gate_passed: bool
    replay_gate_passed: bool

    @property
    def material_harm(self) -> bool:
        return (
            self.worst_readable_loss_regression > 0.05
            and self.worst_readable_regression_ci90_low > 0
        )


@dataclass(frozen=True)
class RuntimeSupplierSelectionV1:
    supplier_policy_id: str
    reason: str
    initial_runtime_efficacy: bool
    changes_h0: bool = False

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


def select_runtime_supplier(
    rows: Sequence[RuntimeSupplierArmAggregateV1],
    prereg: M3RuntimeSupplierPreregV3 | None = None,
) -> RuntimeSupplierSelectionV1:
    prereg = prereg or M3RuntimeSupplierPreregV3()
    by_id = {row.arm_id: row for row in rows}
    if set(by_id) != set(prereg.roster):
        raise ValueError("M3 runtime aggregate does not contain the frozen six-arm roster")

    def safe(row: RuntimeSupplierArmAggregateV1) -> bool:
        return (
            not row.material_harm and row.prepared_valid_fraction == 1.0
            and row.cost_gate_passed and row.replay_gate_passed
        )

    primary = by_id[prereg.primary_llm_arm]
    if (
        safe(primary) and primary.delta_vs_deterministic >= prereg.epsilon
        and primary.ci90_low_vs_deterministic > 0
        and primary.supply_ceiling_delta_vs_deterministic >= 0
    ):
        return RuntimeSupplierSelectionV1(
            primary.arm_id, "primary_llm_runtime_gate_passed", True,
        )
    random = by_id["random_valid_b3"]
    if (
        safe(random) and random.delta_vs_deterministic >= prereg.epsilon
        and random.ci90_low_vs_deterministic > 0
        and random.supply_ceiling_delta_vs_deterministic >= 0
    ):
        return RuntimeSupplierSelectionV1(
            random.arm_id, "random_increment_over_deterministic", False,
        )
    deterministic = by_id["deterministic_b3"]
    if safe(deterministic) and deterministic.ci90_low_vs_frozen_h0 >= -prereg.epsilon:
        return RuntimeSupplierSelectionV1(
            deterministic.arm_id, "deterministic_safe_noninferior", False,
        )
    return RuntimeSupplierSelectionV1(
        "frozen_h0", "all_generated_suppliers_failed_frozen_h0_gate", False,
    )


@dataclass(frozen=True)
class HarnessEvolutionPreregV2:
    tracks: tuple[str, ...] = ("E_det", "E_rand", "E_llm", "E_hybrid")
    max_cycles: int = 3
    slots_per_cycle: int = 3
    mandatory_slots: tuple[str, ...] = ("no_edit", "deterministic_common", "track_specific")
    f0_required: bool = True
    f1_required: bool = True
    f2_max_per_track_cycle: int = 1
    f2_min_groups: int = 8
    f2_epsilon: float = 0.02
    llm_input_token_cap: int = 8192
    llm_output_token_cap: int = 1024
    llm_retries: int = 0
    timeout_rule: str = "ceil_2xp95_init_pilot"
    schema_version: str = "vnext-harness-evolution-prereg/2"

    def __post_init__(self) -> None:
        if self.max_cycles != 3 or self.slots_per_cycle != 3:
            raise ValueError("MVP evolution budget is three cycles by three slots")
        if self.llm_retries != 0:
            raise ValueError("malformed/error LLM slots are ITT no-op without retry")

    @property
    def max_logical_candidates(self) -> int:
        return len(self.tracks) * self.max_cycles * self.slots_per_cycle

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class LLMFactorialSummaryV1:
    evolution_increment: float
    evolution_ci90_low: float
    evolution_worst_harm: float
    mature_runtime_increment: float
    mature_runtime_ci90_low: float
    mature_runtime_worst_harm: float
    complementarity_increment: float
    complementarity_ci90_low: float
    cost_gate_passed: bool
    replay_gate_passed: bool


@dataclass(frozen=True)
class LLMQualificationDecisionV1:
    evolution_llm_qualified: bool
    mature_runtime_llm_qualified: bool
    complementarity_qualified: bool


def llm_factorial_verdict(
    row: LLMFactorialSummaryV1,
    *,
    epsilon: float = 0.02,
    delta: float = 0.05,
) -> LLMQualificationDecisionV1:
    common = row.cost_gate_passed and row.replay_gate_passed
    evolution = (
        common and row.evolution_increment >= epsilon and row.evolution_ci90_low > 0
        and row.evolution_worst_harm <= delta
    )
    runtime = (
        common and row.mature_runtime_increment >= epsilon and row.mature_runtime_ci90_low > 0
        and row.mature_runtime_worst_harm <= delta
    )
    complementarity = (
        common and row.complementarity_increment >= epsilon
        and row.complementarity_ci90_low > 0
    )
    return LLMQualificationDecisionV1(evolution, runtime, complementarity)


@dataclass(frozen=True)
class SAVPromotionInputV2:
    candidate_sha: str
    h0_sha: str
    ex_covid_gain_vs_h0: float
    ex_covid_ci90_low: float
    natural_regression_vs_h0: float
    natural_regression_ci90_low: float
    controlled_regression_vs_h0: float
    controlled_regression_ci90_low: float
    worst_readable_loss_regression: float
    worst_readable_regression_ci90_low: float
    prepared_valid_fraction: float
    unrecorded_fallback_count: int
    dependency_masquerade_count: int
    budget_violation_count: int

    def __post_init__(self) -> None:
        require_sha(self.candidate_sha, "candidate_sha")
        require_sha(self.h0_sha, "h0_sha")
        if self.candidate_sha == self.h0_sha:
            raise ValueError("SA-V must abstain instead of querying H0 against itself")


def support_a_validation_verdict_v2(
    row: SAVPromotionInputV2,
    *,
    epsilon: float = 0.02,
    delta: float = 0.05,
) -> GateVerdict:
    harms = (
        (row.natural_regression_vs_h0, row.natural_regression_ci90_low),
        (row.controlled_regression_vs_h0, row.controlled_regression_ci90_low),
        (row.worst_readable_loss_regression, row.worst_readable_regression_ci90_low),
    )
    material_harm = any(point > delta and ci_low > 0 for point, ci_low in harms)
    passed = (
        row.ex_covid_gain_vs_h0 > epsilon and row.ex_covid_ci90_low > 0
        and not material_harm and row.prepared_valid_fraction == 1.0
        and row.unrecorded_fallback_count == 0
        and row.dependency_masquerade_count == 0
        and row.budget_violation_count == 0
    )
    return GateVerdict(
        passed, "promote" if passed else "retain_h0",
        (f"comparator={row.h0_sha}", f"candidate={row.candidate_sha}"),
    )
