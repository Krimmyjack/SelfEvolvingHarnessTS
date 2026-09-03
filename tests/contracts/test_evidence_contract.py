from __future__ import annotations

from SelfEvolvingHarnessTS.contracts.evidence import (
    CapabilityTemplate,
    EvidenceDisposition,
    LocalBehaviorEvidence,
    ObservationScope,
    PatternNode,
    PolicyInterventionEvidence,
    ScopedObservationReceipt,
    ScopedProgramBinding,
    SignedEvidenceLedger,
)
from SelfEvolvingHarnessTS.contracts.program import Program


def test_scoped_observation_to_policy_evidence_vertical_slice() -> None:
    scope = ObservationScope("a" * 64, "series-1", "load", 10, 20, "support")
    receipt = ScopedObservationReceipt.create(
        tool_name="series_overview",
        arguments={},
        public_result={"level_excursion_score": "medium"},
        context_sha="b" * 64,
        scope=scope,
        coverage={"requested_points": 10, "observed_points": 10},
        reliability={"status": "OK"},
        tool_version="1",
    )
    assert receipt.scope == scope
    assert receipt.reliability["status"] == "OK"

    pattern = PatternNode(
        "localized-level-excursion",
        scope,
        {"level_excursion_score": "medium"},
        (receipt.receipt_sha,),
    )
    program = Program.from_steps(
        [("repair_level_shift", {"offset": 0.5})], source="e0-test"
    )
    binding = ScopedProgramBinding(
        "level-repair-binding",
        program,
        scope,
        {"region": [scope.start, scope.end]},
    )
    capability = CapabilityTemplate(
        "level-shift-repair",
        {"level_excursion_score": "medium"},
        {"allowed_operators": ["repair_level_shift"]},
        {"scope": "observation_interval"},
        {"preserve_outside_scope": True},
        {"task": "forecast"},
        {"model": "fixed:m0"},
        {"registry": "m0"},
    )
    local = LocalBehaviorEvidence(
        "behavior-1",
        capability.capability_id,
        pattern.pattern_id,
        binding.binding_id,
        {"changed_inside_scope": True, "changed_outside_scope": False},
        (receipt.receipt_sha,),
    )
    policy = PolicyInterventionEvidence(
        "policy-1",
        capability.capability_id,
        "identity",
        "level-shift-repair",
        {"cohort": "support-heldout"},
        {"median_target_improvement": 0.2},
        {"regressions": 0},
        "support_cohort_paired_replay",
        ("runs/e0/policy-1.json",),
        (receipt.receipt_sha,),
    )

    ledger = SignedEvidenceLedger("e0-ledger")
    ledger.append(local, EvidenceDisposition.SUPPORTED, recorded_by="runtime")
    assert ledger.compile_verdict(capability.capability_id) is EvidenceDisposition.UNRESOLVED
    ledger.append(policy, EvidenceDisposition.SUPPORTED, recorded_by="judge")
    assert ledger.compile_verdict(capability.capability_id) is EvidenceDisposition.SUPPORTED
    assert ledger.summarize(capability.capability_id)["evidence_count"] == 2
    restored = SignedEvidenceLedger.from_rows("e0-ledger-restored", ledger.to_rows())
    assert restored.to_rows() == ledger.to_rows()
    assert restored.compile_verdict(capability.capability_id) is EvidenceDisposition.SUPPORTED

