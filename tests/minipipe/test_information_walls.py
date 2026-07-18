import json
from dataclasses import replace

import pytest

from SelfEvolvingHarnessTS.evaluation.minipipe.contracts import ArtifactRoots
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.first_fault import CaseFacts, assess_case
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.patterns import mine_failure_patterns
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.sanitize import (
    PublicArtifactReader,
    sanitize_case_feedback,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import PublicAgentInput


FORBIDDEN_PUBLIC_KEYS = {
    "private_family",
    "private_severity",
    "oracle_affected_indices",
    "clean_context",
    "clean_future",
    "candidate_utilities",
    "loss_j",
    "utility_u",
    "r_private",
    "injection_type",
    "confirmed_surface",
}


def _private_feedback(case_id="m0-0001"):
    facts = replace(
        CaseFacts.passing(case_id=case_id),
        private_family="impulsive_outlier",
        oracle_affected_indices=(111, 149),
        candidate_utilities={"identity": -0.4},
        effect_distinct_candidate_ids=(),
        chosen_candidate_id="identity",
        capability_skill_exists=False,
        expressibility_status="PROVEN_EXPRESSIBLE",
        public_probe_gains={"clipping": 0.12, "denoising": 0.01},
        private_probe_gains={"clipping": 0.20},
        behavior_signature={
            "tool_names": ["summarize_observables"],
            "retrieved_skill_ids": [],
            "chosen_candidate_id": "identity",
        },
    )
    return assess_case(facts).feedback


def test_sanitizer_removes_oracle_and_judge_fields():
    public = sanitize_case_feedback(_private_feedback())
    encoded = json.dumps(public.to_json(), sort_keys=True).lower()
    assert not any(key in encoded for key in FORBIDDEN_PUBLIC_KEYS)
    assert public.confirmed_surface is None


def test_applicability_in_public_card_uses_closed_vocabulary():
    card = sanitize_case_feedback(_private_feedback())
    assert card.observable_signature
    with pytest.raises(ValueError, match="unknown observable feature"):
        card.with_applicability(
            {"all": [{"feature": "injection_type", "op": "==", "value": "x"}]}
        )


def test_public_reader_rejects_private_or_parent_paths(tmp_path):
    roots = ArtifactRoots.create(tmp_path)
    reader = PublicArtifactReader(roots.public)
    with pytest.raises(PermissionError):
        reader.read_json(roots.private / "case_feedback.jsonl")
    with pytest.raises(PermissionError):
        reader.read_json("../private/case_feedback.jsonl")


def test_recurring_patterns_are_deterministic_and_have_no_confirmed_surface():
    feedback = (_private_feedback("m0-0001"), _private_feedback("m0-0002"))
    first = mine_failure_patterns(feedback)
    second = mine_failure_patterns(tuple(reversed(feedback)))
    assert len(first) == 1
    assert first[0].pattern_id == second[0].pattern_id
    assert first[0].support_count == 2
    assert first[0].confirmed_surface is None


def test_backend_input_rejects_private_candidate_values_mechanically():
    with pytest.raises(ValueError, match="forbidden private field"):
        PublicAgentInput.create(
            "m0-0001",
            {"safe": {"candidate_utilities": {"identity": -0.4}}},
        )
