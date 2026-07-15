from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.policy.task_spec import forecast_task_spec_v1
from SelfEvolvingHarnessTS.vnext.grammar import ActionEligibilityManifestV1, CandidateGrammarV1
from SelfEvolvingHarnessTS.vnext.pattern import build_pattern_card
from SelfEvolvingHarnessTS.vnext.suppliers import (
    deterministic_b3, llm_plan_compiler_b3, random_valid_b3, semantic_supplier_seed,
)


def _inputs():
    card = build_pattern_card(np.sin(np.arange(96) / 5.0))
    task = forecast_task_spec_v1(horizon=48)
    grammar = CandidateGrammarV1(ActionEligibilityManifestV1.conservative())
    return card, task, grammar


def test_all_supplier_arms_spend_three_slots_and_random_replays():
    card, task, grammar = _inputs()
    deterministic = deterministic_b3(card, task, grammar)
    first = random_valid_b3(card, task, grammar, seed=7)
    second = random_valid_b3(card, task, grammar, seed=7)
    assert len(deterministic.programs) == 3
    assert first == second


def test_llm_malformed_and_duplicate_effects_are_itt_noop_without_replacement():
    card, task, grammar = _inputs()
    batch = llm_plan_compiler_b3(
        card, task, grammar, plan=lambda _summary, _budget: ["v_none", "v_none", "bad"],
    )
    assert batch.request_count == 1
    assert batch.statuses[0] == "valid"
    assert "duplicate_effect_itt_noop" in batch.statuses
    assert "malformed_itt_noop" in batch.statuses


def test_random_seed_is_bound_to_visible_pattern_not_uid():
    card, _, _ = _inputs()
    first = semantic_supplier_seed(
        global_seed=20260713, fold_id="fold-0", slot_namespace="random-b3", pattern=card,
    )
    second = semantic_supplier_seed(
        global_seed=20260713, fold_id="fold-0", slot_namespace="random-b3", pattern=card,
    )
    assert first == second
