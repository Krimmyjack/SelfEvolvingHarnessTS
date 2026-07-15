from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.policy.task_spec import forecast_task_spec_v1
from SelfEvolvingHarnessTS.vnext.grammar import ActionEligibilityManifestV1, CandidateGrammarV1
from SelfEvolvingHarnessTS.vnext.pattern import build_pattern_card


def test_pattern_card_is_deterministic_and_contains_no_uid():
    values = np.sin(np.arange(96) * 2 * np.pi / 12)
    first = build_pattern_card(values)
    second = build_pattern_card(values.copy())
    assert first == second
    assert first.sha256 == second.sha256
    assert "uid" not in first.summary.lower()
    assert len(first.values) == 16


def test_conservative_eligibility_disables_ssm_and_gates_levelshift():
    manifest = ActionEligibilityManifestV1.conservative()
    assert manifest.state("v_ssm") == "disabled"
    assert manifest.state("v_levelshift") == "experimental"
    assert manifest.state("v_ar") == "experimental"
    assert manifest.state("v_hampel") == "experimental"
    grammar = CandidateGrammarV1(manifest)
    task = forecast_task_spec_v1(horizon=48)
    with pytest.raises(ValueError, match="not active"):
        grammar.from_action("v_ssm", task)
    with pytest.raises(ValueError, match="not active"):
        grammar.from_action("v_levelshift", task)
    qualified = ActionEligibilityManifestV1.conservative(ar_reverse_test_passed=True)
    assert CandidateGrammarV1(qualified).from_action("v_ar", task).provenance["menu_action_id"] == "v_ar"


def test_grammar_only_accepts_exact_menu_presets():
    manifest = ActionEligibilityManifestV1.conservative(hampel_reverse_test_passed=True)
    grammar = CandidateGrammarV1(manifest)
    task = forecast_task_spec_v1(horizon=48)
    spec = grammar.from_action("v_hampel", task)
    grammar.validate(spec, task)
    changed = type(spec)(
        **{**spec.__dict__, "steps": (("impute_linear", ()), ("hampel_filter", (("window", 99),)))},
    )
    with pytest.raises(ValueError, match="frozen preset"):
        grammar.validate(changed, task)
