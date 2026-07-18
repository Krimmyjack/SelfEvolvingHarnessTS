import json
from pathlib import Path

import numpy as np

from SelfEvolvingHarnessTS.contracts.observables import OBSERVABLE_FEATURES
from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import build_core_corpus
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.expressibility import (
    ExpressibilityEvaluator,
    ExpressibilityStatus,
    evaluate_expressibility,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.features import extract_public_features
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.panel import M0_PROBE_SPECS, ProbePanel
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import FrozenChronosValuator
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.rolling_observed import (
    RollingObservedValuator,
)
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    extract_public_features as extract_agent_public_features,
)


class FakeChronos:
    def predict_quantiles(self, contexts, *, prediction_length, quantile_levels):
        import torch

        rows = []
        for context in contexts:
            values = np.asarray(context, dtype=np.float64).reshape(-1)
            rows.append(np.full(prediction_length, values[-1], dtype=np.float32))
        mean = torch.as_tensor(np.stack(rows), dtype=torch.float32)
        return mean[:, :, None], mean


def _rules_and_corpus():
    root = Path(__file__).resolve().parents[2]
    rules = load_m0_rules(root / "evaluation/minipipe/config/m0_rules.json")
    return rules, build_core_corpus(rules)


def test_every_repair_probe_has_three_monotonic_strengths():
    for name in ("imputation", "clipping", "denoising", "level_correction"):
        spec = M0_PROBE_SPECS[name]
        assert spec.betas == (0.25, 0.50, 0.75)
        assert spec.aggressiveness == tuple(sorted(spec.aggressiveness))
        assert len(set(spec.aggressiveness)) == 3


def test_public_feature_extractor_is_closed_and_deterministic():
    _, corpus = _rules_and_corpus()
    case = corpus.targets[0]
    first = extract_public_features(case.corrupt_context)
    second = extract_public_features(case.corrupt_context)
    assert set(first.mapping) <= set(OBSERVABLE_FEATURES)
    assert first.feature_context_sha == second.feature_context_sha


def test_period_is_diagnostic_only_and_declares_repair_unavailable():
    rules, corpus = _rules_and_corpus()
    case = next(case for case in corpus.targets if case.private_family == "period_change")
    rolling = RollingObservedValuator(pipeline=FakeChronos())
    panel = ProbePanel(rolling_valuator=rolling, rules=rules)
    result = panel.run_public(case.to_public_view())
    assert result.period_diagnostic.repair_available is False
    assert result.period_diagnostic.evidence_status == "OK"
    assert "period" not in result.response_curves
    assert sum(len(points) for points in result.response_curves.values()) == 12


def test_missing_contamination_makes_period_evidence_unknown_in_both_views():
    values = np.sin(2.0 * np.pi * np.arange(192, dtype=float) / 12.0)
    values[20:52] = np.nan
    grader = extract_public_features(values)
    agent = extract_agent_public_features(values, task_kind="forecast")
    assert grader.period_evidence_status == "UNKNOWN"
    assert grader.mapping["period_change_score"] == 0.0
    assert agent["period_evidence_status"] == "UNKNOWN"
    assert agent["period_change_score"] == 0.0


def test_public_panel_serializer_has_no_absolute_or_private_values():
    rules, corpus = _rules_and_corpus()
    panel = ProbePanel(
        rolling_valuator=RollingObservedValuator(pipeline=FakeChronos()),
        rules=rules,
    )
    receipt = panel.run_public(corpus.targets[0].to_public_view())
    serialized = json.dumps(receipt.to_public_dict(), sort_keys=True).lower()
    for forbidden in (
        "private",
        "clean",
        "candidate",
        "injection",
        '"utility_u"',
        '"loss_j"',
    ):
        assert forbidden not in serialized


def test_private_probe_receipt_has_no_public_serializer():
    rules, corpus = _rules_and_corpus()
    private = FrozenChronosValuator(pipeline=FakeChronos())
    panel = ProbePanel(
        rolling_valuator=RollingObservedValuator(pipeline=private.pipeline),
        private_valuator=private,
        rules=rules,
    )
    receipt = panel.run_private(corpus.targets[0])
    assert not hasattr(receipt, "to_public_dict")
    assert sum(len(points) for points in receipt.response_curves.values()) == 12


def test_oracle_yes_existing_feature_not_derived_is_procedure_gap():
    result = evaluate_expressibility(
        oracle_succeeds=True,
        observable_succeeds=False,
        required_feature_is_in_closed_vocabulary=True,
    )
    assert result.oracle_witness.succeeded is True
    assert result.status is ExpressibilityStatus.EXPRESSIBILITY_UNKNOWN
    assert result.cause_code == "OBSERVABLE_DERIVATION_PROCEDURE_GAP"


def test_missing_required_public_feature_is_noneditable_schema_gap():
    result = evaluate_expressibility(
        oracle_succeeds=True,
        observable_succeeds=False,
        required_feature_is_in_closed_vocabulary=False,
    )
    assert result.status is ExpressibilityStatus.EXPRESSIBILITY_UNKNOWN
    assert result.cause_code == "OBSERVABLE_FEATURE_SCHEMA_GAP"


def test_absent_complete_period_class_proves_unavailable():
    _, corpus = _rules_and_corpus()
    case = next(case for case in corpus.targets if case.private_family == "period_change")
    result = ExpressibilityEvaluator().evaluate(case)
    assert result.status is ExpressibilityStatus.PROVEN_UNAVAILABLE
    assert result.missing_transformation_class == "period_correction"


def test_transformation_class_declaration_is_complete_for_canonical_registry():
    root = Path(__file__).resolve().parents[2]
    declaration = json.loads(
        (root / "evaluation/minipipe/probes/transformation_classes.json").read_text()
    )
    actual = {OPERATOR_METADATA[name]["category"] for name in OPERATOR_NAMES}
    assert set(declaration["complete_operator_categories"]) == actual
    assert "period_correction" not in actual
