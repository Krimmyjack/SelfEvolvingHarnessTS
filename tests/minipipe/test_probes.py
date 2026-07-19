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
    program_requires_external_localization,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.features import extract_public_features
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.panel import (
    M0_PROBE_SPECS,
    PROBE_INSTRUMENT_EPOCH,
    ProbePanel,
    _apply_probe,
    materialize_probe_program,
    public_probe_contracts,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import FrozenChronosValuator
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.rolling_observed import (
    RollingObservedValuator,
)
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    extract_public_features as extract_agent_public_features,
)
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline


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
        assert spec.instrument_epoch == PROBE_INSTRUMENT_EPOCH


def test_imputation_doses_all_change_model_input_and_max_matches_canonical_repair():
    values = np.sin(np.arange(192, dtype=float) / 8.0)
    values[140:152] = np.nan
    canonical = run_pipeline(
        [("impute_linear", {})], values, source="probe-calibration"
    ).artifact
    assert canonical is not None
    for beta in (0.25, 0.50, 0.75):
        transformed, modified_fraction = _apply_probe(values, "imputation", beta)
        assert np.isfinite(transformed).all()
        assert modified_fraction > 0.0
    maximum, _ = _apply_probe(values, "imputation", 0.75)
    np.testing.assert_array_equal(maximum, canonical)


def test_level_max_dose_matches_canonical_public_parameterized_repair():
    _, corpus = _rules_and_corpus()
    case = next(
        case for case in corpus.targets if case.private_family == "level_shift"
    )
    features = extract_public_features(case.corrupt_context)
    canonical = run_pipeline(
        [
            (
                "repair_level_shift",
                {
                    "region_start_fraction": features.mapping[
                        "estimated_region_start_fraction"
                    ],
                    "region_end_fraction": features.mapping[
                        "estimated_region_end_fraction"
                    ],
                    "estimated_offset": features.estimated_excursion_offset,
                },
            )
        ],
        case.corrupt_context,
        source="probe-calibration",
    ).artifact
    assert canonical is not None
    maximum, modified_fraction = _apply_probe(
        case.corrupt_context, "level_correction", 0.75
    )
    assert modified_fraction > 0.0
    np.testing.assert_array_equal(maximum, canonical)


def test_every_fixed_probe_arm_is_exactly_one_materialized_canonical_program():
    _, corpus = _rules_and_corpus()
    cases = {
        family: next(case for case in corpus.targets if case.private_family == family)
        for family in ("missing", "impulsive_outlier", "level_shift")
    }
    inputs = {
        "imputation": cases["missing"].corrupt_context,
        "clipping": cases["impulsive_outlier"].corrupt_context,
        "denoising": cases["impulsive_outlier"].corrupt_context,
        "level_correction": cases["level_shift"].corrupt_context,
    }
    for probe_name, values in inputs.items():
        features = extract_public_features(values)
        for beta in M0_PROBE_SPECS[probe_name].betas:
            steps = materialize_probe_program(probe_name, beta, features.mapping)
            execution = run_pipeline(steps, values, source="probe-equivalence-test")
            assert execution.ok and execution.artifact is not None
            actual, _modified_fraction = _apply_probe(values, probe_name, beta)
            np.testing.assert_array_equal(actual, execution.artifact)


def test_public_probe_contract_discloses_templates_and_current_programs():
    _, corpus = _rules_and_corpus()
    case = next(
        case for case in corpus.targets if case.private_family == "impulsive_outlier"
    )
    features = extract_public_features(case.corrupt_context)
    contracts = public_probe_contracts(features.mapping)
    assert contracts["instrument_epoch"] == PROBE_INSTRUMENT_EPOCH
    clipping = contracts["probes"]["clipping"]
    assert len(clipping["arms"]) == 3
    mild = clipping["arms"][0]["current_context_program_steps"][0]
    assert mild == {
        "op": "hampel_filter",
        "params": {"window": 7, "n_sigmas": 8.0, "global_z_min": 4.0},
    }


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
    assert result.external_localization_required is False


def test_witness_program_contract_distinguishes_internal_and_external_localization():
    assert not program_requires_external_localization(
        [["hampel_filter", {"window": 7, "n_sigmas": 3.0}]]
    )
    assert not program_requires_external_localization([["impute_linear", {}]])
    assert program_requires_external_localization(
        [
            [
                "repair_level_shift",
                {
                    "region_start_fraction_from": "estimated_region_start_fraction",
                    "region_end_fraction_from": "estimated_region_end_fraction",
                    "estimated_offset_from": "estimated_level_offset",
                },
            ]
        ]
    )


def test_transformation_class_declaration_is_complete_for_canonical_registry():
    root = Path(__file__).resolve().parents[2]
    declaration = json.loads(
        (root / "evaluation/minipipe/probes/transformation_classes.json").read_text()
    )
    actual = {OPERATOR_METADATA[name]["category"] for name in OPERATOR_NAMES}
    assert set(declaration["complete_operator_categories"]) == actual
    assert "period_correction" not in actual
