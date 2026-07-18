from pathlib import Path

import numpy as np
import pytest

from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import build_core_corpus
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import FrozenChronosValuator
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.outcomes import evaluate_outcome
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.rolling_observed import (
    RollingObservedValuator,
)


class FakeChronos:
    def predict_quantiles(self, contexts, *, prediction_length, quantile_levels):
        import torch

        mean = torch.zeros((len(contexts), prediction_length), dtype=torch.float32)
        quantiles = mean[:, :, None]
        return quantiles, mean


def test_utility_is_negative_clean_scaled_nrmse_and_fill_is_recorded():
    context = np.arange(192, dtype=float)
    context[10:12] = np.nan
    clean_context = np.arange(192, dtype=float)
    future = np.ones(48, dtype=float)
    receipt = FrozenChronosValuator(pipeline=FakeChronos()).evaluate(
        context, future, scale_context=clean_context
    )
    expected_j = 1.0 / np.std(np.arange(192, dtype=float))
    assert receipt.loss_j == pytest.approx(expected_j)
    assert receipt.utility_u == pytest.approx(-expected_j)
    assert receipt.fill_fraction == pytest.approx(2 / 192)


def test_outcome_definitions_use_higher_is_better_utility():
    outcome = evaluate_outcome(
        clean_u=-0.10,
        corrupt_u=-0.40,
        prepared_u=-0.20,
        identity_u=-0.40,
        candidate_utilities={"identity": -0.40, "agent-0": -0.20},
        chosen_candidate_id="agent-0",
        damage_noise_floor=0.01,
    )
    assert outcome.damage_d == pytest.approx(0.30)
    assert outcome.repair_gain_g == pytest.approx(0.20)
    assert outcome.nrr == pytest.approx(2 / 3)
    assert outcome.selection_regret == 0.0


def test_rolling_transform_cannot_change_held_out_target_identity():
    series = np.sin(np.arange(192, dtype=float) / 8.0)
    valuator = RollingObservedValuator(pipeline=FakeChronos())
    baseline = valuator.evaluate(series)
    transformed = valuator.evaluate(
        series,
        prefix_transform=lambda prefix, _origin: np.zeros_like(prefix),
    )
    assert baseline.target_shas == transformed.target_shas
    assert baseline.input_shas != transformed.input_shas


def test_rolling_invalid_origins_are_excluded_and_all_invalid_is_unknown():
    series = np.arange(192, dtype=float)
    series[96:120] = np.nan
    receipt = RollingObservedValuator(pipeline=FakeChronos()).evaluate(series)
    assert 96 not in receipt.origins
    assert 96 in receipt.excluded_origins

    all_invalid = np.full(192, np.nan)
    unknown = RollingObservedValuator(pipeline=FakeChronos()).evaluate(all_invalid)
    assert unknown.status == "UNKNOWN"
    assert unknown.mean_public_utility is None


def test_rolling_public_serializer_has_no_private_fields():
    receipt = RollingObservedValuator(pipeline=FakeChronos()).evaluate(
        np.arange(192, dtype=float)
    )
    text = str(receipt.to_public_dict()).lower()
    assert "clean_future" not in text
    assert "injection" not in text
    assert "private" not in text


@pytest.mark.frozen_model
def test_pinned_local_model_is_deterministic_without_network():
    rules_path = (
        Path(__file__).resolve().parents[2]
        / "evaluation"
        / "minipipe"
        / "config"
        / "m0_rules.json"
    )
    case = build_core_corpus(load_m0_rules(rules_path)).targets[0]
    private = FrozenChronosValuator()
    first = private.evaluate(
        case.corrupt_context,
        case.clean_future,
        scale_context=case.clean_context,
    )
    second = private.evaluate(
        case.corrupt_context,
        case.clean_future,
        scale_context=case.clean_context,
    )
    assert first.forecast_sha == second.forecast_sha
    assert first.utility_u == pytest.approx(second.utility_u, abs=1e-12)

    public = RollingObservedValuator(pipeline=private.pipeline)
    rolling_first = public.evaluate(case.corrupt_context)
    rolling_second = public.evaluate(case.corrupt_context)
    assert rolling_first.forecast_shas == rolling_second.forecast_shas
    assert rolling_first.mean_public_utility == pytest.approx(
        rolling_second.mean_public_utility, abs=1e-12
    )
