"""P4b statistics: origin-level aggregation and the frozen verdict ladder."""
from __future__ import annotations

import pytest

from evaluation.main_protocol_p4 import p4b_stats as stats


def _rows(arm: str, values: dict[int, list[float]]) -> list[dict[str, object]]:
    return [
        {"arm": arm, "origin": origin, "delta_utility_vs_identity": value}
        for origin, series in values.items()
        for value in series
    ]


def test_replicas_are_averaged_inside_an_origin_not_treated_as_samples():
    # Three replicas walk the same origins and differ only in LLM sampling, so
    # they are one sample, not three.
    rows = _rows("A5-bounded", {1176: [0.1, 0.2, 0.3], 1416: [0.0, 0.0, 0.6]})
    means = stats.by_origin(rows, arm="A5-bounded", field="delta_utility_vs_identity")
    assert means == {1176: pytest.approx(0.2), 1416: pytest.approx(0.2)}


def test_a_contrast_pairs_only_origins_present_on_both_sides():
    rows = _rows("A5-bounded", {1176: [0.5], 1416: [0.5]})
    rows += _rows("A5-strict", {1176: [0.1]})
    result = stats.contrast(
        rows, left="A5-bounded", right="A5-strict",
        field="delta_utility_vs_identity",
    )
    assert result["origins"] == [1176]
    assert result["paired_differences"] == [pytest.approx(0.4)]


def test_the_exact_test_is_enumerated_not_approximated():
    result = stats.wilcoxon_exact([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    # All eight differences positive: the only sign assignment at least as
    # extreme is the all-negative mirror, so p = 2/2^8.
    assert result["p_two_sided"] == pytest.approx(2 / 256)
    assert result["n_nonzero"] == 8


def test_power_note_states_what_n_eight_cannot_detect():
    note = stats.power_note(8)
    assert note["smallest_attainable_two_sided_p"] == pytest.approx(2 / 256)
    assert note["reaches_alpha_05"] is True
    assert "not" in note["reading"]


def _verdict(**overrides):
    base = dict(
        utility={"origins": [1176, 1416], "mean_difference": 0.0,
                 "wilcoxon": {"p_two_sided": 0.5}},
        held_out_worst_single_series_harm=0.1,
        held_out_harm_rate=0.0,
        max_single_series_harm=0.30,
        max_harmed_fraction=0.20,
        any_admission_held_in=True,
        active_skills_formed=1,
        causal_reuse_observed=False,
        writeback_gated=True,
        parallel_selection_face="held_in",
    )
    base.update(overrides)
    return stats.primary_verdict(**base)


def test_an_empty_contrast_is_not_a_neutral_result():
    # The rehearsal has no endpoint phase at all.  Falling through to NEUTRAL
    # would report "no detectable difference" about a comparison never made.
    verdict = _verdict(
        utility={"origins": [], "mean_difference": None,
                 "wilcoxon": {"p_two_sided": None}}
    )
    assert verdict["verdict"] == "NO_ENDPOINT_DATA"
    assert verdict["blocking"] is True


def test_ungated_writeback_voids_the_run_before_anything_is_interpreted():
    assert _verdict(writeback_gated=False)["verdict"] == "LEAKAGE_SUSPECTED"


def test_selecting_on_the_endpoint_face_voids_the_run():
    assert _verdict(
        parallel_selection_face="held_out"
    )["verdict"] == "HELDOUT_CONTAMINATED"


def test_no_admission_at_support_a_closes_the_experiment():
    verdict = _verdict(any_admission_held_in=False)
    assert verdict["verdict"] == "BOUNDED_GATE_STILL_BLOCKING"
    assert verdict["blocking_face"] == "SUPPORT_A"


def test_support_a_admission_without_an_approved_skill_is_blocking_not_neutral():
    # Support-A admission is provisional; full Target-local admission needs both
    # faces.  Reporting this as NEUTRAL would claim a comparison that the run
    # never reached a deployable state to make.
    verdict = _verdict(any_admission_held_in=True, active_skills_formed=0)
    assert verdict["verdict"] == "BOUNDED_GATE_STILL_BLOCKING"
    assert verdict["blocking_face"] == "SUPPORT_B"
    assert verdict["blocking"] is True


def test_neutral_requires_a_formed_skill_and_a_measured_endpoint():
    neutral = _verdict()
    assert neutral["verdict"] == "BOUNDED_GATE_NEUTRAL"
    # Remove either precondition and it is no longer neutral.
    assert _verdict(active_skills_formed=0)["verdict"] != "BOUNDED_GATE_NEUTRAL"
    assert _verdict(
        utility={"origins": [], "mean_difference": None,
                 "wilcoxon": {"p_two_sided": None}}
    )["verdict"] != "BOUNDED_GATE_NEUTRAL"


@pytest.mark.parametrize(
    ("field", "value"),
    [("held_out_worst_single_series_harm", 0.31), ("held_out_harm_rate", 0.21)],
    ids=["single_series", "harm_rate"],
)
def test_a_risk_breach_closes_the_experiment(field, value):
    assert _verdict(**{field: value})["verdict"] == "RISK_BUDGET_BREACHED"


@pytest.mark.parametrize(
    ("mean", "expected"),
    [(0.4, "BOUNDED_GATE_POSITIVE"), (-0.4, "BOUNDED_GATE_NEGATIVE")],
    ids=["positive", "negative"],
)
def test_a_significant_contrast_is_read_by_its_sign(mean, expected):
    verdict = _verdict(
        utility={"origins": [1176, 1416], "mean_difference": mean,
                 "wilcoxon": {"p_two_sided": 0.01}}
    )
    assert verdict["verdict"] == expected
    assert verdict["blocking"] is False


def test_blocking_conditions_outrank_a_significant_result():
    # A leak is checked before the p-value: an invalid run has no result to read.
    verdict = _verdict(
        writeback_gated=False,
        utility={"origins": [1176], "mean_difference": 0.9,
                 "wilcoxon": {"p_two_sided": 0.001}},
    )
    assert verdict["verdict"] == "LEAKAGE_SUSPECTED"
