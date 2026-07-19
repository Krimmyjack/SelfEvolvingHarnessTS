from types import SimpleNamespace

from SelfEvolvingHarnessTS.evaluation.minipipe.cycle import M0CycleRunner


def _feedback(*, damage: float, regret: float = 0.0):
    return SimpleNamespace(
        outcome=SimpleNamespace(damage_d=damage, selection_regret=regret),
        assessments=(SimpleNamespace(status=SimpleNamespace(value="PASS")),),
    )


def test_replay_priority_prefers_recoverable_opportunity_over_support_count():
    feedback = {
        "high-1": _feedback(damage=0.28),
        "high-2": _feedback(damage=0.25),
        "high-3": _feedback(damage=0.36),
        **{f"low-{index}": _feedback(damage=0.05) for index in range(6)},
    }
    high_value = SimpleNamespace(
        case_ids=("high-1", "high-2", "high-3"),
        support_count=3,
        pattern_id="pattern-high",
    )
    high_support = SimpleNamespace(
        case_ids=tuple(f"low-{index}" for index in range(6)),
        support_count=6,
        pattern_id="pattern-low",
    )

    assert M0CycleRunner._priority(high_value, feedback) < M0CycleRunner._priority(
        high_support, feedback
    )
