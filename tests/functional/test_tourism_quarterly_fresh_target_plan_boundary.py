from pathlib import Path

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_tourism_quarterly_fresh_target import (
    CURRENT_CUTOFF,
    _panel,
)


def test_tourism_quarterly_frozen_roster_does_not_materialize_query_future():
    root = Path(__file__).resolve().parents[2]
    train, support, query, arrays = _panel(root, query_end=CURRENT_CUTOFF)

    assert [row.entity_id for row in [*train, *support, *query]] == [
        f"T{index}" for index in range(3, 21)
    ]
    assert [len(arrays[row.series_uid]) for row in train] == [64] * 12
    assert [len(arrays[row.series_uid]) for row in support] == [72] * 3
    assert [len(arrays[row.series_uid]) for row in query] == [64] * 3
