from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from evaluation.main_protocol_p1 import classification_component as p1


EXPECTED = {
    ("Epilepsy2", "support_a"): {
        "fit_n": 40,
        "eval_n": 20,
        "label_names": ("0", "1"),
        "truth_counts": (10, 10),
        "identity_confusion": ((7, 3), (1, 9)),
        "hampel_confusion": ((8, 2), (1, 9)),
        "prediction_flips": 1,
        "wrong_to_right": 1,
        "right_to_wrong": 0,
    },
    ("Epilepsy2", "support_b"): {
        "fit_n": 40,
        "eval_n": 20,
        "label_names": ("0", "1"),
        "truth_counts": (10, 10),
        "identity_confusion": ((6, 4), (0, 10)),
        "hampel_confusion": ((8, 2), (0, 10)),
        "prediction_flips": 2,
        "wrong_to_right": 2,
        "right_to_wrong": 0,
    },
    ("PowerCons", "support_a"): {
        "fit_n": 90,
        "eval_n": 44,
        "label_names": ("1", "2"),
        "truth_counts": (22, 22),
        "identity_confusion": ((17, 5), (1, 21)),
        "hampel_confusion": ((17, 5), (2, 20)),
        "prediction_flips": 7,
        "wrong_to_right": 3,
        "right_to_wrong": 4,
    },
}


def _confusion(truth: np.ndarray, predicted: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Rows are true encoded labels; columns are predicted encoded labels."""
    return tuple(
        tuple(
            int(np.count_nonzero((truth == true_label) & (predicted == predicted_label)))
            for predicted_label in (0, 1)
        )
        for true_label in (0, 1)
    )


def _predict(
    cell: p1.ClassificationCell,
    face: str,
    steps: tuple[tuple[str, Mapping[str, object]], ...],
) -> tuple[np.ndarray, np.ndarray]:
    # This is the current P1 fit-policy exactly: the Workflow transforms only
    # the fit cohort; both models score the same unmodified Support rows.
    prepared_fit, _behavior_points = p1.MacroF1ConsumerAdapter._prepared_fit(
        cell.fit_values, steps
    )
    model = p1.RidgeClassifier(alpha=1.0)
    model.fit(p1.raw_plus_difference(prepared_fit), cell.fit_labels)
    values, truth = cell.surface(face)
    predicted = model.predict(p1.raw_plus_difference(values))
    return np.asarray(truth), np.asarray(predicted)


@pytest.fixture(scope="module")
def diagnostics():
    target, selection, boundary = p1._load_exposed_cells()
    cells = {cell.fixture_id: cell for cell in (target, *selection)}
    rows = {}
    for fixture_id, face in EXPECTED:
        cell = cells[fixture_id]
        truth, identity = _predict(cell, face, p1._steps("identity"))
        hampel_truth, hampel = _predict(cell, face, p1._steps("hampel_filter"))
        np.testing.assert_array_equal(hampel_truth, truth)
        flipped = identity != hampel
        rows[(fixture_id, face)] = {
            "fit_n": int(cell.fit_labels.size),
            "eval_n": int(truth.size),
            "label_names": tuple(cell.label_names),
            "truth_counts": tuple(int(np.count_nonzero(truth == label)) for label in (0, 1)),
            "identity_confusion": _confusion(truth, identity),
            "hampel_confusion": _confusion(truth, hampel),
            # Direction is identity -> Hampel.  The PowerCons K0 -> A5
            # reencounter narrative reverses the two correctness counts but
            # has the same seven prediction flips.
            "prediction_flips": int(np.count_nonzero(flipped)),
            "wrong_to_right": int(
                np.count_nonzero((identity != truth) & (hampel == truth))
            ),
            "right_to_wrong": int(
                np.count_nonzero((identity == truth) & (hampel != truth))
            ),
        }
    return cells, boundary, rows


@pytest.mark.parametrize("key", tuple(EXPECTED))
def test_train_only_confusion_and_prediction_flips(diagnostics, key):
    _cells, boundary, rows = diagnostics
    assert boundary["test_member_bytes_read"] == 0
    assert boundary["held_out_requests"] == 0
    assert boundary["development_query_evaluations"] == 0
    assert boundary["natural_final_outcome_reads"] == 0

    assert rows[key] == EXPECTED[key]
    assert rows[key]["prediction_flips"] == (
        rows[key]["wrong_to_right"] + rows[key]["right_to_wrong"]
    )


def test_three_reported_surfaces_have_84_distinct_evaluation_rows(diagnostics):
    cells, _boundary, rows = diagnostics
    row_keys = {
        (fixture_id, int(index))
        for fixture_id, face in EXPECTED
        for index in (
            cells[fixture_id].support_a_indices
            if face == "support_a"
            else cells[fixture_id].support_b_indices
        )
    }
    assert sum(row["eval_n"] for row in rows.values()) == 84
    assert len(row_keys) == 84
    assert set(cells["Epilepsy2"].support_a_indices).isdisjoint(
        cells["Epilepsy2"].support_b_indices
    )


def test_powercons_reencounter_reverses_the_correctness_direction(diagnostics):
    _cells, _boundary, rows = diagnostics
    identity_to_hampel = rows[("PowerCons", "support_a")]
    # The conflict is reported identity -> Hampel.  The later controlled
    # K0-to-A5 comparison uses the same seven rows in the reverse direction:
    # K0 Hampel -> A5 identity.
    hampel_to_identity_wrong_to_right = identity_to_hampel["right_to_wrong"]
    hampel_to_identity_right_to_wrong = identity_to_hampel["wrong_to_right"]
    assert identity_to_hampel["prediction_flips"] == 7
    assert hampel_to_identity_wrong_to_right == 4
    assert hampel_to_identity_right_to_wrong == 3
