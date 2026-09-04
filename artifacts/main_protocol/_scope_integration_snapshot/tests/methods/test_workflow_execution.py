import numpy as np
import pytest

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_action_conditioned_valuation_proxy import (
    _group_removal_predictions,
    _ridge_reference_and_removal_predictions,
)
from SelfEvolvingHarnessTS.methods.ttha.workflow_execution import (
    execute_rowblock_support_only,
    execute_whole_group_curation_support_only,
    replay_bound_workflow_prediction,
)


def test_support_only_workflow_executors_match_grouped_ridge_formulas():
    rng = np.random.default_rng(17)
    x_train = rng.normal(size=(6, 3))
    targets = rng.normal(size=(6, 4))
    x_eval = rng.normal(size=(3, 3))
    reference = _ridge_reference_and_removal_predictions(
        np,
        x_train=x_train,
        targets=targets,
        x_eval=x_eval,
        candidate_rows=tuple(range(6)),
        target_block=(0, 4),
    )
    baseline = np.asarray(reference["baseline_prediction"], dtype=np.float64)

    # A deterministic synthetic Support criterion makes every non-baseline
    # perturbation beneficial, exercising execution and the all-rows guard.
    def score_support(prediction):
        values = np.asarray(prediction, dtype=np.float64)
        return -np.mean((values - baseline) ** 2, axis=1)

    calls = []

    def group_predict(rows, target_block, removal_strength):
        calls.append((rows, target_block, removal_strength))
        return _group_removal_predictions(
            np,
            reference=reference,
            selected_local_indices=rows,
            target_block=target_block,
            removal_strength=removal_strength,
        )

    groups = (
        {"group_id": "donor_a", "selected_rows": (0, 2, 4)},
        {"group_id": "donor_b", "selected_rows": (1, 3, 5)},
    )
    doses = (
        {"action_id": "ATTENUATE", "removal_strength": 0.75},
        {"action_id": "EXCLUDE", "removal_strength": 1.0},
    )
    curation = execute_whole_group_curation_support_only(
        reference,
        groups,
        doses,
        target_block=(0, 4),
        score_support=score_support,
        group_predict=group_predict,
    )

    manual_candidates = []
    design = np.asarray(reference["evaluation_design"], dtype=np.float64)
    directions = np.asarray(reference["candidate_directions"], dtype=np.float64)
    residual = np.asarray(reference["candidate_full_residual"], dtype=np.float64)
    baseline_losses = score_support(baseline)
    for group in groups:
        selected = np.asarray(group["selected_rows"], dtype=np.int64)
        for dose in doses:
            strength = float(dose["removal_strength"])
            proxy = baseline - strength * (
                (design @ directions[:, selected]) @ residual[selected]
            )
            manual_candidates.append(
                {
                    "group_id": group["group_id"],
                    "action_id": dose["action_id"],
                    "rows": tuple(group["selected_rows"]),
                    "strength": strength,
                    "gain": float(np.mean(baseline_losses - score_support(proxy))),
                }
            )
    expected = max(
        manual_candidates,
        key=lambda row: row["gain"],
    )
    expected_group = _group_removal_predictions(
        np,
        reference=reference,
        selected_local_indices=expected["rows"],
        target_block=(0, 4),
        removal_strength=expected["strength"],
    )["exact_group_prediction"]
    expected_gain = float(
        np.mean(baseline_losses - score_support(expected_group))
    )
    assert len(calls) == 1
    assert curation["bound_action"]["group_id"] == expected["group_id"]
    assert curation["bound_action"]["action_id"] == expected["action_id"]
    assert curation["support_gain"] == pytest.approx(expected_gain)
    assert curation["decision"] == "EXECUTE"
    assert np.allclose(curation["prediction"], expected_group)

    calls.clear()
    blocks = ((0, 2), (2, 4))
    rowblock = execute_rowblock_support_only(
        reference,
        blocks,
        score_support=score_support,
        group_predict=group_predict,
    )
    manual_composed = baseline.copy()
    singleton = np.asarray(
        reference["first_order_proxy_predictions"], dtype=np.float64
    )
    for binding, block in zip(rowblock["bound_groups"], blocks):
        gains = []
        for row_index in range(6):
            proxy = baseline.copy()
            proxy[:, block[0] : block[1]] = singleton[
                row_index, :, block[0] : block[1]
            ]
            gains.append(float(np.mean(baseline_losses - score_support(proxy))))
        retained = min(range(6), key=lambda index: (gains[index], index))
        selected = tuple(index for index in range(6) if index != retained)
        expected_block = _group_removal_predictions(
            np,
            reference=reference,
            selected_local_indices=selected,
            target_block=block,
            removal_strength=1.0,
        )["exact_group_prediction"]
        manual_composed[:, block[0] : block[1]] = expected_block[
            :, block[0] : block[1]
        ]
        assert binding["all_rows_guard_retained_index"] == retained
        assert binding["selected_rows"] == list(selected)
    assert len(calls) == len(blocks)
    assert rowblock["decision"] == "EXECUTE"
    assert rowblock["support_gain"] == pytest.approx(
        float(np.mean(baseline_losses - score_support(manual_composed)))
    )
    assert np.allclose(rowblock["prediction"], manual_composed)

    def zero_score(prediction):
        return np.zeros(np.asarray(prediction).shape[0], dtype=np.float64)

    tied_default = execute_whole_group_curation_support_only(
        reference,
        groups,
        doses,
        target_block=(0, 4),
        score_support=zero_score,
        group_predict=group_predict,
    )
    tied_custom = execute_whole_group_curation_support_only(
        reference,
        groups,
        doses,
        target_block=(0, 4),
        score_support=zero_score,
        group_predict=group_predict,
        candidate_tiebreak=lambda row: (
            str(row["group_id"]),
            str(row["action_id"]),
        ),
    )
    assert tied_default["bound_action"]["group_id"] == "donor_a"
    assert tied_default["bound_action"]["action_id"] == "ATTENUATE"
    assert tied_custom["bound_action"]["group_id"] == "donor_b"
    assert tied_custom["bound_action"]["action_id"] == "EXCLUDE"
    assert tied_custom["decision"] == "ABSTAIN"
    assert tied_custom["support_gain"] == 0.0
    assert np.array_equal(tied_custom["prediction"], baseline)

    forbidden = ("query", "future", "outcome", "metric", "memory")

    def assert_clean_keys(value):
        if isinstance(value, dict):
            assert all(
                not any(token in str(key).lower() for token in forbidden)
                for key in value
            )
            for child in value.values():
                assert_clean_keys(child)
        elif isinstance(value, list):
            for child in value:
                assert_clean_keys(child)

    assert_clean_keys(curation)
    assert_clean_keys(rowblock)
    assert_clean_keys(tied_custom)


def test_bound_workflow_replay_covers_whole_group_rowblock_and_abstain():
    baseline = np.arange(16, dtype=np.float64).reshape(2, 8)
    reference = {
        "baseline_prediction": baseline,
        "candidate_directions": np.eye(4),
        "candidate_full_residual": np.ones((4, 8)),
        "evaluation_design": np.ones((2, 4)),
        "first_order_proxy_predictions": np.repeat(baseline[None, :, :], 4, axis=0),
    }
    calls = []

    def group_predict(rows, target_block, removal_strength):
        calls.append((rows, target_block, removal_strength))
        prediction = baseline.copy()
        start, stop = target_block
        prediction[:, start:stop] += removal_strength * (sum(rows) + 1)
        return {"exact_group_prediction": prediction}

    whole = replay_bound_workflow_prediction(
        reference,
        {
            "decision": "EXECUTE",
            "bound_action": {"selected_rows": [1, 3], "removal_strength": 0.75},
        },
        group_predict=group_predict,
    )
    assert calls == [((1, 3), (0, 8), 0.75)]
    assert np.array_equal(whole, baseline + 3.75)

    calls.clear()
    rowblock = replay_bound_workflow_prediction(
        reference,
        {
            "decision": "EXECUTE",
            "bound_groups": [
                {"block_half_open": [0, 4], "selected_rows": [0, 2]},
                {"block_half_open": [4, 8], "selected_rows": [1]},
            ],
        },
        group_predict=group_predict,
    )
    expected = baseline.copy()
    expected[:, :4] += 3.0
    expected[:, 4:] += 2.0
    assert calls == [((0, 2), (0, 4), 1.0), ((1,), (4, 8), 1.0)]
    assert np.array_equal(rowblock, expected)

    calls.clear()
    abstained = replay_bound_workflow_prediction(
        reference, {"decision": "ABSTAIN"}, group_predict=group_predict
    )
    assert calls == []
    assert np.array_equal(abstained, baseline)
    assert abstained is not baseline
