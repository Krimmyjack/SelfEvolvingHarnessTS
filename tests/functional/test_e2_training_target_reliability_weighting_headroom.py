import numpy as np

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    _apply_stuck_value_censoring,
    _agree_raw_phase_observations,
    _dataset_classification,
    _interval_iou,
    _observe_interval,
    _observe_flatline_interval,
    _oracle_row_weights,
    _oracle_target_cell_weights,
    _phase_residual_observation,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_actionability_credit import (
    _alternating_folds,
    _positive_credit_rows,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_actionability_context_gate import (
    _balanced_accuracy,
    _memory_entries,
    _retrieve_topk_mean,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_query_context_cohort_reweighting import (
    _context_fingerprint,
    _rank_weights,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_action_conditioned_valuation_proxy import (
    ACTION_VALUE_FEEDBACK_BUDGET_SUPPORT_SPLITS,
    _group_removal_predictions,
    _ridge_reference_and_removal_predictions,
    _summarize_action_value_guard_budget,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_temporal_offset_rebind_headroom import (
    _temporal_offset_and_rebind,
)


def test_oracle_weights_and_headroom_decision_are_frozen() -> None:
    weights = _oracle_row_weights(np, 8, {1, 6})
    assert weights.tolist() == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    cells = _oracle_target_cell_weights(np, 8, 6, {1, 6}, (2, 4))
    assert cells.shape == (8, 6)
    assert int(np.count_nonzero(cells == 0.0)) == 4
    assert np.all(cells[[1, 6], 2:4] == 0.0)
    assert np.all(cells[[1, 6], :2] == 1.0)
    assert np.all(cells[[1, 6], 4:] == 1.0)
    assert np.all(cells[[0, 2, 3, 4, 5, 7], :] == 1.0)

    clean_targets = np.arange(24, dtype=np.float64).reshape(4, 6)
    contexts = np.full((4, 5), -10.0, dtype=np.float64)
    censored, restored, changed = _apply_stuck_value_censoring(
        np, clean_targets, contexts, {1, 3}, (2, 5)
    )
    assert changed == 6
    assert np.array_equal(restored, clean_targets)
    assert np.array_equal(censored[[0, 2]], clean_targets[[0, 2]])
    assert np.all(censored[1, 2:5] == contexts[1, -1])
    assert np.all(censored[3, 2:5] == contexts[3, -1])

    nonflat = np.arange(48, dtype=np.float64)
    assert _observe_flatline_interval(np, nonflat)["status"] == "ABSTAIN"
    flatline = nonflat.copy()
    flatline[18:30] = 7.0
    flatline_observation = _observe_flatline_interval(np, flatline)
    assert flatline_observation["status"] == "ACTIVATE"
    assert flatline_observation["predicted_interval"] == [18, 30]

    synthetic = np.zeros(48, dtype=np.float64)
    synthetic[18:30] += 2.0
    observed = _observe_interval(np, synthetic)
    assert observed["status"] == "ACTIVATE"
    assert observed["predicted_interval"] == [16, 28]
    assert _interval_iou(observed["predicted_interval"], (18, 30)) == 10 / 14
    assert _interval_iou(observed["predicted_interval"], (18, 30)) >= 0.5
    assert observed["predicted_interval"] != [18, 30]

    abstained = _observe_interval(np, np.zeros(48, dtype=np.float64))
    assert abstained["status"] == "ABSTAIN"
    assert abstained["predicted_interval"] is None

    period = 12
    one_cycle = 0.1 * np.sin(2 * np.pi * np.arange(period) / period)
    periodic_context = np.tile(one_cycle, 16)
    periodic_target = np.tile(one_cycle, 4)
    clean_phase = _phase_residual_observation(
        np, periodic_context, periodic_target, period=period
    )
    assert clean_phase["status"] == "ABSTAIN"
    corrupt_periodic_target = periodic_target.copy()
    corrupt_periodic_target[18:30] += 2.0
    corrupt_phase = _phase_residual_observation(
        np, periodic_context, corrupt_periodic_target, period=period
    )
    assert corrupt_phase["status"] == "ACTIVATE"
    assert _interval_iou(corrupt_phase["predicted_interval"], (18, 30)) >= 0.5
    corrupt_raw = _observe_interval(np, corrupt_periodic_target)
    agreement = _agree_raw_phase_observations(corrupt_raw, corrupt_phase)
    assert agreement["status"] == "ACTIVATE"
    assert _interval_iou(agreement["predicted_interval"], (18, 30)) >= 0.5
    one_sided = _agree_raw_phase_observations(
        {"status": "ACTIVATE", "predicted_interval": [18, 30], "score": 2.0},
        {"status": "ABSTAIN", "predicted_interval": None, "score": 0.0},
    )
    assert one_sided["status"] == "ABSTAIN"
    assert one_sided["predicted_interval"] is None

    passed, gates = _dataset_classification(
        corruption_mean=0.4,
        corruption_positive_uids=7,
        recovery_mean=0.25,
        recovery_positive_uids=6,
        recovery_fraction=0.625,
    )
    assert passed == "HEADROOM_PASS"
    assert all(gates.values())

    failed, _ = _dataset_classification(
        corruption_mean=0.4,
        corruption_positive_uids=7,
        recovery_mean=0.1,
        recovery_positive_uids=6,
        recovery_fraction=0.25,
    )
    unavailable, _ = _dataset_classification(
        corruption_mean=-0.1,
        corruption_positive_uids=3,
        recovery_mean=0.2,
        recovery_positive_uids=8,
        recovery_fraction=None,
    )
    assert failed == "READABLE_BUT_NO_ORACLE_RECOVERY"
    assert unavailable == "CORRUPTION_UNREADABLE"


def test_actionability_credit_compiles_mask_or_abstain_without_dataset_identity() -> None:
    assert _alternating_folds(8) == {
        "fold_a": (0, 2, 4, 6),
        "fold_b": (1, 3, 5, 7),
    }
    units = [
        {"row_index": 7, "support_credit": -0.2},
        {"row_index": 2, "support_credit": 0.1},
        {"row_index": 4, "support_credit": 0.0},
        {"row_index": 1, "support_credit": 0.3},
    ]
    assert _positive_credit_rows(units, "support_credit") == (1, 2)
    assert ACTION_VALUE_FEEDBACK_BUDGET_SUPPORT_SPLITS == {
        1: tuple((index,) for index in range(8)),
        2: ((0, 1), (2, 3), (4, 5), (6, 7)),
        4: ((0, 2, 4, 6), (1, 3, 5, 7)),
    }
    episode = {
        str(budget): {
            "split_evidence": [
                {
                    "h0_query_gain": 1.0,
                    "h1_guarded_query_gain": 0.0 if budget == 0 else 0.5,
                    "h1_guard_decision": (
                        "ABSTAIN_NO_FEEDBACK" if budget == 0 else "EXECUTE"
                    ),
                    **({} if budget == 0 else {"h1_proposed_fraction": 0.5}),
                }
            ]
        }
        for budget in (0, 1, 2, 4)
    }
    summary = _summarize_action_value_guard_budget([episode])
    assert summary["h0_adapt_auc_budget_grid_mean"] == 1.0
    assert summary["h1_adapt_auc_budget_grid_mean"] == 0.375
    assert summary["budgets"]["1"]["beneficial_gain_retention_fraction"] == 0.5

    raw_values = {
        "series-a": np.arange(100, dtype=np.float64),
        "series-b": np.arange(100, dtype=np.float64) + 100.0,
    }
    provenance = [
        {"series_uid": "series-a", "anchor": 10, "center": 2.0, "scale": 2.0},
        {"series_uid": "series-b", "anchor": 20, "center": 3.0, "scale": 4.0},
    ]
    clean_targets = np.asarray(
        [
            (raw_values["series-a"][10:58] - 2.0) / 2.0,
            (raw_values["series-b"][20:68] - 3.0) / 4.0,
        ]
    )
    offset_targets, rebound_targets, intervals = _temporal_offset_and_rebind(
        np,
        clean_targets=clean_targets,
        row_provenance=provenance,
        raw_values=raw_values,
        selected_rows={0},
        offset=3,
        train_stop=80,
    )
    assert intervals == [[13, 61]]
    assert np.array_equal(
        offset_targets[0], (raw_values["series-a"][13:61] - 2.0) / 2.0
    )
    assert np.array_equal(offset_targets[1], clean_targets[1])
    assert np.array_equal(rebound_targets, clean_targets)


def test_context_gate_marginalizes_repeated_local_evidence_before_retrieval() -> None:
    episodes = [
        {
            "episode_id": "source-a|seed=0|row=1",
            "dataset_id": "source-a",
            "seed": 0,
            "row_key": ("series-1", 240, 48),
            "credit": 0.2,
            "features": {"local_expected": {"local__value": 0.0}},
        },
        {
            "episode_id": "source-a|seed=1|row=1",
            "dataset_id": "source-a",
            "seed": 1,
            "row_key": ("series-1", 240, 48),
            "credit": 0.4,
            "features": {"local_expected": {"local__value": 0.0}},
        },
        {
            "episode_id": "source-b|seed=0|row=2",
            "dataset_id": "source-b",
            "seed": 0,
            "row_key": ("series-2", 300, 48),
            "credit": -0.5,
            "features": {"local_expected": {"local__value": 10.0}},
        },
    ]
    entries = _memory_entries(episodes, "local_expected")
    assert len(entries) == 2
    assert sorted(round(float(row["credit"]), 6) for row in entries) == [-0.5, 0.3]
    prediction, neighbors = _retrieve_topk_mean(
        entries, {"local__value": 0.1}, k=1
    )
    assert prediction == 0.30000000000000004
    assert len(neighbors) == 1
    assert _balanced_accuracy([True, False], [True, False]) == 1.0


def test_query_context_fingerprint_and_rank_program_are_frozen() -> None:
    context = np.sin(2 * np.pi * np.arange(192, dtype=np.float64) / 24)
    fingerprint = _context_fingerprint(np, context, period=24)
    assert fingerprint.shape == (6,)
    assert np.isfinite(fingerprint).all()
    assert fingerprint[4] > 0.99
    assert abs(fingerprint[5]) < 1e-12

    train = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    weights, distances, ranks = _rank_weights(np, train, np.asarray([0.0, 0.0]))
    assert distances.tolist() == [0.0, 1.0, 2.0]
    assert ranks.tolist() == [0, 1, 2]
    assert weights.tolist() == [1.5, 1.0, 0.5]
    assert np.all(weights > 0.0)
    assert np.mean(weights) == 1.0


def test_action_conditioned_ridge_removal_matches_direct_singleton_solve() -> None:
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(9, 4))
    targets = rng.normal(size=(9, 5))
    x_eval = rng.normal(size=(3, 4))
    candidates = (1, 6)
    target_block = (1, 4)
    bundle = _ridge_reference_and_removal_predictions(
        np,
        x_train=x_train,
        targets=targets,
        x_eval=x_eval,
        candidate_rows=candidates,
        target_block=target_block,
    )

    z_train = np.column_stack((x_train, np.ones(len(x_train))))
    z_eval = np.column_stack((x_eval, np.ones(len(x_eval))))
    penalty = np.diag([1.0] * x_train.shape[1] + [0.0])
    for local_index, candidate in enumerate(candidates):
        keep = np.arange(len(x_train)) != candidate
        direct_beta = np.linalg.solve(
            z_train[keep].T @ z_train[keep] + penalty,
            z_train[keep].T @ targets[keep, target_block[0] : target_block[1]],
        )
        direct_block = z_eval @ direct_beta
        exact_prediction = bundle["exact_removal_predictions"][local_index]
        assert np.allclose(
            exact_prediction[:, target_block[0] : target_block[1]],
            direct_block,
            atol=1e-11,
            rtol=1e-11,
        )
        assert np.array_equal(
            exact_prediction[:, : target_block[0]],
            bundle["baseline_prediction"][:, : target_block[0]],
        )
        assert np.array_equal(
            exact_prediction[:, target_block[1] :],
            bundle["baseline_prediction"][:, target_block[1] :],
        )

    grouped = _group_removal_predictions(
        np,
        reference=bundle,
        selected_local_indices=(0, 1),
        target_block=target_block,
    )
    keep = np.ones(len(x_train), dtype=bool)
    keep[list(candidates)] = False
    direct_group_beta = np.linalg.solve(
        z_train[keep].T @ z_train[keep] + penalty,
        z_train[keep].T @ targets[keep, target_block[0] : target_block[1]],
    )
    grouped_exact = grouped["exact_group_prediction"]
    assert np.allclose(
        grouped_exact[:, target_block[0] : target_block[1]],
        z_eval @ direct_group_beta,
        atol=1e-11,
        rtol=1e-11,
    )
    assert grouped["small_matrix_solve_count"] == 1
    assert np.array_equal(
        grouped_exact[:, : target_block[0]],
        bundle["baseline_prediction"][:, : target_block[0]],
    )
    assert np.array_equal(
        grouped_exact[:, target_block[1] :],
        bundle["baseline_prediction"][:, target_block[1] :],
    )

    explicit_hard = _group_removal_predictions(
        np,
        reference=bundle,
        selected_local_indices=(0, 1),
        target_block=target_block,
        removal_strength=1.0,
    )
    assert np.array_equal(
        explicit_hard["exact_group_prediction"], grouped_exact
    )
    assert np.array_equal(
        explicit_hard["first_order_group_proxy_prediction"],
        grouped["first_order_group_proxy_prediction"],
    )

    soft = _group_removal_predictions(
        np,
        reference=bundle,
        selected_local_indices=(0, 1),
        target_block=target_block,
        removal_strength=0.25,
    )
    weights = np.ones(len(x_train), dtype=np.float64)
    weights[list(candidates)] = 0.75
    direct_soft_beta = np.linalg.solve(
        z_train.T @ (weights[:, None] * z_train) + penalty,
        z_train.T @ (weights[:, None] * targets[:, target_block[0] : target_block[1]]),
    )
    assert np.allclose(
        soft["exact_group_prediction"][:, target_block[0] : target_block[1]],
        z_eval @ direct_soft_beta,
        atol=1e-11,
        rtol=1e-11,
    )

    empty = _group_removal_predictions(
        np,
        reference=bundle,
        selected_local_indices=(),
        target_block=target_block,
    )
    assert empty["small_matrix_solve_count"] == 0
    assert np.array_equal(empty["exact_group_prediction"], bundle["baseline_prediction"])
