from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.benchmark.baselines import RawBaseline
from SelfEvolvingHarnessTS.policy.task_spec import forecast_task_spec_v1
from SelfEvolvingHarnessTS.vnext.evaluator import prepare_published_joint_v02
from SelfEvolvingHarnessTS.vnext.method import VNextBenchmarkMethod


def test_generic_evaluator_adapter_exposes_no_private_metadata_and_preserves_joint_corpus():
    histories = {"u1": np.arange(144, dtype=float), "u2": np.arange(144, dtype=float) + 1}
    inner = {uid: values[:-48] for uid, values in histories.items()}
    corpus = prepare_published_joint_v02(
        RawBaseline(), histories, inner, forecast_task_spec_v1(horizon=48),
    )
    assert corpus.estimand == "published_joint_v02"
    assert set(corpus.history) == set(histories)
    assert all(corpus.operators[uid] == () for uid in histories)
    assert len(corpus.artifact_sha) == 64


def test_vnext_method_is_invariant_to_history_inner_call_order():
    histories = {"u1": np.arange(144, dtype=float), "u2": np.arange(144, dtype=float) + 1}
    inner = {uid: values[:-48] for uid, values in histories.items()}
    task = forecast_task_spec_v1(horizon=48)
    first = prepare_published_joint_v02(
        VNextBenchmarkMethod(), histories, inner, task, call_order="history_then_inner",
    )
    second = prepare_published_joint_v02(
        VNextBenchmarkMethod(), histories, inner, task, call_order="inner_then_history",
    )
    assert first.artifact_sha == second.artifact_sha
