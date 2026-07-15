"""Generic benchmark-native method preparation for the published joint v0.2 estimand.

This module deliberately delegates normalization, windowing, training, metrics, folding,
and bootstrap to :mod:`benchmark`.  It only adapts an arbitrary BenchmarkMethod to the
same history/inner corpus shape already consumed by the frozen v0.2 evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ..benchmark.method_api import BenchmarkMethod, MethodSeriesView, validate_prepared
from ..policy.task_spec import TaskSpec
from ._canonical import sha256


@dataclass(frozen=True)
class PreparedJointCorpus:
    history: Mapping[str, np.ndarray]
    inner: Mapping[str, np.ndarray]
    operators: Mapping[str, tuple[str, ...]]
    method_id: str
    estimand: str = "published_joint_v02"
    inner_operators: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def artifact_sha(self) -> str:
        payload = {
            "method_id": self.method_id,
            "estimand": self.estimand,
            "history": {uid: np.asarray(values, dtype="<f8").tobytes().hex()
                        for uid, values in sorted(self.history.items())},
            "inner": {uid: np.asarray(values, dtype="<f8").tobytes().hex()
                      for uid, values in sorted(self.inner.items())},
            "operators": dict(self.operators),
            "inner_operators": dict(self.inner_operators),
        }
        return sha256(payload)


def prepare_published_joint_v02(
    method: BenchmarkMethod,
    histories: Mapping[str, np.ndarray],
    inner: Mapping[str, np.ndarray],
    task_spec: TaskSpec,
    *,
    call_order: str = "history_then_inner",
) -> PreparedJointCorpus:
    """Apply a method to both benchmark-owned joint paths, with no private metadata.

    The caller remains responsible for grouping fits by public ``dataset_id``.  This
    adapter never accepts dataset, regime, split role, future, or labels, preventing those
    fields from reaching the method surface.
    """
    if set(histories) != set(inner) or not histories:
        raise ValueError("published joint history/inner uid sets must match and be non-empty")
    if call_order not in {"history_then_inner", "inner_then_history"}:
        raise ValueError("call_order is not a registered evaluator order")
    prepared_history: dict[str, np.ndarray] = {}
    prepared_inner: dict[str, np.ndarray] = {}
    operators: dict[str, tuple[str, ...]] = {}
    inner_operators: dict[str, tuple[str, ...]] = {}
    for uid in sorted(histories):
        if call_order == "history_then_inner":
            full = method.prepare(MethodSeriesView(uid, histories[uid]), task_spec, {})
            train = method.prepare(MethodSeriesView(uid, inner[uid]), task_spec, {})
        else:
            train = method.prepare(MethodSeriesView(uid, inner[uid]), task_spec, {})
            full = method.prepare(MethodSeriesView(uid, histories[uid]), task_spec, {})
        full_verdict = validate_prepared(full, expected_length=len(histories[uid]))
        train_verdict = validate_prepared(train, expected_length=len(inner[uid]))
        if full.series_uid != uid or train.series_uid != uid:
            raise ValueError("method changed series_uid")
        if not full_verdict.valid or not train_verdict.valid:
            raise ValueError(
                f"method contract failed for {uid}: {full_verdict.code}/{train_verdict.code}"
            )
        prepared_history[uid] = full.values
        prepared_inner[uid] = train.values
        operators[uid] = full.operators
        inner_operators[uid] = train.operators
    return PreparedJointCorpus(
        history=prepared_history, inner=prepared_inner, operators=operators,
        inner_operators=inner_operators, method_id=str(method.method_id),
    )
