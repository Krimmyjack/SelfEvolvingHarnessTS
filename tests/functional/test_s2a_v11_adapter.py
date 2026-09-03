"""S2a v1.1 focused tests: three authorized dispatch sites, zero semantic drift.

(a) The three identity constants default to the classification triple and
    restore to it.
(b) ``_scope_v1_admits`` treats forecast as a legal task-axis value and still
    matches the card's ``task_kind`` against the bound identity.
(c) ``contracted_axes`` dispatches the in-service extractor by ``task_kind``;
    the classification default is the historical call.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

from SelfEvolvingHarnessTS.runtime.public_features import (  # noqa: E402
    extract_public_features,
)
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    skill_revision as rev,
)
import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
import run_e2_s1a_curriculum_oracle_audit as s1a  # noqa: E402

PATTERN = {"local_robust_z_peak": "high", "missing_fraction": "zero"}


def _scope(*, task_kind: str) -> dict:
    return {
        "task_kind": task_kind,
        "consumer_id": s1.CONSUMER_ID,
        "metric": s1.METRIC,
        "pattern_intersection": dict(PATTERN),
        "program_geometry": ["hampel_filter"],
    }


def test_a_identity_defaults_stay_classification_and_restore() -> None:
    s1.bind_curriculum_identity()
    assert s1.TASK_KIND == s1a.TASK_KIND == "classification"
    assert s1.CONSUMER_ID == s1a.CONSUMER_ID
    assert s1.METRIC == s1a.METRIC
    bound = s1.bind_curriculum_identity(
        task_kind=s1.FORECAST_TASK_KIND,
        consumer_id=s1.FORECAST_CONSUMER_ID,
        metric=s1.FORECAST_METRIC)
    assert bound == {
        "task_kind": "forecast",
        "consumer_id": "pooled_ridge_a1",
        "metric": "sMASE",
    }
    restored = s1.bind_curriculum_identity()
    assert restored == {
        "task_kind": "classification",
        "consumer_id": s1a.CONSUMER_ID,
        "metric": s1a.METRIC,
    }


def test_b_scope_axis_admits_forecast_and_still_matches_the_card() -> None:
    s1.bind_curriculum_identity()
    cls_ok = s1._scope_v1_admits(_scope(task_kind="classification"), PATTERN)
    assert cls_ok["admits"] is True
    fc_on_cls = s1._scope_v1_admits(_scope(task_kind="forecast"), PATTERN)
    assert fc_on_cls["admits"] is False
    assert "task_kind" in fc_on_cls["why"]

    s1.bind_curriculum_identity(
        task_kind="forecast",
        consumer_id=s1.FORECAST_CONSUMER_ID,
        metric=s1.FORECAST_METRIC)
    try:
        fc_scope = {
            "task_kind": "forecast",
            "consumer_id": s1.FORECAST_CONSUMER_ID,
            "metric": s1.FORECAST_METRIC,
            "pattern_intersection": dict(PATTERN),
            "program_geometry": ["hampel_filter"],
        }
        fc_ok = s1._scope_v1_admits(fc_scope, PATTERN)
        assert fc_ok["admits"] is True
        cls_on_fc = s1._scope_v1_admits(
            {**fc_scope, "task_kind": "classification"}, PATTERN)
        assert cls_on_fc["admits"] is False
        assert "task_kind" in cls_on_fc["why"]
    finally:
        s1.bind_curriculum_identity()


def test_c_contracted_axes_dispatch_keeps_classification_bytes() -> None:
    probe = np.arange(128, dtype=np.float64)
    historical = set(extract_public_features(
        probe, task_kind="classification").mapping)
    historical.discard("task_kind")
    defaulted = rev.contracted_axes()
    classified = rev.contracted_axes(task_kind="classification")
    forecasted = rev.contracted_axes(task_kind="forecast")
    assert defaulted == classified == frozenset(historical)
    forecast_emitted = set(extract_public_features(
        probe, task_kind="forecast").mapping)
    forecast_emitted.discard("task_kind")
    assert forecasted == frozenset(forecast_emitted)
    try:
        rev.contracted_axes(task_kind="anomaly_detection")
        raise AssertionError("non-forecast non-classification must raise")
    except ValueError:
        pass
