"""Reliability-field passthrough test (no Schema/Gate change)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "evaluation" / "functional")):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.functional.task_episode_harness.t1 import (
    _make_episode,
    _update_episode_delayed,
)
from evaluation.functional.task_episode_harness.t3 import (
    _source_summaries,
)


def test_support_and_delayed_reliability_fields_pass_through() -> None:
    probe = {
        "macro_gain": 0.022,
        "se_block": 0.021,
        "gain_over_se": 0.022 / 0.021,
    }
    episode = _make_episode(
        attempt_index=0,
        program="impute_ema",
        scope=frozenset({"T117", "T118"}),
        observations={
            "T117": {"local_robust_z_peak": 2.0},
            "T118": {"local_robust_z_peak": 3.0},
        },
        probe=probe,
    )
    assert episode.support_response["gain"] == 0.022
    assert episode.support_response["se_block"] == 0.021
    assert abs(episode.support_response["gain_over_se"] - 0.022 / 0.021) < 1e-12
    assert episode.support_response["accepted"] is True

    updated = _update_episode_delayed(
        episode,
        0.010,
        delayed_se_block=0.020,
        delayed_gain_over_se=0.5,
    )
    assert updated.delayed_response["evaluated"] is True
    assert updated.delayed_response["gain"] == 0.010
    assert updated.delayed_response["se_block"] == 0.020
    assert updated.delayed_response["gain_over_se"] == 0.5

    report = {
        "source_bank": {
            "episodes": [
                updated.to_dict(),
                {
                    "context_summary": {"context_class": "legacy"},
                    "workflow_signature": "outlier_mad",
                    "relation": "POSITIVE",
                    "support_response": {"gain": 0.1, "accepted": True},
                    "delayed_response": {"evaluated": False, "gain": None},
                },
            ]
        }
    }
    summaries = _source_summaries(report)
    new_summary = summaries[0]
    legacy_summary = summaries[1]
    assert new_summary["support_se_block"] == 0.021
    assert abs(new_summary["support_gain_over_se"] - 0.022 / 0.021) < 1e-12
    assert new_summary["delayed_se_block"] == 0.020
    assert new_summary["delayed_gain_over_se"] == 0.5
    assert new_summary["reliability"] == "known"
    assert legacy_summary["support_se_block"] is None
    assert legacy_summary["reliability"] == "unknown"
