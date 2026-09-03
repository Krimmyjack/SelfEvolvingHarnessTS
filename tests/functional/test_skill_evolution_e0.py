"""Focused E0 integration test: ADD -> compile -> retrieval -> delayed/revoke.

No live LLM call: the Slow call is monkeypatched with one legal ADD proposal.
Support/delayed evaluators use the already-exposed natural K1 origins; the
revocation phase monkeypatches delayed gain below the material threshold.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import evaluation.functional.task_episode_harness.skill_evolution as se


def _fixed_add(payload: dict) -> dict:
    assert payload["target_task_episode_id"] == se.E0_TARGET_TASK_ID
    assert any(
        payload["source_evidence"][key] is not None
        for key in ("positive", "negative", "conflict")
    )
    return {
        "decision": "ADD",
        "skill_id": "target_local_outlier_mad_v1",
        "workflow": {
            "steps": [{"op": "outlier_mad", "params": {}, "bindings": {}}],
            "requested_observations": [],
            "fallback": "IDENTITY",
            "experience_use": ["natural_k1_01_attempt_1"],
        },
        "scope_rationale": "reuse positive outlier evidence",
        "risk_rationale": "single step, training-window scope only",
    }


def _probe_metrics(gain: float) -> dict:
    return {
        "macro_gain": gain,
        "se_block": 0.02,
        "gain_over_se": gain / 0.02,
        "per_series_mean_gain": {},
        "per_origin_gain": {},
        "positive_series_count": 1 if gain >= se.MATERIAL_THRESHOLD else 0,
        "negative_series_count": 0 if gain >= se.MATERIAL_THRESHOLD else 1,
        "modified_point_count": 1,
        "program_steps": [],
    }


def test_e0_add_compile_retrieval_delayed_and_revocation(monkeypatch, tmp_path):
    report_path = tmp_path / "w1_task_episode_harness_report.json"
    shutil.copy(
        PROJECT_ROOT
        / "artifacts/functional/e2/w1_task_episode_harness_report.json",
        report_path,
    )

    monkeypatch.setattr(se, "_e0_slow_call", _fixed_add)
    passed = se.run_skill_evolution_e0(
        report_path=report_path, set_active=False
    )
    assert passed["verdict"] == "EXPERIENCE_TO_SKILL_ADD_MECHANISM_PASS"
    attempt = passed["attempts"][0]
    assert attempt["body_binding"]["matches_compiled_steps"] is True
    assert attempt["retrieval"]["scope_label"] == (
        "E0_NARROW_SCOPE_RETRIEVAL_PASS"
    )
    assert attempt["retrieval"]["matching_context_match"] is True
    assert attempt["retrieval"]["non_matching_context_match"] is False
    assert attempt["lifecycle"]["delayed_event"]["stage"] == "approved"
    assert attempt["episode"]["local_status"] == "LOCAL_ACTIVE"
    assert attempt["active_snapshot"]["skill_ids"] == [
        "build_contrastive_candidates",
        "fast_winner_forecast_ridge_smase_e0_outlier_mad",
        "inspect_and_localize",
        "select_or_identity_and_verify",
    ]

    # Delayed-harm phase: same ADD proposal, same support path, but delayed
    # gain is material-negative.  Runtime must reject and leave active
    # snapshot unchanged.
    def fake_probe(*args, **kwargs):
        origins = args[3]
        if origins[0] == 1104:
            return _probe_metrics(0.01)
        return _probe_metrics(-0.05)

    monkeypatch.setattr(se, "_probe_compiled", fake_probe)
    restricted = se.run_skill_evolution_e0(
        report_path=report_path, set_active=False
    )
    assert restricted["verdict"] == "SLOW_ADD_DELAYED_RESTRICTED"
    rejected_attempt = restricted["attempts"][0]
    assert rejected_attempt["active_snapshot_changed"] is False
    assert (
        rejected_attempt["lifecycle"]["delayed_event"]["stage"]
        == "delayed_rejected"
    )
    assert rejected_attempt["episode"]["local_status"] == "RESTRICTED"
    assert (
        "fast_winner_forecast_ridge_smase_e0_outlier_mad"
        not in rejected_attempt["active_snapshot_after_delayed_rejection"][
            "skill_ids"
        ]
    )
