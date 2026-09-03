"""Focused E1-v2 protocol repair test.

No live LLM and no real outcome: the Slow calls and Support/delayed evaluators
are monkeypatched.  The test verifies the three requested protocol properties:

1. A3 Target history contains no A5 episode and vice versa;
2. every Support / delayed / cross-Task truth window is pairwise non-overlap
   with HORIZON=48 and uses blocks after E1-v1 exposure;
3. a LOCAL_ACTIVE Target-local Skill is retrieved by the same arm on the next
   Task and is machine-reused instead of re-ADDed.
"""
from __future__ import annotations

import json
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

from evaluation.functional.task_episode_harness import e1
from evaluation.functional.task_episode_harness import runner
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore


def _public_context(observation_cutoff: int) -> dict:
    return {
        "task_kind": "forecast",
        "observation_cutoff": observation_cutoff,
        "scope_feature": "local_robust_z_peak",
        "scope_bin": "high",
        "projection_feature": "estimated_region_start_fraction",
        "scope_series_uids": ["T153", "T154"],
        "representative_uid": "T153",
        "representative_features": {
            "task_kind": "forecast",
            "estimated_region_start_fraction": "medium",
        },
        "task_signature": {
            "task_kind": "forecast",
            "estimated_region_start_fraction": "medium",
        },
        "task_fast_features": {
            "task_kind": "forecast",
            "estimated_region_start_fraction": "medium",
        },
    }


def _probe_metrics(*args, **kwargs) -> dict:
    return {
        "macro_gain": 0.01,
        "se_block": 0.02,
        "gain_over_se": 0.5,
        "per_series_mean_gain": {},
        "per_origin_gain": {},
        "positive_series_count": 2,
        "negative_series_count": 0,
        "modified_point_count": 1,
        "program_steps": [],
    }


def _fake_slow(messages: list[dict]) -> dict:
    system = messages[0]["content"]
    if system == e1._E1_PROPOSAL_SYSTEM:
        payload = json.loads(messages[1]["content"])
        op = "hampel_filter" if payload.get("source_prior") else "outlier_mad"
        return {
            "decision": "PROPOSE",
            "reason": "fixed protocol test",
            "proposals": [
                {
                    "steps": [{"op": op, "params": {}, "bindings": {}}],
                    "requested_observations": [],
                    "fallback": "IDENTITY",
                    "experience_use": [],
                }
            ],
        }
    if system == e1._DECISION_SYSTEM:
        return {"decision": "TRUST_DRAFT", "reason": "fixed protocol test"}
    raise AssertionError(f"unexpected Slow prompt: {system[:80]!r}")


def _fake_compile(proposal, inventory, public_context, *, generation):
    op = proposal["steps"][0]["op"]
    compiled = runner._compiled(op, name=f"test-{op}")
    return compiled, {"decision": "PROPOSE", "steps": proposal["steps"]}


def _make_arm_state(tmp_path: Path, arm: str):
    h0 = compile_snapshot(
        PROJECT_ROOT / "methods/ttha/harness/h0", verify_lock=False
    )
    store = SnapshotStore(tmp_path / arm / "snapshots")
    store.materialize(h0)
    store.set_active(h0.runtime_bundle_sha)
    return e1._ArmState(
        arm=arm,
        memories=[],
        episodes=[],
        store=store,
        active_snapshot=h0,
        active_skill_ids=[],
    )


def test_e1_v2_arm_isolation_window_non_overlap_and_local_skill_reuse(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(e1, "_e1_slow_call", _fake_slow)
    monkeypatch.setattr(e1, "_probe_compiled", _probe_metrics)
    monkeypatch.setattr(e1, "_compile_proposal", _fake_compile)

    # Frozen roster check: HORIZON=48, pairwise non-overlap, unexposed blocks.
    roster = e1._frozen_task_roster()
    assert len(roster) >= e1.N0
    ok, windows, audit = e1._all_truth_windows_non_overlapping(roster)
    assert ok is True
    assert audit["horizon"] == 48
    assert audit["window_count"] == 6 * len(roster)
    assert all(row["end"] - row["start"] == 48 for row in windows)
    assert all(
        windows[i]["start"] >= windows[i - 1]["end"]
        for i in range(1, len(windows))
    )
    assert audit["first_window_start"] == 3072 > 3054

    # Task 1: A3 and A5 run from independent empty arm states.
    spec1 = {
        "task_episode_id": "e1v2_task_01",
        "arm_order": "A3_A5",
        "horizon": 48,
        "support_origins": (100, 148, 196),
        "delayed_origins": (244, 292, 340),
    }
    context1 = _public_context(100)
    a3_state = _make_arm_state(tmp_path, "A3")
    a5_state = _make_arm_state(tmp_path, "A5")
    llm_counter = [0]
    a3_row = e1._run_arm(
        repo_root=PROJECT_ROOT,
        arm_state=a3_state,
        task_spec=spec1,
        public_context=context1,
        source_prior=None,
        inventory=[],
        values={},
        mapped_roster=[],
        config={},
        eval_uids=[],
        llm_counter=llm_counter,
    )
    a5_row = e1._run_arm(
        repo_root=PROJECT_ROOT,
        arm_state=a5_state,
        task_spec=spec1,
        public_context=context1,
        source_prior={"source_card": {"skill_id": "e0_source"}, "source_evidence": {}},
        inventory=[],
        values={},
        mapped_roster=[],
        config={},
        eval_uids=[],
        llm_counter=llm_counter,
    )

    # Task 1 inputs are identical after removing the Source prior block.
    assert (
        e1._normalized_payload_fingerprint(a3_row["payload"])
        == e1._normalized_payload_fingerprint(a5_row["payload"])
    )
    assert a3_row["winner"]["local_status"] == "LOCAL_ACTIVE"
    assert a5_row["winner"]["local_status"] == "LOCAL_ACTIVE"

    # History isolation: A3 history has no A5 episode and vice versa.
    a3_memory_ids = {
        str(memory["episode_id"]) for memory in a3_row["target_memories_after"]
    }
    a5_memory_ids = {
        str(memory["episode_id"]) for memory in a5_row["target_memories_after"]
    }
    assert a3_memory_ids, "A3 history must contain its own episode"
    assert a5_memory_ids, "A5 history must contain its own episode"
    assert all(episode_id.startswith("e1v2_A3_") for episode_id in a3_memory_ids)
    assert not any(
        episode_id.startswith("e1v2_A5_") for episode_id in a3_memory_ids
    )
    assert all(episode_id.startswith("e1v2_A5_") for episode_id in a5_memory_ids)
    assert not any(
        episode_id.startswith("e1v2_A3_") for episode_id in a5_memory_ids
    )
    assert a3_row["active_local_skill_ids_after"] == [
        "fast_winner_forecast_ridge_smase_e1v2_outlier_mad"
    ]
    assert a5_row["active_local_skill_ids_after"] == [
        "fast_winner_forecast_ridge_smase_e1v2_hampel_filter"
    ]

    # Same-arm next-Task retrieval and machine reuse.
    spec2 = dict(
        spec1,
        task_episode_id="e1v2_task_02",
        support_origins=(388, 436, 484),
        delayed_origins=(532, 580, 628),
    )
    context2 = _public_context(388)
    a3_retrieved = e1._retrieve_target_local_skills(
        a3_state.active_snapshot, context2, arm="A3"
    )
    a5_retrieved = e1._retrieve_target_local_skills(
        a5_state.active_snapshot, context2, arm="A5"
    )
    assert [row["skill_id"] for row in a3_retrieved] == [
        "fast_winner_forecast_ridge_smase_e1v2_outlier_mad"
    ]
    assert a3_retrieved[0]["retrieved_in_current_context"] is True
    assert [row["skill_id"] for row in a5_retrieved] == [
        "fast_winner_forecast_ridge_smase_e1v2_hampel_filter"
    ]
    assert a5_retrieved[0]["retrieved_in_current_context"] is True

    a3_row2 = e1._run_arm(
        repo_root=PROJECT_ROOT,
        arm_state=a3_state,
        task_spec=spec2,
        public_context=context2,
        source_prior=None,
        inventory=[],
        values={},
        mapped_roster=[],
        config={},
        eval_uids=[],
        llm_counter=llm_counter,
    )
    assert a3_row2["target_local_skills_before"][0][
        "retrieved_in_current_context"
    ] is True
    assert a3_row2["lifecycle"]["reused_existing_skill"] is True
    assert a3_row2["lifecycle"]["method_event"]["stage"] == (
        "deployed_existing_skill"
    )
    assert a3_row2["winner"]["local_status"] == "LOCAL_ACTIVE"
    assert a3_row2["active_local_skill_ids_after"] == [
        "fast_winner_forecast_ridge_smase_e1v2_outlier_mad"
    ]
    # Reuse did not create a duplicate skill entry.
    local_skill_ids = [
        skill.skill_id
        for skill in a3_state.active_snapshot.skills
        if e1._is_local_skill_id(skill.skill_id)
    ]
    assert local_skill_ids == ["fast_winner_forecast_ridge_smase_e1v2_outlier_mad"]


def _empty_arm_block(arm: str, episode_id: str, skill_id: str) -> dict:
    return {
        "target_memories_after": [{"episode_id": episode_id}],
        "active_local_skill_ids_after": [skill_id],
        "winner": None,
        "metrics": {
            "task_probe_cost": 0,
            "harmful_probe_count": 0,
            "cumulative_support_harm": 0.0,
            "task_local_active": 0,
            "task_delayed_utility": None,
        },
    }


def test_identical_skill_id_in_two_arm_snapshots_is_not_contamination():
    """Minimal review assertion for E1-v2 first-fault review.

    Two independent arm stores may machine-generate the same skill_id.  That
    overlap is observational and must not flip ``history_isolation.pass``;
    only cross-arm Episode history contamination does.
    """
    shared_skill_id = "fast_winner_forecast_ridge_smase_e1v2_repair_level_shift"
    rows = [
        {
            "task_episode_id": "e1v2_task_10",
            "A3": _empty_arm_block(
                "A3", "e1v2_A3_e1v2_task_10_attempt_0", shared_skill_id
            ),
            "A5": _empty_arm_block(
                "A5", "e1v2_A5_e1v2_task_10_attempt_0", shared_skill_id
            ),
        }
    ]
    summary = e1._paired_summary(rows)
    isolation = summary["history_isolation"]
    assert isolation["a3_a5_skill_snapshot_disjoint"] is False
    assert isolation["pass"] is True


def test_e1_v3_source_prior_scope_routing():
    """Source package enters A5 only for the matching medium Context.

    This is the E1-v3 Scope-routing repair test: zero/low/very_low Contexts
    must receive no Source Card and no Source evidence, so no Source workflow
    or operator name can leak into the A5 generation payload.
    """
    source_prior = {
        "source_card": {
            "skill_id": "fast_winner_e0_outlier_mad_repair_level_shift",
            "observable_applicability": (
                "{'all': (mappingproxy({'feature': 'task_kind', "
                "'op': '==', 'value': 'forecast'}), "
                "mappingproxy({'feature': "
                "'estimated_region_start_fraction', 'op': '==', "
                "'value': 'medium'}))}"
            ),
        },
        "source_evidence": {
            "positive": {"workflow": "outlier_mad"},
            "negative": {"workflow": "hampel_filter"},
            "conflict": None,
            "non_empty": True,
        },
    }
    for projection_bin in ("zero", "low", "very_low"):
        context = _public_context(100)
        context["task_fast_features"] = {
            "task_kind": "forecast",
            "estimated_region_start_fraction": projection_bin,
        }
        context["task_signature"] = {
            "task_kind": "forecast",
            "estimated_region_start_fraction": projection_bin,
        }
        assert e1._source_prior_for_task(source_prior, context) is None

    medium = _public_context(100)
    medium["task_fast_features"] = {
        "task_kind": "forecast",
        "estimated_region_start_fraction": "medium",
    }
    medium["task_signature"] = {
        "task_kind": "forecast",
        "estimated_region_start_fraction": "medium",
    }
    routed = e1._source_prior_for_task(source_prior, medium)
    assert routed is not None
    assert routed["source_card"]["skill_id"] == (
        "fast_winner_e0_outlier_mad_repair_level_shift"
    )
    assert routed["source_evidence"] == source_prior["source_evidence"]
