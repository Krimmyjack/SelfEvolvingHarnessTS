"""N5 增长态回归：Memory 增长（pack + Target-local episode）不得断线。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    build_episode, workflow_signature_of)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features)
import signed_radius as resolver  # noqa: E402


def _target_episode(local_pattern):
    return build_episode(
        episode_id="reg_growth",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace="monash_traffic_hourly_dev",
        context_summary={"local_pattern": local_pattern,
                         "delayed_pattern": {},
                         "program_geometry": {"scope": "training_rows"}},
        workflow_signature=workflow_signature_of(
            [{"op": "outlier_mad", "params": {}}]),
        support_response={"gain": 0.05, "accepted": True},
        delayed_response={"evaluated": True, "gain": 0.02},
        relation="POSITIVE", evidence_level="DELAYED",
        local_status="LOCAL_DRAFT", evidence_refs=["reg"])


def test_correct_keys_keep_contrast_branch():
    """extract_public_features 键 → signed 分支不触发（2026-08-15 bug 修复）。"""
    arr = np.sin(np.arange(400, dtype=np.float64) / 6.0)
    lp = {"support_gain": 0.05,
          **extract_public_features(arr, task_kind="forecast")}
    ep = _target_episode(lp)
    assert runner._n5_signed_absent([ep]) is True


def test_window_context_keys_flip_branch_bug_regression():
    """window_context 的 recent./change. 键会触发 signed 分支（事故形态）。"""
    arr = np.sin(np.arange(700, dtype=np.float64) / 6.0)
    lp = {"support_gain": 0.05,
          **resolver.window_context({"s": arr}, 600, 24)}
    assert any(str(k).startswith(("recent.", "change.")) for k in lp)
    ep = _target_episode(lp)
    assert runner._n5_signed_absent([ep]) is False


def test_grown_memory_still_renders_contrast_pack():
    """pack + 正确构造的 Target-local episode → resolve 仍渲染非空。"""
    episodes = runner._n5_load_pack_episodes()
    if not episodes:
        return  # pack 缺失时跳过（环境保护）
    arr = np.sin(np.arange(700, dtype=np.float64) / 6.0)
    lp = {"support_gain": 0.05,
          **extract_public_features(arr[:600], task_kind="forecast")}
    grown = list(episodes) + [_target_episode(lp)]
    feats = extract_public_features(arr[:600], task_kind="forecast")
    out = runner._n5_wiring_check(grown, feats)
    assert out["memory_resolution_status"] == "contrast_pack"
    assert out["rendered_len"] > 0
