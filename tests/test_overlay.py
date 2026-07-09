"""tests/test_overlay.py — 2.0-④ overlay 实验使能守卫（切片表可信的前提）。

守：①forced_program 逐字执行（program note/steps=策略程序，非模板 compose 产物）；
②**当前 harness 的 L4/L1 仍在环上**——契约违规的 forced program 被 Contract gate 拦截并走
recovery（策略路由不旁路安全网）；③routing 证据落 EvidenceRecord 并进 store；
④非 routed 路径旧行为不变（routing=None）。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.fast_path.compose import Program, ProgramStep
from SelfEvolvingHarnessTS.fast_path.pipeline import process
from SelfEvolvingHarnessTS.harness import HarnessState
from SelfEvolvingHarnessTS.memory import EvidenceStore


def _x(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(360, dtype=float)
    x = np.sin(2 * np.pi * t / 24) + 0.2 * rng.standard_normal(360)
    x[rng.choice(360, 12, replace=False)] = np.nan
    return x


def test_forced_program_executes_verbatim_with_routing():
    h = HarnessState.from_minimal()
    store = EvidenceStore()
    prog = Program(steps=[ProgramStep("impute_linear", {}),
                          ProgramStep("denoise_median", {"window": 9})],
                   source="policy_overlay", note="overlay:f0_median_w9")
    rec, art = process(_x(), "forecast", h, store=store,
                       forced_program=prog, routing={"selected_action": "f0_median_w9", "uid": "t:0"})
    assert rec.output_status == "ready"
    assert rec.program["note"] == "overlay:f0_median_w9"
    assert [s["op"] for s in rec.program["steps"]] == ["impute_linear", "denoise_median"]
    assert rec.routing["selected_action"] == "f0_median_w9"
    assert len(store) == 1 and np.all(np.isfinite(art))


def test_current_harness_safety_net_not_bypassed():
    """anomaly 契约禁破坏性平滑：forced program 撞 Contract gate → recovery，不静默执行。"""
    h = HarnessState.from_minimal()
    prog = Program(steps=[ProgramStep("impute_linear", {}),
                          ProgramStep("denoise_median", {"window": 25})],
                   source="policy_overlay", note="overlay:bad_anomaly")
    rec, art = process(_x(1), "anomaly_detection", h, forced_program=prog)
    assert rec.output_status in ("fallback_recovery", "fallback_original")   # 安全网接管
    assert rec.program["note"] != "overlay:bad_anomaly" or rec.output_status == "fallback_original"
    assert rec.verification_result["failure_signature"] is not None


def test_overlay_end_to_end_routed():
    from SelfEvolvingHarnessTS.policy import FrozenArmRouterPolicy, action_menu_v1
    from SelfEvolvingHarnessTS.policy.overlay import routed_process_overlay
    from SelfEvolvingHarnessTS.confirmatory_freeze import FROZEN_ARMS
    if not FROZEN_ARMS.exists():
        pytest.skip("冻结臂缺失（新 clone）")
    h = HarnessState.from_minimal()
    store = EvidenceStore()
    router = FrozenArmRouterPolicy.load_frozen("dp_abstain")
    d, rec, art = routed_process_overlay(_x(2), "forecast", h, router, action_menu_v1(),
                                         store=store, extra_routing={"uid": "t:e2e"})
    assert d.action_id in action_menu_v1()
    assert rec.routing["selected_action"] == d.action_id
    assert rec.routing["uid"] == "t:e2e" and "router_artifact_sha" in rec.routing
    assert rec.output_status == "ready"
    assert rec.program["note"] == f"overlay:{d.action_id}"


def test_non_routed_path_unchanged():
    h = HarnessState.from_minimal()
    rec, _ = process(_x(3), "forecast", h)
    assert rec.routing is None and rec.program["source"] in ("template", "llm_custom")
