"""P0 集成测试（2026-08-15 评审裁定，UPDATE_POLICY_FAULT 修复）：

approved skill → next-round retrieved → support positive → delayed negative
→ removed/restricted → removal 后候选池变化。

零 LLM（SealedProbeBackend 确定性正控）、零真实数据（脚本化 evaluate_fn）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController, FaultRouter, SurfaceRegistry)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore)
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    activate_approved, open_delayed, run_online_round)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway, extract_public_features)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    ScopeExecutor)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    TTHAAgentCore)

DATASET = "revocation_test"
WINNER_OP = "outlier_mad"
# T5 (#41 A5) made the Fast-winner Skill id task-scoped:
# fast_winner_{task_type}_{model_class}_{metric}_{workflow_signature}.
# These arms run the default forecast/ridge/sMASE Consumer, so the
# scope segment is 'forecast_ridge_smase'.
SKILL_ID = f"fast_winner_forecast_ridge_smase_{WINNER_OP}"


def _series() -> np.ndarray:
    # window_context 需要 ≥2 个完整 192 点窗口 → origin ≥ 384；delayed 需要
    # origin+48+96 在界内 → 1024 点，origins 400/448/496
    t = np.arange(1024, dtype=np.float64)
    return np.sin(t / 7.0) + 0.1 * np.sin(t / 3.0) + 5.0


class _ScriptedEval:
    """fake evaluate_fn：按 (origin, ops) + 调用次序出脚本化 mean_smase。

    基线恒 1.0；gain = 1.0 − candidate_mean。
      R1@400  outlier_mad support: 0.90  → +0.10（winner → skill pending）
      @448    第 1 次（R1 delayed）: 0.95 → +0.05（批准）
              第 2 次（R2 首选探测，应为 skill 候选）: 0.98 → +0.02（部署）
              第 3 次起（若首选不是 skill 候选——agent 候选先到）: 1.00 → 0.0
      R2@496  delayed: 1.05 → −0.05（< −M → 必须撤销）
    """

    def __init__(self) -> None:
        self._counts: dict[tuple, int] = {}

    def __call__(self, roster, values, compiled, config, *, origin):
        if compiled is None:
            return {"mean_smase": 1.0, "per_view_smase": [1.0],
                    "behavior_point_count": 10}
        raw = getattr(compiled, "template_steps", None) \
            or getattr(compiled, "steps", None) or ()
        ops_l = []
        for st in raw:
            if isinstance(st, Mapping):
                ops_l.append(str(st.get("op")))
            else:
                op = getattr(st, "op", None)
                if op is None and isinstance(st, (tuple, list)) and st:
                    op = st[0]
                ops_l.append(str(op))
        ops = tuple(ops_l)
        key = (int(origin), ops)
        idx = self._counts.get(key, 0)
        self._counts[key] = idx + 1
        mean = self._script(int(origin), ops, idx)
        return {"mean_smase": mean, "per_view_smase": [mean],
                "behavior_point_count": 10}

    @staticmethod
    def _script(origin: int, ops: tuple[str, ...], idx: int) -> float:
        if ops != (WINNER_OP,):
            return 1.0  # 其他候选恒中性
        if origin == 400:
            return 0.90
        if origin == 448:
            # call0=R1 delayed(+0.05 批准)；call1=R2 skill 候选探测(+0.02
            # 部署)。agent 探索用 denoise_median（程序不同——同程序会按
            # program sha 去重，skill 候选根本不进池）
            return (0.95, 0.98, 1.00, 0.98)[min(idx, 3)]
        if origin == 496:
            return 1.05
        return 1.0


def _round(values, origin, snapshot, store, controller, ev,
           *, prefer_skill: bool):
    # R2/R3 用 detrend_linear 探索：与 skill 的 outlier_mad 程序不同
    # （同程序 sha 去重会让 skill 候选不进池——装置细节，非被测语义）
    agent_op = WINNER_OP if not prefer_skill else "denoise_median"
    backend = sealed.SealedProbeBackend(
        explore=True, operators=(agent_op,), force_pool=True,
        prefer_skill_in_select=prefer_skill)
    series0 = values["s0"]
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(series0[:origin],
                                        task_kind="forecast"))
    method = TTHAMethod(TTHAFastAgent(core), snapshot, ())
    executor = ScopeExecutor(
        [{"series_uid": "s0", "role": "train"},
         {"series_uid": "s0", "role": "eval"}],
        values, {"anchors": []}, evaluate_fn=ev)
    request = runner._a5_request(series0, values, origin, DATASET)
    result = run_online_round(
        method, executor, request, values, origin=origin,
        slow_agent=None, controller=controller, store=store,
        card_builder=runner._a5v2_card,
        round_name=f"revoke_test_{origin}",
        budget=3, allow_slow=False, domain=DATASET, period=24,
        fast_features=dict(extract_public_features(series0[:origin],
                                                   task_kind="forecast")),
        allow_fast_skill=True)
    open_delayed(result, executor, store=store)
    return method, result


def test_delayed_harm_revokes_retrieved_skill(tmp_path):
    values = {"s0": _series()}
    store = SnapshotStore(tmp_path / "store")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    ev = _ScriptedEval()
    h0 = runner._h0_snapshot()

    # R1@400：agent 候选 winner → handle_fast_winner → pending → delayed
    # +0.05 → approved
    method, r1 = _round(values, 400, h0, store, controller, ev,
                        prefer_skill=False)
    assert (r1._fast_skill_event or {}).get("stage") == "pending"
    assert r1.approved_skill_id == SKILL_ID, (
        f"R1 未批准 skill: {r1.approved_skill_id} / {r1._fast_skill_event}")
    activate_approved(r1, store)
    snap1 = method._active_snapshot()  # noqa: SLF001
    assert SKILL_ID in [s.skill_id for s in snap1.skills]

    # R2@448：正常入口——skill retrieved → 部署（support +0.02）→ 不重复
    # ADD（bug 回归点：此前为 apply_failed）→ delayed −0.05 → 撤销
    method2, r2 = _round(values, 448, snap1, store, controller, ev,
                         prefer_skill=True)
    trace2 = method2.last_trace
    assert SKILL_ID in list(getattr(trace2, "retrieved_skill_ids", ()) or ())
    assert r2._winner_candidate_id == f"cand_skill_{SKILL_ID}", (
        "R2 winner 应为 skill 候选，实际探测="
        f"{[(p.get('candidate_id'), p.get('gain'))
            for p in r2.actual_probed_programs]}")
    assert r2.deployed_skill_id == SKILL_ID
    # 事件终态即撤销（revoke 覆写 deployed_existing_skill——部署事实由
    # deployed_skill_id 字段独立断言）
    assert (r2._fast_skill_event or {}).get("stage") \
        == "revoked_delayed_harm"
    assert r2.delayed_utility is not None and r2.delayed_utility < -0.005
    assert r2.revoked_skill_id == SKILL_ID, (
        f"delayed 转害未撤销: event={r2._fast_skill_event}")
    snap2 = method2._active_snapshot()  # noqa: SLF001
    assert SKILL_ID not in [s.skill_id for s in snap2.skills]
    assert r2.revoked_runtime_bundle_sha == snap2.runtime_bundle_sha

    # R3@496：撤销后的正常入口——skill 不再 retrieved，候选池无该候选
    method3, r3 = _round(values, 496, snap2, store, controller, ev,
                         prefer_skill=True)
    trace3 = method3.last_trace
    assert SKILL_ID not in list(
        getattr(trace3, "retrieved_skill_ids", ()) or ())
    probed3 = [p.get("candidate_id") for p in r3.actual_probed_programs]
    assert f"cand_skill_{SKILL_ID}" not in probed3
    assert r3.deployed_skill_id is None
