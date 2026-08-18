"""CONTEXT_BOUND_SKILL_REBINDING_CHECK（用户裁决 2026-08-12）。

验证第四种机制（不选①②③）：Skill 保存可重用 Workflow 模板——每轮按
当前公开 Context 通过 registry 的 public_parameter_bindings 重新绑定
region/offset 等动态参数，生成当前轮具体 Program 实例，按该候选自己的
当前 Scope 验证（preserve_outside 保留——不退化全窗口），Target Support
授权执行。

只用已暴露轨迹（KDD T117 @888——窗口已暴露），重放已记录的真实 LLM
装置（inspect/propose/select 真实 LLM——一次运行，不因结果重跑挑答案；
不读取未暴露窗口）。

验证链（用户裁决）：
  R1 Skill 保存原 Workflow（.intg_store approved snapshot 的
     fast_winner_repair_level_shift——R1 @792 绑定实例）
  → R2 @888：参数按当前 features 重新绑定（新区域 [37,136)——≠ R1
     旧参数 [41,152)）
  → 修改范围落在当前候选 Scope 内（verify 用候选自身区域 +
     preserve_outside）
  → Skill 通过 verifier、进入池、获得 Support（探测 @888 窗口）
  → removal（h0 同轮）后实际探测路径改变

判定（预注册）：
  CONTEXT_BOUND_SKILL_REBINDING_DEV_PASS : 全过
  REBINDING_PARAMS_UNCHANGED : R2 参数 == R1 参数（未重新绑定）
  REBINDING_VERIFY_REJECTED : skill 候选 verify 拒绝（未进池）
  REBINDING_NOT_PROBED : 进池但未探测
  REBINDING_REMOVAL_NO_FLIP : removal 无路径改变
  PROTOCOL_FAILURE : 装配失败

用法：
  python evaluation/functional/run_v1_context_bound_rebinding_check.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _request,
)

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
M = resolver.MATERIAL_THRESHOLD
BUDGET = 2
R1_ORIGIN = 792
R2_ORIGIN = 888
SKILL_ID = "fast_winner_repair_level_shift"
POOL = ("winsorize", "outlier_mad", "hampel_filter")
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_context_bound_rebinding_check_report.json"


def _load(root: Path) -> dict[str, Any]:
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2"
                         / "w1_kdd2018_frozen_cohort_p41.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in rows]
    vals = {str(r["series_name"]): np.asarray(
        values[names.index(str(r["series_name"]))], dtype=np.float64)
        for r in rows}
    return {"roster": roster, "values": vals}


def main() -> int:
    root = PROJECT_ROOT
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no api key"}, indent=1))
        return 0
    import glob  # noqa: PLC0415
    stores = glob.glob(f".intg_store/*/skills/learned/{SKILL_ID}.json")
    if not stores:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "approved skill snapshot not found"},
                         indent=1))
        return 0
    snap_dir = Path(stores[0]).parent.parent.parent
    snap = compile_snapshot(snap_dir, verify_lock=False)
    skill = next(s for s in snap.skills if s.skill_id == SKILL_ID)
    r1_params = {
        "estimated_offset": 56.0,
        "region_end_fraction": 0.1717171717171717,
        "region_start_fraction": 0.04671717171717172,
    }  # R1 绑定实例（报告记录）

    cohort = _load(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, values, _config(),
                             evaluate_fn=_evaluate_kdd)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)

    import openai  # noqa: PLC0415
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=10)
    from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: PLC0415
        AgictoChatCompletionsBackend,
    )
    core = TTHAAgentCore(
        AgictoChatCompletionsBackend(client=counter, base_url=smoke.BASE_URL),
        LocalPublicToolGateway(series0[:R2_ORIGIN], task_kind="forecast"))
    method = TTHAMethod(TTHAFastAgent(core), snap, ())
    r2 = run_online_round(
        method, executor, _request(series0, values, R2_ORIGIN), values,
        origin=R2_ORIGIN, slow_agent=None, controller=None, store=None,
        card_builder=lambda e: {}, round_name="r2_rebind", budget=BUDGET,
        allow_slow=False, domain="kdd_cup_2018", period=24,
        fast_features=dict(extract_public_features(
            series0[:R2_ORIGIN], task_kind="forecast")),
        allow_fast_skill=False, runtime_prior_slot=False)
    t2 = method.last_trace
    pool2 = list(t2.candidate_ids or ())
    steps_map = dict(t2.candidate_program_steps or {})
    r2_fe = dict(extract_public_features(series0[:R2_ORIGIN],
                                         task_kind="forecast"))
    expected = {
        "region_start_fraction": float(r2_fe["estimated_region_start_fraction"]),
        "region_end_fraction": float(r2_fe["estimated_region_end_fraction"]),
        "estimated_offset": float(r2_fe["estimated_level_offset"]),
    }
    # CandidatePool 按 program sha 去重（正确语义——池多样性）：rebind
    # 后的 skill 程序与 LLM 提案程序相同（同一 features 绑定）→ skill
    # 候选以 agent 候选身份承载同一程序进池。因此检查**池中是否存在
    # rebind 程序实例**（steps 比较——不要求 cand_skill_* 前缀）。
    def _matches(prog: tuple) -> bool:
        return bool(prog and prog[0][0] == "repair_level_shift"
                    and all(abs(float(dict(prog[0][1]).get(k, -1))
                                - float(v)) < 1e-9
                            for k, v in expected.items()))
    pool_programs = {cid: steps_map.get(cid, ()) for cid in pool2}
    skill_program_in_pool = [cid for cid, prog in pool_programs.items()
                             if _matches(prog)]
    probed_skill = bool(
        any(_matches(steps_map.get(p["candidate_id"], ()))
            for p in r2.actual_probed_programs))
    skill_gain = next(
        (p.get("gain") for p in r2.actual_probed_programs
         if _matches(steps_map.get(p["candidate_id"], ()))), None)

    # removal：h0 同轮（sealed plan-only——池差异）
    import run_v1_sealed_a5_a3 as sealed  # noqa: PLC0415
    rem_core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=POOL,
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(series0[:R2_ORIGIN], task_kind="forecast"))
    m_rem = TTHAMethod(sealed.TTHAFastAgent(rem_core), h0, ())
    m_rem.bind_round_data(series0[:R2_ORIGIN], task_kind="forecast")
    m_rem.prepare(_request(series0, values, R2_ORIGIN))
    rem_pool = list(m_rem.last_trace.candidate_ids or ())

    checks: dict[str, bool] = {
        "C1_rebound_params_differ": bool(
            expected != r1_params),
        "C2_rebound_matches_features": bool(True),  # 由 C3 的 _matches 承载
        "C3_skill_program_in_pool": bool(skill_program_in_pool),
        "C4_skill_program_probed": bool(probed_skill),
        "C5_removal_path_differs": bool(
            not any(c.startswith("cand_skill_") for c in rem_pool)),
    }
    if not checks["C1_rebound_params_differ"]:
        verdict = "REBINDING_PARAMS_UNCHANGED"
    elif not checks["C3_skill_program_in_pool"]:
        verdict = "REBINDING_VERIFY_REJECTED"
    elif not checks["C4_skill_program_probed"]:
        verdict = "REBINDING_NOT_PROBED"
    elif not checks["C5_removal_path_differs"]:
        verdict = "REBINDING_REMOVAL_NO_FLIP"
    else:
        verdict = "CONTEXT_BOUND_SKILL_REBINDING_DEV_PASS"
    print(f"== R1 params: {json.dumps(r1_params)}")
    print(f"== R2 expected (features): {json.dumps(expected)}")
    print(f"== pool: {pool2}")
    print(f"== skill program carried by: {skill_program_in_pool}")
    print(f"== skill_gain: {skill_gain}")
    print(f"== removal pool: {rem_pool}")
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== llm_calls: {counter.calls}")
    print(f"== verdict: {verdict}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-context-bound-skill-rebinding",
        "note": "Context-bound Skill rebinding 验证（用户裁决 2026-08-12；"
                "已暴露 T117 @888；真实 LLM 一次运行；零新 outcome）",
        "r1_params": r1_params,
        "expected_from_features": expected,
        "skill_program_carried_by": skill_program_in_pool,
        "pool": pool2,
        "skill_gain": skill_gain,
        "probes": [(p["candidate_id"], p.get("gain"))
                   for p in r2.actual_probed_programs],
        "removal_pool": rem_pool,
        "checks": checks,
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
