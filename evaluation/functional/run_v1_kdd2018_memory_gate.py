"""KDD_CUP_2018_MEMORY_ACTIONABILITY_GATE（P4.0，用户裁决 2026-08-11）。

零新数据 plan-only Gate：跨域 Memory（Monash outlier-family Source
Episodes）在 KDD 已暴露 Context 上是否改变进入预算的 top-2 候选集合。

固定条件（用户裁决）：
  - Source 只用 Monash 与 KDD 语义兼容的 outlier-family Episode（winsorize/
    outlier_iqr——排除 repair：日频 vs 小时频参数绑定不可比）；
  - 候选池 ≥3 verifier 合法（KDD @600：winsorize/outlier_mad/hampel——
    outlier_iqr 静态不合法）；Support 预算 2；
  - A5/A3 候选池/顺序/Context/LLM 请求/预算全同——唯一变量 Source Memory；
  - 检查 A5 是否改变进入预算的 top-2 候选集合（不只排序）。

三干预：M_source（真实 Monash Episodes）/ M_remove（空）/ M_signswap
（同候选同 Context——经验符号交换——负控）。每干预 2 次（LLM select——
稳定检查，不投票）。

Gate 通过条件（5 条）：
  1. Source Episode 被 resolver 正确匹配并渲染（M_source 的 Reference 非空）；
  2. 相同输入重复两次 top-2 集合稳定；
  3. M_source 与 M_remove 的 top-2 集合不同；
  4. 变化方向能由 Source Experience 解释（POSITIVE→提前、NEGATIVE→降级）；
  5. 不通过投票掩盖 LLM 方差。

三臂 top-2 相同 → MEMORY_ACTION_SIGNAL_ABSENT（不消耗 virgin 数据）。

用法：
  python evaluation/functional/run_v1_kdd2018_memory_gate.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_bounded_two_candidate_runtime_control import probe_order  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _load_cohort,
)

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
ORIGIN = 600
POOL = ("winsorize", "outlier_mad", "hampel_filter")  # KDD @600 3 合法
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2/w1_kdd2018_memory_gate_report.json"
MONASH_CACHE = PROJECT_ROOT / "data/monash_weather_v1/series_cache.npz"
MONASH_TYPES = PROJECT_ROOT / "data/monash_weather_v1/series_types.json"


def _monash_source_episodes(root: Path) -> list[Any]:
    """重建 Monash outlier-family Source Episodes（已暴露报告——零新
    outcome）：**单条 winsorize 双正**（pair1——n_hist=2 < 3 → weak_
    reference 成对判定——跨域 radius 距离在日频/小时频间不可比（诊断
    2026-08-11：4 条 → n_hist=6 → radius 模式全 UNKNOWN——渲染空）。
    排除 repair（频率参数绑定不可比）。"""
    cache = np.load(root / MONASH_CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    rep = json.loads((root / "artifacts/functional/e2"
                      / "w1_monash_fresh_a5_a3_report_pair1.json")
                     .read_text(encoding="utf-8"))
    src = rep["pairs"][0]["source"]
    eps: list[Any] = []
    for p in src.get("probes", []):
        if p.get("op") != "winsorize":
            continue
        sg = float(p["gain"])
        dg = float(src.get("delayed_gain") or 0.0)
        ep = tll.write_target_episode(
            domain="monash_weather_daily", op="winsorize",
            episode_id_suffix="_p40_src_pair1",
            program_steps=[{"op": "winsorize", "params": {}}],
            support_gain=sg, delayed_gain=None,
            support_context=dict(resolver.window_context(
                _src_values(root, names, values, "pair1"), 600, 24)))
        ep = tll.update_delayed_status(
            ep, dg,
            delayed_context=dict(resolver.window_context(
                _src_values(root, names, values, "pair1"), 648, 24)))
        eps.append(ep)
        break
    return eps


def _src_values(root: Path, names: list[str], values: Any,
                pair: str) -> dict[str, np.ndarray]:
    """两候选版 pair 的 src cohort 首支（roster 文件 C0/C2/C4 首支）。"""
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2"
                         / "w1_monash_frozen_roster.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    idx = {"pair1": 0, "pair2": 40, "pair3": 80}[pair]
    uid = str(rows[idx]["series_name"])
    return {uid: np.asarray(values[names.index(uid)], dtype=np.float64)}


def _signswap(eps: Sequence[Any]) -> list[Any]:
    """经验符号交换（负控）：support/delayed gain 取反——relation 反转。"""
    out: list[Any] = []
    for ep in eps:
        sg = -float((ep.support_response or {}).get("gain") or 0.0)
        dg = -float((ep.delayed_response or {}).get("gain") or 0.0)
        nv = tll.write_target_episode(
            domain="monash_weather_daily",
            op=str(getattr(ep, "workflow_signature", "?")),
            episode_id_suffix=f"_p40_swap_{getattr(ep, 'episode_id', '?')}",
            program_steps=list((ep.context_summary or {})
                               .get("program_geometry", {})
                               .get("program_steps") or []),
            support_gain=sg, delayed_gain=None,
            support_context=dict((ep.context_summary or {})
                                 .get("local_pattern") or {}))
        nv = tll.update_delayed_status(
            nv, dg,
            delayed_context=dict((ep.context_summary or {})
                                 .get("delayed_pattern") or {}))
        out.append(nv)
    return out


def _decide(root: Path, h0: Any, cohort: dict[str, Any], series0: np.ndarray,
            memory: Sequence[Any], counter: Any, *,
            label: str) -> dict[str, Any]:
    """plan-only：真实 LLM select——top-2 = probe_order（chosen 优先 +
    signed 排序）——不 evaluate（不读 Target outcome）。"""
    roster, values = cohort["roster"], cohort["values"]
    ctx = dict(resolver.window_context(values, ORIGIN, PERIOD))
    ctx["bound_period"] = float(PERIOD)
    backend = sealed.LLMSelectBackend(
        explore=True, operators=POOL, client=counter,
        context_plain=dict(ctx), max_propose_candidates=3,
        force_pool=True)
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            backend,
            LocalPublicToolGateway(series0[:ORIGIN],
                                   task_kind="forecast"))),
        h0, tuple(memory))
    method.bind_round_data(series0[:ORIGIN], task_kind="forecast")
    result = method.prepare(sealed._request(series0, values, ORIGIN))
    trace = method.last_trace
    chosen = trace.chosen_candidate_id
    steps_map = dict(trace.candidate_program_steps or {})
    pool_ops = [c[len("cand_"):] for c in trace.candidate_ids
                if c.startswith("cand_") and c in steps_map]
    signed_ranks: dict[str, int] = {}
    for op in (backend._deprioritized or []):  # noqa: SLF001
        signed_ranks[str(op)] = 2
    order = probe_order(pool_ops, chosen, signed_ranks)
    instruction = ""
    for req in backend.requests:
        for m in req.messages:
            # 消息经 _freeze_json 为 MappingProxyType——isinstance(m, dict)
            # 恒 False（审查 2026-08-11）——用 Mapping
            c = m.get("content") if isinstance(m, Mapping) else None
            if isinstance(c, str) and "The following references" in c:
                instruction = c
                break
        if instruction:
            break
    # ref1（POSITIVE）从 instruction 解析——提前（rank 0）
    import re
    ref1_ops: list[str] = []
    m = re.search(r"Reference 1: candidate operators \[([^\]]*)\]",
                  instruction)
    if m:
        ref1_ops = [x.strip().strip("'\"").strip()
                    for x in m.group(1).split(",") if x.strip()]
    for op in ref1_ops:
        signed_ranks.setdefault(op, 0)
    order = probe_order(pool_ops, chosen, signed_ranks)
    return {"label": label, "chosen": chosen,
            "top2": list(order[:2]), "pool": list(trace.candidate_ids),
            "reference_rendered": bool(instruction),
            "instruction_head": instruction[:300]}


def main() -> int:
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    cohort = _load_cohort(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — PROTOCOL_FAILURE")
        return 0
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=10)

    m_source = _monash_source_episodes(root)
    print(f"== M_source episodes: "
          f"{[(getattr(e, 'workflow_signature', '?'), getattr(e, 'relation', '?')) for e in m_source]}")
    arms: dict[str, list[dict[str, Any]]] = {}
    for arm, memory in (("M_source", m_source),
                        ("M_remove", []),
                        ("M_signswap", _signswap(m_source))):
        arms[arm] = [_decide(root, h0, cohort, series0, memory, counter,
                             label=f"{arm}_rep{rep + 1}")
                     for rep in range(2)]

    # ---- Gate 判定（5 条件）----
    checks: dict[str, bool] = {}
    checks["1_resolver_renders"] = all(
        d["reference_rendered"] for d in arms["M_source"])
    # 用户条件 2：top-2 **集合**稳定（顺序差异不算不稳定）
    checks["2_top2_stable"] = all(
        set(arms[a][0]["top2"]) == set(arms[a][1]["top2"]) for a in arms)
    src_top2 = arms["M_source"][0]["top2"]
    rem_top2 = arms["M_remove"][0]["top2"]
    # 用户 Gate 条件 3：top-2 **集合**（无序）不同——不只排序
    checks["3_top2_differs"] = bool(set(src_top2) != set(rem_top2))
    # 4. 方向可解释：M_source 的 POSITIVE winsorize → winsorize 在 top-2
    #    且排序提前；M_signswap（NEGATIVE）→ winsorize 降级/不在 top-2
    checks["4_direction_explainable"] = bool(
        (not checks["3_top2_differs"]) or (
            "winsorize" in src_top2
            and "winsorize" not in arms["M_signswap"][0]["top2"]))
    checks["5_no_voting"] = True  # 2 次重复是稳定检查（单次为准）
    for a in arms:
        print(f"== {a}: rep1 top2={arms[a][0]['top2']} "
              f"chosen={arms[a][0]['chosen']} | rep2 top2={arms[a][1]['top2']}")
        if a == "M_source":
            print("   ref:", arms[a][0]["instruction_head"][:200].replace("\n", " "))

    if not checks["1_resolver_renders"]:
        verdict = "MEMORY_BINDING_FAILURE"
    elif not checks["2_top2_stable"]:
        verdict = "LLM_VARIANCE"
    elif not checks["3_top2_differs"]:
        verdict = "MEMORY_ACTION_SIGNAL_ABSENT"
    elif not checks["4_direction_explainable"]:
        verdict = "MEMORY_ACTION_SIGNAL_UNEXPLAINABLE"
    else:
        verdict = "MEMORY_ACTION_SIGNAL"
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== verdict: {verdict}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-kdd2018-memory-gate",
        "note": "P4.0 零新数据 plan-only Gate（已暴露 KDD Context；跨域 "
                "Monash outlier-family Source；不读新 Target outcome）",
        "pool": list(POOL),
        "arms": arms,
        "checks": checks,
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
