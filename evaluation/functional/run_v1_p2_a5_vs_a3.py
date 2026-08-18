"""P2_A5_VS_A3（Wave 4，2026-08-13：本班次最关键的价值实验——Source
Experience 是否有实际价值——等预算 Target 适应）。

Source：P2-v3 的 36 个 Context-bound Episode（impute_fft——12 train
series × 3 origins——POSITIVE 9 / NEGATIVE 8 / NEUTRAL 19 / CONFLICT
4 对；来源标注 deterministic census probe——不伪装 Fast 自然轨迹）。

Target（outcome-blind feasibility 已查）：NN5 同域 held-out origins
@712（+ delayed @728）——公开 Context 与 Source 重叠（missing_fraction
0.025 vs Source 0.023）；series 参与过 Source 评估 → 结论只称同域
机制检查（不称 fresh）。跨数据集 Target 无足够 Context overlap
（NN5 低缺失率域）。

两臂（同 roster / 同 Fast Agent / 同 Candidate DSL / 同 Target
Support 预算 / 同 LLM 上限 / 同 delayed block）：
  A3：空 Source Experience——Target-only 适应
  A5：读取 Source（resolve_experience_contrast_pack → render 进
    fast_features.experience_reference——真实 LLM Fast 入口自然调用）

指标：首次 material-positive 所需 probes / material-negative probes /
harm / abstention / delayed utility（@728 chosen op gain）/
A5 是否检索到 Source / 检索后是否改变候选选择。

判定（预注册）：
  SOURCE_MEMORY_BENEFIT    : A5 delayed 不劣于 A3 且任一改进
  MEMORY_TO_ACTION_ONLY    : 行为改变但无结果改善
  ADHERENCE_GAP            : 检索到但行为不变
  NEGATIVE_TRANSFER        : A5 更差
  NO_TARGET_CONTEXT_PREVALENCE : 无 Context-matched Target
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_p2_a5_vs_a3.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
from run_v1_kdd2018_natural_slow_update import _request  # noqa: E402
from run_w2_operator_scan import _default_params  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    build_episode,
    resolve_experience_contrast_pack,
    render_experience_pack,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import run_online_round  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgictoChatCompletionsBackend,
)

E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_p2_a5_vs_a3_report.json"
TASK_CONSUMER = "forecast|ridge|sMASE"
DOMAIN = "nn5"
SRC_ORIGINS = (600, 632, 680)
# prequential Target（用户裁决 2026-08-13）：X/Y 必须未被开发使用——
# 672/720 从未被 Source（600/632/680）/开发（712）/S3b（728）消费；
# Y=720 在决策前保持 sealed（720+48=768 ≤ 791 ✓）
TARGET_ORIGIN = 672
DELAYED_ORIGIN = 720
M = 0.005
OP = "impute_fft"
OPS = ("repair_level_shift", "impute_ar", "impute_ssm",
       "impute_fft", "impute_ema", "impute_linear")
BASE_CACHE: dict[int, float] = {}


def _gain_series(sid, op, origin, roster_full, values, cfg):
    compiled = v1.make_compiled(op, _default_params(op, 7))
    try:
        if origin not in BASE_CACHE:
            base = v6._evaluate(roster_full, values, None, cfg,
                                origin=origin)
            BASE_CACHE[origin] = float(base["mean_smase"])
        cand = v6._evaluate(roster_full, values, compiled, cfg,
                            origin=origin,
                            train_series_scope=frozenset({sid}))
        return BASE_CACHE[origin] - float(cand["mean_smase"])
    except Exception:
        return None


def _build_source_episodes(root, roster_full, values, cfg, series_ids):
    """P2-v3 36 个 Context-bound Episode（Wave 1 同款——deterministic
    census probe 来源标注）。"""
    eps = []
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import \
        extract_public_features as _epf
    for sid in series_ids[:12]:
        arr = values[sid]
        for origin in SRC_ORIGINS:
            g = _gain_series(sid, OP, origin, roster_full, values, cfg)
            pub = _epf(arr[:origin], task_kind="forecast")
            steps = [{"op": OP, "params": dict(_default_params(OP, 7))}]
            cls = ("POSITIVE" if g is not None and g >= M
                   else "NEGATIVE" if g is not None and g < -M
                   else "ABSTAIN")
            eps.append(build_episode(
                episode_id=f"nn5_p2src_{sid[:8]}_{OP}_{origin}",
                task_consumer_key=TASK_CONSUMER,
                domain_namespace=DOMAIN,
                context_summary={
                    "local_pattern": {
                        k: pub.get(k) for k in (
                            "missing_fraction",
                            "longest_missing_run_fraction",
                            "estimated_region_start_fraction",
                            "estimated_region_end_fraction",
                            "period_reliability",
                            "period_change_score")},
                    "delayed_pattern": {},
                    "program_geometry": {
                        "scope": "training_rows", "program_steps": steps},
                    "per_view_gain": [], "support_origin": origin},
                workflow_signature=workflow_signature_of(steps),
                support_response={"gain": g, "accepted": False},
                delayed_response={"evaluated": False, "gain": None},
                relation=cls, evidence_level="SUPPORT",
                local_status="EPISODE_ONLY",
                evidence_refs=["p2v3_deterministic_census"]))
    return eps


def main() -> int:
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no api key"}, indent=1))
        return 0
    import openai  # noqa: PLC0415

    root = PROJECT_ROOT
    cfg = dict(v6.DATASET_CONFIGS[DOMAIN])
    roster, values = v6._fixed_roster(root, cfg)
    series_ids = [r["series_uid"] for r in roster]
    roster_full = ([{"series_uid": s, "role": "train"}
                    for s in series_ids[:12]]
                   + [{"series_uid": s, "role": "eval"}
                      for s in series_ids[12:]])
    source_eps_all = _build_source_episodes(root, roster_full, values, cfg,
                                            series_ids)
    # prequential 过滤（用户裁决 2026-08-13）：只注入历史 Episode——
    # support_origin < target_origin（@680 晚于 Target @672——排除，
    # 否则非严格 prequential）
    source_eps = [e for e in source_eps_all
                  if int((e.context_summary or {}).get("support_origin")
                         or 0) < TARGET_ORIGIN]
    target_sid = series_ids[0]
    target_arr = values[target_sid]
    target_features = dict(extract_public_features(
        target_arr[:TARGET_ORIGIN], task_kind="forecast"))

    # ---- A5 检索（A3 空）----
    pack = resolve_experience_contrast_pack(
        source_eps, target_features, TASK_CONSUMER,
        allowed_operators=tuple(OPS))
    # 冲突合并检测（用户裁决）：同一 workflow 正负同命中 + Context 不可
    # 分 → 渲染层合并为 AMBIGUOUS（experience_memory.py 已修）——此处
    # 记录判定用 flag
    pos_ep = pack.positive if pack is not None else None
    neg_ep = pack.negative if pack is not None else None
    ambiguous_memory = bool(
        pos_ep is not None and neg_ep is not None
        and getattr(pos_ep, "workflow_signature", None)
        == getattr(neg_ep, "workflow_signature", None))
    reference_text = (render_experience_pack(pack.to_dict())
                      if pack is not None else None)
    report: dict[str, Any] = {
        "experiment_id": "v1-p2-a5-vs-a3",
        "note": "Wave 4：Source Experience 价值实验（P2-v3 Context-bound "
                "经验 → 等预算 Target 适应——同域机制检查 held-out "
                "origins——不称 fresh——development exposure——零新 Claim）",
        "apparatus": {"domain": DOMAIN, "target_series": target_sid[:8],
                      "target_origin": TARGET_ORIGIN,
                      "delayed_origin": DELAYED_ORIGIN,
                      "candidate_dsl": list(OPS),
                      "source": {"n_episodes": len(source_eps),
                                 "note": "deterministic census probe——"
                                         "非 Fast 自然轨迹"}},
        "a5_retrieval": {"pack_found": pack is not None,
                         "pack": (pack.to_dict() if pack is not None
                                  else None)},
    }

    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL,
                      timeout=120), max_calls=12)
    backend = AgictoChatCompletionsBackend(client=counter,
                                           base_url=smoke.BASE_URL)
    ex = ScopeExecutor(roster_full, values, cfg,
                       evaluate_fn=lambda r, v, c, config, *, origin:
                       v6._evaluate(r, v, c, config, origin=origin))

    def _run_arm(arm: str, memory_eps: tuple) -> dict[str, Any]:
        # 接线修复（用户核查 2026-08-13）：真正的 Memory 注入通道 =
        # TTHAMethod 第三参 experience_episodes（fast_agent.prepare 内部
        # resolve → render 进 instruction）——不是 fast_features。
        method = TTHAMethod(TTHAFastAgent(TTHAAgentCore(
            backend, LocalPublicToolGateway(target_arr[:TARGET_ORIGIN],
                                            task_kind="forecast"),
            model=smoke.MODEL, base_url=smoke.BASE_URL)),
            h0, tuple(memory_eps))
        feats = dict(extract_public_features(
            target_arr[:TARGET_ORIGIN], task_kind="forecast"))
        r = run_online_round(
            method, ex, _request(target_arr, values, TARGET_ORIGIN),
            values, origin=TARGET_ORIGIN, slow_agent=None,
            controller=None, store=None,
            card_builder=lambda e: {"pattern_id": "x",
                                    "observable_signature":
                                        {"task_kind": "forecast"}},
            round_name=f"p2w4_{arm}", budget=2, allow_slow=False,
            domain=f"nn5_p2w4_{arm}", period=7, fast_features=feats,
            allow_fast_skill=False, runtime_prior_slot=False,
            allow_group_slow=False)
        trace = getattr(method, "last_trace", None)
        mem_status = getattr(trace, "memory_resolution_status", None)
        if mem_status is None and isinstance(trace, dict):
            mem_status = trace.get("memory_resolution_status")
        probes = [(p["candidate_id"], p.get("gain"))
                  for p in r.actual_probed_programs]
        gains = [g for _c, g in probes if g is not None]
        return {"probes": probes,
                "memory_resolution_status": mem_status,
                "first_pos_probes": next(
                    (i + 1 for i, (_c, g) in enumerate(probes)
                     if g is not None and g >= M), None),
                "neg_probes": sum(1 for _c, g in probes
                                  if g is not None and g < -M),
                "harm": sum(1 for _c, g in probes
                            if g is not None and g < -M),
                "abstained": len(probes) == 0,
                "chosen_op": (probes[0][0] if probes else None),
                "max_gain": max(gains) if gains else None}

    a3 = _run_arm("A3", ())
    report["arm_A3"] = a3
    print("== A3: " + json.dumps(a3, ensure_ascii=False), flush=True)
    a5 = _run_arm("A5", tuple(source_eps))
    report["arm_A5"] = a5
    print("== A5: " + json.dumps(a5, ensure_ascii=False), flush=True)

    # delayed utility（两臂各自 chosen op @720——candidate ID 映射回算子：
    # 去掉 cand_/_local 后缀——判定修复 2026-08-13：None 不得当作 0.0）
    for arm, res in (("A3", a3), ("A5", a5)):
        op = res["chosen_op"]
        if op is not None and not str(op).startswith("cand_skill"):
            op = str(op).replace("cand_", "").replace("_local", "")
            res["chosen_op_clean"] = op
            res["delayed_gain"] = _gain_series(
                target_sid, op, DELAYED_ORIGIN, roster_full, values, cfg)
        else:
            res["chosen_op_clean"] = None
            res["delayed_gain"] = None

    # ---- 判定（预注册——修复：delayed 未测得（None）不得当 0.0）----
    behavior_changed = (a3["probes"] != a5["probes"])
    u3 = a3["delayed_gain"]
    u5 = a5["delayed_gain"]
    improved = bool(
        (a5["first_pos_probes"] is not None
         and (a3["first_pos_probes"] is None
              or a5["first_pos_probes"] < a3["first_pos_probes"]))
        or a5["harm"] < a3["harm"]
        or (a5["max_gain"] is not None and a3["max_gain"] is not None
            and a5["max_gain"] > a3["max_gain"]))
    # 修复：delayed 未测得（任一 None）→ not_worse 如实标 None（未测得）
    if u3 is None or u5 is None:
        not_worse = None
    else:
        not_worse = bool(u5 >= u3 - 1e-9)
    if pack is None:
        verdict = "NO_TARGET_CONTEXT_PREVALENCE"
    elif ambiguous_memory and a5["abstained"]:
        # 用户裁决：冲突 Memory（正负同 workflow 且 Context 不可分）下
        # abstain = 合理风险响应——不是 Adherence gap
        verdict = "CONFLICT_MEMORY_SAFE_ABSTAIN"
    elif not behavior_changed:
        verdict = "ADHERENCE_GAP"
    elif improved and not_worse is True:
        verdict = "SOURCE_MEMORY_BENEFIT"
    elif improved and not_worse is None:
        verdict = "MEMORY_TO_ACTION_ONLY"  # 改善但 delayed 未测得
    elif not improved:
        verdict = ("NEGATIVE_TRANSFER" if (u5 is not None and u3 is not None
                                           and u5 < u3)
                   else "MEMORY_TO_ACTION_ONLY")
    else:
        verdict = "MEMORY_TO_ACTION_ONLY"
    report["utility"] = {"A3_delayed": u3, "A5_delayed": u5,
                         "A5_not_worse": not_worse, "improved": improved}
    report["a5_retrieval"]["ambiguous_memory"] = ambiguous_memory
    report["a5_retrieval"]["prequential_filter"] = {
        "injected": len(source_eps), "excluded_later_than_target":
            len(source_eps_all) - len(source_eps)}
    report["verdict"] = verdict
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
