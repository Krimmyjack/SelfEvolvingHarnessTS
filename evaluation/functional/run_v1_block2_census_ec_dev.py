"""BLOCK2_CENSUS_EC_DEV（P1，2026-08-13：第二个自然 Development Block
census + 确定性 Evidence Compiler 分支——development exposure——零新
Claim）。

背景：P0 两次 FAIL（LLM_BATCH_EVIDENCE_INTEGRATION_NOT_ESTABLISHED——
语义先验主导弃权）→ 降级设计采纳（EVIDENCE_COMPILER_DEGRADED_CHAIN_PASS）
→ P1 解锁（用户任务书："仅当 P0 PASS 或采用了确定性 Evidence Compiler
后运行"）。

预注册装置（运行前确定——不按 outcome 挑选）：
  - Series（cache 顺序下 Wave 3 未用的前 4 个——SHIFT_REPORT 预注册）：
    T102, T103, T104, T105
  - Origins：600, 792, 888, 984（与 sealed 装置同构）
  - Eval 集：已冻结 p41 评估集（T128, T129, T13, T130, T131, T132,
    T133, T134）
  - 装置：H0 + SealedProbeBackend（explore + force_pool——winsorize
    先探——Wave 3 同构），budget 2，allow_slow=False（只收集 factual
    Action–Response，不触发单条 Slow）
  - 并行：4 series 并发（parallel_eval 线程——单 series 内 4 origin
    顺序——单轮内探测顺序是自适应语义，不可并行）
  - 每轮合法 Action-Response 写 Episode（完整 workflow 指纹 + per_view
    + origin）→ 跨 series 分组（group_first_faults, min_group=2）

Evidence Compiler 分支（只对 top family——确定性排序首项）：
  candidates = OPS − family workflow 算子（replacement 语义）
  headroom（逐 Episode 窗口 replay 替代候选——Wave 3 同构）
  unique_common_positive(headroom, candidates)：
    - 零 → BLOCK2_FAMILY_NO_HEADROOM：与 Wave 3 family（无 headroom）
      连续两个 → 按用户 P1 停止条件关闭 winsorize/outlier ×
      forecast|ridge|sMASE × 当前机制 → 晨间报告
    - ≥2 → BLOCK2_FAMILY_AMBIGUOUS_HEADROOM（确定性 abstain——
      EC 语义：唯一才决策）
    - 唯一 → 完整链（evidence_compiler=True + runtime_selected_patch_id）：
      * 真实 LLM 只编译（1 调用预算 + 1 校验重试；CountingClient
        硬停 2——不再做选择）
      * 组内 replay 门（全部 ≥M）
      * holdout @600（family 首个 series——cache 序）≥ −M
      * pending → 补集检查（失败 op 的正向窗口 ≥ −M）→
        delayed @1032（984 + HORIZON——T117 witness 同款约定）
      * approved → Skill adoption（snapshot SHA 留痕）→ H1 residual
        触发（Wave 5，task #117）

verdict（预注册）：
  BLOCK2_FULL_CHAIN_APPROVED        : 链到 approved（→ H1 residual）
  BLOCK2_CHAIN_COMPLEMENT_REJECTED  : pending 但补集拒绝
  BLOCK2_CHAIN_DELAYED_REJECTED     : pending 但 delayed 拒绝
  BLOCK2_CHAIN_REPLAY_REJECTED      : 组内 replay 门拒绝（Runtime 选择
                                      与门矛盾——如实记录）
  BLOCK2_CHAIN_HOLDOUT_REJECTED     : holdout 拒绝
  BLOCK2_COMPILE_FAILURE            : LLM 编译失败（单选项白名单）
  BLOCK2_FAMILY_NO_HEADROOM         : 零共同正向替代（→ 停止条件检查）
  BLOCK2_FAMILY_AMBIGUOUS_HEADROOM  : ≥2 共同正向（确定性 abstain）
  BLOCK2_NO_INDEPENDENT_FAILURE_FAMILY
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_block2_census_ec_dev.py
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
from SelfEvolvingHarnessTS.methods.ttha.program_supply import controlled_add_only_group_catalog, controlled_add_only_group_decision

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import signed_radius as resolver  # noqa: E402
from parallel_eval import run_parallel  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _request,
)

from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    deployment_constraints_v1,
    forecast_task_context_v1,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    build_episode,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent,
    public_operator_contract,
)
from SelfEvolvingHarnessTS.methods.ttha.group_fault import (  # noqa: E402
    build_contrast_capsule,
    group_first_faults,
    unique_common_positive,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgictoChatCompletionsBackend,
)

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD
BUDGET = 2
# 预注册 block 2（运行前确定——cache 顺序下 Wave 3 未用的前 4 个）
DEV_SERIES = ("T102", "T103", "T104", "T105")
DEV_ORIGINS = (600, 792, 888, 984)
EVAL_SERIES = ("T128", "T129", "T13", "T130",
               "T131", "T132", "T133", "T134")
OPS = ("winsorize", "outlier_mad", "hampel_filter")
# delayed 门约定（T117 witness 同款——最后 Support origin + HORIZON）
DELAYED_ORIGIN = 1032
HOLDOUT_ORIGIN = 600
CENSUS_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_batch_census_dev_report.json"
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_block2_census_ec_dev_report.json"


def _load_series(root: Path, uid: str) -> np.ndarray:
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    return np.asarray(values[names.index(uid)], dtype=np.float64)


def _series_executor(root: Path, series: str) -> ScopeExecutor:
    roster = ([{"series_uid": series, "role": "train"}]
              + [{"series_uid": s, "role": "eval"} for s in EVAL_SERIES])
    values = {s: _load_series(root, s) for s in (series,) + EVAL_SERIES}
    return ScopeExecutor(roster, values, _config(),
                         evaluate_fn=_evaluate_kdd)


def _census_series_task(root: Path, sid: str, h0: Any) -> dict[str, Any]:
    """单 series census 任务（并行格）——4 origin 顺序探测 + Episode
    收集。返回 episodes 附 series 标签（不做 episode_id 字符串解析）。"""
    series_arr = _load_series(root, sid)
    if len(series_arr) < DEV_ORIGINS[-1] + HORIZON:
        return {"series": sid,
                "skipped": {"reason": "too_short",
                            "length": int(len(series_arr))},
                "rounds": [], "episodes": []}
    ex = _series_executor(root, sid)
    vals = {s: _load_series(root, s) for s in (sid,) + EVAL_SERIES}
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=OPS,
                                  max_propose_candidates=3,
                                  force_pool=True),
        LocalPublicToolGateway(series_arr[:DEV_ORIGINS[0]],
                               task_kind="forecast"))
    method = TTHAMethod(sealed.TTHAFastAgent(core), h0, ())
    rounds_log: list[dict[str, Any]] = []
    for origin in DEV_ORIGINS:
        core.backend = sealed.SealedProbeBackend(
            explore=True, operators=OPS, max_propose_candidates=3,
            force_pool=True)
        r = run_online_round(
            method, ex,
            _request(series_arr, vals, origin),
            vals,
            origin=origin, slow_agent=None, controller=None, store=None,
            card_builder=lambda e: {"pattern_id": "x",
                                    "observable_signature":
                                        {"task_kind": "forecast"}},
            round_name=f"block2_{sid}_{origin}", budget=BUDGET,
            allow_slow=False, domain=f"kdd2018_b2_{sid}",
            period=PERIOD,
            fast_features=dict(extract_public_features(
                series_arr[:origin], task_kind="forecast")),
            allow_fast_skill=False, runtime_prior_slot=False,
            allow_group_slow=False)
        rounds_log.append({
            "origin": origin,
            "probes": [(p["candidate_id"], p.get("gain"))
                       for p in r.actual_probed_programs],
            "episodes_written": list(r.episode_ids)})
        print(f"== b2 {sid}@{origin}: probes="
              f"{[(p['candidate_id'], p.get('gain'))
                  for p in r.actual_probed_programs]}", flush=True)
    eps = list(method._experience_episodes)
    return {"series": sid, "rounds": rounds_log,
            "episodes": [{"series": sid, "ep": e} for e in eps]}


def _family_headroom(root: Path, family: Mapping[str, Any],
                     executors: Mapping[str, ScopeExecutor],
                     candidates: Sequence[str]) -> dict[str, Any]:
    """top family 的共同 replacement headroom（per-episode 解析 executor
    ——origin 跨 series 会碰撞——手动循环）。"""
    out: dict[str, Any] = {}
    eps = family["episodes"]
    for alt in candidates:
        per_ep = []
        all_pos = True
        for e in eps:
            sid, origin = e["series"], e["origin"]
            steps = ((alt, dict(wiring.contract_params(alt, PERIOD))),)
            rr = executors[sid].evaluate(tuple(steps), origin)
            g = (float(rr.gain) if rr.gain is not None else None)
            per_ep.append({"series": sid, "origin": origin, "gain": g})
            if g is None or g < M:
                all_pos = False
        out[alt] = {"per_episode_gains": per_ep,
                    "common_positive": all_pos}
    return out


def _rebuild_failure_episode(root: Path, series: str, origin: int,
                             executor: ScopeExecutor,
                             failed_steps: Sequence[tuple[str, dict]],
                             expect_gain: float) -> Any:
    """family 失败窗口重建（replay 失败 op——读数与 census 逐位一致——
    零新 outcome）。"""
    steps = tuple(failed_steps)
    rr = executor.evaluate(steps, origin)
    gain = float(rr.gain) if rr.gain is not None else None
    assert gain is not None and gain < -M, f"{series}@{origin} not failure"
    assert abs(gain - expect_gain) < 1e-9, \
        f"{series}@{origin} drift: {gain} vs census {expect_gain}"
    sig = workflow_signature_of(
        [{"op": s[0], "params": dict(s[1])} for s in steps])
    ctx = dict(resolver.window_context({series: _load_series(root, series)},
                                       origin, PERIOD))
    return build_episode(
        episode_id=f"kdd2018_b2_{series}_target_{sig}_ec_{series}_{origin}",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace=f"kdd2018_b2_{series}",
        context_summary={
            "local_pattern": {"support_gain": gain, **ctx},
            "delayed_pattern": {},
            "program_geometry": {
                "scope": "training_rows",
                "program_steps": [{"op": s[0], "params": dict(s[1])}
                                  for s in steps]},
            "per_view_gain": list(getattr(rr, "per_view_gain", []) or []),
            "support_origin": origin,
        },
        workflow_signature=sig,
        support_response={"gain": gain, "accepted": False},
        delayed_response={"evaluated": False, "gain": None},
        relation="NEGATIVE", evidence_level="SUPPORT",
        local_status="EPISODE_ONLY", evidence_refs=["block2_ec"])


def _positive_contrast_episode(series: str, origin: int,
                               gain: float, op: str) -> Any:
    return build_episode(
        episode_id=f"kdd2018_b2_{series}_target_{op}_pos_{series}"
                   f"_{origin}",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace=f"kdd2018_b2_{series}",
        context_summary={
            "local_pattern": {"support_gain": gain},
            "delayed_pattern": {},
            "program_geometry": {"scope": "training_rows",
                                 "program_steps": [
                                     {"op": op, "params": {}}]},
            "per_view_gain": [],
            "support_origin": origin,
        },
        workflow_signature=op,
        support_response={"gain": gain, "accepted": True},
        delayed_response={"evaluated": False, "gain": None},
        relation="POSITIVE", evidence_level="SUPPORT",
        local_status="LOCAL_DRAFT", evidence_refs=["block2_ec"])


def _ec_card(group: Mapping[str, Any], capsule: Mapping[str, Any],
             headroom: Mapping[str, Any], workflow: str,
             candidates: Sequence[str]) -> dict[str, object]:
    """EC 模式卡：白名单 = 全部 replacement 候选（method 按
    runtime_selected_patch_id 收敛到唯一项）；instruction = 只编译。"""
    options = []
    for alt in candidates:
        options.append({
            "patch_id": f"patch-replace-{workflow}-with-{alt}",
            "program_steps": [{"op": alt,
                               "params": dict(wiring.contract_params(
                                   alt, PERIOD))}]})
    return {
        "pattern_id": f"group-{workflow}-neg-multiseries",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "context": {},
        "workflow": {"steps": [{"op": w, "params": {}}
                               for w in workflow.split("|")]},
        "typed_patch_options": options,
        "facts": {
            "contrast_capsule": dict(capsule),
            "replacement_headroom": dict(headroom),
        },
        "instruction": (
            "A repeated first-fault group across multiple independent "
            "target series has been diagnosed. The runtime has "
            "DETERMINISTICALLY selected exactly one typed patch from "
            "typed_patch_options based on batch evidence (common positive "
            "replacement headroom). Your role is COMPILATION ONLY — do "
            "not re-evaluate the choice and do not declare no_proposal. "
            "Produce the edit_manifest with edit_manifest.patch_id set to "
            "the available patch_id from typed_patch_binding_rule; the "
            "runtime binds the frozen program steps from the whitelist "
            "entry. The deterministic group replay and complement gates "
            "verify the patch after compilation.",
        ),
    }


def _wave3_top_family_no_headroom(root: Path) -> bool:
    """Wave 3 top family 是否无共同正向替代（读 census 报告——确定性
    复核——Wave 4 结论 NO_COMMON_PROGRAM_HEADROOM 的同源检查）。"""
    census = json.loads(CENSUS_REL.read_text(encoding="utf-8"))
    fams = census.get("development_families") or []
    if not fams:
        return False
    hr = fams[0].get("replacement_headroom") or {}
    return not any((hr.get(a) or {}).get("common_positive") for a in OPS)


def main() -> int:
    root = PROJECT_ROOT
    report: dict[str, Any] = {
        "experiment_id": "v1-block2-census-ec-dev",
        "note": "P1：第二个自然 Development Block census + 确定性 "
                "Evidence Compiler 分支（development exposure——零新 "
                "Claim）。装置同 Wave 3（T102-T105 × 4 origin × p41 "
                "eval 集）——唯一机制差异 = evidence_compiler 模式 + "
                "series 级并行。",
        "pre_registered": {"dev_series": list(DEV_SERIES),
                           "origins": list(DEV_ORIGINS),
                           "eval_series": list(EVAL_SERIES),
                           "ops": list(OPS),
                           "holdout": {"series": "family-first-cache-order",
                                       "origin": HOLDOUT_ORIGIN},
                           "delayed": {"series": "family-first-cache-order",
                                       "origin": DELAYED_ORIGIN}},
    }

    # ---- Phase 1：并行 census（4 series 并发——零 LLM）----
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    tasks = [lambda sid=sid: _census_series_task(root, sid, h0)
             for sid in DEV_SERIES]
    results = run_parallel(tasks, workers=4)
    executors: dict[str, ScopeExecutor] = {}
    all_episodes: list[Any] = []
    ep_to_series: dict[int, str] = {}
    rounds_log: dict[str, list[dict[str, Any]]] = {}
    skipped: list[dict[str, Any]] = []
    task_errors: list[dict[str, Any]] = []
    for ok, res in results:
        if not ok:
            task_errors.append({"error": f"{type(res).__name__}: {res}"})
            continue
        sid = res["series"]
        executors[sid] = _series_executor(root, sid)
        if res.get("skipped"):
            skipped.append(res["skipped"])
            continue
        rounds_log[sid] = res["rounds"]
        for item in res["episodes"]:
            all_episodes.append(item["ep"])
            ep_to_series[id(item["ep"])] = item["series"]
    report["development_rounds"] = rounds_log
    report["development_skipped"] = skipped
    report["task_errors"] = task_errors
    if task_errors or not all_episodes:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "census tasks failed"},
                         ensure_ascii=False, indent=1))
        return 0

    # 跨 series 分组（完整 workflow 指纹 × sign）
    groups = group_first_faults(all_episodes, min_group=2)
    families: list[dict[str, Any]] = []
    for g in groups:
        eps = g["episodes"]
        series = sorted({ep_to_series[id(e)] for e in eps})
        gains = [(float((e.support_response or {}).get("gain") or 0.0), e)
                 for e in eps]
        fam = {
            "workflow": g["workflow"],
            "sign": g["sign"],
            "n_episodes": len(eps),
            "independent_series": series,
            "n_series": len(series),
            "origins": sorted({int((e.context_summary or {})
                                  .get("support_origin") or 0) for e in eps}),
            "episodes": [{"series": ep_to_series[id(e)],
                          "origin": int((e.context_summary or {})
                                        .get("support_origin") or 0),
                          "gain": float((e.support_response or {})
                                        .get("gain") or 0.0)}
                         for _, e in sorted(gains)],
            "min_gain": min(g[0] for g in gains),
            "max_gain": max(g[0] for g in gains),
        }
        families.append(fam)
    # 排序：独立 series 数 ↓ × |min harm| ↓ × origin 数 ↓（Wave 3 同构）
    families.sort(key=lambda f: (-f["n_series"], f["min_gain"],
                                 -len(f["origins"])))
    report["development_families"] = families
    print("== b2 families: " + json.dumps(families, ensure_ascii=False,
                                          default=str), flush=True)

    if not families:
        w3_no_headroom = _wave3_top_family_no_headroom(root)
        report["stop_condition"] = {
            "wave3_top_family_no_headroom": w3_no_headroom,
            "block2_top_family_no_headroom": False,
            "note": "block2 无跨 series family——停止条件（连续两个 family "
                    "无 headroom）不适用",
            "two_consecutive_families_no_headroom": False,
        }
        report["verdict"] = "BLOCK2_NO_INDEPENDENT_FAILURE_FAMILY"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: BLOCK2_NO_INDEPENDENT_FAILURE_FAMILY / "
              f"report -> {REPORT_REL}")
        return 0

    top = families[0]
    failed_ops = [w for w in str(top["workflow"]).split("|")
                  if w and w != "unknown"]
    candidates = [op for op in OPS if op not in failed_ops]
    top["headroom_candidates"] = candidates
    top["replacement_headroom"] = _family_headroom(
        root, top, executors, candidates)
    hr = top["replacement_headroom"]
    chosen = unique_common_positive(hr, candidates)
    n_pos = sum(1 for a in candidates
                if (hr.get(a) or {}).get("common_positive"))
    report["ec"] = {"candidates": candidates, "runtime_choice": chosen,
                    "n_common_positive": n_pos}
    print(f"== ec: candidates={candidates} choice={chosen} "
          f"n_pos={n_pos}", flush=True)

    # ---- 分支判定 ----
    if n_pos == 0:
        report["verdict"] = "BLOCK2_FAMILY_NO_HEADROOM"
    elif chosen is None:
        report["verdict"] = "BLOCK2_FAMILY_AMBIGUOUS_HEADROOM"
    else:
        # ---- Phase 2：EC 完整链（真实 LLM 只编译）----
        report["verdict"] = _run_ec_chain(root, report, top, executors,
                                          ep_to_series, rounds_log,
                                          failed_ops, candidates, chosen)
    # 停止条件检查（用户 P1：连续两个 family 无 headroom = 停止条件）
    w3_no_headroom = _wave3_top_family_no_headroom(root)
    b2_no_headroom = (n_pos == 0)
    stop_condition = bool(w3_no_headroom and b2_no_headroom)
    report["stop_condition"] = {
        "wave3_top_family_no_headroom": w3_no_headroom,
        "block2_top_family_no_headroom": b2_no_headroom,
        "two_consecutive_families_no_headroom": stop_condition,
    }
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print("== verdict:", report["verdict"], "stop_condition:",
          stop_condition, flush=True)
    print(f"== report -> {REPORT_REL}")
    return 0


def _run_ec_chain(root: Path, report: dict[str, Any],
                  family: Mapping[str, Any],
                  executors: Mapping[str, ScopeExecutor],
                  ep_to_series: dict[int, str],
                  rounds_log: Mapping[str, list[dict[str, Any]]],
                  failed_ops: Sequence[str], candidates: Sequence[str],
                  chosen: str) -> str:
    """EC 完整链：真实 LLM 只编译 → 组内 replay → holdout → pending →
    补集 → delayed → approved（Skill adoption）。"""
    workflow = str(family["workflow"])
    failed_steps = tuple((w, dict(wiring.contract_params(w, PERIOD)))
                         for w in failed_ops)
    # family 失败窗口重建（读数与 census 逐位一致——零新 outcome）
    by_key = {(e["series"], e["origin"]): e["gain"]
              for e in family["episodes"]}
    eps = []
    for e in family["episodes"]:
        eps.append(_rebuild_failure_episode(
            root, e["series"], e["origin"], executors[e["series"]],
            failed_steps, expect_gain=by_key[(e["series"], e["origin"])]))
        ep_to_series[id(eps[-1])] = e["series"]
    groups = group_first_faults(eps, min_group=2)
    if not groups:
        return "PROTOCOL_FAILURE"
    group = groups[0]
    # 对比案例（失败 op 的正向窗口——block2 census rounds 读数；多算子
    # family 无单算子对照 → 补集 vacuous）
    positive_windows: list[dict[str, Any]] = []
    if len(failed_ops) == 1:
        for sid, rounds in rounds_log.items():
            for r in rounds:
                for cid, gain in r.get("probes") or []:
                    if cid == f"cand_{failed_ops[0]}" and gain is not None \
                            and gain >= M:
                        positive_windows.append({"series": sid,
                                                 "origin": r["origin"],
                                                 "gain": gain})
    contrast_eps = [_positive_contrast_episode(w["series"], w["origin"],
                                               w["gain"], failed_ops[0])
                    for w in positive_windows]
    view_keys = {ep.episode_id: list(EVAL_SERIES) for ep in eps}
    capsule = build_contrast_capsule(group, all_episodes=eps + contrast_eps,
                                     view_keys=view_keys)
    report["ec_chain"] = {
        "family": family,
        "group": {"workflow": group["workflow"], "sign": group["sign"],
                  "episodes": [e.episode_id for e in group["episodes"]]},
        "capsule": capsule,
        "positive_windows": positive_windows,
    }

    # ---- 真实 Slow Agent（只编译——1 调用预算 + 1 校验重试）----
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        return "PROTOCOL_FAILURE"
    import openai  # noqa: PLC0415
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120))
    backend = AgictoChatCompletionsBackend(client=counter,
                                           base_url=smoke.BASE_URL)
    family_first = family["independent_series"][0]
    series0 = _load_series(root, family_first)

    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    store = SnapshotStore(root / ".b2_store")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    slow = TTHASlowAgent(TTHAAgentCore(
        backend, LocalPublicToolGateway(series0[:HOLDOUT_ORIGIN],
                                        task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL))
    method = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(
            backend, LocalPublicToolGateway(series0[:HOLDOUT_ORIGIN],
                                            task_kind="forecast"),
            model=smoke.MODEL, base_url=smoke.BASE_URL)),
        h0, tuple(eps))

    request = _request(series0, {s: _load_series(root, s)
                                 for s in (family_first,) + EVAL_SERIES},
                       HOLDOUT_ORIGIN)
    task_ctx = forecast_task_context_v1(
        task_spec=request.task_spec,
        deployment_constraints=deployment_constraints_v1())
    contracts = tuple(public_operator_contract(op) for op in OPS)

    def _eval_group(steps, ep):
        sid = ep_to_series[id(ep)]
        origin = int(((getattr(ep, "context_summary", {}) or {})
                      .get("support_origin") or 0))
        return executors[sid].evaluate(tuple(steps), origin)

    selected_patch_id = f"patch-replace-{workflow}-with-{chosen}"
    ev = method.handle_group_feedback(group, capsule, slow_agent=slow, controller=controller, store=store, card_builder=lambda g, cap: _ec_card(g, cap, family['replacement_headroom'], workflow, candidates), evaluator_group=_eval_group, holdout_evaluator=lambda s, _m: executors[family_first].evaluate(tuple(s), HOLDOUT_ORIGIN), fast_features=dict(extract_public_features(series0[:HOLDOUT_ORIGIN], task_kind='forecast')), allowed_operator_contracts=contracts, task_context=task_ctx, evidence_compiler=True, runtime_selected_patch_id=selected_patch_id, surface_catalog=controlled_add_only_group_catalog(), route_decision=controlled_add_only_group_decision(case_id='dev-run_v1_block2_census_ec_dev-587'))
    report["ec_chain"]["group_feedback_event"] = ev
    report["ec_chain"]["llm"] = {"model": smoke.MODEL,
                                 "base_url": smoke.BASE_URL,
                                 "calls": counter.calls,
                                 "schema": "slow_edit_v1"}
    print("== ec chain ev: " + json.dumps(ev, ensure_ascii=False,
                                          default=str), flush=True)

    # ---- 补集检查 + delayed（仅 pending 可达）----
    complement: dict[str, Any] = {"checked": False, "windows": []}
    dev = None
    if ev.get("stage") == "pending":
        steps = tuple((f["op"], dict(f.get("params") or {}))
                      for f in ev["frozen_program"])
        complement["checked"] = True
        ok_comp = True
        for w in positive_windows:
            rr = executors[w["series"]].evaluate(tuple(steps), w["origin"])
            g = (float(rr.gain) if rr.gain is not None else None)
            complement["windows"].append(
                {"series": w["series"], "origin": w["origin"],
                 "baseline_gain": w["gain"], "patch_gain": g})
            if g is None or g < -M:
                ok_comp = False
        complement["passed"] = ok_comp
        if not positive_windows:
            complement["vacuous"] = True
        if ok_comp:
            dev = method.handle_feedback_delayed(
                lambda s, _m: executors[family_first].evaluate(
                    tuple(s), DELAYED_ORIGIN),
                episode_id=ev.get("episode_id"))
        else:
            complement["stage"] = "complement_rejected"
    report["ec_chain"]["complement_check"] = complement
    report["ec_chain"]["delayed"] = dev

    # ---- Skill adoption（approved 留痕）----
    if dev is not None and dev.get("stage") == "approved":
        snap = getattr(method, "_snapshot", None)
        report["ec_chain"]["skill_adoption"] = {
            "skill_id": f"group_{workflow}_replacement",
            "patch_id": ev.get("patch_id"),
            "snapshot_sha": (getattr(snap, "harness_content_sha", None)
                             if snap is not None else None),
            "snapshot_updated": bool(dev.get("snapshot_updated")),
        }

    # ---- 判定（预注册——与 docstring 严格一致）----
    if counter.calls > 2:
        return "PROTOCOL_FAILURE"
    stage = ev.get("stage")
    if dev is not None and dev.get("stage") == "approved":
        return "BLOCK2_FULL_CHAIN_APPROVED"
    if stage == "pending":
        if not complement.get("passed"):
            return "BLOCK2_CHAIN_COMPLEMENT_REJECTED"
        if dev is not None and dev.get("stage") == "delayed_rejected":
            return "BLOCK2_CHAIN_DELAYED_REJECTED"
        return "PROTOCOL_FAILURE"  # pending 后未接 delayed——装置错误
    if stage == "group_replay_rejected":
        return "BLOCK2_CHAIN_REPLAY_REJECTED"
    if stage == "holdout_rejected":
        return "BLOCK2_CHAIN_HOLDOUT_REJECTED"
    if stage in ("typed_patch_contract_failed", "budget_exceeded",
                 "no_manifest", "no_frozen_program",
                 "manifest_preflight_failed"):
        return "BLOCK2_COMPILE_FAILURE"
    return "PROTOCOL_FAILURE"


if __name__ == "__main__":
    raise SystemExit(main())
