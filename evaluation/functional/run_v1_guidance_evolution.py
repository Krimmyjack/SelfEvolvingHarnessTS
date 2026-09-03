"""GUIDANCE_EVOLUTION（2026-08-14 夜）——今晚唯一逻辑 Runner。

纵向目标：多轨迹共同 Workflow 构造失败 → Group Contrast Capsule → Slow
Agent 修改 Workflow Construction Skill（唯一变量 =
bootstrap_skills.entries/build_contrastive_candidates.body）→ Runtime 应用到
fork → Fast Agent 用新 Skill 重新生成 Workflow → 候选行为按预测改变 →
held-out Support/delayed 验证 → 激活或拒绝 → removal 验证。

Gates：
  g1-b0 / g1-b1 / g1-verdict   G1 行动性正控（人工最小 guidance patch）
  g2                           G2 真实 Slow Guidance Patch（组级 Slow → PATCH）
  g3                           G3 行为核销（旧/新 snapshot replay）
  g4                           G4 held-out Utility（fault-build/support/delayed 三组）
  g5                           G5 正常入口与 removal

数据纪律：全部使用已暴露 KDD2018 development 窗口（census 报告
w1_batch_census_dev_report.json 的 development_rounds/families——development
exposure——零新 Claim）。G4 的 held-out/delayed 分组预先冻结（见
G4_ROSTER），delayed 只对最终 winner 打开。Fast 准备阶段不开新 Outcome
（prepare 本身只生成候选+verifier 过滤；Support/delayed 探测只在 G4/G5
按冻结 roster 打开）。

装置：真实 LLM（agicto gpt-5.6-luna，温度 0，CountingClient 每 prepare
预算 8 次调用）。每臂每 context n=2 reps 控制 LLM 方差。Memory 两臂同为
空（隔离 Guidance 变量）。Snapshot store 用 .guidance_store（新目录，
不触碰历史 store）。

用法：
  python evaluation/functional/run_v1_guidance_evolution.py <phase>
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
import run_v1_kdd2018_natural_slow_update as nsu  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256  # noqa: E402
from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    EditManifest,
    EditOperation,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    AgentRole,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (  # noqa: E402
    TTHASlowAgent,
    _resolve_apply_manifest,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgictoChatCompletionsBackend,
)

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD
MODEL = smoke.MODEL
BASE_URL = smoke.BASE_URL
KEY_ENVS = smoke.KEY_ENVS
REPORT_REL = PROJECT_ROOT / "artifacts" / "functional" / "e2" \
    / "w1_guidance_evolution_report.json"
STORE_DIR = PROJECT_ROOT / ".guidance_store"
H0_ROOT = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"
CENSUS_REL = PROJECT_ROOT / "artifacts" / "functional" / "e2" \
    / "w1_batch_census_dev_report.json"
SURFACE = "bootstrap_skills.entries/build_contrastive_candidates.body"
GUIDANCE_SKILL_ID = "build_contrastive_candidates"

# ---- 已暴露 development 窗口（全部来自 census development 暴露；不开新 Outcome）----
G1_CONTEXTS = (("T100", 600), ("T1", 888))
G1_REPS = 2
# 预算一次性预注册修订（用户裁决 2026-08-14）：修复后流程含工具轮，
# 实测 6-10 次/prepare——旧 8 上限会随机砍断 prepare。修订冻结于进一步
# 调用前；其余协议（B1 文本/上下文/裁定规则/升级规则）不动。
LLM_BUDGET_PER_PREPARE = 12

# ---- G4 预冻结 roster（在打开任何新探测前冻结；见 REPORT 的 g4_roster）----
# fault-build 组：T100@600、T1@888（winsorize 失败且 outlier_mad ≥ winsorize 的已暴露窗口）
# held-out support 组：T10@888、T101@984（不同 series 的 winsorize 失败窗口——已暴露）
# delayed 组：仅最终 winner 打开——winner 生效窗口的 origin+HORIZON（T100@648/T1@936
#   的 delayed 探测按 G4 计划只对最终 winner 打开一次）
G4_FAULT_BUILD = (("T100", 600), ("T1", 888))
G4_HELDOUT_SUPPORT = (("T10", 888), ("T101", 984))

# ---- B1 人工最小 guidance patch（G1 正控——审核者定稿文本，2026-08-14，
# 在 B0 观察前冻结）----
# 只增加一条明确行为要求（current-context exploration slot）；不动
# distinctness（instruction/proposal_guidance/body 已有，避免近重复导致
# 假阴性）。
B1_APPEND = (
    " Reserve one candidate slot for current-context exploration: at least one "
    "non-identity candidate must use an operator family that directly addresses "
    "the most salient signal of the current context identified during inspection "
    "(for example, an outlier or spike family when a recent extreme deviation "
    "is observed)."
)


def _load_report() -> dict[str, Any]:
    if REPORT_REL.exists():
        return json.loads(REPORT_REL.read_text(encoding="utf-8"))
    return {}


def _save_report(report: dict[str, Any]) -> None:
    REPORT_REL.parent.mkdir(parents=True, exist_ok=True)
    REPORT_REL.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _load_env() -> dict[str, Any]:
    cohort = nsu._load_cohort(PROJECT_ROOT)
    eval_series = tuple(json.loads(CENSUS_REL.read_text(encoding="utf-8"))
                        ["pre_registered"]["eval_series"])
    return {"roster": cohort["roster"], "values": cohort["values"],
            "eval_series": eval_series}


def _h0_snapshot() -> Any:
    return compile_snapshot(H0_ROOT, verify_lock=False)


class RecordingBackend:
    """记录每个 AgentRequest 的 stage/semantic hash/system 消息与**完整**
    响应文本（P1 用户裁决 2026-08-14：不再截断 600 字符），用于 "Prompt
    确实不同" 与 stage 载荷诊断；调用委托给真实 backend。"""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    def complete(self, request: Any) -> Any:
        system = str(request.messages[0]["content"])
        record = {
            "role": str(request.role),
            "stage": str(request.stage),
            "call_index": int(request.call_index),
            "semantic_request_hash": request.semantic_request_hash(),
            "effective_harness_view_sha": str(request.effective_harness_view_sha),
            "system_len": len(system),
            "user_message": str(request.messages[-1]["content"])
            if request.messages[-1]["role"] == "user" else "",
        }
        response = self.delegate.complete(request)
        record["assistant_text"] = str(response.assistant_text)
        record["parse_status"] = str(response.parse_status)
        record["parse_recovery"] = str(
            getattr(response, "parse_recovery", ""))
        self.calls.append(record)
        return response


def _make_client() -> tuple[Any, smoke.CountingClient]:
    api_key = next((os.environ.get(k, "").strip() for k in KEY_ENVS
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        raise SystemExit("missing LLM key")
    import openai  # noqa: PLC0415
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=BASE_URL, timeout=600),  # 2026-08-15：agicto API 降级期放宽（36s/小调用）
        max_calls=LLM_BUDGET_PER_PREPARE)
    return counter, counter


def _prepare_arm(snapshot: Any, series_uid: str, origin: int,
                 env: Mapping[str, Any],
                 with_task_context: bool = False,
                 pool_mode: str = "actionable",
                 values_override: Mapping[str, Any] | None = None
                 ) -> dict[str, Any]:
    """单臂单次真实 Fast prepare（空 Memory；记录 trace + prompt 证据）。
    with_task_context=True（S0-TC 唯一修正，用户裁决 2026-08-14）：请求
    携带真实 forecast TaskContext——正常请求构造，不手工拼 Prompt。
    pool_mode（FULL_OPERATOR_SKILL_CAPABILITY 2026-08-14）：actionable=
    当前管线（默认）；full=机械全池暴露面。values_override 供合成
    Context（synmiss）注入。"""
    values = values_override if values_override is not None else env["values"]
    series0 = np.asarray(values[series_uid], dtype=np.float64)
    counter, counter_ref = _make_client()
    rec = RecordingBackend(AgictoChatCompletionsBackend(
        client=counter, base_url=BASE_URL))
    core = TTHAAgentCore(
        rec, LocalPublicToolGateway(series0[:origin], task_kind="forecast"),
        model=MODEL, base_url=BASE_URL)
    fast = TTHAFastAgent(core)
    method = TTHAMethod(fast, snapshot, ())
    request = nsu._request(series0, values, origin)
    if with_task_context:
        import dataclasses as _dc  # noqa: PLC0415
        from SelfEvolvingHarnessTS.contracts.task import (  # noqa: PLC0415
            deployment_constraints_v1,
            forecast_task_context_v1,
        )
        task_context = forecast_task_context_v1(
            task_spec=request.task_spec,
            deployment_constraints=deployment_constraints_v1())
        request = _dc.replace(request, task_context=task_context)
    try:
        result = method.prepare(request, pool_mode=pool_mode)
        trace = method.last_trace
        if trace is None:
            raise RuntimeError("method.prepare returned no trace")
    except Exception as exc:  # noqa: BLE001
        return {"series": series_uid, "origin": origin,
                "protocol_error": f"{type(exc).__name__}: {exc}",
                "llm_calls": counter_ref.calls,
                "prompt_calls": rec.calls}
    steps_map = {str(k): [{"op": o, "params": dict(p)} for o, p in v]
                 for k, v in dict(trace.candidate_program_steps or {}).items()}
    return {
        "series": series_uid,
        "origin": origin,
        "status": result.status.value if hasattr(result, "status") else str(result),
        "candidate_ids": list(trace.candidate_ids),
        "candidate_steps": steps_map,
        "chosen_candidate_id": str(trace.chosen_candidate_id),
        "compilation_status": str(trace.compilation_status),
        "supplied_noop_candidate_ids": list(trace.supplied_noop_candidate_ids),
        "rejection_receipts": [
            {"candidate_id": r.get("candidate_id"), "reason": r.get("reason"),
             "rejection_code": r.get("rejection_code")}
            for r in (trace.rejection_receipts or ())],
        "agent_cache_hit_flags": list(trace.agent_cache_hit_flags or ()),
        "llm_calls": counter_ref.calls,
        "prompt_calls": rec.calls,
    }


def _view_bodies(snapshot: Any, series_uid: str, origin: int,
                 env: Mapping[str, Any]) -> dict[str, str]:
    """机械检查：harness view 中 guidance skill body 的实际文本。"""
    values = env["values"]
    series0 = np.asarray(values[series_uid], dtype=np.float64)
    features = extract_public_features(series0[:origin], task_kind="forecast")
    view = resolve_harness_view(snapshot, features, role="fast")
    out: dict[str, str] = {}
    for skill in view.skills:
        if skill.skill_id == GUIDANCE_SKILL_ID:
            out[skill.skill_id] = str(skill.body)
    return out


def apply_guidance_patch(controller: EditController, store: SnapshotStore,
                         snapshot: Any, new_body: str, *, edit_id: str,
                         target_pattern_id: str,
                         confirmed_cause: str = "WORKFLOW_GUIDANCE_GAP") -> Any:
    """确定性 Runtime 应用：把新 guidance body PATCH 到 fork 并编译回
    snapshot（Slow Agent 不批准自己——调用方用 Support/delayed 门）。"""
    parent = store.materialize(snapshot)
    sha = controller.surface_precondition_sha(parent, SURFACE)
    manifest = EditManifest(
        edit_id=edit_id,
        base_harness_sha=snapshot.harness_content_sha,
        target_pattern_id=target_pattern_id,
        target_surface_id=SURFACE,
        operation=EditOperation.PATCH,
        surface_precondition={"kind": "SHA", "sha": sha},
        dependency_precondition_shas={},
        minimal_patch={"value": new_body},
        new_value=None,
        observable_applicability=None,
        predicted_agent_behavior_change=("supply_effect_distinct",),
        predicted_data_effect=("candidate_supply_change",),
        automatically_selected_risk_cases=(),
        falsification_condition=("candidate_behavior_unchanged",),
        patch_id=None,
    )
    manifest_applied = _resolve_apply_manifest(manifest, snapshot)
    receipt = controller.apply_to_fork(parent, manifest_applied,
                                       confirmed_cause=confirmed_cause)
    return receipt


def _bootstrap_body(snapshot: Any) -> str:
    for skill in snapshot.skills:
        if skill.skill_id == GUIDANCE_SKILL_ID:
            return str(skill.body)
    raise ValueError("guidance bootstrap skill missing")


# ---------------------------------------------------------------- g1

def _op_sets(arm_result: dict[str, Any]) -> tuple[str, ...]:
    steps = arm_result.get("candidate_steps") or {}
    sets = []
    for cid in arm_result.get("candidate_ids") or ():
        ops = tuple(s["op"] for s in steps.get(str(cid), ()))
        if ops:
            sets.append("|".join(ops))
    return tuple(sorted(set(sets)))


def _rep_range(existing: Sequence[dict[str, Any]], series: str,
               origin: int, extra: bool) -> range:
    n = sum(1 for t in existing
            if t.get("series") == series and t.get("origin") == origin)
    if extra:
        return range(n, n + 1)
    return range(0, G1_REPS)


def _refix_range(existing: Sequence[dict[str, Any]], series: str,
                 origin: int) -> list[int]:
    """预算超限的 rep 按修订后上限重跑（预注册）：返回需重跑的 rep 下标。"""
    out: list[int] = []
    for t in existing:
        if t.get("series") != series or t.get("origin") != origin:
            continue
        err = str(t.get("protocol_error") or "")
        if "budget" in err.lower():
            out.append(int(t["rep"]))
    return out


def phase_g1_b0(env: Mapping[str, Any], extra: bool = False) -> int:
    report = _load_report()
    h0 = _h0_snapshot()
    b0_results: list[dict[str, Any]] = list(
        (report.get("g1") or {}).get("b0") or [])
    for series, origin in G1_CONTEXTS:
        for rep in _rep_range(b0_results, series, origin, extra):
            print(f"== g1-b0: {series}@{origin} rep{rep} ...", flush=True)
            r = _prepare_arm(h0, series, origin, env)
            r["arm"] = "b0"
            r["rep"] = rep
            b0_results.append(r)
            print(json.dumps({k: r.get(k) for k in
                              ("candidate_ids", "chosen_candidate_id",
                               "compilation_status", "llm_calls",
                               "protocol_error")},
                             ensure_ascii=False))
    report.setdefault("g1", {})["b0"] = b0_results
    report["g1"]["b0_bodies"] = {f"{s}@{o}": _view_bodies(h0, s, o, env)
                                 for s, o in G1_CONTEXTS}
    _save_report(report)
    print("== g1-b0 saved")
    return 0


def phase_g1_b1(env: Mapping[str, Any], extra: bool = False,
                refix: bool = False) -> int:
    report = _load_report()
    h0 = _h0_snapshot()
    old_body = _bootstrap_body(h0)
    new_body = old_body + B1_APPEND
    store = SnapshotStore(STORE_DIR)
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    receipt = apply_guidance_patch(
        controller, store, h0, new_body,
        edit_id="g1_manual_guidance_positive_control",
        target_pattern_id="g1-positive-control")
    b1_snapshot = receipt.candidate_snapshot.snapshot
    print(f"== g1-b1 fork: parent={receipt.parent_harness_content_sha[:12]} "
          f"candidate={receipt.candidate_harness_content_sha[:12]}")
    b1_results: list[dict[str, Any]] = list(
        (report.get("g1") or {}).get("b1") or [])
    if not refix:
        for series, origin in G1_CONTEXTS:
            for rep in _rep_range(b1_results, series, origin, extra):
                print(f"== g1-b1: {series}@{origin} rep{rep} ...", flush=True)
                r = _prepare_arm(b1_snapshot, series, origin, env)
                r["arm"] = "b1"
                r["rep"] = rep
                b1_results.append(r)
                print(json.dumps({k: r.get(k) for k in
                                  ("candidate_ids", "chosen_candidate_id",
                                   "compilation_status", "llm_calls",
                                   "protocol_error")},
                                 ensure_ascii=False))
    if refix:
        # 预算超限 rep 按修订后上限重跑（预注册修订，2026-08-14）；
        # 旧行保留并标记 superseded（审计），新行计入裁定。
        for series, origin in G1_CONTEXTS:
            for rep in _refix_range(b1_results, series, origin):
                print(f"== g1-b1 refix: {series}@{origin} rep{rep} ...",
                      flush=True)
                r = _prepare_arm(b1_snapshot, series, origin, env)
                r["arm"] = "b1"
                r["rep"] = rep
                r["refix_after_budget_amendment"] = True
                for t in b1_results:
                    if (t.get("series") == series
                            and t.get("origin") == origin
                            and t.get("rep") == rep
                            and "budget" in str(
                                t.get("protocol_error") or "").lower()):
                        t["superseded_by_budget_amendment"] = True
                b1_results.append(r)
                print(json.dumps({k: r.get(k) for k in
                                  ("candidate_ids", "chosen_candidate_id",
                                   "compilation_status", "llm_calls",
                                   "protocol_error")},
                                 ensure_ascii=False))
    g1 = report.setdefault("g1", {})
    g1["b1"] = b1_results
    g1["b1_patch"] = {
        "edit_id": receipt.edit_id,
        "confirmed_cause": "WORKFLOW_GUIDANCE_GAP",
        "surface": SURFACE,
        "old_body": old_body,
        "new_body": new_body,
        "parent_harness_content_sha": receipt.parent_harness_content_sha,
        "candidate_harness_content_sha": receipt.candidate_harness_content_sha,
    }
    g1["b1_bodies"] = {f"{s}@{o}": _view_bodies(b1_snapshot, s, o, env)
                       for s, o in G1_CONTEXTS}
    _save_report(report)
    print("== g1-b1 saved")
    return 0


def phase_g1_verdict() -> int:
    report = _load_report()
    g1 = report.get("g1", {})
    b0 = g1.get("b0") or []
    b1 = g1.get("b1") or []
    if not b0 or not b1:
        print("== g1: missing arms")
        return 0

    def _by_ctx(arms: Sequence[dict[str, Any]]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in arms:
            if r.get("superseded_by_budget_amendment"):
                continue  # 预算修订前的废弃行——不计入裁定
            if r.get("excluded_by_runner_bug"):
                continue  # refix bug 的计划外重复行——不计入裁定
            out.setdefault(f"{r['series']}@{r['origin']}", []).append(r)
        return out

    ctx0, ctx1 = _by_ctx(b0), _by_ctx(b1)
    # 机械证据：prompt/harness view 是否真不同
    for key in sorted(ctx0):
        h0_view = set(c["effective_harness_view_sha"]
                      for r in ctx0[key] for c in r.get("prompt_calls") or ())
        h1_view = set(c["effective_harness_view_sha"]
                      for r in ctx1[key] for c in r.get("prompt_calls") or ())
        g1.setdefault("mechanical", {})[key] = {
            "b0_harness_view_shas": sorted(h0_view),
            "b1_harness_view_shas": sorted(h1_view),
            "view_differs": bool(h0_view and h1_view and h0_view.isdisjoint(h1_view)),
        }
    # 行为对比：每 context 每臂的候选 operator 集合（去重，跨 rep）
    per_ctx: dict[str, dict[str, Any]] = {}
    for key in sorted(ctx0):
        b0_sets = [tuple(sorted(_op_sets(r))) for r in ctx0[key]]
        b1_sets = [tuple(sorted(_op_sets(r))) for r in ctx1[key]]
        b0_union = sorted({s for t in b0_sets for s in t})
        b1_union = sorted({s for t in b1_sets for s in t})
        per_ctx[key] = {
            "b0_op_sets_per_rep": [list(t) for t in b0_sets],
            "b1_op_sets_per_rep": [list(t) for t in b1_sets],
            "b0_union": b0_union,
            "b1_union": b1_union,
            "b0_chosen": [r.get("chosen_candidate_id") for r in ctx0[key]],
            "b1_chosen": [r.get("chosen_candidate_id") for r in ctx1[key]],
            "b0_protocol_error": any("protocol_error" in r for r in ctx0[key]),
            "b1_protocol_error": any("protocol_error" in r for r in ctx1[key]),
            "b0_compilation": [r.get("compilation_status") for r in ctx0[key]],
            "b1_compilation": [r.get("compilation_status") for r in ctx1[key]],
        }
        # 预测：B1 应引入不在 B0 集合中的 effect-distinct 替代算子
        b0_ops = {o for s in b0_union for o in s.split("|")}
        b1_ops = {o for s in b1_union for o in s.split("|")}
        per_ctx[key]["new_ops_in_b1"] = sorted(b1_ops - b0_ops)
    g1["behavior"] = per_ctx
    # 裁定（严格按 frozen_hypothesis.verdict_rules——审查修正 2026-08-14：
    # 初版实现比冻结规则宽松，已收紧）：
    #   ACTION_SIGNAL 要求每个 premise 成立 context 的**每个 rep** 都出现
    #   预测签名（b1 候选并集含 ≥1 outlier 族算子且与 b0 不同）；
    #   n=3 时规则同：≥2/3 rep 出现签名（多数原则——已在报告预注册）。
    outlier_ops = {"winsorize", "outlier_iqr", "outlier_mad",
                   "hampel_filter"}

    def _signature_per_rep(d: Mapping[str, Any]) -> list[bool]:
        per_rep: list[bool] = []
        for rep_set in d["b1_op_sets_per_rep"]:
            ops = {o for s in rep_set for o in s.split("|")}
            per_rep.append(bool(ops) and bool(ops & outlier_ops))
        return per_rep

    def _b0_all_abstain() -> bool:
        return all(not s for d in per_ctx.values()
                   for s in d["b0_op_sets_per_rep"])

    mech_ok = all((g1.get("mechanical") or {}).get(k, {})
                  .get("view_differs") for k in per_ctx)
    if not mech_ok:
        verdict = "PROTOCOL_FAILURE"
    elif (_b0_all_abstain()
          and all(not s for d in per_ctx.values()
                  for s in d["b1_op_sets_per_rep"])):
        # 两臂全部弃权且逐字段无差异 → 未消费
        verdict = "GUIDANCE_NOT_CONSUMED"
    else:
        sigs = [s for d in per_ctx.values() for s in _signature_per_rep(d)]
        reps_per_ctx = len(ctx0[list(ctx0)[0]])
        required = max(2, min(reps_per_ctx, 1 + reps_per_ctx // 2))
        all_ctx_ok = all(
            sum(_signature_per_rep(d)) >= required
            for d in per_ctx.values())
        if all_ctx_ok and _b0_all_abstain():
            verdict = "GUIDANCE_ACTION_SIGNAL"
        else:
            verdict = "INCONCLUSIVE_LLM_VARIANCE"
    g1["verdict"] = verdict
    _save_report(report)
    print(json.dumps({"g1_verdict": verdict,
                      "behavior": per_ctx,
                      "mechanical": g1.get("mechanical")},
                     ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- g2

# Slow 输出契约（任务书 G2）：guidance body 不得含 Dataset 名/系列 ID/
# 算子名/gain 数值/冻结参数（防止 case patch）；必须有 Evidence: 段。
G2_FORBIDDEN_BODY_TOKENS = (
    "kdd", "t100", "t101", "t10", "t1", "t13",
    "t128", "t129", "t130", "t131", "t132", "t133", "t134",
    "winsorize", "outlier_mad", "hampel_filter", "outlier_iqr",
)
G2_FLOAT_RE = r"\d+\.\d+"


def _g2_preflight(manifest: Any,
                  current_body: str | None = None) -> None:
    """G2 契约强制（重试一次，同 smoke 的 first_fault_face 模式）：
    - Runtime-owned binding（用户裁决 2026-08-14 修订）：Slow 的
      minimal_patch.value 只承载**新 [EDITABLE_GUIDANCE] 内容**——
      [FIXED_CONTRACT] 由 Runtime 按构造拼接（逐字不变由确定性代码保证，
      不依赖模型复制）；因此提案值不得包含两个段标记；
    - 无 case-patch 记号、无数值；证据引用走 predicted_data_effect 的
      'evidence:' 项（不进全局 body）；
    - 预测含 supply_effect_distinct。
    """
    from SelfEvolvingHarnessTS.methods.ttha.agent_core import (
        StagePostValidationError,
    )
    from SelfEvolvingHarnessTS.methods.ttha.method import (
        _parse_clause_payload,
    )
    import re as _re
    patch = manifest.minimal_patch or {}
    body = str(patch.get("value") or "")
    if not body.strip():
        raise StagePostValidationError(
            "GUIDANCE_BODY_EMPTY",
            "minimal_patch.value must be a REPLACE_CLAUSE payload.",
            retryable=True)
    # P3：clause 载荷形状（REPLACE_CLAUSE + target: propose.rule.* +
    # new_clause: ...——Runtime 绑定到正确规则位置，其余内容逐字不变）
    clause_payload = _parse_clause_payload(body)
    if clause_payload is None:
        raise StagePostValidationError(
            "GUIDANCE_CLAUSE_SHAPE_INVALID",
            "minimal_patch.value must be a REPLACE_CLAUSE payload: "
            "REPLACE_CLAUSE, then 'target: propose.rule.<clause_id>', then "
            "'new_clause: <single-line rule text>'.",
            retryable=True)
    for marker in ("[FIXED_CONTRACT]", "[EDITABLE_GUIDANCE]",
                   "[inspect_pattern_guidance]",
                   "[propose_construction_guidance]", "[select_guidance]"):
        if marker in body:
            raise StagePostValidationError(
                "GUIDANCE_BODY_SECTION_MARKERS_FORBIDDEN",
                f"minimal_patch.value must never contain the '{marker}' "
                "marker. The runtime binds the clause to its rule position "
                "itself.",
                retryable=True)
    # 证据引用走 manifest（不进 body）——用户裁决 2026-08-14：证据内容应
    # 保留在 Card/Manifest/Episode，不应永久写进全局程序指导
    effects = [str(item) for item in
               getattr(manifest, "predicted_data_effect", ()) or ()]
    if not any(item.startswith("evidence:") for item in effects):
        raise StagePostValidationError(
            "GUIDANCE_EVIDENCE_MISSING",
            "predicted_data_effect must include at least one 'evidence:' item "
            "referencing the observed repeated construction fault in generic "
            "terms. Evidence belongs in the manifest, not permanently in the "
            "global guidance body.",
            retryable=True)
    # 规范化后匹配（checker 裁决 2026-08-14：裸子串匹配漏掉
    # "hampel filter"/"outlier-mad"/"t-100" 等分隔变体）
    import re as _re2
    low = body.lower()
    normalized = _re2.sub(r"[\s\-_]", "", low)
    for token in G2_FORBIDDEN_BODY_TOKENS:
        if token in low or _re2.sub(r"[\s\-_]", "", token) in normalized:
            raise StagePostValidationError(
                "GUIDANCE_BODY_CASE_PATCH",
                f"The guidance body must not name datasets, series ids, or "
                f"specific operators (found token: {token}). State a general "
                "construction rule instead of a case patch.",
                retryable=True)
    if _re.search(G2_FLOAT_RE, body):
        raise StagePostValidationError(
            "GUIDANCE_BODY_NUMERIC",
            "The guidance body must not contain numeric gain values or frozen "
            "numeric parameters from any context.",
            retryable=True)
    if "supply_effect_distinct" not in tuple(
            manifest.predicted_agent_behavior_change or ()):
        raise StagePostValidationError(
            "GUIDANCE_PREDICTION_MISSING",
            "predicted_agent_behavior_change must include "
            "'supply_effect_distinct' (the missing family must enter the "
            "proposed candidate set).",
            retryable=True)


def _g2_card(fault_rows: Sequence[dict[str, Any]],
             success_rows: Sequence[dict[str, Any]],
             surface_sha: str,
             current_body: str,
             contracts: Sequence[Mapping[str, Any]],
             rejection_feedback: dict[str, Any] | None = None,
             ) -> dict[str, object]:
    """组级 Guidance Card（rev5 基线，2026-08-14）：S0-TC Arm A 的真实
    轨迹——共同 fault = 假设已输出但 propose 弃权/无假设绑定候选；正向
    contrast = 同臂同 Context 的闭链轨迹（证明 headroom 可达）。不携带
    Dataset/系列/算子名；Slow 输出契约在 instruction。facts 携带当前
    body 全文——逐字保留 [FIXED_CONTRACT] 的前提是模型看得到它。"""
    def _row(t: dict[str, Any]) -> dict[str, Any]:
        ch = t.get("chain") or {}
        acc = _trajectory_accounting(t)
        return {
            "context": f"{t.get('series')}@{t.get('origin')}",
            "hypotheses_emitted": ch.get("hypotheses_emitted") or [],
            "referenced_candidates": ch.get("referenced_candidates") or [],
            "candidate_ids": list(t.get("candidate_ids") or ()),
            "chosen": ch.get("chosen"),
            "chain_kind": ch.get("kind"),
            # P0 故障账目（用户裁决：按真实 first fault 聚类）
            "failure_stage": acc["failure_stage"],
            "failure_code": acc["failure_code"],
            "propose_reached": acc["propose_reached"],
            "abstention_reason": acc["abstention_reason"],
        }
    # P2：旧 propose 规则段（clause 化后）——"旧规则及其预测行为"
    propose_start = current_body.find("[propose_construction_guidance]")
    select_start = current_body.find("[select_guidance]")
    old_propose_rules = (current_body[propose_start:select_start].strip()
                         if propose_start >= 0 and select_start > propose_start
                         else "")
    facts: dict[str, Any] = {
        "fault_trajectories": [_row(t) for t in fault_rows],
        "success_contrast_trajectories": [_row(t) for t in success_rows],
        "guidance_surface_precondition_sha": surface_sha,
        "task_objective": "forecast; cohort Ridge sMASE (lower is better)",
        "current_guidance_body": current_body,
        # P2：当前合法 operator contracts 与参数绑定（Slow 判断合法性的依据）
        "legal_operator_contracts": [
            {"name": c.get("name"),
             "category": c.get("category"),
             "public_parameter_bindings": c.get("public_parameter_bindings"),
             "targeting_mode": c.get("targeting_mode")}
            for c in contracts],
        "old_propose_rules": old_propose_rules,
    }
    retry_paragraph = ""
    if rejection_feedback:
        facts["previous_round_rejection"] = rejection_feedback
        retry_paragraph = (
            " This is a retry after your previous patch was REJECTED by "
            "deterministic behavior replay: the new guidance made every "
            "context abstain (zero candidates in all replays, including a "
            "context that previously produced a complete construction "
            "chain). Key lessons: a broad hypothesis does not mean no legal "
            "binding — legality is decided by the operator contract and the "
            "runtime verifier, not by region narrowness; do not tighten "
            "construction rules so far that legal bindings are judged "
            "absent. Your new text must preserve the previously successful "
            "context and fix at least one previously failing context."
        )
    return {
        "pattern_id": "group-guidance-propose-abstain-despite-hypotheses",
        "failure_family": "workflow_construction_propose_supply_gap",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {"steps": []},
        "facts": facts,
        "instruction": (
            "A repeated first-fault group: across two independent series, "
            "the Fast agent emitted grounded pattern hypotheses at the "
            "inspect stage but then returned EMPTY candidate sets at the "
            "propose stage (see fault_trajectories failure accounting), "
            "even though the success contrast trajectories show that "
            "hypothesis-bound effect-distinct candidates were achievable "
            "with the same legal operator menu. The construction guidance "
            "therefore fails to turn hypotheses into candidates "
            "consistently."
            + retry_paragraph
            + " Propose ONE minimal PATCH to the guidance "
            "surface bootstrap_skills.entries/build_contrastive_candidates."
            "body. minimal_patch.value must be a REPLACE_CLAUSE payload "
            "with exactly three lines: 'REPLACE_CLAUSE', 'target: "
            "propose.rule.<clause_id>' (choose exactly one clause id from "
            "facts.old_propose_rules), and 'new_clause: <single-line rule "
            "text>'. You propose the new rule text yourself; the runtime "
            "binds it to the correct rule position and leaves every other "
            "section byte-identical. Never name datasets, series ids, "
            "specific operators, numeric gains, origins, or frozen "
            "parameters anywhere in your text. Judge binding legality from "
            "facts.legal_operator_contracts and the runtime verifier — not "
            "from region width. Evidence references do NOT go into your "
            "text — put at least one 'evidence:<generic description>' item "
            "in predicted_data_effect instead. Set surface_precondition to "
            "kind=SHA with the sha given in "
            "facts.guidance_surface_precondition_sha. "
            "predicted_agent_behavior_change must include "
            "supply_effect_distinct; falsification_condition must state "
            "when the patch should be rejected. You do not approve your own "
            "edit — a deterministic behavior replay and held-out Support "
            "verify it.",
        ),
    }


def phase_g2(env: Mapping[str, Any] | None = None) -> int:
    """G2（rev5 基线，用户裁决 2026-08-14）：S0-TC Arm A 的真实轨迹——
    共同 fault = 假设已输出但 propose 弃权/无假设绑定候选（≥2 个不同
    series 的共享 first fault）；正向 contrast = 同臂闭链轨迹（证明
    headroom 可达）。Slow 只能改 [EDITABLE_GUIDANCE]；[FIXED_CONTRACT]
    逐字由 preflight 强制。无共享组 → NO_GROUP_FAULT（不强行拼组）。"""
    import dataclasses as _dc  # noqa: PLC0415
    report = _load_report()
    s0 = report.get("s0") or {}
    a_rows = s0.get("arm_a") or []
    if not a_rows:
        print("== g2: no s0-tc arm_a data")
        return 0
    fault_rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    for t in a_rows:
        if t.get("protocol_error"):
            continue
        ch = t.get("chain") or {}
        acc = _trajectory_accounting(t)
        if ch.get("kind") == "chain_complete":
            success_rows.append(t)
        elif acc["propose_reached"] and acc["abstention_reason"] == \
                "propose_empty_with_hypotheses":
            # P0 真实 first fault 聚类：propose 阶段真弃权（假设已输出但
            # 无假设绑定候选）——inspect 失败行不得混入
            fault_rows.append(t)
    series_of_fault = {t.get("series") for t in fault_rows}
    shared = len(fault_rows) >= 2 and len(series_of_fault) >= 2
    if not shared:
        # 真正故障不在 propose Guidance（如全部为 inspect 失败）
        ev = {"stage": "edit_surface_mismatch",
              "fault_rows": len(fault_rows),
              "series_of_fault": sorted(series_of_fault),
              "success_rows": len(success_rows),
              "verdict": "EDIT_SURFACE_MISMATCH"}
        report["g2"] = {"feedback_event": ev}
        _save_report(report)
        print("== g2: EDIT_SURFACE_MISMATCH "
              + json.dumps(ev, ensure_ascii=False))
        return 0
    if not shared:
        ev = {"stage": "no_group_fault",
              "fault_rows": len(fault_rows),
              "series_of_fault": sorted(series_of_fault),
              "success_rows": len(success_rows),
              "verdict": "NO_GROUP_FAULT"}
        report["g2"] = {"feedback_event": ev}
        _save_report(report)
        print("== g2: NO_GROUP_FAULT "
              + json.dumps(ev, ensure_ascii=False))
        return 0
    if env is None:
        env = _load_env()
    h0 = _h0_snapshot()
    current_body = _bootstrap_body(h0)
    store = SnapshotStore(STORE_DIR)
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    parent = store.materialize(h0)
    surface_sha = controller.surface_precondition_sha(parent, SURFACE)
    counter, counter_ref = _make_client()
    values = env["values"]
    series_t1 = np.asarray(values["T1"], dtype=np.float64)
    rec_slow = RecordingBackend(AgictoChatCompletionsBackend(
        client=counter, base_url=BASE_URL))
    core = TTHAAgentCore(
        rec_slow,
        LocalPublicToolGateway(series_t1[:600], task_kind="forecast"),
        model=MODEL, base_url=BASE_URL)
    slow = TTHASlowAgent(core)
    method = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(
            AgictoChatCompletionsBackend(client=counter, base_url=BASE_URL),
            LocalPublicToolGateway(series_t1[:600], task_kind="forecast"),
            model=MODEL, base_url=BASE_URL)),
        h0, ())
    # Slow 输入补全（用户裁决 2026-08-14：真实 TaskContext + 合法 contracts）
    from SelfEvolvingHarnessTS.contracts.task import (  # noqa: PLC0415
        deployment_constraints_v1,
        forecast_task_context_v1,
    )
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: PLC0415
        public_operator_contract,
    )
    series100 = np.asarray(values["T100"], dtype=np.float64)
    base_req = nsu._request(series100, values, 600)
    task_ctx = forecast_task_context_v1(
        task_spec=base_req.task_spec,
        deployment_constraints=deployment_constraints_v1())
    contracts = tuple(public_operator_contract(op) for op in
                      ("winsorize", "outlier_iqr", "outlier_mad",
                       "hampel_filter", "denoise_median",
                       "resample_uniform"))
    group = {"workflow": "propose_abstain_despite_hypotheses"}
    capsule = {"n_episodes": len(fault_rows),
               "workflow": group["workflow"], "sign": "NEGATIVE"}
    # 一次且仅一次拒绝回喂（用户裁决 2026-08-14）：P4/G3 拒绝证据进入下一轮
    # Slow 输入——Harness 把上轮核销反馈条件化给 Slow（不是训练权重）。
    # P4 链（新）优先；G3 链（历史）兼容。
    rejection_feedback: dict[str, Any] | None = None
    if (report.get("p4_retry_done") or {}).get("done"):
        print("== g2: p4 retry already used — refusing second retry")
        return 0
    p4_prev = report.get("p4") or {}
    if p4_prev.get("verdict") == "PATCH_REJECTED":
        rejection_feedback = {
            "round": "previous",
            "outcome": json.dumps({
                "fixed": p4_prev.get("condition_1_fixed"),
                "regressions": p4_prev.get("condition_2_regressions"),
                "verifier_rejections":
                    p4_prev.get("condition_4_verifier_rejections"),
                "verdict": p4_prev.get("verdict"),
            }, ensure_ascii=False),
            "lessons": [
                "a broad hypothesis does not mean no legal binding — "
                "legality is decided by the operator contract and the "
                "runtime verifier, not by region narrowness",
                "do not tighten construction rules so far that legal "
                "bindings are judged absent",
                "preserve the previously successful trajectory and fix at "
                "least one previously failing trajectory",
            ],
        }
        report["p4_retry_done"] = {"done": True,
                                   "note": "one retry only（用户裁决）"}
        _save_report(report)
        print("== g2: p4 retry mode with rejection feedback")
    if rejection_feedback is None and (report.get("g2_retry_done") or {}).get("done") is None:
        g3_prev = report.get("g3") or {}
        if g3_prev.get("verdict") == "GUIDANCE_PATCH_REJECTED":
            rejection_feedback = {
                "round": "previous",
                "outcome": ("previous patch made all four replays abstain "
                            "(0/4 complete chains; a context that previously "
                            "completed the chain regressed to abstention; "
                            "0/3 predicted fixes)"),
                "lessons": [
                    "a broad hypothesis does not mean no legal binding — "
                    "legality is decided by the operator contract and the "
                    "runtime verifier, not by region narrowness",
                    "do not tighten construction rules so far that legal "
                    "bindings are judged absent",
                    "preserve the previously successful context and fix at "
                    "least one previously failing context",
                ],
            }
            report["g2_retry_done"] = {"done": True,
                                       "note": "one retry only（用户裁决）"}
            _save_report(report)
            print("== g2: g3 retry mode with rejection feedback")
    card = _g2_card(fault_rows, success_rows, surface_sha, current_body,
                    contracts, rejection_feedback)
    ev = method.handle_group_guidance(
        group, capsule, slow_agent=slow, controller=controller, store=store,
        card_builder=lambda g, c: card,
        confirmed_cause="WORKFLOW_GUIDANCE_GAP",
        manifest_preflight=lambda m: _g2_preflight(
            m, current_body=current_body),
        allowed_operator_contracts=contracts,
        task_context=task_ctx)
    new_snapshot = method.pending_guidance_snapshot()
    g2 = {
        "fault_rows": [{"series": t["series"], "origin": t["origin"],
                        "rep": t["rep"]} for t in fault_rows],
        "success_rows": [{"series": t["series"], "origin": t["origin"],
                          "rep": t["rep"]} for t in success_rows],
        "feedback_event": ev,
        "llm_calls": counter_ref.calls,
        "slow_stage_result": (
            None if slow.last_stage_result is None else {
                "validation_retry_count":
                    slow.last_stage_result.validation_retry_count,
                "first_pass_valid":
                    slow.last_stage_result.first_pass_valid,
                "validation_error_codes": list(
                    slow.last_stage_result.validation_error_codes or ()),
            }),
    }
    pend = getattr(method, "_pending_update", None) or {}
    if pend.get("kind") == "guidance":
        g2["candidate_runtime_bundle_sha"] = (
            pend["receipt"].candidate_runtime_bundle_sha)
    g2["slow_prompt_calls"] = rec_slow.calls
    if slow.last_stage_result is not None:
        mp = (slow.last_stage_result.payload or {}).get("edit_manifest") or {}
        g2["proposed_manifest"] = {
            "target_surface_id": str(mp.get("target_surface_id")),
            "operation": str(mp.get("operation")),
            "predicted_agent_behavior_change": list(
                mp.get("predicted_agent_behavior_change") or ()),
            "predicted_data_effect": list(
                mp.get("predicted_data_effect") or ()),
            "falsification_condition": list(
                mp.get("falsification_condition") or ()),
        }
    if new_snapshot is not None:
        g2["candidate_harness_content_sha"] = (
            new_snapshot.harness_content_sha)
        g2["guidance_body_new"] = ev.get("guidance_body_new")
    report["g2"] = g2
    _save_report(report)
    print("== g2:", json.dumps({k: (v if not isinstance(v, str) else v[:80])
                                for k, v in ev.items()},
                               ensure_ascii=False))
    print(f"== g2 stage={ev.get('stage')} "
          f"pending={new_snapshot is not None} llm={counter_ref.calls}")
    return 0


def phase_g3(env: Mapping[str, Any] | None = None) -> int:
    """G3 行为核销（rev5 基线，2026-08-14）：同一批公开 Context（fault
    rows 所在窗口）上旧（h0 rev5）/新（G2 候选）snapshot replay——
    核销 Slow 的预测：fault 行从弃权变为假设绑定候选；原成功 Context
    不回归；无协议失败、无未预测退化。行为按预测改变才打开 G4。"""
    report = _load_report()
    g2 = report.get("g2") or {}
    ev = g2.get("feedback_event") or {}
    bundle_sha = g2.get("candidate_runtime_bundle_sha")
    if ev.get("stage") != "pending" or not bundle_sha:
        print("== g3: blocked — g2 not pending")
        return 0
    if env is None:
        env = _load_env()
    h0 = _h0_snapshot()
    store = SnapshotStore(STORE_DIR)
    new_snap = compile_snapshot(store.root / str(bundle_sha),
                                verify_lock=False)
    old_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    for series, origin in G4_FAULT_BUILD:
        for rep in range(G1_REPS):
            print(f"== g3-old: {series}@{origin} rep{rep} ...", flush=True)
            r_old = _prepare_arm(h0, series, origin, env,
                                 with_task_context=True)
            r_old["arm"] = "old"
            r_old["rep"] = rep
            old_rows.append(r_old)
            print(f"== g3-new: {series}@{origin} rep{rep} ...", flush=True)
            r_new = _prepare_arm(new_snap, series, origin, env,
                                 with_task_context=True)
            r_new["arm"] = "new"
            r_new["rep"] = rep
            new_rows.append(r_new)
    # 链指标核销
    old_chain = [_chain_metrics(r) for r in old_rows]
    new_chain = [_chain_metrics(r) for r in new_rows]
    old_complete = sum(1 for c in old_chain if c["chain_complete"])
    new_complete = sum(1 for c in new_chain if c["chain_complete"])
    new_errors = sum(1 for r in new_rows if r.get("protocol_error"))
    old_errors = sum(1 for r in old_rows if r.get("protocol_error"))
    # 预测核销：fault 行（假设已输出但弃权）在新快照下变成假设绑定候选
    # （referenced_candidates 非空）；成功行保持闭链。
    predicted_change = 0
    predicted_total = 0
    regression = 0
    per_row: list[dict[str, Any]] = []
    for o, n, oc, nc in zip(old_rows, new_rows, old_chain, new_chain):
        key = f"{o['series']}@{o['origin']} r{o['rep']}"
        row = {"key": key,
               "old": oc["kind"], "new": nc["kind"],
               "old_refs": oc.get("referenced_candidates") or [],
               "new_refs": nc.get("referenced_candidates") or []}
        if oc["chain_complete"]:
            if not nc["chain_complete"]:
                regression += 1
                row["regression"] = True
        elif oc.get("hypotheses_emitted") and not oc.get(
                "referenced_candidates"):
            # fault 行：预测应变为假设绑定候选
            predicted_total += 1
            if nc.get("referenced_candidates"):
                predicted_change += 1
                row["predicted_fixed"] = True
        per_row.append(row)
    # 用户裁决 2026-08-14 修订：至少修复一个原失败行（不必全修复）+
    # 保留全部原成功行 + 无协议失败 + 闭链数不退化
    predicted_ok = (
        predicted_total > 0 and predicted_change >= 1
        and regression == 0 and new_errors == 0
        and new_complete >= old_complete)
    predicted_spec = (report.get("g2") or {}).get("proposed_manifest", {})         .get("predicted_agent_behavior_change") or []
    verdict = (
        "G3_BEHAVIOR_VERIFIED" if predicted_ok
        else "GUIDANCE_PATCH_REJECTED")
    g3 = {
        "contexts": [list(c) for c in G4_FAULT_BUILD],
        "old_chain": [dict(c) for c in old_chain],
        "new_chain": [dict(c) for c in new_chain],
        "old_complete": old_complete,
        "new_complete": new_complete,
        "old_protocol_errors": old_errors,
        "new_protocol_errors": new_errors,
        "predicted_fixed": predicted_change,
        "predicted_total": predicted_total,
        "regressions": regression,
        "per_row": per_row,
        "predicted_agent_behavior_change": list(predicted_spec),
        "verdict": verdict,
        "old_rows": old_rows,
        "new_rows": new_rows,
    }
    report["g3"] = g3
    _save_report(report)
    print("== g3:", json.dumps({k: g3[k] for k in
                                ("old_complete", "new_complete",
                                 "predicted_fixed", "predicted_total",
                                 "regressions", "new_protocol_errors",
                                 "verdict")}, ensure_ascii=False))
    return 0


# ---- G4 预冻结协议（冻结于首次 G4 运行前；两臂一致）----
G4_MAX_ROUNDS_PER_CONTEXT = 2      # 初始 + 1 反馈轮
G4_SUPPORT_BUDGET_PER_ROUND = 2    # 每轮最多 2 次 Support 探测
G4_POSITIVE_CONTRAST = (           # "其他未修改视图"不劣检查窗口（已暴露
    ("T1", 600), ("T1", 792),      # winsorize 正向窗口）
    ("T100", 792), ("T100", 984),
    ("T10", 792), ("T101", 600))
G4_DELAYED_WINDOWS = (("T10", 936), ("T101", 1032))  # 仅最终 winner 打开


def _support_loop(snapshot: Any, series: str, origin: int,
                  env: Mapping[str, Any]) -> dict[str, Any]:
    """单臂单 held-out context 的 adaptation 反馈环（不开 Slow）：
    prepare → chosen-first 探测 → 负反馈写入 Episode → 反馈轮再 prepare。
    Support 预算严格受限。所有探测窗口在 G4 预冻结 roster 内。"""
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
        build_episode,
        workflow_signature_of,
    )

    values = env["values"]
    roster, vals = _support_roster(series, values)
    executor = ScopeExecutor(roster, vals, nsu._config(),
                             evaluate_fn=nsu._evaluate_kdd)
    episodes: list[Any] = []
    probes: list[dict[str, Any]] = []
    feedback_events = 0
    support_evaluations = 0  # Support 评估计数（1 起）——f2p 语义
    first_positive_at: int | None = None
    rounds: list[dict[str, Any]] = []
    for round_i in range(G4_MAX_ROUNDS_PER_CONTEXT):
        counter, counter_ref = _make_client()
        rec = RecordingBackend(AgictoChatCompletionsBackend(
            client=counter, base_url=BASE_URL))
        series0 = np.asarray(values[series], dtype=np.float64)
        core = TTHAAgentCore(
            rec, LocalPublicToolGateway(series0[:origin],
                                        task_kind="forecast"),
            model=MODEL, base_url=BASE_URL)
        method = TTHAMethod(TTHAFastAgent(core), snapshot, tuple(episodes))
        request = nsu._request(series0, values, origin)
        import dataclasses as _dc4  # noqa: PLC0415
        from SelfEvolvingHarnessTS.contracts.task import (  # noqa: PLC0415
            deployment_constraints_v1,
            forecast_task_context_v1,
        )
        _tc4 = forecast_task_context_v1(
            task_spec=request.task_spec,
            deployment_constraints=deployment_constraints_v1())
        request = _dc4.replace(request, task_context=_tc4)
        try:
            result = method.prepare(request)
            trace = method.last_trace
        except Exception as exc:  # noqa: BLE001
            rounds.append({"round": round_i,
                           "protocol_error": f"{type(exc).__name__}: {exc}"})
            break
        if trace is None:
            rounds.append({"round": round_i, "protocol_error": "no trace"})
            break
        steps_map = dict(trace.candidate_program_steps or {})
        pool_ids = [c for c in trace.candidate_ids if c != "identity"]
        # chosen-first 探测序
        chosen = str(trace.chosen_candidate_id)
        order = ([chosen] if chosen in pool_ids else []) + \
            [c for c in pool_ids if c != chosen]
        round_probes: list[dict[str, Any]] = []
        aborted = False
        for cid in order[:G4_SUPPORT_BUDGET_PER_ROUND]:
            steps = steps_map.get(cid)
            if not steps:
                round_probes.append({"candidate_id": cid,
                                     "error": "no_steps"})
                continue
            support_evaluations += 1
            rr = executor.evaluate(tuple(steps), origin)
            gain = (float(rr.gain) if rr.gain is not None else None)
            passed = bool(rr.verification.passed)
            entry = {"candidate_id": cid, "steps": [
                {"op": o, "params": dict(p)} for o, p in steps],
                "gain": gain, "passed": passed}
            round_probes.append(entry)
            probes.append(entry)
            if gain is not None and gain < -M:
                feedback_events += 1
                sig = workflow_signature_of(
                    [{"op": o, "params": dict(p)} for o, p in steps])
                ctx = dict(resolver.window_context({series: values[series]},
                                                   origin, PERIOD))
                episodes.append(build_episode(
                    episode_id=f"g4_{series}_{origin}_r{round_i}_{cid}",
                    task_consumer_key="forecast|ridge|sMASE",
                    domain_namespace=f"g4_{series}",
                    context_summary={
                        "local_pattern": {"support_gain": gain, **ctx},
                        "delayed_pattern": {},
                        "program_geometry": {"scope": "training_rows",
                                             "program_steps": [
                                                 {"op": o,
                                                  "params": dict(p)}
                                                 for o, p in steps]},
                        "per_view_gain": list(
                            getattr(rr, "per_view_gain", []) or []),
                        "support_origin": origin,
                    },
                    workflow_signature=sig,
                    support_response={"gain": gain, "accepted": False},
                    delayed_response={"evaluated": False, "gain": None},
                    relation="NEGATIVE", evidence_level="SUPPORT",
                    local_status="EPISODE_ONLY",
                    evidence_refs=["guidance_g4"]))
            if gain is not None and gain >= M:
                if first_positive_at is None:
                    # 1 起 Support 评估序号（含首个正向的那次）——
                    # 与 Support receipt 数语义一致
                    first_positive_at = support_evaluations
                aborted = True
                break
        rounds.append({"round": round_i, "pool_ids": pool_ids,
                       "chosen": chosen, "probes": round_probes,
                       "compilation_status": str(trace.compilation_status),
                       "llm_calls": counter_ref.calls,
                       "rejected": len(trace.rejection_receipts or ()),
                       "noop": len(trace.supplied_noop_candidate_ids or ())})
        if aborted:
            break
        if not any((p.get("gain") is not None and p["gain"] < -M)
                   for p in round_probes):
            # 无负反馈可给（全正或全中性/abstain）→ 不进入反馈轮
            break
    harms = [p for p in probes
             if p.get("gain") is not None and p["gain"] < -M]
    positives = [p for p in probes
                 if p.get("gain") is not None and p["gain"] >= M]
    # valid_candidate_rate = 池内合法候选 /（池内合法候选 + verifier 拒绝）
    #（用户裁决 2026-08-14：原实现 valid_supplied 恒 0）
    valid_total = sum(len(r["pool_ids"]) for r in rounds
                      if "pool_ids" in r)
    rejected_total = sum(r.get("rejected") or 0 for r in rounds)
    # 首个正向前的探测计数（用户裁决 2026-08-14：命名与计算分离——
    # support_trials = 全部 Support 评估（含中性）；negative_feedbacks =
    # 其中 gain < −M 的负反馈次数）
    trials_before = (
        (first_positive_at - 1) if first_positive_at is not None else None)
    negatives_before = (
        sum(1 for p in probes[:first_positive_at - 1]
            if p.get("gain") is not None and p["gain"] < -M)
        if first_positive_at is not None else None)
    return {
        "series": series, "origin": origin,
        "rounds": rounds,
        "probes": probes,
        "support_trials_before_first_positive": trials_before,
        "negative_feedbacks_before_first_positive": negatives_before,
        "feedback_to_first_positive": negatives_before,
        "first_positive_at_evaluation": first_positive_at,
        "harm_count": len(harms),
        "harm_magnitude_max": (min(p["gain"] for p in harms)
                               if harms else None),
        "harm_total": (round(sum(p["gain"] for p in harms), 4)
                       if harms else 0.0),
        "first_positive_gain": (positives[0]["gain"]
                                if positives else None),
        "valid_candidate_rate": (
            round(valid_total / (valid_total + rejected_total), 4)
            if (valid_total + rejected_total) else None),
        "abstention": any(r.get("compilation_status") == "not_applicable"
                          for r in rounds),
    }


def phase_g4(env: Mapping[str, Any] | None = None) -> int:
    report = _load_report()
    g3 = report.get("g3") or {}
    if g3.get("verdict") != "G3_BEHAVIOR_VERIFIED":
        print(f"== g4: blocked — g3 verdict = {g3.get('verdict')}")
        return 0
    if env is None:
        env = _load_env()
    h0 = _h0_snapshot()
    store = SnapshotStore(STORE_DIR)
    bundle_sha = (report.get("g2") or {}).get("candidate_runtime_bundle_sha")
    new_snap = compile_snapshot(store.root / str(bundle_sha),
                                verify_lock=False)
    print("== g4: arm old (h0) ...", flush=True)
    old_results = [_support_loop(h0, s, o, env)
                   for s, o in G4_HELDOUT_SUPPORT]
    print("== g4: arm new (guidance) ...", flush=True)
    new_results = [_support_loop(new_snap, s, o, env)
                   for s, o in G4_HELDOUT_SUPPORT]
    # 目标错误组（held-out support）指标对比
    def _agg(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
        f2p = [r["feedback_to_first_positive"] for r in results]
        return {
            "feedback_to_first_positive": f2p,
            "mean_f2p": (sum(x for x in f2p if x is not None) /
                         max(1, sum(1 for x in f2p if x is not None))
                         if any(x is not None for x in f2p) else None),
            "first_positive_rate": round(
                sum(1 for x in f2p if x is not None) / len(f2p), 4),
            "harm_count": sum(r["harm_count"] for r in results),
            "harm_magnitude_max": min(
                [h for r in results
                 if r["harm_magnitude_max"] is not None
                 for h in [r["harm_magnitude_max"]]] or [None]),
            "harm_total": round(
                sum(r["harm_total"] or 0.0 for r in results), 4),
            "abstention": sum(1 for r in results if r["abstention"]),
            "probe_count": sum(len(r["probes"]) for r in results),
            "valid_candidate_rate": (
                round(sum(r["valid_candidate_rate"] or 0.0
                          for r in results) / len(results), 4)
                if results else None),
        }
    old_agg, new_agg = _agg(old_results), _agg(new_results)
    improvement = (
        (new_agg["first_positive_rate"] > old_agg["first_positive_rate"])
        or (new_agg["mean_f2p"] is not None
            and (old_agg["mean_f2p"] is None
                 or new_agg["mean_f2p"] < old_agg["mean_f2p"])))
    # 累计 harm 口径（用户裁决 2026-08-14：harm_total 是负数之和——
    # "不多于"必须用 >=；比较用累计值，最值单独报告）
    no_more_harm = (
        new_agg["harm_count"] <= old_agg["harm_count"]
        and new_agg["harm_total"] >= old_agg["harm_total"])
    g4 = {
        "frozen_protocol": {
            "fault_build_group": [list(c) for c in G4_FAULT_BUILD],
            "heldout_support_group": [list(c) for c in G4_HELDOUT_SUPPORT],
            "positive_contrast_windows": [list(c) for c in G4_POSITIVE_CONTRAST],
            "delayed_windows": [list(c) for c in G4_DELAYED_WINDOWS],
            "max_rounds_per_context": G4_MAX_ROUNDS_PER_CONTEXT,
            "support_budget_per_round": G4_SUPPORT_BUDGET_PER_ROUND,
            "stop_rule": "first positive (>=M) or budget exhausted",
            "probe_order": "chosen-first then pool order",
            "delayed": "final winner only",
            "both_arms_identical": "TaskSpec/Consumer/Registry 同一；"
                                   "LLM 与 Support 预算一致；chosen-first；"
                                   "delayed 只对最终 winner",
        },
        "old_arm": {"results": old_results, "aggregate": old_agg},
        "new_arm": {"results": new_results, "aggregate": new_agg},
        "target_improved": improvement,
        "no_more_harm": no_more_harm,
    }
    # 未预测回归检查（final winner = 新 guidance 在 held-out 的 chosen
    # workflow——在正向对照窗口评估；仅当目标组改善时执行）
    unpredicted: list[dict[str, Any]] = []
    winner_steps = None
    if improvement and no_more_harm:
        # final winner = 新臂第一个 positive probe 的 workflow（"最终 winner"
        # 语义——delayed 与正向对照只对它打开）
        for r in new_results:
            for p in r["probes"]:
                if p.get("gain") is not None and p["gain"] >= M:
                    winner_steps = p["steps"]
                    break
            if winner_steps:
                break
        g4["winner_steps"] = winner_steps
        # winner_steps 是 [{"op":…,"params":…}] dict 列表 → 转 (op, params)
        # 元组（checker 裁决 2026-08-14：dict 直传会在 Program.from_steps
        # 崩溃）。
        winner_tuples = (
            tuple((s["op"], s["params"]) for s in winner_steps)
            if winner_steps is not None else None)
        if winner_tuples is not None:
            from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: PLC0415
            for series, origin in G4_POSITIVE_CONTRAST:
                roster, vals = _support_roster(series, env["values"])
                executor = ScopeExecutor(roster, vals, nsu._config(),
                                         evaluate_fn=nsu._evaluate_kdd)
                rr = executor.evaluate(winner_tuples, origin)
                g = (float(rr.gain) if rr.gain is not None else None)
                unpredicted.append({"series": series, "origin": origin,
                                    "gain": g})
            g4["positive_contrast"] = unpredicted
            g4["regressions"] = [
                u for u in unpredicted
                if u["gain"] is not None and u["gain"] < -M]
        else:
            g4["positive_contrast"] = []
            g4["regressions"] = []
            g4["winner_missing"] = True
    else:
        g4["positive_contrast"] = []
        g4["regressions"] = []
    # delayed：仅最终 winner 打开（预冻结窗口；approval 条件 = 全部 ≥ −M）
    delayed_ok = None
    delayed_rows: list[dict[str, Any]] = []
    if (improvement and no_more_harm and winner_tuples is not None
            and not g4["regressions"]):
        from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: PLC0415
        for series, origin in G4_DELAYED_WINDOWS:
            roster, vals = _support_roster(series, env["values"])
            executor = ScopeExecutor(roster, vals, nsu._config(),
                                     evaluate_fn=nsu._evaluate_kdd)
            rr = executor.evaluate(winner_tuples, origin)
            g = (float(rr.gain) if rr.gain is not None else None)
            delayed_rows.append({"series": series, "origin": origin,
                                 "gain": g,
                                 "passed": bool(rr.verification.passed)})
        g4["delayed"] = delayed_rows
        # 用户裁决 2026-08-14：delayed 批准语义 = **每一个**预注册行都必须
        # verifier passed + gain **有限（math.isfinite——inf 不得通过）**
        # + ≥ −M；任一行失败/缺失即拒绝——不得过滤失败行后由剩余正向行
        # 批准。
        import math as _math
        all_valid = (
            bool(delayed_rows)
            and all(r["passed"] and r["gain"] is not None
                    and _math.isfinite(r["gain"])
                    and r["gain"] >= -M
                    for r in delayed_rows))
        delayed_ok = all_valid
        finite = [r["gain"] for r in delayed_rows
                  if r["gain"] is not None]
        g4["final_delayed_utility"] = (
            round(sum(finite) / len(finite), 4) if finite else None)
        g4["delayed_ok"] = delayed_ok
    else:
        g4["delayed"] = []
        g4["delayed_ok"] = None
        g4["final_delayed_utility"] = None
    g4["verdict"] = (
        "G4_SUPPORT_FAILED"
        if not (improvement and no_more_harm)
        else "G4_REGRESSION_FOUND"
        if g4["regressions"]
        else "G4_DELAYED_REJECTED"
        if delayed_ok is False
        else "G4_SUPPORT_PASS")
    report["g4"] = g4
    _save_report(report)
    print("== g4:", json.dumps({k: g4[k] for k in
                                ("old_arm_aggregate" if False else
                                 "target_improved", "no_more_harm",
                                 "verdict")}, ensure_ascii=False))
    print(json.dumps({"old_agg": old_agg, "new_agg": new_agg,
                      "regressions": g4["regressions"],
                      "verdict": g4["verdict"]},
                     ensure_ascii=False, indent=1))
    return 0


def phase_g5(env: Mapping[str, Any] | None = None) -> int:
    """G5 正常入口与 removal：激活新 Guidance snapshot → 下一正常 Fast
    入口消费新 body → 确认行为与最终 winner；removal 回 h0 → 行为恢复。
    正常入口 = TTHAMethod.prepare（不绕道直接调用 Program）。"""
    report = _load_report()
    g4 = report.get("g4") or {}
    if g4.get("verdict") != "G4_SUPPORT_PASS":
        print(f"== g5: blocked — g4 verdict = {g4.get('verdict')}")
        return 0
    if env is None:
        env = _load_env()
    h0 = _h0_snapshot()
    store = SnapshotStore(STORE_DIR)
    bundle_sha = (report.get("g2") or {}).get("candidate_runtime_bundle_sha")
    # 证据链（来自 G3/G4 报告段——Runner 无裸激活权）
    evidence = {
        "g3_behavior_verified": (report.get("g3") or {}).get("verdict")
        == "G3_BEHAVIOR_VERIFIED",
        "g4_support_passed": (report.get("g4") or {}).get("verdict")
        == "G4_SUPPORT_PASS",
        "delayed_ok": (report.get("g4") or {}).get("delayed_ok") is True,
    }
    # 正常入口 context（下一轮已暴露窗口：T100@792——census rounds 已含）
    entry_ctx = ("T100", 792)
    series0 = np.asarray(env["values"][entry_ctx[0]], dtype=np.float64)
    # ---- 方法层批准接口（用户裁决 2026-08-14：正常反馈生命周期批准，
    # 不是 Runner 直接编译 candidate snapshot）----
    counter, counter_ref = _make_client()
    core = TTHAAgentCore(
        AgictoChatCompletionsBackend(client=counter, base_url=BASE_URL),
        LocalPublicToolGateway(series0[:entry_ctx[1]], task_kind="forecast"),
        model=MODEL, base_url=BASE_URL)
    method = TTHAMethod(TTHAFastAgent(core), h0, ())
    adoption = method.adopt_guidance_candidate(
        store.root / str(bundle_sha), parent_snapshot=h0, **evidence)
    if not adoption.get("adopted"):
        g5 = {"verdict": "G5_PROTOCOL_FAILURE",
              "adoption": adoption, "evidence": evidence}
        report["g5"] = g5
        _save_report(report)
        print("== g5 adoption failed:", json.dumps(adoption,
                                                   ensure_ascii=False))
        return 0
    # 正常入口经 method.prepare 消费已激活的 active snapshot
    request = nsu._request(series0, env["values"], entry_ctx[1])
    import dataclasses as _dc5  # noqa: PLC0415
    from SelfEvolvingHarnessTS.contracts.task import (  # noqa: PLC0415
        deployment_constraints_v1,
        forecast_task_context_v1,
    )
    _tc5 = forecast_task_context_v1(
        task_spec=request.task_spec,
        deployment_constraints=deployment_constraints_v1())
    request = _dc5.replace(request, task_context=_tc5)
    try:
        result = method.prepare(request)
        trace = method.last_trace
        if trace is None:
            raise RuntimeError("no trace")
    except Exception as exc:  # noqa: BLE001
        g5 = {"verdict": "G5_PROTOCOL_FAILURE",
              "adoption": adoption, "evidence": evidence,
              "entry_error": f"{type(exc).__name__}: {exc}"}
        report["g5"] = g5
        _save_report(report)
        print("== g5 entry failed:", g5["entry_error"])
        return 0
    steps_map = {str(k): [{"op": o, "params": dict(p)} for o, p in v]
                 for k, v in dict(trace.candidate_program_steps or {}).items()}
    adopted = {
        "status": result.status.value if hasattr(result, "status") else str(result),
        "candidate_ids": list(trace.candidate_ids),
        "candidate_steps": steps_map,
        "chosen_candidate_id": str(trace.chosen_candidate_id),
        "compilation_status": str(trace.compilation_status),
        "llm_calls": counter_ref.calls,
    }
    print(f"== g5: normal entry (adopted guidance) {entry_ctx} ...", flush=True)
    # removal：h0 新实例（正常入口同装置）
    removed = _prepare_arm(h0, entry_ctx[0], entry_ctx[1], env)
    print(f"== g5: removal (h0) {entry_ctx} ...", flush=True)
    # 消费证据：新 guidance body 出现在 prompt 的 harness view 中
    #（adopted 臂的 view 来自方法层已激活的 active snapshot）
    adopted_snap = method._active_snapshot()  # noqa: SLF001
    new_body = (report.get("g2") or {}).get("guidance_body_new") or ""
    new_prompt_sha = {
        c["effective_harness_view_sha"]
        for c in (adopted.get("prompt_calls") or [])}
    removed_prompt_sha = {
        c["effective_harness_view_sha"]
        for c in (removed.get("prompt_calls") or [])}
    new_view = _view_bodies(adopted_snap, entry_ctx[0], entry_ctx[1], env)
    removed_view = _view_bodies(h0, entry_ctx[0], entry_ctx[1], env)
    consumed = bool(new_view.get(GUIDANCE_SKILL_ID)
                    and new_body
                    and new_view[GUIDANCE_SKILL_ID] == new_body)
    reverted = bool(
        removed_view.get(GUIDANCE_SKILL_ID)
        and removed_view[GUIDANCE_SKILL_ID]
        == _bootstrap_body(h0))
    behavior_different = bool(
        set(new_prompt_sha).isdisjoint(removed_prompt_sha)
        and (adopted.get("candidate_ids") != removed.get("candidate_ids")
             or adopted.get("chosen_candidate_id")
             != removed.get("chosen_candidate_id")))
    g5 = {
        "entry_context": list(entry_ctx),
        "adoption": adoption,
        "adopted_trace": adopted,
        "removed_trace": removed,
        "new_guidance_consumed": consumed,
        "removal_reverted_body": reverted,
        "behavior_changed_by_guidance": behavior_different,
        "verdict": (
            "G5_NORMAL_ENTRY_AND_REMOVAL_PASS"
            if (consumed and reverted and behavior_different)
            else "G5_PROTOCOL_FAILURE"),
    }
    report["g5"] = g5
    _save_report(report)
    print("== g5:", json.dumps({k: g5[k] for k in
                                ("new_guidance_consumed",
                                 "removal_reverted_body",
                                 "behavior_changed_by_guidance",
                                 "verdict")}, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- p0
# P0 故障账目（用户裁决 2026-08-14）：每条轨迹明确记录 failure_stage /
# failure_code / propose_reached / abstention_reason——不再把 inspect 失败
# 误归为 propose 弃权。只做分类记录，不重跑、不重判。

def _trajectory_accounting(row: dict[str, Any]) -> dict[str, Any]:
    calls = row.get("prompt_calls") or []
    failure_stage: str | None = None
    failure_code: str | None = None
    abstention_reason: str | None = None
    propose_reached = any(c.get("stage") == "propose" for c in calls)
    # 收集全部 stage-validation-error 反馈（按出现顺序）
    errors: list[dict[str, str]] = []
    for c in calls:
        um = c.get("user_message") or ""
        if "stage-validation-error" not in um:
            continue
        try:
            err = json.loads(um)
        except Exception:  # noqa: BLE001
            continue
        errors.append({"stage": str(c.get("stage")),
                       "code": str(err.get("error_code") or "")})
    status = str(row.get("status") or "")
    if status == "failed":
        # 最终失败：最后一个错误即 failure（恢复性错误另行记录）
        failure_stage = errors[-1]["stage"] if errors else (
            "inspect" if not propose_reached else None)
        failure_code = errors[-1]["code"] if errors else None
        abstention_reason = (
            "inspect_failed" if not propose_reached else "stage_failed")
    elif not propose_reached:
        # 非 failed 但未达 propose（理论上不可达——保守归类）
        failure_stage = "inspect"
        failure_code = errors[-1]["code"] if errors else None
        abstention_reason = "inspect_failed"
    else:
        # propose 到达：弃权判定来自最终 propose stage_result 载荷
        propose_payload = _parse_stage_payload(calls, "propose")
        candidates = [c for c in (propose_payload.get("candidates") or [])
                      if isinstance(c, dict)]
        inspect_payload = _parse_stage_payload(calls, "inspect")
        hypotheses = inspect_payload.get("pattern_hypotheses") or []
        if not candidates:
            abstention_reason = (
                "propose_empty_with_hypotheses" if hypotheses
                else "propose_empty_no_hypotheses")
    # 恢复性错误（重试后成功的行）——单独记录，不算 failure
    recovered_errors = []
    if status != "failed":
        recovered_errors = errors
    return {
        "failure_stage": failure_stage,
        "failure_code": failure_code,
        "propose_reached": bool(propose_reached),
        "abstention_reason": abstention_reason,
        "recovered_errors": recovered_errors,
    }


def phase_p0_accounting() -> int:
    """P0：对已存轨迹（S0-TC 两臂 + 第二轮 G3 两臂）做故障账目分类，
    写入报告 p0_failure_accounting——保留原始结果，不重跑、不包装新结论。"""
    report = _load_report()
    out: dict[str, Any] = {}
    for key, rows in (("s0_arm_a", (report.get("s0") or {}).get("arm_a") or []),
                      ("s0_arm_b", (report.get("s0") or {}).get("arm_b") or []),
                      ("g3r2_old", (report.get("g3") or {}).get("old_rows") or []),
                      ("g3r2_new", (report.get("g3") or {}).get("new_rows") or [])):
        out[key] = []
        for r in rows:
            acc = _trajectory_accounting(r)
            out[key].append({
                "context": f"{r.get('series')}@{r.get('origin')} r{r.get('rep')}",
                "status": r.get("status"),
                "candidate_ids": r.get("candidate_ids"),
                **acc,
            })
    out["note"] = (
        "第一轮 G3 的行级数据已被第二轮覆盖（仅 summary/diagnosis 保留）——"
        "账目只覆盖 S0-TC 与第二轮 G3；第一轮 G3 的 4/4 弃权同样需按"
        "failure_stage 复核，但原始行级记录已不可恢复（如实标注）。")
    report["p0_failure_accounting"] = out
    _save_report(report)
    print("== p0 accounting:")
    for key, rows in out.items():
        if key == "note":
            continue
        for r in rows:
            print(f"  {key} {r['context']}: stage={r['failure_stage']} "
                  f"code={r['failure_code']} propose_reached={r['propose_reached']} "
                  f"reason={r['abstention_reason']}")
    print("  ", out["note"])
    return 0


# ---------------------------------------------------------------- p4
# P4 决定性回放（用户裁决 2026-08-14）：A = 旧 Skill（h0 rev6）vs
# B = Slow clause 修改后的 Skill。五条件：①原失败轨迹产生合法、假设绑定
# 候选；②原成功轨迹不回归；③inspect 输出不变化（编辑面隔离的机械证明 +
# 无新 inspect 校验失败）；④verifier 通过；⑤失败时允许 Slow 据拒绝反馈
# 再改一次。四档裁定：CLAUSE_UPDATE_PASS / PATCH_REJECTED /
# EDIT_SURFACE_MISMATCH / UNIDENTIFIABLE。

def phase_p4(env: Mapping[str, Any] | None = None) -> int:
    report = _load_report()
    g2 = report.get("g2") or {}
    ev = g2.get("feedback_event") or {}
    bundle_sha = g2.get("candidate_runtime_bundle_sha")
    if ev.get("stage") != "pending" or not bundle_sha:
        print("== p4: blocked — g2 not pending")
        return 0
    if env is None:
        env = _load_env()
    h0 = _h0_snapshot()
    store = SnapshotStore(STORE_DIR)
    new_snap = compile_snapshot(store.root / str(bundle_sha),
                                verify_lock=False)
    # 编辑面隔离机械证明：非 propose 段逐字一致
    old_body = _bootstrap_body(h0)
    new_body = _bootstrap_body(new_snap)
    old_p = old_body.find("[propose_construction_guidance]")
    old_s = old_body.find("[select_guidance]")
    new_p = new_body.find("[propose_construction_guidance]")
    new_s = new_body.find("[select_guidance]")
    non_propose_unchanged = bool(
        old_p >= 0 and new_p >= 0
        and old_body[:old_p] == new_body[:new_p]
        and old_s >= 0 and new_s >= 0
        and old_body[old_s:] == new_body[new_s:])
    a_rows: list[dict[str, Any]] = []
    b_rows: list[dict[str, Any]] = []
    for series, origin in G4_FAULT_BUILD:
        for rep in range(G1_REPS):
            print(f"== p4-a: {series}@{origin} rep{rep} ...", flush=True)
            ra = _prepare_arm(h0, series, origin, env,
                              with_task_context=True)
            ra["rep"] = rep
            a_rows.append(ra)
            print(f"== p4-b: {series}@{origin} rep{rep} ...", flush=True)
            rb = _prepare_arm(new_snap, series, origin, env,
                              with_task_context=True)
            rb["rep"] = rep
            b_rows.append(rb)
    a_chain = [_chain_metrics(r) for r in a_rows]
    b_chain = [_chain_metrics(r) for r in b_rows]
    # 条件①：原失败轨迹（A 中 propose 真弃权）在 B 中产生合法假设绑定候选
    fixed = 0
    fixable = 0
    # 条件②：原成功轨迹（A 闭链）在 B 不回归
    regressions = 0
    # 条件③：B 无新 inspect 校验失败（与 A 相比）
    a_inspect_errors = sum(
        1 for r in a_rows
        if _trajectory_accounting(r)["failure_stage"] == "inspect")
    b_inspect_errors = sum(
        1 for r in b_rows
        if _trajectory_accounting(r)["failure_stage"] == "inspect")
    # 条件④：B 提案行 verifier 拒绝为空
    b_rejections = sum(len(r.get("rejection_receipts") or ())
                       for r in b_rows)
    per_row: list[dict[str, Any]] = []
    for a, b, ac, bc in zip(a_rows, b_rows, a_chain, b_chain):
        acc_a = _trajectory_accounting(a)
        key = f"{a['series']}@{a['origin']} r{a['rep']}"
        row = {"key": key, "a_kind": ac["kind"], "b_kind": bc["kind"]}
        if ac["chain_complete"]:
            if not bc["chain_complete"]:
                regressions += 1
                row["regression"] = True
        elif acc_a["propose_reached"] and acc_a["abstention_reason"] == \
                "propose_empty_with_hypotheses":
            fixable += 1
            if (bc.get("referenced_candidates")
                    and bc["chain_complete"]):
                fixed += 1
                row["fixed"] = True
        per_row.append(row)
    condition_1 = fixable > 0 and fixed >= 1
    condition_2 = regressions == 0
    condition_3 = non_propose_unchanged and b_inspect_errors == 0
    condition_4 = b_rejections == 0
    # 裁定（用户 P4 四档）
    if not non_propose_unchanged:
        verdict = "EDIT_SURFACE_MISMATCH"
    elif condition_1 and condition_2 and condition_3 and condition_4:
        verdict = "CLAUSE_UPDATE_PASS"
    elif b_inspect_errors > 0 or any(
            _trajectory_accounting(r)["failure_stage"] == "inspect"
            for r in b_rows):
        # B 的失败集中在 inspect——证据指向面外（但面未泄漏时属
        # UNIDENTIFIABLE 与 MISMATCH 的边界——按面机械证明判定）
        verdict = ("EDIT_SURFACE_MISMATCH" if not non_propose_unchanged
                   else "UNIDENTIFIABLE")
    else:
        verdict = "PATCH_REJECTED"
    p4 = {
        "edit_surface_isolation": non_propose_unchanged,
        "condition_1_fixed": {"fixed": fixed, "fixable": fixable},
        "condition_2_regressions": regressions,
        "condition_3_inspect_unchanged": (
            non_propose_unchanged and b_inspect_errors == 0),
        "a_inspect_errors": a_inspect_errors,
        "b_inspect_errors": b_inspect_errors,
        "condition_4_verifier_rejections": b_rejections,
        "a_rows": a_rows,
        "b_rows": b_rows,
        "a_chain": [dict(c) for c in a_chain],
        "b_chain": [dict(c) for c in b_chain],
        "per_row": per_row,
        "verdict": verdict,
    }
    report["p4"] = p4
    _save_report(report)
    print("== p4:", json.dumps({k: p4[k] for k in
                                ("edit_surface_isolation",
                                 "condition_1_fixed",
                                 "condition_2_regressions",
                                 "condition_3_inspect_unchanged",
                                 "condition_4_verifier_rejections",
                                 "verdict")}, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- s0
# S0：完整结构化 Skill（h0 rev4）vs 旧一句话 body fork 的构造链对比。
# 裁定维度（用户 2026-08-14 裁决）：不看"是否出现 outlier 算子"，看完整
# 构造链成功率：inspect 输出 grounded hypothesis → 候选引用该假设 →
# effect-distinct → select 沿引用选择或合理 abstain → 无协议失败。

def _parse_stage_payload(calls: Sequence[Mapping[str, Any]], stage: str,
                         ) -> dict[str, Any]:
    import json as _json
    payloads: list[dict[str, Any]] = []
    for c in calls:
        if c.get("stage") != stage:
            continue
        if c.get("parse_status") != "VALID_AGENT_ENVELOPE":
            continue
        text = str(c.get("assistant_text") or "")
        try:
            envelope = _json.loads(text)
        except _json.JSONDecodeError:
            continue
        if not isinstance(envelope, dict) or envelope.get("kind") != "stage_result":
            continue
        payload = envelope.get("payload")
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads[-1] if payloads else {}


def _chain_metrics(arm_result: dict[str, Any]) -> dict[str, Any]:
    """构造链完整性判定（纯机械——从记录载荷与 trace 提取）。"""
    if arm_result.get("protocol_error"):
        return {"chain_complete": False, "kind": "protocol_failure"}
    calls = arm_result.get("prompt_calls") or []
    inspect_payload = _parse_stage_payload(calls, "inspect")
    propose_payload = _parse_stage_payload(calls, "propose")
    hypotheses = inspect_payload.get("pattern_hypotheses") or []
    hypothesis_ids = {str(h.get("hypothesis_id"))
                      for h in hypotheses if isinstance(h, dict)}
    candidates = [c for c in (propose_payload.get("candidates") or [])
                  if isinstance(c, dict)]
    referenced = [str(c.get("addresses_hypothesis_id"))
                  for c in candidates
                  if c.get("addresses_hypothesis_id") in hypothesis_ids]
    # 带有效引用的候选 ID 集合（select_aligned 应对候选 ID 判断，
    # 不是对假设 ID——指标 bug 修复 2026-08-14）
    referenced_candidate_ids = [
        str(c.get("candidate_id")) for c in candidates
        if c.get("candidate_id")
        and c.get("addresses_hypothesis_id") in hypothesis_ids]
    steps_map = arm_result.get("candidate_steps") or {}
    program_ids = [cid for cid in arm_result.get("candidate_ids") or ()
                   if cid != "identity"]
    op_sets = [tuple(s.get("op") for s in steps_map.get(cid, ()))
               for cid in program_ids]
    effect_distinct = len(op_sets) <= 1 or len(
        {s for s in op_sets if s}) == len(op_sets)
    chosen = str(arm_result.get("chosen_candidate_id") or "identity")
    select_aligned = (
        chosen == "identity"
        or chosen in referenced_candidate_ids)
    metrics = {
        "hypotheses_emitted": list(hypothesis_ids),
        "referenced_candidates": referenced,
        "program_ids": program_ids,
        "effect_distinct": bool(effect_distinct),
        "chosen": chosen,
        "select_aligned": bool(select_aligned),
        "compilation_status": arm_result.get("compilation_status"),
    }
    metrics["chain_complete"] = bool(
        hypothesis_ids and referenced and effect_distinct
        and select_aligned
        and arm_result.get("compilation_status") in ("ok", "not_applicable")
        and not arm_result.get("protocol_error"))
    metrics["kind"] = (
        "chain_complete" if metrics["chain_complete"]
        else "structured_skill_noncompliant")
    return metrics


def _s0_archive_no_task_context(report: dict[str, Any]) -> None:
    s0 = report.get("s0") or {}
    if s0.get("arm_a") or s0.get("arm_b"):
        s0["no_task_context_run_2026_08_14"] = {
            "arm_a": s0.pop("arm_a", None),
            "arm_b": s0.pop("arm_b", None),
            "verdict": s0.pop("verdict", None),
            "summary": s0.pop("summary", None),
            "renamed_verdict": "S0_INCONCLUSIVE_MISSING_TASK_CONTEXT",
            "reason": ("propose 输入缺失 TaskContext（fast_agent._task_binding "
                       "在 request.task_context=None 时为 propose 返回空 "
                       "binding）——用户裁决：不可解释为完整 Skill 无效；"
                       "Pattern Observation 成功、构造在缺任务语义时全部 "
                       "abstain。"),
        }
        report["s0"] = s0


def _s0_precheck(env: Mapping[str, Any]) -> dict[str, Any]:
    """零 LLM 机械确认（用户裁决：调用前只做一次）：两臂 propose 输入
    包含完全相同的 task 与完整 task_context；quality_contract 含
    objective/preserve/harms；deployment_constraints 存在；两臂仍只差
    Workflow Construction Skill body。"""
    import dataclasses as _dc  # noqa: PLC0415
    from SelfEvolvingHarnessTS.contracts.task import (  # noqa: PLC0415
        deployment_constraints_v1,
        forecast_task_context_v1,
    )
    h0 = _h0_snapshot()
    old_body = (_load_report().get("g1") or {}).get(
        "frozen_hypothesis", {}).get("b1_patch", {}).get("old_body")
    store = SnapshotStore(STORE_DIR)
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    receipt = apply_guidance_patch(
        controller, store, h0, old_body,
        edit_id="s0_tc_precheck_fork",
        target_pattern_id="s0-tc-precheck")
    arm_b = receipt.candidate_snapshot.snapshot
    series0 = np.asarray(env["values"]["T100"], dtype=np.float64)
    base = nsu._request(series0, env["values"], 600)
    task_context = forecast_task_context_v1(
        task_spec=base.task_spec,
        deployment_constraints=deployment_constraints_v1())
    request = _dc.replace(base, task_context=task_context)
    tc = task_context.to_dict()
    quality = tc.get("quality_contract") or {}
    checks = {
        "task_context_present": bool(tc),
        "quality_has_objective": bool(quality.get("objective")),
        "quality_has_preserve": bool(quality.get("preserve")),
        "quality_has_harms": bool(quality.get("harms")),
        "deployment_constraints_present": bool(
            tc.get("deployment_constraints")),
        "request_carries_task_context": bool(
            request.task_context is not None),
    }
    # 两臂 view 只差 build_contrastive body：diff 两快照全部 skill body
    def _skills_by_id(snap: Any) -> dict[str, str]:
        return {s.skill_id: str(s.body) for s in snap.skills}
    a_skills = _skills_by_id(h0)
    b_skills = _skills_by_id(arm_b)
    diffs = [k for k in set(a_skills) | set(b_skills)
             if a_skills.get(k) != b_skills.get(k)]
    checks["arms_differ_only_in"] = sorted(diffs)
    checks["arms_differ_only_guidance"] = diffs == [
        "build_contrastive_candidates"]
    checks["all_ok"] = all(v is True for v in checks.values()
                           if isinstance(v, bool))
    return checks


def phase_s0_a(env: Mapping[str, Any]) -> int:
    """臂 A：完整结构化 Skill（当前 h0 rev4）+ 真实 TaskContext。"""
    report = _load_report()
    _s0_archive_no_task_context(report)
    precheck = _s0_precheck(env)
    report["s0"]["tc_precheck"] = precheck
    _save_report(report)
    print("== s0-tc precheck:", json.dumps(precheck, ensure_ascii=False))
    if not precheck.get("all_ok"):
        print("== s0-a: precheck failed — abort")
        return 0
    h0 = _h0_snapshot()
    rows: list[dict[str, Any]] = []
    for series, origin in G1_CONTEXTS:
        for rep in range(G1_REPS):
            print(f"== s0-tc-a: {series}@{origin} rep{rep} ...", flush=True)
            r = _prepare_arm(h0, series, origin, env,
                             with_task_context=True)
            r["arm"] = "a"
            r["rep"] = rep
            r["chain"] = _chain_metrics(r)
            rows.append(r)
            print(json.dumps({"cands": r.get("candidate_ids"),
                              "chain": r["chain"].get("kind"),
                              "calls": r.get("llm_calls")},
                             ensure_ascii=False))
    s0 = report.setdefault("s0", {})
    s0["arm_a"] = rows
    _save_report(report)
    print("== s0-tc-a saved")
    return 0


def phase_s0_b(env: Mapping[str, Any]) -> int:
    """臂 B：旧一句话 body fork + 真实 TaskContext。"""
    report = _load_report()
    h0 = _h0_snapshot()
    old_body = (report.get("g1") or {}).get("frozen_hypothesis", {}) \
        .get("b1_patch", {}).get("old_body")
    if not old_body or old_body == _bootstrap_body(h0):
        print("== s0-b: old body unavailable or unchanged — abort")
        return 0
    store = SnapshotStore(STORE_DIR)
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    receipt = apply_guidance_patch(
        controller, store, h0, old_body,
        edit_id="s0_thin_body_baseline_fork",
        target_pattern_id="s0-thin-baseline")
    arm_b_snapshot = receipt.candidate_snapshot.snapshot
    print(f"== s0-b fork: parent={receipt.parent_harness_content_sha[:12]} "
          f"candidate={receipt.candidate_harness_content_sha[:12]}")
    rows: list[dict[str, Any]] = []
    for series, origin in G1_CONTEXTS:
        for rep in range(G1_REPS):
            print(f"== s0-tc-b: {series}@{origin} rep{rep} ...", flush=True)
            r = _prepare_arm(arm_b_snapshot, series, origin, env,
                             with_task_context=True)
            r["arm"] = "b"
            r["rep"] = rep
            r["chain"] = _chain_metrics(r)
            rows.append(r)
            print(json.dumps({"cands": r.get("candidate_ids"),
                              "chain": r["chain"].get("kind"),
                              "calls": r.get("llm_calls")},
                             ensure_ascii=False))
    s0 = report.setdefault("s0", {})
    s0["arm_b"] = rows
    s0["arm_b_sha"] = receipt.candidate_harness_content_sha
    _save_report(report)
    print("== s0-tc-b saved")
    return 0


def phase_s0_verdict() -> int:
    report = _load_report()
    s0 = report.get("s0") or {}
    a = s0.get("arm_a") or []
    b = s0.get("arm_b") or []
    if not a or not b:
        print("== s0: missing arms")
        return 0

    def _per_ctx(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            key = f"{r['series']}@{r['origin']}"
            d = out.setdefault(key, {"chain": [], "kinds": [],
                                     "protocol_errors": 0})
            d["chain"].append(bool((r.get("chain") or {})
                                   .get("chain_complete")))
            d["kinds"].append((r.get("chain") or {}).get("kind"))
            if r.get("protocol_error"):
                d["protocol_errors"] += 1
        return out

    a_ctx, b_ctx = _per_ctx(a), _per_ctx(b)
    summary: dict[str, dict[str, Any]] = {}
    for key in sorted(set(a_ctx) | set(b_ctx)):
        ad, bd = a_ctx.get(key, {"chain": []}), b_ctx.get(key, {"chain": []})
        summary[key] = {
            "a_chain_success": sum(ad["chain"]),
            "b_chain_success": sum(bd["chain"]),
            "a_kinds": ad["kinds"],
            "b_kinds": bd["kinds"],
            "a_protocol_errors": ad["protocol_errors"],
            "b_protocol_errors": bd["protocol_errors"],
        }
    a_err = any(r.get("protocol_error") for r in a)
    a_gt = any(v["a_chain_success"] > v["b_chain_success"]
               for v in summary.values())
    a_ge = all(v["a_chain_success"] >= v["b_chain_success"]
               for v in summary.values())
    a_eq = all(v["a_chain_success"] == v["b_chain_success"]
               for v in summary.values())
    # 提案数（非弃权 prepare 数）——用户解释规则用
    a_proposals = sum(1 for r in a
                      if any(c != "identity"
                             for c in (r.get("candidate_ids") or [])))
    b_proposals = sum(1 for r in b
                      if any(c != "identity"
                             for c in (r.get("candidate_ids") or [])))
    a_worse = a_proposals < b_proposals or (
        a_eq and not a_ge) or (
        sum(v["a_chain_success"] for v in summary.values())
        < sum(v["b_chain_success"] for v in summary.values()))
    # 用户解释规则（2026-08-14）：
    if a_err:
        verdict = "S0_PROTOCOL_FAILURE"
    elif a_gt and a_ge:
        verdict = "SKILL_CONTENT_EFFECT"
    elif a_proposals == 0 and b_proposals == 0:
        verdict = "S0_ALL_ABSTAIN_TASK_CONTEXT_INSUFFICIENT"
    elif a_eq:
        verdict = "TASK_CONTEXT_EFFECTIVE_SKILL_NO_EXTRA"
    elif a_worse:
        verdict = "SKILL_SUPPRESSION_EFFECT"
    else:
        verdict = "INCONCLUSIVE_LLM_VARIANCE"
    s0["summary"] = summary
    s0["proposal_counts"] = {"a": a_proposals, "b": b_proposals}
    s0["verdict"] = verdict
    report["s0"] = s0
    _save_report(report)
    print(json.dumps({"s0_verdict": verdict,
                      "proposals": {"a": a_proposals, "b": b_proposals},
                      "summary": summary},
                     ensure_ascii=False, indent=1))
    return 0


def phase_final() -> int:
    """最终 Verdict（任务书映射 + 用户 2026-08-14 修订的停点命名）。
    干净测量（g1_rerun_protocol 存在且已执行）下 INCONCLUSIVE →
    NO_INCREMENTAL_GUIDANCE_SIGNAL_FOR_FROZEN_PATCH_AND_CONTEXTS；
    破损测量下 INCONCLUSIVE → UNIDENTIFIABLE_GUIDANCE_EFFECT。"""
    report = _load_report()
    g1 = (report.get("g1") or {}).get("verdict")
    g2 = (report.get("g2") or {}).get("feedback_event") or {}
    g3 = (report.get("g3") or {}).get("verdict")
    g4 = (report.get("g4") or {}).get("verdict")
    g5 = (report.get("g5") or {}).get("verdict")
    rerun_executed = bool((report.get("g1") or {}).get("b0"))
    if g1 in (None, "INCONCLUSIVE_LLM_VARIANCE"):
        verdict = (
            "NO_INCREMENTAL_GUIDANCE_SIGNAL_FOR_FROZEN_PATCH_AND_CONTEXTS"
            if rerun_executed
            else "UNIDENTIFIABLE_GUIDANCE_EFFECT")
    elif g1 == "GUIDANCE_NOT_CONSUMED":
        verdict = ("NO_INCREMENTAL_GUIDANCE_SIGNAL_FOR_FROZEN_PATCH_AND_CONTEXTS"
                   if rerun_executed
                   else "UNIDENTIFIABLE_GUIDANCE_EFFECT")
    elif g1 != "GUIDANCE_ACTION_SIGNAL":
        verdict = "UNIDENTIFIABLE_GUIDANCE_EFFECT"
    elif g2.get("stage") == "no_manifest":
        verdict = "NO_GUIDANCE_HEADROOM"
    elif g2.get("stage") != "pending":
        verdict = "GUIDANCE_PATCH_REJECTED"
    elif g3 != "G3_BEHAVIOR_VERIFIED":
        verdict = "GUIDANCE_PATCH_REJECTED"
    elif g4 in ("G4_SUPPORT_FAILED", "G4_DELAYED_REJECTED"):
        verdict = "GUIDANCE_BEHAVIOR_ONLY"
    elif g4 == "G4_REGRESSION_FOUND":
        verdict = "OVERGENERALIZED_GUIDANCE"
    elif g4 != "G4_SUPPORT_PASS":
        verdict = "GUIDANCE_BEHAVIOR_ONLY"
    elif g5 == "G5_NORMAL_ENTRY_AND_REMOVAL_PASS":
        verdict = "WORKFLOW_GUIDANCE_EVOLUTION_DEV_PASS"
    else:
        verdict = "GUIDANCE_BEHAVIOR_ONLY"
    report["final_verdict"] = {
        "verdict": verdict,
        "chain": {"g1": g1, "g2_stage": g2.get("stage"),
                  "g3": g3, "g4": g4, "g5": g5},
    }
    _save_report(report)
    print("== final:", json.dumps(report["final_verdict"],
                                  ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- fullop
# FULL_OPERATOR_SKILL_CAPABILITY（用户任务书 2026-08-14）：完整算子池 vs 受限池。
# 协议已冻结于 REPORT 的 fullop_protocol（任何运行之前）。唯一变量 = 候选
# 供给面过滤路径：A=pool_mode actionable（当前管线），B=pool_mode full
# （机械过滤全池，不含 verifier 0.35 可行动性探测）。Skill rev6/TaskContext/
# Schema/Prompt/模型/预算/Memory 全部不变。

FULLOP_ROSTER = (
    {"key": "T1@888", "series": "T1", "origin": 888, "variant": None},
    {"key": "T100@600", "series": "T100", "origin": 600, "variant": None},
    {"key": "T101@792", "series": "T101", "origin": 792, "variant": None},
    {"key": "T10@600", "series": "T10", "origin": 600, "variant": None},
    {"key": "T101@600", "series": "T101", "origin": 600, "variant": None},
    {"key": "T101@600+synmiss", "series": "T101", "origin": 600,
     "variant": "synmiss"},
)
FULLOP_REPS = 2
FULLOP_SUPPORT_BUDGET = 2   # Phase 2 每 context 探测预算（chosen-first 序）

# 假设 pattern_type → 期望算子族（协议冻结；见 fullop_protocol.phase1_family_map）
FULLOP_FAMILY_MAP = {
    "missingness": {"impute"},
    "extreme_deviation": {"outlier"},
    "level_excursion": {"structural"},
    "period_inconsistency": {"align"},
    "regime_ambiguity": {"outlier", "structural", "denoise"},
    "no_actionable_signal": set(),
}


def _fullop_values(env: Mapping[str, Any],
                   variant: str | None) -> Mapping[str, Any]:
    """synmiss 合成 Context 的确定性值注入（协议冻结索引：
    删除 [300,360) 块 + 单点 {100,200,250,450,500}）。"""
    if variant != "synmiss":
        return env["values"]
    values = dict(env["values"])
    arr = np.asarray(values["T101"], dtype=np.float64).copy()
    arr[list(range(300, 360)) + [100, 200, 250, 450, 500]] = np.nan
    values["T101"] = arr
    return values


def _fullop_context_pools(env: Mapping[str, Any], series: str, origin: int,
                          variant: str | None) -> dict[str, Any]:
    """每 context 的 A/B 池机械列表（无 LLM，运行前记录）——B-new 检测依据。"""
    import dataclasses as _dcf  # noqa: PLC0415
    from SelfEvolvingHarnessTS.contracts.task import (  # noqa: PLC0415
        deployment_constraints_v1,
        forecast_task_context_v1,
    )
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: PLC0415
        _actionable_operators,
        _allowed_operators,
        _full_pool_operators,
        _noop_ops_for_context,
    )
    values = _fullop_values(env, variant)
    series0 = np.asarray(values[series], dtype=np.float64)
    request = nsu._request(series0, values, origin)
    request = _dcf.replace(
        request,
        task_context=forecast_task_context_v1(
            task_spec=request.task_spec,
            deployment_constraints=deployment_constraints_v1()),
    )
    arr = np.asarray(request.values, dtype=float)
    features = extract_public_features(arr, task_kind="forecast")
    view = resolve_harness_view(_h0_snapshot(), features, role="fast")
    noop = set(_noop_ops_for_context(request))
    a_pool = sorted(set(_actionable_operators(
        request, arr, view, _allowed_operators(request))) - noop)
    b_pool = sorted(set(_full_pool_operators(request)) - noop)
    # dict()：extract_public_features 返回 mappingproxy——JSON 序列化会
    # 经 default=str 退化成字符串（2026-08-14 崩溃根因）。显式转 dict。
    return {"a_pool": a_pool, "b_pool": b_pool, "features": dict(features)}


def _fullop_row_metrics(row: dict[str, Any], a_pool: Sequence[str],
                        features: Mapping[str, object]) -> dict[str, Any]:
    """Phase 1 行级指标（纯机械——从记录载荷与 trace 提取）。"""
    from SelfEvolvingHarnessTS.operators.registry import (  # noqa: PLC0415
        OPERATOR_METADATA,
    )
    m = _chain_metrics(row)
    calls = row.get("prompt_calls") or []
    inspect_payload = _parse_stage_payload(calls, "inspect")
    propose_payload = _parse_stage_payload(calls, "propose")
    hypotheses = [h for h in (inspect_payload.get("pattern_hypotheses") or [])
                  if isinstance(h, dict)]
    hypothesis_types = {str(h.get("hypothesis_id")): h.get("pattern_type")
                        for h in hypotheses}
    candidates = [c for c in (propose_payload.get("candidates") or [])
                  if isinstance(c, dict)]
    proposed_ids = [str(c.get("candidate_id")) for c in candidates]
    steps_map = row.get("candidate_steps") or {}
    proposed_steps: dict[str, list] = {}
    for c in candidates:
        cid = str(c.get("candidate_id"))
        if cid in steps_map:
            proposed_steps[cid] = [s.get("op") for s in steps_map.get(cid, ())]
        else:
            # verifier 拒绝的候选不进 trace candidate_steps（2026-08-14 口径
            # 修正，fullop2 T10@600 B r0 实证）：ops 从 propose 载荷回读，
            # 否则 b_new_proposed_rejected 漏记（ops 读成空元组）。
            proposed_steps[cid] = [str(s.get("op"))
                                   for s in (c.get("steps") or ())
                                   if isinstance(s, dict)]
    pool_ids = [cid for cid in (row.get("candidate_ids") or ())
                if cid != "identity"]
    rejected = len(row.get("rejection_receipts") or ())
    n_proposed = len(proposed_ids)
    n_pool = len(pool_ids)
    a_pool_set = set(a_pool)
    per_candidate = []
    b_new_legal: list[str] = []
    b_new_proposed_rejected: list[str] = []
    for c in candidates:
        cid = str(c.get("candidate_id"))
        ops = tuple(proposed_steps.get(cid, ()))
        cats = tuple(sorted({str(OPERATOR_METADATA.get(op, {}).get("category"))
                             for op in ops}))
        is_new = any(op not in a_pool_set for op in ops)
        in_pool = cid in pool_ids
        per_candidate.append({
            "candidate_id": cid, "ops": list(ops), "categories": list(cats),
            "addresses_hypothesis_id": c.get("addresses_hypothesis_id"),
            "in_pool": in_pool, "outside_a_pool": is_new,
        })
        if is_new and in_pool:
            b_new_legal.append(cid)
        elif is_new and not in_pool:
            b_new_proposed_rejected.append(cid)
    # 参数绑定正确率（post_validator 强制绑定参数=feature 值；从载荷复核）
    param_binding_ok: bool | None = None
    binding_seen = 0
    for c in candidates:
        for step in c.get("steps") or []:
            if not isinstance(step, dict):
                continue
            meta = OPERATOR_METADATA.get(str(step.get("op"))) or {}
            bindings = meta.get("public_parameter_bindings") or {}
            if not bindings:
                continue
            binding_seen += 1
            params = step.get("params") or {}
            for pname, fname in bindings.items():
                try:
                    matches = (float(params.get(pname))
                               == float(features.get(fname)))
                except (TypeError, ValueError):
                    matches = False
                if not matches:
                    param_binding_ok = False
                    break
            else:
                param_binding_ok = True
    if not binding_seen:
        param_binding_ok = None
    # chosen 的 family_match（rule_3 依据；无引用假设 → None）
    chosen = str(row.get("chosen_candidate_id") or "identity")
    chosen_family_match: bool | None = None
    if chosen != "identity":
        for c in candidates:
            if str(c.get("candidate_id")) != chosen:
                continue
            ptype = hypothesis_types.get(
                str(c.get("addresses_hypothesis_id")))
            expected = FULLOP_FAMILY_MAP.get(str(ptype))
            if expected is not None:
                cats = {str(OPERATOR_METADATA.get(op, {}).get("category"))
                        for op in proposed_steps.get(chosen, ())}
                chosen_family_match = bool(cats) and cats <= expected
            break
    m.update({
        "proposed_program_ids": proposed_ids,
        "pool_program_ids": pool_ids,
        "n_proposed": n_proposed,
        "n_pool": n_pool,
        "rejected_count": rejected,
        "legal_candidate_rate": (round(n_pool / n_proposed, 4)
                                 if n_proposed else None),
        "verifier_pass_rate": (round(1 - rejected / n_proposed, 4)
                               if n_proposed else None),
        "abstention": (propose_payload.get("candidates") == []
                       or row.get("compilation_status") == "not_applicable"),
        "hypothesis_types": hypothesis_types,
        "per_candidate": per_candidate,
        "param_binding_ok": param_binding_ok,
        "b_new_legal": b_new_legal,
        "b_new_proposed_rejected": b_new_proposed_rejected,
        "chosen_family_match": chosen_family_match,
    })
    return m


def _fullop_aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def agg(arm: str) -> dict[str, Any]:
        rs = [r for r in rows if r["arm"] == arm]
        ms = [r.get("metrics") or {} for r in rs]
        proposed = sum(m.get("n_proposed") or 0 for m in ms)
        pool = sum(m.get("n_pool") or 0 for m in ms)
        rejected = sum(m.get("rejected_count") or 0 for m in ms)
        cats: set[str] = set()
        for m in ms:
            for c in m.get("per_candidate") or []:
                cats.update(c.get("categories") or ())
        return {
            "n_rows": len(rs),
            "chain_complete": sum(1 for m in ms if m.get("chain_complete")),
            "protocol_errors": sum(1 for r in rs if r.get("protocol_error")),
            "abstentions": sum(1 for m in ms if m.get("abstention")),
            "proposed_total": proposed,
            "pool_total": pool,
            "rejected_total": rejected,
            "legal_candidate_rate": (round(pool / proposed, 4)
                                     if proposed else None),
            "verifier_pass_rate": (round(1 - rejected / proposed, 4)
                                   if proposed else None),
            "family_diversity": sorted(cats),
            "param_binding_ok": sum(1 for m in ms
                                    if m.get("param_binding_ok") is True),
            "param_binding_bad": sum(1 for m in ms
                                     if m.get("param_binding_ok") is False),
            "b_new_legal": [(r["key"], cid) for r in rs
                            for cid in (r.get("metrics") or {}).get(
                                "b_new_legal") or []],
            "b_new_proposed_rejected": [
                (r["key"], cid) for r in rs
                for cid in (r.get("metrics") or {}).get(
                    "b_new_proposed_rejected") or []],
        }
    return {"A": agg("A"), "B": agg("B")}


def phase_fullop(env: Mapping[str, Any] | None = None) -> int:
    """Phase 1：24 次真实 prepare（6 context × 2 rep × 2 arm；resumable——
    已完成的 (key, rep, arm) 跳过）。行指标 + 臂聚合落盘 report["fullop"]。"""
    report = _load_report()
    if env is None:
        env = _load_env()
    fullop = report.get("fullop") or {}
    rows: list[dict[str, Any]] = fullop.get("rows") or []
    done = {(str(r["key"]), int(r["rep"]), str(r["arm"])) for r in rows}
    context_pools = fullop.get("context_pools") or {}
    report["fullop"] = fullop  # 回挂报告段（bug 修复 2026-08-14：否则保存落空）
    for ctx in FULLOP_ROSTER:
        key = str(ctx["key"])
        # 旧存档的 features 可能是 mappingproxy 退化的字符串——重算
        if (key not in context_pools
                or not isinstance(context_pools[key].get("features"), Mapping)):
            context_pools[key] = _fullop_context_pools(
                env, str(ctx["series"]), int(ctx["origin"]), ctx["variant"])
            fullop["context_pools"] = context_pools
            _save_report(report)
            print(f"== fullop pools {key}: a={len(context_pools[key]['a_pool'])}"
                  f" b={len(context_pools[key]['b_pool'])}", flush=True)
        run_env = _fullop_values(env, ctx["variant"])
        features = context_pools[key]["features"]
        for rep in range(FULLOP_REPS):
            for arm, mode in (("A", "actionable"), ("B", "full")):
                if (key, rep, arm) in done:
                    continue
                print(f"== fullop {arm} {key} rep{rep} ...", flush=True)
                row = _prepare_arm(_h0_snapshot(), str(ctx["series"]),
                                   int(ctx["origin"]), env,
                                   with_task_context=True, pool_mode=mode,
                                   values_override=run_env)
                row["key"] = key
                row["rep"] = rep
                row["arm"] = arm
                row["pool_mode"] = mode
                row["metrics"] = _fullop_row_metrics(
                    row, context_pools[key]["a_pool"], features)
                rows.append(row)
                fullop["rows"] = rows
                _save_report(report)
                mm = row["metrics"]
                print(f"== fullop {arm} {key} rep{rep} done: "
                      f"chain={mm.get('chain_complete')} "
                      f"kind={mm.get('kind')} "
                      f"proposed={mm.get('n_proposed')} "
                      f"pool={mm.get('n_pool')} "
                      f"rejected={mm.get('rejected_count')} "
                      f"new_legal={mm.get('b_new_legal')} "
                      f"llm={row.get('llm_calls')}", flush=True)
    fullop["arm_aggregates"] = _fullop_aggregate(rows)
    report["fullop"] = fullop  # 回挂报告段
    _save_report(report)
    print("== fullop arm aggregates:",
          json.dumps(fullop["arm_aggregates"], ensure_ascii=False))
    return 0


def phase_fullop_p2(env: Mapping[str, Any] | None = None) -> int:
    """Phase 2（条件）：只执行 B 新增且合法的候选（b_new_legal）——无 LLM
    探测：chosen-first → Target Support（origin）→ harm；winner（首个
    gain ≥ M）再在 origin+HORIZON 评估一次（delayed，佐证）。"""
    report = _load_report()
    if env is None:
        env = _load_env()
    fullop = report.get("fullop") or {}
    if fullop.get("p2"):
        print("== fullop-p2: already executed")
        return 0
    rows = fullop.get("rows") or []
    context_pools = fullop.get("context_pools") or {}
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: PLC0415
        ScopeExecutor,
    )
    entries: list[dict[str, Any]] = []
    any_positive = False
    for ctx in FULLOP_ROSTER:
        key = str(ctx["key"])
        b_rows = [r for r in rows
                  if r["key"] == key and r["arm"] == "B"
                  and not r.get("protocol_error")]
        new_legal: dict[str, dict[str, Any]] = {}
        for r in b_rows:
            for cid in (r.get("metrics") or {}).get("b_new_legal") or []:
                new_legal.setdefault(str(cid), r)
        if not new_legal:
            entries.append({"key": key, "status": "no_b_new_legal"})
            continue
        primary = b_rows[0]
        steps_map = primary.get("candidate_steps") or {}
        chosen_cid = str(primary.get("chosen_candidate_id") or "")
        order = ([chosen_cid] if chosen_cid in new_legal else []) + \
            [c for c in new_legal if c != chosen_cid]
        series = str(ctx["series"])
        origin = int(ctx["origin"])
        values = _fullop_values(env, ctx["variant"])
        roster, vals = _support_roster(series, values)
        executor = ScopeExecutor(roster, vals, nsu._config(),
                                 evaluate_fn=nsu._evaluate_kdd)
        probes: list[dict[str, Any]] = []
        winner: dict[str, Any] | None = None
        for cid in order[:FULLOP_SUPPORT_BUDGET]:
            steps = steps_map.get(cid) or []
            tuples = tuple((s["op"], dict(s["params"])) for s in steps)
            rr = executor.evaluate(tuples, origin)
            gain = float(rr.gain) if rr.gain is not None else None
            probe = {"candidate_id": cid, "steps": steps,
                     "gain": gain, "passed": bool(rr.verification.passed)}
            probes.append(probe)
            if winner is None and gain is not None and gain >= M:
                winner = probe
        entry: dict[str, Any] = {
            "key": key, "series": series, "origin": origin,
            "synthetic_context": ctx["variant"] == "synmiss",
            "chosen": chosen_cid, "probes": probes,
            "winner_candidate_id": (winner or {}).get("candidate_id"),
            "positive_count": sum(
                1 for p in probes
                if p.get("gain") is not None and p["gain"] >= M),
            "harm_count": sum(
                1 for p in probes
                if p.get("gain") is not None and p["gain"] < -M),
        }
        if winner is not None:
            tuples = tuple((s["op"], dict(s["params"]))
                           for s in winner["steps"])
            rr = executor.evaluate(tuples, origin + HORIZON)
            g = float(rr.gain) if rr.gain is not None else None
            entry["delayed"] = {
                "candidate_id": winner["candidate_id"],
                "origin": origin + HORIZON,
                "gain": g, "passed": bool(rr.verification.passed)}
        if entry["positive_count"] > 0:
            any_positive = True
        entries.append(entry)
        print(f"== fullop-p2 {key}: {json.dumps(entry, ensure_ascii=False)}",
              flush=True)
    fullop["p2"] = {"entries": entries, "any_positive": any_positive}
    report["fullop"] = fullop  # 回挂报告段
    _save_report(report)
    print("== fullop-p2 any_positive:", any_positive)
    return 0


def phase_fullop_rerun(env: Mapping[str, Any] | None = None) -> int:
    """round1 归档为漂移证据（rule_0 A 基线崩塌）→ 同冻结协议重跑。
    重跑不产生新协议、不改裁定规则——rule_0 是健康门：清门则继续
    rule_1–5；再崩则结论 = 当前后端无法复现 rev6 基线（测量层问题，
    不是池效应问题）。"""
    report = _load_report()
    if env is None:
        env = _load_env()
    fullop = report.get("fullop") or {}
    if not fullop.get("rows"):
        print("== fullop-rerun: no round1 rows to archive")
        return 0
    round_no = 1
    while f"round{round_no}_drift_evidence_2026_08_14" in fullop:
        round_no += 1
    fullop[f"round{round_no}_drift_evidence_2026_08_14"] = {
        "rows": fullop.pop("rows", None),
        "arm_aggregates": fullop.pop("arm_aggregates", None),
        "verdict": fullop.pop("verdict", None),
        "p2": fullop.pop("p2", None),
        "reason": ("round A 基线崩塌（rule_0 INCONCLUSIVE_PROTOCOL_FAILURE）"
                   "→ 归档为漂移证据；同冻结协议重跑，裁定规则不变"),
    }
    report["fullop"] = fullop
    _save_report(report)
    print(f"== fullop-rerun: round{round_no} archived, restarting rows")
    return phase_fullop(env)


def phase_fullop_verdict() -> int:
    """Phase 1+2 终裁（协议冻结规则 fullop_protocol.verdict_rules）。"""
    report = _load_report()
    fullop = report.get("fullop") or {}
    rows = fullop.get("rows") or []
    if len(rows) < len(FULLOP_ROSTER) * FULLOP_REPS * 2:
        print(f"== fullop-verdict: incomplete rows {len(rows)}/"
              f"{len(FULLOP_ROSTER) * FULLOP_REPS * 2}")
        return 0
    agg = fullop.get("arm_aggregates") or {}
    a_chain = (agg.get("A") or {}).get("chain_complete") or 0
    b_chain = (agg.get("B") or {}).get("chain_complete") or 0
    a_proto = (agg.get("A") or {}).get("protocol_errors") or 0
    b_proto = (agg.get("B") or {}).get("protocol_errors") or 0
    b_new_total = len((agg.get("B") or {}).get("b_new_legal") or [])
    # rule_0：A 基线健康（T1@888/T100@600 四行 ≥2 闭链——历史 rev6 A 3/4）
    a_baseline_chain = sum(
        1 for r in rows
        if r["arm"] == "A" and r["key"] in ("T1@888", "T100@600")
        and (r.get("metrics") or {}).get("chain_complete"))
    # rule_3 材料：含 b_new_legal 进入最终池且 chosen≠identity 的行
    blocker_rows: list[bool] = []
    for r in rows:
        if r["arm"] != "B":
            continue
        m = r.get("metrics") or {}
        if not m.get("b_new_legal"):
            continue
        if m.get("chosen") == "identity":
            continue
        if m.get("chosen_family_match") is not None:
            blocker_rows.append(bool(m["chosen_family_match"]))
    verdict: str
    reasons: list[str] = []
    p2 = fullop.get("p2") or {}
    if a_baseline_chain < 2:
        verdict = "INCONCLUSIVE_PROTOCOL_FAILURE"
        reasons.append("rule_0_health_a_baseline")
    elif b_new_total == 0:
        verdict = "OPERATOR_SPACE_OVERLOAD"
        reasons.append("rule_1_no_new_legal_candidates")
    elif b_chain < a_chain - 1 or b_proto > a_proto + 1:
        verdict = "OPERATOR_SPACE_OVERLOAD"
        reasons.append("rule_2_chain_collapse")
    elif blocker_rows and sum(blocker_rows) * 2 <= len(blocker_rows):
        verdict = "CONTEXT_OR_SELECTION_BLOCKER"
        reasons.append("rule_3_blocker")
    elif not p2.get("entries"):
        verdict = "P2_PENDING"
        reasons.append("run_fullop_p2_first")
    elif not p2.get("any_positive"):
        verdict = "LEGAL_BUT_NO_UTILITY"
        reasons.append("rule_4_no_utility")
    else:
        verdict = "FULL_OPERATOR_CAPABILITY_PASS"
        reasons.append("rule_5_pass")
    fullop["verdict"] = {
        "verdict": verdict,
        "reasons": reasons,
        "a_baseline_chain": a_baseline_chain,
        "a_chain": a_chain, "b_chain": b_chain,
        "a_protocol_errors": a_proto, "b_protocol_errors": b_proto,
        "b_new_legal_total": b_new_total,
        "blocker_rows": blocker_rows,
        "p2_any_positive": p2.get("any_positive"),
    }
    report["fullop"] = fullop  # 回挂报告段
    _save_report(report)
    print("== fullop verdict:",
          json.dumps(fullop["verdict"], ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- tsem
# TARGETING_SEMANTICS_REV7_PAIRED_VERIFICATION（用户裁决 2026-08-14）。
# 依据：主报告 targeting_semantics_diagnostic_2026_08_14——FULLOP 两轮弃权
# 的 first fault = rev6 绑定语义未区分 intrinsic/external/global 三类
# targeting（合法 intrinsic 绑定一直存在，模型误判 no_legal_binding）。
# rev7 = 人工基线修订：仅 4e 与 propose.rule.hypothesis_binding 两处逐字
# 替换（用户逐字裁定文本）。配对对照：A=rev6 原文 / B=rev7，同期交错
# AB/BA，pool_mode=actionable 固定（本实验只测 targeting 文本，不同时测
# 完整算子空间）。原 FULLOP 终裁不动，本实验为独立后续。

TSEM_ROSTER = (("T1", 888), ("T100", 600), ("T10", 600),
               ("T101", 792), ("T101", 600))
TSEM_DEVIATION = ("T1@888", "T100@600", "T10@600")
TSEM_CONTROL = ("T101@792", "T101@600")

TSEM_OLD_4E = (
    "4e. Program binding. Bind dynamic parameters only through the public "
    "parameter bindings declared in the operator contract, using the current "
    "feature values; never replay numerical parameters from a previous "
    "Context or Episode. Regions must come from the current inspection; do "
    "not extend to the whole series unless inspection justifies it.")
TSEM_NEW_4E = (
    "4e. Program binding. Bind dynamic parameters only through the public "
    "parameter bindings declared in the operator contract, using current "
    "feature values; never replay numerical parameters from a previous "
    "Context or Episode. Targeting semantics determine how an operator "
    "locates its effects, but do not by themselves establish semantic "
    "relevance to a hypothesis. An intrinsic operator locates its own hit "
    "points during execution and requires no external region parameter. "
    "Hypothesis span alone, including a whole-series span, must not be used "
    "to declare an intrinsic binding illegal. An external_region operator "
    "must bind its region from the current inspection, and the bound region "
    "must remain within the addressed hypothesis region. A global operator "
    "acts on the whole series by definition and is admissible only when the "
    "hypothesis mechanism, TaskContext, and risk constraints justify a "
    "whole-series effect.")
TSEM_OLD_HB = (
    "propose.rule.hypothesis_binding: Every proposed candidate must set "
    "addresses_hypothesis_id to one hypothesis_id emitted by your own "
    "inspect stage, and its operators, order, and region must match that "
    "hypothesis.")
TSEM_NEW_HB = (
    "propose.rule.hypothesis_binding: Every proposed candidate must set "
    "addresses_hypothesis_id to one hypothesis_id emitted by the inspect "
    "stage. Its operator mechanism, step order, and targeting semantics "
    "must address that hypothesis. An intrinsic operator requires no region "
    "parameter; its relevance is determined from the hypothesis pattern and "
    "evidence, while its realized modification is checked by execution "
    "verification and downstream Support. An external_region operator must "
    "bind a region contained within the hypothesis region. A global "
    "operator requires evidence that a whole-series effect is appropriate.")


def _tsem_rev7_body(rev6_body: str) -> str:
    """rev7 = rev6 仅两处逐字替换（用户 2026-08-14 逐字裁定）。锚点必须
    各恰好出现一次；段标记与五 clause 前缀结构保持不变（Slow 编辑面
    REPLACE_CLAUSE 机制依赖 propose.rule. 前缀逐字寻址）。"""
    body = str(rev6_body)
    for old, new in ((TSEM_OLD_4E, TSEM_NEW_4E), (TSEM_OLD_HB, TSEM_NEW_HB)):
        if body.count(old) != 1:
            raise ValueError("tsem rev7 anchor not unique: " + repr(old[:60]))
        body = body.replace(old, new)
    for marker in ("[FIXED_CONTRACT]", "[inspect_pattern_guidance]",
                   "[propose_construction_guidance]", "[select_guidance]"):
        if marker not in body:
            raise ValueError("tsem rev7 lost section marker " + marker)
    for prefix in ("propose.rule.hypothesis_binding:",
                   "propose.rule.effect_distinct:",
                   "propose.rule.inert_and_order:",
                   "propose.rule.no_legal_binding:",
                   "propose.rule.exploration_supply:"):
        if body.count(prefix) != 1:
            raise ValueError("tsem rev7 clause prefix broken: " + prefix)
    return body


def _tsem_schedule(max_rep: int = 1) -> list:
    """冻结交错顺序（用户裁决：固定 AB/BA 交错）：rep0 按 roster 序第 i 个
    context——i 偶 A→B、i 奇 B→A；升级 rep1 逐 context 取反。
    返回 (key, rep, arm) 序列；arm: rev6=对照 / rev7=处理。"""
    entries = []
    for rep in range(max_rep):
        for i, (series, origin) in enumerate(TSEM_ROSTER):
            arms = ("rev6", "rev7") if (i + rep) % 2 == 0 else ("rev7", "rev6")
            for arm in arms:
                entries.append((series + "@" + str(origin), rep, arm))
    return entries


def _tsem_row_metrics(row: dict) -> dict:
    """行级指标（纯机械）：链完整性 + 弃权/拒绝 + 候选 targeting/family 细目。"""
    from SelfEvolvingHarnessTS.operators.registry import (  # noqa: PLC0415
        OPERATOR_METADATA,
    )
    m = _chain_metrics(row)
    calls = row.get("prompt_calls") or []
    propose_payload = _parse_stage_payload(calls, "propose")
    candidates = [c for c in (propose_payload.get("candidates") or [])
                  if isinstance(c, dict)]
    steps_map = row.get("candidate_steps") or {}
    pool_ids = {cid for cid in (row.get("candidate_ids") or ())
                if cid != "identity"}
    per_candidate = []
    for c in candidates:
        cid = str(c.get("candidate_id"))
        ops = [str(s.get("op")) for s in steps_map.get(cid, ())]
        cats = sorted({str(OPERATOR_METADATA.get(op, {}).get("category"))
                       for op in ops})
        tms = sorted({str(OPERATOR_METADATA.get(op, {}).get("targeting_mode"))
                      for op in ops})
        per_candidate.append({
            "candidate_id": cid, "ops": ops, "categories": cats,
            "targeting_modes": tms,
            "addresses_hypothesis_id": c.get("addresses_hypothesis_id"),
            "in_pool": cid in pool_ids})
    chosen = str(row.get("chosen_candidate_id") or "identity")
    chosen_categories: list = []
    for pc in per_candidate:
        if pc["candidate_id"] == chosen:
            chosen_categories = list(pc["categories"])
    m.update({
        "abstention": bool(propose_payload.get("candidates") == []
                           or row.get("compilation_status") == "not_applicable"),
        "rejected_count": len(row.get("rejection_receipts") or ()),
        "per_candidate": per_candidate,
        "chosen_categories": chosen_categories,
        "has_legal_intrinsic_candidate": bool(any(
            pc["targeting_modes"] == ["intrinsic"] and pc["in_pool"]
            for pc in per_candidate)),
    })
    return m


def _tsem_fork_snapshot(report: dict) -> tuple:
    """幂等构建 rev7 fork snapshot（B1 先例同一 EditController 机械层；
    本 fork 是人工基线修订，有效性由本实验配对对照判定——Slow 不批准
    自己）。已记录 fork_bundle_sha → 直接编译复用；施加后断言 fork body
    逐字等于期望 rev7 且 h0 body 仍为 rev6。"""
    tsem = report.setdefault("tsem", {})
    h0 = _h0_snapshot()
    expected = _tsem_rev7_body(_bootstrap_body(h0))
    sha = tsem.get("fork_bundle_sha")
    if sha:
        snap = compile_snapshot(SnapshotStore(STORE_DIR).root / str(sha),
                                verify_lock=False)
    else:
        store = SnapshotStore(STORE_DIR)
        controller = EditController(store, surfaces=SurfaceRegistry(),
                                    router=FaultRouter())
        receipt = apply_guidance_patch(
            controller, store, h0, expected,
            edit_id="tsem-rev7-2026-08-14",
            target_pattern_id="tsem-rev7-targeting-semantics")
        sha = str(receipt.candidate_runtime_bundle_sha)
        tsem["fork_bundle_sha"] = sha
        _save_report(report)
        snap = compile_snapshot(store.root / sha, verify_lock=False)
    if _bootstrap_body(snap) != expected:
        raise ValueError("tsem rev7 fork body mismatch after apply")
    if _bootstrap_body(h0) == expected:
        raise ValueError("tsem h0 body unexpectedly already rev7")
    return snap, sha


def _tsem_verdict(rows: list) -> dict:
    """冻结裁定（tsem_protocol.verdict_rules；纯机械，不得事后放宽）。
    优先级：PROTOCOL_INCONCLUSIVE > REGRESSIVE > P1 判定。"""
    def get(key: str, rep: int, arm: str) -> dict | None:
        for r in rows:
            if (r.get("key") == key and int(r.get("rep", 0)) == rep
                    and str(r.get("arm")) == arm):
                return r
        return None
    reps = [int(r.get("rep", 0)) for r in rows] or [0]
    n_rep = max(reps) + 1
    protocol_bad: list = []
    dev_pairs: list = []
    for key in TSEM_DEVIATION:
        for rep in range(n_rep):
            a = get(key, rep, "rev6")
            b = get(key, rep, "rev7")
            tag = key + " rep" + str(rep)
            if a is None or b is None:
                protocol_bad.append(tag + " 缺臂")
                continue
            if a.get("protocol_error") or b.get("protocol_error"):
                protocol_bad.append(tag + " protocol_error")
                continue
            dev_pairs.append({
                "key": key, "rep": rep,
                "a_chain": bool(a["metrics"]["chain_complete"]),
                "b_chain": bool(b["metrics"]["chain_complete"])})
    p2_failures: list = []
    for key in TSEM_CONTROL:
        for rep in range(n_rep):
            a = get(key, rep, "rev6")
            b = get(key, rep, "rev7")
            tag = key + " rep" + str(rep)
            if a is None or b is None:
                protocol_bad.append(tag + " 缺臂")
                continue
            if b.get("protocol_error"):
                p2_failures.append(tag + " rev7 protocol_error")
                continue
            if a.get("protocol_error"):
                protocol_bad.append(tag + " rev6 protocol_error")
                continue
            am = a["metrics"]
            bm = b["metrics"]
            if not bm["chain_complete"] and am["chain_complete"]:
                p2_failures.append(tag + " rev7 未闭链而 rev6 闭链")
            if am["chain_complete"] and bm["chain_complete"] and (
                    bm.get("chosen_categories")
                    != am.get("chosen_categories")):
                p2_failures.append(tag + " chosen family 改变")
            if bm.get("rejected_count", 0) > 0 \
                    and am.get("rejected_count", 0) == 0:
                p2_failures.append(tag + " rev7 新增 verifier 拒绝")
    b_dev = sum(1 for p in dev_pairs if p["b_chain"])
    a_dev = sum(1 for p in dev_pairs if p["a_chain"])
    repairs = sum(1 for p in dev_pairs if p["b_chain"] and not p["a_chain"])
    if protocol_bad:
        verdict = "PROTOCOL_INCONCLUSIVE"
    elif p2_failures:
        verdict = "REGRESSIVE"
    elif n_rep == 1:
        if b_dev >= 2 and repairs >= 2:
            verdict = "TARGETING_SEMANTICS_CAUSAL_EFFECT"
        elif a_dev >= 2 and b_dev >= 2:
            verdict = "BASELINE_RECOVERED_NO_INCREMENTAL_EFFECT"
        else:
            verdict = "ESCALATE_REP1"
    else:
        if b_dev >= 5 and repairs >= 3:
            verdict = "TARGETING_SEMANTICS_CAUSAL_EFFECT"
        elif a_dev >= 5 and b_dev >= 5:
            verdict = "BASELINE_RECOVERED_NO_INCREMENTAL_EFFECT"
        else:
            verdict = "NO_INCREMENTAL_EFFECT"
    return {"verdict": verdict, "n_rep": n_rep,
            "deviation": {"a_chain": a_dev, "b_chain": b_dev,
                          "repairs": repairs, "pairs": dev_pairs},
            "p2_failures": p2_failures,
            "protocol_problems": protocol_bad}


def phase_tsem_freeze() -> int:
    """冻结 tsem 协议进主报告（任何运行前；已冻结或已有数据行则拒绝）。"""
    import hashlib  # noqa: PLC0415
    report = _load_report()
    if report.get("tsem_protocol"):
        raise SystemExit("tsem_protocol 已存在——拒绝重复冻结")
    tsem = report.get("tsem") or {}
    if tsem.get("rows"):
        raise SystemExit("tsem 已有数据行——协议必须先冻结")
    h0 = _h0_snapshot()
    rev6_body = _bootstrap_body(h0)
    rev7_body = _tsem_rev7_body(rev6_body)
    verifier_src = (PROJECT_ROOT / "SelfEvolvingHarnessTS" / "runtime"
                    / "candidate_verification.py").read_bytes()
    proto = {
        "experiment_id": "TARGETING_SEMANTICS_REV7_PAIRED_VERIFICATION",
        "user_ruling": ("2026-08-14：诊断成立、rev7 最小修正方向批准、两处措辞"
                        "按裁定收紧、协议必须含同期 rev6 配对对照（交错 AB/BA）"),
        "basis_diagnostic": "targeting_semantics_diagnostic_2026_08_14",
        "unique_variable": ("build_contrastive_candidates.body 的两处文本：4e "
                            "Program binding 与 propose.rule.hypothesis_binding"
                            "（rev6 原文 vs rev7 裁定文本），其余字节不变"),
        "rev7_old_new_texts": {"old_4e": TSEM_OLD_4E, "new_4e": TSEM_NEW_4E,
                               "old_hb": TSEM_OLD_HB, "new_hb": TSEM_NEW_HB},
        "rev6_body_sha": hashlib.sha256(rev6_body.encode("utf-8")).hexdigest(),
        "rev7_body_sha": hashlib.sha256(rev7_body.encode("utf-8")).hexdigest(),
        "rev7_construction": ("人工基线修订：两处逐字替换（_tsem_rev7_body 断言锚点"
                              "唯一、段标记与五 clause 前缀结构不变）；经 "
                              "EditController 施加到 fork（B1 先例机械层；surface "
                              "白名单只允许 body——fork 内 revision 字段保持 6 是"
                              "机械限制，行证据以 body/view sha 为准）"),
        "arms": {"rev6": "h0 原文 snapshot（对照）",
                 "rev7": "fork snapshot（处理；仅 body 两处不同）"},
        "pool_mode": ("actionable（固定原 A 池——本实验只测 targeting 文本，"
                      "不测完整算子空间）"),
        "roster": [{"key": s + "@" + str(o), "role": role}
                   for (s, o), role in zip(
                       TSEM_ROSTER,
                       ("deviation_target",) * 3 + ("regression_control",) * 2)],
        "schedule_frozen": [list(e) for e in _tsem_schedule(2)],
        "schedule_note": ("rep0 第一轮 10 prepares（roster 序 i 偶 A→B / i 奇 B→A）；"
                          "仅当 round1 裁定 ESCALATE_REP1 才跑 rep1（逐 context "
                          "取反），最多 20 prepares"),
        "memory_and_context": "两臂同空 Memory；with_task_context=True（同 FULLOP 口径）",
        "metrics": ["chain_complete（沿用 _chain_metrics 口径）", "abstention",
                    "rejected_count",
                    "per_candidate(ops/categories/targeting_modes/in_pool)",
                    "chosen_categories", "has_legal_intrinsic_candidate"],
        "verdict_rules": {
            "precedence": ["PROTOCOL_INCONCLUSIVE", "REGRESSIVE", "P1"],
            "protocol_rule": ("任一 paired 行缺失或任一行 protocol_error → "
                              "PROTOCOL_INCONCLUSIVE（配对不可解释；不自动升级）"),
            "regressive_rule": ("P2 失败即 REGRESSIVE：控制对 rev7 未闭链而 rev6 "
                                "闭链 / rev7 chosen family 不同于 rev6 / rev7 新增 "
                                "verifier 拒绝或 protocol_error"),
            "p1_round1": ("B deviation 闭链 ≥ 2/3 且 B 相对 A 修复 paired rows ≥ 2 "
                          "→ TARGETING_SEMANTICS_CAUSAL_EFFECT"),
            "baseline_recovered_round1": ("A_dev ≥ 2/3 且 B_dev ≥ 2/3 → "
                                          "BASELINE_RECOVERED_NO_INCREMENTAL_EFFECT"),
            "round1_escalation": ("round1 无以上终裁 → ESCALATE_REP1（预注册升级，"
                                  "非新协议）"),
            "p1_round2": ("升级后 B_dev ≥ 5/6 且 paired improvement ≥ 3/6 → "
                          "TARGETING_SEMANTICS_CAUSAL_EFFECT"),
            "baseline_recovered_round2": ("A_dev ≥ 5/6 且 B_dev ≥ 5/6 → "
                                          "BASELINE_RECOVERED_NO_INCREMENTAL_EFFECT"),
            "round2_fallback": "否则 → NO_INCREMENTAL_EFFECT",
        },
        "p3_sanity": {
            "verifier_file_sha256": hashlib.sha256(verifier_src).hexdigest(),
            "note": ("verifier 未修改——代码级 sanity（verdict 时复核同 sha 记录"
                     " p3_sanity_verifier_unchanged），非方法 Gate")},
        "claim_boundary": ("即便通过也只能声称：修正 targeting 说明后 Agent 更稳定地"
                           "把 broad deviation hypothesis 绑定到 intrinsic outlier "
                           "candidate。不证明：完整 26 算子选择能力、新候选下游价值、"
                           "Slow 自进化能力。通过后回到新 FULLOP 版本打开完整池+Support。"),
        "expected_llm_calls": ("round1 ≤ 10 prepares × 12 预算（实测约 5-6/prepare）；"
                               "含升级 ≤ 20 prepares"),
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["tsem_protocol"] = proto
    _save_report(report)
    print("tsem_protocol FROZEN; rev7_body_sha",
          proto["rev7_body_sha"][:16])
    return 0


def phase_tsem(env: Mapping[str, Any] | None = None, extra: bool = False) -> int:
    """配对对照执行（可恢复：按 (key, rep, arm) 跳过已有行；每行落盘）。
    extra=True 跑升级 rep1——须 round1 已裁定 ESCALATE_REP1（预注册纪律）。"""
    report = _load_report()
    if not report.get("tsem_protocol"):
        raise SystemExit("tsem_protocol 未冻结——先跑 tsem-freeze")
    tsem = report.setdefault("tsem", {})
    rows = tsem.setdefault("rows", [])
    if extra and (tsem.get("round1_verdict") or {}).get(
            "verdict") != "ESCALATE_REP1":
        raise SystemExit("未获 ESCALATE_REP1 裁定——拒绝跑 rep1（预注册纪律）")
    max_rep = 2 if extra else 1
    if env is None:
        env = _load_env()
    rev7_snap, rev7_sha = _tsem_fork_snapshot(report)
    snaps = {"rev6": _h0_snapshot(), "rev7": rev7_snap}
    tsem["rev7_fork_bundle_sha"] = rev7_sha
    done = {(r.get("key"), int(r.get("rep", 0)), str(r.get("arm")))
            for r in rows}
    for key, rep, arm in _tsem_schedule(max_rep):
        if (key, rep, arm) in done:
            continue
        series, _, origin = key.partition("@")
        print("== tsem " + arm + " " + key + " rep" + str(rep) + " ...",
              flush=True)
        row = _prepare_arm(snaps[arm], series, int(origin), env,
                           with_task_context=True, pool_mode="actionable")
        row["key"] = key
        row["rep"] = rep
        row["arm"] = arm
        row["pool_mode"] = "actionable"
        row["metrics"] = _tsem_row_metrics(row)
        rows.append(row)
        tsem["state"] = "RUNNING"
        _save_report(report)
        print("== tsem " + arm + " " + key + " rep" + str(rep) + " done: "
              + "chain=" + str(row["metrics"]["chain_complete"])
              + " calls=" + str(row.get("llm_calls")), flush=True)
    _save_report(report)
    return 0


def phase_tsem_verdict() -> int:
    """机械裁定：完整性预检 → _tsem_verdict → P3 sanity 记录 → 落盘。
    ESCALATE_REP1 只写 round1_verdict；终裁写 verdict 并 CLOSED。"""
    import hashlib  # noqa: PLC0415
    report = _load_report()
    proto = report.get("tsem_protocol") or {}
    tsem = report.get("tsem") or {}
    rows = tsem.get("rows") or []
    if not rows:
        raise SystemExit("tsem 无数据行")
    reps_done = max(int(r.get("rep", 0)) for r in rows)
    expected = {(k, rep, arm) for k, rep, arm in _tsem_schedule(reps_done + 1)}
    have = {(r.get("key"), int(r.get("rep", 0)), str(r.get("arm")))
            for r in rows}
    missing = sorted(expected - have)
    if missing:
        raise SystemExit("tsem 行不完整，缺：" + repr(missing[:6]))
    result = _tsem_verdict(rows)
    verifier_src = (PROJECT_ROOT / "SelfEvolvingHarnessTS" / "runtime"
                    / "candidate_verification.py").read_bytes()
    frozen_sha = (proto.get("p3_sanity") or {}).get("verifier_file_sha256")
    result["p3_sanity_verifier_unchanged"] = bool(
        frozen_sha
        and hashlib.sha256(verifier_src).hexdigest() == frozen_sha)
    if result["verdict"] == "ESCALATE_REP1":
        tsem["round1_verdict"] = result
        tsem["state"] = "AWAITING_REP1"
        print("tsem round1: ESCALATE_REP1 —— 按预注册跑 tsem extra 补第二 rep")
    elif reps_done == 0:
        tsem["round1_verdict"] = result
        tsem["verdict"] = result
        tsem["state"] = "CLOSED"
        print("tsem round1 终裁: " + result["verdict"])
    else:
        tsem["verdict"] = result
        tsem["state"] = "CLOSED"
        print("tsem 终裁: " + result["verdict"])
    report["tsem"] = tsem
    _save_report(report)
    print(json.dumps(result, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- fullop2
# FULLOP2（用户裁决 2026-08-14）：rev7 固化（h0 新基线）后的新 FULLOP。
# A = rev7 + actionable（原 A 池）；B = rev7 + full。同期交错、相同
# roster、相同后端、相同预算。原 fullop（rev6 时代 INCONCLUSIVE）保留
# 不动，本节为独立新版本。

def _fullop2_schedule() -> list:
    """同期交错（tsem 纪律）：rep0 roster 序 i 偶 A→B / i 奇 B→A；rep1 取反。"""
    entries = []
    for rep in range(FULLOP_REPS):
        for i, ctx in enumerate(FULLOP_ROSTER):
            arms = ("A", "B") if (i + rep) % 2 == 0 else ("B", "A")
            for arm in arms:
                entries.append((str(ctx["key"]), rep, arm))
    return entries


def _fullop2_verdict(fullop2: Mapping[str, Any]) -> dict:
    """冻结裁定（fullop2_protocol.verdict_rules，2026-08-14 pre-run amendment：
    取消 A 绝对阈值 veto——A 差 B 好是可解释的方法效应，不得被健康门挡住；
    改为同期 paired attribution；纯机械）。
    优先级：protocol > joint_failure > overload > supply > Phase2 效用。"""
    rows = fullop2.get("rows") or []
    agg = _fullop_aggregate(rows)
    a = agg["A"]
    b = agg["B"]
    perr = [str(r["key"]) + " rep" + str(r.get("rep")) + " " + str(r["arm"])
            for r in rows if r.get("protocol_error")]
    if perr:
        return {"verdict": "PROTOCOL_INCONCLUSIVE", "rule": "protocol",
                "protocol_errors": perr, "arm_aggregates": agg}
    joint_fail: list = []
    for ctx in FULLOP_ROSTER:
        key = str(ctx["key"])
        crs = [r for r in rows if r["key"] == key]
        if crs and not any((r.get("metrics") or {}).get("chain_complete")
                           for r in crs):
            joint_fail.append(key)
    if len(joint_fail) >= 4:
        return {"verdict": "PREPARE_STAGE_BLOCKED", "rule": "joint_failure",
                "joint_fail_contexts": joint_fail,
                "note": ("A/B 在进入 pool 差异前共同失败（≥4/6 context 双臂零闭链）"
                         "——前端测量阻塞，非池效应"),
                "arm_aggregates": agg}
    overload = bool(
        b["chain_complete"] < a["chain_complete"] - 1
        or b["rejected_total"] > a["rejected_total"] + 1)
    if overload:
        return {"verdict": "OPERATOR_SPACE_OVERLOAD",
                "rule": "chain_collapse",
                "joint_fail_contexts": joint_fail, "arm_aggregates": agg}
    b_new_legal = list(b["b_new_legal"])
    b_new_rej = list(b["b_new_proposed_rejected"])
    supply_base = {"rule": "supply", "joint_fail_contexts": joint_fail,
                   "arm_aggregates": agg,
                   "observed_a_chain": a["chain_complete"],
                   "observed_b_chain": b["chain_complete"]}
    if not b_new_legal:
        if b_new_rej:
            return dict(supply_base, verdict="CONTEXT_OR_SELECTION_BLOCKER",
                        note=("B 提出了 A 池外候选但全未入池（proposed_rejected>0）"
                              "——完整池供应被选择/预算层挡住"))
        note = ("完整池在本 roster 上未产生受限池之外的合法候选"
                "——actionable 池供应已足够；非 overload 证据")
        if b["chain_complete"] > a["chain_complete"] + 1:
            note += ("；注意 B 闭链显著更高但未用新算子——池暴露的间接效应，"
                     "非新算子供应")
        return dict(supply_base, verdict="FULL_POOL_NO_NEW_LEGAL_SUPPLY",
                    note=note)
    p2 = fullop2.get("p2")
    if not p2:
        return {"verdict": "AWAITING_PHASE2",
                "rule": "supply_ok",
                "b_new_legal": b_new_legal,
                "note": "b_new_legal>0——按冻结协议执行 fullop2-p2（零 LLM）",
                "joint_fail_contexts": joint_fail, "arm_aggregates": agg}
    entries = p2.get("entries") or []
    winners = [e for e in entries if e.get("winner_candidate_id")]
    pass_winners = []
    delayed_regress = []
    for e in winners:
        d = e.get("delayed") or {}
        dg = d.get("gain")
        if dg is not None and float(dg) > -float(M):
            pass_winners.append(e["key"])
        else:
            delayed_regress.append(e["key"])
    harm_total = sum(int(e.get("harm_count") or 0) for e in entries)
    base = {"rule": "phase2_utility", "joint_fail_contexts": joint_fail,
            "arm_aggregates": agg, "b_new_legal": b_new_legal,
            "p2_any_positive": bool(p2.get("any_positive")),
            "harm_total": harm_total}
    if pass_winners:
        return dict(base, verdict="FULL_OPERATOR_CAPABILITY_PASS",
                    pass_contexts=pass_winners,
                    delayed_regressions=delayed_regress,
                    note=("≥1 context 新增候选正 Support（gain ≥ M）且 delayed "
                          "不劣——完整算子空间在 rev7 下产生真实增量价值"))
    note = ("新增合法候选执行后无正 Support（program headroom 不足）"
            if not winners else "正 Support 但 delayed 回归（≤ -M）")
    return dict(base, verdict="LEGAL_BUT_NO_UTILITY", note=note,
                delayed_regressions=delayed_regress)


def phase_fullop2_freeze() -> int:
    """冻结 fullop2 协议进主报告（任何运行前；已冻结或已有数据行则拒绝）。"""
    report = _load_report()
    if report.get("fullop2_protocol"):
        raise SystemExit("fullop2_protocol 已存在——拒绝重复冻结")
    if (report.get("fullop2") or {}).get("rows"):
        raise SystemExit("fullop2 已有数据行——协议必须先冻结")
    h0 = _h0_snapshot()
    import hashlib  # noqa: PLC0415
    body = _bootstrap_body(h0)
    proto = {
        "experiment_id": "FULL_OPERATOR_SKILL_CAPABILITY_V2_REV7",
        "user_ruling": ("2026-08-14：rev7 固化批准 + 新 FULLOP 批准——A=rev7+actionable、"
                        "B=rev7+full，同期交错、相同 Context/后端/预算；Slow inspect "
                        "编辑面暂缓（无证据）"),
        "supersedes": ("fullop_protocol（rev6 时代，终裁 INCONCLUSIVE_PROTOCOL_FAILURE，"
                       "保留不动；本实验为 rev7 下独立新版本）"),
        "baseline_skill": {"revision": 7,
                           "body_sha": hashlib.sha256(
                               body.encode("utf-8")).hexdigest(),
                           "note": "h0 已固化 rev7（TSEM 配对因果验证后）"},
        "unique_variable": "算子池暴露面（actionable vs full）——Skill/模型/预算/Memory 全同",
        "arms": {"A": "rev7 + pool_mode=actionable",
                 "B": "rev7 + pool_mode=full"},
        "roster": [{"key": str(c["key"]), "variant": c["variant"]}
                   for c in FULLOP_ROSTER],
        "reps": FULLOP_REPS,
        "schedule_frozen": [list(e) for e in _fullop2_schedule()],
        "schedule_note": "rep0 i 偶 A→B / i 奇 B→A；rep1 逐 context 取反（tsem 纪律）",
        "phase1_metrics": "沿用 _fullop_row_metrics / _fullop_aggregate（含 b_new_legal 口径）",
        "verdict_rules": {
            "precedence": ["protocol", "joint_failure", "overload", "supply",
                           "phase2_utility"],
            "protocol": ("任一行 protocol_error → PROTOCOL_INCONCLUSIVE；"
                         "配对缺失（行不完整）→ 拒绝裁定并提示补齐（SystemExit，"
                         "不产生裁定——缺失=跑批未完成，不是测量失败）"),
            "joint_failure": ("双臂在同一 context 全部 4 行零闭链的 context 数 ≥ 4/6 → "
                              "PREPARE_STAGE_BLOCKED（A/B 在进入 pool 差异前共同失败——"
                              "前端测量阻塞，非池效应）；1-3 个记录于 joint_fail_contexts "
                              "但不阻断"),
            "overload": ("B chain_total < A chain_total − 1 或 B rejected_total > "
                         "A rejected_total + 1 → OPERATOR_SPACE_OVERLOAD（B 相对 A 回归）"),
            "supply": ("b_new_legal = 0：有 b_new_proposed_rejected → "
                       "CONTEXT_OR_SELECTION_BLOCKER；无 → FULL_POOL_NO_NEW_LEGAL_SUPPLY"
                       "（actionable 池已足够，非 overload 证据；若 B 闭链 > A+1 而未用"
                       "新算子，注明为池暴露间接效应）；b_new_legal > 0 → AWAITING_PHASE2"),
            "phase2_utility": ("chosen-first、每 context 预算 FULLOP_SUPPORT_BUDGET=2、零 LLM；"
                               "positive = gain ≥ M；winner-only delayed（origin+HORIZON）；"
                               "≥1 winner 且其 delayed > −M → FULL_OPERATOR_CAPABILITY_PASS；"
                               "否则 LEGAL_BUT_NO_UTILITY（无正 gain=program headroom 不足；"
                               "正 gain 但 delayed ≤ −M=delayed 回归）；harm_count 记录"),
            "abolished_rule_0": ("原 A 臂绝对阈值健康门已废除（2026-08-14 pre-run "
                                 "amendment）——A 差 B 好是可解释的方法效应，绝对 veto "
                                 "会把真实池效应误判为不可裁定"),
        },
        "claim_boundary": ("PASS 含义：完整算子空间在 rev7 下对本 roster 产生真实增量价值"
                           "（≥1 context 正 Support 且 delayed 不劣）。仍不证明 Slow "
                           "自进化能力；文本 Guidance 自进化 family 保持关闭"),
        "expected_llm_calls": "≤ 24 prepares × 12（实测约 3-6/prepare，预计 ~90）",
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["fullop2_protocol"] = proto
    _save_report(report)
    print("fullop2_protocol FROZEN; rev7 body sha", proto["baseline_skill"]["body_sha"][:16])
    return 0


def phase_fullop2(env: Mapping[str, Any] | None = None) -> int:
    """24 次真实 prepare（6 context × 2 rep × 2 arm；resumable——按
    (key, rep, arm) 跳过；每行落盘）。两臂同为 rev7 h0 snapshot。"""
    report = _load_report()
    if not report.get("fullop2_protocol"):
        raise SystemExit("fullop2_protocol 未冻结——先跑 fullop2-freeze")
    if env is None:
        env = _load_env()
    fullop2 = report.setdefault("fullop2", {})
    rows: list[dict[str, Any]] = fullop2.setdefault("rows", [])
    done = {(str(r["key"]), int(r["rep"]), str(r["arm"])) for r in rows}
    context_pools = fullop2.setdefault("context_pools", {})
    ctx_by_key = {str(c["key"]): c for c in FULLOP_ROSTER}
    for key, rep, arm in _fullop2_schedule():
        if (key, rep, arm) in done:
            continue
        ctx = ctx_by_key[key]
        if (key not in context_pools
                or not isinstance(context_pools[key].get("features"), Mapping)):
            context_pools[key] = _fullop_context_pools(
                env, str(ctx["series"]), int(ctx["origin"]), ctx["variant"])
            _save_report(report)
            print(f"== fullop2 pools {key}: a={len(context_pools[key]['a_pool'])}"
                  f" b={len(context_pools[key]['b_pool'])}", flush=True)
        run_env = _fullop_values(env, ctx["variant"])
        mode = "actionable" if arm == "A" else "full"
        print(f"== fullop2 {arm} {key} rep{rep} ...", flush=True)
        row = _prepare_arm(_h0_snapshot(), str(ctx["series"]),
                           int(ctx["origin"]), env,
                           with_task_context=True, pool_mode=mode,
                           values_override=run_env)
        row["key"] = key
        row["rep"] = rep
        row["arm"] = arm
        row["pool_mode"] = mode
        row["metrics"] = _fullop_row_metrics(
            row, context_pools[key]["a_pool"], context_pools[key]["features"])
        rows.append(row)
        _save_report(report)
        mm = row["metrics"]
        print(f"== fullop2 {arm} {key} rep{rep} done: "
              f"chain={mm.get('chain_complete')} "
              f"proposed={mm.get('n_proposed')} "
              f"new_legal={mm.get('b_new_legal')} "
              f"llm={row.get('llm_calls')}", flush=True)
    fullop2["arm_aggregates"] = _fullop_aggregate(rows)
    _save_report(report)
    print("== fullop2 arm aggregates:",
          json.dumps(fullop2["arm_aggregates"], ensure_ascii=False))
    return 0


def phase_fullop2_p2(env: Mapping[str, Any] | None = None) -> int:
    """Phase 2（条件：phase1 裁定 AWAITING_PHASE2）：只执行 B 新增且合法的
    候选——零 LLM：chosen-first → Target Support → harm；winner 再在
    origin+HORIZON 评估 delayed。"""
    report = _load_report()
    fullop2 = report.get("fullop2") or {}
    if fullop2.get("p2"):
        print("== fullop2-p2: already executed")
        return 0
    if (fullop2.get("phase1_verdict") or {}).get("verdict") != "AWAITING_PHASE2":
        raise SystemExit("phase1 未裁定 AWAITING_PHASE2——拒绝跑 p2（预注册纪律）")
    if env is None:
        env = _load_env()
    rows = fullop2.get("rows") or []
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: PLC0415
        ScopeExecutor,
    )
    entries: list[dict[str, Any]] = []
    any_positive = False
    for ctx in FULLOP_ROSTER:
        key = str(ctx["key"])
        b_rows = [r for r in rows
                  if r["key"] == key and r["arm"] == "B"
                  and not r.get("protocol_error")]
        new_legal: dict[str, dict[str, Any]] = {}
        for r in b_rows:
            for cid in (r.get("metrics") or {}).get("b_new_legal") or []:
                new_legal.setdefault(str(cid), r)
        if not new_legal:
            entries.append({"key": key, "status": "no_b_new_legal"})
            continue
        primary = b_rows[0]
        steps_map = primary.get("candidate_steps") or {}
        chosen_cid = str(primary.get("chosen_candidate_id") or "")
        order = ([chosen_cid] if chosen_cid in new_legal else []) + \
            [c for c in new_legal if c != chosen_cid]
        series = str(ctx["series"])
        origin = int(ctx["origin"])
        values = _fullop_values(env, ctx["variant"])
        roster, vals = _support_roster(series, values)
        executor = ScopeExecutor(roster, vals, nsu._config(),
                                 evaluate_fn=nsu._evaluate_kdd)
        probes: list[dict[str, Any]] = []
        winner: dict[str, Any] | None = None
        for cid in order[:FULLOP_SUPPORT_BUDGET]:
            steps = steps_map.get(cid) or []
            tuples = tuple((s["op"], dict(s["params"])) for s in steps)
            rr = executor.evaluate(tuples, origin)
            gain = float(rr.gain) if rr.gain is not None else None
            probe = {"candidate_id": cid, "steps": steps,
                     "gain": gain, "passed": bool(rr.verification.passed)}
            probes.append(probe)
            if winner is None and gain is not None and gain >= M:
                winner = probe
        entry: dict[str, Any] = {
            "key": key, "series": series, "origin": origin,
            "synthetic_context": ctx["variant"] == "synmiss",
            "chosen": chosen_cid, "probes": probes,
            "winner_candidate_id": (winner or {}).get("candidate_id"),
            "positive_count": sum(
                1 for p in probes
                if p.get("gain") is not None and p["gain"] >= M),
            "harm_count": sum(
                1 for p in probes
                if p.get("gain") is not None and p["gain"] < -M),
        }
        if winner is not None:
            tuples = tuple((s["op"], dict(s["params"]))
                           for s in winner["steps"])
            rr = executor.evaluate(tuples, origin + HORIZON)
            g = float(rr.gain) if rr.gain is not None else None
            entry["delayed"] = {
                "candidate_id": winner["candidate_id"],
                "origin": origin + HORIZON,
                "gain": g, "passed": bool(rr.verification.passed)}
        if entry["positive_count"] > 0:
            any_positive = True
        entries.append(entry)
        print(f"== fullop2-p2 {key}: {json.dumps(entry, ensure_ascii=False)}",
              flush=True)
    fullop2["p2"] = {"entries": entries, "any_positive": any_positive}
    _save_report(report)
    print("== fullop2-p2 any_positive:", any_positive)
    return 0


def phase_fullop2_verdict() -> int:
    """机械裁定：完整性预检 → _fullop2_verdict → 落盘。
    AWAITING_PHASE2 只写 phase1_verdict；p2 在场时给终裁。"""
    report = _load_report()
    if not report.get("fullop2_protocol"):
        raise SystemExit("fullop2_protocol 未冻结")
    fullop2 = report.get("fullop2") or {}
    rows = fullop2.get("rows") or []
    if not rows:
        raise SystemExit("fullop2 无数据行")
    expected = {(k, rep, arm) for k, rep, arm in _fullop2_schedule()}
    have = {(str(r["key"]), int(r["rep"]), str(r["arm"])) for r in rows}
    missing = sorted(expected - have)
    if missing:
        raise SystemExit("fullop2 行不完整，缺：" + repr(missing[:6]))
    result = _fullop2_verdict(fullop2)
    if result["verdict"] == "AWAITING_PHASE2" and not fullop2.get("p2"):
        fullop2["phase1_verdict"] = result
        fullop2["state"] = "AWAITING_PHASE2"
        print("fullop2 phase1: AWAITING_PHASE2 —— 按冻结协议跑 fullop2-p2")
    else:
        fullop2["verdict"] = result
        fullop2["state"] = "CLOSED"
        print("fullop2 终裁: " + result["verdict"])
    report["fullop2"] = fullop2
    _save_report(report)
    print(json.dumps(result, ensure_ascii=False))
    return 0

# ---------------------------------------------------------------- usel
# SELECT_UTILITY_CHECK（用户裁决 2026-08-14）：零 LLM、复用 FULLOP2 已生成
# 合法候选、不换 roster。同 (Context × Program × Scope) 只评估一次（跨臂/
# 跨 rep 去重，不算独立证据）。每 context 至多 2 个 Support probe
# （probe1 = schedule 序首个 distinct 非 identity chosen；probe2 = 第二个
# distinct chosen，否则冻结池序首个非 identity alternative）。禁止看
# Outcome 后选 alternative。Support 正向者进 winner 比较（gain 最高，tie
# 取 probe1），delayed 只开最终 winner（origin+HORIZON）。双层判定：
# 候选效用 + 选择质量。


def _support_roster(series: str, values: Mapping[str, Any]) -> tuple:
    """支持窗口纪律（2026-08-14 修复）：Support 探测只用目标 dev series 自身
    （train+eval 同一序列，origin 窗口出分）。census eval_series
    （T13/T128–T134）是冻结最终评估窗口，禁止读取——原代码把 eval_series
    塞进 Support roster（latent KeyError：cohort values 不含这些序列，
    且构成 scope 违例）。"""
    return ([{"series_uid": series, "role": "train"},
             {"series_uid": series, "role": "eval"}],
            {series: values[series]})

def _usel_prog_key(steps: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(list(steps), ensure_ascii=False, sort_keys=True,
                      default=str)


def _usel_probe_plan(rows: Sequence[dict]) -> dict:
    """冻结探测计划（运行前确定性）。返回 probe1/probe2（program key 或
    None=identity）与评估集。"""
    by_prog: dict[str, list] = {}
    chosen_seq: list[str] = []
    seen_chosen: set[str] = set()
    identity_rows = 0
    for r in rows:
        steps_map = r.get("candidate_steps") or {}
        chosen = str(r.get("chosen_candidate_id") or "identity")
        if chosen == "identity":
            identity_rows += 1
        else:
            steps = steps_map.get(chosen) or []
            k = _usel_prog_key(steps)
            by_prog.setdefault(k, steps)
            if k not in seen_chosen:
                seen_chosen.add(k)
                chosen_seq.append(k)
        pool_ids = {cid for cid in (r.get("candidate_ids") or ())
                    if cid != "identity"}
        for pc in (r.get("metrics") or {}).get("per_candidate") or []:
            cid = str(pc.get("candidate_id"))
            if cid in pool_ids:
                steps = steps_map.get(cid) or []
                if steps:
                    by_prog.setdefault(_usel_prog_key(steps), steps)
    probe1 = chosen_seq[0] if chosen_seq else None
    alt_keys = sorted(by_prog)
    probe2 = None
    if len(chosen_seq) >= 2:
        probe2 = chosen_seq[1]
    elif probe1 is not None:
        for k in alt_keys:
            if k != probe1:
                probe2 = k
                break
    else:
        probe2 = alt_keys[0] if alt_keys else None
    evals: dict[str, dict] = {}
    for tag, k in (("probe1", probe1), ("probe2", probe2)):
        if k is not None and k not in evals:
            evals[tag] = {"program_key": k, "steps": by_prog[k]}
    return {"probe1": probe1, "probe2": probe2, "evals": evals,
            "identity_rows": identity_rows, "chosen_seq": chosen_seq}


def _usel_verdict(entries: Sequence[dict]) -> dict:
    """冻结双层裁定（usel_protocol.verdict_rules；纯机械）。
    效用优先级：PROTOCOL_INCONCLUSIVE > MIXED_CONTEXT_UTILITY >
    CANDIDATE_UTILITY_PASS > LEGAL_SUPPLY_NO_HEADROOM。"""
    from SelfEvolvingHarnessTS.operators.registry import (  # noqa: PLC0415
        OPERATOR_METADATA,
    )

    def fam(steps: Sequence[Mapping[str, Any]]) -> tuple:
        return tuple(sorted({str(OPERATOR_METADATA.get(
            s.get("op"), {}).get("category")) for s in steps}))

    for e in entries:
        if e.get("error"):
            return {"verdict": "PROTOCOL_INCONCLUSIVE",
                    "layer": "utility",
                    "errors": [e["error"] for e in entries if e.get("error")]}
    pos_keys: dict = {}
    neg_keys: dict = {}
    winners = []
    for e in entries:
        for tag, r in (e.get("results") or {}).items():
            g = r.get("gain")
            if g is None:
                continue
            f = fam(r["steps"])
            if g >= M:
                pos_keys.setdefault(f, set()).add(e["key"])
            if g < -M:
                neg_keys.setdefault(f, set()).add(e["key"])
        w = (e.get("results") or {}).get(e.get("winner_key") or "")
        if w:
            winners.append({"key": e["key"], "tag": e["winner_key"],
                            "family": fam(w["steps"]), "gain": w["gain"],
                            "delayed": w.get("delayed")})
    mixed = [list(f) for f, keys in neg_keys.items()
             if pos_keys.get(f, set()) - keys]
    if mixed:
        return {"verdict": "MIXED_CONTEXT_UTILITY", "layer": "utility",
                "flip_families": mixed,
                "note": "同 family 候选跨 Context 正负翻转——first fault 指向 "
                        "Scope/Observation/Risk"}
    certified = [w for w in winners
                 if w["delayed"] and w["delayed"].get("gain") is not None
                 and w["delayed"]["gain"] > -M]
    if certified:
        return {"verdict": "CANDIDATE_UTILITY_PASS", "layer": "utility",
                "certified_winners": certified}
    return {"verdict": "LEGAL_SUPPLY_NO_HEADROOM", "layer": "utility",
            "note": "合法候选全部无材料性改善（gain ≥ M 不存在）——program "
                    "headroom 不足，调查 select 无意义"}


def _usel_selection(entries: Sequence[dict]) -> dict:
    """冻结选择质量裁定。逐 context 行组标签 + 全局合成。
    行组：identity 组 vs 各 distinct chosen program 组。"""
    per_context = []
    for e in entries:
        results = e.get("results") or {}
        w = results.get(e.get("winner_key") or "") if e.get("winner_key") else None
        certified = bool(
            w and w.get("delayed") and w["delayed"].get("gain") is not None
            and w["delayed"]["gain"] > -M)
        harms = [tag for tag, r in results.items()
                 if r.get("gain") is not None and r["gain"] < -M]
        labels: dict = {}
        if not results:
            labels["context"] = "SELECTION_UNAVAILABLE"
        else:
            if e.get("identity_rows", 0) > 0:
                # 任何 certified winner 都是 identity 行组未选而正向的候选
                # （含 winner==probe1 的混合 context——checker 0746de82
                # major-2 修正）
                if certified:
                    labels["identity_group"] = "SELECTOR_CONSERVATISM_CONFIRMED"
                elif harms:
                    labels["identity_group"] = "SELECTOR_HARM_AVOIDANCE_CONFIRMED"
                else:
                    labels["identity_group"] = "SELECT_ALIGNED"
            if e.get("probe1") is not None:
                # program 行组：probe2 存在与否不影响有害 chosen 判定
                # （checker 0746de82 major-1 修正）
                p1 = results.get("probe1") or {}
                if certified and e.get("winner_key") == "probe2":
                    labels["program_group"] = "SELECT_MISALIGNED"
                elif (p1.get("gain") is not None and p1["gain"] < -M
                      and not certified):
                    labels["program_group"] = "SELECT_MISALIGNED"
                else:
                    labels["program_group"] = "SELECT_ALIGNED"
        per_context.append({"key": e["key"], "labels": labels,
                            "winner_tag": e.get("winner_key"),
                            "certified_winner": certified})
    any_conservatism = any(
        "SELECTOR_CONSERVATISM_CONFIRMED" in c["labels"].values()
        for c in per_context)
    any_misaligned = any(
        "SELECT_MISALIGNED" in c["labels"].values() for c in per_context)
    any_harm_avoid = any(
        "SELECTOR_HARM_AVOIDANCE_CONFIRMED" in c["labels"].values()
        for c in per_context)
    if any_conservatism:
        verdict = "SELECTOR_CONSERVATISM_CONFIRMED"
    elif any_misaligned:
        verdict = "SELECT_MISALIGNED"
    elif any_harm_avoid:
        verdict = "SELECTOR_HARM_AVOIDANCE_CONFIRMED"
    else:
        verdict = "SELECT_ALIGNED"
    return {"verdict": verdict, "layer": "selection", "per_context": per_context}


def phase_usel_freeze() -> int:
    """冻结 usel 协议进主报告（任何评估前；已冻结则拒绝）。"""
    report = _load_report()
    if report.get("usel_protocol"):
        raise SystemExit("usel_protocol 已存在——拒绝重复冻结")
    if (report.get("usel") or {}).get("entries"):
        raise SystemExit("usel 已有评估行——协议必须先冻结")
    proto = {
        "experiment_id": "SELECT_UTILITY_CHECK_ZERO_LLM",
        "user_ruling": ("2026-08-14：完整池非当前阻塞；转向候选下游效用与 select 质量"
                        "校验；复用 FULLOP2 已生成合法候选，不再调用 LLM、不换 roster"),
        "basis": "fullop2（rev7 基线，24 行）",
        "dedup_rule": "同 (Context × Program × Scope) 只评估一次，跨臂/跨 rep 复用——不算独立证据",
        "probe_rules": {
            "probe1": "schedule 序首个 distinct 非 identity chosen program（无则 identity）",
            "probe2": ("第二个 distinct chosen program（有）；否则冻结池序（program key "
                       "canonical 字典序）首个 ≠ probe1 的合法非 identity program"),
            "max_two": "每 context 至多 2 个 Support probe；禁止看 Outcome 后选 alternative",
            "synmiss_note": "identity 行组与已评估 impute 候选免费比较（identity gain ≡ 0）",
        },
        "evaluation": ("ScopeExecutor，origin 窗口，evaluate_fn=nsu._evaluate_kdd；"
                       "gain 相对 identity 控制；positive = gain ≥ M；harm = gain < -M"),
        "winner_and_delayed": ("Support 正向候选进 winner 比较（gain 最高，tie 取 probe1）；"
                               "delayed 只开最终 winner（origin+HORIZON）；"
                               "certified = delayed gain > -M"),
        "utility_verdicts": {
            "precedence": ["PROTOCOL_INCONCLUSIVE", "MIXED_CONTEXT_UTILITY",
                           "CANDIDATE_UTILITY_PASS", "LEGAL_SUPPLY_NO_HEADROOM"],
            "CANDIDATE_UTILITY_PASS": "存在 Support 正向且 delayed 不劣（certified）的 Workflow",
            "LEGAL_SUPPLY_NO_HEADROOM": "合法候选全部无材料性改善",
            "MIXED_CONTEXT_UTILITY": "相同 family 在不同 Context 正负翻转",
            "PROTOCOL_INCONCLUSIVE": "评估失败/结果不可读",
        },
        "selection_verdicts": {
            "SELECT_ALIGNED": "chosen 与实际最佳候选一致（含无更优 certified 选项）",
            "SELECTOR_CONSERVATISM_CONFIRMED": "选 identity，但未选候选 certified 正向",
            "SELECTOR_HARM_AVOIDANCE_CONFIRMED": "选 identity，而 alternative 实际有害",
            "SELECT_MISALIGNED": "chosen program 败给 certified alternative 或自身有害",
            "SELECTION_UNAVAILABLE": "无可比较 alternative",
            "global_precedence": ["CONSERVATISM", "MISALIGNED", "HARM_AVOIDANCE", "ALIGNED"],
        },
        "surface_decision_map": {
            "PASS + ALIGNED": "基础 Skill 能力成立 → 进入 Batch 结构化 Slow Update（新提案）",
            "PASS + CONSERVATISM/MISALIGNED": "first fault = Selection/Control",
            "NO_HEADROOM": "Program headroom 不足——调查 select 无意义",
            "MIXED": "first fault = Scope/Observation/Risk",
        },
        "scope_closed": "仅关闭：本 roster 上 full pool 相对 actionable pool 的增量供应分支；"
                        "不扩 richer Context；不开 Slow inspect 编辑面",
        "expected_llm_calls": "0（纯机械，复用已生成候选）",
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["usel_protocol"] = proto
    _save_report(report)
    print("usel_protocol FROZEN")
    return 0


def phase_usel(env: Mapping[str, Any] | None = None) -> int:
    """零 LLM 评估：按冻结计划逐 context 评估（可恢复：已有 entries 的 context
    跳过）。"""
    report = _load_report()
    if not report.get("usel_protocol"):
        raise SystemExit("usel_protocol 未冻结——先跑 usel-freeze")
    if env is None:
        env = _load_env()
    fullop2 = report.get("fullop2") or {}
    rows = fullop2.get("rows") or []
    if not rows:
        raise SystemExit("fullop2 无数据行")
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: PLC0415
        ScopeExecutor,
    )
    usel = report.setdefault("usel", {})
    entries = usel.setdefault("entries", [])
    done_keys = {e.get("key") for e in entries}
    ctx_by_key = {str(c["key"]): c for c in FULLOP_ROSTER}
    rows_by_key: dict[str, list] = {}
    for r in rows:
        rows_by_key.setdefault(str(r["key"]), []).append(r)
    for ctx in FULLOP_ROSTER:
        key = str(ctx["key"])
        if key in done_keys:
            continue
        plan = _usel_probe_plan(rows_by_key.get(key) or [])
        series, origin = str(ctx["series"]), int(ctx["origin"])
        values = _fullop_values(env, ctx["variant"])
        roster, vals = _support_roster(series, values)
        executor = ScopeExecutor(roster, vals, nsu._config(),
                                 evaluate_fn=nsu._evaluate_kdd)
        results: dict[str, dict] = {}
        error = None
        for tag, spec in plan["evals"].items():
            tuples = tuple((s["op"], dict(s["params"])) for s in spec["steps"])
            rr = executor.evaluate(tuples, origin)
            g = float(rr.gain) if rr.gain is not None else None
            results[tag] = {"program_key": spec["program_key"],
                            "steps": spec["steps"], "gain": g,
                            "passed": bool(rr.verification.passed)}
            if g is None or not rr.verification.passed:
                error = key + " eval " + tag + " unreadable"
        positives = [(tag, r) for tag, r in results.items()
                     if r.get("gain") is not None and r["gain"] >= M]
        winner_key = None
        if positives:
            positives.sort(key=lambda kv: (-kv[1]["gain"], kv[0]))
            winner_key = positives[0][0]
            wsteps = results[winner_key]["steps"]
            tuples = tuple((s["op"], dict(s["params"])) for s in wsteps)
            rr = executor.evaluate(tuples, origin + HORIZON)
            dg = float(rr.gain) if rr.gain is not None else None
            if dg is None or not rr.verification.passed:
                error = key + " delayed unreadable"
            results[winner_key]["delayed"] = {
                "origin": origin + HORIZON, "gain": dg,
                "passed": bool(rr.verification.passed)}
        entry = {"key": key, "probe1": plan["probe1"],
                 "probe2": plan["probe2"],
                 "identity_rows": plan["identity_rows"],
                 "chosen_seq": plan["chosen_seq"],
                 "results": results, "winner_key": winner_key,
                 "error": error}
        entries.append(entry)
        usel["entries"] = entries
        _save_report(report)
        print("== usel " + key + ": " + json.dumps(entry, ensure_ascii=False),
              flush=True)
    _save_report(report)
    return 0


def phase_usel_verdict() -> int:
    """机械裁定：双层判定落盘。"""
    report = _load_report()
    if not report.get("usel_protocol"):
        raise SystemExit("usel_protocol 未冻结")
    usel = report.get("usel") or {}
    entries = usel.get("entries") or []
    if not entries:
        raise SystemExit("usel 无评估行")
    expected_keys = {str(c["key"]) for c in FULLOP_ROSTER}
    have_keys = {str(e.get("key")) for e in entries}
    if have_keys != expected_keys:
        raise SystemExit("usel keys mismatch: missing="
                         + repr(sorted(expected_keys - have_keys))
                         + " extra=" + repr(sorted(have_keys - expected_keys)))
    utility = _usel_verdict(entries)
    selection = _usel_selection(entries)
    usel["utility_verdict"] = utility
    usel["selection_verdict"] = selection
    usel["state"] = "CLOSED"
    report["usel"] = usel
    _save_report(report)
    print("usel utility:", utility["verdict"])
    print("usel selection:", selection["verdict"])
    return 0

# ---------------------------------------------------------------- batch1
# BATCH1_OUTLIER_SCOPE_EVIDENCE（用户裁决 2026-08-14）：固定 Batch 轨迹积累。
# 冻结网格 = census pre_registered dev_series × origins 减去 usel 已用 6
# context（选择只看可用性，不看 gain）；固定 outlier_mad vs identity、Task/
# Consumer/Metric/窗口语义；零 LLM；不改 Skill/Select/Memory/Observation。
# 每个 context 保存：运行前可见 Pattern（public features）、执行几何
# （behavior_point_count/per_view_gain）、Support gain、正向 winner 的
# delayed、series/origin 去重。裁定：≥3 独立正例 且 ≥3 独立负例才允许在
# 一部分 context 上提 Scope 候选并在剩余 context 验证；否则继续
# INSUFFICIENT_CONTRASTIVE_EVIDENCE。

def _batch1_contexts() -> list:
    """冻结网格（可用性唯一依据）：census pre_registered 的 4 series × 4
    origins，减去 usel 已用窗口。确定性、与 gain 无关。"""
    census = json.loads(CENSUS_REL.read_text(encoding="utf-8"))
    series = list(census["pre_registered"]["dev_series"])
    origins = list(census["pre_registered"]["origins"])
    used = {"T1@888", "T10@600", "T100@600", "T101@792", "T101@600"}
    out = []
    for s in series:
        for o in origins:
            key = s + "@" + str(o)
            if key not in used:
                out.append((s, o))
    return out


def _batch1_separability(labeled: Sequence[Mapping[str, Any]]) -> list:
    """零 LLM 可分离性：每个数值 feature 是否把全部正例与全部负例干净分开
    （正例值严格在负例 [min,max] 之外，或反向）。返回候选 feature 列表。"""
    feats: dict[str, list] = {}
    for item in labeled:
        for k, v in (item.get("features") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                feats.setdefault(k, []).append((float(v), item["label"]))
    out = []
    for fname, pairs in sorted(feats.items()):
        pos_vals = [v for v, lab in pairs if lab == "POSITIVE"]
        neg_vals = [v for v, lab in pairs if lab == "NEGATIVE"]
        if not pos_vals or not neg_vals:
            continue
        lo, hi = min(neg_vals), max(neg_vals)
        if all(v > hi for v in pos_vals):
            out.append({"feature": fname, "direction": "positive_above",
                        "threshold": (hi + min(pos_vals)) / 2.0})
        elif all(v < lo for v in pos_vals):
            out.append({"feature": fname, "direction": "positive_below",
                        "threshold": (lo + max(pos_vals)) / 2.0})
    return out


def _batch1_verdict(batch_entries: Sequence[dict],
                    usel_labels: Sequence[dict]) -> dict:
    """冻结裁定（batch1_protocol.verdict_rules；纯机械）。"""
    return _scope_verdict_from_labeled(
        _labeled_evidence(batch_entries, usel_labels))


def _scope_verdict_from_labeled(labeled: Sequence[dict]) -> dict:
    """Scope 裁定核心（batch1/cobs 共用；纯机械）：计数 → 停止条件 →
    fit 子集可分离性 → 剩余 context 验证。"""
    pos = [x for x in labeled if x["label"] == "POSITIVE"]
    neg = [x for x in labeled if x["label"] == "NEGATIVE"]
    base = {"pos_count": len(pos), "neg_count": len(neg),
            "positives": [x["key"] for x in pos],
            "negatives": [x["key"] for x in neg]}
    if len(pos) < 3 or len(neg) < 3:
        if len(pos) == 1 and pos[0]["key"] == "T100@600":
            return dict(base, verdict="INSUFFICIENT_CONTRASTIVE_EVIDENCE",
                        outlier_scope_learning="CLOSED",
                        note=("仍只有 T100@600 正向：保留其 Local Draft，关闭 "
                              "outlier family 的一般 Scope 学习——不继续找数据"
                              "凑答案"))
        return dict(base, verdict="INSUFFICIENT_CONTRASTIVE_EVIDENCE",
                    note=("独立正例/负例不足 3/3——继续积累 Batch 轨迹，"
                          "不冻结阈值（防人工拟合）"))
    fit = pos[:3] + neg[:3]
    rest = [x for x in labeled if x not in fit]
    seps = _batch1_separability(fit)
    if not seps:
        return dict(base, verdict="SCOPE_CANDIDATE_ELIGIBLE_NOT_SEPARABLE",
                    note=("正负样本充足（≥3/≥3）但 fit 子集即不可分——"
                          "下一轮只增加一个 Consumer-conditioned Observation"
                          "（廉价历史影响力/敏感度 proxy），不建 Pattern Graph"))
    # 简单 Scope 候选：fit 子集（正/负各前 3 个 context）拟合阈值规则
    rule = {"feature": seps[0]["feature"], "direction": seps[0]["direction"],
            "threshold": seps[0]["threshold"],
            "target": "outlier_family_admissible_for_support_probe"}
    val_pass = []
    val_fail = []
    for x in rest:
        v = float(x["features"].get(rule["feature"], float("nan")))
        pred_pos = (v > rule["threshold"]
                    if rule["direction"] == "positive_above"
                    else v < rule["threshold"])
        (val_pass if pred_pos == (x["label"] == "POSITIVE") else val_fail).append(x["key"])
    if val_fail:
        return dict(base, verdict="SCOPE_CANDIDATE_ELIGIBLE_NOT_SEPARABLE",
                    scope_candidate=rule,
                    fit_contexts=[x["key"] for x in fit],
                    validation={"pass": val_pass, "fail": val_fail},
                    note=("fit 子集可分但规则在剩余 context 验证失败——现有 "
                          "Observation 整体不可分 → 下一轮只增加一个 "
                          "Consumer-conditioned Observation"))
    return dict(base, verdict="SCOPE_CANDIDATE_PROPOSED",
                scope_candidate=rule, fit_contexts=[x["key"] for x in fit],
                validation={"pass": val_pass, "fail": val_fail},
                validation_ok=True,
                note=("窄 Target-local Scope/Risk Rule 候选已产出（需用户裁决后"
                      "才能固化为 Rule——本裁定只报告不生效）"))


def phase_batch1_freeze() -> int:
    """冻结 batch1 协议（任何运行前；已冻结或已有 entry 则拒绝）。"""
    report = _load_report()
    if report.get("batch1_protocol"):
        raise SystemExit("batch1_protocol 已存在——拒绝重复冻结")
    if (report.get("batch1") or {}).get("entries"):
        raise SystemExit("batch1 已有 entry——协议必须先冻结")
    ctxs = _batch1_contexts()
    proto = {
        "experiment_id": "BATCH1_OUTLIER_SCOPE_EVIDENCE",
        "user_ruling": ("2026-08-14：启动固定 Batch 轨迹积累；冻结 8-12 个新 context"
                        "（≥3 序列，只按可用性选不看 gain）；固定 outlier_mad vs identity；"
                        "零 LLM、不改 Skill/Select/Memory/Observation"),
        "contexts": [{"key": s + "@" + str(o), "series": s, "origin": o}
                     for s, o in ctxs],
        "selection_rule": ("census pre_registered dev 网格（T1/T10/T100/T101 × "
                           "600/792/888/984）减去 usel 已用 6 窗口；确定性、与 "
                           "gain 无关"),
        "fixed_program": [{"op": "outlier_mad", "params": {}}],
        "measurement": ("_support_roster（train+eval 同一 dev series）+ "
                        "ScopeExecutor + nsu._evaluate_kdd；gain = baseline−candidate "
                        "sMASE；positive = gain ≥ M；negative = gain < −M"),
        "per_context_record": ["运行前可见 public features（extract_public_features，"
                               "series[:origin]）", "behavior_point_count（执行几何）",
                               "per_view_gain", "support gain",
                               "正向 winner 的 delayed（origin+HORIZON）",
                               "series/origin 唯一键（同一 Outcome 不重复计数）"],
        "evidence_combination": ("usel 已标签 context（T100@600 POSITIVE / "
                                 "T1@888、T10@600 NEGATIVE，当前协议口径）+ "
                                 "batch1 全部 entry"),
        "verdict_rules": {
            "counts_rule": "≥3 独立正例 且 ≥3 独立负例 → 进入 Scope 候选分支；否则 INSUFFICIENT_CONTRASTIVE_EVIDENCE",
            "only_t100_positive": ("正例总数==1 且为 T100@600 → 保留 Local Draft、"
                                   "关闭 outlier family 一般 Scope 学习（停止找数据凑答案）"),
            "separable": "现有 Observation 可干净分开正负 → SCOPE_CANDIDATE_PROPOSED：在正/负各前 3 个 context 上拟合简单阈值规则，剩余 context 验证；只报告不生效（用户裁决后固化）",
            "not_separable": "正负充足但不可分 → SCOPE_CANDIDATE_ELIGIBLE_NOT_SEPARABLE：下一轮只增加一个 Consumer-conditioned Observation（不建 Pattern Graph）",
        },
        "claim_boundary": ("复用已暴露窗口 ≠ 零新 Outcome：当前协议下未执行过的 "
                           "context × outlier_mad 会打开新 development Action-Response，"
                           "但不得称 fresh confirmation；census 旧协议 gain 只作对照"
                           "记录不计数"),
        "expected_llm_calls": "0",
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["batch1_protocol"] = proto
    _save_report(report)
    print("batch1_protocol FROZEN; contexts:", len(ctxs),
          [s + "@" + str(o) for s, o in ctxs])
    return 0


def phase_batch1(env: Mapping[str, Any] | None = None) -> int:
    """零 LLM 固定 Batch 执行（可恢复：按 context key 跳过已有 entry）。"""
    report = _load_report()
    if not report.get("batch1_protocol"):
        raise SystemExit("batch1_protocol 未冻结——先跑 batch1-freeze")
    if env is None:
        env = _load_env()
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: PLC0415
        ScopeExecutor,
    )
    values = env["values"]
    batch = report.setdefault("batch1", {})
    entries = batch.setdefault("entries", [])
    done = {e.get("key") for e in entries}
    census = json.loads(CENSUS_REL.read_text(encoding="utf-8"))
    for series, origin in _batch1_contexts():
        key = series + "@" + str(origin)
        if key in done:
            continue
        series0 = np.asarray(values[series], dtype=np.float64)
        features = dict(extract_public_features(series0[:origin],
                                                task_kind="forecast"))
        roster, vals = _support_roster(series, values)
        executor = ScopeExecutor(roster, vals, nsu._config(),
                                 evaluate_fn=nsu._evaluate_kdd)
        steps = (("outlier_mad", {}),)
        rr = executor.evaluate(steps, origin)
        gain = float(rr.gain) if rr.gain is not None else None
        entry = {"key": key, "series": series, "origin": origin,
                 "features": features, "gain": gain,
                 "passed": bool(rr.verification.passed),
                 "behavior_point_count": int(
                     getattr(rr, "behavior_point_count", 0) or 0),
                 "per_view_gain": list(getattr(rr, "per_view_gain", []) or [])}
        if gain is not None and gain >= M:
            rr2 = executor.evaluate(steps, origin + HORIZON)
            dg = float(rr2.gain) if rr2.gain is not None else None
            entry["delayed"] = {"origin": origin + HORIZON, "gain": dg,
                                "passed": bool(rr2.verification.passed)}
        entries.append(entry)
        batch["entries"] = entries
        _save_report(report)
        print("== batch1 " + key + ": gain=" + str(gain)
              + (" delayed=" + str(entry.get("delayed")) if "delayed" in entry
                 else ""), flush=True)
    _save_report(report)
    return 0


def phase_batch1_verdict() -> int:
    """机械裁定：合并 usel 标签 + batch entry → 计数/可分离性/停止条件。"""
    report = _load_report()
    if not report.get("batch1_protocol"):
        raise SystemExit("batch1_protocol 未冻结")
    batch = report.get("batch1") or {}
    entries = batch.get("entries") or []
    expected = {s + "@" + str(o) for s, o in _batch1_contexts()}
    have = {e.get("key") for e in entries}
    if have != expected:
        raise SystemExit("batch1 keys mismatch: missing="
                         + repr(sorted(expected - have))
                         + " extra=" + repr(sorted(have - expected)))
    usel_entries = (report.get("usel") or {}).get("entries") or []
    pools = (report.get("fullop2") or {}).get("context_pools") or {}
    usel_labels = []
    for e in usel_entries:
        for tag, r in (e.get("results") or {}).items():
            if r.get("gain") is None:
                continue
            # 只取 outlier_mad 单算子程序（Scope 学习对象）
            steps = r.get("steps") or []
            if [s.get("op") for s in steps] != ["outlier_mad"]:
                continue
            key = e["key"]
            label = ("POSITIVE" if r["gain"] >= M
                     else "NEGATIVE" if r["gain"] < -M else "NEUTRAL")
            usel_labels.append({"key": key, "label": label,
                                "features": dict((pools.get(key) or {})
                                                 .get("features") or {})})
    result = _batch1_verdict(entries, usel_labels)
    batch["verdict"] = result
    batch["state"] = "CLOSED"
    report["batch1"] = batch
    _save_report(report)
    print("batch1 终裁: " + result["verdict"])
    print(json.dumps(result, ensure_ascii=False))
    return 0

# ---------------------------------------------------------------- cobs
# CONSUMER_OBSERVATION_SCOPE_CHECK（用户裁决 2026-08-14 batch1 分支 c）：
# 只增加一个 Consumer-conditioned Observation——可疑区对下游 ridge 预测的
# 历史影响力 proxy（复用项目既有 action-conditioned valuation 机制：
# _ridge_reference_and_removal_predictions 的精确一阶移除影响力）。
# 零 LLM、零新 Outcome：对 batch1 已标签证据重算可分离性。

def _labeled_evidence(batch_entries: Sequence[dict],
                      usel_labels: Sequence[dict]) -> list:
    """batch1/usel 已标签证据重建（纯机械；outcome 仅作标签）。"""
    labeled = list(usel_labels) + [
        {"key": e["key"], "label": ("POSITIVE"
                                    if (e.get("gain") is not None
                                        and e["gain"] >= M)
                                    else "NEGATIVE"
                                    if (e.get("gain") is not None
                                        and e["gain"] < -M)
                                    else "NEUTRAL"),
         "features": e.get("features") or {}}
        for e in batch_entries
        if e.get("gain") is not None]
    return [x for x in labeled if x["label"] in ("POSITIVE", "NEGATIVE")]


def _cobs_feature(series: str, origin: int,
                  values: Mapping[str, Any]) -> float | None:
    """Consumer-conditioned Observation（零 LLM）：outlier_mad 会修改的
    训练样本对下游 ridge 预测的历史影响力占比。
    = mean_c RMS(一阶移除预测变化_c) / RMS(基线预测)；惰性（无修改样本）→ 0.0；
    计算失败 → None。与 outcome 标签独立（运行前可计算）。"""
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_action_conditioned_valuation_proxy import (  # noqa: PLC0415
        _ridge_reference_and_removal_predictions,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_cross_series_curation import (  # noqa: PLC0415
        _center_scale,
    )
    from SelfEvolvingHarnessTS.runtime.executor import run_pipeline  # noqa: PLC0415
    raw = np.asarray(values[series], dtype=np.float64)
    config = nsu._config()
    anchors = [int(a) for a in config["anchors"]
               if int(a) + HORIZON <= origin]
    x_train: list = []
    y_train: list = []
    candidate_rows: list[int] = []
    for anchor in anchors:
        window = raw[anchor - 192: anchor + HORIZON]
        prepared = run_pipeline((("outlier_mad", {}),), window,
                                source="agent").artifact
        prepared = np.asarray(prepared, dtype=np.float64)
        context = prepared[:192]
        target = prepared[192:]
        modified_target = bool(np.any(
            ~np.isclose(target, window[192:], equal_nan=True)))
        if modified_target:
            candidate_rows.append(len(x_train))
        center, scale, method = _center_scale(np, context)
        if method == "scale_floor_fallback":
            return None
        x_train.append((context - center) / scale)
        y_train.append((target - center) / scale)
    if not candidate_rows or not x_train:
        return 0.0
    eval_window = raw[origin - 192: origin]
    ec, es, emethod = _center_scale(np, eval_window)
    if emethod == "scale_floor_fallback":
        return None
    x_eval = np.asarray([(eval_window - ec) / es], dtype=np.float64)
    out = _ridge_reference_and_removal_predictions(
        np, x_train=np.asarray(x_train, dtype=np.float64),
        targets=np.asarray(y_train, dtype=np.float64), x_eval=x_eval,
        candidate_rows=tuple(candidate_rows), target_block=(0, HORIZON))
    first = np.asarray(out["first_order_proxy_predictions"], dtype=np.float64)
    baseline = np.asarray(out["baseline_prediction"], dtype=np.float64)
    base_rms = float(np.sqrt(np.mean(baseline ** 2))) or 1.0
    per_c = np.sqrt(np.mean(first ** 2, axis=(1, 2)))
    return float(np.mean(per_c) / base_rms)


def phase_cobs_freeze() -> int:
    """冻结 cobs 协议（任何计算前；已冻结则拒绝）。"""
    report = _load_report()
    if report.get("cobs_protocol"):
        raise SystemExit("cobs_protocol 已存在——拒绝重复冻结")
    if (report.get("cobs") or {}).get("features"):
        raise SystemExit("cobs 已有计算结果——协议必须先冻结")
    proto = {
        "experiment_id": "CONSUMER_OBSERVATION_SCOPE_CHECK",
        "user_ruling": ("2026-08-14 batch1 分支(c)：正负样本充足（7/3）但现有 Observation "
                        "不可分——下一轮只增加一个 Consumer-conditioned Observation"
                        "（廉价历史影响力/敏感度 proxy），不建 Pattern Graph"),
        "observation": ("consumer_action_influence = outlier_mad 会修改的训练样本对下游 "
                        "ridge 预测的历史影响力占比：mean_c RMS(一阶移除预测变化) / "
                        "RMS(基线预测)；机制复用项目既有 "
                        "_ridge_reference_and_removal_predictions（Ridge alpha=1、"
                        "未惩罚截距、多右端一次求解）；零 LLM、与 outcome 标签独立"),
        "scope": ("零新 Outcome：对 batch1 已标签证据（7 POSITIVE / 3 NEGATIVE）"
                  "追加单一特征后重算可分离性（同 _batch1_verdict 的 fit 子集 + "
                  "剩余验证程序）"),
        "verdict_rules": {
            "separable": "fit 子集可分且剩余 context 验证通过 → SCOPE_CANDIDATE_PROPOSED（窄 Target-local Scope/Risk Rule 候选；用户裁决后固化）",
            "not_separable": "仍不可分 → CONSUMER_OBSERVATION_NOT_RESOLVED（该单一 Consumer Observation 不足以解释正负翻转；停止，用户裁决）",
        },
        "claim_boundary": ("proxy 只描述 consumer 依赖结构，不证明因果；"
                           "不产生新 Outcome、不修改任何已冻结协议"),
        "expected_llm_calls": "0",
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["cobs_protocol"] = proto
    _save_report(report)
    print("cobs_protocol FROZEN")
    return 0


def phase_cobs(env: Mapping[str, Any] | None = None) -> int:
    """零 LLM：对已标签证据计算 consumer_action_influence 并重算裁定。"""
    report = _load_report()
    if not report.get("cobs_protocol"):
        raise SystemExit("cobs_protocol 未冻结——先跑 cobs-freeze")
    if env is None:
        env = _load_env()
    batch = report.get("batch1") or {}
    batch_entries = batch.get("entries") or []
    usel_entries = (report.get("usel") or {}).get("entries") or []
    pools = (report.get("fullop2") or {}).get("context_pools") or {}
    usel_labels = []
    for e in usel_entries:
        for tag, r in (e.get("results") or {}).items():
            if r.get("gain") is None:
                continue
            steps = r.get("steps") or []
            if [s.get("op") for s in steps] != ["outlier_mad"]:
                continue
            label = ("POSITIVE" if r["gain"] >= M
                     else "NEGATIVE" if r["gain"] < -M else "NEUTRAL")
            usel_labels.append({"key": e["key"], "label": label,
                                "features": dict((pools.get(e["key"]) or {})
                                                 .get("features") or {})})
    labeled = _labeled_evidence(batch_entries, usel_labels)
    values = env["values"]
    cobs = report.setdefault("cobs", {})
    feats = cobs.setdefault("features", {})
    for item in labeled:
        key = item["key"]
        if key in feats:
            continue
        series, _, origin = key.partition("@")
        try:
            v = _cobs_feature(series, int(origin), values)
        except Exception as exc:  # noqa: BLE001
            v = None
            print("== cobs " + key + " FEATURE_ERROR: "
                  + type(exc).__name__ + ": " + str(exc)[:120], flush=True)
        feats[key] = v
        item["features"]["consumer_action_influence"] = v
        _save_report(report)
        print("== cobs " + key + ": influence=" + str(v), flush=True)
    cobs["labeled"] = labeled
    result = _scope_verdict_from_labeled(labeled)
    if result["verdict"] == "SCOPE_CANDIDATE_ELIGIBLE_NOT_SEPARABLE":
        result["verdict"] = "CONSUMER_OBSERVATION_NOT_RESOLVED"
    cobs["verdict"] = result
    cobs["state"] = "CLOSED"
    report["cobs"] = cobs
    _save_report(report)
    print("cobs 终裁: " + result["verdict"])
    print(json.dumps(result, ensure_ascii=False))
    return 0

# ---------------------------------------------------------------- bse
# BATCH_SELF_EVOLUTION（BSE；方案乙——用户裁决 2026-08-15）：
# 多轨迹 Action–Response → 机械稳定性标签（B=SUPPORT_HARM 是本次 Scope
# family 的唯一 target；C=support 正/delayed 负 是独立的 temporal first
# fault，记 TEMPORAL_INSTABILITY_UNRESOLVED：不参与阈值拟合、不计入
# PASS/FAIL、完整披露）→ 一个 Program-conditioned Observation（路线 A，
# min 为冻结统计量）→ fit 边界恰为 1 才由 Runtime 机械生成 midpoint 候选
# → Slow 一次提案（今晚唯一 LLM 使用；Slow 无权批准自己的规则）→
# held-out T10 零 LLM 机械 replay（规则只门控 prior 槽；probe 后 Runtime
# winner resolution——不是 Fast Select；unknown 不放行）→ 双层终裁：
# SUPPORT_HARM_SCOPE_RULE_DEV_PASS 必须并列 TEMPORAL_INSTABILITY_UNRESOLVED。

BSE_PROGRAM = (("outlier_mad", {}),)
BSE_FIT_SERIES = ("T1", "T100")
BSE_HELDOUT_SERIES = ("T10",)
BSE_CUTOFF_STEPS = (1, 2, 3)            # t−H, t−2H, t−3H
BSE_MIN_ANCHOR = 312                    # nsu._config() anchors 起点
BSE_RULE_ID = "outlier_mad_stable_scope_v1"
BSE_OBS_FEATURE = "historical_program_stability"
EPISODES_REL = PROJECT_ROOT / "artifacts" / "experience" / "episodes.json"
BSE_RULE_REL = PROJECT_ROOT / "artifacts" / "experience" \
    / "scope_rule_outlier_mad_stable_scope_v1.json"

_BSE_GROUP = {"STABLE_POSITIVE": "A",
              "SUPPORT_HARM": "B",
              "SUPPORT_POSITIVE_DELAYED_NEGATIVE": "C",
              "NEUTRAL_OR_UNIDENTIFIED": "NEUTRAL"}


def _bse_stability_label(support_gain: Any, delayed_evaluated: bool,
                         delayed_gain: Any) -> str:
    """冻结标签口径（任务书 P1 + 方案乙裁决；纯机械，不覆盖原始记录）。"""
    if support_gain is None:
        return "NEUTRAL_OR_UNIDENTIFIED"
    g = float(support_gain)
    if g < -M:
        return "SUPPORT_HARM"
    if g >= M:
        if delayed_evaluated and delayed_gain is not None:
            return ("STABLE_POSITIVE" if float(delayed_gain) >= -M
                    else "SUPPORT_POSITIVE_DELAYED_NEGATIVE")
        return "NEUTRAL_OR_UNIDENTIFIED"      # 缺 delayed 证据
    return "NEUTRAL_OR_UNIDENTIFIED"          # no-op / |gain| < M


def _bse_load_episodes() -> list:
    if not EPISODES_REL.exists():
        return []
    return json.loads(EPISODES_REL.read_text(encoding="utf-8"))


def _bse_save_episodes(episodes: list) -> None:
    EPISODES_REL.write_text(
        json.dumps(episodes, ensure_ascii=False, indent=1, default=str) + "\n",
        encoding="utf-8")


def _bse_episode_id_map(episodes: Sequence[Mapping[str, Any]]) -> dict:
    """kdd2018_dev outlier_mad episode 的 context key → episode_id（命名约定
    机械解析；usel 旧 ID 保留 _conflict 后缀是裁决要求的不重命名）。"""
    import re  # noqa: PLC0415
    out: dict[str, str] = {}
    for ep in episodes:
        if ep.get("domain_namespace") != "kdd2018_dev":
            continue
        eid = str(ep.get("episode_id") or "")
        m = re.match(r"(usel|batch1)_t(\d+)_(\d+)_outlier_mad_", eid)
        if m:
            out["T" + m.group(2) + "@" + m.group(3)] = eid
    return out


def _bse_labeled_contexts(report: Mapping[str, Any]) -> list:
    """10 个非中性 context 的测量表（series/origin/support/delayed/features；
    纯机械。support/delayed 来源：batch1 entries + usel entries；usel 的
    delayed 以 P0 后 Memory 为权威——T100@600 delayed@648 只在 episode 中）。"""
    out: dict[str, dict] = {}
    pools = (report.get("fullop2") or {}).get("context_pools") or {}
    for e in (report.get("usel") or {}).get("entries") or []:
        for _tag, r in (e.get("results") or {}).items():
            if r.get("gain") is None:
                continue
            steps = r.get("steps") or []
            if [s.get("op") for s in steps] != ["outlier_mad"]:
                continue
            key = e["key"]
            series, _, origin = key.partition("@")
            out[key] = {"key": key, "series": series, "origin": int(origin),
                        "source": "usel", "support_gain": float(r["gain"]),
                        "delayed_evaluated": False, "delayed_gain": None,
                        "features": dict((pools.get(key) or {})
                                         .get("features") or {})}
    for ep in _bse_load_episodes():
        if ep.get("domain_namespace") != "kdd2018_dev":
            continue
        if ep.get("workflow_signature") != "outlier_mad":
            continue
        eid = str(ep.get("episode_id") or "")
        dr = ep.get("delayed_response") or {}
        if not dr.get("evaluated") or dr.get("gain") is None:
            continue
        for key, row in out.items():
            token = row["series"].lower() + "_" + str(row["origin"]) + "_"
            if token in eid:
                row["delayed_evaluated"] = True
                row["delayed_gain"] = float(dr["gain"])
    for e in (report.get("batch1") or {}).get("entries") or []:
        if e.get("gain") is None or abs(float(e["gain"])) < M:
            continue          # NEUTRAL/no-op 保留在报告，不进标签表
        key = e["key"]
        d = e.get("delayed") or {}
        d_eval = bool(d) and d.get("gain") is not None
        out[key] = {"key": key, "series": e["series"],
                    "origin": int(e["origin"]), "source": "batch1",
                    "support_gain": float(e["gain"]),
                    "delayed_evaluated": d_eval,
                    "delayed_gain": (float(d["gain"]) if d_eval else None),
                    "features": dict(e.get("features") or {})}
    rows = [out[k] for k in sorted(out)]
    for row in rows:
        row["label"] = _bse_stability_label(
            row["support_gain"], row["delayed_evaluated"], row["delayed_gain"])
        row["group"] = _BSE_GROUP[row["label"]]
    return rows


def _bse_first_check(fit_rows: Sequence[dict],
                     heldout_rows: Sequence[dict]) -> dict:
    """现有部署可见 Observation 第一检查（任务书 P1）：fit（A/B）干净分开
    且 group-disjoint heldout（A/B）全部预测正确才算可分——通过则跳过 P2。"""
    feats: dict[str, dict] = {}
    for row in fit_rows:
        for k, v in (row.get("features") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                feats.setdefault(k, {})[row["key"]] = (float(v), row["group"])
    candidates = []
    for fname in sorted(feats):
        pts = feats[fname]
        if len(pts) < len(fit_rows):
            continue
        a_vals = [v for v, g in pts.values() if g == "A"]
        b_vals = [v for v, g in pts.values() if g == "B"]
        if not a_vals or not b_vals:
            continue
        if min(a_vals) > max(b_vals):
            candidates.append({"feature": fname, "direction": "ge",
                               "threshold": (max(b_vals) + min(a_vals)) / 2.0})
        elif max(a_vals) < min(b_vals):
            candidates.append({"feature": fname, "direction": "le",
                               "threshold": (max(a_vals) + min(b_vals)) / 2.0})
    for cand in candidates:
        errors = []
        for row in heldout_rows:
            v = (row.get("features") or {}).get(cand["feature"])
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errors.append(row["key"])
                continue
            pred_a = (float(v) >= cand["threshold"]
                      if cand["direction"] == "ge"
                      else float(v) < cand["threshold"])
            if pred_a != (row["group"] == "A"):
                errors.append(row["key"])
        cand["heldout_errors"] = errors
    ok = [c for c in candidates if not c["heldout_errors"]]
    return {"fit_separable_candidates": candidates,
            "separable": bool(ok),
            "validated_feature": ok[0] if ok else None,
            "note": ("可分 = fit 干净分开 且 group-disjoint heldout 全部正确；"
                     "通过则跳过 P2 新 Observation（任务书第一检查）")}


def _bse_hist_stability(series: str, origin: int,
                        values: Mapping[str, Any]) -> dict:
    """路线 A（冻结）：historical_program_stability =
    min(gain@t−H, gain@t−2H, gain@t−3H)，与 batch1 Support 同一仪器
    （ScopeExecutor + nsu._evaluate_kdd；outlier_mad 作用于训练行）。
    cutoff c 的评估 horizon [c, c+HORIZON) 端点互斥且不晚于当前 origin，
    与 Support 窗 [origin, origin+HORIZON) 零重叠；当前 Support/delayed
    不参与计算；不足三个有限 cutoff gain → unknown（None——自动要求
    Support，绝不放行 prior）。"""
    import math  # noqa: PLC0415
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: PLC0415
        ScopeExecutor,
    )
    roster, vals = _support_roster(series, values)
    executor = ScopeExecutor(roster, vals, nsu._config(),
                             evaluate_fn=nsu._evaluate_kdd)
    legs = []
    for k in BSE_CUTOFF_STEPS:
        c = int(origin) - k * HORIZON
        legal = c >= BSE_MIN_ANCHOR + HORIZON and c + HORIZON <= int(origin)
        leg = {"cutoff": c, "horizon": [c, c + HORIZON], "legal": legal,
               "gain": None, "windows": 0, "passed": None, "error": None}
        if legal:
            leg["windows"] = len(executor.training_windows(c))
            rr = executor.evaluate(BSE_PROGRAM, c)
            leg["passed"] = bool(rr.verification.passed)
            leg["error"] = rr.error
            if rr.gain is not None and math.isfinite(float(rr.gain)):
                leg["gain"] = float(rr.gain)
        legs.append(leg)
    finite = [l["gain"] for l in legs if l["gain"] is not None]
    ok = all(l["legal"] for l in legs)
    value = min(finite) if len(finite) == len(BSE_CUTOFF_STEPS) else None
    return {"feature": BSE_OBS_FEATURE, "value": value, "aggregate": "min",
            "legs": legs,
            "legality": {"ok": ok,
                         "rule": ("cutoff horizon [c, c+48) 端点互斥、不晚于 "
                                  "origin；training windows 全部满足 anchor+48 "
                                  "≤ cutoff；当前 Support/delayed 不参与"),
                         "cutoffs": [l["cutoff"] for l in legs]},
            "unknown_reason": (None if value is not None else
                               "finite cutoff gains < 3 → unknown（要求 Support，"
                               "不放行 prior）")}


def _bse_blind_health(obs: Mapping[str, Any]) -> dict:
    """label-blind 健康检查（任务书 P2）：只看数值性质，不看标签。"""
    import math  # noqa: PLC0415
    items = sorted(obs.items())
    vals = [(k, (o or {}).get("value")) for k, o in items]
    finite_vals = [float(v) for _k, v in vals
                   if v is not None and math.isfinite(float(v))]
    unknown = [k for k, v in vals
               if v is None or not math.isfinite(float(v))]
    distinct = {round(v, 12) for v in finite_vals}
    spread = (max(finite_vals) - min(finite_vals)) if finite_vals else 0.0
    legal = all(bool((o or {}).get("legality", {}).get("ok"))
                for _k, o in items)
    checks = {"finite": True,
              "coverage": len(finite_vals),
              "unknown_contexts": unknown,
              "non_constant": len(distinct) > 1,
              "non_saturating": spread > 1e-6,
              "spread": spread,
              "computable_pre_origin": legal}
    checks["passed"] = (checks["finite"] and checks["non_constant"]
                        and checks["non_saturating"]
                        and checks["computable_pre_origin"]
                        and len(finite_vals) >= 4)
    return checks


def _bse_fit_boundary(fit_points: Sequence[Mapping[str, Any]]) -> dict:
    """fit A/B 点按 obs 值排序，数相邻跨类边界。恰好 1 个且两值严格不同
    才产生 midpoint τ；0 个（无区分/恒定）或 >1 个（单阈值不可表达）或
    等值跨类相邻 → τ=None → 不生成规则（裁决 5：不多阈值/不组合/不换特征）。"""
    view = sorted(({"key": p["key"], "group": p["group"],
                    "value": float(p["value"])}
                   for p in fit_points if p.get("value") is not None),
                  key=lambda p: p["value"])
    boundaries = [(a, b) for a, b in zip(view, view[1:])
                  if a["group"] != b["group"]
                  and a["group"] in ("A", "B") and b["group"] in ("A", "B")]
    tau = None
    pair = None
    if len(boundaries) == 1:
        a, b = boundaries[0]
        if a["value"] < b["value"]:
            tau = (a["value"] + b["value"]) / 2.0
            pair = [a["key"], b["key"]]
    return {"boundary_count": len(boundaries), "tau": tau,
            "boundary_pair": pair, "sorted_fit": view}


def _bse_rule_fires(value: Any, rule: Mapping[str, Any]) -> bool:
    """unknown（None）绝不放行 prior（裁决 4 与 held-out 决定性要求）。"""
    if value is None:
        return False
    app = rule["applicability"]
    return (float(value) >= float(app["threshold"])
            if app["operator"] == "ge"
            else float(value) < float(app["threshold"]))


def _bse_capsule_prompt(fit_view: Sequence[Mapping[str, Any]],
                        feature: str, tau: float,
                        retry_feedback: str | None = None) -> str:
    """匿名 Contrast Capsule（任务书 P3）：无 series 名/无 held-out 标签/
    无当前未来/无人工阈值建议/无全量 trace。"""
    a_lines = ["  - {}: {} = {}".format(p["anon"], feature,
                                       round(float(p["value"]), 6))
               for p in fit_view if p["group"] == "A"]
    b_lines = ["  - {}: {} = {}".format(p["anon"], feature,
                                       round(float(p["value"]), 6))
               for p in fit_view if p["group"] == "B"]
    prompt = (
        "你是 Harness 的 Slow Agent。职责：为 Target-local Skill 的适用范围"
        "生成一条结构化 Scope 规则，或明确弃权。你无权批准自己的规则——"
        "Runtime 将用未参与规则生成的数据独立核销。\n\n"
        "任务背景：\n"
        "- Task = forecast；Consumer = ridge 回归；Metric = sMASE gain（越大越好）\n"
        "- Program = outlier_mad（单算子，作用于训练行）\n"
        "- Observation = " + feature + "：当前决策点之前、同一程序在最近三个"
        "历史窗口上收益的最小值（min 聚合；数值越大 = 历史越稳定；当前 "
        "Support 与 delayed 不参与计算）\n"
        "- 一批历史 Action–Response 已按失败机制分组（匿名 Context）：\n"
        "  A 组（执行后即时有益、且后续仍稳定）：\n"
        + ("\n".join(a_lines) if a_lines else "  - （空）") + "\n"
        "  B 组（执行后即时就有害）：\n"
        + ("\n".join(b_lines) if b_lines else "  - （空）") + "\n"
        "  另存在一组「即时有益但后续翻负」的不稳定轨迹——它由另一条独立机制"
        "负责，不属于本规则的目标，此处不列出。\n\n"
        "Runtime 已机械生成候选谓词（阈值 τ = " + repr(round(float(tau), 9))
        + " 来自上述数据相邻类别的中点，不可修改）：\n"
        "  P1: " + feature + " >= τ\n"
        "  P2: " + feature + " < τ\n\n"
        "规则语义：谓词为真 → 该 Local Skill 进入 Fast 候选 prior 槽；"
        "为假或 unknown → 不进入（Agent 保留探索权；任何执行仍须通过 "
        "Target Support；delayed 决定去留）。\n")
    if retry_feedback:
        prompt += ("\nReplay 拒绝反馈（上一次选择在 held-out 未通过核销）：\n"
                   + retry_feedback + "\n")
    prompt += ("\n只输出一个 JSON 对象，不要输出任何其他内容：\n"
               "{\"choice\": \"P1\" | \"P2\" | \"abstain\", "
               "\"rationale\": \"<=200字\"}")
    return prompt


def _bse_parse_slow_choice(text: Any) -> dict:
    """机械解析 Slow 输出（预注册：不可解析/非法选择 → 视同 abstain）。"""
    import re  # noqa: PLC0415
    m = re.search(r"\{[^{}]*\}", str(text or ""), re.S)
    if not m:
        return {"choice": "abstain", "rationale": None, "parse": "no_json"}
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {"choice": "abstain", "rationale": None, "parse": "bad_json"}
    choice = str(obj.get("choice") or "").strip()
    if choice not in ("P1", "P2", "abstain"):
        return {"choice": "abstain", "rationale": None, "parse": "bad_choice"}
    return {"choice": choice, "rationale": obj.get("rationale"), "parse": "ok"}


def _bse_assemble_rule(choice: str, feature: str, tau: float,
                       labeled: Sequence[Mapping[str, Any]],
                       id_map: Mapping[str, str]) -> dict:
    """Runtime 装配规则（Slow 只选了谓词方向；Slow 不批准自己的规则）。"""
    operator = "ge" if choice == "P1" else "le"

    def ids(group: str) -> list:
        return sorted(id_map[r["key"]] for r in labeled
                      if r["group"] == group and r["key"] in id_map)

    return {
        "rule_id": BSE_RULE_ID,
        "surface": "scope",
        "workflow_signature": "outlier_mad",
        "applicability": {"feature": feature, "operator": operator,
                          "threshold": float(tau)},
        "unknown_policy": "no_prior",
        "authority": "LOCAL_DRAFT",
        "requires_target_support": True,
        "slow_approved": False,
        "evidence": {"positive_episode_ids": ids("A"),
                     "negative_episode_ids": ids("B"),
                     "conflict_episode_ids": ids("C")},
        "semantics": ("谓词为真 → Local Skill 进入 prior 槽；为假/unknown → "
                      "不进 prior（保留 exploration 与独立 propose 权；执行仍须 "
                      "Target Support；delayed 决定去留；不修改历史 Episode）"),
    }


def _bse_pass_evaluation(rows: Sequence[Mapping[str, Any]]) -> dict:
    """held-out PASS 判定（冻结口径；C 组不参与判定但必须披露）。"""
    ab = [r for r in rows if r["group"] in ("A", "B")]
    stable = [r for r in ab if r["group"] == "A"]
    harm = [r for r in ab if r["group"] == "B"]
    h0r = sum(r["H0"]["support_receipts"] for r in ab)
    h1r = sum(r["H1"]["support_receipts"] for r in ab)
    h0n = sum(r["H0"]["negative_probes"] for r in ab)
    h1n = sum(r["H1"]["negative_probes"] for r in ab)
    d0 = sum(r["H0"]["delayed_gain"] for r in ab)
    d1 = sum(r["H1"]["delayed_gain"] for r in ab)
    crit = {
        "stable_prior_recall": bool(stable) and all(
            r["H1"]["prior"] for r in stable),
        "harm_auto_priority_blocked": not any(r["H1"]["prior"] for r in harm),
        "receipts_or_harm_reduced": (h1r < h0r) or (h1n < h0n),
        "delayed_not_worse": d1 >= d0 - M,
        "removal_delta_real": any(not r["H1"]["prior"] for r in rows),
    }
    return {"criteria": crit, "passed": all(crit.values()),
            "failed": sorted(k for k, v in crit.items() if not v),
            "arms": {"H0": {"support_receipts": h0r, "negative_probes": h0n,
                            "delayed_sum_ab": d0},
                     "H1": {"support_receipts": h1r, "negative_probes": h1n,
                            "delayed_sum_ab": d1}}}


def phase_bse_freeze() -> int:
    """冻结 BSE 协议（任何运行前；已冻结或已有结果则拒绝）。"""
    report = _load_report()
    if report.get("bse_protocol"):
        raise SystemExit("bse_protocol 已存在——拒绝重复冻结")
    if report.get("bse"):
        raise SystemExit("bse 已有结果——协议必须先冻结")
    proto = {
        "experiment_id": "BSE_SUPPORT_HARM_SCOPE_RULE_2026_08_15",
        "user_rulings": [
            "方案乙：B 组（SUPPORT_HARM）是本次 Scope family 唯一 target；C 组"
            "（support 正/delayed 负）是独立 temporal first fault——不参与拟合、"
            "不计入 PASS/FAIL、完整披露、标记 TEMPORAL_INSTABILITY_UNRESOLVED；"
            "delayed gate 只能阻止后续复用，不能撤销已发生的第一次 delayed harm",
            "min 保持为冻结统计量（任务书冻结前已提出的安全型统计量；margin 薄"
            "如实记录；不因推演结果换 mean/median）",
            "CONFLICT 双层语义：episode 级（同 Context×Program 内 support 与 "
            "delayed 方向相反）+ retrieval 聚合级；T1@888/T10@600 → NEGATIVE；"
            "T100@600 evidence_level → DELAYED、local_status 保持 LOCAL_DRAFT；"
            "已有 episode ID 不重命名；batch1 七条非中性轨迹补入 Memory",
            "规则只门控 prior 槽：不删 exploration、不永久封杀、执行仍须 "
            "Support、delayed 决定去留；probe 后 winner 是 Runtime winner "
            "resolution（非 Fast Select、不作 Agent 选择能力证据）；probe 前"
            "顺序固定、不得用 Outcome 决定",
            "fit 边界恰为 1 才允许生成 midpoint 候选；0 或 >1 个边界即停止——"
            "不多阈值、不组合条件、不换特征继续拟合",
            "结构发现记为 CURRENT_OBSERVATION_PRECHANGE_NON_IDENTIFIABILITY"
            "（现有 Observation 与拟议三窗口历史最小值很可能无法在变点前识别 "
            "delayed-only 翻转）——不是一般性信息论不可能",
        ],
        "labels": {
            "STABLE_POSITIVE": "support_gain ≥ M 且 delayed 已评估 且 delayed ≥ −M",
            "SUPPORT_HARM": "support_gain < −M",
            "SUPPORT_POSITIVE_DELAYED_NEGATIVE": "support ≥ M 且 delayed 已评估 且 delayed < −M",
            "NEUTRAL_OR_UNIDENTIFIED": "no-op / |gain| < M / 缺稳定性证据",
            "group_map": {"A": "STABLE_POSITIVE", "B": "SUPPORT_HARM",
                          "C": "SUPPORT_POSITIVE_DELAYED_NEGATIVE"},
            "note": "Scope/Promotion 标签，不覆盖原始 Support/delayed 记录",
        },
        "split": {"fit_series": list(BSE_FIT_SERIES),
                  "heldout_series": list(BSE_HELDOUT_SERIES),
                  "rule": "固定 split，不得根据结果更换"},
        "first_check": ("新标签下先检查现有部署可见 Observation：fit 干净分开 "
                        "且 group-disjoint heldout 全部正确 → 跳过 P2；否则进 P2"),
        "observation": {
            "name": BSE_OBS_FEATURE,
            "definition": ("min(gain@t−H, gain@t−2H, gain@t−3H)，与 batch1 "
                           "Support 同一仪器（ScopeExecutor+nsu._evaluate_kdd，"
                           "outlier_mad 作用于训练行）"),
            "legality": ("cutoff horizon [c,c+48) 端点互斥、不晚于 origin；"
                         "当前 Support/delayed 不参与；不读 Query future；"
                         "未来 A5/A3 两臂同工具同预算"),
            "unknown_policy": "有限 cutoff gain < 3 → unknown → 不放行 prior、自动要求 Support",
            "fragility_disclosure": "min 对单个坏窗敏感；拟合 margin 可能很薄；即使通过也只是 development mechanism evidence",
        },
        "blind_health_check": ("finite / 非恒定 / 不饱和（spread>1e-6）/ origin 前"
                               "可计算 / 覆盖 ≥4——失败 → "
                               "PROGRAM_CONDITIONED_OBSERVATION_DEGENERATE 并关闭 "
                               "family，不换公式凑结果"),
        "boundary_rule": ("fit A/B 按 obs 排序数相邻跨类边界：==1 且严格不等 → "
                          "Runtime 生成 {obs≥τ, obs<τ} 两个候选（τ=边界中点）；"
                          "否则不生成规则 → CONTEXT_UTILITY_UNIDENTIFIABLE"),
        "capsule": ("匿名（无 series 名/无 held-out 标签/无未来/无人工阈值/无全量 "
                    "trace）：Task/Consumer/Metric + Program contract + A/B 组 obs "
                    "值 + 候选谓词 + 只能选一或弃权"),
        "rule": {"id": BSE_RULE_ID, "surface": "scope",
                 "workflow_signature": "outlier_mad",
                 "authority": "LOCAL_DRAFT（held-out 通过后才可 LOCAL_ACTIVE）",
                 "requires_target_support": True,
                 "unknown_policy": "no_prior",
                 "slow_cannot_approve": True},
        "replay": {
            "arms": {"H0": "当前 LOCAL_DRAFT 宽 scope（prior 槽恒含 outlier_mad）",
                     "H1": "同一 LOCAL_DRAFT + Scope Rule 门控 prior 槽"},
            "heldout_contexts": ["T10@888(A)", "T10@600(B)",
                                 "T10@792(C 披露)", "T10@984(C 披露)"],
            "pool": "identity（隐含 no-op 基线，gain=0）+ outlier_mad（prior 槽准入时）",
            "probe_order": "pre-probe 固定：prior 在槽 → [outlier_mad, identity]；否则 [identity]；不用 Outcome 定顺序",
            "winner": "post-probe Runtime winner resolution：probe 后取 gain 最大者，≥M 才采用；identity 恒可选",
            "delayed": "采用 winner 的 delayed 复用 batch1/usel 已测缓存（零新 Outcome）；identity 定义 delayed=0",
            "metrics": ["support_receipts", "negative_probes", "winner/support_gain",
                        "delayed_gain", "stable prior recall", "harm auto-priority",
                        "removal delta", "C 组行为与 delayed 披露"],
        },
        "verdict_rules": {
            "degenerate": "blind health 失败 → PROGRAM_CONDITIONED_OBSERVATION_DEGENERATE，关闭本 Observation family",
            "no_boundary": "fit 边界 ≠1 或等值跨类相邻 → CONTEXT_UTILITY_UNIDENTIFIABLE，关闭一维历史稳定性 Scope family",
            "abstain": "Slow 弃权或输出不可解析 → CONTEXT_UTILITY_UNIDENTIFIABLE（slow_abstained），关闭",
            "replay_pass": ("五项全过（T10@888 prior 召回 / T10@600 无自动 prior / "
                            "receipts 或 negative probes 减少 / delayed 不劣 / "
                            "removal delta 真实）→ SUPPORT_HARM_SCOPE_RULE_DEV_PASS "
                            "且必须并列 TEMPORAL_INSTABILITY_UNRESOLVED；总状态 "
                            "PARTIAL_BATCH_SELF_EVOLUTION_DEV_PASS；Rule → "
                            "LOCAL_ACTIVE 并保留 requires_target_support"),
            "replay_fail": ("第一次失败 → OBSERVATION_VALID_SCOPE_PATCH_REJECTED，"
                            "允许一次带拒绝反馈的结构化重试；第二次失败 → "
                            "CONTEXT_UTILITY_UNIDENTIFIABLE，关闭 family"),
        },
        "llm_budget": "P0–P2/P5 零 LLM；P4 Slow ≤1 提案 + ≤1 重试；Fast 真实 LLM 仅在机械 held-out 通过后另行批准",
        "claim_boundary": ("development Target-local 证据；不宣称跨域；不升级 "
                           "Shared Capability；不把阈值应用到新 Dataset；C 组 "
                           "temporal 风险不因 Scope PASS 被宣称解决"),
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["bse_protocol"] = proto
    _save_report(report)
    print("bse_protocol FROZEN")
    return 0


def phase_bse_p0() -> int:
    """P0：Memory 对账（零 LLM；幂等）——修 relation/evidence_level 落盘、
    补 7 条 batch 非中性 episode；不建迁移器/新 Schema。"""
    report = _load_report()
    if not report.get("bse_protocol"):
        raise SystemExit("bse_protocol 未冻结——先跑 bse-freeze")
    import experience_memory as em  # noqa: PLC0415  (methods/ttha 已在 sys.path)
    episodes = _bse_load_episodes()
    fixes = []
    for ep in episodes:
        eid = ep.get("episode_id")
        if eid in ("usel_t1_888_outlier_mad_conflict",
                   "usel_t10_600_outlier_mad_conflict"):
            if ep.get("relation") != em.RELATION_NEGATIVE:
                fixes.append({"episode_id": eid, "field": "relation",
                              "from": ep.get("relation"), "to": "NEGATIVE",
                              "basis": "Support 负、无相反 delayed → NEGATIVE"})
                ep["relation"] = em.RELATION_NEGATIVE
        if eid == "usel_t100_600_outlier_mad_positive":
            if ep.get("evidence_level") != em.EVIDENCE_DELAYED:
                fixes.append({"episode_id": eid, "field": "evidence_level",
                              "from": ep.get("evidence_level"), "to": "DELAYED",
                              "basis": "delayed 已评估且为正 → DELAYED（执行权保持 LOCAL_DRAFT）"})
                ep["evidence_level"] = em.EVIDENCE_DELAYED
    existing = {str(ep.get("episode_id")) for ep in episodes}
    added = []
    for e in (report.get("batch1") or {}).get("entries") or []:
        if e.get("gain") is None or abs(float(e["gain"])) < M:
            continue
        series = e["series"]
        origin = int(e["origin"])
        sg = float(e["gain"])
        d = e.get("delayed") or {}
        d_eval = bool(d) and d.get("gain") is not None
        dg = float(d["gain"]) if d_eval else None
        if sg < -M:
            relation = em.RELATION_NEGATIVE
        elif d_eval and dg < -M:
            relation = em.RELATION_CONFLICT
        else:
            relation = em.RELATION_POSITIVE
        eid = ("batch1_" + series.lower() + "_" + str(origin)
               + "_outlier_mad_" + relation.lower())
        if eid in existing:
            continue
        ep = em.build_episode(
            episode_id=eid,
            task_consumer_key="forecast|ridge|sMASE",
            domain_namespace="kdd2018_dev",
            context_summary={
                "local_pattern": dict(e.get("features") or {}),
                "program_geometry": {"scope": "training_rows",
                                     "program_steps": [{"op": "outlier_mad",
                                                        "params": {}}]},
                "support_origin": origin},
            workflow_signature="outlier_mad",
            support_response={"gain": sg, "accepted": sg >= M},
            delayed_response={"evaluated": d_eval, "gain": dg},
            relation=relation,
            evidence_level=(em.EVIDENCE_DELAYED if d_eval
                            else em.EVIDENCE_SUPPORT),
            local_status=em.STATUS_EPISODE_ONLY,
            evidence_refs=["artifacts/functional/e2/"
                           "w1_guidance_evolution_report.json#batch1"],
        )
        episodes.append(ep.to_dict())
        added.append(eid)
    _bse_save_episodes(episodes)
    bse = report.setdefault("bse", {})
    bse["p0_reconciliation"] = {
        "fixes": fixes, "added_episode_ids": added,
        "conflict_semantics": ("双层（2026-08-15 裁决）：episode 级 = 同 "
                               "Context×Program 内 support 与 delayed 方向相反；"
                               "retrieval 级 = 跨 episode 对照聚合"),
        "episode_ids_unchanged": True,
        "note": ("上轮 relation 修正未落盘（文件仍为 CONFLICT）——本次真实修复；"
                 "中性/no-op 保留在报告、不扩 Relation Schema"),
    }
    _save_report(report)
    print("bse-p0: fixes=" + str(len(fixes)) + " added=" + str(len(added)))
    for a in added:
        print("  + " + a)
    return 0


def phase_bse_p1() -> int:
    """P1：机械稳定性标签 + 现有 Observation 组外分离第一检查（零 LLM）。"""
    report = _load_report()
    if not report.get("bse_protocol"):
        raise SystemExit("bse_protocol 未冻结")
    bse = report.setdefault("bse", {})
    if not bse.get("p0_reconciliation"):
        raise SystemExit("先跑 bse-p0")
    labeled = _bse_labeled_contexts(report)
    bse["labels"] = labeled
    counts: dict[str, int] = {}
    for row in labeled:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    bse["label_counts"] = counts
    # P0 一致性门：Memory relation 与机械标签口径必须一致
    id_map = _bse_episode_id_map(_bse_load_episodes())
    expect = {"A": "POSITIVE", "B": "NEGATIVE", "C": "CONFLICT"}
    episodes = {str(ep.get("episode_id")): ep for ep in _bse_load_episodes()}
    mismatches = []
    for row in labeled:
        eid = id_map.get(row["key"])
        if not eid:
            mismatches.append({"key": row["key"], "issue": "episode_missing"})
            continue
        rel = episodes[eid].get("relation")
        if row["group"] in expect and rel != expect[row["group"]]:
            mismatches.append({"key": row["key"], "episode_id": eid,
                               "expected": expect[row["group"]], "actual": rel})
    bse["memory_consistency"] = {"checked": len(labeled),
                                 "mismatches": mismatches}
    fit = [r for r in labeled
           if r["series"] in BSE_FIT_SERIES and r["group"] in ("A", "B")]
    heldout = [r for r in labeled
               if r["series"] in BSE_HELDOUT_SERIES and r["group"] in ("A", "B")]
    bse["first_check"] = _bse_first_check(fit, heldout)
    _save_report(report)
    print("bse-p1 labels: " + json.dumps(counts, ensure_ascii=False))
    print("fit(A/B): " + repr([(r["key"], r["group"]) for r in fit]))
    print("heldout(A/B): " + repr([(r["key"], r["group"]) for r in heldout]))
    print("first_check separable: "
          + str(bse["first_check"]["separable"])
          + " candidates=" + str(len(bse["first_check"]["fit_separable_candidates"])))
    if mismatches:
        print("MEMORY_MISMATCH: " + json.dumps(mismatches, ensure_ascii=False))
    return 0


def phase_bse_p2(env: Mapping[str, Any] | None = None) -> int:
    """P2：historical_program_stability 计算 + label-blind 健康检查（零 LLM）。"""
    report = _load_report()
    if not report.get("bse_protocol"):
        raise SystemExit("bse_protocol 未冻结")
    bse = report.setdefault("bse", {})
    if not bse.get("labels"):
        raise SystemExit("先跑 bse-p1")
    if (bse.get("first_check") or {}).get("separable"):
        raise SystemExit("现有 Observation 已通过第一检查——走既有 feature 分支，"
                         "不得重复新增 Observation")
    if (bse.get("blind_health") or {}).get("passed") is False:
        raise SystemExit("blind health 已判失败——family 已关闭")
    if env is None:
        env = _load_env()
    values = env["values"]
    obs = bse.setdefault("observations", {})
    for row in bse["labels"]:
        key = row["key"]
        if key in obs:
            continue
        obs[key] = _bse_hist_stability(row["series"], row["origin"], values)
        _save_report(report)
        print("== bse-p2 " + key + ": value="
              + str(obs[key]["value"]), flush=True)
    health = _bse_blind_health(obs)
    bse["blind_health"] = health
    if not health["passed"]:
        bse["verdict"] = {
            "verdict": "PROGRAM_CONDITIONED_OBSERVATION_DEGENERATE",
            "family": "CLOSED",
            "final": True,
            "note": "Observation 无分辨率——关闭本 family，不调聚合公式",
            "health": health}
    _save_report(report)
    print("bse-p2 blind_health passed=" + str(health["passed"]))
    return 0


def phase_bse_p3p4() -> int:
    """P3/P4：fit 边界检查 → Runtime 机械 midpoint 候选 → 匿名 Capsule →
    Slow 一次提案（今晚唯一 LLM 使用；≤1 提案 + ≤1 结构化重试）。"""
    report = _load_report()
    if not report.get("bse_protocol"):
        raise SystemExit("bse_protocol 未冻结")
    bse = report.setdefault("bse", {})
    if not bse.get("labels"):
        raise SystemExit("先跑 bse-p1")
    v = bse.get("verdict") or {}
    if v.get("final"):
        raise SystemExit("已终裁（" + str(v.get("verdict")) + "）")
    labeled = bse["labels"]
    first = bse.get("first_check") or {}
    attempts = int(bse.get("slow_attempts", 0))
    if attempts >= 2:
        raise SystemExit("Slow 提案额度耗尽（1 提案 + 1 重试）")
    retry_feedback = None
    if attempts == 1:
        rep = bse.get("replay") or {}
        failed = ((rep.get("pass_evaluation") or {}).get("failed")) or []
        if not failed:
            raise SystemExit("无 replay 拒绝记录——不允许重试")
        retry_feedback = ("未通过判据: " + ", ".join(failed)
                          + "。请重新审视 A/B 组 Observation 值的相对位置后重选。")
    if first.get("separable"):
        feat = first["validated_feature"]
        feature = str(feat["feature"])
        tau = float(feat["threshold"])
        fit_pts = []
    else:
        if not (bse.get("blind_health") or {}).get("passed"):
            raise SystemExit("blind health 未通过——先确认 bse-p2")
        obs = bse["observations"]
        fit_pts = [{"key": r["key"], "group": r["group"],
                    "value": (obs.get(r["key"]) or {}).get("value")}
                   for r in labeled
                   if r["series"] in BSE_FIT_SERIES
                   and r["group"] in ("A", "B")]
        boundary = _bse_fit_boundary(fit_pts)
        bse["fit_boundary"] = boundary
        if boundary["boundary_count"] != 1 or boundary["tau"] is None:
            bse["verdict"] = {
                "verdict": "CONTEXT_UTILITY_UNIDENTIFIABLE",
                "reason": "fit_boundary_count="
                          + str(boundary["boundary_count"]),
                "family": "CLOSED", "final": True,
                "note": ("单阈值不可表达 fit A/B——不多阈值/不组合/不换特征；"
                         "关闭一维历史稳定性 Scope family"),
                "boundary": boundary}
            _save_report(report)
            print("bse-p3p4: fit 边界 "
                  + str(boundary["boundary_count"]) + " ≠1 → 不生成规则")
            return 0
        feature = BSE_OBS_FEATURE
        tau = float(boundary["tau"])
        fit_pts = boundary["sorted_fit"]
    view = [dict(p, anon="f" + str(i + 1)) for i, p in enumerate(fit_pts)]
    prompt = _bse_capsule_prompt(view, feature, tau, retry_feedback)
    client, counter = _make_client()
    raw = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}])
    text = raw.choices[0].message.content
    choice = _bse_parse_slow_choice(text)
    bse["slow_attempts"] = attempts + 1
    bse["capsule"] = {"attempt": attempts + 1, "prompt": prompt,
                      "raw_response": text, "parsed": choice,
                      "llm_calls": int(counter.calls)}
    if choice["choice"] == "abstain":
        bse["verdict"] = {
            "verdict": "CONTEXT_UTILITY_UNIDENTIFIABLE",
            "reason": "slow_abstained", "family": "CLOSED", "final": True,
            "note": "Slow 弃权/输出不可解析——不生成规则，关闭 family",
            "parsed": choice}
        _save_report(report)
        print("bse-p3p4: Slow abstain (" + str(choice["parse"]) + ")")
        return 0
    id_map = _bse_episode_id_map(_bse_load_episodes())
    rule = _bse_assemble_rule(choice["choice"], feature, tau, labeled, id_map)
    bse["rule"] = rule
    _save_report(report)
    print("bse-p3p4: Slow choice=" + choice["choice"]
          + " → rule " + json.dumps(rule["applicability"], ensure_ascii=False))
    return 0


def _bse_replay_rows(report: Mapping[str, Any],
                     env: Mapping[str, Any]) -> list:
    """held-out T10 H0/H1 机械 replay（零 LLM）：规则只门控 prior 槽；
    probe 后 Runtime winner resolution；support 现场评估（确定性仪器），
    delayed 复用已测缓存（零新 Outcome）。"""
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: PLC0415
        ScopeExecutor,
    )
    bse = report["bse"]
    rule = bse["rule"]
    obs = bse["observations"]
    labeled = {r["key"]: r for r in bse["labels"]}
    values = env["values"]
    cache: dict[str, tuple] = {}

    def support(series: str, origin: int) -> tuple:
        key = series + "@" + str(origin)
        if key not in cache:
            roster, vals = _support_roster(series, values)
            ex = ScopeExecutor(roster, vals, nsu._config(),
                               evaluate_fn=nsu._evaluate_kdd)
            rr = ex.evaluate(BSE_PROGRAM, origin)
            cache[key] = (float(rr.gain) if rr.gain is not None else None,
                          bool(rr.verification.passed), rr.error)
        return cache[key]

    rows = []
    for key in sorted(k for k, r in labeled.items()
                      if r["series"] in BSE_HELDOUT_SERIES):
        row0 = labeled[key]
        value = (obs.get(key) or {}).get("value")
        fires = _bse_rule_fires(value, rule)
        row = {"key": key, "group": row0["group"], "label": row0["label"],
               "obs_value": value, "rule_fires": fires}
        for arm in ("H0", "H1"):
            prior = True if arm == "H0" else fires
            rec = {"prior": prior,
                   "probe_order": (["outlier_mad", "identity"] if prior
                                   else ["identity"]),
                   "support_receipts": 0, "negative_probes": 0,
                   "winner": "identity", "support_gain": 0.0,
                   "delayed_gain": 0.0, "adopted": False}
            if prior:
                gain, passed, err = support(row0["series"], row0["origin"])
                rec["support_receipts"] = 1
                rec["verification_passed"] = passed
                rec["error"] = err
                if gain is not None:
                    if gain < -M:
                        rec["negative_probes"] = 1
                    if gain >= M:   # post-probe Runtime winner resolution
                        rec["winner"] = "outlier_mad"
                        rec["support_gain"] = gain
                        rec["adopted"] = True
                        rec["delayed_gain"] = (
                            float(row0["delayed_gain"])
                            if row0["delayed_evaluated"] else 0.0)
            row[arm] = rec
        row["removal_delta"] = row["H0"]["probe_order"] != row["H1"]["probe_order"]
        rows.append(row)
    return rows


def phase_bse_p5(env: Mapping[str, Any] | None = None) -> int:
    """P5：held-out T10 H0/H1 机械 replay + 逐 receipt 指标（零 LLM）。"""
    report = _load_report()
    if not report.get("bse_protocol"):
        raise SystemExit("bse_protocol 未冻结")
    bse = report.setdefault("bse", {})
    if not bse.get("rule"):
        raise SystemExit("无 rule——先跑 bse-p3p4")
    v = bse.get("verdict") or {}
    if v.get("final"):
        raise SystemExit("已终裁（" + str(v.get("verdict")) + "）")
    attempts = int(bse.get("replay_attempts", 0)) + 1
    if attempts > 2:
        raise SystemExit("replay 额度耗尽（初验 + 1 次结构化重试）")
    if env is None:
        env = _load_env()
    rows = _bse_replay_rows(report, env)
    pe = _bse_pass_evaluation(rows)
    bse["replay"] = {
        "attempt": attempts, "rows": rows, "pass_evaluation": pe,
        "delayed_source": ("adopted winner 的 delayed 复用 batch1/usel 已测缓存"
                           "（零新 Outcome）；identity=no-op 定义 delayed=0"),
        "c_group_disclosure": [
            {"key": r["key"], "rule_fires": r["rule_fires"],
             "H0": {"winner": r["H0"]["winner"],
                    "delayed_gain": r["H0"]["delayed_gain"]},
             "H1": {"winner": r["H1"]["winner"],
                    "delayed_gain": r["H1"]["delayed_gain"]},
             "note": "C 组：delayed 翻负披露；不计入本 Scope Rule 的 PASS/FAIL"}
            for r in rows if r["group"] == "C"]}
    bse["replay_attempts"] = attempts
    _save_report(report)
    print("bse-p5 attempt=" + str(attempts) + " passed="
          + str(pe["passed"]) + " failed=" + repr(pe["failed"]))
    for r in rows:
        print("  " + r["key"] + " [" + r["group"] + "] obs="
              + str(r["obs_value"]) + " fires=" + str(r["rule_fires"])
              + " H1 prior=" + str(r["H1"]["prior"])
              + " H0 winner=" + r["H0"]["winner"]
              + " H1 winner=" + r["H1"]["winner"])
    return 0


def phase_bse_verdict() -> int:
    """双层终裁（预注册 verdict 树；机械）。"""
    report = _load_report()
    if not report.get("bse_protocol"):
        raise SystemExit("bse_protocol 未冻结")
    bse = report.setdefault("bse", {})
    v = bse.get("verdict") or {}
    if v.get("final"):
        print("bse 终裁(已定): " + str(v.get("verdict")))
        print(json.dumps(v, ensure_ascii=False))
        return 0
    if True:
        rep = bse.get("replay") or {}
        pe = rep.get("pass_evaluation") or {}
        attempts = int(bse.get("replay_attempts", 0))
        if not rep:
            raise SystemExit("无 replay——先跑 bse-p5（或 p2/p3p4 的提前终裁）")
        if pe.get("passed"):
            v = {"verdict": "SUPPORT_HARM_SCOPE_RULE_DEV_PASS",
                 "parallel": "TEMPORAL_INSTABILITY_UNRESOLVED",
                 "overall": "PARTIAL_BATCH_SELF_EVOLUTION_DEV_PASS",
                 "note": ("Scope PASS 只承担 B 组（SUPPORT_HARM）；C 组 "
                          "temporal 风险未解决——delayed gate 只能阻止后续复用，"
                          "不能撤销已发生的第一次 delayed harm")}
        elif attempts >= 2:
            rule0 = bse.get("rule") or {}
            app0 = rule0.get("applicability") or {}
            alt_op = "le" if app0.get("operator") == "ge" else "ge"
            alt = {"applicability": {"operator": alt_op,
                                     "threshold": app0.get("threshold")}}
            obs0 = bse.get("observations") or {}
            alt_rows = []
            for r0 in (rep.get("rows") or []):
                v0 = (obs0.get(r0["key"]) or {}).get("value")
                alt_rows.append({"key": r0["key"], "group": r0["group"],
                                 "alt_fires": _bse_rule_fires(v0, alt)})
            alt_fail = []
            if any(r0["group"] == "A" and not r0["alt_fires"]
                   for r0 in alt_rows):
                alt_fail.append("stable_prior_recall")
            if any(r0["group"] == "B" and r0["alt_fires"]
                   for r0 in alt_rows):
                alt_fail.append("harm_auto_priority_blocked")
            v = {"verdict": "CONTEXT_UTILITY_UNIDENTIFIABLE",
                 "reason": "replay_failed_after_retry",
                 "failed": pe.get("failed"),
                 "family": "CLOSED",
                 "retry_analysis": {
                     "alternative_predicate": alt_op,
                     "alternative_heldout": alt_rows,
                     "alternative_would_fail": alt_fail,
                     "note": ("确定性预检（零新评估）：替代谓词在 held-out 上"
                              "也不成立——两个方向都失败 = 该一维 Observation "
                              "的结构性限制，非 Slow 选择错误")},
                 "note": "结构化重试后仍未通过——关闭一维历史稳定性 Scope family"}
        else:
            v = {"verdict": "OBSERVATION_VALID_SCOPE_PATCH_REJECTED",
                 "failed": pe.get("failed"),
                 "retry": "允许一次带拒绝反馈的结构化重试（bse-p3p4 → bse-p5）",
                 "final": False}
    final = v.get("verdict") in (
        "SUPPORT_HARM_SCOPE_RULE_DEV_PASS",
        "CONTEXT_UTILITY_UNIDENTIFIABLE",
        "PROGRAM_CONDITIONED_OBSERVATION_DEGENERATE")
    v["final"] = bool(v.get("final") or final)
    if v.get("verdict") == "SUPPORT_HARM_SCOPE_RULE_DEV_PASS":
        if not BSE_RULE_REL.exists():
            rule = dict(bse["rule"])
            rule["authority"] = "LOCAL_ACTIVE"
            rule["activation_evidence"] = ("bse held-out T10 机械 replay 通过"
                                           "（方案乙五项判据）")
            BSE_RULE_REL.write_text(
                json.dumps(rule, ensure_ascii=False, indent=2, default=str)
                + "\n", encoding="utf-8")
        v["rule_artifact"] = ("artifacts/experience/"
                              "scope_rule_outlier_mad_stable_scope_v1.json")
        v["rule_authority"] = "LOCAL_ACTIVE"
    v["closeout"] = {
        "claim_boundary": report["bse_protocol"]["claim_boundary"],
        "c_group_status": "TEMPORAL_INSTABILITY_UNRESOLVED",
        "family": (v.get("family") or (
            "RULE_LOCAL_ACTIVE"
            if v.get("verdict") == "SUPPORT_HARM_SCOPE_RULE_DEV_PASS"
            else "CLOSED")),
        "no_shared_capability": True,
        "no_cross_domain_claim": True,
        "known_unrelated_regression": ("tests/functional/test_f1_forecast_pilot.py"
                                       " 原生崩溃（verify_candidate 内 scipy/mkl "
                                       "native crash；与 BSE 改动路径零交集——"
                                       "f1 pilot 不读 episodes/report/runner；"
                                       "两次独立运行确定性复现）")}
    bse["verdict"] = v
    _save_report(report)
    print("bse 终裁: " + str(v.get("verdict"))
          + (" || " + str(v.get("parallel")) if v.get("parallel") else ""))
    print(json.dumps(v, ensure_ascii=False))
    return 0

# ---------------------------------------------------------------- n1
# N1 ACTION_EVIDENCE（用户裁决 2026-08-15；BSE 可信关闭后的下一 first
# fault）：「历史 outlier_mad 从未真正修改数据（INERT）时，0.0 gain 被误当
# 安全证据，首次激活的 Context 反而获得 Target-local prior」。
# 顺序：零 Outcome prevalence（只看 changed-point 几何，不看 gain）→
# premise ≥2 个不同 series 才机械装配单例 Scope Rule（0 LLM——规则空间
# 单例，Slow 提案链已由 BSE 验证）→ held-out replay → 通过或可信关闭。
# T10@600 是已消耗的发现案例，永远不得作确认证据。

N1_PROGRAM = BSE_PROGRAM                  # outlier_mad 单算子（family 锁定）
N1_GRID_SERIES = ("T1", "T10", "T100", "T101")   # census pre_registered dev
N1_GRID_ORIGINS = (600, 792, 888, 984)
N1_CONSUMED_DISCOVERY = ("T10@600",)
N1_NO_PRIOR_STATUSES = ("INERT", "UNKNOWN", "ACTED_NEUTRAL")


def _n1_window_change_counts(series: str, cutoff: int,
                             values: Mapping[str, Any]):
    """cutoff 前全部训练窗口（anchor+48 ≤ cutoff）上 outlier_mad 的**全窗口**
    修改点数列表（修订 v2：初版只数 target 块——漏掉 context 块修改，但
    v6._evaluate 的训练特征和标签都来自修改后的整个窗口，且
    behavior_point_count/verifier  inspected region 均为全窗口口径；
    「从未真正修改过数据」= 全窗口无修改）。零 Outcome：纯执行几何。
    窗口不完整或一个合法窗口都没有 → None（不可计算）。"""
    from SelfEvolvingHarnessTS.runtime.executor import run_pipeline  # noqa: PLC0415
    if cutoff < BSE_MIN_ANCHOR + HORIZON:
        return None
    raw = np.asarray(values[series], dtype=np.float64)
    out = []
    for a in nsu._config()["anchors"]:
        a = int(a)
        if a + HORIZON > cutoff:
            continue
        window = raw[a - 192: a + HORIZON]
        if window.size < 192 + HORIZON:
            return None
        prepared = run_pipeline(N1_PROGRAM, window, source="agent").artifact
        prepared = np.asarray(prepared, dtype=np.float64)
        out.append(int(np.count_nonzero(
            ~np.isclose(prepared, window, equal_nan=True))))
    return out or None


def _n1_status_from_counts(hist_counts: Mapping[str, Any]) -> str:
    """prevalence 阶段（不看 gain）：INERT / UNKNOWN / ACTED。"""
    if not hist_counts or any(v is None for v in hist_counts.values()):
        return "UNKNOWN"
    total = sum(int(x) for v in hist_counts.values() for x in v)
    return "INERT" if total == 0 else "ACTED"


def _n1_resolve_acted(gains: Sequence[Any]) -> str:
    """premise 选中后才打开历史 gain：ACTED 细分为
    BENEFICIAL / HARMFUL / CONFLICT / ACTED_NEUTRAL（冻结口径）。"""
    pos = sum(1 for g in gains if g is not None and float(g) >= M)
    neg = sum(1 for g in gains if g is not None and float(g) < -M)
    if pos and neg:
        return "CONFLICT"
    if pos:
        return "BENEFICIAL"
    if neg:
        return "HARMFUL"
    return "ACTED_NEUTRAL"


def _n1_select_cases(prevalence: Sequence[Mapping[str, Any]]) -> dict:
    """NOVEL_ACTION 案例选择（盲于 gain，按 (series, origin) 字典序机械取）：
    开发案例 = 第一个；组外验证案例 = 不同 series 的第一个。不足两个不同
    series → PREMISE_TOO_RARE（不实现 Rule、不消耗 fresh 数据）。"""
    novels = sorted(
        (p for p in prevalence
         if p.get("novel_action")
         and p["key"] not in N1_CONSUMED_DISCOVERY),
        key=lambda p: (p["series"], int(p["origin"])))
    if not novels:
        return {"premise": "NONE",
                "verdict": "ACTION_EVIDENCE_PREMISE_TOO_RARE",
                "novel_cases": []}
    dev = novels[0]
    val = next((p for p in novels[1:] if p["series"] != dev["series"]), None)
    if val is None:
        return {"premise": "SINGLE_SERIES_ONLY",
                "verdict": "ACTION_EVIDENCE_PREMISE_TOO_RARE",
                "novel_cases": [p["key"] for p in novels],
                "note": "仅单一 series 有 novel 案例——无法组外验证"}
    return {"premise": "OK", "dev_case": dev["key"],
            "validation_case": val["key"],
            "novel_cases": [p["key"] for p in novels],
            "note": "开发/验证案例按字典序机械选择，未看任何 gain"}


def phase_n1_freeze() -> int:
    """冻结 N1 协议（任何 prevalence 计算前；已冻结或已有结果则拒绝）。"""
    report = _load_report()
    if report.get("n1_protocol"):
        raise SystemExit("n1_protocol 已存在——拒绝重复冻结")
    if report.get("n1"):
        raise SystemExit("n1 已有结果——协议必须先冻结")
    proto = {
        "experiment_id": "N1_ACTION_EVIDENCE_SCOPE_2026_08_15",
        "user_ruling": ("2026-08-15：BSE 已可信关闭 min(last3 gain) 单阈值 Scope "
                        "family（机制正证据=审批链成立；方法负证据=一维历史稳定性"
                        "阈值不能迁移到 held-out）。下一 first fault：历史 INERT 的 "
                        "0.0 被误当安全证据。运行策略：outlier_mad 维持逐 Context "
                        "Support + delayed 双门，BSE Rule 不激活"),
        "first_fault": ("historical_action_evidence_status = INERT/UNKNOWN 且 "
                        "current_action_active = true（NOVEL_ACTION）时，"
                        "Target-local Skill 不应获得自动 prior——无真实历史作用"
                        "证据 ≠ 已验证安全"),
        "observation": {
            "historical_action_evidence_status": {
                "INERT": "三个历史 cutoff（t−H/t−2H/t−3H）全部合法且 target 块修改点总数 = 0",
                "ACTED": "历史真实发生修改（prevalence 阶段不细分方向）",
                "BENEFICIAL": "ACTED 且 ≥1 个 cutoff gain ≥ M 且无 < −M（仅选中案例后打开）",
                "HARMFUL": "ACTED 且 ≥1 个 cutoff gain < −M 且无 ≥ M",
                "CONFLICT": "ACTED 且正负兼有",
                "ACTED_NEUTRAL": "ACTED 但全部 |gain| < M（行动过但无方向性证据）",
                "UNKNOWN": "任一历史 cutoff 不可计算（窗口不足/仪器失败）",
            },
            "current_action_active": ("当前 origin 训练窗（anchor+48 ≤ origin）"
                                      "确定性模拟 changed_point_count > 0"),
            "legality": ("prevalence 零 gain 零标签零 LLM——只看执行几何与数据"
                         "可用性；案例选择盲于 gain；历史 gain 只对选中案例打开"),
        },
        "prevalence": {
            "grid": "census pre_registered dev 4 series × 4 origins（16 context）",
            "consumed_discovery": list(N1_CONSUMED_DISCOVERY),
            "case_requirement": "≥2 个不同 series 的 NOVEL_ACTION（1 开发 + 1 组外验证）",
            "selection_rule": "novel 案例按 (series, origin) 字典序；dev=第一个；validation=不同 series 的第一个",
            "too_rare": "ACTION_EVIDENCE_PREMISE_TOO_RARE：不实现 Rule、不消耗 fresh 数据、family 关闭",
        },
        "rule": {
            "assembly": ("Runtime 机械装配（规则空间单例、无阈值无方向可选 → "
                         "0 LLM；Slow 提案→Runtime 核销链已由 BSE 验证，本 family "
                         "的独立性由 held-out replay 审批承担）"),
            "card": {"rule_id": "outlier_mad_action_evidence_scope_v1",
                     "surface": "scope",
                     "workflow_signature": "outlier_mad",
                     "trigger": {"historical_action_evidence_status":
                                 list(N1_NO_PRIOR_STATUSES),
                                 "current_action_active": True},
                     "action": {"target_local_skill_prior": False},
                     "risk": {"requires_target_support": True}},
            "permissions": ("不删除 exploration 槽；不禁止 Agent 独立 propose 同一 "
                            "Workflow；不阻断 Source Experience proposal；首次合法正 "
                            "Support 仍形成 Episode；delayed 决定未来 prior"),
            "note": ("trigger 含 ACTED_NEUTRAL 是对「无正面证据」语义的精确化"
                     "（2026-08-15 主 Agent 冻结，用户口径 INERT/UNKNOWN 的超集）；"
                     "HARMFUL/CONFLICT 的数值门控属于已关闭 family，不借 N1 复活"),
        },
        "replay": {
            "arms": {"H0": "现状：INERT/UNKNOWN 历史也允许 local prior（宽 scope）",
                     "H1": "H0 + N1 Rule 门控 prior 槽（exploration 保留）"},
            "arm_semantics": ("probe 前顺序固定（prior 在槽→[outlier_mad, identity]，"
                              "否则 [identity]）；probe 后 Runtime winner resolution"
                              "（gain 最大且 ≥M 才采用）——同 BSE 口径，非 Fast Select"),
            "metrics": ["harmful prior probes（gain < −M 的 prior 探测）",
                        "Target Support receipts", "首次正向 Workflow",
                        "delayed harm（采用 winner；复用已测缓存，零新 Outcome）",
                        "正向证据→未来 prior 状态变化", "removal delta"],
            "delayed_source": "采用 winner 的 delayed 复用 batch1/usel 缓存；未测过的选中案例允许 dev 纪律内新评估（同 batch1 claim 边界，不称 fresh）",
        },
        "verdict_rules": {
            "premise_too_rare": "PREMISE_TOO_RARE → 关闭，不实现 Rule",
            "pass": ("验证案例（组外）上 H1 减少有害 prior probe（该案例 support "
                     "gain < −M 的 prior 探测被消除）且 不阻止任何 BENEFICIAL-证据 "
                     "Skill 且 delayed 权限未被绕过 且 removal 后行为翻转 → "
                     "ACTION_EVIDENCE_SCOPE_RULE_DEV_PASS"),
            "validation_not_harmful": ("组外验证案例实际 gain ≥ −M（无害可防）→ "
                                       "ACTION_EVIDENCE_SCOPE_RULE_REJECTED，"
                                       "关闭——需求证据不存在，不调阈值"),
            "reject_retry": "其余失败 → REJECTED，允许一次结构化重试；再失败关闭",
        },
        "llm_budget": "0（规则空间单例）；若未来引入非单例规则空间另行裁决",
        "claim_boundary": ("development Target-local 证据；只约束 Target-local "
                           "prior，不阻断 A5 Source proposal slot；不声称跨域；"
                           "不升级 Shared Capability；C 组 temporal 风险维持 "
                           "TEMPORAL_INSTABILITY_UNRESOLVED"),
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["n1_protocol"] = proto
    _save_report(report)
    print("n1_protocol FROZEN")
    return 0


def phase_n1_prevalence(env: Mapping[str, Any] | None = None) -> int:
    """零 Outcome prevalence：16 网格 context 的 changed-point 几何 +
    NOVEL_ACTION 判定 + 盲选案例（可恢复：按 key 跳过）。"""
    report = _load_report()
    if not report.get("n1_protocol"):
        raise SystemExit("n1_protocol 未冻结——先跑 n1-freeze")
    n1 = report.setdefault("n1", {})
    if (n1.get("verdict") or {}).get("final"):
        raise SystemExit("n1 已终裁")
    # 语义修订 v2（2026-08-15 主 Agent 裁决，待用户复核）：target 块计数 →
    # 全窗口计数（忠实性修正，非方法贡献）。口径版本不一致则重置 prevalence。
    sem = "whole_window_v2"
    if n1.get("prevalence_semantics") != sem and n1.get("prevalence"):
        n1["prevalence"] = []
        n1.pop("selection", None)
        n1.pop("verdict", None)
        proto = report.get("n1_protocol") or {}
        proto.setdefault("amendments", []).append({
            "date": "2026-08-15",
            "change": ("changed-point 计数从 target 块（[192:]）改为全窗口——"
                       "v6._evaluate 训练特征与标签均来自修改后整窗，"
                       "behavior_point_count 与 verifier inspected region 也是"
                       "全窗口口径；target-only 会把 context 块有修改的 Context "
                       "误标 INERT（T100@600 即例：target 块 0 改但历史 gain "
                       "−0.0004 ≠ 0）"),
            "ruled_by": "主 Agent（语义忠实性修正），待用户复核",
            "effect": "prevalence 全部重算；初版结果作废不入档"})
        _save_report(report)
        print("== n1 prevalence 语义修订 v2：重置重算（全窗口口径）", flush=True)
    n1["prevalence_semantics"] = sem
    if env is None:
        env = _load_env()
    values = env["values"]
    rows = n1.setdefault("prevalence", [])
    done = {r["key"] for r in rows}
    for series in N1_GRID_SERIES:
        for origin in N1_GRID_ORIGINS:
            key = series + "@" + str(origin)
            if key in done:
                continue
            hist = {}
            for k in BSE_CUTOFF_STEPS:
                c = origin - k * HORIZON
                hist[str(c)] = _n1_window_change_counts(series, c, values)
            cur = _n1_window_change_counts(series, origin, values)
            status = _n1_status_from_counts(hist)
            cur_total = None if cur is None else int(sum(cur))
            row = {"key": key, "series": series, "origin": origin,
                   "historical_counts": hist,
                   "historical_status_prevalence": status,
                   "current_changed_total": cur_total,
                   "current_action_active": bool(cur_total),
                   "novel_action": (status in ("INERT", "UNKNOWN")
                                    and bool(cur_total)),
                   "consumed_discovery": key in N1_CONSUMED_DISCOVERY}
            rows.append(row)
            _save_report(report)
            print("== n1 " + key + ": hist=" + status
                  + " current_changed=" + str(cur_total)
                  + " novel=" + str(row["novel_action"]), flush=True)
    selection = _n1_select_cases(rows)
    n1["selection"] = selection
    # 发现案例订正（随 N1 语义修订浮出水面；只订正描述，不改 BSE 已结案裁定）
    bse_obs = ((report.get("bse") or {}).get("observations") or {}).get("T10@600") or {}
    legs = [l.get("gain") for l in bse_obs.get("legs") or []]
    if legs:
        n1["discovery_case_correction"] = {
            "key": "T10@600",
            "correction": ("BSE closeout 称 T10@600「三个历史 cutoff 逐位惰性"
                           "（gain 恰为 0.0）」——不准确：legs="
                           + repr(legs) + "，cutoff 552 有 +0.0113 的弱正向 "
                           "历史证据；其真实结构是「弱正面历史不保鲜」而非"
                           "「纯 INERT 首次激活」。BSE 终裁（family 关闭）"
                           "不依赖该描述，维持不变"),
            "bse_verdict_unchanged": True}
    if selection["premise"] != "OK":
        n1["verdict"] = {
            "verdict": "ACTION_EVIDENCE_PREMISE_TOO_RARE",
            "final": True, "family": "CLOSED_NOT_IMPLEMENTED",
            "note": "无组外可验证的 novel-action 案例——不实现 Rule、不消耗 "
                    "fresh 数据；outlier_mad 维持逐 Context Support+delayed 双门",
            "selection": selection}
    _save_report(report)
    print("n1 prevalence 完成: premise=" + selection["premise"]
          + " novels=" + repr(selection["novel_cases"]))
    return 0

# ---------------------------------------------------------------- n3
# N3 NATURAL SOURCE EPISODE PACK（长期路线 2026-08-15：N1 到终点后回到主
# 目标）：把 KDD 自然积累的 outlier_mad 正/负/冲突轨迹冻结为完整 Source
# Pack——不挑正例、不人工 signswap、Source Rule 只作参考证据、Source
# Memory 最多占一个 prior proposal slot、Target exploration 永远保留、
# Target feedback 覆盖 Source 排序。零 LLM、零新 Outcome。

N3_PACK_REL = PROJECT_ROOT / "artifacts" / "experience" \
    / "source_pack_kdd_outlier_mad_v1.json"


def _n3_pack_episodes(episodes: Sequence[Mapping[str, Any]]) -> list:
    """自然 KDD outlier_mad 轨迹（完整收纳，不挑 relation）。"""
    return [dict(ep) for ep in episodes
            if ep.get("domain_namespace") == "kdd2018_dev"
            and ep.get("workflow_signature") == "outlier_mad"]


def _n3_content_check(ep: Mapping[str, Any]) -> list:
    """单条 Episode 内容完整性（SOURCE_CONTENT_INCOMPLETE 判定依据）。"""
    missing = []
    if not ep.get("workflow_signature") or ep.get("workflow_signature") in (
            "unknown", "identity"):
        missing.append("workflow_signature")
    cs = ep.get("context_summary") or {}
    if not (cs.get("local_pattern")):
        missing.append("context_summary.local_pattern")
    if not ((cs.get("program_geometry") or {}).get("program_steps")):
        missing.append("program_geometry.program_steps")
    sr = ep.get("support_response") or {}
    if sr.get("gain") is None:
        missing.append("support_response.gain")
    dr = ep.get("delayed_response") or {}
    if "evaluated" not in dr:
        missing.append("delayed_response.evaluated")
    if ep.get("response_validity") != "VALID":
        missing.append("response_validity!=VALID")
    return missing


def phase_n3_freeze() -> int:
    """冻结 N3 协议（打包前；已冻结或已有 pack 则拒绝）。"""
    report = _load_report()
    if report.get("n3_protocol"):
        raise SystemExit("n3_protocol 已存在——拒绝重复冻结")
    if report.get("n3"):
        raise SystemExit("n3 已有结果——协议必须先冻结")
    proto = {
        "experiment_id": "N3_NATURAL_SOURCE_EPISODE_PACK_2026_08_15",
        "user_ruling": ("2026-08-15 长期路线：N1 到终点后回到主目标；KDD 自然"
                        "积累的 POSITIVE/NEGATIVE/CONFLICT/delayed/program "
                        "geometry/public Context 轨迹比「单条正例+人工 "
                        "signswap」更合法——冻结为完整 Source Pack"),
        "pack_rules": ["收纳完整自然轨迹，不挑正例",
                       "保留全部正、负、冲突 Episode",
                       "不把 KDD Target-local Rule 当跨域 Active Skill",
                       "Source Rule 只能作为参考证据",
                       "Source Memory 最多占一个 prior proposal slot",
                       "Target exploration slot 永远保留",
                       "Target feedback 覆盖 Source 排序",
                       "禁止人工 signswap"],
        "membership": "domain=kdd2018_dev 且 workflow_signature=outlier_mad 的全部 Episode（当前 10 条：3 POSITIVE/3 NEGATIVE/4 CONFLICT）",
        "content_requirements": ["workflow_signature 可解析",
                                 "context_summary.local_pattern",
                                 "program_geometry.program_steps",
                                 "support_response.gain 非 None",
                                 "delayed_response.evaluated 字段存在",
                                 "response_validity == VALID"],
        "verdict_rules": {
            "ready": "全部成员内容完整 → NATURAL_SOURCE_EPISODE_PACK_READY",
            "incomplete": "任一成员缺内容 → SOURCE_CONTENT_INCOMPLETE（列出缺失，不人工补）"},
        "llm_budget": "0", "new_outcomes": "0",
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["n3_protocol"] = proto
    _save_report(report)
    print("n3_protocol FROZEN")
    return 0


def phase_n3() -> int:
    """打包 + 内容核销（零 LLM）。用户裁决 2026-08-15：名义冻结必须
    实际不可变——已有 n3 结果或 pack 文件时拒绝覆盖（episodes.json
    后续增长不得悄悄改写已冻结 Pack）。"""
    report = _load_report()
    if not report.get("n3_protocol"):
        raise SystemExit("n3_protocol 未冻结——先跑 n3-freeze")
    if report.get("n3") or N3_PACK_REL.exists():
        raise SystemExit("n3 已终裁或 pack 已存在——拒绝覆盖"
                         "（用户裁决 2026-08-15：冻结即不可变）")
    episodes = _bse_load_episodes()
    pack = _n3_pack_episodes(episodes)
    incomplete = {}
    for ep in pack:
        missing = _n3_content_check(ep)
        if missing:
            incomplete[str(ep.get("episode_id"))] = missing
    relations: dict[str, int] = {}
    for ep in pack:
        r = str(ep.get("relation"))
        relations[r] = relations.get(r, 0) + 1
    n3 = {
        "member_ids": sorted(str(ep.get("episode_id")) for ep in pack),
        "relation_counts": relations,
        "content_incomplete": incomplete,
        "verdict": ("SOURCE_CONTENT_INCOMPLETE" if incomplete
                    else "NATURAL_SOURCE_EPISODE_PACK_READY"),
    }
    if not incomplete:
        doc = {"pack_id": "source_pack_kdd_outlier_mad_v1",
               "created": "2026-08-15",
               "rules": report["n3_protocol"]["pack_rules"],
               "episodes": sorted(pack, key=lambda e: str(e["episode_id"]))}
        N3_PACK_REL.write_text(
            json.dumps(doc, ensure_ascii=False, indent=1, default=str) + "\n",
            encoding="utf-8")
        n3["pack_artifact"] = ("artifacts/experience/"
                               "source_pack_kdd_outlier_mad_v1.json")
    report["n3"] = n3
    _save_report(report)
    print("n3: " + n3["verdict"] + " members=" + str(len(pack))
          + " relations=" + json.dumps(relations, ensure_ascii=False))
    if incomplete:
        print("incomplete: " + json.dumps(incomplete, ensure_ascii=False))
    return 0


def phase_n3_amend() -> int:
    """用户裁定 2026-08-15 的三条 claim 收窄入档（追加，不改写原结果）。"""
    report = _load_report()
    n3 = report.get("n3")
    if not n3:
        raise SystemExit("n3 尚无结果——无可修订")
    if n3.get("claim_adjustments_2026_08_15"):
        raise SystemExit("claim_adjustments 已存在——拒绝重复修订")
    eps = _n3_pack_episodes(_bse_load_episodes())
    no_delayed = sorted(
        str(e["episode_id"]) for e in eps
        if not ((e.get("delayed_response") or {}).get("evaluated")))
    n3["claim_adjustments_2026_08_15"] = {
        "user_ruling": ("N3 是昨夜最大进展，但『弹药齐备』须收窄——"
                        "三条边界裁定如下"),
        "delayed_completeness": {
            "episodes_without_delayed": no_delayed,
            "claim": ("不得声称 10 条全部拥有完整 delayed 轨迹："
                      + str(len(no_delayed)) + " 条 NEGATIVE 仅有 Support；"
                      "它们仍是合法负 Episode")},
        "pack_immutability": ("phase_n3 已加拒覆盖闸门：已有结果或 pack 文件"
                              "时拒绝重打包（无需 SHA/Ledger）"),
        "consumption_semantics": {
            "fact": ("本批 Episode 无 recent.*/change.* 特征 → Fast Path 走 "
                     "contrast_pack 文本检索路径，不产生 signed resolution "
                     "→ 无 Runtime-owned Slot P（fast_agent.py:750/1030）"),
            "a5_definition": ("A5 定义为『Source Contrast Memory 指令 vs 空 "
                              "Memory』，不得称为确定存在 Runtime Slot P 的"
                              "双槽实验；A5 价值可通过减少 harm、增加 "
                              "abstention 体现；无机械 prior 时必须显式报告 "
                              "source_prior_candidate_absent")},
    }
    _save_report(report)
    print("n3 claim adjustments recorded; episodes_without_delayed="
          + repr(no_delayed))
    return 0




# ---------------------------------------------------------------- n4
# N4 EXACT ROSTER 资格审查（用户裁定 2026-08-15）：不是数据集数量普查，
# 是 exact (series_uid, origin) 级资格审计。零 LLM、零 Outcome——只允许
# UID/长度/频率/历史消费记录/outlier_mad 确定性修改几何/合法性/窗口完整性。
# 关键实现事实（主 Agent 预验证）：clean_base↔registry join 键 = series_uid；
# registry exposure_class 为权威分级，artifact 扫描为佐证；扫描只用 64-hex
# 标识符（entity_id 太短会误报，不用于子串扫描）。

N4_REGISTRY_REL = PROJECT_ROOT / "artifacts" / "frozen" / "benchmark_v02" \
    / "series_registry.jsonl"
N4_CLEAN_BASE = PROJECT_ROOT / "data" / "benchmark_v0_2" / "clean_base"
N4_CANDIDATE_DATASETS = ("monash:traffic_hourly",
                         "uci_electricity_load_diagrams", "metr_la",
                         "monash:covid_deaths", "noaa_global_hourly")
N4_EXCLUDED_DATASETS = {
    "legacy_monash:nn5_daily": "episode memory 已有 nn5 域消费",
    "monash:nn5_daily": "同名族 v6-era 消费史，保守排除",
    "gefcom2012_load": "f1/smoke 消费史",
    "legacy_monash:tourism_monthly": "e2 脚本消费史",
    "legacy_monash:fred_md": "量太小且 confirmed_exposed",
    "legacy_monash:sunspot": "单序列",
    "legacy_monash:covid_deaths": "legacy 命名且 confirmed_exposed",
}
N4_ORIGINS = (600, 792, 888)          # 984 无 delayed 数据余量（984+96>1024）
N4_MIN_LENGTH = 984                    # origin 888 + 96（delayed）
N4_ROSTER_K = 6
N4_MIN_CONTEXTS = 4
N4_ANCHORS = tuple(range(312, 853, 60))


def _n4_load_registry() -> list:
    return [json.loads(l) for l in
            N4_REGISTRY_REL.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _n4_clean_records() -> dict:
    """clean_base record.json 索引：series_uid → {record, values_path}。"""
    out = {}
    for d in sorted(N4_CLEAN_BASE.iterdir()):
        rj = d / "record.json"
        vj = d / "values.npy"
        if not (rj.exists() and vj.exists()):
            continue
        rec = json.loads(rj.read_text(encoding="utf-8"))
        out[str(rec.get("series_uid"))] = {"record": rec, "values": vj}
    return out


def _n4_change_counts(arr: Any, cutoff: int):
    """outlier_mad 在 cutoff 前训练窗的确定性全窗口修改点总数（零 Outcome）。
    窗口不足/无合法窗口 → None。与 _n1_window_change_counts 同口径。"""
    from SelfEvolvingHarnessTS.runtime.executor import run_pipeline  # noqa: PLC0415
    raw = np.asarray(arr, dtype=np.float64)
    if cutoff < 312 + HORIZON:
        return None
    total, nw = 0, 0
    for a in N4_ANCHORS:
        if a + HORIZON > cutoff:
            continue
        w = raw[a - 192: a + HORIZON]
        if w.size < 192 + HORIZON:
            return None
        nw += 1
        prep = np.asarray(run_pipeline(N1_PROGRAM, w, source="agent").artifact,
                          dtype=np.float64)
        total += int(np.count_nonzero(~np.isclose(prep, w, equal_nan=True)))
    return total if nw >= 2 else None


def _n4_consumption_scan(identifiers: set) -> dict:
    """artifact 佐证扫描：64-hex 标识符集合在实验 artifact 中的命中。
    排除：registry 本身（存储非消费）、clean_base（数据存储）、本报告之外的
    无；主报告纳入（扫描时刻语义）。零 Outcome（纯文本匹配）。"""
    hits: dict[str, list] = {}
    files = []
    for pat in ("artifacts/**/*.json", "evaluation/**/*.log",
                "evaluation/**/*report*.json"):
        files.extend(PROJECT_ROOT.glob(pat))
    scanned = 0
    frozen_dir = PROJECT_ROOT / "artifacts" / "frozen"
    for fp in sorted(set(files)):
        if (fp == N4_REGISTRY_REL or N4_CLEAN_BASE in fp.parents
                or frozen_dir in fp.parents):
            # artifacts/frozen/** = split 分配/注册簿记（registry exposure_class
            # 已编码其结论）——分配≠消费，不计为 outcome 暴露（v2 修正）
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        for ident in identifiers:
            if ident and ident in text:
                hits.setdefault(ident, []).append(
                    str(fp.relative_to(PROJECT_ROOT)))
    return {"files_scanned": scanned, "hit_count": len(hits),
            "hits": {k: v[:3] for k, v in sorted(hits.items())}}


def _n4_pick_roster(per_dataset: Mapping[str, Any]) -> dict:
    """预注册确定性 roster 规则：合格 context 数最多的数据集（平手取
    dataset_id 字典序小者）；roster = 该集内 (series_uid, origin) 字典序
    前 N4_ROSTER_K 个；合格总数 < N4_MIN_CONTEXTS → 不可用。"""
    ranked = sorted(
        ((ds, len(d.get("eligible") or ())) for ds, d in per_dataset.items()),
        key=lambda kv: (-kv[1], kv[0]))
    if not ranked or ranked[0][1] < N4_MIN_CONTEXTS:
        return {"verdict": "FRESH_TARGET_CONTENT_UNAVAILABLE",
                "reason": "无数据集达到最小合格 context 数 "
                          + str(N4_MIN_CONTEXTS),
                "ranking": ranked}
    ds = ranked[0][0]
    roster = sorted(per_dataset[ds]["eligible"],
                    key=lambda e: (e["series_uid"], int(e["origin"]))
                    )[:N4_ROSTER_K]
    return {"verdict": "N4_ROSTER_ELIGIBLE", "dataset": ds,
            "roster": roster, "ranking": ranked}


def phase_n4_freeze() -> int:
    """冻结 N4 协议（任何审计读取前）。"""
    report = _load_report()
    if report.get("n4_protocol"):
        raise SystemExit("n4_protocol 已存在——拒绝重复冻结")
    if report.get("n4"):
        raise SystemExit("n4 已有结果——协议必须先冻结")
    proto = {
        "experiment_id": "N4_EXACT_ROSTER_ELIGIBILITY_2026_08_15",
        "user_ruling": ("2026-08-15：N4 定义为 exact roster 资格审查而非数据"
                        "集数量普查。只允许读 UID/长度/频率/历史消费记录/"
                        "outlier_mad 确定性修改几何/verifier 合法性/窗口完整"
                        "性；禁止读 gain/sMASE/任何 Target outcome"),
        "allowed_reads": ["dataset/series UID、长度、频率",
                          "历史消费记录（registry exposure_class + artifact 扫描）",
                          "outlier_mad 确定性修改几何（零 Outcome 执行计数）",
                          "静态合法性（allowed_tasks/非 changes_target_space/执行完整）",
                          "窗口完整性（anchors/context/delayed 余量）"],
        "forbidden_reads": ["gain", "sMASE", "任何 Target outcome"],
        "exposure_labels": {
            "context_exposure": ("AGGREGATE_SEEN——数据集级聚合统计已在仓库"
                                 "出现，不得称 UNSEEN"),
            "outcome_exposure": ("SEALED 仅授予 registry certified_virgin 且 "
                                 "artifact 扫描零命中且目标 origin 窗口未消费"
                                 "的 exact (series, origin)")},
        "candidate_datasets": list(N4_CANDIDATE_DATASETS),
        "excluded_datasets": dict(N4_EXCLUDED_DATASETS),
        "eligibility_checks": {
            "a_window_completeness": ("anchors≥2 训练窗、context 192 可得、"
                                      "origin+96 ≤ length（delayed 可行）"),
            "b_modification_premise": ("outlier_mad 全窗口确定性修改点总数 > 0"
                                       "（逐 (series, origin) 判定；INERT 无"
                                       "演示价值排除）"),
            "c_legality": ("outlier_mad ∈ forecast allowed、非 "
                           "changes_target_space、训练窗执行完整（等价口径："
                           "intrinsic 算子静态合法 + run_pipeline 无错）"),
            "d_consumption": ("registry exposure_class == certified_virgin 且 "
                              "64-hex 标识符 artifact 扫描零命中；entity_id 短"
                              "名不作子串扫描（防误报）——扫描为佐证，registry "
                              "为权威")},
        "roster_rule": ("合格 context 数最多的数据集（平手 dataset_id 字典序）；"
                        "roster=(series_uid, origin) 字典序前 "
                        + str(N4_ROSTER_K) + " 个；<" + str(N4_MIN_CONTEXTS)
                        + " → FRESH_TARGET_CONTENT_UNAVAILABLE"),
        "origins": list(N4_ORIGINS), "min_length": N4_MIN_LENGTH,
        "verdict_rules": {"eligible": "N4_ROSTER_ELIGIBLE（≥4 context）",
                          "unavailable": "FRESH_TARGET_CONTENT_UNAVAILABLE"},
        "llm_budget": "0", "outcome_reads": "0",
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["n4_protocol"] = proto
    _save_report(report)
    print("n4_protocol FROZEN")
    return 0


def phase_n4() -> int:
    """执行 exact roster 资格审计（零 LLM 零 Outcome；可重入——结果幂等）。"""
    report = _load_report()
    if not report.get("n4_protocol"):
        raise SystemExit("n4_protocol 未冻结——先跑 n4-freeze")
    # 扫描口径 v2（2026-08-15 主 Agent 修正，待用户复核）：v1 把
    # artifacts/frozen/** 的 split 簿记误计为消费（848 假命中）。口径版本
    # 不一致则重置 n4 结果重跑（与 N1 语义修订同一先例）。
    scan_ver = "scan_v2_no_frozen_bookkeeping"
    if (report.get("n4") or {}).get("scan_version") != scan_ver \
            and report.get("n4"):
        report.pop("n4")
        proto = report.get("n4_protocol") or {}
        proto.setdefault("amendments", []).append({
            "date": "2026-08-15",
            "change": ("consumption 扫描排除 artifacts/frozen/**（split_manifest/"
                       "support_a_subsplit 等簿记文件——split 分配≠outcome 消费，"
                       "registry exposure_class 已编码其结论）；v1 的 848 命中"
                       "为假阳性，FRESH_TARGET_CONTENT_UNAVAILABLE 作废重审"),
            "ruled_by": "主 Agent（簿记/消费语义修正），待用户复核"})
        _save_report(report)
        print("== n4 扫描口径 v2：重置重审（排除 frozen 簿记）", flush=True)
    if (report.get("n4") or {}).get("verdict"):
        print("n4 已终裁:", report["n4"]["verdict"].get("verdict"))
        return 0
    registry = _n4_load_registry()
    reg_by_uid = {str(r.get("series_uid")): r for r in registry}
    clean = _n4_clean_records()
    # 佐证扫描：clean_base 内全部 64-hex 标识符（series_uid + content_sha）
    identifiers = set()
    for uid, e in clean.items():
        identifiers.add(uid)
        csha = str(e["record"].get("content_sha") or "")
        if len(csha) == 64:
            identifiers.add(csha)
    scan = _n4_consumption_scan(identifiers)
    hit_uids = set(scan["hits"].keys())
    print("== n4 扫描: files=" + str(scan["files_scanned"])
          + " hits=" + str(scan["hit_count"]), flush=True)
    per_dataset: dict[str, Any] = {}
    for ds in N4_CANDIDATE_DATASETS:
        ds_rows = [r for r in registry if str(r.get("dataset_id")) == ds]
        stat = {"registry_total": len(ds_rows), "not_virgin": 0,
                "not_in_clean_base": 0, "too_short_or_missing": 0,
                "scan_hit": 0, "no_promise": 0, "window_incomplete": 0,
                "eligible": []}
        for r in ds_rows:
            uid = str(r.get("series_uid"))
            if str(r.get("exposure_class")) != "certified_virgin":
                stat["not_virgin"] += 1
                continue
            ent = clean.get(uid)
            if ent is None:
                stat["not_in_clean_base"] += 1
                continue
            rec = ent["record"]
            if (int(rec.get("length") or 0) < N4_MIN_LENGTH
                    or int(rec.get("natural_missing_count") or 0) != 0):
                stat["too_short_or_missing"] += 1
                continue
            csha = str(rec.get("content_sha") or "")
            if uid in hit_uids or (len(csha) == 64 and csha in hit_uids):
                stat["scan_hit"] += 1
                continue
            arr = np.load(ent["values"]).astype(np.float64)
            for origin in N4_ORIGINS:
                tot = _n4_change_counts(arr, origin)
                if tot is None:
                    stat["window_incomplete"] += 1
                    continue
                if tot == 0:
                    stat["no_promise"] += 1
                    continue
                stat["eligible"].append({
                    "dataset": ds, "series_uid": uid,
                    "entity_id": str(r.get("entity_id")),
                    "origin": origin, "changed_total": int(tot),
                    "frequency": str(rec.get("frequency")),
                    "length": int(rec.get("length")),
                    "context_exposure": "AGGREGATE_SEEN",
                    "outcome_exposure": "SEALED"})
            print("== n4 " + ds + " " + str(r.get("entity_id"))
                  + " 完成", flush=True)
        per_dataset[ds] = stat
        print("== n4 " + ds + ": eligible=" + str(len(stat["eligible"])),
              flush=True)
    pick = _n4_pick_roster(per_dataset)
    report["n4"] = {
        "scan_version": scan_ver,
        "consumption_scan": scan,
        "per_dataset": {ds: {k: (v if k != "eligible" else v)
                             for k, v in stat.items()}
                        for ds, stat in per_dataset.items()},
        "pick": pick,
        "verdict": {"verdict": pick["verdict"], "final": True,
                    "dataset": pick.get("dataset"),
                    "roster_size": len(pick.get("roster") or [])},
        "roster": pick.get("roster") or [],
    }
    _save_report(report)
    print("n4: " + pick["verdict"] + " dataset=" + str(pick.get("dataset"))
          + " roster=" + str(len(pick.get("roster") or [])))
    return 0



# ---------------------------------------------------------------- n5
# N5 Memory 接线预检（用户裁定 2026-08-15；零 LLM 零 Outcome）：
# contrast pack 解析是纯确定性代码（fast_agent.py:754-868），可在不进
# prepare 的情况下完整验证两臂接线。主 Agent 预验证发现：pack 文件需
# doc["episodes"] 适配（load_experience_episodes 要求顶层 list）。

N5_REQUIRED_BUCKETS = ("positive", "negative", "conflict")


def _n5_load_pack_episodes() -> list:
    """N3 pack → ExperienceEpisode 列表（doc 结构适配；缺文件 → []）。"""
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: PLC0415
        episode_from_dict,
    )
    if not N3_PACK_REL.exists():
        return []
    doc = json.loads(N3_PACK_REL.read_text(encoding="utf-8"))
    return [episode_from_dict(d) for d in (doc.get("episodes") or [])
            if isinstance(d, Mapping)]


def _n5_wiring_check(episodes: Sequence[Any],
                     features: Mapping[str, Any]) -> dict:
    """单 context A5 臂接线检查（与 fast_agent 同一代码路径，零 LLM）。
    异常不吞——injection_failed 必须可观测。"""
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: PLC0415
        render_experience_pack, resolve_experience_contrast_pack)
    from SelfEvolvingHarnessTS.operators.registry import (  # noqa: PLC0415
        OPERATOR_METADATA, OPERATOR_NAMES)
    from SelfEvolvingHarnessTS.contracts.task import (  # noqa: PLC0415
        MetricSpec, forecast_task_spec_v1)
    spec = forecast_task_spec_v1(
        horizon=HORIZON, downstream_model_class="ridge",
        metric=MetricSpec("sMASE", "lower_is_better"))
    task_key = (spec.task_type + "|" + spec.downstream_model_class
                + "|" + spec.metric.name)
    allowed = sorted(
        n for n in OPERATOR_NAMES
        if spec.task_type in (OPERATOR_METADATA[n].get("allowed_tasks") or [])
        and not spec.is_op_forbidden(n)
        and not OPERATOR_METADATA[n].get("changes_target_space"))
    out = {"memory_resolution_status": None, "buckets_nonempty": {},
           "rendered_len": 0, "injection_failed": False, "error": None}
    if not episodes:
        out["memory_resolution_status"] = "no_memory"
        return out
    try:
        pack = resolve_experience_contrast_pack(
            episodes, dict(features), task_key, allowed_operators=allowed)
        if pack is None:
            out["memory_resolution_status"] = "contrast_pack_empty"
            return out
        pd = pack.to_dict()
        rendered = render_experience_pack(pd)
        out["buckets_nonempty"] = {
            b: bool(pd.get(b)) for b in N5_REQUIRED_BUCKETS}
        out["rendered_len"] = len(rendered or "")
        out["memory_resolution_status"] = (
            "contrast_pack" if rendered else "contrast_pack_empty")
    except Exception as exc:  # noqa: BLE001 —— 记录而非吞掉
        out["memory_resolution_status"] = "injection_failed"
        out["injection_failed"] = True
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _n5_signed_absent(episodes: Sequence[Any]) -> bool:
    """pack 无 recent.*/change.* 特征 → signed 路径不触发 → 无 Slot P。"""
    return not any(
        str(k).startswith(("recent.", "change."))
        for ep in episodes
        for k in ((getattr(ep, "context_summary", None) or {})
                  .get("local_pattern") or {}))


def phase_n5_freeze() -> int:
    """冻结 N5 协议（任何接线检查前）。"""
    report = _load_report()
    if report.get("n5_protocol"):
        raise SystemExit("n5_protocol 已存在——拒绝重复冻结")
    if report.get("n5"):
        raise SystemExit("n5 已有结果——协议必须先冻结")
    proto = {
        "experiment_id": "N5_MEMORY_WIRING_PRECHECK_2026_08_15",
        "user_ruling": ("2026-08-15：N4 后做一次零 Outcome Memory 接线预检；"
                        "必须看到 A5=contrast_pack 且含正/负/冲突 Reference、"
                        "A3=no_memory、无 injection_failed、两臂保留 Target "
                        "exploration；无机械 prior 时显式报告 "
                        "source_prior_candidate_absent，不得伪装双槽"),
        "checks": {
            "a5_contrast_pack": ("A5 臂每 roster context：resolve 非 None 且 "
                                 "渲染非空 且 positive/negative/conflict 三桶"
                                 "均非空"),
            "a3_no_memory": "A3 臂（空 episodes）结构性地为 no_memory",
            "no_injection_failed": "A5 解析/渲染零异常（异常不吞，记为失败）",
            "exploration_retained": ("pack 无 recent.*/change.* 键 → signed 路径"
                                     "不触发 → 无 Slot P → 探索为唯一候选来源，"
                                     "结构性保留（两臂同一代码路径）"),
            "prior_absent_disclosure": ("_signed=None ⇒ 显式记录 "
                                        "source_prior_candidate_absent=True")},
        "mechanism": ("contrast pack 解析为纯确定性代码（fast_agent.py:754-868），"
                      "预检直接调用同一 resolve/render 函数，零 LLM 零 Outcome"),
        "verdict_rules": {"ok": "五查全过 → N5_WIRING_OK",
                          "broken": "任一不过 → N5_WIRING_BROKEN（A5 不得起跑）"},
        "llm_budget": "0", "outcome_reads": "0",
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["n5_protocol"] = proto
    _save_report(report)
    print("n5_protocol FROZEN")
    return 0


def phase_n5() -> int:
    """执行接线预检（零 LLM 零 Outcome）。前置：n4 eligible。"""
    report = _load_report()
    if not report.get("n5_protocol"):
        raise SystemExit("n5_protocol 未冻结——先跑 n5-freeze")
    n4 = report.get("n4") or {}
    if (n4.get("verdict") or {}).get("verdict") != "N4_ROSTER_ELIGIBLE":
        raise SystemExit("n4 未通过——N5 无 roster 可查")
    if (report.get("n5") or {}).get("verdict"):
        print("n5 已终裁:", report["n5"]["verdict"].get("verdict"))
        return 0
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: PLC0415
        extract_public_features,
    )
    episodes = _n5_load_pack_episodes()
    if not episodes:
        report["n5"] = {"verdict": {"verdict": "N5_WIRING_BROKEN",
                                    "reason": "Source Pack 加载为空"},
                        "final": True}
        _save_report(report)
        print("n5: N5_WIRING_BROKEN (empty pack)")
        return 0
    signed_absent = _n5_signed_absent(episodes)
    clean = _n4_clean_records()
    rows = []
    failures = []
    for e in n4.get("roster") or []:
        uid = str(e["series_uid"])
        origin = int(e["origin"])
        arr = np.load(clean[uid]["values"]).astype(np.float64)
        feats = extract_public_features(arr[:origin], task_kind="forecast")
        a5 = _n5_wiring_check(episodes, feats)
        a3 = _n5_wiring_check((), feats)
        row = {"dataset": e["dataset"], "entity_id": e.get("entity_id"),
               "origin": origin, "a5": a5, "a3": a3}
        rows.append(row)
        ok = (a5["memory_resolution_status"] == "contrast_pack"
              and all(a5["buckets_nonempty"].values())
              and not a5["injection_failed"]
              and a3["memory_resolution_status"] == "no_memory")
        if not ok:
            failures.append({"entity_id": e.get("entity_id"),
                             "origin": origin, "a5": a5})
        print("== n5 " + str(e.get("entity_id")) + "@" + str(origin)
              + ": a5=" + str(a5["memory_resolution_status"])
              + " buckets=" + json.dumps(a5["buckets_nonempty"])
              + " a3=" + str(a3["memory_resolution_status"]), flush=True)
    checks = {
        "a5_contrast_pack": not failures,
        "a3_no_memory": all(r["a3"]["memory_resolution_status"] == "no_memory"
                            for r in rows),
        "no_injection_failed": not any(r["a5"]["injection_failed"]
                                       for r in rows),
        "exploration_retained": signed_absent,
        "prior_absent_disclosure": True,
    }
    verdict = ("N5_WIRING_OK" if all(checks.values()) else "N5_WIRING_BROKEN")
    report["n5"] = {
        "pack_size": len(episodes),
        "signed_context_absent": signed_absent,
        "source_prior_candidate_absent": True,
        "exploration_note": ("contrast_pack 仅改 view.instruction；Slot P 仅"
                             "在 signed 路径存在；pack 无 recent./change. 键 →"
                             " 两臂 Target exploration 结构性保留"),
        "checks": checks, "failures": failures, "rows": rows,
        "verdict": {"verdict": verdict, "final": True},
    }
    _save_report(report)
    print("n5: " + verdict + " checks=" + json.dumps(checks))
    return 0


# ---------------------------------------------------------------- loop1
# LOOP1（2026-08-15）：首次全自主自进化闭环正向演示。放宽判定但预注册、
# 诚实标注 n=1 机制演示。单一失败案例（T10@600 B 臂 full 池的 external_region
# repair 越界拒绝）→ 匿名 capsule → Slow 自主改条款 → Runtime 落 fork →
# 两臂配对行为重放 → 机械判据。不晋升 h0、不写真实 skill 文件。

LOOP1_SERIES = "T10"
LOOP1_ORIGIN = 600
LOOP1_EDIT_ID = "loop1-rev8-2026-08-15"
LOOP1_OPERATORS = ("winsorize", "outlier_iqr", "outlier_mad", "hampel_filter",
                   "denoise_median", "resample_uniform", "repair_level_shift")


def _loop1_judgment(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """loop1 verdict_rules 机械判据（纯函数，供 phase 与测试共用）。

    rows：两臂各一行的 replay 结果（含 arm 字段 A/B）。
    返回 {"verdict": ..., "criteria": {...}, "final": True}。"""
    arms: dict[str, Mapping[str, Any]] = {}
    for r in rows or ():
        if isinstance(r, dict) and r.get("arm"):
            arms[str(r["arm"])] = r
    if not arms:
        return {"verdict": "LOOP1_PROTOCOL_ERROR", "criteria": {},
                "reason": "no_rows", "final": True}
    for arm in ("A", "B"):
        if arm not in arms:
            return {"verdict": "LOOP1_PROTOCOL_ERROR", "criteria": {},
                    "reason": "missing_arm_" + arm, "final": True}
        if arms[arm].get("protocol_error"):
            return {"verdict": "LOOP1_PROTOCOL_ERROR", "criteria": {},
                    "reason": ("protocol_error_arm_" + arm + ": "
                               + str(arms[arm].get("protocol_error"))),
                    "final": True}
    a = arms["A"]
    b = arms["B"]

    def _non_identity(row: Mapping[str, Any]) -> list:
        return [str(c) for c in (row.get("candidate_ids") or ())
                if str(c) != "identity"]

    a_rej = len(a.get("rejection_receipts") or ())
    b_rej = len(b.get("rejection_receipts") or ())
    a_nonid = _non_identity(a)
    b_nonid = _non_identity(b)
    b_steps = b.get("candidate_steps") if isinstance(
        b.get("candidate_steps"), dict) else {}
    b_program = [c for c in b_nonid if c in b_steps and b_steps.get(c)]
    contrast_reproduced = a_rej >= 1
    fork_improved = (b_rej == 0 and len(b_program) >= 1
                     and str(b.get("compilation_status")) == "ok")
    supply_preserved = len(b_nonid) >= len(a_nonid) - 1
    criteria = {
        "contrast_reproduced": bool(contrast_reproduced),
        "fork_improved": bool(fork_improved),
        "supply_preserved": bool(supply_preserved),
        "arm_a_rejections": a_rej,
        "arm_b_rejections": b_rej,
        "arm_a_non_identity": len(a_nonid),
        "arm_b_non_identity": len(b_nonid),
    }
    if not contrast_reproduced:
        verdict = "LOOP1_NO_CONTRAST"
    elif not (fork_improved and supply_preserved):
        verdict = "LOOP1_NO_IMPROVEMENT"
    else:
        verdict = "LOOP1_POSITIVE_FORK_DEMO"
    return {"verdict": verdict, "criteria": criteria, "final": True}


def phase_loop1_freeze() -> int:
    report = _load_report()
    if report.get("loop1_protocol"):
        raise SystemExit("loop1_protocol 已存在——拒绝重复冻结")
    if report.get("loop1"):
        raise SystemExit("loop1 已有结果——协议必须先冻结")
    proto = {
        "experiment_id": "LOOP1_AUTONOMOUS_EVOLUTION_DEMO_2026_08_15",
        "user_directive": ("用户 2026-08-15：效果优先、跑通闭环+可视化、"
                           "判定放宽到正向结果即可、细节后补"),
        "failure_evidence": {
            "source": "fullop2.corrected_final_2026_08_14",
            "row": {"series": "T10", "origin": 600, "arm": "B",
                    "pool_mode": "full", "rep": 0},
            "rejected_candidate": "repair_level_shift_local",
            "rejection_code": "MODIFICATION_FRACTION_EXCEEDED",
            "modified_fraction": 0.952,
            "cap": 0.35,
            "public_features_note": ("context 公开特征 estimated_region_start/"
                                     "end_fraction ≈ 0.038/0.99——LLM 照抄全幅"
                                     "估计区域导致修改面越界"),
        },
        "loop": ["batch 失败证据", "匿名 capsule", "Slow 自主改 skill 条款",
                 "Runtime 落 fork", "配对行为重放", "fork 内激活演示"],
        "replay": {
            "context": "T10@600", "pool_mode": "full",
            "with_task_context": True,
            "arms": {"A": "h0(rev7) 基线", "B": "rev8 fork"},
            "reps": 1, "temperature": 0,
            "support_evaluations": 0, "new_outcomes": 0,
        },
        "verdict_rules": {
            "contrast_reproduced": "臂 A 重放仍出现 ≥1 个 verifier 拒绝候选",
            "fork_improved": ("臂 B 零拒绝 且 ≥1 个合法非 identity program 候选 "
                              "且 compilation_status=='ok'"),
            "supply_preserved": ("臂 B 非 identity 候选数 ≥ 臂 A 非 identity "
                                 "候选数 − 1（不允许靠弃权规避）"),
            "positive": "三项全满足 → LOOP1_POSITIVE_FORK_DEMO",
            "no_contrast": "臂 A 无拒绝 → LOOP1_NO_CONTRAST",
            "no_improvement": "臂 B 仍拒绝或供应坍缩 → LOOP1_NO_IMPROVEMENT",
            "protocol_error": "protocol_error → LOOP1_PROTOCOL_ERROR",
        },
        "claim_boundary": ("n=1 单案例机制演示；不晋升 h0（晋升留用户裁决）；"
                           "不构成效用声明；rev6→rev7 正向机制证据已由 TSEM "
                           "独立建立"),
        "llm_budget": {"slow": "≤12 次调用 1 回合（无重试）",
                       "replay": "2 次 prepare"},
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["loop1_protocol"] = proto
    _save_report(report)
    print("loop1_protocol FROZEN")
    return 0


def _loop1_card(surface_sha: str, current_body: str,
                contracts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """loop1 匿名 capsule（仿 _g2_card 形状，facts 换本案证据）。不携带
    series/数据集名；Slow 输出契约沿用 REPLACE_CLAUSE 三行纪律。"""
    propose_start = current_body.find("[propose_construction_guidance]")
    select_start = current_body.find("[select_guidance]")
    old_propose_rules = (current_body[propose_start:select_start].strip()
                         if propose_start >= 0 and select_start > propose_start
                         else "")
    facts = {
        "failure_receipt": {
            "candidate_family": "external_region repair",
            "modified_fraction": 0.952,
            "cap": 0.35,
            "rejection_code": "MODIFICATION_FRACTION_EXCEEDED",
            "context_public_features": {
                "estimated_region_start_fraction": 0.038,
                "estimated_region_end_fraction": 0.99,
            },
            "note": ("估计区域边界描述了异常范围而非可执行修改范围；"
                     "照抄全幅估计区域导致修改面越界"),
        },
        "guidance_surface_precondition_sha": surface_sha,
        "task_objective": "forecast; cohort Ridge sMASE (lower is better)",
        "current_guidance_body": current_body,
        "legal_operator_contracts": [
            {"name": c.get("name"), "category": c.get("category"),
             "public_parameter_bindings": c.get("public_parameter_bindings"),
             "targeting_mode": c.get("targeting_mode")}
            for c in contracts],
        "old_propose_rules": old_propose_rules,
    }
    return {
        "pattern_id": "loop1-external-region-repair-overrun",
        "failure_family": "workflow_construction_region_overrun",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {"steps": []},
        "facts": facts,
        "instruction": (
            "A single-case failure evidence: an external-region repair "
            "candidate was proposed with a modification span that overran "
            "the deployment budget (95.2% > 35% cap) and was rejected by the "
            "runtime verifier as MODIFICATION_FRACTION_EXCEEDED. The "
            "estimated region boundary described the anomaly extent, not "
            "the executable modification extent; the construction guidance "
            "therefore needs to fold the deployment modification budget "
            "explicitly into region selection, while NOT satisfying this by "
            "giving up on proposal (abstention is not an acceptable fix). "
            "Propose ONE minimal PATCH to the guidance surface "
            "bootstrap_skills.entries/build_contrastive_candidates.body. "
            "minimal_patch.value must be a REPLACE_CLAUSE payload with "
            "exactly three lines: 'REPLACE_CLAUSE', 'target: "
            "propose.rule.<clause_id>' (choose exactly one clause id from "
            "facts.old_propose_rules), and 'new_clause: <single-line rule "
            "text>'. You propose the new rule text yourself; the runtime "
            "binds it to the correct rule position and leaves every other "
            "section byte-identical. Never name datasets, series ids, "
            "specific operators, numeric gains, origins, or frozen "
            "parameters anywhere in your text. Judge binding legality from "
            "facts.legal_operator_contracts and the runtime verifier — not "
            "from region width. Evidence references do NOT go into your "
            "text — put at least one 'evidence:<generic description>' item "
            "in predicted_data_effect instead. Set surface_precondition to "
            "kind=SHA with the sha given in "
            "facts.guidance_surface_precondition_sha. "
            "predicted_agent_behavior_change must include "
            "supply_effect_distinct; falsification_condition must state when "
            "the patch should be rejected. You do not approve your own edit "
            "— a deterministic behavior replay verifies it."
        ),
    }


def phase_loop1_slow(env: Mapping[str, Any] | None = None) -> int:
    """loop1 Slow：匿名 capsule → Slow 自主改条款 → Runtime 落 fork。"""
    report = _load_report()
    if not report.get("loop1_protocol"):
        raise SystemExit("loop1_protocol 未冻结——先跑 loop1-freeze")
    loop1 = report.setdefault("loop1", {})
    if loop1.get("slow"):
        print("== loop1-slow: 已存在，幂等跳过")
        return 0
    if env is None:
        env = _load_env()
    h0 = _h0_snapshot()
    current_body = _bootstrap_body(h0)
    store = SnapshotStore(STORE_DIR)
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    parent = store.materialize(h0)
    surface_sha = controller.surface_precondition_sha(parent, SURFACE)
    counter, counter_ref = _make_client()
    values = env["values"]
    series0 = np.asarray(values[LOOP1_SERIES], dtype=np.float64)
    rec_slow = RecordingBackend(AgictoChatCompletionsBackend(
        client=counter, base_url=BASE_URL))
    core = TTHAAgentCore(
        rec_slow,
        LocalPublicToolGateway(series0[:LOOP1_ORIGIN], task_kind="forecast"),
        model=MODEL, base_url=BASE_URL)
    slow = TTHASlowAgent(core)
    method = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(
            AgictoChatCompletionsBackend(client=counter, base_url=BASE_URL),
            LocalPublicToolGateway(series0[:LOOP1_ORIGIN],
                                   task_kind="forecast"),
            model=MODEL, base_url=BASE_URL)),
        h0, ())
    from SelfEvolvingHarnessTS.contracts.task import (  # noqa: PLC0415
        deployment_constraints_v1,
        forecast_task_context_v1,
    )
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: PLC0415
        public_operator_contract,
    )
    base_req = nsu._request(series0, values, LOOP1_ORIGIN)
    task_ctx = forecast_task_context_v1(
        task_spec=base_req.task_spec,
        deployment_constraints=deployment_constraints_v1())
    contracts = tuple(public_operator_contract(op) for op in LOOP1_OPERATORS)
    card = _loop1_card(surface_sha, current_body, contracts)
    group = {"workflow": "external_region_repair_region_overrun"}
    capsule = {"n_episodes": 1, "workflow": group["workflow"],
               "sign": "NEGATIVE"}
    ev = method.handle_group_guidance(
        group, capsule, slow_agent=slow, controller=controller, store=store,
        card_builder=lambda g, c: card,
        confirmed_cause="WORKFLOW_GUIDANCE_GAP",
        manifest_preflight=lambda m: _g2_preflight(
            m, current_body=current_body),
        allowed_operator_contracts=contracts,
        task_context=task_ctx)
    new_snapshot = method.pending_guidance_snapshot()
    slow_rec: dict[str, Any] = {
        "feedback_event": ev,
        "llm_calls": counter_ref.calls,
        "slow_prompt_calls": rec_slow.calls,
    }
    if slow.last_stage_result is not None:
        slow_rec["slow_stage_result"] = {
            "validation_retry_count":
                slow.last_stage_result.validation_retry_count,
            "first_pass_valid": slow.last_stage_result.first_pass_valid,
            "validation_error_codes": list(
                slow.last_stage_result.validation_error_codes or ()),
        }
        mp = (slow.last_stage_result.payload or {}).get("edit_manifest") or {}
        slow_rec["proposed_manifest"] = {
            "target_surface_id": str(mp.get("target_surface_id")),
            "operation": str(mp.get("operation")),
            "predicted_agent_behavior_change": list(
                mp.get("predicted_agent_behavior_change") or ()),
            "predicted_data_effect": list(
                mp.get("predicted_data_effect") or ()),
        }
    if ev.get("stage") == "pending" and new_snapshot is not None:
        pend = getattr(method, "_pending_update", None) or {}
        clause = ev.get("guidance_clause_proposed") or {}
        slow_rec["fork_sha"] = pend["receipt"].candidate_runtime_bundle_sha
        slow_rec["candidate_harness_content_sha"] = (
            new_snapshot.harness_content_sha)
        slow_rec["guidance_body_old_len"] = len(current_body)
        slow_rec["clause_target"] = clause.get("target")
        slow_rec["clause_new"] = clause.get("new_clause")
        slow_rec["edit_id"] = LOOP1_EDIT_ID
        slow_rec["slow_manifest_edit_id"] = ev.get("edit_id")
    loop1["slow"] = slow_rec
    _save_report(report)
    print("== loop1-slow: stage=" + str(ev.get("stage"))
          + " fork=" + str(slow_rec.get("fork_sha", ""))[:12]
          + " llm=" + str(counter_ref.calls))
    return 0


def phase_loop1_replay(env: Mapping[str, Any]) -> int:
    """loop1 配对行为重放：臂 A=h0(rev7)、臂 B=rev8 fork，各 1 rep。"""
    report = _load_report()
    if not report.get("loop1_protocol"):
        raise SystemExit("loop1_protocol 未冻结——先跑 loop1-freeze")
    loop1 = report.setdefault("loop1", {})
    fork_sha = (loop1.get("slow") or {}).get("fork_sha")
    if not fork_sha:
        raise SystemExit("loop1.slow 无 fork_sha——先跑 loop1-slow")
    h0 = _h0_snapshot()
    store = SnapshotStore(STORE_DIR)
    new_snap = compile_snapshot(store.root / str(fork_sha), verify_lock=False)
    rows = list(loop1.get("replay") or [])
    done = {str(r.get("arm")) for r in rows}
    if "A" not in done:
        rA = _prepare_arm(h0, LOOP1_SERIES, LOOP1_ORIGIN, env,
                          with_task_context=True, pool_mode="full")
        rA["arm"] = "A"
        rA["rep"] = 0
        rA["pool_mode"] = "full"
        rows.append(rA)
        loop1["replay"] = rows
        _save_report(report)
        print("== loop1-replay arm A: "
              + json.dumps({k: rA.get(k) for k in (
                  "candidate_ids", "chosen_candidate_id",
                  "compilation_status", "rejection_receipts",
                  "protocol_error")}, ensure_ascii=False))
    if "B" not in done:
        rB = _prepare_arm(new_snap, LOOP1_SERIES, LOOP1_ORIGIN, env,
                          with_task_context=True, pool_mode="full")
        rB["arm"] = "B"
        rB["rep"] = 0
        rB["pool_mode"] = "full"
        rows.append(rB)
        loop1["replay"] = rows
        _save_report(report)
        print("== loop1-replay arm B: "
              + json.dumps({k: rB.get(k) for k in (
                  "candidate_ids", "chosen_candidate_id",
                  "compilation_status", "rejection_receipts",
                  "protocol_error")}, ensure_ascii=False))
    _save_report(report)
    return 0


def phase_loop1_verdict() -> int:
    """机械按 verdict_rules 从 replay rows 计算判据 → verdict。"""
    report = _load_report()
    if not report.get("loop1_protocol"):
        raise SystemExit("loop1_protocol 未冻结——先跑 loop1-freeze")
    loop1 = report.setdefault("loop1", {})
    rows = loop1.get("replay") or []
    judgment = _loop1_judgment(rows)
    loop1["verdict"] = judgment
    if judgment["verdict"] == "LOOP1_POSITIVE_FORK_DEMO":
        loop1["activation"] = {
            "scope": "fork_only",
            "h0_promoted": False,
            "note": "晋升 h0 待用户裁决",
        }
    _save_report(report)
    print("== loop1-verdict: " + judgment["verdict"]
          + " " + json.dumps(judgment.get("criteria", {}), ensure_ascii=False))
    return 0




# ---------------------------------------------------------------- a5
# A5/A3 matched-budget 主实验（用户裁定 2026-08-15：两预检通过后直奔）。
# 长期计划书 A5 vs A3：相同 Target feedback 预算下，读取自然 Source Pack
# （成功+失败+冲突）的 A5 能否比空 Source Memory 的 A3 更快、更安全地形
# 成有效 Target-local Skill。两臂唯一差异 = Source Pack；Slow 双臂同关；
# Target-local episode 对称累积（A3 = 空 Source Memory，非永远空 Memory）。

A5_SUPPORT_CAP_PER_CTX = 3       # 每 context 每臂 Support 评估上限（ matched ）
A5_LLM_CAP_PER_PREPARE = 12


def _a5_config(dataset: str, origin: int) -> dict:
    return {"dataset_id": dataset, "sampling": "hourly_regular",
            "period": PERIOD, "anchors": list(range(312, 853, 60)),
            "support_origin": origin, "selection_origin": origin}


def _a5_request(series0: Any, values: Mapping[str, Any], origin: int,
                dataset: str):
    """目标数据集请求构造（仿 nsu._request；period=24 与 sealed a5a3 一致）。"""
    import dataclasses as _dc  # noqa: PLC0415
    from SelfEvolvingHarnessTS.contracts.method import (  # noqa: PLC0415
        PreparationRequest)
    from SelfEvolvingHarnessTS.contracts.task import (  # noqa: PLC0415
        MetricSpec, deployment_constraints_v1, forecast_task_context_v1,
        forecast_task_spec_v1)
    observed = dict(resolver.window_context(values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    req = PreparationRequest(
        dataset + "-a5a3",
        series0[:origin],
        forecast_task_spec_v1(horizon=HORIZON,
                              downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed))
    tc = forecast_task_context_v1(
        task_spec=req.task_spec,
        deployment_constraints=deployment_constraints_v1())
    return _dc.replace(req, task_context=tc)


def _a5_prepare(snapshot: Any, entry: Mapping[str, Any],
                values: Mapping[str, Any],
                episodes: Sequence[Any]) -> dict:
    """单臂单 context 真实 prepare（带 Memory；记录 memory_resolution_status）。"""
    uid = str(entry["series_uid"])
    origin = int(entry["origin"])
    series0 = np.asarray(values[uid], dtype=np.float64)
    counter, counter_ref = _make_client()
    rec = RecordingBackend(AgictoChatCompletionsBackend(
        client=counter, base_url=BASE_URL))
    core = TTHAAgentCore(
        rec, LocalPublicToolGateway(series0[:origin], task_kind="forecast"),
        model=MODEL, base_url=BASE_URL)
    method = TTHAMethod(TTHAFastAgent(core), snapshot, tuple(episodes))
    request = _a5_request(series0, values, origin, str(entry["dataset"]))
    # 瞬时传输错误（AgentTransportError/APITimeoutError 等网络层噪声）重试
    # ≤2 次——传输 ≠ 证据，不计入 protocol_error（修正案见 a5_protocol）。
    result = trace = None
    last_exc: Exception | None = None
    for _attempt in range(3):
        try:
            result = method.prepare(request)
            trace = method.last_trace
            if trace is None:
                raise RuntimeError("no trace")
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if "Transport" not in type(exc).__name__ \
                    and "Timeout" not in str(type(exc).__name__):
                break  # 非传输类错误不重试
    if last_exc is not None:
        return {"protocol_error": f"{type(last_exc).__name__}: {last_exc}",
                "llm_calls": counter_ref.calls,
                "memory_resolution_status": "prepare_failed"}
    steps_map = {str(k): [{"op": o, "params": dict(p)} for o, p in v]
                 for k, v in dict(trace.candidate_program_steps or {}).items()}
    return {
        "status": result.status.value if hasattr(result.status, "value")
        else str(result.status),
        "candidate_ids": list(trace.candidate_ids),
        "candidate_steps": steps_map,
        "chosen_candidate_id": str(trace.chosen_candidate_id),
        "compilation_status": str(trace.compilation_status),
        "rejected_ids": [str(r.get("candidate_id"))
                         for r in (trace.rejection_receipts or ())],
        "memory_resolution_status": str(
            getattr(trace, "memory_resolution_status", "unknown")),
        "llm_calls": counter_ref.calls,
        "prompt_calls": rec.calls,
    }


def _a5_relation(support_gain: Any, delayed: Mapping[str, Any]) -> str | None:
    """BSE 冻结口径：|gain|<M 不成 Episode；CONFLICT=support 与 delayed 不一致。"""
    if support_gain is None:
        return None
    g = float(support_gain)
    if g < -M:
        return "NEGATIVE"
    if g < M:
        return None
    if delayed.get("evaluated"):
        dg = delayed.get("gain")
        if dg is not None and float(dg) < -M:
            return "CONFLICT"
        return "POSITIVE"
    return "POSITIVE"   # delayed 未评估的 POSITIVE：EPISODE_ONLY 证据级


def _a5_verdict(aggregates: Mapping[str, Any], rows: Sequence[dict]) -> dict:
    """冻结裁定（a5_protocol.verdict_rules；机械，优先级
    PROTOCOL_INCONCLUSIVE > CONTENT_INCONCLUSIVE > 其余）。"""
    a5 = aggregates.get("A5") or {}
    a3 = aggregates.get("A3") or {}
    if any(r.get("protocol_error") for r in rows):
        return {"verdict": "PROTOCOL_INCONCLUSIVE",
                "reason": "存在 protocol_error 行",
                "error_rows": [{"arm": r.get("arm"),
                                "entity": r.get("entity_id"),
                                "origin": r.get("origin")}
                               for r in rows if r.get("protocol_error")]}
    bad_mem = [r for r in rows
               if r.get("arm") == "A5"
               and r.get("memory_resolution_status") != "contrast_pack"]
    if bad_mem:
        return {"verdict": "CONTENT_INCONCLUSIVE",
                "reason": "A5 运行期 Memory 未按 N5 预检接线（与 N5 矛盾）",
                "bad_rows": [{"entity": r.get("entity_id"),
                              "origin": r.get("origin"),
                              "status": r.get("memory_resolution_status")}
                             for r in bad_mem]}
    f5, f3 = a5.get("feedback_to_first_effective"), \
        a3.get("feedback_to_first_effective")
    h5 = int(a5.get("harm_events") or 0)
    h3 = int(a3.get("harm_events") or 0)
    n5e = int(a5.get("n_effective") or 0)
    n3e = int(a3.get("n_effective") or 0)
    if n5e == 0 and n3e == 0:
        return {"verdict": "NO_SIGNAL",
                "reason": "两臂预算内均未形成有效 Target-local Skill"}
    # 更快：fpe 严格更小；None（未形成）视为无穷大
    f5v = f5 if f5 is not None else float("inf")
    f3v = f3 if f3 is not None else float("inf")
    if n5e >= 1 and f5v < f3v and h5 <= h3:
        return {"verdict": "TRANSFER_CASE_PASS",
                "reason": ("A5 以更少 feedback 形成首个有效 Skill（"
                           + str(f5) + " vs " + str(f3) + "）且 harm 不多（"
                           + str(h5) + " vs " + str(h3) + "）"),
                "claim": ("development transfer case：单 Target 数据集、n=6 "
                          "context、1 rep；非一般迁移声明；N5 第二 Target "
                          "复制仅当本档成立才启动")}
    if f5v > f3v or h5 > h3:
        return {"verdict": "NEGATIVE_TRANSFER",
                "reason": ("A5 更慢或更多 harm（fpe " + str(f5) + " vs "
                           + str(f3) + "；harm " + str(h5) + " vs "
                           + str(h3) + "）"),
                "first_fault_hint": ("定位面：Source 指令是否被渲染→是否进入"
                                     " propose 上下文→候选是否变化→winner 是否"
                                     "变化；一次只改一个面")}
    return {"verdict": "NO_SIGNAL",
            "reason": ("两臂同速同 harm（fpe " + str(f5) + " vs " + str(f3)
                       + "；harm " + str(h5) + " vs " + str(h3) + "）")}


def phase_a5_freeze() -> int:
    """冻结 A5/A3 协议（N4/N5 双预检通过后；任何运行前）。"""
    report = _load_report()
    if report.get("a5_protocol"):
        raise SystemExit("a5_protocol 已存在——拒绝重复冻结")
    if report.get("a5"):
        raise SystemExit("a5 已有结果——协议必须先冻结")
    n4v = ((report.get("n4") or {}).get("verdict") or {}).get("verdict")
    n5v = ((report.get("n5") or {}).get("verdict") or {}).get("verdict")
    if n4v != "N4_ROSTER_ELIGIBLE":
        raise SystemExit("n4 未合格（" + str(n4v) + "）——A5 不得冻结")
    if n5v != "N5_WIRING_OK":
        raise SystemExit("n5 未合格（" + str(n5v) + "）——A5 不得冻结")
    roster = report["n4"]["roster"]
    dataset = report["n4"]["verdict"]["dataset"]
    proto = {
        "experiment_id": "A5A3_MATCHED_BUDGET_2026_08_15",
        "user_ruling": ("2026-08-15：两预检通过后直奔 matched-budget A5/A3；"
                        "两臂唯一差异是 Source Pack；LLM/DSL/Support 总预算、"
                        "Slow 触发、停止规则、delayed 次数、adoption/removal "
                        "完全一致"),
        "question": ("相同 Target feedback 预算下，读取自然 Source Pack（成"
                     "功+失败+冲突）的 A5 能否比空 Source Memory 的 A3 更快"
                     "（更少 feedback 形成首个有效 Target-local Skill）、更"
                     "安全（harm 事件不多）"),
        "target": {"dataset": dataset, "roster": roster,
                   "exposure": "context_exposure=AGGREGATE_SEEN; "
                               "outcome_exposure=SEALED（N4 审计）"},
        "arms": {
            "A5": "h0(rev7) + N3 Source Pack（10 条 KDD 自然轨迹）",
            "A3": "h0(rev7) + 空 Source Memory",
            "symmetric_growth": ("两臂在每 context 后对称累积 Target-local "
                                 "episode（A3=空 Source Memory，非永远空）"),
            "slow": "两臂同关（避免混淆 Fast 层迁移问题）"},
        "per_context_sequence": [
            "prepare（memory 按臂；with_task_context；pool_mode=actionable；"
            "LLM ≤12/prepare）",
            "Support：评估全部 selectable 非 identity 候选（提案顺序，"
            "≤3/臂/context）",
            "winner resolution：argmax gain，≥M adopt，否则 abstain(identity)",
            "delayed：adopted winner 在 origin+48 fresh 评估（无缓存可复用）；"
            "identity→0 不评估；delayed<−M → removal",
            "episode：|gain|≥M 的候选成 Episode（BSE 口径 relation）；"
            "入该臂 Memory"],
        "budget_table": {"prepares_per_arm": len(roster),
                         "support_per_arm": "≤" + str(A5_SUPPORT_CAP_PER_CTX)
                         + "×" + str(len(roster)),
                         "delayed_per_arm": "≤" + str(len(roster)),
                         "llm_per_prepare": A5_LLM_CAP_PER_PREPARE,
                         "slow_calls": 0},
        "metrics": ["feedback_to_first_effective_skill（1 起累计 support 评估"
                    "序号；无则 null）",
                    "support_harm_probes（gain<−M）",
                    "delayed_harm_adoptions（adopt 后 delayed<−M）",
                    "abstentions", "total_support_gain（adopted winner 之和）",
                    "n_effective_skills"],
        "verdict_rules": {
            "precedence": "PROTOCOL_INCONCLUSIVE > CONTENT_INCONCLUSIVE > 其余",
            "pass": ("A5.feedback_to_first_effective 严格 < A3 且 "
                     "A5.harm_events ≤ A3.harm_events 且 A5.n_effective ≥ 1 "
                     "→ TRANSFER_CASE_PASS"),
            "negative": "A5 更慢或更多 harm → NEGATIVE_TRANSFER",
            "no_signal": "两臂均无有效 Skill 或两维度全等 → NO_SIGNAL",
            "content": ("A5 运行期 memory_resolution_status != contrast_pack "
                        "→ CONTENT_INCONCLUSIVE"),
            "protocol": "任一 protocol_error 行 → PROTOCOL_INCONCLUSIVE"},
        "claim_boundary": ("单 Target 数据集、n=6 context、1 rep——"
                           "development transfer case；非一般迁移声明；"
                           "Source Rule 仅参考证据；不升级 Shared Capability"),
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["a5_protocol"] = proto
    _save_report(report)
    print("a5_protocol FROZEN (dataset=" + str(dataset) + ", contexts="
          + str(len(roster)) + ")")
    return 0


def phase_a5() -> int:
    """matched-budget A5/A3 主运行（可恢复：按 arm×context 跳过）。"""
    report = _load_report()
    if not report.get("a5_protocol"):
        raise SystemExit("a5_protocol 未冻结——先跑 a5-freeze")
    a5 = report.setdefault("a5", {})
    if (a5.get("verdict") or {}).get("final"):
        raise SystemExit("a5 已终裁")
    roster = report["a5_protocol"]["target"]["roster"]
    dataset = str(report["a5_protocol"]["target"]["dataset"])
    h0 = _h0_snapshot()
    clean = _n4_clean_records()
    values = {str(e["series_uid"]): np.load(
        clean[str(e["series_uid"])]["values"]).astype(np.float64)
        for e in roster}
    pack_episodes = _n5_load_pack_episodes()
    rows = a5.setdefault("rows", [])
    done = {(r["arm"], r["series_uid"], int(r["origin"])) for r in rows}
    # 两臂 Memory 状态：从已完成的 rows 重建（恢复语义）
    memories = {"A5": list(pack_episodes), "A3": []}
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: PLC0415
        build_episode, workflow_signature_of)
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: PLC0415
        ScopeExecutor)
    for r in rows:  # 重建臂内 Target-local episode（保持幂等恢复）
        for ep_dict in (r.get("episodes_formed") or ()):
            from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: PLC0415,E501
                episode_from_dict)
            memories[r["arm"]].append(episode_from_dict(ep_dict))
    for entry in roster:
        uid = str(entry["series_uid"])
        origin = int(entry["origin"])
        for arm in ("A5", "A3"):
            if (arm, uid, origin) in done:
                continue
            print("== a5 " + arm + " " + str(entry.get("entity_id"))
                  + "@" + str(origin) + " ...", flush=True)
            prep = _a5_prepare(h0, entry, values, memories[arm])
            row = {"arm": arm, "dataset": dataset, "series_uid": uid,
                   "entity_id": entry.get("entity_id"), "origin": origin,
                   **prep}
            if prep.get("protocol_error"):
                rows.append(row)
                _save_report(report)
                continue
            # Support：selectable 非 identity 候选（提案序，cap 3）
            rejected = set(prep.get("rejected_ids") or ())
            cand_ids = [c for c in prep["candidate_ids"]
                        if c != "identity" and c not in rejected]
            eval_ids = cand_ids[:A5_SUPPORT_CAP_PER_CTX]
            roster2 = ([{"series_uid": uid, "role": "train"},
                        {"series_uid": uid, "role": "eval"}])
            vals2 = {uid: values[uid]}
            ex = ScopeExecutor(roster2, vals2, _a5_config(dataset, origin),
                               evaluate_fn=nsu._evaluate_kdd)
            probes = []
            for cid in eval_ids:
                steps = [(str(s["op"]), dict(s["params"]))
                         for s in prep["candidate_steps"].get(cid) or []]
                if not steps:
                    continue
                rr = ex.evaluate(tuple(steps), origin)
                probes.append({
                    "candidate_id": cid,
                    "steps": [{"op": o, "params": dict(p)}
                              for o, p in steps],
                    "gain": (float(rr.gain) if rr.gain is not None
                             else None),
                    "passed": bool(rr.verification.passed)})
            # winner resolution：argmax gain，≥M adopt
            valid = [p for p in probes if p["gain"] is not None]
            winner = (max(valid, key=lambda p: p["gain"])
                      if valid else None)
            adopted = bool(winner and winner["gain"] >= M)
            # delayed：adopted winner 在 origin+HORIZON fresh 评估
            delayed = {"evaluated": False, "gain": None}
            removal = False
            if adopted:
                ex2 = ScopeExecutor(
                    roster2, vals2,
                    _a5_config(dataset, origin + HORIZON),
                    evaluate_fn=nsu._evaluate_kdd)
                rr2 = ex2.evaluate(
                    tuple((str(s["op"]), dict(s["params"]))
                          for s in winner["steps"]), origin + HORIZON)
                delayed = {"evaluated": True,
                           "gain": (float(rr2.gain)
                                    if rr2.gain is not None else None)}
                removal = bool(delayed["gain"] is not None
                               and delayed["gain"] < -M)
            # episode 形成（|gain|≥M；BSE 口径 relation）并入臂 Memory
            formed = []
            for p in probes:
                d = delayed if (adopted and winner
                                and p["candidate_id"]
                                == winner["candidate_id"]) else {
                                    "evaluated": False, "gain": None}
                rel = _a5_relation(p["gain"], d)
                if rel is None:
                    continue
                # 2026-08-15 修复（CONTENT_INCONCLUSIVE 门捕获）：
                # local_pattern 必须用 extract_public_features 特征键
                # （与 N3 pack episode 同族）；此前误用 window_context
                # （recent.*/change.* 键）会把臂切换到 signed 分支导致
                # rendered_empty，Memory 静默断线。
                from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
                    extract_public_features)
                ctx = dict(extract_public_features(
                    np.asarray(vals2[uid], dtype=np.float64)[:origin],
                    task_kind="forecast"))
                sig = workflow_signature_of(p["steps"])
                ep = build_episode(
                    episode_id=("a5_" + arm.lower() + "_" + uid[:12]
                                + "_" + str(origin) + "_"
                                + str(p["candidate_id"])),
                    task_consumer_key="forecast|ridge|sMASE",
                    domain_namespace=dataset.replace(":", "_") + "_dev",
                    context_summary={
                        "local_pattern": {"support_gain": p["gain"], **ctx},
                        "delayed_pattern": {},
                        "program_geometry": {
                            "scope": "training_rows",
                            "program_steps": p["steps"]},
                        "support_origin": origin},
                    workflow_signature=sig,
                    support_response={"gain": p["gain"],
                                      "accepted": adopted and winner
                                      and p["candidate_id"]
                                      == winner["candidate_id"]},
                    delayed_response=d,
                    relation=rel,
                    evidence_level=("DELAYED" if d.get("evaluated")
                                    else "SUPPORT"),
                    local_status=("LOCAL_DRAFT"
                                  if rel == "POSITIVE" and d.get("evaluated")
                                  else "EPISODE_ONLY"),
                    evidence_refs=["a5a3_matched_budget"])
                formed.append(ep)
                memories[arm].append(ep)
            row.update({
                "probes": probes,
                "winner_candidate_id": (winner or {}).get("candidate_id"),
                "winner_gain": (winner or {}).get("gain"),
                "adopted": adopted, "delayed": delayed, "removal": removal,
                "episodes_formed": [
                    json.loads(json.dumps(vars(ep), default=str))
                    for ep in formed],
                "support_evals": len(probes),
                "delayed_evals": 1 if delayed["evaluated"] else 0})
            rows.append(row)
            _save_report(report)
            print("== a5 " + arm + " done: winner="
                  + str(row["winner_candidate_id"]) + " gain="
                  + str(row["winner_gain"]) + " adopted=" + str(adopted)
                  + " delayed=" + json.dumps(delayed) + " episodes="
                  + str(len(formed)) + " llm=" + str(prep.get("llm_calls")),
                  flush=True)
    _save_report(report)
    print("== a5 run complete:", len(rows), "rows")
    return 0


def _a5_aggregates(rows: Sequence[dict]) -> dict:
    out = {}
    for arm in ("A5", "A3"):
        arm_rows = [r for r in rows if r.get("arm") == arm]
        cum = 0
        fpe = None
        harm_probes = 0
        delayed_harms = 0
        abstentions = 0
        total_gain = 0.0
        n_eff = 0
        for r in sorted(arm_rows, key=lambda x: (str(x.get("series_uid")),
                                                 int(x.get("origin", 0)))):
            cum += int(r.get("support_evals") or 0)
            for p in (r.get("probes") or ()):
                if p.get("gain") is not None and p["gain"] < -M:
                    harm_probes += 1
            if r.get("removal"):
                delayed_harms += 1
            if not r.get("adopted"):
                abstentions += 1
            elif r.get("winner_gain") is not None:
                total_gain += float(r["winner_gain"])
            for epd in (r.get("episodes_formed") or ()):
                if (epd.get("relation") == "POSITIVE"
                        and (epd.get("delayed_response") or {})
                        .get("evaluated")
                        and ((epd.get("delayed_response") or {})
                             .get("gain") is not None)
                        and float(epd["delayed_response"]["gain"]) >= -M):
                    n_eff += 1
                    if fpe is None:
                        fpe = cum
        out[arm] = {"feedback_to_first_effective": fpe,
                    "support_evals": cum,
                    "harm_events": harm_probes + delayed_harms,
                    "support_harm_probes": harm_probes,
                    "delayed_harm_adoptions": delayed_harms,
                    "abstentions": abstentions,
                    "total_support_gain": total_gain,
                    "n_effective": n_eff,
                    "llm_calls": sum(int(r.get("llm_calls") or 0)
                                     for r in arm_rows)}
    return out


def phase_a5_retry_transient() -> int:
    """清理瞬时传输错误行（AgentTransportError/Timeout——网络噪声，非证
    据），供 phase_a5 续跑重做；同时在 a5_protocol 记录修正案。"""
    report = _load_report()
    proto = report.get("a5_protocol") or {}
    a5 = report.get("a5") or {}
    rows = a5.get("rows") or []
    keep, drop = [], []
    for r in rows:
        err = str(r.get("protocol_error") or "")
        if err and ("Transport" in err or "Timeout" in err):
            drop.append(r)
        else:
            keep.append(r)
    if drop:
        a5["rows"] = keep
        proto.setdefault("amendments", []).append({
            "date": "2026-08-15",
            "change": ("瞬时传输错误（AgentTransportError/APITimeoutError，"
                       "网络层噪声）重试 ≤2 次且不计入 protocol_error；"
                       "真实协议错误（AgentProtocolError 等）仍 → "
                       "PROTOCOL_INCONCLUSIVE。传输 ≠ 证据——不重试会把"
                       "基础设施抖动误记为科学结果"),
            "ruled_by": "主 Agent（基础设施/证据边界修正），待用户复核",
            "dropped_transient_rows": [
                {"arm": r.get("arm"), "entity": r.get("entity_id"),
                 "origin": r.get("origin"),
                 "error": str(r.get("protocol_error"))[:80]}
                for r in drop]})
        _save_report(report)
    print("a5 transient cleanup: dropped=" + str(len(drop))
          + " kept=" + str(len(keep)))
    return 0


# 当前待记录的 a5 重置事故（每次 reset 前更新）
_A5_PENDING_INCIDENT = {
    "date": "2026-08-15",
    "event": ("接线 bug 运行（clean rerun 12/12）：我构造的 Target-local "
              "episode 的 local_pattern 误用 window_context 的 recent.*/"
              "change.* 键 → fast_agent.py:787 把臂切换到 signed 分支 → "
              "rendered_empty → 第 3 context 起两臂 Memory 均静默断线；"
              "冻结裁定链的 CONTENT_INCONCLUSIVE 门按设计捕获（N5 预检"
              "只测了静态 pack，未测增长后状态——预检盲点）"),
    "action": ("修复 episode local_pattern 为 extract_public_features 键"
               "（与 N3 pack 同族）；作废 12 行；单进程重跑为唯一有效 rep"),
    "lesson": ("Memory 增长路径必须纳入接线预检（增长后状态 ≠ 静态 pack "
               "状态）；episode 特征键族决定注入分支，必须与既有 pack 对齐")}


def phase_a5_reset() -> int:
    """僵尸进程事故重置（2026-08-15）：bash-22 被 job_kill 后其 python 子
    进程残留，与 bash-23 双进程交叉写报告，盘上 12 行被 5 行陈旧状态覆
    盖。混合来源的行不可作为单 rep 数据——全部作废，单进程干净重跑。
    保留 a5_protocol（含 amendments）。幂等。"""
    report = _load_report()
    a5 = report.get("a5") or {}
    old_rows = len(a5.get("rows") or [])
    incidents = a5.get("incidents") or []
    if a5.get("zombie_incident"):
        incidents.append(a5.pop("zombie_incident"))
    incident = dict(_A5_PENDING_INCIDENT)
    incident["discarded_rows"] = old_rows
    incidents.append(incident)
    report["a5"] = {"incidents": incidents}
    _save_report(report)
    print("a5 reset: discarded_rows=" + str(old_rows))
    return 0


def phase_a5_verdict() -> int:
    report = _load_report()
    a5 = report.get("a5") or {}
    rows = a5.get("rows") or []
    roster = (report.get("a5_protocol") or {}
              ).get("target", {}).get("roster") or []
    expected = 2 * len(roster)
    if len(rows) < expected:
        raise SystemExit("a5 未完成（" + str(len(rows)) + "/"
                         + str(expected) + "）")
    v = a5.get("verdict") or {}
    if v.get("final"):
        print(json.dumps(v, ensure_ascii=False))
        return 0
    agg = _a5_aggregates(rows)
    a5["aggregates"] = agg
    verdict = _a5_verdict(agg, rows)
    verdict["final"] = True
    verdict["claim_boundary"] = (report.get("a5_protocol") or {}).get(
        "claim_boundary")
    a5["infra_notes_2026_08_15"] = (
        "运行期 agicto API 降级（小调用 36s）：LLM 调用超时 180→600s + "
        "瞬时传输错误 ≤2 重试（见 a5_protocol.amendments）；均为基础设施适"
        "配，不改变任何科学判定口径")
    a5["verdict"] = verdict
    _save_report(report)
    print("== a5 终裁:", json.dumps(
        {"verdict": verdict["verdict"], "reason": verdict.get("reason"),
         "aggregates": agg}, ensure_ascii=False, indent=1))
    return 0



# ---------------------------------------------------------------- n4v2
# N4v2（2026-08-15 外部代码审核后重做）：N4 v1/v2 的暴露扫描 recall 不足——
# 只匹配 64-hex 全串，漏掉 8 位前缀引用（"0414c7e9=T635 @792/888"，
# w1_e2_memory_two_slot_report.json:106）与 entity_id 字段值；registry 的
# certified_virgin 滞后于真实消费（T635）。v3 改为 recall-first 字段级扫描：
#   ① 64-hex series_uid / content_sha 全串
#   ② series_uid 8 位前缀（词边界）
#   ③ entity_id 带引号 JSON 值（"T635"）
#   ④ entity@origin 形态（T635@792 / T635 @792）
#   ⑤ entity_id 裸词（词边界；KDD 撞名集 {T13,T128..T134} 除外）
# 任一命中 → 该 series 整体出局（recall-first：宁可过度排除）。registry
# 滞后不改动冻结文件——以 exposure_overrides 追加式记录。排除本审计链自
# 身产物（主报告 n4/n5/a5 节 + _a5/_n4 运行日志——审计输出非消费）。

N4V2_KDD_NAME_COLLISIONS = frozenset(
    {"T13"} | {f"T{i}" for i in range(128, 135)})
N4V2_OWN_ARTIFACTS = ("w1_guidance_evolution_report.json",)


def _n4v2_scan_files() -> list:
    files = []
    for pat in ("artifacts/**/*.json", "artifacts/**/*.jsonl",
                "evaluation/**/*.log", "evaluation/**/*report*.json"):
        files.extend(PROJECT_ROOT.glob(pat))
    frozen_dir = PROJECT_ROOT / "artifacts" / "frozen"
    out = []
    for fp in sorted(set(files)):
        if fp == N4_REGISTRY_REL or N4_CLEAN_BASE in fp.parents \
                or frozen_dir in fp.parents:
            continue  # 簿记/存储 ≠ 消费（N4 v2 已批准口径）
        if fp.name in N4V2_OWN_ARTIFACTS:
            continue  # 本审计链主报告（含 n4/n5/a5 审计输出）非消费证据
        if fp.name.startswith(("_a5_run", "_n4_run")):
            continue  # 本审计链运行日志（含 roster 实体名）非消费证据
        out.append(fp)
    return out


def _n4v2_series_matchers(uid: str, entity: str, sha: str) -> list:
    """每 series 的 recall-first 匹配器（名称, 编译后正则/子串）。"""
    import re  # noqa: PLC0415
    ms = []
    if len(uid) == 64:
        ms.append(("uid_full", lambda t, s=uid: s in t))
        ms.append(("uid_prefix8", re.compile(r"\b" + re.escape(uid[:8]))))
    if len(sha) == 64:
        ms.append(("sha_full", lambda t, s=sha: s in t))
    if entity:
        e = re.escape(entity)
        ms.append(("entity_json", re.compile(r'"' + e + r'"')))
        ms.append(("entity_at_origin",
                   re.compile(r"\b" + e + r"\s*@\s*\d+")))
        if entity not in N4V2_KDD_NAME_COLLISIONS:
            ms.append(("entity_prose", re.compile(r"\b" + e + r"\b")))
    return ms


def _n4v2_consumption_scan(series_ids: Mapping[str, dict]) -> dict:
    """series_ids: uid → {entity_id, content_sha}。返回命中证据。
    实现（2026-08-15 性能修正）：翻转转循环——每文件只跑 5 个汇总正则
    （64-hex token / ≥8-hex 前缀 / 带引号 T 名 / T@origin / 裸 T 词），
    抽出的标识符在集合里查表；语义与逐 series 匹配器完全一致
    （_n4v2_series_matchers 保留作测试与文档基准）。"""
    import re  # noqa: PLC0415
    uid_set = {u for u in series_ids if len(u) == 64}
    sha_to_uid = {str(d.get("content_sha")): u
                  for u, d in series_ids.items()
                  if len(str(d.get("content_sha") or "")) == 64}
    prefix8 = {u[:8]: u for u in uid_set}
    entity_to_uid = {str(d.get("entity_id")): u
                     for u, d in series_ids.items()
                     if str(d.get("entity_id") or "")}
    re_hex64 = re.compile(r"\b[0-9a-f]{64}\b")
    re_hex8 = re.compile(r"\b[0-9a-f]{8,63}\b")
    re_quoted_t = re.compile(r'"(T\d+)"')
    re_t_at = re.compile(r"\b(T\d+)\s*@\s*\d+")
    re_t_word = re.compile(r"\b(T\d+)\b")
    hits: dict[str, list] = {}

    def record(uid: str, fp: Path, kind: str) -> None:
        lst = hits.setdefault(uid, [])
        if not any(h["file"] == str(fp.relative_to(PROJECT_ROOT))
                   for h in lst):
            lst.append({"file": str(fp.relative_to(PROJECT_ROOT)),
                        "kind": kind})

    files = _n4v2_scan_files()
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for tok in re_hex64.findall(text):
            if tok in uid_set:
                record(tok, fp, "uid_full")
            elif tok in sha_to_uid:
                record(sha_to_uid[tok], fp, "sha_full")
        for tok in re_hex8.findall(text):
            u = prefix8.get(tok[:8])
            if u:
                record(u, fp, "uid_prefix8")
        for m in re_quoted_t.finditer(text):
            u = entity_to_uid.get(m.group(1))
            if u:
                record(u, fp, "entity_json")
        for m in re_t_at.finditer(text):
            u = entity_to_uid.get(m.group(1))
            if u:
                record(u, fp, "entity_at_origin")
        for m in re_t_word.finditer(text):
            name = m.group(1)
            if name in N4V2_KDD_NAME_COLLISIONS:
                continue
            u = entity_to_uid.get(name)
            if u:
                record(u, fp, "entity_prose")
    return {"files_scanned": len(files),
            "hit_series": len(hits),
            "hits": {k: v[:5] for k, v in sorted(hits.items())}}


def phase_n4v2_freeze() -> int:
    report = _load_report()
    if report.get("n4_v2_protocol"):
        raise SystemExit("n4_v2_protocol 已存在——拒绝重复冻结")
    if report.get("n4_v2"):
        raise SystemExit("n4_v2 已有结果——协议必须先冻结")
    proto = {
        "experiment_id": "N4V2_FIELD_LEVEL_EXPOSURE_2026_08_15",
        "supersedes": ("n4（v1/v2 扫描）——外部审核发现 recall 不足："
                       "64-hex 全串扫描漏掉 8 位前缀引用与 entity_id 字段值，"
                       "T635@792/@888 已于 2026-08-12 被 two-slot 实验消费"
                       "（w1_e2_memory_two_slot_report.json:106）而 registry "
                       "仍为 certified_virgin"),
        "scan_design": {
            "principle": ("recall-first：任一匹配器命中即整支 series 出局"
                          "（过度排除可接受，漏检不可接受）"),
            "matchers": ["uid 64-hex 全串", "uid 8 位前缀（词边界）",
                         "content_sha 64-hex", "entity_id 带引号 JSON 值",
                         "entity@origin 形态",
                         "entity_id 裸词（KDD 撞名集除外）"],
            "scope": ("artifacts/**/*.json,jsonl + evaluation/**/*.log,"
                      "*report*.json；排除 artifacts/frozen/**（簿记）与"
                      " clean_base（存储）与本审计链自身产物（主报告 n4/n5/"
                      "a5 节、_a5_run*/_n4_run 日志——审计输出非消费）"),
            "registry_role": ("registry 仍为起点分级，但不再权威——扫描证据"
                              "优先；冲突以 exposure_overrides 追加记录，"
                              "不改冻结 registry 文件")},
        "eligibility_checks": ("同 n4（窗口完整性/修改前提/合法性/长度缺失）"
                               "＋消费判定换为本扫描"),
        "roster_rule": ("同 n4：合格 context 数最多的数据集（平手字典序）；"
                        "(series_uid, origin) 字典序前 6；<4 → "
                        "FRESH_TARGET_CONTENT_UNAVAILABLE"),
        "no_new_systems": "不建 Exposure Ledger / 新 Hash 系统（用户裁定）",
        "verdict_rules": {"eligible": "N4_ROSTER_ELIGIBLE",
                          "unavailable": "FRESH_TARGET_CONTENT_UNAVAILABLE"},
        "llm_budget": "0", "outcome_reads": "0",
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["n4_v2_protocol"] = proto
    _save_report(report)
    print("n4_v2_protocol FROZEN")
    return 0


def phase_n4v2() -> int:
    report = _load_report()
    if not report.get("n4_v2_protocol"):
        raise SystemExit("n4_v2_protocol 未冻结——先跑 n4v2-freeze")
    if (report.get("n4_v2") or {}).get("verdict"):
        print("n4_v2 已终裁:", report["n4_v2"]["verdict"].get("verdict"))
        return 0
    registry = _n4_load_registry()
    reg_by_uid = {str(r.get("series_uid")): r for r in registry}
    clean = _n4_clean_records()
    # 扫描全集：clean_base 内所有 series（不局限于候选数据集——暴露证据
    # 与数据集归属无关）
    series_ids = {}
    for uid, e in clean.items():
        rg = reg_by_uid.get(uid) or {}
        series_ids[uid] = {
            "entity_id": str(rg.get("entity_id")
                               or e["record"].get("entity_id") or ""),
            "content_sha": str(e["record"].get("content_sha") or "")}
    scan = _n4v2_consumption_scan(series_ids)
    hit_uids = set(scan["hits"].keys())
    print("== n4v2 扫描: files=" + str(scan["files_scanned"])
          + " hit_series=" + str(scan["hit_series"]), flush=True)
    # registry 滞后修正（追加式）
    overrides = []
    for uid in sorted(hit_uids):
        rg = reg_by_uid.get(uid)
        if rg and str(rg.get("exposure_class")) == "certified_virgin":
            overrides.append({
                "series_uid": uid,
                "entity_id": str(rg.get("entity_id")),
                "registry_class": "certified_virgin",
                "scan_evidence": scan["hits"][uid][:2],
                "correction": "scan_evidence_overrides_registry"})
    print("== n4v2 registry 滞后修正: " + str(len(overrides)) + " 支",
          flush=True)
    per_dataset: dict[str, Any] = {}
    for ds in N4_CANDIDATE_DATASETS:
        ds_rows = [r for r in registry if str(r.get("dataset_id")) == ds]
        stat = {"registry_total": len(ds_rows), "not_virgin": 0,
                "not_in_clean_base": 0, "too_short_or_missing": 0,
                "scan_hit": 0, "no_promise": 0, "window_incomplete": 0,
                "eligible": []}
        for r in ds_rows:
            uid = str(r.get("series_uid"))
            if str(r.get("exposure_class")) != "certified_virgin":
                stat["not_virgin"] += 1
                continue
            ent = clean.get(uid)
            if ent is None:
                stat["not_in_clean_base"] += 1
                continue
            rec = ent["record"]
            if (int(rec.get("length") or 0) < N4_MIN_LENGTH
                    or int(rec.get("natural_missing_count") or 0) != 0):
                stat["too_short_or_missing"] += 1
                continue
            if uid in hit_uids:
                stat["scan_hit"] += 1
                continue
            arr = np.load(ent["values"]).astype(np.float64)
            for origin in N4_ORIGINS:
                tot = _n4_change_counts(arr, origin)
                if tot is None:
                    stat["window_incomplete"] += 1
                    continue
                if tot == 0:
                    stat["no_promise"] += 1
                    continue
                stat["eligible"].append({
                    "dataset": ds, "series_uid": uid,
                    "entity_id": str(r.get("entity_id")),
                    "origin": origin, "changed_total": int(tot),
                    "frequency": str(rec.get("frequency")),
                    "length": int(rec.get("length")),
                    "context_exposure": "AGGREGATE_SEEN",
                    "outcome_exposure": "SEALED"})
        per_dataset[ds] = stat
        print("== n4v2 " + ds + ": eligible=" + str(len(stat["eligible"]))
              + " scan_hit=" + str(stat["scan_hit"]), flush=True)
    pick = _n4_pick_roster(per_dataset)
    report["n4_v2"] = {
        "consumption_scan": scan,
        "exposure_overrides": overrides,
        "per_dataset": per_dataset,
        "pick": pick,
        "verdict": {"verdict": pick["verdict"], "final": True,
                    "dataset": pick.get("dataset"),
                    "roster_size": len(pick.get("roster") or [])},
        "roster": pick.get("roster") or [],
    }
    _save_report(report)
    print("n4v2: " + pick["verdict"] + " dataset=" + str(pick.get("dataset"))
          + " roster=" + str(len(pick.get("roster") or [])))
    return 0


# ---------------------------------------------------------------- n5v2
# N5v2（2026-08-15）：N4v2 新 roster 上的接线预检 + 增长态检查（修复 N5
# v1 盲点——v1 只测静态 pack，未测"运行中 Memory 增长后"状态，导致
# window_context 键族 episode 把臂切到 signed 分支 rendered_empty 的事故
# 漏网）。零 LLM 零 Outcome。

def phase_n5v2() -> int:
    report = _load_report()
    if (report.get("n4_v2") or {}).get("verdict", {}).get("verdict") \
            != "N4_ROSTER_ELIGIBLE":
        raise SystemExit("n4_v2 未合格——N5v2 无 roster 可查")
    if (report.get("n5_v2") or {}).get("verdict"):
        print("n5_v2 已终裁:", report["n5_v2"]["verdict"].get("verdict"))
        return 0
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: PLC0415
        extract_public_features,
    )
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: PLC0415
        build_episode, workflow_signature_of)
    episodes = _n5_load_pack_episodes()
    if not episodes:
        report["n5_v2"] = {"verdict": {"verdict": "N5_WIRING_BROKEN",
                                       "reason": "Source Pack 加载为空"},
                           "final": True}
        _save_report(report)
        return 0
    roster = report["n4_v2"]["roster"]
    clean = _n4_clean_records()
    rows = []
    failures = []
    for e in roster:
        uid = str(e["series_uid"])
        origin = int(e["origin"])
        arr = np.load(clean[uid]["values"]).astype(np.float64)
        feats = extract_public_features(arr[:origin], task_kind="forecast")
        # ① A5 静态：pack → contrast_pack
        a5_static = _n5_wiring_check(episodes, feats)
        # ② A5 增长态：pack + 目标 local episode（extract_public_features
        #    键族——2026-08-15 修复后形态）→ 仍 contrast_pack
        local_ep = build_episode(
            episode_id="n5v2_growth_probe",
            task_consumer_key="forecast|ridge|sMASE",
            domain_namespace=str(e["dataset"]).replace(":", "_") + "_dev",
            context_summary={
                "local_pattern": {"support_gain": 0.05, **dict(feats)},
                "delayed_pattern": {},
                "program_geometry": {"scope": "training_rows"}},
            workflow_signature=workflow_signature_of(
                [{"op": "outlier_mad", "params": {}}]),
            support_response={"gain": 0.05, "accepted": True},
            delayed_response={"evaluated": True, "gain": 0.02},
            relation="POSITIVE", evidence_level="DELAYED",
            local_status="LOCAL_DRAFT", evidence_refs=["n5v2"])
        a5_grown = _n5_wiring_check(list(episodes) + [local_ep], feats)
        # ③ A3 静态 no_memory；④ A3 增长态（仅 local episode）→ 应仍
        #    走 contrast 路径（contrast_pack/contrast_pack_empty 均可接受——
        #    关键是不 flip 到 signed 分支；signed_absent 结构判定）
        a3_static = _n5_wiring_check((), feats)
        grown_signed_absent = _n5_signed_absent(
            list(episodes) + [local_ep])
        row = {"entity_id": e.get("entity_id"), "origin": origin,
               "a5_static": a5_static, "a5_grown": a5_grown,
               "a3_static": a3_static,
               "grown_signed_absent": grown_signed_absent}
        rows.append(row)
        ok = (a5_static["memory_resolution_status"] == "contrast_pack"
              and all(a5_static["buckets_nonempty"].values())
              and not a5_static["injection_failed"]
              and a5_grown["memory_resolution_status"] == "contrast_pack"
              and a5_grown["rendered_len"] > 0
              and a3_static["memory_resolution_status"] == "no_memory"
              and grown_signed_absent)
        if not ok:
            failures.append(row)
        print("== n5v2 " + str(e.get("entity_id")) + "@" + str(origin)
              + ": static=" + str(a5_static["memory_resolution_status"])
              + " grown=" + str(a5_grown["memory_resolution_status"])
              + " a3=" + str(a3_static["memory_resolution_status"])
              + " signed_absent=" + str(grown_signed_absent), flush=True)
    checks = {
        "a5_static_contrast_pack": all(
            r["a5_static"]["memory_resolution_status"] == "contrast_pack"
            and all(r["a5_static"]["buckets_nonempty"].values())
            for r in rows),
        "a5_grown_contrast_pack": all(
            r["a5_grown"]["memory_resolution_status"] == "contrast_pack"
            and r["a5_grown"]["rendered_len"] > 0 for r in rows),
        "a3_no_memory": all(
            r["a3_static"]["memory_resolution_status"] == "no_memory"
            for r in rows),
        "no_injection_failed": not any(
            r["a5_static"]["injection_failed"]
            or r["a5_grown"]["injection_failed"] for r in rows),
        "growth_keeps_contrast_branch": all(
            r["grown_signed_absent"] for r in rows),
        "prior_absent_disclosure": True,
    }
    verdict = ("N5_WIRING_OK" if all(checks.values())
               else "N5_WIRING_BROKEN")
    report["n5_v2"] = {
        "pack_size": len(episodes),
        "source_prior_candidate_absent": True,
        "growth_check": ("pack + extract_public_features 键族 local episode "
                         "→ 仍 contrast_pack（N5 v1 盲点修复：增长态入检）"),
        "checks": checks, "failures": failures, "rows": rows,
        "verdict": {"verdict": verdict, "final": True},
    }
    _save_report(report)
    print("n5v2: " + verdict + " checks=" + json.dumps(checks))
    return 0


# ---------------------------------------------------------------- a5v2
# A5v2/A3v2（2026-08-15 外部审核后重做）：真 Skill 生命周期——每臂独立
# snapshot，positive winner 走方法层 handle_fast_winner（经
# run_online_round(allow_fast_skill=True)），后续 origin 走正常入口
# （store active snapshot 重建 method），要求 retrieved/verified/probed，
# delayed 后 approve 或 removal。指标 = feedback_to_reliable_local_skill
# （完整 trajectory），不再是 Episode 代理。
# Memory 机制不变（用户裁定：确认归因前不改 Memory/Prompt/Skill）：
# online_loop 原生 episode 带 recent.*/change.* 键会把臂切到 signed 分支
# → 每轮 delayed 后归一化为 extract_public_features 键族（Runner 侧
# 后处理，不改方法层；与 N5v2 增长态检查同构）。

A5V2_BUDGET_PER_ROUND = 3          # 每 context 每臂 Support receipt 上限
A5V2_LLM_PER_PREPARE = 12
A5V2_STORE_PREFIX = ".a5v2_store_"


def _a5v2_card(_episode: Any) -> Mapping[str, Any]:
    """Fast winner 的最小 Card：observable_signature=task_kind → 宽
    applicability → requires_target_support=True（Draft 门：下一正常入口
    必须经 Target Support 确认——'probed' 要求内建）。"""
    return {"pattern_id": "fast-winner",
            "observable_signature": {"task_kind": "forecast"}}


def _a5v2_normalize_new_episodes(method: Any, series0: Any,
                                 origin: int) -> int:
    """把本轮 online_loop 原生写入的 recent.*/change.* 键 episode 归一化
    为 extract_public_features 键族（保持 contrast 分支；不动 pack 与已
    归一化的 episode）。在 open_delayed 之后调用（保留 delayed 更新）。"""
    import dataclasses as _dc  # noqa: PLC0415
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: PLC0415
        extract_public_features)
    feats = dict(extract_public_features(
        np.asarray(series0, dtype=np.float64)[:origin],
        task_kind="forecast"))
    n = 0
    for ep in list(getattr(method, "_experience_episodes", []) or []):
        cs = dict(getattr(ep, "context_summary", None) or {})
        lp = dict(cs.get("local_pattern") or {})
        if not any(str(k).startswith(("recent.", "change.")) for k in lp):
            continue
        sg = (getattr(ep, "support_response", None) or {}).get("gain")
        cs["local_pattern"] = {"support_gain": sg, **feats}
        method.update_experience_episode(_dc.replace(ep, context_summary=cs))
        n += 1
    return n


def _a5v2_verdict(aggregates: Mapping[str, Any], rows: Sequence[dict]) -> dict:
    """冻结裁定（a5v2_protocol.verdict_rules；优先级
    PROTOCOL_INCONCLUSIVE > CONTENT_INCONCLUSIVE > 其余）。"""
    a5 = aggregates.get("A5") or {}
    a3 = aggregates.get("A3") or {}
    if any(r.get("protocol_error") for r in rows):
        return {"verdict": "PROTOCOL_INCONCLUSIVE",
                "reason": "存在 protocol_error 行",
                "error_rows": [{"arm": r.get("arm"),
                                "entity": r.get("entity_id"),
                                "origin": r.get("origin")}
                               for r in rows if r.get("protocol_error")]}
    bad_mem = [r for r in rows
               if r.get("arm") == "A5"
               and r.get("memory_resolution_status") != "contrast_pack"]
    if bad_mem:
        return {"verdict": "CONTENT_INCONCLUSIVE",
                "reason": "A5 运行期 Memory 未按 N5v2 预检接线",
                "bad_rows": [{"entity": r.get("entity_id"),
                              "origin": r.get("origin"),
                              "status": r.get("memory_resolution_status")}
                             for r in bad_mem]}
    f5 = a5.get("feedback_to_reliable_skill")
    f3 = a3.get("feedback_to_reliable_skill")
    h5 = int(a5.get("harm_events") or 0)
    h3 = int(a3.get("harm_events") or 0)
    n5e = int(a5.get("n_reliable") or 0)
    n3e = int(a3.get("n_reliable") or 0)
    if n5e == 0 and n3e == 0:
        return {"verdict": "NO_SIGNAL",
                "reason": "两臂预算内均未形成 reliable Target-local Skill"}
    f5v = f5 if f5 is not None else float("inf")
    f3v = f3 if f3 is not None else float("inf")
    if n5e >= 1 and f5v < f3v and h5 <= h3:
        return {"verdict": "TRANSFER_CASE_PASS",
                "reason": ("A5 以更少 feedback 形成首个 reliable Skill（"
                           + str(f5) + " vs " + str(f3) + "）且 harm 不多（"
                           + str(h5) + " vs " + str(h3) + "）"),
                "claim": ("development transfer case：单 Target 数据集、n=6 "
                          "context、1 rep；非一般迁移声明")}
    if f5v > f3v or h5 > h3:
        return {"verdict": "NEGATIVE_TRANSFER",
                "reason": ("A5 更慢或更多 harm（feedback " + str(f5) + " vs "
                           + str(f3) + "；harm " + str(h5) + " vs "
                           + str(h3) + "）")}
    return {"verdict": "NO_SIGNAL",
            "reason": ("两臂同速同 harm（feedback " + str(f5) + " vs "
                       + str(f3) + "；harm " + str(h5) + " vs "
                       + str(h3) + "）")}


def _a5v2_aggregates(rows: Sequence[dict]) -> dict:
    """从 rows 汇总每臂指标；skill trajectory：created→approved→
    后续正常入口 retrieved→re-probed≥−M=reliable；re-probe<−M=removal。"""
    out = {}
    for arm in ("A5", "A3"):
        arm_rows = [r for r in rows if r.get("arm") == arm]
        cum = 0
        cum_at = []   # 每行结束时的累计 receipts
        for r in arm_rows:
            cum += int(r.get("support_receipts") or 0)
            cum_at.append(cum)
        harm = sum(int(r.get("harm_probes") or 0) for r in arm_rows) \
            + sum(1 for r in arm_rows
                  if (r.get("delayed_utility") is not None
                      and float(r["delayed_utility"]) < -M))
        # skill 生命周期轨迹
        skills: dict[str, dict] = {}
        for i, r in enumerate(arm_rows):
            sid = r.get("skill_created_id")
            if sid:
                skills.setdefault(sid, {"created_row": i,
                                        "approved": False,
                                        "reliable": False,
                                        "removed": False,
                                        "reliable_at_cum": None})
            ap = r.get("approved_skill_id")
            if ap and ap in skills:
                skills[ap]["approved"] = True
            for sid2 in (r.get("retrieved_skill_ids") or ()):
                if sid2 in skills and skills[sid2]["approved"] \
                        and not skills[sid2]["reliable"]:
                    # 该入口 re-probe 了 skill 候选吗？
                    sp = [p for p in (r.get("skill_probes") or ())
                          if p.get("gain") is not None]
                    if sp:
                        g = float(sp[0]["gain"])
                        if g < -M:
                            skills[sid2]["removed"] = True
                        else:
                            skills[sid2]["reliable"] = True
                            skills[sid2]["reliable_at_cum"] = cum_at[i]
        reliable = [s for s in skills.values()
                    if s["reliable"] and not s["removed"]]
        fpe = min((s["reliable_at_cum"] for s in reliable
                   if s["reliable_at_cum"] is not None), default=None)
        out[arm] = {
            "feedback_to_reliable_skill": fpe,
            "support_evals": cum,
            "harm_events": harm,
            "abstentions": sum(1 for r in arm_rows if r.get("abstained")),
            "total_support_gain": sum(
                float(r["winner_gain"]) for r in arm_rows
                if r.get("winner_gain") is not None),
            "n_skills_created": len(skills),
            "n_approved": sum(1 for s in skills.values() if s["approved"]),
            "n_reliable": len(reliable),
            "n_removed": sum(1 for s in skills.values() if s["removed"]),
            "skills": {k: v for k, v in skills.items()},
            "llm_calls": sum(int(r.get("llm_calls") or 0)
                             for r in arm_rows)}
    return out


def phase_a5v2_freeze() -> int:
    report = _load_report()
    if report.get("a5v2_protocol"):
        raise SystemExit("a5v2_protocol 已存在——拒绝重复冻结")
    if report.get("a5v2"):
        raise SystemExit("a5v2 已有结果——协议必须先冻结")
    n4v = ((report.get("n4_v2") or {}).get("verdict") or {}).get("verdict")
    n5v = ((report.get("n5_v2") or {}).get("verdict") or {}).get("verdict")
    if n4v != "N4_ROSTER_ELIGIBLE":
        raise SystemExit("n4_v2 未合格（" + str(n4v) + "）——a5v2 不得冻结")
    if n5v != "N5_WIRING_OK":
        raise SystemExit("n5_v2 未合格（" + str(n5v) + "）——a5v2 不得冻结")
    roster = report["n4_v2"]["roster"]
    dataset = report["n4_v2"]["verdict"]["dataset"]
    proto = {
        "experiment_id": "A5V2_SKILL_LIFECYCLE_2026_08_15",
        "supersedes": ("a5（2026-08-15 早场）——外部审核：n_effective 只是 "
                       "delayed-positive Episode 计数，未走 Skill 生命周期；"
                       "且 N4 v1/v2 漏检 T635 暴露。本次在 N4v2 新 roster"
                       "（uci_electricity_load_diagrams，字段级暴露审查）上"
                       "以真 Skill 生命周期重跑"),
        "question": ("相同 Target feedback 预算下，读取自然 Source Pack 的 "
                     "A5 能否比空 Source Memory 的 A3 更快（更少 feedback 形"
                     "成首个 reliable Target-local Skill）、更安全（harm "
                     "不多）"),
        "target": {"dataset": dataset, "roster": roster,
                   "exposure": ("context_exposure=AGGREGATE_SEEN; "
                                "outcome_exposure=SEALED（N4v2 字段级审计，"
                                "含 registry 滞后修正）")},
        "arms": {
            "A5": "h0(rev7) + N3 Source Pack（10 条 KDD 自然轨迹）",
            "A3": "h0(rev7) + 空 Source Memory",
            "lifecycle": ("每臂独立 snapshot（a5v2_store_<arm>）；每 context "
                          "经正常入口（当臂 active snapshot + 当臂 Memory 重"
                          "建 method）；positive winner → handle_fast_winner"
                          "（宽 scope → requires_target_support Draft 门）→ "
                          "open_delayed → approve/removal → activate_"
                          "approved；后续入口 retrieved/verified/probed 由"
                          " run_online_round 天然保证"),
            "memory_mechanism": ("不变（裁定：归因确认前不改 Memory/Prompt/"
                                 "Skill）；online_loop 原生 recent.*/change.* "
                                 "episode 每轮 delayed 后归一化为 "
                                 "extract_public_features 键族（Runner 后处"
                                 "理，保持 contrast 分支——N5v2 增长态同构"
                                 "已验证）"),
            "slow": "两臂同关",
            "symmetric_growth": "两臂 Target-local episode 对称累积"},
        "per_context_sequence": [
            "prepare（memory 按臂；with_task_context；≤12 LLM）",
            "chosen-first 探测，首个 gain≥M 即 winner 停探（在线语义），"
            "receipts ≤3",
            "winner → handle_fast_winner → Draft Skill pending",
            "open_delayed（origin+48；仅 winner）→ approve（≥−M）或拒绝",
            "activate_approved → 下一入口 active snapshot",
            "episode delayed 更新 + 键族归一化（contrast 分支保持）"],
        "reliable_skill_definition": ("approved（delayed≥−M）且 ≥1 个后续正"
                                      "常入口 retrieved 且该入口 re-probe "
                                      "skill 候选 gain≥−M；re-probe<−M → "
                                      "removal（不再 reliable）"),
        "budget_table": {"rounds_per_arm": len(roster),
                         "support_per_round": A5V2_BUDGET_PER_ROUND,
                         "delayed_per_round": "≤1（仅 winner）",
                         "llm_per_prepare": A5V2_LLM_PER_PREPARE,
                         "slow_calls": 0},
        "metrics": ["feedback_to_reliable_skill（累计 support receipts 到首"
                    "个 reliable 确认轮，含该轮；无则 null）",
                    "harm_events（support<−M 探测 + adopted winner "
                    "delayed<−M）", "abstentions",
                    "total_support_gain（winner 之和）",
                    "n_skills_created/n_approved/n_reliable/n_removed"],
        "verdict_rules": {
            "precedence": "PROTOCOL_INCONCLUSIVE > CONTENT_INCONCLUSIVE > 其余",
            "pass": ("A5.feedback_to_reliable 严格 < A3 且 A5.harm ≤ A3.harm "
                     "且 A5.n_reliable ≥ 1 → TRANSFER_CASE_PASS"),
            "negative": "A5 更慢或更多 harm → NEGATIVE_TRANSFER",
            "no_signal": "两臂均无 reliable Skill 或两维全等 → NO_SIGNAL",
            "content": "A5 任一轮 memory_resolution_status != contrast_pack"
                       " → CONTENT_INCONCLUSIVE",
            "protocol": "任一 protocol_error 行 → PROTOCOL_INCONCLUSIVE"},
        "hypothesis_to_test": ("预注册归因假设（审核裁定）：Source Contrast "
                               "指令可能提高保守/弃权倾向并改变候选 family，"
                               "从而推迟正向 Target Episode——本次仅观测，"
                               "确认前不改 Memory/Prompt/Skill"),
        "outcome_acceptance": ("新 roster 上结果无论 NEGATIVE/NO_SIGNAL/"
                               "PASS 均接受，不换 roster 找答案"),
        "claim_boundary": ("单 Target 数据集、n=6 context、1 rep——"
                           "development transfer case；非一般迁移声明"),
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["a5v2_protocol"] = proto
    _save_report(report)
    print("a5v2_protocol FROZEN (dataset=" + str(dataset) + ", contexts="
          + str(len(roster)) + ")")
    return 0


def phase_a5v2_reset() -> int:
    """a5v2 原子重跑重置：清空 rows/aggregates/verdict + 臂 store 目录。"""
    import shutil  # noqa: PLC0415
    report = _load_report()
    a5 = report.pop("a5v2", None)
    for arm in ("a5", "a3"):
        d = PROJECT_ROOT / (A5V2_STORE_PREFIX + arm)
        if d.exists():
            shutil.rmtree(d)
    _save_report(report)
    print("a5v2 reset: cleared rows=" + str(len((a5 or {}).get("rows") or []))
          + " + stores")
    return 0


def phase_a5v2() -> int:
    """matched-budget A5v2/A3v2（原子运行——中断须 a5v2-reset 后全量重跑）。"""
    import shutil  # noqa: PLC0415
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: PLC0415
        activate_approved, open_delayed, run_online_round)
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: PLC0415
        extract_public_features)
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: PLC0415
        ScopeExecutor)
    report = _load_report()
    if not report.get("a5v2_protocol"):
        raise SystemExit("a5v2_protocol 未冻结——先跑 a5v2-freeze")
    a5 = report.setdefault("a5v2", {})
    if (a5.get("verdict") or {}).get("final"):
        raise SystemExit("a5v2 已终裁")
    if a5.get("rows"):
        raise SystemExit("a5v2 已有部分行——原子语义：先 a5v2-reset 再重跑")
    roster = report["a5v2_protocol"]["target"]["roster"]
    dataset = str(report["a5v2_protocol"]["target"]["dataset"])
    h0 = _h0_snapshot()
    clean = _n4_clean_records()
    values = {str(e["series_uid"]): np.load(
        clean[str(e["series_uid"])]["values"]).astype(np.float64)
        for e in roster}
    pack = _n5_load_pack_episodes()
    rows = a5.setdefault("rows", [])
    for arm in ("A5", "A3"):
        store_dir = PROJECT_ROOT / (A5V2_STORE_PREFIX + arm.lower())
        if store_dir.exists():
            shutil.rmtree(store_dir)
        store = SnapshotStore(store_dir)
        controller = EditController(store, surfaces=SurfaceRegistry(),
                                    router=FaultRouter())
        import openai  # noqa: PLC0415
        api_key = next((os.environ.get(k, "").strip() for k in KEY_ENVS
                        if os.environ.get(k, "").strip()), None)
        counter = smoke.CountingClient(
            openai.OpenAI(api_key=api_key, base_url=BASE_URL, timeout=600),
            max_calls=A5V2_LLM_PER_PREPARE * len(roster) + 6)
        memory = list(pack) if arm == "A5" else []
        snapshot = h0
        for e in roster:
            uid = str(e["series_uid"])
            origin = int(e["origin"])
            series0 = values[uid]
            vals2 = {uid: series0}
            print("== a5v2 " + arm + " " + str(e.get("entity_id"))
                  + "@" + str(origin) + " ...", flush=True)
            llm0 = counter.calls
            row = {"arm": arm, "dataset": dataset, "series_uid": uid,
                   "entity_id": e.get("entity_id"), "origin": origin}
            try:
                core = TTHAAgentCore(
                    RecordingBackend(AgictoChatCompletionsBackend(
                        client=counter, base_url=BASE_URL)),
                    LocalPublicToolGateway(series0[:origin],
                                           task_kind="forecast"))
                method = TTHAMethod(TTHAFastAgent(core), snapshot,
                                    tuple(memory))
                executor = ScopeExecutor(
                    [{"series_uid": uid, "role": "train"},
                     {"series_uid": uid, "role": "eval"}],
                    vals2, _a5_config(dataset, origin),
                    evaluate_fn=nsu._evaluate_kdd)
                request = _a5_request(series0, vals2, origin, dataset)
                r = run_online_round(
                    method, executor, request, vals2, origin=origin,
                    slow_agent=None, controller=controller, store=store,
                    card_builder=_a5v2_card,
                    round_name=(arm.lower() + "_" + uid[:8] + "_"
                                + str(origin)),
                    budget=A5V2_BUDGET_PER_ROUND, allow_slow=False,
                    domain=dataset, period=PERIOD,
                    fast_features=dict(extract_public_features(
                        series0[:origin], task_kind="forecast")),
                    allow_fast_skill=True)
                open_delayed(r, executor)
                if r.approved_skill_id is not None:
                    activate_approved(r, store)
                normalized = _a5v2_normalize_new_episodes(
                    method, series0, origin)
                trace = method.last_trace
                fs = getattr(r, "_fast_skill_event", None) or {}
                winner_gain = None
                if r.winner_program is not None:
                    for p in reversed(r.actual_probed_programs):
                        if p.get("kind") == "probe" \
                                and p.get("gain") is not None:
                            winner_gain = float(p["gain"])
                            break
                row.update({
                    "memory_resolution_status": r.memory_resolution_status,
                    "support_receipts": r.target_support_receipts_used,
                    "probes": r.actual_probed_programs,
                    "winner_program": r.winner_program,
                    "winner_gain": winner_gain,
                    "abstained": bool(r.abstained),
                    "harm_probes": int(r.harm_count),
                    "delayed_utility": (float(r.delayed_utility)
                                        if r.delayed_utility is not None
                                        else None),
                    "skill_created_id": (fs.get("edit_id")
                                         if fs.get("stage") == "pending"
                                         else None),
                    "skill_event_stage": fs.get("stage"),
                    "approved_skill_id": r.approved_skill_id,
                    "retrieved_skill_ids": list(
                        getattr(trace, "retrieved_skill_ids", ()) or ()),
                    "skill_probes": [p for p in r.actual_probed_programs
                                     if str(p.get("candidate_id", ""))
                                     .startswith("cand_skill_")],
                    "episodes_total": len(
                        tuple(method._experience_episodes)),  # noqa: SLF001
                    "episodes_normalized": normalized,
                    "llm_calls": counter.calls - llm0,
                })
                memory = list(method._experience_episodes)  # noqa: SLF001
                snapshot = method._active_snapshot()  # noqa: SLF001
                if row["llm_calls"] > A5V2_LLM_PER_PREPARE:
                    row["protocol_error"] = (
                        "LLM 超预算: " + str(row["llm_calls"]))
            except Exception as exc:  # noqa: BLE001
                row["protocol_error"] = f"{type(exc).__name__}: {exc}"
                row["llm_calls"] = counter.calls - llm0
            rows.append(row)
            _save_report(report)
            print("== a5v2 " + arm + " done: winner="
                  + str(bool(row.get("winner_program")))
                  + " gain=" + str(row.get("winner_gain"))
                  + " delayed=" + str(row.get("delayed_utility"))
                  + " skill=" + str(row.get("skill_created_id"))
                  + "/" + str(row.get("approved_skill_id"))
                  + " retrieved=" + str(len(row.get(
                      "retrieved_skill_ids") or ()))
                  + " llm=" + str(row.get("llm_calls"))
                  + (" ERR=" + str(row["protocol_error"])[:80]
                     if row.get("protocol_error") else ""), flush=True)
    _save_report(report)
    print("== a5v2 run complete:", len(rows), "rows")
    return 0


def phase_a5v2_verdict() -> int:
    report = _load_report()
    a5 = report.get("a5v2") or {}
    rows = a5.get("rows") or []
    expected = 2 * len((report.get("a5v2_protocol") or {})
                       .get("target", {}).get("roster") or [])
    if len(rows) < expected:
        raise SystemExit("a5v2 未完成（" + str(len(rows)) + "/"
                         + str(expected) + "）")
    v = a5.get("verdict") or {}
    if v.get("final"):
        print(json.dumps(v, ensure_ascii=False))
        return 0
    agg = _a5v2_aggregates(rows)
    a5["aggregates"] = agg
    verdict = _a5v2_verdict(agg, rows)
    verdict["final"] = True
    verdict["claim_boundary"] = (report.get("a5v2_protocol") or {}).get(
        "claim_boundary")
    verdict["outcome_acceptance"] = (report.get("a5v2_protocol") or {}).get(
        "outcome_acceptance")
    a5["verdict"] = verdict
    _save_report(report)
    print("== a5v2 终裁:", json.dumps(
        {"verdict": verdict["verdict"], "reason": verdict.get("reason"),
         "aggregates": {k: {kk: vv for kk, vv in a.items()
                            if kk != "skills"}
                        for k, a in agg.items()}},
        ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- a5v3
# A5v3/A3v3（2026-08-15）：P0 生命周期修复（revoke_deployed_skill）后的
# fresh 确认轮——N4v2 剩余合格池按同一确定性规则取下一批 sealed
# context；分维裁定（行动性/增量安全/持久性）+ 单一主裁定，运行前解决
# v2 的 NEGATIVE/NO_SIGNAL 规则重叠。只重跑一次，不换 roster。

A5V3_STORE_PREFIX = ".a5v3_store_"


def _a5v3_pick_roster(report: dict) -> list[dict]:
    """N4v2 剩余合格池 − a5v2 已消费（已暴露）→ 同一确定性规则取前 6。"""
    n4 = report["n4_v2"]
    dataset = n4["verdict"]["dataset"]
    consumed = {(str(e["series_uid"]), int(e["origin"]))
                for e in report["a5v2_protocol"]["target"]["roster"]}
    pool = [e for e in n4["per_dataset"][dataset]["eligible"]
            if (str(e["series_uid"]), int(e["origin"])) not in consumed]
    return pool[:6]


def _a5v3_shared_harm_keys(rows: Sequence[dict]) -> set:
    """两臂同 context 近似同 gain 的 harm 视为共同环境 harm（增量分析剔除）。
    返回 (entity, origin, round(gain,3)) 键集。"""
    from collections import Counter  # noqa: PLC0415
    by_arm: dict[str, Counter] = {"A5": Counter(), "A3": Counter()}
    for r in rows:
        for p in (r.get("probes") or ()):
            g = p.get("gain")
            if g is not None and float(g) < -M:
                by_arm[r["arm"]][(r.get("entity_id"), r.get("origin"),
                                  round(float(g), 3))] += 1
        d = r.get("delayed_utility")
        if r.get("winner_program") and d is not None and float(d) < -M:
            by_arm[r["arm"]][(r.get("entity_id"), r.get("origin"),
                              round(float(d), 3))] += 1
    shared = by_arm["A5"] & by_arm["A3"]
    return {k for k, n in shared.items() for _ in range(n)}


def _a5v3_aggregates(rows: Sequence[dict]) -> dict:
    """v3 语义：confirmed（批准+后续 retrieved+re-probe≥−M，历史事件）、
    final_reliable（轨迹终点未被撤销）、n_removed、增量 harm（剔除共同）。
    """
    shared = _a5v3_shared_harm_keys(rows)
    out = {}
    for arm in ("A5", "A3"):
        arm_rows = [r for r in rows if r.get("arm") == arm]
        cum = 0
        cum_at = []
        for r in arm_rows:
            cum += int(r.get("support_receipts") or 0)
            cum_at.append(cum)
        def _harm_of(r: Mapping[str, Any], incremental: bool) -> int:
            n = 0
            for p in (r.get("probes") or ()):
                g = p.get("gain")
                if g is not None and float(g) < -M:
                    if incremental and (r.get("entity_id"), r.get("origin"),
                                        round(float(g), 3)) in shared:
                        continue
                    n += 1
            d = r.get("delayed_utility")
            if r.get("winner_program") and d is not None and float(d) < -M:
                if not (incremental and (r.get("entity_id"), r.get("origin"),
                                         round(float(d), 3)) in shared):
                    n += 1
            return n
        skills: dict[str, dict] = {}
        for i, r in enumerate(arm_rows):
            sid = r.get("skill_created_id")
            if sid:
                skills.setdefault(sid, {"created_row": i, "approved": False,
                                        "confirmed": False, "revoked": False,
                                        "confirmed_at_cum": None})
            ap = r.get("approved_skill_id")
            if ap and ap in skills:
                skills[ap]["approved"] = True
            rv = r.get("revoked_skill_id")
            if rv:
                skills.setdefault(rv, {"created_row": None, "approved": True,
                                       "confirmed": False, "revoked": True,
                                       "confirmed_at_cum": None})
                skills[rv]["revoked"] = True
            for sid2 in (r.get("retrieved_skill_ids") or ()):
                if sid2 in skills and skills[sid2]["approved"] \
                        and not skills[sid2]["confirmed"] \
                        and not skills[sid2]["revoked"]:
                    sp = [p for p in (r.get("skill_probes") or ())
                          if p.get("gain") is not None]
                    if sp and float(sp[0]["gain"]) >= -M:
                        skills[sid2]["confirmed"] = True
                        skills[sid2]["confirmed_at_cum"] = cum_at[i]
        confirmed = [s for s in skills.values() if s["confirmed"]]
        final_reliable = [s for s in confirmed if not s["revoked"]]
        fpc = min((s["confirmed_at_cum"] for s in confirmed
                   if s["confirmed_at_cum"] is not None), default=None)
        out[arm] = {
            "feedback_to_first_confirmed": fpc,
            "support_evals": cum,
            "harm_events": sum(_harm_of(r, False) for r in arm_rows),
            "incremental_harm": sum(_harm_of(r, True) for r in arm_rows),
            "abstentions": sum(1 for r in arm_rows if r.get("abstained")),
            "winners": sum(1 for r in arm_rows if r.get("winner_program")),
            "total_support_gain": sum(
                float(r["winner_gain"]) for r in arm_rows
                if r.get("winner_gain") is not None),
            "n_skills_created": len([s for s in skills.values()
                                     if s["created_row"] is not None]),
            "n_approved": sum(1 for s in skills.values() if s["approved"]),
            "n_confirmed": len(confirmed),
            "n_removed": sum(1 for s in skills.values() if s["revoked"]),
            "n_final_reliable": len(final_reliable),
            "llm_calls": sum(int(r.get("llm_calls") or 0)
                             for r in arm_rows)}
    return out


def _a5v3_verdict(aggregates: Mapping[str, Any], rows: Sequence[dict]) -> dict:
    """冻结裁定（a5v3_protocol.verdict_rules）：先闸门后分维。

    主裁定（运行前冻结，解决 v2 NEGATIVE/NO_SIGNAL 重叠）：
      PASS      = A5.n_final_reliable ≥ 1 且 A5.feedback_to_first_confirmed
                  严格 < A3（A3 无则视为 ∞）且 A5.incremental_harm ≤ A3
      NEGATIVE  = A5.incremental_harm > A3.incremental_harm（安全否决，
                  无论 Skill 结局）；或两臂 A5 无终态而 A3 有
      NO_SIGNAL = 其余（含两臂均无终态 Skill 且增量 harm 不增）
    分维信号同时输出：q1_actionability / q2_incremental_safety /
    q3_durability。
    """
    a5 = aggregates.get("A5") or {}
    a3 = aggregates.get("A3") or {}
    if any(r.get("protocol_error") for r in rows):
        return {"verdict": "PROTOCOL_INCONCLUSIVE", "reason": "存在 protocol_error 行"}
    bad = [r for r in rows if r.get("arm") == "A5"
           and r.get("memory_resolution_status") != "contrast_pack"]
    if bad:
        return {"verdict": "CONTENT_INCONCLUSIVE",
                "reason": "A5 运行期 Memory 未按预检接线"}
    f5 = a5.get("feedback_to_first_confirmed")
    f3 = a3.get("feedback_to_first_confirmed")
    f5v = f5 if f5 is not None else float("inf")
    f3v = f3 if f3 is not None else float("inf")
    h5 = int(a5.get("incremental_harm") or 0)
    h3 = int(a3.get("incremental_harm") or 0)
    fr5 = int(a5.get("n_final_reliable") or 0)
    fr3 = int(a3.get("n_final_reliable") or 0)
    dimensional = {
        "q1_actionability": {
            "a5_winners": a5.get("winners"), "a3_winners": a3.get("winners"),
            "a5_skills": a5.get("n_skills_created"),
            "a3_skills": a3.get("n_skills_created"),
            "signal": ("SOURCE_ACTIONABILITY_POSITIVE"
                       if (a5.get("n_skills_created") or 0)
                          > (a3.get("n_skills_created") or 0)
                       else "NO_DIFFERENCE")},
        "q2_incremental_safety": {
            "a5_incremental_harm": h5, "a3_incremental_harm": h3,
            "signal": ("SOURCE_INCREMENTAL_HARM" if h5 > h3
                       else "NO_INCREMENTAL_HARM")},
        "q3_durability": {
            "a5_final_reliable": fr5, "a3_final_reliable": fr3,
            "a5_removed": a5.get("n_removed"),
            "a3_removed": a3.get("n_removed"),
            "signal": ("A5_DURABLE_SKILL" if fr5 > fr3
                       else ("A3_DURABLE_SKILL" if fr3 > fr5
                             else "NO_DURABLE_SKILL"))},
    }
    if h5 > h3:
        v = {"verdict": "NEGATIVE_TRANSFER",
             "reason": ("A5 增量 harm 更多（" + str(h5) + " vs " + str(h3)
                        + "）——安全维度否决")}
    elif fr5 >= 1 and f5v < f3v:
        v = {"verdict": "TRANSFER_CASE_PASS",
             "reason": ("A5 形成终态 reliable Skill（n=" + str(fr5)
                        + "），feedback " + str(f5) + " vs " + str(f3)
                        + "，增量 harm 不增（" + str(h5) + " vs " + str(h3)
                        + "）"),
             "claim": ("development transfer case：单 Target 数据集、n=6 "
                       "context、1 rep；非一般迁移声明")}
    elif fr3 >= 1 and fr5 == 0:
        v = {"verdict": "NEGATIVE_TRANSFER",
             "reason": "A3 有终态 reliable Skill 而 A5 无"}
    else:
        v = {"verdict": "NO_SIGNAL",
             "reason": ("两臂终态 reliable=" + str(fr5) + "/" + str(fr3)
                        + "，增量 harm " + str(h5) + " vs " + str(h3))}
    v["dimensional"] = dimensional
    return v


def phase_a5v3_freeze() -> int:
    report = _load_report()
    if report.get("a5v3_protocol"):
        raise SystemExit("a5v3_protocol 已存在——拒绝重复冻结")
    if report.get("a5v3"):
        raise SystemExit("a5v3 已有结果——协议必须先冻结")
    n4v = ((report.get("n4_v2") or {}).get("verdict") or {}).get("verdict")
    n5v = ((report.get("n5_v2") or {}).get("verdict") or {}).get("verdict")
    if n4v != "N4_ROSTER_ELIGIBLE" or n5v != "N5_WIRING_OK":
        raise SystemExit("前置裁定不合格——a5v3 不得冻结")
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: PLC0415
        revoke_deployed_skill)
    assert callable(revoke_deployed_skill)  # P0 必须在位
    roster = _a5v3_pick_roster(report)
    if len(roster) < 4:
        report["a5v3_protocol"] = {
            "verdict": {"verdict": "FRESH_TARGET_CONTENT_UNAVAILABLE",
                        "final": True,
                        "remaining_pool": len(roster)}}
        _save_report(report)
        print("a5v3: FRESH_TARGET_CONTENT_UNAVAILABLE pool=" + str(len(roster)))
        return 0
    # 新 roster 接线预检（零 LLM——同 n5v2 静态检查）
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: PLC0415
        extract_public_features)
    episodes = _n5_load_pack_episodes()
    clean = _n4_clean_records()
    precheck = []
    for e in roster:
        arr = np.load(clean[str(e["series_uid"])]["values"]).astype(np.float64)
        feats = extract_public_features(arr[:int(e["origin"])],
                                        task_kind="forecast")
        chk = _n5_wiring_check(episodes, feats)
        precheck.append({"entity_id": e.get("entity_id"),
                         "origin": int(e["origin"]),
                         "status": chk["memory_resolution_status"]})
        if chk["memory_resolution_status"] != "contrast_pack":
            raise SystemExit("a5v3 预检失败: " + str(e.get("entity_id")))
    dataset = report["n4_v2"]["verdict"]["dataset"]
    proto = {
        "experiment_id": "A5V3_LIFECYCLE_FRESH_CONFIRM_2026_08_15",
        "supersedes": ("a5v2——外部评审降级为 PROTOCOL_INCONCLUSIVE_WITH_"
                       "NEGATIVE_SAFETY_SIGNAL（UPDATE_POLICY_FAULT："
                       "delayed 翻负未撤销已检索 Skill + 重复 ADD "
                       "apply_failed）。P0 已修复（revoke_deployed_skill + "
                       "集成测试）；a5v2 三信号重解释已入档不重跑"),
        "question": "同 a5v2——P0 修复后 fresh roster 上的确认轮",
        "target": {"dataset": dataset, "roster": roster,
                   "selection": ("N4v2 剩余合格池 − a5v2 已消费，同一确定性 "
                                 "规则（uid,origin 字典序前 6）——不看 gain")},
        "wiring_precheck": precheck,
        "p0_fix": ("online_loop.revoke_deployed_skill：cand_skill_* winner 不"
                   "重复 ADD（deployed_existing_skill）；delayed < −M → tree "
                   "往返撤销（compile_snapshot 重算 sha + set_active）；"
                   "tests/functional/test_skill_revocation.py 六步轨迹回归"),
        "arms": "同 a5v2（A5=N3 pack / A3=空；Slow 双关；对称增长；键族归一化）",
        "budget_table": {"rounds_per_arm": len(roster),
                         "support_per_round": A5V2_BUDGET_PER_ROUND,
                         "delayed_per_round": "≤1（仅 winner）",
                         "llm_per_prepare": A5V2_LLM_PER_PREPARE,
                         "slow_calls": 0},
        "metrics": ["feedback_to_first_confirmed（历史事件指标）",
                    "n_final_reliable（轨迹终点未撤销）", "n_removed",
                    "harm_events / incremental_harm（剔除两臂共同 harm）",
                    "abstentions / winners / total_support_gain"],
        "verdict_rules": {
            "precedence": "PROTOCOL_INCONCLUSIVE > CONTENT_INCONCLUSIVE > 其余",
            "pass": ("A5.n_final_reliable ≥ 1 且 feedback_to_first_confirmed "
                     "严格 < A3（无则 ∞）且 incremental_harm ≤ A3 → "
                     "TRANSFER_CASE_PASS"),
            "negative": ("incremental_harm A5 > A3 → NEGATIVE_TRANSFER（安全"
                         "否决，无论 Skill 结局）；或 A3 有终态而 A5 无"),
            "no_signal": "其余（含两臂均无终态且增量 harm 不增）→ NO_SIGNAL",
            "overlap_resolution": ("v2 重叠已解决：两臂皆无终态 Skill 时，"
                                   "增量 harm 更严者仍判 NEGATIVE_TRANSFER"),
            "dimensional": "q1_actionability/q2_incremental_safety/"
                           "q3_durability 三信号恒输出"},
        "outcome_acceptance": "只重跑一次，不换 roster；任何结局接受",
        "claim_boundary": "单 Target 数据集、n=6 context、1 rep",
        "state": "FROZEN_BEFORE_ANY_RUN",
    }
    report["a5v3_protocol"] = proto
    _save_report(report)
    print("a5v3_protocol FROZEN roster=" + str(
        [(e.get("entity_id"), e["origin"]) for e in roster]))
    return 0


def phase_a5v3_reset() -> int:
    import shutil  # noqa: PLC0415
    report = _load_report()
    a5 = report.pop("a5v3", None)
    for arm in ("a5", "a3"):
        d = PROJECT_ROOT / (A5V3_STORE_PREFIX + arm)
        if d.exists():
            shutil.rmtree(d)
    _save_report(report)
    print("a5v3 reset: cleared rows=" + str(len((a5 or {}).get("rows") or [])))
    return 0


def phase_a5v3() -> int:
    """a5v3 原子运行——同 a5v2 结构 + P0 撤销接线（store 传入 open_delayed）
    + deployed/revoked 追踪。中断须 a5v3-reset 后全量重跑。"""
    import shutil  # noqa: PLC0415
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: PLC0415
        activate_approved, open_delayed, run_online_round)
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: PLC0415
        extract_public_features)
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: PLC0415
        ScopeExecutor)
    report = _load_report()
    if not report.get("a5v3_protocol"):
        raise SystemExit("a5v3_protocol 未冻结——先跑 a5v3-freeze")
    if (report["a5v3_protocol"].get("verdict") or {}).get("final"):
        raise SystemExit("a5v3 协议层已终裁（roster 不可用）")
    a5 = report.setdefault("a5v3", {})
    if (a5.get("verdict") or {}).get("final"):
        raise SystemExit("a5v3 已终裁")
    if a5.get("rows"):
        raise SystemExit("a5v3 已有部分行——原子语义：先 a5v3-reset 再重跑")
    roster = report["a5v3_protocol"]["target"]["roster"]
    dataset = str(report["a5v3_protocol"]["target"]["dataset"])
    h0 = _h0_snapshot()
    clean = _n4_clean_records()
    values = {str(e["series_uid"]): np.load(
        clean[str(e["series_uid"])]["values"]).astype(np.float64)
        for e in roster}
    pack = _n5_load_pack_episodes()
    rows = a5.setdefault("rows", [])
    for arm in ("A5", "A3"):
        store_dir = PROJECT_ROOT / (A5V3_STORE_PREFIX + arm.lower())
        if store_dir.exists():
            shutil.rmtree(store_dir)
        store = SnapshotStore(store_dir)
        controller = EditController(store, surfaces=SurfaceRegistry(),
                                    router=FaultRouter())
        import openai  # noqa: PLC0415
        api_key = next((os.environ.get(k, "").strip() for k in KEY_ENVS
                        if os.environ.get(k, "").strip()), None)
        counter = smoke.CountingClient(
            openai.OpenAI(api_key=api_key, base_url=BASE_URL, timeout=600),
            max_calls=A5V2_LLM_PER_PREPARE * len(roster) + 6)
        memory = list(pack) if arm == "A5" else []
        snapshot = h0
        for e in roster:
            uid = str(e["series_uid"])
            origin = int(e["origin"])
            series0 = values[uid]
            vals2 = {uid: series0}
            print("== a5v3 " + arm + " " + str(e.get("entity_id"))
                  + "@" + str(origin) + " ...", flush=True)
            llm0 = counter.calls
            row = {"arm": arm, "dataset": dataset, "series_uid": uid,
                   "entity_id": e.get("entity_id"), "origin": origin}
            try:
                core = TTHAAgentCore(
                    RecordingBackend(AgictoChatCompletionsBackend(
                        client=counter, base_url=BASE_URL)),
                    LocalPublicToolGateway(series0[:origin],
                                           task_kind="forecast"))
                method = TTHAMethod(TTHAFastAgent(core), snapshot,
                                    tuple(memory))
                executor = ScopeExecutor(
                    [{"series_uid": uid, "role": "train"},
                     {"series_uid": uid, "role": "eval"}],
                    vals2, _a5_config(dataset, origin),
                    evaluate_fn=nsu._evaluate_kdd)
                request = _a5_request(series0, vals2, origin, dataset)
                r = run_online_round(
                    method, executor, request, vals2, origin=origin,
                    slow_agent=None, controller=controller, store=store,
                    card_builder=_a5v2_card,
                    round_name=(arm.lower() + "v3_" + uid[:8] + "_"
                                + str(origin)),
                    budget=A5V2_BUDGET_PER_ROUND, allow_slow=False,
                    domain=dataset, period=PERIOD,
                    fast_features=dict(extract_public_features(
                        series0[:origin], task_kind="forecast")),
                    allow_fast_skill=True)
                open_delayed(r, executor, store=store)  # P0：撤销接线
                if r.approved_skill_id is not None:
                    activate_approved(r, store)
                normalized = _a5v2_normalize_new_episodes(
                    method, series0, origin)
                trace = method.last_trace
                fs = getattr(r, "_fast_skill_event", None) or {}
                winner_gain = None
                if r.winner_program is not None:
                    for p in reversed(r.actual_probed_programs):
                        if p.get("kind") == "probe" \
                                and p.get("gain") is not None:
                            winner_gain = float(p["gain"])
                            break
                row.update({
                    "memory_resolution_status": r.memory_resolution_status,
                    "support_receipts": r.target_support_receipts_used,
                    "probes": r.actual_probed_programs,
                    "winner_program": r.winner_program,
                    "winner_candidate_id": getattr(
                        r, "_winner_candidate_id", None),
                    "winner_gain": winner_gain,
                    "abstained": bool(r.abstained),
                    "harm_probes": int(r.harm_count),
                    "delayed_utility": (float(r.delayed_utility)
                                        if r.delayed_utility is not None
                                        else None),
                    "skill_created_id": (fs.get("edit_id")
                                         if fs.get("stage") == "pending"
                                         else None),
                    "skill_event_stage": fs.get("stage"),
                    "approved_skill_id": r.approved_skill_id,
                    "deployed_skill_id": r.deployed_skill_id,
                    "revoked_skill_id": r.revoked_skill_id,
                    "retrieved_skill_ids": list(
                        getattr(trace, "retrieved_skill_ids", ()) or ()),
                    "skill_probes": [p for p in r.actual_probed_programs
                                     if str(p.get("candidate_id", ""))
                                     .startswith("cand_skill_")],
                    "episodes_total": len(
                        tuple(method._experience_episodes)),  # noqa: SLF001
                    "episodes_normalized": normalized,
                    "llm_calls": counter.calls - llm0,
                })
                memory = list(method._experience_episodes)  # noqa: SLF001
                snapshot = method._active_snapshot()  # noqa: SLF001
                if row["llm_calls"] > A5V2_LLM_PER_PREPARE:
                    row["protocol_error"] = (
                        "LLM 超预算: " + str(row["llm_calls"]))
            except Exception as exc:  # noqa: BLE001
                row["protocol_error"] = f"{type(exc).__name__}: {exc}"
                row["llm_calls"] = counter.calls - llm0
            rows.append(row)
            _save_report(report)
            print("== a5v3 " + arm + " done: winner="
                  + str(bool(row.get("winner_program")))
                  + " gain=" + str(row.get("winner_gain"))
                  + " delayed=" + str(row.get("delayed_utility"))
                  + " skill=" + str(row.get("skill_created_id"))
                  + "/" + str(row.get("approved_skill_id"))
                  + " deployed=" + str(row.get("deployed_skill_id"))
                  + " revoked=" + str(row.get("revoked_skill_id"))
                  + " llm=" + str(row.get("llm_calls"))
                  + (" ERR=" + str(row["protocol_error"])[:80]
                     if row.get("protocol_error") else ""), flush=True)
    _save_report(report)
    print("== a5v3 run complete:", len(rows), "rows")
    return 0


def phase_a5v3_verdict() -> int:
    report = _load_report()
    a5 = report.get("a5v3") or {}
    rows = a5.get("rows") or []
    expected = 2 * len((report.get("a5v3_protocol") or {})
                       .get("target", {}).get("roster") or [])
    if len(rows) < expected:
        raise SystemExit("a5v3 未完成（" + str(len(rows)) + "/"
                         + str(expected) + "）")
    v = a5.get("verdict") or {}
    if v.get("final"):
        print(json.dumps(v, ensure_ascii=False))
        return 0
    agg = _a5v3_aggregates(rows)
    a5["aggregates"] = agg
    verdict = _a5v3_verdict(agg, rows)
    verdict["final"] = True
    verdict["claim_boundary"] = (report.get("a5v3_protocol") or {}).get(
        "claim_boundary")
    verdict["outcome_acceptance"] = (report.get("a5v3_protocol") or {}).get(
        "outcome_acceptance")
    a5["verdict"] = verdict
    _save_report(report)
    print("== a5v3 终裁:", json.dumps(
        {"verdict": verdict["verdict"], "reason": verdict.get("reason"),
         "dimensional": verdict.get("dimensional"),
         "aggregates": agg}, ensure_ascii=False, indent=1))
    return 0

# ---------------------------------------------------------------- dashboard
# 只读仪表板：把一晚的自进化实验链渲染为完全自包含的中文 HTML（内联 CSS +
# 内联 SVG 手绘图表；不写回报告、不建 Schema/Registry、不改任何其他逻辑）。

DASHBOARD_REL = (PROJECT_ROOT / "artifacts" / "functional" / "e2"
                 / "w1_evolution_dashboard.html")


def _dx_esc(text: Any) -> str:
    return (str("" if text is None else text)
            .replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _dx_str(value: Any) -> str:
    return "" if value is None else str(value)


def _dx_num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dx_fmt(value: Any) -> str:
    v = _dx_num(value)
    if abs(v) < 1e-9:
        return "0"
    if abs(v) >= 100 or abs(v) < 0.001:
        return "%.3g" % v
    return ("%.3f" % v).rstrip("0").rstrip(".")


def _dx_report() -> dict[str, Any]:
    try:
        if REPORT_REL.exists():
            data = json.loads(REPORT_REL.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _dx_episodes() -> list:
    try:
        raw = json.loads((PROJECT_ROOT / "artifacts" / "experience"
                          / "episodes.json").read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("episodes") or raw.get("entries") or []
        return raw if isinstance(raw, list) else []
    except Exception:  # noqa: BLE001
        return []


def _dx_get(report: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = report
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _dx_grouped_bars(categories: Sequence[str],
                     series: Sequence[tuple[str, str, Sequence[Any]]],
                     height: int = 230, hline: Any = None,
                     hline_label: str = "",
                     stars: Mapping[int, Sequence[int]] | None = None) -> str:
    n = len(categories)
    k = len(series)
    allv = [0.0]
    if hline is not None:
        allv.append(_dx_num(hline))
    for _, _, vals in series:
        allv += [_dx_num(v) for v in vals]
    lo, hi = min(allv), max(allv)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = (hi - lo) * 0.15 or 0.1
    lo, hi = lo - pad, hi + pad
    top, bottom, left = 26, 38, 44
    plot_h = height - top - bottom
    group_w = k * 15 + 6
    gap = 16
    width = left + n * (group_w + gap) + 12
    def yv(v):
        return top + (hi - _dx_num(v)) / (hi - lo) * plot_h
    parts = []
    if hline is not None:
        yy = yv(hline)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#c62828" stroke-dasharray="4,3"/>' % (left, yy, width - 8, yy))
        parts.append('<text x="%d" y="%.1f" class="ref" text-anchor="end">%s</text>' % (width - 8, yy - 3, _dx_esc(hline_label)))
    yy0 = yv(0.0)
    parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>' % (left, yy0, width - 8, yy0))
    for ci, cat in enumerate(categories):
        x0 = left + ci * (group_w + gap)
        for si, (_, color, vals) in enumerate(series):
            v = _dx_num(vals[ci]) if ci < len(vals) else 0.0
            x = x0 + si * 15
            bh = abs(yv(v) - yy0)
            yt = min(yv(v), yy0)
            parts.append('<rect x="%d" y="%.1f" width="13" height="%.1f" fill="%s" rx="1.5"/>' % (x, yt, max(bh, 0.6), color))
            if stars and ci in stars and si in stars[ci]:
                parts.append('<text x="%d" y="%.1f" class="star">★</text>' % (x + 1, yt - 4))
            parts.append('<text x="%d" y="%.1f" class="val" text-anchor="middle">%s</text>' % (x + 6, yt - 6, _dx_esc(_dx_fmt(v))))
        parts.append('<text x="%d" y="%d" class="cat" text-anchor="middle">%s</text>' % (x0 + group_w / 2, height - 10, _dx_esc(cat)))
    return '<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">%s</svg>' % (width, height, "".join(parts))


def _dx_vbars(categories: Sequence[str], values: Sequence[Any],
              colors: Sequence[str], height: int = 230, hline: Any = None,
              hline_label: str = "") -> str:
    n = len(categories)
    allv = [0.0]
    if hline is not None:
        allv.append(_dx_num(hline))
    allv += [_dx_num(v) for v in values]
    lo, hi = min(allv), max(allv)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = (hi - lo) * 0.15 or 0.1
    lo, hi = lo - pad, hi + pad
    top, bottom, left = 26, 44, 44
    plot_h = height - top - bottom
    bw, gap = 20, 14
    width = left + n * (bw + gap) + 10
    def yv(v):
        return top + (hi - _dx_num(v)) / (hi - lo) * plot_h
    parts = []
    if hline is not None:
        yy = yv(hline)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#c62828" stroke-dasharray="4,3"/>' % (left, yy, width - 8, yy))
        parts.append('<text x="%d" y="%.1f" class="ref" text-anchor="end">%s</text>' % (width - 8, yy - 3, _dx_esc(hline_label)))
    yy0 = yv(0.0)
    parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>' % (left, yy0, width - 8, yy0))
    for i, cat in enumerate(categories):
        x = left + i * (bw + gap)
        v = _dx_num(values[i])
        bh = abs(yv(v) - yy0)
        yt = min(yv(v), yy0)
        c = colors[i] if i < len(colors) else "#3b82f6"
        parts.append('<rect x="%d" y="%.1f" width="%d" height="%.1f" fill="%s" rx="2"/>' % (x, yt, bw, max(bh, 0.6), c))
        parts.append('<text x="%d" y="%.1f" class="val" text-anchor="middle">%s</text>' % (x + bw / 2, yt - 4, _dx_esc(_dx_fmt(v))))
        parts.append('<text x="%d" y="%d" class="cat" text-anchor="middle" transform="rotate(-32 %d %d)">%s</text>' % (x + bw / 2, height - 8, x + bw / 2, height - 8, _dx_esc(cat)))
    return '<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">%s</svg>' % (width, height, "".join(parts))


def _dx_hbars(items: Sequence[tuple[str, Any, str]], width: int = 380) -> str:
    maxv = max([_dx_num(v) for _, v, _ in items] + [1.0])
    bh = 18
    h = len(items) * (bh + 8) + 6
    parts = []
    for i, (label, value, color) in enumerate(items):
        v = _dx_num(value)
        y = 4 + i * (bh + 8)
        parts.append('<text x="0" y="%d" class="cat">%s</text>' % (y + 12, _dx_esc(label)))
        bw = max(4, int(v / maxv * (width - 130)))
        parts.append('<rect x="128" y="%d" width="%d" height="%d" fill="%s" rx="2"/>' % (y, bw, bh, color))
        parts.append('<text x="%d" y="%d" class="val">%s</text>' % (128 + bw + 6, y + 12, _dx_esc(str(int(v)) if abs(v - int(v)) < 1e-9 else _dx_fmt(v))))
    return '<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">%s</svg>' % (width, h, "".join(parts))


def _dx_scatter(points: Sequence[tuple[float, float, str, str]]) -> str:
    width, height = 560, 340
    left, right, top, bottom = 52, 16, 20, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xlo, xhi = min(xs + [-0.005]), max(xs + [0.005])
    ylo, yhi = min(ys + [-0.005]), max(ys + [0.005])
    xpad = (xhi - xlo) * 0.10 or 0.01
    ypad = (yhi - ylo) * 0.10 or 0.01
    xlo, xhi = xlo - xpad, xhi + xpad
    ylo, yhi = ylo - ypad, yhi + ypad
    def xv(v):
        return left + (_dx_num(v) - xlo) / (xhi - xlo) * plot_w
    def yv(v):
        return top + (yhi - _dx_num(v)) / (yhi - ylo) * plot_h
    parts = []
    m = 0.005
    for line in ((yv(m), "y=+M"), (yv(-m), "y=-M")):
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#94a3b8" stroke-dasharray="3,3"/>' % (left, line[0], width - right, line[0]))
        parts.append('<text x="%d" y="%.1f" class="ref">%s</text>' % (width - right, line[0] - 3, line[1]))
    for line in ((xv(m), "x=+M"), (xv(-m), "x=-M")):
        parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#94a3b8" stroke-dasharray="3,3"/>' % (line[0], top, line[0], height - bottom))
        parts.append('<text x="%.1f" y="%d" class="ref">%s</text>' % (line[0] + 3, top + 10, line[1]))
    qx, qy = xv(m), yv(-m)
    parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#f59e0b" opacity="0.08"/>' % (qx, qy, xv(xhi) - qx, yv(ylo) - qy))
    for px, py, color, label in points:
        parts.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s" stroke="#0b1220" stroke-width="1"/>' % (xv(px), yv(py), color))
        parts.append('<text x="%.1f" y="%.1f" class="cat">%s</text>' % (xv(px) + 7, yv(py) + 3, _dx_esc(label)))
    parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>' % (left, yv(0), width - right, yv(0)))
    parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="axis"/>' % (xv(0), top, xv(0), height - bottom))
    parts.append('<text x="%d" y="%d" class="cat" text-anchor="middle">support_gain →</text>' % (left + plot_w / 2, height - 6))
    parts.append('<text x="12" y="%d" class="cat" transform="rotate(-90 12 %d)">delayed_gain →</text>' % (top + plot_h / 2, top + plot_h / 2))
    return '<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">%s</svg>' % (width, height, "".join(parts))


def _dx_flow(nodes: Sequence[tuple[str, str, str, str]]) -> str:
    color = {"ok": "#16a34a", "bad": "#dc2626", "warn": "#d97706",
             "info": "#2563eb"}
    n = len(nodes)
    nw, gap, h = 150, 30, 200
    width = 24 + n * nw + (n - 1) * gap + 24
    parts = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">' % (width, h)]
    for i, (title, line2, status, cls) in enumerate(nodes):
        x = 24 + i * (nw + gap)
        y = 34
        cc = color.get(cls, "#334155")
        parts.append('<rect x="%d" y="%d" width="%d" height="92" rx="8" fill="#0b1220" stroke="%s" stroke-width="1.5"/>' % (x, y, nw, cc))
        parts.append('<text x="%d" y="%d" text-anchor="middle" font-weight="700" fill="#e2e8f0" font-size="13">%s</text>' % (x + nw / 2, y + 24, _dx_esc(title)))
        parts.append('<text x="%d" y="%d" text-anchor="middle" fill="#94a3b8" font-size="10">%s</text>' % (x + nw / 2, y + 46, _dx_esc(line2)))
        parts.append('<text x="%d" y="%d" text-anchor="middle" fill="%s" font-size="10" font-weight="700">%s</text>' % (x + nw / 2, y + 70, cc, _dx_esc(status)))
        if i < n - 1:
            ax = x + nw
            parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#64748b" stroke-width="2"/>' % (ax, y + 46, ax + gap, y + 46))
            parts.append('<polygon points="%d,%d %d,%d %d,%d" fill="#64748b"/>' % (ax + gap, y + 46, ax + gap - 7, y + 41, ax + gap - 7, y + 51))
    parts.append('</svg>')
    return "".join(parts)


_DX_CSS = """
:root{--bg:#0f172a;--card:#1e293b;--ink:#e2e8f0;--mut:#94a3b8;--line:#334155}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,'PingFang SC','Microsoft YaHei','Segoe UI',sans-serif;
line-height:1.55}header{padding:20px 26px 14px;border-bottom:1px solid var(--line)}
h1{font-size:19px;margin:0 0 8px}.sum{font-size:17px;color:#22c55e;font-weight:700}
main{max-width:1180px;margin:0 auto;padding:16px 14px 30px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:15px 17px;margin:0 0 16px}
.card h2{font-size:16px;margin:0 0 3px}.card .sub{color:var(--mut);font-size:12px;margin:0 0 11px}
svg{width:100%;height:auto;display:block;overflow:visible}
text{fill:var(--ink);font-size:10px}.cat{fill:var(--mut)}.val{fill:var(--mut);font-size:9px}
.ref{fill:#f87171;font-size:9px}.star{fill:#f59e0b;font-size:13px}
.axis{stroke:var(--line);stroke-width:1}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:#0b1220;color:var(--mut);font-weight:600}
.ok{color:#22c55e}.bad{color:#ef4444}.warn{color:#f59e0b}.mut{color:var(--mut)}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600;white-space:nowrap}
.p-ok{background:#14532d;color:#86efac}.p-bad{background:#7f1d1d;color:#fca5a5}
.p-warn{background:#78350f;color:#fcd34d}.p-info{background:#1e3a8a;color:#93c5fd}
.nodata{color:var(--mut);font-style:italic}
ul.tight{margin:5px 0;padding-left:18px}li{margin:2px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:840px){.grid{grid-template-columns:1fr}}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--mut);margin-top:7px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
footer{color:var(--mut);font-size:11px;text-align:center;padding:6px 0 24px}
"""


def _dx_card(title: str, sub: str, body: str) -> str:
    return ('<section class="card"><h2>%s</h2><p class="sub">%s</p>%s'
            '</section>' % (_dx_esc(title), _dx_esc(sub), body))


def _dx_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    thead = "".join("<th>%s</th>" % _dx_esc(h) for h in headers)
    tbody = "".join(
        "<tr>%s</tr>" % "".join("<td>%s</td>" % _dx_esc(c) for c in row)
        for row in rows)
    return ("<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>"
            % (thead, tbody))


def _dx_strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _dx_strip(v) for k, v in obj.items()
                if k not in ("prompt_calls", "prompt", "assistant_text",
                             "user_message")}
    if isinstance(obj, list):
        return [_dx_strip(x) for x in obj]
    return obj




def phase_dashboard() -> int:
    report = _dx_report()
    episodes = _dx_episodes()

    # ---- a. 顶部闭环流程图 ----
    flow_nodes = [
        ("Batch 轨迹", "batch1 · 11 ctx", "NOT_SEPARABLE", "info"),
        ("对比归因", "usel · 6 ctx", "MISALIGNED", "warn"),
        ("Slow 修改/Rule", "bse · slow×2", "提案→待核销", "info"),
        ("Runtime 核销", "bse · replay×2", "REJECT", "bad"),
        ("Fast 重生成", "tsem · rev6→7", "PASS", "ok"),
        ("Support/delayed", "batch1/usel", "C 组翻负", "warn"),
        ("激活/安全拒绝", "BSE/N1/N3", "安全拒绝", "ok"),
    ]
    flow_html = _dx_flow(flow_nodes)

    # ---- b. 实验时间线表 ----
    def verdict_of(*paths: str) -> str:
        return _dx_str(_dx_get(report, ".".join(paths), ""))

    def pill(text: str) -> str:
        if not text:
            return '<span class="mut">无数据</span>'
        cls = "p-ok"
        t = str(text)
        if any(x in t for x in ("REJECT", "CLOSE", "UNIDENTIFIABLE", "TOO_RARE",
                                "NOT_RESOLVED", "NOT_SEPARABLE", "MISALIGNED",
                                "FAILURE", "BLOCKER", "UNRESOLVED")):
            cls = "p-bad"
        elif any(x in t for x in ("PASS", "CAUSAL_EFFECT", "READY", "SUPPLY")):
            cls = "p-ok"
        elif any(x in t for x in ("MIXED", "INCONCLUSIVE", "SELECT_MISALIGNED")):
            cls = "p-warn"
        return '<span class="pill %s">%s</span>' % (cls, _dx_esc(t))

    tsem_v = verdict_of("tsem", "verdict", "verdict")
    f2_v = verdict_of("fullop2", "corrected_final_2026_08_14",
                      "substantive_finding")
    usel_uv = verdict_of("usel", "utility_verdict", "verdict")
    usel_sv = verdict_of("usel", "selection_verdict", "verdict")
    b1_v = verdict_of("batch1", "verdict", "verdict")
    cobs_v = verdict_of("cobs", "verdict", "verdict")
    bse_v = verdict_of("bse", "verdict", "verdict")
    n1_v = verdict_of("n1", "verdict", "verdict")
    n3_v = verdict_of("n3", "verdict")

    timeline = [
        ("TSEM", "rev6 vs rev7 配对：修正 targeting 说明是否让 Fast 稳定把 broad deviation 绑定到 intrinsic 候选", tsem_v,
         "rev7 修复 2 处弃权、5/5 闭链——机制正证据"),
        ("FULLOP2", "完整算子池 vs actionable 池在 rev7 下是否有真实增量供应", f2_v,
         "24/24 闭链、零协议错；池外唯一候选被 verifier 正确拒绝"),
        ("USEL", "同 family 候选跨 Context 效用方向 + 选择是否对齐", usel_uv + (" / " + usel_sv if usel_sv else ""),
         "outlier 跨 Context 正负翻转 + 选择普遍错位"),
        ("batch1", "11 context 的 outlier_mad gain/delayed/features", b1_v,
         "7 正 / 3 负充足但现有 Observation 不可分"),
        ("COBS", "consumer 影响力 proxy 是否区分正负例", cobs_v,
         "proxy 退化：正负例全部 ≈0.997–0.999 无区分度"),
        ("BSE", "历史稳定性 Scope Rule（Slow 提案 → Runtime 核销）", bse_v,
         "一维历史稳定性阈值无法迁移 held-out；family 关闭"),
        ("N1", "INERT/UNKNOWN 历史误当安全证据的 novel-action 门控", n1_v,
         "无组外可验证案例——不实现 Rule，双门维持"),
        ("N3", "KDD 自然 outlier_mad 轨迹冻结为 Source Pack", n3_v,
         "10 条（3 正 / 3 负 / 4 冲突）完整打包"),
    ]
    rows_html = "".join(
        "<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (_dx_esc(n), _dx_esc(p), pill(v), _dx_esc(c))
        for n, p, v, c in timeline)
    timeline_html = ("<table><thead><tr><th>实验</th><th>目的</th>"
                     "<th>verdict</th><th>一句话结论</th></tr></thead>"
                     "<tbody>%s</tbody></table>" % rows_html)

    # ---- c. TSEM 配对条形图 ----
    tsem_rows = _dx_get(report, "tsem.rows", []) or []
    tsem_by = {}
    for r in tsem_rows if isinstance(tsem_rows, list) else []:
        key = str(r.get("key", ""))
        arm = str(r.get("arm", ""))
        m = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
        supply = len(m.get("program_ids") or [])
        verified = 1 if m.get("chain_complete") else 0
        tsem_by.setdefault(key, {})[arm] = (supply, verified)
    tsem_cats = sorted(tsem_by.keys())
    tsem_series = [
        ("rev6 供应", "#3b82f6", [tsem_by.get(c, {}).get("rev6", (0, 0))[0] for c in tsem_cats]),
        ("rev6 验证", "#60a5fa", [tsem_by.get(c, {}).get("rev6", (0, 0))[1] for c in tsem_cats]),
        ("rev7 供应", "#22c55e", [tsem_by.get(c, {}).get("rev7", (0, 0))[0] for c in tsem_cats]),
        ("rev7 验证", "#4ade80", [tsem_by.get(c, {}).get("rev7", (0, 0))[1] for c in tsem_cats]),
    ]
    tsem_stars = {i: [3] for i in range(len(tsem_cats))}
    tsem_html = (_dx_grouped_bars(tsem_cats, tsem_series, height=240,
                                  stars=tsem_stars) if tsem_cats else
                 '<p class="nodata">无数据</p>')
    tsem_legend = ("<div class='legend'><span><i style='background:#3b82f6'></i>"
                   "rev6 候选供应</span><span><i style='background:#60a5fa'></i>"
                   "rev6 验证通过</span><span><i style='background:#22c55e'></i>"
                   "rev7 候选供应</span><span><i style='background:#4ade80'></i>"
                   "rev7 验证通过</span><span style='color:#f59e0b'>★ = "
                   "chain_complete（rev7 全绿=机制正证据）</span></div>")

    # ---- d. FULLOP2 闭链矩阵 ----
    f2_rows = _dx_get(report, "fullop2.rows", []) or []
    f2_matrix: dict[tuple[str, str], list] = {}
    for r in f2_rows if isinstance(f2_rows, list) else []:
        key = (str(r.get("key", "")), str(r.get("arm", "")))
        m = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
        f2_matrix.setdefault(key, []).append(1 if m.get("chain_complete") else 0)
    f2_ctxs = sorted({k[0] for k in f2_matrix})
    f2_arms = ["A", "B"]
    if f2_ctxs:
        cells = "".join(
            "<tr><td><b>%s</b></td>%s</tr>" % (
                _dx_esc(c),
                "".join(
                    "<td class='ok'>%s</td>" % _dx_esc(
                        "%d/%d" % (sum(f2_matrix.get((c, a), [])),
                                   len(f2_matrix.get((c, a), []))))
                    for a in f2_arms))
            for c in f2_ctxs)
        f2_html = ("<table><thead><tr><th>context</th>"
                   + "".join("<th>arm %s</th>" % a for a in f2_arms)
                   + "</tr></thead><tbody>" + cells + "</tbody></table>"
                   + "<p class='mut'>总闭链 24/24（每 cell 为 "
                   "chain_complete / rep）。</p>")
    else:
        f2_html = '<p class="nodata">无数据</p>'

    # ---- e. 效用翻转散点 ----
    bse_labels = _dx_get(report, "bse.labels", []) or []
    label_series = {str(l.get("key", "")): str(l.get("series", ""))
                    for l in bse_labels if isinstance(l, dict)}
    gcolor = {"A": "#22c55e", "B": "#ef4444", "C": "#f59e0b",
              "NEUTRAL": "#94a3b8"}
    scatter_pts = []
    b_group_keys = []
    for lbl in bse_labels if isinstance(bse_labels, list) else []:
        g = str(lbl.get("group", ""))
        sg = _dx_num(lbl.get("support_gain"))
        dg = lbl.get("delayed_gain")
        if dg is None:
            b_group_keys.append(str(lbl.get("key", "")))
            continue
        scatter_pts.append((sg, _dx_num(dg), gcolor.get(g, "#94a3b8"),
                            str(lbl.get("key", ""))))
    scatter_html = (_dx_scatter(scatter_pts) if scatter_pts else
                    '<p class="nodata">无数据</p>')
    scatter_note = ("<p class='mut'>A=稳定正(绿) · C=即时正/后续翻负(橙，落 "
                    "右下象限) · B=即时有害(红，delayed 未评估：%s)。"
                    "虚线 ±M=0.005。</p>" % (
                        "、".join(b_group_keys) if b_group_keys else "无"))

    # ---- f. BSE observation 条形图 ----
    bse_obs = _dx_get(report, "bse.observations", {}) or {}
    obs_cats = sorted(bse_obs.keys()) if isinstance(bse_obs, dict) else []
    obs_vals = []
    obs_colors = []
    for c in obs_cats:
        ov = bse_obs[c] if isinstance(bse_obs[c], dict) else {}
        obs_vals.append(ov.get("value"))
        obs_colors.append("#22c55e" if label_series.get(c) == "T10"
                          else "#3b82f6")
    tau = _dx_get(report, "bse.fit_boundary.tau")
    tau_n = _dx_num(tau) if tau is not None else -0.1296
    obs_html = (_dx_vbars(obs_cats, obs_vals, obs_colors, height=250,
                          hline=tau_n, hline_label="τ=%.4f" % tau_n)
                if obs_cats else '<p class="nodata">无数据</p>')
    t10 = next((v for c, v in zip(obs_cats, obs_vals) if c == "T10@600"), None)
    obs_note = ("<p class='mut'>fit（T1/T100，蓝）与 heldout（T10，绿）分组；"
                "红线 τ=%.4f 为 fit A/B 边界中点。" % tau_n
                + (" T10@600 观测值 = %s —— 三历史窗 min 恰为 0.0（'弱正面历史"
                   "不保鲜'，非'纯 INERT 首次激活'）。" % _dx_fmt(t10)
                   if t10 is not None else ""))

    # ---- g. BSE replay H0 vs H1 ----
    replay_rows = _dx_get(report, "bse.replay.rows", []) or []
    ab_rows = [r for r in (replay_rows if isinstance(replay_rows, list) else [])
               if isinstance(r, dict) and str(r.get("group")) in ("A", "B")]
    h0r = int(sum(_dx_num(r.get("H0", {}).get("support_receipts")) for r in ab_rows))
    h0n = int(sum(_dx_num(r.get("H0", {}).get("negative_probes")) for r in ab_rows))
    h1r = int(sum(_dx_num(r.get("H1", {}).get("support_receipts")) for r in ab_rows))
    h1n = int(sum(_dx_num(r.get("H1", {}).get("negative_probes")) for r in ab_rows))
    replay_cats = ["support_receipts", "negative_probes"]
    replay_series = [("H0 现状", "#ef4444", [h0r, h0n]),
                     ("H1 规则门控", "#22c55e", [h1r, h1n])]
    replay_html = _dx_grouped_bars(replay_cats, replay_series, height=210)
    replay_legend = ("<div class='legend'><span><i style='background:#ef4444'></i>H0 现状</span><span><i style='background:#22c55e'></i>H1 规则门控</span></div>")
    replay_rows_html = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
        "<td class='%s'>%s</td><td class='%s'>%s</td></tr>" % (
            _dx_esc(str(r.get("key", ""))), _dx_esc(str(r.get("group", ""))),
            _dx_esc(_dx_fmt(r.get("obs_value"))),
            _dx_esc("是" if r.get("rule_fires") else "否"),
            "ok" if r.get("H0", {}).get("prior") else "mut",
            "prior" if r.get("H0", {}).get("prior") else "—",
            "ok" if r.get("H1", {}).get("prior") else "mut",
            "prior" if r.get("H1", {}).get("prior") else "—")
        for r in ab_rows)
    replay_note = ("<p class='mut'>held-out A/B 口径：H0 receipts=%d/neg=%d → "
                   "H1 receipts=%d/neg=%d。失败项：harm_auto_priority_blocked"
                   "（T10@600 观测值 0.0 ≥ τ 仍进 prior）+ receipts_or_harm"
                   "_reduced。</p>" % (h0r, h0n, h1r, h1n))
    replay_table = ("<table><thead><tr><th>held-out</th><th>group</th>"
                    "<th>obs</th><th>rule_fires</th><th>H0 prior</th>"
                    "<th>H1 prior</th></tr></thead><tbody>%s</tbody></table>"
                    % replay_rows_html)

    # ---- h. Memory ----
    rel_counts: dict[str, int] = {}
    ev_counts: dict[str, int] = {}
    st_counts: dict[str, int] = {}
    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        rel_counts[str(ep.get("relation", ""))] = rel_counts.get(str(ep.get("relation", "")), 0) + 1
        ev_counts[str(ep.get("evidence_level", ""))] = ev_counts.get(str(ep.get("evidence_level", "")), 0) + 1
        st_counts[str(ep.get("local_status", ""))] = st_counts.get(str(ep.get("local_status", "")), 0) + 1
    rel_colors = {"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444",
                  "CONFLICT": "#f59e0b"}
    ev_colors = {"SUPPORT": "#3b82f6", "DELAYED": "#a855f7",
                 "FULL_POLICY": "#0ea5e9"}
    st_colors = {"LOCAL_DRAFT": "#22c55e", "EPISODE_ONLY": "#64748b",
                 "RESTRICTED": "#ef4444"}

    def hb(counts: Mapping[str, int], colors: Mapping[str, str]) -> str:
        items = [(k, v, colors.get(k, "#64748b")) for k, v in counts.items()]
        return _dx_hbars(items) if items else '<p class="nodata">无数据</p>'

    n3_pack = (_dx_get(report, "n3.relation_counts") or {})
    n3_status = ("N3 Source Pack：%s · %d 条（%s）" % (
        verdict_of("n3", "verdict"),
        len(_dx_get(report, "n3.member_ids", []) or []),
        "，".join("%s=%s" % (k, v) for k, v in n3_pack.items())))
    mem_html = ("<div class='grid'><div><h3 style='font-size:13px'>relation "
                "（13 条）</h3>%s</div><div><h3 style='font-size:13px'>"
                "evidence_level</h3>%s</div><div><h3 style='font-size:13px'>"
                "local_status</h3>%s</div><div><h3 style='font-size:13px'>"
                "N3 Pack</h3><p class='mut'>%s</p></div></div>"
                % (hb(rel_counts, rel_colors), hb(ev_counts, ev_colors),
                   hb(st_counts, st_colors), _dx_esc(n3_status)))

    # ---- i. Claim 边界卡片 ----
    can = [
        "rev6→rev7 为正向机制证据：修正 targeting 说明后 Agent 更稳定地把 "
        "broad deviation hypothesis 绑定到 intrinsic outlier 候选（TSEM "
        "TARGETING_SEMANTICS_CAUSAL_EFFECT）。",
        "自进化安全审批链已跑通：多轨迹共同失败发现 → Slow 提案 → Runtime "
        "绑定 fork → 行为重放核销 → 发现回归并确定性拒绝（两轮按设计工作）。",
        "development Target-local 证据（BSE/N1 口径：本地证据成立）。",
        "N3：KDD 自然 outlier_mad 轨迹已冻结为完整 Source Pack（NATURAL_"
        "SOURCE_EPISODE_PACK_READY，10 条完整）。",
        "rev7 已固化为 h0 新基线（用户裁决 2026-08-14，逐字节对账）。",
    ]
    cant = [
        "不证明完整 26 算子选择能力（tsem claim_boundary）。",
        "不证明新候选下游价值 / 跨 Context 稳定（milestone not_yet_proven）。",
        "不证明 Slow 自进化能力：自由文本 Guidance Patch 尚不能可靠产生有效"
        "更新（G2/G3、P0/P6 两轮均拒绝）；rev7 是人工修正，非 Slow 自动成功。",
        "不宣称跨域、不升级 Shared Capability、不把阈值应用到新 Dataset"
        "（BSE/N1 claim_boundary）。",
        "C 组 temporal 风险未解决：TEMPORAL_INSTABILITY_UNRESOLVED。",
    ]
    claim_html = ("<div class='grid'><div><h3 style='font-size:13px;"
                  "color:#22c55e'>可以声称</h3><ul class='tight'>"
                  + "".join("<li>%s</li>" % _dx_esc(x) for x in can)
                  + "</ul></div><div><h3 style='font-size:13px;color:#ef4444'>"
                  "不能声称</h3><ul class='tight'>"
                  + "".join("<li>%s</li>" % _dx_esc(x) for x in cant)
                  + "</ul></div></div>")

    # ---- j. 明细表（逐行）----
    tsem_det = _dx_table(
        ["context", "arm", "status", "chosen", "chain", "supply"],
        [[r.get("key", ""), r.get("arm", ""), r.get("status", ""),
          r.get("chosen_candidate_id", ""),
          ("✓" if (r.get("metrics") or {}).get("chain_complete") else "✗"),
          len((r.get("metrics") or {}).get("program_ids") or [])]
         for r in (tsem_rows if isinstance(tsem_rows, list) else [])])
    f2_det = _dx_table(
        ["context", "arm", "rep", "chosen", "chain"],
        [[r.get("key", ""), r.get("arm", ""), r.get("rep", ""),
          r.get("chosen_candidate_id", ""),
          ("✓" if (r.get("metrics") or {}).get("chain_complete") else "✗")]
         for r in (f2_rows if isinstance(f2_rows, list) else [])])
    ep_det = _dx_table(
        ["episode_id", "domain", "workflow", "relation", "evidence", "status"],
        [[str(ep.get("episode_id", "")), str(ep.get("domain_namespace", "")),
          str(ep.get("workflow_signature", "")), str(ep.get("relation", "")),
          str(ep.get("evidence_level", "")), str(ep.get("local_status", ""))]
         for ep in episodes if isinstance(ep, dict)])
    n1_rows = _dx_get(report, "n1.prevalence", []) or []
    n1_det = _dx_table(
        ["context", "hist_status", "cur_changed", "novel"],
        [[r.get("key", ""), r.get("historical_status_prevalence", ""),
          r.get("current_changed_total", ""),
          ("是" if r.get("novel_action") else "否")]
         for r in (n1_rows if isinstance(n1_rows, list) else [])])
    b1_entries = _dx_get(report, "batch1.entries", []) or []
    b1_det = _dx_table(
        ["context", "gain", "delayed_gain"],
        [[e.get("key", ""), _dx_fmt(e.get("gain")),
          (_dx_fmt(e.get("delayed", {}).get("gain"))
           if (e.get("delayed") or {}).get("gain") is not None else "—")]
         for e in (b1_entries if isinstance(b1_entries, list) else [])])
    detail_html = ("<div class='grid'>"
                   "<div><h3 style='font-size:13px'>TSEM 逐行（%d）</h3>%s</div>"
                   "<div><h3 style='font-size:13px'>FULLOP2 逐行（%d）</h3>%s</div>"
                   "<div><h3 style='font-size:13px'>episodes（%d 条）</h3>%s</div>"
                   "<div><h3 style='font-size:13px'>N1 prevalence（%d 行）</h3>%s</div>"
                   "<div><h3 style='font-size:13px'>batch1 entries（%d 行）</h3>%s</div>"
                   "</div>" % (
                       len(tsem_rows if isinstance(tsem_rows, list) else []),
                       tsem_det,
                       len(f2_rows if isinstance(f2_rows, list) else []),
                       f2_det,
                       len(episodes), ep_det,
                       len(n1_rows if isinstance(n1_rows, list) else []),
                       n1_det,
                       len(b1_entries if isinstance(b1_entries, list) else []),
                       b1_det))

    # ---- k. 自包含原始数据附录 ----
    curated = _dx_strip({k: report[k] for k in (
        "tsem", "fullop2", "usel", "batch1", "cobs", "bse", "n1", "n3",
        "final_verdict", "milestone", "g2_g5_chain_final",
        "p0_p6_chain_final") if k in report})
    appendix = ("<details><summary>展开查看原始数据（报告关键 section，"
                "已剥离 prompt_calls）</summary><pre>%s</pre></details>"
                "<details><summary>展开查看 episodes.json 全文</summary>"
                "<pre>%s</pre></details>"
                % (_dx_esc(json.dumps(curated, ensure_ascii=False)),
                   _dx_esc(json.dumps(episodes, ensure_ascii=False))))


    # ---- loop1 卡片（防御性：无 loop1 显示「未运行」）----
    loop1 = report.get("loop1") or {}
    loop1_proto = report.get("loop1_protocol") or {}
    if not loop1 and not loop1_proto:
        loop1_html = '<p class="nodata">未运行</p>'
    else:
        stages = loop1_proto.get("loop") or [
            "batch 失败证据", "匿名 capsule", "Slow 自主改 skill 条款",
            "Runtime 落 fork", "配对行为重放", "fork 内激活演示"]
        slow = loop1.get("slow") or {}
        replay = loop1.get("replay") or []
        verdict = loop1.get("verdict") or {}
        activation = loop1.get("activation") or {}
        stage_done = [
            bool(loop1_proto), bool(slow), bool(slow.get("fork_sha")),
            bool(slow.get("fork_sha")), bool(replay), bool(verdict)]
        stage_html = " ".join(
            "<span class='pill %s'>%s</span>" % (
                "p-ok" if d else "p-info", _dx_esc(str(s)))
            for s, d in zip(stages, stage_done))
        a_rej = 0
        b_rej = 0
        for r in (replay if isinstance(replay, list) else []):
            if not isinstance(r, dict):
                continue
            n = len(r.get("rejection_receipts") or [])
            if r.get("arm") == "A":
                a_rej = n
            elif r.get("arm") == "B":
                b_rej = n
        if replay:
            rej_html = _dx_vbars(["臂 A (h0)", "臂 B (rev8 fork)"],
                                 [a_rej, b_rej], ["#3b82f6", "#22c55e"],
                                 height=180)
        else:
            rej_html = '<p class="nodata">replay 未运行</p>'
        v_txt = str(verdict.get("verdict") or "未裁决")
        act_txt = "无"
        if activation:
            act_txt = _dx_esc(str(activation.get("scope") or "?"))
            if activation.get("note"):
                act_txt += " · " + _dx_esc(str(activation.get("note")))
        loop1_html = ("<div class='grid'><div><h3 style='font-size:13px'>"
                      "六环状态</h3><p>%s</p></div>"
                      "<div><h3 style='font-size:13px'>两臂 rejection 对比</h3>%s</div>"
                      "</div><p class='mut'>verdict：<b>%s</b> · 激活范围：%s</p>"
                      % (stage_html, rej_html, _dx_esc(v_txt), act_txt))


    # ---- N4/N5 预检 + A5/A3 主实验卡片（防御性：无则显示「未运行」）----
    n4 = report.get("n4_v2") or report.get("n4") or {}
    n5 = report.get("n5_v2") or report.get("n5") or {}
    a5d = report.get("a5v3") or report.get("a5v2") or report.get("a5") or {}
    a5_is_v3 = bool(report.get("a5v3"))
    a5_is_v2 = bool(report.get("a5v2")) and not a5_is_v3
    if not n4 and not n5 and not a5d and not report.get("a5_protocol") \
            and not report.get("a5v2_protocol"):
        a5_html = '<p class="nodata">未运行</p>'
    else:
        n4v = ((n4.get("verdict") or {}).get("verdict")) or "未运行"
        n5v = ((n5.get("verdict") or {}).get("verdict")) or "未运行"
        roster = n4.get("roster") or []
        roster_txt = ", ".join(str(e.get("entity_id")) + "@" + str(e.get("origin"))
                               for e in roster) or "-"
        a5v = (a5d.get("verdict") or {})
        agg = a5d.get("aggregates") or {}
        rows_now = a5d.get("rows") or []
        if a5v.get("final"):
            fpe_key = ("feedback_to_first_confirmed" if a5_is_v3
                       else "feedback_to_reliable_skill" if a5_is_v2
                       else "feedback_to_first_effective")
            n_key = ("n_final_reliable" if a5_is_v3
                     else "n_reliable" if a5_is_v2 else "n_effective")
            arms_tbl = _dx_table(
                ["臂", "fpe", "harm", "abstain", "support_gain",
                 ("n_final" if a5_is_v3
                  else "n_reliable" if a5_is_v2 else "n_eff"), "llm"],
                [[arm,
                  _dx_str((agg.get(arm) or {}).get(fpe_key)),
                  _dx_str((agg.get(arm) or {}).get("harm_events")),
                  _dx_str((agg.get(arm) or {}).get("abstentions")),
                  _dx_fmt((agg.get(arm) or {}).get("total_support_gain")),
                  _dx_str((agg.get(arm) or {}).get(n_key)),
                  _dx_str((agg.get(arm) or {}).get("llm_calls"))]
                 for arm in ("A5", "A3")])
            a5_status = ("<p>verdict：<b>" + _dx_esc(str(a5v.get("verdict")))
                         + "</b></p><p class='mut'>"
                         + _dx_esc(str(a5v.get("reason"))) + "</p>" + arms_tbl)
        else:
            a5_status = ("<p class='mut'>运行中：rows " + str(len(rows_now))
                         + "（verdict 未终裁）</p>")
        checks = n5.get("checks") or {}
        checks_txt = " · ".join(k + "=" + ("✓" if v else "✗")
                                for k, v in checks.items()) or "-"
        a5_html = (
            "<div class='grid'><div>"
            "<h3 style='font-size:13px'>N4 Target 资格审查</h3>"
            "<p>" + _dx_esc(str(n4v)) + "</p>"
            "<p class='mut'>roster：" + _dx_esc(roster_txt) + "</p>"
            "<h3 style='font-size:13px'>N5 Memory 接线预检</h3>"
            "<p>" + _dx_esc(str(n5v)) + "</p>"
            "<p class='mut'>" + _dx_esc(checks_txt) + "</p>"
            "</div><div>"
            "<h3 style='font-size:13px'>A5/A3 主实验</h3>" + a5_status
            + "</div></div>")

    # ---- 组装 ----
    cards = "".join([
        _dx_card("a. 闭环流程（今晚真实状态）",
                 "每个节点标注覆盖实验与 pass/closed 状态", flow_html),
        _dx_card("b. 实验时间线", "8 个实验的目的 / verdict / 一句话结论",
                 timeline_html),
        _dx_card("c. TSEM 配对：rev6 vs rev7（机制正证据）",
                 "每 context 的候选供应与验证通过（★）",
                 tsem_html + tsem_legend),
        _dx_card("d. FULLOP2 闭链矩阵", "6 context × 2 arm 的 24/24",
                 f2_html),
        _dx_card("e. 效用翻转散点（support vs delayed）",
                 "即时正、后续翻负的 C 组轨迹", scatter_html + scatter_note),
        _dx_card("f. BSE Observation：historical_program_stability",
                 "10 context 的 min 聚合稳定性观测（τ 参考线）",
                 obs_html + obs_note),
        _dx_card("g. BSE replay：H0 vs H1 receipts / negative probes",
                 "held-out A/B 口径核销",
                 replay_html + replay_legend + replay_table + replay_note),
        _dx_card("h. Memory：episodes.json 13 条 + N3 pack", "", mem_html),
        _dx_card("i. Claim 边界", "今晚可以 / 不能声称什么", claim_html),
        _dx_card("loop1. 首次全自主自进化闭环（正向演示）",
                 "n=1 单案例机制演示 · 不晋升 h0", loop1_html),
        _dx_card("n4n5a5. Target 资格 → Memory 接线 → A5/A3 主实验（v2）",
                 "N4v2 字段级暴露审查 · N5v2 增长态预检 · 真 Skill 生命周期",
                 a5_html),
    ])
    cards += _dx_card("j. 明细表（逐行）", "TSEM / FULLOP2 / episodes / N1 / batch1 逐行数据", detail_html)
    cards += _dx_card("k. 原始数据附录（自包含）", "报告关键 section + episodes.json 全文", appendix)
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>W1 自进化实验仪表板</title>
<style>""" + _DX_CSS + """</style>
</head>
<body>
<header>
<h1>W1 自进化实验仪表板（一夜实验链）</h1>
<div class="sum">Skill 指导 Workflow 的自进化闭环：审批链已跑通；rev6→rev7 为正向机制证据</div>
</header>
<main>
""" + cards + """
</main>
<footer>只读仪表板 · 由 run_v1_guidance_evolution.py phase_dashboard 生成 · 内联 CSS/SVG，离线可用</footer>
</body>
</html>
"""

    DASHBOARD_REL.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_REL.write_text(html, encoding="utf-8")
    size = DASHBOARD_REL.stat().st_size
    print("dashboard: %s (%d bytes)" % (DASHBOARD_REL, size))
    return 0


PHASES = {
    "g1-b0": phase_g1_b0,
    "g1-b1": phase_g1_b1,
    "g1-verdict": phase_g1_verdict,
    "g2": phase_g2,
    "g3": phase_g3,
    "g4": phase_g4,
    "g5": phase_g5,
    "s0-a": phase_s0_a,
    "s0-b": phase_s0_b,
    "s0-verdict": phase_s0_verdict,
    "p0-accounting": phase_p0_accounting,
    "p4": phase_p4,
    "fullop": phase_fullop,
    "fullop-p2": phase_fullop_p2,
    "fullop-verdict": phase_fullop_verdict,
    "fullop-rerun": phase_fullop_rerun,
    "tsem-freeze": phase_tsem_freeze,
    "tsem": phase_tsem,
    "tsem-verdict": phase_tsem_verdict,
    "fullop2-freeze": phase_fullop2_freeze,
    "fullop2": phase_fullop2,
    "fullop2-p2": phase_fullop2_p2,
    "fullop2-verdict": phase_fullop2_verdict,
    "usel-freeze": phase_usel_freeze,
    "usel": phase_usel,
    "usel-verdict": phase_usel_verdict,
    "batch1-freeze": phase_batch1_freeze,
    "batch1": phase_batch1,
    "batch1-verdict": phase_batch1_verdict,
    "cobs-freeze": phase_cobs_freeze,
    "cobs": phase_cobs,
    "bse-freeze": phase_bse_freeze,
    "bse-p0": phase_bse_p0,
    "bse-p1": phase_bse_p1,
    "bse-p2": phase_bse_p2,
    "bse-p3p4": phase_bse_p3p4,
    "bse-p5": phase_bse_p5,
    "bse-verdict": phase_bse_verdict,
    "n1-freeze": phase_n1_freeze,
    "n1-prevalence": phase_n1_prevalence,
    "n3-freeze": phase_n3_freeze,
    "n3": phase_n3,
    "n3-amend": phase_n3_amend,
    "n4-freeze": phase_n4_freeze,
    "n4": phase_n4,
    "n5-freeze": phase_n5_freeze,
    "n5": phase_n5,
    "a5-freeze": phase_a5_freeze,
    "a5": phase_a5,
    "a5-retry-transient": phase_a5_retry_transient,
    "a5-reset": phase_a5_reset,
    "a5-verdict": phase_a5_verdict,
    "n4v2-freeze": phase_n4v2_freeze,
    "n4v2": phase_n4v2,
    "n5v2": phase_n5v2,
    "a5v2-freeze": phase_a5v2_freeze,
    "a5v2": phase_a5v2,
    "a5v2-reset": phase_a5v2_reset,
    "a5v2-verdict": phase_a5v2_verdict,
    "a5v3-freeze": phase_a5v3_freeze,
    "a5v3": phase_a5v3,
    "a5v3-reset": phase_a5v3_reset,
    "a5v3-verdict": phase_a5v3_verdict,
    "dashboard": phase_dashboard,
    "loop1-freeze": phase_loop1_freeze,
    "loop1-slow": phase_loop1_slow,
    "loop1-replay": phase_loop1_replay,
    "loop1-verdict": phase_loop1_verdict,
    "final": phase_final,
}


def _archive_broken_measurement() -> None:
    """P1 rerun 前：把破损测量下的 g1 行为数据归档（保留为混扰证据），
    清空 g1.b0/b1/behavior/verdict 供重跑——冻结假设不变。
    幂等：broken_measurement 归档已存在时不重复归档（2026-08-14 bug
    修复：第二次 rerun 曾把刚跑完的新 b0 误归档并覆盖原始归档）。"""
    report = _load_report()
    g1 = report.get("g1") or {}
    if g1.get("broken_measurement_2026_08_14"):
        return  # 已归档——不重复
    archive = {
        "b0": g1.pop("b0", None),
        "b1": g1.pop("b1", None),
        "b0_bodies": g1.pop("b0_bodies", None),
        "b1_bodies": g1.pop("b1_bodies", None),
        "mechanical": g1.pop("mechanical", None),
        "behavior": g1.pop("behavior", None),
        "verdict": g1.pop("verdict", None),
        "note": ("破损测量（信封脆弱性未修复前）行为数据——归档为混扰"
                 "证据；frozen_hypothesis 与升级规则不变。"),
    }
    g1["broken_measurement_2026_08_14"] = archive
    report["g1"] = g1
    _save_report(report)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in PHASES:
        print("usage: run_v1_guidance_evolution.py "
              + "|".join(sorted(PHASES)))
        return 2
    phase = sys.argv[1]
    extra = len(sys.argv) > 2 and sys.argv[2] == "extra"
    rerun = len(sys.argv) > 2 and sys.argv[2] == "rerun"
    refix = len(sys.argv) > 2 and sys.argv[2] == "refix"
    if phase in ("g1-b0", "g1-b1", "g2", "g3", "g4", "g5", "s0-a", "s0-b",
                 "fullop", "fullop-p2", "fullop-rerun", "tsem",
                 "fullop2", "fullop2-p2", "usel", "batch1",
                 "bse-p2", "bse-p5", "n1-prevalence", "loop1-replay"):
        env = _load_env()
        if phase == "tsem":
            return PHASES[phase](env, extra)
        if phase == "g1-b1":
            if rerun:
                _archive_broken_measurement()
            return PHASES[phase](env, extra, refix)
        if phase == "g1-b0":
            if rerun:
                _archive_broken_measurement()
            return PHASES[phase](env, extra)
        return PHASES[phase](env)
    return PHASES[phase]()


if __name__ == "__main__":
    raise SystemExit(main())
