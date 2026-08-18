"""GROUP_WITNESS_REAL_SLOW_DEV（Wave 2，2026-08-13：受控 Witness 验证——
真实 Slow Agent 消费组级证据——零新 Claim——已暴露窗口 replay）。

目的不是建立自然收益，而是验证 Agent 是否**因果消费**组级证据：

  - 使用已暴露数据（T117 winsorize NEGATIVE @888/@984）；
  - 使用真实 Slow Agent（TTHASlowAgent——真实 LLM——非 ReplayAgent）；
  - 不预选答案：白名单含两个 verifier 合法替代（outlier_mad/hampel）
    ——正确 Patch 不被 Runner 过滤；
  - Agent 自主选择 Typed Patch（whitelist patch_id）/ Abstain——
    Runtime 核销（组内 replay 全 ≥M → holdout 不劣 → pending）；
  - 证据：Capsule（view 对齐的 per-view 响应）+ replacement headroom
    （hampel 共同正向；outlier_mad 非共同——@984 负）+ TaskContext +
    Operator contracts（Wave 1 主链）。

verdict（预注册）：
  GROUP_TYPED_PATCH_MECHANISM_PASS : 真实 Agent 提出白名单 Typed Patch
    （hampel——与组级证据一致）且链到达核销阶段（pending 或如实拒绝）
  GROUP_DIAGNOSTIC_REQUEST_PASS   : Agent 请求诊断（当前 schema 无该
    通道——记录为 no_proposal 原因，仅当原因明确为诊断请求时判此）
  EVIDENCE_GROUNDED_ABSTAIN       : Agent 基于证据主动 abstain（no_
    proposal 原因声明证据不足）
  SLOW_AGENT_EVIDENCE_USE_FAILURE : 选择与证据矛盾（outlier_mad）/
    提不出白名单 Patch / 无证据声明
  PROTOCOL_FAILURE                : 装置/预算故障

API 策略：每 family 一次正常 Slow proposal（schema retry 计入——
validation_retries=1）；不多采样挑答案；模型/温度/Prompt/调用次数写入
主结果。

用法：
  python evaluation/functional/run_v1_group_witness_real_slow_dev.py
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
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import signed_radius as resolver  # noqa: E402
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
    find_common_headroom,
    group_first_faults,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
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
# v3：v1 暴露 typed-patch 绑定规则未注入（两次 no patch_id）；v2 暴露
# surface 实例化缺口（Agent 选对 hampel 但 manifest 内部 skill_id/surface
# 不一致 → EditShapeError）。两次接线修复后重跑（v1/v2 保留）。
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_group_witness_real_slow_report_v3.json"

# 证据锚定的正确替代（hampel——组内共同正向）与对照（outlier_mad——
# @984 负——非共同）——Runner 不过滤白名单，只用于结果分类。
EVIDENCE_PATCH = "patch-replace-winsorize-with-hampel_filter"
CONTRAST_PATCH = "patch-replace-winsorize-with-outlier_mad"


def _load_series(root: Path, uid: str) -> np.ndarray:
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    return np.asarray(values[names.index(uid)], dtype=np.float64)


def _executor(root: Path) -> tuple[ScopeExecutor, dict]:
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2"
                         / "w1_kdd2018_frozen_cohort_p41.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in rows]
    values = {str(r["series_name"]): _load_series(root, str(r["series_name"]))
              for r in rows}
    return ScopeExecutor(roster, values, _config(), evaluate_fn=_evaluate_kdd), \
        values


def _view_keys_for(executor: ScopeExecutor) -> list[str]:
    return [str(row["series_uid"]) for row in executor.roster
            if str(row["role"]) != "train"]


def _rebuild_failure_episode(root: Path, uid: str, origin: int,
                             executor: ScopeExecutor) -> Any:
    steps = (("winsorize", dict(wiring.contract_params("winsorize",
                                                      PERIOD))),)
    rr = executor.evaluate(tuple(steps), origin)
    gain = float(rr.gain) if rr.gain is not None else None
    assert gain is not None and gain < -M, f"{uid}@{origin} not failure"
    ctx = dict(resolver.window_context({uid: _load_series(root, uid)},
                                       origin, PERIOD))
    return build_episode(
        episode_id=f"kdd_cup_2018_target_winsorize_w2_{uid}_{origin}",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace="kdd_cup_2018",
        context_summary={
            "local_pattern": {"support_gain": gain, **ctx},
            "delayed_pattern": {},
            "program_geometry": {"scope": "training_rows",
                                 "program_steps": [
                                     {"op": "winsorize", "params": {}}]},
            "per_view_gain": list(getattr(rr, "per_view_gain", []) or []),
            "support_origin": origin,
        },
        workflow_signature=workflow_signature_of(
            [{"op": "winsorize", "params": {}}]),
        support_response={"gain": gain, "accepted": False},
        delayed_response={"evaluated": False, "gain": None},
        relation="NEGATIVE", evidence_level="SUPPORT",
        local_status="EPISODE_ONLY", evidence_refs=["group_witness_w2"])


def _group_card(group: Mapping[str, Any],
                capsule: Mapping[str, Any],
                headroom: Mapping[str, Any]) -> dict[str, object]:
    """组 Card：白名单 = 两个 verifier 合法替代（不预选）；facts =
    capsule + headroom（Agent 的证据）；instruction = 组级任务语义。"""
    options = []
    for alt_op in ("outlier_mad", "hampel_filter"):
        options.append({
            "patch_id": (EVIDENCE_PATCH if alt_op == "hampel_filter"
                         else CONTRAST_PATCH),
            "program_steps": [{"op": alt_op,
                               "params": dict(wiring.contract_params(
                                   alt_op, PERIOD))}]})
    return {
        "pattern_id": "group-winsorize-neg",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "context": {},
        "workflow": {"steps": [{"op": "winsorize", "params": {}}]},
        "typed_patch_options": options,
        "facts": {
            "contrast_capsule": dict(capsule),
            "replacement_headroom": dict(headroom),
        },
        "instruction": (
            "A single-step workflow (winsorize) produced material negative "
            "Support outcomes at two adaptation decision points on the same "
            "target series, recorded as a repeated first-fault group. The "
            "contrast capsule holds the aligned per-view (per-evaluation-"
            "series) Support gains at both decision points and the "
            "replacement headroom of two verifier-legal alternatives. "
            "Choose exactly one typed patch from typed_patch_options whose "
            "frozen program is the best common replacement for the group "
            "evidence (positive on all in-group decision points; prefer "
            "common positive headroom). Propose exactly one edit: ADD one "
            "new capability skill (patch_id from the whitelist). You do not "
            "approve your own edit — a deterministic group replay and "
            "holdout will verify it. If the evidence is insufficient to "
            "choose, declare no_proposal — abstaining is valid."
        ),
    }


def main() -> int:
    root = PROJECT_ROOT
    ex, vals = _executor(root)
    ep888 = _rebuild_failure_episode(root, "T117", 888, ex)
    ep984 = _rebuild_failure_episode(root, "T117", 984, ex)
    eps = [ep888, ep984]
    groups = group_first_faults(eps, min_group=2)
    if not groups:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no repeated first-fault group"},
                         indent=1))
        return 0
    group = groups[0]
    view_keys = {ep888.episode_id: _view_keys_for(ex),
                 ep984.episode_id: _view_keys_for(ex)}
    capsule = build_contrast_capsule(group, all_episodes=eps,
                                     view_keys=view_keys)

    def _steps_of(op: str):
        return ((op, dict(wiring.contract_params(op, PERIOD))),)

    headroom = find_common_headroom(group, ex.evaluate,
                                    ("outlier_mad", "hampel_filter"),
                                    _steps_of)

    # ---- 真实 Slow Agent（真实 LLM——非 Replay）----
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no api key"}, indent=1))
        return 0
    import openai  # noqa: PLC0415
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120))
    backend = AgictoChatCompletionsBackend(client=counter,
                                           base_url=smoke.BASE_URL)
    series0 = _load_series(root, "T117")
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(series0[:888], task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL)
    slow = TTHASlowAgent(core)

    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    store = SnapshotStore(root / ".w2_store")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    method = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(
            backend, LocalPublicToolGateway(series0[:888],
                                            task_kind="forecast"),
            model=smoke.MODEL, base_url=smoke.BASE_URL)),
        h0, tuple(eps))

    request = _request(series0, vals, 888)
    task_ctx = forecast_task_context_v1(
        task_spec=request.task_spec,
        deployment_constraints=deployment_constraints_v1())
    contracts = tuple(public_operator_contract(op)
                      for op in ("winsorize", "outlier_mad",
                                 "hampel_filter"))

    def _holdout(steps, _mode):
        return ex.evaluate(tuple(steps), 600)

    ev = method.handle_group_feedback(group, capsule, slow_agent=slow, controller=controller, store=store, card_builder=lambda g, cap: _group_card(g, cap, headroom), evaluator_group=lambda s, e: ex.evaluate(tuple(s), int((getattr(e, 'context_summary', {}) or {}).get('support_origin') or 0)), holdout_evaluator=_holdout, fast_features=dict(extract_public_features(series0[:888], task_kind='forecast')), allowed_operator_contracts=contracts, task_context=task_ctx, surface_catalog=controlled_add_only_group_catalog(), route_decision=controlled_add_only_group_decision(case_id='dev-run_v1_group_witness_real_slow_dev-275'))

    # delayed 复核（已暴露窗口 @1032 replay——链完整记录；不改变 witness
    # 判定）
    dev = None
    if ev.get("stage") == "pending":
        dev = method.handle_feedback_delayed(
            lambda s, _m: ex.evaluate(tuple(s), 1032),
            episode_id=ev.get("episode_id"))

    # ---- 判定（预注册）----
    stage = ev.get("stage")
    pid = ev.get("patch_id")
    reason = ev.get("no_proposal_reason") or slow.last_no_proposal_reason
    if stage in ("pending", "group_replay_rejected", "holdout_rejected",
                 "manifest_proposed", "applied", "no_frozen_program",
                 "support_rejected"):
        if pid == EVIDENCE_PATCH:
            verdict = "GROUP_TYPED_PATCH_MECHANISM_PASS"
        elif pid == CONTRAST_PATCH:
            verdict = "SLOW_AGENT_EVIDENCE_USE_FAILURE"
        else:
            verdict = "SLOW_AGENT_EVIDENCE_USE_FAILURE"
    elif stage == "no_manifest":
        r = str(reason or "").lower()
        if any(k in r for k in ("insufficient", "abstain", "evidence",
                                "unable", "cannot")):
            verdict = "EVIDENCE_GROUNDED_ABSTAIN"
        else:
            verdict = "SLOW_AGENT_EVIDENCE_USE_FAILURE"
    elif stage in ("manifest_preflight_failed",
                   "typed_patch_contract_failed"):
        verdict = "SLOW_AGENT_EVIDENCE_USE_FAILURE"
    else:
        verdict = "PROTOCOL_FAILURE"
    if counter.calls > 2:
        verdict = "PROTOCOL_FAILURE"
        reason = (f"{reason} + LLM_CALL_BUDGET_EXCEEDED"
                  if reason else "LLM_CALL_BUDGET_EXCEEDED")

    report = {
        "experiment_id": "v1-group-witness-real-slow",
        "note": "Wave 2 v3：受控 Witness 验证（真实 Slow Agent 消费组级"
                "证据——零新 Claim——已暴露窗口 replay；delayed @1032 为"
                "时间边界窗口 replay）。v1：typed-patch 绑定规则未注入"
                "（两次无 patch_id）。v2：Agent 选对 hampel 但 surface/"
                "entry ID 不一致（EditShapeError）。两次接线修复后重跑。",
        "group": {"workflow": group["workflow"], "sign": group["sign"],
                  "episodes": [e.episode_id for e in group["episodes"]]},
        "capsule": capsule,
        "headroom": headroom,
        "group_feedback_event": ev,
        "delayed_event": dev,
        "slow_agent_choice": pid,
        "no_proposal_reason": reason,
        "llm": {"model": smoke.MODEL, "base_url": smoke.BASE_URL,
                "calls": counter.calls, "schema": "slow_edit_v1",
                "temperature": "default"},
        "verdict": verdict,
    }
    print(f"== group_feedback: {json.dumps(ev, ensure_ascii=False)}")
    print("== delayed: "
          + json.dumps(dev, ensure_ascii=False, default=str))
    print(f"== slow choice: {pid} calls={counter.calls} "
          f"no_proposal_reason={reason}")
    print(f"== verdict: {verdict}")
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
