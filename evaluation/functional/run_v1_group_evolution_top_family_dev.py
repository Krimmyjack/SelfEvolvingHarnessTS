"""GROUP_EVOLUTION_TOP_FAMILY_DEV（Wave 4a，2026-08-13：真实 Group Slow
Evolution on census top family——development exposure——零新 Claim）。

装置（全部预注册）：
  family = census 报告 development_families[0]（确定性排序首项）——
    winsorize NEGATIVE，4 独立 series（T1/T10/T100/T101），6 个失败窗口
  episodes 重建 = 6 个已暴露窗口的 winsorize replay（含 per_view——
    零新 outcome）
  view_keys = 各 series executor roster 的 eval 序（同 EVAL_SERIES——
    8 view 对齐）
  对比案例 = census rounds 中的 winsorize POSITIVE 窗口（8 个——从
    报告读数，不重评）
  headroom = census 报告 reuse（outlier_mad/hampel 均非共同正向）
  Card 白名单 = [outlier_mad, hampel]（两个 verifier 合法替代——不预选；
    证据显示两者在组内均无 common positive headroom）
  instruction = 组级证据消费 + "共同正向 headroom 是组内 replay 门的
    判据；若无替代在全部组内窗口共同正向，请用 no_proposal 信封
    （reason_code=insufficient_public_evidence）弃权——弃权合法"
  Slow = 真实 LLM（一次调用；schema retry 计入 validation_retries=1）
  组内 replay 门 = 全部 ≥M（series 级 executor 解析——origin 碰撞安全）
  补集检查 = 8 个 winsorize POSITIVE 窗口上 patch ≥ −M（runner 侧——
    pending 后执行——"补集不劣"批增益报告）

verdict（预注册）：
  EVIDENCE_GROUNDED_ABSTAIN        : no_manifest 且原因=证据不足/风险
    ——与证据一致（本 family 无共同正向替代）→ family 结论
    NO_COMMON_PROGRAM_HEADROOM
  SLOW_AGENT_EVIDENCE_USE_FAILURE  : 选择白名单 Patch（两者都与
    "共同正向"证据矛盾——组内 replay 门将如实拒绝）/契约失败 → 同上
  GROUP_TYPED_PATCH_MECHANISM_PASS : 选 Patch 且到达核销阶段（本
    family 数据下预期不可达——若达则如实记录）
  TEMPORALLY_UNSTABLE_REPAIR       : pending 且补集/delayed 拒绝
  PROTOCOL_FAILURE

批增益报告（计划要求）：目标组 gain 列表 / positive fraction / tail
harm / 补集 gain / Support receipts 计数 / 稳定性 / unpredicted
regression。

用法：
  python evaluation/functional/run_v1_group_evolution_top_family_dev.py
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
CENSUS_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_batch_census_dev_report.json"
# r2：abstain 通道修复（agent_core 教学 no_proposal 信封 + 重试反馈附
# 模板——checker/reviewer 裁决 REPAIR_CURRENT_WAVE）后对同一 family
# 恰好一次真实重跑（协议修复重试——非重跑挑答案；r1 与 corrected 保留）
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_group_evolution_top_family_report_r2.json"

OPS = ("winsorize", "outlier_mad", "hampel_filter")
PATCH_OUTLIER = "patch-replace-winsorize-with-outlier_mad"
PATCH_HAMPEL = "patch-replace-winsorize-with-hampel_filter"


def _load_series(root: Path, uid: str) -> np.ndarray:
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    return np.asarray(values[names.index(uid)], dtype=np.float64)


def _series_executor(root: Path, series: str,
                     eval_series: Sequence[str]) -> ScopeExecutor:
    roster = ([{"series_uid": series, "role": "train"}]
              + [{"series_uid": s, "role": "eval"} for s in eval_series])
    values = {s: _load_series(root, s) for s in (series,) + tuple(eval_series)}
    return ScopeExecutor(roster, values, _config(),
                         evaluate_fn=_evaluate_kdd)


def _rebuild_failure_episode(root: Path, series: str, origin: int,
                             executor: ScopeExecutor,
                             expect_gain: float | None) -> Any:
    steps = (("winsorize", dict(wiring.contract_params("winsorize",
                                                      PERIOD))),)
    rr = executor.evaluate(tuple(steps), origin)
    gain = float(rr.gain) if rr.gain is not None else None
    assert gain is not None and gain < -M, f"{series}@{origin} not failure"
    if expect_gain is not None:
        # census 读数一致性复核（同装置同窗口——数值应逐位一致）
        assert abs(gain - expect_gain) < 1e-9, \
            f"{series}@{origin} drift: {gain} vs census {expect_gain}"
    ctx = dict(resolver.window_context({series: _load_series(root, series)},
                                       origin, PERIOD))
    return build_episode(
        episode_id=f"kdd2018_dev_{series}_target_winsorize_w4_{series}"
                   f"_{origin}",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace=f"kdd2018_dev_{series}",
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
        local_status="EPISODE_ONLY", evidence_refs=["group_evolution_w4"])


def _positive_contrast_episode(series: str, origin: int,
                               gain: float) -> Any:
    return build_episode(
        episode_id=f"kdd2018_dev_{series}_target_winsorize_pos_{series}"
                   f"_{origin}",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace=f"kdd2018_dev_{series}",
        context_summary={
            "local_pattern": {"support_gain": gain},
            "delayed_pattern": {},
            "program_geometry": {"scope": "training_rows",
                                 "program_steps": [
                                     {"op": "winsorize", "params": {}}]},
            "per_view_gain": [],
            "support_origin": origin,
        },
        workflow_signature="winsorize",
        support_response={"gain": gain, "accepted": True},
        delayed_response={"evaluated": False, "gain": None},
        relation="POSITIVE", evidence_level="SUPPORT",
        local_status="LOCAL_DRAFT", evidence_refs=["group_evolution_w4"])


def _group_card(group: Mapping[str, Any],
                capsule: Mapping[str, Any],
                headroom: Mapping[str, Any]) -> dict[str, object]:
    options = []
    for alt_op, pid in (("outlier_mad", PATCH_OUTLIER),
                        ("hampel_filter", PATCH_HAMPEL)):
        options.append({
            "patch_id": pid,
            "program_steps": [{"op": alt_op,
                               "params": dict(wiring.contract_params(
                                   alt_op, PERIOD))}]})
    return {
        "pattern_id": "group-winsorize-neg-multiseries",
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
            "Support outcomes at six adaptation decision points across four "
            "independent target series, forming a repeated first-fault "
            "group. The contrast capsule holds aligned per-view Support "
            "evidence per decision point and matched positive contrast "
            "cases; replacement_headroom holds the Support gain of each "
            "whitelist alternative on every in-group window. The group "
            "replay gate accepts a patch only if its Support gain is "
            "positive (>= 0.005) on ALL in-group windows. If no whitelist "
            "alternative has common positive headroom on all in-group "
            "windows, declare no_proposal with reason_code "
            "insufficient_public_evidence — abstaining is valid and "
            "preferred over a patch that fails the gate. Otherwise choose "
            "exactly one typed patch from typed_patch_options. You do not "
            "approve your own edit — the deterministic group replay and "
            "complement check verify it.",
        ),
    }


def main() -> int:
    root = PROJECT_ROOT
    census = json.loads(CENSUS_REL.read_text(encoding="utf-8"))
    families = census.get("development_families") or []
    if not families:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no census family"}, indent=1))
        return 0
    family = families[0]
    assert family["n_series"] >= 2, "top family must be cross-series"
    pre = census["pre_registered"]
    eval_series = tuple(pre["eval_series"])

    # per-series executors（episode 级解析——origin 碰撞安全）
    executors = {s: _series_executor(root, s, eval_series)
                 for s in family["independent_series"]}
    by_key = {(e["series"], e["origin"]): e["gain"]
              for e in family["episodes"]}
    eps = []
    for e in family["episodes"]:
        eps.append(_rebuild_failure_episode(
            root, e["series"], e["origin"], executors[e["series"]],
            expect_gain=by_key[(e["series"], e["origin"])]))
    groups = group_first_faults(eps, min_group=2)
    if not groups:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "family did not re-group"}, indent=1))
        return 0
    group = groups[0]
    # 对比案例（winsorize POSITIVE 窗口——census rounds 读数）
    positive_windows: list[dict[str, Any]] = []
    for sid, rounds in (census.get("development_rounds") or {}).items():
        for r in rounds:
            for cid, gain in r.get("probes") or []:
                if cid == "cand_winsorize" and gain is not None \
                        and gain >= M:
                    positive_windows.append({"series": sid,
                                             "origin": r["origin"],
                                             "gain": gain})
    contrast_eps = [_positive_contrast_episode(w["series"], w["origin"],
                                               w["gain"])
                    for w in positive_windows]
    view_keys = {ep.episode_id: list(eval_series) for ep in eps}
    capsule = build_contrast_capsule(group, all_episodes=eps + contrast_eps,
                                     view_keys=view_keys)
    headroom = family["replacement_headroom"]

    # ---- 真实 Slow Agent（一次调用）----
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
    series0 = _load_series(root, "T100")
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(series0[:600], task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL)
    slow = TTHASlowAgent(core)

    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    store = SnapshotStore(root / ".w4_store")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    method = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(
            backend, LocalPublicToolGateway(series0[:600],
                                            task_kind="forecast"),
            model=smoke.MODEL, base_url=smoke.BASE_URL)),
        h0, tuple(eps))

    request = _request(series0, {s: _load_series(root, s)
                                 for s in ("T100",) + eval_series}, 600)
    task_ctx = forecast_task_context_v1(
        task_spec=request.task_spec,
        deployment_constraints=deployment_constraints_v1())
    contracts = tuple(public_operator_contract(op) for op in OPS)

    def _eval_group(steps, ep):
        sid = str(ep.episode_id).split("_dev_")[1].split("_")[0]
        origin = int(((getattr(ep, "context_summary", {}) or {})
                      .get("support_origin") or 0))
        return executors[sid].evaluate(tuple(steps), origin)

    ev = method.handle_group_feedback(group, capsule, slow_agent=slow, controller=controller, store=store, card_builder=lambda g, cap: _group_card(g, cap, headroom), evaluator_group=_eval_group, holdout_evaluator=None, fast_features=dict(extract_public_features(series0[:600], task_kind='forecast')), allowed_operator_contracts=contracts, task_context=task_ctx, surface_catalog=controlled_add_only_group_catalog(), route_decision=controlled_add_only_group_decision(case_id='dev-run_v1_group_evolution_top_family_dev-326'))

    # ---- 补集检查 + delayed（仅 pending 可达——本 family 预期不可达）----
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
                 "winsorize_gain": w["gain"], "patch_gain": g})
            if g is None or g < -M:
                ok_comp = False
        complement["passed"] = ok_comp
        if ok_comp:
            dev = method.handle_feedback_delayed(
                lambda s, _m: executors["T100"].evaluate(tuple(s), 1080),
                episode_id=ev.get("episode_id"))
        else:
            complement["stage"] = "complement_rejected"

    # ---- 判定（预注册——与 docstring 严格一致）----
    # 本 family 无共同正向替代——唯一与证据一致的行为=abstain；
    # 选白名单 Patch = 选择与"共同正向 headroom"证据矛盾 → USE_FAILURE
    # （组内 replay 门如实拒绝）
    stage = ev.get("stage")
    pid = ev.get("patch_id")
    reason = ev.get("no_proposal_reason") or slow.last_no_proposal_reason
    if stage == "no_manifest":
        r = str(reason or "").lower()
        verdict = ("EVIDENCE_GROUNDED_ABSTAIN"
                   if any(k in r for k in ("insufficient", "evidence",
                                           "risk", "abstain", "no_authorized"))
                   else "SLOW_AGENT_EVIDENCE_USE_FAILURE")
    elif stage in ("pending", "group_replay_rejected", "holdout_rejected",
                   "manifest_proposed", "applied", "no_frozen_program"):
        verdict = "SLOW_AGENT_EVIDENCE_USE_FAILURE"  # 预注册：本 family
        # 下任何白名单选择都与共同正向证据矛盾
    elif stage == "typed_patch_contract_failed":
        verdict = "SLOW_AGENT_EVIDENCE_USE_FAILURE"
    else:
        verdict = "PROTOCOL_FAILURE"
    if counter.calls > 2:
        verdict = "PROTOCOL_FAILURE"
    family_conclusion = (
        "NO_COMMON_PROGRAM_HEADROOM"
        if verdict in ("EVIDENCE_GROUNDED_ABSTAIN",
                       "SLOW_AGENT_EVIDENCE_USE_FAILURE")
        else ("TEMPORALLY_UNSTABLE_REPAIR"
              if dev is not None and dev.get("stage") == "delayed_rejected"
              else (stage or "pending")))
    # 判定注记（abstain 意图证据——edit_id 由模型自拟）
    if pid is not None and str(ev.get("edit_id", "")).startswith("abstain"):
        verdict_note = ("edit_id 表明模型曾意图弃权（calls>1=首响应被校验"
                        "重试——疑似 no_proposal 信封格式失配后回退到"
                        "manifest）——abstain 通道格式脆弱待协议级诊断")
    else:
        verdict_note = None

    # ---- 批增益报告（计划要求）----
    fam_gains = [e["gain"] for e in family["episodes"]]
    batch_gain = {
        "target_group": {"windows": len(fam_gains),
                         "gains": fam_gains,
                         "positive_fraction": sum(1 for g in fam_gains
                                                  if g >= M) / len(fam_gains),
                         "tail_harm": min(fam_gains),
                         "mean": sum(fam_gains) / len(fam_gains)},
        "complement": {"windows": len(positive_windows),
                       "winsorize_gains": [w["gain"]
                                           for w in positive_windows]},
        "headroom": headroom,
        "group_replay_receipts": len(ev.get("group_replay") or []),
        "support_receipts_used": (len(ev.get("group_replay") or [])
                                  + len(complement.get("windows") or [])),
        "delayed": dev,
        "unpredicted_regression": [
            w for w in (complement.get("windows") or [])
            if w.get("patch_gain") is not None and w["patch_gain"] < -M],
    }

    report = {
        "experiment_id": "v1-group-evolution-top-family-r2",
        "note": "Wave 4a-r2：abstain 通道修复（agent_core 教学 no_proposal"
                "信封 + 重试反馈附模板——checker/reviewer 裁决 "
                "REPAIR_CURRENT_WAVE）后对同一 family 恰好一次真实重跑"
                "（development exposure——零新 Claim；协议修复重试——"
                "非重跑挑答案；r1/corrected 保留为证据）",
        "family": family,
        "group": {"workflow": group["workflow"], "sign": group["sign"],
                  "episodes": [e.episode_id for e in group["episodes"]]},
        "capsule": capsule,
        "group_feedback_event": ev,
        "complement_check": complement,
        "slow_agent_choice": pid,
        "no_proposal_reason": reason,
        "slow_stage_result": (None if slow.last_stage_result is None else {
            "validation_attempt_count":
                slow.last_stage_result.validation_attempt_count,
            "validation_retry_count":
                slow.last_stage_result.validation_retry_count,
            "first_pass_valid":
                slow.last_stage_result.first_pass_valid,
            "validation_error_codes": list(
                slow.last_stage_result.validation_error_codes or ()),
            "no_proposal_reason":
                slow.last_stage_result.no_proposal_reason,
        }),
        "llm": {"model": smoke.MODEL, "base_url": smoke.BASE_URL,
                "calls": counter.calls, "schema": "slow_edit_v1",
                "temperature": "default"},
        "batch_gain": batch_gain,
        "family_conclusion": family_conclusion,
        "verdict": verdict,
    }
    print("== group_feedback: " + json.dumps(ev, ensure_ascii=False))
    print(f"== slow choice: {pid} calls={counter.calls} "
          f"reason={reason}")
    print(f"== verdict: {verdict} / family: {family_conclusion}")
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
