"""PROGRAM_SUPPLY_DEV（Wave 4b，2026-08-13：Program Supply 分支——
census top family 无共同正向替代（NO_COMMON_PROGRAM_HEADROOM，Agent 已
依公开证据弃权）→ 确定性 DSL 搜索供给候选（TATO 式——候选供给工具，
不回答归因）→ 全过候选全部进白名单（绝不手挑）→ 真实 Agent 一次调用
终选。development exposure——零新 Claim。

预注册搜索空间（固定于本文件——先于任何结果）：
  S1 单算子（seed 复用 census headroom——不重评）：
     winsorize {} / outlier_mad {} / hampel V1（contract）
  S2 hampel 参数变体（3 个，hampel 唯一有参数 schema 的算子）：
     V2 = {window: 7, n_sigmas: 3.0, global_z_min: 1.0}（文献默认）
     V3 = {window: 3, n_sigmas: 0.5, global_z_min: 1.0}（更强裁剪）
     V4 = {window: 5, n_sigmas: 2.0, global_z_min: 1.0}（中间档）
  S3 两步组合（全部有序对，hampel 用 contract 参数）：
     winsorize→outlier_mad / winsorize→hampel / outlier_mad→winsorize /
     outlier_mad→hampel / hampel→winsorize / hampel→outlier_mad

成功判据（同门同 M）：候选在 family 全部 6 个失败窗口上 verifier 通过
且 gain ≥ M（0.005）。

流程：搜索（54 次新评估——预算=搜索空间穷举，不按结果扩张）→ 全过
候选进白名单 → 真实 Slow Agent 一次调用终选（可弃权）→ 组内 replay
门（方法层重放 6 窗口）→ 补集检查（8 个 winsorize 正窗口 ≥ −M）→
delayed（预注册 T100@1080）→ 判定。

verdict（预注册）：
  SUPPLY_FOUND_PATCH_APPROVED       : 供给有解 + Agent 选 + 全门通过
  SUPPLY_FOUND_PATCH_DELAYED_REJECTED : pending 但 delayed 拒绝
    （=TEMPORALLY_UNSTABLE_REPAIR 语义）
  SUPPLY_FOUND_AGENT_ABSTAIN        : 供给有解但 Agent 依证据弃权
  SUPPLY_EXHAUSTED                  : 搜索空间穷举无全过候选（family +
    邻域 development 级关闭）
  PROTOCOL_FAILURE

停止条件（用户任务书）：本 family 是 development block 唯一跨 series
family——SUPPLY_EXHAUSTED 或 AGENT_ABSTAIN 后无安全可解释的下一分支
→ 停实验，生成完整晨间报告。

用法：
  python evaluation/functional/run_v1_program_supply_dev.py
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
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_program_supply_dev_report.json"

# ---- 预注册搜索空间（固定——先于任何结果）----
HAMPEL_VARIANTS = {
    "V2": {"window": 7, "n_sigmas": 3.0, "global_z_min": 1.0},
    "V3": {"window": 3, "n_sigmas": 0.5, "global_z_min": 1.0},
    "V4": {"window": 5, "n_sigmas": 2.0, "global_z_min": 1.0},
}
PAIRS = (("winsorize", "outlier_mad"), ("winsorize", "hampel_filter"),
         ("outlier_mad", "winsorize"), ("outlier_mad", "hampel_filter"),
         ("hampel_filter", "winsorize"),
         ("hampel_filter", "outlier_mad"))
DELAYED_ORIGIN = 1080  # 预注册：T100（最强 harm series）@1080


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
                             executor: ScopeExecutor) -> Any:
    steps = (("winsorize", dict(wiring.contract_params("winsorize",
                                                      PERIOD))),)
    rr = executor.evaluate(tuple(steps), origin)
    gain = float(rr.gain) if rr.gain is not None else None
    assert gain is not None and gain < -M, f"{series}@{origin} not failure"
    ctx = dict(resolver.window_context({series: _load_series(root, series)},
                                       origin, PERIOD))
    return build_episode(
        episode_id=f"kdd2018_dev_{series}_target_winsorize_sup_{series}"
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
        workflow_signature="winsorize",
        support_response={"gain": gain, "accepted": False},
        delayed_response={"evaluated": False, "gain": None},
        relation="NEGATIVE", evidence_level="SUPPORT",
        local_status="EPISODE_ONLY", evidence_refs=["program_supply_w4b"])


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
        local_status="LOCAL_DRAFT", evidence_refs=["program_supply_w4b"])


def _supply_card(group: Mapping[str, Any],
                 capsule: Mapping[str, Any],
                 passing: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    options = []
    for i, cand in enumerate(passing):
        options.append({
            "patch_id": f"patch-supply-{i + 1}",
            "program_steps": [{"op": s[0], "params": dict(s[1])}
                              for s in cand["steps"]]})
    return {
        "pattern_id": "group-winsorize-neg-multiseries",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "context": {},
        "workflow": {"steps": [{"op": "winsorize", "params": {}}]},
        "typed_patch_options": options,
        "facts": {
            "contrast_capsule": dict(capsule),
            "supply_candidates": [{"patch_id": f"patch-supply-{i + 1}",
                                   "steps": cand["steps"],
                                   "per_window_gains":
                                       cand["per_window_gains"]}
                                  for i, cand in enumerate(passing)],
        },
        "instruction": (
            "A repeated first-fault group (winsorize, material negative at "
            "six windows across four independent series) had no common "
            "positive single-operator replacement. A deterministic supply "
            "search over the operator DSL produced the candidate programs "
            "in supply_candidates — each is positive on ALL in-group "
            "windows. Choose exactly one typed patch from "
            "typed_patch_options as the group replacement. You do not "
            "approve your own edit — group replay, complement and delayed "
            "gates verify it. If the evidence is insufficient to choose, "
            "declare no_proposal with reason_code "
            "insufficient_public_evidence — abstaining is valid.",
        ),
    }


def main() -> int:
    root = PROJECT_ROOT
    census = json.loads(CENSUS_REL.read_text(encoding="utf-8"))
    family = (census.get("development_families") or [None])[0]
    if family is None or family["n_series"] < 2:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no census family"}, indent=1))
        return 0
    pre = census["pre_registered"]
    eval_series = tuple(pre["eval_series"])
    executors = {s: _series_executor(root, s, eval_series)
                 for s in family["independent_series"]}

    # ---- 装置重建（episodes + capsule——同 wave4a 口径）----
    eps = []
    for e in family["episodes"]:
        eps.append(_rebuild_failure_episode(
            root, e["series"], e["origin"], executors[e["series"]]))
    groups = group_first_faults(eps, min_group=2)
    if not groups:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "family did not re-group"}, indent=1))
        return 0
    group = groups[0]
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

    # ---- 预注册供给搜索（S2 hampel 变体 + S3 两步组合——seed 复用
    # census headroom 的 winsorize/outlier_mad/hampel-V1 读数）----
    def _contract(op: str) -> dict[str, object]:
        return dict(wiring.contract_params(op, PERIOD))

    candidates: list[dict[str, Any]] = []
    for vid, params in HAMPEL_VARIANTS.items():
        candidates.append({"label": f"hampel_{vid}",
                           "steps": (("hampel_filter", params),)})
    for a, b in PAIRS:
        candidates.append({"label": f"{a}_to_{b}",
                           "steps": ((a, _contract(a)), (b, _contract(b)))})
    # 注意：winsorize{} / outlier_mad{} / hampel-V1 已在 census headroom
    # 中全窗口读过——其读数作为 seed 一并报告（不重评）。
    seed = family.get("replacement_headroom") or {}
    search: list[dict[str, Any]] = []
    for cand in candidates:
        per = []
        all_pass = True
        for e in family["episodes"]:
            sid, origin = e["series"], e["origin"]
            rr = executors[sid].evaluate(tuple(cand["steps"]), origin)
            g = (float(rr.gain) if rr.gain is not None else None)
            per.append({"series": sid, "origin": origin, "gain": g,
                        "passed": bool(rr.verification.passed)})
            if g is None or not rr.verification.passed or g < M:
                all_pass = False
        cand["per_window_gains"] = per
        cand["all_pass"] = all_pass
        search.append(cand)
        print(f"== supply {cand['label']}: all_pass={all_pass}")
    passing = [c for c in search if c["all_pass"]]

    if not passing:
        report = {
            "experiment_id": "v1-program-supply-dev",
            "note": "Wave 4b：Program Supply 分支——预注册搜索空间穷举无"
                    "全过候选（development exposure——零新 Claim）",
            "family": family,
            "search_space": {"hampel_variants": HAMPEL_VARIANTS,
                             "pairs": list(PAIRS),
                             "seed_from_census_headroom": seed},
            "search": search,
            "verdict": "SUPPLY_EXHAUSTED",
        }
        print("== verdict: SUPPLY_EXHAUSTED")
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print(f"== report -> {REPORT_REL}")
        return 0

    # ---- 供给有解：全过候选进白名单 → 真实 Agent 一次调用终选 ----
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
    store = SnapshotStore(root / ".sup_store")
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
    contracts = tuple(public_operator_contract(op)
                      for op in ("winsorize", "outlier_mad",
                                 "hampel_filter"))

    def _eval_group(steps, ep):
        sid = str(ep.episode_id).split("_dev_")[1].split("_")[0]
        origin = int(((getattr(ep, "context_summary", {}) or {})
                      .get("support_origin") or 0))
        return executors[sid].evaluate(tuple(steps), origin)

    ev = method.handle_group_feedback(group, capsule, slow_agent=slow, controller=controller, store=store, card_builder=lambda g, cap: _supply_card(g, cap, passing), evaluator_group=_eval_group, holdout_evaluator=None, fast_features=dict(extract_public_features(series0[:600], task_kind='forecast')), allowed_operator_contracts=contracts, task_context=task_ctx, surface_catalog=controlled_add_only_group_catalog(), route_decision=controlled_add_only_group_decision(case_id='dev-run_v1_program_supply_dev-367'))

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
                lambda s, _m: executors["T100"].evaluate(
                    tuple(s), DELAYED_ORIGIN),
                episode_id=ev.get("episode_id"))
        else:
            complement["stage"] = "complement_rejected"

    stage = ev.get("stage")
    pid = ev.get("patch_id")
    reason = ev.get("no_proposal_reason") or slow.last_no_proposal_reason
    if stage == "no_manifest":
        verdict = "SUPPLY_FOUND_AGENT_ABSTAIN"
    elif stage == "pending":
        verdict = ("SUPPLY_FOUND_PATCH_APPROVED"
                   if dev is not None and dev.get("stage") == "approved"
                   else "SUPPLY_FOUND_PATCH_DELAYED_REJECTED"
                   if dev is not None
                   and dev.get("stage") == "delayed_rejected"
                   else "SUPPLY_FOUND_PATCH_APPROVED")
        if complement.get("checked") and not complement.get("passed"):
            verdict = "SUPPLY_FOUND_PATCH_DELAYED_REJECTED"
    elif stage in ("group_replay_rejected", "holdout_rejected"):
        verdict = "SUPPLY_FOUND_PATCH_DELAYED_REJECTED"
    elif stage == "typed_patch_contract_failed":
        verdict = "PROTOCOL_FAILURE"
    else:
        verdict = "PROTOCOL_FAILURE"
    if counter.calls > 2:
        verdict = "PROTOCOL_FAILURE"

    report = {
        "experiment_id": "v1-program-supply-dev",
        "note": "Wave 4b：Program Supply 分支（development exposure——"
                "零新 Claim；搜索空间预注册；全过候选全进白名单——"
                "不手挑；真实 Agent 一次调用终选）",
        "family": family,
        "search_space": {"hampel_variants": HAMPEL_VARIANTS,
                         "pairs": list(PAIRS),
                         "seed_from_census_headroom": seed},
        "search": search,
        "passing_candidates": passing,
        "group": {"workflow": group["workflow"], "sign": group["sign"],
                  "episodes": [e.episode_id for e in group["episodes"]]},
        "capsule": capsule,
        "group_feedback_event": ev,
        "complement_check": complement,
        "slow_agent_choice": pid,
        "no_proposal_reason": reason,
        "llm": {"model": smoke.MODEL, "base_url": smoke.BASE_URL,
                "calls": counter.calls, "schema": "slow_edit_v1",
                "temperature": "default"},
        "verdict": verdict,
    }
    print("== group_feedback: " + json.dumps(ev, ensure_ascii=False))
    print(f"== slow choice: {pid} calls={counter.calls} reason={reason}")
    print(f"== verdict: {verdict}")
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
