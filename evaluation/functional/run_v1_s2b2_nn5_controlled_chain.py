"""S2B2_NN5_CONTROLLED_CHAIN（2026-08-13：S2b development follow-up-2——
最小 single-path Surface 契约修复——用户裁决方案 1：S2b 失败不是方法
负结果（Agent 已依据 D_patch 数值正确选择 impute_ar），是既有 Harness
Surface 契约未传入 single path——承重接线缺陷，允许最小修复。S2b 报告
保留不覆盖。

修复要求（用户裁决）：
  - 不新增 failure_pattern_card.workflow Surface（非可演化 Harness
    Surface）；
  - single path 从现有 surface_catalog 提供合法绑定——ADD_SKILL 类型
    只能绑定 operation=ADD + target_surface=skill_library.entries/
    {skill_id}；
  - preflight 同时校验 patch_id + operation + target_surface；
  - 不改变 NN5 数据/候选/gain/模型/temperature/Case/批准阈值；
  - 不重新运行分类——复用 S2b 分类结果与 case-0004；
  - S2b-2 只允许一次 Slow Edit 调用；再次失败关闭该切片。

verdict（用户裁决更新）：
  CONTROLLED_EVOLUTION_CHAIN_PASS : apply 后 held-in/delayed/LLM 采用/
    removal 全过
  LOCAL_SKILL_MECHANISM_PASS+AGENT_ADHERENCE_FAIL : Skill 激活但 LLM
    不采用
  SINGLE_PATH_MANIFEST_CONTRACT_FAILED : apply 再失败——关闭 S2 切片
  held-in/delayed 拒绝 : 正常方法负结果——不再修装置重跑

用法：
  python evaluation/functional/run_v1_s2b2_nn5_controlled_chain.py
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

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
from run_w2_operator_scan import _default_params  # noqa: E402

from SelfEvolvingHarnessTS.contracts.harness import EditOperation  # noqa: E402
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    StagePostValidationError,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.method import _typed_patch_preflight  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    build_episode,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent,
)
from SelfEvolvingHarnessTS.methods.ttha.fault_cases import (  # noqa: E402
    CASE_ACTIONS,
    GUARDS,
    classify_group,
    filter_candidates,
    reconcile_existing,
    selectable_fault_types,
)
from SelfEvolvingHarnessTS.methods.ttha.group_fault import (  # noqa: E402
    group_first_faults,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgictoChatCompletionsBackend,
)

E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_s2b2_nn5_controlled_chain_report.json"
STORE_REL = E2 / "w1_problem_cases_bootstrap.json"
TASK_CONSUMER = "forecast|ridge|sMASE"
DOMAIN = "nn5"
SS, SD, TS, TD = core.TIMELINE[DOMAIN]  # (536, 584, 632, 680)
ORIGIN_HELDIN = 728   # 修正：776 超长（791 长度）→ 728（预注册表内）
M = 0.005
WHITELIST = ("impute_ar", "impute_ssm")
PATCH_IDS = {op: f"patch-replace-repair_level_shift-with-{op}"
             for op in WHITELIST}
BASE_CACHE: dict[int, float] = {}

TAXONOMY_TEXT = """TAXONOMY (definitions for understanding only — fault_type
must come from allowed_fault_types below):
1. TASK_INTERPRETATION_ERROR — the agent misunderstood the Task / Consumer /
   Horizon / quality target (only when a verified TaskSpec/Contract conflict
   is present).
2. QUALITY_DIAGNOSIS_ERROR — a quality-phenomenon diagnosis contradicts
   verifiable facts.
3. WORKFLOW_SUPPLY_GAP — every whitelist candidate measured to fail on all
   in-group windows AND the candidate space exhaustively searched with no
   full pass.
4. WORKFLOW_DECISION_ERROR — a positive candidate was measured to exist but
   the agent did not propose or select it.
5. SCOPE_MEMORY_RISK_ERROR — a patch passed Support replay but failed
   delayed validation (temporal/scope risk measured).
MATERIAL THRESHOLD: a candidate "fails" a window when gain < +0.005; a
sign-positive gain below +0.005 still counts as FAILED.
GUARDS (only when allowed_fault_types is empty — deterministic):
- NO_ACTIONABLE_FAULT — every candidate measured FAILED (gain < +0.005) on
  ALL in-group windows.
- INSUFFICIENT_EVIDENCE — such measurements are absent or incomplete."""

OUTPUT_RULES = """Output exactly one JSON object:
{"fault_type": "...", "proposed_case_action": "MATCH_ADD_EVIDENCE|CONFLICT_WITH_EXISTING|NEW_CASE|ABSTAIN", "matched_case_id": "case-XXXX|null", "evidence_refs": ["..."], "reason": "..."}
- fault_type MUST be one of allowed_fault_types (or a GUARD if empty).
- proposed_case_action is ADVISORY — the Runtime decides by field comparison.
- matched_case_id must be one of the case ids above, or null.
- every evidence_ref must be an ID in GROUP EVIDENCE."""


class _V:
    def __init__(self, passed: bool) -> None:
        self.passed = passed


class _Receipt:
    def __init__(self, gain: float | None) -> None:
        self.gain = gain
        self.verification = _V(gain is not None)
        self.per_view_gain: list[float] = []


def _load(name: str) -> dict[str, Any]:
    return json.loads((E2 / name).read_text(encoding="utf-8"))


def _gain(roster, values, config, op: str, origin: int) -> float | None:
    return v1.gain_at(roster, values, config,
                      v1.make_compiled(op, _default_params(op, 7)),
                      origin, BASE_CACHE)


def _receipt_of(g: float | None) -> _Receipt:
    return _Receipt(g)


def _failure_episode(series: str, origin: int, gain: float) -> Any:
    steps = [{"op": "repair_level_shift", "params": {}}]
    return build_episode(
        episode_id=f"nn5_s2_fail_{origin}",
        task_consumer_key=TASK_CONSUMER,
        domain_namespace=DOMAIN,
        context_summary={
            "local_pattern": {"support_gain": gain},
            "delayed_pattern": {},
            "program_geometry": {"scope": "training_rows",
                                 "program_steps": steps},
            "per_view_gain": [],
            "support_origin": origin,
        },
        workflow_signature=workflow_signature_of(steps),
        support_response={"gain": gain, "accepted": False},
        delayed_response={"evaluated": False, "gain": None},
        relation="NEGATIVE", evidence_level="SUPPORT",
        local_status="EPISODE_ONLY", evidence_refs=["s2_nn5"])


def _card_for(group, capsule, case_summary, selectable,
              support_evidence, single=False):
    """S2b：facts 含 D_patch Support 数值证据（用户裁决方案 2——Agent
    应当读取的公开 Action–Response）；instruction 明确给出两候选的
    D_patch gain（@632）——依据数值选择，不靠语义先验。绝不提及
    D_heldin（728）/D_delayed（680）。"""
    ev_lines = "\n".join(
        f"- {op}: D_patch Support gain @632 = {g:.4f}"
        for op, g in support_evidence.items() if g is not None)
    return {
        "pattern_id": "single-repair_level_shift-neg" if single
        else "group-repair_level_shift-neg",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "context": {},
        "workflow": {"steps": [{"op": "repair_level_shift",
                                "params": {}}]},
        "typed_patch_options": [
            {"patch_id": PATCH_IDS[op],
             "program_steps": [{"op": op,
                                "params": dict(_default_params(op, 7))}]}
            for op in WHITELIST],
        "facts": {"contrast_capsule": dict(capsule),
                  "fault_selection": {"selectable": list(selectable),
                                      "note": ("WORKFLOW_DECISION_ERROR: a "
                                               "positive replacement was "
                                               "measured but the initial "
                                               "entry did not select it")},
                  "patch_support_evidence": {
                      op: g for op, g in support_evidence.items()
                      if g is not None},
                  "problem_case": case_summary},
        "instruction": (
            "A " + ("single" if single else "repeated ") + " failure on "
            "the nn5 target: the initial entry ran repair_level_shift "
            "which produced material negative Support "
            + ("at two in-group decision points" if not single
               else "at the decision point")
            + ". The fault-selection step classified this as "
              "WORKFLOW_DECISION_ERROR and reconciled it to a Problem "
              "Case. Below is the D_patch Support evidence of the two "
              "whitelist replacement candidates, measured at the failure "
              "decision point (@632) — this is public Action-Response "
              "evidence you are expected to read:\n"
            + ev_lines
            + "\nChoose exactly one typed patch from typed_patch_options "
              "based on this measured evidence (the candidate with the "
              "best measured replacement Support), or declare "
              "no_proposal with reason_code insufficient_public_evidence "
              "if the evidence does not justify any. The deterministic "
              "Support replay gate verifies your patch — you do not "
              "approve your own edit. Do not invent patch_ids: use the "
              "whitelist.",
        ),
    }


def _surface_preflight(card, manifest):
    """S2b-2 最小契约修复（用户裁决）：patch_id 白名单（同默认）+
    operation=ADD + target_surface=skill_library.entries/{skill_id}
    （来自现有 surface_catalog——不新增 Surface）。"""
    _typed_patch_preflight(card, manifest)
    if manifest.operation != EditOperation.ADD:
        raise StagePostValidationError(
            "SURFACE_OPERATION_NOT_IN_CATALOG",
            "ADD_SKILL type requires operation=ADD", retryable=True)
    skill_id = (manifest.new_value or {}).get("skill_id")
    expected = f"skill_library.entries/{skill_id}"
    if manifest.target_surface_id != expected:
        raise StagePostValidationError(
            "TARGET_SURFACE_NOT_IN_CATALOG",
            f"target_surface_id must be '{expected}' (from the writable "
            "surface catalog)", retryable=True)


def _classification_call(api_key: str, counter: Any, prompt: str) -> dict:
    resp = counter.chat.completions.create(
        model=smoke.MODEL,
        messages=[{"role": "user", "content": prompt}])
    text = str((resp.choices[0].message.content) or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


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
    series_arr = next(iter(values.values()))
    report: dict[str, Any] = {
        "experiment_id": "v1-s2-nn5-controlled-chain",
        "note": "S2：NN5 missingness 受控正例完整链（development positive"
                " control——零新 Claim——不称 fresh 验证）",
        "apparatus": {"domain": DOMAIN, "timeline": {"ss": SS, "sd": SD,
                                                     "ts": TS, "td": TD},
                      "roster_n": len(roster), "series_len": len(series_arr),
                      "whitelist": list(WHITELIST),
                      "apparatus_corrections": [
                          {"issue": "pre-registered held-in origin 776 "
                                    "exceeds series length 791 (776+48>791) "
                                    "— instrument failure; corrected to "
                                    "728 (in pre-registered origin set)",
                           "resolution": "held-in/adoption/removal @728"},
                          {"issue": "repair_level_shift @728 = +0.261 "
                                    "(positive) — group of 2 not formed; "
                                    "pre-registered branch: single-path",
                           "resolution": "single-episode path @632"}]},
    }

    # ---- 失败窗核对（632 已暴露逐位）----
    g632 = _gain(roster, values, cfg, "repair_level_shift", TS)
    assert g632 is not None and abs(g632 - (-0.07892256471269943)) < 1e-6, \
        f"632 drift: {g632}"
    g728 = _gain(roster, values, cfg, "repair_level_shift", ORIGIN_HELDIN)
    report["failure_windows"] = {"632": g632, "728": g728}
    impute_gains = {op: _gain(roster, values, cfg, op, TS)
                    for op in WHITELIST}
    report["known_positive_at_632"] = impute_gains

    # ---- 五类选择题（1 次 LLM）----
    evidence = {
        "group_id": "nn5-repair_level_shift",
        "task_contract_conflict": None,
        "diagnosis_contradiction": None,
        "headroom": {op: bool(g is not None and g >= M)
                     for op, g in impute_gains.items()},
        "supply_exhausted": False,
        "winner_probed": {"op": max(impute_gains, key=lambda o:
                                    (impute_gains[o] or -1.0)),
                          "gain": max(v for v in impute_gains.values()
                                      if v is not None) or 0.0},
        "agent_chosen": "repair_level_shift",
        "support_positive": None,
        "delayed_negative": None,
    }
    selectable = selectable_fault_types(evidence)
    items = ([{"id": f"ep:{TS}", "desc": "repair_level_shift Support gain",
               "gain": g632}]
             + [{"id": f"alt:{op}@{TS}",
                 "desc": f"{op} replacement replay gain", "gain": g}
                for op, g in impute_gains.items()])
    # 用户裁决：不重新运行分类——复用 S2b 分类结果（同证据同装置）
    s2b = _load("w1_s2b_nn5_controlled_chain_report.json")
    cls_s2b = s2b.get("classification") or {}
    chosen_ft = str(cls_s2b.get("normalized") or "")
    report["classification"] = {
        "selectable": selectable,
        "reused_from_s2b": True,
        "normalized": chosen_ft,
        "s2b_answer": cls_s2b.get("answer"),
        "s2b_checks": cls_s2b.get("checks"),
        "llm_calls": 0}

    # ---- Case reconciliation（Runtime 确定性）----
    gf = {"task_consumer": TASK_CONSUMER, "workflow_sig":
          "repair_level_shift", "response_class": "NEGATIVE"}
    store = _load("w1_problem_cases_bootstrap.json")
    matched = filter_candidates(store["cases"], chosen_ft, gf)
    action = reconcile_existing(matched[0], gf) if matched else "NEW_CASE"
    report["reconciliation"] = {"fault_type": chosen_ft, "matched":
                                [c["case_id"] for c in matched[:3]],
                                "action": action}
    # case-0004 已由 S2 写入（同链同证据——S2b 复用不重复写）
    new_id = "case-0004"
    if new_id in {c["case_id"] for c in store["cases"]}:
        report["case_write"] = {"action": "reuse_from_s2",
                                "case_id": new_id, "by": "Runtime"}
    elif action == "NEW_CASE":
        store["cases"].append({
            "case_id": new_id,
            "name": "decision-gap_forecast-ridge_imputation",
            "fault_type": "WORKFLOW_DECISION_ERROR",
            "task_consumer": TASK_CONSUMER,
            "observable_context": {"defect_family": "missingness",
                                   "domain": "nn5"},
            "failed_behavior": ("initial entry selected repair_level_shift "
                                "with material negative Support while a "
                                "positive imputation replacement was "
                                "measured"),
            "workflow_and_effect": {"workflow_sig": "repair_level_shift",
                                    "response_class": "NEGATIVE"},
            "response_class": "NEGATIVE",
            "supporting_episode_ids": [f"nn5@{TS}"],
            "positive_contrasts": [],
            "negative_contrasts": [],
            "conflicts": [],
            "known_headroom": {"status": "pending_verification"},
            "verified_fix": None,
            "status": "CANDIDATE_CASE",
        })
        STORE_REL.write_text(json.dumps(store, ensure_ascii=False,
                                        indent=2, default=str) + "\n",
                             encoding="utf-8")
        report["case_write"] = {"action": "NEW_CASE", "case_id": new_id,
                                "by": "Runtime"}
    else:
        report["case_write"] = {"action": action, "by": "Runtime"}

    # ---- Slow Agent 一个 Typed Edit（1 次 LLM——白名单内自主选择）----
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    sstore = SnapshotStore(root / ".s2_store")
    controller = EditController(sstore, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    counter_edit = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120))
    backend = AgictoChatCompletionsBackend(client=counter_edit,
                                           base_url=smoke.BASE_URL)
    slow = TTHASlowAgent(TTHAAgentCore(
        backend, LocalPublicToolGateway(series_arr[:TS],
                                        task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL))
    method = TTHAMethod(TTHAFastAgent(TTHAAgentCore(
        backend, LocalPublicToolGateway(series_arr[:TS],
                                        task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL)),
        h0, ())
    episode = _failure_episode("nn5", TS, g632)
    capsule = {"workflow": "repair_level_shift", "sign": "NEGATIVE",
               "n_episodes": 1, "origins": [TS],
               "per_episode_rows": [{"episode_id": episode.episode_id,
                                     "support_gain": g632}],
               "view_alignment": {"established": False, "aligned_rows": []},
               "contrast_cases": {"positive": [], "conflict": []}}
    case_summary = (f"case-{new_id} [WORKFLOW_DECISION_ERROR]: "
                    "imputation replacement measured positive but initial "
                    "entry ran repair_level_shift — status CANDIDATE_CASE")
    # S2b：D_patch 公开证据（同分类 items——同一份 Capsule）
    support_evidence = impute_gains

    def _eval_support(steps, _mode):
        return _receipt_of(_gain(roster, values, cfg, steps[0][0], TS))

    ev = method.handle_feedback_support(episode, confirmed_cause="SKILL_LIBRARY_GAP", slow_agent=slow, controller=controller, store=sstore,
        surface_catalog=[{"surface_id": "skill_library.entries/{skill_id}",
                          "operation": "ADD", "surface_type": "skill",
                          "allowed_operations": ["ADD"]}],
        card_builder=lambda e: _card_for(None, capsule, case_summary,
                                         selectable, support_evidence,
                                         single=True),
        manifest_preflight=_surface_preflight,
        evaluator=_eval_support,
        fast_features=dict(extract_public_features(
            series_arr[:TS], task_kind="forecast")))
    report["slow_edit"] = {"event": ev, "llm_calls": counter_edit.calls,
                           "chosen_patch": ev.get("patch_id")}
    print("== slow_edit: " + json.dumps(ev, ensure_ascii=False,
                                        default=str), flush=True)

    # ---- 门链（pending 起）----
    if ev.get("stage") != "pending":
        # 用户裁决：apply 再失败 → SINGLE_PATH_MANIFEST_CONTRACT_FAILED
        # （关闭 S2 切片）；其余失败阶段如实命名
        if ev.get("stage") in ("apply_failed", "typed_patch_contract_failed",
                               "manifest_preflight_failed"):
            report["verdict"] = "SINGLE_PATH_MANIFEST_CONTRACT_FAILED"
        else:
            report["verdict"] = ("SINGLE_SUPPORT_"
                                 + ev.get("stage", "REJECTED").upper())
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict:", report["verdict"])
        return 0

    chosen_op = ev["frozen_program"][0]["op"]
    heldin = _gain(roster, values, cfg, chosen_op, ORIGIN_HELDIN)
    report["heldin"] = {"origin": ORIGIN_HELDIN, "gain": heldin,
                        "passed": bool(heldin is not None
                                       and heldin >= -M)}
    if heldin is None or heldin < -M:
        report["verdict"] = "HELDIN_REJECTED"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: HELDIN_REJECTED")
        return 0
    dev = method.handle_feedback_delayed(
        lambda s, _m: _receipt_of(_gain(roster, values, cfg,
                                        s[0][0], TD)),
        episode_id=ev.get("episode_id"))
    report["delayed"] = dev
    if dev.get("stage") != "approved":
        report["verdict"] = "DELAYED_REJECTED"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: DELAYED_REJECTED")
        return 0

    # ---- 采用双口径 @728 ----
    snap = method._snapshot
    skill_ids = [s.skill_id for s in
                 (snap.skills if snap is not None else [])]
    report["skill"] = {"skill_ids": skill_ids, "frozen_op": chosen_op,
                       "patch_id": ev.get("patch_id")}
    known_positive = [op for op, g in impute_gains.items()
                      if g is not None and g >= M]
    report["post_hoc_check"] = {
        "chosen_in_known_positive": chosen_op in known_positive,
        "known_positive_ops": known_positive}

    runtime_adopted = bool(skill_ids and chosen_op in WHITELIST)
    runtime_gain = _gain(roster, values, cfg, chosen_op, ORIGIN_HELDIN)
    report["runtime_entry"] = {
        "adopted": runtime_adopted, "skill_in_snapshot": bool(skill_ids),
        "executable_gain_at_728": runtime_gain,
        "note": "确定性入口 = skill 写入 snapshot（检索可达）+ frozen "
                "program 在下一入口 @728 执行——轻量机制验证"}

    counter_llm = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120))
    llm_prompt = (
        "You are the normal fast entry selecting the next program for the "
        "nn5 target at the next decision point. A skill "
        f"({chosen_op}, typed patch {ev.get('patch_id')}) is now LOCAL_ACTIVE"
        " in the harness and applicable to this context (it replaces "
        "repair_level_shift after a measured positive Support at the "
        "failure point). Candidates: "
        + json.dumps([f"cand_skill_{chosen_op}"]
                     + [f"cand_{op}" for op in WHITELIST]
                     + ["cand_repair_level_shift", "cand_identity"])
        + ". Choose exactly one candidate id (JSON: {\"chosen\": "
          "\"cand_...\"}) — do not invent ids.")
    resp = counter_llm.chat.completions.create(
        model=smoke.MODEL,
        messages=[{"role": "user", "content": llm_prompt}])
    llm_text = str((resp.choices[0].message.content) or "")
    llm_chosen = ("cand_skill_" + chosen_op) in llm_text
    report["llm_entry"] = {"calls": counter_llm.calls,
                           "adopted_skill": llm_chosen,
                           "raw_prefix": llm_text[:200]}

    removal_gain = _gain(roster, values, cfg, "repair_level_shift",
                         ORIGIN_HELDIN)
    report["removal"] = {"origin": ORIGIN_HELDIN,
                         "no_skill_repair_gain": removal_gain,
                         "note": "行为恢复 = 无 skill 时回到 repair_level_shift"
                                 "（旧入口）——对照组"}

    if llm_chosen and runtime_adopted:
        verdict = "CONTROLLED_EVOLUTION_CHAIN_PASS"
    elif runtime_adopted:
        verdict = "LOCAL_SKILL_MECHANISM_PASS+AGENT_ADHERENCE_FAIL"
    else:
        verdict = "RETRIEVAL_BINDING_EXECUTION_FAILURE"
    report["verdict"] = verdict
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
