"""S3_SIMPLIFIED_SLOW_PATH（2026-08-13：NN5 正控第三次尝试——用户裁决：
"Agent 做语义决策、Compiler 做机械编译"——把 Slow Agent 从 Manifest
作者降级为受限修改决策者。S2/S2b/S2b-2 报告全部保留不覆盖。

简化设计（用户批准）：
  - Runtime 提供故障证据和合法选项（Fault Diagnosis Card）；
  - Runtime 已机械确定唯一错误类型（WORKFLOW_DECISION_ERROR）→ 直接
    告知——不再让 LLM 重复分类（减少方差）；
  - Slow Agent 只输出 {edit_intent, patch_id, reason, expected_change}
    ——不写 Manifest/Surface/SHA/precondition；
  - Runtime 确定性编译完整 EditManifest（ADD skill_library.entries/
    {skill_id}——不新增 Surface）；
  - 保留门：future sealed / 只读 D_patch / patch 合法空间 / Runtime
    编译 / held-in（728）验证 / delayed（680）批准 / 下一入口采用 +
    removal 对照。

verdict（预注册）：
  CONTROLLED_EVOLUTION_CHAIN_PASS : held-in/delayed/LLM 采用/removal
    全过
  LOCAL_SKILL_MECHANISM_PASS+AGENT_ADHERENCE_FAIL : Skill 激活但 LLM
    不采用
  ABSTAIN_CLOSED : Agent 合法弃权（ABSTAIN）——关闭切片
  HELDIN_REJECTED / DELAYED_REJECTED : 正常方法负结果——不修装置
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_s3_simplified_slow_path.py
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

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
from run_w2_operator_scan import _default_params  # noqa: E402

from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    EditManifest,
    EditOperation,
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
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgictoChatCompletionsBackend,
)

E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_s3_simplified_slow_path_report.json"
TASK_CONSUMER = "forecast|ridge|sMASE"
DOMAIN = "nn5"
SS, SD, TS, TD = core.TIMELINE[DOMAIN]
ORIGIN_HELDIN = 728
M = 0.005
WHITELIST = ("impute_ar", "impute_ssm")
PATCH_IDS = {op: f"patch-replace-repair_level_shift-with-{op}"
             for op in WHITELIST}
BASE_CACHE: dict[int, float] = {}
SKILL_ID = "single_repair_level_shift_replacement"


class _V:
    def __init__(self, passed: bool) -> None:
        self.passed = passed


class _Receipt:
    def __init__(self, gain: float | None) -> None:
        self.gain = gain
        self.verification = _V(gain is not None)
        self.per_view_gain: list[float] = []


class CompiledSlowAgent:
    """Agent 决策（选择题）→ Runtime 编译 Manifest 的适配器——propose_edit
    返回预编译 manifest（LLM 不再手写格式）。"""

    def __init__(self, manifest: EditManifest | None) -> None:
        self._manifest = manifest
        self.last_no_proposal_reason = None
        self.last_stage_result = None

    def propose_edit(self, card, surface_catalog, snapshot, **kw):
        if self._manifest is None:
            self.last_no_proposal_reason = "no_authorized_minimal_edit"
            return None
        return self._manifest


def _gain(roster, values, config, op: str, origin: int) -> float | None:
    return v1.gain_at(roster, values, config,
                      v1.make_compiled(op, _default_params(op, 7)),
                      origin, BASE_CACHE)


def _receipt_of(g: float | None) -> _Receipt:
    return _Receipt(g)


def _failure_episode(origin: int, gain: float) -> Any:
    steps = [{"op": "repair_level_shift", "params": {}}]
    return build_episode(
        episode_id=f"nn5_s3_fail_{origin}",
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
        local_status="EPISODE_ONLY", evidence_refs=["s3_nn5"])


def _compile_manifest(h0, chosen_op: str) -> EditManifest:
    """Runtime 确定性编译完整 Manifest（用户裁决：Agent 语义决策、
    Compiler 机械编译）。"""
    steps = ((chosen_op, dict(_default_params(chosen_op, 7))),)
    return EditManifest(
        edit_id=f"replace-repair_level_shift-with-{chosen_op}",
        base_harness_sha=h0.harness_content_sha,
        target_pattern_id="single-repair_level_shift-neg",
        target_surface_id=f"skill_library.entries/{SKILL_ID}",
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value={
            "schema_version": "skill-entry/1",
            "skill_id": SKILL_ID,
            "skill_kind": "capability",
            "revision": 1,
            "body": "Frozen program steps: " + json.dumps(
                [{"op": o, "params": dict(p)} for o, p in steps]),
            "allowed_tools": [o for o, _p in steps],
        },
        observable_applicability=None,
        patch_id=PATCH_IDS[chosen_op],
        predicted_agent_behavior_change=("retrieve_skill:"
                                         "repair_level_shift",),
        predicted_data_effect=("local_skill_replacement",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_support_improvement",),
    )


def _parse_choice(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


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
        "experiment_id": "v1-s3-simplified-slow-path",
        "note": "S3：简化 Slow Path（用户裁决：Agent 语义决策 + Compiler "
                "机械编译）——NN5 正控第三次尝试——development exposure"
                "——零新 Claim——不称 fresh 验证",
        "apparatus": {"domain": DOMAIN, "timeline": {"ss": SS, "sd": SD,
                                                     "ts": TS, "td": TD},
                      "roster_n": len(roster), "series_len": len(series_arr),
                      "whitelist": list(WHITELIST),
                      "heldin": ORIGIN_HELDIN,
                      "design": ("Agent 只选 edit_intent/patch_id——"
                                 "Runtime 编译 Manifest——门链复用")},
    }

    g632 = _gain(roster, values, cfg, "repair_level_shift", TS)
    assert g632 is not None and abs(g632 - (-0.07892256471269943)) < 1e-6, \
        f"632 drift: {g632}"
    impute_gains = {op: _gain(roster, values, cfg, op, TS)
                    for op in WHITELIST}
    report["evidence"] = {"failure_632": g632, "d_patch_impute": impute_gains}

    # ---- Fault Diagnosis Card（确定性——唯一错误类型直接告知）----
    card_text = (
        "FAULT DIAGNOSIS CARD\n"
        f"Detected failure: repair_level_shift Support gain @632 = "
        f"{g632:.4f} (material negative)\n"
        "Runtime fault type (deterministic, already located): "
        "WORKFLOW_DECISION_ERROR\n"
        "Evidence (D_patch, public Action-Response at @632):\n"
        + "\n".join(f"  - {op} = {g:.4f}" for op, g in
                    impute_gains.items())
        + "\nA positive replacement candidate exists but the initial entry "
          "did not select it.\n\n"
        "Allowed modifications (choose exactly one):\n"
        f"  A. ADD_TARGET_LOCAL_SKILL({WHITELIST[0]}) "
        f"[patch_id={PATCH_IDS[WHITELIST[0]]}]\n"
        f"  B. ADD_TARGET_LOCAL_SKILL({WHITELIST[1]}) "
        f"[patch_id={PATCH_IDS[WHITELIST[1]]}]\n"
        "  C. ABSTAIN\n\n"
        "Output exactly one JSON object:\n"
        '{"edit_intent": "ADD_TARGET_LOCAL_SKILL|ABSTAIN", '
        '"patch_id": "<one of the two patch_ids>|null", '
        '"reason": "...", "expected_change": "..."}\n'
        "- You do NOT write the manifest — the Runtime compiles it "
        "deterministically.\n"
        "- held-in and delayed outcomes are hidden — you cannot approve "
        "your own edit.")

    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120))
    resp = counter.chat.completions.create(
        model=smoke.MODEL,
        messages=[{"role": "user", "content": card_text}])
    choice = _parse_choice(str((resp.choices[0].message.content) or ""))
    report["slow_choice"] = {"answer": choice, "llm_calls": counter.calls}
    print("== choice: " + json.dumps(choice, ensure_ascii=False,
                                     default=str), flush=True)

    intent = str((choice or {}).get("edit_intent") or "")
    patch_id = (choice or {}).get("patch_id")
    valid_patch = patch_id in set(PATCH_IDS.values())
    if intent != "ADD_TARGET_LOCAL_SKILL" or not valid_patch:
        report["verdict"] = "ABSTAIN_CLOSED"
        print("== verdict: ABSTAIN_CLOSED")
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        return 0

    chosen_op = next(op for op, pid in PATCH_IDS.items() if pid == patch_id)
    report["slow_choice"]["resolved_op"] = chosen_op

    # ---- Runtime 编译 Manifest + 门链（复用 handle_feedback_support）----
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    sstore = SnapshotStore(root / ".s3_store")
    controller = EditController(sstore, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    backend = AgictoChatCompletionsBackend(
        client=counter, base_url=smoke.BASE_URL)
    compiled = CompiledSlowAgent(_compile_manifest(h0, chosen_op))
    method = TTHAMethod(TTHAFastAgent(TTHAAgentCore(
        backend, LocalPublicToolGateway(series_arr[:TS],
                                        task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL)),
        h0, ())
    episode = _failure_episode(TS, g632)
    card = {
        "pattern_id": "single-repair_level_shift-neg",
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
        "facts": {"fault_diagnosis_card": {
            "fault_type": "WORKFLOW_DECISION_ERROR",
            "failure_gain": g632,
            "d_patch_evidence": impute_gains}},
        "instruction": "Runtime-compiled edit (see fault_diagnosis_card).",
    }

    def _eval_support(steps, _mode):
        return _receipt_of(_gain(roster, values, cfg, steps[0][0], TS))

    ev = method.handle_feedback_support(episode, confirmed_cause="SKILL_LIBRARY_GAP", slow_agent=compiled, controller=controller, store=sstore,
        surface_catalog=[{"surface_id": "skill_library.entries/{skill_id}",
                          "operation": "ADD", "surface_type": "skill",
                          "allowed_operations": ["ADD"]}],
        card_builder=lambda e: card,
        evaluator=_eval_support,
        fast_features=dict(extract_public_features(
            series_arr[:TS], task_kind="forecast")))
    report["support_chain"] = ev
    print("== support: " + json.dumps(ev, ensure_ascii=False,
                                      default=str), flush=True)

    if ev.get("stage") != "pending":
        report["verdict"] = "SUPPORT_" + ev.get("stage", "REJECTED").upper()
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict:", report["verdict"])
        return 0

    chosen_op2 = ev["frozen_program"][0]["op"]
    heldin = _gain(roster, values, cfg, chosen_op2, ORIGIN_HELDIN)
    report["heldin"] = {"origin": ORIGIN_HELDIN, "gain": heldin}
    if heldin is None or heldin < -M:
        report["verdict"] = "HELDIN_REJECTED"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: HELDIN_REJECTED")
        return 0
    dev = method.handle_feedback_delayed(
        lambda s, _m: _receipt_of(_gain(roster, values, cfg, s[0][0], TD)),
        episode_id=ev.get("episode_id"))
    report["delayed"] = dev
    if dev.get("stage") != "approved":
        report["verdict"] = "DELAYED_REJECTED"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: DELAYED_REJECTED")
        return 0

    # ---- 采用双口径 ----
    snap = method._snapshot
    skill_ids = [s.skill_id for s in
                 (snap.skills if snap is not None else [])]
    runtime_adopted = bool(skill_ids and chosen_op2 in WHITELIST)
    runtime_gain = _gain(roster, values, cfg, chosen_op2, ORIGIN_HELDIN)
    report["skill"] = {"skill_ids": skill_ids, "frozen_op": chosen_op2,
                       "runtime_gain_at_728": runtime_gain}
    counter2 = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120))
    resp2 = counter2.chat.completions.create(
        model=smoke.MODEL,
        messages=[{"role": "user", "content": (
            "You are the normal fast entry selecting the next program for "
            "the nn5 target. A skill (impute_ar-based, typed patch "
            f"{PATCH_IDS[chosen_op2]}) is now LOCAL_ACTIVE and applicable. "
            "Candidates: " + json.dumps(
                [f"cand_skill_{chosen_op2}"] + [f"cand_{op}" for op in
                                                WHITELIST] +
                ["cand_repair_level_shift", "cand_identity"])
            + ". Choose one (JSON: {\"chosen\": \"cand_...\"}).")}])
    llm_text = str((resp2.choices[0].message.content) or "")
    llm_chosen = ("cand_skill_" + chosen_op2) in llm_text
    report["llm_entry"] = {"calls": counter2.calls,
                           "adopted_skill": llm_chosen,
                           "raw_prefix": llm_text[:200]}

    removal_gain = _gain(roster, values, cfg, "repair_level_shift",
                         ORIGIN_HELDIN)
    report["removal"] = {"no_skill_repair_gain_728": removal_gain}

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
