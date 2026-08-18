"""S3B_NORMAL_ENTRY_ADOPTION（2026-08-13：审查者 P0——真实正常入口
adoption/removal 对照。S3 verdict 已修正为 CONTROLLED_EDIT_TO_PERSISTED_
SKILL_MECHANISM_PASS + NORMAL_ENTRY_ADOPTION_UNVERIFIED +
REMOVAL_UNVERIFIED（w1_s3_verdict_correction_record.txt）。本 runner
补测两处未验证项 + 时间角色修正。

审查者裁决（四个承重问题的修复）：
  1. 真实正常入口：TTHAFastAgent（真实 LLM backend）→ run_online_round
     完整入口（prepare → 候选池 → select → 执行）——不再用提示式选择；
  2. removal 对照：H1（含 learned Skill）vs H0（移除）同一真实入口
     @728——记录检索/入池/选择/执行/效用；
  3. 时间角色修正：D_patch @632 → held-in @680 → delayed @728
     （S3 曾先 held-in 728 后 delayed 680——逆时间，已修正）；
  4. 效用比较 U(H1)−U(H0) 相对 incumbent（repair_level_shift @728
     = +0.26115——skill 仅相对 raw 正向不足以称有益）。

装置（同 S3——已暴露数据）：NN5 / 632 失败 / 白名单 [impute_ar,
impute_ssm] / Fault Diagnosis Card（选择题结果复用 S3——不重跑）。

verdict（预注册）：
  NORMAL_ENTRY_ADOPTION_REMOVAL_PASS : H1 真实检索+选择+执行该 Skill、
    removal（H0）行为改变、U(H1) ≥ U(H0)（相对 incumbent）
  RETRIEVAL_GAP                      : Skill 在 snapshot 但未进入候选池
  ADHERENCE_SELECTION_GAP            : 检索到但 LLM select 未选择
  SCOPE_RISK_GAP                     : 选择后效用差（U(H1) < U(H0)）
  NO_CAUSAL_ROLE                     : removal 无行为差异
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_s3b_normal_entry_adoption.py
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
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
from run_v1_kdd2018_natural_slow_update import _request  # noqa: E402
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
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
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
REPORT_REL = E2 / "w1_s3b_normal_entry_adoption_report.json"
TASK_CONSUMER = "forecast|ridge|sMASE"
DOMAIN = "nn5"
SS, SD, TS, TD = core.TIMELINE[DOMAIN]  # 536, 584, 632, 680
# 时间角色修正（审查者）：632 D_patch → 680 held-in → 728 delayed
ORIGIN_HELDIN = TD          # 680
ORIGIN_DELAYED = 728        # 修正（原 S3 的 held-in 窗）
ENTRY_ORIGIN = 728          # 下一正常入口（= delayed 窗后）
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


def _nn5_evaluate(roster, values, compiled, config, *, origin):
    return v6._evaluate(roster, values, compiled, config, origin=origin)


def _failure_episode(origin: int, gain: float) -> Any:
    steps = [{"op": "repair_level_shift", "params": {}}]
    return build_episode(
        episode_id=f"nn5_s3b_fail_{origin}",
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
        local_status="EPISODE_ONLY", evidence_refs=["s3b_nn5"])


def _compile_manifest(h0, chosen_op: str) -> EditManifest:
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
        "experiment_id": "v1-s3b-normal-entry-adoption",
        "note": "S3b：真实正常入口 adoption/removal 对照（审查者 P0——"
                "S3 verdict 修正后补测）——development exposure——零新 "
                "Claim——不称 fresh 验证",
        "apparatus": {"domain": DOMAIN,
                      "time_roles": {"d_patch": TS, "heldin": ORIGIN_HELDIN,
                                     "delayed": ORIGIN_DELAYED,
                                     "entry": ENTRY_ORIGIN},
                      "whitelist": list(WHITELIST)},
    }

    g632 = _gain(roster, values, cfg, "repair_level_shift", TS)
    assert g632 is not None and abs(g632 - (-0.07892256471269943)) < 1e-6, \
        f"632 drift: {g632}"
    impute_gains = {op: _gain(roster, values, cfg, op, TS)
                    for op in WHITELIST}
    report["evidence"] = {"failure_632": g632, "d_patch_impute": impute_gains}

    # ---- 选择题复用 S3（已暴露——不重跑）----
    s3 = json.loads((E2 / "w1_s3_simplified_slow_path_report.json")
                    .read_text(encoding="utf-8"))
    s3_choice = (s3.get("slow_choice") or {}).get("answer") or {}
    chosen_op = next((op for op, pid in PATCH_IDS.items()
                      if pid == s3_choice.get("patch_id")), None)
    report["slow_choice"] = {"reused_from_s3": s3_choice,
                             "resolved_op": chosen_op}
    if chosen_op not in WHITELIST:
        report["verdict"] = "PROTOCOL_FAILURE"
        print("== verdict: PROTOCOL_FAILURE (no S3 choice)")
        return 0

    # ---- 门链（时间角色修正：632 → held-in 680 → delayed 728）----
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    sstore = SnapshotStore(root / ".s3b_store")
    controller = EditController(sstore, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    # 真实 LLM fast 入口一轮 = prepare+propose+select ≈ 3+ 次调用；
    # 两臂共享预算——max_calls=12（门链零 LLM——CompiledSlowAgent）
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=12)
    backend = AgictoChatCompletionsBackend(client=counter,
                                           base_url=smoke.BASE_URL)
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
            "failure_gain": g632, "d_patch_evidence": impute_gains}},
        "instruction": "Runtime-compiled edit.",
    }

    def _eval_support(steps, _mode):
        return _receipt_of(_gain(roster, values, cfg, steps[0][0], TS))

    ev = method.handle_feedback_support(episode, confirmed_cause="SKILL_LIBRARY_GAP", slow_agent=compiled, controller=controller, store=sstore,
        surface_catalog=[{"surface_id": "skill_library.entries/{skill_id}",
                          "operation": "ADD", "surface_type": "skill",
                          "allowed_operations": ["ADD"]}],
        card_builder=lambda e: card, evaluator=_eval_support,
        fast_features=dict(extract_public_features(
            series_arr[:TS], task_kind="forecast")))
    report["support_chain"] = ev
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
        lambda s, _m: _receipt_of(_gain(roster, values, cfg, s[0][0],
                                        ORIGIN_DELAYED)),
        episode_id=ev.get("episode_id"))
    report["delayed"] = dev
    if dev.get("stage") != "approved":
        report["verdict"] = "DELAYED_REJECTED"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: DELAYED_REJECTED")
        return 0

    snap = method._snapshot
    skill_ids = [s.skill_id for s in (snap.skills if snap is not None else [])]
    report["skill_persisted"] = {"skill_ids": skill_ids,
                                 "frozen_op": chosen_op2}
    assert SKILL_ID in skill_ids, "skill must be persisted"

    # ---- 真实正常入口（两臂同装置：run_online_round + 真实 LLM fast）----
    ex = ScopeExecutor(roster, values, cfg, evaluate_fn=_nn5_evaluate)
    vals = values

    def _entry(arm: str, method_arm: TTHAMethod) -> dict[str, Any]:
        r = run_online_round(
            method_arm, ex,
            _request(series_arr, vals, ENTRY_ORIGIN),
            vals,
            origin=ENTRY_ORIGIN, slow_agent=None, controller=None,
            store=None,
            card_builder=lambda e: {"pattern_id": "x",
                                    "observable_signature":
                                        {"task_kind": "forecast"}},
            round_name=f"s3b_{arm}_entry", budget=2, allow_slow=False,
            domain=f"nn5_s3b_{arm}", period=7,
            fast_features=dict(extract_public_features(
                series_arr[:ENTRY_ORIGIN], task_kind="forecast")),
            allow_fast_skill=True, runtime_prior_slot=False,
            allow_group_slow=False)
        probes = [(p["candidate_id"], p.get("gain"))
                  for p in r.actual_probed_programs]
        # 判定修复（2026-08-13）：候选 ID 用 skill_id 全名
        # （cand_skill_{SKILL_ID}）——非 op 名
        return {"probes": probes,
                "episodes_written": list(r.episode_ids),
                "skill_retrieved": any(
                    str(c).startswith("cand_skill_") for c, _ in probes),
                "skill_selected": any(
                    str(c) == f"cand_skill_{SKILL_ID}" for c, _ in probes),
                "executed_gains": [g for _c, g in probes if g is not None]}

    # H1：含 learned Skill 的真实入口（同 method——snapshot 已更新）
    h1 = _entry("H1", method)
    report["entry_H1"] = h1
    print("== H1 entry: " + json.dumps(h1, ensure_ascii=False), flush=True)

    # H0：移除 Skill（同一入口——无 skill 的 h0 method）
    method_h0 = TTHAMethod(TTHAFastAgent(TTHAAgentCore(
        backend, LocalPublicToolGateway(series_arr[:ENTRY_ORIGIN],
                                        task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL)),
        h0, ())
    h0_res = _entry("H0", method_h0)
    report["entry_H0"] = h0_res
    print("== H0 entry: " + json.dumps(h0_res, ensure_ascii=False), flush=True)

    # ---- 效用比较（相对 incumbent——@728 上 skill vs repair_level_shift）
    u_h1 = _gain(roster, values, cfg, chosen_op2, ENTRY_ORIGIN)
    u_h0 = _gain(roster, values, cfg, "repair_level_shift", ENTRY_ORIGIN)
    report["utility"] = {"U_H1": u_h1, "U_H0": u_h0,
                         "delta": (u_h1 - u_h0 if u_h1 is not None
                                   and u_h0 is not None else None)}

    # ---- 判定（预注册）----
    retrieved = h1["skill_retrieved"]
    selected = h1["skill_selected"]
    removal_changed = (h1["probes"] != h0_res["probes"])
    utility_ok = bool(report["utility"]["delta"] is not None
                      and report["utility"]["delta"] >= 0.0)
    if not retrieved:
        verdict = "RETRIEVAL_GAP"
    elif not selected:
        verdict = "ADHERENCE_SELECTION_GAP"
    elif not utility_ok:
        verdict = "SCOPE_RISK_GAP"
    elif not removal_changed:
        verdict = "NO_CAUSAL_ROLE"
    else:
        verdict = "NORMAL_ENTRY_ADOPTION_REMOVAL_PASS"
    report["verdict"] = verdict
    report["checks"] = {"retrieved": retrieved, "selected": selected,
                        "removal_changed": removal_changed,
                        "utility_ok": utility_ok}
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
