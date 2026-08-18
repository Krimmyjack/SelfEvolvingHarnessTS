"""P2_NATURAL_BATCH_MISSINGNESS（2026-08-13：真正目标验证——自然多
轨迹 Batch 是否形成 Target-local Skill 并在真实正常入口被采用。
预注册：docs/P2_NATURAL_BATCH_MISSINGNESS_PREREGISTRATION.md。
development exposure——零新 Claim。

链：冻结 H0 → outcome-blind adaptation block（NN5 20 series × origins
{600,632,680}——全算子探测收集 Action–Response）→ 自然 failure family
（≥2 独立 series——单条 → NO_BATCH_FAMILY）→ Fault Diagnosis Card
（确定性 + ≤2 Edit Intent）→ Agent 选择（1 LLM）→ Runtime 编译 →
Support（family 窗全 ≥M）→ held-in @712 → delayed @728 → H0/H1 真实
正常入口（首 series @728——同 DSL 同预算）→ 效用取实际执行 winner
（abstain=0）→ regret 报告。

verdict（预注册）：
  NATURAL_BATCH_LOCAL_SKILL_PASS / NO_BATCH_FAMILY / NO_COMMON_HEADROOM
  / SAFE_ABSTAIN / POLICY_NO_GAIN / PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_p2_natural_batch_missingness.py
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
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fault_cases import (  # noqa: E402
    GUARDS,
    selectable_fault_types,
)
from SelfEvolvingHarnessTS.methods.ttha.group_fault import group_first_faults  # noqa: E402
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
REPORT_REL = E2 / "w1_p2_natural_batch_missingness_report.json"
TASK_CONSUMER = "forecast|ridge|sMASE"
DOMAIN = "nn5"
CENSUS_ORIGINS = (600, 632, 680)
ORIGIN_HELDIN = 712
ORIGIN_DELAYED = 728
M = 0.005
OPS = ("repair_level_shift", "impute_ar", "impute_ssm",
       "impute_fft", "impute_ema", "impute_linear")
BASE_CACHE: dict[int, float] = {}


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


def _gain_series(sid: str, op: str, origin: int) -> tuple[float | None, int]:
    """per-source-series 评估（装置修复 v2，2026-08-13——用户核查）：
    完整冻结 roster（12 train + 8 eval）+ train_series_scope={sid}——
    只在该训练 series 应用算子、相同 8 eval series 测下游。
    返回 (gain, behavior_point_count)——None 表示评估不可用。"""
    compiled = v1.make_compiled(op, _default_params(op, 7))
    try:
        if origin not in BASE_CACHE:
            base = v6._evaluate(_ROSTER_FULL, _SERIES_VALUES, None,
                                _SERIES_CFG, origin=origin)
            BASE_CACHE[origin] = float(base["mean_smase"])
        cand = v6._evaluate(_ROSTER_FULL, _SERIES_VALUES, compiled,
                            _SERIES_CFG, origin=origin,
                            train_series_scope=frozenset({sid}))
        gain = BASE_CACHE[origin] - float(cand["mean_smase"])
        return gain, int(cand.get("behavior_point_count") or 0)
    except Exception:
        return None, 0


def _receipt_of(g: float | None) -> _Receipt:
    return _Receipt(g)


def _nn5_evaluate(roster, values, compiled, config, *, origin):
    return v6._evaluate(roster, values, compiled, config, origin=origin)


def _episode(series: str, origin: int, op: str, gain: float) -> Any:
    steps = [{"op": op, "params": {}}]
    return build_episode(
        episode_id=f"nn5_p2_{series[:8]}_{op}_{origin}",
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
        local_status="EPISODE_ONLY", evidence_refs=["p2_nn5"])


def _compile_manifest(h0, chosen_op: str, skill_id: str) -> EditManifest:
    steps = ((chosen_op, dict(_default_params(chosen_op, 7))),)
    return EditManifest(
        edit_id=f"replace-{chosen_op}-target",
        base_harness_sha=h0.harness_content_sha,
        target_pattern_id="p2-natural-batch",
        target_surface_id=f"skill_library.entries/{skill_id}",
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value={
            "schema_version": "skill-entry/1",
            "skill_id": skill_id,
            "skill_kind": "capability",
            "revision": 1,
            "body": "Frozen program steps: " + json.dumps(
                [{"op": o, "params": dict(p)} for o, p in steps]),
            "allowed_tools": [o for o, _p in steps],
        },
        observable_applicability=None,
        patch_id=f"patch-{chosen_op}",
        predicted_agent_behavior_change=(f"retrieve_skill:{chosen_op}",),
        predicted_data_effect=("natural_batch_skill",),
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
    global _SERIES_CFG, _SERIES_VALUES, _ROSTER_FULL
    _SERIES_CFG = cfg
    _SERIES_VALUES = values
    series_ids = [r["series_uid"] for r in roster]
    # 冻结 roster 划分（用户核查修复 v2）：12 train + 8 eval（roster 序）
    _ROSTER_FULL = ([{"series_uid": s, "role": "train"}
                     for s in series_ids[:12]]
                    + [{"series_uid": s, "role": "eval"}
                       for s in series_ids[12:]])
    TRAIN_SERIES = tuple(series_ids[:12])
    report: dict[str, Any] = {
        "experiment_id": "v1-p2-natural-batch-missingness",
        "note": "P2：自然 Batch Missingness——outcome-blind adaptation "
                "block → 自然 failure family → 受限 Diagnosis → Agent "
                "选择 → Runtime 编译 → held-in/delayed → H0/H1 真实入口"
                "（development exposure——零新 Claim——不称 fresh 验证）",
        "apparatus": {"domain": DOMAIN, "n_series": len(series_ids),
                      "census_origins": list(CENSUS_ORIGINS),
                      "heldin": ORIGIN_HELDIN, "delayed": ORIGIN_DELAYED,
                      "ops": list(OPS)},
    }

    report["apparatus"]["roster_split"] = {
        "n_train": 12, "n_eval": 8,
        "train": list(TRAIN_SERIES[:3]) + ["..."],
        "eval": series_ids[12:]}

    # ---- Phase 1：outcome-blind adaptation block（零 LLM——216 读数）----
    episodes = []
    series_of: dict[int, str] = {}
    census_log: dict[str, dict[str, Any]] = {}
    unavailable: list[str] = []
    n_behaviors = 0
    for sid in TRAIN_SERIES:
        census_log[sid] = {}
        for origin in CENSUS_ORIGINS:
            raw = {op: _gain_series(sid, op, origin) for op in OPS}
            gains = {op: g for op, (g, _b) in raw.items()}
            n_behaviors += sum(1 for _g, b in raw.values() if b > 0)
            census_log[sid][str(origin)] = gains
            for op, (g, _b) in raw.items():
                if g is None:
                    unavailable.append(f"{sid[:8]}@{origin}/{op}")
                elif g < -M:
                    ep = _episode(sid, origin, op, g)
                    episodes.append(ep)
                    series_of[id(ep)] = sid
    report["census"] = {"n_episodes_material_negative": len(episodes),
                        "n_readings": 12 * 3 * len(OPS),
                        "n_unavailable": len(unavailable),
                        "unavailable_sample": unavailable[:10],
                        "n_behavior_changed_readings": n_behaviors,
                        "log": census_log}
    print(f"== census: {len(episodes)} material negative episodes; "
          f"{len(unavailable)} unavailable readings", flush=True)
    # 用户核查修复：关键读数必须非空且程序确实改变数据——否则直接
    # PROTOCOL_FAILURE（不把"评估不可用"误判为"无失败"）
    if unavailable or n_behaviors == 0:
        report["verdict"] = "EVALUATION_UNAVAILABLE"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: EVALUATION_UNAVAILABLE")
        return 0

    # ---- Phase 2：自然 failure family（≥2 独立 series）----
    groups = group_first_faults(episodes, min_group=2)
    families = []
    for g in groups:
        series = sorted({series_of[id(e)] for e in g["episodes"]})
        if len(series) < 2:
            continue
        families.append({
            "workflow": g["workflow"], "sign": g["sign"],
            "n_episodes": len(g["episodes"]), "n_series": len(series),
            "series": series,
            "episodes": [{"series": series_of[id(e)], "origin":
                          int((e.context_summary or {})
                              .get("support_origin") or 0),
                          "gain": float((e.support_response or {})
                                        .get("gain") or 0.0)}
                         for e in g["episodes"]]})
    families.sort(key=lambda f: (-f["n_series"], f["n_episodes"]))
    report["families"] = families
    print("== families: " + json.dumps(families, ensure_ascii=False,
                                       default=str), flush=True)
    if not families:
        report["verdict"] = "NO_BATCH_FAMILY"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: NO_BATCH_FAMILY")
        return 0
    family = families[0]

    # ---- Phase 3：Fault Diagnosis Card（确定性）----
    failed_op = family["workflow"]
    candidates = [op for op in OPS if op != failed_op and op != "identity"]
    fam_episodes = family["episodes"]
    headroom = {}
    for alt in candidates:
        raw = [_gain_series(e["series"], alt, e["origin"])
               for e in fam_episodes]
        per = [g for g, _b in raw]
        headroom[alt] = {
            "gains": per,
            "common_positive": bool(per and all(
                g is not None and g >= M for g in per))}
    pos_cands = [alt for alt, h in headroom.items() if h["common_positive"]]
    report["headroom"] = headroom
    print("== headroom: " + json.dumps(headroom, ensure_ascii=False,
                                       default=str), flush=True)
    evidence = {
        "task_contract_conflict": None, "diagnosis_contradiction": None,
        "headroom": {alt: h["common_positive"]
                     for alt, h in headroom.items()},
        "supply_exhausted": False,
        "winner_probed": None, "agent_chosen": failed_op,
        "support_positive": None, "delayed_negative": None,
    }
    selectable = selectable_fault_types(evidence)
    report["diagnosis"] = {"failed_op": failed_op,
                           "selectable": selectable,
                           "common_positive_candidates": pos_cands}
    if not pos_cands:
        report["verdict"] = "NO_COMMON_HEADROOM"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: NO_COMMON_HEADROOM")
        return 0
    # ≤2 Edit Intent（headroom 最高的两个）
    intents = sorted(pos_cands,
                     key=lambda a: -max(g for g in headroom[a]["gains"]
                                        if g is not None))[:2]
    report["diagnosis"]["intents"] = intents

    # ---- Phase 4：Agent 选择（1 LLM）----
    ev_lines = "\n".join(
        f"  - {alt}: " + ", ".join(
            f"@{e['origin']}={g:.4f}" for e, g in
            zip(fam_episodes, headroom[alt]["gains"]))
        for alt in intents)
    fault_label = (selectable[0] if selectable
                   else "NO_ACTIONABLE_FAULT")
    option_lines = "\n".join(
        f"  {chr(65 + i)}. ADD_TARGET_LOCAL_SKILL({op}) "
        f"[patch_id=patch-{op}]" for i, op in enumerate(intents))
    card_text = (
        "FAULT DIAGNOSIS CARD\n"
        f"Detected failure: {failed_op} material negative at "
        + ", ".join(f"{e['series'][:8]}@{e['origin']} ({e['gain']:.4f})"
                    for e in fam_episodes)
        + f"\nRuntime fault type (deterministic): {fault_label}\n"
        "Evidence (D_patch, public Action-Response on the failure "
        "windows):\n" + ev_lines +
        "\n\nAllowed modifications (choose exactly one):\n"
        + option_lines + "\n  C. ABSTAIN\n\n"
        'Output JSON: {"edit_intent": "ADD_TARGET_LOCAL_SKILL|ABSTAIN", '
        '"patch_id": "<one of the patch ids>|null", "reason": "...", '
        '"expected_change": "..."}\n- You do NOT write the manifest — the '
        "Runtime compiles it. held-in and delayed outcomes are hidden.")
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=14)
    resp = counter.chat.completions.create(
        model=smoke.MODEL,
        messages=[{"role": "user", "content": card_text}])
    choice = _parse_choice(str((resp.choices[0].message.content) or ""))
    report["agent_choice"] = {"answer": choice, "llm_calls": counter.calls}
    print("== choice: " + json.dumps(choice, ensure_ascii=False,
                                     default=str), flush=True)
    intent = str((choice or {}).get("edit_intent") or "")
    patch_id = (choice or {}).get("patch_id")
    if intent != "ADD_TARGET_LOCAL_SKILL" or patch_id not in \
            {f"patch-{op}" for op in intents}:
        report["verdict"] = "SAFE_ABSTAIN"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: SAFE_ABSTAIN")
        return 0
    chosen_op = str(patch_id).replace("patch-", "")
    report["agent_choice"]["resolved_op"] = chosen_op

    # ---- Phase 5：Runtime 编译 + 门链 ----
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    sstore = SnapshotStore(root / ".p2_store")
    controller = EditController(sstore, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    backend = AgictoChatCompletionsBackend(client=counter,
                                           base_url=smoke.BASE_URL)
    skill_id = f"p2_{chosen_op}_skill"
    compiled = CompiledSlowAgent(_compile_manifest(h0, chosen_op, skill_id))
    fam_first = family["series"][0]
    series_arr = values[fam_first]
    method = TTHAMethod(TTHAFastAgent(TTHAAgentCore(
        backend, LocalPublicToolGateway(series_arr[:600],
                                        task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL)),
        h0, ())
    eps = [_episode(e["series"], e["origin"], failed_op, e["gain"])
           for e in fam_episodes]
    card = {
        "pattern_id": "p2-natural-batch",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "context": {},
        "workflow": {"steps": [{"op": failed_op, "params": {}}]},
        "typed_patch_options": [
            {"patch_id": f"patch-{op}",
             "program_steps": [{"op": op,
                                "params": dict(_default_params(op, 7))}]}
            for op in intents],
        "facts": {"fault_diagnosis_card": {
            "fault_type": selectable[0] if selectable
            else "NO_ACTIONABLE_FAULT",
            "failed_op": failed_op, "d_patch_evidence": headroom}},
        "instruction": "Runtime-compiled edit.",
    }

    def _eval_support(steps, _mode):
        g, _b = _gain_series(fam_first, steps[0][0], 632)
        return _receipt_of(g)

    ev = method.handle_feedback_support(eps[0], confirmed_cause="SKILL_LIBRARY_GAP", slow_agent=compiled, controller=controller, store=sstore,
        surface_catalog=[{"surface_id": "skill_library.entries/{skill_id}",
                          "operation": "ADD", "surface_type": "skill",
                          "allowed_operations": ["ADD"]}],
        card_builder=lambda e: card, evaluator=_eval_support,
        fast_features=dict(extract_public_features(
            series_arr[:600], task_kind="forecast")))
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
    heldin, _b = _gain_series(fam_first, chosen_op, ORIGIN_HELDIN)
    report["heldin"] = {"origin": ORIGIN_HELDIN, "gain": heldin}
    if heldin is None or heldin < -M:
        report["verdict"] = "HELDIN_REJECTED"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: HELDIN_REJECTED")
        return 0
    dev = method.handle_feedback_delayed(
        lambda s, _m: _receipt_of(_gain_series(fam_first, s[0][0],
                                               ORIGIN_DELAYED)[0]),
        episode_id=ev.get("episode_id"))
    report["delayed"] = dev
    if dev.get("stage") != "approved":
        report["verdict"] = "DELAYED_REJECTED"
        REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                         indent=2, default=str) + "\n",
                              encoding="utf-8")
        print("== verdict: DELAYED_REJECTED")
        return 0

    # ---- Phase 6：H0/H1 真实正常入口（首 series @728）----
    ex = ScopeExecutor(roster, values, cfg, evaluate_fn=_nn5_evaluate)

    def _entry(arm: str, method_arm: TTHAMethod) -> dict[str, Any]:
        r = run_online_round(
            method_arm, ex,
            _request(series_arr, values, ORIGIN_DELAYED), values,
            origin=ORIGIN_DELAYED, slow_agent=None, controller=None,
            store=None,
            card_builder=lambda e: {"pattern_id": "x",
                                    "observable_signature":
                                        {"task_kind": "forecast"}},
            round_name=f"p2_{arm}_entry", budget=2, allow_slow=False,
            domain=f"nn5_p2_{arm}", period=7,
            fast_features=dict(extract_public_features(
                series_arr[:ORIGIN_DELAYED], task_kind="forecast")),
            allow_fast_skill=True, runtime_prior_slot=False,
            allow_group_slow=False)
        probes = [(p["candidate_id"], p.get("gain"))
                  for p in r.actual_probed_programs]
        return {"probes": probes,
                "skill_retrieved": any(
                    str(c).startswith("cand_skill_") for c, _ in probes),
                "skill_selected": any(
                    str(c) == f"cand_skill_{skill_id}" for c, _ in probes),
                "executed_gains": [g for _c, g in probes if g is not None]}

    h1 = _entry("H1", method)
    report["entry_H1"] = h1
    print("== H1: " + json.dumps(h1, ensure_ascii=False), flush=True)
    method_h0 = TTHAMethod(TTHAFastAgent(TTHAAgentCore(
        backend, LocalPublicToolGateway(series_arr[:ORIGIN_DELAYED],
                                        task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL)),
        h0, ())
    h0_res = _entry("H0", method_h0)
    report["entry_H0"] = h0_res
    print("== H0: " + json.dumps(h0_res, ensure_ascii=False), flush=True)

    # ---- Phase 7：效用 + regret + verdict ----
    u_h1 = max(h1["executed_gains"]) if h1["executed_gains"] else 0.0
    u_h0 = max(h0_res["executed_gains"]) if h0_res["executed_gains"] else 0.0
    best_unselected = max((max(g for g in h["gains"] if g is not None)
                           if any(g is not None for g in h["gains"])
                           else -1.0)
                          for alt, h in headroom.items()
                          if alt != chosen_op) if headroom else -1.0
    report["utility"] = {"U_H1": u_h1, "U_H0": u_h0,
                         "delta_over_abstain": u_h1 - u_h0,
                         "regret_best_unselected": best_unselected}
    if h1["skill_selected"] and u_h1 >= u_h0:
        verdict = "NATURAL_BATCH_LOCAL_SKILL_PASS"
    elif h1["skill_selected"]:
        verdict = "POLICY_NO_GAIN"
    else:
        verdict = "POLICY_NO_GAIN"
    report["verdict"] = verdict
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
