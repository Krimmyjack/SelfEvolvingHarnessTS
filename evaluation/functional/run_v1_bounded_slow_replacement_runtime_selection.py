"""BOUNDED_SLOW_REPLACEMENT_RUNTIME_SELECTION（用户裁决 2026-08-10）。

优先级调整：Program-effect grounding 降为后续效率优化。本实验把
replacement 选择权交给确定性 Runtime，LLM 只做有界候选 supplier。

唯一假设：
> Slow Agent 能在两个有界 replacement 提案内覆盖至少一个有效候选，
> Runtime 能用两个 Support receipts 选择有效修改，并经 delayed、Skill
> 和正常入口完成更新。

案例（复用 P1 development case，不泄露 headroom）：
  GEFCom 904：incumbent denoise_median→outlier_iqr = −0.13021；
  A-only=0.0；B-only=−0.13021。候选池 {C1=winsorize（实际双正，
  不告知）, C2=outlier_mad（实际无 headroom，不告知）}。

流程：
  1. Slow Agent 两次调用产生两个不同候选（调用 2 只告知已提案算子名，
     不提供其 gain）；重复 → CANDIDATE_DIVERSITY_FAILED
  2. Runtime 实测两个候选（两个 Support 计入真实 budget）；冻结选择
     规则：非法拒绝 → gain≥MATERIAL 接纳 → 双正取高 → 相同按
     canonical ID → 全非正 abstain
  3. 两合法候选写 Experience Episode；未选中仅 EPISODE_ONLY；
     赢家 LOCAL_DRAFT → delayed 只开赢家 → ≥M → LOCAL_ACTIVE；
     <M → RESTRICTED/CONFLICT 不得接纳
  4. 赢家 Skill 写入（ADD skill_library.entries/{skill_id}，frozen
     program=赢家 2-step）→ 正常 TTHAMethod.prepare → chosen=
     cand_skill_* → 执行 → remove-skill 对照回退

Verdict（预注册）：
  BOUNDED_SLOW_REPLACEMENT_PASS / CANDIDATE_DIVERSITY_FAILED /
  NO_VALID_REPLACEMENT_SUPPLIED / SUPPORT_SELECTION_FAILED /
  DELAYED_REJECTED / FAST_ADOPTION_FAILED / REMOVAL_CONTROL_NO_FLIP

用法：
  python evaluation/functional/run_v1_bounded_slow_replacement_runtime_selection.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController, SurfaceRegistry, FaultRouter)
from SelfEvolvingHarnessTS.runtime.agent_backend import AgictoChatCompletionsBackend  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA  # noqa: E402

PERIOD = 24
HORIZON = 48
MATERIAL = resolver.MATERIAL_THRESHOLD  # 0.005

# ---- 案例冻结（同 P1；数值承重由运行时 recheck）----
CASE = {
    "domain": "gefcom2012_load",
    "origin": 904,
    "delayed": 952,
    "a": "denoise_median",
    "b": "outlier_iqr",
    "c1": "winsorize",
    "c2": "outlier_mad",
    "gain_AB": -0.1302066421576531,
    "delayed_AB": -0.06125772464287427,
    "gain_A_only": 0.0,
    "gain_B_only": -0.1302066421576531,
    "support_AC1": 0.4000053662007894,
    "delayed_AC1": 0.2572125429345453,
}

REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_bounded_slow_replacement_runtime_selection_report.json")

SURFACE_CATALOG = [
    {
        "surface_id": "skill_library.entries/{skill_id}",
        "operation": "ADD",
        "surface_type": "skill",
        "allowed_operations": ["ADD"],
    },
]

OPS_ALL = tuple(o for o in (
    "denoise_median", "hampel_filter", "impute_ar", "impute_ema",
    "impute_fft", "impute_linear", "impute_ssm", "outlier_iqr",
    "outlier_mad", "period_complete", "period_median_complete",
    "repair_level_shift", "resample_uniform", "winsorize"))


def _contract(op: str) -> dict[str, object]:
    """公开 Operator 契约（LLM 可见）：参数 defaults + category/stage/tags。"""
    meta = OPERATOR_METADATA.get(op) or {}
    return {
        "op": op,
        "params": dict(wiring.contract_params(op, PERIOD)),
        "category": meta.get("category"),
        "stage": meta.get("stage"),
        "tags": list(meta.get("tags") or ()),
    }


def _steps_of(ops: tuple[str, ...]) -> tuple:
    return tuple((op, dict(wiring.contract_params(op, PERIOD))) for op in ops)


def build_card(values: Mapping[str, Any], case: Mapping[str, Any],
               prior_candidate: str | None = None) -> dict[str, object]:
    """信息墙 card（同 P1）：只含 incumbent workflow + Context + failure
    数值 + 可编辑 step index + 两候选契约 + objective + budget。prior_
    candidate 只在调用 2 时给出（已提案算子名，无 gain/验证信息）。"""
    pa = dict(wiring.contract_params(case["a"], PERIOD))
    pb = dict(wiring.contract_params(case["b"], PERIOD))
    ctx = dict(resolver.window_context(values, case["origin"], PERIOD))
    ctx["bound_period"] = float(PERIOD)
    c1 = _contract(case["c1"])
    c2 = _contract(case["c2"])
    card = {
        "pattern_id": f"{case['domain']}-{case['a']}-{case['b']}-replace-step",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {
            "steps": [{"op": case["a"], "params": pa},
                      {"op": case["b"], "params": pb}],
            "scope": "training_windows_only",
            "evaluator": "v6._evaluate (per-training-window, cohort Ridge sMASE)",
        },
        "facts": {
            "window_context": ctx,
            "editable_step_index": 1,
            "replacement_candidates": [c1, c2],
            "task_objective": ("forecast; downstream Support metric = "
                               "cohort-Ridge sMASE over the declared "
                               "training_windows_only scope"),
            "feedback_budget": ("two bounded replacement proposals; "
                                "each proposal gets one formal call and "
                                "one already-existing schema retry"),
        },
        "counterfactual_support": {
            "A_then_B_support_gain": case["gain_AB"],
            "A_then_B_delayed_gain": case["delayed_AB"],
        },
        "instruction": (
            "A two-step workflow was applied at the support decision point "
            "and produced the Support/Delayed outcomes above (negative "
            "material impact relative to identity). The second step "
            "(index 1) is the candidate replacement position: replace it "
            "with one of the two replacement candidates, keeping the "
            "first step (index 0) unchanged. Propose exactly one edit: "
            "ADD one new capability skill whose frozen program is the "
            "repaired two-step workflow (steps JSON after the line "
            "'Frozen program steps:'). Fill falsification_condition so a "
            "deterministic replay can accept or reject it later. You do "
            "not approve your own edit. If the evidence is insufficient, "
            "declare no_proposal — abstaining is valid."
        ),
    }
    if prior_candidate is not None:
        card["instruction"] += (
            f"\nNote: the first replacement proposal has already been "
            f"recorded (replacement operator: {prior_candidate}). "
            f"Propose a DIFFERENT replacement candidate — do not repeat "
            f"the already-proposed operator. The other candidate remains "
            f"available. You must still produce a valid two-step frozen "
            f"program."
        )
        card["facts"]["prior_candidate"] = prior_candidate
    return card


def _parse_frozen(body: str) -> list[dict[str, Any]] | None:
    marker = "Frozen program steps:"
    idx = body.find(marker)
    if idx < 0:
        return None
    rest = body[idx + len(marker):].strip()
    try:
        arr = json.loads(rest)
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    return [{"op": str(s["op"]), "params": dict(s.get("params") or {})}
            for s in arr if isinstance(s, Mapping) and s.get("op")]


def structural_preflight(manifest: Any,
                         case: Mapping[str, Any]) -> dict[str, Any]:
    """结构预检：ADD 面白名单 + schema + 2-step 冻结 Program（steps[0]==A
    且 steps[1]∈{C1,C2}）。"""
    if manifest is None:
        return {"stage": "no_proposal", "preflight": "REJECTED"}
    t = str(manifest.target_surface_id)
    nv = manifest.new_value or {}
    skill_id = str(nv.get("skill_id") or "")
    instantiated = t.replace("{skill_id}", skill_id) if skill_id else t
    ok_surface = instantiated.startswith("skill_library.entries/")
    ok_op = manifest.operation.value == "ADD"
    ok_schema = nv.get("schema_version") == "skill-entry/1"
    body = str(nv.get("body") or "")
    frozen = _parse_frozen(body)
    ok_steps = False
    if frozen is not None:
        ok_steps = (len(frozen) == 2
                    and str(frozen[0].get("op")) == case["a"]
                    and str(frozen[1].get("op")) in (case["c1"], case["c2"]))
    selected = str(frozen[1].get("op")) if ok_steps else None
    return {
        "stage": "manifest",
        "preflight": ("ACCEPTED" if (ok_surface and ok_op and ok_schema
                                     and frozen is not None and ok_steps
                                     and skill_id) else "REJECTED"),
        "target_surface_id": instantiated,
        "skill_id": skill_id,
        "operation": manifest.operation.value,
        "frozen_program": frozen,
        "selected_replacement": selected,
        "surface_template_instantiated_by_harness": bool(t != instantiated),
        "falsification_condition": list(manifest.falsification_condition or ()),
    }


def _declare_winner(candidates: Mapping[str, float]) -> str | None:
    """冻结选择规则（用户裁决）：gain≥MATERIAL 才接纳；双正取高；
    相同按 canonical ID（字母序）；全非正 → abstain(None)。"""
    valid = {c: g for c, g in candidates.items()
             if g is not None and g >= MATERIAL}
    if not valid:
        return None
    return max(sorted(valid), key=lambda c: valid[c])


def main() -> int:
    root = PROJECT_ROOT
    case = dict(CASE)
    for _k in ("domain", "origin", "delayed", "a", "b", "c1", "c2",
               "gain_AB", "delayed_AB", "gain_A_only", "gain_B_only",
               "support_AC1", "delayed_AC1"):
        assert case[_k] is not None, f"案例字段 {_k} 未冻结"
    checks: dict[str, Any] = {"case": dict(case)}
    verdict = "INCONCLUSIVE"

    # ---- 数据 + executor + 案例承重复测（同 P1）----
    config = dict(v6.DATASET_CONFIGS["gefcom"])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)

    r_ab = executor.evaluate(_steps_of((case["a"], case["b"])), case["origin"])
    checks["recheck_gain_AB"] = (float(r_ab.gain) if r_ab.gain is not None
                                 else None)
    checks["recheck_passed_AB"] = bool(r_ab.verification.passed)
    r_a = executor.evaluate(_steps_of((case["a"],)), case["origin"])
    checks["recheck_gain_A_only"] = (float(r_a.gain) if r_a.gain is not None
                                     else None)
    r_ac1 = executor.evaluate(_steps_of((case["a"], case["c1"])),
                              case["origin"])
    checks["recheck_gain_AC1"] = (float(r_ac1.gain) if r_ac1.gain is not None
                                  else None)
    r_ac1d = executor.evaluate(_steps_of((case["a"], case["c1"])),
                               case["delayed"])
    checks["recheck_delayed_AC1"] = (float(r_ac1d.gain)
                                     if r_ac1d.gain is not None else None)
    headroom_ok = bool(
        checks["recheck_passed_AB"]
        and checks["recheck_gain_AB"] is not None
        and checks["recheck_gain_AB"] < -MATERIAL
        and checks["recheck_gain_A_only"] is not None
        and checks["recheck_gain_A_only"] < MATERIAL
        and checks["recheck_gain_AC1"] is not None
        and checks["recheck_gain_AC1"] >= MATERIAL
        and checks["recheck_delayed_AC1"] is not None
        and checks["recheck_delayed_AC1"] >= MATERIAL)
    checks["case_headroom_confirmed"] = headroom_ok
    print(f"== case recheck: AB={checks['recheck_gain_AB']} "
          f"AC1={checks['recheck_gain_AC1']} "
          f"delayed_AC1={checks['recheck_delayed_AC1']}")
    if not headroom_ok:
        verdict = "NO_VALID_REPLACEMENT_SUPPLIED"  # 案例结构失效（防御）
        _write_report(root, case, checks, verdict, None)
        return 0

    # ---- 真实 Slow Agent：两次有界提案 ----
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        verdict = "INCONCLUSIVE"
        _write_report(root, case, checks, verdict, None)
        return 0
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=4)  # 2 候选 × (1 正式 + 1 retry)
    backend = AgictoChatCompletionsBackend(client=counter,
                                           base_url=smoke.BASE_URL)
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(series0[:case["delayed"]],
                               task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL)
    slow = TTHASlowAgent(core)

    # 有界提案：最多 2 个成功候选；LLM abstain 不消耗候选预算（既有语义），
    # 在 4 次 LLM 调用预算内重试。proposal 2 的 prior = 已提案候选（只告知
    # 算子名，不提供 gain/验证信息）。
    proposals: list[dict[str, Any]] = []
    raw_manifests: list[Any] = []
    abstains = 0
    while len(proposals) < 2 and counter.calls < 4:
        prior = (proposals[0].get("selected") if proposals else None)
        card = build_card(values, case, prior_candidate=prior)
        manifest = None
        try:
            manifest = slow.propose_edit(
                card, SURFACE_CATALOG, h0,
                manifest_preflight=lambda m: None,
                allowed_operator_contracts=(),
                task_context=None)
        except RuntimeError as exc:
            print(f"== budget hard stop: {exc}")
            checks["budget_hard_stop"] = True
            break
        preflight = structural_preflight(manifest, case)
        if preflight["preflight"] != "ACCEPTED" or \
                preflight["selected_replacement"] is None:
            abstains += 1
            print(f"== proposal {len(proposals) + 1}: abstain/invalid "
                  f"(llm_calls={counter.calls})")
            continue
        proposals.append({
            "call": len(proposals) + 1,
            "manifest": {
                "edit_id": manifest.edit_id,
                "target_surface_id": preflight["target_surface_id"],
                "skill_id": preflight["skill_id"],
                "frozen_program": preflight["frozen_program"],
                "selected_replacement": preflight["selected_replacement"]},
            "preflight_ok": True,
            "selected": preflight["selected_replacement"],
        })
        raw_manifests.append(manifest)
        print(f"== proposal {len(proposals)}: "
              f"{proposals[-1]['selected']} "
              f"(llm_calls={counter.calls})")
    checks["abstain_count"] = abstains

    checks["llm_calls_le_4"] = counter.calls <= 4
    checks["llm_calls"] = counter.calls

    # ---- 候选有效性 + diversity ----
    sel1 = proposals[0].get("selected")
    sel2 = proposals[1].get("selected")
    checks["diversity_ok"] = bool(
        sel1 is not None and sel2 is not None and sel1 != sel2)
    if not checks["diversity_ok"]:
        verdict = "CANDIDATE_DIVERSITY_FAILED"
        _write_report(root, case, checks, verdict, proposals)
        return 0

    # ---- Runtime 实测两个候选（两个 Support 计入真实 budget）----
    support: dict[str, float] = {}
    episodes: dict[str, Any] = {}
    for sel in (sel1, sel2):
        steps = _steps_of((case["a"], sel))
        r = executor.evaluate(steps, case["origin"])
        g = (float(r.gain) if r.gain is not None else None)
        support[sel] = g if (g is not None and r.verification.passed) \
            else -float("inf")
        # 写 Experience Episode（两个合法候选都写）
        ep = tll.write_target_episode(
            domain=case["domain"], op=f"{case['a']}_{sel}",
            episode_id_suffix=f"_bounded_{sel}",
            program_steps=[{"op": case["a"], "params": dict(
                wiring.contract_params(case["a"], PERIOD))},
                {"op": sel, "params": dict(
                    wiring.contract_params(sel, PERIOD))}],
            support_gain=(g if g is not None else 0.0),
            delayed_gain=None,
            support_context=dict(resolver.window_context(
                values, case["origin"], PERIOD)))
        episodes[sel] = ep
        print(f"== support {sel}: gain={g} "
              f"episode={ep.episode_id} relation={ep.relation}")

    winner = _declare_winner(support)
    checks["runtime_winner"] = winner
    checks["support_gains"] = {
        c: (None if support[c] == -float("inf") else support[c])
        for c in (sel1, sel2)}
    if winner is None:
        verdict = "NO_VALID_REPLACEMENT_SUPPLIED"
        _write_report(root, case, checks, verdict, proposals)
        return 0

    # ---- 赢家 delayed（只打开赢家）----
    winner_steps = _steps_of((case["a"], winner))
    rd = executor.evaluate(winner_steps, case["delayed"])
    gain_d = (float(rd.gain) if rd.gain is not None else None)
    checks["delayed_positive"] = bool(
        gain_d is not None and gain_d >= MATERIAL)
    print(f"== delayed {winner}: gain={gain_d}")
    # 更新赢家 Episode 的 delayed 状态（原位）
    episodes[winner] = tll.update_delayed_status(
        episodes[winner], gain_d if gain_d is not None else 0.0,
        delayed_context=dict(resolver.window_context(
            values, case["delayed"], PERIOD)))
    checks["winner_episode_relation_after_delayed"] = \
        episodes[winner].relation
    if not checks["delayed_positive"]:
        verdict = "DELAYED_REJECTED"
        _write_report(root, case, checks, verdict, proposals)
        return 0

    # ---- 赢家 Skill 写入（apply_to_fork；复用 LLM 原始 manifest 对象，
    #      确定性契约修复同 P1：surface 模板实例化 + required deps）----
    win_idx = 0 if proposals[0]["selected"] == winner else 1
    raw_win = raw_manifests[win_idx]
    winner_manifest = proposals[win_idx]["manifest"]
    reg = SurfaceRegistry()
    resolved = reg.resolve(winner_manifest["target_surface_id"])
    snapshot_deps = dict(h0.dependency_shas)
    declared_dep = {
        key: snapshot_deps[key]
        for key in resolved.definition.required_dependency_keys
        if key in snapshot_deps
    }
    manifest_apply = dataclasses.replace(
        raw_win,
        target_surface_id=winner_manifest["target_surface_id"],
        dependency_precondition_shas=declared_dep)
    store = SnapshotStore(root)
    parent = store.materialize(h0)
    controller = EditController(store, surfaces=reg, router=FaultRouter())
    try:
        receipt = controller.apply_to_fork(
            parent, manifest_apply, confirmed_cause="SKILL_LIBRARY_GAP")
    except Exception as exc:
        print(f"== apply_to_fork EXC: {type(exc).__name__}: {exc}")
        checks["apply_exception"] = f"{type(exc).__name__}: {exc}"
        verdict = "NO_VALID_REPLACEMENT_SUPPLIED"
        _write_report(root, case, checks, verdict, proposals)
        return 0
    checks["controller_applied"] = True
    print(f"== applied: edit={receipt.edit_id}")

    # ---- 正常入口采用 + remove-skill 对照 ----
    skill_cand = f"cand_skill_{winner_manifest['skill_id']}"
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            sealed.SealedProbeBackend(explore=True, operators=OPS_ALL),
            LocalPublicToolGateway(series0[:case["delayed"]],
                                   task_kind="forecast"))),
        receipt.candidate_snapshot.snapshot, ())
    method.bind_round_data(series0[:case["delayed"]], task_kind="forecast")
    obs = dict(resolver.window_context(values, case["delayed"], PERIOD))
    obs["bound_period"] = float(PERIOD)
    r2 = method.prepare(sealed._request(series0, values, case["delayed"]))
    chosen = method.last_trace.chosen_candidate_id
    adopted = chosen == skill_cand
    r2_gain = None
    exec_steps: list[tuple[str, dict]] | None = None
    if r2.program is not None:
        exec_steps = list(r2.program.execution_steps())
        rr2 = executor.evaluate(tuple(exec_steps), case["delayed"])
        r2_gain = (float(rr2.gain) if rr2.gain is not None else None)
    checks["next_round_actual_adoption"] = bool(adopted)
    checks["next_round_executed"] = r2_gain is not None
    checks["executed_program_matches_frozen"] = bool(
        exec_steps is not None
        and [{"op": o, "params": dict(p)} for o, p in exec_steps]
        == winner_manifest["frozen_program"])
    print(f"== @{case['delayed']}: chosen={chosen} adopted={adopted} "
          f"gain={r2_gain}")

    method_ctrl = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            sealed.SealedProbeBackend(explore=True, operators=OPS_ALL),
            LocalPublicToolGateway(series0[:case["delayed"]],
                                   task_kind="forecast"))),
        h0, ())
    method_ctrl.bind_round_data(series0[:case["delayed"]],
                                task_kind="forecast")
    method_ctrl.prepare(sealed._request(series0, values, case["delayed"]))
    chosen_ctrl = method_ctrl.last_trace.chosen_candidate_id
    checks["removal_changes_action"] = bool(chosen_ctrl != skill_cand)
    print(f"== @{case['delayed']} ctrl(no skill): chosen={chosen_ctrl}")

    try:
        store.discard_fork(parent.root)
    except ValueError:
        pass

    passed = all(checks.get(k) is True for k in (
        "case_headroom_confirmed", "diversity_ok", "llm_calls_le_4",
        "controller_applied", "delayed_positive",
        "next_round_actual_adoption", "executed_program_matches_frozen",
        "removal_changes_action"))
    if passed:
        verdict = "BOUNDED_SLOW_REPLACEMENT_PASS"
    elif adopted and not checks.get("removal_changes_action"):
        verdict = "REMOVAL_CONTROL_NO_FLIP"
    else:
        verdict = "FAST_ADOPTION_FAILED"

    print(f"== verdict: {verdict}")
    _write_report(root, case, checks, verdict, proposals)
    return 0


def _write_report(root: Path, case: Mapping[str, Any],
                  checks: dict[str, Any], verdict: str,
                  proposals: Any) -> None:
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-bounded-slow-replacement-runtime-selection",
        "case": dict(case),
        "proposals": proposals,
        "checks": checks,
        "verdict": verdict,
        "llm_api_call_count": checks.get("llm_calls"),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")


if __name__ == "__main__":
    raise SystemExit(main())
