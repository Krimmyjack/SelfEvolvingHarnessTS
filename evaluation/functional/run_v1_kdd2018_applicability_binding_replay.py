"""KDD_CUP_2018_APPLICABILITY_BINDING_REPLAY（P4.3，用户裁决 2026-08-11）。

development replay（用户裁决：**先不用新 virgin 数据**——在已暴露的
P4.2 KDD @984 上验证修复机制有效；**通过只证修复机制有效，不得追溯把旧
实验改称 fresh PASS**）。

修复（A+B，用户裁决 first fault = Applicability-to-Observation Binding
Gap）：
  A：observable_applicability 由 Runtime 从 Failure Card 的
     observable_signature（公开 Observation）机器生成——Slow Agent 不得
     额外编造特征（method.py `_applicability_from_card`——B2 同款 Runtime
     所有权；manifest 级字段同步——controller 校验两者一致）。
  B：批准前机械可达性检查（method.py `_applicability_reachable`）——
     ①特征 ∈ card observable_signature ②∈ 当前 Fast 入口特征空间
     ③当前公开 Context 下可检索（evaluate_applicability）；不满足 →
     ACTION_UNAVAILABLE（不写 active snapshot）；fast_features 缺失 →
     fail-safe 拒绝。

Section 1 门机制测试（零 LLM / 零新数据）：
  - A 单测：task_kind signature → {all:[task_kind==forecast]}；空 signature
    → {"const": True}；
  - B 单测：card 绑定条件 @984 可检索 ✓；LLM 编造条件（clipping_probe_
    direction==negative）→ 拒绝（not_in_card_signature /
    not_retrievable_in_current_context）；
  - handle_feedback_support 集成（stub slow_agent 返回带编造 applicability
    的 EditManifest + stub evaluator——零 LLM/零数据）：clip 签名 card →
    applicability_unreachable（不写 snapshot）；task_kind 签名 card →
    pending（A 绑定后 B 通过）。

Section 2 dev replay（已暴露 @984；sealed 确定性同 P4.1 装置；零 LLM）：
  装载 P4.1 已批准 snapshot（96f83039...）→ 按 A 修正该 skill 的
  applicability（task_kind==forecast——P4.1 card 的 observable_signature）
  → R1 正常 prepare：Skill 被检索（view + cand_skill_* 入池）；
  R2 chosen/program = outlier_mad；R3 实际执行一次（verifier + gain @984；
  delayed@1032 记录）；R4 removal 对照（原 h0 同轮）：候选消失、行为恢复。

verdict（预注册）：
  APPLICABILITY_BINDING_DEV_REPLAY_PASS : Section 1 + Section 2 全过
    （dev-level——不追溯 fresh PASS）
  GATE_MECHANISM_FAILED : Section 1 任一失败
  ADOPTION_STILL_BLOCKED : R1 失败（修复后仍不检索）
  SELECTION_MISSED : R2 失败
  EXECUTION_FAILED : R3 失败
  REMOVAL_UNDIFFERENTIATED : R4 失败
  PROTOCOL_FAILURE : 装载/重建失败

用法：
  python evaluation/functional/run_v1_kdd2018_applicability_binding_replay.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _patch_options,
    _request,
)

from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    CANONICALIZATION_VERSION,
    canonical_sha256,
)
from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    EditManifest,
    EditOperation,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    COMPILER_VERSION,
    RETRIEVAL_COMPILER_VERSION,
    compile_snapshot,
    memory_entry_to_dict,
    skill_entry_to_dict,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import (  # noqa: E402
    TTHAMethod,
    _applicability_from_card,
    _applicability_reachable,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
HORIZON = 48
ORIGIN = 984  # 已暴露（P4.2）——dev replay 不读新 virgin 窗口
DELAYED = ORIGIN + HORIZON
POOL = ("winsorize", "outlier_mad", "hampel_filter")
SKILL_ID = "winsorize_negative_outlier_mad"
SKILL_DIR_REL = "skills/learned/winsorize_negative_outlier_mad.json"
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_kdd2018_applicability_binding_replay_report.json"
NEW_APPLICABILITY = {
    "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]}
FABRICATED_APPLICABILITY = {
    "all": [{"feature": "task_kind", "op": "==", "value": "forecast"},
            {"feature": "clipping_probe_direction", "op": "==",
             "value": "negative"}]}


def _load_cohort_p41(root: Path) -> dict[str, Any]:
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2"
                         / "w1_kdd2018_frozen_cohort_p41.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = np.load(root / CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in rows]
    vals = {str(r["series_name"]): np.asarray(
        values[names.index(str(r["series_name"]))], dtype=np.float64)
        for r in rows}
    return {"roster": roster, "values": vals}


def _rebound_snapshot(approved: Any) -> tuple[Any, str, bool]:
    """按 A 修正已批准 skill 的 applicability（card observable_signature
    机器生成——P4.1 card 只有 task_kind）；sha 按 compiler 公式重算，
    并以"对原 snapshot 重算 == 原 sha"自检公式一致。"""
    skill = next(s for s in approved.skills if s.skill_id == SKILL_ID)
    rebound_skill = dataclasses.replace(
        skill, observable_applicability=NEW_APPLICABILITY)
    rebound = dataclasses.replace(
        approved,
        skills=tuple(rebound_skill if s.skill_id == SKILL_ID else s
                     for s in approved.skills))

    def _shas(snap: Any) -> tuple[str, str]:
        content = {
            "schema_version": "harness-content/1",
            "instruction": snap.instruction,
            "skills": [skill_entry_to_dict(s) for s in snap.skills],
            "memories": [memory_entry_to_dict(m) for m in snap.memories],
            "retrieval": dict(snap.retrieval),
            "candidate_policy": dict(snap.candidate_policy),
            "verification": dict(snap.verification),
        }
        hcs = canonical_sha256(content)
        rbs = canonical_sha256({
            "schema_version": "runtime-bundle/1",
            "harness_content_sha": hcs,
            "operator_bundle_sha": snap.dependency_shas["operator_bundle"],
            "dependency_shas": dict(snap.dependency_shas),
            "canonicalization_version": CANONICALIZATION_VERSION,
            "compiler_version": COMPILER_VERSION,
            "retrieval_compiler_version": RETRIEVAL_COMPILER_VERSION,
        })
        return hcs, rbs

    orig_hcs, orig_rbs = _shas(approved)
    formula_consistent = bool(
        orig_hcs == approved.harness_content_sha
        and orig_rbs == approved.runtime_bundle_sha)
    new_hcs, new_rbs = _shas(rebound)
    rebound = dataclasses.replace(
        rebound, harness_content_sha=new_hcs, runtime_bundle_sha=new_rbs)
    return rebound, f"{new_rbs[:16]}...", formula_consistent


def _dev_card(executor: ScopeExecutor, values: Mapping[str, Any],
              origin: int, *, extra_sig: Mapping[str, object] | None = None
              ) -> dict[str, object]:
    sig = {"task_kind": "forecast", **(extra_sig or {})}
    return {
        "pattern_id": "kdd2018-winsorize-neg",
        "failure_family": "workflow_component_negative",
        "observable_signature": sig,
        "context": dict(__import__("signed_radius", fromlist=["window_context"])
                        .window_context(values, origin, PERIOD)),
        "workflow": {"steps": [{"op": "winsorize", "params": {}}]},
        "typed_patch_options": _patch_options(executor, values, origin,
                                              "winsorize"),
    }


def _stub_manifest(h0: Any, applicability: Mapping[str, object]) -> EditManifest:
    return EditManifest(
        edit_id=SKILL_ID,
        base_harness_sha=h0.harness_content_sha,
        target_pattern_id="kdd2018-winsorize-neg",
        target_surface_id=f"skill_library.entries/{SKILL_ID}",
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value={
            "schema_version": "skill-entry/1",
            "skill_id": SKILL_ID,
            "skill_kind": "capability",
            "revision": 1,
            "body": "LLM-fabricated body",
            "observable_applicability": dict(applicability),
            "allowed_tools": ["outlier_mad"],
            "risk_guards": {
                "explicit_choice_required": True,
                "observable_applicability_only": True,
                "preserve_outside_candidate_region": True,
                "single_surface_only": True,
            },
        },
        observable_applicability=dict(applicability),
        patch_id="patch-winsorize-to-outlier_mad",
        predicted_agent_behavior_change=("retrieve_skill:outlier_mad",),
        predicted_data_effect=("reduce_outlier_tail",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",),
    )


class _StubSlowAgent:
    def __init__(self, manifest: EditManifest) -> None:
        self._manifest = manifest

    def propose_edit(self, card, surface_catalog, snapshot, *,
                     manifest_preflight=None, allowed_operator_contracts=(),
                     task_context=None) -> EditManifest:
        if manifest_preflight is not None:
            manifest_preflight(self._manifest)
        return self._manifest


def _stub_evaluator(steps, mode):
    return SimpleNamespace(gain=0.1,
                           verification=SimpleNamespace(passed=True))


def _integration_case(h0: Any, card: Mapping[str, object],
                      fast_features: Mapping[str, object]) -> dict[str, Any]:
    """handle_feedback_support 集成（stub slow_agent/evaluator——零 LLM、
    零数据读取）。返回 stage 与关键字段。"""
    tmp = Path(tempfile.mkdtemp(prefix="p43-store-")) / "store"
    tmp.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(tmp)
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=POOL,
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(np.zeros(10), task_kind="forecast"))
    method = TTHAMethod(sealed.TTHAFastAgent(core), h0, ())
    ep = SimpleNamespace(support_response={"gain": -0.2},
                         relation="NEGATIVE",
                         episode_id="dev_p43_binding")
    ev = method.handle_feedback_support(ep, confirmed_cause="SKILL_LIBRARY_GAP", slow_agent=_StubSlowAgent(_stub_manifest(h0, FABRICATED_APPLICABILITY)),
        controller=controller, store=store,
        surface_catalog=[{
            "surface_id": "skill_library.entries/{skill_id}",
            "operation": "ADD",
            "surface_type": "skill",
            "allowed_operations": ["ADD"]}],
        card_builder=lambda e: card,
        evaluator=_stub_evaluator,
        fast_features=fast_features)
    return {"stage": ev.get("stage"), "applicability_reason":
            ev.get("applicability_reason"), "action": ev.get("action"),
            "error": ev.get("error")}


def _run_arm(snapshot: Any, series0: np.ndarray, values: Mapping[str, Any],
             executor: ScopeExecutor) -> dict[str, Any]:
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=POOL,
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(series0[:ORIGIN], task_kind="forecast"))
    method = TTHAMethod(sealed.TTHAFastAgent(core), snapshot, ())
    method.bind_round_data(series0[:ORIGIN], task_kind="forecast")
    method.prepare(_request(series0, values, ORIGIN))
    trace = method.last_trace
    steps_map = dict(trace.candidate_program_steps or {})
    chosen = trace.chosen_candidate_id or ""
    chosen_steps = steps_map.get(chosen, ())
    out: dict[str, Any] = {
        "retrieved_skill_ids": list(trace.retrieved_skill_ids or ()),
        "pool": list(trace.candidate_ids or ()),
        "chosen": chosen,
        "chosen_program": [
            {"op": op, "params": dict(p)} for op, p in chosen_steps],
    }
    if chosen_steps:
        rr = executor.evaluate(tuple(chosen_steps), ORIGIN)
        out["support_gain"] = (float(rr.gain) if rr.gain is not None else None)
        out["support_passed"] = bool(rr.verification.passed)
        rd = executor.evaluate(tuple(chosen_steps), DELAYED)
        out["delayed_gain"] = (float(rd.gain) if rd.gain is not None else None)
    else:
        out["support_gain"] = None
    return out


def main() -> int:
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    approved_dir = next((cand.parent.parent.parent
                         for cand in root.glob("*/" + SKILL_DIR_REL)), None)
    if approved_dir is None:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "approved snapshot dir not found"},
                         indent=1))
        return 0
    approved = compile_snapshot(approved_dir, verify_lock=False)
    rebound, rebound_sha, sha_consistent = _rebound_snapshot(approved)

    cohort = _load_cohort_p41(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, values, _config(),
                             evaluate_fn=_evaluate_kdd)
    features = dict(extract_public_features(series0[:ORIGIN],
                                            task_kind="forecast"))
    fe = {k: features.get(k) for k in
          ("task_kind", "clipping_probe_direction",
           "imputation_probe_direction", "denoising_probe_direction",
           "level_probe_direction")}

    # ---- Section 1：门机制测试 ----
    s1: dict[str, Any] = {"a_units": [], "b_units": [], "integration": {}}
    # A 单测
    a_task = _applicability_from_card(
        {"observable_signature": {"task_kind": "forecast"}})
    a_empty = _applicability_from_card({})
    s1["a_units"] = [
        {"case": "task_kind", "ok": a_task == NEW_APPLICABILITY,
         "value": a_task},
        {"case": "empty_sig_const_true", "ok": a_empty == {"const": True},
         "value": a_empty},
    ]
    # B 单测
    card_task = _dev_card(executor, values, ORIGIN)
    card_clip = _dev_card(executor, values, ORIGIN,
                          extra_sig={"clipping_probe_direction": "negative"})
    b1 = _applicability_reachable(card_task, NEW_APPLICABILITY, features)
    b2 = _applicability_reachable(card_task, FABRICATED_APPLICABILITY,
                                  features)
    b3 = _applicability_reachable(card_clip, FABRICATED_APPLICABILITY,
                                  features)
    s1["b_units"] = [
        {"case": "card_bound_reachable", "ok": b1[0], "reason": b1[1]},
        {"case": "fabricated_rejected_provenance", "ok": (not b2[0]),
         "reason": b2[1]},
        {"case": "fabricated_rejected_context", "ok": (not b3[0]),
         "reason": b3[1]},
    ]
    # 集成（handle_feedback_support——stub，零 LLM/零数据）
    # empty_sig_card：observable_signature 为空 → A 生成 {"const": True}
    # → B 检查 ①② 空、③ evaluate_applicability(const True)=True → pending
    # （闭合审查 PLAUSIBLE：const True 端到端过 controller 一致性校验）
    card_nosig = dict(card_task)
    card_nosig["observable_signature"] = {}
    s1["integration"] = {
        "clip_card": _integration_case(h0, card_clip, features),
        "task_card": _integration_case(h0, card_task, features),
        "empty_sig_card": _integration_case(h0, card_nosig, features),
    }
    s1["pass"] = bool(
        all(u["ok"] for u in s1["a_units"])
        and all(u["ok"] for u in s1["b_units"])
        and s1["integration"]["clip_card"]["stage"]
        == "applicability_unreachable"
        and s1["integration"]["task_card"]["stage"] == "pending"
        and s1["integration"]["empty_sig_card"]["stage"] == "pending")
    print(f"== section1 pass={s1['pass']} "
          f"clip_card={s1['integration']['clip_card']} "
          f"task_card={s1['integration']['task_card']} "
          f"empty_sig_card={s1['integration']['empty_sig_card']}")

    # ---- Section 2：dev replay（已暴露 @984）----
    adopt = _run_arm(rebound, series0, values, executor)
    remove = _run_arm(h0, series0, values, executor)
    checks: dict[str, bool] = {
        "R1_skill_retrieved": bool(
            SKILL_ID in adopt["retrieved_skill_ids"]
            and any(str(c).startswith("cand_skill_") for c in adopt["pool"])),
        "R2_chosen_is_skill": bool(
            adopt["chosen"] == f"cand_skill_{SKILL_ID}"
            and [st.get("op") for st in adopt["chosen_program"]]
            == ["outlier_mad"]),
        "R3_executed": bool(adopt.get("support_passed")),
        "R4_removal_restores": bool(
            not any(str(c).startswith("cand_skill_") for c in remove["pool"])
            and remove["chosen"] != adopt["chosen"]
            and [st.get("op") for st in remove["chosen_program"]]
            != [st.get("op") for st in adopt["chosen_program"]]),
    }
    if not s1["pass"]:
        verdict = "GATE_MECHANISM_FAILED"
    elif not checks["R1_skill_retrieved"]:
        verdict = "ADOPTION_STILL_BLOCKED"
    elif not checks["R2_chosen_is_skill"]:
        verdict = "SELECTION_MISSED"
    elif not checks["R3_executed"]:
        verdict = "EXECUTION_FAILED"
    elif not checks["R4_removal_restores"]:
        verdict = "REMOVAL_UNDIFFERENTIATED"
    else:
        verdict = "APPLICABILITY_BINDING_DEV_REPLAY_PASS"
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== verdict: {verdict}")
    print(f"== ADOPT: chosen={adopt['chosen']} program={adopt['chosen_program']} "
          f"support={adopt.get('support_gain')} delayed={adopt.get('delayed_gain')}")
    print(f"== REMOVE: chosen={remove['chosen']} program={remove['chosen_program']} "
          f"support={remove.get('support_gain')}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-kdd2018-applicability-binding-replay",
        "note": "P4.3 development replay（已暴露 @984；零新 LLM/零新数据；"
                "dev-level——只证修复机制有效，不追溯 fresh PASS）",
        "origin": ORIGIN, "delayed": DELAYED,
        "approved_snapshot_sha": approved.runtime_bundle_sha,
        "rebound_snapshot_sha": rebound_sha,
        "sha_formula_consistent": sha_consistent,
        "rebound_applicability": NEW_APPLICABILITY,
        "fabricated_applicability": FABRICATED_APPLICABILITY,
        "features_at_origin": fe,
        "section1_gate_mechanism": s1,
        "arms": {"ADOPT": adopt, "REMOVE": remove},
        "checks": checks,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
