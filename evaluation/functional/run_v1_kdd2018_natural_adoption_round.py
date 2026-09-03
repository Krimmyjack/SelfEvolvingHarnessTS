"""KDD_CUP_2018_NATURAL_ADOPTION_ROUND（P4.2，用户复核裁决 2026-08-11）。

闭合 Target-local 自然更新回路：P4.1 delayed 批准（snapshot_updated=true，
report 中 support +0.1199 / delayed +0.1113）的 snapshot 在**下一轮正常入口**
是否被检索 → 选择 → 实际执行；removal 对照（原 h0 同轮同装置）。

零新 LLM（不重跑投票/不重试挑答案）：
  - 已批准 snapshot 直接从 P4.1 materialized 树装载（compile_snapshot——
    runtime_bundle_sha 96f83039...，skill_library 含
    winsorize_negative_outlier_mad：Frozen program = outlier_mad）；
  - Fast 路径用 P4.1 同款确定性 SealedProbeBackend（force_pool=True，
    候选池/预算同 P4.1）。

R4 origin=984（P4.1 消费终点：R3 delayed@936 的视野终点为 984——984 起为
virgin 窗口）；delayed@1032（评估实际需长度 ≥1080——truths raw[1032:1080]；
npz 实测 cohort 20 支全部 10898）。

双臂（唯一差异 = 批准产物）：
  ADOPT  = 已批准 snapshot（含 outlier_mad skill）
  REMOVE = 原 h0（无该 skill）
相同 cohort（K1 20 支）/Context/候选池/预算/backend。

检查（预注册）：
  C1 skill_in_snapshot : 已批准 snapshot 含 outlier_mad skill（装载验证）
  C2 skill_retrieved   : ADOPT 正常入口 retrieval 渲染该 skill
  C3 chosen_is_skill   : ADOPT chosen 程序 == outlier_mad
  C4 executed          : ADOPT 实际执行一次（verifier 通过 + gain 记录）
  C5 removal_differs   : REMOVE chosen 程序 ≠ ADOPT chosen 程序

verdict（预注册）：
  NATURAL_TARGET_LOCAL_PROGRAM_EVOLUTION_PASS : C1-C5 全过
  ADOPTION_RETRIEVAL_GATE_BLOCKED : C1 过、C2 不过（skill 在 snapshot 但
    下一轮 normal entry 未渲染——applicability 在评估特征空间不可满足）
  ADOPTION_SELECTION_MISSED : C2 过但 C3 不过
  ADOPTION_EXECUTION_FAILED : C3 过但 C4 不过
  ADOPTION_REMOVAL_UNDIFFERENTIATED : C3 过但 C5 不过
  PROTOCOL_FAILURE : snapshot 重建/cohort 装载失败

Claim 限定：本实验只验"批准产物在下一轮正常入口是否生效"（机制闭环）；
不声称跨域价值（P4.1 verdict 已降级为
CROSS_DOMAIN_VALUE_INCONCLUSIVE_EVALUATION_SEMANTICS——见文档）。

用法：
  python evaluation/functional/run_v1_kdd2018_natural_adoption_round.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _request,
)

from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
HORIZON = 48
ORIGIN = 984  # R4：P4.1 消费终点之后的第一个 virgin 起点
DELAYED = ORIGIN + HORIZON
POOL = ("winsorize", "outlier_mad", "hampel_filter")  # 同 P4.1 KDD 池
SKILL_ID = "winsorize_negative_outlier_mad"
SKILL_DIR_REL = "skills/learned/winsorize_negative_outlier_mad.json"
FROZEN_REL = PROJECT_ROOT / "artifacts/functional/e2/w1_kdd2018_frozen_cohort_p41.jsonl"
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2/w1_kdd2018_natural_adoption_round_report.json"


def _find_approved_snapshot_dir(root: Path) -> Path | None:
    """P4.1 delayed 批准的 materialized 树（含该 skill 的 SHA 目录）。"""
    for cand in root.glob("*/" + SKILL_DIR_REL):
        return cand.parent.parent.parent
    return None


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


def _run_arm(snapshot: Any, series0: np.ndarray, values: Mapping[str, Any],
             executor: ScopeExecutor, *, label: str) -> dict[str, Any]:
    """R4 正常入口 prepare + chosen 程序实际执行一次（sealed 确定性——
    零 LLM；与 P4.1 Fast 同款 backend/池/预算）。"""
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=POOL,
                                  max_propose_candidates=3,
                                  force_pool=True),
        LocalPublicToolGateway(series0[:ORIGIN], task_kind="forecast"))
    method = TTHAMethod(sealed.TTHAFastAgent(core), snapshot, ())
    method.bind_round_data(series0[:ORIGIN], task_kind="forecast")
    result = method.prepare(_request(series0, values, ORIGIN))
    trace = method.last_trace
    steps_map = dict(trace.candidate_program_steps or {})
    chosen = trace.chosen_candidate_id or ""
    chosen_steps = steps_map.get(chosen, ())
    out: dict[str, Any] = {
        "arm": label,
        "status": result.status.value,
        "retrieved_skill_ids": list(trace.retrieved_skill_ids or ()),
        "pool": list(trace.candidate_ids or ()),
        "chosen": chosen,
        "chosen_program": [
            {"op": op, "params": dict(p)} for op, p in chosen_steps],
    }
    if chosen_steps:
        rr = executor.evaluate(tuple(chosen_steps), ORIGIN)
        out["executed"] = True
        out["support_gain"] = (float(rr.gain)
                               if rr.gain is not None else None)
        out["support_passed"] = bool(rr.verification.passed)
        rd = executor.evaluate(tuple(chosen_steps), DELAYED)
        out["delayed_gain"] = (float(rd.gain)
                               if rd.gain is not None else None)
    else:
        out["executed"] = False  # identity/abstain——无程序可执行
    return out


def main() -> int:
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    approved_dir = _find_approved_snapshot_dir(root)
    if approved_dir is None:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "approved snapshot dir not found"},
                         indent=1))
        return 0
    approved = compile_snapshot(approved_dir, verify_lock=False)
    approved_skills = {s.skill_id: s for s in approved.skills}
    skill = approved_skills.get(SKILL_ID)
    c1 = bool(skill is not None and "outlier_mad" in (skill.allowed_tools or ()))

    cohort = _load_cohort_p41(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, values, _config(),
                             evaluate_fn=_evaluate_kdd)
    features = dict(extract_public_features(series0[:ORIGIN],
                                            task_kind="forecast"))
    fe = {k: (features.get(k)) for k in
          ("task_kind", "clipping_probe_direction",
           "imputation_probe_direction", "denoising_probe_direction",
           "level_probe_direction")}

    adopt = _run_arm(approved, series0, values, executor, label="ADOPT")
    remove = _run_arm(h0, series0, values, executor, label="REMOVE")

    # ---- 检查（预注册）----
    checks: dict[str, bool] = {
        "C1_skill_in_snapshot": c1,
        "C2_skill_retrieved": SKILL_ID in adopt["retrieved_skill_ids"],
        "C3_chosen_is_skill": any(
            st.get("op") == "outlier_mad" for st in adopt["chosen_program"]),
        "C4_executed": bool(
            adopt.get("executed") and adopt.get("support_passed")),
        "C5_removal_differs": bool(
            [st.get("op") for st in adopt["chosen_program"]]
            != [st.get("op") for st in remove["chosen_program"]]),
    }

    if not checks["C1_skill_in_snapshot"]:
        verdict = "PROTOCOL_FAILURE"
        reason = "approved snapshot lacks the outlier_mad skill"
    elif not checks["C2_skill_retrieved"]:
        verdict = "ADOPTION_RETRIEVAL_GATE_BLOCKED"
        reason = (f"skill in snapshot but not retrieved at next-round normal "
                  f"entry; clipping_probe_direction={fe.get('clipping_probe_direction')!r} "
                  f"(evaluation harness never supplies fixed_probe_panel) vs "
                  f"skill applicability requires 'negative'")
    elif not checks["C3_chosen_is_skill"]:
        verdict = "ADOPTION_SELECTION_MISSED"
        reason = "skill retrieved but not chosen"
    elif not checks["C4_executed"]:
        verdict = "ADOPTION_EXECUTION_FAILED"
        reason = "chosen skill program failed execution"
    elif not checks["C5_removal_differs"]:
        verdict = "ADOPTION_REMOVAL_UNDIFFERENTIATED"
        reason = "removal arm chose identical program"
    else:
        verdict = "NATURAL_TARGET_LOCAL_PROGRAM_EVOLUTION_PASS"
        reason = "C1-C5 all passed"
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== verdict: {verdict}")
    print(f"== reason: {reason}")
    print(f"== ADOPT: chosen={adopt['chosen']} "
          f"program={adopt['chosen_program']} "
          f"support={adopt.get('support_gain')} "
          f"delayed={adopt.get('delayed_gain')}")
    print(f"== REMOVE: chosen={remove['chosen']} "
          f"program={remove['chosen_program']} "
          f"support={remove.get('support_gain')}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-kdd2018-natural-adoption-round",
        "note": "P4.2 自然更新回路闭合检查：P4.1 批准 snapshot 在 R4 正常入口"
                "的检索/选择/执行 + removal 对照（零新 LLM）",
        "origin": ORIGIN, "delayed": DELAYED,
        "approved_snapshot_sha": approved.runtime_bundle_sha,
        "skill": ({"skill_id": SKILL_ID,
                   "allowed_tools": list(skill.allowed_tools or ()),
                   "observable_applicability": _plain_applicability(skill)}
                  if skill is not None else None),
        "features_at_origin": fe,
        "arms": {"ADOPT": adopt, "REMOVE": remove},
        "checks": checks,
        "verdict": verdict,
        "reason": reason,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


def _plain_applicability(skill: Any) -> object:
    def plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value
    return plain(skill.observable_applicability)


if __name__ == "__main__":
    raise SystemExit(main())
