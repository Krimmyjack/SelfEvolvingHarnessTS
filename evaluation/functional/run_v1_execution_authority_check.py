"""EXECUTION_AUTHORITY_CHECK（P1 验收，用户裁决 2026-08-12）。

零 LLM / 零 outcome。三个 case 验证 Skill 执行权限（P1）：
  Case 1 DRAFT：capability Skill（task_kind-only 宽 Scope +
    risk_guards.requires_target_support=true）——**不自动优先**：
    pool 中 cand_skill_* 排在 Agent 候选之后；chosen 为 Agent 候选
    （不经当前 Support 不得成为最终 Program）。
  Case 2 ACTIVE：同款 Skill 无 guard——**保留原优先语义**：cand_skill_*
    在 Agent 之前；chosen 为 skill（ACTIVE 可按现有 signed 证据优先）。
  Case 3 H0：无 capability Skill——原路径不变（pool/chosen 与 P4.1
    装置一致）。

用法：
  python evaluation/functional/run_v1_execution_authority_check.py
"""

from __future__ import annotations

import dataclasses
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
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _request,
)

from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
)

PERIOD = 24
ORIGIN = 984  # 已暴露起点（prepare 只读 [0, origin)——零 outcome）
SKILL_ID = "winsorize_negative_outlier_mad"
SKILL_DIR_REL = "skills/learned/winsorize_negative_outlier_mad.json"
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_execution_authority_check_report.json"
WIDE_APPLICABILITY = {
    "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]}


def _load_cohort(root: Path) -> dict[str, Any]:
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


def _snapshot_with_skill(root: Path, *, draft: bool) -> Any:
    approved_dir = next((cand.parent.parent.parent
                         for cand in root.glob("*/" + SKILL_DIR_REL)), None)
    assert approved_dir is not None, "approved snapshot dir not found"
    snap = compile_snapshot(approved_dir, verify_lock=False)
    skill = next(s for s in snap.skills if s.skill_id == SKILL_ID)
    guards = dict(skill.risk_guards or {})
    if draft:
        guards["requires_target_support"] = True
    rebound = dataclasses.replace(
        skill, observable_applicability=WIDE_APPLICABILITY,
        risk_guards=guards)
    return dataclasses.replace(
        snap, skills=tuple(rebound if s.skill_id == SKILL_ID else s
                           for s in snap.skills))


def _run(root: Path, snapshot: Any, series0: np.ndarray,
         values: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True,
                                  operators=("winsorize", "outlier_mad",
                                             "hampel_filter"),
                                  max_propose_candidates=3,
                                  force_pool=True),
        LocalPublicToolGateway(series0[:ORIGIN], task_kind="forecast"))
    method = TTHAMethod(sealed.TTHAFastAgent(core), snapshot, ())
    method.bind_round_data(series0[:ORIGIN], task_kind="forecast")
    method.prepare(_request(series0, values, ORIGIN))
    trace = method.last_trace
    pool = list(trace.candidate_ids or ())
    skill_cands = [c for c in pool if c.startswith("cand_skill_")]
    agent_cands = [c for c in pool if c.startswith("cand_")
                   and not c.startswith("cand_skill_")]
    chosen = trace.chosen_candidate_id or ""
    return {"label": label, "pool": pool, "chosen": chosen,
            "skill_first": bool(
                skill_cands and agent_cands
                and pool.index(skill_cands[0]) < pool.index(agent_cands[0])),
            "chosen_is_skill": bool(chosen in skill_cands)}


def main() -> int:
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    cohort = _load_cohort(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]

    draft_snap = _snapshot_with_skill(root, draft=True)
    active_snap = _snapshot_with_skill(root, draft=False)
    case1 = _run(root, draft_snap, series0, values, label="DRAFT")
    case2 = _run(root, active_snap, series0, values, label="ACTIVE")
    case3 = _run(root, h0, series0, values, label="H0")
    print(f"== case1 DRAFT : pool={case1['pool']} chosen={case1['chosen']}")
    print(f"== case2 ACTIVE: pool={case2['pool']} chosen={case2['chosen']}")
    print(f"== case3 H0    : pool={case3['pool']} chosen={case3['chosen']}")

    checks = {
        "C1_draft_not_auto_priority": bool(
            not case1["skill_first"] and not case1["chosen_is_skill"]),
        "C2_active_keeps_priority": bool(
            case2["skill_first"] and case2["chosen_is_skill"]),
        "C3_h0_unchanged": bool(
            case3["pool"] == ["identity", "cand_winsorize",
                              "cand_outlier_mad", "cand_hampel_filter"]
            and case3["chosen"] == "cand_winsorize"),
    }
    verdict = ("EXECUTION_AUTHORITY_P1_PASS" if all(checks.values())
               else "EXECUTION_AUTHORITY_P1_FAILED")
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== verdict: {verdict}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-execution-authority-check",
        "note": "P1 验收（零 LLM/零 outcome——prepare 只读 [0,origin)）",
        "origin": ORIGIN,
        "skill_applicability": WIDE_APPLICABILITY,
        "cases": [case1, case2, case3],
        "checks": checks,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
