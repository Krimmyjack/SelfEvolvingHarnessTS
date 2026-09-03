"""DRAFT_EXECUTION_AUTHORITY_GAP_CHECK（P0，用户裁决 2026-08-12）。

零 LLM / 零 outcome——确认正常入口的 Draft 权限缺口。

构造一个 capability Skill：
  - observable_applicability = {all: [task_kind == forecast]}（宽 Scope——
    P4.1 批准产物同款；Scope evidence = insufficient——P4.5 abstain）；
  - risk_guards 无 requires_target_support（当前代码不会写入）；
通过正常 TTHAMethod.prepare()（sealed 确定性 backend）检查：
  G1 是否被检索（retrieved_skill_ids / harness view）；
  G2 是否进入 cand_skill_* 候选池；
  G3 是否被放在 Agent 候选之前（pool 顺序——skill 优先 slot）；
  G4 是否可能不经当前 Target Support 就成为最终 Program
     （sealed select 取首个非 identity → chosen == cand_skill_*）。

预期：G1-G4 全部成立 → DRAFT_EXECUTION_AUTHORITY_GAP 确认
      （宽 Scope Skill 凭历史证据自动获得当前执行权——缺
      requires_target_support 权限门）。

用法：
  python evaluation/functional/run_v1_draft_authority_gap_check.py
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
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
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
    / "w1_draft_authority_gap_check_report.json"


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
    snap = compile_snapshot(approved_dir, verify_lock=False)
    skill = next((s for s in snap.skills if s.skill_id == SKILL_ID), None)
    if skill is None:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "skill missing from approved snapshot"},
                         indent=1))
        return 0
    # P4.2 门修复后的形态（P4.3 rebound 同款）：applicability 由 card
    # observable_signature 机器生成——task_kind-only 宽 Scope（Scope
    # evidence = insufficient——P4.5 abstain）。P0 测试的正是这种"宽
    # Scope 可检索"的 skill 是否有执行权限门。
    import dataclasses  # noqa: PLC0415
    rebound_skill = dataclasses.replace(
        skill, observable_applicability={
            "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]})
    snap = dataclasses.replace(
        snap, skills=tuple(rebound_skill if s.skill_id == SKILL_ID else s
                           for s in snap.skills))
    skill = rebound_skill
    guards = dict(skill.risk_guards or {})
    print(f"== skill: {SKILL_ID} applicability="
          f"{dict(skill.observable_applicability)} "
          f"requires_target_support={guards.get('requires_target_support')}")

    cohort = _load_cohort(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]

    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True,
                                  operators=("winsorize", "outlier_mad",
                                             "hampel_filter"),
                                  max_propose_candidates=3,
                                  force_pool=True),
        LocalPublicToolGateway(series0[:ORIGIN], task_kind="forecast"))
    method = TTHAMethod(sealed.TTHAFastAgent(core), snap, ())
    method.bind_round_data(series0[:ORIGIN], task_kind="forecast")
    method.prepare(_request(series0, values, ORIGIN))
    trace = method.last_trace
    pool = list(trace.candidate_ids or ())
    skill_cands = [c for c in pool if c.startswith("cand_skill_")]
    agent_cands = [c for c in pool if c.startswith("cand_")
                   and not c.startswith("cand_skill_")]
    chosen = trace.chosen_candidate_id or ""

    g1 = bool(SKILL_ID in (trace.retrieved_skill_ids or ()))
    g2 = bool(skill_cands)
    g3 = bool(skill_cands and agent_cands
              and pool.index(skill_cands[0]) < pool.index(agent_cands[0]))
    g4 = bool(chosen in skill_cands)
    checks = {"G1_retrieved": g1, "G2_in_pool": g2,
              "G3_before_agent_candidates": g3,
              "G4_chosen_without_support": g4}
    gap = all(checks.values())
    verdict = "DRAFT_EXECUTION_AUTHORITY_GAP" if gap else "NO_GAP_OBSERVED"
    print(f"== pool: {pool}")
    print(f"== chosen: {chosen}")
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== verdict: {verdict}")

    def _plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): _plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_plain(v) for v in value]
        return value

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-draft-authority-gap-check",
        "note": "P0 缺口确认（零 LLM/零 outcome——prepare 只读 [0,origin)）",
        "origin": ORIGIN,
        "skill": {"skill_id": SKILL_ID,
                  "applicability": _plain(skill.observable_applicability),
                  "requires_target_support": guards.get(
                      "requires_target_support")},
        "pool": pool, "chosen": chosen,
        "retrieved_skill_ids": list(trace.retrieved_skill_ids or ()),
        "checks": checks, "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
