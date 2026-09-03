"""PERSISTENCE_CHECK（P5，用户裁决 2026-08-12；P3 通过后执行）。

最小运行持久化——复用现有 HarnessStore：
  1. 从磁盘 active.json（store.set_active 写入——P3/P4 运行产物）读
     runtime_bundle_sha；
  2. 从 materialized 树 compile_snapshot 重载（=进程重启语义——新进程
     从磁盘恢复）；
  3. 验证 active snapshot 可重载、Draft Guard 仍存在（requires_target_
     support）；
  4. current_status() 字段（active_snapshot/episodes_count/pending/
     last_round/last_delayed/draft|active|restricted skills）；
  5. 下一轮正常入口候选行为一致（DRAFT 不自动优先——已暴露 @984）。

不新增数据库/Ledger/Hash Chain/Receipt Schema。

用法：
  python evaluation/functional/run_v1_persistence_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
from SelfEvolvingHarnessTS.methods.ttha.online_loop import current_status  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
)

ORIGIN = 984  # 已暴露
SKILLS = ("winsorize_negative_outlier_mad", "winsorize_replacement")
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_persistence_check_report.json"


def _load_cohort(root: Path) -> dict[str, Any]:
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2"
                         / "w1_kdd2018_frozen_cohort_p41.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
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
    active_file = root / "active.json"
    if not active_file.is_file():
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "active.json not found"},
                         indent=1))
        return 0
    sha = json.loads(active_file.read_text(encoding="utf-8"))[
        "runtime_bundle_sha"]
    # 在已知 store 树中找该 sha（.p3_store_b / .p4_store_a5 / .p4_store_a3）
    store_root = None
    for cand in (root / ".p3_store_b", root / ".p4_store_a5",
                 root / ".p4_store_a3"):
        if (cand / sha).is_dir():
            store_root = cand
            break
    if store_root is None:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": f"materialized tree {sha[:16]} not found"},
                         indent=1))
        return 0
    snap = compile_snapshot(store_root / sha, verify_lock=False)
    skill_ids = {s.skill_id for s in snap.skills}
    guards = {s.skill_id: dict(s.risk_guards or {})
              for s in snap.skills if s.skill_kind.value == "capability"}

    # 重启语义：新 TTHAMethod 实例 + 重载 snapshot（新进程从磁盘恢复）
    cohort = _load_cohort(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True,
                                  operators=("winsorize", "outlier_mad",
                                             "hampel_filter"),
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(series0[:ORIGIN], task_kind="forecast"))
    method = TTHAMethod(sealed.TTHAFastAgent(core), snap, ())
    method.bind_round_data(series0[:ORIGIN], task_kind="forecast")
    method.prepare(_request(series0, values, ORIGIN))
    trace = method.last_trace
    pool = list(trace.candidate_ids or ())
    skill_cands = [c for c in pool if c.startswith("cand_skill_")]
    agent_cands = [c for c in pool if c.startswith("cand_")
                   and not c.startswith("cand_skill_")]
    status = current_status(None, method)
    checks = {
        "P1_snapshot_reloaded": bool(
            snap.runtime_bundle_sha == sha and skill_ids),
        "P2_draft_guard_persisted": bool(
            any(guards.get(s, {}).get("requires_target_support") is True
                for s in SKILLS if s in skill_ids)),
        "P3_status_fields": bool(
            status["active_snapshot"]["runtime_bundle_sha"] == sha
            and isinstance(status["episodes_count"], int)
            and isinstance(status["draft_skills"], list)
            and isinstance(status["active_skills"], list)
            and isinstance(status["restricted_skills"], list)),
        "P4_next_round_behavior_consistent": bool(
            (not skill_cands) or (
                agent_cands
                and pool.index(skill_cands[0]) > pool.index(agent_cands[0])
                and not (trace.chosen_candidate_id or "") in skill_cands)),
    }
    verdict = "PERSISTENCE_CHECK_PASS" if all(checks.values()) \
        else "PERSISTENCE_CHECK_FAILED"
    print(f"== active sha: {sha[:24]}... store={store_root.name}")
    print(f"== skills: {sorted(skill_ids)}")
    print(f"== guards: {json.dumps(guards)}")
    print(f"== pool: {pool} chosen={trace.chosen_candidate_id}")
    print(f"== status: {json.dumps(status)}")
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== verdict: {verdict}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-persistence-check",
        "note": "P5 最小运行持久化（复用 HarnessStore——active.json + "
                "materialized 树；不新增数据库/Ledger）",
        "active_sha": sha, "store": store_root.name,
        "skills": sorted(skill_ids), "guards": guards,
        "pool": pool, "chosen": trace.chosen_candidate_id,
        "current_status": status,
        "checks": checks,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
