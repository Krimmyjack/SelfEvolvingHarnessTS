"""P3.2 solar cohort 冻结（修正版，2026-08-11）。

从 precheck3 零 outcome 通过列表（solar 类型 − 已消费）贪心构建 20 支
cohort：cohort 级 ScopeExecutor.verify（3 origin × repair/winsorize/
outlier_iqr）+ identity baseline 可计算（3 origin——Ridge 几何失败等
仪器异常排除）。按 series_name 排序取；失败替换（同类型池尾部补入，
不按 gain）。零 outcome。

用法：
  python evaluation/functional/run_v1_solar_p32_freeze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
from run_v1_natural_method_owned_slow_pilot import (  # noqa: E402
    PERIOD,
    _config,
)

from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA  # noqa: E402

SCOPE_ORIGINS = (600, 792, 888)
OPS = ("repair_level_shift", "winsorize", "outlier_iqr")
CACHE = PROJECT_ROOT / "data/monash_weather_v1/series_cache.npz"
TYPES = PROJECT_ROOT / "data/monash_weather_v1/series_types.json"
PARTIAL = PROJECT_ROOT / "artifacts/functional/e2/w1_monash_scope_precheck3.partial.jsonl"
FROZEN_REL = PROJECT_ROOT / "artifacts/functional/e2/w1_monash_frozen_roster_solar_p32.jsonl"
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2/w1_solar_p32_freeze_report.json"


def _candidate_steps(values: np.ndarray, origin: int,
                     op: str) -> tuple[tuple[str, dict], ...]:
    s0 = np.asarray(values[:origin], dtype=np.float64)
    fe = dict(extract_public_features(s0, task_kind="forecast"))
    if op == "repair_level_shift":
        bindings = OPERATOR_METADATA[op].get("public_parameter_bindings") or {}
        params = {p: float(fe[f]) for p, f in bindings.items() if f in fe}
        if len(params) != len(bindings):
            return ()
    else:
        params = dict(wiring.contract_params(op, PERIOD))
    return ((op, params),)


def _cohort_ok(seqs: list[str], values: list[np.ndarray],
               role_seq: list[str]) -> tuple[bool, set[str]]:
    roster = [{"series_uid": n, "role": role_seq[j]} for j, n in enumerate(seqs)]
    vals = {n: v for n, v in zip(seqs, values)}
    executor = ScopeExecutor(roster, vals, _config(), evaluate_fn=v6._evaluate)
    series0 = vals[seqs[0]]
    failures: set[str] = set()
    # 复核修复（2026-08-11）：freeze 只做静态 verify（零 gain——不读
    # downstream future）。Ridge 几何/scale floor 等仪器问题不再预检——
    # 运行时 gain=None → 按 Blocker 1 语义判 PROTOCOL_FAILURE（如实，
    # 不误报 NO_NATURAL_FAILURE）。
    for origin in SCOPE_ORIGINS:
        for op in OPS:
            steps = _candidate_steps(series0, origin, op)
            if not steps:
                failures.add(seqs[0])
                continue
            v = executor.verify(steps, origin)
            if not v.passed:
                for rw in v.rejected_windows:
                    failures.add(str(rw["series_uid"]))
    return (not failures), failures


def main() -> int:
    cache = np.load(CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    types = json.loads(TYPES.read_text(encoding="utf-8"))
    partial = [json.loads(line) for line in
               PARTIAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    consumed = {json.loads(line)["series_name"] for line in
                (PROJECT_ROOT / "artifacts/functional/e2/w1_monash_frozen_roster.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()}
    consumed |= set(json.loads(
        (PROJECT_ROOT / "data/monash_weather_v1/consumed_3cand_old.json")
        .read_text(encoding="utf-8")))
    consumed |= {json.loads(line)["series_name"] for line in
                 (PROJECT_ROOT / "artifacts/functional/e2/w1_monash_frozen_roster_3cand.jsonl")
                 .read_text(encoding="utf-8").splitlines() if line.strip()}
    pool = sorted((r["series_name"] for r in partial
                   if r["ok"] and types.get(r["series_name"]) == "solar"
                   and r["series_name"] not in consumed), key=str)
    print(f"solar pool: {len(pool)}")
    role_seq = ["train"] * 12 + ["support"] * 4 + ["query"] * 4
    taken = pool[:20]
    ok, failing = _cohort_ok(
        taken, [np.asarray(values[names.index(n)], dtype=np.float64)
                for n in taken], role_seq)
    while not ok and len(pool) > 20:
        for uid in sorted(failing):
            if uid in taken:
                repl = pool.pop(20) if len(pool) > 20 else None
                if repl is None:
                    break
                taken[taken.index(uid)] = repl
        ok, failing = _cohort_ok(
            taken, [np.asarray(values[names.index(n)], dtype=np.float64)
                    for n in taken], role_seq)
    if not ok:
        print(json.dumps({"verdict": "INFEASIBLE", "n_pool": len(pool),
                          "failing": sorted(failing)[:10]}, indent=1))
        return 0
    rows = [{"cohort": "S0", "role": role_seq[i], "series_name": n,
             "type": "solar"} for i, n in enumerate(taken)]
    FROZEN_REL.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    report = {"verdict": "FROZEN_OK", "n_pool": len(pool),
              "frozen": len(rows),
              "series": taken}
    REPORT_REL.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
