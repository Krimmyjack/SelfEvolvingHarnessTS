"""MONASH_SCOPE_PRECHECK3（三候选版，2026-08-10，零 outcome）。

用户裁决（三候选、预算二最终验证）：roster 只能依据公开 Context 和三候选
多决策点 verifier 冻结，禁止读取 Target gain 挑选；使用新的 Monash virgin
cohort（剔除两候选实验已消费的 120 条），mintemp/maxtemp 分层平衡。

扫描：全部 3,010 条 × @600/@792/@888 三决策点 ×
{bound repair_level_shift, winsorize, outlier_iqr} 三候选的
ScopeExecutor.verify()（静态窗口 verifier，不读 gain）。通过序列剔除已
消费 120 条（w1_monash_frozen_roster.jsonl）后，分层平衡冻结：mintemp
前 60 + maxtemp 前 60 → 6 cohort × 10+10（角色 train 6/2 per type、
support 2/2、query 2/2）。

不满足 → INFEASIBLE（无法零 outcome 冻结足够的三候选合法 virgin cohort）。

用法：
  python evaluation/functional/run_v1_monash_scope_precheck3.py --batch_idx 0 --batch_total 24
  python evaluation/functional/run_v1_monash_scope_precheck3.py --finalize
"""

from __future__ import annotations

import argparse
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

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    MetricSpec,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA  # noqa: E402

# ---- 冻结配置（与两候选版一致：日频）----
PERIOD = 7
HORIZON = 48
CONTEXT_LENGTH = 192
ANCHORS = (312, 372, 432, 492, 552, 612, 672, 732, 792, 852)
SCOPE_ORIGINS = (600, 792, 888)
OPS = ("repair_level_shift", "winsorize", "outlier_iqr")
CACHE = PROJECT_ROOT / "data/monash_weather_v1/series_cache.npz"
TYPES = PROJECT_ROOT / "data/monash_weather_v1/series_types.json"
CONSUMED_REL = Path("artifacts/functional/e2/w1_monash_frozen_roster.jsonl")
PARTIAL_REL = Path(
    "artifacts/functional/e2/w1_monash_scope_precheck3.partial.jsonl")
FROZEN_REL = Path(
    "artifacts/functional/e2/w1_monash_frozen_roster_3cand.jsonl")
REPORT_REL = Path(
    "artifacts/functional/e2/w1_monash_scope_precheck3_report.json")
FROZEN_N = 120


def _config() -> dict[str, object]:
    return {
        "dataset_id": "monash_weather_daily",
        "sampling": "daily_regular",
        "period": PERIOD,
        "anchors": list(ANCHORS),
        "support_origin": 792,
        "selection_origin": 792,
    }


def _candidate_steps(values: np.ndarray, origin: int,
                     op: str) -> tuple[tuple[str, dict[str, object]], ...]:
    fe = dict(extract_public_features(np.asarray(values[:origin],
                                                 dtype=np.float64),
                                      task_kind="forecast"))
    if op == "repair_level_shift":
        bindings = OPERATOR_METADATA[op].get("public_parameter_bindings") or {}
        params = {p: float(fe[f]) for p, f in bindings.items() if f in fe}
        if len(params) != len(bindings):
            return ()
    else:
        params = dict(wiring.contract_params(op, PERIOD))
    return ((op, params),)


def _scan_one(name: str, values: np.ndarray) -> dict[str, object]:
    s = np.asarray(values, dtype=np.float64)
    executor = ScopeExecutor([{"series_uid": name, "role": "train"}],
                             {name: s}, _config(), evaluate_fn=v6._evaluate)
    failures: list[dict[str, object]] = []
    for origin in SCOPE_ORIGINS:
        for op in OPS:
            steps = _candidate_steps(s, origin, op)
            if not steps:
                failures.append({"origin": origin, "op": op,
                                 "rejection": "bindings_incomplete"})
                continue
            v = executor.verify(steps, origin)
            if not v.passed:
                failures.append({"origin": origin, "op": op,
                                 "rejected": v.rejected_windows[:2]})
    return {"series_name": name, "length": int(s.size),
            "ok": not failures, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--batch_total", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    cache = np.load(CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    types = json.loads(TYPES.read_text(encoding="utf-8"))
    # consumed = 两候选实验 120 条 + 三候选失败尝试的 C0/C1 40 条（Source
    # gain 已打开，不能继续称 fresh——用户裁决 2026-08-10）
    consumed = {json.loads(line)["series_name"] for line in
                (PROJECT_ROOT / CONSUMED_REL).read_text(encoding="utf-8")
                .splitlines() if line.strip()}
    # 旧三候选尝试已消费名单（重建：序列级冻结的前 120——pair1 消费
    # C0/C1、pair2 消费 C2/C3；gain 打开或 verifier 读取均不得再称 fresh）
    consumed_old = PROJECT_ROOT / "data/monash_weather_v1/consumed_3cand_old.json"
    if consumed_old.exists():
        for n in json.loads(consumed_old.read_text(encoding="utf-8")):
            consumed.add(str(n))
    # 当前批次（P0 前的三候选 frozen 120 条——Target outcome 已打开，
    # 全部排除；P0.2 用户裁决：按原冻结顺序选下一批）
    cand3_frozen = PROJECT_ROOT / FROZEN_REL
    if cand3_frozen.exists():
        for line in cand3_frozen.read_text(encoding="utf-8").splitlines():
            if line.strip():
                consumed.add(str(json.loads(line)["series_name"]))
    eligible = [n for n in names if n not in consumed]

    if args.finalize:
        partial = PROJECT_ROOT / PARTIAL_REL
        done: dict[str, dict[str, object]] = {}
        if partial.exists():
            for line in partial.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[str(r["series_name"])] = r
        passed0 = [n for n in eligible if done.get(n, {}).get("ok") is True]
        # 序列级 scale-floor 预筛（快；与 _evaluate 的 _center_scale 同判定
        # ——训练窗口 context + 评估窗口；排除 scale_floor_fallback 序列）
        from SelfEvolvingHarnessTS.evaluation.functional.run_e2_cross_series_curation import (  # noqa: E402
            _center_scale,
        )

        def _scale_floor_ok(s: np.ndarray, origins: tuple[int, ...]) -> bool:
            for origin in origins:
                for anchor in ANCHORS:
                    if anchor + HORIZON > origin:
                        continue
                    w = v6._linear_integrity(
                        s[anchor - CONTEXT_LENGTH: anchor + HORIZON])
                    _c, _sc, method = _center_scale(np, w[:CONTEXT_LENGTH])
                    if method == "scale_floor_fallback":
                        return False
                w_e = v6._linear_integrity(s[origin - CONTEXT_LENGTH: origin])
                _c, _sc, method = _center_scale(np, w_e)
                if method == "scale_floor_fallback":
                    return False
            return True

        passed = [n for n in passed0
                  if _scale_floor_ok(np.asarray(values[names.index(n)],
                                                dtype=np.float64),
                                     SCOPE_ORIGINS)]
        n_scale_floor_dropped = len(passed0) - len(passed)

        # ---- cohort 级验证（用户裁决：三个 origin × 三个 Program 全部
        # verifier 通过——运行时标准是 cohort 全窗口，非序列级）----
        m_pool = sorted((n for n in passed if types.get(n) == "mintemp"),
                        key=str)
        x_pool = sorted((n for n in passed if types.get(n) == "maxtemp"),
                        key=str)

        def _cohort_verify(m10: list[str], x10: list[str],
                           role_seq: list[str]) -> tuple[bool, set[str]]:
            seqs = [(n, "mintemp") for n in m10] + [(n, "maxtemp") for n in x10]
            roster = [{"series_uid": n, "role": role_seq[j], "type": t}
                      for j, (n, t) in enumerate(seqs)]
            vals = {n: np.asarray(values[names.index(n)], dtype=np.float64)
                    for n, _ in seqs}
            executor = ScopeExecutor(roster, vals, _config(),
                                     evaluate_fn=v6._evaluate)
            series0 = vals[seqs[0][0]]
            failures: set[str] = set()
            for origin in SCOPE_ORIGINS:
                for op in OPS:
                    steps = _candidate_steps(series0, origin, op)
                    if not steps:
                        failures.add(seqs[0][0])
                        continue
                    v = executor.verify(steps, origin)
                    if not v.passed:
                        for rw in v.rejected_windows:
                            failures.add(str(rw["series_uid"]))
            return (not failures), failures

        role_seq = (["train"] * 6 + ["support"] * 2 + ["query"] * 2) * 2
        cohort_rows: list[dict[str, object]] = []
        # 贪心构建：每 cohort 从两类型池按序取 10+10；cohort 级验证失败则
        # 替换失败序列（从同类型池尾部补入），直到 6 cohort 或池耗尽。
        while len(cohort_rows) < 120 and len(m_pool) >= 10 \
                and len(x_pool) >= 10:
            m_take, x_take = m_pool[:10], x_pool[:10]
            ok, failing = _cohort_verify(m_take, x_take, role_seq)
            if ok:
                for j, n in enumerate(m_take):
                    cohort_rows.append({"cohort": f"C{len(cohort_rows) // 20}",
                                        "role": role_seq[j],
                                        "series_name": n, "type": "mintemp"})
                for j, n in enumerate(x_take):
                    cohort_rows.append({"cohort": f"C{(len(cohort_rows) - 10) // 20}",
                                        "role": role_seq[j],
                                        "series_name": n, "type": "maxtemp"})
                m_pool, x_pool = m_pool[10:], x_pool[10:]
                continue
            # 替换失败序列（同类型池尾部补入；不按 gain 换序列）
            for uid in sorted(failing):
                if uid in m_take:
                    if len(m_pool) <= 10:
                        break
                    repl = m_pool.pop(10)
                    m_take[m_take.index(uid)] = repl
                elif uid in x_take:
                    if len(x_pool) <= 10:
                        break
                    repl = x_pool.pop(10)
                    x_take[x_take.index(uid)] = repl
        cohort_rows = cohort_rows[:120]
        by_type = {t: [r["series_name"] for r in cohort_rows
                       if r["type"] == t] for t in ("mintemp", "maxtemp")}
        # 最终一次性 baseline 检查（6 cohort × 3 origin——identity 可计算；
        # scale-floor 预筛后失败概率极低；失败 → INFEASIBLE）
        baseline_failures: list[str] = []
        if len(cohort_rows) == 120:
            for i in range(6):
                ci = [{"series_uid": r["series_name"], "role": r["role"],
                       "type": r["type"]}
                      for r in cohort_rows if r["cohort"] == f"C{i}"]
                vals = {r["series_uid"]: np.asarray(
                    values[names.index(r["series_uid"])], dtype=np.float64)
                    for r in ci}
                mapped = [dict(row, role="eval")
                          if str(row["role"]) != "train" else dict(row)
                          for row in ci]
                for origin in SCOPE_ORIGINS:
                    try:
                        v6._evaluate(mapped, vals, None, _config(),
                                     origin=origin)
                    except Exception as exc:
                        baseline_failures.append(
                            f"C{i}@{origin}: {type(exc).__name__}: {exc}")
        if len(cohort_rows) < 120 or len(by_type["mintemp"]) < 60 \
                or len(by_type["maxtemp"]) < 60 or baseline_failures:
            print(json.dumps({
                "experiment_id": "v1-monash-scope-precheck3",
                "note": "零 outcome；三候选多决策点 verifier + cohort 级验证",
                "n_scanned": len(done), "n_passed": len(passed),
                "n_cohort_rows": len(cohort_rows),
                "n_mintemp": len(by_type["mintemp"]),
                "n_maxtemp": len(by_type["maxtemp"]),
                "baseline_failures": baseline_failures,
                "verdict": "INFEASIBLE",
            }, ensure_ascii=False, indent=2))
            return 0
        # 机械断言（冻结时刻）
        assert all(r["series_name"] in done and done[r["series_name"]]["ok"]
                   for r in cohort_rows)
        for i in range(6):
            for j in range(6):
                if i != j:
                    si = {r["series_name"] for r in cohort_rows
                          if r["cohort"] == f"C{i}"}
                    sj = {r["series_name"] for r in cohort_rows
                          if r["cohort"] == f"C{j}"}
                    assert si.isdisjoint(sj)
        for i in range(6):
            ci = [r for r in cohort_rows if r["cohort"] == f"C{i}"]
            assert (sum(r["type"] == "mintemp" for r in ci) == 10
                    and sum(r["type"] == "maxtemp" for r in ci) == 10)
            for t in ("mintemp", "maxtemp"):
                cnt = [r["role"] for r in ci if r["type"] == t]
                assert (cnt.count("train") == 6 and cnt.count("support") == 2
                        and cnt.count("query") == 2)
        assert all(r["series_name"] not in consumed for r in cohort_rows)
        (PROJECT_ROOT / FROZEN_REL).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n"
                    for r in cohort_rows), encoding="utf-8")
        report = {
            "experiment_id": "v1-monash-scope-precheck3",
            "note": "零 outcome：三候选多决策点 ScopeExecutor.verify；"
                    "剔除已消费 120 条；分层平衡版 B",
            "config": {"sampling": "daily_regular", "period": PERIOD,
                       "anchors": list(ANCHORS),
                       "scope_origins": list(SCOPE_ORIGINS),
                       "ops": list(OPS)},
            "n_scanned": len(done),
            "n_eligible_after_consumed": len(eligible),
            "n_passed": len(passed),
            "n_scale_floor_dropped": n_scale_floor_dropped,
            "baseline_failures": baseline_failures,
            "frozen": len(cohort_rows),
            "frozen_mintemp": sum(r["type"] == "mintemp"
                                  for r in cohort_rows),
            "frozen_maxtemp": sum(r["type"] == "maxtemp"
                                  for r in cohort_rows),
            "pairs": [{"pair1": ("C0", "C1")}, {"pair2": ("C2", "C3")},
                      {"pair3": ("C4", "C5")}],
            "verdict": "FROZEN_OK",
        }
        (PROJECT_ROOT / REPORT_REL).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0

    partial = PROJECT_ROOT / PARTIAL_REL
    done_names: set[str] = set()
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_names.add(str(json.loads(line)["series_name"]))
    print(f"checkpoint: {len(done_names)}/{len(eligible)} done", flush=True)
    start, stop = (args.batch_idx * len(eligible) // args.batch_total,
                   (args.batch_idx + 1) * len(eligible) // args.batch_total)
    if args.batch_idx == args.batch_total - 1:
        stop = len(eligible)
    print(f"batch [{start},{stop}) of {len(eligible)}", flush=True)
    with partial.open("a", encoding="utf-8") as fh:
        for i in range(start, stop):
            name = eligible[i]
            if name in done_names:
                continue
            row = _scan_one(name, values[names.index(name)])
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if (i - start + 1) % 100 == 0:
                print(f"  {i + 1}/{stop} scanned", flush=True)
    print(f"== batch {args.batch_idx} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
