"""MONASH_SCOPE_PRECHECK（用户前置修正 1+2，2026-08-10，零 outcome）。

① 多决策点 Scope 检查：ScopeExecutor.verify() 在 Source @600、R1 @792、
   R2 @888 三个决策点确认 bound repair_level_shift 与 winsorize 均存在
   合法候选（逐窗口 verifier；训练窗口 = 12 train × anchors）。若通过
   序列不足 120，从 917 条竞争序列按字典序继续补齐后重新冻结 roster。
② 日频配置冻结（看 outcome 前）：sampling=daily_regular、period=7（周，
   公开 period 规则——contract_params 的 period 键绑定）、anchors 与 uci
   装置同集合、CONTEXT_LENGTH=192/HORIZON=48、support/selection_origin
   =792。断言重冻结 roster 120 条均为 mintemp（series_types.json）。
③ 候选 steps：repair 用各 origin 的 extract_public_features 绑定参数
   （静态公开特征，零 outcome）；winsorize 用 contract_params(op, period)。

两阶段：步骤 A 逐条预筛（cohort of 1，批级拆分 + partial 落盘）；
步骤 B 重冻结 120 条 → 6 cohort（20 支/组）→ 真实 cohort 复核。

用法：
  python evaluation/functional/run_v1_monash_scope_precheck.py --batch_idx 0 --batch_total 12
  python evaluation/functional/run_v1_monash_scope_precheck.py --finalize
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
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA  # noqa: E402

# ---- 日频配置冻结（前置 2：看 outcome 前冻结；两臂一致）----
SAMPLING = "daily_regular"
PERIOD = 7  # 日频公开 period 规则（周周期；contract_params 绑定进 winsorize）
CONTEXT_LENGTH = 192
HORIZON = 48
ANCHORS = (312, 372, 432, 492, 552, 612, 672, 732, 792, 852)  # 与 uci 装置同集合
SOURCE_ORIGIN = 600
SOURCE_DELAYED = 648
TARGET_ROUNDS = [(792, 840), (888, 936)]  # (R1, delayed) / (R2, delayed)
SCOPE_ORIGINS = (SOURCE_ORIGIN, 792, 888)  # 多决策点 Scope 检查
MIN_LEN = 936 + HORIZON  # 984

CACHE = PROJECT_ROOT / "data/monash_weather_v1/series_cache.npz"
TYPES = PROJECT_ROOT / "data/monash_weather_v1/series_types.json"
ROSTER_REL = Path("artifacts/functional/e2/w1_monash_feasibility_roster.jsonl")
PRECHECK_PARTIAL_REL = Path(
    "artifacts/functional/e2/w1_monash_scope_precheck.partial.jsonl")
FROZEN_ROSTER_REL = Path(
    "artifacts/functional/e2/w1_monash_frozen_roster.jsonl")
REPORT_REL = Path("artifacts/functional/e2/w1_monash_scope_precheck_report.json")
FROZEN_N = 120

TRain = "train"


def _config() -> dict[str, object]:
    return {
        "dataset_id": "monash_weather_daily",
        "sampling": SAMPLING,
        "period": PERIOD,
        "anchors": list(ANCHORS),
        "support_origin": TARGET_ROUNDS[0][0],
        "selection_origin": TARGET_ROUNDS[0][0],
    }


def _roster_of_1(name: str) -> list[dict[str, object]]:
    return [{"series_uid": name, "role": "train"}]


def _candidate_steps(name: str, values: np.ndarray, origin: int,
                     op: str) -> tuple[tuple[str, dict[str, object]], ...]:
    fe = dict(extract_public_features(np.asarray(values[:origin],
                                                 dtype=np.float64),
                                      task_kind="forecast"))
    if op == "repair_level_shift":
        bindings = OPERATOR_METADATA[op].get("public_parameter_bindings") or {}
        params = {p: float(fe[f]) for p, f in bindings.items() if f in fe}
        if len(params) != len(bindings):
            return ()  # 绑定不完整 → 不可行动
    else:
        params = dict(wiring.contract_params(op, PERIOD))
    return ((op, params),)


def _scope_ok(executor: ScopeExecutor, values: np.ndarray, name: str,
              origins: tuple[int, ...]) -> tuple[bool, list[dict[str, object]]]:
    """三个决策点 × 两候选的逐窗口 verifier。返回 (ok, 失败明细)。"""
    failures: list[dict[str, object]] = []
    for origin in origins:
        for op in ("repair_level_shift", "winsorize"):
            steps = _candidate_steps(name, values, origin, op)
            if not steps:
                failures.append({"origin": origin, "op": op,
                                 "rejection": "bindings_incomplete"})
                continue
            v = executor.verify(steps, origin)
            if not v.passed:
                failures.append({"origin": origin, "op": op,
                                 "checked": v.checked_windows,
                                 "rejected": v.rejected_windows[:2]})
    return (not failures), failures


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

    # 竞争序列全集（feasibility 产物，字典序）
    comp = sorted(
        [json.loads(line) for line in
         (PROJECT_ROOT / ROSTER_REL).read_text(encoding="utf-8").splitlines()
         if line.strip()]
        + [], key=lambda r: str(r["series_name"]))
    # 注意：feasibility roster 只含前 120——需要从 partial 重建 917 全集
    rows = [json.loads(line) for line in
            (PROJECT_ROOT / "artifacts/functional/e2/w1_monash_feasibility.partial.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    comp_all = sorted([r["series_name"] for r in rows if r["competition"]],
                      key=str)
    if args.finalize:
        partial = PROJECT_ROOT / PRECHECK_PARTIAL_REL
        done: dict[str, dict[str, object]] = {}
        if partial.exists():
            for line in partial.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[str(r["series_name"])] = r
        passed = [n for n, r in done.items() if r["ok"]]
        # 分层平衡版 B（用户裁决 2026-08-10）：多决策点通过序列按
        # series_name 排序，取前 60 mintemp + 前 60 maxtemp；6 cohort 各
        # 10 mintemp + 10 maxtemp；cohort 内角色 train 每类 6 / support
        # 每类 2 / query 每类 2。
        by_type: dict[str, list[str]] = {}
        for t in ("mintemp", "maxtemp"):
            by_type[t] = sorted(
                (str(r["series_name"]) for r in done.values()
                 if r["ok"] and types.get(str(r["series_name"])) == t),
                key=str)[:60]
        assert len(by_type["mintemp"]) == 60 and len(by_type["maxtemp"]) == 60, (
            f"分层不足: mintemp={len(by_type['mintemp'])} "
            f"maxtemp={len(by_type['maxtemp'])}")
        cohort_rows: list[dict[str, object]] = []
        role_seq = ["train"] * 6 + ["support"] * 2 + ["query"] * 2
        for i in range(6):
            m = by_type["mintemp"][i * 10:(i + 1) * 10]
            x = by_type["maxtemp"][i * 10:(i + 1) * 10]
            for j, n in enumerate(m):
                cohort_rows.append({"cohort": f"C{i}", "role": role_seq[j],
                                    "series_name": n, "type": "mintemp"})
            for j, n in enumerate(x):
                cohort_rows.append({"cohort": f"C{i}", "role": role_seq[j],
                                    "series_name": n, "type": "maxtemp"})
        # 机械断言 1-3（运行前）
        ok_names = set(passed)
        assert all(r["series_name"] in ok_names for r in cohort_rows), (
            "断言1: frozen roster 序列必须全部通过多决策点 verify")
        for i in range(6):
            for j in range(6):
                if i != j:
                    si = {r["series_name"] for r in cohort_rows
                          if r["cohort"] == f"C{i}"}
                    sj = {r["series_name"] for r in cohort_rows
                          if r["cohort"] == f"C{j}"}
                    assert si.isdisjoint(sj), f"断言2: C{i}/C{j} 不互斥"
        for i in range(6):
            ci = [r for r in cohort_rows if r["cohort"] == f"C{i}"]
            assert (sum(r["type"] == "mintemp" for r in ci) == 10
                    and sum(r["type"] == "maxtemp" for r in ci) == 10), \
                f"断言3: C{i} 类型组成失衡"
            for t in ("mintemp", "maxtemp"):
                cnt = [r["role"] for r in ci if r["type"] == t]
                assert (cnt.count("train") == 6 and cnt.count("support") == 2
                        and cnt.count("query") == 2), f"断言3: C{i} {t} 角色失衡"
        report = {
            "experiment_id": "v1-monash-scope-precheck",
            "note": "零 outcome：ScopeExecutor.verify 静态窗口 verifier + "
                    "extract_public_features 公开特征；分层平衡版 B 冻结",
            "config_frozen": {
                "sampling": SAMPLING, "period": PERIOD,
                "anchors": list(ANCHORS),
                "context_length": CONTEXT_LENGTH, "horizon": HORIZON,
                "scope_origins": list(SCOPE_ORIGINS),
                "target_rounds": [list(t) for t in TARGET_ROUNDS],
            },
            "n_competition_total": len(comp_all),
            "n_prechecked": len(done),
            "n_multi_origin_ok": len(passed),
            "frozen": len(cohort_rows),
            "frozen_mintemp": sum(r["type"] == "mintemp" for r in cohort_rows),
            "frozen_maxtemp": sum(r["type"] == "maxtemp" for r in cohort_rows),
            "pairs": [{"pair1": ("C0", "C1")}, {"pair2": ("C2", "C3")},
                      {"pair3": ("C4", "C5")}],
            "verdict": ("FROZEN_OK" if len(cohort_rows) == FROZEN_N
                        else "INFEASIBLE_MULTI_ORIGIN_SCOPE"),
        }
        (PROJECT_ROOT / FROZEN_ROSTER_REL).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n"
                    for r in cohort_rows), encoding="utf-8")
        (PROJECT_ROOT / REPORT_REL).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0

    # ---- 步骤 A：逐条预筛（cohort of 1；批级 + partial）----
    partial = PROJECT_ROOT / PRECHECK_PARTIAL_REL
    done_names: set[str] = set()
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_names.add(str(json.loads(line)["series_name"]))
    print(f"checkpoint: {len(done_names)}/{len(comp_all)} done", flush=True)

    start, stop = (args.batch_idx * len(comp_all) // args.batch_total,
                   (args.batch_idx + 1) * len(comp_all) // args.batch_total)
    if args.batch_idx == args.batch_total - 1:
        stop = len(comp_all)
    print(f"batch [{start},{stop}) of {len(comp_all)}", flush=True)

    with partial.open("a", encoding="utf-8") as fh:
        for i in range(start, stop):
            name = comp_all[i]
            if name in done_names:
                continue
            s = np.asarray(values[[comp_all.index(name)]][0], dtype=np.float64)
            executor = ScopeExecutor(_roster_of_1(name), {name: s},
                                     _config(), evaluate_fn=v6._evaluate)
            ok, failures = _scope_ok(executor, s, name, SCOPE_ORIGINS)
            fh.write(json.dumps({
                "series_name": name,
                "length": int(s.size),
                "ok": ok,
                "failures": failures,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            if (i - start + 1) % 50 == 0:
                print(f"  {i + 1}/{stop} scanned", flush=True)
    print(f"== batch {args.batch_idx} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
