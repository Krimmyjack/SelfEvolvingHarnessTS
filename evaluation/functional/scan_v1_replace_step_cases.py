"""P1 案例数值扫描（development 正控；已暴露数据，不消费 virgin）。

任务书 P1：REAL_SLOW_AGENT_REPLACE_STEP_POSITIVE_CONTROL 的案例结构：
  - incumbent A→B：gain < −MATERIAL（NEGATIVE）
  - A-only：近零或不能形成正向 Skill（gain < +MATERIAL）
  - B-only：最多回到 identity（gain < +MATERIAL，不显著正）
  - 存在 C ∈ 合法 DSL（C ≠ A, B）：gain(A→C) ≥ +MATERIAL
    （Support 正）且 delayed(A→C) ≥ −MATERIAL（delayed 不翻转）

在已暴露数据上枚举两步组合，输出满足全部条件的 (origin, A, B, C) 案例。
结果必须由审查者从真实报告核实（不以文字总结为准）。

用法：
  python evaluation/functional/scan_v1_replace_step_cases.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
OPS = ("denoise_median", "hampel_filter", "impute_ar", "impute_ema",
       "impute_fft", "impute_linear", "impute_ssm", "outlier_iqr",
       "outlier_mad", "period_complete", "period_median_complete",
       "repair_level_shift", "resample_uniform", "winsorize")
# 已暴露 origin 窗口（支持 [origin, origin+48)；delayed [origin+48, origin+96)
# 必须 ≤ 1024 → origin ≤ 976；delayed 不越界 → origin ≤ 976-48=928+...
# GEFCom {904,928,952}（952 delayed 窗口 1000-1048 ✓）；traffic {648,744,840}
GEFCOM_ORIGINS = (904, 928, 952)
TRAFFIC_ORIGINS = (648, 744, 840)
PARTIAL = PROJECT_ROOT / "artifacts/functional/e2/scan_v1_replace_step_cases.partial.jsonl"


def _load_partial() -> dict[str, dict[int, set[int]]]:
    """增量落盘恢复：{tag: {origin: {batch_idx}}}——被杀后重启跳过已完成
    (origin, batch)。"""
    done: dict[str, dict[int, set[int]]] = {}
    if PARTIAL.exists():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done.setdefault(rec["tag"], {}).setdefault(
                int(rec["origin"]), set()).add(int(rec.get("batch_idx", 0)))
    return done


def _combos() -> list[tuple[str, str]]:
    """双向枚举（审查 MAJOR 4）：反序 incumbent（有害算子在前）也要扫到，
    否则可能假 INFEASIBLE_NO_TRUE_REPLACEMENT_CONTROL。"""
    pairs: set[tuple[str, str]] = set()
    for i, a in enumerate(OPS):
        for b in OPS[i + 1:]:
            pairs.add((a, b))
            pairs.add((b, a))
    return sorted(pairs)


def _steps(ops: tuple[str, ...]) -> tuple:
    return tuple((op, dict(wiring.contract_params(op, PERIOD))) for op in ops)


def scan(executor: ScopeExecutor, origins: tuple[int, ...],
         tag: str, done_cases: dict[str, dict[int, set[int]]] | None = None,
         combos: list[tuple[str, str]] | None = None,
         batch_idx: int = 0) -> dict[str, object]:
    combos = _combos() if combos is None else combos
    done_cases = done_cases or {}
    results: list[dict[str, object]] = []
    t0 = time.time()
    for origin in origins:
        restored = done_cases.get(tag, {}).get(origin, set())
        if batch_idx in restored:
            # 本批已完成：从 partial 恢复，不重扫
            print(f"== [{tag}] origin {origin} batch {batch_idx} restored",
                  flush=True)
            continue
        origin_cases: list[dict[str, object]] = []
        for (a, b) in combos:
                r = executor.evaluate(_steps((a, b)), origin)
                gab = (float(r.gain) if r.gain is not None else None)
                if gab is None or gab >= -M or not r.verification.passed:
                    continue
                # 负组合 → 反事实 + incumbent delayed（信息墙的 failure 数值）
                ra = executor.evaluate(_steps((a,)), origin)
                rb = executor.evaluate(_steps((b,)), origin)
                ga = (float(ra.gain) if ra.gain is not None else None)
                gb = (float(rb.gain) if rb.gain is not None else None)
                if ga is None or gb is None:
                    continue
                r_abd = executor.evaluate(_steps((a, b)), origin + HORIZON)
                g_abd = (float(r_abd.gain) if r_abd.gain is not None else None)
                # A-only 近零/不能形成正向 Skill；B-only 不显著正（回 identity）
                if ga >= M or gb >= M:
                    continue
                # 候选 C 扫描（support + delayed）
                cs: list[dict[str, object]] = []
                for c in OPS:
                    if c == a or c == b:
                        continue
                    rs = executor.evaluate(_steps((a, c)), origin)
                    gs = (float(rs.gain) if rs.gain is not None else None)
                    if gs is None or gs < M or not rs.verification.passed:
                        continue
                    rd = executor.evaluate(_steps((a, c)), origin + HORIZON)
                    gd = (float(rd.gain) if rd.gain is not None else None)
                    if gd is None or gd < -M:
                        continue
                    cs.append({"c": c, "support_gain": gs,
                               "delayed_gain": gd})
                if cs:
                    case = {
                        "origin": origin,
                        "a": a, "b": b,
                        "gain_AB": gab,
                        "delayed_AB": g_abd,
                        "gain_A_only": ga,
                        "gain_B_only": gb,
                        "replacement_candidates": cs,
                    }
                    origin_cases.append(case)
                    results.append(case)
        # 增量落盘（被杀可恢复；重启跳过已完成 (origin, batch)）
        PARTIAL.parent.mkdir(parents=True, exist_ok=True)
        with open(PARTIAL, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tag": tag, "origin": origin,
                                "batch_idx": batch_idx,
                                "cases": origin_cases}) + "\n")
        print(f"== [{tag}] origin {origin} done "
              f"({time.time() - t0:.0f}s elapsed, "
              f"{len(results)} cases so far)", flush=True)
    print(f"== [{tag}] TOTAL cases: {len(results)} "
          f"({time.time() - t0:.0f}s)")
    return {"tag": tag, "origins": list(origins),
            "material_threshold": M, "cases": results}


def _aggregate_cases() -> dict[str, list[dict[str, object]]]:
    """从 partial 全量汇总（报告承重来源：所有已落盘批的 cases 合并，
    按 (origin, a, b) 去重——旧 origin 级行与新批级行可能重叠）。"""
    agg: dict[str, list[dict[str, object]]] = {"gefcom": [], "traffic": []}
    seen: set[tuple[object, object, object]] = set()
    if PARTIAL.exists():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for c in rec["cases"]:
                key = (c.get("origin"), c.get("a"), c.get("b"))
                if key in seen:
                    continue
                seen.add(key)
                agg.setdefault(rec["tag"], []).append(c)
    return agg


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", choices=("gefcom", "traffic"), default=None,
                        help="只扫一个域")
    parser.add_argument("--origin", type=int, default=None,
                        help="只扫指定 origin")
    parser.add_argument("--batch_idx", type=int, default=None,
                        help="批序号（配合 --batch_total；每批 ~3-4 分钟，"
                             "前台内完成避免被杀）")
    parser.add_argument("--batch_total", type=int, default=None,
                        help="批总数")
    args = parser.parse_args()

    root = PROJECT_ROOT
    out = {}
    done_cases = _load_partial()
    all_combos = _combos()
    combos = all_combos
    if args.batch_total is not None and args.batch_idx is not None:
        n = args.batch_total
        combos = all_combos[args.batch_idx::n]

    def _domains():
        if args.tag == "gefcom" or args.tag is None:
            config = dict(v6.DATASET_CONFIGS["gefcom"])
            roster, values = v6._fixed_roster(root, config)
            executor = ScopeExecutor(roster, values, config,
                                     evaluate_fn=v6._evaluate)
            yield "gefcom", executor, GEFCOM_ORIGINS
        if args.tag == "traffic" or args.tag is None:
            sealed._set_domain("monash:traffic_hourly")
            config_t = sealed._config()
            (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(
                root, offset=120)
            executor_t = ScopeExecutor(tgt_roster, tgt_values, config_t,
                                       evaluate_fn=sealed.v6._evaluate)
            yield "traffic", executor_t, TRAFFIC_ORIGINS

    for tag, executor, origins in _domains():
        if args.origin is not None:
            origins = tuple(o for o in origins if o == args.origin)
            if not origins:
                continue
        out[tag] = scan(executor, origins, tag, done_cases, combos,
                        args.batch_idx or 0)

    # 最终报告从 partial 全量汇总（批级运行后合并）
    agg = _aggregate_cases()
    report = root / "artifacts/functional/e2/scan_v1_replace_step_cases.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "material_threshold": M,
        "gefcom": {"tag": "gefcom",
                   "origins": list(GEFCOM_ORIGINS),
                   "cases": agg.get("gefcom", [])},
        "traffic": {"tag": "traffic",
                    "origins": list(TRAFFIC_ORIGINS),
                    "cases": agg.get("traffic", [])},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {report.relative_to(root)} "
          f"(gefcom={len(agg.get('gefcom', []))} "
          f"traffic={len(agg.get('traffic', []))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
