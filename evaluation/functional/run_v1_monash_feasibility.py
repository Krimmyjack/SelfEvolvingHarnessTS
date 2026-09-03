"""MONASH_WEATHER_FEASIBILITY（用户 Gate P1 feasibility，2026-08-10）。

只读公开 Context + 静态 verifier，**不读任何 gain**。批级拆分
（--batch_idx/--batch_total）+ partial JSONL 增量落盘 + checkpoint
恢复（scan_v1_replace_step_cases 经验：后台任务 ~6 分钟被杀）。

每序列检查（决策点 origin=792，窗口与 uci 装置一致）：
  - 长度 ≥ ORIGIN + 2*HORIZON + HORIZON = 984（R2 delayed 窗口末尾）
  - level 信号合理：level_excursion_score ∈ (1.0, 1e6)（排除数值爆炸
    序列——实测出现 3.9e8/9.9e8 异常值）且 estimated_region_start/
    end_fraction 存在
  - bound repair_level_shift ∈ actionable（静态 verifier 实测，非 identity）
  - winsorize 或 outlier_iqr ∈ actionable
  - **竞争条件** = level 信号 AND repair AND (winsorize|outlier_iqr)

输出：aggregate prevalence（满足竞争条件占比）+ 冻结 roster（JSONL，
只读不消费——fresh A5/A3 用）。

不满足 → STOP_LOW_PREVALENCE（再查 Solar；不能查看 outcome 后挑 series）。

用法：
  python evaluation/functional/run_v1_monash_feasibility.py --batch_idx 0 --batch_total 8
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

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    MetricSpec,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    _actionable_operators,
    _allowed_operators,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402

ORIGIN = 792
HORIZON = 48
MIN_LEN = ORIGIN + 2 * HORIZON + HORIZON  # 984（R2 delayed 末尾）
LEVEL_MIN, LEVEL_MAX = 1.0, 1e6  # level 信号合理范围（排除 0 与数值爆炸）
CACHE = PROJECT_ROOT / "data/monash_weather_v1/series_cache.npz"
PARTIAL_REL = Path(
    "artifacts/functional/e2/w1_monash_feasibility.partial.jsonl")
ROSTER_REL = Path(
    "artifacts/functional/e2/w1_monash_feasibility_roster.jsonl")
REPORT_REL = Path("artifacts/functional/e2/w1_monash_feasibility_report.json")


def _request_for(series0: np.ndarray) -> PreparationRequest:
    return PreparationRequest(
        "monash-feas",
        series0,
        forecast_task_spec_v1(horizon=HORIZON,
                              downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        {},
    )


def _scan_one(name: str, values: np.ndarray, h0: Any) -> dict[str, object]:
    s0 = np.asarray(values[:ORIGIN], dtype=np.float64)
    fe = dict(extract_public_features(s0, task_kind="forecast"))
    level = float(fe.get("level_excursion_score", 0.0))
    req = _request_for(s0)
    view = resolve_harness_view(h0, fe, role="fast")
    act = _actionable_operators(req, s0, view, _allowed_operators(req))
    repair = "repair_level_shift" in act
    winsor = "winsorize" in act
    oiqr = "outlier_iqr" in act
    region_ok = ("estimated_region_start_fraction" in fe
                 and "estimated_region_end_fraction" in fe)
    level_ok = bool(LEVEL_MIN < level < LEVEL_MAX and region_ok)
    return {
        "series_name": str(name),
        "length": int(np.asarray(values).size),
        "length_ok": bool(int(np.asarray(values).size) >= MIN_LEN),
        "level_excursion_score": level,
        "level_signal_ok": level_ok,
        "repair_actionable": bool(repair),
        "winsorize_actionable": bool(winsor),
        "outlier_iqr_actionable": bool(oiqr),
        "competition": bool(level_ok and repair and (winsor or oiqr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--batch_total", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cache = np.load(CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    if args.limit:
        names, values = names[:args.limit], values[:args.limit]
    n_total = len(names)

    # checkpoint：已处理 index 集合（按 (name) 去重恢复）
    partial = PROJECT_ROOT / PARTIAL_REL
    done: set[str] = set()
    rows: list[dict[str, object]] = []
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(str(r["series_name"]))
                rows.append(r)
    print(f"checkpoint: {len(done)}/{n_total} done", flush=True)

    h0 = compile_snapshot(PROJECT_ROOT / "methods/ttha/harness/h0",
                          verify_lock=False)
    start, stop = (args.batch_idx * n_total // args.batch_total,
                   (args.batch_idx + 1) * n_total // args.batch_total)
    if args.batch_idx == args.batch_total - 1:
        stop = n_total
    print(f"batch [{start},{stop}) of {n_total}", flush=True)

    with partial.open("a", encoding="utf-8") as fh:
        for i in range(start, stop):
            name = names[i]
            if name in done:
                continue
            row = _scan_one(name, values[i], h0)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            rows.append(row)
            if (i - start + 1) % 50 == 0:
                print(f"  {i + 1}/{stop} scanned", flush=True)

    # 本批汇总
    scored = [r for r in rows if r["series_name"] in names[start:stop]]
    ok = [r for r in scored if r["competition"]]
    print(f"== batch {args.batch_idx}: {len(scored)} scored, "
          f"{len(ok)} competition", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
