"""Monash feasibility finalize（审查者建议 2026-08-10：报告/roster/verdict
固化为可复现脚本 + 标准中位数口径）。

输入：w1_monash_feasibility.partial.jsonl（批级扫描产物）
输出：w1_monash_feasibility_roster.jsonl + w1_monash_feasibility_report.json
规则（预注册，与扫描脚本常量一致）：
  - 竞争条件 = competition 字段（扫描时固定）；
  - roster = 按 series_name 排序取前 120 条竞争序列；
  - verdict：competition ≥ 120 → PASS，否则 STOP_LOW_PREVALENCE。

用法：
  python evaluation/functional/run_v1_monash_feasibility_finalize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTIAL_REL = Path("artifacts/functional/e2/w1_monash_feasibility.partial.jsonl")
ROSTER_REL = Path("artifacts/functional/e2/w1_monash_feasibility_roster.jsonl")
REPORT_REL = Path("artifacts/functional/e2/w1_monash_feasibility_report.json")
ROSTER_FROZEN = 120


def main() -> int:
    root = PROJECT_ROOT
    rows = [json.loads(line)
            for line in (root / PARTIAL_REL).read_text(encoding="utf-8")
            .splitlines() if line.strip()]
    comp = sorted([r for r in rows if r["competition"]],
                  key=lambda r: str(r["series_name"]))
    roster = [{
        "series_name": r["series_name"],
        "length": r["length"],
        "level_excursion_score": r["level_excursion_score"],
        "repair_actionable": r["repair_actionable"],
        "winsorize_actionable": r["winsorize_actionable"],
        "outlier_iqr_actionable": r["outlier_iqr_actionable"],
    } for r in comp[:ROSTER_FROZEN]]
    scores = sorted(float(r["level_excursion_score"]) for r in rows)
    n = len(scores)
    median = ((scores[n // 2 - 1] + scores[n // 2]) / 2.0 if n % 2 == 0
              else scores[n // 2])  # 标准中位数（偶长取两中值平均）
    report = {
        "experiment_id": "v1-monash-weather-feasibility",
        "dataset": "monash_weather_daily",
        "source": "zenodo.org/record/4654822 (weather_dataset.zip, CC-BY-4.0)",
        "note": "只读公开 Context + 静态 verifier；不读任何 gain",
        "n_scanned": len(rows),
        "n_competition": len(comp),
        "prevalence": round(len(comp) / len(rows), 4),
        "n_length_ok": sum(bool(r["length_ok"]) for r in rows),
        "n_level_signal_ok": sum(bool(r["level_signal_ok"]) for r in rows),
        "n_repair_actionable": sum(bool(r["repair_actionable"]) for r in rows),
        "n_winsorize_actionable": sum(bool(r["winsorize_actionable"])
                                      for r in rows),
        "n_outlier_iqr_actionable": sum(bool(r["outlier_iqr_actionable"])
                                        for r in rows),
        "level_score_median": median,
        "roster_frozen": len(roster),
        "roster_path": ROSTER_REL.as_posix(),
        "verdict": ("PASS" if len(comp) >= ROSTER_FROZEN
                    else "STOP_LOW_PREVALENCE"),
    }
    (root / ROSTER_REL).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in roster),
        encoding="utf-8")
    (root / REPORT_REL).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
