"""Run the Pattern-Batch confirmatory scan on Stage2 records."""
from __future__ import annotations

import argparse
from pathlib import Path

from .evaluators.pattern_batch_scan import build_pattern_batch_report, render_table

ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS = ROOT / "results" / "Stage2" / "S2_replication" / "records_s2.jsonl"
DEFAULT_OUT = ROOT / "results" / "Stage2" / "PatternBatchConfirmatory"
DEFAULT_P1B = ROOT / "results" / "Stage2" / "SkillSliceV2" / "bplus_features.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Pattern-Batch confirmatory scan")
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--p1b-features", type=Path, default=DEFAULT_P1B)
    args = ap.parse_args()

    p1b = args.p1b_features if args.p1b_features.exists() else None
    report = build_pattern_batch_report(args.records, args.out, k=args.k, p1b_features_path=p1b)
    print(render_table(report))
    print(f"Artifacts: {args.out}")


if __name__ == "__main__":
    main()
