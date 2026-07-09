"""Build the E0/E1 offline readiness adversary table.

This runner uses existing S2 utility records only. It does not call external
APIs and does not retrain forecasters.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from .evaluators.readiness_adversary import build_adversary_report, render_markdown


ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS = ROOT / "results" / "Stage2" / "S2_replication" / "records_s2.jsonl"
DEFAULT_OUT = ROOT / "results" / "Stage2" / "ReadinessAdversaries"
DEFAULT_STATIC = ROOT / "results" / "Stage2" / "SkillSliceV2" / "bplus_picks.json"
DEFAULT_ARMS = ("global", "d_lookup", "dp_gbdt", "dp_abstain")


def _external_pairs(values) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--external-picks must use NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        if not name:
            raise SystemExit(f"--external-picks has empty name in {item!r}")
        out[name] = Path(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline readiness adversary evaluator")
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--raw-action", default="v_none")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument(
        "--record-arms",
        default=",".join(DEFAULT_ARMS),
        help="Comma-separated arm names to read from each record's arms field.",
    )
    ap.add_argument(
        "--external-picks",
        action="append",
        default=[],
        help="Additional static policy JSON as NAME=PATH. Can be repeated.",
    )
    ap.add_argument(
        "--no-default-static",
        action="store_true",
        help="Do not auto-load Stage2/SkillSliceV2/bplus_picks.json as P1b-static.",
    )
    args = ap.parse_args()

    arms = [a.strip() for a in args.record_arms.split(",") if a.strip()]
    external = _external_pairs(args.external_picks)
    if not args.no_default_static and DEFAULT_STATIC.exists() and "P1b-static" not in external:
        external["P1b-static"] = DEFAULT_STATIC

    report = build_adversary_report(
        args.records,
        args.out,
        record_arm_names=arms,
        external_pick_paths=external,
        raw_action=args.raw_action,
        margin=args.margin,
    )
    print(render_markdown(report))
    print(f"Artifacts: {args.out}")


if __name__ == "__main__":
    main()
