"""Run the offline SafetyGateLite readiness serving evaluator."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .e32_policy import P_FEATS
from .evaluators.readiness_adversary import load_jsonl, render_markdown
from .evaluators.safety_gate_lite import build_safety_gate_report


ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS = ROOT / "results" / "Stage2" / "S2_replication" / "records_s2.jsonl"
DEFAULT_OUT = ROOT / "results" / "Stage2" / "SafetyGateLite"
DEFAULT_FROZEN_ARMS = ROOT / "results" / "E3_2_confirmatory" / "frozen_arms.joblib"


def conditioning_key_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the P0 router conditioning key from a Stage2 record."""

    x_p = list(record.get("X_p") or [])
    if len(x_p) != len(P_FEATS):
        raise ValueError(f"record {record.get('uid')!r} has X_p length {len(x_p)}, expected {len(P_FEATS)}")
    struct = {
        "SNR": float(record["snr"]),
        "missing_rate": float(record["miss_rate"]),
    }
    struct.update({name: float(value) for name, value in zip(P_FEATS, x_p)})
    return {
        "pattern": {"struct_feats": struct},
        "task": {"type": "forecast"},
        "cell_id": str(record.get("cell") or ""),
    }


def support_scores_from_router(
    records: Sequence[Mapping[str, Any]],
    router: Any,
    *,
    action_menu: Any,
) -> dict[str, float]:
    """Extract per-record support distance from router provenance."""

    scores: dict[str, float] = {}
    for record in records:
        decision = router.predict(conditioning_key_from_record(record), action_menu)
        support = (decision.provenance or {}).get("support") or {}
        if not support.get("available") or "distance" not in support:
            continue
        distance = float(support["distance"])
        if math.isfinite(distance):
            scores[str(record.get("uid"))] = distance
    return scores


def load_router_support_scores(
    records: Sequence[Mapping[str, Any]],
    *,
    arms_path: Path,
    policy_name: str,
    allow_version_mismatch: bool = False,
) -> dict[str, float]:
    """Load the frozen router and collect support distances for Stage2 records."""

    from .policy.action_spec import action_menu_v1
    from .policy.router_policy import FrozenArmRouterPolicy

    router = FrozenArmRouterPolicy.load_frozen(
        policy_name,
        path=arms_path,
        allow_version_mismatch=allow_version_mismatch,
    )
    return support_scores_from_router(records, router, action_menu=action_menu_v1())


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline SafetyGateLite evaluator")
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--policy", default="dp_abstain")
    ap.add_argument("--raw-action", default="v_none")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--support-frozen-arms", type=Path, default=DEFAULT_FROZEN_ARMS)
    ap.add_argument("--support-quantiles", type=float, nargs="*", default=[0.5, 0.75, 0.95])
    ap.add_argument("--no-router-support", action="store_true")
    ap.add_argument("--allow-version-mismatch", action="store_true")
    args = ap.parse_args()

    records = load_jsonl(args.records)
    support_scores = None
    if not args.no_router_support and args.support_frozen_arms.exists():
        support_scores = load_router_support_scores(
            records,
            arms_path=args.support_frozen_arms,
            policy_name=args.policy,
            allow_version_mismatch=args.allow_version_mismatch,
        )

    report = build_safety_gate_report(
        args.records,
        args.out,
        policy_name=args.policy,
        raw_action=args.raw_action,
        margin=args.margin,
        support_scores=support_scores,
        support_quantiles=args.support_quantiles,
    )
    print(render_markdown(report))
    if support_scores is not None:
        print(f"Router support scores: {len(support_scores)}/{len(records)} records")
    print(f"Artifacts: {args.out}")


if __name__ == "__main__":
    main()
