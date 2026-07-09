"""Offline safety-gate variants for readiness policy picks.

SafetyGateLite is intentionally post-hoc: it rewrites existing policy picks into
safer serving semantics, then scores those rewritten policies with the same
readiness adversary evaluator. It does not train, fit, or call an LLM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .readiness_adversary import (
    JsonDict,
    _json_safe,
    _policy_summary_rows,
    _write_csv,
    evaluate_policies,
    load_jsonl,
    policies_from_record_arms,
    render_markdown,
)


def abstain_to_raw_policy(
    picks: Mapping[str, str],
    abstains: Mapping[str, bool],
    *,
    raw_action: str = "v_none",
) -> dict[str, str]:
    """Return a copy where abstained records serve raw/no-op."""

    return {
        str(uid): raw_action if bool(abstains.get(str(uid), False)) else str(action)
        for uid, action in picks.items()
    }


def support_gated_policy(
    picks: Mapping[str, str],
    abstains: Mapping[str, bool],
    support_scores: Mapping[str, float],
    *,
    max_support_score: float,
    raw_action: str = "v_none",
) -> dict[str, str]:
    """Serve raw when abstained or support distance is too weak.

    Lower support scores are better. Missing scores are treated as weak support
    because this gate is a deployment-safety check.
    """

    out: dict[str, str] = {}
    for uid, action in picks.items():
        key = str(uid)
        score = support_scores.get(key)
        weak = score is None or float(score) > max_support_score
        out[key] = raw_action if bool(abstains.get(key, False)) or weak else str(action)
    return out


def _write_report_artifacts(out: Path, report: Mapping[str, object]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    decisions = report["decision_rows"]
    assert isinstance(decisions, Sequence)
    _write_csv(
        out / "decision_rows.csv",
        decisions,  # type: ignore[arg-type]
        [
            "uid",
            "origin",
            "cell",
            "policy",
            "action",
            "valid",
            "abstained",
            "raw_action",
            "raw_loss",
            "oracle_action",
            "oracle_loss",
            "oracle_gain_vs_raw",
            "oracle_actionable",
            "selected_loss",
            "regret",
            "gain_vs_raw",
            "predicted_actionable",
            "harmed_vs_raw",
            "top1_oracle",
        ],
    )
    _write_csv(
        out / "policy_summary.csv",
        _policy_summary_rows(report),
        [
            "policy",
            "n_records",
            "n_valid",
            "coverage",
            "abstain_rate",
            "mean_regret",
            "mean_gain_vs_raw",
            "top1_oracle_rate",
            "harm_rate",
            "mean_harm_when_harmed",
            "readiness_precision",
            "readiness_recall",
            "readiness_f1",
            "readiness_accuracy",
            "tp",
            "fp",
            "tn",
            "fn",
        ],
    )
    (out / "table.md").write_text(render_markdown(report), encoding="utf-8")
    json_report = dict(report)
    json_report.pop("decision_rows", None)
    (out / "report.json").write_text(
        json.dumps(_json_safe(json_report), ensure_ascii=False, indent=1, allow_nan=False),
        encoding="utf-8",
    )


def _support_sweep_policies(
    base: Mapping[str, str],
    base_abstain: Mapping[str, bool],
    support_scores: Mapping[str, float],
    support_quantiles: Sequence[float],
    raw_action: str,
) -> tuple[dict[str, dict[str, str]], list[JsonDict]]:
    scores = np.array([float(v) for v in support_scores.values()], dtype=float)
    scores = scores[np.isfinite(scores)]
    policies: dict[str, dict[str, str]] = {}
    rows: list[JsonDict] = []
    for q in support_quantiles:
        threshold = float(np.quantile(scores, q)) if scores.size else float("nan")
        label = f"support_q{int(round(q * 100)):02d}"
        policy_label = f"dp_abstain_{label}"
        picks = support_gated_policy(
            base,
            base_abstain,
            support_scores,
            max_support_score=threshold,
            raw_action=raw_action,
        )
        served = sum(1 for action in picks.values() if action != raw_action)
        policies[policy_label] = picks
        rows.append(
            {
                "gate_name": label,
                "policy": policy_label,
                "quantile": float(q),
                "threshold": threshold,
                "served": int(served),
                "serve_frac": served / max(1, len(base)),
            }
        )
    return policies, rows


def build_safety_gate_report(
    records_path: Path | str,
    out_dir: Path | str,
    *,
    policy_name: str = "dp_abstain",
    raw_action: str = "v_none",
    margin: float = 0.0,
    support_scores: Mapping[str, float] | None = None,
    support_quantiles: Sequence[float] = (0.5, 0.75, 0.95),
) -> dict[str, object]:
    records = load_jsonl(records_path)
    record_policies, abstains = policies_from_record_arms(records, [policy_name])
    base = record_policies[policy_name]
    base_abstain = abstains[policy_name]
    gated_name = f"{policy_name}_abstain_to_raw"
    policies = {
        "raw": {str(r.get("uid")): raw_action for r in records},
        policy_name: base,
        gated_name: abstain_to_raw_policy(base, base_abstain, raw_action=raw_action),
    }
    support_sweep: list[JsonDict] = []
    if support_scores:
        sweep_policies, support_sweep = _support_sweep_policies(
            base,
            base_abstain,
            support_scores,
            support_quantiles,
            raw_action,
        )
        if policy_name != "dp_abstain":
            sweep_policies = {
                k.replace("dp_abstain_", f"{policy_name}_", 1): v
                for k, v in sweep_policies.items()
            }
            for row in support_sweep:
                row["policy"] = str(row["policy"]).replace("dp_abstain_", f"{policy_name}_", 1)
        policies.update(sweep_policies)

    abstain_by_policy = {policy_name: base_abstain, gated_name: base_abstain}
    for row in support_sweep:
        abstain_by_policy[str(row["policy"])] = base_abstain

    report = evaluate_policies(
        records,
        policies,
        abstain_by_policy=abstain_by_policy,
        raw_action=raw_action,
        margin=margin,
    )
    report["config"] = {
        "records_path": str(records_path),
        "policy_name": policy_name,
        "raw_action": raw_action,
        "margin": margin,
        "variants": list(policies),
    }
    report["safety_gate"] = {
        "abstain_to_raw_policy": gated_name,
        "support_sweep": support_sweep,
    }
    _write_report_artifacts(Path(out_dir), report)
    return report
