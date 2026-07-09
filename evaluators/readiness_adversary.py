"""Offline adversary metrics for data-readiness decisions.

This module converts frozen per-row utility records into a reviewer-facing
readiness table. It does not call forecasters or LLMs; it only scores existing
post-hoc utility evidence.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


JsonDict = Dict[str, object]
PolicyMap = Mapping[str, Mapping[str, str]]
AbstainMap = Mapping[str, Mapping[str, bool]]


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _losses(record: Mapping[str, object]) -> Dict[str, float]:
    raw = record.get("L_test")
    if not isinstance(raw, Mapping):
        raise ValueError(f"record {record.get('uid')!r} has no L_test mapping")
    losses = {str(k): float(v) for k, v in raw.items() if _is_finite_number(v)}
    if not losses:
        raise ValueError(f"record {record.get('uid')!r} has no finite L_test values")
    return losses


def _oracle_action(losses: Mapping[str, float]) -> Tuple[str, float]:
    action = min(losses, key=losses.get)
    return action, float(losses[action])


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def _binary_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = float("nan")
    if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    return {
        "readiness_precision": precision,
        "readiness_recall": recall,
        "readiness_f1": f1,
        "readiness_accuracy": _rate(tp + tn, tp + fp + tn + fn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def load_jsonl(path: Path | str) -> List[JsonDict]:
    p = Path(path)
    rows: List[JsonDict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def policies_from_record_arms(
    records: Iterable[Mapping[str, object]],
    arm_names: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, bool]]]:
    """Extract policy picks and abstain flags from each record's ``arms`` field."""
    rows = list(records)
    if arm_names is None:
        found = set()
        for row in rows:
            arms = row.get("arms")
            if isinstance(arms, Mapping):
                found.update(str(k) for k in arms)
        arm_names = sorted(found)

    policies: Dict[str, Dict[str, str]] = {name: {} for name in arm_names}
    abstain: Dict[str, Dict[str, bool]] = {name: {} for name in arm_names}
    for row in rows:
        uid = str(row.get("uid"))
        arms = row.get("arms")
        if not isinstance(arms, Mapping):
            continue
        for name in arm_names:
            rec = arms.get(name)
            if not isinstance(rec, Mapping):
                continue
            pick = rec.get("pick")
            if pick is None:
                continue
            policies[name][uid] = str(pick)
            abstain[name][uid] = bool(rec.get("abstain", False))
    return policies, abstain


def _decision_rows(
    records: Sequence[Mapping[str, object]],
    policies: PolicyMap,
    abstain_by_policy: Optional[AbstainMap],
    raw_action: str,
    margin: float,
) -> List[JsonDict]:
    rows: List[JsonDict] = []
    abstain_by_policy = abstain_by_policy or {}
    for record in records:
        uid = str(record.get("uid"))
        losses = _losses(record)
        if raw_action not in losses:
            raise ValueError(f"record {uid!r} is missing raw_action={raw_action!r}")
        raw_loss = float(losses[raw_action])
        oracle, oracle_loss = _oracle_action(losses)
        oracle_gain = raw_loss - oracle_loss
        actionable = oracle_gain > margin
        for policy, picks in policies.items():
            action = picks.get(uid)
            valid = action in losses if action is not None else False
            selected_loss = float(losses[action]) if valid and action is not None else float("nan")
            gain_vs_raw = raw_loss - selected_loss if valid else float("nan")
            rows.append(
                {
                    "uid": uid,
                    "origin": str(record.get("origin", "")),
                    "cell": str(record.get("cell", "")),
                    "policy": policy,
                    "action": action or "",
                    "valid": bool(valid),
                    "abstained": bool(abstain_by_policy.get(policy, {}).get(uid, False)),
                    "raw_action": raw_action,
                    "raw_loss": raw_loss,
                    "oracle_action": oracle,
                    "oracle_loss": oracle_loss,
                    "oracle_gain_vs_raw": oracle_gain,
                    "oracle_actionable": bool(actionable),
                    "selected_loss": selected_loss,
                    "regret": selected_loss - oracle_loss if valid else float("nan"),
                    "gain_vs_raw": gain_vs_raw,
                    "predicted_actionable": bool(valid and gain_vs_raw > margin),
                    "harmed_vs_raw": bool(valid and selected_loss > raw_loss + margin),
                    "top1_oracle": bool(valid and action == oracle),
                }
            )
    return rows


def _summarize_policy(rows: Sequence[Mapping[str, object]], n_records: int) -> Dict[str, object]:
    valid_rows = [r for r in rows if r["valid"]]
    tp = sum(1 for r in valid_rows if r["predicted_actionable"] and r["oracle_actionable"])
    fp = sum(1 for r in valid_rows if r["predicted_actionable"] and not r["oracle_actionable"])
    tn = sum(1 for r in valid_rows if not r["predicted_actionable"] and not r["oracle_actionable"])
    fn = sum(1 for r in valid_rows if not r["predicted_actionable"] and r["oracle_actionable"])
    out: Dict[str, object] = {
        "n_records": n_records,
        "n_valid": len(valid_rows),
        "coverage": _rate(len(valid_rows), n_records),
        "abstain_rate": _rate(sum(1 for r in valid_rows if r["abstained"]), len(valid_rows)),
        "mean_regret": _mean([float(r["regret"]) for r in valid_rows]),
        "mean_gain_vs_raw": _mean([float(r["gain_vs_raw"]) for r in valid_rows]),
        "top1_oracle_rate": _rate(sum(1 for r in valid_rows if r["top1_oracle"]), len(valid_rows)),
        "harm_rate": _rate(sum(1 for r in valid_rows if r["harmed_vs_raw"]), len(valid_rows)),
        "mean_harm_when_harmed": _mean(
            [
                float(r["selected_loss"]) - float(r["raw_loss"])
                for r in valid_rows
                if r["harmed_vs_raw"]
            ]
        ),
    }
    out.update(_binary_metrics(tp, fp, tn, fn))
    return out


def evaluate_policies(
    records: Sequence[Mapping[str, object]],
    policies: PolicyMap,
    *,
    abstain_by_policy: Optional[AbstainMap] = None,
    raw_action: str = "v_none",
    margin: float = 0.0,
) -> Dict[str, object]:
    """Evaluate policy picks against an oracle-actionable readiness label.

    The label is positive when the best available action beats ``raw_action`` by
    more than ``margin``. A policy predicts positive when its selected action
    beats raw by more than the same margin.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    rows = list(records)
    decisions = _decision_rows(rows, policies, abstain_by_policy, raw_action, margin)
    oracle_labels = []
    for record in rows:
        losses = _losses(record)
        if raw_action not in losses:
            raise ValueError(f"record {record.get('uid')!r} is missing raw_action={raw_action!r}")
        _, oracle_loss = _oracle_action(losses)
        oracle_labels.append(float(losses[raw_action]) - oracle_loss > margin)

    by_policy: Dict[str, List[Mapping[str, object]]] = {p: [] for p in policies}
    for row in decisions:
        by_policy[str(row["policy"])].append(row)
    summaries = {p: _summarize_policy(pr, len(rows)) for p, pr in by_policy.items()}
    ranking = sorted(
        summaries,
        key=lambda p: (
            float("inf") if not math.isfinite(float(summaries[p]["mean_regret"])) else float(summaries[p]["mean_regret"]),
            float("inf") if not math.isfinite(float(summaries[p]["harm_rate"])) else float(summaries[p]["harm_rate"]),
            -float(summaries[p]["mean_gain_vs_raw"]) if math.isfinite(float(summaries[p]["mean_gain_vs_raw"])) else float("inf"),
        ),
    )
    return {
        "oracle": {
            "n_records": len(rows),
            "n_actionable": sum(1 for v in oracle_labels if v),
            "actionable_rate": _rate(sum(1 for v in oracle_labels if v), len(oracle_labels)),
            "raw_action": raw_action,
            "margin": margin,
        },
        "policies": summaries,
        "policy_ranking": ranking,
        "decision_rows": decisions,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _policy_summary_rows(report: Mapping[str, object]) -> List[JsonDict]:
    policies = report["policies"]
    assert isinstance(policies, Mapping)
    ranking = report["policy_ranking"]
    assert isinstance(ranking, Sequence)
    return [
        {"policy": name, **policies[name]}  # type: ignore[index]
        for name in ranking
    ]


def render_markdown(report: Mapping[str, object]) -> str:
    oracle = report["oracle"]
    rows = _policy_summary_rows(report)
    lines = [
        "# Readiness Adversary Table",
        "",
        (
            f"Raw action: `{oracle['raw_action']}`; margin: {oracle['margin']}; "
            f"records: {oracle['n_records']}; actionable oracle rate: "
            f"{float(oracle['actionable_rate']):.3f}"
        ),
        "",
        "| policy | valid | mean regret | gain vs raw | harm rate | top1 | precision | recall | f1 | abstain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['n_valid']}/{row['n_records']} | "
            f"{float(row['mean_regret']):.4f} | {float(row['mean_gain_vs_raw']):+.4f} | "
            f"{float(row['harm_rate']):.3f} | {float(row['top1_oracle_rate']):.3f} | "
            f"{float(row['readiness_precision']):.3f} | {float(row['readiness_recall']):.3f} | "
            f"{float(row['readiness_f1']):.3f} | {float(row['abstain_rate']):.3f} |"
        )
    lines += [
        "",
        "Interpretation: oracle-actionable is positive only when some available action beats raw. "
        "A policy is useful when it lowers regret and gain-vs-raw without raising harm.",
    ]
    return "\n".join(lines) + "\n"


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def build_adversary_report(
    records_path: Path | str,
    out_dir: Path | str,
    *,
    record_arm_names: Optional[Sequence[str]] = None,
    external_pick_paths: Optional[Mapping[str, Path | str]] = None,
    raw_action: str = "v_none",
    margin: float = 0.0,
) -> Dict[str, object]:
    records = load_jsonl(records_path)
    record_policies, abstain = policies_from_record_arms(records, record_arm_names)
    policies: Dict[str, Dict[str, str]] = {
        "raw": {str(r.get("uid")): raw_action for r in records},
        **record_policies,
    }
    for name, path in (external_pick_paths or {}).items():
        picks = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(picks, Mapping):
            raise ValueError(f"external picks {path!s} must be a JSON object")
        policies[str(name)] = {str(k): str(v) for k, v in picks.items()}
    oracle_picks = {}
    for record in records:
        uid = str(record.get("uid"))
        oracle_picks[uid] = _oracle_action(_losses(record))[0]
    policies["oracle"] = oracle_picks

    report = evaluate_policies(
        records,
        policies,
        abstain_by_policy=abstain,
        raw_action=raw_action,
        margin=margin,
    )
    report["config"] = {
        "records_path": str(records_path),
        "record_arm_names": list(record_arm_names) if record_arm_names is not None else None,
        "external_pick_paths": {k: str(v) for k, v in (external_pick_paths or {}).items()},
    }

    out = Path(out_dir)
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
    return report

