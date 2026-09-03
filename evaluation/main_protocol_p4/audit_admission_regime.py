"""Why every arm so far deploys nothing: the tail budget, not the Agent.

Step 6 of the frozen run order fired -- the Source line formed zero deployable
Skills -- and the run order says to stop rather than run A5 as if it had a
treatment.  Before stopping, this audit answers the question the stopping rule
raises but does not settle: *whose* failure is it?

It reads two already-sealed artifacts and adds no new evaluation, no LLM call
and no held-out read.  What it puts side by side is the same pair of numbers for
every policy measured on this data version:

* the 0-LLM baselines on the Target (``p4v``): a single global program and the
  frozen open-loop tree;
* the live Agent's scoped proposals on the Source cohort (``p4w``).

If the Agent's programs were uniquely reckless, the baselines would clear the
budget and only the Agent would fail.  If instead nothing clears it, then the
admission rule and the operator library, not the proposer, decide the outcome --
and A3 would deploy nothing on the Target for the same reason A5 has no
treatment, which makes the primary contrast vacuous before it is run.

That distinction changes what the next step should be, so it is worth the two
file reads it costs.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import main_experiment_contract as contract
from evaluation.main_protocol_p4 import p4b_contract as bounded
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES = PROJECT_ROOT / "artifacts/main_protocol/p4v_main_baselines.json"
SOURCE = PROJECT_ROOT / "artifacts/main_protocol/p4w_source_line.json"
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4x_admission_regime.json"

HARMED_LINE = bounded.BOUNDED_MAX_HARMED_FRACTION
HARM_LINE = bounded.BOUNDED_MAX_SINGLE_SERIES_HARM


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, arm in report["arms"].items():
        if name == "Static":
            continue  # declines everything by construction; no risk to read
        rows.append({
            "policy": name,
            "where": "Target held-in, 0 LLM",
            "aggregate_gain": arm["mean_gain"],
            "harmed_fraction": arm["harmed_fraction"],
            "max_single_series_harm": arm["max_single_series_harm"],
            "over_harmed_line": arm["harmed_fraction"] > HARMED_LINE,
            "over_harm_line": arm["max_single_series_harm"] > HARM_LINE,
            "clears_bounded_budget": arm["clears_bounded_budget"],
        })
    return rows


def _probe_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in report.get("rounds", ()):
        for probe in entry.get("probes", ()):
            if probe.get("kind") != "probe":
                continue
            risk = probe.get("risk_profile") or {}
            admission = probe.get("admission") or {}
            rows.append({
                "policy": probe.get("candidate_id"),
                "where": "Source held-in, live Fast, scoped",
                "origin": entry["origin"],
                "aggregate_gain": probe.get("gain"),
                "harmed_fraction": risk.get("harmed_fraction"),
                "max_single_series_harm": risk.get("max_single_series_harm"),
                "series_treated": (
                    None if probe.get("resolved_serving_series") is None
                    else len(probe["resolved_serving_series"])),
                "series_served": risk.get("series_read"),
                "admitted": admission.get("admitted"),
                "refusal_reason": admission.get("reason"),
            })
    return rows


def build() -> dict[str, Any]:
    baselines, source = _load(BASELINES), _load(SOURCE)
    rows = _baseline_rows(baselines)
    probes = _probe_rows(source)
    material = [p for p in probes
                if (p["aggregate_gain"] or 0.0) >= admission_policy.MATERIAL_THRESHOLD]
    admitted = [p for p in probes if p.get("admitted")]
    reasons: dict[str, int] = {}
    for probe in probes:
        reason = str(probe.get("refusal_reason"))
        reasons[reason] = reasons.get(reason, 0) + 1
    any_clears = (any(row["clears_bounded_budget"] for row in rows)
                  or bool(admitted))
    return {
        "stage": "P4X_ADMISSION_REGIME",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DERIVED_FROM_SEALED_ARTIFACTS_NO_NEW_EVALUATION",
        "data_version": contract.DATA_VERSION,
        "question": (
            "is the empty treatment an Agent failure, or does nothing in this "
            "action space clear the adjudicated risk budget on this data version"
        ),
        "budget": {
            "rule": bounded.BOUNDED_POLICY.rule,
            "max_harmed_fraction": HARMED_LINE,
            "max_single_series_harm": HARM_LINE,
            "material_threshold": admission_policy.MATERIAL_THRESHOLD,
            "thresholds_changed": 0,
        },
        "baselines_on_target": rows,
        "live_scoped_probes_on_source": probes,
        "counts": {
            "probes": len(probes),
            "probes_clearing_the_material_line": len(material),
            "probes_admitted": len(admitted),
            "refusal_reasons": reasons,
            "baselines_clearing_the_budget": sum(
                1 for row in rows if row["clears_bounded_budget"]),
        },
        "reading": (
            "every policy measured on this data version clears the aggregate "
            "line and then fails the tail: the global best-fixed program and "
            "the frozen open-loop tree both exceed both lines on the Target, "
            "and every materially positive live proposal on the Source exceeds "
            "one of them.  The proposer is not what decides this"
            if not any_clears else
            "at least one policy clears the budget, so the empty treatment is "
            "specific to what the Agent proposed rather than to the regime"
        ),
        "consequence_for_the_run_order": (
            "A3 is gated by the same rule as the Source line, so on this "
            "evidence it would deploy nothing on the Target and the primary "
            "contrast A3 - Static would be zero minus zero.  Running it would "
            "produce a number that says nothing about the Harness"
            if not any_clears else
            "the run order may continue as frozen"
        ),
        "what_this_audit_does_not_decide": [
            "whether the budget is the right budget: it is adjudicated and was "
            "not changed here",
            "whether a finer Scope could bound the tail: the frozen initialiser "
            "selects on defect presence, and the harm lives inside the selected "
            "set, but that is a claim about this initialiser, not about Scopes",
            "anything about held-out, which stays closed",
        ],
        "boundary": {
            **contract.BOUNDARY,
            "llm_calls": 0,
            "new_evaluations": 0,
            "held_out_reads": 0,
        },
        "sources": [BASELINES.name, SOURCE.name],
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print("%-34s %-30s %10s %8s %10s %s" % (
        "policy", "where", "gain", "harmed", "max harm", "clears"))
    for row in report["baselines_on_target"]:
        print("%-34s %-30s %+10.4f %8.2f %10.4f %s" % (
            row["policy"][:34], row["where"], row["aggregate_gain"],
            row["harmed_fraction"], row["max_single_series_harm"],
            row["clears_bounded_budget"]))
    for row in report["live_scoped_probes_on_source"]:
        print("%-34s %-30s %+10.4f %8.2f %10.4f %s  %s" % (
            str(row["policy"])[:34], row["where"], row["aggregate_gain"] or 0.0,
            row["harmed_fraction"] or 0.0, row["max_single_series_harm"] or 0.0,
            row["admitted"], row["refusal_reason"]))
    counts = report["counts"]
    print("\nprobes %d | materially positive %d | admitted %d | baselines clearing %d"
          % (counts["probes"], counts["probes_clearing_the_material_line"],
             counts["probes_admitted"], counts["baselines_clearing_the_budget"]))
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
