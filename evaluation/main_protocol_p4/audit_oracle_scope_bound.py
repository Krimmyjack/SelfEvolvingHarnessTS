"""A: could any Scope Slow is allowed to write have cleared the tail budget?

Run before any Slow call, so that a later Slow failure can be read.  If the
frozen revision class contains no feasible Scope, Slow cannot succeed and the
calls would buy nothing; if it does, a Slow failure is attributable to Slow.

The rules are in ``oracle_scope_contract`` and were frozen before this ran.  The
audit reads Outcomes to choose predicates -- that is what makes it a bound and
not a policy -- so everything it selects is quarantined and may not re-enter the
experiment.  See ``LEAKAGE_GUARD``.

Costs nothing that is protected: no LLM call, no Consumer fit, no new
evaluation, no held-out read.  The per-series gains already sit in ``p4w``, and
the per-series features are recomputed deterministically from the strictly
pre-origin window by the same function the live resolver used.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.main_protocol_p4 import main_experiment_contract as contract
from evaluation.main_protocol_p4 import oracle_scope_contract as rules
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_main_baselines as baselines
from evaluation.main_protocol_p4 import scope_spec as scopes
from evaluation.main_protocol_p4 import smoke_live_scope as smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "artifacts/main_protocol/p4w_source_line.json"
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4y_oracle_scope_bound.json"
FACE = "support_a"

MATERIAL = rules.FEASIBILITY_LINES["min_aggregate_gain"]
MAX_HARMED = rules.FEASIBILITY_LINES["max_harmed_fraction"]
MAX_HARM = rules.FEASIBILITY_LINES["max_single_series_harm"]
MIN_TREATED = rules.FEASIBILITY_LINES["min_treated_series"]


def _score(gains: Mapping[str, float], selected: frozenset[str],
           served: Sequence[str]) -> dict[str, Any]:
    """The gate's own arithmetic, on the subset that would have been treated.

    An excluded series takes the raw pipeline, which is bit-identical to Static,
    so it contributes exactly 0.0 -- the same basis ``_risk_profile`` uses.
    """
    vector = np.array(
        [gains[uid] if uid in selected else 0.0 for uid in served],
        dtype=np.float64)
    worst = float(-vector.min()) if vector.min() < 0.0 else 0.0
    aggregate = float(vector.mean())
    harmed = int((vector < -MATERIAL).sum())
    return {
        "treated": len(selected),
        "coverage": round(len(selected) / len(served), 4),
        "aggregate_gain": round(aggregate, 6),
        "harmed_count": harmed,
        "harmed_fraction": round(harmed / len(served), 4),
        "max_single_series_harm": round(worst, 6),
        "clears_all_lines": bool(
            len(selected) >= MIN_TREATED
            and aggregate >= MATERIAL
            and harmed / len(served) <= MAX_HARMED
            and worst <= MAX_HARM),
    }


def _blocking_lines(scored: Mapping[str, Any]) -> list[str]:
    blocked = []
    if scored["treated"] < MIN_TREATED:
        blocked.append("coverage_floor")
    if scored["aggregate_gain"] < MATERIAL:
        blocked.append("aggregate_below_material_line")
    if scored["harmed_fraction"] > MAX_HARMED:
        blocked.append("harmed_fraction_over_budget")
    if scored["max_single_series_harm"] > MAX_HARM:
        blocked.append("single_series_harm_over_budget")
    return blocked


def _candidate_subsets(resolved: frozenset[str],
                       cards: Mapping[str, Mapping[str, float]],
                       ) -> dict[frozenset, dict[str, Any]]:
    """Every subset reachable by conjoining exactly one more legal clause.

    Keyed by the resulting set, because different clauses that select the same
    series are the same revision; the first one found is kept as its witness.
    """
    shared = set.intersection(*(set(cards[uid]) for uid in resolved))
    subsets: dict[frozenset, dict[str, Any]] = {}
    for feature in sorted(shared):
        values = [float(cards[uid][feature]) for uid in resolved]
        if not all(np.isfinite(values)):
            continue
        for threshold in sorted(set(values)):
            for op in scopes.OPERATORS:
                try:
                    clause = scopes.Clause(feature, op, threshold)
                except scopes.ScopeError:
                    continue  # UID-shaped or malformed: refused by construction
                chosen = frozenset(
                    uid for uid in resolved
                    if clause.holds(float(cards[uid][feature])))
                if not chosen or chosen == resolved:
                    continue  # empty, or not a revision at all
                subsets.setdefault(chosen, clause.to_dict())
    return subsets


def _probe_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in report.get("rounds", ()):
        for probe in entry.get("probes", ()):
            if probe.get("kind") != "probe":
                continue
            admission = probe.get("admission") or {}
            if admission.get("admitted"):
                continue
            if admission.get("reason") not in rules.ELIGIBLE_REFUSAL_REASONS:
                continue
            rows.append({"origin": int(entry["origin"]), "probe": probe})
    return rows


def _audit_one(origin: int, probe: Mapping[str, Any],
               served: Sequence[str],
               cards: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    gains = {uid: float(value)
             for uid, value in zip(served, probe["per_series_gain"])}
    resolved = frozenset(probe["resolved_serving_series"] or ())

    # Alignment is asserted, not assumed: a series outside the Scope took the
    # raw pipeline, so its recorded gain must be exactly 0.0.  If the positional
    # per-series list were misaligned against the roster, this would fail here
    # rather than quietly producing a wrong bound.
    misaligned = sorted(uid for uid in served
                        if uid not in resolved and gains[uid] != 0.0)

    original = _score(gains, resolved, served)
    subsets = _candidate_subsets(resolved, cards, ) if resolved else {}

    feasible: list[dict[str, Any]] = []
    best_attempt: dict[str, Any] | None = None
    for chosen, witness in subsets.items():
        scored = _score(gains, chosen, served)
        row = {"added_clause": witness, "excluded_count": len(resolved) - len(chosen),
               **scored}
        if scored["clears_all_lines"]:
            feasible.append(row)
        elif scored["treated"] >= MIN_TREATED and (
                best_attempt is None
                or scored["aggregate_gain"] > best_attempt["aggregate_gain"]):
            best_attempt = row
    feasible.sort(key=lambda row: -row["aggregate_gain"])

    # Outside the deployable class on purpose: drop every materially harmed
    # series by name.  Keeping a harmed series can only lower the aggregate, so
    # this is the exact optimum over all subsets, not a greedy approximation.
    unconstrained_set = frozenset(
        uid for uid in resolved if gains[uid] >= -MATERIAL)
    unconstrained = _score(gains, unconstrained_set, served)

    return {
        "origin": origin,
        "candidate_id": probe.get("candidate_id"),
        "program": [step.get("op") for step in probe.get("program_steps") or ()],
        "original_scope": probe.get("serving_scope"),
        "original_refusal_reason": (probe.get("admission") or {}).get("reason"),
        "as_deployed": original,
        "alignment_check": {
            "out_of_scope_series_all_scored_zero": not misaligned,
            "misaligned_uids": misaligned,
        },
        "search": {
            "distinct_subsets_reachable_by_one_clause": len(subsets),
            "feasible_count": len(feasible),
            "feasible": bool(feasible),
        },
        "best_near_miss_if_infeasible": (
            None if feasible else
            (None if best_attempt is None else
             {**best_attempt, "blocked_by": _blocking_lines(best_attempt)})),
        "diagnostic_unconstrained_uid_level_bound": {
            **unconstrained,
            "deployable": False,
            "blocked_by": _blocking_lines(unconstrained),
            "reading": (
                "even naming series directly cannot clear the lines: the gain "
                "and the harm live in the same series, so no predicate over any "
                "feature vocabulary could have separated them"
                if not unconstrained["clears_all_lines"] else
                "naming series directly clears the lines, so a separating set "
                "exists; whether a legal predicate can express it is what the "
                "frozen class above tests"),
        },
        "quarantined_do_not_feed_back": {
            "why": rules.LEAKAGE_GUARD["the_bound_is_not_a_policy"],
            "feasible_revisions": feasible[:5],
        },
    }


def build() -> dict[str, Any]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = _probe_rows(source)
    groups = contract.cohorts()
    cell, variant = baselines._cell(groups["source"])

    served_by_origin: dict[int, list[str]] = {}
    cards_by_origin: dict[int, dict[str, dict[str, float]]] = {}
    audited = []
    for row in rows:
        origin = row["origin"]
        if origin not in served_by_origin:
            at = forecast_p4._cell_at(cell, origin)
            served_by_origin[origin] = [
                str(entry["series_uid"]) for entry in at.roster(FACE)
                if entry["role"] == "eval"]
            cards_by_origin[origin] = smoke._feature_cards(
                variant, served_by_origin[origin], origin)
        audited.append(_audit_one(
            origin, row["probe"], served_by_origin[origin],
            cards_by_origin[origin]))

    any_feasible = any(entry["search"]["feasible"] for entry in audited)
    any_unconstrained = any(
        entry["diagnostic_unconstrained_uid_level_bound"]["clears_all_lines"]
        for entry in audited)
    verdict = ("FEASIBLE_SCOPE_EXISTS" if any_feasible
               else "NO_FEASIBLE_SCOPE_IN_FROZEN_CLASS")
    return {
        "stage": "P4Y_ORACLE_SCOPE_BOUND",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "UPPER_BOUND_SELECTED_ON_OUTCOMES_NOT_A_POLICY",
        "data_version": contract.DATA_VERSION,
        "contract": rules.to_dict(),
        "probes_audited": len(audited),
        "probes": audited,
        "counts": {
            "probes_with_a_feasible_scope_in_class": sum(
                1 for entry in audited if entry["search"]["feasible"]),
            "probes_where_uid_level_selection_would_clear": sum(
                1 for entry in audited
                if entry["diagnostic_unconstrained_uid_level_bound"][
                    "clears_all_lines"]),
            "alignment_failures": sum(
                1 for entry in audited
                if not entry["alignment_check"][
                    "out_of_scope_series_all_scored_zero"]),
        },
        "verdict": verdict,
        "verdict_meaning": rules.VERDICTS[verdict],
        "fault_localisation_if_null": (
            None if any_feasible else
            ("the separating set exists but no single legal clause expresses "
             "it: the shortfall is the predicate class or the feature "
             "vocabulary, not the presence of a safe subset"
             if any_unconstrained else
             "no subset of the treated series clears the lines at all, so the "
             "fault is upstream of the Scope entirely -- the Program, or the "
             "Scope's coupling of context preparation with model routing")),
        "boundary": {**rules.BOUNDARY},
        "sources": [SOURCE.name],
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print("%5s %-32s %7s %8s %9s %10s %s" % (
        "orig", "candidate", "scoped", "subsets", "feasible", "uid-level", "as-deployed"))
    for entry in report["probes"]:
        deployed = entry["as_deployed"]
        print("%5d %-32s %7d %8d %9s %10s  g%+.4f h%.2f m%.3f" % (
            entry["origin"], str(entry["candidate_id"])[:32],
            deployed["treated"],
            entry["search"]["distinct_subsets_reachable_by_one_clause"],
            entry["search"]["feasible_count"],
            entry["diagnostic_unconstrained_uid_level_bound"]["clears_all_lines"],
            deployed["aggregate_gain"], deployed["harmed_fraction"],
            deployed["max_single_series_harm"]))
    counts = report["counts"]
    print("\nprobes %d | feasible in class %d | uid-level would clear %d | "
          "alignment failures %d" % (
              report["probes_audited"],
              counts["probes_with_a_feasible_scope_in_class"],
              counts["probes_where_uid_level_selection_would_clear"],
              counts["alignment_failures"]))
    print("VERDICT %s" % report["verdict"])
    if report["fault_localisation_if_null"]:
        print("  -> %s" % report["fault_localisation_if_null"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
