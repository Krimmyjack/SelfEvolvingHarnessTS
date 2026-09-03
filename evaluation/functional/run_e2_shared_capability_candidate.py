"""B + C: compile the Shared Capability candidate and dry-replay it.  0 LLM.

B builds the evidence pool from the traffic line and the NOAA line, restricted
to the outlier-repair family and to an operator-independent per-series harm
guard, dedupes it by domain x window x plan x outcome, and compiles a card
whose applicability conditions are deployment-observable features only.

C is the honest half.  A card compiled from both domains and replayed on both
domains proves nothing about transfer, so it is run and labelled
INTERNAL_CONSISTENCY_ONLY.  The transfer question is asked the only way the
evidence allows: compile from one domain, check against the other's banked
episodes, in both directions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "shared_capability_candidate_v1.json"
OUT_MD = E2 / "shared_capability_candidate_v1.md"

OUTLIER_FAMILY = ("outlier_mad", "outlier_iqr", "hampel_filter", "winsorize")
HARM_LINE = -0.005
MATERIAL = 0.005


# --------------------------------------------------------------- B1: the pool
def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _load(name: str) -> Any:
    path = E2 / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def traffic_rows() -> list[dict[str, Any]]:
    """Every outlier-family reading the traffic cells measured.

    ``menu_scan`` holds the per-eval-series gain vector for each program the
    cell searched, so both halves of the card -- which program, and what the
    harm guard would have seen -- come from the same record.
    """
    rows: list[dict[str, Any]] = []
    payload = _load("batch_recipe_v2_all_cells_v1.json")
    if not payload:
        return rows
    for cell in payload["cells"]:
        if cell["cohort"] != "traffic":
            continue
        recipe = cell["recipe"]
        window = "support%s/delayed%s" % (
            recipe["support_origins"], recipe["delayed_origins"]
        )
        adopted = recipe.get("adopted_plan") or {}
        for program, scan in (recipe.get("menu_scan") or {}).items():
            if program not in OUTLIER_FAMILY:
                continue
            delayed = (scan or {}).get("delayed") or {}
            vector = {
                str(k): float(v)
                for k, v in (delayed.get("per_eval_series_gain") or {}).items()
            }
            if not vector:
                continue
            rows.append({
                "domain": "traffic",
                "source": "batch_recipe_v2_all_cells_v1.json",
                "episode": "traffic/%s/%s" % (cell["consumer_variant"], program),
                "window": window,
                "consumer_variant": cell["consumer_variant"],
                "program": program,
                "scope": "full_batch",
                "delayed_aggregate_gain": float(delayed.get("aggregate_gain", 0.0)),
                "per_eval_series_gain": vector,
                "min_per_series_gain": min(vector.values()),
                "harmed": sorted(u for u, g in vector.items() if g < HARM_LINE),
                "was_adopted": bool(adopted.get("program") == program),
                "in_selection": True,
                "in_selection_note": (
                    "the artifact records that the delayed window took part in "
                    "this cell's adoption decision, so both columns are "
                    "in-selection readings"
                ),
            })
    return rows


def noaa_rows() -> list[dict[str, Any]]:
    """Every outlier-family adoption this line banked, with its own vector."""
    rows: list[dict[str, Any]] = []
    for name in [
        "operational_pipeline_v%d.json" % i for i in range(1, 11)
    ] + ["slow_scope_update_v1.json", "slow_scope_update_v2.json"]:
        payload = _load(name)
        if payload is None:
            continue
        for node in _walk(payload):
            plan = node.get("plan_before_gate")
            vector_raw = node.get("per_eval_series_delayed_before_gate")
            if isinstance(plan, dict) and isinstance(vector_raw, dict) and vector_raw:
                program = str(plan.get("program") or "")
                if program not in OUTLIER_FAMILY:
                    continue
                vector = {str(k): float(v) for k, v in vector_raw.items()}
                rows.append({
                    "domain": "noaa",
                    "source": name,
                    "episode": "noaa/%s/%s" % (node.get("step"), program),
                    "window": str(node.get("window_id") or node.get("step")),
                    "consumer_variant": "pooled",
                    "program": program,
                    "scope": (
                        "full_batch" if not plan.get("excluded_series")
                        else "masked"
                    ),
                    "delayed_aggregate_gain": float(
                        node.get("delayed_before_gate") or 0.0
                    ),
                    "per_eval_series_gain": vector,
                    "min_per_series_gain": min(vector.values()),
                    "harmed": sorted(
                        u for u, g in vector.items() if g < HARM_LINE
                    ),
                    "was_adopted": True,
                    "in_selection": False,
                    "in_selection_note": (
                        "the frozen v2 ladder read the delayed window to set "
                        "its bar, so this reading is not out-of-selection "
                        "either; it is recorded as the ladder produced it"
                    ),
                })
    return rows


def dedupe(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict]]:
    """One vote per domain x window x plan x outcome."""
    seen: dict[tuple, dict[str, Any]] = {}
    collapsed: list[dict[str, Any]] = []
    for row in rows:
        key = (
            row["domain"], row["window"], row["program"], row["scope"],
            json.dumps(row["per_eval_series_gain"], sort_keys=True),
        )
        if key in seen:
            seen[key].setdefault("duplicate_of", []).append(
                "%s @ %s" % (row["episode"], row["source"])
            )
            collapsed.append({"kept": seen[key]["episode"], "dropped": row["episode"],
                              "dropped_source": row["source"]})
            continue
        seen[key] = row
    return list(seen.values()), collapsed


# --------------------------------------------- B2/B3: compile a card from rows
def compile_card(rows: list[dict[str, Any]], provenance: str) -> dict[str, Any]:
    """Deterministic induction: what do these rows agree on?"""
    positives = [r for r in rows if r["delayed_aggregate_gain"] > MATERIAL]
    harmful = [r for r in rows if r["harmed"]]
    programs = sorted({r["program"] for r in positives})
    domains = sorted({r["domain"] for r in rows})
    if not positives:
        return {
            "compiles": False,
            "why": "no evidence row shows a material delayed gain",
            "provenance": provenance,
        }
    return {
        "compiles": True,
        "provenance": provenance,
        "domains_the_evidence_comes_from": domains,
        "capability": {
            "id": "shared_outlier_repair_with_per_series_guard_v1",
            "what_it_recommends": (
                "on a batch whose Context matches the conditions below, put "
                "the outlier-repair family on the shortlist ahead of the rest "
                "of the menu, and attach the per-series harm guard to whatever "
                "the ladder adopts"
            ),
            "programs": programs,
            "guard": {
                "statistic": "min_per_series_gain",
                "window": "delayed",
                "comparator": "lt",
                "threshold": HARM_LINE,
                "actions_allowed": [
                    "VETO_AND_FALL_BACK", "RESCOPE_MASK_HARMED_SERIES",
                ],
                "operator_independent": (
                    "the guard reads the adopted plan's measured per-series "
                    "vector and names no program; it applies whatever the "
                    "ladder adopted"
                ),
            },
        },
        "applicability_conditions": {
            "stated_as": "deployment-observable features of the batch only",
            "forbidden": (
                "no dataset name, no cohort id, no provenance string may "
                "appear in a condition"
            ),
            "conditions": [
                {
                    "feature": "missing_fraction over the training pool",
                    "requirement": "may be zero; the capability does not need missing data",
                    "why": (
                        "traffic's census records missing_region_end_fraction "
                        "all_zero over its 12 training series, and #30's S1 "
                        "found 0 of 24 usable channels with any missing value "
                        "on the third-domain candidate.  A capability that "
                        "required missing data could not apply to either."
                    ),
                },
                {
                    "feature": "local_robust_z_peak over the training pool",
                    "requirement": "at least one series at or above 4.0",
                    "why": (
                        "this is the public signal the outlier family acts "
                        "on.  Traffic's census reports per-series peaks from "
                        "3.70 to 15.72; NOAA's development block has 3 of 20 "
                        "series at or above 4.0"
                    ),
                },
                {
                    "feature": "outlier_point_fraction over the training pool",
                    "requirement": "greater than zero somewhere in the pool",
                    "why": (
                        "traffic's field stats give a mean of 0.0448 with 12 "
                        "of 12 series distinct and non-degenerate"
                    ),
                },
                {
                    "feature": "per-eval-series gain dispersion of the adopted plan",
                    "requirement": (
                        "the guard is required whenever the minimum "
                        "per-series delayed gain can fall below %.3f while "
                        "the aggregate stays positive" % HARM_LINE
                    ),
                    "why": (
                        "that configuration is the failure this line has "
                        "recorded six times; it is invisible at aggregate "
                        "granularity"
                    ),
                },
            ],
        },
        "out_of_scope": {
            "families": ["imputation", "level shift"],
            "declared_because": (
                "imputation has no substrate on two of the three corpora "
                "(traffic: missing_region_end_fraction all_zero; smd: 0 of 24 "
                "usable channels with any missing value), and "
                "level_excursion_score is identically zero on all 20 NOAA "
                "development series, so a level-repair capability has no NOAA "
                "evidence to be induced from"
            ),
            "evidence_pointers": [
                "artifacts/functional/e2/s1_health_v1.json substrate_shape_warning",
                "artifacts/functional/e2/m0a_mask_geometry_census_traffic_v1.json "
                "field_stats_train.missing_region_end_fraction",
            ],
        },
        "status": "SHARED_CANDIDATE",
        "authorization": "GUIDANCE",
        "target_support_required": True,
        "grants_confirmation_free_try": False,
        "evidence_rows": len(rows),
        "positive_rows": len(positives),
        "rows_with_harm": len(harmful),
    }


# ------------------------------------------------------------------ C: replay
def replay(card: dict[str, Any], rows: list[dict[str, Any]], label: str) -> dict:
    """Does the card's direction hold on episodes it did not come from?"""
    if not card.get("compiles"):
        return {"label": label, "ran": False, "why": card.get("why")}
    programs = set(card["capability"]["programs"])
    direction, guard_cases, misses = [], [], []
    for row in rows:
        claimed = row["program"] in programs
        gained = row["delayed_aggregate_gain"] > MATERIAL
        # A row whose program the card does not claim passes vacuously.  That
        # is not evidence, and it is counted apart from the rows that actually
        # put the card at risk.
        tested = claimed
        agree = (not claimed) or gained
        direction.append({
            "episode": row["episode"],
            "program": row["program"],
            "card_claims_this_program": claimed,
            "puts_the_card_at_risk": tested,
            "historical_delayed_gain": row["delayed_aggregate_gain"],
            "direction_agrees": agree,
            "vacuous": not tested,
        })
        if not agree:
            misses.append(row["episode"])
        if row["harmed"]:
            # The non-circular question is not "does a harm line catch a harm"
            # -- it was defined by that line -- but "does the guard catch a
            # harm the aggregate hid".  A harmed row with a negative aggregate
            # would have been caught by the aggregate alone and proves nothing
            # about the guard.
            hidden = row["delayed_aggregate_gain"] > 0
            caught = row["min_per_series_gain"] < HARM_LINE
            guard_cases.append({
                "episode": row["episode"],
                "harmed_series": row["harmed"],
                "min_per_series_gain": row["min_per_series_gain"],
                "delayed_aggregate_gain": row["delayed_aggregate_gain"],
                "aggregate_hid_it": hidden,
                "guard_condition_fires": caught,
                "counts_as_evidence": hidden,
            })
            if hidden and not caught:
                misses.append(row["episode"] + " (guard missed a hidden harm)")
    at_risk = [d for d in direction if d["puts_the_card_at_risk"]]
    hidden_cases = [g for g in guard_cases if g["counts_as_evidence"]]
    return {
        "label": label,
        "ran": True,
        "card_claims_programs": sorted(programs),
        "episodes_checked": len(rows),
        "episodes_that_put_the_card_at_risk": len(at_risk),
        "vacuous_passes": len(direction) - len(at_risk),
        "direction_agreements_among_at_risk": sum(
            1 for d in at_risk if d["direction_agrees"]
        ),
        "direction_rows": direction,
        "harmed_rows": len(guard_cases),
        "hidden_harms_the_aggregate_would_have_missed": len(hidden_cases),
        "hidden_harms_the_guard_catches": sum(
            1 for g in hidden_cases if g["guard_condition_fires"]
        ),
        "harmed_rows_the_aggregate_would_have_caught_anyway": (
            len(guard_cases) - len(hidden_cases)
        ),
        "guard_rows": guard_cases,
        "circularity_note": (
            "the harm line that defines a harmed row is the same line the "
            "guard uses, so 'the guard catches a harmed row' is true by "
            "construction and is not reported as a result.  What is reported "
            "is the subset where the aggregate was positive -- the case the "
            "guard exists for -- and the count of rows the aggregate alone "
            "would have caught."
        ),
        "mismatches": misses,
        "verdict": "SUPPORTED" if not misses else "SOURCE_REPLAY_MISMATCH",
    }


def run() -> int:
    started = time.perf_counter()
    raw_traffic, raw_noaa = traffic_rows(), noaa_rows()
    traffic, dropped_t = dedupe(raw_traffic)
    noaa, dropped_n = dedupe(raw_noaa)

    card_traffic = compile_card(traffic, "traffic evidence only")
    card_noaa = compile_card(noaa, "noaa evidence only")
    card_both = compile_card(traffic + noaa, "both domains")

    c1 = replay(card_traffic, noaa, "C1: compiled on traffic, checked on NOAA")
    c2 = replay(card_noaa, traffic, "C2: compiled on NOAA, checked on traffic")
    c3 = replay(card_both, traffic + noaa, "C3: compiled on both, replayed on both")
    c3["INTERNAL_CONSISTENCY_ONLY"] = (
        "this arm replays a card on the very episodes it was induced from.  "
        "It can only fail, never confirm, and it is not transfer evidence."
    )

    payload: dict[str, Any] = {
        "protocol_version": "shared_capability_candidate_v1",
        "llm_calls": 0,
        "consumer_retrains": 0,
        "smd_identity": (
            "SMD is a third-domain CANDIDATE, not an established third "
            "domain.  Nothing in this file is evidence about SMD; it appears "
            "only as a substrate observation constraining what the card may "
            "lean on."
        ),
        "evidence_pool": {
            "restricted_to": {
                "programs": list(OUTLIER_FAMILY),
                "guard": "operator-independent per-series harm guard, two actions",
            },
            "dedupe_key": "domain x window x plan x outcome",
            "traffic": {
                "before_dedupe": len(raw_traffic),
                "after_dedupe": len(traffic),
                "collapsed": dropped_t,
                "episodes": [r["episode"] for r in traffic],
            },
            "noaa": {
                "before_dedupe": len(raw_noaa),
                "after_dedupe": len(noaa),
                "collapsed_count": len(dropped_n),
                "collapsed_examples": dropped_n[:8],
                "episodes": [r["episode"] for r in noaa],
            },
            "total_after_dedupe": len(traffic) + len(noaa),
        },
        "evidence_rows": {"traffic": traffic, "noaa": noaa},
        "candidate": card_both,
        "lodo": {"C1": c1, "C2": c2, "C3": c3},
    }
    if not card_both.get("compiles"):
        payload["verdict"] = "NO_COMPRESSIBLE_SHARED_CONTEXT"
        payload["verdict_reason"] = card_both.get("why")
    elif c1.get("verdict") == "SOURCE_REPLAY_MISMATCH" or c2.get(
        "verdict"
    ) == "SOURCE_REPLAY_MISMATCH":
        payload["verdict"] = "SOURCE_REPLAY_MISMATCH"
        payload["verdict_reason"] = (
            "C1 mismatches %s; C2 mismatches %s"
            % (c1.get("mismatches"), c2.get("mismatches"))
        )
    else:
        payload["verdict"] = "CANDIDATE_COMPILES"
        payload["verdict_reason"] = (
            "the card compiles from %d deduped evidence rows and both "
            "leave-one-domain-out directions replay without a mismatch"
            % payload["evidence_pool"]["total_after_dedupe"]
        )
        payload["lodo_transfer_supported"] = {
            "traffic_to_noaa": c1.get("verdict"),
            "noaa_to_traffic": c2.get("verdict"),
        }
    payload["wall_seconds"] = time.perf_counter() - started
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("verdict", payload["verdict"], flush=True)
    return 0


def _markdown(payload: dict[str, Any]) -> str:
    pool, card = payload["evidence_pool"], payload["candidate"]
    lines = [
        "# Shared Capability candidate v1",
        "",
        "**Verdict: `%s`** -- %s" % (payload["verdict"], payload["verdict_reason"]),
        "",
        payload["smd_identity"],
        "",
        "## B1 -- evidence pool",
        "",
        "Restricted to %s plus an operator-independent per-series harm guard. "
        "Dedupe key: `%s`." % (", ".join("`%s`" % p for p in OUTLIER_FAMILY),
                               pool["dedupe_key"]),
        "",
        "| domain | before dedupe | after dedupe |",
        "| --- | ---: | ---: |",
        "| traffic | %d | %d |" % (pool["traffic"]["before_dedupe"],
                                    pool["traffic"]["after_dedupe"]),
        "| noaa | %d | %d |" % (pool["noaa"]["before_dedupe"],
                                 pool["noaa"]["after_dedupe"]),
        "| **total** | %d | **%d** |" % (
            pool["traffic"]["before_dedupe"] + pool["noaa"]["before_dedupe"],
            pool["total_after_dedupe"]),
        "",
        "Surviving episodes: %s" % ", ".join(
            "`%s`" % e for e in pool["traffic"]["episodes"] + pool["noaa"]["episodes"]
        ),
        "",
    ]
    if card.get("compiles"):
        cap = card["capability"]
        lines.extend([
            "## The card", "",
            "```json",
            json.dumps({
                "id": cap["id"],
                "status": card["status"],
                "authorization": card["authorization"],
                "target_support_required": card["target_support_required"],
                "grants_confirmation_free_try": card["grants_confirmation_free_try"],
                "programs": cap["programs"],
                "guard": cap["guard"],
            }, indent=1, ensure_ascii=False),
            "```",
            "",
            "**What it recommends.** %s" % cap["what_it_recommends"],
            "",
            "### Applicability conditions (deployment-observable only)",
            "",
            "| feature | requirement | why |",
            "| --- | --- | --- |",
        ])
        for cond in card["applicability_conditions"]["conditions"]:
            lines.append("| `%s` | %s | %s |" % (
                cond["feature"], cond["requirement"], cond["why"]))
        lines.extend([
            "",
            "### Out of scope", "",
            "- Families: %s." % ", ".join(card["out_of_scope"]["families"]),
            "- %s" % card["out_of_scope"]["declared_because"],
        ])
        for p in card["out_of_scope"]["evidence_pointers"]:
            lines.append("- Evidence: `%s`" % p)
        lines.append("")
    lines.extend(["## C -- leave-one-domain-out dry replay", "",
                  "| arm | episodes | at risk | vacuous | direction agrees "
                  "(at risk) | hidden harms | caught | verdict |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for key in ("C1", "C2", "C3"):
        arm = payload["lodo"][key]
        if not arm.get("ran"):
            lines.append("| %s | -- | -- | -- | -- | -- | -- | `did not run: %s` |"
                         % (arm["label"], arm.get("why")))
            continue
        lines.append("| %s | %d | %d | %d | %d / %d | %d | %d | `%s` |" % (
            arm["label"], arm["episodes_checked"],
            arm["episodes_that_put_the_card_at_risk"], arm["vacuous_passes"],
            arm["direction_agreements_among_at_risk"],
            arm["episodes_that_put_the_card_at_risk"],
            arm["hidden_harms_the_aggregate_would_have_missed"],
            arm["hidden_harms_the_guard_catches"], arm["verdict"]))
    lines.extend(["", "Only the *at risk* column is evidence: a row whose "
                  "program the card does not claim passes vacuously.  %s"
                  % payload["lodo"]["C1"].get("circularity_note", ""), ""])
    lines.extend([
        "",
        "**C3 is not transfer evidence.** %s"
        % payload["lodo"]["C3"].get("INTERNAL_CONSISTENCY_ONLY", ""),
        "",
        "## Cost", "",
        "- LLM calls: 0.  Consumer retrains: 0.",
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(run())
