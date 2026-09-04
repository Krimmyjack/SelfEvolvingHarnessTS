"""The course readout: the curve, the descriptive sign pattern, P2, H1-H3.

sol's final ruling of 2026-09-03 §2: no significance gate.  The pre-registered
criteria are qualitative (terminal difference above zero in at least 2 of 3
orderings, cohort difference above zero in at least 3 of 4 cohorts, harm not
worse); the exact binomial probability of the cohort sign pattern is reported
with its floor stated, and the cohort bootstrap is a description of spread.

Reads the scoring ledger, never the bank
----------------------------------------
The evaluation face is scored and never fed back, so the readout is allowed to
see it and no arm is.  Keeping that separation in the *file layout* rather than
in a comment is the point: this module opens ``scoring_ledger`` and never opens
an Episode bank, so there is no code path by which a curve number could reach a
prompt.

What it refuses to say
----------------------
A verdict from fewer than all three orderings.  ``HEC1_INCONCLUSIVE`` is the
honest reading until every ordering is in and each has reached
``hec1_scoreability.MIN_PAIRED_CURVE_POINTS`` -- 19, being ``ceil(0.8 x 23)``
over the **scoreable** units rather than over the 26 scheduled ones, because
three of those can never carry a curve point at all.  The pre-registered
criteria say so before any number exists, which is why the 2026-09-04 morning
report is not allowed to contain a curve conclusion.

0 LLM, 0 fits: every quantity here is arithmetic over readings the course
already took.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import hec1_scoreability as scoreability

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts/main_protocol"
OUT_JSON = ARTIFACTS / "hec1_readout.json"
OUT_MD = ARTIFACTS / "hec1_readout.md"

ORACLE_BANNER = (
    "ORACLE_WALL: computed after the course from the scoring ledger only.  No "
    "Episode bank is opened and no value here may reach a prompt."
)

ONLINE_ARM = "A5-online"
FROZEN_ARM = "A5-frozen"
ONLINE_ARM_EMPTY_K0 = "A3-online"
FROZEN_ARM_EMPTY_K0 = "A3-frozen"


def _sign_pattern(differences: Sequence[float], *, unit_name: str
                  ) -> dict[str, Any]:
    """Sign counts and the exact one-sided binomial probability -- described,
    never judged.

    sol's final ruling §2: the sign test is not a pass gate.  With four cohorts
    the probability has a floor of 1/16 = 0.0625 even at 4/4, so no ``alpha``
    and no ``significant`` field exist here; a reader who wants to call 0.0625
    "significant" has to write that word themselves.  Ties are dropped rather
    than split: a unit where online and frozen deployed the same thing carries
    no information about whether writing back helped.
    """
    positive = sum(1 for value in differences if value > 0)
    negative = sum(1 for value in differences if value < 0)
    ties = sum(1 for value in differences if value == 0)
    trials = positive + negative
    floor = (1.0 / (2 ** trials)) if trials else None
    if trials == 0:
        return {"unit": unit_name, "n": len(differences), "ties": ties,
                "trials": 0, "positive": 0, "negative": 0,
                "exact_binomial_probability": None, "floor_at_all_positive": None,
                "why": "every paired difference was a tie",
                "role": "DESCRIPTIVE"}
    tail = sum(math.comb(trials, k) for k in range(positive, trials + 1))
    return {
        "unit": unit_name,
        "n": len(differences),
        "ties": ties,
        "trials": trials,
        "positive": positive,
        "negative": negative,
        "exact_binomial_probability": round(tail / (2 ** trials), 6),
        "floor_at_all_positive": round(floor, 6),
        "role": "DESCRIPTIVE",
        "note": ("exact one-sided binomial probability of the observed sign "
                 "pattern; not a test, not a gate; the floor is what 100%% "
                 "agreement would read on %d trials" % trials),
    }


def _bootstrap_mean(values: Sequence[float], *, draws: int = 2000,
                    seed: int = 20260903) -> dict[str, Any]:
    """Percentile interval of the mean under resampling of the given units.

    Description of uncertainty only (sol final ruling §2).  With four cohorts
    the interval is crude, and it says so.
    """
    import random  # noqa: PLC0415 - only this description needs it

    values = [float(v) for v in values]
    if not values:
        return {"n": 0, "mean": None, "interval_90": None, "role": "DESCRIPTIVE"}
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo, hi = means[int(0.05 * draws)], means[int(0.95 * draws) - 1]
    return {"n": len(values), "mean": round(sum(values) / len(values), 6),
            "interval_90": [round(lo, 6), round(hi, 6)], "draws": draws,
            "seed": seed, "role": "DESCRIPTIVE",
            "note": "percentile bootstrap over %d units; with n=4 this is a "
                    "coarse description, not a confidence statement" % len(values)}


def _cumulative(rows: Sequence[Mapping[str, Any]], arm: str) -> list[float]:
    total, series = 0.0, []
    for row in rows:
        total += float((row.get("arms") or {}).get(arm, {}).get(
            "aggregate_gain") or 0.0)
        series.append(round(total, 6))
    return series


def _harm_events(rows: Sequence[Mapping[str, Any]], arm: str) -> int:
    events = 0
    for row in rows:
        reading = (row.get("arms") or {}).get(arm) or {}
        if (float(reading.get("harmed_fraction") or 0.0)
                > contract.RISK["max_harmed_fraction"]
                or float(reading.get("max_single_series_harm") or 0.0)
                > contract.RISK["max_single_series_harm"]):
            events += 1
    return events


def _arm_pair(ledger: Mapping[str, Any]) -> tuple[str, str]:
    arms = set()
    for row in ledger.get("units") or ():
        arms.update((row.get("arms") or {}).keys())
    if ONLINE_ARM in arms and FROZEN_ARM in arms:
        return ONLINE_ARM, FROZEN_ARM
    return ONLINE_ARM_EMPTY_K0, FROZEN_ARM_EMPTY_K0


def _gain(row: Mapping[str, Any], arm: str) -> float:
    return float((row.get("arms") or {}).get(arm, {}).get(
        "aggregate_gain") or 0.0)


def _p2_survival_chain(ledger: Mapping[str, Any], online: str, frozen: str
                       ) -> dict[str, Any]:
    """P2, mechanically: revised Draft -> Active on a new unit -> re-encounter
    deployment by the online arm with (online - frozen) > 0 on that unit.

    Read from the cells and the lifecycle only.  A chain is a program signature
    that (a) belongs to a Draft with at least one recorded revision, (b) was the
    deployed program of an online cell with ``activated=True`` at position p,
    and (c) was deployed again by the online arm at a position after p with a
    positive paired difference on that unit's evaluation face.
    """
    from evaluation.main_protocol_p4 import outer_loop  # noqa: PLC0415

    cells = [row for row in (ledger.get("cells") or ())
             if row.get("arm") == online]
    frozen_by_pos = {int(row["position"]): row
                     for row in (ledger.get("cells") or ())
                     if row.get("arm") == frozen}
    revised: dict[str, dict[str, Any]] = {}
    for draft in ((ledger.get("lifecycle") or {}).get(online) or {}).get(
            "drafts") or ():
        if int(draft.get("revisions") or 0) >= 1:
            signature = outer_loop._program_signature(draft.get("program_steps"))
            revised.setdefault(signature, {"draft_id": draft.get("draft_id"),
                                           "revisions": draft.get("revisions")})
    chains = []
    for signature, draft in revised.items():
        activated_at = [int(row["position"]) for row in cells
                        if row.get("activated")
                        and outer_loop._program_signature(row.get("deployed"))
                        == signature]
        if not activated_at:
            continue
        first = min(activated_at)
        for row in cells:
            position = int(row["position"])
            if position <= first:
                continue
            if outer_loop._program_signature(row.get("deployed")) != signature:
                continue
            online_gain = float((row.get("evaluation") or {}).get(
                "aggregate_gain") or 0.0)
            frozen_gain = float(((frozen_by_pos.get(position) or {}).get(
                "evaluation") or {}).get("aggregate_gain") or 0.0)
            if online_gain - frozen_gain > 0:
                chains.append({**draft, "program_signature": signature,
                               "activated_at_position": first,
                               "re_encounter_position": position,
                               "deployed_via": row.get("deployed_via"),
                               "paired_difference": round(
                                   online_gain - frozen_gain, 6)})
                break
    return {"holds": bool(chains), "chains": chains,
            "revised_drafts": len(revised),
            "definition": contract.VERDICTS["P2_definition"]}


def read_ordering(path: Path) -> dict[str, Any]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    ran = list(ledger.get("units") or ())
    online, frozen = _arm_pair(ledger)
    # Only units where **both** arms actually scored enter the curve.  A missing
    # evaluation reads as 0.0 through the ordinary accessor, and a zero
    # difference is indistinguishable from "the arms tied here" -- so the three
    # unscoreable units would otherwise contribute three fabricated ties, each
    # of which dilutes the sign pattern and flattens the cumulative curve.
    rows = scoreability.paired_curve_points(ran, online, frozen)
    unpaired = [
        {"block": (row.get("unit") or {}).get("block"),
         "origin": (row.get("unit") or {}).get("origin"),
         "scoreable": scoreability.unit_is_scoreable(row.get("unit"))}
        for row in ran if row not in rows
    ]
    differences = [round(_gain(row, online) - _gain(row, frozen), 6)
                   for row in rows]
    online_curve = _cumulative(rows, online)
    frozen_curve = _cumulative(rows, frozen)
    cumulative_difference = [round(a - b, 6)
                             for a, b in zip(online_curve, frozen_curve)]
    by_cohort: dict[str, list[float]] = {}
    for row, difference in zip(rows, differences):
        by_cohort.setdefault(str((row.get("unit") or {}).get("block")),
                             []).append(difference)
    midpoint = len(rows) // 2
    return {
        "artifact": path.name,
        "ordering": ledger.get("ordering"),
        "mode": ledger.get("mode"),
        "code_state": ledger.get("code_state"),
        "units_run": len(ran),
        "paired_curve_points": len(rows),
        "units_without_a_paired_point": unpaired,
        "min_paired_curve_points": scoreability.MIN_PAIRED_CURVE_POINTS,
        "meets_completion_floor": (
            len(rows) >= scoreability.MIN_PAIRED_CURVE_POINTS),
        "arm_pair": {"online": online, "frozen": frozen},
        "cumulative_online": online_curve,
        "cumulative_frozen": frozen_curve,
        "cumulative_difference": cumulative_difference,
        "terminal_difference": (cumulative_difference[-1] if rows else None),
        "secondary": {
            "auc_of_cumulative_difference": round(sum(cumulative_difference), 6),
            "midpoint_difference": (cumulative_difference[midpoint - 1]
                                    if midpoint else None),
        },
        "paired_differences": differences,
        "unit_sign_pattern": _sign_pattern(differences, unit_name="unit"),
        "cohort_means_this_ordering": {
            block: round(sum(values) / len(values), 6)
            for block, values in sorted(by_cohort.items()) if values},
        "harm_events": {online: _harm_events(rows, online),
                        frozen: _harm_events(rows, frozen)},
        "harm_online_not_worse": (
            _harm_events(rows, online) <= _harm_events(rows, frozen)),
        "p2_survival_chain": _p2_survival_chain(ledger, online, frozen),
        "lifecycle": ledger.get("lifecycle"),
        "h1_h3": ledger.get("h_readings"),
        "slow_vs_scopefit": ledger.get("shadow_records"),
        "fast_decisions": ledger.get("fast_decisions"),
        "deployed_via_counts": {
            arm: _deployed_via_counts(ledger, arm) for arm in (online, frozen)},
        "attribution_rule": "post hoc, one rule for every ordering; see "
                            "_attribute_cells",
        "lost_activations": sum(
            1 for row in _attribute_cells(ledger, online)
            if row["lost_activation_post_hoc"]),
        "ledgers": ledger.get("ledgers"),
        "replay_fit_allowance": ledger.get("replay_fit_allowance"),
    }


def _attribute_cells(ledger: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
    """One attribution rule for every ordering, applied post hoc.

    forward_live was recorded by the runner as of 2026-09-03 11:28, which knew
    three ``deployed_via`` labels and no ``lost_activation``; later orderings
    may carry richer fields.  Reading the runner's label would compare the
    orderings under two vocabularies, so the readout derives both facts itself
    from what every version records: the deployed program, the activation
    flags, the gate record and the candidate id.  The Active-program set is
    rebuilt by walking the arm's own cells in order (K0 is empty in this
    course, so every Active program was minted inside it).
    """
    from evaluation.main_protocol_p4 import outer_loop  # noqa: PLC0415

    rows = sorted((row for row in (ledger.get("cells") or ())
                   if row.get("arm") == arm), key=lambda r: int(r["position"]))
    active_programs: set[str] = set()
    out = []
    for row in rows:
        steps = row.get("deployed")
        signature = outer_loop._program_signature(steps) if steps else ""
        recorded = str(row.get("deployed_via") or "")
        winner_id = str(row.get("winner_candidate_id") or "")
        from_card = (recorded == "recalled_skill"
                     or bool(row.get("winner_from_skill_candidate")))
        from_draft = (recorded == "resupplied_draft"
                      or winner_id.startswith("resupplied_draft_"))
        in_active = bool(signature) and signature in active_programs
        via = ("identity" if not steps else
               "recalled_skill" if from_card else
               "resupplied_draft" if from_draft else
               "searched_active_program" if in_active else
               "searched_this_unit")
        gate = row.get("gate_disagreement") or {}
        lost = bool(gate.get("may_activate")) and not row.get("activated") \
            and bool(steps) and not in_active and arm.endswith("-online")
        out.append({"position": int(row["position"]),
                    "deployed_via_post_hoc": via,
                    "deployed_via_as_recorded": recorded or None,
                    "program_in_active_set_at_start": in_active,
                    "lost_activation_post_hoc": lost})
        if row.get("activated") and signature:
            active_programs.add(signature)
    return out


def _deployed_via_counts(ledger: Mapping[str, Any], arm: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _attribute_cells(ledger, arm):
        key = row["deployed_via_post_hoc"]
        counts[key] = counts.get(key, 0) + 1
    return counts


#: Only these three labels are the course.  Globbing ``hec1_course_*.json`` swept
#: in offline probes, pytest fixtures and 1-unit chain tests, any of which would
#: have entered the curve as if it were a live ordering.  The label is checked
#: *and* the artifact's own ``offline`` flag is checked, because a name is a
#: convention and the flag is a fact.
LIVE_LABELS = ("forward_live", "reverse_live", "interleaved_live")


#: A resumed run writes ``hec1_course_<label>.resumed.json`` beside the blocked
#: original.  The resumed file is the complete course; the original is the
#: record of the fault.  Both must be recognised or a run that hit a relay
#: outage and resumed could never be read out.
RESUMED_SUFFIX = ".resumed"


#: ``<prefix>forward_live`` etc.  The prefix is what keeps a scientific chain
#: (e.g. ``v11_``) apart from the v1 shakedown that already owns the bare
#: ``forward_live`` label; all three orderings must share one prefix.
LABEL_RE = re.compile(r"^(?P<prefix>.*?)(?P<ordering>forward|reverse|interleaved)_live$")


def _live_courses(prefix: str | None = None
                  ) -> tuple[list[Path], list[dict[str, Any]]]:
    kept: dict[str, Path] = {}
    prefixes: dict[str, str] = {}
    rejected = []
    for path in sorted(ARTIFACTS.glob("hec1_course_*.json")):
        label = path.stem[len("hec1_course_"):]
        resumed = label.endswith(RESUMED_SUFFIX)
        if resumed:
            label = label[:-len(RESUMED_SUFFIX)]
        match = LABEL_RE.match(label)
        if not match:
            rejected.append({"artifact": path.name,
                             "why": "not one of the three live ordering labels"})
            continue
        if prefix is not None and match.group("prefix") != prefix:
            rejected.append({"artifact": path.name,
                             "why": "label prefix %r is not the requested %r"
                                    % (match.group("prefix"), prefix)})
            continue
        ordering_label = "%s_live" % match.group("ordering")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            rejected.append({"artifact": path.name,
                             "why": "unreadable: %s" % str(exc)[:120]})
            continue
        if payload.get("offline"):
            rejected.append({"artifact": path.name,
                             "why": "the artifact declares itself offline"})
            continue
        mode = payload.get("mode")
        if mode == "shakedown":
            rejected.append({"artifact": path.name,
                             "why": "shakedown: instrument-only, enters no curve"})
            continue
        if mode != "scientific" or not (payload.get("code_state") or {}).get(
                "code_commit"):
            rejected.append({"artifact": path.name,
                             "why": "no scientific mode / code_commit stamp "
                                    "(pre-v1.1 run; FORWARD_SHAKEDOWN by ruling)"})
            continue
        if payload.get("status") != "COMPLETE":
            rejected.append({"artifact": path.name,
                             "why": "status is %s" % payload.get("status")})
            continue
        # One prefix for the whole chain: a forward from one chain and a reverse
        # from another are not one experiment.
        if prefixes and match.group("prefix") not in prefixes.values():
            rejected.append({"artifact": path.name,
                             "why": "label prefix %r differs from the chain's %r"
                                    % (match.group("prefix"),
                                       sorted(set(prefixes.values()))[0])})
            continue
        label = ordering_label
        # A complete resumed artifact supersedes a complete original of the
        # same label (it cannot exist unless the original was blocked), and two
        # complete artifacts for one label is itself worth listing.
        if label in kept and not resumed:
            rejected.append({"artifact": path.name,
                             "why": "a resumed artifact for this label is kept"})
            continue
        if label in kept and resumed:
            rejected.append({"artifact": kept[label].name,
                             "why": "superseded by the resumed artifact"})
        kept[label] = path
        prefixes[label] = match.group("prefix")
    return [kept[label] for label in LIVE_LABELS if label in kept], rejected


def _code_commits(orderings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """sol v1.1 R-B: one commit, clean runner files, for every ordering."""
    commits = {row["artifact"]: (row.get("code_state") or {}).get("code_commit")
               for row in orderings}
    clean = {row["artifact"]: bool((row.get("code_state") or {}).get(
        "runner_files_clean")) for row in orderings}
    distinct = sorted({value for value in commits.values() if value})
    return {"code_commit_by_ordering": commits,
            "runner_files_clean_by_ordering": clean,
            "one_commit": len(distinct) == 1 and all(commits.values()),
            "all_clean": all(clean.values()) if clean else False,
            "commit": distinct[0] if len(distinct) == 1 else None}


def build(prefix: str | None = None) -> dict[str, Any]:
    started = time.time()
    found, rejected = _live_courses(prefix)
    orderings = [read_ordering(path) for path in found]
    code = _code_commits(orderings)
    # Against the frozen scoreability manifest, not against the 26 scheduled
    # units: three of those can never carry a curve point, so requiring
    # ceil(0.8 x 26) = 21 would demand two points that do not exist.
    complete = [row for row in orderings if row["meets_completion_floor"]]
    # sol v1.1: the terminal difference has to clear the **material** line,
    # 0.005 x 23 = 0.115, not merely be above zero.  A course that ends +0.02
    # over 23 units is 23 readings of nearly nothing.
    material_line = contract.P1_MATERIAL_TERMINAL_DIFFERENCE
    terminal_positive = sum(
        1 for row in orderings
        if row["terminal_difference"] is not None
        and row["terminal_difference"] >= material_line)
    terminal_above_zero = sum(
        1 for row in orderings
        if row["terminal_difference"] is not None
        and row["terminal_difference"] > 0)

    # Cohort endpoint d_c: the cohort's mean paired difference in each
    # ordering, averaged over the orderings present.  Four numbers.
    cohort_values: dict[str, list[float]] = {}
    for row in orderings:
        for block, value in (row.get("cohort_means_this_ordering") or {}).items():
            cohort_values.setdefault(block, []).append(float(value))
    cohort_d = {block: round(sum(values) / len(values), 6)
                for block, values in sorted(cohort_values.items()) if values}
    cohorts_positive = sum(1 for value in cohort_d.values() if value > 0)
    cohort_pattern = _sign_pattern(list(cohort_d.values()), unit_name="cohort")
    cohort_bootstrap = _bootstrap_mean(list(cohort_d.values()))
    harm_ok = all(row["harm_online_not_worse"] for row in orderings)
    p2_any = any(row["p2_survival_chain"]["holds"] for row in orderings)
    criteria = {
        "D_o_material_in_at_least_2_of_3": terminal_positive >= 2,
        "d_c_positive_in_at_least_3_of_4": cohorts_positive >= 3,
        "harm_online_not_worse_in_every_ordering": harm_ok,
        "P2_survival_chain_in_any_ordering": p2_any,
    }
    material = {
        "line": material_line,
        "derivation": "material 0.005 x %d scoreable units"
                      % scoreability.SCOREABLE_UNITS,
        "orderings_at_or_above_the_line": terminal_positive,
        "orderings_merely_above_zero": terminal_above_zero,
        "why_not_just_positive": (
            "a terminal difference of a few hundredths over 23 units is 23 "
            "readings of nearly nothing; positive is not material"),
    }

    if len(orderings) < len(contract.ORDERINGS) or len(complete) < len(orderings):
        verdict = "HEC1_INCONCLUSIVE"
        why = (
            "%d of %d orderings present and %d of those reached %d paired "
            "curve points; the pre-registered criteria need all three"
            % (len(orderings), len(contract.ORDERINGS), len(complete),
               scoreability.MIN_PAIRED_CURVE_POINTS))
    elif not (code["one_commit"] and code["all_clean"]):
        # sol v1.1 R-B: three orderings from two commits, or from a dirty
        # tree, are not one experiment; the reading is withheld, not averaged.
        verdict = "HEC1_INCONCLUSIVE"
        why = ("the three orderings do not share one clean commit: %s"
               % json.dumps(code["code_commit_by_ordering"], default=str))
    else:
        p1 = (criteria["D_o_material_in_at_least_2_of_3"]
              and criteria["d_c_positive_in_at_least_3_of_4"]
              and criteria["harm_online_not_worse_in_every_ordering"])
        if p1 and p2_any:
            verdict = "HEC1_EVOLUTION_SUPPORTED"
            why = ("P1 held (%d/3 orderings, %d/%d cohorts, harm not worse) and "
                   "at least one revised Draft survived a re-encounter"
                   % (terminal_positive, cohorts_positive, len(cohort_d)))
        elif p1:
            verdict = "HEC1_P1_ONLY__RECALL_ACCUMULATION"
            why = ("P1 held (%d/3 orderings, %d/%d cohorts, harm not worse) but "
                   "no revised Draft survived a re-encounter; the claim narrows "
                   "to ADD / recall-driven accumulation and does not qualify "
                   "for the Phase F seal"
                   % (terminal_positive, cohorts_positive, len(cohort_d)))
        else:
            verdict = "HEC1_EVOLUTION_NOT_SUPPORTED"
            why = ("P1 did not hold (%d/3 orderings, %d/%d cohorts, harm ok=%s); "
                   "see the first-fault map in the contract"
                   % (terminal_positive, cohorts_positive, len(cohort_d), harm_ok))
    return {
        "criteria": criteria,
        "code_freeze": code,
        "cohort_endpoint_d_c": cohort_d,
        "cohort_sign_pattern": cohort_pattern,
        "cohort_bootstrap": cohort_bootstrap,
        "statistics_role": (
            "DESCRIPTIVE development mechanism curve; no significance gate "
            "(sol final ruling 2026-09-03 §2)"),
        "stage": "HEC1_READOUT",
        "written_at": datetime.now().astimezone().isoformat(),
        "oracle_banner": ORACLE_BANNER,
        "contract_version": contract.VERSION,
        "data_version": contract.DATA_VERSION,
        "evidence_grade": "DEVELOPMENT (development mechanism curve)",
        "orderings_found": [row["artifact"] for row in orderings],
        "orderings_expected": list(contract.ORDERINGS),
        "scoreability": scoreability.to_dict(),
        "artifacts_rejected": rejected,
        "why_rejection_is_listed": (
            "a course artifact that is offline, incomplete or a test label must "
            "not enter the curve, and the reason it did not is worth reading"
        ),
        "orderings": orderings,
        "material_terminal_difference": material,
        "terminal_differences_above_zero": terminal_above_zero,
        "terminal_differences_at_or_above_material": terminal_positive,
        "statistics": contract.STATISTICS,
        "preregistered": contract.PREREGISTERED,
        "verdict": verdict,
        "why": why,
        "h1_h3_consistency": (
            "CONSISTENT / MIXED / NOT_OBSERVED, annotated per ordering once "
            "all three are in"),
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": 0,
            "held_out_reads": 0,
            "reads": "the scoring ledger only",
            "opens_episode_bank": False,
        },
        "wall_seconds": round(time.time() - started, 1),
    }


def _md(payload: Mapping[str, Any]) -> str:
    lines = [
        "# HEC-1 course readout",
        "",
        payload["oracle_banner"],
        "",
        "| item | value |",
        "| --- | --- |",
        "| orderings found | %s of %s |" % (
            len(payload["orderings_found"]), len(payload["orderings_expected"])),
        "| terminal differences above zero | %s |" % payload[
            "terminal_differences_above_zero"],
        "| verdict | **%s** |" % payload["verdict"],
        "",
        payload["why"] + ".",
        "",
    ]
    if payload["orderings"]:
        lines += ["| ordering | ran | paired pts (min %d) | terminal diff | AUC "
                  "| harm ok | P2 chain |" % scoreability
                  .MIN_PAIRED_CURVE_POINTS,
                  "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
        for row in payload["orderings"]:
            lines.append("| %s | %s | %s%s | %s | %s | %s | %s |" % (
                row["ordering"], row["units_run"],
                row["paired_curve_points"],
                "" if row["meets_completion_floor"] else " **short**",
                row["terminal_difference"],
                row["secondary"]["auc_of_cumulative_difference"],
                row["harm_online_not_worse"],
                row["p2_survival_chain"]["holds"]))
        lines += ["", "| cohort | d_c |", "| --- | ---: |"]
        for block, value in payload["cohort_endpoint_d_c"].items():
            lines.append("| %s | %s |" % (block, value))
        pattern = payload["cohort_sign_pattern"]
        lines += ["",
                  "Cohort sign pattern: %s positive of %s; exact binomial "
                  "probability %s (floor at all-positive %s) -- descriptive, not "
                  "a test." % (pattern.get("positive"), pattern.get("trials"),
                               pattern.get("exact_binomial_probability"),
                               pattern.get("floor_at_all_positive")),
                  ""]
    else:
        lines += [
            "No course artifact exists yet, so there is no curve to report. "
            "This is the expected state before Phase T runs, and it is not a "
            "negative result.", ""]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None,
                        help="write hec1_readout_<label>.{json,md}; required "
                             "once a readout artifact already exists, so the "
                             "one frozen readout is never overwritten")
    parser.add_argument("--prefix", default=None,
                        help="read only the chain whose labels carry this "
                             "prefix (e.g. v11_); the bare forward_live is the "
                             "v1 shakedown and is excluded by its own stamp")
    args = parser.parse_args(argv)
    payload = build(args.prefix)
    out_json = (OUT_JSON.with_name("hec1_readout_%s.json" % args.label)
                if args.label else OUT_JSON)
    out_md = out_json.with_suffix(".md")
    if out_json.exists():
        print("refusing to overwrite %s; pass --label <new label>"
              % out_json.relative_to(PROJECT_ROOT).as_posix())
        return 2
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    out_md.write_text(_md(payload), encoding="utf-8")
    print("orderings found : %d / %d" % (len(payload["orderings_found"]),
                                        len(payload["orderings_expected"])))
    print("verdict         : %s" % payload["verdict"])
    print("why             : %s" % payload["why"])
    print("wrote %s" % out_json.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
