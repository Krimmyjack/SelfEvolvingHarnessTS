"""What the candidates the replay screen refused would actually have delivered.

DEV-AUTO-1 found that the production replay screen is **absolute** -- any one
applicable processed cell that breaches a risk line eliminates a candidate --
while the card that holds deployment rights is never asked to pass it.  On this
course the parent fails that screen at all sixteen steps, and five proposals
that beat the parent's served-population utility by more than ``material``,
without being worse than the parent on either risk line, were eliminated by it.

That is a statement about the **evidence face**.  This audit turns it into a
statement about the outcome: each distinct blocked program is scored on the
sixteen development units after the k1 boundary, on both faces, over the full
served population, beside the ancestor.

Time discipline
---------------
The programs here were named by the live proposer from pre-boundary and
already-processed evidence only; no future reading reached it.  Scoring them
afterwards on the later units is a **post-hoc oracle read of the candidate
space** -- the plan's L2 layer -- and is reported as one.  It is not a claim
that the screen should have admitted them, and it grants nothing: no Skill is
activated, no counter moves, no gate or threshold is changed.

Cost is billed to DEV-AUTO-1, not to M-R0k: the shared prediction store is
reused for the ancestor's readings (already paid for), and only the new
programs are fitted.

Run:  python -m evaluation.main_protocol_p4.audit_dev_auto1_blocked_candidates
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_m_r0k_scope_workflow_development as m_r0k
from evaluation.main_protocol_p4 import smoke_m_r0k_scope_workflow as smoke

ART = base.ART
FACES = m_r0k.FACES

#: This audit's own ceiling, inside the DEV-AUTO-1 package allowance.
MAX_FITS = 200


def _blocked_programs(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every distinct program the screen refused, with why it is interesting.

    Kept whole: a program that beat the parent and one that did not are both
    listed, so the table is not a selection of the flattering half.
    """
    rows: dict[str, dict[str, Any]] = {}
    for step in report.get("revision_steps") or ():
        for verdict in step.get("evaluations") or ():
            if verdict.get("kind") != "workflow_program":
                continue
            label = verdict.get("program_replayed")
            steps = verdict.get("steps_replayed")
            if not label or not steps:
                continue
            relative = verdict.get("relative_to_the_parent_screen") or {}
            comparison = verdict.get("comparison_with_the_parent") or {}
            row = rows.setdefault(label, {
                "label": label, "steps": steps, "seen_at_positions": [],
                "outcomes": [], "beat_the_parent_at": [],
                "would_be_kept_under_4_5_c_at": [],
                "screen_utility": [], "parent_utility": [],
            })
            row["seen_at_positions"].append(step["position"])
            row["outcomes"].append(verdict.get("outcome"))
            if relative.get("candidate_beats_the_parent_utility_by_material"):
                row["beat_the_parent_at"].append(step["position"])
            if relative.get("would_be_kept_under_4_5_c_with_the_parent_as_the_bar"):
                row["would_be_kept_under_4_5_c_at"].append(step["position"])
            row["screen_utility"].append(
                comparison.get("candidate_mean_aggregate_gain"))
            row["parent_utility"].append(
                comparison.get("parent_mean_aggregate_gain"))
    return sorted(rows.values(), key=lambda r: r["label"])


def _steps_tuple(steps: Sequence[Mapping[str, Any]]
                 ) -> tuple[tuple[str, dict], ...]:
    return tuple((str(s["op"]), dict(s.get("params") or {})) for s in steps)


def _build_program(label: str, steps: Sequence[tuple], population,
                   cache: runner.ReplayPredictionCache,
                   store: dict[str, Any]) -> dict[str, Any]:
    """``m_r0k._build_program`` with its self-check made degenerate-safe.

    M-R0k cross-checks each stored entry against the production cache by asking
    both for the reading under the **whole** served set.  That is exact for its
    three candidates, and it raises for a program that flattens a served
    context: ``cache.reading`` refuses a Scope that reaches a degenerate series,
    which is the production rule and correct.  Some programs proposed here do
    flatten some cells, so the check is run under the legal set -- every served
    series the program may treat -- which is the set any legal Scope is a subset
    of.  The rule is unchanged; only the set the two sides are compared on is.
    """
    built, reused, refused, faults, checked = 0, 0, [], [], 0
    for row in population:
        ctx = m_r0k.opp._unit_ctx(row["unit"])
        for face, origin in m_r0k._faces_for(ctx).items():
            key = m_r0k._entry_key(label, row["position"], face)
            if key in store:
                reused += 1
                continue
            before = cache.physical_fits
            cache_key = cache.key(ctx.unit, origin,
                                  runner.forecast_p4._config(origin), steps)
            if (cache_key not in cache._entries
                    and cache.physical_fits + 2 > MAX_FITS):
                refused.append({"position": row["position"], "face": face,
                                "why": "REFUSED_AT_THE_CEILING",
                                "spent": cache.physical_fits})
                continue
            try:
                entry = cache._build(ctx, origin, tuple(steps))
                cache._entries[cache_key] = entry
            except Exception as exc:  # noqa: BLE001 - recorded, never hidden
                faults.append({"position": row["position"], "face": face,
                               "fault": "%s: %s" % (type(exc).__name__,
                                                    str(exc)[:200]),
                               "fits_spent_before_the_fault":
                                   cache.physical_fits - before})
                continue
            spent = cache.physical_fits - before
            store[key] = {
                "program": label, "position": row["position"], "face": face,
                "origin": origin,
                "verifier_passed": bool(entry.get("verifier_passed")),
                "eval_uids": list(entry.get("eval_uids") or ()),
                "raw_per_view": [float(v)
                                 for v in (entry.get("raw_per_view") or ())],
                "program_per_view": [float(v) for v in
                                     (entry.get("program_per_view") or ())],
                "degenerate_uids": list(entry.get("degenerate_uids") or ()),
                "physical_fits": int(spent),
            }
            built += 1
            if entry.get("verifier_passed"):
                legal = frozenset(
                    uid for uid in ctx.eval_uids
                    if uid not in set(entry.get("degenerate_uids") or ()))
                mine = m_r0k._reading_from_entry(store[key], legal)
                theirs = cache.reading(ctx, origin, tuple(steps), legal)
                for field in ("treated", "served", "aggregate_gain",
                              "harmed_fraction", "max_single_series_harm"):
                    if repr(mine[field]) != repr(theirs[field]):
                        raise m_r0k.InstrumentFault(
                            "the stored re-mask disagrees with the production "
                            "cache on %s at position %d %s: %r vs %r"
                            % (field, row["position"], face, mine[field],
                               theirs[field]))
                checked += 1
            m_r0k._save_store(store)
    return {"program": label,
            "steps": [{"op": op, "params": dict(params)} for op, params in steps],
            "entries_built": built, "entries_reused_from_the_store": reused,
            "entries_cross_checked_against_the_production_cache": checked,
            "refused_at_the_ceiling": refused, "faults": faults,
            "physical_fits_after": cache.physical_fits}


def build(source: str = "run2", only_kept: bool = True) -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    path = ART / ("dev_auto1_skill_revision__%s.json" % source)
    report = json.loads(path.read_text(encoding="utf-8"))
    blocked = _blocked_programs(report)
    wanted = [row for row in blocked
              if row["would_be_kept_under_4_5_c_at"]] if only_kept else blocked

    doc = base.load(live.ORDERING)
    population, _excluded = m_r0k._population(doc)
    after = [row for row in population if row["side"] == "after_the_boundary"]
    store = m_r0k._load_store()
    cache = runner.ReplayPredictionCache(live.ARM)
    cache.physical_fits = 0        # billed to DEV-AUTO-1, not to M-R0k

    builds, scored = [], {}
    labels = ["ANCESTOR"]
    for row in wanted:
        label = "CAND_%s" % row["label"]
        labels.append(label)
        builds.append(_build_program(label, _steps_tuple(row["steps"]),
                                           after, cache, store))
    for label in labels:
        program = "ANCESTOR" if label == "ANCESTOR" else label
        scored[label] = m_r0k._policy_rows(label, program, m_r0k.ANCESTOR_SCOPE,
                                           after, store, "ANCESTOR")

    table = {}
    for label in labels:
        table[label] = {face: m_r0k._summarise(scored[label], face,
                                               side="after_the_boundary")
                        for face in FACES}

    ancestor = table["ANCESTOR"]
    headline = {}
    for label in labels:
        headline[label] = {}
        for face in FACES:
            mine = table[label][face]
            theirs = ancestor[face]
            value = mine["mean_executable_aggregate_gain"]
            base_value = theirs["mean_executable_aggregate_gain"]
            headline[label][face] = {
                "mean_aggregate_gain": value,
                "vs_the_ancestor": (round(float(value) - float(base_value), 6)
                                    if value is not None
                                    and base_value is not None else None),
                "harmed_series": mine["total_harmed_series"],
                "worst_single_series_harm": mine["worst_single_series_harm"],
                "treated": mine["total_treated"],
                "authoritative_gate_passes_out_of_16": mine["gate_passes"],
            }

    return {
        "stage": "DEV_AUTO1_BLOCKED_CANDIDATES",
        "what_this_is": ("a post-hoc read of what the screen-refused programs "
                         "would have delivered on the sixteen units after the "
                         "k1 boundary -- the plan's L2 layer, not a claim "
                         "about what the screen should have done"),
        "source_report": str(path.relative_to(base.ROOT)),
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "selection_rule": (
            "every distinct workflow program the screen refused that beat the "
            "parent's served-population utility by material and was no worse "
            "than the parent on either risk line"
            if only_kept else "every distinct workflow program the screen saw"),
        "every_blocked_program_seen": blocked,
        "programs_scored": [row["label"] for row in wanted],
        "program_builds": builds,
        "headline_on_the_16_units_after_the_boundary": headline,
        "per_candidate": table,
        "per_unit": {label: scored[label] for label in labels},
        "how_to_read_it": (
            "the delayed face is the authoritative one; the Support face is "
            "evidence.  A candidate that beats the ancestor here was still "
            "never deployed and never held a right: this is the size of what "
            "the screen refused, measured, not a verdict on the screen"),
        "actual_cost": {"llm_calls": 0,
                        "physical_consumer_fits": cache.physical_fits,
                        "ceiling": MAX_FITS,
                        "within_ceiling": cache.physical_fits <= MAX_FITS,
                        "billed_to": "DEV-AUTO-1"},
        "boundary": {"skills_activated": 0, "deployment_rights_issued": 0,
                     "lifecycle_counters_moved": 0, "thresholds_changed": 0,
                     "operators_added": 0, "evaluation_face_reads": 0,
                     "sealed_reads": 0, "target_held_in_reads": 0},
        "code_state": base.version_check(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    source = argv[argv.index("--source") + 1] if "--source" in argv else "run2"
    out = ART / ("dev_auto1_blocked_candidates__%s.json" % source)
    if out.exists() and "--allow-overwrite" not in argv:
        sys.stderr.write("refusing to overwrite %s\n" % out)
        return 2
    report = build(source=source, only_kept="--all" not in argv)
    out.write_text(json.dumps(drafts._plain(report), indent=1,
                              ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("wrote %s" % out)
    print(json.dumps(drafts._plain({
        "programs": report["programs_scored"],
        "headline": report["headline_on_the_16_units_after_the_boundary"],
        "cost": report["actual_cost"]}), indent=1, ensure_ascii=False,
        default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
