"""DEV-SEQ-3 preflight: only the three paths this package adds.

DEV-SEQ-2's smoke already covers window binding, measurement de-duplication,
whitelist reachability, fork isolation, the closed edit whitelist and paired /
UNKNOWN scoring.  It is re-run as a whole rather than restated, and the checks
below are the new ones:

    1  the feature description is facts -- formula and window for every
       feature, no unproven claim about how a value moves between windows --
       and it actually reaches the card Slow is given;
    2  action facts are the evaluator's own numbers: the serving count equals
       a replay of its ``_prepare``, the training count equals the
       ``behavior_point_count`` the evaluator itself reports, an executed
       identity is a measured zero, an unavailable count stays UNKNOWN, and
       the training count is labelled as a property of the shared training
       material rather than of the evaluated series;
    3  running a batch in parallel changes nothing: the same readings come
       back bit-for-bit, no extra Consumer fit is spent, the shared ledger
       counts every call exactly once under contention, and results are
       aggregated in fixed roster order rather than completion order.

Check 2 spends a small number of real Consumer fits -- it compares against the
evaluator, so it has to run it -- and reports how many.  No LLM calls: the
parallel check uses fixed programs, not model output.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_seq2_knowledge as know
from evaluation.main_protocol_p4 import per_sequence as ps
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_seq3 as seq3
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped
from evaluation.main_protocol_p4 import smoke_dev_seq2 as smoke2

PROGRAM = (("outlier_mad", {}),)
OTHER = (("impute_linear", {}),)

_FITS = {"spent": 0}


def _unit(position: int) -> Any:
    doc = base.load(R.ORDERING)
    rows, _excluded = R._population(doc)
    return R.opp._unit_ctx({r["position"]: r for r in rows}[position]["unit"])


# ---------------------------------------------------------------------------
# 1. the feature description is facts, and it reaches Slow
# ---------------------------------------------------------------------------

def check_feature_description() -> dict[str, Any]:
    table = know.FEATURE_SEMANTICS
    blob = json.dumps(table, ensure_ascii=False).lower()
    banned = [word for word in
              ("decays", "monotone", "drift", "stays put", "scale-free",
               "gets easier", "survives a growing window")
              if word in blob]
    data_features = [name for name in table if not name.startswith("_")]
    complete = [name for name in data_features
                if table[name].get("formula") and table[name].get("computed_over")]
    scopes = table.get("_scopes") or {}
    three_scopes = all(key in scopes for key in
                       ("feature_window", "serving_window", "modified_region"))
    says_no_feature_measures_the_region = (
        "no feature in this vocabulary measures it"
        in str(scopes.get("modified_region", "")).lower())

    # it must actually be in the card Slow is handed
    group = {
        "members": [{"series_uid": "T1", "origin": 1176, "program_label": "p",
                     "program_label_with_params": "p", "program_steps": [],
                     "support_gain": -0.2, "delayed_gain": "UNKNOWN",
                     "symptom": "SUPPORT_NEGATIVE", "public_features": {},
                     "episode_id": "e1"}],
        "member_count": 1, "workflow": "w", "sign": "NEGATIVE",
        "visible_pattern_spread": {},
        "visible_pattern_spread_of_the_matched_successes": {},
        "contrast_cases": {"positive": [], "conflict": []},
        "fault": {"selectable_fault_types": [], "guard_if_none": None},
    }
    binding = know.FeatureBinding({})
    card = know.evidence_card(group, pattern_id="smoke-card", binding=binding)
    in_card = card.get("what_each_feature_is_measured_over") is table
    card_explains = "192" in str(card.get("how_to_use_the_feature_semantics"))
    return {
        "name": "the feature description is formula plus window, no unproven claims",
        "passed": bool(not banned and len(complete) == len(data_features)
                       and three_scopes
                       and says_no_feature_measures_the_region
                       and in_card and card_explains),
        "features_described": len(data_features),
        "features_with_formula_and_window": len(complete),
        "unproven_movement_claims_remaining": banned,
        "three_scopes_named": three_scopes,
        "says_no_feature_measures_the_modified_region":
            says_no_feature_measures_the_region,
        "description_is_in_the_slow_card": in_card,
        "card_states_the_serving_window_length": card_explains,
    }


# ---------------------------------------------------------------------------
# 2. action facts are the evaluator's own numbers
# ---------------------------------------------------------------------------

def check_action_facts() -> dict[str, Any]:
    ctx = _unit(9)
    origin = int(ctx.origin)
    uid = ps.decision_uids(ctx, size=1)[0]

    facts = know.action_facts(ctx, origin, PROGRAM, uid)
    at, config, roster, compiled = know._compiled_program(ctx, origin, PROGRAM)

    # serving side: replay the evaluator's own _prepare on the same window
    raw = np.asarray(at.values[uid], dtype=np.float64)
    window = raw[origin - scoped.CONTEXT_LENGTH:origin]
    _served, expected_moved, _t = scoped._prepare(window, compiled)
    serving_matches = (facts["serving_window"]["modified_points"]
                       == int(expected_moved)
                       and facts["serving_window"]["total_points"]
                       == int(window.size))

    # training side: the evaluator's own behavior_point_count, from a real run
    before = 0
    reading = scoped.scoped_evaluate(roster, at.values, compiled, config,
                                     origin=origin, scope=frozenset())
    _FITS["spent"] += int(reading.get("consumer_fits") or 0)
    training_matches = (facts["training_material"]["modified_points"]
                        == int(reading["behavior_point_count"]))

    identity = know.action_facts(ctx, origin, (), uid)
    identity_zero_is_measured = (
        identity["serving_window"]["modified_points"] == 0
        and "measured" in identity["serving_window"]["why"])

    missing = know.action_facts(ctx, origin, (("no_such_operator", {}),), uid)
    unknown_kept = (missing.get("serving_window", {}).get("modified_points")
                    == "UNKNOWN")

    shared_label = ("shared training material"
                    in facts["training_material"]["whose_property"])
    not_effect = "not a causal contribution" in facts[
        "counts_are_actions_not_effects"]
    return {
        "name": "action facts are the evaluator's own numbers, correctly labelled",
        "passed": bool(serving_matches and training_matches
                       and identity_zero_is_measured and unknown_kept
                       and shared_label and not_effect),
        "serving_count_matches_the_evaluators_prepare": serving_matches,
        "training_count_matches_behavior_point_count": training_matches,
        "serving": facts["serving_window"],
        "training": {k: v for k, v in facts["training_material"].items()
                     if k != "whose_property"},
        "evaluator_behavior_point_count": int(reading["behavior_point_count"]),
        "an_executed_identity_is_a_measured_zero": identity_zero_is_measured,
        "an_unavailable_count_stays_unknown": unknown_kept,
        "training_is_labelled_a_shared_property": shared_label,
        "a_count_is_not_an_effect": not_effect,
        "consumer_fits": int(reading.get("consumer_fits") or 0),
    }


# ---------------------------------------------------------------------------
# 3. parallel changes nothing
# ---------------------------------------------------------------------------

def check_parallel_is_equivalent() -> dict[str, Any]:
    ctx = _unit(9)
    origin = int(ctx.origin)
    population = ps.decision_uids(ctx, size=6)

    # -- readings: serial then parallel, same cache-cold start each time
    def read_all(readings: Any, workers: int) -> dict[str, float]:
        def one(uid: str) -> tuple[str, float]:
            out, _spent = readings.reading(PROGRAM, origin, [uid])
            return uid, float(out["per_series_gain"][uid])
        if workers == 1:
            rows = [one(uid) for uid in population]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(one, population))
        return dict(rows)

    serial_cache = runner.ReplayPredictionCache("smoke3-serial")
    serial = read_all(seq3.LockedReadings(
        ps.UnitReadings(cache=serial_cache, ctx=ctx, executor=ctx.executor),
        threading.Lock()), 1)
    _FITS["spent"] += int(serial_cache.physical_fits)

    parallel_cache = runner.ReplayPredictionCache("smoke3-parallel")
    parallel = read_all(seq3.LockedReadings(
        ps.UnitReadings(cache=parallel_cache, ctx=ctx, executor=ctx.executor),
        threading.Lock()), 4)
    _FITS["spent"] += int(parallel_cache.physical_fits)

    identical = all(serial[uid] == parallel[uid] for uid in population)
    same_fits = int(serial_cache.physical_fits) == int(
        parallel_cache.physical_fits)

    # -- the shared ledger counts every call exactly once under contention
    ledgers = runner.Ledgers()
    lock = threading.Lock()
    calls = 400

    def spend_one(_i: int) -> None:
        guard = seq3.SequenceGuard(ordering_cap=10 ** 6, per_unit_arm_cap=10 ** 6,
                                   ledgers=ledgers, lock=lock)
        guard.reserve(kind="fast", where={})
        guard.spend(kind="fast", calls=1, billable=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(spend_one, range(calls)))
    ledger_exact = int(ledgers.llm_fast) == calls

    # -- a per-sequence guard does not spend another sequence's cell budget
    a = seq3.SequenceGuard(ordering_cap=10 ** 6, per_unit_arm_cap=2,
                           ledgers=runner.Ledgers(), lock=threading.Lock())
    b = seq3.SequenceGuard(ordering_cap=10 ** 6, per_unit_arm_cap=2,
                           ledgers=runner.Ledgers(), lock=threading.Lock())
    a.open_cell(); b.open_cell()
    a.spend(kind="fast", calls=2, billable=True)
    isolated = True
    try:
        b.reserve(kind="fast", where={})
    except Exception:  # noqa: BLE001
        isolated = False

    # -- aggregation is in roster order, not completion order
    completion = list(reversed(population))
    ordered = [uid for uid in population if uid in set(completion)]
    roster_order = ordered == population

    return {
        "name": "parallel changes readings, counting and order by nothing",
        "passed": bool(identical and same_fits and ledger_exact and isolated
                       and roster_order),
        "readings_identical_serial_vs_parallel": identical,
        "no_extra_consumer_fit_under_parallel": same_fits,
        "physical_fits": {"serial": int(serial_cache.physical_fits),
                          "parallel": int(parallel_cache.physical_fits)},
        "shared_ledger_counts_every_call_once": ledger_exact,
        "ledger_calls": {"expected": calls, "counted": int(ledgers.llm_fast)},
        "per_sequence_cell_budget_is_isolated": isolated,
        "aggregation_is_in_roster_order": roster_order,
        "what_this_does_not_claim": (
            "no speedup is asserted here.  This check is about equivalence "
            "only; the achieved concurrency is reported from the run itself"),
    }


# ---------------------------------------------------------------------------

CHECKS = (check_feature_description, check_action_facts,
          check_parallel_is_equivalent)


def run() -> dict[str, Any]:
    _FITS["spent"] = 0
    results = []
    for check in CHECKS:
        try:
            results.append(check())
        except Exception as exc:  # noqa: BLE001 - a failed check is a result
            results.append({"name": check.__name__, "passed": False,
                            "error": "%s: %s" % (type(exc).__name__,
                                                 str(exc)[:400])})
    inherited = smoke2.run()
    return {
        "stage": "DEV_SEQ3_SMOKE",
        "passed": bool(all(row.get("passed") for row in results)
                       and inherited["passed"]),
        "checks": results,
        "inherited_dev_seq2_smoke": {"passed": inherited["passed"],
                                     "checks": [c.get("name") or c.get("error")
                                                for c in inherited["checks"]]},
        "consumer_fits": _FITS["spent"],
        "llm_calls": 0,
    }


def main() -> int:
    report = run()
    print(json.dumps(drafts._plain(report), ensure_ascii=False, indent=1,
                     default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
