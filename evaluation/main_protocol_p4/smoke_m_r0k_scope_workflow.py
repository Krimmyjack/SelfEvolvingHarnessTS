"""Zero-fit preflight for M-R0k: population, denominator, raw fallback, ceiling.

Six things have to be true before a single Consumer fit is spent, and every one
of them can be checked without spending one:

1. **Population.**  The forward / A5-online ordering holds 26 cells; five carry
   ``TARGET_HELD_IN`` and are excluded *by exposure name*, leaving the 21
   development units M-R0j already read.  The selection boundary is k1 / p04, so
   five units sit before it and sixteen after.
2. **Denominator.**  Every reading is scored over the unit's whole served
   population, which is what makes a narrowing comparable to its ancestor.  The
   served count is read off the roster, not assumed.
3. **Raw fallback.**  A series the Scope does not select must carry the raw
   prediction and score a gain of exactly ``0.0`` -- not "approximately".  This
   is checked by seeding a cache entry with known numbers and re-masking it, so
   the rule is exercised rather than asserted.
4. **Cost ceiling.**  The package is authorised 400 physical Consumer fits, and
   the guard has to refuse *before* spending.  Checked by driving the guard at
   the boundary.
5. **Legality.**  The ancestor program and the three fixed Workflow candidates
   are checked against the operator registry and the typed DSL: present,
   ``forecast`` in ``allowed_tasks``, parameters accepted by the public schema,
   composition length within the frozen program space.  An illegal candidate is
   recorded with its reason; no rule is relaxed to admit one.
6. **Parent rights.**  Whether the parent version was deployable at the boundary
   is read off the rights snapshot, not off any later gate result.

Nothing here fits a Consumer, calls a model, or writes an artifact.

Run:  python -m evaluation.main_protocol_p4.smoke_m_r0k_scope_workflow
"""
from __future__ import annotations

import json
import sys
from typing import Any, Mapping, Sequence

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import audit_m_r0b_revision_opportunity as opp
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live

#: The authorised physical-fit ceiling for this package.  Every fit counts:
#: smoke, retries and the ones spent inside a call that then raises.  Unused
#: allowance from earlier packages is not rolled in.
MAX_FITS = 400

FLOOR = int(contract.RISK["min_treated"])
DELAYED = int(runner.DELAYED_OFFSET)
BOUNDARY = int(live.BOUNDARY_POSITION)

DEVELOPMENT_EXPOSURES = ("SPENT_DEV", "UNREAD")
EXCLUDED_EXPOSURES = ("TARGET_HELD_IN",)

ANCESTOR_SCOPE = {"scope_type": "serving_series_predicate",
                  "predicate": [{"feature": "local_robust_z_peak", "op": ">=",
                                 "threshold": 3.0}]}

#: The ancestor Program, and the three Workflow candidates the package fixes in
#: advance.  ``hampel_filter`` carries no parameters because the registry
#: declares no defaults for it: an empty mapping is what makes the operator use
#: its own declared signature defaults, which is what "registry default" means
#: here and is recorded as such rather than restated as numbers.
ANCESTOR_PROGRAM = (("outlier_mad", {}),)
PMC_PARAMS = {"period": 24, "cycles": 3, "min_donors": 2}
WORKFLOW_CANDIDATES = {
    "W1_hampel_filter": (("hampel_filter", {}),),
    "W2_pmc_then_outlier_mad": (("period_median_complete", dict(PMC_PARAMS)),
                                ("outlier_mad", {})),
    "W3_outlier_mad_then_pmc": (("outlier_mad", {}),
                                ("period_median_complete", dict(PMC_PARAMS))),
}


class FitCeiling(RuntimeError):
    """The authorised ceiling would be crossed.  Nothing is spent."""


def guarded_read(cache: runner.ReplayPredictionCache, ctx: Any, origin: int,
                 steps: Sequence[tuple] | None, resolved) -> dict[str, Any]:
    """``cache.reading`` with the ceiling enforced before the spend.

    A new ``(unit, face, program)`` entry costs at most two fits, so the guard
    refuses when two more would cross the line.  A cached entry costs nothing
    and is always allowed through.
    """
    key = cache.key(ctx.unit, origin, runner.forecast_p4._config(origin),
                    steps or ())
    if key not in cache._entries and cache.physical_fits + 2 > MAX_FITS:
        raise FitCeiling(
            "a new (unit, face, program) entry would take the package past %d "
            "physical fits (spent %d)" % (MAX_FITS, cache.physical_fits))
    return cache.reading(ctx, origin, steps, resolved)


class _FakeCtx:
    """Only what ``ReplayPredictionCache.reading`` reads off a context."""

    def __init__(self, unit: Mapping[str, Any]) -> None:
        self.unit = dict(unit)


def check_population() -> dict[str, Any]:
    doc = base.load(live.ORDERING)
    cells = base.cells(doc, live.ARM)
    kept, excluded = [], []
    for cell in cells:
        exposure = list(cell["unit"].get("exposure") or ())
        row = {"position": int(cell["position"]),
               "origin": int(cell["unit"]["origin"]),
               "block": str(cell["unit"]["block"]),
               "exposure": exposure}
        if any(name in EXCLUDED_EXPOSURES for name in exposure):
            excluded.append({**row, "why": "held for the Target, not development"})
        elif any(name in DEVELOPMENT_EXPOSURES for name in exposure):
            kept.append(row)
        else:
            excluded.append({**row, "why": "exposure is not a development one"})
    before = [r for r in kept if r["position"] <= BOUNDARY]
    after = [r for r in kept if r["position"] > BOUNDARY]
    served = {}
    for row in kept:
        cell = next(c for c in cells if int(c["position"]) == row["position"])
        ctx = opp._unit_ctx(dict(cell["unit"]))
        served[str(row["position"])] = len(ctx.eval_uids)
    ok = (len(cells) == 26 and len(kept) == 21 and len(excluded) == 5
          and len(before) == 5 and len(after) == 16
          and set(served.values()) == {20})
    return {
        "check": "population_and_denominator",
        "passed": bool(ok),
        "cells_in_the_ordering": len(cells),
        "development_units": len(kept),
        "excluded": excluded,
        "excluded_by": "exposure name, never by a property of the reading",
        "boundary_position": BOUNDARY,
        "units_before_the_boundary": [r["position"] for r in before],
        "units_after_the_boundary": [r["position"] for r in after],
        "served_series_per_unit": served,
        "denominator": ("every served series on every unit; an out-of-scope "
                        "series carries the raw prediction and scores 0"),
        "faces": ["the unit's own origin (Support, evidence)",
                  "origin + %d (delayed, the authoritative gate)" % DELAYED],
        "not_read": ["the +144 evaluation face", "TARGET_HELD_IN units",
                     "sealed data"],
    }


def check_raw_fallback() -> dict[str, Any]:
    """Seed one entry, re-mask it, and read what the fallback actually does."""
    cache = runner.ReplayPredictionCache("smoke")
    unit = {"block": "[0:40]", "span": [0, 40], "origin": 1176,
            "exposure": ["SPENT_DEV"]}
    ctx = _FakeCtx(unit)
    origin = 1176
    uids = ["A", "B", "C", "D", "E"]
    key = cache.key(unit, origin, runner.forecast_p4._config(origin),
                    ANCESTOR_PROGRAM)
    cache._entries[key] = {
        "verifier_passed": True,
        "eval_uids": list(uids),
        # raw - program is +0.5, -0.25, 0, +0.1, -1.0 in that order
        "raw_per_view": [1.0, 1.0, 1.0, 1.0, 1.0],
        "program_per_view": [0.5, 1.25, 1.0, 0.9, 2.0],
        "degenerate_uids": [],
        "consumer_fits": 0,
    }
    empty = cache.reading(ctx, origin, ANCESTOR_PROGRAM, frozenset())
    two = cache.reading(ctx, origin, ANCESTOR_PROGRAM, frozenset({"A", "E"}))
    exact_zero = [uid for uid in uids
                  if repr(float(two["per_series_gain"][uid])) == repr(0.0)]
    passed = (
        cache.physical_fits == 0
        and empty["identity"] is True
        and all(repr(float(v)) == repr(0.0)
                for v in empty["per_series_gain"].values())
        and empty["mean_smase"] == 1.0
        and sorted(exact_zero) == ["B", "C", "D"]
        and repr(float(two["per_series_gain"]["A"])) == repr(0.5)
        and repr(float(two["per_series_gain"]["E"])) == repr(-1.0)
        and two["treated"] == 2 and two["served"] == 5
    )
    return {
        "check": "raw_fallback_is_exactly_zero",
        "passed": bool(passed),
        "physical_fits_spent": cache.physical_fits,
        "empty_scope_reproduces_static": {
            "identity": empty["identity"],
            "every_gain_is_exactly_zero": all(
                repr(float(v)) == repr(0.0)
                for v in empty["per_series_gain"].values()),
            "mean_smase_equals_raw": empty["mean_smase"] == empty[
                "static_mean_smase"],
        },
        "partial_scope": {
            "treated": two["treated"], "served": two["served"],
            "per_series_gain": two["per_series_gain"],
            "series_scoring_exactly_zero": sorted(exact_zero),
            "aggregate_gain": two["aggregate_gain"],
        },
        "rule": ("out of scope -> raw prediction -> gain exactly 0.0, so a "
                 "narrowing is scored on its ancestor's denominator"),
    }


def check_cost_ceiling() -> dict[str, Any]:
    """The guard has to refuse before the spend, not report after it."""
    cache = runner.ReplayPredictionCache("smoke")
    unit = {"block": "[0:40]", "span": [0, 40], "origin": 1176,
            "exposure": ["SPENT_DEV"]}
    ctx = _FakeCtx(unit)
    origin = 1176
    key = cache.key(unit, origin, runner.forecast_p4._config(origin),
                    ANCESTOR_PROGRAM)
    cache._entries[key] = {"verifier_passed": True, "eval_uids": ["A"],
                           "raw_per_view": [1.0], "program_per_view": [0.5],
                           "degenerate_uids": [], "consumer_fits": 0}
    cached_still_free = True
    cache.physical_fits = MAX_FITS - 1
    try:
        guarded_read(cache, ctx, origin, ANCESTOR_PROGRAM, frozenset({"A"}))
    except FitCeiling:
        cached_still_free = False
    refused, why = False, None
    try:
        guarded_read(cache, ctx, 9999, ANCESTOR_PROGRAM, frozenset({"A"}))
    except FitCeiling as exc:
        refused, why = True, str(exc)
    spent_after_the_refusal = cache.physical_fits
    return {
        "check": "cost_ceiling_refuses_before_the_spend",
        "passed": bool(refused and cached_still_free
                       and spent_after_the_refusal == MAX_FITS - 1),
        "ceiling": MAX_FITS,
        "a_new_entry_at_the_boundary_is_refused": refused,
        "refusal": why,
        "a_cached_entry_stays_free": cached_still_free,
        "fits_spent_by_the_refusal": 0,
        "counting_rule": ("every physical fit counts, including smoke, retries "
                          "and fits spent inside a call that then raises; no "
                          "allowance is carried over from an earlier package"),
    }


def _legality(label: str, steps: Sequence[tuple]) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.contracts.program import Program  # noqa: PLC0415
    from SelfEvolvingHarnessTS.operators import registry  # noqa: PLC0415

    row: dict[str, Any] = {
        "label": label,
        "steps": [{"op": op, "params": dict(params)} for op, params in steps],
        "length": len(steps),
        "operators": [],
    }
    reasons: list[str] = []
    for op, params in steps:
        meta = registry.OPERATOR_METADATA.get(op)
        if meta is None:
            reasons.append("%s: not in the operator registry" % op)
            row["operators"].append({"op": op, "in_registry": False})
            continue
        allowed = tuple(meta.get("allowed_tasks") or ())
        if "forecast" not in allowed:
            reasons.append("%s: allowed_tasks=%s excludes forecast"
                           % (op, list(allowed)))
        schema = meta.get("public_parameter_schema")
        row["operators"].append({
            "op": op, "in_registry": True, "allowed_tasks": list(allowed),
            "public_parameter_schema": schema,
            "declared_defaults_used": (not dict(params)),
        })
        if isinstance(schema, Mapping):
            properties = dict(schema.get("properties") or {})
            for name in dict(params):
                if (schema.get("additionalProperties") is False
                        and name not in properties):
                    reasons.append("%s: %r is not a public parameter"
                                   % (op, name))
            for name in tuple(schema.get("required") or ()):
                if name not in dict(params):
                    reasons.append("%s: required parameter %r missing"
                                   % (op, name))
            for name, value in dict(params).items():
                rule = dict(properties.get(name) or {})
                kind = rule.get("type")
                if kind == "integer" and (isinstance(value, bool)
                                          or not isinstance(value, int)):
                    reasons.append("%s: %r must be an integer" % (op, name))
                if kind == "number" and not isinstance(value, (int, float)):
                    reasons.append("%s: %r must be a number" % (op, name))
                if "minimum" in rule and isinstance(value, (int, float)) \
                        and float(value) < float(rule["minimum"]):
                    reasons.append("%s: %r below the declared minimum"
                                   % (op, name))
    if len(steps) > 2:
        reasons.append("composition longer than the frozen program space "
                       "allows (length <= 2)")
    try:
        program = Program.from_steps(
            [(op, dict(params)) for op, params in steps], source="m_r0k_smoke")
        row["typed_program"] = [
            {"op": step.op, "params": dict(step.params)}
            for step in program.steps]
    except Exception as exc:  # noqa: BLE001 - recorded, never relaxed
        reasons.append("typed DSL refused the program: %s" % str(exc)[:200])
    row["legal"] = not reasons
    row["reasons"] = reasons
    return row


def check_program_legality() -> dict[str, Any]:
    rows = [_legality("ANCESTOR_outlier_mad", ANCESTOR_PROGRAM)]
    rows.extend(_legality(label, steps)
                for label, steps in WORKFLOW_CANDIDATES.items())
    return {
        "check": "workflow_candidate_legality",
        "passed": all(row["legal"] for row in rows),
        "checked_before_any_scoring": True,
        "programs": rows,
        "program_space": {
            "operators": contract.PROGRAM_SPACE["operators"],
            "compositions": contract.PROGRAM_SPACE["compositions"],
            "operators_added": 0,
        },
        "if_illegal": ("the reason is recorded and the candidate is dropped; "
                       "no rule is relaxed and no substitute is chosen after "
                       "a score is seen"),
    }


def check_parent_rights() -> dict[str, Any]:
    """Was the parent version deployable at the boundary?  Rights, not results."""
    doc = base.load(live.ORDERING)
    k0 = json.loads(base.K0_RECEIPT.read_text(encoding="utf-8"))
    lineage = list(k0.get("lineage_keys") or ())
    active_ids = list(k0.get("active_skill_ids") or ())
    snapshots = []
    for cell in base.cells(doc, live.ARM):
        snapshots.append({
            "position": int(cell["position"]),
            "snapshot_skill_ids_at_start": list(
                cell.get("snapshot_skill_ids_at_start") or ()),
            "deployed": (base.prog(cell["deployed"]) if cell.get("deployed")
                         else None),
            "deployed_serving_scope": cell.get("deployed_serving_scope"),
            "deployed_via": cell.get("deployed_via"),
        })
    at_or_before = [row for row in snapshots if row["position"] <= BOUNDARY]
    held_all_the_way = bool(active_ids) and all(
        set(active_ids) <= set(row["snapshot_skill_ids_at_start"])
        for row in snapshots)
    as_run = [row for row in snapshots if row["deployed"]]
    return {
        "check": "parent_version_rights_at_the_boundary",
        "passed": bool(active_ids and held_all_the_way),
        "k0_active_skill_ids": active_ids,
        "k0_lineage_keys": lineage,
        "parent_is_deployable_at_the_boundary": bool(
            active_ids and all(set(active_ids)
                               <= set(row["snapshot_skill_ids_at_start"])
                               for row in at_or_before)),
        "active_card_never_left_the_snapshot": held_all_the_way,
        "ruling": ("the parent's availability is read off the rights snapshot "
                   "at the boundary.  The Draft under revision carries "
                   "deployable=false; that is the Draft's state, not a loss of "
                   "rights by the K0 ancestor, and it is not read as one"),
        "as_run_deployments": {
            "units_where_the_arm_actually_deployed_the_parent": [
                row["position"] for row in as_run],
            "count": len(as_run),
            "scopes_seen": sorted({json.dumps(row["deployed_serving_scope"],
                                              sort_keys=True)
                                   for row in as_run}),
            "note": ("recorded because it is what the course did.  It is not "
                     "the comparison target: the aligned rule compares against "
                     "the policy the rights snapshot made executable, per the "
                     "plan's clause (c)"),
        },
        "out_of_domain_fallback": "raw (the second fitted pipeline), unchanged",
    }


def check_fixed_scopes() -> dict[str, Any]:
    path = base.ART / "m_r0i_eligible_deterministic_control__run1.json"
    if not path.exists():
        return {"check": "fixed_scopes_cross_check", "passed": False,
                "why": "no M-R0i receipt on disk"}
    written = json.loads(path.read_text(encoding="utf-8")).get(
        "state_written") or {}
    matches = written.get("scope_before") == ANCESTOR_SCOPE
    return {
        "check": "fixed_scopes_cross_check",
        "passed": bool(matches),
        "ancestor": ANCESTOR_SCOPE,
        "matches_the_m_r0i_receipt": matches,
        "m_r0i_outcome": written.get("candidate_outcome"),
        "ancestor_program": [{"op": op, "params": dict(params)}
                             for op, params in ANCESTOR_PROGRAM],
    }


def run() -> dict[str, Any]:
    checks = [check_population(), check_raw_fallback(), check_cost_ceiling(),
              check_program_legality(), check_parent_rights(),
              check_fixed_scopes()]
    return {
        "smoke": "M_R0K_PREFLIGHT",
        "llm_calls": 0,
        "consumer_fits": 0,
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
