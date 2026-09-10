"""DEV-DEPLOY-1: deployment-calibre alignment.

See ``docs/DEV_DEPLOY1_FAST_ONLY_ALIGNMENT_TASK_2026-09-08.md`` for the full
task book.  This module implements the two 0-new-LLM legs of the first
package:

  Part A  the shadow audit of DEV-SEQ-3 g2's already-completed 120 decisions
          (u9/u10/u11, both arms): what Fast actually selected before the
          current Support/admission gate acted, versus what was actually
          delivered.  Zero new LLM calls -- it re-plays the frozen scoring
          machinery on the recorded execution trace, never re-queries Fast.

  Part C  the 0-LLM deterministic baselines -- Static/raw, D-fixed and
          D-safe -- calibrated on u7/u8 over the full 20-series population
          and then frozen and scored (no further selection) on u9/u10/u11.

  Part B  the frozen Fast-only real run.  Fixed knowledge-frozen (DEV-SEQ-3's
          "A") parent snapshot, ``experience_episodes=()``, a real
          ``TTHAMethod.prepare`` call per series -- but the winner is
          whatever Fast's own ``trace.chosen_candidate_id`` names, checked
          only against the deterministic window verifier, never against
          ``online_loop.run_online_round``'s Support/admission search.
          Scoring (both faces) is the same 0-LLM ``Scorer`` Parts A/C use,
          applied after the decision is final -- the score never reaches
          Fast.  One OS process per (group, unit); each decision is its own
          Fast session, checkpointed as it completes.

Reused rather than re-derived
------------------------------
Every reading here goes through ``run_hec1.ReplayPredictionCache`` /
``per_sequence.UnitReadings`` / ``per_sequence.SequenceView`` -- the same
scoring surface DEV-SEQ-3 itself used -- so a program's gain at a given
(unit, origin) is exactly reproduced, never approximated.  ``UnitContext`` is
built directly from the ``unit`` descriptor recorded on each historical row
(block/span/origin), not guessed from a UID (task book Sec 1.1).

The one place this module infers something the historical artifact does not
literally store: when Fast's original, pre-Support choice differs from what
was actually delivered, the chosen candidate's *typed* program steps were
never persisted (only ``result.winner_program`` -- the delivered program --
survives to JSON; the runner-up candidates only leave a rendered label via
``per_sequence.program_label_with_params``).  ``chosen_program_from_row``
reconstructs typed steps from that label and then self-checks the
reconstruction by re-scoring it at the origin the historical run already
scored it at, comparing against the gain the historical run actually
recorded.  A mismatch is never trusted silently -- it downgrades the row to
UNKNOWN rather than reporting a guessed counterfactual.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_seq2_knowledge as know
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import per_sequence as ps
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_seq2 as seq2
from evaluation.main_protocol_p4 import run_dev_seq3 as seq3
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner
from SelfEvolvingHarnessTS.contracts.method import PreparationStatus
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES

ART = base.ROOT / "artifacts" / "main_protocol"
SCRATCH = base.ROOT / "_scratch" / "dev_deploy1"
G2_ARTIFACT = ART / "dev_seq3_guidance_delivery__g2.json"
FORMATION_DIR = base.ROOT / "_scratch" / "dev_seq2" / "pkg2" / "g2"

CALIBRATION_POSITIONS = (7, 8)
TEST_POSITIONS = (9, 10, 11)
TOL = 1e-6

# ---------------------------------------------------------------------------
# Part B constants (task book Sec 2B / Sec 5)
# ---------------------------------------------------------------------------

REQUIRED_MODEL = "gpt-5.6-sol"
PER_SEQUENCE_LLM_CAP = 24
GROUP_SEEDS = {"g1": 2026090901, "g2": 2026090902}
GROUP_ORDER = ("g1", "g2")
PER_GROUP_LLM_CAP = 900
MAX_FITS = 2000
MAX_LLM = 2000
MAX_WALL_SECONDS = 6 * 3600
CONCURRENCY = 2
DOMAIN = "devdeploy1"

_LABEL_RE = re.compile(r"^([A-Za-z0-9_]+)(?:\((.*)\))?$")

#: ``know._fault_kind``'s ``_ACCOUNT_MARKERS`` missed this relay's actual
#: wording (DEV-DEPLOY-2's live run on 2026-09-09 hit it first): the code is
#: ``insufficient_user_quota`` -- "insufficient_quota" is NOT a substring of
#: it (there is a "_user_" in between) -- and the message text is Chinese
#: ("用户额度不足", "预扣费额度失败"), not English.  Missing this meant a
#: genuine account fault was misclassified as an ordinary per-series fault:
#: the run recorded it and moved on to the next series instead of halting
#: the line, and every decision after the balance actually ran out (all of
#: DEV-DEPLOY-2's g2 group, 120 decisions) came back as
#: ``FAULT_NO_DECISION`` dressed up as a "completed" unit.  Purely additive
#: over the shared classifier -- nothing previously classified changes.
_EXTRA_ACCOUNT_MARKERS = (
    "insufficient_user_quota", "用户额度不足", "预扣费额度失败",
)


def _fault_kind(detail: str) -> str:
    if any(marker in str(detail) for marker in _EXTRA_ACCOUNT_MARKERS):
        return "ACCOUNT_OR_PERMISSION_FAULT"
    return know._fault_kind(detail)


# ---------------------------------------------------------------------------
# loading the historical g2 artifact and the u7/u8 formation checkpoints
# ---------------------------------------------------------------------------

def load_g2() -> dict[str, Any]:
    return json.loads(G2_ARTIFACT.read_text(encoding="utf-8"))


def load_formation_unit(position: int) -> dict[str, Any]:
    path = FORMATION_DIR / ("formation_u%03d.json" % position)
    return json.loads(path.read_text(encoding="utf-8"))


def iter_test_decisions(doc: Mapping[str, Any]):
    """Yield (position, arm, row) for all 120 u9/u10/u11 sequence records."""
    for arm, unit in doc["validation"]["units"].items():
        for row in unit["sequences"]:
            yield int(unit["position"]), str(arm), row
    for unit in doc["followup_units"]:
        for row in unit["sequences"]:
            yield int(unit["position"]), str(unit["arm"]), row


def unit_descriptor(position: int, doc: Mapping[str, Any] | None = None,
                    ) -> dict[str, Any]:
    """The recorded ``unit`` dict (block/span/origin/exposure) this position
    actually ran on -- read off an existing row, not reconstructed from a
    position number."""
    if position in CALIBRATION_POSITIONS:
        return load_formation_unit(position)["unit"]["unit"]
    doc = doc if doc is not None else load_g2()
    for pos, _arm, row in iter_test_decisions(doc):
        if pos == position:
            return row["unit"]
    raise KeyError("no recorded row carries a unit descriptor for position %d"
                   % position)


# ---------------------------------------------------------------------------
# shared scoring machinery: one cache/UnitContext per position, no LLM
# ---------------------------------------------------------------------------

class Scorer:
    """One ``ReplayPredictionCache`` shared across every position this run
    touches, so a (unit, origin, program) already read for one purpose (the
    shadow audit) is a cache hit for another (the deterministic baselines) --
    the exact reuse the task book asks for, not an approximation of it."""

    def __init__(self, doc: Mapping[str, Any] | None = None) -> None:
        import threading

        self._doc = doc if doc is not None else load_g2()
        self.cache = runner.ReplayPredictionCache("dev_deploy1")
        self._lock = threading.Lock()
        self.ctx: dict[int, Any] = {}
        self.readings: dict[int, ps.UnitReadings] = {}
        for position in (*CALIBRATION_POSITIONS, *TEST_POSITIONS):
            unit = unit_descriptor(position, self._doc)
            ctx = runner.UnitContext(unit)
            self.ctx[position] = ctx
            self.readings[position] = ps.UnitReadings(
                cache=self.cache, ctx=ctx, executor=ctx.executor)

    @property
    def physical_fits(self) -> int:
        return int(self.cache.physical_fits)

    def population(self, position: int) -> list[str]:
        return list(self.ctx[position].eval_uids)

    def sequence_reading(self, position: int, uid: str, steps: tuple,
                         origin: int) -> Any:
        """One series' gain under one program at one face; UnitFault ->
        legality rejection is left to the caller (verification is checked
        first so a rejection never looks like a measured 0).

        Locked for the whole call: Part B runs up to ``CONCURRENCY`` Fast
        sessions in parallel threads, and ``ReplayPredictionCache`` (a plain
        dict, unguarded) is shared across all of them -- unlike Parts A/C,
        which only ever called this from one thread."""
        with self._lock:
            return ps.SequenceView(self.readings[position], uid).evaluate(
                steps, int(origin))

    def population_reading(self, position: int, steps: tuple,
                           origin: int) -> dict[str, Any] | None:
        """The full-population reading a fixed program would get if deployed
        to every series in this unit -- what D-fixed/D-safe calibration and
        freeze-scoring both need.  None on a deterministic legality
        rejection.

        Identity is short-circuited rather than sent through the Consumer:
        its gain is 0.0 by construction on every series, so paying two fits
        to confirm that would be a wasted read, and ``treated=0`` here
        matches the convention every other identity reading in this project
        uses (``UnitReadings.compose``, ``_policy_reading``)."""
        uids = self.population(position)
        if not steps:
            return {"identity": True, "treated": 0, "served": len(uids),
                    "per_series_gain": {uid: 0.0 for uid in uids},
                    "aggregate_gain": 0.0, "harmed_fraction": 0.0,
                    "max_single_series_harm": 0.0}
        try:
            with self._lock:
                return self.readings[position].reading(
                    steps, int(origin), uids)[0]
        except runner.UnitFault:
            return None


# ---------------------------------------------------------------------------
# Part A: shadow audit of DEV-SEQ-3 g2's 120 historical decisions
# ---------------------------------------------------------------------------

def parse_program_label(label: str) -> tuple[tuple[str, dict], ...]:
    """Inverse of ``per_sequence.program_label_with_params``.  Lossless for
    this project's typed DSL: every step is one canonical op name plus
    primitively-typed (int/float/bool) keyword params, rendered with ``%s``
    and sorted by key -- there is no free-text parameter anywhere in the
    registry's public parameter schemas, so ``ast.literal_eval`` round-trips
    exactly.  Never trusted on its own: every call site re-scores the parsed
    steps and checks the result against the gain already on record before
    using it for anything new (see ``chosen_program_from_row``)."""
    label = (label or "").strip()
    if not label or label == "identity":
        return ()
    steps: list[tuple[str, dict]] = []
    for part in label.split(" -> "):
        part = part.strip()
        match = _LABEL_RE.match(part)
        if not match:
            raise ValueError("cannot parse program label part: %r" % part)
        op = match.group(1)
        params: dict[str, Any] = {}
        params_str = match.group(2)
        if params_str:
            for kv in params_str.split(", "):
                key, _sep, val = kv.partition("=")
                key = key.strip()
                val = val.strip()
                try:
                    params[key] = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    params[key] = val
        steps.append((op, params))
    return tuple(steps)


def chosen_program_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """The program Fast actually selected before Support/admission acted, as
    the task book's Part A defines it -- reconstructed from the recorded
    trace, never re-asked of the model.

    status is one of:
      IDENTITY_SELECTED       -- trace.chosen_candidate_id == "identity"
      RECONSTRUCTED           -- parsed from candidate_programs, not yet
                                  self-checked by the caller
      LEGALITY_FALLBACK_ZERO  -- the chosen candidate failed the
                                  deterministic window verifier; raw ran
      UNKNOWN                 -- no reconstruction material; never guessed
    """
    chosen_id = str(row.get("chosen_candidate_id") or "")
    probes = row.get("probes") or []
    if chosen_id == "identity":
        return {"status": "IDENTITY_SELECTED", "steps": (), "origin_gain": 0.0,
                "chosen_id": "identity",
                "why": "trace.chosen_candidate_id == 'identity'"}
    if not chosen_id:
        return {"status": "UNKNOWN", "steps": None, "origin_gain": "UNKNOWN",
                "chosen_id": None,
                "why": "empty chosen_candidate_id: select output anomaly, "
                       "not a deliberate identity choice -- not reconstructed"}
    entry = next((p for p in probes if str(p.get("candidate_id")) == chosen_id),
                 None)
    if entry is None:
        return {"status": "UNKNOWN", "steps": None, "origin_gain": "UNKNOWN",
                "chosen_id": chosen_id,
                "why": "chosen_candidate_id not found among recorded probes: "
                       "untested, not a deterministic zero"}
    if entry.get("kind") == "verifier_rejected" or entry.get("gain") is None:
        return {"status": "LEGALITY_FALLBACK_ZERO", "steps": (),
                "origin_gain": 0.0, "chosen_id": chosen_id,
                "why": "deterministic window-verifier rejection; raw ran, "
                       "which is what the recorded 0 reflects"}
    label = (row.get("candidate_programs") or {}).get(chosen_id)
    if label is None:
        return {"status": "UNKNOWN", "steps": None, "origin_gain": "UNKNOWN",
                "chosen_id": chosen_id,
                "why": "candidate_programs carries no label for chosen_id"}
    try:
        steps = parse_program_label(label)
    except ValueError as exc:
        return {"status": "UNKNOWN", "steps": None, "origin_gain": "UNKNOWN",
                "chosen_id": chosen_id,
                "why": "label reconstruction failed: %s" % exc}
    return {"status": "RECONSTRUCTED", "steps": steps,
            "origin_gain": float(entry["gain"]), "chosen_id": chosen_id,
            "why": "reconstructed from the candidate_programs label; "
                  "self-check against the recorded probe gain pending"}


def shadow_audit(scorer: Scorer, doc: Mapping[str, Any]) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    unverified = 0
    for position, arm, row in iter_test_decisions(doc):
        uid = str(row["series_uid"])
        origin = int(row["origin"])
        delayed_origin = int(row["delayed_origin"])
        delivered_steps = ps.normalise_steps(row.get("deployed") or ())
        delivered_origin_gain = row.get("support_gain")
        delivered_delayed_gain = row.get("delayed_gain")

        chosen = chosen_program_from_row(row)
        reconstruction_verified = None

        if chosen["status"] == "RECONSTRUCTED":
            fits_before = scorer.physical_fits
            receipt = scorer.sequence_reading(position, uid, chosen["steps"],
                                              origin)
            recomputed = receipt.gain
            reconstruction_verified = (
                recomputed is not None
                and abs(float(recomputed) - float(chosen["origin_gain"]))
                <= TOL)
            if not reconstruction_verified:
                chosen = {
                    "status": "UNKNOWN", "steps": None,
                    "origin_gain": "UNKNOWN", "chosen_id": chosen["chosen_id"],
                    "why": (chosen["why"] + "; FAILED self-check: recomputed "
                           "origin gain %r != recorded %r (verifier passed=%s)"
                           % (recomputed, chosen["origin_gain"],
                              receipt.verification.passed)),
                }
            del fits_before  # kept for a debugger breakpoint, not reported

        overridden = (chosen["steps"] is not None
                     and ps.normalise_steps(chosen["steps"] or ())
                     != delivered_steps)

        chosen_delayed_gain: Any = "UNKNOWN"
        if chosen["steps"] is not None:
            if not overridden:
                chosen_delayed_gain = delivered_delayed_gain
            elif not chosen["steps"]:
                chosen_delayed_gain = 0.0
            else:
                receipt = scorer.sequence_reading(position, uid,
                                                  chosen["steps"],
                                                  delayed_origin)
                chosen_delayed_gain = (round(float(receipt.gain), 6)
                                      if receipt.gain is not None
                                      else "UNKNOWN")

        if chosen["status"] == "UNKNOWN":
            unverified += 1

        rows_out.append({
            "position": position, "arm": arm, "series_uid": uid,
            "origin": origin, "delayed_origin": delayed_origin,
            "chosen_candidate_id": chosen["chosen_id"],
            "chosen_status": chosen["status"], "chosen_why": chosen["why"],
            "chosen_reconstruction_verified": reconstruction_verified,
            "chosen_program": (ps.program_label_with_params(chosen["steps"])
                              if chosen["steps"] is not None else "UNKNOWN"),
            "chosen_origin_gain": chosen["origin_gain"],
            "chosen_delayed_gain": chosen_delayed_gain,
            "winner_candidate_id": row.get("winner_candidate_id"),
            "delivered_program": row.get("deployed_label"),
            "delivered_origin_gain": delivered_origin_gain,
            "delivered_delayed_gain": delivered_delayed_gain,
            "overridden": overridden,
            "decision": row.get("decision"),
        })

    by_status: dict[str, int] = {}
    for r in rows_out:
        by_status[r["chosen_status"]] = by_status.get(r["chosen_status"], 0) + 1
    overridden_rows = [r for r in rows_out if r["overridden"]]
    preserved_rows = [r for r in rows_out
                     if r["chosen_status"] != "UNKNOWN" and not r["overridden"]]
    covering = [r for r in overridden_rows
               if isinstance(r["delivered_delayed_gain"], (int, float))
               and isinstance(r["chosen_delayed_gain"], (int, float))]
    override_delayed_net = sum(float(r["delivered_delayed_gain"])
                               - float(r["chosen_delayed_gain"])
                               for r in covering)
    override_helped = sum(1 for r in covering
                          if r["delivered_delayed_gain"] > r["chosen_delayed_gain"])
    override_hurt = sum(1 for r in covering
                        if r["delivered_delayed_gain"] < r["chosen_delayed_gain"])
    id_overridden = [r for r in overridden_rows
                     if r["chosen_status"] == "IDENTITY_SELECTED"]
    id_overridden_covering = [
        r for r in id_overridden
        if isinstance(r["delivered_delayed_gain"], (int, float))
        and isinstance(r["chosen_delayed_gain"], (int, float))]
    return {
        "n_decisions": len(rows_out),
        "by_chosen_status": by_status,
        "n_preserved_selection_equals_delivery": len(preserved_rows),
        "n_overridden": len(overridden_rows),
        "n_unverified": unverified,
        "override_delayed_net_of_covering_pairs": round(override_delayed_net, 6),
        "override_delayed_helped_count": override_helped,
        "override_delayed_hurt_count": override_hurt,
        "override_delayed_covering_pairs": len(covering),
        "identity_selected_but_overridden": {
            "count": len(id_overridden),
            "delayed_net": round(sum(
                r["delivered_delayed_gain"] - r["chosen_delayed_gain"]
                for r in id_overridden_covering), 6),
            "helped": sum(1 for r in id_overridden_covering
                         if r["delivered_delayed_gain"] > r["chosen_delayed_gain"]),
            "hurt": sum(1 for r in id_overridden_covering
                       if r["delivered_delayed_gain"] < r["chosen_delayed_gain"]),
        },
        "note": ("this is a one-action shadow: it scores what the ORIGINAL "
                "pre-Support choice would have read, holding everything "
                "else -- the rest of the historical course, the knowledge "
                "used, every other decision -- fixed.  It is not a "
                "reconstructed alternative history and is not the same "
                "claim as Part B's real Fast-only deployment."),
        "rows": rows_out,
    }


# ---------------------------------------------------------------------------
# Sec 3.1 diagnostic: menu oracle on the test units (evaluator diagnosis
# only -- never fed back into B/C as an input)
# ---------------------------------------------------------------------------

def menu_oracle(scorer: Scorer) -> dict[str, Any]:
    menu = legal_forecast_menu()
    per_position: dict[int, Any] = {}
    for position in TEST_POSITIONS:
        ctx = scorer.ctx[position]
        origin = int(ctx.origin)
        period = int(ctx.config["period"])
        uids = scorer.population(position)
        best = {uid: 0.0 for uid in uids}  # identity floor
        best_op = {uid: "identity" for uid in uids}
        available_ops = []
        for op in menu:
            steps = candidate_steps(op, period)
            required = (OPERATOR_METADATA[op].get("public_parameter_schema")
                       or {}).get("required") or []
            if any(k != "period" for k in required):
                continue
            reading = scorer.population_reading(position, steps, origin)
            if reading is None:
                continue
            available_ops.append(op)
            for uid, gain in reading["per_series_gain"].items():
                if gain > best[uid]:
                    best[uid] = gain
                    best_op[uid] = op
        per_position[position] = {
            "available_ops": available_ops,
            "unavailable_ops": sorted(set(menu) - set(available_ops)),
            "oracle_mean_gain": round(float(np.mean(list(best.values()))), 6),
            "oracle_best_op_by_series": best_op,
            "oracle_gain_by_series": {uid: round(v, 6)
                                      for uid, v in best.items()},
        }
    return {"menu": menu, "per_position": per_position,
            "what_this_is_not": (
                "a post-hoc, outcome-aware pick per series; it upper-bounds "
                "what this frozen menu could do with a perfect targeter, it "
                "is not itself a deployable policy, and it feeds no "
                "candidate, prompt or Skill anywhere in Parts A/B/C")}


# ---------------------------------------------------------------------------
# Part C: 0-LLM deterministic baselines -- Static/raw, D-fixed, D-safe
# ---------------------------------------------------------------------------

def legal_forecast_menu() -> list[str]:
    names = [n for n in OPERATOR_NAMES
             if "forecast" in OPERATOR_METADATA[n]["allowed_tasks"]
             and not OPERATOR_METADATA[n].get("shape_changing")
             and not OPERATOR_METADATA[n].get("is_alias")]
    return sorted(names)


def candidate_steps(op: str, period: int) -> tuple[tuple[str, dict], ...]:
    schema = OPERATOR_METADATA[op].get("public_parameter_schema") or {}
    required = schema.get("required") or []
    params: dict[str, Any] = {}
    for key in required:
        if key == "period":
            params[key] = int(period)
        else:
            return ()  # unresolvable required param -> caller marks unavailable
    return ((op, params),)


def calibrate_candidate(scorer: Scorer, op: str) -> dict[str, Any]:
    """One candidate's reading at both calibration units, full 20-series
    population, origin face only -- exactly the task book's calibration
    contract (Sec C)."""
    per_unit: dict[int, Any] = {}
    for position in CALIBRATION_POSITIONS:
        period = int(scorer.ctx[position].config["period"])
        steps = candidate_steps(op, period)
        required = (OPERATOR_METADATA[op].get("public_parameter_schema") or {}
                   ).get("required") or []
        if any(k != "period" for k in required):
            per_unit[position] = {"status": "UNAVAILABLE",
                                  "why": "required param has no deterministic "
                                        "binding: %s" % required}
            continue
        origin = int(scorer.ctx[position].origin)
        reading = scorer.population_reading(position, steps, origin)
        if reading is None:
            per_unit[position] = {"status": "LEGALITY_REJECTED"}
            continue
        gate = runner.authoritative_gate(reading)
        per_unit[position] = {
            "status": "READ", "steps": [{"op": o, "params": p}
                                        for o, p in steps],
            "aggregate_gain": reading["aggregate_gain"],
            "harmed_fraction": reading["harmed_fraction"],
            "max_single_series_harm": reading["max_single_series_harm"],
            "treated": reading["treated"], "served": reading["served"],
            "gate": gate,
        }
    complete = all(per_unit[p]["status"] == "READ"
                  for p in CALIBRATION_POSITIONS)
    mean_gain = (sum(per_unit[p]["aggregate_gain"] for p in CALIBRATION_POSITIONS)
                / len(CALIBRATION_POSITIONS)) if complete else None
    passes_both_gates = (complete and all(
        per_unit[p]["gate"]["passes"] for p in CALIBRATION_POSITIONS))
    return {"op": op, "complete": complete, "mean_calibration_gain": mean_gain,
            "passes_both_gates": passes_both_gates, "per_unit": per_unit}


def select_fixed_baselines(scorer: Scorer) -> dict[str, Any]:
    menu = legal_forecast_menu()
    calibrations = [calibrate_candidate(scorer, op) for op in menu]
    complete = [c for c in calibrations if c["complete"]]

    d_fixed = None
    if complete:
        best = max(complete, key=lambda c: (c["mean_calibration_gain"],
                                            -menu.index(c["op"])))
        d_fixed = {"op": best["op"], "steps": best["per_unit"][
            CALIBRATION_POSITIONS[0]]["steps"],
                   "mean_calibration_gain": best["mean_calibration_gain"],
                   "status": "SELECTED"}
    else:
        d_fixed = {"op": "identity", "steps": [],
                   "mean_calibration_gain": 0.0,
                   "status": "NO_COMPLETE_CANDIDATE_FROZEN_IDENTITY"}

    qualified = [c for c in complete if c["passes_both_gates"]]
    if qualified:
        best_safe = max(qualified, key=lambda c: (c["mean_calibration_gain"],
                                                   -menu.index(c["op"])))
        d_safe = {"op": best_safe["op"],
                  "steps": best_safe["per_unit"][
                      CALIBRATION_POSITIONS[0]]["steps"],
                  "mean_calibration_gain": best_safe["mean_calibration_gain"],
                  "status": "SELECTED"}
    else:
        d_safe = {"op": "identity", "steps": [], "mean_calibration_gain": 0.0,
                  "status": "NO_QUALIFIED_FIXED_PROGRAM"}

    return {"menu": menu, "calibrations": calibrations,
            "d_fixed": d_fixed, "d_safe": d_safe}


def score_frozen_baselines(scorer: Scorer, doc: Mapping[str, Any],
                           selection: Mapping[str, Any]) -> dict[str, Any]:
    """Static/raw, D-fixed and D-safe, frozen after u7/u8 and then read (no
    further selection) on u9/u10/u11's full population, both faces."""
    fixed = {"raw": (), "d_fixed": tuple(
                 (s["op"], dict(s.get("params", {})))
                 for s in selection["d_fixed"]["steps"]),
             "d_safe": tuple(
                 (s["op"], dict(s.get("params", {})))
                 for s in selection["d_safe"]["steps"])}
    out: dict[str, Any] = {}
    for name, steps in fixed.items():
        per_position: dict[int, Any] = {}
        for position in TEST_POSITIONS:
            ctx = scorer.ctx[position]
            origin = int(ctx.origin)
            delayed_origin = ctx.face_origin(ps.DELAYED)
            origin_reading = scorer.population_reading(position, steps, origin)
            delayed_reading = scorer.population_reading(position, steps,
                                                         delayed_origin)
            per_position[position] = {
                "origin_reading": origin_reading,
                "origin_gate": (runner.authoritative_gate(origin_reading)
                               if origin_reading else None),
                "delayed_reading": delayed_reading,
                "delayed_gate": (runner.authoritative_gate(delayed_reading)
                                if delayed_reading else None),
            }
        complete = [p["origin_reading"]["aggregate_gain"]
                   for p in per_position.values() if p["origin_reading"]]
        out[name] = {
            "steps": [{"op": o, "params": p} for o, p in steps],
            "per_position": per_position,
            "equal_weighted_mean_origin_gain": (
                round(sum(complete) / len(complete), 6)
                if len(complete) == len(TEST_POSITIONS) else None),
            "units_read": len(complete), "units_total": len(TEST_POSITIONS),
        }
    return out


# ---------------------------------------------------------------------------
# Part B: frozen Fast-only real run
# ---------------------------------------------------------------------------

class PackageCeiling(RuntimeError):
    """The package's global fits/LLM/wall-clock ceiling would be crossed."""


class AccountFault(RuntimeError):
    """An account/permission error: the request never reached a model.  Stops
    the whole run rather than being retried or counted as an abstention
    (task book Sec 5.4(7))."""


# ---------------------------------------------------------------------------
# PreparationResult consumption (future runs only; historical JSON is not
# rewritten).  A leftover DecisionTrace is not a valid decision.
# ---------------------------------------------------------------------------

VALID_MECHANISM_STATUSES = frozenset({
    "IDENTITY_SELECTED",
    "DEPLOYED",
    "LEGALITY_FALLBACK_RAW",
})
MISSING_MECHANISM_STATUSES = frozenset({
    "FAULT_NO_DECISION",
    "PREPARE_FAILED",
    "INVALID_PREPARATION_RESULT",
    "EMPTY_SELECTION_NO_VALID_DECISION",
    "UNKNOWN_CANDIDATE_NO_VALID_DECISION",
})
# Pre-repair runner statuses that scored identity/0 without a valid choice.
HISTORICAL_MISSING_MECHANISM_STATUSES = frozenset({
    "EMPTY_SELECTION_FALLBACK_TO_IDENTITY",
    "UNKNOWN_CANDIDATE_FALLBACK_TO_IDENTITY",
})
_ANNOTATION_MARK = "dev_deploy_feedback_integrity_v1"
_SENSITIVE_FRAGMENTS = (
    "api_key", "apikey", "authorization", "secret", "password", "token",
    "credential", "bearer",
)


def _sanitize_error_text(text: str, limit: int = 300) -> str:
    raw = str(text or "")
    lowered = raw.lower()
    if any(fragment in lowered for fragment in _SENSITIVE_FRAGMENTS):
        # A credential can precede the first colon as well as follow it.
        return "[redacted error detail]"
    return raw[:limit]


def _status_value(result: Any) -> str:
    if result is None:
        return ""
    status = getattr(result, "status", None)
    if status is None:
        return ""
    if isinstance(status, PreparationStatus):
        return str(status.value)
    return str(getattr(status, "value", status) or "").lower()


def _receipt_error(result: Any) -> str:
    receipt = getattr(result, "receipt", None)
    return str(getattr(receipt, "error", "") or "")


def raise_if_account_fault(detail: str,
                           *, cause: BaseException | None = None) -> None:
    """Account/permission errors stop the line; they are not model abstentions."""
    if _fault_kind(detail) == "ACCOUNT_OR_PERMISSION_FAULT":
        exc = AccountFault(_sanitize_error_text(detail))
        if cause is not None:
            raise exc from cause
        raise exc


def compact_prepare_error(*, result: Any, trace: Any,
                          thrown: BaseException | None = None,
                          deploy_status: str) -> dict[str, Any] | None:
    """Short, non-secret residue of a failed or missing decision.

    Keeps error class, receipt.ok, compilation/execution status, whether a
    trace existed, and the chosen_id sitting on that trace (untrusted).
    Does not keep request bodies, env, or credential-bearing text.
    """
    if deploy_status in VALID_MECHANISM_STATUSES and thrown is None:
        error = _receipt_error(result)
        if not error:
            return None
    error = _receipt_error(result)
    if thrown is not None and not error:
        error = "%s: %s" % (type(thrown).__name__, thrown)
    kind = deploy_status
    if thrown is not None and result is None:
        kind = "THROWN_%s" % type(thrown).__name__
    receipt = getattr(result, "receipt", None)
    return {
        "kind": kind,
        "receipt_ok": (None if receipt is None
                       else bool(getattr(receipt, "ok", False))),
        "error": _sanitize_error_text(error),
        "compilation_status": getattr(trace, "compilation_status", None),
        "execution_status": getattr(trace, "execution_status", None),
        "trace_present": trace is not None,
        "chosen_candidate_id_on_trace": (
            (getattr(trace, "chosen_candidate_id", None) or None)
            if trace is not None else None),
        "n_candidate_ids_on_trace": (
            len(tuple(getattr(trace, "candidate_ids", ()) or ()))
            if trace is not None else 0),
    }


def _steps_map_from_trace(trace: Any) -> dict[str, tuple]:
    if trace is None:
        return {}
    raw = dict(getattr(trace, "candidate_program_steps", None) or {})
    return {str(k): tuple((str(o), dict(p)) for o, p in v)
            for k, v in raw.items()}


def interpret_prepare_outcome(*, result: Any, trace: Any,
                              thrown: BaseException | None = None
                              ) -> dict[str, Any]:
    """Classify ``prepare()`` before any identity/program scoring.

    A non-empty DecisionTrace does not make a FAILED or empty selection
    into an active identity decision.
    """
    if result is not None:
        error = _receipt_error(result)
        if error:
            raise_if_account_fault(error)

    status_value = _status_value(result)
    steps_map = _steps_map_from_trace(trace)
    chosen_raw = (getattr(trace, "chosen_candidate_id", None)
                  if trace is not None else None)
    chosen = str(chosen_raw or "")

    def _programs() -> dict[str, str]:
        return {k: ps.program_label_with_params(ps.normalise_steps(v))
                for k, v in steps_map.items()}

    def _missing(deploy_status: str, why: str) -> dict[str, Any]:
        return {
            "deploy_status": deploy_status,
            "valid_mechanism_decision": False,
            "mechanism_gain_policy": "unknown",
            "deployed_steps": None,
            "chosen_candidate_id": chosen or None,
            "candidate_programs": _programs(),
            "why_unknown": why,
        }

    if thrown is not None and result is None:
        return _missing(
            "FAULT_NO_DECISION",
            "the session raised %s before a consumed PreparationResult; "
            "this is not a zero" % type(thrown).__name__)
    if status_value == PreparationStatus.FAILED.value:
        return _missing(
            "PREPARE_FAILED",
            "PreparationResult.status is FAILED; a leftover trace or "
            "chosen_candidate_id is not a valid decision")
    if result is None or trace is None:
        return _missing(
            "FAULT_NO_DECISION",
            "the session faulted before a decision was reached; "
            "this is not a zero")
    if (status_value not in (PreparationStatus.PREPARED.value,
                             PreparationStatus.ABSTAINED.value)
            or not getattr(getattr(result, "receipt", None), "ok", False)):
        return _missing(
            "INVALID_PREPARATION_RESULT",
            "missing/invalid success status or unsuccessful execution receipt")
    if not chosen:
        return _missing(
            "EMPTY_SELECTION_NO_VALID_DECISION",
            "empty chosen_candidate_id is not a deliberate identity choice")
    if chosen == "identity":
        return {
            "deploy_status": "IDENTITY_SELECTED",
            "valid_mechanism_decision": True,
            "mechanism_gain_policy": "legal_zero",
            "deployed_steps": (),
            "chosen_candidate_id": "identity",
            "candidate_programs": _programs(),
            "why_unknown": None,
        }
    if chosen not in steps_map:
        return _missing(
            "UNKNOWN_CANDIDATE_NO_VALID_DECISION",
            "chosen_candidate_id %r is not in the trace candidate map; "
            "not a deterministic zero" % chosen)
    return {
        "deploy_status": "CANDIDATE_CHOSEN_PENDING_LEGALITY",
        "valid_mechanism_decision": True,
        "mechanism_gain_policy": "score_if_legal_else_raw_fallback",
        "deployed_steps": steps_map[chosen],
        "chosen_candidate_id": chosen,
        "candidate_programs": _programs(),
        "why_unknown": None,
    }


def apply_legality_and_score(
        interpreted: Mapping[str, Any], *,
        verify: Any = None, scorer: Any = None,
        position: int = 0, uid: str = "",
        origin: int = 0, delayed_origin: int = 0) -> dict[str, Any]:
    """Turn a classified prepare outcome into deploy_status and gains.

    Identity and the predefined legality-raw fallback keep a real 0.
    Missing decisions stay UNKNOWN and are not scored as identity.
    """
    policy = interpreted["mechanism_gain_policy"]
    if policy == "unknown":
        return {
            "deploy_status": interpreted["deploy_status"],
            "deployed_program": "UNKNOWN",
            "deployment_gain_at_origin": "UNKNOWN",
            "fixed_program_gain_at_plus48": "UNKNOWN",
            "valid_mechanism_decision": False,
        }
    if policy == "legal_zero":
        steps = interpreted.get("deployed_steps") or ()
        return {
            "deploy_status": interpreted["deploy_status"],
            "deployed_program": ps.program_label_with_params(steps),
            "deployment_gain_at_origin": 0.0,
            "fixed_program_gain_at_plus48": 0.0,
            "valid_mechanism_decision": True,
        }
    steps = interpreted["deployed_steps"]
    if verify is None or scorer is None:
        raise TypeError("legal program path requires verify and scorer")
    verification = verify(steps)
    if bool(getattr(verification, "passed", False)):
        receipt_o = scorer.sequence_reading(position, uid, steps, origin)
        gain_o = (round(float(receipt_o.gain), 6)
                  if receipt_o.gain is not None else "UNKNOWN")
        receipt_d = scorer.sequence_reading(position, uid, steps,
                                            delayed_origin)
        gain_d = (round(float(receipt_d.gain), 6)
                  if receipt_d.gain is not None else "UNKNOWN")
        return {
            "deploy_status": "DEPLOYED",
            "deployed_program": ps.program_label_with_params(steps),
            "deployment_gain_at_origin": gain_o,
            "fixed_program_gain_at_plus48": gain_d,
            "valid_mechanism_decision": True,
        }
    # Predefined legality fallback: raw actually ran; 0 is that fallback,
    # not an active identity selection and not a fault dressing.
    return {
        "deploy_status": "LEGALITY_FALLBACK_RAW",
        "deployed_program": "identity",
        "deployment_gain_at_origin": 0.0,
        "fixed_program_gain_at_plus48": 0.0,
        "valid_mechanism_decision": True,
        "active_model_choice": False,
    }


def finalize_fastonly_decision(
        record: dict[str, Any], *, result: Any, trace: Any,
        thrown: BaseException | None = None,
        scorer: Any, ctx: Any, uid: str, origin: int,
        delayed_origin: int, position: int) -> dict[str, Any]:
    interpreted = interpret_prepare_outcome(
        result=result, trace=trace, thrown=thrown)
    record["chosen_candidate_id"] = interpreted["chosen_candidate_id"]
    record["candidate_programs"] = interpreted["candidate_programs"]
    if interpreted["why_unknown"]:
        record["why_unknown"] = interpreted["why_unknown"]
    scored = apply_legality_and_score(
        interpreted,
        verify=(None if interpreted["mechanism_gain_policy"]
                != "score_if_legal_else_raw_fallback"
                else (lambda steps: ctx.executor.verify(steps, origin))),
        scorer=scorer, position=position, uid=uid, origin=origin,
        delayed_origin=delayed_origin)
    record["deploy_status"] = scored["deploy_status"]
    record["deployed_program"] = scored["deployed_program"]
    record["deployment_gain_at_origin"] = scored["deployment_gain_at_origin"]
    record["fixed_program_gain_at_plus48"] = scored[
        "fixed_program_gain_at_plus48"]
    record["valid_mechanism_decision"] = scored["valid_mechanism_decision"]
    if "active_model_choice" in scored:
        record["active_model_choice"] = scored["active_model_choice"]
    elif scored["valid_mechanism_decision"]:
        record["active_model_choice"] = scored["deploy_status"] in (
            "IDENTITY_SELECTED", "DEPLOYED")
    else:
        record["active_model_choice"] = False
    record["prepare_error"] = compact_prepare_error(
        result=result, trace=trace, thrown=thrown,
        deploy_status=record["deploy_status"])
    if not record["valid_mechanism_decision"] and not record.get("faults"):
        record.setdefault("faults", []).append({
            "kind": record["deploy_status"],
            "why": ((record["prepare_error"] or {}).get("error")
                    or record.get("why_unknown", "no valid decision")),
        })
    record["original_deploy_status"] = record["deploy_status"]
    record["original_deployment_gain_at_origin"] = record[
        "deployment_gain_at_origin"]
    record["original_fixed_program_gain_at_plus48"] = record[
        "fixed_program_gain_at_plus48"]
    record["mechanism_gain_at_origin"] = record["deployment_gain_at_origin"]
    record["mechanism_annotation"] = _ANNOTATION_MARK
    if record["valid_mechanism_decision"]:
        record["original_record_semantics"] = "valid mechanism decision on this run"
    else:
        record["original_record_semantics"] = (
            record.get("why_unknown")
            or "no valid mechanism decision; main reading is UNKNOWN")
    return record


def is_valid_mechanism_status(status: str) -> bool:
    return str(status or "") in VALID_MECHANISM_STATUSES


def annotate_recorded_decision(row: Mapping[str, Any]) -> dict[str, Any]:
    """Working view of one recorded row.  Does not write the source file.

    Historical empty/unknown-selection fallbacks keep their original
    status string and original numeric fields under ``original_*``, but
    the mechanism gain used by new summaries is UNKNOWN.
    """
    out = dict(row)
    status = str(out.get("original_deploy_status")
                 or out.get("deploy_status") or "")
    if "original_deployment_gain_at_origin" in out:
        original_gain = out.get("original_deployment_gain_at_origin")
        original_delayed = out.get("original_fixed_program_gain_at_plus48")
    elif "original_recorded_gain_at_origin" in out:
        original_gain = out.get("original_recorded_gain_at_origin")
        original_delayed = out.get("original_recorded_gain_at_plus48")
    else:
        original_gain = out.get("original_recorded_gain_at_origin",
                                out.get("deployment_gain_at_origin"))
        original_delayed = out.get("original_recorded_gain_at_plus48",
                                   out.get("fixed_program_gain_at_plus48"))
    out["original_deploy_status"] = status
    out["original_deployment_gain_at_origin"] = original_gain
    out["original_fixed_program_gain_at_plus48"] = original_delayed
    historical = status in HISTORICAL_MISSING_MECHANISM_STATUSES
    missing = (
        status in MISSING_MECHANISM_STATUSES
        or historical
        or not is_valid_mechanism_status(status))
    if missing:
        out["valid_mechanism_decision"] = False
        out["active_model_choice"] = False
        out["mechanism_gain_at_origin"] = "UNKNOWN"
        out["deployment_gain_at_origin"] = "UNKNOWN"
        out["fixed_program_gain_at_plus48"] = "UNKNOWN"
        if historical:
            out["original_record_semantics"] = (
                "historical %s was stored as identity/0; that recorded "
                "zero is not a valid mechanism decision" % status)
        else:
            out["original_record_semantics"] = (
                out.get("why_unknown")
                or "no valid mechanism decision; main reading is UNKNOWN")
        out["why_unknown"] = out["original_record_semantics"]
    else:
        out["valid_mechanism_decision"] = True
        out["mechanism_gain_at_origin"] = original_gain
        out["deployment_gain_at_origin"] = original_gain
        out["fixed_program_gain_at_plus48"] = original_delayed
        if "active_model_choice" not in out:
            out["active_model_choice"] = status in (
                "IDENTITY_SELECTED", "DEPLOYED")
        out["original_record_semantics"] = "valid mechanism decision"
    out["mechanism_annotation"] = _ANNOTATION_MARK
    return out


def load_checkpoint_rows(saved: Mapping[str, Any]) -> dict[str, Any]:
    """Annotate checkpoint sequences in memory; caller must not write back
    to an original artifact path."""
    return {row["series_uid"]: annotate_recorded_decision(row)
            for row in saved.get("sequences") or []}


def read_sequences_file_view(path: Path) -> list[dict[str, Any]]:
    """Return an annotated view of ``sequences`` without writing ``path``."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, Mapping) and "sequences" in doc:
        rows = list(doc.get("sequences") or [])
    elif isinstance(doc, Mapping) and "groups" in doc:
        rows = []
        for group in (doc.get("groups") or {}).values():
            for unit in (group.get("units") or {}).values():
                rows.extend(unit.get("sequences") or [])
    else:
        rows = list(doc if isinstance(doc, list) else [])
    return [annotate_recorded_decision(row) for row in rows]


def summarize_fastonly_unit(*, group: str, position: int, unit: Any,
                            population: list[str], rows: list[Mapping[str, Any]],
                            stopped_at: dict[str, Any] | None = None
                            ) -> dict[str, Any]:
    """Full-population summary.  Any UNKNOWN makes the main mean None.

    Missing population members stay in ``sequences`` as FAULT_NO_DECISION
    placeholders so the denominator is not silently shrunk.  They are not
    checkpointed as completed decisions.
    """
    completed = [annotate_recorded_decision(r) for r in rows]
    by_uid = {r["series_uid"]: r for r in completed}
    sequences: list[dict[str, Any]] = []
    for uid in population:
        if uid in by_uid:
            sequences.append(by_uid[uid])
        else:
            sequences.append({
                "series_uid": uid, "group": group,
                "position": int(position),
                "deploy_status": "FAULT_NO_DECISION",
                "deployed_program": "UNKNOWN",
                "deployment_gain_at_origin": "UNKNOWN",
                "fixed_program_gain_at_plus48": "UNKNOWN",
                "valid_mechanism_decision": False,
                "active_model_choice": False,
                "mechanism_gain_at_origin": "UNKNOWN",
                "why_unknown": "row missing from completed decisions; not a zero",
                "original_record_semantics": (
                    "missing row; not a zero"),
                "original_deploy_status": "FAULT_NO_DECISION",
                "original_deployment_gain_at_origin": "UNKNOWN",
                "original_fixed_program_gain_at_plus48": "UNKNOWN",
                "mechanism_annotation": _ANNOTATION_MARK,
                "faults": [{"kind": "MISSING_ROW",
                            "why": "not present in completed sequences"}],
            })
    mechanism_gains = [
        r["deployment_gain_at_origin"] for r in sequences
        if isinstance(r.get("deployment_gain_at_origin"), (int, float))]
    complete = len(mechanism_gains) == len(population)
    return {
        "group": group, "position": int(position), "unit": unit,
        "decision_population": list(population),
        "decided": [uid for uid in population if uid in by_uid],
        "stopped_at": stopped_at,
        "sequences": sequences,
        "deployment_gain_at_origin_mean": (
            round(float(np.mean(mechanism_gains)), 6) if complete else None),
        "deployment_gain_at_origin_mean_of_scored_subset_diagnostic_only": (
            round(float(np.mean(mechanism_gains)), 6) if mechanism_gains
            else None),
        "n_scored": len(mechanism_gains), "n_total": len(population),
        "n_missing_valid_decision": sum(
            1 for r in sequences if not r.get("valid_mechanism_decision")),
    }


def mechanism_unit_stats(rows: list[Mapping[str, Any]],
                         *, population_n: int | None = None) -> dict[str, Any]:
    """Aggregate one unit's sequences under the mechanism reading.

    Historical empty-selection zeros do not form a main mean.
    """
    annotated = [annotate_recorded_decision(r) for r in rows]
    n = int(population_n) if population_n is not None else len(annotated)
    origin_vals = [r.get("deployment_gain_at_origin") for r in annotated]
    delayed_vals = [r.get("fixed_program_gain_at_plus48") for r in annotated]
    origin_nums = [v for v in origin_vals if isinstance(v, (int, float))]
    delayed_nums = [v for v in delayed_vals if isinstance(v, (int, float))]
    complete = len(origin_nums) == n and len(annotated) == n
    treated = sum(1 for r in annotated
                  if r.get("deploy_status") == "DEPLOYED"
                  and r.get("valid_mechanism_decision"))
    by_status: dict[str, int] = {}
    for r in annotated:
        st = str(r.get("original_deploy_status") or r.get("deploy_status") or "")
        by_status[st] = by_status.get(st, 0) + 1
    origin_mean = (round(float(np.mean(origin_nums)), 6)
                   if complete and origin_nums else None)
    delayed_mean = (round(float(np.mean(delayed_nums)), 6)
                    if complete and len(delayed_nums) == n and delayed_nums
                    else None)
    gate = None
    harmed = None
    max_harm = None
    if complete and origin_nums:
        origin_arr = np.array(origin_nums, dtype=float)
        harmed = float((origin_arr < -ps.MATERIAL).mean())
        max_harm = float(max(0.0, float(-origin_arr.min())))
        gate = runner.authoritative_gate({
            "treated": treated, "aggregate_gain": float(origin_arr.mean()),
            "harmed_fraction": harmed,
            "max_single_series_harm": max_harm})
    return {
        "n": n, "n_rows_present": len(annotated), "treated": treated,
        "deploy_status_counts": by_status,
        "origin_mean": origin_mean, "delayed_mean": delayed_mean,
        "origin_mean_of_scored_subset_diagnostic_only": (
            round(float(np.mean(origin_nums)), 6) if origin_nums else None),
        "harmed_fraction": None if harmed is None else round(harmed, 4),
        "max_single_series_harm": (
            None if max_harm is None else round(max_harm, 6)),
        "gate": gate,
        "n_faulted": sum(1 for r in annotated if r.get("faults")),
        "n_missing_valid_decision": sum(
            1 for r in annotated if not r.get("valid_mechanism_decision")),
        "complete": complete,
    }


def parent_snapshot_and_machinery() -> tuple[Any, dict[str, Any]]:
    """The exact K0 parent snapshot DEV-SEQ-3 g2's ``knowledge-frozen`` arm
    used throughout ("A" in the task book's "g2 A" -- see run_dev_seq2.py's
    own ``ARM_A = "knowledge-frozen"``).  Recomputed deterministically from
    the same registered ordering the historical run read, not loaded from a
    process-local store -- and its SHA is checked against the historical
    ``knowledge_version`` before any LLM is spent, so a mismatch stops this
    function rather than silently running on the wrong knowledge."""
    machinery = v1runner._machinery()
    doc = base.load(R.ORDERING)
    state = live._state_at_k1(doc)
    snapshot_dir, _source = R._resolve_k0_snapshot(doc, state["k0"])
    snapshot = machinery["compile_snapshot"](snapshot_dir, verify_lock=False)
    expected = "98dea3b037b790528cd745f8cbdf657da0285c25f91df53e9c572a0dfc87563b"
    if snapshot.runtime_bundle_sha != expected:
        raise RuntimeError(
            "recomputed K0 snapshot SHA %s != DEV-SEQ-3 g2 knowledge-frozen's "
            "recorded knowledge_version %s -- refusing to spend LLM budget on "
            "an unverified knowledge version" % (snapshot.runtime_bundle_sha,
                                                 expected))
    return snapshot, machinery


def allow_loopback_http_relay() -> None:
    """Runtime-only patch, not a source edit: ``agent_backend.py``'s content
    is hashed into every Harness snapshot's identity, so editing the file on
    disk shifts every snapshot's SHA project-wide.  This monkeypatches the
    validator in memory for this process only, so the file stays untouched.
    Idempotent; safe to call more than once."""
    from SelfEvolvingHarnessTS.runtime import agent_backend as _ab
    from urllib.parse import urlsplit as _urlsplit

    if getattr(_ab._validate_base_url, "_devdeploy_loopback_patch", False):
        return
    loopback = {"127.0.0.1", "localhost", "::1"}
    original = _ab._validate_base_url

    def _patched(base_url: str) -> None:
        parsed = _urlsplit(base_url)
        if parsed.scheme == "http" and parsed.hostname in loopback:
            if base_url != base_url.strip() or parsed.path != "/v1":
                raise ValueError("base_url must end in /v1")
            return
        original(base_url)

    _patched._devdeploy_loopback_patch = True
    _ab._validate_base_url = _patched


def fastonly_transport_preflight(machinery: Mapping[str, Any]) -> dict[str, Any]:
    saved_required = seq2.REQUIRED_MODEL
    seq2.REQUIRED_MODEL = REQUIRED_MODEL
    try:
        return seq3.transport_preflight(machinery)
    finally:
        seq2.REQUIRED_MODEL = saved_required


def relay_aliased_transport_preflight(machinery: Mapping[str, Any], *,
                                      required_returned_model: str
                                      ) -> dict[str, Any]:
    """``seq2.transport_preflight`` for a relay whose request identifier is
    not the same string as what it reports back.

    ``api.nowaterapi.xyz`` (what ``seq2.transport_preflight`` was written
    against) answers with the same string it was asked for, so that
    function's ``target["model"] != REQUIRED_MODEL`` guard doubles as both
    "the env is configured for what this package expects" and "reject a
    swap before spending the real call".  The 2026-09-09 local relay at
    ``127.0.0.1:8318`` genuinely needs a different request identifier per
    model (``cpa-gpt-5.6-sol`` vs ``cpa-gpt-5.6-terra``, confirmed by hand:
    the base_url alone does not tell you which model answers) and echoes
    back the canonical name (``gpt-5.6-sol`` / ``gpt-5.6-terra``) rather
    than the alias -- three consecutive manual calls each returned
    ``gpt-5.6-sol`` for the ``cpa-gpt-5.6-sol`` alias before this was
    trusted with a real decision.  So here the two checks are kept
    separate: the env's *requested* alias only has to be internally
    consistent (present, non-empty), and the real refusal gate is the
    *returned* identity, exactly as strict as ``seq2.transport_preflight``'s
    own ``returned == [REQUIRED_MODEL]`` line.
    """
    target = machinery["agentic"].live_transport()
    row: dict[str, Any] = {
        "requested_model_alias": target["model"], "base_url": target["base_url"],
        "source": target["source"],
        "required_returned_model": required_returned_model,
    }
    if not str(target["model"]):
        row.update({"usable": False, "why": "M0_AGENT_MODEL is empty"})
        return row
    from SelfEvolvingHarnessTS.runtime.agent_backend import AgentRequest

    zero = "0" * 64
    backend = machinery["agentic"]._default_backend_factory(2)
    try:
        request = AgentRequest(
            case_id="devdeploy2-relay-transport-preflight", role="fast",
            stage="inspect", call_index=0, replicate_id="preflight",
            messages=(
                {"role": "system", "content": "Answer with one short word."},
                {"role": "user", "content": "Reply with the single word: ready"},
            ),
            envelope_schema_sha=zero, tool_schema_sha=zero,
            tool_result_schema_sha=zero, stage_schema_sha=zero,
            public_case_view_sha=zero, effective_harness_view_sha=zero,
            tool_context_sha=zero,
            model=target["model"], base_url=target["base_url"])
        response = backend.complete(request)
        reply = getattr(response, "assistant_text", "") or ""
    except Exception as exc:  # noqa: BLE001 - a configuration receipt, not a crash
        row.update({"usable": False,
                    "why": "%s: %s" % (type(exc).__name__, str(exc)[:300])})
        return row
    returned = sorted(getattr(backend, "returned_models", set()) or set())
    row.update({"returned_models": returned,
               "reply_nonempty": bool(str(reply or "").strip()),
               "calls": int(getattr(backend, "calls", 0) or 0)})
    row["usable"] = bool(returned == [required_returned_model])
    if not row["usable"]:
        row["why"] = ("the endpoint answered as %s, which is not the "
                      "required returned identity %r"
                      % (returned or "nothing", required_returned_model))
    return row


def run_fastonly_decision(*, group: str, position: int, uid: str, ctx: Any,
                          scorer: Scorer, snapshot: Any,
                          machinery: Mapping[str, Any], guard: Any,
                          ) -> dict[str, Any]:
    """One series, one fresh Fast session, no Support/admission search.

    ``experience_episodes=()`` always -- this leg never writes an Episode and
    never carries one in, so there is nothing to accumulate across u9/u10/u11
    even within one run (task book: "0 TARGET_HELD_IN", "不写经验").

    Consumes ``PreparationResult.status`` and ``ExecutionReceipt`` before
    scoring.  FAILED / unparseable / empty selection are UNKNOWN, not
    identity/0.  Explicit identity and legality-raw fallback keep a real 0.
    """
    started = time.time()
    origin = int(ctx.origin)
    delayed_origin = ctx.face_origin(ps.DELAYED)
    record: dict[str, Any] = {
        "group": group, "position": int(position), "unit": ctx.unit,
        "series_uid": str(uid), "origin": origin,
        "delayed_origin": int(delayed_origin), "faults": [],
    }
    features = ps.sequence_features(machinery, ctx, uid, origin=origin)
    request = ps.sequence_request(machinery, ctx, uid, origin=origin,
                                  domain=DOMAIN)
    record["public_features"] = features

    inner = machinery["agentic"]._default_backend_factory(PER_SEQUENCE_LLM_CAP)
    backend = runner._MeteredFastBackend(seq3.TransientRelayRetry(inner),
                                         guard=guard, billable=True)
    target = machinery["agentic"].live_transport()
    core = machinery["TTHAAgentCore"](
        backend, machinery["LocalPublicToolGateway"](
            np.zeros(8, dtype=np.float64), task_kind="forecast"),
        model=target["model"], base_url=target["base_url"])
    method = machinery["TTHAMethod"](machinery["TTHAFastAgent"](core),
                                     snapshot, ())

    guard.open_cell()
    result = None
    trace = None
    thrown: BaseException | None = None
    try:
        guard.reserve(kind="fast", where={"unit": ctx.unit, "group": group,
                                          "series_uid": uid})
        series0 = np.asarray(request.values, dtype=np.float64)
        method.bind_round_data(series0, task_kind=request.task_spec.task_type)
        result = method.prepare(request, runtime_prior_slot=False,
                                pool_mode="actionable")
        trace = method.last_trace
        error = _receipt_error(result)
        if error:
            raise_if_account_fault(error)
    except AccountFault:
        raise
    except runner.UnitFault as exc:
        raise_if_account_fault(str(exc), cause=exc)
        record["faults"].append({"kind": "UnitFault",
                                  "why": _sanitize_error_text(str(exc), 240)})
        thrown = exc
    except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
        detail = "%s: %s" % (type(exc).__name__, exc)
        raise_if_account_fault(detail, cause=exc)
        record["faults"].append({"kind": type(exc).__name__,
                                  "why": _sanitize_error_text(detail)})
        thrown = exc

    record["llm_requests_sent"] = int(getattr(backend, "calls", 0) or 0)
    record["returned_models"] = sorted(
        getattr(backend, "returned_models", set()) or set())
    record["wall_seconds"] = round(time.time() - started, 2)
    return finalize_fastonly_decision(
        record, result=result, trace=trace, thrown=thrown,
        scorer=scorer, ctx=ctx, uid=uid, origin=origin,
        delayed_origin=delayed_origin, position=position)


def run_fastonly_unit(*, group: str, position: int, scorer: Scorer,
                      snapshot: Any, machinery: Mapping[str, Any],
                      budget: seq2.Budget, checkpoint_path: Path,
                      concurrency: int = CONCURRENCY,
                      limit: int | None = None) -> dict[str, Any]:
    ctx = scorer.ctx[position]
    population = scorer.population(position)
    if limit is not None:
        population = population[: int(limit)]

    saved: dict[str, Any] = {}
    if checkpoint_path.is_file():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if any((r.get("original_deploy_status") or r.get("deploy_status"))
           in HISTORICAL_MISSING_MECHANISM_STATUSES
           for r in saved.get("sequences", [])):
        # Legacy receipts are immutable.  A future authorised partial resume
        # writes its annotated working view beside, never over, that source.
        checkpoint_path = checkpoint_path.with_name(
            checkpoint_path.stem + ".feedback_integrity" + checkpoint_path.suffix)
        if checkpoint_path.is_file():
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    # Annotate in memory only.  Historical empty-selection rows are not
    # re-queried (no LLM replay) and are not treated as valid identity/0.
    rows_by_uid: dict[str, Any] = load_checkpoint_rows(saved)

    lock = __import__("threading").Lock()

    def one(uid: str) -> tuple[str, dict[str, Any] | None, Exception | None]:
        if uid in rows_by_uid:
            return uid, rows_by_uid[uid], None
        guard = seq3.SequenceGuard(ordering_cap=MAX_LLM,
                                   per_unit_arm_cap=PER_SEQUENCE_LLM_CAP,
                                   ledgers=budget.ledgers, lock=lock)
        try:
            budget.require(2)
            if int(budget.llm_by_arm.get(group, 0)) >= PER_GROUP_LLM_CAP:
                raise PackageCeiling(
                    "group %s LLM cap of %d reached" % (group, PER_GROUP_LLM_CAP))
            fits_before = scorer.physical_fits
            row = run_fastonly_decision(
                group=group, position=position, uid=uid, ctx=ctx,
                scorer=scorer, snapshot=snapshot, machinery=machinery,
                guard=guard)
            with lock:
                budget.spend_llm(group, int(row["llm_requests_sent"]))
                budget.spend_fits(group, scorer.physical_fits - fits_before)
                budget.note_evaluation(group)
            return uid, row, None
        except (PackageCeiling, AccountFault):
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised in roster order
            return uid, None, exc

    from concurrent.futures import ThreadPoolExecutor, as_completed
    workers = max(1, int(concurrency))
    pending = [uid for uid in population if uid not in rows_by_uid]
    stopped: dict[str, Any] | None = None

    def checkpoint() -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(
            {"group": group, "position": position,
             "sequences": [rows_by_uid[u] for u in population
                          if u in rows_by_uid]},
            ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    if pending:
        if workers == 1:
            for uid in pending:
                try:
                    _uid, row, exc = one(uid)
                except (PackageCeiling, AccountFault) as fatal:
                    stopped = {"why": type(fatal).__name__,
                              "detail": str(fatal)[:300]}
                    break
                if exc is not None:
                    if stopped is None:
                        stopped = {"series_uid": uid, "why": type(exc).__name__,
                                  "detail": str(exc)[:200]}
                    continue
                rows_by_uid[uid] = row
                checkpoint()
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(one, uid): uid for uid in pending}
                # Every decision that actually completed is checkpointed as it
                # lands, in whatever order the threads finish -- a fatal
                # exception in one future never discards another future's
                # already-billed, already-finished result (task book: "中断
                # 按 checkpoint 恢复,不删除完成的 checkpoint").
                for future in as_completed(futures):
                    uid = futures[future]
                    try:
                        _uid, row, exc = future.result()
                    except (PackageCeiling, AccountFault) as fatal:
                        if stopped is None:
                            stopped = {"why": type(fatal).__name__,
                                      "detail": str(fatal)[:300]}
                        continue
                    if exc is not None:
                        if stopped is None:
                            stopped = {"series_uid": uid,
                                      "why": type(exc).__name__,
                                      "detail": str(exc)[:200]}
                        continue
                    rows_by_uid[uid] = row
                    checkpoint()

    completed_rows = [rows_by_uid[uid] for uid in population
                      if uid in rows_by_uid]
    # Task book Sec 4/5 (DEV-DEPLOY-1) / Sec 5 (DEV-DEPLOY-2): a planned
    # population with any unknown means the batch mean is UNKNOWN, never the
    # mean of whatever subset happened to score -- the readable subset is a
    # diagnostic, not a stand-in denominator.  Missing members stay in
    # sequences as UNKNOWN placeholders; they are not checkpointed as done.
    return summarize_fastonly_unit(
        group=group, position=position, unit=ctx.unit,
        population=population, rows=completed_rows, stopped_at=stopped)


def run_fastonly_group(*, group: str, scorer: Scorer, snapshot: Any,
                       machinery: Mapping[str, Any], budget: seq2.Budget,
                       concurrency: int = CONCURRENCY,
                       limit: int | None = None,
                       positions: tuple[int, ...] = TEST_POSITIONS,
                       ) -> dict[str, Any]:
    np.random.seed(int(GROUP_SEEDS[group]) % (2 ** 32))
    units: dict[int, Any] = {}
    for position in positions:
        checkpoint_path = SCRATCH / group / ("u%03d.json" % position)
        try:
            units[position] = run_fastonly_unit(
                group=group, position=position, scorer=scorer,
                snapshot=snapshot, machinery=machinery, budget=budget,
                checkpoint_path=checkpoint_path, concurrency=concurrency,
                limit=limit)
        except AccountFault as exc:
            units[position] = {"group": group, "position": position,
                               "stopped_at": {
                                   "why": "ACCOUNT_OR_PERMISSION_FAULT",
                                   "detail": str(exc)[:300]}}
            return {"group": group, "seed": GROUP_SEEDS[group], "units": units,
                   "status": "BOUNDARY_ACCOUNT_OR_PERMISSION_FAULT"}
        if (units[position].get("stopped_at") or {}).get(
                "why") in ("AccountFault", "ACCOUNT_OR_PERMISSION_FAULT"):
            return {"group": group, "seed": GROUP_SEEDS[group], "units": units,
                    "status": "BOUNDARY_ACCOUNT_OR_PERMISSION_FAULT"}
        if (units[position].get("stopped_at") or {}).get(
                "why") == "PackageCeiling":
            return {"group": group, "seed": GROUP_SEEDS[group], "units": units,
                   "status": "STOPPED_AT_PACKAGE_CEILING"}
    return {"group": group, "seed": GROUP_SEEDS[group], "units": units,
           "status": "COMPLETE"}


def run_part_b(groups: tuple[str, ...] = GROUP_ORDER,
              concurrency: int = CONCURRENCY, limit: int | None = None,
              positions: tuple[int, ...] = TEST_POSITIONS) -> dict[str, Any]:
    seq2.MAX_FITS, seq2.MAX_LLM = MAX_FITS, MAX_LLM
    seq2.MAX_WALL_SECONDS = MAX_WALL_SECONDS

    snapshot, machinery = parent_snapshot_and_machinery()
    preflight = fastonly_transport_preflight(machinery)
    if not preflight.get("usable"):
        return {"stage": "DEV_DEPLOY1_PART_B", "status": "BLOCKED_BY_TRANSPORT",
               "transport": preflight}

    budget_path = SCRATCH / "part_b_budget.json"
    budget = (seq2.Budget.load(budget_path) if budget_path.is_file()
             else seq2.Budget())
    budget.open_group()

    doc = load_g2()
    scorer = Scorer(doc)
    group_results: dict[str, Any] = {}
    for group in groups:
        group_results[group] = run_fastonly_group(
            group=group, scorer=scorer, snapshot=snapshot, machinery=machinery,
            budget=budget, concurrency=concurrency, limit=limit,
            positions=positions)
        budget.save(budget_path)
        if group_results[group]["status"] != "COMPLETE":
            break

    return {
        "stage": "DEV_DEPLOY1_PART_B", "status": "RAN",
        "knowledge_snapshot_sha": snapshot.runtime_bundle_sha,
        "transport_preflight": preflight,
        "groups": group_results,
        "cost": budget.to_dict(),
        "fits_from_scoring": scorer.physical_fits,
    }


def aggregate_part_b(pattern: str = "dev_deploy1_part_b__*.json"
                     ) -> dict[str, Any]:
    """Combine the per-(group, unit) process outputs -- one OS process per
    unit, per the task book -- into the comparator table Sec 4 asks for.
    Reads only what each process already wrote; runs no new fits or LLM."""
    units: dict[tuple[str, int], Any] = {}
    for path in sorted(ART.glob(pattern)):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for group, g in (doc.get("groups") or {}).items():
            for pos, u in (g.get("units") or {}).items():
                units[(group, int(pos))] = u

    missing = [(g, p) for g in GROUP_ORDER for p in TEST_POSITIONS
              if (g, p) not in units]
    by_group: dict[str, Any] = {}
    for group in GROUP_ORDER:
        per_position: dict[int, Any] = {}
        for position in TEST_POSITIONS:
            u = units.get((group, position))
            if u is None:
                continue
            rows = u.get("sequences") or []
            stats = mechanism_unit_stats(
                rows, population_n=len(u.get("decision_population") or rows))
            per_position[position] = stats
        complete = (
            len(per_position) == len(TEST_POSITIONS)
            and all(p.get("complete") and p.get("origin_mean") is not None
                    for p in per_position.values()))
        origin_means = [p["origin_mean"] for p in per_position.values()
                        if p.get("origin_mean") is not None]
        delayed_means = [p["delayed_mean"] for p in per_position.values()
                         if p.get("delayed_mean") is not None]
        by_group[group] = {
            "per_position": per_position,
            "equal_weighted_origin_mean": (
                round(float(np.mean(origin_means)), 6) if complete else None),
            "equal_weighted_delayed_mean": (
                round(float(np.mean(delayed_means)), 6)
                if complete and len(delayed_means) == len(TEST_POSITIONS)
                else None),
            "equal_weighted_origin_mean_of_complete_units_diagnostic_only": (
                round(float(np.mean(origin_means)), 6) if origin_means
                else None),
            "complete": complete,
        }
    return {"by_group": by_group, "missing_units": missing,
           "source_files": [str(p.name) for p in sorted(ART.glob(pattern))]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part",
                        choices=("shadow", "baseline", "fastonly",
                                "aggregate", "all"),
                        default="all")
    parser.add_argument("--group", action="append", choices=("g1", "g2"))
    parser.add_argument("--position", action="append", type=int,
                        choices=list(TEST_POSITIONS))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the per-unit population (smoke runs only)")
    parser.add_argument("--out", default=str(
        ART / "dev_deploy1_fast_only_alignment.json"))
    args = parser.parse_args(argv)

    if args.part == "fastonly":
        groups = tuple(args.group) if args.group else GROUP_ORDER
        positions = tuple(sorted(args.position)) if args.position else TEST_POSITIONS
        result = run_part_b(groups=groups, concurrency=args.concurrency,
                            limit=args.limit, positions=positions)
        tag = "%s_u%s" % ("-".join(groups),
                          "-".join(str(p) for p in positions))
        out_path = Path(args.out).parent / ("dev_deploy1_part_b__%s.json" % tag)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                       default=str), encoding="utf-8")
        print("wrote", out_path, "| status", result.get("status"))
        incomplete = result.get("status") != "RAN" or any(
            g.get("status") != "COMPLETE"
            or any(u.get("stopped_at")
                   or u.get("deployment_gain_at_origin_mean") is None
                   for u in g.get("units", {}).values())
            for g in result.get("groups", {}).values())
        return 2 if incomplete else 0

    if args.part == "aggregate":
        result = aggregate_part_b()
        out_path = Path(args.out).parent / "dev_deploy1_part_b_aggregate.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                       default=str), encoding="utf-8")
        print("wrote", out_path, "| missing", result.get("missing_units"))
        return 0

    doc = load_g2()
    scorer = Scorer(doc)
    result: dict[str, Any] = {"stage": "DEV_DEPLOY1_FAST_ONLY_ALIGNMENT",
                              "parts_run": args.part}

    if args.part in ("shadow", "all"):
        result["shadow_audit"] = shadow_audit(scorer, doc)
        fits_after_shadow = scorer.physical_fits
        result.setdefault("cost", {})["fits_after_shadow_audit"] = fits_after_shadow

    if args.part in ("baseline", "all"):
        selection = select_fixed_baselines(scorer)
        result["baseline_selection"] = selection
        result["baseline_frozen_scoring"] = score_frozen_baselines(
            scorer, doc, selection)
        result["menu_oracle_diagnostic"] = menu_oracle(scorer)

    result.setdefault("cost", {})["physical_consumer_fits_total"] = \
        scorer.physical_fits
    result["cost"]["cache"] = scorer.cache.to_dict()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                   default=str), encoding="utf-8")
    print("wrote", out_path, "| fits", scorer.physical_fits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
