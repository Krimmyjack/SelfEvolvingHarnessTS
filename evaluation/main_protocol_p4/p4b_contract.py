"""P4b frozen contract: origins, arms, budget, and admission scoping.

Everything the bounded-risk experiment is not allowed to decide at run time
lives here, so the runner reads it rather than restating it.  See
``docs/P4B_BOUNDED_RISK_GATE_PREREGISTRATION_2026-08-31.md``.

Nothing in this module reads data, calls a provider, or writes an artifact.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

HORIZON = forecast_p1.HORIZON            # 48
PERIOD = forecast_p1.PERIOD              # 24
CONTEXT_LENGTH = 192                     # the recent window window_context reports on

# Origins are spaced one context window plus one horizon apart, so origin o's
# context [o-192, o) begins exactly where the previous origin's evaluation
# window [o-240, o-192) ended.  The old P4 spacing of 48 overlapped 75% of the
# context between neighbours, which is why its 24 readings were not 24 samples.
SPACING = CONTEXT_LENGTH + HORIZON       # 240
ISOLATION = CONTEXT_LENGTH               # 192

OLD_P4_ORIGINS = (600, 648, 696, 744, 792, 840, 888, 936)
OLD_P4_EVAL_END = OLD_P4_ORIGINS[-1] + HORIZON          # 984
ORIGINS_PER_BLOCK = 8

# Origins are searched on the horizon grid and accepted greedily, subject to a
# minimum spacing of one context plus one horizon.  A fixed arithmetic block
# cannot be used: some KDD context windows are flat enough that the Consumer's
# robust scale collapses to its floor and the window is simply not evaluable,
# so the plan has to skip those and take the next admissible origin.
ORIGIN_GRID_STEP = HORIZON                              # 48
MIN_SPACING = SPACING                                   # 240
HELD_IN_SEARCH_START = OLD_P4_EVAL_END + ISOLATION      # 1176
MAX_ORIGIN = 10_850                                     # geometric ceiling


def resolve_block(
    start: int, count: int, is_viable: Any, *, maximum: int = MAX_ORIGIN
) -> tuple[int, ...]:
    """Earliest ``count`` viable origins from ``start``, spaced >= MIN_SPACING.

    ``is_viable`` is a structural predicate on the pre-origin context window --
    it reads observed history only, never a Consumer outcome -- so screening
    with it is the same kind of act as checking the geometry.
    """
    chosen: list[int] = []
    origin = int(start)
    while len(chosen) < count and origin <= maximum:
        if (not chosen or origin - chosen[-1] >= MIN_SPACING) and is_viable(origin):
            chosen.append(origin)
        origin += ORIGIN_GRID_STEP
    if len(chosen) < count:
        raise ValueError(
            "only %d viable origins from %d; %d were required"
            % (len(chosen), start, count)
        )
    return tuple(chosen)


def resolve_origins(is_viable: Any) -> dict[str, Any]:
    """The frozen origin plan, derived rather than written down."""
    held_in = resolve_block(HELD_IN_SEARCH_START, ORIGINS_PER_BLOCK, is_viable)
    held_in_eval_end = held_in[-1] + HORIZON
    held_out = resolve_block(
        held_in_eval_end + ISOLATION, ORIGINS_PER_BLOCK, is_viable
    )
    return {
        "rule": (
            "greedy on the %d-step grid, minimum spacing %d, skipping origins "
            "whose context window is not evaluable" % (ORIGIN_GRID_STEP, MIN_SPACING)
        ),
        "viability_screen": (
            "structural: the Consumer's robust scale on [origin-%d, origin) must "
            "not collapse to its floor for any evaluation series; reads observed "
            "history only, no Outcome" % CONTEXT_LENGTH
        ),
        "held_in_origins": list(held_in),
        "held_in_eval_end": held_in_eval_end,
        "held_out_origins": list(held_out),
        "held_out_eval_end": held_out[-1] + HORIZON,
        "held_in_spacings": [b - a for a, b in zip(held_in, held_in[1:])],
        "held_out_spacings": [b - a for a, b in zip(held_out, held_out[1:])],
        "replica_orders": {
            name: list(order) for name, order in replica_orders(held_in).items()
        },
        "outcome_read_during_selection": False,
    }


def replica_orders(held_in: Sequence[int]) -> dict[str, tuple[int, ...]]:
    """Three visit orders over the same held-in origins."""
    forward = tuple(int(origin) for origin in held_in)
    half = len(forward) // 2
    interleaved = tuple(
        value
        for pair in zip(forward[:half], forward[half:])
        for value in pair
    )
    return {
        "Forward": forward,
        "Reverse": tuple(reversed(forward)),
        "Interleaved": interleaved,
    }

# The frozen budget, calibrated on the eight old origins and scoped to this
# Consumer and metric only (preregistration section 1.1).
BOUNDED_MAX_HARMED_FRACTION = 0.20
BOUNDED_MAX_SINGLE_SERIES_HARM = 0.30
# Conservative sensitivity point.  Re-read offline from the collected
# per-series log; never a second tuning run.
SENSITIVITY_MAX_HARMED_FRACTION = 0.05
SENSITIVITY_MAX_SINGLE_SERIES_HARM = 0.10

STRICT_POLICY = admission_policy.DEFAULT
BOUNDED_POLICY = admission_policy.AdmissionPolicy(
    rule=admission_policy.BOUNDED_V1,
    max_harmed_fraction=BOUNDED_MAX_HARMED_FRACTION,
    max_single_series_harm=BOUNDED_MAX_SINGLE_SERIES_HARM,
)


# What this experiment is, and -- just as bindingly -- what it is not.
#
# The audited Source card installed in ``shared_initial`` carries an
# ``observable_applicability`` that matches no KDD origin in this study: 0 of 8
# held-in, 0 of 8 held-out, and 0 of 8 old-P4 origins.  The card is behaving
# correctly -- it stays silent where it does not apply -- but it means the
# cross-domain accumulation *treatment is empty here*, so no arm contrast can
# measure accumulation benefit.  Rather than swap in a different card (which
# would fold a Source-card selection experiment into a risk-gate experiment, and
# re-introduce exactly the confound this study is meant to avoid), the study is
# narrowed: it tests only what the bounded-risk gate does to Target-local
# adaptation, Skill formation, and held-out performance.
EXPERIMENT_QUESTION = (
    "does the bounded-risk admission gate change Target-local adaptation, Skill "
    "formation, and held-out utility, relative to the strict 20/20 gate?"
)
NOT_TESTED_HERE = (
    "cross-domain accumulation benefit: both online arms carry the same audited "
    "Source card and it is applicable at none of this study's origins, so the "
    "accumulation treatment is empty and no arm difference could express it"
)
SOURCE_TREATMENT_ACTIVE = False
# The gate a genuine A5 accumulation study must pass before it may run: at least
# one pre-audited Source Skill must match the deployment-visible context of at
# least one held-in origin.  Widening an existing card's Scope to satisfy this
# is not an admissible way to pass it.
SOURCE_SCOPE_MATCH_MINIMUM = 1


@dataclass(frozen=True)
class Arm:
    """One same-session arm.

    ``snapshot_source`` is where it starts: ``h0`` is the public cold start,
    ``shared_initial`` is the audited accumulated knowledge both online arms
    share.  ``carries_state`` says whether its snapshot and Episodes thread
    across held-in origins.

    Both arms here start from ``shared_initial`` and differ only in the
    admission rule, because that is the only contrast this data can express:
    with the Source card inapplicable everywhere (see ``NOT_TESTED_HERE``), an
    ``h0`` arm and a ``shared_initial`` arm would differ only by a store entry
    that never fires.
    """

    name: str
    snapshot_source: str
    carries_state: bool
    bounded: bool
    held_out_role: str


ARMS: tuple[Arm, ...] = (
    Arm("A5-strict", "shared_initial", True, False, "primary"),
    Arm("A5-bounded", "shared_initial", True, True, "primary"),
)
ARMS_BY_NAME: Mapping[str, Arm] = {arm.name: arm for arm in ARMS}
ADAPTIVE_ARM_NAMES = tuple(arm.name for arm in ARMS)
DETERMINISTIC_REFERENCES = ("Static", "Parallel Best-of-N@8")

# Per-cell budget, unchanged from P4 (B=8).
MAX_SUPPORT_A = 7
MAX_SUPPORT_B = 1
MAX_FULL_SUPPORT_EVALUATIONS = 8
MAX_CHEAP_PROBES = 24
MAX_LLM_CALLS = 8
MAX_TOKENS = 60_000
MAX_UPDATES = 1
MAX_WALL_SECONDS = 45 * 60

# Held-in is the only phase that spends LLM: held-out scoring is
# ``_frozen_recall`` plus one Consumer read, both deterministic.
REPLICA_COUNT = 3
HELD_IN_CELLS = len(ARMS) * ORIGINS_PER_BLOCK * REPLICA_COUNT
HELD_OUT_CELLS = HELD_IN_CELLS
GLOBAL_LLM_CALL_CAP = HELD_IN_CELLS * MAX_LLM_CALLS

# P4b does not exercise Slow-generated Patches, so the Slow Support gate is
# untouched and strict/bounded stay comparable on that path.
ALLOW_SLOW = False


@contextmanager
def admission_scope(arm: Arm | str) -> Iterator[Any]:
    """Install this arm's admission rule, and always put the default back.

    ``install_policy`` is process-global.  Without the ``finally`` an arm would
    inherit whatever the previous arm installed, and ``A5-strict`` would quietly
    stop being the old policy -- which is the one thing the strict/bounded
    contrast cannot survive.
    """
    resolved = ARMS_BY_NAME[arm] if isinstance(arm, str) else arm
    policy = BOUNDED_POLICY if resolved.bounded else STRICT_POLICY
    admission_policy.install_policy(policy)
    try:
        yield policy
    finally:
        admission_policy.reset_policy()


def geometry(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The resolved origin plan, with the isolation each block actually got."""
    held_in = list(plan["held_in_origins"])
    held_out = list(plan["held_out_origins"])
    return {
        "grid_step": ORIGIN_GRID_STEP,
        "minimum_spacing": MIN_SPACING,
        "minimum_spacing_rule": "CONTEXT_LENGTH + HORIZON",
        "old_p4_origins": list(OLD_P4_ORIGINS),
        "old_p4_eval_end": OLD_P4_EVAL_END,
        **{key: plan[key] for key in (
            "rule", "viability_screen", "held_in_origins", "held_in_eval_end",
            "held_out_origins", "held_out_eval_end", "held_in_spacings",
            "held_out_spacings", "outcome_read_during_selection",
        )},
        # Where each block's earliest context window opens.  It must land at or
        # after the previous block's evaluation end; landing exactly on it
        # (slack 0) is the intended tight packing, not a missing isolation band.
        "held_in_context_opens_at": held_in[0] - CONTEXT_LENGTH,
        "held_out_context_opens_at": held_out[0] - CONTEXT_LENGTH,
        "slack_beyond_isolation_old_to_held_in": (
            held_in[0] - CONTEXT_LENGTH - OLD_P4_EVAL_END
        ),
        "slack_beyond_isolation_held_in_to_held_out": (
            held_out[0] - CONTEXT_LENGTH - plan["held_in_eval_end"]
        ),
    }


def validate_geometry(
    plan: Mapping[str, Any], minimum_series_length: int | None = None
) -> list[str]:
    """Mechanical checks on a resolved plan.  Empty list means it holds."""
    failures: list[str] = []
    held_in = tuple(plan["held_in_origins"])
    held_out = tuple(plan["held_out_origins"])
    for name, origins in (("held_in", held_in), ("held_out", held_out)):
        if len(set(origins)) != ORIGINS_PER_BLOCK:
            failures.append("%s block does not hold %d distinct origins"
                            % (name, ORIGINS_PER_BLOCK))
        # Even spacing is not required; non-overlap is.  A gap larger than the
        # minimum only means the screen skipped an unevaluable origin.
        for earlier, later in zip(origins, origins[1:]):
            if later - earlier < MIN_SPACING:
                failures.append(
                    "%s origins %d and %d are closer than %d"
                    % (name, earlier, later, MIN_SPACING))
        if any(origin % ORIGIN_GRID_STEP != held_in[0] % ORIGIN_GRID_STEP
               for origin in origins):
            failures.append("%s block left the %d-step grid"
                            % (name, ORIGIN_GRID_STEP))
    if held_in[0] - CONTEXT_LENGTH < OLD_P4_EVAL_END:
        failures.append("held-in context overlaps the old P4 evaluation region")
    if held_out[0] - CONTEXT_LENGTH < plan["held_in_eval_end"]:
        failures.append("held-out context overlaps the held-in evaluation region")
    if set(held_in) & set(held_out):
        failures.append("held-in and held-out share an origin")
    if set(held_in) & set(OLD_P4_ORIGINS):
        failures.append("held-in reuses an old P4 origin")
    if set(held_out) & set(OLD_P4_ORIGINS):
        failures.append("held-out reuses an old P4 origin")
    for name, order in replica_orders(held_in).items():
        if sorted(order) != sorted(held_in):
            failures.append("replica order %s is not a permutation of held-in" % name)
    if (minimum_series_length is not None
            and plan["held_out_eval_end"] > minimum_series_length):
        failures.append(
            "held-out evaluation end %d exceeds the shortest series (%d)"
            % (plan["held_out_eval_end"], minimum_series_length))
    return failures


def contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The whole frozen contract, for the runner to copy into its receipt."""
    return {
        "task": "forecast",
        "consumer": "pooled_ridge_a1",
        "primary_metric": "sMASE",
        "preregistration": (
            "docs/P4B_BOUNDED_RISK_GATE_PREREGISTRATION_2026-08-31.md"
        ),
        "does_not_supersede": (
            "artifacts/main_protocol/p4_forecast_performance_b8_llm8_run2_20260830.json"
        ),
        "experiment_label": "PROSPECTIVE_RISK_UTILITY_POLICY_EXPERIMENT",
        "question": EXPERIMENT_QUESTION,
        "not_tested_here": NOT_TESTED_HERE,
        "source_treatment": {
            "active": SOURCE_TREATMENT_ACTIVE,
            "reading": (
                "the audited Source card is installed in both arms and is "
                "applicable at no origin in this study; it is correctly silent, "
                "and the accumulation treatment is therefore empty"
            ),
            "arm_names_are_historical": (
                "the arms are coded A5-strict / A5-bounded for continuity with "
                "P2-P4; read them as Online-strict / Online-bounded, since "
                "neither arm's Source knowledge activates here"
            ),
            "scope_match_gate_for_a_real_a5_study": {
                "requirement": (
                    "at least %d pre-audited Source Skill must match the "
                    "deployment-visible context of at least one held-in origin"
                    % SOURCE_SCOPE_MATCH_MINIMUM
                ),
                "inadmissible_way_to_pass": (
                    "widening an existing card's observable_applicability so it "
                    "matches; that manufactures the treatment instead of finding it"
                ),
            },
        },
        "geometry": geometry(plan),
        "arms": [
            {
                "name": arm.name,
                "snapshot_source": arm.snapshot_source,
                "carries_state_across_origins": arm.carries_state,
                "admission_rule": (
                    BOUNDED_POLICY if arm.bounded else STRICT_POLICY
                ).to_dict(),
                "held_out_role": arm.held_out_role,
            }
            for arm in ARMS
        ],
        "deterministic_references": list(DETERMINISTIC_REFERENCES),
        "bounded_budget": {
            "max_harmed_fraction": BOUNDED_MAX_HARMED_FRACTION,
            "max_single_series_harm": BOUNDED_MAX_SINGLE_SERIES_HARM,
            "scope": "pooled_ridge_a1 + sMASE only; does not transfer",
        },
        "offline_sensitivity_point": {
            "max_harmed_fraction": SENSITIVITY_MAX_HARMED_FRACTION,
            "max_single_series_harm": SENSITIVITY_MAX_SINGLE_SERIES_HARM,
            "method": "offline re-read of the collected per-series log; no new run",
        },
        "per_cell_budget": {
            "full_support_evaluations": MAX_FULL_SUPPORT_EVALUATIONS,
            "support_a_max": MAX_SUPPORT_A,
            "support_b_max": MAX_SUPPORT_B,
            "cheap_probe_max": MAX_CHEAP_PROBES,
            "llm_call_max": MAX_LLM_CALLS,
            "token_max": MAX_TOKENS,
            "accepted_update_max": MAX_UPDATES,
            "wall_seconds_max": MAX_WALL_SECONDS,
        },
        "cells": {
            "held_in": HELD_IN_CELLS,
            "held_out": HELD_OUT_CELLS,
            "held_out_spends_llm": False,
            "global_llm_call_cap": GLOBAL_LLM_CALL_CAP,
        },
        "allow_slow": ALLOW_SLOW,
        "statistics": {
            "unit": "origin",
            "aggregation": "mean over the 3 replicas within an origin, then paired",
            "n": ORIGINS_PER_BLOCK,
            "test": "paired Wilcoxon signed-rank, two-sided, alpha=0.05",
            "interval": "BCa bootstrap 95% CI, 10000 resamples, clustered on origin",
            "power_note": (
                "n=8: the smallest attainable two-sided p is 0.0078 and p<0.05 "
                "needs W<=3, so only a near-unanimous sign split can reach "
                "significance; a neutral verdict means undetectable here, not zero"
            ),
        },
    }
