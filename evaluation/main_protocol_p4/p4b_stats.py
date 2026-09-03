"""P4b statistics: origin is the unit, not the reading.

Three replicas walk the same eight held-in origins and differ only in LLM
sampling, so 24 held-out readings per arm are 24 records of 8 samples.  The
frozen aggregation is therefore: average the replicas inside an origin, then
pair across origins.  That is n = 8, and the power that follows is stated
rather than worked around -- an exact signed-rank test on 8 pairs can only
reach p < 0.05 with a near-unanimous sign split.

The signed-rank p here is exact by enumeration (2^8 = 256 sign assignments),
not a normal approximation, because at this n the approximation is the wrong
tool and the exact value is cheap.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260831
ALPHA = 0.05


def by_origin(
    rows: Sequence[Mapping[str, Any]], *, arm: str, field: str
) -> dict[int, float]:
    """Mean of ``field`` over the replicas of each origin, for one arm."""
    buckets: dict[int, list[float]] = {}
    for row in rows:
        if str(row.get("arm")) != arm or row.get(field) is None:
            continue
        buckets.setdefault(int(row["origin"]), []).append(float(row[field]))
    return {origin: float(np.mean(values)) for origin, values in buckets.items()}


def paired(
    left: Mapping[int, float], right: Mapping[int, float]
) -> tuple[list[int], list[float]]:
    """Origins present on both sides, and left minus right on each."""
    origins = sorted(set(left) & set(right))
    return origins, [left[origin] - right[origin] for origin in origins]


def _signed_ranks(differences: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([d for d in differences if d != 0.0], dtype=np.float64)
    if values.size == 0:
        return values, values
    order = np.argsort(np.abs(values), kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    magnitudes = np.abs(values)[order]
    # Average ranks within ties, the standard signed-rank handling.
    position = 0
    rank_of_sorted = np.empty(values.size, dtype=np.float64)
    while position < magnitudes.size:
        stop = position
        while stop + 1 < magnitudes.size and magnitudes[stop + 1] == magnitudes[position]:
            stop += 1
        rank_of_sorted[position:stop + 1] = np.mean(
            np.arange(position + 1, stop + 2, dtype=np.float64)
        )
        position = stop + 1
    ranks[order] = rank_of_sorted
    return values, ranks


def wilcoxon_exact(differences: Sequence[float]) -> dict[str, Any]:
    """Two-sided exact signed-rank test.  Zero differences are dropped."""
    values, ranks = _signed_ranks(differences)
    n = int(values.size)
    if n == 0:
        return {
            "statistic": None, "p_two_sided": None, "n_pairs": len(differences),
            "n_nonzero": 0,
            "note": "every paired difference was exactly zero; no test is defined",
        }
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    statistic = min(positive, negative)
    # Exact null: every sign assignment is equally likely.
    total = 0
    at_least_as_extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        signed = np.asarray(signs) * ranks
        candidate = min(float(signed[signed > 0].sum()),
                        float(-signed[signed < 0].sum()))
        total += 1
        if candidate <= statistic + 1e-12:
            at_least_as_extreme += 1
    return {
        "statistic": statistic,
        "p_two_sided": at_least_as_extreme / total,
        "n_pairs": len(differences),
        "n_nonzero": n,
        "positive_rank_sum": positive,
        "negative_rank_sum": negative,
        "method": "exact enumeration over 2^n sign assignments",
    }


def bca_interval(
    differences: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """BCa CI for the mean paired difference, resampling whole origins."""
    values = np.asarray(list(differences), dtype=np.float64)
    n = int(values.size)
    observed = float(values.mean()) if n else None
    if n < 2 or float(values.std()) == 0.0:
        return {
            "point_estimate": observed, "low": observed, "high": observed,
            "resamples": 0, "alpha": alpha,
            "note": "degenerate sample; interval collapses to the point estimate",
        }
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    proportion_below = float((draws < observed).mean())
    proportion_below = min(max(proportion_below, 1.0 / resamples),
                           1.0 - 1.0 / resamples)
    from statistics import NormalDist

    normal = NormalDist()
    z0 = normal.inv_cdf(proportion_below)
    # Jackknife acceleration over origins.
    jackknife = np.asarray(
        [np.delete(values, index).mean() for index in range(n)], dtype=np.float64
    )
    centred = jackknife.mean() - jackknife
    denominator = 6.0 * float((centred ** 2).sum()) ** 1.5
    acceleration = (
        float((centred ** 3).sum()) / denominator if denominator else 0.0
    )

    def endpoint(probability: float) -> float:
        z = normal.inv_cdf(probability)
        adjusted = z0 + (z0 + z) / (1.0 - acceleration * (z0 + z))
        return float(np.percentile(draws, 100.0 * normal.cdf(adjusted)))

    return {
        "point_estimate": observed,
        "low": endpoint(alpha / 2.0),
        "high": endpoint(1.0 - alpha / 2.0),
        "resamples": resamples,
        "alpha": alpha,
        "bias_correction_z0": z0,
        "acceleration": acceleration,
        "resample_unit": "origin",
    }


def contrast(
    rows: Sequence[Mapping[str, Any]], *, left: str, right: str, field: str
) -> dict[str, Any]:
    """One paired contrast on origin means, with an exact test and a BCa CI."""
    left_means = by_origin(rows, arm=left, field=field)
    right_means = by_origin(rows, arm=right, field=field)
    origins, differences = paired(left_means, right_means)
    return {
        "left": left,
        "right": right,
        "field": field,
        "unit": "origin",
        "origins": origins,
        "left_origin_means": [left_means[o] for o in origins],
        "right_origin_means": [right_means[o] for o in origins],
        "paired_differences": differences,
        "mean_difference": float(np.mean(differences)) if differences else None,
        "median_difference": float(np.median(differences)) if differences else None,
        "positive_origins": sum(1 for d in differences if d > 0),
        "negative_origins": sum(1 for d in differences if d < 0),
        "wilcoxon": wilcoxon_exact(differences),
        "bca_95": bca_interval(differences),
    }


def power_note(n: int) -> dict[str, Any]:
    """What this n can and cannot detect, stated before the data opens."""
    smallest = 2.0 / (2.0 ** n) if n else None
    return {
        "n": n,
        "smallest_attainable_two_sided_p": smallest,
        "reaches_alpha_05": bool(smallest is not None and smallest < ALPHA),
        "reading": (
            "only a near-unanimous sign split can reach significance at this n; "
            "a neutral verdict means undetectable by this design, not zero"
        ),
    }


def primary_verdict(
    *,
    utility: Mapping[str, Any],
    held_out_worst_single_series_harm: float | None,
    held_out_harm_rate: float | None,
    max_single_series_harm: float,
    max_harmed_fraction: float,
    any_admission_held_in: bool,
    active_skills_formed: int,
    causal_reuse_observed: bool,
    writeback_gated: bool,
    parallel_selection_face: str,
) -> dict[str, Any]:
    """The frozen verdict ladder.  Blocking conditions are checked first.

    Support-A admission is provisional, not deployment: full Target-local
    admission requires **both** faces.  So a run where Support-A admitted
    candidates and Support-B approved none has not produced a neutral result --
    it has been blocked, at a nameable face.  ``BOUNDED_GATE_NEUTRAL`` is
    reserved for a run that formed at least one Active Skill and carried it
    through a completed held-out phase; anything less has no endpoint to be
    neutral about.
    """
    if not writeback_gated:
        return {"verdict": "LEAKAGE_SUSPECTED", "blocking": True,
                "reading": "the Skill store moved on a cell where the admission "
                           "gate admitted nothing"}
    if parallel_selection_face != "held_in":
        return {"verdict": "HELDOUT_CONTAMINATED", "blocking": True,
                "reading": "a comparator selected on the endpoint face"}
    if not any_admission_held_in:
        return {"verdict": "BOUNDED_GATE_STILL_BLOCKING", "blocking": True,
                "blocking_face": "SUPPORT_A",
                "reading": "no candidate was admitted at Support-A anywhere in "
                           "held-in"}
    if int(active_skills_formed) <= 0:
        return {"verdict": "BOUNDED_GATE_STILL_BLOCKING", "blocking": True,
                "blocking_face": "SUPPORT_B",
                "reading": "Support-A admitted candidates but the independent "
                           "Support-B confirmation approved none, so no Skill "
                           "was formed and nothing reached a deployable state"}
    # An empty contrast is not a neutral result.  Without this the ladder falls
    # through to BOUNDED_GATE_NEUTRAL and reports "no detectable difference"
    # about a comparison that was never made -- which is how a rehearsal with no
    # endpoint phase came back reading like a finding.
    if not (utility.get("origins") or ()):
        return {"verdict": "NO_ENDPOINT_DATA", "blocking": True,
                "reading": "the primary contrast has no paired origin; nothing "
                           "was measured on the endpoint face"}
    breached = []
    if (held_out_worst_single_series_harm is not None
            and held_out_worst_single_series_harm > max_single_series_harm):
        breached.append("worst single-series harm %.4f > %.2f" % (
            held_out_worst_single_series_harm, max_single_series_harm))
    if (held_out_harm_rate is not None and held_out_harm_rate > max_harmed_fraction):
        breached.append("harm rate %.4f > %.2f" % (
            held_out_harm_rate, max_harmed_fraction))
    if breached:
        return {"verdict": "RISK_BUDGET_BREACHED", "blocking": True,
                "reading": "; ".join(breached)}
    p = (utility.get("wilcoxon") or {}).get("p_two_sided")
    mean = utility.get("mean_difference")
    significant = p is not None and p < ALPHA
    if significant and mean is not None and mean > 0:
        verdict = "BOUNDED_GATE_POSITIVE"
        reading = "bounded beats strict on held-out utility with no risk breach"
    elif significant and mean is not None and mean < 0:
        verdict = "BOUNDED_GATE_NEGATIVE"
        reading = "bounded is worse than strict on held-out utility"
    else:
        verdict = "BOUNDED_GATE_NEUTRAL"
        reading = "no detectable difference at n=8; not evidence of no effect"
    return {
        "verdict": verdict,
        "blocking": False,
        "reading": reading,
        "causal_reuse_observed": bool(causal_reuse_observed),
        "secondary_note": (
            None if causal_reuse_observed
            else "no causal reuse: the mechanism claim is NO_CAUSAL_REUSE and "
                 "the primary result stands on its own"
        ),
    }
