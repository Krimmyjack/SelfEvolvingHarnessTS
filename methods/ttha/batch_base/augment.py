"""Augmentation primitives with explicit parameters (no module-level config). Draw order and
arithmetic are those of _scratch/ts_aug_source_grounding_pilot/aug_ops.py (freqmask / freqmix /
timemixup on the joint normalized [X;y] window, one RandomState per (entity, step)), plus the
same-clock-hour donor rule U of _scratch/ts_aug_donor_pattern_probe/dp_materials.py. With the
default parameters (w=0.25, mu=0.10, rule R) the outputs are bit-identical to those packages'
materials; batch_base_smoke.py checks this against the probe's frozen children.

Each step draws from RandomState(spec.aug_seed(entity_index, step_index)); donors always come from
the entity's ORIGINAL parent pool, never from a previously derived child.
"""
from __future__ import annotations

import numpy as np

from . import spec

N_BINS = (spec.L + spec.H) // 2 + 1


def derangement(n: int, rng: np.random.RandomState) -> np.ndarray:
    perm = rng.permutation(n)
    guard = 0
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
        guard += 1
        if guard > 1000:
            raise RuntimeError("derangement did not converge")
    return perm


def donors_R(rng: np.random.RandomState, n: int = spec.N_PARENTS) -> np.ndarray:
    return derangement(n, rng)


def donors_U(rng: np.random.RandomState, hours: np.ndarray) -> np.ndarray:
    n = hours.shape[0]
    donor = np.full(n, -1, dtype=np.int64)
    for h in range(24):
        members = np.flatnonzero(hours == h)
        if len(members) < 2:
            raise RuntimeError("INSTRUMENT_FAIL hour group %d has %d members" % (h, len(members)))
        perm = derangement(len(members), rng)
        donor[members] = members[perm]
    if (donor < 0).any():
        raise RuntimeError("INSTRUMENT_FAIL unassigned parents in rule U")
    return donor


def _donors(rule: str, rng: np.random.RandomState, hours: np.ndarray) -> np.ndarray:
    if rule == "R":
        return donors_R(rng, hours.shape[0])
    if rule == "U":
        return donors_U(rng, hours)
    raise ValueError("unknown donor rule %r" % rule)


def step_timemixup(current: np.ndarray, orig: np.ndarray, hours: np.ndarray, rng, donor_rule: str, w: float):
    donor = _donors(donor_rule, rng, hours)
    out = (1.0 - w) * current + w * orig[donor]
    return out, {"op": "timemixup", "donor_rule": donor_rule, "w": w, "donor_idx": donor.tolist()}


def step_freqmask(current: np.ndarray, rng, mu: float):
    thresh = rng.random_sample((current.shape[0], N_BINS)) < mu
    freq = np.fft.rfft(current, axis=-1)
    freq = np.where(thresh, 0.0 + 0.0j, freq)
    out = np.fft.irfft(freq, n=current.shape[-1], axis=-1)
    return out, {"op": "freqmask", "mu": mu, "n_masked_bins_mean": float(thresh.mean() * N_BINS), "masked_bins_by_parent": [np.flatnonzero(row).tolist() for row in thresh]}


def step_freqmix(current: np.ndarray, orig: np.ndarray, hours: np.ndarray, rng, donor_rule: str, mu: float):
    thresh = rng.random_sample((current.shape[0], N_BINS)) < mu          # drawn BEFORE the donor, as in a94
    donor = _donors(donor_rule, rng, hours)
    freq_p = np.fft.rfft(current, axis=-1)
    freq_d = np.fft.rfft(orig[donor], axis=-1)
    freq = np.where(thresh, freq_d, freq_p)
    out = np.fft.irfft(freq, n=current.shape[-1], axis=-1)
    return out, {"op": "freqmix", "donor_rule": donor_rule, "mu": mu, "donor_idx": donor.tolist(), "mixed_bins_by_parent": [np.flatnonzero(row).tolist() for row in thresh]}


def apply_steps(X_e: np.ndarray, y_e: np.ndarray, hours: np.ndarray, steps: list, entity_index: int):
    """steps: list of 0-2 dicts {op, ...params}. Returns (X_child, y_child, log) or (None, None, []) for identity."""
    if not steps:
        return None, None, []
    if len(steps) > spec.MAX_STEPS:
        raise ValueError("at most %d steps" % spec.MAX_STEPS)
    orig = np.concatenate([X_e, y_e], axis=-1)
    current = orig.copy()
    log = []
    for s, st in enumerate(steps):
        rng = np.random.RandomState(spec.aug_seed(entity_index, s))
        op = st["op"]
        if op == "timemixup":
            current, meta = step_timemixup(current, orig, hours, rng, st["donor_rule"], float(st["w"]))
        elif op == "freqmask":
            current, meta = step_freqmask(current, rng, float(st["mu"]))
        elif op == "freqmix":
            current, meta = step_freqmix(current, orig, hours, rng, st["donor_rule"], float(st["mu"]))
        else:
            raise ValueError("unknown op %r" % op)
        meta["step_index"] = s
        meta["seed"] = spec.aug_seed(entity_index, s)
        log.append(meta)
    return current[:, : spec.L], current[:, spec.L:], log
