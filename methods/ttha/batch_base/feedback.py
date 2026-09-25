"""Paired feedback between two complete materials (comparison doc §5 "反馈包含配对差、时间块与实体贡献").
d(seed) = loss(A) - loss(B); positive = B better. Per block: n, mean, sample SD, SE (null if n<2),
sign counts, df=n-1 Student-t 95% interval when n>=2 (descriptive; no significance claim), and the
working resolution delta = 1% of the reference (A) mean. Entity contributions: per-entity mean paired
difference, positive/negative top-2 shares, leave-one-out max shift -- diagnostics of the shared
model's prediction losses, never causal labels for any entity's own material.
"""
from __future__ import annotations

import math

import numpy as np

T_CRIT_95 = {2: 12.706204736, 3: 4.302652730, 4: 3.182446305, 5: 2.776445105, 6: 2.570581836, 8: 2.364624252}


def _scores(cells: list, block: str) -> dict:
    out = {}
    for c in cells:
        if c.get("status") == "OK" and block in c.get("scores", {}):
            out[int(c["model_seed"])] = c["scores"][block]
    return out


def paired(cells_a: list, cells_b: list, block: str = "c_a", label_a: str = "A", label_b: str = "B") -> dict:
    sa, sb = _scores(cells_a, block), _scores(cells_b, block)
    seeds = sorted(set(sa) & set(sb))
    d = [sa[s]["normalized_mse_macro"] - sb[s]["normalized_mse_macro"] for s in seeds]
    n = len(d)
    ref_mean = float(np.mean([sa[s]["normalized_mse_macro"] for s in seeds])) if seeds else None
    delta = 0.01 * ref_mean if ref_mean is not None else None
    out = {"A": label_a, "B": label_b, "block": block, "seeds": seeds, "n": n, "d_by_seed": d,
           "loss_A_by_seed": [sa[s]["normalized_mse_macro"] for s in seeds], "loss_B_by_seed": [sb[s]["normalized_mse_macro"] for s in seeds],
           "delta_1pct_of_A": delta, "convention": "d = loss(A) - loss(B); positive = B better"}
    if n == 0:
        out.update(mean=None, sd=None, se=None, pos=0, neg=0, zero=0, ci95=None, reading="n_insufficient")
        return out
    arr = np.array(d)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n >= 2 else None
    se = sd / math.sqrt(n) if sd is not None else None
    out.update(mean=mean, sd=sd, se=se, pos=int((arr > 0).sum()), neg=int((arr < 0).sum()), zero=int((arr == 0).sum()))
    if se is not None and n in T_CRIT_95:
        lo, hi = mean - T_CRIT_95[n] * se, mean + T_CRIT_95[n] * se
        out["ci95"] = [lo, hi]
        if lo > delta:
            out["reading"] = "candidate_improvement_of_B"
        elif hi < -delta:
            out["reading"] = "candidate_harm_of_B"
        elif lo >= -delta and hi <= delta:
            out["reading"] = "practically_close"
        else:
            out["reading"] = "uncertain"
    else:
        out["ci95"] = None
        out["reading"] = "n_insufficient"
    # entity contributions (shared-model prediction-loss decomposition; not causal labels)
    pe = [np.array(sa[s]["per_entity_normalized_mse"]) - np.array(sb[s]["per_entity_normalized_mse"]) for s in seeds]
    dv = np.mean(np.stack(pe), axis=0)
    pos, neg = dv[dv > 0], dv[dv < 0]
    order = np.argsort(-np.abs(dv))
    loo = [(dv.sum() - dv[i]) / (len(dv) - 1) for i in range(len(dv))]
    j = int(np.argmax(np.abs(np.array(loo) - mean)))
    out["entity_contributions"] = {
        "per_entity_mean_d": {"entity_%d" % i: float(dv[i]) for i in range(len(dv))},
        "n_pos": int(len(pos)), "n_neg": int(len(neg)),
        "pos_top2_share_of_pos_total": float(np.sort(pos)[::-1][:2].sum() / pos.sum()) if len(pos) else None,
        "neg_top2_share_of_neg_total": float(np.sort(neg)[:2].sum() / neg.sum()) if len(neg) else None,
        "top2_abs_entities": ["entity_%d" % i for i in order[:2]], "top2_abs_values": [float(dv[i]) for i in order[:2]],
        "loo_max_shift_entity": "entity_%d" % j, "loo_mean_without_it": float(loo[j]),
        "loo_sign_flips": bool(np.sign(loo[j]) != np.sign(mean) and mean != 0),
        "note": "prediction-loss decomposition of two shared models; an entity's value here is not the causal value of its own training material"}
    po = [np.array(sa[s]["per_origin_normalized_mse_mean"]) - np.array(sb[s]["per_origin_normalized_mse_mean"]) for s in seeds]
    out["per_origin_mean_d"] = np.mean(np.stack(po), axis=0).tolist()
    return out

