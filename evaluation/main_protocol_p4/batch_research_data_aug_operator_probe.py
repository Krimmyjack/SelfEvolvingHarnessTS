"""DEV-DATA-AUG-OPERATOR-PROBE: bounded 0-LLM augmentation development probe (user decision 2026-09-18; METHOD.md in ROOT).

Question: on natural readiness data, does choosing an augmentation from a larger (TempoPFN-inspired) operator pool by C_A deliver
more than a fixed default, random choice or short training -- measured as actual delivered gain, selection regret and cost.
  4 exposed dev jobs (RD01B_Q1, RD02_T1, RD01B_STP1, RD02_STP1) x 24 materials (None, 3 old ops, 20 new single ops) x 3 seeds;
  one 2000-update trajectory per cell with the 100-update checkpoint scored as the short-training Consumer.
  Serving is Baseline-Linear for every model; augmentation only adds a child view (parent 0.5 + child 0.5, existing train_arm).
Order: preflight -> children -> fits (C_A only) -> checks -> selections frozen -> C_B -> E frozen -> barrier -> E -> readout.
The controller never imports torch; each fit / label stage runs in its own subprocess.
  --smoke | --preflight | --run | --result      (workers: --worker-children JOB | --worker-fit JOB MID SEED [--tag T] |
                                                  --worker-label STAGE JOB | --worker-compare A B)
"""
from __future__ import annotations

import os

KMP_AT_START = os.environ.get('KMP_DUPLICATE_LIB_OK')

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import augment, context, spec
from methods.ttha.batch_base import readiness as rd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_data_aug_operator_probe'
RUN = ROOT / 'jobs'
TL = REPO / '_scratch' / 'dev_data_readiness_training_length'
MODULE = 'evaluation.main_protocol_p4.batch_research_data_aug_operator_probe'
PRSA_CLOSED_FROM_ROW = 24864
JOBS = ('RD01B_Q1', 'RD02_T1', 'RD01B_STP1', 'RD02_STP1')
DOMAIN = {'RD01B_Q1': 'RD01B', 'RD01B_STP1': 'RD01B', 'RD02_T1': 'RD02', 'RD02_STP1': 'RD02'}
SEEDS = {'RD01B_Q1': (20261001, 20261002, 20261003), 'RD02_T1': (20261001, 20261002, 20261003),
         'RD01B_STP1': (20261011, 20261012, 20261013), 'RD02_STP1': (20261011, 20261012, 20261013)}
CACHE = {'RD01B_Q1': TL / 'dev' / 'RD01B_Q1', 'RD02_T1': TL / 'dev' / 'RD02_T1',
         'RD01B_STP1': TL / 'follow' / 'RD01B_STP1', 'RD02_STP1': TL / 'follow' / 'RD02_STP1'}
N_FULL, N_SHORT = 2000, 100
CONSUMERS = (N_FULL, N_SHORT)
AUG_SEED_BASE = 918_000_000                      # RandomState(918000000 + 1000 * material_no + entity)
TIE_TOL = 1e-12

# (material_id, family, op, params). material_no = position (None = 0). One operator step per material (no combinations this round).
MATERIALS = (
    ('None', 'none', None, {}),
    ('FixedMixup', 'old', 'timemixup', {'donor_rule': 'R', 'w': 0.25}),
    ('FreqMask10', 'old', 'freqmask', {'mu': 0.10}),
    ('FreqMixR10', 'old', 'freqmix', {'donor_rule': 'R', 'mu': 0.10}),
    ('AmpMod30', 'new', 'amp_mod', {'a': 0.3}),
    ('AmpMod60', 'new', 'amp_mod', {'a': 0.6}),
    ('Shift25', 'new', 'regime_shift', {'s': 0.25}),
    ('Shift50', 'new', 'regime_shift', {'s': 0.5}),
    ('Shock10', 'new', 'shock', {'A': 1.0}),
    ('Shock20', 'new', 'shock', {'A': 2.0}),
    ('DayLvl05', 'new', 'day_level', {'d': 0.5}),
    ('DayLvl10', 'new', 'day_level', {'d': 1.0}),
    ('Mix3', 'new', 'mixup_multi', {}),
    ('MixCross40', 'new', 'mixup_cross', {'w_max': 0.4}),
    ('MixPath50', 'new', 'mixup_path', {'w_max': 0.5}),
    ('Resamp2', 'new', 'resample', {'k': 2}),
    ('Resamp4', 'new', 'resample', {'k': 4}),
    ('Conv3', 'new', 'conv', {'L': 3}),
    ('Conv7', 'new', 'conv', {'L': 7}),
    ('Censor95', 'new', 'censor', {'q': 0.95}),
    ('Censor90', 'new', 'censor', {'q': 0.90}),
    ('Noise05', 'new', 'noise', {'sigma': 0.05}),
    ('Noise15', 'new', 'noise', {'sigma': 0.15}),
    ('OU30', 'new', 'ou_mix', {'w': 0.3}),
)
MIDS = tuple(m[0] for m in MATERIALS)
MAT = {m[0]: {'no': i, 'family': m[1], 'op': m[2], 'params': m[3]} for i, m in enumerate(MATERIALS)}
OLD_POOL = tuple(m for m in MIDS if MAT[m]['family'] in ('none', 'old'))
BUDGET2_FIXED = ('None', 'FixedMixup')
BUDGET2_DRAW = tuple(m for m in MIDS if m not in BUDGET2_FIXED)
CAPS = {'fit': len(JOBS) * len(MIDS) * 3, 'check': 1, 'retry': 6}
CAPS['hard'] = CAPS['fit'] + CAPS['check'] + CAPS['retry']
WALL_NUMERIC_S = 4.0 * 3600                      # amendment 1: was 2.5 h; external CPU contention (METHOD.md)
FIT_TIMEOUT_S = 300.
OP_TEXT = {
    'timemixup': 'old: (1-w) z + w z[donor], donor = derangement of the entity\'s own legal parents (augment.step_timemixup, rule R)',
    'freqmask': 'old: zero each rFFT bin of the 240-point joint window with prob mu (augment.step_freqmask)',
    'freqmix': 'old: replace each rFFT bin with the donor\'s with prob mu, donor rule R (augment.step_freqmix)',
    'amp_mod': 'TempoPFN amplitude modulation: one random segment (24-96 points) scaled around its own mean by g ~ U(1-a, 1+a)',
    'regime_shift': 'TempoPFN regime change (level part): from one change point c ~ U{24..216} add delta ~ N(0, s^2)',
    'shock': 'TempoPFN shock & recovery: add +/- A*U(0.5,1.5)*exp(-(t-t0)/tau), t0 ~ U{0..239}, tau ~ U(3,24) hours',
    'day_level': 'TempoPFN calendar effect: one whole clock day (hour 0..23) inside the window shifted by +/- d*U(0.5,1), 3-point linear edge ramps',
    'mixup_multi': 'TempoPFN TS-mixup: w0 z + w1 z[d1] + w2 z[d2], w0 ~ U(0.5,1), (w1,w2) = (1-w0) Dirichlet(1,1), two own-entity derangements',
    'mixup_cross': 'TS-mixup across series: (1-w) z + w z_other, other = random other entity\'s legal window at the SAME start (same clock time), w ~ U(0, w_max); own-entity donor if none (counted)',
    'mixup_path': 'TempoPFN time-dependent mixup: w(t) linear from w_a to w_b, both ~ U(0, w_max), own-entity derangement donor',
    'resample': 'TempoPFN resampling artifact: keep every k-th point (random phase), linear interpolation back to 240 points',
    'conv': 'TempoPFN stochastic convolution: 1-3 random positive kernels (softmax of N(0,1), length L), reflect padding',
    'censor': 'TempoPFN censoring: clip the window to its own [1-q, q] quantiles',
    'noise': 'TempoPFN finishing noise: add N(0, sigma^2) (normalized units)',
    'ou_mix': 'TempoPFN SDE generator (OU, theta ~ U(0.05,0.5)) standardized, rescaled to the window mean/std, mixed with weight w',
}
EXCLUDED = {'time_reversal': 'breaks causal/diurnal direction for in-domain forecasting', 'sign_inversion': 'impossible negative loads / concentrations',
            'differential_operators': 'changes the signal type', 'numerical_integration': 'changes the signal type',
            'quantization': 'left out to keep the round at 20 new variants'}


def ds_of(job: str) -> str:
    return job.split('_')[0]


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def cid(job: str, mid: str, seed: int, n: int, tag: str = '') -> str:
    return '%s__%s__s%d__u%d%s' % (job, mid, seed, n, ('__' + tag) if tag else '')


# ============================================================================= augmentation operators (numpy only)
def _derange(n: int, rng) -> np.ndarray:
    return augment.derangement(n, rng)


def _softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def op_amp_mod(Z, rng, a):
    out = Z.copy()
    for i in range(Z.shape[0]):
        L = int(rng.randint(24, 97))
        s = int(rng.randint(0, Z.shape[1] - L + 1))
        g = rng.uniform(1 - a, 1 + a)
        seg = out[i, s:s + L]
        m = seg.mean()
        out[i, s:s + L] = m + g * (seg - m)
    return out, {}


def op_regime_shift(Z, rng, s):
    out = Z.copy()
    for i in range(Z.shape[0]):
        c = int(rng.randint(24, 217))
        out[i, c:] += rng.normal(0.0, s)
    return out, {}


def op_shock(Z, rng, A):
    out = Z.copy()
    T = Z.shape[1]
    for i in range(Z.shape[0]):
        t0 = int(rng.randint(0, T))
        tau = rng.uniform(3.0, 24.0)
        sign = 1.0 if rng.randint(2) else -1.0
        amp = A * rng.uniform(0.5, 1.5)
        out[i, t0:] += sign * amp * np.exp(-np.arange(T - t0) / tau)
    return out, {}


RAMP = np.array([0.25, 0.5, 0.75] + [1.0] * 18 + [0.75, 0.5, 0.25])


def op_day_level(Z, rng, d, hours):
    out = Z.copy()
    T = Z.shape[1]
    for i in range(Z.shape[0]):
        starts = [p for p in np.flatnonzero(hours[i] == 0) if p + 24 <= T]
        if not starts:
            raise RuntimeError('no whole clock day inside a 240-point window')
        p = int(starts[int(rng.randint(len(starts)))])
        delta = (1.0 if rng.randint(2) else -1.0) * d * rng.uniform(0.5, 1.0)
        out[i, p:p + 24] += delta * RAMP
    return out, {}


def op_mixup_multi(Z, rng):
    n = Z.shape[0]
    d1, d2 = _derange(n, rng), _derange(n, rng)
    w0 = rng.uniform(0.5, 1.0, size=n)
    u = rng.dirichlet([1.0, 1.0], size=n)
    w1, w2 = (1 - w0) * u[:, 0], (1 - w0) * u[:, 1]
    return w0[:, None] * Z + w1[:, None] * Z[d1] + w2[:, None] * Z[d2], {}


def op_mixup_path(Z, rng, w_max):
    n, T = Z.shape
    d = _derange(n, rng)
    wa, wb = rng.uniform(0, w_max, size=n), rng.uniform(0, w_max, size=n)
    w = wa[:, None] + (wb - wa)[:, None] * (np.arange(T)[None, :] / (T - 1))
    return (1 - w) * Z + w * Z[d], {}


def op_mixup_cross(Z, rng, w_max, e, k, others):
    """others: {entity: (Z_other, {start_k: row})} for every OTHER entity."""
    out = Z.copy()
    fallback = 0
    own = None
    ents = sorted(others)
    for i in range(Z.shape[0]):
        cand = [o for o in ents if int(k[i]) in others[o][1]]
        w = rng.uniform(0, w_max)
        if cand:
            o = cand[int(rng.randint(len(cand)))]
            donor = others[o][0][others[o][1][int(k[i])]]
        else:
            if own is None:
                own = _derange(Z.shape[0], np.random.RandomState(AUG_SEED_BASE + 999_000 + e))
            donor = Z[own[i]]
            fallback += 1
        out[i] = (1 - w) * Z[i] + w * donor
    return out, {'own_entity_fallback_windows': fallback}


def op_resample(Z, rng, k):
    out = np.empty_like(Z)
    T = Z.shape[1]
    grid = np.arange(T)
    for i in range(Z.shape[0]):
        phase = int(rng.randint(0, k))
        idx = np.arange(phase, T, k)
        out[i] = np.interp(grid, idx, Z[i, idx])
    return out, {}


def op_conv(Z, rng, L):
    out = Z.copy()
    h = L // 2
    for i in range(Z.shape[0]):
        for _ in range(int(rng.randint(1, 4))):
            w = _softmax(rng.normal(0.0, 1.0, size=L))
            out[i] = np.convolve(np.pad(out[i], h, mode='reflect'), w[::-1], mode='valid')
    return out, {}


def op_censor(Z, rng, q):
    lo = np.quantile(Z, 1 - q, axis=1, keepdims=True)
    hi = np.quantile(Z, q, axis=1, keepdims=True)
    return np.clip(Z, lo, hi), {}


def op_noise(Z, rng, sigma):
    return Z + rng.normal(0.0, sigma, size=Z.shape), {}


def op_ou_mix(Z, rng, w):
    n, T = Z.shape
    out = Z.copy()
    for i in range(n):
        theta = rng.uniform(0.05, 0.5)
        eps = rng.normal(0.0, 1.0, size=T)
        s = np.empty(T)
        s[0] = eps[0] / math.sqrt(max(1 - (1 - theta) ** 2, 1e-12))
        for t in range(1, T):
            s[t] = (1 - theta) * s[t - 1] + eps[t]
        sd = s.std()
        zs = Z[i].std()
        if sd > 0 and zs > 0:
            out[i] = (1 - w) * Z[i] + w * (Z[i].mean() + zs * (s - s.mean()) / sd)
    return out, {}


def apply_material(mid: str, Z: np.ndarray, e: int, k: np.ndarray, hours: np.ndarray, others: dict) -> tuple:
    """Z: (n_e, 240) joint normalized [X;y] windows of entity e (legal order). Returns (child, meta)."""
    m = MAT[mid]
    if m['op'] is None:
        return None, {}
    rng = np.random.RandomState(AUG_SEED_BASE + 1000 * m['no'] + e)
    op, p = m['op'], m['params']
    if op == 'timemixup':
        out, meta = augment.step_timemixup(Z, Z, np.zeros(Z.shape[0]), rng, p['donor_rule'], p['w'])
        meta = {}
    elif op == 'freqmask':
        out, _ = augment.step_freqmask(Z, rng, p['mu'])
        meta = {}
    elif op == 'freqmix':
        out, _ = augment.step_freqmix(Z, Z, np.zeros(Z.shape[0]), rng, p['donor_rule'], p['mu'])
        meta = {}
    elif op == 'day_level':
        out, meta = op_day_level(Z, rng, p['d'], hours)
    elif op == 'mixup_cross':
        out, meta = op_mixup_cross(Z, rng, p['w_max'], e, k, others)
    else:
        out, meta = {'amp_mod': op_amp_mod, 'regime_shift': op_regime_shift, 'shock': op_shock, 'mixup_multi': op_mixup_multi,
                     'mixup_path': op_mixup_path, 'resample': op_resample, 'conv': op_conv, 'censor': op_censor, 'noise': op_noise,
                     'ou_mix': op_ou_mix}[op](Z, rng, **p)
    if out.shape != Z.shape or not np.isfinite(out).all():
        raise RuntimeError('operator %s produced a bad child for entity %d' % (op, e))
    return out, meta


# ============================================================================= workers (subprocess entries)
def job_arrays(ctx) -> tuple:
    """Joint normalized windows of the legal parents: X from the Baseline-Linear material, raw y; + clock hours of each window."""
    idx = rd.load_index(ctx.job_dir)
    ref = rd.MaterialRef(**idx['P0_Linear'])
    X, y = rd.training_arrays(ctx, ref)
    lg = ctx.legal
    t0 = ctx.job.train_range[0]
    hrs = ctx.slice.hours_of(*ctx.job.train_range)
    H = np.stack([hrs[lg.k[p]: lg.k[p] + spec.L + spec.H, lg.ent[p]] for p in range(lg.n)])
    assert t0 == ctx.job.train_range[0]
    return ref, X, y, H


def worker_children(job: str) -> None:
    ctx = rd.open_job(ds_of(job), job, RUN, stage='material')
    idx = rd.load_index(ctx.job_dir)
    if 'P0_Linear' not in idx:
        ov = ctx.overview()
        rd.build(ctx, rd.compile_policy(rd.BASELINE_LINEAR, ov['entities'])['assignment'], 'P0_Linear')
    ref, X, y, H = job_arrays(ctx)
    lg = ctx.legal
    Zall = np.concatenate([X, y], axis=1)
    rows_by_ent = {e: np.flatnonzero(lg.ent == e) for e in range(len(lg.counts))}
    kmap = {e: {int(lg.k[r]): j for j, r in enumerate(rows_by_ent[e])} for e in rows_by_ent}
    zent = {e: Zall[rows] for e, rows in rows_by_ent.items()}
    cdir = ctx.job_dir / 'children'
    cdir.mkdir(exist_ok=True)
    summary = {'job_id': job, 'legal_parents': int(lg.n), 'materials': {}, 'written_local': now()}
    for mid in MIDS:
        if MAT[mid]['op'] is None:
            continue
        t0 = time.time()
        C = np.empty_like(Zall)
        metas = {}
        for e, rows in rows_by_ent.items():
            others = {o: (zent[o], kmap[o]) for o in rows_by_ent if o != e} if MAT[mid]['op'] == 'mixup_cross' else {}
            out, meta = apply_material(mid, Zall[rows], e, lg.k[rows], H[rows], others)
            C[rows] = out
            for kk, v in meta.items():
                metas[kk] = metas.get(kk, 0) + v
        secs = time.time() - t0
        d = C - Zall
        np.savez(cdir / (mid + '.npz'), Xc=C[:, :spec.L].astype(np.float32), yc=C[:, spec.L:].astype(np.float32))
        summary['materials'][mid] = {'op': MAT[mid]['op'], 'params': MAT[mid]['params'], 'build_seconds': secs,
                                     'rms_change_norm': float(np.sqrt((d ** 2).mean())), 'rms_change_norm_X': float(np.sqrt((d[:, :spec.L] ** 2).mean())),
                                     'rms_change_norm_y': float(np.sqrt((d[:, spec.L:] ** 2).mean())),
                                     'fraction_points_changed': float((np.abs(d) > 1e-12).mean()), **metas}
        print('CHILD', job, mid, round(secs, 2), flush=True)
    context.write_json(ctx.job_dir / 'children_summary.json', summary)


def worker_fit(job: str, mid: str, seed: int, tag: str = '') -> None:
    rd._init_openmp_before_torch()
    from methods.ttha.batch_base import train
    import torch
    t0 = time.time()
    ctx = rd.open_job(ds_of(job), job, RUN, stage='evaluate')         # rows [t-672, t+96): no C_B / E row in memory
    ref, X, y, _ = job_arrays(ctx)
    lg, sc = ctx.legal, ctx.scaler
    child = None
    if MAT[mid]['op'] is not None:
        with np.load(ctx.job_dir / 'children' / (mid + '.npz')) as z:
            child = (z['Xc'].copy(), z['yc'].copy())
        if child[0].shape != X.shape or child[1].shape != y.shape:
            raise RuntimeError('child does not cover the legal parent set')
    for d in ('runs', 'cells', 'pred_c_a'):
        (ctx.job_dir / d).mkdir(exist_ok=True)
    states = {}

    def keep(step, model, batch_loss, seconds):
        states[step] = ({k: v.detach().clone() for k, v in model.state_dict().items()}, batch_loss, seconds)

    bidx = train.batch_indices(seed, n_pool=lg.n, n_updates=N_FULL)
    model, tres = train.train_arm(X, y, child, bidx, model_seed=seed, timeout_seconds=FIT_TIMEOUT_S - 30, n_updates=N_FULL,
                                  checkpoints=[N_SHORT], on_checkpoint=keep)
    Xs, serve = rd.serve_inputs(ctx.slice, ctx.job.c_a, ref.assignment, sc)
    truth = rd.read_truth(ctx.slice, ctx.job.c_a)
    short = train.make_model()().to(train.device())
    short.load_state_dict(states[N_SHORT][0])
    for n, m, secs, loss in ((N_SHORT, short, states[N_SHORT][2], states[N_SHORT][1]), (N_FULL, model, tres.seconds, tres.final_train_loss)):
        c = cid(job, mid, seed, n, tag)
        mp = ctx.job_dir / 'runs' / (c + '.pt')
        train.save_model(m, mp)
        pred = rd.predict_inputs(m, Xs, sc)
        np.savez(ctx.job_dir / 'pred_c_a' / (c + '.npz'), pred_raw=pred, origins=np.array(ctx.job.c_a))
        rec = {'cell_id': c, 'status': 'OK', 'job_id': job, 'material_id': mid, 'model_seed': seed, 'n_updates': n, 'tag': tag,
               'physical_fit': n == N_FULL, 'trajectory_updates': N_FULL, 'child_view': child is not None, 'pool_size': int(lg.n),
               'batch_stream': 'first n rows of train.batch_indices(seed, n_pool)', 'model_path': str(mp),
               'train': {'seconds_at_this_length': secs, 'batch_loss_at_this_length': loss, 'trajectory_seconds': tres.seconds,
                         'first_step_loss': tres.first_step_loss},
               'serve_inputs_c_a': serve, 'scores': {'c_a': rd.score_block(pred, truth, sc)},
               'device': str(train.device()), 'torch': torch.__version__, 'torch_threads': torch.get_num_threads(),
               'python': sys.version.split()[0], 'rows_read': list(ctx.job.evaluate_rows), 'worker_seconds_total': time.time() - t0}
        context.write_json(ctx.job_dir / 'cells' / (c + '.json'), rec)
    print('CELL_OK', cid(job, mid, seed, N_FULL, tag), round(tres.seconds, 2), flush=True)


def _cells(job_dir: Path) -> dict:
    out = {}
    for p in sorted((job_dir / 'cells').glob('*.json')):
        r = context.read_json(p)
        if r.get('status') == 'OK' and not r.get('tag'):
            out[r['cell_id']] = r
    return out


def label_prerequisite(stage: str, job: str) -> None:
    """Raises before any label row is loaded or torch is imported."""
    jd = RUN / job
    if stage == 'c_b' and not (ROOT / 'selections.json').exists():
        raise PermissionError('C_B refused: selections.json (frozen from C_A) missing')
    if stage == 'freeze_e' and not (jd / 'c_b_scores.json').exists():
        raise PermissionError('E inputs refused: C_B not scored')
    if stage == 'score_e':
        b = ROOT / 'all_e_predictions_frozen.json'
        if not b.exists() or job not in context.read_json(b)['jobs']:
            raise PermissionError('E targets refused: barrier missing or job not frozen')


def worker_label(stage: str, job: str) -> None:
    label_prerequisite(stage, job)
    rd._init_openmp_before_torch()
    from methods.ttha.batch_base import train
    jd = RUN / job
    ds = ds_of(job)
    js = rd.resolve_job(ds, job)
    lin = None
    if stage in ('c_b', 'freeze_e'):
        ctx = rd.open_job(ds, job, RUN, stage='c_b' if stage == 'c_b' else 'e_input')
        origins = js.c_b if stage == 'c_b' else js.e
        N = len(js.roster)
        Xs, serve = rd.serve_inputs(ctx.slice, origins, [[] for _ in range(N)], ctx.scaler)
        cells = _cells(jd)
        if stage == 'c_b':
            truth = rd.read_truth(ctx.slice, origins)
            out = {'job_id': job, 'opened_local': now(), 'rows_read': list(js.c_b_rows), 'serve': serve, 'cells': {}}
            for c, r in cells.items():
                pred = rd.predict_inputs(train.load_model(r['model_path']), Xs, ctx.scaler)
                out['cells'][c] = {'material_id': r['material_id'], 'model_seed': r['model_seed'], 'n_updates': r['n_updates'],
                                   'c_b': rd.score_block(pred, truth, ctx.scaler)}
            context.write_json(jd / 'c_b_scores.json', out)
        else:
            (jd / 'pred_e').mkdir(exist_ok=True)
            out = {'job_id': job, 'started_local': now(), 'rows_read': list(js.e_input_rows), 'serve': serve, 'models': {}}
            for c, r in cells.items():
                pred = rd.predict_inputs(train.load_model(r['model_path']), Xs, ctx.scaler)
                p = jd / 'pred_e' / (c + '.npz')
                np.savez(p, pred_raw=pred, origins=np.array(js.e))
                out['models'][c] = {'path': str(p), 'material_id': r['material_id'], 'model_seed': r['model_seed'], 'n_updates': r['n_updates']}
            out['frozen_local'] = now()
            context.write_json(jd / 'e_frozen.json', out)
    elif stage == 'score_e':
        ctx = rd.open_job(ds, job, RUN, stage='e_target')
        fz = context.read_json(jd / 'e_frozen.json')
        truth = rd.read_truth(ctx.slice, js.e)
        out = {'job_id': job, 'scored_local': now(), 'e_frozen_local': fz['frozen_local'], 'rows_read': list(js.e_target_rows), 'cells': {}}
        for c, m in fz['models'].items():
            with np.load(m['path']) as z:
                if not np.array_equal(z['origins'], np.array(js.e)):
                    raise RuntimeError('frozen E origins differ')
                pred = z['pred_raw'].copy()
            out['cells'][c] = {**{k: m[k] for k in ('material_id', 'model_seed', 'n_updates')}, 'e': rd.score_block(pred, truth, ctx.scaler)}
        context.write_json(jd / 'e_scores.json', out)
    else:
        raise ValueError(stage)
    print('LABEL_OK', stage, job, flush=True)


def worker_compare(a: str, b: str) -> None:
    import torch
    sa, sb = torch.load(a, map_location='cpu'), torch.load(b, map_location='cpu')
    same = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    print(json.dumps({'identical': bool(same)}))


# ============================================================================= controller side
def worker_env() -> dict:
    env = dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    return env


def run_sub(args: list, log: Path, timeout: float) -> tuple:
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with log.open('w', encoding='utf-8') as fh:
            p = subprocess.run([sys.executable, '-B', '-m', MODULE] + args, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout, env=worker_env())
        return p.returncode, time.time() - t0
    except subprocess.TimeoutExpired:
        return -999, time.time() - t0


class BudgetStop(RuntimeError):
    pass


class Ledger:
    def __init__(self):
        self.p = ROOT / 'budget.json'
        self.d = context.read_json(self.p) if self.p.exists() else {'caps': CAPS, 'fit': 0, 'check': 0, 'retry': 0, 'physical': 0,
                                                                     'failed': [], 'fit_seconds': [], 'started_epoch': time.time(), 'events': []}

    def save(self):
        context.write_json(self.p, self.d)

    def reserve(self, kind: str, what: str):
        if self.d['physical'] >= CAPS['hard'] or self.d[kind] >= CAPS[kind]:
            raise BudgetStop('%s cap reached before %s' % (kind, what))
        if time.time() - self.d['started_epoch'] > WALL_NUMERIC_S:
            raise BudgetStop('numeric wall clock reached before %s' % what)
        self.d[kind] += 1
        self.d['physical'] += 1
        self.save()

    def done(self, what: str, ok: bool, secs: float, kind: str):
        self.d['events'].append({'what': what, 'kind': kind, 'ok': ok, 'seconds': secs, 'local': now()})
        if ok and kind == 'fit':
            self.d['fit_seconds'].append(secs)
        if not ok:
            self.d['failed'].append(what)
        self.save()


def cell_path(job: str, mid: str, seed: int, n: int, tag: str = '') -> Path:
    return RUN / job / 'cells' / (cid(job, mid, seed, n, tag) + '.json')


def fit(ledger: Ledger, job: str, mid: str, seed: int, kind: str = 'fit', tag: str = '') -> bool:
    if all(cell_path(job, mid, seed, n, tag).exists() for n in CONSUMERS):
        return True
    for attempt in (0, 1):
        ledger.reserve(kind if attempt == 0 else 'retry', cid(job, mid, seed, N_FULL, tag))
        args = ['--worker-fit', job, mid, str(seed)] + (['--tag', tag] if tag else [])
        rc, secs = run_sub(args, RUN / job / 'fit_logs' / (cid(job, mid, seed, N_FULL, tag) + ('.log' if attempt == 0 else '.retry.log')), FIT_TIMEOUT_S)
        ok = rc == 0 and all(cell_path(job, mid, seed, n, tag).exists() for n in CONSUMERS)
        ledger.done(cid(job, mid, seed, N_FULL, tag), ok, secs, kind)
        print('FIT', cid(job, mid, seed, N_FULL, tag), 'OK' if ok else 'FAILED rc=%s' % rc, round(secs, 1), flush=True)
        if ok:
            return True
    return False


def environment() -> dict:
    probe = subprocess.run([sys.executable, '-B', '-c', 'import json, sys, torch, numpy; print(json.dumps({"python": sys.version.split()[0], '
                            '"torch": torch.__version__, "numpy": numpy.__version__, "threads": torch.get_num_threads()}))'],
                           capture_output=True, text=True, env=worker_env(), cwd=str(REPO))
    got = json.loads(probe.stdout.strip().splitlines()[-1])
    ref = context.read_json(next((CACHE['RD01B_Q1'] / 'cells').glob('*P0_Linear*u2000.json')))
    out = {'executable': sys.executable, **got, 'cache_python': ref['python'], 'cache_torch': ref['torch'],
           'kmp_duplicate_lib_ok_in_shell_at_start': KMP_AT_START}
    out['ok'] = got['python'] == ref['python'] and got['torch'] == ref['torch'] and KMP_AT_START is None
    return out


def preflight() -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    env = environment()
    if not env['ok']:
        raise RuntimeError('environment differs from the cached Consumer runs: %s' % env)
    sealing = {}
    for job in JOBS:
        js = rd.resolve_job(ds_of(job), job)
        top = max(js.e_target_rows)
        sealing[job] = {'t': js.t, 'rows_max': int(top), 'prsa_closed_from': PRSA_CLOSED_FROM_ROW,
                        'ok': ds_of(job) != 'RD02' or top <= PRSA_CLOSED_FROM_ROW}
        for s in SEEDS[job]:
            for n in CONSUMERS:
                p = CACHE[job] / 'cells' / ('%s__P0_Linear__s%d__u%d.json' % (job, s, n))
                if not p.exists():
                    raise RuntimeError('cached reference cell missing: %s' % p)
    if not all(v['ok'] for v in sealing.values()):
        raise RuntimeError('sealed rows would be read')
    cfg = {'package': 'DEV-DATA-AUG-OPERATOR-PROBE', 'frozen_local': now(), 'environment': env, 'sealing': sealing, 'jobs': JOBS, 'seeds': SEEDS,
           'consumers': {'primary': N_FULL, 'short': N_SHORT, 'note': 'the 100-update model is the checkpoint of the same trajectory'},
           'materials': [{'material_id': m, **MAT[m], 'semantics': OP_TEXT.get(MAT[m]['op'], 'no augmentation (Baseline-Linear only)')} for m in MIDS],
           'excluded_tempopfn_ops': EXCLUDED, 'aug_seed': 'RandomState(%d + 1000*material_no + entity)' % AUG_SEED_BASE,
           'serving': 'Baseline-Linear inputs for every model', 'training': 'parent 0.5 + child 0.5 pooled MSE when a child exists (train.train_arm)',
           'rules': RULE_TEXT, 'caps': CAPS, 'wall_numeric_s': WALL_NUMERIC_S, 'llm_calls': 0, 'api_calls': 0}
    context.write_json(ROOT / 'frozen_config.json', cfg)
    return cfg


# ============================================================================= selection rules (pure)
RULE_TEXT = {
    'R_None': 'Baseline-Linear, no augmentation (the current Harness default Consumer input)',
    'R_Fixed': 'always FixedMixup (the fixed default of the augmentation line)',
    'R_Random': 'uniform choice over all 24 materials; reported as the exact expectation',
    'R_CA': 'argmin 3-seed mean C_A over all 24 materials (exhaustive search upper bound of a C_A-driven Harness)',
    'R_CA_old': 'argmin C_A over the old library {None, FixedMixup, FreqMask10, FreqMixR10}',
    'R_CA_b2': 'budget-2 search: {None, FixedMixup} + 2 of the other 22 drawn uniformly; argmin C_A; exact expectation over all 231 pairs',
    'consumers': 'every rule is applied inside the 2000-update Consumer and inside the 100-update Consumer; R_Short = R_None at 100',
    'R_CA_joint': 'argmin C_A over all (material, consumer) pairs',
    'ties': '|difference| <= 1e-12 -> earlier material in the frozen list (None first); 2000 before 100',
    'readouts': 'delivered gain = (E(None,2000) - E(rule)) / mean E(None,2000), per seed, then 3-seed mean; jobs equal weight (= domains equal)',
    'regret': 'E(pick) - E(post-hoc best of the same pool): DEVELOPMENT CLUE ONLY, never evidence of method success; exact first place not required',
}


def pick(means: dict, order) -> str | None:
    best = None
    for m in order:
        v = means.get(m)
        if v is None:
            continue
        if best is None or v < means[best] - TIE_TOL:
            best = m
    return best


def select_job(ca: dict) -> dict:
    """ca[mid][n] = 3-seed mean C_A (None when not scorable). Returns the frozen picks per rule and consumer."""
    out = {}
    for n in CONSUMERS:
        means = {m: ca[m][n] for m in MIDS}
        out[str(n)] = {'R_None': 'None', 'R_Fixed': 'FixedMixup', 'R_CA': pick(means, MIDS), 'R_CA_old': pick(means, OLD_POOL),
                       'R_CA_b2_pairs': {'%s|%s' % pr: pick(means, BUDGET2_FIXED + pr) for pr in itertools.combinations(BUDGET2_DRAW, 2)}}
    joint = {(m, n): ca[m][n] for m in MIDS for n in CONSUMERS if ca[m][n] is not None}
    best = None
    for n in CONSUMERS:
        for m in MIDS:
            if (m, n) in joint and (best is None or joint[(m, n)] < joint[best] - TIE_TOL):
                best = (m, n)
    out['R_CA_joint'] = {'material': best[0], 'consumer': best[1]} if best else None
    return out


# ============================================================================= run
def check_none() -> dict:
    eq = []
    for job in JOBS:
        for s in SEEDS[job]:
            for n in CONSUMERS:
                mine = context.read_json(cell_path(job, 'None', s, n))
                ref = context.read_json(CACHE[job] / 'cells' / ('%s__P0_Linear__s%d__u%d.json' % (job, s, n)))
                a, b = mine['scores']['c_a']['normalized_mse_macro'], ref['scores']['c_a']['normalized_mse_macro']
                row = {'job': job, 'seed': s, 'n': n, 'mine': a, 'cache': b, 'equal': a == b}
                if n == N_FULL:
                    rc = subprocess.run([sys.executable, '-B', '-m', MODULE, '--worker-compare', mine['model_path'], ref['model_path']],
                                        capture_output=True, text=True, cwd=str(REPO), env=worker_env())
                    row['weights_identical'] = json.loads(rc.stdout.strip().splitlines()[-1])['identical'] if rc.returncode == 0 else None
                eq.append(row)
    out = {'none_reproduces_training_length_cache': {'rows': eq, 'ok': all(r['equal'] and r.get('weights_identical', True) for r in eq)}}
    context.write_json(ROOT / 'checks_none.json', out)
    return out


def check_rerun(ledger: Ledger) -> dict:
    out = context.read_json(ROOT / 'checks_none.json')
    job, mid, s = 'RD02_T1', 'FixedMixup', SEEDS['RD02_T1'][0]
    fit(ledger, job, mid, s, kind='check', tag='rerun')
    a, b = context.read_json(cell_path(job, mid, s, N_FULL)), context.read_json(cell_path(job, mid, s, N_FULL, 'rerun'))
    rc = subprocess.run([sys.executable, '-B', '-m', MODULE, '--worker-compare', a['model_path'], b['model_path']], capture_output=True, text=True,
                        cwd=str(REPO), env=worker_env())
    out['augmented_refit_identical'] = {'cell': a['cell_id'], 'c_a_equal': a['scores']['c_a']['normalized_mse_macro'] == b['scores']['c_a']['normalized_mse_macro'],
                                        'weights_identical': json.loads(rc.stdout.strip().splitlines()[-1])['identical'] if rc.returncode == 0 else None}
    out['augmented_refit_identical']['ok'] = bool(out['augmented_refit_identical']['c_a_equal'] and out['augmented_refit_identical']['weights_identical'])
    out['all_ok'] = out['none_reproduces_training_length_cache']['ok'] and out['augmented_refit_identical']['ok']
    context.write_json(ROOT / 'checks.json', out)
    return out


def ca_table(job: str) -> dict:
    ca = {m: {} for m in MIDS}
    for m in MIDS:
        for n in CONSUMERS:
            v = []
            for s in SEEDS[job]:
                r = context.read_json(cell_path(job, m, s, n))
                v.append(r['scores']['c_a']['normalized_mse_macro'])
            ca[m][n] = None if any(x is None for x in v) else statistics.fmean(v)
    return ca


def run() -> None:
    cfg = context.read_json(ROOT / 'frozen_config.json')
    ledger = Ledger()
    status = {'started_local': now(), 'stages': {}}
    try:
        for job in JOBS:
            if not (RUN / job / 'children_summary.json').exists():
                rc, secs = run_sub(['--worker-children', job], RUN / job / 'children.log', 1800)
                if rc != 0:
                    raise RuntimeError('children failed for %s' % job)
                status['stages']['children_' + job] = secs
        for mid in MIDS:
            for job in JOBS:
                for s in SEEDS[job]:
                    if not fit(ledger, job, mid, s):
                        raise RuntimeError('technical failure %s %s %s' % (job, mid, s))
            if mid == 'None':
                c0 = check_none()
                if not c0['none_reproduces_training_length_cache']['ok']:
                    status['exit'] = 'CHECK_FAILED'
                    status['failed_check'] = 'none_reproduces_training_length_cache'
                    return
        chk = check_rerun(ledger)
        status['checks_ok'] = chk['all_ok']
        if not chk['all_ok']:
            status['exit'] = 'CHECK_FAILED'
            return
        if not (ROOT / 'selections.json').exists():
            sel = {'frozen_local': now(), 'frozen_epoch': time.time(), 'rules': RULE_TEXT, 'jobs': {}}
            for job in JOBS:
                ca = ca_table(job)
                sel['jobs'][job] = {'picks': select_job(ca), 'c_a_means': {m: {str(n): ca[m][n] for n in CONSUMERS} for m in MIDS}}
            context.write_json(ROOT / 'selections.json', sel)
        for stage in ('c_b', 'freeze_e'):
            for job in JOBS:
                target = RUN / job / ('c_b_scores.json' if stage == 'c_b' else 'e_frozen.json')
                if not target.exists():
                    rc, secs = run_sub(['--worker-label', stage, job], RUN / job / ('label_%s.log' % stage), 1800)
                    if rc != 0:
                        raise RuntimeError('%s failed for %s' % (stage, job))
                    status['stages']['%s_%s' % (stage, job)] = secs
        barrier = ROOT / 'all_e_predictions_frozen.json'
        if not barrier.exists():
            context.write_json(barrier, {'jobs': {j: context.read_json(RUN / j / 'e_frozen.json')['frozen_local'] for j in JOBS}, 'written_local': now()})
        for job in JOBS:
            if not (RUN / job / 'e_scores.json').exists():
                rc, secs = run_sub(['--worker-label', 'score_e', job], RUN / job / 'label_score_e.log', 1800)
                if rc != 0:
                    raise RuntimeError('score_e failed for %s' % job)
                status['stages']['score_e_' + job] = secs
        status['exit'] = 'COMPLETE'
    except BudgetStop as exc:
        status['exit'] = 'BUDGET_STOP'
        status['reason'] = str(exc)
    finally:
        status['finished_local'] = now()
        status['numeric_wall_seconds'] = time.time() - ledger.d['started_epoch']
        context.write_json(ROOT / 'run_status.json', status)
    if status.get('exit') == 'COMPLETE':
        readout()


# ============================================================================= readout
def _ms(v: list) -> dict:
    v = [x for x in v if x is not None]
    if not v:
        return {'n': 0, 'mean': None, 'se': None}
    return {'n': len(v), 'mean': statistics.fmean(v), 'se': (statistics.stdev(v) / math.sqrt(len(v))) if len(v) > 1 else None}


def scores(job: str) -> dict:
    """S[block][mid][n][seed]"""
    jd = RUN / job
    cb = context.read_json(jd / 'c_b_scores.json')['cells']
    ee = context.read_json(jd / 'e_scores.json')['cells']
    S = {b: {m: {n: {} for n in CONSUMERS} for m in MIDS} for b in ('c_a', 'c_b', 'e')}
    for m in MIDS:
        for n in CONSUMERS:
            for s in SEEDS[job]:
                c = cid(job, m, s, n)
                S['c_a'][m][n][s] = context.read_json(cell_path(job, m, s, n))['scores']['c_a']['normalized_mse_macro']
                S['c_b'][m][n][s] = cb[c]['c_b']['normalized_mse_macro']
                S['e'][m][n][s] = ee[c]['e']['normalized_mse_macro']
    return S


def delivered(S: dict, block: str, job: str, choice) -> list:
    """Per-seed relative gain vs None@2000 of a choice: (mid, n) or a list of (weight, mid, n) for an expectation."""
    seeds = SEEDS[job]
    base = S[block]['None'][N_FULL]
    den = statistics.fmean(base[s] for s in seeds)
    if isinstance(choice, tuple):
        choice = [(1.0, choice[0], choice[1])]
    out = []
    for s in seeds:
        v = sum(w * S[block][m][n][s] for w, m, n in choice)
        out.append((base[s] - v) / den)
    return out


def readout() -> dict:
    sel = context.read_json(ROOT / 'selections.json')
    res = {'package': 'DEV-DATA-AUG-OPERATOR-PROBE', 'written_local': now(), 'jobs': {}, 'pooled': {}}
    rules_all = {}
    for job in JOBS:
        S = scores(job)
        pk = sel['jobs'][job]['picks']
        rows = {}
        for n in CONSUMERS:
            p = pk[str(n)]
            b2 = [(1.0 / len(p['R_CA_b2_pairs']), v, n) for v in p['R_CA_b2_pairs'].values()]
            rand = [(1.0 / len(MIDS), m, n) for m in MIDS]
            tag = '' if n == N_FULL else '@100'
            rows.update({'R_None' + tag: ('None', n), 'R_Fixed' + tag: ('FixedMixup', n), 'R_Random' + tag: rand,
                         'R_CA' + tag: (p['R_CA'], n), 'R_CA_old' + tag: (p['R_CA_old'], n), 'R_CA_b2' + tag: b2})
        if pk['R_CA_joint']:
            rows['R_CA_joint'] = (pk['R_CA_joint']['material'], pk['R_CA_joint']['consumer'])
        jr = {'picks': {k: (v[0] + ('' if v[1] == N_FULL else '@100')) if isinstance(v, tuple) else 'expectation' for k, v in rows.items()},
              'rules': {}, 'candidates': {}}
        for k, ch in rows.items():
            jr['rules'][k] = {b: _ms(delivered(S, b, job, ch)) for b in ('c_a', 'c_b', 'e')}
            rules_all.setdefault(k, []).append(jr['rules'][k])
        # candidate table (development clues): effect vs None@2000 on each block, both consumers
        for n in CONSUMERS:
            for m in MIDS:
                jr['candidates']['%s@%d' % (m, n)] = {b: _ms(delivered(S, b, job, (m, n))) for b in ('c_a', 'c_b', 'e')}
        for n in CONSUMERS:
            e_means = {m: statistics.fmean(S['e'][m][n].values()) for m in MIDS}
            best = min(e_means, key=e_means.get)
            order = sorted(MIDS, key=e_means.get)
            den = statistics.fmean(S['e']['None'][N_FULL].values())
            pick_ca = pk[str(n)]['R_CA']
            diffs = [S['e'][pick_ca][n][s] - S['e'][best][n][s] for s in SEEDS[job]]
            st = _ms(diffs)
            jr['regret_dev_clue_%d' % n] = {'post_hoc_best_e': best, 'R_CA_pick': pick_ca, 'R_CA_rank_on_e': order.index(pick_ca) + 1, 'of': len(MIDS),
                                            'R_CA_regret_rel': (e_means[pick_ca] - e_means[best]) / den,
                                            'R_Fixed_regret_rel': (e_means['FixedMixup'] - e_means[best]) / den,
                                            'R_None_regret_rel': (e_means['None'] - e_means[best]) / den,
                                            'R_Random_regret_rel': (statistics.fmean(e_means.values()) - e_means[best]) / den,
                                            'R_CA_within_2se_of_best': st['se'] is not None and abs(st['mean']) <= 2 * st['se'],
                                            'note': 'post-hoc best on already-exposed E: development clue only'}
        # C_A/E sign agreement over candidates (vs None, same consumer)
        agree = []
        for n in CONSUMERS:
            for m in MIDS:
                if m == 'None':
                    continue
                dca = statistics.fmean(S['c_a']['None'][n][s] - S['c_a'][m][n][s] for s in SEEDS[job])
                de = statistics.fmean(S['e']['None'][n][s] - S['e'][m][n][s] for s in SEEDS[job])
                agree.append({'m': m, 'n': n, 'dca_rel': dca / statistics.fmean(S['c_a']['None'][n].values()), 'same_sign': (dca > 0) == (de > 0)})
        jr['ca_e_agreement'] = agree
        res['jobs'][job] = jr
    for k, lst in rules_all.items():
        res['pooled'][k] = {}
        for b in ('c_a', 'c_b', 'e'):
            ms = [x[b] for x in lst]
            if any(x['mean'] is None for x in ms):
                res['pooled'][k][b] = None
                continue
            se = [x['se'] or 0.0 for x in ms]
            res['pooled'][k][b] = {'mean': statistics.fmean(x['mean'] for x in ms), 'se': math.sqrt(sum(v * v for v in se)) / len(ms),
                                   'jobs_positive': sum(x['mean'] > 0 for x in ms), 'jobs': len(ms)}
    res['cost'] = cost_readout()
    res['children'] = {j: context.read_json(RUN / j / 'children_summary.json')['materials'] for j in JOBS}
    res['checks'] = context.read_json(ROOT / 'checks.json')
    res['run_status'] = context.read_json(ROOT / 'run_status.json')
    context.write_json(ROOT / 'result.json', res)
    (ROOT / 'tables.md').write_text(tables(res), encoding='utf-8')
    return res


def cost_readout() -> dict:
    led = context.read_json(ROOT / 'budget.json')
    fits = [e for e in led['events'] if e['kind'] == 'fit' and e['ok']]
    by_mat = {}
    for e in fits:
        m = e['what'].split('__')[1]
        by_mat.setdefault(m, []).append(e['seconds'])
    mean_fit = {m: statistics.fmean(v) for m, v in by_mat.items()}
    short_train = []
    for job in JOBS:
        for s in SEEDS[job]:
            r = context.read_json(cell_path(job, 'None', s, N_SHORT))
            short_train.append(r['train']['seconds_at_this_length'])
    full_none = mean_fit.get('None')
    none_train = statistics.fmean(context.read_json(cell_path(j, 'None', s, N_FULL))['train']['seconds_at_this_length'] for j in JOBS for s in SEEDS[j])
    per_job_fits = {'R_None': 3, 'R_Fixed': 3, 'R_Random': 3, 'R_Short (None@100)': 3, 'R_CA_old': 12, 'R_CA_b2': 12, 'R_CA': 72}
    avg_aug = statistics.fmean(v for m, v in mean_fit.items() if m != 'None')
    est = {'R_None': 3 * full_none, 'R_Fixed': 3 * mean_fit['FixedMixup'], 'R_Random': 3 * statistics.fmean(mean_fit.values()),
           'R_Short (None@100)': 3 * (full_none - none_train + statistics.fmean(short_train)),
           'R_CA_old': sum(3 * mean_fit[m] for m in OLD_POOL), 'R_CA_b2': 3 * (full_none + mean_fit['FixedMixup'] + 2 * avg_aug),
           'R_CA': sum(3 * v for v in mean_fit.values())}
    children = {j: sum(v['build_seconds'] for v in context.read_json(RUN / j / 'children_summary.json')['materials'].values()) for j in JOBS}
    # amendment 1: the first ~70 fits ran under external CPU contention. Unit costs for the rule comparison therefore use the median
    # augmented fit wall of this run (robust to the contended minority) and the uncontended None fit wall measured in TRAINING-LENGTH.
    aug_walls = [e['seconds'] for e in fits if e['what'].split('__')[1] != 'None']
    aug_train = [context.read_json(cell_path(j, m, s, N_FULL))['train']['trajectory_seconds'] for j in JOBS for m in MIDS if m != 'None' for s in SEEDS[j]]
    tl = context.read_json(REPO / '_scratch' / 'dev_data_readiness_short_adapt_signal' / 'result.json')['cost']['full_training_comparator_ESTIMATE']
    none_unit = tl['per_fit_wall_seconds_measured_in_TRAINING_LENGTH']['mean_of_86_trajectory_workers']
    none_train = tl['per_fit_wall_seconds_measured_in_TRAINING_LENGTH']['mean_2000_update_train_seconds']
    aug_unit = statistics.median(aug_walls)
    short_unit = none_unit - none_train + none_train * N_SHORT / N_FULL
    unit = {'None_fit_wall_s (TRAINING-LENGTH, uncontended)': none_unit, 'augmented_fit_wall_s (median this run)': aug_unit,
            'augmented_train_s (median this run)': statistics.median(aug_train), 'None_train_s (TRAINING-LENGTH)': none_train,
            'short_None_fit_wall_s (ESTIMATE)': short_unit}
    unc = {'R_None': 3 * none_unit, 'R_Fixed': 3 * aug_unit, 'R_Random': 3 * (none_unit + 23 * aug_unit) / 24, 'R_Short (None@100)': 3 * short_unit,
           'R_CA_old': 3 * (none_unit + 3 * aug_unit), 'R_CA_b2': 3 * (none_unit + 3 * aug_unit), 'R_CA': 3 * (none_unit + 23 * aug_unit)}
    return {'uncontended_unit_costs_ESTIMATE': unit, 'per_job_seconds_by_rule_uncontended_ESTIMATE': unc,
            'contention_note': 'amendment 1: the None..AmpMod60 fits (first ~70) ran while an external sweep used ~13 cores; later fits ran uncontended',
            'physical_fits': led['physical'], 'fit_count': led['fit'], 'check_fits': led['check'], 'retries': led['retry'], 'failed': led['failed'],
            'fit_wall_seconds_total': sum(e['seconds'] for e in led['events']), 'mean_fit_wall_by_material': mean_fit,
            'per_job_fits_by_rule': per_job_fits, 'per_job_seconds_by_rule_ESTIMATE_from_measured_fits': est,
            'short_note': 'R_Short cost = None fit wall - None 2000-update train seconds + measured train seconds to update 100 (ESTIMATE; no separate short run)',
            'children_build_seconds_by_job': children, 'llm_calls': 0, 'api_calls': 0}


def _f(x, pct=True, nd=1):
    if x is None:
        return 'n/a'
    return ('%+.' + str(nd) + 'f%%') % (100 * x) if pct else ('%.' + str(nd) + 'f') % x


def tables(res: dict) -> str:
    L = ['# DEV-DATA-AUG-OPERATOR-PROBE tables', '',
         'Gain = (E(None,2000) - E(rule)) / mean E(None,2000); positive = better than no augmentation with the current Consumer. '
         'Jobs equal weight (two per domain). SE = per-job 3-seed SE combined. Post-hoc-best columns are development clues only.', '',
         '## 1. Delivered gain by rule (pooled over 4 jobs)', '', '| Rule | E gain (SE) | jobs + | C_B gain | C_A gain |', '|---|---:|---:|---:|---:|']
    for k, v in res['pooled'].items():
        e, cb, ca = v['e'], v['c_b'], v['c_a']
        L.append('| %s | %s (%s) | %d/%d | %s | %s |' % (k, _f(e['mean']), _f(e['se']), e['jobs_positive'], e['jobs'], _f(cb['mean']), _f(ca['mean'])))
    L += ['', '## 2. Per job: picks and E gain', '']
    for job, jr in res['jobs'].items():
        L += ['### %s' % job, '', '| Rule | pick | E gain (SE) | C_B gain |', '|---|---|---:|---:|']
        for k, v in jr['rules'].items():
            L.append('| %s | %s | %s (%s) | %s |' % (k, jr['picks'][k], _f(v['e']['mean']), _f(v['e']['se']), _f(v['c_b']['mean'])))
        for n in CONSUMERS:
            r = jr['regret_dev_clue_%d' % n]
            L.append('')
            L.append('Consumer %d (dev clue): post-hoc best E = %s; R_CA pick %s ranks %d/%d; regret R_CA %s, Fixed %s, None %s, Random %s; '
                     'R_CA within 2 SE of best: %s' % (n, r['post_hoc_best_e'], r['R_CA_pick'], r['R_CA_rank_on_e'], r['of'], _f(r['R_CA_regret_rel']),
                                                      _f(r['R_Fixed_regret_rel']), _f(r['R_None_regret_rel']), _f(r['R_Random_regret_rel']),
                                                      r['R_CA_within_2se_of_best']))
        L.append('')
    L += ['## 3. Candidate E gains vs None@2000 (3-seed mean; development clues)', '',
          '| Material | ' + ' | '.join('%s @2000' % j for j in JOBS) + ' | ' + ' | '.join('%s @100' % j for j in JOBS) + ' |',
          '|---|' + '---:|' * (2 * len(JOBS))]
    for m in MIDS:
        cells = [_f(res['jobs'][j]['candidates']['%s@%d' % (m, n)]['e']['mean']) for n in CONSUMERS for j in JOBS]
        L.append('| %s | %s |' % (m, ' | '.join(cells)))
    L += ['', '## 4. Cost', '', '```json', json.dumps(res['cost'], indent=1, ensure_ascii=False), '```', '',
          '## 5. Operator change magnitude (joint window, normalized units)', '', '| Material | ' + ' | '.join(JOBS) + ' |', '|---|' + '---:|' * len(JOBS)]
    for m in MIDS[1:]:
        L.append('| %s | %s |' % (m, ' | '.join('rms %.3f / %.0f%%' % (res['children'][j][m]['rms_change_norm'], 100 * res['children'][j][m]['fraction_points_changed'])
                                              for j in JOBS)))
    return '\n'.join(L) + '\n'


# ============================================================================= smoke (numpy only; no torch in the controller)
def smoke() -> dict:
    checks = {}
    rng = np.random.RandomState(0)
    n, T = 40, 240
    Z = np.cumsum(rng.normal(0, 0.2, size=(n, T)), axis=1)
    hours = (np.arange(T)[None, :] + rng.randint(0, 24, size=(n, 1))) % 24
    k = np.arange(n)
    others = {1: (Z[::-1].copy(), {int(i): int(i) for i in range(n)})}
    ok_all = True
    rms = {}
    for mid in MIDS[1:]:
        a, _ = apply_material(mid, Z, 0, k, hours, others)
        b, _ = apply_material(mid, Z, 0, k, hours, others)
        good = a.shape == Z.shape and np.isfinite(a).all() and np.array_equal(a, b) and not np.array_equal(a, Z)
        rms[mid] = float(np.sqrt(((a - Z) ** 2).mean()))
        ok_all &= bool(good)
    checks['operators_deterministic_finite_nonidentity'] = ok_all
    checks['none_is_identity'] = apply_material('None', Z, 0, k, hours, others)[0] is None
    checks['strength_orders_change'] = all(rms[a] < rms[b] for a, b in (('AmpMod30', 'AmpMod60'), ('Shift25', 'Shift50'), ('Shock10', 'Shock20'),
                                                                       ('DayLvl05', 'DayLvl10'), ('Noise05', 'Noise15'), ('Censor95', 'Censor90')))
    c = op_censor(Z, None, 0.9)[0]
    checks['censor_bounds'] = bool((c.max(1) <= np.quantile(Z, 0.9, axis=1) + 1e-12).all())
    z1 = np.zeros((3, T))
    hz = np.tile(np.arange(T) % 24, (3, 1))
    d1, _ = op_day_level(z1, np.random.RandomState(1), 1.0, hz)
    nz = np.flatnonzero(d1[0])
    checks['day_level_whole_clock_day'] = len(nz) == 24 and hz[0, nz[0]] == 0
    fixed = apply_material('FixedMixup', Z, 0, k, hours, others)[0]
    ref, _ = augment.step_timemixup(Z, Z, np.zeros(n), np.random.RandomState(AUG_SEED_BASE + 1000 * MAT['FixedMixup']['no']), 'R', 0.25)
    checks['fixed_mixup_is_existing_timemixup'] = bool(np.array_equal(fixed, ref))
    # rules
    ca = {m: {N_FULL: 1.0, N_SHORT: 1.0} for m in MIDS}
    ca['Shock10'][N_FULL] = 0.5
    ca['FreqMask10'][N_FULL] = 0.7
    ca['Noise05'][N_SHORT] = 0.4
    sj = select_job(ca)
    checks['rule_picks'] = (sj[str(N_FULL)]['R_CA'] == 'Shock10' and sj[str(N_FULL)]['R_CA_old'] == 'FreqMask10'
                            and sj[str(N_SHORT)]['R_CA'] == 'Noise05' and sj['R_CA_joint'] == {'material': 'Noise05', 'consumer': N_SHORT}
                            and pick({m: 1.0 for m in MIDS}, MIDS) == 'None' and len(sj[str(N_FULL)]['R_CA_b2_pairs']) == 231
                            and sum(v == 'Shock10' for v in sj[str(N_FULL)]['R_CA_b2_pairs'].values()) == 21)
    # expectation arithmetic of delivered()
    S = {'e': {m: {N_FULL: {s: 1.0 for s in SEEDS['RD02_T1']}, N_SHORT: {s: 1.0 for s in SEEDS['RD02_T1']}} for m in MIDS}}
    S['e']['Shock10'][N_FULL] = {s: 0.8 for s in SEEDS['RD02_T1']}
    g = delivered(S, 'e', 'RD02_T1', [(1.0 / len(MIDS), m, N_FULL) for m in MIDS])
    checks['random_expectation'] = all(abs(x - 0.2 / len(MIDS)) < 1e-12 for x in g)
    # barriers refuse before labels / torch
    global ROOT, RUN
    saved = ROOT, RUN
    with tempfile.TemporaryDirectory() as td:
        ROOT, RUN = Path(td), Path(td) / 'jobs'
        refusals = []
        for stage in ('c_b', 'freeze_e', 'score_e'):
            try:
                label_prerequisite(stage, 'RD02_T1')
                refusals.append(False)
            except PermissionError:
                refusals.append(True)
        ROOT, RUN = saved
    checks['label_barriers_refuse'] = all(refusals)
    checks['controller_has_no_torch'] = 'torch' not in sys.modules
    checks = {k: bool(v) for k, v in checks.items()}
    res = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks, 'smoke_rms_by_material': rms, 'written_local': now()}
    ROOT.mkdir(parents=True, exist_ok=True)
    context.write_json(ROOT / 'smoke_record.json', res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--worker-children')
    ap.add_argument('--worker-fit', nargs=3)
    ap.add_argument('--tag', default='')
    ap.add_argument('--worker-label', nargs=2)
    ap.add_argument('--worker-compare', nargs=2)
    a = ap.parse_args()
    if a.smoke:
        r = smoke()
        print(json.dumps(r['checks'], indent=1))
        print(r['status'])
    elif a.preflight:
        print(json.dumps(preflight()['environment'], indent=1))
    elif a.run:
        run()
    elif a.result:
        readout()
    elif a.worker_children:
        worker_children(a.worker_children)
    elif a.worker_fit:
        worker_fit(a.worker_fit[0], a.worker_fit[1], int(a.worker_fit[2]), a.tag)
    elif a.worker_label:
        worker_label(a.worker_label[0], a.worker_label[1])
    elif a.worker_compare:
        worker_compare(*a.worker_compare)
    else:
        ap.error('choose a mode')


if __name__ == '__main__':
    main()
