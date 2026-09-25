"""DEV-AUG-DLINEAR-SOURCE (docs/DEV_AUG_DLINEAR_SOURCE_TASK_2026-09-24.md): official DLinear (cure-lab/LTSF-Linear 0c11366) as the next
small Consumer. No LLM, no card learning, no Test.

Source calibration (None only, Source C_A+C_B, the PatchTST protocol) and then 24 Source cases x 4 programs (None, NoMix, Comp[censor],
Comp[shock]) x 3 seeds. Case split, parent windows, scalers and augmentation materials are read-only from the PatchTST offline-skill package
(inherited from OFFLINE-SKILL / the task-family screen). Fit workers are CPU processes, one torch thread each, run in parallel as memory
allows; each fit freezes its E prediction from E input windows only; targets are read after every fit exists. Output
_scratch/dev_aug_dlinear_source/.

  --run       plan, smoke, wiring, calibration, 288 fits, scoring, result.json / REPORT.md, then stop
  --readout   aggregate only
  worker:     --fit-batch <spec.json>
"""
from __future__ import annotations

import os

import argparse
import concurrent.futures as cf
import importlib.util
import json
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import context, data
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_aug_patchtst_offline_skill as M
from evaluation.main_protocol_p4 import batch_research_aug_offline_skill as OS
from evaluation.main_protocol_p4 import batch_research_domain_aug_decision_priority as DP

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_aug_dlinear_source'
SRC = M.ROOT                                                   # case split, parents, scalers, material registry (read-only)
MLP_SCREEN = REPO / '_scratch' / 'dev_aug_task_family_screen' / 'cases'
REF = ROOT / 'ref' / 'LTSF-Linear'
REF_COMMIT = '0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6'
MODULE = 'evaluation.main_protocol_p4.batch_research_aug_dlinear_source'
TASK = 'docs/DEV_AUG_DLINEAR_SOURCE_TASK_2026-09-24.md'
DOMAINS = ['D01', 'D02', 'D03']
SEEDS = list(M.SEEDS)
PROGRAMS = ['None', 'P_NoMixRecipe', 'C_censor', 'C_shock']
LABEL = {'None': 'None', 'P_NoMixRecipe': 'NoMix', 'C_censor': 'Comp[censor]', 'C_shock': 'Comp[shock]'}
ARCH = {'seq_len': ec.L, 'pred_len': ec.H, 'enc_in': 1, 'individual': False}          # official kernel_size 25 is fixed inside DLinear.py
LRS = [1e-4, 1e-3, 1e-2, 5e-2]                                  # spans the official LTSF-Linear scripts (1e-4 default ... 5e-2 traffic)
MARKS = [250, 500, 1000, 2000]
CALIB_CASES = ['%s_SCR_A%d_G1' % (d, a) for d in DOMAINS for a in (1, 2)]
WEIGHT_DECAY = 1e-4
WORKERS_MAX = int(os.environ.get('SEH_DLINEAR_WORKERS_MAX', '8'))
FIT_TIMEOUT_S = 1800


def read(p):
    return context.read_json(Path(p))


def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    context.write_json(Path(p), obj)


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def source_cases() -> list:
    plan = read(SRC / 'plan.json')
    return [c for d in DOMAINS for c in sorted(plan['cases']['source'][d])]


def freeze_plan() -> dict:
    p = ROOT / 'plan.json'
    if p.exists():
        return read(p)
    cases = source_cases()
    if len(cases) != 24 or not set(CALIB_CASES) <= set(cases):
        raise RuntimeError('Source cases differ from the frozen split')
    cc = M.css()
    plan = {'package': 'DEV-AUG-DLINEAR-SOURCE', 'task': TASK, 'frozen_local': now(), 'llm': 0,
            'identity': 'SOURCE_DEVELOPMENT_CONSUMER_SCREEN (Source cases only; no Select / Test read, no card learning)',
            'consumer': {'model': 'DLinear', 'reference': 'https://github.com/cure-lab/LTSF-Linear', 'commit': REF_COMMIT, 'file': 'models/DLinear.py',
                         'architecture': ARCH, 'kernel_size': 25, 'parameters': 2 * (ec.L * ec.H + ec.H)},
            'training': {'optimizer': 'AdamW', 'weight_decay': WEIGHT_DECAY, 'parent_batch': 64, 'parent_child_loss': [0.5, 0.5], 'dtype': 'float32',
                         'device': 'cpu, one torch thread per worker, deterministic algorithms',
                         'batch_stream': 'unchanged train.batch_indices prefix (same as MLP / PatchTST)', 'inputs': 'T-scaler normalized windows, no instance normalization',
                         'note': 'controlled task adaptation with the PatchTST package loop; not the official epoch / early-stopping benchmark recipe'},
            'calibration': {'material': 'None', 'cases': CALIB_CASES, 'lrs': LRS, 'checkpoints': MARKS, 'seed': SEEDS[0],
                            'objective': 'equal-domain mean C_A+C_B nMSE on the six Source calibration cases; exact tie: fewer updates then lower lr'},
            'cases': cases, 'periods': {c: 'A%s' % c.split('_A')[1][0] for c in cases}, 'cut_t': {c: cc[c].t for c in cases},
            'programs': PROGRAMS, 'labels': LABEL, 'seeds': SEEDS,
            'metric': 'G = 100 x (E(None) - E(plan)) / E(None), seed-mean E of the same case and Consumer; domain = mean of its 8 cases; three-domain = equal weight',
            'budget': {'fits': 3 + len(CALIB_CASES) * len(LRS) + len(cases) * len(PROGRAMS) * len(SEEDS), 'llm': 0},
            'reads_only': [str(SRC.relative_to(REPO)), str(MLP_SCREEN.relative_to(REPO)) + ' (MLP side table)',
                           str((SRC / 'source' / 'scores').relative_to(REPO)) + ' (PatchTST side table)']}
    write(p, plan)
    return plan


# ============================================================================= worker (CPU)
_TORCH = None


def setup():
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    os.environ.pop('KMP_DUPLICATE_LIB_OK', None)
    from methods.ttha.batch_base import readiness as rd
    rd._init_openmp_before_torch()
    os.environ.pop('KMP_DUPLICATE_LIB_OK', None)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _TORCH = torch
    return torch


def model_new(torch, seed: int):
    sp = importlib.util.spec_from_file_location('official_dlinear', REF / 'models' / 'DLinear.py')
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    torch.manual_seed(seed)
    return mod.Model(argparse.Namespace(**ARCH))


def scaler_of(case: str):
    with np.load(SRC / 'common' / case / 'scaler.npz') as z:
        return data.Scaler(mean=z['mean'].copy(), std=z['std'].copy(), scale=z['scale'].copy(), floor_hits=0), z['mase'].copy()


def e_inputs_only(case: str) -> np.ndarray:
    """E input windows without reading any target row."""
    cs = M.css()[case]
    sc, _ = scaler_of(case)
    origins = list(cs.e)
    sl = ec.load_case_slice(cs, min(origins) - ec.L, max(origins))
    return np.stack([w.X_norm for w in data.build_windows(sl, origins, sc, with_truth=False)], axis=1).astype('float32')


def with_truth(case: str, origins: list):
    cs = M.css()[case]
    sc, mase = scaler_of(case)
    sl = ec.load_case_slice(cs, min(origins) - ec.L, max(origins) + ec.H)
    wins = data.build_windows(sl, origins, sc, with_truth=True)
    return np.stack([w.X_norm for w in wins], axis=1).astype('float32'), np.stack([w.y_true_raw for w in wins], axis=1), sc, mase


def predict(torch, model, x: np.ndarray) -> np.ndarray:
    model.eval()
    z = torch.as_tensor(x.reshape(-1, ec.L))
    with torch.no_grad():
        out = np.concatenate([model(z[k:k + 256, :, None]).squeeze(-1).numpy() for k in range(0, len(z), 256)])
    return out.reshape(*x.shape[:-1], ec.H)


def _peak_rss_mb():
    return M._peak_rss_mb()


def fit_one(torch, train, j: dict) -> None:
    c, seed, lr, steps = j['case'], int(j['seed']), float(j['lr']), int(j['steps'])
    out = REPO / j['output']
    out.mkdir(parents=True, exist_ok=True)
    with np.load(SRC / 'common' / c / 'parents.npz') as z:
        xp, yp = torch.as_tensor(z['Xp'].copy()), torch.as_tensor(z['yp'].copy())
    child = None
    if j['phys'] != 'None':
        with np.load(ec.load_registry(SRC / 'common' / c)[j['phys']]['path']) as z:
            if z['Xc'].shape != tuple(xp.shape) or z['yc'].shape != tuple(yp.shape):
                raise RuntimeError('child-parent shape mismatch')
            if not np.isfinite(z['Xc']).all() or not np.isfinite(z['yc']).all():
                raise RuntimeError('nonfinite frozen material')
            child = (torch.as_tensor(z['Xc'].copy()), torch.as_tensor(z['yc'].copy()))
    idxs = torch.as_tensor(train.batch_indices(seed, n_pool=len(xp), n_updates=steps))
    model = model_new(torch, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    val = with_truth(c, list(M.css()[c].c_a) + list(M.css()[c].c_b)) if j.get('marks') else None
    curve, scores = [], {}
    start = time.time()
    for step in range(1, steps + 1):
        model.train()
        idx = idxs[step - 1]
        opt.zero_grad(set_to_none=True)
        loss_p = ((model(xp[idx, :, None]).squeeze(-1) - yp[idx]) ** 2).mean()
        loss = loss_p if child is None else 0.5 * loss_p + 0.5 * ((model(child[0][idx, :, None]).squeeze(-1) - child[1][idx]) ** 2).mean()
        loss.backward()
        opt.step()
        if step == 1 or step % 250 == 0 or step == steps:
            v = float(loss.detach())
            if not np.isfinite(v):
                raise RuntimeError('nonfinite training loss')
            curve.append([step, v])
        if val is not None and step in j['marks']:
            x, y, sc, _ = val
            e = ((predict(torch, model, x) - (y - sc.mean[:, None, None]) / sc.scale[:, None, None]) ** 2).mean(axis=(0, 2))
            scores[str(step)] = {'nmse': float(e.mean()), 'per_origin': e.tolist()}
    elapsed = time.time() - start
    torch.save({k: v.detach().clone() for k, v in model.state_dict().items()}, out / 'model.pt')
    if not j.get('marks'):
        np.savez(out / 'pred_e.npz', pred=predict(torch, model, e_inputs_only(c)))           # frozen before any target is read
    write(out / 'fit.json', {'status': 'OK', 'job': j, 'train_seconds': elapsed, 'loss_curve': curve, 'validation': scores,
                             'parameters': sum(p.numel() for p in model.parameters()), 'peak_rss_mb_process': _peak_rss_mb(),
                             'torch': str(torch.__version__), 'completed_local': now()})
    print('FIT_OK', j['output'], round(elapsed, 2), flush=True)


def fit_batch(spec_path: str) -> None:
    torch = setup()
    from methods.ttha.batch_base import train
    for j in read(spec_path)['jobs']:
        fp = REPO / j['output'] / 'fit.json'
        if fp.exists() and read(fp).get('status') == 'OK':
            continue
        try:
            fit_one(torch, train, j)
        except Exception as exc:  # noqa: BLE001
            write(REPO / j['output'] / 'fit_error.json', {'job': j, 'error': repr(exc)[:500], 'traceback': traceback.format_exc()[-3000:], 'local': now()})
            print('FIT_FAIL', j['output'], repr(exc)[:200], flush=True)


# ============================================================================= controller
def job(case, phys, seed, lr, steps, out, marks=None) -> dict:
    return {'case': case, 'phys': phys, 'seed': int(seed), 'lr': lr, 'steps': steps, 'marks': marks or [], 'output': (ROOT / out).relative_to(REPO).as_posix()}


def run_jobs(batches: list, tag: str, workers: int) -> list:
    """batches: [[job, ...], ...]; one CPU worker process per batch, `workers` at a time; every job must end OK."""
    (ROOT / 'specs').mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)

    def one(k, jobs):
        if all((REPO / j['output'] / 'fit.json').exists() for j in jobs):
            return 0
        sp = ROOT / 'specs' / ('%s_%03d.json' % (tag, k))
        write(sp, {'jobs': jobs})
        with (ROOT / 'logs' / ('%s_%03d.log' % (tag, k))).open('w', encoding='utf-8') as f:
            return subprocess.run([sys.executable, '-B', '-u', '-m', MODULE, '--fit-batch', str(sp)], cwd=str(REPO), env=M.worker_env(), stdout=f,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                                  timeout=FIT_TIMEOUT_S * max(1, len(jobs))).returncode
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda a: one(*a), enumerate(batches)))
    recs, bad = [], []
    for b in batches:
        for j in b:
            fp = REPO / j['output'] / 'fit.json'
            (recs.append(read(fp)) if fp.exists() else bad.append(j['output']))
    if bad:
        raise RuntimeError('%s fits failed: %s' % (tag, bad[:4]))
    return recs


def workers_for(rss_mb: float) -> int:
    free_mb = OS.free_gb() * 1024
    return max(1, min(WORKERS_MAX, int((free_mb - 1500) / max(1.5 * rss_mb, 300))))


def smoke() -> dict:
    torch = setup()
    a, b = model_new(torch, SEEDS[0]), model_new(torch, SEEDS[0])
    x = torch.linspace(0, 1, 2 * ec.L).reshape(2, ec.L, 1)
    with torch.no_grad():
        ya, yb = a(x), b(x)
    res = {'output_shape': list(ya.shape), 'same_seed_same_init': bool(torch.equal(ya, yb)), 'parameters': sum(p.numel() for p in a.parameters()),
           'reference_file_present': (REF / 'models' / 'DLinear.py').exists(), 'e_input_shape': list(e_inputs_only(CALIB_CASES[0]).shape), 'local': now()}
    res['status'] = 'PASS' if res['output_shape'] == [2, ec.H, 1] and res['same_seed_same_init'] and res['reference_file_present'] else 'FAIL'
    write(ROOT / 'smoke.json', res)
    return res


def run() -> None:
    plan = freeze_plan()
    with DP.PackageLock(ROOT, 'run'):
        t0 = time.time()
        write(ROOT / 'status.json', {'status': 'RUNNING', 'phase': 'smoke', 'pid': os.getpid(), 'updated': now()})
        if smoke()['status'] != 'PASS':
            raise RuntimeError('smoke FAIL: %s' % read(ROOT / 'smoke.json'))
        # wiring: the same child fit twice is bitwise equal, differs from None (the child view is used)
        c0 = CALIB_CASES[0]
        w = run_jobs([[job(c0, 'None', SEEDS[0], 1e-3, 100, 'wiring/none'), job(c0, 'P_NoMixRecipe', SEEDS[0], 1e-3, 100, 'wiring/nomix_a'),
                       job(c0, 'P_NoMixRecipe', SEEDS[0], 1e-3, 100, 'wiring/nomix_b')]], 'wiring', 1)
        torch = setup()
        sd = {n: torch.load(ROOT / 'wiring' / n / 'model.pt', weights_only=True) for n in ('none', 'nomix_a', 'nomix_b')}
        same = all(torch.equal(sd['nomix_a'][k], sd['nomix_b'][k]) for k in sd['nomix_a'])
        differs = any(not torch.equal(sd['none'][k], sd['nomix_a'][k]) for k in sd['none'])
        rss = max(r['peak_rss_mb_process'] or 0 for r in w)
        workers = workers_for(rss)
        write(ROOT / 'wiring.json', {'status': 'PASS' if same and differs else 'FAIL', 'repeat_bitwise': same, 'child_changes_model': differs,
                                     'fit_seconds_100_steps': [r['train_seconds'] for r in w], 'peak_rss_mb': rss, 'workers_chosen': workers,
                                     'free_gb_at_choice': round(OS.free_gb(), 2), 'local': now()})
        if not (same and differs):
            raise RuntimeError('wiring FAIL')
        # calibration (None only, Source C_A + C_B only)
        write(ROOT / 'status.json', {'status': 'RUNNING', 'phase': 'calibration', 'pid': os.getpid(), 'workers': workers, 'updated': now()})
        if not (ROOT / 'consumer_frozen.json').exists():
            cj = [[job(c, 'None', SEEDS[0], lr, MARKS[-1], 'calibration/%s/lr%g' % (c, lr), MARKS)] for c in CALIB_CASES for lr in LRS]
            recs = run_jobs(cj, 'calibration', workers)
            opts = []
            for lr in LRS:
                for m in MARKS:
                    vs = {d: statistics.fmean(r['validation'][str(m)]['nmse'] for r in recs if r['job']['case'][:3] == d and r['job']['lr'] == lr) for d in DOMAINS}
                    opts.append({'lr': lr, 'steps': m, 'objective': statistics.fmean(vs.values()), 'domains': vs})
            best = min(opts, key=lambda o: (o['objective'], o['steps'], o['lr']))
            write(ROOT / 'consumer_frozen.json', {**best, 'all_source_options': opts, 'architecture': ARCH, 'frozen_local': now(),
                                                  'note': 'None only, Source C_A+C_B only; E and Test never read'})
        cfz = read(ROOT / 'consumer_frozen.json')
        # 24 Source cases x 4 programs x 3 seeds; one worker batch per case
        write(ROOT / 'status.json', {'status': 'RUNNING', 'phase': 'fits', 'pid': os.getpid(), 'workers': workers, 'consumer': {k: cfz[k] for k in ('lr', 'steps')},
                                     'updated': now()})
        batches = [[job(c, ph, s, cfz['lr'], cfz['steps'], 'fits/%s/%s/%d' % (c, ph, s)) for ph in PROGRAMS for s in SEEDS] for c in plan['cases']]
        run_jobs(batches, 'fits', workers)
        write(ROOT / 'models_frozen.json', {'frozen_local': now(), 'fits': sum(len(b) for b in batches)})
        # scoring behind the barrier
        write(ROOT / 'status.json', {'status': 'RUNNING', 'phase': 'score', 'pid': os.getpid(), 'updated': now()})
        for c in plan['cases']:
            score_case(c)
        res = readout(wall_s=time.time() - t0, workers=workers)
        write(ROOT / 'status.json', {'status': 'COMPLETE', 'finished_local': now(), 'three_domain': res['three_domain'], 'wall_s': round(time.time() - t0, 1)})
        print('COMPLETE', json.dumps(res['three_domain']), flush=True)


def score_case(c: str) -> None:
    if not (ROOT / 'models_frozen.json').exists():
        raise RuntimeError('E barrier: models not frozen')
    out = ROOT / 'scores' / ('%s.json' % c)
    if out.exists():
        return
    cs = M.css()[c]
    _, y, sc, mase = with_truth(c, list(cs.e))
    cells = {}
    for ph in PROGRAMS:
        for s in SEEDS:
            with np.load(ROOT / 'fits' / c / ph / str(s) / 'pred_e.npz') as z:
                pr = z['pred'] * sc.scale[:, None, None] + sc.mean[:, None, None]
            r = ec.score_with_status(pr, y, sc, mase)
            if r['status'] != 'SCORABLE':
                raise RuntimeError('non-scorable %s %s %d' % (c, ph, s))
            cells['%s__%d' % (ph, s)] = {'phys': ph, 'seed': s, 'nmse': r['normalized_mse_macro'],
                                         'per_origin': (((pr - y) / sc.scale[:, None, None]) ** 2).mean(axis=(0, 2)).tolist()}
    write(out, {'case': c, 'origins': list(cs.e), 'cells': cells, 'scored_local': now()})


# ============================================================================= readout
def e_by_consumer(c: str) -> dict:
    """Seed-mean E by program for DLinear (this package), PatchTST (its Source stage) and MLP (the task-family screen), same case and E origins."""
    out = {}
    d = read(ROOT / 'scores' / ('%s.json' % c))
    out['DLinear'] = {ph: statistics.fmean(d['cells']['%s__%d' % (ph, s)]['nmse'] for s in SEEDS) for ph in PROGRAMS}
    p = read(SRC / 'source' / 'scores' / ('%s.json' % c))
    if list(p['origins']) != list(d['origins']):
        raise RuntimeError('PatchTST E origins differ for %s' % c)
    out['PatchTST'] = {ph: statistics.fmean(v['nmse'] for v in p['cells'].values() if v['phys'] == ph) for ph in PROGRAMS}
    m = read(MLP_SCREEN / c / 'e_scores.json')
    if list(m['e_origins']) != list(d['origins']):
        raise RuntimeError('MLP E origins differ for %s' % c)
    out['MLP'] = {ph: statistics.fmean(v['e'] for v in m['cells'].values() if v['material_id'] == ph) for ph in PROGRAMS}
    return out


def G(e, e0):
    return 100.0 * (e0 - e) / e0


def readout(wall_s=None, workers=None) -> dict:
    plan = read(ROOT / 'plan.json')
    cases, per = plan['cases'], {}
    for c in cases:
        E = e_by_consumer(c)
        sd = read(ROOT / 'scores' / ('%s.json' % c))['cells']
        n0 = statistics.fmean(sd['None__%d' % s]['nmse'] for s in SEEDS)
        per[c] = {'domain': c[:3], 'period': plan['periods'][c], 'E': E,
                  'G': {k: {ph: G(E[k][ph], E[k]['None']) for ph in PROGRAMS[1:]} for k in E},
                  'G_seed_DLinear': {ph: [G(sd['%s__%d' % (ph, s)]['nmse'], n0) for s in SEEDS] for ph in PROGRAMS[1:]},
                  'G_by_origin_DLinear': {ph: [G(statistics.fmean(sd['%s__%d' % (ph, s)]['per_origin'][i] for s in SEEDS),
                                                 statistics.fmean(sd['None__%d' % s]['per_origin'][i] for s in SEEDS)) for i in range(4)] for ph in PROGRAMS[1:]}}

    def agg(k, ph, sel):
        cs = [c for c in cases if sel(c)]
        return statistics.fmean(per[c]['G'][k][ph] for c in cs) if cs else None
    consumers = ['DLinear', 'PatchTST', 'MLP']
    dom = {k: {ph: {d: agg(k, ph, lambda c, d=d: c[:3] == d) for d in DOMAINS} for ph in PROGRAMS[1:]} for k in consumers}
    three = {k: {ph: statistics.fmean(dom[k][ph].values()) for ph in PROGRAMS[1:]} for k in consumers}
    period = {ph: {d: {a: agg('DLinear', ph, lambda c, d=d, a=a: c[:3] == d and per[c]['period'] == a) for a in ('A1', 'A2')} for d in DOMAINS} for ph in PROGRAMS[1:]}
    wins = {ph: {d: [sum(per[c]['G']['DLinear'][ph] > 0 for c in cases if c[:3] == d), sum(per[c]['G']['DLinear'][ph] < 0 for c in cases if c[:3] == d)] for d in DOMAINS}
            for ph in PROGRAMS[1:]}
    seed_level = {ph: [statistics.fmean(statistics.fmean(per[c]['G_seed_DLinear'][ph][i] for c in cases if c[:3] == d) for d in DOMAINS) for i in range(len(SEEDS))]
                  for ph in PROGRAMS[1:]}
    origin = {ph: [statistics.fmean(statistics.fmean(per[c]['G_by_origin_DLinear'][ph][i] for c in cases if c[:3] == d) for d in DOMAINS) for i in range(4)] for ph in PROGRAMS[1:]}
    e_none = {k: {d: statistics.fmean(per[c]['E'][k]['None'] for c in cases if c[:3] == d) for d in DOMAINS} for k in consumers}
    best = {d: max(PROGRAMS[1:], key=lambda ph: dom['DLinear'][ph][d]) for d in DOMAINS}
    fits = [read(q) for q in (ROOT / 'fits').glob('*/*/*/fit.json')]
    calib = [read(q) for q in (ROOT / 'calibration').glob('*/*/fit.json')]
    wiring = read(ROOT / 'wiring.json')
    cost = {'llm_requests': 0, 'fits': {'main': len(fits), 'calibration': len(calib), 'wiring': 3},
            'train_seconds': {'main_sum': sum(r['train_seconds'] for r in fits), 'main_mean': statistics.fmean(r['train_seconds'] for r in fits),
                              'main_child_mean': statistics.fmean(r['train_seconds'] for r in fits if r['job']['phys'] != 'None'),
                              'calibration_sum': sum(r['train_seconds'] for r in calib)},
            'peak_rss_mb_worker': max(r['peak_rss_mb_process'] or 0 for r in fits + calib), 'workers': workers or wiring['workers_chosen'],
            'wall_s_package': wall_s, 'device': 'CPU (1 thread per worker)'}
    res = {'package': plan['package'], 'written_local': now(), 'consumer': read(ROOT / 'consumer_frozen.json'), 'wiring': wiring,
           'three_domain': three['DLinear'], 'by_domain': dom['DLinear'], 'by_period': period, 'wins_losses': wins, 'seed_level_three_domain': seed_level,
           'by_origin_three_domain': origin, 'best_program_by_domain': best, 'cross_consumer': {'three_domain': three, 'by_domain': dom, 'E_None_by_domain': e_none},
           'cut_t': {d: {a: sorted({plan['cut_t'][c] for c in cases if c[:3] == d and plan['periods'][c] == a}) for a in ('A1', 'A2')} for d in DOMAINS},
           'cost': cost, 'per_case': per}
    write(ROOT / 'result.json', res)
    (ROOT / 'REPORT.md').write_text(report_md(res), encoding='utf-8')
    return res


def report_md(r: dict) -> str:
    f = lambda x: '%+.2f' % x
    P = PROGRAMS[1:]
    cz = r['consumer']
    L = ['# DLinear 作为下一个小型 Consumer：Source 校准与 24 案 × 4 方案比较（不调 LLM）', '',
         '官方 DLinear（cure-lab/LTSF-Linear %s，`models/DLinear.py`，单通道、共享线性层、核 25）。训练循环与 PatchTST 包相同（AdamW wd 1e-4、每步父窗 64、父 / 子视图 0.5 / 0.5、'
         'T scaler 输入），CPU、每个进程一个线程。案例划分、父窗、scaler 与增强材料全部只读复用。只看 Source，没有学卡、没有读 Select / Test。' % REF_COMMIT[:7], '',
         '## 1. 校准（只用不增强，只看 Source C_A+C_B，6 个校准案例）', '',
         '选中 lr %g、%d 步，目标 %.4f。' % (cz['lr'], cz['steps'], cz['objective']), '', '| lr | 250 | 500 | 1000 | 2000 |', '|---|---:|---:|---:|---:|']
    for lr in LRS:
        L.append('| %g | %s |' % (lr, ' | '.join('%.4f' % next(o['objective'] for o in cz['all_source_options'] if o['lr'] == lr and o['steps'] == m) for m in MARKS)))
    L += ['', '## 2. 增强相对不增强的收益（pp of DLinear None，正 = 更好）', '', '| 方案 | 三域 | 电力 | 交通 | 太阳能 | seed 三域 |', '|---|---:|---:|---:|---:|---|']
    for ph in P:
        L.append('| %s | %s | %s | %s | %s | %s |' % (LABEL[ph], f(r['three_domain'][ph]), *[f(r['by_domain'][ph][d]) for d in DOMAINS],
                                                     ' / '.join(f(x) for x in r['seed_level_three_domain'][ph])))
    L += ['', '胜 / 负（8 例中 G>0 / G<0 的案例数）：']
    for ph in P:
        L.append('- %s：%s' % (LABEL[ph], '；'.join('%s %d/%d' % (d, *r['wins_losses'][ph][d]) for d in DOMAINS)))
    L += ['', '## 3. 分时期（每域 A1 / A2 各 4 例）', '', '| 方案 | 电力 A1 | 电力 A2 | 交通 A1 | 交通 A2 | 太阳能 A1 | 太阳能 A2 |', '|---|---:|---:|---:|---:|---:|---:|']
    for ph in P:
        L.append('| %s | %s |' % (LABEL[ph], ' | '.join(f(r['by_period'][ph][d][a]) for d in DOMAINS for a in ('A1', 'A2'))))
    L += ['', '切点 t：%s。' % json.dumps(r['cut_t']), '', '分起点（三域等权）：']
    for ph in P:
        L.append('- %s：%s' % (LABEL[ph], ' / '.join(f(x) for x in r['by_origin_three_domain'][ph])))
    cc = r['cross_consumer']
    L += ['', '## 4. 同 24 案、同 4 方案的跨 Consumer 对照（各自相对自己的 None；MLP 与 PatchTST 为已有读数）', '',
          '| Consumer | 不增强 E 电力 / 交通 / 太阳能 | NoMix 三域 | censor 三域 | shock 三域 | NoMix 电力 / 交通 / 太阳能 |', '|---|---|---:|---:|---:|---|']
    for k in ('MLP', 'PatchTST', 'DLinear'):
        L.append('| %s | %s | %s | %s | %s | %s |' % (k, ' / '.join('%.3f' % cc['E_None_by_domain'][k][d] for d in DOMAINS), *[f(cc['three_domain'][k][ph]) for ph in P],
                                                   ' / '.join(f(cc['by_domain'][k]['P_NoMixRecipe'][d]) for d in DOMAINS)))
    c = r['cost']
    L += ['', '## 5. 成本', '',
          '- LLM 0 次；拟合：主比较 %d + 校准 %d + 接线 %d。' % (c['fits']['main'], c['fits']['calibration'], c['fits']['wiring']),
          '- 单次拟合 %.2f s（带子视图 %.2f s），主比较训练合计 %.0f s，校准合计 %.0f s；并行 %d 个 CPU 进程，单进程峰值内存 %.0f MB；整包墙钟 %s。'
          % (c['train_seconds']['main_mean'], c['train_seconds']['main_child_mean'], c['train_seconds']['main_sum'], c['train_seconds']['calibration_sum'],
             c['workers'], c['peak_rss_mb_worker'], ('%.1f min' % (c['wall_s_package'] / 60)) if c['wall_s_package'] else '—'),
          '', '按任务书到此停止：不学新卡、不进入 Test。']
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--readout', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--fit-batch')
    a = ap.parse_args()
    if a.fit_batch:
        return fit_batch(a.fit_batch)
    if a.smoke:
        freeze_plan()
        print(json.dumps(smoke()))
    if a.run:
        run()
    if a.readout:
        readout()


if __name__ == '__main__':
    main()
