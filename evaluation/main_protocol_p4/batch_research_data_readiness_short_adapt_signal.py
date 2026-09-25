"""DEV-DATA-READINESS-SHORT-ADAPT-SIGNAL (docs/DEV_DATA_READINESS_SHORT_ADAPT_SIGNAL_TASK_2026-09-18.md).

A cheap material-value signal. From one common checkpoint theta0 (100 updates on P0 Linear) every material gets a fresh AdamW and
the same 25 parent batches; its C_A loss is compared with the equally adapted P0. The full Consumer (REF2000) and its 100-update
checkpoint (REF100) are read from the TRAINING-LENGTH cache. Only the training X changes: every model is served the P0 Linear inputs.
0 LLM, 0 API, 0 new 2000-update fits, no hashing.

  preflight   environment, job geometry, exposure records, cache inventory + snapshot, material rebuild check (worker), config frozen
  proxy       4 jobs x 3 proxy seeds: theta0 (100 updates) + 3 branches x 25 updates, T rows only; branch states at update 1 and 25
  checks      direct readiness 100-update fit == theta0; a lone P2 branch from the saved theta0 == the planned branch
  c_a         per job, rows [t-672, t+96): proxy models, theta0 and REF100/REF2000 x 3 materials x reference seeds, Linear inputs
  selections  R25 / R1 / R100 / R2000 / Always-P0 frozen in selections.json before any C_B/E row is loaded
  later       C_B/E predictions of the 120 reference models -> barrier -> C_B/E scores
  readout     result.json, tables.md

The controller never imports torch; each worker initializes OpenMP before torch and gets an environment without KMP_DUPLICATE_LIB_OK.
  --preflight | --smoke | --run | --result
"""
from __future__ import annotations

import os

# scikit-learn's __init__ sets KMP_DUPLICATE_LIB_OK in its own process; record the shell state before any import and never pass the
# variable to a worker (TRAINING-LENGTH side finding).
KMP_AT_START = os.environ.get('KMP_DUPLICATE_LIB_OK')

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import context, spec
from methods.ttha.batch_base import readiness as rd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_data_readiness_short_adapt_signal'
TL = REPO / '_scratch' / 'dev_data_readiness_training_length'
TASK = 'docs/DEV_DATA_READINESS_SHORT_ADAPT_SIGNAL_TASK_2026-09-18.md'
MODULE = 'evaluation.main_protocol_p4.batch_research_data_readiness_short_adapt_signal'
PRSA_CLOSED_FROM_ROW = 24864
JOBS = ('RD01B_Q1', 'RD02_T1', 'RD01B_STP1', 'RD02_STP1')
DOMAINS = ('RD01B', 'RD02')
CACHE = {'RD01B_Q1': TL / 'dev' / 'RD01B_Q1', 'RD02_T1': TL / 'dev' / 'RD02_T1',
         'RD01B_STP1': TL / 'follow' / 'RD01B_STP1', 'RD02_STP1': TL / 'follow' / 'RD02_STP1'}
DEV_SEEDS = (20261001, 20261002, 20261003, 20261004)
FOLLOW_SEEDS = (20261011, 20261012, 20261013, 20261014, 20261015, 20261016)
REF_SEEDS = {'RD01B_Q1': DEV_SEEDS, 'RD02_T1': DEV_SEEDS, 'RD01B_STP1': FOLLOW_SEEDS, 'RD02_STP1': FOLLOW_SEEDS}
GEOMETRY = {'RD01B_Q1': {'t': 8160, 'T': [7488, 8160], 'C_A': [8160, 8256], 'C_B': [8256, 8352], 'E': [8352, 8544]},
            'RD02_T1': {'t': 6960, 'T': [6288, 6960], 'C_A': [6960, 7056], 'C_B': [7056, 7152], 'E': [7152, 7344]},
            'RD01B_STP1': {'t': 10320, 'T': [9648, 10320], 'C_A': [10320, 10416], 'C_B': [10416, 10512], 'E': [10512, 10704]},
            'RD02_STP1': {'t': 14160, 'T': [13488, 14160], 'C_A': [14160, 14256], 'C_B': [14256, 14352], 'E': [14352, 14544]}}
EXPOSURE = {'RD01B_Q1': REPO / '_scratch' / 'dev_data_readiness_profile_transfer' / 'test' / 'RD01B_Q1_F0' / 'RD01B_Q1',
            'RD02_T1': REPO / '_scratch' / 'dev_data_readiness_domain_skill_v1' / 'target' / 'RD02_T1_no_skill' / 'RD02_T1',
            'RD01B_STP1': CACHE['RD01B_STP1'], 'RD02_STP1': CACHE['RD02_STP1']}
PROXY_SEEDS = (20261101, 20261102, 20261103)
START_UPDATES = 100
ADAPT_ROWS = (100, 125)                  # rows of the unchanged 2000 x 64 parent-batch stream of the proxy seed
ADAPT_UPDATES = ADAPT_ROWS[1] - ADAPT_ROWS[0]
PROXY_K = (1, 25)
REF_N = (100, 2000)
P2_POLICY = rd.uniform_policy([{'op': 'impute_linear', 'strength': 1.0}, {'op': 'hampel_filter', 'window': 5, 'n_sigmas': 3.0}],
                              'P2 Linear+Hampel: linear completion then Hampel(window=5, n_sigmas=3.0)')
MATERIALS = (('P0_Linear', rd.BASELINE_LINEAR), ('P1_Seasonal', rd.FIXED_SEASONAL), ('P2_Linear_Hampel', P2_POLICY))
MIDS = tuple(m for m, _ in MATERIALS)
PAIRS = (('P1_Seasonal', 'P0_Linear'), ('P2_Linear_Hampel', 'P0_Linear'))
RULES = ('R25', 'R1', 'R100', 'R2000', 'Always_P0')
COMPARISONS = (('R2000', 'R25'), ('Always_P0', 'R25'), ('R100', 'R25'))
BLOCKS = ('c_a', 'c_b', 'e')
CAPS = {'start': 12, 'adapt': 36, 'check': 2, 'retry': 2, 'hard': 52}
WALL_S = 3600.
WORKER_TIMEOUT_S = 900.
TIE_TOL = 1e-12
CHECK = ('RD02_T1', PROXY_SEEDS[0], 'P2_Linear_Hampel')
SELECTION_RULES = {
    'R25': 'mean over the 3 proxy seeds of the C_A macro loss (Linear inputs) of the branch after 25 updates; argmin over P0/P1/P2',
    'R1': 'same with the branch state after update 1 (sensitivity reading of the same trajectories)',
    'R100': 'mean over the reference seeds of the C_A macro loss (Linear inputs) of the cached 100-update checkpoint; argmin',
    'R2000': 'mean over the reference seeds of the C_A macro loss (Linear inputs) of the cached 2000-update model; argmin',
    'Always_P0': 'P0_Linear',
    'ties': 'a loss within 1e-12 of the minimum counts as tied; order P0 -> P1 -> P2 decides',
    'missing': 'a rule whose three material losses are not all SCORABLE selects nothing for that job (NOT_SELECTABLE)'}
READOUT_RULES = {
    'proxy_delta': 'd[j,p,s,k] = L_A(P0 branch after k) - L_A(p branch after k), same seed and theta0; positive = p better than P0. k=25 main, k=1 extra.',
    'reference_delta': 'd[j,p,s,n] = L_A(REF P0 model) - L_A(REF p model), same reference seed, all served the Linear inputs; n=2000 main, 100 sensitivity.',
    'resolution': 'reference cell REFERENCE_UNRESOLVED when the fixed-endpoint Student t 95% interval over reference seeds contains 0; the interval '
                  'covers training randomness of the cached Consumer only (not time, multiplicity or proxy error). Proxy cells get no interval.',
    'sign_agreement': 'sign(mean proxy d) == sign(mean reference d) per job/pair cell; named agreement with a finite-seed reference mean, not accuracy.',
    'selection_value': 'per job, per Consumer (REF2000 main, REF100 sensitivity): loss of the model trained on the selected material; paired per '
                       'reference seed, other rule minus R25 (positive = R25 better); identical materials give exact zeros.',
    'regret': 'selected material seed-mean loss minus the lowest seed-mean loss among the three materials (finite-seed diagnostic, not an oracle).',
    'pooling': 'relative = mean paired difference / REF2000 P0 seed-mean loss of the same job and block (for both Consumers); jobs equal within '
               'a domain, then domains equal. Raw per-job differences are kept.',
    'consumer_dependent': 'CONSUMER_DEPENDENT when the pooled (or per-job) mean difference has opposite nonzero signs under REF2000 and REF100.'}


def ds_of(job: str) -> str:
    return job.split('_')[0]


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def domain_jobs(jobs=JOBS) -> dict:
    return {d: [j for j in jobs if ds_of(j) == d] for d in DOMAINS}


def proxy_model_name(mid: str, seed: int, k: int) -> str:
    return '%s__s%d__k%d.pt' % (mid, seed, k)


# ----------------------------------------------------------------------------------------------------------- budget ledger
class BudgetStop(RuntimeError):
    pass


class Ledger:
    """Units are trajectories (a theta0 start or one 25-update branch), charged before the worker starts."""

    def __init__(self, root: Path):
        self.path = root / 'budget.json'
        self.d = context.read_json(self.path) if self.path.exists() else {
            'caps': CAPS, 'wall_s': WALL_S, 'first_launch_epoch': None, 'physical': {'start': 0, 'adapt': 0, 'check': 0}, 'retries_used': 0,
            'failed_workers': 0, 'events': []}

    def save(self):
        context.write_json(self.path, self.d)

    def total(self) -> int:
        return sum(self.d['physical'].values()) + self.d['retries_used']

    def reserve(self, units: list, name: str, retry: bool):
        """units: list of (stage, unit name)."""
        if self.d['first_launch_epoch'] is not None and time.time() - self.d['first_launch_epoch'] > WALL_S:
            raise BudgetStop('numeric run wall clock limit reached')
        if self.total() + len(units) > CAPS['hard']:
            raise BudgetStop('trajectory hard cap reached')
        if retry:
            if self.d['retries_used'] + len(units) > CAPS['retry']:
                raise BudgetStop('retry cap reached')
            self.d['retries_used'] += len(units)
        else:
            for stage, _ in units:
                if self.d['physical'][stage] + sum(1 for s, _ in units if s == stage) > CAPS[stage]:
                    raise BudgetStop('%s trajectory cap reached' % stage)
            for stage, _ in units:
                self.d['physical'][stage] += 1
        if self.d['first_launch_epoch'] is None:
            self.d['first_launch_epoch'] = time.time()
        self.d['events'].append({'kind': 'worker_started', 'name': name, 'units': [list(u) for u in units], 'retry': retry, 'epoch': time.time()})
        self.save()

    def finish(self, name: str, ok: bool, seconds: float, reason: str = '', units_done=()):
        if not ok:
            self.d['failed_workers'] += 1
        self.d['events'].append({'kind': 'worker_finished', 'name': name, 'ok': ok, 'seconds': seconds, 'reason': reason,
                                 'units_done': list(units_done), 'epoch': time.time()})
        self.save()

    def stage(self, name: str, ok: bool, seconds: float, reason: str = ''):
        if self.d['first_launch_epoch'] is not None and time.time() - self.d['first_launch_epoch'] > WALL_S:
            raise BudgetStop('numeric run wall clock limit reached')
        self.d['events'].append({'kind': 'stage_worker', 'name': name, 'ok': ok, 'seconds': seconds, 'reason': reason, 'epoch': time.time()})
        self.save()


# ----------------------------------------------------------------------------------------------------------- preflight
def worker_env() -> dict:
    env = dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    return env


PROBE = '''
import json, os, sys
from SelfEvolvingHarnessTS.operators import _provenance as p
fp = p.dependency_fingerprint()
import torch
opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3, weight_decay=1e-4)
d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.defaults.items()}
print(json.dumps({"fp": fp, "torch": torch.__version__, "python": sys.version.split()[0], "adamw_defaults": d, "threads": torch.get_num_threads()}))
'''


def environment() -> dict:
    import inspect
    parent = context.read_json(TL / 'frozen_config.json')['environment']
    probe = subprocess.run([sys.executable, '-B', '-c', PROBE], cwd=str(REPO), capture_output=True, text=True, timeout=300, env=worker_env())
    if probe.returncode != 0:
        raise RuntimeError('environment probe failed: %s' % probe.stderr[-600:])
    got = json.loads(probe.stdout.strip().splitlines()[-1])
    main_src = inspect.getsource(rd.main)
    out = {'executable': sys.executable, 'python': got['python'], 'torch': got['torch'], 'dependency_fingerprint': got['fp'],
           'torch_threads': got['threads'], 'adamw_defaults_probe': got['adamw_defaults'],
           'probe': 'fingerprint, torch version and AdamW defaults read in a separate process; the controller never imports scikit-learn or torch',
           'parent': 'TRAINING-LENGTH frozen_config environment', 'kmp_duplicate_lib_ok_in_shell_at_start': KMP_AT_START,
           'kmp_duplicate_lib_ok_in_controller_now': os.environ.get('KMP_DUPLICATE_LIB_OK'),
           'worker_env_kmp_duplicate_lib_ok': worker_env().get('KMP_DUPLICATE_LIB_OK'),
           'readiness_worker_initializes_openmp_before_torch': '_init_openmp_before_torch()' in main_src.split('ap = argparse')[0],
           'controller_imports_torch': 'torch' in sys.modules}
    ok = (got['fp'] == parent['dependency_fingerprint'] and out['python'] == parent['python'] and out['torch'] == parent['torch']
          and Path(sys.executable).resolve() == Path(parent['executable']).resolve() and KMP_AT_START is None
          and out['worker_env_kmp_duplicate_lib_ok'] is None and out['readiness_worker_initializes_openmp_before_torch'] and not out['controller_imports_torch'])
    if not ok:
        raise RuntimeError('environment differs from TRAINING-LENGTH: %s' % out)
    return out


def geometry(js) -> dict:
    return {'t': js.t, 'T': list(js.train_range), 'C_A': [min(js.c_a), max(js.c_a) + rd.H], 'C_B': [min(js.c_b), max(js.c_b) + rd.H],
            'E': [min(js.e), max(js.e) + rd.H]}


def structure(job: str) -> dict:
    """T eligibility and C_A coverage (rows [t-672, t+96) only)."""
    ds, js = ds_of(job), rd.resolve_job(ds_of(job), job)
    if geometry(js) != GEOMETRY[job]:
        raise RuntimeError('%s geometry differs from the task table: %s' % (job, geometry(js)))
    if ds == 'RD02' and max(js.e) + rd.H > PRSA_CLOSED_FROM_ROW:
        raise RuntimeError('%s reaches the closed PRSA 2016+ region' % job)
    lo, hi = js.evaluate_rows
    sl = rd.load_slice(ds, lo, hi)
    seg = sl.rows(*js.train_range)
    fin = np.isfinite(seg)
    lg = rd.legal_parents(seg)
    bad = [e for e in range(seg.shape[1]) if fin[:, e].sum() < rd.MIN_T_FINITE or not seg[fin[:, e], e].std() > spec.ZERO_SCALE_FLOOR
           or lg.counts[e] < rd.MIN_LEGAL_PARENTS]
    obs = np.isfinite(rd.read_truth(sl, js.c_a)).sum(axis=(1, 2)) / (len(js.c_a) * rd.H)
    row = {'job': job, 'dataset': ds, 'geometry': geometry(js), 'rows_read': [lo, hi], 'n_entities': seg.shape[1], 'legal_parents': lg.n,
           'ineligible_entities': bad, 'c_a_min_observed_fraction': float(obs.min()), 'c_a_not_scorable_entities': [int(e) for e in np.flatnonzero(obs < rd.COVERAGE_MIN)],
           'purpose': 'T eligibility and C_A mask coverage; C_B/E rows are not loaded before the selections are frozen'}
    row['status'] = 'ELIGIBLE' if not bad and not row['c_a_not_scorable_entities'] else 'INELIGIBLE'
    return row


def exposure_identity(job: str) -> dict:
    d, js = EXPOSURE[job], rd.resolve_job(ds_of(job), job)
    es = context.read_json(d / 'e_scores.json')
    with np.load(d / 'scaler.npz') as sc:
        binding = (str(sc['profile']) == rd.PROFILE and str(sc['dataset']) == js.dataset and int(sc['t']) == js.t
                   and [str(x) for x in sc['roster']] == js.roster)
    ok = es['job_id'] == job and es['rows_read'] == GEOMETRY[job]['E'] and binding
    if not ok:
        raise RuntimeError('exposure record of %s does not bind to the job: %s' % (job, d))
    return {'record': str(d / 'e_scores.json'), 'job_id': es['job_id'], 'e_rows_read': es['rows_read'], 'scored_at_local': es['scored_at_local'],
            'n_cells': len(es['cells']), 'scaler_binding_profile_dataset_t_roster': binding, 'status': 'EXPOSED_DEVELOPMENT'}


def ref_paths(cache_dir: Path, job: str, mid: str, seed: int, n: int) -> tuple:
    cid = rd.length_cell_id(job, mid, seed, n)
    return cid, cache_dir / 'runs' / (cid + '.pt'), cache_dir / 'cells' / (cid + '.json')


def cache_inventory(job: str) -> dict:
    cdir, rows = CACHE[job], {}
    for mid in MIDS:
        for seed in REF_SEEDS[job]:
            for n in REF_N:
                cid, mp, cp = ref_paths(cdir, job, mid, seed, n)
                ok = mp.exists() and cp.exists()
                if ok:
                    c = context.read_json(cp)
                    ok = (c.get('status') == 'OK' and c['job_id'] == job and c['material_id'] == mid and c['model_seed'] == seed
                          and rd.cell_trained_updates(c) == n and Path(c['model_path']).resolve() == mp.resolve())
                rows[cid] = {'model': str(mp), 'cell': str(cp), 'bound': ok}
    return {'cache_dir': str(cdir), 'models_expected': len(MIDS) * len(REF_SEEDS[job]) * len(REF_N), 'models_bound': sum(r['bound'] for r in rows.values()),
            'models': rows}


def cache_files() -> list:
    files = []
    for job in JOBS:
        c = CACHE[job]
        for mid in MIDS:
            for seed in REF_SEEDS[job]:
                for n in REF_N:
                    _, mp, cp = ref_paths(c, job, mid, seed, n)
                    files += [mp, cp]
        files += sorted((c / 'materials').glob('*')) + [c / 'scaler.npz'] + [c / f for f in ('c_b_scores.json', 'e_scores.json') if (c / f).exists()]
    for job in JOBS:
        files.append(EXPOSURE[job] / 'e_scores.json')
    return sorted(set(files))


def snapshot(files) -> dict:
    return {str(p): ([p.stat().st_size, p.stat().st_mtime_ns] if p.exists() else None) for p in files}


def run_worker(cmd: list, log: Path, timeout: float = WORKER_TIMEOUT_S) -> tuple:
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        with log.open('w', encoding='utf-8') as fh:
            p = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, env=worker_env())
        return p.returncode, time.time() - start, ''
    except subprocess.TimeoutExpired:
        return -999, time.time() - start, 'worker_timeout'


def worker_cmd(*args) -> list:
    return [sys.executable, '-B', '-m', MODULE] + [str(a) for a in args]


def preflight() -> dict:
    p = ROOT / 'frozen_config.json'
    if p.exists():
        return context.read_json(p)
    (ROOT / 'checks').mkdir(parents=True, exist_ok=True)
    env = environment()
    struct = {j: structure(j) for j in JOBS}
    exposure = {j: exposure_identity(j) for j in JOBS}
    inventory = {j: cache_inventory(j) for j in JOBS}
    if any(v['models_bound'] != v['models_expected'] for v in inventory.values()):
        raise RuntimeError('reference cache incomplete or unbound: %s' % {j: (v['models_bound'], v['models_expected']) for j, v in inventory.items()})
    snap = snapshot(cache_files())
    context.write_json(ROOT / 'checks' / 'cache_snapshot_before.json', {'written_local': now(), 'files': snap})
    mc_path = ROOT / 'checks' / 'material_rebuild.json'
    if not mc_path.exists():
        rc, secs, reason = run_worker(worker_cmd('--worker-material-check'), ROOT / 'checks' / 'material_rebuild.log')
        if rc != 0 or not mc_path.exists():
            raise RuntimeError('material rebuild check failed (%s)' % (reason or rc))
        mc = context.read_json(mc_path)
        mc['worker_wall_seconds'] = secs
        context.write_json(mc_path, mc)
    mc = context.read_json(mc_path)
    if not all(r['scaler_and_legal_order_equal'] and all(m['X_equal_to_cache'] and m['key_equal_to_cache'] and m['alias_of'] is None
                                                         for m in r['materials'].values()) for r in mc['jobs'].values()):
        raise RuntimeError('rebuilt materials differ from the cache: %s' % mc_path)
    mats = {}
    for mid, pol in MATERIALS:
        comp = rd.compile_policy(pol, [{}] * len(rd.DATASETS['RD02']['roster']))
        mats[mid] = {'policy': pol, 'canonical_steps': comp['assignment'][0]}
    cfg = {'package': 'DEV-DATA-READINESS-SHORT-ADAPT-SIGNAL', 'task': TASK, 'frozen_epoch': time.time(), 'frozen_local': now(),
           'exposure': 'EXPOSED_DEVELOPMENT for all four jobs; freezing before the delayed scores only prevents re-tuning inside this package',
           'llm_calls': 0, 'api_calls': 0, 'new_hashes': 0, 'new_full_2000_update_fits': 0, 'environment': env,
           'jobs': list(JOBS), 'geometry': GEOMETRY, 'structure': struct, 'exposure_identity': exposure,
           'sealing': {'prsa_closed_from_row': PRSA_CLOSED_FROM_ROW, 'max_prsa_row_to_read': max(GEOMETRY[j]['E'][1] for j in JOBS if ds_of(j) == 'RD02'),
                       'max_kdd_row_to_read': max(GEOMETRY[j]['E'][1] for j in JOBS if ds_of(j) == 'RD01B'), 'natural_final_read': False},
           'materials': mats, 'material_rebuild_check': {'path': str(mc_path), 'all_equal': True},
           'reference_cache': {j: {k: v for k, v in inventory[j].items() if k != 'models'} for j in JOBS},
           'reference_models_total': sum(v['models_bound'] for v in inventory.values()), 'reference_seeds': {j: list(REF_SEEDS[j]) for j in JOBS},
           'reference_updates': list(REF_N), 'cache_snapshot': str(ROOT / 'checks' / 'cache_snapshot_before.json'),
           'proxy': {'seeds': list(PROXY_SEEDS), 'start': {'material': 'P0_Linear', 'updates': START_UPDATES, 'batch_rows': [0, START_UPDATES],
                                                           'init': 'train.train_arm from torch.manual_seed(seed), as a direct 100-update fit'},
                     'branches': {'materials': list(MIDS), 'from': 'a copy of theta0 (parameters and buffers)', 'optimizer': 'fresh AdamW per branch',
                                  'lr': spec.LR, 'weight_decay': spec.WEIGHT_DECAY, 'adamw_defaults': env['adamw_defaults_probe'], 'parent_batch': spec.PARENT_BATCH,
                                  'batch_rows': list(ADAPT_ROWS), 'batch_stream': 'train.batch_indices(seed, n_pool)[100:125], identical for P0/P1/P2',
                                  'updates': ADAPT_UPDATES, 'states_saved': list(PROXY_K), 'main_k': 25, 'loss': 'pooled MSE, as train_arm'},
                     'serving': 'every proxy and reference model is served the P0 Linear inputs (empty program); training y, scaler, entities fixed'},
           'selection_rules': SELECTION_RULES, 'readout_rules': READOUT_RULES, 'comparisons': [list(c) for c in COMPARISONS],
           'caps': CAPS, 'wall_s_numeric': WALL_S,
           'wiring_checks': {'check_1': 'direct readiness worker fit (--n-updates 100 --no-feedback) of %s P0_Linear seed %d on the rebuilt materials '
                                        'versus theta0 (exact weights)' % CHECK[:2],
                             'check_2': 'a lone %s branch from the saved theta0 of %s seed %d in its own process versus the planned branch at '
                                        'update 1 and 25 (exact weights)' % (CHECK[2], CHECK[0], CHECK[1])}}
    context.write_json(p, cfg)
    return cfg


# ----------------------------------------------------------------------------------------------------------- workers (torch)
def bound_arrays(ctx, cache_dir: Path) -> tuple:
    """Training X of the three cached materials and y, after binding scaler, legal parent order and material keys to the cache."""
    lg, sc = ctx.legal, ctx.scaler
    with np.load(cache_dir / 'scaler.npz') as old:
        rd._check_binding(old, ctx.job)
        same = (np.array_equal(old['mean'], sc.mean) and np.array_equal(old['scale'], sc.scale) and np.array_equal(old['legal_ent'], lg.ent)
                and np.array_equal(old['legal_k'], lg.k))
    if not same:
        raise RuntimeError('scaler / legal parent order differs from the cache for %s' % ctx.job.job_id)
    index = context.read_json(cache_dir / 'materials' / 'index.json')
    N = len(ctx.job.roster)
    X = {}
    for mid, pol in MATERIALS:
        key = json.dumps(rd.compile_policy(pol, [{}] * N)['assignment'], sort_keys=True, separators=(',', ':'))
        ref, path = index[mid], cache_dir / 'materials' / (mid + '.npz')
        if ref['key'] != key or ref['alias_of'] is not None or Path(ref['path']).resolve() != path.resolve():
            raise RuntimeError('cached material %s of %s does not bind' % (mid, ctx.job.job_id))
        with np.load(path) as z:
            X[mid] = z['X'].copy()
        if X[mid].shape != lg.X_raw.shape or not np.isfinite(X[mid]).all():
            raise RuntimeError('cached material %s does not cover the legal parents' % mid)
    y = (lg.y_raw - sc.mean[lg.ent][:, None]) / sc.scale[lg.ent][:, None]
    return X, y, {'scaler_and_legal_order_equal': True, 'material_keys_bound': True, 'legal_parents': lg.n}


def adapt(state0: dict, X: np.ndarray, y: np.ndarray, rows: np.ndarray, checkpoints=(), on_checkpoint=None):
    """Continue from a copy of state0 with a fresh AdamW on the given parent-batch rows; same pooled loss as train.train_arm.
    on_checkpoint(step, model, batch_loss, seconds) runs after the opt.step() of `step` and must only copy state."""
    from methods.ttha.batch_base import train
    torch, _ = train._torch()
    dev = train.device()
    model = train.make_model()().to(dev)
    model.load_state_dict({k: v.clone() for k, v in state0.items()})
    opt = torch.optim.AdamW(model.parameters(), lr=spec.LR, weight_decay=spec.WEIGHT_DECAY)
    Xp = torch.as_tensor(X, dtype=torch.float32, device=dev)
    Yp = torch.as_tensor(y, dtype=torch.float32, device=dev)
    losses, t0 = [], time.time()
    for step in range(len(rows)):
        idx = torch.as_tensor(rows[step], dtype=torch.long, device=dev)
        opt.zero_grad(set_to_none=True)
        loss = ((model(Xp[idx]) - Yp[idx]) ** 2).mean(dim=-1).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step + 1 in checkpoints:
            on_checkpoint(step + 1, model, losses[-1], time.time() - t0)
    return model, {'batch_losses': losses, 'seconds': time.time() - t0,
                   'optimizer_defaults': {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.defaults.items()}}


def _peak_mb() -> float | None:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, 'peak_wset', getattr(mi, 'rss', 0)) / 2 ** 20, 1)
    except Exception:
        return None


def proxy_train(job: str, seed: int, study: Path, cache_dir: Path) -> dict:
    """theta0 (100 updates on P0) and three 25-update branches for one job/seed; T rows only. Resumable per unit."""
    t_entry, kmp_entry = time.time(), os.environ.get('KMP_DUPLICATE_LIB_OK')
    rd._init_openmp_before_torch()
    kmp_before_torch = os.environ.get('KMP_DUPLICATE_LIB_OK')
    import torch
    from methods.ttha.batch_base import train
    t_import = time.time()
    ctx = rd.open_job(ds_of(job), job, study / 'jobs', stage='material')
    X, y, bind = bound_arrays(ctx, cache_dir)
    t_ready = time.time()
    pdir = ctx.job_dir / 'proxy'
    mdir = pdir / 'models'
    mdir.mkdir(parents=True, exist_ok=True)
    rp = pdir / ('proxy__s%d.json' % seed)
    rec = context.read_json(rp) if rp.exists() else {'job_id': job, 'seed': seed, 'rows_read': list(ctx.job.material_rows), 'units': {}, 'attempts': []}
    att = {'started_epoch': t_entry, 'kmp_duplicate_lib_ok_at_worker_entry': kmp_entry, 'kmp_duplicate_lib_ok_before_torch': kmp_before_torch,
           'import_seconds': t_import - t_entry, 'open_and_bind_seconds': t_ready - t_import, 'binding': bind, 'units_run': []}
    full = train.batch_indices(seed, n_pool=ctx.legal.n)
    rows = full[ADAPT_ROWS[0]:ADAPT_ROWS[1]]
    t0p = mdir / ('theta0__s%d.pt' % seed)
    if 'start' in rec['units'] and t0p.exists():
        state0 = torch.load(t0p, map_location='cpu')
    else:
        ts = time.time()
        model0, tres = train.train_arm(X['P0_Linear'], y, None, full[:START_UPDATES], model_seed=seed, n_updates=START_UPDATES)
        call_s = time.time() - ts
        state0 = {k: v.detach().clone() for k, v in model0.state_dict().items()}
        ts = time.time()
        train.save_model(model0, t0p)
        rec['units']['start'] = {'material': 'P0_Linear', 'model_path': str(t0p), 'updates': START_UPDATES, 'batch_rows': [0, START_UPDATES],
                                 'train_seconds': tres.seconds, 'call_seconds': call_s, 'save_seconds': time.time() - ts,
                                 'final_batch_loss': tres.final_train_loss, 'loss_curve': tres.loss_curve, 'finished_epoch': time.time()}
        att['units_run'].append('start')
        context.write_json(rp, rec)
    changed = {mid: (X[mid] != X['P0_Linear']).any(axis=1) for mid in MIDS}
    for mid in MIDS:
        if mid in rec['units']:
            continue
        keep = {}

        def on_ck(step, model, loss, secs):
            keep[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}

        ts = time.time()
        model, ares = adapt(state0, X[mid], y, rows, checkpoints=(1,), on_checkpoint=on_ck)
        call_s = time.time() - ts
        ts = time.time()
        paths = {}
        for k in PROXY_K:
            path = mdir / proxy_model_name(mid, seed, k)
            torch.save(keep[1] if k == 1 else model.state_dict(), path)
            paths[str(k)] = str(path)
        save_s = time.time() - ts
        mask = changed[mid]
        drawn = {}
        for k in PROXY_K:
            d = rows[:k].ravel()
            drawn[str(k)] = {'draws': int(d.size), 'changed_draws': int(mask[d].sum()), 'unique_changed_windows': int(np.unique(d[mask[d]]).size)}
        rec['units'][mid] = {'material': mid, 'from': str(t0p), 'batch_rows': list(ADAPT_ROWS), 'updates': ADAPT_UPDATES, 'model_paths': paths,
                             'train_seconds': ares['seconds'], 'call_seconds': call_s, 'save_seconds': save_s, 'batch_losses': ares['batch_losses'],
                             'optimizer_defaults': ares['optimizer_defaults'], 'fresh_optimizer': True,
                             'material_windows_changed_vs_p0': int(mask.sum()), 'material_windows': int(mask.size), 'drawn': drawn,
                             'finished_epoch': time.time()}
        att['units_run'].append(mid)
        context.write_json(rp, rec)
    disk0 = torch.load(t0p, map_location='cpu')
    att['theta0_unchanged_after_branches'] = all(torch.equal(state0[k], disk0[k]) for k in state0) and list(state0) == list(disk0)
    att['kmp_duplicate_lib_ok_at_worker_end'] = os.environ.get('KMP_DUPLICATE_LIB_OK')
    att['peak_working_set_mb'] = _peak_mb()
    att['torch_threads'] = torch.get_num_threads()
    att['body_seconds'] = time.time() - t_entry
    rec['attempts'].append(att)
    rec['complete'] = all(u in rec['units'] for u in ('start',) + MIDS) and att['theta0_unchanged_after_branches']
    context.write_json(rp, rec)
    return rec


def adapt_only(job: str, seed: int, mid: str, study: Path, cache_dir: Path, out_dir: Path) -> dict:
    """Wiring check 2: one branch from the saved theta0 in its own process."""
    rd._init_openmp_before_torch()
    import torch
    from methods.ttha.batch_base import train
    ctx = rd.open_job(ds_of(job), job, study / 'jobs', stage='material')
    X, y, _ = bound_arrays(ctx, cache_dir)
    state0 = torch.load(ctx.job_dir / 'proxy' / 'models' / ('theta0__s%d.pt' % seed), map_location='cpu')
    rows = train.batch_indices(seed, n_pool=ctx.legal.n)[ADAPT_ROWS[0]:ADAPT_ROWS[1]]
    keep = {}
    model, ares = adapt(state0, X[mid], y, rows, checkpoints=(1,), on_checkpoint=lambda s, m, l, t: keep.__setitem__(s, {k: v.detach().clone() for k, v in m.state_dict().items()}))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k in PROXY_K:
        paths[str(k)] = str(out_dir / proxy_model_name(mid, seed, k))
        torch.save(keep[1] if k == 1 else model.state_dict(), paths[str(k)])
    rec = {'job': job, 'seed': seed, 'material': mid, 'model_paths': paths, 'train_seconds': ares['seconds']}
    context.write_json(out_dir / 'adapt_only.json', rec)
    return rec


def compact(s: dict) -> dict:
    ok = s['status'] == 'SCORABLE'
    return {'status': s['status'], 'macro': s['normalized_mse_macro'],
            'per_origin_macro': [statistics.fmean(x for x in row if x is not None) for row in s['per_origin_entity_normalized_mse']] if ok else None,
            'min_observed_fraction': s['coverage']['min_observed_fraction'], 'n_nonfinite': s['n_nonfinite_predictions']}


def ca_eval(job: str, study: Path, cache_dir: Path, proxy_seeds=PROXY_SEEDS, ref_seeds=None, ref_n=REF_N) -> dict:
    """C_A of every proxy state, theta0 and reference model on the Linear inputs; rows [t-672, t+96) only."""
    t_entry = time.time()
    rd._init_openmp_before_torch()
    from methods.ttha.batch_base import train
    t_import = time.time()
    ref_seeds = REF_SEEDS[job] if ref_seeds is None else ref_seeds
    jd = study / 'jobs' / job
    recs = {s: context.read_json(jd / 'proxy' / ('proxy__s%d.json' % s)) for s in proxy_seeds}
    if not all(r.get('complete') for r in recs.values()):
        raise RuntimeError('proxy states of %s incomplete; C_A is evaluated only after every branch exists' % job)
    ctx = rd.open_job(ds_of(job), job, study / 'jobs', stage='evaluate')
    js = ctx.job
    if ctx.slice.row_end > min(js.c_b):
        raise PermissionError('C_A stage slice reaches C_B rows')
    t_open = time.time()
    N = len(js.roster)
    Xs, serve_rec = rd.serve_inputs(ctx.slice, js.c_a, [[] for _ in range(N)], ctx.scaler)
    truth = rd.read_truth(ctx.slice, js.c_a)
    t_serve = time.time()
    preds, timers = {}, {'theta0': [], 'proxy_k1': [], 'proxy_k25': [], 'ref': []}
    out = {'job_id': job, 'rows_read': list(js.evaluate_rows), 'slice_row_end': ctx.slice.row_end, 'serving': 'Linear inputs (empty program) for every model',
           'serve_inputs': serve_rec, 'theta0': {}, 'proxy': {}, 'ref': {}, 'cache_consistency_p0_c_a': []}

    def run(path, key, timer):
        t = time.time()
        model = train.load_model(path)
        pred = rd.predict_inputs(model, Xs, ctx.scaler)
        s = compact(rd.score_block(pred, truth, ctx.scaler))
        preds[key] = pred
        timers[timer].append(time.time() - t)
        return s

    for seed in proxy_seeds:
        out['theta0'][str(seed)] = run(recs[seed]['units']['start']['model_path'], 'theta0__s%d' % seed, 'theta0')
        for mid in MIDS:
            for k in PROXY_K:
                out['proxy'].setdefault(str(k), {}).setdefault(mid, {})[str(seed)] = run(recs[seed]['units'][mid]['model_paths'][str(k)],
                                                                                        'proxy__%s__s%d__k%d' % (mid, seed, k), 'proxy_k%d' % k)
    for n in ref_n:
        for mid in MIDS:
            for seed in ref_seeds:
                cid, mp, cp = ref_paths(cache_dir, job, mid, seed, n)
                s = run(mp, 'ref__' + cid, 'ref')
                out['ref'].setdefault(str(n), {}).setdefault(mid, {})[str(seed)] = s
                if mid == 'P0_Linear':
                    cached = (context.read_json(cp).get('scores', {}).get('c_a') or {}).get('normalized_mse_macro')
                    out['cache_consistency_p0_c_a'].append({'cell': cid, 'cached': cached, 'recomputed': s['macro'], 'equal': cached == s['macro']})
    np.savez(jd / 'ca_predictions.npz', origins=np.array(js.c_a), **preds)
    out['timing'] = {'import_seconds': t_import - t_entry, 'open_job_seconds': t_open - t_import, 'serve_and_truth_seconds': t_serve - t_open,
                     'per_model_seconds': {k: v for k, v in timers.items()}, 'body_seconds': time.time() - t_entry}
    out['peak_working_set_mb'] = _peak_mb()
    out['finished_epoch'] = time.time()
    context.write_json(jd / 'ca_scores.json', out)
    return out


def later_predict(job: str, study: Path, cache_dir: Path, ref_seeds=None, ref_n=REF_N) -> dict:
    """C_B / E predictions of the reference models on the Linear inputs; refused before the selections are frozen."""
    sel_path = study / 'selections.json'
    if not sel_path.exists():
        raise PermissionError('selections are not frozen; no C_B/E row is loaded')
    sel = context.read_json(sel_path)
    if job not in sel['jobs']:
        raise PermissionError('job %s is not in the frozen selections' % job)
    t_entry = time.time()
    rd._init_openmp_before_torch()
    from methods.ttha.batch_base import train
    ref_seeds = REF_SEEDS[job] if ref_seeds is None else ref_seeds
    ds, js = ds_of(job), rd.resolve_job(ds_of(job), job)
    ctx_e = rd.open_job(ds, job, study / 'jobs', stage='e_input')              # frozen scaler; E input rows only
    cb_rows = (min(js.c_b) - rd.L, max(js.c_b))                                 # C_B input rows only
    sl_cb = rd.load_slice(ds, *cb_rows)
    N = len(js.roster)
    Xcb, rec_cb = rd.serve_inputs(sl_cb, js.c_b, [[] for _ in range(N)], ctx_e.scaler)
    Xe, rec_e = rd.serve_inputs(ctx_e.slice, js.e, [[] for _ in range(N)], ctx_e.scaler)
    preds, models, per_model = {}, [], []
    for n in ref_n:
        for mid in MIDS:
            for seed in ref_seeds:
                cid, mp, _ = ref_paths(cache_dir, job, mid, seed, n)
                t = time.time()
                model = train.load_model(mp)
                preds['c_b__' + cid] = rd.predict_inputs(model, Xcb, ctx_e.scaler)
                preds['e__' + cid] = rd.predict_inputs(model, Xe, ctx_e.scaler)
                per_model.append(time.time() - t)
                models.append({'cell_id': cid, 'material_id': mid, 'model_seed': seed, 'n_updates': n, 'model_path': str(mp)})
    jd = study / 'jobs' / job
    np.savez(jd / 'later_predictions.npz', c_b_origins=np.array(js.c_b), e_origins=np.array(js.e), **preds)
    rec = {'job_id': job, 'selections_frozen_epoch': sel['frozen_epoch'], 'started_epoch': t_entry, 'rows_read': {'c_b_inputs': list(cb_rows), 'e_inputs': list(js.e_input_rows)},
           'serving': 'Linear inputs for every model', 'serve_inputs': {'c_b': rec_cb, 'e': rec_e}, 'models': models,
           'per_model_seconds': per_model, 'body_seconds': time.time() - t_entry, 'peak_working_set_mb': _peak_mb(), 'frozen_epoch': time.time(), 'frozen_local': now()}
    context.write_json(jd / 'later_frozen.json', rec)
    return rec


def later_score(job: str, study: Path, cache_dir: Path) -> dict:
    """C_B / E scores after the whole-package prediction barrier."""
    barrier = study / 'later_barrier.json'
    if not barrier.exists() or job not in context.read_json(barrier)['jobs']:
        raise PermissionError('prediction barrier missing or job not in it; no C_B/E target row is loaded')
    t_entry = time.time()
    ds, js = ds_of(job), rd.resolve_job(ds_of(job), job)
    jd = study / 'jobs' / job
    fz = context.read_json(jd / 'later_frozen.json')
    ctx_cb = rd.open_job(ds, job, study / 'jobs', stage='c_b')
    ctx_et = rd.open_job(ds, job, study / 'jobs', stage='e_target')
    y_cb, y_e = rd.read_truth(ctx_cb.slice, js.c_b), rd.read_truth(ctx_et.slice, js.e)
    cached = {}
    for f, key in (('c_b_scores.json', 'c_b'), ('e_scores.json', 'e')):
        if (cache_dir / f).exists():
            cached[key] = context.read_json(cache_dir / f)['cells']
    out = {'job_id': job, 'barrier_epoch': context.read_json(barrier)['written_epoch'], 'rows_read': {'c_b': list(js.c_b_rows), 'e_target': list(js.e_target_rows)},
           'scores': {}, 'cache_consistency_shadow': []}
    with np.load(jd / 'later_predictions.npz') as z:
        if not (np.array_equal(z['c_b_origins'], np.array(js.c_b)) and np.array_equal(z['e_origins'], np.array(js.e))):
            raise RuntimeError('frozen origins differ from the job table')
        for m in fz['models']:
            cid = m['cell_id']
            row = {'c_b': compact(rd.score_block(z['c_b__' + cid], y_cb, ctx_cb.scaler)), 'e': compact(rd.score_block(z['e__' + cid], y_e, ctx_et.scaler))}
            out['scores'][cid] = row
            for b, shadow_key in (('c_b', 'c_b_shadow_baseline_inputs'), ('e', 'e_shadow_baseline_inputs')):
                if b in cached and cid in cached[b]:
                    old = (cached[b][cid].get(shadow_key) or {}).get('normalized_mse_macro')
                    out['cache_consistency_shadow'].append({'cell': cid, 'block': b, 'cached_shadow': old, 'recomputed': row[b]['macro'], 'equal': old == row[b]['macro']})
    out['body_seconds'] = time.time() - t_entry
    out['finished_epoch'] = time.time()
    context.write_json(jd / 'later_scores.json', out)
    return out


def material_check(study: Path, jobs=JOBS) -> dict:
    """Rebuild the three materials from T with the frozen programs and compare them with the cached arrays (preflight)."""
    out = {'jobs': {}, 'started_local': now()}
    for job in jobs:
        t0 = time.time()
        ctx = rd.open_job(ds_of(job), job, study / 'checks' / 'material_rebuild', stage='material')
        rec = {'open_job_seconds': time.time() - t0, 'legal_parents': ctx.legal.n, 'materials': {}}
        with np.load(CACHE[job] / 'scaler.npz') as old:
            rec['scaler_and_legal_order_equal'] = bool(np.array_equal(old['mean'], ctx.scaler.mean) and np.array_equal(old['scale'], ctx.scaler.scale)
                                                       and np.array_equal(old['legal_ent'], ctx.legal.ent) and np.array_equal(old['legal_k'], ctx.legal.k))
        old_index = context.read_json(CACHE[job] / 'materials' / 'index.json')
        N = len(ctx.job.roster)
        for mid, pol in MATERIALS:
            idx = rd.load_index(ctx.job_dir)
            ts = time.time()
            ref = rd.MaterialRef(**idx[mid]) if mid in idx else rd.build(ctx, rd.compile_policy(pol, [{}] * N)['assignment'], mid)
            secs = time.time() - ts
            with np.load(ref.path) as a, np.load(CACHE[job] / 'materials' / (mid + '.npz')) as b:
                same = bool(np.array_equal(a['X'], b['X']))
            rec['materials'][mid] = {'build_seconds': secs, 'X_equal_to_cache': same, 'key_equal_to_cache': ref.key == old_index[mid]['key'],
                                     'alias_of': ref.alias_of}
        out['jobs'][job] = rec
    out['finished_local'] = now()
    context.write_json(study / 'checks' / 'material_rebuild.json', out)
    return out


def _compare_models_main(a: str, b: str) -> None:
    rd._init_openmp_before_torch()
    import torch
    sa, sb = torch.load(a, map_location='cpu'), torch.load(b, map_location='cpu')
    same_keys = list(sa) == list(sb)
    diffs = {k: float((sa[k].double() - sb[k].double()).abs().max()) for k in sa} if same_keys else {}
    print(json.dumps({'same_keys': same_keys, 'exact_equal': same_keys and all(torch.equal(sa[k], sb[k]) for k in sa),
                      'max_abs_diff': max(diffs.values()) if diffs else None}))


def compare_models(a, b) -> dict:
    p = subprocess.run(worker_cmd('--compare-models', a, b), cwd=str(REPO), capture_output=True, text=True, timeout=120, env=worker_env())
    if p.returncode != 0:
        raise RuntimeError('model comparison failed: %s' % p.stderr[-800:])
    return json.loads(p.stdout.strip().splitlines()[-1])


# ----------------------------------------------------------------------------------------------------------- selections
def pick(losses: dict):
    """losses {mid: value}; None when any is missing; ties within 1e-12 -> P0, P1, P2 order."""
    if any(losses.get(m) is None for m in MIDS):
        return None
    best = min(losses[m] for m in MIDS)
    return next(m for m in MIDS if losses[m] - best <= TIE_TOL)


def rule_losses(ca: dict, proxy_seeds, ref_seeds, ref_n) -> dict:
    """{rule: {mid: seed-mean C_A macro or None}}."""
    def mean(block):
        vals = [block[str(s)]['macro'] for s in block] if block else []
        return statistics.fmean(vals) if vals and all(v is not None for v in vals) else None

    src = {'R25': ca['proxy']['25'], 'R1': ca['proxy']['1'], 'R100': ca['ref'][str(ref_n[0])], 'R2000': ca['ref'][str(ref_n[1])]}
    expected = {'R25': len(proxy_seeds), 'R1': len(proxy_seeds), 'R100': len(ref_seeds), 'R2000': len(ref_seeds)}
    return {r: {m: (mean(src[r].get(m)) if len(src[r].get(m, {})) == expected[r] else None) for m in MIDS} for r in src}


def select(study: Path, jobs=JOBS, proxy_seeds=PROXY_SEEDS, ref_seeds=None, ref_n=REF_N) -> dict:
    p = study / 'selections.json'
    if p.exists():
        return context.read_json(p)
    later = [str(x) for x in study.glob('jobs/*/later_*')] + ([str(study / 'later_barrier.json')] if (study / 'later_barrier.json').exists() else [])
    if later:
        raise RuntimeError('C_B/E outputs exist before the selections: %s' % later[:3])
    rec = {'status': 'FROZEN', 'rules': SELECTION_RULES, 'jobs': {}, 'incomplete_jobs': []}
    for job in jobs:
        cp = study / 'jobs' / job / 'ca_scores.json'
        if not cp.exists():
            rec['incomplete_jobs'].append(job)
            continue
        ls = rule_losses(context.read_json(cp), proxy_seeds, REF_SEEDS[job] if ref_seeds is None else ref_seeds, ref_n)
        sel = {r: pick(ls[r]) for r in ls}
        sel['Always_P0'] = 'P0_Linear'
        rec['jobs'][job] = {'selected': sel, 'c_a_seed_mean_losses': ls,
                            'not_selectable': [r for r in RULES if sel[r] is None]}
    rec['c_b_e_outputs_at_selection'] = later
    rec['frozen_local'] = now()
    rec['frozen_epoch'] = time.time()
    context.write_json(p, rec)
    return rec


# ----------------------------------------------------------------------------------------------------------- controller stages
def proxy_stage(ledger: Ledger) -> dict:
    out = {'started_local': now(), 'workers': {}}
    for job in JOBS:
        for seed in PROXY_SEEDS:
            name = '%s__s%d' % (job, seed)
            rp = ROOT / 'jobs' / job / 'proxy' / ('proxy__s%d.json' % seed)
            done = lambda: set((context.read_json(rp)['units'] if rp.exists() else {}).keys())
            missing = [u for u in ('start',) + MIDS if u not in done()]
            if not missing and context.read_json(rp).get('complete'):
                out['workers'][name] = 'CACHE'
                continue
            status = 'TECHNICAL_FAILURE'
            for attempt in (0, 1):
                ledger.reserve([('start' if u == 'start' else 'adapt', '%s|%s' % (name, u)) for u in missing], name, retry=attempt > 0)
                rc, secs, reason = run_worker(worker_cmd('--worker-proxy', job, seed), ROOT / 'jobs' / job / 'proxy' / 'logs' / (name + ('.log' if attempt == 0 else '.retry.log')))
                now_done = done()
                ok = rc == 0 and rp.exists() and context.read_json(rp).get('complete')
                ledger.finish(name, bool(ok), secs, '' if ok else (reason or 'worker_failed rc=%s' % rc), [u for u in missing if u in now_done])
                if ok:
                    rec = context.read_json(rp)
                    rec['attempts'][-1]['controller_wall_seconds'] = secs
                    context.write_json(rp, rec)
                    status = 'OK'
                    break
                missing = [u for u in ('start',) + MIDS if u not in now_done]
                if ledger.d['retries_used'] + len(missing) > CAPS['retry']:
                    break
            print('PROXY', name, status, flush=True)
            out['workers'][name] = status
    out['finished_local'] = now()
    out['finished_epoch'] = time.time()
    context.write_json(ROOT / 'proxy_stage.json', out)
    return out


def wiring_checks(ledger: Ledger) -> dict:
    p = ROOT / 'checks' / 'wiring_checks.json'
    if p.exists():
        return context.read_json(p)
    job, seed, mid = CHECK
    rp = context.read_json(ROOT / 'jobs' / job / 'proxy' / ('proxy__s%d.json' % seed))
    out = {'started_local': now()}
    # check 1: the readiness worker's direct 100-update fit on the rebuilt materials == theta0
    d1 = ROOT / 'checks' / 'material_rebuild'
    cid = rd.length_cell_id(job, 'P0_Linear', seed, START_UPDATES)
    cell = d1 / job / 'cells' / (cid + '.json')
    if not cell.exists():
        ledger.reserve([('check', cid)], 'check_1', retry=False)
        cmd = [sys.executable, '-B', '-m', rd.FIT_MODULE, '--dataset', ds_of(job), '--job', job, '--run-dir', str(d1), '--material', 'P0_Linear',
               '--seed', str(seed), '--n-updates', str(START_UPDATES), '--no-feedback']
        rc, secs, reason = run_worker(cmd, ROOT / 'checks' / 'check_1.log')
        ledger.finish('check_1', rc == 0 and cell.exists(), secs, reason)
        out['check_1_wall_seconds'] = secs
    if cell.exists():
        c = context.read_json(cell)
        cmp_ = compare_models(c['model_path'], rp['units']['start']['model_path'])
        out['check_1'] = {'direct_cell': cid, 'direct_train_seconds': c['train']['seconds'], 'theta0': rp['units']['start']['model_path'],
                          'weights': cmp_, 'pass': cmp_['exact_equal']}
    else:
        out['check_1'] = {'status': 'TECHNICAL_FAILURE', 'pass': False}
    # check 2: a lone branch from the saved theta0 == the planned branch at update 1 and 25
    d2 = ROOT / 'checks' / 'adapt_only'
    if not (d2 / 'adapt_only.json').exists():
        ledger.reserve([('check', '%s|%d|%s' % CHECK)], 'check_2', retry=False)
        rc, secs, reason = run_worker(worker_cmd('--worker-adapt-only', job, seed, mid), ROOT / 'checks' / 'check_2.log')
        ledger.finish('check_2', rc == 0 and (d2 / 'adapt_only.json').exists(), secs, reason)
        out['check_2_wall_seconds'] = secs
    if (d2 / 'adapt_only.json').exists():
        a = context.read_json(d2 / 'adapt_only.json')
        cmps = {k: compare_models(a['model_paths'][k], rp['units'][mid]['model_paths'][k]) for k in ('1', '25')}
        out['check_2'] = {'branch': list(CHECK), 'planned_branch_order': list(MIDS), 'weights': cmps, 'pass': all(v['exact_equal'] for v in cmps.values())}
    else:
        out['check_2'] = {'status': 'TECHNICAL_FAILURE', 'pass': False}
    out['finished_local'] = now()
    context.write_json(p, out)
    return out


def ca_stage(ledger: Ledger, jobs) -> list:
    done = []
    for job in jobs:
        if not (ROOT / 'jobs' / job / 'ca_scores.json').exists():
            rc, secs, reason = run_worker(worker_cmd('--worker-ca', job), ROOT / 'jobs' / job / 'logs' / 'ca.log')
            ok = rc == 0 and (ROOT / 'jobs' / job / 'ca_scores.json').exists()
            ledger.stage('ca|' + job, ok, secs, reason)
            if ok:
                rec = context.read_json(ROOT / 'jobs' / job / 'ca_scores.json')
                rec['timing']['controller_wall_seconds'] = secs
                context.write_json(ROOT / 'jobs' / job / 'ca_scores.json', rec)
            print('C_A', job, 'OK' if ok else 'FAILED', round(secs, 1), flush=True)
        if (ROOT / 'jobs' / job / 'ca_scores.json').exists():
            done.append(job)
    return done


def later_stage(ledger: Ledger, sel: dict) -> dict:
    jobs = [j for j in JOBS if j in sel['jobs']]
    walls = {}
    for job in jobs:
        if not (ROOT / 'jobs' / job / 'later_frozen.json').exists():
            rc, secs, reason = run_worker(worker_cmd('--worker-later-predict', job), ROOT / 'jobs' / job / 'logs' / 'later_predict.log')
            ledger.stage('later_predict|' + job, rc == 0, secs, reason)
            walls['predict|' + job] = secs
            if rc != 0:
                raise RuntimeError('C_B/E prediction failed for %s' % job)
    barrier = ROOT / 'later_barrier.json'
    if not barrier.exists():
        context.write_json(barrier, {'jobs': {j: context.read_json(ROOT / 'jobs' / j / 'later_frozen.json')['frozen_local'] for j in jobs},
                                     'written_local': now(), 'written_epoch': time.time()})
    for job in jobs:
        if not (ROOT / 'jobs' / job / 'later_scores.json').exists():
            rc, secs, reason = run_worker(worker_cmd('--worker-later-score', job), ROOT / 'jobs' / job / 'logs' / 'later_score.log')
            ledger.stage('later_score|' + job, rc == 0, secs, reason)
            walls['score|' + job] = secs
            if rc != 0:
                raise RuntimeError('C_B/E scoring failed for %s' % job)
    return {'jobs': jobs, 'walls': walls, 'finished_local': now()}


# ----------------------------------------------------------------------------------------------------------- readout
def _stats(v: list, interval: bool = True) -> dict:
    n = len(v)
    if n == 0:
        return {'n': 0}
    mean = statistics.fmean(v)
    sd = statistics.stdev(v) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    out = {'n': n, 'mean': mean, 'sd': sd, 'se': se, 'positive': sum(x > 0 for x in v), 'negative': sum(x < 0 for x in v), 'zero': sum(x == 0 for x in v),
           'values': list(v)}
    if interval and se is not None:
        from scipy.stats import t as _t
        q = float(_t.ppf(0.975, n - 1))
        out['t95_interval'] = [mean - q * se, mean + q * se]
    return out


def _sign(x) -> int | None:
    return None if x is None else (1 if x > 0 else -1 if x < 0 else 0)


def _resolution(st: dict) -> str:
    iv = st.get('t95_interval')
    if not iv:
        return 'REFERENCE_UNRESOLVED'
    return 'RESOLVED_POSITIVE' if iv[0] > 0 else 'RESOLVED_NEGATIVE' if iv[1] < 0 else 'REFERENCE_UNRESOLVED'


def _pair_d(block: dict, p0: str, pi: str, seeds) -> tuple:
    """(per-seed macro differences, per-origin lists) of P0 minus Pi over the seeds where both are SCORABLE."""
    d, per_o = [], None
    for s in seeds:
        a, b = block.get(p0, {}).get(str(s)), block.get(pi, {}).get(str(s))
        if a and b and a['macro'] is not None and b['macro'] is not None:
            d.append(a['macro'] - b['macro'])
            po = [x - y for x, y in zip(a['per_origin_macro'], b['per_origin_macro'])]
            per_o = [[] for _ in po] if per_o is None else per_o
            for i, x in enumerate(po):
                per_o[i].append(x)
    return d, per_o or []


def approximation(ca: dict, job: str) -> dict:
    out = {}
    src = {'proxy25': (ca['proxy']['25'], PROXY_SEEDS, False), 'proxy1': (ca['proxy']['1'], PROXY_SEEDS, False),
           'ref2000': (ca['ref']['2000'], REF_SEEDS[job], True), 'ref100': (ca['ref']['100'], REF_SEEDS[job], True)}
    for pi, p0 in PAIRS:
        key = '%s-%s' % (pi, p0)
        row = {}
        for name, (block, seeds, iv) in src.items():
            d, per_o = _pair_d(block, p0, pi, seeds)
            row[name] = {**_stats(d, iv), 'per_origin': [_stats(x, iv) for x in per_o]}
        for ref in ('ref2000', 'ref100'):
            row[ref]['resolution'] = _resolution(row[ref])
            for o in row[ref]['per_origin']:
                o['resolution'] = _resolution(o)
        row['signs'] = {name: _sign(row[name].get('mean')) for name in src}
        row['proxy_seed_signs_unanimous'] = {name: (row[name].get('n', 0) > 0 and (row[name]['positive'] == row[name]['n'] or row[name]['negative'] == row[name]['n']))
                                             for name in ('proxy25', 'proxy1')}
        out[key] = row
    return out


def approximation_summary(appr: dict) -> dict:
    cells = [(job, key, row) for job, jr in appr.items() for key, row in jr.items()]
    out = {'cells': len(cells), 'note': 'job x pair cells are not independent jobs; the cross-context units are 2 jobs per domain'}
    for proxy in ('proxy25', 'proxy1'):
        for ref in ('ref2000', 'ref100'):
            agree = [c for c in cells if c[2]['signs'][proxy] is not None and c[2]['signs'][proxy] == c[2]['signs'][ref]]
            resolved = [c for c in cells if c[2][ref]['resolution'] != 'REFERENCE_UNRESOLVED']
            agree_res = [c for c in resolved if c[2]['signs'][proxy] == c[2]['signs'][ref]]
            out['%s_vs_%s' % (proxy, ref)] = {'agreement_with_finite_seed_reference_mean': '%d/%d' % (len(agree), len(cells)),
                                             'resolved_reference_cells': len(resolved), 'agreement_on_resolved': '%d/%d' % (len(agree_res), len(resolved)),
                                             'resolved_cells': ['%s|%s|%s' % (j, k, r[ref]['resolution']) for j, k, r in resolved],
                                             'by_domain': {d: '%d/%d' % (sum(1 for c in agree if ds_of(c[0]) == d), sum(1 for c in cells if ds_of(c[0]) == d)) for d in DOMAINS}}
    out['ref2000_vs_ref100_opposite_mean_sign'] = ['%s|%s' % (j, k) for j, k, r in cells if r['signs']['ref2000'] and r['signs']['ref100'] and r['signs']['ref2000'] != r['signs']['ref100']]
    out['proxy25_unanimous_seed_sign_cells'] = sum(1 for _, _, r in cells if r['proxy_seed_signs_unanimous']['proxy25'])
    return out


def later_losses(job: str) -> dict:
    """{n: {block: {mid: {seed: compact}}}} for the reference models (C_A from ca_scores, C_B/E from later_scores)."""
    ca = context.read_json(ROOT / 'jobs' / job / 'ca_scores.json')
    ls = context.read_json(ROOT / 'jobs' / job / 'later_scores.json') if (ROOT / 'jobs' / job / 'later_scores.json').exists() else None
    out = {}
    for n in REF_N:
        out[n] = {'c_a': ca['ref'][str(n)]}
        if ls:
            for b in ('c_b', 'e'):
                out[n][b] = {m: {str(s): ls['scores'][rd.length_cell_id(job, m, s, n)][b] for s in REF_SEEDS[job]} for m in MIDS}
    return out


def selection_value(sel: dict) -> dict:
    by_job = {}
    for job, sj in sel['jobs'].items():
        L = later_losses(job)
        chosen = sj['selected']
        jr = {'selected': chosen, 'consumers': {}}
        p0_2000 = {b: statistics.fmean(v['macro'] for v in L[2000][b]['P0_Linear'].values()) if b in L[2000] and all(
            v['macro'] is not None for v in L[2000][b]['P0_Linear'].values()) else None for b in BLOCKS}
        jr['ref2000_p0_mean'] = p0_2000
        for n, cname in ((2000, 'ref2000'), (100, 'ref100')):
            if 'e' not in L[n]:
                continue
            cr = {'material_means': {}, 'rule_means': {}, 'paired': {}, 'regret': {}}
            for m in MIDS:
                cr['material_means'][m] = {b: (statistics.fmean(v['macro'] for v in L[n][b][m].values())
                                              if all(v['macro'] is not None for v in L[n][b][m].values()) else None) for b in BLOCKS}
            for r in RULES:
                m = chosen.get(r)
                cr['rule_means'][r] = {'material': m, **({b: cr['material_means'][m][b] for b in BLOCKS} if m else {})}
                if m:
                    cr['regret'][r] = {b: (None if cr['material_means'][m][b] is None or any(cr['material_means'][x][b] is None for x in MIDS)
                                           else cr['material_means'][m][b] - min(cr['material_means'][x][b] for x in MIDS)) for b in BLOCKS}
            for other, base in COMPARISONS:
                mo, mb = chosen.get(other), chosen.get(base)
                key = '%s-%s' % (other, base)
                if not (mo and mb):
                    cr['paired'][key] = {'status': 'NOT_SELECTABLE'}
                    continue
                cr['paired'][key] = {'materials': [mo, mb], 'same_material': mo == mb}
                for b in BLOCKS:
                    d = [L[n][b][mo][str(s)]['macro'] - L[n][b][mb][str(s)]['macro'] for s in REF_SEEDS[job]
                         if L[n][b][mo][str(s)]['macro'] is not None and L[n][b][mb][str(s)]['macro'] is not None]
                    st = _stats(d)
                    st['relative_to_ref2000_p0_mean'] = (st['mean'] / p0_2000[b]) if st.get('n') and p0_2000[b] else None
                    cr['paired'][key][b] = st
            jr['consumers'][cname] = cr
        by_job[job] = jr
    pooled = {}
    for cname in ('ref2000', 'ref100'):
        for other, base in COMPARISONS:
            key = '%s-%s' % (other, base)
            for b in BLOCKS:
                per_dom = {}
                for d, jobs in domain_jobs(list(by_job)).items():
                    vals = [by_job[j]['consumers'].get(cname, {}).get('paired', {}).get(key, {}).get(b, {}).get('relative_to_ref2000_p0_mean') for j in jobs]
                    if jobs and all(v is not None for v in vals) and len(jobs) == 2:
                        per_dom[d] = statistics.fmean(vals)
                pooled.setdefault(cname, {}).setdefault(key, {})[b] = {'by_domain': per_dom,
                                                                      'domains_equal': statistics.fmean(per_dom.values()) if len(per_dom) == len(DOMAINS) else None}
        for r in RULES:
            for b in BLOCKS:
                per_dom = {}
                for d, jobs in domain_jobs(list(by_job)).items():
                    vals = []
                    for j in jobs:
                        reg = by_job[j]['consumers'].get(cname, {}).get('regret', {}).get(r, {}).get(b)
                        den = by_job[j]['ref2000_p0_mean'][b]
                        vals.append(None if reg is None or not den else reg / den)
                    if len(jobs) == 2 and all(v is not None for v in vals):
                        per_dom[d] = statistics.fmean(vals)
                pooled.setdefault(cname + '_regret', {}).setdefault(r, {})[b] = {'by_domain': per_dom,
                                                                                'domains_equal': statistics.fmean(per_dom.values()) if len(per_dom) == len(DOMAINS) else None}
    dependence = {}
    for other, base in COMPARISONS:
        key = '%s-%s' % (other, base)
        for b in BLOCKS:
            a2, a1 = pooled['ref2000'][key][b]['domains_equal'], pooled['ref100'][key][b]['domains_equal']
            per_job = {j: [_sign((by_job[j]['consumers']['ref2000']['paired'][key].get(b) or {}).get('mean')),
                           _sign((by_job[j]['consumers']['ref100']['paired'][key].get(b) or {}).get('mean'))]
                       for j in by_job if 'ref100' in by_job[j]['consumers'] and 'status' not in by_job[j]['consumers']['ref2000']['paired'][key]}
            dependence.setdefault(key, {})[b] = {
                'pooled': 'CONSUMER_DEPENDENT' if _sign(a2) and _sign(a1) and _sign(a2) != _sign(a1) else 'SAME_OR_ZERO',
                'jobs_consumer_dependent': [j for j, (x, y) in per_job.items() if x and y and x != y]}
    same = {r: [j for j in by_job if by_job[j]['selected'].get(r) == by_job[j]['selected'].get('R25')] for r in ('R2000', 'R100', 'Always_P0', 'R1')}
    return {'by_job': by_job, 'pooled_relative': pooled, 'consumer_dependence': dependence, 'jobs_where_rule_selects_same_as_R25': same}


def proxy_diagnostics(job: str) -> dict:
    ca = context.read_json(ROOT / 'jobs' / job / 'ca_scores.json')
    m = lambda block: statistics.fmean(v['macro'] for v in block.values()) if block and all(v['macro'] is not None for v in block.values()) else None
    out = {'c_a_seed_mean': {'theta0': m(ca['theta0']), **{'%s_k%s' % (mid, k): m(ca['proxy'][k][mid]) for k in ('1', '25') for mid in MIDS},
                             **{'ref%s_%s' % (n, mid): m(ca['ref'][n][mid]) for n in ('100', '2000') for mid in MIDS}}, 'drawn_changed_windows': {}}
    for mid in MIDS[1:]:
        rows = [context.read_json(ROOT / 'jobs' / job / 'proxy' / ('proxy__s%d.json' % s))['units'][mid] for s in PROXY_SEEDS]
        out['drawn_changed_windows'][mid] = {'material_windows_changed_vs_p0': rows[0]['material_windows_changed_vs_p0'], 'material_windows': rows[0]['material_windows'],
                                             **{'k%s' % k: {'changed_draws_per_seed': [r['drawn'][k]['changed_draws'] for r in rows],
                                                            'unique_changed_windows_per_seed': [r['drawn'][k]['unique_changed_windows'] for r in rows],
                                                            'draws': rows[0]['drawn'][k]['draws']} for k in ('1', '25')}}
    return out


def cost_readout(led: dict) -> dict:
    tl_wc = context.read_json(TL / 'checks' / 'wiring_checks.json')
    tl_led = context.read_json(TL / 'budget.json')
    tl_walls = [e['seconds'] for e in tl_led['events'] if e['kind'] == 'fit_finished']
    tl_train = []
    for p in list((TL / 'dev').rglob('cells/*__u2000.json')) + list((TL / 'follow').rglob('cells/*__u2000.json')):
        c = context.read_json(p)
        if c.get('physical_fit'):
            tl_train.append(c['train']['seconds'])
    mc = context.read_json(ROOT / 'checks' / 'material_rebuild.json')
    jobs = {}
    all_proxy = []
    for job in JOBS:
        prox = [context.read_json(ROOT / 'jobs' / job / 'proxy' / ('proxy__s%d.json' % s)) for s in PROXY_SEEDS]
        ca_p = ROOT / 'jobs' / job / 'ca_scores.json'
        ca = context.read_json(ca_p) if ca_p.exists() else None
        walls = [sum(a.get('controller_wall_seconds', 0) for a in r['attempts']) for r in prox]
        start_train = [r['units']['start']['train_seconds'] for r in prox]
        adapt_train = [r['units'][m]['train_seconds'] for r in prox for m in MIDS]
        pm_ = ca['timing']['per_model_seconds'] if ca else {}
        # the whole C_A worker wall (process start, torch import inside the first model load, open, serve) minus the models R25 does not need
        ca_for_r25 = (ca['timing']['controller_wall_seconds'] - sum(pm_['ref']) - sum(pm_['proxy_k1']) - sum(pm_['theta0'][1:])) if ca else None
        r25 = sum(walls) + ca_for_r25 if ca else None
        jobs[job] = {'proxy_worker_walls': walls, 'proxy_import_seconds': [r['attempts'][-1]['import_seconds'] for r in prox],
                     'proxy_open_and_bind_seconds': [r['attempts'][-1]['open_and_bind_seconds'] for r in prox],
                     'start_train_seconds': start_train, 'adapt_train_seconds_mean': statistics.fmean(adapt_train),
                     'proxy_peak_working_set_mb': max((r['attempts'][-1].get('peak_working_set_mb') or 0) for r in prox),
                     'ca_worker': (ca['timing'] | {'peak_working_set_mb': ca.get('peak_working_set_mb')}) if ca else None,
                     'material_rebuild_seconds': {m: v['build_seconds'] for m, v in mc['jobs'][job]['materials'].items()},
                     'R25_per_job_seconds': r25, 'R25_c_a_worker_share_seconds': ca_for_r25,
                     'R25_composition': '3 proxy worker walls (process start, torch import, T load + binding, theta0 100 updates, 3 x 25-update branches, '
                                        '7 model saves) + the C_A worker wall minus the reference, step-1 and repeated theta0 model timers (process start, '
                                        'torch import, rows, Linear inputs, 9 step-25 models + first theta0 load); MEASURED parts',
                     'proxy_worker_process_start_and_import_seconds': [a_w - r['attempts'][-1]['body_seconds'] + r['attempts'][-1]['import_seconds']
                                                                       for a_w, r in zip(walls, prox)],
                     'R25_training_seconds': sum(start_train) + sum(r['units'][m]['train_seconds'] for r in prox for m in MIDS)}
        all_proxy += walls
    fits = [e for e in led.get('events', []) if e['kind'] == 'worker_finished']
    stage_walls = [e for e in led.get('events', []) if e['kind'] == 'stage_worker']
    est_fit_wall = tl_wc['check_2']['legacy_wall_seconds']
    out = {'actual': {'trajectory_units': led.get('physical'), 'retries_used': led.get('retries_used'), 'failed_workers': led.get('failed_workers'),
                      'units_total': sum((led.get('physical') or {}).values()) + (led.get('retries_used') or 0), 'hard_cap': CAPS['hard'],
                      'updates_normal': len(JOBS) * len(PROXY_SEEDS) * START_UPDATES + len(JOBS) * len(PROXY_SEEDS) * len(MIDS) * ADAPT_UPDATES,
                      'check_updates': START_UPDATES + ADAPT_UPDATES,
                      'proxy_and_check_worker_seconds': sum(e['seconds'] for e in fits), 'stage_worker_seconds': {e['name']: e['seconds'] for e in stage_walls},
                      'material_rebuild_worker_wall_seconds': mc.get('worker_wall_seconds'),
                      'numeric_run_wall_seconds': (max(e['epoch'] for e in led['events']) - led['first_launch_epoch']) if led.get('first_launch_epoch') else None,
                      'llm_calls': 0, 'api_calls': 0, 'new_full_fits': 0},
           'per_job': jobs,
           'full_training_comparator_ESTIMATE': {
               'definition': '3 materials x 3 seeds x 2000 updates from scratch, each fit + its C_A scoring in its own worker (the current path)',
               'per_fit_wall_seconds_measured_in_TRAINING_LENGTH': {'legacy_default_path_wiring_check': est_fit_wall,
                                                                     'mean_of_86_trajectory_workers': statistics.fmean(tl_walls),
                                                                     'mean_2000_update_train_seconds': statistics.fmean(tl_train)},
               'per_job_seconds_ESTIMATE': 9 * est_fit_wall, 'per_job_training_seconds_ESTIMATE': 9 * statistics.fmean(tl_train),
               'per_job_updates': 9 * spec.N_UPDATES, 'proxy_per_job_updates': len(PROXY_SEEDS) * (START_UPDATES + len(MIDS) * ADAPT_UPDATES),
               'note': 'ESTIMATE from TRAINING-LENGTH measurements in the same environment; no full fit was run for timing; cache reads are not training cost. '
                       'The small-MLP ratio does not transfer to a TSFM.'}}
    return out


def protocol_checks(sel: dict, led: dict) -> dict:
    before = context.read_json(ROOT / 'checks' / 'cache_snapshot_before.json')['files']
    after = snapshot([Path(p) for p in before])
    proxy_end = max(context.read_json(ROOT / 'jobs' / j / 'proxy' / ('proxy__s%d.json' % s))['units'][u]['finished_epoch']
                    for j in JOBS for s in PROXY_SEEDS for u in ('start',) + MIDS)
    ca_starts = [e['epoch'] - e['seconds'] for e in led['events'] if e['kind'] == 'stage_worker' and e['name'].startswith('ca|')]
    ca_ends = [context.read_json(ROOT / 'jobs' / j / 'ca_scores.json')['finished_epoch'] for j in JOBS if (ROOT / 'jobs' / j / 'ca_scores.json').exists()]
    lf = [context.read_json(ROOT / 'jobs' / j / 'later_frozen.json') for j in sel['jobs'] if (ROOT / 'jobs' / j / 'later_frozen.json').exists()]
    ls = [context.read_json(ROOT / 'jobs' / j / 'later_scores.json') for j in sel['jobs'] if (ROOT / 'jobs' / j / 'later_scores.json').exists()]
    barrier = context.read_json(ROOT / 'later_barrier.json') if (ROOT / 'later_barrier.json').exists() else None
    proxy_recs = [context.read_json(ROOT / 'jobs' / j / 'proxy' / ('proxy__s%d.json' % s)) for j in JOBS for s in PROXY_SEEDS]
    new_paths = [p for r in proxy_recs for u in r['units'].values() for p in ([u['model_path']] if 'model_path' in u else list(u['model_paths'].values()))]
    ca_all = [context.read_json(ROOT / 'jobs' / j / 'ca_scores.json') for j in JOBS]
    checks = {
        'cache_files_unchanged_size_and_mtime': after == before,
        'new_models_written_under_package_root': all(Path(p).resolve().is_relative_to(ROOT.resolve()) for p in new_paths),
        'all_proxy_units_before_first_c_a': bool(ca_starts) and proxy_end < min(ca_starts),
        'selections_after_all_c_a': bool(ca_ends) and sel['frozen_epoch'] > max(ca_ends),
        'no_c_b_e_output_at_selection': sel['c_b_e_outputs_at_selection'] == [],
        'later_predictions_after_selections': bool(lf) and all(r['started_epoch'] > sel['frozen_epoch'] for r in lf),
        'barrier_after_all_predictions': barrier is not None and barrier['written_epoch'] >= max(r['frozen_epoch'] for r in lf),
        'scores_after_barrier': bool(ls) and all(r['finished_epoch'] > barrier['written_epoch'] for r in ls),
        'c_a_slices_end_before_c_b': all(c['slice_row_end'] <= min(rd.resolve_job(ds_of(c['job_id']), c['job_id']).c_b) for c in ca_all),
        'theta0_unchanged_in_every_worker': all(a['theta0_unchanged_after_branches'] for r in proxy_recs for a in r['attempts']),
        'fresh_optimizer_every_branch': all(r['units'][m]['fresh_optimizer'] for r in proxy_recs for m in MIDS),
        'worker_kmp_unset_at_entry': all(a['kmp_duplicate_lib_ok_at_worker_entry'] is None for r in proxy_recs for a in r['attempts']),
        'recomputed_p0_c_a_equals_cache': all(x['equal'] for c in ca_all for x in c['cache_consistency_p0_c_a']),
        'recomputed_shadow_c_b_e_equals_cache_stp1': bool(ls) and all(x['equal'] for r in ls for x in r['cache_consistency_shadow']),
        'units_within_hard_cap': sum(led['physical'].values()) + led['retries_used'] <= CAPS['hard'],
        'prsa_rows_below_closed_region': all(r['rows_read']['e_target'][1] <= PRSA_CLOSED_FROM_ROW for r in ls if ds_of(r['job_id']) == 'RD02'),
    }
    info = {'kmp_at_worker_end_values': sorted({str(a.get('kmp_duplicate_lib_ok_at_worker_end')) for r in proxy_recs for a in r['attempts']}),
            'shadow_consistency_comparisons': sum(len(r['cache_consistency_shadow']) for r in ls),
            'p0_c_a_consistency_comparisons': sum(len(c['cache_consistency_p0_c_a']) for c in ca_all),
            'changed_cache_files': [p for p in before if before[p] != after.get(p)]}
    return {'checks': checks, 'all_true': all(checks.values()), 'n_checks': len(checks), 'info': info}


def readout() -> dict:
    cfg = context.read_json(ROOT / 'frozen_config.json')
    sel = context.read_json(ROOT / 'selections.json') if (ROOT / 'selections.json').exists() else None
    led = context.read_json(ROOT / 'budget.json') if (ROOT / 'budget.json').exists() else {}
    res = {'package': cfg['package'], 'written_local': now(), 'run_status': context.read_json(ROOT / 'run_status.json') if (ROOT / 'run_status.json').exists() else None,
           'selections': sel, 'wiring_checks': context.read_json(ROOT / 'checks' / 'wiring_checks.json') if (ROOT / 'checks' / 'wiring_checks.json').exists() else None}
    jobs_ca = [j for j in JOBS if (ROOT / 'jobs' / j / 'ca_scores.json').exists()]
    res['approximation'] = {j: approximation(context.read_json(ROOT / 'jobs' / j / 'ca_scores.json'), j) for j in jobs_ca}
    res['approximation_summary'] = approximation_summary(res['approximation'])
    res['proxy_diagnostics'] = {j: proxy_diagnostics(j) for j in jobs_ca}
    if sel and all((ROOT / 'jobs' / j / 'later_scores.json').exists() for j in sel['jobs']):
        res['selection_value'] = selection_value(sel)
        res['protocol_checks'] = protocol_checks(sel, led)
        context.write_json(ROOT / 'protocol_checks.json', res['protocol_checks'])
    res['cost'] = cost_readout(led)
    res['budget'] = {k: v for k, v in led.items() if k != 'events'}
    context.write_json(ROOT / 'result.json', res)
    write_tables(json.loads(json.dumps(res, default=str)))
    return res


def write_tables(res: dict) -> str:
    f = lambda x, d=3: '—' if x is None else ('%.*f' % (d, x))
    pm = lambda s, d=3: '—' if not s or s.get('n', 0) == 0 else '%s ± %s' % (f(s.get('mean'), d), f(s.get('se'), d))
    sg = lambda s: '—' if not s or s.get('n', 0) == 0 else '%d/%d/%d' % (s['positive'], s['negative'], s['zero'])
    iv = lambda s: '—' if not s or not s.get('t95_interval') else '[%s, %s]' % (f(s['t95_interval'][0]), f(s['t95_interval'][1]))
    lines = ['# DEV-DATA-READINESS-SHORT-ADAPT-SIGNAL tables', '',
             'Generated from result.json by `--result`. Losses: missing-aware macro NMSE on C_A/C_B/E (lower is better); every model is served the '
             'P0 Linear inputs. Material differences d = loss(P0) − loss(Pi): positive = Pi better. Selection differences = loss(other rule) − loss(R25): '
             'positive = R25 better. "±" is one SE over seeds; +/−/0 counts seeds.', '']
    sel = res.get('selections') or {}
    lines += ['## 1. Frozen selections', '', '| Job | R25 | R1 | R100 | R2000 | Always_P0 |', '|---|---|---|---|---|---|']
    for j, sj in (sel.get('jobs') or {}).items():
        lines.append('| %s | %s |' % (j, ' | '.join(str(sj['selected'].get(r)) for r in RULES)))
    lines += ['', '| Job | Rule | P0 C_A | P1 C_A | P2 C_A |', '|---|---|---:|---:|---:|']
    for j, sj in (sel.get('jobs') or {}).items():
        for r, v in sj['c_a_seed_mean_losses'].items():
            lines.append('| %s | %s | %s |' % (j, r, ' | '.join(f(v[m], 4) for m in MIDS)))
    lines += ['', '## 2. Same C_A: proxy difference versus reference difference', '',
              '| Job | Pair | proxy25 mean ± SE | +/−/0 | proxy1 mean ± SE | REF2000 mean ± SE | REF2000 t95 | REF2000 | REF100 mean ± SE | REF100 t95 | REF100 |',
              '|---|---|---:|---|---:|---:|---|---|---:|---|---|']
    for j, jr in (res.get('approximation') or {}).items():
        for k, r in jr.items():
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                j, k, pm(r['proxy25'], 4), sg(r['proxy25']), pm(r['proxy1'], 4), pm(r['ref2000'], 4), iv(r['ref2000']), r['ref2000']['resolution'],
                pm(r['ref100'], 4), iv(r['ref100']), r['ref100']['resolution']))
    lines += ['', '### 2b. Per C_A origin (mean d; origin t, t+48)', '', '| Job | Pair | proxy25 | proxy1 | REF2000 | REF100 |', '|---|---|---|---|---|---|']
    for j, jr in (res.get('approximation') or {}).items():
        for k, r in jr.items():
            cell = lambda s: ' / '.join(f(o.get('mean'), 4) for o in s['per_origin']) or '—'
            lines.append('| %s | %s | %s | %s | %s | %s |' % (j, k, cell(r['proxy25']), cell(r['proxy1']), cell(r['ref2000']), cell(r['ref100'])))
    lines += ['', '### 2c. Agreement summary', '', '```json', json.dumps(res.get('approximation_summary'), indent=1, ensure_ascii=False), '```', '']
    lines += ['## 3. Proxy diagnostics (C_A seed means; changed parent windows drawn)', '',
              '| Job | theta0 | P0 k1 | P0 k25 | P1 k25 | P2 k25 | REF100 P0 | REF2000 P0 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for j, d in (res.get('proxy_diagnostics') or {}).items():
        c = d['c_a_seed_mean']
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (j, f(c['theta0'], 4), f(c['P0_Linear_k1'], 4), f(c['P0_Linear_k25'], 4), f(c['P1_Seasonal_k25'], 4),
                                                                f(c['P2_Linear_Hampel_k25'], 4), f(c['ref100_P0_Linear'], 4), f(c['ref2000_P0_Linear'], 4)))
    lines += ['', '| Job | Material | windows changed vs P0 | k1 changed draws/64 per seed | k25 changed draws/1600 per seed | k25 unique changed windows per seed |',
              '|---|---|---|---|---|---|']
    for j, d in (res.get('proxy_diagnostics') or {}).items():
        for m, v in d['drawn_changed_windows'].items():
            lines.append('| %s | %s | %d/%d | %s | %s | %s |' % (j, m, v['material_windows_changed_vs_p0'], v['material_windows'], v['k1']['changed_draws_per_seed'],
                                                               v['k25']['changed_draws_per_seed'], v['k25']['unique_changed_windows_per_seed']))
    sv = res.get('selection_value')
    if sv:
        for cname, title in (('ref2000', '4. Selection value under REF2000 (main)'), ('ref100', '5. Selection value under REF100 (Consumer sensitivity; same frozen selections)')):
            lines += ['', '## ' + title, '', '| Job | Rule | Material | C_A | C_B | E | regret C_A | regret C_B | regret E |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
            for j, jr in sv['by_job'].items():
                cr = jr['consumers'].get(cname)
                if not cr:
                    continue
                for r in RULES:
                    rm, rg = cr['rule_means'][r], cr['regret'].get(r, {})
                    lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (j, r, rm.get('material'), f(rm.get('c_a'), 4), f(rm.get('c_b'), 4), f(rm.get('e'), 4),
                                                                                  f(rg.get('c_a'), 4), f(rg.get('c_b'), 4), f(rg.get('e'), 4)))
            lines += ['', '| Job | Comparison | Materials | Block | n | mean ± SE | +/−/0 | t95 | relative to REF2000 P0 |', '|---|---|---|---|---:|---:|---|---|---:|']
            for j, jr in sv['by_job'].items():
                cr = jr['consumers'].get(cname)
                if not cr:
                    continue
                for key, v in cr['paired'].items():
                    if 'status' in v:
                        lines.append('| %s | %s | — | — | — | %s | — | — | — |' % (j, key, v['status']))
                        continue
                    for b in ('e', 'c_b', 'c_a'):
                        s = v[b]
                        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (j, key, ' vs '.join(v['materials']), b, s.get('n'), pm(s, 4), sg(s), iv(s),
                                                                                      f(s.get('relative_to_ref2000_p0_mean'), 4)))
            lines += ['', '| Comparison | Block | RD01B (jobs equal) | RD02 (jobs equal) | domains equal |', '|---|---|---:|---:|---:|']
            for key, bb in sv['pooled_relative'][cname].items():
                for b in ('e', 'c_b', 'c_a'):
                    p = bb[b]
                    lines.append('| %s | %s | %s | %s | %s |' % (key, b, f(p['by_domain'].get('RD01B'), 4), f(p['by_domain'].get('RD02'), 4), f(p['domains_equal'], 4)))
            lines += ['', '| Rule | Block | regret RD01B | regret RD02 | domains equal |', '|---|---|---:|---:|---:|']
            for r, bb in sv['pooled_relative'][cname + '_regret'].items():
                for b in ('e', 'c_b', 'c_a'):
                    p = bb[b]
                    lines.append('| %s | %s | %s | %s | %s |' % (r, b, f(p['by_domain'].get('RD01B'), 4), f(p['by_domain'].get('RD02'), 4), f(p['domains_equal'], 4)))
        lines += ['', '## 6. Consumer dependence and repeated selections', '', '```json',
                  json.dumps({'consumer_dependence': sv['consumer_dependence'], 'jobs_where_rule_selects_same_as_R25': sv['jobs_where_rule_selects_same_as_R25']},
                             indent=1, ensure_ascii=False), '```', '']
    lines += ['## 7. Wiring and protocol checks', '', '```json', json.dumps({'wiring_checks': res.get('wiring_checks'), 'protocol_checks': res.get('protocol_checks')},
                                                                         indent=1, ensure_ascii=False, default=str)[:6000], '```', '',
              '## 8. Cost', '', '```json', json.dumps(res.get('cost'), indent=1, default=str), '```', '']
    text = '\n'.join(lines)
    (ROOT / 'tables.md').write_text(text, encoding='utf-8')
    return text


# ----------------------------------------------------------------------------------------------------------- smoke (synthetic; 0 real-data fits)
def smoke() -> dict:
    rd._init_openmp_before_torch()
    import torch
    from methods.ttha.batch_base import train
    from evaluation.main_protocol_p4 import batch_research_data_readiness as drs
    checks = {}
    eq = lambda a, b: list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    checks['tie_rule'] = (pick({'P0_Linear': 1.0, 'P1_Seasonal': 1.0, 'P2_Linear_Hampel': 0.5}) == 'P2_Linear_Hampel'
                          and pick({'P0_Linear': 1.0, 'P1_Seasonal': 1.0 - 1e-13, 'P2_Linear_Hampel': 2.0}) == 'P0_Linear'
                          and pick({'P0_Linear': 0.5, 'P1_Seasonal': 0.5, 'P2_Linear_Hampel': 0.5}) == 'P0_Linear'
                          and pick({'P0_Linear': None, 'P1_Seasonal': 0.5, 'P2_Linear_Hampel': 0.5}) is None)
    job, ds, ref_seed, pseed, ref_n = 'RDX_V1', 'RDX', 5, 11, (100, 200)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        drs._register_synthetic()
        try:
            cache_root, study = td / 'cache', td / 'study'
            ctx = rd.open_job(ds, job, cache_root, 'material')
            for mid, pol in MATERIALS:
                rd.build(ctx, rd.compile_policy(pol, [{}] * len(ctx.job.roster))['assignment'], mid)
            for mid in MIDS:
                rd.fit_one(ds, job, str(cache_root), mid, ref_seed, n_updates=200, checkpoints=[100], score_checkpoints=[100])
            cache = cache_root / job
            before = snapshot([p for p in cache.rglob('*') if p.is_file()])
            rec = proxy_train(job, pseed, study, cache)
            checks['proxy_units_complete_theta0_unchanged'] = bool(rec['complete']) and rec['attempts'][-1]['theta0_unchanged_after_branches']
            # P0/P0 copy, execution order and step-1 saving, all from the saved theta0
            c2 = rd.open_job(ds, job, study / 'jobs', 'material')
            X, y, _ = bound_arrays(c2, cache)
            state0 = torch.load(c2.job_dir / 'proxy' / 'models' / ('theta0__s%d.pt' % pseed), map_location='cpu')
            rows = train.batch_indices(pseed, n_pool=c2.legal.n)[ADAPT_ROWS[0]:ADAPT_ROWS[1]]
            saved = {m: {k: torch.load(rec['units'][m]['model_paths'][str(k)], map_location='cpu') for k in PROXY_K} for m in MIDS}
            rev = {m: adapt(state0, X[m], y, rows)[0].state_dict() for m in reversed(MIDS)}
            p0_again = adapt(state0, X['P0_Linear'], y, rows)[0].state_dict()
            checks['p0_copy_bitwise_equal'] = eq(p0_again, saved['P0_Linear'][25]) and eq(rev['P0_Linear'], saved['P0_Linear'][25])
            checks['branch_order_does_not_matter'] = all(eq(rev[m], saved[m][25]) for m in MIDS) and eq(state0, torch.load(c2.job_dir / 'proxy' / 'models' / ('theta0__s%d.pt' % pseed)))
            one = adapt(state0, X['P1_Seasonal'], y, rows[:1])[0].state_dict()
            checks['step1_state_equals_one_update_and_step25_unchanged'] = eq(one, saved['P1_Seasonal'][1]) and eq(rev['P1_Seasonal'], saved['P1_Seasonal'][25])
            checks['branches_differ_only_in_training_X'] = (not np.array_equal(X['P1_Seasonal'], X['P0_Linear']) and not np.array_equal(X['P2_Linear_Hampel'], X['P0_Linear'])
                                                            and all(rec['units'][m]['batch_rows'] == list(ADAPT_ROWS) for m in MIDS)
                                                            and rec['units']['P0_Linear']['drawn']['25']['changed_draws'] == 0)
            try:
                later_predict(job, study, cache, ref_seeds=(ref_seed,), ref_n=ref_n)
                checks['c_b_e_refused_before_selections'] = False
            except PermissionError:
                checks['c_b_e_refused_before_selections'] = True
            ca = ca_eval(job, study, cache, proxy_seeds=(pseed,), ref_seeds=(ref_seed,), ref_n=ref_n)
            checks['c_a_rows_end_before_c_b_and_linear_serving'] = ca['slice_row_end'] <= min(ctx.job.c_b) and ca['serve_inputs']['windows'] == len(ctx.job.roster) * 2
            checks['recomputed_p0_c_a_equals_cached_cell'] = len(ca['cache_consistency_p0_c_a']) == 2 and all(x['equal'] for x in ca['cache_consistency_p0_c_a'])
            ce = rd.open_job(ds, job, study / 'jobs', 'evaluate')
            m2 = train.make_model()()
            m2.load_state_dict(p0_again)
            Xs, _ = rd.serve_inputs(ce.slice, ce.job.c_a, [[] for _ in ce.job.roster], ce.scaler)
            again = rd.score_block(rd.predict_inputs(m2, Xs, ce.scaler), rd.read_truth(ce.slice, ce.job.c_a), ce.scaler)['normalized_mse_macro']
            checks['p0_copy_c_a_delta_zero'] = ca['proxy']['25']['P0_Linear'][str(pseed)]['macro'] - again == 0.0
            sel = select(study, jobs=(job,), proxy_seeds=(pseed,), ref_seeds=(ref_seed,), ref_n=ref_n)
            checks['selections_written'] = sel['jobs'][job]['selected']['Always_P0'] == 'P0_Linear' and not sel['jobs'][job]['not_selectable']
            try:
                later_score(job, study, cache)
                checks['score_refused_before_barrier'] = False
            except PermissionError:
                checks['score_refused_before_barrier'] = True
            fz = later_predict(job, study, cache, ref_seeds=(ref_seed,), ref_n=ref_n)
            context.write_json(study / 'later_barrier.json', {'jobs': {job: fz['frozen_local']}, 'written_local': now(), 'written_epoch': time.time()})
            sc = later_score(job, study, cache)
            checks['later_scores_scorable'] = len(sc['scores']) == 6 and all(v[b]['status'] == 'SCORABLE' for v in sc['scores'].values() for b in ('c_b', 'e'))
            (study / 'selections.json').unlink()
            try:
                select(study, jobs=(job,), proxy_seeds=(pseed,), ref_seeds=(ref_seed,), ref_n=ref_n)
                checks['selection_refused_after_c_b_e_outputs'] = False
            except RuntimeError:
                checks['selection_refused_after_c_b_e_outputs'] = True
            checks['cache_read_only'] = snapshot([Path(p) for p in before]) == before and sorted(before) == sorted(str(p) for p in cache.rglob('*') if p.is_file())
        finally:
            rd.DATASETS.pop('RDX', None)
    ctl = subprocess.run([sys.executable, '-B', '-c', 'import sys; import %s; print("torch" in sys.modules)' % MODULE], cwd=str(REPO),
                         capture_output=True, text=True, timeout=120, env=worker_env())
    checks['controller_module_does_not_import_torch'] = ctl.stdout.strip().splitlines()[-1:] == ['False']
    rec = {'checks': checks, 'all_pass': all(checks.values()), 'real_data_fits': 0, 'written_local': now()}
    ROOT.mkdir(parents=True, exist_ok=True)
    context.write_json(ROOT / 'smoke_record.json', rec)
    return rec


# ----------------------------------------------------------------------------------------------------------- entry
def run() -> None:
    preflight()
    sm = ROOT / 'smoke_record.json'
    if not sm.exists() or not context.read_json(sm)['all_pass']:
        raise RuntimeError('smoke has not passed')
    ledger = Ledger(ROOT)
    status = {'started_local': now(), 'started_epoch': time.time()}
    try:
        ps = proxy_stage(ledger)
        status['proxy_workers'] = ps['workers']
        status['wiring_checks_pass'] = {k: v.get('pass') for k, v in wiring_checks(ledger).items() if k.startswith('check_') and isinstance(v, dict)}
        complete = [j for j in JOBS if all(ps['workers'].get('%s__s%d' % (j, s)) in ('OK', 'CACHE') for s in PROXY_SEEDS)]
        ca_jobs = ca_stage(ledger, complete)
        sel = select(ROOT, jobs=tuple(ca_jobs))
        status['selections'] = {j: v['selected'] for j, v in sel['jobs'].items()}
        status['later'] = later_stage(ledger, sel)
        status['exit'] = 'COMPLETE' if len(sel['jobs']) == len(JOBS) else 'PARTIAL_JOBS'
    except BudgetStop as ex:
        status['exit'] = 'PARTIAL'
        status['stop'] = str(ex)
    status['finished_local'] = now()
    status['finished_epoch'] = time.time()
    context.write_json(ROOT / 'run_status.json', status)
    readout()
    print(json.dumps(status, ensure_ascii=False, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--compare-models', nargs=2)
    ap.add_argument('--worker-material-check', action='store_true')
    ap.add_argument('--worker-proxy', nargs=2)
    ap.add_argument('--worker-adapt-only', nargs=3)
    ap.add_argument('--worker-ca')
    ap.add_argument('--worker-later-predict')
    ap.add_argument('--worker-later-score')
    a = ap.parse_args()
    if a.compare_models:
        _compare_models_main(*a.compare_models)
    elif a.worker_material_check:
        rd._init_openmp_before_torch()
        material_check(ROOT)
    elif a.worker_proxy:
        job, seed = a.worker_proxy[0], int(a.worker_proxy[1])
        proxy_train(job, seed, ROOT, CACHE[job])
    elif a.worker_adapt_only:
        job, seed, mid = a.worker_adapt_only[0], int(a.worker_adapt_only[1]), a.worker_adapt_only[2]
        adapt_only(job, seed, mid, ROOT, CACHE[job], ROOT / 'checks' / 'adapt_only')
    elif a.worker_ca:
        ca_eval(a.worker_ca, ROOT, CACHE[a.worker_ca])
    elif a.worker_later_predict:
        later_predict(a.worker_later_predict, ROOT, CACHE[a.worker_later_predict])
    elif a.worker_later_score:
        later_score(a.worker_later_score, ROOT, CACHE[a.worker_later_score])
    elif a.smoke:
        print(json.dumps(smoke(), indent=1))
    elif a.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, default=str)[:4000])
    elif a.run:
        run()
    elif a.result:
        print(json.dumps(readout(), ensure_ascii=False, default=str)[:3000])


if __name__ == '__main__':
    main()
