"""DEV-DATA-READINESS-TRAINING-LENGTH (docs/DEV_DATA_READINESS_TRAINING_LENGTH_TASK_2026-09-17.md).

Only the number of AdamW updates of the shared readiness Consumer changes. 0 LLM, 0 API, no hashing.
  preflight: environment, sealing, structure (dev: T + C_A masks; follow-up: T + C_A/C_B/E masks, counts only), materials frozen
  -> dev: RD01B_V1, RD01B_Q1, RD02_V1, RD02_T1 x {P0 Linear, P1 Seasonal, P2 Linear+Hampel} x 4 seeds; one 2000-update trajectory
     per cell with checkpoints 100/250, C_A for all three (C_B/E of these jobs never read)
  -> 2 wiring checks (direct short run == checkpoint prefix; old default path == 2000 checkpoint trajectory)
  -> selection: r = mean_s C_A(n) / mean_s C_A(2000); J = domains equal -> jobs equal -> materials equal; argmin, ties fewer updates
  -> selected_consumer.json (before any follow-up fit)
  -> follow-up: RD01B_STP1, RD02_STP1 x 3 materials x 6 seeds (checkpoints saved; only t_star and 2000 get cells)
     -> consumer_frozen -> C_B -> freeze E -> barrier -> E (+ shadow Linear inputs) -> readout

Fits run one trajectory per subprocess (methods.ttha.batch_base.readiness worker); this controller never imports torch.
  --smoke | --preflight | --run | --result
"""
from __future__ import annotations

import os

# scikit-learn's __init__ runs os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True"); a controller that imported it would hand that
# mask to every worker. Record the shell state before any import and never pass the variable to a worker (task §6).
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
ROOT = REPO / '_scratch' / 'dev_data_readiness_training_length'
TASK = 'docs/DEV_DATA_READINESS_TRAINING_LENGTH_TASK_2026-09-17.md'
MODULE = 'evaluation.main_protocol_p4.batch_research_data_readiness_training_length'
PARENT_CONFIG = REPO / '_scratch' / 'dev_data_readiness_profile_transfer' / 'package_config.json'
PRSA_CLOSED_FROM_ROW = 24864                      # 2016+ region of the hourly PRSA files stays closed
DEV_JOBS = ('RD01B_V1', 'RD01B_Q1', 'RD02_V1', 'RD02_T1')
FOLLOW_JOBS = ('RD01B_STP1', 'RD02_STP1')
DOMAINS = ('RD01B', 'RD02')
DEV_SEEDS = (20261001, 20261002, 20261003, 20261004)
FOLLOW_SEEDS = (20261011, 20261012, 20261013, 20261014, 20261015, 20261016)
CANDIDATES = (100, 250, 2000)
OLD = 2000
SHORT = (100, 250)
TIE_TOL = 1e-12
P2_POLICY = rd.uniform_policy([{'op': 'impute_linear', 'strength': 1.0}, {'op': 'hampel_filter', 'window': 5, 'n_sigmas': 3.0}],
                              'P2 Linear+Hampel: linear completion then Hampel(window=5, n_sigmas=3.0)')
MATERIALS = (('P0_Linear', rd.BASELINE_LINEAR), ('P1_Seasonal', rd.FIXED_SEASONAL), ('P2_Linear_Hampel', P2_POLICY))
MIDS = tuple(m for m, _ in MATERIALS)
PAIRS = (('P1_Seasonal', 'P0_Linear'), ('P2_Linear_Hampel', 'P0_Linear'))
BLOCKS = ('c_a', 'c_b', 'e')
CAPS = {'dev': 48, 'follow': 36, 'check': 2, 'retry': 2, 'hard': 88}
WALL_S = 5400.
FIT_TIMEOUT_S = 300.
CHECK_CELL = ('RD02_V1', 'P0_Linear', DEV_SEEDS[0])
FOLLOW_GEOMETRY = {'RD01B_STP1': {'t': 10320, 'T': [9648, 10320], 'C_A': [10320, 10416], 'C_B': [10416, 10512], 'E': [10512, 10704]},
                   'RD02_STP1': {'t': 14160, 'T': [13488, 14160], 'C_A': [14160, 14256], 'C_B': [14256, 14352], 'E': [14352, 14544]}}
OLD_NAMES = {'RD02_V1': 5760, 'RD02_T1': 6960, 'RD02_T3': 9360, 'RD02_T4': 10560, 'RD02_Q1': 11760, 'RD02_Q2': 12960,
             'RD01B_V1': 6960, 'RD01B_Q1': 8160, 'RD01B_Q2': 9360, 'RD01_S1': 3360}
PARENT_COMMON = {'RD01B': REPO / '_scratch' / 'dev_data_readiness_profile_transfer' / 'test' / 'RD01B_Q1_common' / 'RD01B_Q1' / 'materials' / 'index.json',
                 'RD02': REPO / '_scratch' / 'dev_data_readiness_profile_transfer' / 'test' / 'RD02_Q1_common' / 'RD02_Q1' / 'materials' / 'index.json'}
NOISE_DIAGNOSTIC = [r'C:\Users\辉\AppData\Local\Temp\claude\C--Users---Desktop-Agent\4909bb9e-adaa-4ef3-ade3-ae76918c33c6\scratchpad\%s' % n
                    for n in ('noise_diag.py', 'noise_analyze.py', 'noise_RD02_Q1.json', 'noise_RD01B_Q2.json')]
READOUT_RULES = {
    'prediction': 'per job/material/block: ratio = mean_s loss(t_star) / mean_s loss(2000); materials equal within a job, one job per domain, '
                  'domains equal. Direction per domain on E is IMPROVED (<1) / WORSE (>1); overall IMPROVED_BOTH_DOMAINS / WORSE_BOTH_DOMAINS / MIXED. '
                  'Paired old2000_loss - selected_loss per seed reported with mean, sd, SE and sign counts.',
    'precision': 'pairs P1-P0 and P2-P0 only; d[s] = loss(P0,s) - loss(Pi,s); SE = sd(d)/sqrt(n). Primary: E block, 2 pairs x 2 jobs = 4 SE comparisons '
                 '(t_star vs 2000): SE_DOWN_ALL (4/4), SE_DOWN_MAJORITY (3/4), SE_NOT_DOWN (<=2/4). C_A and C_B reported per block, never averaged into seed noise.',
    'direction_conflict': 'per job/pair/checkpoint: sign of mean d on C_A versus C_B and E, plus per-seed signs; described, not called accuracy.',
    'cost': 'trajectory seconds at the t_star checkpoint versus at 2000 inside the same worker; direct short-run seconds from the wiring check; '
            'physical trajectories, logical models, cache/alias counts.',
    'no_change': 't_star == 2000: every follow-up comparison is NO_CHANGE; 2000 results still reported; unselected checkpoints are never scored.'}


def ds_of(job: str) -> str:
    return job.split('_')[0]


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


# ----------------------------------------------------------------------------------------------------------- budget ledger
class BudgetStop(RuntimeError):
    pass


class Ledger:
    def __init__(self, root: Path):
        self.path = root / 'budget.json'
        self.d = context.read_json(self.path) if self.path.exists() else {
            'caps': CAPS, 'wall_s': WALL_S, 'first_fit_epoch': None, 'physical': {'dev': 0, 'follow': 0, 'check': 0}, 'retries_used': 0,
            'failed': 0, 'cache': 0, 'alias': 0, 'events': []}

    def save(self):
        context.write_json(self.path, self.d)

    def total(self) -> int:
        return sum(self.d['physical'].values()) + self.d['retries_used']

    def reserve(self, stage: str, cell: str, retry: bool):
        if self.d['first_fit_epoch'] is not None and time.time() - self.d['first_fit_epoch'] > WALL_S:
            raise BudgetStop('real-run wall clock limit reached')
        if self.total() >= CAPS['hard']:
            raise BudgetStop('physical fit attempt hard cap reached')
        if retry:
            if self.d['retries_used'] >= CAPS['retry']:
                raise BudgetStop('retry cap reached')
            self.d['retries_used'] += 1
        else:
            if self.d['physical'][stage] >= CAPS[stage]:
                raise BudgetStop('%s trajectory cap reached' % stage)
            self.d['physical'][stage] += 1
        if self.d['first_fit_epoch'] is None:
            self.d['first_fit_epoch'] = time.time()
        self.d['events'].append({'kind': 'fit_started', 'stage': stage, 'cell': cell, 'retry': retry, 'epoch': time.time()})
        self.save()

    def finish(self, cell: str, ok: bool, seconds: float, reason: str = ''):
        if not ok:
            self.d['failed'] += 1
        self.d['events'].append({'kind': 'fit_finished', 'cell': cell, 'ok': ok, 'seconds': seconds, 'reason': reason, 'epoch': time.time()})
        self.save()

    def note(self, kind: str, cell: str):
        self.d[kind] += 1
        self.d['events'].append({'kind': kind, 'cell': cell, 'epoch': time.time()})
        self.save()


# ----------------------------------------------------------------------------------------------------------- preflight
def worker_env() -> dict:
    env = dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    return env


def environment() -> dict:
    import inspect
    parent = context.read_json(PARENT_CONFIG)['interpreter']
    probe = subprocess.run([sys.executable, '-B', '-c', 'import json, os, sys; from SelfEvolvingHarnessTS.operators import _provenance as p; '
                            'fp = p.dependency_fingerprint(); import torch; '
                            'print(json.dumps({"fp": fp, "torch": torch.__version__, "python": sys.version.split()[0]}))'],
                           cwd=str(REPO), capture_output=True, text=True, timeout=300, env=worker_env())
    if probe.returncode != 0:
        raise RuntimeError('environment probe failed: %s' % probe.stderr[-600:])
    got = json.loads(probe.stdout.strip().splitlines()[-1])
    main_src = inspect.getsource(rd.main)
    out = {'executable': sys.executable, 'python': got['python'], 'torch': got['torch'], 'dependency_fingerprint': got['fp'],
           'probe': 'fingerprint and torch version read in a separate process, so the controller never imports scikit-learn or torch',
           'parent_interpreter': parent, 'kmp_duplicate_lib_ok_in_shell_at_start': KMP_AT_START,
           'kmp_duplicate_lib_ok_in_controller_now': os.environ.get('KMP_DUPLICATE_LIB_OK'),
           'worker_env_kmp_duplicate_lib_ok': worker_env().get('KMP_DUPLICATE_LIB_OK'),
           'worker_initializes_openmp_before_torch': '_init_openmp_before_torch()' in main_src.split('ap = argparse')[0],
           'controller_imports_torch': 'torch' in sys.modules}
    ok = (got['fp'] == parent['dependency_fingerprint'] and out['python'] == parent['python'] and out['torch'] == parent['torch']
          and Path(sys.executable).resolve() == Path(parent['executable']).resolve() and KMP_AT_START is None
          and out['worker_env_kmp_duplicate_lib_ok'] is None and out['worker_initializes_openmp_before_torch'] and not out['controller_imports_torch'])
    if not ok:
        raise RuntimeError('environment differs from the parent readiness packages: %s' % out)
    return out


def _t_eligibility(seg: np.ndarray) -> dict:
    fin = np.isfinite(seg)
    lg = rd.legal_parents(seg)
    bad = [e for e in range(seg.shape[1]) if fin[:, e].sum() < rd.MIN_T_FINITE or not seg[fin[:, e], e].std() > spec.ZERO_SCALE_FLOOR
           or lg.counts[e] < rd.MIN_LEGAL_PARENTS]
    return {'eligible': not bad, 'ineligible_entities': bad, 'finite_per_entity_min': int(fin.sum(axis=0).min()), 'legal_parents_total': lg.n,
            'legal_parents_min': int(lg.counts.min())}


def _coverage(sl, origins) -> dict:
    obs = np.isfinite(rd.read_truth(sl, origins)).sum(axis=(1, 2))
    cells = len(origins) * rd.H
    frac = obs / cells
    return {'observed_cells_per_entity': [int(x) for x in obs], 'cells_per_entity': cells, 'min_fraction': float(frac.min()),
            'not_scorable_entities': [int(e) for e in np.flatnonzero(frac < rd.COVERAGE_MIN)], 'scorable': bool((frac >= rd.COVERAGE_MIN).all())}


def structure(job: str, follow: bool) -> dict:
    ds = ds_of(job)
    js = rd.resolve_job(ds, job)
    hi = max(js.e) + rd.H if follow else js.evaluate_rows[1]
    lo = js.t - rd.TRAIN_SPAN
    if ds == 'RD02' and hi > PRSA_CLOSED_FROM_ROW:
        raise RuntimeError('%s reaches the closed PRSA 2016+ region' % job)
    sl = rd.load_slice(ds, lo, hi)
    row = {'job': job, 'dataset': ds, 't': js.t, 'rows_read': [lo, hi], 'T': _t_eligibility(sl.rows(*js.train_range)), 'c_a': _coverage(sl, js.c_a),
           'purpose': ('eligibility and finite-mask coverage only (task §3.2); no value, error or ranking leaves this function' if follow
                       else 'T eligibility and C_A coverage of a development job; C_B/E rows not loaded')}
    if follow:
        g = FOLLOW_GEOMETRY[job]
        geom = {'t': js.t, 'T': list(js.train_range), 'C_A': [min(js.c_a), max(js.c_a) + rd.H], 'C_B': [min(js.c_b), max(js.c_b) + rd.H],
                'E': [min(js.e), max(js.e) + rd.H]}
        if geom != g:
            raise RuntimeError('follow-up geometry differs from the task table: %s' % geom)
        row.update({'geometry': geom, 'c_b': _coverage(sl, js.c_b), 'e': _coverage(sl, js.e)})
        row['status'] = 'ELIGIBLE' if row['T']['eligible'] and all(row[b]['scorable'] for b in BLOCKS) else 'INELIGIBLE'
    else:
        row['status'] = 'ELIGIBLE' if row['T']['eligible'] and row['c_a']['scorable'] else 'INELIGIBLE'
    return row


def material_check() -> dict:
    out = {}
    for d, p in PARENT_COMMON.items():
        idx = context.read_json(p)
        n = len(rd.DATASETS[d]['roster'])
        for mid, pol, parent_id in (('P0_Linear', rd.BASELINE_LINEAR, 'Baseline_Linear'), ('P1_Seasonal', rd.FIXED_SEASONAL, 'Fixed_Seasonal')):
            comp = rd.compile_policy(pol, [{}] * n)
            key = json.dumps(comp['assignment'], sort_keys=True, separators=(',', ':'))
            out['%s_%s' % (d, mid)] = {'parent_common': str(p), 'parent_id': parent_id, 'same_assignment': key == idx[parent_id]['key']}
    if not all(v['same_assignment'] for v in out.values()):
        raise RuntimeError('P0/P1 differ from the common baselines: %s' % out)
    return out


def preflight() -> dict:
    p = ROOT / 'frozen_config.json'
    if p.exists():
        return context.read_json(p)
    ROOT.mkdir(parents=True, exist_ok=True)
    env = environment()
    for j, t in OLD_NAMES.items():
        if rd.resolve_job(ds_of(j), j).t != t:
            raise RuntimeError('old job name changed meaning: %s' % j)
    dev = {j: structure(j, False) for j in DEV_JOBS}
    follow = {j: structure(j, True) for j in FOLLOW_JOBS}
    n_of = {d: len(rd.DATASETS[d]['roster']) for d in DOMAINS}
    mats = {}
    for mid, pol in MATERIALS:
        comp = rd.compile_policy(pol, [{}] * n_of['RD02'])
        mats[mid] = {'policy': pol, 'requested_steps': pol['default']['steps'], 'canonical_steps': comp['assignment'][0],
                     'canonicalized': comp['assignment'] != comp['requested_assignment']}
    cfg = {'package': 'DEV-DATA-READINESS-TRAINING-LENGTH', 'task': TASK, 'frozen_epoch': time.time(), 'frozen_local': now(),
           'exposure': 'EXPOSED_DEVELOPMENT for every job; RD01B_Q1 and RD02_T1 are former Test jobs explicitly demoted to development; '
                       'the STP1 jobs are only "post-freeze follow-up checks", not independent tests',
           'llm_calls': 0, 'api_calls': 0, 'new_hashes': 0, 'environment': env,
           'consumer_fixed': {k: v for k, v in rd.consumer_record(OLD).items() if k != 'n_updates'},
           'candidates': list(CANDIDATES), 'old_n_updates': OLD, 'checkpoints': list(SHORT),
           'jobs': {'dev': list(DEV_JOBS), 'follow': list(FOLLOW_JOBS), 'follow_geometry': FOLLOW_GEOMETRY},
           'structure': {'dev': dev, 'follow': follow},
           'sealing': {'prsa_closed_from_row': PRSA_CLOSED_FROM_ROW, 'max_prsa_row_read': max([dev[j]['rows_read'][1] for j in dev if ds_of(j) == 'RD02']
                                                                                              + [follow['RD02_STP1']['rows_read'][1]]),
                       'natural_final_read': False, 'kdd_rows_read_max': max([dev[j]['rows_read'][1] for j in dev if ds_of(j) == 'RD01B']
                                                                              + [follow['RD01B_STP1']['rows_read'][1]])},
           'materials': mats, 'material_check_against_common_baselines': material_check(),
           'seeds': {'dev': list(DEV_SEEDS), 'follow': list(FOLLOW_SEEDS), 'note': 'Consumer repetitions, not datasets or agent runs'},
           'selection_rule': 'r[j,p,n] = mean_s C_A(j,p,s,n) / max(mean_s C_A(j,p,s,2000), 1e-12); J(n) = mean over domains of mean over the '
                             'domain\'s dev jobs of mean over materials of r; t_star = argmin J, |J - min| <= 1e-12 counts as a tie and the '
                             'fewest updates win. Any dev job incomplete -> CALIBRATION_INCOMPLETE, no global selection.',
           'readout_rules': READOUT_RULES, 'caps': CAPS, 'wall_s_real_runs': WALL_S, 'fit_timeout_s': FIT_TIMEOUT_S,
           'wiring_checks': {'cell': list(CHECK_CELL), 'check_1': 'direct worker run n_updates=250 with checkpoint 100 (no feedback) versus the dev '
                                                                    'trajectory checkpoints u250/u100 (exact weights)',
                             'check_2': 'old default worker path (no training-length flags) versus the dev trajectory u2000 (exact weights, same C_A)'},
           'noise_diagnostic_sources': [{'path': x, 'exists': Path(x).exists(), 'use': 'read-only citation; it used opened E (not prospective)'}
                                        for x in NOISE_DIAGNOSTIC],
           'fit_command': [sys.executable, '-B', '-m', rd.FIT_MODULE, '--dataset', '<ds>', '--job', '<job>', '--run-dir', '<dir>', '--material', '<id>',
                           '--seed', '<seed>', '--n-updates', '2000', '--checkpoints', '100,250', '[--score-checkpoints <t_star>]']}
    context.write_json(p, cfg)
    return cfg


# ----------------------------------------------------------------------------------------------------------- materials and fits
def build_materials(run_dir: Path, job: str) -> dict:
    ctx = rd.open_job(ds_of(job), job, run_dir, stage='material')
    ov = ctx.overview()
    index = rd.load_index(ctx.job_dir)
    out = {}
    for mid, pol in MATERIALS:
        ref = rd.MaterialRef(**index[mid]) if mid in index else rd.build(ctx, rd.compile_policy(pol, ov['entities'])['assignment'], mid)
        index = rd.load_index(ctx.job_dir)
        out[mid] = {'alias_of': ref.alias_of, 'canonical_steps_entity0': ref.assignment[0], 'uniform': all(a == ref.assignment[0] for a in ref.assignment),
                    'legal_parents': ctx.legal.n}
    return out


def _cell_path(run_dir: Path, job: str, mid: str, seed: int, n: int) -> Path:
    return run_dir / job / 'cells' / (rd.length_cell_id(job, mid, seed, n) + '.json')


def _ok_cell(p: Path, n: int):
    if not p.exists():
        return None
    rec = context.read_json(p)
    if rd.cell_trained_updates(rec) != n:
        raise RuntimeError('cell %s carries another training length' % p.name)
    return rec if rec.get('status') == 'OK' else None


def run_worker(cmd: list, log: Path, timeout: float) -> tuple:
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        with log.open('w', encoding='utf-8') as fh:
            p = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, env=worker_env())
        return p.returncode, time.time() - start, ''
    except subprocess.TimeoutExpired:
        return -999, time.time() - start, 'worker_timeout'


def trajectory(ledger: Ledger, stage: str, run_dir: Path, job: str, mid: str, seed: int, n: int, checkpoints, score_checkpoints,
               feedback: bool = True, alias_of: str | None = None) -> dict:
    """One trajectory (or a verified cache / same-job alias copy). Returns {'status', 'cells': {n: record}}."""
    want = [n] + [c for c in checkpoints if c in score_checkpoints]
    cached = {c: _ok_cell(_cell_path(run_dir, job, mid, seed, c), c) for c in want}
    if all(cached.values()):
        ledger.note('cache', rd.length_cell_id(job, mid, seed, n))
        return {'status': 'CACHE', 'cells': cached}
    if alias_of:
        src = {c: _ok_cell(_cell_path(run_dir, job, alias_of, seed, c), c) for c in want}
        if all(src.values()):
            out = {}
            for c, rec in src.items():
                rec = {**rec, 'cell_id': rd.length_cell_id(job, mid, seed, c), 'material_id': mid, 'material_alias_of': alias_of, 'source_cell': rec['cell_id'],
                       'physical_fit': False}
                context.write_json(_cell_path(run_dir, job, mid, seed, c), rec)
                out[c] = rec
            ledger.note('alias', rd.length_cell_id(job, mid, seed, n))
            return {'status': 'ALIAS', 'cells': out}
    cid = rd.length_cell_id(job, mid, seed, n)
    cmd = [sys.executable, '-B', '-m', rd.FIT_MODULE, '--dataset', ds_of(job), '--job', job, '--run-dir', str(run_dir), '--material', mid,
           '--seed', str(seed), '--train-deadline', str(FIT_TIMEOUT_S - 30), '--n-updates', str(n)]
    if checkpoints:
        cmd += ['--checkpoints', ','.join(str(c) for c in checkpoints), '--score-checkpoints', ','.join(str(c) for c in score_checkpoints)]
    if not feedback:
        cmd.append('--no-feedback')
    for attempt in (0, 1):
        ledger.reserve(stage, cid, retry=attempt > 0)
        rc, secs, reason = run_worker(cmd, run_dir / job / 'fit_logs' / (cid + ('.log' if attempt == 0 else '.retry.log')), FIT_TIMEOUT_S)
        cells = {c: _ok_cell(_cell_path(run_dir, job, mid, seed, c), c) for c in want} if rc == 0 else {}
        ok = rc == 0 and all(cells.values())
        ledger.finish(cid, ok, secs, '' if ok else (reason or 'worker_failed rc=%s' % rc))
        print('FIT', cid, 'OK' if ok else 'FAILED', round(secs, 1), flush=True)
        if ok:
            return {'status': 'FIT', 'cells': cells, 'wall_seconds': secs, 'attempts': attempt + 1}
        if ledger.d['retries_used'] >= CAPS['retry']:
            break
    return {'status': 'TECHNICAL_FAILURE', 'cells': {}}


def compare_models(a: str, b: str) -> dict:
    p = subprocess.run([sys.executable, '-B', '-m', MODULE, '--compare-models', a, b], cwd=str(REPO), capture_output=True, text=True, timeout=120,
                       env=worker_env())
    if p.returncode != 0:
        raise RuntimeError('model comparison failed: %s' % p.stderr[-800:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def _compare_models_main(a: str, b: str) -> None:
    rd._init_openmp_before_torch()
    import torch
    sa, sb = torch.load(a, map_location='cpu'), torch.load(b, map_location='cpu')
    same_keys = list(sa) == list(sb)
    diffs = {k: float((sa[k].double() - sb[k].double()).abs().max()) for k in sa} if same_keys else {}
    print(json.dumps({'same_keys': same_keys, 'exact_equal': same_keys and all(torch.equal(sa[k], sb[k]) for k in sa),
                      'max_abs_diff': max(diffs.values()) if diffs else None}))


# ----------------------------------------------------------------------------------------------------------- stages
def dev_stage(cfg: dict, ledger: Ledger) -> dict:
    run_dir = ROOT / 'dev'
    out = {'started_local': now(), 'jobs': {}}
    for job in DEV_JOBS:
        if cfg['structure']['dev'][job]['status'] != 'ELIGIBLE':
            out['jobs'][job] = {'status': 'INELIGIBLE'}
            continue
        mats = build_materials(run_dir, job)
        rows = {}
        for mid in MIDS:
            for seed in DEV_SEEDS:
                r = trajectory(ledger, 'dev', run_dir, job, mid, seed, OLD, SHORT, SHORT, alias_of=mats[mid]['alias_of'])
                rows['%s|%d' % (mid, seed)] = r['status']
        out['jobs'][job] = {'status': 'DONE', 'materials': mats, 'trajectories': rows}
    out['finished_local'] = now()
    context.write_json(run_dir / 'dev_stage.json', out)
    return out


def wiring_checks(ledger: Ledger) -> dict:
    p = ROOT / 'checks' / 'wiring_checks.json'
    if p.exists():
        return context.read_json(p)
    job, mid, seed = CHECK_CELL
    dev_dir = ROOT / 'dev'
    dev_cells = {c: _ok_cell(_cell_path(dev_dir, job, mid, seed, c), c) for c in CANDIDATES}
    if not all(dev_cells.values()):
        return {'status': 'NOT_RUN', 'reason': 'dev trajectory of the check cell missing'}
    out = {'cell': list(CHECK_CELL), 'started_local': now()}
    # check 1: direct short run through the worker (250 updates, checkpoint 100, no feedback)
    d1 = ROOT / 'checks' / 'direct_short'
    build_materials(d1, job)
    with np.load(dev_dir / job / 'materials' / (mid + '.npz')) as a, np.load(d1 / job / 'materials' / (mid + '.npz')) as b:
        same_material = bool(np.array_equal(a['X'], b['X']))
    r1 = trajectory(ledger, 'check', d1, job, mid, seed, 250, (100,), (100,), feedback=False)
    if r1['status'] in ('FIT', 'CACHE'):
        c250, c100 = r1['cells'][250], r1['cells'][100]
        out['check_1'] = {'same_material_X': same_material, 'worker_status': r1['status'], 'direct_cell_id': c250['cell_id'],
                          'consumer_n_updates_recorded': [c250['consumer']['n_updates'], c100['consumer']['n_updates']],
                          'u250_vs_dev_checkpoint': compare_models(c250['model_path'], dev_cells[250]['model_path']),
                          'u100_vs_dev_checkpoint': compare_models(c100['model_path'], dev_cells[100]['model_path']),
                          'direct_250_train_seconds': c250['train']['seconds'], 'direct_250_wall_seconds': r1.get('wall_seconds'),
                          'dev_trajectory_seconds_at_250': dev_cells[250]['train']['seconds'], 'dev_trajectory_seconds_at_2000': dev_cells[2000]['train']['seconds']}
        out['check_1']['pass'] = same_material and out['check_1']['u250_vs_dev_checkpoint']['exact_equal'] and out['check_1']['u100_vs_dev_checkpoint']['exact_equal']
    else:
        out['check_1'] = {'status': r1['status'], 'pass': False}
    # check 2: the old default worker path (no training-length flags) in its own run dir
    d2 = ROOT / 'checks' / 'default_path'
    build_materials(d2, job)
    legacy_cell = d2 / job / 'cells' / (rd.train.cell_id(job, mid, seed) + '.json')
    if legacy_cell.exists():
        rec2, st2, w2 = context.read_json(legacy_cell), 'CACHE', None
    else:
        cmd = [sys.executable, '-B', '-m', rd.FIT_MODULE, '--dataset', ds_of(job), '--job', job, '--run-dir', str(d2), '--material', mid, '--seed', str(seed)]
        ledger.reserve('check', rd.train.cell_id(job, mid, seed), retry=False)
        rc, w2, reason = run_worker(cmd, d2 / job / 'fit_logs' / 'default_path.log', FIT_TIMEOUT_S)
        ok = rc == 0 and legacy_cell.exists()
        ledger.finish(rd.train.cell_id(job, mid, seed), ok, w2, '' if ok else (reason or 'worker_failed'))
        rec2, st2 = (context.read_json(legacy_cell), 'FIT') if ok else (None, 'TECHNICAL_FAILURE')
    if rec2:
        c2000 = dev_cells[2000]
        out['check_2'] = {'worker_status': st2, 'legacy_cell_id': rec2['cell_id'], 'legacy_record_has_consumer_field': 'consumer' in rec2,
                          'legacy_n_updates_done': rec2['train']['n_updates_done'], 'weights_vs_dev_u2000': compare_models(rec2['model_path'], c2000['model_path']),
                          'c_a_equal': rec2['scores']['c_a']['normalized_mse_macro'] == c2000['scores']['c_a']['normalized_mse_macro'],
                          'legacy_train_seconds': rec2['train']['seconds'], 'legacy_wall_seconds': w2}
        out['check_2']['pass'] = (out['check_2']['weights_vs_dev_u2000']['exact_equal'] and out['check_2']['c_a_equal']
                                  and not out['check_2']['legacy_record_has_consumer_field'] and rec2['cell_id'] == rd.train.cell_id(job, mid, seed))
    else:
        out['check_2'] = {'status': st2, 'pass': False}
    out['finished_local'] = now()
    context.write_json(p, out)
    return out


def dev_losses() -> dict:
    """L[job][mid][seed][n] = C_A macro NMSE (None when missing or not scorable)."""
    out = {}
    for job in DEV_JOBS:
        for mid in MIDS:
            for seed in DEV_SEEDS:
                for n in CANDIDATES:
                    rec = _ok_cell(_cell_path(ROOT / 'dev', job, mid, seed, n), n)
                    v = (rec or {}).get('scores', {}).get('c_a') or {}
                    out.setdefault(job, {}).setdefault(mid, {}).setdefault(str(seed), {})[str(n)] = v.get('normalized_mse_macro') if v.get('status') == 'SCORABLE' else None
    return out


def select() -> dict:
    p = ROOT / 'selected_consumer.json'
    if p.exists():
        return context.read_json(p)
    if (ROOT / 'follow').exists() and any((ROOT / 'follow').rglob('cells/*.json')):
        raise RuntimeError('follow-up cells exist before the selection')
    L = dev_losses()
    missing = sorted('%s|%s|%s|%s' % (j, m, s, n) for j in L for m in L[j] for s in L[j][m] for n, v in L[j][m][s].items() if v is None)
    if missing:
        rec = {'status': 'CALIBRATION_INCOMPLETE', 'missing_cells': missing, 'selected_local': now(), 'selected_epoch': time.time()}
        context.write_json(p, rec)
        return rec
    mean = {j: {m: {n: statistics.fmean(L[j][m][str(s)][str(n)] for s in DEV_SEEDS) for n in CANDIDATES} for m in MIDS} for j in DEV_JOBS}
    r = {j: {m: {n: mean[j][m][n] / max(mean[j][m][OLD], 1e-12) for n in CANDIDATES} for m in MIDS} for j in DEV_JOBS}
    dom_jobs = {d: [j for j in DEV_JOBS if ds_of(j) == d] for d in DOMAINS}
    by_domain = {d: {n: statistics.fmean(statistics.fmean(r[j][m][n] for m in MIDS) for j in dom_jobs[d]) for n in CANDIDATES} for d in DOMAINS}
    by_material = {m: {n: statistics.fmean(statistics.fmean(r[j][m][n] for j in dom_jobs[d]) for d in DOMAINS) for n in CANDIDATES} for m in MIDS}
    J = {n: statistics.fmean(by_domain[d][n] for d in DOMAINS) for n in CANDIDATES}
    best = min(J.values())
    t_star = min(n for n in CANDIDATES if J[n] - best <= TIE_TOL)
    rec = {'status': 'SELECTED', 't_star': t_star, 'J': {str(n): J[n] for n in CANDIDATES},
           'components': {'by_domain': {d: {str(n): v for n, v in by_domain[d].items()} for d in DOMAINS},
                          'by_material': {m: {str(n): v for n, v in by_material[m].items()} for m in MIDS},
                          'r': {j: {m: {str(n): r[j][m][n] for n in CANDIDATES} for m in MIDS} for j in DEV_JOBS},
                          'seed_mean_c_a': {j: {m: {str(n): mean[j][m][n] for n in CANDIDATES} for m in MIDS} for j in DEV_JOBS}},
           'rule': context.read_json(ROOT / 'frozen_config.json')['selection_rule'],
           'selected_consumer': rd.consumer_record(t_star),
           'direct_short_training_command': [sys.executable, '-B', '-m', rd.FIT_MODULE, '--dataset', '<ds>', '--job', '<job>', '--run-dir', '<dir>',
                                             '--material', '<id>', '--seed', '<seed>', '--n-updates', str(t_star)],
           'follow_up_compares': [t_star, OLD] if t_star != OLD else [OLD], 'no_change': t_star == OLD,
           'selected_local': now(), 'selected_epoch': time.time(), 'written_before_any_follow_up_fit': True,
           'note': 'Development C_A only; the ratio is used only to pool across jobs; this choice is final for the package.'}
    context.write_json(p, rec)
    return rec


def worker_stage(stage: str, job: str, run_dir: Path, **extra) -> dict:
    cmd = [sys.executable, '-B', '-m', rd.FIT_MODULE, '--stage', stage, '--output', str(run_dir), '--dataset', ds_of(job), '--job', job]
    for k, v in extra.items():
        cmd += ['--' + k.replace('_', '-'), str(v)]
    rc, secs, reason = run_worker(cmd, run_dir / job / 'label_logs' / (stage + '.log'), 600.)
    if rc != 0:
        raise RuntimeError('%s stage failed for %s (%s)' % (stage, job, reason or rc))
    return {'stage': stage, 'job': job, 'seconds': secs, 'finished_local': now()}


def follow_stage(cfg: dict, sel: dict, ledger: Ledger) -> dict:
    run_dir = ROOT / 'follow'
    t_star = sel['t_star']
    score = [t_star] if t_star != OLD else []
    jobs = [j for j in FOLLOW_JOBS if cfg['structure']['follow'][j]['status'] == 'ELIGIBLE']
    out = {'started_local': now(), 'eligible_jobs': jobs, 'ineligible_jobs': [j for j in FOLLOW_JOBS if j not in jobs], 'jobs': {}, 'labels': []}
    for job in jobs:
        mats = build_materials(run_dir, job)
        rows = {}
        for mid in MIDS:
            for seed in FOLLOW_SEEDS:
                r = trajectory(ledger, 'follow', run_dir, job, mid, seed, OLD, SHORT, score, alias_of=mats[mid]['alias_of'])
                rows['%s|%d' % (mid, seed)] = r['status']
        out['jobs'][job] = {'materials': mats, 'trajectories': rows}
    # every planned model exists (or is recorded missing) before any follow-up label
    for job in jobs:
        cells = sorted(p.stem for p in (run_dir / job / 'cells').glob('*.json'))
        cf = run_dir / job / 'consumer_frozen.json'
        if not cf.exists():
            context.write_json(cf, {'job_id': job, 'selected_consumer': sel['selected_consumer'], 't_star': t_star, 'compared': sel['follow_up_compares'],
                                    'cells': cells, 'frozen_local': now(), 'frozen_epoch': time.time(),
                                    'note': 'no agent: the frozen object is the Consumer and the full evaluation set'})
    for job in jobs:
        if not (run_dir / job / 'c_b_scores.json').exists():
            out['labels'].append(worker_stage('c_b', job, run_dir, prerequisite='consumer_frozen.json'))
    for job in jobs:
        if not (run_dir / job / 'e_frozen.json').exists():
            out['labels'].append(worker_stage('freeze_e', job, run_dir))
    barrier = run_dir / 'all_e_predictions_frozen.json'
    if not barrier.exists():
        if not all((run_dir / j / 'e_frozen.json').exists() for j in jobs):
            raise RuntimeError('E predictions incomplete; barrier not written')
        context.write_json(barrier, {'jobs': {j: context.read_json(run_dir / j / 'e_frozen.json')['frozen_at_local'] for j in jobs},
                                     'written_local': now(), 'written_epoch': time.time()})
    for job in jobs:
        if not (run_dir / job / 'e_scores.json').exists():
            score_e_after_barrier(run_dir, job)
            out['labels'].append({'stage': 'score_e', 'job': job, 'finished_local': now()})
    out['finished_local'] = now()
    context.write_json(run_dir / 'follow_stage.json', out)
    return out


def score_e_after_barrier(run_dir: Path, job: str) -> None:
    barrier = run_dir / 'all_e_predictions_frozen.json'
    if not barrier.exists() or job not in context.read_json(barrier)['jobs']:
        raise PermissionError('E barrier missing or job not in the frozen cohort')
    worker_stage('score_e', job, run_dir)


# ----------------------------------------------------------------------------------------------------------- readout
def _stats(v: list) -> dict:
    n = len(v)
    if n == 0:
        return {'n': 0}
    mean = statistics.fmean(v)
    sd = statistics.stdev(v) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    out = {'n': n, 'mean': mean, 'sd': sd, 'se': se, 'positive': sum(x > 0 for x in v), 'negative': sum(x < 0 for x in v), 'zero': sum(x == 0 for x in v)}
    if se is not None and n > 1:
        from scipy.stats import t as _t
        q = float(_t.ppf(0.975, n - 1))
        out['t95_interval'] = [mean - q * se, mean + q * se]
        out['t95_assumption'] = 'fixed-endpoint Student t over seeds (df=n-1), treating seed differences as i.i.d.; not a guaranteed error rate'
    return out


def follow_losses(run_dir: Path, job: str) -> dict:
    """loss[mid][n][block][seed] (main) and shadow[mid][n][block][seed] (Linear serving inputs; C_B/E only)."""
    cells = {p.stem: context.read_json(p) for p in (run_dir / job / 'cells').glob('*.json')}
    cb = context.read_json(run_dir / job / 'c_b_scores.json')['cells'] if (run_dir / job / 'c_b_scores.json').exists() else {}
    es = context.read_json(run_dir / job / 'e_scores.json')['cells'] if (run_dir / job / 'e_scores.json').exists() else {}
    main, shadow, per_origin = {}, {}, {}
    for cid, c in cells.items():
        if c.get('status') != 'OK':
            continue
        n, mid, seed = rd.cell_trained_updates(c), c['material_id'], str(c['model_seed'])
        blocks = {'c_a': c['scores'].get('c_a'), 'c_b': (cb.get(cid) or {}).get('c_b'), 'e': (es.get(cid) or {}).get('e')}
        sh = {'c_b': (cb.get(cid) or {}).get('c_b_shadow_baseline_inputs'), 'e': (es.get(cid) or {}).get('e_shadow_baseline_inputs')}
        for b, s in blocks.items():
            if s and s.get('status') == 'SCORABLE':
                main.setdefault(mid, {}).setdefault(n, {}).setdefault(b, {})[seed] = s['normalized_mse_macro']
                per_origin.setdefault(mid, {}).setdefault(n, {}).setdefault(b, {})[seed] = [statistics.fmean(x for x in row if x is not None)
                                                                                        for row in s['per_origin_entity_normalized_mse']]
        for b, s in sh.items():
            if s and s.get('status') == 'SCORABLE':
                shadow.setdefault(mid, {}).setdefault(n, {}).setdefault(b, {})[seed] = s['normalized_mse_macro']
    return {'main': main, 'shadow': shadow, 'per_origin_mean': per_origin}


def readout() -> dict:
    cfg = context.read_json(ROOT / 'frozen_config.json')
    sel = context.read_json(ROOT / 'selected_consumer.json') if (ROOT / 'selected_consumer.json').exists() else None
    led = context.read_json(ROOT / 'budget.json') if (ROOT / 'budget.json').exists() else {}
    res = {'package': cfg['package'], 'written_local': now(), 'selection': sel, 'wiring_checks': context.read_json(ROOT / 'checks' / 'wiring_checks.json')
           if (ROOT / 'checks' / 'wiring_checks.json').exists() else None}
    # development
    L = dev_losses()
    dev = {'c_a_macro': L, 'pairs_c_a': {}}
    for j in DEV_JOBS:
        for pi, p0 in PAIRS:
            for n in CANDIDATES:
                d = [L[j][p0][str(s)][str(n)] - L[j][pi][str(s)][str(n)] for s in DEV_SEEDS
                     if L[j][p0][str(s)][str(n)] is not None and L[j][pi][str(s)][str(n)] is not None]
                dev['pairs_c_a'].setdefault(j, {}).setdefault('%s-%s' % (pi, p0), {})[str(n)] = _stats(d)
    dev['train_seconds_at_checkpoint'] = {}
    secs = {n: [] for n in CANDIDATES}
    for p in (ROOT / 'dev').rglob('cells/*__u2000.json'):
        c = context.read_json(p)
        if c.get('physical_fit'):
            secs[OLD].append(c['train']['seconds'])
            for s_ in c.get('checkpoints_saved', []):
                secs[s_['n_updates']].append(s_['train_seconds_at_checkpoint'])
    dev['train_seconds_at_checkpoint'] = {str(n): _stats(v) for n, v in secs.items()}
    res['development'] = dev
    # follow-up
    if sel and sel.get('status') == 'SELECTED' and (ROOT / 'follow' / 'follow_stage.json').exists():
        t_star = sel['t_star']
        fol = {'t_star': t_star, 'no_change': t_star == OLD, 'jobs': {}}
        run_dir = ROOT / 'follow'
        for job in context.read_json(run_dir / 'follow_stage.json')['eligible_jobs']:
            fl = follow_losses(run_dir, job)
            M, S = fl['main'], fl['shadow']
            jr = {'losses': M, 'shadow_losses': S, 'per_origin_mean': fl['per_origin_mean'], 'prediction': {}, 'pairs': {}, 'shadow_pairs': {}, 'coverage': {}}
            for mid in MIDS:
                for b in BLOCKS:
                    old = M.get(mid, {}).get(OLD, {}).get(b, {})
                    new = M.get(mid, {}).get(t_star, {}).get(b, {})
                    common = sorted(set(old) & set(new))
                    jr['coverage']['%s|%s' % (mid, b)] = {'old_seeds': len(old), 'selected_seeds': len(new), 'paired': len(common)}
                    jr['prediction'].setdefault(mid, {})[b] = {
                        'old2000_mean': statistics.fmean(old.values()) if old else None, 'selected_mean': statistics.fmean(new.values()) if new else None,
                        'ratio_selected_over_old': (statistics.fmean(new[s] for s in common) / statistics.fmean(old[s] for s in common)) if common else None,
                        'old_minus_selected': _stats([old[s] - new[s] for s in common])}
            for pi, p0 in PAIRS:
                key = '%s-%s' % (pi, p0)
                for n in sorted({t_star, OLD}):
                    for b in BLOCKS:
                        a0, a1 = M.get(p0, {}).get(n, {}).get(b, {}), M.get(pi, {}).get(n, {}).get(b, {})
                        jr['pairs'].setdefault(key, {}).setdefault(str(n), {})[b] = {
                            'd_per_seed': {s: a0[s] - a1[s] for s in sorted(set(a0) & set(a1))}, **_stats([a0[s] - a1[s] for s in sorted(set(a0) & set(a1))])}
                        if b != 'c_a':
                            sh1 = S.get(pi, {}).get(n, {}).get(b, {})
                            jr['shadow_pairs'].setdefault(key, {}).setdefault(str(n), {})[b] = {
                                'note': 'P0 main loss minus Pi model fed the Linear serving inputs (training-material side); not an additive decomposition',
                                **_stats([a0[s] - sh1[s] for s in sorted(set(a0) & set(sh1))])}
                se = {str(n): {b: jr['pairs'][key][str(n)][b].get('se') for b in BLOCKS} for n in sorted({t_star, OLD})}
                jr['pairs'][key]['se_selected_over_old'] = {b: (None if not se[str(OLD)][b] else (se[str(t_star)][b] / se[str(OLD)][b] if se[str(t_star)][b] is not None else None))
                                                            for b in BLOCKS}
                jr['pairs'][key]['direction'] = {str(n): {b: (None if jr['pairs'][key][str(n)][b].get('mean') is None else
                                                             int(np.sign(jr['pairs'][key][str(n)][b]['mean']))) for b in BLOCKS} for n in sorted({t_star, OLD})}
            p0_shadow_equal = all(abs(M.get('P0_Linear', {}).get(n, {}).get(b, {}).get(s, 0) - S.get('P0_Linear', {}).get(n, {}).get(b, {}).get(s, 0)) == 0
                                  for n in (t_star, OLD) for b in ('c_b', 'e') for s in M.get('P0_Linear', {}).get(n, {}).get(b, {}))
            jr['p0_shadow_equals_main'] = p0_shadow_equal
            fol['jobs'][job] = jr
        # pooled prediction ratios: materials equal within a job, one job per domain, domains equal
        pooled = {}
        for b in BLOCKS:
            per_dom = {}
            for job, jr in fol['jobs'].items():
                vals = [jr['prediction'][m][b]['ratio_selected_over_old'] for m in MIDS if jr['prediction'][m][b]['ratio_selected_over_old'] is not None]
                if len(vals) == len(MIDS):
                    per_dom[ds_of(job)] = statistics.fmean(vals)
            pooled[b] = {'by_domain': per_dom, 'domains_equal': statistics.fmean(per_dom.values()) if len(per_dom) == len(DOMAINS) else None}
        fol['pooled_ratio_selected_over_old'] = pooled
        e_dom = pooled['e']['by_domain']
        if t_star == OLD:
            fol['prediction_verdict'] = 'NO_CHANGE'
        elif len(e_dom) < len(DOMAINS):
            fol['prediction_verdict'] = 'INCOMPLETE'
        else:
            dirs = {d: ('IMPROVED' if v < 1 else 'WORSE' if v > 1 else 'EQUAL') for d, v in e_dom.items()}
            fol['prediction_by_domain_e'] = dirs
            fol['prediction_verdict'] = ('IMPROVED_BOTH_DOMAINS' if all(x == 'IMPROVED' for x in dirs.values()) else
                                         'WORSE_BOTH_DOMAINS' if all(x == 'WORSE' for x in dirs.values()) else 'MIXED')
        if t_star == OLD:
            fol['precision_verdict'] = 'NO_CHANGE'
        else:
            comps = [(job, key, jr['pairs'][key]['se_selected_over_old']['e']) for job, jr in fol['jobs'].items() for key in jr['pairs']]
            down = sum(1 for _, _, v in comps if v is not None and v < 1)
            fol['precision_e_se_comparisons'] = [{'job': j, 'pair': k, 'se_ratio': v} for j, k, v in comps]
            fol['precision_verdict'] = ('INCOMPLETE' if len(comps) < 4 or any(v is None for _, _, v in comps) else
                                        'SE_DOWN_ALL' if down == 4 else 'SE_DOWN_MAJORITY' if down == 3 else 'SE_NOT_DOWN')
        conflicts = []
        for job, jr in fol['jobs'].items():
            for key in (k for k in jr['pairs'] if '-' in k):
                for n, dd in jr['pairs'][key]['direction'].items():
                    ca = dd['c_a']
                    conflicts.append({'job': job, 'pair': key, 'n_updates': int(n), 'sign_c_a': ca, 'sign_c_b': dd['c_b'], 'sign_e': dd['e'],
                                      'c_a_disagrees_with': [b for b in ('c_b', 'e') if dd[b] is not None and ca is not None and dd[b] != ca]})
        fol['direction_conflicts'] = conflicts
        secs = {n: [] for n in CANDIDATES}
        for p in run_dir.rglob('cells/*__u2000.json'):
            c = context.read_json(p)
            if c.get('physical_fit'):
                secs[OLD].append(c['train']['seconds'])
                for s_ in c.get('checkpoints_saved', []):
                    secs[s_['n_updates']].append(s_['train_seconds_at_checkpoint'])
        fol['train_seconds_at_checkpoint'] = {str(n): _stats(v) for n, v in secs.items()}
        res['follow_up'] = fol
    fits = [e for e in led.get('events', []) if e['kind'] == 'fit_finished']
    res['cost'] = {'physical_trajectories': led.get('physical'), 'retries_used': led.get('retries_used'), 'failed_attempts': led.get('failed'),
                   'cache': led.get('cache'), 'alias': led.get('alias'), 'physical_attempts_total': sum((led.get('physical') or {}).values()) + (led.get('retries_used') or 0),
                   'hard_cap': CAPS['hard'], 'fit_wall_seconds_total': sum(e['seconds'] for e in fits),
                   'logical_models': {'dev_scored_cells': sum(1 for _ in (ROOT / 'dev').rglob('cells/*.json')) if (ROOT / 'dev').exists() else 0,
                                      'follow_scored_cells': sum(1 for _ in (ROOT / 'follow').rglob('cells/*.json')) if (ROOT / 'follow').exists() else 0,
                                      'follow_saved_unscored_checkpoints': sum(1 for p in (ROOT / 'follow').rglob('runs/*.pt')) - sum(1 for _ in (ROOT / 'follow').rglob('cells/*.json'))
                                      if (ROOT / 'follow').exists() else 0},
                   'real_run_wall_seconds': (max(e['epoch'] for e in led['events']) - led['first_fit_epoch']) if led.get('first_fit_epoch') else None,
                   'llm_calls': 0, 'api_calls': 0}
    context.write_json(ROOT / 'result.json', res)
    write_tables(json.loads(json.dumps(res, default=str)))
    return res


def write_tables(res: dict) -> str:
    """Markdown tables recomputed from result.json (no new numbers)."""
    f = lambda x, d=3: '—' if x is None else ('%.*f' % (d, x))
    lines = ['# DEV-DATA-READINESS-TRAINING-LENGTH tables', '', 'Generated from result.json by `--result`. Losses are missing-aware macro NMSE '
             '(lower is better). d = loss(P0) − loss(Pi): positive means the repair Pi is better. old2000 − selected: positive means shorter training is better.', '']
    sel = res.get('selection') or {}
    dev = res.get('development') or {}
    L = dev.get('c_a_macro') or {}
    lines += ['## 1. Development C_A (seed means, 4 seeds)', '', '| Job | Material | 100 | 250 | 2000 | r(100) | r(250) |', '|---|---|---:|---:|---:|---:|---:|']
    comp = sel.get('components') or {}
    for j in DEV_JOBS:
        for m in MIDS:
            mm = (comp.get('seed_mean_c_a') or {}).get(j, {}).get(m, {})
            rr = (comp.get('r') or {}).get(j, {}).get(m, {})
            lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (j, m, f(mm.get('100')), f(mm.get('250')), f(mm.get('2000')), f(rr.get('100')), f(rr.get('250'))))
    if sel.get('status') == 'SELECTED':
        lines += ['', '| J component | 100 | 250 | 2000 |', '|---|---:|---:|---:|',
                  '| J (domains equal) | %s |' % ' | '.join(f(sel['J'][str(n)], 4) for n in CANDIDATES)]
        for d, v in comp['by_domain'].items():
            lines.append('| domain %s | %s |' % (d, ' | '.join(f(v[str(n)], 4) for n in CANDIDATES)))
        for m, v in comp['by_material'].items():
            lines.append('| material %s | %s |' % (m, ' | '.join(f(v[str(n)], 4) for n in CANDIDATES)))
        lines += ['', '**t_star = %d** (%s)' % (sel['t_star'], sel['rule']), '']
    else:
        lines += ['', 'Selection status: %s' % sel.get('status'), '']
    lines += ['## 2. Development C_A treatment differences (descriptive only; not used for selection)', '',
              '| Job | Pair | n | mean d | SE | +/−/0 |', '|---|---|---:|---:|---:|---|']
    for j, pairs in (dev.get('pairs_c_a') or {}).items():
        for k, byn in pairs.items():
            for n, s in byn.items():
                lines.append('| %s | %s | %s | %s | %s | %s/%s/%s |' % (j, k, n, f(s.get('mean')), f(s.get('se')), s.get('positive'), s.get('negative'), s.get('zero')))
    wc = res.get('wiring_checks') or {}
    lines += ['', '## 3. Wiring checks', '', '```json', json.dumps(wc, indent=1, ensure_ascii=False, default=str)[:3000], '```', '']
    fol = res.get('follow_up')
    if fol:
        ts = fol['t_star']
        lines += ['## 4. Follow-up prediction (6 seeds): selected = %d vs old = 2000' % ts, '',
                  '| Job | Material | Block | old2000 mean | selected mean | ratio sel/old | old−sel mean | SE | +/−/0 |', '|---|---|---|---:|---:|---:|---:|---:|---|']
        for job, jr in fol['jobs'].items():
            for m in MIDS:
                for b in BLOCKS:
                    p = jr['prediction'][m][b]
                    s = p['old_minus_selected']
                    lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s/%s/%s |' % (job, m, b, f(p['old2000_mean']), f(p['selected_mean']), f(p['ratio_selected_over_old']),
                                                                                    f(s.get('mean')), f(s.get('se')), s.get('positive'), s.get('negative'), s.get('zero')))
        lines += ['', '| Block | RD01B ratio | RD02 ratio | domains equal |', '|---|---:|---:|---:|']
        for b in BLOCKS:
            p = fol['pooled_ratio_selected_over_old'][b]
            lines.append('| %s | %s | %s | %s |' % (b, f(p['by_domain'].get('RD01B'), 4), f(p['by_domain'].get('RD02'), 4), f(p['domains_equal'], 4)))
        lines += ['', 'Prediction verdict (E): **%s**; precision verdict (E SE): **%s**' % (fol.get('prediction_verdict'), fol.get('precision_verdict')), '',
                  '## 5. Treatment-difference precision (6 seeds)', '', '| Job | Pair | Block | mean d @2000 | SE @2000 | +/−/0 @2000 | mean d @sel | SE @sel | +/−/0 @sel | SE ratio sel/old |',
                  '|---|---|---|---:|---:|---|---:|---:|---|---:|']
        for job, jr in fol['jobs'].items():
            for k, v in jr['pairs'].items():
                for b in BLOCKS:
                    o, s_ = v[str(OLD)][b], v[str(ts)][b]
                    lines.append('| %s | %s | %s | %s | %s | %s/%s/%s | %s | %s | %s/%s/%s | %s |' % (
                        job, k, b, f(o.get('mean')), f(o.get('se')), o.get('positive'), o.get('negative'), o.get('zero'),
                        f(s_.get('mean')), f(s_.get('se')), s_.get('positive'), s_.get('negative'), s_.get('zero'), f(v['se_selected_over_old'][b])))
        lines += ['', '## 6. Training-material side (Pi model fed Linear serving inputs; C_B/E)', '', '| Job | Pair | n | Block | mean | SE |', '|---|---|---:|---|---:|---:|']
        for job, jr in fol['jobs'].items():
            for k, byn in jr['shadow_pairs'].items():
                for n, bb in byn.items():
                    for b, s in bb.items():
                        lines.append('| %s | %s | %s | %s | %s | %s |' % (job, k, n, b, f(s.get('mean')), f(s.get('se'))))
        lines += ['', '## 7. Direction of mean d by block', '', '| Job | Pair | n | C_A | C_B | E | C_A disagrees with |', '|---|---|---:|---:|---:|---:|---|']
        for c in fol['direction_conflicts']:
            lines.append('| %s | %s | %d | %s | %s | %s | %s |' % (c['job'], c['pair'], c['n_updates'], c['sign_c_a'], c['sign_c_b'], c['sign_e'], ', '.join(c['c_a_disagrees_with']) or '—'))
        lines += ['', '## 8. P0 Linear per-origin block means (seed-averaged)', '', '| Job | n | Block | per-origin means |', '|---|---:|---|---|']
        for job, jr in fol['jobs'].items():
            po = jr['per_origin_mean'].get('P0_Linear', {})
            for n in sorted({ts, OLD}):
                for b in BLOCKS:
                    rows = list((po.get(n) or po.get(str(n)) or {}).get(b, {}).values())
                    if rows:
                        means = [statistics.fmean(r[i] for r in rows) for i in range(len(rows[0]))]
                        lines.append('| %s | %d | %s | %s |' % (job, n, b, ', '.join(f(x) for x in means)))
        lines += ['', '## 9. Training seconds at checkpoints (inside the same worker)', '', '| Stage | 100 | 250 | 2000 |', '|---|---:|---:|---:|']
        for name, blk in (('dev', dev.get('train_seconds_at_checkpoint')), ('follow-up', fol.get('train_seconds_at_checkpoint'))):
            if blk:
                lines.append('| %s | %s |' % (name, ' | '.join(f((blk.get(str(n)) or {}).get('mean'), 2) for n in CANDIDATES)))
    lines += ['', '## 10. Cost', '', '```json', json.dumps(res.get('cost'), indent=1, default=str), '```', '']
    text = '\n'.join(lines)
    (ROOT / 'tables.md').write_text(text, encoding='utf-8')
    return text


# ----------------------------------------------------------------------------------------------------------- smoke (synthetic; no real-data fit)
def smoke() -> dict:
    rd._init_openmp_before_torch()
    import torch
    from methods.ttha.batch_base import train
    from evaluation.main_protocol_p4 import batch_research_data_readiness as drs
    checks = {}
    s = 20261001
    full = train.batch_indices(s, n_pool=777)
    checks['batch_prefix'] = bool(np.array_equal(train.batch_indices(s, n_pool=777, n_updates=250), full[:250])) and full.shape == (2000, 64)
    rng = np.random.RandomState(3)
    X, Y = rng.randn(400, 192).astype(np.float32), rng.randn(400, 48).astype(np.float32)
    bidx = train.batch_indices(s, n_pool=400)
    states = {}

    def keep(step, model, loss, secs):
        states[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}

    eq = lambda a, b: all(torch.equal(a[k], b[k]) for k in a)
    mA, _ = train.train_arm(X, Y, None, train.batch_indices(s, 400, 300), s, n_updates=300, checkpoints=(100, 250), on_checkpoint=keep)
    mB, rB = train.train_arm(X, Y, None, train.batch_indices(s, 400, 100), s, n_updates=100)
    mC, _ = train.train_arm(X, Y, None, train.batch_indices(s, 400, 250), s, n_updates=250)
    mD, _ = train.train_arm(X, Y, None, train.batch_indices(s, 400, 300), s, n_updates=300)
    checks['checkpoint_100_equals_direct_100'] = eq(states[100], mB.state_dict()) and rB.n_updates_done == 100
    checks['checkpoint_250_equals_direct_250'] = eq(states[250], mC.state_dict())
    checks['checkpoints_do_not_change_final'] = eq(mA.state_dict(), mD.state_dict())
    mE, rE = train.train_arm(X, Y, None, bidx, s)
    mF, _ = train.train_arm(X, Y, None, bidx, s, n_updates=2000, checkpoints=(100, 250), on_checkpoint=keep)
    checks['default_path_2000_equals_explicit_2000_with_checkpoints'] = rE.n_updates_done == 2000 and eq(mE.state_dict(), mF.state_dict())
    bad = 0
    for f in (lambda: train.check_n_updates(0), lambda: train.check_n_updates(2001), lambda: train.check_n_updates(2.5),
              lambda: train.train_arm(X, Y, None, bidx, s, n_updates=100, checkpoints=(100,), on_checkpoint=keep),
              lambda: train.train_arm(X, Y, None, bidx[:50], s, n_updates=100)):
        try:
            f()
        except ValueError:
            bad += 1
    checks['invalid_lengths_refused'] = bad == 5
    checks['cell_ids'] = (rd.length_cell_id('RD02_V1', 'P0_Linear', 1, 100) == 'RD02_V1__P0_Linear__s1__u100'
                          and rd.cell_trained_updates({}) == 2000 and rd.cell_trained_updates({'consumer': {'n_updates': 250}}) == 250)
    checks['old_job_names_unchanged'] = all(rd.resolve_job(ds_of(j), j).t == t for j, t in OLD_NAMES.items())
    geo_ok = True
    for j, g in FOLLOW_GEOMETRY.items():
        js = rd.resolve_job(ds_of(j), j)
        geo_ok &= {'t': js.t, 'T': list(js.train_range), 'C_A': [min(js.c_a), max(js.c_a) + 48], 'C_B': [min(js.c_b), max(js.c_b) + 48],
                   'E': [min(js.e), max(js.e) + 48]} == g
    checks['follow_up_job_binding'] = geo_ok
    # worker path on a registered synthetic dataset (in-process; the real worker CLI is exercised by the two wiring checks)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        drs._register_synthetic()
        try:
            ctx = rd.open_job('RDX', 'RDX_V1', td, 'material')
            ov = ctx.overview()
            rd.build(ctx, rd.compile_policy(rd.BASELINE_LINEAR, ov['entities'])['assignment'], 'P0_Linear')
            r1 = rd.fit_one('RDX', 'RDX_V1', str(td), 'P0_Linear', 7, n_updates=300, checkpoints=[100, 250], score_checkpoints=[250])
            cdir = td / 'RDX_V1' / 'cells'
            names = sorted(p.name for p in cdir.glob('*.json'))
            checks['worker_records_consumer_and_checkpoint_cells'] = (names == ['RDX_V1__P0_Linear__s7__u250.json', 'RDX_V1__P0_Linear__s7__u300.json']
                                                                      and r1['consumer']['n_updates'] == 300 and r1['physical_fit'] is True
                                                                      and (td / 'RDX_V1' / 'runs' / 'RDX_V1__P0_Linear__s7__u100.pt').exists()
                                                                      and context.read_json(cdir / 'RDX_V1__P0_Linear__s7__u250.json')['consumer']['n_updates'] == 250
                                                                      and context.read_json(cdir / 'RDX_V1__P0_Linear__s7__u250.json')['scores']['c_a']['status'] == 'SCORABLE')
            fake = context.read_json(cdir / 'RDX_V1__P0_Linear__s7__u250.json')
            fake['consumer']['n_updates'] = 2000
            context.write_json(cdir / 'RDX_V1__P0_Linear__s7__u250.json', fake)
            try:
                rd.fit_one('RDX', 'RDX_V1', str(td), 'P0_Linear', 7, n_updates=300, checkpoints=[100, 250])
                checks['cache_refuses_other_length'] = False
            except RuntimeError:
                checks['cache_refuses_other_length'] = True
            try:
                _ok_cell(cdir / 'RDX_V1__P0_Linear__s7__u250.json', 250)
                checks['controller_cache_refuses_other_length'] = False
            except RuntimeError:
                checks['controller_cache_refuses_other_length'] = True
            r0 = rd.fit_one('RDX', 'RDX_V1', str(td), 'P0_Linear', 8)
            checks['old_default_worker_path'] = r0['cell_id'] == 'RDX_V1__P0_Linear__s8' and 'consumer' not in r0 and r0['train']['n_updates_done'] == 2000
            try:
                rd.open_c_b(td, 'RDX', 'RDX_V1', prerequisite='consumer_frozen.json')
                checks['c_b_refused_without_frozen_consumer'] = False
            except PermissionError:
                checks['c_b_refused_without_frozen_consumer'] = True
            try:
                rd.open_c_b(td, 'RDX', 'RDX_V1', prerequisite='anything.json')
                checks['c_b_unknown_prerequisite_refused'] = False
            except ValueError:
                checks['c_b_unknown_prerequisite_refused'] = True
            try:
                score_e_after_barrier(td, 'RDX_V1')
                checks['e_refused_without_barrier'] = False
            except PermissionError:
                checks['e_refused_without_barrier'] = True
        finally:
            rd.DATASETS.pop('RDX', None)
    ctl = subprocess.run([sys.executable, '-B', '-c', 'import sys; import %s; print("torch" in sys.modules)' % MODULE], cwd=str(REPO),
                         capture_output=True, text=True, timeout=120)
    checks['controller_module_does_not_import_torch'] = ctl.stdout.strip().splitlines()[-1:] == ['False']
    rec = {'checks': checks, 'all_pass': all(checks.values()), 'real_data_fits': 0, 'written_local': now()}
    ROOT.mkdir(parents=True, exist_ok=True)
    context.write_json(ROOT / 'smoke_record.json', rec)
    return rec


# ----------------------------------------------------------------------------------------------------------- entry
def run() -> None:
    cfg = preflight()
    if not all(context.read_json(ROOT / 'smoke_record.json')['checks'].values()):
        raise RuntimeError('smoke has not passed')
    ledger = Ledger(ROOT)
    status = {'started_local': now()}
    try:
        dev_stage(cfg, ledger)
        status['wiring_checks'] = wiring_checks(ledger)
        sel = select()
        status['selection'] = sel.get('status')
        if sel.get('status') == 'SELECTED':
            follow_stage(cfg, sel, ledger)
            status['exit'] = 'COMPLETE'
        else:
            status['exit'] = 'CALIBRATION_INCOMPLETE'
    except BudgetStop as ex:
        status['exit'] = 'PARTIAL'
        status['stop'] = str(ex)
    status['finished_local'] = now()
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
    a = ap.parse_args()
    if a.compare_models:
        _compare_models_main(*a.compare_models)
    elif a.smoke:
        print(json.dumps(smoke(), indent=1))
    elif a.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, default=str)[:4000])
    elif a.run:
        run()
    elif a.result:
        print(json.dumps(readout(), ensure_ascii=False, default=str)[:4000])


if __name__ == '__main__':
    main()
