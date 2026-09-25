"""DEV-DATA-READINESS-LTSV-CORE-REPAIR-SIGNAL (docs/DEV_DATA_READINESS_LTSV_CORE_REPAIR_SIGNAL_TASK_2026-09-18.md).

Project adaptation of the LTSV block valuation on the pretrained Time-MoE-50M (method table: METHOD.md in the output root).
This controller runs in the base environment and never imports torch. Data preparation and scoring run as base-environment
workers of this module (readiness.score_block unchanged); every TSFM update / prediction runs in the isolated venv through
evaluation.main_protocol_p4.ltsv_tsfm_helper.

  preflight   environment, exposure, MLP/score reuse identity, cache snapshot, T-only arrays + stratified blocks + confirmation
              index streams (worker), frozen_config.json
  checks      C_A Linear inputs (worker); TSFM real checks and timing (<= 12 extra updates); resource estimate x1.5
  blocks      per job: theta0 + one AdamW step per (material, block) -> C_A reference predictions; scored; R_LTSV frozen
  confirm     2 jobs x 3 materials x 3 seeds x 100 updates (batch 4); C_A scored; R_TSF_CA / R_P0 frozen (selections.json)
  later       C_B/E Linear inputs -> predictions of theta0 + 18 models -> barrier -> scores
  readout     result.json, tables.md, checks.json

  --preflight | --smoke | --run | --result
"""
from __future__ import annotations

import os

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
ROOT = REPO / '_scratch' / 'dev_data_readiness_ltsv_core_repair_signal'
TL = REPO / '_scratch' / 'dev_data_readiness_training_length'
SA = REPO / '_scratch' / 'dev_data_readiness_short_adapt_signal'
TASK = 'docs/DEV_DATA_READINESS_LTSV_CORE_REPAIR_SIGNAL_TASK_2026-09-18.md'
MODULE = 'evaluation.main_protocol_p4.batch_research_data_readiness_ltsv_core'
HELPER = 'evaluation.main_protocol_p4.ltsv_tsfm_helper'
VENV_PY = ROOT / 'tsfm_env' / 'venv' / 'Scripts' / 'python.exe'
PRSA_CLOSED_FROM_ROW = 24864
JOBS = ('RD01B_Q1', 'RD02_T1')
DOMAINS = {'RD01B_Q1': 'RD01B', 'RD02_T1': 'RD02'}
CACHE = {'RD01B_Q1': TL / 'dev' / 'RD01B_Q1', 'RD02_T1': TL / 'dev' / 'RD02_T1'}
GEOMETRY = {'RD01B_Q1': {'t': 8160, 'T': [7488, 8160], 'C_A': [8160, 8256], 'C_B': [8256, 8352], 'E': [8352, 8544], 'entities': 32},
            'RD02_T1': {'t': 6960, 'T': [6288, 6960], 'C_A': [6960, 7056], 'C_B': [7056, 7152], 'E': [7152, 7344], 'entities': 12}}
EXPOSURE = {'RD01B_Q1': REPO / '_scratch' / 'dev_data_readiness_profile_transfer' / 'test' / 'RD01B_Q1_F0' / 'RD01B_Q1',
            'RD02_T1': REPO / '_scratch' / 'dev_data_readiness_domain_skill_v1' / 'target' / 'RD02_T1_no_skill' / 'RD02_T1'}
MLP_SEEDS = (20261001, 20261002, 20261003, 20261004)
CONFIRM_SEEDS = (20261201, 20261202, 20261203)
CONFIRM_UPDATES, CONFIRM_BATCH = 100, 4
N_STRATA = 4
P2_POLICY = rd.uniform_policy([{'op': 'impute_linear', 'strength': 1.0}, {'op': 'hampel_filter', 'window': 5, 'n_sigmas': 3.0}],
                              'P2 Linear+Hampel: linear completion then Hampel(window=5, n_sigmas=3.0)')
MATERIALS = (('P0_Linear', rd.BASELINE_LINEAR), ('P1_Seasonal', rd.FIXED_SEASONAL), ('P2_Linear_Hampel', P2_POLICY))
MIDS = tuple(m for m, _ in MATERIALS)
REPAIRS = MIDS[1:]
BLOCKS = ('c_a', 'c_b', 'e')
RULES = ('R_LTSV', 'R_TSF_CA', 'R_P0')
CAPS = {'block': 528, 'confirm': 1800, 'extra': 12, 'retry': 100, 'hard': 2440}
WALL_NUMERIC_S = 3 * 3600.
TIE_TOL = 1e-12
CHECK_JOB = 'RD01B_Q1'


def ds_of(job: str) -> str:
    return job.split('_')[0]


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def jd(job: str, root: Path = ROOT) -> Path:
    return root / 'jobs' / job


# ----------------------------------------------------------------------------------------------------------- environment
def worker_env() -> dict:
    env = dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    return env


PROBE = '''
import json, sys
from SelfEvolvingHarnessTS.operators import _provenance as p
fp = p.dependency_fingerprint()
import torch
print(json.dumps({"fp": fp, "torch": torch.__version__, "python": sys.version.split()[0]}))
'''


def environment() -> dict:
    parent = context.read_json(TL / 'frozen_config.json')['environment']
    pr = subprocess.run([sys.executable, '-B', '-c', PROBE], cwd=str(REPO), capture_output=True, text=True, timeout=300, env=worker_env())
    if pr.returncode != 0:
        raise RuntimeError('base environment probe failed: %s' % pr.stderr[-600:])
    got = json.loads(pr.stdout.strip().splitlines()[-1])
    base = {'executable': sys.executable, 'python': got['python'], 'torch': got['torch'], 'dependency_fingerprint': got['fp'],
            'same_as_training_length': got['fp'] == parent['dependency_fingerprint'] and got['torch'] == parent['torch'] and got['python'] == parent['python'],
            'kmp_duplicate_lib_ok_in_shell_at_start': KMP_AT_START, 'controller_imports_torch': 'torch' in sys.modules}
    probe = context.read_json(ROOT / 'checks' / 'probe_load_generate.json')
    pip = subprocess.run([str(VENV_PY), '-m', 'pip', 'list', '--format', 'json'], cwd=str(REPO), capture_output=True, text=True, timeout=300, env=worker_env())
    pkgs = {r['name'].lower(): r['version'] for r in json.loads(pip.stdout)} if pip.returncode == 0 else {}
    tsfm = {'venv_python': str(VENV_PY), 'base_interpreter': 'D:/Anaconda_envs/envs/project/python.exe (venv --system-site-packages; that env unchanged)',
            'probe': probe, 'packages': {k: pkgs.get(k) for k in ('torch', 'transformers', 'tokenizers', 'huggingface-hub', 'safetensors', 'numpy')},
            'installed_into_venv': 'transformers 4.40.1, tokenizers 0.19.1, huggingface-hub 0.36.2 from pypi.org (the configured mirror had no transformers 4.40.1)',
            'model': {'repo': 'Maple728/TimeMoE-50M', 'revision': '446753ee48ff3726d0606a81d0092d54acee995e', 'files': {
                p.name: p.stat().st_size for p in (ROOT / 'tsfm_env' / 'model').iterdir()}},
            'time_moe_source': {'repo': 'Time-MoE/Time-MoE', 'commit': 'd3524fdaa5c0e83a29a6cc01828194dd870ec7b6',
                                'files': sorted(str(p.relative_to(ROOT / 'tsfm_env' / 'src')) for p in (ROOT / 'tsfm_env' / 'src').rglob('*.py'))},
            'downloads_measured': {'model_files_seconds': 36.7, 'model_bytes': 226760264 + 891 + 69, 'pip_seconds_pypi': 23, 'pip_seconds_failed_mirror': 11,
                                   'pip_wheels_mb': 9.0 + 2.2 + 0.57}}
    if not base['same_as_training_length'] or KMP_AT_START is not None or base['controller_imports_torch']:
        raise RuntimeError('base environment differs from TRAINING-LENGTH: %s' % base)
    if not probe['generate_vs_predict']['within_atol1e-5_rtol1e-4']:
        raise RuntimeError('differentiable predict does not match the official generate')
    return {'base': base, 'tsfm': tsfm}


def exposure_identity(job: str) -> dict:
    d, js = EXPOSURE[job], rd.resolve_job(ds_of(job), job)
    es = context.read_json(d / 'e_scores.json')
    with np.load(d / 'scaler.npz') as sc:
        binding = (str(sc['profile']) == rd.PROFILE and str(sc['dataset']) == js.dataset and int(sc['t']) == js.t and [str(x) for x in sc['roster']] == js.roster)
    if not (es['job_id'] == job and es['rows_read'] == GEOMETRY[job]['E'] and binding and len(js.roster) == GEOMETRY[job]['entities']):
        raise RuntimeError('exposure record of %s does not bind' % job)
    return {'record': str(d / 'e_scores.json'), 'e_rows_read': es['rows_read'], 'scored_at_local': es['scored_at_local'], 'status': 'EXPOSED_DEVELOPMENT'}


def geometry(js) -> dict:
    return {'t': js.t, 'T': list(js.train_range), 'C_A': [min(js.c_a), max(js.c_a) + rd.H], 'C_B': [min(js.c_b), max(js.c_b) + rd.H],
            'E': [min(js.e), max(js.e) + rd.H], 'entities': len(js.roster)}


def mlp_inventory(job: str) -> dict:
    """The 24 cached 2000-update MLP models and the SHORT-ADAPT Linear-input scores of exactly those models (reused, read after selections)."""
    sa_cfg = context.read_json(SA / 'frozen_config.json')
    if sa_cfg['reference_seeds'][job] != list(MLP_SEEDS) or 'Linear inputs' not in sa_cfg['proxy']['serving']:
        raise RuntimeError('SHORT-ADAPT reference seeds / serving differ for %s' % job)
    fz = context.read_json(SA / 'jobs' / job / 'later_frozen.json')
    by_cell = {m['cell_id']: m for m in fz['models']}
    rows = {}
    for mid in MIDS:
        for s in MLP_SEEDS:
            cid = rd.length_cell_id(job, mid, s, 2000)
            mp, cp = CACHE[job] / 'runs' / (cid + '.pt'), CACHE[job] / 'cells' / (cid + '.json')
            c = context.read_json(cp)
            ok = (mp.exists() and c.get('status') == 'OK' and c['material_id'] == mid and c['model_seed'] == s and rd.cell_trained_updates(c) == 2000
                  and Path(c['model_path']).resolve() == mp.resolve() and cid in by_cell and Path(by_cell[cid]['model_path']).resolve() == mp.resolve())
            rows[cid] = {'model': str(mp), 'bound': ok}
    if not all(r['bound'] for r in rows.values()):
        raise RuntimeError('MLP models / SHORT-ADAPT scores do not bind for %s' % job)
    return {'models': len(rows), 'score_sources': {'c_a': str(SA / 'jobs' / job / 'ca_scores.json'), 'c_b_e': str(SA / 'jobs' / job / 'later_scores.json')},
            'semantics': 'same cached models, P0 Linear serving inputs, readiness.score_block (SHORT-ADAPT, protocol checks 16/16)'}


def cache_files() -> list:
    files = []
    for job in JOBS:
        c = CACHE[job]
        files += sorted((c / 'materials').glob('*')) + [c / 'scaler.npz', EXPOSURE[job] / 'e_scores.json']
        for mid in MIDS:
            for s in MLP_SEEDS:
                cid = rd.length_cell_id(job, mid, s, 2000)
                files += [c / 'runs' / (cid + '.pt'), c / 'cells' / (cid + '.json')]
        files += [SA / 'jobs' / job / f for f in ('ca_scores.json', 'later_scores.json', 'later_frozen.json')]
    return sorted(set(files))


def snapshot(files) -> dict:
    return {str(p): ([p.stat().st_size, p.stat().st_mtime_ns] if p.exists() else None) for p in files}


# ----------------------------------------------------------------------------------------------------------- pure rules (smoke-tested)
def block_table(ent: np.ndarray, k: np.ndarray, n_entities: int, t0: int, differs: dict) -> list:
    """4 contiguous strata of each entity's time-ordered legal parents (np.array_split); the representative is index (size-1)//2."""
    total, out = int(ent.size), []
    for e in range(n_entities):
        idx = np.flatnonzero(ent == e)
        if idx.size < N_STRATA or np.any(np.diff(k[idx]) <= 0):
            raise RuntimeError('entity %d: fewer than %d legal parents or unsorted windows' % (e, N_STRATA))
        for s, st in enumerate(np.array_split(idx, N_STRATA)):
            p = int(st[(st.size - 1) // 2])
            out.append({'block': len(out), 'entity': e, 'stratum': s, 'stratum_size': int(st.size), 'stratum_parents': [int(st[0]), int(st[-1])],
                        'parent': p, 'k': int(k[p]), 'x_rows': [t0 + int(k[p]), t0 + int(k[p]) + rd.L], 'y_rows': [t0 + int(k[p]) + rd.L, t0 + int(k[p]) + rd.L + rd.H],
                        'weight': st.size / total, 'x_differs_from_p0': {m: bool(differs[m][p]) for m in REPAIRS}})
    return out


def aggregate_u(blocks: list, d: dict):
    """d: {block: value or None}; None when any block is missing."""
    if any(d.get(b['block']) is None for b in blocks):
        return None
    return float(sum(b['weight'] * d[b['block']] for b in blocks))


def coverage_map(blocks: list, d: dict, n_entities: int, t_len: int = rd.TRAIN_SPAN) -> tuple:
    """Point value = mean d of the valued blocks covering that T position (X and y rows of the window); uncovered = NaN."""
    sums, cnt = np.zeros((n_entities, t_len)), np.zeros((n_entities, t_len))
    for b in blocks:
        v = d.get(b['block'])
        if v is None:
            continue
        sums[b['entity'], b['k']: b['k'] + rd.L + rd.H] += v
        cnt[b['entity'], b['k']: b['k'] + rd.L + rd.H] += 1
    m = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    segs = []
    for e in range(n_entities):
        cov = cnt[e] > 0
        runs, i = [], 0
        while i < t_len:
            if cov[i]:
                j = i
                while j < t_len and cov[j]:
                    j += 1
                runs.append({'t_offset': [i, j], 'mean': float(m[e, i:j].mean())})
                i = j
            else:
                i += 1
        segs.append({'entity': e, 'covered_fraction': float(cov.mean()), 'covered_mean': float(m[e][cov].mean()) if cov.any() else None, 'segments': runs})
    return m, segs


def pick(values: dict, maximize: bool):
    if any(values.get(m) is None for m in MIDS):
        return None
    best = max(values.values()) if maximize else min(values.values())
    return next(m for m in MIDS if abs(values[m] - best) <= TIE_TOL)


def nulls(a: np.ndarray) -> list:
    return [[None if not np.isfinite(x) else round(float(x), 8) for x in row] for row in a]


# ----------------------------------------------------------------------------------------------------------- base-environment workers
def bound_arrays(ctx, cache_dir: Path) -> tuple:
    lg, sc = ctx.legal, ctx.scaler
    with np.load(cache_dir / 'scaler.npz') as old:
        rd._check_binding(old, ctx.job)
        same = (np.array_equal(old['mean'], sc.mean) and np.array_equal(old['scale'], sc.scale) and np.array_equal(old['legal_ent'], lg.ent)
                and np.array_equal(old['legal_k'], lg.k))
    if not same:
        raise RuntimeError('scaler / legal parent order differs from the cache for %s' % ctx.job.job_id)
    index = context.read_json(cache_dir / 'materials' / 'index.json')
    N, X = len(ctx.job.roster), {}
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
    return X, y


def prep_t(job: str, root: Path = ROOT, cache_dir: Path | None = None) -> dict:
    ctx = rd.open_job(ds_of(job), job, root / 'jobs', stage='material')
    X, y = bound_arrays(ctx, cache_dir or CACHE[job])
    lg = ctx.legal
    differs = {m: (X[m] != X['P0_Linear']).any(axis=1) for m in REPAIRS}
    blocks = block_table(lg.ent, lg.k, len(ctx.job.roster), ctx.job.train_range[0], differs)
    idx = {s: np.random.RandomState(s).randint(0, lg.n, size=(CONFIRM_UPDATES, CONFIRM_BATCH)) for s in CONFIRM_SEEDS}
    np.savez(jd(job, root) / 'train_arrays.npz', y=y, ent=lg.ent, k=lg.k, **{'X_' + m: X[m] for m in MIDS}, **{'confirm_idx_%d' % s: v for s, v in idx.items()})
    rec = {'job_id': job, 'rows_read': list(ctx.job.material_rows), 'legal_parents': lg.n, 'entities': len(ctx.job.roster), 'blocks': blocks,
           'material_windows_changed_vs_p0': {m: int(differs[m].sum()) for m in REPAIRS},
           'blocks_changed_vs_p0': {m: sum(b['x_differs_from_p0'][m] for b in blocks) for m in REPAIRS},
           'block_weight_changed_vs_p0': {m: float(sum(b['weight'] for b in blocks if b['x_differs_from_p0'][m])) for m in REPAIRS},
           'planned_physical_block_updates': len(blocks) + sum(b['x_differs_from_p0'][m] for b in blocks for m in REPAIRS),
           'confirm_idx': {str(s): v.tolist() for s, v in idx.items()}, 'written_local': now()}
    context.write_json(jd(job, root) / 'blocks.json', rec)
    return rec


def prep_ca(job: str, root: Path = ROOT) -> dict:
    ctx = rd.open_job(ds_of(job), job, root / 'jobs', stage='evaluate')
    if ctx.slice.row_end > min(ctx.job.c_b):
        raise PermissionError('C_A slice reaches C_B rows')
    Xs, rec = rd.serve_inputs(ctx.slice, ctx.job.c_a, [[] for _ in ctx.job.roster], ctx.scaler)
    np.savez(jd(job, root) / 'ca_inputs.npz', X_ref=Xs, origins=np.array(ctx.job.c_a))
    return {'job_id': job, 'rows_read': list(ctx.job.evaluate_rows), 'serve_inputs': rec}


def later_outputs(root: Path) -> list:
    return [str(p) for p in root.glob('jobs/*/later_*')] + ([str(root / 'later_barrier.json')] if (root / 'later_barrier.json').exists() else [])


def _losses(pred_norm: np.ndarray, truth: np.ndarray, sc) -> dict:
    raw = pred_norm * sc.scale[:, None, None] + sc.mean[:, None, None]
    full = rd.score_block(raw, truth, sc)
    per = [rd.score_block(raw[:, [o], :], truth[:, [o], :], sc) for o in range(truth.shape[1])]
    return {'A': full['normalized_mse_macro'], 'status': full['status'], 'o': [p['normalized_mse_macro'] for p in per], 'o_status': [p['status'] for p in per],
            'min_observed_fraction': full['coverage']['min_observed_fraction']}


def score_blocks(job: str, root: Path = ROOT) -> dict:
    if later_outputs(root) or (root / 'selections.json').exists():
        raise RuntimeError('block scores must precede every later output and the selections')
    bdir = jd(job, root) / 'blocks'
    meta = context.read_json(jd(job, root) / 'blocks.json')
    ctx = rd.open_job(ds_of(job), job, root / 'jobs', stage='evaluate')
    truth, sc = rd.read_truth(ctx.slice, ctx.job.c_a), ctx.scaler
    with np.load(bdir / 'theta0_ca.npz') as z:
        L0 = _losses(z['pred'], truth, sc)
    blocks, out = meta['blocks'], {'job_id': job, 'theta0': L0, 'blocks': [], 'U': {}, 'U_origin': {}}
    N = meta['entities']
    d_all = {m: {} for m in MIDS}
    d_o = {m: {0: {}, 1: {}} for m in MIDS}
    for b in blocks:
        with np.load(bdir / ('b%03d.npz' % b['block'])) as z:
            preds = z['pred']
        Ls = {m: _losses(preds[i], truth, sc) for i, m in enumerate(MIDS)}
        v = {m: (None if Ls[m]['A'] is None or L0['A'] is None else L0['A'] - Ls[m]['A']) for m in MIDS}
        vo = {m: [(None if Ls[m]['o'][o] is None or L0['o'][o] is None else L0['o'][o] - Ls[m]['o'][o]) for o in range(2)] for m in MIDS}
        row = {'block': b['block'], 'entity': b['entity'], 'stratum': b['stratum'], 'weight': b['weight'], 'loss': {m: Ls[m] for m in MIDS}, 'v': v, 'v_origin': vo, 'd': {}, 'd_origin': {}}
        for m in MIDS:
            row['d'][m] = None if v[m] is None or v['P0_Linear'] is None else v[m] - v['P0_Linear']
            row['d_origin'][m] = [None if vo[m][o] is None or vo['P0_Linear'][o] is None else vo[m][o] - vo['P0_Linear'][o] for o in range(2)]
            d_all[m][b['block']] = row['d'][m]
            for o in range(2):
                d_o[m][o][b['block']] = row['d_origin'][m][o]
        out['blocks'].append(row)
    maps = {}
    for m in MIDS:
        out['U'][m] = 0.0 if m == 'P0_Linear' else aggregate_u(blocks, d_all[m])
        out['U_origin'][m] = [0.0 if m == 'P0_Linear' else aggregate_u(blocks, d_o[m][o]) for o in range(2)]
        if m != 'P0_Linear':
            arr, segs = coverage_map(blocks, d_all[m], N)
            maps[m] = arr
            out.setdefault('coverage_summary', {})[m] = segs
    out['v_P0_weighted'] = aggregate_u(blocks, {b['block']: r['v']['P0_Linear'] for b, r in zip(blocks, out['blocks'])})
    out['exact_zero_d_for_identical_x'] = all(r['d'][m] == 0.0 for b, r in zip(blocks, out['blocks']) for m in REPAIRS if not b['x_differs_from_p0'][m])
    out['R_LTSV'] = pick(out['U'], maximize=True)
    out['R_LTSV_by_origin'] = [pick({m: out['U_origin'][m][o] for m in MIDS}, maximize=True) for o in range(2)]
    out['origin_sign_conflicts'] = {m: [None if u is None else int(np.sign(u)) for u in out['U_origin'][m]] for m in REPAIRS}
    np.savez(jd(job, root) / 'coverage_maps.npz', **{m: maps[m] for m in maps})
    out['coverage_maps_json'] = {m: nulls(maps[m]) for m in maps}
    out['scored_epoch'] = time.time()
    context.write_json(jd(job, root) / 'block_scores.json', out)
    return out


def score_confirm(job: str, root: Path = ROOT) -> dict:
    if later_outputs(root) or (root / 'selections.json').exists():
        raise RuntimeError('confirmation C_A scores must precede the selections and every later output')
    cdir = jd(job, root) / 'confirm'
    ctx = rd.open_job(ds_of(job), job, root / 'jobs', stage='evaluate')
    truth, sc = rd.read_truth(ctx.slice, ctx.job.c_a), ctx.scaler
    out = {'job_id': job, 'models': {}}
    for m in MIDS:
        for s in CONFIRM_SEEDS:
            with np.load(cdir / ('%s__s%d_ca.npz' % (m, s))) as z:
                out['models']['%s__s%d' % (m, s)] = _losses(z['pred'], truth, sc)
    with np.load(jd(job, root) / 'blocks' / 'theta0_ca.npz') as z:
        out['theta0'] = _losses(z['pred'], truth, sc)
    out['scored_epoch'] = time.time()
    context.write_json(jd(job, root) / 'confirm_ca_scores.json', out)
    return out


def prep_later(job: str, root: Path = ROOT) -> dict:
    if not (root / 'selections.json').exists():
        raise PermissionError('selections are not frozen; no C_B/E row is loaded')
    js = rd.resolve_job(ds_of(job), job)
    ctx_e = rd.open_job(ds_of(job), job, root / 'jobs', stage='e_input')
    cb_rows = (min(js.c_b) - rd.L, max(js.c_b))
    sl = rd.load_slice(ds_of(job), *cb_rows)
    Xcb, rcb = rd.serve_inputs(sl, js.c_b, [[] for _ in js.roster], ctx_e.scaler)
    Xe, re_ = rd.serve_inputs(ctx_e.slice, js.e, [[] for _ in js.roster], ctx_e.scaler)
    np.savez(jd(job, root) / 'later_inputs.npz', X_c_b=Xcb, X_e=Xe, c_b_origins=np.array(js.c_b), e_origins=np.array(js.e))
    rec = {'job_id': job, 'rows_read': {'c_b_inputs': list(cb_rows), 'e_inputs': list(js.e_input_rows)}, 'serve_inputs': {'c_b': rcb, 'e': re_},
           'selections_epoch': context.read_json(root / 'selections.json')['frozen_epoch'], 'written_epoch': time.time()}
    context.write_json(jd(job, root) / 'later_inputs.json', rec)
    return rec


def score_later(job: str, root: Path = ROOT) -> dict:
    barrier = root / 'later_barrier.json'
    if not barrier.exists() or job not in context.read_json(barrier)['jobs']:
        raise PermissionError('prediction barrier missing; no C_B/E target row is loaded')
    js = rd.resolve_job(ds_of(job), job)
    fz = context.read_json(jd(job, root) / 'later_frozen.json')
    ctx_cb = rd.open_job(ds_of(job), job, root / 'jobs', stage='c_b')
    ctx_et = rd.open_job(ds_of(job), job, root / 'jobs', stage='e_target')
    y_cb, y_e = rd.read_truth(ctx_cb.slice, js.c_b), rd.read_truth(ctx_et.slice, js.e)
    out = {'job_id': job, 'barrier_epoch': context.read_json(barrier)['written_epoch'], 'rows_read': {'c_b': list(js.c_b_rows), 'e_target': list(js.e_target_rows)}, 'models': {}}
    with np.load(jd(job, root) / 'later_predictions.npz') as z:
        if not (np.array_equal(z['c_b_origins'], np.array(js.c_b)) and np.array_equal(z['e_origins'], np.array(js.e))):
            raise RuntimeError('frozen origins differ from the job table')
        for name in fz['models']:
            out['models'][name] = {'c_b': _losses(z['c_b__' + name], y_cb, ctx_cb.scaler), 'e': _losses(z['e__' + name], y_e, ctx_et.scaler)}
    out['scored_epoch'] = time.time()
    context.write_json(jd(job, root) / 'later_scores.json', out)
    return out


# ----------------------------------------------------------------------------------------------------------- controller stages
class BudgetStop(RuntimeError):
    pass


class Ledger:
    """Optimizer updates, counted from the workers' update logs (each executed update is logged right after opt.step())."""

    def __init__(self, root: Path = ROOT):
        self.path = root / 'budget.json'
        self.d = context.read_json(self.path) if self.path.exists() else {'caps': CAPS, 'wall_numeric_s': WALL_NUMERIC_S, 'numeric_start_epoch': None,
                                                                          'workers': [], 'failures': 0, 'retries': 0}

    def save(self):
        context.write_json(self.path, self.d)

    @staticmethod
    def used(stage: str) -> int:
        n = 0
        for p in ROOT.glob('jobs/*/%s_updates.jsonl' % stage) if stage != 'extra' else [ROOT / 'checks' / 'extra_updates.jsonl']:
            if p.exists():
                n += sum(1 for line in p.read_text(encoding='utf-8').splitlines() if line.strip())
        return n

    def totals(self) -> dict:
        u = {s: self.used(s) for s in ('block', 'confirm', 'extra')}
        u['total'] = sum(u.values())
        return u

    def allowance(self, stage: str, retry: bool) -> int:
        u = self.totals()
        extra = CAPS['retry'] if retry else 0
        return max(0, min(CAPS[stage] + extra - u[stage], CAPS['hard'] - u['total']))

    def start_numeric(self):
        if self.d['numeric_start_epoch'] is None:
            self.d['numeric_start_epoch'] = time.time()
            self.save()

    def deadline(self) -> float:
        return (self.d['numeric_start_epoch'] or time.time()) + WALL_NUMERIC_S

    def run(self, name: str, cmd: list, log: Path, timeout: float = 4 * 3600.) -> tuple:
        if self.d['numeric_start_epoch'] is not None and time.time() > self.deadline():
            raise BudgetStop('numeric wall clock limit reached before %s' % name)
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            with log.open('w', encoding='utf-8') as fh:
                rc = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, env=worker_env()).returncode
        except subprocess.TimeoutExpired:
            rc = -999
        secs = time.time() - t0
        self.d['workers'].append({'name': name, 'rc': rc, 'seconds': secs, 'start_epoch': t0, 'end_epoch': time.time(), 'updates_after': self.totals()})
        if rc != 0:
            self.d['failures'] += 1
        self.save()
        print('WORKER', name, 'rc=%s' % rc, round(secs, 1), flush=True)
        return rc, secs


def base_cmd(*args) -> list:
    return [sys.executable, '-B', '-m', MODULE] + [str(a) for a in args]


def tsfm_cmd(*args) -> list:
    return [str(VENV_PY), '-B', '-m', HELPER] + [str(a) for a in args]


def preflight() -> dict:
    p = ROOT / 'frozen_config.json'
    if p.exists():
        return context.read_json(p)
    (ROOT / 'checks').mkdir(parents=True, exist_ok=True)
    env = environment()
    for job in JOBS:
        js = rd.resolve_job(ds_of(job), job)
        if geometry(js) != GEOMETRY[job]:
            raise RuntimeError('%s geometry differs from the task table' % job)
    exposure = {j: exposure_identity(j) for j in JOBS}
    mlp = {j: mlp_inventory(j) for j in JOBS}
    context.write_json(ROOT / 'checks' / 'cache_snapshot_before.json', {'written_local': now(), 'files': snapshot(cache_files())})
    prep = {}
    for job in JOBS:
        if not (jd(job) / 'blocks.json').exists():
            pr = subprocess.run(base_cmd('--worker-prep-t', job), cwd=str(REPO), capture_output=True, text=True, timeout=1200, env=worker_env())
            if pr.returncode != 0:
                raise RuntimeError('T preparation failed for %s: %s' % (job, pr.stderr[-800:]))
        b = context.read_json(jd(job) / 'blocks.json')
        prep[job] = {k: v for k, v in b.items() if k not in ('blocks', 'confirm_idx')}
        prep[job]['block_parents'] = [x['parent'] for x in b['blocks']]
        prep[job]['confirm_idx'] = b['confirm_idx']
    planned_block = sum(prep[j]['planned_physical_block_updates'] for j in JOBS)
    cfg = {'package': 'DEV-DATA-READINESS-LTSV-CORE-REPAIR-SIGNAL', 'task': TASK, 'frozen_epoch': time.time(), 'frozen_local': now(),
           'implementation_label': 'PAPER_REIMPLEMENTATION (author code located at LTSV commit 82d3f94, not executed; differences in METHOD.md)',
           'method_file': str(ROOT / 'METHOD.md'), 'source_check': str(ROOT / 'source_check.json'),
           'user_decisions_2026_09_18': {'downloads': 'model, pinned Time-MoE source and an isolated venv with transformers 4.40.1 allowed',
                                         'update_contract': 'task-book contract (48-step MSE, fixed T scaler, no aux loss, beta2 0.999) with the decoupled AdamW '
                                                            'weight decay of the author code path'},
           'exposure': exposure, 'contamination': 'UNKNOWN (Time-300B membership of KDD Cup 2018 / Beijing PM2.5 not verified)',
           'environment': env, 'geometry': GEOMETRY, 'sealing': {'prsa_closed_from_row': PRSA_CLOSED_FROM_ROW, 'max_prsa_row_to_read': GEOMETRY['RD02_T1']['E'][1],
                                                                  'max_kdd_row_to_read': GEOMETRY['RD01B_Q1']['E'][1], 'natural_final_read': False},
           'materials': {m: {'policy': pol, 'canonical_steps': rd.compile_policy(pol, [{}])['assignment'][0]} for m, pol in MATERIALS},
           'update': {'optimizer': 'torch.optim.AdamW', 'lr': 1e-5, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.1, 'no_decay': 'parameter names containing bias',
                      'clip_grad_norm': 1.0, 'precision': 'fp32, TF32 off, deterministic algorithms', 'loss': '48-step autoregressive MSE on T-scaler-normalized raw y',
                      'prediction': 'no-cache unroll with the official head choice (32, 8, 8), shared by updates and scoring', 'mode': 'eval() with autograd'},
           'blocks': {'rule': 'np.array_split of each entity time-ordered legal parents into 4; representative index (size-1)//2; weight = stratum size / legal parents',
                      'planned_logical_updates': sum(3 * len(prep[j]['block_parents']) for j in JOBS), 'planned_physical_updates_after_identical_x_reuse': planned_block},
           'value': {'v': 'L(theta0; P0 C_A inputs) - L(one_step(theta0, X[p,k], y[k]); P0 C_A inputs); L_A = score_block over C_A, L_o = score_block per origin',
                     'd': 'v[p] - v[P0]', 'U': 'sum_k weight_k d[p,k]', 'R_LTSV': 'argmax U over {P0: 0, P1, P2}; ties <= 1e-12 -> P0, P1, P2'},
           'confirm': {'seeds': list(CONFIRM_SEEDS), 'updates': CONFIRM_UPDATES, 'batch': CONFIRM_BATCH, 'index_stream': 'np.random.RandomState(seed).randint(0, legal parents, (100, 4)), shared by materials',
                       'loss': 'batch mean of the same 48-step MSE', 'R_TSF_CA': 'argmin 3-seed mean C_A L_A; ties -> P0, P1, P2', 'R_P0': 'P0_Linear',
                       'zero_shot': 'theta0 predictions kept for context only'},
           'mlp_transfer': mlp, 'prep': prep, 'caps': CAPS, 'wall_numeric_s': WALL_NUMERIC_S, 'llm_calls': 0, 'api_calls': 0, 'new_hashes': 0, 'new_mlp_fits': 0}
    context.write_json(p, cfg)
    return cfg


def checks_stage(ledger: Ledger) -> dict:
    for job in JOBS:
        if not (jd(job) / 'ca_inputs.npz').exists():
            rc, _ = ledger.run('prep_ca|' + job, base_cmd('--worker-prep-ca', job), jd(job) / 'logs' / 'prep_ca.log')
            if rc != 0:
                raise RuntimeError('C_A input preparation failed for %s' % job)
    p = ROOT / 'checks' / 'tsfm_checks.json'
    if not p.exists():
        ledger.start_numeric()
        rc, secs = ledger.run('checks|' + CHECK_JOB, tsfm_cmd('--checks', CHECK_JOB, '--max-updates', min(CAPS['extra'] - ledger.used('extra'), CAPS['extra'])),
                              ROOT / 'checks' / 'tsfm_checks.log')
        if rc != 0 or not p.exists():
            raise RuntimeError('TSFM checks failed')
    ch = context.read_json(p)
    t = ch['timing']
    plan = {j: context.read_json(jd(j) / 'blocks.json')['planned_physical_block_updates'] for j in JOBS}
    est = {'block_updates': sum(plan.values()) * t['block_update_with_reference_seconds'],
           'confirm_updates': CAPS['confirm'] * t['confirm_update_seconds'],
           'confirm_reference_and_save': 18 * (t['reference_predict_seconds'] + t['save_model_seconds']),
           'later_predictions': 19 * 3 * t['reference_predict_seconds'] + 18 * t['load_model_state_seconds'],
           'worker_cold_starts': 8 * t['cold_start_seconds'], 'scoring_workers': 12 * 15.0}
    est['total'] = sum(est.values())
    est['with_margin_x1_5'] = 1.5 * est['total']
    est['limit_s'] = WALL_NUMERIC_S
    est['status'] = 'OK' if est['with_margin_x1_5'] <= WALL_NUMERIC_S and all(ch['checks'].values()) else ('CHECK_FAILED' if not all(ch['checks'].values()) else 'RESOURCE_LIMITED')
    context.write_json(ROOT / 'checks' / 'resource_estimate.json', est)
    return {'checks': ch['checks'], 'estimate': est}


def block_stage(ledger: Ledger) -> dict:
    for job in JOBS:
        meta = context.read_json(jd(job) / 'blocks.json')
        for attempt in (0, 1):
            if (jd(job) / 'blocks' / 'done.json').exists():
                break
            allow = ledger.allowance('block', retry=attempt > 0)
            rc, _ = ledger.run('blocks|%s|%d' % (job, attempt), tsfm_cmd('--blocks', job, '--max-updates', allow, '--deadline', ledger.deadline()),
                               jd(job) / 'logs' / ('blocks_%d.log' % attempt))
            if rc == 3:
                raise BudgetStop('block update budget exhausted in %s' % job)
            if rc == 4:
                raise BudgetStop('numeric wall clock limit reached in blocks of %s' % job)
            if rc != 0 and attempt == 0:
                ledger.d['retries'] += 1
                ledger.save()
        if not (jd(job) / 'blocks' / 'done.json').exists():
            raise RuntimeError('block valuation failed for %s' % job)
        if not (jd(job) / 'block_scores.json').exists():
            rc, _ = ledger.run('score_blocks|' + job, base_cmd('--worker-score-blocks', job), jd(job) / 'logs' / 'score_blocks.log')
            if rc != 0:
                raise RuntimeError('block scoring failed for %s' % job)
    p = ROOT / 'selection_r_ltsv.json'
    if not p.exists():
        if any(jd(j).joinpath('confirm').exists() for j in JOBS) or later_outputs(ROOT):
            raise RuntimeError('confirmation or later outputs exist before R_LTSV')
        rec = {'jobs': {}, 'frozen_local': now(), 'frozen_epoch': time.time(), 'rule': 'argmax U, ties -> P0, P1, P2'}
        for j in JOBS:
            bs = context.read_json(jd(j) / 'block_scores.json')
            rec['jobs'][j] = {'R_LTSV': bs['R_LTSV'], 'U': bs['U'], 'U_origin': bs['U_origin'], 'R_LTSV_by_origin': bs['R_LTSV_by_origin']}
        context.write_json(p, rec)
    return context.read_json(p)


def confirm_stage(ledger: Ledger) -> dict:
    if not (ROOT / 'selection_r_ltsv.json').exists():
        raise PermissionError('R_LTSV must be frozen before the confirmation')
    for job in JOBS:
        for attempt in (0, 1):
            if (jd(job) / 'confirm' / 'done.json').exists():
                break
            allow = ledger.allowance('confirm', retry=attempt > 0)
            rc, _ = ledger.run('confirm|%s|%d' % (job, attempt), tsfm_cmd('--confirm', job, '--max-updates', allow, '--deadline', ledger.deadline()),
                               jd(job) / 'logs' / ('confirm_%d.log' % attempt))
            if rc == 3:
                raise BudgetStop('confirmation update budget exhausted in %s' % job)
            if rc == 4:
                raise BudgetStop('numeric wall clock limit reached in the confirmation of %s' % job)
            if rc != 0 and attempt == 0:
                ledger.d['retries'] += 1
                ledger.save()
        if not (jd(job) / 'confirm' / 'done.json').exists():
            raise RuntimeError('confirmation failed for %s' % job)
        if not (jd(job) / 'confirm_ca_scores.json').exists():
            rc, _ = ledger.run('score_confirm|' + job, base_cmd('--worker-score-confirm', job), jd(job) / 'logs' / 'score_confirm.log')
            if rc != 0:
                raise RuntimeError('confirmation C_A scoring failed for %s' % job)
    p = ROOT / 'selections.json'
    if not p.exists():
        if later_outputs(ROOT):
            raise RuntimeError('later outputs exist before the selections')
        lt = context.read_json(ROOT / 'selection_r_ltsv.json')
        rec = {'jobs': {}, 'r_ltsv_frozen_epoch': lt['frozen_epoch'], 'rules': {'R_LTSV': lt['rule'], 'R_TSF_CA': 'argmin 3-seed mean C_A of the 100-update TSFM, ties -> P0, P1, P2',
                                                                             'R_P0': 'always P0_Linear'}}
        for j in JOBS:
            cs = context.read_json(jd(j) / 'confirm_ca_scores.json')
            means = {m: (statistics.fmean(cs['models']['%s__s%d' % (m, s)]['A'] for s in CONFIRM_SEEDS)
                         if all(cs['models']['%s__s%d' % (m, s)]['A'] is not None for s in CONFIRM_SEEDS) else None) for m in MIDS}
            rec['jobs'][j] = {'R_LTSV': lt['jobs'][j]['R_LTSV'], 'R_TSF_CA': pick(means, maximize=False), 'R_P0': 'P0_Linear', 'tsfm_c_a_seed_means': means}
        rec['frozen_local'], rec['frozen_epoch'] = now(), time.time()
        context.write_json(p, rec)
    return context.read_json(p)


def later_stage(ledger: Ledger) -> dict:
    for job in JOBS:
        if not (jd(job) / 'later_inputs.npz').exists():
            rc, _ = ledger.run('prep_later|' + job, base_cmd('--worker-prep-later', job), jd(job) / 'logs' / 'prep_later.log')
            if rc != 0:
                raise RuntimeError('later input preparation failed for %s' % job)
        if not (jd(job) / 'later_frozen.json').exists():
            rc, _ = ledger.run('later_predict|' + job, tsfm_cmd('--later', job, '--deadline', ledger.deadline()), jd(job) / 'logs' / 'later_predict.log')
            if rc != 0:
                raise RuntimeError('later prediction failed for %s' % job)
    b = ROOT / 'later_barrier.json'
    if not b.exists():
        context.write_json(b, {'jobs': {j: context.read_json(jd(j) / 'later_frozen.json')['frozen_local'] for j in JOBS}, 'written_local': now(), 'written_epoch': time.time()})
    for job in JOBS:
        if not (jd(job) / 'later_scores.json').exists():
            rc, _ = ledger.run('score_later|' + job, base_cmd('--worker-score-later', job), jd(job) / 'logs' / 'score_later.log')
            if rc != 0:
                raise RuntimeError('later scoring failed for %s' % job)
    return {'done': True}


# ----------------------------------------------------------------------------------------------------------- readout
def _stats(v: list) -> dict:
    n = len(v)
    if n == 0:
        return {'n': 0}
    mean = statistics.fmean(v)
    sd = statistics.stdev(v) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    out = {'n': n, 'mean': mean, 'sd': sd, 'se': se, 'positive': sum(x > 0 for x in v), 'negative': sum(x < 0 for x in v), 'zero': sum(x == 0 for x in v), 'values': list(v)}
    if se is not None:
        from scipy.stats import t as _t
        q = float(_t.ppf(0.975, n - 1))
        out['t95_interval'] = [mean - q * se, mean + q * se]
        out['resolution'] = ('ZERO' if se == 0 and mean == 0 else 'RESOLVED_POSITIVE' if out['t95_interval'][0] > 0 else
                             'RESOLVED_NEGATIVE' if out['t95_interval'][1] < 0 else 'UNRESOLVED')
    return out


def _sign(x):
    return None if x is None else (1 if x > 0 else -1 if x < 0 else 0)


def consumer_losses(job: str) -> dict:
    """{consumer: {block: {material: {seed: L_A}}}} plus theta0 zero-shot for the TSFM."""
    out = {'tsfm': {b: {m: {} for m in MIDS} for b in BLOCKS}, 'mlp': {b: {m: {} for m in MIDS} for b in BLOCKS}, 'tsfm_zero_shot': {}}
    cs = context.read_json(jd(job) / 'confirm_ca_scores.json')
    ls = context.read_json(jd(job) / 'later_scores.json') if (jd(job) / 'later_scores.json').exists() else None
    for m in MIDS:
        for s in CONFIRM_SEEDS:
            name = '%s__s%d' % (m, s)
            out['tsfm']['c_a'][m][s] = cs['models'][name]['A']
            if ls:
                out['tsfm']['c_b'][m][s] = ls['models'][name]['c_b']['A']
                out['tsfm']['e'][m][s] = ls['models'][name]['e']['A']
    out['tsfm_zero_shot'] = {'c_a': cs['theta0']['A'], 'c_b': ls['models']['theta0']['c_b']['A'] if ls else None, 'e': ls['models']['theta0']['e']['A'] if ls else None}
    sa_ca = context.read_json(SA / 'jobs' / job / 'ca_scores.json')['ref']['2000']
    sa_l = context.read_json(SA / 'jobs' / job / 'later_scores.json')['scores']
    for m in MIDS:
        for s in MLP_SEEDS:
            out['mlp']['c_a'][m][s] = sa_ca[m][str(s)]['macro']
            cid = rd.length_cell_id(job, m, s, 2000)
            out['mlp']['c_b'][m][s] = sa_l[cid]['c_b']['macro']
            out['mlp']['e'][m][s] = sa_l[cid]['e']['macro']
    return out


def readout() -> dict:
    cfg = context.read_json(ROOT / 'frozen_config.json')
    res = {'package': cfg['package'], 'written_local': now(), 'implementation_label': cfg['implementation_label'],
           'run_status': context.read_json(ROOT / 'run_status.json') if (ROOT / 'run_status.json').exists() else None}
    sel = context.read_json(ROOT / 'selections.json') if (ROOT / 'selections.json').exists() else None
    res['selections'] = sel
    table2, table3, origin = {}, {}, {}
    for job in JOBS:
        if not (jd(job) / 'block_scores.json').exists():
            continue
        bs = context.read_json(jd(job) / 'block_scores.json')
        meta = context.read_json(jd(job) / 'blocks.json')
        row = {'U': bs['U'], 'U_origin': bs['U_origin'], 'R_LTSV': bs['R_LTSV'], 'R_LTSV_by_origin': bs['R_LTSV_by_origin'], 'theta0_c_a': bs['theta0'],
               'v_P0_weighted': bs['v_P0_weighted'], 'pairs': {}}
        CL = consumer_losses(job) if sel and (jd(job) / 'confirm_ca_scores.json').exists() else None
        for m in REPAIRS:
            pr = {'U': bs['U'][m], 'U_origin': bs['U_origin'][m], 'blocks_changed': '%d/%d' % (meta['blocks_changed_vs_p0'][m], len(meta['blocks'])),
                  'block_weight_changed': meta['block_weight_changed_vs_p0'][m],
                  'material_windows_changed': '%d/%d' % (meta['material_windows_changed_vs_p0'][m], meta['legal_parents']),
                  'block_d_values': [r['d'][m] for r in bs['blocks']]}
            if CL:
                for cons, seeds in (('tsfm', CONFIRM_SEEDS), ('mlp', MLP_SEEDS)):
                    for b in BLOCKS:
                        dd = [CL[cons][b]['P0_Linear'][s] - CL[cons][b][m][s] for s in seeds if CL[cons][b]['P0_Linear'].get(s) is not None and CL[cons][b][m].get(s) is not None]
                        pr.setdefault(cons, {})[b] = _stats(dd)
                pr['sign_U_vs_tsfm_c_a_mean'] = [_sign(pr['U']), _sign(pr['tsfm']['c_a'].get('mean'))]
                pr['mlp_transfer_c_a'] = 'TRANSFER_UNRESOLVED' if pr['mlp']['c_a'].get('resolution') == 'UNRESOLVED' else pr['mlp']['c_a'].get('resolution')
            row['pairs'][m] = pr
        table2[job] = row
        if CL and sel:
            chosen = sel['jobs'][job]
            t3 = {}
            for cons, seeds in (('tsfm', CONFIRM_SEEDS), ('mlp', MLP_SEEDS)):
                means = {m: {b: (statistics.fmean(CL[cons][b][m][s] for s in seeds) if all(CL[cons][b][m].get(s) is not None for s in seeds) else None)
                             for b in BLOCKS} for m in MIDS}
                cr = {'material_means': means, 'rules': {}, 'paired_vs_R_LTSV': {}, 'regret': {}}
                for r in RULES:
                    mm = chosen[r]
                    cr['rules'][r] = {'material': mm, **({b: means[mm][b] for b in BLOCKS} if mm else {})}
                    if mm:
                        cr['regret'][r] = {b: (None if means[mm][b] is None or any(means[x][b] is None for x in MIDS) else means[mm][b] - min(means[x][b] for x in MIDS))
                                           for b in BLOCKS}
                for other in ('R_P0', 'R_TSF_CA'):
                    mo, ml = chosen[other], chosen['R_LTSV']
                    key = '%s-R_LTSV' % other
                    if not (mo and ml):
                        cr['paired_vs_R_LTSV'][key] = {'status': 'NOT_SELECTABLE'}
                        continue
                    cr['paired_vs_R_LTSV'][key] = {'materials': [mo, ml], 'same_material': mo == ml}
                    for b in BLOCKS:
                        dd = [CL[cons][b][mo][s] - CL[cons][b][ml][s] for s in seeds if CL[cons][b][mo].get(s) is not None and CL[cons][b][ml].get(s) is not None]
                        st = _stats(dd)
                        den = means['P0_Linear'][b]
                        st['relative_to_p0_mean'] = st['mean'] / den if st.get('n') and den else None
                        cr['paired_vs_R_LTSV'][key][b] = st
                t3[cons] = cr
            t3['tsfm_zero_shot'] = CL['tsfm_zero_shot']
            table3[job] = t3
        origin[job] = {'U_origin': bs['U_origin'], 'R_LTSV_by_origin': bs['R_LTSV_by_origin'], 'origin_signs': bs['origin_sign_conflicts'],
                       'coverage_summary': {m: [{k: v for k, v in s.items() if k != 'segments'} for s in bs['coverage_summary'][m]] for m in REPAIRS}}
    res['table2_values_vs_real_adaptation'] = table2
    res['table3_selection_value'] = table3
    res['origin_and_coverage'] = origin
    pooled = {}
    for cons in ('tsfm', 'mlp'):
        for key in ('R_P0-R_LTSV', 'R_TSF_CA-R_LTSV'):
            for b in BLOCKS:
                vals = {DOMAINS[j]: table3[j][cons]['paired_vs_R_LTSV'].get(key, {}).get(b, {}).get('relative_to_p0_mean') for j in table3}
                pooled.setdefault(cons, {}).setdefault(key, {})[b] = {'by_domain': vals, 'domains_equal': statistics.fmean(vals.values()) if vals and all(v is not None for v in vals.values()) and len(vals) == 2 else None}
    res['pooled_relative'] = pooled
    res['checks'] = build_checks(sel)
    context.write_json(ROOT / 'checks.json', res['checks'])
    res['cost'] = cost_readout()
    res['budget'] = {**(context.read_json(ROOT / 'budget.json') if (ROOT / 'budget.json').exists() else {}), 'updates': Ledger().totals()}
    context.write_json(ROOT / 'result.json', res)
    write_tables(json.loads(json.dumps(res, default=str)))
    return res


def build_checks(sel) -> dict:
    out = {'smoke': context.read_json(ROOT / 'smoke_record.json') if (ROOT / 'smoke_record.json').exists() else None,
           'tsfm': context.read_json(ROOT / 'checks' / 'tsfm_checks.json') if (ROOT / 'checks' / 'tsfm_checks.json').exists() else None,
           'resource_estimate': context.read_json(ROOT / 'checks' / 'resource_estimate.json') if (ROOT / 'checks' / 'resource_estimate.json').exists() else None}
    if not sel:
        return out
    before = context.read_json(ROOT / 'checks' / 'cache_snapshot_before.json')['files']
    after = snapshot([Path(p) for p in before])
    lt = context.read_json(ROOT / 'selection_r_ltsv.json')
    blk_done = [context.read_json(jd(j) / 'blocks' / 'done.json') for j in JOBS]
    cf_done = [context.read_json(jd(j) / 'confirm' / 'done.json') for j in JOBS]
    li = [context.read_json(jd(j) / 'later_inputs.json') for j in JOBS if (jd(j) / 'later_inputs.json').exists()]
    lf = [context.read_json(jd(j) / 'later_frozen.json') for j in JOBS if (jd(j) / 'later_frozen.json').exists()]
    ls = [context.read_json(jd(j) / 'later_scores.json') for j in JOBS if (jd(j) / 'later_scores.json').exists()]
    bs = [context.read_json(jd(j) / 'block_scores.json') for j in JOBS]
    barrier = context.read_json(ROOT / 'later_barrier.json') if (ROOT / 'later_barrier.json').exists() else None
    u = Ledger().totals()
    protocol = {
        'cache_files_unchanged': after == before,
        'r_ltsv_after_all_block_scores': lt['frozen_epoch'] > max(b['scored_epoch'] for b in bs),
        'confirmation_started_after_r_ltsv': all(d['started_epoch'] > lt['frozen_epoch'] for d in cf_done),
        'selections_after_confirmation': sel['frozen_epoch'] > max(d['finished_epoch'] for d in cf_done),
        'later_inputs_after_selections': bool(li) and all(r['written_epoch'] > sel['frozen_epoch'] for r in li),
        'later_predictions_after_selections': bool(lf) and all(r['started_epoch'] > sel['frozen_epoch'] for r in lf),
        'barrier_after_all_predictions': barrier is not None and bool(lf) and barrier['written_epoch'] >= max(r['frozen_epoch'] for r in lf),
        'scores_after_barrier': bool(ls) and barrier is not None and all(r['scored_epoch'] > barrier['written_epoch'] for r in ls),
        'identical_x_gives_exact_zero_d': all(b['exact_zero_d_for_identical_x'] for b in bs),
        'theta0_restore_exact_after_blocks': all(d['theta0_restore_exact'] for d in blk_done),
        'updates_within_caps': u['block'] <= CAPS['block'] + CAPS['retry'] and u['confirm'] <= CAPS['confirm'] + CAPS['retry'] and u['extra'] <= CAPS['extra'] and u['total'] <= CAPS['hard'],
        'no_kmp_in_tsfm_workers': all(d.get('kmp_duplicate_lib_ok') is None for d in blk_done + cf_done),
        'prsa_rows_below_closed_region': all(r['rows_read']['e_target'][1] <= PRSA_CLOSED_FROM_ROW for r in ls),
    }
    out['protocol'] = {'checks': protocol, 'all_true': all(protocol.values()), 'changed_cache_files': [p for p in before if before[p] != after.get(p)]}
    return out


def cost_readout() -> dict:
    led = context.read_json(ROOT / 'budget.json') if (ROOT / 'budget.json').exists() else {}
    workers = led.get('workers', [])
    by_kind = {}
    for w in workers:
        by_kind.setdefault(w['name'].split('|')[0], []).append(w['seconds'])
    out = {'downloads_and_install': context.read_json(ROOT / 'frozen_config.json')['environment']['tsfm']['downloads_measured'],
           'worker_seconds_by_kind': {k: {'n': len(v), 'sum': sum(v)} for k, v in by_kind.items()},
           'numeric_wall_seconds': (max(w['end_epoch'] for w in workers) - led['numeric_start_epoch']) if workers and led.get('numeric_start_epoch') else None,
           'updates': Ledger().totals(), 'jobs': {}}
    for job in JOBS:
        row = {}
        for stage in ('blocks', 'confirm'):
            p = jd(job) / stage / 'done.json'
            if p.exists():
                row[stage] = context.read_json(p).get('timing')
        p = jd(job) / 'later_frozen.json'
        if p.exists():
            row['later'] = context.read_json(p).get('timing')
        out['jobs'][job] = row
    out['per_job_valuation_vs_confirmation'] = {j: {'ltsv_valuation_worker_seconds': sum(w['seconds'] for w in workers if w['name'].startswith('blocks|%s|' % j)),
                                                    'confirmation_worker_seconds': sum(w['seconds'] for w in workers if w['name'].startswith('confirm|%s|' % j)),
                                                    'note': 'each includes its own cold model load; confirmation covers 9 x 100 updates + C_A predictions + 9 model saves'}
                                                for j in JOBS}
    return out


def write_tables(res: dict) -> str:
    f = lambda x, d=4: '—' if x is None else ('%.*f' % (d, x))
    pm = lambda s, d=4: '—' if not s or not s.get('n') else '%s ± %s' % (f(s.get('mean'), d), f(s.get('se'), d))
    iv = lambda s: '—' if not s or not s.get('t95_interval') else '[%s, %s]' % (f(s['t95_interval'][0]), f(s['t95_interval'][1]))
    sg = lambda s: '—' if not s or not s.get('n') else '%d/%d/%d' % (s['positive'], s['negative'], s['zero'])
    lines = ['# DEV-DATA-READINESS-LTSV-CORE-REPAIR-SIGNAL tables', '',
             'Generated from result.json by `--result`. Losses: missing-aware macro NMSE (lower is better), every model served the P0 Linear inputs. '
             'U and material differences are oriented so that positive = the repair is better than P0. Selection differences = loss(other rule) − loss(R_LTSV): '
             'positive = R_LTSV better. "±" = one SE over seeds (TSFM n=3, MLP n=4).', '',
             '## 1. Implementation comparison', '', 'See METHOD.md (paper / author code / actual / nature). Label: **%s**.' % res.get('implementation_label'), '']
    lines += ['## 2. Block values versus real adaptation (C_A)', '',
              '| Job | Repair | U (L_A) | U origins | blocks changed | window share changed | TSFM100 d mean ± SE | TSFM t95 | TSFM | MLP2000 d mean ± SE | MLP t95 | MLP |',
              '|---|---|---:|---|---|---|---:|---|---|---:|---|---|']
    for j, row in (res.get('table2_values_vs_real_adaptation') or {}).items():
        for m, pr in row['pairs'].items():
            t, ml = (pr.get('tsfm') or {}).get('c_a'), (pr.get('mlp') or {}).get('c_a')
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                j, m, f(pr['U'], 6), ' / '.join(f(x, 6) for x in pr['U_origin']), pr['blocks_changed'], pr['material_windows_changed'],
                pm(t), iv(t), (t or {}).get('resolution'), pm(ml), iv(ml), pr.get('mlp_transfer_c_a')))
    lines += ['', '### 2b. Same material differences on C_B / E (d = P0 − repair)', '', '| Job | Repair | Consumer | C_A | C_B | E |', '|---|---|---|---:|---:|---:|']
    for j, row in (res.get('table2_values_vs_real_adaptation') or {}).items():
        for m, pr in row['pairs'].items():
            for cons in ('tsfm', 'mlp'):
                if cons in pr:
                    lines.append('| %s | %s | %s | %s | %s | %s |' % (j, m, cons, *('%s %s %s' % (pm(pr[cons][b]), sg(pr[cons][b]), (pr[cons][b] or {}).get('resolution', '')) for b in BLOCKS)))
    lines += ['', '### 2c. theta0 and block-level readings', '', '| Job | theta0 C_A L_A | weighted v(P0) (continued-training effect) | R_LTSV | R_LTSV by origin |', '|---|---:|---:|---|---|']
    for j, row in (res.get('table2_values_vs_real_adaptation') or {}).items():
        lines.append('| %s | %s | %s | %s | %s |' % (j, f(row['theta0_c_a']['A']), f(row['v_P0_weighted'], 6), row['R_LTSV'], ' / '.join(str(x) for x in row['R_LTSV_by_origin'])))
    for cons, title in (('tsfm', '3a. Selection value — TSFM 100-update Consumer (3 seeds)'), ('mlp', '3b. Selection value — current MLP 2000 (4 seeds, transfer reading)')):
        lines += ['', '## ' + title, '', '| Job | Rule | Material | C_A | C_B | E | regret C_A | regret C_B | regret E |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
        for j, t3 in (res.get('table3_selection_value') or {}).items():
            cr = t3[cons]
            for r, rm in cr['rules'].items():
                rg = cr['regret'].get(r, {})
                lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (j, r, rm.get('material'), f(rm.get('c_a')), f(rm.get('c_b')), f(rm.get('e')), f(rg.get('c_a')), f(rg.get('c_b')), f(rg.get('e'))))
        lines += ['', '| Job | Comparison | Materials | Block | n | mean ± SE | +/−/0 | t95 | relative to P0 |', '|---|---|---|---|---:|---:|---|---|---:|']
        for j, t3 in (res.get('table3_selection_value') or {}).items():
            for key, v in t3[cons]['paired_vs_R_LTSV'].items():
                if 'status' in v:
                    lines.append('| %s | %s | — | — | — | %s | — | — | — |' % (j, key, v['status']))
                    continue
                for b in ('e', 'c_b', 'c_a'):
                    s = v[b]
                    lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (j, key, ' vs '.join(v['materials']), b, s.get('n'), pm(s), sg(s), iv(s), f(s.get('relative_to_p0_mean'))))
        pr = (res.get('pooled_relative') or {}).get(cons, {})
        lines += ['', '| Comparison | Block | RD01B | RD02 | domains equal |', '|---|---|---:|---:|---:|']
        for key, bb in pr.items():
            for b in ('e', 'c_b', 'c_a'):
                p = bb[b]
                lines.append('| %s | %s | %s | %s | %s |' % (key, b, f(p['by_domain'].get('RD01B')), f(p['by_domain'].get('RD02')), f(p['domains_equal'])))
    lines += ['', '### 3c. TSFM zero-shot theta0 (context only)', '', '| Job | C_A | C_B | E |', '|---|---:|---:|---:|']
    for j, t3 in (res.get('table3_selection_value') or {}).items():
        z = t3['tsfm_zero_shot']
        lines.append('| %s | %s | %s | %s |' % (j, f(z['c_a']), f(z['c_b']), f(z['e'])))
    lines += ['', '## 4. Cost and checks', '', '```json', json.dumps({'cost': res.get('cost'), 'budget': res.get('budget')}, indent=1, default=str)[:12000], '```', '',
              '```json', json.dumps(res.get('checks'), indent=1, default=str)[:12000], '```', '']
    text = '\n'.join(lines)
    (ROOT / 'tables.md').write_text(text, encoding='utf-8')
    return text


# ----------------------------------------------------------------------------------------------------------- smoke (numpy rules; the TSFM checks run in the checks stage)
def smoke() -> dict:
    checks = {}
    ent = np.array([0] * 10 + [1] * 6)
    k = np.array(list(range(10)) + list(range(6)))
    differs = {m: np.array([i % 2 == 0 for i in range(16)]) for m in REPAIRS}
    bt = block_table(ent, k, 2, 1000, differs)
    checks['strata_sizes_and_medians'] = ([b['stratum_size'] for b in bt] == [3, 3, 2, 2, 2, 2, 1, 1] and [b['parent'] for b in bt] == [1, 4, 6, 8, 10, 12, 14, 15]
                                          and abs(sum(b['weight'] for b in bt) - 1.0) < 1e-12 and bt[0]['x_rows'] == [1001, 1193] and bt[0]['y_rows'] == [1193, 1241])
    d = {b['block']: float(b['block'] + 1) for b in bt}
    hand = (3 * 1 + 3 * 2 + 2 * 3 + 2 * 4 + 2 * 5 + 2 * 6 + 1 * 7 + 1 * 8) / 16
    checks['u_weighted_sum_by_hand'] = abs(aggregate_u(bt, d) - hand) < 1e-12 and aggregate_u(bt, {**d, 3: None}) is None
    small = [{'block': 0, 'entity': 0, 'k': 0}, {'block': 1, 'entity': 0, 'k': 100}, {'block': 2, 'entity': 1, 'k': 0}]
    m, segs = coverage_map(small, {0: 1.0, 1: 3.0, 2: None}, 2, t_len=400)
    checks['coverage_map_by_hand'] = (m[0, 50] == 1.0 and m[0, 150] == 2.0 and m[0, 300] == 3.0 and np.isnan(m[0, 350]) and np.isnan(m[1]).all()
                                      and segs[0]['segments'] == [{'t_offset': [0, 340], 'mean': float(m[0, :340].mean())}] and segs[1]['covered_mean'] is None
                                      and nulls(m[:, 345:346]) == [[None], [None]])
    checks['tie_rules'] = (pick({'P0_Linear': 0.0, 'P1_Seasonal': 1e-13, 'P2_Linear_Hampel': -1.0}, True) == 'P0_Linear'
                           and pick({'P0_Linear': 0.0, 'P1_Seasonal': 0.2, 'P2_Linear_Hampel': 0.2}, True) == 'P1_Seasonal'
                           and pick({'P0_Linear': 1.0, 'P1_Seasonal': 0.5, 'P2_Linear_Hampel': 0.5}, False) == 'P1_Seasonal'
                           and pick({'P0_Linear': 0.0, 'P1_Seasonal': None, 'P2_Linear_Hampel': 0.1}, True) is None)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / 'jobs' / 'RDX_V1').mkdir(parents=True)
        try:
            prep_later('RDX_V1', td)
            checks['later_inputs_refused_before_selections'] = False
        except PermissionError:
            checks['later_inputs_refused_before_selections'] = True
        try:
            score_later('RDX_V1', td)
            checks['later_scores_refused_before_barrier'] = False
        except PermissionError:
            checks['later_scores_refused_before_barrier'] = True
        context.write_json(td / 'later_barrier.json', {'jobs': {}})
        try:
            score_blocks('RDX_V1', td)
            checks['block_scores_refused_after_later_outputs'] = False
        except RuntimeError:
            checks['block_scores_refused_after_later_outputs'] = True
    ctl = subprocess.run([sys.executable, '-B', '-c', 'import sys; import %s; print("torch" in sys.modules)' % MODULE], cwd=str(REPO), capture_output=True, text=True,
                         timeout=120, env=worker_env())
    checks['controller_module_does_not_import_torch'] = ctl.stdout.strip().splitlines()[-1:] == ['False']
    rec = {'checks': checks, 'all_pass': all(checks.values()), 'written_local': now(), 'note': 'numpy rules only; TSFM reset/order/zero/gradient checks run in checks/tsfm_checks.json'}
    context.write_json(ROOT / 'smoke_record.json', rec)
    return rec


# ----------------------------------------------------------------------------------------------------------- entry
def run() -> None:
    preflight()
    if not (ROOT / 'smoke_record.json').exists() or not context.read_json(ROOT / 'smoke_record.json')['all_pass']:
        raise RuntimeError('smoke has not passed')
    ledger = Ledger()
    status = {'started_local': now(), 'started_epoch': time.time()}
    try:
        ch = checks_stage(ledger)
        status['checks'] = ch
        if ch['estimate']['status'] != 'OK':
            status['exit'] = ch['estimate']['status']
        else:
            status['r_ltsv'] = block_stage(ledger)['jobs']
            status['selections'] = {j: {r: v[r] for r in RULES} for j, v in confirm_stage(ledger)['jobs'].items()}
            later_stage(ledger)
            status['exit'] = 'COMPLETE'
    except BudgetStop as ex:
        status['exit'] = 'PARTIAL'
        status['stop'] = str(ex)
    status['finished_local'] = now()
    status['finished_epoch'] = time.time()
    context.write_json(ROOT / 'run_status.json', status)
    readout()
    print(json.dumps({k: v for k, v in status.items() if k != 'checks'}, ensure_ascii=False, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    for flag in ('--smoke', '--preflight', '--run', '--result'):
        ap.add_argument(flag, action='store_true')
    for flag in ('--worker-prep-t', '--worker-prep-ca', '--worker-score-blocks', '--worker-score-confirm', '--worker-prep-later', '--worker-score-later'):
        ap.add_argument(flag)
    a = ap.parse_args()
    workers = {'worker_prep_t': prep_t, 'worker_prep_ca': prep_ca, 'worker_score_blocks': score_blocks, 'worker_score_confirm': score_confirm,
               'worker_prep_later': prep_later, 'worker_score_later': score_later}
    for name, fn in workers.items():
        if getattr(a, name):
            fn(getattr(a, name))
            return
    if a.smoke:
        print(json.dumps(smoke(), indent=1))
    elif a.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, default=str)[:3000])
    elif a.run:
        run()
    elif a.result:
        print(json.dumps(readout(), ensure_ascii=False, default=str)[:3000])


if __name__ == '__main__':
    main()
