"""DEV-TEMPO-AUG-SOURCE-ALIGNMENT (docs/DEV_TEMPO_AUG_SOURCE_ALIGNMENT_TASK_2026-09-18.md): TempoPFN-sourced augmentation primitives
wired into the batch Harness, plus a <=28-fit integration run. 0 experiment LLM calls.

  primitives   methods/ttha/batch_base/tempo_source.py (verbatim Apache-2.0 copy) + tempo_aug.py (streams, DSL, materials)
  adapter      TempoAugAdapter: the unchanged seven-tool BatchAdapter protocol (methods/ttha/batch_research.run_job); opt-in only
  run          4 exposed dev jobs x {P_AmpResample, P_NoMixRecipe} x 3 seeds = 24 fits + 4 checks; None / FixedMixup reused from
               DEV-DATA-AUG-OPERATOR-PROBE after binding checks. Shared MLP, 2000 updates, batch 64, parent 0.5 + child 0.5.
  order        preflight -> build/bind -> materials frozen -> None checks -> fits (C_A) -> rerun checks -> report objects frozen
               -> C_B -> E predictions frozen -> barrier -> E -> readout. The controller never imports torch.
  --smoke | --run | --result | --wiring      (workers: --worker-build JOB | --worker-fit JOB MID SEED [--tag T] |
                                              --worker-label STAGE JOB | --worker-compare A B | --worker-parity | --worker-wiring)
"""
from __future__ import annotations

import os

KMP_AT_START = os.environ.get('KMP_DUPLICATE_LIB_OK')

import argparse
import ast
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from methods.ttha import batch_research as br
from methods.ttha.batch_base import context, spec, tempo_aug as ta
from methods.ttha.batch_base import readiness as rd
from evaluation.main_protocol_p4 import batch_research_data_aug_operator_probe as P

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_tempo_aug_source_alignment'
RUN = ROOT / 'jobs'
WIRING_RUN = ROOT / 'wiring_run'
TEMPOPFN = REPO.parent / 'TempoPFN'
TEMPOPFN_COMMIT = '5969ec6ddbded5a8d2976d42504c1ae436a2f1b3'
MODULE = 'evaluation.main_protocol_p4.batch_research_tempo_aug_source_alignment'
JOBS = ('RD01B_Q1', 'RD01B_STP1', 'RD02_T1', 'RD02_STP1')          # task §6 table order = job index 1..4
JOB_INDEX = {j: i + 1 for i, j in enumerate(JOBS)}
DOMAIN = {'RD01B_Q1': 'RD01B', 'RD01B_STP1': 'RD01B', 'RD02_T1': 'RD02', 'RD02_STP1': 'RD02'}
SEEDS = P.SEEDS
N = 2000
MATERIALS = {'P_AmpResample': {'index': 1, 'steps': [{'op': 'tp_amplitude'}, {'op': 'tp_resample'}]},
             'P_NoMixRecipe': {'index': 2, 'steps': [{'op': ta.RECIPE}]}}
NEW = tuple(MATERIALS)
REPORTED = ('None', 'FixedMixup') + NEW
CHECK_NONE = {'RD01B': ('RD01B_Q1', 20261001), 'RD02': ('RD02_T1', 20261001)}
CHECK_RERUN = {'RD01B': ('RD01B_Q1', 'P_NoMixRecipe', 20261001), 'RD02': ('RD02_T1', 'P_AmpResample', 20261001)}
CAPS = {'fit': 24, 'check': 4, 'hard': 28}                       # every attempt counts; no retry allowance
WALL_NUMERIC_S = 3600.
FIT_TIMEOUT_S = 300.
T975 = {2: 4.302652729911275}
PARITY_TOL = {'float32': 0.0}                                   # frozen before the first parity run: bitwise for float32 outputs


def ds_of(job: str) -> str:
    return job.split('_')[0]


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def cid(job: str, mid: str, seed: int, tag: str = '') -> str:
    return P.cid(job, mid, seed, N, tag)


def plan_policy(mid: str) -> dict:
    return {'default': {'steps': MATERIALS[mid]['steps']}, 'rules': [], 'rationale': 'frozen package material %s' % mid,
            'observation_fields_used': []}


# ============================================================================= workers (subprocess entries; torch allowed)
def worker_build(job: str) -> None:
    """P0_Linear, binding against the probe's parents / scaler / serving inputs, then the two frozen materials."""
    rd._init_openmp_before_torch()
    ctx = rd.open_job(ds_of(job), job, RUN, stage='material')
    Z, ent, k = ta.joint_parents(ctx)
    pctx = rd.open_job(ds_of(job), job, P.RUN, stage='material')
    pX, py = rd.training_arrays(pctx, rd.MaterialRef(**rd.load_index(pctx.job_dir)['P0_Linear']))
    with np.load(ctx.job_dir / 'scaler.npz') as a, np.load(P.RUN / job / 'scaler.npz') as b:
        scaler_same = {f: bool(np.array_equal(a[f], b[f])) for f in ('mean', 'std', 'scale', 'legal_ent', 'legal_k', 'legal_counts')}
    bind = {'job': job, 'checked_local': now(), 'scaler_and_legal_equal': scaler_same,
            'parent_X_equal': bool(np.array_equal(Z[:, :spec.L], pX)), 'parent_y_equal': bool(np.array_equal(Z[:, spec.L:], py)),
            'legal_parents': int(Z.shape[0]), 'train_rows': list(ctx.job.train_range), 'evaluate_rows': list(ctx.job.evaluate_rows),
            'probe_consumer': 'shared MLP %s, lr %s, wd %s, batch %d, 2000 updates, parent 0.5 + child 0.5' % (spec.CONSUMER['arch'], spec.LR, spec.WEIGHT_DECAY, spec.PARENT_BATCH)}
    bind['window_dates'] = {'first': str(ta.window_starts(ctx)[0]), 'n': int(Z.shape[0]), 'binding': 'hour of every window point checked'}
    bind['ok'] = all(scaler_same.values()) and bind['parent_X_equal'] and bind['parent_y_equal']
    context.write_json(ctx.job_dir / 'binding.json', bind)
    if not bind['ok']:
        raise RuntimeError('BINDING_FAILED %s' % job)
    reg = ta.load_registry(ctx.job_dir)
    table = ctx.overview()['entities']
    for mid, m in MATERIALS.items():
        if mid in reg:
            continue
        compiled = ta.compile_plan(plan_policy(mid), table)
        rec = ta.build_material(ctx, compiled['assignment'], mid, job_index=JOB_INDEX[job], material_index=m['index'], compiled=compiled)
        s = context.read_json(rec['summary_path'])
        print('MATERIAL', job, mid, 'seed', rec['seed'], 'build_s %.1f' % s['build_seconds'], 'rmsX %.3f rmsY %.3f' % (s['rms_change_X'], s['rms_change_y']), flush=True)


def worker_fit(job: str, mid: str, seed: int, tag: str = '') -> None:
    rd._init_openmp_before_torch()
    from methods.ttha.batch_base import train
    import torch
    t0 = time.time()
    ctx = rd.open_job(ds_of(job), job, RUN, stage='evaluate')          # rows [t-672, t+96): no C_B / E row in memory
    ref = rd.MaterialRef(**rd.load_index(ctx.job_dir)['P0_Linear'])
    X, y = rd.training_arrays(ctx, ref)
    lg, sc = ctx.legal, ctx.scaler
    child = None if mid == 'None' else ta.load_child(ctx.job_dir, mid)
    if mid != 'None' and child is None:
        raise RuntimeError('material %s has no child view' % mid)
    if child is not None and (child[0].shape != X.shape or child[1].shape != y.shape):
        raise RuntimeError('child does not cover the legal parent set')
    for d in ('runs', 'cells', 'pred_c_a'):
        (ctx.job_dir / d).mkdir(exist_ok=True)
    bidx = train.batch_indices(seed, n_pool=lg.n, n_updates=N)
    model, tres = train.train_arm(X, y, child, bidx, model_seed=seed, timeout_seconds=FIT_TIMEOUT_S - 30, n_updates=N)
    Xs, serve = rd.serve_inputs(ctx.slice, ctx.job.c_a, ref.assignment, sc)
    truth = rd.read_truth(ctx.slice, ctx.job.c_a)
    c = cid(job, mid, seed, tag)
    mp = ctx.job_dir / 'runs' / (c + '.pt')
    train.save_model(model, mp)
    pred = rd.predict_inputs(model, Xs, sc)
    np.savez(ctx.job_dir / 'pred_c_a' / (c + '.npz'), pred_raw=pred, origins=np.array(ctx.job.c_a))
    reg = ta.resolve(ctx.job_dir, mid) if mid != 'None' else {}
    rec = {'cell_id': c, 'status': 'OK', 'job_id': job, 'material_id': mid, 'material_seed': reg.get('seed'), 'model_seed': seed, 'n_updates': N,
           'tag': tag, 'physical_fit': True, 'child_view': child is not None, 'pool_size': int(lg.n),
           'batch_stream': 'train.batch_indices(seed, n_pool, 2000)', 'model_path': str(mp),
           'train': {'seconds': tres.seconds, 'final_train_loss': tres.final_train_loss, 'first_step_loss': tres.first_step_loss},
           'serve_inputs_c_a': serve, 'scores': {'c_a': rd.score_block(pred, truth, sc)},
           'device': str(train.device()), 'torch': torch.__version__, 'torch_threads': torch.get_num_threads(),
           'python': sys.version.split()[0], 'rows_read': list(ctx.job.evaluate_rows), 'worker_seconds_total': time.time() - t0}
    context.write_json(ctx.job_dir / 'cells' / (c + '.json'), rec)
    print('CELL_OK', c, round(tres.seconds, 2), flush=True)


def _cells(job_dir: Path) -> dict:
    out = {}
    for p in sorted((job_dir / 'cells').glob('*.json')) if (job_dir / 'cells').exists() else []:
        r = context.read_json(p)
        if r.get('status') == 'OK' and not r.get('tag') and r['material_id'] in NEW:
            out[r['cell_id']] = r
    return out


def label_prerequisite(stage: str, job: str) -> None:
    """Raises before any label row is loaded or torch is imported."""
    jd = RUN / job
    if stage == 'c_b' and not (ROOT / 'report_objects.json').exists():
        raise PermissionError('C_B refused: report_objects.json (frozen after all C_A fits) missing')
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
    ds, js = ds_of(job), rd.resolve_job(ds_of(job), job)
    cells = _cells(jd)
    if stage in ('c_b', 'freeze_e'):
        ctx = rd.open_job(ds, job, RUN, stage='c_b' if stage == 'c_b' else 'e_input')
        origins = js.c_b if stage == 'c_b' else js.e
        Xs, serve = rd.serve_inputs(ctx.slice, origins, [[] for _ in js.roster], ctx.scaler)
        if stage == 'c_b':
            truth = rd.read_truth(ctx.slice, origins)
            out = {'job_id': job, 'opened_local': now(), 'rows_read': list(js.c_b_rows), 'serve': serve, 'cells': {}}
            for c, r in cells.items():
                pred = rd.predict_inputs(train.load_model(r['model_path']), Xs, ctx.scaler)
                out['cells'][c] = {'material_id': r['material_id'], 'model_seed': r['model_seed'], 'c_b': rd.score_block(pred, truth, ctx.scaler)}
            # binding: the probe's None models re-scored here must reproduce the probe's frozen C_B
            pcb = context.read_json(P.RUN / job / 'c_b_scores.json')['cells']
            out['probe_none_rescore_equal'] = all(
                rd.score_block(rd.predict_inputs(train.load_model(_probe_cell(job, 'None', s)['model_path']), Xs, ctx.scaler),
                               truth, ctx.scaler)['normalized_mse_macro'] == pcb[P.cid(job, 'None', s, P.N_FULL)]['c_b']['normalized_mse_macro']
                for s in SEEDS[job])
            context.write_json(jd / 'c_b_scores.json', out)
        else:
            (jd / 'pred_e').mkdir(exist_ok=True)
            out = {'job_id': job, 'started_local': now(), 'rows_read': list(js.e_input_rows), 'serve': serve, 'models': {}}
            for c, r in cells.items():
                pred = rd.predict_inputs(train.load_model(r['model_path']), Xs, ctx.scaler)
                p = jd / 'pred_e' / (c + '.npz')
                np.savez(p, pred_raw=pred, origins=np.array(js.e))
                out['models'][c] = {'path': str(p), 'material_id': r['material_id'], 'model_seed': r['model_seed']}
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
            out['cells'][c] = {'material_id': m['material_id'], 'model_seed': m['model_seed'], 'e': rd.score_block(pred, truth, ctx.scaler)}
        pe = context.read_json(P.RUN / job / 'e_scores.json')['cells']
        eq = []
        for s in SEEDS[job]:
            with np.load(P.RUN / job / 'pred_e' / (P.cid(job, 'None', s, P.N_FULL) + '.npz')) as z:
                eq.append(rd.score_block(z['pred_raw'], truth, ctx.scaler)['normalized_mse_macro'] == pe[P.cid(job, 'None', s, P.N_FULL)]['e']['normalized_mse_macro'])
        out['probe_none_rescore_equal'] = all(eq)
        context.write_json(jd / 'e_scores.json', out)
    else:
        raise ValueError(stage)
    print('LABEL_OK', stage, job, flush=True)


def worker_compare(a: str, b: str) -> None:
    import torch
    sa, sb = torch.load(a, map_location='cpu'), torch.load(b, map_location='cpu')
    same = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    print(json.dumps({'identical': bool(same)}))


# ----------------------------------------------------------------------------- parity against the original (read-only clone)
def _original_namespace() -> dict:
    """Execute the ORIGINAL class sources, read at test time from the read-only clone, without importing their heavy modules
    (src.data.augmentations imports gluonts / datasets via src.gift_eval). Frequency helpers come from the original module."""
    sys.dont_write_bytecode = True
    if str(TEMPOPFN) not in sys.path:
        sys.path.append(str(TEMPOPFN))
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from src.data.frequency import Frequency, parse_frequency            # original module (numpy/pandas/constants only)
    ns = {'np': np, 'torch': torch, 'nn': nn, 'F': F, 'pd': pd, 'Frequency': Frequency, 'parse_frequency': parse_frequency,
          'Any': object, 'MixUpAugmenter': type('MixUpAugmenter', (), {}), 'TimeFlipAugmenter': None, 'YFlipAugmenter': None,
          'QuantizationAugmenter': None, 'DifferentialAugmenter': None, 'CensorAugmenter': None, 'RandomConvAugmenter': None}
    for rel, names in (('src/data/augmentations.py', ('CensorAugmenter', 'RandomConvAugmenter')),
                       ('src/synthetic_generation/augmentations/offline_per_sample_iid_augmentations.py', ('UnivariateOfflineAugmentor',))):
        text = (TEMPOPFN / rel).read_text(encoding='utf-8')
        for node in ast.parse(text).body:
            if isinstance(node, ast.ClassDef) and node.name in names:
                exec(compile(ast.get_source_segment(text, node), str(TEMPOPFN / rel), 'exec'), ns)
    return ns


def worker_parity() -> None:
    rd._init_openmp_before_torch()
    import pandas as pd
    import torch
    ns = _original_namespace()
    Orig = ns['UnivariateOfflineAugmentor']

    def original(name, series, rng, start):
        host = object.__new__(Orig)                   # original methods, without the original __init__'s global reseeding
        host.rng = rng
        if name == 'tp_regime':
            return host._apply_regime_change(series, p_apply=1.0)
        if name == 'tp_shock':
            return host._apply_shock_recovery(series, p_apply=1.0)
        if name == 'tp_calendar':
            return host._apply_calendar_injections(series, [start], ['h'], p_apply=1.0)
        if name == 'tp_amplitude':
            return host._apply_seasonality_amplitude_modulation(series, p_apply=1.0)
        if name == 'tp_resample':
            return host._apply_resample_artifacts(series, p_apply=1.0)
        if name == 'tp_censor':
            return ns['CensorAugmenter']().transform(series)
        if name == 'tp_random_conv':
            return ns['RandomConvAugmenter'](p_transform=1.0).transform(series)
        raise ValueError(name)

    # inputs: a non-constant synthetic series and real legal T joint windows of two jobs (with real dates)
    t = np.arange(ta.L_JOINT)
    synth = (np.sin(2 * np.pi * t / 24) + 0.01 * t + np.random.RandomState(7).normal(0, 0.3, t.size)).astype(np.float64)
    inputs = [('synthetic', synth, pd.Timestamp('2020-02-27 05:00:00'))]
    for job in ('RD01B_Q1', 'RD02_T1'):
        ctx = rd.open_job(ds_of(job), job, RUN, stage='material')
        Z, ent, k = ta.joint_parents(ctx)
        starts = ta.window_starts(ctx)
        for p in (0, int(Z.shape[0] // 2)):
            inputs.append(('%s_window_%d' % (job, p), Z[p], starts[p]))
    rows, all_ok = [], True
    seeds = (11, 12, 13, 14, 15)
    for label, z, start in inputs:
        for name in ta.PRIMITIVES:
            for sd in seeds:
                guard_t, guard_n = torch.get_rng_state(), np.random.get_state()
                st = ta.Streams(sd)
                w, _, _ = ta.run_program(z, [{'op': name}], st, start)
                restored = torch.equal(guard_t, torch.get_rng_state()) and all(
                    np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(guard_n, np.random.get_state()))
                torch.manual_seed(sd)
                np.random.seed(sd)
                o = original(name, torch.as_tensor(z, dtype=torch.float32).reshape(1, -1, 1).clone(), np.random.default_rng(sd), start)
                o = o.reshape(-1).numpy()
                diff = float(np.max(np.abs(o.astype(np.float64) - w.astype(np.float64))))
                ok = diff <= PARITY_TOL['float32'] and o.dtype == np.float32 and restored
                all_ok &= ok
                rows.append({'input': label, 'primitive': name, 'seed': sd, 'max_abs_diff': diff, 'dtype': str(o.dtype),
                             'caller_rng_restored': restored, 'changed': bool(np.any(o != z.astype(np.float32))), 'ok': ok})
    # chains and multi-window stream continuity: wrapper Streams vs the original objects consumed continuously
    chains = ([{'op': 'tp_shock'}, {'op': 'tp_resample'}, {'op': 'tp_censor'}], [{'op': 'tp_regime'}, {'op': 'tp_calendar'}, {'op': 'tp_random_conv'}],
              [{'op': 'tp_amplitude'}, {'op': 'tp_censor'}, {'op': 'tp_random_conv'}])
    chain_rows = []
    for ch in chains:
        for sd in seeds[:3]:
            st = ta.Streams(sd)
            ws = [ta.run_program(z, ch, st, start)[0] for _, z, start in inputs]
            torch.manual_seed(sd)
            np.random.seed(sd)
            g = np.random.default_rng(sd)
            os_ = []
            for _, z, start in inputs:
                s = torch.as_tensor(z, dtype=torch.float32).reshape(1, -1, 1).clone()
                for stp in ch:
                    s = original(stp['op'], s, g, start)
                os_.append(s.reshape(-1).numpy())
            diff = max(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) for a, b in zip(ws, os_))
            ok = diff <= PARITY_TOL['float32']
            all_ok &= ok
            chain_rows.append({'chain': [s['op'] for s in ch], 'seed': sd, 'windows_in_sequence': len(inputs), 'max_abs_diff': diff, 'ok': ok})
    # frequency subset (M2) against the original module
    from methods.ttha.batch_base import tempo_source as vs
    fq = {'original_parse_h': ns['parse_frequency']('h').value, 'vendored_parse_h': vs.parse_frequency('h').value,
          'original_alias': ns['parse_frequency']('h').to_pandas_freq(for_date_range=True), 'vendored_alias': vs.parse_frequency('h').to_pandas_freq(True)}
    fq['ok'] = fq['original_parse_h'] == fq['vendored_parse_h'] == 'h' and fq['original_alias'] == fq['vendored_alias'] == 'h'
    all_ok &= fq['ok']
    # verbatim text check of the vendored copy against the clone
    vend = (REPO / 'methods' / 'ttha' / 'batch_base' / 'tempo_source.py').read_text(encoding='utf-8')
    segs = {}
    for rel, spans in (('src/synthetic_generation/augmentations/offline_per_sample_iid_augmentations.py', ((505, 556), (558, 589), (591, 662), (664, 684), (686, 732))),
                       ('src/data/augmentations.py', ((319, 371), (1013, 1213)))):
        lines = (TEMPOPFN / rel).read_text(encoding='utf-8').splitlines()
        for a, b in spans:
            segs['%s:%d-%d' % (rel, a, b)] = '\n'.join(lines[a - 1:b]) in vend
    all_ok &= all(segs.values())
    head = subprocess.run(['git', '-C', str(TEMPOPFN), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(TEMPOPFN), 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()
    out = {'written_local': now(), 'tolerance': PARITY_TOL, 'reference_head': head, 'reference_expected': TEMPOPFN_COMMIT,
           'reference_clean': dirty == '', 'n_single': len(rows), 'n_single_ok': sum(r['ok'] for r in rows),
           'single_unchanged_outputs': {n: sum(not r['changed'] for r in rows if r['primitive'] == n) for n in ta.PRIMITIVES},
           'chains': chain_rows, 'frequency_subset': fq, 'verbatim_segments': segs, 'single': rows,
           'ok': bool(all_ok and head == TEMPOPFN_COMMIT and dirty == '')}
    context.write_json(ROOT / 'parity.json', out)
    print('PARITY', out['ok'], out['n_single_ok'], '/', out['n_single'], flush=True)


# ============================================================================= Harness adapter (seven tools; opt-in)
CONTRACTS = {
    'overview': {'arguments': {}, 'meaning': 'All N entity rows of T-only, missing-aware observations, batch summaries, the Consumer, and the augmentation action table (7 TempoPFN-sourced primitives, the P_NoMixRecipe preset, composition rules, window and randomness semantics).'},
    'inspect_data': {'arguments': {'entity_indices': '[1..8 integers in 0..N-1]', 'kind': 'gaps|segment|daily_means|hour_profile', 'sub_range': 'optional [absolute_start, absolute_end) inside overview.train_rows'},
                     'meaning': 'Raw current-T views with missing positions kept (null), normalized by the frozen T scaler.'},
    'build_material': {'arguments': {'plan_id': 'new short alphanumeric/underscore id',
                                     'policy': {'default': {'steps': [{'op': '<primitive name from overview.actions.primitives, or P_NoMixRecipe alone>'}]},
                                                'rules': [{'when': {'feature': '<overview field>', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': []}],
                                                'rationale': '<current hypothesis; no data source or job name>', 'observation_fields_used': []}},
                       'meaning': 'Construct one complete whole-batch augmentation plan and freeze its child view; does not train. Up to 3 primitives per program, no repeats, regime/shock and calendar/amplitude mutually exclusive, executed in the fixed order regardless of listing; P_NoMixRecipe is a complete program. Empty steps = no augmentation for that entity; an all-empty plan is None. First matching rule wins, a null field never fires a rule. No seed, strength or distribution argument exists; identical plans return the same material.'},
    'inspect_material': {'arguments': {'plan_id': '<built id>', 'entity_index': 'optional integer: also return one parent/child window', 'window': "optional 'largest_change' (default) | 'first' | 'last'"},
                         'meaning': 'What the plan changed: executed steps, identity (no-op) counts, change fraction and RMS for inputs and targets separately. Not predictive utility.'},
    'evaluate': {'arguments': {'plan_id': '<built id>'}, 'meaning': 'Train the complete plan (parent 0.5 + child 0.5, shared MLP, 2000 updates) under the frozen paired seeds and return C_A only (missing-aware loss on observed future cells, Baseline-Linear serving).'},
    'compare': {'arguments': {'a': '<evaluated id>', 'b': '<evaluated id>'}, 'meaning': 'Positive paired delta means b has lower C_A loss; per seed, origin and entity; seed uncertainty, not future certainty.'},
    'commit': {'arguments': {'plan_id': '<evaluated id>', 'reason': '<brief evidence-based decision>'}, 'meaning': 'Final action. Any evaluated plan including None or FixedMixup. No automatic argmin, no later selection override.'},
}


class WiringStop(RuntimeError):
    pass


class TempoAugAdapter:
    """BatchAdapter over the readiness T view + tempo_aug materials. `allow_fit=False` (wiring test): evaluate only replays
    already-fitted materials (their frozen cells) and rejects anything that would need a new fit, before any slot or fit."""

    def __init__(self, run_dir: Path, job: str, *, seeds, job_index: int, allow_fit: bool = False, fit_fn=None):
        self.ctx = rd.open_job(ds_of(job), job, run_dir, 'material')
        self.job, self.seeds, self.job_index = job, tuple(seeds), job_index
        self.allow_fit, self.fit_fn = allow_fit, fit_fn
        self.n = len(self.ctx.job.roster)
        self.specs = {}

    # --- read tools
    def overview(self):
        ov = br.json_copy(self.ctx.overview())
        ov['batch_features'] = {'batch_median_' + k: v['median'] for k, v in ov['summary'].items() if v is not None}
        ov['actions'] = ta.action_table()                      # replaces the readiness repair table: this profile augments
        ov['consumer'] = {**ov.get('consumer', {}), 'training': 'parent 0.5 + child 0.5 (existing augmentation protocol), 2000 updates, batch 64'}
        return ov

    def inspect_data(self, arguments, *, remaining_seconds):
        if set(arguments) - {'entity_indices', 'kind', 'sub_range'}:
            raise br.ToolInputError('unknown inspection argument')
        idx = arguments.get('entity_indices')
        if not isinstance(idx, list) or not 1 <= len(idx) <= rd.MAX_INSPECT_ENTITIES or any(type(i) != int or not 0 <= i < self.n for i in idx):
            raise br.ToolInputError('entity_indices must be 1..%d integers in 0..%d' % (rd.MAX_INSPECT_ENTITIES, self.n - 1))
        kind = arguments.get('kind', 'gaps')
        if kind not in rd.INSPECT_KINDS:
            raise br.ToolInputError('kind must be one of %s' % list(rd.INSPECT_KINDS))
        span = arguments.get('sub_range')
        lo, hi = self.ctx.job.train_range
        if span is not None and (not isinstance(span, list) or len(span) != 2 or any(type(x) != int for x in span) or not lo <= span[0] < span[1] <= hi):
            raise br.ToolInputError('sub_range must be two increasing absolute rows inside T=[%d,%d)' % (lo, hi))
        return self.ctx.inspect_data(idx, kind, span)

    def build_material(self, arguments, *, remaining_seconds):
        if set(arguments) != {'plan_id', 'policy'}:
            raise br.ToolInputError('build needs plan_id and policy')
        mid = arguments['plan_id']
        if not isinstance(mid, str) or not mid[:1].isalpha() or not mid.replace('_', '').isalnum() or len(mid) > 40 or mid in ('None', 'FixedMixup'):
            raise br.ToolInputError('unsafe or reserved plan id')
        try:
            compiled = ta.compile_plan(arguments['policy'], self.ctx.overview()['entities'])
        except rd.PolicyError as exc:
            raise br.ToolInputError(str(exc), legal_ops=list(ta.PRIMITIVES) + [ta.RECIPE], valid_observation_fields=list(rd.FIELDS)) from None
        if mid in ta.load_registry(self.ctx.job_dir):
            raise br.ToolInputError('plan id already exists; use it or a new id')
        rec = ta.build_material(self.ctx, compiled['assignment'], mid, job_index=self.job_index, compiled=compiled)
        programs, idx = [], []
        for steps in compiled['assignment']:
            if steps not in programs:
                programs.append(steps)
            idx.append(programs.index(steps))
        ms = {'profile': ta.PROFILE_VERSION, 'policy': compiled['policy'], 'programs': programs, 'entity_program': idx,
              'rule_index': compiled['rule_index'], 'resolved_thresholds': compiled['resolved_thresholds'], 'n_unknown': compiled['n_unknown'],
              'alias_of': rec.get('alias_of'), 'execution_order': 'fixed: ' + ' -> '.join(ta.PRIMITIVES)}
        self.specs[mid] = ms
        return br.Candidate(mid, ms, self.job)

    def inspect_material(self, plan_id, arguments, *, remaining_seconds):
        if set(arguments) - {'plan_id', 'entity_index', 'window'}:
            raise br.ToolInputError('inspect_material takes plan_id and optional entity_index / window')
        ei, w = arguments.get('entity_index'), arguments.get('window', 'largest_change')
        if ei is not None and (type(ei) != int or not 0 <= ei < self.n):
            raise br.ToolInputError('entity_index must be an integer in 0..%d' % (self.n - 1))
        if w not in ('largest_change', 'first', 'last'):
            raise br.ToolInputError("window must be 'largest_change', 'first' or 'last'")
        if plan_id in ('None', 'FixedMixup'):
            return {'material_id': plan_id, 'note': 'historical baseline of the previous package; no TempoPFN primitive; see its own records'}
        return ta.inspect(self.ctx, plan_id, ei, w)

    # --- training
    def _resolved(self, plan_id: str) -> str:
        if plan_id in ('None', 'FixedMixup'):
            return plan_id
        rec = ta.resolve(self.ctx.job_dir, plan_id)
        return 'None' if rec.get('alias_of') == 'None' else rec['material_id']

    def _cell_records(self, mid: str):
        if mid in ('None', 'FixedMixup'):
            paths = [P.cell_path(self.job, mid, s, P.N_FULL) for s in self.seeds]
        else:
            paths = [RUN / self.job / 'cells' / (cid(self.job, mid, s) + '.json') for s in self.seeds]
        return [context.read_json(p) for p in paths] if all(p.exists() for p in paths) else None

    def check_evaluate(self, plan_id, candidate):
        if not self.allow_fit and self._cell_records(self._resolved(plan_id)) is None:
            raise br.ToolInputError('wiring test: this plan has no frozen fitted cells; new fits are disabled in this run (no fit, no slot)',
                                    execution_status='FIT_DISABLED_IN_WIRING_TEST')

    def evaluate(self, candidate, seeds, *, feedback, remaining_seconds):
        mid = self._resolved(candidate.plan_id)
        recs = self._cell_records(mid)
        if recs is None:
            if not self.allow_fit or self.fit_fn is None:
                raise WiringStop('fit disabled')
            self.fit_fn(self.job, mid, tuple(seeds))
            recs = self._cell_records(mid)
        if any(r['scores']['c_a']['status'] != 'SCORABLE' for r in recs):
            raise RuntimeError('C_A_NOT_SCORABLE')
        losses = tuple(tuple(tuple(row) for row in r['scores']['c_a']['per_origin_entity_normalized_mse']) for r in recs) if feedback else ()
        return replace(candidate, model_seeds=tuple(seeds), model_refs=tuple(r['model_path'] for r in recs), ca_losses=losses)

    def baselines(self):
        out = []
        for mid, spec_ in (('None', {'kind': 'historical_baseline', 'augmentation': 'none (parent view only)'}),
                           ('FixedMixup', {'kind': 'historical_baseline', 'augmentation': 'timemixup donor rule R, w=0.25 (pre-existing augment.py; not TempoPFN)'})):
            c = br.Candidate(mid, spec_, self.job)
            out.append(self.evaluate(c, self.seeds, feedback=True, remaining_seconds=1e9))
        return tuple(out)


def scripted_wiring_client(log: list):
    """Scripted Fast (WIRING TEST, not an agent and not a research strategy): observe -> build a composition -> inspect ->
    build the preset (alias of the frozen material) -> evaluate (replay) -> try to evaluate the unfitted composition (rejected)
    -> compare -> commit. Records what the real request exposes."""
    script = [
        [{'tool': 'overview', 'arguments': {}}],
        [{'tool': 'inspect_data', 'arguments': {'entity_indices': [0, 1], 'kind': 'hour_profile'}}],
        [{'tool': 'build_material', 'arguments': {'plan_id': 'W_comp', 'policy': {
            'default': {'steps': [{'op': 'tp_censor'}, {'op': 'tp_shock'}]},
            'rules': [{'when': {'feature': 'missing_fraction', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': [{'op': 'tp_resample'}, {'op': 'tp_amplitude'}, {'op': 'tp_random_conv'}]}],
            'rationale': 'wiring test composition', 'observation_fields_used': ['missing_fraction']}}}],
        [{'tool': 'inspect_material', 'arguments': {'plan_id': 'W_comp', 'entity_index': 0}}],
        [{'tool': 'build_material', 'arguments': {'plan_id': 'W_recipe', 'policy': {'default': {'steps': [{'op': 'P_NoMixRecipe'}]}, 'rules': [],
                                                                                    'rationale': 'wiring test preset', 'observation_fields_used': []}}}],
        [{'tool': 'inspect_material', 'arguments': {'plan_id': 'W_recipe'}}],
        [{'tool': 'evaluate', 'arguments': {'plan_id': 'W_recipe'}}],
        [{'tool': 'evaluate', 'arguments': {'plan_id': 'W_comp'}}],
        [{'tool': 'compare', 'arguments': {'a': 'None', 'b': 'W_recipe'}}],
        None,                                                           # commit decided from the returned comparison (scripted)
    ]
    state = {'i': 0}

    def client(request):
        i = state['i']
        state['i'] += 1
        log.append({'call': i + 1, 'tools': request['tools'], 'tool_contract_names': sorted(request['tool_contracts']),
                    'overview_primitives': sorted(request['overview'].get('actions', {}).get('primitives', {})),
                    'overview_preset': sorted(request['overview'].get('actions', {}).get('preset', {})),
                    'remaining': request['remaining']})
        if script[i] is not None:
            return {'actions': script[i]}
        cmp_ = [e for e in request['current_trace'] if e['event'] == 'tool_completed' and e['tool'] == 'compare'][-1]['output']
        pick = 'W_recipe' if cmp_['mean_delta'] > 0 else 'None'
        return {'actions': [{'tool': 'commit', 'arguments': {'plan_id': pick, 'reason': 'wiring test: scripted rule on the returned C_A comparison'}}]}
    return client


def worker_wiring() -> None:
    """0 fits. Runs the unchanged batch_research.run_job with TempoAugAdapter in a SEPARATE run directory whose frozen materials are
    rebuilt from the same seeds (so the preset aliases to the fitted P_NoMixRecipe) and whose cells are the package's own."""
    rd._init_openmp_before_torch()
    out = {'written_local': now(), 'label': 'WIRING_TEST (scripted client; no LLM; not an agent result)', 'run_dir': str(WIRING_RUN), 'jobs': {}}
    for job in ('RD01B_Q1', 'RD02_T1'):
        reg = ta.load_registry(RUN / job)
        wdir = ta._mdir(WIRING_RUN / job)
        if not (wdir / 'index.json').exists():           # shared physical cache: the frozen materials' registry rows (paths into RUN)
            context.write_json(wdir / 'index.json', {m: reg[m] for m in NEW})
        ad = TempoAugAdapter(WIRING_RUN, job, seeds=SEEDS[job], job_index=JOB_INDEX[job], allow_fit=False)
        before = ta.load_registry(ad.ctx.job_dir)
        log = []
        res = br.run_job(job_id=job, knowledge=br.Knowledge(), adapter=ad, client=scripted_wiring_client(log), seeds=SEEDS[job],
                         baselines=ad.baselines(), allowed_features=frozenset('batch_median_' + f for f in rd.FIELDS),
                         tool_contracts=CONTRACTS, entity_count=len(ad.ctx.job.roster), ca_origins=2,
                         limits=br.Limits(max_calls=16, max_tools=24, max_new_evaluations=2, wall_seconds=1800.),
                         evidence_roundtrip=True, max_tool_corrections=1)
        after = ta.load_registry(ad.ctx.job_dir)
        new = sorted(set(after) - set(before))
        fits_after = len(list((RUN / job / 'runs').glob('*.pt'))) + len(list((WIRING_RUN / job).glob('**/*.pt')))
        out['jobs'][job] = {'status': res.status, 'failure_kind': res.failure_kind, 'reason': res.reason, 'committed': res.committed_plan_id,
                            'calls': res.calls, 'tool_calls': res.tool_calls, 'new_evaluations': res.new_evaluations,
                            'materials_added': {m: {'alias_of': after[m].get('alias_of'), 'material_index': after[m].get('material_index'),
                                                    'seed': after[m].get('seed')} for m in new},
                            'model_files_in_run_dir': fits_after, 'request_log': log,
                            'events': [{k: v for k, v in e.items() if k in ('event', 'tool', 'error', 'plan_id')} for e in res.trace]}
        tr = res.trace
        out['jobs'][job]['checks'] = {
            'all_7_primitives_visible_in_every_request': all(set(ta.PRIMITIVES) <= set(r['overview_primitives']) for r in log),
            'preset_visible': all(ta.RECIPE in r['overview_preset'] for r in log),
            'seven_tools_in_contracts': all(set(r['tool_contract_names']) == br.TOOLS for r in log),
            'preset_aliased_to_frozen_material': after.get('W_recipe', {}).get('alias_of') == 'P_NoMixRecipe',
            'composition_built_new_material': after.get('W_comp', {}).get('alias_of') is None and bool(after.get('W_comp', {}).get('seed')),
            'unfitted_evaluate_rejected_without_fit': any(e['event'] == 'tool_rejected' and e['tool'] == 'evaluate' for e in tr),
            'committed': res.status == 'COMPLETE'}
    context.write_json(ROOT / 'wiring.json', out)
    print('WIRING', {j: v['checks'] for j, v in out['jobs'].items()}, flush=True)


# ============================================================================= controller side (never imports torch)
def worker_env() -> dict:
    env = dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
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
        self.d = context.read_json(self.p) if self.p.exists() else {'caps': CAPS, 'fit': 0, 'check': 0, 'physical': 0, 'failed': [],
                                                                     'numeric_started_epoch': None, 'events': []}

    def save(self):
        context.write_json(self.p, self.d)

    def start_numeric(self):
        if self.d['numeric_started_epoch'] is None:
            self.d['numeric_started_epoch'] = time.time()
            self.save()

    def reserve(self, kind: str, what: str):
        if self.d['physical'] >= CAPS['hard'] or self.d[kind] >= CAPS[kind]:
            raise BudgetStop('%s cap reached before %s' % (kind, what))
        if self.d['numeric_started_epoch'] and time.time() - self.d['numeric_started_epoch'] > WALL_NUMERIC_S:
            raise BudgetStop('numeric wall clock reached before %s' % what)
        self.d[kind] += 1
        self.d['physical'] += 1
        self.save()

    def done(self, what: str, ok: bool, secs: float, kind: str):
        self.d['events'].append({'what': what, 'kind': kind, 'ok': ok, 'seconds': secs, 'local': now()})
        if not ok:
            self.d['failed'].append(what)
        self.save()


def cell_path(job: str, mid: str, seed: int, tag: str = '') -> Path:
    return RUN / job / 'cells' / (cid(job, mid, seed, tag) + '.json')


def fit(ledger: Ledger, job: str, mid: str, seed: int, kind: str = 'fit', tag: str = '') -> bool:
    if cell_path(job, mid, seed, tag).exists():
        return True
    ledger.reserve(kind, cid(job, mid, seed, tag))                  # every attempt counts; no retry allowance in this package
    args = ['--worker-fit', job, mid, str(seed)] + (['--tag', tag] if tag else [])
    rc, secs = run_sub(args, RUN / job / 'fit_logs' / (cid(job, mid, seed, tag) + '.log'), FIT_TIMEOUT_S)
    ok = rc == 0 and cell_path(job, mid, seed, tag).exists()
    ledger.done(cid(job, mid, seed, tag), ok, secs, kind)
    print('FIT', cid(job, mid, seed, tag), 'OK' if ok else 'FAILED rc=%s' % rc, round(secs, 1), flush=True)
    return ok


def environment() -> dict:
    code = 'import sys, torch, numpy, pandas; print(sys.version.split()[0], torch.__version__, numpy.__version__, pandas.__version__, torch.get_num_threads())'
    got = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, env=worker_env()).stdout.split()
    return {'python': got[0], 'torch': got[1], 'numpy': got[2], 'pandas': got[3], 'threads': int(got[4]), 'executable': sys.executable,
            'KMP_DUPLICATE_LIB_OK_at_start': KMP_AT_START, 'workers_env_kmp_removed': True}


def preflight() -> dict:
    env = environment()
    pf = context.read_json(P.ROOT / 'frozen_config.json').get('environment') or {}
    js_max = max(rd.resolve_job('RD02', j).e_target_rows[1] for j in JOBS if ds_of(j) == 'RD02')
    probe_files = {j: all((P.RUN / j / f).exists() for f in ('c_b_scores.json', 'e_scores.json', 'scaler.npz')) and
                   all(P.cell_path(j, m, s, P.N_FULL).exists() for m in ('None', 'FixedMixup') for s in SEEDS[j]) for j in JOBS}
    out = {'checked_local': now(), 'environment': env, 'probe_environment': pf,
           'env_matches_probe': bool(pf) and pf.get('python') == env['python'] and pf.get('torch') == env['torch'] and pf.get('numpy') == env['numpy'] and pf.get('threads') == env['threads'],
           'prsa_rows_used_max': js_max, 'prsa_sealed_from': P.PRSA_CLOSED_FROM_ROW, 'sealing_ok': js_max <= P.PRSA_CLOSED_FROM_ROW,
           'probe_cache_complete': probe_files, 'reference_repo': str(TEMPOPFN)}
    out['ok'] = out['env_matches_probe'] and out['sealing_ok'] and all(probe_files.values())
    context.write_json(ROOT / 'preflight.json', out)
    return out


def _probe_cell(job, mid, seed):
    return context.read_json(P.cell_path(job, mid, seed, P.N_FULL))


def check_none(ledger: Ledger) -> dict:
    res = {}
    for dom, (job, seed) in CHECK_NONE.items():
        ok = fit(ledger, job, 'None', seed, kind='check', tag='chk')
        mine = context.read_json(cell_path(job, 'None', seed, 'chk')) if ok else None
        cache = _probe_cell(job, 'None', seed)
        rc = subprocess.run([sys.executable, '-B', '-m', MODULE, '--worker-compare', mine['model_path'], cache['model_path']],
                            capture_output=True, text=True, cwd=str(REPO), env=worker_env()) if ok else None
        same = ok and json.loads(rc.stdout.strip().splitlines()[-1])['identical']
        res[dom] = {'job': job, 'seed': seed, 'fit_ok': ok, 'c_a_mine': mine and mine['scores']['c_a']['normalized_mse_macro'],
                    'c_a_cache': cache['scores']['c_a']['normalized_mse_macro'],
                    'c_a_equal': bool(ok and mine['scores']['c_a']['normalized_mse_macro'] == cache['scores']['c_a']['normalized_mse_macro']),
                    'weights_identical': bool(same)}
    res['ok'] = all(v['c_a_equal'] and v['weights_identical'] for k, v in res.items() if k != 'ok')
    context.write_json(ROOT / 'checks_none.json', res)
    return res


def check_rerun(ledger: Ledger) -> dict:
    res = {}
    for dom, (job, mid, seed) in CHECK_RERUN.items():
        ok = fit(ledger, job, mid, seed, kind='check', tag='rerun')
        a, b = context.read_json(cell_path(job, mid, seed)), (context.read_json(cell_path(job, mid, seed, 'rerun')) if ok else None)
        rc = subprocess.run([sys.executable, '-B', '-m', MODULE, '--worker-compare', a['model_path'], b['model_path']],
                            capture_output=True, text=True, cwd=str(REPO), env=worker_env()) if ok else None
        res[dom] = {'job': job, 'material': mid, 'seed': seed, 'fit_ok': ok,
                    'c_a_equal': bool(ok and a['scores']['c_a']['normalized_mse_macro'] == b['scores']['c_a']['normalized_mse_macro']),
                    'weights_identical': bool(ok and json.loads(rc.stdout.strip().splitlines()[-1])['identical'])}
    res['ok'] = all(v['c_a_equal'] and v['weights_identical'] for k, v in res.items() if k != 'ok')
    context.write_json(ROOT / 'checks_rerun.json', res)
    return res


def run() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / 'frozen_config.json').exists():
        raise RuntimeError('freeze frozen_config.json (via --smoke) before the run')
    st = {'started_local': now(), 'stages': {}}
    stp = ROOT / 'run_status.json'
    if stp.exists():
        st = context.read_json(stp)

    def mark(stage, **kw):
        st['stages'][stage] = {'local': now(), **kw}
        context.write_json(stp, st)

    pf = preflight()
    if not pf['ok']:
        mark('preflight', ok=False)
        raise RuntimeError('preflight failed')
    mark('preflight', ok=True)
    for job in JOBS:                                             # binding + frozen materials (0 fits)
        if not all(m in ta.load_registry(RUN / job) for m in NEW):
            rc, secs = run_sub(['--worker-build', job], RUN / job / 'build.log', 1800)
            if rc != 0:
                mark('build', ok=False, job=job)
                raise RuntimeError('build failed for %s' % job)
    frozen = {j: {m: {k: ta.load_registry(RUN / j)[m][k] for k in ('seed', 'material_index', 'key')} for m in NEW} for j in JOBS}
    context.write_json(ROOT / 'materials_frozen.json', {'frozen_local': now(), 'materials': frozen})
    mark('materials_frozen', ok=True)
    led = Ledger()
    led.start_numeric()
    t_num = time.time()
    cn = check_none(led)
    mark('check_none', ok=cn['ok'])
    if not cn['ok']:
        raise RuntimeError('None check failed: stop before new fits')
    for job in JOBS:
        for mid in NEW:
            for s in SEEDS[job]:
                if not fit(led, job, mid, s):
                    mark('fits', ok=False, failed=cid(job, mid, s))
                    raise RuntimeError('fit failed: %s (no retry allowance)' % cid(job, mid, s))
    mark('fits', ok=True)
    cr = check_rerun(led)
    mark('check_rerun', ok=cr['ok'])
    context.write_json(ROOT / 'report_objects.json', {'frozen_local': now(), 'objects': list(REPORTED),
                                                      'note': 'no selector: the two frozen materials are reported as they are, next to None and FixedMixup'})
    for job in JOBS:
        rc, _ = run_sub(['--worker-label', 'c_b', job], RUN / job / 'label_c_b.log', 900)
        if rc != 0:
            raise RuntimeError('C_B failed %s' % job)
    for job in JOBS:
        rc, _ = run_sub(['--worker-label', 'freeze_e', job], RUN / job / 'label_freeze_e.log', 900)
        if rc != 0:
            raise RuntimeError('freeze E failed %s' % job)
    context.write_json(ROOT / 'all_e_predictions_frozen.json', {'frozen_local': now(), 'jobs': list(JOBS)})
    for job in JOBS:
        rc, _ = run_sub(['--worker-label', 'score_e', job], RUN / job / 'label_score_e.log', 900)
        if rc != 0:
            raise RuntimeError('score E failed %s' % job)
    mark('labels', ok=True)
    mark('numeric_done', ok=True, numeric_seconds=time.time() - t_num)
    st['exit'] = 'COMPLETE'
    context.write_json(stp, st)


# ============================================================================= readout
def _stats(v: list) -> dict:
    m = statistics.fmean(v)
    se = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None
    t = T975.get(len(v) - 1)
    return {'mean': m, 'se': se, 't95': [m - t * se, m + t * se] if se is not None and t else None, 'per_seed': v}


def _scores(job: str) -> dict:
    """S[block][mid][seed] -> dict score (macro + per_origin_entity)."""
    S = {b: {m: {} for m in REPORTED} for b in ('c_a', 'c_b', 'e')}
    pcb = context.read_json(P.RUN / job / 'c_b_scores.json')['cells']
    pe = context.read_json(P.RUN / job / 'e_scores.json')['cells']
    ncb = context.read_json(RUN / job / 'c_b_scores.json')['cells']
    ne = context.read_json(RUN / job / 'e_scores.json')['cells']
    for s in SEEDS[job]:
        for m in ('None', 'FixedMixup'):
            c = P.cid(job, m, s, P.N_FULL)
            S['c_a'][m][s] = _probe_cell(job, m, s)['scores']['c_a']
            S['c_b'][m][s], S['e'][m][s] = pcb[c]['c_b'], pe[c]['e']
        for m in NEW:
            c = cid(job, m, s)
            S['c_a'][m][s] = context.read_json(cell_path(job, m, s))['scores']['c_a']
            S['c_b'][m][s], S['e'][m][s] = ncb[c]['c_b'], ne[c]['e']
    return S


def readout() -> dict:
    res = {'package': 'DEV-TEMPO-AUG-SOURCE-ALIGNMENT', 'written_local': now(), 'unit': {
        'loss_diff': 'paired per seed: loss(material) - loss(reference); negative = material better (normalized MSE macro)',
        'gain_pp': 'paired per seed: (loss(reference) - loss(material)) / 3-seed mean loss of None@2000 of that job, in percentage points of None'},
        'jobs': {}, 'pooled': {}}
    for job in JOBS:
        S = _scores(job)
        seeds = SEEDS[job]
        jr = {'macro': {b: {m: _stats([S[b][m][s]['normalized_mse_macro'] for s in seeds]) for m in REPORTED} for b in ('c_a', 'c_b', 'e')},
              'vs': {}}
        for b in ('c_a', 'c_b', 'e'):
            den = statistics.fmean(S[b]['None'][s]['normalized_mse_macro'] for s in seeds)
            for m in NEW:
                for ref in ('None', 'FixedMixup'):
                    d = [S[b][m][s]['normalized_mse_macro'] - S[b][ref][s]['normalized_mse_macro'] for s in seeds]
                    g = [-x / den * 100 for x in d]
                    entry = {'loss_diff': _stats(d), 'gain_pp': _stats(g)}
                    O = len(S[b][m][seeds[0]]['per_origin_entity_normalized_mse'])
                    per_o = []
                    for o in range(O):
                        do = [statistics.fmean(S[b][m][s]['per_origin_entity_normalized_mse'][o]) / O -
                              statistics.fmean(S[b][ref][s]['per_origin_entity_normalized_mse'][o]) / O for s in seeds]
                        per_o.append(_stats(do))
                    entry['origin_contrib_diff'] = per_o
                    entry['median_origin_diff_mean'] = statistics.median(x['mean'] * O for x in per_o)
                    entry['ref_origin_share'] = [statistics.fmean(statistics.fmean(S[b][ref][s]['per_origin_entity_normalized_mse'][o]) / O
                                                                  for s in seeds) / statistics.fmean(S[b][ref][s]['normalized_mse_macro'] for s in seeds) for o in range(O)]
                    jr['vs'].setdefault(b, {}).setdefault(m, {})[ref] = entry
        res['jobs'][job] = jr
    for b in ('c_a', 'c_b', 'e'):
        for m in NEW:
            for ref in ('None', 'FixedMixup'):
                dom = {}
                for d in ('RD01B', 'RD02'):
                    js = [j for j in JOBS if DOMAIN[j] == d]
                    xs = [res['jobs'][j]['vs'][b][m][ref]['gain_pp'] for j in js]
                    dom[d] = {'mean': statistics.fmean(x['mean'] for x in xs), 'se': math.sqrt(sum(x['se'] ** 2 for x in xs)) / len(xs),
                              'jobs_positive': sum(x['mean'] > 0 for x in xs), 'jobs': len(xs)}
                res['pooled'].setdefault(b, {}).setdefault(m, {})[ref] = {
                    'domains': dom, 'domain_equal_mean': statistics.fmean(v['mean'] for v in dom.values()),
                    'domain_equal_se': math.sqrt(sum(v['se'] ** 2 for v in dom.values())) / len(dom)}
    res['cost'] = cost_readout()
    res['materials'] = {j: {m: {k: v for k, v in context.read_json(ta.resolve(RUN / j, m)['summary_path']).items() if k != 'entities'}
                            for m in NEW} for j in JOBS}
    for f in ('preflight', 'checks_none', 'checks_rerun', 'parity', 'wiring', 'smoke_record', 'run_status', 'materials_frozen'):
        p = ROOT / (f + '.json')
        res[f] = context.read_json(p) if p.exists() else None
    res['binding'] = {j: context.read_json(RUN / j / 'binding.json') for j in JOBS}
    res['label_binding'] = {j: {'c_b_probe_none_rescore_equal': context.read_json(RUN / j / 'c_b_scores.json')['probe_none_rescore_equal'],
                                'e_probe_none_rescore_equal': context.read_json(RUN / j / 'e_scores.json')['probe_none_rescore_equal']} for j in JOBS}
    context.write_json(ROOT / 'result.json', res)
    (ROOT / 'tables.md').write_text(tables(res), encoding='utf-8')
    return res


def cost_readout() -> dict:
    led = context.read_json(ROOT / 'budget.json')
    ev = led['events']
    fits = [e for e in ev if e['kind'] == 'fit' and e['ok']]
    checks = [e for e in ev if e['kind'] == 'check']
    build = {j: {m: context.read_json(ta.resolve(RUN / j, m)['summary_path'])['build_seconds'] for m in NEW} for j in JOBS}
    st = context.read_json(ROOT / 'run_status.json')
    return {'physical_fits': led['physical'], 'fits': led['fit'], 'checks': led['check'], 'failed': led['failed'],
            'fit_worker_seconds_total': sum(e['seconds'] for e in fits), 'fit_worker_seconds_median': statistics.median(e['seconds'] for e in fits) if fits else None,
            'check_worker_seconds_total': sum(e['seconds'] for e in checks), 'material_build_seconds': build,
            'material_build_seconds_total': sum(v for d in build.values() for v in d.values()),
            'numeric_wall_seconds': (st.get('stages', {}).get('numeric_done') or {}).get('numeric_seconds'),
            'cache_reused_cells': {'None': 12, 'FixedMixup': 12, 'source': 'DEV-DATA-AUG-OPERATOR-PROBE (u2000 cells, C_B / E scores)'},
            'llm_calls': 0}


def _f(x, nd=1):
    return '—' if x is None else ('%+.*f' % (nd, x))


def tables(res: dict) -> str:
    out = ['# DEV-TEMPO-AUG-SOURCE-ALIGNMENT tables', '', 'gain_pp = (loss_ref − loss_material) / 3-seed mean None loss of the job, percentage points; positive = material better. SE over 3 paired seeds; t95 = n=3 two-sided t interval.', '']
    for b in ('e', 'c_b', 'c_a'):
        out += ['## %s' % b.upper(), '', '| Job | material | vs | gain_pp | SE | 95% t | per seed | median-origin diff (loss) |', '|---|---|---|---:|---:|---|---|---:|']
        for job in JOBS:
            for m in NEW:
                for ref in ('None', 'FixedMixup'):
                    e = res['jobs'][job]['vs'][b][m][ref]
                    g = e['gain_pp']
                    out.append('| %s | %s | %s | %s | %.1f | [%s, %s] | %s | %s |' % (
                        job, m, ref, _f(g['mean']), g['se'], _f(g['t95'][0]), _f(g['t95'][1]), ', '.join(_f(x) for x in g['per_seed']),
                        _f(e['median_origin_diff_mean'], 3)))
        out += ['', 'Domain-equal pooling (%s):' % b.upper(), '', '| material | vs | RD01B | RD02 | domain-equal (SE) |', '|---|---|---:|---:|---:|']
        for m in NEW:
            for ref in ('None', 'FixedMixup'):
                p = res['pooled'][b][m][ref]
                out.append('| %s | %s | %s (%.1f) | %s (%.1f) | %s (%.1f) |' % (m, ref, _f(p['domains']['RD01B']['mean']), p['domains']['RD01B']['se'],
                                                                         _f(p['domains']['RD02']['mean']), p['domains']['RD02']['se'],
                                                                         _f(p['domain_equal_mean']), p['domain_equal_se']))
        out.append('')
    out += ['## E origin contributions (loss diff of material vs None per origin; share of None macro)', '']
    for job in JOBS:
        for m in NEW:
            e = res['jobs'][job]['vs']['e'][m]['None']
            out.append('- %s %s: ' % (job, m) + '; '.join('o%d %s±%.3f (None share %.0f%%)' % (o, _f(x['mean'], 3), x['se'], 100 * sh)
                                                         for o, (x, sh) in enumerate(zip(e['origin_contrib_diff'], e['ref_origin_share']))))
    return '\n'.join(out) + '\n'


# ============================================================================= smoke (0 fits)
def smoke() -> dict:
    """Fast checks of the new risks only (task §7). Fits: 0. Parity runs in its own subprocess (reads the clone read-only)."""
    import pandas as pd  # noqa: F401  (controller may use pandas; never torch)
    ROOT.mkdir(parents=True, exist_ok=True)
    checks, notes = {}, {}
    # 5 (part): label barrier refuses before any file exists
    for stage in ('c_b', 'freeze_e', 'score_e'):
        try:
            if not (ROOT / 'report_objects.json').exists():
                label_prerequisite(stage, 'RD02_T1')
                checks['label_refusal_%s' % stage] = False
        except PermissionError:
            checks['label_refusal_%s' % stage] = True
    # 3: DSL order, exclusion, max steps, no mixup, preset alone
    def rejects(steps):
        try:
            ta.validate_steps(steps)
            return False
        except rd.PolicyError:
            return True
    checks['dsl_fixed_order'] = ta.validate_steps([{'op': 'tp_random_conv'}, {'op': 'tp_censor'}, {'op': 'tp_shock'}]) == [{'op': 'tp_shock'}, {'op': 'tp_censor'}, {'op': 'tp_random_conv'}]
    checks['dsl_exclusive'] = rejects([{'op': 'tp_regime'}, {'op': 'tp_shock'}]) and rejects([{'op': 'tp_calendar'}, {'op': 'tp_amplitude'}])
    checks['dsl_max3_no_repeat'] = rejects([{'op': 'tp_shock'}, {'op': 'tp_calendar'}, {'op': 'tp_resample'}, {'op': 'tp_censor'}]) and rejects([{'op': 'tp_censor'}, {'op': 'tp_censor'}])
    checks['dsl_no_mixup_or_params'] = rejects([{'op': 'timemixup'}]) and rejects([{'op': 'tp_shock', 'mag': 2.0}]) and rejects([{'op': 'tp_shock', 'profile': 'strong'}])
    checks['dsl_preset_alone'] = rejects([{'op': ta.RECIPE}, {'op': 'tp_censor'}]) and ta.validate_steps([{'op': ta.RECIPE}]) == [{'op': ta.RECIPE}]
    rc, _ = run_sub(['--worker-parity'], ROOT / 'smoke_parity.log', 1800)
    par = context.read_json(ROOT / 'parity.json') if (ROOT / 'parity.json').exists() else {'ok': False}
    checks['parity_7_primitives_chains_frequency_verbatim'] = rc == 0 and par['ok']
    # 2 + determinism of material rebuild: done inside a worker (needs torch) on a scratch copy of RD02_T1
    rc2, _ = run_sub(['--worker-smoke-materials'], ROOT / 'smoke_materials.log', 1800)
    sm = context.read_json(ROOT / 'smoke_materials.json') if (ROOT / 'smoke_materials.json').exists() else {}
    for k, v in sm.get('checks', {}).items():
        checks[k] = bool(v) and rc2 == 0
    notes['materials'] = sm.get('notes')
    rec = {'written_local': now(), 'checks': checks, 'passed': sum(checks.values()), 'of': len(checks), 'notes': notes}
    context.write_json(ROOT / 'smoke_record.json', rec)
    return rec


def worker_smoke_materials() -> None:
    """Window/date/split binding, parent invariance, recipe structure, rebuild determinism and alias semantics on a scratch run dir."""
    rd._init_openmp_before_torch()
    import shutil
    scratch = ROOT / 'smoke_scratch'
    if scratch.exists():
        shutil.rmtree(scratch)
    ctx = rd.open_job('RD02', 'RD02_T1', scratch, stage='material')
    Z, ent, k = ta.joint_parents(ctx)
    starts = ta.window_starts(ctx)
    lg = ctx.legal
    checks, notes = {}, {}
    # dates: first window start = absolute row t-672 + k[0]; PRSA base 2013-03-01 00:00
    r0 = ctx.job.train_range[0] + int(lg.k[0])
    import pandas as pd
    checks['dates_bound_to_rows'] = starts[0] == pd.Timestamp(2013, 3, 1) + pd.Timedelta(hours=r0) and starts[-1] == pd.Timestamp(2013, 3, 1) + pd.Timedelta(hours=ctx.job.train_range[0] + int(lg.k[-1]))
    notes['first_window_start'] = str(starts[0])
    checks['split_192_48'] = Z.shape[1] == 240 and np.array_equal(Z[:, 192:], (lg.y_raw - ctx.scaler.mean[lg.ent][:, None]) / ctx.scaler.scale[lg.ent][:, None])
    table = ctx.overview()['entities']
    comp = ta.compile_plan({'default': {'steps': [{'op': 'tp_resample'}, {'op': 'tp_shock'}, {'op': 'tp_calendar'}]}, 'rules': []}, table)
    X_before = Z.copy()
    rec1 = ta.build_material(ctx, comp['assignment'], 'S_comp', job_index=9, compiled=comp)
    checks['parents_unchanged_after_build'] = np.array_equal(ta.joint_parents(ctx)[0], X_before)
    c1 = ta.load_child(ctx.job_dir, 'S_comp')
    # rebuild determinism: a second registry with the same seed index gives the identical child
    scratch2 = ROOT / 'smoke_scratch2'
    if scratch2.exists():
        shutil.rmtree(scratch2)
    ctx2 = rd.open_job('RD02', 'RD02_T1', scratch2, stage='material')
    ta.build_material(ctx2, comp['assignment'], 'S_comp', job_index=9, material_index=rec1['material_index'], compiled=comp)
    c2 = ta.load_child(ctx2.job_dir, 'S_comp')
    checks['material_rebuild_deterministic'] = np.array_equal(c1[0], c2[0]) and np.array_equal(c1[1], c2[1])
    comp_perm = ta.compile_plan({'default': {'steps': [{'op': 'tp_calendar'}, {'op': 'tp_resample'}, {'op': 'tp_shock'}]}, 'rules': []}, table)
    rec_alias = ta.build_material(ctx, comp_perm['assignment'], 'S_comp_again', job_index=9, compiled=comp_perm)
    checks['same_semantics_alias_no_new_seed'] = rec_alias['alias_of'] == 'S_comp' and rec_alias['seed'] is None
    empty = ta.compile_plan({'default': {'steps': []}, 'rules': []}, table)
    checks['all_empty_is_None'] = ta.build_material(ctx, empty['assignment'], 'S_none', job_index=9, compiled=empty)['alias_of'] == 'None'
    rec_r = ta.build_material(ctx, ta.compile_plan({'default': {'steps': [{'op': ta.RECIPE}]}, 'rules': []}, table)['assignment'], 'S_recipe', job_index=9)
    sr = context.read_json(rec_r['summary_path'])
    sets = sr['recipe_step_sets']
    lens = {len(s.split('+')) for s in sets}
    bad = [s for s in sets if ('tp_regime' in s and 'tp_shock' in s) or ('tp_calendar' in s and 'tp_amplitude' in s) or 'mixup' in s]
    checks['recipe_2_to_3_steps_fixed_order_exclusive'] = lens <= {2, 3} and not bad and all(
        [ta.FIXED_ORDER[x] for x in s.split('+')] == sorted(ta.FIXED_ORDER[x] for x in s.split('+')) for s in sets)
    conv_rate = sum(v for s, v in sets.items() if 'tp_random_conv' in s) / sum(sets.values())
    notes['recipe_conv_rate'] = conv_rate
    notes['recipe_step_sets'] = sets
    notes['recipe_censor_identity'] = sr['per_step']['tp_censor']
    checks['recipe_conv_rate_near_0.3'] = abs(conv_rate - 0.3) < 0.03
    checks['children_finite_float32'] = c1[0].dtype == np.float32 and np.isfinite(c1[0]).all() and np.isfinite(c1[1]).all()
    # a mixed plan: identity entities carry their parent in the child slot
    mixed = ta.compile_plan({'default': {'steps': []}, 'rules': [{'when': {'feature': 'missing_fraction', 'op': '>=', 'value': {'quantile': 0.5}},
                                                                   'steps': [{'op': 'tp_censor'}]}]}, table)
    ta.build_material(ctx, mixed['assignment'], 'S_mixed', job_index=9, compiled=mixed)
    cm = ta.load_child(ctx.job_dir, 'S_mixed')
    idle = [e for e, st in enumerate(mixed['assignment']) if not st]
    rows = np.isin(ent, idle)
    checks['identity_entities_parent_in_child_slot'] = bool(idle) and np.array_equal(cm[0][rows], Z[rows, :192].astype(np.float32))
    # tool catalog of the real Fast request comes from the adapter overview + contracts
    ad = TempoAugAdapter(scratch, 'RD02_T1', seeds=SEEDS['RD02_T1'], job_index=9)
    ov = ad.overview()
    checks['catalog_all_7_and_preset'] = sorted(ov['actions']['primitives']) == sorted(ta.PRIMITIVES) and ta.RECIPE in ov['actions']['preset'] and set(CONTRACTS) == br.TOOLS
    context.write_json(ROOT / 'smoke_materials.json', {'checks': {k: bool(v) for k, v in checks.items()}, 'notes': notes})
    shutil.rmtree(scratch, ignore_errors=True)
    shutil.rmtree(scratch2, ignore_errors=True)
    print('SMOKE_MATERIALS', checks, flush=True)


def freeze_config() -> None:
    cfg = {'package': 'DEV-TEMPO-AUG-SOURCE-ALIGNMENT', 'frozen_local': now(), 'task': 'docs/DEV_TEMPO_AUG_SOURCE_ALIGNMENT_TASK_2026-09-18.md',
           'jobs': list(JOBS), 'job_index': JOB_INDEX, 'seeds': {j: list(SEEDS[j]) for j in JOBS}, 'n_updates': N,
           'materials': {m: {'index': v['index'], 'policy': plan_policy(m), 'seed_rule': 'base %d + 10000 x job_index + 100 x material_index' % ta.SEED_BASE,
                             'seeds': {j: ta.material_seed(JOB_INDEX[j], v['index']) for j in JOBS}} for m, v in MATERIALS.items()},
           'recipe': ta.RECIPE_TEXT, 'catalog': ta.CATALOG, 'fixed_order': list(ta.PRIMITIVES), 'exclusive': [list(x) for x in ta.EXCLUSIVE],
           'reused_from_probe': {'None': 'u2000 cells + C_B/E', 'FixedMixup': 'u2000 cells + C_B/E', 'probe_root': str(P.ROOT)},
           'checks': {'none_retrain': CHECK_NONE, 'rerun': CHECK_RERUN}, 'caps': CAPS, 'wall_numeric_s': WALL_NUMERIC_S,
           'parity_tolerance': PARITY_TOL, 'tempopfn_commit': TEMPOPFN_COMMIT, 'llm_calls': 0}
    context.write_json(ROOT / 'frozen_config.json', cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--worker-build')
    ap.add_argument('--worker-fit', nargs=3)
    ap.add_argument('--tag', default='')
    ap.add_argument('--worker-label', nargs=2)
    ap.add_argument('--worker-compare', nargs=2)
    ap.add_argument('--worker-parity', action='store_true')
    ap.add_argument('--worker-smoke-materials', action='store_true')
    ap.add_argument('--worker-wiring', action='store_true')
    a = ap.parse_args()
    if a.worker_build:
        worker_build(a.worker_build)
    elif a.worker_fit:
        worker_fit(a.worker_fit[0], a.worker_fit[1], int(a.worker_fit[2]), a.tag)
    elif a.worker_label:
        worker_label(*a.worker_label)
    elif a.worker_compare:
        worker_compare(*a.worker_compare)
    elif a.worker_parity:
        worker_parity()
    elif a.worker_smoke_materials:
        worker_smoke_materials()
    elif a.worker_wiring:
        worker_wiring()
    elif a.smoke:
        print(json.dumps(smoke(), indent=1))
    elif a.freeze:
        freeze_config()
    elif a.run:
        run()
    elif a.wiring:
        rc, _ = run_sub(['--worker-wiring'], ROOT / 'wiring.log', 1800)
        print('wiring rc', rc)
    elif a.result:
        readout()


if __name__ == '__main__':
    main()
