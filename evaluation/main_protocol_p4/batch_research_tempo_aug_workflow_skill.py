"""DEV-TEMPO-AUG-WORKFLOW-SKILL (docs/DEV_TEMPO_AUG_WORKFLOW_SKILL_TASK_DRAFT_2026-09-18.md): same-domain augmentation Workflow
learning (Eval-Skill style generation -> selection, Workflow only) and its check on four unscored follow-up batches.

One package, one ledger: preflight (0 cost) -> wiring acceptance (RD02_T1, <=5 fits, scripted client, 0 LLM) -> Source (4 no-Skill Fast,
C_B only) -> formation (census + ONE Slow) -> Select (no_skill / W1 / W2 on T4 and Q2, J(v) on C_B) -> freeze -> Target (L1-L4:
F_noSkill, F_domainSkill, RandomSearch_B4, Menu_CA + four public references; whole-stage C_B -> E barrier -> E) -> readout.

Reused unchanged: the Fast controller (batch_research.run_job via run_batch_research_v1.branch / resume_branch), the metered client and
ledger, domain_skill parse/freeze, the readiness T view / scoring, tempo_aug primitives / DSL / inspection. New here (opt-in only):
the per-(job, entity, program) material RNG (profile tempo_source_v2_entity_program), the shared physical cache with per-branch views,
the fit / material / label workers of this profile, RandomSearch_B4, Menu_CA, the J(v) selection and the readout.

  --smoke | --preflight | --wiring | --run | --resume-stage STAGE [--accept-unknown-usage] | --result
  workers: --stage-worker CFG | --worker-build ROOT JOB JOB_INDEX | --worker-material ROOT JOB PHYS JOB_INDEX |
           --worker-fit ROOT JOB PHYS SEED [--tag T] | --worker-label STAGE BRANCH JOB STAGE_ROOT | --worker-compare A B | --worker-smoke OUT
The controller never imports torch; every torch use is a subprocess that first calls rd._init_openmp_before_torch().
"""
from __future__ import annotations

import os

KMP_AT_START = os.environ.get('KMP_DUPLICATE_LIB_OK')

import argparse
import copy
import itertools
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha.batch_base import augment, budget, context, spec, tempo_aug as ta
from methods.ttha.batch_base import readiness as rd
from evaluation.main_protocol_p4 import batch_research_domain_skill as dsks
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import run_batch_research_v1 as base
from evaluation.main_protocol_p4 import batch_research_data_aug_operator_probe as P      # read-only: historical RD02_T1 None / FixedMixup

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_tempo_aug_workflow_skill'
TASK = 'docs/DEV_TEMPO_AUG_WORKFLOW_SKILL_TASK_DRAFT_2026-09-18.md'
MODULE = 'evaluation.main_protocol_p4.batch_research_tempo_aug_workflow_skill'
PACKAGE = 'DEV-TEMPO-AUG-WORKFLOW-SKILL'
DATASET = 'RD02'
DOMAIN = 'RD02'
# Amendment 1 (before any formal fit): the task book's [2026091801, 2026091802, 2026091803] overflow the frozen Consumer batch stream
# (spec.batch_seed = 800000 + 100 x seed + 9 must be < 2**32; the Consumer path is not modified). Same seeds in the 8-digit encoding
# "2026-918-k"; two failed worker attempts with the overflowing seed stay in the ledger (RD02_S1 None s2026091801 + its retry).
TASK_BOOK_SEEDS = (2026091801, 2026091802, 2026091803)
SEEDS = (20269181, 20269182, 20269183)
WIRING_SEEDS = (20261001, 20261002, 20261003)
WIRING_JOB = 'RD02_T1'
JOB_INDEX = {'RD02_T1': 0, 'RD02_S1': 1, 'RD02_S2': 2, 'RD02_V1': 3, 'RD02_T2': 4, 'RD02_T4': 5, 'RD02_Q2': 6,
             'RD02_L1': 7, 'RD02_L2': 8, 'RD02_L3': 9, 'RD02_L4': 10}
JOB_T = {'RD02_T1': 6960, 'RD02_S1': 3360, 'RD02_S2': 4560, 'RD02_V1': 5760, 'RD02_T2': 8160, 'RD02_T4': 10560, 'RD02_Q2': 12960,
         'RD02_L1': 15360, 'RD02_L2': 16560, 'RD02_L3': 18960, 'RD02_L4': 20160}
SOURCE_JOBS = ('RD02_S1', 'RD02_S2', 'RD02_V1', 'RD02_T2')
SELECT_JOBS = ('RD02_T4', 'RD02_Q2')
TARGET_JOBS = ('RD02_L1', 'RD02_L2', 'RD02_L3', 'RD02_L4')
SELECT_ORDER = {'RD02_T4': ('no_skill', 'cand_W1', 'cand_W2'), 'RD02_Q2': ('cand_W2', 'cand_W1', 'no_skill')}
TARGET_ORDER = {'RD02_L1': ('f_no_skill', 'f_domain_skill', 'random'), 'RD02_L2': ('f_domain_skill', 'random', 'f_no_skill'),
                'RD02_L3': ('random', 'f_no_skill', 'f_domain_skill'), 'RD02_L4': ('f_domain_skill', 'f_no_skill', 'random')}
PUBLIC = ('None', 'FixedMixup', 'P_AmpResample', 'P_NoMixRecipe')
WIRING_PUBLIC = ('None', 'FixedMixup')
PUBLIC_STEPS = {'None': [], 'P_AmpResample': [{'op': 'tp_amplitude'}, {'op': 'tp_resample'}], 'P_NoMixRecipe': [{'op': ta.RECIPE}]}
FIXED_MIXUP = {'donor_rule': 'R', 'w': 0.25, 'seed_rule': 'RandomState(918000000 + 1000 x 1 + entity) per entity (DEV-DATA-AUG-OPERATOR-PROBE material_no 1)',
               'definition': 'augment.step_timemixup on the joint normalized [X;y] legal windows of each entity: (1-w) z + w z[donor], donor = derangement of the entity\'s own legal parents'}
LIMITS = {'max_calls': 16, 'max_tools': 24, 'max_new_evaluations': 4}
FAST_TOKEN_CAP = 400_000
MAX_OUTPUT_TOKENS = 12_000
MAX_TOOL_CORRECTIONS = 2
EVIDENCE_ROUNDTRIP = True
TOTAL = {'max_fit_attempts': 395, 'max_llm_requests': 290, 'max_llm_tokens': 7_600_000, 'max_wall_s': 36_000, 'max_retries': 6}
STAGE_RETRIES = 2
HTTP_CAP = 294
ALLOC = {'wiring': {'fits': 5, 'requests': 0, 'tokens': 0},
         'source': {'fits': 96, 'requests': 64, 'tokens': 1_600_000},
         'formation': {'fits': 0, 'requests': 2, 'tokens': 400_000},
         'select': {'fits': 96, 'requests': 96, 'tokens': 2_400_000},
         'target': {'fits': 192, 'requests': 128, 'tokens': 3_200_000}}
NUMERIC_WALL_S = 4 * 3600.
FIT_TIMEOUT_S = 300.
MODEL = {'requested': 'cpa-grok-4.6', 'returned_required': 'grok-4.6-build', 'temperature': 0, 'base_url': 'http://127.0.0.1:8318/v1',
         'max_output_tokens': MAX_OUTPUT_TOKENS, 'other_decoding': 'openai chat.completions defaults (no top_p / penalties set), as in every readiness package'}
PROFILE_VERSION = 'tempo_source_v2_entity_program'
SEED_ROOT = 2026091800
RANDOM_SEED_ROOT = 2026091801
RANDOM_QUANTILES = (0.25, 0.5, 0.75)
RANDOM_MAX_TRIES = 256
T975_DF2 = 4.302652729911275
ALLOWED_FEATURES = frozenset('batch_median_' + f for f in rd.FIELDS)
PRSA_SEALED_FROM_ROW = P.PRSA_CLOSED_FROM_ROW
AVAILABILITY = REPO / '_scratch' / 'dev_tempo_aug_source_alignment' / 'data_availability' / 'availability.json'
CONSUMER = {**spec.CONSUMER,
            'loss': 'pooled MSE on normalized targets: 0.5 x parent view (all legal parent windows) + 0.5 x child view (the plan\'s augmented copy of '
                    'every legal parent; identity entities put their parent in the child slot); None trains the parent view only',
            'training': 'shared MLP, AdamW, 2000 updates, batch 64, the same legal parent pool, T scaler and batch index stream for every plan',
            'normalization': rd.CONSUMER['normalization'], 'metric': rd.CONSUMER['metric'],
            'serving': 'prediction inputs are always the Baseline-Linear filled 192 points; scoring targets are the raw observed future (never augmented)'}


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def n_of(job: str) -> int:
    return len(rd.DATASETS[DATASET]['roster'])


# ============================================================================= program table and the per-(job, entity, program) RNG
def program_table() -> list:
    """index -> canonical steps. 0 = empty; 1..51 = legal explicit compositions by length 1, 2, 3 in PRIMITIVES (= execution) order;
    52 = P_NoMixRecipe. Frozen before the first real material of this package."""
    out = [[]]
    for n in (1, 2, 3):
        for combo in itertools.combinations(ta.PRIMITIVES, n):
            if any(a in combo and b in combo for a, b in ta.EXCLUSIVE):
                continue
            out.append([{'op': x} for x in combo])
    out.append([{'op': ta.RECIPE}])
    if len(out) != 53:
        raise RuntimeError('program table must hold 53 programs')
    return out


PROGRAMS = program_table()
_PKEY = lambda steps: json.dumps(steps, sort_keys=True, separators=(',', ':'))
PROGRAM_INDEX = {_PKEY(p): i for i, p in enumerate(PROGRAMS)}


def program_index(steps: list) -> int:
    return PROGRAM_INDEX[_PKEY(ta.validate_steps(steps))]


def material_key(assignment: list) -> str:
    return json.dumps({'profile': PROFILE_VERSION, 'assignment': assignment}, sort_keys=True, separators=(',', ':'))


def entity_program_seed(job_index: int, prog_index: int, entity_index: int) -> int:
    return int(np.random.SeedSequence([SEED_ROOT, int(job_index), int(prog_index), int(entity_index)]).generate_state(1, dtype=np.uint32)[0])


def uniform_policy(steps: list, rationale: str) -> dict:
    return {'default': {'steps': steps}, 'rules': [], 'rationale': rationale, 'observation_fields_used': []}


def public_policy(mid: str) -> dict:
    return uniform_policy(PUBLIC_STEPS[mid], 'public reference %s' % mid)


# ============================================================================= physical cache (per job) and branch views
def aug_dir(job_dir: Path) -> Path:
    d = Path(job_dir) / 'aug_materials'
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_registry(job_dir: Path) -> dict:
    p = Path(job_dir) / 'aug_materials' / 'index.json'
    return context.read_json(p) if p.exists() else {}


def save_registry(job_dir: Path, reg: dict) -> None:
    context.write_json(aug_dir(job_dir) / 'index.json', reg)


def phys_cell_id(job: str, phys: str, seed: int, tag: str = '') -> str:
    return '%s__%s__s%d%s' % (job, phys, seed, ('__' + tag) if tag else '')


def build_material_v2(ctx, assignment: list, phys_id: str, *, job_index: int) -> dict:
    """Freeze one physical material (child view of every legal parent) under the per-(job, entity, program) RNG. Torch process only."""
    reg = load_registry(ctx.job_dir)
    if phys_id in reg:
        raise ValueError('physical material id exists')
    key = material_key(assignment)
    if any(r['key'] == key for r in reg.values()):
        raise ValueError('a physical material with this assignment already exists')
    if ta.is_identity(assignment):
        raise ValueError('the all-empty plan is None; no physical material')
    t0 = time.time()
    Z, ent, k = ta.joint_parents(ctx)
    starts = ta.window_starts(ctx) if any(st['op'] in ('tp_calendar', ta.RECIPE) for steps in assignment for st in steps) else None
    C = np.empty(Z.shape, dtype=np.float32)
    per_step = {n: {'windows': 0, 'identity': 0} for n in ta.PRIMITIVES}
    recipe_sets, window_identity, seeds = {}, 0, {}
    for e in range(len(ctx.job.roster)):
        rows = np.flatnonzero(ent == e)
        prog = assignment[e]
        if not prog:
            C[rows] = Z[rows].astype(np.float32)
            continue
        pi = program_index(prog)
        seed = entity_program_seed(job_index, pi, e)
        seeds['entity_%d' % e] = {'program_index': pi, 'seed': seed}
        streams = ta.Streams(seed)
        for p in rows:                                             # frozen window order inside the entity
            out, executed, ident = ta.run_program(Z[p], prog, streams, starts[p] if starts is not None else None)
            C[p] = out
            if prog == [{'op': ta.RECIPE}]:
                recipe_sets['+'.join(executed)] = recipe_sets.get('+'.join(executed), 0) + 1
            for n, i in zip(executed, ident):
                per_step[n]['windows'] += 1
                per_step[n]['identity'] += int(i)
            window_identity += int(np.array_equal(out, Z[p].astype(np.float32)))
    secs = time.time() - t0
    d = C.astype(np.float64) - Z
    mdir = aug_dir(ctx.job_dir)
    npz = mdir / (phys_id + '.npz')
    np.savez(npz, Xc=C[:, :spec.L], yc=C[:, spec.L:])
    ents = []
    for e in range(len(ctx.job.roster)):
        rows = np.flatnonzero(ent == e)
        de = d[rows]
        ents.append({'steps': assignment[e], 'windows': int(rows.size),
                     'rms_change_X': float(np.sqrt((de[:, :spec.L] ** 2).mean())), 'rms_change_y': float(np.sqrt((de[:, spec.L:] ** 2).mean())),
                     'fraction_points_changed_X': float((np.abs(de[:, :spec.L]) > 0).mean()), 'fraction_points_changed_y': float((np.abs(de[:, spec.L:]) > 0).mean())})
    summary = {'material_id': phys_id, 'job_id': ctx.job.job_id, 'profile': PROFILE_VERSION, 'job_index': job_index, 'entity_program_seeds': seeds,
               'seed_rule': 'SeedSequence([%d, job_index, program_index, entity_index]).generate_state(1, uint32)[0]' % SEED_ROOT,
               'legal_parents': int(Z.shape[0]), 'build_seconds': secs,
               'rms_change_X': float(np.sqrt((d[:, :spec.L] ** 2).mean())), 'rms_change_y': float(np.sqrt((d[:, spec.L:] ** 2).mean())),
               'fraction_points_changed_X': float((np.abs(d[:, :spec.L]) > 0).mean()), 'fraction_points_changed_y': float((np.abs(d[:, spec.L:]) > 0).mean()),
               'windows_bitwise_unchanged': window_identity, 'per_step': per_step, 'recipe_step_sets': recipe_sets,
               'units': 'normalized by the frozen per-entity T scaler (child - parent)', 'entities': ents}
    sp = mdir / (phys_id + '__summary.json')
    context.write_json(sp, summary)
    rec = {'material_id': phys_id, 'job_id': ctx.job.job_id, 'key': key, 'assignment': assignment, 'profile': PROFILE_VERSION, 'alias_of': None,
           'path': str(npz), 'summary_path': str(sp), 'seed': None, 'material_index': int(phys_id[2:]) if phys_id.startswith('TA') else None,
           'kind': 'tempo_v2', 'built_local': now()}
    reg = load_registry(ctx.job_dir)
    reg[phys_id] = rec
    save_registry(ctx.job_dir, reg)
    return rec


def build_fixed_mixup(ctx) -> dict:
    """The old public reference: timemixup donor rule R, w = 0.25, the probe's per-entity RandomState rule; no TempoPFN primitive."""
    Z, ent, k = ta.joint_parents(ctx)
    C = np.empty(Z.shape, dtype=np.float64)
    for e in range(len(ctx.job.roster)):
        rows = np.flatnonzero(ent == e)
        rng = np.random.RandomState(P.AUG_SEED_BASE + 1000 * P.MAT['FixedMixup']['no'] + e)
        out, _ = augment.step_timemixup(Z[rows], Z[rows], np.zeros(rows.size), rng, FIXED_MIXUP['donor_rule'], FIXED_MIXUP['w'])
        C[rows] = out
    C32 = C.astype(np.float32)
    mdir = aug_dir(ctx.job_dir)
    npz = mdir / 'FixedMixup.npz'
    np.savez(npz, Xc=C32[:, :spec.L], yc=C32[:, spec.L:])
    d = C32.astype(np.float64) - Z
    sp = mdir / 'FixedMixup__summary.json'
    context.write_json(sp, {'material_id': 'FixedMixup', 'job_id': ctx.job.job_id, 'legal_parents': int(Z.shape[0]), **FIXED_MIXUP,
                            'rms_change_X': float(np.sqrt((d[:, :spec.L] ** 2).mean())), 'rms_change_y': float(np.sqrt((d[:, spec.L:] ** 2).mean()))})
    return {'material_id': 'FixedMixup', 'job_id': ctx.job.job_id, 'key': 'FixedMixup:R:w0.25:probe_rng', 'assignment': None, 'profile': 'timemixup_probe',
            'alias_of': None, 'path': str(npz), 'summary_path': str(sp), 'seed': None, 'material_index': None, 'kind': 'fixed_mixup', 'built_local': now()}


def none_record(job: str, n: int) -> dict:
    return {'material_id': 'None', 'job_id': job, 'key': material_key([[] for _ in range(n)]), 'assignment': [[] for _ in range(n)], 'profile': PROFILE_VERSION,
            'alias_of': None, 'path': None, 'summary_path': None, 'seed': None, 'material_index': None, 'kind': 'none'}


# ============================================================================= workers (subprocess entries; torch allowed)
def worker_build(root: str, job: str, job_index: int) -> None:
    """P0_Linear (readiness builder), the None record, the FixedMixup child and the two public tempo materials of one physical cache."""
    rd._init_openmp_before_torch()
    ctx = rd.open_job(DATASET, job, Path(root), stage='material')
    ta.joint_parents(ctx)                                          # builds P0_Linear on demand
    reg = load_registry(ctx.job_dir)
    if 'None' not in reg:
        reg['None'] = none_record(job, len(ctx.job.roster))
        save_registry(ctx.job_dir, reg)
    if 'FixedMixup' not in reg:
        reg = load_registry(ctx.job_dir)
        reg['FixedMixup'] = build_fixed_mixup(ctx)
        save_registry(ctx.job_dir, reg)
    table = ctx.overview()['entities']
    for mid in ('P_AmpResample', 'P_NoMixRecipe'):
        if mid in load_registry(ctx.job_dir):
            continue
        compiled = ta.compile_plan(public_policy(mid), table)
        rec = build_material_v2(ctx, compiled['assignment'], mid, job_index=job_index)
        print('MATERIAL', job, mid, 'build_s %.1f' % context.read_json(rec['summary_path'])['build_seconds'], flush=True)
    print('BUILD_OK', job, sorted(load_registry(ctx.job_dir)), flush=True)


def worker_material(root: str, job: str, phys: str, job_index: int) -> None:
    """One new physical material from aug_materials/<phys>__request.json (written by the controller)."""
    rd._init_openmp_before_torch()
    ctx = rd.open_job(DATASET, job, Path(root), stage='material')
    req = context.read_json(aug_dir(ctx.job_dir) / (phys + '__request.json'))
    try:
        rec = build_material_v2(ctx, req['assignment'], phys, job_index=job_index)
    except rd.ProgramExecutionError as exc:
        context.write_json(aug_dir(ctx.job_dir) / (phys + '__error.json'), {'kind': 'PROGRAM_EXECUTION_REJECTED', 'message': str(exc)[:300]})
        raise
    print('MATERIAL_OK', job, phys, 'build_s %.1f' % context.read_json(rec['summary_path'])['build_seconds'], flush=True)


def worker_fit(root: str, job: str, phys: str, seed: int, tag: str = '') -> None:
    rd._init_openmp_before_torch()
    from methods.ttha.batch_base import train
    import torch
    t0 = time.time()
    ctx = rd.open_job(DATASET, job, Path(root), stage='evaluate')          # rows [t-672, t+96): no C_B / E row in memory
    ref = rd.MaterialRef(**rd.load_index(ctx.job_dir)['P0_Linear'])
    X, y = rd.training_arrays(ctx, ref)
    lg, sc = ctx.legal, ctx.scaler
    reg = load_registry(ctx.job_dir)
    rec_m = reg[phys]
    child = None
    if rec_m['path'] is not None:
        with np.load(rec_m['path']) as z:
            child = (z['Xc'].copy(), z['yc'].copy())
        if child[0].shape != X.shape or child[1].shape != y.shape:
            raise RuntimeError('child does not cover the legal parent set')
    for d in ('runs', 'cells', 'pred_c_a'):
        (ctx.job_dir / d).mkdir(exist_ok=True)
    bidx = train.batch_indices(seed, n_pool=lg.n, n_updates=spec.N_UPDATES)
    model, tres = train.train_arm(X, y, child, bidx, model_seed=seed, timeout_seconds=FIT_TIMEOUT_S - 30, n_updates=spec.N_UPDATES)
    Xs, serve = rd.serve_inputs(ctx.slice, ctx.job.c_a, [[] for _ in ctx.job.roster], sc)
    truth = rd.read_truth(ctx.slice, ctx.job.c_a)
    c = phys_cell_id(job, phys, seed, tag)
    mp = ctx.job_dir / 'runs' / (c + '.pt')
    train.save_model(model, mp)
    pred = rd.predict_inputs(model, Xs, sc)
    np.savez(ctx.job_dir / 'pred_c_a' / (c + '.npz'), pred_raw=pred, origins=np.array(ctx.job.c_a))
    rec = {'cell_id': c, 'status': 'OK', 'profile': PROFILE_VERSION, 'job_id': job, 'dataset': DATASET, 'material_id': phys, 'material_key': rec_m['key'],
           'model_seed': seed, 'batch_seed': spec.batch_seed(seed), 'n_updates': spec.N_UPDATES, 'tag': tag, 'physical_fit': True, 'child_view': child is not None,
           'pool_size': int(lg.n), 'roster': list(ctx.job.roster), 'roster_explicit': False, 'batch_stream': 'train.batch_indices(seed, n_pool, 2000)',
           'model_path': str(mp), 'train': {'seconds': tres.seconds, 'n_updates_done': tres.n_updates_done, 'final_train_loss': tres.final_train_loss,
                                            'first_step_loss': tres.first_step_loss, 'loss_curve': tres.loss_curve},
           'serve_inputs_c_a': serve, 'scores': {'c_a': rd.score_block(pred, truth, sc)}, 'consumer': CONSUMER,
           'device': str(train.device()), 'torch': torch.__version__, 'torch_threads': torch.get_num_threads(), 'python': sys.version.split()[0],
           'rows_read': list(ctx.job.evaluate_rows), 'worker_seconds_total': time.time() - t0}
    context.write_json(ctx.job_dir / 'cells' / (c + '.json'), rec)
    print('CELL_OK', c, round(tres.seconds, 2), flush=True)


def worker_compare(a: str, b: str) -> None:
    import torch
    sa, sb = torch.load(a, map_location='cpu'), torch.load(b, map_location='cpu')
    same = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    print(json.dumps({'identical': bool(same)}))


def branch_cells(job_dir: Path) -> dict:
    out = {}
    for p in sorted((job_dir / 'cells').glob('*.json')) if (job_dir / 'cells').exists() else []:
        r = context.read_json(p)
        if r.get('status') == 'OK' and not r.get('tag'):
            out[r['cell_id']] = r
    return out


def label_prerequisite(stage: str, branch: Path, job: str, stage_root: Path) -> None:
    """Raises before any label row is loaded or torch is imported."""
    jd = Path(branch) / job
    if stage == 'c_b' and not (jd / 'commit.json').exists():
        raise PermissionError('C_B refused: commit.json missing')
    if stage == 'freeze_e' and not (jd / 'c_b_scores.json').exists():
        raise PermissionError('E inputs refused: C_B not scored')
    if stage == 'score_e':
        from methods.ttha.batch_base import commit as _commit
        _commit.require_cohort_frozen(branch, job, stage_root)       # the existing whole-stage E barrier, unchanged
        if not (jd / 'e_frozen.json').exists():
            raise PermissionError('E targets refused: predictions not frozen')


def worker_label(stage: str, branch: str, job: str, stage_root: str) -> None:
    branch, stage_root = Path(branch), Path(stage_root)
    label_prerequisite(stage, branch, job, stage_root)
    rd._init_openmp_before_torch()
    from methods.ttha.batch_base import train
    jd = branch / job
    js = rd.resolve_job(DATASET, job)
    cells = branch_cells(jd)
    if stage in ('c_b', 'freeze_e'):
        ctx = rd.open_job(DATASET, job, branch, stage='c_b' if stage == 'c_b' else 'e_input')
        origins = js.c_b if stage == 'c_b' else js.e
        Xs, serve = rd.serve_inputs(ctx.slice, origins, [[] for _ in js.roster], ctx.scaler)
        if stage == 'c_b':
            truth = rd.read_truth(ctx.slice, origins)
            out = {'job_id': job, 'profile': PROFILE_VERSION, 'opened_local': now(), 'rows_read': list(js.c_b_rows), 'serve': serve, 'cells': {}}
            for c, r in cells.items():
                s = rd.score_block(rd.predict_inputs(train.load_model(r['model_path']), Xs, ctx.scaler), truth, ctx.scaler)
                out['cells'][c] = {'material_id': r['material_id'], 'model_seed': r['model_seed'], 'c_b': s}
                r['scores']['c_b'] = s
                context.write_json(jd / 'cells' / (c + '.json'), r)
            context.write_json(jd / 'c_b_scores.json', out)
        else:
            (jd / 'predictions_e').mkdir(exist_ok=True)
            out = {'job_id': job, 'profile': PROFILE_VERSION, 'started_local': now(), 'rows_read': list(js.e_input_rows), 'serve': serve, 'models': {}}
            for c, r in cells.items():
                pred = rd.predict_inputs(train.load_model(r['model_path']), Xs, ctx.scaler)
                p = jd / 'predictions_e' / (c + '.npz')
                np.savez(p, pred_raw=pred, origins=np.array(js.e))
                out['models'][c] = {'path': str(p), 'material_id': r['material_id'], 'model_seed': r['model_seed']}
            out['frozen_at_local'] = out['frozen_local'] = now()
            context.write_json(jd / 'e_frozen.json', out)
    elif stage == 'score_e':
        ctx = rd.open_job(DATASET, job, branch, stage='e_target')
        fz = context.read_json(jd / 'e_frozen.json')
        truth = rd.read_truth(ctx.slice, js.e)
        out = {'job_id': job, 'profile': PROFILE_VERSION, 'scored_local': now(), 'e_frozen_local': fz['frozen_local'], 'rows_read': list(js.e_target_rows), 'cells': {}}
        for c, m in fz['models'].items():
            with np.load(m['path']) as z:
                if not np.array_equal(z['origins'], np.array(js.e)):
                    raise RuntimeError('frozen E origins differ')
                pred = z['pred_raw'].copy()
            out['cells'][c] = {'material_id': m['material_id'], 'model_seed': m['model_seed'], 'e': rd.score_block(pred, truth, ctx.scaler)}
        context.write_json(jd / 'e_scores.json', out)
    else:
        raise ValueError(stage)
    print('LABEL_OK', stage, branch.name, job, flush=True)


# ============================================================================= controller-side subprocess plumbing (never imports torch)
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
            p = subprocess.run([sys.executable, '-B', '-m', MODULE] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout, env=worker_env())
        return p.returncode, time.time() - t0
    except subprocess.TimeoutExpired:
        return -999, time.time() - t0


def compare_models(a: str, b: str) -> bool:
    rc = subprocess.run([sys.executable, '-B', '-m', MODULE, '--worker-compare', a, b], capture_output=True, text=True, cwd=str(REPO), env=worker_env())
    return bool(json.loads(rc.stdout.strip().splitlines()[-1])['identical'])


def fit_physical(ledger, root: Path, job: str, phys: str, seed: int, *, tag: str = '') -> dict:
    """Serial, reserved-before-launch physical fit into the job's cache; one same-configuration retry when FIT_RETRY and the ledger allows."""
    cp = Path(root) / job / 'cells' / (phys_cell_id(job, phys, seed, tag) + '.json')
    if cp.exists():
        rec = context.read_json(cp)
        if rec.get('status') == 'OK':
            return rec
    cid = phys_cell_id(job, phys, seed, tag)
    for attempt in (0, 1):
        ledger.reserve_fit(cid)
        timeout = min(FIT_TIMEOUT_S, ledger.remaining())
        t0 = time.time()
        rc, secs = run_sub(['--worker-fit', root, job, phys, seed] + (['--tag', tag] if tag else []),
                           Path(root) / job / 'fit_logs' / (cid + ('.log' if attempt == 0 else '.retry.log')), timeout)
        ok = rc == 0 and cp.exists() and context.read_json(cp).get('status') == 'OK'
        reason = '' if ok else ('worker_timeout' if rc == -999 else 'worker_failed')
        ledger.finish_fit(cid, ok, time.time() - t0, reason)
        print('FIT', cid, 'OK' if ok else reason, round(secs, 1), flush=True)
        if ok:
            break
        if attempt == 0 and rt.FIT_RETRY and ledger.can_retry():
            ledger.s['retries_used'] += 1
            ledger.s['events'].append({'kind': 'fit_retry', 'cell': cid, 'reason': reason, 'epoch': time.time()})
            ledger._save()
            continue
        raise RuntimeError(reason)
    rec = context.read_json(cp)
    if rec['scores']['c_a']['n_nonfinite_predictions']:
        raise RuntimeError('nonfinite predictions')
    return rec


def ensure_material(root: Path, job: str, assignment: list, *, job_index: int) -> dict:
    """The physical record of an assignment in the job cache: None / existing by key / built now in a worker (serial)."""
    reg = load_registry(Path(root) / job)
    key = material_key(assignment)
    for r in reg.values():
        if r['key'] == key:
            return r
    used = [int(r['material_index']) for r in reg.values() if r.get('material_index')]
    phys = 'TA%03d' % (max(used + [0]) + 1)
    context.write_json(aug_dir(Path(root) / job) / (phys + '__request.json'), {'assignment': assignment, 'requested_local': now()})
    rc, secs = run_sub(['--worker-material', root, job, phys, job_index], Path(root) / job / 'fit_logs' / ('material_%s.log' % phys), 900)
    err = aug_dir(Path(root) / job) / (phys + '__error.json')
    if rc != 0:
        if err.exists():
            raise br.ToolInputError('plan rejected at execution (no fit, no material): %s' % context.read_json(err)['message'], execution_status='PROGRAM_EXECUTION_REJECTED')
        raise RuntimeError('material worker failed for %s %s' % (job, phys))
    return load_registry(Path(root) / job)[phys]


# ============================================================================= adapter (BatchAdapter protocol + resume hooks)
CONTRACTS = {
    'overview': {'arguments': {}, 'meaning': 'All N entity rows of T-only, missing-aware observations (formulas in field_definitions), batch summaries, the Consumer (parent 0.5 + child 0.5 training) and the augmentation action table: 7 TempoPFN-sourced primitives, the P_NoMixRecipe preset, composition rules, window and randomness semantics.'},
    'inspect_data': {'arguments': {'entity_indices': '[1..8 integers in 0..N-1]', 'kind': 'gaps|segment|daily_means|hour_profile', 'sub_range': 'optional [absolute_start, absolute_end): ABSOLUTE row indices inside overview.train_rows (not 0-based offsets); segment at most 336 rows (default: last 168 T rows)'},
                     'meaning': 'Raw current-T views with missing positions kept (null), normalized by the frozen T scaler, with absolute row indices.'},
    'build_material': {'arguments': {'plan_id': 'new short alphanumeric/underscore id',
                                     'policy': {'default': {'steps': [{'op': '<primitive name from overview.actions.primitives, or P_NoMixRecipe alone>'}]},
                                                'rules': [{'when': {'feature': '<overview field>', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': []}],
                                                'rationale': '<current hypothesis; no data source or job name>', 'observation_fields_used': []}},
                       'meaning': 'Construct one complete whole-batch augmentation plan and freeze its child view; does not train. Up to 3 primitives per program, no repeats, regime/shock and calendar/amplitude mutually exclusive, executed in the fixed order regardless of listing; P_NoMixRecipe is a complete program on its own. Empty steps = no augmentation for that entity; an all-empty plan is None. At most 8 rules; first matching rule wins, a null field never fires a rule; thresholds are numbers or a current-batch quantile. No seed, strength or distribution argument exists; the child view of a program on an entity is frozen per (entity, program), so plans with identical complete assignments are the same material and are rejected as duplicates of the existing id (no new material, fit or slot).'},
    'inspect_material': {'arguments': {'plan_id': '<built id>', 'entity_index': 'optional integer: also return one parent/child window', 'window': "optional 'largest_change' (default) | 'first' | 'last'"},
                         'meaning': 'What the plan changed: executed steps, identity (no-op) counts, change fraction and RMS for inputs and targets separately. Not predictive utility.'},
    'evaluate': {'arguments': {'plan_id': '<built id>'}, 'meaning': 'Train the complete plan (parent 0.5 + child 0.5, shared MLP, 2000 updates) under the 3 frozen paired seeds and return C_A only (missing-aware loss on observed future cells, Baseline-Linear serving inputs). At most 4 new complete plans per job; the four public references are already evaluated.'},
    'compare': {'arguments': {'a': '<evaluated id>', 'b': '<evaluated id>'}, 'meaning': 'Positive paired delta means b has lower C_A loss; per seed, origin and entity; seed uncertainty, not future certainty.'},
    'commit': {'arguments': {'plan_id': '<evaluated id>', 'reason': '<brief evidence-based decision>'}, 'meaning': 'Final action. Any evaluated plan including None, FixedMixup, P_AmpResample or P_NoMixRecipe. No automatic argmin, no later selection override.'},
}

FAST_SYSTEM = '''You are the Fast path of a batch training-data augmentation research Harness. Your unit is one entire batch of N entities (see overview) and one shared MLP Consumer, not separate per-entity models.
你的任务是为当前整批数据构造一份完整的训练数据增强方案（对每个合法训练窗口生成一个子视图，与不变的父视图共同训练），使固定共享 Consumer 对原始观测未来的预测更好。阅读全批现象与原语语义，用实际工具证据决定是否增强、用哪些原语、是否分组，组织预算内的比较，允许直接保留任一公共参照。子视图只改训练材料：预测输入和评分真值从不增强。改动更大、步骤更多、实体分组更细都不是目标。只使用 T 和本轨迹的 C_A。单实体预测变化不是该实体材料的独立因果贡献。输出真实工具动作，不输出伪执行的说明文字。
(Task: construct one complete whole-batch augmentation plan - a child view of every legal training window trained next to the unchanged parent view - so that the fixed shared Consumer predicts the originally observed future better. Read the batch-wide phenomena and the primitive semantics; decide from actual tool evidence whether to augment, with which primitives and whether to group entities; organize comparisons within budget; keeping any public reference directly is allowed. The child view changes training material only: serving inputs and scoring targets are never augmented. Larger changes, more steps or finer grouping are not goals. Use only T and this trajectory's C_A. A change in one entity's prediction is not the independent causal contribution of that entity's material. Output real tool actions, not text that pretends execution.)
None, FixedMixup, P_AmpResample and P_NoMixRecipe are fully evaluated public references; you may commit any of them without spending more fits. No C_B or E is available; do not invent future results. Uniform and heterogeneous plans are both legal. Guidance is advice and does not prohibit public tools. Return exactly one JSON object {"actions":[{"tool":"listed_name","arguments":{...}}]}, no markdown. Batched actions execute in order and must use valid IDs. Commit must be last. State brief hypotheses in material rationale or commit reason, not hidden chain-of-thought. Do not include data source or job identifiers in rationale. Do not change seeds, model, training budget, scaler, legal windows, targets, scoring or population.'''

SLOW_SYSTEM = '''You are the offline Slow of a batch training-data augmentation research Harness. You receive a deterministic census of the completed or explicitly failed Source branches of ONE neutral domain id: T-only missing-aware observations, the actual Fast trajectories (tool calls and returned results), complete augmentation plans, material diagnostics, C_A feedback, commits, post-commit delayed C_B, costs and failures. No E values exist in this input, and no Select or Target data. The branches ran the same no-Skill Fast start on different earlier batches of this domain; they are not repeats. Goal of a Workflow: improve the whole-batch prediction gain and the research cost of Fast on NEW batches of the same domain. Propose at most TWO reusable domain Workflows, or KEEP if the census does not support a reusable Workflow. A Workflow explains how to observe the batch, which hypotheses and complete augmentation plans are worth constructing, how to organize comparisons and use C_A feedback within the fixed budget (4 new complete plans, 16 calls, 24 tool calls), and when to stop or keep a public reference. Two candidates may follow different investigation orders or experiment allocations. A Workflow may state evidence-bounded primitive preferences and may recommend a simple uniform strategy or a public reference; neither candidate needs to be complex, diverse or aggressive. Neutral fact: rankings on one evaluation time block are not guaranteed to hold on later blocks. Distinguish observed facts from hypotheses; an undistinguished comparison is not evidence that a primitive is generally worse; an untested action is not harmful; a per-entity prediction difference is not the causal contribution of that entity's material. Do not change model, scoring, tools, permissions, budgets, randomness, targets or legal windows. Do not write data source names, job labels, entity ids, row numbers or a historical assignment as the answer. Output exact JSON, no markdown: {"decision":"KEEP","rationale":"..."} OR {"decision":"PROPOSE","candidates":[{"candidate_id":"W1","research_mode":"<short label>","workflow":"...","principles":null,"observable_applicability":{"const":true},"applicability_summary":"<=400 characters","evidence_refs":["copied from legal_evidence_refs"],"rationale":"..."}]}. In this package principles MUST be null and observable_applicability MUST be exactly {"const":true} (express any applicability condition inside the Workflow text as observable conditions). The Workflow renders to at most 1200 characters. KEEP is valid and will not be resampled; every candidate is CANDIDATE_TEST_ONLY until real selection.'''


def skill_text_check(text: str, where: str) -> None:
    rd.validate_text(text, where, allow_entity_ids=False)


class WorkflowAdapter:
    """BatchAdapter over the readiness T view + tempo_aug v2 materials. Physical cache = common/<job>/ (materials by assignment key,
    fits by physical id); this branch sees only its own plan ids, its copies of the cells and the public references."""

    def __init__(self, run_dir, job, ledger, repo, *, common, job_index, public_ids, tool_error_feedback=True, roster=None, seeds=SEEDS, dataset=DATASET, fit_fn=None):
        if dataset != DATASET or roster is not None:
            raise ValueError('this profile is bound to %s with its frozen roster' % DATASET)
        self.root, self.common, self.job = Path(run_dir), Path(common), job
        self.job_index, self.public_ids = int(job_index), tuple(public_ids)
        self.ledger, self.repo, self.seeds = ledger, repo, tuple(seeds)
        self.fit_fn = fit_fn or (lambda phys, seed: fit_physical(self.ledger, self.common, self.job, phys, seed))
        self.tool_error_feedback = tool_error_feedback
        init_branch_dir(self.common, self.root, job, self.public_ids)
        self.ctx = rd.open_job(DATASET, job, self.root, 'material')
        self.n = len(self.ctx.job.roster)
        self.specs, self.fitted = {}, {}

    # --- helpers
    def _reject(self, message, **details):
        if self.tool_error_feedback:
            raise br.ToolInputError(message, **details)
        raise ValueError(message)

    def _reg(self) -> dict:
        return load_registry(self.ctx.job_dir)

    def _spec_of(self, mid: str) -> dict:
        if mid in self.specs:
            return self.specs[mid]
        p = aug_dir(self.ctx.job_dir) / (mid + '__compiled.json')
        self.specs[mid] = context.read_json(p)['material_spec'] if p.exists() else public_spec(mid)
        return self.specs[mid]

    # --- read tools
    def overview(self):
        ov = br.json_copy(self.ctx.overview())
        ov['batch_features'] = {'batch_median_' + k: v['median'] for k, v in ov['summary'].items() if v is not None}
        ov['batch_feature_definition'] = 'Median across all N T-only entity values of the field; a field with no computable value is omitted.'
        ov['actions'] = ta.action_table()                              # the augmentation profile replaces the readiness repair table
        ov['actions']['public_references'] = {'None': 'no augmentation (parent view only)', 'FixedMixup': FIXED_MIXUP['definition'] + '; not a TempoPFN primitive',
                                              'P_AmpResample': 'uniform tp_amplitude -> tp_resample', 'P_NoMixRecipe': 'uniform ' + ta.RECIPE}
        ov['consumer'] = CONSUMER
        return ov

    def inspect_data(self, arguments, *, remaining_seconds):
        if set(arguments) - {'entity_indices', 'kind', 'sub_range'}:
            self._reject('unknown inspection argument')
        idx = arguments.get('entity_indices')
        if not isinstance(idx, list) or not 1 <= len(idx) <= rd.MAX_INSPECT_ENTITIES or any(type(i) != int or not 0 <= i < self.n for i in idx):
            self._reject('entity_indices must be 1..%d integers in 0..%d' % (rd.MAX_INSPECT_ENTITIES, self.n - 1))
        kind = arguments.get('kind', 'gaps')
        if kind not in rd.INSPECT_KINDS:
            self._reject('kind must be one of %s' % list(rd.INSPECT_KINDS))
        span = arguments.get('sub_range')
        if span is not None and (not isinstance(span, list) or len(span) != 2 or any(type(x) != int for x in span) or span[0] >= span[1]):
            self._reject('sub_range must be two increasing integer boundaries')
        lo, hi = self.ctx.job.train_range
        if span is not None and (span[0] < lo or span[1] > hi):
            self._reject('sub_range must use absolute row indices inside T=[%d,%d) (overview.train_rows), not 0-based offsets' % (lo, hi))
        if kind == 'segment' and span is not None and span[1] - span[0] > rd.MAX_SEGMENT_ROWS:
            self._reject('segment is limited to %d rows' % rd.MAX_SEGMENT_ROWS)
        return self.ctx.inspect_data(idx, kind, span)

    def build_material(self, arguments, *, remaining_seconds):
        if set(arguments) != {'plan_id', 'policy'}:
            self._reject('build needs plan_id and policy')
        mid = arguments['plan_id']
        if not isinstance(mid, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}', mid) or mid in PUBLIC or mid.startswith('TA'):
            self._reject('unsafe or reserved plan id')
        reg = self._reg()
        if mid in reg:
            self._reject('plan id already exists; use it or a new id')
        try:
            compiled = ta.compile_plan(arguments['policy'], self.ctx.overview()['entities'])
        except rd.PolicyError as exc:
            self._reject(str(exc), legal_ops=list(ta.PRIMITIVES) + [ta.RECIPE], valid_observation_fields=list(rd.FIELDS))
        key = material_key(compiled['assignment'])
        same = next((m for m, r in reg.items() if r.get('key') == key), None)
        if same is not None:
            self._reject('identical complete assignment to existing plan %s (same material): reuse that id; no new material, fit or slot' % same, alias_of=same,
                         execution_status='DUPLICATE_ASSIGNMENT')
        phys = ensure_material(self.common, self.job, compiled['assignment'], job_index=self.job_index)
        rec = {**phys, 'material_id': mid, 'phys_id': phys['material_id'], 'compiled': compiled}
        reg = self._reg()
        reg[mid] = rec
        save_registry(self.ctx.job_dir, reg)
        programs, idx = [], []
        for steps in compiled['assignment']:
            if steps not in programs:
                programs.append(steps)
            idx.append(programs.index(steps))
        ms = {'profile': PROFILE_VERSION, 'policy': compiled['policy'], 'programs': programs, 'entity_program': idx, 'rule_index': compiled['rule_index'],
              'resolved_thresholds': compiled['resolved_thresholds'], 'n_unknown': compiled['n_unknown'], 'alias_of': None,
              'execution_order': 'fixed: ' + ' -> '.join(ta.PRIMITIVES)}
        self.specs[mid] = ms
        context.write_json(aug_dir(self.ctx.job_dir) / (mid + '__compiled.json'), {'material_spec': ms, 'assignment': compiled['assignment'], 'key': key, 'phys_id': phys['material_id']})
        return br.Candidate(mid, ms, self.job)

    def inspect_material(self, plan_id, arguments, *, remaining_seconds):
        if 'plan_id' not in arguments or set(arguments) - {'plan_id', 'entity_index', 'window'}:
            self._reject('inspect_material takes plan_id and optional entity_index / window')
        ei, w = arguments.get('entity_index'), arguments.get('window', 'largest_change')
        if ei is not None and (type(ei) != int or not 0 <= ei < self.n):
            self._reject('entity_index must be an integer in 0..%d' % (self.n - 1))
        if w not in ('largest_change', 'first', 'last'):
            self._reject("window must be 'largest_change', 'first' or 'last'")
        rec = self._reg()[plan_id]
        if rec.get('kind') == 'none':
            return {'material_id': plan_id, 'note': 'no augmentation: the parent view only, no child view'}
        if rec.get('kind') == 'fixed_mixup':
            return {'material_id': plan_id, 'note': 'historical public reference: ' + FIXED_MIXUP['definition'] + '; no TempoPFN primitive', 'change_rms': {k: v for k, v in context.read_json(rec['summary_path']).items() if k.startswith('rms')}}
        return ta.inspect(self.ctx, plan_id, ei, w)

    # --- training
    def evaluate(self, candidate, seeds, *, feedback, remaining_seconds):
        rec = self._reg()[candidate.plan_id]
        phys = rec.get('phys_id', rec['material_id'])
        recs = []
        for seed in seeds:
            bc = self.ctx.job_dir / 'cells' / (phys_cell_id(self.job, candidate.plan_id, seed) + '.json')
            if bc.exists():
                recs.append(context.read_json(bc))
                continue
            before = self.ledger.s['fit_attempts']
            pr = self.fit_fn(phys, seed)
            if self.ledger.s['fit_attempts'] == before:
                self.ledger.note_cache(phys_cell_id(self.job, candidate.plan_id, seed))
            if pr['job_id'] != self.job or pr['model_seed'] != seed or pr.get('material_key') != rec['key'] or not Path(pr['model_path']).exists():
                raise RuntimeError('physical cell binding failed')
            mine = {**pr, 'cell_id': phys_cell_id(self.job, candidate.plan_id, seed), 'material_id': candidate.plan_id, 'physical_cell': pr['cell_id'], 'physical_material': phys}
            bc.parent.mkdir(exist_ok=True)
            context.write_json(bc, mine)
            recs.append(mine)
        if any(r['scores']['c_a']['status'] != 'SCORABLE' for r in recs):
            raise RuntimeError('C_A_NOT_SCORABLE')
        losses = tuple(tuple(tuple(row) for row in r['scores']['c_a']['per_origin_entity_normalized_mse']) for r in recs) if feedback else ()
        fitted = replace(candidate, model_seeds=tuple(seeds), model_refs=tuple(r['model_path'] for r in recs), ca_losses=losses)
        self.fitted[candidate.plan_id] = fitted
        return fitted

    def restore(self, plan_ids):
        out = []
        for mid in plan_ids:
            before = self.ledger.s['fit_attempts']
            out.append(self.evaluate(br.Candidate(mid, self._spec_of(mid), self.job), self.seeds, feedback=True, remaining_seconds=self.ledger.remaining()))
            if self.ledger.s['fit_attempts'] != before:
                raise RuntimeError('restore must not fit')
        return out

    def restore_built(self, mid):
        return br.Candidate(mid, self._spec_of(mid), self.job)

    def baselines(self):
        return tuple(self.evaluate(br.Candidate(mid, public_spec(mid), self.job), self.seeds, feedback=True, remaining_seconds=self.ledger.remaining()) for mid in self.public_ids)


def public_spec(mid: str) -> dict:
    if mid == 'FixedMixup':
        return {'kind': 'public_reference', 'augmentation': 'timemixup donor rule R, w=0.25 (pre-existing augment.py; not TempoPFN)'}
    steps = PUBLIC_STEPS[mid]
    return {'kind': 'public_reference', 'profile': PROFILE_VERSION, 'policy': uniform_policy(steps, 'public reference %s' % mid), 'programs': [steps],
            'entity_program': [0] * n_of(''), 'alias_of': None, 'execution_order': 'fixed: ' + ' -> '.join(ta.PRIMITIVES)}


def init_branch_dir(common: Path, branch_root: Path, job: str, public_ids) -> None:
    """A branch view of the job cache: static T objects copied, the public references registered; no other material, cell or trajectory."""
    src, dst = Path(common) / job, Path(branch_root) / job
    if (dst / 'aug_materials' / 'index.json').exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for name in ('scaler.npz', 'overview.json'):
        if not (dst / name).exists():
            shutil.copy2(src / name, dst / name)
    if not (dst / 'materials').exists():
        shutil.copytree(src / 'materials', dst / 'materials')
    reg = load_registry(src)
    save_registry(dst, {m: {**reg[m], 'phys_id': m} for m in public_ids})


def commit_branch(root, dataset: str, job: str, material_id: str, reason: str, delivery_seed: int, extra: dict | None = None) -> dict:
    jd = Path(root) / job
    p = jd / 'commit.json'
    if p.exists():
        raise RuntimeError('job %s already committed in this branch' % job)
    cells = branch_cells(jd)
    cid = phys_cell_id(job, material_id, delivery_seed)
    if cid not in cells:
        raise ValueError('commit refused: %s has no OK cell for seed %d in this branch' % (material_id, delivery_seed))
    rec = {'job_id': job, 'profile': PROFILE_VERSION, 'material_id': material_id, 'physical_material': cells[cid].get('physical_material', material_id),
           'material_key': cells[cid].get('material_key'), 'delivery_seed': delivery_seed, 'delivery_cell': cid, 'reason': str(reason)[:1000],
           'committed_at_local': now(), 'fitted_materials_at_commit': sorted({c['material_id'] for c in cells.values()}), 'extra': extra or {}}
    context.write_json(p, rec)
    return rec


# ============================================================================= metered Fast client with the per-branch token cap
class FastClient:
    def __init__(self, client, stage_root: Path, system: str | None = None):
        # system: a follow-up study's frozen Fast system prompt (carried in its stage config); None = this package's FAST_SYSTEM.
        self.client, self.spent, self.system = client, {}, system or FAST_SYSTEM
        for unit, row in dsks._unit_tokens(stage_root).items():
            if unit.startswith('fast:'):
                self.spent[unit[5:]] = row['prompt_tokens'] + row['completion_tokens']

    def fast(self, unit):
        def call(payload):
            if self.spent.get(unit, 0) >= FAST_TOKEN_CAP:
                raise budget.BudgetExhausted('per-branch token cap %d reached' % FAST_TOKEN_CAP)
            if not proxy_reachable():          # free TCP pre-check: no request number, no HTTP attempt, no unknown usage; the never-sent call is resumable
                raise rt.llm.TransportFault('PROXY_UNREACHABLE_PRECHECK')
            led = self.client.ledger
            before = led.s['llm_tokens_in'] + led.s['llm_tokens_out']
            try:
                return self.client.call('fast', unit, payload, self.system, max_tokens=MAX_OUTPUT_TOKENS)
            finally:
                self.spent[unit] = self.spent.get(unit, 0) + (led.s['llm_tokens_in'] + led.s['llm_tokens_out'] - before)
        return call


# ============================================================================= arms
def paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'configs': root / 'stage_configs', 'logs': root / 'logs',
            'wiring': root / 'wiring', 'source': root / 'source', 'formation': root / 'formation', 'select': root / 'select', 'target': root / 'target'}


def knowledge_from_json(d: dict) -> br.Knowledge:
    return br.Knowledge(int(d['version']), tuple(br.Guidance(e['hook'], e['body'], copy.deepcopy(e['applicability']), tuple(e['evidence_refs'])) for e in d['entries']))


def adapter_factory(common: Path, job_index: int, public_ids, seeds):
    def make(root, job, led, repo, *, tool_error_feedback=True, roster=None, seeds=seeds, dataset=DATASET):
        return WorkflowAdapter(root, job, led, repo, common=common, job_index=job_index, public_ids=public_ids, tool_error_feedback=tool_error_feedback,
                               roster=roster, seeds=seeds, dataset=dataset)
    return make


def fast_branch(path, job, knowledge, led, client, common, *, job_index, public_ids, seeds):
    return base.branch(path, job, knowledge, led, client, None, evidence_roundtrip=EVIDENCE_ROUNDTRIP, max_tool_corrections=MAX_TOOL_CORRECTIONS,
                       tool_contracts=CONTRACTS, seeds=seeds, adapter_factory=adapter_factory(common, job_index, public_ids, seeds), limits=LIMITS, dataset=DATASET,
                       commit_fn=commit_branch, allowed_features=ALLOWED_FEATURES, entity_count=n_of(job))


def resume_fast_branch(path, job, knowledge, led, client, common, *, job_index, public_ids, seeds):
    return base.resume_branch(path, job, knowledge, led, client, max_tool_corrections=MAX_TOOL_CORRECTIONS, tool_contracts=CONTRACTS, seeds=seeds,
                              adapter_factory=adapter_factory(common, job_index, public_ids, seeds), evidence_roundtrip=EVIDENCE_ROUNDTRIP, limits=LIMITS,
                              dataset=DATASET, commit_fn=commit_branch, allowed_features=ALLOWED_FEATURES, entity_count=n_of(job))


def draw_random_supply(job: str, job_index: int, table: list) -> dict:
    """RandomSearch_B4's frozen distribution (task §6): two uniform plans from the 53 programs minus the public assignments, then two
    conditional plans (field ~ U(FIELDS), quantile ~ U{.25,.5,.75}, '>=', default / hit programs two distinct of 53); both groups must hold
    an entity on T; complete assignments deduplicated against the public references and this arm's earlier plans; 256 tries per conditional
    slot, then a fallback uniform program from the same stream. Drawn before the job's first fit; never shown to Fast."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([RANDOM_SEED_ROOT, int(job_index)])))
    n = len(table)
    public_keys = {material_key(ta.compile_plan(public_policy(m), table)['assignment']) for m in PUBLIC_STEPS}
    taken_keys, drawn_uniform, plans = set(public_keys), set(), []
    eligible = [i for i in range(len(PROGRAMS)) if material_key([PROGRAMS[i]] * n) not in public_keys]

    def draw_uniform():
        pool = [i for i in eligible if i not in drawn_uniform]
        i = int(pool[int(rng.integers(len(pool)))])
        drawn_uniform.add(i)
        return i

    for k in (1, 2):
        i = draw_uniform()
        pol = uniform_policy(PROGRAMS[i], 'random uniform program %d (control arm)' % i)
        comp = ta.compile_plan(pol, table)
        taken_keys.add(material_key(comp['assignment']))
        plans.append({'slot': 'R%d' % k, 'kind': 'uniform', 'program_index': i, 'policy': pol, 'fallback': False, 'tries': 1})
    for k in (3, 4):
        found = None
        for t in range(1, RANDOM_MAX_TRIES + 1):
            f = rd.FIELDS[int(rng.integers(len(rd.FIELDS)))]
            q = RANDOM_QUANTILES[int(rng.integers(len(RANDOM_QUANTILES)))]
            a, b = [int(x) for x in rng.choice(len(PROGRAMS), size=2, replace=False)]
            pol = {'default': {'steps': PROGRAMS[a]}, 'rules': [{'when': {'feature': f, 'op': '>=', 'value': {'quantile': q}}, 'steps': PROGRAMS[b]}],
                   'rationale': 'random conditional plan (control arm)', 'observation_fields_used': [f]}
            comp = ta.compile_plan(pol, table)
            key = material_key(comp['assignment'])
            groups = set(comp['rule_index'])
            if groups == {-1, 0} and key not in taken_keys:
                found = {'slot': 'R%d' % k, 'kind': 'conditional', 'field': f, 'quantile': q, 'default_program_index': a, 'hit_program_index': b, 'policy': pol,
                         'fallback': False, 'tries': t, 'entities_hit': int(sum(x == 0 for x in comp['rule_index'])), 'n_unknown': comp['n_unknown']}
                taken_keys.add(key)
                break
        if found is None:
            i = draw_uniform()
            pol = uniform_policy(PROGRAMS[i], 'random uniform program %d (conditional slot fallback)' % i)
            taken_keys.add(material_key(ta.compile_plan(pol, table)['assignment']))
            found = {'slot': 'R%d' % k, 'kind': 'uniform', 'program_index': i, 'policy': pol, 'fallback': True, 'tries': RANDOM_MAX_TRIES}
        plans.append(found)
    return {'job': job, 'job_index': job_index, 'rng': 'Generator(PCG64(SeedSequence([%d, %d])))' % (RANDOM_SEED_ROOT, job_index), 'drawn_local': now(),
            'plans': plans, 'hidden_from_fast': True}


def random_branch(root: Path, job: str, led, common: Path, supply: dict, *, job_index, public_ids, seeds) -> br.RunResult:
    """RandomSearch_B4: 0 LLM, four frozen plans evaluated (4 logical slots), delivery = argmin three-seed C_A mean over public + random;
    ties in public order then draw order."""
    root.mkdir(parents=True, exist_ok=False)
    ad = WorkflowAdapter(root, job, led, REPO, common=common, job_index=job_index, public_ids=public_ids, seeds=seeds)
    n = ad.n
    trace = []

    def emit(event, **kw):
        row = br.json_copy({'event_id': '%s:%d' % (job, len(trace)), 'event': event, **kw})
        trace.append(row)
        base.event_sink(root / 'trace.jsonl')(row)
    context.write_json(root / 'knowledge.json', asdict(br.Knowledge()))
    cands = list(ad.baselines())
    emit('job_started', mode='random_search_b4', baseline_ids=[c.plan_id for c in cands], seeds=list(seeds), supply_rng=supply['rng'])
    for pl in supply['plans']:
        c = ad.build_material({'plan_id': pl['slot'], 'policy': pl['policy']}, remaining_seconds=led.remaining())
        emit('random_material', plan_id=c.plan_id, material_spec=c.material_spec, draw={k: v for k, v in pl.items() if k != 'policy'})
        c = ad.evaluate(c, seeds, feedback=True, remaining_seconds=led.remaining())
        cands.append(c)
        emit('random_evaluated', plan_id=c.plan_id, feedback=c.feedback(seeds, 2, n))
    order = [c.plan_id for c in cands]
    chosen = min(cands, key=lambda c: (c.feedback(seeds, 2, n)['mean_loss'], order.index(c.plan_id)))
    reason = 'RandomSearch_B4: minimum three-seed C_A mean among %s; ties in that order.' % ', '.join(order)
    commit_branch(root, DATASET, job, chosen.plan_id, reason, seeds[0])
    emit('committed', plan_id=chosen.plan_id, delivery_model_ref=chosen.model_refs[0], reason=reason)
    result = br.RunResult('COMPLETE', job, 0, chosen.plan_id, chosen.model_refs[0], 0, 0, len(supply['plans']), trace)
    context.write_json(root / 'branch_result.json', asdict(result))
    print('RANDOM', job, 'commit', chosen.plan_id, flush=True)
    return result


def menu_branch(root: Path, job: str, led, common: Path, *, job_index, public_ids, seeds) -> br.RunResult:
    """Menu_CA: 0 LLM, 0 extra fits; argmin three-seed C_A mean over the four public references, ties in public order."""
    root.mkdir(parents=True, exist_ok=False)
    ad = WorkflowAdapter(root, job, led, REPO, common=common, job_index=job_index, public_ids=public_ids, seeds=seeds)
    before = led.s['fit_attempts']
    cands = list(ad.baselines())
    if led.s['fit_attempts'] != before:
        raise RuntimeError('Menu_CA must not fit')
    order = [c.plan_id for c in cands]
    fb = {c.plan_id: c.feedback(seeds, 2, ad.n) for c in cands}
    chosen = min(cands, key=lambda c: (fb[c.plan_id]['mean_loss'], order.index(c.plan_id)))
    reason = 'Menu_CA: minimum three-seed C_A mean among %s; ties in that order.' % ', '.join(order)
    trace = [br.json_copy({'event_id': '%s:0' % job, 'event': 'menu_decision', 'candidates': {m: fb[m]['loss_by_seed'] for m in order}, 'chosen': chosen.plan_id})]
    base.event_sink(root / 'trace.jsonl')(trace[0])
    context.write_json(root / 'knowledge.json', asdict(br.Knowledge()))
    commit_branch(root, DATASET, job, chosen.plan_id, reason, seeds[0])
    result = br.RunResult('COMPLETE', job, 0, chosen.plan_id, chosen.model_refs[0], 0, 0, 0, trace)
    context.write_json(root / 'branch_result.json', asdict(result))
    print('MENU', job, 'commit', chosen.plan_id, flush=True)
    return result


# ============================================================================= budget (one package ledger, cumulative stage caps, per-stage retries)
def stage_caps(root: Path, stage: str) -> dict:
    P_ = paths(root)
    rec = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    if stage in rec:
        return rec[stage]
    led = rt.RuntimeLedger(P_['ledger'], **TOTAL)
    s, alloc = led.s, ALLOC[stage]
    snap = {k: s.get(k, 0) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts',
                                     'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'fit_wall_seconds')}
    snap['elapsed_s'] = time.time() - s['started_epoch']
    retries_stage = min(STAGE_RETRIES, TOTAL['max_retries'] - s['retries_used'])
    caps = {'max_fit_attempts': min(TOTAL['max_fit_attempts'], s['fit_attempts'] + alloc['fits'] + retries_stage),
            'max_llm_requests': min(TOTAL['max_llm_requests'], s['llm_requests'] + alloc['requests']),
            'max_llm_tokens': min(TOTAL['max_llm_tokens'], s['llm_tokens_in'] + s['llm_tokens_out'] + alloc['tokens']),
            'max_wall_s': TOTAL['max_wall_s'], 'max_retries': min(TOTAL['max_retries'], s['retries_used'] + retries_stage)}
    rec[stage] = {'epoch_start': time.time(), 'allocation': alloc, 'snapshot_at_start': snap, 'caps': caps,
                  'http_cap': min(HTTP_CAP, s['llm_http_attempts'] + alloc['requests'] + 4)}
    context.write_json(P_['stages'], rec)
    return rec[stage]


def numeric_seconds(root: Path) -> float:
    led = context.read_json(paths(root)['ledger'])
    return float(led.get('fit_wall_seconds', 0.0)) + float(led.get('label_wall_seconds', 0.0)) + float(led.get('material_wall_seconds', 0.0))


# ============================================================================= stage worker (subprocess; labels withheld on any stop)
def label_stage(branch_dir: Path, job: str, name: str, led, stage_root: Path) -> None:
    t0 = time.time()
    rc, secs = run_sub(['--worker-label', name, branch_dir, job, stage_root], stage_root / 'logs' / ('label_%s_%s.log' % (name, branch_dir.name)), max(1.0, led.remaining()))
    led.s['label_wall_seconds'] = led.s.get('label_wall_seconds', 0.0) + (time.time() - t0)
    led._save()
    if rc:
        raise RuntimeError('label stage failed: %s %s' % (name, branch_dir.name))


def _safe_message(exc) -> str | None:
    return None if isinstance(exc, (rt.llm.AccountFault, rt.llm.TransportFault)) else str(exc)[:300]


def open_labels(root: Path, cfg: dict, branches, failures, led, *, withhold_reason=None, resumed=False, runner=label_stage) -> None:
    if withhold_reason:
        context.write_json(root / 'labels_withheld.json', {'epoch': time.time(), 'reason': withhold_reason, 'branches': [(str(p), j) for p, j in branches],
                                                          'unknown_usage': led.s['llm_tokens_unknown'], 'next': '--resume-stage (operator decision) or stop'})
        print('LABELS_WITHHELD', withhold_reason, flush=True)
        return
    eligible = [(p, j) for p, j in branches if (p / j / 'commit.json').exists()]
    try:
        for p, j in eligible:
            if not (p / j / 'c_b_scores.json').exists():
                runner(p, j, 'c_b', led, root)
        if not cfg['labels_e']:
            context.write_json(root / 'labels_c_b_only.json', {'epoch': time.time(), 'branches': [str(p) for p, j in eligible], 'resumed': resumed})
        else:
            for p, j in eligible:
                if not (p / j / 'e_frozen.json').exists():
                    runner(p, j, 'freeze_e', led, root)
            context.write_json(root / 'all_e_predictions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in eligible], 'resumed': resumed})
            for p, j in eligible:
                if not (p / j / 'e_scores.json').exists():
                    runner(p, j, 'score_e', led, root)
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'external', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
        print('EXTERNAL_STOP', type(exc).__name__, flush=True)
    context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': resumed})


def copy_physical_cache(src_job_dir: Path, common: Path, job: str, *, seeds) -> dict:
    """Opt-in cross-package physical cache (DEV-TEMPO-AUG-WORKFLOW-LEARNING-LOOP §3): copies T objects, materials, models and the
    C_A-only physical cells of an earlier package's <job>_common/<job>/ into this package's cache after binding checks. Nothing of the
    old package is written; no C_B / E file exists in a physical cache (labels live in branch copies). Model paths are rewritten."""
    dst = common / job
    if (dst / 'aug_materials' / 'index.json').exists():
        return {'copied': False, 'reason': 'cache already present'}
    src_job_dir = Path(src_job_dir)
    if not (src_job_dir / 'aug_materials' / 'index.json').exists():
        return {'copied': False, 'reason': 'no source cache'}
    with np.load(src_job_dir / 'scaler.npz') as z:
        if str(z['dataset']) != DATASET or int(z['t']) != rd.resolve_job(DATASET, job).t or [str(x) for x in z['roster']] != rd.resolve_job(DATASET, job).roster:
            raise RuntimeError('source cache belongs to another dataset / cut / roster')
    forbidden = [p.name for p in src_job_dir.rglob('*') if p.name in ('c_b_scores.json', 'e_scores.json', 'e_frozen.json', 'commit.json') or p.name.startswith('predictions_e')]
    if forbidden:
        raise RuntimeError('source cache carries labels or deliveries: %s' % forbidden[:3])
    dst.mkdir(parents=True, exist_ok=True)
    for name in ('scaler.npz', 'overview.json'):
        shutil.copy2(src_job_dir / name, dst / name)
    shutil.copytree(src_job_dir / 'materials', dst / 'materials')
    shutil.copytree(src_job_dir / 'aug_materials', dst / 'aug_materials')
    reg = load_registry(dst)
    for m, r in reg.items():
        for k in ('path', 'summary_path'):
            if r.get(k):
                r[k] = str(dst / 'aug_materials' / Path(r[k]).name)
    save_registry(dst, reg)
    (dst / 'runs').mkdir(exist_ok=True)
    (dst / 'cells').mkdir(exist_ok=True)
    (dst / 'pred_c_a').mkdir(exist_ok=True)
    n = 0
    for cp in sorted((src_job_dir / 'cells').glob('*.json')):
        rec = context.read_json(cp)
        if rec.get('status') != 'OK' or rec.get('tag') or rec['model_seed'] not in seeds or rec.get('profile') not in (PROFILE_VERSION, None):
            continue
        if set(rec.get('scores', {})) - {'c_a'}:
            raise RuntimeError('physical cell %s carries a label block' % rec['cell_id'])
        mp = Path(rec['model_path'])
        shutil.copy2(mp, dst / 'runs' / mp.name)
        rec['model_path'] = str(dst / 'runs' / mp.name)
        rec['copied_from_package_cache'] = str(cp)
        context.write_json(dst / 'cells' / cp.name, rec)
        pc = src_job_dir / 'pred_c_a' / (rec['cell_id'] + '.npz')
        if pc.exists():
            shutil.copy2(pc, dst / 'pred_c_a' / pc.name)
        n += 1
    out = {'copied': True, 'source': str(src_job_dir), 'materials': sorted(reg), 'cells': n, 'copied_local': now()}
    context.write_json(dst / 'cache_copied_from.json', out)
    return out


def prepare_common(root: Path, job: str, led, cfg: dict) -> Path:
    """Public references of one job: build (0 fits) -> random supply frozen (Target) -> reference fits -> Menu_CA decision written later by its arm."""
    common = root / (job + '_common')
    ji, seeds, public_ids = cfg['job_index'][job], tuple(cfg['seeds']), tuple(cfg['public_ids'])
    src = (cfg.get('cache_from') or {}).get(job)
    if src and not (common / job / 'aug_materials' / 'index.json').exists():
        copy_physical_cache(Path(src), common, job, seeds=seeds)
    if not all(m in load_registry(common / job) for m in public_ids):
        t0 = time.time()
        rc, _ = run_sub(['--worker-build', root / (job + '_common'), job, ji], root / 'logs' / ('build_%s.log' % job), 1800)
        led.s['material_wall_seconds'] = led.s.get('material_wall_seconds', 0.0) + (time.time() - t0)
        led._save()
        if rc != 0:
            raise RuntimeError('common build failed for %s' % job)
    if cfg.get('random') and not (common / 'random_supply.json').exists():
        table = context.read_json(common / job / 'overview.json')['entities']
        dsks.write_once(common / 'random_supply.json', draw_random_supply(job, ji, table))
    for m in public_ids:
        for s in seeds:
            rec = fit_physical(led, common, job, m, s)
            if rec['scores']['c_a']['status'] != 'SCORABLE':
                raise RuntimeError('C_A_NOT_SCORABLE: %s' % rec['cell_id'])
    return common


def _run_arms(root: Path, cfg: dict, led, client, branches, failures, *, resumed: bool) -> None:
    fc = FastClient(client, root, cfg.get('fast_system')) if client is not None else None
    for job in cfg['jobs']:
        if (root / (job + '_not_scorable.json')).exists():
            continue
        ji, seeds, public_ids = cfg['job_index'][job], tuple(cfg['seeds']), tuple(cfg['public_ids'])
        try:
            common = prepare_common(root, job, led, cfg)
        except RuntimeError as exc:
            if 'C_A_NOT_SCORABLE' not in str(exc):
                raise
            context.write_json(root / (job + '_not_scorable.json'), {'job': job, 'status': 'C_A_NOT_SCORABLE'})
            failures.append({'job': job, 'kind': 'C_A_NOT_SCORABLE'})
            continue
        for arm in cfg['order'][job]:
            path = root / ('%s_%s' % (job, arm))
            kn = cfg['knowledge'].get(job, {}).get(arm)
            if arm not in ('random', 'menu_ca') and kn is None:
                if not (root / ('%s_%s_treatment.json' % (job, arm))).exists():
                    context.write_json(root / ('%s_%s_treatment.json' % (job, arm)), cfg.get('treatment_note', {}).get(arm) or
                                       {'status': 'NO_TREATMENT', 'note': 'no frozen card for this arm; not run, no substitute, no budget transfer'})
                continue
            branches.append((path, job))
            try:
                if path.exists() and (path / 'branch_result.json').exists():
                    prior = context.read_json(path / 'branch_result.json')
                    if prior['status'] == 'COMPLETE':
                        result = br.RunResult(**{k: v for k, v in prior.items() if k != 'trace'})
                    elif arm in ('random', 'menu_ca'):
                        raise RuntimeError('control branch cannot be resumed')
                    else:
                        result = resume_fast_branch(path, job, knowledge_from_json(kn), led, fc, common, job_index=ji, public_ids=public_ids, seeds=seeds)
                elif path.exists():
                    raise RuntimeError('branch directory without a result; inspect before any replay')
                elif arm == 'random':
                    result = random_branch(path, job, led, common, context.read_json(common / 'random_supply.json'), job_index=ji, public_ids=public_ids, seeds=seeds)
                elif arm == 'menu_ca':
                    result = menu_branch(path, job, led, common, job_index=ji, public_ids=public_ids, seeds=seeds)
                else:
                    result = fast_branch(path, job, knowledge_from_json(kn), led, fc, common, job_index=ji, public_ids=public_ids, seeds=seeds)
                if result.status != 'COMPLETE':
                    failures.append({'branch': path.name, 'kind': result.failure_kind, 'reason': result.reason})
            except Exception as exc:  # noqa: BLE001
                failures.append({'branch': path.name, 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
                print('BRANCH_FAILED', path.name, type(exc).__name__, flush=True)
            led.check_wall()
            if client is not None:
                led.check_llm()
                if getattr(client, 'fatal', False) or rt.unknown_usage_blocks(led):
                    raise RuntimeError('backend fatal or unknown usage')
                if arm not in ('random', 'menu_ca') and not proxy_reachable():
                    raise RuntimeError('LLM proxy unreachable; stage stopped with labels withheld (resume when the proxy is back)')
            if numeric_seconds(root.parent) > NUMERIC_WALL_S:
                raise RuntimeError('numeric wall clock (4 h) exhausted')


def stage_worker(cfg_path: Path, *, client=None, runner=label_stage) -> None:
    cfg = context.read_json(cfg_path)
    root = Path(cfg['output'])
    if (root / 'experiment_started.json').exists():
        raise RuntimeError('stage already started; no paid replay')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    context.write_json(root / 'config.json', cfg)
    led = rt.RuntimeLedger(cfg['ledger_path'], **cfg['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage; paid stage refused without an operator decision')
    rt.FIT_RETRY = bool(cfg.get('fit_retry'))
    context.write_json(root / 'experiment_started.json', {'epoch': time.time(), 'pid': os.getpid(), 'stage': cfg['stage']})
    branches, failures, stopped = [], [], None
    try:
        if client is None and cfg.get('llm'):
            client = rt.MeteredClient(led, root / 'raw_responses', http_cap=int(cfg['http_cap']))     # config only; no unmetered ping
        _run_arms(root, cfg, led, client, branches, failures, resumed=False)
        context.write_json(root / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches]})
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
        stopped = type(exc).__name__
        print('EXECUTION_STOP', stopped, flush=True)
    finally:
        context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures})
    open_labels(root, cfg, branches, failures, led, runner=runner, withhold_reason=stopped and 'execution stopped before every planned arm was attempted (%s)' % stopped)


def proxy_reachable() -> bool:
    import socket
    from urllib.parse import urlparse
    u = urlparse(MODEL['base_url'])
    try:
        with socket.create_connection((u.hostname, u.port or 80), timeout=3):
            return True
    except OSError:
        return False


def resume_stage(root: Path, stage: str, *, accept_unknown_usage: bool = False, restart=()) -> None:
    """One operator continuation: labels still withheld (existing boundary check), completed branches kept, a never-sent Fast call
    continued, missing arms run as planned, then labels. Unknown usage needs an explicit acceptance."""
    from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp
    sroot = paths(root)[stage]
    cfg = context.read_json(sroot / 'config.json')
    if (sroot / 'resume.json').exists():
        raise RuntimeError('already resumed once; no second replay')
    blocked = rtp.labels_boundary_violations(sroot)
    if blocked:
        raise RuntimeError('resume refused; labels are no longer withheld: ' + '; '.join(blocked))
    led = rt.RuntimeLedger(cfg['ledger_path'], **cfg['caps'])
    if rt.unknown_usage_blocks(led):
        if not accept_unknown_usage:
            raise RuntimeError('ledger holds unknown usage; pass --accept-unknown-usage to record an explicit operator decision')
        led.s['unknown_usage_accepted'] = led.s['llm_tokens_unknown']
        led._save()
    for name in ('execution_finished.json', 'labels_withheld.json'):
        if (sroot / name).exists():
            shutil.move(sroot / name, sroot / name.replace('.json', '_before_resume.json'))
    moved = []
    for name in restart:
        b = sroot / name
        prior = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
        if not b.is_dir() or (prior and (prior['status'] == 'COMPLETE' or prior['new_evaluations'])) or any(b.rglob('commit.json')):
            raise RuntimeError('only an interrupted branch without commit or new evaluation can be restarted: %s' % name)
        dest = sroot / ('%s__interrupted_1' % name)
        shutil.move(b, dest)
        moved.append({'branch': name, 'kept_as': dest.name, 'prior_status': prior and prior['status'], 'prior_failure_kind': prior and prior['failure_kind']})
    context.write_json(sroot / 'resume.json', {'epoch': time.time(), 'accepted_unknown_usage': bool(accept_unknown_usage), 'slow_recalled': False, 'restarted_branches': moved})
    rt.FIT_RETRY = bool(cfg.get('fit_retry'))
    branches, failures, stopped = [], [], None
    try:
        client = rt.MeteredClient(led, sroot / 'raw_responses', http_cap=int(cfg['http_cap'])) if cfg.get('llm') else None
        _run_arms(sroot, cfg, led, client, branches, failures, resumed=True)
        context.write_json(sroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches], 'resumed': True})
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
        stopped = type(exc).__name__
    finally:
        context.write_json(sroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': True})
    open_labels(sroot, cfg, branches, failures, led, resumed=True, withhold_reason=stopped and 'resumed stage stopped again (%s)' % stopped)


def stage_config(root: Path, stage: str, *, jobs, order, knowledge, labels_e: bool, llm: bool, random: bool, treatment_note=None) -> Path:
    P_ = paths(root)
    path = P_['configs'] / ('%s.json' % stage)
    if path.exists():
        return path
    sc = stage_caps(root, stage)
    cfg = {'study': 'tempo_aug_workflow_skill', 'status': 'FROZEN', 'stage': stage, 'output': str(P_[stage]), 'jobs': list(jobs),
           'order': {j: list(order[j]) for j in jobs}, 'knowledge': knowledge, 'labels_e': bool(labels_e), 'llm': bool(llm), 'random': bool(random),
           'seeds': list(SEEDS), 'job_index': {j: JOB_INDEX[j] for j in jobs}, 'public_ids': list(PUBLIC), 'limits': LIMITS, 'max_tool_corrections': MAX_TOOL_CORRECTIONS,
           'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'fast_token_cap': FAST_TOKEN_CAP, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'caps': sc['caps'], 'http_cap': sc['http_cap'],
           'ledger_path': str(P_['ledger']), 'fit_retry': True, 'fit_attempts_at_stage_start': context.read_json(P_['ledger'])['fit_attempts'],
           'treatment_note': treatment_note or {}}
    dsks.write_once(path, cfg)
    return path


def prepare_baselines(root: Path, stage: str = 'source') -> dict:
    """0-LLM part of the NEXT stage run ahead (public reference fits of its jobs) under that stage's cumulative caps; the stage worker later
    finds the cells and charges nothing twice. Only the next stage keeps the cumulative accounting exact, so only 'source' is allowed here."""
    if stage != 'source':
        raise RuntimeError('only the next stage (source) may be prepared ahead')
    preflight(root)
    sc = stage_caps(root, stage)
    led = rt.RuntimeLedger(paths(root)['ledger'], **sc['caps'])
    rt.FIT_RETRY = True
    sroot = paths(root)[stage]
    (sroot / 'logs').mkdir(parents=True, exist_ok=True)
    cfg = {'jobs': list(SOURCE_JOBS), 'seeds': list(SEEDS), 'job_index': {j: JOB_INDEX[j] for j in SOURCE_JOBS}, 'public_ids': list(PUBLIC), 'random': False}
    out = {'stage': stage, 'started_local': now(), 'jobs': {}}
    for job in SOURCE_JOBS:
        before = led.s['fit_attempts']
        prepare_common(sroot, job, led, cfg)
        out['jobs'][job] = {'fits_charged': led.s['fit_attempts'] - before, 'materials': sorted(load_registry(sroot / (job + '_common') / job))}
    out['finished_local'] = now()
    context.write_json(sroot / 'baselines_prepared.json', out)
    print('BASELINES_PREPARED', stage, out['jobs'], flush=True)
    return out


def run_stage(root: Path, stage: str, cfg_path: Path) -> dict:
    P_ = paths(root)
    sroot = P_[stage]
    if not (sroot / 'experiment_started.json').exists():
        led = context.read_json(P_['ledger'])
        remaining = TOTAL['max_wall_s'] - (time.time() - led['started_epoch'])
        if remaining <= 0:
            raise RuntimeError('package wall exhausted')
        P_['logs'].mkdir(parents=True, exist_ok=True)
        print('STAGE_START', stage, now(), flush=True)
        with (P_['logs'] / ('%s.log' % stage)).open('a', encoding='utf-8') as log:
            p = subprocess.Popen([sys.executable, '-B', '-m', MODULE, '--stage-worker', str(cfg_path)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=worker_env())
            try:
                rc = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)
                context.write_json(root / ('%s_supervisor_timeout.json' % stage), {'epoch': time.time(), 'pid': p.pid})
                rc = 124
        print('STAGE_EXIT', stage, rc, now(), flush=True)
    return dsks.stage_status(sroot)


# ============================================================================= preflight (0 cost) and frozen configuration
def environment() -> dict:
    code = 'import sys, torch, numpy, pandas; print(sys.version.split()[0], torch.__version__, numpy.__version__, pandas.__version__, torch.get_num_threads())'
    got = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, env=worker_env()).stdout.split()
    return {'python': got[0], 'torch': got[1], 'numpy': got[2], 'pandas': got[3], 'threads': int(got[4]), 'executable': sys.executable,
            'KMP_DUPLICATE_LIB_OK_at_start': KMP_AT_START, 'workers_env_kmp_removed': True}


def preflight(root: Path = ROOT) -> dict:
    """Frozen before the first fit of the package: jobs, roster, windows, Consumer, environment, model, prompts, program table, seed rules,
    quotas, orders, selection and aggregation formulas. Reads no T row of any planned job (eligibility is checked at each stage start)."""
    import inspect
    root.mkdir(parents=True, exist_ok=True)
    p = root / 'frozen_config.json'
    if p.exists():
        return context.read_json(p)
    env = environment()
    prev = context.read_json(REPO / '_scratch' / 'dev_tempo_aug_source_alignment' / 'preflight.json')['environment']
    env_ok = all(prev[k] == env[k] for k in ('python', 'torch', 'numpy', 'threads'))
    src = inspect.getsource(rt.MeteredClient)
    if not all(x in src for x in ("model='cpa-grok-4.6'", "'grok-4.6-build'", 'temperature=0', "base_url='http://127.0.0.1:8318/v1'")):
        raise RuntimeError('metered client no longer carries the frozen model identity')
    jobs = {}
    for j, t in JOB_T.items():
        js = rd.resolve_job(DATASET, j)
        if js.t != t:
            raise RuntimeError('job table drift %s: %d != %d' % (j, js.t, t))
        jobs[j] = {'t': t, 'job_index': JOB_INDEX[j], 'train_rows': list(js.train_range), 'c_a_origins': list(js.c_a), 'c_b_origins': list(js.c_b), 'e_origins': list(js.e),
                   'last_row_read_exclusive': js.e_target_rows[1]}
    avail = context.read_json(AVAILABILITY)['candidates']
    targets = {j: avail['RD02@%d' % JOB_T[j]] for j in TARGET_JOBS}
    if not all(v['eligible_and_scorable'] and not v['E_rows_scored_by_any_known_line'] for v in targets.values()):
        raise RuntimeError('a Target batch is not marked eligible / unscored in the availability audit')
    sealing_ok = max(v['last_row_read_exclusive'] for v in jobs.values()) <= PRSA_SEALED_FROM_ROW
    cfg = {'package': PACKAGE, 'task': TASK, 'frozen_local': now(), 'frozen_epoch': time.time(), 'method_reference': 'Eval-Skill arXiv:2606.07040v2 §3-4, App. B (Workflow generation-selection only)',
           'dataset': {'id': DATASET, 'field': rd.DATASETS[DATASET]['field'], 'roster': rd.DATASETS[DATASET]['roster'], 'exposure': rd.EXPOSURE,
                       'prsa_sealed_from_row': PRSA_SEALED_FROM_ROW, 'max_row_read_exclusive': max(v['last_row_read_exclusive'] for v in jobs.values()), 'sealing_ok': sealing_ok},
           'geometry': {'L': rd.L, 'H': rd.H, 'T': rd.TRAIN_SPAN, 'c_a': 't, t+48', 'c_b': 't+96, t+144', 'e': 't+192..t+336 step 48', 'legal_parent': 'X >= 32 finite and y fully finite',
                        'eligibility': '>=336 finite T values, std > 1e-6, >= 32 legal parents per entity', 'coverage_min': rd.COVERAGE_MIN},
           'jobs': jobs, 'stages': {'wiring': [WIRING_JOB], 'source': list(SOURCE_JOBS), 'select': list(SELECT_JOBS), 'target': list(TARGET_JOBS)},
           'orders': {'select': {j: list(o) for j, o in SELECT_ORDER.items()}, 'target': {j: list(o) for j, o in TARGET_ORDER.items()}},
           'seeds': {'formal': list(SEEDS), 'wiring': list(WIRING_SEEDS)}, 'consumer': CONSUMER, 'environment': env, 'environment_matches_source_alignment': env_ok,
           'model': MODEL, 'fast_system': FAST_SYSTEM, 'slow_system': SLOW_SYSTEM, 'contracts': CONTRACTS, 'limits': LIMITS, 'fast_token_cap': FAST_TOKEN_CAP,
           'max_tool_corrections': MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': EVIDENCE_ROUNDTRIP,
           'materials': {'profile': PROFILE_VERSION, 'program_table': PROGRAMS, 'program_count': len(PROGRAMS),
                         'entity_program_seed': 'int(SeedSequence([%d, job_index, program_index, entity_index]).generate_state(1, uint32)[0]); one Streams per (entity, program), windows in legal order' % SEED_ROOT,
                         'public_references': {'None': 'parent view only', 'FixedMixup': FIXED_MIXUP, 'P_AmpResample': PUBLIC_STEPS['P_AmpResample'], 'P_NoMixRecipe': PUBLIC_STEPS['P_NoMixRecipe']},
                         'catalog': ta.CATALOG, 'recipe': ta.RECIPE_TEXT, 'fixed_order': list(ta.PRIMITIVES), 'exclusive': [list(x) for x in ta.EXCLUSIVE], 'max_steps': ta.MAX_STEPS, 'max_rules': ta.MAX_RULES,
                         'old_v1_materials_reused': 'none (job or name equality never reuses a v1 AmpResample / NoMixRecipe model)'},
           'random_search_b4': {'rng': 'Generator(PCG64(SeedSequence([%d, job_index])))' % RANDOM_SEED_ROOT, 'uniform_slots': 2, 'conditional_slots': 2, 'quantiles': list(RANDOM_QUANTILES),
                                'predicate_op': '>=', 'max_tries_per_conditional_slot': RANDOM_MAX_TRIES, 'fallback': 'a not-yet-drawn uniform program from the same stream',
                                'delivery': 'argmin three-seed C_A mean over public references (in public order) then random plans (draw order)'},
           'menu_ca': 'argmin three-seed C_A mean over the four public references; ties None, FixedMixup, P_AmpResample, P_NoMixRecipe; 0 LLM, 0 extra fits',
           'selection': {'formula': 'J(v) = mean over the two Select jobs of mean_seed C_B(committed_v) / mean_seed C_B(None); lower is better',
                         'tie': 'differences <= 1e-12 resolve in the order no_skill, W1, W2', 'incomplete': 'a non-positive or unscorable denominator or an incomplete option -> SELECTION_INCOMPLETE'},
           'main_readout': {'d': 'd[j,s] = E(F_noSkill,j,s) - E(F_domainSkill,j,s); positive = Skill better',
                            'G': 'G = mean_j(100 x mean_s d[j,s] / mean_s E(None,j,s)), four Target jobs equal weight, percentage points of that job\'s None loss',
                            'interval': 'paired SE and n=3 two-sided 95%% t interval (df=2, t=%.6f)' % T975_DF2},
           'budget': {'total': TOTAL, 'stage_allocations': ALLOC, 'stage_retries': STAGE_RETRIES, 'http_cap': HTTP_CAP, 'numeric_wall_s': NUMERIC_WALL_S,
                      'fit_formula': 'wiring 5; Source 4x(4+4)x3=96; Select 2x(4+3x4)x3=96; Target 4x(4+3x4)x3=192; +6 technical retries = 395 physical attempts'},
           'label_barriers': 'Source / Select: all planned arms end -> C_B only. Target: all four jobs\' arms end and Menu_CA frozen -> C_B -> all E predictions frozen -> barrier -> E scoring',
           'availability_targets': targets}
    if not (env_ok and sealing_ok):
        raise RuntimeError('preflight failed: environment %s sealing %s' % (env_ok, sealing_ok))
    dsks.write_once(p, cfg)
    return cfg


# ============================================================================= wiring acceptance (RD02_T1, scripted client, <= 5 fits, 0 LLM)
WIRING_PLAN = {'default': {'steps': [{'op': 'tp_censor'}, {'op': 'tp_shock'}]},
               'rules': [{'when': {'feature': 'missing_fraction', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': [{'op': 'tp_resample'}, {'op': 'tp_amplitude'}, {'op': 'tp_random_conv'}]}],
               'rationale': 'wiring acceptance composition (not an agent decision)', 'observation_fields_used': ['missing_fraction']}
WIRING_PLAN_PERMUTED = {'default': {'steps': [{'op': 'tp_shock'}, {'op': 'tp_censor'}]},
                        'rules': [{'when': {'feature': 'missing_fraction', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': [{'op': 'tp_random_conv'}, {'op': 'tp_amplitude'}, {'op': 'tp_resample'}]}],
                        'rationale': 'same composition, other listing order and other construction order', 'observation_fields_used': ['missing_fraction']}


class ScriptedClient:
    """Scripted Fast for the wiring acceptance and the smoke; not an agent and not a research strategy."""
    fatal = False

    def __init__(self, scripts: dict):
        self.scripts, self.count, self.requests = scripts, {}, []

    def fast(self, unit):
        def call(request):
            k = self.count[unit] = self.count.get(unit, 0) + 1
            self.requests.append({'unit': unit, 'n': k, 'candidates': [c['plan_id'] for c in request['candidates']], 'loaded': request['guidance']['loaded'],
                                  'consumer': request['overview'].get('consumer', {}), 'remaining': request['remaining']})
            step = self.scripts[unit][min(k, len(self.scripts[unit])) - 1]
            return {'actions': copy.deepcopy(step(request) if callable(step) else step)}
        return call


def _probe_binding(common: Path) -> dict:
    """RD02_T1 only: the probe's None / FixedMixup cells may be reused when scaler, legal set, parent arrays and the FixedMixup child are equal."""
    job = WIRING_JOB
    mine, probe = common / job, P.RUN / job
    out = {'job': job, 'checked_local': now()}
    with np.load(mine / 'scaler.npz') as a, np.load(probe / 'scaler.npz') as b:
        out['scaler_and_legal_equal'] = {f: bool(np.array_equal(a[f], b[f])) for f in ('mean', 'std', 'scale', 'legal_ent', 'legal_k', 'legal_counts')}
    with np.load(rd.load_index(mine)['P0_Linear']['path']) as a, np.load(rd.load_index(probe)['P0_Linear']['path']) as b:
        out['parent_X_equal'] = bool(np.array_equal(a['X'], b['X']))
    with np.load(load_registry(mine)['FixedMixup']['path']) as a, np.load(probe / 'children' / 'FixedMixup.npz') as b:
        out['fixed_mixup_child_equal'] = bool(np.array_equal(a['Xc'], b['Xc']) and np.array_equal(a['yc'], b['yc']))
    out['ok'] = all(out['scaler_and_legal_equal'].values()) and out['parent_X_equal'] and out['fixed_mixup_child_equal']
    return out


def _link_probe_cells(common: Path) -> None:
    job = WIRING_JOB
    for m in WIRING_PUBLIC:
        for s in WIRING_SEEDS:
            pc = context.read_json(P.cell_path(job, m, s, P.N_FULL))
            if pc['status'] != 'OK' or pc['model_seed'] != s or pc['material_id'] != m or list(pc['rows_read']) != list(rd.resolve_job(DATASET, job).evaluate_rows):
                raise RuntimeError('probe cell binding failed for %s %s' % (m, s))
            rec = {**pc, 'cell_id': phys_cell_id(job, m, s), 'source_cell': pc['cell_id'], 'source_package': 'DEV-DATA-AUG-OPERATOR-PROBE', 'tag': '',
                   'material_key': load_registry(common / job)[m]['key'], 'reused_after_binding_check': True}
            (common / job / 'cells').mkdir(exist_ok=True)
            context.write_json(common / job / 'cells' / (rec['cell_id'] + '.json'), rec)


def wiring(root: Path = ROOT) -> dict:
    """Task §4: one non-public composition really trained (3 fits) through the seven tools on branch A; branch B builds the same
    composition in another order (bitwise-identical material, fits reused); both C_B -> E frozen -> barrier -> E; rerun one seed of the
    composition and one historical None seed (2 check fits). Every fit is charged to the package ledger."""
    preflight(root)
    P_ = paths(root)
    wroot = P_['wiring']
    rec_path = wroot / 'wiring.json'
    if rec_path.exists():
        return context.read_json(rec_path)
    sc = stage_caps(root, 'wiring')
    led = rt.RuntimeLedger(P_['ledger'], **sc['caps'])
    rt.FIT_RETRY = False
    wroot.mkdir(parents=True, exist_ok=True)
    (wroot / 'logs').mkdir(exist_ok=True)
    job, ji, seeds = WIRING_JOB, JOB_INDEX[WIRING_JOB], WIRING_SEEDS
    cfg = {'stage': 'wiring', 'output': str(wroot), 'jobs': [job], 'seeds': list(seeds), 'job_index': {job: ji}, 'public_ids': list(WIRING_PUBLIC), 'labels_e': True, 'random': False, 'llm': False}
    context.write_json(wroot / 'config.json', cfg)
    context.write_json(wroot / 'experiment_started.json', {'epoch': time.time(), 'pid': os.getpid(), 'stage': 'wiring'})
    common = wroot / (job + '_common')
    rc, _ = run_sub(['--worker-build', common, job, ji], wroot / 'logs' / 'build.log', 1800)
    if rc != 0:
        raise RuntimeError('wiring build failed')
    binding = _probe_binding(common)
    context.write_json(wroot / 'probe_binding.json', binding)
    if not binding['ok']:
        raise RuntimeError('probe binding failed; None / FixedMixup cannot be reused')
    _link_probe_cells(common)
    out = {'label': 'WIRING_ACCEPTANCE (scripted client; no LLM; not an agent result)', 'job': job, 'seeds': list(seeds), 'probe_binding': binding, 'fits_before': led.s['fit_attempts']}
    scripts = {
        job + '_A': [[{'tool': 'overview', 'arguments': {}}], [{'tool': 'inspect_data', 'arguments': {'entity_indices': [0, 1], 'kind': 'hour_profile'}}],
                     [{'tool': 'build_material', 'arguments': {'plan_id': 'W_comp', 'policy': WIRING_PLAN}}], [{'tool': 'inspect_material', 'arguments': {'plan_id': 'W_comp', 'entity_index': 0}}],
                     [{'tool': 'evaluate', 'arguments': {'plan_id': 'W_comp'}}], [{'tool': 'compare', 'arguments': {'a': 'None', 'b': 'W_comp'}}],
                     [{'tool': 'commit', 'arguments': {'plan_id': 'W_comp', 'reason': 'wiring acceptance: scripted commit of the trained composition'}}]],
        job + '_B': [[{'tool': 'build_material', 'arguments': {'plan_id': 'W_other', 'policy': uniform_policy([{'op': 'tp_censor'}], 'built first, never evaluated')}}],
                     [{'tool': 'build_material', 'arguments': {'plan_id': 'W_same', 'policy': WIRING_PLAN_PERMUTED}}], [{'tool': 'inspect_material', 'arguments': {'plan_id': 'W_same'}}],
                     [{'tool': 'evaluate', 'arguments': {'plan_id': 'W_same'}}], [{'tool': 'compare', 'arguments': {'a': 'FixedMixup', 'b': 'W_same'}}],
                     [{'tool': 'commit', 'arguments': {'plan_id': 'W_same', 'reason': 'wiring acceptance: scripted commit of the reused composition'}}]]}
    client = ScriptedClient(scripts)
    branches, failures = [], []
    for arm in ('A', 'B'):
        path = wroot / ('%s_%s' % (job, arm))
        branches.append((path, job))
        before = led.s['fit_attempts']
        res = fast_branch(path, job, br.Knowledge(), led, client, common, job_index=ji, public_ids=WIRING_PUBLIC, seeds=seeds)
        out['branch_' + arm] = {'status': res.status, 'failure_kind': res.failure_kind, 'reason': res.reason, 'committed': res.committed_plan_id, 'calls': res.calls,
                                'tool_calls': res.tool_calls, 'new_evaluations': res.new_evaluations, 'fits_charged': led.s['fit_attempts'] - before,
                                'events': [{k: v for k, v in e.items() if k in ('event', 'tool', 'error', 'plan_id')} for e in res.trace]}
        if res.status != 'COMPLETE':
            failures.append({'branch': path.name, 'kind': res.failure_kind})
    ra, rb = load_registry(wroot / (job + '_A') / job), load_registry(wroot / (job + '_B') / job)
    out['same_physical_material'] = ra['W_comp']['phys_id'] == rb['W_same']['phys_id'] and ra['W_comp']['key'] == rb['W_same']['key']
    out['branch_B_sees_only_own_plans'] = sorted(rb) == sorted(list(WIRING_PUBLIC) + ['W_other', 'W_same'])
    # bitwise identity under another construction order: rebuild the same assignment alone in a scratch cache and compare arrays
    scratch = wroot / 'rng_check_scratch'
    if scratch.exists():
        shutil.rmtree(scratch)
    rc, _ = run_sub(['--worker-build', scratch, job, ji], wroot / 'logs' / 'rng_check_build.log', 1800)
    asg = ra['W_comp']['assignment']
    context.write_json(aug_dir(scratch / job) / 'TA001__request.json', {'assignment': asg})
    rc2, _ = run_sub(['--worker-material', scratch, job, 'TA001', ji], wroot / 'logs' / 'rng_check_material.log', 900)
    with np.load(ra['W_comp']['path']) as a, np.load(load_registry(scratch / job)['TA001']['path']) as b:
        out['material_bitwise_identical_other_order'] = rc == 0 and rc2 == 0 and bool(np.array_equal(a['Xc'], b['Xc']) and np.array_equal(a['yc'], b['yc']))
    shutil.rmtree(scratch, ignore_errors=True)
    context.write_json(wroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches]})
    context.write_json(wroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures})
    open_labels(wroot, cfg, branches, failures, led)
    esA = context.read_json(wroot / (job + '_A') / job / 'e_scores.json')['cells'] if (wroot / (job + '_A') / job / 'e_scores.json').exists() else {}
    esB = context.read_json(wroot / (job + '_B') / job / 'e_scores.json')['cells'] if (wroot / (job + '_B') / job / 'e_scores.json').exists() else {}
    out['new_composition_scored_e'] = {'A_W_comp_cells': sorted(c for c in esA if '__W_comp__' in c), 'B_W_same_cells': sorted(c for c in esB if '__W_same__' in c)}
    out['labels'] = {'c_b_A': (wroot / (job + '_A') / job / 'c_b_scores.json').exists(), 'c_b_B': (wroot / (job + '_B') / job / 'c_b_scores.json').exists(),
                     'barrier': (wroot / 'all_e_predictions_frozen.json').exists(), 'e_A': bool(esA), 'e_B': bool(esB)}
    # check fits: same composition one seed rerun (weights identical), None one historical seed vs the probe cache
    phys = ra['W_comp']['phys_id']
    rr = fit_physical(led, common, job, phys, seeds[0], tag='rerun')
    orig = context.read_json(common / job / 'cells' / (phys_cell_id(job, phys, seeds[0]) + '.json'))
    out['rerun_composition'] = {'seed': seeds[0], 'weights_identical': compare_models(rr['model_path'], orig['model_path']),
                                'c_a_equal': rr['scores']['c_a']['normalized_mse_macro'] == orig['scores']['c_a']['normalized_mse_macro']}
    chk = fit_physical(led, common, job, 'None', seeds[0], tag='chk')
    pc = context.read_json(P.cell_path(job, 'None', seeds[0], P.N_FULL))
    out['none_vs_probe'] = {'seed': seeds[0], 'weights_identical': compare_models(chk['model_path'], pc['model_path']),
                            'c_a_equal': chk['scores']['c_a']['normalized_mse_macro'] == pc['scores']['c_a']['normalized_mse_macro']}
    out['fits_charged_total'] = led.s['fit_attempts'] - out['fits_before']
    out['checks'] = {'branch_A_complete_with_new_composition': out['branch_A']['status'] == 'COMPLETE' and out['branch_A']['committed'] == 'W_comp' and out['branch_A']['fits_charged'] == 3,
                     'branch_B_reused_fits': out['branch_B']['status'] == 'COMPLETE' and out['branch_B']['committed'] == 'W_same' and out['branch_B']['fits_charged'] == 0 and out['branch_B']['new_evaluations'] == 1,
                     'same_physical_material': out['same_physical_material'], 'material_bitwise_identical_other_order': out['material_bitwise_identical_other_order'],
                     'branch_isolation': out['branch_B_sees_only_own_plans'], 'labels_opened_after_both_commits_with_barrier': all(out['labels'].values()),
                     'new_composition_in_e_scores': len(out['new_composition_scored_e']['A_W_comp_cells']) == 3 and len(out['new_composition_scored_e']['B_W_same_cells']) == 3,
                     'rerun_weights_identical': out['rerun_composition']['weights_identical'] and out['rerun_composition']['c_a_equal'],
                     'none_retrain_equals_probe': out['none_vs_probe']['weights_identical'] and out['none_vs_probe']['c_a_equal'],
                     'fits_at_most_5': out['fits_charged_total'] <= 5}
    out['status'] = 'PASS' if all(out['checks'].values()) else 'FAIL'
    out['written_local'] = now()
    context.write_json(rec_path, out)
    print('WIRING', out['status'], out['checks'], flush=True)
    return out


# ============================================================================= formation: census (deterministic, compression rules frozen first) + ONE Slow
COMPRESSION_RULES = {
    'frozen_before_first_slow_call': True,
    'rules': ['job_started.overview and tool_completed(overview).output -> "see overview (identical)" (the overview is given once per branch)',
              'fast_request.guidance -> "see first request (identical)" after the first request of a branch (a no-Skill Source branch always renders loaded=[])',
              'inspect_data outputs: any numeric list longer than 48 -> {n, first_8, last_8, min, max, mean, n_null} (raw T views are re-derivable; no action, candidate, failure, sign or C_A/C_B number is dropped)',
              'inspect_material outputs: window.parent / window.child (240-point arrays) -> {n, min, max, rms_diff_child_minus_parent}; all counts, RMS and fractions kept',
              'every float inside a trajectory tool output is rounded to 5 decimals (the overview itself is already 4-decimal); evaluate / compare / build_material / commit outputs and every Fast response otherwise kept in full',
              'per plan: material_spec, physical material diagnostics without per-entity windows, C_A by seed, post-commit C_B by seed, commit reason, status, cost'],
    'tier_2_only_if_the_frozen_client_reservation_would_exceed_the_formation_cap': [
        'evaluate / compare outputs: the per-entity vectors loss_by_seed_entity / delta_by_seed_entity -> per seed {n, min, max, mean}; per-seed and per-origin values, means, signs and SE untouched',
        'build_material outputs: entity_program / rule_index integer lists kept (12 entries); nothing else changes'],
    'tier_3_only_if_tier_2_still_exceeds_the_reservation (added after the tier-2 refusal of 2026-09-18 22:19, before any Slow request was sent; size-only)': [
        'inspect_data outputs: every numeric list longer than 8 -> {n, min, max, mean, n_null} (first_8 / last_8 dropped); hour_profile / daily_means row lists -> one columnar record with 2-decimal values (same numbers); gap lists -> first 5 gaps + n_gaps + total missing (raw T views are re-derivable from T)',
        'inspect_material outputs: the fixed explanatory strings field_note / interpretation / units (verbatim tool-contract text) dropped; every count, RMS and fraction kept',
        'plans[*].material_diagnostics: per_step entries with 0 windows, units and seed_rule dropped; rms / fraction / identity counts / recipe step sets kept',
        'tool_started(*).arguments -> "see fast_response" (every argument is verbatim in the Fast response of the same call); compare.scope (fixed text) dropped',
        'tool_completed(build_material).output.material_spec -> "see plans[<id>].material_spec" (identical object kept once per plan); inspect_material.entities -> columnar {steps, rms_X, rms_y} lists',
        'plans[*]: material_spec of the four public references -> "see public_semantics.public_references" (identical in every branch); material_diagnostics id/profile/build_seconds fields dropped',
        'overview.entities -> columnar {field: [12 values]}; overview.geometry once in public_semantics; the compression rule text itself replaced by a pointer to census_compression_rules.json'],
    'reason': 'the metered client reserves 2 x payload bytes against the 400,000-token formation cap before sending; the rules are size-only and were fixed before any Source result was read',
}


def _shrink_diagnostics(summ: dict | None, tier: int):
    if summ is None or tier < 3:
        return summ
    out = {k: v for k, v in summ.items() if k not in ('units', 'seed_rule', 'material_id', 'job_id', 'profile', 'job_index', 'build_seconds')}
    if isinstance(out.get('per_step'), dict):
        out['per_step'] = {k: v for k, v in out['per_step'].items() if v.get('windows')}
    return out


def _columnar(rows: list) -> dict:
    """[{k: v}, ...] -> {k: [v, ...]} (same values, keys once)."""
    keys = list(rows[0]) if rows else []
    return {'n': len(rows), 'columns': {k: [r.get(k) for r in rows] for k in keys}}


def _compress(value, tool: str, tier: int = 1):
    def rnd(x):
        return round(x, 5) if isinstance(x, float) else x

    def per_seed_stats(v):
        return [{'n': len(row), 'min': rnd(min(row)), 'max': rnd(max(row)), 'mean': rnd(statistics.fmean(row))} for row in v]

    def squeeze_list(v):
        nums = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
        return {'n': len(v), 'first_8': v[:8], 'last_8': v[-8:], 'min': min(nums) if nums else None, 'max': max(nums) if nums else None,
                'mean': statistics.fmean(nums) if nums else None, 'n_null': sum(x is None for x in v)}

    def walk(x, path=''):
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if tool == 'inspect_material' and path.endswith('.window') and k in ('parent', 'child'):
                    continue
                if tier >= 2 and tool in ('evaluate', 'compare') and k in ('loss_by_seed_entity', 'delta_by_seed_entity') and isinstance(v, list) and v and isinstance(v[0], list):
                    out[k + '_stats'] = per_seed_stats(v)
                    continue
                if tier >= 3 and ((tool == 'inspect_material' and k in ('field_note', 'interpretation', 'units')) or (tool == 'compare' and k in ('scope', 'block'))
                                  or (tool == 'inspect_data' and k in ('units', 'gap_format')) or (tool == 'commit' and k == 'delivery_model_ref')):
                    continue
                if tier >= 3 and tool == 'inspect_material' and k == 'entities' and isinstance(v, dict) and path == '':
                    out[k] = {'columns': {'entity': list(v), 'steps': [e.get('steps') for e in v.values()], 'rms_X': [e.get('rms_X') for e in v.values()], 'rms_y': [e.get('rms_y') for e in v.values()]}}
                    continue
                if tier >= 3 and tool == 'build_material' and k == 'material_spec' and path == '':
                    out[k] = 'see plans[%s].material_spec (identical)' % x.get('plan_id')
                    continue
                if tier >= 3 and tool == 'inspect_data' and k == 'gaps' and isinstance(v, list):
                    out[k] = {'first_5': v[:5], 'n_gaps': len(v), 'total_missing_hours': sum(g[2] for g in v if isinstance(g, list) and len(g) == 3)}
                    continue
                out[k] = walk(v, path + '.' + str(k))
            if tool == 'inspect_material' and path.endswith('.window') and 'parent' in x and 'child' in x:
                pa, ch = np.asarray(x['parent'], dtype=float), np.asarray(x['child'], dtype=float)
                out['parent_child_summary'] = {'n': int(pa.size), 'parent_min': float(pa.min()), 'parent_max': float(pa.max()), 'child_min': float(ch.min()), 'child_max': float(ch.max()),
                                               'rms_diff_child_minus_parent': float(np.sqrt(((ch - pa) ** 2).mean()))}
            return out
        if isinstance(x, list):
            limit = 8 if tier >= 3 else 48
            if tier >= 3 and tool == 'inspect_data' and len(x) > limit and all(isinstance(v, dict) for v in x):
                # hour_profile / daily_means rows -> one columnar record (2-decimal values), same numbers, ~1/6 of the bytes
                cols = {}
                for k in x[0]:
                    vals = [v.get(k) for v in x]
                    if k == 'hour' and vals == list(range(len(x))):
                        continue                                   # implied index 0..n-1
                    cols[k] = [round(v, 2) if isinstance(v, float) else v for v in vals]
                return {'n': len(x), 'columns': cols}
            if tool == 'inspect_data' and len(x) > limit and all(isinstance(v, (int, float, type(None))) and not isinstance(v, bool) for v in x):
                sq = {k: (rnd(v) if not isinstance(v, list) else [rnd(y) for y in v]) for k, v in squeeze_list(x).items()}
                return {k: v for k, v in sq.items() if k not in ('first_8', 'last_8')} if tier >= 3 else sq
            return [walk(v, path + '[]') for v in x]
        return rnd(x)
    return walk(value)


def _plan_blocks(bdir: Path, job: str) -> dict:
    out = {}
    cb = context.read_json(bdir / job / 'c_b_scores.json')['cells'] if (bdir / job / 'c_b_scores.json').exists() else {}
    for c, cell in branch_cells(bdir / job).items():
        row = out.setdefault(cell['material_id'], {})
        row[cell['model_seed']] = {'c_a': cell['scores']['c_a']['normalized_mse_macro'], 'c_b': (cb.get(c) or {}).get('c_b', {}).get('normalized_mse_macro'),
                                   'c_b_status': (cb.get(c) or {}).get('c_b', {}).get('status')}
    return out


def _trace(bdir: Path) -> list:
    tr = bdir / 'trace.jsonl'
    return [json.loads(x) for x in tr.read_text(encoding='utf-8').splitlines() if x.strip()] if tr.exists() else []


def census(root: Path, tier: int = 1) -> dict:
    P_ = paths(root)
    records = []
    for j in SOURCE_JOBS:
        ns = P_['source'] / ('%s_not_scorable.json' % j)
        if ns.exists():
            records.append({'job': j, 'status': 'C_A_NOT_SCORABLE', 'note': 'C_A below the coverage rule; references fitted, no Fast trajectory'})
    for old in sorted(P_['source'].glob('RD02_*__interrupted_*')):
        r = old / 'branch_result.json'
        records.append({'branch_attempt': old.name, 'status': 'INTERRUPTED_TECHNICAL', 'result': ({k: v for k, v in context.read_json(r).items() if k != 'trace'} if r.exists() else None)})
    branches = [(P_['source'] / ('%s_no_skill' % j), j) for j in SOURCE_JOBS if (P_['source'] / ('%s_no_skill' % j)).exists()]
    complete = [(b, j) for b, j in branches if dsks._branch_complete_with_c_b(b, j)]
    for b, j in branches:
        if (b, j) not in complete:
            r = b / 'branch_result.json'
            records.append({'job': j, 'status': 'INCOMPLETE', 'result': ({k: v for k, v in context.read_json(r).items() if k != 'trace'} if r.exists() else None)})
    if not complete:
        return {'census_status': 'SOURCE_INCOMPLETE', 'domain_id': DOMAIN, 'records': records, 'legal_evidence_refs': [], 'note': 'no COMPLETE Source trajectory with post-commit C_B; no substitute'}
    refs, out = [], []
    for b, j in complete:
        res = context.read_json(b / 'branch_result.json')
        commit_rec = context.read_json(b / j / 'commit.json')
        rows, first_guidance = [], None
        for r in _trace(b):
            r = copy.deepcopy(r)
            if r['event'] == 'job_started':
                r['overview'] = 'see overview (identical)'
            if r['event'] == 'fast_request':
                if first_guidance is None:
                    first_guidance = r.get('guidance')
                elif r.get('guidance') == first_guidance:
                    r['guidance'] = 'see first request (identical)'
            if r['event'] == 'tool_completed':
                r['output'] = 'see overview (identical)' if r['tool'] == 'overview' else _compress(r['output'], r['tool'], tier)
            refs.append(r['event_id'])
            if tier >= 3 and r['event'] == 'tool_started':
                continue                                           # arguments are verbatim in the Fast response; results in tool_completed
            rows.append(r)
        ov = context.read_json(b / j / 'overview.json')
        reg = load_registry(b / j)
        blocks = _plan_blocks(b, j)
        plans = {}
        for mid, rec in reg.items():
            sc = blocks.get(mid, {})
            seeds_ok = set(sc) == set(SEEDS)
            summ = {k: v for k, v in context.read_json(rec['summary_path']).items() if k not in ('entities', 'entity_program_seeds')} if rec.get('summary_path') else None
            spec_p = aug_dir(b / j) / (mid + '__compiled.json')
            mspec = context.read_json(spec_p)['material_spec'] if spec_p.exists() else ('see public_semantics.public_references' if tier >= 3 else public_spec(mid))
            if tier >= 3 and isinstance(mspec, dict):
                mspec = {k: v for k, v in mspec.items() if k not in ('execution_order', 'profile', 'alias_of')}
            plans[mid] = {'material_spec': mspec, 'material_diagnostics': _shrink_diagnostics(summ, tier),
                          'c_a_by_seed': [sc[s]['c_a'] for s in SEEDS] if seeds_ok else None,
                          'c_b_after_commit_by_seed': [sc[s]['c_b'] for s in SEEDS] if seeds_ok else None,
                          'c_b_status': sorted({str(sc[s]['c_b_status']) for s in SEEDS}) if seeds_ok else None, 'fitted': seeds_ok, 'committed': mid == res['committed_plan_id']}
            for k in ('c_a_by_seed', 'c_b_after_commit_by_seed'):
                v = plans[mid][k]
                plans[mid][k.replace('_by_seed', '_mean')] = statistics.fmean(v) if v and all(x is not None for x in v) else None
        refs += ['%s:overview' % j, '%s:c_b_after_commit' % j]
        tokens = dsks._unit_tokens(P_['source']).get('fast:%s' % b.name, {})
        out.append({'job': j, 'branch_status': res['status'], 'committed_plan_id': res['committed_plan_id'], 'commit_reason': commit_rec['reason'],
                    'cost': {'fast_calls': res['calls'], 'tool_calls': res['tool_calls'], 'new_evaluations': res['new_evaluations'], 'fast_tokens': tokens},
                    'failure_kind': res['failure_kind'],
                    'overview': {'reference': '%s:overview' % j, 'n_entities': ov['n_entities'], 'summary': ov['summary'],
                                 'entities': _columnar(ov['entities']) if tier >= 3 else ov['entities'],
                                 **({'geometry': ov['geometry']} if tier < 3 else {'geometry': 'see public_semantics.geometry (identical for every batch)'})},
                    'plans': plans, 'delayed_block': {'reference': '%s:c_b_after_commit' % j, 'block': 'C_B (opened after every Source branch committed)'}, 'trajectory': rows})
    geometry = context.read_json(complete[0][0] / complete[0][1] / 'overview.json')['geometry']
    actions = ta.action_table()
    if tier >= 3:
        actions['primitives'] = {n: {k: v for k, v in p.items() if k != 'source'} for n, p in actions['primitives'].items()}
    ev = {'census_status': 'CENSUS_COMPLETE', 'domain_id': DOMAIN, 'profile': PROFILE_VERSION, 'branches': out, 'structural_and_technical_records': records,
          'legal_evidence_refs': sorted(set(refs)),
          'compression': ({**COMPRESSION_RULES, 'tier_used': tier} if tier < 3 else
                          {'tier_used': 3, 'rules': 'size-only compression, see census_compression_rules.json (tiers 1-3); no action, candidate, failure, sign or C_A/C_B number is dropped; long raw T lists and repeated text are summarized or pointed to; tool_started rows omitted (arguments are in the Fast response, results in tool_completed)'}),
          'public_semantics': {'field_definitions': rd.FIELD_DEFINITIONS, 'actions': actions, 'tool_contracts': CONTRACTS, 'consumer': CONSUMER, 'geometry': geometry,
                               'public_references': {m: public_spec(m) for m in PUBLIC}, 'budget_per_job': LIMITS,
                               'metric': 'missing-aware normalized MSE on observed future cells; lower is better'},
          'neutral_reminder': 'Rankings on one evaluation time block are not guaranteed to hold on later blocks.',
          'purpose': 'Offline Slow input: can the Source processes of this neutral domain be consolidated into at most two reusable augmentation Workflows?'}
    low = json.dumps(ev, ensure_ascii=False).lower()
    if any(n in low for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')):
        raise PermissionError('census names a data source')
    return ev


def _ref_ranges(refs: list) -> dict:
    """Event ids are '<job>:<index>' (contiguous per branch) plus '<job>:overview' / '<job>:c_b_after_commit'."""
    out = {}
    for r in refs:
        job, _, tail = r.partition(':')
        row = out.setdefault(job, {'event_ids': [], 'other': []})
        (row['event_ids'] if tail.isdigit() else row['other']).append(int(tail) if tail.isdigit() else r)
    return {job: {'event_id_range': '%s:%d .. %s:%d (every integer index)' % (job, min(v['event_ids']), job, max(v['event_ids'])) if v['event_ids'] else None,
                  'other_refs': v['other']} for job, v in out.items()}


def slow_payload(census_: dict) -> dict:
    refs = census_['legal_evidence_refs']
    compact = census_.get('compression', {}).get('tier_used', 1) >= 3
    cen = {k: v for k, v in census_.items() if k != 'legal_evidence_refs'} if compact else census_
    return {'domain_id': DOMAIN, 'census': cen, 'allowed_batch_features': sorted(ALLOWED_FEATURES), 'max_candidates': 2,
            **({'legal_evidence_refs': _ref_ranges(refs), 'legal_evidence_refs_format': 'evidence_refs must be exact ids inside these ranges, e.g. "<job>:<integer>", or one of other_refs'}
               if compact else {'legal_evidence_refs': refs}),
            'required': {'principles': None, 'observable_applicability': {'const': True}}, 'status': 'CANDIDATE_TEST_ONLY'}


def propose_workflow(census_: dict, call) -> dict:
    """One scientific formation call; one contract correction only (format / contract, incl. principles != null or scope != const:true).
    Never a semantic resample; transport / account / budget faults end as PROPOSE_CALL_FAILED."""
    refs = census_['legal_evidence_refs']
    payload = slow_payload(census_)
    low = json.dumps(payload, ensure_ascii=False).lower()
    if any(n in low for n in tuple(spec.DATASETS) + tuple(x for x in rd.SOURCE_NAMES if x not in ('rd01', 'rd02'))):
        raise PermissionError('Slow payload names a data source')
    attempts, prior = [], None
    for i in range(2):
        try:
            raw = call(payload)
        except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
            attempts.append({'attempt': i, 'fault_kind': type(exc).__name__})
            return {'status': 'PROPOSE_CALL_FAILED', 'attempts': attempts, 'skills': []}
        except ValueError as exc:
            attempts.append({'attempt': i, 'error': 'response not JSON: %s' % str(exc)[:200]})
            prior = None
        else:
            try:
                out = dsk.parse_proposal(raw, domain_id=DOMAIN, legal_evidence_refs=refs, max_candidates=2, allowed_features=ALLOWED_FEATURES, text_check=skill_text_check)
                for s in out['skills']:
                    if s.principles is not None:
                        raise ValueError('principles must be null in this package (Workflow only)')
                    if s.observable_applicability != {'const': True}:
                        raise ValueError('observable_applicability must be exactly {"const": true}')
                attempts.append({'attempt': i, 'ok': True})
                return {'status': 'KEEP' if out['decision'] == 'KEEP' else 'PROPOSED', 'attempts': attempts, 'raw': raw, **out}
            except ValueError as exc:
                attempts.append({'attempt': i, 'error': str(exc)[:300]})
                prior = raw if dsk._jsonable(raw) else None
        if i == 0:
            payload = {**payload, 'correction': {'previous_output': prior, 'error': attempts[-1].get('error'),
                                                 'instruction': 'Correct this contract error only. KEEP is allowed. Do not seek a different outcome.'}}
    return {'status': 'PROPOSE_PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'skills': []}


def formation(root: Path) -> None:
    P_ = paths(root)
    out = P_['formation']
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'census_compression_rules.json').exists():
        dsks.write_once(out / 'census_compression_rules.json', {**COMPRESSION_RULES, 'frozen_local': now()})
    if not (out / 'census.json').exists():
        sizes = {}
        for tier in (1, 2, 3):
            cen = census(root, tier=tier)
            if cen['census_status'] != 'CENSUS_COMPLETE':
                break
            # the frozen client's reservation: 2 x (payload bytes + 2048 + max_tokens) must fit the formation allocation, with room for one correction round
            sizes['tier_%d_bytes' % tier] = len(json.dumps([{'role': 'system', 'content': SLOW_SYSTEM}, {'role': 'user', 'content': json.dumps(slow_payload(cen), ensure_ascii=False)}], ensure_ascii=False).encode('utf-8'))
            cen['compression']['payload_bytes_by_tier'] = dict(sizes)
            if 2 * (sizes['tier_%d_bytes' % tier] + 8192 + 2048 + MAX_OUTPUT_TOKENS) <= ALLOC['formation']['tokens']:
                break
        dsks.write_once(out / 'census.json', cen)
    if (out / 'propose.json').exists():
        return
    cen = context.read_json(out / 'census.json')
    if cen['census_status'] != 'CENSUS_COMPLETE':
        dsks.write_once(out / 'propose.json', {'status': 'NO_PROPOSAL_INPUT', 'census_status': cen['census_status'], 'skills': [], 'formation': 'INCOMPLETE'})
        return
    sc = stage_caps(root, 'formation')
    led = rt.RuntimeLedger(P_['ledger'], **sc['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage; paid formation refused')
    if not proxy_reachable():
        raise RuntimeError('LLM proxy not reachable; formation refused before any paid call')
    client = rt.MeteredClient(led, out / 'raw_responses', http_cap=sc['http_cap'])
    res = propose_workflow(cen, lambda p: client.call('slow_domain', DOMAIN, p, SLOW_SYSTEM, max_tokens=MAX_OUTPUT_TOKENS))
    dsks.write_once(out / 'propose.json', {**{k: v for k, v in res.items() if k != 'skills'}, 'skills': [s.to_json() for s in res['skills']],
                                           'census_legal_ref_count': len(cen['legal_evidence_refs']), 'slow_requests': led.s['llm_requests'], 'written_local': now()})
    if client.fatal or rt.unknown_usage_blocks(led):
        raise RuntimeError('backend fatal or unknown usage during formation; stop')


def candidates_of(root: Path) -> list:
    prop = context.read_json(paths(root)['formation'] / 'propose.json')
    return [dsk.skill_from_json(s) for s in prop.get('skills', [])] if prop.get('status') == 'PROPOSED' else []


# ============================================================================= Select (T4, Q2) with J(v), freeze
def select_stage(root: Path) -> dict:
    skills = candidates_of(root)
    if not skills:
        if not (paths(root)['formation'] / 'select_skipped.json').exists():
            dsks.write_once(paths(root)['formation'] / 'select_skipped.json', {'status': 'NO_CANDIDATES', 'note': 'KEEP / no proposal / formation failure: the no-Skill option wins by default; empty Skill frozen'})
        return {'status': 'SKIPPED'}
    arms = {'cand_' + s.skill_id.split('-')[1]: s for s in skills}
    order = {j: [a for a in SELECT_ORDER[j] if a == 'no_skill' or a in arms] for j in SELECT_JOBS}
    knowledge = {j: {'no_skill': asdict(br.Knowledge()), **{a: asdict(dsk.skill_knowledge(s)) for a, s in arms.items()}} for j in SELECT_JOBS}
    cfgp = stage_config(root, 'select', jobs=SELECT_JOBS, order=order, knowledge=br.json_copy(knowledge), labels_e=False, llm=True, random=False)
    return run_stage(root, 'select', cfgp)


def _committed_block(bdir: Path, job: str, block: str) -> list | None:
    """Three-seed block losses of the branch's committed plan (None when incomplete / unscorable)."""
    if not (bdir / 'branch_result.json').exists():
        return None
    res = context.read_json(bdir / 'branch_result.json')
    if res['status'] != 'COMPLETE' or not (bdir / job / 'commit.json').exists():
        return None
    plan = res['committed_plan_id']
    return _block_vec(bdir, job, plan, block)


def _block_vec(bdir: Path, job: str, plan: str, block: str) -> list | None:
    jd = bdir / job
    cells = branch_cells(jd)
    if block == 'c_a':
        vals = {c['model_seed']: c['scores']['c_a'] for c in cells.values() if c['material_id'] == plan}
    else:
        f = jd / ('%s_scores.json' % block)
        if not f.exists():
            return None
        sc = context.read_json(f)['cells']
        vals = {v['model_seed']: v[block] for c, v in sc.items() if v['material_id'] == plan}
    seeds = tuple(context.read_json(bdir.parent / 'config.json')['seeds']) if (bdir.parent / 'config.json').exists() else SEEDS
    if set(vals) < set(seeds) or any(vals[s]['status'] != 'SCORABLE' for s in seeds):
        return None
    return [vals[s]['normalized_mse_macro'] for s in seeds]


def select_workflow(root: Path) -> dict:
    """J(v) = mean over the two Select jobs of mean_seed C_B(committed_v) / mean_seed C_B(None); lower is better; ties (<= 1e-12) in the order
    no_skill, W1, W2; every option must be COMPLETE and scorable on both jobs, None's C_B must be positive, else SELECTION_INCOMPLETE."""
    P_ = paths(root)
    skills = candidates_of(root)
    options = ['no_skill'] + ['cand_' + s.skill_id.split('-')[1] for s in skills]
    runs, J, incomplete = {}, {}, []
    for o in options:
        runs[o], ratios = {}, []
        for j in SELECT_JOBS:
            b = P_['select'] / ('%s_%s' % (j, o))
            cb = _committed_block(b, j, 'c_b')
            none_cb = _block_vec(b, j, 'None', 'c_b') if b.exists() else None
            res = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
            row = {'status': res['status'] if res else 'NOT_RUN', 'committed': res['committed_plan_id'] if res else None, 'failure_kind': res['failure_kind'] if res else None,
                   'c_b_by_seed': cb, 'none_c_b_by_seed': none_cb, 'c_a_by_seed': _committed_block(b, j, 'c_a')}
            if cb is None or none_cb is None or statistics.fmean(none_cb) <= 0:
                incomplete.append('%s/%s' % (o, j))
                row['ratio'] = None
            else:
                row['ratio'] = statistics.fmean(cb) / statistics.fmean(none_cb)
                ratios.append(row['ratio'])
            runs[o][j] = row
        if len(ratios) == len(SELECT_JOBS):
            J[o] = statistics.fmean(ratios)
    rec = {'formula': 'J(v) = mean_j(mean_seed C_B(committed_v) / mean_seed C_B(None)); lower is better', 'tie_rule': '<= 1e-12 -> order no_skill, W1, W2',
           'options': options, 'runs': runs, 'J': J, 'incomplete': incomplete, 'block': 'c_b', 'dev_jobs': list(SELECT_JOBS), 'written_local': now()}
    if 'no_skill' not in J or len(J) != len(options):
        rec.update(status='SELECTION_INCOMPLETE', selected=None, note='an option is not complete / scorable on both Select jobs; empty Skill frozen, no success claim')
        return rec
    best = min(J.values())
    winner = next(o for o in options if J[o] - best <= 1e-12)
    rec['selected'] = winner
    rec['status'] = 'NO_TREATMENT' if winner == 'no_skill' else 'SELECTED'
    rec['margins_vs_no_skill'] = {o: J[o] - J['no_skill'] for o in options}
    return rec


def freeze_skill(root: Path) -> dict:
    P_ = paths(root)
    out = P_['formation'] / 'frozen_skill.json'
    if out.exists():
        return context.read_json(out)
    skills = candidates_of(root)
    prop = context.read_json(P_['formation'] / 'propose.json')
    if prop.get('status') not in ('PROPOSED', 'KEEP'):
        raise RuntimeError('formation status %s is not a scientific outcome; nothing is frozen' % prop.get('status'))
    if not skills:
        rec = {'domain_id': DOMAIN, 'status': 'NO_TREATMENT', 'skill': None, 'propose_status': prop.get('status'), 'selection': None,
               'note': 'KEEP / no proposal input / formation failure: empty Skill frozen; Generic is not substituted', 'frozen_local': now()}
    else:
        sel = select_workflow(root)
        dsks.write_once(P_['formation'] / 'selection.json', sel)
        if sel['status'] == 'SELECTED':
            s = next(x for x in skills if 'cand_' + x.skill_id.split('-')[1] == sel['selected'])
            frozen = replace(s, status='FROZEN_SELECTED', source_stage='freeze', derived_from=s.skill_id)
            rec = {'domain_id': DOMAIN, 'status': 'FROZEN_SELECTED', 'skill': frozen.to_json(), 'selection': {k: sel[k] for k in ('status', 'selected', 'J', 'margins_vs_no_skill', 'dev_jobs', 'block')},
                   'candidates': [x.to_json() for x in skills], 'injected_body': frozen.rendered_body, 'frozen_local': now(),
                   'note': 'selected on the two Select jobs\' C_B; selection is not promotion to a verified Skill'}
        else:
            rec = {'domain_id': DOMAIN, 'status': sel['status'], 'skill': None, 'selection': {k: sel.get(k) for k in ('status', 'selected', 'J', 'incomplete', 'dev_jobs', 'block')},
                   'candidates': [x.to_json() for x in skills], 'frozen_local': now(), 'note': 'no-Skill won or the selection basis is incomplete: empty Skill frozen'}
    dsks.write_once(out, rec)
    dsks.write_once(root / 'selected_skill.json', rec)
    return rec


# ============================================================================= Target
def target_stage(root: Path) -> dict:
    fr = freeze_skill(root)
    if fr['status'] == 'FROZEN_SELECTED':
        s = dsk.skill_from_json(fr['skill'])
        if s.status != 'FROZEN_SELECTED' or s.domain_id != DOMAIN:
            raise PermissionError('only the FROZEN_SELECTED card of this domain enters Target')
        kn_skill, note = asdict(dsk.skill_knowledge(s)), {}
        order = {j: list(TARGET_ORDER[j]) + ['menu_ca'] for j in TARGET_JOBS}
    else:
        kn_skill = None
        note = {'f_domain_skill': {'status': 'EMPTY_SKILL_ALIAS', 'alias_of_arm': 'f_no_skill', 'frozen_skill_status': fr['status'],
                                   'note': 'the frozen Skill is empty: every actual request condition equals the no-Skill arm; the no-Skill trajectory and delivery are reused, no second LLM run'}}
        order = {j: [a for a in TARGET_ORDER[j] if a != 'f_domain_skill'] + ['menu_ca'] for j in TARGET_JOBS}
    knowledge = {j: {'f_no_skill': asdict(br.Knowledge()), 'f_domain_skill': kn_skill, 'random': None, 'menu_ca': None} for j in TARGET_JOBS}
    cat = paths(root)['formation'] / 'target_catalog.json'
    if not cat.exists():
        dsks.write_once(cat, {'status': 'TARGET_FROZEN', 'epoch': time.time(), 'frozen_local': now(), 'skill': fr.get('skill'), 'skill_status': fr['status'],
                              'frozen_before_first_target_fit': True, 'orders': order})
    cfgp = stage_config(root, 'target', jobs=TARGET_JOBS, order=order, knowledge=br.json_copy(knowledge), labels_e=True, llm=True, random=True, treatment_note=note)
    return run_stage(root, 'target', cfgp)


# ============================================================================= package driver
def package_run(root: Path = ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    status = context.read_json(root / 'package_status.json') if (root / 'package_status.json').exists() else {}

    def stop(where, st):
        status.update({'stopped_at': where, 'stage_status': st, 'epoch': time.time(), 'local': now()})
        context.write_json(root / 'package_status.json', status)
        print('PACKAGE_STOP', where, st.get('status'), flush=True)
        return status
    preflight(root)
    w = wiring(root)
    status['wiring'] = w['status']
    if w['status'] != 'PASS':
        return stop('wiring', w)
    st = run_stage(root, 'source', stage_config(root, 'source', jobs=SOURCE_JOBS, order={j: ['no_skill'] for j in SOURCE_JOBS},
                                                knowledge={j: {'no_skill': asdict(br.Knowledge())} for j in SOURCE_JOBS}, labels_e=False, llm=True, random=False))
    status['source'] = st
    if st['status'] != 'FINISHED':
        return stop('source', st)
    formation(root)
    prop = context.read_json(paths(root)['formation'] / 'propose.json')
    status['formation'] = prop.get('status')
    if prop.get('status') not in ('PROPOSED', 'KEEP'):
        # an instrument / transport / contract failure is not a scientific KEEP: no empty Skill is frozen, no Select/Target starts
        return stop('formation', {'status': prop.get('status'), 'attempts': prop.get('attempts'), 'note': 'formation did not produce a scientific outcome; operator decision needed'})
    st = select_stage(root)
    status['select'] = st
    if st['status'] not in ('FINISHED', 'SKIPPED'):
        return stop('select', st)
    fr = freeze_skill(root)
    status['frozen_skill'] = fr['status']
    st = target_stage(root)
    status['target'] = st
    if st['status'] != 'FINISHED':
        return stop('target', st)
    status['finished_epoch'] = time.time()
    status['finished_local'] = now()
    context.write_json(root / 'package_status.json', status)
    readout(root)
    print('PACKAGE_FINISHED', flush=True)
    return status


# ============================================================================= readout (task §9)
def _stats(v: list) -> dict:
    m = statistics.fmean(v)
    se = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None
    return {'mean': m, 'se': se, 't95': [m - T975_DF2 * se, m + T975_DF2 * se] if se is not None and len(v) == 3 else None, 'per_seed': v,
            'signs': [int(x > 0) - int(x < 0) for x in v]}


def _paired(a, b, den=None) -> dict | None:
    """a - b per seed (positive = b lower loss); gain_pp = 100 x mean(a-b) / den (den = that job's three-seed None mean)."""
    if a is None or b is None:
        return None
    d = [x - y for x, y in zip(a, b)]
    out = {'loss_diff': _stats(d)}
    if den:
        out['gain_pp'] = _stats([100 * x / den for x in d])
    return out


def _e_per_origin(bdir: Path, job: str, plan: str) -> list | None:
    f = bdir / job / 'e_scores.json'
    if not f.exists():
        return None
    sc = context.read_json(f)['cells']
    rows = {v['model_seed']: v['e'] for v in sc.values() if v['material_id'] == plan}
    seeds = tuple(context.read_json(bdir.parent / 'config.json')['seeds'])
    if set(rows) < set(seeds) or any(rows[s]['status'] != 'SCORABLE' for s in seeds):
        return None
    O = rows[seeds[0]]['n_origins']
    return [[statistics.fmean(rows[s]['per_origin_entity_normalized_mse'][o]) / O for s in seeds] for o in range(O)]     # contribution to the macro


def _arm_record(bdir: Path, job: str) -> dict:
    res = context.read_json(bdir / 'branch_result.json')
    plan = res['committed_plan_id']
    reg = load_registry(bdir / job)
    rec = {'status': res['status'], 'failure_kind': res['failure_kind'], 'reason': res['reason'], 'commit': plan, 'fast_calls': res['calls'], 'tool_calls': res['tool_calls'],
           'new_evaluations': res['new_evaluations'], 'behavior': dsks._behavior(bdir), 'fast_tokens': dsks._unit_tokens(bdir.parent).get('fast:%s' % bdir.name),
           'plans_built': sorted(m for m in reg if m not in PUBLIC), 'commit_reason': context.read_json(bdir / job / 'commit.json')['reason'] if (bdir / job / 'commit.json').exists() else None}
    if plan:
        comp = reg[plan].get('compiled') or {}
        rec['committed_material'] = {'phys_id': reg[plan].get('phys_id', plan), 'public': plan in PUBLIC,
                                     'assignment': comp.get('assignment') if plan not in PUBLIC else reg[plan].get('assignment'),
                                     'rule_index': comp.get('rule_index'), 'policy': comp.get('policy')}
        rec.update({b: _block_vec(bdir, job, plan, b) for b in ('c_a', 'c_b', 'e')})
        rec['e_per_origin_contrib'] = _e_per_origin(bdir, job, plan)
    rec['all_evaluated'] = {m: {b: _block_vec(bdir, job, m, b) for b in ('c_a', 'c_b', 'e')} for m in sorted({c['material_id'] for c in branch_cells(bdir / job).values()})}
    return rec


def readout(root: Path = ROOT) -> dict:
    P_ = paths(root)
    res = {'package': PACKAGE, 'written_local': now(), 'units': {'loss': 'missing-aware normalized MSE macro (lower is better)',
           'gain_pp': 'paired per seed: 100 x (loss_reference - loss_arm) / three-seed mean None loss of that job; positive = arm better',
           'd': 'd[j,s] = E(F_noSkill) - E(F_domainSkill); positive = Skill better', 'G': 'mean over Target jobs of 100 x mean_s d / mean_s E(None)'},
           'stages': {}, 'source': {}, 'formation': {}, 'select': {}, 'target': {}, 'main': {}, 'cost': {}}
    for st in ('wiring', 'source', 'select', 'target'):
        if (P_[st] / 'execution_finished.json').exists():
            res['stages'][st] = dsks.stage_status(P_[st]) if st != 'wiring' else {'status': context.read_json(P_['wiring'] / 'wiring.json')['status'] if (P_['wiring'] / 'wiring.json').exists() else 'NOT_FINISHED'}
    for j in SOURCE_JOBS:
        b = P_['source'] / ('%s_no_skill' % j)
        if (b / 'branch_result.json').exists():
            res['source'][j] = _arm_record(b, j)
    for name in ('census_compression_rules', 'propose', 'selection', 'frozen_skill', 'target_catalog', 'select_skipped'):
        f = P_['formation'] / (name + '.json')
        if f.exists():
            res['formation'][name] = context.read_json(f) if name != 'propose' else {k: v for k, v in context.read_json(f).items() if k != 'raw'}
    if (P_['formation'] / 'census.json').exists():
        cen = context.read_json(P_['formation'] / 'census.json')
        res['formation']['census_status'] = cen['census_status']
        res['formation']['census_bytes'] = (P_['formation'] / 'census.json').stat().st_size
    for j in SELECT_JOBS:
        for arm in ('no_skill', 'cand_W1', 'cand_W2'):
            b = P_['select'] / ('%s_%s' % (j, arm))
            if (b / 'branch_result.json').exists():
                res['select'].setdefault(j, {})[arm] = _arm_record(b, j)
    # Target
    alias = None
    tcfg = context.read_json(P_['configs'] / 'target.json') if (P_['configs'] / 'target.json').exists() else None
    if tcfg and tcfg.get('treatment_note', {}).get('f_domain_skill'):
        alias = tcfg['treatment_note']['f_domain_skill']
    G_rows, complete_jobs = [], []
    for j in TARGET_JOBS:
        row = {'arms': {}, 'alias': alias}
        for arm in ('f_no_skill', 'f_domain_skill', 'random', 'menu_ca'):
            b = P_['target'] / ('%s_%s' % (j, arm))
            if arm == 'f_domain_skill' and alias:
                src = P_['target'] / ('%s_f_no_skill' % j)
                row['arms'][arm] = {**(_arm_record(src, j) if (src / 'branch_result.json').exists() else {'status': 'NOT_RUN'}), 'alias': 'EMPTY_SKILL_ALIAS of f_no_skill'}
                continue
            if (b / 'branch_result.json').exists():
                row['arms'][arm] = _arm_record(b, j)
            else:
                t = P_['target'] / ('%s_%s_treatment.json' % (j, arm))
                row['arms'][arm] = {'status': context.read_json(t)['status'] if t.exists() else 'NOT_RUN'}
        anyb = next((P_['target'] / ('%s_%s' % (j, a)) for a in ('f_no_skill', 'random', 'menu_ca') if (P_['target'] / ('%s_%s' % (j, a)) / job_e_exists(j)).exists()), None)
        if anyb is not None:
            row['public'] = {m: {b_: _block_vec(anyb, j, m, b_) for b_ in ('c_a', 'c_b', 'e')} for m in PUBLIC}
            none_e = row['public']['None']['e']
            den = statistics.fmean(none_e) if none_e else None
            e_of = lambda a: row['arms'].get(a, {}).get('e')
            row['none_mean_e'] = den
            row['skill_vs'] = {'f_no_skill': _paired(e_of('f_no_skill'), e_of('f_domain_skill'), den), 'random': _paired(e_of('random'), e_of('f_domain_skill'), den),
                               'menu_ca': _paired(e_of('menu_ca'), e_of('f_domain_skill'), den),
                               **{m: _paired(row['public'][m]['e'], e_of('f_domain_skill'), den) for m in PUBLIC}}
            row['no_skill_vs'] = {'random': _paired(e_of('random'), e_of('f_no_skill'), den), 'menu_ca': _paired(e_of('menu_ca'), e_of('f_no_skill'), den),
                                  **{m: _paired(row['public'][m]['e'], e_of('f_no_skill'), den) for m in PUBLIC}}
            row['public_vs_none'] = {m: _paired(row['public']['None']['e'], row['public'][m]['e'], den) for m in PUBLIC if m != 'None'}
            row['random_vs_menu'] = _paired(e_of('menu_ca'), e_of('random'), den)
            d = row['skill_vs']['f_no_skill']
            if d and den:
                G_rows.append(d['gain_pp']['mean'])
                complete_jobs.append(j)
            # oracle description: every evaluated candidate's E in every arm (did an arm evaluate but not deliver a better plan?)
            row['oracle_evaluated_e'] = {a: {m: (statistics.fmean(v['e']) if v.get('e') else None) for m, v in row['arms'][a].get('all_evaluated', {}).items()}
                                         for a in row['arms'] if row['arms'][a].get('all_evaluated')}
        res['target'][j] = row
    res['main'] = {'G_skill_vs_no_skill_pp': statistics.fmean(G_rows) if G_rows else None, 'jobs_in_G': complete_jobs, 'complete': len(complete_jobs) == len(TARGET_JOBS),
                   'per_job_gain_pp': {j: res['target'][j]['skill_vs']['f_no_skill']['gain_pp'] for j in complete_jobs},
                   'signs_positive_jobs': sum(res['target'][j]['skill_vs']['f_no_skill']['gain_pp']['mean'] > 0 for j in complete_jobs),
                   'alias': alias}
    for ref in ('random', 'menu_ca', 'P_AmpResample', 'P_NoMixRecipe', 'FixedMixup', 'None'):
        vals = [res['target'][j]['skill_vs'][ref]['gain_pp']['mean'] for j in complete_jobs if res['target'][j].get('skill_vs', {}).get(ref)]
        res['main']['G_skill_vs_%s_pp' % ref] = statistics.fmean(vals) if len(vals) == len(TARGET_JOBS) else None
        vals2 = [res['target'][j]['no_skill_vs'][ref]['gain_pp']['mean'] for j in complete_jobs if res['target'][j].get('no_skill_vs', {}).get(ref)]
        res['main']['G_no_skill_vs_%s_pp' % ref] = statistics.fmean(vals2) if len(vals2) == len(TARGET_JOBS) else None
    res['cost'] = cost_readout(root)
    context.write_json(root / 'result.json', res)
    (root / 'tables.md').write_text(tables(res), encoding='utf-8')
    return res


def job_e_exists(job: str) -> str:
    return job + '/e_scores.json'


def cost_readout(root: Path) -> dict:
    P_ = paths(root)
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    stages = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    per_stage = {}
    names = list(stages)
    for i, st in enumerate(names):
        a = stages[st]['snapshot_at_start']
        b = stages[names[i + 1]]['snapshot_at_start'] if i + 1 < len(names) else {k: led.get(k, 0) for k in a}
        per_stage[st] = {k: (b.get(k, 0) - a.get(k, 0)) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts',
                                                                   'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'fit_wall_seconds')}
    tokens = {s: dsks._unit_tokens(P_[s]) for s in ('source', 'select', 'target') if P_[s].exists()}
    tokens['formation'] = dsks._unit_tokens(P_['formation']) if P_['formation'].exists() else {}
    per_arm_fits = {}
    for s in ('source', 'select', 'target'):
        if not P_[s].exists():
            continue
        for b in sorted(x for x in P_[s].iterdir() if x.is_dir() and (x / 'branch_result.json').exists()):
            job = b.name.split('_')[0] + '_' + b.name.split('_')[1]
            cells = branch_cells(b / job)
            per_arm_fits[b.name] = {'logical_cells': len(cells), 'new_plan_cells': sum(c['material_id'] not in PUBLIC for c in cells.values())}
    return {'ledger': {k: v for k, v in led.items() if k != 'events'}, 'per_stage_delta': per_stage, 'tokens_by_unit': tokens, 'per_arm_logical_cells': per_arm_fits,
            'numeric_seconds': numeric_seconds(root) if P_['ledger'].exists() else None,
            'formation_one_time': {'stages': ['source', 'formation', 'select'], 'note': 'Source Fast + one Slow + Select Fast (fits, requests, tokens, seconds from per_stage_delta)'},
            'deployment_per_job': {'public_references': '4 x 3 fits', 'F_arm': 'up to 4 x 3 fits + <= 16 calls', 'random': 'up to 4 x 3 fits, 0 LLM', 'menu_ca': '0 fits, 0 LLM',
                                   'note': 'shared cache hits are physical savings only; each method\'s independent cost = its logical cells x 3 seeds'}}


def _f(x, nd=2):
    return '—' if x is None else ('%+.*f' % (nd, x))


def tables(res: dict) -> str:
    out = ['# %s tables' % PACKAGE, '', res['units']['gain_pp'], '', '## Main: E of F_domainSkill vs references (gain_pp; positive = Skill better)', '',
           '| Job | vs F_noSkill | vs Random | vs Menu_CA | vs AmpResample | vs NoMixRecipe | vs FixedMixup | vs None | None mean E |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for j in TARGET_JOBS:
        r = res['target'].get(j, {})
        sv = r.get('skill_vs') or {}

        def cell(k):
            v = sv.get(k)
            if not v:
                return '—'
            g = v['gain_pp']
            return '%s (SE %.2f) [%s]' % (_f(g['mean']), g['se'], ''.join('+' if s > 0 else '-' if s < 0 else '0' for s in g['signs']))
        out.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (j, cell('f_no_skill'), cell('random'), cell('menu_ca'), cell('P_AmpResample'), cell('P_NoMixRecipe'), cell('FixedMixup'), cell('None'),
                                                                     _f(r.get('none_mean_e'), 4) if r.get('none_mean_e') is not None else '—'))
    m = res['main']
    out += ['', 'G (Skill vs no-Skill, four jobs equal weight): %s pp; complete: %s; positive jobs: %s/4' % (_f(m.get('G_skill_vs_no_skill_pp')), m.get('complete'), m.get('signs_positive_jobs')), '']
    for ref in ('random', 'menu_ca', 'P_AmpResample', 'P_NoMixRecipe', 'FixedMixup', 'None'):
        out.append('- G Skill vs %s: %s pp; G no-Skill vs %s: %s pp' % (ref, _f(m.get('G_skill_vs_%s_pp' % ref)), ref, _f(m.get('G_no_skill_vs_%s_pp' % ref))))
    out += ['', '## Deliveries and per-seed E', '', '| Job | arm | commit | E per seed | C_B per seed | C_A per seed | calls | new evals |', '|---|---|---|---|---|---|---:|---:|']
    for j in TARGET_JOBS:
        for arm, a in (res['target'].get(j, {}).get('arms') or {}).items():
            fmt = lambda v: ', '.join('%.4f' % x for x in v) if v else '—'
            out.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (j, arm + (' (alias)' if a.get('alias') else ''), a.get('commit'), fmt(a.get('e')), fmt(a.get('c_b')), fmt(a.get('c_a')), a.get('fast_calls', '—'), a.get('new_evaluations', '—')))
        pub = res['target'].get(j, {}).get('public') or {}
        for mname, v in pub.items():
            fmt = lambda v: ', '.join('%.4f' % x for x in v) if v else '—'
            out.append('| %s | ref %s | %s | %s | %s | %s | 0 | 0 |' % (j, mname, mname, fmt(v.get('e')), fmt(v.get('c_b')), fmt(v.get('c_a'))))
    out += ['', '## E origin contributions (Skill vs no-Skill loss difference per origin, three-seed mean; share of the macro)', '']
    for j in TARGET_JOBS:
        r = res['target'].get(j, {})
        a, b = (r.get('arms') or {}).get('f_no_skill', {}).get('e_per_origin_contrib'), (r.get('arms') or {}).get('f_domain_skill', {}).get('e_per_origin_contrib')
        if a and b:
            out.append('- %s: ' % j + '; '.join('o%d %s' % (o, _f(statistics.fmean(x - y for x, y in zip(a[o], b[o])), 4)) for o in range(len(a))))
    out += ['', '## Select (J(v), lower is better)', '']
    sel = res['formation'].get('selection') or {}
    for o, v in (sel.get('J') or {}).items():
        out.append('- %s: J = %.5f' % (o, v))
    out.append('- status: %s, selected: %s' % (sel.get('status'), sel.get('selected')))
    return '\n'.join(out) + '\n'


# ============================================================================= smoke (new risks only; synthetic fits, 0 API, no label row of a planned job)
def smoke(out: Path) -> dict:
    """Runs in a torch-capable worker (rd._init_openmp_before_torch first): program table / RNG identity across order and arms, duplicate
    rejection, logical slots with cross-arm physical cache, random supply, menu, J(v), label refusals, empty vs injected Skill."""
    rd._init_openmp_before_torch()
    import tempfile
    out.mkdir(parents=True, exist_ok=True)
    checks, notes = {}, {}
    # 1. program table and canonicalization
    idx_a = program_index([{'op': 'tp_random_conv'}, {'op': 'tp_censor'}, {'op': 'tp_shock'}])
    idx_b = program_index([{'op': 'tp_shock'}, {'op': 'tp_censor'}, {'op': 'tp_random_conv'}])
    checks['program_table_53_and_canonical'] = len(PROGRAMS) == 53 and idx_a == idx_b and program_index([]) == 0 and program_index([{'op': ta.RECIPE}]) == 52 and len(set(map(_PKEY, PROGRAMS))) == 53
    s1, s2, s3 = entity_program_seed(1, 5, 0), entity_program_seed(1, 5, 1), entity_program_seed(2, 5, 0)
    checks['entity_program_seed_distinct_and_stable'] = len({s1, s2, s3}) == 3 and s1 == entity_program_seed(1, 5, 0) and 0 <= s1 < 2 ** 32
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td = Path(td)
        job, ji = WIRING_JOB, 9
        cA, cB = td / 'cacheA', td / 'cacheB'
        for c in (cA, cB):
            worker_build(str(c), job, ji)
        table = context.read_json(cA / job / 'overview.json')['entities']
        # 2. cross-order / cross-cache bitwise identity; different program differs; None never physical
        asg = ta.compile_plan(WIRING_PLAN, table)['assignment']
        asg_perm = ta.compile_plan(WIRING_PLAN_PERMUTED, table)['assignment']
        ctxA, ctxB = rd.open_job(DATASET, job, cA, 'material'), rd.open_job(DATASET, job, cB, 'material')
        build_material_v2(ctxA, ta.compile_plan(uniform_policy([{'op': 'tp_censor'}], 'x'), table)['assignment'], 'TA001', job_index=ji)     # A builds another program first
        ra = build_material_v2(ctxA, asg, 'TA002', job_index=ji)
        rb = build_material_v2(ctxB, asg_perm, 'TA001', job_index=ji)
        with np.load(ra['path']) as a, np.load(rb['path']) as b:
            same = np.array_equal(a['Xc'], b['Xc']) and np.array_equal(a['yc'], b['yc'])
        with np.load(ra['path']) as a, np.load(load_registry(cA / job)['TA001']['path']) as b:
            diff = not np.array_equal(a['Xc'], b['Xc'])
        checks['material_bitwise_identical_across_order_and_cache'] = bool(same and diff and asg == asg_perm and ra['key'] == rb['key'])
        # public references v2 (not the v1 job-material seeds): the AmpResample child differs from the source-alignment v1 material of the same job
        v1 = REPO / '_scratch' / 'dev_tempo_aug_source_alignment' / 'jobs' / job / 'aug_materials' / 'P_AmpResample.npz'
        if v1.exists():
            with np.load(v1) as a, np.load(load_registry(cA / job)['P_AmpResample']['path']) as b:
                checks['v2_public_material_not_v1'] = not np.array_equal(a['Xc'], b['Xc'])
        # 3. branch views + logical slots + duplicate rejection + cross-arm physical cache, with a fake fit (no training)
        fits = {'n': 0}
        led = rt.RuntimeLedger(td / 'budget.json', max_fit_attempts=999, max_llm_requests=0, max_llm_tokens=0, max_wall_s=3600, max_retries=0)

        def fake_fit(common):
            def f(phys, seed):
                cp = common / job / 'cells' / (phys_cell_id(job, phys, seed) + '.json')
                if cp.exists():
                    return context.read_json(cp)
                fits['n'] += 1
                led.reserve_fit(phys_cell_id(job, phys, seed))
                led.finish_fit(phys_cell_id(job, phys, seed), True, 0.0, '')
                rng = np.random.RandomState(abs(hash((phys, seed))) % (2 ** 31))
                n = len(table)
                mp = common / job / 'runs' / (phys_cell_id(job, phys, seed) + '.pt')
                mp.parent.mkdir(exist_ok=True)
                mp.write_bytes(b'fake')
                rec = {'cell_id': phys_cell_id(job, phys, seed), 'status': 'OK', 'job_id': job, 'material_id': phys, 'material_key': load_registry(common / job)[phys]['key'],
                       'model_seed': seed, 'model_path': str(mp), 'tag': '', 'scores': {'c_a': {'status': 'SCORABLE', 'normalized_mse_macro': float(rng.rand()),
                                                                                                 'per_origin_entity_normalized_mse': rng.rand(2, n).tolist(), 'n_nonfinite_predictions': 0}}}
                cp.parent.mkdir(exist_ok=True)
                context.write_json(cp, rec)
                return rec
            return f
        common = cA
        seeds = WIRING_SEEDS
        adA = WorkflowAdapter(td / 'armA', job, led, REPO, common=common, job_index=ji, public_ids=PUBLIC, seeds=seeds, fit_fn=fake_fit(common))
        baseA = adA.baselines()
        n_pub_fits = fits['n']
        plans = {'X1': uniform_policy([{'op': 'tp_shock'}], 'a'), 'X2': uniform_policy([{'op': 'tp_regime'}, {'op': 'tp_censor'}], 'b'),
                 'X3': uniform_policy([{'op': 'tp_calendar'}], 'c'), 'X4': uniform_policy([{'op': 'tp_resample'}, {'op': 'tp_random_conv'}], 'd'),
                 'X5': uniform_policy([{'op': 'tp_amplitude'}, {'op': 'tp_censor'}], 'e')}
        scriptA = [[{'tool': 'build_material', 'arguments': {'plan_id': k, 'policy': v}} for k, v in plans.items()],
                   [{'tool': 'evaluate', 'arguments': {'plan_id': 'X1'}}], [{'tool': 'evaluate', 'arguments': {'plan_id': 'X2'}}], [{'tool': 'evaluate', 'arguments': {'plan_id': 'X3'}}],
                   [{'tool': 'evaluate', 'arguments': {'plan_id': 'X4'}}], [{'tool': 'evaluate', 'arguments': {'plan_id': 'X5'}}]]
        client = ScriptedClient({'armA': scriptA})
        resA = br.run_job(job_id=job, knowledge=br.Knowledge(), adapter=adA, client=client.fast('armA'), seeds=seeds, baselines=baseA, allowed_features=ALLOWED_FEATURES,
                          tool_contracts=CONTRACTS, entity_count=len(table), ca_origins=2, limits=br.Limits(wall_seconds=600, **LIMITS), evidence_roundtrip=False, max_tool_corrections=2)
        fits_after_A = fits['n']
        checks['logical_slots_4_then_budget_exhausted'] = resA.status == 'INCOMPLETE' and resA.failure_kind == 'BUDGET_EXHAUSTED' and resA.new_evaluations == 4 and fits_after_A - n_pub_fits == 12
        checks['fast_request_carries_4_public_with_feedback_and_child_view_consumer'] = client.requests[0]['candidates'] == list(PUBLIC) and 'child view' in client.requests[0]['consumer']['loss'] \
            and 'no parent/child view mixing' not in json.dumps(client.requests[0]['consumer'])
        # arm B: duplicate of a public reference rejected (no slot); X1's material hits the physical cache but costs a logical slot; other arm's ids invisible
        adB = WorkflowAdapter(td / 'armB', job, led, REPO, common=common, job_index=ji, public_ids=PUBLIC, seeds=seeds, fit_fn=fake_fit(common))
        baseB = adB.baselines()
        scriptB = [[{'tool': 'build_material', 'arguments': {'plan_id': 'Dup', 'policy': public_policy('P_AmpResample')}}],
                   [{'tool': 'build_material', 'arguments': {'plan_id': 'Y1', 'policy': plans['X1']}}, {'tool': 'evaluate', 'arguments': {'plan_id': 'Y1'}}],
                   [{'tool': 'commit', 'arguments': {'plan_id': 'Y1', 'reason': 'smoke'}}]]
        clientB = ScriptedClient({'armB': scriptB})
        resB = br.run_job(job_id=job, knowledge=br.Knowledge(), adapter=adB, client=clientB.fast('armB'), seeds=seeds, baselines=baseB, allowed_features=ALLOWED_FEATURES,
                          tool_contracts=CONTRACTS, entity_count=len(table), ca_origins=2, limits=br.Limits(wall_seconds=600, **LIMITS), evidence_roundtrip=False, max_tool_corrections=2)
        rej = [e for e in resB.trace if e['event'] == 'tool_rejected']
        regB = load_registry(td / 'armB' / job)
        checks['duplicate_public_rejected_no_slot'] = len(rej) == 1 and rej[0]['error'].get('alias_of') == 'P_AmpResample' and 'Dup' not in regB
        checks['cross_arm_cache_hit_costs_slot_not_fit'] = resB.status == 'COMPLETE' and resB.new_evaluations == 1 and fits['n'] == fits_after_A and regB['Y1']['phys_id'] == load_registry(td / 'armA' / job)['X1']['phys_id']
        checks['branch_isolation'] = sorted(regB) == sorted(list(PUBLIC) + ['Y1']) and all(c['plan_id'] in list(PUBLIC) + ['Y1'] for r in clientB.requests for c in [{'plan_id': x} for x in r['candidates']])
        cB_cells = branch_cells(td / 'armB' / job)
        checks['branch_cells_named_by_plan_point_to_physical_models'] = all(c['physical_material'] == regB[c['material_id']]['phys_id'] for c in cB_cells.values() if c['material_id'] == 'Y1') and \
            all(Path(c['model_path']).parent.parent == common / job for c in cB_cells.values())
        commit_branch(td / 'armB', DATASET, job, 'Y1', 'smoke', seeds[0])
        checks['commit_written_and_second_refused'] = (td / 'armB' / job / 'commit.json').exists()
        try:
            commit_branch(td / 'armB', DATASET, job, 'None', 'again', seeds[0])
            checks['commit_written_and_second_refused'] = False
        except RuntimeError:
            pass
        # 4. random supply: deterministic, uniform slots outside public assignments, conditional both groups, dedup
        sup1, sup2 = draw_random_supply(job, 7, table), draw_random_supply(job, 7, table)
        pub_keys = {material_key(ta.compile_plan(public_policy(m), table)['assignment']) for m in PUBLIC_STEPS}
        keys = [material_key(ta.compile_plan(p['policy'], table)['assignment']) for p in sup1['plans']]
        conds = [p for p in sup1['plans'] if p['kind'] == 'conditional']
        checks['random_supply_frozen_public_free_unique'] = sup1['plans'] == sup2['plans'] and len(set(keys)) == 4 and not (set(keys) & pub_keys) and \
            all(0 < p['entities_hit'] < len(table) for p in conds) and draw_random_supply(job, 8, table)['plans'] != sup1['plans']
        notes['random_supply_kinds'] = [p['kind'] + ('(fallback)' if p['fallback'] else '') for p in sup1['plans']]
        # random + menu decisions with fake fits (random_branch itself binds fit_physical; its selection rule is exercised here)
        adR = WorkflowAdapter(td / 'armR', job, led, REPO, common=common, job_index=ji, public_ids=PUBLIC, seeds=seeds, fit_fn=fake_fit(common))
        cands = list(adR.baselines())
        for pl in sup1['plans']:
            c = adR.build_material({'plan_id': pl['slot'], 'policy': pl['policy']}, remaining_seconds=60)
            cands.append(adR.evaluate(c, seeds, feedback=True, remaining_seconds=60))
        order = [c.plan_id for c in cands]
        chosen = min(cands, key=lambda c: (c.feedback(seeds, 2, len(table))['mean_loss'], order.index(c.plan_id)))
        checks['random_arm_argmin_over_public_and_R'] = chosen.plan_id in order and len(order) == 8
        fbs = {c.plan_id: c.feedback(seeds, 2, len(table))['mean_loss'] for c in cands[:4]}
        tie_order = sorted(fbs, key=lambda m: (fbs[m], list(PUBLIC).index(m)))
        checks['menu_argmin_public_order'] = tie_order[0] == min(fbs, key=lambda m: (fbs[m], list(PUBLIC).index(m)))
        # 5. label refusals before commit / before barrier (no row read)
        for stage_, want in (('c_b', td / 'armA'), ('score_e', td / 'armB')):
            try:
                label_prerequisite(stage_, want, job, td)
                checks['label_refusal_' + stage_] = False
            except PermissionError:
                checks['label_refusal_' + stage_] = True
        # 6. selection J(v) on synthetic numbers + empty / injected Skill rendering
        card = dsk.make_skill(skill_id='RD02-W1-r1', domain_id=DOMAIN, revision=1, workflow='Observe the batch summary first; construct at most two complete plans that test one stated hypothesis each; commit on the paired comparison.',
                              principles=None, applicability_summary='smoke card', compatibility_note='smoke', observable_applicability={'const': True}, evidence_refs=['fixture:smoke'],
                              legal_evidence_refs=['fixture:smoke'], source_stage='fixture', status='SMOKE_FIXTURE', allowed_features=ALLOWED_FEATURES, text_check=skill_text_check)
        feats = {'batch_median_missing_fraction': 0.1}
        checks['empty_vs_injected_skill'] = br.Knowledge().render(feats, ALLOWED_FEATURES)['loaded'] == [] and dsk.skill_knowledge(card).render(feats, ALLOWED_FEATURES)['loaded'] == [{'hook': 'experiment_guidance', 'body': card.rendered_body}]
        try:
            dsk.make_skill(skill_id='RD02-W2-r1', domain_id=DOMAIN, revision=1, workflow='Use the T1 batch answer table', principles=None, applicability_summary='x', compatibility_note='x',
                           observable_applicability={'const': True}, evidence_refs=['fixture:smoke'], legal_evidence_refs=['fixture:smoke'], source_stage='fixture', status='SMOKE_FIXTURE',
                           allowed_features=ALLOWED_FEATURES, text_check=skill_text_check)
            checks['skill_text_bans_job_tokens'] = False
        except (ValueError, rd.PolicyError):
            checks['skill_text_bans_job_tokens'] = True
        checks['proxy_tcp_probe_is_free'] = isinstance(proxy_reachable(), bool)
    rec = {'written_local': now(), 'checks': {k: bool(v) for k, v in checks.items()}, 'passed': sum(bool(v) for v in checks.values()), 'of': len(checks), 'notes': notes,
           'real_fits': 0, 'api_calls': 0, 'label_rows_of_planned_jobs': 0}
    rec['status'] = 'PASS' if rec['passed'] == rec['of'] else 'FAIL'
    context.write_json(out / 'smoke.json', rec)
    print('SMOKE', rec['status'], rec['passed'], '/', rec['of'], flush=True)
    return rec


def smoke_driver(root: Path = ROOT) -> dict:
    out = root / 'smoke'
    out.mkdir(parents=True, exist_ok=True)
    rc, secs = run_sub(['--worker-smoke', out], out / 'smoke_worker.log', 3600)
    rec = context.read_json(out / 'smoke.json') if (out / 'smoke.json').exists() else {'status': 'FAIL', 'checks': {}, 'worker_rc': rc}
    rec['existing_controls'] = []
    for cmd in (['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/functional/test_batch_research.py'],):
        t0 = time.time()
        pr = subprocess.run([sys.executable, '-B'] + cmd, cwd=REPO, capture_output=True, text=True, timeout=1800, env=worker_env())
        rec['existing_controls'].append({'command': ' '.join(cmd), 'returncode': pr.returncode, 'seconds': round(time.time() - t0, 1), 'tail': (pr.stdout + pr.stderr)[-400:]})
        if pr.returncode:
            rec['status'] = 'FAIL'
    rec['worker_rc'] = rc
    context.write_json(out / 'smoke.json', rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--prepare-baselines', choices=['source'])
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--stage-worker', type=Path)
    ap.add_argument('--resume-stage', choices=['source', 'select', 'target'])
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--restart', default='')
    ap.add_argument('--continue-package', action='store_true')
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--worker-build', nargs=3)
    ap.add_argument('--worker-material', nargs=4)
    ap.add_argument('--worker-fit', nargs=4)
    ap.add_argument('--tag', default='')
    ap.add_argument('--worker-label', nargs=4)
    ap.add_argument('--worker-compare', nargs=2)
    ap.add_argument('--worker-smoke')
    a = ap.parse_args()
    root = a.root.resolve()
    if a.worker_build:
        worker_build(a.worker_build[0], a.worker_build[1], int(a.worker_build[2]))
    elif a.worker_material:
        worker_material(a.worker_material[0], a.worker_material[1], a.worker_material[2], int(a.worker_material[3]))
    elif a.worker_fit:
        worker_fit(a.worker_fit[0], a.worker_fit[1], a.worker_fit[2], int(a.worker_fit[3]), a.tag)
    elif a.worker_label:
        worker_label(*a.worker_label)
    elif a.worker_compare:
        worker_compare(*a.worker_compare)
    elif a.worker_smoke:
        smoke(Path(a.worker_smoke))
    elif a.stage_worker:
        stage_worker(a.stage_worker.resolve())
    elif a.resume_stage:
        resume_stage(root, a.resume_stage, accept_unknown_usage=a.accept_unknown_usage, restart=[x for x in a.restart.split(',') if x])
        if a.continue_package:
            package_run(root)
    elif a.smoke:
        rec = smoke_driver(root)
        print('SMOKE_DRIVER', rec['status'], flush=True)
        if rec['status'] != 'PASS':
            raise SystemExit(1)
    elif a.preflight:
        print(json.dumps({k: v for k, v in preflight(root).items() if k in ('frozen_local', 'environment_matches_source_alignment', 'dataset', 'jobs')}, ensure_ascii=False)[:3000])
    elif a.wiring:
        wiring(root)
    elif a.prepare_baselines:
        prepare_baselines(root, a.prepare_baselines)
    elif a.run:
        package_run(root)
    elif a.result:
        readout(root)
    else:
        ap.error('choose an action')


if __name__ == '__main__':
    from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_skill as _canonical
    _canonical.main()
