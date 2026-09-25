"""DEV-DOMAIN-AUG-MATCHED-LAG-FEEDBACK (docs/DEV_DOMAIN_AUG_MATCHED_LAG_FEEDBACK_TASK_2026-09-21.md): for four frozen replay cases
(D01/D02 x Q03/Q04 of the decision-priority package) and their 31 already-evaluated F0 candidates, replay every candidate program
at two earlier cuts h1 = t-768 and h2 = t-384 of the SAME entities (training [h-672, h), the same Consumer / seeds / RNG rule), score
the historical models on a near block [h, h+192) and a late block [h+192, h+384) (four 48-step origins each), and compare four
deterministic selectors on the identical pool: R_CA (current C_A argmin), R_HistLate (late-block historical argmin, the new
strategy), R_HistNear (near-block control on the same historical models) and Fixed_dev (the public program frozen on other
entities). The current C_B / E of the original F0 models are connected only after the four selections are frozen.

0 LLM / HTTP, 0 Fast / Slow / Skill generation, 0 hashes, 0 commits. Reused read-only: entity_case (CaseSpec, slice loader, policy
compiler, material builder, RNG rule, fit_one, scoring); the decision-priority package (DP) for PackageLock / SafeLedger / Pools /
memory sampling; the entity-split package (ES) for the worker environment and OpenMP initialisation. New here (opt-in, thin):
the label-free pool binding from the F0 caches, the historical case instances (case_id {case}_H1/H2, case_index kept), the
historical build / fit / score workers, the selector arithmetic of task §5 and the readout of task §9.

  --preflight | --smoke | --wiring | --run | --result | --monitor [--interval S] [--once]
  subprocess entries: --worker-build ROOT CASEJSON POOLJSON | --worker-fit ROOT CASEJSON PHYS SEED [--n-updates N] |
                      --worker-histscore ROOT CASEJSON | --lock-probe ROOT
The coordinator never imports torch.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import budget, context, data, policy, spec, train, tempo_aug as ta
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_domain_aug_entity_split as ES              # read-only: worker env, OpenMP init, environment()
from evaluation.main_protocol_p4 import batch_research_domain_aug_decision_priority as DP        # read-only: lock, ledger, pools, memory sampler, split binding

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_domain_aug_matched_lag_feedback'
MODULE = 'evaluation.main_protocol_p4.batch_research_domain_aug_matched_lag_feedback'
TASK = 'docs/DEV_DOMAIN_AUG_MATCHED_LAG_FEEDBACK_TASK_2026-09-21.md'
PACKAGE = 'DEV-DOMAIN-AUG-MATCHED-LAG-FEEDBACK'
EXPOSURE = 'EXPOSED_DEVELOPMENT_REPLAY'
HIST_IDENTITY = 'historical_calibration'
F0_ROOT = DP.ROOT / 'metatest'                                # the original F0 caches (materials, cells, C_A; C_B / E only for the evaluator)
FP_ROOT = REPO / '_scratch' / 'dev_domain_aug_fixed_pool_selection'
SPLIT_DOC = DP.SPLIT_DOC                                       # docs/DOMAIN_AUG_DECISION_PRIORITY_V1.json (rosters of Q03 / Q04)
CASES = ('D01_Q03', 'D01_Q04', 'D02_Q03', 'D02_Q04')
DOMAINS = ('D01', 'D02')
LAGS = {'H1': -768, 'H2': -384}                                # h = t + lag, ascending in time
HS = ('H1', 'H2')
NEAR_OFFSETS = (0, 48, 96, 144)
LATE_OFFSETS = (192, 240, 288, 336)
BLOCKS = ('near', 'late')
HIST_RANGE = (-1440, 0)                                        # the explicit historical access range, relative to t
SEEDS = ES.SEEDS
PUBLIC = ec.PUBLIC
PUBLIC_STEPS = ec.PUBLIC_STEPS
FIXED_MIXUP_KEY = 'FixedMixup:R:w0.25:aug_seed'
STRATEGIES = ('R_CA', 'R_HistLate', 'R_HistNear', 'Fixed_dev')
TOL = 1e-12
DISPLAY_SIG = 8
WIRING_CASES = ('D01_Q03', 'D02_Q03')
FIT_THREADS = ES.FIT_THREADS
FIT_TIMEOUT_S = ES.FIT_TIMEOUT_S
TOTAL = {'max_fit_attempts': 196, 'max_llm_requests': 0, 'max_llm_tokens': 0, 'max_wall_s': 48 * 3600, 'max_retries': 4}
ALLOC = {'wiring': {'fits': 6}, 'historical': {'fits': 186}}
CONCURRENCY = {'cases': 4, 'numeric': 2, 'numeric_max': 3, 'http': 1, 'fit_threads': FIT_THREADS}
NUMERIC_WALL_WARNINGS_S = (3600, 7200)
T975_DF2 = 4.302652729911275


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def paths(root: Path) -> dict:
    root = Path(root)
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'logs': root / 'logs', 'pool': root / 'pool', 'historical': root / 'historical',
            'wiring': root / 'wiring', 'smoke': root / 'smoke', 'warnings': root / 'warnings', 'lock': root / 'package.lock',
            'selections': root / 'selections.json', 'feedback': root / 'feedback.json', 'result': root / 'result.json', 'tables': root / 'tables.md',
            'frozen': root / 'frozen_config.json', 'progress': root / 'progress.json', 'concurrency': root / 'concurrency_effective.json'}


def write_once(path, obj) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError('refusing to overwrite a frozen record: %s' % path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    context.write_json(path, obj)


def sig(x, n: int = DISPLAY_SIG):
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        return x if not math.isfinite(x) else float('%.*g' % (n, x))
    if isinstance(x, dict):
        return {k: sig(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sig(v, n) for v in x]
    return x


def fmean(v):
    return statistics.fmean(v)


def se3(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None


def stats3(v: list) -> dict:
    m = fmean(v)
    se = se3(v)
    return {'mean': m, 'se': se, 't95': [m - T975_DF2 * se, m + T975_DF2 * se] if se is not None and len(v) == 3 else None, 'per_seed': list(v),
            'signs': [int(x > 0) - int(x < 0) for x in v]}


def argmin_display(scores: dict) -> tuple:
    """Lowest value; ties within TOL broken by display id order (lexicographic C00 < C01 ...). Returns (winner, tie_group)."""
    lo = min(scores.values())
    tie = sorted(c for c, v in scores.items() if v - lo <= TOL)
    return tie[0], tie


# ============================================================================= cases (current cuts) and historical instances
def current_cases(root: Path) -> dict:
    split = context.read_json(SPLIT_DOC)
    bound = DP.bound_split(split)                               # the document's /mnt/c path bound to the spec file (identical file)
    allc = ec.cases_from_split(bound)
    return {c: allc[c] for c in CASES}


def hist_case(cs: ec.CaseSpec, H: str) -> ec.CaseSpec:
    """The historical instance of a case: same dataset / domain / roster / case_index (RNG rule), cut h = t + lag, case_id {case}_{H}.
    Its CaseSpec geometry gives train [h-672, h) and c_a = (h, h+48) = the first two near origins; the late origins equal its `e` field."""
    h = cs.t + LAGS[H]
    return ec.CaseSpec(dataset=cs.dataset, domain=cs.domain, case_id='%s_%s' % (cs.case_id, H), role='test', t=h, roster=cs.roster, case_index=cs.case_index,
                       csv_path=cs.csv_path, total_hours=cs.total_hours)


def hist_geometry(cs: ec.CaseSpec, H: str) -> dict:
    ch = hist_case(cs, H)
    h = ch.t
    near = [h + o for o in NEAR_OFFSETS]
    late = [h + o for o in LATE_OFFSETS]
    g = {'H': H, 'case_id': ch.case_id, 'parent_case': cs.case_id, 'current_t': cs.t, 'h': h, 'lag': LAGS[H], 'train_range': list(ch.train_range),
         'near_origins': near, 'near_target_union': [h, h + 192], 'late_origins': late, 'late_target_union': [h + 192, h + 384],
         'fit_rows_read': list(ch.evaluate_rows), 'score_rows_read': [h - spec.L, h + 384], 'allowed_history': [cs.t + HIST_RANGE[0], cs.t + HIST_RANGE[1]],
         'identity': HIST_IDENTITY}
    lo, hi = g['allowed_history']
    if not (lo <= ch.train_range[0] and h + 384 <= hi and h + 384 <= cs.t):
        raise RuntimeError('historical geometry leaves the allowed range for %s' % ch.case_id)
    if list(ch.c_a) != near[:2] or list(ch.e) != late:
        raise RuntimeError('historical CaseSpec origins disagree with the near / late blocks')
    return g


def hist_dir(root: Path, case: str, H: str) -> Path:
    """run_dir of the historical instance; the case directory inside it is {case}_{H}."""
    return paths(root)['historical'] / ('%s_%s' % (case, H))


def case_json_path(root: Path, case: str, H: str) -> Path:
    return hist_dir(root, case, H) / 'case.json'


def write_case_json(root: Path, cs: ec.CaseSpec, H: str) -> Path:
    p = case_json_path(root, cs.case_id, H)
    ch = hist_case(cs, H)
    rec = {'spec': ch.to_json(), 'geometry': hist_geometry(cs, H), 'package': PACKAGE, 'identity': HIST_IDENTITY, 'exposure': EXPOSURE}
    if p.exists():
        if context.read_json(p)['spec'] != rec['spec']:
            raise RuntimeError('historical case.json drifted for %s' % ch.case_id)
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    context.write_json(p, rec)
    return p


def load_case_json(p: Path) -> ec.CaseSpec:
    return ec.case_from_json(context.read_json(p)['spec'])


# ============================================================================= label-free pool binding (task §2: clean_inputs + F0 legal fields only)
def f0_dirs(case: str) -> tuple:
    return F0_ROOT / ('%s_f0' % case) / case, F0_ROOT / ('%s_common' % case) / case


def semantic_key(mid: str, rec: dict) -> str:
    if mid == 'FixedMixup':
        return FIXED_MIXUP_KEY
    if mid == 'None':
        return ec.material_key([[] for _ in range(ec.COHORT_SIZE)])
    if mid in PUBLIC_STEPS:
        return ec.material_key([PUBLIC_STEPS[mid]] * ec.COHORT_SIZE)
    return rec['key']


def block_from_cells(cells: dict, plan_ids: list, block: str) -> dict:
    """Three-seed summary of one score block from cell records ({cell_id: {'material_id', 'model_seed', <block>: score}} or fit records)."""
    vals = {}
    for r in cells.values():
        if r['material_id'] in plan_ids:
            s = r['scores'][block] if 'scores' in r else r[block]
            vals[r['model_seed']] = s
    if set(vals) != set(SEEDS):
        raise RuntimeError('block %s: seeds %s found for %s' % (block, sorted(vals), plan_ids))
    if any(vals[s]['status'] != 'SCORABLE' for s in SEEDS):
        raise RuntimeError('block %s not scorable for %s' % (block, plan_ids))
    mats = [[list(map(float, row)) for row in vals[s]['per_origin_entity_normalized_mse']] for s in SEEDS]
    return block_summary(mats)


def block_summary(mats: list) -> dict:
    n_o, n_e = len(mats[0]), len(mats[0][0])
    by_seed_origin = [[fmean(row) for row in m] for m in mats]
    by_seed = [fmean(x) for x in by_seed_origin]
    by_origin = [fmean(by_seed_origin[s][o] for s in range(len(mats))) for o in range(n_o)]
    per_entity = [fmean(mats[s][o][e] for s in range(len(mats)) for o in range(n_o)) for e in range(n_e)]
    return {'by_seed': by_seed, 'by_seed_origin': by_seed_origin, 'by_origin': by_origin, 'mean': fmean(by_seed), 'per_entity_mean': per_entity, 'n_origins': n_o, 'n_entities': n_e}


def clean_policy(pol: dict) -> dict:
    """The frozen program definition: default, rules, observation fields; the constructor's rationale is not part of the program."""
    return {'default': {'steps': copy.deepcopy(pol['default']['steps'])}, 'rules': [{'when': copy.deepcopy(r['when']), 'steps': copy.deepcopy(r['steps'])} for r in pol.get('rules', []) or []],
            'observation_fields_used': list(pol.get('observation_fields_used', []) or [])}


def policy_of(mid: str, rec: dict, bdir: Path) -> dict | None:
    if mid == 'FixedMixup':
        return None
    if mid in PUBLIC_STEPS:
        return ec.uniform_policy(PUBLIC_STEPS[mid], '')
    comp = rec.get('compiled')
    if comp is None:
        p = bdir / 'aug_materials' / (mid + '__compiled.json')
        if not p.exists():
            raise RuntimeError('original policy missing for evaluated plan %s' % mid)
        comp = {'policy': context.read_json(p)['material_spec']['policy']}
    return clean_policy(comp['policy'])


def bind_pool(case: str, cs: ec.CaseSpec) -> dict:
    """The F0 pool of one case WITHOUT any later-block label: registry + OK untagged fit records (C_A only) + the T-only overview; canonical
    display ids by lexicographic semantic key (the fixed_pool rule). Checked against the label-free clean_inputs packet."""
    bdir, common = f0_dirs(case)
    cs_rec = context.read_json(bdir / 'case_spec.json')
    if cs_rec['case_id'] != case or int(cs_rec['t']) != cs.t or list(cs_rec['roster']) != list(cs.roster) or cs_rec['dataset'] != cs.dataset:
        raise RuntimeError('F0 case_spec disagrees with the frozen split for %s' % case)
    reg = ec.load_registry(bdir)
    cells = ec.branch_cells(bdir)
    table = context.read_json(bdir / 'overview.json')['entities']
    by_phys = {}
    for mid, rec in reg.items():
        mine = {r['model_seed']: r for r in cells.values() if r['material_id'] == mid}
        if not mine:
            continue                                            # built, never evaluated: not a candidate
        if set(mine) != set(SEEDS):
            raise RuntimeError('%s %s evaluated on %d/3 seeds' % (case, mid, len(mine)))
        for s, r in mine.items():
            if r['job_id'] != case or list(r['roster']) != list(cs.roster) or (mid != 'FixedMixup' and r.get('material_key') != rec['key']) or list(r['rows_read']) != list(cs.evaluate_rows):
                raise RuntimeError('cell binding conflict %s %s seed %d' % (case, mid, s))
        phys = rec.get('phys_id', mid)
        pol = policy_of(mid, rec, bdir)
        row = by_phys.get(phys)
        if row is None:
            assignment = 'FixedMixup' if mid == 'FixedMixup' else ([PUBLIC_STEPS[mid]] * ec.COHORT_SIZE if mid in PUBLIC_STEPS else rec['assignment'])
            row = by_phys[phys] = {'phys_f0': phys, 'plan_ids': [], 'public_id': mid if mid in PUBLIC else None, 'key': semantic_key(mid, rec),
                                   'label': ec.assignment_label(assignment, mid if mid in PUBLIC else None), 'assignment_t': assignment, 'policy': pol,
                                   'program_source_plan': mid, 'c_a': block_from_cells(cells, [mid], 'c_a'), 'kind': rec.get('kind'), 'summary_path_f0': rec.get('summary_path')}
            if pol is not None and mid not in PUBLIC_STEPS:
                comp = rec.get('compiled') or {}
                row['resolved_thresholds_t'] = comp.get('resolved_thresholds', {})
                row['rule_index_t'] = comp.get('rule_index')
                row['n_unknown_t'] = comp.get('n_unknown', 0)
                # task §4.1: the current-t compilation of the frozen policy must reproduce the old assignment (construction only, 0 fits)
                again = ec.compile_plan_full(pol, table)
                row['current_t_recompile_ok'] = again['assignment'] == rec['assignment'] and again['resolved_thresholds'] == row['resolved_thresholds_t']
                if not row['current_t_recompile_ok']:
                    raise RuntimeError('current-t recompilation does not reproduce the F0 assignment for %s %s' % (case, mid))
        row['plan_ids'].append(mid)
        if mid in PUBLIC:
            row['public_id'] = mid
    missing = [p for p in PUBLIC if not any(r['public_id'] == p for r in by_phys.values())]
    if missing:
        raise RuntimeError('public references missing from the evaluated pool of %s: %s' % (case, missing))
    order = sorted(by_phys.values(), key=lambda r: r['key'])
    pool = {'C%02d' % i: {**r, 'display_id': 'C%02d' % i} for i, r in enumerate(order)}
    ref_ids = {r['public_id']: cid for cid, r in pool.items() if r['public_id']}
    # cross-check with the fixed_pool clean packet (label-free by construction)
    packet = context.read_json(FP_ROOT / 'clean_inputs' / 'replay' / (case + '.json'))
    if packet['candidate_ids'] != sorted(pool) or packet['reference_ids'] != ref_ids:
        raise RuntimeError('pool ids differ from the clean packet for %s' % case)
    for cid, r in pool.items():
        pc = packet['candidates'][cid]
        if pc['label'] != r['label'] or sig(pc['ca']['mean']) != sig(r['c_a']['mean']) or sig(pc['ca']['by_seed']) != sig(r['c_a']['by_seed']):
            raise RuntimeError('candidate %s of %s differs from the clean packet (label / C_A)' % (cid, case))
        if r['policy'] is not None and pc['program'].get('policy') is not None:
            pp = pc['program']['policy']
            if pp['default'] != r['policy']['default'] or [dict(x) for x in pp.get('rules', [])] != r['policy']['rules']:
                raise RuntimeError('policy of %s %s differs from the clean packet' % (case, cid))
    return {'case': case, 'domain': cs.domain, 'branch_dir_f0': str(bdir), 'common_dir_f0': str(common), 'pool': pool, 'pool_size': len(pool), 'reference_ids': ref_ids,
            'order_rule': 'lexicographic on the semantic material key (fixed_pool rule); independent of any score', 'labels_read': 'none (C_A of the F0 fit records only)',
            'overview_t_entities': table, 'bound_local': now()}


def load_pool(root: Path, case: str) -> dict:
    return context.read_json(paths(root)['pool'] / (case + '.json'))


# ============================================================================= preflight (0 fits; reads only the authorized historical rows for eligibility)
def environment() -> dict:
    env = ES.environment()
    env['fit_parallel'] = 'numeric pool of this package (%d default, %d max)' % (CONCURRENCY['numeric'], CONCURRENCY['numeric_max'])
    return env


def data_readiness(cs: ec.CaseSpec) -> dict:
    """Finite values, hourly timestamps, non-degenerate scaler and 16 x 433 windows at each h, inside [t-1440, t) of the roster only."""
    lo, hi = cs.t + HIST_RANGE[0], cs.t + HIST_RANGE[1]
    sl = ec.load_case_slice(cs, lo, hi)                          # raises on non-finite / non-hourly rows
    out = {'rows_read': [lo, hi], 'finite': bool(np.isfinite(sl.values).all()), 'hourly': True, 'columns_in_file': sl.columns_in_file, 'columns_converted': ec.COHORT_SIZE, 'h': {}}
    for H in HS:
        ch = hist_case(cs, H)
        sc = data.compute_scaler(sl, ch.t)
        P = data.build_parents(sl, ch.t, sc)
        out['h'][H] = {'h': ch.t, 'floor_hits': sc.floor_hits, 'min_std': float(sc.std.min()), 'n_parents': int(P.n_parents), 'pool': int(P.X_norm.shape[0] * P.n_parents),
                       'ok': sc.floor_hits == 0 and P.n_parents == spec.N_PARENTS and P.X_norm.shape == (ec.COHORT_SIZE, spec.N_PARENTS, spec.L)}
    out['ok'] = out['finite'] and all(v['ok'] for v in out['h'].values())
    return out


def fixed_dev_of(domain: str) -> dict:
    fr = context.read_json(FP_ROOT / 'freeze' / (domain + '.json'))
    return {'public_id': fr['fixed_dev']['public_id'], 'selected_on': fr['cases'], 'source': str(FP_ROOT / 'freeze' / (domain + '.json')), 'skill_text_imported': False}


def preflight(root: Path = ROOT) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    P_ = paths(root)
    if P_['frozen'].exists():
        return context.read_json(P_['frozen'])
    cases = current_cases(root)
    P_['pool'].mkdir(exist_ok=True)
    pools_, geometry, readiness, hist_specs = {}, {}, {}, {}
    for c, cs in cases.items():
        pp = P_['pool'] / (c + '.json')
        if not pp.exists():
            context.write_json(pp, bind_pool(c, cs))
        pools_[c] = context.read_json(pp)
        geometry[c] = {H: hist_geometry(cs, H) for H in HS}
        readiness[c] = data_readiness(cs)
        if not readiness[c]['ok']:
            raise RuntimeError('data readiness failed for %s: %s' % (c, readiness[c]))
        for H in HS:
            write_case_json(root, cs, H)
            hist_specs['%s_%s' % (c, H)] = hist_case(cs, H).to_json()
    n_cand = sum(p['pool_size'] for p in pools_.values())
    if n_cand != 31:
        raise RuntimeError('expected 31 candidates, bound %d' % n_cand)
    fixed = {d: fixed_dev_of(d) for d in DOMAINS}
    cfg = {'package': PACKAGE, 'task': TASK, 'exposure': EXPOSURE, 'historical_identity': HIST_IDENTITY, 'split_document': str(SPLIT_DOC), 'f0_root': str(F0_ROOT), 'fixed_pool_root': str(FP_ROOT),
           'cases': {c: cs.to_json() for c, cs in cases.items()}, 'historical_instances': hist_specs, 'geometry': geometry, 'lags': LAGS, 'near_offsets': list(NEAR_OFFSETS), 'late_offsets': list(LATE_OFFSETS),
           'historical_access_range_relative_to_t': list(HIST_RANGE), 'data_readiness': readiness,
           'pools': {c: {'pool_size': p['pool_size'], 'reference_ids': p['reference_ids'], 'candidates': {cid: {'label': r['label'], 'public_id': r['public_id'], 'plan_ids': r['plan_ids'], 'phys_f0': r['phys_f0'],
                                                                                                                   'policy': r['policy'], 'c_a_mean': r['c_a']['mean']} for cid, r in p['pool'].items()}} for c, p in pools_.items()},
           'n_candidates': n_cand, 'fixed_dev': fixed, 'seeds': list(SEEDS), 'consumer': ec.CONSUMER, 'cohort_size': ec.COHORT_SIZE,
           'seed_rule': 'entity_program_seed(domain, ORIGINAL case_index, program_index, entity_column) - identical stream definition at t, h1, h2; FixedMixup RandomState(900090 + entity_index)',
           'training': {'train_span': spec.TRAIN_SPAN, 'n_updates': spec.N_UPDATES, 'parent_batch': spec.PARENT_BATCH, 'fit_threads': FIT_THREADS, 'warm_start': False, 'scaler': 'per h from [h-672, h) only',
                        'child_weight': '0.5 parent / 0.5 child; None parent-only', 'timeout_s': FIT_TIMEOUT_S},
           'selectors': {'L': 'L(j,h,m,b) = mean over 3 seeds of normalized MSE (entity macro, 4 origins equal) of candidate m at cut h on block b',
                         'A': 'A(j,h,m,b) = L(j,h,m,b) / max(L(j,h,None,b), 1e-12)', 'J_hist': 'J_hist(j,m,b) = mean over h1, h2 of A',
                         'R_CA': 'argmin over the pool of the three-seed mean current C_A (full precision F0 cells)', 'R_HistLate': 'argmin J_hist(j, m, late)',
                         'R_HistNear': 'argmin J_hist(j, m, near)', 'Fixed_dev': 'the fixed_pool freeze fixed_dev.public_id of the domain', 'ties': '<= 1e-12 -> display id order; no preset priority'},
           'readout': {'G': 'G_j(A over B) = 100 x [mean_seed E_j(B) - mean_seed E_j(A)] / mean_seed E_j(None); positive = A better; pp of None',
                       'aggregation': 'domain = equal weight of its two cases; overall = equal weight of the two domains', 'main_block': 'E; C_B attached'},
           'budget': {'total': TOTAL, 'allocation': ALLOC, 'concurrency_planned': CONCURRENCY, 'numeric_wall_warnings_s': list(NUMERIC_WALL_WARNINGS_S), 'llm_requests': 0, 'http': 0, 'sha': 0, 'commit': 0},
           'environment': environment(), 'frozen_local': now(), 'no_sha': True}
    write_once(P_['frozen'], cfg)
    return cfg


# ============================================================================= ledger, caps, warnings
def stage_caps(root: Path, stage: str) -> dict:
    P_ = paths(root)
    rec = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    if stage in rec:
        return rec[stage]
    led = DP.SafeLedger(P_['ledger'], **TOTAL)
    s = led.s
    caps = {'max_fit_attempts': min(TOTAL['max_fit_attempts'], s['fit_attempts'] + ALLOC[stage]['fits'] + (TOTAL['max_retries'] - s['retries_used'])),
            'max_llm_requests': 0, 'max_llm_tokens': 0, 'max_wall_s': TOTAL['max_wall_s'], 'max_retries': TOTAL['max_retries']}
    rec[stage] = {'epoch_start': time.time(), 'local_start': now(), 'allocation': ALLOC[stage], 'snapshot_at_start': {k: s.get(k, 0) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'fit_wall_seconds')},
                  'caps': caps}
    context.write_json(P_['stages'], rec)
    return rec[stage]


def numeric_warnings(root: Path, where: str) -> list:
    P_ = paths(root)
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    st = led.get('historical_started_epoch')
    if not st:
        return []
    el = time.time() - st
    P_['warnings'].mkdir(parents=True, exist_ok=True)
    out = []
    for th in NUMERIC_WALL_WARNINGS_S:
        p = P_['warnings'] / ('numeric_wall_%dh.json' % (th // 3600))
        if el >= th and not p.exists():
            context.write_json(p, {'threshold_s': th, 'elapsed_s': el, 'where': where, 'local': now(), 'action': 'progress / remaining estimate only; no data, candidate or seed change'})
            out.append(str(p))
            print('NUMERIC_WALL_WARNING', p, flush=True)
    return out


# ============================================================================= workers (subprocess entries; torch allowed)
def _init_worker() -> None:
    ES._init_worker()


def worker_build(root: str, case_json: str, pool_json: str) -> None:
    """One historical instance: T-only overview at h, the public materials, then every constructed candidate compiled on the h table and
    built (or aliased when its h-assignment equals an existing physical material). Single writer of the instance registry."""
    _init_worker()
    ch = load_case_json(Path(case_json))
    run_dir = Path(case_json).parent
    pool = context.read_json(pool_json)['pool']
    ctx = ec.open_case(ch, run_dir, stage='material')
    if ctx.slice.row_start != ch.train_range[0] or ctx.slice.row_end != ch.train_range[1]:
        raise RuntimeError('material stage slice is not [h-672, h)')
    t0 = time.time()
    ec.build_public(ctx)
    table = ctx.overview()['entities']
    reg = ec.load_registry(ctx.job_dir)
    used = [int(r['material_index']) for r in reg.values() if r.get('material_index')]
    nxt = max(used + [0]) + 1
    out = {'case_id': ch.case_id, 'h': ch.t, 'identity': HIST_IDENTITY, 'rows_read_material': list(ch.train_range), 'candidates': {}, 'built_local': now()}
    for cid in sorted(pool):
        r = pool[cid]
        rec = {'display_id': cid, 'label_t': r['label'], 'public_id': r['public_id'], 'phys_f0': r['phys_f0']}
        if r['public_id'] == 'FixedMixup':
            rec.update({'phys_h': 'FixedMixup', 'alias_of': None, 'assignment_h': 'FixedMixup', 'label_h': 'FixedMixup', 'assignment_changed': False, 'key_h': FIXED_MIXUP_KEY})
        elif r['public_id'] == 'None':
            rec.update({'phys_h': 'None', 'alias_of': None, 'assignment_h': [[] for _ in range(ec.COHORT_SIZE)], 'label_h': 'None', 'assignment_changed': False, 'key_h': semantic_key('None', {})})
        else:
            comp = ec.compile_plan_full(r['policy'], table)
            asg = comp['assignment']
            key = ec.material_key(asg)
            rec.update({'assignment_h': asg, 'key_h': key, 'label_h': ec.assignment_label(asg, r['public_id']), 'rule_index_h': comp['rule_index'], 'resolved_thresholds_h': comp['resolved_thresholds'],
                        'n_unknown_h': comp['n_unknown'], 'assignment_changed': asg != r['assignment_t'],
                        'entities_changed': [e for e in range(ec.COHORT_SIZE) if asg[e] != r['assignment_t'][e]],
                        'resolved_thresholds_t': r.get('resolved_thresholds_t', {}), 'rule_index_t': r.get('rule_index_t')})
            reg = ec.load_registry(ctx.job_dir)
            if r['public_id']:
                if reg[r['public_id']]['key'] != key:
                    raise RuntimeError('public material %s key drift at h' % r['public_id'])
                rec.update({'phys_h': r['public_id'], 'alias_of': None})
            else:
                same = [m for m, rr in reg.items() if rr['key'] == key and not rr.get('alias_of')]
                if same:
                    rq = ec.aug_dir(ctx.job_dir) / (same[0] + '__request.json')
                    own = rq.exists() and context.read_json(rq).get('display_id') == cid          # resume: this candidate's own material already built
                    rec.update({'phys_h': same[0], 'alias_of': None if own else same[0]})
                elif ec.is_identity(asg):
                    rec.update({'phys_h': 'None', 'alias_of': 'None'})
                else:
                    phys = 'TA%03d' % nxt
                    nxt += 1
                    context.write_json(ec.aug_dir(ctx.job_dir) / (phys + '__request.json'), {'assignment': asg, 'display_id': cid, 'requested_local': now(), 'package': PACKAGE})
                    ec.build_material(ctx, asg, phys)
                    context.write_json(ec.aug_dir(ctx.job_dir) / (phys + '__compiled.json'), {'display_id': cid, 'policy': r['policy'], 'assignment': asg, 'key': key, 'rule_index': comp['rule_index'],
                                                                                                'resolved_thresholds': comp['resolved_thresholds'], 'n_unknown': comp['n_unknown'], 'phys_id': phys})
                    rec.update({'phys_h': phys, 'alias_of': None})
        out['candidates'][cid] = rec
    out['physical_materials'] = sorted({r['phys_h'] for r in out['candidates'].values()})
    out['build_seconds_total'] = time.time() - t0
    context.write_json(ctx.job_dir / 'hist_pool.json', out)
    print('BUILD_OK', ch.case_id, len(out['candidates']), 'physical', len(out['physical_materials']), 'secs %.1f' % out['build_seconds_total'], flush=True)


def worker_fit(root: str, case_json: str, phys: str, seed: int, n_updates=None) -> None:
    _init_worker()
    ch = load_case_json(Path(case_json))
    run_dir = Path(case_json).parent
    rec = ec.fit_one(ch, run_dir, phys, int(seed), threads=FIT_THREADS, timeout_s=FIT_TIMEOUT_S - 30, n_updates=n_updates)
    print('CELL_OK', rec['cell_id'], round(rec['train']['seconds'], 2), 'near2 %.5f' % rec['scores']['c_a']['normalized_mse_macro'], flush=True)


def worker_histscore(root: str, case_json: str) -> None:
    """Every OK cell of the instance: the two cached C_A predictions (origins h, h+48) are reused, the other six origins are inferred once; the near and
    late blocks are scored with the instance's frozen T scaler / MASE denominators. Rows read: [h-192, h+384) only (all before the current t)."""
    _init_worker()
    ch = load_case_json(Path(case_json))
    run_dir = Path(case_json).parent
    jd = run_dir / ch.case_id
    sc = np.load(jd / 'scaler.npz')
    if [str(x) for x in sc['roster']] != list(ch.roster) or int(sc['t']) != ch.t or str(sc['dataset']) != ch.dataset:
        raise RuntimeError('scaler binding drift for %s' % ch.case_id)
    scaler = data.Scaler(mean=sc['mean'], std=sc['std'], scale=sc['scale'], floor_hits=0)
    mase = sc['mase']
    h = ch.t
    lo, hi = h - spec.L, h + 384
    sl = ec.load_case_slice(ch, lo, hi)
    near = [h + o for o in NEAR_OFFSETS]
    late = [h + o for o in LATE_OFFSETS]
    wins = {o: w for o, w in zip(near + late, data.build_windows(sl, near + late, scaler, with_truth=True))}
    cells = ec.branch_cells(jd)
    reg = ec.load_registry(jd)
    out = {'case_id': ch.case_id, 'h': h, 'identity': HIST_IDENTITY, 'rows_read': [lo, hi], 'near_origins': near, 'late_origins': late, 'scored_local': now(), 'cells': {},
           'metric': 'normalized MSE (frozen T scaler of this h), entity macro, origins equal-weighted inside a block', 'reused_origins_per_model': 2, 'inferred_origins_per_model': 6}
    t0 = time.time()
    n_inf = 0
    for c, r in cells.items():
        if r['job_id'] != ch.case_id or list(r['rows_read']) != list(ch.evaluate_rows) or list(r['roster']) != list(ch.roster):
            raise RuntimeError('cell %s does not belong to this historical instance' % c)
        with np.load(jd / 'pred_c_a' / (c + '.npz')) as z:
            if list(z['origins']) != near[:2]:
                raise RuntimeError('cached C_A origins differ for %s' % c)
            pred_ca = z['pred_raw'].copy()
        model = train.load_model(r['model_path'])
        rest = near[2:] + late
        pred_rest = train.predict_windows(model, [wins[o] for o in rest], scaler)
        n_inf += len(rest)
        pred = {o: pred_ca[:, i, :] for i, o in enumerate(near[:2])}
        pred.update({o: pred_rest[:, i, :] for i, o in enumerate(rest)})
        rec = {'material_id': r['material_id'], 'material_key': r['material_key'], 'model_seed': r['model_seed'], 'reused_c_a_origins': near[:2], 'inferred_origins': rest}
        for b, origins in (('near', near), ('late', late)):
            p = np.stack([pred[o] for o in origins], axis=1)
            truth = np.stack([wins[o].y_true_raw for o in origins], axis=1)
            rec[b] = ec.score_with_status(p, truth, scaler, mase)
        if rec['near']['status'] != 'SCORABLE' or rec['late']['status'] != 'SCORABLE':
            raise RuntimeError('historical block not scorable for %s' % c)
        out['cells'][c] = rec
    out['n_models'] = len(cells)
    out['n_origins_inferred'] = n_inf
    out['n_origins_reused'] = 2 * len(cells)
    out['seconds'] = time.time() - t0
    out['physical_materials'] = sorted({r['material_id'] for r in out['cells'].values()})
    (jd / 'hist_scores').mkdir(exist_ok=True)
    for c, rec in out['cells'].items():
        context.write_json(jd / 'hist_scores' / (c + '.json'), rec)
    context.write_json(jd / 'hist_scores.json', out)
    print('HISTSCORE_OK', ch.case_id, 'models', len(cells), 'inferred', n_inf, 'secs %.1f' % out['seconds'], flush=True)


# ============================================================================= coordinator plumbing (never imports torch)
def run_worker(args: list, log: Path, timeout: float) -> tuple:
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with log.open('w', encoding='utf-8') as fh:
            p = subprocess.run([sys.executable, '-B', '-m', MODULE] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, env=ES.worker_env())
        return p.returncode, time.time() - t0
    except subprocess.TimeoutExpired:
        return -999, time.time() - t0


def cell_cached(cp: Path, ch: ec.CaseSpec, phys: str, key: str, n_updates=None) -> dict | None:
    """An OK record bound to THIS historical instance (case_id, rows, roster, key, model on disk); a record of another cut is a conflict, never a hit."""
    if not cp.exists():
        return None
    r = context.read_json(cp)
    if r.get('status') != 'OK':
        return None
    ok = (r['job_id'] == ch.case_id and list(r['rows_read']) == list(ch.evaluate_rows) and list(r['roster']) == list(ch.roster) and r['material_id'] == phys
          and (phys == 'FixedMixup' or r['material_key'] == key) and r.get('dataset') == ch.dataset and Path(r['model_path']).exists() and r['n_updates'] == (n_updates or spec.N_UPDATES))
    if not ok:
        raise RuntimeError('cached cell %s is not bound to %s (another cut / roster / material / training length)' % (cp.name, ch.case_id))
    return r


def fit_cells(ledger, root: Path, case_json: Path, ch: ec.CaseSpec, jobs: list, *, n_updates=None) -> dict:
    """jobs: [(phys, seed)] of ONE historical instance; reserved-before-launch physical fits inside the numeric pool, at most one same-configuration
    retry per cell from the package cap; returns {cell_id: {'record', 'fit_started', 'attempts'}}."""
    run_dir = case_json.parent
    jd = run_dir / ch.case_id
    reg = ec.load_registry(jd)
    out, pending = {}, []
    for phys, seed in jobs:
        cid = ec.cell_id(ch.case_id, phys, seed)
        cp = jd / 'cells' / (cid + '.json')
        rec = cell_cached(cp, ch, phys, reg[phys]['key'], n_updates)
        if rec is not None:
            out[cid] = {'record': rec, 'fit_started': False, 'attempts': 0}
            continue
        pending.append((phys, seed, cid, cp))
    errors = []

    def run_one(phys, seed, cid, cp):
        try:
            for attempt in (0, 1):
                log = jd / 'fit_logs' / (cid + ('.log' if attempt == 0 else '.retry.log'))
                log.parent.mkdir(parents=True, exist_ok=True)
                args = ['--worker-fit', root, case_json, phys, seed] + (['--n-updates', n_updates] if n_updates else [])
                with DP.pools().numeric():
                    ledger.reserve_fit(cid)
                    timeout = min(FIT_TIMEOUT_S, ledger.remaining())
                    t0 = time.time()
                    rc, _ = run_worker(args, log, timeout)
                    ok = rc == 0 and cp.exists() and context.read_json(cp).get('status') == 'OK'
                    reason = '' if ok else ('worker_timeout' if rc == -999 else 'worker_failed')
                    ledger.finish_fit(cid, ok, time.time() - t0, reason)
                print('FIT', cid, 'OK' if ok else reason, round(time.time() - t0, 1), flush=True)
                if ok:
                    rec = context.read_json(cp)
                    if rec['scores']['c_a']['n_nonfinite_predictions']:
                        raise RuntimeError('nonfinite predictions')
                    out[cid] = {'record': rec, 'fit_started': True, 'attempts': attempt + 1}
                    return
                if attempt == 0 and rt.FIT_RETRY and ledger.can_retry():
                    with ledger.lock:
                        ledger.s['retries_used'] += 1
                        ledger.s['events'].append({'kind': 'fit_retry', 'cell': cid, 'reason': reason, 'epoch': time.time()})
                        ledger._save()
                    continue
                raise RuntimeError(reason)
        except Exception as exc:  # noqa: BLE001
            errors.append((cid, exc))

    threads = [threading.Thread(target=run_one, args=j, name='fit:' + j[2], daemon=True) for j in pending]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError('fit failed: %s (%s)' % (errors[0][0], errors[0][1]))
    return out


class Progress:
    def __init__(self, path: Path):
        self.path, self.lock, self.state = Path(path), threading.Lock(), {'items': {}, 'started_local': now()}
        if self.path.exists():
            self.state = context.read_json(self.path)

    def set(self, item: str, **kw) -> None:
        with self.lock:
            self.state['items'].setdefault(item, {}).update(kw)
            self.state['updated_local'] = now()
            self.state['pools'] = DP.pools().snapshot()
            context.write_json(self.path, self.state)


def effective_numeric(root: Path) -> int:
    p = paths(root)['concurrency']
    return int(context.read_json(p)['concurrency']['numeric']) if p.exists() else CONCURRENCY['numeric']


def run_item(root: Path, ledger, case: str, H: str, progress: Progress, *, pool_json: Path, n_updates=None) -> dict:
    """One historical instance end to end: build (0 fits) -> fits (pool candidates x seeds) -> historical scoring (0 fits). Idempotent."""
    item = '%s_%s' % (case, H)
    cj = case_json_path(root, case, H)
    ch = load_case_json(cj)
    jd = cj.parent / ch.case_id
    hp = jd / 'hist_pool.json'
    t_item = time.time()
    if not hp.exists():
        progress.set(item, status='building', started_local=now())
        t0 = time.time()
        with DP.pools().numeric():
            rc, _ = run_worker(['--worker-build', root, cj, pool_json], paths(root)['logs'] / ('build_%s.log' % item), 1800)
        ledger.add('material_wall_seconds', time.time() - t0)
        if rc != 0 or not hp.exists():
            raise RuntimeError('build worker failed for %s' % item)
    hpool = context.read_json(hp)
    phys = hpool['physical_materials']
    jobs = [(p, s) for p in phys for s in SEEDS]
    progress.set(item, status='fitting', n_physical=len(phys), n_cells=len(jobs))
    res = fit_cells(ledger, root, cj, ch, jobs, n_updates=n_updates)
    n_new = sum(1 for v in res.values() if v['fit_started'])
    hs = jd / 'hist_scores.json'
    if not hs.exists() or set(context.read_json(hs)['cells']) != set(res):
        progress.set(item, status='scoring', fits_new=n_new)
        t0 = time.time()
        with DP.pools().numeric():
            rc, _ = run_worker(['--worker-histscore', root, cj], paths(root)['logs'] / ('histscore_%s.log' % item), 1800)
        ledger.add('histscore_wall_seconds', time.time() - t0)
        if rc != 0 or not hs.exists():
            raise RuntimeError('historical scoring worker failed for %s' % item)
    sc = context.read_json(hs)
    progress.set(item, status='done', fits_new=n_new, cells=len(res), models_scored=sc['n_models'], origins_inferred=sc['n_origins_inferred'], item_wall_s=time.time() - t_item, finished_local=now())
    return {'item': item, 'fits_new': n_new, 'cells': len(res), 'physical': len(phys), 'models_scored': sc['n_models'], 'origins_inferred': sc['n_origins_inferred'], 'item_wall_s': time.time() - t_item}


def historical_stage(root: Path = ROOT, *, cases=CASES, n_updates=None, numeric: int | None = None) -> dict:
    """All (case, H) items with up to CONCURRENCY['cases'] active item threads; every process runs inside the one numeric pool."""
    P_ = paths(root)
    sc = stage_caps(root, 'historical')
    led = DP.SafeLedger(P_['ledger'], **sc['caps'])
    rt.FIT_RETRY = True
    DP.set_pools(numeric or effective_numeric(root), CONCURRENCY['http'])
    if led.s.get('historical_started_epoch') is None:
        led.s['historical_started_epoch'] = time.time()
        led.s['historical_started_local'] = now()
        led._save()
    P_['logs'].mkdir(exist_ok=True)
    progress = Progress(P_['progress'])
    items = [(c, H) for c in cases for H in HS]
    gate = threading.BoundedSemaphore(CONCURRENCY['cases'])
    results, errors = {}, []

    def run(c, H):
        with gate:
            try:
                results['%s_%s' % (c, H)] = run_item(root, led, c, H, progress, pool_json=P_['pool'] / (c + '.json'), n_updates=n_updates)
            except Exception as exc:  # noqa: BLE001
                errors.append(('%s_%s' % (c, H), repr(exc)))
                progress.set('%s_%s' % (c, H), status='failed', error=repr(exc)[:300])
            numeric_warnings(root, 'historical')
    ts = [threading.Thread(target=run, args=i, name='item:%s_%s' % i, daemon=True) for i in items]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    led.s['historical_finished_local'] = now()
    led.s['historical_finished_epoch'] = time.time()
    led.s['pools_snapshot'] = DP.pools().snapshot()
    led._save()
    if errors:
        raise RuntimeError('historical stage: %d item(s) failed: %s' % (len(errors), errors[:3]))
    return {'items': results, 'ledger': {k: v for k, v in led.s.items() if k != 'events'}, 'pools': DP.pools().snapshot()}


# ============================================================================= selectors (task §5) - frozen before any current C_B / E is read
def hist_scores_of(root: Path, case: str) -> dict:
    """{H: {'pool': hist_pool, 'cells': hist_scores cells}} of one case."""
    out = {}
    for H in HS:
        jd = hist_dir(root, case, H) / ('%s_%s' % (case, H))
        out[H] = {'pool': context.read_json(jd / 'hist_pool.json'), 'scores': context.read_json(jd / 'hist_scores.json')}
    return out


def candidate_hist_blocks(hist: dict, cid: str) -> dict:
    """{H: {'near': block_summary, 'late': block_summary, 'phys_h', 'alias_of'}} of one display id (aliases share the physical model)."""
    out = {}
    for H, rec in hist.items():
        cand = rec['pool']['candidates'][cid]
        phys = cand['phys_h']
        cells = {c: v for c, v in rec['scores']['cells'].items() if v['material_id'] == phys}
        out[H] = {'phys_h': phys, 'alias_of': cand.get('alias_of'), 'label_h': cand['label_h'], 'assignment_changed': cand['assignment_changed']}
        for b in BLOCKS:
            out[H][b] = block_from_cells(cells, [phys], b)
    return out


def selections_of_case(root: Path, case: str, pool: dict, fixed_id: str) -> dict:
    hist = hist_scores_of(root, case)
    none_id = pool['reference_ids']['None']
    blocks = {cid: candidate_hist_blocks(hist, cid) for cid in pool['pool']}
    L = {cid: {H: {b: blocks[cid][H][b]['mean'] for b in BLOCKS} for H in HS} for cid in pool['pool']}
    A = {cid: {H: {b: L[cid][H][b] / max(L[none_id][H][b], 1e-12) for b in BLOCKS} for H in HS} for cid in pool['pool']}
    J = {cid: {b: fmean(A[cid][H][b] for H in HS) for b in BLOCKS} for cid in pool['pool']}
    ca = {cid: pool['pool'][cid]['c_a']['mean'] for cid in pool['pool']}
    r_ca, tie_ca = argmin_display(ca)
    r_late, tie_late = argmin_display({cid: J[cid]['late'] for cid in J})
    r_near, tie_near = argmin_display({cid: J[cid]['near'] for cid in J})
    picks = {'R_CA': r_ca, 'R_HistLate': r_late, 'R_HistNear': r_near, 'Fixed_dev': fixed_id}
    ranks = {'c_a': sorted(ca, key=lambda c: (ca[c], c)), 'hist_late': sorted(J, key=lambda c: (J[c]['late'], c)), 'hist_near': sorted(J, key=lambda c: (J[c]['near'], c))}
    return {'case': case, 'domain': pool['domain'], 'none_id': none_id, 'picks': picks, 'pick_labels': {k: pool['pool'][v]['label'] for k, v in picks.items()},
            'ties': {'R_CA': tie_ca, 'R_HistLate': tie_late, 'R_HistNear': tie_near}, 'ranks': ranks,
            'scores_at_selection': {cid: {'c_a_mean': ca[cid], 'c_a_by_seed': pool['pool'][cid]['c_a']['by_seed'], 'L': L[cid], 'A': A[cid], 'J_hist': J[cid],
                                          'by_h': {H: {b: {'by_seed': blocks[cid][H][b]['by_seed'], 'by_origin': blocks[cid][H][b]['by_origin']} for b in BLOCKS} for H in HS},
                                          'phys_h': {H: blocks[cid][H]['phys_h'] for H in HS}, 'alias_of_h': {H: blocks[cid][H]['alias_of'] for H in HS},
                                          'label_h': {H: blocks[cid][H]['label_h'] for H in HS}, 'assignment_changed_h': {H: blocks[cid][H]['assignment_changed'] for H in HS}} for cid in pool['pool']},
            'same_as': {'R_HistLate_eq_R_CA': r_late == r_ca, 'R_HistLate_eq_Fixed_dev': r_late == fixed_id, 'R_HistNear_eq_R_HistLate': r_near == r_late, 'R_HistNear_eq_Fixed_dev': r_near == fixed_id,
                        'R_CA_eq_Fixed_dev': r_ca == fixed_id}}


def paired_hist(blocks_a: dict, blocks_b: dict) -> dict:
    """Historical paired difference b - a per h / block (positive = candidate a lower loss than reference b): seeds, mean, SE, per-origin means, signs."""
    out = {}
    for H in HS:
        out[H] = {}
        for b in BLOCKS:
            d = [blocks_b[H][b]['by_seed'][i] - blocks_a[H][b]['by_seed'][i] for i in range(len(SEEDS))]
            d_o = [blocks_b[H][b]['by_origin'][o] - blocks_a[H][b]['by_origin'][o] for o in range(4)]
            out[H][b] = {'by_seed_delta': d, 'mean_delta': fmean(d), 'seed_se': se3(d), 'signs_by_seed': [int(x > 0) - int(x < 0) for x in d], 'by_origin_mean_delta': d_o,
                         'signs_by_origin': [int(x > 0) - int(x < 0) for x in d_o], 'relative_to_reference': fmean(d) / max(fmean(blocks_b[H][b]['by_seed']), 1e-12)}
    out['both_h'] = {b: {'mean_delta': fmean(out[H][b]['mean_delta'] for H in HS), 'signs_by_h': [int(out[H][b]['mean_delta'] > 0) - int(out[H][b]['mean_delta'] < 0) for H in HS],
                         'seed_signs_all': [s for H in HS for s in out[H][b]['signs_by_seed']], 'origin_signs_all': [s for H in HS for s in out[H][b]['signs_by_origin']]} for b in BLOCKS}
    return out


def feedback_of_case(root: Path, case: str, pool: dict, fixed_id: str) -> dict:
    hist = hist_scores_of(root, case)
    none_id = pool['reference_ids']['None']
    blocks = {cid: candidate_hist_blocks(hist, cid) for cid in pool['pool']}
    out = {'case': case, 'domain': pool['domain'], 'none_id': none_id, 'fixed_id': fixed_id, 'candidates': {}}
    for cid in pool['pool']:
        rec = {'label': pool['pool'][cid]['label'], 'vs_None': paired_hist(blocks[cid], blocks[none_id]) if cid != none_id else None,
               'vs_Fixed_dev': paired_hist(blocks[cid], blocks[fixed_id]) if cid != fixed_id else None,
               'current_c_a_vs_None': None if cid == none_id else {'by_seed_delta': [pool['pool'][none_id]['c_a']['by_seed'][i] - pool['pool'][cid]['c_a']['by_seed'][i] for i in range(3)]},
               'assignment_at_h': {H: {'phys_h': blocks[cid][H]['phys_h'], 'alias_of': blocks[cid][H]['alias_of'], 'label_h': blocks[cid][H]['label_h'], 'changed_vs_t': blocks[cid][H]['assignment_changed'],
                                       'entities_changed': hist[H]['pool']['candidates'][cid].get('entities_changed'), 'resolved_thresholds_h': hist[H]['pool']['candidates'][cid].get('resolved_thresholds_h'),
                                       'resolved_thresholds_t': hist[H]['pool']['candidates'][cid].get('resolved_thresholds_t')} for H in HS}}
        if rec['current_c_a_vs_None']:
            rec['current_c_a_vs_None']['mean_delta'] = fmean(rec['current_c_a_vs_None']['by_seed_delta'])
        out['candidates'][cid] = rec
    return out


def consistency_checks(root: Path, cases: list, n_updates=None) -> dict:
    """Real-data checks that cost nothing: every historical instance complete (all pool candidates scored on 3 seeds), the reused near origins
    0-1 equal the fit's own C_A per-origin values, the scoring slice ends at or before the current t, every cell bound to its instance."""
    out = {'items': {}, 'ok': True}
    for c in cases:
        for H in HS:
            jd = hist_dir(root, c, H) / ('%s_%s' % (c, H))
            hp, hs = context.read_json(jd / 'hist_pool.json'), context.read_json(jd / 'hist_scores.json')
            ch = load_case_json(case_json_path(root, c, H))
            cells = ec.branch_cells(jd)
            phys = set(hp['physical_materials'])
            complete = all(sum(1 for v in hs['cells'].values() if v['material_id'] == p) == len(SEEDS) for p in phys) and set(hs['cells']) == set(cells)
            reuse = all(hs['cells'][k]['near']['per_origin_normalized_mse_mean'][:2] == cells[k]['scores']['c_a']['per_origin_normalized_mse_mean'] for k in hs['cells'])
            bound = all(cells[k]['job_id'] == ch.case_id and list(cells[k]['rows_read']) == list(ch.evaluate_rows) and cells[k]['n_updates'] == (n_updates or spec.N_UPDATES) for k in cells)
            slice_ok = hs['rows_read'][1] <= ch.t + 384 and hs['rows_read'][0] == ch.t - spec.L and hs['late_origins'][-1] + spec.H <= ch.t + 384
            rec = {'complete': complete, 'near01_equal_c_a': reuse, 'cells_bound': bound, 'score_slice_ok': slice_ok, 'n_cells': len(cells), 'n_physical': len(phys)}
            rec['ok'] = all(rec[k] for k in ('complete', 'near01_equal_c_a', 'cells_bound', 'score_slice_ok'))
            out['items']['%s_%s' % (c, H)] = rec
            out['ok'] = out['ok'] and rec['ok']
    return out


def select_stage(root: Path = ROOT, *, n_updates=None) -> dict:
    """Freezes selections.json (write-once) and feedback.json from the historical scores and the current C_A only; refuses to run without every item."""
    P_ = paths(root)
    if P_['selections'].exists():
        return context.read_json(P_['selections'])
    cfg = context.read_json(P_['frozen'])
    cases = list(cfg['cases'])
    chk = consistency_checks(root, cases, n_updates)
    if not chk['ok']:
        raise RuntimeError('historical instances incomplete or inconsistent: %s' % {k: v for k, v in chk['items'].items() if not v['ok']})
    sel, fb = {'package': PACKAGE, 'frozen_local': now(), 'labels_read': 'none (historical near / late blocks before t, current C_A only)', 'consistency_checks': chk, 'cases': {}}, {'package': PACKAGE, 'cases': {}}
    for c in cases:
        pool = load_pool(root, c)
        fixed_id = pool['reference_ids'][cfg['fixed_dev'][pool['domain']]['public_id']]
        sel['cases'][c] = selections_of_case(root, c, pool, fixed_id)
        fb['cases'][c] = feedback_of_case(root, c, pool, fixed_id)
    sel['summary'] = {s: {c: sel['cases'][c]['picks'][s] for c in cases} for s in STRATEGIES}
    write_once(P_['selections'], sel)
    context.write_json(P_['feedback'], fb)
    return sel


# ============================================================================= evaluator (after the freeze): the current C_B / E of the ORIGINAL F0 models
def f0_labels(pool: dict) -> dict:
    bdir = Path(pool['branch_dir_f0'])
    es = context.read_json(bdir / 'e_scores.json')['cells']
    cb = context.read_json(bdir / 'c_b_scores.json')['cells']
    out = {}
    for cid, r in pool['pool'].items():
        out[cid] = {'e': block_from_cells(es, r['plan_ids'], 'e'), 'c_b': block_from_cells(cb, r['plan_ids'], 'c_b')}
    return out


def G(a_e: dict, b_e: dict, den: float) -> dict:
    """100 x (E(B) - E(A)) / E(None) paired per seed; positive = A better."""
    per = [100.0 * (b_e['by_seed'][i] - a_e['by_seed'][i]) / den for i in range(len(SEEDS))]
    return {'value': 100.0 * (b_e['mean'] - a_e['mean']) / den, **stats3(per)}


def concordance(f: dict, e: dict) -> dict:
    """Pairwise same-direction rate between a feedback score and the current E over candidate pairs (lower = better in both); ties listed separately."""
    ids = sorted(f)
    agree = disagree = ties = 0
    for a, b in itertools.combinations(ids, 2):
        df, de = f[a] - f[b], e[a] - e[b]
        if abs(df) <= TOL or abs(de) <= TOL:
            ties += 1
            continue
        if (df < 0) == (de < 0):
            agree += 1
        else:
            disagree += 1
    n = agree + disagree
    return {'agree': agree, 'disagree': disagree, 'ties': ties, 'rate': agree / n if n else None, 'n_pairs': n}


def case_readout(root: Path, case: str, sel: dict, pool: dict) -> dict:
    lab = f0_labels(pool)
    none_id = pool['reference_ids']['None']
    den = lab[none_id]['e']['mean']
    den_cb = lab[none_id]['c_b']['mean']
    picks = sel['picks']
    e_mean = {cid: lab[cid]['e']['mean'] for cid in pool['pool']}
    best, _ = argmin_display(e_mean)
    out = {'case': case, 'domain': pool['domain'], 'none_e_mean': den, 'picks': picks, 'pick_labels': sel['pick_labels'], 'post_hoc_best': best, 'post_hoc_best_label': pool['pool'][best]['label'],
           'strategies': {}, 'pairs': {}, 'alignment': {}, 'candidates': {}}
    for s, cid in picks.items():
        out['strategies'][s] = {'pick': cid, 'label': pool['pool'][cid]['label'], 'e_mean': e_mean[cid], 'e_by_seed': lab[cid]['e']['by_seed'], 'c_b_mean': lab[cid]['c_b']['mean'],
                                'G_over_None': G(lab[cid]['e'], lab[none_id]['e'], den), 'G_cb_over_None': G(lab[cid]['c_b'], lab[none_id]['c_b'], den_cb),
                                'regret_pp': 100.0 * (e_mean[cid] - e_mean[best]) / den, 'is_post_hoc_best': cid == best}
    for a, b in (('R_HistLate', 'R_CA'), ('R_HistLate', 'Fixed_dev'), ('R_HistNear', 'R_CA'), ('R_HistNear', 'Fixed_dev'), ('R_HistLate', 'R_HistNear'), ('R_CA', 'Fixed_dev')):
        ca_, cb_ = picks[a], picks[b]
        same = ca_ == cb_
        out['pairs']['%s_over_%s' % (a, b)] = {'same_material': same, 'G': None if same else G(lab[ca_]['e'], lab[cb_]['e'], den), 'G_cb': None if same else G(lab[ca_]['c_b'], lab[cb_]['c_b'], den_cb),
                                               'outcome': 'tie' if same else ('win' if e_mean[ca_] < e_mean[cb_] else 'loss')}
    ssel = sel['scores_at_selection']
    feeds = {'c_a': {cid: ssel[cid]['c_a_mean'] for cid in pool['pool']}, 'hist_near': {cid: ssel[cid]['J_hist']['near'] for cid in pool['pool']}, 'hist_late': {cid: ssel[cid]['J_hist']['late'] for cid in pool['pool']}}
    for H in HS:
        feeds['hist_near_%s' % H] = {cid: ssel[cid]['A'][H]['near'] for cid in pool['pool']}
        feeds['hist_late_%s' % H] = {cid: ssel[cid]['A'][H]['late'] for cid in pool['pool']}
    for k, f in feeds.items():
        out['alignment'][k] = concordance(f, e_mean)
    out['alignment']['c_b_vs_e'] = concordance({cid: lab[cid]['c_b']['mean'] for cid in pool['pool']}, e_mean)
    fixed_id = picks['Fixed_dev']
    for cid in pool['pool']:
        out['candidates'][cid] = {'label': pool['pool'][cid]['label'], 'e_mean': e_mean[cid], 'e_by_seed': lab[cid]['e']['by_seed'], 'c_b_mean': lab[cid]['c_b']['mean'],
                                  'G_over_None': G(lab[cid]['e'], lab[none_id]['e'], den)['value'] if cid != none_id else 0.0,
                                  'G_over_Fixed_dev': G(lab[cid]['e'], lab[fixed_id]['e'], den)['value'] if cid != fixed_id else 0.0,
                                  'c_a_mean': ssel[cid]['c_a_mean'], 'J_hist': ssel[cid]['J_hist'], 'A': ssel[cid]['A'], 'rank_e': None}
    order = sorted(pool['pool'], key=lambda c: (e_mean[c], c))
    for i, cid in enumerate(order):
        out['candidates'][cid]['rank_e'] = i + 1
    out['opportunity'] = {'post_hoc_best': best, 'available_over_Fixed_dev_pp': 100.0 * (e_mean[fixed_id] - e_mean[best]) / den, 'available_over_R_CA_pp': 100.0 * (e_mean[picks['R_CA']] - e_mean[best]) / den,
                          'n_candidates_better_than_Fixed_dev': sum(1 for c in pool['pool'] if e_mean[c] < e_mean[fixed_id] - TOL)}
    return out


def _mean_or_none(vals: list):
    return fmean(vals) if vals else None


def readout(root: Path = ROOT) -> dict:
    P_ = paths(root)
    if not P_['selections'].exists():
        raise PermissionError('evaluator refused: selections.json is not frozen')
    sel = context.read_json(P_['selections'])
    cfg = context.read_json(P_['frozen'])
    cases = list(cfg['cases'])
    per_case = {c: case_readout(root, c, sel['cases'][c], load_pool(root, c)) for c in cases}
    res = {'package': PACKAGE, 'exposure': EXPOSURE, 'evaluated_local': now(), 'cases': per_case, 'domains': {}, 'overall': {}, 'labels_connected_after_freeze': sel['frozen_local']}
    pairs = list(next(iter(per_case.values()))['pairs'])
    doms = sorted({r['domain'] for r in per_case.values()})
    dom_recs = {d: [per_case[c] for c in cases if per_case[c]['domain'] == d] for d in doms}
    for d, recs in dom_recs.items():
        res['domains'][d] = {'cases': [r['case'] for r in recs], 'pairs': {}, 'strategies': {}}
        for p in pairs:
            gs = [(r['pairs'][p]['G']['value'] if r['pairs'][p]['G'] else 0.0) for r in recs]
            res['domains'][d]['pairs'][p] = {'G_mean': fmean(gs), 'per_case': gs, 'outcomes': [r['pairs'][p]['outcome'] for r in recs]}
        for s in STRATEGIES:
            res['domains'][d]['strategies'][s] = {'G_over_None_mean': fmean(r['strategies'][s]['G_over_None']['value'] for r in recs), 'regret_mean': fmean(r['strategies'][s]['regret_pp'] for r in recs)}
    for p in pairs:
        per_dom = [res['domains'][d]['pairs'][p]['G_mean'] for d in doms]
        gs = [(per_case[c]['pairs'][p]['G']['value'] if per_case[c]['pairs'][p]['G'] else 0.0) for c in cases]
        oc = [per_case[c]['pairs'][p]['outcome'] for c in cases]
        res['overall'][p] = {'G_mean_domains_equal': fmean(per_dom), 'per_domain': dict(zip(doms, per_dom)), 'per_case': dict(zip(cases, gs)), 'wins': oc.count('win'), 'ties': oc.count('tie'), 'losses': oc.count('loss'),
                             'max_harm_pp': min(gs), 'max_gain_pp': max(gs)}
    res['overall']['strategies'] = {s: {'G_over_None_mean': fmean(res['domains'][d]['strategies'][s]['G_over_None_mean'] for d in doms), 'regret_mean': fmean(res['domains'][d]['strategies'][s]['regret_mean'] for d in doms),
                                        'picks': {c: per_case[c]['picks'][s] for c in cases}, 'pick_labels': {c: per_case[c]['pick_labels'][s] for c in cases}} for s in STRATEGIES}
    res['overall']['alignment'] = {k: {'rate_cases_equal': _mean_or_none([per_case[c]['alignment'][k]['rate'] for c in cases if per_case[c]['alignment'][k]['rate'] is not None]),
                                       'per_case': {c: per_case[c]['alignment'][k] for c in cases}} for k in next(iter(per_case.values()))['alignment']}
    res['overall']['same_as'] = {k: sum(1 for c in cases if sel['cases'][c]['same_as'][k]) for k in next(iter(sel['cases'].values()))['same_as']}
    res['verdict'] = verdict(res, cases)
    res['cost'] = cost_readout(root)
    context.write_json(P_['result'], res)
    P_['tables'].write_text(tables(res, sel), encoding='utf-8')
    return res


def verdict(res: dict, cases: list) -> dict:
    """Task §9 categories from the actual readings (no threshold tuned after the fact)."""
    o = res['overall']
    late_ca, late_fd, near_ca, near_fd = o['R_HistLate_over_R_CA'], o['R_HistLate_over_Fixed_dev'], o['R_HistNear_over_R_CA'], o['R_HistNear_over_Fixed_dev']
    n = len(cases)
    late_is_fixed = o['same_as']['R_HistLate_eq_Fixed_dev']
    late_eq_near = o['same_as']['R_HistNear_eq_R_HistLate']
    reduces_misselection = late_ca['G_mean_domains_equal'] > 0 and late_ca['losses'] == 0
    realized_vs_fixed = late_fd['G_mean_domains_equal'] > 0 and late_fd['losses'] == 0
    if late_is_fixed == n or (late_fd['ties'] + late_fd['losses'] == n and late_ca['G_mean_domains_equal'] > 0):
        cat = 'DEFAULT_RECOVERY_ONLY'
    elif reduces_misselection and realized_vs_fixed:
        cat = 'NEAR_EQUALLY_EFFECTIVE' if (late_eq_near == n or (near_ca['G_mean_domains_equal'] >= late_ca['G_mean_domains_equal'] - TOL and near_fd['G_mean_domains_equal'] > 0)) else 'NON_DEFAULT_IMPROVEMENT_SIGNAL'
    else:
        cat = 'MIXED_OR_WORSE'
    return {'category': cat, 'R_HistLate_over_R_CA': late_ca['G_mean_domains_equal'], 'R_HistLate_over_Fixed_dev': late_fd['G_mean_domains_equal'], 'R_HistNear_over_R_CA': near_ca['G_mean_domains_equal'],
            'R_HistNear_over_Fixed_dev': near_fd['G_mean_domains_equal'], 'late_equals_fixed_cases': late_is_fixed, 'late_equals_near_cases': late_eq_near, 'n_cases': n,
            'rule': 'DEFAULT_RECOVERY_ONLY if R_HistLate == Fixed_dev in every case or it never beats Fixed_dev while beating R_CA; NON_DEFAULT_IMPROVEMENT_SIGNAL if R_HistLate beats R_CA and Fixed_dev on the '
                    'domain-equal mean with no losing case; NEAR_EQUALLY_EFFECTIVE if the near-block control delivers the same or as much; otherwise MIXED_OR_WORSE'}


def cost_readout(root: Path) -> dict:
    P_ = paths(root)
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    st = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    items = {}
    cfg = context.read_json(P_['frozen'])
    build_s = score_s = 0.0
    n_models = n_inf = n_reused = n_phys = n_alias = 0
    for c in cfg['cases']:
        for H in HS:
            jd = hist_dir(root, c, H) / ('%s_%s' % (c, H))
            hp, hs = jd / 'hist_pool.json', jd / 'hist_scores.json'
            if hp.exists() and hs.exists():
                a, b = context.read_json(hp), context.read_json(hs)
                build_s += a['build_seconds_total']
                score_s += b['seconds']
                n_models += b['n_models']
                n_inf += b['n_origins_inferred']
                n_reused += b['n_origins_reused']
                n_phys += len(a['physical_materials'])
                n_alias += sum(1 for r in a['candidates'].values() if r.get('alias_of'))
                items['%s_%s' % (c, H)] = {'physical': len(a['physical_materials']), 'aliases': sum(1 for r in a['candidates'].values() if r.get('alias_of')), 'build_s': a['build_seconds_total'], 'score_s': b['seconds'], 'models': b['n_models']}
    wiring = context.read_json(P_['wiring'] / 'wiring_result.json') if (P_['wiring'] / 'wiring_result.json').exists() else {}
    hist_wall = (led.get('historical_finished_epoch') or time.time()) - led['historical_started_epoch'] if led.get('historical_started_epoch') else None
    return {'fit_attempts_total': led.get('fit_attempts'), 'fits_ok': led.get('fits_ok'), 'fits_failed': led.get('fits_failed'), 'retries_used': led.get('retries_used'), 'cache_hits': led.get('cache_hits'),
            'wiring_fits': (st.get('historical') or {}).get('snapshot_at_start', {}).get('fit_attempts'), 'fit_wall_seconds_sum': led.get('fit_wall_seconds'),
            'material_wall_seconds_sum': led.get('material_wall_seconds', 0.0), 'histscore_wall_seconds_sum': led.get('histscore_wall_seconds', 0.0), 'build_seconds_in_worker': build_s,
            'score_seconds_in_worker': score_s, 'historical_models': n_models, 'historical_physical_materials': n_phys, 'historical_alias_candidates': n_alias, 'origins_inferred': n_inf, 'origins_reused_from_c_a': n_reused,
            'historical_stage_wall_s': hist_wall, 'per_item': items, 'pools': led.get('pools_snapshot'), 'concurrency_effective': effective_numeric(root), 'wiring_memory': wiring.get('memory'),
            'llm_requests': 0, 'http_attempts': 0, 'sha': 0, 'commits': 0,
            'note': 'the four selectors read the same historical scores; the near / late contrast shares the models (a deployment of both would still train once per h); the current-t pool fits are sunk F0 cost'}


def _f(x, nd=2):
    return '-' if x is None else ('%.*f' % (nd, x))


def tables(res: dict, sel: dict) -> str:
    L_ = ['# %s - tables (pp of None; positive = first arm better; E main block)' % PACKAGE, '', '## Verdict: %s' % res['verdict']['category'], '',
          '| pair | overall (domains equal) | per domain | wins/ties/losses | max harm | max gain |', '|---|---:|---|---|---:|---:|']
    for p, v in res['overall'].items():
        if p in ('strategies', 'alignment', 'same_as'):
            continue
        L_.append('| %s | %s | %s | %d/%d/%d | %s | %s |' % (p, _f(v['G_mean_domains_equal']), ', '.join('%s %s' % (d, _f(g)) for d, g in v['per_domain'].items()), v['wins'], v['ties'], v['losses'], _f(v['max_harm_pp']), _f(v['max_gain_pp'])))
    L_ += ['', '## Delivery table', '', '| case | strategy | pick | label | E mean | G over None | G over R_CA | G over Fixed_dev | regret (pp) | post-hoc best |', '|---|---|---|---|---:|---:|---:|---:|---:|---|']
    for c, r in res['cases'].items():
        for s in STRATEGIES:
            st = r['strategies'][s]
            g_ca = r['pairs'].get('%s_over_R_CA' % s, {}).get('G') if s != 'R_CA' else None
            g_fd = r['pairs'].get('%s_over_Fixed_dev' % s, {}).get('G') if s != 'Fixed_dev' else None
            L_.append('| %s | %s | %s | %s | %.5f | %s | %s | %s | %s | %s |' % (c, s, st['pick'], st['label'], st['e_mean'], _f(st['G_over_None']['value']),
                                                                         '0.00 (same)' if s != 'R_CA' and r['picks'][s] == r['picks']['R_CA'] else (_f(g_ca['value']) if g_ca else '-'),
                                                                         '0.00 (same)' if s != 'Fixed_dev' and r['picks'][s] == r['picks']['Fixed_dev'] else (_f(g_fd['value']) if g_fd else '-'),
                                                                         _f(st['regret_pp']), r['post_hoc_best'] + ' ' + r['post_hoc_best_label']))
    L_ += ['', '## Opportunity table', '', '| case | post-hoc best | available over Fixed_dev | available over R_CA | # better than Fixed_dev | HistLate changed vs R_CA | HistLate = Fixed_dev |', '|---|---|---:|---:|---:|---|---|']
    for c, r in res['cases'].items():
        o = r['opportunity']
        sa = sel['cases'][c]['same_as']
        L_.append('| %s | %s %s | %s | %s | %d | %s | %s |' % (c, o['post_hoc_best'], r['post_hoc_best_label'], _f(o['available_over_Fixed_dev_pp']), _f(o['available_over_R_CA_pp']), o['n_candidates_better_than_Fixed_dev'],
                                                       'no' if sa['R_HistLate_eq_R_CA'] else 'yes', 'yes' if sa['R_HistLate_eq_Fixed_dev'] else 'no'))
    L_ += ['', '## Feedback alignment (pairwise same-direction rate with current E; ties excluded and counted)', '', '| feedback | cases-equal rate | ' + ' | '.join(res['cases']) + ' |', '|---|---:|' + '---:|' * len(res['cases'])]
    for k, v in res['overall']['alignment'].items():
        L_.append('| %s | %s | %s |' % (k, _f(v['rate_cases_equal'], 3), ' | '.join('%s (%d/%d, ties %d)' % (_f(v['per_case'][c]['rate'], 3), v['per_case'][c]['agree'], v['per_case'][c]['n_pairs'], v['per_case'][c]['ties']) for c in res['cases'])))
    L_ += ['', '## Candidate table (per case: current C_A, J_hist near / late, current E, G over None / Fixed_dev)', '']
    for c, r in res['cases'].items():
        L_ += ['### %s (None E = %.5f; picks: %s)' % (c, r['none_e_mean'], ', '.join('%s=%s' % (s, r['picks'][s]) for s in STRATEGIES)), '',
               '| id | label | C_A | A near H1 | A near H2 | J near | A late H1 | A late H2 | J late | E | rank E | G over None | G over Fixed_dev |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for cid, v in r['candidates'].items():
            L_.append('| %s | %s | %.5f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.5f | %d | %s | %s |' % (cid, v['label'], v['c_a_mean'], v['A']['H1']['near'], v['A']['H2']['near'], v['J_hist']['near'],
                                                                                                              v['A']['H1']['late'], v['A']['H2']['late'], v['J_hist']['late'], v['e_mean'], v['rank_e'], _f(v['G_over_None']), _f(v['G_over_Fixed_dev'])))
        L_.append('')
    cst = res['cost']
    L_ += ['## Cost', '', '| item | value |', '|---|---:|']
    for k in ('fit_attempts_total', 'fits_ok', 'fits_failed', 'retries_used', 'cache_hits', 'wiring_fits', 'historical_models', 'historical_physical_materials', 'historical_alias_candidates', 'origins_inferred',
              'origins_reused_from_c_a', 'fit_wall_seconds_sum', 'material_wall_seconds_sum', 'histscore_wall_seconds_sum', 'historical_stage_wall_s', 'concurrency_effective', 'llm_requests', 'http_attempts'):
        v = cst.get(k)
        L_.append('| %s | %s |' % (k, _f(v, 1) if isinstance(v, float) else v))
    return '\n'.join(L_) + '\n'


# ============================================================================= wiring (real D01_Q03 / D02_Q03 at the current t; <= 6 fits; 0 labels)
def wiring(root: Path = ROOT) -> dict:
    P_ = paths(root)
    out = P_['wiring'] / 'wiring_result.json'
    if out.exists():
        return context.read_json(out)
    import psutil
    cfg = preflight(root)
    sc = stage_caps(root, 'wiring')
    led = DP.SafeLedger(P_['ledger'], **sc['caps'])
    rt.FIT_RETRY = True
    DP.set_pools(CONCURRENCY['numeric_max'], CONCURRENCY['http'])
    P_['wiring'].mkdir(parents=True, exist_ok=True)
    P_['logs'].mkdir(exist_ok=True)
    cases = current_cases(root)
    res = {'status': 'PASS', 'started_local': now(), 'checks': [], 'cases': {}, 'available_mb_before': psutil.virtual_memory().available / 1e6}
    mem, stop = {}, threading.Event()
    sampler = threading.Thread(target=DP._sample_memory, args=(stop, mem), daemon=True)
    sampler.start()
    t_all = time.time()
    results, errs = {}, []

    def one(case):
        cs = cases[case]
        wd = P_['wiring'] / case
        wd.mkdir(exist_ok=True)
        cj = wd / 'case.json'
        if not cj.exists():
            context.write_json(cj, {'spec': cs.to_json(), 'identity': 'wiring_current_t', 'package': PACKAGE})
        pool = load_pool(root, case)
        bdir, common = f0_dirs(case)
        checks = []
        # 1. build the public materials at the current t (0 fits) and compare P_AmpResample bytes with the F0 common cache
        with DP.pools().numeric():
            rc, secs = run_worker(['--worker-build', root, cj, P_['pool'] / (case + '.json')], P_['logs'] / ('wiring_build_%s.log' % case), 1800)
        if rc != 0:
            raise RuntimeError('wiring build failed for %s' % case)
        hp = context.read_json(wd / case / 'hist_pool.json')
        reg = ec.load_registry(wd / case)
        with np.load(reg['P_AmpResample']['path']) as z1, np.load(common / 'aug_materials' / 'P_AmpResample.npz') as z2:
            amp_eq = np.array_equal(z1['Xc'], z2['Xc']) and np.array_equal(z1['yc'], z2['yc'])
        checks.append({'case': case, 'check': 'P_AmpResample rebuilt at the current t bitwise equal to the F0 common material (0 fits)', 'ok': bool(amp_eq)})
        with np.load(reg['P_NoMixRecipe']['path']) as z1, np.load(common / 'aug_materials' / 'P_NoMixRecipe.npz') as z2:
            nomix_eq = np.array_equal(z1['Xc'], z2['Xc']) and np.array_equal(z1['yc'], z2['yc'])
        checks.append({'case': case, 'check': 'P_NoMixRecipe rebuilt at the current t bitwise equal to the F0 common material', 'ok': bool(nomix_eq)})
        # 2. every constructed candidate compiled on the current-t table reproduces the F0 assignment (task §4.1) and its rebuilt material equals the F0 bytes
        asg_ok, bytes_ok = [], []
        for cid, r in hp['candidates'].items():
            pr = pool['pool'][cid]
            if pr['public_id']:
                continue
            asg_ok.append(r['assignment_h'] == pr['assignment_t'] and not r['assignment_changed'])
            with np.load(reg[r['phys_h']]['path']) as z1, np.load(common / 'aug_materials' / (pr['phys_f0'] + '.npz')) as z2:
                bytes_ok.append(bool(np.array_equal(z1['Xc'], z2['Xc']) and np.array_equal(z1['yc'], z2['yc'])))
        checks.append({'case': case, 'check': 'constructed candidates recompiled at t: assignment == F0 assignment (incl. conditional policies)', 'ok': all(asg_ok), 'n': len(asg_ok)})
        checks.append({'case': case, 'check': 'constructed candidates rebuilt at t: material bytes == F0 common material', 'ok': all(bytes_ok), 'n': len(bytes_ok)})
        with np.load(wd / case / 'scaler.npz') as z1, np.load(bdir / 'scaler.npz') as z2:
            sc_eq = np.array_equal(z1['mean'], z2['mean']) and np.array_equal(z1['scale'], z2['scale'])
        checks.append({'case': case, 'check': 'current-t scaler identical to the F0 scaler', 'ok': bool(sc_eq)})
        # 3. None x 3 seeds at the current t under the concurrent scheduler: C_A identical to the F0 cells
        res_fit = fit_cells(led, root, cj, cs, [('None', s) for s in SEEDS])
        mine, theirs, eq = {}, {}, []
        f0_cells = ec.branch_cells(bdir)
        for s in SEEDS:
            a = res_fit[ec.cell_id(case, 'None', s)]['record']
            b = f0_cells.get(ec.cell_id(case, 'None', s))
            mine[s] = a['scores']['c_a']['normalized_mse_macro']
            theirs[s] = b['scores']['c_a']['normalized_mse_macro'] if b else None
            eq.append(b is not None and mine[s] == theirs[s] and a['pool_size'] == b['pool_size'] == ec.COHORT_SIZE * spec.N_PARENTS and a['torch_threads'] == b['torch_threads'] == FIT_THREADS)
        checks.append({'case': case, 'check': 'None x 3 seeds at the current t: C_A identical to the F0 cells (same pool, same threads)', 'ok': all(eq), 'mine': mine, 'f0': theirs})
        res['cases'][case] = {'none_c_a_by_seed': mine, 'f0_c_a_by_seed': theirs, 'fit_seconds': [round(res_fit[ec.cell_id(case, 'None', s)]['record']['train']['seconds'], 1) for s in SEEDS],
                              'worker_seconds': [round(res_fit[ec.cell_id(case, 'None', s)]['record']['worker_seconds_total'], 1) for s in SEEDS], 'build_worker_seconds': secs}
        return checks

    def run(case):
        try:
            results[case] = one(case)
        except Exception as exc:  # noqa: BLE001
            errs.append((case, repr(exc)))
    ts = [threading.Thread(target=run, args=(c,), daemon=True) for c in WIRING_CASES]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    stop.set()
    sampler.join(timeout=5)
    for c in WIRING_CASES:
        res['checks'] += results.get(c, [{'case': c, 'check': 'case ran', 'ok': False, 'error': str(errs)[:400]}])
    res['wall_seconds_both_cases'] = time.time() - t_all
    res['fits_total'] = led.s['fit_attempts']
    res['fit_wall_seconds_sum'] = led.s['fit_wall_seconds']
    res['pools'] = DP.pools().snapshot()
    res['memory'] = mem
    res['checks'].append({'check': 'exactly 6 physical fits, all OK', 'ok': led.s['fit_attempts'] == 6 and led.s['fits_ok'] == 6})
    no_labels = not any(p_.name in ('c_b_scores.json', 'e_frozen.json', 'e_scores.json') for p_ in P_['wiring'].rglob('*.json'))
    res['checks'].append({'check': 'no C_B / E label file in wiring', 'ok': no_labels})
    peak = mem.get('peak_worker_rss_mb') or 0.0
    avail = res['available_mb_before']
    numeric = CONCURRENCY['numeric_max'] if (3 * peak + 1500 <= avail and peak > 0) else CONCURRENCY['numeric']
    decision = {'concurrency': {**CONCURRENCY, 'numeric': numeric}, 'peak_worker_rss_mb': peak, 'peak_workers_total_rss_mb': mem.get('peak_workers_total_rss_mb'), 'available_mb_before': avail,
                'available_mb_min_during': mem.get('available_mb_min'), 'rule': '3 x peak worker RSS + 1500 MB margin <= available before start -> 3, else 2 (task §7)', 'decided_local': now(),
                'note': 'threads / seeds / steps / candidates unchanged; the pool only limits how many worker processes run at once'}
    context.write_json(P_['concurrency'], decision)
    res['concurrency_decision'] = decision
    # projection for the task's wall-clock warning (single measured fit vs. 186 fits at the decided concurrency)
    fs = [v for c in res['cases'].values() for v in c['worker_seconds']]
    res['projection'] = {'worker_seconds_per_fit_mean': fmean(fs) if fs else None, 'main_fits': ALLOC['historical']['fits'], 'numeric': numeric,
                         'projected_fit_wall_s': (fmean(fs) * ALLOC['historical']['fits'] / numeric) if fs else None, 'note': 'fits only; material builds (8 instances) and scoring add on top; contention may raise the per-fit time'}
    res['status'] = 'PASS' if all(c['ok'] for c in res['checks']) and led.s['fit_attempts'] <= ALLOC['wiring']['fits'] else 'FAIL'
    res['finished_local'] = now()
    write_once(out, res)
    print('WIRING', res['status'], 'fits', res['fits_total'], 'numeric', numeric, 'projected_fit_wall_s', res['projection']['projected_fit_wall_s'], flush=True)
    return res


# ============================================================================= package driver
def package_run(root: Path = ROOT) -> dict:
    P_ = paths(root)
    with DP.PackageLock(root, 'driver'):
        context.write_json(root / 'driver_launch.json', {'pid': os.getpid(), 'local': now(), 'package': PACKAGE})
        cfg = preflight(root)
        w = wiring(root)
        if w['status'] != 'PASS':
            raise RuntimeError('wiring FAIL: %s' % [c for c in w['checks'] if not c['ok']])
        hist = historical_stage(root)
        sel = select_stage(root)
        res = readout(root)
        context.write_json(root / 'package_status.json', {'status': 'COMPLETE', 'finished_local': now(), 'verdict': res['verdict'], 'fits': hist['ledger']['fit_attempts']})
        print('PACKAGE COMPLETE', res['verdict']['category'], flush=True)
        return res


def monitor(root: Path = ROOT, interval: float = 60.0, once: bool = False) -> None:
    import psutil
    P_ = paths(root)
    while True:
        line = ['MONITOR', now()]
        q = root / 'driver_launch.json'
        if q.exists():
            try:
                pid = int(context.read_json(q)['pid'])
                line.append('driver=%d:%s' % (pid, 'alive' if psutil.pid_exists(pid) else 'exited'))
            except Exception:  # noqa: BLE001
                pass
        held = None
        if P_['lock'].exists():
            try:
                with open(P_['lock'], 'rb') as fh:
                    fh.read(1)
                held = False
            except OSError:
                held = True
        line.append('lock=%s' % ({None: 'absent', True: 'held', False: 'free'}[held]))
        if P_['ledger'].exists():
            led = context.read_json(P_['ledger'])
            ev = led.get('events', [])
            last_fit = max((e['epoch'] for e in ev if e.get('kind') == 'fit_finished'), default=None)
            line.append('fits=%d/%d ok=%d failed=%d cache=%d retries=%d' % (led['fit_attempts'], led['caps']['max_fit_attempts'], led['fits_ok'], led['fits_failed'], led['cache_hits'], led['retries_used']))
            line.append('last_fit=%s' % ('%.0fs ago' % (time.time() - last_fit) if last_fit else '-'))
            if led.get('historical_started_epoch'):
                line.append('hist=%.1fmin' % ((time.time() - led['historical_started_epoch']) / 60))
        if P_['progress'].exists():
            pr = context.read_json(P_['progress'])
            st = {}
            for v in pr.get('items', {}).values():
                st[v.get('status', '?')] = st.get(v.get('status', '?'), 0) + 1
            pl = pr.get('pools') or {}
            line.append('items %s pool(num %s/%s peak %s)' % (st, pl.get('numeric_active'), pl.get('numeric_limit'), pl.get('numeric_peak')))
        for p_ in sorted(P_['logs'].glob('*.log'), key=lambda p: p.stat().st_mtime)[-2:] if P_['logs'].exists() else []:
            tail = p_.read_text(encoding='utf-8', errors='replace').splitlines()[-1:]
            if tail:
                line.append('%s: %s' % (p_.name, tail[0][:100]))
        try:
            line.append('avail=%.1fGB' % (psutil.virtual_memory().available / 1e9))
        except Exception:  # noqa: BLE001
            pass
        print(' '.join(line), flush=True)
        if once:
            return
        time.sleep(interval)


# ============================================================================= smoke (synthetic file; tiny training; every guard of task §6.B)
def _synthetic_case(out: Path, n_cols: int = 40, hours: int = 3000, t: int = 2400) -> ec.CaseSpec:
    import datetime as _dt
    rng = np.random.RandomState(11)
    t0 = _dt.datetime(2016, 7, 1, 2)
    csvp = out / 'synthetic.csv'
    base_ = rng.rand(n_cols) * 5 + 1
    with csvp.open('w', newline='') as f:
        f.write('date,' + ','.join(str(i) for i in range(n_cols)) + ',OT\n')
        for h in range(hours):
            ts = t0 + _dt.timedelta(hours=h)
            drift = 1 + 0.4 * np.sin(2 * np.pi * h / 900.0 + np.arange(n_cols) * 0.3)     # slow regime change so that T-only features move between cuts
            vals = base_ * drift * (1 + 0.3 * np.sin(2 * np.pi * (h % 24) / 24 + np.arange(n_cols))) + 0.1 * rng.randn(n_cols)
            f.write(ts.strftime('%Y-%m-%d %H:%M:%S') + ',' + ','.join('%.4f' % v for v in vals) + ',%.3f\n' % rng.rand())
    cols = [str(i) for i in range(n_cols)]
    return ec.CaseSpec(dataset='synthetic', domain='SM', case_id='SM_Q03', role='test', t=t, roster=tuple(cols[0:16]), case_index=14, csv_path=str(csvp), total_hours=hours)


def _synthetic_pool(cs: ec.CaseSpec, table_t: list) -> dict:
    """A pool in the exact layout of bind_pool: four public references, one edit, one conditional (quantile rule), with fake current C_A."""
    rng = np.random.RandomState(3)
    defs = [('FixedMixup', None, 'FixedMixup'), ('None', ec.uniform_policy([], ''), 'None'), ('P_NoMixRecipe', ec.uniform_policy(PUBLIC_STEPS['P_NoMixRecipe'], ''), 'P_NoMixRecipe'),
            ('P_AmpResample', ec.uniform_policy(PUBLIC_STEPS['P_AmpResample'], ''), 'P_AmpResample'),
            (None, {'default': {'steps': [ta.edit_step(['tp_shock'])]}, 'rules': [], 'observation_fields_used': []}, 'edit'),
            (None, {'default': {'steps': [{'op': 'tp_regime'}]}, 'rules': [{'when': {'feature': 'lag24_corr', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': []}], 'observation_fields_used': ['lag24_corr']}, 'cond')]
    rows = []
    for pub, pol, name in defs:
        if pub == 'FixedMixup':
            asg, key = 'FixedMixup', FIXED_MIXUP_KEY
            comp = None
        else:
            comp = ec.compile_plan_full(pol, table_t)
            asg = comp['assignment']
            key = ec.material_key(asg)
        mats = [[[float(rng.rand() * 0.2 + 0.5 + (0.05 if name == 'edit' else 0.0)) for _ in range(ec.COHORT_SIZE)] for _ in range(2)] for _ in SEEDS]
        rows.append({'phys_f0': pub or name, 'plan_ids': [pub or name], 'public_id': pub, 'key': key, 'label': ec.assignment_label(asg, pub), 'assignment_t': asg, 'policy': None if pub == 'FixedMixup' else clean_policy(pol),
                     'program_source_plan': pub or name, 'c_a': block_summary(mats), 'kind': 'smoke', 'resolved_thresholds_t': comp['resolved_thresholds'] if comp else {}, 'rule_index_t': comp['rule_index'] if comp else None})
    order = sorted(rows, key=lambda r: r['key'])
    pool = {'C%02d' % i: {**r, 'display_id': 'C%02d' % i} for i, r in enumerate(order)}
    return {'case': cs.case_id, 'domain': 'SM', 'branch_dir_f0': None, 'common_dir_f0': None, 'pool': pool, 'pool_size': len(pool), 'reference_ids': {r['public_id']: cid for cid, r in pool.items() if r['public_id']},
            'order_rule': 'lexicographic semantic key', 'labels_read': 'none', 'overview_t_entities': table_t}


def _fake_f0_labels(out: Path, pool: dict, case: str) -> Path:
    """Fake e_scores.json / c_b_scores.json in the F0 layout so that the evaluator code path runs in smoke."""
    rng = np.random.RandomState(5)
    bdir = out / 'f0' / case
    bdir.mkdir(parents=True, exist_ok=True)
    for block in ('e', 'c_b'):
        cells = {}
        for cid, r in pool['pool'].items():
            for s in SEEDS:
                mat = [[float(rng.rand() * 0.2 + 0.5) for _ in range(ec.COHORT_SIZE)] for _ in range(4 if block == 'e' else 2)]
                sc = {'status': 'SCORABLE', 'per_origin_entity_normalized_mse': mat, 'normalized_mse_macro': float(np.mean(mat)), 'n_nonfinite_predictions': 0}
                cells[ec.cell_id(case, r['plan_ids'][0], s)] = {'material_id': r['plan_ids'][0], 'model_seed': s, block: sc}
        context.write_json(bdir / ('%s_scores.json' % block), {'job_id': case, 'cells': cells, 'smoke': True})
    return bdir


def smoke(root: Path = ROOT) -> dict:
    out = paths(root)['smoke']
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    checks = []
    n_up = 5
    cs = _synthetic_case(out)
    sroot = out / 'pkg'
    sroot.mkdir()
    P_ = paths(sroot)
    for k in ('pool', 'historical', 'logs'):
        P_[k].mkdir(exist_ok=True)
    # geometry + case json
    geo = {H: hist_geometry(cs, H) for H in HS}
    checks.append({'check': 'historical geometry inside [t-1440, t), late target union ends <= t, CaseSpec c_a == near[:2] and e == late', 'ok': all(g['late_target_union'][1] <= cs.t for g in geo.values())})
    for H in HS:
        write_case_json(sroot, cs, H)
    # a current-t table for the pool (T-only overview at t) via the material stage of the current case
    cur_dir = out / 'current'
    ctx_t = ec.open_case(cs, cur_dir, stage='material')
    table_t = ctx_t.overview()['entities']
    pool = _synthetic_pool(cs, table_t)
    context.write_json(P_['pool'] / (cs.case_id + '.json'), pool)
    readiness = data_readiness(cs)
    checks.append({'check': 'data readiness on the synthetic roster (finite, hourly, scaler non-degenerate, 16 x 433 at both h)', 'ok': readiness['ok']})
    # frozen config stub + ledger (the smoke package keeps its own budget)
    context.write_json(P_['frozen'], {'package': PACKAGE + '-SMOKE', 'cases': {cs.case_id: cs.to_json()}, 'fixed_dev': {'SM': {'public_id': 'P_NoMixRecipe'}}, 'smoke': True})
    # ---- 1. poison test: rows >= h1 scaled x1000 must not change the h1 scaler, materials or trained weights (only the scored C_A / near / late change)
    poison = out / 'poison'
    poison.mkdir()
    src = Path(cs.csv_path)
    h1 = hist_case(cs, 'H1').t
    with src.open() as f, (poison / 'synthetic.csv').open('w', newline='') as g:
        for i, line in enumerate(f):
            if i == 0 or i - 1 < h1:
                g.write(line)
            else:
                parts = line.rstrip('\n').split(',')
                g.write(parts[0] + ',' + ','.join('%.4f' % (float(v) * 1000.0) for v in parts[1:]) + '\n')
    cs_p = ec.CaseSpec(dataset=cs.dataset, domain=cs.domain, case_id=cs.case_id, role=cs.role, t=cs.t, roster=cs.roster, case_index=cs.case_index, csv_path=str(poison / 'synthetic.csv'), total_hours=cs.total_hours)
    proot = out / 'pkg_poison'
    (proot / 'pool').mkdir(parents=True)
    (proot / 'historical').mkdir()
    (proot / 'logs').mkdir()
    context.write_json(proot / 'pool' / (cs.case_id + '.json'), pool)
    write_case_json(proot, cs_p, 'H1')
    t_smoke = time.time()
    led_p = DP.SafeLedger(proot / 'budget.json', max_fit_attempts=10, max_llm_requests=0, max_llm_tokens=0, max_wall_s=3600, max_retries=0)
    DP.set_pools(2, 1)
    rc, _ = run_worker(['--worker-build', proot, case_json_path(proot, cs.case_id, 'H1'), proot / 'pool' / (cs.case_id + '.json')], proot / 'logs' / 'build.log', 900)
    checks.append({'check': 'poisoned-file build worker ran', 'ok': rc == 0})
    rc, _ = run_worker(['--worker-build', sroot, case_json_path(sroot, cs.case_id, 'H1'), P_['pool'] / (cs.case_id + '.json')], P_['logs'] / 'build_H1.log', 900)
    checks.append({'check': 'clean-file H1 build worker ran', 'ok': rc == 0})
    jd_c, jd_p = hist_dir(sroot, cs.case_id, 'H1') / (cs.case_id + '_H1'), hist_dir(proot, cs.case_id, 'H1') / (cs.case_id + '_H1')
    with np.load(jd_c / 'scaler.npz') as a, np.load(jd_p / 'scaler.npz') as b:
        sc_eq = np.array_equal(a['mean'], b['mean']) and np.array_equal(a['scale'], b['scale']) and np.array_equal(a['mase'], b['mase'])
    reg_c, reg_p = ec.load_registry(jd_c), ec.load_registry(jd_p)
    mat_eq = True
    for m, r in reg_c.items():
        if r['path'] is None:
            continue
        with np.load(r['path']) as a, np.load(reg_p[m]['path']) as b:
            mat_eq = mat_eq and np.array_equal(a['Xc'], b['Xc']) and np.array_equal(a['yc'], b['yc'])
    ov_eq = context.read_json(jd_c / 'overview.json')['entities'] == context.read_json(jd_p / 'overview.json')['entities']
    hp_c, hp_p = context.read_json(jd_c / 'hist_pool.json'), context.read_json(jd_p / 'hist_pool.json')
    checks.append({'check': 'material stage reads [h-672, h) only: scaler / MASE / overview / every material identical with rows >= h poisoned x1000', 'ok': bool(sc_eq and mat_eq and ov_eq),
                   'scaler': bool(sc_eq), 'materials': bool(mat_eq), 'overview': bool(ov_eq), 'materials_compared': sorted(m for m, r in reg_c.items() if r['path'])})
    checks.append({'check': 'hist_pool identical (assignments, aliases, thresholds) on the poisoned file', 'ok': hp_c['candidates'] == hp_p['candidates']})
    phys_edit = next(r['phys_h'] for r in hp_c['candidates'].values() if r['label_t'].startswith('Edit'))
    seed = SEEDS[0]
    rc1, _ = run_worker(['--worker-fit', sroot, case_json_path(sroot, cs.case_id, 'H1'), phys_edit, seed, '--n-updates', n_up], P_['logs'] / 'fit_clean.log', 600)
    rc2, _ = run_worker(['--worker-fit', proot, case_json_path(proot, cs.case_id, 'H1'), phys_edit, seed, '--n-updates', n_up], proot / 'logs' / 'fit_poison.log', 600)
    cid = ec.cell_id(cs.case_id + '_H1', phys_edit, seed)
    a, b = context.read_json(jd_c / 'cells' / (cid + '.json')), context.read_json(jd_p / 'cells' / (cid + '.json'))
    code = ('import sys, torch; a = torch.load(sys.argv[1]); b = torch.load(sys.argv[2]); print(all(torch.equal(a[k], b[k]) for k in a) and set(a) == set(b))')
    w_eq = subprocess.run([sys.executable, '-B', '-c', code, a['model_path'], b['model_path']], capture_output=True, text=True, cwd=str(REPO), env=ES.worker_env()).stdout.strip() == 'True'
    checks.append({'check': 'training reads no row >= h: trained weights identical on the poisoned file while the scored C_A differs', 'ok': rc1 == 0 and rc2 == 0 and w_eq and a['scores']['c_a']['normalized_mse_macro'] != b['scores']['c_a']['normalized_mse_macro'],
                   'weights_equal': w_eq, 'c_a_clean': a['scores']['c_a']['normalized_mse_macro'], 'c_a_poison': b['scores']['c_a']['normalized_mse_macro'], 'rows_read_fit': a['rows_read']})
    checks.append({'check': 'fit record rows_read == [h-672, h+96) (the C_A block is scored, never trained; same protocol as the current t)', 'ok': a['rows_read'] == list(hist_case(cs, 'H1').evaluate_rows)})
    # ---- 2. full historical stage on the clean package (both h, tiny training), then scoring and the selectors
    led = DP.SafeLedger(P_['ledger'], max_fit_attempts=60, max_llm_requests=0, max_llm_tokens=0, max_wall_s=3600, max_retries=2)
    context.write_json(P_['stages'], {'historical': {'caps': {'max_fit_attempts': 60, 'max_llm_requests': 0, 'max_llm_tokens': 0, 'max_wall_s': 3600, 'max_retries': 2}, 'allocation': ALLOC['historical'], 'epoch_start': time.time(), 'local_start': now(), 'snapshot_at_start': {}}})
    DP.POOLS = None
    hist = historical_stage(sroot, cases=(cs.case_id,), n_updates=n_up, numeric=2)
    led = DP.SafeLedger(P_['ledger'], **TOTAL)
    n_phys = len(hp_c['physical_materials'])
    expected_fits = n_phys * 3 * 2 - 1                               # one clean H1 cell already fitted above
    checks.append({'check': 'historical stage: %d physical x 3 seeds x 2 h fitted (one cached), 0 failed, numeric pool used in parallel' % n_phys,
                   'ok': led.s['fit_attempts'] == expected_fits and led.s['fits_failed'] == 0 and hist['pools']['numeric_peak'] >= 2, 'fit_attempts': led.s['fit_attempts'], 'expected': expected_fits, 'pools': hist['pools']})
    hs = {H: context.read_json(hist_dir(sroot, cs.case_id, H) / (cs.case_id + '_' + H) / 'hist_scores.json') for H in HS}
    ok_blocks = all(hs[H]['rows_read'] == [hist_case(cs, H).t - spec.L, hist_case(cs, H).t + 384] and all(len(v['near']['per_origin_normalized_mse_mean']) == 4 and len(v['late']['per_origin_normalized_mse_mean']) == 4
                    and set(v['near']) == set(v['late']) for v in hs[H]['cells'].values()) for H in HS)
    checks.append({'check': 'historical scoring: rows [h-192, h+384), near and late each 4 origins, identical metric keys, 6 inferred + 2 reused origins per model', 'ok': ok_blocks and all(hs[H]['n_origins_inferred'] == 6 * hs[H]['n_models'] for H in HS)})
    # near origins 0-1 reproduce the fit's C_A per-origin values exactly
    rep = all(hs[H]['cells'][c]['near']['per_origin_normalized_mse_mean'][:2] == context.read_json(hist_dir(sroot, cs.case_id, H) / (cs.case_id + '_' + H) / 'cells' / (c + '.json'))['scores']['c_a']['per_origin_normalized_mse_mean']
              for H in HS for c in hs[H]['cells'])
    checks.append({'check': 'near origins 0-1 == cached C_A per-origin values (reuse, no re-inference)', 'ok': rep})
    # conditional policy: thresholds re-resolved on each h's T-only table; assignment == fresh compile on the h overview; h1 != h2 thresholds
    hp = {H: context.read_json(hist_dir(sroot, cs.case_id, H) / (cs.case_id + '_' + H) / 'hist_pool.json') for H in HS}
    cond_id = next(cid for cid, r in pool['pool'].items() if r['policy'] and r['policy']['rules'])
    fresh_ok, thr = True, {}
    for H in HS:
        tab = context.read_json(hist_dir(sroot, cs.case_id, H) / (cs.case_id + '_' + H) / 'overview.json')['entities']
        comp = ec.compile_plan_full(pool['pool'][cond_id]['policy'], tab)
        fresh_ok = fresh_ok and comp['assignment'] == hp[H]['candidates'][cond_id]['assignment_h'] and comp['resolved_thresholds'] == hp[H]['candidates'][cond_id]['resolved_thresholds_h']
        thr[H] = hp[H]['candidates'][cond_id]['resolved_thresholds_h']
    thr_t = pool['pool'][cond_id]['resolved_thresholds_t']
    checks.append({'check': 'conditional candidate: quantile threshold re-resolved on each h (differs from t and between h1 / h2), assignment == fresh compile on the h table', 'ok': fresh_ok and thr['H1'] != thr['H2'] and thr['H1'] != thr_t,
                   'thresholds': {'t': thr_t, **thr}, 'changed_vs_t': {H: hp[H]['candidates'][cond_id]['assignment_changed'] for H in HS}})
    # ---- 3. cross-h cache rejection: an H1 cell record copied under the H2 instance is refused (never a hit, never silently refitted)
    c1 = next(iter(hs['H1']['cells']))
    src_rec = context.read_json(hist_dir(sroot, cs.case_id, 'H1') / (cs.case_id + '_H1') / 'cells' / (c1 + '.json'))
    ch2 = hist_case(cs, 'H2')
    fake = hist_dir(sroot, cs.case_id, 'H2') / (cs.case_id + '_H2') / 'cells' / ('%s__%s__s%d.json' % (ch2.case_id, src_rec['material_id'], src_rec['model_seed']))
    backup = context.read_json(fake)
    context.write_json(fake, src_rec)
    try:
        cell_cached(fake, ch2, src_rec['material_id'], ec.load_registry(hist_dir(sroot, cs.case_id, 'H2') / (cs.case_id + '_H2'))[src_rec['material_id']]['key'])
        rejected = False
    except RuntimeError:
        rejected = True
    context.write_json(fake, backup)
    checks.append({'check': 'a cell record of another cut under the same material id is rejected as a cache conflict', 'ok': rejected})
    # ---- 4. selection: frozen, deterministic, write-once; evaluator refused before the freeze; re-run of the stage = 0 new fits
    try:
        readout(sroot)
        refused = False
    except PermissionError:
        refused = True
    checks.append({'check': 'evaluator refuses before selections.json is frozen', 'ok': refused})
    sel1 = select_stage(sroot, n_updates=n_up)
    sel_copy = copy.deepcopy(sel1)
    P_['selections'].unlink()
    sel2 = select_stage(sroot, n_updates=n_up)
    same = {k: sel_copy['cases'][cs.case_id][k] == sel2['cases'][cs.case_id][k] for k in ('picks', 'ranks', 'scores_at_selection')}
    checks.append({'check': 'selection deterministic (picks / ranks / scores identical on recomputation) and write-once', 'ok': all(same.values()), 'detail': same})
    try:
        write_once(P_['selections'], {})
        wo = False
    except RuntimeError:
        wo = True
    checks.append({'check': 'selections.json cannot be overwritten', 'ok': wo})
    before = led.s['fit_attempts']
    DP.POOLS = None
    historical_stage(sroot, cases=(cs.case_id,), n_updates=n_up, numeric=2)
    after = context.read_json(P_['ledger'])['fit_attempts']
    checks.append({'check': 'resume of the historical stage: 0 new fits (cells / materials / scores cached)', 'ok': before == after})
    s = sel2['cases'][cs.case_id]
    checks.append({'check': 'four strategies pick a pool id; Fixed_dev == the frozen public id; ties broken by display id', 'ok': all(s['picks'][k] in pool['pool'] for k in STRATEGIES) and s['picks']['Fixed_dev'] == pool['reference_ids']['P_NoMixRecipe']})
    # ---- 5. evaluator on fake F0 labels
    bdir = _fake_f0_labels(out, pool, cs.case_id)
    pool['branch_dir_f0'] = str(bdir)
    context.write_json(P_['pool'] / (cs.case_id + '.json'), pool)
    cfg = context.read_json(P_['frozen'])
    context.write_json(P_['frozen'], cfg)
    res = readout(sroot)
    checks.append({'check': 'evaluator: readout, verdict, tables written on fake labels', 'ok': P_['result'].exists() and P_['tables'].exists() and res['verdict']['category'] in ('DEFAULT_RECOVERY_ONLY', 'NON_DEFAULT_IMPROVEMENT_SIGNAL', 'NEAR_EQUALLY_EFFECTIVE', 'MIXED_OR_WORSE')})
    # ---- 6. lock
    with DP.PackageLock(sroot, 'smoke'):
        rc_probe = subprocess.run([sys.executable, '-B', '-m', MODULE, '--lock-probe', str(sroot)], cwd=str(REPO), env=ES.worker_env()).returncode
    rc_free = subprocess.run([sys.executable, '-B', '-m', MODULE, '--lock-probe', str(sroot)], cwd=str(REPO), env=ES.worker_env()).returncode
    checks.append({'check': 'package lock: probe fails while held (3), succeeds when free (0)', 'ok': rc_probe == 3 and rc_free == 0})
    no_llm = not any(p_.name.endswith('.json') and 'llm' in p_.name for p_ in out.rglob('*'))
    checks.append({'check': 'no LLM client / request file created', 'ok': no_llm and led.s['llm_requests'] == 0})
    result = {'status': 'PASS' if all(c['ok'] for c in checks) else 'FAIL', 'n_checks': len(checks), 'checks': checks, 'seconds': time.time() - t_smoke, 'n_updates_smoke': n_up, 'finished_local': now()}
    context.write_json(out / 'smoke_result.json', result)
    print('SMOKE', result['status'], '%d/%d' % (sum(c['ok'] for c in checks), len(checks)), 'secs %.0f' % result['seconds'], flush=True)
    for c in checks:
        if not c['ok']:
            print('  FAILED:', c['check'], {k: v for k, v in c.items() if k not in ('check', 'ok')}, flush=True)
    return result


# ============================================================================= CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--monitor', action='store_true')
    ap.add_argument('--interval', type=float, default=60.0)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--worker-build', nargs=3, metavar=('ROOT', 'CASEJSON', 'POOLJSON'))
    ap.add_argument('--worker-fit', nargs=4, metavar=('ROOT', 'CASEJSON', 'PHYS', 'SEED'))
    ap.add_argument('--worker-histscore', nargs=2, metavar=('ROOT', 'CASEJSON'))
    ap.add_argument('--n-updates', type=int, default=None)
    ap.add_argument('--lock-probe')
    a = ap.parse_args()
    root = Path(a.root)
    if a.lock_probe:
        sys.exit(DP.lock_probe(Path(a.lock_probe)))
    elif a.worker_build:
        worker_build(*a.worker_build)
    elif a.worker_fit:
        worker_fit(a.worker_fit[0], a.worker_fit[1], a.worker_fit[2], int(a.worker_fit[3]), n_updates=a.n_updates)
    elif a.worker_histscore:
        worker_histscore(*a.worker_histscore)
    elif a.preflight:
        cfg = preflight(root)
        print(json.dumps({k: cfg[k] for k in ('package', 'n_candidates', 'geometry', 'fixed_dev', 'environment', 'budget')}, indent=1, ensure_ascii=False, default=str))
    elif a.smoke:
        smoke(root)
    elif a.wiring:
        print(json.dumps(wiring(root), indent=1, ensure_ascii=False, default=str))
    elif a.run:
        package_run(root)
    elif a.result:
        readout(root)
        print(paths(root)['tables'].read_text(encoding='utf-8'))
    elif a.monitor:
        monitor(root, a.interval, a.once)


if __name__ == '__main__':
    main()
