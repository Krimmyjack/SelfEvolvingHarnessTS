"""DEV-AUG-OFFLINE-SKILL (docs/DEV_AUG_OFFLINE_SKILL_TASK_2026-09-23.md): offline-learned experience x zero-feedback Fast construction on
electricity D01 / traffic D02 / solar D03 (case ranges frozen from docs/AUG_TASK_FAMILY_DIVISION_V1.json).

This file implements stage A of the two-part execution (task §5): 24 no-card zero-feedback Fast trajectories on the reused screen Source cases,
one external evaluation after every commit is frozen, and the checkpoint report with the measured trajectory cost. Stages B-D (formation,
zero-feedback card selection, 40-case test) are specified in the task book and start only after the remaining budget is confirmed.

Reused unchanged (read-only imports): the entity-split case profile (entity_case), the parallel pieces of DEV-DOMAIN-AUG-DECISION-PRIORITY
(package lock, thread-safe ledger, numeric / HTTP pools, metered client, fit_cells over the entity-split fit worker), the screen's cached
materials / fits / E scores (as a keyed physical cache, never rewritten) and its original formulas for the three added T-only fields.
New here: the zero-feedback Fast loop (methods/ttha/batch_zero_feedback.py), a zero-feedback case adapter (compact case card, per-entity
table on demand, extended predicate vocabulary compiled for real, unfitted commit), the per-trajectory token gate, the post-freeze evaluator
with the screen cache lookup, the stage-A readout.

  --smoke | --wiring | --stage-a [--numeric N] | --evaluate-a | --report-a | --status
  subprocess entries: --worker-material ROOT SPLIT CASE PHYS | --worker-score ROOT SPLIT CASE OUT PHYS...
"""
from __future__ import annotations

import os

KMP_AT_START = os.environ.get('KMP_DUPLICATE_LIB_OK')

import argparse
import copy
import ctypes
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha import batch_zero_feedback as zf
from methods.ttha.batch_base import budget, context, data, policy, spec, tempo_aug as ta
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_skill as W              # read-only: MODEL, the Fast system text
from evaluation.main_protocol_p4 import batch_research_domain_aug_entity_split as ES               # read-only: fit worker, adapter base, texts
from evaluation.main_protocol_p4 import batch_research_domain_aug_decision_priority as DP          # read-only: lock, ledger, pools, client, fit_cells
from evaluation.main_protocol_p4 import batch_research_aug_task_family_screen as SC                # read-only: screen cache, field formulas

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_aug_offline_skill'
MODULE = 'evaluation.main_protocol_p4.batch_research_aug_offline_skill'
TASK = 'docs/DEV_AUG_OFFLINE_SKILL_TASK_2026-09-23.md'
DIVISION = REPO / 'docs' / 'AUG_TASK_FAMILY_DIVISION_V1.json'
PACKAGE = 'DEV-AUG-OFFLINE-SKILL'
EXPOSURE = 'DEVELOPMENT_ZERO_FEEDBACK_REUSE'           # task §2: not Natural Final; Source cases were exposed by the screen
DOMAINS = ('D01', 'D02', 'D03')
SEEDS = SC.SEEDS                                        # (20269181, 20269182, 20269183)
MODEL = W.MODEL                                         # requested cpa-grok-4.6, returned identity grok-4.6-build, temperature 0 (previous formal package)
MAX_OUTPUT_TOKENS = W.MAX_OUTPUT_TOKENS                 # 12,000 (unchanged sampling configuration)
LIMITS = zf.Limits(max_calls=6, max_tools=18, max_materials=4)     # task §4
TRAJ_TOKEN_CAP = 200_000                                # input + output per trajectory (task §4)
MAX_CORRECTIONS = 2                                     # the previous package's tool-correction budget; response-format errors share it
EVIDENCE_ROUNDTRIP = True
PUBLIC = ec.PUBLIC                                      # None, FixedMixup, P_AmpResample, P_NoMixRecipe
REFS = {'None': 'None', 'NoMix': 'P_NoMixRecipe', 'Fixed_source': {'D01': 'C_censor', 'D02': 'C_censor', 'D03': 'P_NoMixRecipe'},
        'Fixed_global_source': 'C_shock'}               # task §8, frozen from the screen Source table
EXTRA_FIELDS = ('zero_fraction', 'flat_nonzero_fraction', 'weekly_excess_r2')     # task §4: screen formulas, all arms
OBS_FIELDS_ZF = list(spec.OBS_FIELDS) + list(EXTRA_FIELDS)
ALLOWED_FEATURES = frozenset('batch_median_' + f for f in OBS_FIELDS_ZF)
STAGE_A = {'trajectories': 24, 'max_logical_requests': 144, 'extra_transport_attempts': 8, 'max_tokens': 5_000_000,
           'max_new_fits': 82, 'fits_final': 72, 'fits_wiring': 6, 'fit_retries': 4}
CONCURRENCY = {'http': 4, 'numeric': 2}                 # numeric 2 by default (host free memory ~1.6 GB at launch); wiring measures
MIN_FREE_GB = 1.0                                       # a numeric subprocess starts only above this free physical memory
FIT_THREADS = ES.FIT_THREADS                            # 8 (the entity-split fit worker; the screen used the same)
AUTO_ACCEPT_CLASS = ('TRANSPORT_TRANSIENT_FAULT on attempt 0 or on the bounded attempt 1 of the same logical request (the user decisions recorded in '
                     '_scratch/dev_domain_aug_temporal_coverage/operator_decision_transient_class.json, applied here per task §11); any other class stops dispatch')

now = DP.now
write_new = DP.write_new
PackageLock = DP.PackageLock
SafeLedger = DP.SafeLedger


def paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'split': root / 'split_copy.json', 'frozen': root / 'frozen_config.json', 'common': root / 'common',
            'stage_a': root / 'stage_a', 'llm': root / 'stage_a' / 'llm', 'eval': root / 'stage_a' / 'evaluation', 'smoke': root / 'smoke',
            'wiring': root / 'wiring', 'lock': root / 'package.lock', 'incidents': root / 'incidents', 'logs': root / 'logs'}


# ============================================================================= split (frozen case ranges, task §2)
def build_split() -> dict:
    d = context.read_json(DIVISION)
    out = {'setting_id': 'AUG_OFFLINE_SKILL_V1', 'package': PACKAGE, 'status': 'FROZEN_FOR_%s' % PACKAGE.replace('-', '_'),
           'source_document': str(DIVISION.relative_to(REPO)), 'source_status_at_freeze': d['status'], 'exposure': EXPOSURE,
           'cohort_size': d['cohort_size'], 'model_seeds': d['model_seeds'], 'domains': {}}
    for dom in DOMAINS:
        D = copy.deepcopy(d['domains'][dom])
        out['domains'][dom] = D
    return out


def stage_cases(split: dict) -> dict:
    """task §2 table: Source = reuses_screen_case only (8 / 8 / 8); Select 9 / 9 / 6; Test 16 / 16 / 8."""
    out = {'source': {}, 'select': {}, 'test': {}}
    for dom, D in split['domains'].items():
        for g in D['groups']:
            if g['stage'] == 'source' and not g['reuses_screen_case']:
                continue
            out[g['stage']].setdefault(dom, []).append(g['case_id'])
    return out


def cases_of(split: dict) -> dict:
    return ec.cases_from_split(split)


# ============================================================================= the three added T-only fields (screen formulas) and the extended vocabulary
def extra_fields(ctx: ec.CaseContext) -> list:
    lo, hi = ctx.job.train_range
    seg = ctx.slice.rows(lo, hi)
    ts = [ctx.slice.timestamp_of_row(r) for r in range(lo, hi)]
    hours, wd = np.array([x.hour for x in ts]), np.array([x.weekday() for x in ts])
    rows = []
    for j in range(ec.COHORT_SIZE):
        f = SC.entity_features(seg[:, j], hours, wd)
        rows.append({k: float(f[k]) for k in EXTRA_FIELDS})
    return rows


def _validate_predicate_zf(p) -> None:
    if not isinstance(p, dict):
        raise policy.PolicyError('predicate must be an object')
    if set(p) == {'const'}:
        if not isinstance(p['const'], bool):
            raise policy.PolicyError('const must be a boolean')
        return
    if set(p) == {'not'}:
        return _validate_predicate_zf(p['not'])
    if set(p) in ({'all'}, {'any'}):
        k = next(iter(p))
        if not isinstance(p[k], list) or not p[k]:
            raise policy.PolicyError('%s must be a non-empty list' % k)
        for q in p[k]:
            _validate_predicate_zf(q)
        return
    if set(p) != {'feature', 'op', 'value'}:
        raise policy.PolicyError('predicate leaf must contain exactly feature, op, value (got %s)' % sorted(p))
    if p['feature'] not in OBS_FIELDS_ZF:
        raise policy.PolicyError('unknown feature %r' % p['feature'])
    if p['op'] not in spec.PREDICATE_OPS:
        raise policy.PolicyError('unknown predicate op %r' % p['op'])
    v = p['value']
    if isinstance(v, dict):
        if set(v) != {'quantile'} or isinstance(v['quantile'], bool) or not isinstance(v['quantile'], (int, float)) or not 0.0 <= float(v['quantile']) <= 1.0:
            raise policy.PolicyError("value object must be {'quantile': q in [0,1]}")
    elif not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
        raise policy.PolicyError("value must be a finite number or {'quantile': q}")


def compile_zf(pol: dict, table: list) -> dict:
    """ec.compile_plan_full with the extended predicate vocabulary (OBS_FIELDS + EXTRA_FIELDS) compiled for real on the extended table."""
    if not isinstance(pol, dict) or 'default' not in pol:
        raise policy.PolicyError("policy must be an object with a 'default'")
    if set(pol) - {'default', 'rules', 'rationale', 'observation_fields_used'}:
        raise policy.PolicyError('unknown policy keys %s' % sorted(set(pol) - {'default', 'rules', 'rationale', 'observation_fields_used'}))
    if not isinstance(pol['default'], dict) or set(pol['default']) != {'steps'}:
        raise policy.PolicyError("default must be {'steps': [...]}")
    clean = {'default': {'steps': ec.validate_steps_full(pol['default']['steps'])}, 'rules': []}
    rules = pol.get('rules', []) or []
    if not isinstance(rules, list) or len(rules) > ta.MAX_RULES:
        raise policy.PolicyError('rules must be a list of at most %d rules' % ta.MAX_RULES)
    for r in rules:
        if not isinstance(r, dict) or set(r) != {'when', 'steps'}:
            raise policy.PolicyError("each rule must be {'when': predicate, 'steps': [...]}")
        _validate_predicate_zf(r['when'])
        clean['rules'].append({'when': r['when'], 'steps': ec.validate_steps_full(r['steps'])})
    rationale = str(pol.get('rationale', ''))[:500]
    policy.validate_text(rationale)
    clean['rationale'] = rationale
    clean['observation_fields_used'] = [f for f in (pol.get('observation_fields_used') or []) if f in OBS_FIELDS_ZF]
    resolved, assignment, rule_index, n_unknown = {}, [], [], 0
    for row in table:
        chosen, ri = clean['default']['steps'], -1
        for i, r in enumerate(clean['rules']):
            v = policy._eval(r['when'], row, table, resolved)
            if v is None:
                n_unknown += 1
            if v is True:
                chosen, ri = r['steps'], i
                break
        assignment.append([dict(s) for s in chosen])
        rule_index.append(ri)
    return {'policy': clean, 'assignment': assignment, 'rule_index': rule_index, 'resolved_thresholds': resolved, 'n_unknown': n_unknown}


FIELD_DEFINITIONS = {**ES.FIELD_DEFINITIONS,
                     'zero_fraction': 'share of the 672 raw T values that are exactly 0',
                     'flat_nonzero_fraction': 'share of consecutive T steps with an unchanged NONZERO value (diff == 0 and value != 0), over 671 steps',
                     'weekly_excess_r2': 'max(0, R2 of hour-of-week means - R2 of hour-of-day means) on the standardized T: weekly structure beyond the daily profile'}


def _sig(x, n=4):
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return float('%.*g' % (n, x))
    if isinstance(x, dict):
        return {k: _sig(v, n) for k, v in x.items()}
    if isinstance(x, list):
        return [_sig(v, n) for v in x]
    return x


def _q(vals: list) -> dict:
    v = np.array([x for x in vals if x is not None and math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return None
    return {'min': float(v.min()), 'p25': float(np.quantile(v, .25)), 'median': float(np.median(v)), 'p75': float(np.quantile(v, .75)), 'max': float(v.max())}


# ============================================================================= physical cache (package common dir; the screen cache is read, never written)
_CASE_LOCKS: dict = {}
_CASE_LOCKS_GUARD = threading.Lock()


def case_lock(case: str) -> threading.Lock:
    with _CASE_LOCKS_GUARD:
        return _CASE_LOCKS.setdefault(case, threading.Lock())


def screen_dir(case: str) -> Path:
    return SC.case_dir() / case


def prepare_common_source(common: Path, cs: ec.CaseSpec) -> dict:
    """common/<case> = the package view of a reused screen case: T scaler / overview computed here (T rows only), the screen registry entries
    (materials by assignment key; files stay in the screen directory) as the keyed physical cache. Binding: the scaler must equal the screen's."""
    jd = common / cs.case_id
    reg_p = ec.aug_dir(jd) / 'index.json'
    if reg_p.exists():
        return {'case': cs.case_id, 'prepared': 'existing'}
    ctx = ec.open_case(cs, common, 'material')
    ctx.overview()
    mine, scr = np.load(jd / 'scaler.npz'), np.load(screen_dir(cs.case_id) / 'scaler.npz')
    if not (np.array_equal(mine['mean'], scr['mean']) and np.array_equal(mine['scale'], scr['scale']) and [str(x) for x in scr['roster']] == list(cs.roster)):
        raise RuntimeError('screen cache binding failed for %s (scaler / roster)' % cs.case_id)
    sreg = ec.load_registry(screen_dir(cs.case_id))
    ec.save_registry(jd, {m: {**r, 'screen_cache': True} for m, r in sreg.items()})
    return {'case': cs.case_id, 'prepared': 'new', 'screen_materials': sorted(sreg)}


def free_gb() -> float:
    return SC.free_gb()


def wait_memory(tag: str) -> None:
    t0 = time.time()
    while free_gb() < MIN_FREE_GB:
        if time.time() - t0 > 1800:
            raise RuntimeError('host free memory below %.1f GB for 30 min before %s' % (MIN_FREE_GB, tag))
        time.sleep(5)


def worker_env() -> dict:
    return ES.worker_env()


def run_numeric(args: list, log: Path, timeout: float, module: str = MODULE) -> tuple:
    log.parent.mkdir(parents=True, exist_ok=True)
    with DP.pools().numeric():
        wait_memory(' '.join(str(a) for a in args[:1]))
        t0 = time.time()
        try:
            with log.open('w', encoding='utf-8') as fh:
                p = subprocess.run([sys.executable, '-B', '-m', module] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=timeout, env=worker_env())
            return p.returncode, time.time() - t0
        except subprocess.TimeoutExpired:
            return -999, time.time() - t0


def ensure_material_zf(common: Path, split_path: Path, case: str, assignment: list, *, ledger=None) -> dict:
    """The physical record of an assignment in common/<case>: an existing entry by key (screen or earlier build), None, or a new TA material
    built by one worker (torch 1 thread, as the screen built its materials). Serial per case (case lock)."""
    with case_lock(case):
        reg = ec.load_registry(common / case)
        key = ec.material_key(assignment)
        for r in reg.values():
            if r['key'] == key:
                return r
        if ec.is_identity(assignment):
            return reg['None']
        used = [int(r['material_index']) for r in reg.values() if r.get('material_index')]
        phys = 'TA%03d' % (max(used + [0]) + 1)
        context.write_json(ec.aug_dir(common / case) / (phys + '__request.json'), {'assignment': assignment, 'requested_local': now(), 'package': PACKAGE})
        rc, secs = run_numeric(['--worker-material', common, split_path, case, phys], common / case / 'fit_logs' / ('material_%s.log' % phys), 900)
        if ledger is not None:
            ledger.add('material_wall_seconds', secs)
            ledger.event(kind='material_built', case=case, phys=phys, seconds=round(secs, 2), ok=rc == 0)
        err = ec.aug_dir(common / case) / (phys + '__error.json')
        if rc != 0:
            if err.exists():
                raise br.ToolInputError('plan rejected at execution (no material): %s' % context.read_json(err)['message'], execution_status='PROGRAM_EXECUTION_REJECTED')
            raise RuntimeError('material worker failed for %s %s' % (case, phys))
        return ec.load_registry(common / case)[phys]


# ============================================================================= zero-feedback adapter
PUBLIC_LABEL = {'None': 'None (no augmentation: parent view only)', 'FixedMixup': 'FixedMixup (historical public reference)',
                'P_AmpResample': 'P_AmpResample (uniform tp_amplitude -> tp_resample)', 'P_NoMixRecipe': 'P_NoMixRecipe (uniform preset; "NoMix")'}


def public_candidates(case: str) -> tuple:
    out = []
    for mid in PUBLIC:
        ms = {'kind': 'public_reference', 'label': PUBLIC_LABEL[mid]}
        if mid in ec.PUBLIC_STEPS:
            ms.update({'programs': [ec.PUBLIC_STEPS[mid]], 'entity_program': [0] * ec.COHORT_SIZE})
        else:
            ms['definition'] = ec.FIXED_MIXUP['definition']
        out.append(br.Candidate(mid, ms, case))
    return tuple(out)


class ZFAdapter(ES.CaseAdapter):
    """Zero-feedback view of one case for one branch: no evaluate, no score, no cell; construction and inspection are the parent's tools
    (the material worker is this package's); predicates may use the three added T-only fields, which the compiler really uses."""

    def __init__(self, run_dir, case, ledger, *, common, split_path, cs: ec.CaseSpec):
        super().__init__(run_dir, case, ledger, REPO, common=common, split_path=split_path, cs=cs, public_ids=PUBLIC, tool_error_feedback=True, seeds=SEEDS,
                         fit_fn=lambda *a: (_ for _ in ()).throw(PermissionError('zero-feedback branch: no fit')))
        base = self.ctx.overview()['entities']
        extra = extra_fields(self.ctx)
        self.table = [{**row, **ex} for row, ex in zip(base, extra)]
        self.build_seconds = 0.0

    # --- request pieces (compact; per-entity rows only through overview)
    def case_card(self) -> dict:
        ov = self.ctx.overview()
        geo = {k: v for k, v in ov['geometry'].items() if k != 'c_a_origins_relative_to_t'}
        geo['scored_block'] = ('computed externally after every commit is frozen and never shown here: four 48-hour forecasts at origins t+192, t+240, t+288, '
                               't+336 (t = end of T), normalized MSE, entity macro')
        return {'case_id': self.case, 'domain_id': self.cs.domain, 'role': self.cs.role, 'dataset_exposure': EXPOSURE, 'train_rows': ov['train_rows'],
                'n_entities': ec.COHORT_SIZE, 'geometry': geo, 'consumer': ec.CONSUMER, 'fields': OBS_FIELDS_ZF,
                'field_definitions': FIELD_DEFINITIONS, 'actions': ES.action_table(),
                'entity_rows': 'per-entity values of every field: overview tool (all rows or chosen entity_indices); T-only'}

    def batch_summary(self) -> dict:
        summ = {f: _sig(_q([r.get(f) for r in self.table])) for f in OBS_FIELDS_ZF}
        return {'per_field_quantiles_over_entities': {k: v for k, v in summ.items() if v is not None},
                'batch_features': {'batch_median_' + f: v['median'] for f, v in summ.items() if v is not None},
                'note': 'min / p25 / median / p75 / max across the N entities of T-only values (4 significant digits); batch_median_<field> is the guidance-matching vocabulary'}

    def overview_table(self, arguments) -> dict:
        if set(arguments) - {'entity_indices'}:
            self._reject('overview takes only optional entity_indices')
        idx = arguments.get('entity_indices', list(range(self.n)))
        if not isinstance(idx, list) or not idx or any(type(i) != int or not 0 <= i < self.n for i in idx):
            self._reject('entity_indices must be a non-empty list of integers in 0..%d' % (self.n - 1))
        return {'fields': OBS_FIELDS_ZF, 'rows': {'entity_%d' % i: _sig({f: self.table[i].get(f) for f in OBS_FIELDS_ZF}) for i in idx},
                'note': 'T-only; 4 significant digits (the compiler uses full precision)'}

    def inspect_data(self, arguments, *, remaining_seconds):
        return _sig(super().inspect_data(arguments, remaining_seconds=remaining_seconds), 5)

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
            compiled = compile_zf(arguments['policy'], self.table)
        except policy.PolicyError as exc:
            self._reject(str(exc), legal_program_forms=['[] (no augmentation)', '1-3 of %s' % list(ta.PRIMITIVES), '[{"op":"%s"}]' % ta.RECIPE,
                                                        '[{"op":"%s","disabled_ops":[0-2 names]}]' % ta.RECIPE_EDIT], valid_observation_fields=OBS_FIELDS_ZF)
        key = ec.material_key(compiled['assignment'])
        same = next((m for m, r in reg.items() if r.get('key') == key), None)
        if same is not None:
            self._reject('identical complete assignment to existing plan %s (same material): reuse that id; no new material or slot' % same, alias_of=same,
                         execution_status='DUPLICATE_ASSIGNMENT')
        t0 = time.time()
        phys = ensure_material_zf(self.common, self.split_path, self.case, compiled['assignment'], ledger=self.ledger)
        self.build_seconds += time.time() - t0
        rec = {**phys, 'material_id': mid, 'phys_id': phys['material_id'], 'compiled': compiled}
        reg = self._reg()
        reg[mid] = rec
        ec.save_registry(self.ctx.job_dir, reg)
        programs, idx = [], []
        for steps in compiled['assignment']:
            if steps not in programs:
                programs.append(steps)
            idx.append(programs.index(steps))
        ms = {'kind': 'constructed', 'profile': ec.PROFILE_VERSION, 'policy': compiled['policy'], 'programs': programs, 'entity_program': idx,
              'rule_index': compiled['rule_index'], 'resolved_thresholds': _sig(compiled['resolved_thresholds'], 6), 'n_unknown': compiled['n_unknown'],
              'label': ec.assignment_label(compiled['assignment']), 'execution_order': 'fixed: ' + ' -> '.join(ta.PRIMITIVES)}
        self.specs[mid] = ms
        context.write_json(ec.aug_dir(self.ctx.job_dir) / (mid + '__compiled.json'), {'material_spec': ms, 'assignment': compiled['assignment'], 'key': key,
                                                                                       'phys_id': phys['material_id']})
        return br.Candidate(mid, ms, self.case)

    def evaluate(self, candidate, seeds, *, feedback, remaining_seconds):
        raise PermissionError('zero-feedback branch: evaluate does not exist')

    def baselines(self):
        raise PermissionError('zero-feedback branch: public references are unfitted candidates')

    def restore_built(self, mid):
        p = ec.aug_dir(self.ctx.job_dir) / (mid + '__compiled.json')
        return br.Candidate(mid, context.read_json(p)['material_spec'], self.case)

    def check_commit(self, plan_id, candidate) -> None:
        reg = self._reg()
        if plan_id not in reg:
            raise br.ToolInputError('commit needs a constructed plan or a public reference of this case')
        r = reg[plan_id]
        if r.get('kind') != 'none' and r.get('path') and not Path(r['path']).exists():
            raise RuntimeError('physical material missing for %s' % plan_id)
        if (self.ctx.job_dir / 'cells').exists() and any((self.ctx.job_dir / 'cells').iterdir()):
            raise RuntimeError('a zero-feedback branch holds fitted cells')

    def commit_record(self, plan_id: str, reason: str) -> dict:
        r = self._reg()[plan_id]
        return {'job_id': self.case, 'profile': ec.PROFILE_VERSION, 'plan_id': plan_id, 'physical_material': r.get('phys_id', r['material_id']),
                'material_key': r['key'], 'label': self._spec_of(plan_id).get('label') if plan_id not in PUBLIC else plan_id, 'reason': str(reason)[:1000],
                'status': 'COMMITTED_UNFITTED', 'fitted': False, 'committed_at_local': now(), 'constructed_plans': [m for m in self._reg() if m not in PUBLIC]}


# ============================================================================= metered client (raw text) and per-trajectory token gate
class UnknownUsageBlock(RuntimeError):
    """A never-sent request (unknown usage outside the accepted class is pending an operator decision): the trajectory stays resumable."""


class ZFMeteredClient(DP.MeteredClient):
    """DP's metered client with two changes: the raw response text is returned (the zero-feedback loop parses it, so a malformed envelope is a
    correctable contract error); a failed HTTP attempt of the standing accepted class is accepted in-process with one incident file each."""

    def __init__(self, ledger, out, *, http_cap: int, stage: str, incidents: Path, extra_cap: int):
        super().__init__(ledger, out, http_cap=http_cap, stage=stage)
        self.incidents, self.extra_cap = Path(incidents), int(extra_cap)

    def call(self, role, unit, payload, system, *, max_tokens=MAX_OUTPUT_TOKENS, meta=None, request_timeout=300.):
        led, meta = self.ledger, dict(meta or {})
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        upper = len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 2048 + max_tokens
        with led.lock:
            if self.fatal:
                raise rt.llm.AccountFault('backend identity/authentication previously failed')
            led.check_llm()
            if rt.unknown_usage_blocks(led):
                raise UnknownUsageBlock('unknown usage of a non-accepted class prevents further budgeted calls')
            used = led.s['llm_tokens_in'] + led.s['llm_tokens_out'] + led.s.get('token_reserved_failed_upper', 0)
            if used + 2 * upper > led.s['caps']['max_llm_tokens']:
                raise br.BudgetExhausted('insufficient conservative remaining stage token budget')
            led.s['llm_requests'] += 1
            number = led.s['llm_requests']
            led.s['events'].append({'kind': 'llm_request', 'request': number, 'role': role, 'unit': unit, 'stage': self.stage, **meta, 'reserved_upper': upper, 'epoch': time.time()})
            led._save()
            prefix = self.out / ('%03d_%s' % (number, role))
            write_new(str(prefix) + '_request.json', {'role': role, 'unit': unit, 'stage': self.stage, **meta, 'request': number, 'messages': messages,
                                                     'requested_model': MODEL['requested'], 'temperature': MODEL['temperature'], 'max_tokens': max_tokens, 'sent_local': now()})
        t_q = time.time()
        with DP.pools().http():
            wait = time.time() - t_q
            for attempt in range(2):
                with led.lock:
                    led.check_wall()
                    if led.s['llm_http_attempts'] >= self.http_cap:
                        raise br.BudgetExhausted('HTTP cap')
                    if attempt == 1 and led.s.get('extra_transport_attempts', 0) >= self.extra_cap:
                        raise rt.llm.TransportFault('EXTRA_TRANSPORT_BUDGET_EXHAUSTED')
                    led.s['llm_http_attempts'] += 1
                    if attempt == 1:
                        led.s['extra_transport_attempts'] = led.s.get('extra_transport_attempts', 0) + 1
                    led._save()
                try:
                    response = self._transport(messages, max_tokens, min(float(request_timeout), led.remaining()))
                except Exception as exc:  # noqa: BLE001
                    kind = rt.llm.classify_fault(exc)
                    write_new(str(prefix) + '_attempt%d.json' % attempt, {'failure_kind': kind, 'exception_type': type(exc).__name__, 'request': number, 'stage': self.stage, **meta, 'epoch': time.time()})
                    with led.lock:
                        led.s['llm_tokens_unknown'] += 1
                        led.s['llm_failed_attempts'] = led.s.get('llm_failed_attempts', 0) + 1
                        led.s['token_reserved_failed_upper'] = led.s.get('token_reserved_failed_upper', 0) + upper
                        led.s['events'].append({'kind': 'llm_attempt_failed', 'request': number, 'attempt': attempt, 'failure_kind': kind, 'stage': self.stage, **meta, 'epoch': time.time()})
                        if kind == 'TRANSPORT_TRANSIENT_FAULT':
                            led.s['unknown_usage_accepted'] = led.s.get('unknown_usage_accepted', 0) + 1
                            n_inc = led.s['unknown_usage_accepted']
                            led.s['events'].append({'kind': 'unknown_usage_accepted', 'request': number, 'attempt': attempt, 'rule': 'standing transient class', 'epoch': time.time()})
                        led._save()
                        if kind == 'ACCOUNT_OR_PERMISSION_FAULT':
                            self.fatal = True
                    if kind == 'TRANSPORT_TRANSIENT_FAULT':
                        write_new(self.incidents / ('operator_incident_%d_transient.json' % n_inc),
                                  {'local': now(), 'request': number, 'attempt': attempt, 'stage': self.stage, **meta, 'exception_type': type(exc).__name__,
                                   'reserved_upper_tokens': upper, 'accepted_under': AUTO_ACCEPT_CLASS})
                    if kind == 'ACCOUNT_OR_PERMISSION_FAULT':
                        raise rt.llm.AccountFault(kind) from None
                    if kind == 'TRANSPORT_TRANSIENT_FAULT' and attempt == 0:
                        continue
                    raise rt.llm.TransportFault(kind) from None
                write_new(str(prefix) + '_response.json', response.model_dump(mode='json'))
                usage = response.usage
                pt, ct = getattr(usage, 'prompt_tokens', None), getattr(usage, 'completion_tokens', None)
                with led.lock:
                    if pt is None or ct is None:
                        led.s['llm_tokens_unknown'] += 1
                    else:
                        led.s['llm_tokens_in'] += pt
                        led.s['llm_tokens_out'] += ct
                        ut = self.unit_tokens.setdefault(unit, {'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0})
                        ut['requests'] += 1
                        ut['prompt_tokens'] += pt
                        ut['completion_tokens'] += ct
                    led.s['events'].append({'kind': 'llm_finished', 'role': role, 'unit': unit, 'request': number, 'stage': self.stage, **meta, 'attempts': attempt + 1,
                                            'prompt_tokens': pt, 'completion_tokens': ct, 'queue_wait_s': round(wait, 3), 'returned_model': response.model,
                                            'message_bytes': upper - 2048 - max_tokens, 'epoch': time.time()})
                    if response.model != MODEL['returned_required']:
                        self.fatal = True
                    led._save()
                if response.model != MODEL['returned_required']:
                    raise rt.llm.AccountFault('returned model mismatch')
                text = response.choices[0].message.content if response.choices else ''
                print('LLM', role, unit, 'request', number, 'usage', pt, ct, 'wait %.1fs' % wait, flush=True)
                return (text if isinstance(text, str) else None), {'request': number, 'prompt_tokens': pt, 'completion_tokens': ct, 'attempts': attempt + 1,
                                                                   'message_bytes': upper - 2048 - max_tokens}
        raise AssertionError('unreachable')


class TrajectoryClient:
    """One trajectory's view: the free proxy pre-check, the token gate (task §4: 200k per trajectory; a final commit-only request is issued
    while the next one would not fit), per-trajectory token accounting from the returned usage."""

    def __init__(self, client, unit: str, system: str, *, case: str, arm: str, spent: int = 0, ratio: float | None = None):
        self.client, self.unit, self.system, self.case, self.arm = client, unit, system, case, arm
        self.spent, self.ratio, self.calls = int(spent), ratio, 0
        self.sys_bytes = len(json.dumps([{'role': 'system', 'content': system}], ensure_ascii=False).encode('utf-8'))

    def _est(self, req_bytes: int) -> float:
        r = self.ratio if self.ratio else 0.5          # tokens per message byte; conservative until the first answered call calibrates it
        return (self.sys_bytes + req_bytes + 64) * r * 1.05 + MAX_OUTPUT_TOKENS

    def gate(self, req_bytes: int) -> dict:
        est = self._est(req_bytes)
        if self.spent + est > TRAJ_TOKEN_CAP:
            return {'send': False, 'final': True, 'reason': 'spent %d + next request estimate %d > %d' % (self.spent, est, TRAJ_TOKEN_CAP)}
        if self.spent + est + 1.15 * est > TRAJ_TOKEN_CAP:
            return {'send': True, 'final': True, 'reason': 'token budget: the request after this one would not fit (%d spent)' % self.spent}
        return {'send': True, 'final': False, 'reason': ''}

    def __call__(self, payload: dict) -> str:
        if not DP.proxy_reachable():
            raise rt.llm.TransportFault('PROXY_UNREACHABLE_PRECHECK')
        self.calls += 1
        try:
            text, rec = self.client.call('fast', self.unit, payload, self.system, max_tokens=MAX_OUTPUT_TOKENS, meta={'case': self.case, 'arm': self.arm, 'call': self.calls})
        except budget.BudgetExhausted as exc:          # the ledger's own caps (requests / tokens / wall) are terminal budget stops of the loop
            raise br.BudgetExhausted(str(exc)) from None
        if rec['prompt_tokens'] is not None:
            self.spent += rec['prompt_tokens'] + (rec['completion_tokens'] or 0)
            self.ratio = rec['prompt_tokens'] / max(1, rec['message_bytes'])
        else:
            self.spent += int(rec['message_bytes'] * 0.5) + MAX_OUTPUT_TOKENS          # unknown usage: conservative
        return text


# ============================================================================= Fast system text (zero-feedback version of the previous package's text)
FAST_SYSTEM_ZF = ('''You are the Fast path of a batch training-data augmentation Harness, deployed WITHOUT downstream feedback. Your unit is one entire batch of N entities (see case) and one shared MLP Consumer, not separate per-entity models.
你的任务是为当前整批数据构造并提交一份完整的训练数据增强方案（对每个合法训练窗口生成一个子视图，与不变的父视图共同训练），使固定共享 Consumer 对原始观测未来的预测更好。本轨迹没有任何下游反馈：不训练、不打分、不排名，公共参照也没有分数。只依据合法历史 T 的观察、原语与预设的语义、材料检查结果和已加载的指导来决定是否增强、用哪些原语、是否按观察分组，然后提交一份方案；提交冻结后由外部评估器统一训练和评分，结果不返回本轨迹。允许直接提交任一公共参照，也允许提交自己构造的方案；不强制偏离或保留 NoMix（P_NoMixRecipe），不强制分组、多步或用满预算。子视图只改训练材料：预测输入和评分真值从不增强。单实体预测变化不是该实体材料的独立因果贡献；允许按合法 T 观察构造统一或条件化方案。材料检查确认处理是否生效、作用在哪里、数值是否有效；改动大或平滑本身不证明有害或有益。
(Task: construct and commit one complete whole-batch augmentation plan - a child view of every legal training window trained next to the unchanged parent view - so that the fixed shared Consumer predicts the originally observed future better. There is NO downstream feedback in this trajectory: nothing is trained, scored or ranked, and the public references carry no scores. Decide from legal historical T observations, the semantics of the primitives and the preset, material inspection and any loaded guidance whether to augment, with which primitives and whether to group entities by observation; then commit one plan. After every commit is frozen an external evaluator trains and scores it; nothing comes back to this trajectory. Committing any public reference directly is allowed, and so is committing a plan you constructed; deviating from or keeping NoMix (P_NoMixRecipe), grouping, several steps or using the whole budget are not required. The child view changes training material only: serving inputs and scoring targets are never augmented. A change in one entity's prediction is not the independent causal contribution of that entity's material; uniform or observation-conditional plans may be constructed. Material inspection confirms that a treatment took effect, where, and with valid numbers; a larger change or smoothing does not by itself show harm or benefit.)
Action space: each entity's program is (a) no augmentation (empty steps); (b) an explicit 1-3 step composition of the seven primitives (fixed execution order, regime/shock and calendar/amplitude mutually exclusive, no repeats); (c) the public preset P_NoMixRecipe alone; (d) one P_NoMixRecipe_Edit step disabling 0-2 preset components. A default plus up to 8 T-observation predicate rules (first match wins) may give different entities different programs, or all the same. Plans with identical complete assignments are the same material. Budget per trajectory: at most 6 requests (corrections included), 18 tool calls, 4 constructed materials, one commit; `remaining` states what is left and the last request offers only commit.
Return exactly one JSON object {"actions":[{"tool":"listed_name","arguments":{...}}]}, no markdown. Batched actions execute in order and must use valid IDs; commit must be last. State brief hypotheses in the material rationale or the commit reason, not hidden chain-of-thought. Do not include data source or job identifiers in rationale. Do not invent downstream results. Do not change seeds, model, training budget, scaler, legal windows, targets, scoring or population.''')

CONTRACTS_ZF = {
    'overview': {'arguments': {'entity_indices': 'optional [integers in 0..N-1]; default all N'},
                 'meaning': 'Per-entity T-only values of every field in case.fields (formulas in case.field_definitions), 4 significant digits. No downstream number.'},
    'inspect_data': ES.CONTRACTS['inspect_data'],
    'build_material': {**ES.CONTRACTS['build_material'],
                       'meaning': ES.CONTRACTS['build_material']['meaning'] + ' Predicate features: any name in case.fields (including zero_fraction, flat_nonzero_fraction, '
                                  'weekly_excess_r2); the compiler evaluates them on this batch. At most 4 constructed materials per trajectory; a duplicate is rejected '
                                  'without using a material slot. Nothing is trained.'},
    'inspect_material': ES.CONTRACTS['inspect_material'],
    'commit': {'arguments': {'plan_id': '<a plan id listed in `plans`: constructed or public reference>', 'reason': '<brief evidence-based decision>'},
               'meaning': 'Final action: freeze this plan for the external evaluator (trained and scored only after every commit is frozen; no result returns). '
                          'No automatic replacement, no later override.'},
}


# ============================================================================= one branch (run / resume)
def event_sink(path: Path):
    lock = threading.Lock()

    def emit(row):
        with lock:
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    return emit


def _trace(bdir: Path) -> list:
    p = bdir / 'trace.jsonl'
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def run_branch(bdir: Path, case: str, knowledge: br.Knowledge, led, client, *, common: Path, split_path: Path, cs: ec.CaseSpec, arm: str,
               system: str = FAST_SYSTEM_ZF, limits: zf.Limits = LIMITS, resume: bool = False) -> dict:
    """One zero-feedback trajectory (fresh, or resumed after a never-answered request). Writes trace.jsonl, branch_result.json and, when
    COMPLETE, <case>/commit.json (COMMITTED_UNFITTED)."""
    kn_p = bdir / 'knowledge.json'
    res_state, spent, ratio = None, 0, None
    if not resume:
        bdir.mkdir(parents=True, exist_ok=False)          # before the adapter: the adapter populates the branch view of the case
    adapter = ZFAdapter(bdir, case, led, common=common, split_path=split_path, cs=cs)
    if resume:
        prior = context.read_json(bdir / 'branch_result.json')
        if prior['status'] != 'INCOMPLETE' or prior['failure_kind'] != 'AGENT_CALL_FAILED' or (bdir / case / 'commit.json').exists():
            raise RuntimeError('branch %s is not resumable' % bdir.name)
        if context.read_json(kn_p) != br.json_copy(asdict(knowledge)):
            raise RuntimeError('frozen knowledge differs')
        rows = _trace(bdir)
        pend = max(i for i, r in enumerate(rows) if r['event'] == 'fast_request')
        if any(r['event'] in ('fast_response', 'response_rejected') for r in rows[pend:]):
            raise RuntimeError('last request was answered')
        prefix = rows[:pend]
        answered = sum(r['event'] in ('fast_response', 'response_rejected') for r in prefix)
        built = [r['output']['plan_id'] for r in prefix if r['event'] == 'tool_completed' and r['tool'] == 'build_material']
        feats = {}
        for r in prefix:
            if r['event'] == 'tool_completed' and r['tool'] == 'inspect_data':
                feats.update(r['output'].get('batch_features', {}))
        n_res = 1 + len(list(bdir.glob('resume*.json')))
        shutil.move(bdir / 'trace.jsonl', bdir / ('trace_before_resume%d.jsonl' % n_res))
        shutil.move(bdir / 'branch_result.json', bdir / ('branch_result_before_resume%d.json' % n_res))
        sink = event_sink(bdir / 'trace.jsonl')
        for r in prefix:
            sink(r)
        sink({'event_id': case + ':resume%d' % n_res, 'event': 'job_resumed', 'calls': answered, 'epoch': time.time()})
        prefix = prefix + [{'event_id': case + ':resume%d' % n_res, 'event': 'job_resumed'}]
        res_state = {'trace': prefix, 'calls': answered, 'tools_used': sum(r['event'] == 'tool_started' for r in prefix), 'materials': len(built),
                     'corrections': sum(r['event'] in ('tool_rejected', 'response_rejected') for r in prefix), 'built': [adapter.restore_built(m) for m in built],
                     'features': feats}
        context.write_json(bdir / ('resume%d.json' % n_res), {'epoch': time.time(), 'reason': 'the next Fast request was never answered (transport fault); continued from the identical prefix',
                                                              'prefix_events': len(prefix) - 1, 'calls_before': answered, 'restored_plans': built})
        spent = sum((e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0) for e in led.s['events'] if e.get('kind') == 'llm_finished' and e.get('unit') == bdir.name)
    else:
        context.write_json(kn_p, asdict(knowledge))
        sink = event_sink(bdir / 'trace.jsonl')
    tc = TrajectoryClient(client, bdir.name, system, case=case, arm=arm, spent=spent, ratio=ratio)
    result = zf.run_job_zf(job_id=case, knowledge=knowledge, adapter=adapter, client=tc, public=public_candidates(case), allowed_features=ALLOWED_FEATURES,
                           tool_contracts=CONTRACTS_ZF, entity_count=ec.COHORT_SIZE, limits=zf.Limits(limits.max_calls, limits.max_tools, limits.max_materials,
                                                                                                          max(60.0, led.remaining())),
                           on_event=sink, guard=led.check_wall, evidence_roundtrip=EVIDENCE_ROUNDTRIP, max_corrections=MAX_CORRECTIONS, token_gate=tc.gate,
                           resume=res_state)
    rec = asdict(result)
    rec.pop('trace')
    rec.update({'arm': arm, 'tokens_spent': tc.spent, 'material_build_seconds': round(adapter.build_seconds, 2), 'finished_local': now()})
    context.write_json(bdir / 'branch_result.json', rec)
    if result.status == 'COMPLETE':
        reason = next(r['output']['reason'] for r in reversed(result.trace) if r['event'] == 'tool_completed' and r['tool'] == 'commit')
        cr = adapter.commit_record(result.committed_plan_id, reason)
        cr['constructed_plans'] = [m for m in adapter._reg() if m not in PUBLIC]
        write_new(bdir / case / 'commit.json', cr)
    print('BRANCH', bdir.name, result.status, result.failure_kind or '', 'commit', result.committed_plan_id, 'calls', result.calls, 'tokens', tc.spent, flush=True)
    return rec


# ============================================================================= stage A
def no_card() -> br.Knowledge:
    return br.Knowledge(0, ())


def ledger_for(root: Path) -> SafeLedger:
    return SafeLedger(paths(root)['ledger'], max_fit_attempts=STAGE_A['max_new_fits'], max_llm_requests=STAGE_A['max_logical_requests'] + STAGE_A['extra_transport_attempts'],
                      max_llm_tokens=STAGE_A['max_tokens'], max_wall_s=24 * 3600, max_retries=STAGE_A['fit_retries'])


def freeze_config(root: Path) -> dict:
    P_ = paths(root)
    if P_['frozen'].exists():
        return context.read_json(P_['frozen'])
    split = build_split()
    context.write_json(P_['split'], split)
    sc = stage_cases(split)
    cfg = {'package': PACKAGE, 'task': TASK, 'frozen_local': now(), 'exposure': EXPOSURE, 'domains': list(DOMAINS), 'seeds': list(SEEDS),
           'model': MODEL, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'limits': {'max_calls': LIMITS.max_calls, 'max_tools': LIMITS.max_tools,
           'max_materials': LIMITS.max_materials, 'tokens_per_trajectory': TRAJ_TOKEN_CAP, 'max_corrections': MAX_CORRECTIONS, 'commit': 1},
           'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'extra_fields': list(EXTRA_FIELDS), 'refs': REFS, 'stage_a': STAGE_A, 'concurrency': CONCURRENCY,
           'min_free_gb': MIN_FREE_GB, 'auto_accept_unknown_usage': AUTO_ACCEPT_CLASS,
           'cases': {k: v for k, v in sc.items()}, 'counts': {k: {d: len(v) for d, v in s.items()} for k, s in sc.items()},
           'fast_system': FAST_SYSTEM_ZF, 'tool_contracts': CONTRACTS_ZF,
           'stage_a_order': 'interleaved D01, D02, D03 by screen anchor then group (A1_G1.. A2_G4); at most 4 trajectories / HTTP requests in flight'}
    want = {'source': {'D01': 8, 'D02': 8, 'D03': 8}, 'select': {'D01': 9, 'D02': 9, 'D03': 6}, 'test': {'D01': 16, 'D02': 16, 'D03': 8}}
    if cfg['counts'] != want:
        raise RuntimeError('case counts differ from task §2: %s' % cfg['counts'])
    context.write_json(P_['frozen'], cfg)
    return cfg


def stage_a_order(cfg: dict) -> list:
    src = cfg['cases']['source']
    order = []
    for k in range(8):
        for d in DOMAINS:
            order.append(sorted(src[d])[k])
    return order


def stage_a(root: Path = ROOT, *, numeric: int = CONCURRENCY['numeric'], only=None) -> dict:
    """task §5: 24 no-card zero-feedback trajectories (<= 4 in flight); no fit, no score inside. Resumes never-answered trajectories once
    each within the extra-transport budget. Stops paid dispatch when done."""
    P_ = paths(root)
    cfg = freeze_config(root)
    split = context.read_json(P_['split'])
    css = cases_of(split)
    DP.set_pools(numeric, CONCURRENCY['http'])
    with PackageLock(root, 'stage_a'):
        led = ledger_for(root)
        if 'paid_clock_start' not in led.s:
            with led.lock:
                led.s['paid_clock_start'] = time.time()
                led.s['paid_clock_start_local'] = now()
                led._save()
        client = ZFMeteredClient(led, P_['llm'], http_cap=STAGE_A['max_logical_requests'] + STAGE_A['extra_transport_attempts'], stage='stage_a',
                                 incidents=P_['incidents'], extra_cap=STAGE_A['extra_transport_attempts'])
        order = [c for c in stage_a_order(cfg) if (only is None or c in only)]
        for case in order:
            prepare_common_source(P_['common'], css[case])
        results, errors = {}, []
        q = list(order)
        qlock = threading.Lock()

        def worker():
            while True:
                with qlock:
                    if not q:
                        return
                    case = q.pop(0)
                bdir = P_['stage_a'] / 'branches' / ('%s__f0' % case)
                try:
                    if bdir.exists() and not (bdir / 'branch_result.json').exists():
                        # the controller died inside this trajectory: keep its files, restart it whole (its paid calls stay in the ledger; reported)
                        k = 1 + len(list(bdir.parent.glob(bdir.name + '__killed*')))
                        shutil.move(bdir, bdir.parent / ('%s__killed%d' % (bdir.name, k)))
                        led.event(kind='trajectory_restarted_after_kill', case=case, moved_to='%s__killed%d' % (bdir.name, k))
                    if (bdir / 'branch_result.json').exists():
                        r = context.read_json(bdir / 'branch_result.json')
                        if r['status'] == 'COMPLETE' or r['failure_kind'] != 'AGENT_CALL_FAILED':
                            results[case] = r
                            continue
                        results[case] = run_branch(bdir, case, no_card(), led, client, common=P_['common'], split_path=P_['split'], cs=css[case], arm='f0', resume=True)
                    else:
                        results[case] = run_branch(bdir, case, no_card(), led, client, common=P_['common'], split_path=P_['split'], cs=css[case], arm='f0')
                except Exception as exc:  # noqa: BLE001
                    errors.append((case, repr(exc)[:300]))
                    print('BRANCH_ERROR', case, repr(exc)[:300], flush=True)

        threads = [threading.Thread(target=worker, name='fast-%d' % i, daemon=True) for i in range(CONCURRENCY['http'])]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # one resume pass for never-answered requests (transport), within the extra-transport budget
        again = [c for c, r in results.items() if r.get('status') == 'INCOMPLETE' and r.get('failure_kind') == 'AGENT_CALL_FAILED']
        for case in again:
            if led.s.get('extra_transport_attempts', 0) >= STAGE_A['extra_transport_attempts'] or rt.unknown_usage_blocks(led):
                break
            bdir = P_['stage_a'] / 'branches' / ('%s__f0' % case)
            led.s['extra_transport_attempts'] = led.s.get('extra_transport_attempts', 0) + 1       # the resend is an extra transport attempt
            led._save()
            try:
                results[case] = run_branch(bdir, case, no_card(), led, client, common=P_['common'], split_path=P_['split'], cs=css[case], arm='f0', resume=True)
            except Exception as exc:  # noqa: BLE001
                errors.append((case, repr(exc)[:300]))
        with led.lock:
            led.s['paid_clock_stop'] = time.time()
            led.s['paid_clock_stop_local'] = now()
            led.s['stage_a_dispatch_stopped'] = True
            led._save()
        summary = {'finished_local': now(), 'n': len(results), 'complete': sum(r.get('status') == 'COMPLETE' for r in results.values()),
                   'incomplete': {c: [r.get('failure_kind'), r.get('reason')] for c, r in results.items() if r.get('status') != 'COMPLETE'},
                   'errors': errors, 'pools': DP.pools().snapshot(), 'ledger': led.summary(), 'unknown_usage_blocking': rt.unknown_usage_blocks(led)}
        context.write_json(P_['stage_a'] / 'dispatch_summary.json', summary)
        print('STAGE_A_DISPATCH_DONE', summary['complete'], '/', summary['n'], 'errors', len(errors), flush=True)
        return summary


# ============================================================================= post-freeze evaluator (task §5: after all 24 commits are frozen)
def screen_e(case: str) -> dict:
    """{material_id: {seed: e}} from the screen's E scores, with the cell configuration checks for cache reuse."""
    es = context.read_json(screen_dir(case) / 'e_scores.json')
    out = {}
    for cid, r in es['cells'].items():
        out.setdefault(r['material_id'], {})[int(r['model_seed'])] = float(r['e'])
    return out


def screen_cell_ok(case: str, phys: str, key: str, cs: ec.CaseSpec) -> bool:
    for s in SEEDS:
        p = screen_dir(case) / 'cells' / (ec.cell_id(case, phys, s) + '.json')
        if not p.exists():
            return False
        r = context.read_json(p)
        if not (r.get('status') == 'OK' and r.get('material_key') == key and r.get('n_updates') == spec.N_UPDATES and r.get('profile') == ec.PROFILE_VERSION
                and r.get('roster') == list(cs.roster) and r.get('entity_count') == ec.COHORT_SIZE and r.get('model_seed') == s and r.get('torch_threads') == FIT_THREADS):
            return False
    return True


def worker_material(root: str, split_path: str, case: str, phys: str) -> None:
    ES._init_worker()
    import torch
    torch.set_num_threads(1)          # the screen built its materials with one torch thread (measured faster); identical setting here
    ES.worker_material(root, split_path, case, phys)


def worker_score(root: str, split_path: str, case: str, out: str, physes: list) -> None:
    """Evaluator only: E of the given physical materials' OK cells (all seeds) with the case's frozen T scaler; rows [t, t+384) are read here."""
    ES._init_worker()
    from methods.ttha.batch_base import train
    cs = ES._case(split_path, case)
    jd = Path(root) / case
    sc = np.load(jd / 'scaler.npz')
    if [str(x) for x in sc['roster']] != list(cs.roster) or int(sc['t']) != cs.t or str(sc['dataset']) != cs.dataset:
        raise RuntimeError('scaler binding drift for %s' % case)
    scaler = data.Scaler(mean=sc['mean'], std=sc['std'], scale=sc['scale'], floor_hits=0)
    sl = ec.load_case_slice(cs, cs.t, max(cs.e) + ec.H)
    wins = data.build_windows(sl, list(cs.e), scaler, with_truth=True)
    truth = np.stack([w.y_true_raw for w in wins], axis=1)
    cells = ec.branch_cells(jd)
    res = {'case_id': case, 'rows_read': [cs.t, max(cs.e) + ec.H], 'e_origins': list(cs.e), 'scored_local': now(), 'cells': {}}
    for phys in physes:
        for s in SEEDS:
            cid = ec.cell_id(case, phys, s)
            r = cells[cid]
            pred = train.predict_windows(train.load_model(r['model_path']), wins, scaler)
            sco = ec.score_with_status(pred, truth, scaler, sc['mase'])
            if sco['status'] != 'SCORABLE':
                raise RuntimeError('E not scorable for %s' % cid)
            res['cells'][cid] = {'material_id': phys, 'model_seed': s, 'c_a': r['scores']['c_a']['normalized_mse_macro'], 'e': sco['normalized_mse_macro'],
                                 'e_per_entity': sco['per_entity_normalized_mse'], 'e_per_origin': sco['per_origin_normalized_mse_mean']}
    context.write_json(Path(out), res)
    print('SCORE_OK', case, len(res['cells']), flush=True)


def fit_new(led, common: Path, split_path: Path, case: str, phys: str) -> dict:
    """Three seeds of one new physical material in common/<case> (reserved-before-launch, the entity-split fit worker, <= 1 retry each)."""
    rt.FIT_RETRY = True
    out = {}
    for s in SEEDS:
        wait_memory('fit')
        out.update(DP.fit_cells(led, common, split_path, case, [(phys, s)]))
    return out


def evaluate_stage_a(root: Path = ROOT, *, numeric: int = CONCURRENCY['numeric']) -> dict:
    P_ = paths(root)
    cfg = context.read_json(P_['frozen'])
    split = context.read_json(P_['split'])
    css = cases_of(split)
    order = stage_a_order(cfg)
    bdirs = {c: P_['stage_a'] / 'branches' / ('%s__f0' % c) for c in order}
    unfinished = [c for c, b in bdirs.items() if not (b / 'branch_result.json').exists()]
    if unfinished:
        raise RuntimeError('stage A not frozen: %s' % unfinished)
    if not context.read_json(P_['ledger']).get('stage_a_dispatch_stopped'):
        raise RuntimeError('stage A dispatch has not stopped; no evaluation before every commit is frozen')
    DP.set_pools(numeric, CONCURRENCY['http'])
    with PackageLock(root, 'evaluate_a'):
        led = ledger_for(root)
        freeze = {'frozen_local': now(), 'commits': {}}
        for c, b in bdirs.items():
            cp = b / c / 'commit.json'
            freeze['commits'][c] = context.read_json(cp) if cp.exists() else None
        if not (P_['eval'] / 'freeze.json').exists():
            write_new(P_['eval'] / 'freeze.json', freeze)
        out = {}
        lock = threading.Lock()
        jobs = []
        for c in order:
            cm = freeze['commits'][c]
            if cm is None:
                out[c] = {'status': 'INCOMPLETE', 'delivery': None}
                continue
            phys, key = cm['physical_material'], cm['material_key']
            sreg = ec.load_registry(screen_dir(c))
            if phys in sreg and sreg[phys]['key'] == key and screen_cell_ok(c, phys, key, css[c]):
                se = screen_e(c)
                out[c] = {'status': 'COMPLETE', 'phys': phys, 'cache_hit': True, 'e_by_seed': {str(s): se[phys][s] for s in SEEDS}}
                for s in SEEDS:
                    led.note_cache(ec.cell_id(c, phys, s))
            else:
                jobs.append((c, phys))

        def one(job):
            c, phys = job
            fp = P_['eval'] / ('%s__%s__e.json' % (c, phys))
            if not fp.exists():
                fit_new(led, P_['common'], P_['split'], c, phys)
                rc, secs = run_numeric(['--worker-score', P_['common'], P_['split'], c, fp, phys], P_['logs'] / ('score_%s_%s.log' % (c, phys)), 900)
                if rc != 0:
                    raise RuntimeError('score worker failed for %s %s' % (c, phys))
            r = context.read_json(fp)
            with lock:
                out[c] = {'status': 'COMPLETE', 'phys': phys, 'cache_hit': False,
                          'e_by_seed': {str(s): r['cells'][ec.cell_id(c, phys, s)]['e'] for s in SEEDS},
                          'c_a_by_seed_diagnostic': {str(s): r['cells'][ec.cell_id(c, phys, s)]['c_a'] for s in SEEDS}}

        errs = []

        def run(job):
            try:
                one(job)
            except Exception as exc:  # noqa: BLE001
                errs.append((job, repr(exc)[:300]))

        threads = [threading.Thread(target=run, args=(j,), daemon=True) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        res = {'evaluated_local': now(), 'cases': out, 'new_fit_jobs': [list(j) for j in jobs], 'errors': errs, 'ledger': led.summary()}
        context.write_json(P_['eval'] / 'evaluation.json', res)
        print('STAGE_A_EVAL', len(out), 'new', len(jobs), 'errors', len(errs), flush=True)
        return res


# ============================================================================= stage-A readout and checkpoint (task §5)
def G(a, b, den):
    return 100.0 * (b - a) / den


def _st(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    s = sorted(v)
    return {'n': len(v), 'mean': statistics.fmean(v), 'median': statistics.median(v), 'p90': s[min(len(s) - 1, int(math.ceil(0.9 * len(s))) - 1)], 'max': max(v), 'min': min(v)}


def readout_a(root: Path = ROOT) -> dict:
    P_ = paths(root)
    cfg = context.read_json(P_['frozen'])
    ev = context.read_json(P_['eval'] / 'evaluation.json')
    led = context.read_json(P_['ledger'])
    order = stage_a_order(cfg)
    per_case, dom_rows = {}, {d: [] for d in DOMAINS}
    for c in order:
        dom = c[:3]
        b = P_['stage_a'] / 'branches' / ('%s__f0' % c)
        br_ = context.read_json(b / 'branch_result.json')
        rows = _trace(b)
        se = screen_e(c)
        E = {m: statistics.fmean(se[m][s] for s in SEEDS) for m in se}
        den = E['None']
        ev_c = ev['cases'][c]
        cm = context.read_json(b / c / 'commit.json') if (b / c / 'commit.json').exists() else None
        tok = [(e.get('prompt_tokens') or 0, e.get('completion_tokens') or 0) for e in led['events'] if e.get('kind') == 'llm_finished' and e.get('unit') == b.name]
        builds = [r for r in rows if r['event'] == 'tool_completed' and r['tool'] == 'build_material']
        rec = {'domain': dom, 'status': br_['status'], 'failure_kind': br_.get('failure_kind'), 'reason': br_.get('reason'), 'calls': br_['calls'],
               'tool_calls': br_['tool_calls'], 'materials_built': br_['new_evaluations'], 'tokens': sum(a + b_ for a, b_ in tok), 'prompt_tokens': sum(a for a, _ in tok),
               'completion_tokens': sum(b_ for _, b_ in tok), 'requests': len(tok), 'material_build_seconds': br_.get('material_build_seconds'),
               'tools_used': [r['tool'] for r in rows if r['event'] == 'tool_started'], 'corrections': sum(r['event'] in ('tool_rejected', 'response_rejected') for r in rows),
               'built_labels': [r['output'].get('label') for r in builds], 'built_conditional': sum(1 for r in builds if len(r['output'].get('programs') or []) > 1),
               'commit': cm and {'plan_id': cm['plan_id'], 'phys': cm['physical_material'], 'label': cm['label'], 'reason': cm['reason'],
                                 'public': cm['plan_id'] in PUBLIC, 'constructed': cm['plan_id'] not in PUBLIC}}
        refs = {'None': 'None', 'NoMix': 'P_NoMixRecipe', 'Fixed_source': REFS['Fixed_source'][dom], 'Fixed_global_source': REFS['Fixed_global_source']}
        rec['refs_G_vs_None'] = {k: G(E[m], den, den) for k, m in refs.items()}
        rec['menu_G_vs_None'] = {m: G(E[m], den, den) for m in SC.MENU_IDS}
        best = max(SC.MENU_IDS, key=lambda m: rec['menu_G_vs_None'][m])
        rec['menu_best_post_hoc'] = {'program': best, 'label': SC.LABEL[best], 'G_vs_None': rec['menu_G_vs_None'][best]}
        if ev_c['status'] == 'COMPLETE':
            ef = statistics.fmean(ev_c['e_by_seed'][str(s)] for s in SEEDS)
            rec['f0'] = {'E': ef, 'cache_hit': ev_c['cache_hit'], 'phys': ev_c['phys'], 'G_vs_None': G(ef, den, den),
                         'G_F0_minus': {k: G(ef, E[m], den) for k, m in refs.items()}, 'G_F0_minus_menu_best': G(ef, E[best], den),
                         'seed_G_vs_None': [G(ev_c['e_by_seed'][str(s)], se['None'][s], den) for s in SEEDS],
                         'menu_equivalent': next((m for m in SC.MENU_IDS if m == ev_c['phys']), None)}
        per_case[c] = rec
        dom_rows[dom].append(rec)
    doms = {}
    for d, rs in dom_rows.items():
        ok = [r for r in rs if 'f0' in r]
        doms[d] = {'n': len(rs), 'complete': len(ok),
                   'F0_minus': {k: statistics.fmean(r['f0']['G_F0_minus'][k] for r in ok) for k in ('None', 'NoMix', 'Fixed_source', 'Fixed_global_source')} if ok else None,
                   'F0_minus_menu_best_post_hoc': statistics.fmean(r['f0']['G_F0_minus_menu_best'] for r in ok) if ok else None,
                   'wins_ties_losses_vs_NoMix': [sum(r['f0']['G_F0_minus']['NoMix'] > 1e-9 for r in ok), sum(abs(r['f0']['G_F0_minus']['NoMix']) <= 1e-9 for r in ok),
                                                 sum(r['f0']['G_F0_minus']['NoMix'] < -1e-9 for r in ok)] if ok else None,
                   'wins_ties_losses_vs_Fixed_source': [sum(r['f0']['G_F0_minus']['Fixed_source'] > 1e-9 for r in ok), sum(abs(r['f0']['G_F0_minus']['Fixed_source']) <= 1e-9 for r in ok),
                                                        sum(r['f0']['G_F0_minus']['Fixed_source'] < -1e-9 for r in ok)] if ok else None,
                   'worst_vs_None': min((r['f0']['G_vs_None'] for r in ok), default=None),
                   'commits': [r['commit']['label'] if r['commit'] else None for r in rs]}
    eq = {k: statistics.fmean(doms[d]['F0_minus'][k] for d in DOMAINS if doms[d]['F0_minus']) for k in ('None', 'NoMix', 'Fixed_source', 'Fixed_global_source')}
    cost = {k: _st([per_case[c][k] for c in order]) for k in ('tokens', 'prompt_tokens', 'completion_tokens', 'requests', 'tool_calls', 'material_build_seconds', 'materials_built')}
    res = {'package': PACKAGE, 'stage': 'A', 'written_local': now(), 'per_case': per_case, 'per_domain': doms, 'three_domain_equal_weight_F0_minus': eq,
           'three_domain_equal_weight_F0_minus_menu_best_post_hoc': statistics.fmean(doms[d]['F0_minus_menu_best_post_hoc'] for d in DOMAINS if doms[d]['F0_minus_menu_best_post_hoc'] is not None),
           'cost_per_trajectory': cost, 'ledger': {k: v for k, v in led.items() if k != 'events'},
           'fits': {'new_fit_jobs': ev['new_fit_jobs'], 'cache_hit_cases': sum(1 for c in order if ev['cases'][c].get('cache_hit')),
                    'new_material_cases': sum(1 for c in order if ev['cases'][c].get('status') == 'COMPLETE' and not ev['cases'][c].get('cache_hit'))}}
    res['projection'] = project_remaining(res)
    context.write_json(P_['stage_a'] / 'stage_a_result.json', res)
    return res


def project_remaining(res: dict) -> dict:
    """Stages B-D cost from the measured trajectory cost (task §5 / §10). Card text adds <= 6000 characters (~2.5k tokens) to every request of a
    card arm; Slow calls are estimated from the census size bound (<= 530 KB message)."""
    c = res['cost_per_trajectory']
    if not c['tokens']:
        return {}
    mean_t, p90_t, max_t = c['tokens']['mean'], c['tokens']['p90'], c['tokens']['max']
    req_mean = c['requests']['mean']
    card_extra = 2_500 * req_mean
    traj = {'select': 96, 'test': 120}
    card_traj = {'select': 96, 'test': 80}
    point = traj['select'] * mean_t + traj['test'] * mean_t + (card_traj['select'] + card_traj['test']) * card_extra
    cons = (traj['select'] + traj['test']) * min(TRAJ_TOKEN_CAP, max(p90_t, mean_t * 1.5)) + (card_traj['select'] + card_traj['test']) * card_extra
    slow = {'calls': 8, 'tokens_point': 8 * 150_000, 'tokens_conservative': 8 * 2 * 230_000}
    return {'basis': {'measured_mean_tokens_per_trajectory': mean_t, 'p90': p90_t, 'max': max_t, 'requests_mean': req_mean, 'card_extra_tokens_per_card_trajectory': card_extra},
            'trajectories': {'select': 96, 'test': 120, 'total_remaining': 216},
            'fast_tokens_point': point, 'fast_tokens_conservative': cons, 'slow': slow,
            'total_point': point + slow['tokens_point'], 'total_conservative': cons + slow['tokens_conservative'],
            'proposed_hard_caps': {'tokens': int(math.ceil((cons + slow['tokens_conservative']) * 1.15 / 1e6)) * 1_000_000,
                                   'logical_requests': 216 * LIMITS.max_calls + 8 * 2, 'extra_transport_attempts': 16,
                                   'fits': 1300 - (res['ledger']['fit_attempts'] + context.read_json(paths(ROOT)['wiring'] / 'wiring_result.json')['fits']),
                                   'note': 'fits: package cap 1300 minus stage A (final-commit fits + wiring fits); select 288 + None 72, test 360 + dedup refs 456'},
            'fits_worst_case_remaining': {'select_commits': 288, 'select_none': 72, 'test_commits': 360, 'test_refs_dedup': 456}}


def report_a(root: Path = ROOT) -> str:
    res = readout_a(root)
    P_ = paths(root)
    L = ['# %s — Stage A checkpoint (auto tables, %s)' % (PACKAGE, now()), '',
         'Units: pp of the case None mean, G = 100 x (E(B) - E(A)) / E(None), three-seed means; positive = F0 better. Domain = equal weight of its 8 cases.', '']
    L.append('| domain | complete | F0 − None | F0 − NoMix | F0 − Fixed_source | F0 − Fixed_global | F0 − menu best (post hoc) | W/T/L vs NoMix | W/T/L vs Fixed_source | worst vs None |')
    L.append('|---|---|---:|---:|---:|---:|---:|---|---|---:|')
    for d in DOMAINS:
        x = res['per_domain'][d]
        f = x['F0_minus'] or {}
        L.append('| %s | %d/%d | %s | %s | %s | %s | %s | %s | %s | %s |' % (d, x['complete'], x['n'], _f(f.get('None')), _f(f.get('NoMix')), _f(f.get('Fixed_source')),
                 _f(f.get('Fixed_global_source')), _f(x['F0_minus_menu_best_post_hoc']), x['wins_ties_losses_vs_NoMix'], x['wins_ties_losses_vs_Fixed_source'], _f(x['worst_vs_None'])))
    L.append('')
    L.append('| case | status | commit | constructed | cache | F0 vs None | NoMix vs None | Fixed_source vs None | menu best (post hoc) | calls | tools | built | tokens |')
    L.append('|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|')
    for c, r in res['per_case'].items():
        f0 = r.get('f0') or {}
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s %s | %d | %d | %d | %d |' % (c, r['status'] + ('' if not r['failure_kind'] else ' ' + r['failure_kind']),
                 (r['commit'] or {}).get('label'), (r['commit'] or {}).get('constructed'), f0.get('cache_hit'), _f(f0.get('G_vs_None')), _f(r['refs_G_vs_None']['NoMix']),
                 _f(r['refs_G_vs_None']['Fixed_source']), r['menu_best_post_hoc']['label'], _f(r['menu_best_post_hoc']['G_vs_None']), r['calls'], r['tool_calls'],
                 r['materials_built'], r['tokens']))
    L.append('')
    L.append('Cost per trajectory: ' + json.dumps({k: {kk: round(vv, 1) for kk, vv in v.items()} if v else None for k, v in res['cost_per_trajectory'].items()}))
    (P_['stage_a'] / 'tables_a.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    return '\n'.join(L)


def _f(x, nd=2):
    return '—' if x is None else ('%+.*f' % (nd, x))


# ============================================================================= smoke (scripted client; no LLM, no fit)
class ScriptedText:
    """Returns scripted response texts (a dict is JSON-encoded, a str is sent as is); records every request; can raise a transport fault once."""

    def __init__(self, steps: list, *, fail_at: int | None = None):
        self.steps, self.fail_at, self.requests, self.n = list(steps), fail_at, [], 0

    def __call__(self, payload):
        self.n += 1
        self.requests.append(copy.deepcopy(payload))
        if self.fail_at is not None and self.n == self.fail_at:
            self.fail_at = None
            raise rt.llm.TransportFault('SCRIPTED')
        s = self.steps.pop(0)
        return s if isinstance(s, str) else json.dumps(s)


def smoke(root: Path = ROOT) -> dict:
    """Zero-feedback isolation, unfitted commit, extended-vocabulary compilation, budgets / final request, corrections, resume (scripted; the
    material worker runs for one new material; no LLM client object is created, no fit, no score)."""
    sm = paths(root)['smoke']
    if sm.exists():
        shutil.rmtree(sm)
    sm.mkdir(parents=True)
    split = build_split()
    context.write_json(sm / 'split_copy.json', split)
    css = cases_of(split)
    case = 'D03_SCR_A1_G1'
    common = sm / 'common'
    prepare_common_source(common, css[case])
    led = SafeLedger(sm / 'budget.json', max_fit_attempts=0, max_llm_requests=0, max_llm_tokens=0, max_wall_s=3600, max_retries=0)
    DP.set_pools(2, 1)
    checks = []

    def check(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})
        print('SMOKE', 'OK ' if ok else 'FAIL', name, info if not ok else '', flush=True)

    def run(tag, steps, *, knowledge=no_card(), limits=LIMITS, gate=None, fail_at=None, resume=False, client=None):
        bdir = sm / 'branches' / tag
        cl = client or ScriptedText(steps, fail_at=fail_at)
        adapter = ZFAdapter(bdir if resume else (bdir.mkdir(parents=True, exist_ok=True) or bdir), case, led, common=common, split_path=sm / 'split_copy.json', cs=css[case])
        sink = event_sink(bdir / 'trace.jsonl')
        res_state = None
        if resume:
            rows = _trace(bdir)
            pend = max(i for i, r in enumerate(rows) if r['event'] == 'fast_request')
            prefix = rows[:pend]
            built = [r['output']['plan_id'] for r in prefix if r['event'] == 'tool_completed' and r['tool'] == 'build_material']
            res_state = {'trace': prefix, 'calls': sum(r['event'] in ('fast_response', 'response_rejected') for r in prefix),
                         'tools_used': sum(r['event'] == 'tool_started' for r in prefix), 'materials': len(built),
                         'corrections': sum(r['event'] in ('tool_rejected', 'response_rejected') for r in prefix), 'built': [adapter.restore_built(m) for m in built]}
        r = zf.run_job_zf(job_id=case, knowledge=knowledge, adapter=adapter, client=cl, public=public_candidates(case), allowed_features=ALLOWED_FEATURES,
                          tool_contracts=CONTRACTS_ZF, entity_count=ec.COHORT_SIZE, limits=limits, on_event=sink, guard=None, evidence_roundtrip=True,
                          max_corrections=MAX_CORRECTIONS, token_gate=gate, resume=res_state)
        if r.status == 'COMPLETE':
            reason = next(x['output']['reason'] for x in reversed(r.trace) if x['event'] == 'tool_completed' and x['tool'] == 'commit')
            write_new(bdir / case / 'commit.json', adapter.commit_record(r.committed_plan_id, reason))
        return r, cl, adapter

    cond = {'default': {'steps': [{'op': ta.RECIPE}]}, 'rules': [{'when': {'feature': 'zero_fraction', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': [{'op': 'tp_censor'}]}],
            'rationale': 'smoke: zero-heavy entities get censor', 'observation_fields_used': ['zero_fraction']}
    # S1: happy path through the final commit-only request
    r1, c1, a1 = run('s1_happy', [
        {'actions': [{'tool': 'inspect_data', 'arguments': {'entity_indices': [0, 1], 'kind': 'hour_profile'}}]},
        {'actions': [{'tool': 'overview', 'arguments': {}}]},
        {'actions': [{'tool': 'build_material', 'arguments': {'plan_id': 'cens', 'policy': {'default': {'steps': [{'op': 'tp_censor'}]}, 'rules': [], 'rationale': 'smoke', 'observation_fields_used': []}}},
                     {'tool': 'build_material', 'arguments': {'plan_id': 'cond', 'policy': cond}},
                     {'tool': 'inspect_material', 'arguments': {'plan_id': 'cond'}},
                     {'tool': 'build_material', 'arguments': {'plan_id': 'deferred', 'policy': {'default': {'steps': [{'op': 'tp_shock'}]}, 'rules': []}}}]},
        {'actions': [{'tool': 'inspect_material', 'arguments': {'plan_id': 'cens'}}]},
        {'actions': [{'tool': 'inspect_data', 'arguments': {'entity_indices': [2], 'kind': 'daily_means'}}]},
        {'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'cond', 'reason': 'smoke final commit'}}]}])
    reqs = c1.requests
    context.write_json(sm / 's1_requests.json', reqs)
    check('s1_complete_unfitted_commit', r1.status == 'COMPLETE' and r1.committed_plan_id == 'cond' and r1.delivery_model_ref is None, status=r1.status, reason=r1.reason)
    cm = context.read_json(sm / 'branches' / 's1_happy' / case / 'commit.json') if (sm / 'branches' / 's1_happy' / case / 'commit.json').exists() else {}
    check('s1_commit_record_unfitted', cm.get('status') == 'COMMITTED_UNFITTED' and cm.get('fitted') is False and cm.get('physical_material', '').startswith('TA'), cm=cm)
    check('s1_no_cells_in_branch', not (sm / 'branches' / 's1_happy' / case / 'cells').exists())
    check('s1_no_evaluate_tool', all('evaluate' not in q['tools'] and 'compare' not in q['tools'] and 'evaluate' not in q['tool_contracts'] for q in reqs))
    txt = json.dumps(reqs)
    check('s1_no_score_keys_in_requests', not re.search(r'"(c_a|c_b|e|feedback|normalized_mse_macro|ca_losses|loss_by_seed)"\s*:', txt))
    check('s1_final_request_commit_only', reqs[-1]['tools'] == ['commit'] and reqs[-1]['remaining']['final_request'] is True and all(q['tools'] != ['commit'] for q in reqs[:-1]))
    check('s1_card_compact_no_entity_rows', all('entities' not in q['case'] and 'entities' not in q['batch_summary'] for q in reqs), first_request_bytes=len(json.dumps(reqs[0])))
    check('s1_extra_fields_in_summary_and_table', all(('batch_median_' + f) in reqs[0]['batch_summary']['batch_features'] for f in EXTRA_FIELDS)
          and all(f in json.dumps(reqs[2]['history']) for f in EXTRA_FIELDS))
    comp = context.read_json(sm / 'branches' / 's1_happy' / case / 'aug_materials' / 'cond__compiled.json')
    zfv = [row['zero_fraction'] for row in a1.table]
    thr = float(np.quantile(zfv, 0.5))
    want = [[{'op': 'tp_censor'}] if z >= thr else [{'op': ta.RECIPE}] for z in zfv]
    check('s1_zero_fraction_rule_compiled_for_real', comp['assignment'] == want, threshold=thr, n_censor=sum(z >= thr for z in zfv))
    reg1 = ec.load_registry(sm / 'branches' / 's1_happy' / case)
    check('s1_menu_key_maps_to_screen_material', reg1['cens']['phys_id'] == 'C_censor', phys=reg1['cens'].get('phys_id'))
    rows1 = _trace(sm / 'branches' / 's1_happy')
    check('s1_evidence_roundtrip_deferral', any(r['event'] == 'action_batch_deferred' for r in rows1))
    hist = reqs[-1]['history']
    n_resp = sum(1 for h in hist if 'response' in h)
    check('s1_history_each_response_once', n_resp == 5 and not any('guidance' in h for h in hist), n_resp=n_resp)
    check('s1_screen_dir_unchanged', sorted(ec.load_registry(screen_dir(case))) == sorted(SC.MENU_IDS))
    # S2: corrections (evaluate not available, malformed text, unknown feature) -> INCOMPLETE after the budget
    r2, c2, _ = run('s2_errors', [
        {'actions': [{'tool': 'evaluate', 'arguments': {'plan_id': 'P_NoMixRecipe'}}]},
        'this is not json',
        {'actions': [{'tool': 'build_material', 'arguments': {'plan_id': 'bad', 'policy': {'default': {'steps': []}, 'rules': [{'when': {'feature': 'future_mean', 'op': '>', 'value': 0}, 'steps': []}]}}}]}])
    rows2 = _trace(sm / 'branches' / 's2_errors')
    check('s2_errors_are_corrections_then_incomplete', r2.status == 'INCOMPLETE' and r2.failure_kind == 'BUDGET_EXHAUSTED'
          and sum(r['event'] == 'response_rejected' for r in rows2) == 2 and sum(r['event'] == 'tool_rejected' for r in rows2) == 1, status=r2.status, kind=r2.failure_kind)
    check('s2_rejection_visible_next_request', 'response_rejected' in json.dumps(c2.requests[1]['history']) and 'not available' in json.dumps(c2.requests[1]['history']))
    # S3: material budget
    r3, c3, _ = run('s3_materials', [
        {'actions': [{'tool': 'build_material', 'arguments': {'plan_id': 'm1', 'policy': {'default': {'steps': [{'op': 'tp_censor'}]}, 'rules': []}}},
                     {'tool': 'build_material', 'arguments': {'plan_id': 'm2', 'policy': {'default': {'steps': [{'op': 'tp_shock'}]}, 'rules': []}}}]},
        {'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'm1', 'reason': 'smoke'}}]}], limits=zf.Limits(6, 18, 1))
    rows3 = _trace(sm / 'branches' / 's3_materials')
    check('s3_material_cap', r3.status == 'COMPLETE' and any(r['event'] == 'tool_rejected' and 'material budget' in r['error']['message'] for r in rows3))
    # S4: token gate makes the second request final
    calls = {'n': 0}

    def gate(nbytes):
        calls['n'] += 1
        return {'send': True, 'final': calls['n'] >= 2, 'reason': 'smoke token gate'}
    r4, c4, _ = run('s4_gate', [{'actions': [{'tool': 'overview', 'arguments': {'entity_indices': [0]}}]},
                                {'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'P_NoMixRecipe', 'reason': 'smoke'}}]}], gate=gate)
    check('s4_token_gate_final', r4.status == 'COMPLETE' and c4.requests[1]['tools'] == ['commit'] and 'smoke token gate' in c4.requests[1]['remaining']['final_request_note'])
    tcl = TrajectoryClient(None, 'u', FAST_SYSTEM_ZF, case=case, arm='f0', spent=165_000, ratio=0.3)
    g1, g2 = tcl.gate(30_000), tcl.gate(200_000)
    check('s4_gate_arithmetic', g1['send'] and g1['final'] and not g2['send'], g1=g1, g2=g2)
    # S5: resume after a never-answered request: the resent request equals the original
    steps5 = [{'actions': [{'tool': 'overview', 'arguments': {}}]},
              {'actions': [{'tool': 'build_material', 'arguments': {'plan_id': 'cc', 'policy': {'default': {'steps': [{'op': 'tp_censor'}]}, 'rules': []}}}]},
              {'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'cc', 'reason': 'smoke'}}]}]
    cl5 = ScriptedText(copy.deepcopy(steps5), fail_at=3)
    r5a, _, _ = run('s5_resume', None, client=cl5)
    sent_before = cl5.requests[-1]
    cl5b = ScriptedText([steps5[2]])
    r5b, _, _ = run('s5_resume', None, client=cl5b, resume=True)
    check('s5_resume_identical_request', r5a.failure_kind == 'AGENT_CALL_FAILED' and r5b.status == 'COMPLETE'
          and {k: v for k, v in cl5b.requests[0].items() if k != 'remaining'} == {k: v for k, v in sent_before.items() if k != 'remaining'},
          a=r5a.failure_kind, b=r5b.status)
    # S6: adapter refuses fits
    try:
        a1.evaluate(None, SEEDS, feedback=False, remaining_seconds=10)
        ok6 = False
    except PermissionError:
        ok6 = True
    check('s6_adapter_has_no_evaluate', ok6)
    out = {'written_local': now(), 'checks': checks, 'passed': sum(c['ok'] for c in checks), 'total': len(checks),
           'note': 'scripted client only; no LLM client object, no fit, no score; one real material build (cond) and screen-cache hits'}
    context.write_json(sm / 'smoke_result.json', out)
    print('SMOKE_DONE', out['passed'], '/', out['total'], flush=True)
    return out


# ============================================================================= wiring (real fits <= 6, Source only; task §4)
def wiring(root: Path = ROOT) -> dict:
    """W1/W2: the package fit + scorer reproduce the screen cells (cache equivalence) on D01 and D03; W3: a new conditional material through
    build -> fit -> score; W4: the package scorer on existing screen models of D02 (no fit). Wiring products are not agent experience."""
    P_ = paths(root)
    wd = P_['wiring']
    wd.mkdir(parents=True, exist_ok=True)
    split = build_split()
    sp = wd / 'split_copy.json'
    context.write_json(sp, split)
    css = cases_of(split)
    common = wd / 'common'
    DP.set_pools(CONCURRENCY['numeric'], 1)
    led = SafeLedger(wd / 'budget.json', max_fit_attempts=STAGE_A['fits_wiring'], max_llm_requests=0, max_llm_tokens=0, max_wall_s=3 * 3600, max_retries=1)
    checks, mem = [], {'free_gb_before': round(free_gb(), 2)}

    def check(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})
        print('WIRING', 'OK ' if ok else 'FAIL', name, info, flush=True)

    for case, phys, seed in (('D01_SCR_A1_G1', 'C_censor', SEEDS[0]), ('D03_SCR_A1_G1', 'P_NoMixRecipe', SEEDS[1])):
        prepare_common_source(common, css[case])
        # a cache-equivalence refit needs its own physical id: register the same screen material under a wiring alias (same file, same key)
        reg = ec.load_registry(common / case)
        alias = 'W_' + phys
        if alias not in reg:
            reg[alias] = {**reg[phys], 'material_id': alias}
            ec.save_registry(common / case, reg)
        t0 = time.time()
        got = DP.fit_cells(led, common, sp, case, [(alias, seed)])
        rec = got[ec.cell_id(case, alias, seed)]['record']
        mem['fit_seconds_%s' % case] = round(time.time() - t0, 1)
        scr = context.read_json(screen_dir(case) / 'cells' / (ec.cell_id(case, phys, seed) + '.json'))
        check('W_cache_equivalence_c_a_%s' % case, rec['scores']['c_a']['normalized_mse_macro'] == scr['scores']['c_a']['normalized_mse_macro'],
              mine=rec['scores']['c_a']['normalized_mse_macro'], screen=scr['scores']['c_a']['normalized_mse_macro'])
        fp = wd / ('%s__%s__e.json' % (case, alias))
        # scorer covers all seeds of a phys; wiring fitted one seed -> score through a one-seed copy of the worker logic
        rc, secs = run_numeric(['--worker-score-one', common, sp, case, fp, alias, seed], P_['logs'] / ('wiring_score_%s.log' % case), 600)
        mine_e = context.read_json(fp)['cells'][ec.cell_id(case, alias, seed)]['e'] if rc == 0 else None
        check('W_cache_equivalence_e_%s' % case, mine_e is not None and mine_e == screen_e(case)[phys][seed], mine=mine_e, screen=screen_e(case)[phys][seed])
    # W3: a new conditional material end to end (build -> fit one seed -> score)
    case = 'D01_SCR_A1_G1'
    ctx = ec.open_case(css[case], common, 'material')
    table = [{**r, **x} for r, x in zip(ctx.overview()['entities'], extra_fields(ctx))]
    comp = compile_zf({'default': {'steps': [{'op': 'tp_censor'}]}, 'rules': [{'when': {'feature': 'weekly_excess_r2', 'op': '>=', 'value': {'quantile': 0.5}},
                                                                             'steps': [{'op': ta.RECIPE}]}], 'rationale': 'wiring'}, table)
    rec_m = ensure_material_zf(common, sp, case, comp['assignment'], ledger=led)
    got = DP.fit_cells(led, common, sp, case, [(rec_m['material_id'], SEEDS[0])])
    fp = wd / ('%s__%s__e.json' % (case, rec_m['material_id']))
    rc, _ = run_numeric(['--worker-score-one', common, sp, case, fp, rec_m['material_id'], SEEDS[0]], P_['logs'] / 'wiring_score_new.log', 600)
    e_new = context.read_json(fp)['cells'][ec.cell_id(case, rec_m['material_id'], SEEDS[0])]['e'] if rc == 0 else None
    check('W_new_material_build_fit_score', rec_m['material_id'].startswith('TA') and e_new is not None and math.isfinite(e_new),
          phys=rec_m['material_id'], n_recipe=sum(1 for a in comp['assignment'] if a == [{'op': ta.RECIPE}]), e=e_new)
    # W4: the package scorer on an existing screen model (D02; no fit): same E as the screen
    case2 = 'D02_SCR_A1_G1'
    fp2 = wd / 'D02_screen_rescore.json'
    rc, _ = run_numeric(['--worker-score-dir', screen_dir(case2).parent, sp, case2, fp2, 'P_NoMixRecipe', SEEDS[2]], P_['logs'] / 'wiring_score_d02.log', 600)
    e2 = context.read_json(fp2)['cells'][ec.cell_id(case2, 'P_NoMixRecipe', SEEDS[2])]['e'] if rc == 0 else None
    check('W_scorer_equals_screen_D02', e2 is not None and e2 == screen_e(case2)['P_NoMixRecipe'][SEEDS[2]], mine=e2, screen=screen_e(case2)['P_NoMixRecipe'][SEEDS[2]])
    check('W_screen_dirs_untouched', all(sorted(ec.load_registry(screen_dir(c))) == sorted(SC.MENU_IDS) for c in ('D01_SCR_A1_G1', 'D03_SCR_A1_G1', 'D02_SCR_A1_G1')))
    mem['free_gb_after'] = round(free_gb(), 2)
    out = {'written_local': now(), 'checks': checks, 'passed': sum(c['ok'] for c in checks), 'total': len(checks), 'fits': led.s['fit_attempts'],
           'memory': mem, 'ledger': led.summary()}
    context.write_json(wd / 'wiring_result.json', out)
    print('WIRING_DONE', out['passed'], '/', out['total'], 'fits', out['fits'], flush=True)
    return out


def worker_score_one(root: str, split_path: str, case: str, out: str, phys: str, seed: int, *, screen_layout: bool = False) -> None:
    """Wiring scorer: one (phys, seed) cell, the same E computation as worker_score (for screen_layout the case json of the screen is used)."""
    ES._init_worker()
    from methods.ttha.batch_base import train
    cs = ES._case(split_path, case)
    jd = Path(root) / case
    sc = np.load(jd / 'scaler.npz')
    scaler = data.Scaler(mean=sc['mean'], std=sc['std'], scale=sc['scale'], floor_hits=0)
    sl = ec.load_case_slice(cs, cs.t, max(cs.e) + ec.H)
    wins = data.build_windows(sl, list(cs.e), scaler, with_truth=True)
    truth = np.stack([w.y_true_raw for w in wins], axis=1)
    cid = ec.cell_id(case, phys, int(seed))
    r = context.read_json(jd / 'cells' / (cid + '.json'))
    pred = train.predict_windows(train.load_model(r['model_path']), wins, scaler)
    sco = ec.score_with_status(pred, truth, scaler, sc['mase'])
    context.write_json(Path(out), {'cells': {cid: {'e': sco['normalized_mse_macro'], 'c_a': r['scores']['c_a']['normalized_mse_macro']}}, 'scored_local': now()})
    print('SCORE_ONE_OK', cid, sco['normalized_mse_macro'], flush=True)


# ============================================================================= stages B-D (task §6-§10; budget frozen by the user / Planner 2026-09-23 after the stage-A checkpoint)
BD_CAPS = {'tokens': 57_000_000, 'logical_requests': 1312, 'extra_transport': 16, 'http': 1328, 'fits': 1258, 'fit_retries': 20,
           'wall_warn_s': 12 * 3600, 'wall_cap_s': 24 * 3600}
PACKAGE_FIT_CAP = 1300
SCOPES = ('D01', 'D02', 'D03', 'SHARED')
SLOTS = (1, 2)
SELECT_ARMS = ('d1', 'd2', 's1', 's2')
TEST_ARMS = ('f0', 'f_shared', 'f_domain')
CARD_BODY_LIMIT = 6000
SLOW_TIMEOUT_S = 600.
SLOW_BYTE_LIMIT = 530_000                       # the envelope the backend accepted in earlier packages
MENU_STEPS = {m[0]: m[2] for m in SC.MENU}
MENU_LABEL = SC.LABEL
TEST_REFS = {'D01': ('None', 'P_NoMixRecipe', 'C_censor', 'C_shock'), 'D02': ('None', 'P_NoMixRecipe', 'C_censor', 'C_shock'),
             'D03': ('None', 'P_NoMixRecipe', 'C_shock')}          # Fixed_source D03 = NoMix (shared result)
TEST_EXTRA_BUILD = {'D01': ('C_censor', 'C_shock'), 'D02': ('C_censor', 'C_shock'), 'D03': ('C_shock',)}


def paths_bd(root: Path) -> dict:
    return {'ledger': root / 'budget_bd.json', 'frozen': root / 'frozen_bd.json', 'slow': root / 'slow', 'select': root / 'select', 'test': root / 'test',
            'freeze': root / 'freeze', 'common': root / 'common', 'split': root / 'split_copy.json', 'incidents': root / 'incidents', 'logs': root / 'logs'}


def ledger_bd(root: Path) -> SafeLedger:
    return SafeLedger(paths_bd(root)['ledger'], max_fit_attempts=BD_CAPS['fits'], max_llm_requests=BD_CAPS['http'], max_llm_tokens=BD_CAPS['tokens'],
                      max_wall_s=BD_CAPS['wall_cap_s'], max_retries=BD_CAPS['fit_retries'])


def freeze_bd(root: Path) -> dict:
    P_ = paths_bd(root)
    if P_['frozen'].exists():
        return context.read_json(P_['frozen'])
    a = context.read_json(paths(root)['ledger'])
    wiring_fits = context.read_json(paths(root)['wiring'] / 'wiring_result.json')['fits']
    rec = {'package': PACKAGE, 'frozen_local': now(), 'decision': 'user / Planner 2026-09-23 (after the stage-A checkpoint): continue B -> C -> D at unchanged scale; keep '
                                                                  'the tool outputs of stage A (no lossy compression, no shortened observation windows)',
           'caps_B_to_D': BD_CAPS, 'package_fit_cap': PACKAGE_FIT_CAP,
           'stage_a_used': {'fits': a['fit_attempts'] + wiring_fits, 'fits_final_commits': a['fit_attempts'], 'fits_wiring': wiring_fits,
                            'tokens': a['llm_tokens_in'] + a['llm_tokens_out'], 'requests': a['llm_requests']},
           'correction': 'the stage-A machine projection wrote 1261 remaining fits (it subtracted only the 39 final-commit fits); with the 3 wiring fits the remaining '
                         'package allowance is 1300 - 42 = 1258, the frozen cap here (stage-A records are not rerun)',
           'concurrency': {'http': 4, 'numeric': '2-4 by measured memory', 'min_free_gb': MIN_FREE_GB},
           'scale': {'slow_calls': 8, 'select_trajectories': 96, 'test_trajectories': 120},
           'incomplete_rule': 'a trajectory without a legal commit is INCOMPLETE; the Runner never supplies a plan; for J and G it is scored as the '
                              'no-augmentation delivery (ratio 1 / G 0 vs None) and counted separately',
           'shared_card_rule': 'deployed identically in all three domains; may branch on observations the tools compute, never on a domain id (text check), as in the '
                               'previous formal package', 'slow_max_output_tokens': MAX_OUTPUT_TOKENS, 'slow_request_timeout_s': SLOW_TIMEOUT_S}
    context.write_json(P_['frozen'], rec)
    return rec


def _prepared(common: Path, case: str) -> bool:
    return (common / case / 'prepared.json').exists()


def worker_prepare(split_path: str, case: str, root: str, extra: str) -> None:
    """New case (Select / Test): T scaler + overview, public materials and the frozen fixed references (torch 1 thread, as all materials of this family)."""
    ES._init_worker()
    import torch
    torch.set_num_threads(1)
    cs = ES._case(split_path, case)
    ctx = ec.open_case(cs, Path(root), stage='material')
    ctx.overview()
    ec.build_public(ctx)
    table = ctx.overview()['entities']
    for mid in [x for x in extra.split(',') if x]:
        if mid in ec.load_registry(ctx.job_dir):
            continue
        comp = ec.compile_plan_full(ec.uniform_policy(MENU_STEPS[mid], 'fixed reference %s' % MENU_LABEL[mid]), table)
        ec.build_material(ctx, comp['assignment'], mid)
    context.write_json(ctx.job_dir / 'prepared.json', {'case': case, 'materials': sorted(ec.load_registry(ctx.job_dir)), 'prepared_local': now()})
    print('PREPARED', case, flush=True)


def prepare_case(common: Path, split_path: Path, cs: ec.CaseSpec, extra=(), *, ledger=None) -> None:
    with case_lock(cs.case_id):
        if _prepared(common, cs.case_id):
            return
        rc, secs = run_numeric(['--worker-prepare', split_path, cs.case_id, common, ','.join(extra)], common / cs.case_id / 'fit_logs' / 'prepare.log', 1800)
        if rc != 0 or not _prepared(common, cs.case_id):
            raise RuntimeError('prepare failed for %s' % cs.case_id)
        if ledger is not None:
            ledger.add('material_wall_seconds', secs)
            ledger.event(kind='case_prepared', case=cs.case_id, seconds=round(secs, 1))


_FIT_LOCKS: dict = {}


def fit_phys(led, common: Path, split_path: Path, case: str, phys: str) -> None:
    """Three seeds of one physical material of one case; one owner per (case, phys), others wait and reuse (task §11)."""
    with _CASE_LOCKS_GUARD:
        lk = _FIT_LOCKS.setdefault((case, phys), threading.Lock())
    with lk:
        cells = ec.branch_cells(common / case)
        if all(ec.cell_id(case, phys, s) in cells for s in SEEDS):
            return
        fit_new(led, common, split_path, case, phys)


def score_case(common: Path, split_path: Path, case: str, physes: list, out: Path) -> dict:
    if out.exists():
        r = context.read_json(out)
        if all(ec.cell_id(case, p, s) in r['cells'] for p in physes for s in SEEDS):
            return r
    out.parent.mkdir(parents=True, exist_ok=True)
    rc, _ = run_numeric(['--worker-score', common, split_path, case, out] + sorted(physes), out.parent / ('score_%s.log' % case), 900)
    if rc != 0:
        raise RuntimeError('score worker failed for %s' % case)
    return context.read_json(out)


# ----------------------------------------------------------------------------- Slow census (task §6): trajectory, outcome and scripted records in the same case record
def _q5(vals):
    q = _q(vals)
    return _sig(q) if q else None


def _obs_summary(common: Path, cs: ec.CaseSpec) -> dict:
    ctx = ec.open_case(cs, common, 'material')
    table = [{**r, **x} for r, x in zip(ctx.overview()['entities'], extra_fields(ctx))]
    return {f: _q5([r.get(f) for r in table]) for f in OBS_FIELDS_ZF if f not in ('missing_count', 'n_parent_pairs')}


def _inspect_compact(out: dict) -> dict:
    if 'change_X' not in out:
        return {k: v for k, v in out.items() if k in ('material_id', 'note')}
    ents = out.get('entities') or {}
    rx = [v['rms_X'] for v in ents.values()]
    ry = [v['rms_y'] for v in ents.values()]
    c = {'material_id': out['material_id'], 'change_X': out['change_X'], 'change_y': out['change_y'], 'windows_bitwise_unchanged': out.get('windows_bitwise_unchanged'),
         'legal_windows': out.get('legal_windows'), 'entity_rms_X_min_median_max': _sig([min(rx), float(np.median(rx)), max(rx)]) if rx else None,
         'entity_rms_y_min_median_max': _sig([min(ry), float(np.median(ry)), max(ry)]) if ry else None}
    if out.get('recipe_edit'):
        c['recipe_edit_skip_fraction'] = out['recipe_edit'].get('skip_fraction_of_legal_windows')
    w = out.get('window')
    if w:
        p, ch = np.array(w['parent'], dtype=float), np.array(w['child'], dtype=float)
        c['window_seen'] = {'entity': w['entity'], 'choice': w['choice'], 'parent_min_max': _sig([float(p.min()), float(p.max())]),
                            'child_min_max': _sig([float(ch.min()), float(ch.max())]), 'max_abs_change': _sig(float(np.abs(ch - p).max())),
                            'mean_abs_change': _sig(float(np.abs(ch - p).mean())), 'note': 'the 240-point parent / child values Fast saw, reduced to these numbers'}
    return c


def _menu_key_label(case: str) -> dict:
    return {r['key']: MENU_LABEL[m] for m, r in ec.load_registry(screen_dir(case)).items()}


def compact_trajectory(bdir: Path, case: str) -> dict:
    rows = _trace(bdir)
    reg = ec.load_registry(bdir / case)
    k2l = _menu_key_label(case)
    out = []
    for r in rows:
        ev = r['event']
        if ev == 'fast_response':
            acts = []
            for a in r['response']['actions']:
                if a['tool'] == 'build_material':
                    acts.append({'tool': 'build_material', 'plan_id': a['arguments'].get('plan_id'), 'policy': a['arguments'].get('policy')})
                else:
                    acts.append(a)
            out.append({'call': r['number'], 'actions': acts})
        elif ev == 'response_rejected':
            out.append({'call': r['number'], 'response_rejected': r['error']['message']})
        elif ev == 'tool_completed':
            t, o = r['tool'], r['output']
            if t == 'overview':
                out.append({'tool': 'overview', 'result': 'per-entity T table read (see observations)'})
            elif t == 'inspect_data':
                out.append({'tool': 'inspect_data', 'result': {'kind': o.get('kind'), 'entities': sorted((o.get('entities') or {}).keys()), 'values': 'omitted (raw T views)'}})
            elif t == 'build_material':
                rec = reg.get(o['plan_id'], {})
                same = k2l.get(rec.get('key'))
                out.append({'tool': 'build_material', 'plan_id': o['plan_id'], 'label': o.get('label'),
                            'same_material_as_scripted_program': same,
                            'downstream_evidence': ('the scripted record of %s (below) is this exact material' % same) if same else 'none (never trained)'})
            elif t == 'inspect_material':
                out.append({'tool': 'inspect_material', 'result': _inspect_compact(o)})
            elif t == 'commit':
                out.append({'tool': 'commit', 'plan_id': o['plan_id'], 'reason': o['reason']})
        elif ev == 'tool_rejected':
            out.append({'tool': r['tool'], 'rejected': r['error']['message']})
        elif ev == 'action_batch_deferred':
            out.append({'deferred_not_executed': [a['tool'] for a in r['actions']], 'reason': r['reason']})
    return {'steps': out, 'note': 'the real no-card zero-feedback Fast trajectory: its actions (with plan rationales), compacted tool results and commit reason'}


def scripted_records(case: str) -> dict:
    se = screen_e(case)
    den = statistics.fmean(se['None'][s] for s in SEEDS)
    out = {}
    for m in SC.MENU_IDS:
        g_seed = [100.0 * (se['None'][s] - se[m][s]) / den for s in SEEDS]
        rec = {'program': MENU_LABEL[m], 'steps': MENU_STEPS[m] if MENU_STEPS[m] is not None else ('none' if m == 'None' else 'FixedMixup (historical w=0.25 same-entity mixup)'),
               'G_vs_None_mean': _sig(statistics.fmean(g_seed)), 'G_vs_None_by_seed': _sig(g_seed)}
        if m != 'None':
            sp = screen_dir(case) / 'aug_materials' / (m + '__summary.json')
            if sp.exists():
                s = context.read_json(sp)
                rec['material_change'] = _sig({'rms_X': s.get('rms_change_X'), 'rms_y': s.get('rms_change_y'), 'fraction_points_changed_X': s.get('fraction_points_changed_X'),
                                               'windows_bitwise_unchanged': s.get('windows_bitwise_unchanged')})
        out[MENU_LABEL[m]] = rec
    return out


def census_case(root: Path, case: str, css: dict) -> dict:
    P_ = paths(root)
    bdir = P_['stage_a'] / 'branches' / ('%s__f0' % case)
    ev = context.read_json(P_['eval'] / 'evaluation.json')['cases'][case]
    se = screen_e(case)
    den = statistics.fmean(se['None'][s] for s in SEEDS)
    cm = context.read_json(bdir / case / 'commit.json')
    g_seed = [100.0 * (se['None'][s] - ev['e_by_seed'][str(s)]) / den for s in SEEDS]
    scr = scripted_records(case)
    gm = statistics.fmean(g_seed)
    rank = 1 + sum(1 for v in scr.values() if v['G_vs_None_mean'] > gm)
    same = _menu_key_label(case).get(cm['material_key'])
    return {'case_ref': case, 'domain_id': case[:3], 'observations': _obs_summary(P_['common'], css[case]),
            'fast_trajectory': compact_trajectory(bdir, case),
            'fast_outcome': {'committed_plan': cm['label'], 'same_material_as_scripted_program': same, 'commit_reason': cm['reason'],
                             'G_vs_None_mean': _sig(gm), 'G_vs_None_by_seed': _sig(g_seed), 'rank_among_12_scripted_plus_commit': '%d of 13' % rank},
            'scripted_records': {'note': 'SCRIPTED experiment records: an evaluator trained each of these 12 fixed uniform programs on this case (3 seeds) and scored E '
                                         'as pp of the case None mean (positive = better than no augmentation). Produced by a screen before this study, not by an agent\'s '
                                         'research, and not available to a deployed Fast.', 'programs': scr}}


def legal_refs_of(cases: list) -> list:
    out = []
    for c in cases:
        out += ['source/%s/observations' % c, 'source/%s/trajectory' % c, 'source/%s/outcome' % c] + ['source/%s/scripted/%s' % (c, MENU_LABEL[m]) for m in SC.MENU_IDS]
    return out


def build_census(root: Path, scope: str, css: dict) -> dict:
    cfg = context.read_json(paths(root)['frozen'])
    doms = DOMAINS if scope == 'SHARED' else (scope,)
    cases = [c for d in doms for c in sorted(cfg['cases']['source'][d])]
    return {'scope': scope, 'domains': list(doms), 'n_cases': len(cases), 'cases': [census_case(root, c, css) for c in cases], 'legal_evidence_refs': legal_refs_of(cases),
            'units': 'G = 100 x (E(None) - E(plan)) / E(None) of that case, three-seed mean unless by_seed; E = four 48-hour forecasts at t+192..t+336 of the case'}


# ----------------------------------------------------------------------------- Slow contract
_NAME_TOKEN = re.compile(r'\b(?:electricity|traffic|solar|photovoltaic)\b', re.I)
_CASE_TOKEN_ZF = re.compile(r'(?<![A-Za-z0-9])(?:D0\d(?:_[A-Za-z0-9]+)?|SCR|A\d_G\d|[VQ]\d_G\d|SHARED)(?![A-Za-z0-9])')
_DATE_TOKEN = re.compile(r'\b(?:19|20)\d{2}-\d{1,2}\b|\b(?:january|february|april|june|july|august|september|october|november|december)\b', re.I)
_FEEDBACK_TOKEN = re.compile(r'(?<![A-Za-z0-9])C_?[AB](?![A-Za-z0-9])')


_OPERATOR_VOCAB = re.compile(r'(?<![A-Za-z0-9_])t0(?![A-Za-z0-9_])')      # the shock-onset parameter of the supplied operator table (instrument correction 2026-09-23 21:40)


def card_text_check(text: str, where: str) -> None:
    t = str(text or '')
    W.skill_text_check(_OPERATOR_VOCAB.sub('onset', t), where)      # the readiness entity rule (T\d{1,3}, case-insensitive) must not flag operator vocabulary
    for rx, what in ((_NAME_TOKEN, 'data source name'), (_CASE_TOKEN_ZF, 'case / domain / period / group id'), (ES_ENTITY_TOKEN, 'entity / column number'),
                     (_DATE_TOKEN, 'calendar date / month'), (_FEEDBACK_TOKEN, 'downstream feedback block (deployment has no downstream feedback)')):
        m = rx.search(t)
        if m:
            raise ValueError('%s carries a %s token %r' % (where, what, m.group(0)))


ES_ENTITY_TOKEN = DP.ENTITY_TOKEN

SLOW_ROLE = ('You are the offline Slow of a batch training-data augmentation Harness. You write ONE deployable card (a Workflow plus optional Principles) that a Fast '
             'agent loads read-only at deployment, or KEEP (no card). DEPLOYMENT IS ZERO-FEEDBACK: the Fast reads the case card (task, Consumer, T-only fields with a '
             'per-entity table on demand), inspects raw T views, builds complete augmentation plans and inspects what they changed, then commits ONE plan (a plan it '
             'constructed or a public reference: None, FixedMixup, P_AmpResample, P_NoMixRecipe); nothing is trained or scored before every commit is frozen, and no '
             'result returns to it. Its tools are exactly overview, inspect_data, build_material, inspect_material, commit; per case at most 6 requests, 18 tool calls, 4 '
             'constructed materials, one commit. The action space and the observable fields are in `deployment`.')
SLOW_EVIDENCE = ('The census holds completed Source cases (entity groups of 16 at two periods). Each case record puts together: (1) T observations (quantiles over the 16 '
                 'entities); (2) the REAL no-card zero-feedback Fast trajectory - what it observed, which plans it built with which stated hypotheses, what its material '
                 'inspections showed (reduced to numbers) and why it committed or abandoned a material; (3) the downstream result of its actual commit (E, three seeds, as '
                 'pp of the case\'s no-augmentation None); (4) SCRIPTED EXPERIMENT RECORDS: twelve fixed uniform programs that an evaluator trained and scored on the same '
                 'case (three seeds) together with their material-change statistics. The scripted records were produced by a screen before this study; they are not an '
                 'agent\'s research process and they are not available to a deployed Fast. A plan the Fast built that equals a scripted program is marked; other built plans '
                 'were never trained and have no downstream number.')
SLOW_GUIDE = ('Learn how to observe and construct. You need not presume that domains must differ. Relate every judgement in the trajectories to its recorded result '
              '(judgement -> result), including counterexamples: material appearance (smoother, flatter, larger change, windows that look unrealistic) shows WHAT changed, '
              'not downstream utility in either direction. An untried action is not harmful; a change in one entity\'s loss is not a reason to forbid grouping. The Fast '
              'at deployment has no downstream feedback: never write a step that needs a score, a training run or any feedback block. A default strategy may be learned; '
              'attribute it to the actual results, and say when the evidence is mixed. Do not hard-code case ids, entity lists, dates, periods or data source names; '
              'state every condition as an observable fact the tools compute (field values, inspections). Soft guidance never changes tool permissions.')
SLOW_SHARED = ('This is the SHARED card: the census holds three domains with eight cases each (domain ids are metadata only). Weight the three domains EQUALLY. The card is '
               'deployed identically in every domain: it may branch on observations the tools compute, never on a domain id. Say in cross_domain_note how the same text '
               'behaves when the domains\' observations differ.')
SLOW_FORMAT = ('Output exact JSON, no markdown: {"decision":"KEEP","rationale":"..."} OR {"decision":"PROPOSE","card":{"research_mode":"<short label>",'
               '"workflow":"<executable zero-feedback Workflow: observe / construct / inspect / commit>","principles":"<conditions, actions, reasons, exceptions>" or null,'
               '"applicability_summary":"<=400 characters","evidence_refs":["exact refs from legal_evidence_refs"],'
               '"rule_status":[{"rule":"<one main rule>","status":"supported|hypothesis|unresolved","support":["refs or short facts"],"counter":["refs or short facts"]}],'
               '"decision_change_vs_no_card":"<the concrete decision the card changes relative to the no-card Fast>",'
               '"counterexample":{"ref":"<one legal ref>","note":"<where the card should hold or may fail>"}%s}}. Workflow plus Principles render to at most %d characters. '
               'KEEP is valid and is not resampled.')


def slow_system_zf(scope: str) -> str:
    extra = ',"cross_domain_note":"<how the same text behaves across domains>"' if scope == 'SHARED' else ''
    return ' '.join([SLOW_ROLE, SLOW_EVIDENCE, SLOW_GUIDE] + ([SLOW_SHARED] if scope == 'SHARED' else []) + [SLOW_FORMAT % (extra, CARD_BODY_LIMIT)])


def slow_payload_zf(cen: dict, scope: str, slot: int) -> dict:
    cases = list(cen['cases']) if slot == 1 else list(reversed(cen['cases']))
    return {'scope': scope, 'proposal_slot': slot,
            'proposal_note': 'proposal %d of two independent proposals for this scope formed from the identical evidence (case order differs); the other is not visible to you' % slot,
            'deployment': {'tools': CONTRACTS_ZF, 'actions': ES.action_table(), 'fields': OBS_FIELDS_ZF, 'field_definitions': FIELD_DEFINITIONS, 'consumer': ec.CONSUMER,
                           'limits': {'requests': LIMITS.max_calls, 'tool_calls': LIMITS.max_tools, 'constructed_materials': LIMITS.max_materials, 'commit': 1},
                           'visible_domain_identity': 'the case card shows a neutral domain id; a domain card is loaded by it; a shared card must not use it'},
            'census': {k: v for k, v in cen.items() if k not in ('legal_evidence_refs', 'cases')} | {'cases': cases},
            'legal_evidence_refs': cen['legal_evidence_refs'], 'body_limit_characters': CARD_BODY_LIMIT}


CARD_KEYS = {'research_mode', 'workflow', 'principles', 'applicability_summary', 'evidence_refs', 'rule_status', 'decision_change_vs_no_card', 'counterexample'}


def parse_card(resp, *, scope: str, slot: int, legal_refs) -> dict:
    legal = set(legal_refs)
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'PROPOSE'):
        raise ValueError('decision must be KEEP or PROPOSE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale'}:
            raise ValueError('KEEP carries only rationale')
        return {'decision': 'KEEP', 'skill': None, 'meta': {'rationale': str(resp.get('rationale', ''))[:3000]}}
    if set(resp) != {'decision', 'card'} or not isinstance(resp['card'], dict):
        raise ValueError('PROPOSE carries exactly decision and one card object')
    c = resp['card']
    want = CARD_KEYS | ({'cross_domain_note'} if scope == 'SHARED' else set())
    if set(c) != want:
        raise ValueError('card keys must be exactly %s' % sorted(want))
    for k in ('decision_change_vs_no_card',) + (('cross_domain_note',) if scope == 'SHARED' else ()):
        if not isinstance(c[k], str) or not c[k].strip():
            raise ValueError('%s must be nonempty text' % k)
    ce = c['counterexample']
    if not isinstance(ce, dict) or set(ce) != {'ref', 'note'} or ce['ref'] not in legal or not isinstance(ce['note'], str) or not ce['note'].strip():
        raise ValueError('counterexample is {ref: one legal ref, note: nonempty text}')
    if not isinstance(c['research_mode'], str) or not c['research_mode'].strip() or len(c['research_mode']) > 80:
        raise ValueError('research_mode must be short nonempty text')
    card_text_check(c['research_mode'], 'research_mode')
    rs = ES.check_rule_status(c['rule_status'])
    s = dsk.make_skill(skill_id='%s-W%d' % (scope, slot), domain_id=scope, revision=1, workflow=c['workflow'], principles=c['principles'],
                       applicability_summary=c['applicability_summary'], compatibility_note='Formed for this study\'s zero-feedback deployment, fixed Consumer, L/H, hourly T of 672 and 16-entity cases.',
                       observable_applicability={'const': True}, evidence_refs=c['evidence_refs'], legal_evidence_refs=legal_refs, source_stage='propose',
                       status='CANDIDATE_TEST_ONLY', allowed_features=ALLOWED_FEATURES, text_check=card_text_check, body_limit=CARD_BODY_LIMIT)
    meta = {'research_mode': c['research_mode'], 'rule_status': rs, 'decision_change_vs_no_card': c['decision_change_vs_no_card'][:2000], 'counterexample': ce}
    if scope == 'SHARED':
        meta['cross_domain_note'] = c['cross_domain_note'][:2000]
    return {'decision': 'PROPOSE', 'skill': s, 'meta': meta}


def propose_card(payload: dict, call, *, scope: str, slot: int, legal_refs) -> dict:
    """One Slow call + at most one format / contract correction (task §6); technical faults are PROPOSE_CALL_FAILED, never a card."""
    attempts, receipts = [], []
    for i in range(2):
        try:
            text, rec = call(payload)
            receipts.append(rec)
        except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted, UnknownUsageBlock) as exc:
            attempts.append({'attempt': i, 'fault_kind': type(exc).__name__})
            return {'status': 'PROPOSE_CALL_FAILED', 'attempts': attempts, 'receipts': receipts, 'skill': None, 'meta': {}}
        prior = text
        try:
            raw = json.loads(text) if isinstance(text, str) else None
            out = parse_card(raw, scope=scope, slot=slot, legal_refs=legal_refs)
            attempts.append({'attempt': i, 'ok': True})
            return {'status': 'NO_CARD' if out['decision'] == 'KEEP' else 'PROPOSED', 'attempts': attempts, 'receipts': receipts, 'raw': raw, **out}
        except (ValueError, policy.PolicyError) as exc:
            attempts.append({'attempt': i, 'error': str(exc)[:400]})
        if i == 0:
            payload = {**payload, 'correction': {'previous_output_excerpt': (prior or '')[:4000], 'error': attempts[-1]['error'],
                                                 'instruction': 'Correct this format / contract error only; KEEP is allowed; do not seek a different outcome.'}}
    return {'status': 'PROPOSE_PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'receipts': receipts, 'skill': None, 'meta': {}}


def slow_stage_bd(root: Path, led, client, css: dict) -> dict:
    """task §6: 8 main Slow calls (D01/D02/D03 x 2 domain cards, 2 shared cards), in parallel (HTTP pool)."""
    P_ = paths_bd(root)
    summ_p = P_['slow'] / 'proposals.json'
    if summ_p.exists():
        return context.read_json(summ_p)
    cens, legal = {}, {}
    for scope in SCOPES:
        cp = P_['slow'] / scope / 'census.json'
        if not cp.exists():
            cp.parent.mkdir(parents=True, exist_ok=True)
            context.write_json(cp, build_census(root, scope, css))
        cens[scope] = context.read_json(cp)
        legal[scope] = cens[scope]['legal_evidence_refs']
    sizes = {}
    for scope in SCOPES:
        for k in SLOTS:
            b = len(json.dumps([{'role': 'system', 'content': slow_system_zf(scope)}, {'role': 'user', 'content': json.dumps(slow_payload_zf(cens[scope], scope, k), ensure_ascii=False)}],
                               ensure_ascii=False).encode('utf-8'))
            sizes['%s_%d' % (scope, k)] = b
            if b > SLOW_BYTE_LIMIT:
                raise RuntimeError('Slow payload %s slot %d is %d bytes > %d' % (scope, k, b, SLOW_BYTE_LIMIT))
    context.write_json(P_['slow'] / 'payload_sizes.json', sizes)
    results, lock = {}, threading.Lock()

    def one(scope, k):
        out_p = P_['slow'] / scope / ('proposal_%d.json' % k)
        if out_p.exists():
            r = context.read_json(out_p)
        else:
            call = lambda p: client.call('slow', '%s_%d' % (scope, k), p, slow_system_zf(scope), max_tokens=MAX_OUTPUT_TOKENS, meta={'scope': scope, 'slot': k},
                                         request_timeout=SLOW_TIMEOUT_S)
            res = propose_card(slow_payload_zf(cens[scope], scope, k), call, scope=scope, slot=k, legal_refs=legal[scope])
            r = {'scope': scope, 'slot': k, 'status': res['status'], 'attempts': res['attempts'], 'receipts': res['receipts'],
                 'skill': res['skill'].to_json() if res.get('skill') else None, 'meta': res.get('meta', {}), 'raw': res.get('raw'), 'written_local': now()}
            write_new(out_p, r)
        with lock:
            results[(scope, k)] = r
        print('SLOW', scope, k, r['status'], flush=True)

    threads = [threading.Thread(target=one, args=(s, k), daemon=True) for s in SCOPES for k in SLOTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summ = {'written_local': now(), 'scopes': {}}
    for scope in SCOPES:
        rows = []
        for k in SLOTS:
            r = results[(scope, k)]
            alias = None
            if r['status'] == 'PROPOSED' and k == 2 and results[(scope, 1)]['status'] == 'PROPOSED' and \
                    results[(scope, 1)]['skill']['rendered_body'] == r['skill']['rendered_body']:
                alias = 1
            arm = ('s%d' % k) if scope == 'SHARED' else ('d%d' % k)
            rows.append({'slot': k, 'arm': arm, 'status': r['status'], 'alias_of_slot': alias,
                         'body_chars': len(r['skill']['rendered_body']) if r['skill'] else 0})
        summ['scopes'][scope] = rows
    if any(results[k]['status'] in ('PROPOSE_CALL_FAILED',) for k in results) and rt.unknown_usage_blocks(led):
        raise RuntimeError('Slow stage stopped by unknown usage; resume after an operator decision')
    write_new(summ_p, summ)
    return summ


def card_of(root: Path, scope: str, slot: int):
    r = context.read_json(paths_bd(root)['slow'] / scope / ('proposal_%d.json' % slot))
    return dsk.skill_from_json(r['skill']) if r['status'] == 'PROPOSED' else None


def knowledge_of_card(skill) -> br.Knowledge:
    return dsk.skill_knowledge(skill) if skill is not None else no_card()


# ----------------------------------------------------------------------------- multi-arm zero-feedback stage (Select / Test)
def run_fast_stage(root: Path, stage: str, jobs: list, knowledge_of, *, led, client, css: dict, prep_extra, prefit, sdir: Path | None = None) -> dict:
    """jobs: ordered [(case, arm)]; every case is prepared once (numeric pool) before its trajectories; prefit(case) -> [phys] reference fits run
    alongside (numeric pool) and never feed any trajectory. <= 4 trajectories in flight; a killed trajectory restarts whole; a never-answered
    request is resumed once within the extra-transport budget. Dispatch stops on unknown usage of a non-accepted class."""
    P_ = paths_bd(root)
    sdir = sdir or P_[stage]
    common, split_path = P_['common'], P_['split']
    cases = list(dict.fromkeys(c for c, _ in jobs))
    ready = {c: threading.Event() for c in cases}
    failed_prep = {}
    errors, results = [], {}
    stop = threading.Event()

    def prep_all():
        q = list(cases)
        fq = []
        ql = threading.Lock()

        def w():
            while True:                                          # phase 1: prepare every case (its trajectories start as soon as it is ready)
                with ql:
                    if not q:
                        break
                    c = q.pop(0)
                try:
                    prepare_case(common, split_path, css[c], prep_extra(c), ledger=led)
                    with ql:
                        fq.extend((c, phys) for phys in prefit(c))
                except Exception as exc:  # noqa: BLE001
                    failed_prep[c] = repr(exc)[:300]
                finally:
                    ready[c].set()
            while True:                                          # phase 2: reference fits (never read by any trajectory)
                with ql:
                    if not fq and not q:
                        return
                    if not fq:
                        continue
                    c, phys = fq.pop(0)
                try:
                    fit_phys(led, common, split_path, c, phys)
                except Exception as exc:  # noqa: BLE001
                    errors.append(('prefit', c, phys, repr(exc)[:300]))
        ts = [threading.Thread(target=w, daemon=True) for _ in range(DP.pools().numeric_limit)]
        for t in ts:
            t.start()
        return ts

    prep_threads = prep_all()
    q = list(jobs)
    qlock = threading.Lock()

    def worker():
        while True:
            with qlock:
                if not q or stop.is_set():
                    return
                case, arm = q.pop(0)
            ready[case].wait()
            if case in failed_prep:
                errors.append(('prep', case, arm, failed_prep[case]))
                continue
            if rt.unknown_usage_blocks(led):
                stop.set()
                return
            bdir = sdir / 'branches' / ('%s__%s' % (case, arm))
            try:
                if bdir.exists() and not (bdir / 'branch_result.json').exists():
                    k = 1 + len(list(bdir.parent.glob(bdir.name + '__killed*')))
                    shutil.move(bdir, bdir.parent / ('%s__killed%d' % (bdir.name, k)))
                    led.event(kind='trajectory_restarted_after_kill', case=case, arm=arm)
                if (bdir / 'branch_result.json').exists():
                    r = context.read_json(bdir / 'branch_result.json')
                    if r['status'] == 'COMPLETE' or r['failure_kind'] != 'AGENT_CALL_FAILED':
                        results[(case, arm)] = r
                        continue
                    results[(case, arm)] = run_branch(bdir, case, knowledge_of(case, arm), led, client, common=common, split_path=split_path, cs=css[case], arm=arm, resume=True)
                else:
                    results[(case, arm)] = run_branch(bdir, case, knowledge_of(case, arm), led, client, common=common, split_path=split_path, cs=css[case], arm=arm)
            except Exception as exc:  # noqa: BLE001
                errors.append(('branch', case, arm, repr(exc)[:300]))
                print('BRANCH_ERROR', stage, case, arm, repr(exc)[:300], flush=True)
            wall = time.time() - led.s['started_epoch']
            if wall > BD_CAPS['wall_warn_s'] and not led.s.get('wall_warn_12h'):
                with led.lock:
                    led.s['wall_warn_12h'] = now()
                    led._save()
                print('WALL_WARNING_12H', flush=True)

    threads = [threading.Thread(target=worker, name='fast-%d' % i, daemon=True) for i in range(CONCURRENCY['http'])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    again = [k for k, r in results.items() if r.get('status') == 'INCOMPLETE' and r.get('failure_kind') == 'AGENT_CALL_FAILED']
    for case, arm in again:
        if led.s.get('extra_transport_attempts', 0) >= BD_CAPS['extra_transport'] or rt.unknown_usage_blocks(led):
            break
        with led.lock:
            led.s['extra_transport_attempts'] = led.s.get('extra_transport_attempts', 0) + 1
            led._save()
        bdir = sdir / 'branches' / ('%s__%s' % (case, arm))
        try:
            results[(case, arm)] = run_branch(bdir, case, knowledge_of(case, arm), led, client, common=common, split_path=split_path, cs=css[case], arm=arm, resume=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(('resume', case, arm, repr(exc)[:300]))
    for t in prep_threads:
        t.join()
    missing = [j for j in jobs if j not in results]
    summ = {'stage': stage, 'finished_local': now(), 'n_jobs': len(jobs), 'complete': sum(r.get('status') == 'COMPLETE' for r in results.values()),
            'incomplete': {'%s__%s' % k: [r.get('failure_kind'), r.get('reason')] for k, r in results.items() if r.get('status') != 'COMPLETE'},
            'missing': ['%s__%s' % j for j in missing], 'errors': errors, 'unknown_usage_blocking': rt.unknown_usage_blocks(led), 'pools': DP.pools().snapshot()}
    sdir.mkdir(parents=True, exist_ok=True)
    context.write_json(sdir / 'dispatch_summary.json', summ)
    print('STAGE_DISPATCH_DONE', stage, summ['complete'], '/', len(jobs), 'errors', len(errors), 'missing', len(missing), flush=True)
    return summ


def evaluate_bd(root: Path, stage: str, jobs: list, refs_of, *, led, css: dict, sdir: Path | None = None) -> dict:
    """After every commit of the stage is frozen: fit each needed physical material once per case (3 seeds), then score E per case."""
    P_ = paths_bd(root)
    sdir = sdir or P_[stage]
    common, split_path = P_['common'], P_['split']
    freeze_p = sdir / 'freeze.json'
    if not freeze_p.exists():
        commits = {}
        for case, arm in jobs:
            b = sdir / 'branches' / ('%s__%s' % (case, arm))
            if not (b / 'branch_result.json').exists():
                raise RuntimeError('stage %s not frozen: %s %s' % (stage, case, arm))
            cp = b / case / 'commit.json'
            commits['%s__%s' % (case, arm)] = context.read_json(cp) if cp.exists() else None
        write_new(freeze_p, {'frozen_local': now(), 'commits': commits})
    commits = context.read_json(freeze_p)['commits']
    need = {}
    for case, arm in jobs:
        need.setdefault(case, set(refs_of(case)))
        cm = commits['%s__%s' % (case, arm)]
        if cm is not None:
            need[case].add(cm['physical_material'])
    tasks = [(c, p) for c, ps in need.items() for p in sorted(ps)]
    errs, lock = [], threading.Lock()
    q = list(tasks)

    def w():
        while True:
            with lock:
                if not q:
                    return
                c, p = q.pop(0)
            try:
                fit_phys(led, common, split_path, c, p)
            except Exception as exc:  # noqa: BLE001
                errs.append(('fit', c, p, repr(exc)[:300]))
    ts = [threading.Thread(target=w, daemon=True) for _ in range(DP.pools().numeric_limit)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise RuntimeError('fits failed: %s' % errs[:3])
    scores = {}
    q2 = sorted(need)

    def s_w():
        while True:
            with lock:
                if not q2:
                    return
                c = q2.pop(0)
            try:
                r = score_case(common, split_path, c, sorted(need[c]), sdir / 'scores' / ('%s.json' % c))
                with lock:
                    scores[c] = {p: {str(s): r['cells'][ec.cell_id(c, p, s)]['e'] for s in SEEDS} for p in need[c]}
            except Exception as exc:  # noqa: BLE001
                errs.append(('score', c, repr(exc)[:300]))
    ts = [threading.Thread(target=s_w, daemon=True) for _ in range(DP.pools().numeric_limit)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise RuntimeError('scoring failed: %s' % errs[:3])
    res = {'stage': stage, 'evaluated_local': now(), 'e_by_case_phys_seed': scores, 'commits': commits, 'fit_tasks': len(tasks), 'ledger': led.summary()}
    context.write_json(sdir / 'evaluation.json', res)
    print('STAGE_EVAL_DONE', stage, len(scores), 'cases', len(tasks), 'phys', flush=True)
    return res


def _e_mean(ev: dict, case: str, phys: str) -> float:
    return statistics.fmean(ev['e_by_case_phys_seed'][case][phys][str(s)] for s in SEEDS)


def delivered_phys(ev: dict, case: str, arm: str) -> str:
    cm = ev['commits']['%s__%s' % (case, arm)]
    return cm['physical_material'] if cm else 'None'           # INCOMPLETE scored as the no-augmentation delivery (frozen_bd incomplete_rule)


# ----------------------------------------------------------------------------- Select (task §7)
def select_jobs(root: Path, cfg: dict) -> list:
    P_ = paths_bd(root)
    summ = context.read_json(P_['slow'] / 'proposals.json')
    jobs = []
    for d in DOMAINS:
        cases = sorted(cfg['cases']['select'][d])
        for i, c in enumerate(cases):
            arms = []
            for scope, pre in ((d, 'd'), ('SHARED', 's')):
                for row in summ['scopes'][scope]:
                    if row['status'] in ('PROPOSED', 'NO_CARD') and row['alias_of_slot'] is None:
                        arms.append(row['arm'])
            k = i % len(arms) if arms else 0
            for a in arms[k:] + arms[:k]:                       # rotate the start arm per case (order effects spread)
                jobs.append((c, a))
    # interleave domains: round-robin over the per-domain lists
    by = {d: [j for j in jobs if j[0][:3] == d] for d in DOMAINS}
    out = []
    while any(by.values()):
        for d in DOMAINS:
            if by[d]:
                out.append(by[d].pop(0))
    return out


def select_knowledge(root: Path):
    def k(case, arm):
        scope, slot = (case[:3], int(arm[1])) if arm.startswith('d') else ('SHARED', int(arm[1]))
        return knowledge_of_card(card_of(root, scope, slot))
    return k


def _arm_tokens(led: dict, stage_dir_name: str, bname: str) -> int:
    return sum((e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0) for e in led['events'] if e.get('kind') == 'llm_finished' and e.get('unit') == bname)


def selection(root: Path, cfg: dict) -> dict:
    P_ = paths_bd(root)
    out_p = P_['freeze'] / 'cards_frozen.json'
    if out_p.exists():
        return context.read_json(out_p)
    ev = context.read_json(P_['select'] / 'evaluation.json')
    led = context.read_json(P_['ledger'])
    summ = context.read_json(P_['slow'] / 'proposals.json')
    jobs = select_jobs(root, cfg)
    J, tok = {}, {}
    for case, arm in jobs:
        d = case[:3]
        ratio = _e_mean(ev, case, delivered_phys(ev, case, arm)) / _e_mean(ev, case, 'None')
        J.setdefault(arm, {}).setdefault(d, []).append(ratio)
        tok.setdefault(arm, {}).setdefault(d, []).append(_arm_tokens(led, 'select', '%s__%s' % (case, arm)))
    Jd = {arm: {d: statistics.fmean(v) for d, v in dd.items()} for arm, dd in J.items()}
    choice = {}
    for d in DOMAINS:
        cands = [a for a in ('d1', 'd2') if a in Jd and d in Jd[a]]
        if not cands:
            choice[d] = {'arm': None, 'slot': None, 'J': {}, 'tie_break': None, 'note': 'no usable domain card (technical failures); F_domain runs without a card'}
            continue
        best = min(cands, key=lambda a: (Jd[a][d], statistics.fmean(tok[a][d]), ('d1', 'd2').index(a)))
        choice[d] = {'arm': best, 'slot': int(best[1]), 'J': {a: Jd[a][d] for a in cands},
                     'tie_break': 'tokens' if len(cands) == 2 and Jd[cands[0]][d] == Jd[cands[1]][d] else None}
    sc = [a for a in ('s1', 's2') if a in Jd]
    Js = {a: statistics.fmean(Jd[a][d] for d in DOMAINS) for a in sc}
    best_s = min(sc, key=lambda a: (Js[a], statistics.fmean(v for d in DOMAINS for v in tok[a][d]), ('s1', 's2').index(a)))
    # alias slots (identical rendered body) share their source's readings
    rec = {'frozen_local': now(), 'rule': 'task §7: J_d(W) = mean over Select_d of mean-seed E(c,W) / E(c,None); domain card = argmin own J_d; shared = argmin equal-weight mean '
                                          'over the three domains; exact ties -> fewer deployment tokens -> fixed order; INCOMPLETE = None delivery (ratio 1)',
           'J_by_arm_domain': Jd, 'J_shared_equal_weight': Js, 'domain': choice, 'shared': {'arm': best_s, 'slot': int(best_s[1]), 'J': Js},
           'incomplete': {'%s__%s' % (c, a): True for c, a in jobs if ev['commits']['%s__%s' % (c, a)] is None},
           'proposals': summ}
    for d in DOMAINS:
        sk = card_of(root, d, choice[d]['slot']) if choice[d]['slot'] else None
        rec['domain'][d]['skill'] = sk.to_json() if sk else None
    sk = card_of(root, 'SHARED', rec['shared']['slot'])
    rec['shared']['skill'] = sk.to_json() if sk else None
    write_new(out_p, rec)
    return rec


# ----------------------------------------------------------------------------- Test (task §8)
def test_jobs(cfg: dict) -> list:
    by = {}
    for d in DOMAINS:
        cases = sorted(cfg['cases']['test'][d])
        lst = []
        for i, c in enumerate(cases):
            k = i % 3
            lst += [(c, a) for a in (TEST_ARMS[k:] + TEST_ARMS[:k])]
        by[d] = lst
    out = []
    while any(by.values()):
        for d in DOMAINS:
            if by[d]:
                out.append(by[d].pop(0))
    return out


def test_knowledge(root: Path):
    fr = context.read_json(paths_bd(root)['freeze'] / 'cards_frozen.json')

    def k(case, arm):
        if arm == 'f0':
            return no_card()
        sk = fr['shared']['skill'] if arm == 'f_shared' else fr['domain'][case[:3]]['skill']
        return knowledge_of_card(dsk.skill_from_json(sk) if sk else None)
    return k


# ----------------------------------------------------------------------------- readout (task §9)
def G2(a, b, den):
    return 100.0 * (b - a) / den


def _period(case: str) -> str:
    return case.split('_')[2]                                           # Q1..Q4


def _group(case: str) -> str:
    return case.split('_')[3]


def boot_ci(per_case: dict, key: str, B: int = 4000, seed: int = 2026092401) -> dict:
    """Cluster bootstrap preserving period and entity-group structure: within each domain, resample periods and groups with replacement and take
    the cases of the resampled period x group cross; domain mean, then equal-weight three-domain mean."""
    rng = np.random.default_rng(seed)
    dom_cases = {d: {c: r[key] for c, r in per_case.items() if c[:3] == d and r.get(key) is not None} for d in DOMAINS}
    vals = []
    for _ in range(B):
        ds = []
        for d, cc in dom_cases.items():
            ps = sorted({_period(c) for c in cc})
            gs = sorted({_group(c) for c in cc})
            P = rng.choice(ps, len(ps), replace=True)
            Gs = rng.choice(gs, len(gs), replace=True)
            v = [cc['%s_Q_%s_%s' % (d, p, g)] for p in P for g in Gs if '%s_Q_%s_%s' % (d, p, g) in cc]
            ds.append(statistics.fmean(v) if v else statistics.fmean(cc.values()))
        vals.append(statistics.fmean(ds))
    return {'lo95': float(np.percentile(vals, 2.5)), 'hi95': float(np.percentile(vals, 97.5)), 'B': B}


def readout_bd(root: Path = ROOT) -> dict:
    P_ = paths_bd(root)
    cfg = context.read_json(paths(root)['frozen'])
    ev = context.read_json(P_['test'] / 'evaluation.json')
    led = context.read_json(P_['ledger'])
    fr = context.read_json(P_['freeze'] / 'cards_frozen.json')
    jobs = test_jobs(cfg)
    per_case = {}
    for case in sorted({c for c, _ in jobs}):
        d = case[:3]
        den = _e_mean(ev, case, 'None')
        E = {arm: _e_mean(ev, case, delivered_phys(ev, case, arm)) for arm in TEST_ARMS}
        refs = {'None': 'None', 'NoMix': 'P_NoMixRecipe', 'Fixed_source': REFS['Fixed_source'][d], 'Fixed_global_source': 'C_shock'}
        ER = {k: _e_mean(ev, case, m) for k, m in refs.items()}
        rec = {'domain': d, 'period': _period(case), 'group': _group(case), 'E': E, 'E_refs': ER,
               'delivered': {arm: (ev['commits']['%s__%s' % (case, arm)] or {}).get('label', 'INCOMPLETE') for arm in TEST_ARMS},
               'delivered_phys': {arm: delivered_phys(ev, case, arm) for arm in TEST_ARMS},
               'G_vs_None': {arm: G2(E[arm], den, den) for arm in TEST_ARMS}, 'G_refs_vs_None': {k: G2(v, den, den) for k, v in ER.items()}}
        rec['dom_minus_f0'] = G2(E['f_domain'], E['f0'], den)
        rec['shared_minus_f0'] = G2(E['f_shared'], E['f0'], den)
        rec['dom_minus_shared'] = G2(E['f_domain'], E['f_shared'], den)
        for arm in TEST_ARMS:
            for k, v in ER.items():
                rec['%s_minus_%s' % (arm, k)] = G2(E[arm], v, den)
        se = ev['e_by_case_phys_seed'][case]
        rec['seed_dom_minus_f0'] = [100 * (se[rec['delivered_phys']['f0']][str(s)] - se[rec['delivered_phys']['f_domain']][str(s)]) / den for s in SEEDS]
        rec['seed_shared_minus_f0'] = [100 * (se[rec['delivered_phys']['f0']][str(s)] - se[rec['delivered_phys']['f_shared']][str(s)]) / den for s in SEEDS]
        rec['seed_dom_minus_shared'] = [100 * (se[rec['delivered_phys']['f_shared']][str(s)] - se[rec['delivered_phys']['f_domain']][str(s)]) / den for s in SEEDS]
        rec['tokens'] = {arm: _arm_tokens(led, 'test', '%s__%s' % (case, arm)) for arm in TEST_ARMS}
        per_case[case] = rec

    def agg(key, cases=None):
        cs = cases or list(per_case)
        dm = {d: statistics.fmean(per_case[c][key] for c in cs if per_case[c]['domain'] == d) for d in DOMAINS if any(per_case[c]['domain'] == d for c in cs)}
        return {'three_domain': statistics.fmean(dm.values()), 'by_domain': dm}

    def wtl(key, a=None, b=None):
        w = t = l = 0
        for c, r in per_case.items():
            same = a is not None and r['delivered_phys'][a] == r['delivered_phys'][b]
            if same or abs(r[key]) <= 1e-9:
                t += 1
            elif r[key] > 0:
                w += 1
            else:
                l += 1
        return [w, t, l]

    main = {}
    for key, a, b in (('dom_minus_f0', 'f_domain', 'f0'), ('shared_minus_f0', 'f_shared', 'f0'), ('dom_minus_shared', 'f_domain', 'f_shared')):
        main[key] = {**agg(key), 'wins_same_losses': wtl(key, a, b), 'max_harm': min(r[key] for r in per_case.values()), 'max_gain': max(r[key] for r in per_case.values()),
                     'cluster_bootstrap_95': boot_ci(per_case, key),
                     'leave_one_period_out': {p: agg(key, [c for c in per_case if per_case[c]['period'] != p])['three_domain'] for p in ('Q1', 'Q2', 'Q3', 'Q4')},
                     'by_period': {p: agg(key, [c for c in per_case if per_case[c]['period'] == p])['three_domain'] for p in ('Q1', 'Q2', 'Q3', 'Q4')},
                     'seed_level': [statistics.fmean(statistics.fmean(per_case[c]['seed_' + key][i] for c in per_case if per_case[c]['domain'] == d) for d in DOMAINS) for i in range(3)]}
    vs_refs = {}
    for arm in TEST_ARMS:
        for k in ('None', 'NoMix', 'Fixed_source', 'Fixed_global_source'):
            key = '%s_minus_%s' % (arm, k)
            vs_refs[key] = {**agg(key), 'wins_ties_losses': wtl(key)}
    arms_vs_none = {arm: agg_g(per_case, arm) for arm in TEST_ARMS}
    refs_vs_none = {k: {'three_domain': statistics.fmean(statistics.fmean(r['G_refs_vs_None'][k] for r in per_case.values() if r['domain'] == d) for d in DOMAINS),
                        'by_domain': {d: statistics.fmean(r['G_refs_vs_None'][k] for r in per_case.values() if r['domain'] == d) for d in DOMAINS}}
                    for k in ('NoMix', 'Fixed_source', 'Fixed_global_source')}
    deliv = {arm: dict(collections_counter(r['delivered'][arm] for r in per_case.values())) for arm in TEST_ARMS}
    behaviour = test_behaviour(root, jobs)
    res = {'package': PACKAGE, 'written_local': now(), 'cards': {'domain': {d: fr['domain'][d]['arm'] for d in DOMAINS}, 'shared': fr['shared']['arm']},
           'main': main, 'vs_refs': vs_refs, 'arms_vs_none': arms_vs_none, 'refs_vs_none': refs_vs_none, 'delivered_counts': deliv, 'behaviour': behaviour,
           'per_case': per_case, 'selection': {k: fr[k] for k in ('J_by_arm_domain', 'J_shared_equal_weight')}, 'costs': cost_bd(root)}
    context.write_json(root / 'result.json', res)
    return res


def collections_counter(it):
    import collections
    return collections.Counter(it)


def agg_g(per_case: dict, arm: str) -> dict:
    dm = {d: statistics.fmean(r['G_vs_None'][arm] for r in per_case.values() if r['domain'] == d) for d in DOMAINS}
    return {'three_domain': statistics.fmean(dm.values()), 'by_domain': dm}


def test_behaviour(root: Path, jobs: list) -> dict:
    P_ = paths_bd(root)
    out = {}
    for case, arm in jobs:
        b = P_['test'] / 'branches' / ('%s__%s' % (case, arm))
        rows = _trace(b)
        br_ = context.read_json(b / 'branch_result.json')
        builds = [r['output'] for r in rows if r['event'] == 'tool_completed' and r['tool'] == 'build_material']
        o = out.setdefault(arm, {'n': 0, 'complete': 0, 'calls': [], 'tools': [], 'materials': [], 'conditional_built': 0, 'preset_family_built': 0, 'inspect_material': 0,
                                 'commit_public': 0, 'commit_constructed': 0, 'tokens': []})
        o['n'] += 1
        o['complete'] += br_['status'] == 'COMPLETE'
        o['calls'].append(br_['calls'])
        o['tools'].append(br_['tool_calls'])
        o['materials'].append(br_['new_evaluations'])
        o['tokens'].append(br_.get('tokens_spent', 0))
        o['conditional_built'] += sum(1 for x in builds if len(x.get('programs') or []) > 1)
        o['preset_family_built'] += sum(1 for x in builds if 'Preset' in (x.get('label') or '') or 'Edit[' in (x.get('label') or ''))
        o['inspect_material'] += sum(1 for r in rows if r['event'] == 'tool_started' and r['tool'] == 'inspect_material')
        cp = b / case / 'commit.json'
        if cp.exists():
            cm = context.read_json(cp)
            o['commit_public' if cm['plan_id'] in PUBLIC else 'commit_constructed'] += 1
    for arm, o in out.items():
        for k in ('calls', 'tools', 'materials', 'tokens'):
            o[k] = _st(o[k])
    return out


def cost_bd(root: Path) -> dict:
    P_ = paths_bd(root)
    led = context.read_json(P_['ledger'])
    a = context.read_json(paths(root)['ledger'])
    wire = context.read_json(paths(root)['wiring'] / 'wiring_result.json')
    scr = context.read_json(SC.ROOT / 'ledger.json')
    by_stage = {}
    for e in led['events']:
        if e.get('kind') == 'llm_finished':
            st = e.get('stage', '?')
            row = by_stage.setdefault(st, {'requests': 0, 'tokens': 0})
            row['requests'] += 1
            row['tokens'] += (e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0)
    fits_by_stage = {}
    for e in led['events']:
        if e.get('kind') == 'fit_started':
            st = 'select' if '_V_' in e['cell'] else ('test' if '_Q_' in e['cell'] else '?')
            fits_by_stage[st] = fits_by_stage.get(st, 0) + 1
    return {'screen': {'fits': scr.get('fits_ok'), 'llm': 0},
            'stage_a': {'tokens': a['llm_tokens_in'] + a['llm_tokens_out'], 'requests': a['llm_requests'], 'fits': a['fit_attempts'] + wire['fits']},
            'b_to_d': {'tokens': led['llm_tokens_in'] + led['llm_tokens_out'], 'requests': led['llm_requests'], 'http': led['llm_http_attempts'],
                       'extra_transport': led.get('extra_transport_attempts', 0), 'unknown_usage': led['llm_tokens_unknown'], 'fits': led['fit_attempts'],
                       'fits_ok': led['fits_ok'], 'fits_failed': led['fits_failed'], 'retries': led['retries_used'], 'by_stage_llm': by_stage, 'fits_by_stage': fits_by_stage,
                       'material_wall_seconds': led.get('material_wall_seconds'), 'wall_hours': (time.time() - led['started_epoch']) / 3600.0},
            'package_fits_total': led['fit_attempts'] + a['fit_attempts'] + wire['fits']}


# ----------------------------------------------------------------------------- driver
def run_bd(root: Path = ROOT, *, numeric: int = CONCURRENCY['numeric']) -> dict:
    """B -> C -> D -> readout, idempotent per step (task §10: continuous after the budget freeze)."""
    P_ = paths_bd(root)
    freeze_bd(root)
    cfg = context.read_json(paths(root)['frozen'])
    split = context.read_json(P_['split'])
    css = cases_of(split)
    DP.set_pools(numeric, CONCURRENCY['http'])
    rt.FIT_RETRY = True
    with PackageLock(root, 'run_bd'):
        led = ledger_bd(root)
        a_fits = context.read_json(paths(root)['ledger'])['fit_attempts'] + context.read_json(paths(root)['wiring'] / 'wiring_result.json')['fits']
        if a_fits + BD_CAPS['fits'] > PACKAGE_FIT_CAP:
            raise RuntimeError('package fit cap')
        mk = lambda stage: ZFMeteredClient(led, root / stage / 'llm', http_cap=BD_CAPS['http'], stage=stage, incidents=P_['incidents'], extra_cap=BD_CAPS['extra_transport'])
        # B
        summ = slow_stage_bd(root, led, mk('slow'), css)
        print('STAGE_B_DONE', json.dumps({s: [r['status'] for r in rows] for s, rows in summ['scopes'].items()}), flush=True)
        # C
        sj = select_jobs(root, cfg)
        if not (P_['select'] / 'evaluation.json').exists():
            ds = run_fast_stage(root, 'select', sj, select_knowledge(root), led=led, client=mk('select'), css=css, prep_extra=lambda c: (), prefit=lambda c: ['None'])
            if ds['missing'] or ds['unknown_usage_blocking']:
                raise RuntimeError('select stage incomplete: %s' % ds['missing'][:5])
            evaluate_bd(root, 'select', sj, lambda c: ['None'], led=led, css=css)
        fr = selection(root, cfg)
        print('STAGE_C_DONE', json.dumps({'domain': {d: fr['domain'][d]['arm'] for d in DOMAINS}, 'shared': fr['shared']['arm']}), flush=True)
        # D
        tj = test_jobs(cfg)
        if not (P_['test'] / 'evaluation.json').exists():
            ds = run_fast_stage(root, 'test', tj, test_knowledge(root), led=led, client=mk('test'), css=css, prep_extra=lambda c: TEST_EXTRA_BUILD[c[:3]],
                                prefit=lambda c: list(TEST_REFS[c[:3]]))
            if ds['missing'] or ds['unknown_usage_blocking']:
                raise RuntimeError('test stage incomplete: %s' % ds['missing'][:5])
            evaluate_bd(root, 'test', tj, lambda c: list(TEST_REFS[c[:3]]), led=led, css=css)
        res = readout_bd(root)
        (root / 'tables.md').write_text(tables_bd(res), encoding='utf-8')
        print('STAGE_D_DONE', json.dumps({k: round(v['three_domain'], 3) for k, v in res['main'].items()}), flush=True)
        return res


# ----------------------------------------------------------------------------- smoke for B-D (scripted; no LLM client object, no fit)
def smoke_bd(root: Path = ROOT) -> dict:
    sm = root / 'smoke_bd'
    if sm.exists():
        shutil.rmtree(sm)
    sm.mkdir(parents=True)
    checks = []

    def check(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})
        print('SMOKE_BD', 'OK ' if ok else 'FAIL', name, info if not ok else '', flush=True)

    split = context.read_json(paths(root)['split'])
    css = cases_of(split)
    # census (real stage-A evidence; read-only)
    sizes = {}
    for scope in SCOPES:
        cen = build_census(root, scope, css)
        txt = json.dumps(cen, ensure_ascii=False)
        sizes[scope] = len(txt.encode('utf-8'))
        if scope == 'D01':
            cen_d01 = cen
        if scope == 'SHARED':
            b = len(json.dumps([{'role': 'system', 'content': slow_system_zf(scope)}, {'role': 'user', 'content': json.dumps(slow_payload_zf(cen, scope, 2), ensure_ascii=False)}],
                               ensure_ascii=False).encode('utf-8'))
            check('census_shared_within_byte_limit', b <= SLOW_BYTE_LIMIT, bytes=b)
    context.write_json(sm / 'census_sizes.json', sizes)
    c0 = cen_d01['cases'][0]
    check('census_case_has_trajectory_outcome_scripted', set(c0) >= {'observations', 'fast_trajectory', 'fast_outcome', 'scripted_records'}
          and len(c0['scripted_records']['programs']) == 12 and 'G_vs_None_mean' in c0['fast_outcome'])
    check('census_no_raw_windows', not re.search(r'"(parent|child)":\s*\[', json.dumps(cen_d01)), )
    check('census_marks_scripted_equivalent_builds', any(s.get('same_material_as_scripted_program') for c in cen_d01['cases'] for s in c['fast_trajectory']['steps'] if s.get('tool') == 'build_material'))
    check('census_domain_scope_only_own_domain', all(c['domain_id'] == 'D01' for c in cen_d01['cases']) and cen_d01['n_cases'] == 8)
    # card contract
    legal = cen_d01['legal_evidence_refs']
    good = {'decision': 'PROPOSE', 'card': {'research_mode': 'observe then construct', 'workflow': 'Read the batch summary; if lag24_corr is high build the preset and commit it unless inspections show invalid numbers.',
                                             'principles': None, 'applicability_summary': 'dense periodic batches', 'evidence_refs': legal[:3],
                                             'rule_status': [{'rule': 'prefer the preset', 'status': 'hypothesis', 'support': [legal[2]], 'counter': []}],
                                             'decision_change_vs_no_card': 'commits the preset instead of a mild single primitive', 'counterexample': {'ref': legal[1], 'note': 'may fail'}}}
    out = parse_card(good, scope='D01', slot=1, legal_refs=legal)
    check('card_parse_ok', out['decision'] == 'PROPOSE' and out['skill'].rendered_body.startswith('Workflow:'))
    kn = knowledge_of_card(out['skill'])
    r = kn.render({}, ALLOWED_FEATURES)
    check('card_renders_loaded', len(r['loaded']) == 1 and 'preset' in r['loaded'][0]['body'], render=r)
    for bad_text, what in (('Use C_A to decide.', 'feedback'), ('On D01_SCR_A1_G1 use censor.', 'case id'), ('For traffic use censor.', 'source name'),
                           ('Treat entity_3 separately.', 'entity')):
        bad = json.loads(json.dumps(good))
        bad['card']['workflow'] = bad_text
        try:
            parse_card(bad, scope='D01', slot=1, legal_refs=legal)
            ok = False
        except (ValueError, policy.PolicyError):
            ok = True
        check('card_rejects_%s' % what.replace(' ', '_'), ok)
    check('card_keep_is_no_card', parse_card({'decision': 'KEEP', 'rationale': 'x'}, scope='D01', slot=1, legal_refs=legal)['decision'] == 'KEEP')
    # correction path: first response malformed, second valid
    seq = ['not json', json.dumps(good)]
    res = propose_card({'x': 1}, lambda p: (seq.pop(0), {'request': 0}), scope='D01', slot=1, legal_refs=legal)
    check('propose_one_correction', res['status'] == 'PROPOSED' and len(res['attempts']) == 2)
    # a scripted Fast branch with the card: the card body reaches the request
    smc = paths(root)['smoke'] / 'common'
    case = 'D03_SCR_A1_G1'
    led = SafeLedger(sm / 'budget.json', max_fit_attempts=0, max_llm_requests=5, max_llm_tokens=10 ** 6, max_wall_s=3600, max_retries=0)
    DP.set_pools(2, 1)
    cap = []

    class FC:
        def call(self, role, unit, payload, system, *, max_tokens, meta):
            cap.append(payload)
            return json.dumps({'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'P_NoMixRecipe', 'reason': 'smoke'}}]}), {'request': 0, 'prompt_tokens': 100, 'completion_tokens': 10, 'message_bytes': 1000}
    old = DP.proxy_reachable
    DP.proxy_reachable = lambda: True
    try:
        rb = run_branch(sm / 'branches' / 'card_branch', case, kn, led, FC(), common=smc, split_path=paths(root)['smoke'] / 'split_copy.json', cs=cases_of(context.read_json(paths(root)['smoke'] / 'split_copy.json'))[case], arm='d1')
    finally:
        DP.proxy_reachable = old
    check('card_branch_guidance_loaded', rb['status'] == 'COMPLETE' and cap and cap[0]['guidance']['loaded'] and 'preset' in cap[0]['guidance']['loaded'][0]['body'])
    # job lists and selection arithmetic
    cfg = context.read_json(paths(root)['frozen'])
    tj = test_jobs(cfg)
    check('test_jobs_120_balanced', len(tj) == 120 and len(set(tj)) == 120 and all(sum(1 for c, a in tj if a == x) == 40 for x in TEST_ARMS))
    # bootstrap on a synthetic per-case table (a constant difference gives a degenerate interval at that constant)
    pc = {c: {'x': 1.0} for c in {c for c, _ in tj}}
    ci = boot_ci(pc, 'x', B=200)
    check('bootstrap_constant', abs(ci['lo95'] - 1.0) < 1e-12 and abs(ci['hi95'] - 1.0) < 1e-12)
    # new-case preparation (one real Test case; T rows only; builds public + fixed references, 0 fits)
    tc = 'D03_Q_Q1_G1'
    sp = sm / 'split_copy.json'
    context.write_json(sp, split)
    prepare_case(sm / 'common', sp, css[tc], TEST_EXTRA_BUILD['D03'])
    reg = ec.load_registry(sm / 'common' / tc)
    check('prepare_test_case', set(reg) >= {'None', 'FixedMixup', 'P_AmpResample', 'P_NoMixRecipe', 'C_shock'} and not (sm / 'common' / tc / 'cells').exists(), materials=sorted(reg))
    out = {'written_local': now(), 'checks': checks, 'passed': sum(c['ok'] for c in checks), 'total': len(checks), 'census_bytes': sizes}
    context.write_json(sm / 'smoke_bd_result.json', out)
    print('SMOKE_BD_DONE', out['passed'], '/', out['total'], flush=True)
    return out


# ----------------------------------------------------------------------------- tables.md of the test readout
def _fx(x, nd=2):
    return '—' if x is None else ('%+.*f' % (nd, x))


def tables_bd(res: dict) -> str:
    L = ['# DEV-AUG-OFFLINE-SKILL — Test tables (auto)', '',
         'Units: pp of the case None mean, G = 100 x (E(B) - E(A)) / E(None), three-seed means; positive = first arm better. Domain = equal weight of its cases; '
         'three-domain = equal weight of the domains. Cluster bootstrap resamples periods and entity groups within each domain (B = 4000).', '']
    L.append('## 1. Main comparisons')
    L.append('')
    L.append('| comparison | three-domain | D01 | D02 | D03 | 95% cluster CI | W / same / L | max harm | max gain | seed 1 / 2 / 3 |')
    L.append('|---|---:|---:|---:|---:|---|---|---:|---:|---|')
    names = {'dom_minus_f0': 'F_domain − F0', 'shared_minus_f0': 'F_shared − F0', 'dom_minus_shared': 'F_domain − F_shared'}
    for k, lab in names.items():
        m = res['main'][k]
        ci = m['cluster_bootstrap_95']
        L.append('| %s | **%s** | %s | %s | %s | [%s, %s] | %s | %s | %s | %s |' % (lab, _fx(m['three_domain']), _fx(m['by_domain']['D01']), _fx(m['by_domain']['D02']), _fx(m['by_domain']['D03']),
                 _fx(ci['lo95']), _fx(ci['hi95']), '/'.join(map(str, m['wins_same_losses'])), _fx(m['max_harm']), _fx(m['max_gain']), ' / '.join(_fx(x) for x in m['seed_level'])))
    L.append('')
    L.append('Leave one period out (three-domain): ' + '; '.join('%s: %s' % (lab, ', '.join('−%s %s' % (p, _fx(v)) for p, v in res['main'][k]['leave_one_period_out'].items()))
                                                               for k, lab in names.items()))
    L.append('')
    L.append('By period (three-domain): ' + '; '.join('%s: %s' % (lab, ', '.join('%s %s' % (p, _fx(v)) for p, v in res['main'][k]['by_period'].items())) for k, lab in names.items()))
    L.append('')
    L.append('## 2. Arms and references vs None')
    L.append('')
    L.append('| arm / reference | three-domain | D01 | D02 | D03 |')
    L.append('|---|---:|---:|---:|---:|')
    for arm, lab in (('f0', 'F0 (no card)'), ('f_shared', 'F_shared'), ('f_domain', 'F_domain')):
        a = res['arms_vs_none'][arm]
        L.append('| %s | %s | %s | %s | %s |' % (lab, _fx(a['three_domain']), _fx(a['by_domain']['D01']), _fx(a['by_domain']['D02']), _fx(a['by_domain']['D03'])))
    for k in ('NoMix', 'Fixed_source', 'Fixed_global_source'):
        a = res['refs_vs_none'][k]
        L.append('| %s | %s | %s | %s | %s |' % (k, _fx(a['three_domain']), _fx(a['by_domain']['D01']), _fx(a['by_domain']['D02']), _fx(a['by_domain']['D03'])))
    L.append('')
    L.append('## 3. Arms minus fixed references')
    L.append('')
    L.append('| arm − reference | three-domain | D01 | D02 | D03 | W / T / L |')
    L.append('|---|---:|---:|---:|---:|---|')
    for arm in ('f0', 'f_shared', 'f_domain'):
        for k in ('NoMix', 'Fixed_source', 'Fixed_global_source'):
            v = res['vs_refs']['%s_minus_%s' % (arm, k)]
            L.append('| %s − %s | %s | %s | %s | %s | %s |' % (arm, k, _fx(v['three_domain']), _fx(v['by_domain']['D01']), _fx(v['by_domain']['D02']), _fx(v['by_domain']['D03']),
                                                              '/'.join(map(str, v['wins_ties_losses']))))
    L.append('')
    L.append('## 4. Delivered programs (count of 40)')
    L.append('')
    for arm in ('f0', 'f_shared', 'f_domain'):
        L.append('- %s: %s' % (arm, ', '.join('%s %d' % (k, v) for k, v in sorted(res['delivered_counts'][arm].items(), key=lambda x: -x[1]))))
    L.append('')
    L.append('## 5. Behaviour and cost per trajectory')
    L.append('')
    L.append('| arm | complete | calls mean | tools mean | materials mean | conditional built | preset family built | inspect_material | commit public / constructed | tokens mean | tokens p90 |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|')
    for arm in ('f0', 'f_shared', 'f_domain'):
        b = res['behaviour'][arm]
        L.append('| %s | %d/%d | %.1f | %.1f | %.1f | %d | %d | %d | %d / %d | %.0f | %.0f |' % (arm, b['complete'], b['n'], b['calls']['mean'], b['tools']['mean'], b['materials']['mean'],
                 b['conditional_built'], b['preset_family_built'], b['inspect_material'], b['commit_public'], b['commit_constructed'], b['tokens']['mean'], b['tokens']['p90']))
    L.append('')
    L.append('## 6. Per case')
    L.append('')
    L.append('| case | F0 | F_shared | F_domain | F0 vs None | shared vs None | domain vs None | NoMix vs None | Fixed_source vs None | dom − F0 | shared − F0 | dom − shared |')
    L.append('|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for c, r in sorted(res['per_case'].items()):
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (c, r['delivered']['f0'], r['delivered']['f_shared'], r['delivered']['f_domain'], _fx(r['G_vs_None']['f0']),
                 _fx(r['G_vs_None']['f_shared']), _fx(r['G_vs_None']['f_domain']), _fx(r['G_refs_vs_None']['NoMix']), _fx(r['G_refs_vs_None']['Fixed_source']), _fx(r['dom_minus_f0']),
                 _fx(r['shared_minus_f0']), _fx(r['dom_minus_shared'])))
    L.append('')
    L.append('## 7. Selection (J = mean E(W) / E(None) on Select; lower is better)')
    L.append('')
    for arm, dd in res['selection']['J_by_arm_domain'].items():
        L.append('- %s: %s' % (arm, ', '.join('%s %.4f' % (d, v) for d, v in dd.items())))
    L.append('- shared equal-weight: ' + ', '.join('%s %.4f' % (a, v) for a, v in res['selection']['J_shared_equal_weight'].items()))
    L.append('- frozen: domain ' + json.dumps(res['cards']['domain']) + ', shared ' + res['cards']['shared'])
    return '\n'.join(L) + '\n'



# ============================================================================= DEV-AUG-OFFLINE-SKILL-NAIVE-CONTROL (post-hoc mechanism ablation on the exposed Test cases)
NAIVE = {'package': 'DEV-AUG-OFFLINE-SKILL-NAIVE-CONTROL', 'identity': 'POST_HOC_MECHANISM_ABLATION_ON_EXPOSED_TEST_CASES (not an independent validation; the main '
                                                                       'experiment, its frozen cards and arms are unchanged and not rerun)',
         'caps': {'fast_trajectories': 40, 'tokens': 9_000_000, 'fits': 126, 'fit_retries': 6, 'http_inflight': 4, 'numeric': 2,
                  'logical_requests': 40 * 6 + 3 * 2, 'extra_transport': 8, 'wall_cap_s': 12 * 3600}}
NAIVE_ARM = 'f_naive'


def paths_naive(root: Path) -> dict:
    d = root / 'naive'
    return {'dir': d, 'ledger': d / 'budget_naive.json', 'frozen': d / 'frozen_naive.json', 'cards': d / 'cards', 'naive': d, 'common': root / 'common',
            'split': root / 'split_copy.json', 'incidents': d / 'incidents'}


def ledger_naive(root: Path) -> SafeLedger:
    c = NAIVE['caps']
    return SafeLedger(paths_naive(root)['ledger'], max_fit_attempts=c['fits'], max_llm_requests=c['logical_requests'] + c['extra_transport'], max_llm_tokens=c['tokens'],
                      max_wall_s=c['wall_cap_s'], max_retries=c['fit_retries'])


NAIVE_EVIDENCE = ('You receive NO downstream evidence: no research trajectories, no plan results, no scores of any block, no earlier cards and no test results. You receive only '
                  'the task, the Consumer, the complete operator descriptions (deployment.actions, deployment.tools) and the T-side observation summaries (quantiles over the 16 '
                  'entities of every T-only field) of eight development cases of ONE domain. Write the card from task knowledge and these observations.')
NAIVE_GUIDE = ('Learn how to observe and construct. You need not presume anything about which treatment is best. An untried action is not harmful; a change in one entity\'s '
               'loss is not a reason to forbid grouping. The Fast at deployment has no downstream feedback: never write a step that needs a score, a training run or any feedback '
               'block. Material appearance (smoother, flatter, larger change, windows that look unrealistic) shows WHAT changed, not downstream utility in either direction. Do '
               'not hard-code case ids, entity lists, dates, periods or data source names; state every condition as an observable fact the tools compute (field values, '
               'inspections). Soft guidance never changes tool permissions.')
NAIVE_FORMAT = ('Output exact JSON, no markdown: {"decision":"KEEP","rationale":"..."} OR {"decision":"PROPOSE","card":{"research_mode":"<short label>",'
                '"workflow":"<executable zero-feedback Workflow: observe / construct / inspect / commit>","principles":"<conditions, actions, reasons, exceptions>" or null,'
                '"applicability_summary":"<=400 characters","evidence_refs":["the observation refs you relied on, exact ids from legal_evidence_refs"],'
                '"rationale":"<why, from task knowledge and the observations>"}}. Workflow plus Principles render to at most %d characters. KEEP is valid and is not resampled.' % CARD_BODY_LIMIT)
NAIVE_KEYS = {'research_mode', 'workflow', 'principles', 'applicability_summary', 'evidence_refs', 'rationale'}


def naive_system() -> str:
    return ' '.join([SLOW_ROLE, NAIVE_EVIDENCE, NAIVE_GUIDE, NAIVE_FORMAT])


def naive_payload(root: Path, domain: str, css: dict) -> dict:
    cfg = context.read_json(paths(root)['frozen'])
    cases = sorted(cfg['cases']['source'][domain])
    obs = [{'case_ref': c, 'observations': _obs_summary(paths(root)['common'], css[c])} for c in cases]
    refs = ['source/%s/observations' % c for c in cases]
    return {'scope': domain, 'deployment': {'tools': CONTRACTS_ZF, 'actions': ES.action_table(), 'fields': OBS_FIELDS_ZF, 'field_definitions': FIELD_DEFINITIONS,
                                            'consumer': ec.CONSUMER, 'limits': {'requests': LIMITS.max_calls, 'tool_calls': LIMITS.max_tools,
                                                                                'constructed_materials': LIMITS.max_materials, 'commit': 1},
                                            'visible_domain_identity': 'the case card shows a neutral domain id; this card is loaded by it'},
            'development_observations': {'note': 'T-only observation quantiles of eight development cases of this domain (16 entities each); no downstream result exists here',
                                         'cases': obs},
            'legal_evidence_refs': refs, 'body_limit_characters': CARD_BODY_LIMIT}


def parse_naive(resp, *, domain: str, legal_refs) -> dict:
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'PROPOSE'):
        raise ValueError('decision must be KEEP or PROPOSE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale'}:
            raise ValueError('KEEP carries only rationale')
        return {'decision': 'KEEP', 'skill': None, 'meta': {'rationale': str(resp.get('rationale', ''))[:3000]}}
    if set(resp) != {'decision', 'card'} or not isinstance(resp['card'], dict):
        raise ValueError('PROPOSE carries exactly decision and one card object')
    c = resp['card']
    if set(c) != NAIVE_KEYS:
        raise ValueError('card keys must be exactly %s' % sorted(NAIVE_KEYS))
    if not isinstance(c['research_mode'], str) or not c['research_mode'].strip() or len(c['research_mode']) > 80:
        raise ValueError('research_mode must be short nonempty text')
    card_text_check(c['research_mode'], 'research_mode')
    if not isinstance(c['rationale'], str) or not c['rationale'].strip():
        raise ValueError('rationale must be nonempty text')
    s = dsk.make_skill(skill_id='NAIVE-%s' % domain, domain_id=domain, revision=1, workflow=c['workflow'], principles=c['principles'],
                       applicability_summary=c['applicability_summary'], compatibility_note='Naive control: task knowledge and T-side observations only (no downstream evidence).',
                       observable_applicability={'const': True}, evidence_refs=c['evidence_refs'], legal_evidence_refs=legal_refs, source_stage='propose',
                       status='CANDIDATE_TEST_ONLY', allowed_features=ALLOWED_FEATURES, text_check=card_text_check, body_limit=CARD_BODY_LIMIT)
    return {'decision': 'PROPOSE', 'skill': s, 'meta': {'research_mode': c['research_mode'], 'rationale': c['rationale'][:3000]}}


def naive_cards(root: Path, led, client, css: dict) -> dict:
    """One naive card per domain (one call + at most one format correction; no selection, no revision), in parallel."""
    P_ = paths_naive(root)
    summ_p = P_['cards'] / 'naive_cards.json'
    if summ_p.exists():
        return context.read_json(summ_p)
    out, lock = {}, threading.Lock()

    def one(d):
        p_ = P_['cards'] / ('%s.json' % d)
        if p_.exists():
            r = context.read_json(p_)
        else:
            payload = naive_payload(root, d, css)
            low = json.dumps(payload, ensure_ascii=False)
            for bad in ('G_vs_None', 'fast_trajectory', 'scripted_records', 'fast_outcome', 'e_by_seed', 'normalized_mse'):
                if bad in low:
                    raise PermissionError('naive payload carries downstream evidence: %s' % bad)
            legal = payload['legal_evidence_refs']
            attempts, receipts, res = [], [], None
            for i in range(2):
                try:
                    text, rec = client.call('slow_naive', 'naive_%s' % d, payload, naive_system(), max_tokens=MAX_OUTPUT_TOKENS, meta={'scope': d}, request_timeout=SLOW_TIMEOUT_S)
                    receipts.append(rec)
                except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted, UnknownUsageBlock) as exc:
                    attempts.append({'attempt': i, 'fault_kind': type(exc).__name__})
                    res = {'status': 'PROPOSE_CALL_FAILED'}
                    break
                try:
                    raw = json.loads(text) if isinstance(text, str) else None
                    pr = parse_naive(raw, domain=d, legal_refs=legal)
                    attempts.append({'attempt': i, 'ok': True})
                    res = {'status': 'NO_CARD' if pr['decision'] == 'KEEP' else 'PROPOSED', 'skill': pr['skill'].to_json() if pr['skill'] else None, 'meta': pr['meta'], 'raw': raw}
                    break
                except ValueError as exc:
                    attempts.append({'attempt': i, 'error': str(exc)[:400]})
                    if i == 0:
                        payload = {**payload, 'correction': {'previous_output_excerpt': (text or '')[:4000], 'error': attempts[-1]['error'],
                                                             'instruction': 'Correct this format / contract error only; KEEP is allowed; do not seek a different outcome.'}}
            if res is None:
                res = {'status': 'PROPOSE_PARSE_OR_VALIDATION_FAILED'}
            r = {'domain': d, **res, 'attempts': attempts, 'receipts': receipts, 'written_local': now(), 'payload_bytes': len(json.dumps(payload, ensure_ascii=False).encode('utf-8'))}
            write_new(p_, r)
        with lock:
            out[d] = r
        print('NAIVE_CARD', d, r['status'], flush=True)

    ts = [threading.Thread(target=one, args=(d,), daemon=True) for d in DOMAINS]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    summ = {'written_local': now(), 'cards': {d: {'status': out[d]['status'], 'attempts': len(out[d]['attempts']),
                                                  'body_chars': len(out[d]['skill']['rendered_body']) if out[d].get('skill') else 0} for d in DOMAINS}}
    if any(out[d]['status'] == 'PROPOSE_CALL_FAILED' for d in DOMAINS):
        raise RuntimeError('a naive card call failed technically; resume after the fault is cleared')
    write_new(summ_p, summ)
    return summ


def naive_knowledge(root: Path):
    def k(case, arm):
        r = context.read_json(paths_naive(root)['cards'] / ('%s.json' % case[:3]))
        return knowledge_of_card(dsk.skill_from_json(r['skill']) if r.get('skill') else None)
    return k


def naive_jobs(cfg: dict) -> list:
    by = {d: [(c, NAIVE_ARM) for c in sorted(cfg['cases']['test'][d])] for d in DOMAINS}
    out = []
    while any(by.values()):
        for d in DOMAINS:
            if by[d]:
                out.append(by[d].pop(0))
    return out


def readout_naive(root: Path = ROOT) -> dict:
    P_ = paths_naive(root)
    cfg = context.read_json(paths(root)['frozen'])
    evt = context.read_json(paths_bd(root)['test'] / 'evaluation.json')
    evn = context.read_json(P_['naive'] / 'evaluation.json')
    ledn = context.read_json(P_['ledger'])
    ledt = context.read_json(paths_bd(root)['ledger'])
    per_case = {}
    none_check = []
    for case in sorted(cfg['cases']['test']['D01'] + cfg['cases']['test']['D02'] + cfg['cases']['test']['D03']):
        d = case[:3]
        den = _e_mean(evt, case, 'None')
        none_check.append(abs(_e_mean(evn, case, 'None') - den))
        E = {arm: _e_mean(evt, case, delivered_phys(evt, case, arm)) for arm in TEST_ARMS}
        E[NAIVE_ARM] = _e_mean(evn, case, delivered_phys(evn, case, NAIVE_ARM))
        refs = {'None': 'None', 'NoMix': 'P_NoMixRecipe', 'Fixed_source': REFS['Fixed_source'][d], 'Fixed_global_source': 'C_shock'}
        ER = {k: _e_mean(evt, case, m) for k, m in refs.items()}
        dp = {arm: delivered_phys(evt, case, arm) for arm in TEST_ARMS}
        dp[NAIVE_ARM] = delivered_phys(evn, case, NAIVE_ARM)
        lab = {arm: (evt['commits']['%s__%s' % (case, arm)] or {}).get('label', 'INCOMPLETE') for arm in TEST_ARMS}
        lab[NAIVE_ARM] = (evn['commits']['%s__%s' % (case, NAIVE_ARM)] or {}).get('label', 'INCOMPLETE')
        rec = {'domain': d, 'period': _period(case), 'group': _group(case), 'E': E, 'delivered': lab, 'delivered_phys': dp, 'ref_phys': refs,
               'G_vs_None': {a: G2(E[a], den, den) for a in E}, 'G_refs_vs_None': {k: G2(v, den, den) for k, v in ER.items()}}
        for a, b, key in (('f_domain', NAIVE_ARM, 'dom_minus_naive'), (NAIVE_ARM, 'f0', 'naive_minus_f0'), ('f_shared', NAIVE_ARM, 'shared_minus_naive')):
            rec[key] = G2(E[a], E[b], den)
            rec['seed_' + key] = [100 * (_seed_e(evt, evn, case, dp[b], s) - _seed_e(evt, evn, case, dp[a], s)) / den for s in SEEDS]
        for k, v in ER.items():
            rec['naive_minus_%s' % k] = G2(E[NAIVE_ARM], v, den)
        rec['tokens_naive'] = _arm_tokens(ledn, 'naive', '%s__%s' % (case, NAIVE_ARM))
        per_case[case] = rec

    def agg(key, cases=None):
        cs = cases or list(per_case)
        dm = {d: statistics.fmean(per_case[c][key] for c in cs if per_case[c]['domain'] == d) for d in DOMAINS}
        return {'three_domain': statistics.fmean(dm.values()), 'by_domain': dm}

    def wtl(key, a, b):
        w = t = l = 0
        for r in per_case.values():
            if r['delivered_phys'][a] == r['delivered_phys'][b] or abs(r[key]) <= 1e-9:
                t += 1
            elif r[key] > 0:
                w += 1
            else:
                l += 1
        return [w, t, l]

    main = {}
    for key, a, b in (('dom_minus_naive', 'f_domain', NAIVE_ARM), ('naive_minus_f0', NAIVE_ARM, 'f0'), ('shared_minus_naive', 'f_shared', NAIVE_ARM)):
        main[key] = {**agg(key), 'wins_same_losses': wtl(key, a, b), 'max_harm': min(r[key] for r in per_case.values()), 'max_gain': max(r[key] for r in per_case.values()),
                     'cluster_bootstrap_95': boot_ci(per_case, key),
                     'leave_one_period_out': {p: agg(key, [c for c in per_case if per_case[c]['period'] != p])['three_domain'] for p in ('Q1', 'Q2', 'Q3', 'Q4')},
                     'seed_level': [statistics.fmean(statistics.fmean(per_case[c]['seed_' + key][i] for c in per_case if per_case[c]['domain'] == d) for d in DOMAINS) for i in range(3)]}
    def wtl_ref(k):
        w = t = l = 0
        for r in per_case.values():
            v = r['naive_minus_%s' % k]
            if r['delivered_phys'][NAIVE_ARM] == r['ref_phys'][k] or abs(v) <= 1e-9:
                t += 1
            elif v > 0:
                w += 1
            else:
                l += 1
        return [w, t, l]
    naive_vs_refs = {k: {**agg('naive_minus_%s' % k), 'wins_same_losses': wtl_ref(k)} for k in ('None', 'NoMix', 'Fixed_source', 'Fixed_global_source')}
    arms_vs_none = {a: {'three_domain': statistics.fmean(statistics.fmean(r['G_vs_None'][a] for r in per_case.values() if r['domain'] == d) for d in DOMAINS),
                        'by_domain': {d: statistics.fmean(r['G_vs_None'][a] for r in per_case.values() if r['domain'] == d) for d in DOMAINS}} for a in TEST_ARMS + (NAIVE_ARM,)}
    beh = test_behaviour_stage(root, P_['naive'], naive_jobs(cfg))
    deliv = dict(collections_counter(r['delivered'][NAIVE_ARM] for r in per_case.values()))
    same_as = {a: sum(1 for r in per_case.values() if r['delivered_phys'][NAIVE_ARM] == r['delivered_phys'][a]) for a in TEST_ARMS}
    same_as.update({'NoMix': sum(1 for r in per_case.values() if r['delivered_phys'][NAIVE_ARM] == 'P_NoMixRecipe'),
                    'None': sum(1 for r in per_case.values() if r['delivered_phys'][NAIVE_ARM] == 'None')})
    toks = [r['tokens_naive'] for r in per_case.values()]
    res = {'package': NAIVE['package'], 'identity': NAIVE['identity'], 'written_local': now(), 'main': main, 'naive_vs_refs': naive_vs_refs, 'arms_vs_none': arms_vs_none,
           'naive_delivered_counts': deliv, 'naive_same_delivery_as': same_as, 'behaviour_naive': beh, 'per_case': per_case,
           'none_consistency_max_abs_diff': max(none_check),
           'cost': {'naive_cards': {e.get('unit'): (e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0) for e in ledn['events'] if e.get('kind') == 'llm_finished' and str(e.get('unit', '')).startswith('naive_')},
                    'tokens_total': ledn['llm_tokens_in'] + ledn['llm_tokens_out'], 'requests': ledn['llm_requests'], 'http': ledn['llm_http_attempts'],
                    'unknown_usage': ledn['llm_tokens_unknown'], 'fits': ledn['fit_attempts'], 'fits_ok': ledn['fits_ok'], 'fits_failed': ledn['fits_failed'],
                    'tokens_per_naive_trajectory': _st(toks)}}
    context.write_json(P_['naive'] / 'result_naive.json', res)
    return res


def _seed_e(evt: dict, evn: dict, case: str, phys: str, seed: int) -> float:
    src = evt['e_by_case_phys_seed'][case] if phys in evt['e_by_case_phys_seed'][case] else evn['e_by_case_phys_seed'][case]
    return src[phys][str(seed)]


def test_behaviour_stage(root: Path, sdir: Path, jobs: list) -> dict:
    o = {'n': 0, 'complete': 0, 'calls': [], 'tools': [], 'materials': [], 'conditional_built': 0, 'preset_family_built': 0, 'inspect_material': 0,
         'commit_public': 0, 'commit_constructed': 0, 'tokens': [], 'built_labels': {}}
    for case, arm in jobs:
        b = sdir / 'branches' / ('%s__%s' % (case, arm))
        rows = _trace(b)
        br_ = context.read_json(b / 'branch_result.json')
        builds = [r['output'] for r in rows if r['event'] == 'tool_completed' and r['tool'] == 'build_material']
        o['n'] += 1
        o['complete'] += br_['status'] == 'COMPLETE'
        o['calls'].append(br_['calls'])
        o['tools'].append(br_['tool_calls'])
        o['materials'].append(br_['new_evaluations'])
        o['tokens'].append(br_.get('tokens_spent', 0))
        o['conditional_built'] += sum(1 for x in builds if len(x.get('programs') or []) > 1)
        o['preset_family_built'] += sum(1 for x in builds if 'Preset' in (x.get('label') or '') or 'Edit[' in (x.get('label') or ''))
        o['inspect_material'] += sum(1 for r in rows if r['event'] == 'tool_started' and r['tool'] == 'inspect_material')
        for x in builds:
            o['built_labels'][x.get('label')] = o['built_labels'].get(x.get('label'), 0) + 1
        cp = b / case / 'commit.json'
        if cp.exists():
            cm = context.read_json(cp)
            o['commit_public' if cm['plan_id'] in PUBLIC else 'commit_constructed'] += 1
    for k in ('calls', 'tools', 'materials', 'tokens'):
        o[k] = _st(o[k])
    return o


def run_naive(root: Path = ROOT, *, numeric: int = 2) -> dict:
    P_ = paths_naive(root)
    P_['dir'].mkdir(parents=True, exist_ok=True)
    if not P_['frozen'].exists():
        context.write_json(P_['frozen'], {**NAIVE, 'frozen_local': now(), 'task': 'user 2026-09-23 (DEV-AUG-OFFLINE-SKILL-NAIVE-CONTROL)',
                                          'arm': NAIVE_ARM, 'cases': 'the 40 Test cases of the main experiment', 'same_as_main': 'zero-feedback loop, tools, fields, per-trajectory caps, '
                                          'Fast system text, model and sampling of the main Test stage; E / fits / cache of common/<case>',
                                          'naive_slow_system': naive_system(), 'incomplete_rule': 'INCOMPLETE = no-augmentation delivery for G (counted separately)'})
    cfg = context.read_json(paths(root)['frozen'])
    split = context.read_json(P_['split'])
    css = cases_of(split)
    DP.set_pools(numeric, 4)
    rt.FIT_RETRY = True
    with PackageLock(root, 'run_naive'):
        led = ledger_naive(root)
        mk = lambda stage: ZFMeteredClient(led, P_['dir'] / stage / 'llm', http_cap=NAIVE['caps']['logical_requests'] + NAIVE['caps']['extra_transport'], stage=stage,
                                           incidents=P_['incidents'], extra_cap=NAIVE['caps']['extra_transport'])
        summ = naive_cards(root, led, mk('cards'), css)
        print('NAIVE_CARDS_DONE', json.dumps({d: v['status'] for d, v in summ['cards'].items()}), flush=True)
        jobs = naive_jobs(cfg)
        if not (P_['naive'] / 'evaluation.json').exists():
            ds = run_fast_stage(root, 'naive', jobs, naive_knowledge(root), led=led, client=mk('fast'), css=css, prep_extra=lambda c: (), prefit=lambda c: [],
                                sdir=P_['naive'])
            if ds['missing'] or ds['unknown_usage_blocking']:
                raise RuntimeError('naive stage incomplete: %s' % ds['missing'][:5])
            evaluate_bd(root, 'naive', jobs, lambda c: ['None'], led=led, css=css, sdir=P_['naive'])
        res = readout_naive(root)
        print('NAIVE_DONE', json.dumps({k: round(v['three_domain'], 3) for k, v in res['main'].items()}), flush=True)
        return res


def smoke_naive(root: Path = ROOT) -> dict:
    """Scripted (no LLM client object, no fit): naive payload carries no downstream evidence, card contract, and the stage-dir plumbing end to end
    on one exposed Test case whose committed material is already fitted (one scoring subprocess)."""
    sm = root / 'naive_smoke'
    if sm.exists():
        shutil.rmtree(sm)
    sm.mkdir(parents=True)
    checks = []

    def check(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})
        print('SMOKE_NAIVE', 'OK ' if ok else 'FAIL', name, info if not ok else '', flush=True)

    split = context.read_json(paths(root)['split'])
    css = cases_of(split)
    cfg = context.read_json(paths(root)['frozen'])
    pl = naive_payload(root, 'D01', css)
    txt = json.dumps(pl, ensure_ascii=False)
    b = len(json.dumps([{'role': 'system', 'content': naive_system()}, {'role': 'user', 'content': txt}], ensure_ascii=False).encode('utf-8'))
    check('naive_payload_no_downstream_evidence', not any(k in txt for k in ('G_vs_None', 'fast_trajectory', 'scripted_records', 'fast_outcome', 'e_by_seed', 'normalized_mse',
                                                                             'commit_reason', 'P_NoMixRecipe_card')), bytes=b)
    check('naive_payload_observations_only', len(pl['development_observations']['cases']) == 8 and set(pl['development_observations']['cases'][0]) == {'case_ref', 'observations'})
    check('naive_system_has_no_learned_answer', not re.search(r'shock|censor|NoMix|preset', NAIVE_EVIDENCE + NAIVE_GUIDE, re.I))
    legal = pl['legal_evidence_refs']
    good = {'decision': 'PROPOSE', 'card': {'research_mode': 'observe then construct', 'workflow': 'Read the batch summary and build one plan that matches the observed structure; inspect it; commit.',
                                             'principles': None, 'applicability_summary': 'any batch', 'evidence_refs': legal[:2], 'rationale': 'task knowledge'}}
    out = parse_naive(good, domain='D01', legal_refs=legal)
    check('naive_card_parse_ok', out['decision'] == 'PROPOSE' and out['skill'].skill_id == 'NAIVE-D01')
    bad = json.loads(json.dumps(good))
    bad['card']['workflow'] = 'Use C_A.'
    try:
        parse_naive(bad, domain='D01', legal_refs=legal)
        ok = False
    except ValueError:
        ok = True
    check('naive_card_rejects_feedback_step', ok)
    nj = naive_jobs(cfg)
    check('naive_jobs_40', len(nj) == 40 and len(set(nj)) == 40)
    # plumbing on one Test case: a scripted commit of an already fitted material -> sdir override, freeze, evaluation (0 fits), E equals the main evaluation
    case = 'D03_Q_Q1_G1'
    led = SafeLedger(sm / 'budget.json', max_fit_attempts=0, max_llm_requests=5, max_llm_tokens=10 ** 6, max_wall_s=3600, max_retries=0)
    DP.set_pools(2, 1)

    class FC:
        def call(self, role, unit, payload, system, *, max_tokens, meta):
            return json.dumps({'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'P_NoMixRecipe', 'reason': 'smoke'}}]}), {'request': 0, 'prompt_tokens': 100, 'completion_tokens': 10, 'message_bytes': 1000}
    old = DP.proxy_reachable
    DP.proxy_reachable = lambda: True
    try:
        jobs = [(case, NAIVE_ARM)]
        ds = run_fast_stage(root, 'naive', jobs, lambda c, a: knowledge_of_card(out['skill']), led=led, client=FC(), css=css, prep_extra=lambda c: (), prefit=lambda c: [], sdir=sm)
        ev = evaluate_bd(root, 'naive', jobs, lambda c: ['None'], led=led, css=css, sdir=sm)
    finally:
        DP.proxy_reachable = old
    evt = context.read_json(paths_bd(root)['test'] / 'evaluation.json')
    check('plumbing_complete_no_fit', ds['complete'] == 1 and led.s['fit_attempts'] == 0, fits=led.s['fit_attempts'])
    check('plumbing_E_equals_main', all(ev['e_by_case_phys_seed'][case][p][str(s)] == evt['e_by_case_phys_seed'][case][p][str(s)] for p in ('None', 'P_NoMixRecipe') for s in SEEDS))
    check('main_stage_untouched', not (paths_bd(root)['test'] / 'branches' / ('%s__%s' % (case, NAIVE_ARM))).exists())
    res = {'written_local': now(), 'checks': checks, 'passed': sum(c['ok'] for c in checks), 'total': len(checks)}
    context.write_json(sm / 'smoke_naive_result.json', res)
    print('SMOKE_NAIVE_DONE', res['passed'], '/', res['total'], flush=True)
    return res


# ============================================================================= status
def status(root: Path = ROOT) -> dict:
    P_ = paths(root)
    out = {'local': now()}
    if P_['ledger'].exists():
        led = context.read_json(P_['ledger'])
        out['ledger'] = {k: v for k, v in led.items() if k != 'events'}
    bd = P_['stage_a'] / 'branches'
    if bd.exists():
        rs = {}
        for b in sorted(bd.iterdir()):
            p = b / 'branch_result.json'
            rs[b.name] = context.read_json(p)['status'] if p.exists() else 'RUNNING'
        out['branches'] = rs
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--stage-a', action='store_true')
    ap.add_argument('--evaluate-a', action='store_true')
    ap.add_argument('--report-a', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--smoke-bd', action='store_true')
    ap.add_argument('--run-bd', action='store_true')
    ap.add_argument('--readout-bd', action='store_true')
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--worker-prepare', nargs=4)
    ap.add_argument('--smoke-naive', action='store_true')
    ap.add_argument('--run-naive', action='store_true')
    ap.add_argument('--readout-naive', action='store_true')
    ap.add_argument('--numeric', type=int, default=CONCURRENCY['numeric'])
    ap.add_argument('--only', default=None)
    ap.add_argument('--worker-material', nargs=4)
    ap.add_argument('--worker-score', nargs='+')
    ap.add_argument('--worker-score-one', nargs=6)
    ap.add_argument('--worker-score-dir', nargs=6)
    a = ap.parse_args()
    if a.worker_material:
        return worker_material(*a.worker_material)
    if a.worker_prepare:
        return worker_prepare(*a.worker_prepare)
    if a.worker_score:
        r, sp, case, out, *physes = a.worker_score
        return worker_score(r, sp, case, out, physes)
    if a.worker_score_one:
        r, sp, case, out, phys, seed = a.worker_score_one
        return worker_score_one(r, sp, case, out, phys, int(seed))
    if a.worker_score_dir:
        r, sp, case, out, phys, seed = a.worker_score_dir
        return worker_score_one(r, sp, case, out, phys, int(seed))
    if a.smoke:
        smoke()
    if a.wiring:
        wiring()
    if a.stage_a:
        stage_a(numeric=a.numeric, only=a.only.split(',') if a.only else None)
    if a.evaluate_a:
        evaluate_stage_a(numeric=a.numeric)
    if a.report_a:
        print(report_a())
    if a.accept_unknown_usage:
        led = ledger_bd(ROOT)
        if rt.unknown_usage_blocks(led):
            led.s['unknown_usage_accepted'] = led.s['llm_tokens_unknown']
            led.event(kind='unknown_usage_accepted', operator='explicit --accept-unknown-usage')
    if a.smoke_bd:
        smoke_bd()
    if a.run_bd:
        run_bd(numeric=a.numeric)
    if a.readout_bd:
        res = readout_bd()
        (ROOT / 'tables.md').write_text(tables_bd(res), encoding='utf-8')
    if a.smoke_naive:
        smoke_naive()
    if a.run_naive:
        run_naive(numeric=a.numeric)
    if a.readout_naive:
        readout_naive()
    if a.status:
        status()


if __name__ == '__main__':
    main()
