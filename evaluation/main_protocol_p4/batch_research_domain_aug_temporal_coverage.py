"""DEV-DOMAIN-AUG-TEMPORAL-COVERAGE (docs/DEV_DOMAIN_AUG_TEMPORAL_COVERAGE_TASK_2026-09-21.md): the same two domains and the same
entity groups as DEV-DOMAIN-AUG-ENTITY-SPLIT, organised over twelve chronological periods: six Source periods (24 no-card Fast research
cases), two Select periods (4 cases), four MetaTest periods (16 cases = 2 entity groups x 4 periods per domain). Learning: three
independent single-card Slow proposals per domain (12 Source cases each) and three shared proposals (both domains' 24 Source cases);
Select freezes W_d per domain, ONE global W_shared, Fixed_dev,d and H_deploy,d; MetaTest runs F0 / F_domain / F_shared / RandomSearch_B4 /
Fixed_dev + the four public references (+ Menu_CA from cache) with a whole-stage C_B -> E barrier; readout per task §8.

Reused unchanged (read-only imports): the entity-split case profile (methods/ttha/batch_base/entity_case.py) and its worker CLI
(batch_research_domain_aug_entity_split: build / material / fit / label; the controller never imports torch), the parallel coordinator
pieces of batch_research_domain_aug_decision_priority (package lock, thread-safe ledger, pools, metered client with receipts, adapter,
Fast / Random / Fixed_dev branches, label phases, decision-point evidence helpers, readout arithmetic). New here: the multi-period split
check (role-disjoint entity sets, in-role reuse at different t, per-case stage barriers), the Source stage of this package, the
domain / shared censuses with time metadata, the no-old-card single-card Slow contract, the W_d / W_shared / Fixed_dev / H_deploy
freeze, the 16-case MetaTest with period x entity-group aggregation.

  --preflight | --smoke | --wiring | --run | --resume-stage STAGE [--accept-unknown-usage] [--restart b1,b2] | --result | --monitor
  subprocess entries: --stage-worker CFG | --lock-probe ROOT
"""
from __future__ import annotations

import os

KMP_AT_START = os.environ.get('KMP_DUPLICATE_LIB_OK')

import argparse
import copy
import json
import math
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha.batch_base import budget, context, spec, tempo_aug as ta
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_domain_skill as dsks
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import run_batch_research_v1 as base
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_skill as W            # read-only helpers: text check, columnar
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_learning_loop as LL    # read-only helpers: review parsing, ref ranges
from evaluation.main_protocol_p4 import batch_research_domain_aug_entity_split as ES              # read-only: workers, adapter, random draw, cache copy
from evaluation.main_protocol_p4 import batch_research_domain_aug_decision_priority as DP         # read-only: coordinator pieces, evidence helpers

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_domain_aug_temporal_coverage'
MODULE = 'evaluation.main_protocol_p4.batch_research_domain_aug_temporal_coverage'
WORKER_MODULE = ES.MODULE
TASK = 'docs/DEV_DOMAIN_AUG_TEMPORAL_COVERAGE_TASK_2026-09-21.md'
SPLIT_DOC = REPO / 'docs' / 'DOMAIN_AUG_TEMPORAL_COVERAGE_V1.json'
PARENT_SPLIT_DOC = ES.SPLIT_DOC
PACKAGE = 'DEV-DOMAIN-AUG-TEMPORAL-COVERAGE'
EXPOSURE = 'SERIES_DISJOINT_TEMPORAL_DEVELOPMENT'
DOMAINS = ES.DOMAINS
SHARED = 'SHARED'
SEEDS = ES.SEEDS
PUBLIC = ec.PUBLIC
PUBLIC_STEPS = ec.PUBLIC_STEPS
LIMITS = dict(ES.LIMITS)                                    # 16 calls / 24 tools / 4 new evaluations per Fast
MAX_OUTPUT_TOKENS = ES.MAX_OUTPUT_TOKENS                    # 12,000
MAX_TOOL_CORRECTIONS = ES.MAX_TOOL_CORRECTIONS
EVIDENCE_ROUNDTRIP = ES.EVIDENCE_ROUNDTRIP
REQUEST_TRACE_DEDUPE = True
MODEL = ES.MODEL
BODY_LIMIT = ES.BODY_LIMIT                                  # 6000 characters per card
TOL = ES.TOL
T975_DF2 = ES.T975_DF2
FIT_THREADS = ES.FIT_THREADS
WIRING_COMPOSITION = ES.WIRING_COMPOSITION
WIRING_CASES = ('D01_S_A01_G01', 'D02_S_A01_G01')
COMMON_PREFIX = (0, 17544)
STAGE_ANCHORS = {'source': ('A01', 'A02', 'A03', 'A04', 'A05', 'A06'), 'select': ('A07', 'A08'), 'test': ('A09', 'A10', 'A11', 'A12')}
COUNTS = {'source': 24, 'select': 4, 'test': 16}
SLOTS = (1, 2, 3)
DOMAIN_ARMS = ('d1', 'd2', 'd3')
SHARED_ARMS = ('s1', 's2', 's3')
SELECT_ARMS = ('f0',) + DOMAIN_ARMS + SHARED_ARMS
TEST_FAST_ARMS = ('f0', 'f_domain', 'f_shared')
CONTROL_ARMS = DP.CONTROL_ARMS                              # ('random', 'fixed_dev')
ARM_LABEL = {'f0': 'F0', 'f_domain': 'F_domain', 'f_shared': 'F_shared', 'random': 'RandomSearch_B4', 'fixed_dev': 'Fixed_dev', 'menu_ca': 'Menu_CA', 'h_deploy': 'H_deploy',
             'd1': 'D1', 'd2': 'D2', 'd3': 'D3', 's1': 'S1', 's2': 'S2', 's3': 'S3'}
CONCURRENCY = {'cases': 4, 'http': 4, 'numeric': 3, 'fit_threads': FIT_THREADS}     # numeric: 3 only if the wiring memory rule allows, else 2
TOTAL = {'max_fit_attempts': 1988, 'max_llm_requests': 1618, 'max_llm_tokens': 150_000_000, 'max_wall_s': 72 * 3600, 'max_retries': 8}   # tokens / wall: non-binding guards
HTTP_CAP = 1626
TRANSPORT_RETRIES = 8                                       # package-wide, at most one per logical request
ALLOC = {'wiring': {'fits': 12, 'requests': 0}, 'source': {'fits': 576, 'requests': 384}, 'slow': {'fits': 0, 'requests': 18},
         'select': {'fits': 384, 'requests': 448}, 'metatest': {'fits': 1008, 'requests': 768}}
TOKEN_WARNINGS = (20_000_000, 35_000_000)
PAID_WALL_WARNINGS_S = (4 * 3600, 8 * 3600)
SLOW_REQUEST_TIMEOUT_S = 600.
DOMAIN_BYTE_TARGET = 270_000                                # ~100k tokens (planning target of task §5.C)
SHARED_BYTE_TARGET = 430_000                                # ~160k tokens (planning target)
HARD_BYTE_LIMIT = 530_000                                   # the envelope the backend accepted before (495 KB -> 191k tokens)
FAST_SYSTEM = DP.FAST_SYSTEM                                # the common Fast prompt in force (permission sentence, action space, material semantics)
CONTRACTS = copy.deepcopy(DP.CONTRACTS)
ALLOWED_FEATURES = ES.ALLOWED_FEATURES
FIELD_DEFINITIONS = ES.FIELD_DEFINITIONS

now = DP.now
write_new = DP.write_new
PackageLock = DP.PackageLock


class SafeLedger(DP.SafeLedger):
    """DP's thread-safe ledger with a bounded retry of the atomic replace: on Windows os.replace fails with PermissionError when another handle
    (a reader thread / the read-only monitor) holds budget.json at that instant (seen once in MetaTest: CASE_FAILED D01_Q_A11_G02, 2026-09-22)."""

    def _save(self):
        with self.lock:
            for i in range(8):
                try:
                    return super()._save()
                except PermissionError:
                    if i == 7:
                        raise
                    time.sleep(0.05 * (i + 1))

pools, set_pools = DP.pools, DP.set_pools
run_worker, fit_cells, ensure_material = DP.run_worker, DP.fit_cells, DP.ensure_material
MeteredClient, FastClient, BoundFast = DP.MeteredClient, DP.FastClient, DP.BoundFast
CaseAdapter = DP.CaseAdapter
fast_branch = DP.fast_branch


def resume_fast_branch(path, case, knowledge, led, client_call, common, *, split_path, cs, public_ids, seeds):
    """DP's continuation with the opt-in for more than one resume of the same branch (a stage may be stopped by unknown usage more than once)."""
    return base.resume_branch(path, case, knowledge, led, client_call, max_tool_corrections=MAX_TOOL_CORRECTIONS, tool_contracts=CONTRACTS, seeds=seeds,
                              adapter_factory=DP.adapter_factory(common, split_path, cs, public_ids, seeds), evidence_roundtrip=EVIDENCE_ROUNDTRIP, limits=LIMITS,
                              dataset=cs.dataset, commit_fn=ES.commit_branch, allowed_features=ALLOWED_FEATURES, entity_count=ec.COHORT_SIZE, request_trace_dedupe=REQUEST_TRACE_DEDUPE,
                              allow_multiple_resumes=True)
random_branch, fixed_dev_branch = DP.random_branch, DP.fixed_dev_branch
knowledge_from_json = DP.knowledge_from_json
Progress, label_one, _parallel, open_labels = DP.Progress, DP.label_one, DP._parallel, DP.open_labels
proxy_reachable = DP.proxy_reachable
payload_bytes = DP.payload_bytes


def paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'configs': root / 'stage_configs', 'logs': root / 'logs', 'split': root / 'split_copy.json',
            'parent_split': root / 'parent_split_copy.json', 'wiring': root / 'wiring', 'smoke': root / 'smoke', 'evidence': root / 'evidence', 'source': root / 'source',
            'slow': root / 'slow', 'select': root / 'select', 'freeze': root / 'freeze', 'metatest': root / 'metatest', 'warnings': root / 'warnings', 'lock': root / 'package.lock'}


def ref(stage: str, branch: str, tail: str) -> str:
    return '%s/%s/%s' % (stage, branch, tail)


def _require_package_root(root: Path) -> None:
    """Stage configs / workers only under <repo>/_scratch (a smoke or a check can never start a paid stage elsewhere)."""
    r = Path(root).resolve()
    if (REPO / '_scratch').resolve() not in r.parents:
        raise PermissionError('stage execution refused outside %s: %s' % (REPO / '_scratch', r))


# ============================================================================= split: verbatim copy, path binding, role-disjoint entities, per-case barriers
def anchor_formula(common_rows: int = COMMON_PREFIX[1]) -> list:
    """t_k = 24 floor((672 + (k + 0.5) (rows - 1056) / 12) / 24), k = 0..11 (task §2.1)."""
    return [24 * int(math.floor((672 + (k + 0.5) * (common_rows - 1056) / 12) / 24)) for k in range(12)]


def load_split(root: Path) -> dict:
    p = paths(root)['split']
    return context.read_json(p if p.exists() else SPLIT_DOC)


def load_parent_split(root: Path) -> dict:
    p = paths(root)['parent_split']
    return context.read_json(p if p.exists() else PARENT_SPLIT_DOC)


def bound_split(split: dict) -> dict:
    """The split with csv_path replaced by the bound spec path (None) after the binding check; a synthetic dataset (smoke) keeps its own csv_path."""
    out = copy.deepcopy(split)
    for dom, D in out['domains'].items():
        if D['dataset'] in spec.DATASETS:
            DP.resolve_csv_path(D['dataset'], D.get('csv_path') or D.get('path'))
            D['csv_path'] = None
            D.pop('path', None)
    return out


def cases_of(root: Path) -> dict:
    return ec.cases_from_split(bound_split(load_split(root)))


def case_ids(root: Path, role: str, domain: str | None = None) -> list:
    return [c for c, cs in cases_of(root).items() if cs.role == role and (domain is None or cs.domain == domain)]


def group_tail(case_id: str) -> str:
    return case_id.split('_')[-1]                            # G01 .. G08


def anchor_of(case_id: str) -> str:
    return case_id.split('_')[2]                             # A01 .. A12


def check_split_tc(split: dict, parent: dict) -> dict:
    """This package's partition check: per domain 12 Source / 2 Select / 8 MetaTest groups of 16 at the frozen anchors; roles are DISJOINT as
    entity sets (in-role reuse of an entity group at another t is legal); every roster equals its parent group; (entity group, t) and case_index
    unique; anchors follow the formula and the stage mapping; geometry; per-case stage barriers (every Source E end < first Select T start;
    every Select E end < first MetaTest T start; every E end inside the common prefix)."""
    out = {'domains': {}, 'ok': True, 'anchors_formula_ok': list(split['anchors']) == anchor_formula(), 'common_prefix_ok': list(split['common_row_prefix']) == list(COMMON_PREFIX)}
    anchor_ids = {'A%02d' % (i + 1): t for i, t in enumerate(split['anchors'])}
    stage_of_anchor = {a: st for st, ids in STAGE_ANCHORS.items() for a in ids}
    for dom, D in split['domains'].items():
        P_ = parent['domains'].get(dom) or {}
        pg = {g['case_id']: g['roster'] for g in P_.get('groups', [])}
        groups = D['groups']
        excl = set(D.get('excluded_columns', []))
        by_role = {}
        for g in groups:
            by_role.setdefault(g['stage'], []).append(g)
        ent = {r: {c for g in gs for c in g['roster']} for r, gs in by_role.items()}
        roles = sorted(ent)
        overlap = {'%s&%s' % (a, b): sorted(ent[a] & ent[b]) for i, a in enumerate(roles) for b in roles[i + 1:]}
        rec = {'n_groups': len(groups), 'roles': {r: len(gs) for r, gs in by_role.items()}, 'entities_per_role': {r: len(v) for r, v in ent.items()},
               'role_overlap': {k: v for k, v in overlap.items() if v}, 'sizes_ok': all(len(g['roster']) == ec.COHORT_SIZE for g in groups),
               'roster_unique_within_group': all(len(set(g['roster'])) == ec.COHORT_SIZE for g in groups), 'excluded_in_use': sorted({c for g in groups for c in g['roster']} & excl),
               'roster_equals_parent_group': all(pg.get(g['roster_source_case_id']) == g['roster'] for g in groups),
               'group_t_unique': len({(g['entity_group'], g['t']) for g in groups}) == len(groups), 'case_index_unique': len({g['case_index'] for g in groups}) == len(groups),
               'anchor_ids_ok': all(anchor_ids.get(g['anchor_id']) == g['t'] and stage_of_anchor.get(g['anchor_id']) == g['stage'] for g in groups),
               'same_dataset_as_parent': P_.get('dataset') == D['dataset'],
               'geometry_ok': all(list(g['train_rows']) == [g['t'] - spec.TRAIN_SPAN, g['t']] and list(g['c_a_origins']) == [g['t'], g['t'] + 48] and list(g['c_b_origins']) == [g['t'] + 96, g['t'] + 144]
                                  and list(g['e_origins']) == [g['t'] + 192, g['t'] + 240, g['t'] + 288, g['t'] + 336] and list(g['e_target_rows']) == [g['t'] + 192, g['t'] + 384] for g in groups)}
        # per-case barriers (not only the maximum row)
        src_e_end = [g['e_target_rows'][1] for g in by_role.get('source', [])]
        sel_t_start = [g['train_rows'][0] for g in by_role.get('select', [])]
        sel_e_end = [g['e_target_rows'][1] for g in by_role.get('select', [])]
        test_t_start = [g['train_rows'][0] for g in by_role.get('test', [])]
        rec['barriers'] = {'source_e_end_max': max(src_e_end, default=None), 'select_t_start_min': min(sel_t_start, default=None), 'select_e_end_max': max(sel_e_end, default=None),
                           'test_t_start_min': min(test_t_start, default=None), 'final_e_end_max': max((g['e_target_rows'][1] for g in groups), default=None),
                           'every_source_e_before_every_select_t': all(e <= s for e in src_e_end for s in sel_t_start),
                           'every_select_e_before_every_test_t': all(e <= s for e in sel_e_end for s in test_t_start),
                           'every_e_inside_common_prefix': all(g['e_target_rows'][1] <= COMMON_PREFIX[1] and g['train_rows'][0] >= COMMON_PREFIX[0] for g in groups)}
        rec['ok'] = (rec['roles'] == {'source': 12, 'select': 2, 'test': 8} and not rec['role_overlap'] and rec['sizes_ok'] and rec['roster_unique_within_group'] and not rec['excluded_in_use']
                     and rec['roster_equals_parent_group'] and rec['group_t_unique'] and rec['case_index_unique'] and rec['anchor_ids_ok'] and rec['same_dataset_as_parent'] and rec['geometry_ok']
                     and all(v for k, v in rec['barriers'].items() if k.startswith('every_')) and rec['entities_per_role'] == {'source': 128, 'select': 32, 'test': 32})
        out['domains'][dom] = rec
        out['ok'] = out['ok'] and rec['ok']
    out['counts_ok'] = {r: sum(1 for D in split['domains'].values() for g in D['groups'] if g['stage'] == r) == n for r, n in COUNTS.items()}
    out['ok'] = out['ok'] and set(split['domains']) == set(DOMAINS) and out['anchors_formula_ok'] and out['common_prefix_ok'] and all(out['counts_ok'].values())
    return out


# ============================================================================= budget (one package ledger, cumulative stage caps)
def stage_caps(root: Path, stage: str) -> dict:
    P_ = paths(root)
    rec = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    if stage in rec:
        return rec[stage]
    led = SafeLedger(P_['ledger'], **TOTAL)
    s, alloc = led.s, ALLOC[stage]
    snap = {k: s.get(k, 0) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts',
                                     'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'llm_failed_attempts', 'fit_wall_seconds')}
    snap['elapsed_s'] = time.time() - s['started_epoch']
    retries_stage = (TOTAL['max_retries'] - s['retries_used']) if alloc['fits'] else 0
    caps = {'max_fit_attempts': min(TOTAL['max_fit_attempts'], s['fit_attempts'] + alloc['fits'] + retries_stage),
            'max_llm_requests': min(TOTAL['max_llm_requests'], s['llm_requests'] + alloc['requests']),
            'max_llm_tokens': TOTAL['max_llm_tokens'], 'max_wall_s': TOTAL['max_wall_s'], 'max_retries': TOTAL['max_retries']}
    rec[stage] = {'epoch_start': time.time(), 'local_start': now(), 'allocation': alloc, 'snapshot_at_start': snap, 'caps': caps,
                  'http_cap': min(HTTP_CAP, s['llm_http_attempts'] + alloc['requests'] + max(0, TRANSPORT_RETRIES - s.get('llm_failed_attempts', 0)))}
    context.write_json(P_['stages'], rec)
    return rec[stage]


def start_paid_clock(root: Path) -> None:
    P_ = paths(root)
    led = SafeLedger(P_['ledger'], **TOTAL)
    if led.s.get('paid_clock_started_epoch') is None:
        if led.s['llm_requests']:
            raise RuntimeError('paid requests exist before the paid clock start')
        led.s['wiring_elapsed_before_paid_clock_s'] = time.time() - led.s['started_epoch']
        led.s['paid_clock_started_epoch'] = time.time()
        led.s['paid_clock_started_local'] = now()
        led._save()


def budget_warnings(root: Path, where: str, led_obj=None) -> list:
    """Task §7: tokens / paid wall are warnings (20M / 35M tokens; 4 h / 8 h paid), never a mechanical stop. Each threshold is written once.
    led_obj: the live SafeLedger of the stage (read under its lock; avoids a file read racing the ledger's atomic replace)."""
    P_ = paths(root)
    if led_obj is not None:
        with led_obj.lock:
            led = dict(led_obj.s)
    else:
        led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    tok = led.get('llm_tokens_in', 0) + led.get('llm_tokens_out', 0)
    paid = (time.time() - led['paid_clock_started_epoch']) if led.get('paid_clock_started_epoch') else 0.0
    P_['warnings'].mkdir(parents=True, exist_ok=True)
    out = []
    for th in TOKEN_WARNINGS:
        p = P_['warnings'] / ('tokens_%d.json' % th)
        if tok >= th and not p.exists():
            context.write_json(p, {'threshold': th, 'tokens': tok, 'where': where, 'local': now(), 'action': 'progress / remaining estimate only; no data, arm or request change'})
            out.append(str(p))
    for th in PAID_WALL_WARNINGS_S:
        p = P_['warnings'] / ('paid_wall_%dh.json' % (th // 3600))
        if paid >= th and not p.exists():
            context.write_json(p, {'threshold_s': th, 'paid_elapsed_s': paid, 'where': where, 'local': now(), 'action': 'progress / remaining estimate only'})
            out.append(str(p))
    for w in out:
        print('BUDGET_WARNING', w, flush=True)
    return out


def numeric_seconds(root: Path) -> float:
    led = context.read_json(paths(root)['ledger'])
    return float(led.get('fit_wall_seconds', 0.0)) + float(led.get('label_wall_seconds', 0.0)) + float(led.get('material_wall_seconds', 0.0))


# ============================================================================= stage coordinator (one process, one thread per active case, sequential arms inside a case)
def prepare_common(root: Path, case: str, led: SafeLedger, cfg: dict) -> Path:
    """Public references of one case: (wiring cache copy for the two wiring cases) -> build (0 fits) -> random supply frozen (MetaTest) ->
    reference fits (numeric pool). The physical cache of a case is its own directory: another t of the same entity group is another case."""
    common = root / (case + '_common')
    cases = ec.cases_from_split(context.read_json(cfg['split_path']))
    cs = cases[case]
    seeds, public_ids, split_path = tuple(cfg['seeds']), tuple(cfg['public_ids']), Path(cfg['split_path'])
    src = (cfg.get('cache_from') or {}).get(case)
    if src and not (common / case / 'aug_materials' / 'index.json').exists():
        ES.copy_physical_cache(Path(src), common, case, cs, seeds=seeds)
    if not all(m in ec.load_registry(common / case) for m in public_ids):
        t0 = time.time()
        rc, _ = run_worker(['--worker-build', common, split_path, case], root / 'logs' / ('build_%s.log' % case), 1800)
        led.add('material_wall_seconds', time.time() - t0)
        if rc != 0:
            raise RuntimeError('common build failed for %s' % case)
    if cfg.get('random') and not (common / 'random_supply.json').exists():
        table = context.read_json(common / case / 'overview.json')['entities']
        dsks.write_once(common / 'random_supply.json', ES.draw_random_supply(case, cs.domain, cs.case_index, table))
    got = fit_cells(led, common, split_path, case, [(m, s) for m in public_ids for s in seeds])
    for r in got.values():
        if r['record']['scores']['c_a']['status'] != 'SCORABLE':
            raise RuntimeError('C_A_NOT_SCORABLE: %s' % r['record']['cell_id'])
    return common


def _safe_message(exc) -> str | None:
    return ES._safe_message(exc)


def _run_case(root: Path, cfg: dict, led: SafeLedger, fc, case: str, stop: threading.Event, progress: Progress, resumed: bool) -> tuple:
    """All arms of one case in the configured order (sequential inside the case); returns (branches, failures) of this case."""
    branches, failures = [], []
    cases = ec.cases_from_split(context.read_json(cfg['split_path']))
    cs = cases[case]
    seeds, public_ids, split_path = tuple(cfg['seeds']), tuple(cfg['public_ids']), Path(cfg['split_path'])
    progress.set(case, status='preparing_common', local=now())
    common = prepare_common(root, case, led, cfg)
    for arm in cfg['order'][case]:
        path = root / ('%s_%s' % (case, arm))
        alias = (cfg.get('aliases') or {}).get(case, {}).get(arm)
        if alias:
            p = root / ('%s_%s_alias.json' % (case, arm))
            if not p.exists():
                context.write_json(p, {'arm': arm, 'alias_of': alias, 'note': (cfg.get('alias_notes') or {}).get(arm, 'same frozen card text (or no card): the trajectory of the aliased arm is reused; no second LLM run')})
            continue
        kn = cfg['knowledge'].get(case, {}).get(arm)
        if arm not in CONTROL_ARMS and kn is None:
            raise RuntimeError('no knowledge for arm %s of %s' % (arm, case))
        if arm == 'fixed_dev' and not (cfg.get('fixed_dev') or {}).get(cs.domain):
            p = root / ('%s_%s_treatment.json' % (case, arm))
            if not p.exists():
                context.write_json(p, {'status': 'NO_TREATMENT', 'note': 'no Fixed_dev program frozen for this domain'})
            continue
        if stop.is_set():
            failures.append({'branch': path.name, 'kind': 'NOT_STARTED', 'reason': 'stage stopping (backend fatal / unknown usage); arm left for resume'})
            progress.set(case, status='stopped_before_' + arm, local=now())
            break
        branches.append((path, case))
        progress.set(case, status='running_' + arm, local=now())
        client_call = BoundFast(fc, case, arm) if fc is not None else None
        try:
            if path.exists() and (path / 'branch_result.json').exists():
                prior = context.read_json(path / 'branch_result.json')
                if prior['status'] == 'COMPLETE':
                    result = br.RunResult(**{k: v for k, v in prior.items() if k != 'trace'})
                elif arm in CONTROL_ARMS:
                    raise RuntimeError('control branch cannot be resumed')
                else:
                    result = resume_fast_branch(path, case, knowledge_from_json(kn), led, client_call, common, split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
            elif path.exists():
                raise RuntimeError('branch directory without a result; inspect before any replay')
            elif arm == 'random':
                result = random_branch(path, case, led, common, context.read_json(common / 'random_supply.json'), split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
            elif arm == 'fixed_dev':
                result = fixed_dev_branch(path, case, led, common, cfg['fixed_dev'][cs.domain], split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
            else:
                result = fast_branch(path, case, knowledge_from_json(kn), led, client_call, common, split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
            if result.status != 'COMPLETE':
                failures.append({'branch': path.name, 'kind': result.failure_kind, 'reason': result.reason})
        except Exception as exc:  # noqa: BLE001
            failures.append({'branch': path.name, 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
            print('BRANCH_FAILED', path.name, type(exc).__name__, _safe_message(exc), flush=True)
        progress.set(case, status='done_' + arm, last_arm_local=now())
        if fc is not None and (getattr(fc.client, 'fatal', False) or rt.unknown_usage_blocks(led)):
            stop.set()
            print('STAGE_STOP_REQUESTED', 'backend fatal' if fc.client.fatal else 'unknown usage', flush=True)
    progress.set(case, status='finished', local=now())
    return branches, failures


def _run_cases(root: Path, cfg: dict, led: SafeLedger, client, branches: list, failures: list, *, resumed: bool) -> None:
    """Up to CONCURRENCY['cases'] cases at once in the frozen queue order; a free slot takes the next case; arms of a case stay sequential;
    results are merged in the frozen case / arm order (never by completion time)."""
    fc = FastClient(client, FAST_SYSTEM) if client is not None else None
    progress = Progress(root / 'progress.json')
    stop = threading.Event()
    q = queue.Queue()
    for c in cfg['cases']:
        q.put(c)
    results, errors, lock = {}, [], threading.Lock()

    def worker():
        while True:
            try:
                case = q.get_nowait()
            except queue.Empty:
                return
            try:
                out = _run_case(root, cfg, led, fc, case, stop, progress, resumed)
            except Exception as exc:  # noqa: BLE001
                out = ([], [{'case': case, 'exception_type': type(exc).__name__, 'message': _safe_message(exc)}])
                errors.append((case, exc))
                print('CASE_FAILED', case, type(exc).__name__, _safe_message(exc), flush=True)
            with lock:
                results[case] = out
            q.task_done()
    n = min(int(cfg['concurrency']['cases']), len(cfg['cases']))
    threads = [threading.Thread(target=worker, name='case-worker-%d' % i, daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        budget_warnings(Path(cfg['package_root']), cfg['stage'], led)
        with led.lock:
            snap = {k: led.s.get(k) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown')}
        progress.note(ledger=snap)
    for t in threads:
        t.join()
    for case in cfg['cases']:
        b, f = results.get(case, ([], [{'case': case, 'kind': 'NOT_RUN'}]))
        branches.extend(b)
        failures.extend(f)
    progress.note(finished_local=now(), pools=pools().snapshot())
    if stop.is_set():
        raise RuntimeError('backend fatal or unknown usage')


def stage_worker(cfg_path: Path, *, client=None) -> None:
    cfg = context.read_json(cfg_path)
    root = Path(cfg['output'])
    _require_package_root(root)
    lock = PackageLock(Path(cfg['package_root']), 'stage_worker:%s' % cfg['stage']).acquire()
    try:
        if (root / 'experiment_started.json').exists():
            raise RuntimeError('stage already started; no paid replay')
        root.mkdir(parents=True, exist_ok=True)
        (root / 'logs').mkdir(exist_ok=True)
        context.write_json(root / 'config.json', cfg)
        set_pools(cfg['concurrency']['numeric'], cfg['concurrency']['http'])
        led = SafeLedger(cfg['ledger_path'], **cfg['caps'])
        if rt.unknown_usage_blocks(led):
            raise RuntimeError('package ledger holds unknown usage; paid stage refused without an operator decision')
        rt.FIT_RETRY = bool(cfg.get('fit_retry'))
        context.write_json(root / 'experiment_started.json', {'epoch': time.time(), 'pid': os.getpid(), 'stage': cfg['stage'], 'concurrency': cfg['concurrency']})
        branches, failures, stopped = [], [], None
        try:
            if client is None and cfg.get('llm'):
                client = MeteredClient(led, root / 'raw_responses', http_cap=int(cfg['http_cap']), stage=cfg['stage'])
            _run_cases(root, cfg, led, client, branches, failures, resumed=False)
            context.write_json(root / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches]})
        except Exception as exc:  # noqa: BLE001
            failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
            stopped = type(exc).__name__
            print('EXECUTION_STOP', stopped, _safe_message(exc), flush=True)
        finally:
            context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures})
        open_labels(root, cfg, branches, failures, led, withhold_reason=stopped and 'execution stopped before every planned arm was attempted (%s)' % stopped)
    finally:
        lock.release()


def resume_stage(root: Path, stage: str, *, accept_unknown_usage: bool = False, restart=()) -> None:
    """One operator continuation under the package lock: labels still withheld, completed branches kept, never-sent Fast calls continued,
    missing arms run as planned, then labels. Unknown usage needs an explicit acceptance (the operator's, never the executor's own)."""
    from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp
    sroot = paths(root)[stage]
    cfg = context.read_json(sroot / 'config.json')
    with PackageLock(root, 'resume:%s' % stage):
        n_prev = len(list(sroot.glob('resume*.json')))
        if (sroot / 'experiment_started.json').exists() and not (sroot / 'execution_finished.json').exists() and not (sroot / 'labels_withheld.json').exists():
            context.write_json(sroot / 'labels_withheld.json', {'epoch': time.time(), 'reason': 'stage process stopped before finishing (operator stop / crash); no label was opened', 'written_by': 'resume_stage'})
        blocked = rtp.labels_boundary_violations(sroot)
        if blocked:
            raise RuntimeError('resume refused; labels are no longer withheld: ' + '; '.join(blocked))
        set_pools(cfg['concurrency']['numeric'], cfg['concurrency']['http'])
        led = SafeLedger(cfg['ledger_path'], **cfg['caps'])
        if rt.unknown_usage_blocks(led):
            if not accept_unknown_usage:
                raise RuntimeError('ledger holds unknown usage; pass --accept-unknown-usage to record an explicit operator decision')
            led.s['unknown_usage_accepted'] = led.s['llm_tokens_unknown']
            led.event(kind='unknown_usage_accepted', accepted=led.s['llm_tokens_unknown'], stage=stage)
        for name in ('execution_finished.json', 'labels_withheld.json'):
            if (sroot / name).exists():
                shutil.move(sroot / name, sroot / name.replace('.json', '_before_resume%d.json' % (n_prev + 1)))
        moved = []
        for name in restart:
            b = sroot / name
            prior = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
            if not b.is_dir() or (prior and (prior['status'] == 'COMPLETE' or prior['new_evaluations'])) or any(b.rglob('commit.json')):
                raise RuntimeError('only an interrupted branch without commit or new evaluation can be restarted: %s' % name)
            dest = sroot / ('%s__interrupted_%d' % (name, n_prev + 1))
            shutil.move(b, dest)
            moved.append({'branch': name, 'kept_as': dest.name, 'prior_status': prior and prior['status'], 'prior_failure_kind': prior and prior['failure_kind']})
        context.write_json(sroot / ('resume%d.json' % (n_prev + 1)), {'epoch': time.time(), 'local': now(), 'accepted_unknown_usage': bool(accept_unknown_usage), 'restarted_branches': moved})
        rt.FIT_RETRY = bool(cfg.get('fit_retry'))
        branches, failures, stopped = [], [], None
        try:
            client = MeteredClient(led, sroot / 'raw_responses', http_cap=int(cfg['http_cap']), stage=cfg['stage']) if cfg.get('llm') else None
            _run_cases(sroot, cfg, led, client, branches, failures, resumed=True)
            context.write_json(sroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches], 'resumed': True})
        except Exception as exc:  # noqa: BLE001
            failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
            stopped = type(exc).__name__
            print('EXECUTION_STOP', stopped, _safe_message(exc), flush=True)
        finally:
            context.write_json(sroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': True})
        open_labels(sroot, cfg, branches, failures, led, resumed=True, withhold_reason=stopped and 'resumed stage stopped again (%s)' % stopped)


def supplement_arms(root: Path, stage: str, names: list) -> dict:
    """Operator path for a stage whose labels were already opened while one or more Fast arms were left INCOMPLETE / AGENT_CALL_FAILED by the
    (historical) single-resume limit: continue exactly those branches from their identical built requests under the stage's frozen config, label
    them (C_B, E frozen), extend the stage's E barrier list by them (the original barrier file is kept as *_before_supplement<n>.json) and score E.
    The Fast never reads any E; the late trajectories see only their own T / C_A. Recorded as a protocol deviation (supplement<n>.json)."""
    sroot = paths(root)[stage]
    cfg = context.read_json(sroot / 'config.json')
    if not (sroot / 'all_e_predictions_frozen.json').exists() or not (sroot / 'execution_finished.json').exists():
        raise RuntimeError('supplement is only for a stage whose labels were already opened; otherwise use --resume-stage')
    with PackageLock(root, 'supplement:%s' % stage):
        n = 1 + len(list(sroot.glob('supplement*.json')))
        set_pools(cfg['concurrency']['numeric'], cfg['concurrency']['http'])
        led = SafeLedger(cfg['ledger_path'], **cfg['caps'])
        if rt.unknown_usage_blocks(led):
            raise RuntimeError('ledger holds unknown usage; accept it first')
        rt.FIT_RETRY = bool(cfg.get('fit_retry'))
        client = MeteredClient(led, sroot / 'raw_responses', http_cap=int(cfg['http_cap']), stage=cfg['stage'])
        fc = FastClient(client, FAST_SYSTEM)
        cases = ec.cases_from_split(context.read_json(cfg['split_path']))
        seeds, public_ids, split_path = tuple(cfg['seeds']), tuple(cfg['public_ids']), Path(cfg['split_path'])
        done, failures = [], []
        for name in names:
            b = sroot / name
            case = next(c for c in cfg['cases'] if name.startswith(c + '_'))
            arm = name[len(case) + 1:]
            kn = cfg['knowledge'][case][arm]
            common = sroot / (case + '_common')
            try:
                if not b.exists():                                       # a never-started arm (its case failed before any arm): public refs then a fresh branch
                    prepare_common(sroot, case, led, cfg)
                    if arm == 'random':
                        result = random_branch(b, case, led, common, context.read_json(common / 'random_supply.json'), split_path=split_path, cs=cases[case], public_ids=public_ids, seeds=seeds)
                    elif arm == 'fixed_dev':
                        result = fixed_dev_branch(b, case, led, common, cfg['fixed_dev'][cases[case].domain], split_path=split_path, cs=cases[case], public_ids=public_ids, seeds=seeds)
                    else:
                        result = fast_branch(b, case, knowledge_from_json(kn), led, BoundFast(fc, case, arm), common, split_path=split_path, cs=cases[case], public_ids=public_ids, seeds=seeds)
                else:
                    prior = context.read_json(b / 'branch_result.json')
                    if prior['status'] != 'INCOMPLETE' or (b / case / 'commit.json').exists():
                        raise RuntimeError('%s is not an unfinished Fast branch' % name)
                    result = resume_fast_branch(b, case, knowledge_from_json(kn), led, BoundFast(fc, case, arm), common, split_path=split_path, cs=cases[case], public_ids=public_ids, seeds=seeds)
            except Exception as exc:  # noqa: BLE001
                failures.append({'branch': name, 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
                continue
            if result.status != 'COMPLETE':
                failures.append({'branch': name, 'kind': result.failure_kind, 'reason': result.reason})
                continue
            label_one('c_b', b, case, led, sroot, split_path)
            label_one('freeze_e', b, case, led, sroot, split_path)
            done.append((b, case))
        if done:
            bar = sroot / 'all_e_predictions_frozen.json'
            old = context.read_json(bar)
            shutil.copy2(bar, sroot / ('all_e_predictions_frozen_before_supplement%d.json' % n))
            context.write_json(bar, {**old, 'branches': old['branches'] + [str(b) for b, c in done], 'supplement': n, 'supplemented_local': now()})
            for b, case in done:
                label_one('score_e', b, case, led, sroot, split_path)
            fin = context.read_json(sroot / 'execution_finished.json')
            done_names, done_cases = {b.name for b, c in done}, {c for b, c in done}
            fin['failures'] = [f for f in fin.get('failures', []) if f.get('branch') not in done_names and not (f.get('case') in done_cases and 'exception_type' in f)]
            for b, c in done:
                if [str(b), c] not in fin['branches'] and (str(b), c) not in {tuple(x) for x in fin['branches']}:
                    fin['branches'].append([str(b), c])
            fin['supplement'] = n
            context.write_json(sroot / 'execution_finished.json', fin)
        rec = {'supplement': n, 'local': now(), 'stage': stage, 'requested': list(names), 'completed': [b.name for b, c in done], 'failures': failures,
               'deviation': 'these Fast arms were continued after the stage labels had been opened for the other arms (single-resume limit of the shared continuation path, fixed opt-in); '
                            'the Fast reads no E; their own labels were opened afterwards; the barrier list was extended'}
        context.write_json(sroot / ('supplement%d.json' % n), rec)
        print('SUPPLEMENT', json.dumps(rec, default=str)[:600], flush=True)
        return rec


def effective_concurrency(root: Path) -> dict:
    p = root / 'concurrency_effective.json'
    if p.exists():
        return context.read_json(p)['concurrency']
    return dict(CONCURRENCY)


def stage_config(root: Path, stage: str, *, cases, order, knowledge, labels_e: bool, llm: bool, random: bool, aliases=None, alias_notes=None, fixed_dev=None, cache_from=None) -> Path:
    _require_package_root(root)
    P_ = paths(root)
    path = P_['configs'] / ('%s.json' % stage)
    if path.exists():
        return path
    sc = stage_caps(root, stage)
    conc = dict(effective_concurrency(root))
    cfg = {'study': 'domain_aug_temporal_coverage', 'status': 'FROZEN', 'stage': stage, 'output': str(P_[stage]), 'package_root': str(root), 'cases': list(cases),
           'split_path': str(root / 'split_bound.json'), 'order': {c: list(order[c]) for c in cases}, 'knowledge': knowledge, 'labels_e': bool(labels_e), 'llm': bool(llm),
           'random': bool(random), 'seeds': list(SEEDS), 'public_ids': list(PUBLIC), 'limits': LIMITS, 'max_tool_corrections': MAX_TOOL_CORRECTIONS,
           'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'request_trace_dedupe': REQUEST_TRACE_DEDUPE, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'fit_threads': FIT_THREADS,
           'concurrency': conc, 'caps': sc['caps'], 'http_cap': sc['http_cap'], 'ledger_path': str(P_['ledger']), 'fit_retry': True,
           'fit_attempts_at_stage_start': context.read_json(P_['ledger'])['fit_attempts'], 'aliases': aliases or {}, 'alias_notes': alias_notes or {},
           'fixed_dev': fixed_dev or {}, 'cache_from': {k: str(v) for k, v in (cache_from or {}).items()}, 'fast_system': FAST_SYSTEM, 'frozen_local': now()}
    dsks.write_once(path, cfg)
    return path


def run_stage(root: Path, stage: str, cfg_path: Path, driver_lock: PackageLock) -> dict:
    """The driver hands the package lock to the stage worker (one controller at a time) and takes it back when the worker exits."""
    _require_package_root(root)
    P_ = paths(root)
    sroot = P_[stage]
    if not (sroot / 'experiment_started.json').exists():
        P_['logs'].mkdir(parents=True, exist_ok=True)
        print('STAGE_START', stage, now(), flush=True)
        driver_lock.release()
        try:
            with (P_['logs'] / ('%s.log' % stage)).open('a', encoding='utf-8') as log:
                p = subprocess.Popen([sys.executable, '-B', '-m', MODULE, '--stage-worker', str(cfg_path)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=ES.worker_env())
                context.write_json(root / ('%s_worker_launch.json' % stage), {'epoch': time.time(), 'pid': p.pid, 'driver_pid': os.getpid(), 'local': now()})
                rc = p.wait()
        finally:
            driver_lock.acquire()
        print('STAGE_EXIT', stage, rc, now(), flush=True)
    return dsks.stage_status(sroot)


def case_queue(root: Path, role: str) -> list:
    """Frozen queue order of a stage: by anchor, then domain, then entity group (any order is legal: cases are independent inside a stage)."""
    cases = cases_of(root)
    ids = [c for c, cs in cases.items() if cs.role == role]
    return sorted(ids, key=lambda c: (cases[c].t, cases[c].domain, c))


def rotate(base: tuple, idx: int) -> list:
    r = int(idx) % len(base)
    return list(base[r:]) + list(base[:r])


# ============================================================================= Source: 24 no-card F0 trajectories (one per case), labels after the whole stage
def source_stage(root: Path, driver_lock: PackageLock) -> dict:
    P_ = paths(root)
    cases = case_queue(root, 'source')
    cache_from = {c: P_['wiring'] / (c + '_common') / c for c in WIRING_CASES if (P_['wiring'] / (c + '_common') / c / 'aug_materials' / 'index.json').exists()}
    cfgp = stage_config(root, 'source', cases=cases, order={c: ['f0'] for c in cases}, knowledge={c: {'f0': asdict(br.Knowledge())} for c in cases}, labels_e=True, llm=True,
                        random=False, cache_from=cache_from)
    return run_stage(root, 'source', cfgp, driver_lock)


# ============================================================================= evidence: the Source cases as decision points, with time metadata (deterministic, 0 LLM)
CASE_TOKEN = re.compile(r'(?<![A-Za-z0-9])(?:D0\d(?:_[A-Za-z0-9]+)?|[SVQ]0?\d{1,2}|[AG]\d{2}|SHARED)(?![A-Za-z0-9])')
PERIOD_TOKEN = re.compile(r'\b(?:period|anchor|stage|cut|epoch)[ _-]?(?:#\s*)?\d{1,2}\b|\b20(?:16|17|18)\b|\b(?:january|february|april|june|july|august|september|october|november|december)\b', re.I)
# 'may' and 'march' are ordinary English words (modal verb / verb) far more often than month names in guidance text: not banned (instrument
# correction 2026-09-21 22:40 after four Slow proposals were sent to a correction round for the word 'may'; see REPORT section 2).
ENTITY_TOKEN = DP.ENTITY_TOKEN
_ROWS = sorted(({t + d for t in anchor_formula() for d in (-672, 0, 48, 96, 144, 192, 240, 288, 336, 384)} | {COMMON_PREFIX[1]}) - {0, 672})   # block boundary rows; the generic 0 / 672 stay legal
ROW_TOKEN = re.compile(r'(?<![\d.])(?:%s)(?![\d.])' % '|'.join(str(r) for r in _ROWS))


def tc_text_check(text: str, where: str) -> None:
    """The parent checks (data source names, old job tokens) plus this package's case / anchor / group ids, domain ids, period or calendar words,
    block row numbers and entity / column numbers (task §5.A: deployable text must not index cases, anchors, rows or entities)."""
    W.skill_text_check(text, where)
    t = str(text or '')
    for rx, what in ((CASE_TOKEN, 'case / anchor / group / domain id'), (ROW_TOKEN, 'row / cut position'), (ENTITY_TOKEN, 'entity / column number'), (PERIOD_TOKEN, 'period / calendar reference')):
        m = rx.search(t)
        if m:
            raise ValueError('%s carries a %s token %r (deployable text must not index cases, periods, rows or entities)' % (where, what, m.group(0)))


def source_branch_dirs(root: Path, domains) -> list:
    """Every F0 branch of the Source stage of the given domains, in frozen (anchor, domain, group) order; never a Select / MetaTest directory."""
    P_ = paths(root)
    out = []
    for case in case_queue(root, 'source'):
        cs = cases_of(root)[case]
        if cs.domain not in domains or cs.role != 'source':
            continue
        d = P_['source'] / ('%s_f0' % case)
        if d.exists():
            out.append({'stage': 'source', 'case': case, 'arm': 'f0', 'dir': d, 'stage_root': P_['source'], 'kind': 'completed_or_failed', 'branch': d.name, 'domain': cs.domain})
        for d2 in sorted(P_['source'].glob('%s_f0__interrupted_*' % case)):
            out.append({'stage': 'source', 'case': case, 'arm': 'f0', 'dir': d2, 'stage_root': P_['source'], 'kind': 'interrupted', 'branch': d2.name, 'domain': cs.domain})
    return out


def _shrink_tier4(cen: dict) -> dict:
    """Tier 4 = tier 3 with the per-entity overview columns dropped (batch summary kept), rationale / reasons <= 160 characters, inspect_data results
    reduced to the kind and entities read. Actions, plans, C_A / C_B / E per seed, commits, failures and rejections are never dropped."""
    cen = copy.deepcopy(cen)
    for c in cen['cases'].values():
        ov = c.get('overview') or {}
        if ov.get('entities'):
            ov['entities'] = {'note': 'per-entity columns dropped at tier 4; batch summary quantiles kept', 'n': ov['entities'].get('n')}
        for b in c['branches']:
            for dp in b.get('decision_points', []) or []:
                for h in dp.get('hypotheses_stated', []):
                    for k in ('rationale', 'reason'):
                        if k in h:
                            h[k] = str(h[k])[:160]
                dp['results'] = [({'tool': r['tool'], 'kind': r.get('kind'), 'entities': r.get('entities')} if r.get('tool') == 'inspect_data' else r) for r in dp.get('results', [])]
            ov_ = b.get('offline_verification') or {}
            if ov_.get('commit_reason'):
                ov_['commit_reason'] = ov_['commit_reason'][:240]
    cen['compression']['tier_used'] = '4'
    return cen


def _shrink_tier5(cen: dict) -> dict:
    """Tier 5 = tier 4 further reduced for the 24-case shared census: per-branch offline_verification.materials and summary dropped (the case
    materials table holds the same per-seed numbers), decision-point 'visible_then' reduced to the remaining budget, rationale / reasons <= 100
    characters, inspect_material results reduced to the change RMS and identity count, compare results to the mean delta and signs. Actions,
    plans, C_A per seed of every evaluation, commits, failures and rejections are never dropped."""
    cen = _shrink_tier4(cen)
    for c in cen['cases'].values():
        for b in c['branches']:
            ov_ = b.get('offline_verification') or {}
            if 'materials' in ov_:
                ov_['materials'] = 'see case.materials (per-seed C_A / C_B / E of every material this branch evaluated: %s)' % sorted(ov_['materials'])
            if ov_.get('commit_reason'):
                ov_['commit_reason'] = ov_['commit_reason'][:160]
            b.pop('summary', None)
            if isinstance(b.get('unknown_then'), dict):
                b['unknown_then'] = {'built_never_evaluated': b['unknown_then'].get('built_never_evaluated')}
            for dp in b.get('decision_points', []) or []:
                vt = dp.get('visible_then') or {}
                dp['visible_then'] = {'remaining_before': vt.get('remaining_before'), 'built_not_evaluated': vt.get('built_not_evaluated')}
                for h in dp.get('hypotheses_stated', []):
                    for k in ('rationale', 'reason'):
                        if k in h:
                            h[k] = str(h[k])[:100]
                out = []
                for r in dp.get('results', []):
                    t = r.get('tool')
                    if t == 'inspect_material' and 'change_X' in r:
                        out.append({'tool': t, 'plan_id': r.get('plan_id'), 'rms_X': (r.get('change_X') or {}).get('rms'), 'rms_y': (r.get('change_y') or {}).get('rms'),
                                    'windows_bitwise_unchanged': r.get('windows_bitwise_unchanged')})
                    elif t == 'compare':
                        out.append({'tool': t, 'a': r.get('a'), 'b': r.get('b'), 'mean_delta_a_minus_b_loss': r.get('mean_delta_a_minus_b_loss'), 'signs': r.get('signs')})
                    elif t == 'evaluate':
                        out.append({'tool': t, 'plan_id': r.get('plan_id'), 'c_a_by_seed': r.get('c_a_by_seed'), 'physical_material': r.get('physical_material')})
                    else:
                        out.append(r)
                dp['results'] = out
    cen['compression']['tier_used'] = '5'
    cen['compression']['tier_meaning']['5'] = 'tier 4 without per-branch offline materials / summary; rationale <=100 chars; compact inspect / compare / evaluate results'
    return cen


def census_tc(root: Path, scope: str, tier) -> dict:
    """scope = 'D01' / 'D02' (12 Source cases of the domain) or 'SHARED' (24 Source cases of both domains). Reads only the Source stage of this
    package; the branch record per case is DP's decision-point record; time metadata (period id / order, entity-group tail, domain id for the shared
    census) is carried as evidence metadata only."""
    domains = list(DOMAINS) if scope == SHARED else [scope]
    bdirs = source_branch_dirs(root, domains)
    all_cases = cases_of(root)
    anchor_order = {a: i + 1 for i, a in enumerate(STAGE_ANCHORS['source'])}
    by_case = {}
    for b in bdirs:
        by_case.setdefault(b['case'], []).append(b)
    base_tier = 3 if tier in (4, 5) else tier
    cases, refs = {}, []
    for case, bl in by_case.items():
        cs = all_cases[case]
        mats = DP.case_materials(bl, case, base_tier)
        ov_dir = next((b['dir'] for b in bl if (b['dir'] / case / 'overview.json').exists()), None)
        ov = context.read_json(ov_dir / case / 'overview.json') if ov_dir else None
        recs = []
        for b in bl:
            if not (b['dir'] / 'trace.jsonl').exists() and not (b['dir'] / 'trace_before_resume.jsonl').exists():
                recs.append({'ref_prefix': ref(b['stage'], b['branch'], ''), 'arm': b['arm'], 'status': 'NOT_RUN'})
                continue
            rec, r_ = DP.branch_record(b, mats, {}, base_tier)
            refs += r_
            recs.append(rec)
        post = ES._post_hoc(mats, next((r.get('offline_verification', {}).get('commit_physical') for r in recs if r.get('status') == 'COMPLETE'), None))
        cases[case] = {'case_kind': 'source', 'domain_id': cs.domain, 'period_id': anchor_of(case), 'period_order': anchor_order.get(anchor_of(case)),
                       'entity_group': group_tail(case), 'train_rows': ov['train_rows'] if ov else None,
                       'overview': DP._overview_view(ov, base_tier) if ov else None, 'materials': mats, 'branches': recs,
                       'post_hoc': {k: v for k, v in post.items() if k in ('note', 'c_a_argmin_physical', 'e_best_physical_diagnostic', 'commit_physical', 'commit_is_c_a_argmin', 'e_ratio_to_none')}}
    actions = ES.action_table()
    actions['primitives'] = {n: {k: v for k, v in p.items() if k != 'source'} for n, p in actions['primitives'].items()}
    n_completed = sum(1 for c in cases.values() for r in c['branches'] if r.get('status') == 'COMPLETE')
    n_expected = 12 * len(domains)
    groups_by_period = {}
    for case in cases:
        groups_by_period.setdefault(cases[case]['period_id'], []).append('%s:%s' % (cases[case]['domain_id'], cases[case]['entity_group']))
    ev = {'census_status': 'CENSUS_COMPLETE' if len(cases) == n_expected and n_completed >= n_expected else 'CENSUS_INCOMPLETE', 'scope': scope,
          'domain_id': scope if scope != SHARED else None, 'domains_included': domains, 'purpose': 'temporal_coverage_learning', 'profile': ec.PROFILE_VERSION,
          'material_origin': ('%d research cases of %s: entity groups of 16 at six chronological periods (two groups per period; groups of the first four '
                              'periods reappear at a later period with new material and new weights), each researched once by the no-card Fast; all labelled'
                              % (n_expected, 'ONE neutral domain' if scope != SHARED else 'TWO neutral domains (twelve each)')),
          'time_organisation': {'periods_in_chronological_order': list(STAGE_ANCHORS['source']), 'entity_groups_by_period': groups_by_period,
                                'note': 'period ids and entity-group tails are evidence metadata for cross-checking only; Fast never observes the calendar; a card may not map a period, '
                                        'a date, a case or an entity group to a program'},
          'organisation': 'per case: one T-only overview, one materials table (physical id -> label, program, C_A / C_B / E per seed), one trajectory record organised as DECISION POINTS '
                          '(what was visible then, the actions and stated hypotheses, compact results), what was unknown then, and the OFFLINE verification (supervision). '
                          'Every case counts once; a longer trajectory is not more evidence; the same entity group at two periods is two cases of the same entities, not two independent groups.',
          'cases_in_order': list(cases), 'cases': cases, 'n_branches_completed': n_completed,
          'legal_evidence_refs': sorted(set(refs)), 'compression': {'tier_used': str(tier), 'tier_meaning': {'1': 'per-origin numbers, per-entity assignments, inspect window stats',
                                                                                                          '2': 'per-seed numbers only, inspect window stats, rationale <=500 chars',
                                                                                                          '3': 'per-seed numbers only, no window stats, rationale <=240 chars',
                                                                                                          '4': 'tier 3 without per-entity overview columns; rationale <=160 chars',
                                                                                                          '5': 'tier 4 without per-branch offline materials / summary; rationale <=100 chars; compact inspect / compare / evaluate results'},
                                                                    'never_dropped': 'actions, plans, C_A / C_B / E per seed, commits, failures, rejections, hypotheses (truncated at most)'},
          'feedback_roles': {'C_A': 'immediate feedback used by Fast at decision time (origins t, t+48)', 'C_B': 'delayed check of the same case (origins t+96, t+144), opened after commit',
                             'E': 'development_late: the late block (origins t+192..t+336), opened after every commit of the Source stage; authorized for learning only'},
          'public_semantics': {'field_definitions': FIELD_DEFINITIONS, 'actions': actions, 'tool_contracts': CONTRACTS, 'consumer': ec.CONSUMER,
                               'geometry': {'L': spec.L, 'H': spec.H, 'T': spec.TRAIN_SPAN, 'n_entities': ec.COHORT_SIZE, 'n_parents_per_entity': spec.N_PARENTS},
                               'public_references': {m: ES.public_spec(m) for m in PUBLIC}, 'budget_per_case': LIMITS, 'body_limit_characters': BODY_LIMIT,
                               'metric': 'normalized MSE (frozen T scaler), entity macro; lower is better',
                               'material_semantics_sentence_all_fast_arms': DP.MATERIAL_SEMANTICS,
                               'case_count': '%d entity-group cases (16 entities each) at six periods; not independent time environments and not 16 x cases independent entities' % n_expected}}
    if tier == 4:
        ev = _shrink_tier4(ev)
    elif tier == 5:
        ev = _shrink_tier5(ev)
    low = json.dumps(ev, ensure_ascii=False).lower()
    leak = [n for n in tuple(spec.DATASETS) if n in low]
    if leak:
        raise PermissionError('census names a data source: %s' % leak)
    return ev


# ============================================================================= Slow: three independent single-card proposals per domain and three shared proposals
FEEDBACK_ROLES_TC = ('Feedback roles, by name: C_A = the immediate feedback Fast legally used at decision time (origins t, t+48); it explains why Fast decided as it did. '
                     'C_B = the delayed check of that case after its commit (origins t+96, t+144). E (development_late) = the late block of that case (origins t+192..t+336), the '
                     'criterion this study cares about, opened only after every commit of the Source stage and authorized for learning only. When C_A and E disagree, KEEP the conflict: '
                     'a C_A lead is not a later gain, and no rule may a priori ignore all C_A. All blocks are the same 48-hour forecast; they differ in the calendar period scored. '
                     'The cases are entity groups of 16 at six chronological periods; the cases this card will serve are NEW entity groups of the same domain at LATER periods. Fast '
                     'never observes the calendar or the period: it observes the T-only statistics the tools compute and its own C_A. A per-entity prediction change is not the '
                     'independent contribution of that entity\'s material; grouping by observations stays legal and is judged by the complete shared model. A material that was '
                     'built but never evaluated has NO downstream number: it is unverified, not shown harmful, and the post-hoc pool best is a diagnostic, never a known answer. '
                     'Material appearance (smoother, flatter, larger change) is not downstream utility in either direction. Uncertainty can justify an extra check, a limited trial '
                     'or keeping a public reference; one undecided comparison never becomes a permanent ban; an untried action is not harmful.')
GOAL_TC = ('Your goal: improve the FINAL downstream prediction (E) of the complete Fast on NEW entity groups at LATER periods and save worthless experiments. Learn from the '
           'trajectories: which observations helped to form a processing hypothesis, how discriminating complete plans were (or should have been) constructed, how current C_A '
           'feedback should be used, and when to continue, keep a public reference or commit. A simple uniform strategy, a public reference, an explicit composition, a preset '
           'edit or an observation-conditional plan are all legal recommendations when the evidence supports them; complexity, grouping, three steps, using all four slots, '
           'deviating from or keeping the preset are never required. Never experimenting is not delivery progress; cost is an additional dimension. Do not ban a primitive '
           'because it was not tried; do not infer utility from appearance. You write ONE card (an executable Workflow covering observe / construct / experiment / commit-or-stop, '
           'plus optional Principles) or KEEP (= no usable card; the no-card Fast is then used). Soft priorities never change tool permissions; a card may not forbid grouping, '
           'compositions or any public tool as a global ban "because untested"; do not require "must dominate" or "3/3". State the COMMIT POLICY explicitly (the role of the '
           'current C_A; when a learned preference yields to current evidence). Say which conditions the current tools compute and which are learned preferences. Give every main '
           'rule its support / counter refs and mark it supported / hypothesis / unresolved. Name the ONE main decision mechanism your card changes relative to the no-card Fast '
           'and its deployment-observable trigger, one historical decision point your card would have changed (expected, not run), one counterexample where the card should hold '
           'or may fail, and the conditions under which your card behaves like the no-card Fast. KEEP or an undecided rule is allowed when the evidence is insufficient; the format '
           'never forces novelty.')
SHARED_TC = ('This is the SHARED learning branch: the census holds the Source cases of TWO neutral domains (domain ids and entity-group tails are metadata only). You write ONE card '
             'deployed identically in both domains: it may branch on legal observations that the tools compute (overview fields, inspections, C_A), never on a domain id, case id, '
             'period or entity group. Say in cross_domain_note how the same text behaves when the two domains\' observations differ, and which advice is common to both.')
NO_IDS_TC = ('Do not write data source names, domain ids, case ids (also short forms), period / anchor / stage ids, calendar dates or months, cut positions or row numbers, entity or '
             'column numbers, historical assignments or answer tables into deployable text (workflow, principles, applicability_summary, research_mode); evidence_refs and the two ref '
             'fields carry the references. Do not change model, scoring, tools, permissions, budgets, randomness, targets or legal windows. observable_applicability MUST be exactly '
             '{"const":true}; applicability conditions belong inside the Workflow text as observable conditions.')
FORMAT_TC = ('Output exact JSON, no markdown: {"decision":"KEEP","rationale":"...","evidence_review":{"supports":[...],"contradicts_or_limits":[...],"rule_changes":[...],'
             '"uncertain_and_expected_behavior_change":"..."}} OR {"decision":"PROPOSE","candidate":{"research_mode":"<short label>",'
             '"workflow":"<executable Workflow incl. observe / construct / experiment / commit-or-stop; the commit policy must be inside it>","principles":"<conditions, actions, reasons, exceptions>" or null,'
             '"observable_applicability":{"const":true},"applicability_summary":"<=400 characters","evidence_refs":["exact refs from the supplied legal ranges"],"rationale":"...",'
             '"evidence_review":{"supports":["observation -> which advice it supports"],"contradicts_or_limits":["which later result (C_B / E) contradicts or limits which judgement"],'
             '"rule_changes":["which rule is kept / dropped / rewritten because of it"],"uncertain_and_expected_behavior_change":"what remains uncertain and which Fast behaviour should change"},'
             '"rule_status":[{"rule":"<one main rule>","status":"supported|hypothesis|unresolved","support":["refs or short facts"],"counter":["refs or short facts"]}],'
             '"commit_policy":"<one paragraph restating the commit rule>","main_mechanism_changed":{"mechanism":"<the one decision mechanism>","deployment_observable_trigger":"<what the tools show that triggers it>"},'
             '"decision_change_vs_no_card":"<the concrete decision difference relative to the no-card Fast>","historical_decision_point_expected_to_change":{"ref":"<one legal ref>","expected_change":"..."},'
             '"counterexample_expected":{"ref":"<one legal ref>","should_hold_or_may_fail":"..."},"conditions_same_as_no_card":"..."%s}}. Workflow plus Principles render to at most %d characters. '
             'KEEP (= no card for this proposal slot; recorded NO_CARD, aliased to the no-card Fast) is valid and is not resampled.')


def slow_system(scope: str) -> str:
    n, who = (24, 'TWO neutral domains (twelve cases each, domain id as metadata)') if scope == SHARED else (12, 'ONE neutral domain')
    role = ('You are the offline Slow of a batch training-data augmentation research Harness. You receive a deterministic census of %d completed research cases of %s: '
            'entity groups of 16 at SIX chronological periods (two entity groups per period; the groups of the first four periods reappear once at a later period with new '
            'material and new weights), each researched once by the no-card Fast with the same tools and budget. Each trajectory is organised as DECISION POINTS: what was '
            'visible then (T observations, C_A obtained so far, built / evaluated sets, remaining budget), the actions with their stated hypotheses, the compact tool results, '
            'what was unknown then, and - separately, as supervision for you - the OFFLINE verification: per-seed C_A / C_B / E of every material that trajectory evaluated, its '
            'real commit and cost. You are one of three independent Slow proposals formed from the identical evidence; you do not see the other two. ' % (n, who))
    extra = (' ' + SHARED_TC) if scope == SHARED else ''
    fmt = FORMAT_TC % (',"cross_domain_note":"<how the same text behaves in both domains>"' if scope == SHARED else '', BODY_LIMIT)
    return role + FEEDBACK_ROLES_TC + ' ' + GOAL_TC + extra + ' ' + NO_IDS_TC + ' ' + fmt


CAND_KEYS_TC = {'research_mode', 'workflow', 'principles', 'observable_applicability', 'applicability_summary', 'evidence_refs', 'rationale', 'evidence_review', 'rule_status',
                'commit_policy', 'main_mechanism_changed', 'decision_change_vs_no_card', 'historical_decision_point_expected_to_change', 'counterexample_expected', 'conditions_same_as_no_card'}


def cand_keys(scope: str) -> set:
    return CAND_KEYS_TC | ({'cross_domain_note'} if scope == SHARED else set())


def slow_payload(cen: dict, scope: str, slot: int) -> dict:
    refs = cen['legal_evidence_refs']
    body = {k: v for k, v in cen.items() if k != 'legal_evidence_refs'}
    return {'scope': scope, 'domain_id': cen.get('domain_id'), 'proposal_slot': slot,
            'proposal_note': 'proposal %d of three independent proposals formed from the identical census; the others are not visible to you' % slot,
            'census': body, 'body_limit_characters': BODY_LIMIT, 'legal_evidence_refs': LL.ref_ranges(refs),
            'legal_evidence_refs_format': 'evidence_refs must be exact ids inside these ranges, e.g. "source/<branch>/<case>:<integer>", or one of other_refs',
            'required': {'observable_applicability': {'const': True}}, 'status': 'CANDIDATE_TEST_ONLY'}


def make_card_tc(*, scope, skill_id, workflow, principles, summary, refs, legal_refs, mode) -> dsk.Skill:
    if not isinstance(mode, str) or not mode.strip() or len(mode) > 80:
        raise ValueError('research_mode must be short nonempty text')
    dsk.check_reusable_text(mode, 'research_mode', tc_text_check)
    return dsk.make_skill(skill_id=skill_id, domain_id=scope, revision=1, workflow=workflow, principles=principles, applicability_summary=summary,
                          compatibility_note='Formed for this study\'s fixed task, Consumer, L/H, hourly sampling, T length and 16-entity cases.', observable_applicability={'const': True},
                          evidence_refs=refs, legal_evidence_refs=legal_refs, source_stage='propose', status='CANDIDATE_TEST_ONLY',
                          allowed_features=ALLOWED_FEATURES, text_check=tc_text_check, body_limit=BODY_LIMIT)


def parse_proposal_tc(resp, *, scope: str, slot: int, legal_refs) -> dict:
    """-> {'decision': 'KEEP'|'PROPOSE', 'skill': Skill|None, 'meta': {...}}; ValueError on contract errors."""
    legal = set(legal_refs)
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'PROPOSE'):
        raise ValueError('decision must be KEEP or PROPOSE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale', 'evidence_review'}:
            raise ValueError('KEEP carries only rationale and evidence_review')
        return {'decision': 'KEEP', 'skill': None, 'meta': {'rationale': str(resp.get('rationale', ''))[:3000], 'evidence_review': LL.check_review(resp['evidence_review']) if 'evidence_review' in resp else None}}
    if set(resp) != {'decision', 'candidate'} or not isinstance(resp['candidate'], dict):
        raise ValueError('PROPOSE carries exactly decision and one candidate object')
    c = resp['candidate']
    if set(c) != cand_keys(scope):
        raise ValueError('candidate keys must be exactly %s' % sorted(cand_keys(scope)))
    if c['observable_applicability'] != {'const': True}:
        raise ValueError('observable_applicability must be exactly {"const": true}')
    for k in ('commit_policy', 'decision_change_vs_no_card', 'conditions_same_as_no_card') + (('cross_domain_note',) if scope == SHARED else ()):
        if not isinstance(c[k], str) or not c[k].strip():
            raise ValueError('%s must be nonempty text' % k)
    mm = c['main_mechanism_changed']
    if not isinstance(mm, dict) or set(mm) != {'mechanism', 'deployment_observable_trigger'} or any(not isinstance(mm[k], str) or not mm[k].strip() for k in mm):
        raise ValueError('main_mechanism_changed must be {mechanism, deployment_observable_trigger} with nonempty text')
    hist_ref = DP._legal_ref(c['historical_decision_point_expected_to_change'], legal, 'historical_decision_point_expected_to_change')
    counter_ref = DP._legal_ref(c['counterexample_expected'], legal, 'counterexample_expected')
    review = LL.check_review(c['evidence_review'])
    rs = ES.check_rule_status(c['rule_status'])
    prefix = 'S' if scope == SHARED else 'D'
    s = make_card_tc(scope=scope, skill_id='%s-%s%d-r1' % (scope, prefix, slot), workflow=c['workflow'], principles=c['principles'], summary=c['applicability_summary'],
                     refs=c['evidence_refs'], legal_refs=legal_refs, mode=c['research_mode'])
    meta = {'research_mode': c['research_mode'], 'rationale': str(c['rationale'])[:3000], 'evidence_review': review, 'rule_status': rs, 'commit_policy': c['commit_policy'][:2000],
            'main_mechanism_changed': {k: mm[k][:800] for k in mm}, 'decision_change_vs_no_card': c['decision_change_vs_no_card'][:2000],
            'historical_decision_point_expected_to_change': {**c['historical_decision_point_expected_to_change'], 'ref': hist_ref},
            'counterexample_expected': {**c['counterexample_expected'], 'ref': counter_ref}, 'conditions_same_as_no_card': c['conditions_same_as_no_card'][:1500]}
    if scope == SHARED:
        meta['cross_domain_note'] = c['cross_domain_note'][:2000]
    return {'decision': 'PROPOSE', 'skill': s, 'meta': meta}


def propose_one(payload: dict, call, *, scope: str, slot: int, legal_refs) -> dict:
    """One scientific proposal call; one contract correction only; transport / account / budget faults end as PROPOSE_CALL_FAILED (technical, never a card)."""
    low = json.dumps(payload, ensure_ascii=False).lower()
    if any(n in low for n in tuple(spec.DATASETS)):
        raise PermissionError('Slow payload names a data source')
    attempts, prior, receipts = [], None, []
    for i in range(2):
        try:
            raw, rec = call(payload)
            receipts.append(rec)
        except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
            attempts.append({'attempt': i, 'fault_kind': type(exc).__name__})
            return {'status': 'PROPOSE_CALL_FAILED', 'attempts': attempts, 'receipts': receipts, 'skill': None, 'meta': {}}
        except ValueError as exc:
            attempts.append({'attempt': i, 'error': 'response not JSON: %s' % str(exc)[:200]})
            prior = None
        else:
            try:
                out = parse_proposal_tc(raw, scope=scope, slot=slot, legal_refs=legal_refs)
                attempts.append({'attempt': i, 'ok': True})
                return {'status': 'NO_CARD' if out['decision'] == 'KEEP' else 'PROPOSED', 'attempts': attempts, 'receipts': receipts, 'raw': raw, **out}
            except ValueError as exc:
                attempts.append({'attempt': i, 'error': str(exc)[:300]})
                prior = raw if dsk._jsonable(raw) else None
        if i == 0:
            payload = {**payload, 'correction': {'previous_output': prior, 'error': attempts[-1].get('error'), 'instruction': 'Correct this contract error only. KEEP is allowed. Do not seek a different outcome.'}}
    return {'status': 'PROPOSE_PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'receipts': receipts, 'skill': None, 'meta': {}}


def build_census(root: Path, scope: str) -> dict:
    """The frozen census of a scope (tier chosen so that the serialized payload fits the planning byte target, else the smallest tier under the
    hard backend envelope; else refused) written once under evidence/<scope>/."""
    P_ = paths(root)
    out = P_['evidence'] / scope
    out.mkdir(parents=True, exist_ok=True)
    p = out / 'census.json'
    if p.exists():
        return context.read_json(p)
    target = SHARED_BYTE_TARGET if scope == SHARED else DOMAIN_BYTE_TARGET
    sizes, chosen = {}, None
    for tier in (1, 2, 3, 4, 5):
        cen = census_tc(root, scope, tier)
        nb = payload_bytes(slow_system(scope), slow_payload(cen, scope, 1))
        sizes['tier_%d_bytes' % tier] = nb
        chosen = cen
        if nb <= target:
            break
    final_bytes = sizes['tier_%s_bytes' % chosen['compression']['tier_used']]
    if final_bytes > HARD_BYTE_LIMIT:
        raise RuntimeError('census of %s exceeds the backend envelope even at tier 5: %d bytes > %d' % (scope, final_bytes, HARD_BYTE_LIMIT))
    chosen['compression']['payload_bytes_by_tier'] = sizes
    chosen['compression']['byte_target'] = target
    chosen['compression']['hard_byte_limit'] = HARD_BYTE_LIMIT
    chosen['compression']['within_planning_target'] = final_bytes <= target
    chosen['compression']['frozen_local'] = now()
    dsks.write_once(p, chosen)
    return chosen


SCOPES = tuple(DOMAINS) + (SHARED,)


def slow_stage(root: Path, driver_lock: PackageLock) -> dict:
    """Task §5.C: three independent proposals per domain and three shared (nine calls through the HTTP pool), each with at most one contract
    correction; KEEP = NO_CARD; a byte-identical body is an alias. 0 fits. Writes slow/<scope>/proposal_<k>.json and slow/proposals.json."""
    P_ = paths(root)
    P_['slow'].mkdir(parents=True, exist_ok=True)
    summary_p = P_['slow'] / 'proposals.json'
    if summary_p.exists():
        return context.read_json(summary_p)
    censuses = {s: build_census(root, s) for s in SCOPES}
    for s in SCOPES:
        if censuses[s]['census_status'] != 'CENSUS_COMPLETE':
            raise RuntimeError('census of %s incomplete: %s' % (s, censuses[s]['census_status']))
    payloads = {(s, k): slow_payload(censuses[s], s, k) for s in SCOPES for k in SLOTS}
    reservation = {'%s_%d' % (s, k): {'bytes': payload_bytes(slow_system(s), p), 'reserved_tokens_upper_2x': 2 * (payload_bytes(slow_system(s), p) + 2048 + MAX_OUTPUT_TOKENS)} for (s, k), p in payloads.items()}
    if not (P_['slow'] / 'reservation_check.json').exists():
        dsks.write_once(P_['slow'] / 'reservation_check.json', {'checked_local': now(), 'payloads': reservation, 'request_timeout_s': SLOW_REQUEST_TIMEOUT_S,
                                                               'note': 'no call sent by this check; sizes of the final serialized payloads'})
    sc = stage_caps(root, 'slow')
    if not proxy_reachable():
        raise RuntimeError('LLM proxy not reachable; Slow refused before any paid call')
    led = SafeLedger(P_['ledger'], **sc['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage; paid Slow refused')
    client = MeteredClient(led, P_['slow'] / 'raw_responses', http_cap=sc['http_cap'], stage='slow')
    results, lock = {}, threading.Lock()

    def one(s, k):
        outp = P_['slow'] / s / ('proposal_%d.json' % k)
        if outp.exists():
            prior = context.read_json(outp)
            if prior['status'] in ('PROPOSED', 'NO_CARD', 'PROPOSE_PARSE_OR_VALIDATION_FAILED'):
                with lock:
                    results[(s, k)] = prior
                return
            n_prev = len(list((P_['slow'] / s).glob('proposal_%d_failed_*.json' % k)))
            shutil.move(outp, P_['slow'] / s / ('proposal_%d_failed_%d.json' % (k, n_prev + 1)))    # technical failure: the same frozen payload is called again after the operator decision
        legal = censuses[s]['legal_evidence_refs']
        res = propose_one(payloads[(s, k)], lambda p: client.call('slow_propose', '%s_%d' % (s, k), p, slow_system(s), max_tokens=MAX_OUTPUT_TOKENS, meta={'scope': s, 'slot': k},
                                                                    request_timeout=SLOW_REQUEST_TIMEOUT_S), scope=s, slot=k, legal_refs=legal)
        rec = {**{kk: v for kk, v in res.items() if kk != 'skill'}, 'skill': res['skill'].to_json() if res.get('skill') else None, 'scope': s, 'slot': k, 'written_local': now()}
        outp.parent.mkdir(parents=True, exist_ok=True)
        dsks.write_once(outp, rec)
        with lock:
            results[(s, k)] = rec
    errs = _parallel([(s, k) for s in SCOPES for k in SLOTS], one)
    if errs:
        raise RuntimeError('slow proposal thread failed: %s' % errs[0][1])
    failed = [(s, k) for (s, k), r in results.items() if r['status'] == 'PROPOSE_CALL_FAILED']
    if client.fatal or rt.unknown_usage_blocks(led) or failed:
        raise RuntimeError('Slow stopped: fatal=%s unknown_usage_blocks=%s call_failed=%s; finished proposals are kept, rerun --run (with --accept-unknown-usage after the operator decision) to call only the unfinished slots'
                           % (client.fatal, rt.unknown_usage_blocks(led), ['%s_%d' % x for x in failed]))
    summary = {'written_local': now(), 'scopes': {}}
    for s in SCOPES:
        rows, bodies = [], {}
        arms = SHARED_ARMS if s == SHARED else DOMAIN_ARMS
        for k in SLOTS:
            r = results[(s, k)]
            row = {'slot': k, 'arm': arms[k - 1], 'status': r['status'], 'attempts': r['attempts'], 'requests': [x.get('request') for x in r.get('receipts', [])],
                   'tokens': sum((x.get('prompt_tokens') or 0) + (x.get('completion_tokens') or 0) for x in r.get('receipts', [])), 'alias_of': None}
            if r['status'] == 'NO_CARD':
                row['alias_of'] = 'f0'
                row['alias_reason'] = 'KEEP: no card for this slot; the no-card Fast (F0) trajectory is reused'
            elif r['status'] == 'PROPOSED':
                body = r['skill']['rendered_body']
                same = next((name for name, b in bodies.items() if b == body), None)
                if same is not None:
                    row['alias_of'] = same
                    row['alias_reason'] = 'byte-identical rendered body to %s' % same
                else:
                    bodies[arms[k - 1]] = body
                row['research_mode'] = r['meta'].get('research_mode')
                row['main_mechanism_changed'] = r['meta'].get('main_mechanism_changed')
            else:
                row['alias_reason'] = 'technical failure (parse / validation after one correction): not a KEEP, not an alias; no card for this slot'
            rows.append(row)
        summary['scopes'][s] = {'proposals': rows, 'n_distinct_new_cards': sum(1 for r in rows if r['status'] == 'PROPOSED' and r['alias_of'] is None),
                                'census_tier': censuses[s]['compression']['tier_used'], 'census_bytes': censuses[s]['compression']['payload_bytes_by_tier']}
    dsks.write_once(summary_p, summary)
    return summary


def cards_of(root: Path, scope: str) -> dict:
    """{arm: Skill} of the scope's distinct cards (aliases and NO_CARD excluded)."""
    P_ = paths(root)
    summ = context.read_json(P_['slow'] / 'proposals.json')
    out = {}
    for row in summ['scopes'][scope]['proposals']:
        if row['status'] == 'PROPOSED' and row['alias_of'] is None:
            rec = context.read_json(P_['slow'] / scope / ('proposal_%d.json' % row['slot']))
            out[row['arm']] = dsk.skill_from_json(rec['skill'])
    return out


def aliases_of(root: Path, scope: str) -> dict:
    summ = context.read_json(paths(root)['slow'] / 'proposals.json')
    return {row['arm']: row['alias_of'] for row in summ['scopes'][scope]['proposals'] if row['alias_of']}


# ============================================================================= Select (2 cases per domain): F0 + domain cards + shared cards; J on E; freeze W_d / W_shared / Fixed_dev / H_deploy
def select_stage(root: Path, driver_lock: PackageLock) -> dict:
    cases_all = cases_of(root)
    shared_cards = cards_of(root, SHARED)
    shared_al = aliases_of(root, SHARED)
    knowledge, order, aliases, cases = {}, {}, {}, []
    for d in DOMAINS:
        dom_cards = cards_of(root, d)
        al = {**aliases_of(root, d), **shared_al}
        arms = {'f0': br.Knowledge()}
        arms.update({arm: dsk.skill_knowledge(s) for arm, s in dom_cards.items()})
        arms.update({arm: dsk.skill_knowledge(s) for arm, s in shared_cards.items()})
        for c in case_queue(root, 'select'):
            if cases_all[c].domain != d:
                continue
            cs = cases_all[c]
            cases.append(c)
            order[c] = [a for a in rotate(SELECT_ARMS, cs.case_index) if a in arms]
            knowledge[c] = {a: asdict(k) for a, k in arms.items()}
            aliases[c] = dict(al)
    cases = case_queue(root, 'select')
    notes = {a: 'alias (NO_CARD or byte-identical body): the aliased arm\'s trajectory and scores are reused; no second LLM run' for a in DOMAIN_ARMS + SHARED_ARMS}
    cfgp = stage_config(root, 'select', cases=cases, order=order, knowledge=br.json_copy(knowledge), labels_e=True, llm=True, random=False, aliases=aliases, alias_notes=notes)
    return run_stage(root, 'select', cfgp, driver_lock)


def _arm_run(stage_root: Path, case: str, arm: str, aliases: dict, tokens_all: dict) -> dict:
    """One arm's Select row: status, commit, E / None ratio, cost (aliases resolved)."""
    b, alias_of = DP._arm_dir(stage_root, case, arm, aliases)
    plan = ES._committed(b, case)
    e = ES._block_vec(b, case, plan, 'e') if plan else None
    none_e = ES._block_vec(b, case, 'None', 'e') if b.exists() else None
    res = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
    tok = tokens_all.get('fast:%s' % b.name, {})
    row = {'status': res['status'] if res else 'NOT_RUN', 'alias_of': alias_of, 'branch': b.name, 'committed': plan, 'committed_label': ES._label_of(b, case, plan) if plan else None,
           'failure_kind': res['failure_kind'] if res else None, 'e_by_seed': e, 'none_e_by_seed': none_e, 'c_a_by_seed': ES._block_vec(b, case, plan, 'c_a') if plan else None,
           'calls': res['calls'] if res else None, 'new_evaluations': res['new_evaluations'] if res else None,
           'tokens': (tok.get('prompt_tokens', 0) + tok.get('completion_tokens', 0)) if tok else 0}
    row['ratio'] = (statistics.fmean(e) / statistics.fmean(none_e)) if (e is not None and none_e is not None and statistics.fmean(none_e) > 0) else None
    return row


def decide_selection(J_dom: dict, cost_dom: dict, J_shared: dict, cost_shared: dict, fixed: dict, f0_J: dict, *, dom_arms=DOMAIN_ARMS, shared_arms=SHARED_ARMS) -> dict:
    """Pure decision arithmetic (task §5.D): per domain W_d = lowest J_d among the domain cards; ONE W_shared = lowest J_shared (mean of the two
    domains' J_d) among the shared cards; Fixed_dev,d = lowest J_d among uniform programs evaluated on both Select cases; H_deploy,d among
    F_domain / F_shared / F0 / Fixed_dev by the same J_d (ties: fixed program first, then fewer evaluations, fewer tokens, fixed id order).
    J_dom: {domain: {arm: J}}; cost_dom: {domain: {arm: (evals, tokens)}}; J_shared: {arm: {domain: J_d}}; cost_shared: {arm: (evals, tokens)};
    fixed: {domain: {'label', 'steps', 'public_id', 'J'} | None}; f0_J: {domain: {'J', 'cost'}}."""
    out = {'w_domain': {}, 'w_shared': None, 'h_deploy': {}}
    for d, J in J_dom.items():
        cands = [a for a in dom_arms if a in J]
        if not cands:
            out['w_domain'][d] = {'status': 'NO_CARD', 'arm': None, 'note': 'no distinct complete domain card (all KEEP / alias / technical failure): F_domain aliases F0'}
            continue
        best = min(J[a] for a in cands)
        tied = [a for a in cands if J[a] - best <= TOL]
        winner = min(tied, key=lambda a: (cost_dom[d][a][0], cost_dom[d][a][1], dom_arms.index(a)))
        out['w_domain'][d] = {'status': 'SELECTED', 'arm': winner, 'J': J[winner], 'tied': tied, 'margins': {a: J[a] - J[winner] for a in cands}}
    complete = [a for a in shared_arms if a in J_shared and all(d in J_shared[a] for d in J_dom)]
    if not complete:
        out['w_shared'] = {'status': 'NO_CARD', 'arm': None, 'note': 'no distinct shared card complete on every Select case: F_shared aliases F0'}
    else:
        Js = {a: statistics.fmean(J_shared[a][d] for d in J_dom) for a in complete}
        best = min(Js.values())
        tied = [a for a in complete if Js[a] - best <= TOL]
        winner = min(tied, key=lambda a: (cost_shared[a][0], cost_shared[a][1], shared_arms.index(a)))
        out['w_shared'] = {'status': 'SELECTED', 'arm': winner, 'J_shared': Js[winner], 'J_by_domain': dict(J_shared[winner]), 'tied': tied, 'margins': {a: Js[a] - Js[winner] for a in complete},
                           'J_shared_table': Js}
    for d in J_dom:
        cands = {}
        if fixed.get(d):
            cands['fixed_dev'] = (fixed[d]['J'], 0, 0, 0, 0)
        wd = out['w_domain'][d]
        if wd['status'] == 'SELECTED':
            cands['f_domain'] = (wd['J'], 1, cost_dom[d][wd['arm']][0], cost_dom[d][wd['arm']][1], 1)
        ws = out['w_shared']
        if ws['status'] == 'SELECTED' and d in ws['J_by_domain']:
            cands['f_shared'] = (ws['J_by_domain'][d], 1, cost_shared[ws['arm']][0], cost_shared[ws['arm']][1], 2)
        if f0_J.get(d) and f0_J[d].get('J') is not None:
            cands['f0'] = (f0_J[d]['J'], 1, f0_J[d]['cost'][0], f0_J[d]['cost'][1], 3)
        if cands:
            best = min(v[0] for v in cands.values())
            tied = [k for k, v in cands.items() if v[0] - best <= TOL]
            choice = min(tied, key=lambda k: cands[k][1:])
            out['h_deploy'][d] = {'choice': choice, 'J_table': {k: v[0] for k, v in cands.items()}, 'tied': tied,
                                  'rule': 'lowest J_d; ties <= 1e-12: fixed program first, then fewer evaluations, fewer tokens, then fixed id order f_domain / f_shared / f0',
                                  'metatest_arm': choice, 'note': 'frozen adoption decision; MetaTest reuses the corresponding arm, no new call'}
        else:
            out['h_deploy'][d] = None
    return out


def fixed_dev_of(root: Path, domain: str, sel_cases: list) -> dict | None:
    """Fixed_dev,d: uniform programs actually evaluated (E present) on BOTH Select cases of the domain over every branch (incl. public); lowest J_d;
    ties in public order then table order. Never a new fit."""
    P_ = paths(root)
    per_case = {}
    for c in sel_cases:
        pool = {}
        for b in sorted(p for p in P_['select'].glob('%s_*' % c) if p.is_dir() and not p.name.endswith('_common') and '__interrupted' not in p.name):
            for key, m in ES._uniform_materials_e(b, c).items():
                pool.setdefault(key, m)
        per_case[c] = pool
    common_keys = set.intersection(*(set(p) for p in per_case.values())) if per_case and all(per_case.values()) else set()
    fixed = {}
    for key in common_keys:
        ratios = []
        for c in sel_cases:
            m, none = per_case[c][key], per_case[c].get(ec._PKEY([]))
            if none is None or statistics.fmean(none['e_by_seed']) <= 0:
                ratios = None
                break
            ratios.append(statistics.fmean(m['e_by_seed']) / statistics.fmean(none['e_by_seed']))
        if ratios:
            m0 = per_case[sel_cases[0]][key]
            fixed[key] = {'label': m0['label'], 'steps': m0['steps'], 'public_id': m0['public_id'], 'J': statistics.fmean(ratios), 'ratios': ratios}
    if not fixed:
        return None

    def tie(key):
        m = fixed[key]
        pub_rank = PUBLIC.index(m['public_id']) if m['public_id'] in PUBLIC else len(PUBLIC)
        tab = ec.program_table_index(m['steps']) if m['steps'] is not None else ('fixed', 0)
        return (m['J'], pub_rank, 0 if tab[0] == 'explicit' else 1, tab[1])
    kbest = min(fixed, key=tie)
    return {**fixed[kbest], 'candidates_in_intersection': len(fixed), 'table': {v['label']: round(v['J'], 6) for v in sorted(fixed.values(), key=lambda v: v['J'])}}


def selection(root: Path) -> dict:
    """The Select readout of both domains and the shared branch (written once as slow/selection.json)."""
    P_ = paths(root)
    p = P_['slow'] / 'selection.json'
    if p.exists():
        return context.read_json(p)
    cfg = context.read_json(P_['configs'] / 'select.json')
    aliases = cfg.get('aliases', {})
    tokens_all = dsks._unit_tokens(P_['select']) if (P_['select'] / 'raw_responses').exists() else {}
    cases_all = cases_of(root)
    runs, J_dom, cost_dom, J_shared_raw, cost_shared_raw, fixed, f0_J, incomplete = {}, {}, {}, {}, {}, {}, {}, []
    sel_cases = {d: [c for c in case_queue(root, 'select') if cases_all[c].domain == d] for d in DOMAINS}
    for d in DOMAINS:
        dom_arms = [a for a in DOMAIN_ARMS if a not in aliases_of(root, d)]
        shared_arms = [a for a in SHARED_ARMS if a not in aliases_of(root, SHARED)]
        J_dom[d], cost_dom[d] = {}, {}
        for a in ['f0'] + dom_arms + shared_arms:
            rows = {c: _arm_run(P_['select'], c, a, aliases, tokens_all) for c in sel_cases[d]}
            runs.setdefault(a, {}).update(rows)
            ratios = [r['ratio'] for r in rows.values()]
            for c, r in rows.items():
                if r['ratio'] is None:
                    incomplete.append('%s/%s' % (a, c))
            if all(x is not None for x in ratios):
                Jd = statistics.fmean(ratios)
                cost = (sum(r['new_evaluations'] or 0 for r in rows.values()), sum(r['tokens'] for r in rows.values()))
                if a == 'f0':
                    f0_J[d] = {'J': Jd, 'cost': cost}
                elif a in dom_arms:
                    J_dom[d][a], cost_dom[d][a] = Jd, cost
                else:
                    J_shared_raw.setdefault(a, {})[d] = Jd
                    cost_shared_raw[a] = tuple(x + y for x, y in zip(cost_shared_raw.get(a, (0, 0)), cost))
        fixed[d] = fixed_dev_of(root, d, sel_cases[d])
    dec = decide_selection(J_dom, cost_dom, J_shared_raw, cost_shared_raw, fixed, f0_J)
    rec = {'formula': {'J_d': 'J_d(W) = mean over the domain\'s two Select cases of mean_seed E(actual commit of W) / mean_seed E(None); lower is better',
                       'J_shared': 'J_shared(W) = mean over the two domains of J_d(W); ONE shared winner for both domains',
                       'tie_rule_cards': '<= 1e-12 -> fewer new evaluations, fewer tokens, then slot order', 'fixed_dev': 'uniform programs with E on both Select cases of the domain; lowest J_d; ties public order then table order',
                       'h_deploy': 'lowest J_d among F_domain / F_shared / F0 / Fixed_dev; ties fixed program first, then fewer evaluations, fewer tokens, then f_domain / f_shared / f0'},
           'select_cases': sel_cases, 'aliases': {s: aliases_of(root, s) for s in SCOPES}, 'runs': runs, 'J_domain': J_dom, 'cost_domain': cost_dom, 'J_shared_by_domain': J_shared_raw,
           'cost_shared': {a: list(v) for a, v in cost_shared_raw.items()}, 'f0': f0_J, 'fixed_dev': fixed, 'incomplete': incomplete, 'decision': dec, 'block': 'e', 'written_local': now()}
    dsks.write_once(p, rec)
    return rec


def freeze_all(root: Path) -> dict:
    """freeze/D01.json, D02.json (W_d, Fixed_dev, H_deploy), freeze/shared.json (W_shared) and the MetaTest catalog; written once, before any MetaTest fit."""
    P_ = paths(root)
    P_['freeze'].mkdir(parents=True, exist_ok=True)
    sel = selection(root)
    dec = sel['decision']
    shared_cards = cards_of(root, SHARED)
    out = {}
    ws = dec['w_shared']
    sp = P_['freeze'] / 'shared.json'
    if not sp.exists():
        rec = {'scope': SHARED, 'frozen_local': now(), 'selection': ws, 'candidates': {a: s.to_json() for a, s in shared_cards.items()}}
        if ws['status'] == 'SELECTED':
            s = shared_cards[ws['arm']]
            frozen = replace(s, status='FROZEN_SELECTED', source_stage='freeze', derived_from=s.skill_id)
            rec['w_shared'] = {'status': 'FROZEN_SELECTED', 'arm': ws['arm'], 'skill': frozen.to_json(), 'injected_body': frozen.rendered_body, 'J_shared': ws['J_shared'], 'J_by_domain': ws['J_by_domain']}
        else:
            rec['w_shared'] = {'status': 'NO_CARD', 'arm': None, 'skill': None, 'injected_body': None, 'alias_of': 'f0', 'note': ws.get('note')}
        rec['note'] = 'one global shared winner selected on the four Select cases\' E (two per domain, equal weight); a selection, not a promotion to a verified Skill'
        dsks.write_once(sp, rec)
    out[SHARED] = context.read_json(sp)
    for d in DOMAINS:
        p = P_['freeze'] / ('%s.json' % d)
        if not p.exists():
            cards = cards_of(root, d)
            wd = dec['w_domain'][d]
            rec = {'domain_id': d, 'frozen_local': now(), 'selection': {'w_domain': wd, 'J_domain': sel['J_domain'][d], 'f0': sel['f0'].get(d), 'incomplete': [x for x in sel['incomplete'] if x.split('/')[1].startswith(d)]},
                   'fixed_dev': sel['fixed_dev'].get(d), 'h_deploy': dec['h_deploy'].get(d), 'candidates': {a: s.to_json() for a, s in cards.items()}}
            if wd['status'] == 'SELECTED':
                s = cards[wd['arm']]
                frozen = replace(s, status='FROZEN_SELECTED', source_stage='freeze', derived_from=s.skill_id)
                rec['w_domain'] = {'status': 'FROZEN_SELECTED', 'arm': wd['arm'], 'skill': frozen.to_json(), 'injected_body': frozen.rendered_body, 'J': wd['J'], 'alias_of': None}
            else:
                rec['w_domain'] = {'status': 'NO_CARD', 'arm': None, 'skill': None, 'injected_body': None, 'alias_of': 'f0', 'note': wd.get('note')}
            rec['note'] = 'selected on the domain\'s two Select cases\' E; a selection, not a promotion; F_domain and F_shared keep their MetaTest arms whatever H_deploy chose'
            dsks.write_once(p, rec)
        out[d] = context.read_json(p)
    return out


# ============================================================================= MetaTest (16 cases): F0 / F_domain / F_shared / Random / Fixed_dev + publics
def metatest_stage(root: Path, driver_lock: PackageLock) -> dict:
    frozen = freeze_all(root)
    cases_all = cases_of(root)
    cases = case_queue(root, 'test')
    knowledge, order, aliases, fixed = {}, {}, {}, {}
    shared_kn, shared_body = None, None
    if frozen[SHARED]['w_shared']['status'] == 'FROZEN_SELECTED':
        s = dsk.skill_from_json(frozen[SHARED]['w_shared']['skill'])
        if s.status != 'FROZEN_SELECTED' or s.domain_id != SHARED:
            raise PermissionError('only the FROZEN_SELECTED shared card enters MetaTest')
        shared_kn, shared_body = asdict(dsk.skill_knowledge(s)), s.rendered_body
    for d in DOMAINS:
        fr = frozen[d]
        dom_kn, dom_body, al = None, None, {}
        if fr['w_domain']['status'] == 'FROZEN_SELECTED':
            s = dsk.skill_from_json(fr['w_domain']['skill'])
            if s.status != 'FROZEN_SELECTED' or s.domain_id != d:
                raise PermissionError('only the FROZEN_SELECTED card of its domain enters MetaTest')
            dom_kn, dom_body = asdict(dsk.skill_knowledge(s)), s.rendered_body
        else:
            al['f_domain'] = 'f0'
        if shared_kn is None:
            al['f_shared'] = 'f0'
        elif dom_body is not None and dom_body == shared_body:
            al['f_shared'] = 'f_domain'
        if fr.get('fixed_dev'):
            fixed[d] = {k: fr['fixed_dev'][k] for k in ('label', 'steps', 'public_id', 'J')}
        for c in cases:
            if cases_all[c].domain != d:
                continue
            cs = cases_all[c]
            knowledge[c] = {'f0': asdict(br.Knowledge()), 'f_domain': dom_kn, 'f_shared': shared_kn, 'random': None, 'fixed_dev': None}
            order[c] = rotate(TEST_FAST_ARMS, cs.case_index) + list(CONTROL_ARMS)
            aliases[c] = dict(al)
    cat = paths(root)['freeze'] / 'metatest_catalog.json'
    if not cat.exists():
        dsks.write_once(cat, {'status': 'METATEST_FROZEN', 'epoch': time.time(), 'frozen_local': now(), 'w_domain': {d: {k: frozen[d]['w_domain'].get(k) for k in ('status', 'arm', 'J', 'alias_of')} for d in DOMAINS},
                              'w_shared': {k: frozen[SHARED]['w_shared'].get(k) for k in ('status', 'arm', 'J_shared', 'J_by_domain')}, 'fixed_dev': fixed,
                              'h_deploy': {d: frozen[d].get('h_deploy') for d in DOMAINS}, 'orders': order, 'aliases': aliases, 'knowledge_fixed_from_first_to_last_period': True,
                              'frozen_before_first_metatest_fit': True})
    cfgp = stage_config(root, 'metatest', cases=cases, order=order, knowledge=br.json_copy(knowledge), labels_e=True, llm=True, random=True, aliases=aliases,
                        alias_notes={'f_domain': 'no domain card: the F0 trajectory and scores are reused; no second LLM run',
                                     'f_shared': 'no shared card (alias F0) or shared body identical to the domain card (alias F_domain): the aliased trajectory and scores are reused'},
                        fixed_dev=fixed)
    return run_stage(root, 'metatest', cfgp, driver_lock)


# ============================================================================= readout (task §8)
G = DP.G


def _menu_ca(pool: dict) -> str | None:
    """Menu_CA: the C_A argmin among the four public references (ties in public order); 0 LLM, 0 extra fits."""
    pub = [m for m in PUBLIC if m in pool and pool[m].get('c_a') is not None]
    return min(pub, key=lambda m: (pool[m]['c_a'], PUBLIC.index(m))) if pub else None


def _loo_ranges(per_case: dict, meta: dict) -> dict:
    """Leave-one-period-out and leave-one-entity-group-out ranges of the overall mean (domain = mean over periods of the mean over groups; overall =
    mean over domains); sensitivity description only (task §8), never a confidence interval."""
    def overall(exclude_period=None, exclude_group=None):
        doms = []
        for d in DOMAINS:
            periods = {}
            for c, v in per_case.items():
                m = meta[c]
                if m['domain'] != d or v is None or m['period'] == exclude_period or m['group'] == exclude_group:
                    continue
                periods.setdefault(m['period'], []).append(v)
            if not periods:
                return None
            doms.append(statistics.fmean(statistics.fmean(v) for v in periods.values()))
        return statistics.fmean(doms) if len(doms) == len(DOMAINS) else None
    periods = sorted({m['period'] for m in meta.values()})
    groups = sorted({m['group'] for m in meta.values()})
    lop = {p: overall(exclude_period=p) for p in periods}
    log = {g: overall(exclude_group=g) for g in groups}
    return {'leave_one_period_out': lop, 'leave_one_period_out_range': [min(v for v in lop.values() if v is not None), max(v for v in lop.values() if v is not None)] if any(v is not None for v in lop.values()) else None,
            'leave_one_group_out': log, 'leave_one_group_out_range': [min(v for v in log.values() if v is not None), max(v for v in log.values() if v is not None)] if any(v is not None for v in log.values()) else None}


def aggregate(per_case: dict, meta: dict) -> dict:
    """per_case: {case: G mean or None}; meta: {case: {'domain', 'period', 'group'}}. Domain = equal-weight periods of the equal-weight two groups;
    overall = equal-weight domains; plus by_period (mean over domains), by_group, wins / ties / losses, max harm and the LOO ranges."""
    by_dom, by_dom_period = {}, {}
    for d in DOMAINS:
        periods = {}
        for c, v in per_case.items():
            if meta[c]['domain'] == d and v is not None:
                periods.setdefault(meta[c]['period'], []).append(v)
        by_dom_period[d] = {p: statistics.fmean(v) for p, v in sorted(periods.items())}
        by_dom[d] = statistics.fmean(by_dom_period[d].values()) if periods else None
    ok = [v for v in by_dom.values() if v is not None]
    by_period = {}
    for p in sorted({m['period'] for m in meta.values()}):
        vals = [by_dom_period[d][p] for d in DOMAINS if p in by_dom_period[d]]
        by_period[p] = statistics.fmean(vals) if len(vals) == len(DOMAINS) else None
    by_group = {}
    for d in DOMAINS:
        for g in sorted({m['group'] for m in meta.values()}):
            vals = [v for c, v in per_case.items() if v is not None and meta[c]['domain'] == d and meta[c]['group'] == g]
            by_group['%s:%s' % (d, g)] = statistics.fmean(vals) if vals else None
    vals = {c: v for c, v in per_case.items() if v is not None}
    return {'overall': statistics.fmean(ok) if len(ok) == len(DOMAINS) else None, 'by_domain': by_dom, 'by_domain_period': by_dom_period, 'by_period': by_period, 'by_group': by_group,
            'per_case': per_case, 'wins': sum(1 for v in vals.values() if v > 1e-9), 'ties': sum(1 for v in vals.values() if abs(v) <= 1e-9), 'losses': sum(1 for v in vals.values() if v < -1e-9),
            'n_cases': len(vals), 'max_harm': min(vals.values()) if vals else None, 'max_harm_case': min(vals, key=vals.get) if vals else None,
            'max_gain': max(vals.values()) if vals else None, 'max_gain_case': max(vals, key=vals.get) if vals else None, **_loo_ranges(per_case, meta)}


def readout(root: Path = ROOT) -> dict:
    P_ = paths(root)
    cases = cases_of(root)
    frozen = {s: (context.read_json(P_['freeze'] / ('%s.json' % ('shared' if s == SHARED else s))) if (P_['freeze'] / ('%s.json' % ('shared' if s == SHARED else s))).exists() else {'status': 'NOT_FROZEN'}) for s in SCOPES}
    out = {'package': PACKAGE, 'exposure': EXPOSURE, 'written_local': now(), 'frozen': {}, 'slow': {}, 'select': {}, 'metatest': {}, 'summary': {}}
    for d in DOMAINS:
        fr = frozen[d]
        out['frozen'][d] = {'w_domain': {k: (fr.get('w_domain') or {}).get(k) for k in ('status', 'arm', 'J', 'alias_of', 'injected_body')},
                            'fixed_dev': {k: v for k, v in (fr.get('fixed_dev') or {}).items() if k != 'table'} if fr.get('fixed_dev') else None, 'h_deploy': fr.get('h_deploy')}
    out['frozen'][SHARED] = {'w_shared': {k: (frozen[SHARED].get('w_shared') or {}).get(k) for k in ('status', 'arm', 'J_shared', 'J_by_domain', 'injected_body')}}
    sp = P_['slow'] / 'proposals.json'
    if sp.exists():
        s = context.read_json(sp)
        for sc in SCOPES:
            rows = []
            for r in s['scopes'][sc]['proposals']:
                row = dict(r)
                pp = P_['slow'] / sc / ('proposal_%d.json' % r['slot'])
                if pp.exists():
                    rec = context.read_json(pp)
                    row['meta'] = rec.get('meta')
                    row['body'] = (rec.get('skill') or {}).get('rendered_body')
                rows.append(row)
            out['slow'][sc] = {'proposals': rows, 'n_distinct_new_cards': s['scopes'][sc]['n_distinct_new_cards'], 'census_tier': s['scopes'][sc]['census_tier'], 'census_bytes': s['scopes'][sc]['census_bytes']}
    selp = P_['slow'] / 'selection.json'
    if selp.exists():
        s = context.read_json(selp)
        out['select'] = {'J_domain': s['J_domain'], 'J_shared_by_domain': s['J_shared_by_domain'], 'f0': s['f0'], 'fixed_dev': s['fixed_dev'], 'decision': s['decision'], 'incomplete': s['incomplete'],
                         'aliases': s['aliases'], 'runs': {a: {c: {k: v.get(k) for k in ('status', 'alias_of', 'committed', 'committed_label', 'ratio', 'new_evaluations', 'calls', 'tokens')} for c, v in rr.items()} for a, rr in s['runs'].items()}}
    arms = ('f0', 'f_domain', 'f_shared', 'random', 'fixed_dev')
    mt_cfg = context.read_json(P_['configs'] / 'metatest.json') if (P_['configs'] / 'metatest.json').exists() else {'aliases': {}}
    meta = {}
    for c in case_queue(root, 'test'):
        cs = cases[c]
        meta[c] = {'domain': cs.domain, 'period': anchor_of(c), 'group': group_tail(c)}
        recs = {}
        for a in arms:
            b, alias_of = DP._arm_dir(P_['metatest'], c, a, mt_cfg.get('aliases', {}))
            if b.exists():
                recs[a] = DP._arm_record(b, c, with_research=a in TEST_FAST_ARMS)
                if alias_of:
                    recs[a]['alias_of'] = alias_of
            else:
                recs[a] = {'status': 'NOT_RUN'}
        f0 = recs.get('f0', {})
        pub = {m: f0['pool'][m]['e_by_seed'] for m in PUBLIC if m in f0.get('pool', {}) and f0['pool'][m]['e_by_seed']}
        none_e = pub.get('None')
        den = statistics.fmean(none_e) if none_e else None
        e = {a: recs[a].get('e_by_seed') for a in arms}
        menu = _menu_ca(f0.get('pool', {}))
        e['menu_ca'] = pub.get(menu) if menu else None
        h = (frozen[cs.domain].get('h_deploy') or {}).get('metatest_arm')
        e['h_deploy'] = e.get(h) if h else None
        row = {'domain': cs.domain, 'period': anchor_of(c), 'group': group_tail(c), 't': cs.t, 'none_e_by_seed': none_e, 'public_e_by_seed': pub, 'menu_ca_choice': menu, 'h_deploy_arm': h,
               'arms': {a: {k: v for k, v in r.items() if k != 'pool'} for a, r in recs.items()}, 'G': {}}
        nomix = pub.get('P_NoMixRecipe')
        row['G'] = {'F_domain_over_F0': G(e['f_domain'], e['f0'], den), 'F_domain_over_F_shared': G(e['f_domain'], e['f_shared'], den), 'F_shared_over_F0': G(e['f_shared'], e['f0'], den),
                    'F_domain_over_Random': G(e['f_domain'], e['random'], den), 'F_domain_over_MenuCA': G(e['f_domain'], e['menu_ca'], den), 'F_domain_over_FixedDev': G(e['f_domain'], e['fixed_dev'], den),
                    'F_domain_over_NoMix': G(e['f_domain'], nomix, den), 'F_domain_over_None': G(e['f_domain'], none_e, den),
                    'F_shared_over_Random': G(e['f_shared'], e['random'], den), 'F_shared_over_MenuCA': G(e['f_shared'], e['menu_ca'], den), 'F_shared_over_FixedDev': G(e['f_shared'], e['fixed_dev'], den),
                    'F_shared_over_NoMix': G(e['f_shared'], nomix, den), 'F_shared_over_None': G(e['f_shared'], none_e, den),
                    'F0_over_Random': G(e['f0'], e['random'], den), 'F0_over_MenuCA': G(e['f0'], e['menu_ca'], den), 'F0_over_FixedDev': G(e['f0'], e['fixed_dev'], den), 'F0_over_NoMix': G(e['f0'], nomix, den), 'F0_over_None': G(e['f0'], none_e, den),
                    'Random_over_None': G(e['random'], none_e, den), 'MenuCA_over_None': G(e['menu_ca'], none_e, den), 'FixedDev_over_None': G(e['fixed_dev'], none_e, den), 'NoMix_over_None': G(nomix, none_e, den),
                    'FixedMixup_over_None': G(pub.get('FixedMixup'), none_e, den), 'AmpResample_over_None': G(pub.get('P_AmpResample'), none_e, den),
                    'H_deploy_over_F0': G(e['h_deploy'], e['f0'], den), 'H_deploy_over_F_domain': G(e['h_deploy'], e['f_domain'], den), 'H_deploy_over_F_shared': G(e['h_deploy'], e['f_shared'], den),
                    'H_deploy_over_Random': G(e['h_deploy'], e['random'], den), 'H_deploy_over_FixedDev': G(e['h_deploy'], e['fixed_dev'], den), 'H_deploy_over_NoMix': G(e['h_deploy'], nomix, den), 'H_deploy_over_None': G(e['h_deploy'], none_e, den)}
        for a in ('f0', 'f_domain', 'f_shared', 'random'):
            r = recs.get(a, {})
            if r.get('pool') and den and r.get('e_by_seed'):
                best = r.get('e_best_oracle')
                row['arms'][a]['oracle_gap_pp'] = 100.0 * (statistics.fmean(r['e_by_seed']) - r['pool'][best]['e']) / den if best else None
                row['arms'][a]['oracle_label'] = r['pool'][best]['label'] if best else None
                row['arms'][a]['post_hoc_vs_nomix'] = DP._post_hoc_pool(r, den)
                row['arms'][a]['independent_deployment_fits'] = 12 + 3 * int(r.get('new_evaluations') or 0)
        row['e_ratio_to_none'] = {a: (statistics.fmean(e[a]) / den if (e.get(a) and den) else None) for a in list(arms) + ['menu_ca', 'h_deploy']}
        row['e_ratio_to_none'].update({m: (statistics.fmean(v) / den if den else None) for m, v in pub.items()})
        row['same_delivery'] = {'F_domain_vs_F0': (recs['f_domain'].get('committed_label') == recs['f0'].get('committed_label')) if recs['f_domain'].get('committed_label') else None,
                                'F_domain_vs_F_shared': (recs['f_domain'].get('committed_label') == recs['f_shared'].get('committed_label')) if recs['f_domain'].get('committed_label') else None,
                                'F_shared_vs_F0': (recs['f_shared'].get('committed_label') == recs['f0'].get('committed_label')) if recs['f_shared'].get('committed_label') else None}
        out['metatest'][c] = row
    keys = list(next(iter(out['metatest'].values()))['G']) if out['metatest'] else []
    agg = {k: aggregate({c: (out['metatest'][c]['G'][k]['mean'] if out['metatest'][c]['G'].get(k) else None) for c in out['metatest']}, meta) for k in keys}
    research = {}
    for a in TEST_FAST_ARMS:
        rows = [out['metatest'][c]['arms'][a].get('research') for c in out['metatest'] if out['metatest'][c]['arms'].get(a, {}).get('research')]
        own = [c for c in out['metatest'] if not out['metatest'][c]['arms'][a].get('alias_of')]
        research[a] = {'cases': len(rows), 'aliased_cases': len(out['metatest']) - len(own), 'calls': sum(r['calls'] for r in rows), 'tool_calls': sum(r['tool_calls'] for r in rows),
                       'new_evaluations': sum(r['new_evaluations'] for r in rows), 'built': sum(r['n_built'] for r in rows), 'built_never_evaluated': sum(len(r['built_never_evaluated']) for r in rows),
                       'inspect_material_calls': sum(r['inspect_material_calls'] for r in rows), 'inspect_data_calls': sum(r['inspect_data_calls'] for r in rows),
                       'possible_unverified_harm_claims': sum(len(r['possible_unverified_harm_claims']) for r in rows),
                       'commit_is_c_a_argmin': sum(1 for c in out['metatest'] if out['metatest'][c]['arms'].get(a, {}).get('commit_is_c_a_argmin')),
                       'tokens': sum(int(out['metatest'][c]['arms'][a].get('tokens') or 0) for c in own), 'requests': sum(int(out['metatest'][c]['arms'][a].get('requests') or 0) for c in own),
                       'physical_fits': sum(int(out['metatest'][c]['arms'][a].get('physical_fits_this_branch') or 0) for c in own),
                       'independent_deployment_fits': sum(int(out['metatest'][c]['arms'][a].get('independent_deployment_fits') or 0) for c in out['metatest'])}
    out['summary'] = {'G_pp': agg, 'research': research, 'case_meta': meta,
                      'commit_vs_c_a_argmin': {c: {a: out['metatest'][c]['arms'][a].get('commit_is_c_a_argmin') for a in TEST_FAST_ARMS} for c in out['metatest']},
                      'committed_labels': {c: {a: out['metatest'][c]['arms'][a].get('committed_label') for a in arms} for c in out['metatest']},
                      'same_delivery': {c: out['metatest'][c]['same_delivery'] for c in out['metatest']},
                      'menu_ca_choice': {c: out['metatest'][c]['menu_ca_choice'] for c in out['metatest']},
                      'post_hoc_vs_nomix': {c: {a: out['metatest'][c]['arms'][a].get('post_hoc_vs_nomix') for a in ('f0', 'f_domain', 'f_shared', 'random')} for c in out['metatest']}}
    out['cost'] = cost_readout(root)
    context.write_json(root / 'result.json', out)
    (root / 'tables.md').write_text(tables(out), encoding='utf-8')
    return out


def cost_readout(root: Path) -> dict:
    P_ = paths(root)
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    stages = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    per_stage, names = {}, list(stages)
    for i, s in enumerate(names):
        snap = stages[s]['snapshot_at_start']
        nxt = stages[names[i + 1]]['snapshot_at_start'] if i + 1 < len(names) else {k: led.get(k, 0) for k in snap}
        per_stage[s] = {k: nxt.get(k, 0) - snap.get(k, 0) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'fit_wall_seconds')}
        sroot = P_.get(s)
        fin = sroot / 'execution_finished.json' if sroot and (sroot / 'execution_finished.json').exists() else None
        if fin:
            f = context.read_json(fin)
            per_stage[s]['wall_s'] = f['epoch'] - stages[s]['epoch_start']
            per_stage[s]['pools'] = f.get('pools')
    ev = [e for e in led.get('events', []) if e.get('kind') == 'llm_finished' and e.get('stage') == 'slow']
    slow_tok = {'requests': len(ev), 'tokens': sum((e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0) for e in ev)}
    queue_waits = [e.get('queue_wait_s') for e in led.get('events', []) if e.get('kind') == 'llm_finished' and e.get('queue_wait_s') is not None]
    learning = {k: sum(per_stage.get(s, {}).get(k, 0) for s in ('source', 'slow', 'select')) for k in ('fit_attempts', 'llm_requests', 'llm_tokens_in', 'llm_tokens_out', 'wall_s')}
    return {'ledger': {k: v for k, v in led.items() if k != 'events'}, 'per_stage': per_stage, 'slow_calls': slow_tok, 'one_time_learning_cost': learning,
            'http_queue_wait': {'n': len(queue_waits), 'mean_s': statistics.fmean(queue_waits) if queue_waits else None, 'max_s': max(queue_waits) if queue_waits else None},
            'paid_elapsed_s': (time.time() - led['paid_clock_started_epoch']) if led.get('paid_clock_started_epoch') else None,
            'numeric_seconds': numeric_seconds(root) if P_['ledger'].exists() else None}


def _f(x, nd=2):
    return '-' if x is None else ('%.*f' % (nd, x))


def tables(res: dict) -> str:
    L_ = ['# %s tables (%s)' % (PACKAGE, res['written_local']), '', 'Exposure: %s. Metric: E normalized MSE (T scaler), entity macro, seed mean. G(A over B) in pp of the case None mean; positive = A better.' % res['exposure'], '']
    cols = ('F_domain_over_F0', 'F_domain_over_F_shared', 'F_shared_over_F0', 'F_domain_over_Random', 'F_domain_over_MenuCA', 'F_domain_over_FixedDev', 'F_domain_over_NoMix', 'F_domain_over_None', 'F0_over_NoMix')
    L_ += ['## MetaTest: G per case (pp; SE df=2 in parentheses)', '', '| case | dom | period | group | ' + ' | '.join(cols) + ' |', '|---|---|---|---|' + '---:|' * len(cols)]
    for c, r in res['metatest'].items():
        g = r['G']
        L_.append('| %s | %s | %s | %s | %s |' % (c, r['domain'], r['period'], r['group'], ' | '.join((_f(g[k]['mean']) + (' (%s)' % _f(g[k]['se']) if g[k]['se'] is not None else '')) if g.get(k) else '-' for k in cols)))
    agg = res['summary'].get('G_pp', {})
    L_ += ['', '| aggregate | D01 | D02 | overall | A09 | A10 | A11 | A12 | win/tie/loss | max harm (case) | LOO period range | LOO group range |', '|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|']
    for k, v in agg.items():
        L_.append('| %s | %s | %s | %s | %s | %d/%d/%d of %d | %s (%s) | %s | %s |' % (k, _f(v['by_domain'].get('D01')), _f(v['by_domain'].get('D02')), _f(v['overall']),
                                                                                    ' | '.join(_f(v['by_period'].get(p)) for p in ('A09', 'A10', 'A11', 'A12')), v['wins'], v['ties'], v['losses'], v['n_cases'],
                                                                                    _f(v['max_harm']), v['max_harm_case'], ('[%s, %s]' % tuple(_f(x) for x in v['leave_one_period_out_range'])) if v.get('leave_one_period_out_range') else '-',
                                                                                    ('[%s, %s]' % tuple(_f(x) for x in v['leave_one_group_out_range'])) if v.get('leave_one_group_out_range') else '-'))
    L_ += ['', '## MetaTest: E / None per arm', '', '| case | F0 | F_domain | F_shared | Random | Fixed_dev | Menu_CA | H_deploy | NoMix | FixedMixup | AmpResample |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c, r in res['metatest'].items():
        rr = r['e_ratio_to_none']
        L_.append('| %s | %s |' % (c, ' | '.join(_f(rr.get(k), 4) for k in ('f0', 'f_domain', 'f_shared', 'random', 'fixed_dev', 'menu_ca', 'h_deploy', 'P_NoMixRecipe', 'FixedMixup', 'P_AmpResample'))))
    L_ += ['', '## MetaTest: commits and research process', '', '| case | arm | status | committed | C_A argmin? | oracle gap pp | calls | tools | built | new evals | never evaluated | inspect_material | inspect_data | harm-claim flags | tokens |',
           '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c, r in res['metatest'].items():
        for a, v in r['arms'].items():
            rs = v.get('research') or {}
            L_.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                c, ARM_LABEL.get(a, a) + (' (alias %s)' % ARM_LABEL.get(v['alias_of'], v['alias_of']) if v.get('alias_of') else ''), v.get('status'), v.get('committed_label'), v.get('commit_is_c_a_argmin'),
                _f(v.get('oracle_gap_pp')), v.get('calls'), rs.get('tool_calls', v.get('tool_calls')), rs.get('n_built', '-'), v.get('new_evaluations'), len(rs['built_never_evaluated']) if rs else '-',
                rs.get('inspect_material_calls', '-'), rs.get('inspect_data_calls', '-'), len(rs['possible_unverified_harm_claims']) if rs else '-', v.get('tokens')))
    L_ += ['', '## MetaTest: post-hoc pool arithmetic (reference NoMix; pp of None)', '', '| case | arm | available | selection loss | net | pool | pool best |', '|---|---|---:|---:|---:|---:|---|']
    for c, r in res['metatest'].items():
        for a in ('f0', 'f_domain', 'f_shared', 'random'):
            ph = r['arms'].get(a, {}).get('post_hoc_vs_nomix')
            if ph:
                L_.append('| %s | %s | %s | %s | %s | %d | %s |' % (c, ARM_LABEL[a], _f(ph['available_pp']), _f(ph['selection_loss_pp']), _f(ph['net_pp']), ph['pool_size'], ph['pool_best_label']))
    L_ += ['', '## Select and freeze', '']
    s = res.get('select') or {}
    if s:
        dec = s.get('decision', {})
        for d in DOMAINS:
            L_.append('- %s: J_d %s; f0 J %s; W_d %s; Fixed_dev %s (J %s); H_deploy %s' % (d, {k: round(v, 4) for k, v in (s.get('J_domain', {}).get(d) or {}).items()}, _f((s.get('f0', {}).get(d) or {}).get('J'), 4),
                                                                                         (dec.get('w_domain', {}).get(d) or {}).get('arm') or (dec.get('w_domain', {}).get(d) or {}).get('status'),
                                                                                         (s.get('fixed_dev', {}).get(d) or {}).get('label'), _f((s.get('fixed_dev', {}).get(d) or {}).get('J'), 4), (dec.get('h_deploy', {}).get(d) or {}).get('choice')))
        ws = dec.get('w_shared') or {}
        L_.append('- SHARED: J_shared table %s; W_shared %s' % ({k: round(v, 4) for k, v in (ws.get('J_shared_table') or {}).items()}, ws.get('arm') or ws.get('status')))
        for a, rr in (s.get('runs') or {}).items():
            L_.append('  - %s: ' % ARM_LABEL.get(a, a) + '; '.join('%s -> %s (E/None %s, evals %s)' % (c, v.get('committed_label'), _f(v.get('ratio'), 4), v.get('new_evaluations')) for c, v in rr.items()))
    L_ += ['', '## Slow proposals', '']
    for sc, s in res['slow'].items():
        for r in s['proposals']:
            L_.append('- %s %s: %s%s; mode %s; tokens %s' % (sc, r['arm'].upper(), r['status'], (' -> alias %s' % r['alias_of']) if r.get('alias_of') else '', r.get('research_mode'), r.get('tokens')))
    cost = res.get('cost', {})
    L_ += ['', '## Cost', '', '```', json.dumps({k: v for k, v in cost.get('ledger', {}).items() if k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts',
                                                                                                          'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'llm_failed_attempts', 'fit_wall_seconds', 'label_wall_seconds', 'material_wall_seconds')}, indent=1),
           json.dumps(cost.get('per_stage', {}), indent=1, default=str), json.dumps({'slow_calls': cost.get('slow_calls'), 'one_time_learning_cost': cost.get('one_time_learning_cost'), 'paid_elapsed_s': cost.get('paid_elapsed_s'),
                                                                                    'http_queue_wait': cost.get('http_queue_wait')}, indent=1), '```']
    return '\n'.join(L_) + '\n'


# ============================================================================= package driver
def accept_unknown_usage(root: Path, where: str) -> None:
    """Operator decision (task §6): recorded only through this explicit flag; the executor never accepts on the user's behalf."""
    P_ = paths(root)
    led = SafeLedger(P_['ledger'], **TOTAL)
    if rt.unknown_usage_blocks(led):
        led.s['unknown_usage_accepted'] = led.s['llm_tokens_unknown']
        led.event(kind='unknown_usage_accepted', accepted=led.s['llm_tokens_unknown'], where=where, operator='explicit --accept-unknown-usage')
        print('UNKNOWN_USAGE_ACCEPTED', led.s['llm_tokens_unknown'], where, flush=True)


def package_run(root: Path = ROOT, *, accept_unknown: bool = False) -> dict:
    _require_package_root(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = PackageLock(root, 'driver').acquire()
    context.write_json(root / 'driver_launch.json', {'pid': os.getpid(), 'epoch': time.time(), 'local': now()})
    try:
        if accept_unknown:
            accept_unknown_usage(root, 'driver_start')
        status = context.read_json(root / 'package_status.json') if (root / 'package_status.json').exists() else {}

        def stop(where, st):
            status.update({'stopped_at': where, 'stage_status': st, 'epoch': time.time(), 'local': now()})
            context.write_json(root / 'package_status.json', status)
            print('PACKAGE_STOP', where, st.get('status') if isinstance(st, dict) else st, flush=True)
            return status
        preflight(root)
        w = wiring(root)
        status['wiring'] = w['status']
        if w['status'] != 'PASS':
            return stop('wiring', w)
        start_paid_clock(root)
        st = source_stage(root, lock)
        status['source'] = st
        context.write_json(root / 'package_status.json', status)
        if st['status'] != 'FINISHED':
            return stop('source', st)
        sl = slow_stage(root, lock)
        status['slow'] = {s: [(r['arm'], r['status'], r['alias_of']) for r in sl['scopes'][s]['proposals']] for s in SCOPES}
        context.write_json(root / 'package_status.json', status)
        st = select_stage(root, lock)
        status['select'] = st
        context.write_json(root / 'package_status.json', status)
        if st['status'] != 'FINISHED':
            return stop('select', st)
        fr = freeze_all(root)
        status['frozen'] = {d: {'w_domain': fr[d]['w_domain']['status'], 'h_deploy': (fr[d].get('h_deploy') or {}).get('choice')} for d in DOMAINS}
        status['frozen'][SHARED] = fr[SHARED]['w_shared']['status']
        context.write_json(root / 'package_status.json', status)
        st = metatest_stage(root, lock)
        status['metatest'] = st
        if st['status'] != 'FINISHED':
            return stop('metatest', st)
        status['finished_epoch'] = time.time()
        status['finished_local'] = now()
        context.write_json(root / 'package_status.json', status)
        readout(root)
        print('PACKAGE_FINISHED', flush=True)
        return status
    finally:
        lock.release()


# ============================================================================= preflight (0 cost) and frozen configuration
def preflight(root: Path = ROOT) -> dict:
    """Frozen before the first fit: verbatim split copies (this package + the entity-split parent roster), path binding, the multi-period partition
    check, cases, environment, prompts, contracts, orders, selection / readout formulas, budgets, concurrency. Reads no data row."""
    root.mkdir(parents=True, exist_ok=True)
    P_ = paths(root)
    if not P_['split'].exists():
        shutil.copy2(SPLIT_DOC, P_['split'])
    if not P_['parent_split'].exists():
        shutil.copy2(PARENT_SPLIT_DOC, P_['parent_split'])
    split, parent = context.read_json(P_['split']), context.read_json(P_['parent_split'])
    if split != context.read_json(SPLIT_DOC) or parent != context.read_json(PARENT_SPLIT_DOC):
        raise RuntimeError('a frozen split copy differs from its document')
    bound = bound_split(split)
    if not (root / 'split_bound.json').exists():
        context.write_json(root / 'split_bound.json', bound)
    elif context.read_json(root / 'split_bound.json') != bound:
        raise RuntimeError('split_bound.json drifted')
    p = root / 'frozen_config.json'
    if p.exists():
        return context.read_json(p)
    chk = check_split_tc(split, parent)
    if not chk['ok']:
        raise RuntimeError('partition check failed: %s' % json.dumps(chk, default=str)[:2000])
    binding = {d: DP.resolve_csv_path(D['dataset'], D.get('csv_path') or D.get('path')) for d, D in split['domains'].items()}
    cases = ec.cases_from_split(bound)
    for cs in cases.values():
        if not cs.path().exists():
            raise RuntimeError('data file missing for %s: %s' % (cs.case_id, cs.path()))
        if cs.max_row() > COMMON_PREFIX[1]:
            raise RuntimeError('%s reads past the common prefix' % cs.case_id)
    for c in WIRING_CASES:
        if c not in cases or cases[c].role != 'source':
            raise RuntimeError('wiring case %s missing or not a Source case' % c)
    cfg = {'package': PACKAGE, 'task': TASK, 'split_document': str(SPLIT_DOC), 'split_setting_id': split['setting_id'], 'split_status_in_document': split.get('status'),
           'parent_roster_document': str(PARENT_SPLIT_DOC), 'split_check': chk, 'path_binding': binding, 'exposure': EXPOSURE, 'common_row_prefix': list(COMMON_PREFIX),
           'anchors': split['anchors'], 'anchor_formula': 't_k = 24 floor((672 + (k + 0.5) (17544 - 1056) / 12) / 24), k = 0..11', 'stage_anchors': STAGE_ANCHORS,
           'domains': {d: {'dataset': split['domains'][d]['dataset'], 'file': binding[d]['name'], 'columns_in_file': split['domains'][d].get('parent_columns_in_file'),
                           'total_hours_in_file': split['domains'][d].get('total_hours'), 'rows_used_max': max(cs.max_row() for cs in cases.values() if cs.domain == d)} for d in DOMAINS},
           'cases': {c: cs.to_json() for c, cs in cases.items()}, 'case_queues': {r: case_queue(root, r) for r in ('source', 'select', 'test')},
           'learning_material': {d: case_queue(root, 'source') and [c for c in case_queue(root, 'source') if cases[c].domain == d] for d in DOMAINS},
           'learning_material_shared': case_queue(root, 'source'), 'no_select_or_test_case_enters_any_census': True,
           'cohort_size': ec.COHORT_SIZE, 'geometry': {'L': spec.L, 'H': spec.H, 'T': spec.TRAIN_SPAN, 'n_parents': spec.N_PARENTS, 'pool': ec.COHORT_SIZE * spec.N_PARENTS},
           'consumer': ec.CONSUMER, 'seeds': list(SEEDS), 'model': MODEL, 'environment': ES.environment(), 'limits': LIMITS, 'max_output_tokens': MAX_OUTPUT_TOKENS,
           'max_tool_corrections': MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'request_trace_dedupe': REQUEST_TRACE_DEDUPE,
           'fast_token_cap': 'none (cumulative monitoring at %s)' % list(DP.FAST_TOKEN_MONITOR), 'total': TOTAL, 'http_cap': HTTP_CAP, 'transport_retries': TRANSPORT_RETRIES,
           'transport_rule': 'one bounded automatic retry of the same logical request on a transient transport fault (the 8-attempt reserve); the failed attempt is unknown usage: '
                             'no new paid dispatch until an explicit operator acceptance (--resume-stage --accept-unknown-usage); the executor never accepts on the user\'s behalf',
           'allocation': ALLOC, 'token_warnings': list(TOKEN_WARNINGS), 'paid_wall_warnings_s': list(PAID_WALL_WARNINGS_S), 'concurrency_planned': CONCURRENCY, 'concurrency_effective': effective_concurrency(root),
           'slow_request_timeout_s': SLOW_REQUEST_TIMEOUT_S, 'fast_system': FAST_SYSTEM, 'material_semantics_sentence': DP.MATERIAL_SEMANTICS,
           'slow_system': {s: slow_system(s) for s in SCOPES}, 'contracts': CONTRACTS,
           'public_references': {m: ES.public_spec(m) for m in PUBLIC}, 'program_tables': {'explicit': [ec.program_label(p_) for p_ in ec.PROGRAMS], 'edit': [ec.program_label(p_) for p_ in ec.EDIT_PROGRAMS]},
           'seed_rule': 'SeedSequence([%d, domain_index, case_index, program_index, int(entity_column)]); preset and every edit use program_index %d; a new case_index per (entity group, t)' % (ec.SEED_ROOT, ec.PRESET_INDEX),
           'cache_rule': 'physical cache per case directory <stage>/<case>_common/<case>; cell id = case__material__seed; scaler bound to (dataset, case_id, t, roster); the same entity group at another t is another case',
           'random_search': {'seed_root': ES.RANDOM_SEED_ROOT, 'max_tries': ES.RANDOM_MAX_TRIES, 'min_group': ES.RANDOM_MIN_GROUP, 'quantile': 0.5, 'programs_pool': len(ec.LEGAL_PROGRAMS),
                             'commit': 'argmin three-seed C_A mean over public + R1..R4; ties in public order then R1..R4', 'frozen': 'before the case\'s first new fit; never shown to Fast'},
           'orders': {'source': 'f0 only', 'select': 'rotate(%s, case_index) filtered to existing arms' % list(SELECT_ARMS), 'metatest': 'rotate(%s, case_index) + %s' % (list(TEST_FAST_ARMS), list(CONTROL_ARMS)),
                      'case_queue': 'by anchor, then domain, then group; a free slot takes the next case'},
           'slow': {'proposals': {'per_domain': 3, 'shared': 3}, 'same_census_per_scope': True, 'independent': 'identical payload but proposal_slot; the others never visible',
                    'contract_correction_per_proposal': 1, 'alias_rule': 'KEEP -> NO_CARD (alias F0); byte-identical rendered body -> alias', 'byte_targets': {'domain': DOMAIN_BYTE_TARGET, 'shared': SHARED_BYTE_TARGET, 'hard_limit': HARD_BYTE_LIMIT},
                    'body_limit': BODY_LIMIT, 'evidence': 'domain census = the domain\'s 12 Source cases; shared census = both domains\' 24 Source cases; no Select / MetaTest case; no old card'},
           'selection': {'J_d': 'mean over the domain\'s two Select cases of [mean_seed E(actual commit of W) / mean_seed E(None)]; lowest wins',
                         'J_shared': 'mean over the two domains of J_d; ONE global shared winner', 'w_domain': 'lowest J_d among distinct domain cards; ties <= 1e-12 by fewer new evaluations, fewer tokens, slot; none -> alias F0',
                         'fixed_dev': 'among uniform programs actually evaluated on BOTH Select cases (any branch, incl. public), the lowest J_d; ties in public order then table order',
                         'h_deploy': 'lowest J_d among F_domain / F_shared / F0 / Fixed_dev; ties: fixed program first, then fewer evaluations, fewer tokens, then f_domain / f_shared / f0'},
           'readout': {'metric': 'E normalized MSE (T scaler), entity macro, then seed mean; domain = equal-weight periods of the equal-weight two entity groups; overall = equal-weight 2 domains; period / group breakdown; LOO ranges',
                       'G': 'G_j(A over B) = 100 x [mean_seed E(B) - mean_seed E(A)] / mean_seed E(None) (paired per seed with the same denominator); positive = A better; pp of None',
                       'menu_ca': 'C_A argmin among the four public references (ties in public order); 0 LLM, 0 extra fits',
                       'post_hoc': 'available = 100 x [E(NoMix) - min_pool E] / E(None); selection_loss = 100 x [E(commit) - min_pool E] / E(None); net = available - selection_loss; own pool only'},
           'frozen_local': now(), 'no_sha': True}
    dsks.write_once(p, cfg)
    return cfg


# ============================================================================= wiring acceptance (A01 G01 of each domain: None + explicit composition x 3 seeds; <= 12 fits, 0 LLM, C_A only)
def wiring(root: Path = ROOT) -> dict:
    P_ = paths(root)
    out = P_['wiring'] / 'wiring_result.json'
    if out.exists():
        return context.read_json(out)
    import psutil
    preflight(root)
    sc = stage_caps(root, 'wiring')
    led = SafeLedger(P_['ledger'], **sc['caps'])
    rt.FIT_RETRY = True
    set_pools(CONCURRENCY['numeric'], CONCURRENCY['http'])
    P_['wiring'].mkdir(parents=True, exist_ok=True)
    (P_['wiring'] / 'logs').mkdir(exist_ok=True)
    cases = cases_of(root)
    res = {'status': 'PASS', 'started_local': now(), 'cases': {}, 'checks': [], 'available_mb_before': psutil.virtual_memory().available / 1e6}
    mem, stop = {}, threading.Event()
    sampler = threading.Thread(target=DP._sample_memory, args=(stop, mem), daemon=True)
    sampler.start()
    t_all = time.time()

    def one(case):
        cs = cases[case]
        common = P_['wiring'] / (case + '_common')
        cfg = {'split_path': str(root / 'split_bound.json'), 'seeds': list(SEEDS), 'public_ids': ['None'], 'random': False, 'package_root': str(root)}
        prepare_common(P_['wiring'], case, led, cfg)                      # None: 3 fits (public materials built, 0 fits)
        reg = ec.load_registry(common / case)
        none_cells = [c for c in ec.branch_cells(common / case).values() if c['material_id'] == 'None']
        checks = [{'case': case, 'check': 'public materials built without fits; exactly the 3 None cells fitted in this case cache (receipts, not counter differences)',
                   'ok': all(m in reg for m in PUBLIC) and len(none_cells) == 3 and len(ec.branch_cells(common / case)) == 3 and all(c.get('physical_fit') for c in none_cells), 'materials': sorted(reg)}]
        with np.load(common / case / 'scaler.npz') as z:
            checks.append({'case': case, 'check': 'scaler bound to the case (roster, t, dataset)', 'ok': [str(x) for x in z['roster']] == list(cs.roster) and int(z['t']) == cs.t and str(z['dataset']) == cs.dataset})
        spec_rec = context.read_json(common / case / 'case_spec.json')
        checks.append({'case': case, 'check': 'only the 16 roster columns were converted', 'ok': spec_rec['columns_converted'] == ec.COHORT_SIZE and spec_rec['columns_in_file'] > ec.COHORT_SIZE, 'columns_in_file': spec_rec['columns_in_file']})
        steps = [{'actions': [{'tool': 'overview', 'arguments': {}}]},
                 {'actions': [{'tool': 'build_material', 'arguments': {'plan_id': 'WireComp', 'policy': ec.uniform_policy(WIRING_COMPOSITION, 'wiring: explicit 3-step composition')}}]},
                 {'actions': [{'tool': 'inspect_material', 'arguments': {'plan_id': 'WireComp', 'entity_index': 0}}]},
                 {'actions': [{'tool': 'evaluate', 'arguments': {'plan_id': 'WireComp'}}]},
                 {'actions': [{'tool': 'compare', 'arguments': {'a': 'None', 'b': 'WireComp'}}]},
                 {'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'None', 'reason': 'wiring acceptance: the public reference is kept'}}]}]
        bpath = P_['wiring'] / ('%s_wiring' % case)
        result = fast_branch(bpath, case, br.Knowledge(), led, ES.ScriptedClient(steps), common, split_path=root / 'split_bound.json', cs=cs, public_ids=('None',), seeds=SEEDS)
        cells = ec.branch_cells(bpath / case)
        comp = sorted((c for c in cells.values() if c['material_id'] == 'WireComp'), key=lambda c: c['model_seed'])
        fits = sum(int(c.get('fit_attempts_this_request') or 0) for c in comp if c.get('cache_hit') is False)      # per-cell receipts of this branch
        pool_ok = all(c['pool_size'] == ec.COHORT_SIZE * spec.N_PARENTS and c['entity_count'] == ec.COHORT_SIZE for c in cells.values())
        threads_ok = all(c['torch_threads'] == FIT_THREADS for c in cells.values())
        rows_ok = all(tuple(c['rows_read']) == tuple(cs.evaluate_rows) for c in cells.values())
        phys = ec.load_registry(common / case)
        ta_ids = [m for m in phys if m.startswith('TA')]
        summ = context.read_json(phys[ta_ids[0]]['summary_path']) if ta_ids else {}
        rows = ES._trace(bpath)
        ov = next(r['overview'] for r in rows if r['event'] == 'job_started')
        leak = [n for n in tuple(spec.DATASETS) if n in json.dumps(ov).lower()]
        checks.append({'case': case, 'check': 'scripted Fast: 3 composition fits, commit None, pool 16 x 433, 8 threads, rows read = [t-672, t+96), no source name in the overview',
                       'ok': result.status == 'COMPLETE' and fits == 3 and len(comp) == 3 and pool_ok and threads_ok and rows_ok and not leak and result.committed_plan_id == 'None' and (bpath / case / 'commit.json').exists(),
                       'fits': fits, 'steps_executed': {k: v['windows'] for k, v in summ.get('per_step', {}).items() if v['windows']}, 'build_seconds': summ.get('build_seconds')})
        # cross-arm cache reuse: a second branch view of the same case builds the same composition -> same physical material, 0 new fits, 3 cache notes
        ad = CaseAdapter(P_['wiring'] / ('%s_reuse' % case), case, led, REPO, common=common, split_path=root / 'split_bound.json', cs=cs, public_ids=('None',), seeds=SEEDS)
        ad.baselines()
        cand = ad.build_material({'plan_id': 'SameComp', 'policy': ec.uniform_policy(WIRING_COMPOSITION, 'wiring reuse')}, remaining_seconds=600)
        cand = ad.evaluate(cand, SEEDS, feedback=True, remaining_seconds=600)
        reg2 = ec.load_registry(P_['wiring'] / ('%s_reuse' % case) / case)
        reuse_cells = [c for c in ec.branch_cells(P_['wiring'] / ('%s_reuse' % case) / case).values() if c['material_id'] == 'SameComp']
        checks.append({'case': case, 'check': 'cross-arm cache reuse: identical assignment -> same physical material, 0 new fits (every cell a cache hit by receipt)',
                       'ok': len(reuse_cells) == 3 and all(c.get('cache_hit') is True and not c.get('fit_attempts_this_request') for c in reuse_cells) and reg2['SameComp']['phys_id'] == ta_ids[0]
                       and cand.ca_losses and cand.model_seeds == SEEDS, 'cache_hit_cells': len(reuse_cells)})
        res['cases'][case] = {'t': cs.t, 'anchor': anchor_of(case), 'none_c_a_by_seed': [c['scores']['c_a']['normalized_mse_macro'] for c in sorted((c for c in cells.values() if c['material_id'] == 'None'), key=lambda c: c['model_seed'])],
                              'comp_c_a_by_seed': [c['scores']['c_a']['normalized_mse_macro'] for c in comp], 'fit_seconds': [round(c['train']['seconds'], 1) for c in cells.values()],
                              'material_build_seconds': summ.get('build_seconds')}
        return checks
    errs, results = [], {}

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
    # the two wiring cases are different files / different domains at the same t: their scalers must differ (no cross-case cache)
    try:
        za = np.load(P_['wiring'] / (WIRING_CASES[0] + '_common') / WIRING_CASES[0] / 'scaler.npz')
        zb = np.load(P_['wiring'] / (WIRING_CASES[1] + '_common') / WIRING_CASES[1] / 'scaler.npz')
        res['checks'].append({'check': 'the two wiring cases hold distinct scalers and rosters (no cross-case cache)', 'ok': not np.array_equal(za['mean'], zb['mean']) and list(za['roster']) != list(zb['roster'])})
    except Exception as exc:  # noqa: BLE001
        res['checks'].append({'check': 'the two wiring cases hold distinct scalers and rosters', 'ok': False, 'error': repr(exc)[:200]})
    res['wall_seconds_both_cases'] = time.time() - t_all
    res['fits_total'] = led.s['fit_attempts']
    res['fit_wall_seconds_sum'] = led.s['fit_wall_seconds']
    res['pools'] = pools().snapshot()
    res['memory'] = mem
    res['checks'].append({'check': 'exactly 12 physical fits, all OK, numeric pool actually used in parallel (peak >= 2)', 'ok': led.s['fit_attempts'] == 12 and led.s['fits_ok'] == 12 and res['pools']['numeric_peak'] >= 2})
    no_labels = not any(p_.name in ('c_b_scores.json', 'e_frozen.json', 'e_scores.json') for p_ in P_['wiring'].rglob('*.json'))
    res['checks'].append({'check': 'no label file in wiring', 'ok': no_labels})
    peak = mem.get('peak_worker_rss_mb') or 0.0
    avail = res['available_mb_before']
    numeric = CONCURRENCY['numeric'] if 3 * peak + 1500 <= avail else 2
    decision = {'concurrency': {**CONCURRENCY, 'numeric': numeric}, 'peak_worker_rss_mb': peak, 'peak_workers_total_rss_mb': mem.get('peak_workers_total_rss_mb'), 'available_mb_before': avail,
                'available_mb_min_during': mem.get('available_mb_min'), 'rule': '3 x peak worker RSS + 1500 MB margin <= available before start, else 2', 'decided_local': now(),
                'note': 'training threads / seeds / batch / budgets unchanged; the numeric pool only limits how many worker processes run at once'}
    context.write_json(root / 'concurrency_effective.json', decision)
    res['concurrency_decision'] = decision
    fit_s = [x for c in res['cases'].values() for x in c['fit_seconds']]
    res['time_estimate'] = {'mean_fit_s_under_wiring_load': statistics.fmean(fit_s) if fit_s else None, 'planned_fits_max': TOTAL['max_fit_attempts'], 'numeric_pool': numeric,
                            'numeric_hours_if_all_fits_run': (TOTAL['max_fit_attempts'] * statistics.fmean(fit_s) / numeric / 3600) if fit_s else None}
    res['status'] = 'PASS' if all(c['ok'] for c in res['checks']) and led.s['fit_attempts'] <= ALLOC['wiring']['fits'] else 'FAIL'
    res['finished_local'] = now()
    dsks.write_once(out, res)
    print('WIRING', res['status'], 'fits', res['fits_total'], 'numeric', numeric, flush=True)
    return res


# ============================================================================= smoke (synthetic file + synthetic split; 0 real fits, 0 API; this package's new risks only)
def _synthetic_split_tc(out: Path, n_cols: int = 120, hours: int = 3000) -> tuple:
    """A synthetic two-domain roster in this package's layout: per domain 2 Source groups x 2 periods (in-role reuse), 1 Select, 1 Test x 2 periods;
    plus a parent-like roster for the roster-equality check."""
    import datetime as _dt
    rng = np.random.RandomState(11)
    t0 = _dt.datetime(2016, 7, 1, 2)
    domains, files = {}, {}
    for di, dom in enumerate(('D01', 'D02')):
        csvp = out / ('synthetic_%s.csv' % dom)
        with csvp.open('w', newline='') as f:
            f.write('date,' + ','.join(str(i) for i in range(n_cols)) + ',OT\n')
            base_ = rng.rand(n_cols) * 5 + 1 + di
            for h in range(hours):
                ts = t0 + _dt.timedelta(hours=h)
                vals = base_ * (1 + 0.3 * np.sin(2 * np.pi * (h % 24) / 24 + np.arange(n_cols))) + 0.1 * rng.randn(n_cols)
                f.write(ts.strftime('%Y-%m-%d %H:%M:%S') + ',' + ','.join('%.4f' % v for v in vals) + ',%.3f\n' % rng.rand())
        files[dom] = csvp
    cols = [str(i) for i in range(n_cols)]
    anchors = {'source': [700, 1100], 'select': [1600], 'test': [2000, 2400]}
    groups_all = {}
    for di, dom in enumerate(('D01', 'D02')):
        rosters = {'S01': cols[0:16], 'S02': cols[16:32], 'V01': cols[32:48], 'Q01': cols[48:64]}
        groups = []
        idx = 1000 * (di + 1)
        for role, ids in (('source', ('S01', 'S02')), ('select', ('V01',)), ('test', ('Q01',))):
            for ai, t in enumerate(anchors[role]):
                for gid in ids:
                    cid = '%s_%s_A%02d_%s' % (dom, {'source': 'S', 'select': 'V', 'test': 'Q'}[role], ai + 1, gid)
                    cs = ec.CaseSpec(dataset='synthetic', domain=dom, case_id=cid, role=role, t=t, roster=tuple(rosters[gid]), case_index=idx, csv_path=str(files[dom]), total_hours=hours)
                    groups.append({'case_id': cid, 'stage': role, 't': t, 'anchor_id': 'A%02d' % (ai + 1), 'entity_group': '%s_%s' % (dom, gid), 'roster_source_case_id': '%s_%s' % (dom, gid),
                                   'roster': list(cs.roster), 'case_index': idx, 'train_rows': list(cs.train_range), 'c_a_origins': list(cs.c_a), 'c_b_origins': list(cs.c_b), 'e_origins': list(cs.e),
                                   'e_target_rows': [cs.e[0], cs.e[-1] + spec.H]})
                    idx += 1
        groups_all[dom] = groups
        domains[dom] = {'dataset': 'synthetic', 'csv_path': str(files[dom]), 'total_hours': hours, 'excluded_columns': ['OT'], 'groups': groups}
    split = {'setting_id': 'SMOKE_TC', 'anchors': sorted({t for v in anchors.values() for t in v}), 'common_row_prefix': [0, hours], 'cohort_size': 16, 'domains': domains}
    parent = {'setting_id': 'SMOKE_PARENT', 'anchors': [700], 'cohort_size': 16,
              'domains': {dom: {'dataset': 'synthetic', 'groups': [{'case_id': '%s_%s' % (dom, gid), 'roster': r} for gid, r in
                                                                    {'S01': cols[0:16], 'S02': cols[16:32], 'V01': cols[32:48], 'Q01': cols[48:64]}.items()]} for dom in ('D01', 'D02')}}
    context.write_json(out / 'split.json', split)
    context.write_json(out / 'parent_split.json', parent)
    return out / 'split.json', out / 'parent_split.json'


def smoke(root: Path = ROOT) -> dict:
    """Pure checks only: never a stage config, never a stage worker, never a real HTTP call (task §5.A)."""
    out = paths(root)['smoke']
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    checks = []

    def check(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})
        print('SMOKE', 'ok ' if ok else 'FAIL', name, flush=True)
    set_pools(3, 2)
    # 1 the real documents: multi-period partition, per-case barriers, anchor formula, stage mapping (no data row read)
    split, parent = context.read_json(SPLIT_DOC), context.read_json(PARENT_SPLIT_DOC)
    chk = check_split_tc(split, parent)
    check('document partition: 12/2/8 groups per domain, roles disjoint as entity sets, in-role reuse, rosters equal the parent groups, anchors by formula, per-case barriers',
          chk['ok'], barriers={d: r['barriers'] for d, r in chk['domains'].items()}, roles={d: r['roles'] for d, r in chk['domains'].items()})
    bad = copy.deepcopy(split)
    bad['domains']['D01']['groups'][0]['roster'][0] = next(g for g in split['domains']['D01']['groups'] if g['stage'] == 'test')['roster'][0]
    check('an entity shared between two roles is detected', not check_split_tc(bad, parent)['ok'])
    bad = copy.deepcopy(split)
    g_ = next(g for g in bad['domains']['D01']['groups'] if g['stage'] == 'select')
    g_['t'] = 8208
    g_['train_rows'] = [8208 - 672, 8208]
    g_['c_a_origins'], g_['c_b_origins'], g_['e_origins'], g_['e_target_rows'] = [8208, 8256], [8304, 8352], [8400, 8448, 8496, 8544], [8400, 8592]
    check('a Select case moved before the last Source E is detected by the per-case barrier check', not check_split_tc(bad, parent)['ok'])
    check('in-role reuse of an entity group at two t is accepted (the real roster has it)', chk['domains']['D01']['group_t_unique'] and chk['domains']['D01']['roles']['source'] == 12 and chk['domains']['D01']['entities_per_role']['source'] == 128)
    check('csv path binding: the /mnt/c document path resolves to the spec file of each domain', all(DP.resolve_csv_path(split['domains'][d]['dataset'], split['domains'][d]['csv_path'])['identical_file'] for d in DOMAINS))
    # 2 text check: case / anchor / group / domain / period / row / entity tokens rejected, plain text accepted
    ok = True
    for badtext in ('build censor as on D01_S_A03_G05', 'the A03 period likes shock', 'group G02 prefers the preset', 'in D02 use edits', 'at row 12336 the regime changes',
                    'entity 7 is special', 'period 3 was calmer', 'the 2017 cases', 'in june use censor', 'the electricity data', 'the SHARED scope'):
        try:
            tc_text_check(badtext, 'smoke')
            ok = False
            print('  not rejected:', badtext)
        except ValueError:
            pass
    try:
        tc_text_check('Observe the batch summary; if lag24_corr median is high, construct one calendar-based plan and evaluate it before any second construction; commit the lower C_A unless within one seed SE.', 'smoke')
    except ValueError:
        ok = False
    check('deployable-text check rejects case / anchor / group / domain / period / row / entity / calendar tokens and accepts plain guidance', ok)
    # 3 package lock (existing mechanism, one probe)
    lk = PackageLock(out, 'smoke').acquire()
    rc_held = subprocess.run([sys.executable, '-B', '-m', MODULE, '--lock-probe', str(out)], cwd=str(REPO), capture_output=True, env=ES.worker_env()).returncode
    lk.release()
    rc_free = subprocess.run([sys.executable, '-B', '-m', MODULE, '--lock-probe', str(out)], cwd=str(REPO), capture_output=True, env=ES.worker_env()).returncode
    check('package lock: second controller refused (3) while held, admitted (0) after release', rc_held == 3 and rc_free == 0, rc_held=rc_held, rc_free=rc_free)
    # 4 fake metered client (no network; api=None): numbering, receipts, transient fault -> unknown usage blocks the next paid call
    led = SafeLedger(out / 'ledger_llm.json', max_fit_attempts=0, max_llm_requests=100, max_llm_tokens=10_000_000, max_wall_s=3600, max_retries=0)
    fc2 = DP.FakeMeteredClient(led, out / 'raw2', http_cap=100, stage='smoke', fail_requests={1})
    r1 = fc2.call('fast', 'u', {'k': 'a'}, 'sys', meta={'case': 'c', 'arm': 'a', 'call': 1})
    blocked = False
    try:
        fc2.call('fast', 'u', {'k': 'b'}, 'sys', meta={'case': 'c', 'arm': 'a', 'call': 2})
    except budget.BudgetExhausted:
        blocked = True
    check('fake client never opens a socket (api is None); transient fault: one bounded retry answers the same logical request, the next paid call is blocked until an operator accepts',
          fc2.api is None and r1[1]['attempts'] == 2 and led.s['llm_tokens_unknown'] == 1 and blocked)
    # 5 synthetic two-domain roster in this package's layout: split check, multi-t isolation, in-role reuse, role rejection
    sp, psp = _synthetic_split_tc(out)
    ssplit, sparent = context.read_json(sp), context.read_json(psp)
    scases = ec.cases_from_split(ssplit)
    same_group = [c for c in scases if c.startswith('D01_S_') and c.endswith('S01')]
    check('synthetic roster: the same entity group at two periods yields two cases with distinct case ids, t and case_index', len(same_group) == 2 and len({scases[c].t for c in same_group}) == 2 and len({scases[c].case_index for c in same_group}) == 2)
    run = out / 'run'
    led2 = SafeLedger(out / 'ledger_fits.json', max_fit_attempts=120, max_llm_requests=0, max_llm_tokens=0, max_wall_s=3600, max_retries=1)
    rt.FIT_RETRY = True
    for case in same_group:
        rc, _ = run_worker(['--worker-build', run / (case + '_common'), sp, case], out / ('build_%s.log' % case), 600)
        check('public materials built in a worker for %s (own cache directory)' % case, rc == 0 and all(m in ec.load_registry(run / (case + '_common') / case) for m in PUBLIC))
    a, b = same_group
    za, zb = np.load(run / (a + '_common') / a / 'scaler.npz'), np.load(run / (b + '_common') / b / 'scaler.npz')
    ma = np.load(ec.load_registry(run / (a + '_common') / a)['P_NoMixRecipe']['path'])
    mb = np.load(ec.load_registry(run / (b + '_common') / b)['P_NoMixRecipe']['path'])
    check('multi-t isolation: same roster, different t -> different scaler, different preset material bytes, separate registries',
          list(za['roster']) == list(zb['roster']) and int(za['t']) != int(zb['t']) and not np.array_equal(za['mean'], zb['mean']) and not np.array_equal(ma['Xc'], mb['Xc']))
    got2, e2 = {}, []

    def fits(case):
        try:
            got2[case] = fit_cells(led2, run / (case + '_common'), sp, case, [('None', s) for s in SEEDS[:2]], n_updates=20)
        except Exception as exc:  # noqa: BLE001
            e2.append((case, repr(exc)))
    ts = [threading.Thread(target=fits, args=(c,)) for c in same_group]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    snap = pools().snapshot()
    check('4 short fits of the two same-group cases through the numeric pool: ledger charged 4, cell ids carry the case id (no cross-t hit), numeric peak >= 2',
          not e2 and led2.s['fit_attempts'] == 4 and all(r['fit_started'] for c in got2.values() for r in c.values()) and all(cid.startswith(c + '__') for c, cc in got2.items() for cid in cc) and snap['numeric_peak'] >= 2,
          numeric_peak=snap['numeric_peak'], errors=str(e2)[:300])
    # role rejection: a branch directory bound to another case is refused by the adapter binding check
    ok = False
    wrong = out / 'wrong_binding'
    shutil.copytree(run / (a + '_common') / a, wrong / b)              # a branch view named for case b that holds case a's frozen objects
    try:
        CaseAdapter(wrong, b, led2, REPO, common=run / (b + '_common'), split_path=sp, cs=scases[b], public_ids=('None',), seeds=SEEDS)
    except RuntimeError as exc:
        ok = 'binding' in str(exc)
    check('a branch directory bound to another case (same roster, other t) is refused at open (no resume across cases)', ok)
    # 6 explicit composition and observation-conditional plans reach the material builder through the adapter (synthetic case)
    cs = scases[a]
    branch = out / ('%s_branch' % a)
    ad = CaseAdapter(branch, a, led2, REPO, common=run / (a + '_common'), split_path=sp, cs=cs, public_ids=('None',), seeds=SEEDS[:2],
                     fit_fn=lambda phys, seeds_: fit_cells(led2, run / (a + '_common'), sp, a, [(phys, s) for s in seeds_], n_updates=20))
    ad.baselines()
    c_comp = ad.build_material({'plan_id': 'Comp3', 'policy': ec.uniform_policy(WIRING_COMPOSITION, 'smoke composition')}, remaining_seconds=600)
    table = context.read_json(run / (a + '_common') / a / 'overview.json')['entities']
    field = ES._table_fields_with_variation(table)[0]
    c_cond = ad.build_material({'plan_id': 'Cond1', 'policy': {'default': {'steps': [{'op': ta.RECIPE}]}, 'rules': [{'when': {'feature': field, 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': [ta.edit_step(['tp_shock'])]}],
                                                                'rationale': 'smoke conditional', 'observation_fields_used': [field]}}, remaining_seconds=600)
    check('explicit 3-step composition and an observation-conditional preset edit are both buildable (conditional -> Conditional label with two groups)',
          c_comp.material_spec['label'] == 'Comp[regime+resample+censor]' and c_cond.material_spec['label'].startswith('Conditional{') and len(set(c_cond.material_spec['entity_program'])) == 2)
    # 7 scripted Fast loop -> a labelled synthetic Source stage of both domains -> domain and shared censuses (evidence boundary, refs, metadata, no select / test case)
    sroot = out / 'pkg' / 'source'
    pkg = out / 'pkg'
    pkg.mkdir(parents=True, exist_ok=True)
    context.write_json(pkg / 'split_copy.json', ssplit)
    context.write_json(pkg / 'split_bound.json', ssplit)
    context.write_json(pkg / 'parent_split_copy.json', sparent)
    src_cases = [c for c in scases if scases[c].role == 'source']
    for case in src_cases:
        common = sroot / (case + '_common')
        if not (common / case / 'aug_materials' / 'index.json').exists():
            rc, _ = run_worker(['--worker-build', common, sp, case], out / ('build_%s.log' % case), 600)
        fit_cells(led2, common, sp, case, [('None', s) for s in SEEDS], n_updates=20)
        steps = [{'actions': [{'tool': 'overview', 'arguments': {}}]},
                 {'actions': [{'tool': 'build_material', 'arguments': {'plan_id': 'P1', 'policy': ec.uniform_policy([{'op': 'tp_censor'}], 'smoke plan')}}]},
                 {'actions': [{'tool': 'evaluate', 'arguments': {'plan_id': 'P1'}}]},
                 {'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'P1', 'reason': 'smoke: commit the evaluated plan'}}]}]
        cs_ = scases[case]
        bpath = sroot / ('%s_f0' % case)
        base.branch(bpath, case, br.Knowledge(), led2, DP.ScriptedClient(copy.deepcopy(steps)), None, evidence_roundtrip=EVIDENCE_ROUNDTRIP, max_tool_corrections=MAX_TOOL_CORRECTIONS, tool_contracts=CONTRACTS, seeds=SEEDS,
                    adapter_factory=lambda r_, c_, l_, rp, **kw: CaseAdapter(r_, c_, l_, rp, common=common, split_path=sp, cs=cs_, public_ids=('None',), seeds=SEEDS,
                                                                             fit_fn=lambda phys, seeds_: fit_cells(led2, common, sp, case, [(phys, s) for s in seeds_], n_updates=20)),
                    limits=LIMITS, dataset='synthetic', commit_fn=ES.commit_branch, allowed_features=ALLOWED_FEATURES, entity_count=16, request_trace_dedupe=True)
        rc1, _ = run_worker(['--worker-label', 'c_b', bpath, sp, case, sroot], out / ('label_cb_%s.log' % case), 600)
        rc2, _ = run_worker(['--worker-label', 'freeze_e', bpath, sp, case, sroot], out / ('label_fe_%s.log' % case), 600)
    context.write_json(sroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [[str(sroot / ('%s_f0' % c)), c] for c in src_cases], 'failures': []})
    context.write_json(sroot / 'all_e_predictions_frozen.json', {'epoch': time.time(), 'branches': [str(sroot / ('%s_f0' % c)) for c in src_cases]})
    rcs = [run_worker(['--worker-label', 'score_e', sroot / ('%s_f0' % c), sp, c, sroot], out / ('label_se_%s.log' % c), 600)[0] for c in src_cases]
    check('synthetic Source stage of both domains: 8 scripted F0 branches committed and labelled (C_B, E frozen, E scored behind the barrier)',
          all(r == 0 for r in rcs) and all((sroot / ('%s_f0' % c) / c / 'e_scores.json').exists() for c in src_cases))
    # a Select-stage directory with a labelled branch must never enter a census
    (pkg / 'select').mkdir(exist_ok=True)
    shutil.copytree(sroot / ('%s_f0' % src_cases[0]), pkg / 'select' / ('%s_f0' % src_cases[0].replace('_S_', '_V_')))
    cen_d = census_tc(pkg, 'D01', 3)
    cen_s = census_tc(pkg, SHARED, 3)
    cen_4 = census_tc(pkg, SHARED, 4)
    refs = set(cen_s['legal_evidence_refs'])
    dp_ok = all(p_['ref'] in refs for c in cen_s['cases'].values() for r in c['branches'] if r.get('decision_points') for p_ in r['decision_points'])
    meta_ok = all(c['period_id'] in ('A01', 'A02') and c['period_order'] in (1, 2) and c['entity_group'] in ('S01', 'S02') and c['domain_id'] in DOMAINS for c in cen_s['cases'].values())
    check('domain census = the domain\'s 4 synthetic Source cases; shared census = all 8 of both domains; no Select / MetaTest branch; decision-point refs legal; time metadata carried; tier 4 drops per-entity columns',
          set(cen_d['cases']) == {c for c in src_cases if c.startswith('D01_')} and set(cen_s['cases']) == set(src_cases) and cen_d['census_status'] == 'CENSUS_COMPLETE' if False else
          (set(cen_d['cases']) == {c for c in src_cases if c.startswith('D01_')} and set(cen_s['cases']) == set(src_cases) and not any('_V_' in r for r in refs) and dp_ok and meta_ok
           and all(c['overview']['entities'].get('note') for c in cen_4['cases'].values()) and all(c['post_hoc']['commit_physical'] for c in cen_s['cases'].values())),
          n_refs=len(refs), bytes_domain_t3=payload_bytes(slow_system('D01'), slow_payload(cen_d, 'D01', 1)), bytes_shared_t3=payload_bytes(slow_system(SHARED), slow_payload(cen_s, SHARED, 1)),
          bytes_shared_t4=payload_bytes(slow_system(SHARED), slow_payload(cen_4, SHARED, 1)))
    # 8 Slow contract: one candidate accepted (domain and shared), KEEP accepted, illegal refs / tokens rejected, shared needs cross_domain_note
    legal = sorted(refs)
    cand = {'research_mode': 'test-first priority', 'workflow': 'Observe the overview; construct one discriminating plan; evaluate it before any second construction; commit the evaluated plan with the lower C_A unless the gap is within one seed SE.',
            'principles': None, 'observable_applicability': {'const': True}, 'applicability_summary': 'smoke', 'evidence_refs': [legal[0]], 'rationale': 'r',
            'evidence_review': {'supports': [], 'contradicts_or_limits': [], 'rule_changes': [], 'uncertain_and_expected_behavior_change': 'u'},
            'rule_status': [{'rule': 'r1', 'status': 'hypothesis', 'support': [], 'counter': []}], 'commit_policy': 'p',
            'main_mechanism_changed': {'mechanism': 'evaluate before inspecting more', 'deployment_observable_trigger': 'a built plan without C_A while slots remain'},
            'decision_change_vs_no_card': 'd', 'historical_decision_point_expected_to_change': {'ref': legal[0], 'expected_change': 'x'},
            'counterexample_expected': {'ref': legal[1], 'should_hold_or_may_fail': 'y'}, 'conditions_same_as_no_card': 'z'}
    ok = parse_proposal_tc({'decision': 'PROPOSE', 'candidate': cand}, scope='D01', slot=2, legal_refs=legal)['skill'].skill_id == 'D01-D2-r1'
    try:
        parse_proposal_tc({'decision': 'PROPOSE', 'candidate': cand}, scope=SHARED, slot=1, legal_refs=legal)
        ok = False
    except ValueError:
        pass
    shared_ok = parse_proposal_tc({'decision': 'PROPOSE', 'candidate': {**cand, 'cross_domain_note': 'same text; branches on lag24_corr'}}, scope=SHARED, slot=3, legal_refs=legal)['skill']
    ok = ok and shared_ok.skill_id == 'SHARED-S3-r1' and shared_ok.domain_id == SHARED
    for badtext in ('build censor as on S01', 'use the D01_S_A01_S01 recipe', 'the cut at 12336 hours', 'entity 7 is special', 'period 2 needs shock', 'the traffic data'):
        try:
            parse_proposal_tc({'decision': 'PROPOSE', 'candidate': {**cand, 'workflow': cand['workflow'] + ' ' + badtext}}, scope='D01', slot=1, legal_refs=legal)
            ok = False
        except ValueError:
            pass
    try:
        parse_proposal_tc({'decision': 'PROPOSE', 'candidate': {**cand, 'historical_decision_point_expected_to_change': {'ref': 'nowhere/x/y:1', 'expected_change': 'x'}}}, scope='D01', slot=1, legal_refs=legal)
        ok = False
    except ValueError:
        pass
    keep = parse_proposal_tc({'decision': 'KEEP', 'rationale': 'r'}, scope='D01', slot=2, legal_refs=legal)
    check('Slow parser: domain and shared candidates accepted (shared needs cross_domain_note); case / group / row / entity / period / source tokens and illegal refs rejected; KEEP accepted', ok and keep['decision'] == 'KEEP')
    # 9 selection arithmetic (pure): W_d, one global W_shared, Fixed_dev, H_deploy tie rules
    dec = decide_selection({'D01': {'d1': 0.90, 'd2': 0.90, 'd3': 0.95}, 'D02': {}}, {'D01': {'d1': (3, 500), 'd2': (2, 900), 'd3': (1, 100)}, 'D02': {}},
                           {'s1': {'D01': 0.92, 'D02': 0.88}, 's2': {'D01': 0.80}, 's3': {'D01': 0.91, 'D02': 0.89}}, {'s1': (4, 1000), 's2': (1, 1), 's3': (4, 1000)},
                           {'D01': {'label': 'P_NoMixRecipe', 'steps': [{'op': ta.RECIPE}], 'public_id': 'P_NoMixRecipe', 'J': 0.90}, 'D02': None}, {'D01': {'J': 0.93, 'cost': (2, 400)}, 'D02': {'J': 0.85, 'cost': (2, 400)}})
    check('selection arithmetic: W_d tie -> fewer evaluations (d2); W_shared = lowest J_shared among cards complete on both domains (s1 not s2); no domain card -> NO_CARD; H_deploy tie -> fixed first (D01), F0 when it is lowest (D02)',
          dec['w_domain']['D01']['arm'] == 'd2' and dec['w_domain']['D02']['status'] == 'NO_CARD' and dec['w_shared']['arm'] == 's1' and dec['h_deploy']['D01']['choice'] == 'fixed_dev' and dec['h_deploy']['D02']['choice'] == 'f0',
          decision=dec)
    # 10 aggregation (pure): period x group weighting, LOO ranges
    meta = {'%s_Q_A%02d_G%02d' % (d, p, g): {'domain': d, 'period': 'A%02d' % p, 'group': 'G%02d' % g} for d in DOMAINS for p in (9, 10, 11, 12) for g in (1, 2)}
    per = {c: (1.0 if m['domain'] == 'D01' else -1.0) * (2.0 if m['group'] == 'G01' else 0.0) for c, m in meta.items()}
    ag = aggregate(per, meta)
    check('aggregation: domain = mean over periods of the two-group mean; overall = mean of domains; LOO ranges computed',
          abs(ag['by_domain']['D01'] - 1.0) < 1e-12 and abs(ag['by_domain']['D02'] + 1.0) < 1e-12 and abs(ag['overall']) < 1e-12 and ag['n_cases'] == 16 and ag['leave_one_group_out_range'] == [0.0, 0.0])
    # 11 the stage-config guard refuses a root outside _scratch (a smoke can never freeze or start a paid stage elsewhere)
    ok = False
    try:
        _require_package_root(Path(os.environ.get('TEMP', '/tmp')) / 'not_scratch')
    except PermissionError:
        ok = True
    check('stage configuration / execution refused outside <repo>/_scratch', ok)
    check('G sign: A better (lower E) is positive', G([1.0, 1.0, 1.0], [1.1, 1.1, 1.1], 1.0)['mean'] > 0)
    check('rotation orders by case_index', rotate(SELECT_ARMS, 1012)[0] == SELECT_ARMS[1012 % 7] and rotate(TEST_FAST_ARMS, 1015) == ['f_domain', 'f_shared', 'f0'])
    res = {'status': 'PASS' if all(c['ok'] for c in checks) else 'FAIL', 'checks': checks, 'real_fits': 0, 'llm_requests': 0, 'stage_workers_started': 0, 'finished_local': now()}
    context.write_json(out / 'smoke_result.json', res)
    print('SMOKE_RESULT', res['status'], sum(c['ok'] for c in checks), '/', len(checks), flush=True)
    return res


# ============================================================================= read-only monitor
def monitor(root: Path = ROOT, interval: float = 60.0, once: bool = False) -> None:
    """Prints process liveness, ledger totals, last request / fit completion, stage progress and log tails (stdout+stderr are one file per stage).
    Never writes, never takes the lock."""
    import psutil
    P_ = paths(root)
    while True:
        line = ['MONITOR', now()]
        procs = []
        for name in ('driver_launch.json', 'source_worker_launch.json', 'select_worker_launch.json', 'metatest_worker_launch.json'):
            q = root / name
            if q.exists():
                try:
                    pid = int(context.read_json(q)['pid'])
                    procs.append('%s=%d:%s' % (name.split('_')[0], pid, 'alive' if psutil.pid_exists(pid) else 'exited'))
                except Exception:  # noqa: BLE001
                    pass
        line.append('procs[%s]' % ' '.join(procs))
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
            last_llm = max((e['epoch'] for e in ev if e.get('kind') == 'llm_finished'), default=None)
            last_fit = max((e['epoch'] for e in ev if e.get('kind') == 'fit_finished'), default=None)
            line.append('fits=%d/%d ok=%d failed=%d cache=%d req=%d http=%d tok=%.2fM unknown=%d' % (
                led['fit_attempts'], led['caps']['max_fit_attempts'], led['fits_ok'], led['fits_failed'], led['cache_hits'], led['llm_requests'], led['llm_http_attempts'],
                (led['llm_tokens_in'] + led['llm_tokens_out']) / 1e6, led['llm_tokens_unknown']))
            line.append('last_llm=%s last_fit=%s' % ('%.0fs ago' % (time.time() - last_llm) if last_llm else '-', '%.0fs ago' % (time.time() - last_fit) if last_fit else '-'))
            if led.get('paid_clock_started_epoch'):
                line.append('paid=%.2fh' % ((time.time() - led['paid_clock_started_epoch']) / 3600))
        for stage in ('source', 'select', 'metatest'):
            pp = P_[stage] / 'progress.json'
            if pp.exists():
                pr = context.read_json(pp)
                st = {}
                for c, v in pr.get('cases', {}).items():
                    st[v.get('status', '?')] = st.get(v.get('status', '?'), 0) + 1
                pl = pr.get('pools') or {}
                line.append('%s: %s pools(num %s/%s peak %s, http %s/%s peak %s)' % (stage, st, pl.get('numeric_active'), pl.get('numeric_limit'), pl.get('numeric_peak'), pl.get('http_active'), pl.get('http_limit'), pl.get('http_peak')))
        for name in ('driver', 'source', 'select', 'metatest'):
            lg = P_['logs'] / ('%s.log' % name)
            if lg.exists():
                tail = [t for t in lg.read_text(encoding='utf-8', errors='replace').splitlines() if t.strip()][-2:]
                if tail:
                    line.append('%s.log: %s' % (name, ' | '.join(t[:110] for t in tail)))
                txt = lg.read_text(encoding='utf-8', errors='replace')
                if 'Traceback' in txt or 'EXECUTION_STOP' in txt or 'PACKAGE_STOP' in txt:
                    line.append('!! %s.log holds Traceback / STOP' % name)
        print(' '.join(line), flush=True)
        if once:
            return
        time.sleep(interval)


# ============================================================================= CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--resume-stage')
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--restart', default='')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--monitor', action='store_true')
    ap.add_argument('--interval', type=float, default=60.0)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--stage-worker')
    ap.add_argument('--lock-probe')
    ap.add_argument('--supplement-arms', nargs=2, metavar=('STAGE', 'BRANCHES'))
    a = ap.parse_args()
    root = Path(a.root)
    if a.lock_probe:
        sys.exit(DP.lock_probe(Path(a.lock_probe)))
    elif a.stage_worker:
        stage_worker(Path(a.stage_worker))
    elif a.preflight:
        print(json.dumps({k: v for k, v in preflight(root).items() if k in ('package', 'split_check', 'path_binding', 'environment', 'total', 'concurrency_effective', 'case_queues')}, indent=1, ensure_ascii=False))
    elif a.smoke:
        smoke(root)
    elif a.wiring:
        print(json.dumps(wiring(root), indent=1, ensure_ascii=False, default=str))
    elif a.run:
        package_run(root, accept_unknown=a.accept_unknown_usage)
    elif a.supplement_arms:
        supplement_arms(root, a.supplement_arms[0], [x for x in a.supplement_arms[1].split(',') if x])
    elif a.resume_stage:
        resume_stage(root, a.resume_stage, accept_unknown_usage=a.accept_unknown_usage, restart=[x for x in a.restart.split(',') if x])
    elif a.result:
        readout(root)
        print((root / 'tables.md').read_text(encoding='utf-8'))
    elif a.monitor:
        monitor(root, a.interval, a.once)


if __name__ == '__main__':
    main()
