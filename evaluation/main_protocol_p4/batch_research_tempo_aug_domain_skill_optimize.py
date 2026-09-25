"""DEV-TEMPO-AUG-DOMAIN-SKILL-OPTIMIZE (docs/DEV_TEMPO_AUG_DOMAIN_SKILL_OPTIMIZE_TASK_2026-09-19.md): form a RD02 domain Skill from the
research processes AND the already-exposed late results (E) of nine earlier batches, try two cards on L4, revise once from L4's E, select the
best card against no-card on L5, freeze it, and compare card vs no-card Fast on L6 / L7 (frozen development replay).

What is new: Slow reads development_late (E) of the whitelisted Source batches and of the Practice / Select batches after their commits; the
current-batch Fast still sees only T, its own C_A and the frozen card. Everything else is reused: the recipe-edit adapter, physical caches and
workers of recipe_edit_pilot (imported as E), the Skill carrier / Slow call / compression / reservation helpers of workflow_learning_loop
(imported as LL), and the parent package's numerical workers, label stages and ledger (W).

  --preflight | --smoke | --wiring | --run | --resume-stage STAGE [--accept-unknown-usage] [--restart b1,b2] | --result
  workers (subprocess only): --stage-worker CFG
The controller never imports torch.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha.batch_base import budget, context, spec, tempo_aug as ta
from methods.ttha.batch_base import readiness as rd
from evaluation.main_protocol_p4 import batch_research_domain_skill as dsks
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_skill as W
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_learning_loop as LL
from evaluation.main_protocol_p4 import batch_research_tempo_aug_recipe_edit_pilot as E

REPO = W.REPO
ROOT = REPO / '_scratch' / 'dev_tempo_aug_domain_skill_optimize'
MODULE = 'evaluation.main_protocol_p4.batch_research_tempo_aug_domain_skill_optimize'
TASK = 'docs/DEV_TEMPO_AUG_DOMAIN_SKILL_OPTIMIZE_TASK_2026-09-19.md'
PACKAGE = 'DEV-TEMPO-AUG-DOMAIN-SKILL-OPTIMIZE'
EXPOSURE = 'EXPOSED_DEVELOPMENT_REPLAY'
DATASET, DOMAIN = W.DATASET, W.DOMAIN
SEEDS = W.SEEDS
PUBLIC = W.PUBLIC
P_ROOT = E.ROOT / 'pilot'                                            # recipe_edit_pilot stage root (read-only)
TR_ROOT = REPO / '_scratch' / 'dev_tempo_aug_temporal_feedback_replay'   # historical_late (read-only)
SOURCE_JOBS = ('RD02_S1', 'RD02_S2', 'RD02_V1', 'RD02_T2', 'RD02_T4', 'RD02_Q2', 'RD02_L1', 'RD02_L2', 'RD02_L3')
PRACTICE_JOB, SELECT_JOB, REPLAY_JOBS = 'RD02_L4', 'RD02_L5', ('RD02_L6', 'RD02_L7')
JOB_INDEX = {**W.JOB_INDEX, 'RD02_L5': 11, 'RD02_L6': 12, 'RD02_L7': 13}
JOB_T = {**W.JOB_T, 'RD02_L5': 21360, 'RD02_L6': 22560, 'RD02_L7': 24480}
HIST_LATE_JOBS = ('RD02_V1', 'RD02_T2', 'RD02_T4', 'RD02_Q2')
SOURCE_BRANCHES = ([(W.ROOT / 'source', '%s_no_skill' % j, j, 'hist_source', 'no-Skill Fast (explicit-composition DSL)') for j in ('RD02_S1', 'RD02_S2', 'RD02_V1', 'RD02_T2')]
                   + [(W.ROOT / 'select', '%s_no_skill' % j, j, 'hist_select', 'no-Skill Fast (explicit-composition DSL)') for j in ('RD02_T4', 'RD02_Q2')]
                   + [(W.ROOT / 'target', '%s_f_no_skill' % j, j, 'hist_target', 'no-Skill Fast (explicit-composition DSL)') for j in ('RD02_L1', 'RD02_L2', 'RD02_L3')]
                   + [(P_ROOT, '%s_f_edit' % j, j, 'edit_pilot', 'no-Skill Fast (recipe-edit DSL, the program space of this package)') for j in ('RD02_L1', 'RD02_L3')]
                   + [(P_ROOT, '%s_random_edit' % j, j, 'edit_pilot', 'random control (0 LLM): four frozen random local edits, delivery = argmin C_A') for j in ('RD02_L1', 'RD02_L3')]
                   + [(P_ROOT, '%s_ablation' % j, j, 'edit_pilot', 'diagnostic view (0 LLM): the seven single-component ablations; the preset recorded as reference delivery, no decision') for j in ('RD02_L1', 'RD02_L3')])
STAGE_NAMES = {str(W.ROOT / 'source'): 'hist_source', str(W.ROOT / 'select'): 'hist_select', str(W.ROOT / 'target'): 'hist_target', str(P_ROOT): 'edit_pilot'}
CACHE_FROM = {'RD02_L4': W.ROOT / 'target' / 'RD02_L4_common' / 'RD02_L4', 'RD02_L5': P_ROOT / 'RD02_L5_common' / 'RD02_L5',
              'RD02_L6': E.DL / 'test' / 'RD02_L6_common' / 'RD02_L6', 'RD02_L7': P_ROOT / 'RD02_L7_common' / 'RD02_L7'}
WIRING_JOB, WIRING_SEEDS, WIRING_PUBLIC, WIRING_JOB_INDEX = E.WIRING_JOB, E.WIRING_SEEDS, E.WIRING_PUBLIC, E.WIRING_JOB_INDEX
WIRING_CACHE = E.ROOT / 'wiring' / 'RD02_T1_common' / 'RD02_T1'
LIMITS, FAST_TOKEN_CAP, MAX_OUTPUT_TOKENS, MAX_TOOL_CORRECTIONS, EVIDENCE_ROUNDTRIP = E.LIMITS, E.FAST_TOKEN_CAP, E.MAX_OUTPUT_TOKENS, E.MAX_TOOL_CORRECTIONS, E.EVIDENCE_ROUNDTRIP
FAST_SYSTEM = E.FAST_SYSTEM_EDIT                                      # identical for every arm; only experiment_guidance differs
CONTRACTS = E.CONTRACTS_EDIT
BODY_LIMIT = LL.BODY_LIMIT                                            # 6000 (opt-in)
ALLOWED_FEATURES = W.ALLOWED_FEATURES
MODEL = W.MODEL
TOTAL = {'max_fit_attempts': 180, 'max_llm_requests': 165, 'max_llm_tokens': 5_300_000, 'max_wall_s': 8 * 3600, 'max_retries': 3}   # amendment_1_wall_cap.json: 4 h -> 8 h after the idle crash gap (user instruction)
HTTP_CAP = 169
ALLOC = {'wiring': {'fits': 3, 'requests': 0, 'tokens': 0}, 'formation': {'fits': 0, 'requests': 3, 'tokens': 800_000}, 'practice': {'fits': 36, 'requests': 32, 'tokens': 800_000},
         'revision': {'fits': 0, 'requests': 2, 'tokens': 500_000}, 'select': {'fits': 60, 'requests': 64, 'tokens': 1_600_000}, 'replay': {'fits': 78, 'requests': 64, 'tokens': 1_600_000}}
NUMERIC_WALL_S = 4 * 3600.
T975_DF2 = W.T975_DF2
TOL = 1e-12
DECISION_SUMMARY_SECONDS = 20 * 60


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'configs': root / 'stage_configs', 'logs': root / 'logs', 'wiring': root / 'wiring', 'smoke': root / 'smoke',
            'evidence': root / 'evidence', 'formation': root / 'formation', 'practice': root / 'practice', 'revision': root / 'revision', 'select': root / 'select', 'replay': root / 'replay'}


def ref(stage: str, branch: str, tail: str) -> str:
    return '%s/%s/%s' % (stage, branch, tail)


# ============================================================================= program naming (semantic, cross-package)
def program_label(assignment, public_id=None) -> str:
    """A cross-package name of a complete assignment: public ids; uniform programs by canonical steps ('Edit[-tp_x]', 'Comp[tp_a+tp_b]', 'Preset');
    entity-conditional plans as 'Conditional{...}' (never merged across batches)."""
    if public_id:
        return public_id
    if assignment == 'FixedMixup':
        return 'FixedMixup'
    if not isinstance(assignment, list) or not assignment:
        return 'unknown'
    if all(not s for s in assignment):
        return 'None'
    def one(steps):
        if not steps:
            return 'none'
        if steps == [{'op': ta.RECIPE}]:
            return 'Preset'
        if ta.is_edit_program(steps):
            return 'Edit[-%s]' % '-'.join(steps[0]['disabled_ops'])
        return 'Comp[%s]' % '+'.join(s['op'] for s in steps)
    labels = [one(s) for s in assignment]
    if len(set(labels)) == 1:
        return labels[0]
    groups = sorted(set(labels))
    return 'Conditional{%s}' % ', '.join('%s x%d' % (g, labels.count(g)) for g in groups)


def uniform_steps(assignment):
    if isinstance(assignment, list) and assignment and all(s == assignment[0] for s in assignment):
        return assignment[0]
    return None


# ============================================================================= census with development_late (whitelist §3.1)
def _e_allowed(stage_name: str, job: str, allowed: dict) -> bool:
    return job in allowed.get(stage_name, ())


def branch_evidence(b: tuple, tier, allowed_e: dict, seeds=SEEDS) -> tuple:
    """(record, refs, materials keyed by material key). Adds e_by_seed where the branch holds e_scores.json and (stage, job) is whitelisted for E."""
    stage_root, bname, job, sname, kind = b
    d = stage_root / bname
    res = context.read_json(d / 'branch_result.json') if (d / 'branch_result.json').exists() else None
    reg = W.load_registry(d / job)
    cells = W.branch_cells(d / job)
    cb = context.read_json(d / job / 'c_b_scores.json')['cells'] if (d / job / 'c_b_scores.json').exists() else {}
    e_ok = _e_allowed(sname, job, allowed_e)
    es = context.read_json(d / job / 'e_scores.json')['cells'] if (d / job / 'e_scores.json').exists() else {}
    if es and not e_ok:
        raise PermissionError('E present for a job outside the whitelist: %s/%s' % (sname, job))
    commit = context.read_json(d / job / 'commit.json') if (d / job / 'commit.json').exists() else None
    plan_to_key, materials = {}, {}
    for mid, r in reg.items():
        key = r.get('key') if mid not in PUBLIC else ('public:' + mid)
        plan_to_key[mid] = key
        asg = 'FixedMixup' if mid == 'FixedMixup' else (r.get('assignment') if mid not in PUBLIC else (W.public_spec(mid).get('programs', [[]]) * 12 if mid != 'None' else [[]] * 12))
        if mid in PUBLIC and mid in ('P_AmpResample', 'P_NoMixRecipe'):
            asg = [W.PUBLIC_STEPS[mid]] * 12
        row = materials.setdefault(key, {'label': program_label(asg, mid if mid in PUBLIC else None), 'public_id': mid if mid in PUBLIC else None,
                                         'uniform_steps': uniform_steps(asg) if isinstance(asg, list) else None, 'assignment': None if mid in PUBLIC else asg,
                                         'program_space': 'public' if mid in PUBLIC else ('recipe_edit_v1' if isinstance(asg, list) and any(ta.is_edit_program(s) or s == [{'op': ta.RECIPE}] for s in asg) else 'explicit_compositions_v2'),
                                         'c_a_by_seed': None, 'c_b_by_seed': None, 'e_by_seed': None, 'e_status': 'unavailable'})
        ca = {c['model_seed']: c['scores']['c_a'] for c in cells.values() if c['material_id'] == mid}
        if set(ca) >= set(seeds) and row['c_a_by_seed'] is None:
            row['c_a_by_seed'] = [ca[s]['normalized_mse_macro'] for s in seeds]
            row['c_a_by_seed_origin'] = [[statistics.fmean(ca[s]['per_origin_entity_normalized_mse'][o]) for o in range(ca[s]['n_origins'])] for s in seeds]
        cbv = {v['model_seed']: v['c_b'] for c, v in cb.items() if v['material_id'] == mid}
        if set(cbv) >= set(seeds) and row['c_b_by_seed'] is None and all(cbv[s]['status'] == 'SCORABLE' for s in seeds):
            row['c_b_by_seed'] = [cbv[s]['normalized_mse_macro'] for s in seeds]
            row['c_b_by_seed_origin'] = [[statistics.fmean(cbv[s]['per_origin_entity_normalized_mse'][o]) for o in range(cbv[s]['n_origins'])] for s in seeds]
        ev = {v['model_seed']: v['e'] for c, v in es.items() if v['material_id'] == mid}
        if set(ev) >= set(seeds) and row['e_by_seed'] is None and all(ev[s]['status'] == 'SCORABLE' for s in seeds):
            row['e_by_seed'] = [ev[s]['normalized_mse_macro'] for s in seeds]
            row['e_by_seed_origin'] = [[statistics.fmean(ev[s]['per_origin_entity_normalized_mse'][o]) for o in range(ev[s]['n_origins'])] for s in seeds]
            row['e_status'] = 'development_late'
    rows = [json.loads(x) for x in (d / 'trace.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()] if (d / 'trace.jsonl').exists() else []
    refs = [ref(sname, bname, r['event_id']) for r in rows] + [ref(sname, bname, 'overview'), ref(sname, bname, 'labels_after_commit')]
    labels = {mid: materials[k]['label'] for mid, k in plan_to_key.items() if k in materials}
    traj = summarize_trajectory(rows, labels) if tier == 'minimal' else LL.compress_trajectory(rows, tier)
    for r in traj:
        r['ref'] = ref(sname, bname, r['event_id'])
    tokens = dsks._unit_tokens(stage_root).get('fast:%s' % bname, {}) if (stage_root / 'raw_responses').exists() else {}
    kn = context.read_json(d / 'knowledge.json') if (d / 'knowledge.json').exists() else {'entries': []}
    rec = {'ref_prefix': ref(sname, bname, ''), 'job_ref': job, 'kind': kind, 'knowledge_loaded': [e_['body'] for e_ in kn.get('entries', [])] or None,
           'status': res['status'] if res else ('DIAGNOSTIC_VIEW' if commit and (commit.get('extra') or {}).get('diagnostic_view') else 'NOT_RUN'),
           'failure_kind': res.get('failure_kind') if res else None, 'committed_plan_id': (res.get('committed_plan_id') if res else (commit['material_id'] if commit else None)),
           'committed_material_key': plan_to_key.get(res.get('committed_plan_id') if res else (commit['material_id'] if commit else None)),
           'commit_reason': commit['reason'] if commit else None, 'plan_to_material_key': plan_to_key, 'plan_specs': {},
           'cost': {'fast_calls': res['calls'] if res else 0, 'tool_calls': res['tool_calls'] if res else 0, 'new_evaluations': res['new_evaluations'] if res else 0, 'tokens': tokens},
           'trajectory': traj}
    for mid in reg:
        p = W.aug_dir(d / job) / (mid + '__compiled.json')
        if p.exists():
            ms = context.read_json(p)['material_spec']
            if tier == 'minimal':
                pol = ms.get('policy', {})
                rec['plan_specs'][mid] = {'label': labels.get(mid), 'rationale': str(pol.get('rationale', ''))[:300],
                                          'rules': [{'when': r_['when'], 'steps': program_label([r_['steps']] * 1)} for r_ in pol.get('rules', [])] or None,
                                          'entities_per_program': {program_label([ms['programs'][i]]): ms['entity_program'].count(i) for i in range(len(ms.get('programs', [])))} if ms.get('programs') else None,
                                          'n_unknown': ms.get('n_unknown')}
            else:
                rec['plan_specs'][mid] = {k: v for k, v in ms.items() if k not in ('execution_order', 'profile', 'alias_of', 'edit_profile')}
    return rec, refs, materials


def summarize_trajectory(rows: list, labels: dict) -> list:
    """'minimal' tier: one compact row per action (build / evaluate / compare / inspect / commit / rejection / incomplete); every C_A number, plan label,
    commit and failure kept; Fast prose kept only through the plan rationales (plan_specs) and the commit reason."""
    out = []
    for r in rows:
        ev = r['event']
        if ev == 'tool_started':
            a = r.get('arguments') or {}
            if r['tool'] == 'build_material':
                out.append({'event_id': r['event_id'], 'action': 'build', 'plan_id': a.get('plan_id'), 'label': labels.get(a.get('plan_id')), 'n_rules': len((a.get('policy') or {}).get('rules') or [])})
            elif r['tool'] in ('inspect_data', 'inspect_material', 'overview'):
                out.append({'event_id': r['event_id'], 'action': r['tool'], 'arguments': {k: v for k, v in a.items() if k in ('entity_indices', 'kind', 'plan_id', 'entity_index')}})
        elif ev == 'tool_completed':
            o = r.get('output') or {}
            if r['tool'] == 'evaluate' and isinstance(o, dict) and o.get('feedback'):
                out.append({'event_id': r['event_id'], 'action': 'evaluate', 'plan_id': o.get('plan_id'), 'label': labels.get(o.get('plan_id')), 'c_a_mean': round(o['feedback']['mean_loss'], 5), 'c_a_by_seed': [round(x, 5) for x in o['feedback']['loss_by_seed']]})
            elif r['tool'] == 'compare' and isinstance(o, dict):
                out.append({'event_id': r['event_id'], 'action': 'compare', 'a': o.get('a'), 'b': o.get('b'), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items() if k in ('delta_mean', 'mean_delta', 'delta_by_seed', 'sign_agreement')}})
            elif r['tool'] == 'commit' and isinstance(o, dict):
                out.append({'event_id': r['event_id'], 'action': 'commit', 'plan_id': o.get('plan_id'), 'label': labels.get(o.get('plan_id')), 'reason': str(o.get('reason', ''))[:500]})
        elif ev == 'tool_rejected':
            out.append({'event_id': r['event_id'], 'action': 'rejected', 'tool': r.get('tool'), 'message': str((r.get('error') or {}).get('message', ''))[:200]})
        elif ev == 'job_incomplete':
            out.append({'event_id': r['event_id'], 'action': 'incomplete', 'failure_kind': r.get('failure_kind'), 'reason': r.get('reason')})
        elif ev == 'action_batch_deferred':
            out.append({'event_id': r['event_id'], 'action': 'deferred'})
    return out


def historical_late_rows(allowed_e: dict) -> dict:
    """E of the four public references on V1/T2/T4/Q2 from temporal_feedback_replay/historical_late (already opened there)."""
    out = {}
    for j in HIST_LATE_JOBS:
        f = TR_ROOT / 'historical_late' / j / 'historical_late_scores.json'
        if not f.exists() or not _e_allowed('hist_late', j, allowed_e):
            continue
        cells = context.read_json(f)['cells']
        for m in PUBLIC:
            ev = {v['model_seed']: v['e'] for v in cells.values() if v['material_id'] == m}
            if set(ev) >= set(SEEDS) and all(ev[s]['status'] == 'SCORABLE' for s in SEEDS):
                out.setdefault(j, {})['public:' + m] = {'e_by_seed': [ev[s]['normalized_mse_macro'] for s in SEEDS], 'e_by_seed_origin': [[statistics.fmean(ev[s]['per_origin_entity_normalized_mse'][o]) for o in range(4)] for s in SEEDS],
                                                       'e_status': 'development_late (opened by the temporal replay package)'}
    return out


def census(branches: list, tier, allowed_e: dict, *, purpose: str, stage_names: dict = STAGE_NAMES, hist_late: bool = False) -> dict:
    jobs, refs, records = {}, [], []
    for b in branches:
        stage_root, bname, job, sname, kind = b
        d = stage_root / bname
        if not (d / job / 'cells').exists():
            records.append({'branch': ref(sname, bname, ''), 'status': 'NOT_RUN'})
            continue
        rec, r_, mats = branch_evidence(b, tier, allowed_e)
        refs += r_
        J = jobs.setdefault(job, {'t': JOB_T.get(job) or rd.resolve_job(DATASET, job).t, 'overview': None, 'materials': {}, 'branches': []})
        if J['overview'] is None:
            ov = context.read_json(d / job / 'overview.json')
            J['overview'] = {'n_entities': ov['n_entities'], 'summary': ov['summary'], 'entities': W._columnar(ov['entities'])}
        for key, m in mats.items():
            cur = J['materials'].setdefault(key, m)
            for k in ('c_a_by_seed', 'c_b_by_seed', 'e_by_seed', 'c_a_by_seed_origin', 'c_b_by_seed_origin', 'e_by_seed_origin'):
                if cur.get(k) is None and m.get(k) is not None:
                    cur[k] = m[k]
                    if k == 'e_by_seed':
                        cur['e_status'] = m['e_status']
        J['branches'].append(rec)
    if hist_late:
        for j, rows in historical_late_rows(allowed_e).items():
            if j in jobs:
                for key, v in rows.items():
                    cur = jobs[j]['materials'].get(key)
                    if cur is not None and cur.get('e_by_seed') is None:
                        cur.update(v)
    for J in jobs.values():
        J['materials'] = {k: v for k, v in sorted(J['materials'].items(), key=lambda kv: (0 if kv[1]['public_id'] else 1, kv[1]['label']))}
        J['material_index'] = {k: 'M%02d' % i for i, k in enumerate(J['materials'])}
        J['materials'] = {J['material_index'][k]: (compact_material(v) if tier == 'minimal' else {**v, 'material_key': k}) for k, v in J['materials'].items()}
        for b_ in J['branches']:
            b_['plan_to_material'] = {p: J['material_index'].get(k) for p, k in b_['plan_to_material_key'].items()}
            b_['committed_material'] = J['material_index'].get(b_['committed_material_key'])
            del b_['plan_to_material_key'], b_['committed_material_key']
    actions = E.edit_action_table()
    actions['recipe_components'] = {n: {k: v for k, v in p.items() if k != 'source'} for n, p in actions['recipe_components'].items()}
    first = next((b for b in branches if (b[0] / b[1] / b[2] / 'overview.json').exists()), None)
    geometry = context.read_json(first[0] / first[1] / first[2] / 'overview.json')['geometry'] if first else None
    ev = {'census_status': 'CENSUS_COMPLETE' if jobs else 'CENSUS_EMPTY', 'domain_id': DOMAIN, 'purpose': purpose, 'profile': W.PROFILE_VERSION,
          'jobs_in_order': sorted(jobs, key=lambda j: jobs[j]['t']), 'jobs': jobs, 'not_run': records, 'legal_evidence_refs': sorted(set(refs)),
          'compression': {'tier_used': str(tier), 'rules': 'the parent package\'s size-only rules (tier 3) or the skeleton tier; actions, plans, C_A/C_B/E numbers, commits, failures and loaded card texts are never dropped'},
          'feedback_roles': {'C_A': 'immediate feedback Fast legally used at decision time (origins t, t+48; two 48-hour forecasts right after training)',
                             'C_B': 'delayed feedback of the same episode (origins t+96, t+144), opened after the commit',
                             'E': 'development_late: the late block (origins t+192..t+336, the deployment-matched period) of that batch, already opened in earlier development; the criterion the study cares about. Present ONLY for whitelisted earlier batches and marked e_status; "unavailable" means it was never opened for that material and must not be guessed',
                             'all_blocks': 'every block is the same 48-hour forecast task; they differ only in which calendar period after training is scored'},
          'program_spaces': {'explicit_compositions_v2': 'historical no-Skill Fast could compose 1-3 primitives explicitly; that DSL is NOT offered to the Fast of this study',
                             'recipe_edit_v1': 'the Fast of this study: every entity program is the preset P_NoMixRecipe or one P_NoMixRecipe_Edit step (0-2 disabled components), uniform or with up to 8 observation-predicate rules; public references may be committed directly',
                             'materials_labels': 'Preset = P_NoMixRecipe; Edit[-a-b] = preset with components a,b disabled on every entity; Comp[..] = explicit composition; Conditional{..} = entity-conditional plan of one batch (never merged across batches)'},
          'public_semantics': {'field_definitions': rd.FIELD_DEFINITIONS, 'actions': actions, 'tool_contracts': CONTRACTS, 'consumer': W.CONSUMER, 'geometry': geometry,
                               'public_references': {m: W.public_spec(m) for m in PUBLIC}, 'budget_per_job': LIMITS, 'body_limit_characters': BODY_LIMIT,
                               'metric': 'missing-aware normalized MSE on observed future cells; lower is better; ratios to None of the same batch are comparable across batches'}}
    low = json.dumps(ev, ensure_ascii=False).lower()
    if any(n in low for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')):
        raise PermissionError('census names a data source')
    return ev


def compact_material(v: dict) -> dict:
    """'minimal' tier: the label IS the complete assignment for uniform plans; conditional plans list the per-entity program labels; numbers rounded."""
    def r4(x, nd=4):
        return None if x is None else [([round(y, nd) for y in row] if isinstance(row, list) else round(row, nd)) for row in x]
    out = {'label': v['label'], 'public_id': v['public_id'], 'program_space': v['program_space'], 'e_status': v['e_status']}
    if v['label'].startswith('Conditional') and isinstance(v.get('assignment'), list):
        out['program_by_entity'] = [program_label([s]) for s in v['assignment']]
    for k in ('c_a_by_seed', 'c_b_by_seed', 'e_by_seed'):
        out[k] = r4(v.get(k))
    for k in ('c_a_by_seed_origin', 'c_b_by_seed_origin', 'e_by_seed_origin'):
        if v.get(k) is not None:
            out[k] = r4(v[k], 3)
    return out


SOURCE_E_ALLOWED = {'hist_source': ('RD02_S1', 'RD02_S2', 'RD02_V1', 'RD02_T2'), 'hist_select': ('RD02_T4', 'RD02_Q2'), 'hist_target': ('RD02_L1', 'RD02_L2', 'RD02_L3'),
                    'edit_pilot': ('RD02_L1', 'RD02_L3'), 'hist_late': HIST_LATE_JOBS}


def source_census(tier) -> dict:
    return census(SOURCE_BRANCHES, tier, SOURCE_E_ALLOWED, hist_late=True,
                  purpose='Source: nine earlier batches of one neutral domain -- no-Skill research trajectories (two program spaces), a random control and an ablation view on two of them, '
                          'with C_A / post-commit C_B and, where already opened in development, the late block E of every scored material. Can this be consolidated into a domain Skill for the recipe-edit Fast?')


# ============================================================================= decision summary (§5.A; 0 LLM, 0 fits, <= 20 minutes; joins the census)
def _mean(v):
    return statistics.fmean(v) if v else None


def decision_summary(cen: dict) -> dict:
    t0 = time.time()
    out = {'built_local': now(), 'note': 'deterministic readout of the Source census; oracle rows are POST-HOC DIAGNOSTICS (E was opened after the fact), not a deployable signal; '
                                        'no utility gate, no k / aggregation / threshold search'}
    jobs = cen['jobs_in_order']
    # 1) coverage: uniform strategy x Source job (which blocks hold three-seed scores); conditional plans listed per job, never merged
    labels = sorted({m['label'] for j in jobs for m in cen['jobs'][j]['materials'].values() if not m['label'].startswith('Conditional')})
    cov = {}
    for lab in labels:
        row = {}
        for j in jobs:
            ms = [m for m in cen['jobs'][j]['materials'].values() if m['label'] == lab]
            row[j] = ''.join(k for k, f in (('A', 'c_a_by_seed'), ('B', 'c_b_by_seed'), ('E', 'e_by_seed')) if ms and ms[0].get(f)) or '-'
        cov[lab] = row
    out['coverage_uniform_strategy_x_job'] = {'legend': 'A = C_A, B = C_B, E = development_late scored with three seeds; - = not evaluated / not opened', 'rows': cov,
                                              'conditional_plans_per_job': {j: [m['label'] for m in cen['jobs'][j]['materials'].values() if m['label'].startswith('Conditional')] for j in jobs}}
    # 2) per trajectory: commit vs shadow C_A argmin vs post-hoc E best inside its OWN evaluated pool
    traj = []
    for j in jobs:
        J = cen['jobs'][j]
        none_e = next((m['e_by_seed'] for m in J['materials'].values() if m['public_id'] == 'None'), None)
        for b in J['branches']:
            if b['status'] not in ('COMPLETE',) or not b.get('committed_material'):
                continue
            pool = sorted({mi for mi in b['plan_to_material'].values() if mi and J['materials'][mi].get('c_a_by_seed')})
            if not pool:
                continue
            ca = {mi: _mean(J['materials'][mi]['c_a_by_seed']) for mi in pool}
            shadow = min(pool, key=lambda mi: (ca[mi], pool.index(mi)))
            e_pool = [mi for mi in pool if J['materials'][mi].get('e_by_seed')]
            row = {'branch': b['ref_prefix'], 'job': j, 'kind': b['kind'], 'committed': b['committed_material'], 'committed_label': J['materials'][b['committed_material']]['label'],
                   'shadow_c_a_argmin': shadow, 'shadow_label': J['materials'][shadow]['label'], 'commit_equals_shadow': shadow == b['committed_material'],
                   'pool_size': len(pool), 'pool_with_e': len(e_pool), 'e_status': 'complete' if len(e_pool) == len(pool) else ('partial' if e_pool else 'unavailable')}
            if e_pool and none_e:
                den = _mean(none_e)
                em = {mi: _mean(J['materials'][mi]['e_by_seed']) for mi in e_pool}
                best = min(e_pool, key=lambda mi: (em[mi], e_pool.index(mi)))
                pub = [mi for mi in e_pool if J['materials'][mi]['public_id']]
                own = [mi for mi in e_pool if not J['materials'][mi]['public_id']]
                bp = min(pub, key=lambda mi: em[mi]) if pub else None
                bo = min(own, key=lambda mi: em[mi]) if own else None
                row.update({'post_hoc_e_best': best, 'post_hoc_e_best_label': J['materials'][best]['label'],
                            'gap_commit_vs_e_best_pp': 100 * (em.get(b['committed_material'], float('nan')) - em[best]) / den if b['committed_material'] in em else None,
                            'gap_shadow_vs_e_best_pp': 100 * (em.get(shadow, float('nan')) - em[best]) / den if shadow in em else None,
                            'best_public_e_ratio': em[bp] / den if bp else None, 'best_own_e_ratio': em[bo] / den if bo else None,
                            'opportunity_from_own_materials_pp': (100 * (em[bp] - em[bo]) / den) if (bp and bo) else None})
            traj.append(row)
    out['trajectories_commit_vs_shadow_vs_post_hoc'] = traj
    # 3) the single L1 -> L3 check on the uniform intersection (public + Edit[-x]) with three-seed E on both ends
    a, c = 'RD02_L1', 'RD02_L3'
    if a in cen['jobs'] and c in cen['jobs']:
        def uni(j):
            return {m['label']: m for m in cen['jobs'][j]['materials'].values() if (m['public_id'] or (m['label'].startswith('Edit[-') and m['label'].count('-') == 1)) and m.get('e_by_seed') and m.get('c_a_by_seed')}
        ua, uc = uni(a), uni(c)
        inter = sorted(set(ua) & set(uc), key=lambda l: (0 if l in PUBLIC else 1, l))
        if 'None' in inter and len(inter) >= 2:
            den_a, den_c = _mean(ua['None']['e_by_seed']), _mean(uc['None']['e_by_seed'])
            ratio_a = {l: _mean(ua[l]['e_by_seed']) / den_a for l in inter}
            hist_pick = min(inter, key=lambda l: (ratio_a[l], inter.index(l)))
            ca_c = {l: _mean(uc[l]['c_a_by_seed']) for l in inter}
            ca_pick = min(inter, key=lambda l: (ca_c[l], inter.index(l)))
            e_c = {l: _mean(uc[l]['e_by_seed']) for l in inter}
            best_c = min(inter, key=lambda l: (e_c[l], inter.index(l)))
            out['l1_to_l3_check'] = {'intersection': inter, 'earlier_batch_e_ratio_to_none': ratio_a, 'historical_recommendation_from_earlier_e': hist_pick,
                                     'later_batch_c_a_selection_on_same_set': ca_pick, 'later_batch_e_ratio_to_none': {l: e_c[l] / den_c for l in inter},
                                     'delta_pp_on_later_batch': {'historical_pick_vs_c_a_pick': 100 * (e_c[ca_pick] - e_c[hist_pick]) / den_c, 'historical_pick_vs_preset': 100 * (e_c['P_NoMixRecipe'] - e_c[hist_pick]) / den_c,
                                                                 'c_a_pick_vs_preset': 100 * (e_c['P_NoMixRecipe'] - e_c[ca_pick]) / den_c, 'post_hoc_best_on_later_batch': best_c},
                                     'note': 'ONE earlier batch and ONE cross-batch check: a case, not evidence that rolling averages or dynamic matching work'}
        else:
            out['l1_to_l3_check'] = {'status': 'NOT_COMPUTABLE', 'intersection': inter}
    # 4) the public rolling replay rows of the whitelisted batches only (L1..L3)
    sel = TR_ROOT / 'selections.json'
    if sel.exists():
        s = context.read_json(sel)
        rows = {}
        for jj in s['jobs']:
            if jj['job'] in ('RD02_L1', 'RD02_L2', 'RD02_L3'):
                rows[jj['job']] = {'history_used': jj['history'], 'R_CA_pick': jj['R_CA']['material'], 'R_HistLate_pick': jj['R_HistLate']['material'], 'R_HistNear_pick': jj['R_HistNear']['material'],
                                   'note': 'four public references only; from the temporal replay package; E outcomes of these picks are in this census (materials table)'}
        out['public_rolling_replay_whitelisted_rows'] = rows
    out['seconds'] = time.time() - t0
    out['within_time_cap'] = out['seconds'] <= DECISION_SUMMARY_SECONDS
    return out


# ============================================================================= Slow prompts (§4)
FEEDBACK_ROLES = ('Feedback roles, by name: C_A = the immediate feedback Fast legally used at decision time; it explains why Fast decided as it did. C_B = the delayed check '
                  'of that episode after its commit. E (development_late) = the late block of that batch, the criterion this study cares about, already opened in earlier '
                  'development for the whitelisted batches only; it tells which guidance deserves reuse, revision or should stay a hypothesis. When C_A and E disagree, KEEP the '
                  'conflict: a C_A lead is not a later gain, and no rule may a priori ignore all C_A. All blocks are the same 48-hour forecast; they differ in the calendar period scored. '
                  'A per-entity prediction change is not the independent contribution of that entity\'s material; this does not forbid grouping by observations, whose plans are '
                  'still judged by the complete shared model. Uncertainty can justify an extra check, a limited trial or keeping a baseline; one undecided comparison never becomes a '
                  'permanent operator ban; a historical conclusion supports only the treatments and situations it actually covered; an untried action is not harmful.')
GOAL = ('Your goal: improve the FINAL downstream prediction on later batches of the same domain and save worthless experiments. Propose an executable Workflow from the visible '
        'data problems and action-effect evidence: what to inspect, when to construct which discriminating complete plans (the preset, single or double disables, entity-conditional '
        'switches, or no augmentation), how to use feedback, when to stop, and how to commit. State the COMMIT POLICY explicitly: what role the current C_A plays; when a learned '
        'historical preference is kept and when it yields to current evidence; you may advise committing an evaluated plan whose C_A is not the lowest, and you may keep the plain '
        'C_A-minimum rule if nothing justifies deviating. Never require "must dominate" or "must win 3/3" as an unexamined default. Say which conditions the current tools compute '
        'and which are learned historical preferences. Name ONE concrete decision the card should change relative to a no-card Fast (especially the commit rule). Do not write job '
        'ids, entity ids, row numbers or a lookup table from batch to answer; do not hand-fit thresholds to future results. Distinguish a "common domain default" from the evidence '
        'for "when to deviate".')
NO_IDS = LL.NO_IDS
FORMAT_FORM = ('Output exact JSON, no markdown: {"decision":"KEEP","rationale":"...","evidence_review":{...}} OR {"decision":"PROPOSE","candidates":[{"candidate_id":"%s","research_mode":"<short label>",'
               '"workflow":"<executable Workflow incl. observe / construct / experiment / commit-or-stop; the commit policy must be inside it>","principles":"<conditions, actions, reasons, exceptions>" or null,'
               '"observable_applicability":{"const":true},"applicability_summary":"<=400 characters","evidence_refs":["exact refs from the supplied legal ranges"],"rationale":"...",'
               '"evidence_review":{"supports":["observation -> which advice it supports"],"contradicts_or_limits":["which later result (C_B / E) contradicts or limits which judgement"],'
               '"rule_changes":["which rule is kept / dropped / rewritten because of it"],"uncertain_and_expected_behavior_change":"what remains uncertain and which Fast behaviour should change"},'
               '"rule_status":[{"rule":"<one main rule>","status":"supported|hypothesis|unresolved","support":["refs or short facts"],"counter":["refs or short facts"]}],'
               '"commit_policy":"<one paragraph restating the commit rule>","decision_change_vs_no_card":"<the one concrete decision this card changes>"}]}. '
               'Exactly one candidate. Workflow plus Principles render to at most %d characters. KEEP is valid and is not resampled.')
SLOW_FORM_1 = ('You are the offline Slow of a batch training-data augmentation research Harness. You receive a deterministic census of nine earlier batches of ONE neutral domain: the '
               'no-Skill Fast research trajectories (two program spaces: the historical explicit-composition DSL and the recipe-edit DSL that the future Fast of this study will use), '
               'a random control and an ablation view on two batches, complete plans, per-seed C_A, post-commit C_B and -- where already opened -- the late block E of every scored '
               'material, costs, failures, plus a deterministic decision summary (coverage, commit vs shadow C_A argmin vs post-hoc E best, one earlier-to-later batch check). '
               + FEEDBACK_ROLES + ' ' + GOAL + ' ' + NO_IDS + ' ' + FORMAT_FORM % ('W1', BODY_LIMIT))
SLOW_FORM_2 = ('You are the offline Slow of a batch training-data augmentation research Harness. You receive the same census and decision summary as the first formation call, plus the '
               'verbatim output of that first call (card W1). Propose ONE card W2 that follows a genuinely different research strategy from W1 for a stated reason (a different diagnosis, '
               'construction priority, experiment allocation or commit judgement), not a rewording; or KEEP if the census supports no second strategy. This is not an independent '
               'statistical repeat. ' + FEEDBACK_ROLES + ' ' + GOAL + ' ' + NO_IDS + ' ' + FORMAT_FORM % ('W2', BODY_LIMIT))
FORMAT_REV = ('Output exact JSON, no markdown: {"decision":"KEEP","rationale":"...","evidence_review":{...},"counterexample_responses":["..."]} OR {"decision":"REVISE","child":{'
              '"workflow":"<full child Workflow text>","principles":"<Principles text>" or null,"changed_mechanism":"<the one decision mechanism changed and how>",'
              '"supporting_feedback":"<which actual Practice feedback supports the change>","expected_trigger":"<on which class of legal inputs the child decides differently from the parent>",'
              '"research_mode":"<short label>","applicability_summary":"<=400 characters","evidence_refs":["exact refs from the supplied legal ranges"],"rationale":"...",'
              '"evidence_review":{"supports":[...],"contradicts_or_limits":[...],"rule_changes":[...],"uncertain_and_expected_behavior_change":"..."},'
              '"rule_status":[{"rule":"...","status":"supported|hypothesis|unresolved","support":[...],"counter":[...]}],"commit_policy":"...","decision_change_vs_no_card":"...",'
              '"counterexample_responses":["..."]}}. Workflow plus Principles render to at most %d characters. KEEP is valid, is not resampled and is recorded as an alias of the parent.')
SLOW_REV = ('You are the offline Slow revising ONE parent card after its real use. You receive: the parent card (text, rationale, evidence review, rule status, commit policy and its own '
            'prediction of the decision it would change), the Source census and decision summary the parent was formed from (condensed), and the complete Practice trajectories of BOTH '
            'cards on one new batch: the loaded card text, tool calls, plans, per-seed C_A, the commit and reason, the post-commit C_B and the development_late E of every evaluated '
            'material of that batch, plus deterministic numeric counterexamples (facts only). ' + FEEDBACK_ROLES + ' Your job: identify where the parent\'s advice was NOT executed and '
            'where it WAS executed without benefit; answer every listed numeric counterexample explicitly (accept, bound or refute it with the given numbers); then either KEEP the parent '
            'unchanged or produce ONE child that changes ONE main decision mechanism (observation, candidate priority, or the comparison / commit rule; a limited edit, not a rewrite). '
            'State which actual feedback supports the change and on which class of legal inputs the child would decide differently; untriggered branches stay untested. You are not '
            'required to be more conservative, observe more, change components or grouping, and you need not change anything to produce a version. ' + NO_IDS + ' ' + FORMAT_REV % BODY_LIMIT)

CAND_KEYS = set(LL.CAND_KEYS) | {'rule_status', 'commit_policy', 'decision_change_vs_no_card'}
CHILD_KEYS = {'workflow', 'principles', 'changed_mechanism', 'supporting_feedback', 'expected_trigger', 'research_mode', 'applicability_summary', 'evidence_refs', 'rationale', 'evidence_review',
              'rule_status', 'commit_policy', 'decision_change_vs_no_card', 'counterexample_responses'}


def check_rule_status(rs) -> list:
    if not isinstance(rs, list) or not rs:
        raise ValueError('rule_status must be a nonempty list')
    out = []
    for r in rs:
        if not isinstance(r, dict) or set(r) != {'rule', 'status', 'support', 'counter'} or r['status'] not in ('supported', 'hypothesis', 'unresolved'):
            raise ValueError('each rule_status entry is {rule, status in supported|hypothesis|unresolved, support[], counter[]}')
        if not isinstance(r['rule'], str) or not r['rule'].strip() or any(not isinstance(r[k], list) for k in ('support', 'counter')):
            raise ValueError('rule_status entry malformed')
        out.append({'rule': r['rule'][:400], 'status': r['status'], 'support': [str(x)[:300] for x in r['support']][:8], 'counter': [str(x)[:300] for x in r['counter']][:8]})
    return out


def make_card(*, skill_id, revision, workflow, principles, summary, refs, legal_refs, mode, derived_from=None, source_stage='propose') -> dsk.Skill:
    if not isinstance(mode, str) or not mode.strip() or len(mode) > 80:
        raise ValueError('research_mode must be short nonempty text')
    dsk.check_reusable_text(mode, 'research_mode', W.skill_text_check)
    return dsk.make_skill(skill_id=skill_id, domain_id=DOMAIN, revision=revision, workflow=workflow, principles=principles, applicability_summary=summary,
                          compatibility_note='Formed for this study\'s fixed task, Consumer, L/H, sampling interval and T length.', observable_applicability={'const': True},
                          evidence_refs=refs, legal_evidence_refs=legal_refs, source_stage=source_stage, status='CANDIDATE_TEST_ONLY', derived_from=derived_from,
                          allowed_features=ALLOWED_FEATURES, text_check=W.skill_text_check, body_limit=BODY_LIMIT)


def parse_formation(resp, *, cid: str, legal_refs) -> dict:
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'PROPOSE'):
        raise ValueError('decision must be KEEP or PROPOSE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale', 'evidence_review'}:
            raise ValueError('KEEP carries only rationale and evidence_review')
        return {'decision': 'KEEP', 'rationale': str(resp.get('rationale', ''))[:3000], 'evidence_review': LL.check_review(resp['evidence_review']) if 'evidence_review' in resp else None, 'skill': None, 'meta': {}}
    if set(resp) != {'decision', 'candidates'} or not isinstance(resp['candidates'], list) or len(resp['candidates']) != 1:
        raise ValueError('PROPOSE carries exactly decision and one candidate')
    c = resp['candidates'][0]
    if not isinstance(c, dict) or set(c) != CAND_KEYS:
        raise ValueError('candidate keys must be exactly %s' % sorted(CAND_KEYS))
    if c['candidate_id'] != cid:
        raise ValueError('candidate_id must be %s' % cid)
    if c['observable_applicability'] != {'const': True}:
        raise ValueError('observable_applicability must be exactly {"const": true}')
    for k in ('commit_policy', 'decision_change_vs_no_card'):
        if not isinstance(c[k], str) or not c[k].strip():
            raise ValueError('%s must be nonempty text' % k)
    review = LL.check_review(c['evidence_review'])
    rs = check_rule_status(c['rule_status'])
    s = make_card(skill_id='%s-%s-r1' % (DOMAIN, cid), revision=1, workflow=c['workflow'], principles=c['principles'], summary=c['applicability_summary'], refs=c['evidence_refs'],
                  legal_refs=legal_refs, mode=c['research_mode'])
    meta = {'candidate_id': cid, 'research_mode': c['research_mode'], 'rationale': str(c['rationale'])[:3000], 'evidence_review': review, 'rule_status': rs,
            'commit_policy': c['commit_policy'][:2000], 'decision_change_vs_no_card': c['decision_change_vs_no_card'][:2000]}
    return {'decision': 'PROPOSE', 'skill': s, 'meta': meta}


def parse_revision(resp, *, parent: dsk.Skill, legal_refs) -> dict:
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'REVISE'):
        raise ValueError('decision must be KEEP or REVISE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale', 'evidence_review', 'counterexample_responses'}:
            raise ValueError('KEEP carries only rationale, evidence_review and counterexample_responses')
        return {'decision': 'KEEP', 'rationale': str(resp.get('rationale', ''))[:3000], 'evidence_review': LL.check_review(resp.get('evidence_review')),
                'counterexample_responses': [str(x) for x in (resp.get('counterexample_responses') or [])], 'skill': None, 'meta': {}}
    if set(resp) != {'decision', 'child'} or not isinstance(resp['child'], dict) or set(resp['child']) != CHILD_KEYS:
        raise ValueError('REVISE carries exactly decision and child with keys %s' % sorted(CHILD_KEYS))
    c = resp['child']
    for k in ('changed_mechanism', 'supporting_feedback', 'expected_trigger', 'commit_policy', 'decision_change_vs_no_card'):
        if not isinstance(c[k], str) or not c[k].strip():
            raise ValueError('%s must be nonempty text' % k)
    if c['principles'] is not None and (not isinstance(c['principles'], str) or not c['principles'].strip()):
        raise ValueError('principles must be null or nonempty text')
    if not isinstance(c['counterexample_responses'], list) or any(not isinstance(x, str) for x in c['counterexample_responses']):
        raise ValueError('counterexample_responses must be a list of strings')
    review = LL.check_review(c['evidence_review'])
    rs = check_rule_status(c['rule_status'])
    child_id = re.sub(r'-r\d+$', '', parent.skill_id) + '-r%d' % (parent.revision + 1)
    s = make_card(skill_id=child_id, revision=parent.revision + 1, workflow=c['workflow'], principles=c['principles'], summary=c['applicability_summary'], refs=c['evidence_refs'],
                  legal_refs=legal_refs, mode=c['research_mode'], derived_from=parent.skill_id, source_stage='revision')
    if s.rendered_body == parent.rendered_body:
        raise ValueError('the child renders identically to the parent; return KEEP instead')
    return {'decision': 'REVISE', 'skill': s, 'meta': {k: (c[k][:3000] if isinstance(c[k], str) else c[k]) for k in ('changed_mechanism', 'supporting_feedback', 'expected_trigger', 'research_mode', 'rationale', 'commit_policy', 'decision_change_vs_no_card')},
            'evidence_review': review, 'rule_status': rs, 'counterexample_responses': c['counterexample_responses']}


def arm_of(skill_id: str) -> str:
    return 'cand_' + skill_id.split('-', 1)[1].replace('-', '')            # RD02-W1-r1 -> cand_W1r1


# ============================================================================= budget / stage config / launch / resume
def stage_caps(root: Path, stage: str) -> dict:
    P_ = paths(root)
    rec = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    if stage in rec:
        return rec[stage]
    led = rt.RuntimeLedger(P_['ledger'], **TOTAL)
    s, alloc = led.s, ALLOC[stage]
    snap = {k: s.get(k, 0) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'fit_wall_seconds')}
    snap['elapsed_s'] = time.time() - s['started_epoch']
    retries = min(TOTAL['max_retries'] - s['retries_used'], TOTAL['max_retries']) if alloc['fits'] else 0
    caps = {'max_fit_attempts': min(TOTAL['max_fit_attempts'], s['fit_attempts'] + alloc['fits'] + retries), 'max_llm_requests': min(TOTAL['max_llm_requests'], s['llm_requests'] + alloc['requests']),
            'max_llm_tokens': min(TOTAL['max_llm_tokens'], s['llm_tokens_in'] + s['llm_tokens_out'] + alloc['tokens']), 'max_wall_s': TOTAL['max_wall_s'], 'max_retries': min(TOTAL['max_retries'], s['retries_used'] + retries)}
    rec[stage] = {'epoch_start': time.time(), 'local': now(), 'allocation': alloc, 'snapshot_at_start': snap, 'caps': caps, 'http_cap': min(HTTP_CAP, s['llm_http_attempts'] + alloc['requests'] + 1)}
    context.write_json(P_['stages'], rec)
    return rec[stage]


def stage_config(root: Path, stage: str, *, jobs, order, knowledge, labels_e: bool, fixed_dev=None) -> Path:
    P_ = paths(root)
    path = P_['configs'] / ('%s.json' % stage)
    if path.exists():
        return path
    sc = stage_caps(root, stage)
    cfg = {'study': 'tempo_aug_domain_skill_optimize', 'status': 'FROZEN', 'stage': stage, 'output': str(P_[stage]), 'jobs': list(jobs), 'order': {j: list(order[j]) for j in jobs},
           'knowledge': knowledge, 'labels_e': bool(labels_e), 'llm': True, 'random': False, 'ablation': [], 'fixed_dev': fixed_dev, 'seeds': list(SEEDS), 'job_index': {j: JOB_INDEX[j] for j in jobs},
           'public_ids': list(PUBLIC), 'limits': LIMITS, 'max_tool_corrections': MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'fast_token_cap': FAST_TOKEN_CAP,
           'max_output_tokens': MAX_OUTPUT_TOKENS, 'fast_system': FAST_SYSTEM, 'contracts': CONTRACTS, 'caps': sc['caps'], 'http_cap': sc['http_cap'], 'ledger_path': str(P_['ledger']),
           'fit_retry': True, 'fit_attempts_at_stage_start': context.read_json(P_['ledger'])['fit_attempts'], 'cache_from': {j: str(CACHE_FROM[j]) for j in jobs}, 'model': MODEL,
           'e_whitelist': {stage: list(jobs)}}
    dsks.write_once(path, cfg)
    return path


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
            p = subprocess.Popen([sys.executable, '-B', '-m', MODULE, '--stage-worker', str(cfg_path)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=W.worker_env())
            try:
                rc = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)
                context.write_json(root / ('%s_supervisor_timeout.json' % stage), {'epoch': time.time(), 'pid': p.pid})
                rc = 124
        print('STAGE_EXIT', stage, rc, now(), flush=True)
    return dsks.stage_status(sroot)


def _run_arms(root: Path, cfg: dict, led, client, branches, failures) -> None:
    fc = W.FastClient(client, root, cfg['fast_system']) if client is not None else None
    for job in cfg['jobs']:
        if (root / (job + '_not_scorable.json')).exists():
            continue
        ji, seeds, public_ids = cfg['job_index'][job], tuple(cfg['seeds']), tuple(cfg['public_ids'])
        try:
            common = E.prepare_common_edit(root, job, led, cfg)
        except RuntimeError as exc:
            if 'C_A_NOT_SCORABLE' not in str(exc):
                raise
            context.write_json(root / (job + '_not_scorable.json'), {'job': job, 'status': 'C_A_NOT_SCORABLE'})
            failures.append({'job': job, 'kind': 'C_A_NOT_SCORABLE'})
            continue
        for arm in cfg['order'][job]:
            if arm == 'fixed_dev':
                continue                                                # 0-LLM view, built after every Fast arm of the stage (see fixed_dev_views)
            path = root / ('%s_%s' % (job, arm))
            branches.append((path, job))
            kn = W.knowledge_from_json(cfg['knowledge'][job][arm])
            try:
                if path.exists() and (path / 'branch_result.json').exists():
                    prior = context.read_json(path / 'branch_result.json')
                    result = br.RunResult(**{k: v for k, v in prior.items() if k != 'trace'}) if prior['status'] == 'COMPLETE' else \
                        E.resume_fast_branch_edit(path, job, kn, led, fc, common, job_index=ji, public_ids=public_ids, seeds=seeds)
                elif path.exists():
                    raise RuntimeError('branch directory without a result; inspect before any replay')
                else:
                    result = E.fast_branch_edit(path, job, kn, led, fc, common, job_index=ji, public_ids=public_ids, seeds=seeds)
                if result.status != 'COMPLETE':
                    failures.append({'branch': path.name, 'kind': result.failure_kind, 'reason': result.reason})
            except Exception as exc:  # noqa: BLE001
                failures.append({'branch': path.name, 'exception_type': type(exc).__name__, 'message': W._safe_message(exc)})
                print('BRANCH_FAILED', path.name, type(exc).__name__, flush=True)
            E._guard(root, led, client, 'f_edit')


def fixed_dev_views(root: Path, cfg: dict, led, branches, failures) -> None:
    """Replay only: the frozen Fixed_dev program materialised per batch in a 0-LLM view (alias of a public reference, or the uniform edit built and fitted)."""
    fd = cfg.get('fixed_dev')
    if not fd:
        return
    for job in cfg['jobs']:
        if 'fixed_dev' not in cfg['order'][job] or (root / (job + '_not_scorable.json')).exists():
            continue
        ji, seeds, public_ids = cfg['job_index'][job], tuple(cfg['seeds']), tuple(cfg['public_ids'])
        common = root / (job + '_common')
        path = root / (job + '_fixed_dev')
        try:
            if (path / job / 'commit.json').exists():
                branches.append((path, job))
                continue
            if path.exists():
                shutil.rmtree(path)
            ad = E.EditAdapter(path, job, led, REPO, common=common, job_index=ji, public_ids=public_ids, seeds=seeds)
            ad.baselines()
            if fd['public_id']:
                mid = fd['public_id']
            else:
                mid = 'FixedDev'
                c = ad.build_material({'plan_id': mid, 'policy': E.uniform_edit_policy(fd['disabled_ops'], 'Fixed_dev: the frozen uniform program of the Select stage, regenerated on this batch (0 LLM)')}, remaining_seconds=led.remaining())
                ad.evaluate(c, seeds, feedback=True, remaining_seconds=led.remaining())
            W.commit_branch(path, DATASET, job, mid, 'Fixed_dev view: the program frozen at Select, regenerated as this batch\'s own material; no research decision, 0 LLM', seeds[0],
                            extra={'diagnostic_view': True, 'fixed_dev': fd})
            branches.append((path, job))
            print('FIXED_DEV', job, mid, flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({'branch': path.name, 'exception_type': type(exc).__name__, 'message': W._safe_message(exc)})
            print('FIXED_DEV_FAILED', job, type(exc).__name__, flush=True)
        E._guard(root, led, None, 'fixed_dev')


def stage_worker(cfg_path: Path) -> None:
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
        client = rt.MeteredClient(led, root / 'raw_responses', http_cap=int(cfg['http_cap'])) if cfg.get('llm') else None
        _run_arms(root, cfg, led, client, branches, failures)
        context.write_json(root / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches]})
        fixed_dev_views(root, cfg, led, branches, failures)
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': W._safe_message(exc)})
        stopped = type(exc).__name__
        print('EXECUTION_STOP', stopped, flush=True)
    finally:
        context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures})
    W.open_labels(root, cfg, branches, failures, led, withhold_reason=stopped and 'execution stopped before every planned arm was attempted (%s)' % stopped)


def resume_stage(root: Path, stage: str, *, accept_unknown_usage: bool = False, restart=()) -> None:
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
        moved.append({'branch': name, 'kept_as': dest.name, 'prior_status': prior and prior['status']})
    context.write_json(sroot / 'resume.json', {'epoch': time.time(), 'accepted_unknown_usage': bool(accept_unknown_usage), 'restarted_branches': moved})
    rt.FIT_RETRY = bool(cfg.get('fit_retry'))
    branches, failures, stopped = [], [], None
    try:
        client = rt.MeteredClient(led, sroot / 'raw_responses', http_cap=int(cfg['http_cap']))
        _run_arms(sroot, cfg, led, client, branches, failures)
        context.write_json(sroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches], 'resumed': True})
        fixed_dev_views(sroot, cfg, led, branches, failures)
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': W._safe_message(exc)})
        stopped = type(exc).__name__
    finally:
        context.write_json(sroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': True})
    W.open_labels(sroot, cfg, branches, failures, led, resumed=True, withhold_reason=stopped and 'resumed stage stopped again (%s)' % stopped)


# ============================================================================= Slow calls
def slow_client(root: Path, stage: str):
    P_ = paths(root)
    sc = stage_caps(root, stage)
    led = rt.RuntimeLedger(P_['ledger'], **sc['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage; paid %s refused' % stage)
    if not W.proxy_reachable():
        raise RuntimeError('LLM proxy not reachable; %s refused before any paid call' % stage)
    (P_[stage] / 'raw_responses').mkdir(parents=True, exist_ok=True)
    return led, rt.MeteredClient(led, P_[stage] / 'raw_responses', http_cap=sc['http_cap']), sc


def formation_payload(cen: dict, summary: dict, *, w1_output=None) -> dict:
    p = {'domain_id': DOMAIN, 'census': {k: v for k, v in cen.items() if k != 'legal_evidence_refs'}, 'decision_summary': summary, 'legal_evidence_refs': LL.ref_ranges(cen['legal_evidence_refs']),
         'legal_evidence_refs_format': 'evidence_refs must be exact strings "<stage>/<branch>/<job>:<integer>" inside these ranges, or one of other_refs',
         'allowed_batch_features': sorted(ALLOWED_FEATURES), 'body_limit_characters': BODY_LIMIT, 'required': {'observable_applicability': {'const': True}, 'candidates': 1}, 'status': 'CANDIDATE_TEST_ONLY',
         'future_fast': {'program_space': 'recipe_edit_v1 (see census.program_spaces)', 'budget_per_job': LIMITS, 'feedback_available_to_fast': 'T observations, tool results, C_A only', 'system_prompt_fixed': True}}
    if w1_output is not None:
        p['first_formation_call_output'] = w1_output
    return p


def formation(root: Path) -> dict:
    P_ = paths(root)
    out = P_['formation']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'candidates.json').exists():
        return context.read_json(out / 'candidates.json')
    P_['evidence'].mkdir(exist_ok=True)
    led, client, sc = slow_client(root, 'formation')
    if not (P_['evidence'] / 'census_source.json').exists():
        sizes, chosen = {}, None
        for tier in (3, 'skeleton', 'minimal'):
            cen = source_census(tier)
            summ = decision_summary(cen)
            sizes[str(tier)] = LL.payload_bytes(SLOW_FORM_2, formation_payload(cen, summ, w1_output={'placeholder_for_w1_output_bytes': 'x' * 16384}))
            chk = LL.check_reservation(led, sc, sizes[str(tier)], calls_left=2)
            cen['compression']['payload_bytes_by_tier'] = dict(sizes)
            cen['compression']['reservation_check_at_choice'] = chk
            if chk['fits_all_remaining_calls']:
                chosen = tier
                break
        if chosen is None:                                                # the frozen client checks every call before sending; the smallest tier must at least fit one call + correction
            chk = LL.check_reservation(led, sc, sizes['minimal'], calls_left=1)
            cen['compression']['reservation_check_at_choice'] = {**chk, 'note': 'no tier fits two calls under the 2x byte reservation; the minimal tier is used and each call is checked before sending against the actual usage'}
            if not chk['fits_with_correction']:
                raise RuntimeError('even the minimal census does not fit one formation call: %s' % chk)
        dsks.write_once(P_['evidence'] / 'census_source.json', cen)
        dsks.write_once(P_['evidence'] / 'decision_summary.json', summ)
    cen, summ = context.read_json(P_['evidence'] / 'census_source.json'), context.read_json(P_['evidence'] / 'decision_summary.json')
    if cen['census_status'] != 'CENSUS_COMPLETE':
        raise RuntimeError('source census incomplete')
    legal = frozenset(cen['legal_evidence_refs'])
    forbidden = [n for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')]
    w1_raw = None
    for call_name, cid, system in (('F1', 'W1', SLOW_FORM_1), ('F2', 'W2', SLOW_FORM_2)):
        f = out / ('%s.json' % call_name)
        if f.exists():
            rec = context.read_json(f)
        else:
            payload = formation_payload(cen, summ, w1_output=w1_raw if call_name == 'F2' else None)
            size = LL.payload_bytes(system, payload)
            chk = LL.check_reservation(led, sc, size)
            if not chk['fits_with_correction']:
                raise RuntimeError('formation payload does not fit the frozen reservation with a correction margin: %s' % chk)
            res = LL.slow_call(client, 'slow_form_%s' % cid.lower(), call_name, payload, system, lambda r, cid=cid: parse_formation(r, cid=cid, legal_refs=legal), forbidden=forbidden)
            rec = {**{k: v for k, v in res.items() if k != 'skill'}, 'skill': LL.skill_json(res.get('skill')), 'reservation_check': chk, 'requests_after': led.s['llm_requests'], 'written_local': now()}
            dsks.write_once(f, rec)
            if client.fatal or rt.unknown_usage_blocks(led):
                raise RuntimeError('backend fatal or unknown usage during formation')
        if call_name == 'F1':
            w1_raw = rec.get('raw')
    f1, f2 = context.read_json(out / 'F1.json'), context.read_json(out / 'F2.json')
    statuses = {'F1': f1['status'], 'F2': f2['status']}
    if any(s in ('CALL_FAILED', 'PARSE_OR_VALIDATION_FAILED') for s in statuses.values()):
        rec = {'status': 'FORMATION_TECHNICAL_FAILURE', 'calls': statuses, 'skills': {}, 'meta': {}, 'written_local': now()}
    else:
        skills, meta, aliases = {}, {}, {}
        for r in (f1, f2):
            if r.get('skill'):
                sj = r['skill']
                same = next((k for k, v in skills.items() if v['rendered_body'] == sj['rendered_body']), None)
                if same:
                    aliases[sj['skill_id']] = same
                else:
                    skills[sj['skill_id']] = sj
                    meta[sj['skill_id']] = r['meta']
        rec = {'status': 'PROPOSED' if skills else 'NO_INITIAL_PROPOSAL', 'calls': statuses, 'skills': skills, 'meta': meta, 'aliases': aliases, 'n_candidates': len(skills),
               'keep_rationales': {k: r.get('rationale') for k, r in (('F1', f1), ('F2', f2)) if r['status'] == 'KEEP'}, 'written_local': now()}
    dsks.write_once(out / 'candidates.json', rec)
    return rec


# ============================================================================= practice on L4, parent choice, one revision
def practice_stage(root: Path, cands: dict) -> dict:
    skills = {k: dsk.skill_from_json(v) for k, v in cands['skills'].items()}
    ids = sorted(skills)
    arms = {arm_of(k): skills[k] for k in ids}
    order = {PRACTICE_JOB: [arm_of(k) for k in ids]}
    knowledge = {PRACTICE_JOB: {a: asdict(dsk.skill_knowledge(s)) for a, s in arms.items()}}
    cfgp = stage_config(root, 'practice', jobs=(PRACTICE_JOB,), order=order, knowledge=br.json_copy(knowledge), labels_e=True)
    return run_stage(root, 'practice', cfgp)


def _arm_e_ratio(sroot: Path, job: str, arm: str):
    b = sroot / ('%s_%s' % (job, arm))
    res = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
    e = W._committed_block(b, job, 'e')
    none = W._block_vec(b, job, 'None', 'e') if b.exists() else None
    ratio = statistics.fmean(e) / statistics.fmean(none) if e and none and statistics.fmean(none) > 0 else None
    tokens = dsks._unit_tokens(sroot).get('fast:%s_%s' % (job, arm), {}) if (sroot / 'raw_responses').exists() else {}
    return {'status': res['status'] if res else 'NOT_RUN', 'committed': res['committed_plan_id'] if res else None, 'e_by_seed': e, 'none_e_by_seed': none, 'ratio': ratio,
            'new_evaluations': res['new_evaluations'] if res else None, 'tokens': (tokens.get('prompt_tokens', 0) + tokens.get('completion_tokens', 0)) if tokens else None,
            'c_a_by_seed': W._committed_block(b, job, 'c_a') if res else None, 'c_b_by_seed': W._committed_block(b, job, 'c_b') if res else None}


def choose_parent(root: Path, cands: dict) -> dict:
    P_ = paths(root)
    out = P_['formation'] / 'parent.json'
    if out.exists():
        return context.read_json(out)
    ids = sorted(cands['skills'])
    runs = {sid: _arm_e_ratio(P_['practice'], PRACTICE_JOB, arm_of(sid)) for sid in ids}
    ok = [sid for sid in ids if runs[sid]['ratio'] is not None]
    parent = None
    if ok:
        best = min(runs[s]['ratio'] for s in ok)
        tied = [s for s in ok if runs[s]['ratio'] - best <= TOL]
        parent = min(tied, key=lambda s: (runs[s]['new_evaluations'] or 0, runs[s]['tokens'] or 0, ids.index(s)))
    rec = {'rule': 'parent = the card whose actual L4 commit has the lowest mean_seed E / mean_seed E(None); ties <= 1e-12 by fewer new evaluations, fewer tokens, W1 first',
           'runs': runs, 'parent': parent, 'status': 'PARENT_CHOSEN' if parent else 'PRACTICE_INCOMPLETE', 'written_local': now()}
    dsks.write_once(out, rec)
    return rec


PRACTICE_E_ALLOWED = {'practice': (PRACTICE_JOB,)}


def practice_census(root: Path, cands: dict, tier) -> dict:
    arms = [arm_of(k) for k in sorted(cands['skills'])]
    branches = [(paths(root)['practice'], '%s_%s' % (PRACTICE_JOB, a), PRACTICE_JOB, 'practice', 'card-loaded Fast (recipe-edit DSL); the card text is in knowledge_loaded') for a in arms]
    return census(branches, tier, PRACTICE_E_ALLOWED, stage_names={str(paths(root)['practice']): 'practice'},
                  purpose='Practice: real use of both candidate cards on one new batch, with post-commit C_B and development_late E of every evaluated material')


def counterexamples(prac: dict, parent_arm: str) -> list:
    out = []
    J = prac['jobs'].get(PRACTICE_JOB)
    if not J:
        return [{'fact': 'no practice census'}]
    for b in J['branches']:
        mats = J['materials']
        cm = mats.get(b.get('committed_material') or '', {})
        pool = {mi: mats[mi] for mi in set(b['plan_to_material'].values()) if mi and mats.get(mi, {}).get('e_by_seed')}
        none_e = next((m['e_by_seed'] for m in mats.values() if m.get('public_id') == 'None'), None)
        den = _mean(none_e) if none_e else None
        best = min(pool.items(), key=lambda kv: _mean(kv[1]['e_by_seed'])) if pool else None
        ca_pool = {mi: mats[mi] for mi in set(b['plan_to_material'].values()) if mi and mats.get(mi, {}).get('c_a_by_seed')}
        shadow = min(ca_pool.items(), key=lambda kv: _mean(kv[1]['c_a_by_seed'])) if ca_pool else None
        out.append({'branch_ref': b['ref_prefix'], 'is_parent': b['ref_prefix'].endswith('/%s_%s/' % (PRACTICE_JOB, parent_arm)), 'committed': b.get('committed_material'), 'committed_label': cm.get('label'),
                    'committed_c_a_mean': _mean(cm.get('c_a_by_seed')), 'committed_c_b_mean': _mean(cm.get('c_b_by_seed')), 'committed_e_mean': _mean(cm.get('e_by_seed')), 'none_e_mean': den,
                    'preset_e_mean': _mean(next((m['e_by_seed'] for m in mats.values() if m.get('public_id') == 'P_NoMixRecipe'), None)),
                    'lowest_e_in_own_pool': {'material': best[0], 'label': best[1]['label'], 'e_mean': _mean(best[1]['e_by_seed']), 'c_a_mean': _mean(best[1].get('c_a_by_seed'))} if best else None,
                    'shadow_c_a_argmin': {'material': shadow[0], 'label': shadow[1]['label'], 'e_mean': _mean(shadow[1].get('e_by_seed'))} if shadow else None,
                    'commit_is_lowest_e': bool(best and best[0] == b.get('committed_material')), 'commit_is_c_a_argmin': bool(shadow and shadow[0] == b.get('committed_material')),
                    'gap_commit_vs_lowest_e_pp_of_none': (100 * (_mean(cm['e_by_seed']) - _mean(best[1]['e_by_seed'])) / den) if (best and cm.get('e_by_seed') and den) else None})
    return out


def source_digest(cen: dict) -> dict:
    """Revision input: the Source census condensed to the materials tables (labels + numbers) and each branch's commit / reason / cost; observations and
    trajectories are dropped (the decision summary carries the commit / shadow / post-hoc facts). Refs stay legal."""
    jobs = {}
    for j in cen['jobs_in_order']:
        J = cen['jobs'][j]
        jobs[j] = {'t': J['t'], 'materials': J['materials'],
                   'branches': [{'ref_prefix': b['ref_prefix'], 'kind': b['kind'], 'status': b['status'], 'committed_material': b.get('committed_material'),
                                 'committed_label': (J['materials'].get(b.get('committed_material') or '') or {}).get('label'), 'commit_reason': (b.get('commit_reason') or '')[:300],
                                 'new_evaluations': (b.get('cost') or {}).get('new_evaluations')} for b in J['branches']]}
    return {'census_status': cen['census_status'], 'domain_id': cen['domain_id'], 'condensed': 'materials + commits only; observations / trajectories dropped for size',
            'jobs_in_order': cen['jobs_in_order'], 'jobs': jobs, 'feedback_roles': cen['feedback_roles'], 'program_spaces': cen['program_spaces'],
            'public_semantics': {k: cen['public_semantics'][k] for k in ('budget_per_job', 'body_limit_characters', 'metric')}, 'legal_evidence_refs': cen['legal_evidence_refs']}


def revision(root: Path, cands: dict, parent: dict) -> dict:
    P_ = paths(root)
    out = P_['revision']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'child.json').exists():
        return context.read_json(out / 'child.json')
    sid = parent['parent']
    skills = {k: dsk.skill_from_json(v) for k, v in cands['skills'].items()}
    led, client, sc = slow_client(root, 'revision')
    cen_src = context.read_json(P_['evidence'] / 'census_source.json')
    summ = context.read_json(P_['evidence'] / 'decision_summary.json')
    if not (P_['evidence'] / 'census_source_digest.json').exists():
        dsks.write_once(P_['evidence'] / 'census_source_digest.json', source_digest(source_census('minimal')))
    cen_skel = context.read_json(P_['evidence'] / 'census_source_digest.json')
    if not (P_['evidence'] / 'census_practice.json').exists():
        sizes = {}
        for tier in (3, 'skeleton', 'minimal'):
            prac = practice_census(root, cands, tier)
            probe = {'parent_card': {'rendered_body': 'x' * BODY_LIMIT, 'rationale': 'x' * 3000, 'evidence_review': 'x' * 2000, 'rule_status': 'x' * 2000},
                     'source_census_condensed': {k: v for k, v in cen_skel.items() if k != 'legal_evidence_refs'}, 'decision_summary': summ,
                     'practice_census': {k: v for k, v in prac.items() if k != 'legal_evidence_refs'}, 'numeric_counterexamples': 'x' * 4000,
                     'legal_evidence_refs': LL.ref_ranges(sorted(set(cen_skel['legal_evidence_refs']) | set(prac['legal_evidence_refs'])))}
            sizes[str(tier)] = LL.payload_bytes(SLOW_REV, probe)
            chk = LL.check_reservation(led, sc, sizes[str(tier)], calls_left=1)
            prac['compression']['payload_bytes_by_tier'] = dict(sizes)
            prac['compression']['reservation_check_at_choice'] = chk
            if chk['fits_with_correction']:
                break
        dsks.write_once(P_['evidence'] / 'census_practice.json', prac)
    prac = context.read_json(P_['evidence'] / 'census_practice.json')
    legal = frozenset(cen_src['legal_evidence_refs']) | frozenset(prac['legal_evidence_refs'])
    forbidden = [n for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')]
    ps = skills[sid]
    m = cands['meta'][sid]
    payload = {'domain_id': DOMAIN, 'parent_card': {'skill_id': sid, 'workflow': ps.workflow, 'principles': ps.principles, 'rendered_body': ps.rendered_body, 'research_mode': m['research_mode'],
                                                    'rationale': m['rationale'], 'evidence_review': m['evidence_review'], 'rule_status': m['rule_status'], 'commit_policy': m['commit_policy'],
                                                    'own_prediction_decision_change_vs_no_card': m['decision_change_vs_no_card']},
               'parent_practice_arm': arm_of(sid), 'other_card_practice_arm': [arm_of(k) for k in cands['skills'] if k != sid],
               'source_census_condensed': {k: v for k, v in cen_skel.items() if k != 'legal_evidence_refs'}, 'decision_summary': summ,
               'practice_census': {k: v for k, v in prac.items() if k != 'legal_evidence_refs'}, 'numeric_counterexamples': counterexamples(prac, arm_of(sid)),
               'legal_evidence_refs': LL.ref_ranges(sorted(legal)), 'legal_evidence_refs_format': 'evidence_refs must be exact strings "<stage>/<branch>/<job>:<integer>" inside these ranges, or one of other_refs',
               'allowed_batch_features': sorted(ALLOWED_FEATURES), 'body_limit_characters': BODY_LIMIT, 'required': {'observable_applicability': {'const': True}, 'max_children': 1}, 'status': 'CANDIDATE_TEST_ONLY'}
    size = LL.payload_bytes(SLOW_REV, payload)
    chk = LL.check_reservation(led, sc, size)
    if not chk['fits_with_correction']:
        raise RuntimeError('revision payload does not fit the frozen reservation with a correction margin: %s' % chk)
    res = LL.slow_call(client, 'slow_revise', sid, payload, SLOW_REV, lambda r: parse_revision(r, parent=ps, legal_refs=legal), forbidden=forbidden)
    rec = {**{k: v for k, v in res.items() if k != 'skill'}, 'skill': LL.skill_json(res.get('skill')), 'parent': sid, 'reservation_check': chk, 'requests_after': led.s['llm_requests'], 'written_local': now()}
    rec['child_status'] = 'REVISED' if rec['status'] == 'REVISE' else ('KEEP_ALIAS_OF_PARENT' if rec['status'] == 'KEEP' else 'REVISION_TECHNICAL_FAILURE')
    dsks.write_once(out / 'child.json', rec)
    return rec


# ============================================================================= select on L5, freeze W* and Fixed_dev, replay on L6/L7
def select_options(cands: dict, child: dict) -> dict:
    """{arm: (skill_id or None, Skill or None)} in the order f0, W1, W2, child (a KEEP child is an alias and is not run)."""
    skills = {k: dsk.skill_from_json(v) for k, v in cands['skills'].items()}
    opts = {'f0': (None, None)}
    for sid in sorted(skills):
        opts[arm_of(sid)] = (sid, skills[sid])
    if child.get('status') == 'REVISE' and child.get('skill'):
        s = dsk.skill_from_json(child['skill'])
        opts[arm_of(s.skill_id)] = (s.skill_id, s)
    return opts


def select_stage(root: Path, cands: dict, child: dict) -> dict:
    opts = select_options(cands, child)
    order = {SELECT_JOB: list(opts)}
    knowledge = {SELECT_JOB: {a: asdict(dsk.skill_knowledge(s)) for a, (sid, s) in opts.items()}}
    cfgp = stage_config(root, 'select', jobs=(SELECT_JOB,), order=order, knowledge=br.json_copy(knowledge), labels_e=True)
    return run_stage(root, 'select', cfgp)


def _uniform_materials_e(sroot: Path, job: str) -> dict:
    """Every public reference and every UNIFORM edit material evaluated in any branch of the job, with three-seed E: {label: {'public_id', 'disabled_ops', 'e_by_seed'}}."""
    out = {}
    for b in sorted(sroot.glob('%s_*' % job)):
        if not (b / job / 'e_scores.json').exists():
            continue
        reg = W.load_registry(b / job)
        for mid, r in reg.items():
            e = W._block_vec(b, job, mid, 'e')
            if not e:
                continue
            if mid in PUBLIC:
                out.setdefault(mid, {'label': mid, 'public_id': mid, 'disabled_ops': None, 'e_by_seed': e})
                continue
            steps = uniform_steps(r.get('assignment'))
            if steps is None or not (ta.is_edit_program(steps) or steps == [{'op': ta.RECIPE}]):
                continue
            lab = program_label(r['assignment'])
            if lab == 'Preset':
                continue
            out.setdefault(lab, {'label': lab, 'public_id': None, 'disabled_ops': steps[0]['disabled_ops'], 'e_by_seed': e, 'first_seen_in': b.name})
    return out


def freeze_choices(root: Path, cands: dict, child: dict) -> dict:
    P_ = paths(root)
    out = P_['formation'] / 'frozen_choices.json'
    if out.exists():
        return context.read_json(out)
    opts = select_options(cands, child)
    ids = sorted(cands['skills'])
    runs = {a: _arm_e_ratio(P_['select'], SELECT_JOB, a) for a in opts}
    J = {a: r['ratio'] for a, r in runs.items() if r['ratio'] is not None}
    card_arms = [a for a in opts if a != 'f0']
    child_arms = [a for a in card_arms if opts[a][0] not in ids]
    ok = [a for a in card_arms if a in J]
    rec = {'formula': 'J(W) = mean_seed E(L5, actual_commit_W) / mean_seed E(L5, None); lower is better', 'runs': runs, 'J': J, 'card_arms': card_arms, 'child_arms': child_arms, 'incomplete': [a for a in opts if a not in J]}
    if not ok or 'f0' not in J:
        rec.update(status='SELECT_INCOMPLETE', W_star=None, Fixed_dev=None)
        dsks.write_once(out, rec)
        return rec
    best = min(J[a] for a in ok)
    tied = [a for a in ok if J[a] - best <= TOL]
    w = min(tied, key=lambda a: (runs[a]['new_evaluations'] or 0, runs[a]['tokens'] or 0, 1 if a in child_arms else 0, card_arms.index(a)))
    uni = _uniform_materials_e(P_['select'], SELECT_JOB)
    none_e = uni.get('None', {}).get('e_by_seed')
    fixed = None
    if none_e and statistics.fmean(none_e) > 0:
        den = statistics.fmean(none_e)
        ratio = {lab: statistics.fmean(v['e_by_seed']) / den for lab, v in uni.items()}
        lo = min(ratio.values())
        tied_f = [lab for lab in uni if ratio[lab] - lo <= TOL]
        lab = min(tied_f, key=lambda l: (0 if uni[l]['public_id'] else 1, list(PUBLIC).index(l) if uni[l]['public_id'] else 0, json.dumps(uni[l]['disabled_ops'] or [])))
        fixed = {'label': lab, 'public_id': uni[lab]['public_id'], 'disabled_ops': uni[lab]['disabled_ops'], 'ratio': ratio[lab], 'candidates_ratio': ratio,
                 'rule': 'lowest mean_seed E / E(None) among the public references and the UNIFORM edit materials actually evaluated on L5 in any branch; ties public order first, then canonical program text; conditional plans excluded; the program definition is copied, never the L5 model'}
    rec.update(status='FROZEN', W_star={'arm': w, 'skill_id': opts[w][0], 'J': J[w], 'is_child': w in child_arms, 'f0_J': J['f0'], 'card_minus_f0_pp': 100 * (J['f0'] - J[w])},
               Fixed_dev=fixed, skills={opts[w][0]: LL.skill_json(opts[w][1])}, rules='quality only; ties <= 1e-12: fewer new evaluations, fewer tokens, original order, child last; f0 winning does not remove W*',
               frozen_local=now(), frozen_epoch=time.time())
    dsks.write_once(out, rec)
    return rec


def replay_stage(root: Path, frozen: dict) -> dict:
    if frozen['status'] != 'FROZEN':
        raise RuntimeError('choices are not frozen; Replay refused')
    skill = dsk.skill_from_json(frozen['skills'][frozen['W_star']['skill_id']])
    order = {'RD02_L6': ['f0', 'f_skill', 'fixed_dev'], 'RD02_L7': ['f_skill', 'f0', 'fixed_dev']}
    knowledge = {j: {'f0': asdict(br.Knowledge()), 'f_skill': asdict(dsk.skill_knowledge(skill))} for j in REPLAY_JOBS}
    fd = frozen['Fixed_dev'] and {'public_id': frozen['Fixed_dev']['public_id'], 'disabled_ops': frozen['Fixed_dev']['disabled_ops'], 'label': frozen['Fixed_dev']['label']}
    cfgp = stage_config(root, 'replay', jobs=REPLAY_JOBS, order=order, knowledge=br.json_copy(knowledge), labels_e=True, fixed_dev=fd)
    return run_stage(root, 'replay', cfgp)


# ============================================================================= preflight (0 cost) / smoke (0 fits, 0 LLM) / wiring (<= 3 fits)
def preflight(root: Path = ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    fc = root / 'frozen_config.json'
    if fc.exists():
        return context.read_json(fc)
    caches = {j: E.cache_binding(j, CACHE_FROM[j]) for j in (PRACTICE_JOB, SELECT_JOB) + REPLAY_JOBS}
    src = {}
    for stage_root, bname, job, sname, kind in SOURCE_BRANCHES:
        d = stage_root / bname / job
        src['%s/%s' % (sname, bname)] = {'cells': (d / 'cells').exists(), 'c_b': (d / 'c_b_scores.json').exists(), 'e': (d / 'e_scores.json').exists(), 'trace': (stage_root / bname / 'trace.jsonl').exists()}
    hl = {j: (TR_ROOT / 'historical_late' / j / 'historical_late_scores.json').exists() for j in HIST_LATE_JOBS}
    t0 = time.time()
    cen = source_census(3)
    summ = decision_summary(cen)
    sizes = {'tier3_form2': LL.payload_bytes(SLOW_FORM_2, formation_payload(cen, summ, w1_output={'x': 'x' * 16384}))}
    cen_s = source_census('skeleton')
    sizes['skeleton_form2'] = LL.payload_bytes(SLOW_FORM_2, formation_payload(cen_s, summ, w1_output={'x': 'x' * 16384}))
    cen_m = source_census('minimal')
    sizes['minimal_form2'] = LL.payload_bytes(SLOW_FORM_2, formation_payload(cen_m, summ, w1_output={'x': 'x' * 16384}))
    low = json.dumps(formation_payload(cen_m, summ), ensure_ascii=False).lower()
    sizes['forbidden_names_in_minimal_payload'] = [n for n in tuple(spec.DATASETS) + tuple(x for x in rd.SOURCE_NAMES if x not in ('rd01', 'rd02')) if n.lower() in low]
    cfg = {'package': PACKAGE, 'task': TASK, 'frozen_local': now(), 'exposure': EXPOSURE, 'dataset': DATASET, 'seeds': list(SEEDS), 'public_ids': list(PUBLIC),
           'jobs': {'source': list(SOURCE_JOBS), 'practice': PRACTICE_JOB, 'select': SELECT_JOB, 'replay': list(REPLAY_JOBS)}, 'job_index': {j: JOB_INDEX[j] for j in (PRACTICE_JOB, SELECT_JOB) + REPLAY_JOBS},
           'orders': {'practice': 'W1 then W2 (ascending id)', 'select': 'f0, W1, W2, child', 'replay': {'RD02_L6': ['f0', 'f_skill', 'fixed_dev'], 'RD02_L7': ['f_skill', 'f0', 'fixed_dev']}},
           'source_whitelist': [{'stage': s, 'branch': b, 'job': j, 'kind': k} for _, b, j, s, k in SOURCE_BRANCHES], 'source_e_allowed': {k: list(v) for k, v in SOURCE_E_ALLOWED.items()},
           'e_semantics': 'development_late: Slow may read the E of the Source whitelist and of Practice / Select after their commits; Fast never reads its own C_B / E; Replay E is used only for the readout',
           'fast': {'system_prompt': FAST_SYSTEM, 'contracts': CONTRACTS, 'limits': LIMITS, 'token_cap': FAST_TOKEN_CAP, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'model': MODEL, 'program_space': 'recipe_edit_v1 (EditAdapter of recipe_edit_pilot)'},
           'slow': {'form_1': SLOW_FORM_1, 'form_2': SLOW_FORM_2, 'revision': SLOW_REV, 'body_limit': BODY_LIMIT, 'one_contract_correction_per_call': True},
           'selection': {'parent': 'lowest E/None of the actual L4 commit; ties fewer new evaluations, fewer tokens, W1', 'W_star': 'J(W) = E(L5, commit)/E(L5, None) among cards; ties fewer new evaluations, fewer tokens, order, child last',
                         'Fixed_dev': 'lowest E/None among public + uniform edit materials evaluated on L5; ties public order then canonical program text', 'no_second_revision': True},
           'budget': {**TOTAL, 'http_cap': HTTP_CAP, 'alloc': ALLOC, 'numeric_wall_s': NUMERIC_WALL_S, 'max_fast_runs': 10, 'new_sha': 0, 'git_commit': 0},
           'metrics': {'delta_pp': 'Delta_j(A,B) = 100 (mean_s E_j(A) - mean_s E_j(B)) / mean_s E_j(None); > 0 = B better; paired seed vectors with the same denominator; SE and df=2 t95', 'unit': 'job; two replay jobs = development clues only'},
           'caches': caches, 'source_files': src, 'historical_late': hl, 'decision_summary_seconds': summ['seconds'], 'formation_payload_bytes': sizes, 'environment': E.environment()}
    if not all(c['ok'] for c in caches.values()):
        context.write_json(root / 'preflight_failed.json', cfg)
        raise RuntimeError('cache preflight failed: %s' % {j: c.get('ok') for j, c in caches.items()})
    dsks.write_once(fc, cfg)
    print('PREFLIGHT_OK', {j: c['public_cells'] for j, c in caches.items()}, 'source census jobs', cen['jobs_in_order'], 'summary_s', round(summ['seconds'], 1), 'bytes', sizes)
    return cfg


def smoke(root: Path = ROOT) -> dict:
    sdir = paths(root)['smoke']
    if (sdir / 'smoke.json').exists():
        return context.read_json(sdir / 'smoke.json')
    sdir.mkdir(parents=True, exist_ok=True)
    rec = {'local': now(), 'checks': {}}
    # 1) knowledge loading: a SMOKE_FIXTURE card renders into the Fast request guidance; f0 renders nothing
    fx = dsk.make_skill(skill_id='RD02-SMOKE-r0', domain_id=DOMAIN, revision=0, workflow='Smoke fixture: inspect the overview, then build one single-component edit and evaluate it; commit the evaluated plan you consider best.',
                        principles=None, applicability_summary='smoke fixture', compatibility_note='smoke', observable_applicability={'const': True}, evidence_refs=['fixture:smoke'], legal_evidence_refs=frozenset({'fixture:smoke'}),
                        source_stage='fixture', status='SMOKE_FIXTURE', allowed_features=ALLOWED_FEATURES, text_check=W.skill_text_check, body_limit=BODY_LIMIT)
    kn = W.knowledge_from_json(asdict(dsk.skill_knowledge(fx)))
    rendered = kn.render({}, ALLOWED_FEATURES)
    rec['checks']['knowledge_loads'] = {'loaded_hooks': [x['hook'] for x in rendered['loaded']], 'version': rendered['version'], 'f0_loaded': W.knowledge_from_json(asdict(br.Knowledge())).render({}, ALLOWED_FEATURES)['loaded'],
                                        'pass': rendered['loaded'] and rendered['loaded'][0]['hook'] == 'experiment_guidance' and 'Smoke fixture' in rendered['loaded'][0]['body'] and rendered['version'] == 1}
    # 2) E whitelist: the census refuses E outside the whitelist and accepts it inside
    try:
        census([(P_ROOT, 'RD02_L5_f_edit', 'RD02_L5', 'edit_pilot', 'x')], 'skeleton', {'edit_pilot': ('RD02_L1', 'RD02_L3')}, purpose='smoke')
        refused = False
    except PermissionError as exc:
        refused = str(exc)[:120]
    c_ok = census([(P_ROOT, 'RD02_L1_f_edit', 'RD02_L1', 'edit_pilot', 'x')], 'skeleton', {'edit_pilot': ('RD02_L1',)}, purpose='smoke')
    has_e = any(m.get('e_by_seed') for m in c_ok['jobs']['RD02_L1']['materials'].values())
    rec['checks']['e_whitelist'] = {'refuses_outside': refused, 'accepts_inside_with_e': has_e, 'pass': bool(refused) and has_e}
    # 3) parsers: a well-formed candidate passes, a bad status / missing key fails
    legal = frozenset(c_ok['legal_evidence_refs'])
    good = {'decision': 'PROPOSE', 'candidates': [{'candidate_id': 'W1', 'research_mode': 'smoke', 'workflow': 'Observe the batch summary; build one edit; commit the evaluated plan with the lowest C_A unless it is worse than the preset by less than seed noise.',
                                                  'principles': None, 'observable_applicability': {'const': True}, 'applicability_summary': 'smoke', 'evidence_refs': [sorted(legal)[0]], 'rationale': 'smoke',
                                                  'evidence_review': {'supports': ['x'], 'contradicts_or_limits': [], 'rule_changes': [], 'uncertain_and_expected_behavior_change': 'x'},
                                                  'rule_status': [{'rule': 'r', 'status': 'hypothesis', 'support': [], 'counter': []}], 'commit_policy': 'argmin C_A', 'decision_change_vs_no_card': 'none'}]}
    ok1 = parse_formation(good, cid='W1', legal_refs=legal)['decision'] == 'PROPOSE'
    bad = copy.deepcopy(good)
    bad['candidates'][0]['rule_status'][0]['status'] = 'proven'
    try:
        parse_formation(bad, cid='W1', legal_refs=legal)
        ok2 = False
    except ValueError:
        ok2 = True
    rec['checks']['parsers'] = {'good_passes': ok1, 'bad_status_rejected': ok2, 'pass': ok1 and ok2}
    # 4) decision summary within its time cap and structurally complete
    cen = context.read_json(root / 'evidence' / 'census_source.json') if (root / 'evidence' / 'census_source.json').exists() else source_census('skeleton')
    summ = decision_summary(cen)
    rec['checks']['decision_summary'] = {'seconds': summ['seconds'], 'within_cap': summ['within_time_cap'], 'n_trajectories': len(summ['trajectories_commit_vs_shadow_vs_post_hoc']),
                                         'l1_to_l3': summ.get('l1_to_l3_check', {}).get('historical_recommendation_from_earlier_e') or summ.get('l1_to_l3_check', {}).get('status'),
                                         'pass': summ['within_time_cap'] and len(summ['trajectories_commit_vs_shadow_vs_post_hoc']) > 0}
    rec['checks']['controller_imports_torch'] = 'torch' in sys.modules
    rec['all_pass'] = all(v['pass'] for k, v in rec['checks'].items() if isinstance(v, dict)) and not rec['checks']['controller_imports_torch']
    context.write_json(sdir / 'smoke.json', rec)
    print('SMOKE', rec['all_pass'], {k: v.get('pass') for k, v in rec['checks'].items() if isinstance(v, dict)}, flush=True)
    if not rec['all_pass']:
        raise RuntimeError('smoke failed')
    return rec


def wiring(root: Path = ROOT) -> dict:
    """RD02_T1, <= 3 fits, scripted client with a SMOKE_FIXTURE card loaded: build one edit, evaluate, then commit a plan that is NOT the C_A argmin
    of the pool (the commit permission), C_B only; E never read."""
    preflight(root)
    P_ = paths(root)
    wroot = P_['wiring']
    if (wroot / 'wiring.json').exists():
        return context.read_json(wroot / 'wiring.json')
    sc = stage_caps(root, 'wiring')
    led = rt.RuntimeLedger(P_['ledger'], **sc['caps'])
    rt.FIT_RETRY = False
    wroot.mkdir(parents=True, exist_ok=True)
    (wroot / 'logs').mkdir(exist_ok=True)
    job, ji, seeds = WIRING_JOB, WIRING_JOB_INDEX, WIRING_SEEDS
    cfg = {'stage': 'wiring', 'output': str(wroot), 'jobs': [job], 'seeds': list(seeds), 'job_index': {job: ji}, 'public_ids': list(WIRING_PUBLIC), 'labels_e': False, 'random': False, 'llm': False}
    context.write_json(wroot / 'config.json', cfg)
    context.write_json(wroot / 'experiment_started.json', {'epoch': time.time(), 'pid': os.getpid(), 'stage': 'wiring'})
    common = wroot / (job + '_common')
    copied = W.copy_physical_cache(WIRING_CACHE, common, job, seeds=seeds)
    fx = dsk.make_skill(skill_id='RD02-SMOKE-r0', domain_id=DOMAIN, revision=0, workflow='Wiring fixture: build one single-component edit, evaluate it, then commit.', principles=None,
                        applicability_summary='wiring fixture', compatibility_note='wiring', observable_applicability={'const': True}, evidence_refs=['fixture:smoke'], legal_evidence_refs=frozenset({'fixture:smoke'}),
                        source_stage='fixture', status='SMOKE_FIXTURE', allowed_features=ALLOWED_FEATURES, text_check=W.skill_text_check, body_limit=BODY_LIMIT)
    plan = E.uniform_edit_policy(['tp_censor'], 'wiring: the preset with censor disabled on every entity')
    chosen = {}

    def commit_non_argmin(request):
        cands = {c['plan_id']: c['feedback']['mean_loss'] for c in request['candidates'] if c['trained'] and c['feedback']}
        argmin = min(cands, key=lambda k: (cands[k], list(cands).index(k)))
        pick = next(k for k in ('None', 'FixedMixup', 'E_nocensor') if k in cands and k != argmin)
        chosen.update({'pool_c_a': cands, 'argmin': argmin, 'committed': pick, 'loaded': request['guidance']['loaded']})
        return [{'tool': 'commit', 'arguments': {'plan_id': pick, 'reason': 'wiring: scripted commit of an evaluated plan that is NOT the C_A argmin (permission check)'}}]
    scripts = {job + '_W': [[{'tool': 'overview', 'arguments': {}}], [{'tool': 'build_material', 'arguments': {'plan_id': 'E_nocensor', 'policy': plan}}],
                            [{'tool': 'evaluate', 'arguments': {'plan_id': 'E_nocensor'}}], commit_non_argmin]}
    client = W.ScriptedClient(scripts)
    path = wroot / (job + '_W')
    before = led.s['fit_attempts']
    res = E.fast_branch_edit(path, job, dsk.skill_knowledge(fx), led, client, common, job_index=ji, public_ids=WIRING_PUBLIC, seeds=seeds)
    out = {'label': 'WIRING_ACCEPTANCE (scripted client; fixture card; no LLM; not an agent result)', 'job': job, 'seeds': list(seeds), 'cache': copied, 'fits_before': before,
           'branch': {'status': res.status, 'failure_kind': res.failure_kind, 'reason': res.reason, 'committed': res.committed_plan_id, 'calls': res.calls, 'tool_calls': res.tool_calls, 'new_evaluations': res.new_evaluations,
                      'fits_charged': led.s['fit_attempts'] - before}, 'scripted_choice': chosen}
    branches, failures = [(path, job)], ([] if res.status == 'COMPLETE' else [{'branch': path.name, 'kind': res.failure_kind}])
    context.write_json(wroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches]})
    context.write_json(wroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures})
    W.open_labels(wroot, cfg, branches, failures, led)
    commit = context.read_json(path / job / 'commit.json') if (path / job / 'commit.json').exists() else None
    cb = context.read_json(path / job / 'c_b_scores.json')['cells'] if (path / job / 'c_b_scores.json').exists() else {}
    out['commit_json'] = commit and {'material_id': commit['material_id'], 'reason': commit['reason']}
    out['checks'] = {'branch_complete': res.status == 'COMPLETE', 'fixture_card_loaded_in_request': bool(chosen.get('loaded')) and 'Wiring fixture' in json.dumps(chosen.get('loaded')),
                     'commit_is_not_c_a_argmin_and_not_overwritten': bool(commit and chosen and commit['material_id'] == chosen['committed'] != chosen['argmin']),
                     'edit_evaluated_with_3_fits': out['branch']['fits_charged'] == 3 and res.new_evaluations == 1, 'c_b_collected_edit': sum('__E_nocensor__' in c for c in cb) == 3,
                     'e_never_read': not (path / job / 'e_scores.json').exists() and not (path / job / 'e_frozen.json').exists(), 'fits_at_most_3': led.s['fit_attempts'] - before <= 3}
    out['status'] = 'PASS' if all(out['checks'].values()) else 'FAIL'
    out['written_local'] = now()
    context.write_json(wroot / 'wiring.json', out)
    print('WIRING', out['status'], out['checks'], flush=True)
    return out


# ============================================================================= readout (§7)
def _branch_pool(sroot: Path, job: str, arm: str) -> dict:
    b = sroot / ('%s_%s' % (job, arm))
    if not (b / 'branch_result.json').exists() and not (b / job / 'commit.json').exists():
        return {'status': 'MISSING'}
    cells = E._cells_of(b, job) if (b / job / 'cells').exists() else {}
    res = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
    commit = context.read_json(b / job / 'commit.json') if (b / job / 'commit.json').exists() else None
    reg = W.load_registry(b / job)
    none_e = E._vec(cells, 'None', 'e')
    out = {'status': res['status'] if res else ('VIEW' if commit else 'NOT_RUN'), 'calls': res['calls'] if res else 0, 'tool_calls': res['tool_calls'] if res else 0, 'new_evaluations': res['new_evaluations'] if res else 0,
           'commit': commit and commit['material_id'], 'commit_reason': commit and commit['reason'][:500], 'commit_is_public': bool(commit and commit['material_id'] in PUBLIC),
           'commit_label': commit and (commit['material_id'] if commit['material_id'] in PUBLIC else program_label(reg.get(commit['material_id'], {}).get('assignment'))),
           'own_materials': {m: {'label': program_label(reg[m]['assignment']), 'n_rules': len((reg[m].get('compiled') or {}).get('policy', {}).get('rules', [])), 'c_a': E._vec(cells, m, 'c_a'), 'c_b': E._vec(cells, m, 'c_b'), 'e': E._vec(cells, m, 'e')}
                             for m in cells if m not in PUBLIC and m in reg},
           'conditional_used': any(len((reg[m].get('compiled') or {}).get('policy', {}).get('rules', [])) > 0 for m in cells if m not in PUBLIC and m in reg)}
    if commit and E._vec(cells, commit['material_id'], 'e') and none_e:
        den = statistics.fmean(none_e)
        ce = E._vec(cells, commit['material_id'], 'e')
        pool = [m for m in list(PUBLIC) + [m for m in cells if m not in PUBLIC] if E._vec(cells, m, 'e')]
        pm = {m: statistics.fmean(E._vec(cells, m, 'e')) for m in pool}
        best = min(pool, key=lambda m: (pm[m], pool.index(m)))
        sh = E._argmin_ca(cells, pool)
        out.update({'commit_e': ce, 'commit_e_mean': statistics.fmean(ce), 'commit_c_a': E._vec(cells, commit['material_id'], 'c_a'), 'commit_c_b': E._vec(cells, commit['material_id'], 'c_b'),
                    'commit_e_per_origin': [statistics.fmean(x) for x in zip(*[cells[commit['material_id']][s]['e_per_origin'] for s in SEEDS])],
                    'public_e': {m: E._vec(cells, m, 'e') for m in PUBLIC}, 'e_none_mean': den,
                    'pool': {'materials': pool, 'best': best, 'best_label': best if best in PUBLIC else program_label(reg[best]['assignment']), 'pool_gain_pp': 100 * (pm['P_NoMixRecipe'] - pm[best]) / den,
                             'selection_loss_pp': 100 * (statistics.fmean(ce) - pm[best]) / den, 'net_pp': 100 * (pm['P_NoMixRecipe'] - statistics.fmean(ce)) / den,
                             'public_best_minus_pool_best_pp': 100 * (min(pm[m] for m in PUBLIC) - pm[best]) / den},
                    'shadow_ca_argmin': {'material': sh, 'label': sh if sh in PUBLIC else program_label(reg[sh]['assignment']), 'same_as_commit': sh == commit['material_id'],
                                         'paired_commit_vs_shadow': E._paired(E._vec(cells, sh, 'e'), ce, den)}})
    return out


def readout(root: Path = ROOT) -> dict:
    P_ = paths(root)
    res = {'package': PACKAGE, 'task': TASK, 'exposure': EXPOSURE, 'finished_local': now(), 'stages': {s: dsks.stage_status(P_[s]) if P_[s].exists() else {'status': 'NOT_STARTED'} for s in ('practice', 'select', 'replay')}}
    cands = context.read_json(P_['formation'] / 'candidates.json') if (P_['formation'] / 'candidates.json').exists() else {}
    parent = context.read_json(P_['formation'] / 'parent.json') if (P_['formation'] / 'parent.json').exists() else {}
    child = context.read_json(P_['revision'] / 'child.json') if (P_['revision'] / 'child.json').exists() else {}
    frozen = context.read_json(P_['formation'] / 'frozen_choices.json') if (P_['formation'] / 'frozen_choices.json').exists() else {}
    res['cards'] = {'formation': cands, 'parent': parent, 'revision': child, 'frozen': frozen}
    res['decision_summary'] = context.read_json(P_['evidence'] / 'decision_summary.json') if (P_['evidence'] / 'decision_summary.json').exists() else None
    # practice / select / replay branches
    for stage, jobs in (('practice', (PRACTICE_JOB,)), ('select', (SELECT_JOB,)), ('replay', REPLAY_JOBS)):
        sroot = P_[stage]
        if not sroot.exists():
            continue
        cfg = context.read_json(sroot / 'config.json') if (sroot / 'config.json').exists() else {}
        res[stage] = {}
        for job in jobs:
            arms = list((cfg.get('order') or {}).get(job, []))
            res[stage][job] = {a: _branch_pool(sroot, job, a) for a in arms}
    # replay main readout
    rep = {}
    if 'replay' in res:
        for job in REPLAY_JOBS:
            R = res['replay'].get(job, {})
            f0, fs, fd = R.get('f0', {}), R.get('f_skill', {}), R.get('fixed_dev', {})
            if fs.get('commit_e') and f0.get('commit_e'):
                den = fs['e_none_mean']
                pub = fs['public_e']
                cells_fs = E._cells_of(P_['replay'] / ('%s_f_skill' % job), job)
                menu = E._argmin_ca({m: cells_fs[m] for m in PUBLIC if m in cells_fs}, list(PUBLIC)) if pub else None
                row = {'e_none_mean': den, 'f_skill_commit': fs['commit_label'], 'f0_commit': f0['commit_label'], 'fixed_dev_commit': fd.get('commit_label'),
                       'skill_vs_f0': E._paired(f0['commit_e'], fs['commit_e'], den), 'skill_vs_nomix': E._paired(pub['P_NoMixRecipe'], fs['commit_e'], den), 'skill_vs_none': E._paired(pub['None'], fs['commit_e'], den),
                       'skill_vs_fixed_dev': E._paired(fd['commit_e'], fs['commit_e'], den) if fd.get('commit_e') else None, 'f0_vs_nomix': E._paired(pub['P_NoMixRecipe'], f0['commit_e'], den),
                       'public_e_means': {m: statistics.fmean(pub[m]) for m in PUBLIC}, 'menu_ca_shadow': {'material': menu, 'delta_skill_vs_menu_pp': E._paired(pub[menu], fs['commit_e'], den)['delta_pp'] if menu else None},
                       'same_delivery_skill_f0': fs['commit'] == f0['commit'], 'skill_same_material_as_fixed_dev': fs.get('commit_label') == fd.get('commit_label'),
                       'shadow': {'f_skill': fs['shadow_ca_argmin'], 'f0': f0['shadow_ca_argmin']}, 'pool': {'f_skill': fs['pool'], 'f0': f0['pool']}, 'conditional_used': {'f_skill': fs['conditional_used'], 'f0': f0['conditional_used']}}
                rep[job] = row
        if rep:
            keys = ('skill_vs_f0', 'skill_vs_nomix', 'skill_vs_none', 'skill_vs_fixed_dev', 'f0_vs_nomix')
            agg = {}
            for k in keys:
                d = [rep[j][k]['delta_pp'] for j in rep if rep[j].get(k)]
                agg[k] = {'n': len(d), 'mean_pp': statistics.fmean(d) if d else None, 'per_job': {j: rep[j][k]['delta_pp'] for j in rep if rep[j].get(k)}, 'max_harm_pp': min(d) if d else None,
                          'wins': sum(x > 0 for x in d), 'ties': sum(x == 0 for x in d), 'losses': sum(x < 0 for x in d)}
            res['replay_aggregate'] = agg
    res['replay_readout'] = rep
    res['cost'] = cost_readout(root)
    context.write_json(root / 'result.json', res)
    (root / 'tables.md').write_text(tables(res), encoding='utf-8')
    write_report(root, res)
    return res


def cost_readout(root: Path) -> dict:
    led = context.read_json(paths(root)['ledger']) if paths(root)['ledger'].exists() else {}
    stages = context.read_json(paths(root)['stages']) if paths(root)['stages'].exists() else {}
    per_stage = {}
    for s in ('formation', 'practice', 'revision', 'select', 'replay'):
        sroot = paths(root)[s]
        if (sroot / 'raw_responses').exists():
            per_stage[s] = dsks._unit_tokens(sroot)
    return {k: led.get(k) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'fit_wall_seconds', 'label_wall_seconds', 'material_wall_seconds', 'llm_requests', 'llm_http_attempts',
                                    'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'llm_failed_attempts', 'unknown_usage_accepted', 'caps')} | \
        {'elapsed_s_since_ledger': (time.time() - led['started_epoch']) if led else None, 'stage_snapshots': {s: v.get('snapshot_at_start') for s, v in stages.items()}, 'per_stage_units': per_stage, 'slow_requests': sum(u['requests'] for s in ('formation', 'revision') for u in per_stage.get(s, {}).values()),
         'new_sha': 0, 'git_commits': 0}


def _f(x, nd=4):
    return '—' if x is None else ('%.*f' % (nd, x))


def _p(x):
    return '—' if x is None else ('%+.2f' % x)


def tables(res: dict) -> str:
    L = ['# %s — tables' % PACKAGE, '', 'E = missing-aware normalized MSE macro (3-seed mean); Δ in pp of the job\'s None mean E (> 0 = second better).', '']
    for stage in ('practice', 'select', 'replay'):
        if stage not in res:
            continue
        L += ['## %s' % stage.capitalize(), '', '| Job | arm | status | calls / tools / new | commit | E mean | Δ vs NoMix | Δ vs None | pool best | pool gain | sel. loss | net | shadow C_A argmin | same? | commit−shadow Δ | conditional |', '|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---|']
        for job, arms in res[stage].items():
            for a, A in arms.items():
                if A.get('commit_e_mean') is None:
                    L.append('| %s | %s | %s | | %s | | | | | | | | | | | |' % (job.replace('RD02_', ''), a, A.get('status'), A.get('commit')))
                    continue
                pub, den = A['public_e'], A['e_none_mean']
                L.append('| %s | %s | %s | %s / %s / %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                    job.replace('RD02_', ''), a, A['status'], A['calls'], A['tool_calls'], A['new_evaluations'], A['commit_label'], _f(A['commit_e_mean']),
                    _p(100 * (statistics.fmean(pub['P_NoMixRecipe']) - A['commit_e_mean']) / den), _p(100 * (den - A['commit_e_mean']) / den), A['pool']['best_label'], _p(A['pool']['pool_gain_pp']),
                    _p(A['pool']['selection_loss_pp']), _p(A['pool']['net_pp']), A['shadow_ca_argmin']['label'], A['shadow_ca_argmin']['same_as_commit'], _p(A['shadow_ca_argmin']['paired_commit_vs_shadow']['delta_pp']), A['conditional_used']))
        L += ['', '### own materials (%s)' % stage, '', '| Job | arm | plan | label | rules | C_A | C_B | E |', '|---|---|---|---|---:|---:|---:|---:|']
        for job, arms in res[stage].items():
            for a, A in arms.items():
                for m, r in (A.get('own_materials') or {}).items():
                    L.append('| %s | %s | %s | %s | %d | %s | %s | %s |' % (job.replace('RD02_', ''), a, m, r['label'], r['n_rules'], _f(statistics.fmean(r['c_a']) if r['c_a'] else None), _f(statistics.fmean(r['c_b']) if r['c_b'] else None), _f(statistics.fmean(r['e']) if r['e'] else None)))
        L.append('')
    if res.get('replay_readout'):
        L += ['## Replay main readout (pp of None; > 0 = F_skill better)', '', '| Job | F_skill | F0 | Fixed_dev | skill vs F0 [t95] | skill vs NoMix | skill vs Fixed_dev | skill vs None | F0 vs NoMix | Menu_CA shadow | skill vs Menu |', '|---|---|---|---|---|---:|---:|---:|---:|---|---:|']
        for j, r in res['replay_readout'].items():
            L.append('| %s | %s | %s | %s | %s [%s, %s] | %s | %s | %s | %s | %s | %s |' % (j.replace('RD02_', ''), r['f_skill_commit'], r['f0_commit'], r['fixed_dev_commit'], _p(r['skill_vs_f0']['delta_pp']), _p(r['skill_vs_f0']['t95_pp'][0]), _p(r['skill_vs_f0']['t95_pp'][1]),
                                                                                       _p(r['skill_vs_nomix']['delta_pp']), _p((r['skill_vs_fixed_dev'] or {}).get('delta_pp')), _p(r['skill_vs_none']['delta_pp']), _p(r['f0_vs_nomix']['delta_pp']), r['menu_ca_shadow']['material'], _p(r['menu_ca_shadow']['delta_skill_vs_menu_pp'])))
        L += ['', '| comparison | n | mean pp | per job | W/T/L | max harm |', '|---|---:|---:|---|---|---:|']
        for k, v in res.get('replay_aggregate', {}).items():
            L.append('| %s | %d | %s | %s | %d/%d/%d | %s |' % (k, v['n'], _p(v['mean_pp']), {jj.replace('RD02_', ''): round(x, 2) for jj, x in v['per_job'].items()}, v['wins'], v['ties'], v['losses'], _p(v['max_harm_pp'])))
    return '\n'.join(L) + '\n'


def write_report(root: Path, res: dict) -> None:
    C = res['cards']
    cands, parent, child, frozen = C['formation'], C['parent'], C['revision'], C['frozen']
    rep, agg = res.get('replay_readout', {}), res.get('replay_aggregate', {})
    cost = res['cost']
    L = ['# %s — REPORT' % PACKAGE, '', '任务书：`%s`。身份：`%s`（所有批次已在历史研究中曝光；本包为开发复验）。完成：%s。' % (TASK, EXPOSURE, res['finished_local']),
         '阶段状态：%s。' % {s: v.get('status') for s, v in res['stages'].items()}, '', '## 0. 主结果（Replay L6/L7：F_skill 对 F0）', '']
    if agg:
        a = agg['skill_vs_f0']
        L.append('- **F_skill − F0**：等权 %s pp（逐 Job %s；胜/平/负 %d/%d/%d；最大伤害 %s）。' % (_p(a['mean_pp']), {j.replace('RD02_', ''): round(x, 2) for j, x in a['per_job'].items()}, a['wins'], a['ties'], a['losses'], _p(a['max_harm_pp'])))
        for k, lab in (('skill_vs_nomix', 'F_skill − NoMix'), ('skill_vs_fixed_dev', 'F_skill − Fixed_dev'), ('skill_vs_none', 'F_skill − None'), ('f0_vs_nomix', 'F0 − NoMix')):
            v = agg.get(k, {})
            L.append('- **%s**：等权 %s pp（逐 Job %s；W/T/L %s/%s/%s）。' % (lab, _p(v.get('mean_pp')), {j.replace('RD02_', ''): round(x, 2) for j, x in (v.get('per_job') or {}).items()}, v.get('wins'), v.get('ties'), v.get('losses')))
        for j, r in rep.items():
            L.append('- %s：F_skill 交付 %s，F0 交付 %s，Fixed_dev %s；同交付 %s；F_skill 与 Fixed_dev 同材料 %s；影子 C_A argmin = commit：F_skill %s / F0 %s；条件化使用：F_skill %s / F0 %s。' % (
                j.replace('RD02_', ''), r['f_skill_commit'], r['f0_commit'], r['fixed_dev_commit'], r['same_delivery_skill_f0'], r['skill_same_material_as_fixed_dev'], r['shadow']['f_skill']['same_as_commit'], r['shadow']['f0']['same_as_commit'],
                r['conditional_used']['f_skill'], r['conditional_used']['f0']))
    else:
        L.append('- Replay 未完成或未评分。')
    L += ['', '## 1. 卡片', '']
    for sid, sj in (cands.get('skills') or {}).items():
        m = cands['meta'][sid]
        L += ['### %s（%s）' % (sid, m.get('research_mode')), '', '```', sj['rendered_body'], '```', '', '- commit policy: %s' % m.get('commit_policy'), '- decision change vs no card: %s' % m.get('decision_change_vs_no_card'),
              '- rule status: %s' % '; '.join('%s [%s]' % (r['rule'][:120], r['status']) for r in m.get('rule_status', [])), '']
    if child:
        L += ['### 修订（父卡 %s）：%s' % (child.get('parent'), child.get('child_status')), '']
        if child.get('skill'):
            L += ['```', child['skill']['rendered_body'], '```', '', '- changed mechanism: %s' % child['meta'].get('changed_mechanism'), '- supporting feedback: %s' % child['meta'].get('supporting_feedback'), '- expected trigger: %s' % child['meta'].get('expected_trigger'), '']
        else:
            L += ['- rationale: %s' % (child.get('rationale') or '')[:1500], '']
    if frozen:
        L += ['### 冻结：W* = %s（J %s vs f0 %s）；Fixed_dev = %s' % ((frozen.get('W_star') or {}).get('skill_id'), _f((frozen.get('W_star') or {}).get('J')), _f((frozen.get('W_star') or {}).get('f0_J')), (frozen.get('Fixed_dev') or {}).get('label')), '']
    L += ['## 2. 费用', '', '- 物理拟合 %s（成功 %s / 失败 %s / 重试 %s），缓存命中 %s；Fast+Slow 请求 %s（HTTP %s），token in/out %s/%s，未知 %s；新增 SHA 0，git commit 0。' % (
        cost.get('fit_attempts'), cost.get('fits_ok'), cost.get('fits_failed'), cost.get('retries_used'), cost.get('cache_hits'), cost.get('llm_requests'), cost.get('llm_http_attempts'), cost.get('llm_tokens_in'), cost.get('llm_tokens_out'), cost.get('llm_tokens_unknown')),
          '', '详细表：`tables.md`；机器可读：`result.json`、`frozen_config.json`、`budget.json`、`evidence/`、`formation/`、`revision/`、各阶段目录。']
    (root / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


# ============================================================================= package driver
def package_run(root: Path = ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    status = context.read_json(root / 'package_status.json') if (root / 'package_status.json').exists() else {}

    def stop(where, st):
        status.update({'stopped_at': where, 'stage_status': st, 'epoch': time.time(), 'local': now()})
        context.write_json(root / 'package_status.json', status)
        print('PACKAGE_STOP', where, st.get('status') if isinstance(st, dict) else st, flush=True)
        return status
    preflight(root)
    status['smoke'] = smoke(root)['all_pass']
    w = wiring(root)
    status['wiring'] = w['status']
    if w['status'] != 'PASS':
        return stop('wiring', w)
    cands = formation(root)
    status['formation'] = cands['status']
    if cands['status'] != 'PROPOSED' or cands['n_candidates'] < 1:
        return stop('formation', cands)
    st = practice_stage(root, cands)
    status['practice'] = st
    if st['status'] != 'FINISHED':
        return stop('practice', st)
    parent = choose_parent(root, cands)
    status['parent'] = parent['status']
    if parent['status'] != 'PARENT_CHOSEN':
        return stop('parent', parent)
    child = revision(root, cands, parent)
    status['revision'] = child['child_status']
    if child['child_status'] == 'REVISION_TECHNICAL_FAILURE':
        return stop('revision', child)
    st = select_stage(root, cands, child)
    status['select'] = st
    if st['status'] != 'FINISHED':
        return stop('select', st)
    frozen = freeze_choices(root, cands, child)
    status['frozen'] = frozen['status']
    if frozen['status'] != 'FROZEN':
        return stop('freeze', frozen)
    st = replay_stage(root, frozen)
    status['replay'] = st
    if st['status'] != 'FINISHED':
        return stop('replay', st)
    status['finished_local'] = now()
    context.write_json(root / 'package_status.json', status)
    readout(root)
    print('PACKAGE_FINISHED', flush=True)
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--resume-stage', metavar='STAGE')
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--restart', default='')
    ap.add_argument('--stage-worker', metavar='CFG')
    a = ap.parse_args()
    if a.stage_worker:
        stage_worker(Path(a.stage_worker))
        return
    assert 'torch' not in sys.modules, 'controller must not import torch'
    if a.preflight:
        preflight()
    elif a.smoke:
        preflight()
        smoke()
    elif a.wiring:
        wiring()
    elif a.resume_stage:
        if a.resume_stage not in ('practice', 'select', 'replay'):
            raise SystemExit('resumable stages: practice, select, replay')
        resume_stage(ROOT, a.resume_stage, accept_unknown_usage=a.accept_unknown_usage, restart=tuple(x for x in a.restart.split(',') if x))
    elif a.run:
        package_run()
    elif a.result:
        readout()
    assert 'torch' not in sys.modules, 'controller imported torch'


if __name__ == '__main__':
    main()
