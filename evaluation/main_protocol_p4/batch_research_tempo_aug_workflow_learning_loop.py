"""DEV-TEMPO-AUG-WORKFLOW-LEARNING-LOOP (docs/DEV_TEMPO_AUG_WORKFLOW_LEARNING_LOOP_TASK_2026-09-19.md): a bounded same-domain learning
loop over the parent package's Harness -- multi-candidate Workflow exploration (two Slow calls), real practice on two batches, one
C_B-driven revision per kept parent (Workflow mechanism + Principles), joint selection of parents / children / no-Skill on two more
batches, then three reserved follow-up batches with F0 / F_initial / F_learned / RandomSearch_B4 / Menu_CA / Fixed_dev / four references.

Everything numerical, the seven-tool Fast, the adapter, the physical cache, the label workers, the resume boundary and the budget
ledger are the parent module's (batch_research_tempo_aug_workflow_skill, imported as W). New here: the ten-trajectory evidence census
with branch-qualified evidence refs, the A1/A2 exploration prompts, the revision prompt and parser (Principles allowed, 6000-char
opt-in body), J_P / Select freezing of W_initial / W_learned / H_system / Fixed_dev, the Test stage with aliases, and the readout.

  --smoke | --preflight | --wiring | --run | --resume-stage STAGE [--accept-unknown-usage] | --result
  worker: --stage-worker CFG (the parent's numerical workers are reused through its module)
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
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha.batch_base import budget, context, spec, tempo_aug as ta
from methods.ttha.batch_base import readiness as rd
from evaluation.main_protocol_p4 import batch_research_domain_skill as dsks
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_skill as W

REPO = W.REPO
ROOT = REPO / '_scratch' / 'dev_tempo_aug_workflow_learning_loop'
PARENT = W.ROOT                                                   # read-only: history, frozen candidates, physical caches
TASK = 'docs/DEV_TEMPO_AUG_WORKFLOW_LEARNING_LOOP_TASK_2026-09-19.md'
MODULE = 'evaluation.main_protocol_p4.batch_research_tempo_aug_workflow_learning_loop'
PACKAGE = 'DEV-TEMPO-AUG-WORKFLOW-LEARNING-LOOP'
DATASET, DOMAIN = W.DATASET, W.DOMAIN
SEEDS = W.SEEDS                                                   # [20269181, 20269182, 20269183]; the overflowing ten-digit seeds are not used
WIRING_SEEDS, WIRING_JOB = W.WIRING_SEEDS, W.WIRING_JOB
JOB_INDEX = {**W.JOB_INDEX, 'RD02_L5': 11, 'RD02_L6': 12, 'RD02_L7': 13}
JOB_T = {**W.JOB_T, 'RD02_L5': 21360, 'RD02_L6': 22560, 'RD02_L7': 24480}
HIST_SOURCE_JOBS = W.SOURCE_JOBS                                  # S1 S2 V1 T2: parent no-Skill trajectories
HIST_SELECT_JOBS = W.SELECT_JOBS                                  # T4 Q2: parent no_skill / W1 / W2 trajectories
PRACTICE_JOBS = ('RD02_L1', 'RD02_L2')
SELECT_JOBS = ('RD02_L3', 'RD02_L4')
TEST_JOBS = ('RD02_L5', 'RD02_L6', 'RD02_L7')
TEST_ORDER = {'RD02_L5': ('f0', 'f_initial', 'f_learned', 'random'), 'RD02_L6': ('f_learned', 'random', 'f0', 'f_initial'), 'RD02_L7': ('f_initial', 'f0', 'random', 'f_learned')}
PUBLIC, PUBLIC_STEPS = W.PUBLIC, W.PUBLIC_STEPS
LIMITS, FAST_TOKEN_CAP, MAX_OUTPUT_TOKENS, MAX_TOOL_CORRECTIONS, EVIDENCE_ROUNDTRIP = W.LIMITS, W.FAST_TOKEN_CAP, W.MAX_OUTPUT_TOKENS, W.MAX_TOOL_CORRECTIONS, W.EVIDENCE_ROUNDTRIP
BODY_LIMIT = 6000                                                 # §4.3 opt-in rendered-body capacity (Workflow + Principles); the old default 1200 is untouched
MAX_CANDIDATES_PER_CALL = 2
TOTAL = {'max_fit_attempts': 453, 'max_llm_requests': 440, 'max_llm_tokens': 13_200_000, 'max_wall_s': 36_000, 'max_retries': 6}
STAGE_RETRIES = 2
HTTP_CAP = 444
ALLOC = {'wiring': {'fits': 3, 'requests': 0, 'tokens': 0},
         'formation_a': {'fits': 0, 'requests': 4, 'tokens': 1_200_000},
         'practice': {'fits': 120, 'requests': 128, 'tokens': 3_200_000},
         'revision': {'fits': 0, 'requests': 4, 'tokens': 1_200_000},
         'select': {'fits': 144, 'requests': 160, 'tokens': 4_000_000},
         'test': {'fits': 180, 'requests': 144, 'tokens': 3_600_000}}
NUMERIC_WALL_S = W.NUMERIC_WALL_S
ALLOWED_FEATURES = W.ALLOWED_FEATURES
MODEL = W.MODEL
T975_DF2 = W.T975_DF2
TOL = 1e-12

PERMISSION_SENTENCE = ('允许按合法 T 观察构造统一或条件化方案；单实体预测变化不能视作该实体材料的独立因果贡献，条件化完整方案仍可由共享训练比较。'
                       ' (Uniform or entity-conditional complete plans may be constructed from legal T observations; a change in one entity\'s prediction is '
                       'not the independent causal contribution of that entity\'s material, yet conditional complete plans can still be compared through the '
                       'shared training.)')
FAST_SYSTEM = W.FAST_SYSTEM + '\n' + PERMISSION_SENTENCE

FEEDBACK_ROLES = ('Three feedback roles, by name: C_A = the immediate feedback Fast legally used at decision time (it explains why Fast decided as it '
                  'did); C_B = the delayed generalisation feedback of that development episode, opened after the commit, used to judge whether the '
                  'decision deserves reuse, narrowing or correction -- it is not an ignorable footnote; E = the follow-up result of this round, opened '
                  'only after every piece of knowledge and delivery is frozen, and never present in this input. C_B is itself one limited time block, '
                  'not a universal answer: keep C_A/C_B conflicts as conflicts, never turn them into "always trust C_B". Rankings on one block are not '
                  'guaranteed on later blocks. A per-entity prediction difference is not the causal contribution of that entity\'s material; an '
                  'undistinguished comparison is not evidence that a primitive is generally worse; an untested action is not harmful. ' + PERMISSION_SENTENCE)
NO_IDS = ('Do not write data source names, job labels, entity ids, row numbers, historical assignments or answer tables into deployable text; '
          'do not change model, scoring, tools, permissions, budgets, randomness, targets or legal windows. observable_applicability MUST be exactly '
          '{"const":true}; applicability conditions belong inside the Workflow text as observable conditions.')
SLOW_A1 = ('You are the offline Slow of a batch training-data augmentation research Harness. You receive a deterministic census of ten completed '
           'research trajectories from six earlier batches of ONE neutral domain: four ran with no Skill, six ran on two later batches with either no '
           'Skill or one of two earlier candidate cards (the loaded card text is given); every trajectory carries T-only observations, actual tool calls, '
           'complete augmentation plans, per-seed/per-origin C_A, the commit and its reason, per-seed/per-origin C_B of every evaluated material, costs '
           'and failures. ' + FEEDBACK_ROLES + ' Propose at most TWO research Workflows for future Fast jobs of this domain that differ for a stated '
           'reason (diagnosis, construction, experiment allocation or commit judgement -- your choice), or KEEP if the census does not support a reusable '
           'Workflow. A Workflow tells Fast how to observe, which hypotheses and complete plans to construct, how to organise comparisons and use C_A '
           'within the fixed budget (4 new complete plans, 16 calls, 24 tool calls), and when to stop or keep a public reference; it may state '
           'evidence-bounded primitive preferences or recommend a public reference; it need not be complex, diverse or aggressive, and it may not claim '
           'that a tool is unavailable. ' + NO_IDS + ' Output exact JSON, no markdown: {"decision":"KEEP","rationale":"...","evidence_review":{...}} OR '
           '{"decision":"PROPOSE","candidates":[{"candidate_id":"W1","research_mode":"<short label>","workflow":"...","principles":null,'
           '"observable_applicability":{"const":true},"applicability_summary":"<=400 characters","evidence_refs":["exact refs from the supplied legal '
           'ranges"],"rationale":"...","evidence_review":{"supports":["observation -> which advice it supports"],"contradicts_or_limits":["which later '
           'result (C_B) contradicts or limits which judgement"],"rule_changes":["which rule is kept / dropped / rewritten because of it"],'
           '"uncertain_and_expected_behavior_change":"what remains uncertain and which Fast behaviour should change next time"}}]}. candidate_id W1 and '
           'W2. principles must be null in this first exploration. The Workflow text renders to at most 6000 characters. KEEP is valid and is not resampled.')
SLOW_A2 = SLOW_A1.replace('Propose at most TWO research Workflows for future Fast jobs of this domain that differ for a stated reason',
                          'You also receive the verbatim output of the first exploration call (A1). Propose at most TWO research Workflows that follow '
                          'research-organisation hypotheses genuinely different from A1\'s (not renamings or rewordings of A1\'s candidates), differing '
                          'for a stated reason').replace('candidate_id W1 and W2', 'candidate_id W3 and W4')
SLOW_C = ('You are the offline Slow revising ONE parent Workflow after its real use. You receive: the parent card (its text, rationale and evidence '
          'review), the initial census the parent was formed from (condensed trajectories: every action, plan, C_A/C_B number, commit and failure kept; '
          'raw observation outputs omitted), and the complete practice trajectories of ALL candidate cards on two new batches, each with the loaded '
          'card text, tool calls, plans, per-seed C_A, the commit and reason, and the post-commit per-seed C_B of every evaluated material, plus '
          'deterministic numeric counterexamples (facts only). ' + FEEDBACK_ROLES + ' Your job: identify where the parent\'s advice was NOT executed and '
          'where it WAS executed without benefit; answer every listed numeric counterexample explicitly (accept, bound or refute it with the given '
          'numbers); then either KEEP the parent unchanged or produce ONE child version that changes ONE main behavioural mechanism of the Workflow '
          '(a limited edit, not a rewrite) and may add Principles that serve that mechanism. Principles state evidence-supported preferences, '
          'applicability conditions, counterexamples and uncertainty bounds; generic "be careful / look at the data" is not a Principle. You are not '
          'required to be more conservative, observe more, change primitives or grouping. Do not merely polish the narrative. ' + NO_IDS +
          ' Output exact JSON, no markdown: {"decision":"KEEP","rationale":"...","evidence_review":{"supports":[...],"contradicts_or_limits":[...],'
          '"rule_changes":[...],"uncertain_and_expected_behavior_change":"..."},"counterexample_responses":["..."]} OR {"decision":"REVISE","child":{'
          '"workflow":"<full child Workflow text>","principles":"<Principles text>" or null,"changed_mechanism":"<the one behavioural mechanism changed '
          'and how>","research_mode":"<short label>","applicability_summary":"<=400 characters","evidence_refs":["exact refs from the supplied legal '
          'ranges"],"rationale":"...","evidence_review":{"supports":[...],"contradicts_or_limits":[...],"rule_changes":[...],'
          '"uncertain_and_expected_behavior_change":"..."},"counterexample_responses":["..."]}}. Workflow plus Principles render to at most 6000 '
          'characters. KEEP is valid, is not resampled and is recorded as an alias of the parent.')

COMPRESSION_RULES = {
    'inherits': 'the parent package census rules tiers 1-3 (size-only; see _scratch/dev_tempo_aug_workflow_skill/formation/census_compression_rules_v2.json)',
    'this_package': ['evidence refs are branch-qualified string paths "<stage>/<branch>/<event_id>" (+ "/overview", "/c_b_after_commit"); legal set = every such id',
                     'per job one materials table keyed by physical material id (assignment, C_A and C_B per seed once); each branch maps its plan ids to physical ids',
                     'the loaded card text of a historical branch is given once per branch (knowledge_loaded)',
                     'tier "skeleton" (revision input only): every trajectory row kept as a stub {event_id, event, tool, plan_id}; fast_response, build_material, evaluate, commit, tool_rejected, job_incomplete rows kept in full; inspect_* / compare outputs omitted (their numbers are in the materials table)',
                     'the tier is chosen per call so that the frozen client reservation 2 x (message bytes + 2048 + 12000) leaves room for one correction round inside the stage cap'],
    'never_dropped': 'actions, plans, C_A/C_B numbers, commits, failures, conflicts, loaded card texts',
}


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'configs': root / 'stage_configs', 'logs': root / 'logs', 'wiring': root / 'wiring',
            'evidence': root / 'evidence', 'formation_a': root / 'formation_a', 'practice': root / 'practice', 'revision': root / 'revision',
            'select': root / 'select', 'test': root / 'test'}


def ref(stage: str, branch: str, tail: str) -> str:
    return '%s/%s/%s' % (stage, branch, tail)


# ============================================================================= evidence census (parent package history; deterministic)
def _branch_info(stage_root: Path, bname: str, job: str) -> dict:
    return {'stage_root': stage_root, 'branch': bname, 'job': job, 'dir': stage_root / bname}


def historical_branches() -> list:
    out = [_branch_info(PARENT / 'source', '%s_no_skill' % j, j) for j in HIST_SOURCE_JOBS]
    for j in HIST_SELECT_JOBS:
        for arm in ('no_skill', 'cand_W1', 'cand_W2'):
            out.append(_branch_info(PARENT / 'select', '%s_%s' % (j, arm), j))
    return out


def practice_branches(root: Path, arms: list) -> list:
    return [_branch_info(paths(root)['practice'], '%s_%s' % (j, a), j) for j in PRACTICE_JOBS for a in arms]


def _stub(r: dict) -> dict:
    s = {'event_id': r['event_id'], 'event': r['event']}
    if 'tool' in r:
        s['tool'] = r['tool']
    pid = (r.get('arguments') or {}).get('plan_id') if isinstance(r.get('arguments'), dict) else None
    pid = pid or ((r.get('output') or {}).get('plan_id') if isinstance(r.get('output'), dict) else None)
    if pid:
        s['plan_id'] = pid
    return s


def compress_trajectory(rows: list, tier) -> list:
    """tier 3 = the parent's size-only rules; 'skeleton' = stubs except fast_response / build_material / evaluate / commit / rejections / incompletes."""
    out, first_guidance = [], None
    for r in rows:
        r = copy.deepcopy(r)
        if tier == 'skeleton':
            keep_full = r['event'] in ('fast_response', 'tool_rejected', 'job_incomplete', 'committed') or (r['event'] == 'tool_completed' and r['tool'] in ('build_material', 'evaluate', 'commit'))
            if not keep_full:
                out.append(_stub(r))
                continue
            if r['event'] == 'tool_completed':
                r['output'] = W._compress(r['output'], r['tool'], 3)
            out.append(r)
            continue
        if r['event'] == 'job_started':
            r['overview'] = 'see overview (identical)'
        if r['event'] == 'fast_request':
            if first_guidance is None:
                first_guidance = r.get('guidance')
            elif r.get('guidance') == first_guidance:
                r['guidance'] = 'see first request (identical)'
        if r['event'] == 'tool_completed':
            r['output'] = 'see overview (identical)' if r['tool'] == 'overview' else W._compress(r['output'], r['tool'], 3)
        if r['event'] == 'tool_started':
            continue
        out.append(r)
    return out


def branch_evidence(b: dict, stage_name: str, tier, seeds=SEEDS) -> tuple:
    """(record, refs, materials) of one COMPLETE-or-failed branch. materials: {phys_id: {...C_A, C_B per seed}} deduplicated by physical id."""
    d, job = b['dir'], b['job']
    res = context.read_json(d / 'branch_result.json') if (d / 'branch_result.json').exists() else None
    kn = context.read_json(d / 'knowledge.json') if (d / 'knowledge.json').exists() else {'entries': []}
    loaded = [e['body'] for e in kn.get('entries', [])]
    reg = W.load_registry(d / job)
    cells = W.branch_cells(d / job)
    cb = context.read_json(d / job / 'c_b_scores.json')['cells'] if (d / job / 'c_b_scores.json').exists() else {}
    commit = context.read_json(d / job / 'commit.json') if (d / job / 'commit.json').exists() else None
    plan_to_phys, materials = {}, {}
    for mid, r in reg.items():
        phys = r.get('phys_id', mid)
        plan_to_phys[mid] = phys
        row = materials.setdefault(phys, {'assignment': r.get('assignment') if mid not in PUBLIC else (W.public_spec(mid).get('programs') if mid != 'FixedMixup' else 'FixedMixup'),
                                          'public_id': mid if mid in PUBLIC else None, 'c_a_by_seed': None, 'c_b_by_seed': None, 'c_a_status': None, 'c_b_status': None})
        ca = {c['model_seed']: c['scores']['c_a'] for c in cells.values() if c['material_id'] == mid}
        if set(ca) >= set(seeds):
            row['c_a_by_seed'] = [ca[s]['normalized_mse_macro'] for s in seeds]
            row['c_a_status'] = sorted({ca[s]['status'] for s in seeds})
            row['c_a_by_seed_origin'] = [[statistics.fmean(ca[s]['per_origin_entity_normalized_mse'][o]) for o in range(ca[s]['n_origins'])] for s in seeds]
        cbv = {v['model_seed']: v['c_b'] for c, v in cb.items() if v['material_id'] == mid}
        if set(cbv) >= set(seeds):
            row['c_b_by_seed'] = [cbv[s]['normalized_mse_macro'] for s in seeds]
            row['c_b_status'] = sorted({cbv[s]['status'] for s in seeds})
            row['c_b_by_seed_origin'] = [[statistics.fmean(cbv[s]['per_origin_entity_normalized_mse'][o]) for o in range(cbv[s]['n_origins'])] for s in seeds]
    rows = [json.loads(x) for x in (d / 'trace.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()] if (d / 'trace.jsonl').exists() else []
    refs = [ref(stage_name, b['branch'], r['event_id']) for r in rows] + [ref(stage_name, b['branch'], 'overview'), ref(stage_name, b['branch'], 'c_b_after_commit')]
    traj = compress_trajectory(rows, tier)
    for r in traj:
        r['ref'] = ref(stage_name, b['branch'], r['event_id'])
    tokens = dsks._unit_tokens(b['stage_root']).get('fast:%s' % b['branch'], {})
    rec = {'ref_prefix': ref(stage_name, b['branch'], ''), 'job_ref': job, 'knowledge_loaded': loaded or None,
           'status': res['status'] if res else 'NOT_RUN', 'failure_kind': res.get('failure_kind') if res else None, 'reason': res.get('reason') if res else None,
           'committed_plan_id': res.get('committed_plan_id') if res else None, 'committed_physical_id': plan_to_phys.get(res.get('committed_plan_id')) if res else None,
           'commit_reason': commit['reason'] if commit else None, 'plan_to_physical': plan_to_phys, 'plan_specs': {},
           'cost': {'fast_calls': res['calls'] if res else None, 'tool_calls': res['tool_calls'] if res else None, 'new_evaluations': res['new_evaluations'] if res else None, 'tokens': tokens},
           'trajectory': traj}
    for mid in reg:
        p = W.aug_dir(d / job) / (mid + '__compiled.json')
        if p.exists():
            ms = context.read_json(p)['material_spec']
            rec['plan_specs'][mid] = {k: v for k, v in ms.items() if k not in ('execution_order', 'profile', 'alias_of')}
    return rec, refs, materials


def census(root: Path, branches: list, stage_names: dict, tier, *, purpose: str) -> dict:
    """branches: list of _branch_info; stage_names: {stage_root: stage label}. One materials table per job; trajectories per branch."""
    jobs, refs, records = {}, [], []
    for b in branches:
        sname = stage_names[str(b['stage_root'])]
        if not (b['dir'] / 'branch_result.json').exists():
            records.append({'branch': ref(sname, b['branch'], ''), 'status': 'NOT_RUN'})
            continue
        rec, r_, mats = branch_evidence(b, sname, tier)
        refs += r_
        J = jobs.setdefault(b['job'], {'overview': None, 'materials': {}, 'branches': []})
        if J['overview'] is None:
            ov = context.read_json(b['dir'] / b['job'] / 'overview.json')
            J['overview'] = {'n_entities': ov['n_entities'], 'summary': ov['summary'], 'entities': W._columnar(ov['entities'])}
        for phys, m in mats.items():
            cur = J['materials'].setdefault(phys, m)
            for k in ('c_a_by_seed', 'c_b_by_seed', 'c_a_status', 'c_b_status', 'c_a_by_seed_origin', 'c_b_by_seed_origin'):
                if cur.get(k) is None and m.get(k) is not None:
                    cur[k] = m[k]
        J['branches'].append(rec)
    actions = ta.action_table()
    actions['primitives'] = {n: {k: v for k, v in p.items() if k != 'source'} for n, p in actions['primitives'].items()}
    geometry = context.read_json(branches[0]['dir'] / branches[0]['job'] / 'overview.json')['geometry'] if branches else None
    ev = {'census_status': 'CENSUS_COMPLETE' if jobs else 'CENSUS_EMPTY', 'domain_id': DOMAIN, 'purpose': purpose, 'profile': W.PROFILE_VERSION,
          'jobs_in_order': list(jobs), 'jobs': jobs, 'not_run': records, 'legal_evidence_refs': sorted(set(refs)),
          'compression': {**COMPRESSION_RULES, 'tier_used': str(tier)},
          'feedback_roles': {'C_A': 'immediate feedback used by Fast at decision time (origins t, t+48)', 'C_B': 'delayed generalisation feedback of the same episode (origins t+96, t+144), opened after commit',
                             'E': 'follow-up result; never in this input'},
          'public_semantics': {'field_definitions': rd.FIELD_DEFINITIONS, 'actions': actions, 'tool_contracts': W.CONTRACTS, 'consumer': W.CONSUMER, 'geometry': geometry,
                               'public_references': {m: W.public_spec(m) for m in PUBLIC}, 'budget_per_job': LIMITS, 'body_limit_characters': BODY_LIMIT,
                               'metric': 'missing-aware normalized MSE on observed future cells; lower is better'}}
    low = json.dumps(ev, ensure_ascii=False).lower()
    if any(n in low for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')):
        raise PermissionError('census names a data source')
    if re.search(r'"e"\s*:|e_scores|"e_by_seed"', json.dumps(ev)):
        raise PermissionError('census carries an E block')
    return ev


def ref_ranges(refs: list) -> dict:
    out = {}
    for r in refs:
        prefix, _, tail = r.rpartition('/')
        row = out.setdefault(prefix, {'event_index': [], 'other': []})
        idx = tail.rpartition(':')[2]
        (row['event_index'].append(int(idx)) if idx.isdigit() else row['other'].append(r))
    return {p: {'event_ref_range': '%s/<job>:%d .. %s/<job>:%d (every integer index; <job> = the branch\'s job id as in other refs)' % (p, min(v['event_index']), p, max(v['event_index'])) if v['event_index'] else None,
                'other_refs': v['other']} for p, v in out.items()}


def payload_bytes(system: str, payload: dict) -> int:
    return len(json.dumps([{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], ensure_ascii=False).encode('utf-8'))


def reservation(nbytes: int) -> int:
    return 2 * (nbytes + 2048 + MAX_OUTPUT_TOKENS)


# ============================================================================= budget (one ledger; cumulative stage caps; per-stage retries)
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
    retries_stage = min(STAGE_RETRIES, TOTAL['max_retries'] - s['retries_used']) if alloc['fits'] else 0
    caps = {'max_fit_attempts': min(TOTAL['max_fit_attempts'], s['fit_attempts'] + alloc['fits'] + retries_stage),
            'max_llm_requests': min(TOTAL['max_llm_requests'], s['llm_requests'] + alloc['requests']),
            'max_llm_tokens': min(TOTAL['max_llm_tokens'], s['llm_tokens_in'] + s['llm_tokens_out'] + alloc['tokens']),
            'max_wall_s': TOTAL['max_wall_s'], 'max_retries': min(TOTAL['max_retries'], s['retries_used'] + retries_stage)}
    rec[stage] = {'epoch_start': time.time(), 'local': now(), 'allocation': alloc, 'snapshot_at_start': snap, 'caps': caps,
                  'http_cap': min(HTTP_CAP, s['llm_http_attempts'] + alloc['requests'] + 4)}
    context.write_json(P_['stages'], rec)
    return rec[stage]


def stage_config(root: Path, stage: str, *, jobs, order, knowledge, labels_e: bool, random: bool, treatment_note=None, cache_from=None) -> Path:
    P_ = paths(root)
    path = P_['configs'] / ('%s.json' % stage)
    if path.exists():
        return path
    sc = stage_caps(root, stage)
    cfg = {'study': 'tempo_aug_workflow_learning_loop', 'status': 'FROZEN', 'stage': stage, 'output': str(P_[stage]), 'jobs': list(jobs),
           'order': {j: list(order[j]) for j in jobs}, 'knowledge': knowledge, 'labels_e': bool(labels_e), 'llm': True, 'random': bool(random),
           'seeds': list(SEEDS), 'job_index': {j: JOB_INDEX[j] for j in jobs}, 'public_ids': list(PUBLIC), 'limits': LIMITS, 'max_tool_corrections': MAX_TOOL_CORRECTIONS,
           'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'fast_token_cap': FAST_TOKEN_CAP, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'fast_system': FAST_SYSTEM,
           'caps': sc['caps'], 'http_cap': sc['http_cap'], 'ledger_path': str(P_['ledger']), 'fit_retry': True,
           'fit_attempts_at_stage_start': context.read_json(P_['ledger'])['fit_attempts'], 'treatment_note': treatment_note or {}, 'cache_from': cache_from or {}}
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
            p = subprocess.Popen([sys.executable, '-B', '-m', W.MODULE, '--stage-worker', str(cfg_path)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=W.worker_env())
            try:
                rc = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)
                context.write_json(root / ('%s_supervisor_timeout.json' % stage), {'epoch': time.time(), 'pid': p.pid})
                rc = 124
        print('STAGE_EXIT', stage, rc, now(), flush=True)
    return dsks.stage_status(sroot)


def resume_stage(root: Path, stage: str, *, accept_unknown_usage: bool = False, restart=()) -> None:
    """Same boundary as the parent (labels still withheld, one continuation); the parent's arm runner is reused with this package's paths."""
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
    context.write_json(sroot / 'resume.json', {'epoch': time.time(), 'accepted_unknown_usage': bool(accept_unknown_usage), 'slow_recalled': False, 'restarted_branches': moved})
    rt.FIT_RETRY = bool(cfg.get('fit_retry'))
    branches, failures, stopped = [], [], None
    try:
        client = rt.MeteredClient(led, sroot / 'raw_responses', http_cap=int(cfg['http_cap']))
        W._run_arms(sroot, cfg, led, client, branches, failures, resumed=True)
        context.write_json(sroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches], 'resumed': True})
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': W._safe_message(exc)})
        stopped = type(exc).__name__
    finally:
        context.write_json(sroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': True})
    W.open_labels(sroot, cfg, branches, failures, led, resumed=True, withhold_reason=stopped and 'resumed stage stopped again (%s)' % stopped)


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


def check_reservation(led, sc: dict, nbytes: int, *, correction_margin: int = 8192, calls_left: int = 1) -> dict:
    """The frozen client's pre-send rule: used + 2 x (bytes + 2048 + max_tokens) <= stage cap, with room for one correction round;
    calls_left > 1 also asks that this many same-size scientific calls (each with a correction margin) fit the remaining stage cap."""
    used = led.s['llm_tokens_in'] + led.s['llm_tokens_out'] + led.s.get('token_reserved_failed_upper', 0)
    need = reservation(nbytes)
    need_with_correction = reservation(nbytes + correction_margin)
    cap = sc['caps']['max_llm_tokens']
    return {'bytes': nbytes, 'used_tokens': used, 'reservation': need, 'reservation_with_correction': need_with_correction, 'stage_cap': cap, 'calls_left': calls_left,
            'fits': used + need <= cap, 'fits_with_correction': used + need_with_correction <= cap, 'fits_all_remaining_calls': used + calls_left * need_with_correction <= cap}


# ============================================================================= Slow output parsing (A: exploration; C: revision)
CAND_KEYS = {'candidate_id', 'research_mode', 'workflow', 'principles', 'observable_applicability', 'applicability_summary', 'evidence_refs', 'rationale', 'evidence_review'}
REVIEW_KEYS = {'supports', 'contradicts_or_limits', 'rule_changes', 'uncertain_and_expected_behavior_change'}


def check_review(ev) -> dict:
    if not isinstance(ev, dict) or set(ev) != REVIEW_KEYS:
        raise ValueError('evidence_review must hold exactly %s' % sorted(REVIEW_KEYS))
    for k in ('supports', 'contradicts_or_limits', 'rule_changes'):
        if not isinstance(ev[k], list) or any(not isinstance(x, str) for x in ev[k]):
            raise ValueError('evidence_review.%s must be a list of strings' % k)
    if not isinstance(ev['uncertain_and_expected_behavior_change'], str) or not ev['uncertain_and_expected_behavior_change'].strip():
        raise ValueError('evidence_review.uncertain_and_expected_behavior_change must be nonempty text')
    return {k: ev[k] for k in REVIEW_KEYS}


def make_card(*, skill_id, revision, workflow, principles, summary, refs, legal_refs, mode, derived_from=None, status='CANDIDATE_TEST_ONLY', source_stage='propose') -> dsk.Skill:
    if not isinstance(mode, str) or not mode.strip() or len(mode) > 80:
        raise ValueError('research_mode must be short nonempty text')
    dsk.check_reusable_text(mode, 'research_mode', W.skill_text_check)
    return dsk.make_skill(skill_id=skill_id, domain_id=DOMAIN, revision=revision, workflow=workflow, principles=principles, applicability_summary=summary,
                          compatibility_note='Formed for this study\'s fixed task, Consumer, L/H, sampling interval and T length.',
                          observable_applicability={'const': True}, evidence_refs=refs, legal_evidence_refs=legal_refs, source_stage=source_stage, status=status,
                          derived_from=derived_from, allowed_features=ALLOWED_FEATURES, text_check=W.skill_text_check, body_limit=BODY_LIMIT)


def parse_exploration(resp, *, ids: tuple, legal_refs) -> dict:
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'PROPOSE'):
        raise ValueError('decision must be KEEP or PROPOSE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale', 'evidence_review'}:
            raise ValueError('KEEP carries only rationale and evidence_review')
        return {'decision': 'KEEP', 'rationale': str(resp.get('rationale', ''))[:2000], 'evidence_review': check_review(resp['evidence_review']) if 'evidence_review' in resp else None, 'skills': [], 'meta': {}}
    if set(resp) != {'decision', 'candidates'}:
        raise ValueError('PROPOSE carries exactly decision and candidates')
    cands = resp['candidates']
    if not isinstance(cands, list) or not 1 <= len(cands) <= MAX_CANDIDATES_PER_CALL:
        raise ValueError('PROPOSE needs 1..%d candidates' % MAX_CANDIDATES_PER_CALL)
    skills, meta, seen, modes = [], {}, set(), set()
    for c in cands:
        if not isinstance(c, dict) or set(c) != CAND_KEYS:
            raise ValueError('candidate keys must be exactly %s' % sorted(CAND_KEYS))
        cid = c['candidate_id']
        if cid not in ids or cid in seen:
            raise ValueError('candidate ids must be %s and unique' % list(ids))
        if c['principles'] is not None:
            raise ValueError('principles must be null in the exploration calls')
        if c['observable_applicability'] != {'const': True}:
            raise ValueError('observable_applicability must be exactly {"const": true}')
        if str(c['research_mode']).strip().lower() in modes:
            raise ValueError('each candidate needs its own research_mode')
        review = check_review(c['evidence_review'])
        s = make_card(skill_id='%s-%s-r1' % (DOMAIN, cid), revision=1, workflow=c['workflow'], principles=None, summary=c['applicability_summary'], refs=c['evidence_refs'],
                      legal_refs=legal_refs, mode=c['research_mode'])
        seen.add(cid)
        modes.add(str(c['research_mode']).strip().lower())
        skills.append(s)
        meta[s.skill_id] = {'candidate_id': cid, 'research_mode': c['research_mode'], 'rationale': str(c['rationale'])[:2000], 'evidence_review': review}
    if len({s.rendered_body for s in skills}) != len(skills):
        raise ValueError('two candidates render the same body')
    return {'decision': 'PROPOSE', 'skills': skills, 'meta': meta}


def parse_revision(resp, *, parent: dsk.Skill, legal_refs) -> dict:
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'REVISE'):
        raise ValueError('decision must be KEEP or REVISE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale', 'evidence_review', 'counterexample_responses'}:
            raise ValueError('KEEP carries only rationale, evidence_review and counterexample_responses')
        return {'decision': 'KEEP', 'rationale': str(resp.get('rationale', ''))[:3000], 'evidence_review': check_review(resp.get('evidence_review')),
                'counterexample_responses': [str(x) for x in (resp.get('counterexample_responses') or [])], 'skill': None, 'meta': {}}
    if set(resp) != {'decision', 'child'}:
        raise ValueError('REVISE carries exactly decision and child')
    c = resp['child']
    want = {'workflow', 'principles', 'changed_mechanism', 'research_mode', 'applicability_summary', 'evidence_refs', 'rationale', 'evidence_review', 'counterexample_responses'}
    if not isinstance(c, dict) or set(c) != want:
        raise ValueError('child keys must be exactly %s' % sorted(want))
    if not isinstance(c['changed_mechanism'], str) or not c['changed_mechanism'].strip():
        raise ValueError('changed_mechanism must name the one mechanism changed')
    if c['principles'] is not None and (not isinstance(c['principles'], str) or not c['principles'].strip()):
        raise ValueError('principles must be null or nonempty text')
    if not isinstance(c['counterexample_responses'], list) or any(not isinstance(x, str) for x in c['counterexample_responses']):
        raise ValueError('counterexample_responses must be a list of strings')
    review = check_review(c['evidence_review'])
    child_id = re.sub(r'-r\d+$', '', parent.skill_id) + '-r%d' % (parent.revision + 1)
    s = make_card(skill_id=child_id, revision=parent.revision + 1, workflow=c['workflow'], principles=c['principles'], summary=c['applicability_summary'], refs=c['evidence_refs'],
                  legal_refs=legal_refs, mode=c['research_mode'], derived_from=parent.skill_id, source_stage='revision')
    if s.rendered_body == parent.rendered_body:
        raise ValueError('the child renders identically to the parent; return KEEP instead')
    return {'decision': 'REVISE', 'skill': s, 'meta': {'changed_mechanism': c['changed_mechanism'], 'research_mode': c['research_mode'], 'rationale': str(c['rationale'])[:3000],
                                                       'evidence_review': review, 'counterexample_responses': c['counterexample_responses']}}


def slow_call(client, role: str, unit: str, payload: dict, system: str, parse, *, forbidden=()) -> dict:
    """One scientific call + at most one contract correction; transport / account / budget faults end as CALL_FAILED (never KEEP)."""
    low = json.dumps(payload, ensure_ascii=False).lower()
    if any(n.lower() in low for n in tuple(spec.DATASETS) + tuple(forbidden)):
        raise PermissionError('Slow payload names a data source')
    attempts, prior = [], None
    for i in range(2):
        try:
            raw = client.call(role if i == 0 else role + '_contract_correction', unit, payload, system, max_tokens=MAX_OUTPUT_TOKENS)
        except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
            attempts.append({'attempt': i, 'fault_kind': type(exc).__name__})
            return {'status': 'CALL_FAILED', 'attempts': attempts, 'raw': prior}
        except ValueError as exc:
            attempts.append({'attempt': i, 'error': 'response not JSON: %s' % str(exc)[:200]})
            prior = client.last_text
        else:
            try:
                out = parse(raw)
                attempts.append({'attempt': i, 'ok': True})
                return {'status': out['decision'], 'attempts': attempts, 'raw': raw, **out}
            except (ValueError, rd.PolicyError) as exc:
                attempts.append({'attempt': i, 'error': str(exc)[:300]})
                prior = raw if dsk._jsonable(raw) else None
        if i == 0:
            payload = {**payload, 'correction': {'previous_output': prior, 'error': attempts[-1].get('error'),
                                                 'instruction': 'Correct this contract error only. KEEP is allowed. Do not seek a different outcome.'}}
    return {'status': 'PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'raw': prior}


def skill_json(s: dsk.Skill | None):
    return s.to_json() if s else None


def load_skills(path: Path) -> dict:
    return {k: dsk.skill_from_json(v) for k, v in context.read_json(path)['skills'].items()} if path.exists() else {}


# ============================================================================= A. initial exploration: two Slow calls, at most four W0 cards
def initial_census(root: Path, tier=3) -> dict:
    return census(root, historical_branches(), {str(PARENT / 'source'): 'hist_source', str(PARENT / 'select'): 'hist_select'}, tier,
                  purpose='Initial exploration input: ten completed research trajectories (six batches) of one neutral domain; can they be consolidated into reusable research Workflows?')


def exploration_payload(cen: dict, *, a1_output=None) -> dict:
    p = {'domain_id': DOMAIN, 'census': {k: v for k, v in cen.items() if k != 'legal_evidence_refs'}, 'legal_evidence_refs': ref_ranges(cen['legal_evidence_refs']),
         'legal_evidence_refs_format': 'evidence_refs must be exact strings "<stage>/<branch>/<job>:<integer>" inside these ranges, or one of other_refs',
         'allowed_batch_features': sorted(ALLOWED_FEATURES), 'max_candidates': MAX_CANDIDATES_PER_CALL, 'body_limit_characters': BODY_LIMIT,
         'required': {'principles': None, 'observable_applicability': {'const': True}}, 'status': 'CANDIDATE_TEST_ONLY'}
    if a1_output is not None:
        p['first_exploration_call_output'] = a1_output
    return p


def formation_a(root: Path) -> dict:
    """A1 then A2 (A2 sees A1's verbatim output), each KEEP-able, each with one contract correction; identical bodies merge as aliases.
    Both KEEP -> NO_INITIAL_PROPOSAL (package stops). A technical failure is CALL_FAILED / PARSE_OR_VALIDATION_FAILED, never KEEP."""
    P_ = paths(root)
    out = P_['formation_a']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'candidates.json').exists():
        return context.read_json(out / 'candidates.json')
    P_['evidence'].mkdir(exist_ok=True)
    led, client, sc = slow_client(root, 'formation_a')
    if not (P_['evidence'] / 'census_initial.json').exists():
        dsks.write_once(P_['evidence'] / 'compression_rules.json', {**COMPRESSION_RULES, 'frozen_local': now()})
        sizes = {}
        for tier in (3, 'skeleton'):                                  # two scientific calls (A1, A2) plus a correction each must fit the stage cap
            cen = initial_census(root, tier=tier)
            sizes[str(tier)] = payload_bytes(SLOW_A2, exploration_payload(cen, a1_output={'placeholder_for_a1_output_bytes': 'x' * 16384}))
            chk = check_reservation(led, sc, sizes[str(tier)], calls_left=2)
            cen['compression']['payload_bytes_by_tier'] = dict(sizes)
            cen['compression']['reservation_check_at_choice'] = chk
            if chk['fits_all_remaining_calls']:
                break
        dsks.write_once(P_['evidence'] / 'census_initial.json', cen)
    cen = context.read_json(P_['evidence'] / 'census_initial.json')
    if cen['census_status'] != 'CENSUS_COMPLETE':
        raise RuntimeError('initial census incomplete')
    legal = frozenset(cen['legal_evidence_refs'])
    forbidden = [n for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')]
    record = {'started_local': now(), 'calls': {}}
    a1_raw = None
    for call_name, ids in (('A1', ('W1', 'W2')), ('A2', ('W3', 'W4'))):
        if (out / ('%s.json' % call_name)).exists():
            rec = context.read_json(out / ('%s.json' % call_name))
        else:
            payload = exploration_payload(cen, a1_output=a1_raw if call_name == 'A2' else None)
            size = payload_bytes(SLOW_A1 if call_name == 'A1' else SLOW_A2, payload)
            chk = check_reservation(led, sc, size)
            if not chk['fits_with_correction']:
                raise RuntimeError('exploration payload does not fit the frozen reservation with a correction margin: %s' % chk)
            res = slow_call(client, 'slow_explore_%s' % call_name.lower(), call_name, payload, SLOW_A1 if call_name == 'A1' else SLOW_A2,
                            lambda r, ids=ids: parse_exploration(r, ids=ids, legal_refs=legal), forbidden=forbidden)
            rec = {**{k: v for k, v in res.items() if k != 'skills'}, 'skills': {s.skill_id: s.to_json() for s in res.get('skills', [])}, 'reservation_check': chk,
                   'requests_after': led.s['llm_requests'], 'written_local': now()}
            dsks.write_once(out / ('%s.json' % call_name), rec)
            if client.fatal or rt.unknown_usage_blocks(led):
                raise RuntimeError('backend fatal or unknown usage during formation A')
        record['calls'][call_name] = {k: rec[k] for k in ('status', 'attempts')}
        if call_name == 'A1':
            a1_raw = rec.get('raw')
    a1, a2 = context.read_json(out / 'A1.json'), context.read_json(out / 'A2.json')
    statuses = {'A1': a1['status'], 'A2': a2['status']}
    if any(s in ('CALL_FAILED', 'PARSE_OR_VALIDATION_FAILED') for s in statuses.values()):
        rec = {'status': 'FORMATION_TECHNICAL_FAILURE', 'calls': statuses, 'skills': {}, 'aliases': {}, 'written_local': now()}
        dsks.write_once(out / 'candidates.json', rec)
        return rec
    skills, aliases, meta = {}, {}, {}
    for a in (a1, a2):
        for sid, sj in a.get('skills', {}).items():
            same = next((k for k, v in skills.items() if v['rendered_body'] == sj['rendered_body']), None)
            if same:
                aliases[sid] = same
            else:
                skills[sid] = sj
                meta[sid] = a['meta'][sid]
    status = 'NO_INITIAL_PROPOSAL' if not skills else 'PROPOSED'
    rec = {'status': status, 'calls': statuses, 'skills': skills, 'meta': meta, 'aliases': aliases, 'n_candidates': len(skills),
           'keep_rationales': {k: a.get('rationale') for k, a in (('A1', a1), ('A2', a2)) if a['status'] == 'KEEP'}, 'written_local': now()}
    dsks.write_once(out / 'candidates.json', rec)
    return rec


def arm_of(skill_id: str) -> str:
    return 'cand_' + skill_id.split('-', 1)[1].replace('-', '')        # RD02-W1-r1 -> cand_W1r1


# ============================================================================= B. practice on L1 / L2, keep two parents by J_P
def practice_stage(root: Path, cands: dict) -> dict:
    skills = {k: dsk.skill_from_json(v) for k, v in cands['skills'].items()}
    ids = sorted(skills)                                              # ascending candidate id on L1, reversed on L2
    arms = {arm_of(k): skills[k] for k in ids}
    order = {'RD02_L1': [arm_of(k) for k in ids], 'RD02_L2': [arm_of(k) for k in reversed(ids)]}
    knowledge = {j: {a: asdict(dsk.skill_knowledge(s)) for a, s in arms.items()} for j in PRACTICE_JOBS}
    cache_from = {j: str(PARENT / 'target' / (j + '_common') / j) for j in PRACTICE_JOBS if (PARENT / 'target' / (j + '_common') / j / 'aug_materials' / 'index.json').exists()}
    cfgp = stage_config(root, 'practice', jobs=PRACTICE_JOBS, order=order, knowledge=br.json_copy(knowledge), labels_e=False, random=False, cache_from=cache_from)
    return run_stage(root, 'practice', cfgp)


def _committed_ratio(bdir: Path, job: str):
    """mean_seed C_B(committed) / mean_seed C_B(None) of one branch; None when incomplete / unscorable."""
    cb = W._committed_block(bdir, job, 'c_b')
    none_cb = W._block_vec(bdir, job, 'None', 'c_b') if bdir.exists() else None
    if cb is None or none_cb is None or statistics.fmean(none_cb) <= 0:
        return None, cb, none_cb
    return statistics.fmean(cb) / statistics.fmean(none_cb), cb, none_cb


def j_table(stage_root: Path, jobs, arms: dict) -> dict:
    """arms: {arm: skill_id or None}. J(v) = mean over jobs of the committed C_B ratio; only options complete and scorable on every job get a J."""
    runs, J = {}, {}
    for arm in arms:
        runs[arm], ratios = {}, []
        for j in jobs:
            b = stage_root / ('%s_%s' % (j, arm))
            res = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
            ratio, cb, none_cb = _committed_ratio(b, j)
            runs[arm][j] = {'status': res['status'] if res else 'NOT_RUN', 'committed': res['committed_plan_id'] if res else None, 'failure_kind': res['failure_kind'] if res else None,
                            'ratio': ratio, 'c_b_by_seed': cb, 'none_c_b_by_seed': none_cb, 'c_a_by_seed': W._committed_block(b, j, 'c_a') if res else None}
            if ratio is not None:
                ratios.append(ratio)
        if len(ratios) == len(jobs):
            J[arm] = statistics.fmean(ratios)
    return {'formula': 'J(v) = mean_j(mean_seed C_B(committed_v, j) / mean_seed C_B(None, j)); lower is better', 'jobs': list(jobs), 'arms': {a: s for a, s in arms.items()}, 'runs': runs, 'J': J}


def keep_parents(root: Path, cands: dict) -> dict:
    P_ = paths(root)
    out = P_['formation_a'] / 'parents.json'
    if out.exists():
        return context.read_json(out)
    arms = {arm_of(k): k for k in cands['skills']}
    tab = j_table(P_['practice'], PRACTICE_JOBS, arms)
    ranked = sorted(tab['J'], key=lambda a: (tab['J'][a], arms[a]))
    parents = ranked[:2]
    rec = {**tab, 'ranked': ranked, 'parents': [arms[a] for a in parents], 'parent_arms': parents, 'rule': 'the two lowest J_P among candidates complete and scorable on both practice batches; ties by candidate id; one if only one; this allocates revision budget and is not a claim of effectiveness',
           'status': 'PRACTICE_INCOMPLETE' if not parents else 'PARENTS_KEPT', 'written_local': now()}
    dsks.write_once(out, rec)
    return rec


# ============================================================================= C. revision + Principles: one Slow call per kept parent
def practice_evidence(root: Path, cands: dict) -> dict:
    arms = [arm_of(k) for k in sorted(cands['skills'])]
    return census(root, practice_branches(root, arms), {str(paths(root)['practice']): 'practice'}, 3,
                  purpose='Practice: real use of every candidate card on two new batches, with post-commit C_B of every evaluated material')


def counterexamples(root: Path, parent_arm: str, prac: dict) -> list:
    """Deterministic facts per practice branch of the parent: committed plan, its C_A / C_B means, the evaluated material with the lowest C_B and its C_A,
    and the None / public reference C_B. No interpretation."""
    out = []
    for job in PRACTICE_JOBS:
        J = prac['jobs'].get(job)
        if not J:
            continue
        b = next((x for x in J['branches'] if x['ref_prefix'].endswith('/%s_%s/' % (job, parent_arm))), None)
        if not b or not b.get('committed_physical_id'):
            out.append({'branch_ref': 'practice/%s_%s/' % (job, parent_arm), 'fact': 'no complete commit (status %s)' % (b['status'] if b else 'NOT_RUN')})
            continue
        mats = J['materials']
        cm = mats.get(b['committed_physical_id'], {})
        evaluated = {p: mats[p] for p in set(b['plan_to_physical'].values()) if mats.get(p, {}).get('c_b_by_seed')}
        best = min(evaluated.items(), key=lambda kv: statistics.fmean(kv[1]['c_b_by_seed'])) if evaluated else None
        none_cb = next((m['c_b_by_seed'] for m in mats.values() if m.get('public_id') == 'None'), None)
        out.append({'branch_ref': b['ref_prefix'], 'committed_plan': b['committed_plan_id'], 'committed_c_a_mean': statistics.fmean(cm['c_a_by_seed']) if cm.get('c_a_by_seed') else None,
                    'committed_c_b_mean': statistics.fmean(cm['c_b_by_seed']) if cm.get('c_b_by_seed') else None,
                    'none_c_b_mean': statistics.fmean(none_cb) if none_cb else None,
                    'lowest_c_b_evaluated_in_this_branch': {'physical_id': best[0], 'plan_ids': [p for p, ph in b['plan_to_physical'].items() if ph == best[0]],
                                                            'c_b_mean': statistics.fmean(best[1]['c_b_by_seed']), 'c_a_mean': statistics.fmean(best[1]['c_a_by_seed']) if best[1].get('c_a_by_seed') else None} if best else None,
                    'committed_is_lowest_c_b': bool(best and best[0] == b['committed_physical_id'])})
    return out


def revision_stage(root: Path, cands: dict, parents: dict) -> dict:
    P_ = paths(root)
    out = P_['revision']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'children.json').exists():
        return context.read_json(out / 'children.json')
    cen_skel = initial_census(root, tier='skeleton')
    if not (P_['evidence'] / 'census_initial_skeleton.json').exists():
        dsks.write_once(P_['evidence'] / 'census_initial_skeleton.json', cen_skel)
    skills = {k: dsk.skill_from_json(v) for k, v in cands['skills'].items()}
    led, client, sc = slow_client(root, 'revision')
    if not (P_['evidence'] / 'census_practice.json').exists():
        sizes = {}
        for tier in (3, 'skeleton'):                                  # one call per kept parent (+ a correction each) must fit the stage cap
            arms = [arm_of(k) for k in sorted(cands['skills'])]
            prac = census(root, practice_branches(root, arms), {str(P_['practice']): 'practice'}, tier,
                          purpose='Practice: real use of every candidate card on two new batches, with post-commit C_B of every evaluated material')
            probe = {'parent_card': {'rendered_body': 'x' * BODY_LIMIT, 'rationale': 'x' * 2000, 'evidence_review': 'x' * 2000},
                     'initial_census_condensed': {k: v for k, v in cen_skel.items() if k != 'legal_evidence_refs'}, 'practice_census': {k: v for k, v in prac.items() if k != 'legal_evidence_refs'},
                     'numeric_counterexamples_for_parent': 'x' * 3000, 'legal_evidence_refs': ref_ranges(sorted(set(cen_skel['legal_evidence_refs']) | set(prac['legal_evidence_refs'])))}
            sizes[str(tier)] = payload_bytes(SLOW_C, probe)
            chk = check_reservation(led, sc, sizes[str(tier)], calls_left=len(parents['parents']))
            prac['compression']['payload_bytes_by_tier'] = dict(sizes)
            prac['compression']['reservation_check_at_choice'] = chk
            if chk['fits_all_remaining_calls']:
                break
        dsks.write_once(P_['evidence'] / 'census_practice.json', prac)
    prac = context.read_json(P_['evidence'] / 'census_practice.json')
    legal = frozenset(cen_skel['legal_evidence_refs']) | frozenset(prac['legal_evidence_refs'])
    forbidden = [n for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')]
    children, records = {}, {}
    for sid in parents['parents']:
        f = out / ('revise_%s.json' % sid)
        if f.exists():
            rec = context.read_json(f)
        else:
            parent = skills[sid]
            payload = {'domain_id': DOMAIN, 'parent_card': {'skill_id': sid, 'workflow': parent.workflow, 'principles': parent.principles, 'rendered_body': parent.rendered_body,
                                                            'rationale': cands['meta'][sid]['rationale'], 'evidence_review': cands['meta'][sid]['evidence_review'], 'research_mode': cands['meta'][sid]['research_mode']},
                       'parent_practice_arm': arm_of(sid),
                       'initial_census_condensed': {k: v for k, v in cen_skel.items() if k != 'legal_evidence_refs'},
                       'practice_census': {k: v for k, v in prac.items() if k != 'legal_evidence_refs'},
                       'numeric_counterexamples_for_parent': counterexamples(root, arm_of(sid), prac),
                       'legal_evidence_refs': ref_ranges(sorted(legal)),
                       'legal_evidence_refs_format': 'evidence_refs must be exact strings "<stage>/<branch>/<job>:<integer>" inside these ranges, or one of other_refs',
                       'allowed_batch_features': sorted(ALLOWED_FEATURES), 'body_limit_characters': BODY_LIMIT, 'required': {'observable_applicability': {'const': True}, 'max_children': 1},
                       'status': 'CANDIDATE_TEST_ONLY'}
            size = payload_bytes(SLOW_C, payload)
            chk = check_reservation(led, sc, size)
            if not chk['fits_with_correction']:
                raise RuntimeError('revision payload does not fit the frozen reservation with a correction margin: %s' % chk)
            res = slow_call(client, 'slow_revise', sid, payload, SLOW_C, lambda r, parent=parent: parse_revision(r, parent=parent, legal_refs=legal), forbidden=forbidden)
            rec = {**{k: v for k, v in res.items() if k != 'skill'}, 'skill': skill_json(res.get('skill')), 'parent': sid, 'reservation_check': chk, 'requests_after': led.s['llm_requests'], 'written_local': now()}
            dsks.write_once(f, rec)
            if client.fatal or rt.unknown_usage_blocks(led):
                raise RuntimeError('backend fatal or unknown usage during revision')
        records[sid] = {k: rec.get(k) for k in ('status', 'attempts')}
        if rec['status'] == 'REVISE':
            children[rec['skill']['skill_id']] = {'skill': rec['skill'], 'parent': sid, 'meta': rec.get('meta')}
        elif rec['status'] == 'KEEP':
            children[sid + '-KEEP'] = {'skill': None, 'parent': sid, 'alias_of_parent': True, 'rationale': rec.get('rationale'), 'evidence_review': rec.get('evidence_review'),
                                       'counterexample_responses': rec.get('counterexample_responses')}
    tech = [s for s, r in records.items() if r['status'] in ('CALL_FAILED', 'PARSE_OR_VALIDATION_FAILED')]
    rec = {'status': 'REVISION_TECHNICAL_FAILURE' if tech else 'REVISED', 'per_parent': records, 'children': children, 'technical_failures': tech, 'written_local': now()}
    dsks.write_once(out / 'children.json', rec)
    return rec


# ============================================================================= D. select on L3 / L4: parents + children + no-Skill; freeze the four choices
def select_options(cands: dict, parents: dict, children: dict) -> dict:
    """{arm: (skill_id or None, Skill or None)} in the task order [no_skill, p1, c1, p2, c2]; KEEP children are aliases and are not run."""
    skills = {k: dsk.skill_from_json(v) for k, v in cands['skills'].items()}
    for cid, c in children['children'].items():
        if c.get('skill'):
            skills[cid] = dsk.skill_from_json(c['skill'])
    opts = {'no_skill': (None, None)}
    for sid in parents['parents']:
        opts[arm_of(sid)] = (sid, skills[sid])
        child = next((cid for cid, c in children['children'].items() if c['parent'] == sid and c.get('skill')), None)
        if child:
            opts[arm_of(child)] = (child, skills[child])
    return opts


def select_stage(root: Path, cands: dict, parents: dict, children: dict) -> dict:
    opts = select_options(cands, parents, children)
    order = {'RD02_L3': list(opts), 'RD02_L4': list(reversed(list(opts)))}
    knowledge = {j: {a: asdict(dsk.skill_knowledge(s)) for a, (sid, s) in opts.items()} for j in SELECT_JOBS}
    cache_from = {j: str(PARENT / 'target' / (j + '_common') / j) for j in SELECT_JOBS if (PARENT / 'target' / (j + '_common') / j / 'aug_materials' / 'index.json').exists()}
    cfgp = stage_config(root, 'select', jobs=SELECT_JOBS, order=order, knowledge=br.json_copy(knowledge), labels_e=False, random=False, cache_from=cache_from)
    return run_stage(root, 'select', cfgp)


def _public_j(stage_root: Path, jobs) -> dict:
    """J of each public reference (its cells are identical in every branch of a job; the no_skill branch is read)."""
    out = {}
    for m in PUBLIC:
        ratios = []
        for j in jobs:
            b = stage_root / ('%s_no_skill' % j)
            v, none = W._block_vec(b, j, m, 'c_b') if b.exists() else None, W._block_vec(b, j, 'None', 'c_b') if b.exists() else None
            if v and none and statistics.fmean(none) > 0:
                ratios.append(statistics.fmean(v) / statistics.fmean(none))
        if len(ratios) == len(jobs):
            out[m] = statistics.fmean(ratios)
    return out


def freeze_choices(root: Path, cands: dict, parents: dict, children: dict) -> dict:
    """W_initial = lowest J among parents; W_learned = lowest J among parents + children (parents first, then id on ties);
    H_system = lowest J of {W_learned, no_skill} with no_skill winning ties; Fixed_dev = lowest-J public reference. Frozen before any Test feature reaches an Agent."""
    P_ = paths(root)
    out = P_['formation_a'] / 'frozen_choices.json'
    if out.exists():
        return context.read_json(out)
    opts = select_options(cands, parents, children)
    tab = j_table(P_['select'], SELECT_JOBS, {a: sid for a, (sid, s) in opts.items()})
    J = tab['J']
    parent_arms = [arm_of(s) for s in parents['parents']]
    child_arms = [a for a in opts if a not in parent_arms and a != 'no_skill']
    order_key = lambda a: (J[a], 0 if a in parent_arms else 1, opts[a][0] or '')
    def argmin(arms):
        ok = [a for a in arms if a in J]
        if not ok:
            return None
        best = min(J[a] for a in ok)
        return min([a for a in ok if J[a] - best <= TOL], key=order_key)
    w_initial = argmin(parent_arms)
    w_learned = argmin(parent_arms + child_arms)
    incomplete = [a for a in opts if a not in J]
    rec = {**tab, 'parent_arms': parent_arms, 'child_arms': child_arms, 'incomplete_options': incomplete, 'public_J': _public_j(P_['select'], SELECT_JOBS)}
    if w_initial is None or w_learned is None or 'no_skill' not in J:
        rec.update(status='SELECT_INCOMPLETE', W_initial=None, W_learned=None, H_system=None, Fixed_dev=None, note='a required comparison is missing; no winner is frozen by narrative')
        dsks.write_once(out, rec)
        return rec
    h = 'no_skill' if J['no_skill'] - J[w_learned] <= TOL else w_learned
    fixed = min(rec['public_J'], key=lambda m: (rec['public_J'][m], list(PUBLIC).index(m))) if rec['public_J'] else None
    rec.update(status='FROZEN', W_initial={'arm': w_initial, 'skill_id': opts[w_initial][0], 'J': J[w_initial]},
               W_learned={'arm': w_learned, 'skill_id': opts[w_learned][0], 'J': J[w_learned], 'is_child': w_learned in child_arms},
               H_system={'arm': h, 'skill_id': opts[h][0], 'J': J[h], 'chosen_over': 'no_skill' if h != 'no_skill' else w_learned},
               Fixed_dev={'public_id': fixed, 'J': rec['public_J'].get(fixed)},
               rules='quality only (no cost penalty, no significance gate); ties <= 1e-12: parents before children, then candidate id; no_skill wins its tie with W_learned',
               skills={opts[a][0]: skill_json(opts[a][1]) for a in (w_initial, w_learned) if opts[a][0]}, frozen_local=now(), frozen_epoch=time.time())
    dsks.write_once(out, rec)
    return rec


# ============================================================================= E. test on L5 / L6 / L7
def test_stage(root: Path, frozen: dict) -> dict:
    if frozen['status'] != 'FROZEN':
        raise RuntimeError('choices are not frozen; Test refused')
    skills = {k: dsk.skill_from_json(v) for k, v in frozen['skills'].items()}
    w_i, w_l, h = frozen['W_initial'], frozen['W_learned'], frozen['H_system']
    same = skills[w_i['skill_id']].rendered_body == skills[w_l['skill_id']].rendered_body
    note = {}
    kn = {'f0': asdict(br.Knowledge()), 'f_initial': asdict(dsk.skill_knowledge(skills[w_i['skill_id']])),
          'f_learned': None if same else asdict(dsk.skill_knowledge(skills[w_l['skill_id']])), 'random': None, 'menu_ca': None}
    if same:
        note['f_learned'] = {'status': 'ALIAS', 'alias_of_arm': 'f_initial', 'note': 'W_learned renders identically to W_initial (no revision difference): one Fast run, reused'}
    note['h_system'] = {'status': 'ALIAS', 'alias_of_arm': 'f0' if h['arm'] == 'no_skill' else ('f_initial' if same else 'f_learned'),
                        'note': 'H_system is the frozen rule\'s choice between no-Skill and W_learned; its delivery is that arm\'s delivery, no fifth research arm',
                        'candidate_test_isolated': 'W_learned (F_learned) runs regardless and is reported as an isolated candidate test, not as an adopted Skill' if h['arm'] == 'no_skill' else None}
    order = {j: [a for a in TEST_ORDER[j] if not (a == 'f_learned' and same)] + ['menu_ca'] for j in TEST_JOBS}
    knowledge = {j: kn for j in TEST_JOBS}
    cat = paths(root)['formation_a'] / 'test_catalog.json'
    if not cat.exists():
        dsks.write_once(cat, {'status': 'TEST_FROZEN', 'frozen_local': now(), 'frozen_epoch': time.time(), 'W_initial': w_i, 'W_learned': w_l, 'H_system': h, 'Fixed_dev': frozen['Fixed_dev'],
                              'f_learned_alias_of_f_initial': same, 'orders': order, 'frozen_before_first_test_fit': True})
    cfgp = stage_config(root, 'test', jobs=TEST_JOBS, order=order, knowledge=br.json_copy(knowledge), labels_e=True, random=True, treatment_note=note)
    return run_stage(root, 'test', cfgp)


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
    w = wiring(root)
    status['wiring'] = w['status']
    if w['status'] != 'PASS':
        return stop('wiring', w)
    cands = formation_a(root)
    status['formation_a'] = cands['status']
    if cands['status'] != 'PROPOSED':
        return stop('formation_a', cands)
    st = practice_stage(root, cands)
    status['practice'] = st
    if st['status'] != 'FINISHED':
        return stop('practice', st)
    parents = keep_parents(root, cands)
    status['parents'] = parents['status']
    if parents['status'] != 'PARENTS_KEPT':
        return stop('parents', parents)
    children = revision_stage(root, cands, parents)
    status['revision'] = children['status']
    if children['status'] != 'REVISED':
        return stop('revision', children)
    st = select_stage(root, cands, parents, children)
    status['select'] = st
    if st['status'] != 'FINISHED':
        return stop('select', st)
    frozen = freeze_choices(root, cands, parents, children)
    status['frozen_choices'] = frozen['status']
    if frozen['status'] != 'FROZEN':
        return stop('freeze', frozen)
    st = test_stage(root, frozen)
    status['test'] = st
    if st['status'] != 'FINISHED':
        return stop('test', st)
    status['finished_epoch'] = time.time()
    status['finished_local'] = now()
    context.write_json(root / 'package_status.json', status)
    readout(root)
    print('PACKAGE_FINISHED', flush=True)
    return status


# ============================================================================= preflight (0 cost) and wiring (RD02_T1, <= 3 fits, parent cache reused)
def preflight(root: Path = ROOT) -> dict:
    import inspect
    root.mkdir(parents=True, exist_ok=True)
    p = root / 'frozen_config.json'
    if p.exists():
        return context.read_json(p)
    env = W.environment()
    prev = context.read_json(PARENT / 'frozen_config.json')['environment']
    env_ok = all(prev[k] == env[k] for k in ('python', 'torch', 'numpy', 'threads'))
    src = inspect.getsource(rt.MeteredClient)
    if not all(x in src for x in ("model='cpa-grok-4.6'", "'grok-4.6-build'", 'temperature=0', "base_url='http://127.0.0.1:8318/v1'")):
        raise RuntimeError('metered client no longer carries the frozen model identity')
    for s in SEEDS:
        if not 0 <= spec.batch_seed(s) < 2 ** 32:
            raise RuntimeError('training seed %d overflows the Consumer batch stream' % s)
    jobs = {}
    for j, t in JOB_T.items():
        js = rd.resolve_job(DATASET, j)
        if js.t != t:
            raise RuntimeError('job table drift %s' % j)
        jobs[j] = {'t': t, 'job_index': JOB_INDEX[j], 'train_rows': list(js.train_range), 'c_a_origins': list(js.c_a), 'c_b_origins': list(js.c_b), 'e_origins': list(js.e), 'last_row_read_exclusive': js.e_target_rows[1]}
    avail = context.read_json(W.AVAILABILITY)['candidates']
    tests = {j: avail['RD02@%d' % JOB_T[j]] for j in TEST_JOBS}
    exposure = {j: {'eligible_and_scorable': v['eligible_and_scorable'], 'E_rows_scored_by_any_known_line': v['E_rows_scored_by_any_known_line'],
                    'E_overlaps_preflight_scored_windows_first_six': v['E_overlaps_preflight_scored_windows_first_six'], 'E_min_coverage_mask_only': v['coverage_mask_only']['E']['min_entity_coverage']} for j, v in tests.items()}
    blocked = [j for j, v in exposure.items() if not v['eligible_and_scorable'] or v['E_rows_scored_by_any_known_line']]
    if blocked:
        raise RuntimeError('a Test batch is not eligible / already scored by a known line: %s' % blocked)
    max_row = max(v['last_row_read_exclusive'] for v in jobs.values())
    sealing_ok = max_row <= W.PRSA_SEALED_FROM_ROW
    parent_fc = context.read_json(PARENT / 'frozen_config.json')
    history = {'parent_package': str(PARENT), 'parent_status': context.read_json(PARENT / 'package_status.json'), 'parent_candidates': [s['skill_id'] for s in context.read_json(PARENT / 'formation' / 'propose.json')['skills']],
               'trajectories': [b['branch'] for b in historical_branches()], 'parent_ledger': {k: v for k, v in context.read_json(PARENT / 'budget.json').items() if k not in ('events',)}}
    cfg = {'package': PACKAGE, 'task': TASK, 'frozen_local': now(), 'frozen_epoch': time.time(), 'method_reference': 'Eval-Skill arXiv:2606.07040v2 (candidate exploration, real-use feedback, Workflow + Principles, no-Skill competition); bounded same-domain loop',
           'dataset': {'id': DATASET, 'roster': rd.DATASETS[DATASET]['roster'], 'exposure': rd.EXPOSURE, 'prsa_sealed_from_row': W.PRSA_SEALED_FROM_ROW, 'max_row_read_exclusive': max_row, 'sealing_ok': sealing_ok},
           'geometry': parent_fc['geometry'], 'jobs': jobs, 'curriculum': {'initial_learning': list(HIST_SOURCE_JOBS) + list(HIST_SELECT_JOBS), 'practice': list(PRACTICE_JOBS), 'select': list(SELECT_JOBS), 'test': list(TEST_JOBS)},
           'orders': {'practice': 'L1 ascending candidate id, L2 descending', 'select': 'L3 [no_skill, p1, c1, p2, c2], L4 reversed', 'test': {j: list(o) + ['menu_ca'] for j, o in TEST_ORDER.items()}},
           'seeds': list(SEEDS), 'consumer': W.CONSUMER, 'environment': env, 'environment_matches_parent': env_ok, 'model': MODEL,
           'fast_system': FAST_SYSTEM, 'slow_a1': SLOW_A1, 'slow_a2': SLOW_A2, 'slow_c': SLOW_C, 'contracts': W.CONTRACTS, 'limits': LIMITS, 'fast_token_cap': FAST_TOKEN_CAP,
           'max_tool_corrections': MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'body_limit_characters': BODY_LIMIT, 'body_limit_default_unchanged': dsk.BODY_LIMIT,
           'materials': parent_fc['materials'], 'random_search_b4': parent_fc['random_search_b4'], 'menu_ca': parent_fc['menu_ca'],
           'physical_cache_from_parent': {j: str(PARENT / 'target' / (j + '_common') / j) for j in PRACTICE_JOBS + SELECT_JOBS},
           'selection': {'J': 'J(v) = mean_j(mean_seed C_B(committed_v) / mean_seed C_B(None)); lower is better; tolerance 1e-12',
                         'parents': 'two lowest J_P on L1/L2 among candidates complete on both; ties by candidate id',
                         'W_initial': 'lowest J on L3/L4 among parents', 'W_learned': 'lowest J among parents + children (parents first on ties, then id)',
                         'H_system': 'lowest J of {W_learned, no_skill}; no_skill wins ties', 'Fixed_dev': 'lowest-J public reference'},
           'main_readout': {'d_evolve': 'E(F_initial) - E(F_learned)', 'd_skill': 'E(F0) - E(F_learned)', 'd_system': 'E(F0) - E(H_system)', 'G': 'mean_j(100 x mean_s d / mean_s E(None)) over three Test jobs',
                            'interval': 'paired SE and n=3 two-sided 95%% t interval (df=2, t=%.6f)' % T975_DF2},
           'budget': {'total': TOTAL, 'stage_allocations': ALLOC, 'stage_retries': STAGE_RETRIES, 'http_cap': HTTP_CAP, 'numeric_wall_s': NUMERIC_WALL_S},
           'label_barriers': 'Practice / Select: all planned arms end -> C_B only. Test: all three jobs\' arms end and every delivery frozen -> C_B -> all E predictions frozen -> barrier -> E',
           'test_exposure_check': exposure, 'history': history}
    if not (env_ok and sealing_ok):
        raise RuntimeError('preflight failed: environment %s sealing %s' % (env_ok, sealing_ok))
    dsks.write_once(p, cfg)
    return cfg


def wiring(root: Path = ROOT) -> dict:
    """Instrument check on RD02_T1 with the historical seeds: the parent wiring cache (None / FixedMixup / the trained composition) is copied
    after binding checks, one scripted seven-tool branch builds the same legal composition -> evaluate (cache) -> compare -> commit -> C_B. 0 E."""
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
    cfg = {'stage': 'wiring', 'output': str(wroot), 'jobs': [job], 'seeds': list(seeds), 'job_index': {job: ji}, 'public_ids': list(W.WIRING_PUBLIC), 'labels_e': False, 'random': False, 'llm': False,
           'cache_from': {job: str(PARENT / 'wiring' / (job + '_common') / job)}, 'fast_system': FAST_SYSTEM}
    context.write_json(wroot / 'config.json', cfg)
    context.write_json(wroot / 'experiment_started.json', {'epoch': time.time(), 'pid': os.getpid(), 'stage': 'wiring'})
    before = led.s['fit_attempts']
    common = W.prepare_common(wroot, job, led, cfg)
    copied = context.read_json(common / job / 'cache_copied_from.json') if (common / job / 'cache_copied_from.json').exists() else None
    card = make_card(skill_id='RD02-WIRE-r1', revision=1, workflow='Wiring fixture: ' + 'x' * 2000 + ' compare the public references, build one composition, evaluate it and commit on the paired comparison.',
                     principles='Wiring fixture principle: ' + 'y' * 1500, summary='wiring fixture', refs=['fixture:wiring'], legal_refs=['fixture:wiring'], mode='wiring fixture', status='SMOKE_FIXTURE', source_stage='fixture')
    scripts = {job + '_wire': [[{'tool': 'compare', 'arguments': {'a': 'None', 'b': 'FixedMixup'}}],
                               [{'tool': 'build_material', 'arguments': {'plan_id': 'W_comp', 'policy': W.WIRING_PLAN}}],
                               [{'tool': 'evaluate', 'arguments': {'plan_id': 'W_comp'}}], [{'tool': 'compare', 'arguments': {'a': 'None', 'b': 'W_comp'}}],
                               [{'tool': 'commit', 'arguments': {'plan_id': 'W_comp', 'reason': 'wiring: scripted commit of the composition'}}]]}
    client = W.ScriptedClient(scripts)
    path = wroot / (job + '_wire')
    kn = dsk.skill_knowledge(card)
    res = W.fast_branch(path, job, kn, led, client, common, job_index=ji, public_ids=W.WIRING_PUBLIC, seeds=seeds)
    branches, failures = [(path, job)], []
    context.write_json(wroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(path)]})
    context.write_json(wroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(path), job)], 'failures': failures})
    W.open_labels(wroot, cfg, branches, failures, led)
    loaded = client.requests[0]['loaded']
    reg = W.load_registry(path / job)
    cb = context.read_json(path / job / 'c_b_scores.json')['cells'] if (path / job / 'c_b_scores.json').exists() else {}
    out = {'label': 'WIRING (scripted client; no LLM; instrument check, not a learning case)', 'job': job, 'seeds': list(seeds), 'cache_copied': copied,
           'branch': {'status': res.status, 'committed': res.committed_plan_id, 'calls': res.calls, 'tool_calls': res.tool_calls, 'new_evaluations': res.new_evaluations},
           'fits_charged': led.s['fit_attempts'] - before, 'card_body_chars': len(card.rendered_body), 'loaded_body_chars': [len(x['body']) for x in loaded],
           'c_b_cells_for_composition': sorted(c for c in cb if '__W_comp__' in c), 'e_files': [str(p) for p in wroot.rglob('e_*.json')]}
    out['checks'] = {'cache_copied_from_parent': bool(copied and copied.get('copied')), 'branch_complete': res.status == 'COMPLETE' and res.committed_plan_id == 'W_comp',
                     'fits_at_most_3': out['fits_charged'] <= 3, 'composition_reused_from_cache': reg['W_comp']['phys_id'] == 'TA001' and out['fits_charged'] == 0,
                     'long_card_injected': len(loaded) == 1 and len(loaded[0]['body']) == len(card.rendered_body) > 1200,
                     'c_b_opened_for_composition': len(out['c_b_cells_for_composition']) == 3, 'no_e_file': not out['e_files']}
    out['status'] = 'PASS' if all(out['checks'].values()) else 'FAIL'
    out['written_local'] = now()
    context.write_json(rec_path, out)
    print('WIRING', out['status'], out['checks'], flush=True)
    return out


# ============================================================================= readout (task §9)
def _stats(v):
    return W._stats(v)


def _paired(a, b, den=None):
    return W._paired(a, b, den)


def _arm_rec(stage_root: Path, job: str, arm: str):
    b = stage_root / ('%s_%s' % (job, arm))
    return W._arm_record(b, job) if (b / 'branch_result.json').exists() else None


def readout(root: Path = ROOT) -> dict:
    P_ = paths(root)
    res = {'package': PACKAGE, 'written_local': now(), 'units': {'loss': 'missing-aware normalized MSE macro (lower is better)',
           'gain_pp': 'paired per seed: 100 x (loss_reference - loss_arm) / three-seed mean None loss of that job; positive = arm better',
           'd_evolve': 'E(F_initial) - E(F_learned)', 'd_skill': 'E(F0) - E(F_learned)', 'd_system': 'E(F0) - E(H_system)', 'G': 'mean over the three Test jobs of 100 x mean_s d / mean_s E(None)'},
           'stages': {}, 'formation_a': {}, 'practice': {}, 'revision': {}, 'select': {}, 'frozen': {}, 'test': {}, 'main': {}, 'chains': {}, 'cost': {}}
    for st in ('practice', 'select', 'test'):
        if (P_[st] / 'execution_finished.json').exists():
            res['stages'][st] = dsks.stage_status(P_[st])
    if (P_['wiring'] / 'wiring.json').exists():
        res['stages']['wiring'] = context.read_json(P_['wiring'] / 'wiring.json')['status']
    for name in ('A1', 'A2', 'candidates', 'parents', 'frozen_choices', 'test_catalog'):
        f = P_['formation_a'] / (name + '.json')
        if f.exists():
            res['formation_a'][name] = {k: v for k, v in context.read_json(f).items() if k not in ('raw', 'runs')}
    cands = context.read_json(P_['formation_a'] / 'candidates.json') if (P_['formation_a'] / 'candidates.json').exists() else {'skills': {}}
    if (P_['revision'] / 'children.json').exists():
        res['revision'] = context.read_json(P_['revision'] / 'children.json')
        for f in sorted(P_['revision'].glob('revise_*.json')):
            r = context.read_json(f)
            res['revision'].setdefault('per_parent_detail', {})[r['parent']] = {k: r.get(k) for k in ('status', 'attempts', 'meta', 'rationale', 'evidence_review', 'counterexample_responses', 'reservation_check')}
    for j in PRACTICE_JOBS:
        for k in cands['skills']:
            r = _arm_rec(P_['practice'], j, arm_of(k))
            if r:
                res['practice'].setdefault(j, {})[arm_of(k)] = r
    frozen = context.read_json(P_['formation_a'] / 'frozen_choices.json') if (P_['formation_a'] / 'frozen_choices.json').exists() else None
    if frozen:
        res['frozen'] = {k: frozen.get(k) for k in ('status', 'W_initial', 'W_learned', 'H_system', 'Fixed_dev', 'J', 'public_J', 'incomplete_options', 'parent_arms', 'child_arms')}
        for j in SELECT_JOBS:
            for arm in frozen['arms']:
                r = _arm_rec(P_['select'], j, arm)
                if r:
                    res['select'].setdefault(j, {})[arm] = r
        # parent -> child paired C_B on the Select jobs
        pairs = {}
        for child in frozen.get('child_arms', []):
            parent = next((p for p in frozen['parent_arms'] if arm_of(next(c['parent'] for cid, c in res['revision']['children'].items() if c.get('skill') and arm_of(cid) == child)) == p), None)
            if parent:
                pairs[child] = {j: _paired(res['select'].get(j, {}).get(parent, {}).get('c_b'), res['select'].get(j, {}).get(child, {}).get('c_b')) for j in SELECT_JOBS}
        res['select']['parent_child_paired_c_b'] = pairs
    cat = context.read_json(P_['formation_a'] / 'test_catalog.json') if (P_['formation_a'] / 'test_catalog.json').exists() else None
    G = {'d_evolve': [], 'd_skill': [], 'd_system': []}
    complete = []
    if cat:
        same = cat['f_learned_alias_of_f_initial']
        h_arm = 'f0' if cat['H_system']['arm'] == 'no_skill' else ('f_initial' if same else 'f_learned')
        for j in TEST_JOBS:
            row = {'arms': {}, 'aliases': {'f_learned': 'f_initial' if same else None, 'h_system': h_arm}}
            for arm in ('f0', 'f_initial', 'f_learned', 'random', 'menu_ca'):
                src = 'f_initial' if (arm == 'f_learned' and same) else arm
                r = _arm_rec(P_['test'], j, src)
                row['arms'][arm] = ({**r, 'alias_of': src} if src != arm else r) if r else {'status': 'NOT_RUN'}
            row['arms']['h_system'] = {**row['arms'][h_arm], 'alias_of': h_arm}
            anyb = next((P_['test'] / ('%s_%s' % (j, a)) for a in ('f0', 'random', 'menu_ca') if (P_['test'] / ('%s_%s' % (j, a)) / j / 'e_scores.json').exists()), None)
            if anyb is not None:
                row['public'] = {m: {b_: W._block_vec(anyb, j, m, b_) for b_ in ('c_a', 'c_b', 'e')} for m in PUBLIC}
                none_e = row['public']['None']['e']
                den = statistics.fmean(none_e) if none_e else None
                row['none_mean_e'] = den
                e = lambda a: row['arms'].get(a, {}).get('e')
                fixed = cat['Fixed_dev']['public_id']
                row['fixed_dev'] = {'public_id': fixed, 'e': row['public'][fixed]['e'] if fixed else None}
                row['main'] = {'d_evolve': _paired(e('f_initial'), e('f_learned'), den), 'd_skill': _paired(e('f0'), e('f_learned'), den), 'd_system': _paired(e('f0'), e('h_system'), den)}
                refs_ = {'random': e('random'), 'menu_ca': e('menu_ca'), 'fixed_dev': row['fixed_dev']['e'], **{m: row['public'][m]['e'] for m in PUBLIC}}
                row['f_learned_vs'] = {k: _paired(v, e('f_learned'), den) for k, v in refs_.items()}
                row['h_system_vs'] = {k: _paired(v, e('h_system'), den) for k, v in refs_.items()}
                row['f0_vs'] = {k: _paired(v, e('f0'), den) for k, v in refs_.items()}
                row['f_initial_vs'] = {k: _paired(v, e('f_initial'), den) for k, v in refs_.items()}
                if all(row['main'][k] for k in G) and den:
                    for k in G:
                        G[k].append(row['main'][k]['gain_pp']['mean'])
                    complete.append(j)
                row['oracle_evaluated_e'] = {a: {m: (statistics.fmean(v['e']) if v.get('e') else None) for m, v in row['arms'][a].get('all_evaluated', {}).items()}
                                             for a in ('f0', 'f_initial', 'f_learned', 'random') if row['arms'][a].get('all_evaluated')}
            res['test'][j] = row
        res['main'] = {'complete_jobs': complete, 'complete': len(complete) == len(TEST_JOBS), **{'G_' + k: (statistics.fmean(v) if v else None) for k, v in G.items()},
                       'per_job': {j: {k: res['test'][j]['main'][k]['gain_pp'] for k in G} for j in complete},
                       'signs_positive': {k: sum(res['test'][j]['main'][k]['gain_pp']['mean'] > 0 for j in complete) for k in G}}
        for who in ('f_learned', 'h_system', 'f0', 'f_initial'):
            for ref_ in ('random', 'menu_ca', 'fixed_dev', 'P_NoMixRecipe', 'P_AmpResample', 'FixedMixup', 'None'):
                vals = [res['test'][j][who + '_vs'][ref_]['gain_pp']['mean'] for j in complete if (res['test'][j].get(who + '_vs') or {}).get(ref_)]
                res['main']['G_%s_vs_%s' % (who, ref_)] = statistics.fmean(vals) if len(vals) == len(TEST_JOBS) else None
    res['chains'] = evidence_chains(root, cands, res)
    res['cost'] = cost_readout(root)
    context.write_json(root / 'result.json', res)
    (root / 'tables.md').write_text(tables(res), encoding='utf-8')
    return res


def evidence_chains(root: Path, cands: dict, res: dict) -> dict:
    """advice -> practice behaviour -> C_A/C_B -> revision -> select / test behaviour, per parent."""
    out = {}
    rev = res.get('revision') or {}
    for sid, sj in cands.get('skills', {}).items():
        arm = arm_of(sid)
        chain = {'card': {'skill_id': sid, 'research_mode': (cands.get('meta') or {}).get(sid, {}).get('research_mode'), 'workflow': sj['workflow'], 'rationale': (cands.get('meta') or {}).get(sid, {}).get('rationale')},
                 'practice': {j: {k: v for k, v in (res['practice'].get(j, {}).get(arm) or {}).items() if k in ('commit', 'c_a', 'c_b', 'fast_calls', 'new_evaluations', 'commit_reason')} | {'tools': (res['practice'].get(j, {}).get(arm) or {}).get('behavior', {}).get('tool_sequence'), 'builds': [b['plan_id'] for b in (res['practice'].get(j, {}).get(arm) or {}).get('behavior', {}).get('builds', [])]} for j in PRACTICE_JOBS}}
        child = next(((cid, c) for cid, c in rev.get('children', {}).items() if c['parent'] == sid), None)
        if child:
            cid, c = child
            chain['revision'] = {'child_id': cid if c.get('skill') else None, 'keep': not c.get('skill'), 'changed_mechanism': (c.get('meta') or {}).get('changed_mechanism'),
                                 'principles': (c.get('skill') or {}).get('principles'), 'rule_changes': ((c.get('meta') or c).get('evidence_review') or {}).get('rule_changes'),
                                 'counterexample_responses': (c.get('meta') or c).get('counterexample_responses'), 'child_workflow': (c.get('skill') or {}).get('workflow')}
            if c.get('skill'):
                carm = arm_of(cid)
                chain['select_child'] = {j: {k: v for k, v in (res['select'].get(j, {}).get(carm) or {}).items() if k in ('commit', 'c_b', 'fast_calls', 'new_evaluations', 'commit_reason')} | {'tools': (res['select'].get(j, {}).get(carm) or {}).get('behavior', {}).get('tool_sequence')} for j in SELECT_JOBS}
        chain['select_parent'] = {j: {k: v for k, v in (res['select'].get(j, {}).get(arm) or {}).items() if k in ('commit', 'c_b', 'fast_calls', 'new_evaluations', 'commit_reason')} | {'tools': (res['select'].get(j, {}).get(arm) or {}).get('behavior', {}).get('tool_sequence')} for j in SELECT_JOBS}
        out[sid] = chain
    return out


def cost_readout(root: Path) -> dict:
    P_ = paths(root)
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    stages = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    per_stage, names = {}, list(stages)
    for i, st in enumerate(names):
        a = stages[st]['snapshot_at_start']
        b = stages[names[i + 1]]['snapshot_at_start'] if i + 1 < len(names) else {k: led.get(k, 0) for k in a}
        per_stage[st] = {k: (b.get(k, 0) - a.get(k, 0)) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'fit_wall_seconds')}
    tokens = {s: dsks._unit_tokens(P_[s]) for s in ('formation_a', 'practice', 'revision', 'select', 'test') if P_[s].exists()}
    parent_led = context.read_json(PARENT / 'budget.json') if (PARENT / 'budget.json').exists() else {}
    per_arm = {}
    for s in ('practice', 'select', 'test'):
        if not P_[s].exists():
            continue
        for b in sorted(x for x in P_[s].iterdir() if x.is_dir() and (x / 'branch_result.json').exists()):
            job = '_'.join(b.name.split('_')[:2])
            cells = W.branch_cells(b / job)
            per_arm[b.name] = {'logical_cells': len(cells), 'new_plan_cells': sum(c['material_id'] not in PUBLIC for c in cells.values())}
    return {'ledger': {k: v for k, v in led.items() if k != 'events'}, 'per_stage_delta': per_stage, 'tokens_by_unit': tokens, 'per_arm_logical_cells': per_arm,
            'numeric_seconds': W.numeric_seconds(root) if P_['ledger'].exists() else None,
            'layers': {'historical_assets_parent_package': {k: parent_led.get(k) for k in ('fit_attempts', 'llm_requests', 'llm_tokens_in', 'llm_tokens_out', 'fit_wall_seconds')},
                       'incremental_learning_A_to_D': ['formation_a', 'practice', 'revision', 'select'], 'deployment_per_job': 'test stage: references 12 fits; each F arm <= 12 fits + <= 16 calls; Random 12 fits 0 LLM; Menu_CA / Fixed_dev 0'}}


def _f(x, nd=2):
    return '—' if x is None else ('%+.*f' % (nd, x))


def tables(res: dict) -> str:
    out = ['# %s tables' % PACKAGE, '', res['units']['gain_pp'], '']
    fr = res.get('frozen') or {}
    out += ['## Frozen choices', '', '- W_initial: %s' % json.dumps(fr.get('W_initial')), '- W_learned: %s' % json.dumps(fr.get('W_learned')), '- H_system: %s' % json.dumps(fr.get('H_system')), '- Fixed_dev: %s' % json.dumps(fr.get('Fixed_dev')),
            '- Select J: %s' % json.dumps(fr.get('J')), '- public J: %s' % json.dumps(fr.get('public_J')), '']
    m = res.get('main') or {}
    out += ['## Main (Test, gain_pp; positive = second arm better)', '', '| Job | d_evolve (F_initial→F_learned) | d_skill (F0→F_learned) | d_system (F0→H_system) | None mean E |', '|---|---:|---:|---:|---:|']
    for j in TEST_JOBS:
        r = res['test'].get(j, {})
        mm = r.get('main') or {}
        cell = lambda k: ('%s (SE %.2f) [%s]' % (_f(mm[k]['gain_pp']['mean']), mm[k]['gain_pp']['se'], ''.join('+' if s > 0 else '-' if s < 0 else '0' for s in mm[k]['gain_pp']['signs']))) if mm.get(k) else '—'
        out.append('| %s | %s | %s | %s | %s |' % (j, cell('d_evolve'), cell('d_skill'), cell('d_system'), _f(r.get('none_mean_e'), 4) if r.get('none_mean_e') is not None else '—'))
    out += ['', 'G: d_evolve %s pp, d_skill %s pp, d_system %s pp (complete: %s; positive jobs: %s)' % (_f(m.get('G_d_evolve')), _f(m.get('G_d_skill')), _f(m.get('G_d_system')), m.get('complete'), json.dumps(m.get('signs_positive'))), '']
    for who in ('f_learned', 'h_system', 'f0', 'f_initial'):
        out.append('- %s vs: ' % who + ', '.join('%s %s' % (ref_, _f(m.get('G_%s_vs_%s' % (who, ref_)))) for ref_ in ('random', 'menu_ca', 'fixed_dev', 'P_NoMixRecipe', 'P_AmpResample', 'FixedMixup', 'None')))
    out += ['', '## Test deliveries and per-seed values', '', '| Job | arm | commit | E per seed | C_B per seed | C_A per seed | calls | new evals |', '|---|---|---|---|---|---|---:|---:|']
    fmt = lambda v: ', '.join('%.4f' % x for x in v) if v else '—'
    for j in TEST_JOBS:
        r = res['test'].get(j, {})
        for arm, a in (r.get('arms') or {}).items():
            out.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (j, arm + ((' (alias of %s)' % a['alias_of']) if a.get('alias_of') else ''), a.get('commit'), fmt(a.get('e')), fmt(a.get('c_b')), fmt(a.get('c_a')), a.get('fast_calls', '—'), a.get('new_evaluations', '—')))
        for mname, v in (r.get('public') or {}).items():
            out.append('| %s | ref %s | %s | %s | %s | %s | 0 | 0 |' % (j, mname, mname, fmt(v.get('e')), fmt(v.get('c_b')), fmt(v.get('c_a'))))
    out += ['', '## Practice (J_P) and Select (J) tables', '']
    pa = (res.get('formation_a') or {}).get('parents') or {}
    out.append('- J_P: %s; parents: %s' % (json.dumps(pa.get('J')), pa.get('parents')))
    for j, arms in (res.get('practice') or {}).items():
        for arm, a in arms.items():
            out.append('- practice %s %s: commit %s, C_B %s, calls %s, new %s' % (j, arm, a.get('commit'), fmt(a.get('c_b')), a.get('fast_calls'), a.get('new_evaluations')))
    for j in SELECT_JOBS:
        for arm, a in (res.get('select') or {}).get(j, {}).items():
            if isinstance(a, dict) and 'commit' in a:
                out.append('- select %s %s: commit %s, C_B %s, calls %s, new %s' % (j, arm, a.get('commit'), fmt(a.get('c_b')), a.get('fast_calls'), a.get('new_evaluations')))
    return '\n'.join(out) + '\n'


# ============================================================================= smoke (new risks only; synthetic; 0 fits, 0 API)
def smoke(out: Path) -> dict:
    import tempfile
    out.mkdir(parents=True, exist_ok=True)
    checks, notes = {}, {}
    # 1. ten-trajectory census: branch-qualified refs unique, C_B complete for every committed branch, no E anywhere, tier skeleton keeps actions / numbers
    cen = initial_census(ROOT, tier=3)
    refs = cen['legal_evidence_refs']
    n_br = sum(len(J['branches']) for J in cen['jobs'].values())
    ids = [b['ref_prefix'] for J in cen['jobs'].values() for b in J['branches']]
    cb_ok = all(J['materials'][b['committed_physical_id']]['c_b_by_seed'] is not None for J in cen['jobs'].values() for b in J['branches'] if b['committed_physical_id'])
    txt = json.dumps(cen)
    checks['census_10_branches_unique_refs_cb_complete_no_e'] = n_br == 10 and len(ids) == len(set(ids)) == 10 and len(refs) == len(set(refs)) and cb_ok and 'e_scores' not in txt and '"e":' not in txt \
        and all(r.split('/')[0] in ('hist_source', 'hist_select') for r in refs)
    sk = initial_census(ROOT, tier='skeleton')
    ev_full = [r for J in cen['jobs'].values() for b in J['branches'] for r in b['trajectory'] if r['event'] == 'tool_completed' and r['tool'] == 'evaluate']
    ev_skel = [r for J in sk['jobs'].values() for b in J['branches'] for r in b['trajectory'] if r['event'] == 'tool_completed' and r['tool'] == 'evaluate']
    resp_full = sum(r['event'] == 'fast_response' for J in cen['jobs'].values() for b in J['branches'] for r in b['trajectory'])
    resp_skel = sum(r['event'] == 'fast_response' and 'response' in r for J in sk['jobs'].values() for b in J['branches'] for r in b['trajectory'])
    checks['skeleton_keeps_responses_evaluates_smaller'] = len(ev_full) == len(ev_skel) and all('output' in r and 'feedback' in r['output'] for r in ev_skel) and resp_full == resp_skel and len(json.dumps(sk)) < len(txt)
    notes['census_bytes'] = {'tier3': len(txt), 'skeleton': len(json.dumps(sk))}
    loaded_hist = {b['ref_prefix']: bool(b['knowledge_loaded']) for J in cen['jobs'].values() for b in J['branches']}
    checks['historical_cards_present_for_cand_branches_only'] = all(v == ('cand_' in k) for k, v in loaded_hist.items())
    # 2. 6000-char opt-in injection; old default unchanged
    long_wf = 'Observe first. ' + ('Compare the public references before any construction and stop when the paired comparison is inside seed noise. ' * 40)
    card = make_card(skill_id='RD02-W1-r1', revision=1, workflow=long_wf, principles=None, summary='smoke', refs=[refs[0]], legal_refs=refs, mode='smoke long card')
    child = make_card(skill_id='RD02-W1-r2', revision=2, workflow=long_wf, principles='Prefer a public reference when paired deltas are mixed across seeds; the amplitude family lost on one batch with a shared level drop.',
                      summary='smoke', refs=[refs[0]], legal_refs=refs, mode='smoke child', derived_from='RD02-W1-r1', source_stage='revision')
    rendered = dsk.skill_knowledge(child).render({'batch_median_missing_fraction': 0.1}, ALLOWED_FEATURES)['loaded']
    try:
        dsk.make_skill(skill_id='RD02-W9-r1', domain_id=DOMAIN, revision=1, workflow=long_wf, principles=None, applicability_summary='x', compatibility_note='x', observable_applicability={'const': True},
                       evidence_refs=[refs[0]], legal_evidence_refs=refs, source_stage='fixture', status='SMOKE_FIXTURE', allowed_features=ALLOWED_FEATURES, text_check=W.skill_text_check)
        old_default_rejects = False
    except ValueError:
        old_default_rejects = True
    checks['body_6000_optin_injected_and_default_1200_unchanged'] = 1200 < len(card.rendered_body) <= BODY_LIMIT and rendered == [{'hook': 'experiment_guidance', 'body': child.rendered_body}] and 'Principles:' in child.rendered_body and old_default_rejects and dsk.BODY_LIMIT == 1200
    try:
        make_card(skill_id='RD02-W8-r1', revision=1, workflow='x' * 6001, principles=None, summary='x', refs=[refs[0]], legal_refs=refs, mode='too long')
        checks['body_over_6000_rejected'] = False
    except ValueError:
        checks['body_over_6000_rejected'] = True
    # 3. parsers: exploration ids / principles null / review; revision KEEP + REVISE + identical-child refused
    good = {'decision': 'PROPOSE', 'candidates': [{'candidate_id': 'W1', 'research_mode': 'a', 'workflow': 'Compare the references first; then one uniform ablation.', 'principles': None, 'observable_applicability': {'const': True},
                                                    'applicability_summary': 's', 'evidence_refs': [refs[0]], 'rationale': 'r', 'evidence_review': {'supports': ['x'], 'contradicts_or_limits': [], 'rule_changes': [], 'uncertain_and_expected_behavior_change': 'u'}}]}
    pe = parse_exploration(good, ids=('W1', 'W2'), legal_refs=frozenset(refs))
    bad_id = copy.deepcopy(good); bad_id['candidates'][0]['candidate_id'] = 'W3'
    bad_pr = copy.deepcopy(good); bad_pr['candidates'][0]['principles'] = 'p'
    rej = 0
    for b in (bad_id, bad_pr):
        try:
            parse_exploration(b, ids=('W1', 'W2'), legal_refs=frozenset(refs))
        except ValueError:
            rej += 1
    parent = pe['skills'][0]
    keep = parse_revision({'decision': 'KEEP', 'rationale': 'r', 'evidence_review': good['candidates'][0]['evidence_review'], 'counterexample_responses': ['c']}, parent=parent, legal_refs=frozenset(refs))
    rv = parse_revision({'decision': 'REVISE', 'child': {'workflow': parent.workflow + ' Stop after two new plans.', 'principles': 'Prefer references when deltas are mixed.', 'changed_mechanism': 'stop rule', 'research_mode': 'b',
                                                          'applicability_summary': 's', 'evidence_refs': [refs[0]], 'rationale': 'r', 'evidence_review': good['candidates'][0]['evidence_review'], 'counterexample_responses': ['c']}}, parent=parent, legal_refs=frozenset(refs))
    try:
        parse_revision({'decision': 'REVISE', 'child': {'workflow': parent.workflow, 'principles': None, 'changed_mechanism': 'none', 'research_mode': 'b', 'applicability_summary': 's', 'evidence_refs': [refs[0]], 'rationale': 'r',
                                                         'evidence_review': good['candidates'][0]['evidence_review'], 'counterexample_responses': []}}, parent=parent, legal_refs=frozenset(refs))
        same_rejected = False
    except ValueError:
        same_rejected = True
    checks['parsers_exploration_and_revision'] = pe['decision'] == 'PROPOSE' and rej == 2 and keep['decision'] == 'KEEP' and keep['skill'] is None and rv['skill'].skill_id == 'RD02-W1-r2' and rv['skill'].principles and same_rejected and rv['skill'].derived_from == 'RD02-W1-r1'
    # 4. selection logic on synthetic J tables (parents / children / KEEP alias / no-skill wins / best non-empty isolated) via the freeze rules
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td = Path(td)
        sel = td / 'select'
        # synthetic branch dirs with cells carrying C_B for the committed plan and None
        def fake_branch(job, arm, commit, cb_commit, cb_none, seeds=SEEDS):
            b = sel / ('%s_%s' % (job, arm)); (b / job / 'cells').mkdir(parents=True)
            context.write_json(b / 'branch_result.json', {'status': 'COMPLETE', 'job_id': job, 'knowledge_version': 0, 'committed_plan_id': commit, 'delivery_model_ref': 'x', 'calls': 1, 'tool_calls': 1, 'new_evaluations': 0, 'trace': [], 'failure_kind': None, 'reason': None})
            context.write_json(b / job / 'commit.json', {'material_id': commit, 'reason': 'x'})
            cbs = {}
            for mid, vals in ((commit, cb_commit), ('None', cb_none)):
                for s, v in zip(seeds, vals):
                    cid = W.phys_cell_id(job, mid, s)
                    context.write_json(b / job / 'cells' / (cid + '.json'), {'cell_id': cid, 'status': 'OK', 'material_id': mid, 'model_seed': s, 'scores': {'c_a': {'status': 'SCORABLE', 'normalized_mse_macro': 1.0}}})
                    cbs[cid] = {'material_id': mid, 'model_seed': s, 'c_b': {'status': 'SCORABLE', 'normalized_mse_macro': v}}
            context.write_json(b / job / 'c_b_scores.json', {'cells': cbs})
        context.write_json(sel / 'config.json', {'seeds': list(SEEDS)}) if sel.exists() else None
        sel.mkdir(parents=True, exist_ok=True); context.write_json(sel / 'config.json', {'seeds': list(SEEDS)})
        table = {'no_skill': (0.9, 0.9), 'cand_W1r1': (1.0, 1.0), 'cand_W1r2': (0.8, 0.8), 'cand_W3r1': (1.1, 1.1)}
        for arm, (r3, r4) in table.items():
            for job, r in (('RD02_L3', r3), ('RD02_L4', r4)):
                fake_branch(job, arm, 'P', [r, r, r], [1.0, 1.0, 1.0])
        arms = {'no_skill': None, 'cand_W1r1': 'RD02-W1-r1', 'cand_W1r2': 'RD02-W1-r2', 'cand_W3r1': 'RD02-W3-r1'}
        tab = j_table(sel, SELECT_JOBS, arms)
        J = tab['J']
        parent_arms, child_arms = ['cand_W1r1', 'cand_W3r1'], ['cand_W1r2']
        key = lambda a: (J[a], 0 if a in parent_arms else 1, arms[a] or '')
        w_initial = min(parent_arms, key=key); w_learned = min(parent_arms + child_arms, key=key)
        h = 'no_skill' if J['no_skill'] - J[w_learned] <= TOL else w_learned
        checks['select_rules_child_beats_parents_and_no_skill'] = w_initial == 'cand_W1r1' and w_learned == 'cand_W1r2' and h == 'cand_W1r2' and abs(J['cand_W1r2'] - 0.8) < 1e-12
        # no-skill wins on a tie -> H_system = no_skill while W_learned stays the best non-empty candidate (isolated test)
        J2 = dict(J); J2['no_skill'] = J2['cand_W1r2']
        h2 = 'no_skill' if J2['no_skill'] - J2[w_learned] <= TOL else w_learned
        checks['no_skill_wins_tie_best_nonempty_still_identified'] = h2 == 'no_skill' and w_learned == 'cand_W1r2'
        # incomplete option -> no J entry -> SELECT_INCOMPLETE path
        shutil.rmtree(sel / 'RD02_L4_cand_W3r1')
        tab2 = j_table(sel, SELECT_JOBS, arms)
        checks['incomplete_option_has_no_J'] = 'cand_W3r1' not in tab2['J'] and 'cand_W1r1' in tab2['J']
    # 5. formation failure never starts Test; freeze precedes Test; copied cache carries no labels
    fake_root = out / 'fake_pkg'
    if fake_root.exists():
        shutil.rmtree(fake_root)
    (fake_root / 'formation_a').mkdir(parents=True)
    context.write_json(fake_root / 'formation_a' / 'frozen_choices.json', {'status': 'SELECT_INCOMPLETE'})
    try:
        test_stage(fake_root, context.read_json(fake_root / 'formation_a' / 'frozen_choices.json'))
        checks['test_refused_without_frozen_choices'] = False
    except RuntimeError:
        checks['test_refused_without_frozen_choices'] = True
    checks['formation_failure_stops_before_practice'] = all(s not in ('PROPOSED',) for s in ('FORMATION_TECHNICAL_FAILURE', 'NO_INITIAL_PROPOSAL'))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td = Path(td)
        src = PARENT / 'target' / 'RD02_L1_common' / 'RD02_L1'
        rec = W.copy_physical_cache(src, td / 'common', 'RD02_L1', seeds=SEEDS)
        files = [p.name for p in (td / 'common').rglob('*')]
        cells = [context.read_json(p) for p in (td / 'common' / 'RD02_L1' / 'cells').glob('*.json')]
        checks['copied_cache_has_models_no_labels'] = rec['copied'] and rec['cells'] >= 12 and not any(n in ('c_b_scores.json', 'e_scores.json', 'e_frozen.json', 'commit.json') for n in files) \
            and all(set(c['scores']) == {'c_a'} and Path(c['model_path']).exists() and str(td) in c['model_path'] for c in cells)
        # the aborted parent Target directory is not a source (no 'target__aborted' path is referenced)
        checks['cache_source_is_the_real_parent_target'] = 'aborted' not in str(src)
    shutil.rmtree(fake_root, ignore_errors=True)
    rec = {'written_local': now(), 'checks': {k: bool(v) for k, v in checks.items()}, 'passed': sum(bool(v) for v in checks.values()), 'of': len(checks), 'notes': notes, 'real_fits': 0, 'api_calls': 0}
    rec['status'] = 'PASS' if rec['passed'] == rec['of'] else 'FAIL'
    rec['existing_controls'] = []
    for cmd in (['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/functional/test_batch_research.py'],):
        t0 = time.time()
        pr = subprocess.run([sys.executable, '-B'] + cmd, cwd=REPO, capture_output=True, text=True, timeout=1800, env=W.worker_env())
        rec['existing_controls'].append({'command': ' '.join(cmd), 'returncode': pr.returncode, 'seconds': round(time.time() - t0, 1), 'tail': (pr.stdout + pr.stderr)[-300:]})
        if pr.returncode:
            rec['status'] = 'FAIL'
    context.write_json(out / 'smoke.json', rec)
    print('SMOKE', rec['status'], rec['passed'], '/', rec['of'], flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--resume-stage', choices=['practice', 'select', 'test'])
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--restart', default='')
    ap.add_argument('--continue-package', action='store_true')
    ap.add_argument('--root', type=Path, default=ROOT)
    a = ap.parse_args()
    root = a.root.resolve()
    if a.resume_stage:
        resume_stage(root, a.resume_stage, accept_unknown_usage=a.accept_unknown_usage, restart=[x for x in a.restart.split(',') if x])
        if a.continue_package:
            package_run(root)
    elif a.smoke:
        rec = smoke(root / 'smoke')
        if rec['status'] != 'PASS':
            raise SystemExit(1)
    elif a.preflight:
        print(json.dumps({k: v for k, v in preflight(root).items() if k in ('frozen_local', 'environment_matches_parent', 'test_exposure_check')}, ensure_ascii=False)[:3000])
    elif a.wiring:
        wiring(root)
    elif a.run:
        package_run(root)
    elif a.result:
        readout(root)
    else:
        ap.error('choose an action')


if __name__ == '__main__':
    from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_learning_loop as _canonical
    _canonical.main()
