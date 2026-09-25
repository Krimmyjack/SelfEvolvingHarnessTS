"""DEV-DATA-READINESS-DOMAIN-WORKFLOW-EVOLVE-R2 (docs/DEV_DATA_READINESS_DOMAIN_WORKFLOW_EVOLVE_R2_TASK_2026-09-17.md).

RD02 only. One package, one ledger, parent package read-only:
  preflight (0 cost: T-only eligibility of RD02_T3/T4, exposure check, semantic checks of the reused parent branches)
  -> Stage A: evidence census of the parent's legal RD02 use (no E) + ONE Slow revision (KEEP or C1/C2; 1 contract correction)
  -> Stage B: each new candidate runs one complete Fast on RD02_V1 and RD02_T1 (parent common baselines reused as cache);
     every new commit before any new C_B; NO_SKILL / H_old dev readings reused from the parent's actual commits
  -> freeze H_challenger (best new candidate) and H_selected (NO_SKILL / H_old / H_challenger) before any Test fit
  -> Stage C: RD02_T3 and RD02_T4 with F0, F_old, F_new, F_generic, Random (+ the two common programs);
     all Test commits -> C_B -> freeze every E prediction -> barrier -> E (+ shadow)
  -> readout.

Reused unchanged: the Fast loop and adapter (batch_research_data_readiness.fast_branch / random_branch / open_labels /
ReadinessAdapter, FAST_SYSTEM, CONTRACTS), the metered client and ledger, rt.fit, the Skill carrier / make_skill /
two-phase select, the Generic control and the E cohort barrier.

  --smoke                 new-risk smoke (0 real fits, 0 API, no label row of a planned Test job) + existing controls
  --run                   whole package (stops at the first stage that does not finish; no automatic extension)
  --resume-stage STAGE    one operator continuation of dev/test while its labels are still withheld
  --result                readout only
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

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha.batch_base import budget, context, llm
from methods.ttha.batch_base import readiness as rd
from evaluation.main_protocol_p4 import batch_research_data_readiness as drs
from evaluation.main_protocol_p4 import batch_research_domain_skill as dsks
from evaluation.main_protocol_p4 import batch_research_runtime as rt

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_data_readiness_domain_workflow_evolve_r2'
PARENT = drs.ROOT
TASK = 'docs/DEV_DATA_READINESS_DOMAIN_WORKFLOW_EVOLVE_R2_TASK_2026-09-17.md'
DOMAIN = 'RD02'
H_OLD_ID = 'RD02-W1-r1'
SEEDS = drs.SEEDS
LIMITS = drs.LIMITS
DEV_JOBS = ('RD02_V1', 'RD02_T1')
TEST_JOBS = ('RD02_T3', 'RD02_T4')
TEST_TABLE = {'RD02_T3': {'t': 9360, 'missing_rate_pct': 1.5253, 'min_legal_parents': 321, 'T': [8688, 9360], 'C_A': [9360, 9456], 'C_B': [9456, 9552], 'E': [9552, 9744]},
              'RD02_T4': {'t': 10560, 'missing_rate_pct': 4.1295, 'min_legal_parents': 192, 'T': [9888, 10560], 'C_A': [10560, 10656], 'C_B': [10656, 10752], 'E': [10752, 10944]}}
SEALED_FROM_ROW = 24864                         # 2016-01-01 00:00 of the hourly file (2016+ region kept closed, docs/HARNESS_RESEARCH_DIRECTION_AND_PLAN.md §2.2)
PARENT_COMMON = {'RD02_V1': 'select/RD02_V1_common', 'RD02_T1': 'target/RD02_T1_common'}
PARENT_DEV = {dsk.NO_SKILL: {'RD02_V1': 'select/RD02_V1_no_skill', 'RD02_T1': 'target/RD02_T1_no_skill'},
              H_OLD_ID: {'RD02_V1': 'select/RD02_V1_cand_W1', 'RD02_T1': 'target/RD02_T1_known_domain'}}
DEV_ORDER = {'RD02_V1': ('C1', 'C2'), 'RD02_T1': ('C2', 'C1')}
TEST_ORDER = {'RD02_T3': ('F0', 'F_old', 'F_new', 'F_generic', 'Random'), 'RD02_T4': ('F_generic', 'F_new', 'F_old', 'F0', 'Random')}
TEST_RANDOM_SEEDS = {'RD02_T3': (2026091711, 2026091712), 'RD02_T4': (2026091713, 2026091714)}
ARM_OF_OPTION = {dsk.NO_SKILL: 'F0', H_OLD_ID: 'F_old'}          # H_challenger -> F_new
TOTAL = {'max_fit_attempts': 98, 'max_llm_requests': 194, 'max_llm_tokens': 4_000_000, 'max_wall_s': 21600, 'max_retries': 2}
HTTP_CAP = 388
ALLOC = {'revision': {'fits': 0, 'requests': 2, 'tokens': 250_000},
         'dev': {'fits': 24, 'requests': 64, 'tokens': 1_250_000},
         'test': {'fits': 72, 'requests': 128, 'tokens': 2_500_000}}
REVISION_MAX_TOKENS = 12000
EVIDENCE_MAX_BYTES = 82_000                    # whole request bytes B: first check 2*(B+2048+12000) = 192k; correction (prior <= 4000 chars) 2*(B+5000+14048) = 202k
                                               # plus first usage <= B/2.4+12000 = 46k -> 248k <= 250k (the client's conservative token rule)
CORRECTION_PRIOR_MAX = 4000
MODEL = drs.MODEL
COMPAT_NOTE = 'Formed for this study\'s fixed task, Consumer, L/H, sampling interval and T length.'
CHANGE_TARGETS = ('observation', 'construction', 'experiment', 'commit')
REVISION_KEYS = frozenset(dsk.CANDIDATE_KEYS | {'change_target', 'process_evidence', 'expected_behavior_change'})
EVIDENCE_BRANCHES = (   # parent-relative branch, job, role, role note (roles are labels of how the trajectory was produced)
    ('source/RD02_S1_no_skill', 'RD02_S1', 'no_skill', 'Fast without any Skill (earlier batch; the parent formation input)'),
    ('source/RD02_S2_no_skill', 'RD02_S2', 'no_skill', 'Fast without any Skill (earlier batch; the parent formation input)'),
    ('select/RD02_V1_no_skill', 'RD02_V1', 'no_skill', 'Fast without any Skill (parent selection reference)'),
    ('select/RD02_V1_cand_W1', 'RD02_V1', 'old_card', 'Fast with the parent card while it was candidate W1 (then selected and frozen as the parent card)'),
    ('select/RD02_V1_cand_W2', 'RD02_V1', 'parent_candidate_W2', 'Fast with the parent candidate W2; not selected (tied with W1 on C_B, lost by option order); not validated knowledge'),
    ('target/RD02_T1_no_skill', 'RD02_T1', 'no_skill', 'Fast without any Skill'),
    ('target/RD02_T1_generic', 'RD02_T1', 'generic_control', 'Fast with the frozen hand-written Generic guidance (a control, not a learned Skill)'),
    ('target/RD02_T1_known_domain', 'RD02_T1', 'old_card', 'Fast with the frozen parent card'),
    ('target/RD02_T1_random', 'RD02_T1', 'random_control', 'Not an Agent trajectory: two frozen random plans; the lowest three-seed C_A mean among the two common programs and the two random plans was committed'),
    ('target/RD02_T2_no_skill', 'RD02_T2', 'no_skill', 'Fast without any Skill'),
    ('target/RD02_T2_generic', 'RD02_T2', 'generic_control', 'Fast with the frozen hand-written Generic guidance (a control, not a learned Skill)'),
    ('target/RD02_T2_known_domain', 'RD02_T2', 'old_card', 'Fast with the frozen parent card'),
    ('target/RD02_T2_random', 'RD02_T2', 'random_control', 'Not an Agent trajectory: two frozen random plans; the lowest three-seed C_A mean among the two common programs and the two random plans was committed'),
)
TECHNICAL_RECORDS = (('source/RD02_S1_no_skill__interrupted_1', 'RD02_S1'),)
EVIDENCE_JOBS = ('RD02_S1', 'RD02_S2', 'RD02_V1', 'RD02_T1', 'RD02_T2')
BRANCH_CODE = {rel: 'b%02d' % i for i, (rel, _, _, _) in enumerate(EVIDENCE_BRANCHES, 1)}
FORBIDDEN_READ = re.compile(r'e_scores|e_frozen|predictions_e|all_e_predictions|REPORT\.md|(?<![A-Za-z_])result\.json|CLAIMS_FROM_RESULTS', re.I)

REVISION_TASK_TEXT = '你正在为同一域改进一份可复用的数据准备 Workflow。依据给定旧卡、真实使用轨迹、材料与 C_A/延迟 C_B 结果，识别一个主要的可改善研究行为；也可以判断证据不足而 KEEP。若提案，最多给两种不同流程，说明观察什么、用什么证据构造或选择对照、如何根据返回结果继续或停止。目标是固定 Consumer 的预测效用和完整成本，不能以多改数据、少改数据、多观察、换算子或更多步骤作为成功。事实、假设与未知分开；未测不等于有害。不要把一次胜负扩大为全域禁令；不要根据实体预测差给训练材料贴因果标签。允许优先具体程序，但不得写历史站点/Job答案表。保持执行工具、预算、训练和计分不变。'

REVISE_SYSTEM = '''You are the offline Slow of a batch data-readiness research Harness. You revise ONE neutral domain's reusable data-preparation Workflow card, once. Input: the parent card (Workflow, Principles, scope, its formation and selection record, and where it was actually loaded), the parent's candidate that was not selected (labeled; not validated knowledge), and a deterministic census of real use of this domain on five earlier batches: T-only observations, Fast trajectories with explicit roles (no Skill; the frozen hand-written Generic control; the parent card; the unselected parent candidate) with the actual tool calls and returned results, complete input-preparation plans, material diagnostics, C_A feedback, commits, post-commit delayed C_B, costs and failures, plus a Random control (two frozen random plans committed by C_A; not an Agent trajectory). Structural and technical records are listed apart from completed trajectories. The Fast public prompt, tools and budgets were the same for every role. No E values, E rankings or future-batch data exist in this input.
''' + REVISION_TASK_TEXT + '''
(You are improving a reusable data-preparation Workflow for the same domain. From the given parent card, real use trajectories, materials and C_A / delayed C_B results, identify ONE main research behavior that can be improved, or judge that the evidence is insufficient and KEEP. If you propose, give at most two different processes, stating what to observe, what evidence to use to construct or choose comparisons, and how to continue or stop from the returned results. The goal is the fixed Consumer's predictive utility and the full cost; changing more data, changing less data, observing more, switching operators or taking more steps is not success. Keep facts, hypotheses and unknowns apart; untested is not harmful. Do not widen one win or loss into a domain-wide ban; do not attach causal labels to training material from entity prediction differences. Preferring concrete programs is allowed, but do not write a table of answers for historical stations or jobs. Keep the execution tools, budgets, training and scoring unchanged.)
Each candidate focuses on one main research behavior; two candidates must differ in research priority or process (they need not use different operators). A card grants no new permission and cannot change tools, budgets, model, scoring, targets or legal windows. Do not write data source names, job labels or historical entity ids in any text. Output exact JSON, no markdown: {"decision":"KEEP","rationale":"..."} OR {"decision":"PROPOSE","candidates":[{"candidate_id":"C1","research_mode":"<short label>","workflow":"...","principles":"..." or null,"observable_applicability":{"const":true},"applicability_summary":"<=400 characters","evidence_refs":["ref strings copied from the evidence"],"rationale":"...","change_target":"observation|construction|experiment|commit","process_evidence":"<=400 characters: the process evidence this revision answers","expected_behavior_change":"<=300 characters: which observable Fast behavior should change on a new batch"}]}. Candidate ids are C1, then C2. Workflow plus Principles render to at most 1200 characters. const:true is explicit unconditional; null is incomplete. Conditional scopes use only the supplied batch feature names with numeric {feature,op,value} leaves. KEEP is valid and will not be resampled; every candidate is CANDIDATE_TEST_ONLY until real development selection.'''


def paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'configs': root / 'stage_configs', 'logs': root / 'logs',
            'preflight': root / 'preflight', 'revision': root / 'revision', 'dev': root / 'dev', 'freeze': root / 'freeze', 'test': root / 'test',
            'smoke': root / 'smoke'}


def stage_caps(root: Path, stage: str) -> dict:
    P = paths(root)
    return dsks.stage_caps(root, stage, total=TOTAL, alloc=ALLOC, http_cap=HTTP_CAP, stages_path=P['stages'], ledger_path=P['ledger'])


def _r(x, nd: int = 4):
    if isinstance(x, float):
        return round(x, nd) if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    return x


def parent_card() -> dsk.Skill:
    fr = context.read_json(PARENT / 'formation' / DOMAIN / 'frozen_skill.json')
    if fr['status'] != 'FROZEN_SELECTED' or fr['skill']['skill_id'] != H_OLD_ID:
        raise RuntimeError('parent frozen card is not %s' % H_OLD_ID)
    return dsk.skill_from_json(fr['skill'])


def _same_policy(k1: dict, k2: dict) -> bool:
    """Knowledge equality as Fast sees it: version and each entry's hook, body and scope (evidence refs are not rendered)."""
    strip = lambda k: (k['version'], [(e['hook'], e['body'], json.dumps(e['applicability'], sort_keys=True)) for e in k['entries']])
    return strip(k1) == strip(k2)


def _knowledge(skill) -> dict:
    return br.json_copy(asdict(dsk.skill_knowledge(skill)))


# ============================================================================= Stage A: evidence census (0 cost) and one revision
def _compact_output(tool: str, out):
    """Short deterministic summaries of what each tool returned (the call arguments stay in the Fast actions)."""
    if not isinstance(out, dict):
        return out
    if tool == 'overview':
        return 'job overview'
    if tool == 'inspect_data':
        kind, ents = out.get('kind'), {}
        for e, v in (out.get('entities') or {}).items():
            k = e.replace('entity_', 'e')
            if kind == 'gaps':
                g = v['gaps']
                ents[k] = [v['missing'], len(g), max((x[2] for x in g), default=0), [[x[0], x[2]] for x in g if x[2] >= 6]]
            elif kind == 'hour_profile':
                fin = [(h['mean'], i) for i, h in enumerate(v) if h['mean'] is not None]
                ents[k] = [max(fin)[1], min(fin)[1], round(max(fin)[0] - min(fin)[0], 2)] if fin else None
            elif kind == 'daily_means':
                m = [d['mean'] for d in v if d['mean'] is not None]
                ents[k] = [round(min(m), 2), round(max(m), 2), sum(x > 1 for x in m), sum(d['missing'] for d in v)] if m else None
            elif kind == 'segment':
                vals = v['values']
                fin = [x for x in vals if x is not None]
                ents[k] = [v['start_row'], len(vals), len(vals) - len(fin), round(min(fin), 2) if fin else None, round(max(fin), 2) if fin else None]
            else:
                ents[k] = v
        return {'kind': kind, 'entities': ents}
    if tool == 'inspect_material':
        bt = out.get('batch_totals') or {}
        rec = {'material_id': out.get('material_id'), 'alias_of': out.get('alias_of'),
               'totals': {k: bt.get(k) for k in ('windows_with_missing', 'filled_points', 'observed_changed_points', 'residual_nan_linear_fill')},
               'operator_records': bt.get('operator_records')}
        rows = {}
        for e, v in (out.get('entities') or {}).items():
            if (v.get('fill_difference_from_linear_rms_norm') or 0) >= 0.5 or (v.get('observed_changed_fraction') or 0) > 0:
                rows[e.replace('entity_', 'e')] = [v.get('filled_points'), _r(v.get('observed_changed_fraction'), 3), _r(v.get('fill_difference_from_linear_rms_norm'), 2),
                                                   _r(v.get('window_std_ratio_vs_linear'), 3)]
        rec['entities'] = rows
        w = out.get('window')
        if isinstance(w, dict):
            bl, pr = w.get('baseline_linear'), w.get('prepared')
            win = {'entity': str(w.get('entity')).replace('entity_', 'e'), 'raw_missing': sum(x is None for x in (w.get('raw') or []))}
            if isinstance(bl, list) and isinstance(pr, list):
                d = [p - b for p, b in zip(pr, bl) if p is not None and b is not None]
                win['prepared_minus_linear_rms'] = _r(math.sqrt(sum(x * x for x in d) / len(d)), 3) if d else None
            rec['window'] = win
        return rec
    if tool == 'build_material':
        ms = out.get('material_spec') or {}
        rec = {'plan_id': out.get('plan_id')}
        rec.update({k: _r(ms[k]) for k in ('alias_of', 'canonicalized', 'resolved_thresholds') if ms.get(k)})
        return rec
    if tool == 'evaluate':
        fb = out.get('feedback') or {}
        return {'plan_id': out.get('plan_id'), 'c_a_by_seed': _r(fb.get('loss_by_seed')), 'c_a_by_seed_origin': _r(fb.get('loss_by_seed_origin'), 3)}
    if tool == 'compare':
        return {k: _r(out.get(k)) for k in ('a', 'b', 'mean_delta', 'seed_se', 'signs')}
    return _r(out)


def _dedupe(item: dict, ref: str, seen: dict) -> dict:
    """Within one job, a tool output (or one entity of an inspection) identical to an earlier one is replaced by that earlier ref."""
    out = item.get('output')
    if not isinstance(out, dict):
        return item
    if item['tool'] == 'inspect_data' and isinstance(out.get('entities'), dict):
        ents = {}
        for e, v in out['entities'].items():
            key = json.dumps([out.get('kind'), out.get('rows_read'), e, v], sort_keys=True, ensure_ascii=False)
            ents[e] = v if key not in seen else 'same as %s' % seen[key]
            seen.setdefault(key, ref)
        return {**item, 'output': {**out, 'entities': ents}}
    if item['tool'] == 'inspect_material':
        body = {k: v for k, v in out.items() if k != 'window'}
        key = json.dumps([item['tool'], body], sort_keys=True, ensure_ascii=False)
        if key in seen:
            return {**item, 'output': {'material_id': out.get('material_id'), 'summary': 'same as %s' % seen[key], **({'window': out['window']} if 'window' in out else {})}}
        seen.setdefault(key, ref)
        return item
    key = json.dumps([item['tool'], out], sort_keys=True, ensure_ascii=False)
    if key in seen and item['tool'] == 'compare':
        return {**item, 'output': 'same as %s' % seen[key]}
    seen.setdefault(key, ref)
    return item


def _compact_trace(rows: list, prefix: str, seen: dict | None = None) -> list:
    seen = {} if seen is None else seen
    out = []
    for r in rows:
        ev = r['event']
        if ev in ('job_started', 'tool_started', 'committed'):
            continue
        ref = '%s:%s' % (prefix, str(r['event_id']).rsplit(':', 1)[-1])
        if ev == 'fast_request' or (ev == 'tool_completed' and r.get('tool') in ('commit', 'overview')):
            continue
        if ev == 'fast_response':
            resp = r.get('response')
            acts = resp.get('actions') if isinstance(resp, dict) else None
            item = {'fast_response': r.get('number'), 'actions': [[x.get('tool'), x.get('arguments')] if isinstance(x, dict) else x for x in acts] if isinstance(acts, list) else resp}
        elif ev == 'tool_completed':
            item = _dedupe({'tool': r['tool'], 'output': _compact_output(r['tool'], r.get('output'))}, ref, seen)
        elif ev == 'tool_rejected':
            item = {'rejected_tool': r.get('tool'), 'arguments': r.get('arguments'), 'error': (r.get('error') or {}).get('message')}
        elif ev == 'random_material':
            ms = r.get('material_spec') or {}
            item = {'random_plan': r['plan_id'], 'default': (ms.get('policy') or {}).get('default'), 'rules': (ms.get('policy') or {}).get('rules')}
        elif ev == 'random_evaluated':
            fb = r.get('feedback') or {}
            item = {'random_evaluated': r['plan_id'], 'c_a_by_seed': _r(fb.get('loss_by_seed'))}
        elif ev == 'job_incomplete':
            item = {'incomplete': r.get('failure_kind'), 'reason': r.get('reason')}
        else:
            s = json.dumps(_r({k: v for k, v in r.items() if k not in ('event_id', 'delivery_model_ref')}), ensure_ascii=False)
            item = {'event': ev, 'content': s[:600]}
        out.append({'ref': ref, **item})
    return out


def _material_totals(summary_path) -> dict:
    summ = context.read_json(summary_path)
    tot = {k: int(sum(e[k] for e in summ['entities'])) for k in ('windows', 'windows_with_missing', 'filled_points', 'observed_points', 'observed_changed_points')}
    tot['x_fill_fraction'] = _r(tot['filled_points'] / (tot['windows'] * rd.L), 5) if tot['windows'] else None
    tot['observed_changed_fraction'] = _r(tot['observed_changed_points'] / tot['observed_points'], 5) if tot['observed_points'] else None
    return tot


def _job_block(bdir: Path, job: str) -> dict:
    ov = context.read_json(bdir / job / 'overview.json')
    fields = ['missing_fraction', 'longest_gap', 'lag24_corr', 'robust_z_max', 'recent168_level_shift', 'recent168_volatility_ratio', 'n_legal_parents']
    blocks = drs._plan_blocks(bdir, job)
    index = rd.load_index(bdir / job)
    base = {}
    for mid, _ in drs.BASELINES:
        sc = blocks.get(mid, {})
        tot = _material_totals(index[mid]['summary_path'])
        base[mid] = {'c_a_by_seed': _r([sc[s]['c_a'] for s in SEEDS]), 'c_b_by_seed': _r([sc[s]['c_b'] for s in SEEDS]), 'x_fill_fraction': tot['x_fill_fraction']}
    return {'train_rows': ov['train_rows'], 'legal_parents_total': ov['geometry']['legal_parents_total'],
            'summary_p25_median_p75_max': {f: (None if v is None else _r([v['p25'], v['median'], v['p75'], v['max']], 3)) for f, v in ov['summary'].items()
                                           if f not in ('head_gap', 'tail_gap', 'finite_mean', 'finite_std')},
            'entity_rows': {'columns': fields, 'rows': [[_r(e.get(f), 2) for f in fields] for e in ov['entities']]},
            'common_programs': base}


def _branch_record(bdir: Path, job: str, role: str, note: str, stage_root: Path, seen: dict, code: str) -> tuple:
    res = context.read_json(bdir / 'branch_result.json')
    prefix = code
    rows = drs._trace(bdir)
    traj = _compact_trace(rows, prefix, seen)
    index = rd.load_index(bdir / job)
    blocks = drs._plan_blocks(bdir, job)
    plans = {}
    for mid, rec in index.items():
        if mid in dict(drs.BASELINES):
            continue
        comp = context.read_json(bdir / job / 'materials' / (mid + '__compiled.json'))['material_spec']
        sc = blocks.get(mid, {})
        fitted = set(sc) == set(SEEDS)
        tot = _material_totals(rec['summary_path'])
        plans[mid] = {**({'alias_of': rec['alias_of']} if rec['alias_of'] else {}), 'entity_program': comp.get('entity_program'),
                      'x_fill_fraction': tot['x_fill_fraction'], 'observed_changed_fraction': tot['observed_changed_fraction'],
                      'c_a_by_seed': _r([sc[s]['c_a'] for s in SEEDS]) if fitted else None,
                      'c_b_after_commit_by_seed': _r([sc[s]['c_b'] for s in SEEDS]) if fitted else None}
        if fitted and {str(sc[s]['c_b_status']) for s in SEEDS} != {'SCORABLE'}:
            plans[mid]['c_b_status'] = sorted({str(sc[s]['c_b_status']) for s in SEEDS})
    tok = dsks._unit_tokens(stage_root).get('fast:%s' % bdir.name, {})
    beh = dsks._behavior(bdir)
    rec = {'branch': prefix, 'status': res['status'], **({'failure_kind': res['failure_kind']} if res['failure_kind'] else {}), 'committed_plan_id': res['committed_plan_id'],
           'cost': {'fast_calls': res['calls'], 'tool_calls': res['tool_calls'], 'new_evaluations': res['new_evaluations'],
                    'fast_tokens': tok.get('prompt_tokens', 0) + tok.get('completion_tokens', 0)},
           'requests_with_loaded_guidance': '%d/%d' % (beh['requests_with_loaded_guidance'], beh['fast_requests']),
           'trajectory': traj, 'plans_built_in_branch': plans, 'delayed_block_ref': '%s:c_b_after_commit' % prefix}
    return rec, [t['ref'] for t in traj] + [rec['delayed_block_ref']]


def revision_evidence(parent: Path = PARENT, *, branches=EVIDENCE_BRANCHES, technical=TECHNICAL_RECORDS) -> dict:
    """Whitelist census of the parent's legal RD02 use: traces, materials, C_A cells and post-commit C_B; never an E file,
    a report/result/claims file, or any raw data row."""
    h_old = parent_card()
    jobs, recs, refs, seen_by_job = {}, [], [], {}
    for rel, job, role, note in branches:
        b = parent / rel
        if not drs._branch_complete_with_c_b(b, job):
            raise RuntimeError('evidence branch is not COMPLETE with post-commit C_B: %s' % rel)
        stage_root = b.parent
        jb = _job_block(b, job)
        if job not in jobs:
            jobs[job] = jb
            refs.append('%s:overview' % job)
        elif jb != jobs[job]:
            raise RuntimeError('branches of %s disagree on the job overview or common programs' % job)
        rec, r = _branch_record(b, job, role, note, stage_root, seen_by_job.setdefault(job, {}), BRANCH_CODE[rel])
        recs.append(rec)
        refs += r
    tech = []
    for rel, job in technical:
        b = parent / rel
        res = context.read_json(b / 'branch_result.json')
        tech.append({'attempt': '%s/no_skill_interrupted_attempt' % job, 'status': 'INTERRUPTED_TECHNICAL', 'failure_kind': res['failure_kind'], 'reason': res['reason'],
                     'calls': res['calls'], 'tool_calls': res['tool_calls'], 'new_evaluations': res['new_evaluations'],
                     'note': 'first request used 0-based sub_range offsets and hit the hard out-of-T permission check before any build or evaluation; '
                             'the interface was then amended (out-of-T sub_range is a recoverable input error) and the branch restarted; not a scientific outcome'})
    prop = context.read_json(parent / 'formation' / DOMAIN / 'propose.json')
    sel = context.read_json(parent / 'formation' / DOMAIN / 'selection.json')
    w2 = next(s for s in prop['skills'] if s['skill_id'] == 'RD02-W2-r1')
    loading = {}
    for rel, job, role, _ in branches:
        if role == 'old_card':
            beh = dsks._behavior(parent / rel)
            loading[BRANCH_CODE[rel]] = {'fast_requests': beh['fast_requests'], 'requests_with_loaded_card': beh['requests_with_loaded_guidance'], 'scope': 'const:true (MATCH)'}
    state = {
        'parent_card': {'ref': 'parent_card:%s' % H_OLD_ID, 'skill_id': H_OLD_ID, 'workflow': h_old.workflow, 'principles': h_old.principles,
                        'observable_applicability': h_old.observable_applicability, 'applicability_summary': h_old.applicability_summary,
                        'status': 'FROZEN_SELECTED in the parent package; used as the old card below'},
        'parent_formation': {'input_trajectories': [BRANCH_CODE['source/RD02_S1_no_skill'], BRANCH_CODE['source/RD02_S2_no_skill']], 'decision': prop['status'], 'research_modes': prop.get('research_modes'),
                             'rationales': prop.get('rationales')},
        'parent_candidate_not_selected': {'ref': 'parent_candidate:RD02-W2-r1', 'skill_id': 'RD02-W2-r1', 'workflow': w2['workflow'], 'principles': w2['principles'],
                                          'observable_applicability': w2['observable_applicability'],
                                          'status': 'CANDIDATE_TEST_ONLY; tied with W1 on the selection C_B and not selected by option order; not validated knowledge'},
        'parent_selection': {'ref': 'parent_selection:RD02_V1', 'dev_jobs': sel['dev_jobs'], 'block': sel['block'], 'rule': sel['rule'], 'status': sel['status'],
                             'selected': sel['selected'], 'eligible_scores': _r(sel['eligible_scores']),
                             'runs': {o: {j: {'committed_plan_id': v['committed_plan_id'], 'c_b_by_seed': _r(v.get('loss_by_seed')), 'cost': v.get('cost')} for j, v in per.items()}
                                      for o, per in sel['runs'].items()}},
        'actual_loading_of_parent_card': loading,
    }
    refs += ['parent_card:%s' % H_OLD_ID, 'parent_candidate:RD02-W2-r1', 'parent_selection:RD02_V1']
    ev = {'domain_id': DOMAIN, 'profile': 'data_readiness',
          'reading_notes': {'blocks': 'C_A = two origins right after T (the only feedback Fast had); C_B = two later origins opened only after the branch committed; '
                                      'normalized MSE on observed future cells, lower is better; three seeds are Consumer repeats of one trajectory, not independent trajectories',
                            'batches': 'five earlier batches of the same entities at increasing times; different branches of one batch share T, the two common programs and their fitted models',
                            'refs': 'every ref string in this evidence is a legal evidence ref (branch code:event number, <job>:overview, <branch code>:c_b_after_commit, parent_*)',
                            'inspect_data_summaries': {'gaps': '[missing hours, number of gaps, longest gap hours, [[start row, hours] of gaps >= 6h]]',
                                                       'hour_profile': '[peak clock hour, trough clock hour, range of hourly means]',
                                                       'daily_means': '[min daily mean, max daily mean, days with mean > 1, missing hours]',
                                                       'segment': '[start row, rows, missing, min, max]', 'units': 'normalized by the frozen per-entity T scaler; eN = entity index N'},
                            'inspect_material_summary': 'totals, operator records, and entities with fill difference from linear >= 0.5 or changed observations: '
                                                        '[filled points, observed changed fraction, fill difference from linear rms, window std ratio vs linear]',
                            'actions': '[tool, arguments] in the order Fast sent them; identical later outputs of the same batch say same as <ref>',
                            'omitted': 'raw values, material arrays and the unchanged public Fast prompt; no E exists here'},
          'branch_legend': {BRANCH_CODE[rel]: {'batch': job, 'role': role, 'note': note} for rel, job, role, note in branches},
          'parent_state': state, 'jobs': jobs, 'trajectories': recs, 'structural_and_technical_records': tech,
          'public_semantics': {'tool_arguments': {k: v['arguments'] for k, v in drs.CONTRACTS.items()}, 'limits': {**LIMITS, 'max_tool_corrections': drs.MAX_TOOL_CORRECTIONS},
                               'field_definitions': rd.FIELD_DEFINITIONS, 'ops_parameters': rd.OPS, 'op_semantics': rd.OP_SEMANTICS, 'canonicalization': rd.CANONICAL_RULE,
                               'common_programs': {'Baseline_Linear': 'empty program: minimum linear completion', 'Fixed_Seasonal': 'uniform period_median_complete(24,3,2)'}}}
    ev['legal_evidence_refs'] = sorted(set(refs))
    low = json.dumps(ev, ensure_ascii=False).lower()
    if any(n in low for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')):
        raise PermissionError('revision evidence names a data source')
    return ev


def revision_payload(evidence: dict) -> dict:
    return {'domain_id': DOMAIN, 'evidence': {k: v for k, v in evidence.items() if k != 'legal_evidence_refs'},
            'allowed_batch_features': sorted(drs.ALLOWED_FEATURES), 'max_candidates': 2, 'change_targets': list(CHANGE_TARGETS), 'status': 'CANDIDATE_TEST_ONLY'}


def request_bytes(payload: dict) -> int:
    messages = [{'role': 'system', 'content': REVISE_SYSTEM}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
    return len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))


def parse_revision(resp, *, legal_refs, h_old: dsk.Skill) -> dict:
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'PROPOSE'):
        raise ValueError('decision must be KEEP or PROPOSE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale'}:
            raise ValueError('KEEP cannot carry candidates')
        return {'decision': 'KEEP', 'rationale': str(resp.get('rationale', ''))[:1000], 'candidates': []}
    if set(resp) != {'decision', 'candidates'}:
        raise ValueError('PROPOSE carries exactly decision and candidates')
    cands = resp['candidates']
    if not isinstance(cands, list) or not 1 <= len(cands) <= 2:
        raise ValueError('PROPOSE needs 1..2 candidates')
    out, modes = [], set()
    for i, c in enumerate(cands):
        if not isinstance(c, dict) or set(c) != REVISION_KEYS:
            raise ValueError('candidate keys must be exactly %s' % sorted(REVISION_KEYS))
        cid = c['candidate_id']
        if cid != 'C%d' % (i + 1):
            raise ValueError('candidate ids are C1, then C2')
        mode = c['research_mode']
        if not isinstance(mode, str) or not mode.strip() or len(mode) > 80 or mode.strip().lower() in modes:
            raise ValueError('each candidate needs its own short research_mode')
        modes.add(mode.strip().lower())
        dsk.check_reusable_text(mode, 'research_mode', drs.skill_text_check)
        if c['change_target'] not in CHANGE_TARGETS:
            raise ValueError('change_target must be one of %s' % list(CHANGE_TARGETS))
        for k, n in (('process_evidence', 400), ('expected_behavior_change', 300)):
            if not isinstance(c[k], str) or not c[k].strip() or len(c[k]) > n:
                raise ValueError('%s must be nonempty text of at most %d characters' % (k, n))
        skill = dsk.make_skill(skill_id='%s-%s-r2' % (DOMAIN, cid), domain_id=DOMAIN, revision=2, workflow=c['workflow'], principles=c['principles'],
                               applicability_summary=c['applicability_summary'], compatibility_note=COMPAT_NOTE, observable_applicability=c['observable_applicability'],
                               evidence_refs=c['evidence_refs'], legal_evidence_refs=legal_refs, source_stage='revise', status='CANDIDATE_TEST_ONLY', derived_from=H_OLD_ID,
                               allowed_features=drs.ALLOWED_FEATURES, text_check=drs.skill_text_check)
        out.append({'candidate_id': cid, 'research_mode': mode.strip(), 'change_target': c['change_target'], 'process_evidence': c['process_evidence'].strip(),
                    'expected_behavior_change': c['expected_behavior_change'].strip(), 'rationale': str(c['rationale'])[:1000], 'skill': skill.to_json()})
    return {'decision': 'PROPOSE', 'candidates': classify_candidates(out, h_old)}


def classify_candidates(cands: list, h_old: dsk.Skill) -> list:
    """RUN, IDENTICAL_TO_H_OLD (same body and scope as the parent card: its readings are referenced, no run) or
    SEMANTIC_DUPLICATE_OF_C1 (same body and scope as C1: not run, not a separate option)."""
    key = lambda s: (s['rendered_body'], json.dumps(s['observable_applicability'], sort_keys=True))
    for i, c in enumerate(cands):
        if key(c['skill']) == key(h_old.to_json()):
            c['dev_status'] = 'IDENTICAL_TO_H_OLD'
        elif any(key(c['skill']) == key(p['skill']) for p in cands[:i]):
            c['dev_status'] = 'SEMANTIC_DUPLICATE_OF_C1'
        else:
            c['dev_status'] = 'RUN'
    return cands


def revise(evidence: dict, call, *, h_old: dsk.Skill, last_text=lambda: None) -> dict:
    """call(payload) -> parsed JSON (metered). One contract correction, no semantic resampling; transport/account/budget faults
    end as REVISE_CALL_FAILED (never KEEP)."""
    refs = frozenset(evidence['legal_evidence_refs'])
    payload = revision_payload(evidence)
    low = json.dumps(payload, ensure_ascii=False).lower()
    if any(n in low for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')):
        raise PermissionError('Slow payload names a data source')
    attempts = []
    for i in range(2):
        prior = None
        try:
            raw = call(payload)
        except (llm.AccountFault, llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
            attempts.append({'attempt': i, 'fault_kind': type(exc).__name__})
            return {'status': 'REVISE_CALL_FAILED', 'attempts': attempts, 'candidates': []}
        except ValueError as exc:
            attempts.append({'attempt': i, 'error': 'response not JSON: %s' % str(exc)[:200]})
            prior = (last_text() or '')[:CORRECTION_PRIOR_MAX] or None
        else:
            try:
                out = parse_revision(raw, legal_refs=refs, h_old=h_old)
                attempts.append({'attempt': i, 'ok': True})
                return {'status': 'KEEP' if out['decision'] == 'KEEP' else 'PROPOSED', 'attempts': attempts, **out, 'raw_response': br.json_copy(raw)}
            except (ValueError, rd.PolicyError) as exc:
                attempts.append({'attempt': i, 'error': str(exc)[:300]})
                prior = br.json_copy(raw) if dsk._jsonable(raw) else None
                if prior is not None and len(json.dumps(prior, ensure_ascii=False)) > CORRECTION_PRIOR_MAX:
                    prior = json.dumps(prior, ensure_ascii=False)[:CORRECTION_PRIOR_MAX]
        if i == 0:
            payload = {**payload, 'correction': {'previous_output': prior, 'error': attempts[-1].get('error'),
                                                 'instruction': 'Correct this contract error only. KEEP is allowed. Do not seek a different outcome.'}}
    return {'status': 'REVISE_PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'candidates': []}


def revision_stage(root: Path, client=None) -> dict:
    P = paths(root)
    out = P['revision']
    if (out / 'proposal.json').exists():
        return context.read_json(out / 'proposal.json')
    if not (out / 'evidence.json').exists():
        ev = revision_evidence()
        dsks.write_once(out / 'evidence.json', ev)
    ev = context.read_json(out / 'evidence.json')
    nbytes = request_bytes(revision_payload(ev))
    if nbytes > EVIDENCE_MAX_BYTES:
        raise RuntimeError('revision request %d bytes exceeds the frozen %d-byte bound' % (nbytes, EVIDENCE_MAX_BYTES))
    sc = stage_caps(root, 'revision')
    led = rt.RuntimeLedger(P['ledger'], **sc['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage; paid revision refused')
    client = client or rt.MeteredClient(led, out / 'raw_responses', http_cap=sc['http_cap'])
    res = revise(ev, lambda p: client.call('slow_revision', DOMAIN, p, REVISE_SYSTEM, max_tokens=REVISION_MAX_TOKENS), h_old=parent_card(),
                 last_text=lambda: getattr(client, 'last_text', None))
    dsks.write_once(out / 'proposal.json', {**res, 'request_bytes_first_attempt': nbytes, 'legal_ref_count': len(ev['legal_evidence_refs']), 'epoch': time.time()})
    if getattr(client, 'fatal', False) or rt.unknown_usage_blocks(led):
        raise RuntimeError('backend fatal or unknown usage during revision; stop')
    return context.read_json(out / 'proposal.json')


# ============================================================================= arms (same worker shape as the parent study)
def _run_arms(root: Path, cfg: dict, led, client, branches, failures, *, resumed: bool) -> None:
    for job in cfg['jobs']:
        if (root / (job + '_not_scorable.json')).exists():
            continue
        shared = root / (job + '_common')
        src = (cfg.get('parent_common') or {}).get(job)
        if src:
            if not shared.exists():               # the parent's C_A-only common baselines (checked in preflight); models read in place
                shared.mkdir(parents=True)
                shutil.copytree(Path(src) / job, shared / job, ignore=shutil.ignore_patterns('fit_logs'))
        else:
            shared.mkdir(exist_ok=resumed)
        before = led.s['fit_attempts']
        try:
            drs.ReadinessAdapter(shared, job, led, REPO, seeds=SEEDS).baselines()
        except RuntimeError as exc:
            if str(exc) != 'C_A_NOT_SCORABLE':
                raise
            context.write_json(root / (job + '_not_scorable.json'), {'job': job, 'status': 'C_A_NOT_SCORABLE', 'note': 'common observed-cell mask below the 25% rule; no arm of this job is run'})
            failures.append({'job': job, 'kind': 'C_A_NOT_SCORABLE'})
            continue
        if src and led.s['fit_attempts'] != before:
            raise RuntimeError('reused parent baselines must be cache hits')
        for arm in cfg['order'][job]:
            spec_ = cfg['arms'][job][arm]
            path = root / ('%s_%s' % (job, arm))
            if spec_['kind'] in ('absent', 'reference'):
                t = root / ('%s_%s_treatment.json' % (job, arm))
                if not t.exists():
                    context.write_json(t, spec_)
                continue
            branches.append((path, job))
            try:
                if path.exists() and (path / 'branch_result.json').exists():
                    prior = context.read_json(path / 'branch_result.json')
                    if prior['status'] == 'COMPLETE':
                        result = br.RunResult(**{k: v for k, v in prior.items() if k != 'trace'})
                    elif spec_['kind'] == 'random':
                        raise RuntimeError('control branch cannot be resumed')
                    else:
                        result = drs.resume_fast_branch(path, job, drs.knowledge_from_json(spec_['knowledge']), led, client)
                elif path.exists():
                    raise RuntimeError('branch directory without a result; inspect before any replay')
                elif spec_['kind'] == 'random':
                    result = drs.random_branch(path, job, led, shared, spec_['policy_seeds'])
                else:
                    result = drs.fast_branch(path, job, drs.knowledge_from_json(spec_['knowledge']), led, client, shared)
                if result.status != 'COMPLETE':
                    failures.append({'branch': path.name, 'kind': result.failure_kind, 'reason': result.reason})
            except Exception as exc:  # noqa: BLE001
                failures.append({'branch': path.name, 'exception_type': type(exc).__name__, 'message': drs._safe_message(exc)})
                print('BRANCH_FAILED', path.name, type(exc).__name__, flush=True)
            led.check_wall()
            led.check_llm()
            if getattr(client, 'fatal', False) or rt.unknown_usage_blocks(led):
                raise RuntimeError('backend fatal or unknown usage')


def stage_worker(cfg_path: Path, *, client=None, runner=drs.label_stage) -> None:
    cfg = context.read_json(cfg_path)
    root = Path(cfg['output'])
    if (root / 'experiment_started.json').exists():
        raise RuntimeError('stage already started; no paid replay')
    root.mkdir(parents=True, exist_ok=True)
    context.write_json(root / 'config.json', cfg)
    led = rt.RuntimeLedger(cfg['ledger_path'], **cfg['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage; paid stage refused without an operator decision')
    rt.FIT_RETRY = bool(cfg.get('fit_retry'))
    context.write_json(root / 'experiment_started.json', {'epoch': time.time(), 'pid': os.getpid(), 'stage': cfg['stage']})
    branches, failures, stopped = [], [], None
    try:
        client = client or rt.MeteredClient(led, root / 'raw_responses', http_cap=int(cfg['http_cap']))
        _run_arms(root, cfg, led, client, branches, failures, resumed=False)
        context.write_json(root / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches]})
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': drs._safe_message(exc)})
        stopped = type(exc).__name__
        print('EXECUTION_STOP', stopped, flush=True)
    finally:
        context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures})
    drs.open_labels(root, cfg, branches, failures, led, runner=runner,
                    withhold_reason=stopped and 'execution stopped before every planned arm was attempted (%s)' % stopped)


def resume_stage(root: Path, stage: str, *, accept_unknown_usage: bool = False, restart=()) -> None:
    """One operator continuation (labels still withheld; completed branches kept; missing arms run as planned)."""
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
    context.write_json(sroot / 'resume.json', {'epoch': time.time(), 'accepted_unknown_usage': bool(accept_unknown_usage), 'restarted_branches': moved})
    rt.FIT_RETRY = bool(cfg.get('fit_retry'))
    branches, failures, stopped = [], [], None
    try:
        client = rt.MeteredClient(led, sroot / 'raw_responses', http_cap=int(cfg['http_cap']))
        _run_arms(sroot, cfg, led, client, branches, failures, resumed=True)
        context.write_json(sroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches], 'resumed': True})
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': drs._safe_message(exc)})
        stopped = type(exc).__name__
    finally:
        context.write_json(sroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': True})
    drs.open_labels(sroot, cfg, branches, failures, led, resumed=True, withhold_reason=stopped and 'resumed stage stopped again (%s)' % stopped)


def stage_config(root: Path, stage: str, *, jobs, order, arms, labels_e: bool, parent_common=None) -> Path:
    P = paths(root)
    path = P['configs'] / ('%s.json' % stage)
    if path.exists():
        return path
    sc = stage_caps(root, stage)
    cfg = {'study': 'data_readiness_r2', 'status': 'FROZEN', 'stage': stage, 'output': str(P[stage]), 'jobs': list(jobs), 'order': {j: list(order[j]) for j in jobs},
           'arms': br.json_copy(arms), 'parent_common': parent_common or {}, 'labels_e': bool(labels_e), 'seeds': list(SEEDS), 'limits': LIMITS,
           'max_tool_corrections': drs.MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': drs.EVIDENCE_ROUNDTRIP, 'caps': sc['caps'], 'http_cap': sc['http_cap'],
           'ledger_path': str(P['ledger']), 'fit_retry': True, 'fit_attempts_at_stage_start': context.read_json(P['ledger'])['fit_attempts']}
    dsks.write_once(path, cfg)
    return path


def run_stage(root: Path, stage: str, cfg_path: Path) -> dict:
    P = paths(root)
    sroot = P[stage]
    if not sroot.exists():
        led = context.read_json(P['ledger'])
        remaining = TOTAL['max_wall_s'] - (time.time() - led['started_epoch'])
        if remaining <= 0:
            raise RuntimeError('package wall exhausted')
        P['logs'].mkdir(parents=True, exist_ok=True)
        print('STAGE_START', stage, flush=True)
        with (P['logs'] / ('%s.log' % stage)).open('a', encoding='utf-8') as log:
            p = subprocess.Popen([sys.executable, '-B', '-m', 'evaluation.main_protocol_p4.batch_research_data_readiness_r2', '--stage-worker', str(cfg_path)],
                                 cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)
                context.write_json(root / ('%s_supervisor_timeout.json' % stage), {'epoch': time.time(), 'pid': p.pid})
                rc = 124
        print('STAGE_EXIT', stage, rc, flush=True)
    return dsks.stage_status(sroot)


# ============================================================================= preflight (0 cost)
def test_eligibility() -> dict:
    """T rows only of RD02_T3/T4 vs the task table (4 decimals) and the exact job geometry."""
    import numpy as np
    out = {}
    for job, want in TEST_TABLE.items():
        js = rd.resolve_job(DOMAIN, job)
        t = js.t
        sl = rd.load_slice(DOMAIN, t - rd.TRAIN_SPAN, t)
        seg = sl.rows(t - rd.TRAIN_SPAN, t)
        lg = rd.legal_parents(seg)
        fin = np.isfinite(seg)
        per = [{'finite': int(fin[:, e].sum()), 'std': float(seg[fin[:, e], e].std()), 'legal_parents': int(lg.counts[e])} for e in range(seg.shape[1])]
        ok = all(p['finite'] >= rd.MIN_T_FINITE and p['std'] > 1e-6 and p['legal_parents'] >= rd.MIN_LEGAL_PARENTS for p in per)
        geom = {'T': list(js.train_range), 'C_A': [min(js.c_a), max(js.c_a) + rd.H], 'C_B': [min(js.c_b), max(js.c_b) + rd.H], 'E': [min(js.e), max(js.e) + rd.H]}
        row = {'t': t, 'missing_rate_pct': round(float(np.isnan(seg).mean() * 100), 4), 'min_legal_parents': int(lg.counts.min()), 'legal_parents_total': lg.n,
               'all_entities_eligible': ok, 'per_entity': per, 'geometry': geom, 'origins': {'c_a': list(js.c_a), 'c_b': list(js.c_b), 'e': list(js.e)},
               'rows_loaded': [sl.row_start, sl.row_end]}
        row['matches_task_table'] = (row['missing_rate_pct'] == want['missing_rate_pct'] and row['min_legal_parents'] == want['min_legal_parents']
                                     and all(geom[k] == want[k] for k in ('T', 'C_A', 'C_B', 'E')) and t == want['t'])
        if not ok or not row['matches_task_table']:
            raise RuntimeError('Test eligibility/table mismatch for %s: %s' % (job, {k: v for k, v in row.items() if k != 'per_entity'}))
        out[job] = row
    return out


def exposure_check() -> dict:
    last_row = max(TEST_TABLE[j]['E'][1] for j in TEST_JOBS)
    if last_row > SEALED_FROM_ROW:
        raise RuntimeError('Test rows reach the closed 2016+ region')
    return {'status': 'NO_SEALED_REGION_TOUCHED', 'dataset_exposure': rd.EXPOSURE, 'last_label_row_exclusive': last_row, 'closed_region_from_row': SEALED_FROM_ROW,
            'records_checked': ['docs/HARNESS_RESEARCH_DIRECTION_AND_PLAN.md §2.2 (2016+ boundary) and §2.3 (Beijing PM2.5 2013-03..2015-12 exposed development material)',
                                'docs/DEV_TRAIN3_CROSS_JOB_HARNESS_TASK_2026-09-11.md §4.2 (six stations, blocks from rows 3600 and 10800: exposed development)',
                                'AGENTS.md (Natural Final stays sealed; not this file region)'],
            'note': 'RD02_T3/T4 labels are rows [9360,10944) = 2014-03-26..2014-05-31 of the same hourly files; exposed development data, planned evaluation batches, not Natural Final'}


def verify_parent_reuse() -> dict:
    """Semantic / array checks that the reused parent dev branches and common baselines match this package's environment."""
    import glob
    import numpy as np
    h_old = parent_card()
    want_k = {dsk.NO_SKILL: br.json_copy(asdict(br.Knowledge())), H_OLD_ID: _knowledge(h_old)}
    want_bodies = {dsk.NO_SKILL: [], H_OLD_ID: [h_old.rendered_body]}
    requests = {}
    for st in ('select', 'target'):
        for f in sorted(glob.glob(str(PARENT / st / 'raw_responses' / '*_fast_request.json'))):
            q = context.read_json(f)
            requests.setdefault(q['unit'], []).append(f)
    stage_cfg = {st: context.read_json(PARENT / st / 'config.json') for st in ('select', 'target')}
    out = {'options': {}, 'common': {}, 'stage_configs': {}}
    for st, c in stage_cfg.items():
        row = {'seeds': c['seeds'] == list(SEEDS), 'limits': c['limits'] == LIMITS, 'max_tool_corrections': c['max_tool_corrections'] == drs.MAX_TOOL_CORRECTIONS,
               'evidence_roundtrip': c['evidence_roundtrip'] == drs.EVIDENCE_ROUNDTRIP, 'fit_retry': c['fit_retry'] is True}
        out['stage_configs'][st] = row
        if not all(row.values()):
            raise RuntimeError('parent %s configuration differs: %s' % (st, row))
    for job, rel in PARENT_COMMON.items():
        cdir = PARENT / rel / job
        cells = [context.read_json(p) for p in sorted((cdir / 'cells').glob('*.json'))]
        fresh = rd.open_job(DOMAIN, job, Path(os.environ.get('TEMP', '.')) / 'r2_preflight_scaler', 'material')   # recomputed from raw T; compared below
        with np.load(cdir / 'scaler.npz') as z:
            arrays = all(np.array_equal(z[k], v) for k, v in (('mean', fresh.scaler.mean), ('scale', fresh.scaler.scale), ('legal_ent', fresh.legal.ent), ('legal_k', fresh.legal.k)))
            binding = str(z['profile']) == rd.PROFILE and str(z['dataset']) == DOMAIN and int(z['t']) == fresh.job.t and [str(x) for x in z['roster']] == fresh.job.roster
        row = {'scaler_and_legal_parents_equal_recomputed': arrays, 'binding': binding,
               'cells': sorted((c['material_id'], c['model_seed']) for c in cells) == sorted((m, s) for m, _ in drs.BASELINES for s in SEEDS),
               'c_a_only': all(set(c['scores']) == {'c_a'} and c['scores']['c_a']['status'] == 'SCORABLE' for c in cells),
               'models_exist': all(Path(c['model_path']).exists() for c in cells),
               'no_label_files': not any((cdir / n).exists() for n in ('c_b_scores.json', 'e_frozen.json', 'e_scores.json', 'predictions_e')),
               'rows_read_evaluate': all(list(c['rows_read']) == list(fresh.job.evaluate_rows) for c in cells),
               'overview_equal_recomputed': context.read_json(cdir / 'overview.json') == br.json_copy(fresh.overview())}
        shutil.rmtree(Path(os.environ.get('TEMP', '.')) / 'r2_preflight_scaler', ignore_errors=True)
        out['common'][job] = row
        if not all(row.values()):
            raise RuntimeError('parent common %s not reusable: %s' % (job, row))
    for option, per in PARENT_DEV.items():
        for job, rel in per.items():
            b = PARENT / rel
            res = context.read_json(b / 'branch_result.json')
            reqs = [context.read_json(f) for f in requests.get(b.name, [])]
            pay = [json.loads(q['messages'][1]['content']) for q in reqs]
            first = pay[0] if pay else {}
            cells = {(c['material_id'], c['model_seed']): c for c in (context.read_json(p) for p in (b / job / 'cells').glob('*.json'))}
            plan = res['committed_plan_id']
            cdir = PARENT / PARENT_COMMON[job] / job
            with np.load(cdir / 'scaler.npz') as z0, np.load(b / job / 'scaler.npz') as z1:
                same_scaler = all(np.array_equal(z0[k], z1[k]) for k in ('mean', 'scale', 'legal_ent', 'legal_k', 'roster', 't'))
            row = {'complete_with_commit_and_c_b': drs._branch_complete_with_c_b(b, job),
                   'knowledge_equal': context.read_json(b / 'knowledge.json') == want_k[option],
                   'same_job_scaler_legal_parents_as_common': same_scaler,
                   'overview_equal_common': context.read_json(b / job / 'overview.json') == context.read_json(cdir / 'overview.json'),
                   'requests': len(reqs) == res['calls'] and len(reqs) > 0,
                   'fast_system_equal': all(q['messages'][0]['content'] == drs.FAST_SYSTEM for q in reqs),
                   'tool_contracts_equal': all(p['tool_contracts'] == drs.CONTRACTS for p in pay),
                   'requested_model_temperature': all(q['requested_model'] == MODEL['requested'] and q['temperature'] == 0 for q in reqs),
                   'returned_model': all(context.read_json(f.replace('_request.json', '_response.json'))['model'] == MODEL['returned_required'] for f in requests.get(b.name, [])),
                   'first_request_budget': {k: first.get('remaining', {}).get(k) for k in ('calls', 'tools', 'new_evaluations', 'tool_corrections')}
                   == {'calls': LIMITS['max_calls'], 'tools': LIMITS['max_tools'], 'new_evaluations': LIMITS['max_new_evaluations'], 'tool_corrections': drs.MAX_TOOL_CORRECTIONS},
                   'loaded_guidance_bodies': all([g['body'] for g in p['guidance']['loaded']] == want_bodies[option] for p in pay),
                   'committed_plan_c_b_three_seeds': all(cells.get((plan, s), {}).get('scores', {}).get('c_b', {}).get('status') == 'SCORABLE' for s in SEEDS)}
            out['options'].setdefault(option, {})[job] = {**row, 'branch': rel, 'committed_plan_id': plan}
            if not all(v for k, v in row.items()):
                raise RuntimeError('parent dev branch %s not reusable: %s' % (rel, row))
    out['note'] = ('The only public payload difference is remaining.seconds (the package wall shown to Fast), a budget display, not a tool, prompt, '
                   'limit or knowledge change. Reused readings are this round\'s development readings, not new independent runs.')
    return out


def preflight(root: Path) -> dict:
    import inspect
    from SelfEvolvingHarnessTS.operators import _provenance
    if (root / 'package_config.json').exists():
        return context.read_json(root / 'package_config.json')
    src = inspect.getsource(rt.MeteredClient)
    if not all(x in src for x in ("model='cpa-grok-4.6'", "'grok-4.6-build'", 'temperature=0', "base_url='http://127.0.0.1:8318/v1'")):
        raise RuntimeError('metered client no longer carries the frozen model identity')
    gc = drs.generic_control_record()
    if drs.generic_knowledge() != br.Knowledge(1, (br.Guidance('experiment_guidance', gc['body'], {'const': True}, ('generic:frozen',)),)):
        raise RuntimeError('generic control differs from the historical Generic')
    for rel, job, _, _ in EVIDENCE_BRANCHES:
        if not drs._branch_complete_with_c_b(PARENT / rel, job):
            raise RuntimeError('evidence branch missing: %s' % rel)
    h_old = parent_card()
    import torch
    fp = _provenance.dependency_fingerprint()
    parent_fp = context.read_json(PARENT / 'package_config.json')['dependency_fingerprint']
    if fp != parent_fp:
        raise RuntimeError('numerical dependency fingerprint differs from the parent package: %s vs %s' % (fp, parent_fp))
    interpreter = {'executable': sys.executable, 'python': sys.version.split()[0], 'torch': torch.__version__, 'dependency_fingerprint_equals_parent': True,
                   'note': 'same interpreter family as the parent package whose baselines and development readings are reused (task text names the conda project env; '
                           'that env carries other numpy/scipy/torch builds, so it would not be comparable with the reused parent fits)'}
    elig = test_eligibility()
    reuse = verify_parent_reuse()
    exposure = exposure_check()
    cfg = {'package': 'DEV-DATA-READINESS-DOMAIN-WORKFLOW-EVOLVE-R2', 'task': TASK, 'parent_package': str(PARENT), 'parent_read_only': True, 'exposure': rd.EXPOSURE,
           'frozen_epoch': time.time(), 'domain': DOMAIN, 'population': rd.RD02_ROSTER,
           'freeze_note': 'Frozen before the first paid call. This preflight read T rows of RD02_T3/T4 (eligibility) and of RD02_V1/T1 (scaler recomputation) only.',
           'interpreter': interpreter, 'test_eligibility': elig, 'exposure_check': exposure, 'parent_reuse_checks': reuse,
           'geometry': {'L': rd.L, 'H': rd.H, 'T': rd.TRAIN_SPAN, 'c_a': 't, t+48', 'c_b': 't+96, t+144', 'e': 't+192..t+336 step 48', 'legal_parent': 'X >= 32 finite and y fully finite',
                        'coverage_min': rd.COVERAGE_MIN},
           'h_old': h_old.to_json(), 'evidence_branches': [list(x) for x in EVIDENCE_BRANCHES], 'technical_records': [list(x) for x in TECHNICAL_RECORDS],
           'revise_system': REVISE_SYSTEM, 'revision_max_tokens': REVISION_MAX_TOKENS, 'evidence_max_request_bytes': EVIDENCE_MAX_BYTES,
           'dev_jobs': list(DEV_JOBS), 'dev_order': {j: list(o) for j, o in DEV_ORDER.items()}, 'parent_dev_readings': PARENT_DEV, 'parent_common': PARENT_COMMON,
           'test_jobs': list(TEST_JOBS), 'test_order': {j: list(o) for j, o in TEST_ORDER.items()},
           'test_random': {j: {'seeds': list(s), 'policies': [rd.random_policy(x) for x in s]} for j, s in TEST_RANDOM_SEEDS.items()},
           'seeds': list(SEEDS), 'limits': LIMITS, 'max_tool_corrections': drs.MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': drs.EVIDENCE_ROUNDTRIP,
           'fast_system': drs.FAST_SYSTEM, 'contracts': drs.CONTRACTS, 'generic_control': gc, 'model': MODEL,
           'total_caps': TOTAL, 'http_cap': HTTP_CAP, 'stage_allocations': ALLOC,
           'fit_budget_formula': 'dev 2 candidates x 2 jobs x 2 new plans x 3 seeds = 24 (parent baselines reused); test 2 x (6 common + 4 Fast x 6 + Random 6) = 72; +2 same-configuration retries = 98',
           'selection_rules': {'score': 'per option: per dev job three-seed mean C_B of the actual commit, then equal-weight mean over RD02_V1 and RD02_T1; complete on both jobs required',
                               'H_challenger': 'lowest among eligible C1/C2; tie C1', 'H_selected': 'lowest among NO_SKILL, H_old, H_challenger; ties NO_SKILL -> H_old -> H_challenger',
                               'incomplete': 'NO_SKILL or H_old not eligible -> SELECTION_INCOMPLETE, H_selected keeps H_old flagged as unfinished'},
           'metric': 'missing-aware normalized MSE per entity over observed origin x horizon cells, equal-weight entity macro; block NOT_SCORABLE if an entity < 25% coverage',
           'dependency_fingerprint': _provenance.dependency_fingerprint()}
    root.mkdir(parents=True, exist_ok=True)
    dsks.write_once(root / 'package_config.json', cfg)
    return cfg


# ============================================================================= Stage B: dev candidates, selection, freeze
def dev_candidates(root: Path) -> list:
    prop = context.read_json(paths(root)['revision'] / 'proposal.json')
    return prop.get('candidates', []) if prop.get('status') == 'PROPOSED' else []


def dev_stage(root: Path) -> dict:
    run = [c for c in dev_candidates(root) if c['dev_status'] == 'RUN']
    if not run:
        rec = paths(root)['dev'] / 'dev_skipped.json'
        if not rec.exists():
            dsks.write_once(rec, {'status': 'NO_NEW_CANDIDATE_TO_RUN', 'revision_status': context.read_json(paths(root)['revision'] / 'proposal.json')['status']})
        return {'status': 'SKIPPED'}
    by_id = {c['candidate_id']: c for c in run}
    order = {j: [c for c in DEV_ORDER[j] if c in by_id] for j in DEV_JOBS}
    arms = {j: {c: {'kind': 'fast', 'option': by_id[c]['skill']['skill_id'], 'knowledge': _knowledge(dsk.skill_from_json(by_id[c]['skill']))} for c in order[j]} for j in DEV_JOBS}
    cfgp = stage_config(root, 'dev', jobs=DEV_JOBS, order=order, arms=arms, labels_e=False, parent_common={j: str(PARENT / r) for j, r in PARENT_COMMON.items()})
    return run_stage(root, 'dev', cfgp)


def run_selection(options: dict, jobs, arm_dir, *, cand_ids, h_old_id: str, reference_ids=(), _read_run=None, _score=None) -> tuple:
    """options: {NO_SKILL: None, h_old_id: skill, cand skill ids...}; arm_dir(option_id, job) -> branch dir. Two phases (dsk.select):
    every option's actual commit is read before any C_B; eligibility needs COMPLETE + scorable C_B of the commit on every job.
    _read_run / _score: smoke-only stand-ins for fixed score tables."""
    def read_run(oid, knowledge, job):
        b = Path(arm_dir(oid, job))
        if not (b / 'branch_result.json').exists():
            return {'status': 'NOT_RUN', 'synthetic': False}
        got, want = context.read_json(b / 'knowledge.json'), br.json_copy(asdict(knowledge))
        if not (_same_policy(got, want) if oid in reference_ids else got == want):
            raise RuntimeError('selection branch %s ran other knowledge than option %s' % (b.name, oid))
        res = context.read_json(b / 'branch_result.json')
        return {'status': res['status'], 'committed_plan_id': res['committed_plan_id'], 'failure_kind': res['failure_kind'], 'synthetic': False, 'branch': str(b),
                'reference_of_parent_card': oid in reference_ids, 'cost': {'fast_calls': res['calls'], 'tool_calls': res['tool_calls'], 'new_evaluations': res['new_evaluations']}}
    sel = dsk.select(options, list(jobs), _read_run or read_run, block='c_b', score=_score or drs.c_b_scorer(arm_dir))
    elig = sel['eligible_scores']
    cands = [c for c in cand_ids if c in elig]
    challenger = min(cands, key=lambda c: (elig[c], cand_ids.index(c))) if cands else None
    dec = {'H_challenger': challenger, 'challenger_rule': 'lowest equal-weight dev C_B among eligible new candidates; tie C1',
           'eligible_new_candidates': cands, 'new_candidates_not_eligible': [c for c in cand_ids if c not in elig]}
    if dsk.NO_SKILL not in elig or h_old_id not in elig:
        dec.update(selection_status='SELECTION_INCOMPLETE', H_selected=h_old_id, note='a parent/no-card development reference is incomplete: H_old kept, not a passed selection')
    else:
        pool = [dsk.NO_SKILL, h_old_id] + ([challenger] if challenger else [])
        dec.update(selection_status='SELECTED', H_selected=min(pool, key=lambda o: (elig[o], pool.index(o))), pool=pool,
                   rule='lowest equal-weight dev C_B among NO_SKILL, H_old, H_challenger; ties in that order')
    return sel, dec


def select_and_freeze(root: Path) -> dict:
    P = paths(root)
    rec_path = P['freeze'] / 'frozen.json'
    if rec_path.exists():
        return context.read_json(rec_path)
    h_old = parent_card()
    cands = dev_candidates(root)
    options = {dsk.NO_SKILL: None, H_OLD_ID: h_old}
    dirs = {(o, j): PARENT / rel for o, per in PARENT_DEV.items() for j, rel in per.items()}
    cand_ids, refs, skills = [], [], {}
    for c in cands:
        s = dsk.skill_from_json(c['skill'])
        skills[s.skill_id] = (c, s)
        if c['dev_status'] == 'SEMANTIC_DUPLICATE_OF_C1':
            continue
        options[s.skill_id] = s
        cand_ids.append(s.skill_id)
        for j in DEV_JOBS:
            if c['dev_status'] == 'IDENTICAL_TO_H_OLD':
                dirs[(s.skill_id, j)] = dirs[(H_OLD_ID, j)]
                refs.append(s.skill_id)
            else:
                dirs[(s.skill_id, j)] = P['dev'] / ('%s_%s' % (j, c['candidate_id']))
    dev_status = dsks.stage_status(P['dev']) if (P['dev'] / 'execution_finished.json').exists() else {'status': 'SKIPPED' if (P['dev'] / 'dev_skipped.json').exists() else 'NOT_RUN'}
    sel, dec = run_selection(options, DEV_JOBS, lambda o, j: dirs[(o, j)], cand_ids=cand_ids, h_old_id=H_OLD_ID, reference_ids=tuple(set(refs)))
    challenger = skills[dec['H_challenger']] if dec['H_challenger'] else None
    selected = dec['H_selected']
    if selected == dsk.NO_SKILL:
        sel_skill, sel_arm = None, 'F0'
    elif selected == H_OLD_ID:
        sel_skill, sel_arm = h_old, 'F_old'
    else:
        sel_skill, sel_arm = skills[selected][1], 'F_new'
    ch_json = None
    if challenger:
        ch_json = {**challenger[0], 'skill': {**challenger[1].to_json(), 'status': 'CANDIDATE_TEST_ONLY'},
                   'role': 'H_challenger: best revision candidate of this round; tested in isolation, no deployment/promotion identity from that test'}
    identical_new = bool(challenger) and challenger[0]['dev_status'] == 'IDENTICAL_TO_H_OLD'
    arms = {}
    for job in TEST_JOBS:
        a = {'F0': {'kind': 'fast', 'option': dsk.NO_SKILL, 'knowledge': br.json_copy(asdict(br.Knowledge()))},
             'F_old': {'kind': 'fast', 'option': H_OLD_ID, 'knowledge': _knowledge(h_old)},
             'F_generic': {'kind': 'fast', 'option': 'GENERIC_CONTROL', 'knowledge': br.json_copy(asdict(drs.generic_knowledge()))},
             'Random': {'kind': 'random', 'policy_seeds': list(TEST_RANDOM_SEEDS[job])}}
        if not challenger:
            a['F_new'] = {'kind': 'absent', 'status': 'NO_NEW_CANDIDATE', 'note': 'no eligible revision candidate; not run, no substitute, no budget transfer'}
        elif identical_new:
            a['F_new'] = {'kind': 'reference', 'status': 'IDENTICAL_POLICY_REFERENCE', 'reference_arm': 'F_old', 'option': challenger[1].skill_id,
                          'note': 'candidate body and scope equal the parent card; the F_old arm is referenced, not rerun'}
        else:
            a['F_new'] = {'kind': 'fast', 'option': challenger[1].skill_id, 'knowledge': _knowledge(challenger[1])}
        arms[job] = a
    frozen = {'status': 'TEST_FROZEN', 'epoch': time.time(), 'frozen_before_first_test_fit': not P['test'].exists(),
              'revision_status': context.read_json(P['revision'] / 'proposal.json')['status'], 'dev_stage_status': dev_status.get('status'),
              'dev_selection': {k: sel[k] for k in ('block', 'dev_jobs', 'option_order', 'runs', 'eligible_scores', 'rule')}, 'decision': dec,
              'candidates': [{k: v for k, v in c.items() if k != 'skill'} | {'skill_id': c['skill']['skill_id']} for c in cands],
              'H_old': h_old.to_json(), 'H_challenger': ch_json,
              'H_selected': {'option': selected, 'arm': sel_arm, 'selection_status': dec['selection_status'],
                             'skill': ({**sel_skill.to_json(), 'status': 'FROZEN_SELECTED', 'source_stage': 'freeze' if sel_skill.skill_id != H_OLD_ID else sel_skill.source_stage}
                                       if sel_skill else None),
                             'delivery': 'the pre-frozen %s arm of each Test job is the system delivery; no extra Fast run' % sel_arm},
              'test': {'jobs': list(TEST_JOBS), 'order': {j: list(o) for j, o in TEST_ORDER.items()}, 'random_seeds': {j: list(s) for j, s in TEST_RANDOM_SEEDS.items()},
                       'arms': arms, 'geometry': TEST_TABLE},
              'budget': {'total': TOTAL, 'allocations': ALLOC, 'http_cap': HTTP_CAP}}
    if not frozen['frozen_before_first_test_fit']:
        raise RuntimeError('test directory exists before the freeze')
    dsks.write_once(rec_path, frozen)
    return frozen


def test_stage(root: Path) -> dict:
    fr = context.read_json(paths(root)['freeze'] / 'frozen.json')
    cfgp = stage_config(root, 'test', jobs=TEST_JOBS, order=TEST_ORDER, arms=fr['test']['arms'], labels_e=True)
    return run_stage(root, 'test', cfgp)


# ============================================================================= package driver
def package_run(root: Path = ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    status = {}

    def stop(where, st):
        status.update({'stopped_at': where, 'stage_status': st, 'epoch': time.time()})
        context.write_json(root / 'package_status.json', status)
        print('PACKAGE_STOP', where, st.get('status'), flush=True)
        return status
    sm = paths(root)['smoke'] / 'smoke.json'
    if not sm.exists() or context.read_json(sm)['status'] != 'PASS':
        raise RuntimeError('the new-risk smoke has not passed')
    preflight(root)
    prop = revision_stage(root)
    status['revision'] = prop['status']
    st = dev_stage(root)
    status['dev'] = st
    if st['status'] not in ('FINISHED', 'SKIPPED'):
        return stop('dev', st)
    fr = select_and_freeze(root)
    status['selection'] = fr['decision']
    st = test_stage(root)
    status['test'] = st
    if st['status'] != 'FINISHED':
        return stop('test', st)
    status['finished_epoch'] = time.time()
    context.write_json(root / 'package_status.json', status)
    package_result(root)
    print('PACKAGE_FINISHED', flush=True)
    return status


# ============================================================================= readout
def _arm_row(bdir: Path, job: str, tokens: dict) -> dict:
    r = context.read_json(bdir / 'branch_result.json')
    cells = drs._cells(bdir, job)
    plan = r['committed_plan_id']
    comp = context.read_json(bdir / job / 'materials' / (plan + '__compiled.json')) if plan else None
    idx = rd.load_index(bdir / job)
    beh = dsks._behavior(bdir)
    rows = drs._trace(bdir)
    fields_used = sorted({f for rr in rows if rr['event'] == 'tool_started' and rr.get('tool') == 'build_material'
                          for f in ((rr['arguments'].get('policy') or {}).get('observation_fields_used') or [])})
    evaluated = {m: {'c_a': drs._vec(cells, m, 'c_a'), 'c_b': drs._vec(cells, m, 'c_b'), 'e': drs._vec(cells, m, 'e')} for m in cells}
    return {'status': r['status'], 'failure_kind': r['failure_kind'], 'commit': plan, 'fast_calls': r['calls'], 'tool_calls': r['tool_calls'], 'new_evaluations': r['new_evaluations'],
            'e': drs._vec(cells, plan, 'e'), 'e_shadow': drs._vec(cells, plan, 'e_shadow'), 'c_a': drs._vec(cells, plan, 'c_a'), 'c_b': drs._vec(cells, plan, 'c_b'),
            'e_mae': drs._vec(cells, plan, 'e_mae'), 'material_spec': comp['material_spec'] if comp else None,
            'material_totals': _material_totals(idx[plan]['summary_path']) if plan else None, 'plans_in_branch': evaluated,
            'behavior': {**beh, 'observation_fields_used_in_builds': fields_used}, 'fast_tokens': tokens.get('fast:%s' % bdir.name)}


def _mean(v):
    return statistics.mean(v) if v else None


def package_result(root: Path = ROOT) -> dict:
    P = paths(root)
    fr = context.read_json(P['freeze'] / 'frozen.json') if (P['freeze'] / 'frozen.json').exists() else None
    res = {'package': 'DEV-DATA-READINESS-DOMAIN-WORKFLOW-EVOLVE-R2', 'exposure': rd.EXPOSURE, 'scope': 'RD02 only (12 PM2.5 stations), fixed Consumer; not cross-domain',
           'revision': None, 'dev': {}, 'freeze': fr, 'test': {}, 'summary': {}, 'costs': {}}
    if (P['revision'] / 'proposal.json').exists():
        prop = context.read_json(P['revision'] / 'proposal.json')
        res['revision'] = {k: v for k, v in prop.items() if k != 'raw_response'}
    if (P['dev'] / 'execution_finished.json').exists():
        res['dev_status'] = dsks.stage_status(P['dev'])
        tok = dsks._unit_tokens(P['dev'])
        for b in sorted(P['dev'].glob('RD02_*_C[12]')):
            job = '_'.join(b.name.split('_')[:2])
            if (b / 'branch_result.json').exists():
                res['dev'][b.name] = _arm_row_dev(b, job, tok)
    if (P['test'] / 'execution_finished.json').exists():
        res['test_status'] = dsks.stage_status(P['test'])
    tok = dsks._unit_tokens(P['test']) if P['test'].exists() else {}
    sel_arm = fr['H_selected']['arm'] if fr else None
    for job in TEST_JOBS:
        if not fr or not P['test'].exists():
            break
        row = {'arms': {}, 'H_selected_arm': sel_arm}
        ns = P['test'] / ('%s_not_scorable.json' % job)
        if ns.exists():
            row['status'] = 'C_A_NOT_SCORABLE'
            res['test'][job] = row
            continue
        for arm in TEST_ORDER[job]:
            b = P['test'] / ('%s_%s' % (job, arm))
            t = P['test'] / ('%s_%s_treatment.json' % (job, arm))
            if (b / 'branch_result.json').exists():
                row['arms'][arm] = _arm_row(b, job, tok)
            elif t.exists():
                row['arms'][arm] = context.read_json(t)
            else:
                row['arms'][arm] = {'status': 'NOT_RUN'}
        e = {}
        for arm, v in row['arms'].items():
            if v.get('kind') == 'reference':
                e[arm] = row['arms'][v['reference_arm']].get('e')
            else:
                e[arm] = v.get('e')
        some = next((P['test'] / ('%s_%s' % (job, a)) for a in TEST_ORDER[job] if (P['test'] / ('%s_%s' % (job, a)) / job / 'e_scores.json').exists()), None)
        if some:
            cells = drs._cells(some, job)
            row['Baseline_Linear'] = {k: drs._vec(cells, 'Baseline_Linear', k) for k in ('e', 'e_shadow', 'c_a', 'c_b', 'e_mae')}
            row['Fixed_Seasonal'] = {k: drs._vec(cells, 'Fixed_Seasonal', k) for k in ('e', 'e_shadow', 'c_a', 'c_b', 'e_mae')}
            anyb = next(iter(cells.values()))
            row['e_status'] = sorted({str(v['e_status']) for v in anyb.values()})
            row['e_min_coverage'] = min((v['e_min_coverage'] for v in anyb.values() if v['e_min_coverage'] is not None), default=None)
            cb = context.read_json(some / job / 'c_b_scores.json')['cells']
            row['c_b_status'] = sorted({v['c_b']['status'] for v in cb.values()})
            bl, fs = row['Baseline_Linear']['e'], row['Fixed_Seasonal']['e']
            hs = e.get(sel_arm)
            row['main'] = {'delta_skill_F0_minus_H_selected': drs._paired(e.get('F0'), hs), 'delta_round_F_old_minus_H_selected': drs._paired(e.get('F_old'), hs),
                           'delta_revision_F_old_minus_F_new': drs._paired(e.get('F_old'), e.get('F_new'))}
            row['H_selected_vs_others'] = {'F_generic_minus_H_selected': drs._paired(e.get('F_generic'), hs), 'Random_minus_H_selected': drs._paired(e.get('Random'), hs),
                                           'Baseline_Linear_minus_H_selected': drs._paired(bl, hs), 'Fixed_Seasonal_minus_H_selected': drs._paired(fs, hs)}
            row['F_new_vs_others'] = {'F0_minus_F_new': drs._paired(e.get('F0'), e.get('F_new')), 'F_generic_minus_F_new': drs._paired(e.get('F_generic'), e.get('F_new')),
                                      'Random_minus_F_new': drs._paired(e.get('Random'), e.get('F_new')), 'Baseline_Linear_minus_F_new': drs._paired(bl, e.get('F_new'))}
            row['pipeline_vs_baseline_linear'] = {a: drs._paired(bl, v) for a, v in {**e, 'Fixed_Seasonal': fs}.items()}
            row['shadow_vs_baseline_linear'] = {a: drs._paired(bl, row['arms'][a].get('e_shadow')) for a in row['arms'] if row['arms'][a].get('e_shadow')}
            row['shadow_vs_baseline_linear']['Fixed_Seasonal'] = drs._paired(bl, row['Fixed_Seasonal']['e_shadow'])
            row['e_mean'] = {**{a: _mean(v) for a, v in e.items()}, 'Baseline_Linear': _mean(bl), 'Fixed_Seasonal': _mean(fs)}
        res['test'][job] = row
    for key in ('delta_skill_F0_minus_H_selected', 'delta_round_F_old_minus_H_selected', 'delta_revision_F_old_minus_F_new'):
        per = {j: (res['test'].get(j, {}).get('main') or {}).get(key) for j in TEST_JOBS}
        vals = [v['mean'] for v in per.values() if v]
        res['summary'][key] = {'per_job_mean': {j: (v['mean'] if v else None) for j, v in per.items()},
                               'equal_weight_two_jobs': statistics.mean(vals) if len(vals) == 2 else None, 'scorable_jobs': len(vals),
                               'note': 'description only; seeds are Consumer repeats, jobs are the unit'}
    led = context.read_json(P['ledger']) if P['ledger'].exists() else {}
    res['costs'] = {'ledger': {k: v for k, v in led.items() if k != 'events'}, 'stages': context.read_json(P['stages']) if P['stages'].exists() else {},
                    'tokens_by_unit': {s: dsks._unit_tokens(P[s]) for s in ('revision', 'dev', 'test') if (P[s] / 'raw_responses').exists()},
                    'parent_sunk': parent_sunk_costs()}
    context.write_json(root / 'result.json', res)
    return res


def _arm_row_dev(bdir: Path, job: str, tokens: dict) -> dict:
    r = context.read_json(bdir / 'branch_result.json')
    cells = drs._cells(bdir, job)
    plan = r['committed_plan_id']
    return {'status': r['status'], 'failure_kind': r['failure_kind'], 'commit': plan, 'fast_calls': r['calls'], 'tool_calls': r['tool_calls'], 'new_evaluations': r['new_evaluations'],
            'c_a': drs._vec(cells, plan, 'c_a'), 'c_b': drs._vec(cells, plan, 'c_b'),
            'plans_in_branch': {m: {'c_a': drs._vec(cells, m, 'c_a'), 'c_b': drs._vec(cells, m, 'c_b')} for m in cells},
            'material_spec': context.read_json(bdir / job / 'materials' / (plan + '__compiled.json'))['material_spec'] if plan else None,
            'behavior': dsks._behavior(bdir), 'fast_tokens': tokens.get('fast:%s' % bdir.name)}


def parent_sunk_costs() -> dict:
    out = {}
    form = dsks._unit_tokens(PARENT / 'formation')
    out['formation_slow_RD02'] = form.get('slow_domain:RD02')
    for st in ('source', 'select', 'target'):
        tok = dsks._unit_tokens(PARENT / st)
        for rel, job, role, _ in EVIDENCE_BRANCHES:
            if rel.startswith(st + '/'):
                name = rel.split('/')[1]
                r = context.read_json(PARENT / rel / 'branch_result.json')
                out[rel] = {'role': role, 'fast_tokens': tok.get('fast:%s' % name), 'new_evaluations': r['new_evaluations'], 'new_plan_fits': 3 * r['new_evaluations'] if role != 'random_control' else 6}
    out['note'] = 'historical parent costs of the evidence and of the reused dev readings; not charged to this package ledger'
    return out


# ============================================================================= smoke (new risks; 0 real fits, 0 API, no label row of a planned Test job)
class _ScriptedSlow:
    def __init__(self, responses):
        self.responses, self.payloads = list(responses), []

    def __call__(self, payload):
        self.payloads.append(payload)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return copy.deepcopy(r)


def _fixture_candidate(cid: str, workflow: str, refs) -> dict:
    return {'candidate_id': cid, 'research_mode': 'mode %s' % cid, 'workflow': workflow, 'principles': None, 'observable_applicability': {'const': True},
            'applicability_summary': 'smoke', 'evidence_refs': list(refs)[:2], 'rationale': 'smoke', 'change_target': 'experiment',
            'process_evidence': 'smoke evidence', 'expected_behavior_change': 'smoke change'}


def smoke(out: Path) -> dict:
    import tempfile
    import numpy as np
    from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td = Path(td)
        # 1. old job mapping unchanged; new jobs exact; strings cannot pass through; CLI worker resolves the new jobs before any label read
        old_map = {'S1': 3360, 'S2': 4560, 'V1': 5760, 'T1': 6960, 'T2': 8160}
        old_ok = rd.JOB_T == old_map and all(rd.resolve_job(d, '%s_%s' % (d, s)).t == t for d in ('RD01', 'RD02') for s, t in old_map.items())
        new = {}
        for job, want in TEST_TABLE.items():
            js = rd.resolve_job(DOMAIN, job)
            new[job] = {'t': js.t, 'T': list(js.train_range), 'C_A': [min(js.c_a), max(js.c_a) + rd.H], 'C_B': [min(js.c_b), max(js.c_b) + rd.H], 'E': [min(js.e), max(js.e) + rd.H],
                        'c_a_origins': list(js.c_a), 'c_b_origins': list(js.c_b), 'e_origins': list(js.e), 'c_b_rows': list(js.c_b_rows), 'e_input_rows': list(js.e_input_rows),
                        'e_target_rows': list(js.e_target_rows)}
        exact = all(new[j]['t'] == w['t'] and all(new[j][k] == w[k] for k in ('T', 'C_A', 'C_B', 'E')) for j, w in TEST_TABLE.items()) and \
            new['RD02_T3']['c_a_origins'] == [9360, 9408] and new['RD02_T3']['c_b_origins'] == [9456, 9504] and new['RD02_T3']['e_origins'] == [9552, 9600, 9648, 9696] and \
            new['RD02_T4']['c_a_origins'] == [10560, 10608] and new['RD02_T4']['c_b_origins'] == [10656, 10704] and new['RD02_T4']['e_origins'] == [10752, 10800, 10848, 10896]
        rejected = {}
        for ds, job in (('RD01', 'RD01_T3'), ('RD02', 'RD02_T5'), ('RD02', 'RD02_T3 '), ('RD02', 'rd02_t3'), ('RD02', 'RD02_T3x'), ('RD01', 'RD02_T3'), ('RDX', 'RDX_T3')):
            try:
                rd.resolve_job(ds, job)
                rejected[job + '@' + ds] = False
            except ValueError:
                rejected[job + '@' + ds] = True
        text_ban = {}
        for s in ('prefer T3 plans', 'the T4 batch'):
            try:
                rd.validate_text(s, 'smoke')
                text_ban[s] = False
            except rd.PolicyError:
                text_ban[s] = True
        cli = {}
        for job in ('RD02_T3', 'RD02_T5'):
            pr = subprocess.run([sys.executable, '-B', '-m', rd.FIT_MODULE, '--stage', 'c_b', '--output', str(td / 'cli'), '--dataset', DOMAIN, '--job', job],
                                cwd=REPO, capture_output=True, text=True, timeout=300)
            cli[job] = {'returncode': pr.returncode, 'tail': pr.stderr.strip().splitlines()[-1][:200] if pr.stderr.strip() else ''}
        checks['1_job_mapping_old_unchanged_new_exact'] = {
            'ok': old_ok and exact and all(rejected.values()) and all(text_ban.values()) and cli['RD02_T3']['returncode'] != 0
            and 'c_b prerequisite missing before loading any labels' in cli['RD02_T3']['tail'] and 'ValueError' in cli['RD02_T5']['tail'],
            'detail': {'old_mapping_unchanged': old_ok, 'new_jobs': new, 'rejected': rejected, 'text_ban': text_ban, 'cli': cli}}

        # 2. census: whitelist reads only (no E / report / result / claims file, no data row), no E keys or E numbers, no Test content, size bound
        seen = {'read_json': [], 'read_text': [], 'np_load': [], 'load_slice': 0}
        orig_rj, orig_rt, orig_ls, orig_np = context.read_json, Path.read_text, rd.load_slice, np.load

        def rj(p, *a, **k):
            seen['read_json'].append(str(p))
            return orig_rj(p, *a, **k)

        def rtx(self, *a, **k):
            seen['read_text'].append(str(self))
            return orig_rt(self, *a, **k)

        def ls(*a, **k):
            seen['load_slice'] += 1
            return orig_ls(*a, **k)

        def npl(p, *a, **k):
            seen['np_load'].append(str(p))
            return orig_np(p, *a, **k)
        context.read_json, Path.read_text, rd.load_slice, np.load = rj, rtx, ls, npl
        try:
            ev = revision_evidence()
        finally:
            context.read_json, Path.read_text, rd.load_slice, np.load = orig_rj, orig_rt, orig_ls, orig_np
        text = json.dumps(ev, ensure_ascii=False)
        bad_reads = [p for p in seen['read_json'] + seen['read_text'] + seen['np_load'] if FORBIDDEN_READ.search(p)]
        keys = set()

        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    keys.add(k)
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(ev)
        e_vals = set()
        for f in PARENT.glob('*/RD02_*/RD02_*/e_scores.json'):
            for c in orig_rj(f)['cells'].values():
                for blk in ('e', 'e_shadow_baseline_inputs'):
                    v = (c.get(blk) or {}).get('normalized_mse_macro')
                    if v is not None:
                        e_vals.add(repr(v))
        e_hits = sorted(v for v in e_vals if v in text)
        nbytes = request_bytes(revision_payload(ev))
        checks['2_census_no_e_no_test_content'] = {
            'ok': not bad_reads and seen['load_slice'] == 0 and not seen['np_load'] and not (keys & {'e', 'e_shadow', 'e_scores', 'e_shadow_baseline_inputs', 'e_frozen', 'predictions_e'})
            and not re.search(r'RD02_T[34]|\b(9360|10560)\b', text) and not e_hits and nbytes <= EVIDENCE_MAX_BYTES and len(ev['trajectories']) == len(EVIDENCE_BRANCHES),
            'detail': {'files_read': len(seen['read_json']) + len(seen['read_text']), 'forbidden_reads': bad_reads[:5], 'data_slices_loaded': seen['load_slice'],
                       'e_like_keys': sorted(keys & {'e', 'e_shadow', 'e_scores', 'e_shadow_baseline_inputs'}), 'e_value_hits': e_hits[:5],
                       'request_bytes': nbytes, 'bound': EVIDENCE_MAX_BYTES, 'legal_refs': len(ev['legal_evidence_refs'])}}

        # 3. revision contract: KEEP, duplicates/identical classification, one contract correction, faults are not KEEP
        h_old = parent_card()
        refs = ev['legal_evidence_refs']
        keep = revise(ev, _ScriptedSlow([{'decision': 'KEEP', 'rationale': 'insufficient'}]), h_old=h_old)
        c1 = _fixture_candidate('C1', 'Inspect gaps first; compare one completion plan with the linear program; stop when C_A differences are within seed noise.', refs)
        c2 = dict(c1, candidate_id='C2', research_mode='other mode')
        c_same_old = _fixture_candidate('C1', h_old.workflow, refs) | {'principles': h_old.principles}
        dup = revise(ev, _ScriptedSlow([{'decision': 'PROPOSE', 'candidates': [c1, c2]}]), h_old=h_old)
        ident = revise(ev, _ScriptedSlow([{'decision': 'PROPOSE', 'candidates': [c_same_old]}]), h_old=h_old)
        bad = dict(c1)
        bad.pop('change_target')
        corrected = _ScriptedSlow([{'decision': 'PROPOSE', 'candidates': [bad]}, {'decision': 'PROPOSE', 'candidates': [c1]}])
        corr = revise(ev, corrected, h_old=h_old)
        twice_bad = revise(ev, _ScriptedSlow([{'decision': 'PROPOSE', 'candidates': [bad]}, {'decision': 'PROPOSE', 'candidates': [dict(c1, candidate_id='W1')]}]), h_old=h_old)
        fault = revise(ev, _ScriptedSlow([llm.TransportFault('x')]), h_old=h_old)
        leak = revise(ev, _ScriptedSlow([{'decision': 'PROPOSE', 'candidates': [dict(c1, workflow='Use the T3 answer table.')]}] * 2), h_old=h_old)
        checks['3_revision_contract_keep_duplicate_identical_correction'] = {
            'ok': keep['status'] == 'KEEP' and not keep['candidates'] and dup['status'] == 'PROPOSED' and [c['dev_status'] for c in dup['candidates']] == ['RUN', 'SEMANTIC_DUPLICATE_OF_C1']
            and [c['dev_status'] for c in ident['candidates']] == ['IDENTICAL_TO_H_OLD'] and corr['status'] == 'PROPOSED' and len(corrected.payloads) == 2
            and 'correction' in corrected.payloads[1] and twice_bad['status'] == 'REVISE_PARSE_OR_VALIDATION_FAILED' and fault['status'] == 'REVISE_CALL_FAILED'
            and leak['status'] == 'REVISE_PARSE_OR_VALIDATION_FAILED' and dup['candidates'][0]['skill']['skill_id'] == 'RD02-C1-r2'
            and dup['candidates'][0]['skill']['derived_from'] == H_OLD_ID and dup['candidates'][0]['skill']['status'] == 'CANDIDATE_TEST_ONLY',
            'detail': {'keep': keep['status'], 'dup': [c['dev_status'] for c in dup['candidates']], 'identical': [c['dev_status'] for c in ident['candidates']],
                       'corrected_attempts': corr['attempts'], 'twice_bad': twice_bad['attempts'], 'fault': fault['status'], 'leak_attempts': leak['attempts']}}

        # 4-5. synthetic package: parent dev readings, R2 dev stage on copied parent commons, two-phase selection of actual commits, Test stage labels order
        drs._register_synthetic()
        counter = {}
        orig_fit = rt.fit
        rt.fit = drs._fake_fit_factory(counter)
        try:
            caps = {'max_fit_attempts': 999, 'max_llm_requests': 999, 'max_llm_tokens': 10 ** 9, 'max_wall_s': 3600, 'max_retries': 0}
            ledger = td / 'pkg' / 'budget.json'
            (td / 'pkg').mkdir()
            rt.RuntimeLedger(ledger, **caps)
            inproc = lambda p, j, name, led: {'c_b': rd.open_c_b, 'freeze_e': rd.freeze_e, 'score_e': rd.score_e}[name](p, drs.ds_of(j), j)
            fx_refs = ['fixture:smoke']
            mk = lambda sid, wf, rev=1: dsk.make_skill(skill_id=sid, domain_id='RDX', revision=rev, workflow=wf, principles=None, applicability_summary='smoke', compatibility_note='smoke',
                                                      observable_applicability={'const': True}, evidence_refs=fx_refs, legal_evidence_refs=fx_refs, source_stage='fixture',
                                                      status='SMOKE_FIXTURE', allowed_features=drs.ALLOWED_FEATURES, text_check=drs.skill_text_check)
            old = mk('RDX-W1-r1', 'Inspect gaps; compare the two common programs; stop when differences are within seed noise.')
            ca = mk('RDX-C1-r2', 'Build one gated completion plan before committing; compare it with the linear program.', 2)
            cb = mk('RDX-C2-r2', 'Inspect material diagnostics of one anomaly plan; evaluate it only if observed changes are small.', 2)
            client = drs._ScriptedClient()
            dev_jobs = ('RDX_V1', 'RDX_T1')
            base_cfg = {'labels_e': False, 'caps': caps, 'http_cap': 999, 'ledger_path': str(ledger), 'fit_retry': False, 'parent_common': {}}
            pcfg = {**base_cfg, 'stage': 'parent', 'output': str(td / 'pkg' / 'parent'), 'jobs': list(dev_jobs), 'order': {j: ['no_skill', 'old'] for j in dev_jobs},
                    'arms': {j: {'no_skill': {'kind': 'fast', 'knowledge': br.json_copy(asdict(br.Knowledge()))}, 'old': {'kind': 'fast', 'knowledge': _knowledge(old)}} for j in dev_jobs}}
            context.write_json(td / 'pkg' / 'parent.json', pcfg)
            stage_worker(td / 'pkg' / 'parent.json', client=client, runner=inproc)
            proot = td / 'pkg' / 'parent'
            fits_before_dev = counter.get('fits', 0)
            dcfg = {**base_cfg, 'stage': 'dev', 'output': str(td / 'pkg' / 'dev'), 'jobs': list(dev_jobs), 'order': {'RDX_V1': ['C1', 'C2'], 'RDX_T1': ['C2', 'C1']},
                    'arms': {j: {'C1': {'kind': 'fast', 'knowledge': _knowledge(ca)}, 'C2': {'kind': 'fast', 'knowledge': _knowledge(cb)}} for j in dev_jobs},
                    'parent_common': {j: str(proot / ('%s_common' % j)) for j in dev_jobs}}
            context.write_json(td / 'pkg' / 'dev.json', dcfg)
            stage_worker(td / 'pkg' / 'dev.json', client=client, runner=inproc)
            droot = td / 'pkg' / 'dev'
            dev_fits = counter.get('fits', 0) - fits_before_dev
            commits = [droot / ('%s_%s' % (j, c)) / j / 'commit.json' for j in dev_jobs for c in ('C1', 'C2')]
            cbs = [droot / ('%s_%s' % (j, c)) / j / 'c_b_scores.json' for j in dev_jobs for c in ('C1', 'C2')]
            e_files = [p for p in droot.rglob('*') if p.name in ('e_frozen.json', 'e_scores.json', 'all_e_predictions_frozen.json', 'predictions_e')]
            dirs = {(dsk.NO_SKILL, j): proot / ('%s_no_skill' % j) for j in dev_jobs}
            dirs.update({(old.skill_id, j): proot / ('%s_old' % j) for j in dev_jobs})
            dirs.update({(s.skill_id, j): droot / ('%s_%s' % (j, c)) for s, c in ((ca, 'C1'), (cb, 'C2')) for j in dev_jobs})
            opts = {dsk.NO_SKILL: None, old.skill_id: old, ca.skill_id: ca, cb.skill_id: cb}
            sel, dec = run_selection(opts, dev_jobs, lambda o, j: dirs[(o, j)], cand_ids=[ca.skill_id, cb.skill_id], h_old_id=old.skill_id)
            # actual commit, not the candidate's best plan: lower every other plan's C_B in a copy and re-select
            oracle_root = td / 'pkg' / 'oracle'
            shutil.copytree(droot, oracle_root)
            for bdir in oracle_root.glob('RDX_*_C[12]'):
                job = '_'.join(bdir.name.split('_')[:2])
                plan = context.read_json(bdir / 'branch_result.json')['committed_plan_id']
                for cpath in (bdir / job / 'cells').glob('*.json'):
                    cell = context.read_json(cpath)
                    if cell['material_id'] != plan and (cell['scores'].get('c_b') or {}).get('status') == 'SCORABLE':
                        cell['scores']['c_b']['normalized_mse_macro'] = -1.0
                        context.write_json(cpath, cell)
            odirs = {k: (oracle_root / v.name if v.parent == droot else v) for k, v in dirs.items()}
            osel, odec = run_selection(opts, dev_jobs, lambda o, j: odirs[(o, j)], cand_ids=[ca.skill_id, cb.skill_id], h_old_id=old.skill_id)
            # selection rules on fixed score tables (ties, incomplete, no candidate)
            rules = {}
            for label, elig, cand_ids in (('tie_no_skill_first', {dsk.NO_SKILL: 1.0, 'O': 1.0, 'A': 1.0}, ['A']),
                                          ('tie_c1_first', {dsk.NO_SKILL: 2.0, 'O': 2.0, 'A': 1.0, 'B': 1.0}, ['A', 'B']),
                                          ('old_over_challenger_tie', {dsk.NO_SKILL: 2.0, 'O': 1.0, 'A': 1.0}, ['A']),
                                          ('incomplete', {'O': 1.0, 'A': 0.5}, ['A']), ('keep_no_candidates', {dsk.NO_SKILL: 2.0, 'O': 1.0}, [])):
                fake = lambda o, k, j, elig=elig: {'status': 'COMPLETE' if o in elig else 'INCOMPLETE', 'committed_plan_id': 'x', 'synthetic': True}
                fscore = lambda o, j, r, elig=elig: {'loss_by_seed': [elig[o]] * 3}
                _, d = run_selection({dsk.NO_SKILL: None, 'O': old, **{c: ca for c in cand_ids}}, ['J'], None, cand_ids=cand_ids, h_old_id='O',
                                     _read_run=fake, _score=fscore)
                rules[label] = (d['selection_status'], d['H_challenger'], d['H_selected'])
            want_rules = {'tie_no_skill_first': ('SELECTED', 'A', dsk.NO_SKILL), 'tie_c1_first': ('SELECTED', 'A', 'A'), 'old_over_challenger_tie': ('SELECTED', 'A', 'O'),
                          'incomplete': ('SELECTION_INCOMPLETE', 'A', 'O'), 'keep_no_candidates': ('SELECTED', None, 'O')}
            checks['4_dev_reuses_parent_commons_selects_actual_commits'] = {
                'ok': dev_fits == 4 * 3 and all(p.exists() for p in commits + cbs) and max(p.stat().st_mtime for p in commits) <= min(p.stat().st_mtime for p in cbs)
                and not e_files and (droot / 'labels_c_b_only.json').exists() and set(sel['eligible_scores']) == set(opts)
                and odec == dec and osel['eligible_scores'] == sel['eligible_scores'] and rules == want_rules
                and all(r['system'] == drs.FAST_SYSTEM for r in client.requests),
                'detail': {'dev_new_fits': dev_fits, 'expected': '4 branches x 1 scripted plan x 3 seeds = 12; 0 baseline fits', 'eligible_scores': sel['eligible_scores'], 'decision': dec, 'oracle_decision_equal': odec == dec,
                           'rules': rules, 'e_files_in_dev': [str(p) for p in e_files]}}
            # Test stage: F0, F_old, F_new (real on one job, identical reference on the other), F_generic, Random; labels after every commit
            tjobs = ('RDX_T1', 'RDX_T2')
            tarms = {}
            for j in tjobs:
                tarms[j] = {'F0': {'kind': 'fast', 'knowledge': br.json_copy(asdict(br.Knowledge()))}, 'F_old': {'kind': 'fast', 'knowledge': _knowledge(old)},
                            'F_new': ({'kind': 'fast', 'knowledge': _knowledge(ca)} if j == 'RDX_T1' else
                                      {'kind': 'reference', 'status': 'IDENTICAL_POLICY_REFERENCE', 'reference_arm': 'F_old'}),
                            'F_generic': {'kind': 'fast', 'knowledge': br.json_copy(asdict(drs.generic_knowledge()))},
                            'Random': {'kind': 'random', 'policy_seeds': [2026091711, 2026091712]}}
            tcfg = {**base_cfg, 'stage': 'test', 'output': str(td / 'pkg' / 'test'), 'jobs': list(tjobs), 'labels_e': True, 'parent_common': {},
                    'order': {'RDX_T1': list(TEST_ORDER['RD02_T3']), 'RDX_T2': list(TEST_ORDER['RD02_T4'])}, 'arms': tarms}
            context.write_json(td / 'pkg' / 'test.json', tcfg)
            nreq = len(client.requests)
            stage_worker(td / 'pkg' / 'test.json', client=client, runner=inproc)
            troot = td / 'pkg' / 'test'
            ran = [troot / ('%s_%s' % (j, a)) for j in tjobs for a in TEST_ORDER['RD02_T3'] if (troot / ('%s_%s' % (j, a))).is_dir()]
            tcommits = [b / b.name[:6] / 'commit.json' for b in ran]
            tcb = [b / b.name[:6] / 'c_b_scores.json' for b in ran]
            tef = [b / b.name[:6] / 'e_frozen.json' for b in ran]
            tes = [b / b.name[:6] / 'e_scores.json' for b in ran]
            barrier = troot / 'all_e_predictions_frozen.json'
            first = {}
            for r_ in client.requests[nreq:]:
                first.setdefault(r_['unit'], [x['body'] for x in r_['payload']['guidance']['loaded']])
            want_first = {'RDX_T1_F0': [], 'RDX_T1_F_old': [old.rendered_body], 'RDX_T1_F_new': [ca.rendered_body], 'RDX_T1_F_generic': [drs.generic_control_record()['body']],
                          'RDX_T2_F0': [], 'RDX_T2_F_old': [old.rendered_body], 'RDX_T2_F_generic': [drs.generic_control_record()['body']]}
            order_seen = [u for u in dict.fromkeys(r_['unit'] for r_ in client.requests[nreq:])]
            ref_t = troot / 'RDX_T2_F_new_treatment.json'
            checks['5_test_arms_order_reference_labels_after_all_commits'] = {
                'ok': len(ran) == 9 and all(p.exists() for p in tcommits + tcb + tef + tes) and not (troot / 'RDX_T2_F_new').exists()
                and ref_t.exists() and context.read_json(ref_t)['status'] == 'IDENTICAL_POLICY_REFERENCE'
                and max(p.stat().st_mtime for p in tcommits) <= min(p.stat().st_mtime for p in tcb) and max(p.stat().st_mtime for p in tcb) <= min(p.stat().st_mtime for p in tef)
                and max(p.stat().st_mtime for p in tef) <= barrier.stat().st_mtime <= min(p.stat().st_mtime for p in tes)
                and first == want_first and order_seen == ['RDX_T1_F0', 'RDX_T1_F_old', 'RDX_T1_F_new', 'RDX_T1_F_generic', 'RDX_T2_F_generic', 'RDX_T2_F_old', 'RDX_T2_F0']
                and bool(rtp.labels_boundary_violations(troot)) and context.read_json(troot / 'RDX_T1_Random' / 'branch_result.json')['status'] == 'COMPLETE',
                'detail': {'branches_run': [b.name for b in ran], 'fast_unit_order': order_seen, 'first_request_bodies': {u: len(v) for u, v in first.items()},
                           'resume_refused_after_labels': rtp.labels_boundary_violations(troot)[:2]}}
        finally:
            rt.fit = orig_fit
            rd.DATASETS.pop('RDX', None)
    res = {'status': 'PASS' if all(c['ok'] for c in checks.values()) else 'FAIL', 'checks': checks, 'real_fits': 0, 'api_calls': 0, 'label_rows_of_planned_test_jobs': 0}
    res['existing_controls'] = []
    for cmd in (['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/functional/test_batch_research.py'],
                ['-c', 'import json,sys; from pathlib import Path; from evaluation.main_protocol_p4 import batch_research_data_readiness as m; r=m.smoke(Path(sys.argv[1])); '
                       'sys.exit(0 if r["status"]=="PASS" else 1)', str(out / 'readiness_control')]):
        t0 = time.time()
        pr = subprocess.run([sys.executable, '-B'] + cmd, cwd=REPO, capture_output=True, text=True, timeout=3600)
        res['existing_controls'].append({'command': ' '.join(cmd[:4])[:120], 'returncode': pr.returncode, 'seconds': round(time.time() - t0, 1), 'tail': (pr.stdout + pr.stderr)[-400:]})
        if pr.returncode:
            res['status'] = 'FAIL'
    context.write_json(out / 'smoke.json', res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--stage-worker', type=Path)
    ap.add_argument('--resume-stage', choices=['dev', 'test'])
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--restart', default='')
    ap.add_argument('--continue-package', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--evidence', action='store_true', help='0-cost census only (printed size; nothing written)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--root', type=Path, default=ROOT)
    a = ap.parse_args()
    root = a.root.resolve()
    if a.stage_worker:
        stage_worker(a.stage_worker.resolve())
    elif a.resume_stage:
        resume_stage(root, a.resume_stage, accept_unknown_usage=a.accept_unknown_usage, restart=[x for x in a.restart.split(',') if x])
        if a.continue_package:
            package_run(root)
    elif a.run:
        package_run(root)
    elif a.result:
        package_result(root)
    elif a.preflight:
        cfg = preflight(root)
        print(json.dumps({'test_eligibility': {j: {k: v for k, v in r.items() if k != 'per_entity'} for j, r in cfg['test_eligibility'].items()},
                          'exposure': cfg['exposure_check']['status']}, ensure_ascii=False)[:3000])
    elif a.evidence:
        ev = revision_evidence()
        print('EVIDENCE_BYTES', request_bytes(revision_payload(ev)), 'REFS', len(ev['legal_evidence_refs']),
              {k: len(json.dumps(v, ensure_ascii=False)) for k, v in ev.items()})
    elif a.smoke:
        res = smoke(root / 'smoke')
        print('R2_SMOKE', res['status'], {k: v['ok'] for k, v in res['checks'].items()}, [c['returncode'] for c in res['existing_controls']], flush=True)
        if res['status'] != 'PASS':
            raise SystemExit(1)
    else:
        ap.error('choose --run, --stage-worker, --resume-stage, --result, --preflight, --evidence or --smoke')


if __name__ == '__main__':
    from evaluation.main_protocol_p4 import batch_research_data_readiness_r2 as _canonical
    _canonical.main()
