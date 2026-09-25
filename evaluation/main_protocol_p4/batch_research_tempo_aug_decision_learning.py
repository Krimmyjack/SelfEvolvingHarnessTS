"""DEV-TEMPO-AUG-DECISION-LEARNING (docs/DEV_TEMPO_AUG_DECISION_LEARNING_TASK_2026-09-19.md): the same historical evidence under two
Slow learning contracts -- O (the parent's SLOW_A1 / SLOW_C trajectory-consolidation contract) and N (a decision-hypothesis contract:
observable condition -> actual decision -> delayed outcome -> strong simple default, with per-rule support / counterexample handling) --
each formed twice independently (O1, N1, N2, O2), every candidate really practised on L1/L2, one parent per line revised once with
Principles allowed, parents + children + no-Skill selected on L3/L4 with the fixed public programs inside the adoption set (H_O / H_N),
then an EXPOSED development replay on L5-L7 (F0 / F_O / F_N / RandomSearch_B4 / Menu_CA / Fixed_dev / P_source / four references).

Everything numerical, the seven-tool Fast, the adapter, the physical cache, the label workers, the resume boundary and the budget ledger
are the parent modules' (W = batch_research_tempo_aug_workflow_skill, L = batch_research_tempo_aug_workflow_learning_loop). New here: the
N-contract prompts, the N input organisation (decision_cases + materials / actions + historical_hypotheses linked by card_ref) derived
deterministically from the O census, the zero-fit evidence summary (P_source, public C_A -> C_B concordance), the four-call formation, the
per-line parent / revision / selection with fixed strategies in the adoption set, the replay stage and the readout.

  --smoke | --preflight | --wiring | --run | --resume-stage STAGE [--accept-unknown-usage] | --result
  worker: the parent module W runs every numerical / Fast stage worker (--stage-worker CFG) with this package's config.
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
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_learning_loop as L

REPO = W.REPO
ROOT = REPO / '_scratch' / 'dev_tempo_aug_decision_learning'
PARENT_LL = L.ROOT                                                 # read-only: L1-L7 physical caches, wiring cache, the parent ledger
PARENT_W = W.ROOT                                                  # read-only: the ten historical trajectories and the two old cards
TASK = 'docs/DEV_TEMPO_AUG_DECISION_LEARNING_TASK_2026-09-19.md'
MODULE = 'evaluation.main_protocol_p4.batch_research_tempo_aug_decision_learning'
PACKAGE = 'DEV-TEMPO-AUG-DECISION-LEARNING'
DATASET, DOMAIN, SEEDS = W.DATASET, W.DOMAIN, W.SEEDS
WIRING_SEEDS, WIRING_JOB = W.WIRING_SEEDS, W.WIRING_JOB
JOB_INDEX, JOB_T = L.JOB_INDEX, L.JOB_T
HIST_SOURCE_JOBS, HIST_SELECT_JOBS = L.HIST_SOURCE_JOBS, L.HIST_SELECT_JOBS
INITIAL_JOBS = tuple(HIST_SOURCE_JOBS) + tuple(HIST_SELECT_JOBS)
PRACTICE_JOBS, SELECT_JOBS, TEST_JOBS = L.PRACTICE_JOBS, L.SELECT_JOBS, L.TEST_JOBS
TEST_ORDER = {'RD02_L5': ('f0', 'f_o', 'f_n', 'random'), 'RD02_L6': ('f_n', 'random', 'f0', 'f_o'), 'RD02_L7': ('f_o', 'f0', 'random', 'f_n')}
PUBLIC, PUBLIC_STEPS = W.PUBLIC, W.PUBLIC_STEPS
LIMITS, FAST_TOKEN_CAP, MAX_OUTPUT_TOKENS, MAX_TOOL_CORRECTIONS, EVIDENCE_ROUNDTRIP = W.LIMITS, W.FAST_TOKEN_CAP, W.MAX_OUTPUT_TOKENS, W.MAX_TOOL_CORRECTIONS, W.EVIDENCE_ROUNDTRIP
BODY_LIMIT = L.BODY_LIMIT                                          # 6000-character opt-in (Workflow + Principles); the old default 1200 untouched
MAX_CANDIDATES_PER_CALL = 2
LINES = ('O', 'N')
FORMATION_CALLS = (('O1', 'O'), ('N1', 'N'), ('N2', 'N'), ('O2', 'O'))    # task 6.A order; no call reads another call's output
TOTAL = {'max_fit_attempts': 549, 'max_llm_requests': 572, 'max_llm_tokens': 13_200_000, 'max_wall_s': 8 * 3600, 'max_retries': 6}
STAGE_RETRIES = 2
HTTP_CAP = 576
ALLOC = {'wiring': {'fits': 3, 'requests': 0, 'tokens': 0},
         'formation_a': {'fits': 0, 'requests': 8, 'tokens': 2_400_000},
         'practice': {'fits': 216, 'requests': 256, 'tokens': 3_200_000},
         'revision': {'fits': 0, 'requests': 4, 'tokens': 2_400_000},
         'select': {'fits': 144, 'requests': 160, 'tokens': 2_800_000},
         'test': {'fits': 180, 'requests': 144, 'tokens': 2_400_000}}
NUMERIC_WALL_S = W.NUMERIC_WALL_S
ALLOWED_FEATURES = W.ALLOWED_FEATURES
MODEL = W.MODEL
T975_DF2 = W.T975_DF2
TOL = 1e-12
FAST_SYSTEM = L.FAST_SYSTEM                                        # unchanged for every arm (task 4): the parent's FAST_SYSTEM + permission sentence
EXPOSURE_TAG = 'EXPOSED_DEVELOPMENT_REPLAY'

# ----------------------------------------------------------------------------- contracts
SLOW_O_FORM = L.SLOW_A1                                            # contract O, both formation calls (never the A1-reading SLOW_A2)
SLOW_O_REV = L.SLOW_C
FEEDBACK_ROLES, NO_IDS = L.FEEDBACK_ROLES, L.NO_IDS
N_CORE = ('Your goal is to learn a reusable research Workflow that improves the delayed forecasting utility of Fast\'s actual committed material '
          'under the fixed budget. Compare the research policy with the source-selected fixed default and the other public references. Prediction '
          'quality is primary; costs are reported separately. Saving experiments alone is not evidence of a quality improvement. '
          'Historical cards are hypotheses being evaluated, not instructions for you. No recommendation or prohibition inherits authority merely '
          'because a past card contained it. Evidence requirements apply equally to retained and new rules. Legal grouping and multi-step plans '
          'remain available; neither is mandatory. '
          'Distinguish the C_A evidence available to Fast at the decision from the C_B evidence used to evaluate whether that decision should be '
          'reused. Acknowledge limited time coverage. Agreement across training seeds is not evidence of agreement across future time periods. '
          'Propose up to two alternative Workflows, or KEEP. Explain each alternative\'s decision hypothesis relative to the simple default: what '
          'observable information could justify a different construction, experiment allocation, or commit; what outcome would contradict that '
          'hypothesis; and what to do when the available information cannot distinguish the alternatives. '
          'For every reusable rule, including any retained rule, identify its supporting observations, delayed outcomes, and counterexamples. If '
          'there is no supporting evidence, mark it as an exploratory hypothesis, not an established principle. A counterexample need not force '
          'deletion, but explain why the rule is retained, narrowed, changed into a test, or left unresolved. Do not handle it merely by appending '
          'a generic warning while claiming the rule has been validated. '
          'Do not invent unmeasured causal explanations or performance. Do not impose a universal 3/3-seed gate, a universal trust/distrust of C_A, '
          'or a preferred primitive unless the supplied evidence supports the scoped recommendation. A short, uniform or stopping Workflow is '
          'legitimate; so is conditional construction. Material complexity or diversity is not an objective.')
N_RULE_SCHEMA = ('{"rule":"<the concrete rule as it stands in the Workflow text>","observable_condition":"<a condition Fast can legally obtain from T '
                 'or its own C_A; if the information is missing, what to check>","change_vs_default":"<where the construction, experiment allocation or '
                 'commit may differ from the default>","support_refs":["exact evidence refs" (may be empty)],"counter_refs":["exact evidence refs" (may be '
                 'empty)],"evidence_status":"supported_in_these_cases" | "exploratory" | "unresolved","counterexample_handling":"<why the rule is '
                 'retained, narrowed, turned into a test, dropped or left unresolved, and its limits>","predicted_behavior_and_failure":"<the Fast '
                 'behaviour expected next time and which result would refute it>"}')
N_HYPOTHESIS_SCHEMA = ('{"vs_default":"<what observable information could justify a construction, experiment allocation or commit different from the '
                       'simple default>","contradicting_outcome":"<which outcome would contradict this hypothesis>","when_indistinguishable":"<what to do '
                       'when the available information cannot distinguish the alternatives>"}')
N_INPUT_DESCRIPTION = ('The facts are organised as: (1) public_contract_and_default: field definitions, tools, Consumer, budget, the four public '
                       'references and the source-selected fixed default with its zero-fit evidence summary; (2) per batch, decision_cases: the '
                       'information visible to Fast, its actual decisions, the numeric consequences, the difference against the default and the other '
                       'materials it evaluated; (3) the complete material tables and action sequences; (4) historical_hypotheses: the earlier cards some '
                       'trajectories ran with, given once each and linked from the trajectories by card_ref. Nothing outside these facts exists: no E '
                       'value, no later batch, no other learning call\'s output.')
SLOW_N_FORM = ('You are the offline Slow of a batch training-data augmentation research Harness. You receive the deterministic facts of ten completed '
               'research trajectories from six earlier batches of ONE neutral domain (four ran with no card; six ran on two later batches with either '
               'no card or one of two earlier candidate cards). ' + N_INPUT_DESCRIPTION + ' ' + FEEDBACK_ROLES + ' The Fast you write for has a '
               'fixed budget per batch (4 new complete plans, 16 calls, 24 tool calls) and may commit any evaluated material including the public '
               'references; it may not claim a tool is unavailable. ' + N_CORE + ' ' + NO_IDS +
               ' Output exact JSON, no markdown: {"decision":"KEEP","rationale":"...","evidence_review":{"decision_rules":[' + N_RULE_SCHEMA +
               ', ...]}} OR {"decision":"PROPOSE","candidates":[{"candidate_id":"W1","research_mode":"<short label>","workflow":"...","principles":null,'
               '"observable_applicability":{"const":true},"applicability_summary":"<=400 characters","evidence_refs":["exact refs from the supplied '
               'legal ranges"],"decision_hypothesis":' + N_HYPOTHESIS_SCHEMA + ',"rationale":"...","evidence_review":{"decision_rules":[' + N_RULE_SCHEMA +
               ', ...]}}]}. candidate_id W1 and W2. principles must be null in this formation. Every reusable rule of the Workflow appears as one '
               'decision_rules entry; a rule without support_refs cannot be supported_in_these_cases. The Workflow text renders to at most 6000 '
               'characters. The whole response must fit within 12000 output tokens: keep every entry concise. KEEP is valid and is not resampled.')
SLOW_N_REV = ('You are the offline Slow revising ONE parent Workflow after its real use. You receive: the parent card (text, decision hypothesis, '
              'decision rules), the initial facts it was formed from (condensed: every action, plan, C_A/C_B number, commit and failure kept), the '
              'practice facts of ALL candidate cards of BOTH learning contracts on two new batches, organised as decision_cases (the information '
              'visible to Fast, its actual decisions, per-seed C_A, the commit and reason, post-commit per-seed C_B of every evaluated material, the '
              'difference against the default, the best evaluated material of each branch, costs and failures), the complete material tables and '
              'action sequences, the candidate cards given once each and linked by card_ref, and the source-selected fixed default with the public '
              'references. ' + FEEDBACK_ROLES + ' ' + N_CORE + ' Separate the directly observable facts -- advice not executed; executed without '
              'benefit; a worse choice among the materials actually evaluated; the better material not covered by the candidates -- from '
              'explanations. Training randomness, time change and material mechanism are hypotheses to be tested, never assigned as the cause from a '
              'single sign reversal. Relative to the parent you may change ONE main decision mechanism, and you may rewrite the related flow and the '
              'Principles that serve it (not limited to appending a gate at the end); Principles state evidence-supported preferences, applicability '
              'conditions, counterexamples and uncertainty bounds, never generic caution. Every added or retained principle must handle its evidence '
              'and counterexamples. KEEP is a complete, legitimate result. State the scenario in which the change is expected to trigger; an '
              'untriggered change is not a validated revision. ' + NO_IDS +
              ' Output exact JSON, no markdown: {"decision":"KEEP","rationale":"...","evidence_review":{"decision_rules":[' + N_RULE_SCHEMA + ', ...]}} '
              'OR {"decision":"REVISE","child":{"workflow":"<full child Workflow text>","principles":"<Principles text>" or null,"changed_mechanism":'
              '"<the one decision mechanism changed and how>","expected_trigger_scenario":"<the observable situation in which the child decides '
              'differently from the parent>","research_mode":"<short label>","applicability_summary":"<=400 characters","evidence_refs":["exact refs '
              'from the supplied legal ranges"],"decision_hypothesis":' + N_HYPOTHESIS_SCHEMA + ',"rationale":"...","evidence_review":{"decision_rules":['
              + N_RULE_SCHEMA + ', ...]}}}. Workflow plus Principles render to at most 6000 characters. The whole response must fit within 12000 output '
              'tokens: keep every entry concise. KEEP is valid, is not resampled and is recorded as an alias of the parent.')

COMPRESSION_RULES = {
    'inherits': 'the parent learning-loop census (tiers 3 / skeleton) and its branch-qualified evidence refs; see _scratch/dev_tempo_aug_workflow_learning_loop/evidence/compression_rules.json',
    'this_package': ['one common tier for both contracts, chosen before the first paid call on the SERIALISED sizes of all four formation payloads: the least-compressed tier '
                     'for which every remaining call of the stage fits the frozen client reservation 2 x (bytes + 2048 + 12000) with an 8192-byte correction margin, '
                     'assuming each earlier call of the stage consumes at most its own conservative upper (bytes + 2048 + 12000 tokens)',
                     'contract N input = a deterministic reorganisation of the contract O census: the same materials tables, the same compressed trajectory rows, the same '
                     'commits / costs / failures; the loaded card texts move out of the trajectories into historical_hypotheses (once each) and are linked by card_ref; '
                     'decision_cases are derived only from those rows and tables (no new measurement)',
                     'the zero-fit evidence summary (P_source, public C_A -> C_B concordance) is computed once from the six initial batches and given to both contracts'],
    'never_dropped': 'actions, plans, C_A/C_B numbers, commits, failures, conflicts, loaded card texts; no contract loses a case or number the other keeps',
}


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def paths(root: Path) -> dict:
    return L.paths(root)


def _mean(v):
    return statistics.fmean(v) if v else None


def _r(x, nd=6):
    return round(x, nd) if isinstance(x, float) else x


# ============================================================================= evidence: contract O census (parent code), zero-fit summary, contract N reorganisation
def o_census(root: Path, tier, *, purpose='Initial formation input (contract O): ten completed research trajectories of six batches of one neutral domain') -> dict:
    return L.census(root, L.historical_branches(), {str(PARENT_W / 'source'): 'hist_source', str(PARENT_W / 'select'): 'hist_select'}, tier, purpose=purpose)


def _pair_stats(a: list, b: list) -> dict:
    """b - a per seed with SE and the n=3 two-sided 95% t interval (df=2); positive = b higher loss."""
    d = [y - x for x, y in zip(a, b)]
    m = statistics.fmean(d)
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else None
    return {'diff_by_seed': [_r(x) for x in d], 'mean': _r(m), 'se': _r(se) if se is not None else None,
            't95': [_r(m - T975_DF2 * se), _r(m + T975_DF2 * se)] if se is not None and len(d) == 3 else None, 'sign_of_mean': int(m > TOL) - int(m < -TOL)}


def evidence_summary(cen: dict, jobs=INITIAL_JOBS) -> dict:
    """Task 5.1: J_source(p) over the initial six jobs and P_source (tolerance 1e-12, fixed public order on ties); zero-fit pairwise public C_A -> C_B
    same-direction rate, rankings and loss differences per job, then equal-weight over jobs; per-seed paired differences with SE and the df=2 t interval.
    No threshold is fitted; nothing here is an E oracle or a commit rule."""
    per_job, ratios = {}, {m: [] for m in PUBLIC}
    for j in jobs:
        mats = cen['jobs'][j]['materials']
        pub = {m: next(v for v in mats.values() if v.get('public_id') == m) for m in PUBLIC}
        none_cb = pub['None']['c_b_by_seed']
        row = {'c_a_mean': {}, 'c_b_mean': {}, 'c_b_ratio_to_none': {}, 'c_a_by_seed': {}, 'c_b_by_seed': {}}
        for m in PUBLIC:
            ca, cb = pub[m].get('c_a_by_seed'), pub[m].get('c_b_by_seed')
            row['c_a_by_seed'][m], row['c_b_by_seed'][m] = ca, cb
            row['c_a_mean'][m], row['c_b_mean'][m] = _r(_mean(ca)) if ca else None, _r(_mean(cb)) if cb else None
            row['c_b_ratio_to_none'][m] = _r(_mean(cb) / _mean(none_cb)) if cb and none_cb and _mean(none_cb) > 0 else None
            if row['c_b_ratio_to_none'][m] is not None:
                ratios[m].append(row['c_b_ratio_to_none'][m])
        row['rank_c_a'] = sorted([m for m in PUBLIC if row['c_a_mean'][m] is not None], key=lambda m: (row['c_a_mean'][m], PUBLIC.index(m)))
        row['rank_c_b'] = sorted([m for m in PUBLIC if row['c_b_mean'][m] is not None], key=lambda m: (row['c_b_mean'][m], PUBLIC.index(m)))
        pairs, agree, ties = {}, 0, 0
        for i, a in enumerate(PUBLIC):
            for b in PUBLIC[i + 1:]:
                if not (row['c_a_by_seed'][a] and row['c_a_by_seed'][b] and row['c_b_by_seed'][a] and row['c_b_by_seed'][b]):
                    pairs['%s|%s' % (a, b)] = {'status': 'MISSING'}
                    continue
                pa, pb = _pair_stats(row['c_a_by_seed'][a], row['c_a_by_seed'][b]), _pair_stats(row['c_b_by_seed'][a], row['c_b_by_seed'][b])
                same = pa['sign_of_mean'] == pb['sign_of_mean'] and pa['sign_of_mean'] != 0
                tie = pa['sign_of_mean'] == 0 or pb['sign_of_mean'] == 0
                pairs['%s|%s' % (a, b)] = {'c_a_b_minus_a': pa, 'c_b_b_minus_a': pb, 'same_direction': same, 'tie': tie}
                agree += same
                ties += tie
        n_pairs = sum(1 for p in pairs.values() if p.get('status') != 'MISSING')
        row['pairs'] = pairs
        row['same_direction_rate'] = {'agree': agree, 'ties': ties, 'of_pairs': n_pairs, 'rate_excluding_ties': _r(agree / (n_pairs - ties)) if n_pairs - ties else None}
        per_job[j] = row
    J = {m: _r(_mean(v)) for m, v in ratios.items() if len(v) == len(jobs)}
    p_source = min(J, key=lambda m: (J[m] if J[m] - min(J.values()) > TOL else min(J.values()), PUBLIC.index(m))) if J else None
    rates = [per_job[j]['same_direction_rate']['rate_excluding_ties'] for j in jobs if per_job[j]['same_direction_rate']['rate_excluding_ties'] is not None]
    return {'definition': {'J_source': 'J_source(p) = mean_j[ mean_s C_B(p,j,s) / mean_s C_B(None,j,s) ] over the six initial batches; lower is better; ties <= 1e-12 by the fixed order None, FixedMixup, P_AmpResample, P_NoMixRecipe',
                           'P_source': 'the public reference with the lowest J_source: the initial development default and an evidence summary, not an E oracle and not a mandatory commit rule; not rewritten by Practice / Select',
                           'concordance': 'per job, for each of the six public pairs: sign of mean_s(C_A(b) - C_A(a)) vs sign of mean_s(C_B(b) - C_B(a)); same_direction counts only non-tied pairs; ties listed separately; equal-weight mean over jobs; no threshold is fitted from these six jobs',
                           'interval': 'per-seed paired differences with SE and the n=3 two-sided 95%% t interval (df=2, t=%.6f); 2SE is not that interval' % T975_DF2},
            'jobs': list(jobs), 'J_source': J, 'P_source': p_source, 'per_job': per_job,
            'pooled': {'mean_same_direction_rate_excluding_ties': _r(_mean(rates)) if rates else None, 'jobs_with_rate': len(rates),
                       'agree_total': sum(per_job[j]['same_direction_rate']['agree'] for j in jobs), 'ties_total': sum(per_job[j]['same_direction_rate']['ties'] for j in jobs),
                       'pairs_total': sum(per_job[j]['same_direction_rate']['of_pairs'] for j in jobs)},
            'computed_local': now(), 'fits': 0, 'llm': 0}


def _card_body_ref(cards: dict, body: str, known: dict | None, branch_ref: str) -> str:
    known = known or {}
    ref_ = next((k for k, v in known.items() if v == body), None)
    if ref_ is None:
        ref_ = next((k for k, v in cards.items() if v['text'] == body), None)
    if ref_ is None:
        ref_ = 'H%d' % (1 + sum(1 for k in cards if k.startswith('H') and k[1:].isdigit()))
    cards.setdefault(ref_, {'text': body, 'loaded_by': [], 'characters': len(body)})
    if branch_ref not in cards[ref_]['loaded_by']:
        cards[ref_]['loaded_by'].append(branch_ref)
    return ref_


def _actions_of(traj: list) -> list:
    """The Fast decisions in order, taken from the fast_response rows (verbatim arguments) -- identical facts to the trajectory."""
    out = []
    for r in traj:
        if r['event'] == 'fast_response' and isinstance(r.get('response'), dict):
            for a in r['response'].get('actions') or []:
                if isinstance(a, dict):
                    out.append({'ref': r.get('ref'), 'call': r.get('number'), 'tool': a.get('tool'), 'arguments': a.get('arguments')})
    return out


def decision_case(b: dict, mats: dict, p_source: str | None, *, card_ref) -> dict:
    """Observable information -> actual decision -> numeric consequence -> difference against the default -> other evaluated materials.
    Derived only from the branch record and the job materials table (same facts as contract O); missing values stay missing."""
    traj = b['trajectory']
    actions = _actions_of(traj)
    p2p = b['plan_to_physical']
    evaluated = sorted({ph for ph in p2p.values() if mats.get(ph, {}).get('c_a_by_seed')})
    committed = b.get('committed_physical_id')
    rows = {}
    for ph in evaluated:
        m = mats[ph]
        rows[ph] = {'plan_ids': [p for p, x in p2p.items() if x == ph], 'public_id': m.get('public_id'), 'c_a_by_seed': m.get('c_a_by_seed'), 'c_b_by_seed': m.get('c_b_by_seed'),
                    'c_a_mean': _r(_mean(m['c_a_by_seed'])) if m.get('c_a_by_seed') else None, 'c_b_mean': _r(_mean(m['c_b_by_seed'])) if m.get('c_b_by_seed') else None,
                    'c_b_status': m.get('c_b_status')}
    none_ = next((mats[ph] for ph in mats if mats[ph].get('public_id') == 'None'), {})
    default = next((mats[ph] for ph in mats if mats[ph].get('public_id') == p_source), {}) if p_source else {}
    cm = mats.get(committed, {}) if committed else {}
    with_cb = {ph: r for ph, r in rows.items() if r['c_b_by_seed']}
    best_cb = min(with_cb, key=lambda ph: (with_cb[ph]['c_b_mean'], ph)) if with_cb else None
    argmin_ca = min(rows, key=lambda ph: (rows[ph]['c_a_mean'], ph)) if rows else None
    regret = (_mean(cm['c_b_by_seed']) - rows[best_cb]['c_b_mean']) if best_cb and cm.get('c_b_by_seed') else None
    cons = {'committed_physical_id': committed, 'committed_plan_id': b.get('committed_plan_id'), 'committed_c_a_by_seed': cm.get('c_a_by_seed'), 'committed_c_b_by_seed': cm.get('c_b_by_seed'),
            'committed_c_a_mean': _r(_mean(cm['c_a_by_seed'])) if cm.get('c_a_by_seed') else None, 'committed_c_b_mean': _r(_mean(cm['c_b_by_seed'])) if cm.get('c_b_by_seed') else None,
            'vs_none': {'c_a_committed_minus_none': _pair_stats(none_['c_a_by_seed'], cm['c_a_by_seed']) if cm.get('c_a_by_seed') and none_.get('c_a_by_seed') else None,
                        'c_b_committed_minus_none': _pair_stats(none_['c_b_by_seed'], cm['c_b_by_seed']) if cm.get('c_b_by_seed') and none_.get('c_b_by_seed') else None},
            'vs_default': {'default_public_id': p_source,
                           'c_a_committed_minus_default': _pair_stats(default['c_a_by_seed'], cm['c_a_by_seed']) if cm.get('c_a_by_seed') and default.get('c_a_by_seed') else None,
                           'c_b_committed_minus_default': _pair_stats(default['c_b_by_seed'], cm['c_b_by_seed']) if cm.get('c_b_by_seed') and default.get('c_b_by_seed') else None},
            'lowest_c_b_among_evaluated': {'physical_id': best_cb, 'plan_ids': rows[best_cb]['plan_ids'], 'c_b_mean': rows[best_cb]['c_b_mean'], 'c_a_mean': rows[best_cb]['c_a_mean']} if best_cb else None,
            'committed_is_lowest_c_b_evaluated': bool(best_cb and best_cb == committed), 'c_a_argmin_among_evaluated': argmin_ca,
            'committed_is_c_a_argmin': bool(argmin_ca and argmin_ca == committed),
            'c_b_regret_vs_lowest_evaluated': _r(regret) if regret is not None else None}
    visible = {'batch_information': 'jobs[<job>].overview (all N entity rows of T-only observations and the batch summary; identical for every branch of the batch)',
               'card_ref': card_ref,
               'observations_requested': [a for a in actions if a['tool'] in ('overview', 'inspect_data', 'inspect_material', 'compare')],
               'c_a_available_at_decision': {ph: {'plan_ids': r['plan_ids'], 'c_a_by_seed': r['c_a_by_seed']} for ph, r in rows.items()}}
    decisions = {'builds': [a for a in actions if a['tool'] == 'build_material'], 'evaluates': [a['arguments'].get('plan_id') for a in actions if a['tool'] == 'evaluate' and isinstance(a.get('arguments'), dict)],
                 'compares': [[a['arguments'].get('a'), a['arguments'].get('b')] for a in actions if a['tool'] == 'compare' and isinstance(a.get('arguments'), dict)],
                 'commit': next(({'plan_id': a['arguments'].get('plan_id'), 'reason': a['arguments'].get('reason'), 'ref': a['ref']} for a in actions if a['tool'] == 'commit' and isinstance(a.get('arguments'), dict)), None),
                 'commit_recorded': {'plan_id': b.get('committed_plan_id'), 'physical_id': committed, 'reason': b.get('commit_reason')},
                 'rejections': [r['ref'] for r in traj if r['event'] == 'tool_rejected'], 'status': b['status'], 'failure_kind': b.get('failure_kind'), 'reason': b.get('reason'),
                 'cost': b['cost'], 'unused_new_evaluation_slots': (LIMITS['max_new_evaluations'] - b['cost']['new_evaluations']) if b['cost'].get('new_evaluations') is not None else None}
    return {'branch_ref': b['ref_prefix'], 'job_ref': b['job_ref'], 'visible_information': visible, 'actual_decisions': decisions, 'numeric_consequences': cons,
            'other_evaluated_materials': {ph: r for ph, r in rows.items() if ph != committed}}


def reorganize_for_n(cen: dict, summary: dict, *, known_cards: dict | None = None, card_table: str = 'historical_hypotheses', purpose: str) -> dict:
    """Contract N input from the contract O census: same materials tables, same compressed trajectory rows and costs; loaded card texts moved into
    one table (once each, card_ref links); decision_cases derived per branch. known_cards: {card_ref: body} for this package's own candidates."""
    cards, jobs_n = {}, {}
    p_source = summary.get('P_source')
    for j in cen['jobs_in_order']:
        J = cen['jobs'][j]
        cases, actions = [], []
        for b in J['branches']:
            refs = [_card_body_ref(cards, body, known_cards, b['ref_prefix']) for body in (b.get('knowledge_loaded') or [])]
            card_ref = refs[0] if len(refs) == 1 else (refs or None)
            traj = copy.deepcopy(b['trajectory'])
            for r in traj:
                g = r.get('guidance')
                if isinstance(g, dict) and isinstance(g.get('loaded'), list):
                    g['loaded'] = [{**{k: v for k, v in e.items() if k != 'body'}, 'card_ref': _card_body_ref(cards, e['body'], known_cards, b['ref_prefix'])} if isinstance(e, dict) and 'body' in e else e for e in g['loaded']]
            bb = {**b, 'trajectory': traj}
            cases.append(decision_case(bb, J['materials'], p_source, card_ref=card_ref))
            actions.append({'branch_ref': b['ref_prefix'], 'card_ref': card_ref, 'status': b['status'], 'failure_kind': b.get('failure_kind'), 'reason': b.get('reason'),
                            'committed_plan_id': b.get('committed_plan_id'), 'committed_physical_id': b.get('committed_physical_id'), 'commit_reason': b.get('commit_reason'),
                            'plan_to_physical': b['plan_to_physical'], 'plan_specs': b['plan_specs'], 'cost': b['cost'], 'action_sequence': traj})
        jobs_n[j] = {'overview': J['overview'], 'decision_cases': cases, 'materials': J['materials'], 'branches': actions}
    pub = cen['public_semantics']
    return {'census_status': cen['census_status'], 'domain_id': cen['domain_id'], 'purpose': purpose, 'profile': cen['profile'], 'organisation': 'contract_N_decision_cases',
            'public_contract_and_default': {**pub, 'feedback_roles': cen['feedback_roles'], 'source_selected_default': {'P_source': p_source, 'J_source': summary['J_source'], 'definition': summary['definition']},
                                            'zero_fit_evidence_summary': {k: summary[k] for k in ('jobs', 'per_job', 'pooled')}},
            'jobs_in_order': cen['jobs_in_order'], 'jobs': jobs_n, 'not_run': cen['not_run'], card_table: cards,
            'legal_evidence_refs': cen['legal_evidence_refs'], 'compression': cen['compression'],
            'field_semantics': {'decision_cases': 'visible_information = what Fast could see (batch overview, its own observations, the C_A of what it had evaluated); actual_decisions = the tool actions it took; '
                                                  'numeric_consequences = per-seed C_A / C_B of the committed material, its paired difference against None and against the default (positive = committed worse), '
                                                  'the lowest-C_B material among those this branch evaluated and the C_B regret against it; other_evaluated_materials = every other material with its numbers',
                                'card_ref': 'the card loaded in that branch, given once in %s; a branch without card_ref ran the common no-card start' % card_table,
                                'materials': 'physical materials of the batch keyed by physical id: assignment, C_A and C_B per seed / origin; plan ids of each branch map to physical ids'}}


def facts_of(payload: dict, contract: str) -> dict:
    """Canonical scientific facts of one formation input: per job materials, per branch identity / commit / cost / failure / card text / actions."""
    if contract == 'O':
        cen = payload
        cards = {}
        def card_text(b):
            return list(b.get('knowledge_loaded') or [])
        def traj_of(b):
            rows = copy.deepcopy(b['trajectory'])
            for r in rows:
                g = r.get('guidance')
                if isinstance(g, dict) and isinstance(g.get('loaded'), list):
                    g['loaded'] = [{**{k: v for k, v in e.items() if k != 'body'}, 'card_text': e['body']} if isinstance(e, dict) and 'body' in e else e for e in g['loaded']]
            return rows
    else:
        cen = payload
        cards = payload.get('historical_hypotheses') or payload.get('candidate_cards') or {}
        def card_text(b):
            ref_ = b.get('card_ref')
            refs = ref_ if isinstance(ref_, list) else ([ref_] if ref_ else [])
            return [cards[x]['text'] for x in refs]
        def traj_of(b):
            rows = copy.deepcopy(b['action_sequence'])
            for r in rows:
                g = r.get('guidance')
                if isinstance(g, dict) and isinstance(g.get('loaded'), list):
                    g['loaded'] = [{**{k: v for k, v in e.items() if k != 'card_ref'}, 'card_text': cards[e['card_ref']]['text']} if isinstance(e, dict) and 'card_ref' in e else e for e in g['loaded']]
            return rows
    out = {'jobs': {}}
    for j in cen['jobs_in_order']:
        J = cen['jobs'][j]
        out['jobs'][j] = {'overview': J['overview'], 'materials': J['materials'], 'branches': {}}
        for b in J['branches']:
            out['jobs'][j]['branches'][b.get('ref_prefix') or b.get('branch_ref')] = {'status': b['status'], 'failure_kind': b.get('failure_kind'), 'reason': b.get('reason'), 'committed_plan_id': b.get('committed_plan_id'),
                                                            'committed_physical_id': b.get('committed_physical_id'), 'commit_reason': b.get('commit_reason'), 'plan_to_physical': b['plan_to_physical'],
                                                            'plan_specs': b['plan_specs'], 'cost': b['cost'], 'cards': card_text(b), 'actions': traj_of(b)}
    out['legal_evidence_refs'] = sorted(cen['legal_evidence_refs'])
    return out


def facts_equivalent(o_payload: dict, n_payload: dict) -> dict:
    fo, fn = facts_of(o_payload, 'O'), facts_of(n_payload, 'N')
    diffs = []
    for j in set(fo['jobs']) | set(fn['jobs']):
        a, b = fo['jobs'].get(j), fn['jobs'].get(j)
        if a is None or b is None:
            diffs.append('job %s missing on one side' % j)
            continue
        for k in ('overview', 'materials'):
            if a[k] != b[k]:
                diffs.append('%s.%s differs' % (j, k))
        for br_ in set(a['branches']) | set(b['branches']):
            x, y = a['branches'].get(br_), b['branches'].get(br_)
            if x is None or y is None:
                diffs.append('%s branch missing on one side' % br_)
                continue
            for k in x:
                if x[k] != y.get(k):
                    diffs.append('%s.%s differs' % (br_, k))
    if fo['legal_evidence_refs'] != fn['legal_evidence_refs']:
        diffs.append('legal refs differ')
    n_cards = n_payload.get('historical_hypotheses') or n_payload.get('candidate_cards') or {}
    o_texts = sorted({t for J in fo['jobs'].values() for b in J['branches'].values() for t in b['cards']})
    return {'equivalent': not diffs, 'differences': diffs, 'n_branches': sum(len(J['branches']) for J in fo['jobs'].values()),
            'cards_once_each': sorted({v['text'] for v in n_cards.values()}) == o_texts and len(n_cards) == len(o_texts),
            'card_refs_traceable': all(any(lb in J['branches'] for J in fo['jobs'].values()) for v in n_cards.values() for lb in v['loaded_by'])}


def payload_bytes(system: str, payload: dict) -> int:
    return L.payload_bytes(system, payload)


def reservation(nbytes: int) -> int:
    return L.reservation(nbytes)


def upper_tokens(nbytes: int) -> int:
    return nbytes + 2048 + MAX_OUTPUT_TOKENS


def plan_check(led, sc: dict, sizes_in_order: list, *, correction_margin: int = 8192) -> dict:
    """Task 8: every remaining call of the stage must fit the frozen client's reservation with a correction margin, assuming each earlier call
    consumed at most its own conservative upper (bytes + 2048 + max_output tokens). Sizes are the actual serialised payloads in call order."""
    used0 = led.s['llm_tokens_in'] + led.s['llm_tokens_out'] + led.s.get('token_reserved_failed_upper', 0)
    cap = sc['caps']['max_llm_tokens']
    rows, used, ok = [], used0, True
    for i, nb in enumerate(sizes_in_order):
        need = reservation(nb + correction_margin)
        fits = used + need <= cap
        rows.append({'call_index': i, 'bytes': nb, 'assumed_used_before': used, 'reservation_with_correction': need, 'fits': fits})
        ok = ok and fits
        used += upper_tokens(nb)
    return {'stage_cap': cap, 'used_at_check': used0, 'calls': rows, 'fits': ok}


def check_reservation(led, sc, nbytes, **kw):
    return L.check_reservation(led, sc, nbytes, **kw)


# ============================================================================= budget / stage plumbing (this package's caps; the parent's workers)
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


def _require_package_root(root: Path) -> None:
    # INCIDENT_2026-09-19: a smoke check once started a real stage worker in a temporary directory; no stage may run outside the repo's _scratch tree.
    if not str(Path(root).resolve()).startswith(str((REPO / '_scratch').resolve())):
        raise RuntimeError('stage refused: root %s is outside %s' % (root, REPO / '_scratch'))


def stage_config(root: Path, stage: str, *, jobs, order, knowledge, labels_e: bool, random: bool, treatment_note=None, cache_from=None) -> Path:
    _require_package_root(root)
    P_ = paths(root)
    path = P_['configs'] / ('%s.json' % stage)
    if path.exists():
        return path
    sc = stage_caps(root, stage)
    cfg = {'study': 'tempo_aug_decision_learning', 'status': 'FROZEN', 'stage': stage, 'output': str(P_[stage]), 'jobs': list(jobs),
           'order': {j: list(order[j]) for j in jobs}, 'knowledge': knowledge, 'labels_e': bool(labels_e), 'llm': True, 'random': bool(random),
           'seeds': list(SEEDS), 'job_index': {j: JOB_INDEX[j] for j in jobs}, 'public_ids': list(PUBLIC), 'limits': LIMITS, 'max_tool_corrections': MAX_TOOL_CORRECTIONS,
           'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'fast_token_cap': FAST_TOKEN_CAP, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'fast_system': FAST_SYSTEM,
           'caps': sc['caps'], 'http_cap': sc['http_cap'], 'ledger_path': str(P_['ledger']), 'fit_retry': True, 'exposure': EXPOSURE_TAG,
           'fit_attempts_at_stage_start': context.read_json(P_['ledger'])['fit_attempts'], 'treatment_note': treatment_note or {}, 'cache_from': cache_from or {}}
    dsks.write_once(path, cfg)
    return path


def run_stage(root: Path, stage: str, cfg_path: Path) -> dict:
    _require_package_root(root)
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
    L.resume_stage(root, stage, accept_unknown_usage=accept_unknown_usage, restart=restart)


ACCEPT_UNKNOWN_USAGE = False                                       # set by --run --accept-unknown-usage (an explicit operator decision, recorded in the ledger)


def slow_client(root: Path, stage: str):
    P_ = paths(root)
    sc = stage_caps(root, stage)
    led = rt.RuntimeLedger(P_['ledger'], **sc['caps'])
    if rt.unknown_usage_blocks(led):
        if not ACCEPT_UNKNOWN_USAGE:
            raise RuntimeError('package ledger holds unknown usage; paid %s refused (pass --accept-unknown-usage to record an explicit operator decision)' % stage)
        led.s['unknown_usage_accepted'] = led.s['llm_tokens_unknown']
        led.s['events'].append({'kind': 'unknown_usage_accepted', 'stage': stage, 'count': led.s['llm_tokens_unknown'], 'epoch': time.time(), 'by': 'operator flag --accept-unknown-usage'})
        led._save()
    if not W.proxy_reachable():
        raise RuntimeError('LLM proxy not reachable; %s refused before any paid call' % stage)
    (P_[stage] / 'raw_responses').mkdir(parents=True, exist_ok=True)
    return led, rt.MeteredClient(led, P_[stage] / 'raw_responses', http_cap=sc['http_cap']), sc


# ============================================================================= Slow output parsing: contract O (parent parsers, prefixed ids) and contract N
CAND_KEYS_O = L.CAND_KEYS
CAND_KEYS_N = L.CAND_KEYS | {'decision_hypothesis'}
RULE_KEYS = {'rule', 'observable_condition', 'change_vs_default', 'support_refs', 'counter_refs', 'evidence_status', 'counterexample_handling', 'predicted_behavior_and_failure'}
EVIDENCE_STATUS = ('supported_in_these_cases', 'exploratory', 'unresolved')
HYP_KEYS = {'vs_default', 'contradicting_outcome', 'when_indistinguishable'}


def cand_id(call: str, cid: str) -> str:
    return '%s_%s' % (call, cid)                                   # runner prefix (task 5.2): O1_W1 ... N2_W2; the model's own id stays W1 / W2


def skill_id_of(call: str, cid: str, revision: int = 1) -> str:
    return '%s-%s-r%d' % (DOMAIN, cand_id(call, cid), revision)


def arm_of(skill_id: str) -> str:
    return L.arm_of(skill_id)                                       # RD02-O1_W1-r1 -> cand_O1_W1r1


def line_of(skill_id: str) -> str:
    return skill_id.split('-', 1)[1][0]                             # 'O' / 'N'


def call_of(skill_id: str) -> str:
    return skill_id.split('-', 1)[1].split('_', 1)[0]               # 'O1' ...


def formation_order_key(skill_id: str) -> tuple:
    calls = [c for c, _ in FORMATION_CALLS]
    return (calls.index(call_of(skill_id)), skill_id)


def _check_hypothesis(h) -> dict:
    if not isinstance(h, dict) or set(h) != HYP_KEYS or any(not isinstance(h[k], str) or not h[k].strip() for k in HYP_KEYS):
        raise ValueError('decision_hypothesis must hold exactly %s as nonempty text' % sorted(HYP_KEYS))
    return {k: h[k] for k in sorted(HYP_KEYS)}


def check_review_n(ev, *, legal_refs, allow_empty: bool) -> dict:
    if not isinstance(ev, dict) or set(ev) != {'decision_rules'}:
        raise ValueError('evidence_review must hold exactly decision_rules')
    rules = ev['decision_rules']
    if not isinstance(rules, list) or (not rules and not allow_empty):
        raise ValueError('decision_rules must be a non-empty list (one entry per reusable rule)')
    out = []
    for i, r in enumerate(rules):
        if not isinstance(r, dict) or set(r) != RULE_KEYS:
            raise ValueError('decision_rules[%d] must hold exactly %s' % (i, sorted(RULE_KEYS)))
        for k in ('rule', 'observable_condition', 'change_vs_default', 'counterexample_handling', 'predicted_behavior_and_failure'):
            if not isinstance(r[k], str) or not r[k].strip():
                raise ValueError('decision_rules[%d].%s must be nonempty text' % (i, k))
        for k in ('support_refs', 'counter_refs'):
            if not isinstance(r[k], list) or any(not isinstance(x, str) for x in r[k]):
                raise ValueError('decision_rules[%d].%s must be a list of strings' % (i, k))
            bad = [x for x in r[k] if x not in legal_refs]
            if bad:
                raise ValueError('decision_rules[%d].%s holds refs outside the legal ranges: %s' % (i, k, bad[:3]))
        if r['evidence_status'] not in EVIDENCE_STATUS:
            raise ValueError('decision_rules[%d].evidence_status must be one of %s' % (i, list(EVIDENCE_STATUS)))
        if r['evidence_status'] == 'supported_in_these_cases' and not r['support_refs']:
            raise ValueError('decision_rules[%d] is supported_in_these_cases without support_refs; mark it exploratory or unresolved' % i)
        out.append({k: r[k] for k in sorted(RULE_KEYS)})
    return {'decision_rules': out}


def make_card(**kw) -> dsk.Skill:
    return L.make_card(**kw)


def parse_formation(resp, *, contract: str, call: str, legal_refs) -> dict:
    """Both contracts: KEEP or PROPOSE with 1..2 candidates (W1 / W2), principles null, const:true, unique research modes, distinct bodies.
    Contract O = the parent's exploration schema; contract N adds decision_hypothesis and the decision_rules evidence review."""
    ids = ('W1', 'W2')
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'PROPOSE'):
        raise ValueError('decision must be KEEP or PROPOSE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale', 'evidence_review'}:
            raise ValueError('KEEP carries only rationale and evidence_review')
        ev = resp.get('evidence_review')
        review = (L.check_review(ev) if contract == 'O' else check_review_n(ev, legal_refs=legal_refs, allow_empty=True)) if ev is not None else None
        return {'decision': 'KEEP', 'rationale': str(resp.get('rationale', ''))[:3000], 'evidence_review': review, 'skills': [], 'meta': {}}
    if set(resp) != {'decision', 'candidates'}:
        raise ValueError('PROPOSE carries exactly decision and candidates')
    cands = resp['candidates']
    if not isinstance(cands, list) or not 1 <= len(cands) <= MAX_CANDIDATES_PER_CALL:
        raise ValueError('PROPOSE needs 1..%d candidates' % MAX_CANDIDATES_PER_CALL)
    want = CAND_KEYS_O if contract == 'O' else CAND_KEYS_N
    skills, meta, seen, modes = [], {}, set(), set()
    for c in cands:
        if not isinstance(c, dict) or set(c) != want:
            raise ValueError('candidate keys must be exactly %s' % sorted(want))
        cid = c['candidate_id']
        if cid not in ids or cid in seen:
            raise ValueError('candidate ids must be %s and unique' % list(ids))
        if c['principles'] is not None:
            raise ValueError('principles must be null in the formation calls')
        if c['observable_applicability'] != {'const': True}:
            raise ValueError('observable_applicability must be exactly {"const": true}')
        if str(c['research_mode']).strip().lower() in modes:
            raise ValueError('each candidate needs its own research_mode')
        review = L.check_review(c['evidence_review']) if contract == 'O' else check_review_n(c['evidence_review'], legal_refs=legal_refs, allow_empty=False)
        hyp = _check_hypothesis(c['decision_hypothesis']) if contract == 'N' else None
        s = make_card(skill_id=skill_id_of(call, cid), revision=1, workflow=c['workflow'], principles=None, summary=c['applicability_summary'], refs=c['evidence_refs'],
                      legal_refs=legal_refs, mode=c['research_mode'])
        seen.add(cid)
        modes.add(str(c['research_mode']).strip().lower())
        skills.append(s)
        meta[s.skill_id] = {'candidate_id': cand_id(call, cid), 'model_candidate_id': cid, 'line': contract, 'call': call, 'research_mode': c['research_mode'],
                            'rationale': str(c['rationale'])[:3000], 'evidence_review': review, 'decision_hypothesis': hyp}
    if len({s.rendered_body for s in skills}) != len(skills):
        raise ValueError('two candidates render the same body')
    return {'decision': 'PROPOSE', 'skills': skills, 'meta': meta}


def parse_revision(resp, *, contract: str, parent: dsk.Skill, legal_refs) -> dict:
    if contract == 'O':
        out = L.parse_revision(resp, parent=parent, legal_refs=legal_refs)
        if out['decision'] == 'REVISE':
            out['meta'] = {**out['meta'], 'line': 'O', 'expected_trigger_scenario': None, 'decision_hypothesis': None}
        return out
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'REVISE'):
        raise ValueError('decision must be KEEP or REVISE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale', 'evidence_review'}:
            raise ValueError('KEEP carries only rationale and evidence_review')
        return {'decision': 'KEEP', 'rationale': str(resp.get('rationale', ''))[:3000], 'evidence_review': check_review_n(resp.get('evidence_review'), legal_refs=legal_refs, allow_empty=True),
                'counterexample_responses': None, 'skill': None, 'meta': {}}
    if set(resp) != {'decision', 'child'}:
        raise ValueError('REVISE carries exactly decision and child')
    c = resp['child']
    want = {'workflow', 'principles', 'changed_mechanism', 'expected_trigger_scenario', 'research_mode', 'applicability_summary', 'evidence_refs', 'decision_hypothesis', 'rationale', 'evidence_review'}
    if not isinstance(c, dict) or set(c) != want:
        raise ValueError('child keys must be exactly %s' % sorted(want))
    for k in ('changed_mechanism', 'expected_trigger_scenario'):
        if not isinstance(c[k], str) or not c[k].strip():
            raise ValueError('%s must be nonempty text' % k)
    if c['principles'] is not None and (not isinstance(c['principles'], str) or not c['principles'].strip()):
        raise ValueError('principles must be null or nonempty text')
    review = check_review_n(c['evidence_review'], legal_refs=legal_refs, allow_empty=False)
    hyp = _check_hypothesis(c['decision_hypothesis'])
    child_id = re.sub(r'-r\d+$', '', parent.skill_id) + '-r%d' % (parent.revision + 1)
    s = make_card(skill_id=child_id, revision=parent.revision + 1, workflow=c['workflow'], principles=c['principles'], summary=c['applicability_summary'], refs=c['evidence_refs'],
                  legal_refs=legal_refs, mode=c['research_mode'], derived_from=parent.skill_id, source_stage='revision')
    if s.rendered_body == parent.rendered_body:
        raise ValueError('the child renders identically to the parent; return KEEP instead')
    return {'decision': 'REVISE', 'skill': s, 'meta': {'line': 'N', 'changed_mechanism': c['changed_mechanism'], 'expected_trigger_scenario': c['expected_trigger_scenario'],
                                                       'research_mode': c['research_mode'], 'rationale': str(c['rationale'])[:3000], 'evidence_review': review, 'decision_hypothesis': hyp},
            'counterexample_responses': None}


def skill_json(s):
    return L.skill_json(s)


def load_skills(path: Path) -> dict:
    return L.load_skills(path)


# ============================================================================= A. formation: four independent calls (O1, N1, N2, O2), at most eight W0 cards
def o_payload(cen: dict, summary: dict) -> dict:
    """Contract O input: the parent's exploration payload (old-style census with knowledge_loaded) + the common P_source / zero-fit summary."""
    p = L.exploration_payload(cen)
    p['source_selected_default_and_zero_fit_summary'] = {'P_source': summary['P_source'], 'J_source': summary['J_source'], 'definition': summary['definition'],
                                                         'per_job': summary['per_job'], 'pooled': summary['pooled']}
    return p


def n_payload(n_in: dict) -> dict:
    return {'domain_id': DOMAIN, 'facts': {k: v for k, v in n_in.items() if k != 'legal_evidence_refs'}, 'legal_evidence_refs': L.ref_ranges(n_in['legal_evidence_refs']),
            'legal_evidence_refs_format': 'evidence_refs / support_refs / counter_refs must be exact strings "<stage>/<branch>/<job>:<integer>" inside these ranges, or one of other_refs',
            'allowed_batch_features': sorted(ALLOWED_FEATURES), 'max_candidates': MAX_CANDIDATES_PER_CALL, 'body_limit_characters': BODY_LIMIT,
            'required': {'principles': None, 'observable_applicability': {'const': True}}, 'status': 'CANDIDATE_TEST_ONLY'}


def formation_inputs(root: Path, led=None, sc=None) -> dict:
    """Frozen once: the common tier, the contract O census, the zero-fit summary, the contract N input, the equivalence check and all four sizes."""
    P_ = paths(root)
    P_['evidence'].mkdir(parents=True, exist_ok=True)
    f = P_['evidence'] / 'formation_inputs.json'
    if f.exists():
        return context.read_json(f)
    dsks.write_once(P_['evidence'] / 'compression_rules.json', {**COMPRESSION_RULES, 'frozen_local': now()})
    chosen = None
    for tier in (3, 'skeleton'):
        cen = o_census(root, tier)
        if cen['census_status'] != 'CENSUS_COMPLETE':
            raise RuntimeError('initial census incomplete')
        summ = evidence_summary(cen)
        n_in = reorganize_for_n(cen, summ, purpose='Initial formation input (contract N): the same ten trajectories organised as decision cases, materials / actions and historical hypotheses')
        sizes = {'O': payload_bytes(SLOW_O_FORM, o_payload(cen, summ)), 'N': payload_bytes(SLOW_N_FORM, n_payload(n_in))}
        order = [sizes[line] for _, line in FORMATION_CALLS]
        chk = plan_check(led, sc, order) if led is not None else {'fits': True, 'note': 'no ledger (dry run)'}
        chosen = {'tier': str(tier), 'sizes': sizes, 'sizes_in_call_order': order, 'plan_check': chk}
        if chk['fits']:
            break
    if not chosen['plan_check']['fits']:
        raise RuntimeError('no common tier lets the four formation calls fit the frozen reservation: %s' % chosen)
    cen['compression']['tier_used'] = chosen['tier']
    cen['compression']['formation_plan'] = chosen
    n_in['compression'] = cen['compression']
    eq = facts_equivalent(cen, n_in)
    if not eq['equivalent']:
        raise RuntimeError('contract inputs are not fact-equivalent: %s' % eq['differences'][:5])
    dsks.write_once(P_['evidence'] / 'census_initial_o.json', cen)
    dsks.write_once(P_['evidence'] / 'evidence_summary.json', summ)
    dsks.write_once(P_['evidence'] / 'census_initial_n.json', n_in)
    rec = {**chosen, 'equivalence': eq, 'frozen_local': now(), 'calls': [c for c, _ in FORMATION_CALLS], 'system_prompt_chars': {'O': len(SLOW_O_FORM), 'N': len(SLOW_N_FORM)}}
    dsks.write_once(f, rec)
    return rec


def interrupted_request(stage_root: Path, role: str, unit: str):
    """The last request file of (role*, unit) without a response file: a call that was in flight when the driver died. Returns (path, record) or None."""
    reqs = sorted((stage_root / 'raw_responses').glob('*_request.json')) if (stage_root / 'raw_responses').exists() else []
    for req in reversed(reqs):
        q = context.read_json(req)
        if q.get('unit') != unit or not str(q.get('role', '')).startswith(role):
            continue
        if not req.with_name(req.name.replace('_request.json', '_response.json')).exists():
            return req, q
        return None
    return None


def resend_interrupted(client, led, req: Path, q: dict, parse, *, stage_root: Path) -> dict:
    """Task 8 recovery: the identical messages are sent once more (one extra HTTP attempt, counted as a request number by the frozen client but logged
    as a verbatim resend, not a new scientific question). The interrupted request's usage stays UNKNOWN in the ledger."""
    log = stage_root / 'resends.json'
    rec = context.read_json(log) if log.exists() else {'resends': []}
    if any(r['of_request'] == req.name for r in rec['resends']):
        return {'status': 'RESEND_ALREADY_USED', 'attempts': [{'resend_of': req.name, 'error': 'one identical resend per request; not repeated'}], 'raw': None}
    if len(rec['resends']) >= 4:
        return {'status': 'CALL_FAILED', 'attempts': [{'resend_of': req.name, 'error': 'package resend cap (4) reached'}], 'raw': None}
    system, payload = q['messages'][0]['content'], json.loads(q['messages'][1]['content'])
    role = q['role'] + '_resend'
    rec['resends'].append({'of_request': req.name, 'role': role, 'unit': q['unit'], 'epoch': time.time(), 'local': now(), 'request_number_after': None})
    context.write_json(log, rec)
    attempts = [{'attempt': 'interrupted', 'request_file': req.name, 'note': 'driver killed while in flight; usage UNKNOWN; identical messages re-sent once'}]
    try:
        raw = client.call(role, q['unit'], payload, system, max_tokens=int(q['max_tokens']))
    except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
        attempts.append({'attempt': 'resend', 'fault_kind': type(exc).__name__})
        return {'status': 'CALL_FAILED', 'attempts': attempts, 'raw': None}
    except ValueError as exc:
        attempts.append({'attempt': 'resend', 'error': 'response not JSON: %s' % str(exc)[:200]})
        return {'status': 'PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'raw': None}
    finally:
        rec['resends'][-1]['request_number_after'] = led.s['llm_requests']
        context.write_json(log, rec)
    try:
        out = parse(raw)
        attempts.append({'attempt': 'resend', 'ok': True})
        return {'status': out['decision'], 'attempts': attempts, 'raw': raw, **out}
    except (ValueError, rd.PolicyError) as exc:
        attempts.append({'attempt': 'resend', 'error': str(exc)[:300]})
        return {'status': 'PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'raw': raw if dsk._jsonable(raw) else None}


def formation_a(root: Path) -> dict:
    """O1, N1, N2, O2: each reads only its contract's frozen input (no other call's output, no Practice), each KEEP-able, one contract correction each.
    Identical bodies across calls merge as aliases (first call in order wins). Per line: PROPOSED / NO_PROPOSAL; any technical failure -> FORMATION_TECHNICAL_FAILURE."""
    P_ = paths(root)
    out = P_['formation_a']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'candidates.json').exists():
        return context.read_json(out / 'candidates.json')
    led, client, sc = slow_client(root, 'formation_a')
    inputs = formation_inputs(root, led, sc)
    cen = context.read_json(P_['evidence'] / 'census_initial_o.json')
    summ = context.read_json(P_['evidence'] / 'evidence_summary.json')
    n_in = context.read_json(P_['evidence'] / 'census_initial_n.json')
    legal = frozenset(cen['legal_evidence_refs'])
    forbidden = [n for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')]
    payloads = {'O': o_payload(cen, summ), 'N': n_payload(n_in)}
    systems = {'O': SLOW_O_FORM, 'N': SLOW_N_FORM}
    for call, line in FORMATION_CALLS:
        f = out / ('%s.json' % call)
        if f.exists():
            continue
        parse = lambda r, call=call, line=line: parse_formation(r, contract=line, call=call, legal_refs=legal)
        hit = interrupted_request(out, 'slow_form_%s' % call.lower(), call)
        if hit is not None:                                       # recovery: the interrupted request is re-sent verbatim once; Slow is not asked a new question
            req, q = hit
            size, chk = len(json.dumps(q['messages'], ensure_ascii=False).encode('utf-8')), check_reservation(led, sc, len(json.dumps(q['messages'], ensure_ascii=False).encode('utf-8')))
            if not chk['fits']:
                raise RuntimeError('resend of %s does not fit the frozen reservation: %s' % (req.name, chk))
            res = resend_interrupted(client, led, req, q, parse, stage_root=out)
        else:
            payload = copy.deepcopy(payloads[line])
            payload['formation_call'] = {'call': call, 'contract': line, 'note': 'independent formation; no output of any other call is present; identical outputs are recorded as aliases, not resampled'}
            size = payload_bytes(systems[line], payload)
            chk = check_reservation(led, sc, size)
            if not chk['fits_with_correction']:
                raise RuntimeError('formation payload %s does not fit the frozen reservation with a correction margin: %s' % (call, chk))
            res = L.slow_call(client, 'slow_form_%s' % call.lower(), call, payload, systems[line], parse, forbidden=forbidden)
        rec = {**{k: v for k, v in res.items() if k != 'skills'}, 'skills': {s.skill_id: s.to_json() for s in res.get('skills', [])}, 'call': call, 'contract': line, 'system_prompt': systems[line],
               'reservation_check': chk, 'payload_bytes': size, 'requests_after': led.s['llm_requests'], 'written_local': now()}
        dsks.write_once(f, rec)
        if client.fatal or rt.unknown_usage_blocks(led):
            raise RuntimeError('backend fatal or unknown usage during formation A')
    recs = {call: context.read_json(out / ('%s.json' % call)) for call, _ in FORMATION_CALLS}
    statuses = {c: r['status'] for c, r in recs.items()}
    if any(s in ('CALL_FAILED', 'PARSE_OR_VALIDATION_FAILED') for s in statuses.values()):
        rec = {'status': 'FORMATION_TECHNICAL_FAILURE', 'calls': statuses, 'skills': {}, 'meta': {}, 'aliases': {}, 'lines': {}, 'written_local': now()}
        dsks.write_once(out / 'candidates.json', rec)
        return rec
    skills, aliases, meta = {}, {}, {}
    for call, line in FORMATION_CALLS:
        for sid, sj in recs[call].get('skills', {}).items():
            same = next((k for k, v in skills.items() if v['rendered_body'] == sj['rendered_body']), None)
            if same:
                aliases[sid] = same
            else:
                skills[sid] = sj
                meta[sid] = recs[call]['meta'][sid]
    lines = {}
    for line in LINES:
        own = [s for s in skills if line_of(s) == line]
        own_alias = {s: t for s, t in aliases.items() if line_of(s) == line}
        lines[line] = {'candidates': sorted(own, key=formation_order_key), 'aliases': own_alias, 'n_effective': len(own), 'calls': {c: statuses[c] for c, l in FORMATION_CALLS if l == line},
                       'status': 'PROPOSED' if own else 'NO_PROPOSAL', 'keep_rationales': {c: recs[c].get('rationale') for c, l in FORMATION_CALLS if l == line and statuses[c] == 'KEEP'}}
    status = 'PROPOSED' if any(v['status'] == 'PROPOSED' for v in lines.values()) else 'NO_PROPOSAL_BOTH_LINES'
    rec = {'status': status, 'calls': statuses, 'skills': skills, 'meta': meta, 'aliases': aliases, 'lines': lines, 'n_candidates': len(skills), 'inputs': inputs, 'written_local': now()}
    dsks.write_once(out / 'candidates.json', rec)
    return rec


# ============================================================================= B. practice on L1 / L2 (all candidates, interleaved), one parent per line by J_P
def practice_arms(cands: dict) -> list:
    """Interleaved by line in candidate order: O1_W1, N1_W1, O1_W2, N1_W2, O2_W1, N2_W1, ... (missing entries skipped)."""
    o = cands['lines']['O']['candidates'] if cands.get('lines') else []
    n = cands['lines']['N']['candidates'] if cands.get('lines') else []
    out = []
    for i in range(max(len(o), len(n))):
        if i < len(o):
            out.append(o[i])
        if i < len(n):
            out.append(n[i])
    return out


def practice_stage(root: Path, cands: dict) -> dict:
    skills = {k: dsk.skill_from_json(v) for k, v in cands['skills'].items()}
    ids = practice_arms(cands)
    arms = {arm_of(k): skills[k] for k in ids}
    order = {'RD02_L1': [arm_of(k) for k in ids], 'RD02_L2': [arm_of(k) for k in reversed(ids)]}
    knowledge = {j: {a: asdict(dsk.skill_knowledge(s)) for a, s in arms.items()} for j in PRACTICE_JOBS}
    cache_from = {j: str(PARENT_LL / 'practice' / (j + '_common') / j) for j in PRACTICE_JOBS if (PARENT_LL / 'practice' / (j + '_common') / j / 'aug_materials' / 'index.json').exists()}
    cfgp = stage_config(root, 'practice', jobs=PRACTICE_JOBS, order=order, knowledge=br.json_copy(knowledge), labels_e=False, random=False, cache_from=cache_from,
                        treatment_note={'aliases': cands.get('aliases', {})})
    return run_stage(root, 'practice', cfgp)


def j_table(stage_root: Path, jobs, arms: dict) -> dict:
    return L.j_table(stage_root, jobs, arms)


def keep_parents(root: Path, cands: dict) -> dict:
    """Per line: the lowest J_P among candidates complete and scorable on both practice batches; ties by formation call order then candidate id.
    Allocates the revision budget; not a claim of effectiveness; a line without a complete candidate gets NO_PARENT and the other continues."""
    P_ = paths(root)
    out = P_['formation_a'] / 'parents.json'
    if out.exists():
        return context.read_json(out)
    arms = {arm_of(k): k for k in cands['skills']}
    tab = j_table(P_['practice'], PRACTICE_JOBS, arms)
    lines = {}
    for line in LINES:
        own = [a for a in arms if line_of(arms[a]) == line]
        ok = [a for a in own if a in tab['J']]
        ranked = sorted(ok, key=lambda a: (tab['J'][a], formation_order_key(arms[a])))
        best = ranked[0] if ranked else None
        if best is not None:
            top = tab['J'][best]
            best = min([a for a in ok if tab['J'][a] - top <= TOL], key=lambda a: formation_order_key(arms[a]))
        lines[line] = {'arms': own, 'ranked': ranked, 'J': {a: tab['J'][a] for a in ok}, 'parent': arms[best] if best else None, 'parent_arm': best,
                       'status': 'PARENT_KEPT' if best else 'NO_PARENT', 'incomplete': [a for a in own if a not in tab['J']]}
    parents = [lines[l]['parent'] for l in LINES if lines[l]['parent']]
    rec = {**tab, 'lines': lines, 'parents': parents, 'rule': 'per line, the lowest J_P among candidates complete and scorable on both practice batches; ties <= 1e-12 by formation call order then candidate id; '
                                                         'one parent per line allocates the revision budget and is not a claim of effectiveness',
           'status': 'PARENTS_KEPT' if parents else 'PRACTICE_INCOMPLETE', 'written_local': now()}
    dsks.write_once(out, rec)
    return rec


# ============================================================================= C. one revision per line (O: parent SLOW_C; N: SLOW_N_REV), Principles allowed
def practice_census(root: Path, cands: dict, tier) -> dict:
    arms = [arm_of(k) for k in practice_arms(cands)]
    return L.census(root, L.practice_branches(root, arms), {str(paths(root)['practice']): 'practice'}, tier,
                    purpose='Practice: real use of every candidate card of both contracts on two new batches, with post-commit C_B of every evaluated material')


def branch_facts(prac: dict, p_source: str | None) -> dict:
    """Deterministic per-branch facts of EVERY practice branch (both lines): committed material, its C_A / C_B means, the lowest-C_B evaluated material,
    None and default C_B. The same numbers both contracts receive (contract O additionally gets the parent-local counterexample list)."""
    out = {}
    for job in PRACTICE_JOBS:
        J = prac['jobs'].get(job)
        if not J:
            continue
        for b in J['branches']:
            dc = decision_case(b, J['materials'], p_source, card_ref=None)
            nc = dc['numeric_consequences']
            out[b['ref_prefix']] = {'job': job, 'status': b['status'], 'committed': nc['committed_plan_id'], 'committed_physical_id': nc['committed_physical_id'],
                                    'committed_c_a_mean': nc['committed_c_a_mean'], 'committed_c_b_mean': nc['committed_c_b_mean'],
                                    'c_b_committed_minus_none': nc['vs_none']['c_b_committed_minus_none'], 'c_b_committed_minus_default': nc['vs_default']['c_b_committed_minus_default'],
                                    'lowest_c_b_among_evaluated': nc['lowest_c_b_among_evaluated'], 'committed_is_lowest_c_b_evaluated': nc['committed_is_lowest_c_b_evaluated'],
                                    'c_b_regret_vs_lowest_evaluated': nc['c_b_regret_vs_lowest_evaluated'], 'cost': b['cost']}
    return out


def revision_stage(root: Path, cands: dict, parents: dict) -> dict:
    P_ = paths(root)
    out = P_['revision']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'children.json').exists():
        return context.read_json(out / 'children.json')
    summ = context.read_json(P_['evidence'] / 'evidence_summary.json')
    p_source = summ['P_source']
    cen_o_skel = o_census(root, 'skeleton', purpose='Initial evidence, condensed (skeleton) for revision')
    n_skel = reorganize_for_n(cen_o_skel, summ, purpose='Initial evidence (contract N organisation), condensed for revision')
    if not (P_['evidence'] / 'census_initial_o_skeleton.json').exists():
        dsks.write_once(P_['evidence'] / 'census_initial_o_skeleton.json', cen_o_skel)
        dsks.write_once(P_['evidence'] / 'census_initial_n_skeleton.json', n_skel)
    skills = {k: dsk.skill_from_json(v) for k, v in cands['skills'].items()}
    known = {k: v['rendered_body'] for k, v in cands['skills'].items()}
    led, client, sc = slow_client(root, 'revision')
    f_in = P_['evidence'] / 'revision_inputs.json'
    if not f_in.exists():
        chosen = None
        for tier in (3, 'skeleton'):
            prac = practice_census(root, cands, tier)
            prac_n = reorganize_for_n(prac, summ, known_cards=known, card_table='candidate_cards', purpose='Practice facts (contract N organisation): every candidate of both contracts on two new batches')
            probe_card = {'rendered_body': 'x' * BODY_LIMIT, 'rationale': 'x' * 3000, 'evidence_review': 'x' * 6000, 'decision_hypothesis': 'x' * 1500}
            sizes = {}
            for line, sysm in (('O', SLOW_O_REV), ('N', SLOW_N_REV)):
                probe = {'parent_card': probe_card, 'initial_condensed': {k: v for k, v in (cen_o_skel if line == 'O' else n_skel).items() if k != 'legal_evidence_refs'},
                         'practice': {k: v for k, v in (prac if line == 'O' else prac_n).items() if k != 'legal_evidence_refs'}, 'branch_facts': branch_facts(prac, p_source),
                         'numeric_counterexamples_for_parent': 'x' * 3000, 'source_default_and_public_references': summ,
                         'legal_evidence_refs': L.ref_ranges(sorted(set(cen_o_skel['legal_evidence_refs']) | set(prac['legal_evidence_refs'])))}
                sizes[line] = payload_bytes(sysm, probe)
            order = [sizes[l] for l in LINES if parents['lines'][l]['parent']]
            chk = plan_check(led, sc, order)
            chosen = {'tier': str(tier), 'sizes': sizes, 'sizes_in_call_order': order, 'plan_check': chk}
            if chk['fits']:
                break
        if not chosen['plan_check']['fits']:
            raise RuntimeError('no common tier lets the revision calls fit the frozen reservation: %s' % chosen)
        prac['compression']['tier_used'] = chosen['tier']
        prac['compression']['revision_plan'] = chosen
        prac_n['compression'] = prac['compression']
        eq = facts_equivalent(prac, prac_n)
        if not eq['equivalent']:
            raise RuntimeError('practice inputs are not fact-equivalent: %s' % eq['differences'][:5])
        dsks.write_once(P_['evidence'] / 'census_practice_o.json', prac)
        dsks.write_once(P_['evidence'] / 'census_practice_n.json', prac_n)
        dsks.write_once(f_in, {**chosen, 'equivalence': eq, 'frozen_local': now()})
    prac = context.read_json(P_['evidence'] / 'census_practice_o.json')
    prac_n = context.read_json(P_['evidence'] / 'census_practice_n.json')
    facts = branch_facts(prac, p_source)
    legal = frozenset(cen_o_skel['legal_evidence_refs']) | frozenset(prac['legal_evidence_refs'])
    forbidden = [n for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')]
    children, records = {}, {}
    for line in LINES:
        sid = parents['lines'][line]['parent']
        if not sid:
            records[line] = {'status': 'NO_PARENT', 'attempts': []}
            continue
        f = out / ('revise_%s.json' % sid)
        if f.exists():
            rec = context.read_json(f)
        else:
            parent = skills[sid]
            m = cands['meta'][sid]
            common = {'domain_id': DOMAIN, 'parent_practice_arm': arm_of(sid), 'branch_facts_all_practice_branches': facts, 'source_default_and_public_references': summ,
                      'legal_evidence_refs': L.ref_ranges(sorted(legal)),
                      'legal_evidence_refs_format': 'evidence_refs must be exact strings "<stage>/<branch>/<job>:<integer>" inside these ranges, or one of other_refs',
                      'allowed_batch_features': sorted(ALLOWED_FEATURES), 'body_limit_characters': BODY_LIMIT, 'required': {'observable_applicability': {'const': True}, 'max_children': 1},
                      'status': 'CANDIDATE_TEST_ONLY'}
            if line == 'O':
                payload = {**common, 'parent_card': {'skill_id': sid, 'workflow': parent.workflow, 'principles': parent.principles, 'rendered_body': parent.rendered_body, 'rationale': m['rationale'],
                                                     'evidence_review': m['evidence_review'], 'research_mode': m['research_mode']},
                           'initial_census_condensed': {k: v for k, v in cen_o_skel.items() if k != 'legal_evidence_refs'},
                           'practice_census': {k: v for k, v in prac.items() if k != 'legal_evidence_refs'},
                           'numeric_counterexamples_for_parent': L.counterexamples(root, arm_of(sid), prac)}
                system = SLOW_O_REV
            else:
                payload = {**common, 'parent_card': {'skill_id': sid, 'workflow': parent.workflow, 'principles': parent.principles, 'rendered_body': parent.rendered_body, 'rationale': m['rationale'],
                                                     'decision_hypothesis': m.get('decision_hypothesis'), 'evidence_review': m['evidence_review'], 'research_mode': m['research_mode']},
                           'initial_facts_condensed': {k: v for k, v in n_skel.items() if k != 'legal_evidence_refs'},
                           'practice_facts': {k: v for k, v in prac_n.items() if k != 'legal_evidence_refs'},
                           'numeric_counterexamples_for_parent': L.counterexamples(root, arm_of(sid), prac)}
                system = SLOW_N_REV
            size = payload_bytes(system, payload)
            chk = check_reservation(led, sc, size)
            if not chk['fits_with_correction']:
                raise RuntimeError('revision payload %s does not fit the frozen reservation with a correction margin: %s' % (sid, chk))
            res = L.slow_call(client, 'slow_revise_%s' % line.lower(), sid, payload, system, lambda r, parent=parent, line=line: parse_revision(r, contract=line, parent=parent, legal_refs=legal), forbidden=forbidden)
            rec = {**{k: v for k, v in res.items() if k != 'skill'}, 'skill': skill_json(res.get('skill')), 'parent': sid, 'line': line, 'system_prompt': system, 'reservation_check': chk, 'payload_bytes': size,
                   'requests_after': led.s['llm_requests'], 'written_local': now()}
            dsks.write_once(f, rec)
            if client.fatal or rt.unknown_usage_blocks(led):
                raise RuntimeError('backend fatal or unknown usage during revision')
        records[line] = {k: rec.get(k) for k in ('status', 'attempts')}
        records[line]['parent'] = sid
        if rec['status'] == 'REVISE':
            children[rec['skill']['skill_id']] = {'skill': rec['skill'], 'parent': sid, 'line': line, 'meta': rec.get('meta')}
        elif rec['status'] == 'KEEP':
            children[sid + '-KEEP'] = {'skill': None, 'parent': sid, 'line': line, 'alias_of_parent': True, 'rationale': rec.get('rationale'), 'evidence_review': rec.get('evidence_review'),
                                       'counterexample_responses': rec.get('counterexample_responses')}
    tech = [l for l, r in records.items() if r['status'] in ('CALL_FAILED', 'PARSE_OR_VALIDATION_FAILED')]
    rec = {'status': 'REVISION_TECHNICAL_FAILURE' if tech else 'REVISED', 'per_line': records, 'children': children, 'technical_failures': tech, 'written_local': now()}
    dsks.write_once(out / 'children.json', rec)
    return rec


def amend_stage_cap(root: Path, stage: str, extra_tokens: int, reason: str, decided_by: str) -> dict:
    """Operator-approved amendment of one frozen stage token cap (recorded in stages.json and frozen_config_amendment_N.json); the package total is untouched."""
    P_ = paths(root)
    rec = context.read_json(P_['stages'])
    n = 1 + len(list(root.glob('frozen_config_amendment_*.json')))
    before = rec[stage]['caps']['max_llm_tokens']
    rec[stage]['caps']['max_llm_tokens'] = min(TOTAL['max_llm_tokens'], before + int(extra_tokens))
    rec[stage].setdefault('amendments', []).append({'n': n, 'field': 'caps.max_llm_tokens', 'before': before, 'after': rec[stage]['caps']['max_llm_tokens'], 'reason': reason, 'decided_by': decided_by, 'local': now()})
    context.write_json(P_['stages'], rec)
    am = {'amendment': n, 'stage': stage, 'field': 'caps.max_llm_tokens', 'before': before, 'after': rec[stage]['caps']['max_llm_tokens'], 'extra_tokens': int(extra_tokens), 'reason': reason,
          'decided_by': decided_by, 'package_total_unchanged': TOTAL, 'local': now(), 'epoch': time.time()}
    dsks.write_once(root / ('frozen_config_amendment_%d.json' % n), am)
    return am


def resume_revision(root: Path, cands: dict, parents: dict) -> dict:
    """Recovery of a revision whose ONE contract correction was refused by the frozen reservation (never sent): the correction round of that same
    scientific call is sent now, under the (amended) stage cap. No new scientific question; a line whose scientific call itself failed is not resumed."""
    P_ = paths(root)
    out = P_['revision']
    ch = out / 'children.json'
    if not ch.exists() or context.read_json(ch)['status'] != 'REVISION_TECHNICAL_FAILURE':
        return context.read_json(ch) if ch.exists() else {'status': 'NOTHING_TO_RESUME'}
    n_prev = len(list(out.glob('children.failed_*.json'))) + 1
    shutil.move(ch, out / ('children.failed_%d.json' % n_prev))
    summ = context.read_json(P_['evidence'] / 'evidence_summary.json')
    cen_o_skel = context.read_json(P_['evidence'] / 'census_initial_o_skeleton.json')
    prac = context.read_json(P_['evidence'] / 'census_practice_o.json')
    legal = frozenset(cen_o_skel['legal_evidence_refs']) | frozenset(prac['legal_evidence_refs'])
    skills = {k: dsk.skill_from_json(v) for k, v in cands['skills'].items()}
    led, client, sc = slow_client(root, 'revision')
    resumed = {}
    for line in LINES:
        sid = parents['lines'][line]['parent']
        f = out / ('revise_%s.json' % sid) if sid else None
        if not f or not f.exists():
            continue
        rec = context.read_json(f)
        att = rec.get('attempts') or []
        pending = rec['status'] == 'CALL_FAILED' and len(att) == 2 and 'error' in att[0] and att[1].get('fault_kind') == 'BudgetExhausted' \
            and not list((out / 'raw_responses').glob('*_slow_revise_%s_contract_correction_request.json' % line.lower()))
        if not pending:
            continue
        first = sorted((out / 'raw_responses').glob('*_slow_revise_%s_request.json' % line.lower()))[-1]
        q = context.read_json(first)
        raw_prev = context.read_json(first.with_name(first.name.replace('_request.json', '_response.json')))['choices'][0]['message']['content']
        prior = json.loads(raw_prev)
        payload = {**json.loads(q['messages'][1]['content']), 'correction': {'previous_output': prior, 'error': att[0]['error'],
                                                                              'instruction': 'Correct this contract error only. KEEP is allowed. Do not seek a different outcome.'}}
        system = q['messages'][0]['content']
        chk = check_reservation(led, sc, payload_bytes(system, payload))
        if not chk['fits']:
            raise RuntimeError('the pending correction still does not fit the amended reservation: %s' % chk)
        shutil.move(f, out / ('revise_%s.failed_%d.json' % (sid, n_prev)))
        parent = skills[sid]
        parse = lambda r, parent=parent, line=line: parse_revision(r, contract=line, parent=parent, legal_refs=legal)
        attempts = list(att[:1]) + [{'attempt': 1, 'refused_by_reservation': att[1], 'resumed_after_amendment': True}]
        try:
            raw = client.call('slow_revise_%s_contract_correction' % line.lower(), sid, payload, system, max_tokens=MAX_OUTPUT_TOKENS)
        except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
            attempts.append({'attempt': 'resumed_correction', 'fault_kind': type(exc).__name__})
            res = {'status': 'CALL_FAILED', 'attempts': attempts, 'raw': None}
        except ValueError as exc:
            attempts.append({'attempt': 'resumed_correction', 'error': 'response not JSON: %s' % str(exc)[:200]})
            res = {'status': 'PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'raw': None}
        else:
            try:
                o = parse(raw)
                attempts.append({'attempt': 'resumed_correction', 'ok': True})
                res = {'status': o['decision'], 'attempts': attempts, 'raw': raw, **o}
            except (ValueError, rd.PolicyError) as exc:
                attempts.append({'attempt': 'resumed_correction', 'error': str(exc)[:300]})
                res = {'status': 'PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'raw': raw if dsk._jsonable(raw) else None}
        rec2 = {**{k: v for k, v in res.items() if k != 'skill'}, 'skill': skill_json(res.get('skill')), 'parent': sid, 'line': line, 'system_prompt': system, 'reservation_check': chk,
                'payload_bytes': payload_bytes(system, payload), 'requests_after': led.s['llm_requests'], 'written_local': now(), 'resumed_correction': True}
        dsks.write_once(f, rec2)
        resumed[line] = rec2['status']
    context.write_json(out / 'resume_revision.json', {'local': now(), 'resumed_lines': resumed, 'archived_children': 'children.failed_%d.json' % n_prev})
    return revision_stage(root, cands, parents)


# ============================================================================= D. select on L3 / L4: F0, O parent, O child, N parent, N child; freeze W_O / W_N / Fixed_dev / H_O / H_N
def select_options(cands: dict, parents: dict, children: dict) -> tuple:
    """({arm: (skill_id, Skill)} in task order [no_skill, O_parent, O_child, N_parent, N_child], {alias_arm: run_arm}); KEEP children and identical
    bodies are strict aliases (one execution, recorded once)."""
    skills = {k: dsk.skill_from_json(v) for k, v in cands['skills'].items()}
    for cid, c in children['children'].items():
        if c.get('skill'):
            skills[cid] = dsk.skill_from_json(c['skill'])
    opts, aliases = {'no_skill': (None, None)}, {}
    for line in LINES:
        sid = parents['lines'][line]['parent']
        if not sid:
            continue
        for cand in (sid, next((cid for cid, c in children['children'].items() if c['parent'] == sid and c.get('skill')), None)):
            if not cand:
                continue
            same = next((a for a, (s2, sk) in opts.items() if sk is not None and sk.rendered_body == skills[cand].rendered_body), None)
            if same:
                aliases[arm_of(cand)] = same
            else:
                opts[arm_of(cand)] = (cand, skills[cand])
    return opts, aliases


def select_stage(root: Path, cands: dict, parents: dict, children: dict) -> dict:
    opts, aliases = select_options(cands, parents, children)
    order = {'RD02_L3': list(opts), 'RD02_L4': list(reversed(list(opts)))}
    knowledge = {j: {a: asdict(dsk.skill_knowledge(s)) for a, (sid, s) in opts.items()} for j in SELECT_JOBS}
    cache_from = {j: str(PARENT_LL / 'select' / (j + '_common') / j) for j in SELECT_JOBS if (PARENT_LL / 'select' / (j + '_common') / j / 'aug_materials' / 'index.json').exists()}
    keep_alias = {cid: c['parent'] for cid, c in children['children'].items() if c.get('alias_of_parent')}
    cfgp = stage_config(root, 'select', jobs=SELECT_JOBS, order=order, knowledge=br.json_copy(knowledge), labels_e=False, random=False, cache_from=cache_from,
                        treatment_note={'strict_aliases': aliases, 'keep_aliases': keep_alias})
    return run_stage(root, 'select', cfgp)


def _public_j(stage_root: Path, jobs) -> dict:
    return L._public_j(stage_root, jobs)


def _arm_cost(stage_root: Path, jobs, arm: str) -> dict:
    cells, toks = [], []
    for j in jobs:
        b = stage_root / ('%s_%s' % (j, arm))
        if (b / 'branch_result.json').exists():
            cells.append(len(W.branch_cells(b / j)))
            t = dsks._unit_tokens(stage_root).get('fast:%s' % b.name) or {}
            toks.append(t.get('prompt_tokens', 0) + t.get('completion_tokens', 0))
    return {'mean_logical_cells': _mean(cells), 'mean_tokens': _mean(toks)}


def _argmin_fixed_order(J: dict) -> str | None:
    if not J:
        return None
    best = min(J.values())
    return min([m for m in J if J[m] - best <= TOL], key=lambda m: PUBLIC.index(m))


def freeze_choices(root: Path, cands: dict, parents: dict, children: dict) -> dict:
    """W_O / W_N: the lowest-J_S non-empty card of each line (parent before child, then candidate id). Fixed_dev: the lowest-J_S public program.
    H_O / H_N: the lowest J_S among {W_line, no_skill, None, FixedMixup, P_AmpResample, P_NoMixRecipe}; ties <= 1e-12 prefer a fixed strategy (public order),
    then among non-fixed by mean logical cells, mean actual tokens, then id order. Frozen before any replay feature reaches an Agent."""
    P_ = paths(root)
    out = P_['formation_a'] / 'frozen_choices.json'
    if out.exists():
        return context.read_json(out)
    opts, aliases = select_options(cands, parents, children)
    tab = j_table(P_['select'], SELECT_JOBS, {a: sid for a, (sid, s) in opts.items()})
    J = dict(tab['J'])
    for alias, run_arm in aliases.items():
        if run_arm in J:
            J[alias] = J[run_arm]
    pub_J = _public_j(P_['select'], SELECT_JOBS)
    costs = {a: _arm_cost(P_['select'], SELECT_JOBS, a) for a in opts}
    all_skills = {**cands['skills'], **{cid: c['skill'] for cid, c in children['children'].items() if c.get('skill')}}
    line_cards = {line: {'parent': parents['lines'][line]['parent'],
                         'child': next((cid for cid, c in children['children'].items() if c['parent'] == parents['lines'][line]['parent'] and c.get('skill')), None) if parents['lines'][line]['parent'] else None}
                  for line in LINES}
    rec = {**tab, 'J_with_aliases': J, 'strict_aliases': aliases, 'public_J': pub_J, 'select_costs': costs, 'incomplete_options': [a for a in opts if a not in J],
           **apply_freeze_rules(J, pub_J, costs, aliases, line_cards)}
    if rec['status'] == 'FROZEN':
        rec['skills'] = {l['W']['skill_id']: all_skills[l['W']['skill_id']] for l in rec['lines'].values() if l['W']}
        rec['frozen_local'], rec['frozen_epoch'] = now(), time.time()
    dsks.write_once(out, rec)
    return rec


# ============================================================================= E. development replay on L5 / L6 / L7 (EXPOSED): F0, F_O, F_N, Random, Menu_CA; fixed strategies by their real models
def replay_plan(frozen: dict, p_source: str) -> dict:
    """Pure: which replay arms run, with which frozen knowledge, which are aliases (no execution, no file)."""
    if frozen['status'] != 'FROZEN':
        raise RuntimeError('choices are not frozen; replay refused')
    skills = {k: dsk.skill_from_json(v) for k, v in frozen['skills'].items()}
    w_o, w_n = frozen['W_O'], frozen['W_N']
    same = bool(w_o and w_n and skills[w_o['skill_id']].rendered_body == skills[w_n['skill_id']].rendered_body)
    note = {'exposure': EXPOSURE_TAG}
    kn = {'f0': asdict(br.Knowledge()), 'f_o': asdict(dsk.skill_knowledge(skills[w_o['skill_id']])) if w_o else None,
          'f_n': (None if same else asdict(dsk.skill_knowledge(skills[w_n['skill_id']]))) if w_n else None, 'random': None, 'menu_ca': None}
    if not w_o:
        note['f_o'] = {'status': 'NO_TREATMENT', 'note': 'contract O produced no non-empty card that completed Select; F_O is not run and no F0 alias substitutes for it'}
    if not w_n:
        note['f_n'] = {'status': 'NO_TREATMENT', 'note': 'contract N produced no non-empty card that completed Select; F_N is not run and no F0 alias substitutes for it'}
    if same:
        note['f_n'] = {'status': 'ALIAS', 'alias_of_arm': 'f_o', 'note': 'W_N renders identically to W_O: one Fast run under the identical frozen knowledge, reused once and reported as an alias'}
    for line, h in (('O', frozen['H_O']), ('N', frozen['H_N'])):
        arm = 'f0' if h['kind'] == 'no_skill' else (('f_o' if line == 'O' or same else 'f_n') if h['kind'] == 'card' else None)
        note['h_%s' % line.lower()] = {'status': 'ALIAS', 'kind': h['kind'], 'alias_of_arm': arm, 'public_id': h['choice'] if h['kind'] == 'fixed_public' else None,
                                       'note': 'H_%s is the Select-frozen policy; its delivery is that arm\'s delivery (a fixed program = its own real model of the batch); no fifth research arm and no re-selection at replay' % line,
                                       'candidate_test_isolated': ('the line\'s best non-empty card still runs (F_%s) as an isolated candidate test, not as the adopted system' % line) if h['kind'] != 'card' else None}
    order = {j: [a for a in TEST_ORDER[j] if not (a == 'f_n' and (same or not w_n)) and not (a == 'f_o' and not w_o)] + ['menu_ca'] for j in TEST_JOBS}
    catalog = {'status': 'REPLAY_FROZEN', 'exposure': EXPOSURE_TAG, 'W_O': w_o, 'W_N': w_n, 'H_O': frozen['H_O'], 'H_N': frozen['H_N'], 'Fixed_dev': frozen['Fixed_dev'], 'P_source': p_source,
               'f_n_alias_of_f_o': same, 'orders': order, 'treatment_note': note, 'frozen_before_first_replay_fit': True}
    return {'same': same, 'knowledge': kn, 'note': note, 'order': order, 'catalog': catalog}


def test_stage(root: Path, frozen: dict) -> dict:
    _require_package_root(root)
    plan = replay_plan(frozen, context.read_json(paths(root)['evidence'] / 'evidence_summary.json')['P_source'])
    cat = paths(root)['formation_a'] / 'test_catalog.json'
    if not cat.exists():
        dsks.write_once(cat, {**plan['catalog'], 'frozen_local': now(), 'frozen_epoch': time.time()})
    cfgp = stage_config(root, 'test', jobs=TEST_JOBS, order=plan['order'], knowledge=br.json_copy({j: plan['knowledge'] for j in TEST_JOBS}), labels_e=True, random=True, treatment_note=plan['note'],
                        cache_from={j: str(PARENT_LL / 'test' / (j + '_common') / j) for j in TEST_JOBS if (PARENT_LL / 'test' / (j + '_common') / j / 'aug_materials' / 'index.json').exists()})
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
def replay_exposure() -> dict:
    """L5-L7 were scored (E) by the parent learning-loop package: this package's replay is EXPOSED development, never fresh. Read from the parent's artefacts."""
    out = {}
    for j in TEST_JOBS:
        scored = sorted(p.parent.parent.name for p in (PARENT_LL / 'test').glob('RD02_*/%s/e_scores.json' % j)) if (PARENT_LL / 'test').exists() else []
        out[j] = {'E_scored_by_parent_learning_loop_branches': scored, 'exposure': EXPOSURE_TAG, 'fresh': False}
    return out


def preflight(root: Path = ROOT) -> dict:
    import inspect
    root.mkdir(parents=True, exist_ok=True)
    p = root / 'frozen_config.json'
    if p.exists():
        return context.read_json(p)
    env = W.environment()
    prev = context.read_json(PARENT_LL / 'frozen_config.json')['environment']
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
    rex = replay_exposure()
    exposure = {j: {'eligible_and_scorable': avail['RD02@%d' % JOB_T[j]]['eligible_and_scorable'], **rex[j]} for j in TEST_JOBS}
    blocked = [j for j, v in exposure.items() if not v['eligible_and_scorable']]
    if blocked:
        raise RuntimeError('a replay batch is not eligible / scorable: %s' % blocked)
    max_row = max(v['last_row_read_exclusive'] for v in jobs.values())
    sealing_ok = max_row <= W.PRSA_SEALED_FROM_ROW
    ll_fc = context.read_json(PARENT_LL / 'frozen_config.json')
    ll_status = context.read_json(PARENT_LL / 'package_status.json')
    history = {'parent_learning_loop': str(PARENT_LL), 'parent_learning_loop_finished': ll_status.get('finished_local'), 'parent_workflow_skill': str(PARENT_W),
               'historical_trajectories': [b['branch'] for b in L.historical_branches()], 'old_cards': [s['skill_id'] for s in context.read_json(PARENT_W / 'formation' / 'propose.json')['skills']],
               'parent_ledgers': {'learning_loop': {k: v for k, v in context.read_json(PARENT_LL / 'budget.json').items() if k != 'events'},
                                  'workflow_skill': {k: v for k, v in context.read_json(PARENT_W / 'budget.json').items() if k != 'events'}}}
    cfg = {'package': PACKAGE, 'task': TASK, 'frozen_local': now(), 'frozen_epoch': time.time(), 'exposure': EXPOSURE_TAG,
           'hypothesis': 'organising the same evidence as observable condition -> actual decision -> delayed outcome -> strong simple default, with per-rule support / counterexample handling (contract N), '
                         'yields a more useful Workflow than the trajectory-consolidation contract O; a joint change of Slow task contract + input organisation + rule evidence expression, not attributed to one sentence',
           'dataset': {'id': DATASET, 'roster': rd.DATASETS[DATASET]['roster'], 'exposure': rd.EXPOSURE, 'prsa_sealed_from_row': W.PRSA_SEALED_FROM_ROW, 'max_row_read_exclusive': max_row, 'sealing_ok': sealing_ok},
           'geometry': ll_fc['geometry'], 'jobs': jobs, 'curriculum': {'initial_evidence': list(INITIAL_JOBS), 'practice': list(PRACTICE_JOBS), 'select': list(SELECT_JOBS), 'development_replay': list(TEST_JOBS)},
           'information_permissions': {'initial_slow': 'T / trajectories / C_A / C_B of the ten historical branches of the six initial jobs only; no L1-L7 experience; no E',
                                       'revision_slow': 'initial evidence + this package\'s own Practice T / trajectories / C_A / C_B of every candidate of both contracts; no Select / replay feedback; no E',
                                       'fast': 'T and its own C_A of the current job plus one frozen card; the four public references; never another arm\'s candidates, C_B or E',
                                       'replay_e': 'read only after every replay delivery is frozen and C_B is opened; historical E (already exposed) is not a learning input'},
           'orders': {'formation': [c for c, _ in FORMATION_CALLS], 'practice': 'L1 candidate order interleaved by line (O1_W1, N1_W1, O1_W2, ...), L2 reversed',
                      'select': 'L3 [no_skill, O_parent, O_child, N_parent, N_child], L4 reversed; strict aliases run once', 'replay': {j: list(o) + ['menu_ca'] for j, o in TEST_ORDER.items()}},
           'seeds': list(SEEDS), 'consumer': W.CONSUMER, 'environment': env, 'environment_matches_parent': env_ok, 'model': MODEL,
           'fast_system': FAST_SYSTEM, 'fast_system_identical_to_parent': FAST_SYSTEM == L.FAST_SYSTEM, 'slow_o_form': SLOW_O_FORM, 'slow_o_rev': SLOW_O_REV, 'slow_n_form': SLOW_N_FORM, 'slow_n_rev': SLOW_N_REV,
           'contract_difference': 'contract O = parent SLOW_A1 (formation, both calls) / SLOW_C (revision) with the old-style census; contract N = SLOW_N_FORM / SLOW_N_REV with the decision-case organisation and '
                                  'decision_rules evidence review; everything else (Fast, tools, budget, materials, seeds, Consumer, selection rules, adoption set) identical',
           'contracts': W.CONTRACTS, 'limits': LIMITS, 'fast_token_cap': FAST_TOKEN_CAP, 'max_tool_corrections': MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': EVIDENCE_ROUNDTRIP,
           'body_limit_characters': BODY_LIMIT, 'body_limit_default_unchanged': dsk.BODY_LIMIT, 'materials': ll_fc['materials'], 'random_search_b4': ll_fc['random_search_b4'], 'menu_ca': ll_fc['menu_ca'],
           'physical_cache_from_parent': {'practice': {j: str(PARENT_LL / 'practice' / (j + '_common') / j) for j in PRACTICE_JOBS}, 'select': {j: str(PARENT_LL / 'select' / (j + '_common') / j) for j in SELECT_JOBS},
                                          'test': {j: str(PARENT_LL / 'test' / (j + '_common') / j) for j in TEST_JOBS}, 'wiring': str(PARENT_LL / 'wiring' / (WIRING_JOB + '_common') / WIRING_JOB)},
           'selection': {'J': 'J(v) = mean_j(mean_seed C_B(committed_v) / mean_seed C_B(None)); lower is better; tolerance 1e-12',
                         'parents': 'per line the lowest J_P on L1/L2 among candidates complete on both; ties by formation call order then candidate id',
                         'W_line': 'lowest J_S on L3/L4 among the line\'s parent + child (parent first on ties, then id)', 'Fixed_dev': 'lowest-J_S public program (public order on ties)',
                         'H_line': 'lowest J_S among {W_line, no_skill, None, FixedMixup, P_AmpResample, P_NoMixRecipe}; ties prefer a fixed strategy, then non-fixed by mean logical cells, mean tokens, id order',
                         'P_source': 'lowest J_source over the six initial jobs (public order on ties); an initial default and evidence summary only'},
           'main_readout': {'d_contract': 'E(F_O) - E(F_N)', 'd_skill': 'E(F0) - E(F_N)', 'd_default': 'E(Fixed_dev) - E(F_N)', 'd_system': 'E(H_O) - E(H_N)',
                            'G': 'mean_j(100 x mean_s d / mean_s E(None)) over the three replay jobs (percentage points of that job\'s None loss)',
                            'interval': 'paired SE and n=3 two-sided 95%% t interval (df=2, t=%.6f)' % T975_DF2},
           'budget': {'total': TOTAL, 'stage_allocations': ALLOC, 'stage_retries': STAGE_RETRIES, 'http_cap': HTTP_CAP, 'numeric_wall_s': NUMERIC_WALL_S},
           'label_barriers': 'Practice / Select: all planned arms end -> C_B only. Replay: all three jobs\' arms end and every delivery frozen -> C_B -> all E predictions frozen -> barrier -> E',
           'replay_exposure': exposure, 'history': history}
    if not (env_ok and sealing_ok):
        raise RuntimeError('preflight failed: environment %s sealing %s' % (env_ok, sealing_ok))
    dsks.write_once(p, cfg)
    return cfg


def wiring(root: Path = ROOT) -> dict:
    """Instrument check on RD02_T1 with the historical seeds (parent wiring cache copied after binding checks; scripted seven-tool branch:
    compare -> build the same legal composition -> evaluate (cache) -> compare -> commit -> C_B; 0 E; 0 LLM). Not a learning case."""
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
           'cache_from': {job: str(PARENT_LL / 'wiring' / (job + '_common') / job)}, 'fast_system': FAST_SYSTEM}
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


# ============================================================================= readout (task 7)
def _stats(v):
    return W._stats(v)


def _paired(a, b, den=None):
    return W._paired(a, b, den)


def _arm_rec(stage_root: Path, job: str, arm: str):
    return L._arm_rec(stage_root, job, arm)


REF_NAMES = ('random', 'menu_ca', 'fixed_dev', 'p_source', 'None', 'FixedMixup', 'P_AmpResample', 'P_NoMixRecipe')


def _sign(x):
    return int(x > TOL) - int(x < -TOL)


def concordance_public(pub: dict) -> dict:
    """Six public pairs of one job: sign agreement of the paired mean difference across C_A -> C_B, C_A -> E, C_B -> E; ties listed."""
    out = {'pairs': {}, 'rates': {}}
    counts = {k: {'agree': 0, 'ties': 0, 'of': 0} for k in ('c_a_to_c_b', 'c_a_to_e', 'c_b_to_e')}
    for i, a in enumerate(PUBLIC):
        for b in PUBLIC[i + 1:]:
            row = {}
            for blk in ('c_a', 'c_b', 'e'):
                va, vb = pub[a].get(blk), pub[b].get(blk)
                row[blk] = _pair_stats(va, vb) if va and vb else None
            for k, (x, y) in {'c_a_to_c_b': ('c_a', 'c_b'), 'c_a_to_e': ('c_a', 'e'), 'c_b_to_e': ('c_b', 'e')}.items():
                if row[x] and row[y]:
                    counts[k]['of'] += 1
                    sx, sy = row[x]['sign_of_mean'], row[y]['sign_of_mean']
                    if sx == 0 or sy == 0:
                        counts[k]['ties'] += 1
                        row[k] = 'tie'
                    else:
                        counts[k]['agree'] += sx == sy
                        row[k] = 'same' if sx == sy else 'opposite'
            out['pairs']['%s|%s' % (a, b)] = row
    for k, c in counts.items():
        out['rates'][k] = {**c, 'rate_excluding_ties': _r(c['agree'] / (c['of'] - c['ties'])) if c['of'] - c['ties'] else None}
    return out


def branch_private_selection(arm_rec: dict) -> dict | None:
    """One Fast / Random arm's own evaluated set: argmin C_A, argmin C_B, argmin E, the commit, the E regret against the best evaluated and against the C_A argmin."""
    ev = arm_rec.get('all_evaluated') or {}
    with_e = {m: v for m, v in ev.items() if v.get('e') and v.get('c_a')}
    if not with_e or not arm_rec.get('e'):
        return None
    mean = lambda m, b: statistics.fmean(with_e[m][b]) if with_e[m].get(b) else None
    argmin = lambda b: min([m for m in with_e if with_e[m].get(b)], key=lambda m: (mean(m, b), m))
    commit = arm_rec['commit']
    a_ca, a_e = argmin('c_a'), argmin('e')
    a_cb = argmin('c_b') if all(with_e[m].get('c_b') for m in with_e) else None
    e_commit = statistics.fmean(arm_rec['e'])
    return {'n_evaluated': len(with_e), 'committed': commit, 'argmin_c_a': a_ca, 'argmin_c_b': a_cb, 'argmin_e': a_e, 'committed_is_argmin_e': commit == a_e, 'c_a_argmin_is_argmin_e': a_ca == a_e,
            'c_b_argmin_is_argmin_e': (a_cb == a_e) if a_cb else None,
            'e_regret_committed_vs_best_evaluated': _r(e_commit - mean(a_e, 'e')), 'e_regret_c_a_argmin_vs_best_evaluated': _r(mean(a_ca, 'e') - mean(a_e, 'e')),
            'e_means': {m: _r(mean(m, 'e')) for m in with_e}}


def readout(root: Path = ROOT) -> dict:
    P_ = paths(root)
    res = {'package': PACKAGE, 'written_local': now(), 'exposure': EXPOSURE_TAG,
           'units': {'loss': 'missing-aware normalized MSE macro (lower is better)',
                     'gain_pp': 'paired per seed: 100 x (loss_reference - loss_arm) / three-seed mean None loss of that job; positive = arm (second) better; percentage points of that job\'s None loss, never a percentage of Fixed',
                     'd_contract': 'E(F_O) - E(F_N)', 'd_skill': 'E(F0) - E(F_N)', 'd_default': 'E(Fixed_dev) - E(F_N)', 'd_system': 'E(H_O) - E(H_N)',
                     'G': 'mean over the three replay jobs of 100 x mean_s d / mean_s E(None); three jobs equal weight; nine seed x job cells are not nine independent tasks'},
           'stages': {}, 'formation_a': {}, 'evidence_summary': {}, 'practice': {}, 'revision': {}, 'select': {}, 'frozen': {}, 'replay': {}, 'main': {}, 'evidence_table': {}, 'concordance': {}, 'cost': {}}
    for st in ('practice', 'select', 'test'):
        if (P_[st] / 'execution_finished.json').exists():
            res['stages'][st] = dsks.stage_status(P_[st])
    if (P_['wiring'] / 'wiring.json').exists():
        res['stages']['wiring'] = context.read_json(P_['wiring'] / 'wiring.json')['status']
    for name in ('O1', 'N1', 'N2', 'O2', 'candidates', 'parents', 'frozen_choices', 'test_catalog'):
        f = P_['formation_a'] / (name + '.json')
        if f.exists():
            res['formation_a'][name] = {k: v for k, v in context.read_json(f).items() if k not in ('raw', 'runs', 'system_prompt', 'inputs')}
    if (P_['evidence'] / 'evidence_summary.json').exists():
        s = context.read_json(P_['evidence'] / 'evidence_summary.json')
        res['evidence_summary'] = {k: s[k] for k in ('J_source', 'P_source', 'pooled')} | {'per_job_same_direction': {j: v['same_direction_rate'] for j, v in s['per_job'].items()}}
    cands = context.read_json(P_['formation_a'] / 'candidates.json') if (P_['formation_a'] / 'candidates.json').exists() else {'skills': {}, 'lines': {}}
    if (P_['revision'] / 'children.json').exists():
        res['revision'] = context.read_json(P_['revision'] / 'children.json')
        for f in sorted(P_['revision'].glob('revise_*.json')):
            r = context.read_json(f)
            res['revision'].setdefault('per_parent_detail', {})[r['parent']] = {k: r.get(k) for k in ('status', 'line', 'attempts', 'meta', 'rationale', 'evidence_review', 'counterexample_responses', 'reservation_check', 'payload_bytes')}
    for j in PRACTICE_JOBS:
        for k in cands['skills']:
            r = _arm_rec(P_['practice'], j, arm_of(k))
            if r:
                res['practice'].setdefault(j, {})[arm_of(k)] = r
    frozen = context.read_json(P_['formation_a'] / 'frozen_choices.json') if (P_['formation_a'] / 'frozen_choices.json').exists() else None
    if frozen:
        res['frozen'] = {k: frozen.get(k) for k in ('status', 'W_O', 'W_N', 'Fixed_dev', 'H_O', 'H_N', 'J', 'J_with_aliases', 'public_J', 'strict_aliases', 'select_costs', 'incomplete_options', 'lines')}
        for j in SELECT_JOBS:
            for arm in frozen['arms']:
                r = _arm_rec(P_['select'], j, arm)
                if r:
                    res['select'].setdefault(j, {})[arm] = r
        pairs = {}
        for line, lrec in (frozen.get('lines') or {}).items():
            if lrec.get('parent') and lrec.get('child'):
                pa, ch = arm_of(lrec['parent']), arm_of(lrec['child'])
                ch_run = (frozen.get('strict_aliases') or {}).get(ch, ch)
                pairs[line] = {j: _paired(res['select'].get(j, {}).get(pa, {}).get('c_b'), res['select'].get(j, {}).get(ch_run, {}).get('c_b')) for j in SELECT_JOBS}
        res['select']['parent_child_paired_c_b'] = pairs
    cat = context.read_json(P_['formation_a'] / 'test_catalog.json') if (P_['formation_a'] / 'test_catalog.json').exists() else None
    G = {'d_contract': [], 'd_skill': [], 'd_default': [], 'd_system': []}
    complete, vs_acc = [], {}
    if cat:
        same = cat['f_n_alias_of_f_o']
        note = cat['treatment_note']
        for j in TEST_JOBS:
            row = {'arms': {}, 'aliases': {}}
            for arm in ('f0', 'f_o', 'f_n', 'random', 'menu_ca'):
                src = 'f_o' if (arm == 'f_n' and same) else arm
                r = _arm_rec(P_['test'], j, src) if not (note.get(arm) or {}).get('status') == 'NO_TREATMENT' else None
                row['arms'][arm] = ({**r, 'alias_of': src} if src != arm else r) if r else {'status': 'NOT_RUN', 'reason': (note.get(arm) or {}).get('note')}
                if src != arm:
                    row['aliases'][arm] = src
            anyb = next((P_['test'] / ('%s_%s' % (j, a)) for a in ('f0', 'random', 'menu_ca') if (P_['test'] / ('%s_%s' % (j, a)) / j / 'e_scores.json').exists()), None)
            if anyb is not None:
                row['public'] = {m: {b_: W._block_vec(anyb, j, m, b_) for b_ in ('c_a', 'c_b', 'e')} for m in PUBLIC}
                none_e = row['public']['None']['e']
                den = statistics.fmean(none_e) if none_e else None
                row['none_mean_e'] = den
                fixed, psrc = cat['Fixed_dev']['public_id'], cat['P_source']
                row['fixed_dev'] = {'public_id': fixed, **row['public'][fixed]}
                row['p_source'] = {'public_id': psrc, **row['public'][psrc]}
                for line in ('o', 'n'):
                    h = note['h_' + line]
                    if h['kind'] == 'fixed_public':
                        row['arms']['h_' + line] = {'status': 'ALIAS_FIXED', 'public_id': h['public_id'], 'commit': h['public_id'], **row['public'][h['public_id']], 'alias_of': 'public:' + h['public_id']}
                    else:
                        base = row['arms'].get(h['alias_of_arm']) or {}
                        row['arms']['h_' + line] = {**base, 'alias_of': h['alias_of_arm']}
                    row['aliases']['h_' + line] = row['arms']['h_' + line].get('alias_of')
                e = lambda a: (row['arms'].get(a) or {}).get('e')
                row['main'] = {'d_contract': _paired(e('f_o'), e('f_n'), den), 'd_skill': _paired(e('f0'), e('f_n'), den), 'd_default': _paired(row['fixed_dev']['e'], e('f_n'), den),
                               'd_system': _paired(e('h_o'), e('h_n'), den)}
                refs_ = {'random': e('random'), 'menu_ca': e('menu_ca'), 'fixed_dev': row['fixed_dev']['e'], 'p_source': row['p_source']['e'], **{m: row['public'][m]['e'] for m in PUBLIC}}
                for who in ('f_n', 'h_n', 'f_o', 'h_o', 'f0'):
                    row[who + '_vs'] = {k: _paired(v, e(who), den) for k, v in refs_.items()}
                    for k, v in row[who + '_vs'].items():
                        if v:
                            vs_acc.setdefault((who, k), []).append(v['gain_pp']['mean'])
                if den:
                    for k in G:
                        if row['main'][k]:
                            G[k].append(row['main'][k]['gain_pp']['mean'])
                    complete.append(j)
                row['branch_private_selection'] = {a: branch_private_selection(row['arms'][a]) for a in ('f0', 'f_o', 'f_n', 'random') if row['arms'][a].get('all_evaluated')}
                row['public_concordance'] = concordance_public(row['public'])
            res['replay'][j] = row
        res['main'] = {'complete_jobs': complete, 'complete': len(complete) == len(TEST_JOBS),
                       **{'G_' + k: (statistics.fmean(v) if len(v) == len(TEST_JOBS) else None) for k, v in G.items()},
                       'available': {k: len(v) for k, v in G.items()},
                       'per_job': {j: {k: (res['replay'][j]['main'][k] or {}).get('gain_pp') for k in G} for j in complete},
                       'signs_positive': {k: sum((res['replay'][j]['main'][k] or {}).get('gain_pp', {}).get('mean', 0) > 0 for j in complete if res['replay'][j]['main'][k]) for k in G}}
        for (who, k), vals in vs_acc.items():
            res['main']['G_%s_vs_%s' % (who, k)] = statistics.fmean(vals) if len(vals) == len(TEST_JOBS) else None
        res['concordance'] = {'public_by_job': {j: res['replay'][j].get('public_concordance') for j in TEST_JOBS if res['replay'].get(j, {}).get('public_concordance')},
                              'public_pooled': _pool_concordance([res['replay'][j]['public_concordance'] for j in TEST_JOBS if res['replay'].get(j, {}).get('public_concordance')]),
                              'branch_private_by_job': {j: res['replay'][j].get('branch_private_selection') for j in TEST_JOBS if res['replay'].get(j)},
                              'note': 'descriptive only, computed at report time after every E was scored; public pairs are the common candidate set (six pairs per job); branch-private sets differ in size and are not pooled into one accuracy; regret alongside hit rates'}
    res['evidence_table'] = evidence_table(root, cands, res)
    res['cost'] = cost_readout(root)
    context.write_json(root / 'result.json', res)
    (root / 'tables.md').write_text(tables(res), encoding='utf-8')
    return res


def _pool_concordance(rows: list) -> dict:
    out = {}
    for k in ('c_a_to_c_b', 'c_a_to_e', 'c_b_to_e'):
        agree = sum(r['rates'][k]['agree'] for r in rows)
        ties = sum(r['rates'][k]['ties'] for r in rows)
        of = sum(r['rates'][k]['of'] for r in rows)
        out[k] = {'agree': agree, 'ties': ties, 'of': of, 'rate_excluding_ties': _r(agree / (of - ties)) if of - ties else None}
    return out


def evidence_table(root: Path, cands: dict, res: dict) -> dict:
    """Task 7 short evidence rows per card: text, rules / review, practice behaviour, revision, Select and replay behaviour (tools, builds, commit, C_B / E).
    Mechanical facts only; whether a rule was triggered is read from the trajectories and reported as 'not examined' when no trigger is observable."""
    out = {}
    rev = res.get('revision') or {}
    frozen = res.get('frozen') or {}
    cat = (res.get('formation_a') or {}).get('test_catalog') or {}
    beh = lambda r: {k: v for k, v in (r or {}).items() if k in ('commit', 'commit_reason', 'c_a', 'c_b', 'e', 'fast_calls', 'new_evaluations')} | {'tools': ((r or {}).get('behavior') or {}).get('tool_sequence'), 'builds': [b['plan_id'] for b in ((r or {}).get('behavior') or {}).get('builds', [])],
                                                                                                                                                         'rules_in_builds': [bool(b.get('rules')) for b in ((r or {}).get('behavior') or {}).get('builds', [])]}
    all_cards = dict(cands.get('skills', {}))
    for cid, c in (rev.get('children') or {}).items():
        if c.get('skill'):
            all_cards[cid] = c['skill']
    for sid, sj in all_cards.items():
        arm = arm_of(sid)
        m = (cands.get('meta') or {}).get(sid) or ((rev.get('children') or {}).get(sid) or {}).get('meta') or {}
        chain = {'line': line_of(sid), 'card': {'skill_id': sid, 'revision': sj.get('revision'), 'derived_from': sj.get('derived_from'), 'research_mode': m.get('research_mode'), 'workflow': sj['workflow'], 'principles': sj.get('principles'),
                                                'body_chars': len(sj['rendered_body']), 'decision_hypothesis': m.get('decision_hypothesis'), 'evidence_review': m.get('evidence_review'), 'rationale': m.get('rationale')},
                 'old_prohibitions_in_text': {k: bool(re.search(p, sj['rendered_body'], re.I)) for k, p in {'no_3_step': r'(never|no|not)\s+(build\s+)?3-step|three-step|3 step', 'no_grouping': r'(do not|never|no)\s+(group|write rules|entity-conditional|conditional)',
                                                                                                             '3_of_3_seed_gate': r'3/3|all (3|three) seeds|non-mixed', 'trust_c_a': r'trust C_A|C_A rank is not', 'keep_public_on_mixed': r'mixed'}.items()},
                 'practice': {j: beh(res['practice'].get(j, {}).get(arm)) for j in PRACTICE_JOBS if res['practice'].get(j, {}).get(arm)}}
        child = next(((k, v) for k, v in (rev.get('children') or {}).items() if v['parent'] == sid), None)
        if child:
            k, v = child
            chain['revision'] = {'child_id': k if v.get('skill') else None, 'keep': not v.get('skill'), 'changed_mechanism': (v.get('meta') or {}).get('changed_mechanism'), 'expected_trigger_scenario': (v.get('meta') or {}).get('expected_trigger_scenario'),
                                 'principles': (v.get('skill') or {}).get('principles'), 'rule_changes_or_rules': ((v.get('meta') or v).get('evidence_review') or {}).get('rule_changes') or ((v.get('meta') or v).get('evidence_review') or {}).get('decision_rules'),
                                 'counterexample_responses': (v.get('meta') or v).get('counterexample_responses'), 'rationale': v.get('rationale') or (v.get('meta') or {}).get('rationale')}
        sel_arm = (frozen.get('strict_aliases') or {}).get(arm, arm)
        chain['select'] = {j: beh(res['select'].get(j, {}).get(sel_arm)) | ({'alias_of': sel_arm} if sel_arm != arm else {}) for j in SELECT_JOBS if res['select'].get(j, {}).get(sel_arm)}
        rep_arm = None
        for line_arm, key in (('f_o', 'W_O'), ('f_n', 'W_N')):
            if (cat.get(key) or {}).get('skill_id') == sid:
                rep_arm = line_arm
        if rep_arm:
            chain['replay_arm'] = rep_arm
            chain['replay'] = {j: beh(res['replay'].get(j, {}).get('arms', {}).get(rep_arm)) | {'alias_of': res['replay'].get(j, {}).get('aliases', {}).get(rep_arm)} for j in TEST_JOBS if res['replay'].get(j)}
            chain['delivery_vs_f0_replay'] = {j: {'same_commit_as_f0': (res['replay'][j]['arms'].get(rep_arm) or {}).get('commit') == (res['replay'][j]['arms'].get('f0') or {}).get('commit'),
                                                  'same_commit_as_fixed_dev': (res['replay'][j]['arms'].get(rep_arm) or {}).get('commit') == (res['replay'][j].get('fixed_dev') or {}).get('public_id')} for j in TEST_JOBS if res['replay'].get(j)}
        out[sid] = chain
    return out


def cost_readout(root: Path) -> dict:
    P_ = paths(root)
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    stages = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    per_stage, names = {}, list(stages)
    keys = ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'fit_wall_seconds')
    for i, st in enumerate(names):
        a = stages[st]['snapshot_at_start']
        b = stages[names[i + 1]]['snapshot_at_start'] if i + 1 < len(names) else {k: led.get(k, 0) for k in a}
        per_stage[st] = {k: (b.get(k, 0) - a.get(k, 0)) for k in keys}
    tokens = {s: dsks._unit_tokens(P_[s]) for s in ('formation_a', 'practice', 'revision', 'select', 'test') if P_[s].exists()}
    per_arm, logical = {}, {}
    for s in ('practice', 'select', 'test'):
        if not P_[s].exists():
            continue
        for b in sorted(x for x in P_[s].iterdir() if x.is_dir() and (x / 'branch_result.json').exists()):
            job = '_'.join(b.name.split('_')[:2])
            cells = W.branch_cells(b / job)
            per_arm[b.name] = {'logical_cells': len(cells), 'new_plan_cells': sum(c['material_id'] not in PUBLIC for c in cells.values()), 'tokens': tokens[s].get('fast:%s' % b.name)}
        commons = [x for x in P_[s].iterdir() if x.is_dir() and x.name.endswith('_common')]
        stage_jobs = tuple(c.name[:-len('_common')] for c in commons)
        logical[s] = {'reference_cells_per_job': len(PUBLIC) * len(SEEDS), 'jobs': len(commons),
                      'logical_fits_no_cache': len(commons) * len(PUBLIC) * len(SEEDS) + sum(v['new_plan_cells'] for k, v in per_arm.items() if k.startswith(stage_jobs) and (P_[s] / k).is_dir())}
    ll = context.read_json(PARENT_LL / 'budget.json') if (PARENT_LL / 'budget.json').exists() else {}
    ww = context.read_json(PARENT_W / 'budget.json') if (PARENT_W / 'budget.json').exists() else {}
    inc = {k: sum(per_stage.get(s, {}).get(k, 0) for s in ('formation_a', 'practice', 'revision', 'select')) for k in keys}
    return {'ledger': {k: v for k, v in led.items() if k != 'events'}, 'per_stage_delta': per_stage, 'tokens_by_unit': tokens, 'per_arm': per_arm, 'logical_no_cache': logical,
            'numeric_seconds': W.numeric_seconds(root) if P_['ledger'].exists() else None,
            'layers': {'historical_assets': {'workflow_skill_package': {k: ww.get(k) for k in ('fit_attempts', 'llm_requests', 'llm_tokens_in', 'llm_tokens_out', 'fit_wall_seconds')},
                                             'learning_loop_package': {k: ll.get(k) for k in ('fit_attempts', 'llm_requests', 'llm_tokens_in', 'llm_tokens_out', 'fit_wall_seconds')}},
                       'incremental_learning_A_to_D_this_package': inc,
                       'deployment_per_batch_replay': {'physical': per_stage.get('test'), 'logical_rule': 'references 12 fits; each F arm <= 12 fits + <= 16 calls; Random 12 fits 0 LLM; Menu_CA / Fixed_dev / P_source 0 extra (their models are the references)',
                                                       'fixed_strategy_independent_deployment': 'training its own delivery model only (3 fits per batch); the other three references are the common experiment overhead of this study, not the fixed strategy\'s cost'}}}


def _f(x, nd=2):
    return '—' if x is None else ('%+.*f' % (nd, x))


def tables(res: dict) -> str:
    out = ['# %s tables' % PACKAGE, '', '%s. Replay batches are %s (already scored by the parent learning loop).' % (res['units']['gain_pp'], EXPOSURE_TAG), '']
    fr = res.get('frozen') or {}
    out += ['## Frozen choices', '']
    for k in ('W_O', 'W_N', 'Fixed_dev', 'H_O', 'H_N'):
        out.append('- %s: %s' % (k, json.dumps(fr.get(k), ensure_ascii=False)))
    out += ['- Select J (with aliases): %s' % json.dumps(fr.get('J_with_aliases')), '- public J: %s' % json.dumps(fr.get('public_J')), '- P_source / J_source: %s / %s' % ((res.get('evidence_summary') or {}).get('P_source'), json.dumps((res.get('evidence_summary') or {}).get('J_source'))), '']
    m = res.get('main') or {}
    out += ['## Main (replay, gain_pp; positive = second arm better)', '', '| Job | d_contract (F_O→F_N) | d_skill (F0→F_N) | d_default (Fixed_dev→F_N) | d_system (H_O→H_N) | None mean E |', '|---|---:|---:|---:|---:|---:|']
    for j in TEST_JOBS:
        r = res['replay'].get(j, {})
        mm = r.get('main') or {}
        cell = lambda k: ('%s (SE %s) [%s]' % (_f(mm[k]['gain_pp']['mean']), _f(mm[k]['gain_pp']['se']), ''.join('+' if s > 0 else '-' if s < 0 else '0' for s in mm[k]['gain_pp']['signs']))) if mm.get(k) else 'NOT_AVAILABLE'
        out.append('| %s | %s | %s | %s | %s | %s |' % (j, cell('d_contract'), cell('d_skill'), cell('d_default'), cell('d_system'), ('%.4f' % r['none_mean_e']) if r.get('none_mean_e') is not None else '—'))
    out += ['', 'G: d_contract %s, d_skill %s, d_default %s, d_system %s pp (complete jobs: %s; positive jobs: %s)' % (_f(m.get('G_d_contract')), _f(m.get('G_d_skill')), _f(m.get('G_d_default')), _f(m.get('G_d_system')), m.get('complete_jobs'), json.dumps(m.get('signs_positive'))), '']
    for who in ('f_n', 'h_n', 'f_o', 'h_o', 'f0'):
        out.append('- %s vs: ' % who + ', '.join('%s %s' % (ref_, _f(m.get('G_%s_vs_%s' % (who, ref_)))) for ref_ in REF_NAMES))
    out += ['', '## Replay deliveries and per-seed values', '', '| Job | arm | commit | E per seed | C_B per seed | C_A per seed | calls | new evals |', '|---|---|---|---|---|---|---:|---:|']
    fmt = lambda v: ', '.join('%.4f' % x for x in v) if v else '—'
    for j in TEST_JOBS:
        r = res['replay'].get(j, {})
        for arm, a in (r.get('arms') or {}).items():
            out.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (j, arm + ((' (alias of %s)' % a['alias_of']) if a.get('alias_of') else ''), a.get('commit'), fmt(a.get('e')), fmt(a.get('c_b')), fmt(a.get('c_a')), a.get('fast_calls', '—'), a.get('new_evaluations', '—')))
        for mname, v in (r.get('public') or {}).items():
            out.append('| %s | ref %s | %s | %s | %s | %s | 0 | 0 |' % (j, mname, mname, fmt(v.get('e')), fmt(v.get('c_b')), fmt(v.get('c_a'))))
    out += ['', '## Practice (J_P) and Select (J_S)', '']
    pa = (res.get('formation_a') or {}).get('parents') or {}
    out.append('- J_P: %s; parents: %s' % (json.dumps(pa.get('J')), pa.get('parents')))
    for j, arms in (res.get('practice') or {}).items():
        for arm, a in arms.items():
            out.append('- practice %s %s: commit %s, C_B %s, C_A %s, calls %s, new %s' % (j, arm, a.get('commit'), fmt(a.get('c_b')), fmt(a.get('c_a')), a.get('fast_calls'), a.get('new_evaluations')))
    for j in SELECT_JOBS:
        for arm, a in (res.get('select') or {}).get(j, {}).items():
            if isinstance(a, dict) and 'commit' in a:
                out.append('- select %s %s: commit %s, C_B %s, C_A %s, calls %s, new %s' % (j, arm, a.get('commit'), fmt(a.get('c_b')), fmt(a.get('c_a')), a.get('fast_calls'), a.get('new_evaluations')))
    con = res.get('concordance') or {}
    if con:
        out += ['', '## Zero-fit concordance (report time only)', '', '- public pairs pooled: %s' % json.dumps(con.get('public_pooled'))]
        for j, v in (con.get('branch_private_by_job') or {}).items():
            for arm, b in (v or {}).items():
                if b:
                    out.append('- %s %s: n=%d committed %s, argmin C_A %s, argmin E %s, E regret committed %s / C_A-argmin %s' % (j, arm, b['n_evaluated'], b['committed'], b['argmin_c_a'], b['argmin_e'], _f(b['e_regret_committed_vs_best_evaluated'], 4), _f(b['e_regret_c_a_argmin_vs_best_evaluated'], 4)))
    return '\n'.join(out) + '\n'


# ============================================================================= freeze rules as a pure function (used by freeze_choices and the smoke)
def apply_freeze_rules(J: dict, pub_J: dict, costs: dict, aliases: dict, line_cards: dict) -> dict:
    """J: {arm: J_S} including strict aliases; pub_J: {public id: J_S}; costs: {arm: {mean_logical_cells, mean_tokens}}; line_cards: {line: {parent, child}}."""
    lines = {}
    for line in LINES:
        sid, child = line_cards[line].get('parent'), line_cards[line].get('child')
        card_arms = [arm_of(x) for x in (sid, child) if x]
        ok = [a for a in card_arms if a in J]
        w = None
        if ok:
            best = min(J[a] for a in ok)
            w = min([a for a in ok if J[a] - best <= TOL], key=lambda a: (0 if a == arm_of(sid) else 1, a))
        w_sid = (sid if w == arm_of(sid) else child) if w else None
        lines[line] = {'parent': sid, 'child': child, 'card_arms': card_arms,
                       'W': {'arm': w, 'skill_id': w_sid, 'J': J.get(w), 'is_child': bool(w and w != arm_of(sid)), 'run_arm': aliases.get(w, w)} if w else None,
                       'no_learning_proposal': w is None, 'card_comparison': 'AVAILABLE' if w else 'NOT_AVAILABLE'}
    fixed = _argmin_fixed_order(pub_J)
    if 'no_skill' not in J or not pub_J or len(pub_J) < len(PUBLIC):
        return {'status': 'SELECT_INCOMPLETE', 'lines': lines, 'W_O': None, 'W_N': None, 'Fixed_dev': None, 'H_O': None, 'H_N': None,
                'note': 'no_skill or a public reference is missing on a Select job; no winner frozen by narrative'}

    def system_choice(line):
        cand = {'no_skill': J['no_skill'], **{m: pub_J[m] for m in PUBLIC}}
        w = lines[line]['W']
        if w:
            cand[w['arm']] = w['J']
        best = min(cand.values())
        tied = [k for k in cand if cand[k] - best <= TOL]

        def key(k):
            if k in PUBLIC:
                return (0, PUBLIC.index(k), 0, 0, '')
            c = costs.get(aliases.get(k, k)) or {}
            return (1, 0, c['mean_logical_cells'] if c.get('mean_logical_cells') is not None else float('inf'), c['mean_tokens'] if c.get('mean_tokens') is not None else float('inf'), '' if k == 'no_skill' else k)
        h = min(tied, key=key)
        kind = 'fixed_public' if h in PUBLIC else ('no_skill' if h == 'no_skill' else 'card')
        return {'choice': h, 'kind': kind, 'J': cand[h], 'candidates': cand, 'tied': tied, 'skill_id': (w['skill_id'] if kind == 'card' else None), 'run_arm': (w['run_arm'] if kind == 'card' else None)}
    return {'status': 'FROZEN', 'lines': lines, 'W_O': lines['O']['W'], 'W_N': lines['N']['W'], 'Fixed_dev': {'public_id': fixed, 'J': pub_J.get(fixed)}, 'H_O': system_choice('O'), 'H_N': system_choice('N'),
            'rules': 'quality only, no cost penalty, no significance gate; ties <= 1e-12: cards parent before child then id; systems prefer a fixed strategy (public order), then non-fixed by mean logical '
                     'cells, mean tokens, id order; the fixed programs enter both adoption sets identically (adoption-layer change, reported separately from the Slow contract effect)'}


# ============================================================================= smoke (task 8: new risks only; synthetic; 0 fits, 0 API)
def merge_formation(recs: dict) -> tuple:
    """(skills, aliases, meta): identical rendered bodies across the four calls alias to the first in call order."""
    skills, aliases, meta = {}, {}, {}
    for call, line in FORMATION_CALLS:
        for sid, sj in (recs.get(call) or {}).get('skills', {}).items():
            same = next((k for k, v in skills.items() if v['rendered_body'] == sj['rendered_body']), None)
            if same:
                aliases[sid] = same
            else:
                skills[sid] = sj
                meta[sid] = recs[call]['meta'][sid]
    return skills, aliases, meta


def smoke(out: Path) -> dict:
    import tempfile
    out.mkdir(parents=True, exist_ok=True)
    checks, notes = {}, {}
    # 1. contract inputs are fact-equivalent (both tiers); old cards moved once each and traceable; initial inputs cover the six initial jobs only; no E anywhere
    for tier in (3, 'skeleton'):
        cen = o_census(ROOT, tier)
        summ = evidence_summary(cen)
        n_in = reorganize_for_n(cen, summ, purpose='smoke')
        eq = facts_equivalent(cen, n_in)
        txt_o, txt_n = json.dumps(cen), json.dumps(n_in)
        checks['facts_equivalent_tier_%s' % tier] = eq['equivalent'] and eq['n_branches'] == 10 and eq['cards_once_each'] and eq['card_refs_traceable'] and len(n_in['historical_hypotheses']) == 2
        checks['no_e_and_initial_jobs_only_tier_%s' % tier] = not re.search(r'"e"\s*:|e_scores|"e_by_seed"', txt_o + txt_n) and list(cen['jobs_in_order']) == list(INITIAL_JOBS) \
            and all(r.split('/')[0] in ('hist_source', 'hist_select') for r in cen['legal_evidence_refs']) and not any('RD02_L' in r for r in cen['legal_evidence_refs'])
        notes['bytes_tier_%s' % tier] = {'O': payload_bytes(SLOW_O_FORM, o_payload(cen, summ)), 'N': payload_bytes(SLOW_N_FORM, n_payload(n_in))}
    # card text appears exactly once in the N input (in the table), never inside a trajectory row
    hyp = n_in['historical_hypotheses']
    checks['old_card_text_once_in_n_input'] = all(json.dumps(n_in['jobs']).count(json.dumps(v['text'])[1:-1][:80]) == 0 for v in hyp.values()) and all(len(v['loaded_by']) == 2 for v in hyp.values())
    checks['p_source_and_summary'] = summ['P_source'] in PUBLIC and set(summ['J_source']) == set(PUBLIC) and summ['pooled']['pairs_total'] == 36 and all(v['same_direction_rate']['of_pairs'] == 6 for v in summ['per_job'].values())
    notes['P_source'] = {'P_source': summ['P_source'], 'J_source': summ['J_source'], 'pooled': summ['pooled']}
    # 2. formation independence (no other call output in any payload); rule refs + 6000 chars; the N core text is the task's text and is what the system prompt carries
    po, pn = o_payload(cen, summ), n_payload(n_in)
    checks['no_other_call_output_in_payloads'] = 'first_exploration_call_output' not in po and 'first_exploration_call_output' not in pn and 'A1' not in json.dumps(list(pn)) and SLOW_O_FORM == L.SLOW_A1 and 'first exploration call' not in SLOW_O_FORM
    core_sentences = ['Historical cards are hypotheses being evaluated, not instructions for you.', 'Agreement across training seeds is not evidence of agreement across future time periods.',
                      'A short, uniform or stopping Workflow is legitimate; so is conditional construction. Material complexity or diversity is not an objective.',
                      'Do not handle it merely by appending a generic warning while claiming the rule has been validated.']
    checks['n_core_text_in_both_n_prompts'] = all(s in SLOW_N_FORM and s in SLOW_N_REV for s in core_sentences) and 'decision_rules' in SLOW_N_FORM and 'expected_trigger_scenario' in SLOW_N_REV
    refs = cen['legal_evidence_refs']
    long_wf = 'Observe first. ' + ('Compare the public references before any construction and stop when the paired comparison is inside seed noise. ' * 40)
    card = make_card(skill_id=skill_id_of('N1', 'W1'), revision=1, workflow=long_wf, principles=None, summary='smoke', refs=[refs[0]], legal_refs=refs, mode='smoke long card')
    rendered = dsk.skill_knowledge(card).render({'batch_median_missing_fraction': 0.1}, ALLOWED_FEATURES)['loaded']
    checks['body_6000_optin_injected_prefixed_ids'] = 1200 < len(card.rendered_body) <= BODY_LIMIT and rendered == [{'hook': 'experiment_guidance', 'body': card.rendered_body}] and card.skill_id == 'RD02-N1_W1-r1' \
        and arm_of(card.skill_id) == 'cand_N1_W1r1' and line_of(card.skill_id) == 'N' and call_of(card.skill_id) == 'N1' and dsk.BODY_LIMIT == 1200
    try:
        make_card(skill_id=skill_id_of('O1', 'W2'), revision=1, workflow='x' * 6001, principles=None, summary='x', refs=[refs[0]], legal_refs=refs, mode='too long')
        checks['body_over_6000_rejected'] = False
    except ValueError:
        checks['body_over_6000_rejected'] = True
    # 3. parsers: O / N formation (prefix, KEEP, rules refs, supported-without-refs rejected), N revision KEEP / REVISE / identical refused, alias merge, per-line parents, strict aliases
    rule = {'rule': 'Compare the references first', 'observable_condition': 'always', 'change_vs_default': 'none', 'support_refs': [refs[0]], 'counter_refs': [], 'evidence_status': 'supported_in_these_cases',
            'counterexample_handling': 'none observed', 'predicted_behavior_and_failure': 'compares first; refuted if it builds first'}
    hyp_ = {'vs_default': 'a', 'contradicting_outcome': 'b', 'when_indistinguishable': 'c'}
    good_n = {'decision': 'PROPOSE', 'candidates': [{'candidate_id': 'W1', 'research_mode': 'a', 'workflow': 'Compare the references first; then one uniform ablation.', 'principles': None, 'observable_applicability': {'const': True},
                                                      'applicability_summary': 's', 'evidence_refs': [refs[0]], 'decision_hypothesis': hyp_, 'rationale': 'r', 'evidence_review': {'decision_rules': [rule]}}]}
    good_o = {'decision': 'PROPOSE', 'candidates': [{'candidate_id': 'W1', 'research_mode': 'a', 'workflow': 'Compare the references first; keep the leading public reference.', 'principles': None, 'observable_applicability': {'const': True},
                                                      'applicability_summary': 's', 'evidence_refs': [refs[0]], 'rationale': 'r', 'evidence_review': {'supports': ['x'], 'contradicts_or_limits': [], 'rule_changes': [], 'uncertain_and_expected_behavior_change': 'u'}}]}
    pn1 = parse_formation(good_n, contract='N', call='N1', legal_refs=frozenset(refs))
    po1 = parse_formation(good_o, contract='O', call='O2', legal_refs=frozenset(refs))
    rej = 0
    bad1 = copy.deepcopy(good_n); bad1['candidates'][0]['evidence_review']['decision_rules'][0]['support_refs'] = []               # supported without refs
    bad2 = copy.deepcopy(good_n); bad2['candidates'][0]['evidence_review']['decision_rules'][0]['counter_refs'] = ['practice/x/RD02_L1:1']   # ref outside legal ranges
    bad3 = copy.deepcopy(good_n); del bad3['candidates'][0]['decision_hypothesis']                                                 # missing hypothesis
    bad4 = copy.deepcopy(good_o); bad4['candidates'][0]['candidate_id'] = 'W3'
    bad5 = copy.deepcopy(good_n); bad5['candidates'][0]['principles'] = 'p'
    for b in (bad1, bad2, bad3, bad4, bad5):
        try:
            parse_formation(b, contract='N' if b is not bad4 else 'O', call='N1', legal_refs=frozenset(refs))
        except ValueError:
            rej += 1
    keep_n = parse_formation({'decision': 'KEEP', 'rationale': 'r', 'evidence_review': {'decision_rules': []}}, contract='N', call='N2', legal_refs=frozenset(refs))
    checks['formation_parsers'] = pn1['skills'][0].skill_id == 'RD02-N1_W1-r1' and po1['skills'][0].skill_id == 'RD02-O2_W1-r1' and pn1['meta']['RD02-N1_W1-r1']['decision_hypothesis'] == {k: hyp_[k] for k in sorted(hyp_)} \
        and rej == 5 and keep_n['decision'] == 'KEEP' and not keep_n['skills']
    parent = pn1['skills'][0]
    keep_r = parse_revision({'decision': 'KEEP', 'rationale': 'r', 'evidence_review': {'decision_rules': [rule]}}, contract='N', parent=parent, legal_refs=frozenset(refs))
    child_ok = {'decision': 'REVISE', 'child': {'workflow': parent.workflow + ' Stop after two new plans.', 'principles': 'Prefer references when deltas are mixed.', 'changed_mechanism': 'stop rule',
                                                'expected_trigger_scenario': 'mixed deltas', 'research_mode': 'b', 'applicability_summary': 's', 'evidence_refs': [refs[0]], 'decision_hypothesis': hyp_, 'rationale': 'r',
                                                'evidence_review': {'decision_rules': [rule]}}}
    rv = parse_revision(child_ok, contract='N', parent=parent, legal_refs=frozenset(refs))
    same_c = copy.deepcopy(child_ok); same_c['child']['workflow'] = parent.workflow; same_c['child']['principles'] = None
    try:
        parse_revision(same_c, contract='N', parent=parent, legal_refs=frozenset(refs))
        same_rejected = False
    except ValueError:
        same_rejected = True
    rv_o = parse_revision({'decision': 'KEEP', 'rationale': 'r', 'evidence_review': good_o['candidates'][0]['evidence_review'], 'counterexample_responses': ['c']}, contract='O', parent=po1['skills'][0], legal_refs=frozenset(refs))
    checks['revision_parsers'] = keep_r['decision'] == 'KEEP' and keep_r['skill'] is None and rv['skill'].skill_id == 'RD02-N1_W1-r2' and rv['skill'].principles and rv['skill'].derived_from == 'RD02-N1_W1-r1' \
        and rv['meta']['expected_trigger_scenario'] == 'mixed deltas' and same_rejected and rv_o['decision'] == 'KEEP'
    # alias merge across calls: O2 identical to O1 -> alias; per-line NO_PROPOSAL when a line only KEEPs
    recs = {'O1': {'skills': {po1['skills'][0].skill_id.replace('O2', 'O1'): {**po1['skills'][0].to_json(), 'skill_id': 'RD02-O1_W1-r1'}}, 'meta': {'RD02-O1_W1-r1': {}}},
            'N1': {'skills': {}, 'meta': {}}, 'N2': {'skills': {}, 'meta': {}}, 'O2': {'skills': {po1['skills'][0].skill_id: po1['skills'][0].to_json()}, 'meta': {po1['skills'][0].skill_id: {}}}}
    sk, al, _ = merge_formation(recs)
    checks['alias_merge_and_no_proposal_line'] = list(sk) == ['RD02-O1_W1-r1'] and al == {'RD02-O2_W1-r1': 'RD02-O1_W1-r1'} and not [s for s in sk if line_of(s) == 'N']
    checks['practice_order_interleaved'] = practice_arms({'lines': {'O': {'candidates': ['RD02-O1_W1-r1', 'RD02-O1_W2-r1', 'RD02-O2_W1-r1']}, 'N': {'candidates': ['RD02-N1_W1-r1', 'RD02-N2_W1-r1']}}}) \
        == ['RD02-O1_W1-r1', 'RD02-N1_W1-r1', 'RD02-O1_W2-r1', 'RD02-N2_W1-r1', 'RD02-O2_W1-r1']
    # strict alias in select_options: a child identical to the other line's parent is aliased, a KEEP child is not run
    cands_s = {'skills': {'RD02-O1_W1-r1': po1['skills'][0].to_json() | {'skill_id': 'RD02-O1_W1-r1'}, 'RD02-N1_W1-r1': parent.to_json()}}
    parents_s = {'lines': {'O': {'parent': 'RD02-O1_W1-r1'}, 'N': {'parent': 'RD02-N1_W1-r1'}}}
    children_s = {'children': {'RD02-O1_W1-r1-KEEP': {'skill': None, 'parent': 'RD02-O1_W1-r1', 'alias_of_parent': True}, 'RD02-N1_W1-r2': {'skill': rv['skill'].to_json(), 'parent': 'RD02-N1_W1-r1'}}}
    opts, aliases = select_options(cands_s, parents_s, children_s)
    twin = copy.deepcopy(children_s); twin['children']['RD02-N1_W1-r2']['skill'] = po1['skills'][0].to_json() | {'skill_id': 'RD02-N1_W1-r2'}
    opts2, aliases2 = select_options(cands_s, parents_s, twin)
    checks['select_options_keep_and_strict_alias'] = list(opts) == ['no_skill', 'cand_O1_W1r1', 'cand_N1_W1r1', 'cand_N1_W1r2'] and not aliases and list(opts2) == ['no_skill', 'cand_O1_W1r1', 'cand_N1_W1r1'] \
        and aliases2 == {'cand_N1_W1r2': 'cand_O1_W1r1'}
    # 4. freeze rules: fixed inside the adoption set; ties prefer fixed; card wins when lower; child wins its line when lower; a line without cards; isolated card still runs when the system picks a fixed program
    lc = {'O': {'parent': 'RD02-O1_W1-r1', 'child': None}, 'N': {'parent': 'RD02-N1_W1-r1', 'child': 'RD02-N1_W1-r2'}}
    costs = {'no_skill': {'mean_logical_cells': 18, 'mean_tokens': 100}, 'cand_O1_W1r1': {'mean_logical_cells': 15, 'mean_tokens': 80}, 'cand_N1_W1r1': {'mean_logical_cells': 15, 'mean_tokens': 60}, 'cand_N1_W1r2': {'mean_logical_cells': 12, 'mean_tokens': 50}}
    pubJ = {'None': 1.0, 'FixedMixup': 1.2, 'P_AmpResample': 1.05, 'P_NoMixRecipe': 0.56}
    f1 = apply_freeze_rules({'no_skill': 0.82, 'cand_O1_W1r1': 0.78, 'cand_N1_W1r1': 0.70, 'cand_N1_W1r2': 0.60}, pubJ, costs, {}, lc)
    f2 = apply_freeze_rules({'no_skill': 0.82, 'cand_O1_W1r1': 0.78, 'cand_N1_W1r1': 0.70, 'cand_N1_W1r2': 0.50}, pubJ, costs, {}, lc)
    f3 = apply_freeze_rules({'no_skill': 0.56, 'cand_O1_W1r1': 0.56, 'cand_N1_W1r1': 0.70, 'cand_N1_W1r2': 0.70}, pubJ, costs, {}, lc)
    f4 = apply_freeze_rules({'no_skill': 0.50, 'cand_O1_W1r1': 0.50, 'cand_N1_W1r1': 0.50, 'cand_N1_W1r2': 0.50}, pubJ, costs, {}, lc)
    f5 = apply_freeze_rules({'no_skill': 0.82, 'cand_N1_W1r1': 0.70, 'cand_N1_W1r2': 0.70}, pubJ, costs, {}, {'O': {'parent': None, 'child': None}, 'N': lc['N']})
    f6 = apply_freeze_rules({'no_skill': 0.82, 'cand_O1_W1r1': 0.78}, {'None': 1.0, 'FixedMixup': 1.2}, costs, {}, lc)
    checks['freeze_fixed_in_adoption_set_and_isolated_card'] = f1['status'] == 'FROZEN' and f1['Fixed_dev']['public_id'] == 'P_NoMixRecipe' and f1['H_O']['choice'] == 'P_NoMixRecipe' and f1['H_N']['choice'] == 'P_NoMixRecipe' \
        and f1['W_N']['skill_id'] == 'RD02-N1_W1-r2' and f1['W_N']['is_child'] and f1['W_O']['skill_id'] == 'RD02-O1_W1-r1'
    checks['freeze_card_beats_fixed_and_ties'] = f2['H_N']['choice'] == 'cand_N1_W1r2' and f2['H_N']['kind'] == 'card' and f2['H_O']['choice'] == 'P_NoMixRecipe' \
        and f3['H_O']['choice'] == 'P_NoMixRecipe' and f3['W_N']['skill_id'] == 'RD02-N1_W1-r1' and not f3['W_N']['is_child'] \
        and f4['W_N']['skill_id'] == 'RD02-N1_W1-r1' and f4['H_N']['choice'] == 'cand_N1_W1r1' and f4['H_O']['choice'] == 'cand_O1_W1r1' and 'P_NoMixRecipe' not in f4['H_N']['tied']
    checks['freeze_line_without_card_and_incomplete'] = f5['W_O'] is None and f5['lines']['O']['card_comparison'] == 'NOT_AVAILABLE' and f5['H_O']['choice'] == 'P_NoMixRecipe' and f5['W_N']['skill_id'] == 'RD02-N1_W1-r1' \
        and f6['status'] == 'SELECT_INCOMPLETE'
    # replay orders (pure planning; no stage, no file): no F_O when line O has no card; H fixed -> the card still runs; alias when identical; refusal without FROZEN
    frozen_fake = {**f5, 'skills': {'RD02-N1_W1-r1': parent.to_json()}}
    plan = replay_plan(frozen_fake, 'P_NoMixRecipe')
    cat_ = plan['catalog']
    checks['replay_orders_no_f_o_isolated_f_n'] = cat_['orders']['RD02_L5'] == ['f0', 'f_n', 'random', 'menu_ca'] and cat_['orders']['RD02_L6'] == ['f_n', 'random', 'f0', 'menu_ca'] \
        and cat_['treatment_note']['f_o']['status'] == 'NO_TREATMENT' and cat_['treatment_note']['h_n']['kind'] == 'fixed_public' and cat_['treatment_note']['h_n']['candidate_test_isolated'] \
        and cat_['exposure'] == EXPOSURE_TAG and plan['knowledge']['f_o'] is None and plan['knowledge']['f_n']['version'] == 1
    twin_frozen = {**f1, 'skills': {'RD02-O1_W1-r1': parent.to_json() | {'skill_id': 'RD02-O1_W1-r1'}, 'RD02-N1_W1-r2': parent.to_json() | {'skill_id': 'RD02-N1_W1-r2'}}}
    plan2 = replay_plan(twin_frozen, 'P_NoMixRecipe')
    checks['replay_identical_cards_alias_once'] = plan2['same'] and plan2['order']['RD02_L5'] == ['f0', 'f_o', 'random', 'menu_ca'] and plan2['note']['f_n']['status'] == 'ALIAS' and plan2['note']['h_n']['alias_of_arm'] is None
    try:
        replay_plan({'status': 'SELECT_INCOMPLETE'}, 'P_NoMixRecipe')
        checks['replay_refused_without_frozen_choices'] = False
    except RuntimeError:
        checks['replay_refused_without_frozen_choices'] = True
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            run_stage(Path(td), 'test', Path(td) / 'x.json')
            checks['stage_refused_outside_package_root'] = False
        except RuntimeError:
            checks['stage_refused_outside_package_root'] = True
        checks['stage_refused_outside_package_root'] = checks['stage_refused_outside_package_root'] and not list(Path(td).glob('**/stage_configs'))
    # 5. copied cache carries models and C_A only (no labels, no other arm's candidates by name); a learning failure never starts the next stage
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td = Path(td)
        src = PARENT_LL / 'practice' / 'RD02_L1_common' / 'RD02_L1'
        rec = W.copy_physical_cache(src, td / 'common', 'RD02_L1', seeds=SEEDS)
        files = [p.name for p in (td / 'common').rglob('*')]
        cells = [context.read_json(p) for p in (td / 'common' / 'RD02_L1' / 'cells').glob('*.json')]
        reg = W.load_registry(td / 'common' / 'RD02_L1')
        checks['copied_cache_models_c_a_only_no_plan_names'] = rec['copied'] and rec['cells'] >= 12 and not any(n in ('c_b_scores.json', 'e_scores.json', 'e_frozen.json', 'commit.json') for n in files) \
            and all(set(c['scores']) == {'c_a'} and Path(c['model_path']).exists() for c in cells) and all(m in PUBLIC or m.startswith('TA') for m in reg)
    checks['learning_failure_stops_package'] = all(s != 'PROPOSED' for s in ('FORMATION_TECHNICAL_FAILURE', 'NO_PROPOSAL_BOTH_LINES')) and 'PRACTICE_INCOMPLETE' != 'PARENTS_KEPT' and 'REVISION_TECHNICAL_FAILURE' != 'REVISED'
    checks['budget_table_matches_task'] = sum(a['fits'] for a in ALLOC.values()) == 543 and sum(a['requests'] for a in ALLOC.values()) == 572 and sum(a['tokens'] for a in ALLOC.values()) == 13_200_000 \
        and TOTAL['max_fit_attempts'] == 549 and HTTP_CAP == 576 and TOTAL['max_wall_s'] == 8 * 3600
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
    global ACCEPT_UNKNOWN_USAGE
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--parent-smoke', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--resume-stage', choices=['practice', 'select', 'test'])
    ap.add_argument('--resume-revision', action='store_true', help='send the pending contract correction of a revision refused by the reservation (after an approved cap amendment), then continue the package')
    ap.add_argument('--amend-revision-cap', type=int, default=0, help='operator-approved extra tokens on the revision stage cap (recorded as frozen_config_amendment_N.json)')
    ap.add_argument('--amend-reason', default='')
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--restart', default='')
    ap.add_argument('--continue-package', action='store_true')
    ap.add_argument('--root', type=Path, default=ROOT)
    a = ap.parse_args()
    root = a.root.resolve()
    if a.resume_revision:
        ACCEPT_UNKNOWN_USAGE = bool(a.accept_unknown_usage)
        if a.amend_revision_cap:
            print(json.dumps(amend_stage_cap(root, 'revision', a.amend_revision_cap, a.amend_reason, 'user (chat decision)'), ensure_ascii=False))
        cands = context.read_json(paths(root)['formation_a'] / 'candidates.json')
        parents = context.read_json(paths(root)['formation_a'] / 'parents.json')
        rec = resume_revision(root, cands, parents)
        print('REVISION_RESUMED', rec['status'], flush=True)
        if a.continue_package and rec['status'] == 'REVISED':
            package_run(root)
    elif a.resume_stage:
        resume_stage(root, a.resume_stage, accept_unknown_usage=a.accept_unknown_usage, restart=[x for x in a.restart.split(',') if x])
        if a.continue_package:
            package_run(root)
    elif a.smoke:
        rec = smoke(root / 'smoke')
        if rec['status'] != 'PASS':
            raise SystemExit(1)
    elif a.parent_smoke:
        rec = L.smoke(root / 'smoke' / 'parent_learning_loop_smoke')
        if rec['status'] != 'PASS':
            raise SystemExit(1)
    elif a.preflight:
        print(json.dumps({k: v for k, v in preflight(root).items() if k in ('frozen_local', 'environment_matches_parent', 'replay_exposure')}, ensure_ascii=False)[:3000])
    elif a.wiring:
        wiring(root)
    elif a.run:
        ACCEPT_UNKNOWN_USAGE = bool(a.accept_unknown_usage)
        package_run(root)
    elif a.result:
        readout(root)
    else:
        ap.error('choose an action')


if __name__ == '__main__':
    from evaluation.main_protocol_p4 import batch_research_tempo_aug_decision_learning as _canonical
    _canonical.main()
