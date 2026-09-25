"""DEV-BATCH-RESEARCH-EXPLORATION-QUOTA: thin study module for run_batch_research_roundtrip --study exploration_quota.

Same frozen draft flow as the draft_construction study (four unscored pool drafts per job, T only, before any fit),
plus an independent frozen stream that permutes the four draft ids: Q = pi[0] is the random exploration probe.
Five arms on G4 (CSV string-sorted names [128:160]) at a65/a85, empty H0, no Slow; every Fast arm may only build
pool / baseline assignments (DraftAdapter select_only):
  F_auto   Q is a public random marker only; no suggestion, no gate
  F_text   the core rule is given as a suggestion that may be declined
  F_quota  the core rule is this arm's contract, executed by the Runtime: a second non-Q new evaluation before Q,
           or any commit before Q is fully evaluated, is rejected as a public tool input error (correction budget)
  Random   evaluates pi[0], pi[1]; argmin three-seed C_A (Fixed, None, pi[0], pi[1] tie order); 0 LLM
  Menu     fixed U-Mixup 0.25 / FreqMask 0.10; argmin C_A; 0 LLM
The quota is a researcher-fixed process intervention of this study only; it is not Slow output and grants Slow nothing.
"""
from __future__ import annotations
import functools
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from methods.ttha import batch_research as br
from methods.ttha.batch_base import context, policy, spec
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import run_batch_research_v1 as base
from evaluation.main_protocol_p4 import batch_research_source_process as sp
from evaluation.main_protocol_p4 import batch_research_draft_construction as dc

REPO = Path(__file__).resolve().parents[2]
CONTRACTS = dc.CONTRACTS
SEEDS = (20261001, 20261002, 20261003)
ARMS = ('f_auto', 'f_text', 'f_quota', 'random', 'menu')
FAST_ARMS = ('f_auto', 'f_text', 'f_quota')
POOL_ARMS = ('f_auto', 'f_text', 'f_quota', 'random')
JOBS = ('a65_g4', 'a85_g4')
ORDER = {'a65_g4': ('f_auto', 'f_text', 'f_quota', 'random', 'menu'),
         'a85_g4': ('menu', 'random', 'f_quota', 'f_text', 'f_auto')}
G4 = ('213', '214', '215', '216', '217', '218', '219', '22', '220', '221', '222', '223', '224', '225', '226', '227',
      '228', '229', '23', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '24', '240', '241')
ROSTERS = {'a65_g4': G4, 'a85_g4': G4}
MENU, MENU_TIE_ORDER = dc.MENU, dc.MENU_TIE_ORDER
PROPOSAL_SEED_BASE, PROBE_SEED_BASE = 2026091500, 2026091600
CORE_RULE = ('Within the two new-candidate evaluation slots, make sure the random probe Q (overview.random_probe_id) is actually '
             'evaluated; at most one other slot is yours to choose. You may evaluate Q first or another draft first. Q carries no '
             'quality advantage; the final choice still follows the current legal evidence, and None / FixedMixup may be delivered.')
PROBE_NOTE = ('random_probe_id names one of the four unscored drafts, drawn by a frozen independent random stream after the drafts were '
              'fixed. It is a random marker only: no evidence, no confidence, no quality order, no recommended operator.')


# ----------------------------------------------------------------------------- frozen streams
def generate_pool(job, table):
    return dc.generate_pool(job, table, job_index=JOBS.index(job), seed_base=PROPOSAL_SEED_BASE)


def draw_probe(job, pool):
    """Independent frozen stream over the fixed draft ids: pi = permutation, Q = pi[0]; Random evaluates pi[0], pi[1]."""
    if pool['status'] != 'COMPLETE':
        return None
    job_index = JOBS.index(job)
    seed = PROBE_SEED_BASE + 1000 * job_index
    ids = [d['draft_id'] for d in pool['drafts']]
    pi = [ids[i] for i in np.random.RandomState(seed).permutation(len(ids))]
    by_id = {d['draft_id']: d for d in pool['drafts']}
    return {'job': job, 'seed': seed, 'rule': 'rng = RandomState(%d + 1000*job_index); pi = draft ids in rng.permutation(4) order; Q = pi[0]; Random evaluates pi[0], pi[1]' % PROBE_SEED_BASE,
            'pi': pi, 'q': pi[0], 'q_key': policy.assignment_key(by_id[pi[0]]['assignment']), 'q_policy': by_id[pi[0]]['policy'],
            'meaning': 'random exploration position; not confidence, quality, distance or a recommended operator'}


class QuotaAdapter(dc.DraftAdapter):
    """DraftAdapter (pool-only admission for every Fast arm of this study) + the public random probe in overview.
    For arm == 'f_quota' the two execution checks of task §6 run inside run_job before any slot count, fit or commit."""
    def __init__(self, run_dir, job, ledger, repo, *, pool, arm, **kw):
        super().__init__(run_dir, job, ledger, repo, pool=pool, select_only=True, **kw)
        if arm not in FAST_ARMS or pool.get('probe') is None or pool['probe']['job'] != self.ctx.job.job_id:
            raise ValueError('quota adapter needs a Fast arm and this job\'s frozen probe')
        self.arm, self.probe = arm, pool['probe']
        self.baseline_ids = {'None', 'FixedMixup'}

    def overview(self):
        ov = super().overview()
        ov['random_probe_id'] = self.probe['q']
        ov['random_probe_note'] = PROBE_NOTE
        return ov

    # --- quota state, derived from the real fitted candidates of this adapter (also after restore) ---
    def _key(self, plan_id):
        return policy.assignment_key(self.specs[plan_id]['assignment'])

    def q_complete(self):
        return any(self._key(pid) == self.probe['q_key'] and len(c.model_refs) == len(self.seeds) and all(c.model_refs) for pid, c in self.fitted.items() if pid in self.specs)

    def non_q_new(self):
        return sorted(pid for pid, c in self.fitted.items() if pid not in self.baseline_ids and pid in self.specs and self._key(pid) != self.probe['q_key'] and len(c.model_refs) == len(self.seeds))

    def _reject(self, message, **details):
        if self.tool_error_feedback:
            raise br.ToolInputError(message, random_probe_id=self.probe['q'], q_evaluated=self.q_complete(), non_q_new_candidates=self.non_q_new(), **details)
        raise ValueError(message)

    def check_evaluate(self, plan_id, candidate):
        if self.arm != 'f_quota' or self.q_complete() or self._key(plan_id) == self.probe['q_key']:
            return
        if self.non_q_new():
            self._reject('quota: the random probe %s is not yet evaluated and one non-Q new candidate already used a slot; the remaining new '
                         'evaluation must be Q. Build a plan with Q\'s policy under any plan id and evaluate it (an identical assignment counts), '
                         'or commit becomes available only after Q completes.' % self.probe['q'], legal_next=['build_material with the policy of ' + self.probe['q'], 'evaluate that plan'])

    def check_commit(self, plan_id, candidate):
        if self.arm != 'f_quota' or self.q_complete():
            return
        self._reject('quota: commit is refused until the random probe %s is fully evaluated (all paired seeds). Evaluate Q first (build it with its '
                     'policy under any plan id; an identical assignment counts), then commit any evaluated candidate including None / FixedMixup.' % self.probe['q'],
                     legal_next=['build_material with the policy of ' + self.probe['q'], 'evaluate that plan', 'then commit'])


def arm_contracts(arm):
    """Shared contracts + pool-only admission for every Fast arm; the accurate commit sentence for all three; the arm's own rule."""
    out = dc.arm_contracts('f_select')
    out['commit']['evaluation_requirement'] = ('The common baselines (None, FixedMixup) are always legal outputs, but a commit must satisfy this arm\'s publicly '
                                               'stated evaluation requirement: ' + {'f_auto': 'none for this arm.', 'f_text': 'none enforced for this arm (see the suggestion under evaluate.arm_rule).',
                                                                                    'f_quota': 'the random probe Q must be fully evaluated before any commit (see evaluate.arm_rule).'}[arm])
    if arm == 'f_auto':
        out['evaluate']['arm_rule'] = 'overview.random_probe_id is a random marker only; this arm has no evaluation suggestion and no gate attached to it.'
    elif arm == 'f_text':
        out['evaluate']['arm_rule'] = 'SUGGESTION for this arm (not enforced; you may decline it): ' + CORE_RULE
    elif arm == 'f_quota':
        out['evaluate']['arm_rule'] = ('CONTRACT of this arm, executed by the Runtime: ' + CORE_RULE + ' Enforcement: after one non-Q new evaluation, a further non-Q new '
                                       'evaluation before Q is rejected; any commit before Q is fully evaluated is rejected. Rejections are tool input errors within the '
                                       'correction budget; a rejected action and the rest of that response are not executed, and no slot or fit is consumed by them.')
    else:
        raise ValueError(arm)
    return out


# ----------------------------------------------------------------------------- frozen config, preparation
def frozen_config(caps):
    jobs = {}
    for job in JOBS:
        js = context.resolve_job('electricity', job, roster=ROSTERS[job])
        jobs[job] = {'t': js.t, 'train_rows': list(js.train_range), 'c_a': list(js.c_a), 'c_b': list(js.c_b), 'e': list(js.e),
                     'workload_end_row_exclusive': max(js.e) + spec.H, 'roster': list(js.roster), 'roster_explicit': True}
    return {'study': 'exploration_quota', 'jobs': jobs, 'seeds': list(SEEDS), 'order': {j: list(o) for j, o in ORDER.items()}, 'arms': list(ARMS),
            'pool_arms': list(POOL_ARMS), 'fast_arms': list(FAST_ARMS),
            'draft_pool': {'n_drafts': dc.N_DRAFTS, 'max_proposals': dc.MAX_PROPOSALS, 'seed_base': PROPOSAL_SEED_BASE,
                           'rule': dc.DRAFT_RULE.replace(str(dc.PROPOSAL_SEED_BASE), str(PROPOSAL_SEED_BASE)), 'generator': dc.GENERATOR_NOTE,
                           'dedup': 'expanded assignment equality against None, FixedMixup and earlier drafts; no distance, complexity, proxy or quality filter',
                           'frozen_before': 'any fit of the package; both pools and probes generated from T only; C_A never opened for generation'},
            'probe': {'seed_base': PROBE_SEED_BASE, 'rule': 'rng = RandomState(%d + 1000*job_index); pi = draft ids in rng.permutation(4) order; Q = pi[0]' % PROBE_SEED_BASE,
                      'meaning': 'random exploration position drawn after the drafts were fixed; not confidence, quality, distance or a recommended operator; visible to every Fast arm before the run'},
            'quota': {'applies_to': 'f_quota only', 'rule': CORE_RULE,
                      'execution': ['build/inspect never consume a slot', 'first new evaluation may be Q or another pool draft',
                                    'with Q incomplete and one non-Q new candidate fitted, a further non-Q new evaluation is rejected before slot count and fit',
                                    'with Q incomplete, any commit (including None/FixedMixup) is rejected before the delivery is bound or written',
                                    'Q identity by compiled assignment (rename is fine; baselines never satisfy Q); Q complete = all paired seeds fitted',
                                    'rejections are ToolInputError under the 2-correction budget; nothing is auto-fitted or auto-committed'],
                      'origin': 'researcher-fixed process intervention of this study; not Slow output; grants Slow nothing'},
            'arm_contracts': {a: arm_contracts(a) for a in FAST_ARMS}, 'core_rule': CORE_RULE, 'probe_note': PROBE_NOTE,
            'menu': {'plans': [[m, p] for m, p in MENU], 'rule': 'argmin three-seed C_A mean among FixedMixup, None, MenuU, MenuFM; ties in that order; 0 LLM'},
            'random': {'plans': 'pi[0], pi[1] of the job probe', 'rule': 'argmin three-seed C_A mean among FixedMixup, None, pi[0], pi[1]; ties in that order; 0 LLM'},
            'caps': caps, 'planned_fits': 72, 'http_cap': 2 * caps['max_llm_requests'], 'exposure': 'EXPOSED_DEVELOPMENT',
            'population_note': 'one 32-entity group G4 (CSV string-sorted names [128:160]) at two disjoint times a=.65/.85; the next whole group after G3; same source, not independent replications',
            'knowledge': 'H0 empty for every Fast arm; no Slow; no historical card, generic note, construction seed or old drafts',
            'per_fast_job_limits': {**asdict(br.Limits()), 'wall_seconds': 'remaining package wall'}, 'max_tool_corrections': 2, 'evidence_roundtrip': False,
            'tool_contracts': CONTRACTS, 'tool_semantics': rt.TOOL_SEMANTICS, 'consumer': spec.CONSUMER,
            'environment': dc.environment(),
            'unknown_usage_rule': 'a failed transport attempt or a response without usage leaves cost UNKNOWN; at most one same-request retransmission; later paid calls refused; executor may not accept unknown usage',
            'utility_rule': ('main: Delta_text = loss(F_text) - loss(F_quota); necessary: F_auto/Random/Menu/None/FixedMixup - F_quota; auxiliary: F_auto - F_text; '
                             'three-seed E means per job, positive = F_quota better; C_A/C_B intermediate; first-seed delivery separate; no post-hoc significance gate.')}


def prepare(root, led, caps):
    """Stage 2: both Targets' T only (eligibility, roster binding), both draft pools and probes frozen before any fit."""
    context.write_json(root / 'frozen_config.json', frozen_config(caps))
    if led.s['fit_attempts']:
        raise RuntimeError('draft pools must be frozen before any fit')
    commons, pools = {}, {}
    for job in JOBS:
        shared = root / (job + '_common')
        shared.mkdir()
        try:
            adapter = rt.Adapter(shared, job, led, REPO, roster=ROSTERS[job], seeds=SEEDS)
        except RuntimeError as exc:
            if 'ELIGIBILITY_FAIL' not in str(exc):
                raise
            context.write_json(root / (job + '_eligibility.json'), {'job': job, 'status': 'ELIGIBILITY_FAILED', 'reason': str(exc), 'note': 'no population or time change; the other job continues'})
            pools[job] = None
            continue
        commons[job] = adapter
        ov = adapter.overview()
        dc.target_checks(root, job, adapter)
        pool = generate_pool(job, ov['entities'])
        context.write_json(root / (job + '_draft_pool.json'), pool)
        probe = draw_probe(job, pool)
        context.write_json(root / (job + '_probe.json'), probe if probe is not None else {'job': job, 'status': pool['status'], 'q': None})
        pools[job] = {**pool, 'probe': probe}
    if led.s['fit_attempts']:
        raise RuntimeError('a fit happened during draft generation')
    context.write_json(root / 'drafts_frozen.json', {'epoch': time.time(), 'jobs': {j: (p['status'] if p else 'ELIGIBILITY_FAILED') for j, p in pools.items()},
                                                    'probes': {j: (p['probe']['q'] if p and p.get('probe') else None) for j, p in pools.items()},
                                                    'fit_attempts_at_freeze': led.s['fit_attempts'], 'note': 'both pools and probes generated from T only before any fit; no C_A opened'})
    return commons, pools


def load_pools(root):
    out = {}
    for job in JOBS:
        pf, qf = root / (job + '_draft_pool.json'), root / (job + '_probe.json')
        if not pf.exists():
            out[job] = None
            continue
        pool = context.read_json(pf)
        probe = context.read_json(qf) if qf.exists() else None
        out[job] = {**pool, 'probe': probe if probe and probe.get('q') else None}
    return out


# ----------------------------------------------------------------------------- arms
def adapter_factory(arm, pool):
    return functools.partial(QuotaAdapter, pool=pool, arm=arm)


def run_arm(arm, path, job, led, client, shared, pool, *, seeds, roster, max_tool_corrections):
    if arm in FAST_ARMS:
        return base.branch(path, job, br.Knowledge(), led, client, shared, evidence_roundtrip=False, max_tool_corrections=max_tool_corrections,
                           tool_contracts=arm_contracts(arm), seeds=seeds, roster=roster, adapter_factory=adapter_factory(arm, pool))
    if arm == 'random':
        by_id = {d['draft_id']: d for d in pool['drafts']}
        pi = pool['probe']['pi']
        plans = [(m, by_id[m]['policy']) for m in pi[:2]]
        return dc.fixed_candidates_branch(path, job, led, shared, plans=plans, tie_order=('FixedMixup', 'None', pi[0], pi[1]), tag='random', seeds=seeds, roster=roster,
                                          reason='Random draft baseline: minimum C_A three-seed mean among FixedMixup, None, %s, %s (frozen probe order); frozen tie order.' % (pi[0], pi[1]),
                                          expected_assignments={m: by_id[m]['assignment'] for m in pi[:2]})
    if arm == 'menu':
        return dc.fixed_candidates_branch(path, job, led, shared, plans=list(MENU), tie_order=MENU_TIE_ORDER, tag='menu', seeds=seeds, roster=roster,
                                          reason='Fixed menu: minimum C_A three-seed mean among FixedMixup, None, MenuU, MenuFM; frozen tie order; no history read.')
    raise ValueError(arm)


def resume_arm(arm, path, job, led, client, pool, *, seeds, roster, max_tool_corrections):
    if arm not in FAST_ARMS:
        raise RuntimeError('control branch cannot be resumed')
    return base.resume_branch(path, job, br.Knowledge(), led, client, max_tool_corrections=max_tool_corrections, tool_contracts=arm_contracts(arm),
                              seeds=seeds, roster=roster, adapter_factory=adapter_factory(arm, pool))


# ----------------------------------------------------------------------------- report
def quota_events(root, name, pool):
    """Mechanism record of one Fast branch from its real tool events: which drafts were evaluated, when Q completed,
    whether Q came by the Agent's own choice or after a contract rejection, and how the two slots were used."""
    rows = [json.loads(l) for l in (root / name / 'trace.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    probe = (pool or {}).get('probe') or {}
    q_key = probe.get('q_key')
    idx = root / name / (name.split('_', 2)[0] + '_' + name.split('_', 2)[1]) / 'materials' / 'index.json'
    keys = {}
    if idx.exists():
        for mid in context.read_json(idx):
            comp = idx.parent / (mid + '__compiled.json')
            if comp.exists():
                keys[mid] = policy.assignment_key(context.read_json(comp)['assignment'])
    named = {policy.assignment_key(d['assignment']): d['draft_id'] for d in (pool or {}).get('drafts', [])}
    resp_of, current = {}, 0
    evals, rejections = [], []
    for i, x in enumerate(rows):
        if x['event'] == 'fast_response':
            current = x['number']
        if x['event'] == 'tool_completed' and x['tool'] == 'evaluate':
            pid = x['output']['plan_id']
            evals.append({'plan_id': pid, 'draft': named.get(keys.get(pid)), 'is_q': keys.get(pid) == q_key, 'response': current, 'event_index': i})
        if x['event'] == 'tool_rejected':
            msg = x['error'].get('message', '')
            rejections.append({'tool': x['tool'], 'response': current, 'quota': msg.startswith('quota:'), 'message': msg[:160], 'unexecuted': [a['tool'] for a in x.get('unexecuted_actions', [])]})
    first_q = next((e for e in evals if e['is_q']), None)
    quota_rej = [r for r in rejections if r['quota']]
    commit = next((x for x in rows if x['event'] == 'tool_completed' and x['tool'] == 'commit'), None)
    out = {'random_probe_id': probe.get('q'), 'pi': probe.get('pi'), 'evaluated': [{k: e[k] for k in ('plan_id', 'draft', 'is_q', 'response')} for e in evals],
           'new_evaluations': len({e['plan_id'] for e in evals}), 'q_evaluated': first_q is not None, 'q_response': first_q['response'] if first_q else None,
           'q_order': (evals.index(first_q) + 1) if first_q else None,
           'q_after_quota_rejection': bool(first_q and any(r['response'] < first_q['response'] for r in quota_rej)),
           'quota_rejections': len(quota_rej), 'other_rejections': len(rejections) - len(quota_rej), 'rejections': rejections,
           'second_slot_after_first_feedback': (len(evals) >= 2 and evals[1]['response'] > evals[0]['response']) if len(evals) >= 2 else None,
           'committed': commit['output']['plan_id'] if commit else None,
           'committed_draft': named.get(keys.get(commit['output']['plan_id'])) if commit and commit['output']['plan_id'] in keys else (commit['output']['plan_id'] if commit else None),
           'committed_is_q': bool(commit and keys.get(commit['output']['plan_id']) == q_key)}
    return out


def finish_report(root, r, jobs, order, seeds):
    r['study'] = 'exploration_quota'
    r.pop('treatment_present', None)
    pools = load_pools(root)
    r['draft_pools'] = {j: ({k: p[k] for k in ('status', 'n_proposals_tried', 'fits', 'materials_built')} | {'drafts': [{k: d[k] for k in ('draft_id', 'j', 'proposal_seed', 'n_rules', 'n_identity', 'n_entities_changed')} for d in p['drafts']]}) if p else None for j, p in pools.items()}
    r['probes'] = {j: ({k: p['probe'][k] for k in ('seed', 'pi', 'q')} if p and p.get('probe') else None) for j, p in pools.items()}
    r['eligibility'] = {j: (context.read_json(root / (j + '_eligibility.json')) if (root / (j + '_eligibility.json')).exists() else {'status': 'ELIGIBLE'}) for j in jobs}
    r['skipped'] = {p.name[:-13]: context.read_json(p) for p in sorted(root.glob('*_skipped.json'))}
    r['drafts_frozen'] = context.read_json(root / 'drafts_frozen.json') if (root / 'drafts_frozen.json').exists() else None
    r['resume'] = context.read_json(root / 'resume.json') if (root / 'resume.json').exists() else None
    r['target_checks'] = {j: context.read_json(root / (j + '_target_checks.json')) for j in jobs if (root / (j + '_target_checks.json')).exists()}
    dc.annotate_branches(root, r, pools, seeds)
    seeds = tuple(seeds)
    for name, b in r['branches'].items():
        b['quota'] = quota_events(root, name, pools.get(b['job'])) if b['arm'] in FAST_ARMS else None

    def delta(cells_a, cells_b, block):
        ds = [cells_a[s][block]['normalized_mse_macro'] - cells_b[s][block]['normalized_mse_macro'] for s in seeds
              if s in cells_a and s in cells_b and cells_a[s].get(block) and cells_b[s].get(block)]
        if len(ds) != 3:
            return None
        return {'delta_by_seed': ds, 'mean_delta': statistics.mean(ds), 'seed_se': statistics.stdev(ds) / 3 ** .5, 'first_seed_delta': ds[0], 'positive_means': 'second_is_better'}

    def committed_cells(b):
        return sp._seed_cells(b, b['controller']['committed_plan_id']) if b and b['controller']['status'] == 'COMPLETE' else None

    r['comparisons'] = {}
    for job in jobs:
        quota = r['branches'].get(job + '_f_quota')
        comp = {'positive_means': 'F_quota has lower loss', 'reference': 'f_quota committed plan'}
        ref = committed_cells(quota)
        if ref:
            for arm in ('f_text', 'f_auto', 'random', 'menu'):
                other = committed_cells(r['branches'].get(job + '_' + arm))
                comp[arm + '_minus_f_quota'] = {block: delta(other, ref, block) for block in ('c_a', 'c_b', 'external')} if other else None
            for b0 in ('None', 'FixedMixup'):
                comp[b0 + '_minus_f_quota'] = {block: delta(sp._seed_cells(quota, b0), ref, block) for block in ('c_a', 'c_b', 'external')}
            comp['delta_text'], comp['delta_auto'] = comp['f_text_minus_f_quota'], comp['f_auto_minus_f_quota']
            comp['delta_random'], comp['delta_menu'] = comp['random_minus_f_quota'], comp['menu_minus_f_quota']
        else:
            comp['status'] = 'UNKNOWN: F_quota branch missing or incomplete'
        auto, text = committed_cells(r['branches'].get(job + '_f_auto')), committed_cells(r['branches'].get(job + '_f_text'))
        comp['f_auto_minus_f_text'] = {block: delta(auto, text, block) for block in ('c_a', 'c_b', 'external')} if auto and text else None
        keys, lineage = {}, {}
        for arm in ARMS:
            b = r['branches'].get(job + '_' + arm)
            if b and b['controller']['status'] == 'COMPLETE':
                p = b['controller']['committed_plan_id']
                keys[arm] = (b['new_candidate_materials'].get(p, {}).get('assignment_key') if p not in ('None', 'FixedMixup') else p)
                lineage[arm] = b['committed_lineage']
        comp['committed_material_key_by_arm'], comp['committed_lineage_by_arm'] = keys, lineage
        # retrospective only: Q's E rank among every complete plan fitted by any arm of this job (no Agent saw E)
        pooled = {}
        for arm in ARMS:
            b = r['branches'].get(job + '_' + arm)
            for p, cells in (b or {}).get('plans', {}).items():
                if len(cells) == 3 and all(c.get('external') for c in cells):
                    lin = (b['new_candidate_materials'].get(p, {}).get('lineage') or {}).get('identical_to') if p not in ('None', 'FixedMixup') else p
                    pooled[arm + ':' + p] = {'e': statistics.mean(c['external']['normalized_mse_macro'] for c in cells), 'identical_to': lin}
        q = (pools.get(job) or {}).get('probe', {}) or {}
        ranked = sorted(pooled.items(), key=lambda kv: kv[1]['e'])
        comp['retrospective_best_e_across_arms'] = (ranked[0][0], ranked[0][1]['e']) if ranked else None
        comp['q_retrospective'] = {'q': q.get('q'), 'fitted_by_any_arm': any(v['identical_to'] == q.get('q') for v in pooled.values()),
                                   'e_rank_among_fitted': next((i + 1 for i, (k, v) in enumerate(ranked) if v['identical_to'] == q.get('q')), None), 'n_fitted': len(ranked)}
        r['comparisons'][job] = comp
    planned = sum(len(order[j]) for j in jobs)
    r['planned_arms'], r['skipped_arms'] = planned, len(r['skipped'])
    r['status'] = 'COMPLETE' if (len(r['branches']) == planned and not r['failures'] and all(b['controller']['status'] == 'COMPLETE' and b['delivery_e'] is not None for b in r['branches'].values())) else 'PARTIAL'
    r['utility_supported'] = None
    r['utility_scope'] = 'Researcher-fixed process intervention; one Fast trajectory per arm and job, three training seeds per complete plan, one population at two disjoint times; development pilot; no Slow.'
    context.write_json(root / 'result.json', r)
    lines = ['# DEV-BATCH-RESEARCH-EXPLORATION-QUOTA：机器汇总（result.json 派生）', '', '状态：' + r['status'], '',
             '|分支|状态|commit|谱系|Q|Q已评估|Q次序|配额拒绝|第一seed E|三seed E均值|Fast请求|工具尝试|私有拟合|集合内E遗憾|',
             '|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, b in r['branches'].items():
        q = b.get('quota') or {}
        lines.append('|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|' % (name, b['controller']['status'], b['controller']['committed_plan_id'], (b['committed_lineage'] or {}).get('identical_to'),
                                                                   q.get('random_probe_id'), q.get('q_evaluated'), q.get('q_order'), q.get('quota_rejections'),
                                                                   b['delivery_e'], b['paired_mean_e'], b['controller']['calls'], b['controller']['tool_calls'], b['private_physical_fits'], b['commit_regret_within_observed_candidates']))
    lines += ['', '各作业配对比较（正值 = F_quota 损失更低）：']
    for j, x in r['comparisons'].items():
        lines.append(j + ': ' + json.dumps({k: v for k, v in x.items() if k.startswith('delta_') or k.endswith('_minus_f_quota') or k == 'f_auto_minus_f_text' or k == 'q_retrospective'}, ensure_ascii=False))
    lines += ['', '预算：' + json.dumps(r.get('budget'), ensure_ascii=False), '', '主报告见 REPORT.md；本文件由 report() 自动生成，不含叙述判断。']
    (root / 'MACHINE_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('EXPLORATION_QUOTA_REPORT', r['status'], flush=True)
    return r
