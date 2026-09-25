"""DEV-BATCH-RESEARCH-SKILL-REVISION: thin study module for run_batch_research_roundtrip --study skill_revision.

H_old (the previous package's frozen F_source card) + a deterministic census of its actual use (ten a89/a93
branches: T-only observations, plans, tool results, C_A, post-commit C_B, costs, unknown-cost and stage-order
metadata; no E) -> one Slow revision or KEEP -> H_new frozen -> F0 / F_generic / F_old / F_new / Menu on two
later jobs that share t=25512 but use disjoint 32-entity populations (G0, G1). Census helpers, Slow client,
apply_update and the report skeleton are reused from batch_research_source_process; nothing numerical lives here.
"""
from __future__ import annotations
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from methods.ttha import batch_research as br
from methods.ttha.batch_base import context, policy, spec
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_source_process as sp

REPO = Path(__file__).resolve().parents[2]
PREV = REPO / '_scratch' / 'dev_batch_research_source_process'
CONTRACTS = sp.CONTRACTS
SEEDS = (20260925, 20260926, 20260927)
ARMS = ('f0', 'f_generic', 'f_old', 'f_new', 'menu')
JOBS = ('a97_g0', 'a97_g1')
ORDER = {'a97_g0': ('f0', 'f_generic', 'f_old', 'f_new', 'menu'),
         'a97_g1': ('menu', 'f_new', 'f_old', 'f_generic', 'f0')}
G0 = ('0', '1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '11', '110', '111',
      '112', '113', '114', '115', '116', '117', '118', '119', '12', '120', '121', '122', '123', '124', '125', '126')
G1 = ('127', '128', '129', '13', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '14', '140',
      '141', '142', '143', '144', '145', '146', '147', '148', '149', '15', '150', '151', '152', '153', '154', '155')
ROSTERS = {'a97_g0': G0, 'a97_g1': G1}
# Menu arm (task §6): the two fixed plans of the previous package's de-facto menu, deterministic argmin C_A, frozen tie order.
MENU = (('MenuU', policy.uniform_policy([{'op': 'timemixup', 'donor_rule': 'U', 'w': 0.25}], 'fixed menu: uniform hour-aligned TimeMixup w=0.25')),
        ('MenuFM', policy.uniform_policy([{'op': 'freqmask', 'mu': 0.10}], 'fixed menu: uniform FreqMask mu=0.10')))
MENU_TIE_ORDER = ('FixedMixup', 'None', 'MenuU', 'MenuFM')
USAGE_BRANCHES = tuple((job + '_' + arm, job) for job in ('a89', 'a93') for arm in sp.ARMS)

SLOW_SYSTEM = '''You are the boundary Slow of a batch training-data research Harness. Input: parent_knowledge, the existing experiment_guidance exactly as later Fast runs loaded it; historical_source_census, the legal record it was formed from; and usage_branches, a deterministic record of its actual use on two later jobs by five arms (Fast with no knowledge, Fast with a generic note, Fast with this guidance, Fast with a construction seed, and a random-policy control): T-only observations, constructed complete plans, tool results, C_A feedback, commits, post-commit delayed C_B, known costs, and metadata on unknown costs and stage order. No E values or E rankings and no data of any later job are present; usage branches with the same job share the same public baselines. Task: check the original guidance's intent against what actually happened when it was used, then either KEEP it or propose ONE bounded revision of the experiment_guidance hook (the only editable hook), and say which kind of later research decision the revision is expected to change and what the evidence can and cannot support. KEEP is a legitimate answer when the record does not justify a change; a proposal is CANDIDATE_TEST_ONLY. Do not change model, scoring, tool implementation, permissions, budgets or candidate counts. Distinguish observed facts, hypotheses under test and untested actions; never write an untested action as harmful. Shared-plan loss differences do not identify causal entity-level training value. Do not use dataset names, job labels, entity IDs, group names or dates as conditions; keep evidence in evidence_refs. Uniform plans, direct reuse, zero observation and stopping are all legitimate; nothing requires more observation, heterogeneity or extra calls. Output exact JSON: {"decision":"KEEP","rationale":"..."} OR {"decision":"PROPOSE","hook":"experiment_guidance","body":"<=1200 characters","observable_applicability":{"const":true} or a conditional AST,"evidence_refs":["supplied reference", "..."],"rationale":"..."}. const:true is explicit unconditional; null/missing is incomplete, never unconditional. Conditional scopes use only supplied batch feature names with numeric {feature,op,value} leaves (>,>=,<,<=,==), all/any/not/const. evidence_refs must be copied from legal_evidence_refs. No markdown.'''


# ----------------------------------------------------------------------------- frozen inputs
def _knowledge(d):
    return br.Knowledge(d['version'], tuple(br.Guidance(e['hook'], e['body'], e['applicability'], tuple(e['evidence_refs'])) for e in d['entries']))


def load_h_old():
    k = context.read_json(PREV / 'frozen_knowledge.json')['f_source']
    if not k or not k['entries']:
        raise RuntimeError('previous F_source knowledge is missing; nothing to revise')
    return _knowledge(k)


def load_generic():
    k = context.read_json(PREV / 'frozen_knowledge.json')['f_generic']
    assert k['entries'][0]['body'] == sp.GENERIC_TEXT
    return _knowledge(k)


def frozen_config(caps):
    jobs = {}
    for job in JOBS:
        js = context.resolve_job('electricity', job, roster=ROSTERS[job])
        jobs[job] = {'t': js.t, 'train_rows': list(js.train_range), 'c_a': list(js.c_a), 'c_b': list(js.c_b), 'e': list(js.e),
                     'workload_end_row_exclusive': max(js.e) + spec.H, 'roster': list(js.roster), 'roster_explicit': True}
    return {'study': 'skill_revision', 'jobs': jobs, 'seeds': list(SEEDS), 'order': {j: list(o) for j, o in ORDER.items()}, 'arms': list(ARMS),
            'menu': {'plans': [[m, p] for m, p in MENU], 'rule': 'argmin three-seed C_A mean among FixedMixup, None, MenuU, MenuFM; ties in that order; 0 LLM'},
            'caps': caps, 'planned_fits': 72, 'http_cap': 2 * caps['max_llm_requests'], 'exposure': 'EXPOSED_DEVELOPMENT',
            'population_note': 'same later cutoff t=25512, two disjoint 32-entity groups (CSV string-sorted names [0:32] and [32:64]); not two independent time replications',
            'knowledge': 'f0 empty; f_generic previous frozen generic; f_old previous frozen F_source card; f_new one Slow revision of f_old (or absent)',
            'slow': {'revisions': 1, 'contract_corrections': 1, 'editable_hooks': ['experiment_guidance'], 'reads': 'H_old, historical source census, usage census with C_A and post-commit C_B; no E, no later-job data'},
            'per_fast_job_limits': {**asdict(br.Limits()), 'wall_seconds': 'remaining package wall'}, 'max_tool_corrections': 2, 'evidence_roundtrip': False,
            'tool_contracts': CONTRACTS, 'tool_semantics': rt.TOOL_SEMANTICS, 'consumer': spec.CONSUMER, 'generic_body': sp.GENERIC_TEXT,
            'unknown_usage_rule': 'a failed transport attempt or a response without usage leaves cost UNKNOWN; at most one same-request retransmission; later paid calls refused; executor may not accept unknown usage',
            'utility_rule': 'Delta_revision = loss(F_old) - loss(F_new) on three-seed E means per job; no post-hoc significance gate; first-seed delivery listed separately.'}


# ----------------------------------------------------------------------------- usage census (no E, no later-job data)
def _compact_events(events):
    """Bound the request: per-entity vectors already live in candidates (by_entity_mean_over_seeds); drop them
    from tool events, keep arguments, loss_by_seed, means, signs and material summaries. Deterministic."""
    out = []
    for e in events:
        e = json.loads(json.dumps(e))
        if e.get('event') == 'tool' and isinstance(e.get('output'), dict):
            o = e['output']
            if e['tool'] == 'inspect_data':
                e['output'] = {'kind': o.get('kind'), 'rows_read': o.get('rows_read'), 'n_entities_returned': len(o.get('entities', {})), 'values': 'omitted (T-only views were returned to Fast)'}
            elif e['tool'] == 'evaluate':
                fb = o.get('feedback', {})
                e['output'] = {'plan_id': o.get('plan_id'), 'feedback': {k: fb.get(k) for k in ('block', 'loss_by_seed', 'mean_loss', 'loss_by_seed_origin')}}
            elif e['tool'] == 'compare':
                e['output'] = {k: o.get(k) for k in ('a', 'b', 'block', 'positive_means', 'delta_by_seed', 'mean_delta', 'seed_sd', 'seed_se', 'signs') if k in o}
            elif e['tool'] == 'inspect_material':
                groups = [{k: v for k, v in g.items() if k != 'entities'} for g in o.get('groups_by_steps', [])]
                e['output'] = {'material_id': o.get('material_id'), 'alias_of': o.get('alias_of'), 'n_identity': o.get('n_identity'), 'summary': o.get('summary'), 'groups_by_steps': groups}
        out.append(e)
    return out


def _failed_attempts_by_unit(root):
    out = {}
    for f in sorted((root / 'raw_responses').glob('*_attempt*.json')):
        req = f.with_name(f.name.split('_attempt')[0] + '_request.json')
        if req.exists():
            u = context.read_json(req).get('unit')
            out[u] = out.get(u, 0) + 1
    return out


def build_usage_evidence(prev=PREV):
    prior = context.read_json(prev / 'source_evidence.json')
    prior_compact = {k: prior[k] for k in ('boundary', 'jobs', 'branches', 'branch_labels', 'historical_tool_description') if k in prior}
    for job in prior_compact['jobs'].values():          # keep quantile summaries; drop the per-entity columns of the old jobs to bound the request
        job['overview'].pop('entity_columns', None)
        job['overview'].pop('entity_rows', None)
        job['overview']['entity_table'] = 'omitted here; summary_quantiles kept (same census as the original formation input)'
    for b in prior_compact['branches']:
        b['events'] = _compact_events(b['events'])
        for cand in b['candidates'].values():
            for blk in ('c_a', 'c_b'):
                if isinstance(cand.get(blk), dict):
                    cand[blk].pop('by_entity_mean_over_seeds', None)
    for job in prior_compact['jobs'].values():
        for cand in job['common_baselines'].values():
            for blk in ('c_a', 'c_b'):
                if isinstance(cand.get(blk), dict):
                    cand[blk].pop('by_entity_mean_over_seeds', None)
    prior_compact['note'] = 'per-entity vectors of the historical census omitted here (the parent guidance was formed from the full census); usage branches keep per-entity means'
    resume = context.read_json(prev / 'resume.json') if (prev / 'resume.json').exists() else {}
    budget = context.read_json(prev / 'budget.json')
    e_before = [Path(x).name for x in resume.get('e_opened_before_resume', [])]
    failed = _failed_attempts_by_unit(prev)
    run_meta = {
        'package_execution': 'all ten branches completed; one instrument stop (a89_f_source paused before its second call was sent) and a resume from the identical built request',
        'stage_order': {'a89': 'C_B and E of %s were opened by the stop path BEFORE the other a89 arms committed (protocol order deviation; the resumed arms had no read path to those files)' % '/'.join(e_before),
                        'a93': 'local stage order held within this job: every a93 commit preceded any a93 C_B/E'},
        'budget_rule_change': 'the frozen rule "unknown usage stops later paid calls" was set aside at the resume; paid calls continued',
        'unknown_cost': {'failed_transport_attempts': budget.get('llm_failed_attempts', 0), 'by_unit': failed,
                         'reserve_estimate_tokens': budget.get('token_reserved_failed_upper', 0), 'status': 'UNKNOWN; the reserve is an accounting estimate, not a bound'},
        'known_usage_tokens': budget['llm_tokens_in'] + budget['llm_tokens_out'],
        'physical_fits': budget['fit_attempts'], 'cache_hits': budget['cache_hits'],
        'exposure': 'EXPOSED_DEVELOPMENT usage feedback; not a validation of the parent guidance',
    }
    jobs, branches, refs = {}, [], []
    for name, job in USAGE_BRANCHES:
        broot = prev / name
        jdir = broot / job
        result = context.read_json(broot / 'branch_result.json')
        if job not in jobs:
            jobs[job] = {'overview': sp._job_overview(jdir), 'common_baselines': {}}
            refs.append('%s:overview' % job)
        cells = sp._cells(jdir)
        index = context.read_json(jdir / 'materials' / 'index.json')
        rows = [json.loads(l) for l in (broot / 'trace.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        events = _compact_events(sp._events(name, rows))
        refs += [e['ref'] for e in events]
        knowledge = context.read_json(broot / 'knowledge.json') if (broot / 'knowledge.json').exists() else {'version': 0, 'entries': []}
        for entry in knowledge.get('entries', []):
            entry.pop('evidence_refs', None)
        arm = name.split('_', 1)[1]
        started = next((r for r in rows if r['event'] == 'job_started'), {})
        candidates = {}
        for mid, byseed in cells.items():
            rec = index.get(mid, {})
            cand = {'alias_of': rec.get('alias_of') if rec.get('alias_of') != mid else None, 'n_identity': rec.get('n_identity'), 'policy': None,
                    'c_a': sp._block(byseed, 'c_a'), 'c_b': sp._block(byseed, 'c_b')}
            comp = jdir / 'materials' / (mid + '__compiled.json')
            if comp.exists():
                pol = context.read_json(comp).get('policy', {})
                cand['policy'] = {k: pol.get(k) for k in ('default', 'rules', 'rationale')}
            if mid in ('None', 'FixedMixup'):
                have = jobs[job]['common_baselines'].get(mid)
                if have is None or (have.get('c_b') is None and cand['c_b'] is not None):
                    jobs[job]['common_baselines'][mid] = cand
                candidates[mid] = {'see': 'usage_jobs.%s.common_baselines.%s' % (job, mid)}
            else:
                candidates[mid] = cand
        commit = context.read_json(jdir / 'commit.json') if (jdir / 'commit.json').exists() else None
        cb_means = {mid: sp._block(byseed, 'c_b') for mid, byseed in cells.items()}
        delayed = {'ref': '%s:delayed' % name, 'opened_after_commit': True, 'c_b_mean_by_plan': {mid: blk['mean'] for mid, blk in cb_means.items() if blk}} if (jdir / 'c_b_scores.json').exists() else None
        if delayed:
            refs.append(delayed['ref'])
        private = [m for m, rec in index.items() if m not in ('None', 'FixedMixup') and rec.get('alias_of') in (None, m) and m in cells]
        cost = {**sp._llm_cost(prev, name), 'failed_transport_attempts_cost_unknown': failed.get(name, 0), 'fast_calls_recorded': result.get('calls'),
                'tool_attempts': result.get('tool_calls'), 'new_complete_candidates_evaluated': result.get('new_evaluations'),
                'private_physical_fits': 3 * len(private), 'common_baseline_fits_shared_per_job': 6}
        meta = {'arm': arm, 'initial_knowledge_role': {'f0': 'no guidance', 'f_generic': 'generic experiment note', 'f_source': 'the parent experiment_guidance under review',
                                                        'f_program': 'construction seed carrying one historical complete policy', 'random': 'random-policy control, 0 LLM'}[arm]}
        if job == 'a89':
            meta['stage_order'] = ('this branch\'s C_B/E were opened before sibling arms committed' if name in e_before else 'committed after sibling C_B/E files existed on disk; no read path')
        if name == 'a89_f_source':
            meta['resume'] = 'paused after call 1 (call 2 never sent); resumed from the identical built request; 0 new fits'
        branches.append({'branch': name, 'job': job, 'execution_order': len(branches) + 1, 'status': result.get('status'),
                         'failure': None if result.get('status') == 'COMPLETE' else {'failure_kind': result.get('failure_kind'), 'reason': result.get('reason')},
                         'metadata': meta, 'initial_knowledge': knowledge,
                         'control_mode': ({'kind': 'random_policy sampler', 'llm_requests': 0, 'commit_rule': 'minimum C_A three-seed mean among None, FixedMixup, p1, p2'} if arm == 'random'
                                          else {'kind': 'tool-loop Fast', 'evidence_roundtrip': False, 'max_tool_corrections': started.get('max_tool_corrections', 0), 'limits': {'max_calls': 16, 'max_tools': 24, 'max_new_evaluations': 2}}),
                         'cost': cost, 'events': events, 'candidates': candidates,
                         'commit': ({'plan_id': commit['material_id'], 'reason': commit['reason'], 'fitted_materials_at_commit': commit['fitted_materials_at_commit']} if commit else None),
                         'delayed_c_b': delayed})
        refs += ['%s:knowledge' % name, '%s:cost' % name, '%s:metadata' % name]
    for b in branches:
        for ev in b['events']:
            if ev.get('event') == 'tool' and ev['tool'] == 'build_material':
                text = json.dumps(ev['arguments'].get('policy', {}).get('rationale', ''), ensure_ascii=False).lower()
                ev['rationale_mentions_cross_entity_donor'] = any(m in text for m in ('cross-entity', 'cross entity', 'across entit', 'other entit'))
    evidence = {
        'status': 'CANDIDATE_TEST_ONLY',
        'purpose': 'Boundary Slow input for ONE revision (or KEEP) of the parent experiment_guidance after its actual use.',
        'boundary': {'usage_jobs': ['a89', 'a93'], 'excluded': 'all E values and E rankings, any later job, any planner interpretation of outcomes', 'e_present': False},
        'consumer': spec.CONSUMER, 'actions': spec.ACTIONS, 'observation_fields': list(spec.OBS_FIELDS),
        'allowed_batch_features': sorted(rt.BATCH_FIELDS),
        'batch_field_semantics': 'Each batch_median_FIELD is the median of the current 32 T-only entity FIELD values. Material predicates use per-entity field names separately.',
        'public_semantics_now': rt.TOOL_SEMANTICS,
        'tool_contracts_seen_by_usage_fast': CONTRACTS,
        'historical_source_census': {'source_marker': 'the legal a55/a65/a75/a85 census the parent guidance was formed from (source_process package input)', **prior_compact},
        'usage_run_metadata': run_meta,
        'usage_jobs': jobs, 'usage_branches': branches,
        'legal_evidence_refs': sorted(set(refs) | set(prior.get('legal_evidence_refs', []))),
    }
    text = json.dumps(evidence, ensure_ascii=False)
    import re
    for bad in (r'a97', r'"e":', r'"external"', r'e_scores', r'g0', r'g1', r'25512'):
        if re.search(bad, text):
            raise PermissionError('forbidden token %r in usage evidence' % bad)
    return sp.assert_no_private(br.json_copy(evidence), 'usage evidence')


# ----------------------------------------------------------------------------- one natural revision
def revise(root, client, h_old, evidence):
    refs = frozenset(evidence['legal_evidence_refs'])
    payload = {**evidence, 'parent_knowledge': asdict(h_old), 'editable_hooks': ['experiment_guidance']}
    context.write_json(root / 'slow_evidence.json', payload)
    old = next(e for e in h_old.entries if e.hook == 'experiment_guidance')
    prior = None
    for i in range(2):
        try:
            proposal = client.call('slow' if i == 0 else 'slow_contract_correction', 'usage_boundary', payload, SLOW_SYSTEM, max_tokens=8192, request_timeout=600.)
            prior = proposal
            if isinstance(proposal, dict) and proposal.get('decision') == 'PROPOSE' and proposal.get('hook') != 'experiment_guidance':
                raise ValueError('this study allows only the experiment_guidance hook')
            h_new = br.apply_update(h_old, proposal, allowed_features=rt.BATCH_FIELDS, legal_evidence_refs=refs)
            for item in h_new.entries:
                policy.validate_text(item.body, allow_entity_ids=False)
            if h_new.version == h_old.version:
                status, diff = 'NO_REVISION_PROPOSED', None
            else:
                new = next(e for e in h_new.entries if e.hook == 'experiment_guidance')
                diff = {'body_changed': new.body != old.body, 'scope_changed': new.applicability != old.applicability,
                        'old_body': old.body, 'new_body': new.body, 'old_scope': old.applicability, 'new_scope': new.applicability,
                        'old_refs': list(old.evidence_refs), 'new_refs': list(new.evidence_refs)}
                status = 'CANDIDATE_TEST_ONLY' if (diff['body_changed'] or diff['scope_changed']) else 'NO_ACTIONABLE_EDIT'
            context.write_json(root / 'slow_result.json', {'status': status, 'proposal': proposal, 'h_old': asdict(h_old), 'h_new': asdict(h_new), 'diff': diff, 'contract_corrections_used': i})
            return status, (h_new if status == 'CANDIDATE_TEST_ONLY' else None)
        except (ValueError, policy.PolicyError) as exc:
            context.write_json(root / ('slow_contract_error_%d.json' % i), {'type': type(exc).__name__, 'message': str(exc), 'proposal': prior if prior is not None else client.last_text})
            if i == 0:
                payload = {**payload, 'correction': {'previous_output': prior if prior is not None else client.last_text, 'error': str(exc),
                                                     'instruction': 'Correct this contract error only. KEEP is allowed. Do not seek a different outcome.'}}
                continue
    context.write_json(root / 'slow_result.json', {'status': 'PARSE_OR_VALIDATION_FAILED', 'h_old': asdict(h_old), 'h_new': None, 'contract_corrections_used': 1})
    return 'PARSE_OR_VALIDATION_FAILED', None


def prepare(root, client, caps):
    """Freeze config, H_old, Generic and the usage census; run the one revision. Returns per-arm knowledge or None (early stop)."""
    context.write_json(root / 'frozen_config.json', frozen_config(caps))
    h_old, generic = load_h_old(), load_generic()
    context.write_json(root / 'h_old.json', asdict(h_old))
    context.write_json(root / 'generic_guidance.json', asdict(generic))
    evidence = build_usage_evidence()
    context.write_json(root / 'usage_evidence.json', evidence)
    try:
        status, h_new = revise(root, client, h_old, evidence)
    except rt.llm.AccountFault:
        context.write_json(root / 'slow_result.json', {'status': 'ACCOUNT_OR_PERMISSION_FAULT', 'h_old': asdict(h_old), 'h_new': None})
        raise
    except Exception as exc:
        context.write_json(root / 'slow_result.json', {'status': 'FORMATION_FAILED', 'exception_type': type(exc).__name__, 'h_old': asdict(h_old), 'h_new': None})
        status, h_new = 'FORMATION_FAILED', None
    knowledge = {'f0': br.Knowledge(), 'f_generic': generic, 'f_old': h_old, 'f_new': h_new}
    context.write_json(root / 'frozen_knowledge.json', {arm: (asdict(h) if h is not None else None) for arm, h in knowledge.items()})
    if h_new is None:
        context.write_json(root / 'early_stop.json', {'status': status, 'note': 'no revision treatment; Target fits not started; no same-knowledge double run'})
        return None
    return knowledge


def target_checks(root, job, adapter, knowledge):
    """After the T of a Target is opened: population binding and scope states per arm (no knowledge edits)."""
    ov = adapter.overview()
    out = {'job': job, 'roster': list(adapter.ctx.job.roster), 'roster_explicit': adapter.ctx.job.roster_override is not None,
           'n_entities': ov['n_entities'], 'train_rows': ov['train_rows'], 'scope': {}}
    for arm, h in knowledge.items():
        if h is None:
            continue
        r = h.render(ov['batch_features'], rt.BATCH_FIELDS)
        out['scope'][arm] = {'loaded_hooks': [x['hook'] for x in r['loaded']], 'applicability': r['applicability']}
    context.write_json(root / (job + '_target_checks.json'), out)
    return out


# ----------------------------------------------------------------------------- report
def finish_report(root, r, jobs, order, seeds):
    r['study'] = 'skill_revision'
    r.pop('treatment_present', None)
    slow = context.read_json(root / 'slow_result.json') if (root / 'slow_result.json').exists() else {'status': 'NOT_RUN'}
    frozen = context.read_json(root / 'frozen_knowledge.json') if (root / 'frozen_knowledge.json').exists() else {}
    r['revision'] = {'slow_status': slow.get('status'), 'proposal': slow.get('proposal'), 'diff': slow.get('diff'), 'contract_corrections_used': slow.get('contract_corrections_used'),
                     'frozen_knowledge': frozen}
    r['early_stop'] = context.read_json(root / 'early_stop.json') if (root / 'early_stop.json').exists() else None
    r['resume'] = context.read_json(root / 'resume.json') if (root / 'resume.json').exists() else None
    r['target_checks'] = {j: context.read_json(root / (j + '_target_checks.json')) for j in jobs if (root / (j + '_target_checks.json')).exists()}
    for name, b in r['branches'].items():
        job, arm = name.split('_', 2)[0] + '_' + name.split('_', 2)[1], name.split('_', 2)[2]
        b['arm'], b['job'] = arm, job
        b.pop('mode_enabled', None)
        b['guidance'] = sp._guidance_readout(root, name)
        expected = (frozen.get(arm) or {}).get('entries') or []
        b['loaded_bodies_match_frozen'] = (sorted(b['guidance']['loaded_bodies']) == sorted(e['body'] for e in expected)) if arm in ('f_generic', 'f_old', 'f_new') and b['guidance']['requests_with_loaded_guidance'] else None
        b['new_candidate_materials'] = {}
        for p, cells in b['plans'].items():
            if p in ('None', 'FixedMixup'):
                continue
            first = cells[0] if cells else {}
            b['new_candidate_materials'][p] = {'n_identity_entities': first.get('n_identity_entities'), 'material_alias_of': first.get('material_alias_of')}
            comp = root / name / job / 'materials' / (p + '__compiled.json')
            if comp.exists():
                ms = context.read_json(comp)
                b['new_candidate_materials'][p].update({'n_identity_entities': sum(1 for a in ms['assignment'] if not a), 'policy': ms['policy'], 'assignment_key': policy.assignment_key(ms['assignment'])})
        tokens_in = tokens_out = n = 0
        for req in sorted((root / 'raw_responses').glob('*_request.json')):
            q = context.read_json(req)
            if q.get('unit') != name:
                continue
            resp = req.with_name(req.name.replace('_request.json', '_response.json'))
            if resp.exists():
                u = context.read_json(resp).get('usage') or {}
                n += 1
                tokens_in += int(u.get('prompt_tokens') or 0)
                tokens_out += int(u.get('completion_tokens') or 0)
        b['llm_cost'] = {'requests_with_response': n, 'prompt_tokens': tokens_in, 'completion_tokens': tokens_out}
    seeds = tuple(seeds)

    def delta(cells_a, cells_b, block):
        ds = [cells_a[s][block]['normalized_mse_macro'] - cells_b[s][block]['normalized_mse_macro'] for s in seeds
              if s in cells_a and s in cells_b and cells_a[s].get(block) and cells_b[s].get(block)]
        if len(ds) != 3:
            return None
        return {'delta_by_seed': ds, 'mean_delta': statistics.mean(ds), 'seed_se': statistics.stdev(ds) / 3 ** .5, 'first_seed_delta': ds[0], 'positive_means': 'second_is_better'}

    r['comparisons'] = {}
    for job in jobs:
        new = r['branches'].get(job + '_f_new')
        comp = {'positive_means': 'F_new has lower loss', 'reference': 'f_new committed plan'}
        if new and new['controller']['status'] == 'COMPLETE':
            ref = sp._seed_cells(new, new['controller']['committed_plan_id'])
            for arm in ('f_old', 'f0', 'f_generic', 'menu'):
                other = r['branches'].get(job + '_' + arm)
                comp[arm + '_minus_f_new'] = ({block: delta(sp._seed_cells(other, other['controller']['committed_plan_id']), ref, block) for block in ('c_a', 'c_b', 'external')}
                                              if other and other['controller']['status'] == 'COMPLETE' else None)
            for base in ('None', 'FixedMixup'):
                comp[base + '_minus_f_new'] = {block: delta(sp._seed_cells(new, base), ref, block) for block in ('c_a', 'c_b', 'external')}
            comp['delta_revision'] = comp['f_old_minus_f_new']
        else:
            comp['status'] = 'UNKNOWN: F_new branch missing or incomplete'
        # same-material flags between arms (identical expanded assignment on this job)
        keys = {}
        for arm in ARMS:
            b = r['branches'].get(job + '_' + arm)
            if b and b['controller']['status'] == 'COMPLETE':
                p = b['controller']['committed_plan_id']
                keys[arm] = (b['new_candidate_materials'].get(p, {}).get('assignment_key') if p not in ('None', 'FixedMixup') else p)
        comp['committed_material_key_by_arm'] = keys
        r['comparisons'][job] = comp
    planned = sum(len(order[j]) for j in jobs)
    if r['early_stop']:
        r['status'] = 'EARLY_STOP_' + r['early_stop']['status']
    else:
        r['status'] = 'COMPLETE' if (len(r['branches']) == planned and not r['failures'] and all(b['controller']['status'] == 'COMPLETE' and b['delivery_e'] is not None for b in r['branches'].values())) else 'PARTIAL'
    r['utility_supported'] = None
    r['utility_scope'] = 'One Slow revision, one Fast trajectory per arm and job, three training seeds per complete plan, two disjoint populations at one cutoff; development pilot.'
    context.write_json(root / 'result.json', r)
    lines = ['# DEV-BATCH-RESEARCH-SKILL-REVISION：机器汇总（result.json 派生）', '', '状态：' + r['status'], '', 'Slow：' + str(slow.get('status')), '',
             '|分支|状态|commit|第一seed E|三seed E均值|Fast请求|工具尝试|指导加载请求数|集合内E遗憾|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for name, b in r['branches'].items():
        lines.append('|%s|%s|%s|%s|%s|%s|%s|%s|%s|' % (name, b['controller']['status'], b['controller']['committed_plan_id'], b['delivery_e'], b['paired_mean_e'],
                                                    b['controller']['calls'], b['controller']['tool_calls'], b['guidance']['requests_with_loaded_guidance'], b['commit_regret_within_observed_candidates']))
    lines += ['', '各作业配对比较（正值 = F_new 损失更低）：']
    for j, x in r['comparisons'].items():
        lines.append(j + ': ' + json.dumps(x, ensure_ascii=False))
    lines += ['', '预算：' + json.dumps(r.get('budget'), ensure_ascii=False), '', '主报告见 REPORT.md；本文件由 report() 自动生成，不含叙述判断。']
    (root / 'MACHINE_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('SKILL_REVISION_REPORT', r['status'], flush=True)
    return r
