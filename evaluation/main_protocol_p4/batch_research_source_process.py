"""DEV-BATCH-RESEARCH-SOURCE-PROCESS: thin study module for run_batch_research_roundtrip --study source_process.

Deterministic Source census (no E values), one Slow formation restricted to experiment_guidance, the
frozen Generic text, the historical-program seed P*, per-arm frozen knowledge and the study report.
No trainer, Skill store, clustering or scorer is built here; numerics stay in methods.ttha.batch_base.
"""
from __future__ import annotations
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from methods.ttha import batch_research as br
from methods.ttha.batch_base import context, policy, spec
from evaluation.main_protocol_p4 import batch_research_runtime as rt

REPO = Path(__file__).resolve().parents[2]
CONTRACTS = rt.contracts_with_semantics()
ARMS = ('f0', 'f_generic', 'f_source', 'f_program', 'random')
ORDER = {'a89': ('f0', 'f_generic', 'f_source', 'f_program', 'random'),
         'a93': ('random', 'f_program', 'f_source', 'f_generic', 'f0')}
RANDOM_SEEDS = {'a89': (910891, 910892), 'a93': (910931, 910932)}

# Source whitelist (task §4): completed or explicitly failed historical branches only, in execution order.
SOURCE_BRANCHES = (
    ('dev_batch_research_workflow_v1', 'a55_H0', 'a55'),
    ('dev_batch_research_workflow_v1', 'a65_parent', 'a65'),
    ('dev_batch_research_workflow_v1', 'a65_candidate', 'a65'),
    ('dev_batch_research_roundtrip', 'a75_old', 'a75'),
    ('dev_batch_research_roundtrip', 'a75_roundtrip', 'a75'),
    ('dev_batch_research_roundtrip', 'a75_random', 'a75'),
    ('dev_batch_research_roundtrip', 'a85_roundtrip', 'a85'),
    ('dev_batch_research_roundtrip', 'a85_old', 'a85'),
    ('dev_batch_research_roundtrip', 'a85_random', 'a85'),
)
LAST_SOURCE_JOB, SOURCE_WORKLOAD_END_ROW = 'a85', 22728
PRIVATE_KEYS = frozenset({'e', 'external', 'e_scores', 'evaluation_truth', 'query_future'})

# Frozen Generic arm text (task §6). Never edited after Source or Target results are seen.
GENERIC_TEXT = ('在预算内提出与当前证据对应的候选，并说明比较希望区分什么。如果希望归因，尽量保持其余训练条件一致。'
                '根据实际执行的工具结果组织后续实验；差异不清楚时说明不确定性。可以直接复用、继续探索或停止，'
                '由当前合法反馈与成本决定。不要把未测试的做法写成禁令。')

SLOW_SYSTEM = '''You are the boundary Slow of a batch training-data research Harness. You receive a deterministic census of several completed or explicitly failed historical source branches: their legal episodes (T-only observations, constructed complete plans, tool results, C_A feedback, commits, costs, failures) and post-commit delayed C_B. No E values or E rankings exist in this input, and no next-job data. Branches differ in initial knowledge and control mode; they are labeled and are NOT same-configuration repeats. The public tool description has since been corrected (see public_semantics_now vs historical_tool_description); interface facts that are now stated are not something future Fast must rediscover. Propose at most ONE reusable experiment_guidance change or KEEP. experiment_guidance is the only editable hook in this study: it tells a future Fast, on a new batch, in which observable situations which hypotheses are worth posing, which existing tool evidence to obtain, how to organize complete-plan comparisons within the fixed budget, and how strong an interpretation the results support. Direct reuse or stopping may be recommended; extra experiments are not mandatory. Do not change model, scoring, tool implementation, permissions or budgets. Distinguish observed facts, hypotheses under test and untested actions; never write an untested action as harmful. Shared-plan loss differences do not identify causal entity-level training value. Do not cite dataset names, job labels or entity IDs, and do not prescribe one fixed assignment or a full historical winning plan as the answer; specific operators may appear as bounded comparison examples. Output exact JSON: {"decision":"KEEP","rationale":"..."} OR {"decision":"PROPOSE","hook":"experiment_guidance","body":"<=1200 characters","observable_applicability":{"const":true},"evidence_refs":["supplied reference", "..."],"rationale":"..."}. const:true is explicit unconditional; null/missing is incomplete, never unconditional. Conditional scopes use only supplied batch feature names with numeric {feature,op,value} leaves (>,>=,<,<=,==), all/any/not/const. evidence_refs must be copied from legal_evidence_refs. No markdown. KEEP is a valid outcome and will not be resampled; the proposal is CANDIDATE_TEST_ONLY.'''


# ----------------------------------------------------------------------------- helpers
def _r(x, nd):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int, float)):
        return round(float(x), nd)
    if isinstance(x, list):
        return [_r(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    return x


def assert_no_private(value, where='source evidence'):
    def walk(item):
        if isinstance(item, dict):
            for k, v in item.items():
                if str(k).lower() in PRIVATE_KEYS:
                    raise PermissionError('private channel %r inside %s' % (k, where))
                walk(v)
        elif isinstance(item, list):
            for v in item:
                walk(v)
    walk(value)
    return value


def _mean(xs):
    return statistics.mean(xs) if xs else None


# ----------------------------------------------------------------------------- Source census
def _compact_args(tool, args):
    if tool == 'build_material':
        return {'plan_id': args.get('plan_id'), 'policy': args.get('policy')}
    return args


def _compact_output(tool, out):
    if tool == 'inspect_data':
        ents = {}
        for name, vals in out.get('entities', {}).items():
            if out.get('kind') == 'segment':
                v = [x for x in vals if x is not None]
                ents[name] = {'n': len(v), 'mean': _r(_mean(v), 3), 'std': _r(statistics.pstdev(v), 3) if len(v) > 1 else None, 'min': _r(min(v), 3), 'max': _r(max(v), 3)}
            else:
                ents[name] = _r(vals, 2)
        return {'kind': out.get('kind'), 'rows_read': out.get('rows_read'), 'entities': ents}
    if tool == 'build_material':
        ms = out.get('material_spec', {})
        assignment = ms.get('assignment', [])
        hits = {}
        for ri in ms.get('rule_index', []):
            hits[str(ri)] = hits.get(str(ri), 0) + 1
        return {'plan_id': out.get('plan_id'), 'n_identity': sum(1 for a in assignment if not a), 'rule_hits_by_rule_index': hits,
                'resolved_thresholds': _r(ms.get('resolved_thresholds'), 4), 'n_unknown': ms.get('n_unknown')}
    if tool == 'inspect_material':
        # Aggregate the per-entity diagnostics by identical step list (deterministic; identity entities listed by name).
        groups, identity = {}, []
        for name, rec in out.get('entities', {}).items():
            if rec.get('identity'):
                identity.append(name)
                continue
            key = json.dumps(rec.get('steps'), sort_keys=True, separators=(',', ':'))
            g = groups.setdefault(key, {'steps': rec.get('steps'), 'entities': [], 'child_x_change_rms': [], 'child_y_change_rms': [], 'donor_same_hour_ratio': [], 'donor_abs_time_distance_mean_h': []})
            g['entities'].append(name)
            for k in ('child_x_change_rms', 'child_y_change_rms', 'donor_same_hour_ratio', 'donor_abs_time_distance_mean_h'):
                if k in rec:
                    g[k].append(rec[k])
        agg = []
        for g in groups.values():
            row = {'steps': g['steps'], 'n_entities': len(g['entities']), 'entities': g['entities']}
            for k in ('child_x_change_rms', 'child_y_change_rms', 'donor_same_hour_ratio', 'donor_abs_time_distance_mean_h'):
                if g[k]:
                    row[k + '_mean'] = _r(_mean(g[k]), 3)
                    row[k + '_min_max'] = [_r(min(g[k]), 3), _r(max(g[k]), 3)]
            agg.append(row)
        return {'material_id': out.get('material_id'), 'alias_of': out.get('alias_of'), 'n_identity': out.get('n_identity'), 'summary': _r(out.get('summary'), 4),
                'identity_entities': identity, 'groups_by_steps': agg}
    if tool == 'evaluate':
        fb = out.get('feedback', {})
        return {'plan_id': out.get('plan_id'), 'feedback': {'block': 'C_A', 'loss_by_seed': _r(fb.get('loss_by_seed'), 6), 'mean_loss': _r(fb.get('mean_loss'), 6),
                                                            'loss_by_seed_origin': _r(fb.get('loss_by_seed_origin'), 4), 'loss_by_seed_entity': _r(fb.get('loss_by_seed_entity'), 3)}}
    if tool == 'compare':
        return {k: (_r(out.get(k), 6) if k in ('delta_by_seed', 'mean_delta', 'seed_sd', 'seed_se') else _r(out.get(k), 4) if k == 'delta_by_seed_origin' else _r(out.get(k), 3) if k == 'delta_by_seed_entity' else out.get(k))
                for k in ('a', 'b', 'block', 'positive_means', 'delta_by_seed', 'mean_delta', 'seed_sd', 'seed_se', 'signs', 'delta_by_seed_origin', 'delta_by_seed_entity', 'scope') if k in out}
    if tool == 'commit':
        return {'plan_id': out.get('plan_id'), 'reason': out.get('reason')}
    if tool == 'overview':
        return {'note': 'overview re-read; identical to jobs.<job>.overview'}
    return out


def _events(branch_name, rows):
    """Merge tool_started/tool_completed pairs; drop raw fast_response bodies (their actions appear as tools)."""
    out, pending = [], None
    for row in rows:
        ref = '%s:%s' % (branch_name, row['event_id'])
        ev = row['event']
        if ev == 'job_started':
            out.append({'ref': ref, 'event': ev, 'mode': row.get('mode'), 'knowledge_version': row.get('knowledge_version'), 'baseline_ids': row.get('baseline_ids'),
                        'evidence_roundtrip': row.get('evidence_roundtrip', False), 'max_tool_corrections': row.get('max_tool_corrections', 0)})
        elif ev == 'fast_request':
            g = row.get('guidance', {})
            out.append({'ref': ref, 'event': ev, 'number': row.get('number'), 'guidance_loaded_hooks': [x['hook'] for x in g.get('loaded', [])], 'applicability': g.get('applicability', [])})
        elif ev == 'fast_response':
            acts = (row.get('response') or {}).get('actions') or []
            out.append({'ref': ref, 'event': ev, 'number': row.get('number'), 'n_actions': len(acts), 'tools': [a.get('tool') for a in acts if isinstance(a, dict)]})
        elif ev == 'tool_started':
            pending = {'ref': ref, 'event': 'tool', 'tool': row['tool'], 'arguments': _compact_args(row['tool'], row.get('arguments', {})), 'completed': False}
            out.append(pending)
        elif ev == 'tool_completed':
            if pending is not None and pending['tool'] == row['tool'] and not pending['completed']:
                pending['completed'] = True
                pending['output'] = _compact_output(row['tool'], row.get('output', {}))
            else:
                out.append({'ref': ref, 'event': ev, 'tool': row['tool'], 'output': _compact_output(row['tool'], row.get('output', {}))})
            pending = None
        elif ev == 'tool_rejected':
            out.append({'ref': ref, 'event': ev, 'tool': row.get('tool'), 'error': row.get('error'), 'correction_allowed': row.get('correction_allowed'),
                        'unexecuted_tools': [a.get('tool') for a in row.get('unexecuted_actions', [])]})
        elif ev == 'action_batch_deferred':
            out.append({'ref': ref, 'event': ev, 'deferred_tools': [a.get('tool') for a in row.get('actions', [])], 'reason': row.get('reason')})
        elif ev == 'committed':
            out.append({'ref': ref, 'event': ev, 'plan_id': row.get('plan_id')})
        elif ev == 'job_incomplete':
            out.append({'ref': ref, 'event': ev, 'failure_kind': row.get('failure_kind'), 'reason': row.get('reason')})
        elif ev == 'random_material':
            ms = row.get('material_spec', {})
            out.append({'ref': ref, 'event': ev, 'sampling_seed': row.get('sampling_seed'), 'policy': {k: ms.get('policy', {}).get(k) for k in ('default', 'rules')},
                        'n_identity': sum(1 for a in ms.get('assignment', []) if not a), 'resolved_thresholds': _r(ms.get('resolved_thresholds'), 4)})
        elif ev == 'random_evaluated':
            out.append({'ref': ref, 'event': ev, 'plan_id': row.get('plan_id'), 'feedback': _compact_output('evaluate', {'feedback': row.get('feedback', {})})['feedback']})
        else:
            out.append({'ref': ref, 'event': ev})
    return out


def _cells(job_dir):
    out = {}
    cdir = job_dir / 'cells'
    for p in sorted(cdir.glob('*.json')) if cdir.exists() else []:
        c = context.read_json(p)
        if c.get('status') == 'OK':
            out.setdefault(c['material_id'], {})[c['model_seed']] = c
    return out


def _block(cells_by_seed, block):
    seeds = sorted(cells_by_seed)
    rows = [cells_by_seed[s]['scores'].get(block) for s in seeds]
    if any(r is None for r in rows) or len(rows) != len(rt.SEEDS):
        return None
    ent = [statistics.mean(r['per_entity_normalized_mse'][e] for r in rows) for e in range(32)]
    return {'seeds': seeds, 'by_seed': _r([r['normalized_mse_macro'] for r in rows], 6), 'mean': _r(statistics.mean(r['normalized_mse_macro'] for r in rows), 6),
            'by_origin_mean_over_seeds': _r([statistics.mean(r['per_origin_normalized_mse_mean'][o] for r in rows) for o in range(len(rows[0]['per_origin_normalized_mse_mean']))], 4),
            'by_entity_mean_over_seeds': _r(ent, 3)}


def _llm_cost(package_root, unit):
    tokens_in = tokens_out = n = 0
    for req in sorted((package_root / 'raw_responses').glob('*_request.json')):
        q = context.read_json(req)
        if q.get('unit') != unit or q.get('role') not in ('fast',):
            continue
        resp = req.with_name(req.name.replace('_request.json', '_response.json'))
        if not resp.exists():
            continue
        u = context.read_json(resp).get('usage') or {}
        n += 1
        tokens_in += int(u.get('prompt_tokens') or 0)
        tokens_out += int(u.get('completion_tokens') or 0)
    return {'fast_requests': n, 'prompt_tokens': tokens_in, 'completion_tokens': tokens_out}


def _job_overview(job_dir, dataset='electricity'):
    ov = context.read_json(job_dir / 'overview.json')
    js = context.resolve_job(dataset, ov['job_id'])
    return {'t': js.t, 'train_rows': list(js.train_range), 'c_a_origins': list(js.c_a), 'c_b_origins': list(js.c_b), 'workload_end_row_exclusive': max(js.e) + spec.H,
            'batch_features': _r({'batch_median_' + k: v['median'] for k, v in ov['summary'].items() if v}, 4),
            'summary_quantiles': _r(ov['summary'], 4), 'entity_rows': [row['entity'] for row in ov['entities']],
            'entity_columns': {f: _r([row.get(f) for row in ov['entities']], 3) for f in spec.OBS_FIELDS},
            'entity_table_layout': 'entity_columns[field][i] belongs to entity_rows[i]'}


def build_source_evidence(repo=REPO, *, source_branches=SOURCE_BRANCHES, dataset='electricity', last_source_job=LAST_SOURCE_JOB,
                          workload_end_row=SOURCE_WORKLOAD_END_ROW):
    """Deterministic census of the whitelisted branches. Never opens e_scores.json or any Target file.
    The keyword defaults are this study's frozen whitelist; other studies pass their own whitelist, dataset and boundary."""
    jobs, branches, refs = {}, [], []
    for package, name, job in source_branches:
        proot = repo / '_scratch' / package
        broot = proot / name
        jdir = broot / job
        result = context.read_json(broot / 'branch_result.json')
        if job not in jobs:
            js = context.resolve_job(dataset, job)
            assert max(js.e) + spec.H <= workload_end_row, 'source job runs past the frozen boundary'
            jobs[job] = {'overview': _job_overview(jdir, dataset), 'common_baselines': {}}
            refs.append('%s:overview' % job)
        cells = _cells(jdir)
        index = context.read_json(jdir / 'materials' / 'index.json') if (jdir / 'materials' / 'index.json').exists() else {}
        rows = [json.loads(l) for l in (broot / 'trace.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        events = _events(name, rows)
        refs += [e['ref'] for e in events]
        knowledge = context.read_json(broot / 'knowledge.json') if (broot / 'knowledge.json').exists() else {'version': 0, 'entries': []}
        for entry in knowledge.get('entries', []):
            entry.pop('evidence_refs', None)
        started = next((r for r in rows if r['event'] == 'job_started'), {})
        is_random = name.endswith('_random')
        candidates = {}
        for mid, byseed in cells.items():
            rec = index.get(mid, {})
            cand = {'alias_of': rec.get('alias_of') if rec.get('alias_of') != mid else None, 'n_identity': rec.get('n_identity'),
                    'policy': None, 'c_a': _block(byseed, 'c_a'), 'c_b': _block(byseed, 'c_b')}
            comp = jdir / 'materials' / (mid + '__compiled.json')
            if comp.exists():
                pol = context.read_json(comp).get('policy', {})
                cand['policy'] = {k: pol.get(k) for k in ('default', 'rules', 'rationale')}
            elif mid == 'None':
                cand['policy'] = {'default': {'steps': []}, 'rules': [], 'rationale': 'no augmentation'}
            elif mid == 'FixedMixup':
                cand['policy'] = {'default': {'steps': list(spec.FIXED_MIXUP_STEPS)}, 'rules': [], 'rationale': 'fixed incumbent'}
            if mid in ('None', 'FixedMixup'):
                have = jobs[job]['common_baselines'].get(mid)
                if have is None or (have.get('c_b') is None and cand['c_b'] is not None):
                    jobs[job]['common_baselines'][mid] = cand
                candidates[mid] = {'see': 'jobs.%s.common_baselines.%s' % (job, mid), 'c_b_opened_in_this_branch': cand['c_b'] is not None}
            else:
                candidates[mid] = cand
        commit = context.read_json(jdir / 'commit.json') if (jdir / 'commit.json').exists() else None
        delayed = None
        if (jdir / 'c_b_scores.json').exists():
            cb_means = {mid: _block(byseed, 'c_b') for mid, byseed in cells.items()}
            delayed = {'ref': '%s:delayed' % name, 'opened_after_commit': True, 'c_b_mean_by_plan': {mid: blk['mean'] for mid, blk in cb_means.items() if blk}}
            refs.append(delayed['ref'])
        failure = None
        if result.get('status') != 'COMPLETE':
            failure = {'failure_kind': result.get('failure_kind'), 'reason': result.get('reason')}
            bad = [r for r in rows if r['event'] == 'tool_started' and r['tool'] == 'build_material']
            if bad and result.get('reason') == 'PolicyError':
                pol = bad[-1]['arguments'].get('policy', {})
                unknown = sorted({leaf['feature'] for rule in pol.get('rules', []) for leaf in _leaves(rule.get('when', {})) if leaf.get('feature') not in spec.OBS_FIELDS})
                failure['mechanical_note'] = {'last_build_plan_id': bad[-1]['arguments'].get('plan_id'), 'predicate_features_outside_public_vocabulary': unknown,
                                              'branch_max_tool_corrections': started.get('max_tool_corrections', 0)}
        private_materials = [m for m, rec in index.items() if m not in ('None', 'FixedMixup') and (rec.get('alias_of') in (None, m)) and m in cells]
        branches.append({
            'branch': name, 'package': package, 'job': job, 'execution_order': len(branches) + 1,
            'status': result.get('status'), 'failure': failure,
            'initial_knowledge': knowledge,
            'control_mode': ({'kind': 'random_policy sampler', 'llm_requests': 0, 'commit_rule': 'minimum C_A three-seed mean among None, FixedMixup, p1, p2 (frozen tie order)', 'n_sampled_candidates': 2}
                             if is_random else {'kind': 'tool-loop Fast', 'evidence_roundtrip': started.get('evidence_roundtrip', False), 'max_tool_corrections': started.get('max_tool_corrections', 0),
                                                'limits': {'max_calls': 16, 'max_tools': 24, 'max_new_evaluations': 2}}),
            'cost': {**(_llm_cost(proot, name) if not is_random else {'fast_requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0}),
                     'fast_calls_recorded': result.get('calls'), 'tool_attempts': result.get('tool_calls'), 'new_complete_candidates_evaluated': result.get('new_evaluations'),
                     'private_physical_fits': 3 * len(private_materials), 'common_baseline_fits_shared_per_job': 6},
            'events': events,
            'candidates': candidates,
            'commit': ({'plan_id': commit['material_id'], 'reason': commit['reason'], 'fitted_materials_at_commit': commit['fitted_materials_at_commit']} if commit else None),
            'delayed_c_b': delayed,
        })
        refs.append('%s:knowledge' % name)
        refs.append('%s:cost' % name)
    hist_contract = context.read_json(repo / '_scratch' / 'dev_batch_research_roundtrip' / 'config.json')['tool_contracts']
    evidence = {
        'status': 'CANDIDATE_TEST_ONLY',
        'purpose': 'Boundary Slow input: can past research processes be consolidated into ONE reusable experiment_guidance for a future batch job?',
        'boundary': {'last_source_job': last_source_job, 'source_workload_end_row_exclusive': workload_end_row,
                     'excluded': 'later jobs, any Target observation/result, all E values and E rankings', 'e_present': False},
        'consumer': spec.CONSUMER, 'actions': spec.ACTIONS, 'observation_fields': list(spec.OBS_FIELDS),
        'allowed_batch_features': sorted(rt.BATCH_FIELDS),
        'batch_field_semantics': 'Each batch_median_FIELD is the median of the current 32 T-only entity FIELD values. Material predicates use per-entity field names separately.',
        'public_semantics_now': rt.TOOL_SEMANTICS,
        'historical_tool_description': {'build_material_meaning_seen_by_source_fast': hist_contract['build_material']['meaning'],
                                        'note': 'Source branches saw overview.actions (R/U enumerations) and the meanings above only; the same-entity donor, joint [X;y] and shared-training facts in public_semantics_now were not stated to them and are stated to every future arm.'},
        'branch_labels': 'Branches differ in initial knowledge and control mode and are labeled individually; they are not same-configuration repeats.',
        'jobs': jobs, 'branches': branches,
        'legal_evidence_refs': sorted(set(refs)),
    }
    for b in branches:
        for ev in b['events']:
            if ev.get('event') == 'tool' and ev['tool'] == 'build_material':
                text = json.dumps(ev['arguments'].get('policy', {}).get('rationale', ''), ensure_ascii=False).lower()
                ev['rationale_mentions_cross_entity_donor'] = any(m in text for m in ('cross-entity', 'cross entity', 'across entit', 'other entit'))
    return assert_no_private(br.json_copy(evidence))


def _leaves(p):
    if not isinstance(p, dict):
        return []
    if 'feature' in p:
        return [p]
    if 'not' in p:
        return _leaves(p['not'])
    for k in ('all', 'any'):
        if k in p:
            return [x for q in p[k] for x in _leaves(q)]
    return []


# ----------------------------------------------------------------------------- P*: historical program seed
def select_program_seed(repo=REPO):
    """argmin three-seed C_A mean over every actually trained complete candidate of the last Source job;
    aliases (same expanded assignment) count once; ties by frozen full path lexicographic order."""
    pkg = repo / '_scratch' / 'dev_batch_research_roundtrip'
    table = {}
    for name in sorted(p.name for p in pkg.glob(LAST_SOURCE_JOB + '_*') if p.is_dir()):
        jdir = pkg / name / LAST_SOURCE_JOB
        if not (jdir / 'materials' / 'index.json').exists():
            continue
        index = context.read_json(jdir / 'materials' / 'index.json')
        cells = _cells(jdir)
        for mid, rec in sorted(index.items()):
            if mid not in cells or len(cells[mid]) != len(rt.SEEDS):
                continue
            path = '%s/%s/%s/materials/%s' % (pkg.name, name, LAST_SOURCE_JOB, mid)
            comp = jdir / 'materials' / (mid + '__compiled.json')
            pol = context.read_json(comp)['policy'] if comp.exists() else (policy.identity_policy() if mid == 'None' else policy.fixed_mixup_policy())
            mean = statistics.mean(cells[mid][s]['scores']['c_a']['normalized_mse_macro'] for s in cells[mid])
            key = rec['key']
            row = {'path': path, 'material_id': mid, 'branch': name, 'c_a_mean': mean, 'policy': {k: pol[k] for k in ('default', 'rules')}}
            if key not in table or path < table[key]['path']:
                table[key] = row
    ranked = sorted(table.values(), key=lambda r: (r['c_a_mean'], r['path']))
    best = ranked[0]
    clean = policy.validate_policy({'default': best['policy']['default'], 'rules': best['policy']['rules'], 'rationale': '', 'observation_fields_used': []})
    normalized = {'default': clean['default'], 'rules': clean['rules']}
    body = ('Optional starting point for the current job, not a recommendation: one complete policy that may be re-instantiated on the current batch, inspected, evaluated, modified or ignored. '
            'If used, it counts as one of the new candidates. Quantile thresholds resolve against the current batch. Policy: '
            + json.dumps(normalized, separators=(',', ':')))
    policy.validate_text(body, allow_entity_ids=False)
    if len(body) > 1200:
        raise ValueError('program seed text exceeds the 1200-character guidance body contract')
    return {'rule': 'argmin C_A three-seed mean over actually trained complete candidates of the last Source job; aliases once; ties by frozen path',
            'selected': best, 'ranked_unique_materials': ranked, 'policy': normalized, 'guidance_body': body, 'hook': 'construction_guidance'}


# ----------------------------------------------------------------------------- Slow formation
def form_source_guidance(root, client, evidence):
    refs = frozenset(evidence['legal_evidence_refs'])
    payload = {**evidence, 'parent_knowledge': asdict(br.Knowledge()), 'editable_hooks': ['experiment_guidance']}
    context.write_json(root / 'slow_evidence.json', payload)
    prior = None
    for i in range(2):
        try:
            proposal = client.call('slow' if i == 0 else 'slow_contract_correction', 'source_boundary', payload, SLOW_SYSTEM, max_tokens=8192, request_timeout=600.)
            prior = proposal
            if isinstance(proposal, dict) and proposal.get('decision') == 'PROPOSE' and proposal.get('hook') != 'experiment_guidance':
                raise ValueError('this study allows only the experiment_guidance hook')
            h = br.apply_update(br.Knowledge(), proposal, allowed_features=rt.BATCH_FIELDS, legal_evidence_refs=refs)
            for item in h.entries:
                policy.validate_text(item.body, allow_entity_ids=False)
            context.write_json(root / 'slow_result.json', {'status': 'KEEP' if h.version == 0 else 'CANDIDATE_TEST_ONLY', 'proposal': proposal, 'knowledge': asdict(h), 'contract_corrections_used': i})
            return h
        except (ValueError, policy.PolicyError) as exc:
            context.write_json(root / ('slow_contract_error_%d.json' % i), {'type': type(exc).__name__, 'message': str(exc), 'proposal': prior if prior is not None else client.last_text})
            if i == 0:
                payload = {**payload, 'correction': {'previous_output': prior if prior is not None else client.last_text, 'error': str(exc),
                                                     'instruction': 'Correct this contract error only. KEEP is allowed. Do not seek a different outcome.'}}
                continue
    context.write_json(root / 'slow_result.json', {'status': 'PARSE_OR_VALIDATION_FAILED', 'knowledge': asdict(br.Knowledge()), 'contract_corrections_used': 1})
    return None


def prepare(root, client, repo=REPO):
    """Freeze Source census, P*, Generic and the one Slow output before any Target adapter opens."""
    evidence = build_source_evidence(repo)
    context.write_json(root / 'source_evidence.json', evidence)
    seed = select_program_seed(repo)
    context.write_json(root / 'program_seed.json', seed)
    context.write_json(root / 'generic_guidance.json', {'hook': 'experiment_guidance', 'body': GENERIC_TEXT, 'observable_applicability': {'const': True}, 'frozen_before_source_and_target': True})
    knowledge = {'f0': br.Knowledge(),
                 'f_generic': br.Knowledge(1, (br.Guidance('experiment_guidance', GENERIC_TEXT, {'const': True}, ('generic:frozen',)),)),
                 'f_program': br.Knowledge(1, (br.Guidance('construction_guidance', seed['guidance_body'], {'const': True}, ('program_seed:argmin_c_a_last_source_job',)),)),
                 'f_source': None}
    try:
        knowledge['f_source'] = form_source_guidance(root, client, evidence)
    except rt.llm.AccountFault:
        context.write_json(root / 'slow_result.json', {'status': 'ACCOUNT_OR_PERMISSION_FAULT', 'knowledge': asdict(br.Knowledge())})
        raise
    except Exception as exc:
        context.write_json(root / 'slow_result.json', {'status': 'FORMATION_FAILED', 'exception_type': type(exc).__name__, 'knowledge': asdict(br.Knowledge())})
        knowledge['f_source'] = None
    context.write_json(root / 'frozen_knowledge.json', {arm: (asdict(h) if h is not None else None) for arm, h in knowledge.items()})
    return knowledge


# ----------------------------------------------------------------------------- report
def _seed_cells(entry, plan):
    return {c['seed']: c for c in entry['plans'].get(plan, [])}


def _delta(cells_a, cells_b, block):
    """a - b per seed; positive means b (the second) has lower loss."""
    ds = [cells_a[s][block]['normalized_mse_macro'] - cells_b[s][block]['normalized_mse_macro'] for s in rt.SEEDS
          if s in cells_a and s in cells_b and cells_a[s].get(block) and cells_b[s].get(block)]
    if len(ds) != 3:
        return None
    return {'delta_by_seed': ds, 'mean_delta': statistics.mean(ds), 'seed_se': statistics.stdev(ds) / 3 ** .5, 'first_seed_delta': ds[0], 'positive_means': 'second_is_better'}


def _guidance_readout(root, name):
    rows = [json.loads(x) for x in (root / name / 'trace.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    reqs = [r for r in rows if r['event'] == 'fast_request']
    loaded = [r for r in reqs if r.get('guidance', {}).get('loaded')]
    states = sorted({a['state'] for r in reqs for a in r.get('guidance', {}).get('applicability', [])})
    bodies = sorted({x['body'] for r in loaded for x in r['guidance']['loaded']})
    return {'requests': len(reqs), 'requests_with_loaded_guidance': len(loaded), 'applicability_states': states, 'loaded_bodies': bodies,
            'tool_sequence': [r['tool'] for r in rows if r['event'] == 'tool_completed'],
            'rejections': sum(r['event'] == 'tool_rejected' for r in rows),
            'built_policies': {r['arguments']['plan_id']: r['arguments'].get('policy', {'derived_from': r['arguments'].get('parent_plan_id'), 'parameter_edit': r['arguments'].get('parameter_edit')})
                               for r in rows if r['event'] == 'tool_started' and r['tool'] == 'build_material'}}   # derived build form (paired_refinement) carries no policy argument


def _program_reuse(root, name, job, seed):
    """Did F_program actually re-instantiate P* (same normalized policy or same expanded assignment on the current T)?"""
    jdir = root / name / job
    index = context.read_json(jdir / 'materials' / 'index.json') if (jdir / 'materials' / 'index.json').exists() else {}
    ov = context.read_json(jdir / 'overview.json')
    target_key = policy.compile_policy({**seed['policy'], 'rationale': '', 'observation_fields_used': []}, ov['entities'])['key']
    out = {'program_seed_assignment_key_on_current_T_matches': [], 'program_seed_policy_verbatim': []}
    for mid, rec in index.items():
        if mid in ('None', 'FixedMixup'):
            continue
        if rec['key'] == target_key:
            out['program_seed_assignment_key_on_current_T_matches'].append(mid)
        comp = jdir / 'materials' / (mid + '__compiled.json')
        if comp.exists():
            pol = context.read_json(comp)['policy']
            if {k: pol[k] for k in ('default', 'rules')} == seed['policy']:
                out['program_seed_policy_verbatim'].append(mid)
    return out


def finish_report(root, r, jobs, order):
    r['study'] = 'source_process'
    r.pop('treatment_present', None)
    slow = context.read_json(root / 'slow_result.json') if (root / 'slow_result.json').exists() else {'status': 'NOT_RUN'}
    seed = context.read_json(root / 'program_seed.json') if (root / 'program_seed.json').exists() else None
    frozen = context.read_json(root / 'frozen_knowledge.json') if (root / 'frozen_knowledge.json').exists() else {}
    r['source'] = {'slow_status': slow.get('status'), 'proposal': slow.get('proposal'), 'contract_corrections_used': slow.get('contract_corrections_used'),
                   'program_seed': (None if seed is None else {'selected_material': seed['selected']['material_id'], 'selected_branch': seed['selected']['branch'], 'policy': seed['policy']}),
                   'generic_body': GENERIC_TEXT, 'frozen_knowledge': frozen}
    r['resume'] = context.read_json(root / 'resume.json') if (root / 'resume.json').exists() else None
    r['f_source_treatment'] = {}
    for job in jobs:
        t = root / (job + '_f_source_treatment.json')
        if t.exists():
            r['f_source_treatment'][job] = context.read_json(t)
    for name, b in r['branches'].items():
        job, arm = name.split('_', 1)
        b['arm'] = arm
        b['job'] = job
        b.pop('treatment_present', None)
        b.pop('mode_enabled', None)
        b['guidance'] = _guidance_readout(root, name)
        if arm == 'f_program' and seed is not None:
            b['program_seed_reuse'] = _program_reuse(root, name, job, seed)
        b['loaded_bodies_match_frozen'] = None
        expected = (frozen.get(arm) or {}).get('entries') or []
        if arm in ('f_generic', 'f_source', 'f_program'):
            b['loaded_bodies_match_frozen'] = sorted(b['guidance']['loaded_bodies']) == sorted(e['body'] for e in expected) if b['guidance']['requests_with_loaded_guidance'] else False
        b['new_candidate_materials'] = {}
        for p, cells in b['plans'].items():
            if p in ('None', 'FixedMixup'):
                continue
            first = cells[0] if cells else {}
            b['new_candidate_materials'][p] = {'n_identity_entities': first.get('n_identity_entities'), 'material_alias_of': first.get('material_alias_of')}
            comp = root / name / job / 'materials' / (p + '__compiled.json')
            if comp.exists():
                ms = context.read_json(comp)
                b['new_candidate_materials'][p].update({'n_identity_entities': sum(1 for a in ms['assignment'] if not a), 'policy': ms['policy']})
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
    r['comparisons'] = {}
    for job in jobs:
        src = r['branches'].get(job + '_f_source')
        comp = {'positive_means': 'F_source has lower loss', 'reference': 'f_source committed plan'}
        if src and src['controller']['status'] == 'COMPLETE':
            ref = _seed_cells(src, src['controller']['committed_plan_id'])
            for arm in ('f0', 'f_generic', 'f_program', 'random'):
                other = r['branches'].get(job + '_' + arm)
                if other and other['controller']['status'] == 'COMPLETE':
                    cells = _seed_cells(other, other['controller']['committed_plan_id'])
                    comp[arm + '_minus_f_source'] = {block: _delta(cells, ref, block) for block in ('c_a', 'c_b', 'external')}
                else:
                    comp[arm + '_minus_f_source'] = None
            for base in ('None', 'FixedMixup'):
                comp[base + '_minus_f_source'] = {block: _delta(_seed_cells(src, base), ref, block) for block in ('c_a', 'c_b', 'external')}
        else:
            comp['status'] = 'UNKNOWN: F_source branch missing or incomplete'
        # Secondary pairwise table among all arms (E only) for the reader; same positive convention as key order.
        arms_done = {a: r['branches'].get(job + '_' + a) for a in ARMS}
        table = {}
        for a in ARMS:
            for c in ARMS:
                if a >= c or not arms_done[a] or not arms_done[c]:
                    continue
                if arms_done[a]['controller']['status'] != 'COMPLETE' or arms_done[c]['controller']['status'] != 'COMPLETE':
                    continue
                d = _delta(_seed_cells(arms_done[a], arms_done[a]['controller']['committed_plan_id']), _seed_cells(arms_done[c], arms_done[c]['controller']['committed_plan_id']), 'external')
                table['%s_minus_%s' % (a, c)] = d
        comp['all_pairs_external'] = table
        r['comparisons'][job] = comp
    planned = sum(len(order[j]) for j in jobs)
    skipped = len(r['f_source_treatment'])
    r['status'] = 'COMPLETE' if (len(r['branches']) + skipped == planned and not r['failures'] and all(b['controller']['status'] == 'COMPLETE' and b['delivery_e'] is not None for b in r['branches'].values())) else 'PARTIAL'
    r['utility_supported'] = None
    r['utility_scope'] = 'One Source formation, one Fast trajectory per arm and job, three training seeds per complete plan; development pilot, not a Skill promotion.'
    context.write_json(root / 'result.json', r)
    lines = ['# DEV-BATCH-RESEARCH-SOURCE-PROCESS：机器汇总（result.json 派生）', '', '状态：' + r['status'], '', 'Slow：' + str(slow.get('status')), '',
             '|分支|状态|commit|第一seed E|三seed E均值|Fast请求|工具尝试|指导加载请求数|集合内E遗憾|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for name, b in r['branches'].items():
        lines.append('|%s|%s|%s|%s|%s|%s|%s|%s|%s|' % (name, b['controller']['status'], b['controller']['committed_plan_id'], b['delivery_e'], b['paired_mean_e'],
                                                    b['controller']['calls'], b['controller']['tool_calls'], b['guidance']['requests_with_loaded_guidance'], b['commit_regret_within_observed_candidates']))
    lines += ['', '各作业配对比较（正值 = F_source 损失更低）：']
    for j, x in r['comparisons'].items():
        lines.append(j + ': ' + json.dumps({k: v for k, v in x.items() if k != 'all_pairs_external'}, ensure_ascii=False))
    lines += ['', '预算：' + json.dumps(r.get('budget'), ensure_ascii=False), '', '主报告见 REPORT.md；本文件由 report() 自动生成，不含叙述判断。']
    (root / 'MACHINE_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('SOURCE_PROCESS_REPORT', r['status'], flush=True)
    return r
