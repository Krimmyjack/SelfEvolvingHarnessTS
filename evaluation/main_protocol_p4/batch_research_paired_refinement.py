"""DEV-BATCH-RESEARCH-PAIRED-REFINEMENT: thin study module for run_batch_research_roundtrip --study paired_refinement.

Fixed Harness: Fast constructs one complete plan, gets its real paired C_A, then revises ONE numeric leaf of an already
evaluated parent (the derived build form of task §5) so that the next fit answers a stated material question; a third
slot is free; commit is possible at any stage. Five arms on G5 (CSV string-sorted names [160:192]) at a65/a85, empty H0,
no Slow, no drafts shown to Fast:
  F_refine       three slots: free -> derived single-parameter revision of a fully evaluated parent -> free or derived
  F_free         same tools, same round trip, same three slots; the derived form is optional, never required
  Random-global  R1, R2, R3 of the frozen random stream co-trained; argmin three-seed C_A (Fixed, None, R1, R2, R3); 0 LLM
  Random-local   R1 -> one random legal single-parameter neighbour of the C_A-best parent among Fixed/None/R1 -> first of
                 R2..R4 not duplicating a registered assignment; argmin C_A; 0 LLM
  Menu           fixed U-Mixup 0.25 / FreqMask 0.10; argmin C_A (Fixed, None, MenuU, MenuFM); 0 LLM
Both Fast arms use evidence_roundtrip (a build/evaluate/commit after new evidence in the same response is deferred to
the next request) and semantic dedup (a new plan equal to a registered expanded assignment is rejected before material
or fit). The second-slot contract is enforced for F_refine only, in the controller's check_evaluate hook, before the slot
is counted or any fit reserved. Nothing numerical lives here; run_job, Adapter, roster and label stages are reused.
"""
from __future__ import annotations
import copy
import functools
import json
import re
import shutil
import statistics
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from methods.ttha import batch_research as br
from methods.ttha.batch_base import commit, context, materials, policy, spec
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import run_batch_research_v1 as base
from evaluation.main_protocol_p4 import batch_research_source_process as sp
from evaluation.main_protocol_p4 import batch_research_draft_construction as dc

REPO = Path(__file__).resolve().parents[2]
CONTRACTS = dc.CONTRACTS                      # public tool facts shared by every arm
SEEDS = (20261004, 20261005, 20261006)
ARMS = ('f_refine', 'f_free', 'random_global', 'random_local', 'menu')
FAST_ARMS = ('f_refine', 'f_free')
POOL_ARMS = ('random_global', 'random_local')  # the only arms that read the frozen random stream; Fast never sees it
POOL_NAME = 'random supply'
FROZEN_MARKER = 'random_supply_frozen.json'
JOBS = ('a65_g5', 'a85_g5')
ORDER = {'a65_g5': ('f_refine', 'f_free', 'random_global', 'random_local', 'menu'),
         'a85_g5': ('menu', 'random_local', 'random_global', 'f_free', 'f_refine')}
# G5: task §3, the next whole group after G4 in the fixed string order; chosen by order, never by any score.
G5 = ('242', '243', '244', '245', '246', '247', '248', '249', '25', '250', '251', '252', '253', '254', '255', '256',
      '257', '258', '259', '26', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '27', '270')
ROSTERS = {'a65_g5': G5, 'a85_g5': G5}
MENU, MENU_TIE_ORDER = dc.MENU, dc.MENU_TIE_ORDER
BASELINE_IDS = ('None', 'FixedMixup')
PROPOSAL_SEED_BASE, LOCAL_SEED_BASE = 2026091700, 2026091800
N_RANDOM = 4
LIMITS = {'max_calls': 16, 'max_tools': 32, 'max_new_evaluations': 3}   # wall = remaining package wall (bound at branch time)
GLOBAL_TIE_ORDER = ('FixedMixup', 'None', 'R1', 'R2', 'R3')
FIT_PLAN = '2 jobs x (6 common + F_refine 9 + F_free 9 + Random-global 9 + Random-local 9 + Menu 6) = at most 96; cap 98 reserves 2 same-configuration non-scientific retries'
PARAM_OF = {'timemixup': 'w', 'freqmask': 'mu', 'freqmix': 'mu'}
EDIT_KEYS = ('location', 'rule_index', 'step_index', 'parameter', 'new_value')
PROCESS_NOTE = ('Use current batch evidence and actual paired C_A feedback to seek useful complete training material. State a brief testable '
                'hypothesis in the material rationale, name the comparison, and revise or stop as useful. Stronger, weaker, uniform and '
                'conditional constructions are all hypotheses, not defaults. Extra observations, edits and fits are not goals.')
DERIVED_FORM = {'arguments': {'plan_id': 'new short alphanumeric/underscore id',
                              'parent_plan_id': '<a plan of this run listed in candidates with trained=true (all three seeds fitted, C_A returned); FixedMixup qualifies; plans of other runs, unbuilt or untrained plans do not>',
                              'parameter_edit': {'location': 'default | rule', 'rule_index': 'null when location is default; the 0-based index of an existing rule when location is rule',
                                                 'step_index': '0-based index of an existing step of that branch', 'parameter': 'w for timemixup, mu for freqmask/freqmix (the numeric parameter of that step)',
                                                 'new_value': 'a legal value of overview.actions for that parameter, different from the current value; booleans are not numbers'},
                              'rationale': '<brief current-evidence hypothesis and the comparison being tested>'},
                'example': {'plan_id': 'P2', 'parent_plan_id': 'P1', 'parameter_edit': {'location': 'default', 'rule_index': None, 'step_index': 0, 'parameter': 'mu', 'new_value': 0.20},
                            'rationale': 'Brief current-evidence hypothesis and the comparison being tested.'},
                'meaning': ('Alternative to the policy form. The parent policy is deep-copied and exactly this one numeric leaf is changed; every other operation, '
                            'condition, order and rule stays as in the parent; the result is compiled by the same compiler and built as a new material. The edited '
                            'branch must apply to at least one current entity. Entities outside that branch keep bit-identical material; changed entities are '
                            'regenerated by the same frozen random streams. The tool returns the actual entity/array change, not a predicted effect. The example is '
                            'syntax only, not a recommended operator, parameter or direction. The parent keeps its own material and scores.')}
IDENTITY_RULE = ('A new plan whose expanded assignment equals a registered plan of this run (None, FixedMixup or any earlier plan of yours) is rejected before '
                 'material or fit, naming the existing id; evaluate that id instead (a cached read: no new slot, no fit).')


# ----------------------------------------------------------------------------- single-leaf derivation (shared by both Fast arms and Random-local)
class EditError(ValueError):
    """Public, pre-execution rejection of a parameter_edit."""


def derive_policy(parent_policy, edit):
    """Deep-copy `parent_policy` (a validated policy) and change exactly one numeric leaf. Returns (child_policy, edit_record).
    Raises EditError with a public message; never touches materials or fits."""
    if not isinstance(edit, dict) or set(edit) != set(EDIT_KEYS):
        raise EditError('parameter_edit needs exactly the keys location, rule_index, step_index, parameter, new_value')
    loc, ri, si, par, nv = (edit[k] for k in EDIT_KEYS)
    rules = parent_policy.get('rules', []) or []
    if loc == 'default':
        if ri is not None:
            raise EditError('rule_index must be null when location is default')
        branch = parent_policy['default']
    elif loc == 'rule':
        if type(ri) is not int or not 0 <= ri < len(rules):
            raise EditError('rule_index must be the 0-based index of an existing rule (the parent has %d rules)' % len(rules))
        branch = rules[ri]
    else:
        raise EditError('location must be default or rule')
    steps = branch['steps']
    if type(si) is not int or not 0 <= si < len(steps):
        raise EditError('step_index must be the 0-based index of an existing step (this branch has %d steps)' % len(steps))
    step = steps[si]
    op = step['op']
    if par not in ('w', 'mu') or par not in spec.ACTIONS[op]:
        raise EditError('parameter must be the numeric parameter of that step: %s has %s' % (op, PARAM_OF[op]))
    allowed = list(spec.ACTIONS[op][par])
    if isinstance(nv, bool) or not isinstance(nv, (int, float)) or not any(abs(float(nv) - a) < 1e-12 for a in allowed):
        raise EditError('new_value must be one of %s for %s.%s' % (allowed, op, par))
    nv = float(next(a for a in allowed if abs(float(nv) - a) < 1e-12))
    old = float(step[par])
    if abs(nv - old) < 1e-12:
        raise EditError('new_value equals the current value %s of that leaf; an edit must change it' % old)
    child = copy.deepcopy(parent_policy)
    (child['default'] if loc == 'default' else child['rules'][ri])['steps'][si][par] = nv
    return child, {'location': loc, 'rule_index': ri, 'step_index': si, 'op': op, 'parameter': par, 'old_value': old, 'new_value': nv}


def enumerate_neighbours(parent_policy, table, registered_keys, parent_key):
    """Every single-leaf edit of the parent in a fixed order (default steps, then rules in order; within a leaf the other
    allowed values ascending), compiled on `table`. status: kept | inactive_branch (assignment equals the parent) |
    duplicate (equals a registered assignment or an earlier kept neighbour). Deterministic; no scores read."""
    out, seen = [], set(registered_keys)
    branches = [('default', None, parent_policy['default'])] + [('rule', i, r) for i, r in enumerate(parent_policy.get('rules', []) or [])]
    for loc, ri, branch in branches:
        for si, step in enumerate(branch['steps']):
            par = PARAM_OF[step['op']]
            for v in spec.ACTIONS[step['op']][par]:
                if abs(float(v) - float(step[par])) < 1e-12:
                    continue
                child, rec = derive_policy(parent_policy, {'location': loc, 'rule_index': ri, 'step_index': si, 'parameter': par, 'new_value': v})
                c = policy.compile_policy(child, table)
                status = 'inactive_branch' if c['key'] == parent_key else 'duplicate' if c['key'] in seen else 'kept'
                out.append({**rec, 'status': status, 'key': c['key'], 'policy': c['policy'], 'assignment': c['assignment']})
                if status == 'kept':
                    seen.add(c['key'])
    return out


def array_diff(parent_ref, child_ref, changed_assignment):
    """Actual material change of a derived child against its parent, per entity (arrays, not effects)."""
    changed = set(int(i) for i in changed_assignment)
    with np.load(parent_ref.path) as p, np.load(child_ref.path) as c:
        pX, pY, cX, cY = p['Xc'], p['Yc'], c['Xc'], c['Yc']
        arrays_changed = [e for e in range(pX.shape[0]) if not (np.array_equal(pX[e], cX[e]) and np.array_equal(pY[e], cY[e]))]
        max_abs = {str(e): float(max(np.abs(cX[e] - pX[e]).max(), np.abs(cY[e] - pY[e]).max())) for e in arrays_changed}
        rms = {str(e): float(np.sqrt(((cX[e] - pX[e]) ** 2).mean())) for e in arrays_changed}
    unchanged_identical = all(e in changed for e in arrays_changed)   # no array moved outside the edited assignment
    return {'entities_changed_assignment': sorted(changed), 'entities_changed_arrays': arrays_changed, 'n_entities_changed_arrays': len(arrays_changed),
            'unchanged_entities_bit_identical': unchanged_identical, 'max_abs_change_by_entity': max_abs, 'x_rms_change_by_entity': rms,
            'note': 'actual array change of the child material relative to the parent material; not a prediction of training effect'}


def attach_derivation(adapter, mid, parent, edit_record, changed, branch_entities):
    """Record parent / single-leaf edit / actual array change in the built plan's spec and its compiled.json."""
    ref, pref = adapter.refs[mid], adapter.refs[parent]
    if ref.alias_of is not None:
        raise RuntimeError('derived child %s aliased an existing material despite semantic dedup' % mid)
    diff = array_diff(pref, ref, changed)
    if not diff['unchanged_entities_bit_identical'] or set(diff['entities_changed_arrays']) - set(changed):
        raise RuntimeError('derived child %s changed material outside the edited branch' % mid)
    ms = adapter.specs[mid]
    ms.update({'parent_plan_id': parent, 'parameter_edit': dict(edit_record), 'derivation': {**diff, 'branch_entities': list(branch_entities)}})
    context.write_json(adapter.ctx.job_dir / 'materials' / (mid + '__compiled.json'), ms)
    return ms


class RefineAdapter(rt.Adapter):
    """rt.Adapter plus: the derived build form (both Fast arms), semantic dedup before material/fit (both arms), and the
    F_refine second-slot contract in check_evaluate (before the slot is counted or any fit reserved)."""
    def __init__(self, run_dir, job, ledger, repo, *, arm, **kw):
        super().__init__(run_dir, job, ledger, repo, **kw)
        if arm not in FAST_ARMS:
            raise ValueError('refine adapter needs a Fast arm of this study')
        self.arm = arm

    # --- real candidate state (also after restore) ---
    def _complete(self, pid):
        c = self.fitted.get(pid)
        return c is not None and len(c.model_refs) == len(self.seeds) and all(c.model_refs) and len(c.ca_losses) == len(self.seeds)

    def complete_ids(self):
        return [p for p in self.specs if self._complete(p)]

    def n_new_complete(self):
        return sum(1 for p in self.complete_ids() if p not in BASELINE_IDS)

    def is_derived(self, pid):
        return (self.specs.get(pid) or {}).get('parent_plan_id') is not None

    def _reject(self, message, **details):
        if self.tool_error_feedback:
            raise br.ToolInputError(message, **details)
        raise ValueError(message)

    def check_compiled(self, plan_id, compiled):
        for pid, ms in self.specs.items():
            if policy.assignment_key(ms['assignment']) == compiled['key']:
                self._reject('the expanded assignment of %r is identical to the registered plan %r; use that id instead (evaluate on an evaluated id '
                             'is a cached read and takes no new slot)' % (plan_id, pid), existing_plan_id=pid)

    def build_material(self, arguments, *, remaining_seconds):
        if set(arguments) == {'plan_id', 'policy'}:
            return super().build_material(arguments, remaining_seconds=remaining_seconds)
        if set(arguments) != {'plan_id', 'parent_plan_id', 'parameter_edit', 'rationale'}:
            self._reject('build_material takes {plan_id, policy} or the derived form {plan_id, parent_plan_id, parameter_edit, rationale}')
        mid, parent, rationale = arguments['plan_id'], arguments['parent_plan_id'], arguments['rationale']
        if not isinstance(mid, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}', mid):
            self._reject('unsafe plan id')
        if mid in self.refs:
            self._reject('plan id already exists; use existing plan or a new id')
        legal = [p for p in self.complete_ids() if self.specs[p]['policy']['default']['steps'] or self.specs[p]['policy']['rules']]
        if self.arm == 'f_refine' and self.n_new_complete() == 0:
            self._reject('the derived form is not available in this arm before the first new evaluation has completed; build the first plan as a complete '
                         'policy (or commit an evaluated plan)', new_evaluations_completed=0)
        if not isinstance(parent, str) or parent not in self.specs:
            self._reject('parent_plan_id must be a plan of this run', legal_parents=legal)
        if not self._complete(parent):
            self._reject('parent %r is not fully evaluated in this run (all three seeds fitted with C_A returned)' % parent, legal_parents=legal)
        if not isinstance(rationale, str) or not rationale.strip():
            self._reject('rationale must be a nonempty string')
        try:
            child, rec = derive_policy(self.specs[parent]['policy'], arguments['parameter_edit'])
        except EditError as exc:
            self._reject(str(exc), parent_plan_id=parent, parent_policy=self.specs[parent]['policy'], legal_parents=legal)
        child['rationale'] = rationale
        try:
            compiled = policy.compile_policy(child, self.overview()['entities'])
        except policy.PolicyError as exc:
            self._reject(str(exc), parent_plan_id=parent)
        pa = self.specs[parent]['assignment']
        changed = [i for i, (a, b) in enumerate(zip(compiled['assignment'], pa)) if a != b]
        target = -1 if rec['location'] == 'default' else rec['rule_index']
        branch_entities = [i for i, r in enumerate(self.specs[parent]['rule_index']) if r == target]
        if not changed:
            self._reject('the edited branch (%s) applies to no current entity of the parent; the material would equal the parent' % ('default' if target == -1 else 'rule %d' % target),
                         parent_plan_id=parent, entities_by_branch={str(i): self.specs[parent]['rule_index'].count(i) for i in range(-1, len(self.specs[parent]['policy']['rules']))})
        c = super().build_material({'plan_id': mid, 'policy': child}, remaining_seconds=remaining_seconds)   # compile again, dedup, build
        ms = attach_derivation(self, mid, parent, rec, changed, branch_entities)
        return br.Candidate(mid, ms, c.job_id)

    def check_evaluate(self, plan_id, candidate):
        if self.arm != 'f_refine' or self.n_new_complete() != 1 or self.is_derived(plan_id):
            return
        self._reject('second-slot contract of this arm: the second new evaluation must be a plan built with the derived form (parent_plan_id + parameter_edit) '
                     'from a fully evaluated parent of this run; %r was built as a complete policy. Build a derived plan and evaluate it, keep %r for the third '
                     'slot, or commit an evaluated plan.' % (plan_id, plan_id), legal_parents=self.complete_ids(), new_evaluations_completed=1)


def arm_contracts(arm):
    """Shared contracts + the three-slot facts, the derived form and the identity rule (both arms) + this arm's slot contract."""
    out = br.json_copy(CONTRACTS)
    out['build_material']['derived_form'] = br.json_copy(DERIVED_FORM)
    out['build_material']['identity_rule'] = IDENTITY_RULE
    out['build_material']['process_note'] = PROCESS_NOTE
    out['evaluate']['meaning'] = 'Train the complete shared plan under 3 frozen paired seeds; return C_A only. At most 3 new complete plans in this run.'
    if arm == 'f_refine':
        out['evaluate']['slot_contract'] = ('CONTRACT of this arm, executed by the Runtime: the first new evaluation is a plan built as a complete policy (observe first '
                                            'or commit an evaluated baseline instead, as you judge). If a second new evaluation is used, it must be a plan built with the '
                                            'derived form (one numeric leaf of a fully evaluated parent of this run, chosen by you; FixedMixup qualifies). The derived '
                                            'form is unavailable before the first new evaluation has completed. A third new evaluation may be a complete policy or '
                                            'derived. A rejected evaluate is a tool input error within the correction budget; it consumes no slot or fit. No slot has '
                                            'to be used; any evaluated plan, including None or FixedMixup, may be committed at any time.')
    elif arm == 'f_free':
        out['evaluate']['slot_contract'] = ('This arm has no requirement on whether or when the derived form is used: up to three new evaluations, each built as a '
                                            'complete policy or with the derived form, in any order; any evaluated plan, including None or FixedMixup, may be '
                                            'committed at any time.')
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
    return {'study': 'paired_refinement', 'jobs': jobs, 'seeds': list(SEEDS), 'order': {j: list(o) for j, o in ORDER.items()}, 'arms': list(ARMS),
            'pool_arms': list(POOL_ARMS), 'fast_arms': list(FAST_ARMS),
            'random_supply': {'n': N_RANDOM, 'max_proposals': dc.MAX_PROPOSALS, 'seed_base': PROPOSAL_SEED_BASE,
                              'rule': 'proposal_seed(j) = %d + 1000*job_index + j, j ascending 0..63; policy.random_policy(seed) compiled on the T-only overview; '
                                      'assignments equal to None, FixedMixup or an earlier kept proposal are skipped; the first four kept are R1..R4' % PROPOSAL_SEED_BASE,
                              'generator': dc.GENERATOR_NOTE, 'visible_to_fast': False,
                              'incomplete': 'SUPPLY_INCOMPLETE marks the job; random_global and random_local are not run; Fast arms and Menu still run'},
            'random_global': {'plans': 'R1, R2, R3', 'rule': 'argmin three-seed C_A mean among FixedMixup, None, R1, R2, R3; ties in that order; 0 LLM'},
            'random_local': {'slot1': 'R1', 'slot2': 'parent = argmin three-seed C_A among FixedMixup, None, R1 that have >=1 kept single-leaf neighbour (ties Fixed, None, R1); '
                                              'neighbours enumerated in fixed order (default steps, rules in order, other legal values ascending), inactive and duplicate '
                                              'assignments dropped; index = RandomState(%d + 1000*job_index).randint(n_kept); drawn once' % LOCAL_SEED_BASE,
                             'slot3': 'first of R2, R3, R4 whose assignment is not registered in this arm', 'empty': 'LOCAL_NEIGHBORHOOD_EMPTY: stop and deliver from completed plans',
                             'rule': 'argmin three-seed C_A mean among FixedMixup, None, then evaluation order; 0 LLM'},
            'menu': {'plans': [[m, p] for m, p in MENU], 'rule': 'argmin three-seed C_A mean among FixedMixup, None, MenuU, MenuFM; ties in that order; 0 LLM'},
            'derived_form': DERIVED_FORM, 'identity_rule': IDENTITY_RULE, 'process_note': PROCESS_NOTE,
            'second_slot_contract': {'applies_to': 'f_refine only', 'checked_in': 'RefineAdapter.check_evaluate inside run_job, before the slot is counted and before any fit is reserved',
                                     'build_rule': 'derived form refused in f_refine while no new evaluation has completed; parent must be fully evaluated in this arm (FixedMixup qualifies; None has no leaf)',
                                     'identity': 'derived = built with the derived form (spec.parent_plan_id); a complete policy that happens to equal a neighbour is not derived',
                                     'origin': 'researcher-fixed process intervention of this study; not Slow output; grants Slow nothing'},
            'arm_contracts': {a: arm_contracts(a) for a in FAST_ARMS},
            'caps': caps, 'planned_fits': 96, 'http_cap': 2 * caps['max_llm_requests'], 'exposure': 'EXPOSED_DEVELOPMENT',
            'population_note': 'one 32-entity group G5 (CSV string-sorted names [160:192]) at two disjoint times a=.65/.85; the next whole group after G4; same source, not independent replications',
            'knowledge': 'H0 empty for every Fast arm; no Slow; no historical card, draft, D1, old E ranking or construction seed',
            'per_fast_job_limits': {**LIMITS, 'wall_seconds': 'remaining package wall'}, 'max_tool_corrections': 2, 'evidence_roundtrip': True,
            'tool_contracts': CONTRACTS, 'tool_semantics': rt.TOOL_SEMANTICS, 'consumer': spec.CONSUMER,
            'environment': dc.environment(),
            'unknown_usage_rule': 'a failed transport attempt or a response without usage leaves cost UNKNOWN; at most one same-request retransmission; later paid calls refused; executor may not accept unknown usage without explicit operator authorization',
            'utility_rule': ('main: Delta_free = loss(F_free) - loss(F_refine); necessary: None/FixedMixup/Menu/Random-global/Random-local - F_refine; sub-mechanism: parent - child per '
                             'actual derived edit (C_A/C_B/E by seed); Random-global - Random-local; three-seed E means per job, positive = F_refine better; first-seed delivery separate; no significance gate.')}


def generate_supply(job, table):
    """R1..R4 by the frozen stream of task §7.1 (dc.generate_pool with this study's seed base; ids renamed D->R)."""
    pool = dc.generate_pool(job, table, job_index=JOBS.index(job), seed_base=PROPOSAL_SEED_BASE)
    for d in pool['drafts']:
        d['draft_id'] = 'R' + d['draft_id'][1:]
    for a in pool['attempts']:
        if 'draft_id' in a:
            a['draft_id'] = 'R' + a['draft_id'][1:]
        if a['status'].startswith('duplicate_of_D'):
            a['status'] = 'duplicate_of_R' + a['status'][len('duplicate_of_D'):]
    pool['status'] = 'COMPLETE' if pool['status'] == 'COMPLETE' else 'SUPPLY_INCOMPLETE'
    pool['visible_to_fast'] = False
    return pool


def prepare(root, led, caps):
    """Stage 2: both Targets' T only (eligibility, roster binding); both random supplies generated and frozen before any fit.
    Returns (common adapters of eligible jobs, supplies by job; None = ELIGIBILITY_FAILED). Fast arms never receive the supply."""
    context.write_json(root / 'frozen_config.json', frozen_config(caps))
    if led.s['fit_attempts']:
        raise RuntimeError('random supply must be frozen before any fit')
    commons, pools = {}, {}
    for job in JOBS:
        shared = root / (job + '_common')
        shared.mkdir()
        try:
            adapter = rt.Adapter(shared, job, led, REPO, roster=ROSTERS[job], seeds=SEEDS)
        except RuntimeError as exc:
            if 'ELIGIBILITY_FAIL' not in str(exc):
                raise
            context.write_json(root / (job + '_eligibility.json'), {'job': job, 'status': 'ELIGIBILITY_FAILED', 'reason': str(exc), 'note': 'no population, column or time change; the other job continues'})
            pools[job] = None
            continue
        commons[job] = adapter
        ov = adapter.overview()
        dc.target_checks(root, job, adapter)
        pools[job] = generate_supply(job, ov['entities'])
        context.write_json(root / (job + '_random_pool.json'), pools[job])
    if led.s['fit_attempts']:
        raise RuntimeError('a fit happened during random supply generation')
    context.write_json(root / FROZEN_MARKER, {'epoch': time.time(), 'jobs': {j: (p['status'] if p else 'ELIGIBILITY_FAILED') for j, p in pools.items()},
                                              'fit_attempts_at_freeze': led.s['fit_attempts'], 'note': 'both random supplies generated from T only before any fit; no C_A opened; not shown to Fast'})
    return commons, pools


def load_pools(root):
    return {job: (context.read_json(root / (job + '_random_pool.json')) if (root / (job + '_random_pool.json')).exists() else None) for job in JOBS}


# ----------------------------------------------------------------------------- arms
def adapter_factory(arm, pool):
    return functools.partial(RefineAdapter, arm=arm)


def random_local_branch(root, job, led, shared, pool, *, seeds, roster):
    """0-LLM local search (task §7.2): R1 -> one random legal single-leaf neighbour of the C_A-best parent -> first fresh R;
    argmin three-seed C_A among FixedMixup, None, then evaluation order."""
    seeds = tuple(seeds)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(shared / job, root / job)
    adapter = rt.Adapter(root, job, led, REPO, roster=roster, seeds=seeds)
    by_id = {c.plan_id: c for c in adapter.baselines()}
    order = ['FixedMixup', 'None']
    trace = []
    drafts = {d['draft_id']: d for d in pool['drafts']}

    def emit(event, **kw):
        row = {'event_id': f'{job}:{len(trace)}', 'event': event, **kw}
        trace.append(row)
        base.event_sink(root / 'trace.jsonl')(row)

    def train(mid, pol, expected, **extra):
        c = adapter.build_material({'plan_id': mid, 'policy': pol}, remaining_seconds=led.remaining())
        if c.material_spec['assignment'] != expected:
            raise RuntimeError('frozen plan %s compiled to a different assignment' % mid)
        if extra:
            ms = attach_derivation(adapter, mid, extra['parent_plan_id'], extra['parameter_edit'], extra['changed'], extra['branch_entities'])
            c = br.Candidate(mid, ms, c.job_id)
        emit('local_material', plan_id=mid, material_spec=c.material_spec, **{k: v for k, v in extra.items() if k in ('parent_plan_id', 'parameter_edit')})
        c = adapter.evaluate(c, seeds, feedback=True, remaining_seconds=led.remaining())
        by_id[mid] = c
        order.append(mid)
        emit('local_evaluated', plan_id=mid, feedback=c.feedback(seeds, 2, 32))
        return c
    record = {'job': job, 'job_index': int(pool['job_index']), 'seed': LOCAL_SEED_BASE + 1000 * int(pool['job_index']), 'status': 'COMPLETE'}
    train('R1', drafts['R1']['policy'], drafts['R1']['assignment'])
    table = adapter.overview()['entities']
    registered = {pid: policy.assignment_key(adapter.specs[pid]['assignment']) for pid in adapter.specs}
    parents = {}
    for pid in ('FixedMixup', 'None', 'R1'):
        nb = enumerate_neighbours(adapter.specs[pid]['policy'], table, set(registered.values()), registered[pid])
        parents[pid] = {'c_a_mean': by_id[pid].feedback(seeds, 2, 32)['mean_loss'], 'n_kept': sum(n['status'] == 'kept' for n in nb), 'n_enumerated': len(nb),
                        'neighbours': [{k: n[k] for k in ('location', 'rule_index', 'step_index', 'op', 'parameter', 'old_value', 'new_value', 'status')} for n in nb], '_full': nb}
    record['parents'] = {p: {k: v for k, v in parents[p].items() if k != '_full'} for p in parents}
    eligible = [p for p in ('FixedMixup', 'None', 'R1') if parents[p]['n_kept']]
    emit('local_parent_selection', candidates=record['parents'], eligible=eligible)
    if eligible:
        parent = min(eligible, key=lambda p: (parents[p]['c_a_mean'], ('FixedMixup', 'None', 'R1').index(p)))
        kept = [n for n in parents[parent]['_full'] if n['status'] == 'kept']
        idx = int(np.random.RandomState(record['seed']).randint(len(kept)))
        chosen = kept[idx]
        target = -1 if chosen['location'] == 'default' else chosen['rule_index']
        pa = adapter.specs[parent]['assignment']
        changed = [i for i, (a, b) in enumerate(zip(chosen['assignment'], pa)) if a != b]
        record.update({'parent': parent, 'n_kept': len(kept), 'drawn_index': idx,
                       'edit': {k: chosen[k] for k in ('location', 'rule_index', 'step_index', 'op', 'parameter', 'old_value', 'new_value')}})
        train('L2', chosen['policy'], chosen['assignment'], parent_plan_id=parent, parameter_edit=record['edit'], changed=changed,
              branch_entities=[i for i, r in enumerate(adapter.specs[parent]['rule_index']) if r == target])
        registered['L2'] = chosen['key']
        third = next((d for d in ('R2', 'R3', 'R4') if policy.assignment_key(drafts[d]['assignment']) not in registered.values()), None)
        record['third'] = third
        if third is not None:
            train(third, drafts[third]['policy'], drafts[third]['assignment'])
        else:
            record['status'] = 'THIRD_SLOT_EMPTY'
    else:
        record.update({'status': 'LOCAL_NEIGHBORHOOD_EMPTY', 'parent': None})
    context.write_json(root / 'local_search.json', record)
    chosen_c = min((by_id[m] for m in order), key=lambda c: (c.feedback(seeds, 2, 32)['mean_loss'], order.index(c.plan_id)))
    reason = 'Random-local baseline: minimum C_A three-seed mean among FixedMixup, None, then evaluation order (%s); frozen tie order; 0 LLM.' % ', '.join(order[2:])
    commit.commit(root, 'electricity', job, chosen_c.plan_id, reason, seeds[0])
    emit('committed', plan_id=chosen_c.plan_id, delivery_model_ref=chosen_c.model_refs[0])
    result = br.RunResult('COMPLETE', job, 0, chosen_c.plan_id, chosen_c.model_refs[0], 0, 0, len(order) - 2, trace)
    context.write_json(root / 'branch_result.json', asdict(result))
    print('RANDOM_LOCAL', job, 'commit', chosen_c.plan_id, record['status'], flush=True)
    return result


def run_arm(arm, path, job, led, client, shared, pool, *, seeds, roster, max_tool_corrections):
    if arm in FAST_ARMS:
        return base.branch(path, job, br.Knowledge(), led, client, shared, evidence_roundtrip=True, max_tool_corrections=max_tool_corrections,
                           tool_contracts=arm_contracts(arm), seeds=seeds, roster=roster, adapter_factory=adapter_factory(arm, pool), limits=LIMITS)
    if arm == 'random_global':
        plans = [(d['draft_id'], d['policy']) for d in pool['drafts'][:3]]
        return dc.fixed_candidates_branch(path, job, led, shared, plans=plans, tie_order=GLOBAL_TIE_ORDER, tag='random', seeds=seeds, roster=roster,
                                          reason='Random-global baseline: minimum C_A three-seed mean among FixedMixup, None, R1, R2, R3; frozen tie order; frozen random stream; 0 LLM.',
                                          expected_assignments={d['draft_id']: d['assignment'] for d in pool['drafts'][:3]})
    if arm == 'random_local':
        return random_local_branch(path, job, led, shared, pool, seeds=seeds, roster=roster)
    if arm == 'menu':
        return dc.fixed_candidates_branch(path, job, led, shared, plans=list(MENU), tie_order=MENU_TIE_ORDER, tag='menu', seeds=seeds, roster=roster,
                                          reason='Fixed menu: minimum C_A three-seed mean among FixedMixup, None, MenuU, MenuFM; frozen tie order; no history read.')
    raise ValueError(arm)


def resume_arm(arm, path, job, led, client, pool, *, seeds, roster, max_tool_corrections):
    if arm not in FAST_ARMS:
        raise RuntimeError('control branch cannot be resumed')
    return base.resume_branch(path, job, br.Knowledge(), led, client, max_tool_corrections=max_tool_corrections, tool_contracts=arm_contracts(arm),
                              seeds=seeds, roster=roster, adapter_factory=adapter_factory(arm, pool), evidence_roundtrip=True, limits=LIMITS)


# ----------------------------------------------------------------------------- report
def refinement_events(root, name, baseline_ids=BASELINE_IDS):
    """Process record of one Fast branch from its real trace: per-response tools, builds (form, parent, edit), evaluations
    with the response that issued them, deferrals, rejections, commit and stop point. No scores are interpreted here."""
    rows = [json.loads(l) for l in (root / name / 'trace.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    current, per_response, builds, evals, rejections, deferred, commit_row = 0, {}, {}, [], [], [], None
    for i, x in enumerate(rows):
        if x['event'] == 'fast_response':
            current = x['number']
            per_response[current] = [a['tool'] for a in x['response'].get('actions', [])] if isinstance(x['response'], dict) else None
        elif x['event'] == 'tool_completed' and x['tool'] == 'build_material':
            a = next(r['arguments'] for r in reversed(rows[:i]) if r['event'] == 'tool_started' and r['tool'] == 'build_material')
            ms = x['output']['material_spec']
            builds[x['output']['plan_id']] = {'response': current, 'form': 'derived' if 'parent_plan_id' in a else 'policy', 'parent_plan_id': ms.get('parent_plan_id'),
                                              'parameter_edit': ms.get('parameter_edit'), 'n_entities_changed_arrays': (ms.get('derivation') or {}).get('n_entities_changed_arrays'),
                                              'rationale': ms['policy'].get('rationale')}
        elif x['event'] == 'tool_completed' and x['tool'] == 'evaluate':
            evals.append({'plan_id': x['output']['plan_id'], 'response': current, 'new': x['output']['plan_id'] not in baseline_ids and x['output']['plan_id'] not in [e['plan_id'] for e in evals]})
        elif x['event'] == 'tool_rejected':
            msg = x['error'].get('message', '')
            rejections.append({'tool': x['tool'], 'response': current, 'kind': 'second_slot' if msg.startswith('second-slot') else 'derived_form' if 'derived form' in msg else 'identity' if 'identical to the registered' in msg else 'edit' if x['tool'] == 'build_material' else 'other',
                               'message': msg[:200], 'unexecuted': [a['tool'] for a in x.get('unexecuted_actions', [])]})
        elif x['event'] == 'action_batch_deferred':
            deferred.append({'response': current, 'actions': [a['tool'] for a in x['actions']]})
        elif x['event'] == 'tool_completed' and x['tool'] == 'commit':
            commit_row = {'plan_id': x['output']['plan_id'], 'response': current, 'reason': x['output'].get('reason')}
    new = [e for e in evals if e['new']]
    derived_evaluated = [b for p, b in builds.items() if b['form'] == 'derived' and any(e['plan_id'] == p for e in new)]
    return {'responses': per_response, 'n_responses': len(per_response), 'builds': builds, 'evaluations': evals, 'new_evaluations': [e['plan_id'] for e in new],
            'new_evaluation_responses': [e['response'] for e in new], 'second_after_first_feedback': (new[1]['response'] > new[0]['response']) if len(new) >= 2 else None,
            'derived_built': [p for p, b in builds.items() if b['form'] == 'derived'], 'derived_evaluated': [{'plan_id': p, **b} for p, b in builds.items() if b in derived_evaluated],
            'built_not_evaluated': [p for p in builds if p not in [e['plan_id'] for e in evals]], 'rejections': rejections, 'deferred': deferred,
            'commit': commit_row, 'stop_point_new_evaluations': len(new), 'inspections': sum(1 for x in rows if x['event'] == 'tool_completed' and x['tool'] in ('inspect_data', 'inspect_material')),
            'compares': sum(1 for x in rows if x['event'] == 'tool_completed' and x['tool'] == 'compare')}


def finish_report(root, r, jobs, order, seeds):
    r['study'] = 'paired_refinement'
    r.pop('treatment_present', None)
    pools = load_pools(root)
    r['random_supply'] = {j: ({k: p[k] for k in ('status', 'n_proposals_tried', 'visible_to_fast')} | {'plans': [{k: d[k] for k in ('draft_id', 'j', 'proposal_seed', 'n_rules', 'n_identity', 'n_entities_changed')} for d in p['drafts']]}) if p else None for j, p in pools.items()}
    r['eligibility'] = {j: (context.read_json(root / (j + '_eligibility.json')) if (root / (j + '_eligibility.json')).exists() else {'status': 'ELIGIBLE'}) for j in jobs}
    r['skipped'] = {p.name[:-13]: context.read_json(p) for p in sorted(root.glob('*_skipped.json'))}
    r['supply_frozen'] = context.read_json(root / FROZEN_MARKER) if (root / FROZEN_MARKER).exists() else None
    r['resume'] = context.read_json(root / 'resume.json') if (root / 'resume.json').exists() else None
    r['target_checks'] = {j: context.read_json(root / (j + '_target_checks.json')) for j in jobs if (root / (j + '_target_checks.json')).exists()}
    dc.annotate_branches(root, r, pools, seeds)
    seeds = tuple(seeds)

    def delta(cells_a, cells_b, block):
        ds = [cells_a[s][block]['normalized_mse_macro'] - cells_b[s][block]['normalized_mse_macro'] for s in seeds
              if s in cells_a and s in cells_b and cells_a[s].get(block) and cells_b[s].get(block)]
        if len(ds) != 3:
            return None
        return {'delta_by_seed': ds, 'mean_delta': statistics.mean(ds), 'seed_se': statistics.stdev(ds) / 3 ** .5, 'first_seed_delta': ds[0], 'positive_means': 'second_is_better'}

    def blocks(cells_a, cells_b):
        return {block: delta(cells_a, cells_b, block) for block in ('c_a', 'c_b', 'external')} if cells_a and cells_b else None

    def committed_cells(b):
        return sp._seed_cells(b, b['controller']['committed_plan_id']) if b and b['controller']['status'] == 'COMPLETE' else None

    for name, b in r['branches'].items():
        b['refinement'] = refinement_events(root, name) if b['arm'] in FAST_ARMS else None
        b['local_search'] = context.read_json(root / name / 'local_search.json') if (root / name / 'local_search.json').exists() else None
        # every actual parent -> child edit of this branch (Fast or Random-local), with paired deltas where both are complete
        b['derived_edits'] = {}
        for p, rec in b['new_candidate_materials'].items():
            comp = root / name / b['job'] / 'materials' / (p + '__compiled.json')
            ms = context.read_json(comp) if comp.exists() else {}
            if ms.get('parent_plan_id') is None:
                continue
            b['derived_edits'][p] = {'parent_plan_id': ms['parent_plan_id'], 'parameter_edit': ms['parameter_edit'], 'derivation': {k: v for k, v in ms['derivation'].items() if k not in ('max_abs_change_by_entity', 'x_rms_change_by_entity')},
                                     'parent_minus_child': blocks(sp._seed_cells(b, ms['parent_plan_id']), sp._seed_cells(b, p)), 'child_committed': b['controller']['committed_plan_id'] == p}
    r['comparisons'] = {}
    for job in jobs:
        refine = r['branches'].get(job + '_f_refine')
        comp = {'positive_means': 'F_refine has lower loss', 'reference': 'f_refine committed plan'}
        ref = committed_cells(refine)
        if ref:
            for arm in ('f_free', 'random_global', 'random_local', 'menu'):
                comp[arm + '_minus_f_refine'] = blocks(committed_cells(r['branches'].get(job + '_' + arm)), ref)
            for b0 in BASELINE_IDS:
                comp[b0 + '_minus_f_refine'] = blocks(sp._seed_cells(refine, b0), ref)
            comp['delta_free'] = comp['f_free_minus_f_refine']
        else:
            comp['status'] = 'UNKNOWN: F_refine branch missing or incomplete'
        comp['random_global_minus_random_local'] = blocks(committed_cells(r['branches'].get(job + '_random_global')), committed_cells(r['branches'].get(job + '_random_local')))
        keys, lineage = {}, {}
        for arm in ARMS:
            b = r['branches'].get(job + '_' + arm)
            if b and b['controller']['status'] == 'COMPLETE':
                p = b['controller']['committed_plan_id']
                keys[arm] = (b['new_candidate_materials'].get(p, {}).get('assignment_key') if p not in BASELINE_IDS else p)
                lineage[arm] = b['committed_lineage']
        comp['committed_material_key_by_arm'], comp['committed_lineage_by_arm'] = keys, lineage
        pooled = {}
        for arm in ARMS:
            b = r['branches'].get(job + '_' + arm)
            for p, cells in (b or {}).get('plans', {}).items():
                if len(cells) == 3 and all(c.get('external') for c in cells):
                    pooled[arm + ':' + p] = statistics.mean(c['external']['normalized_mse_macro'] for c in cells)
        comp['retrospective_best_e_across_arms'] = (min(pooled, key=pooled.get), pooled[min(pooled, key=pooled.get)]) if pooled else None
        r['comparisons'][job] = comp
    planned = sum(len(order[j]) for j in jobs)
    r['planned_arms'], r['skipped_arms'] = planned, len(r['skipped'])
    r['status'] = 'COMPLETE' if (len(r['branches']) == planned and not r['failures'] and all(b['controller']['status'] == 'COMPLETE' and b['delivery_e'] is not None for b in r['branches'].values())) else 'PARTIAL'
    r['utility_supported'] = None
    r['utility_scope'] = 'Researcher-fixed process Harness; one Fast trajectory per arm and job, three training seeds per complete plan, one population at two disjoint times; development pilot; no Slow.'
    context.write_json(root / 'result.json', r)
    lines = ['# DEV-BATCH-RESEARCH-PAIRED-REFINEMENT：机器汇总（result.json 派生）', '', '状态：' + r['status'], '',
             '|分支|状态|commit|谱系|新评估|派生已评估|第二槽在反馈后|拒绝|第一seed E|三seed E均值|Fast请求|工具尝试|私有拟合|集合内E遗憾|',
             '|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, b in r['branches'].items():
        q = b.get('refinement') or {}
        lines.append('|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|' % (name, b['controller']['status'], b['controller']['committed_plan_id'], (b['committed_lineage'] or {}).get('identical_to'),
                                                                   q.get('stop_point_new_evaluations', b['controller']['new_evaluations']), len(q.get('derived_evaluated', [])) if q else len(b['derived_edits']),
                                                                   q.get('second_after_first_feedback'), len(q.get('rejections', [])) if q else 0,
                                                                   b['delivery_e'], b['paired_mean_e'], b['controller']['calls'], b['controller']['tool_calls'], b['private_physical_fits'], b['commit_regret_within_observed_candidates']))
    lines += ['', '各作业配对比较（正值 = F_refine 损失更低）：']
    for j, x in r['comparisons'].items():
        lines.append(j + ': ' + json.dumps({k: v for k, v in x.items() if k.endswith('_minus_f_refine') or k == 'delta_free' or k == 'random_global_minus_random_local'}, ensure_ascii=False))
    lines += ['', '预算：' + json.dumps(r.get('budget'), ensure_ascii=False), '', '主报告见 REPORT.md；本文件由 report() 自动生成，不含叙述判断。']
    (root / 'MACHINE_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('PAIRED_REFINEMENT_REPORT', r['status'], flush=True)
    return r
