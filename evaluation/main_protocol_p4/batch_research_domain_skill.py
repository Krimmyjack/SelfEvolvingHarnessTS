"""DEV-DOMAIN-SKILL-V1 study module (docs/DEV_DOMAIN_SKILL_V1_BUILD_TASK_2026-09-16.md).

  --build-profiles   real T-only batch profiles of two neutral domains (D01/D02), domain reference catalog with
                     SMOKE_FIXTURE Skills, source asset registry and the two live configurations (A zero budget, B draft)
  --smoke            the one integration smoke (0 fits, 0 API, 0 new label reads; real T for profiles and materials,
                     scripted clients and explicitly synthetic feedback for control flow)
  --formation STAGE  offline census / propose / select / freeze entry (paid stages refuse unless a FROZEN config with budget)

Live Target runs reuse run_batch_research_roundtrip (--study domain_skill --study-config <json>): the same worker,
label withholding, E barrier and resume boundary; this module supplies the STUDY_MOD hooks (prepare with routing,
run_arm, resume_arm, load_pools, before_labels, finish_report). Numerics stay in methods.ttha.batch_base; the Skill,
profile and routing logic in methods.ttha.domain_skill. No hashing.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha.batch_base import context, policy, spec
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_source_process as sp
from evaluation.main_protocol_p4 import run_batch_research_v1 as base

REPO = Path(__file__).resolve().parents[2]
BUILD_ROOT = REPO / '_scratch' / 'dev_domain_skill_v1_build'
DOMAIN_BINDINGS = {'D01': 'electricity', 'D02': 'traffic'}      # Runtime only; never a matching feature
REFERENCE_JOBS = ('a30', 'a40')                                  # formation reference batches (EXPOSED_DEVELOPMENT)
QUERY_JOBS = ('a50',)                                            # query profile / material smoke (EXPOSED_DEVELOPMENT)
CONTRACTS = rt.contracts_with_semantics()
FIXTURE_REFS = frozenset({'fixture:smoke'})
CONFIG_STATUSES = ('FROZEN', 'BUILD_ZERO_BUDGET', 'DRAFT_NOT_FROZEN', 'SMOKE_CONTROL_FLOW')
STAGES = ('material', 'llm', 'fit', 'labels_c_b', 'labels_e')
REQUIRED_LIVE_STAGES = ('material', 'llm', 'fit', 'labels_c_b')          # labels_e only where the stage opens E (Target)
STUDY_ARMS = dsk.ROUTE_MODES + ('random', 'cand_W1', 'cand_W2')


def _rel(p: Path, root: Path) -> str:
    return str(Path(p).resolve().relative_to(Path(root).resolve())).replace('\\', '/')


def write_once(path, obj) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError('refusing to overwrite a frozen record: %s' % path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    context.write_json(path, obj)


# ============================================================================= fixtures, profiles, catalog
def fixture_skills() -> list:
    """Hand-written SMOKE_FIXTURE cards: generic procedural text, no effect answer, never routable in a live run."""
    common = dict(revision=0, applicability_summary='Smoke fixture for wiring checks: generic procedural text without any domain-specific claim.',
                  compatibility_note='Same task, Consumer, L/H, hourly sampling and T length as this build.',
                  observable_applicability={'const': True}, evidence_refs=['fixture:smoke'], legal_evidence_refs=FIXTURE_REFS,
                  source_stage='fixture', status='SMOKE_FIXTURE')
    return [
        dsk.make_skill(skill_id='SK-FIX-D01', domain_id='D01', principles=None, **common,
                       workflow='[SMOKE_FIXTURE SK-FIX-D01] Read the overview summary before building anything and state one hypothesis about the '
                                'shared training material. If you construct a plan, compare it with the evaluated baselines under the same seeds '
                                'before deciding. Commit an evaluated plan when the remaining budget or the comparison does not justify another fit.'),
        dsk.make_skill(skill_id='SK-FIX-D02', domain_id='D02', **common,
                       workflow='[SMOKE_FIXTURE SK-FIX-D02] Inspect a few rows whose summary values sit at the batch extremes before choosing a '
                                'construction. Build at most one uniform or heterogeneous plan that tests a stated hypothesis, evaluate it, compare '
                                'it with the closest baseline and commit on the paired evidence.',
                       principles='Paired C_A differences carry seed uncertainty only. Row-level loss changes are not causal labels. '
                                  'This text is a smoke fixture, not learned knowledge.'),
    ]


def build_profiles(root: Path = BUILD_ROOT) -> dict:
    """6 real T-only profiles (2 domains x a30/a40/a50) + catalog/runtime bindings + registry + live configs."""
    root = Path(root)
    summary = {'profiles': {}, 'not_qualified': [], 'reads': 'material stage only: rows [t-672, t) of each job'}
    for d, ds in DOMAIN_BINDINGS.items():
        for job in REFERENCE_JOBS + QUERY_JOBS:
            pdir = root / 'profiles' / d / job
            try:
                ctx = context.open_job(ds, job, root / 't_runtime' / d, 'material')
            except RuntimeError as exc:
                if 'ELIGIBILITY_FAIL' not in str(exc) and 'non-finite' not in str(exc):
                    raise
                rec = {'domain_id': d, 'job': job, 'status': 'NOT_QUALIFIED', 'reason': str(exc), 'note': 'no roster or cut substitution'}
                summary['not_qualified'].append(rec)
                pdir.mkdir(parents=True, exist_ok=True)
                context.write_json(pdir / 'not_qualified.json', rec)
                continue
            view, binding = dsk.batch_profile(ctx.overview(), ctx.job)
            pdir.mkdir(parents=True, exist_ok=True)
            context.write_json(pdir / 'profile.json', view)
            context.write_json(pdir / 'runtime_binding.json', {**binding, 'domain_id': d, 'role': 'formation_reference' if job in REFERENCE_JOBS else 'query_smoke'})
            summary['profiles']['%s/%s' % (d, job)] = {'t': ctx.job.t, 'train_rows': list(ctx.job.train_range),
                                                        'null_fields': [f for f, v in view['fields'].items() if v is None]}
    write_catalog(root)
    context.write_json(root / 'source_assets.json', source_assets())
    context.write_json(root / 'live_config_A_zero_budget.json', live_config_a(root))
    context.write_json(root / 'live_config_B_draft.json', live_config_b_draft(root))
    context.write_json(root / 'profiles' / 'build_summary.json', summary)
    print('PROFILES_BUILT', json.dumps({k: len(v) for k, v in summary.items() if isinstance(v, (list, dict))}), flush=True)
    return summary


def write_catalog(root: Path) -> None:
    catalog = {'catalog_version': 1, 'status': 'BUILD_SMOKE_ONLY',
               'note': 'Neutral domain ids and formation reference batches. No domain Skill is formed in this build; the two cards are SMOKE_FIXTURE.',
               'domains': {}, 'skills': [s.to_json() for s in fixture_skills()], 'generic_skill_id': None}
    bindings = {}
    for d, ds in DOMAIN_BINDINGS.items():
        refs = {}
        for i, job in enumerate(REFERENCE_JOBS, 1):
            p = root / 'profiles' / d / job / 'profile.json'
            if p.exists():
                refs['%s.ref%d' % (d, i)] = {'job': job, 'profile': _rel(p, root), 'runtime_binding': _rel(p.with_name('runtime_binding.json'), root)}
        queries = {'%s.query%d' % (d, i): {'job': job, 'profile': _rel(root / 'profiles' / d / job / 'profile.json', root)}
                   for i, job in enumerate(QUERY_JOBS, 1) if (root / 'profiles' / d / job / 'profile.json').exists()}
        catalog['domains'][d] = {'reference_batch_ids': sorted(refs), 'formation_status': 'NOT_FORMED__B_PACKAGE'}
        bindings[d] = {'dataset': ds, 'reference_batches': refs, 'query_smoke_batches': queries}
    (root / 'catalog').mkdir(parents=True, exist_ok=True)
    context.write_json(root / 'catalog' / 'domain_catalog.json', catalog)
    context.write_json(root / 'catalog' / 'runtime_bindings.json', {'note': 'RUNTIME ONLY: never part of a matching view', 'domains': bindings})


def load_catalog(root: Path):
    root = Path(root)
    catalog = context.read_json(root / 'catalog' / 'domain_catalog.json')
    bindings = context.read_json(root / 'catalog' / 'runtime_bindings.json')['domains']
    skills = [dsk.skill_from_json(s) for s in catalog['skills']]
    if len({s.skill_id for s in skills}) != len(skills):
        raise ValueError('duplicate skill ids in catalog')
    return catalog, bindings, skills


def domain_refs(root: Path, catalog: dict, bindings: dict, *, query_binding: dict | None = None):
    """{domain_id: [reference profile views]} from formation batches only. A reference of the query's own source whose T
    overlaps the query workload is refused (the target batch never enters its own reference)."""
    out, used = {}, []
    for d in sorted(catalog['domains']):
        views = []
        for ref_id in catalog['domains'][d]['reference_batch_ids']:
            b = bindings[d]['reference_batches'][ref_id]
            rb = context.read_json(Path(root) / b['runtime_binding'])
            if query_binding is not None and rb['dataset'] == query_binding['dataset']:
                lo, hi = query_binding['workload_rows']
                if rb['train_rows'][0] < hi and lo < rb['train_rows'][1]:
                    raise PermissionError('reference batch %s overlaps the query workload' % ref_id)
            view = context.read_json(Path(root) / b['profile'])
            dsk.assert_identity_free(view)
            views.append(view)
            used.append(ref_id)
        out[d] = views
    return out, used


def source_assets() -> dict:
    """Registry only: existing paths, source domain, evidence type and gaps. No result is re-read or re-scored here."""
    s = REPO / '_scratch'

    def item(rel, domain, evidence, note):
        return {'path': '_scratch/' + rel, 'exists': (s / rel).exists(), 'domain_id': domain, 'evidence_type': evidence, 'note': note}
    return {
        'status': 'REGISTRY_ONLY', 'purpose': 'formation-material availability for B; nothing here was fed to a model',
        'rules': ['source boundary = dataset + roster + time', 'E-bearing files are never Slow input; census reads C_A and opened C_B only',
                  'an Electricity trajectory is never relabeled as Traffic; effect-only evidence is not process evidence'],
        'domains': {
            'D01': {'process_trajectories': [
                item('dev_batch_research_workflow_v1', 'D01', 'batch Fast tool-loop trajectories + C_A + post-commit C_B (+E files present)', 'a55/a65; H0 and one Slow card'),
                item('dev_batch_research_roundtrip', 'D01', 'batch Fast trajectories + Random control', 'a75/a85; old vs evidence roundtrip'),
                item('dev_batch_research_fast_baselines', 'D01', 'batch Fast trajectories + Random control', 'a90/a95'),
                item('dev_batch_research_source_process', 'D01', 'batch Fast trajectories under 4 knowledge arms + Random', 'a89/a93; includes one Slow-formed experiment_guidance'),
                item('dev_batch_research_skill_revision', 'D01', 'batch Fast trajectories + Menu', 'a97 G0/G1 explicit rosters'),
                item('dev_batch_research_draft_construction_g3', 'D01', 'batch Fast trajectories + Random + Menu', 'a65_g3/a85_g3 explicit roster'),
                item('dev_batch_research_exploration_quota', 'D01', 'batch Fast trajectories + Random + Menu', 'a65_g4/a85_g4'),
                item('dev_batch_research_paired_refinement', 'D01', 'batch Fast trajectories + local/global random + Menu', 'a65_g5/a85_g5'),
                item('dev_batch_research_measured_start', 'D01', 'batch Fast trajectories with trained random start + controls', 'a65_g7/a85_g7'),
            ], 'effect_only': [
                item('ts_augmentation_preflight', 'D01', 'fixed-operator arms, MLP, other C/E layout', 'a30/a40 E opened'),
                item('ts_augmentation_confirmation', 'D01', 'fixed-operator arms, MLP, other C/E layout', 'a50/a60 E opened'),
            ], 'gap': 'process trajectories exist; B still has to freeze a whitelist and a workload boundary before any formation call'},
            'D02': {'process_trajectories': [], 'effect_only': [
                item('ts_augmentation_preflight', 'D02', 'fixed-operator arms (None/Copy/FreqMask/FreqMix/TimeMixup/selector), MLP, other C/E layout', 'a30/a40 E opened'),
                item('ts_augmentation_confirmation', 'D02', 'fixed-operator arms, MLP, other C/E layout', 'a50/a60 E opened'),
                item('substrate_screen', 'D02', 'one-program shared Ridge terrain (conditional parts retracted)', 'different Consumer; 6 columns'),
                item('flash_cross_domain_dev', 'D02', 'seven fixed Workflows, shared Ridge, other geometry', 'different Consumer and geometry'),
            ], 'gap': 'EFFECT_EVIDENCE_PRESENT__PROCESS_TRAJECTORIES_ABSENT: no batch Fast tool-loop trajectory exists; B must generate legal D02 trajectories within budget before a D02 Workflow can be formed'},
        },
        'exposure_note': 'a30/a40/a50 of both sources are EXPOSED_DEVELOPMENT (earlier packages opened their E); this build reads their T only',
    }


# ============================================================================= live study hooks (run_batch_research_roundtrip --study domain_skill)
CFG: dict = {}
JOBS: tuple = ()
ORDER: dict = {}
ROSTERS: dict = {}
SEEDS: tuple = ()
JOB_DATASETS: dict = {}
JOB_DOMAINS: dict = {}
LIMITS: dict = {}
PLANNED_FITS = 0
POOL_ARMS = ()
POOL_NAME = 'route record'
FROZEN_MARKER = 'routes_frozen.json'
PREPARE_WITH_CLIENT = True


def live_config_a(root: Path) -> dict:
    return {'study': 'domain_skill', 'status': 'BUILD_ZERO_BUDGET', 'catalog_root': str(root), 'output': str(root / 'live_not_run'),
            'jobs': [{'job_id': 'a50_D01', 'domain_id': 'D01', 'roster': None}, {'job_id': 'a50_D02', 'domain_id': 'D02', 'roster': None}],
            'order': {'a50_D01': ['no_skill', 'known_domain', 'profile_match'], 'a50_D02': ['profile_match', 'known_domain', 'no_skill']},
            'seeds': list(rt.SEEDS), 'caps': {'max_fit_attempts': 0, 'max_llm_requests': 0, 'max_llm_tokens': 0, 'max_wall_s': 0, 'max_retries': 0},
            'limits': {'max_calls': 16, 'max_tools': 24, 'max_new_evaluations': 2}, 'max_tool_corrections': 2, 'evidence_roundtrip': False,
            'allow_fixture_skills': False, 'stages_authorized': ['material'],
            'note': 'Package A configuration: material stage only, 0 fits, 0 LLM, 0 labels. The live entry refuses it before any client, fit or label read.'}


def live_config_b_draft(root: Path) -> dict:
    return {'study': 'domain_skill', 'status': 'DRAFT_NOT_FROZEN', 'catalog_root': str(root), 'output': None,
            'note': 'Draft for the Planner; the loader refuses DRAFT_NOT_FROZEN. Jobs, whitelists, selection block, model identity and budgets are B decisions.',
            'formation': {'D01': {'whitelist': 'FREEZE_IN_B (subset of source_assets D01.process_trajectories)', 'dev_jobs': 'FREEZE_IN_B', 'max_candidates': 2},
                          'D02': {'whitelist': [], 'trajectory_generation': 'FREEZE_IN_B: no_skill Fast on legal D02 source jobs first', 'dev_jobs': 'FREEZE_IN_B', 'max_candidates': 2},
                          'select_block': 'FREEZE_IN_B (c_a or c_b; never E)'},
            'target_jobs': 'FREEZE_IN_B', 'arms': ['no_skill', 'known_domain', 'profile_match'],
            'seeds': list(rt.SEEDS), 'limits': {'max_calls': 16, 'max_tools': 24, 'max_new_evaluations': 2}, 'caps': 'FREEZE_IN_B from the cost formula in REPORT.md'}


def load_study_config(path) -> dict:
    cfg = context.read_json(path) if not isinstance(path, dict) else copy.deepcopy(path)
    if cfg.get('study') != 'domain_skill' or cfg.get('status') not in CONFIG_STATUSES:
        raise ValueError('not a domain_skill study configuration')
    if cfg['status'] == 'DRAFT_NOT_FROZEN':
        raise PermissionError('LIVE_REFUSED: DRAFT_NOT_FROZEN configuration')
    need = {'catalog_root', 'output', 'jobs', 'order', 'seeds', 'caps', 'limits', 'max_tool_corrections', 'evidence_roundtrip', 'allow_fixture_skills', 'stages_authorized'}
    if need - set(cfg):
        raise ValueError('study config misses %s' % sorted(need - set(cfg)))
    bindings = context.read_json(Path(cfg['catalog_root']) / 'catalog' / 'runtime_bindings.json')['domains']
    ids = [j['job_id'] for j in cfg['jobs']]
    if not ids or len(set(ids)) != len(ids) or set(cfg['order']) != set(ids):
        raise ValueError('jobs must be unique and each must have an arm order')
    cfg['job_datasets'], cfg['job_domains'] = {}, {}
    for j in cfg['jobs']:
        if j['domain_id'] not in bindings:
            raise ValueError('unknown domain id %r' % j['domain_id'])
        ds = bindings[j['domain_id']]['dataset']
        context.resolve_job(ds, j['job_id'], roster=j.get('roster'))            # geometry/file bounds fail here, not later
        cfg['job_datasets'][j['job_id']], cfg['job_domains'][j['job_id']] = ds, j['domain_id']
    for job, arms in cfg['order'].items():
        if not arms or len(set(arms)) != len(arms) or any(a not in STUDY_ARMS for a in arms):
            raise ValueError('arms must be distinct members of %s' % (STUDY_ARMS,))
        if 'random' in arms:
            ps = (cfg.get('random_policy_seeds') or {}).get(job)
            if not isinstance(ps, list) or len(ps) != 2 or len(set(ps)) != 2 or any(type(x) is not int for x in ps):
                raise ValueError('random arm of %s needs two distinct frozen policy seeds' % job)
        if any(a.startswith('cand_') for a in arms) and not Path((cfg.get('candidates') or {}).get(job, '')).is_file():
            raise ValueError('candidate arms of %s need a frozen propose record' % job)
    if len(cfg['seeds']) != 3 or len(set(cfg['seeds'])) != 3:
        raise ValueError('three distinct paired seeds')
    if set(cfg['caps']) != {'max_fit_attempts', 'max_llm_requests', 'max_llm_tokens', 'max_wall_s', 'max_retries'}:
        raise ValueError('caps must name exactly the ledger caps')
    if set(cfg['limits']) != {'max_calls', 'max_tools', 'max_new_evaluations'} or any(s not in STAGES for s in cfg['stages_authorized']):
        raise ValueError('bad limits or stage names')
    return cfg


def configure(cfg: dict) -> None:
    global CFG, JOBS, ORDER, ROSTERS, SEEDS, JOB_DATASETS, JOB_DOMAINS, LIMITS, PLANNED_FITS
    CFG = cfg
    JOBS = tuple(j['job_id'] for j in cfg['jobs'])
    ORDER = {j: tuple(a) for j, a in cfg['order'].items()}
    ROSTERS = {j['job_id']: j['roster'] for j in cfg['jobs'] if j.get('roster')}
    SEEDS = tuple(cfg['seeds'])
    JOB_DATASETS, JOB_DOMAINS = dict(cfg['job_datasets']), dict(cfg['job_domains'])
    LIMITS = dict(cfg['limits'])
    PLANNED_FITS = sum(6 + 3 * LIMITS['max_new_evaluations'] * len(ORDER[j]) for j in JOBS)
    rt.FIT_RETRY = bool(cfg.get('fit_retry', False))


def preflight(cfg: dict) -> None:
    """Refuse a live run before any client, fit or label read unless the configuration is FROZEN with nonzero budgets.
    labels_e is optional: formation-stage runs (Source/Select) stop at C_B."""
    reasons = []
    if cfg['status'] != 'FROZEN':
        reasons.append('status %s is not FROZEN' % cfg['status'])
    zero = [k for k in ('max_fit_attempts', 'max_llm_requests', 'max_llm_tokens', 'max_wall_s') if not cfg['caps'][k] > 0]
    if zero:
        reasons.append('zero budget: %s' % zero)
    missing = [s for s in REQUIRED_LIVE_STAGES if s not in cfg['stages_authorized']]
    if missing:
        reasons.append('stages not authorized: %s' % missing)
    if cfg['allow_fixture_skills']:
        reasons.append('SMOKE_FIXTURE Skills cannot enter a live run')
    if reasons:
        raise PermissionError('LIVE_REFUSED: ' + '; '.join(reasons))


def require_stage(name: str) -> None:
    if name not in CFG.get('stages_authorized', ()):
        raise PermissionError('STAGE_NOT_AUTHORIZED: %s' % name)


def labels_e_allowed() -> bool:
    return 'labels_e' in CFG.get('stages_authorized', ())


def frozen_config(caps) -> dict:
    catalog, bindings, skills = load_catalog(Path(CFG['catalog_root']))
    jobs = {}
    for job in JOBS:
        js = context.resolve_job(JOB_DATASETS[job], job, roster=ROSTERS.get(job))
        jobs[job] = {'domain_id': JOB_DOMAINS[job], 'dataset_runtime_only': JOB_DATASETS[job], 't': js.t, 'train_rows': list(js.train_range),
                     'c_a': list(js.c_a), 'c_b': list(js.c_b), 'e': list(js.e), 'roster': list(js.roster), 'roster_explicit': js.roster_override is not None}
    return {'study': 'domain_skill', 'stage': CFG.get('stage'), 'config_status': CFG['status'], 'jobs': jobs, 'order': {j: list(o) for j, o in ORDER.items()}, 'seeds': list(SEEDS),
            'arms': {'no_skill': 'common public start, no Skill', 'generic': 'common start + hand-frozen Generic CONTROL (not learned, not selected)',
                     'known_domain': 'external domain label -> that domain FROZEN_SELECTED Skill; no model; NO_TREATMENT without a card',
                     'profile_match': 'one frozen router call per job on the identity-free profile; ABSTAIN/parse failure -> common start',
                     'random': 'zero-LLM control: two frozen random complete plans, real fits, commit argmin C_A (ties None, FixedMixup, p1, p2)',
                     'cand_W1': 'selection only: Slow candidate W1 (CANDIDATE_TEST_ONLY)', 'cand_W2': 'selection only: Slow candidate W2 (CANDIDATE_TEST_ONLY)'},
            'route_rules': 'routes of every job are frozen before any fit of that run and never revisited; router failure stops the run (no abstain rewrite); '
                           'a selected Skill still passes the unchanged scope render; selection grants no action permission',
            'catalog_skills': [{'skill_id': s.skill_id, 'domain_id': s.domain_id, 'status': s.status, 'variant': s.variant} for s in skills],
            'generic_control': catalog.get('generic_control'), 'candidates': CFG.get('candidates'), 'random_policy_seeds': CFG.get('random_policy_seeds'),
            'routable_statuses': list(dsk.ROUTABLE) + (['SMOKE_FIXTURE'] if CFG['allow_fixture_skills'] else []),
            'labels': 'C_B then E' if labels_e_allowed() else 'C_B only (formation stage; no E prediction or score)',
            'limits': LIMITS, 'caps': caps, 'planned_fits': PLANNED_FITS, 'http_cap': CFG.get('http_cap', 2 * caps['max_llm_requests']), 'ledger_path': CFG.get('ledger_path'),
            'fit_retry': bool(CFG.get('fit_retry', False)), 'exposure': spec.EXPOSURE,
            'max_tool_corrections': CFG['max_tool_corrections'], 'evidence_roundtrip': CFG['evidence_roundtrip'], 'tool_contracts': CONTRACTS,
            'consumer': spec.CONSUMER, 'knowledge': 'H0 empty common start; at most one experiment_guidance entry per arm',
            'cost_accounting': 'router, fast and slow_domain requests counted by role from raw_responses; fits from the ledger; formation cost is separate',
            'unknown_usage_rule': 'a failed transport attempt leaves cost UNKNOWN and stops later paid calls unless an operator accepts it at resume'}


def candidate_skill(job: str, arm: str):
    prop = context.read_json(CFG['candidates'][job])
    sid = '%s-%s-r1' % (JOB_DOMAINS[job], arm[len('cand_'):])
    hit = [dsk.skill_from_json(s) for s in prop.get('skills', []) if s['skill_id'] == sid]
    if prop.get('status') != 'PROPOSED' or not hit:
        return None
    if hit[0].status != 'CANDIDATE_TEST_ONLY' or hit[0].domain_id != JOB_DOMAINS[job]:
        raise PermissionError('selection candidate %s is not a CANDIDATE_TEST_ONLY card of this domain' % sid)
    return hit[0]


def route_job(root: Path, job: str, adapter, client, *, catalog_root: Path) -> dict:
    p = root / (job + '_routes.json')
    if p.exists():
        raise RuntimeError('routes already frozen for %s; no re-route' % job)
    catalog, bindings, skills_all = load_catalog(catalog_root)
    skills = dsk.routable(skills_all, allow_fixture=CFG['allow_fixture_skills'])
    domain_skills = [s for s in skills if s.domain_id is not None]
    gc = catalog.get('generic_control')
    feats = adapter.overview()['batch_features']
    view, binding = dsk.batch_profile(adapter.ctx.overview(), adapter.ctx.job)
    context.write_json(root / (job + '_profile.json'), view)
    context.write_json(root / (job + '_runtime_binding.json'), binding)
    routes = {}
    for arm in ORDER[job]:
        if arm == 'no_skill':
            routes[arm] = {'route_mode': 'no_skill', 'status': 'NO_SKILL', 'selected_skill_id': None, 'llm_requests': 0}
        elif arm == 'generic':
            routes[arm] = {'route_mode': 'generic', 'status': 'GENERIC_CONTROL' if gc else 'NO_GENERIC_CONTROL', 'selected_skill_id': None,
                           'control_id': gc['control_id'] if gc else None, 'llm_requests': 0}
        elif arm == 'known_domain':
            routes[arm] = dsk.route_known_domain(JOB_DOMAINS[job], domain_skills, feats)
        elif arm == 'profile_match':
            require_stage('llm')
            refs, ref_ids = domain_refs(catalog_root, catalog, bindings, query_binding=binding)
            routes[arm] = dsk.route_profile_match(view, refs, domain_skills, lambda payload: client.call('router', job, payload, dsk.ROUTER_SYSTEM, max_tokens=1024),
                                                  feats, diagnostic_domain_id=JOB_DOMAINS[job])
            routes[arm]['reference_batch_ids'] = ref_ids
        elif arm == 'random':
            routes[arm] = {'route_mode': 'random', 'status': 'ZERO_LLM_CONTROL_NOT_ROUTED', 'selected_skill_id': None, 'policy_seeds': CFG['random_policy_seeds'][job], 'llm_requests': 0}
        else:
            skill = candidate_skill(job, arm)
            routes[arm] = {'route_mode': 'selection_candidate', 'status': 'CANDIDATE_TEST_ONLY' if skill else 'NO_CANDIDATE',
                           'selected_skill_id': skill.skill_id if skill else None, 'scope_at_route': dsk.scope_at(skill, feats), 'llm_requests': 0}
    context.write_json(p, {'job': job, 'domain_id_runtime_only': JOB_DOMAINS[job], 'routes': routes, 'frozen_before_any_fit': True, 'epoch': time.time()})
    if any(r['status'] == 'ROUTER_CALL_FAILED' for r in routes.values()):
        raise RuntimeError('router call failed for %s; recorded, no abstain rewrite and no fallback' % job)
    return {'status': 'COMPLETE', 'routes': routes}


def prepare(root, led, caps, client=None):
    require_stage('material')
    context.write_json(root / 'frozen_config.json', frozen_config(caps))
    if led.s['fit_attempts'] > CFG.get('fit_attempts_at_stage_start', 0):
        raise RuntimeError('routes must be frozen before any fit of this stage')
    before = led.s['fit_attempts']
    commons, pools = {}, {}
    for job in JOBS:
        shared = root / (job + '_common')
        shared.mkdir()
        try:
            adapter = rt.Adapter(shared, job, led, REPO, roster=ROSTERS.get(job), seeds=SEEDS, dataset=JOB_DATASETS[job])
        except RuntimeError as exc:
            if 'ELIGIBILITY_FAIL' not in str(exc):
                raise
            context.write_json(root / (job + '_eligibility.json'), {'job': job, 'status': 'ELIGIBILITY_FAILED', 'reason': str(exc), 'note': 'no population, column or time change'})
            pools[job] = None
            continue
        commons[job] = adapter
        pools[job] = route_job(root, job, adapter, client, catalog_root=Path(CFG['catalog_root']))
    if led.s['fit_attempts'] != before:
        raise RuntimeError('a fit happened during routing')
    context.write_json(root / FROZEN_MARKER, {'epoch': time.time(), 'jobs': {j: ('COMPLETE' if p else 'ELIGIBILITY_FAILED') for j, p in pools.items()},
                                              'fit_attempts_at_freeze': led.s['fit_attempts']})
    return commons, pools


def load_pools(root):
    return {job: ({'status': 'COMPLETE', 'routes': context.read_json(root / (job + '_routes.json'))['routes']} if (root / (job + '_routes.json')).exists() else None)
            for job in JOBS}


def arm_knowledge(job: str, arm: str, pool: dict):
    """(route, Knowledge|None, treatment record). None = NO_TREATMENT: never a silent no-skill duplicate."""
    catalog, bindings, skills_all = load_catalog(Path(CFG['catalog_root']))
    route = pool['routes'][arm]
    if arm == 'no_skill':
        return route, br.Knowledge(), None
    if arm == 'generic':
        gc = catalog.get('generic_control')
        if not gc:
            return route, None, None
        return route, generic_knowledge(gc), gc
    if arm in ('known_domain', 'profile_match'):
        skills = dsk.routable(skills_all, allow_fixture=CFG['allow_fixture_skills'])
        runnable = route['status'] == 'KNOWN_DOMAIN_SELECTED' if arm == 'known_domain' else route['status'] != 'NO_CANDIDATE_SKILLS'
        if not runnable:
            return route, None, None
        skill = dsk.routed_skill(route, skills)
        return route, dsk.skill_knowledge(skill), (skill.to_json() if skill else None)
    if arm.startswith('cand_'):
        skill = candidate_skill(job, arm)
        return (route, None, None) if skill is None else (route, dsk.skill_knowledge(skill), skill.to_json())
    raise ValueError(arm)


def generic_knowledge(gc: dict) -> br.Knowledge:
    """Exactly the historical frozen Generic control of batch_research_source_process (version 1, one experiment_guidance entry)."""
    return br.Knowledge(int(gc['knowledge_version']), (br.Guidance(gc['hook'], gc['body'], copy.deepcopy(gc['observable_applicability']), tuple(gc['evidence_refs'])),))


def run_arm(arm, path, job, led, client, shared, pool, *, seeds, roster, max_tool_corrections):
    if arm == 'random':
        require_stage('fit')
        from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp   # lazy: the runner imports this module
        result = rtp.random_branch(path, job, led, shared, seeds=seeds, roster=roster, dataset=JOB_DATASETS[job], policy_seeds=tuple(CFG['random_policy_seeds'][job]))
        context.write_json(path / 'route_used.json', {'arm': arm, 'route': pool['routes'][arm], 'treatment': None, 'knowledge_entries': 0, 'llm_requests': 0})
        return result
    route, h, treatment = arm_knowledge(job, arm, pool)
    if h is None:
        path.mkdir(parents=True, exist_ok=False)
        (path / 'trace.jsonl').write_text('', encoding='utf-8')          # the runner's report reads every branch trace
        context.write_json(path / 'treatment.json', {'status': 'NO_TREATMENT', 'route': route, 'note': 'arm not run; no common-start substitute and no budget transfer'})
        result = br.RunResult('INCOMPLETE', job, 0, None, None, 0, 0, 0, [], 'NO_TREATMENT', route['status'])
        context.write_json(path / 'branch_result.json', asdict(result))
        return result
    require_stage('fit')
    require_stage('llm')
    result = base.branch(path, job, h, led, client, shared, dataset=JOB_DATASETS[job], evidence_roundtrip=bool(CFG['evidence_roundtrip']),
                         max_tool_corrections=max_tool_corrections, tool_contracts=CONTRACTS, seeds=seeds, roster=roster, limits=LIMITS)
    context.write_json(path / 'route_used.json', {'arm': arm, 'route': route, 'treatment': treatment, 'knowledge_version': h.version, 'knowledge_entries': len(h.entries)})
    return result


def resume_arm(arm, path, job, led, client, pool, *, seeds, roster, max_tool_corrections):
    if arm == 'random':
        raise RuntimeError('control branch cannot be resumed')
    route, h, treatment = arm_knowledge(job, arm, pool)
    if h is None:
        raise RuntimeError('a NO_TREATMENT arm has nothing to resume')
    require_stage('fit')
    require_stage('llm')
    return base.resume_branch(path, job, h, led, client, max_tool_corrections=max_tool_corrections, tool_contracts=CONTRACTS,
                              seeds=seeds, roster=roster, evidence_roundtrip=bool(CFG['evidence_roundtrip']), limits=LIMITS, dataset=JOB_DATASETS[job])


def before_labels(root, branches):
    require_stage('labels_c_b')


def cost_by_role(root: Path) -> dict:
    out = {}
    for req in sorted((Path(root) / 'raw_responses').glob('*_request.json')):
        q = context.read_json(req)
        row = out.setdefault(q.get('role'), {'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'responses_missing': 0})
        row['requests'] += 1
        resp = req.with_name(req.name.replace('_request.json', '_response.json'))
        if not resp.exists():
            row['responses_missing'] += 1
            continue
        u = context.read_json(resp).get('usage') or {}
        row['prompt_tokens'] += int(u.get('prompt_tokens') or 0)
        row['completion_tokens'] += int(u.get('completion_tokens') or 0)
    return out


def finish_report(root, r, jobs, order, seeds):
    r['stage'] = CFG.get('stage')
    r['routes'] = {j: context.read_json(root / (j + '_routes.json'))['routes'] for j in jobs if (root / (j + '_routes.json')).exists()}
    r['cost_by_role'] = cost_by_role(root)
    if CFG.get('ledger_path') and Path(CFG['ledger_path']).exists():
        r['package_ledger'] = {k: v for k, v in context.read_json(CFG['ledger_path']).items() if k != 'events'}
    lines = ['# DOMAIN-SKILL live run (%s)' % CFG.get('stage'), '', '状态：' + r['status'], '', '|分支|状态|路由|选中 Skill|首请求加载|commit|三seed E均值|', '|---|---|---|---|---|---|---:|']
    for name, b in sorted(r['branches'].items()):
        used = root / name / 'route_used.json'
        rec = context.read_json(used) if used.exists() else None
        b['route_used'] = rec
        trace = root / name / 'trace.jsonl'
        first = next((json.loads(x) for x in trace.read_text(encoding='utf-8').splitlines() if '"fast_request"' in x), None) if trace.exists() else None
        b['first_request_loaded_hooks'] = [x['hook'] for x in first['guidance']['loaded']] if first else None
        lines.append('|%s|%s|%s|%s|%s|%s|%s|' % (name, b['controller']['status'], (rec or {}).get('route', {}).get('status'), (rec or {}).get('route', {}).get('selected_skill_id'),
                                                b['first_request_loaded_hooks'], b['controller']['committed_plan_id'], b['paired_mean_e']))
    for p in sorted(root.glob('*/treatment.json')):
        lines.append('|%s|NO_TREATMENT|%s|||||' % (p.parent.name, context.read_json(p)['route']['status']))
    lines += ['', '成本（按角色）：' + json.dumps(r['cost_by_role'], ensure_ascii=False), '包账本：' + json.dumps(r.get('package_ledger'), ensure_ascii=False),
              '', 'known_domain 是已知域路由参照，不是模型推断。形成/选优成本单列。']
    context.write_json(root / 'result.json', r)
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ============================================================================= offline formation entry
def census_domain(domain_id: str, whitelist, *, repo: Path = REPO, workload_end_row: int, bindings=None, tool_description_note: str | None = None) -> dict:
    """Deterministic census of explicitly whitelisted branches of ONE domain (no E). Every branch must carry the domain's
    own dataset in its fitted cells; relabeling another source is refused. Empty whitelist = recorded process gap."""
    ds = (bindings or DOMAIN_BINDINGS)[domain_id]
    whitelist = [tuple(x) for x in whitelist]
    if not whitelist:
        return {'census_status': 'NO_PROCESS_TRAJECTORIES', 'domain_id': domain_id, 'legal_evidence_refs': [], 'branches': [],
                'gap': 'no whitelisted batch Fast trajectory for this domain; effect-only evidence cannot stand in for process evidence'}
    source_binding = []
    for package, name, job in whitelist:
        cells = [context.read_json(c) for c in sorted((Path(repo) / '_scratch' / package / name / job / 'cells').glob('*.json'))]
        if not cells:
            raise ValueError('whitelisted branch without fitted cells: %s' % name)
        if any(c.get('dataset', 'electricity') != ds for c in cells):
            raise PermissionError('branch %s belongs to another source; relabeling refused' % name)
        populations = {tuple(c['roster']) for c in cells if c.get('roster') is not None}
        if len(populations) > 1:
            raise PermissionError('branch %s mixes populations' % name)
        js = context.resolve_job(ds, job)
        if max(js.e) + spec.H > workload_end_row:
            raise PermissionError('branch %s runs past the formation time boundary' % name)
        source_binding.append({'branch': name, 't': js.t, 'workload_end_row_exclusive': max(js.e) + spec.H,
                               'roster_explicit': any(c.get('roster_explicit') for c in cells), 'populations': max(1, len(populations))})
    ev = sp.build_source_evidence(repo, source_branches=tuple(whitelist), dataset=ds, last_source_job=whitelist[-1][2], workload_end_row=workload_end_row)
    ev = {**ev, 'census_status': 'CENSUS_COMPLETE', 'domain_id': domain_id, 'source_binding': source_binding,
          'purpose': 'Offline Slow input: can these source processes of one neutral domain be consolidated into at most two reusable domain Workflows?'}
    if tool_description_note:
        ev['historical_tool_description'] = {**ev.get('historical_tool_description', {}), 'note': tool_description_note}
    if any(n in json.dumps(ev, ensure_ascii=False).lower() for n in spec.DATASETS):
        raise PermissionError('census names a data source')
    return ev


def block_scorer(ds: str, block: str, seeds, branch_dir, *, led=None, rosters=None):
    """score(option_id, job, record) for dsk.select: reads the committed plan's `block` loss of that branch. C_B is opened
    here (base.stage) only when the branch has not had it opened yet and a ledger is given; E is never touched."""
    if block not in ('c_a', 'c_b'):
        raise ValueError('selection block must be c_a or c_b')
    rosters = rosters or {}

    def score(option_id, job, rec):
        path = Path(branch_dir(option_id, job))
        jdir = path / context.resolve_job(ds, job).job_id
        if block == 'c_b' and not (jdir / 'c_b_scores.json').exists():
            if led is None:
                return {'loss_by_seed': None, 'status': 'BLOCK_INCOMPLETE'}
            base.stage(path, job, 'c_b', led, roster=rosters.get(job), dataset=ds)
        by_seed = {}
        for c in (jdir / 'cells').glob('*.json'):
            cell = context.read_json(c)
            if cell.get('status') == 'OK' and cell['material_id'] == rec['committed_plan_id'] and block in cell.get('scores', {}):
                by_seed[cell['model_seed']] = cell['scores'][block]['normalized_mse_macro']
        if not set(by_seed) >= set(seeds):
            return {'loss_by_seed': None, 'status': 'BLOCK_INCOMPLETE'}
        return {'loss_by_seed': [by_seed[s] for s in seeds]}
    return score


def selection_runner(root: Path, led, client, *, domain_id: str, shared_by_job: dict, block: str, seeds, limits, max_tool_corrections, evidence_roundtrip, rosters=None):
    """(run, score) for dsk.select. run: one complete Fast job per (option, dev job) under the same budget, commit only.
    score: the delayed block of the committed plan, called by dsk.select after EVERY option has committed; never E."""
    ds = DOMAIN_BINDINGS[domain_id]
    rosters = rosters or {}
    branch_dir = lambda option_id, job: Path(root) / 'selection' / domain_id / ('%s__%s' % (job, option_id))

    def run(option_id, knowledge, job):
        result = base.branch(branch_dir(option_id, job), job, knowledge, led, client, shared_by_job[job], dataset=ds, evidence_roundtrip=evidence_roundtrip,
                             max_tool_corrections=max_tool_corrections, tool_contracts=CONTRACTS, seeds=seeds, roster=rosters.get(job), limits=limits)
        return {'status': result.status, 'committed_plan_id': result.committed_plan_id, 'failure_kind': result.failure_kind, 'synthetic': False,
                'cost': {'fast_calls': result.calls, 'tool_calls': result.tool_calls, 'new_evaluations': result.new_evaluations}}
    return run, block_scorer(ds, block, seeds, branch_dir, led=led, rosters=rosters)


def formation(stage: str, cfg_path: Path) -> None:
    """census (0 cost) -> propose (Slow, paid) -> select (Fast + fits, paid) -> freeze (0 cost). Each stage writes once.
    Paid stages charge ONE package ledger (cfg['ledger_path']) under the caps frozen in the configuration; no entry gets its own budget."""
    cfg = context.read_json(cfg_path)
    if cfg.get('status') != 'FROZEN':
        raise PermissionError('FORMATION_REFUSED: configuration status %r is not FROZEN' % cfg.get('status'))
    d = cfg['domain_id']
    out = Path(cfg['output']) / d
    if stage == 'census':
        write_once(out / 'census.json', census_domain(d, cfg['whitelist'], workload_end_row=cfg['workload_end_row']))
        return
    if stage in ('propose', 'select'):
        zero = [k for k in ('max_fit_attempts', 'max_llm_requests', 'max_llm_tokens', 'max_wall_s') if not cfg['caps'][k] > 0]
        if zero:
            raise PermissionError('FORMATION_REFUSED_ZERO_BUDGET: %s' % zero)
        if not cfg.get('ledger_path'):
            raise PermissionError('FORMATION_REFUSED: paid formation stages need the package ledger_path')
        led = rt.RuntimeLedger(cfg['ledger_path'], **cfg['caps'])
        client = rt.MeteredClient(led, Path(cfg['output']) / 'raw_responses', http_cap=int(cfg.get('http_cap', 2 * cfg['caps']['max_llm_requests'])))
    if stage == 'propose':
        census = context.read_json(out / 'census.json')
        if census['census_status'] != 'CENSUS_COMPLETE':
            write_once(out / 'propose.json', {'status': 'NO_PROPOSAL_INPUT', 'census_status': census['census_status']})
            return
        res = dsk.propose(census, domain_id=d, call=lambda p: client.call('slow_domain', d, p, dsk.SLOW_SYSTEM, max_tokens=8192))
        write_once(out / 'propose.json', {**{k: v for k, v in res.items() if k != 'skills'}, 'skills': [s.to_json() for s in res['skills']]})
        return
    if stage == 'select':
        prop = context.read_json(out / 'propose.json')
        options = {dsk.NO_SKILL: None, **{s['skill_id']: dsk.skill_from_json(s) for s in prop.get('skills', [])}}
        seeds = tuple(cfg['seeds'])
        shared = {}
        for job in cfg['dev_jobs']:
            sdir = out / 'selection_common' / job
            sdir.mkdir(parents=True, exist_ok=False)
            rt.Adapter(sdir, job, led, REPO, roster=(cfg.get('rosters') or {}).get(job), seeds=seeds, dataset=DOMAIN_BINDINGS[d]).baselines()
            shared[job] = sdir
        run, score = selection_runner(out, led, client, domain_id=d, shared_by_job=shared, block=cfg['select_block'], seeds=seeds, limits=cfg['limits'],
                                      max_tool_corrections=cfg['max_tool_corrections'], evidence_roundtrip=cfg['evidence_roundtrip'], rosters=cfg.get('rosters'))
        write_once(out / 'selection.json', dsk.select(options, cfg['dev_jobs'], run, block=cfg['select_block'], score=score))
        return
    if stage == 'freeze':
        prop, sel = context.read_json(out / 'propose.json'), context.read_json(out / 'selection.json')
        options = {dsk.NO_SKILL: None, **{s['skill_id']: dsk.skill_from_json(s) for s in prop.get('skills', [])}}
        write_once(out / 'frozen_skill.json', dsk.freeze(d, sel, options, evidence_scope={'census': str(out / 'census.json'), 'dev_jobs': sel['dev_jobs'], 'block': sel['block']}))
        return
    raise ValueError(stage)


# ============================================================================= EVOLVE-PILOT (docs/DEV_DOMAIN_SKILL_V1_EVOLVE_PILOT_TASK_2026-09-17.md)
PILOT_ROOT = REPO / '_scratch' / 'dev_domain_skill_v1_evolve_pilot'
PILOT_TASK = 'docs/DEV_DOMAIN_SKILL_V1_EVOLVE_PILOT_TASK_2026-09-17.md'
PILOT_SEEDS = (20260922, 20260923, 20260924)
PILOT_LIMITS = {'max_calls': 16, 'max_tools': 24, 'max_new_evaluations': 2}
PILOT_TOTAL = {'max_fit_attempts': 194, 'max_llm_requests': 324, 'max_llm_tokens': 6_000_000, 'max_wall_s': 28800, 'max_retries': 2}
PILOT_HTTP_CAP = 648
PILOT_ALLOC = {'source': {'fits': 24, 'requests': 32, 'tokens': 800_000},
               'formation': {'fits': 48, 'requests': 100, 'tokens': 2_000_000},
               'target': {'fits': 120, 'requests': 192, 'tokens': 3_200_000}}
PILOT_T = {'D01': {'a55': 14448, 'a65': 17088, 'a75': 19728, 'a85': 22344, 'a95': 24984},
           'D02': {'a55': 9648, 'a65': 11400, 'a75': 13152, 'a85': 14904, 'a95': 16656}}
PILOT_SOURCE_END = {'D01': 17472, 'D02': 11784}
D01_HISTORY = (('dev_batch_research_workflow_v1', 'a55_H0', 'a55'), ('dev_batch_research_workflow_v1', 'a65_parent', 'a65'))
D02_SOURCE_JOBS = ('a55_D02', 'a65_D02')
REFERENCE_PROFILE_JOBS = ('a55', 'a65')
SELECT_JOBS = {'D01': 'a75_D01', 'D02': 'a75_D02'}
TARGET_JOBS = ('a85_D01', 'a85_D02', 'a95_D01', 'a95_D02')
TARGET_ORDER = {'a85_D01': ('no_skill', 'generic', 'known_domain', 'random'), 'a85_D02': ('generic', 'known_domain', 'no_skill', 'random'),
                'a95_D01': ('known_domain', 'no_skill', 'generic', 'random'), 'a95_D02': ('no_skill', 'known_domain', 'generic', 'random')}
TARGET_RANDOM_SEEDS = {'a85_D01': [2026091701, 2026091702], 'a85_D02': [2026091703, 2026091704], 'a95_D01': [2026091705, 2026091706], 'a95_D02': [2026091707, 2026091708]}
MODEL = {'requested': 'cpa-grok-4.6', 'returned_required': 'grok-4.6-build', 'temperature': 0, 'base_url': 'http://127.0.0.1:8318/v1'}
D02_SOURCE_NOTE = ('D02 source branches were run in this package with the current public tool semantics (public_semantics_now), evidence_roundtrip=true '
                   'and up to 2 tool-input corrections; unlike D01 history, they saw those stated facts.')


def _domain_of(job: str) -> str:
    return job.rsplit('_', 1)[1]


def pilot_paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'configs': root / 'stage_configs',
            'source': root / 'source', 'formation': root / 'formation', 'pre_catalog': root / 'formation' / 'pre', 'select': root / 'formation' / 'select',
            'catalog': root / 'formation', 'target': root / 'target'}


def generic_control_record() -> dict:
    return {'control_id': 'GENERIC-CONTROL', 'status': 'CONTROL', 'hook': 'experiment_guidance', 'body': sp.GENERIC_TEXT, 'observable_applicability': {'const': True},
            'evidence_refs': ['generic:frozen'], 'knowledge_version': 1,
            'source': 'evaluation.main_protocol_p4.batch_research_source_process.GENERIC_TEXT; hand-frozen control, not learned and not selected'}


def pilot_preflight(root: Path) -> dict:
    """0 cost: frozen geometry vs the task table, T eligibility (T rows only), D01 history binding, model constants, Generic
    control and the pre-formation catalog. Writes pilot_config.json / preflight.json once."""
    import inspect
    from methods.ttha.batch_base import data
    P = pilot_paths(root)
    checks = {}
    for d, ds in DOMAIN_BINDINGS.items():
        prev_end = None
        for a, t in PILOT_T[d].items():
            js = spec.job_spec(ds, int(a[1:]) / 100.0)
            if js.t != t:
                raise RuntimeError('geometry drift %s %s: %d != %d' % (d, a, js.t, t))
            sl = data.load_slice(ds, js.t - spec.TRAIN_SPAN, js.t)
            sc = data.compute_scaler(sl, js.t)
            data.build_parents(sl, js.t, sc)
            if sl.roster != spec.DATASETS[ds]['roster']:
                raise RuntimeError('roster drift %s' % d)
            if prev_end is not None and js.train_range[0] < prev_end:
                raise RuntimeError('stage time overlap at %s %s' % (d, a))
            checks['%s_%s' % (d, a)] = {'t': js.t, 'train_rows': list(js.train_range), 'c_a': list(js.c_a), 'c_b': list(js.c_b), 'e': list(js.e),
                                        'workload_end_row_exclusive': max(js.e) + spec.H, 'zero_scale_entities': sc.floor_hits, 'eligible': sc.floor_hits == 0}
            if sc.floor_hits:
                checks['%s_%s' % (d, a)]['status'] = 'ELIGIBILITY_FAILED'
            prev_end = max(js.e) + spec.H
        if checks['%s_a65' % d]['workload_end_row_exclusive'] != PILOT_SOURCE_END[d]:
            raise RuntimeError('source boundary drift %s' % d)
    hist = []
    for package, name, job in D01_HISTORY:
        b = REPO / '_scratch' / package / name
        res, kn = context.read_json(b / 'branch_result.json'), context.read_json(b / 'knowledge.json')
        cells = [context.read_json(p) for p in sorted((b / job / 'cells').glob('*.json'))]
        js = context.resolve_job('electricity', job)
        ok = (res['status'] == 'COMPLETE' and kn == {'version': 0, 'entries': []} and cells and all(c.get('dataset') == 'electricity' and c.get('roster') is None
              and list(c['rows_read']) == list(js.evaluate_rows) for c in cells) and {c['model_seed'] for c in cells} == set(PILOT_SEEDS)
              and (b / job / 'commit.json').exists() and (b / job / 'c_b_scores.json').exists())
        if not ok:
            raise RuntimeError('D01 history branch %s does not match the whitelist contract' % name)
        hist.append({'branch': name, 'job': job, 't': js.t, 'committed_plan_id': res['committed_plan_id'], 'fast_calls': res['calls'], 'new_evaluations': res['new_evaluations'],
                     'cells': len(cells), 'seeds': sorted({c['model_seed'] for c in cells}), 'note': 'historical seeds/budget/controls are not rewritten; not a same-configuration repeat of this package'})
    src = inspect.getsource(rt.MeteredClient)
    if not all(x in src for x in ("model='cpa-grok-4.6'", "'grok-4.6-build'", 'temperature=0', "base_url='http://127.0.0.1:8318/v1'")):
        raise RuntimeError('metered client no longer carries the frozen model identity')
    gc = generic_control_record()
    if generic_knowledge(gc) != sp.br.Knowledge(1, (sp.br.Guidance('experiment_guidance', sp.GENERIC_TEXT, {'const': True}, ('generic:frozen',)),)):
        raise RuntimeError('generic control differs from the historical frozen Generic knowledge')
    cat_dir = P['pre_catalog'] / 'catalog'
    write_once(cat_dir / 'domain_catalog.json', {'catalog_version': 1, 'status': 'PRE_FORMATION', 'note': 'no domain Skill yet; Generic CONTROL frozen before the first Source call',
                                                 'domains': {d: {'reference_batch_ids': [], 'formation_status': 'NOT_FORMED'} for d in DOMAIN_BINDINGS}, 'skills': [],
                                                 'generic_control': gc, 'generic_skill_id': None})
    write_once(cat_dir / 'runtime_bindings.json', {'note': 'RUNTIME ONLY', 'domains': {d: {'dataset': ds, 'reference_batches': {}} for d, ds in DOMAIN_BINDINGS.items()}})
    cfg = {'package': 'DEV-DOMAIN-SKILL-V1-EVOLVE-PILOT', 'task': PILOT_TASK, 'domains_runtime_only': DOMAIN_BINDINGS, 'seeds': list(PILOT_SEEDS), 'limits': PILOT_LIMITS,
           'max_tool_corrections': 2, 'evidence_roundtrip': True, 'model': MODEL, 'total_caps': PILOT_TOTAL, 'http_cap': PILOT_HTTP_CAP, 'stage_allocations': PILOT_ALLOC,
           'source': {'D01_history_whitelist': [list(x) for x in D01_HISTORY], 'D02_run_jobs': list(D02_SOURCE_JOBS), 'workload_end_row_exclusive': PILOT_SOURCE_END},
           'select': {'jobs': SELECT_JOBS, 'block': 'c_b', 'option_order': ['NO_SKILL', 'W1', 'W2'], 'rule': 'argmin three-seed C_B of each option\'s own commit; ties NO_SKILL, W1, W2'},
           'target': {'jobs': list(TARGET_JOBS), 'order': {j: list(o) for j, o in TARGET_ORDER.items()}, 'random_policy_seeds': TARGET_RANDOM_SEEDS},
           'reference_profile_jobs': list(REFERENCE_PROFILE_JOBS), 'generic_control': gc, 'exposure': spec.EXPOSURE, 'geometry': checks, 'D01_history': hist,
           'fit_retry': 'same-configuration re-launch of a failed cell, at most 2 for the whole package (ledger max_retries)'}
    write_once(root / 'pilot_config.json', cfg)
    return cfg


def stage_caps(root: Path, stage: str, *, total=None, alloc=None, http_cap=None, stages_path=None, ledger_path=None) -> dict:
    """Cumulative caps for a stage, frozen at its first start: consumption so far + this stage's allocation (never an earlier
    stage's unused share), bounded by the package totals. A restart reads the frozen record; nothing is reset.
    total / alloc / http_cap / stages_path / ledger_path: another package's frozen budget table; defaults = this pilot."""
    P = pilot_paths(root)
    PILOT_TOTAL_, PILOT_ALLOC_, PILOT_HTTP_CAP_ = total or PILOT_TOTAL, alloc or PILOT_ALLOC, http_cap or PILOT_HTTP_CAP
    stages_path, ledger_path = Path(stages_path or P['stages']), Path(ledger_path or P['ledger'])
    rec = context.read_json(stages_path) if stages_path.exists() else {}
    if stage in rec:
        return rec[stage]
    led = rt.RuntimeLedger(ledger_path, **PILOT_TOTAL_)
    s, alloc = led.s, PILOT_ALLOC_[stage]
    snap = {k: s.get(k, 0) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts', 'llm_tokens_in',
                                     'llm_tokens_out', 'llm_tokens_unknown', 'fit_wall_seconds')}
    snap['elapsed_s'] = time.time() - s['started_epoch']
    retries_left = PILOT_TOTAL_['max_retries'] - s['retries_used']
    caps = {'max_fit_attempts': min(PILOT_TOTAL_['max_fit_attempts'], s['fit_attempts'] + alloc['fits'] + retries_left),
            'max_llm_requests': min(PILOT_TOTAL_['max_llm_requests'], s['llm_requests'] + alloc['requests']),
            'max_llm_tokens': min(PILOT_TOTAL_['max_llm_tokens'], s['llm_tokens_in'] + s['llm_tokens_out'] + alloc['tokens']),
            'max_wall_s': PILOT_TOTAL_['max_wall_s'], 'max_retries': PILOT_TOTAL_['max_retries']}
    rec[stage] = {'epoch_start': time.time(), 'allocation': alloc, 'snapshot_at_start': snap, 'caps': caps,
                  'http_cap': min(PILOT_HTTP_CAP_, s['llm_http_attempts'] + 2 * alloc['requests'])}
    context.write_json(stages_path, rec)
    return rec[stage]


def pilot_stage_config(root: Path, stage: str, *, jobs, order, catalog_root: Path, labels_e: bool, candidates=None, random_seeds=None) -> Path:
    P = pilot_paths(root)
    path = P['configs'] / ('%s.json' % stage)
    if path.exists():
        return path
    sc = stage_caps(root, 'formation' if stage == 'select' else stage)
    cfg = {'study': 'domain_skill', 'status': 'FROZEN', 'stage': stage, 'catalog_root': str(catalog_root), 'output': str(P[stage]),
           'jobs': [{'job_id': j, 'domain_id': _domain_of(j), 'roster': None} for j in jobs], 'order': {j: list(order[j]) for j in jobs},
           'seeds': list(PILOT_SEEDS), 'caps': sc['caps'], 'http_cap': sc['http_cap'], 'ledger_path': str(P['ledger']),
           'fit_attempts_at_stage_start': context.read_json(P['ledger'])['fit_attempts'], 'limits': PILOT_LIMITS, 'max_tool_corrections': 2,
           'evidence_roundtrip': True, 'allow_fixture_skills': False, 'fit_retry': True,
           'stages_authorized': list(REQUIRED_LIVE_STAGES) + (['labels_e'] if labels_e else []),
           'candidates': candidates or {}, 'random_policy_seeds': random_seeds or {}}
    load_study_config(cfg)                     # validate before freezing
    write_once(path, cfg)
    return path


def run_stage(root: Path, stage: str, cfg_path: Path) -> dict:
    import subprocess
    sroot = pilot_paths(root)[stage]
    if not sroot.exists():
        print('PILOT_STAGE_START', stage, flush=True)
        subprocess.run([sys.executable, '-B', '-m', 'evaluation.main_protocol_p4.run_batch_research_roundtrip', '--run', '--study', 'domain_skill',
                        '--study-config', str(cfg_path), '--output', str(sroot)], cwd=REPO)
    return stage_status(sroot)


def stage_status(sroot: Path) -> dict:
    fin = sroot / 'execution_finished.json'
    if not fin.exists():
        return {'status': 'NOT_FINISHED'}
    f = context.read_json(fin)
    if (sroot / 'labels_withheld.json').exists():
        return {'status': 'LABELS_WITHHELD', 'failures': f['failures'], 'withheld': context.read_json(sroot / 'labels_withheld.json')}
    opened = (sroot / 'labels_c_b_only.json').exists() or (sroot / 'all_e_predictions_frozen.json').exists()
    if not opened or any(x.get('stage') in ('external', 'execution') for x in f['failures']):
        return {'status': 'LABELS_NOT_OPENED_OR_EXTERNAL_FAILURE', 'failures': f['failures']}
    return {'status': 'FINISHED', 'failures': f['failures'], 'branches': f['branches']}


def _branch_complete_with_c_b(bdir: Path, job: str) -> bool:
    r = bdir / 'branch_result.json'
    return r.exists() and context.read_json(r)['status'] == 'COMPLETE' and (bdir / job / 'commit.json').exists() and (bdir / job / 'c_b_scores.json').exists()


def pilot_formation_calls(root: Path) -> None:
    """Census (0 cost) and one Slow formation per domain on the package ledger under the formation-stage caps."""
    P = pilot_paths(root)
    for d in DOMAIN_BINDINGS:
        out = P['formation'] / d
        if not (out / 'census.json').exists():
            if d == 'D01':
                census = census_domain(d, D01_HISTORY, workload_end_row=PILOT_SOURCE_END[d])
            else:
                wl = [('dev_domain_skill_v1_evolve_pilot/source', '%s_no_skill' % j, j) for j in D02_SOURCE_JOBS]
                missing = [n for _, n, j in wl if not _branch_complete_with_c_b(P['source'] / n, j)]
                census = ({'census_status': 'SOURCE_INCOMPLETE', 'domain_id': d, 'incomplete_branches': missing, 'legal_evidence_refs': [],
                           'note': 'both D02 source trajectories must be COMPLETE with post-commit C_B; no substitute trajectory'} if missing else
                          census_domain(d, wl, workload_end_row=PILOT_SOURCE_END[d], tool_description_note=D02_SOURCE_NOTE))
            write_once(out / 'census.json', census)
    sc = stage_caps(root, 'formation')
    led = None
    for d in DOMAIN_BINDINGS:
        out = P['formation'] / d
        if (out / 'propose.json').exists():
            continue
        census = context.read_json(out / 'census.json')
        if census['census_status'] != 'CENSUS_COMPLETE':
            write_once(out / 'propose.json', {'status': 'NO_PROPOSAL_INPUT', 'census_status': census['census_status'], 'skills': []})
            continue
        if led is None:
            led = rt.RuntimeLedger(P['ledger'], **sc['caps'])
            if rt.unknown_usage_blocks(led):
                raise RuntimeError('package ledger holds unknown usage; paid formation refused without an operator decision')
            client = rt.MeteredClient(led, P['formation'] / 'raw_responses', http_cap=sc['http_cap'])
        res = dsk.propose(census, domain_id=d, call=lambda p, d=d: client.call('slow_domain', d, p, dsk.SLOW_SYSTEM, max_tokens=8192))
        write_once(out / 'propose.json', {**{k: v for k, v in res.items() if k != 'skills'}, 'skills': [s.to_json() for s in res['skills']],
                                          'census_legal_ref_count': len(census['legal_evidence_refs'])})
        if client.fatal or rt.unknown_usage_blocks(led):
            raise RuntimeError('backend fatal or unknown usage during formation; stop')


def select_options(root: Path, d: str) -> tuple:
    prop = context.read_json(pilot_paths(root)['formation'] / d / 'propose.json')
    skills = [dsk.skill_from_json(s) for s in prop.get('skills', [])] if prop.get('status') == 'PROPOSED' else []
    return prop, skills


def pilot_select_stage(root: Path) -> dict:
    P = pilot_paths(root)
    jobs, order, cands = [], {}, {}
    for d in DOMAIN_BINDINGS:
        prop, skills = select_options(root, d)
        if not skills:
            continue
        job = SELECT_JOBS[d]
        jobs.append(job)
        order[job] = ['no_skill'] + ['cand_' + s.skill_id.split('-')[1] for s in skills]
        cands[job] = str(P['formation'] / d / 'propose.json')
    if not jobs:
        write_once(P['formation'] / 'select_skipped.json', {'status': 'NO_CANDIDATES_IN_ANY_DOMAIN', 'note': 'no empty selection course is started'})
        return {'status': 'SKIPPED'}
    cfgp = pilot_stage_config(root, 'select', jobs=jobs, order=order, catalog_root=P['pre_catalog'], labels_e=False, candidates=cands)
    return run_stage(root, 'select', cfgp)


def pilot_freeze(root: Path) -> None:
    P = pilot_paths(root)
    for d, ds in DOMAIN_BINDINGS.items():
        out = P['formation'] / d
        if (out / 'frozen_skill.json').exists():
            continue
        prop, skills = select_options(root, d)
        if not skills:
            write_once(out / 'frozen_skill.json', {'domain_id': d, 'status': 'NO_CANDIDATE', 'skill': None, 'propose_status': prop.get('status'),
                                                   'census_status': prop.get('census_status'), 'note': 'KEEP / no proposal input / formation failure: no selection course, no card'})
            continue
        job = SELECT_JOBS[d]
        arm_of = {dsk.NO_SKILL: 'no_skill', **{s.skill_id: 'cand_' + s.skill_id.split('-')[1] for s in skills}}
        options = {dsk.NO_SKILL: None, **{s.skill_id: s for s in skills}}
        bdir = lambda oid, j: P['select'] / ('%s_%s' % (j, arm_of[oid]))

        def read_run(oid, knowledge, j):
            b = bdir(oid, j)
            if not (b / 'branch_result.json').exists():
                return {'status': 'NOT_RUN', 'synthetic': False}
            if context.read_json(b / 'knowledge.json') != br.json_copy(asdict(knowledge)):
                raise RuntimeError('selection branch %s ran other knowledge than the option' % b.name)
            res = context.read_json(b / 'branch_result.json')
            return {'status': res['status'], 'committed_plan_id': res['committed_plan_id'], 'failure_kind': res['failure_kind'], 'synthetic': False, 'branch': b.name,
                    'cost': {'fast_calls': res['calls'], 'tool_calls': res['tool_calls'], 'new_evaluations': res['new_evaluations']}}
        sel = dsk.select(options, [job], read_run, block='c_b', score=block_scorer(ds, 'c_b', PILOT_SEEDS, bdir))
        write_once(out / 'selection.json', sel)
        write_once(out / 'frozen_skill.json', dsk.freeze(d, sel, options, evidence_scope={'census': 'formation/%s/census.json' % d, 'select_job': job, 'block': 'c_b',
                                                                                          'source': 'D01 history a55/a65' if d == 'D01' else 'this package D02 source a55/a65'}))


def pilot_target_catalog(root: Path) -> None:
    P = pilot_paths(root)
    cat = P['catalog'] / 'catalog' / 'domain_catalog.json'
    if cat.exists():
        return
    skills, domains, bindings = [], {}, {}
    for d, ds in DOMAIN_BINDINGS.items():
        fr = context.read_json(P['formation'] / d / 'frozen_skill.json')
        if fr['status'] == 'FROZEN_SELECTED':
            s = dsk.skill_from_json(fr['skill'])
            if s.status != 'FROZEN_SELECTED' or s.domain_id != d:
                raise PermissionError('only a real FROZEN_SELECTED card of its own domain enters the Target catalog')
            skills.append(s.to_json())
        refs = {}
        for i, job in enumerate(REFERENCE_PROFILE_JOBS, 1):
            ctx = context.open_job(ds, job, P['formation'] / 'reference_t' / d, 'material')
            view, binding = dsk.batch_profile(ctx.overview(), ctx.job)
            pdir = P['formation'] / 'reference_profiles' / d / job
            write_once(pdir / 'profile.json', view)
            write_once(pdir / 'runtime_binding.json', {**binding, 'domain_id': d, 'role': 'formation_reference'})
            refs['%s.ref%d' % (d, i)] = {'job': job, 'profile': _rel(pdir / 'profile.json', P['catalog']), 'runtime_binding': _rel(pdir / 'runtime_binding.json', P['catalog'])}
        domains[d] = {'reference_batch_ids': sorted(refs), 'formation_status': fr['status']}
        bindings[d] = {'dataset': ds, 'reference_batches': refs}
    write_once(P['catalog'] / 'catalog' / 'runtime_bindings.json', {'note': 'RUNTIME ONLY', 'domains': bindings})
    write_once(cat, {'catalog_version': 1, 'status': 'EVOLVE_PILOT_FROZEN', 'domains': domains, 'skills': skills, 'generic_control': generic_control_record(),
                     'generic_skill_id': None, 'frozen_before_first_target_fit': True, 'epoch': time.time()})


def pilot_run(root: Path = PILOT_ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    P = pilot_paths(root)
    status = {}

    def stop(where, st):
        status.update({'stopped_at': where, 'stage_status': st, 'epoch': time.time()})
        context.write_json(root / 'pilot_status.json', status)
        print('PILOT_STOP', where, st.get('status'), flush=True)
        return status
    if not (root / 'pilot_config.json').exists():
        pilot_preflight(root)
    # Source: D02 no_skill trajectories (commit both, then C_B; no E)
    cfgp = pilot_stage_config(root, 'source', jobs=list(D02_SOURCE_JOBS), order={j: ['no_skill'] for j in D02_SOURCE_JOBS}, catalog_root=P['pre_catalog'], labels_e=False)
    st = run_stage(root, 'source', cfgp)
    status['source'] = st
    if st['status'] != 'FINISHED':
        return stop('source', st)
    # Formation: census + one Slow per domain; Select on a75 (all commits, then C_B); freeze
    pilot_formation_calls(root)
    st = pilot_select_stage(root)
    status['select'] = st
    if st['status'] not in ('FINISHED', 'SKIPPED'):
        return stop('select', st)
    pilot_freeze(root)
    pilot_target_catalog(root)
    # Target: four jobs, frozen cards, labels C_B then E after every branch
    cfgp = pilot_stage_config(root, 'target', jobs=list(TARGET_JOBS), order=TARGET_ORDER, catalog_root=P['catalog'], labels_e=True, random_seeds=TARGET_RANDOM_SEEDS)
    st = run_stage(root, 'target', cfgp)
    status['target'] = st
    if st['status'] != 'FINISHED':
        return stop('target', st)
    status['finished_epoch'] = time.time()
    context.write_json(root / 'pilot_status.json', status)
    pilot_result(root)
    print('PILOT_FINISHED', flush=True)
    return status


# ----------------------------------------------------------------------------- readouts
def _plan_scores(bdir: Path, job: str) -> dict:
    jdir = bdir / job
    e = context.read_json(jdir / 'e_scores.json')['cells'] if (jdir / 'e_scores.json').exists() else {}
    out = {}
    for c in (jdir / 'cells').glob('*.json') if (jdir / 'cells').exists() else []:
        cell = context.read_json(c)
        if cell.get('status') != 'OK':
            continue
        row = out.setdefault(cell['material_id'], {})
        row[cell['model_seed']] = {'c_a': cell['scores'].get('c_a', {}).get('normalized_mse_macro'), 'c_b': cell['scores'].get('c_b', {}).get('normalized_mse_macro'),
                                   'e': (e.get(cell['cell_id']) or {}).get('e', {}).get('normalized_mse_macro')}
    return out


def _vec(scores: dict, plan: str | None, block: str):
    if plan is None or plan not in scores or set(scores[plan]) != set(PILOT_SEEDS) or any(scores[plan][s][block] is None for s in PILOT_SEEDS):
        return None
    return [scores[plan][s][block] for s in PILOT_SEEDS]


def _paired(a, b) -> dict | None:
    if a is None or b is None:
        return None
    d = [x - y for x, y in zip(a, b)]
    return {'delta_by_seed': d, 'mean': statistics.mean(d), 'se': statistics.stdev(d) / len(d) ** 0.5, 'positive_means': 'second (reference) arm better'}


def _behavior(bdir: Path) -> dict:
    tr = bdir / 'trace.jsonl'
    rows = [json.loads(x) for x in tr.read_text(encoding='utf-8').splitlines() if x.strip()] if tr.exists() else []
    req = [r for r in rows if r['event'] == 'fast_request']
    tools = [r for r in rows if r['event'] == 'tool_completed']
    builds = [{'plan_id': r['arguments'].get('plan_id'), 'default': (r['arguments'].get('policy') or {}).get('default'), 'rules': (r['arguments'].get('policy') or {}).get('rules'),
               'rationale': (r['arguments'].get('policy') or {}).get('rationale')} for r in rows if r['event'] == 'tool_started' and r['tool'] == 'build_material']
    commit = next((r for r in rows if r['event'] == 'tool_completed' and r['tool'] == 'commit'), None)
    return {'fast_requests': len(req), 'requests_with_loaded_guidance': sum(bool(r.get('guidance', {}).get('loaded')) for r in req),
            'tool_sequence': [r['tool'] for r in tools], 'builds': builds, 'evaluated': [r['output'].get('plan_id') for r in tools if r['tool'] == 'evaluate'],
            'compares': [[r['output'].get('a'), r['output'].get('b')] for r in tools if r['tool'] == 'compare'],
            'rejections': sum(r['event'] == 'tool_rejected' for r in rows), 'deferred': sum(r['event'] == 'action_batch_deferred' for r in rows),
            'commit': commit['output'] if commit else None, 'random_materials': [r.get('material_spec', {}).get('policy') for r in rows if r['event'] == 'random_material']}


def _unit_tokens(stage_root: Path) -> dict:
    out = {}
    for req in sorted((stage_root / 'raw_responses').glob('*_request.json')):
        q = context.read_json(req)
        resp = req.with_name(req.name.replace('_request.json', '_response.json'))
        u = (context.read_json(resp).get('usage') or {}) if resp.exists() else {}
        row = out.setdefault('%s:%s' % (q.get('role'), q.get('unit')), {'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'missing_responses': 0})
        row['requests'] += 1
        row['missing_responses'] += 0 if resp.exists() else 1
        row['prompt_tokens'] += int(u.get('prompt_tokens') or 0)
        row['completion_tokens'] += int(u.get('completion_tokens') or 0)
    return out


def pilot_result(root: Path = PILOT_ROOT) -> dict:
    P = pilot_paths(root)
    res = {'package': 'DEV-DOMAIN-SKILL-V1-EVOLVE-PILOT', 'exposure': spec.EXPOSURE, 'formation': {}, 'select': {}, 'target': {}, 'costs': {}}
    for d in DOMAIN_BINDINGS:
        f = P['formation'] / d
        prop = context.read_json(f / 'propose.json') if (f / 'propose.json').exists() else None
        cen = context.read_json(f / 'census.json') if (f / 'census.json').exists() else None
        res['formation'][d] = {'census_status': cen and cen['census_status'], 'census_branches': [b['branch'] for b in (cen or {}).get('branches', [])],
                               'propose_status': prop and prop['status'], 'attempts': prop and prop.get('attempts'),
                               'candidates': [{k: s[k] for k in ('skill_id', 'variant', 'workflow', 'principles', 'observable_applicability', 'applicability_summary', 'rendered_body')}
                                              for s in (prop or {}).get('skills', [])],
                               'research_modes': (prop or {}).get('research_modes'), 'rationales': (prop or {}).get('rationales'),
                               'keep_rationale': (prop or {}).get('rationale') if (prop or {}).get('status') == 'KEEP' else None,
                               'frozen': context.read_json(f / 'frozen_skill.json') if (f / 'frozen_skill.json').exists() else None}
        if (f / 'selection.json').exists():
            sel = context.read_json(f / 'selection.json')
            res['select'][d] = {'status': sel['status'], 'selected': sel['selected'], 'eligible_scores': sel['eligible_scores'], 'runs': sel['runs']}
            job = SELECT_JOBS[d]
            for oid, runs in sel['runs'].items():
                b = P['select'] / runs[job].get('branch', '') if runs[job].get('branch') else None
                if b is not None and b.exists():
                    sc = _plan_scores(b, job)
                    res['select'][d]['runs'][oid][job].update({'behavior': _behavior(b), 'c_a_delivered': _vec(sc, runs[job].get('committed_plan_id'), 'c_a'),
                                                               'baselines_c_b': {m: _vec(sc, m, 'c_b') for m in ('None', 'FixedMixup')}})
    if (P['target'] / 'execution_finished.json').exists():
        summary_rel = []
        for job in TARGET_JOBS:
            arms, vec = {}, {}
            for arm in TARGET_ORDER[job]:
                b = P['target'] / ('%s_%s' % (job, arm))
                if (b / 'treatment.json').exists():
                    arms[arm] = {'status': 'NO_TREATMENT', 'route': context.read_json(b / 'treatment.json')['route']}
                    continue
                if not (b / 'branch_result.json').exists():
                    arms[arm] = {'status': 'NOT_RUN'}
                    continue
                r = context.read_json(b / 'branch_result.json')
                sc = _plan_scores(b, job)
                plan = r['committed_plan_id'] if r['status'] == 'COMPLETE' else None
                vec[arm] = _vec(sc, plan, 'e')
                arms[arm] = {'status': r['status'], 'failure_kind': r.get('failure_kind'), 'committed_plan_id': plan, 'calls': r['calls'], 'new_evaluations': r['new_evaluations'],
                             'e_by_seed': vec[arm], 'e_mean': statistics.mean(vec[arm]) if vec[arm] else None, 'c_a_by_seed': _vec(sc, plan, 'c_a'), 'c_b_by_seed': _vec(sc, plan, 'c_b'),
                             'route_used': context.read_json(b / 'route_used.json') if (b / 'route_used.json').exists() else None, 'behavior': _behavior(b),
                             'n_plans_fitted': len(sc)}
                for m in ('None', 'FixedMixup'):
                    if vec.get(m) is None and _vec(sc, m, 'e') is not None:
                        vec[m] = _vec(sc, m, 'e')
            ref = vec.get('known_domain')
            comps = {'no_skill_minus_skill': _paired(vec.get('no_skill'), ref), 'generic_minus_skill': _paired(vec.get('generic'), ref),
                     'random_minus_skill': _paired(vec.get('random'), ref), 'fixed_minus_skill': _paired(vec.get('FixedMixup'), ref), 'none_minus_skill': _paired(vec.get('None'), ref),
                     'none_minus_no_skill': _paired(vec.get('None'), vec.get('no_skill')), 'fixed_minus_no_skill': _paired(vec.get('FixedMixup'), vec.get('no_skill')),
                     'no_skill_minus_generic': _paired(vec.get('no_skill'), vec.get('generic')), 'random_minus_no_skill': _paired(vec.get('random'), vec.get('no_skill')),
                     'random_minus_generic': _paired(vec.get('random'), vec.get('generic'))}
            if comps['no_skill_minus_skill'] is not None:
                summary_rel.append(comps['no_skill_minus_skill']['mean'] / statistics.mean(vec['no_skill']))
            res['target'][job] = {'arms': arms, 'baselines_e_by_seed': {m: vec.get(m) for m in ('None', 'FixedMixup')}, 'comparisons': comps}
        res['target_summary'] = {'relative_improvement_of_skill_over_no_skill_equal_weight': statistics.mean(summary_rel) if summary_rel else None,
                                 'complete_pairs': len(summary_rel), 'planned_pairs': len(TARGET_JOBS)}
    stages = context.read_json(P['stages']) if P['stages'].exists() else {}
    led = {k: v for k, v in context.read_json(P['ledger']).items() if k != 'events'} if P['ledger'].exists() else {}
    names = [n for n in ('source', 'formation', 'target') if n in stages]
    for i, n in enumerate(names):
        a = stages[n]['snapshot_at_start']
        bsnap = stages[names[i + 1]]['snapshot_at_start'] if i + 1 < len(names) else {**{k: led.get(k, 0) for k in a}, 'elapsed_s': time.time() - led.get('started_epoch', time.time())}
        res['costs'][n] = {'allocation': stages[n]['allocation'], 'caps': stages[n]['caps'],
                           'used': {k: bsnap[k] - a[k] for k in a}}
    res['costs']['package_ledger'] = led
    res['costs']['tokens_by_role_unit'] = {n: _unit_tokens(P[k]) for n, k in (('source', 'source'), ('formation_slow', 'formation'), ('select', 'select'), ('target', 'target'))}
    context.write_json(root / 'pilot_result.json', res)
    return res


# ============================================================================= the integration smoke
class _Scripted:
    """Scripted stand-in: call(payload) for router/Slow, fast(unit) for Fast. Records every request; raises queued exceptions."""

    def __init__(self, outs=(), fast_outs=()):
        self.outs, self.fast_outs, self.seen, self.fast_seen, self.fatal = list(outs), list(fast_outs), [], [], False

    def __call__(self, payload):
        self.seen.append(copy.deepcopy(payload))
        out = self.outs.pop(0)
        if isinstance(out, Exception):
            raise out
        return copy.deepcopy(out)

    def call(self, role, unit, payload, system, max_tokens=4096, request_timeout=300.):
        self.seen.append({'role': role, 'unit': unit, 'payload': copy.deepcopy(payload), 'system_head': system[:40]})
        out = self.outs.pop(0)
        if isinstance(out, Exception):
            raise out
        return copy.deepcopy(out)

    def fast(self, unit):
        def f(req):
            self.fast_seen.append(copy.deepcopy(req))
            return copy.deepcopy(self.fast_outs.pop(0))
        return f


def _synthetic_baselines(adapter, seeds):
    """None/FixedMixup built for real on T; their C_A is SYNTHETIC (explicit numbers, control flow only, no model exists)."""
    out = []
    for i, (mid, pol) in enumerate((('None', policy.identity_policy()), ('FixedMixup', policy.fixed_mixup_policy()))):
        c = adapter.build_material({'plan_id': mid, 'policy': pol}, remaining_seconds=600.) if mid not in adapter.refs else br.Candidate(mid, adapter.specs[mid], adapter.ctx.job.job_id)
        losses = tuple(tuple(tuple([0.5 + 0.01 * i + 0.001 * r] * 32) for _ in range(2)) for r in range(len(seeds)))
        out.append(replace(c, model_seeds=tuple(seeds), model_refs=tuple('SYNTHETIC_NO_MODEL_%s_%d' % (mid, s) for s in seeds), ca_losses=losses))
    return tuple(out)


def smoke(root: Path = BUILD_ROOT, out: Path | None = None) -> dict:
    import subprocess as _subprocess
    import numpy as np
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from methods.ttha.batch_base import budget, commit, data, llm, materials
    from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp

    root = Path(root)
    if not (root / 'catalog' / 'domain_catalog.json').exists():
        raise RuntimeError('run --build-profiles first')
    checks, loads, launches = {}, [], []
    started = time.time()

    def check(name, ok, **detail):
        checks[name] = {'ok': bool(ok), **detail}
        if not ok:
            raise AssertionError('%s: %s' % (name, detail))

    def refused(fn, exc_types, contains=''):
        try:
            fn()
        except exc_types as exc:
            return contains in str(exc)
        return False

    real_load = data.load_slice

    def logged_load(dataset, a, b, roster=None):
        loads.append((dataset, a, b))
        return real_load(dataset, a, b, roster=roster)

    def no_launch(*a, **k):
        launches.append(a[0] if a else k.get('args'))
        raise AssertionError('a subprocess/fit/stage launch was attempted inside the smoke')

    out = Path(out or root)
    (out / 'routing').mkdir(parents=True, exist_ok=True)
    catalog, bindings, skills_all = load_catalog(root)
    fixtures = {s.skill_id: s for s in skills_all}
    feats_allowed = rt.BATCH_FIELDS
    with TemporaryDirectory(ignore_cleanup_errors=True) as tdn, patch.object(data, 'load_slice', logged_load), patch.object(_subprocess, 'run', no_launch), patch.object(_subprocess, 'Popen', no_launch):
        td = Path(tdn)
        led0 = rt.RuntimeLedger(td / 'ledger_zero.json', max_fit_attempts=0, max_llm_requests=0, max_llm_tokens=0, max_wall_s=36000, max_retries=0)

        # ---- 1. identity-free profiles; references exclude the query
        views = {}
        for d in DOMAIN_BINDINGS:
            for job in REFERENCE_JOBS + QUERY_JOBS:
                v = context.read_json(root / 'profiles' / d / job / 'profile.json')
                b = context.read_json(root / 'profiles' / d / job / 'runtime_binding.json')
                dsk.assert_identity_free(v)
                check('profile_fields_%s_%s' % (d, job), set(v['fields']) == set(dsk.PROFILE_FIELDS) and b['dataset'] == DOMAIN_BINDINGS[d] and
                      all(x is None or {'p25', 'median', 'p75', 'min', 'max', 'n_valid'} == set(x) for x in v['fields'].values()) and
                      'a50' not in json.dumps(v) and 'missing_count' not in v['fields'], t=b['t'])
                views[(d, job)] = (v, b)
        check('identity_injection_refused', all(refused(lambda bad=bad: dsk.assert_identity_free(bad), PermissionError) for bad in (
            {**views[('D01', 'a50')][0], 'job_id': 'x'}, {**views[('D01', 'a50')][0], 'notes': {'x': 'traffic like'}},
            {**views[('D01', 'a50')][0], 'notes': {'x': 'C:\\data\\x.csv'}}, {'fields': {'e': 1}}, {'fields': {'x': 'a50_D01'}})))
        refs, ref_ids = domain_refs(root, catalog, bindings, query_binding=views[('D02', 'a50')][1])
        check('references_exclude_query', ref_ids == ['D01.ref1', 'D01.ref2', 'D02.ref1', 'D02.ref2'] and
              all(r != views[(d, 'a50')][0] for d in DOMAIN_BINDINGS for r in refs[d]), reference_batch_ids=ref_ids)
        check('overlapping_reference_refused', refused(lambda: domain_refs(root, catalog, bindings, query_binding={**views[('D02', 'a50')][1], 'workload_rows': views[('D02', 'a40')][1]['train_rows']}), PermissionError, 'overlaps'))

        # ---- 2. two domains: T, population, scaler and materials bound per dataset; old default unchanged
        adapters, mats = {}, {}
        for d, ds in DOMAIN_BINDINGS.items():
            ad = rt.Adapter(td / d / 'adapter', 'a50', led0, REPO, dataset=ds, seeds=rt.SEEDS)
            base_c = _synthetic_baselines(ad, rt.SEEDS)
            direct = context.open_job(ds, 'a50', td / d / 'direct', 'material')
            same = [np.array_equal(ad.ctx.scaler.mean, direct.scaler.mean), np.array_equal(ad.ctx.scaler.scale, direct.scaler.scale),
                    np.array_equal(ad.ctx.parents.X_norm, direct.parents.X_norm), np.array_equal(ad.ctx.parents.y_norm, direct.parents.y_norm)]
            for mid, pol in (('None', policy.identity_policy()), ('FixedMixup', policy.fixed_mixup_policy())):
                ref = materials.build(direct, policy.compile_policy(pol, direct.overview()['entities'])['assignment'], mid)
                with np.load(ad.refs[mid].path) as x, np.load(ref.path) as y:
                    same += [np.array_equal(x['Xc'], y['Xc']), np.array_equal(x['Yc'], y['Yc']), np.array_equal(x['has_child'], y['has_child'])]
            js = ad.ctx.job
            insp = materials.inspect_material(ad.ctx, ad.refs['FixedMixup'])['summary']
            with np.load(ad.ctx.job_dir / 'scaler.npz') as z:
                sc = {k: z[k] for k in ('dataset', 't')}
            check('materials_bound_%s' % d, all(same) and js.dataset == ds and js.t == spec.t_of(ds, 0.5) and ad.ctx.slice.roster == spec.DATASETS[ds]['roster'] and
                  str(sc['dataset']) == ds and int(sc['t']) == js.t and ad.refs['None'].n_identity == 32 and ad.refs['FixedMixup'].n_identity == 0 and
                  views[(d, 'a50')][0] == dsk.batch_profile(ad.ctx.overview(), js)[0],
                  dataset_runtime_only=ds, t=js.t, train_rows=list(js.train_range), equal_checks=len(same), fixedmixup_changed_entities=insp['n_changed_entities'],
                  fixedmixup_child_x_change_rms_mean=insp['child_x_change_rms_mean'], scaler_floor_hits=ad.ctx.scaler.floor_hits)
            adapters[d], mats[d] = (ad, base_c), ad.refs['FixedMixup'].path
        with np.load(mats['D01']) as x, np.load(mats['D02']) as y:
            differ = not np.array_equal(x['Xc'], y['Xc'])
        check('same_a_not_mixed', differ and adapters['D01'][0].ctx.job.t != adapters['D02'][0].ctx.job.t and
              not np.array_equal(adapters['D01'][0].ctx.scaler.mean, adapters['D02'][0].ctx.scaler.mean) and
              refused(lambda: context.open_job('traffic', 'a50', td / 'D01' / 'adapter', 'material'), RuntimeError, 'dataset drift'), t_d01=adapters['D01'][0].ctx.job.t, t_d02=adapters['D02'][0].ctx.job.t)
        dflt = rt.Adapter(td / 'default', 'a50', led0, REPO)
        dflt.build_material({'plan_id': 'FixedMixup', 'policy': policy.fixed_mixup_policy()}, remaining_seconds=600.)
        hist = rt.Adapter(td / 'hist', 'a55', led0, REPO)
        hist.build_material({'plan_id': 'R', 'policy': policy.fixed_mixup_policy()}, remaining_seconds=600.)
        probe = REPO / '_scratch' / 'ts_aug_donor_pattern_probe' / 'materials' / 'a55__children.npz'
        with np.load(dflt.refs['FixedMixup'].path) as x, np.load(mats['D01']) as y, np.load(probe) as old, np.load(hist.refs['R'].path) as new:
            check('old_default_electricity_materials_unchanged', dflt.ctx.job.dataset == 'electricity' and np.array_equal(x['Xc'], y['Xc']) and np.array_equal(x['Yc'], y['Yc']) and
                  np.array_equal(old['RX'], new['Xc']) and np.array_equal(old['RY'], new['Yc']),
                  reference='historical _scratch/ts_aug_donor_pattern_probe/materials/a55__children.npz (read-only)')
        # semantic cache binding (existing fields, no hash) and zero-cap fit refusal before any launch
        ad2 = adapters['D02'][0]
        cid = 'a50__None__s%d' % rt.SEEDS[0]
        (ad2.ctx.job_dir / 'cells').mkdir(exist_ok=True)
        (td / 'fake.pt').write_bytes(b'')
        good = {'cell_id': cid, 'status': 'OK', 'job_id': 'a50', 'dataset': 'traffic', 'material_id': 'None', 'model_seed': rt.SEEDS[0], 'roster': ad2.ctx.job.roster,
                'rows_read': list(ad2.ctx.job.evaluate_rows), 'model_path': str(td / 'fake.pt'), 'scores': {'c_a': {'normalized_mse_macro': 0.5}}, 'synthetic': True}
        cache_ok = []
        for mutate, expect in ((dict(), None), ({'dataset': 'electricity'}, 'another dataset'), ({'roster': list(reversed(ad2.ctx.job.roster))}, 'another population'),
                               ({'rows_read': [0, 1]}, 'another training'), ({'material_id': 'FixedMixup'}, 'another candidate')):
            context.write_json(ad2.ctx.job_dir / 'cells' / (cid + '.json'), {**good, **mutate})
            if expect is None:
                cache_ok.append(rt.fit(ad2.ctx, ad2.refs['None'], (rt.SEEDS[0],), led0, REPO)[0]['dataset'] == 'traffic')
            else:
                cache_ok.append(refused(lambda: rt.fit(ad2.ctx, ad2.refs['None'], (rt.SEEDS[0],), led0, REPO), RuntimeError, expect))
        (ad2.ctx.job_dir / 'cells' / (cid + '.json')).unlink()
        check('cache_hit_semantic_binding', all(cache_ok) and led0.s['fit_attempts'] == 0, cases=len(cache_ok))
        check('zero_fit_cap_refuses_before_launch', refused(lambda: rt.fit(ad2.ctx, ad2.refs['FixedMixup'], rt.SEEDS, led0, REPO), budget.BudgetExhausted) and not launches)

        # ---- 3/4. routing -> the same run_job loads the fixture; unmatched/unknown keep every public tool
        fixture_skills_routable = dsk.routable(skills_all, allow_fixture=True)
        check('fixtures_not_routable_in_live', dsk.routable(skills_all) == [] and all(s.status == 'SMOKE_FIXTURE' for s in skills_all))
        commit_none = {'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'None', 'reason': 'smoke: scripted commit of a synthetic baseline'}}]}

        def fast_run(d, h):
            ad, base_c = adapters[d]
            client = _Scripted(fast_outs=[commit_none])
            res = br.run_job(job_id='a50', knowledge=h, adapter=ad, client=client.fast('smoke'), seeds=rt.SEEDS, baselines=base_c, allowed_features=feats_allowed,
                             tool_contracts=CONTRACTS, limits=br.Limits(wall_seconds=600.))
            return res, client.fast_seen[0]

        requests = {}
        for d in DOMAIN_BINDINGS:
            ad, _ = adapters[d]
            feats = ad.overview()['batch_features']
            qview = views[(d, 'a50')][0]
            known = dsk.route_known_domain(d, fixture_skills_routable, feats)
            router = _Scripted(outs=[{'selected_skill_id': 'SK-FIX-' + d, 'status': 'SELECTED', 'evidence_fields': ['lag24_corr', 'T_length'], 'rationale': 'smoke script'}])
            prof = dsk.route_profile_match(qview, refs, fixture_skills_routable, router, feats, diagnostic_domain_id=d)
            payload_text = json.dumps(router.seen[0], ensure_ascii=False)
            check('router_payload_identity_free_%s' % d, not any(n in payload_text.lower() for n in spec.DATASETS) and 'a50' not in payload_text and 'a30' not in payload_text
                  and 'Workflow:' not in payload_text and len(router.seen[0]['candidate_domains']) == 2, payload_chars=len(payload_text))
            context.write_json(out / 'routing' / ('%s_profile_match_router_request.json' % d), {'kind': 'SCRIPTED_ROUTER_REQUEST', 'system': dsk.ROUTER_SYSTEM, 'payload': router.seen[0],
                                                                                              'scripted_response': 'SELECTED SK-FIX-%s' % d, 'route': prof})
            for mode, route in (('known_domain', known), ('profile_match', prof)):
                skill = dsk.routed_skill(route, fixture_skills_routable)
                res, req = fast_run(d, dsk.skill_knowledge(skill))
                loaded = req['guidance']['loaded']
                check('fast_loads_%s_%s' % (mode, d), res.status == 'COMPLETE' and route['status'] in ('KNOWN_DOMAIN_SELECTED', 'SELECTED') and skill.skill_id == 'SK-FIX-' + d and
                      len(loaded) == 1 and loaded[0] == {'hook': 'experiment_guidance', 'body': fixtures['SK-FIX-' + d].rendered_body} and req['tools'] == sorted(br.TOOLS),
                      route_status=route['status'], variant=skill.variant, body_chars=len(loaded[0]['body']))
                requests[(d, mode)] = req
                context.write_json(out / 'routing' / ('%s_%s_fast_request.json' % (d, mode)), {
                    'kind': 'ACTUAL_FAST_REQUEST_SCRIPTED_CLIENT', 'route': route, 'skill_id': skill.skill_id, 'skill_status': skill.status, 'rendered_body': skill.rendered_body,
                    'synthetic': 'baseline C_A feedback in candidates[] is SYNTHETIC; no model was trained; T observations and materials are real', 'request': req})
            check('known_domain_is_labeled_reference', known['inference'] == 'NONE__EXTERNAL_DOMAIN_LABEL' and known['llm_requests'] == 0 and prof['llm_requests'] == 1)

        d = 'D02'
        ad, _ = adapters[d]
        feats = ad.overview()['batch_features']
        qview = views[(d, 'a50')][0]
        ids = [s.skill_id for s in fixture_skills_routable]
        rec = {}
        for name, out in (('abstain', {'selected_skill_id': None, 'status': 'ABSTAIN', 'evidence_fields': [], 'rationale': 'smoke: insufficient'}),
                          ('foreign_id', {'selected_skill_id': 'SK-OTHER', 'status': 'SELECTED', 'evidence_fields': ['lag24_corr'], 'rationale': 'x'}),
                          ('bad_field', {'selected_skill_id': ids[0], 'status': 'SELECTED', 'evidence_fields': ['dataset'], 'rationale': 'x'}),
                          ('not_json', json.JSONDecodeError('Expecting value', 'x', 0)), ('transport', llm.TransportFault('TRANSPORT_TRANSIENT_FAULT')),
                          ('account', llm.AccountFault('ACCOUNT_OR_PERMISSION_FAULT')), ('budget', budget.BudgetExhausted('cap'))):
            rec[name] = dsk.route_profile_match(qview, refs, fixture_skills_routable, _Scripted(outs=[out]), feats, diagnostic_domain_id=d)['status']
        rec['no_candidates'] = dsk.route_profile_match(qview, refs, [], _Scripted(), feats)['status']
        check('route_outcomes_kept_apart', rec == {'abstain': 'ABSTAIN', 'foreign_id': 'ROUTER_PARSE_OR_VALIDATION_FAILED', 'bad_field': 'ROUTER_PARSE_OR_VALIDATION_FAILED',
                                                   'not_json': 'ROUTER_PARSE_OR_VALIDATION_FAILED', 'transport': 'ROUTER_CALL_FAILED', 'account': 'ROUTER_CALL_FAILED',
                                                   'budget': 'ROUTER_CALL_FAILED', 'no_candidates': 'NO_CANDIDATE_SKILLS'}, outcomes=rec)
        nomatch = dsk.make_skill(skill_id='SK-FIX-NOMATCH', domain_id='D02', revision=0, workflow='[SMOKE_FIXTURE] scope control text.', principles=None,
                                 applicability_summary='scope control', compatibility_note='smoke', observable_applicability={'feature': 'batch_median_lag24_corr', 'op': '>', 'value': 2.0},
                                 evidence_refs=['fixture:smoke'], legal_evidence_refs=FIXTURE_REFS, source_stage='fixture', status='SMOKE_FIXTURE')
        r_nm = dsk.route_profile_match(qview, refs, [nomatch], _Scripted(outs=[{'selected_skill_id': 'SK-FIX-NOMATCH', 'status': 'SELECTED', 'evidence_fields': ['lag24_corr'], 'rationale': 'x'}]), feats)
        res_nm, req_nm = fast_run(d, dsk.skill_knowledge(dsk.routed_skill(r_nm, [nomatch])))
        res_ab, req_ab = fast_run(d, dsk.skill_knowledge(None))
        sel_req = requests[(d, 'profile_match')]
        same_perm = lambda q: q['tools'] == sel_req['tools'] and q['tool_contracts'] == sel_req['tool_contracts'] and {k: v for k, v in q['remaining'].items() if k != 'seconds'} == {k: v for k, v in sel_req['remaining'].items() if k != 'seconds'} and q['candidates'] == sel_req['candidates']
        check('unmatched_or_abstain_keeps_fast_permissions', r_nm['status'] == 'SELECTED_SCOPE_NOT_MATCHED' and req_nm['guidance']['loaded'] == [] and
              req_nm['guidance']['applicability'][0]['state'] == 'NO_MATCH' and req_ab['guidance']['loaded'] == [] and res_nm.status == res_ab.status == 'COMPLETE' and same_perm(req_nm) and same_perm(req_ab))
        unknown_state = dsk.skill_knowledge(fixtures['SK-FIX-D01']).render({}, feats_allowed)
        nomatch_unknown = br.Knowledge(0, (br.Guidance('experiment_guidance', 'x', {'feature': 'batch_median_r_gap', 'op': '>', 'value': 0.0}),)).render({}, feats_allowed)
        null_scope = br.Knowledge(0, (br.Guidance('experiment_guidance', 'x', None),)).render(feats, feats_allowed)
        check('scope_semantics', unknown_state['loaded'] and nomatch_unknown['applicability'][0]['state'] == 'UNKNOWN' and nomatch_unknown['loaded'] == [] and
              null_scope['applicability'][0]['state'] == 'INCOMPLETE_NO_APPLICABILITY' and null_scope['loaded'] == [] and
              refused(lambda: dsk.make_skill(skill_id='SK-NULL', domain_id='D01', revision=0, workflow='x text', principles=None, applicability_summary='s', compatibility_note='c',
                                             observable_applicability=None, evidence_refs=['fixture:smoke'], legal_evidence_refs=FIXTURE_REFS, source_stage='fixture', status='SMOKE_FIXTURE'), ValueError, 'null'),
              const_true_loads=bool(unknown_state['loaded']))
        check('skill_text_bans', all(refused(lambda w=w: dsk.make_skill(skill_id='SK-BAD', domain_id='D01', revision=0, workflow=w, principles=None, applicability_summary='s', compatibility_note='c',
                                                                         observable_applicability={'const': True}, evidence_refs=['fixture:smoke'], legal_evidence_refs=FIXTURE_REFS,
                                                                         source_stage='fixture', status='SMOKE_FIXTURE'), ValueError) for w in
                                         ('use the traffic answer', 'as in a50_D02 do this', 'entity_3 needs mixup', 'x' * 1300)))

        # ---- 5. offline formation: census / propose (workflow_only, workflow+principles, KEEP, failures) / select / freeze
        trepo = td / 'repo'
        (trepo / '_scratch' / 'dev_batch_research_roundtrip').mkdir(parents=True)
        shutil.copy(REPO / '_scratch' / 'dev_batch_research_roundtrip' / 'config.json', trepo / '_scratch' / 'dev_batch_research_roundtrip' / 'config.json')
        census = _fixture_census_tree(trepo, root)
        c1 = census_domain('D01', [('fixture_pkg', 'a30_fast', 'a30')], repo=trepo, workload_end_row=spec.t_of('electricity', 0.3) + 400)
        refs_legal = c1['legal_evidence_refs']
        check('census_domain_bound', c1['census_status'] == 'CENSUS_COMPLETE' and len(refs_legal) > 5 and '"e_scores"' not in json.dumps(c1) and
              refused(lambda: census_domain('D02', [('fixture_pkg', 'a30_fast', 'a30')], repo=trepo, workload_end_row=10 ** 6), PermissionError, 'relabeling') and
              refused(lambda: census_domain('D01', [('fixture_pkg', 'a30_fast', 'a30')], repo=trepo, workload_end_row=spec.t_of('electricity', 0.3) + 100), PermissionError, 'time boundary') and
              c1['source_binding'][0]['t'] == spec.t_of('electricity', 0.3) and
              census_domain('D02', [], workload_end_row=0)['census_status'] == 'NO_PROCESS_TRAJECTORIES', n_refs=len(refs_legal), fixture=census)
        cand = lambda cid, mode, wf, pr: {'candidate_id': cid, 'research_mode': mode, 'workflow': wf, 'principles': pr, 'observable_applicability': {'const': True},
                                          'applicability_summary': 'smoke candidate', 'evidence_refs': [refs_legal[0]], 'rationale': 'smoke'}
        two = {'decision': 'PROPOSE', 'candidates': [cand('W1', 'observe then one uniform test', 'Read the summary, build one uniform plan, compare with baselines, commit.', None),
                                                     cand('W2', 'stop early unless evidence', 'Compare the evaluated baselines first; construct only when observations support a hypothesis.', 'Seed uncertainty only; no causal row labels.')]}
        bad = {'decision': 'PROPOSE', 'candidates': [cand('W1', 'm', 'copy the traffic winner', None)]}
        outs = {'two': dsk.propose(c1, domain_id='D01', call=_Scripted(outs=[two])),
                'keep': dsk.propose(c1, domain_id='D01', call=_Scripted(outs=[{'decision': 'KEEP', 'rationale': 'no reusable workflow'}])),
                'bad_bad': dsk.propose(c1, domain_id='D01', call=_Scripted(outs=[bad, bad])),
                'bad_fixed': dsk.propose(c1, domain_id='D01', call=_Scripted(outs=[bad, two])),
                'transport': dsk.propose(c1, domain_id='D01', call=_Scripted(outs=[llm.TransportFault('TRANSPORT_TRANSIENT_FAULT')])),
                'same_mode': dsk.propose(c1, domain_id='D01', call=_Scripted(outs=[{'decision': 'PROPOSE', 'candidates': [two['candidates'][0], {**two['candidates'][1], 'research_mode': 'observe then one uniform test'}]}] * 2))}
        variants = [s.variant for s in outs['two']['skills']]
        check('propose_paths', outs['two']['status'] == 'PROPOSED' and variants == ['workflow_only', 'workflow+principles'] and
              all(s.status == 'CANDIDATE_TEST_ONLY' for s in outs['two']['skills']) and dsk.routable(outs['two']['skills']) == [] and
              outs['keep']['status'] == 'KEEP' and outs['keep']['skills'] == [] and outs['bad_bad']['status'] == 'PROPOSE_PARSE_OR_VALIDATION_FAILED' and
              outs['bad_fixed']['status'] == 'PROPOSED' and len(outs['bad_fixed']['attempts']) == 2 and outs['transport']['status'] == 'PROPOSE_CALL_FAILED' and
              outs['same_mode']['status'] == 'PROPOSE_PARSE_OR_VALIDATION_FAILED', statuses={k: v['status'] for k, v in outs.items()}, variants=variants)
        W1, W2 = outs['two']['skills']
        options = {dsk.NO_SKILL: None, W1.skill_id: W1, W2.skill_id: W2}
        seen_k = []

        def synth(losses, incomplete=()):
            def run(oid, knowledge, job):
                seen_k.append((oid, [e.body for e in knowledge.entries]))
                if (oid, job) in incomplete:
                    return {'status': 'INCOMPLETE', 'synthetic': True}
                return {'status': 'COMPLETE', 'loss_by_seed': [losses[oid]] * 3, 'synthetic': True, 'committed_plan_id': 'p1'}
            return run
        s1 = dsk.select(options, ['a30', 'a40'], synth({dsk.NO_SKILL: .30, W1.skill_id: .31, W2.skill_id: .28}), block='c_b')
        s2 = dsk.select(options, ['a30'], synth({dsk.NO_SKILL: .30, W1.skill_id: .31, W2.skill_id: .32}), block='c_b')
        s3 = dsk.select(options, ['a30', 'a40'], synth({dsk.NO_SKILL: .30, W1.skill_id: .29, W2.skill_id: .28}, incomplete={(W2.skill_id, 'a40')}), block='c_a')
        s4 = dsk.select(options, ['a30'], synth({dsk.NO_SKILL: .30, W1.skill_id: .29, W2.skill_id: .28}, incomplete={(dsk.NO_SKILL, 'a30')}), block='c_a')
        f1, f2, f4 = (dsk.freeze('D01', s, options, evidence_scope={'smoke': True}) for s in (s1, s2, s4))
        f1_before = copy.deepcopy(f1)
        add = dsk.principles_addendum(f1, 'Treat single-job differences as development evidence only.', evidence_refs=[refs_legal[0]], legal_evidence_refs=refs_legal)
        frozen_path = td / 'frozen_skill.json'
        write_once(frozen_path, f1)
        check('select_freeze_paths', s1['status'] == 'SELECTED' and s1['selected'] == W2.skill_id and s1['synthetic'] and f1['status'] == 'SMOKE_FIXTURE' and
              f1['skill']['status'] == 'SMOKE_FIXTURE' and s2['status'] == 'NO_EFFECTIVE_CANDIDATE' and f2['status'] == 'NOT_PROMOTED' and f2['skill'] is None and
              s3['selected'] == W1.skill_id and W2.skill_id not in s3['eligible_scores'] and s4['status'] == 'SELECTION_INCOMPLETE' and f4['status'] == 'NOT_PROMOTED' and
              add['status'] == 'PRINCIPLES_ADDENDUM_CANDIDATE' and add['skill']['revision'] == 2 and f1 == f1_before and
              dsk.routable([dsk.skill_from_json(add['skill'])]) == [] and refused(lambda: write_once(frozen_path, f1), RuntimeError, 'overwrite') and
              refused(lambda: dsk.select(options, ['a30'], synth({}), block='e'), ValueError) and ('NO_SKILL', []) in seen_k and
              (W2.skill_id, [W2.rendered_body]) in seen_k, statuses=[s1['status'], s2['status'], s3['status'], s4['status']])
        # the live selection runner binds the domain source, commits every option first, then opens only the selection block
        cap, order_log = {'branch': [], 'stages': []}, []

        def fake_branch(path, job, h, led, client, shared, **kw):
            cap['branch'].append(kw)
            order_log.append(('run', Path(path).name))
            jd = Path(path) / job / 'cells'
            jd.mkdir(parents=True)
            for s in rt.SEEDS:
                context.write_json(jd / ('%s__p1__s%d.json' % (job, s)), {'status': 'OK', 'material_id': 'p1', 'model_seed': s, 'scores': {'c_a': {'normalized_mse_macro': .4}, 'c_b': {'normalized_mse_macro': .41 if 'W1' in Path(path).name else .42}}})
            return br.RunResult('COMPLETE', job, 0, 'p1', 'm', 1, 1, 1)

        def fake_stage(*a, **k):
            cap['stages'].append((a[2], k))
            order_log.append(('score', Path(a[0]).name))
        with patch.object(base, 'branch', side_effect=fake_branch), patch.object(base, 'stage', side_effect=fake_stage):
            run, score = selection_runner(td / 'sel', led0, None, domain_id='D02', shared_by_job={'a30_D02': td}, block='c_b', seeds=rt.SEEDS,
                                          limits={'max_calls': 16, 'max_tools': 24, 'max_new_evaluations': 2}, max_tool_corrections=2, evidence_roundtrip=False)
            s_live = dsk.select({dsk.NO_SKILL: None, W1.skill_id: W1}, ['a30_D02'], run, block='c_b', score=score)
        check('selection_runner_binding', all(k['dataset'] == 'traffic' for k in cap['branch']) and cap['stages'] == [('c_b', {'roster': None, 'dataset': 'traffic'})] * 2 and
              [x[0] for x in order_log] == ['run', 'run', 'score', 'score'] and s_live['selected'] == W1.skill_id and s_live['runs'][W1.skill_id]['a30_D02']['loss_by_seed'] == [.41] * 3,
              order=order_log)

        # ---- 6. zero budget refuses network / fit / labels; dataset reaches every stage and resume
        cfg_a = root / 'live_config_A_zero_budget.json'
        for argv, why in ((['--run', '--study', 'domain_skill', '--study-config', str(cfg_a), '--output', str(td / 'live_A')], 'LIVE_REFUSED'),
                          (['--run', '--study', 'domain_skill', '--study-config', str(root / 'live_config_B_draft.json'), '--output', str(td / 'live_B')], 'DRAFT')):
            with patch.object(sys, 'argv', ['roundtrip'] + argv):
                check('live_entry_refuses_%s' % why, refused(rtp.main, PermissionError, why) and not (td / 'live_A').exists() and not (td / 'live_B').exists() and 'openai' not in sys.modules)
        zero_frozen = {**context.read_json(cfg_a), 'status': 'FROZEN', 'stages_authorized': list(STAGES)}
        check('zero_caps_refused_even_if_frozen', refused(lambda: preflight(load_study_config(zero_frozen)), PermissionError, 'zero budget'))
        mc = object.__new__(rt.MeteredClient)
        mc.ledger, mc.out, mc.fatal, mc.http_cap, mc.last_text = led0, td / 'raw', False, 0, None
        mc.out.mkdir()
        mc.api = None                                   # any send would raise AttributeError, not BudgetExhausted
        check('zero_token_cap_refuses_before_send', refused(lambda: mc.call('router', 'a50_D02', {'x': 1}, 'system'), budget.BudgetExhausted) and not list(mc.out.iterdir()) and led0.s['llm_requests'] == 0)
        metered_written = len(list(mc.out.iterdir()))
        n_loads = len(loads)
        check('labels_refuse_without_prerequisites', refused(lambda: commit.open_c_b(td / 'nolabels', 'traffic', 'a50'), PermissionError, 'before loading') and
              refused(lambda: commit.freeze_e(td / 'nolabels', 'traffic', 'a50'), PermissionError, 'before loading') and
              refused(lambda: commit.score_e(td / 'nolabels', 'traffic', 'a50'), PermissionError, 'before loading') and len(loads) == n_loads)
        calls = []

        class _Led:
            def remaining(self):
                return 5
        with patch.object(base.subprocess, 'run', side_effect=lambda cmd, **kw: calls.append(cmd) or type('R', (), {'returncode': 0})()):
            base.stage(Path('b'), 'a50_D02', 'c_b', _Led(), dataset='traffic')
            base.stage(Path('b'), 'a75', 'score_e', _Led())
        got = []
        with patch.object(commit, 'open_c_b', side_effect=lambda *a, **k: got.append(('c_b', a[1], a[2]))), patch.object(commit, 'freeze_e', side_effect=lambda *a, **k: got.append(('freeze_e', a[1], a[2]))), \
                patch.object(commit, 'score_e', side_effect=lambda *a, **k: got.append(('score_e', a[1], a[2]))):
            for st in ('c_b', 'freeze_e', 'score_e'):
                with patch.object(sys, 'argv', ['v1', '--stage', st, '--job', 'a50_D02', '--output', str(td / 'o'), '--dataset', 'traffic']):
                    base.main()
        check('stage_cli_dataset_binding', calls[0][calls[0].index('--dataset') + 1] == 'traffic' and calls[1][calls[1].index('--dataset') + 1] == 'electricity' and
              got == [('c_b', 'traffic', 'a50_D02'), ('freeze_e', 'traffic', 'a50_D02'), ('score_e', 'traffic', 'a50_D02')])
        # roundtrip study hooks with a SMOKE_CONTROL_FLOW configuration (scripted router, patched branch/stage; real T for routing)
        ctl = {**context.read_json(cfg_a), 'status': 'SMOKE_CONTROL_FLOW', 'allow_fixture_skills': True, 'stages_authorized': list(STAGES), 'output': str(td / 'ctl'),
               'caps': {'max_fit_attempts': 60, 'max_llm_requests': 40, 'max_llm_tokens': 10 ** 6, 'max_wall_s': 3600, 'max_retries': 2}}
        ctl_path = td / 'ctl_config.json'
        context.write_json(ctl_path, ctl)
        rtp.configure_study('domain_skill', ctl_path)
        cfgd = rtp.configuration()
        check('roundtrip_configuration_datasets', cfgd['jobs']['a50_D01']['dataset'] == 'electricity' and cfgd['jobs']['a50_D02']['dataset'] == 'traffic' and
              cfgd['jobs']['a50_D02']['t'] == spec.t_of('traffic', .5) and rtp.study_dataset('a50_D02') == 'traffic' and rtp.study_dataset('a75') == 'electricity' and
              cfgd['planned_fits'] == PLANNED_FITS == 2 * (6 + 3 * 2 * 3) and cfgd['max_tool_corrections'] == 2 and refused(lambda: preflight(CFG), PermissionError, 'SMOKE_FIXTURE'))
        run_root = td / 'ctl'
        run_root.mkdir()
        ledc = rt.RuntimeLedger(run_root / 'budget.json', **ctl['caps'])
        router = _Scripted(outs=[{'selected_skill_id': 'SK-FIX-D01', 'status': 'SELECTED', 'evidence_fields': ['lag168_corr'], 'rationale': 'smoke'},
                                 {'selected_skill_id': None, 'status': 'ABSTAIN', 'evidence_fields': [], 'rationale': 'smoke'}])
        commons, pools = prepare(run_root, ledc, ctl['caps'], client=router)
        check('prepare_routes_frozen_before_fits', ledc.s['fit_attempts'] == 0 and (run_root / FROZEN_MARKER).exists() and [c['role'] for c in router.seen] == ['router', 'router'] and
              pools['a50_D01']['routes']['profile_match']['status'] == 'SELECTED' and pools['a50_D02']['routes']['profile_match']['status'] == 'ABSTAIN' and
              pools['a50_D02']['routes']['known_domain']['selected_skill_id'] == 'SK-FIX-D02' and commons['a50_D02'].ctx.job.dataset == 'traffic' and
              refused(lambda: route_job(run_root, 'a50_D02', commons['a50_D02'], _Scripted(), catalog_root=root), RuntimeError, 'no re-route') and
              load_pools(run_root)['a50_D01']['routes'] == pools['a50_D01']['routes'])
        got_b, got_r = [], []
        with patch.object(base, 'branch', side_effect=lambda path, job, h, *a, **k: got_b.append((job, [e.body for e in h.entries], k['dataset'], k['limits'])) or br.RunResult('COMPLETE', job, 0, 'None', 'm', 1, 1, 0)), \
                patch.object(base, 'resume_branch', side_effect=lambda path, job, h, *a, **k: got_r.append((job, [e.body for e in h.entries], k['dataset'])) or br.RunResult('COMPLETE', job, 0, 'None', 'm', 1, 1, 0)):
            for job in JOBS:
                for arm in ORDER[job]:
                    p = run_root / ('%s_%s' % (job, arm))
                    p.mkdir()
                    run_arm(arm, p, job, ledc, router, run_root / (job + '_common'), pools[job], seeds=SEEDS, roster=None, max_tool_corrections=2)
                    resume_arm(arm, p, job, ledc, router, load_pools(run_root)[job], seeds=SEEDS, roster=None, max_tool_corrections=2)
        body = {s: fixtures[s].rendered_body for s in fixtures}
        expect = [('a50_D01', [], 'electricity'), ('a50_D01', [body['SK-FIX-D01']], 'electricity'), ('a50_D01', [body['SK-FIX-D01']], 'electricity'),
                  ('a50_D02', [], 'traffic'), ('a50_D02', [body['SK-FIX-D02']], 'traffic'), ('a50_D02', [], 'traffic')]
        check('run_and_resume_arm_binding', [(j, b, ds) for j, b, ds, _ in got_b] == expect and got_r == expect and all(l == LIMITS for *_, l in got_b) and len(router.seen) == 2)
        branches = [(run_root / 'x1', 'a50_D01'), (run_root / 'x2', 'a50_D02')]
        for p, j in branches:
            (p / j).mkdir(parents=True)
            (p / j / 'commit.json').write_text('{}', encoding='utf-8')
        stages_seen = []
        with patch.object(base, 'stage', side_effect=lambda p, j, name, led, roster=None, dataset='electricity': stages_seen.append((j, name, dataset))):
            rtp.open_labels(run_root, branches, [], ledc)
            CFG['stages_authorized'] = ['material', 'llm', 'fit']
            label_refused = refused(lambda: rtp.open_labels(run_root, branches, [], ledc), PermissionError, 'labels')
        check('labels_stage_dataset_and_authorization', stages_seen == [('a50_D01', 'c_b', 'electricity'), ('a50_D02', 'c_b', 'traffic'), ('a50_D01', 'freeze_e', 'electricity'),
                                                                        ('a50_D02', 'freeze_e', 'traffic'), ('a50_D01', 'score_e', 'electricity'), ('a50_D02', 'score_e', 'traffic')] and label_refused)
        check('no_fit_or_api_or_label_read', not launches and led0.s['fit_attempts'] == 0 and ledc.s['fit_attempts'] == 0 and 'openai' not in sys.modules and
              all(b - a == spec.TRAIN_SPAN and b in {spec.t_of(ds, x) for x in (.3, .4, .5, .55)} for ds, a, b in loads), t_reads=sorted(set(loads)))

    result = {'status': 'PASS', 'package': 'DEV-DOMAIN-SKILL-V1-BUILD', 'seconds': round(time.time() - started, 1), 'checks': checks,
              'resources': {'consumer_fits': led0.s['fit_attempts'] + ledc.s['fit_attempts'], 'fit_subprocess_launches': len(launches),
                            'api_calls': 0 if 'openai' not in sys.modules else 'UNKNOWN_OPENAI_IMPORTED', 'openai_module_imported': 'openai' in sys.modules,
                            'metered_requests_written': metered_written, 'new_label_reads': 0 if checks['no_fit_or_api_or_label_read']['ok'] else 'CHECK_FAILED', 'new_sha': 0,
                            't_slices_loaded': [{'dataset_runtime_only': ds, 'rows': [a, b]} for ds, a, b in sorted(set(loads))],
                            'note': 'every loaded slice is a material-stage T [t-672, t); label stages were refused or patched'},
              'synthetic': 'baseline C_A in Fast requests, Slow/router responses, selection losses and fake cells are SCRIPTED/SYNTHETIC control-flow inputs; no utility claim'}
    return result


def _fixture_census_tree(trepo: Path, root: Path) -> dict:
    """A tiny SYNTHETIC completed source branch (real a30 T overview, scripted trace, synthetic C_A/C_B) for census wiring."""
    jdir = trepo / '_scratch' / 'fixture_pkg' / 'a30_fast' / 'a30'
    (jdir / 'cells').mkdir(parents=True)
    (jdir / 'materials').mkdir()
    shutil.copy(root / 't_runtime' / 'D01' / 'a30' / 'overview.json', jdir / 'overview.json')
    score = lambda v: {'normalized_mse_macro': v, 'per_entity_normalized_mse': [v] * 32, 'per_origin_normalized_mse_mean': [v, v]}
    for i, mid in enumerate(('None', 'FixedMixup', 'p1')):
        for s in rt.SEEDS:
            context.write_json(jdir / 'cells' / ('a30__%s__s%d.json' % (mid, s)), {'cell_id': 'a30__%s__s%d' % (mid, s), 'status': 'OK', 'job_id': 'a30', 'dataset': 'electricity',
                                                                                'material_id': mid, 'model_seed': s, 'scores': {'c_a': score(.5 + .01 * i), 'c_b': score(.6 + .01 * i)}, 'synthetic': True})
    pol = policy.uniform_policy([{'op': 'freqmask', 'mu': 0.1}], 'smoke hypothesis')
    context.write_json(jdir / 'materials' / 'index.json', {'p1': {'material_id': 'p1', 'job_id': 'a30', 'path': 'x', 'key': 'k', 'assignment': [], 'n_identity': 0, 'alias_of': None, 'steps_log_path': 'x'}})
    context.write_json(jdir / 'materials' / 'p1__compiled.json', {'policy': pol})
    context.write_json(jdir / 'commit.json', {'material_id': 'p1', 'reason': 'smoke', 'fitted_materials_at_commit': ['FixedMixup', 'None', 'p1']})
    context.write_json(jdir / 'c_b_scores.json', {'synthetic': True})
    b = jdir.parent
    rows = [{'event_id': 'a30:0', 'event': 'job_started', 'mode': 'held_in', 'knowledge_version': 0, 'baseline_ids': ['None', 'FixedMixup']},
            {'event_id': 'a30:1', 'event': 'fast_request', 'number': 1, 'guidance': {'loaded': [], 'applicability': []}},
            {'event_id': 'a30:2', 'event': 'fast_response', 'number': 1, 'response': {'actions': [{'tool': 'build_material', 'arguments': {}}]}},
            {'event_id': 'a30:3', 'event': 'tool_started', 'tool': 'build_material', 'arguments': {'plan_id': 'p1', 'policy': pol}},
            {'event_id': 'a30:4', 'event': 'tool_completed', 'tool': 'build_material', 'output': {'plan_id': 'p1', 'material_spec': {'assignment': [[{'op': 'freqmask', 'mu': 0.1}]] * 32, 'rule_index': [-1] * 32}}},
            {'event_id': 'a30:5', 'event': 'tool_started', 'tool': 'commit', 'arguments': {'plan_id': 'p1', 'reason': 'smoke'}},
            {'event_id': 'a30:6', 'event': 'tool_completed', 'tool': 'commit', 'output': {'plan_id': 'p1', 'reason': 'smoke'}},
            {'event_id': 'a30:7', 'event': 'committed', 'plan_id': 'p1'}]
    (b / 'trace.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    context.write_json(b / 'branch_result.json', {'status': 'COMPLETE', 'job_id': 'a30', 'calls': 1, 'tool_calls': 2, 'new_evaluations': 1, 'committed_plan_id': 'p1'})
    context.write_json(b / 'knowledge.json', {'version': 0, 'entries': []})
    return {'kind': 'SYNTHETIC_SOURCE_BRANCH', 'events': len(rows), 'cells': 9}


def smoke_pilot() -> dict:
    """Integration smoke of the EVOLVE-PILOT differences only (0 fits, 0 API, 0 label reads; real T for preflight and materials,
    fake fits and synthetic cells for control flow): frozen constants, cumulative stage caps without transfer or reset, stage
    configs and preflight, Generic CONTROL, NO_TREATMENT, candidate arms, Random with dataset and frozen policy seeds, C_B-only
    labels, the select reader + freeze, target catalog admission, same-configuration fit retry, ledger/http hooks."""
    import subprocess as _subprocess
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from methods.ttha.batch_base import commit, data
    from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp
    checks, launches, loads = {}, [], []
    started = time.time()

    def check(name, ok, **detail):
        checks[name] = {'ok': bool(ok), **detail}
        if not ok:
            raise AssertionError('%s: %s' % (name, detail))

    def refused(fn, exc_types, contains=''):
        try:
            fn()
        except exc_types as exc:
            return contains in str(exc)
        return False
    real_load = data.load_slice

    def logged_load(dataset, a, b, roster=None):
        loads.append((dataset, a, b))
        return real_load(dataset, a, b, roster=roster)

    def no_launch(*a, **k):
        launches.append(a[0] if a else k.get('args'))
        raise AssertionError('a subprocess/fit/stage launch was attempted inside the smoke')

    def fake_fit(ctx, ref, seeds, ledger, repo, feedback=True):
        out = []
        v = {'None': .50, 'FixedMixup': .48}.get(ref.material_id, .49)
        for s in seeds:
            cid = '%s__%s__s%d' % (ctx.job.job_id, ref.material_id, s)
            rec = {'cell_id': cid, 'status': 'OK', 'job_id': ctx.job.job_id, 'dataset': ctx.job.dataset, 'material_id': ref.material_id, 'model_seed': s,
                   'roster': ctx.job.roster, 'rows_read': list(ctx.job.evaluate_rows), 'model_path': 'SYNTHETIC_NO_MODEL', 'synthetic': True,
                   'scores': {'c_a': {'per_origin_entity_normalized_mse': [[v] * 32, [v] * 32], 'per_origin_normalized_mse_mean': [v, v], 'normalized_mse_macro': v, 'n_nonfinite_predictions': 0}}}
            (ctx.job_dir / 'cells').mkdir(exist_ok=True)
            context.write_json(ctx.job_dir / 'cells' / (cid + '.json'), rec)
            out.append(rec)
        return out

    with TemporaryDirectory(ignore_cleanup_errors=True) as tdn, patch.object(data, 'load_slice', logged_load), patch.object(_subprocess, 'run', no_launch), patch.object(_subprocess, 'Popen', no_launch):
        td = Path(tdn)
        # 1. frozen constants
        chain = all(spec.job_spec(DOMAIN_BINDINGS[d], .75).train_range[0] >= PILOT_SOURCE_END[d] and
                    max(spec.job_spec(DOMAIN_BINDINGS[d], .75).e) + spec.H <= spec.job_spec(DOMAIN_BINDINGS[d], .85).train_range[0] and
                    max(spec.job_spec(DOMAIN_BINDINGS[d], .85).e) + spec.H <= spec.job_spec(DOMAIN_BINDINGS[d], .95).train_range[0] for d in DOMAIN_BINDINGS)
        check('pilot_constants', all(spec.job_spec(DOMAIN_BINDINGS[d], int(a[1:]) / 100).t == t for d in PILOT_T for a, t in PILOT_T[d].items()) and chain and
              all(PILOT_T[d]['a65'] + 384 == PILOT_SOURCE_END[d] for d in PILOT_T) and
              sum(x['fits'] for x in PILOT_ALLOC.values()) == 192 == PILOT_TOTAL['max_fit_attempts'] - PILOT_TOTAL['max_retries'] and
              sum(x['requests'] for x in PILOT_ALLOC.values()) == PILOT_TOTAL['max_llm_requests'] == 324 and sum(x['tokens'] for x in PILOT_ALLOC.values()) == PILOT_TOTAL['max_llm_tokens'] and
              PILOT_HTTP_CAP == 648 and 2 * (6 + 6) == PILOT_ALLOC['source']['fits'] and 2 * (6 + 3 * 6) == PILOT_ALLOC['formation']['fits'] and 4 * (6 + 3 * 6 + 6) == PILOT_ALLOC['target']['fits'] and
              all(sorted(o) == sorted(('no_skill', 'generic', 'known_domain', 'random')) for o in TARGET_ORDER.values()) and
              len({x for v in TARGET_RANDOM_SEEDS.values() for x in v}) == 8 and PILOT_SEEDS == rt.SEEDS)
        # 2. real preflight (T rows only) in a temp root, then cumulative stage caps
        root = td / 'pilot'
        root.mkdir()
        cfg0 = pilot_preflight(root)
        P = pilot_paths(root)
        check('preflight_real', all(v['eligible'] for v in cfg0['geometry'].values()) and len(cfg0['geometry']) == 10 and [h['branch'] for h in cfg0['D01_history']] == ['a55_H0', 'a65_parent'] and
              context.read_json(P['pre_catalog'] / 'catalog' / 'domain_catalog.json')['generic_control']['body'] == sp.GENERIC_TEXT and refused(lambda: pilot_preflight(root), RuntimeError, 'overwrite'))
        src = stage_caps(root, 'source')
        led = rt.RuntimeLedger(P['ledger'], **PILOT_TOTAL)
        epoch0 = led.s['started_epoch']
        led.s.update({'fit_attempts': 10, 'llm_requests': 5, 'llm_tokens_in': 90_000, 'llm_tokens_out': 10_000, 'llm_http_attempts': 6})
        led._save()
        form = stage_caps(root, 'formation')
        again = stage_caps(root, 'source')
        led.s.update({'fit_attempts': 190, 'retries_used': 1})
        led._save()
        tgt = stage_caps(root, 'target')
        check('stage_caps_no_transfer_no_reset', src['caps']['max_fit_attempts'] == 26 and src['caps']['max_llm_requests'] == 32 and src['caps']['max_llm_tokens'] == 800_000 and src['http_cap'] == 64 and
              form['caps']['max_fit_attempts'] == 60 and form['caps']['max_llm_requests'] == 105 and form['caps']['max_llm_tokens'] == 2_100_000 and form['http_cap'] == 206 and
              again == src and tgt['caps']['max_fit_attempts'] == 194 and context.read_json(P['ledger'])['started_epoch'] == epoch0 and
              all(c['caps']['max_wall_s'] == 28800 for c in (src, form, tgt)), source=src['caps'], formation=form['caps'], target=tgt['caps'])
        # 3. stage configs + preflight: Source/Select stop at C_B, Target also opens E
        led.s.update({'fit_attempts': 0, 'retries_used': 0})
        led._save()
        scfg = context.read_json(pilot_stage_config(root, 'source', jobs=list(D02_SOURCE_JOBS), order={j: ['no_skill'] for j in D02_SOURCE_JOBS}, catalog_root=P['pre_catalog'], labels_e=False))
        preflight(load_study_config(scfg))
        bad = {**scfg, 'stages_authorized': ['material', 'llm', 'fit']}
        check('stage_config_and_preflight', scfg['stages_authorized'] == ['material', 'llm', 'fit', 'labels_c_b'] and scfg['ledger_path'] == str(P['ledger']) and scfg['fit_retry'] and
              scfg['caps'] == src['caps'] and scfg['http_cap'] == 64 and scfg['evidence_roundtrip'] and scfg['limits'] == PILOT_LIMITS and scfg['max_tool_corrections'] == 2 and
              refused(lambda: preflight(load_study_config(bad)), PermissionError, 'labels_c_b') and
              refused(lambda: pilot_stage_config(root, 'source', jobs=['a55_D02'], order={'a55_D02': ['no_skill']}, catalog_root=P['pre_catalog'], labels_e=False) and write_once(P['configs'] / 'source.json', {}), RuntimeError, 'overwrite'))
        # 4. roundtrip hooks: package ledger path, http cap, and restoring another study's defaults
        rtp.configure_study('domain_skill', P['configs'] / 'source.json')
        hooks = (rtp.study_ledger_path(td / 'x') == P['ledger'], rtp.study_http_cap() == 64, rtp.study_dataset('a55_D02') == 'traffic', rt.FIT_RETRY is True)
        rtp.configure_study('measured_start')
        hooks += (rtp.study_ledger_path(td / 'x') == td / 'x' / 'budget.json', rtp.study_http_cap() == 2 * rtp.CAPS['max_llm_requests'])
        check('runner_ledger_and_http_hooks', all(hooks), hooks=hooks)
        # 5. Generic CONTROL / NO_TREATMENT / candidate arms (target-like config on the pre catalog)
        W = [dsk.make_skill(skill_id='D02-W%d-r1' % k, domain_id='D02', revision=1, workflow='Smoke candidate workflow %d: compare with the baselines before committing.' % k,
                            principles=None if k == 1 else 'Seed uncertainty only.', applicability_summary='smoke', compatibility_note='smoke', observable_applicability={'const': True},
                            evidence_refs=['r:1'], legal_evidence_refs={'r:1'}, source_stage='propose', status='CANDIDATE_TEST_ONLY') for k in (1, 2)]
        prop_path = td / 'propose.json'
        context.write_json(prop_path, {'status': 'PROPOSED', 'skills': [w.to_json() for w in W]})
        tcfg = {**scfg, 'stage': 'smoke_arms', 'output': str(td / 'arms'), 'jobs': [{'job_id': 'a85_D02', 'domain_id': 'D02', 'roster': None}, {'job_id': 'a75_D02', 'domain_id': 'D02', 'roster': None}],
                'order': {'a85_D02': list(TARGET_ORDER['a85_D02']), 'a75_D02': ['no_skill', 'cand_W1', 'cand_W2']}, 'stages_authorized': list(STAGES),
                'random_policy_seeds': {'a85_D02': TARGET_RANDOM_SEEDS['a85_D02']}, 'candidates': {'a75_D02': str(prop_path)}}
        configure(load_study_config(tcfg))
        gc = context.read_json(P['pre_catalog'] / 'catalog' / 'domain_catalog.json')['generic_control']
        pool = {'routes': {'generic': {'route_mode': 'generic', 'status': 'GENERIC_CONTROL'}, 'known_domain': dsk.route_known_domain('D02', [], {}),
                           'cand_W1': {'status': 'CANDIDATE_TEST_ONLY'}, 'cand_W2': {'status': 'CANDIDATE_TEST_ONLY'}, 'no_skill': {'status': 'NO_SKILL'},
                           'random': {'route_mode': 'random', 'status': 'ZERO_LLM_CONTROL_NOT_ROUTED'}}}
        _, hg, _ = arm_knowledge('a85_D02', 'generic', pool)
        _, hw2, tw2 = arm_knowledge('a75_D02', 'cand_W2', pool)
        _, h0, _ = arm_knowledge('a85_D02', 'no_skill', pool)
        with patch.object(base, 'branch', side_effect=AssertionError('NO_TREATMENT must not run Fast')):
            nt = run_arm('known_domain', td / 'arms' / 'a85_D02_known_domain', 'a85_D02', led, None, td, pool, seeds=PILOT_SEEDS, roster=None, max_tool_corrections=2)
        bad_prop = td / 'bad_propose.json'
        context.write_json(bad_prop, {'status': 'PROPOSED', 'skills': [{**W[0].to_json(), 'status': 'FROZEN_SELECTED'}]})
        CFG['candidates']['a75_D02'] = str(bad_prop)
        wrong_status = refused(lambda: arm_knowledge('a75_D02', 'cand_W1', pool), PermissionError, 'CANDIDATE_TEST_ONLY')
        CFG['candidates']['a75_D02'] = str(prop_path)
        check('generic_control_no_treatment_candidates', hg == sp.br.Knowledge(1, (sp.br.Guidance('experiment_guidance', sp.GENERIC_TEXT, {'const': True}, ('generic:frozen',)),)) and
              gc['status'] == 'CONTROL' and hw2 == dsk.skill_knowledge(W[1]) and hw2.version == 1 == hg.version and tw2['status'] == 'CANDIDATE_TEST_ONLY' and h0 == br.Knowledge() and
              nt.status == 'INCOMPLETE' and nt.failure_kind == 'NO_TREATMENT' and (td / 'arms' / 'a85_D02_known_domain' / 'trace.jsonl').exists() and
              (td / 'arms' / 'a85_D02_known_domain' / 'treatment.json').exists() and wrong_status and dsk.routable(W) == [])
        # 6. Random: real traffic T and materials, fake fits; dataset, frozen policy seeds, commit binding
        shared = td / 'arms' / 'a85_D02_common'
        committed = []
        real_commit = commit.commit
        with patch.object(rt, 'fit', side_effect=fake_fit), patch.object(commit, 'commit', side_effect=lambda *a, **k: committed.append(a[1]) or real_commit(*a, **k)):
            rt.Adapter(shared, 'a85_D02', led, REPO, dataset='traffic', seeds=PILOT_SEEDS).baselines()
            rr = run_arm('random', td / 'arms' / 'a85_D02_random', 'a85_D02', led, None, shared, pool, seeds=PILOT_SEEDS, roster=None, max_tool_corrections=2)
        rows = [json.loads(x) for x in (td / 'arms' / 'a85_D02_random' / 'trace.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        mats = [r for r in rows if r['event'] == 'random_material']
        cells = [context.read_json(c) for c in (td / 'arms' / 'a85_D02_random' / 'a85_D02' / 'cells').glob('*.json')]
        check('random_arm_dataset_and_seeds', rr.status == 'COMPLETE' and committed == ['traffic'] and [m['sampling_seed'] for m in mats] == TARGET_RANDOM_SEEDS['a85_D02'] and
              all(m['material_spec']['policy'] == policy.validate_policy(policy.random_policy(m['sampling_seed'])) for m in mats) and
              {c['dataset'] for c in cells} == {'traffic'} and context.read_json(td / 'arms' / 'a85_D02_random' / 'a85_D02' / 'commit.json')['material_id'] == rr.committed_plan_id and
              (td / 'arms' / 'a85_D02_random' / 'route_used.json').exists() and led.s['fit_attempts'] == 0, committed_plan=rr.committed_plan_id)
        # 7. same-configuration fit retry (patched worker launch; ledger-charged; capped)
        ad = rt.Adapter(td / 'retry', 'a85_D02', led, REPO, dataset='traffic', seeds=PILOT_SEEDS)
        ad.build_material({'plan_id': 'q1', 'policy': policy.uniform_policy([{'op': 'freqmask', 'mu': 0.1}], 'smoke')}, remaining_seconds=600.)
        rled = rt.RuntimeLedger(td / 'retry_budget.json', max_fit_attempts=10, max_llm_requests=1, max_llm_tokens=1, max_wall_s=3600, max_retries=2)
        launched = []

        def flaky(cmd, **kw):
            launched.append(cmd)
            if len(launched) % 2 == 1:
                return type('R', (), {'returncode': 1})()
            seed = int(cmd[cmd.index('--seed') + 1])
            fake_fit(ad.ctx, ad.refs['q1'], (seed,), rled, REPO)
            return type('R', (), {'returncode': 0})()
        with patch.object(rt.subprocess, 'run', side_effect=flaky):
            rt.FIT_RETRY = True
            ok_rec = rt.fit(ad.ctx, ad.refs['q1'], (PILOT_SEEDS[0],), rled, REPO)
            rt.FIT_RETRY = False
            no_retry = refused(lambda: rt.fit(ad.ctx, ad.refs['q1'], (PILOT_SEEDS[1],), rled, REPO), RuntimeError, 'worker_failed')
        check('fit_retry_same_configuration', ok_rec[0]['model_seed'] == PILOT_SEEDS[0] and rled.s['retries_used'] == 1 and rled.s['fit_attempts'] == 3 and
              sum(e.get('kind') == 'fit_retry' for e in rled.s['events']) == 1 and no_retry and all(c[c.index('--seed') + 1] in {str(s) for s in PILOT_SEEDS} for c in launched), attempts=rled.s['fit_attempts'])
        # 8. labels: Source/Select open C_B only; Target opens C_B then E
        branches = [(td / 'lab' / 'b1', 'a55_D02'), (td / 'lab' / 'b2', 'a65_D02')]
        for p, j in branches:
            (p / j).mkdir(parents=True)
            (p / j / 'commit.json').write_text('{}', encoding='utf-8')
        seen = []
        rtp.configure_study('domain_skill', P['configs'] / 'source.json')
        with patch.object(base, 'stage', side_effect=lambda p, j, name, led, roster=None, dataset='electricity': seen.append((j, name, dataset))):
            rtp.open_labels(td / 'lab', branches, [], led)
            c_b_only = list(seen)
            full_cfg = td / 'full.json'
            context.write_json(full_cfg, {**scfg, 'stages_authorized': list(STAGES)})
            rtp.configure_study('domain_skill', full_cfg)
            seen.clear()
            (td / 'lab2').mkdir()
            rtp.open_labels(td / 'lab2', branches, [], led)
        check('labels_c_b_only_for_formation_stages', c_b_only == [('a55_D02', 'c_b', 'traffic'), ('a65_D02', 'c_b', 'traffic')] and (td / 'lab' / 'labels_c_b_only.json').exists() and
              not (td / 'lab' / 'all_e_predictions_frozen.json').exists() and [x[1] for x in seen] == ['c_b', 'c_b', 'freeze_e', 'freeze_e', 'score_e', 'score_e'])
        # 9. select reader + freeze, knowledge binding, target catalog admission
        froot = td / 'fz'
        FP = pilot_paths(froot)
        (FP['formation'] / 'D01').mkdir(parents=True)
        context.write_json(FP['formation'] / 'D01' / 'propose.json', {'status': 'KEEP', 'skills': [], 'rationale': 'smoke keep'})
        (FP['formation'] / 'D02').mkdir(parents=True)
        context.write_json(FP['formation'] / 'D02' / 'propose.json', {'status': 'PROPOSED', 'skills': [w.to_json() for w in W]})
        cb = {'no_skill': .50, 'cand_W1': .48, 'cand_W2': .49}
        for arm, knowledge in (('no_skill', br.Knowledge()), ('cand_W1', dsk.skill_knowledge(W[0])), ('cand_W2', dsk.skill_knowledge(W[1]))):
            b = FP['select'] / ('a75_D02_%s' % arm)
            (b / 'a75_D02' / 'cells').mkdir(parents=True)
            context.write_json(b / 'knowledge.json', asdict(knowledge))
            context.write_json(b / 'branch_result.json', asdict(br.RunResult('COMPLETE', 'a75_D02', knowledge.version, 'p1', 'm', 2, 3, 1)))
            context.write_json(b / 'a75_D02' / 'c_b_scores.json', {'synthetic': True})
            for s in PILOT_SEEDS:
                context.write_json(b / 'a75_D02' / 'cells' / ('a75_D02__p1__s%d.json' % s), {'cell_id': 'a75_D02__p1__s%d' % s, 'status': 'OK', 'material_id': 'p1', 'model_seed': s,
                                                                                            'scores': {'c_a': {'normalized_mse_macro': .4}, 'c_b': {'normalized_mse_macro': cb[arm]}}})
        pilot_freeze(froot)
        f1, f2 = (context.read_json(FP['formation'] / d / 'frozen_skill.json') for d in ('D01', 'D02'))
        sel2 = context.read_json(FP['formation'] / 'D02' / 'selection.json')
        froot2 = td / 'fz2'
        shutil.copytree(froot, froot2)
        for d in ('D01', 'D02'):
            for n in ('frozen_skill.json', 'selection.json'):
                if (froot2 / 'formation' / d / n).exists():
                    (froot2 / 'formation' / d / n).unlink()
        context.write_json(froot2 / 'formation' / 'select' / 'a75_D02_cand_W2' / 'knowledge.json', asdict(br.Knowledge()))
        mismatch = refused(lambda: pilot_freeze(froot2), RuntimeError, 'other knowledge')
        froot3 = td / 'fz3'
        (froot3 / 'formation' / 'D01').mkdir(parents=True)
        context.write_json(froot3 / 'formation' / 'D01' / 'frozen_skill.json', {'status': 'FROZEN_SELECTED', 'skill': W[0].to_json()})
        admission = refused(lambda: pilot_target_catalog(froot3), PermissionError, 'FROZEN_SELECTED')
        check('select_reader_freeze_and_catalog_admission', f1['status'] == 'NO_CANDIDATE' and f2['status'] == 'FROZEN_SELECTED' and f2['skill']['skill_id'] == 'D02-W1-r1' and
              sel2['runs']['NO_SKILL']['a75_D02']['loss_by_seed'] == [.50] * 3 and sel2['selected'] == 'D02-W1-r1' and not sel2['synthetic'] is None and mismatch and admission,
              note='synthetic C_B in a temp dir; the rule, not a result')
        # 10. every T read was a material-stage slice [t-672, t)
        check('no_fit_api_or_label_read', not launches and 'openai' not in sys.modules and led.s['llm_requests'] == 5 and
              all(b - a == spec.TRAIN_SPAN for ds, a, b in loads), t_reads=sorted(set(loads)))
    return {'status': 'PASS', 'package': 'DEV-DOMAIN-SKILL-V1-EVOLVE-PILOT', 'seconds': round(time.time() - started, 1), 'checks': checks,
            'resources': {'consumer_fits': 0, 'fit_subprocess_launches': len(launches), 'api_calls': 0 if 'openai' not in sys.modules else 'UNKNOWN', 'new_label_reads': 0, 'new_sha': 0},
            'synthetic': 'fake fits, synthetic C_A/C_B cells and a simulated ledger in temp dirs; control flow only'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-profiles', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-out', type=Path, default=None, help='where smoke.json and routing/ are written (default: --root)')
    ap.add_argument('--smoke-pilot', action='store_true')
    ap.add_argument('--pilot', choices=['run', 'result'])
    ap.add_argument('--formation', choices=['census', 'propose', 'select', 'freeze'])
    ap.add_argument('--formation-config', type=Path)
    ap.add_argument('--root', type=Path, default=BUILD_ROOT)
    a = ap.parse_args()
    if a.build_profiles:
        build_profiles(a.root.resolve())
        return
    if a.smoke:
        import subprocess
        out = (a.smoke_out or a.root).resolve()
        out.mkdir(parents=True, exist_ok=True)
        try:
            res = smoke(a.root.resolve(), out)
        except Exception as exc:  # noqa: BLE001  - the failed check is recorded, then re-raised
            context.write_json(out / 'smoke.json', {'status': 'FAIL', 'error': '%s: %s' % (type(exc).__name__, str(exc)[:2000])})
            raise
        # related existing controls, outside every patch: controller tests; Source census + fault paths (worker stop, withheld labels,
        # resume/finalize); adapter-factory branch path + fault paths
        res['existing_controls'] = []
        for cmd in (['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/functional/test_batch_research.py'],
                    ['-m', 'evaluation.main_protocol_p4.run_batch_research_roundtrip', '--smoke', '--study', 'source_process'],
                    ['-m', 'evaluation.main_protocol_p4.run_batch_research_roundtrip', '--smoke', '--study', 'paired_refinement']):
            t0 = time.time()
            pr = subprocess.run([sys.executable, '-B'] + cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
            res['existing_controls'].append({'command': ' '.join(cmd[1:]), 'returncode': pr.returncode, 'seconds': round(time.time() - t0, 1),
                                             'tail': (pr.stdout + pr.stderr)[-400:]})
            if pr.returncode:
                res['status'] = 'FAIL'
        context.write_json(out / 'smoke.json', res)
        print('DOMAIN_SKILL_SMOKE', res['status'], '%d checks' % len(res['checks']), flush=True)
        if res['status'] != 'PASS':
            raise SystemExit(1)
        return
    if a.smoke_pilot:
        out = PILOT_ROOT / 'smoke'
        out.mkdir(parents=True, exist_ok=True)
        try:
            res = smoke_pilot()
        except Exception as exc:  # noqa: BLE001
            context.write_json(out / 'smoke_pilot.json', {'status': 'FAIL', 'error': '%s: %s' % (type(exc).__name__, str(exc)[:2000])})
            raise
        context.write_json(out / 'smoke_pilot.json', res)
        print('PILOT_SMOKE', res['status'], '%d checks' % len(res['checks']), flush=True)
        return
    if a.pilot == 'run':
        pilot_run(PILOT_ROOT)
        return
    if a.pilot == 'result':
        pilot_result(PILOT_ROOT)
        return
    if a.formation:
        formation(a.formation, a.formation_config)
        return
    ap.error('choose --build-profiles, --smoke, --smoke-pilot, --pilot or --formation')


if __name__ == '__main__':
    # run inside the canonical module object: run_batch_research_roundtrip imports this module by name and configures its globals
    from evaluation.main_protocol_p4 import batch_research_domain_skill as _canonical
    _canonical.main()
