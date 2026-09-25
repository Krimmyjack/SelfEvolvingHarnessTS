"""DEV-DATA-READINESS-PROFILE-TRANSFER (docs/DEV_DATA_READINESS_PROFILE_TRANSFER_TASK_2026-09-17.md).

One package, one ledger. Domains: RD01B (new fixed cohort of the KDD with-missing file) and RD02 (12 PM2.5 stations).
  preflight (0 cost: T tables, RD01B roster rule, exposure, interpreter, RD02 card from R2)
  -> Source: RD01B_S1/S2 no-Skill Fast (C_B after both)          -> census + ONE Slow formation (KEEP or W1/W2)
  -> dev: RD01B_S2 (W1, W2; no-Skill = Source reading) and RD01B_V1 (W2, NO_SKILL, W1); every commit before any C_B
  -> select + freeze RD01B (NO_SKILL / W1 / W2) -> frozen catalog (RD01B card if any, RD02-C1-r2, reference T profiles)
  -> routing: one profile_match call per Test job before any Test fit (anonymous catalog ids; Q1 forward, Q2 reversed)
  -> Test: RD01B_Q1, RD02_Q1, RD01B_Q2, RD02_Q2 x {F0, F_known, F_match, F_generic, Random} + common programs;
     identical executions decided before Test effects; all commits -> C_B -> freeze E -> barrier -> E (+ shadow)
  -> readout.

Reused unchanged: the Fast loop and adapter (batch_research_data_readiness), stage worker / arm runner / label order
(batch_research_data_readiness_r2), Skill carrier / propose / select / freeze / profile_match routing (domain_skill, with
opt-in vocabularies), the metered client and ledger.

  --smoke | --preflight | --run | --resume-stage STAGE [--accept-unknown-usage] | --result
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha.batch_base import context, llm
from methods.ttha.batch_base import readiness as rd
from evaluation.main_protocol_p4 import batch_research_data_readiness as drs
from evaluation.main_protocol_p4 import batch_research_data_readiness_r2 as r2
from evaluation.main_protocol_p4 import batch_research_domain_skill as dsks
from evaluation.main_protocol_p4 import batch_research_runtime as rt

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_data_readiness_profile_transfer'
TASK = 'docs/DEV_DATA_READINESS_PROFILE_TRANSFER_TASK_2026-09-17.md'
PARENT = drs.ROOT
R2_ROOT = r2.ROOT
SEEDS = drs.SEEDS
LIMITS = drs.LIMITS
DOMAINS = ('RD01B', 'RD02')
SOURCE_JOBS = ('RD01B_S1', 'RD01B_S2')
DEV_JOBS = ('RD01B_S2', 'RD01B_V1')
DEV_ORDER = {'RD01B_S2': ('W1', 'W2'), 'RD01B_V1': ('W2', 'no_skill', 'W1')}
TEST_JOBS = ('RD01B_Q1', 'RD02_Q1', 'RD01B_Q2', 'RD02_Q2')
TEST_ORDER = {'RD01B_Q1': ('F0', 'F_known', 'F_match', 'F_generic', 'Random'), 'RD02_Q1': ('F_generic', 'F_match', 'F_known', 'F0', 'Random'),
              'RD01B_Q2': ('F_known', 'F0', 'F_generic', 'F_match', 'Random'), 'RD02_Q2': ('F_match', 'F_generic', 'F0', 'F_known', 'Random')}
TEST_RANDOM_SEEDS = {'RD01B_Q1': (2026091801, 2026091802), 'RD02_Q1': (2026091803, 2026091804),
                     'RD01B_Q2': (2026091805, 2026091806), 'RD02_Q2': (2026091807, 2026091808)}
REF_JOBS = {'RD01B': ('RD01B_S1', 'RD01B_S2', 'RD01B_V1'), 'RD02': ('RD02_S1', 'RD02_S2', 'RD02_V1')}
T_TABLE = {'RD01B_S1': (4560, 15.178571, 68), 'RD01B_S2': (5760, 4.110863, 89), 'RD01B_V1': (6960, 9.263393, 53),
           'RD01B_Q1': (8160, 1.446243, 139), 'RD01B_Q2': (9360, 1.274182, 112), 'RD02_Q1': (11760, 1.847718, 132), 'RD02_Q2': (12960, 2.703373, 216)}
RD02_Q_GEOMETRY = {'RD02_Q1': {'T': [11088, 11760], 'C_A': [11760, 11856], 'C_B': [11856, 11952], 'E': [11952, 12144]},
                   'RD02_Q2': {'T': [12288, 12960], 'C_A': [12960, 13056], 'C_B': [13056, 13152], 'E': [13152, 13344]}}
PRSA_CLOSED_FROM_ROW = r2.SEALED_FROM_ROW
TOTAL = {'max_fit_attempts': 206, 'max_llm_requests': 374, 'max_llm_tokens': 6_000_000, 'max_wall_s': 28800, 'max_retries': 2}
HTTP_CAP = 748
ALLOC = {'source': {'fits': 24, 'requests': 32, 'tokens': 700_000},
         'formation': {'fits': 0, 'requests': 2, 'tokens': 250_000},
         'dev': {'fits': 36, 'requests': 80, 'tokens': 1_600_000},
         'routing': {'fits': 0, 'requests': 4, 'tokens': 50_000},
         'test': {'fits': 144, 'requests': 256, 'tokens': 3_400_000}}
SLOW_MAX_TOKENS = 12000
CENSUS_MAX_BYTES = 78_000                     # 2*(B+14048) <= 250k and a correction (B + prior + 14048) after the first usage
ROUTER_MAX_TOKENS = 3500
ROUTER_MAX_BYTES = 8_500                      # 2*(B+2048+3500) = 28.1k; before the 4th call <= 3*(B/2.5+3500) = 20.7k -> 48.8k <= 50k routing cap
MODEL = drs.MODEL
FAST_ARMS = ('F0', 'F_known', 'F_match')
PROFILE_VERSION = 'data_readiness_profile_v1'
ROUTER_FIELDS = ('missing_fraction', 'longest_gap', 'head_gap', 'tail_gap', 'lag24_corr', 'lag168_corr', 'constant_run_fraction',
                 'robust_deviation_fraction', 'robust_z_max', 'recent168_level_shift', 'recent168_volatility_ratio')
PROFILE_STATS = ('p25', 'median', 'p75', 'valid_fraction')
COMPAT_KEYS = ('task', 'consumer', 'training', 'lookback_L', 'horizon_H', 'sampling_interval_hours', 'T_length', 'readiness_geometry')
ROUTER_FORBIDDEN_KEYS = frozenset({'n_entities', 'n_valid', 'n', 'legal_parents_total', 'finite_mean', 'finite_std', 'job_id', 'train_rows', 'entities',
                                   'workflow', 'principles', 'rendered_body', 'evidence_refs', 'c_a', 'c_b', 'e', 'scores', 'roster', 'dataset', 'path', 't'})


def ds_of(job: str) -> str:
    return job.split('_')[0]


def paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'configs': root / 'stage_configs', 'logs': root / 'logs',
            'source': root / 'source', 'formation': root / 'formation', 'dev': root / 'dev', 'catalog': root / 'catalog', 'routing': root / 'routing',
            'freeze': root / 'freeze', 'test': root / 'test', 'smoke': root / 'smoke'}


def stage_caps(root: Path, stage: str) -> dict:
    P = paths(root)
    return dsks.stage_caps(root, stage, total=TOTAL, alloc=ALLOC, http_cap=HTTP_CAP, stages_path=P['stages'], ledger_path=P['ledger'])


def request_bytes(system: str, payload: dict) -> int:
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
    return len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))


def _knowledge(skill) -> dict:
    return br.json_copy(asdict(dsk.skill_knowledge(skill)))


# ============================================================================= readiness profile and routing
def readiness_compatibility() -> dict:
    return {'task': 'hourly point forecast of the next 48 hours from 192 hours, naturally missing values, input preparation only',
            'consumer': rd.CONSUMER['arch'], 'training': 'one shared model per seed over all legal windows of the batch; AdamW, 2000 updates, batch 64',
            'lookback_L': rd.L, 'horizon_H': rd.H, 'sampling_interval_hours': 1, 'T_length': rd.TRAIN_SPAN,
            'readiness_geometry': 'legal training window: X has >= 32 observed values and the 48-point target is fully observed; only X is prepared'}


def readiness_profile(overview: dict) -> dict:
    """T-only view: per field [p25, median, p75, valid_fraction] from the overview's own summary (computed before any filling).
    An all-unknown field keeps null quantiles; no entity count, raw level, window count or identity is carried."""
    n = int(overview['n_entities'])
    fields = {}
    for f in ROUTER_FIELDS:
        s = overview['summary'].get(f)
        vf = round((int(s['n']) if s else 0) / n, 4)
        fields[f] = [None, None, None, vf] if s is None else [round(float(s['p25']), 4), round(float(s['median']), 4), round(float(s['p75']), 4), vf]
    return {'profile_version': PROFILE_VERSION, 'fields': fields}


def batch_features(overview: dict) -> dict:
    return {'batch_median_' + k: v['median'] for k, v in overview['summary'].items() if v is not None}


def _identity_scan(payload: dict, *, forbidden_strings=()) -> list:
    """Router request must not carry identity, population size, windows, raw levels, outcomes or Skill bodies."""
    bad = []
    names = tuple(n.lower() for n in rd.SOURCE_NAMES) + tuple(str(x).lower() for x in forbidden_strings)
    uid = re.compile(r'(?<![A-Za-z0-9_])T\d{1,3}(?![A-Za-z0-9_])')

    def walk(x, where):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ROUTER_FORBIDDEN_KEYS:
                    bad.append('key %s at %s' % (k, where))
                walk(v, where + '.' + str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, '%s[%d]' % (where, i))
        elif isinstance(x, str):
            low = x.lower()
            if any(n in low for n in names) or rd._JOB_TOKEN.search(x) or uid.search(x) or re.search(r'entity_\d+|[A-Za-z]:\\|_scratch', x):
                bad.append('string at %s: %r' % (where, x[:60]))
        elif isinstance(x, bool) or x is None or isinstance(x, float):
            return
        elif isinstance(x, int) and x not in (1, rd.L, rd.H, rd.TRAIN_SPAN):
            bad.append('integer %d at %s' % (x, where))
    walk(payload, '$')
    return bad


def router_payload(query: dict, candidates: list) -> dict:
    """candidates: [(catalog_id, reference profiles, skill)] in the frozen presentation order."""
    comp = readiness_compatibility()
    payload = {'current_batch_profile': dsk._rounded(query, 3),
               'profile_stats_order': list(PROFILE_STATS),
               'candidate_domains': [{'domain_id': cid, 'reference_batch_profiles': dsk._rounded(refs, 3)} for cid, refs, _ in candidates],
               'candidate_skills': [{'skill_id': cid, 'domain_id': cid, 'compatibility_note': s.compatibility_note, 'applicability_summary': s.applicability_summary}
                                    for cid, _, s in candidates],
               'compatibility': {'current_batch_and_every_candidate': comp},
               'contract': {'response': {'selected_skill_id': 'id or null', 'status': 'SELECTED|ABSTAIN', 'evidence_fields': ['profile field or compatibility key'], 'rationale': ''},
                            'note': 'Frozen for the whole job; grants no permission; the Skill scope is still checked. Each profile field is '
                                    '[p25, median, p75, valid_fraction]; valid_fraction only says how often the field is defined.'}}
    return payload


def route_job(query_overview: dict, catalog: dict, order, call, *, true_domain: str) -> dict:
    """One routing record. Transport/format faults are ROUTING_FAILED (never ABSTAIN); no card -> NO_CANDIDATE_SKILLS without a call."""
    cards = []
    for d in order:
        entry = catalog['cards'].get(d)
        if not entry:
            continue
        if entry.get('profile') != 'data_readiness' or catalog['reference_profiles'][d]['profile_version'] != PROFILE_VERSION:
            raise PermissionError('a readiness job can only route to readiness Skills with readiness reference profiles')
        cards.append((d, dsk.skill_from_json(entry['skill'])))
    anon = [('C%d' % i, d, replace(s, skill_id='C%d' % i, domain_id='C%d' % i)) for i, (d, s) in enumerate(cards, 1)]
    query = readiness_profile(query_overview)
    if query['profile_version'] != PROFILE_VERSION:
        raise PermissionError('query profile version')
    payload = router_payload(query, [(cid, catalog['reference_profiles'][d]['profiles'], s) for cid, d, s in anon])
    rec = {'runtime_catalog_map': {cid: {'domain': d, 'skill_id': s.skill_id} for (cid, d, _), (_, s) in zip(anon, cards)},
           'presentation_order': list(order), 'true_domain_diagnostic_only': true_domain, 'query_profile': query}
    bad = _identity_scan(payload, forbidden_strings=list(rd.RD01B_ROSTER) + list(rd.RD01_ROSTER) + [s.workflow[:40] for _, s in cards])
    nbytes = request_bytes(dsk.ROUTER_SYSTEM, payload)
    rec.update(request_bytes=nbytes, identity_scan=bad)
    if bad:
        raise PermissionError('router request carries identity/outcome content: %s' % bad[:3])
    if anon and nbytes > ROUTER_MAX_BYTES:
        raise RuntimeError('router request %d bytes exceeds the frozen bound %d' % (nbytes, ROUTER_MAX_BYTES))
    feats = batch_features(query_overview)
    got = dsk.route_profile_match(None, {}, [a for _, _, a in anon], call, feats, payload=payload, profile_fields=ROUTER_FIELDS,
                                  compatibility_keys=COMPAT_KEYS, allowed_features=drs.ALLOWED_FEATURES)
    got = br.json_copy({k: v for k, v in got.items() if k not in ('candidate_domain_ids',)})
    rec['router'] = got
    status = got['status']
    sel_cid = got.get('selected_skill_id')
    real = rec['runtime_catalog_map'].get(sel_cid) if sel_cid else None
    if status in ('ROUTER_CALL_FAILED', 'ROUTER_PARSE_OR_VALIDATION_FAILED'):
        rec['routing_status'] = 'ROUTING_FAILED'
    else:
        rec['routing_status'] = status                 # NO_CANDIDATE_SKILLS | ABSTAIN | SELECTED | SELECTED_SCOPE_NOT_MATCHED
    rec['selected_domain'] = real['domain'] if real and status in ('SELECTED', 'SELECTED_SCOPE_NOT_MATCHED') else None
    rec['selected_skill_id'] = real['skill_id'] if rec['selected_domain'] else None
    rec['scope_state'] = (got.get('scope_at_route') or {}).get('state')
    rec['load'] = 'CARD' if rec['selected_domain'] else 'NO_CARD'
    rec['selected_equals_true_domain'] = (rec['selected_domain'] == true_domain) if rec['selected_domain'] else None
    rec['llm_requests'] = got.get('llm_requests', 0)
    return rec


# ============================================================================= stage runner (r2 worker; this package's paths and caps)
def stage_config(root: Path, stage: str, *, jobs, order, arms, labels_e: bool, parent_common=None) -> Path:
    P = paths(root)
    path = P['configs'] / ('%s.json' % stage)
    if path.exists():
        return path
    sc = stage_caps(root, stage)
    cfg = {'study': 'data_readiness_profile_transfer', 'status': 'FROZEN', 'stage': stage, 'output': str(P[stage]), 'jobs': list(jobs),
           'order': {j: list(order[j]) for j in jobs}, 'arms': br.json_copy(arms), 'parent_common': parent_common or {}, 'labels_e': bool(labels_e),
           'seeds': list(SEEDS), 'limits': LIMITS, 'max_tool_corrections': drs.MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': drs.EVIDENCE_ROUNDTRIP,
           'caps': sc['caps'], 'http_cap': sc['http_cap'], 'ledger_path': str(P['ledger']), 'fit_retry': True,
           'fit_attempts_at_stage_start': context.read_json(P['ledger'])['fit_attempts']}
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
            p = subprocess.Popen([sys.executable, '-B', '-m', 'evaluation.main_protocol_p4.batch_research_data_readiness_transfer', '--stage-worker', str(cfg_path)],
                                 cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)
                context.write_json(root / ('%s_supervisor_timeout.json' % stage), {'epoch': time.time(), 'pid': p.pid})
                rc = 124
        print('STAGE_EXIT', stage, rc, flush=True)
    return dsks.stage_status(sroot)


def resume_stage(root: Path, stage: str, *, accept_unknown_usage: bool = False) -> None:
    """One user-authorized continuation of a stage whose labels are still withheld (r2 semantics, this package's paths)."""
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
            raise RuntimeError('ledger holds unknown usage; pass --accept-unknown-usage to record an explicit user decision')
        led.s['unknown_usage_accepted'] = led.s['llm_tokens_unknown']
        led._save()
    for name in ('execution_finished.json', 'labels_withheld.json'):
        if (sroot / name).exists():
            shutil.move(sroot / name, sroot / name.replace('.json', '_before_resume.json'))
    context.write_json(sroot / 'resume.json', {'epoch': time.time(), 'accepted_unknown_usage': bool(accept_unknown_usage)})
    rt.FIT_RETRY = bool(cfg.get('fit_retry'))
    branches, failures, stopped = [], [], None
    try:
        client = rt.MeteredClient(led, sroot / 'raw_responses', http_cap=int(cfg['http_cap']))
        r2._run_arms(sroot, cfg, led, client, branches, failures, resumed=True)
        context.write_json(sroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches], 'resumed': True})
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': drs._safe_message(exc)})
        stopped = type(exc).__name__
    finally:
        context.write_json(sroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': True})
    drs.open_labels(sroot, cfg, branches, failures, led, resumed=True, withhold_reason=stopped and 'resumed stage stopped again (%s)' % stopped)


# ============================================================================= preflight (0 cost)
def rd01b_roster_rule() -> list:
    import io
    import zipfile
    import numpy as np
    toks = {}
    with zipfile.ZipFile(rd.DATASETS['RD01B']['path']) as z, z.open(z.namelist()[0]) as h:
        in_data = False
        for line in io.TextIOWrapper(h, encoding='utf-8'):
            if not in_data:
                in_data = line.strip().lower() == '@data'
                continue
            f = line.rstrip().split(':')
            toks[f[0]] = f[-1].split(',')
    roster = []
    for uid in sorted(toks)[80:]:
        if uid in rd.RD01_ROSTER or len(toks[uid]) < 9744:
            continue
        ok = True
        for job in ('RD01B_S1', 'RD01B_S2', 'RD01B_V1', 'RD01B_Q1', 'RD01B_Q2'):
            t = T_TABLE[job][0]
            seg = np.array([np.nan if x == '?' else float(x) for x in toks[uid][t - rd.TRAIN_SPAN:t]])[:, None]
            fin = np.isfinite(seg[:, 0])
            if fin.sum() < rd.MIN_T_FINITE or not seg[fin, 0].std() > 1e-6 or rd.legal_parents(seg).counts[0] < rd.MIN_LEGAL_PARENTS:
                ok = False
                break
        if ok:
            roster.append(uid)
        if len(roster) == 32:
            break
    return roster


def t_eligibility() -> dict:
    import numpy as np
    out = {}
    for job, (t, miss, minp) in T_TABLE.items():
        js = rd.resolve_job(ds_of(job), job)
        sl = rd.load_slice(ds_of(job), t - rd.TRAIN_SPAN, t)
        seg = sl.rows(t - rd.TRAIN_SPAN, t)
        lg = rd.legal_parents(seg)
        fin = np.isfinite(seg)
        ok = all(fin[:, e].sum() >= rd.MIN_T_FINITE and seg[fin[:, e], e].std() > 1e-6 and lg.counts[e] >= rd.MIN_LEGAL_PARENTS for e in range(seg.shape[1]))
        row = {'t': js.t, 'missing_rate_pct': round(float(np.isnan(seg).mean() * 100), 6), 'min_legal_parents': int(lg.counts.min()), 'legal_parents_total': lg.n,
               'all_entities_eligible': ok, 'geometry': {'T': list(js.train_range), 'C_A': [min(js.c_a), max(js.c_a) + rd.H], 'C_B': [min(js.c_b), max(js.c_b) + rd.H],
                                                         'E': [min(js.e), max(js.e) + rd.H]}}
        row['matches_task_table'] = row['t'] == t and row['missing_rate_pct'] == miss and row['min_legal_parents'] == minp and ok \
            and (job not in RD02_Q_GEOMETRY or row['geometry'] == RD02_Q_GEOMETRY[job])
        if not row['matches_task_table']:
            raise RuntimeError('T table mismatch for %s: %s' % (job, row))
        out[job] = row
    return out


def exposure_check() -> dict:
    last_prsa = max(RD02_Q_GEOMETRY[j]['E'][1] for j in RD02_Q_GEOMETRY)
    if last_prsa > PRSA_CLOSED_FROM_ROW:
        raise RuntimeError('RD02 rows reach the closed 2016+ region')
    return {'status': 'NO_SEALED_REGION_TOUCHED', 'dataset_exposure': rd.EXPOSURE,
            'RD02': {'last_label_row_exclusive': last_prsa, 'closed_from_row': PRSA_CLOSED_FROM_ROW, 'dates': '2014-06-06 .. 2014-09-08 of the hourly files (rows [11088, 13344))'},
            'RD01B': {'last_label_row_exclusive': T_TABLE['RD01B_Q2'][0] + 336 + rd.H, 'identity': 'EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING (AGENTS.md §5.1/§8.1)',
                      'note': 'no sealed KDD with-missing region is declared; the sealed sets are Yahoo remaining 41, NOAA beyond 17520, SMD test, UCR TEST, PRSA 2016+ and Natural Final'},
            'records_checked': ['AGENTS.md §5, §5.1, §8.1', 'docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md §3', 'docs/HARNESS_RESEARCH_DIRECTION_AND_PLAN.md §2.2-2.3']}


def rd02_card() -> dict:
    fr = context.read_json(R2_ROOT / 'freeze' / 'frozen.json')
    hs = fr['H_selected']
    if hs['option'] != 'RD02-C1-r2' or hs['skill']['skill_id'] != 'RD02-C1-r2' or hs['skill']['status'] != 'FROZEN_SELECTED':
        raise RuntimeError('R2 frozen selection is not RD02-C1-r2')
    prop = context.read_json(R2_ROOT / 'revision' / 'proposal.json')
    cand = next(c['skill'] for c in prop['candidates'] if c['skill']['skill_id'] == 'RD02-C1-r2')
    if any(hs['skill'][k] != cand[k] for k in ('workflow', 'principles', 'observable_applicability', 'rendered_body', 'applicability_summary')):
        raise RuntimeError('R2 frozen card differs from its development candidate text')
    return hs['skill']


def preflight(root: Path) -> dict:
    if (root / 'package_config.json').exists():
        return context.read_json(root / 'package_config.json')
    # imported only here: torch brings its own OpenMP runtime, which must not be loaded in the driver process that later runs
    # numpy/MKL-threaded T overviews (fits run in their own worker processes)
    import inspect
    import torch
    from SelfEvolvingHarnessTS.operators import _provenance
    src = inspect.getsource(rt.MeteredClient)
    if not all(x in src for x in ("model='cpa-grok-4.6'", "'grok-4.6-build'", 'temperature=0', "base_url='http://127.0.0.1:8318/v1'")):
        raise RuntimeError('metered client no longer carries the frozen model identity')
    fp = _provenance.dependency_fingerprint()
    if fp != context.read_json(PARENT / 'package_config.json')['dependency_fingerprint']:
        raise RuntimeError('numerical dependency fingerprint differs from the parent readiness packages')
    roster = rd01b_roster_rule()
    if roster != rd.RD01B_ROSTER:
        raise RuntimeError('RD01B roster rule does not reproduce the frozen roster')
    elig = t_eligibility()
    card = rd02_card()
    gc = drs.generic_control_record()
    for j in REF_JOBS['RD02']:
        if not (PARENT / 'source' / ('%s_common' % j) / j / 'overview.json').exists() and not (PARENT / 'select' / ('%s_common' % j) / j / 'overview.json').exists():
            raise RuntimeError('parent RD02 reference overview missing: %s' % j)
    cfg = {'package': 'DEV-DATA-READINESS-PROFILE-TRANSFER', 'task': TASK, 'exposure': rd.EXPOSURE, 'frozen_epoch': time.time(),
           'freeze_note': 'Frozen before the first paid call; this preflight read T rows of the planned jobs only (eligibility, roster rule).',
           'interpreter': {'executable': sys.executable, 'python': sys.version.split()[0], 'torch': torch.__version__, 'dependency_fingerprint': fp,
                           'note': 'the base Anaconda interpreter of the parent readiness and R2 packages (fingerprint equal)'},
           'domains': {'RD01B': {'roster': rd.RD01B_ROSTER, 'roster_rule': rd.DATASETS['RD01B']['roster_rule'], 'path': str(rd.DATASETS['RD01B']['path'])},
                       'RD02': {'roster': rd.RD02_ROSTER, 'card': card}},
           't_eligibility': elig, 'exposure_check': exposure_check(),
           'geometry': {'L': rd.L, 'H': rd.H, 'T': rd.TRAIN_SPAN, 'c_a': 't, t+48', 'c_b': 't+96, t+144', 'e': 't+192..t+336 step 48', 'coverage_min': rd.COVERAGE_MIN},
           'jobs': {'source': list(SOURCE_JOBS), 'dev': list(DEV_JOBS), 'dev_order': {j: list(o) for j, o in DEV_ORDER.items()}, 'test': list(TEST_JOBS),
                    'test_order': {j: list(o) for j, o in TEST_ORDER.items()}, 'reference_profiles': {d: list(j) for d, j in REF_JOBS.items()}},
           'test_random': {j: {'seeds': list(s), 'policies': [rd.random_policy(x) for x in s]} for j, s in TEST_RANDOM_SEEDS.items()},
           'profile': {'version': PROFILE_VERSION, 'router_fields': list(ROUTER_FIELDS), 'stats': list(PROFILE_STATS), 'compatibility': readiness_compatibility(),
                       'forbidden_router_keys': sorted(ROUTER_FORBIDDEN_KEYS), 'router_system': dsk.ROUTER_SYSTEM, 'router_max_tokens': ROUTER_MAX_TOKENS,
                       'router_max_request_bytes': ROUTER_MAX_BYTES, 'presentation_order': 'Q1 jobs: RD01B then RD02; Q2 jobs: reversed; anonymous ids C1/C2'},
           'formation': {'slow_system': drs.SLOW_SYSTEM, 'max_tokens': SLOW_MAX_TOKENS, 'census_max_request_bytes': CENSUS_MAX_BYTES},
           'seeds': list(SEEDS), 'limits': LIMITS, 'max_tool_corrections': drs.MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': drs.EVIDENCE_ROUNDTRIP,
           'fast_system': drs.FAST_SYSTEM, 'contracts': drs.CONTRACTS, 'generic_control': gc, 'model': MODEL,
           'total_caps': TOTAL, 'http_cap': HTTP_CAP, 'stage_allocations': ALLOC,
           'fit_budget_formula': 'Source 2x(6+6)=24; dev 2x6 (S2 cards) + 6 (V1 common) + 3x6 (V1 arms) = 36; Test 4x(6+5x6)=144; +2 retries = 206',
           'selection_rule': 'RD01B on S2 and V1: per option three-seed C_B mean of the actual commit, equal weight over both jobs; complete on both; ties NO_SKILL, W1, W2',
           'equivalence_rule': 'before Test effects: among F0/F_known/F_match an arm whose knowledge (version, hook, body, scope) equals an earlier-running arm of the job '
                               'references it (IDENTICAL_EXECUTION_REFERENCE); F_generic never merged',
           'readout': 'delta_skill=E(F0)-E(F_known); delta_routing=E(F_known)-E(F_match); delta_system=E(F0)-E(F_match); relative = difference / same-job Baseline_Linear E mean; '
                      'job equal weight within domain, domains equal weight',
           'dependency_fingerprint': fp}
    root.mkdir(parents=True, exist_ok=True)
    dsks.write_once(root / 'package_config.json', cfg)
    return cfg


# ============================================================================= Stage A: Source, formation, dev selection, freeze
def source_stage(root: Path) -> dict:
    arms = {j: {'no_skill': {'kind': 'fast', 'option': dsk.NO_SKILL, 'knowledge': br.json_copy(asdict(br.Knowledge()))}} for j in SOURCE_JOBS}
    cfgp = stage_config(root, 'source', jobs=SOURCE_JOBS, order={j: ['no_skill'] for j in SOURCE_JOBS}, arms=arms, labels_e=False)
    return run_stage(root, 'source', cfgp)


def formation_census(root: Path, *, level: int = 0) -> dict:
    """Compact whitelist census of this package's RD01B Source (r2 summaries); level 1 drops per-entity job rows,
    level 2 also drops per-entity inspection details. No E, no Test profile, no other package's outcome."""
    P = paths(root)
    records, jobs, trajs, refs, legend = [], {}, [], [], {}
    for i, job in enumerate(SOURCE_JOBS, 1):
        b = P['source'] / ('%s_no_skill' % job)
        ns = P['source'] / ('%s_not_scorable.json' % job)
        if ns.exists():
            records.append({'batch': job, 'status': 'C_A_NOT_SCORABLE', 'note': 'common observed-cell mask of C_A below the 25% rule; no Fast trajectory'})
            continue
        if not drs._branch_complete_with_c_b(b, job):
            r = b / 'branch_result.json'
            res = context.read_json(r) if r.exists() else None
            records.append({'batch': job, 'status': 'INCOMPLETE', 'result': {k: v for k, v in res.items() if k != 'trace'} if res else None})
            continue
        code = 's%d' % i
        legend[code] = {'batch': job, 'role': 'no_skill', 'note': 'Fast without any Skill on an earlier batch of this domain'}
        jb = r2._job_block(b, job)
        if level >= 1:
            jb = {k: v for k, v in jb.items() if k != 'entity_rows'}
        jobs[job] = jb
        refs.append('%s:overview' % job)
        rec, rr = r2._branch_record(b, job, 'no_skill', legend[code]['note'], b.parent, {}, code)
        if level >= 2:
            for t in rec['trajectory']:
                if t.get('tool') in ('inspect_data', 'inspect_material') and isinstance(t.get('output'), dict):
                    t['output'] = {k: (v if k not in ('entities',) else 'omitted at compaction level 2') for k, v in t['output'].items()}
        trajs.append(rec)
        refs += rr
    base = {'domain_id': 'RD01B', 'profile': 'data_readiness', 'structural_and_technical_records': records}
    if not trajs:
        return {**base, 'census_status': 'SOURCE_INCOMPLETE', 'legal_evidence_refs': [], 'note': 'no COMPLETE Source trajectory with post-commit C_B; no Slow call'}
    ev = {**base, 'census_status': 'CENSUS_COMPLETE', 'compaction_level': level, 'evidence_reduced_to_one_trajectory': len(trajs) < 2,
          'reading_notes': {'blocks': 'C_A = two origins right after T (the only feedback Fast had); C_B = two later origins opened only after the branch committed; '
                                      'missing-aware normalized MSE on observed future cells, lower is better; three seeds are Consumer repeats of one trajectory',
                            'batches': 'earlier batches of the same fixed population at increasing times; not repeats',
                            'refs': 'every ref string in this census is a legal evidence ref (branch code:event number, <batch>:overview, <branch code>:c_b_after_commit)',
                            'inspect_data_summaries': {'gaps': '[missing hours, number of gaps, longest gap hours, [[start row, hours] of gaps >= 6h]]',
                                                       'hour_profile': '[peak clock hour, trough clock hour, range of hourly means]',
                                                       'daily_means': '[min daily mean, max daily mean, days with mean > 1, missing hours]',
                                                       'segment': '[start row, rows, missing, min, max]', 'units': 'normalized by the frozen per-entity T scaler; eN = entity index N'},
                            'inspect_material_summary': 'totals, operator records, and entities with fill difference from linear >= 0.5 or changed observations: '
                                                        '[filled points, observed changed fraction, fill difference from linear rms, window std ratio vs linear]',
                            'actions': '[tool, arguments] in the order Fast sent them',
                            'omitted': 'raw values and material arrays are summarized; no E exists here'},
          'branch_legend': legend, 'jobs': jobs, 'branches': trajs,
          'public_semantics': {'tool_arguments': {k: v['arguments'] for k, v in drs.CONTRACTS.items()}, 'limits': {**LIMITS, 'max_tool_corrections': drs.MAX_TOOL_CORRECTIONS},
                               'field_definitions': rd.FIELD_DEFINITIONS, 'ops_parameters': rd.OPS, 'op_semantics': rd.OP_SEMANTICS, 'canonicalization': rd.CANONICAL_RULE,
                               'common_programs': {'Baseline_Linear': 'empty program: minimum linear completion', 'Fixed_Seasonal': 'uniform period_median_complete(24,3,2)'}},
          'purpose': 'Offline Slow input: can the Source processes of this neutral domain be consolidated into at most two reusable data-readiness Workflows?',
          'legal_evidence_refs': sorted(set(refs))}
    low = json.dumps(ev, ensure_ascii=False).lower()
    if any(n in low for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')):
        raise PermissionError('census names a data source')
    return ev


def _propose_payload(census: dict) -> dict:          # the exact payload dsk.propose builds
    return {'domain_id': 'RD01B', 'census': census, 'legal_evidence_refs': census['legal_evidence_refs'], 'allowed_batch_features': sorted(drs.ALLOWED_FEATURES),
            'max_candidates': 2, 'status': 'CANDIDATE_TEST_ONLY'}


def formation_stage(root: Path, client=None) -> dict:
    P = paths(root)
    out = P['formation'] / 'RD01B'
    if (out / 'propose.json').exists():
        return context.read_json(out / 'propose.json')
    if not (out / 'census.json').exists():
        census = None
        for level in (0, 1, 2):
            census = formation_census(root, level=level)
            if census['census_status'] != 'CENSUS_COMPLETE' or request_bytes(drs.SLOW_SYSTEM, _propose_payload(census)) <= CENSUS_MAX_BYTES:
                break
        dsks.write_once(out / 'census.json', census)
    census = context.read_json(out / 'census.json')
    if census['census_status'] != 'CENSUS_COMPLETE':
        dsks.write_once(out / 'propose.json', {'status': 'NO_PROPOSAL_INPUT', 'census_status': census['census_status'], 'skills': []})
        return context.read_json(out / 'propose.json')
    nbytes = request_bytes(drs.SLOW_SYSTEM, _propose_payload(census))
    if nbytes > CENSUS_MAX_BYTES:
        raise RuntimeError('formation request %d bytes exceeds the frozen bound even at the last compaction level' % nbytes)
    sc = stage_caps(root, 'formation')
    led = rt.RuntimeLedger(P['ledger'], **sc['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage; paid formation refused')
    client = client or rt.MeteredClient(led, P['formation'] / 'raw_responses', http_cap=sc['http_cap'])
    res = dsk.propose(census, domain_id='RD01B', call=lambda p: client.call('slow_domain', 'RD01B', p, drs.SLOW_SYSTEM, max_tokens=SLOW_MAX_TOKENS),
                      allowed_features=drs.ALLOWED_FEATURES, text_check=drs.skill_text_check, forbidden_names=[n for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')])
    dsks.write_once(out / 'propose.json', {**{k: v for k, v in res.items() if k != 'skills'}, 'skills': [s.to_json() for s in res['skills']],
                                           'census_legal_ref_count': len(census['legal_evidence_refs']), 'request_bytes_first_attempt': nbytes})
    if getattr(client, 'fatal', False) or rt.unknown_usage_blocks(led):
        raise RuntimeError('backend fatal or unknown usage during formation; stop')
    return context.read_json(out / 'propose.json')


def candidates_rd01b(root: Path) -> list:
    prop = context.read_json(paths(root)['formation'] / 'RD01B' / 'propose.json')
    return [dsk.skill_from_json(s) for s in prop.get('skills', [])] if prop.get('status') == 'PROPOSED' else []


def dev_stage(root: Path) -> dict:
    P = paths(root)
    skills = candidates_rd01b(root)
    if not skills:
        rec = P['dev'] / 'dev_skipped.json'
        if not rec.exists():
            dsks.write_once(rec, {'status': 'NO_CANDIDATE', 'formation_status': context.read_json(P['formation'] / 'RD01B' / 'propose.json')['status'],
                                  'note': 'no card to select; RD01B stays NO_SKILL; no dev fits'})
        return {'status': 'SKIPPED'}
    by = {s.skill_id.split('-')[1]: s for s in skills}                     # W1 / W2
    order = {j: [a for a in DEV_ORDER[j] if a == 'no_skill' or a in by] for j in DEV_JOBS}
    arms = {}
    for j in DEV_JOBS:
        arms[j] = {}
        for a in order[j]:
            arms[j][a] = ({'kind': 'fast', 'option': dsk.NO_SKILL, 'knowledge': br.json_copy(asdict(br.Knowledge()))} if a == 'no_skill'
                          else {'kind': 'fast', 'option': by[a].skill_id, 'knowledge': _knowledge(by[a])})
    order['RD01B_S2'] = [a for a in order['RD01B_S2'] if a != 'no_skill']
    arms['RD01B_S2'].pop('no_skill', None)
    cfgp = stage_config(root, 'dev', jobs=DEV_JOBS, order=order, arms=arms, labels_e=False,
                        parent_common={'RD01B_S2': str(P['source'] / 'RD01B_S2_common')})
    return run_stage(root, 'dev', cfgp)


def select_and_freeze_rd01b(root: Path) -> dict:
    P = paths(root)
    out = P['formation'] / 'RD01B'
    if (out / 'frozen_skill.json').exists():
        return context.read_json(out / 'frozen_skill.json')
    skills = candidates_rd01b(root)
    prop = context.read_json(out / 'propose.json')
    if not skills:
        dsks.write_once(out / 'frozen_skill.json', {'domain_id': 'RD01B', 'status': 'NO_CANDIDATE', 'skill': None, 'propose_status': prop.get('status'),
                                                    'note': 'KEEP / no proposal input / formation failure: no selection, RD01B stays NO_SKILL'})
        return context.read_json(out / 'frozen_skill.json')
    options = {dsk.NO_SKILL: None, **{s.skill_id: s for s in skills}}
    arm = {dsk.NO_SKILL: 'no_skill', **{s.skill_id: s.skill_id.split('-')[1] for s in skills}}

    def bdir(oid, job):
        if oid == dsk.NO_SKILL and job == 'RD01B_S2':
            return P['source'] / 'RD01B_S2_no_skill'
        return P['dev'] / ('%s_%s' % (job, arm[oid]))

    def read_run(oid, knowledge, job):
        b = bdir(oid, job)
        if not (b / 'branch_result.json').exists():
            return {'status': 'NOT_RUN', 'synthetic': False}
        if context.read_json(b / 'knowledge.json') != br.json_copy(asdict(knowledge)):
            raise RuntimeError('selection branch %s ran other knowledge than option %s' % (b.name, oid))
        res = context.read_json(b / 'branch_result.json')
        return {'status': res['status'], 'committed_plan_id': res['committed_plan_id'], 'failure_kind': res['failure_kind'], 'synthetic': False, 'branch': str(b.relative_to(root)),
                'cost': {'fast_calls': res['calls'], 'tool_calls': res['tool_calls'], 'new_evaluations': res['new_evaluations']}}
    sel = dsk.select(options, list(DEV_JOBS), read_run, block='c_b', score=drs.c_b_scorer(bdir))
    dsks.write_once(out / 'selection.json', sel)
    fr = dsk.freeze('RD01B', sel, options, evidence_scope={'census': 'formation/RD01B/census.json', 'dev_jobs': list(DEV_JOBS), 'block': 'c_b',
                                                            'note': 'RD01B_S2 was read by the Slow: development selection, not independent validation'})
    dsks.write_once(out / 'frozen_skill.json', fr)
    return fr


def freeze_catalog(root: Path) -> dict:
    P = paths(root)
    cat_path = P['catalog'] / 'frozen_catalog.json'
    if cat_path.exists():
        return context.read_json(cat_path)
    if P['routing'].exists() or P['test'].exists():
        raise RuntimeError('routing or Test directory exists before the catalog freeze')
    fr = context.read_json(P['formation'] / 'RD01B' / 'frozen_skill.json')
    cards, status = {}, {}
    if fr['status'] == 'FROZEN_SELECTED':
        s = dsk.skill_from_json(fr['skill'])
        if s.domain_id != 'RD01B' or s.status != 'FROZEN_SELECTED':
            raise PermissionError('only a real FROZEN_SELECTED RD01B card enters the catalog')
        cards['RD01B'] = {'profile': 'data_readiness', 'skill': s.to_json()}
    else:
        cards['RD01B'] = None
    status['RD01B'] = fr['status']
    cards['RD02'] = {'profile': 'data_readiness', 'skill': rd02_card()}
    status['RD02'] = 'FROZEN_SELECTED (R2 development selection; Test did not show an increment)'
    profiles = {}
    for d, jobs in REF_JOBS.items():
        rows = []
        for j in jobs:
            ov = rd.open_job(d, j, P['catalog'] / 'profile_overviews', 'material').overview()
            if d == 'RD02':
                parent_ov = next(p for p in (PARENT / 'source' / ('%s_common' % j) / j / 'overview.json', PARENT / 'select' / ('%s_common' % j) / j / 'overview.json') if p.exists())
                if context.read_json(parent_ov) != br.json_copy(ov):
                    raise RuntimeError('recomputed RD02 reference overview differs from the parent package: %s' % j)
            rows.append({'runtime_job': j, 'profile': readiness_profile(ov)})
        profiles[d] = {'profile_version': PROFILE_VERSION, 'profiles': [r['profile'] for r in rows], 'runtime_jobs': [r['runtime_job'] for r in rows]}
    rec = {'status': 'CATALOG_FROZEN', 'epoch': time.time(), 'cards': cards, 'card_status': status, 'reference_profiles': profiles,
           'routable_card_count': sum(1 for v in cards.values() if v), 'compatibility': readiness_compatibility(), 'router_fields': list(ROUTER_FIELDS),
           'presentation_order': {'Q1': list(DOMAINS), 'Q2': list(reversed(DOMAINS))}, 'generic_control': drs.generic_control_record(),
           'frozen_before_any_test_router_or_fast': True,
           'note': 'a single-card catalog tests applicability / abstention only, not multi-Skill selection'}
    dsks.write_once(cat_path, rec)
    return rec


# ============================================================================= Stage B: routing; Stage C: Test
def routing_stage(root: Path, client=None) -> dict:
    P = paths(root)
    cat = context.read_json(P['catalog'] / 'frozen_catalog.json')
    if P['test'].exists():
        raise RuntimeError('Test directory exists before routing')
    led, out = None, {}
    for job in TEST_JOBS:
        path = P['routing'] / ('%s.json' % job)
        if path.exists():
            out[job] = context.read_json(path)
            continue
        ov = rd.open_job(ds_of(job), job, P['routing'] / 'query_overviews', 'material').overview()
        order = DOMAINS if job.endswith('Q1') else tuple(reversed(DOMAINS))
        if led is None and cat['routable_card_count']:
            sc = stage_caps(root, 'routing')
            led = rt.RuntimeLedger(P['ledger'], **sc['caps'])
            if rt.unknown_usage_blocks(led):
                raise RuntimeError('package ledger holds unknown usage; paid routing refused')
            client = client or rt.MeteredClient(led, P['routing'] / 'raw_responses', http_cap=sc['http_cap'])
        call = (lambda p, job=job: client.call('router', job, p, dsk.ROUTER_SYSTEM, max_tokens=ROUTER_MAX_TOKENS)) if client else (lambda p: None)
        rec = route_job(ov, cat, order, call, true_domain=ds_of(job))
        rec.update(job=job, epoch=time.time(), before_any_test_fit=not P['test'].exists())
        dsks.write_once(path, rec)
        out[job] = rec
        if client is not None and (getattr(client, 'fatal', False) or (led is not None and rt.unknown_usage_blocks(led))):
            raise RuntimeError('backend fatal or unknown usage during routing; stop')
    return out


def plan_test(root: Path) -> dict:
    P = paths(root)
    plan_path = P['freeze'] / 'test_plan.json'
    if plan_path.exists():
        return context.read_json(plan_path)
    if P['test'].exists():
        raise RuntimeError('Test directory exists before the Test plan freeze')
    cat = context.read_json(P['catalog'] / 'frozen_catalog.json')
    routes = {j: context.read_json(P['routing'] / ('%s.json' % j)) for j in TEST_JOBS}
    arms, equivalence = {}, {}
    for job in TEST_JOBS:
        d = ds_of(job)
        known = cat['cards'].get(d)
        k = {'F0': br.json_copy(asdict(br.Knowledge())),
             'F_known': _knowledge(dsk.skill_from_json(known['skill'])) if known else br.json_copy(asdict(br.Knowledge())),
             'F_generic': br.json_copy(asdict(drs.generic_knowledge()))}
        r = routes[job]
        k['F_match'] = _knowledge(dsk.skill_from_json(cat['cards'][r['selected_domain']]['skill'])) if r['load'] == 'CARD' else br.json_copy(asdict(br.Knowledge()))
        opt = {'F0': dsk.NO_SKILL, 'F_known': known['skill']['skill_id'] if known else dsk.NO_SKILL,
               'F_match': r['selected_skill_id'] or dsk.NO_SKILL, 'F_generic': 'GENERIC_CONTROL'}
        a, running = {}, []
        for arm in TEST_ORDER[job]:
            if arm == 'Random':
                a[arm] = {'kind': 'random', 'policy_seeds': list(TEST_RANDOM_SEEDS[job])}
                continue
            ref = next((x for x in running if arm in FAST_ARMS and x in FAST_ARMS and r2._same_policy(k[arm], k[x])), None)
            if ref:
                a[arm] = {'kind': 'reference', 'status': 'IDENTICAL_EXECUTION_REFERENCE', 'reference_arm': ref, 'option': opt[arm],
                          'note': 'knowledge (version, hook, body, scope) equals the earlier-running %s arm of this job; one shared execution, not an independent LLM repeat' % ref}
            else:
                a[arm] = {'kind': 'fast', 'option': opt[arm], 'knowledge': k[arm]}
                running.append(arm)
        arms[job] = a
        equivalence[job] = {arm: v.get('reference_arm') for arm, v in a.items() if v['kind'] == 'reference'}
    rec = {'status': 'TEST_FROZEN', 'epoch': time.time(), 'arms': arms, 'equivalence': equivalence, 'routes': {j: {k2: routes[j][k2] for k2 in
           ('routing_status', 'selected_domain', 'selected_skill_id', 'scope_state', 'load', 'selected_equals_true_domain')} for j in TEST_JOBS},
           'order': {j: list(o) for j, o in TEST_ORDER.items()}, 'random_seeds': {j: list(s) for j, s in TEST_RANDOM_SEEDS.items()},
           'frozen_before_first_test_fit': True}
    dsks.write_once(plan_path, rec)
    return rec


def test_stage(root: Path) -> dict:
    plan = context.read_json(paths(root)['freeze'] / 'test_plan.json')
    cfgp = stage_config(root, 'test', jobs=TEST_JOBS, order=TEST_ORDER, arms=plan['arms'], labels_e=True)
    return run_stage(root, 'test', cfgp)


# ============================================================================= driver
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
    if not paths(root)['ledger'].exists():
        stage_caps(root, 'source')
    st = source_stage(root)
    status['source'] = st
    if st['status'] != 'FINISHED':
        return stop('source', st)
    status['formation'] = formation_stage(root)['status']
    st = dev_stage(root)
    status['dev'] = st
    if st['status'] not in ('FINISHED', 'SKIPPED'):
        return stop('dev', st)
    status['rd01b_freeze'] = select_and_freeze_rd01b(root)['status']
    cat = freeze_catalog(root)
    status['catalog_cards'] = cat['routable_card_count']
    routes = routing_stage(root)
    status['routing'] = {j: r['routing_status'] for j, r in routes.items()}
    plan_test(root)
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
def _arm_record(bdir: Path, job: str, tokens: dict) -> dict:
    r = context.read_json(bdir / 'branch_result.json')
    cells = drs._cells(bdir, job)
    plan = r['committed_plan_id']
    idx = rd.load_index(bdir / job)
    beh = dsks._behavior(bdir)
    return {'status': r['status'], 'failure_kind': r['failure_kind'], 'commit': plan, 'fast_calls': r['calls'], 'tool_calls': r['tool_calls'],
            'new_evaluations': r['new_evaluations'], 'e': drs._vec(cells, plan, 'e'), 'e_shadow': drs._vec(cells, plan, 'e_shadow'), 'c_a': drs._vec(cells, plan, 'c_a'),
            'c_b': drs._vec(cells, plan, 'c_b'), 'e_mae': drs._vec(cells, plan, 'e_mae'),
            'material_spec': context.read_json(bdir / job / 'materials' / (plan + '__compiled.json'))['material_spec'] if plan else None,
            'material_totals': r2._material_totals(idx[plan]['summary_path']) if plan else None,
            'plans_in_branch': {m: {k: drs._vec(cells, m, k) for k in ('c_a', 'c_b', 'e')} for m in cells},
            'behavior': beh, 'fast_tokens': tokens.get('fast:%s' % bdir.name)}


def _rel(pair, base_mean):
    if not pair or base_mean is None or not math.isfinite(base_mean) or base_mean == 0:
        return None
    return pair['mean'] / base_mean


def package_result(root: Path = ROOT) -> dict:
    P = paths(root)
    res = {'package': 'DEV-DATA-READINESS-PROFILE-TRANSFER', 'exposure': rd.EXPOSURE, 'source': {}, 'formation': {}, 'dev': {}, 'catalog': None, 'routing': {},
           'test': {}, 'summary': {}, 'costs': {}}
    tok_src = dsks._unit_tokens(P['source']) if P['source'].exists() else {}
    for j in SOURCE_JOBS:
        b = P['source'] / ('%s_no_skill' % j)
        if (b / 'branch_result.json').exists():
            res['source'][j] = _arm_record(b, j, tok_src)
        elif (P['source'] / ('%s_not_scorable.json' % j)).exists():
            res['source'][j] = {'status': 'C_A_NOT_SCORABLE'}
    f = P['formation'] / 'RD01B'
    for name in ('propose', 'selection', 'frozen_skill'):
        if (f / (name + '.json')).exists():
            res['formation'][name] = context.read_json(f / (name + '.json'))
    tok_dev = dsks._unit_tokens(P['dev']) if P['dev'].exists() else {}
    for b in sorted(P['dev'].glob('RD01B_*_*')) if P['dev'].exists() else []:
        if b.is_dir() and (b / 'branch_result.json').exists():
            res['dev'][b.name] = _arm_record(b, '_'.join(b.name.split('_')[:2]), tok_dev)
    if (P['catalog'] / 'frozen_catalog.json').exists():
        res['catalog'] = context.read_json(P['catalog'] / 'frozen_catalog.json')
    for j in TEST_JOBS:
        if (P['routing'] / ('%s.json' % j)).exists():
            res['routing'][j] = context.read_json(P['routing'] / ('%s.json' % j))
    plan = context.read_json(P['freeze'] / 'test_plan.json') if (P['freeze'] / 'test_plan.json').exists() else None
    tok = dsks._unit_tokens(P['test']) if P['test'].exists() else {}
    rel_rows = {d: {} for d in DOMAINS}
    for job in TEST_JOBS:
        if not plan or not P['test'].exists():
            break
        row = {'arms': {}}
        if (P['test'] / ('%s_not_scorable.json' % job)).exists():
            row['status'] = 'C_A_NOT_SCORABLE'
            res['test'][job] = row
            continue
        for arm in TEST_ORDER[job]:
            b = P['test'] / ('%s_%s' % (job, arm))
            spec_ = plan['arms'][job][arm]
            if spec_['kind'] == 'reference':
                row['arms'][arm] = {'kind': 'reference', 'reference_arm': spec_['reference_arm'], 'status': spec_['status']}
            elif (b / 'branch_result.json').exists():
                row['arms'][arm] = _arm_record(b, job, tok)
            elif b.exists():
                fails = context.read_json(P['test'] / 'execution_finished.json')['failures'] if (P['test'] / 'execution_finished.json').exists() else []
                row['arms'][arm] = {'status': 'TECHNICAL_FAILURE', 'failure': next((f for f in fails if f.get('branch') == b.name), None),
                                    'note': 'attempted, ended without commit; not rerun and not replaced (see package_config_amendment_1.json)'}
            else:
                row['arms'][arm] = {'status': 'NOT_RUN'}
        e = {arm: (row['arms'][v['reference_arm']].get('e') if v.get('kind') == 'reference' else v.get('e')) for arm, v in row['arms'].items()}
        some = next((P['test'] / ('%s_%s' % (job, a)) for a in TEST_ORDER[job] if (P['test'] / ('%s_%s' % (job, a)) / job / 'e_scores.json').exists()), None)
        if some:
            cells = drs._cells(some, job)
            row['Baseline_Linear'] = {k: drs._vec(cells, 'Baseline_Linear', k) for k in ('e', 'e_shadow', 'c_a', 'c_b')}
            row['Fixed_Seasonal'] = {k: drs._vec(cells, 'Fixed_Seasonal', k) for k in ('e', 'e_shadow', 'c_a', 'c_b')}
            anyb = next(iter(cells.values()))
            row['e_status'] = sorted({str(v['e_status']) for v in anyb.values()})
            row['e_min_coverage'] = min((v['e_min_coverage'] for v in anyb.values() if v['e_min_coverage'] is not None), default=None)
            bl, fs = row['Baseline_Linear']['e'], row['Fixed_Seasonal']['e']
            blm = statistics.mean(bl) if bl else None
            row['main'] = {'delta_skill_F0_minus_F_known': drs._paired(e.get('F0'), e.get('F_known')),
                           'delta_routing_F_known_minus_F_match': drs._paired(e.get('F_known'), e.get('F_match')),
                           'delta_system_F0_minus_F_match': drs._paired(e.get('F0'), e.get('F_match'))}
            row['F_match_vs'] = {'F_generic_minus_F_match': drs._paired(e.get('F_generic'), e.get('F_match')), 'Random_minus_F_match': drs._paired(e.get('Random'), e.get('F_match')),
                                 'Baseline_Linear_minus_F_match': drs._paired(bl, e.get('F_match')), 'Fixed_Seasonal_minus_F_match': drs._paired(fs, e.get('F_match'))}
            row['relative_to_baseline_linear_e_mean'] = {'denominator': blm, **{k: _rel(v, blm) for k, v in {**row['main'], **row['F_match_vs']}.items()}}
            row['pipeline_vs_baseline_linear'] = {a: drs._paired(bl, v) for a, v in {**e, 'Fixed_Seasonal': fs}.items()}
            row['shadow_vs_baseline_linear'] = {a: drs._paired(bl, row['arms'][a].get('e_shadow')) for a in row['arms'] if row['arms'][a].get('e_shadow')}
            row['e_mean'] = {**{a: (statistics.mean(v) if v else None) for a, v in e.items()}, 'Baseline_Linear': blm, 'Fixed_Seasonal': statistics.mean(fs) if fs else None}
            rel_rows[ds_of(job)][job] = row['relative_to_baseline_linear_e_mean']
        row['routing'] = plan['routes'][job]
        res['test'][job] = row
    keys = ('delta_skill_F0_minus_F_known', 'delta_routing_F_known_minus_F_match', 'delta_system_F0_minus_F_match', 'F_generic_minus_F_match',
            'Random_minus_F_match', 'Baseline_Linear_minus_F_match', 'Fixed_Seasonal_minus_F_match')
    for k in keys:
        per_domain = {}
        for d in DOMAINS:
            vals = [v[k] for v in rel_rows[d].values() if v.get(k) is not None]
            per_domain[d] = {'jobs': {j: v.get(k) for j, v in rel_rows[d].items()}, 'equal_weight': statistics.mean(vals) if vals else None, 'n_jobs': len(vals)}
        dom = [v['equal_weight'] for v in per_domain.values() if v['equal_weight'] is not None]
        complete = all(v['n_jobs'] == 2 for v in per_domain.values())
        res['summary'][k] = {'relative_by_domain': per_domain, 'two_domain_equal_weight': statistics.mean(dom) if len(dom) == 2 else None,
                             'status': 'COMPLETE_FOUR_JOBS' if complete else 'PARTIAL'}
    led = context.read_json(P['ledger']) if P['ledger'].exists() else {}
    res['costs'] = {'ledger': {k: v for k, v in led.items() if k != 'events'}, 'stages': context.read_json(P['stages']) if P['stages'].exists() else {},
                    'tokens_by_unit': {s: dsks._unit_tokens(P[s]) for s in ('source', 'formation', 'dev', 'routing', 'test') if (P[s] / 'raw_responses').exists()}}
    context.write_json(root / 'result.json', res)
    return res


# ============================================================================= smoke (new risks; 0 real fits, 0 API, no label row of a planned job)
class _Scripted:
    """Per-unit scripted Fast responses; a list item that is an Exception is raised instead."""
    fatal = False

    def __init__(self, scripts, default):
        self.scripts, self.default, self.requests, self.count = scripts, default, [], {}
        self.last_text = None

    def call(self, role, unit, payload, system, max_tokens=4096, request_timeout=300.):
        self.requests.append({'role': role, 'unit': unit, 'payload': copy.deepcopy(payload), 'system': system})
        k = self.count[unit] = self.count.get(unit, 0) + 1
        script = self.scripts.get(unit, self.default)
        item = script[min(k, len(script)) - 1]
        if isinstance(item, Exception):
            raise item
        return {'actions': copy.deepcopy(item)} if role == 'fast' else copy.deepcopy(item)


def smoke(out: Path) -> dict:
    import tempfile
    import numpy as np
    from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    plan_p1 = {'default': {'steps': [{'op': 'hampel_filter', 'window': 7, 'n_sigmas': 3.0}]}, 'rules': [], 'rationale': 'smoke', 'observation_fields_used': []}
    fast_default = [[{'tool': 'inspect_data', 'arguments': {'entity_indices': [0], 'kind': 'gaps'}}],
                    [{'tool': 'build_material', 'arguments': {'plan_id': 'P1', 'policy': plan_p1}}, {'tool': 'evaluate', 'arguments': {'plan_id': 'P1'}}],
                    [{'tool': 'commit', 'arguments': {'plan_id': 'P1', 'reason': 'smoke commit'}}]]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td = Path(td)
        drs._register_synthetic()
        counter = {}
        orig_fit = rt.fit
        rt.fit = drs._fake_fit_factory(counter)
        inproc = lambda p, j, name, led: {'c_b': rd.open_c_b, 'freeze_e': rd.freeze_e, 'score_e': rd.score_e}[name](p, drs.ds_of(j), j)
        try:
            # 1. resume of a branch holding a built-but-unevaluated plan; same id evaluated once; ledger not reset; refused after labels
            caps = {'max_fit_attempts': 999, 'max_llm_requests': 999, 'max_llm_tokens': 10 ** 9, 'max_wall_s': 3600, 'max_retries': 0}
            ledger_path = td / 'r' / 'budget.json'
            (td / 'r').mkdir()
            led = rt.RuntimeLedger(ledger_path, **caps)
            started = led.s['started_epoch']
            common = td / 'r' / 'RDX_V1_common'
            common.mkdir()
            drs.ReadinessAdapter(common, 'RDX_V1', led, REPO).baselines()
            build_only = [[{'tool': 'build_material', 'arguments': {'plan_id': 'P1', 'policy': plan_p1}}], llm.TransportFault('smoke')]
            c1 = _Scripted({'RDX_V1_arm': build_only}, fast_default)
            b = td / 'r' / 'RDX_V1_arm'
            first = drs.fast_branch(b, 'RDX_V1', br.Knowledge(), led, c1, common)
            fits_before = counter.get('fits', 0)
            c2 = _Scripted({'RDX_V1_arm': [[{'tool': 'evaluate', 'arguments': {'plan_id': 'P1'}}], [{'tool': 'commit', 'arguments': {'plan_id': 'P1', 'reason': 'resumed'}}]]}, fast_default)
            resumed = drs.resume_fast_branch(b, 'RDX_V1', br.Knowledge(), led, c2)
            resumed_state = c2.requests[0]['payload']['candidates']
            rd.open_c_b(b, 'RDX', 'RDX_V1')
            refused = None
            try:
                drs.resume_fast_branch(b, 'RDX_V1', br.Knowledge(), led, c2)
            except RuntimeError as exc:
                refused = str(exc)
            checks['1_resume_built_unevaluated'] = {
                'ok': first.status == 'INCOMPLETE' and first.failure_kind == 'AGENT_CALL_FAILED' and first.new_evaluations == 0 and resumed.status == 'COMPLETE'
                and resumed.committed_plan_id == 'P1' and resumed.new_evaluations == 1 and counter.get('fits', 0) - fits_before == 3
                and [(c['plan_id'], c['trained']) for c in resumed_state] == [('Baseline_Linear', True), ('Fixed_Seasonal', True), ('P1', False)]
                and rt.RuntimeLedger(ledger_path, **caps).s['started_epoch'] == started and refused is not None
                and bool(rtp.labels_boundary_violations(td / 'r')) and (b / 'trace_before_resume.jsonl').exists(),
                'detail': {'first': [first.status, first.failure_kind], 'resumed': [resumed.status, resumed.committed_plan_id, resumed.new_evaluations],
                           'resumed_first_request_candidates': [(c['plan_id'], c['trained']) for c in resumed_state], 'fits_after_resume': counter.get('fits', 0) - fits_before,
                           'refused_after_labels': refused}}
            # 2. new cohort / cuts through the worker CLI, the cache binding and label prerequisites; old jobs unchanged
            old = {(d, s): rd.resolve_job(d, '%s_%s' % (d, s)).t for d in ('RD01', 'RD02') for s in ('S1', 'S2', 'V1', 'T1', 'T2')}
            old_ok = old == {(d, s): t for d in ('RD01', 'RD02') for s, t in {'S1': 3360, 'S2': 4560, 'V1': 5760, 'T1': 6960, 'T2': 8160}.items()} \
                and rd.resolve_job('RD02', 'RD02_T3').t == 9360 and rd.resolve_job('RD02', 'RD02_T4').t == 10560 and rd.JOB_T == {'S1': 3360, 'S2': 4560, 'V1': 5760, 'T1': 6960, 'T2': 8160}
            new_ok = all(rd.resolve_job(ds_of(j), j).t == t for j, (t, _, _) in T_TABLE.items())
            rej = {}
            for d, j in (('RD01B', 'RD01B_T1'), ('RD01', 'RD01_Q1'), ('RD02', 'RD02_Q3'), ('RD01', 'RD01B_S1'), ('RD01B', 'rd01b_s1')):
                try:
                    rd.resolve_job(d, j)
                    rej[j + '@' + d] = False
                except ValueError:
                    rej[j + '@' + d] = True
            cli = {}
            for d, j in (('RD01B', 'RD01B_Q1'), ('RD02', 'RD02_Q2'), ('RD01B', 'RD01B_T3')):
                pr = subprocess.run([sys.executable, '-B', '-m', rd.FIT_MODULE, '--stage', 'c_b', '--output', str(td / 'cli'), '--dataset', d, '--job', j],
                                    cwd=REPO, capture_output=True, text=True, timeout=300)
                cli[j] = pr.stderr.strip().splitlines()[-1][:160] if pr.stderr.strip() else 'rc=%d' % pr.returncode
            ctx = rd.open_job('RD01B', 'RD01B_S1', td / 'bind', 'material')
            with np.load(ctx.job_dir / 'scaler.npz') as z:
                binding = (str(z['dataset']), int(z['t']), [str(x) for x in z['roster']])
            model = td / 'bind' / 'm.pt'
            model.write_bytes(b'0')
            ref = rd.MaterialRef('BL', ctx.job.job_id, str(model), 'k', [[] for _ in rd.RD01B_ROSTER], None, str(model))
            good = {'status': 'OK', 'model_seed': 1, 'job_id': 'RD01B_S1', 'dataset': 'RD01B', 'roster': rd.RD01B_ROSTER, 'rows_read': list(ctx.job.evaluate_rows),
                    'material_id': 'BL', 'scores': {'c_a': {'status': 'SCORABLE'}}, 'model_path': str(model), 'cell_id': 'x'}
            cell = ctx.job_dir / 'cells' / (rd.train.cell_id('RD01B_S1', 'BL', 1) + '.json')
            cell.parent.mkdir(exist_ok=True)
            context.write_json(cell, good)
            cached = orig_fit(ctx, ref, (1,), rt.RuntimeLedger(td / 'bind' / 'b.json', **caps), REPO)
            wrong = {}
            for label, change in (('dataset', {'dataset': 'RD01'}), ('roster', {'roster': rd.RD01_ROSTER})):
                context.write_json(cell, {**good, **change})
                try:
                    orig_fit(ctx, ref, (1,), rt.RuntimeLedger(td / 'bind' / 'b.json', **caps), REPO)
                    wrong[label] = 'ACCEPTED'
                except RuntimeError as exc:
                    wrong[label] = str(exc)
            checks['2_new_cohort_and_cuts_bound_end_to_end'] = {
                'ok': old_ok and new_ok and all(rej.values()) and 'c_b prerequisite missing' in cli['RD01B_Q1'] and 'c_b prerequisite missing' in cli['RD02_Q2']
                and 'ValueError' in cli['RD01B_T3'] and binding == ('RD01B', 4560, rd.RD01B_ROSTER) and len(cached) == 1
                and all('another' in v for v in wrong.values()) and rd.DATASETS['RD01']['roster'] == rd.RD01_ROSTER,
                'detail': {'old_unchanged': old_ok, 'new_cuts': new_ok, 'rejected': rej, 'cli': cli, 'scaler_binding': binding[:2], 'cache_binding_errors': wrong}}
            # 3. profile: nulls/quantiles, 12 vs 32 populations, identity-free router request
            ov12 = rd.open_job('RDX', 'RDX_T1', td / 'p', 'material').overview()
            ov32 = copy.deepcopy(ov12)                   # same distribution, three times the population
            ov32['n_entities'] *= 3
            ov32['entities'] = ov32['entities'] * 3
            for v in ov32['summary'].values():
                if v:
                    v['n'] *= 3
            ov_null = copy.deepcopy(ov12)
            ov_null['summary']['lag168_corr'] = None
            p12, p32, pn = readiness_profile(ov12), readiness_profile(ov32), readiness_profile(ov_null)
            card = dsk.make_skill(skill_id='RDX-W1-r1', domain_id='RDX', revision=1, workflow='Inspect gaps; compare the two common programs; stop when differences are within seed noise.',
                                  principles=None, applicability_summary='hourly series with short holes', compatibility_note='readiness smoke', observable_applicability={'const': True},
                                  evidence_refs=['fixture:smoke'], legal_evidence_refs=['fixture:smoke'], source_stage='fixture', status='FROZEN_SELECTED',
                                  allowed_features=drs.ALLOWED_FEATURES, text_check=drs.skill_text_check)
            cat = {'cards': {'RD01B': {'profile': 'data_readiness', 'skill': replace(card, skill_id='RD01B-W1-r1', domain_id='RD01B').to_json()},
                             'RD02': {'profile': 'data_readiness', 'skill': replace(card, skill_id='RD02-C1-r2', domain_id='RD02', workflow='Other workflow text for smoke.').to_json()}},
                   'reference_profiles': {'RD01B': {'profile_version': PROFILE_VERSION, 'profiles': [p12, p32, p12]}, 'RD02': {'profile_version': PROFILE_VERSION, 'profiles': [p12, p12, p12]}}}
            router = _Scripted({}, [{'selected_skill_id': 'C2', 'status': 'SELECTED', 'evidence_fields': ['missing_fraction'], 'rationale': 'smoke'}])
            rec = route_job(ov12, cat, ('RD01B', 'RD02'), lambda p: router.call('router', 'q', p, dsk.ROUTER_SYSTEM), true_domain='RD02')
            raw = json.dumps(router.requests[0]['payload'], ensure_ascii=False)
            leaks = [s for s in ('RD01', 'RD02', 'RDX', 'n_entities', 'n_valid', '"n":', 'legal_parents_total', 'finite_mean', 'c_a', 'c_b', 'Workflow', 'Inspect gaps', 'entity_', 'train_rows')
                     if s in raw]
            checks['3_profile_nulls_population_identity_free_router'] = {
                'ok': set(p12['fields']) == set(ROUTER_FIELDS) and p12 == p32 and pn['fields']['lag168_corr'][:3] == [None, None, None] and pn['fields']['lag168_corr'][3] == 0.0
                and all(v[3] == 1.0 or 0 <= v[3] <= 1 for v in p12['fields'].values()) and not leaks and not rec['identity_scan']
                and rec['selected_domain'] == 'RD02' and rec['runtime_catalog_map']['C2']['domain'] == 'RD02' and rec['request_bytes'] <= ROUTER_MAX_BYTES
                and not _identity_scan(router.requests[0]['payload']),
                'detail': {'fields': list(p12['fields']), 'population_invariant': p12 == p32, 'null_field': pn['fields']['lag168_corr'], 'leaks': leaks,
                           'request_bytes': rec['request_bytes'], 'route': [rec['routing_status'], rec['selected_domain']]}}
            # 4. readiness scope; cross-profile refusal; abstain / scope failure / technical failure kept apart
            cond = replace(card, skill_id='RD02-C1-r2', domain_id='RD02', observable_applicability={'feature': 'batch_median_missing_fraction', 'op': '>', 'value': 0.5})
            feats = batch_features(ov12)
            sc_ready = dsk.scope_at(cond, feats, drs.ALLOWED_FEATURES)
            try:
                sc_aug = dsk.scope_at(cond, feats)
            except Exception as exc:  # noqa: BLE001
                sc_aug = {'state': 'REJECTED', 'error': type(exc).__name__}
            cat_scope = copy.deepcopy(cat)
            cat_scope['cards']['RD02']['skill'] = cond.to_json()
            outcomes = {}
            for label, reply in (('abstain', {'selected_skill_id': None, 'status': 'ABSTAIN', 'evidence_fields': [], 'rationale': 'insufficient'}),
                                 ('scope_fail', {'selected_skill_id': 'C2', 'status': 'SELECTED', 'evidence_fields': ['missing_fraction'], 'rationale': 'smoke'}),
                                 ('transport', llm.TransportFault('x')), ('format', {'selected_skill_id': 'C9', 'status': 'SELECTED', 'evidence_fields': ['missing_fraction'], 'rationale': 's'}),
                                 ('aug_field', {'selected_skill_id': 'C1', 'status': 'SELECTED', 'evidence_fields': ['nn1_input_distance'], 'rationale': 's'})):
                cl = _Scripted({}, [reply])
                r_ = route_job(ov12, cat_scope, ('RD01B', 'RD02'), lambda p, cl=cl: cl.call('router', 'q', p, dsk.ROUTER_SYSTEM), true_domain='RD02')
                outcomes[label] = (r_['routing_status'], r_['load'], r_['scope_state'])
            cross = copy.deepcopy(cat)
            cross['cards']['RD02'] = {'profile': 'augmentation', 'skill': cat['cards']['RD02']['skill']}
            try:
                route_job(ov12, cross, ('RD01B', 'RD02'), lambda p: None, true_domain='RD02')
                cross_refused = False
            except PermissionError:
                cross_refused = True
            empty = route_job(ov12, {'cards': {'RD01B': None, 'RD02': None}, 'reference_profiles': cat['reference_profiles']}, ('RD01B', 'RD02'),
                              lambda p: (_ for _ in ()).throw(AssertionError('no call without candidates')), true_domain='RD01B')
            checks['4_readiness_scope_cross_profile_and_outcome_kinds'] = {
                'ok': sc_ready['state'] == 'NO_MATCH' and sc_aug['state'] != 'NO_MATCH' and cross_refused and empty['routing_status'] == 'NO_CANDIDATE_SKILLS' and empty['llm_requests'] == 0
                and outcomes == {'abstain': ('ABSTAIN', 'NO_CARD', None), 'scope_fail': ('SELECTED_SCOPE_NOT_MATCHED', 'CARD', 'NO_MATCH'),
                                 'transport': ('ROUTING_FAILED', 'NO_CARD', None), 'format': ('ROUTING_FAILED', 'NO_CARD', None), 'aug_field': ('ROUTING_FAILED', 'NO_CARD', None)},
                'detail': {'readiness_scope': sc_ready, 'augmentation_vocabulary_scope': sc_aug, 'outcomes': outcomes, 'cross_profile_refused': cross_refused,
                           'no_candidates': empty['routing_status']}}
            # 5-6. synthetic package: catalog -> routing -> Test plan with identical executions -> Test worker (labels order), caps not reset
            proot = td / 'pkg'
            P = paths(proot)
            synth_jobs = ('RDX_T1', 'RDX_T2')
            proot.mkdir()
            rt.RuntimeLedger(P['ledger'], **TOTAL)
            caps_first = stage_caps(proot, 'test')
            caps_again = stage_caps(proot, 'test')
            known_card = replace(card, skill_id='RD02-C1-r2', domain_id='RD02')
            routes = {'RDX_T1': {'routing_status': 'SELECTED', 'selected_domain': 'RD02', 'selected_skill_id': 'RD02-C1-r2', 'scope_state': 'MATCH', 'load': 'CARD', 'selected_equals_true_domain': True},
                      'RDX_T2': {'routing_status': 'ABSTAIN', 'selected_domain': None, 'selected_skill_id': None, 'scope_state': None, 'load': 'NO_CARD', 'selected_equals_true_domain': None}}
            k0, kc, kg = br.json_copy(asdict(br.Knowledge())), _knowledge(known_card), br.json_copy(asdict(drs.generic_knowledge()))
            eq = {}
            arms = {}
            for job, order in (('RDX_T1', TEST_ORDER['RD02_Q1']), ('RDX_T2', TEST_ORDER['RD02_Q2'])):
                k = {'F0': k0, 'F_known': kc, 'F_match': kc if routes[job]['load'] == 'CARD' else k0, 'F_generic': kg}
                a, running = {}, []
                for arm in order:
                    if arm == 'Random':
                        a[arm] = {'kind': 'random', 'policy_seeds': [2026091801, 2026091802]}
                        continue
                    ref = next((x for x in running if arm in FAST_ARMS and x in FAST_ARMS and r2._same_policy(k[arm], k[x])), None)
                    a[arm] = {'kind': 'reference', 'status': 'IDENTICAL_EXECUTION_REFERENCE', 'reference_arm': ref} if ref else {'kind': 'fast', 'knowledge': k[arm]}
                    if not ref:
                        running.append(arm)
                arms[job] = a
                eq[job] = {x: v['reference_arm'] for x, v in a.items() if v['kind'] == 'reference'}
            tcfg = {'stage': 'test', 'output': str(P['test']), 'jobs': list(synth_jobs), 'order': {'RDX_T1': list(TEST_ORDER['RD02_Q1']), 'RDX_T2': list(TEST_ORDER['RD02_Q2'])},
                    'arms': br.json_copy(arms), 'parent_common': {}, 'labels_e': True, 'caps': caps, 'http_cap': 999, 'ledger_path': str(P['ledger']), 'fit_retry': False}
            context.write_json(td / 'test.json', tcfg)
            fast = _Scripted({}, fast_default)
            r2.stage_worker(td / 'test.json', client=fast, runner=inproc)
            units = [u for u in dict.fromkeys(r_['unit'] for r_ in fast.requests)]
            ran = [P['test'] / ('%s_%s' % (j, a)) for j in synth_jobs for a in arms[j] if arms[j][a]['kind'] != 'reference']
            commits = [b_ / b_.name[:6] / 'commit.json' for b_ in ran]
            cbs = [b_ / b_.name[:6] / 'c_b_scores.json' for b_ in ran]
            efz = [b_ / b_.name[:6] / 'e_frozen.json' for b_ in ran]
            esc = [b_ / b_.name[:6] / 'e_scores.json' for b_ in ran]
            barrier = P['test'] / 'all_e_predictions_frozen.json'
            caps_after = stage_caps(proot, 'test')
            checks['5_identical_execution_charged_once'] = {
                'ok': eq == {'RDX_T1': {'F_known': 'F_match'}, 'RDX_T2': {'F0': 'F_match'}},
                'detail': {'equivalence': eq, 'fast_units_run': units}}
            checks['5_identical_execution_charged_once']['ok'] = checks['5_identical_execution_charged_once']['ok'] and \
                units == ['RDX_T1_F_generic', 'RDX_T1_F_match', 'RDX_T1_F0', 'RDX_T2_F_match', 'RDX_T2_F_generic', 'RDX_T2_F_known'] and \
                all((P['test'] / ('%s_%s_treatment.json' % (j, a))).exists() for j in eq for a in eq[j]) and not any((P['test'] / ('%s_%s' % (j, a))).exists() for j in eq for a in eq[j])
            checks['6_synthetic_chain_labels_order_caps_not_reset'] = {
                'ok': all(p_.exists() for p_ in commits + cbs + efz + esc) and max(p_.stat().st_mtime for p_ in commits) <= min(p_.stat().st_mtime for p_ in cbs)
                and max(p_.stat().st_mtime for p_ in cbs) <= min(p_.stat().st_mtime for p_ in efz) and max(p_.stat().st_mtime for p_ in efz) <= barrier.stat().st_mtime <= min(p_.stat().st_mtime for p_ in esc)
                and caps_first == caps_again == caps_after and bool(rtp.labels_boundary_violations(P['test']))
                and context.read_json(P['test'] / 'RDX_T1_Random' / 'branch_result.json')['status'] == 'COMPLETE',
                'detail': {'branches_run': [b_.name for b_ in ran], 'caps_frozen': caps_first['caps'], 'resume_refused_after_labels': rtp.labels_boundary_violations(P['test'])[:2]}}
        finally:
            rt.fit = orig_fit
            rd.DATASETS.pop('RDX', None)
    res = {'status': 'PASS' if all(c['ok'] for c in checks.values()) else 'FAIL', 'checks': checks, 'real_fits': 0, 'api_calls': 0, 'label_rows_of_planned_jobs': 0}
    res['existing_controls'] = []
    for cmd in (['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/functional/test_batch_research.py'],
                ['-c', 'import sys; from pathlib import Path; from evaluation.main_protocol_p4 import batch_research_data_readiness as m; r=m.smoke(Path(sys.argv[1])); '
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
    ap.add_argument('--resume-stage', choices=['source', 'dev', 'test'])
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--continue-package', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--root', type=Path, default=ROOT)
    a = ap.parse_args()
    root = a.root.resolve()
    if a.stage_worker:
        r2.stage_worker(a.stage_worker.resolve())
    elif a.resume_stage:
        resume_stage(root, a.resume_stage, accept_unknown_usage=a.accept_unknown_usage)
        if a.continue_package:
            package_run(root)
    elif a.run:
        package_run(root)
    elif a.result:
        package_result(root)
    elif a.preflight:
        cfg = preflight(root)
        print(json.dumps({'t_eligibility': {j: {k: v for k, v in r.items() if k != 'geometry'} for j, r in cfg['t_eligibility'].items()},
                          'exposure': cfg['exposure_check']['status'], 'rd02_card': cfg['domains']['RD02']['card']['skill_id']}, ensure_ascii=False)[:3000])
    elif a.smoke:
        res = smoke(root / 'smoke')
        print('TRANSFER_SMOKE', res['status'], {k: v['ok'] for k, v in res['checks'].items()}, [c['returncode'] for c in res['existing_controls']], flush=True)
        if res['status'] != 'PASS':
            raise SystemExit(1)
    else:
        ap.error('choose --run, --stage-worker, --resume-stage, --result, --preflight or --smoke')


if __name__ == '__main__':
    from evaluation.main_protocol_p4 import batch_research_data_readiness_transfer as _canonical
    _canonical.main()
