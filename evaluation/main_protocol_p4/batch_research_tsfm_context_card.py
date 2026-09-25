"""DEV-TSFM-CONTEXT-CARD (docs/DEV_TSFM_CONTEXT_CARD_TASK_2026-09-24.md): can offline experience make a zero-feedback Agent prepare the inference
context of a zero-shot TSFM (Time-MoE-50M) more reliably?

Action per case: one (context length, normalization) of the cached context screen (lengths 192 / 336 / 672 / 1344 / 2688, normalization
'instance' or 'tscale'); scored by the cached zero-shot E of that variant (server numbers, read-only), never re-run. Source: the no-card Agent's
real choices; Slow: two shared and two domain cards from the Source choice experience and the scored menu; Select: J picks and freezes; Test:
no card vs shared card vs domain card, with the 192 baseline, the Source-best uniform 672 and the Source-frozen per-domain choice as references.
No downstream number ever reaches a Fast request.

  --smoke | --run | --readout | --status
"""
from __future__ import annotations

import os

import argparse
import json
import re
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import context
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_aug_offline_skill as OS
from evaluation.main_protocol_p4 import batch_research_aug_patchtst_offline_skill as PTS
from evaluation.main_protocol_p4 import batch_research_aug_tsfm_offline_skill as TS
from evaluation.main_protocol_p4 import batch_research_domain_aug_decision_priority as DP

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_tsfm_context_card'
PULL = REPO / '_scratch' / 'dev_aug_tsfm_native_server_pull'       # cached zero-shot context screen (server numbers; read-only)
PARENT = OS.ROOT                                                     # case split and T scalers (read-only)
OBS_COMMON = PTS.ROOT / 'common'                                     # prepared cases for the T observation summary (read-only)
FRESH = REPO / '_scratch' / 'dev_tsfm_context_fresh'                 # never-scored forecast origins of the Test entity groups (T-only scalers)
TASK = 'docs/DEV_TSFM_CONTEXT_CARD_TASK_2026-09-24.md'
PACKAGE = 'DEV-TSFM-CONTEXT-CARD'
IDENTITY = 'EXPOSED_DEVELOPMENT_REVALIDATION (every stage\'s context-screen readings were exposed; this package tests the Agent decision only)'
DOMAINS = OS.DOMAINS
LENGTHS = [192, 336, 672, 1344, 2688]
NORMS = ('instance', 'tscale')
VARIANTS = ['%s_%d' % (n, L) for L in LENGTHS for n in NORMS]
BASE = 'instance_192'
UNIFORM = 'instance_672'                                              # the Source-best variant every case can run (Source +25.85)
STAGE_FILE = {'source': 'context_screen.json', 'select': 'context_screen_select.json', 'test': 'context_screen_test.json', 'fresh': 'context_screen_fresh.json'}
CAPS = {'tokens': 12_000_000, 'logical_requests': 1000, 'http': 1200, 'extra_transport': 200, 'wall_s': 12 * 3600}
HTTP = int(os.environ.get('SEH_HTTP', '6'))
FAST_MAX_CALLS = 3
FAST_MAX_TOKENS = 2000
CARD_LIMIT = 1500
SLOTS = (1, 2)
SCOPES = ['shared'] + list(DOMAINS)
TEST_ARMS = ('f0', 'f_shared', 'f_domain')
NAIVE_ARM = 'f_naive'                                        # control added 2026-09-25 (user): generic knowledge, no Source outcome

now = DP.now
SafeLedger = DP.SafeLedger


def read(p):
    return context.read_json(Path(p))


def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    context.write_json(Path(p), obj)


def cases_by_stage() -> dict:
    return read(PARENT / 'frozen_config.json')['cases']


def stage_cases(stage: str) -> list:
    if stage == 'fresh':
        return [c for d in DOMAINS for c in sorted(read(FRESH / 'plan.json')['cases']['fresh'][d])]
    return [c for d in DOMAINS for c in sorted(cases_by_stage()[stage][d])]


def common_of(case: str) -> Path:
    return FRESH / 'common' if '_N_' in case else PARENT / 'common'


_CSS = None


def css() -> dict:
    global _CSS
    if _CSS is None:
        _CSS = ec.cases_from_split(read(PARENT / 'split_copy.json'))
        if (FRESH / 'split_copy.json').exists():
            _CSS.update(ec.cases_from_split(read(FRESH / 'split_copy.json')))
    return _CSS


_SCORES = {}


def scores(stage: str) -> dict:
    if stage not in _SCORES:
        _SCORES[stage] = {c: r['E'] for c, r in read(PULL / STAGE_FILE[stage])['cases'].items()}
    return _SCORES[stage]


def G(e, e0):
    return 100.0 * (e0 - e) / e0


# ============================================================================= what every Fast sees
CONSUMER = {'model': 'Time-MoE-50M, a pretrained decoder-only time-series foundation model (mixture of experts); used zero-shot: it is not trained or '
                     'tuned on this case, its weights never change',
            'forecast': 'four 48-hour forecasts of every one of the 16 entities, at the first forecast origin t0 and at t0+48, t0+96, t0+144 hours; each '
                        'forecast reads the most recent L hours before its own origin (never anything at or after the origin)',
            'context_length': 'L hours of the entity\'s own hourly history; the model accepts up to 4096 points',
            'normalization': {'instance': 'each context window is normalized by its own mean and standard deviation before the model and the forecast is '
                                          'mapped back (the model\'s official convention)',
                              'tscale': 'each context window is scaled with the per-entity mean and standard deviation of the case\'s training segment T '
                                        '(the 672 hours that end 192 hours before t0) and fed as is'},
            'default': 'L = 192 with instance normalization is the default preparation'}
FAST_SYSTEM = ('You prepare the inference context of a pretrained time-series forecasting model for one case of 16 related series. This is a '
               'zero-feedback deployment: no forecast accuracy or downstream result is available to you now or later in this case. Decide from the '
               'case card, the observations and, if you request it, the history profile. If a knowledge card is given, it holds experience learned '
               'offline from other cases; treat it as guidance and check it against this case\'s observations. Reply with ONE JSON object and '
               'nothing else, either {"tool": "history_profile", "arguments": {}} or {"tool": "commit", "arguments": {"length": <an available '
               'length from the menu>, "normalization": "instance" or "tscale", "reason": "<brief, evidence-based>"}}. You have at most %d requests '
               'and the last one must commit.' % FAST_MAX_CALLS)


def history_profile(case: str) -> dict:
    """Weekly blocks of the entities' own history before the first forecast origin (T-scaled), most recent first; nothing at or after t0."""
    p = ROOT / 'obs' / ('%s__history.json' % case)
    if p.exists():
        return read(p)
    cs = css()[case]
    first = min(cs.e)
    weeks = min(max(LENGTHS), first) // 168
    with np.load(common_of(case) / case / 'scaler.npz') as z:
        mean, scale = z['mean'].copy(), z['scale'].copy()
    sl = ec.load_case_slice(cs, first - weeks * 168, first)
    q = lambda v: [float('%.3g' % x) for x in np.quantile(v, [0.25, 0.5, 0.75])]
    blocks = []
    for k in range(1, weeks + 1):
        raw = sl.rows(first - k * 168, first - (k - 1) * 168)
        zv = (raw - mean[None, :]) / scale[None, :]
        daily = zv.reshape(7, 24, -1).mean(axis=1)
        blocks.append({'weeks_before_t0': k, 'mean_q25_q50_q75': q(zv.mean(axis=0)), 'std_q25_q50_q75': q(zv.std(axis=0)),
                       'day_to_day_range_q25_q50_q75': q(daily.max(axis=0) - daily.min(axis=0)), 'zero_fraction_median': float('%.3g' % np.median((raw == 0).mean(axis=0)))})
    out = {'unit': 'values scaled by the per-entity mean / std of the training segment T; quantiles over the 16 entities',
           'weeks_available': weeks, 'blocks_most_recent_first': blocks}
    write(p, out)
    return out


def case_card(case: str) -> dict:
    p = ROOT / 'obs' / ('%s__card.json' % case)
    if p.exists():
        return read(p)
    cs = css()[case]
    first = min(cs.e)
    card = {'domain_id': case[:3], 'entities': ec.COHORT_SIZE, 'consumer': CONSUMER, 'available_history_hours_before_t0': int(first),
            'menu': [{'length': L, 'normalization': n, 'available': first >= L} for L in LENGTHS for n in NORMS],
            'training_segment_observations': {'note': 'quantiles (q10, q25, q50, q75, q90) over the 16 entities of T-only statistics; field meanings in field_definitions',
                                              'fields': OS._obs_summary(FRESH / 'common' if '_N_' in case else OBS_COMMON, cs)},
            'field_definitions': {k: v for k, v in OS.FIELD_DEFINITIONS.items() if k in OS.OBS_FIELDS_ZF}}
    write(p, card)
    return card


# ============================================================================= one zero-feedback trajectory
def parse_action(text: str, case: str, calls_left: int) -> dict:
    t = (text or '').strip()
    t = re.sub(r'^```(?:json)?|```$', '', t, flags=re.M).strip()
    a = json.loads(t)
    if not isinstance(a, dict) or a.get('tool') not in ('history_profile', 'commit'):
        raise ValueError('reply with one JSON object whose "tool" is "history_profile" or "commit"')
    if a['tool'] == 'history_profile':
        if calls_left <= 1:
            raise ValueError('this is the last request: it must commit')
        return a
    arg = a.get('arguments') or {}
    L, n = arg.get('length'), arg.get('normalization')
    if not isinstance(L, int) or L not in LENGTHS or n not in NORMS:
        raise ValueError('commit needs "length" in %s and "normalization" in %s' % (LENGTHS, list(NORMS)))
    if min(css()[case].e) < L:
        raise ValueError('length %d is not available for this case' % L)
    if not str(arg.get('reason') or '').strip():
        raise ValueError('commit needs a brief "reason"')
    return a


def run_trajectory(bdir: Path, case: str, card_body, client, stage: str, arm: str) -> dict:
    rp = bdir / 'result.json'
    if rp.exists():
        r = read(rp)
        if r['status'] != 'CALL_FAILED':
            return r
    bdir.mkdir(parents=True, exist_ok=True)
    payload = {'case': case_card(case), 'knowledge_card': card_body, 'transcript': []}
    trace, calls, corrections, tokens = [], 0, 0, 0
    status, choice, reason = 'INCOMPLETE', None, None
    while calls < FAST_MAX_CALLS:
        payload['requests_left_including_this'] = FAST_MAX_CALLS - calls
        try:
            text, rec = client.call('fast', bdir.name, payload, FAST_SYSTEM, max_tokens=FAST_MAX_TOKENS, meta={'stage': stage, 'case': case, 'arm': arm, 'call': calls + 1},
                                    request_timeout=300.0)
        except (rt.llm.AccountFault, rt.llm.TransportFault, OS.UnknownUsageBlock) as exc:
            status = 'CALL_FAILED'
            trace.append({'call': calls + 1, 'fault': type(exc).__name__, 'detail': str(exc)[:200]})
            break
        calls += 1
        tokens += (rec.get('prompt_tokens') or 0) + (rec.get('completion_tokens') or 0)
        try:
            act = parse_action(text, case, FAST_MAX_CALLS - calls + 1)
        except (ValueError, json.JSONDecodeError) as exc:
            trace.append({'call': calls, 'response_excerpt': (text or '')[:600], 'rejected': str(exc)[:300]})
            if corrections >= 1:
                continue
            corrections += 1
            payload['transcript'].append({'your_previous_reply_rejected': str(exc)[:300]})
            continue
        trace.append({'call': calls, 'action': act})
        if act['tool'] == 'history_profile':
            payload['transcript'].append({'tool': 'history_profile', 'output': history_profile(case)})
            continue
        status, choice, reason = 'COMPLETE', '%s_%d' % (act['arguments']['normalization'], act['arguments']['length']), act['arguments']['reason']
        break
    r = {'case': case, 'arm': arm, 'stage': stage, 'status': status, 'choice': choice, 'delivered': choice if status == 'COMPLETE' else BASE, 'reason': reason,
         'calls': calls, 'corrections': corrections, 'used_history_profile': any(t.get('action', {}).get('tool') == 'history_profile' for t in trace),
         'tokens': tokens, 'trace': trace, 'finished_local': now()}
    write(rp, r)
    print('TRAJ', stage, bdir.name, status, choice, 'calls', calls, 'tokens', tokens, flush=True)
    return r


def run_stage(stage: str, jobs: list, card_of, client, summary_name: str = 'dispatch_summary.json') -> dict:
    """jobs: [(case, arm)]; HTTP threads; call failures are retried once after TS.RESUME_WAIT_S, then the stage stops (a resume continues)."""
    sdir = ROOT / stage / 'branches'
    results, lock = {}, threading.Lock()

    def go(todo):
        q = list(todo)
        qlock = threading.Lock()

        def worker():
            while True:
                with qlock:
                    if not q:
                        return
                    case, arm = q.pop(0)
                r = run_trajectory(sdir / ('%s__%s' % (case, arm)), case, card_of(case, arm), client, stage, arm)
                with lock:
                    results[(case, arm)] = r
        ts = [threading.Thread(target=worker, daemon=True) for _ in range(HTTP)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    go(jobs)
    failed = [k for k, r in results.items() if r['status'] == 'CALL_FAILED']
    if failed:
        print('TRANSPORT_RESUME_WAIT', stage, len(failed), 'trajectories', TS.RESUME_WAIT_S, 's', flush=True)
        time.sleep(TS.RESUME_WAIT_S)
        go(failed)
    failed = ['%s__%s' % k for k, r in results.items() if r['status'] == 'CALL_FAILED']
    summ = {'stage': stage, 'finished_local': now(), 'jobs': len(jobs), 'complete': sum(r['status'] == 'COMPLETE' for r in results.values()),
            'incomplete': sum(r['status'] == 'INCOMPLETE' for r in results.values()), 'call_failed': failed}
    write(ROOT / stage / summary_name, summ)
    print('STAGE_DONE', json.dumps(summ), flush=True)
    if failed:
        raise RuntimeError('%s: transport-interrupted trajectories %s; resume after the fault is cleared' % (stage, failed[:5]))
    return results


# ============================================================================= Slow: two shared and two domain cards from the Source experience
SLOW_SYSTEM = ('You write a knowledge card for a zero-feedback agent that prepares the inference context (context length and normalization) of a '
               'pretrained time-series foundation model used zero-shot. The census holds Source cases. For each: its training-segment observations, '
               'the history available before the first forecast origin, a compact history profile, what the no-card agent did (its requests, choice '
               'and reason) and SCRIPTED RECORDS: the zero-shot error of every menu variant on that case, as pp improvement over the default '
               '(instance, 192); positive = better. The scripted records come from an evaluator run, not from an agent, and a deployed agent never '
               'sees any such number. Write guidance that a deployed agent can apply from the observations it will actually see: which variant to '
               'prefer by default, which observable conditions change the choice, and what to avoid. Do not name cases, do not quote case-specific '
               'numbers as rules unless they are stable across the census, do not claim more certainty than the census supports. KEEP (no card) is '
               'allowed if no guidance would help. Reply with ONE JSON object: {"decision": "PROPOSE" or "KEEP", "card": {"title": "<short>", '
               '"body": "<at most %d characters>"}}.' % CARD_LIMIT)
SLOT_FRAME = {1: 'Write a short, rule-first card: the default, then at most three observable conditions that change it.',
              2: 'Write a card that states the decisive observations, the default, the exceptions and the known failure modes of the no-card agent.'}


def census(scope: str) -> dict:
    cases = [c for c in stage_cases('source') if scope == 'shared' or c[:3] == scope]
    sc = scores('source')
    rows = []
    for c in cases:
        r = read(ROOT / 'source' / 'branches' / ('%s__f0' % c) / 'result.json')
        hp = history_profile(c)
        rows.append({'domain_id': c[:3], 'available_history_hours_before_t0': int(min(css()[c].e)),
                     'training_segment_observations': case_card(c)['training_segment_observations']['fields'],
                     'history_profile_recent_weeks': hp['blocks_most_recent_first'][:4], 'history_weeks_available': hp['weeks_available'],
                     'no_card_agent': {'used_history_profile': r['used_history_profile'], 'choice': r['delivered'], 'status': r['status'], 'reason': r['reason']},
                     'scripted_records_pp_vs_default': {v: float('%.3g' % G(sc[c][v], sc[c][BASE])) for v in VARIANTS if v in sc[c]},
                     'no_card_choice_pp_vs_default': float('%.3g' % G(sc[c][r['delivered']], sc[c][BASE]))})
    return {'scope': scope, 'n_cases': len(rows), 'menu': VARIANTS, 'default': BASE, 'consumer': CONSUMER, 'cases': rows}


def validate_card(obj) -> dict:
    if not isinstance(obj, dict) or obj.get('decision') not in ('PROPOSE', 'KEEP'):
        raise ValueError('reply with {"decision": "PROPOSE" or "KEEP", "card": {...}}')
    if obj['decision'] == 'KEEP':
        return {'status': 'NO_CARD', 'body': None, 'title': None}
    card = obj.get('card') or {}
    body, title = str(card.get('body') or '').strip(), str(card.get('title') or '').strip()
    if not body or len(body) > CARD_LIMIT:
        raise ValueError('card.body must be non-empty and at most %d characters (got %d)' % (CARD_LIMIT, len(body)))
    if re.search(r'D0\d_[A-Z]', body):
        raise ValueError('the card must not name cases')
    return {'status': 'PROPOSED', 'body': body, 'title': title}


def slow_stage(client) -> dict:
    out = ROOT / 'slow' / 'proposals.json'
    if out.exists():
        return read(out)
    res, lock = {}, threading.Lock()

    def one(scope, k):
        p = ROOT / 'slow' / ('%s_%d.json' % (scope, k))
        if p.exists():
            r = read(p)
        else:
            payload = {'census': census(scope), 'framing': SLOT_FRAME[k]}
            r, attempts = None, []
            for i in range(2):
                try:
                    text, rec = client.call('slow', '%s_%d' % (scope, k), payload, SLOW_SYSTEM, max_tokens=4000, meta={'scope': scope, 'slot': k}, request_timeout=600.0)
                except (rt.llm.AccountFault, rt.llm.TransportFault, OS.UnknownUsageBlock) as exc:
                    r = {'status': 'CALL_FAILED', 'fault': type(exc).__name__}
                    break
                try:
                    v = validate_card(json.loads(re.sub(r'^```(?:json)?|```$', '', (text or '').strip(), flags=re.M).strip()))
                    r = {**v, 'attempts': i + 1, 'raw': text}
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    attempts.append(str(exc)[:300])
                    payload = {**payload, 'correction': {'previous_output_excerpt': (text or '')[:3000], 'error': str(exc)[:300],
                                                         'instruction': 'Correct this format / contract error only; KEEP is allowed.'}}
            if r is None:
                r = {'status': 'PARSE_OR_VALIDATION_FAILED', 'errors': attempts}
            r.update({'scope': scope, 'slot': k, 'written_local': now()})
            if r['status'] != 'CALL_FAILED':
                write(p, r)
        with lock:
            res['%s_%d' % (scope, k)] = r
        print('SLOW', scope, k, r['status'], flush=True)
    ts = [threading.Thread(target=one, args=(s, k), daemon=True) for s in SCOPES for k in SLOTS]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if any(r['status'] == 'CALL_FAILED' for r in res.values()):
        raise RuntimeError('a Slow call failed technically; resume after the fault is cleared')
    summ = {'written_local': now(), 'cards': {k: {'status': r['status'], 'chars': len(r['body']) if r.get('body') else 0} for k, r in res.items()}}
    write(out, summ)
    return summ


def card_body(scope: str, slot: int):
    r = read(ROOT / 'slow' / ('%s_%d.json' % (scope, slot)))
    return r['body'] if r['status'] == 'PROPOSED' else None


NAIVE_SYSTEM = ('You write a knowledge card for a zero-feedback agent that prepares the inference context (context length and normalization) of a '
                'pretrained time-series foundation model used zero-shot. No downstream result of any kind exists for you: you see the deployment '
                'description (model, menu, default, the history profile tool) and the observations of eight development cases of one domain (their '
                'training-segment statistics and the weekly history profile before the first forecast origin). Using your general knowledge of '
                'forecasting and of such models, write guidance the agent can apply from the observations it will see: the default choice, the '
                'observable conditions that change it, what to avoid. Do not name cases. KEEP (no card) is allowed. Reply with ONE JSON object: '
                '{"decision": "PROPOSE" or "KEEP", "card": {"title": "<short>", "body": "<at most %d characters>"}}.' % CARD_LIMIT)


def naive_payload(domain: str) -> dict:
    cases = [c for c in stage_cases('source') if c[:3] == domain]
    obs = []
    for c in cases:
        hp = history_profile(c)
        obs.append({'available_history_hours_before_t0': int(min(css()[c].e)), 'training_segment_observations': case_card(c)['training_segment_observations']['fields'],
                    'history_profile_recent_weeks': hp['blocks_most_recent_first'][:4], 'history_weeks_available': hp['weeks_available']})
    return {'domain_id': domain, 'deployment': {'consumer': CONSUMER, 'menu': VARIANTS, 'default': BASE, 'agent_tools': ['history_profile', 'commit'],
                                                'field_definitions': case_card(cases[0])['field_definitions']},
            'development_observations': {'note': 'observations only; no forecast of any variant was scored for you', 'cases': obs}}


def naive_stage(client) -> dict:
    res = {}
    for d in DOMAINS:
        p = ROOT / 'naive' / ('%s.json' % d)
        if p.exists():
            res[d] = read(p)
            continue
        payload = naive_payload(d)
        low = json.dumps(payload)
        if re.search(r'"(E|G_[a-z_]*|nmse|scripted_records[a-z_]*|pp_vs_default|no_card_agent)"', low):
            raise PermissionError('naive payload carries a downstream number or agent outcome')
        r = None
        for i in range(2):
            try:
                text, rec = client.call('slow_naive', 'naive_%s' % d, payload, NAIVE_SYSTEM, max_tokens=4000, meta={'scope': d}, request_timeout=600.0)
            except (rt.llm.AccountFault, rt.llm.TransportFault, OS.UnknownUsageBlock) as exc:
                raise RuntimeError('a naive card call failed technically (%s); resume after the fault is cleared' % type(exc).__name__)
            try:
                r = {**validate_card(json.loads(re.sub(r'^```(?:json)?|```$', '', (text or '').strip(), flags=re.M).strip())), 'attempts': i + 1, 'raw': text}
                break
            except (ValueError, json.JSONDecodeError) as exc:
                payload = {**payload, 'correction': {'previous_output_excerpt': (text or '')[:3000], 'error': str(exc)[:300],
                                                     'instruction': 'Correct this format / contract error only; KEEP is allowed.'}}
        r = r or {'status': 'PARSE_OR_VALIDATION_FAILED', 'body': None}
        r.update({'domain': d, 'written_local': now()})
        write(p, r)
        res[d] = r
        print('NAIVE', d, r['status'], flush=True)
    return res


def run_naive() -> None:
    """The naive-card control on the Test cases (cards once; one f_naive trajectory per Test case); then the readout (extended)."""
    DP.set_pools(2, HTTP)
    with DP.PackageLock(ROOT, 'run'):
        led = SafeLedger(ROOT / 'budget.json', max_fit_attempts=1, max_llm_requests=CAPS['logical_requests'], max_llm_tokens=CAPS['tokens'],
                         max_wall_s=CAPS['wall_s'], max_retries=0)
        naive_stage(make_client(led, 'naive'))
        cards = {d: read(ROOT / 'naive' / ('%s.json' % d)).get('body') for d in DOMAINS}
        run_stage('test', [(c, NAIVE_ARM) for c in stage_cases('test')], lambda c, a: cards[c[:3]], make_client(led, 'test'), summary_name='dispatch_summary_naive.json')
        readout()
        print('NAIVE_CONTROL_COMPLETE', flush=True)


# ============================================================================= Select and Test
def select_jobs() -> list:
    jobs = []
    for c in stage_cases('select'):
        jobs += [(c, 's%d' % k) for k in SLOTS] + [(c, 'd%d' % k) for k in SLOTS]
    return jobs


def select_card(case, arm):
    return card_body('shared' if arm[0] == 's' else case[:3], int(arm[1]))


def selection() -> dict:
    out = ROOT / 'freeze' / 'cards_frozen.json'
    if out.exists():
        return read(out)
    sc = scores('select')
    J, tok = {}, {}
    for c, arm in select_jobs():
        r = read(ROOT / 'select' / 'branches' / ('%s__%s' % (c, arm)) / 'result.json')
        J.setdefault(arm, {}).setdefault(c[:3], []).append(sc[c][r['delivered']] / sc[c][BASE])
        tok.setdefault(arm, []).append(r['tokens'])
    Jd = {a: {d: statistics.fmean(v) for d, v in dd.items()} for a, dd in J.items()}
    mt = {a: statistics.fmean(v) for a, v in tok.items()}
    s_best = min(['s%d' % k for k in SLOTS], key=lambda a: (statistics.fmean(Jd[a].values()), mt[a], a))
    rec = {'frozen_local': now(), 'rule': 'J = mean over Select cases of E(delivered) / E(default); domain card per domain argmin; shared card argmin of the '
                                          'three-domain mean; exact ties -> fewer mean tokens -> lower slot', 'J': Jd, 'mean_tokens': mt,
           'shared': {'arm': s_best, 'body': card_body('shared', int(s_best[1]))}, 'domain': {}}
    for d in DOMAINS:
        b = min(['d%d' % k for k in SLOTS], key=lambda a: (Jd[a][d], mt[a], a))
        rec['domain'][d] = {'arm': b, 'body': card_body(d, int(b[1]))}
    write(out, rec)
    return rec


def test_card(case, arm):
    if arm == 'f0':
        return None
    fr = read(ROOT / 'freeze' / 'cards_frozen.json')
    return fr['shared']['body'] if arm == 'f_shared' else fr['domain'][case[:3]]['body']


def test_jobs() -> list:
    return [(c, a) for c in stage_cases('test') for a in TEST_ARMS]


# ============================================================================= controller, readout
def make_client(led, stage):
    return TS.ZFStreamClient(led, ROOT / stage / 'llm', http_cap=CAPS['http'], stage=stage, incidents=ROOT / 'incidents', extra_cap=CAPS['extra_transport'])


def freeze_plan() -> dict:
    p = ROOT / 'plan.json'
    if p.exists():
        return read(p)
    plan = {'package': PACKAGE, 'task': TASK, 'identity': IDENTITY, 'frozen_local': now(), 'variants': VARIANTS, 'default': BASE,
            'references': {'default': BASE, 'uniform_source_best': UNIFORM, 'fixed_domain': 'context_choice_source.json (Source-frozen per-domain choice)'},
            'cases': {st: len(stage_cases(st)) for st in ('source', 'select', 'test')}, 'fast': {'system': FAST_SYSTEM, 'max_calls': FAST_MAX_CALLS,
            'max_tokens': FAST_MAX_TOKENS, 'consumer': CONSUMER}, 'slow': {'system': SLOW_SYSTEM, 'slots': SLOT_FRAME, 'scopes': SCOPES, 'card_limit': CARD_LIMIT},
            'caps': CAPS, 'http_in_flight': HTTP, 'model': OS.MODEL, 'scores_source': {st: str((PULL / f).relative_to(REPO)) for st, f in STAGE_FILE.items()},
            'incomplete_rule': 'no legal commit -> the default (instance, 192) is delivered and counted separately',
            'same_for_all_arms': 'history access, menu, Consumer description, request budget; only the knowledge card differs'}
    write(p, plan)
    return plan


def run() -> None:
    freeze_plan()
    DP.set_pools(2, HTTP)
    for st in ('source', 'select', 'test'):
        for c in stage_cases(st):
            case_card(c)
            history_profile(c)
    with DP.PackageLock(ROOT, 'run'):
        led = SafeLedger(ROOT / 'budget.json', max_fit_attempts=1, max_llm_requests=CAPS['logical_requests'], max_llm_tokens=CAPS['tokens'],
                         max_wall_s=CAPS['wall_s'], max_retries=0)

        def status(phase, **kw):
            write(ROOT / 'status.json', {'status': 'RUNNING', 'phase': phase, 'pid': os.getpid(), 'updated': now(),
                                         'ledger': {k: led.s.get(k) for k in ('llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown')}, **kw})
        try:
            status('source')
            run_stage('source', [(c, 'f0') for c in stage_cases('source')], lambda c, a: None, make_client(led, 'source'))
            status('slow')
            slow_stage(make_client(led, 'slow'))
            status('select')
            run_stage('select', select_jobs(), select_card, make_client(led, 'select'))
            fr = selection()
            print('SELECTED', json.dumps({'shared': fr['shared']['arm'], **{d: fr['domain'][d]['arm'] for d in DOMAINS}}), flush=True)
            status('test')
            run_stage('test', test_jobs(), test_card, make_client(led, 'test'))
        except Exception as exc:  # noqa: BLE001
            write(ROOT / 'status.json', {'status': 'STOPPED_ERROR', 'error': repr(exc)[:800], 'updated': now()})
            raise
        res = readout()
        write(ROOT / 'status.json', {'status': 'COMPLETE', 'finished_local': now(), 'main': {k: v['three_domain'] for k, v in res['arms_vs_default'].items()}})
        print('COMPLETE', flush=True)


def _period(c):
    return OS._period(c)


def readout(stage: str = 'test') -> dict:
    sc = scores(stage)
    choice = read(PULL / 'context_choice_source.json')['domain']
    per = {}
    arms_t = list(TEST_ARMS) + ([NAIVE_ARM] if all((ROOT / stage / 'branches' / ('%s__%s' % (c, NAIVE_ARM)) / 'result.json').exists() for c in stage_cases(stage)) else [])
    for c in stage_cases(stage):
        rs = {a: read(ROOT / stage / 'branches' / ('%s__%s' % (c, a)) / 'result.json') for a in arms_t}
        e = {a: sc[c][rs[a]['delivered']] for a in arms_t}
        e.update({'default': sc[c][BASE], 'uniform_672': sc[c][UNIFORM], 'fixed_domain': sc[c][choice[c[:3]]['choice']]})
        rec = {'domain': c[:3], 'period': _period(c), 'delivered': {a: rs[a]['delivered'] for a in arms_t}, 'status': {a: rs[a]['status'] for a in arms_t},
               'E': e, 'G_vs_default': {k: G(v, e['default']) for k, v in e.items()}, 'tokens': {a: rs[a]['tokens'] for a in arms_t},
               'calls': {a: rs[a]['calls'] for a in arms_t}}
        extra = ((('f_domain', NAIVE_ARM), (NAIVE_ARM, 'f0'), ('f_shared', NAIVE_ARM), (NAIVE_ARM, 'uniform_672'), (NAIVE_ARM, 'fixed_domain'))
                 if NAIVE_ARM in arms_t else ())
        for a, b in (('f_domain', 'f0'), ('f_shared', 'f0'), ('f_domain', 'f_shared'), ('f0', 'fixed_domain'), ('f_domain', 'fixed_domain'),
                     ('f_shared', 'fixed_domain'), ('f0', 'uniform_672'), ('f_domain', 'uniform_672'), ('f_shared', 'uniform_672')) + extra:
            rec['%s_minus_%s' % (a, b)] = G(e[a], e['default']) - G(e[b], e['default'])
        per[c] = rec

    def agg(key, sel=None):
        dm = {d: statistics.fmean(sel(per[c]) if sel else per[c][key] for c in per if per[c]['domain'] == d) for d in DOMAINS}
        return {'three_domain': statistics.fmean(dm.values()), 'by_domain': dm}
    arms = list(arms_t) + ['uniform_672', 'fixed_domain']
    arms_vs = {a: {**agg(None, lambda r, a=a: r['G_vs_default'][a]), 'wins_vs_default': sum(r['G_vs_default'][a] > 0 for r in per.values())} for a in arms}
    pairs = {}
    for k in [k for k in next(iter(per.values())) if '_minus_' in k]:
        a, b = k.split('_minus_')
        same = sum(1 for r in per.values() if abs(r[k]) < 1e-12)
        pairs[k] = {**agg(k), 'wins_same_losses': [sum(r[k] > 1e-12 for r in per.values()), same, sum(r[k] < -1e-12 for r in per.values())],
                    'cluster_bootstrap_95': OS.boot_ci({c.replace('_N_', '_Q_', 1): r for c, r in per.items()}, k)}   # boot_ci keys cases as D0x_Q_<period>_<group>
    beh = {}
    for st, arm_list in (('source', ['f0']), ('select', ['s1', 's2', 'd1', 'd2']), (stage, list(arms_t))):
        for a in arm_list:
            rs = [read(p) for p in (ROOT / st / 'branches').glob('*__%s/result.json' % a)]
            if rs:
                beh['%s:%s' % (st, a)] = {'n': len(rs), 'choices': dict(sorted(OS.collections_counter(r['delivered'] for r in rs).items())),
                                          'incomplete': sum(r['status'] != 'COMPLETE' for r in rs), 'used_history_profile': sum(r['used_history_profile'] for r in rs),
                                          'mean_calls': statistics.fmean(r['calls'] for r in rs), 'mean_tokens': statistics.fmean(r['tokens'] for r in rs)}
    src = scores('source')
    f0s = {c: read(ROOT / 'source' / 'branches' / ('%s__f0' % c) / 'result.json')['delivered'] for c in stage_cases('source')}
    source_f0 = {d: statistics.fmean(G(src[c][f0s[c]], src[c][BASE]) for c in f0s if c[:3] == d) for d in DOMAINS}
    led = read(ROOT / 'budget.json')
    res = {'package': PACKAGE, 'identity': IDENTITY, 'written_local': now(), 'cards': read(ROOT / 'freeze' / 'cards_frozen.json'),
           'arms_vs_default': arms_vs, 'pairs': pairs, 'behaviour': beh, 'source_no_card_G_by_domain': source_f0,
           'source_no_card_G_three_domain': statistics.fmean(source_f0.values()), 'per_case': per,
           'costs': {'requests': led['llm_requests'], 'http': led['llm_http_attempts'], 'tokens_in': led['llm_tokens_in'], 'tokens_out': led['llm_tokens_out'],
                     'unknown_usage': led['llm_tokens_unknown']}}
    res['stage'] = stage
    suffix = '' if stage == 'test' else '_' + stage
    write(ROOT / ('result%s.json' % suffix), res)
    (ROOT / ('REPORT%s.md' % suffix)).write_text(report_md(res), encoding='utf-8')
    return res


def run_fresh() -> None:
    """The frozen cards (and the no-card agent, and the naive cards when they exist) deployed on the never-scored fresh origins."""
    DP.set_pools(2, HTTP)
    cases = stage_cases('fresh')
    for c in cases:
        case_card(c)
        history_profile(c)
    arms = list(TEST_ARMS) + ([NAIVE_ARM] if all((ROOT / 'naive' / ('%s.json' % d)).exists() for d in DOMAINS) else [])
    naive = {d: read(ROOT / 'naive' / ('%s.json' % d)).get('body') for d in DOMAINS} if NAIVE_ARM in arms else {}
    card = lambda c, a: naive[c[:3]] if a == NAIVE_ARM else test_card(c, a)
    with DP.PackageLock(ROOT, 'run'):
        led = SafeLedger(ROOT / 'budget.json', max_fit_attempts=1, max_llm_requests=CAPS['logical_requests'], max_llm_tokens=CAPS['tokens'],
                         max_wall_s=CAPS['wall_s'], max_retries=0)
        run_stage('fresh', [(c, a) for c in cases for a in arms], card, make_client(led, 'fresh'))
        if (PULL / STAGE_FILE['fresh']).exists():
            readout('fresh')
        print('FRESH_COMPLETE', flush=True)


def report_md(r: dict) -> str:
    f = lambda x: '%+.2f' % x
    stage = r.get('stage', 'test')
    head = {'test': 'Test（40 例，以 192 点、逐窗口标准化为基线的 pp，正 = 更好）',
            'fresh': '新起点留出（40 例：Test 实体组在从未评分过的新预测起点；卡、选择、参照全部冻结；以 192 点、逐窗口标准化为基线的 pp）'}[stage]
    L = ['# TSFM 上下文准备卡（DEV-TSFM-CONTEXT-CARD）' + ('' if stage == 'test' else '：新起点留出验证'), '',
         '身份：%s。评分全部来自缓存的零样本上下文筛查，没有重新推理。' % (r['identity'] if stage == 'test' else 'FRESH_ORIGIN_HOLDOUT（新预测起点的读数在部署全部冻结后才读取）'), '',
         '## 1. ' + head, '', '| 臂 / 参照 | 三域 | 电力 | 交通 | 太阳能 | 胜基线 |', '|---|---:|---:|---:|---:|---:|']
    for a, v in r['arms_vs_default'].items():
        L.append('| %s | %s | %s | %s | %s | %d/40 |' % (a, f(v['three_domain']), *[f(v['by_domain'][d]) for d in DOMAINS], v['wins_vs_default']))
    L += ['', '## 2. 成对差值（pp；胜 / 同 / 负；聚类 bootstrap 95%）', '', '| 比较 | 三域 | 电力 | 交通 | 太阳能 | 胜/同/负 | 95% 区间 |', '|---|---:|---:|---:|---:|---|---|']
    for k, v in r['pairs'].items():
        ci = v['cluster_bootstrap_95']
        L.append('| %s | %s | %s | %s | %s | %s | [%s, %s] |' % (k, f(v['three_domain']), *[f(v['by_domain'][d]) for d in DOMAINS], '/'.join(map(str, v['wins_same_losses'])),
                                                          f(ci['lo95']), f(ci['hi95'])))
    L += ['', '## 3. 行为（选择分布、请求次数、token）', '']
    for k, v in r['behaviour'].items():
        L.append('- %s：n=%d，INCOMPLETE %d，调用历史概况 %d 次，平均请求 %.2f，平均 token %.0f；选择 %s' % (k, v['n'], v['incomplete'], v['used_history_profile'], v['mean_calls'],
                                                                                 v['mean_tokens'], json.dumps(v['choices'])))
    L += ['', 'Source 上无卡 Agent 的选择相对基线：三域 %s（%s）。' % (f(r['source_no_card_G_three_domain']), json.dumps({d: round(x, 2) for d, x in r['source_no_card_G_by_domain'].items()})),
          '', '## 4. 选中的卡', '', '- 共享卡：%s' % r['cards']['shared']['arm']]
    for d in DOMAINS:
        L.append('- %s：%s' % (d, r['cards']['domain'][d]['arm']))
    c = r['costs']
    L += ['', '## 5. 成本', '', '- 逻辑请求 %d，HTTP %d，token %d 输入 / %d 输出，未知用量 %d。' % (c['requests'], c['http'], c['tokens_in'], c['tokens_out'], c['unknown_usage'])]
    return '\n'.join(L) + '\n'


def smoke() -> dict:
    """Scripted client, one Source case: card / history build, no score in any Fast payload, commit parsing, INCOMPLETE default, census size."""
    freeze_plan()
    checks = []

    def check(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})
        print('SMOKE', 'OK ' if ok else 'FAIL', name, info if not ok else '', flush=True)
    for st in ('source', 'select', 'test'):
        check('scores_%s' % st, len(scores(st)) == len(stage_cases(st)) and all(BASE in v for v in scores(st).values()), n=len(scores(st)))
    case = stage_cases('source')[0]
    seen = []

    class FC:
        def __init__(self, replies):
            self.replies = list(replies)

        def call(self, role, unit, payload, system, *, max_tokens, meta, request_timeout):
            seen.append(json.dumps(payload))
            return self.replies.pop(0), {'request': len(seen), 'prompt_tokens': 10, 'completion_tokens': 5}
    sm = ROOT / 'smoke'
    import shutil
    if sm.exists():
        shutil.rmtree(sm)
    r1 = run_trajectory(sm / 'a', case, None, FC(['{"tool": "history_profile", "arguments": {}}',
                                                  '{"tool": "commit", "arguments": {"length": 672, "normalization": "instance", "reason": "smoke"}}']), 'smoke', 'f0')
    check('commit_after_profile', r1['status'] == 'COMPLETE' and r1['delivered'] == 'instance_672' and r1['used_history_profile'] and r1['calls'] == 2)
    r2 = run_trajectory(sm / 'b', case, 'card text', FC(['not json', '{"tool": "history_profile", "arguments": {}}', '{"tool": "history_profile", "arguments": {}}']),
                        'smoke', 'f_domain')
    check('incomplete_defaults', r2['status'] == 'INCOMPLETE' and r2['delivered'] == BASE and r2['corrections'] == 1)
    txt = ' '.join(seen)
    check('no_scores_in_fast_payloads', not re.search(r'"(E|G_[a-z_]*|nmse|scripted_records[a-z_]*|pp_vs_default)"', txt) and 'context_screen' not in txt)
    check('card_reaches_fast', any('"knowledge_card": "card text"' in s for s in seen))
    hp = history_profile(case)
    check('history_before_t0_only', hp['weeks_available'] * 168 <= min(css()[case].e), weeks=hp['weeks_available'])
    for p in (sm / 'a', sm / 'b'):
        (p / 'result.json').unlink()
    res = {'written_local': now(), 'checks': checks, 'passed': sum(c['ok'] for c in checks), 'total': len(checks)}
    write(sm / 'smoke_result.json', res)
    print('SMOKE_DONE', res['passed'], '/', res['total'], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    for a in ('smoke', 'run', 'readout', 'status', 'naive', 'fresh', 'readout_fresh'):
        ap.add_argument('--' + a.replace('_', '-'), action='store_true')
    a = ap.parse_args()
    if a.smoke:
        smoke()
    if a.run:
        run()
    if a.readout:
        readout()
    if a.naive:
        run_naive()
    if a.fresh:
        run_fresh()
    if a.readout_fresh:
        readout('fresh')
    if a.status:
        print(json.dumps(read(ROOT / 'status.json'), indent=1, ensure_ascii=False) if (ROOT / 'status.json').exists() else '{}')


if __name__ == '__main__':
    main()
