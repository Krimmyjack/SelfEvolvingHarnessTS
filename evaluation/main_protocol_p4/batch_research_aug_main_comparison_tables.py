"""Main table / cost table / result.json of DEV-AUG-MAIN-COMPARISON (0 fits, 0 LLM): the parent DEV-AUG-OFFLINE-SKILL arms and references from
their frozen evaluations, the naive-card control when its line has closed, and AutoDA-Timeseries from this package. Pending columns stay empty."""
from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import context

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_aug_main_comparison'
PARENT = REPO / '_scratch' / 'dev_aug_offline_skill'
SCREEN = REPO / '_scratch' / 'dev_aug_task_family_screen'
DOMAINS = ('D01', 'D02', 'D03')
DNAME = {'D01': 'Electricity', 'D02': 'Traffic', 'D03': 'Solar'}
SEEDS = ('20269181', '20269182', '20269183')
REFS = {'None': 'None', 'NoMix': 'P_NoMixRecipe', 'Fixed_global_source': 'C_shock'}
FIXED_SOURCE = {'D01': 'C_censor', 'D02': 'C_censor', 'D03': 'P_NoMixRecipe'}
ARM_LABEL = {'None': 'None (no augmentation)', 'NoMix': 'NoMix preset (P_NoMixRecipe)', 'Fixed_source': 'Fixed_source (domain best on Source)',
             'Fixed_global_source': 'Fixed_global_source (Comp[shock])', 'f0': 'F0 zero-feedback agent, no card', 'f_naive': 'F_naive (task-knowledge card)',
             'f_shared': 'F_shared (learned shared card)', 'f_domain': 'F_domain (learned domain card)', 'autoda': 'AutoDA-Timeseries (adapted)'}
ORDER = ('None', 'NoMix', 'Fixed_source', 'Fixed_global_source', 'autoda', 'f0', 'f_naive', 'f_shared', 'f_domain')


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def _period(c):
    return c.split('_')[2]


def _group(c):
    return c.split('_')[3]


def boot_ci(diff: dict, B: int = 4000, seed: int = 2026092401) -> dict:
    """Same cluster bootstrap as the parent readout: within each domain resample periods and entity groups, domain mean, equal-weight domains."""
    rng = np.random.default_rng(seed)
    dom = {d: {c: v for c, v in diff.items() if c[:3] == d} for d in DOMAINS}
    vals = []
    for _ in range(B):
        ds = []
        for d, cc in dom.items():
            ps = sorted({_period(c) for c in cc})
            gs = sorted({_group(c) for c in cc})
            P = rng.choice(ps, len(ps), replace=True)
            G = rng.choice(gs, len(gs), replace=True)
            v = [cc['%s_Q_%s_%s' % (d, p, g)] for p in P for g in G if '%s_Q_%s_%s' % (d, p, g) in cc]
            ds.append(statistics.fmean(v) if v else statistics.fmean(cc.values()))
        vals.append(statistics.fmean(ds))
    return {'lo95': float(np.percentile(vals, 2.5)), 'hi95': float(np.percentile(vals, 97.5))}


def collect() -> dict:
    """{case: {arm: {'e_seed': {seed: e}, 'label': str, 'phys': str}}} over the 40 Test cases."""
    ev = context.read_json(PARENT / 'test' / 'evaluation.json')
    out = {}
    for case, per in ev['e_by_case_phys_seed'].items():
        d = case[:3]
        rec = {}
        for arm, phys in list(REFS.items()) + [('Fixed_source', FIXED_SOURCE[d])]:
            rec[arm] = {'e_seed': per[phys], 'phys': phys, 'label': arm}
        for arm in ('f0', 'f_shared', 'f_domain'):
            cm = ev['commits']['%s__%s' % (case, arm)]
            phys = cm['physical_material'] if cm else 'None'
            rec[arm] = {'e_seed': per[phys], 'phys': phys, 'label': cm['label'] if cm else 'INCOMPLETE'}
        out[case] = rec
    nv = PARENT / 'naive' / 'evaluation.json'
    if nv.exists():
        evn = context.read_json(nv)
        for case in out:
            cm = evn['commits'].get('%s__f_naive' % case)
            phys = cm['physical_material'] if cm else 'None'
            src = evn['e_by_case_phys_seed'][case] if phys in evn['e_by_case_phys_seed'][case] else ev['e_by_case_phys_seed'][case]
            out[case]['f_naive'] = {'e_seed': src[phys], 'phys': phys, 'label': cm['label'] if cm else 'INCOMPLETE', 'incomplete': cm is None}
    ad = ROOT / 'test' / 'autoda_test_e.json'
    if ad.exists():
        a = context.read_json(ad)
        for case in out:
            out[case]['autoda'] = {'e_seed': a['e_by_case_seed'][case], 'phys': 'autoda:%s' % a['config'], 'label': 'AutoDA-Timeseries (%s)' % a['config']}
    return out


def tables() -> dict:
    data = collect()
    arms = [a for a in ORDER if all(a in r for r in data.values())]
    em = {c: {a: statistics.fmean(float(r[a]['e_seed'][s]) for s in SEEDS) for a in arms} for c, r in data.items()}
    G = {c: {a: 100.0 * (em[c]['None'] - em[c][a]) / em[c]['None'] for a in arms} for c in em}

    def dom_mean(f):
        dm = {d: statistics.fmean(f(c) for c in em if c[:3] == d) for d in DOMAINS}
        return dm, statistics.fmean(dm.values())

    rows = []
    for a in arms:
        e_dm, e_all = dom_mean(lambda c: em[c][a])
        g_dm, g_all = dom_mean(lambda c: G[c][a])
        rows.append({'arm': a, 'label': ARM_LABEL[a], 'nmse_three_domain': e_all, **{'nmse_%s' % d: e_dm[d] for d in DOMAINS},
                     'G_vs_None_three_domain': g_all, **{'G_vs_None_%s' % d: g_dm[d] for d in DOMAINS}})
    pairs = [(x, y) for x, y in (('f_domain', 'f0'), ('f_shared', 'f0'), ('f_domain', 'f_shared'), ('f_domain', 'f_naive'), ('f_naive', 'f0'),
                                 ('autoda', 'None'), ('autoda', 'f0'), ('autoda', 'f_domain'), ('autoda', 'f_shared'), ('autoda', 'NoMix'),
                                 ('f_domain', 'NoMix'), ('f_domain', 'Fixed_source'), ('f_domain', 'Fixed_global_source')) if x in arms and y in arms]
    pair_rows = []
    for x, y in pairs:
        diff = {c: 100.0 * (em[c][y] - em[c][x]) / em[c]['None'] for c in em}
        dm = {d: statistics.fmean(v for c, v in diff.items() if c[:3] == d) for d in DOMAINS}
        same = sum(1 for c in em if data[c][x]['phys'] == data[c][y]['phys'])
        w = sum(1 for c in em if data[c][x]['phys'] != data[c][y]['phys'] and diff[c] > 1e-9)
        l = sum(1 for c in em if data[c][x]['phys'] != data[c][y]['phys'] and diff[c] < -1e-9)
        pair_rows.append({'pair': '%s - %s' % (x, y), 'three_domain': statistics.fmean(dm.values()), **{d: dm[d] for d in DOMAINS}, 'ci95': boot_ci(diff),
                          'wins_same_losses': [w, same, l], 'max_harm': min(diff.values())})
    res = {'package': 'DEV-AUG-MAIN-COMPARISON', 'written_local': now(), 'identity': 'supplementary comparison on the exposed Test cases of DEV-AUG-OFFLINE-SKILL',
           'arms_present': arms, 'pending': [a for a in ORDER if a not in arms], 'arms': rows, 'pairs': pair_rows,
           'per_case': {c: {a: {'e_mean': em[c][a], 'e_seed': data[c][a]['e_seed'], 'G_vs_None': G[c][a], 'delivered': data[c][a]['label']} for a in arms} for c in em},
           'unit_note': 'nmse = normalized MSE on E (mean over seeds, entity macro, cases then domains equal weight); G = pp of the case None mean, NOT a % error '
                        'reduction relative to the no-card agent'}
    context.write_json(ROOT / 'result.json', res)
    with open(ROOT / 'main_table.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['arm', 'label', 'nmse_3dom', 'nmse_D01', 'nmse_D02', 'nmse_D03', 'G_vs_None_3dom', 'G_D01', 'G_D02', 'G_D03'])
        for r in rows:
            w.writerow([r['arm'], r['label'], '%.5f' % r['nmse_three_domain']] + ['%.5f' % r['nmse_%s' % d] for d in DOMAINS] +
                       ['%.2f' % r['G_vs_None_three_domain']] + ['%.2f' % r['G_vs_None_%s' % d] for d in DOMAINS])
    L = ['# Main table (DEV-AUG-MAIN-COMPARISON, %s)' % now(), '',
         '40 Test cases of DEV-AUG-OFFLINE-SKILL (Electricity 16 / Traffic 16 / Solar 8), shared MLP 192->128->64->48, 3 seeds, E = four 48-h origins t+192..t+336. '
         'nMSE = normalized MSE (lower is better), cases then domains equal weight. G = pp of the case None mean (higher is better). Exposed development test; '
         'supplementary comparison. Pending: %s.' % (', '.join(res['pending']) or 'none'), '',
         '| Method | nMSE (3 dom) | Elec | Traffic | Solar | G vs None (3 dom) | Elec | Traffic | Solar |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        L.append('| %s | %.4f | %.4f | %.4f | %.4f | %+.2f | %+.2f | %+.2f | %+.2f |' % (r['label'], r['nmse_three_domain'], r['nmse_D01'], r['nmse_D02'], r['nmse_D03'],
                 r['G_vs_None_three_domain'], r['G_vs_None_D01'], r['G_vs_None_D02'], r['G_vs_None_D03']))
    L += ['', '## Paired differences (pp of None; positive = first better; 95% cluster CI over periods x entity groups)', '',
          '| Pair | 3 dom | Elec | Traffic | Solar | 95% CI | W / same / L | max harm |', '|---|---:|---:|---:|---:|---|---|---:|']
    for p in pair_rows:
        L.append('| %s | %+.2f | %+.2f | %+.2f | %+.2f | [%+.2f, %+.2f] | %s | %+.2f |' % (p['pair'], p['three_domain'], p['D01'], p['D02'], p['D03'], p['ci95']['lo95'],
                 p['ci95']['hi95'], '/'.join(map(str, p['wins_same_losses'])), p['max_harm']))
    (ROOT / 'main_table.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    return res


if __name__ == '__main__':
    r = tables()
    print('arms', r['arms_present'], 'pending', r['pending'])


def cost_table() -> str:
    """One-time learning cost, per-case decision cost and per-case Consumer training cost, from the ledgers and run records (0 fits)."""
    import glob
    led_a = context.read_json(PARENT / 'budget.json')
    led_bd = context.read_json(PARENT / 'budget_bd.json')
    led_n = context.read_json(PARENT / 'naive' / 'budget_naive.json') if (PARENT / 'naive' / 'budget_naive.json').exists() else None
    scr = context.read_json(SCREEN / 'ledger.json')
    wire = context.read_json(PARENT / 'wiring' / 'wiring_result.json')
    res = context.read_json(PARENT / 'result.json')
    stage_tok = {}
    for e in led_bd['events']:
        if e.get('kind') == 'llm_finished':
            r = stage_tok.setdefault(e.get('stage'), [0, 0])
            r[0] += 1
            r[1] += (e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0)
    fits_sel = sum(1 for e in led_bd['events'] if e.get('kind') == 'fit_started' and '_V_' in e.get('cell', ''))
    fit_secs = [e['seconds'] for e in led_bd['events'] if e.get('kind') == 'fit_finished' and e.get('ok')]
    mlp_fit_s = statistics.fmean(fit_secs)
    beh = res['behaviour']
    ad_recs = [context.read_json(Path(p)) for p in glob.glob(str(ROOT / 'test' / '*__autoda.json'))]
    ad_src = [context.read_json(Path(p)) for p in glob.glob(str(ROOT / 'source' / '*__autoda.json'))]
    ad_s = statistics.fmean(r['train_seconds'] for r in ad_recs) if ad_recs else None
    ad_feat = statistics.fmean(r.get('feature_seconds', 0) for r in ad_recs) if ad_recs else None
    naive_tok = None
    if (PARENT / 'naive' / 'result_naive.json').exists():
        naive_tok = context.read_json(PARENT / 'naive' / 'result_naive.json')['cost']['tokens_per_naive_trajectory']['mean']
    L = ['# Cost table (DEV-AUG-MAIN-COMPARISON, %s)' % now(), '',
         'Fit = one Consumer training (2000 updates, CPU, 8 threads per process). Wall seconds are measured per fit process (parallel runs share the host).', '',
         '## One-time (learning / selection)', '', '| Item | LLM requests | Tokens | Fits | Notes |', '|---|---:|---:|---:|---|',
         '| Screen: 12 fixed programs x 38 Source cases x 3 seeds | 0 | 0 | %d | reused as scripted records and as the Source cache; fit wall sum %.0f s |' % (scr['fits_ok'], scr.get('fit_seconds', 0)),
         '| Stage A: 24 no-card Source trajectories + evaluation | %d | %.2fM | %d | incl. 3 wiring fits |' % (led_a['llm_requests'], (led_a['llm_tokens_in'] + led_a['llm_tokens_out']) / 1e6, led_a['fit_attempts'] + wire['fits']),
         '| Formation: 8 Slow calls (+1 correction) | %d | %.2fM | 0 | |' % (stage_tok.get('slow', [0, 0])[0], stage_tok.get('slow', [0, 0])[1] / 1e6),
         '| Select: 96 zero-feedback trajectories + evaluation | %d | %.2fM | %d | |' % (stage_tok.get('select', [0, 0])[0], stage_tok.get('select', [0, 0])[1] / 1e6, fits_sel)]
    if led_n:
        card_tok = sum((e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0) for e in led_n['events'] if e.get('kind') == 'llm_finished' and str(e.get('unit', '')).startswith('naive_'))
        L.append('| Naive cards (3 domains, one generation each, no selection) | 6 | %.2fM | 0 | control arm |' % (card_tok / 1e6))
    if ad_src:
        L.append('| AutoDA config selection (3 configs x 9 Source cases x 1 seed) | 0 | 0 | %d | mean %.0f s per joint fit |' % (len(ad_src), statistics.fmean(r['train_seconds'] for r in ad_src)))
    L += ['', '## Per new case (deployment)', '', '| Method | Agent tokens / trajectory | Agent calls | Consumer fits | Mean wall s per fit | Notes |', '|---|---:|---:|---:|---:|---|']
    for arm, lab in (('f0', 'F0 no card'), ('f_shared', 'F_shared'), ('f_domain', 'F_domain')):
        b = beh[arm]
        L.append('| %s | %.0f | %.1f | 3 | %.1f | plan construction by the agent; training after commit |' % (lab, b['tokens']['mean'], b['calls']['mean'], mlp_fit_s))
    if naive_tok:
        L.append('| F_naive | %.0f | — | 3 | %.1f | |' % (naive_tok, mlp_fit_s))
    L.append('| Fixed programs (NoMix / Fixed_source / Fixed_global) | 0 | 0 | 3 | %.1f | no decision cost; choice made once on Source |' % mlp_fit_s)
    if ad_s:
        L.append('| AutoDA-Timeseries | 0 | 0 | 3 joint (policy + MLP) | %.1f | incl. Catch22 features %.1f s; %.1fx the plain fit |' % (ad_s, ad_feat, ad_s / mlp_fit_s))
    else:
        L.append('| AutoDA-Timeseries | 0 | 0 | 3 joint (policy + MLP) | pending | |')
    txt = '\n'.join(L) + '\n'
    (ROOT / 'cost_table.md').write_text(txt, encoding='utf-8')
    return txt
