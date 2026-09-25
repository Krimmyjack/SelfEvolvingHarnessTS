"""Read-only development analysis; standard library, no Consumer/LLM calls.

Run from the repository root:
    python3 idea-stage/observation_mechanism/analyze.py

Only cached losses and pre-origin contexts of cached development cases are used.
The already reported Hampel threshold 6 is examined, never tuned here.
Local operator footprints are transparent Python translations of s1_outlier.py;
they are diagnostics, not a replacement production executor or a new policy.
"""
import io
import json
import math
import statistics as st
import zipfile
from collections import defaultdict
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROGRAMS = ('ANCESTOR', 'W1_hampel_filter', 'W2_pmc_then_outlier_mad')
FACES = ('support_face', 'delayed_face')
CONTEXT = 192


def robust(x):
    x = [v for v in x if math.isfinite(v)]
    c = st.median(x)
    s = 1.4826 * st.median(abs(v - c) for v in x)
    if s <= 1e-12:
        s = st.pstdev(x)
    if s <= 1e-12:
        s = 1.0
    return c, max(s, 1e-8)


def interpolate(x):
    y = list(x)
    known = [i for i, v in enumerate(x) if math.isfinite(v)]
    assert known
    for i in range(known[0]):
        y[i] = x[known[0]]
    for left, right in zip(known, known[1:]):
        for i in range(left + 1, right):
            y[i] = x[left] + (x[right] - x[left]) * ((i-left)/(right-left))
    for i in range(known[-1] + 1, len(x)):
        y[i] = x[known[-1]]
    return y


def rolling_median(x):
    # Width 7; numpy.pad(mode='symmetric') duplicates the boundary point.
    padded = list(reversed(x[:3])) + x + list(reversed(x[-3:]))
    return [st.median(padded[i:i+7]) for i in range(len(x))]


def card(x):
    y = interpolate(x)
    c, _ = robust(y)
    mad = st.median(abs(v-c) for v in y)
    bound = 3.5 * 1.4826 * mad
    mad_hits = [mad > 1e-12 and abs(v-c) > bound for v in y]
    med = rolling_median(y)
    local_mad = rolling_median([abs(v-m) for v, m in zip(y, med)])
    hampel_hits = [m > 0 and abs(v-c0) > 3*1.4826*m
                   for v, c0, m in zip(y, med, local_mad)]
    rc, rs = robust(x)
    _, ts = robust(x[-48:])
    peak = max(abs(v-rc) for v in x if math.isfinite(v)) / ts
    rz = [abs(v-rc)/rs if math.isfinite(v) else None for v in x]
    gaps = [i for i,v in enumerate(x) if not math.isfinite(v)]
    enough = sum(sum(i-k*24 >= 0 and math.isfinite(x[i-k*24])
                     for k in (1,2,3)) >= 2 for i in gaps)
    return {
        'spike_peak_over_tail_sd': peak,
        'missing_points': len(gaps),
        'gaps_with_two_original_prior_phase_donors': enough,
        'mad_hits': sum(mad_hits),
        'hampel_hits': sum(hampel_hits),
        'hampel_hits_observed': sum(h and z is not None for h,z in zip(hampel_hits,rz)),
        'hampel_hits_observed_below_global_z3': sum(h and z is not None and z < 3
                                                 for h,z in zip(hampel_hits,rz)),
    }


def block(p):
    return '[0:40]' if p <= 6 else '[40:80]' if p <= 15 else '[80:120]' if p <= 17 else '[120:160]'


def summarize(g):
    return {'n': len(g), 'mean_gain': st.mean(g) if g else None,
            'helped': sum(v > 0 for v in g), 'material_harmed': sum(v < -.005 for v in g),
            'severe_harmed': sum(v < -.30 for v in g),
            'max_harm': max([0.] + [-v for v in g])}


def run():
    store = json.loads((ROOT/'_scratch/m_r0k_prediction_store.json').read_text())
    entries = [e for e in store.values() if e['program'] in PROGRAMS and e['verifier_passed']]
    needed = defaultdict(set)
    for e in entries:
        assert e['origin'] < 4056
        for u in e['eval_uids']:
            needed[u].add(e['origin'])
    cards = {}
    archive = ROOT/'data/kdd2018/raw/kdd_cup_2018_dataset_with_missing_values.zip'
    with zipfile.ZipFile(archive) as z:
        with z.open(z.namelist()[0]) as f:
            for line in io.TextIOWrapper(f, encoding='utf-8', errors='replace'):
                uid = line.split(':',1)[0]
                if uid not in needed:
                    continue
                # Suffix tokens are discarded without numeric parsing.
                tokens = islice(line.rstrip().rsplit(':',1)[-1].split(','), max(needed[uid]))
                prefix = [float('nan') if s.strip() == '?' else float(s) for s in tokens]
                for o in needed[uid]:
                    assert o <= len(prefix)
                    cards[o,uid] = card(prefix[o-CONTEXT:o])
    gains = {}
    for e in entries:
        deg = set(e.get('degenerate_uids',[]))
        for u,r,p in zip(e['eval_uids'],e['raw_per_view'],e['program_per_view']):
            if u not in deg:
                gains[e['program'],e['position'],e['face'],u] = r-p
    out = {'boundary': {'consumer_fits':0,'llm_calls':0,'horizon_values_parsed':0,
                       'max_numeric_index':max(o for o,u in cards)-1,
                       'context_cards':len(cards),'uids':len(needed),'new_hashes':0},
           'evidence': 'EXPOSED_DEVELOPMENT_DIAGNOSTIC; no prospective/generalization claim',
           'footprint_implementation': 'stdlib translation of current operators; no production-executor equality assertion',
           'all_programs':{},'unchanged_serving_context':{},'hampel_rule_6':{},
           'r4a_rule_support_reproduction':{},'program_relative_to_ancestor':{}}
    r4a=json.loads((ROOT/'artifacts/main_protocol/r4a_pattern_identifiability.json').read_text())
    for face in FACES:
        face_entries=[e for e in entries if e['face']==face and e['program']=='ANCESTOR']
        cases=[(e['position'],e['origin'],u) for e in face_entries for u in e['eval_uids']]
        for prog in PROGRAMS:
            out['all_programs'][prog+'|'+face]=summarize([gains[prog,p,face,u] for p,o,u in cases])
            if prog!='ANCESTOR':
                out['program_relative_to_ancestor'][prog+'|'+face]=summarize([
                    gains[prog,p,face,u]-gains['ANCESTOR',p,face,u] for p,o,u in cases])
        for prog,feature in [('ANCESTOR','mad_hits'),('W1_hampel_filter','hampel_hits')]:
            zero=[gains[prog,p,face,u] for p,o,u in cases if cards[o,u][feature]==0]
            out['unchanged_serving_context'][prog+'|'+face]=summarize(zero)
        per_block={}
        for b in sorted({block(p) for p,o,u in cases}):
            subset=[(p,o,u) for p,o,u in cases if block(p)==b]
            selected=[cards[o,u]['spike_peak_over_tail_sd']>=6 for p,o,u in subset]
            hampel=[gains['W1_hampel_filter',p,face,u] for p,o,u in subset]
            old=[gains['ANCESTOR',p,face,u] for p,o,u in subset]
            rule=[g if s else 0 for g,s in zip(hampel,selected)]
            per_block[b]={'n':len(subset),'coverage':st.mean(selected),
                          'rule_gain':st.mean(rule),'hampel_all_gain':st.mean(hampel),
                          'ancestor_all_gain':st.mean(old),
                          'delta_vs_hampel_all':st.mean(rule)-st.mean(hampel),
                          'delta_vs_ancestor_all':st.mean(rule)-st.mean(old)}
            if face=='support_face':
                ref=r4a['main_table']['W1_hampel_filter|support_face']['per_feature']['spike_peak_over_tail_sd']['helped']['stump']['folds'][b]
                delta=abs(ref['utility_rule']-st.mean(rule))
                assert delta < 5.1e-7, (b,delta)
                out['r4a_rule_support_reproduction'][b]=delta
        unit_rows=[]
        for e in face_entries:
            p,o=e['position'],e['origin']
            selected=[cards[o,u]['spike_peak_over_tail_sd']>=6 for u in e['eval_uids']]
            rule=[gains['W1_hampel_filter',p,face,u] if s else 0
                  for u,s in zip(e['eval_uids'],selected)]
            q=summarize(rule);q['position']=p;q['treated']=sum(selected)
            q['numeric_screen_pass']=q['mean_gain']>=.005 and q['material_harmed']/q['n']<=.2 and q['max_harm']<=.3 and q['treated']>=5
            unit_rows.append(q)
        out['hampel_rule_6'][face]={'folds':per_block,
            'macro_delta_vs_hampel_all':st.mean(v['delta_vs_hampel_all'] for v in per_block.values()),
            'macro_delta_vs_ancestor_all':st.mean(v['delta_vs_ancestor_all'] for v in per_block.values()),
            'numeric_screen_pass_units':sum(r['numeric_screen_pass'] for r in unit_rows),'units':unit_rows,
            'interpretation':'fixed threshold 6; exposed-data shadow comparison; numeric screen is not lifecycle approval; ancestor-all is not necessarily the legal incumbent'}
    out['operator_observation_mismatch']={
        'observed_hampel_hits':sum(c['hampel_hits_observed'] for c in cards.values()),
        'observed_hampel_hits_below_global_z3':sum(c['hampel_hits_observed_below_global_z3'] for c in cards.values()),
        'context_gaps':sum(c['missing_points'] for c in cards.values()),
        'context_gaps_with_two_original_prior_phase_donors':sum(c['gaps_with_two_original_prior_phase_donors'] for c in cards.values()),
        'note':'overlapping context counts; not independent point samples'}
    (HERE/'readout.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k!='hampel_rule_6'},ensure_ascii=False,indent=2))
    print(json.dumps({f:{k:v for k,v in s.items() if k!='units'} for f,s in out['hampel_rule_6'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    run()
