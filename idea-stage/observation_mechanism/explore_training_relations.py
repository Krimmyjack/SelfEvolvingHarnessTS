"""Exploratory relations in already exposed development results; zero fits.

Run: python3 idea-stage/observation_mechanism/explore_training_relations.py
Reads numeric raw data only before index 900, the frozen training frontier.
No candidate selection, thresholds for acceptance, or confirmatory claims.
"""
import io
import json
import math
import statistics as st
import zipfile
from collections import defaultdict
from itertools import islice
from pathlib import Path

from analyze import interpolate, robust

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ANCHORS = tuple(range(312, 853, 60))
ANC = 'ANCESTOR'
W2 = 'W2_pmc_then_outlier_mad'


def corr(x, y):
    if len(x) < 3:
        return None
    mx, my = st.mean(x), st.mean(y)
    xx, yy = [v-mx for v in x], [v-my for v in y]
    den = math.sqrt(sum(v*v for v in xx)*sum(v*v for v in yy))
    return sum(a*b for a, b in zip(xx, yy))/den if den > 1e-15 else None


def ranks(x):
    order = sorted(range(len(x)), key=x.__getitem__)
    out = [0.0]*len(x)
    left = 0
    while left < len(x):
        right = left+1
        while right < len(x) and x[order[right]] == x[order[left]]:
            right += 1
        for i in order[left:right]:
            out[i] = (left+right-1)/2
        left = right
    return out


def rho(x, y):
    return corr(ranks(x), ranks(y))


def mean_defined(values):
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None


def profile(raw):
    parts = defaultdict(list)
    for a in ANCHORS:
        original = raw[a-192:a+48]
        x = interpolate(original)
        c, s = robust(x[:192])
        parts['context_gap_fraction'].append(st.mean(not math.isfinite(v) for v in original[:192]))
        parts['target_gap_fraction'].append(st.mean(not math.isfinite(v) for v in original[192:]))
        parts['acf_1'].append(corr(x[:191], x[1:192]))
        parts['acf_24'].append(corr(x[:168], x[24:192]))
        parts['context_spike_fraction'].append(st.mean(abs(v-c)/s > 3.5 for v in x[:192]))
        parts['target_extreme_fraction'].append(st.mean(abs(v-c)/s > 3.5 for v in x[192:]))
        parts['context_trend'].append(abs(st.mean(x[144:192])-st.mean(x[:48]))/s)
        parts['target_context_shift'].append(abs(st.mean(x[192:])-c)/s)
        parts['historical_seasonal_error'].append(mean_defined(
            abs(original[192+j]-x[168+j % 24])/s
            for j in range(48) if math.isfinite(original[192+j])))
    result = {k: mean_defined(v) for k, v in parts.items()}
    phase = []
    for j in range(24):
        values = [raw[t] for t in range(120+j, 900, 24) if math.isfinite(raw[t])]
        phase.append(st.median(values) if values else 0.0)
    center, scale = st.mean(phase), max(st.pstdev(phase), 1e-8)
    result['_phase_shape'] = [(v-center)/scale for v in phase]
    return result


def load_rows(name):
    data = json.loads((ROOT/name).read_text())
    out = {}
    for e in data['entries'].values():
        assert e['origin'] < 4056
        for i, uid in enumerate(e['eval_uids']):
            key = (uid, e['position'], e['face'], e['program'])
            out[key] = {k: e[k][i] for k in ['L_rr', 'route', 'ctx', 'total']}
    return out


def main():
    supply = json.loads((ROOT/'artifacts/main_protocol/p4s_main_experiment_supply.json').read_text())['readable_uids']
    blocks = {f'[{lo}:{lo+40}]': supply[lo:lo+40] for lo in [0,40,80,120]}
    needed = {uid for uids in blocks.values() for uid in uids}
    raw = {}
    archive = ROOT/'data/kdd2018/raw/kdd_cup_2018_dataset_with_missing_values.zip'
    with zipfile.ZipFile(archive) as z:
        with z.open(z.namelist()[0]) as handle:
            for line in io.TextIOWrapper(handle, encoding='utf8', errors='replace'):
                uid = line.split(':', 1)[0]
                if uid not in needed:
                    continue
                tokens = islice(line.rstrip().rsplit(':', 1)[-1].split(','), 900)
                raw[uid] = [math.nan if t.strip() == '?' else float(t) for t in tokens]
    assert set(raw) == needed and all(len(v) == 900 for v in raw.values())
    profiles = {uid: profile(v) for uid, v in raw.items()}
    print('Training profiles computed for 160 series; numeric frontier 899.', flush=True)
    pooled = load_rows('_scratch/r4d_a_three_cell_store.json')
    pc = load_rows('_scratch/r4d_b_perchannel_three_cell_store.json')
    assert pc.keys() == pooled.keys()
    values = defaultdict(lambda: defaultdict(list))
    by_face = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for uid, pos, face, prog in pc:
        if prog != ANC:
            continue
        pa, ca = pooled[uid,pos,face,ANC], pc[uid,pos,face,ANC]
        pw, cw = pooled[uid,pos,face,W2], pc[uid,pos,face,W2]
        outcomes = {
            'pc_static_advantage': pa['L_rr']-ca['L_rr'],
            'pc_mad_train_gain': -ca['route'],
            'pc_mad_total_gain': -ca['total'],
            'pc_w2_vs_mad_train_gain': ca['route']-cw['route'],
            'pc_w2_vs_mad_total_gain': ca['total']-cw['total'],
            'pooled_mad_train_gain': -pa['route'],
            'pooled_w2_vs_mad_train_gain': pa['route']-pw['route'],
            'pc_mad_train_effect_magnitude': abs(ca['route']),
        }
        for target, value in outcomes.items():
            values[uid][target].append(value)
            by_face[face][uid][target].append(value)
    rows = []
    for block, uids in blocks.items():
        train, served = uids[20:40], uids[:20]
        assert all(uid in values for uid in served)
        pool_phase = [st.mean(profiles[u]['_phase_shape'][j] for u in train) for j in range(24)]
        structure_keys = ('acf_1','acf_24','context_trend')
        reference = {k:(st.mean(profiles[u][k] for u in train),
                        max(st.pstdev(profiles[u][k] for u in train), 0.1)) for k in structure_keys}
        for uid in served:
            p = profiles[uid]
            features = {k:v for k,v in p.items() if not k.startswith('_')}
            features['phase_shape_distance_to_pool'] = math.sqrt(st.mean((a-b)**2 for a,b in zip(p['_phase_shape'],pool_phase)))
            features['structure_distance_to_pool'] = math.sqrt(st.mean(((p[k]-reference[k][0])/reference[k][1])**2 for k in structure_keys))
            features['acf_24_above_pool'] = p['acf_24']-reference['acf_24'][0]
            rows.append({'uid':uid,'block':block,'features':features,
                         'outcomes':{k:st.mean(v) for k,v in values[uid].items()},
                         'by_face':{f:{k:st.mean(v) for k,v in by_face[f][uid].items()} for f in by_face},
                         'repeated_windows':len(values[uid]['pc_static_advantage'])})
    assert {r['uid'] for r in rows} == set(values) and len(rows) == 80
    weighted_advantage = sum(r['outcomes']['pc_static_advantage']*r['repeated_windows'] for r in rows)/sum(r['repeated_windows'] for r in rows)
    assert abs(weighted_advantage-0.375471045) < 1e-9
    relations = []
    for feature in rows[0]['features']:
        for target in rows[0]['outcomes']:
            available = [r for r in rows if r['features'][feature] is not None]
            x = [r['features'][feature] for r in available]
            y = [r['outcomes'][target] for r in available]
            within = {}
            group_deltas = {}
            for block in blocks:
                subset = [r for r in available if r['block'] == block]
                within[block] = rho([r['features'][feature] for r in subset], [r['outcomes'][target] for r in subset])
                ordered = sorted(subset,key=lambda r:r['features'][feature])
                group_deltas[block] = st.mean(r['outcomes'][target] for r in ordered[-7:])-st.mean(r['outcomes'][target] for r in ordered[:7])
            relations.append({'feature':feature,'outcome':target,'available_series':len(available),'spearman_80':rho(x,y),
                              'within_block_spearman':within,'mean_within_block_spearman':mean_defined(within.values()),
                              'face_spearman':{f:rho(x,[r['by_face'][f][target] for r in available]) for f in by_face},
                              'within_block_top7_minus_bottom7':group_deltas,
                              'note':'Descriptive exploration; ties may split quantile groups. No held-out selection or causal claim.'})
    result = {
        'purpose':'Discover candidate relations that can inform Observation and Workflow design; not acceptance testing.',
        'boundary':{'physical_fits':0,'llm_calls':0,'new_hashes':0,'raw_numeric_max_index':899,
                    'new_raw_deployment_horizon_values':0,'training_series':160,'served_series':80,'blocks':4,
                    'cached_outcomes':'exposed R4D-A/B only; each served UID averaged over its available windows'},
        'definitions':{'training_windows':'10 fixed anchors 312..852; 192 context + 48 historical target',
                       'raw_features':'linear interpolation; robust center/scale per historical context; standard-library descriptors',
                       'historical_seasonal_error':'MAE of repeating last 24 context points over observed historical targets, divided by context robust scale',
                       'positive_outcomes':'positive means pc better, processing helps, or W2 beats MAD; magnitude is unsigned',
                       'search_size':f'{len(rows[0]["features"])} descriptors x {len(rows[0]["outcomes"])} outcomes',
                       'uncertainty':'same-data descriptive search; correlated windows; only four shared training cohorts; zero formal significance claims'},
        'verification':{'matched_served_uid_set':True,'paired_instances':sum(r['repeated_windows'] for r in rows),
                        'weighted_pc_static_advantage':weighted_advantage,'r4d_b_reference':0.375471045},
        'relations':relations,'series':rows}
    (HERE/'training_relations.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    ranked = sorted(relations,key=lambda r:abs(r['mean_within_block_spearman'] or 0),reverse=True)
    print(json.dumps({'definitions':result['definitions'],'top_descriptive_relations':ranked[:14]},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
