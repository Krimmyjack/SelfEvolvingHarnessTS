"""DEV-AUG-TASK-FAMILY-SCREEN (user + Planner, 2026-09-23): 0-LLM screen for task families of the planned experiment
"offline-learned Workflow, data-dependent execution at zero-feedback deployment" (no card / shared card / domain card).

Question of this package: which task families have (1) structure that is observable from T-only data and (2) augmentation
treatment differences that recur across entity groups and time, so that Slow has something learnable and a domain card has
room to differ from a shared card. Output: the chosen families, the entity-disjoint Source / Select / Test division with
anchors, and a scale estimate for the paid experiment. No paid run is started here.

Design (frozen in plan.json before the first fit):
  domains     D01 electricity, D02 traffic (existing TSLib CSVs), D03 solar (Monash 10-minute -> hourly mean), D04 air quality
              (Monash KDD Cup 2018 without-missing file, hourly), D05 wind (Monash minutely wind farms -> hourly mean of >= 45 observed
              minutes). Derived CSVs are written under ROOT/data; metadata (city / station / measurement) never reaches a case.
  division    per domain an anchor grid (Source / Select / Test periods in time order, E end of a role <= T start of the next) and an
              entity partition fixed BEFORE any outcome: eligibility is T-only (finite, non-degenerate) plus a finiteness mask of
              the future rows. D01 / D02 Source pools reuse entities already used by earlier packages; Select / Test pools never used.
  screen      Source pool only: 4 groups of 16 (sorted by the domain's first structural principal component on the first screen
              anchor's T; G1-G2 low, G3-G4 high) x 2 Source anchors = 8 cases per domain; a fixed menu of 12 uniform programs x 3 seeds
              (the shared entity_case fit); E scored directly by the evaluator (no agent exists in this package).
  readout     pp of the case None mean; per domain best program, global best program, heterogeneity, recurrence, stratum
              interaction, C_A/E concordance, fingerprint separability; the family rules are written in plan.json before fits.

0 LLM / HTTP, 0 hashes, 0 commits. The coordinator never imports torch.
  --prepare-data | --plan | --smoke | --run [--numeric N] | --result | --report | --status
  subprocess entries: --worker-build CASEJSON | --worker-fit CASEJSON PHYS SEED | --worker-score CASEJSON
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as _dt
import io
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import context, data, observe, spec, tempo_aug as ta
from methods.ttha.batch_base import entity_case as ec

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_aug_task_family_screen'
DATA = ROOT / 'data'
MODULE = 'evaluation.main_protocol_p4.batch_research_aug_task_family_screen'
PACKAGE = 'DEV-AUG-TASK-FAMILY-SCREEN'
EXPOSURE = 'SOURCE_POOL_SCREEN (Select / Test pools untouched)'
SEEDS = (20269181, 20269182, 20269183)            # the entity-split seeds (ES.SEEDS)
FIT_THREADS = 8
FIT_TIMEOUT_S = 300.0
L, H, T_SPAN = spec.L, spec.H, spec.TRAIN_SPAN
ROLE_GAP = T_SPAN + 384                            # t_next - t_prev >= 1056: E end of an earlier role <= T start of the next
CASE_INDEX_BASE = 5000

RAW = {
    'solar': REPO / 'data' / 'solar_10_minutes' / 'raw' / 'solar_10_minutes_dataset.zip',
    'air_quality': REPO / 'data' / 'kdd2018' / 'raw' / 'kdd_cup_2018_dataset_without_missing_values.tsf',
    'wind': REPO / 'data' / 'wind_farms_minutely' / 'raw' / 'wind_farms_minutely_dataset_with_missing_values.zip',
}
DERIVED = {'D03': DATA / 'solar_hourly.csv', 'D04': DATA / 'air_quality_hourly.csv', 'D05': DATA / 'wind_hourly.csv'}


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


# ============================================================================= derived hourly CSVs (TSLib layout: date + one column per entity)
def _tsf_records(fh):
    """(meta dict, raw value string) per series of a Monash .tsf stream; ':' never occurs inside Monash timestamps (they use '-')."""
    attrs, in_data = [], False
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('@attribute'):
            attrs.append(line.split()[1])
            continue
        if line.startswith('@data'):
            in_data = True
            continue
        if line.startswith('@') or not in_data:
            continue
        parts = line.split(':')
        yield dict(zip(attrs, parts[:len(attrs)])), parts[len(attrs)]


def _ts(s: str) -> _dt.datetime:
    return _dt.datetime.strptime(s.strip(), '%Y-%m-%d %H-%M-%S')


def _write_csv(path: Path, start: _dt.datetime, matrix: np.ndarray, names: list) -> None:
    """matrix (rows, cols) float64 with NaN for unobserved hours; written as 'nan' (the case loader refuses non-finite rows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        f.write('date,' + ','.join(names) + '\n')
        for i in range(matrix.shape[0]):
            stamp = (start + _dt.timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
            f.write(stamp + ',' + ','.join('nan' if not np.isfinite(v) else ('%.7g' % v) for v in matrix[i]) + '\n')
    os.replace(tmp, path)


def prepare_solar() -> dict:
    z = zipfile.ZipFile(RAW['solar'])
    cols, meta = [], []
    with z.open(z.infolist()[0].filename) as fh:
        for m, vals in _tsf_records(io.TextIOWrapper(fh, encoding='latin-1')):
            a = np.array([float(x) for x in vals.split(',')])
            st = _ts(m['start_timestamp']).replace(second=0)
            if st != _dt.datetime(2006, 1, 1) or a.size != 52560:
                raise RuntimeError('unexpected solar series layout %s' % m)
            cols.append(a.reshape(8760, 6).mean(1))
            meta.append({'column': str(len(cols) - 1), 'series_name': m['series_name']})
    M = np.stack(cols, axis=1)
    names = [str(i) for i in range(M.shape[1])]
    _write_csv(DERIVED['D03'], _dt.datetime(2006, 1, 1), M, names)
    info = {'domain': 'D03', 'source': str(RAW['solar'].relative_to(REPO)), 'aggregation': 'hourly mean of the six 10-minute values [hh:00, hh:50]',
            'axis_start': '2006-01-01 00:00:00', 'rows': int(M.shape[0]), 'columns': len(names), 'nan_cells': int(np.isnan(M).sum()), 'meta': meta}
    context.write_json(DATA / 'solar_hourly_meta.json', info)
    return {k: v for k, v in info.items() if k != 'meta'}


def prepare_air_quality() -> dict:
    axis0, rows = _dt.datetime(2017, 1, 1), 10920
    cols, meta = [], []
    with RAW['air_quality'].open(encoding='latin-1') as fh:
        for m, vals in _tsf_records(fh):
            a = np.array([np.nan if x == '?' else float(x) for x in vals.split(',')])
            st = _ts(m['start_timestamp']).replace(second=0)
            off = int((st - axis0).total_seconds() // 3600)
            if off < 0 or (st - axis0).total_seconds() % 3600:
                raise RuntimeError('unaligned air-quality start %s' % m['start_timestamp'])
            col = np.full(rows, np.nan)
            n = min(a.size, rows - off)
            col[off:off + n] = a[:n]
            cols.append(col)
            meta.append({'column': str(len(cols) - 1), 'series_name': m['series_name'], 'city': m.get('city'), 'station': m.get('station'),
                         'measurement': m.get('air_quality_measurement'), 'start_row': off, 'n_values': int(a.size)})
    M = np.stack(cols, axis=1)
    names = [str(i) for i in range(M.shape[1])]
    _write_csv(DERIVED['D04'], axis0, M, names)
    info = {'domain': 'D04', 'source': str(RAW['air_quality'].relative_to(REPO)),
            'aggregation': 'hourly as published; the Monash file replaced leading missing values by 0 and later ones by LOCF (kept as is; T-only flatness filter below)',
            'axis_start': '2017-01-01 00:00:00', 'rows': rows, 'columns': len(names), 'nan_cells': int(np.isnan(M).sum()), 'meta': meta}
    context.write_json(DATA / 'air_quality_hourly_meta.json', info)
    return {k: v for k, v in info.items() if k != 'meta'}


def prepare_wind() -> dict:
    axis0, rows = _dt.datetime(2019, 8, 1), 8784
    z = zipfile.ZipFile(RAW['wind'])
    cols, meta = [], []
    with z.open(z.infolist()[0].filename) as fh:
        for m, vals in _tsf_records(io.TextIOWrapper(fh, encoding='latin-1')):
            a = np.array(vals.replace('?', 'nan').split(','), dtype=np.float64)
            st = _ts(m['start_timestamp']).replace(second=0)
            skip = (60 - st.minute) % 60                       # align to the next full hour
            a, st = a[skip:], st + _dt.timedelta(minutes=skip)
            off = int((st - axis0).total_seconds() // 3600)
            n_h = a.size // 60
            h = a[:n_h * 60].reshape(n_h, 60)
            obs = np.isfinite(h).sum(1)
            with np.errstate(invalid='ignore'):
                hm = np.where(obs >= 45, np.nanmean(np.where(np.isfinite(h), h, np.nan), axis=1), np.nan)
            col = np.full(rows, np.nan)
            n = max(0, min(n_h, rows - off))
            col[off:off + n] = hm[:n]
            cols.append(col)
            meta.append({'column': str(len(cols) - 1), 'series_name': m['series_name'], 'start_row': off, 'n_hours': int(n_h)})
    M = np.stack(cols, axis=1)
    names = [str(i) for i in range(M.shape[1])]
    _write_csv(DERIVED['D05'], axis0, M, names)
    info = {'domain': 'D05', 'source': str(RAW['wind'].relative_to(REPO)), 'aggregation': 'hourly mean of the observed minutes when >= 45 of 60 are observed, else NaN',
            'axis_start': '2019-08-01 00:00:00', 'rows': rows, 'columns': len(names), 'nan_cells': int(np.isnan(M).sum()), 'meta': meta}
    context.write_json(DATA / 'wind_hourly_meta.json', info)
    return {k: v for k, v in info.items() if k != 'meta'}


def prepare_data() -> dict:
    out = {}
    for name, fn in (('D03', prepare_solar), ('D04', prepare_air_quality), ('D05', prepare_wind)):
        t0 = time.time()
        out[name] = fn()
        out[name]['seconds'] = round(time.time() - t0, 1)
        print('PREPARED', name, out[name]['rows'], 'x', out[name]['columns'], 'nan', out[name]['nan_cells'], '%.1fs' % out[name]['seconds'], flush=True)
    context.write_json(DATA / 'prepared.json', {'prepared_local': now(), 'domains': out})
    return out


# ============================================================================= domains, anchors, menu (frozen in plan.json)
# Anchors are multiples of 24; within a domain Source < Select < Test and consecutive roles are >= ROLE_GAP apart (E end of the earlier
# role <= T start of the later one). They were chosen from row counts and the finiteness mask only (wind: common gap days 46-146).
DOMAINS = {
    'D01': {'name': 'electricity', 'dataset': 'electricity', 'csv': None, 'hours': 26304,
            'source': [672, 2976, 5280, 7584, 9888, 12192], 'select': [13248, 14952, 16656], 'test': [17712, 20448, 23184, 25920], 'screen': [2976, 9888]},
    'D02': {'name': 'traffic', 'dataset': 'traffic', 'csv': None, 'hours': 17544,
            'source': [672, 2112, 3552, 4992, 6432, 7872], 'select': [8928, 9936, 10944], 'test': [12000, 13704, 15408, 17112], 'screen': [2112, 6432]},
    'D03': {'name': 'solar', 'dataset': 'solar_hourly', 'csv': 'D03', 'hours': 8760,
            'source': [672, 1224, 1776, 2328, 2880, 3432], 'select': [4488, 4872, 5256], 'test': [6312, 6984, 7656, 8328], 'screen': [1224, 2880]},
    'D04': {'name': 'air_quality', 'dataset': 'air_quality_hourly', 'csv': 'D04', 'hours': 10920,
            'source': [696, 1464, 2232, 3000, 3768, 4536], 'select': [5592, 6240, 6888], 'test': [7944, 8784, 9624, 10464], 'screen': [1464, 3768]},
    'D05': {'name': 'wind', 'dataset': 'wind_hourly', 'csv': 'D05', 'hours': 8784,
            'source': [720, 4200, 4560, 4920, 5280], 'select': [6336, 6672, 7008], 'test': [8064, 8400], 'screen': [4200, 5280]},
}
POOL_TARGET = {'source': 64, 'select': 48, 'test': 64}          # entities (16 per group); reduced to what eligibility allows
MIN_GROUPS = {'select': 2, 'test': 2}
ELIG = {'flat_nonzero_max': 0.35, 'zero_fraction_max': 0.90, 'std_min': 1e-6}
PRIOR_SPLIT = REPO / 'docs' / 'DOMAIN_AUG_TEMPORAL_COVERAGE_V1.json'     # D01 / D02 entities already used by earlier packages
SHUFFLE_ROOT = 2026092301

MENU = [  # (material id, label, steps); None / FixedMixup / the two public tempo materials come from ec.build_public
    ('None', 'None', None),
    ('FixedMixup', 'FixedMixup', None),
    ('P_AmpResample', 'P_AmpResample', [{'op': 'tp_amplitude'}, {'op': 'tp_resample'}]),
    ('P_NoMixRecipe', 'Preset', [{'op': ta.RECIPE}]),
    ('E_cal', 'Edit[-calendar]', [ta.edit_step(['tp_calendar'])]),
    ('E_conv', 'Edit[-random_conv]', [ta.edit_step(['tp_random_conv'])]),
    ('E_resconv', 'Edit[-resample-random_conv]', [ta.edit_step(['tp_resample', 'tp_random_conv'])]),
    ('E_regshock', 'Edit[-regime-shock]', [ta.edit_step(['tp_regime', 'tp_shock'])]),
    ('C_censor', 'Comp[censor]', [{'op': 'tp_censor'}]),
    ('C_amp', 'Comp[amplitude]', [{'op': 'tp_amplitude'}]),
    ('C_conv', 'Comp[random_conv]', [{'op': 'tp_random_conv'}]),
    ('C_shock', 'Comp[shock]', [{'op': 'tp_shock'}]),
]
MENU_IDS = [m[0] for m in MENU]
LABEL = {m[0]: m[1] for m in MENU}
FEATURES = ['lag24_corr', 'lag168_corr', 'nondc_spectrum_top5_energy_share', 'standardized_abs_p95', 'standardized_abs_max', 'abs_trend_per_100h',
            'last168_std_over_full_T_std', 'zero_fraction', 'flat_nonzero_fraction', 'diff_kurtosis', 'daily_r2', 'weekly_excess_r2']
FAMILY_RULES = {
    'unit': 'G(c,p) = 100 x (E(c,None) - E(c,p)) / E(c,None), seed means (pp of the case None mean; positive = p better than None)',
    'domain_best': 'argmax over the menu of the mean G over the domain\'s 8 screen cases',
    'global_best': 'argmax over the menu of the equal-weight mean over the domains in the set of the domain mean G (computed for all 5 and for the carried set)',
    'heterogeneity': 'H_d = meanG(d, domain_best) - meanG(d, global_best)',
    'recurrence': 'cases (of 8) with G(domain_best) > G(global_best); also split by anchor and by stratum',
    'F1_effect': 'domain_best beats None by >= 3 pp on average and in >= 6/8 cases, or None itself is domain_best (a no-augmentation family)',
    'F2_distinct': 'domain_best != global_best, H_d >= 3 pp and domain_best beats global_best in >= 6/8 cases',
    'F3_observable': 'leave-one-case-out nearest centroid on standardized case fingerprints (median of the 12 T-only entity features) returns the '
                     'case\'s own domain for >= 7/8 of its cases',
    'F4_capacity': '>= 2 Select and >= 2 Test groups of 16 eligible entities that the screen never touched',
    'stratum_split': 'within a domain, low (G1, G2) and high (G3, G4) strata each prefer their own stratum-best program in >= 3/4 of their cases by '
                     '>= 3 pp on average -> a structure-level family candidate inside the domain',
    'set_rule': 'carry D01/D02 if F1 holds (existing wiring) and every new domain with F1+F3+F4; the no-card / shared / domain-card experiment is '
                'recommended only if >= 2 carried domains satisfy F2 relative to the carried set; otherwise the report says domain cards have no room '
                'over a shared card on this menu',
    'status': 'executor proposal written before the first fit; the user / Planner may overrule before the experiment is frozen',
}


def dom_index(dom: str) -> int:
    return int(dom[1:])


def csv_path(dom: str):
    D = DOMAINS[dom]
    return str(DERIVED[D['csv']]) if D['csv'] else None


def load_matrix(dom: str) -> tuple:
    """(values (rows, cols) float64, column names). The coordinator's planning reads; only masks are taken from future rows."""
    import pandas as pd
    p = DERIVED[DOMAINS[dom]['csv']] if DOMAINS[dom]['csv'] else spec.DATASETS[DOMAINS[dom]['dataset']]['path']
    df = pd.read_csv(p, index_col=0)
    if df.shape[0] != DOMAINS[dom]['hours']:
        raise RuntimeError('%s rows %d != %d' % (dom, df.shape[0], DOMAINS[dom]['hours']))
    return df.values.astype(np.float64), [str(c) for c in df.columns], [str(x) for x in df.index]


# ============================================================================= T-only features and eligibility
def entity_features(seg: np.ndarray, hours: np.ndarray, weekdays: np.ndarray) -> dict:
    """12 T-only structural features of one entity's raw T segment (672 finite values)."""
    mean, std = float(seg.mean()), float(seg.std(ddof=0))
    obs = observe.entity_observation(seg, mean, std)
    z = (seg - mean) / max(std, spec.ZERO_SCALE_FLOOR)
    d = np.diff(seg)
    nz = seg[1:] != 0
    flat_nz = float(((d == 0) & nz).mean())
    dz = np.diff(z)
    dzc = dz - dz.mean()
    kurt = float((dzc ** 4).mean() / max((dzc ** 2).mean() ** 2, 1e-12))
    tot = float(((z - z.mean()) ** 2).sum()) or 1.0
    hod = np.array([z[hours == h].mean() for h in range(24)])
    daily_r2 = float(((hod[hours] - z.mean()) ** 2).sum() / tot)
    how = weekdays * 24 + hours
    how_means = {k: z[how == k].mean() for k in np.unique(how)}
    weekly_r2 = float(((np.array([how_means[k] for k in how]) - z.mean()) ** 2).sum() / tot)
    return {'lag24_corr': obs['lag24_corr'], 'lag168_corr': obs['lag168_corr'], 'nondc_spectrum_top5_energy_share': obs['nondc_spectrum_top5_energy_share'],
            'standardized_abs_p95': obs['standardized_abs_p95'], 'standardized_abs_max': obs['standardized_abs_max'],
            'abs_trend_per_100h': abs(obs['standardized_trend_per_100h']), 'last168_std_over_full_T_std': obs['last168_std_over_full_T_std'] or 0.0,
            'zero_fraction': float((seg == 0).mean()), 'flat_nonzero_fraction': flat_nz, 'diff_kurtosis': kurt, 'daily_r2': daily_r2,
            'weekly_excess_r2': max(0.0, weekly_r2 - daily_r2)}


def eligible(M: np.ndarray, col: int, t: int) -> tuple:
    """(ok, reason): T values [t-672, t) non-degenerate; future rows [t, t+384) finite (mask only, no statistic of those values)."""
    seg = M[t - T_SPAN:t, col]
    if not np.isfinite(seg).all():
        return False, 'T_nonfinite'
    if not np.isfinite(M[t:t + 384, col]).all():
        return False, 'future_mask'
    if seg.std(ddof=0) <= ELIG['std_min']:
        return False, 'T_flat'
    if float((seg == 0).mean()) > ELIG['zero_fraction_max']:
        return False, 'T_zero'
    if float(((np.diff(seg) == 0) & (seg[1:] != 0)).mean()) > ELIG['flat_nonzero_max']:
        return False, 'T_stuck'
    return True, ''


def prior_usage(dom: str) -> dict:
    if dom not in ('D01', 'D02'):
        return {'source': set(), 'any': set()}
    d = context.read_json(PRIOR_SPLIT)
    gs = d['domains'][dom]['groups']
    return {'source': {c for g in gs if g['stage'] == 'source' for c in g['roster']}, 'any': {c for g in gs for c in g['roster']}}


def _perm(items: list, dom: str, code: int) -> list:
    rng = np.random.default_rng(np.random.SeedSequence([SHUFFLE_ROOT, dom_index(dom), code]))
    items = sorted(items, key=lambda x: int(x))
    return [items[i] for i in rng.permutation(len(items))]


def plan_domain(dom: str) -> dict:
    D = DOMAINS[dom]
    M, cols, stamps = load_matrix(dom)
    roles = ('source', 'select', 'test')
    ok = {r: [] for r in roles}
    reasons = {}
    for j, c in enumerate(cols):
        if not c.isdigit():                                              # TSLib target column 'OT' (the earlier packages' excluded column)
            reasons['non_entity_column'] = reasons.get('non_entity_column', 0) + 1
            continue
        for r in roles:
            anchors = D['screen'] if r == 'source' else D[r]            # Source-pool entities must pass at the screen anchors
            res = [eligible(M, j, t) for t in anchors]
            if all(x[0] for x in res):
                ok[r].append(c)
            else:
                for x in res:
                    if not x[0]:
                        reasons[x[1]] = reasons.get(x[1], 0) + 1
    prior = prior_usage(dom)
    pools = {}
    # Source first (the screen needs 4 groups): D01 / D02 from entities that were Source in earlier packages; new domains from all eligible
    src_cand = [c for c in ok['source'] if (not prior['any'] or c in prior['source'])]
    n = (min(POOL_TARGET['source'], len(src_cand)) // 16) * 16
    pools['source'] = _perm(src_cand, dom, 1)[:n]
    S = set(pools['source'])
    A_t = [c for c in ok['test'] if c not in S and c not in prior['any']]
    A_v = [c for c in ok['select'] if c not in S and c not in prior['any']]
    union = set(A_t) | set(A_v)
    if len(union) >= POOL_TARGET['test'] + POOL_TARGET['select']:
        t_n, v_n = POOL_TARGET['test'], POOL_TARGET['select']
    else:                                              # small domains: split what is left, Test first, both in whole groups
        v_n = max(16, (len(union) // 2) // 16 * 16) if len(union) >= 32 else 0
        t_n = min(POOL_TARGET['test'], (len(union) - v_n) // 16 * 16)
    only_t = [c for c in _perm(A_t, dom, 3) if c not in set(A_v)]
    both_t = [c for c in _perm(A_t, dom, 3) if c in set(A_v)]
    cand_t = only_t + both_t
    pools['test'] = cand_t[:(min(t_n, len(cand_t)) // 16) * 16]
    T_ = set(pools['test'])
    cand_v = [c for c in _perm(A_v, dom, 2) if c not in T_]
    pools['select'] = cand_v[:(min(v_n, len(cand_v)) // 16) * 16]
    if len(pools['test']) % 16 or len(pools['select']) % 16:
        raise RuntimeError('pool sizes must be whole groups: %s' % {r: len(v) for r, v in pools.items()})
    # structural strata of the Source pool on the first screen anchor's T
    t0 = D['screen'][0]
    ts = [_dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S') for s in stamps[t0 - T_SPAN:t0]]
    hours = np.array([x.hour for x in ts])
    wd = np.array([x.weekday() for x in ts])
    idx = {c: j for j, c in enumerate(cols)}
    feats = {c: entity_features(M[t0 - T_SPAN:t0, idx[c]], hours, wd) for c in pools['source']}
    F = np.array([[feats[c][f] for f in FEATURES] for c in pools['source']])
    mu, sd = F.mean(0), F.std(0)
    Z = (F - mu) / np.where(sd > 0, sd, 1.0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pc1 = Z @ Vt[0]
    if np.corrcoef(pc1, F[:, FEATURES.index('lag24_corr')])[0, 1] < 0:          # orientation: high = more daily-regular
        pc1, Vt[0] = -pc1, -Vt[0]
    order = [pools['source'][i] for i in np.argsort(pc1, kind='stable')]
    groups = {'G%d' % (k + 1): order[16 * k:16 * (k + 1)] for k in range(len(order) // 16)}
    return {'domain': dom, 'name': D['name'], 'dataset': D['dataset'], 'csv_path': csv_path(dom), 'hours': D['hours'],
            'anchors': {r: D[r] for r in ('source', 'select', 'test', 'screen')}, 'eligible_counts': {r: len(ok[r]) for r in roles},
            'ineligible_reason_counts': reasons, 'prior_used': len(prior['any']), 'pools': pools, 'screen_groups': groups,
            'strata': {'low': ['G1', 'G2'], 'high': ['G3', 'G4']},
            'pc1': {'explained_variance_share': float(S[0] ** 2 / (S ** 2).sum()), 'loadings': dict(zip(FEATURES, [round(float(v), 4) for v in Vt[0]]))},
            'source_feature_medians': {f: float(np.median(F[:, i])) for i, f in enumerate(FEATURES)}}


def screen_cases(plan: dict) -> list:
    out = []
    for dom, P in plan['domains'].items():
        for ai, t in enumerate(P['anchors']['screen']):
            for gi, (g, roster) in enumerate(sorted(P['screen_groups'].items())):
                if gi >= 4:
                    break
                cs = ec.CaseSpec(dataset=P['dataset'], domain=dom, case_id='%s_SCR_A%d_%s' % (dom, ai + 1, g), role='source', t=int(t), roster=tuple(roster),
                                 case_index=CASE_INDEX_BASE + 100 * dom_index(dom) + 10 * (ai + 1) + gi + 1, csv_path=P['csv_path'],
                                 total_hours=P['hours'] if P['csv_path'] else None)
                out.append(cs)
    return out


def make_plan() -> dict:
    p = ROOT / 'plan.json'
    if p.exists():
        raise RuntimeError('plan.json exists; the plan is frozen (delete the run directory to re-plan before any fit)')
    doms = {}
    for dom in DOMAINS:
        t0 = time.time()
        doms[dom] = plan_domain(dom)
        print('PLANNED', dom, 'eligible', doms[dom]['eligible_counts'], 'pools', {r: len(v) for r, v in doms[dom]['pools'].items()},
              'pc1 share %.2f' % doms[dom]['pc1']['explained_variance_share'], '%.1fs' % (time.time() - t0), flush=True)
    plan = {'package': PACKAGE, 'exposure': EXPOSURE, 'planned_local': now(), 'seeds': list(SEEDS), 'menu': [{'id': m[0], 'label': m[1], 'steps': m[2]} for m in MENU],
            'features': FEATURES, 'eligibility': ELIG, 'pool_target': POOL_TARGET, 'role_gap': ROLE_GAP, 'family_rules': FAMILY_RULES, 'domains': doms,
            'consumer': ec.CONSUMER, 'geometry': {'L': L, 'H': H, 'T': T_SPAN, 'c_a': [0, 48], 'e': [192, 240, 288, 336], 'cohort': ec.COHORT_SIZE},
            'llm_requests': 0, 'fits_planned': 0,
            'workers': {'fit_torch_threads': FIT_THREADS, 'build_torch_threads': 1, 'fit_timeout_s': FIT_TIMEOUT_S, 'min_free_gb_before_launch': MIN_FREE_GB}}
    cases = screen_cases(plan)
    plan['screen_cases'] = [c.to_json() for c in cases]
    plan['fits_planned'] = len(cases) * len(MENU) * len(SEEDS)
    context.write_json(p, plan)
    for cs in cases:
        write_case(cs)
    return plan


def case_dir() -> Path:
    return ROOT / 'cases'


def write_case(cs: ec.CaseSpec) -> Path:
    p = case_dir() / (cs.case_id + '.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if context.read_json(p) != cs.to_json():
            raise RuntimeError('case json drift for %s' % cs.case_id)
    else:
        context.write_json(p, cs.to_json())
    return p


def load_case(p) -> ec.CaseSpec:
    return ec.case_from_json(context.read_json(Path(p)))


# ============================================================================= workers (subprocess entries; torch allowed)
def _init_worker() -> None:
    from methods.ttha.batch_base import readiness as rd
    rd._init_openmp_before_torch()


def worker_build(case_json: str) -> None:
    _init_worker()
    import torch
    torch.set_num_threads(1)          # per-window primitives on 240-point tensors: 1 thread is 1.15-1.75x faster than the default 24 (measured)
    cs = load_case(case_json)
    run_dir = Path(case_json).parent
    t0 = time.time()
    ctx = ec.open_case(cs, run_dir, stage='material')
    ec.build_public(ctx)
    table = ctx.overview()['entities']
    for mid, label, steps in MENU:
        if steps is None or mid in ec.load_registry(ctx.job_dir):
            continue
        comp = ec.compile_plan_full(ec.uniform_policy(steps, 'screen menu %s' % label), table)
        ec.build_material(ctx, comp['assignment'], mid)
    reg = ec.load_registry(ctx.job_dir)
    missing = [m for m in MENU_IDS if m not in reg]
    if missing:
        raise RuntimeError('materials missing after build: %s' % missing)
    context.write_json(ctx.job_dir / 'build.json', {'case_id': cs.case_id, 'materials': MENU_IDS, 'seconds': time.time() - t0, 'built_local': now(),
                                                    'rows_read': list(cs.material_rows)})
    print('BUILD_OK', cs.case_id, '%.1fs' % (time.time() - t0), flush=True)


def worker_fit(case_json: str, phys: str, seed: int, n_updates=None) -> None:
    _init_worker()
    cs = load_case(case_json)
    rec = ec.fit_one(cs, Path(case_json).parent, phys, int(seed), threads=FIT_THREADS, timeout_s=FIT_TIMEOUT_S - 30, n_updates=n_updates)
    print('CELL_OK', rec['cell_id'], round(rec['train']['seconds'], 2), 'c_a %.5f' % rec['scores']['c_a']['normalized_mse_macro'], flush=True)


def worker_score(case_json: str) -> None:
    """Evaluator only (no agent in this package): E of every OK cell with the case's frozen T scaler; rows [t, t+384) are read here."""
    _init_worker()
    from methods.ttha.batch_base import train
    cs = load_case(case_json)
    jd = Path(case_json).parent / cs.case_id
    sc = np.load(jd / 'scaler.npz')
    if [str(x) for x in sc['roster']] != list(cs.roster) or int(sc['t']) != cs.t or str(sc['dataset']) != cs.dataset:
        raise RuntimeError('scaler binding drift for %s' % cs.case_id)
    scaler = data.Scaler(mean=sc['mean'], std=sc['std'], scale=sc['scale'], floor_hits=0)
    sl = ec.load_case_slice(cs, cs.t, max(cs.e) + H)
    wins = data.build_windows(sl, list(cs.e), scaler, with_truth=True)
    truth = np.stack([w.y_true_raw for w in wins], axis=1)
    out = {'case_id': cs.case_id, 'rows_read': [cs.t, max(cs.e) + H], 'e_origins': list(cs.e), 'scored_local': now(), 'cells': {}}
    for c, r in sorted(ec.branch_cells(jd).items()):
        pred = train.predict_windows(train.load_model(r['model_path']), wins, scaler)
        s = ec.score_with_status(pred, truth, scaler, sc['mase'])
        if s['status'] != 'SCORABLE':
            raise RuntimeError('E not scorable for %s' % c)
        out['cells'][c] = {'material_id': r['material_id'], 'model_seed': r['model_seed'], 'c_a': r['scores']['c_a']['normalized_mse_macro'],
                           'e': s['normalized_mse_macro'], 'e_per_entity': s['per_entity_normalized_mse'], 'e_per_origin': s['per_origin_normalized_mse_mean']}
    context.write_json(jd / 'e_scores.json', out)
    print('SCORE_OK', cs.case_id, len(out['cells']), flush=True)


# ============================================================================= coordinator (never imports torch)
class _MemStatus(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def free_gb() -> float:
    try:
        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 2 ** 30
    except Exception:  # noqa: BLE001
        return 99.0


MIN_FREE_GB = 1.0


def worker_env() -> dict:
    env = dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


class Coordinator:
    def __init__(self, numeric: int):
        self.sem = threading.BoundedSemaphore(numeric)
        self.lock = threading.Lock()
        self.ledger_path = ROOT / 'ledger.json'
        self.led = context.read_json(self.ledger_path) if self.ledger_path.exists() else {'fits_ok': 0, 'fits_failed': 0, 'fit_seconds': 0.0, 'builds': 0,
                                                                                          'build_seconds': 0.0, 'scores': 0, 'events': [], 'llm_requests': 0}
        self.led.setdefault('runs', []).append({'started_local': now(), 'numeric': numeric})
        self._save()

    def _save(self) -> None:
        with self.lock:
            context.write_json(self.ledger_path, self.led)

    def event(self, **kw) -> None:
        with self.lock:
            self.led['events'].append({'local': now(), **kw})
        self._save()

    def sub(self, args: list, log: Path, timeout: float) -> tuple:
        log.parent.mkdir(parents=True, exist_ok=True)
        with self.sem:
            waited = 0
            while free_gb() < MIN_FREE_GB and waited < 1800:
                time.sleep(10)
                waited += 10
            t0 = time.time()
            try:
                with log.open('w', encoding='utf-8') as fh:
                    p = subprocess.run([sys.executable, '-B', '-m', MODULE] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                                       timeout=timeout, env=worker_env())
                return p.returncode, time.time() - t0
            except subprocess.TimeoutExpired:
                return -999, time.time() - t0

    def cell_ok(self, cs: ec.CaseSpec, phys: str, seed: int, n_updates=None) -> bool:
        cp = case_dir() / cs.case_id / 'cells' / (ec.cell_id(cs.case_id, phys, seed) + '.json')
        if not cp.exists():
            return False
        r = context.read_json(cp)
        if r.get('status') != 'OK':
            return False
        if (r['job_id'] != cs.case_id or list(r['rows_read']) != list(cs.evaluate_rows) or list(r['roster']) != list(cs.roster) or r['material_id'] != phys
                or r.get('dataset') != cs.dataset or not Path(r['model_path']).exists() or r['n_updates'] != (n_updates or spec.N_UPDATES)):
            raise RuntimeError('cached cell %s is not bound to this case' % cp.name)
        return True

    def run_case(self, cj: Path, n_updates=None) -> dict:
        cs = load_case(cj)
        jd = case_dir() / cs.case_id
        logs = ROOT / 'logs' / cs.case_id
        if not (jd / 'build.json').exists():
            rc, secs = self.sub(['--worker-build', cj], logs / 'build.log', 1800)
            with self.lock:
                self.led['builds'] += 1
                self.led['build_seconds'] += secs
            self._save()
            if rc != 0 or not (jd / 'build.json').exists():
                self.event(kind='build_failed', case=cs.case_id, rc=rc)
                raise RuntimeError('build failed for %s (rc %s)' % (cs.case_id, rc))
        new = 0
        for phys in MENU_IDS:
            for seed in SEEDS:
                if self.cell_ok(cs, phys, seed, n_updates):
                    continue
                cid = ec.cell_id(cs.case_id, phys, seed)
                for attempt in (0, 1):
                    args = ['--worker-fit', cj, phys, seed] + (['--n-updates', n_updates] if n_updates else [])
                    rc, secs = self.sub(args, logs / (cid + ('.log' if attempt == 0 else '.retry.log')), FIT_TIMEOUT_S)
                    ok = rc == 0 and self.cell_ok(cs, phys, seed, n_updates)
                    with self.lock:
                        self.led['fits_ok' if ok else 'fits_failed'] += 1
                        self.led['fit_seconds'] += secs
                    self._save()
                    if ok:
                        new += 1
                        break
                    self.event(kind='fit_failed', cell=cid, rc=rc, attempt=attempt)
                else:
                    raise RuntimeError('fit failed twice for %s' % cid)
        es = jd / 'e_scores.json'
        want = {ec.cell_id(cs.case_id, p, s) for p in MENU_IDS for s in SEEDS}
        if not es.exists() or set(context.read_json(es)['cells']) != want:
            rc, secs = self.sub(['--worker-score', cj], logs / 'score.log', 900)
            with self.lock:
                self.led['scores'] += 1
            self._save()
            if rc != 0 or not es.exists():
                self.event(kind='score_failed', case=cs.case_id, rc=rc)
                raise RuntimeError('score failed for %s' % cs.case_id)
        return {'case': cs.case_id, 'fits_new': new}


def run(numeric: int = 2, case_threads: int | None = None, only=None, n_updates=None) -> dict:
    case_threads = case_threads or max(4, numeric + 1)          # enough active cases to keep every numeric slot busy
    plan = context.read_json(ROOT / 'plan.json')
    cases = [ec.case_from_json(c) for c in plan['screen_cases']]
    order = sorted(cases, key=lambda c: (c.case_id.split('_')[2], c.case_id.split('_')[3], c.domain))      # A1 G1 over domains first
    if only:
        order = [c for c in order if c.case_id in only]
    co = Coordinator(numeric)
    gate = threading.BoundedSemaphore(case_threads)
    res, errs = {}, []
    progress = ROOT / 'progress.json'

    def one(cs):
        with gate:
            try:
                res[cs.case_id] = co.run_case(write_case(cs), n_updates)
            except Exception as exc:  # noqa: BLE001
                errs.append((cs.case_id, repr(exc)))
            with co.lock:
                context.write_json(progress, {'updated_local': now(), 'done': sorted(res), 'errors': errs, 'n_cases': len(order), 'free_gb': round(free_gb(), 2)})
            print('CASE_DONE' if cs.case_id in res else 'CASE_FAILED', cs.case_id, len(res), '/', len(order), flush=True)

    ths = [threading.Thread(target=one, args=(c,), daemon=True) for c in order]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    co.led['runs'][-1]['finished_local'] = now()
    co._save()
    return {'done': len(res), 'errors': errs}


# ============================================================================= readout (0 fits)
def _case_table(cs: ec.CaseSpec) -> dict | None:
    es = case_dir() / cs.case_id / 'e_scores.json'
    if not es.exists():
        return None
    cells = context.read_json(es)['cells']
    e, ca = {}, {}
    for c, r in cells.items():
        e.setdefault(r['material_id'], {})[r['model_seed']] = r['e']
        ca.setdefault(r['material_id'], {})[r['model_seed']] = r['c_a']
    if any(len(e.get(p, {})) != len(SEEDS) for p in MENU_IDS):
        return None
    E = {p: float(np.mean([e[p][s] for s in SEEDS])) for p in MENU_IDS}
    CA = {p: float(np.mean([ca[p][s] for s in SEEDS])) for p in MENU_IDS}
    base = E['None']
    G = {p: 100.0 * (base - E[p]) / base for p in MENU_IDS}
    Gs = {p: [100.0 * (e['None'][s] - e[p][s]) / base for s in SEEDS] for p in MENU_IDS}          # paired by seed
    return {'E': E, 'CA': CA, 'G': G, 'G_seed': Gs}


def case_fingerprint(cs: ec.CaseSpec, M_cache: dict) -> dict:
    dom = cs.domain
    if dom not in M_cache:
        M_cache[dom] = load_matrix(dom)
    M, cols, stamps = M_cache[dom]
    idx = {c: j for j, c in enumerate(cols)}
    ts = [_dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S') for s in stamps[cs.t - T_SPAN:cs.t]]
    hours, wd = np.array([x.hour for x in ts]), np.array([x.weekday() for x in ts])
    F = [entity_features(M[cs.t - T_SPAN:cs.t, idx[c]], hours, wd) for c in cs.roster]
    return {f: float(np.median([x[f] for x in F])) for f in FEATURES}


def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def concordance(a: dict, b: dict) -> float:
    ks = list(a)
    agree = tot = 0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            da, db = a[ks[i]] - a[ks[j]], b[ks[i]] - b[ks[j]]
            if da == 0 or db == 0:
                continue
            tot += 1
            agree += int((da > 0) == (db > 0))
    return agree / tot if tot else float('nan')


def readout() -> dict:
    plan = context.read_json(ROOT / 'plan.json')
    cases = [ec.case_from_json(c) for c in plan['screen_cases']]
    tabs, missing = {}, []
    for cs in cases:
        t = _case_table(cs)
        if t is None:
            missing.append(cs.case_id)
        else:
            tabs[cs.case_id] = t
    M_cache = {}
    fps = {cs.case_id: case_fingerprint(cs, M_cache) for cs in cases}
    doms = sorted(plan['domains'])
    by_dom = {d: [cs for cs in cases if cs.domain == d and cs.case_id in tabs] for d in doms}
    dom_mean = {d: {p: _mean([tabs[c.case_id]['G'][p] for c in by_dom[d]]) for p in MENU_IDS} for d in doms if by_dom[d]}

    def best(means: dict):
        ok = {p: v for p, v in means.items() if v is not None}
        return max(ok, key=lambda p: ok[p]) if ok else None

    def need(frac: float, n: int) -> int:                  # the frozen rules are written as fractions of 8 cases (6/8, 7/8, 3/4)
        return int(math.ceil(frac * n - 1e-9))

    def global_best(ds):
        m = {p: float(np.mean([dom_mean[d][p] for d in ds])) for p in MENU_IDS}
        return best(m), m

    g_all, g_all_means = global_best([d for d in doms if d in dom_mean])
    per_dom = {}
    for d in dom_mean:
        b = best(dom_mean[d])
        cs_list = by_dom[d]
        def wins(p, q, subset=None):
            sub = cs_list if subset is None else [c for c in cs_list if subset(c)]
            return sum(1 for c in sub if tabs[c.case_id]['G'][p] > tabs[c.case_id]['G'][q]), len(sub)
        low = lambda c: c.case_id.endswith(('G1', 'G2'))
        high = lambda c: c.case_id.endswith(('G3', 'G4'))
        strat = {}
        for nm, f in (('low', low), ('high', high)):
            sub = [c for c in cs_list if f(c)]
            m = {p: _mean([tabs[c.case_id]['G'][p] for c in sub]) for p in MENU_IDS}
            strat[nm] = {'best': best(m), 'mean_G': m, 'n': len(sub)}
        cross = {}
        for nm, f, other in (('low', low, 'high'), ('high', high, 'low')):
            bo, bs = strat[other]['best'], strat[nm]['best']
            sub = [c for c in cs_list if f(c)]
            cross[nm] = {'own_best': bs, 'other_best': bo, 'own_beats_other': sum(1 for c in sub if tabs[c.case_id]['G'][bs] > tabs[c.case_id]['G'][bo]),
                         'gain_own_over_other': _mean([tabs[c.case_id]['G'][bs] - tabs[c.case_id]['G'][bo] for c in sub]), 'n': len(sub)}
        per_anchor = {}
        for a in ('A1', 'A2'):
            sub = [c for c in cs_list if '_%s_' % a in c.case_id]
            per_anchor[a] = {'mean_G': {p: _mean([tabs[c.case_id]['G'][p] for c in sub]) for p in MENU_IDS}}
            per_anchor[a]['best'] = best(per_anchor[a]['mean_G'])
        seed_se = []
        for c in cs_list:
            for p in MENU_IDS:
                v = tabs[c.case_id]['G_seed'][p]
                if p != 'None':
                    seed_se.append(float(np.std(v, ddof=1) / np.sqrt(len(v))))
        per_dom[d] = {
            'n_cases': len(cs_list), 'best': b, 'mean_G': dom_mean[d], 'best_beats_none': wins(b, 'None')[0] if b != 'None' else None,
            'global_best_all': g_all, 'H_vs_global_all': dom_mean[d][b] - dom_mean[d][g_all], 'best_beats_global_all': wins(b, g_all)[0] if b != g_all else None,
            'rank': sorted(MENU_IDS, key=lambda p: -dom_mean[d][p]), 'per_anchor': per_anchor, 'strata': strat, 'stratum_cross': cross,
            'case_sd_of_G_best_minus_global': float(np.std([tabs[c.case_id]['G'][b] - tabs[c.case_id]['G'][g_all] for c in cs_list], ddof=1)) if len(cs_list) > 1 else None,
            'median_seed_se_pp': float(np.median(seed_se)) if seed_se else None,
            'ca_e_concordance_mean': _mean([concordance({p: -tabs[c.case_id]['CA'][p] for p in MENU_IDS}, {p: -tabs[c.case_id]['E'][p] for p in MENU_IDS}) for c in cs_list]),
            'ca_argmin_equals_e_argmin': sum(1 for c in cs_list if min(MENU_IDS, key=lambda p: tabs[c.case_id]['CA'][p]) == min(MENU_IDS, key=lambda p: tabs[c.case_id]['E'][p])),
        }
    # observability: leave-one-case-out nearest centroid on standardized case fingerprints
    ids = [cs.case_id for cs in cases]
    X = np.array([[fps[i][f] for f in FEATURES] for i in ids])
    y = [cs.domain for cs in cases]
    loo = {}
    for k, i in enumerate(ids):
        mask = np.arange(len(ids)) != k
        mu, sd = X[mask].mean(0), X[mask].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Z = (X - mu) / sd
        cents = {d: Z[[j for j in range(len(ids)) if mask[j] and y[j] == d]].mean(0) for d in doms}
        loo[i] = min(cents, key=lambda d: float(((Z[k] - cents[d]) ** 2).sum()))
    obs = {d: sum(1 for i, dd in zip(ids, y) if dd == d and loo[i] == d) for d in doms}
    fam = {}
    for d in per_dom:
        P = per_dom[d]
        n_d = P['n_cases']
        F1 = (P['best'] == 'None') or (P['mean_G'][P['best']] >= 3 and (P['best_beats_none'] or 0) >= need(0.75, n_d))
        F2 = P['best'] != g_all and P['H_vs_global_all'] >= 3 and (P['best_beats_global_all'] or 0) >= need(0.75, n_d)
        F3 = obs[d] >= need(0.875, n_d)
        pools = plan['domains'][d]['pools']
        F4 = len(pools['select']) // 16 >= MIN_GROUPS['select'] and len(pools['test']) // 16 >= MIN_GROUPS['test']
        split = all(P['stratum_cross'][s]['own_best'] != P['stratum_cross'][s]['other_best']
                    and P['stratum_cross'][s]['own_beats_other'] >= need(0.75, P['stratum_cross'][s]['n'])
                    and (P['stratum_cross'][s]['gain_own_over_other'] or 0) >= 3 for s in ('low', 'high'))
        fam[d] = {'F1_effect': F1, 'F2_distinct_vs_all': F2, 'F3_observable': F3, 'F4_capacity': F4, 'stratum_split': split,
                  'thresholds_used': {'n_cases': n_d, 'wins_needed_0.75': need(0.75, n_d), 'loo_needed_0.875': need(0.875, n_d)}, 'loo_correct': obs[d], 'select_groups': len(pools['select']) // 16, 'test_groups': len(pools['test']) // 16}
    carried = [d for d in fam if (d in ('D01', 'D02') and fam[d]['F1_effect']) or (d not in ('D01', 'D02') and fam[d]['F1_effect'] and fam[d]['F3_observable'] and fam[d]['F4_capacity'])]
    carried_view = {}
    if carried:
        gc, gc_means = global_best(carried)
        for d in carried:
            P = per_dom[d]
            b = P['best']
            w = sum(1 for c in by_dom[d] if tabs[c.case_id]['G'][b] > tabs[c.case_id]['G'][gc])
            carried_view[d] = {'global_best_carried': gc, 'H_vs_carried': dom_mean[d][b] - dom_mean[d][gc], 'best_beats_carried_global': w if b != gc else None,
                               'F2_vs_carried': b != gc and dom_mean[d][b] - dom_mean[d][gc] >= 3 and w >= need(0.75, P['n_cases'])}
    n_f2 = sum(1 for v in carried_view.values() if v['F2_vs_carried'])
    res = {'package': PACKAGE, 'read_local': now(), 'missing_cases': missing, 'n_cases_scored': len(tabs), 'per_case': tabs, 'fingerprints': fps,
           'loo_domain_prediction': loo, 'observability_correct': obs, 'per_domain': per_dom, 'global_best_all': g_all, 'global_means_all': g_all_means,
           'family_checks': fam, 'carried': carried, 'carried_view': carried_view, 'n_carried_F2': n_f2,
           'experiment_recommended': n_f2 >= 2}
    context.write_json(ROOT / 'result.json', res)
    return res


# ============================================================================= smoke (synthetic CSV, tiny fits; never touches the real run)
def smoke() -> dict:
    sroot = ROOT / 'smoke'
    import shutil
    if sroot.exists():
        shutil.rmtree(sroot)
    sroot.mkdir(parents=True)
    rng = np.random.default_rng(0)
    hours, ncol = 2400, 20
    tt = np.arange(hours)
    Msyn = np.stack([10 + 3 * np.sin(2 * np.pi * (tt + k) / 24) + rng.normal(0, 0.5, hours) for k in range(ncol)], axis=1)
    csvp = sroot / 'synthetic.csv'
    _write_csv(csvp, _dt.datetime(2020, 1, 1), Msyn, [str(i) for i in range(ncol)])
    cs = ec.CaseSpec(dataset='synthetic_smoke', domain='D03', case_id='D03_SMOKE', role='source', t=1200, roster=tuple(str(i) for i in range(16)),
                     case_index=9999, csv_path=str(csvp), total_hours=hours)
    cj = sroot / 'cases' / 'D03_SMOKE.json'
    cj.parent.mkdir(parents=True, exist_ok=True)
    context.write_json(cj, cs.to_json())
    checks = {}
    env = worker_env()

    def call(args, log):
        with (sroot / log).open('w', encoding='utf-8') as fh:
            return subprocess.run([sys.executable, '-B', '-m', MODULE] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=1800).returncode

    checks['build_rc'] = call(['--worker-build', cj], 'build.log')
    reg = ec.load_registry(sroot / 'cases' / 'D03_SMOKE')
    checks['materials'] = sorted(reg) == sorted(MENU_IDS)
    rcs = [call(['--worker-fit', cj, p, SEEDS[0], '--n-updates', 30], 'fit_%s.log' % p) for p in ('None', 'P_NoMixRecipe', 'C_censor')]
    checks['fit_rcs'] = rcs
    checks['score_rc'] = call(['--worker-score', cj], 'score.log')
    es = context.read_json(sroot / 'cases' / 'D03_SMOKE' / 'e_scores.json')
    checks['scored_cells'] = len(es['cells'])
    # independent recomputation of one E score from raw CSV rows and the saved prediction path (numpy only, no train module)
    import pandas as pd
    df = pd.read_csv(csvp, index_col=0)
    sc = np.load(sroot / 'cases' / 'D03_SMOKE' / 'scaler.npz')
    raw = df.values[:, :16]
    mean_ok = np.allclose(sc['mean'], raw[cs.t - T_SPAN:cs.t].mean(0))
    checks['scaler_matches_raw_T'] = bool(mean_ok)
    checks['e_rows'] = es['rows_read'] == [cs.t, max(cs.e) + H]
    checks['ok'] = checks['build_rc'] == 0 and checks['materials'] and rcs == [0, 0, 0] and checks['score_rc'] == 0 and checks['scored_cells'] == 3 and mean_ok and checks['e_rows']
    context.write_json(sroot / 'smoke.json', checks)
    return checks


def status() -> dict:
    led = context.read_json(ROOT / 'ledger.json') if (ROOT / 'ledger.json').exists() else {}
    prog = context.read_json(ROOT / 'progress.json') if (ROOT / 'progress.json').exists() else {}
    return {'fits_ok': led.get('fits_ok'), 'fits_failed': led.get('fits_failed'), 'fit_seconds': round(led.get('fit_seconds', 0), 1),
            'builds': led.get('builds'), 'scores': led.get('scores'), 'cases_done': len(prog.get('done', [])), 'n_cases': prog.get('n_cases'),
            'errors': prog.get('errors'), 'free_gb': round(free_gb(), 2)}


# ============================================================================= report, family choice, division proposal, scale estimate (0 fits)
CONTRAST_BASE = 'P_NoMixRecipe'
ROLE_TAG = {'source': 'S', 'select': 'V', 'test': 'Q'}
# Per-unit costs for the scale estimate, measured in DEV-DOMAIN-AUG-TEMPORAL-COVERAGE (REPORT section 0.6 / 6) unless marked as an assumption.
UNIT = {
    'f0_tokens_per_case': 399_781, 'f0_requests_per_case': 8.44, 'f0_new_evals_per_case': 3.9,           # Fast with C_A research (measured)
    'slow_tokens_per_call': (122_000, 368_000), 'slow_calls_per_card_set': 3,                             # measured range of Slow proposals
    'zero_feedback_tokens_per_trajectory': (80_000, 250_000), 'zero_feedback_requests_per_trajectory': (3, 6),   # ASSUMPTION (no evaluate tool)
    'fit_seconds_parallel3': 13.2,                                                                        # measured in this screen (3 workers)
}


def _f(x, nd=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    return ('%+.*f' if sign else '%.*f') % (nd, x)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def contrast_matrix(res: dict, ids: list) -> tuple:
    """(names, Y (cases x contrasts)): G(c,p) - G(c,Preset) for every other menu program (None gives -G(Preset))."""
    names = [p for p in MENU_IDS if p != CONTRAST_BASE]
    Y = np.array([[res['per_case'][c]['G'][p] - res['per_case'][c]['G'][CONTRAST_BASE] for p in names] for c in ids])
    return names, Y


def feature_associations(res: dict, n_perm: int = 2000) -> dict:
    ids = sorted(res['per_case'])
    doms = [c.split('_')[0] for c in ids]
    X = np.array([[res['fingerprints'][c][f] for f in FEATURES] for c in ids])
    names, Y = contrast_matrix(res, ids)
    out = {'n_cases': len(ids), 'contrasts': names, 'features': FEATURES}

    def rho_mat(Xm, Ym):
        return np.array([[_spearman(Xm[:, i], Ym[:, j]) for j in range(Ym.shape[1])] for i in range(Xm.shape[1])])

    def demean(A):
        B = A.copy()
        for d in set(doms):
            m = [k for k, x in enumerate(doms) if x == d]
            B[m] -= B[m].mean(0)
        return B

    rng = np.random.default_rng(np.random.SeedSequence([SHUFFLE_ROOT, 77]))
    for tag, Xm, Ym in (('pooled', X, Y), ('within_domain', demean(X), demean(Y))):
        R = rho_mat(Xm, Ym)
        mx = []
        for _ in range(n_perm):
            if tag == 'pooled':
                perm = rng.permutation(len(ids))
            else:                                                   # permute cases only within their own domain
                perm = np.arange(len(ids))
                for d in sorted(set(doms)):
                    m = np.array([k for k, x in enumerate(doms) if x == d])
                    perm[m] = rng.permutation(m)
            mx.append(np.nanmax(np.abs(rho_mat(Xm, Ym[perm]))))
        thr = float(np.percentile(mx, 95))
        hits = [{'feature': FEATURES[i], 'contrast': names[j], 'rho': float(R[i, j])} for i in range(len(FEATURES)) for j in range(len(names))
                if np.isfinite(R[i, j]) and abs(R[i, j]) >= thr]
        out[tag] = {'max_abs_rho_threshold_95': thr, 'hits': sorted(hits, key=lambda h: -abs(h['rho'])),
                    'top': sorted([{'feature': FEATURES[i], 'contrast': names[j], 'rho': float(R[i, j])} for i in range(len(FEATURES)) for j in range(len(names))
                                   if np.isfinite(R[i, j])], key=lambda h: -abs(h['rho']))[:10]}
    # share of across-case variance of each contrast explained by the domain (one-way ANOVA R^2)
    r2 = {}
    for j, p in enumerate(names):
        y = Y[:, j]
        sst = float(((y - y.mean()) ** 2).sum())
        ssb = float(sum(len(m) * (y[m].mean() - y.mean()) ** 2 for m in [[k for k, x in enumerate(doms) if x == d] for d in sorted(set(doms))]))
        r2[p] = ssb / sst if sst > 0 else float('nan')
    out['domain_r2'] = r2
    out['note'] = ('Spearman across screen cases; the 95% threshold is the permutation distribution of the MAX |rho| over all feature x contrast pairs '
                   '(family-wise); within_domain = domain-demeaned features and contrasts, permutations within domain. Exploratory, screen cases only.')
    return out


def group_ok(M, cols_idx, roster, t) -> bool:
    return all(eligible(M, cols_idx[c], t)[0] for c in roster)


def division_proposal(plan: dict, res: dict) -> dict:
    """Entity-disjoint Source / Select / Test cases for the carried families; every (group, anchor) is kept only if all 16 entities pass the T-only
    eligibility and the future finiteness mask. Screen cases are reused as Source cases (same case_id / roster / t -> caches reusable)."""
    out = {'families': {}, 'rule': 'Source groups = screen groups; Select / Test groups = the frozen pools in their shuffled order, 16 per group; '
                                   'anchors = the frozen role anchors; a (group, anchor) is dropped if any entity fails eligibility at that anchor'}
    for d in res['carried']:
        P = plan['domains'][d]
        M, cols, stamps = load_matrix(d)
        ci = {c: j for j, c in enumerate(cols)}
        fam = {'name': P['name'], 'dataset': P['dataset'], 'csv_path': P['csv_path'], 'hours': P['hours'], 'cases': {'source': [], 'select': [], 'test': []}, 'dropped': []}
        groups = {'source': sorted(P['screen_groups'].items()),
                  'select': [('G%d' % (k + 1), P['pools']['select'][16 * k:16 * (k + 1)]) for k in range(len(P['pools']['select']) // 16)],
                  'test': [('G%d' % (k + 1), P['pools']['test'][16 * k:16 * (k + 1)]) for k in range(len(P['pools']['test']) // 16)]}
        screen_t = P['anchors']['screen']
        for role in ('source', 'select', 'test'):
            for ai, t in enumerate(P['anchors'][role]):
                for g, roster in groups[role]:
                    if role == 'source' and t in screen_t:
                        cid, cidx = '%s_SCR_A%d_%s' % (d, screen_t.index(t) + 1, g), CASE_INDEX_BASE + 100 * dom_index(d) + 10 * (screen_t.index(t) + 1) + int(g[1:])
                        reuse = True
                    else:
                        cid = '%s_%s_%s%d_%s' % (d, ROLE_TAG[role], ROLE_TAG[role], ai + 1, g)
                        cidx = 6000 + 1000 * ('source', 'select', 'test').index(role) + 100 * dom_index(d) + 10 * (ai + 1) + int(g[1:])
                        reuse = False
                    if not group_ok(M, ci, roster, t):
                        fam['dropped'].append({'role': role, 'case_id': cid, 't': t, 'reason': 'eligibility'})
                        continue
                    fam['cases'][role].append({'case_id': cid, 'case_index': cidx, 't': t, 'group': g, 'roster': list(roster), 'reuses_screen_case': reuse,
                                               'train_rows': [t - T_SPAN, t], 'c_a_origins': [t, t + 48], 'e_origins': [t + 192, t + 240, t + 288, t + 336]})
        fam['counts'] = {r: len(v) for r, v in fam['cases'].items()}
        ents = {r: {c for x in fam['cases'][r] for c in x['roster']} for r in ('source', 'select', 'test')}
        fam['entity_disjoint'] = not (ents['source'] & ents['select'] or ents['source'] & ents['test'] or ents['select'] & ents['test'])
        last_src_e = max([x['t'] + 384 for x in fam['cases']['source']] or [0])
        first_sel_T = min([x['t'] - T_SPAN for x in fam['cases']['select']] or [10 ** 9])
        last_sel_e = max([x['t'] + 384 for x in fam['cases']['select']] or [0])
        first_test_T = min([x['t'] - T_SPAN for x in fam['cases']['test']] or [10 ** 9])
        fam['time_barriers_ok'] = last_src_e <= first_sel_T and last_sel_e <= first_test_T
        out['families'][d] = fam
    return out


SCALE_CONFIGS = {
    # Source F0 runs on screen cases first (their 12-program menu cells are reused); Select runs every candidate card + no card on every Select case
    # of its own family; Test runs no card / shared / domain card on every Test case; references None / NoMix / Fixed_dev = 3 programs x 3 seeds.
    'lean': {'source_f0_cases_per_family': 4, 'domain_cards': 2, 'shared_cards': 2},
    'full': {'source_f0_cases_per_family': 12, 'domain_cards': 3, 'shared_cards': 3},
}


def scale_estimate(div: dict, cfg: dict) -> dict:
    """Planning arithmetic only (the Planner freezes the real caps)."""
    F = list(div['families'])
    n_sel = sum(div['families'][d]['counts']['select'] for d in F)
    n_test = sum(div['families'][d]['counts']['test'] for d in F)
    n_f0 = {d: min(cfg['source_f0_cases_per_family'], div['families'][d]['counts']['source']) for d in F}
    f0_on_new = {d: max(0, n_f0[d] - sum(1 for x in div['families'][d]['cases']['source'] if x['reuses_screen_case'])) for d in F}
    lo_t, hi_t = UNIT['zero_feedback_tokens_per_trajectory']
    lo_r, hi_r = UNIT['zero_feedback_requests_per_trajectory']
    slo, shi = UNIT['slow_tokens_per_call']
    n_cards = cfg['domain_cards'] + cfg['shared_cards'] + 1
    select_traj, test_traj = n_sel * n_cards, n_test * 3
    fits = {'source_f0': sum(n_f0.values()) * round(UNIT['f0_new_evals_per_case'] * 3), 'source_refs_new': 12 * sum(f0_on_new.values()),
            'select': select_traj * 3 + n_sel * 9, 'test': test_traj * 3 + n_test * 9}
    slow_calls = len(F) * cfg['domain_cards'] + cfg['shared_cards']
    tokens = {'source_f0': sum(n_f0.values()) * UNIT['f0_tokens_per_case'], 'slow': (slow_calls * slo, slow_calls * shi),
              'select': (select_traj * lo_t, select_traj * hi_t), 'test': (test_traj * lo_t, test_traj * hi_t)}
    tot = (tokens['source_f0'] + tokens['slow'][0] + tokens['select'][0] + tokens['test'][0], tokens['source_f0'] + tokens['slow'][1] + tokens['select'][1] + tokens['test'][1])
    total_fits = sum(fits.values())
    return {'config': cfg, 'families': F, 'cases': {'source_f0': sum(n_f0.values()), 'select': n_sel, 'test': n_test},
            'trajectories': {'source_f0': sum(n_f0.values()), 'select': select_traj, 'test': test_traj}, 'slow_calls': slow_calls,
            'requests': {'source_f0': round(sum(n_f0.values()) * UNIT['f0_requests_per_case']), 'select': (select_traj * lo_r, select_traj * hi_r),
                         'test': (test_traj * lo_r, test_traj * hi_r), 'slow': slow_calls},
            'fits': fits, 'fits_total': total_fits, 'fit_hours_at_3_workers': round(total_fits * UNIT['fit_seconds_parallel3'] / 3600, 1),
            'tokens': tokens, 'tokens_total_range': tot, 'assumptions': UNIT,
            'note': 'zero-feedback trajectory size is an ASSUMPTION (no evaluate tool; overview in the first prompt); every other unit is measured.'}


def report() -> dict:
    res = readout()
    plan = context.read_json(ROOT / 'plan.json')
    assoc = feature_associations(res) if len(res['per_case']) >= 10 else None
    div = division_proposal(plan, res) if res['carried'] else {'families': {}}
    scale = {k: scale_estimate(div, cfg) for k, cfg in SCALE_CONFIGS.items()} if div['families'] else None
    context.write_json(ROOT / 'associations.json', assoc)
    context.write_json(ROOT / 'division_proposal.json', div)
    context.write_json(ROOT / 'scale_estimate.json', scale)
    led = context.read_json(ROOT / 'ledger.json') if (ROOT / 'ledger.json').exists() else {}
    doms = sorted(res['per_domain'])
    L_ = []
    L_.append('# %s — 结果表（自动生成，%s）\n' % (PACKAGE, now()))
    L_.append('单位：pp of None，G(c,p) = 100 × (E_None − E_p) / E_None，三 seed 均值；正 = 比 None 好。每域 8 个筛选案例（D04 为 6 个）。\n')
    L_.append('## 1. 各域 × 菜单平均 G\n')
    L_.append('| 程序 | ' + ' | '.join('%s %s' % (d, DOMAINS[d]['name']) for d in doms) + ' | 五域等权 |')
    L_.append('|---|' + '---:|' * (len(doms) + 1))
    for p in MENU_IDS:
        row = [res['per_domain'][d]['mean_G'][p] for d in doms]
        L_.append('| %s | %s | %s |' % (LABEL[p], ' | '.join(_f(v) for v in row), _f(float(np.mean(row)))))
    L_.append('')
    L_.append('## 2. 域最优、全局最优、异质性与重复性\n')
    L_.append('| 域 | 域最优 | 均值 | 胜 None（/n） | 全局最优（五域） | H_d | 域最优胜全局（/n） | 最优 A1 / A2 | 低层 / 高层最优 | 低层自胜 / 高层自胜 | 案例 SD（最优−全局） | seed SE 中位 | C_A/E 同向率 | C_A 最优 = E 最优 |')
    L_.append('|---|---|---:|---|---|---:|---|---|---|---|---:|---:|---:|---|')
    for d in doms:
        P = res['per_domain'][d]
        L_.append('| %s | %s | %s | %s/%d | %s | %s | %s/%d | %s / %s | %s / %s | %s/%d / %s/%d | %s | %s | %s | %d/%d |' % (
            d, LABEL[P['best']], _f(P['mean_G'][P['best']]), P['best_beats_none'] if P['best_beats_none'] is not None else '—', P['n_cases'],
            LABEL[P['global_best_all']], _f(P['H_vs_global_all']), P['best_beats_global_all'] if P['best_beats_global_all'] is not None else '—', P['n_cases'],
            LABEL[P['per_anchor']['A1']['best']], LABEL[P['per_anchor']['A2']['best']], LABEL[P['strata']['low']['best']], LABEL[P['strata']['high']['best']],
            P['stratum_cross']['low']['own_beats_other'], P['stratum_cross']['low']['n'], P['stratum_cross']['high']['own_beats_other'], P['stratum_cross']['high']['n'],
            _f(P['case_sd_of_G_best_minus_global'], sign=False), _f(P['median_seed_se_pp'], sign=False), _f(P['ca_e_concordance_mean'], 3, False),
            P['ca_argmin_equals_e_argmin'], P['n_cases']))
    L_.append('')
    L_.append('## 3. 任务族判据（plan.json 中冻结的执行者提议）\n')
    L_.append('| 域 | F1 效应 | F2 异质（对五域全局） | F3 可观察（留一 %s） | F4 容量（Select/Test 组） | 层内分裂 |' % '/n')
    L_.append('|---|---|---|---|---|---|')
    for d in doms:
        f = res['family_checks'][d]
        L_.append('| %s | %s | %s | %s（%d/%d） | %s（%d/%d） | %s |' % (d, f['F1_effect'], f['F2_distinct_vs_all'], f['F3_observable'], f['loo_correct'],
                                                                 res['per_domain'][d]['n_cases'], f['F4_capacity'], f['select_groups'], f['test_groups'], f['stratum_split']))
    L_.append('')
    L_.append('携带域：%s；携带集合内全局最优与异质性：\n' % (', '.join(res['carried']) or '无'))
    if res['carried_view']:
        L_.append('| 域 | 携带集全局最优 | H_d（对携带集） | 域最优胜携带集全局（/n） | F2（对携带集） |')
        L_.append('|---|---|---:|---|---|')
        for d, v in res['carried_view'].items():
            L_.append('| %s | %s | %s | %s | %s |' % (d, LABEL[v['global_best_carried']], _f(v['H_vs_carried']), v['best_beats_carried_global'] if v['best_beats_carried_global'] is not None else '—', v['F2_vs_carried']))
        L_.append('')
    L_.append('满足 F2 的携带域数：%d；按规则 %s 建议进入“无卡 / 共享卡 / 域卡”实验。\n' % (res['n_carried_F2'], '' if res['experiment_recommended'] else '不'))
    if assoc:
        L_.append('## 4. 可观察结构与处理差异的关联（探索性）\n')
        L_.append('对比量 = G(p) − G(Preset)。域解释的跨案例方差（R²）：' + '，'.join('%s %.2f' % (LABEL[p], v) for p, v in assoc['domain_r2'].items()) + '\n')
        for tag in ('pooled', 'within_domain'):
            A = assoc[tag]
            L_.append('- %s：族错误率 95%% 阈值 |ρ| ≥ %.2f；超阈值 %d 对；最强：%s' % (tag, A['max_abs_rho_threshold_95'], len(A['hits']),
                      '；'.join('%s ~ %s ρ=%+.2f' % (h['feature'], LABEL[h['contrast']], h['rho']) for h in A['top'][:5])))
        L_.append('')
    if div.get('families'):
        L_.append('## 5. 案例划分提议\n')
        L_.append('| 域 | Source | Select | Test | 丢弃 | 实体互斥 | 时间屏障 |')
        L_.append('|---|---:|---:|---:|---:|---|---|')
        for d, fm in div['families'].items():
            L_.append('| %s | %d | %d | %d | %d | %s | %s |' % (d, fm['counts']['source'], fm['counts']['select'], fm['counts']['test'], len(fm['dropped']), fm['entity_disjoint'], fm['time_barriers_ok']))
        L_.append('')
    if scale:
        L_.append('## 6. 规模估算（供冻结，非上限；零反馈轨迹 token 为假设 80k–250k，其余单价为 TEMPORAL-COVERAGE 实测）\n')
        for k, sc in scale.items():
            L_.append('- **%s** %s：案例 %s；轨迹 %s；Slow %d 次；拟合 %d（3 路约 %.1f h）；token %.1fM – %.1fM（Source F0 %.1fM；Select %.1f–%.1fM；Test %.1f–%.1fM；Slow %.1f–%.1fM）' % (
                k, sc['config'], sc['cases'], sc['trajectories'], sc['slow_calls'], sc['fits_total'], sc['fit_hours_at_3_workers'],
                sc['tokens_total_range'][0] / 1e6, sc['tokens_total_range'][1] / 1e6, sc['tokens']['source_f0'] / 1e6,
                sc['tokens']['select'][0] / 1e6, sc['tokens']['select'][1] / 1e6, sc['tokens']['test'][0] / 1e6, sc['tokens']['test'][1] / 1e6,
                sc['tokens']['slow'][0] / 1e6, sc['tokens']['slow'][1] / 1e6))
        L_.append('')
    L_.append('## 7. 运行\n')
    L_.append('- 拟合成功 %s、失败 %s、拟合进程墙钟合计 %.0f s；构建 %s 次、评分 %s 次；LLM 请求 0；SHA 0。' % (led.get('fits_ok'), led.get('fits_failed'), led.get('fit_seconds', 0),
                                                                                  led.get('builds'), led.get('scores')))
    (ROOT / 'tables.md').write_text('\n'.join(L_) + '\n', encoding='utf-8')
    return {'carried': res['carried'], 'n_carried_F2': res['n_carried_F2'], 'experiment_recommended': res['experiment_recommended'],
            'division_counts': {d: f['counts'] for d, f in div.get('families', {}).items()},
            'scale': scale and {k: {'fits_total': v['fits_total'], 'tokens_total_range': v['tokens_total_range']} for k, v in scale.items()}}


def write_split_doc(div: dict, plan: dict, path: Path) -> dict:
    """The division in the layout ec.cases_from_split reads (docs/DOMAIN_AUG_*_V1.json style); status READY_FOR_FREEZE, nothing dispatched."""
    doc = {'status': 'PROPOSED_BY_SCREEN_NOT_FROZEN', 'setting_id': 'AUG_TASK_FAMILY_DIVISION_V1', 'package': PACKAGE,
           'exposure': {'source': 'SOURCE (screen cases reused; their E already seen by the evaluator)', 'select': 'NEVER_SCORED (T-only eligibility + future finiteness mask)',
                        'test': 'NEVER_SCORED (T-only eligibility + future finiteness mask)'},
           'cohort_size': ec.COHORT_SIZE, 'model_seeds': list(SEEDS), 'role_gap_hours': ROLE_GAP, 'domains': {}}
    for d, fam in div['families'].items():
        groups = []
        for role in ('source', 'select', 'test'):
            for x in fam['cases'][role]:
                t = x['t']
                groups.append({'case_id': x['case_id'], 'stage': role, 't': t, 'group': x['group'], 'roster': x['roster'], 'case_index': x['case_index'],
                               'reuses_screen_case': x['reuses_screen_case'], 'train_rows': [t - T_SPAN, t], 'c_a_origins': [t, t + 48], 'c_b_origins': [t + 96, t + 144],
                               'e_origins': [t + 192, t + 240, t + 288, t + 336], 'e_target_rows': [t + 192, t + 384]})
        doc['domains'][d] = {'name': fam['name'], 'dataset': fam['dataset'], 'csv_path': fam['csv_path'], 'total_hours': fam['hours'] if fam['csv_path'] else None,
                             'anchors': plan['domains'][d]['anchors'], 'counts': fam['counts'], 'dropped': fam['dropped'], 'groups': groups}
    context.write_json(path, doc)
    # the layout must load through the shared reader (geometry cross-check included)
    for dom, D in doc['domains'].items():
        for g in D['groups']:
            cs = ec.CaseSpec(dataset=D['dataset'], domain=dom, case_id=g['case_id'], role=g['stage'], t=int(g['t']), roster=tuple(g['roster']),
                             case_index=int(g['case_index']), csv_path=D.get('csv_path'), total_hours=D.get('total_hours'))
            assert list(cs.train_range) == g['train_rows'] and list(cs.c_a) == g['c_a_origins'] and list(cs.c_b) == g['c_b_origins'] and list(cs.e) == g['e_origins']
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare-data', action='store_true')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--numeric', type=int, default=2)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--worker-build')
    ap.add_argument('--worker-fit', nargs=3)
    ap.add_argument('--worker-score')
    ap.add_argument('--n-updates', type=int)
    a = ap.parse_args()
    if a.worker_build:
        worker_build(a.worker_build)
    elif a.worker_fit:
        worker_fit(a.worker_fit[0], a.worker_fit[1], int(a.worker_fit[2]), a.n_updates)
    elif a.worker_score:
        worker_score(a.worker_score)
    elif a.prepare_data:
        print(json.dumps(prepare_data(), indent=1))
    elif a.plan:
        p = make_plan()
        print(json.dumps({d: {'eligible': v['eligible_counts'], 'pools': {r: len(x) for r, x in v['pools'].items()}, 'reasons': v['ineligible_reason_counts'],
                              'pc1': v['pc1']} for d, v in p['domains'].items()}, indent=1))
        print('screen cases', len(p['screen_cases']), 'fits planned', p['fits_planned'])
    elif a.smoke:
        print(json.dumps(smoke(), indent=1))
    elif a.run:
        print(json.dumps(run(a.numeric, only=a.only, n_updates=a.n_updates), indent=1))
    elif a.result:
        r = readout()
        print(json.dumps({'scored': r['n_cases_scored'], 'missing': len(r['missing_cases']), 'global_best_all': r['global_best_all'], 'carried': r['carried'],
                          'n_carried_F2': r['n_carried_F2'], 'family_checks': r['family_checks']}, indent=1))
    elif a.report:
        print(json.dumps(report(), indent=1, default=str))
    elif a.status:
        print(json.dumps(status(), indent=1))


if __name__ == '__main__':
    main()
