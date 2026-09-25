"""Readiness data availability after DEV-TEMPO-AUG-SOURCE-ALIGNMENT: exposure + eligibility only (0 fits, 0 LLM).

Question (Planner, 2026-09-18): which later batches of RD01B / RD01 / RD02 exist in the unsealed range, which of their rows were ever
scored or read by earlier lines, and are they eligible / scorable under the frozen readiness rules?

Reads:
  * JSON records of this project (rows_read / origin fields) -- which cells were evaluated, never an error or gain;
  * T rows of candidate batches (eligibility: >=336/672 finite, finite std > 1e-6, >=32 legal parents);
  * the MISSING MASK ONLY of candidate C_A / C_B / E rows (coverage rule >= 25% observed cells per entity and block), reported as counts.
    No value of an unexposed row is printed, stored, summarized or used (same line as audit_exposure_ledger: definedness is a mask
    question). PRSA rows >= 24864 (2016+) are never requested.
Writes _scratch/dev_tempo_aug_source_alignment/data_availability/{availability.json, availability.md}.
  python -B -m evaluation.main_protocol_p4.audit_readiness_data_availability
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import readiness as rd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / '_scratch' / 'dev_tempo_aug_source_alignment' / 'data_availability'
PRSA_SEALED_FROM = 24864
SPAN = rd.TRAIN_SPAN
# Non-readiness lines that scored KDD / PRSA cells (from their code; maximum absolute row of any scored truth window).
OTHER_LINES = {
    'KDD': [
        ('evaluation/functional/run_v1_kdd2018_natural_slow_update.py:75-83', 'anchors 312..852 step 60, support/selection origin 600, horizon 48', 900),
        ('evaluation/functional/run_e2_cross_series_curation.py:4657-4662', 'KDD historical origin 880, safe anchors ANCHORS (small)', 928),
        ('evaluation/main_protocol_p4/audit_exposure_ledger.py:33-34', 'P4 main protocol, sorted-UID slice 80..120, origins 4056..5016', 5064),
        ('evaluation/main_protocol_p4/dev_train1_runner.py:2250', 'DEV-TRAIN-1/2 KDD course, metadata origins 1656/2136/2376 (job-local blocks)', 2424),
        ('_scratch/flash_cross_domain_dev/run_cross_domain_dev.py:9-15', 'Flash two-source dev, block b in {0, 3600}, max raw index 7128', 7128),
    ],
    'PRSA_first_six': [
        ('_scratch/train3_preflight/preflight_landscape.py:47,58', 'Aotizhongxin..Gucheng; truth windows [o, o+48) for o in 3000..24000 step 1000 minus 16000 (21 windows)', 24048),
        ('evaluation/main_protocol_p4/dev_train3_cross_job.py:84', 'DEV-TRAIN-3 Beijing J1 b=3600, J2 b=10800 (local indices <= 3528)', 14328),
        ('_scratch/flash_cross_domain_dev/run_cross_domain_dev.py:9-15', 'Flash two-source dev, b in {0, 3600}', 7128),
    ],
}
PREFLIGHT_WINDOWS = [(o, o + 48) for o in range(3000, 24001, 1000) if o != 16000]
PREFLIGHT_STATIONS = ('Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng')


def geometry(t: int) -> dict:
    js = rd.Job('RD02', 'probe', t)                         # geometry only (dataset irrelevant)
    return {'t': t, 'T': list(js.train_range), 'C_A_target': [js.c_a[0], js.c_a[1] + rd.H], 'C_B_rows': list(js.c_b_rows),
            'C_B_target': [js.c_b[0], js.c_b[1] + rd.H], 'E_input': list(js.e_input_rows), 'E_target': list(js.e_target_rows)}


def scan_records() -> dict:
    """Every JSON record under _scratch/ that names a readiness job id and rows_read (cells, label files, stage records)."""
    job_re = re.compile(r'^(RD0[12]B?|RD01B)_(S1|S2|V1|T1|T2|T3|T4|Q1|Q2|STP1)$')
    found = {}

    def walk(v, path):
        if isinstance(v, dict):
            jid = v.get('job_id')
            rr = v.get('rows_read')
            if isinstance(jid, str) and job_re.match(jid) and rr is not None:
                pairs = []

                def pr(x):
                    if isinstance(x, list) and len(x) == 2 and all(isinstance(i, int) for i in x):
                        pairs.append(x)
                    elif isinstance(x, (list, tuple)):
                        for y in x:
                            pr(y)
                    elif isinstance(x, dict):
                        for y in x.values():
                            pr(y)
                pr(rr)
                if pairs:
                    rec = found.setdefault(jid, {'max_row': 0, 'files': set()})
                    rec['max_row'] = max(rec['max_row'], max(b for _, b in pairs))
                    rec['files'].add(str(path.relative_to(REPO)).split(os.sep)[1])
            for c in v.values():
                walk(c, path)
        elif isinstance(v, list):
            for c in v:
                walk(c, path)

    n = 0
    for root, dirs, files in os.walk(REPO / '_scratch'):
        dirs[:] = [d for d in dirs if d not in ('tsfm_env', '__pycache__', 'runs', 'pred_e', 'pred_c_a', 'materials', 'aug_materials', 'children')]
        for f in files:
            if not f.endswith('.json'):
                continue
            p = Path(root) / f
            try:
                if p.stat().st_size > 20_000_000:
                    continue
                walk(json.loads(p.read_text(encoding='utf-8')), p)
                n += 1
            except Exception:
                continue
    return {'files_scanned': n, 'jobs': {j: {'max_row_read': v['max_row'], 'packages': sorted(v['files'])} for j, v in sorted(found.items())}}


def origin_scan(threshold: int) -> list:
    """Any JSON value under a key naming an origin / anchor / block start that is >= threshold (outside the readiness records)."""
    hits = []
    key_re = re.compile(r'origin|anchor|block_start|stop', re.I)

    def walk(v, path, key=''):
        if isinstance(v, dict):
            for k, c in v.items():
                walk(c, path, str(k))
        elif isinstance(v, list):
            for c in v:
                walk(c, path, key)
        elif isinstance(v, int) and not isinstance(v, bool) and key_re.search(key) and threshold <= v < 30000:
            hits.append((str(path.relative_to(REPO)), key, v))

    for base in ('artifacts', '_scratch', 'results', 'runs'):
        for root, dirs, files in os.walk(REPO / base):
            dirs[:] = [d for d in dirs if d not in ('tsfm_env', '__pycache__', 'pred_e', 'pred_c_a', 'aug_materials', 'children')]
            for f in files:
                if not f.endswith(('.json', '.jsonl')):
                    continue
                p = Path(root) / f
                try:
                    if p.stat().st_size > 20_000_000:
                        continue
                    txt = p.read_text(encoding='utf-8')
                    docs = [json.loads(line) for line in txt.splitlines() if line.strip()] if f.endswith('.jsonl') else [json.loads(txt)]
                    for d in docs:
                        walk(d, p)
                except Exception:
                    continue
    return hits


def eligibility(dataset: str, t: int) -> dict:
    """T-only eligibility (frozen readiness rules) + mask-only coverage counts of the C_A / C_B / E target cells."""
    g = geometry(t)
    js = rd.Job(dataset, 'candidate', t)
    top = g['E_target'][1]
    if dataset == 'RD02' and top > PRSA_SEALED_FROM:
        return {'status': 'REFUSED_SEALED_REGION', 'E_target_end': top}
    sl = rd.load_slice(dataset, js.train_range[0], top)
    seg = sl.rows(*js.train_range)
    fin = np.isfinite(seg)
    scaler = rd.compute_scaler(seg)
    legal = rd.legal_parents(seg)
    N = seg.shape[1]
    t_ok = [bool(fin[:, e].sum() >= rd.MIN_T_FINITE and scaler.std[e] > rd.spec.ZERO_SCALE_FLOOR and legal.counts[e] >= rd.MIN_LEGAL_PARENTS)
            for e in range(N)]
    cov = {}
    for blk, origins in (('C_A', js.c_a), ('C_B', js.c_b), ('E', js.e)):
        mask = np.stack([np.isfinite(sl.rows(o, o + rd.H)).T for o in origins], axis=1)    # (N, O, H) booleans only
        frac = mask.sum(axis=(1, 2)) / (len(origins) * rd.H)
        cov[blk] = {'min_entity_coverage': round(float(frac.min()), 3), 'entities_below_25pct': int((frac < rd.COVERAGE_MIN).sum()),
                    'scorable': bool((frac >= rd.COVERAGE_MIN).all())}
    del sl
    return {'geometry': g, 'T_eligible_entities': int(sum(t_ok)), 'of': N, 'T_eligible': all(t_ok),
            'legal_parents_min': int(legal.counts.min()), 'coverage_mask_only': cov,
            'eligible_and_scorable': all(t_ok) and all(c['scorable'] for c in cov.values())}


def kdd_meta() -> dict:
    out = {}
    import io
    import zipfile
    p = rd.DATASETS['RD01B']['path']
    want = set(rd.RD01_ROSTER) | set(rd.RD01B_ROSTER)
    with zipfile.ZipFile(p) as z, z.open(z.namelist()[0]) as h:
        indata = False
        for line in io.TextIOWrapper(h, encoding='utf-8'):
            if not indata:
                indata = line.strip().lower() == '@data'
                continue
            f = line.split(':')
            if f[0] in want:
                out[f[0]] = {'city': f[1], 'station': f[2], 'measurement': f[3], 'start': f[4], 'length': len(f[-1].split(','))}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {'rules': {'eligibility': 'T: >=336/672 finite, finite std > 1e-6, >=32 legal parents per entity (frozen readiness rule)',
                     'scorable': 'every entity >= 25% observed cells in each block (frozen readiness rule; mask only)',
                     'exposure': 'a row range counts as exposed for a (series, rows) pair if any earlier line scored a truth window there'}}
    res['readiness_records'] = scan_records()
    res['other_lines'] = OTHER_LINES
    meta = kdd_meta()
    res['kdd_cohorts'] = {c: {'min_length': min(meta[u]['length'] for u in r), 'cities': {k: sum(meta[u]['city'] == k for u in r) for k in ('Beijing', 'London')},
                              'measurements': sorted({meta[u]['measurement'] for u in r}),
                              'beijing_stations': sorted({meta[u]['station'] for u in r if meta[u]['city'] == 'Beijing'})}
                          for c, r in (('RD01', rd.RD01_ROSTER), ('RD01B', rd.RD01B_ROSTER))}
    res['kdd_prsa_overlap'] = {'kdd_start': '2017-01-01 14:00 (Beijing series)', 'prsa_end': '2017-02-28 23:00',
                               'overlap_kdd_rows': [0, 1402], 'rd01_rd01b_beijing_pm25_series': sorted(
                                   u for u in set(rd.RD01_ROSTER) | set(rd.RD01B_ROSTER) if meta[u]['city'] == 'Beijing' and meta[u]['measurement'] == 'PM2.5')}
    # candidates
    cands = {'RD01B': [10512], 'RD01': [9360, 10512], 'RD02': [15360, 16560, 17760, 18960, 20160, 21360, 22560, 23760, 24480]}
    res['candidates'] = {}
    for ds, ts in cands.items():
        for t in ts:
            key = '%s@%d' % (ds, t)
            r = eligibility(ds, t)
            g = geometry(t)
            e0, e1 = g['E_target']
            if ds == 'RD02':
                hit = [w for w in PREFLIGHT_WINDOWS if w[0] < e1 and w[1] > e0]
                r['E_overlaps_preflight_scored_windows_first_six'] = hit
            r['E_rows_scored_by_any_known_line'] = (ds == 'RD02' and bool(r.get('E_overlaps_preflight_scored_windows_first_six')))
            res['candidates'][key] = r
            print(key, json.dumps({k: v for k, v in r.items() if k != 'geometry'}), flush=True)
    res['late_origin_mentions_ge_9000'] = origin_scan(9000)
    (OUT / 'availability.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=list), encoding='utf-8')
    print('written', OUT / 'availability.json')


if __name__ == '__main__':
    main()
