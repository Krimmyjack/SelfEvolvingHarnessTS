"""Fresh forecast origins for the frozen context cards (DEV-TSFM-CONTEXT-CARD §6): the Test entity groups at new cut points whose forecast targets
were never scored by any package. Builds <FRESH>/split_copy.json, <FRESH>/plan.json, <FRESH>/common/<case>/{scaler.npz, case_spec.json,
prepared.json} (T-only) and copies the Source-frozen context choice, so the Time-MoE module can run --context-screen with SEH_CTX_STAGE=fresh
and SEH_TSFM_ROOT=<FRESH>. No target row is read here.

Rule (fixed before any fresh score): periods Q1-Q3 -> t' = t + 240 (the new targets [t'+192, t'+384) fall in the gap after the case's own E and
before the next period); the last period -> t' = t(Q1) - 1056 (targets end where the Q1 case's training segment starts). Every new target row
range must be disjoint from every scored window (C_A, C_B, E) of any case holding one of its entities.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from methods.ttha.batch_base import context
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_aug_offline_skill as OS

REPO = Path(__file__).resolve().parents[2]
FRESH = REPO / '_scratch' / 'dev_tsfm_context_fresh'
PULL = REPO / '_scratch' / 'dev_aug_tsfm_native_server_pull'
DOMAINS = OS.DOMAINS


def read(p):
    return context.read_json(Path(p))


def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    context.write_json(Path(p), obj)


def scored_windows(cs: ec.CaseSpec) -> list:
    return [(cs.c_a[0], cs.c_a[1] + ec.H), (cs.c_b[0], cs.c_b[1] + ec.H), (min(cs.e), max(cs.e) + ec.H)]


def build() -> dict:
    split = read(OS.ROOT / 'split_copy.json')
    old = ec.cases_from_split(split)
    new_split = {k: v for k, v in split.items() if k != 'domains'}
    new_split.update({'setting_id': 'TSFM_CONTEXT_FRESH_V1', 'package': 'DEV-TSFM-CONTEXT-CARD fresh origins', 'status': 'BUILT_BEFORE_ANY_FRESH_SCORE',
                      'rule': __doc__.split('Rule (fixed before any fresh score): ')[1].strip(), 'domains': {}})
    report = {}
    for d, D in split['domains'].items():
        tests = [g for g in D['groups'] if g['stage'] == 'test']
        anchors = sorted({g['t'] for g in tests})
        others = [g for g in D['groups'] if g['stage'] != 'test']
        test_ents = {c for g in tests for c in g['roster']}
        if any(set(g['roster']) & test_ents for g in others):
            raise RuntimeError('%s: a Source / Select roster shares an entity with a Test roster' % d)
        groups = []
        for g in tests:
            t = g['t']
            t_new = t + 240 if t != anchors[-1] else anchors[0] - 1056
            cid = g['case_id'].replace('_Q_', '_N_', 1)
            cs = ec.CaseSpec(dataset=D['dataset'], domain=d, case_id=cid, role='test', t=t_new, roster=tuple(g['roster']), case_index=int(g['case_index']),
                             csv_path=D.get('csv_path'), total_hours=D.get('total_hours'))
            lo, hi = min(cs.e), max(cs.e) + ec.H
            if cs.train_range[0] < 0 or hi > cs.hours():
                raise RuntimeError('%s outside the file' % cid)
            for oc in old.values():
                if oc.domain == cs.domain and set(oc.roster) & set(cs.roster):
                    for a, b in scored_windows(oc):
                        if a < hi and lo < b:
                            raise RuntimeError('%s targets [%d,%d) overlap a scored window [%d,%d) of %s' % (cid, lo, hi, a, b, oc.case_id))
            groups.append({'case_id': cid, 'stage': 'test', 'fresh': True, 't': t_new, 'group': g['group'], 'roster': list(cs.roster), 'case_index': g['case_index'],
                           'from_test_case': g['case_id'], 'train_rows': list(cs.train_range), 'c_a_origins': list(cs.c_a), 'c_b_origins': list(cs.c_b),
                           'e_origins': list(cs.e)})
            report[cid] = {'from': g['case_id'], 't_old': t, 't_new': t_new, 'targets': [lo, hi], 'history_before_t0': min(cs.e)}
        new_split['domains'][d] = {**{k: v for k, v in D.items() if k not in ('groups', 'anchors', 'counts', 'dropped')}, 'groups': groups}
    write(FRESH / 'split_copy.json', new_split)
    cases = ec.cases_from_split(new_split)
    for cid, cs in cases.items():
        ec.open_case(cs, FRESH / 'common', 'material')                  # T rows only: scaler, case_spec
        write(FRESH / 'common' / cid / 'prepared.json', {'case': cid, 'fresh': True, 'note': 'T-only scaler; E inputs are built by the context screen'})
    plan = {'package': 'DEV-TSFM-CONTEXT-CARD fresh origins', 'cases': {'fresh': {d: sorted(c for c in cases if c[:3] == d) for d in DOMAINS}},
            'rule': new_split['rule'], 'built_local': context.now() if hasattr(context, 'now') else None}
    write(FRESH / 'plan.json', plan)
    shutil.copy2(PULL / 'context_choice_source.json', FRESH / 'context_choice_source.json')
    write(FRESH / 'fresh_cases.json', report)
    print('FRESH', len(cases), 'cases', json.dumps({d: len(plan['cases']['fresh'][d]) for d in DOMAINS}))
    return report


if __name__ == '__main__':
    build()
