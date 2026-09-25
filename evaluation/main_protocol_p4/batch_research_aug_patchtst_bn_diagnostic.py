"""DEV-AUG-PATCHTST-BN-DIAGNOSTIC (docs/DEV_AUG_PATCHTST_BN_DIAGNOSTIC_TASK_2026-09-24.md): does part of the augmented PatchTST loss come from the
BatchNorm running statistics? Existing checkpoints only - no LLM, no optimizer update.

Electricity, the 16 exposed Test cases x 3 seeds x {None, NoMix (P_NoMixRecipe), Edit[-resample-random_conv]} checkpoints of
DEV-AUG-PATCHTST-OFFLINE-SKILL (read-only). For each checkpoint: (1) score as it is (the original scorer); (2) a copy whose learned parameters are
frozen and whose BatchNorm running statistics are re-estimated from the case's T parent input windows only (no C_A / C_B / E rows), then scored
the same way. The original package and its models are never written; every output goes to _scratch/dev_aug_patchtst_bn_diagnostic/.

  --run      freeze the plan, run the workers (GPU, parallel by memory), write result.json / REPORT.md
  --report   aggregate only
  worker:    --worker K N
"""
from __future__ import annotations

import os

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import context
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_aug_patchtst_transfer as PT
from evaluation.main_protocol_p4 import batch_research_aug_patchtst_offline_skill as M
from evaluation.main_protocol_p4 import batch_research_aug_offline_skill as OS

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_aug_patchtst_bn_diagnostic'
SRC = M.ROOT                                               # _scratch/dev_aug_patchtst_offline_skill (read-only)
MODULE = 'evaluation.main_protocol_p4.batch_research_aug_patchtst_bn_diagnostic'
TASK = 'docs/DEV_AUG_PATCHTST_BN_DIAGNOSTIC_TASK_2026-09-24.md'
SEEDS = list(M.SEEDS)
CASES = ['D01_Q_Q%d_G%d' % (q, g) for q in (1, 2, 3, 4) for g in (1, 2, 3, 4)]
REEST = {'data': 'the case T parent input windows X only (common/<case>/parents.npz Xp of the PatchTST package; 16 entities x 433 = 6928 windows); '
                 'never C_A / C_B / E inputs or any label',
         'order': 'row order of parents.npz (entity-major, window order within entity)', 'batch': 64, 'passes': 1,
         'statistic': 'BatchNorm reset_running_stats(), momentum=None (cumulative average of the per-batch statistics over all batches of the pass)',
         'mode': 'model.eval() (dropout off); only the BatchNorm modules in train(); torch.no_grad(); RevIN unchanged (instance statistics, no buffers)',
         'check': 'every learned parameter (named_parameters) bitwise unchanged after re-estimation'}
TRIM = {'fraction_per_side': 0.10, 'rule': 'within each domain drop round(0.10 x n) cases (at least 1) from each end of the sorted per-case values, mean '
                                          'the rest; three-domain = equal-weight mean of the domain trimmed means (electricity 16 -> 2+2, traffic 16 -> 2+2, solar 8 -> 1+1)'}
WORKERS = int(os.environ.get('SEH_BN_WORKERS', '2'))


def read(p):
    return context.read_json(Path(p))


def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    context.write_json(Path(p), obj)


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def materials(case: str) -> dict:
    return {'None': 'None', 'NoMix': 'P_NoMixRecipe', 'Edit': read(SRC / 'test' / 'fixed_ref_phys.json')['phys'][case]}


def freeze_plan() -> dict:
    p = ROOT / 'plan.json'
    if p.exists():
        return read(p)
    plan = {'task': TASK, 'frozen_local': now(), 'identity': 'MECHANISM diagnostic on exposed development Test checkpoints (no LLM, no optimizer update)',
            'source_package': str(SRC.relative_to(REPO)), 'cases': CASES, 'seeds': SEEDS, 'materials': {c: materials(c) for c in CASES},
            'bn_reestimation': REEST, 'scoring': 'the package scorer: E origins t+192..t+336, raw-scale predictions, ec.score_with_status normalized MSE',
            'readouts': ['NoMix / Edit gain vs the None of the same variant (orig vs orig, re vs re), 16-case mean, by period, by case, by origin',
                         'Q4_G1 origin 2: error and prediction bias (mean normalized pred - truth)',
                         'system f_patch electricity under the frozen card delivery (None stays None)',
                         'original full-sample results kept; median, trimmed mean (TRIM) and leave-out-Q4_G1 sensitivity added'],
            'trimmed_mean': TRIM, 'workers': WORKERS}
    write(p, plan)
    return plan


# ============================================================================= worker (GPU)
def bn_layers(torch, model):
    return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def reestimate(torch, model, xp):
    params0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    buf0 = {n: b.detach().clone() for n, b in model.named_buffers()}
    model.eval()
    bns = bn_layers(torch, model)
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    with torch.no_grad():
        for i in range(0, xp.shape[0], REEST['batch']):
            model(xp[i:i + REEST['batch'], :, None])
    model.eval()
    same = all(torch.equal(params0[n], p) for n, p in model.named_parameters())
    shift = {}
    for n, b in model.named_buffers():
        if n.endswith('running_mean') or n.endswith('running_var'):
            a, c = buf0[n].float(), b.float()
            shift[n] = float((c - a).norm() / max(float(a.norm()), 1e-12))
    return same, len(bns), shift


def score(torch, model, x, y, sc, mase):
    pr = PT.predict(torch, model, x) * sc.scale[:, None, None] + sc.mean[:, None, None]
    s = ec.score_with_status(pr, y, sc, mase)
    if s['status'] != 'SCORABLE':
        raise RuntimeError('non-scorable')
    dn = (pr - y) / sc.scale[:, None, None]
    return {'nmse': s['normalized_mse_macro'], 'per_origin': (dn ** 2).mean(axis=(0, 2)).tolist(), 'bias_per_origin': dn.mean(axis=(0, 2)).tolist()}


def worker(k: int, n: int) -> None:
    torch = PT.setup()
    plan = read(ROOT / 'plan.json')
    for case in CASES[k::n]:
        out_p = ROOT / 'cells' / ('%s.json' % case)
        if out_p.exists():
            continue
        stored = read(SRC / 'test' / 'scores' / ('%s.json' % case))['cells']
        with np.load(SRC / 'common' / case / 'parents.npz') as z:
            xp = torch.as_tensor(z['Xp'].copy(), device='cuda')
        x, y, sc, mase, origins = M.e_inputs(case)          # the package scorer's E inputs (SRC common scaler)
        res = {'case': case, 'origins': origins, 'n_parent_windows': int(xp.shape[0]), 'cells': {}}
        for arm, phys in plan['materials'][case].items():
            for seed in SEEDS:
                cell = next(v for v in stored.values() if v['phys'] == phys and int(v['seed']) == seed)
                model = PT.model_new(torch, seed)
                model.load_state_dict(torch.load(Path(cell['model_dir']) / 'model.pt', map_location='cuda', weights_only=True))
                model.eval()
                orig = score(torch, model, x, y, sc, mase)
                m2 = copy.deepcopy(model)
                same, nbn, shift = reestimate(torch, m2, xp)
                if not same:
                    raise RuntimeError('learned parameters changed during BN re-estimation: %s %s %d' % (case, phys, seed))
                re_ = score(torch, m2, x, y, sc, mase)
                od = ROOT / 'models' / case / phys / str(seed)
                od.mkdir(parents=True, exist_ok=True)
                torch.save({kk: v.detach().cpu() for kk, v in m2.state_dict().items()}, od / 'model_bn_reestimated.pt')
                res['cells']['%s__s%d' % (arm, seed)] = {'arm': arm, 'phys': phys, 'seed': seed, 'model_dir': cell['model_dir'], 'orig': orig, 'bn_re': re_,
                                                         'orig_equals_package_score': orig['nmse'] == cell['nmse'], 'params_bitwise_unchanged': same,
                                                         'bn_layers': nbn, 'bn_relative_shift_mean': float(np.mean(list(shift.values())))}
                del model, m2
        write(out_p, res)
        print('CASE_OK', case, flush=True)


# ============================================================================= aggregate
def G(e_mat, e_none):
    return 100.0 * (e_none - e_mat) / e_none


def trimmed(vals: list) -> float:
    v = sorted(vals)
    k = max(1, int(round(TRIM['fraction_per_side'] * len(v))))
    return statistics.fmean(v[k:len(v) - k])


def report() -> dict:
    plan = read(ROOT / 'plan.json')
    per = {}
    for case in CASES:
        r = read(ROOT / 'cells' / ('%s.json' % case))
        cells = r['cells']
        rec = {'period': case.split('_')[2], 'orig_equals_package': all(c['orig_equals_package_score'] for c in cells.values()),
               'params_unchanged': all(c['params_bitwise_unchanged'] for c in cells.values())}
        for v in ('orig', 'bn_re'):
            E = {arm: statistics.fmean(cells['%s__s%d' % (arm, s)][v]['nmse'] for s in SEEDS) for arm in ('None', 'NoMix', 'Edit')}
            PO = {arm: [statistics.fmean(cells['%s__s%d' % (arm, s)][v]['per_origin'][i] for s in SEEDS) for i in range(4)] for arm in ('None', 'NoMix', 'Edit')}
            BI = {arm: [statistics.fmean(cells['%s__s%d' % (arm, s)][v]['bias_per_origin'][i] for s in SEEDS) for i in range(4)] for arm in ('None', 'NoMix', 'Edit')}
            rec[v] = {'E': E, 'G_NoMix': G(E['NoMix'], E['None']), 'G_Edit': G(E['Edit'], E['None']),
                      'G_by_origin': {arm: [G(PO[arm][i], PO['None'][i]) for i in range(4)] for arm in ('NoMix', 'Edit')}, 'per_origin': PO, 'bias': BI}
        rec['None_bn_re_vs_orig'] = G(rec['bn_re']['E']['None'], rec['orig']['E']['None'])
        rec['NoMix_bn_re_vs_orig_None'] = G(rec['bn_re']['E']['NoMix'], rec['orig']['E']['None'])
        rec['Edit_bn_re_vs_orig_None'] = G(rec['bn_re']['E']['Edit'], rec['orig']['E']['None'])
        rec['bn_shift_mean'] = {arm: statistics.fmean(cells['%s__s%d' % (arm, s)]['bn_relative_shift_mean'] for s in SEEDS) for arm in ('None', 'NoMix', 'Edit')}
        per[case] = rec
    mean = lambda f, cs=None: statistics.fmean(f(per[c]) for c in (cs or CASES))
    summ = {}
    for v in ('orig', 'bn_re'):
        summ[v] = {'G_NoMix': mean(lambda r: r[v]['G_NoMix']), 'G_Edit': mean(lambda r: r[v]['G_Edit']),
                   'G_NoMix_by_period': {p: mean(lambda r: r[v]['G_NoMix'], [c for c in CASES if per[c]['period'] == p]) for p in ('Q1', 'Q2', 'Q3', 'Q4')},
                   'G_Edit_by_period': {p: mean(lambda r: r[v]['G_Edit'], [c for c in CASES if per[c]['period'] == p]) for p in ('Q1', 'Q2', 'Q3', 'Q4')},
                   'G_by_origin': {arm: [mean(lambda r, i=i: r[v]['G_by_origin'][arm][i]) for i in range(4)] for arm in ('NoMix', 'Edit')},
                   'G_NoMix_without_Q4_G1': mean(lambda r: r[v]['G_NoMix'], [c for c in CASES if c != 'D01_Q_Q4_G1']),
                   'G_Edit_without_Q4_G1': mean(lambda r: r[v]['G_Edit'], [c for c in CASES if c != 'D01_Q_Q4_G1']),
                   'G_NoMix_median': statistics.median(per[c][v]['G_NoMix'] for c in CASES), 'G_Edit_median': statistics.median(per[c][v]['G_Edit'] for c in CASES),
                   'wins_NoMix': sum(per[c][v]['G_NoMix'] > 0 for c in CASES), 'wins_Edit': sum(per[c][v]['G_Edit'] > 0 for c in CASES)}
    summ['None_bn_re_vs_orig'] = mean(lambda r: r['None_bn_re_vs_orig'])
    # system f_patch (electricity) under the frozen card delivery; None deliveries stay None (G 0 against the same variant's None)
    src = read(SRC / 'result.json')
    deliv = {c: src['per_case'][c]['delivered_phys']['f_patch'] for c in CASES}
    if any(deliv[c] not in ('None', plan['materials'][c]['Edit']) for c in CASES):
        raise RuntimeError('an f_patch delivery is outside {None, Edit}')
    sysG = {v: {c: (per[c][v]['G_Edit'] if deliv[c] != 'None' else 0.0) for c in CASES} for v in ('orig', 'bn_re')}
    system = {v: {'mean': statistics.fmean(sysG[v].values()), 'without_Q4_G1': statistics.fmean(x for c, x in sysG[v].items() if c != 'D01_Q_Q4_G1'),
                  'by_case': sysG[v]} for v in ('orig', 'bn_re')}
    system['delivered'] = deliv
    system['package_D01_f_patch_vs_None'] = src['arms_vs_none']['f_patch']['by_domain']['D01']
    q = per['D01_Q_Q4_G1']
    focus = {v: {arm: {'error_origin2': q[v]['per_origin'][arm][1], 'bias_origin2': q[v]['bias'][arm][1]} for arm in ('None', 'NoMix', 'Edit')} for v in ('orig', 'bn_re')}
    robust = original_robustness()
    res = {'task': TASK, 'written_local': now(), 'plan': plan, 'checks': {'orig_equals_package_scores': all(per[c]['orig_equals_package'] for c in CASES),
                                                                          'learned_params_bitwise_unchanged': all(per[c]['params_unchanged'] for c in CASES),
                                                                          'system_orig_equals_package': abs(system['orig']['mean'] - system['package_D01_f_patch_vs_None']) < 1e-9,
                                                                          'nomix_orig_equals_package': abs(summ['orig']['G_NoMix'] - src['refs_vs_none']['NoMix']['by_domain']['D01']) < 1e-9},
           'summary': summ, 'system_f_patch_electricity': system, 'q4_g1_origin2': focus, 'per_case': per, 'original_robustness': robust}
    write(ROOT / 'result.json', res)
    (ROOT / 'REPORT.md').write_text(report_md(res), encoding='utf-8')
    return res


def original_robustness() -> dict:
    """The original PatchTST package results (kept as the main result) with median, trimmed mean (TRIM) and leave-out-Q4_G1 added."""
    r = read(SRC / 'result.json')
    pc = r['per_case']
    doms = ('D01', 'D02', 'D03')
    getters = {k: (lambda c, k=k: pc[c][k]) for k in ('patch_minus_f0', 'patch_minus_mlp', 'patch_minus_naive', 'mlp_minus_f0', 'naive_minus_f0')}
    getters.update({'%s_vs_None' % a: (lambda c, a=a: pc[c]['G_vs_None'][a]) for a in ('f_patch', 'f_mlp', 'f0', 'f_naive')})
    getters.update({'%s_vs_None' % a: (lambda c, a=a: pc[c]['G_refs_vs_None'][a]) for a in ('NoMix', 'Fixed_source_P')})
    out = {}
    for name, get in getters.items():
        dm = {d: [get(c) for c in pc if pc[c]['domain'] == d] for d in doms}
        dm_wo = {d: [get(c) for c in pc if pc[c]['domain'] == d and c != 'D01_Q_Q4_G1'] for d in doms}
        out[name] = {'mean_three_domain': statistics.fmean(statistics.fmean(v) for v in dm.values()),
                     'median_three_domain': statistics.fmean(statistics.median(v) for v in dm.values()),
                     'median_all_cases': statistics.median(get(c) for c in pc),
                     'trimmed_three_domain': statistics.fmean(trimmed(v) for v in dm.values()),
                     'without_Q4_G1_three_domain': statistics.fmean(statistics.fmean(v) for v in dm_wo.values()),
                     'electricity': {'mean': statistics.fmean(dm['D01']), 'median': statistics.median(dm['D01']), 'trimmed': trimmed(dm['D01']),
                                     'without_Q4_G1': statistics.fmean(dm_wo['D01'])}}
        main = r['main'].get(name, {}).get('three_domain')
        if main is not None and abs(main - out[name]['mean_three_domain']) > 1e-9:
            raise RuntimeError('recomputed mean differs from the package main result: %s' % name)
    return out


def report_md(res: dict) -> str:
    f = lambda x: '%+.2f' % x
    s, sysm, fo, rb = res['summary'], res['system_f_patch_electricity'], res['q4_g1_origin2'], res['original_robustness']
    L = ['# PatchTST BatchNorm 机制诊断（电力 16 例 × 3 seed，已有 checkpoint，不训练、不调 LLM）', '',
         '身份：机制诊断（已曝光开发 Test）。每个 checkpoint 按原流程评分，并在复制品上冻结全部学习参数、只用该案例 T 段父窗输入重估 BatchNorm 运行统计后再评分。'
         '重估：批 64、按 parents.npz 行序、1 遍、momentum=None（累计平均）、仅 BN 层 train()、no_grad。', '',
         '核对：原流程评分与原包存值逐位相同 %s；重估前后学习参数逐位不变 %s。' % (res['checks']['orig_equals_package_scores'], res['checks']['learned_params_bitwise_unchanged']), '',
         '## 1. 增强相对各自 None 的收益（pp，电力 16 例等权）', '',
         '| | NoMix 原 | NoMix 重估 | Edit 原 | Edit 重估 |', '|---|---:|---:|---:|---:|',
         '| 均值 | %s | %s | %s | %s |' % (f(s['orig']['G_NoMix']), f(s['bn_re']['G_NoMix']), f(s['orig']['G_Edit']), f(s['bn_re']['G_Edit'])),
         '| 中位数 | %s | %s | %s | %s |' % (f(s['orig']['G_NoMix_median']), f(s['bn_re']['G_NoMix_median']), f(s['orig']['G_Edit_median']), f(s['bn_re']['G_Edit_median'])),
         '| 去掉 Q4_G1 | %s | %s | %s | %s |' % (f(s['orig']['G_NoMix_without_Q4_G1']), f(s['bn_re']['G_NoMix_without_Q4_G1']), f(s['orig']['G_Edit_without_Q4_G1']), f(s['bn_re']['G_Edit_without_Q4_G1'])),
         '| 胜例数 / 16 | %d | %d | %d | %d |' % (s['orig']['wins_NoMix'], s['bn_re']['wins_NoMix'], s['orig']['wins_Edit'], s['bn_re']['wins_Edit'])]
    for p in ('Q1', 'Q2', 'Q3', 'Q4'):
        L.append('| 时期 %s | %s | %s | %s | %s |' % (p, f(s['orig']['G_NoMix_by_period'][p]), f(s['bn_re']['G_NoMix_by_period'][p]),
                                                    f(s['orig']['G_Edit_by_period'][p]), f(s['bn_re']['G_Edit_by_period'][p])))
    for i in range(4):
        L.append('| 起点 %d | %s | %s | %s | %s |' % (i + 1, f(s['orig']['G_by_origin']['NoMix'][i]), f(s['bn_re']['G_by_origin']['NoMix'][i]),
                                                    f(s['orig']['G_by_origin']['Edit'][i]), f(s['bn_re']['G_by_origin']['Edit'][i])))
    L += ['', '重估对 None 自身：%s pp（正 = 重估后更好）。' % f(s['None_bn_re_vs_orig']), '',
          '## 2. Q4_G1 第二起点（误差 = 归一化 MSE；偏差 = 归一化预测 − 真值的均值）', '', '| 模型 | 原误差 | 重估误差 | 原偏差 | 重估偏差 |', '|---|---:|---:|---:|---:|']
    for arm in ('None', 'NoMix', 'Edit'):
        L.append('| %s | %.3f | %.3f | %+.3f | %+.3f |' % (arm, fo['orig'][arm]['error_origin2'], fo['bn_re'][arm]['error_origin2'], fo['orig'][arm]['bias_origin2'], fo['bn_re'][arm]['bias_origin2']))
    L += ['', '## 3. 系统读数：按冻结卡的原交付（11 例 None 仍为 None）', '',
          '- 新卡（f_patch）电力，相对同一口径的 None：原 %s → 重估 %s；去掉 Q4_G1：原 %s → 重估 %s。' % (f(sysm['orig']['mean']), f(sysm['bn_re']['mean']), f(sysm['orig']['without_Q4_G1']), f(sysm['bn_re']['without_Q4_G1'])),
          '', '## 4. 原包结果的稳健性补充（原全样本主结果保留）', '', '截尾规则：%s' % TRIM['rule'], '',
          '| 比较 | 原均值 | 中位数（域内再三域等权） | 截尾均值 | 去掉 Q4_G1 | 电力 均值 / 中位 / 截尾 / 去掉 Q4_G1 |', '|---|---:|---:|---:|---:|---|']
    for k, v in rb.items():
        e = v['electricity']
        L.append('| %s | %s | %s | %s | %s | %s / %s / %s / %s |' % (k, f(v['mean_three_domain']), f(v['median_three_domain']), f(v['trimmed_three_domain']), f(v['without_Q4_G1_three_domain']),
                                                               f(e['mean']), f(e['median']), f(e['trimmed']), f(e['without_Q4_G1'])))
    return '\n'.join(L) + '\n'


def run() -> None:
    freeze_plan()
    env = dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env.update(PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    procs = []
    for k in range(WORKERS):
        f = (ROOT / 'logs' / ('worker_%d.log' % k)).open('w', encoding='utf-8')
        procs.append((subprocess.Popen([sys.executable, '-B', '-u', '-m', MODULE, '--worker', str(k), str(WORKERS)], cwd=str(REPO), env=env, stdout=f,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)), f))
    rcs = [p.wait() for p, _ in procs]
    for _, f in procs:
        f.close()
    if any(rcs) or any(not (ROOT / 'cells' / ('%s.json' % c)).exists() for c in CASES):
        raise RuntimeError('worker failure: %s' % rcs)
    res = report()
    print('DONE', json.dumps({k: res['checks'][k] for k in res['checks']}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--worker', nargs=2, type=int)
    a = ap.parse_args()
    if a.worker:
        return worker(*a.worker)
    if a.run:
        run()
    if a.report:
        report()


if __name__ == '__main__':
    main()
