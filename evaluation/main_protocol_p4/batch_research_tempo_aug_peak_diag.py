"""DEV-TEMPO-AUG-SOURCE-ALIGNMENT §3.2: RD02 peak diagnostic on cached predictions (0 fits, 0 LLM, no rerun).

Reads only the already-exposed E predictions of DEV-DATA-AUG-OPERATOR-PROBE (None, FixedMixup, Shock20, MixCross40; 2000-update
Consumer; all origins and seeds of RD02_T1 / RD02_STP1) and the E truth rows those predictions were scored against. Writes
_scratch/dev_tempo_aug_source_alignment/peak_diagnostic/{peak_diagnostic.json, peak_table.md, peak_figure.png}.
The top-10% segment is a post-hoc description on exposed truth, not a deployment-visible feature. No origin is removed.
  python -B -m evaluation.main_protocol_p4.batch_research_tempo_aug_peak_diag
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import readiness as rd
from evaluation.main_protocol_p4 import batch_research_data_aug_operator_probe as P

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / '_scratch' / 'dev_tempo_aug_source_alignment' / 'peak_diagnostic'
JOBS = ('RD02_T1', 'RD02_STP1')
MIDS = ('None', 'FixedMixup', 'Shock20', 'MixCross40')
TOP_Q = 0.90


def _load(job: str):
    js = rd.resolve_job('RD02', job)
    ctx = rd.open_job('RD02', job, P.RUN, stage='e_target')            # exposed E truth rows only; scaler from the probe run
    truth = rd.read_truth(ctx.slice, js.e)                             # (N, O, H) raw, NaN = unobserved
    preds = {}
    for m in MIDS:
        for s in P.SEEDS[job]:
            with np.load(P.RUN / job / 'pred_e' / (P.cid(job, m, s, P.N_FULL) + '.npz')) as z:
                if not np.array_equal(z['origins'], np.array(js.e)):
                    raise RuntimeError('cached E origins differ')
                preds[(m, s)] = z['pred_raw'].copy()
    return js, ctx.scaler, truth, preds


def _job(job: str) -> dict:
    js, sc, truth, preds = _load(job)
    N, O, H = truth.shape
    obs = np.isfinite(truth)
    scale = sc.scale[:, None, None]
    seeds = P.SEEDS[job]
    # reproduce the frozen probe score (binding check)
    frozen = P.context.read_json(P.RUN / job / 'e_scores.json')['cells']
    for (m, s), pr in preds.items():
        v = rd.score_block(pr, truth, sc)['normalized_mse_macro']
        if abs(v - frozen[P.cid(job, m, s, P.N_FULL)]['e']['normalized_mse_macro']) > 1e-12:
            raise RuntimeError('E score not reproduced for %s %s' % (m, s))
    # top-10% observed truth cells per (entity, origin): post-hoc peak segment
    top = np.zeros_like(obs)
    for e in range(N):
        for o in range(O):
            v = truth[e, o][obs[e, o]]
            if v.size:
                top[e, o] = obs[e, o] & (truth[e, o] >= np.quantile(v, TOP_Q))
    n_obs_e = obs.sum(axis=(1, 2))
    rows = {}
    for m in MIDS:
        per = []
        for s in seeds:
            pr = preds[(m, s)]
            sq = np.where(obs, ((pr - np.where(obs, truth, 0.0)) / scale) ** 2, 0.0)
            res = np.where(obs, (pr - np.where(obs, truth, 0.0)) / scale, np.nan)
            # origin contribution to the macro: mean_e (SSE_eo / n_obs_e)
            contrib_o = [float(np.mean(sq[:, o].sum(axis=1) / n_obs_e)) for o in range(O)]
            macro = float(sum(contrib_o))
            ent_o = [(sq[:, o].sum(axis=1) / n_obs_e) for o in range(O)]
            peak_bias, peak_under = [], 0
            for e in range(N):
                for o in range(O):
                    if obs[e, o].any():
                        h = int(np.nanargmax(np.where(obs[e, o], truth[e, o], -np.inf)))
                        b = float((pr[e, o, h] - truth[e, o, h]) / sc.scale[e])
                        peak_bias.append(b)
                        peak_under += b < 0
            per.append({'seed': s, 'macro': macro, 'origin_contrib': contrib_o,
                        'origin_top1_entity_share': [float(x.max() / x.sum()) if x.sum() > 0 else None for x in ent_o],
                        'origin_top3_entity_share': [float(np.sort(x)[-3:].sum() / x.sum()) if x.sum() > 0 else None for x in ent_o],
                        'origin_mean_signed_residual': [float(np.nanmean(res[:, o])) for o in range(O)],
                        'origin_top10_mean_signed_residual': [float(np.nanmean(np.where(top[:, o], res[:, o], np.nan))) for o in range(O)],
                        'origin_rest_mean_signed_residual': [float(np.nanmean(np.where(obs[:, o] & ~top[:, o], res[:, o], np.nan))) for o in range(O)],
                        'origin_top10_share_of_sse': [float(np.where(top[:, o], sq[:, o], 0).sum() / sq[:, o].sum()) for o in range(O)],
                        'peak_time_bias_mean': float(np.mean(peak_bias)), 'peak_time_under_fraction': peak_under / len(peak_bias),
                        'peak_time_bias_by_origin': [float(np.mean([peak_bias[e * O + o] for e in range(N)])) for o in range(O)]})
        rows[m] = per

    def avg(m, key, o=None):
        vals = [(r[key][o] if o is not None else r[key]) for r in rows[m]]
        return statistics.fmean(vals)

    def paired(m, o=None):
        d = [(a['origin_contrib'][o] if o is not None else a['macro']) - (b['origin_contrib'][o] if o is not None else b['macro'])
             for a, b in zip(rows[m], rows['None'])]
        return {'mean': statistics.fmean(d), 'se': statistics.stdev(d) / math.sqrt(len(d)), 'per_seed': d}

    none_macro = avg('None', 'macro')
    summary = {'origins': list(js.e), 'none_macro_mean': none_macro,
               'truth_origin_level': [float(np.nanmean(truth[:, o])) for o in range(O)],
               'truth_origin_p90_over_entities': [float(np.nanmean([np.nanquantile(truth[e, o], .9) for e in range(N) if obs[e, o].any()])) for o in range(O)],
               'T_mean_over_entities': float(np.mean(sc.mean)),
               'materials': {}}
    for m in MIDS:
        summary['materials'][m] = {
            'macro_mean': avg(m, 'macro'),
            'paired_macro_minus_none': paired(m) if m != 'None' else None,
            'origin_share_of_macro': [avg(m, 'origin_contrib', o) / avg(m, 'macro') for o in range(O)],
            'origin_paired_minus_none': [paired(m, o) for o in range(O)] if m != 'None' else None,
            'origin_top1_entity_share': [avg(m, 'origin_top1_entity_share', o) for o in range(O)],
            'origin_top3_entity_share': [avg(m, 'origin_top3_entity_share', o) for o in range(O)],
            'origin_mean_signed_residual': [avg(m, 'origin_mean_signed_residual', o) for o in range(O)],
            'origin_top10_mean_signed_residual': [avg(m, 'origin_top10_mean_signed_residual', o) for o in range(O)],
            'origin_rest_mean_signed_residual': [avg(m, 'origin_rest_mean_signed_residual', o) for o in range(O)],
            'origin_top10_share_of_sse': [avg(m, 'origin_top10_share_of_sse', o) for o in range(O)],
            'peak_time_bias_mean': avg(m, 'peak_time_bias_mean'), 'peak_time_under_fraction': avg(m, 'peak_time_under_fraction'),
            'peak_time_bias_by_origin': [avg(m, 'peak_time_bias_by_origin', o) for o in range(O)],
            'per_seed': rows[m]}
    return summary, (js, sc, truth, preds)


def _figure(data: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, job in zip(axes, JOBS):
        summ, (js, sc, truth, preds) = data[job]
        o = int(np.argmax(summ['materials']['None']['origin_share_of_macro']))
        share = summ['materials']['None']['origin_top1_entity_share'][o]
        ent_loss = np.nanmean(np.stack([np.nanmean(np.where(np.isfinite(truth[:, o]), ((preds[('None', s)][:, o] - truth[:, o]) / sc.scale[:, None]) ** 2, np.nan), axis=1)
                                        for s in P.SEEDS[job]]), axis=0)
        e = int(np.nanargmax(ent_loss))
        ax.plot(truth[e, o], 'k-', lw=2, label='observed truth')
        for m, c in zip(MIDS, ('tab:blue', 'tab:orange', 'tab:red', 'tab:green')):
            ax.plot(np.mean([preds[(m, s)][e, o] for s in P.SEEDS[job]], axis=0), color=c, lw=1.2, label=m + ' (3-seed mean)')
        ax.set_title('%s, highest-loss origin %d (row %d), entity_%d\nNone: origin share %.0f%% of macro, top entity %.0f%% of origin' % (
            job, o, js.e[o], e, 100 * summ['materials']['None']['origin_share_of_macro'][o], 100 * share), fontsize=9)
        ax.set_xlabel('horizon hour')
        ax.set_ylabel('raw PM2.5')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def _table(res: dict) -> str:
    out = ['# RD02 峰值诊断（0 拟合，读已缓存预测；全部 origin，不删除）', '',
           '单位：E 宏损失分量为 normalized MSE；残差为 (预测 − 真值)/T scale，负值 = 低估。top10% = 每个实体在该 origin 已观测真值最高 10% 的时刻（事后描述，部署不可见）。', '']
    for job in JOBS:
        s = res[job]
        O = len(s['origins'])
        out += ['## %s（None 宏均值 %.3f；T 均值 %.1f）' % (job, s['none_macro_mean'], s['T_mean_over_entities']), '',
                '| origin（行） | 真值均值 | 实体 p90 均值 | None 占宏损失 | None top1 实体占比 | top3 占比 | None 平均残差 | None top10% 残差 | 其余残差 | top10% 占 SSE | None 峰值时刻偏差 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        n = s['materials']['None']
        for o in range(O):
            out.append('| %d（%d） | %.1f | %.1f | %.0f%% | %.0f%% | %.0f%% | %+.2f | %+.2f | %+.2f | %.0f%% | %+.2f |' % (
                o, s['origins'][o], s['truth_origin_level'][o], s['truth_origin_p90_over_entities'][o], 100 * n['origin_share_of_macro'][o],
                100 * n['origin_top1_entity_share'][o], 100 * n['origin_top3_entity_share'][o], n['origin_mean_signed_residual'][o],
                n['origin_top10_mean_signed_residual'][o], n['origin_rest_mean_signed_residual'][o], 100 * n['origin_top10_share_of_sse'][o],
                n['peak_time_bias_by_origin'][o]))
        out += ['', '材料相对 None 的配对差（宏损失分量，正 = 更差；3 seed 均值 ± SE）：', '',
                '| 材料 | 宏损失 | ' + ' | '.join('origin %d' % o for o in range(O)) + ' | 峰值时刻偏差（全 origin） | top10% 残差（主 origin） |',
                '|---|---:|' + '---:|' * O + '---:|---:|']
        main_o = int(np.argmax(n['origin_share_of_macro']))
        for m in MIDS[1:]:
            r = s['materials'][m]
            cells = ['%+.3f ± %.3f' % (x['mean'], x['se']) for x in r['origin_paired_minus_none']]
            out.append('| %s | %+.3f ± %.3f | %s | %+.2f（None %+.2f） | %+.2f（None %+.2f） |' % (
                m, r['paired_macro_minus_none']['mean'], r['paired_macro_minus_none']['se'], ' | '.join(cells), r['peak_time_bias_mean'],
                n['peak_time_bias_mean'], r['origin_top10_mean_signed_residual'][main_o], n['origin_top10_mean_signed_residual'][main_o]))
        out.append('')
    return '\n'.join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {job: _job(job) for job in JOBS}
    res = {job: data[job][0] for job in JOBS}
    res['_meta'] = {'source': str(P.ROOT), 'materials': MIDS, 'consumer_updates': P.N_FULL, 'top_quantile': TOP_Q,
                    'note': 'Exposed development data; post-hoc description. No origin removed; high-loss periods are not called contamination.'}
    (OUT / 'peak_diagnostic.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    (OUT / 'peak_table.md').write_text(_table(res), encoding='utf-8')
    _figure(data, OUT / 'peak_figure.png')
    print(_table(res))


if __name__ == '__main__':
    main()
