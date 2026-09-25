"""DEV-TEMPO-AUG-TEMPORAL-FEEDBACK-REPLAY (docs/DEV_TEMPO_AUG_TEMPORAL_FEEDBACK_REPLAY_TASK_2026-09-19.md): does selecting one of the four
public augmentation plans by the *late* feedback (t_h+192..336, the deployment-matched block) of the four most recent finished batches beat the
current batch's single-block C_A argmin (R_CA), the *early* feedback of the same four batches (R_HistNear) and two fixed defaults, on the E of
the seven sequential RD02 batches L1..L7?

Everything is a replay over frozen caches: 0 fits, 0 optimizer updates, 0 LLM. The only new numbers are the E-block predictions and scores of
the 48 initial-history models (V1/T2/T4/Q2, `historical_late`), computed once in a torch subprocess from the cached weights, the cached scaler
and the Baseline-Linear serving inputs, then scored with the unchanged missing-aware scorer. Old caches are read only; every new file goes to
_scratch/dev_tempo_aug_temporal_feedback_replay/.

  --preflight   inventory + frozen_config.json (0 torch)          --smoke     roster / cutoff / merge-vs-scorer / selection-API checks + one
  --historical  freeze then score the 48 historical E predictions              cached-model C_A re-computation (wiring, torch subprocess)
  --replay      L1 -> L7 selections, then the independent E scorer, result.json / tables.md / REPORT.md
  --run         all of the above in order under the 30-minute numeric cap
  workers (torch allowed, subprocess only): --worker-wiring OUT | --worker-hist-freeze JOB | --worker-hist-score JOB
The controller never imports torch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import context, spec
from methods.ttha.batch_base import readiness as rd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / '_scratch' / 'dev_tempo_aug_temporal_feedback_replay'
MODULE = 'evaluation.main_protocol_p4.batch_research_tempo_aug_temporal_feedback_replay'
TASK = 'docs/DEV_TEMPO_AUG_TEMPORAL_FEEDBACK_REPLAY_TASK_2026-09-19.md'
PACKAGE = 'DEV-TEMPO-AUG-TEMPORAL-FEEDBACK-REPLAY'
DATASET = 'RD02'
EXPOSURE = 'EXPOSED_DEVELOPMENT_REPLAY'
PUBLIC = ('None', 'FixedMixup', 'P_AmpResample', 'P_NoMixRecipe')
SEEDS = (20269181, 20269182, 20269183)
W = REPO / '_scratch' / 'dev_tempo_aug_workflow_skill'
D = REPO / '_scratch' / 'dev_tempo_aug_decision_learning'
BRANCH = {'RD02_V1': W / 'source' / 'RD02_V1_no_skill', 'RD02_T2': W / 'source' / 'RD02_T2_no_skill',
          'RD02_T4': W / 'select' / 'RD02_T4_no_skill', 'RD02_Q2': W / 'select' / 'RD02_Q2_no_skill',
          'RD02_L1': W / 'target' / 'RD02_L1_f_no_skill', 'RD02_L2': W / 'target' / 'RD02_L2_f_no_skill',
          'RD02_L3': W / 'target' / 'RD02_L3_f_no_skill', 'RD02_L4': W / 'target' / 'RD02_L4_f_no_skill',
          'RD02_L5': D / 'test' / 'RD02_L5_f0', 'RD02_L6': D / 'test' / 'RD02_L6_f0', 'RD02_L7': D / 'test' / 'RD02_L7_f0'}
JOB_T = {'RD02_V1': 5760, 'RD02_T2': 8160, 'RD02_T4': 10560, 'RD02_Q2': 12960, 'RD02_L1': 15360, 'RD02_L2': 16560, 'RD02_L3': 18960,
         'RD02_L4': 20160, 'RD02_L5': 21360, 'RD02_L6': 22560, 'RD02_L7': 24480}
INITIAL = ('RD02_V1', 'RD02_T2', 'RD02_T4', 'RD02_Q2')
REPLAY = ('RD02_L1', 'RD02_L2', 'RD02_L3', 'RD02_L4', 'RD02_L5', 'RD02_L6', 'RD02_L7')
ORDER = INITIAL + REPLAY
K_HIST = 4
FEEDBACK_SPAN = 384                      # a history h is usable for j only when t_h + 384 <= t_j (its E block [t_h+192, t_h+384) is finished)
HISTORY = {'RD02_L1': ('RD02_V1', 'RD02_T2', 'RD02_T4', 'RD02_Q2'), 'RD02_L2': ('RD02_T2', 'RD02_T4', 'RD02_Q2', 'RD02_L1'),
           'RD02_L3': ('RD02_T4', 'RD02_Q2', 'RD02_L1', 'RD02_L2'), 'RD02_L4': ('RD02_Q2', 'RD02_L1', 'RD02_L2', 'RD02_L3'),
           'RD02_L5': ('RD02_L1', 'RD02_L2', 'RD02_L3', 'RD02_L4'), 'RD02_L6': ('RD02_L2', 'RD02_L3', 'RD02_L4', 'RD02_L5'),
           'RD02_L7': ('RD02_L3', 'RD02_L4', 'RD02_L5', 'RD02_L6')}
STRATEGIES = ('R_CA', 'R_HistNear', 'R_HistLate', 'Fixed_initial', 'Fixed_NoMix', 'UniformRandom_expectation')
TIE_ABS = 1e-12
WIRING_TOL_ABS = 1e-5                    # frozen before the wiring run: |macro| and max |per-entity| difference to the cached C_A
WIRING_CELL = ('RD02_V1', 'None', 20269181)
NUMERIC_WALL_S = 1800.
SUB_TIMEOUT_S = 600.
SEALED_FROM_ROW = 24864
T975_DF2 = 4.302652729911275
PRIMARY = ('R_CA', 'R_HistLate')          # Delta_j(A, B) > 0 means B is better
COMPARISONS = (('R_CA', 'R_HistLate'), ('Fixed_initial', 'R_HistLate'), ('Fixed_NoMix', 'R_HistLate'), ('R_HistNear', 'R_HistLate'),
               ('UniformRandom_expectation', 'R_HistLate'), ('R_CA', 'R_HistNear'), ('Fixed_NoMix', 'R_CA'), ('Fixed_initial', 'R_CA'),
               ('Fixed_NoMix', 'Fixed_initial'), ('UniformRandom_expectation', 'R_CA'))


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def wj(path, obj) -> None:
    context.write_json(path, obj)


def rj(path):
    return context.read_json(path)


def cell_id(job: str, m: str, s: int) -> str:
    return '%s__%s__s%d' % (job, m, s)


def job_t(job: str) -> int:
    return rd.resolve_job(DATASET, job).t


# ============================================================================= cache readers (old roots read-only)
def resolve_model_path(p: str) -> Path:
    """The cached model_path is a Windows absolute path of this repository; resolve it by its suffix below `_scratch` so the same cell
    resolves on any mount of the same repository. Never guessed from a file name."""
    parts = Path(p.replace('\\', '/')).parts
    if '_scratch' not in parts:
        raise FileNotFoundError('model_path not inside _scratch: %s' % p)
    rel = Path(*parts[parts.index('_scratch'):])
    q = REPO / rel
    if not q.exists():
        raise FileNotFoundError('cached model missing: %s' % q)
    return q


def public_cells(job: str) -> dict:
    out = {}
    for p in sorted((BRANCH[job] / job / 'cells').glob('*.json')):
        c = rj(p)
        if c.get('status') == 'OK' and not c.get('tag') and c['material_id'] in PUBLIC and c['model_seed'] in SEEDS:
            out[(c['material_id'], c['model_seed'])] = c
    missing = [(m, s) for m in PUBLIC for s in SEEDS if (m, s) not in out]
    if missing:
        raise RuntimeError('public cells missing in %s: %s' % (job, missing))
    return out


def block_scores(job: str, blk: str) -> dict:
    """{(material, seed): score dict} of one cached block: c_a from the cells, c_b / e from the label files of the same branch."""
    if blk == 'c_a':
        return {k: c['scores']['c_a'] for k, c in public_cells(job).items()}
    f = BRANCH[job] / job / ('%s_scores.json' % blk)
    if not f.exists():
        raise FileNotFoundError('%s has no cached %s block' % (job, blk))
    out = {}
    for v in rj(f)['cells'].values():
        if v['material_id'] in PUBLIC and v['model_seed'] in SEEDS:
            out[(v['material_id'], v['model_seed'])] = v[blk]
    return out


def merge_blocks(a: dict, b: dict) -> dict:
    """Missing-aware merge of two scored blocks of the same entities: per-entity SSE = per_entity_normalized_mse x observed cells, summed over
    the blocks, divided by the summed observed cells; the coverage rule is re-applied to the merged block. Equals score_block on the
    concatenated origins (smoke-checked), unlike the 0.5/0.5 average of the two macros when the observed counts differ."""
    na, nb = a['coverage']['observed_cells_per_entity'], b['coverage']['observed_cells_per_entity']
    N = len(na)
    cells = a['coverage']['cells_per_entity'] + b['coverage']['cells_per_entity']
    nonfinite = a['n_nonfinite_predictions'] + b['n_nonfinite_predictions']
    per, n_obs = [], []
    for e in range(N):
        pa, pb = a['per_entity_normalized_mse'][e], b['per_entity_normalized_mse'][e]
        sse = (pa * na[e] if pa is not None else 0.0) + (pb * nb[e] if pb is not None else 0.0)
        n = na[e] + nb[e]
        ok = n > 0 and (pa is not None or na[e] == 0) and (pb is not None or nb[e] == 0)
        per.append(float(sse / n) if ok else None)
        n_obs.append(n)
    coverage = [n / cells for n in n_obs]
    not_scorable = [e for e in range(N) if coverage[e] < rd.COVERAGE_MIN]
    status = 'NONFINITE_PREDICTIONS' if nonfinite else ('NOT_SCORABLE' if not_scorable or any(p is None for p in per) else 'SCORABLE')
    return {'status': status, 'normalized_mse_macro': float(np.mean(per)) if status == 'SCORABLE' else None, 'n_entities': N,
            'n_origins': a['n_origins'] + b['n_origins'], 'n_nonfinite_predictions': nonfinite, 'per_entity_normalized_mse': per,
            'coverage': {'cells_per_entity': cells, 'observed_cells_per_entity': n_obs, 'min_fraction_rule': rd.COVERAGE_MIN,
                         'min_observed_fraction': float(min(coverage)), 'not_scorable_entities': not_scorable, 'observed_cells_total': int(sum(n_obs))},
            'block_macros': {'first': a['normalized_mse_macro'], 'second': b['normalized_mse_macro']},
            'semantics': 'merged observed cells of both blocks per entity (SSE / observed cells), equal-weight entity mean'}


def early_scores(job: str) -> dict:
    ca, cb = block_scores(job, 'c_a'), block_scores(job, 'c_b')
    return {k: merge_blocks(ca[k], cb[k]) for k in ca if k in cb}


def macro3(scores: dict, m: str):
    """[3 seed macros] of one material or (None, reason) when the material is incomplete under the original coverage rule."""
    vals = []
    for s in SEEDS:
        v = scores.get((m, s))
        if v is None:
            return None, 'cell missing %s' % cell_id('?', m, s)
        if v['status'] != 'SCORABLE' or v['normalized_mse_macro'] is None:
            return None, 'status %s for seed %d' % (v['status'], s)
        vals.append(float(v['normalized_mse_macro']))
    return vals, ''


# ============================================================================= history bank with the time cutoff (the only path to history)
class HistoryBank:
    """Serves early / late feedback of a history job h to a current job j only when h is in the frozen H(j) and t_h + 384 <= t_j. Late for the
    initial four comes from this package's historical_late_scores.json; late for L1..L7 from the cached e_scores.json. It never serves the
    current job's own C_B / E, and logs every access."""

    def __init__(self, hist_root: Path):
        self.hist_root = hist_root
        self.log = []

    def _late_file(self, h: str) -> Path:
        return (self.hist_root / h / 'historical_late_scores.json') if h in INITIAL else (BRANCH[h] / h / 'e_scores.json')

    def check(self, h: str, j: str) -> None:
        if h == j:
            raise PermissionError('%s asked for its own feedback' % j)
        if h not in HISTORY.get(j, ()):
            raise PermissionError('%s is not in the frozen history of %s' % (h, j))
        if job_t(h) + FEEDBACK_SPAN > job_t(j):
            raise PermissionError('%s (t=%d) is not finished before %s (t=%d)' % (h, job_t(h), j, job_t(j)))

    def get(self, h: str, j: str, blk: str) -> dict:
        self.check(h, j)
        if blk == 'early':
            sc, src = early_scores(h), [str(BRANCH[h] / h / 'cells'), str(BRANCH[h] / h / 'c_b_scores.json')]
        elif blk == 'late':
            f = self._late_file(h)
            if not f.exists():
                raise FileNotFoundError('late feedback of %s not available: %s' % (h, f))
            sc = {(v['material_id'], v['model_seed']): v['e'] for v in rj(f)['cells'].values() if v['material_id'] in PUBLIC and v['model_seed'] in SEEDS}
            src = [str(f)]
        else:
            raise ValueError(blk)
        self.log.append({'current_job': j, 'history_job': h, 'block': blk, 'sources': src, 't_h_plus_384': job_t(h) + FEEDBACK_SPAN, 't_j': job_t(j)})
        return sc


def derive_history(j: str) -> tuple:
    """The K_HIST most recent table jobs before j whose feedback is finished; must equal the frozen table."""
    done = [h for h in ORDER if ORDER.index(h) < ORDER.index(j) and job_t(h) + FEEDBACK_SPAN <= job_t(j)]
    return tuple(done[-K_HIST:])


# ============================================================================= the six deterministic strategies
def argmin_public(values: dict) -> str:
    """Lowest value; ties within TIE_ABS broken by the public order."""
    lo = min(values[m] for m in PUBLIC)
    return next(m for m in PUBLIC if values[m] - lo <= TIE_ABS)


def select_ca(current_ca: dict) -> dict:
    means, per_seed = {}, {}
    for m in PUBLIC:
        v, why = macro3(current_ca, m)
        if v is None:
            return {'status': 'INCOMPLETE', 'reason': why, 'material': None}
        means[m], per_seed[m] = statistics.fmean(v), v
    return {'status': 'OK', 'material': argmin_public(means), 'criterion': means, 'per_seed': per_seed}


def select_history(bank: HistoryBank, j: str, hist: tuple, blk: str) -> dict:
    """J_blk(j, m) = mean over h in hist of [mean_s L(h, m, s, blk) / mean_s L(h, None, s, blk)]; argmin with public tie-break."""
    terms = {m: [] for m in PUBLIC}
    detail = {}
    for h in hist:
        sc = bank.get(h, j, blk)
        none, why = macro3(sc, 'None')
        if none is None:
            return {'status': 'INCOMPLETE', 'reason': '%s None %s' % (h, why), 'material': None, 'history': list(hist)}
        den = statistics.fmean(none)
        if not den > 0:
            return {'status': 'INCOMPLETE', 'reason': '%s None denominator non-positive' % h, 'material': None, 'history': list(hist)}
        detail[h] = {'none_mean': den}
        for m in PUBLIC:
            v, why = macro3(sc, m)
            if v is None:
                return {'status': 'INCOMPLETE', 'reason': '%s %s %s' % (h, m, why), 'material': None, 'history': list(hist)}
            terms[m].append(statistics.fmean(v) / den)
            detail[h][m] = {'per_seed': v, 'mean': statistics.fmean(v), 'ratio_to_none': statistics.fmean(v) / den}
    J = {m: statistics.fmean(terms[m]) for m in PUBLIC}
    return {'status': 'OK', 'material': argmin_public(J), 'criterion': J, 'per_history': detail, 'history': list(hist), 'block': blk}


def selections_for(j: str, current_ca: dict, bank: HistoryBank, fixed_initial: dict) -> dict:
    """All selections of job j from the current C_A and the legal history only. The current C_B / E never enter here."""
    hist = HISTORY[j]
    if tuple(hist) != derive_history(j):
        raise RuntimeError('frozen history of %s differs from the rule' % j)
    for h in hist:
        bank.check(h, j)
    out = {'job': j, 't': job_t(j), 'history': list(hist), 'selected_local': now(),
           'R_CA': select_ca(current_ca), 'R_HistNear': select_history(bank, j, hist, 'early'), 'R_HistLate': select_history(bank, j, hist, 'late'),
           'Fixed_initial': {'status': fixed_initial['status'], 'material': fixed_initial['material'], 'frozen_from': list(INITIAL)},
           'Fixed_NoMix': {'status': 'OK', 'material': 'P_NoMixRecipe'},
           'UniformRandom_expectation': {'status': 'OK', 'material': None, 'note': 'expectation over the four public materials, no draw'}}
    return out


# ============================================================================= independent E scorer (after the selections are saved)
def e_scores_of(j: str) -> dict:
    return block_scores(j, 'e')


def per_origin_macro(score: dict) -> list:
    return [float(np.mean([x for x in row])) if all(x is not None for x in row) else None for row in score['per_origin_entity_normalized_mse']]


def score_job(j: str, sel: dict, e: dict) -> dict:
    E = {}
    for m in PUBLIC:
        v, why = macro3(e, m)
        if v is None:
            raise RuntimeError('E of %s %s incomplete: %s' % (j, m, why))
        E[m] = v
    none_mean = statistics.fmean(E['None'])
    means = {m: statistics.fmean(E[m]) for m in PUBLIC}
    ranked = sorted(PUBLIC, key=lambda m: (means[m], PUBLIC.index(m)))
    rank = {m: ranked.index(m) + 1 for m in PUBLIC}
    oracle = argmin_public(means)
    origins = {m: [statistics.fmean(x) for x in zip(*[per_origin_macro(e[(m, s)]) for s in SEEDS])] for m in PUBLIC}
    strat = {}
    for st in STRATEGIES:
        s = sel[st]
        if s['status'] != 'OK':
            strat[st] = {'status': s['status'], 'material': None, 'e_per_seed': None, 'e_mean': None, 'e_rank': None, 'regret_pp': None}
            continue
        if st == 'UniformRandom_expectation':
            per_seed = [statistics.fmean(E[m][i] for m in PUBLIC) for i in range(len(SEEDS))]
            mean = statistics.fmean(per_seed)
            strat[st] = {'status': 'OK', 'material': None, 'e_per_seed': per_seed, 'e_mean': mean, 'e_rank': None,
                         'regret_pp': 100 * (mean - means[oracle]) / none_mean, 'e_per_origin': [statistics.fmean(origins[m][o] for m in PUBLIC) for o in range(4)]}
        else:
            m = s['material']
            strat[st] = {'status': 'OK', 'material': m, 'e_per_seed': E[m], 'e_mean': means[m], 'e_rank': rank[m],
                         'regret_pp': 100 * (means[m] - means[oracle]) / none_mean, 'e_per_origin': origins[m]}
    comp = {}
    for a, b in COMPARISONS:
        A, B = strat[a], strat[b]
        if A['status'] != 'OK' or B['status'] != 'OK':
            comp['%s->%s' % (a, b)] = {'status': 'INCOMPLETE'}
            continue
        d = [100 * (x - y) / none_mean for x, y in zip(A['e_per_seed'], B['e_per_seed'])]
        se = statistics.stdev(d) / math.sqrt(len(d))
        same = A['material'] is not None and A['material'] == B['material']
        comp['%s->%s' % (a, b)] = {'status': 'OK', 'delta_pp': statistics.fmean(d), 'per_seed_pp': d, 'se_pp': se, 't95_pp': [statistics.fmean(d) - T975_DF2 * se, statistics.fmean(d) + T975_DF2 * se],
                                   'same_delivery': same, 'outcome': 'tie' if same else ('B_better' if statistics.fmean(d) > 0 else ('A_better' if statistics.fmean(d) < 0 else 'equal'))}
    return {'job': j, 't': job_t(j), 'e_none_mean': none_mean, 'e_materials': {m: {'per_seed': E[m], 'mean': means[m], 'rank': rank[m], 'per_origin_mean': origins[m],
                                                                                      'delta_vs_none_pp': 100 * (none_mean - means[m]) / none_mean} for m in PUBLIC},
            'oracle': oracle, 'e_origins': list(rd.resolve_job(DATASET, j).e), 'strategies': strat, 'comparisons': comp,
            'e_source': str(BRANCH[j] / j / 'e_scores.json')}


# ============================================================================= aggregation over jobs (job = unit)
def aggregate(scored: list) -> dict:
    out = {}
    for a, b in COMPARISONS:
        key = '%s->%s' % (a, b)
        rows = [(r['job'], r['comparisons'][key]) for r in scored if r['comparisons'][key]['status'] == 'OK']
        if not rows:
            out[key] = {'status': 'INCOMPLETE', 'n_jobs': 0}
            continue
        d = [c['delta_pp'] for _, c in rows]
        loo = {j: statistics.fmean(x for jj, x in zip([r for r, _ in rows], d) if jj != j) for j, _ in rows} if len(rows) > 1 else {}
        out[key] = {'status': 'OK', 'n_jobs': len(rows), 'jobs': [j for j, _ in rows], 'mean_delta_pp': statistics.fmean(d), 'per_job_delta_pp': dict(zip([j for j, _ in rows], d)),
                    'se_over_jobs_pp': (statistics.stdev(d) / math.sqrt(len(d))) if len(d) > 1 else None,
                    'wins_B': sum(c['outcome'] == 'B_better' for _, c in rows), 'ties': sum(c['outcome'] in ('tie', 'equal') for _, c in rows),
                    'losses_B': sum(c['outcome'] == 'A_better' for _, c in rows), 'changed_delivery': sum(not c['same_delivery'] for _, c in rows),
                    'max_harm_to_B_pp': min(d), 'max_gain_to_B_pp': max(d), 'leave_one_job_out_mean_pp': loo,
                    'loo_min_pp': min(loo.values()) if loo else None, 'loo_max_pp': max(loo.values()) if loo else None,
                    'sign_stable_under_loo': (all(v > 0 for v in loo.values()) or all(v < 0 for v in loo.values())) if loo else None}
    per_strategy = {}
    for st in STRATEGIES:
        rows = [r for r in scored if r['strategies'][st]['status'] == 'OK']
        per_strategy[st] = {'n_jobs': len(rows), 'materials': {r['job']: r['strategies'][st]['material'] for r in rows},
                            'mean_regret_pp': statistics.fmean(r['strategies'][st]['regret_pp'] for r in rows) if rows else None,
                            'ranks': {r['job']: r['strategies'][st]['e_rank'] for r in rows},
                            'n_rank1': sum(r['strategies'][st]['e_rank'] == 1 for r in rows), 'n_rank4': sum(r['strategies'][st]['e_rank'] == 4 for r in rows),
                            'mean_delta_vs_none_pp': statistics.fmean(100 * (r['e_none_mean'] - r['strategies'][st]['e_mean']) / r['e_none_mean'] for r in rows) if rows else None}
    return {'comparisons': out, 'per_strategy': per_strategy, 'unit': 'job (7 batches); the 3 seeds only describe training randomness'}


# ============================================================================= budget / wall ledger (numeric subprocesses only)
class Ledger:
    def __init__(self):
        self.p = OUT / 'budget.json'
        self.s = rj(self.p) if self.p.exists() else {'package': PACKAGE, 'numeric_wall_cap_s': NUMERIC_WALL_S, 'numeric_wall_used_s': 0.0, 'subprocesses': [],
                                                     'model_inferences': 0, 'fits': 0, 'optimizer_updates': 0, 'llm_requests': 0, 'http_requests': 0, 'new_sha': 0, 'git_commits': 0,
                                                     'stopped_for_wall': False}

    def save(self):
        wj(self.p, self.s)

    def remaining(self) -> float:
        return NUMERIC_WALL_S - self.s['numeric_wall_used_s']

    def run(self, args: list, log: Path, inferences: int) -> int:
        if self.remaining() <= 0:
            self.s['stopped_for_wall'] = True
            self.save()
            raise RuntimeError('numeric wall cap reached; partial facts kept')
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.pop('KMP_DUPLICATE_LIB_OK', None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        t0 = time.time()
        try:
            with log.open('w', encoding='utf-8') as fh:
                p = subprocess.run([sys.executable, '-B', '-m', MODULE] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=min(SUB_TIMEOUT_S, max(1.0, self.remaining())), env=env)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -999
        secs = time.time() - t0
        self.s['numeric_wall_used_s'] += secs
        self.s['subprocesses'].append({'args': [str(a) for a in args], 'rc': rc, 'seconds': secs, 'log': str(log), 'local': now()})
        if rc == 0:
            self.s['model_inferences'] += inferences
        self.save()
        return rc


# ============================================================================= preflight + frozen config
def environment() -> dict:
    return {'python': sys.version.split()[0], 'executable': sys.executable, 'numpy': np.__version__, 'KMP_DUPLICATE_LIB_OK_at_start': os.environ.get('KMP_DUPLICATE_LIB_OK'),
            'workers_env_kmp_removed': True, 'controller_imports_torch': 'torch' in sys.modules}


def inventory() -> dict:
    inv = {}
    for j in ORDER:
        cells = public_cells(j)
        models = {cell_id(j, m, s): str(resolve_model_path(cells[(m, s)]['model_path'])) for m in PUBLIC for s in SEEDS}
        cb = block_scores(j, 'c_b')
        cb_equal = all(abs(cells[k]['scores']['c_b']['normalized_mse_macro'] - cb[k]['normalized_mse_macro']) == 0 for k in cells if 'c_b' in cells[k]['scores'])
        jd = BRANCH[j] / j
        inv[j] = {'t': job_t(j), 'branch': str(BRANCH[j]), 'public_cells': len(cells), 'models_present': sum(Path(v).exists() for v in models.values()),
                  'n_updates': sorted({c['n_updates'] for c in cells.values()}), 'profile': sorted({c['profile'] for c in cells.values()}),
                  'c_a_origins': sorted({c['scores']['c_a']['n_origins'] for c in cells.values()}), 'c_b_scores': (jd / 'c_b_scores.json').exists(),
                  'c_b_in_cells_equals_file': cb_equal, 'e_scores': (jd / 'e_scores.json').exists(), 'scaler': (jd / 'scaler.npz').exists(),
                  'model_paths': models, 'rows_read_c_a': sorted({tuple(c['rows_read']) for c in cells.values()})}
    return inv


def frozen_config() -> dict:
    jobs = {}
    for j in ORDER:
        js = rd.resolve_job(DATASET, j)
        jobs[j] = {'t': js.t, 'train_range': list(js.train_range), 'c_a': list(js.c_a), 'c_b': list(js.c_b), 'e': list(js.e),
                   'e_target_rows': list(js.e_target_rows), 'e_input_rows': list(js.e_input_rows), 'role': 'initial_history' if j in INITIAL else 'sequential_replay'}
        assert js.e_target_rows[1] <= SEALED_FROM_ROW
    return {'package': PACKAGE, 'task': TASK, 'frozen_local': now(), 'exposure': EXPOSURE, 'dataset': DATASET, 'roster': list(rd.DATASETS[DATASET]['roster']),
            'public_materials_order': list(PUBLIC), 'seeds': list(SEEDS), 'consumer': 'cached identity (shared MLP, 2000 updates, batch 64; see any cell.consumer)',
            'jobs': jobs, 'order': list(ORDER), 'history_table': {j: list(h) for j, h in HISTORY.items()}, 'history_rule': 'the %d most recent table jobs with t_h + %d <= t_j; no similarity, recency weighting or performance filter' % (K_HIST, FEEDBACK_SPAN),
            'feedback_span': FEEDBACK_SPAN, 'cache_branches': {j: str(p) for j, p in BRANCH.items()},
            'blocks': {'early': 'merged C_A (t+0, t+48) and C_B (t+96, t+144): per-entity SSE / observed cells over the four origins, coverage rule 0.25 re-applied',
                       'late': 'E (t+192, t+240, t+288, t+336) scored by the unchanged missing-aware scorer; initial four computed here as historical_late, L1..L7 read from cache'},
            'strategies': {'R_CA': 'argmin_m mean_s C_A_j(m)', 'R_HistNear': 'argmin_m mean_h [mean_s early_h(m) / mean_s early_h(None)]',
                           'R_HistLate': 'argmin_m mean_h [mean_s late_h(m) / mean_s late_h(None)]', 'Fixed_initial': 'R_HistLate formula on V1/T2/T4/Q2 once, frozen before L1, never updated',
                           'Fixed_NoMix': 'always P_NoMixRecipe', 'UniformRandom_expectation': 'mean over the four public materials of E (per seed), no draw',
                           'tie_break': 'public order when |difference| <= %g' % TIE_ABS, 'incomplete': 'any missing / NOT_SCORABLE material or non-positive None denominator -> INCOMPLETE, no fallback'},
            'metrics': {'delta_pp': 'Delta_j(A,B) = 100 (mean_s E_j(A) - mean_s E_j(B)) / mean_s E_j(None); > 0 means B better',
                        'paired_seed_vector': '100 (E_js(A) - E_js(B)) / mean_s E_j(None); SE over 3 seeds, df=2 t95 = %.6f' % T975_DF2,
                        'aggregate': 'equal-weight mean over the 7 jobs; wins / ties / losses; max harm; leave-one-job-out means', 'oracle': 'argmin_m mean_s E_j(m), diagnostic only; regret in the same pp units'},
            'wiring': {'cell': list(WIRING_CELL), 'tolerance_abs': WIRING_TOL_ABS, 'what': 'recompute C_A of one cached model with the recomputed T scaler and Baseline-Linear inputs; compare macro and per-entity to the cached cell'},
            'budget': {'fits': 0, 'optimizer_updates': 0, 'llm': 0, 'http': 0, 'new_sha': 0, 'git_commit': 0, 'max_model_inferences': 49, 'numeric_wall_cap_s': NUMERIC_WALL_S, 'single_worker': True},
            'sealed_from_row': SEALED_FROM_ROW, 'environment': environment()}


def preflight() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    for j in ORDER:
        hist_ok = (j in INITIAL) or (tuple(HISTORY[j]) == derive_history(j))
        if not hist_ok:
            raise RuntimeError('frozen history table of %s violates the rule' % j)
    cfg = frozen_config()
    cfg['inventory'] = inv
    fc = OUT / 'frozen_config.json'
    hl = OUT / 'historical_late'
    if fc.exists() and any((hl / h / 'historical_late_scores.json').exists() for h in INITIAL):
        old = rj(fc)
        keys = ('history_table', 'strategies', 'metrics', 'wiring', 'blocks', 'public_materials_order', 'seeds', 'jobs')
        if any(old[k] != cfg[k] for k in keys):
            raise RuntimeError('frozen_config.json differs from the frozen formulas after historical results were opened; refusing to re-freeze')
        return old
    wj(fc, cfg)
    print('PREFLIGHT_OK', {j: (v['public_cells'], v['models_present'], v['e_scores']) for j, v in inv.items()})
    return cfg


# ============================================================================= smoke (controller: 0 torch) + wiring (subprocess)
def smoke_controller() -> dict:
    out = {'local': now(), 'checks': {}}
    # 1) history rule reproduces the frozen table; every pair satisfies the cutoff; a future / same / unlisted job is refused
    out['checks']['history_table_equals_rule'] = all(tuple(HISTORY[j]) == derive_history(j) for j in REPLAY)
    out['checks']['cutoff_all_pairs'] = all(job_t(h) + FEEDBACK_SPAN <= job_t(j) for j in REPLAY for h in HISTORY[j])
    bank = HistoryBank(OUT / 'historical_late')
    refused, accepted = {}, {}
    for h, j in (('RD02_L2', 'RD02_L1'), ('RD02_L1', 'RD02_L1'), ('RD02_L7', 'RD02_L6'), ('RD02_V1', 'RD02_L2')):     # future / same / future / dropped-out-of-window
        try:
            bank.check(h, j)
            refused['%s->%s' % (h, j)] = False
        except PermissionError as exc:
            refused['%s->%s' % (h, j)] = str(exc)
    for h, j in (('RD02_L5', 'RD02_L6'), ('RD02_Q2', 'RD02_L1'), ('RD02_L3', 'RD02_L7')):                              # legal, finished histories
        try:
            bank.check(h, j)
            accepted['%s->%s' % (h, j)] = True
        except PermissionError as exc:
            accepted['%s->%s' % (h, j)] = str(exc)
    out['checks']['refusals'] = refused
    out['checks']['accepted'] = accepted
    out['checks']['refuses_future_same_unlisted'] = all(bool(v) for v in refused.values())
    out['checks']['accepts_legal_history'] = all(v is True for v in accepted.values())
    # the cutoff predicate itself: a history whose E block is not finished before t_j (t_h + 384 > t_j) is unusable
    finished = lambda t_h, t_j: t_h + FEEDBACK_SPAN <= t_j
    out['checks']['rule_excludes_unfinished'] = finished(job_t('RD02_L6'), job_t('RD02_L7')) and not finished(24200, 24480) and not finished(job_t('RD02_L7') - 383, job_t('RD02_L7'))
    # 2) merged four-origin score == direct scorer on the concatenated origins (with missing cells and unequal observed counts)
    rng = np.random.RandomState(20260919)
    N, O, H = 12, 4, spec.H
    truth = rng.randn(N, O, H) * 50 + 80
    truth[rng.rand(N, O, H) < 0.15] = np.nan
    truth[3, 0, :] = np.nan                      # one origin entirely unobserved for one entity -> unequal counts between the halves
    truth[7, 2:, :30] = np.nan
    pred = truth + rng.randn(N, O, H) * 20
    pred = np.where(np.isnan(pred), rng.randn(N, O, H) * 50 + 80, pred)
    from methods.ttha.batch_base import data as _data
    sc = _data.Scaler(mean=np.full(N, 80.0), std=np.full(N, 50.0), scale=np.full(N, 50.0), floor_hits=0)
    direct = rd.score_block(pred, truth, sc)
    a, b = rd.score_block(pred[:, :2], truth[:, :2], sc), rd.score_block(pred[:, 2:], truth[:, 2:], sc)
    merged = merge_blocks(a, b)
    naive = 0.5 * a['normalized_mse_macro'] + 0.5 * b['normalized_mse_macro']
    out['checks']['merge_equals_direct_macro'] = abs(merged['normalized_mse_macro'] - direct['normalized_mse_macro']) <= 1e-12
    out['checks']['merge_equals_direct_per_entity'] = max(abs(x - y) for x, y in zip(merged['per_entity_normalized_mse'], direct['per_entity_normalized_mse'])) <= 1e-12
    out['checks']['merge_coverage_equals_direct'] = merged['coverage']['observed_cells_per_entity'] == direct['coverage']['observed_cells_per_entity']
    out['checks']['naive_half_half_differs'] = abs(naive - direct['normalized_mse_macro']) > 1e-9
    out['checks']['merge_values'] = {'direct': direct['normalized_mse_macro'], 'merged': merged['normalized_mse_macro'], 'naive_half_half': naive}
    # coverage rule: an entity below 0.25 in the merged block makes it NOT_SCORABLE
    t2 = truth.copy()
    t2[5, :, :] = np.nan
    t2[5, 0, :10] = truth[5, 0, :10] if np.isfinite(truth[5, 0, :10]).all() else 1.0
    a2, b2 = rd.score_block(pred[:, :2], t2[:, :2], sc), rd.score_block(pred[:, 2:], t2[:, 2:], sc)
    out['checks']['merge_applies_coverage_rule'] = merge_blocks(a2, b2)['status'] == 'NOT_SCORABLE' and rd.score_block(pred, t2, sc)['status'] == 'NOT_SCORABLE'
    # 3) the selection functions accept only the current C_A and the bank; the bank refuses the current job; tie-break follows the public order
    out['checks']['tie_break_public_order'] = argmin_public({'None': 1.0, 'FixedMixup': 1.0 - 5e-13, 'P_AmpResample': 1.0 + 1e-13, 'P_NoMixRecipe': 2.0}) == 'None'
    out['checks']['argmin_strict'] = argmin_public({'None': 1.0, 'FixedMixup': 1.0 - 1e-6, 'P_AmpResample': 1.0, 'P_NoMixRecipe': 2.0}) == 'FixedMixup'
    try:
        bank.get('RD02_L1', 'RD02_L1', 'late')
        out['checks']['bank_refuses_current_job'] = False
    except PermissionError:
        out['checks']['bank_refuses_current_job'] = True
    import inspect
    out['checks']['selection_signature'] = str(inspect.signature(selections_for))
    out['checks']['selection_has_no_current_e_argument'] = all(k not in str(inspect.signature(selections_for)) for k in ('c_b', 'e_scores', 'current_e'))
    out['checks']['controller_imports_torch'] = 'torch' in sys.modules
    out['all_pass'] = all(v is True for k, v in out['checks'].items() if isinstance(v, bool) and k != 'controller_imports_torch') and not out['checks']['controller_imports_torch']
    return out


def worker_wiring(out_dir: str) -> None:
    """One cached model: recompute its C_A with the recomputed T scaler / legal parents (must equal the cached scaler.npz) and Baseline-Linear
    inputs; compare to the cached cell. 0 optimizer updates. Writes wiring.json into out_dir."""
    rd._init_openmp_before_torch()
    from methods.ttha.batch_base import train
    job, m, s = WIRING_CELL
    cell = public_cells(job)[(m, s)]
    run_dir = Path(out_dir)
    ctx = rd.open_job(DATASET, job, run_dir, stage='evaluate')             # rows [t-672, t+96) only; recomputes scaler + legal parents
    with np.load(BRANCH[job] / job / 'scaler.npz') as old:
        same_scaler = all(np.array_equal(old[k], v) for k, v in (('mean', ctx.scaler.mean), ('scale', ctx.scaler.scale), ('legal_ent', ctx.legal.ent), ('legal_k', ctx.legal.k)))
    Xs, serve = rd.serve_inputs(ctx.slice, ctx.job.c_a, [[] for _ in ctx.job.roster], ctx.scaler)
    truth = rd.read_truth(ctx.slice, ctx.job.c_a)
    model = train.load_model(str(resolve_model_path(cell['model_path'])))
    pred = rd.predict_inputs(model, Xs, ctx.scaler)
    sc = rd.score_block(pred, truth, ctx.scaler)
    ref = cell['scores']['c_a']
    d_macro = abs(sc['normalized_mse_macro'] - ref['normalized_mse_macro'])
    d_ent = max(abs(x - y) for x, y in zip(sc['per_entity_normalized_mse'], ref['per_entity_normalized_mse']))
    import torch
    rec = {'cell': cell_id(job, m, s), 'model_path': str(resolve_model_path(cell['model_path'])), 'scaler_and_legal_parents_identical_to_cache': bool(same_scaler),
           'serve_inputs_equal_to_cache': serve == cell['serve_inputs_c_a'], 'recomputed_macro': sc['normalized_mse_macro'], 'cached_macro': ref['normalized_mse_macro'],
           'abs_diff_macro': d_macro, 'max_abs_diff_per_entity': d_ent, 'tolerance_abs': WIRING_TOL_ABS, 'pass': bool(same_scaler and d_macro <= WIRING_TOL_ABS and d_ent <= WIRING_TOL_ABS),
           'optimizer_updates': 0, 'rows_read': list(ctx.job.evaluate_rows), 'torch': torch.__version__, 'torch_threads': torch.get_num_threads(), 'cached_torch': cell.get('torch'),
           'cached_threads': cell.get('torch_threads'), 'local': now()}
    wj(run_dir / 'wiring.json', rec)
    print('WIRING', json.dumps({k: rec[k] for k in ('pass', 'abs_diff_macro', 'max_abs_diff_per_entity', 'scaler_and_legal_parents_identical_to_cache')}), flush=True)


def smoke(ledger: Ledger) -> dict:
    sdir = OUT / 'smoke'
    sdir.mkdir(parents=True, exist_ok=True)
    rec = smoke_controller()
    wj(sdir / 'smoke_controller.json', rec)
    if not rec['all_pass']:
        raise RuntimeError('controller smoke failed: %s' % json.dumps(rec['checks'], default=str)[:800])
    rc = ledger.run(['--worker-wiring', str(sdir / 'wiring')], sdir / 'wiring.log', inferences=1)
    w = rj(sdir / 'wiring' / 'wiring.json') if rc == 0 and (sdir / 'wiring' / 'wiring.json').exists() else {'pass': False, 'rc': rc}
    rec['wiring'] = w
    wj(sdir / 'smoke.json', rec)
    if not w.get('pass'):
        raise RuntimeError('wiring check failed: %s' % json.dumps(w, default=str)[:800])
    print('SMOKE_OK', json.dumps({k: v for k, v in w.items() if k in ('abs_diff_macro', 'max_abs_diff_per_entity', 'pass')}))
    return rec


# ============================================================================= historical_late (initial four): freeze predictions, then score
def worker_hist_freeze(job: str) -> None:
    rd._init_openmp_before_torch()
    from methods.ttha.batch_base import train
    run_dir = OUT / 'historical_late'
    jd = run_dir / job
    cells = public_cells(job)
    ctx = rd.open_job(DATASET, job, run_dir, stage='e_input')              # rows [t, t+336): inputs of the four E origins; no E target row
    js = ctx.job
    assert js.e_input_rows[1] <= SEALED_FROM_ROW
    Xs, serve = rd.serve_inputs(ctx.slice, js.e, [[] for _ in js.roster], ctx.scaler)
    (jd / 'predictions_e').mkdir(exist_ok=True)
    out = {'job_id': job, 'package': PACKAGE, 'block': 'historical_late', 'started_local': now(), 'rows_read': list(js.e_input_rows), 'origins': list(js.e), 'serve': serve, 'models': {}}
    for m in PUBLIC:
        for s in SEEDS:
            c = cells[(m, s)]
            mp = resolve_model_path(c['model_path'])
            pred = rd.predict_inputs(train.load_model(str(mp)), Xs, ctx.scaler)
            cid = cell_id(job, m, s)
            p = jd / 'predictions_e' / (cid + '.npz')
            np.savez(p, pred_raw=pred, origins=np.array(js.e))
            out['models'][cid] = {'path': str(p), 'material_id': m, 'model_seed': s, 'model_path': str(mp), 'cached_cell': str(BRANCH[job] / job / 'cells' / (cid + '.json')),
                                  'n_nonfinite': int((~np.isfinite(pred)).sum())}
    out['frozen_local'] = now()
    wj(jd / 'historical_late_frozen.json', out)
    print('HIST_FREEZE_OK', job, len(out['models']), flush=True)


def worker_hist_score(job: str) -> None:
    rd._init_openmp_before_torch()
    run_dir = OUT / 'historical_late'
    jd = run_dir / job
    fz = rj(jd / 'historical_late_frozen.json')
    ctx = rd.open_job(DATASET, job, run_dir, stage='e_target')             # rows [t+192, t+384): the newly opened historical targets
    js = ctx.job
    assert js.e_target_rows[1] <= SEALED_FROM_ROW
    truth = rd.read_truth(ctx.slice, js.e)
    out = {'job_id': job, 'package': PACKAGE, 'block': 'historical_late', 'scored_local': now(), 'frozen_local': fz['frozen_local'], 'rows_read': list(js.e_target_rows),
           'origins': list(js.e), 'new_exposure_rows': list(js.e_target_rows), 'cells': {}}
    for cid, m in fz['models'].items():
        with np.load(m['path']) as z:
            if not np.array_equal(z['origins'], np.array(js.e)):
                raise RuntimeError('frozen origins differ')
            pred = z['pred_raw'].copy()
        out['cells'][cid] = {'material_id': m['material_id'], 'model_seed': m['model_seed'], 'e': rd.score_block(pred, truth, ctx.scaler)}
    wj(jd / 'historical_late_scores.json', out)
    print('HIST_SCORE_OK', job, {c: round(v['e']['normalized_mse_macro'], 4) if v['e']['normalized_mse_macro'] is not None else v['e']['status'] for c, v in out['cells'].items()}, flush=True)


def historical(ledger: Ledger) -> dict:
    run_dir = OUT / 'historical_late'
    rec = {}
    for h in INITIAL:
        jd = run_dir / h
        jd.mkdir(parents=True, exist_ok=True)
        if not (jd / 'scaler.npz').exists():
            shutil.copyfile(BRANCH[h] / h / 'scaler.npz', jd / 'scaler.npz')      # byte copy of the cached scaler / legal-parent identity (old dir untouched)
        if not (jd / 'historical_late_frozen.json').exists():
            rc = ledger.run(['--worker-hist-freeze', h], jd / 'freeze.log', inferences=12)
            if rc != 0:
                raise RuntimeError('historical freeze failed for %s (rc %s)' % (h, rc))
        if not (jd / 'historical_late_scores.json').exists():
            rc = ledger.run(['--worker-hist-score', h], jd / 'score.log', inferences=0)
            if rc != 0:
                raise RuntimeError('historical score failed for %s (rc %s)' % (h, rc))
        sc = rj(jd / 'historical_late_scores.json')
        rec[h] = {'new_exposure_rows': sc['new_exposure_rows'], 'origins': sc['origins'], 'n_models': len(sc['cells']),
                  'statuses': sorted({v['e']['status'] for v in sc['cells'].values()}), 'scaler_copied_from': str(BRANCH[h] / h / 'scaler.npz')}
    wj(OUT / 'historical_late' / 'summary.json', {'package': PACKAGE, 'local': now(), 'jobs': rec,
                                                   'note': 'new exposure: the E targets [t+192, t+384) of V1/T2/T4/Q2 are scored for the first time in this package (development feedback, recorded, not claimed as 0 new labels)'})
    print('HISTORICAL_OK', json.dumps(rec, default=str)[:600])
    return rec


# ============================================================================= replay L1 -> L7
def replay() -> dict:
    bank = HistoryBank(OUT / 'historical_late')
    # Fixed_initial: the R_HistLate formula on the initial four, frozen before L1; the bank check is done with L1 as the requester
    fixed_initial = select_history(bank, 'RD02_L1', INITIAL, 'late')
    fixed_initial['frozen_before'] = 'RD02_L1'
    fixed_initial['frozen_local'] = now()
    wj(OUT / 'fixed_initial.json', fixed_initial)
    selections, scored = [], []
    for j in REPLAY:
        sel = selections_for(j, block_scores(j, 'c_a'), bank, fixed_initial)     # only current C_A + legal history enter
        selections.append(sel)
        wj(OUT / 'selections.json', {'package': PACKAGE, 'fixed_initial': fixed_initial, 'jobs': selections, 'history_access_log': bank.log, 'saved_local': now()})
        # independent scorer: only after this job's selections are on disk does the cached E of the job get read
        scored.append(score_job(j, sel, e_scores_of(j)))
        wj(OUT / 'scored_jobs.json', {'package': PACKAGE, 'jobs': scored, 'saved_local': now()})
    agg = aggregate(scored)
    res = {'package': PACKAGE, 'task': TASK, 'exposure': EXPOSURE, 'finished_local': now(), 'fixed_initial': fixed_initial, 'selections': selections, 'scored': scored,
           'aggregate': agg, 'history_access_log': bank.log, 'budget': rj(OUT / 'budget.json') if (OUT / 'budget.json').exists() else None,
           'historical_late': rj(OUT / 'historical_late' / 'summary.json') if (OUT / 'historical_late' / 'summary.json').exists() else None,
           'smoke': rj(OUT / 'smoke' / 'smoke.json') if (OUT / 'smoke' / 'smoke.json').exists() else None}
    wj(OUT / 'result.json', res)
    write_tables(res)
    write_report(res)
    print('REPLAY_OK', json.dumps({k: (v.get('mean_delta_pp'), v.get('wins_B'), v.get('ties'), v.get('losses_B')) for k, v in agg['comparisons'].items()}, default=str))
    return res


# ============================================================================= tables + report
def f(x, nd=4):
    return '—' if x is None else ('%.*f' % (nd, x))


def fp(x):
    return '—' if x is None else ('%+.2f' % x)


def write_tables(res: dict) -> None:
    L = ['# %s — tables' % PACKAGE, '', 'Units: E = missing-aware normalized MSE macro (raw); Δ in pp of the job\'s None mean E (> 0 = second strategy better). Job is the unit; 3 seeds = training randomness only.', '']
    L += ['## A. Per job: materials on E (mean of 3 seeds), rank, Δ vs None', '', '| Job | t | E(None) | ' + ' | '.join(PUBLIC[1:]) + ' | oracle |', '|---|---:|---:|' + '---:|' * len(PUBLIC[1:]) + '---|']
    for r in res['scored']:
        L.append('| %s | %d | %s | %s | %s |' % (r['job'], r['t'], f(r['e_none_mean']),
                                                 ' | '.join('%s (r%d, %s)' % (f(r['e_materials'][m]['mean']), r['e_materials'][m]['rank'], fp(r['e_materials'][m]['delta_vs_none_pp'])) for m in PUBLIC[1:]), r['oracle']))
    L += ['', '## B. Per job x strategy: history, selection criterion, delivery, C_A, E per seed, rank, regret', '',
          '| Job | strategy | history H(j) | selected | criterion (C_A mean / J) | E s1 | E s2 | E s3 | E mean | rank | regret pp | Δ vs R_CA | Δ vs Fixed_initial | Δ vs Fixed_NoMix |',
          '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r, s in zip(res['scored'], res['selections']):
        for st in STRATEGIES:
            x, sel = r['strategies'][st], s[st]
            crit = ''
            if sel.get('status') == 'OK' and sel.get('criterion'):
                crit = ' / '.join('%s %.4f' % (m[:6], sel['criterion'][m]) for m in PUBLIC)
            hist = ', '.join(h.replace('RD02_', '') for h in sel.get('history', [])) if st in ('R_HistNear', 'R_HistLate') else ('V1,T2,T4,Q2 (frozen)' if st == 'Fixed_initial' else '—')
            dcol = lambda a: (fp(r['comparisons']['%s->%s' % (a, st)]['delta_pp']) if '%s->%s' % (a, st) in r['comparisons'] and r['comparisons']['%s->%s' % (a, st)]['status'] == 'OK' else ('0.00' if a == st else '—'))
            es = x['e_per_seed'] or [None] * 3
            L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (r['job'], st, hist, x['material'] or ('%s' % x['status'] if x['status'] != 'OK' else 'expectation'), crit,
                                                                                                    f(es[0]), f(es[1]), f(es[2]), f(x['e_mean']), x['e_rank'] or '—', fp(x['regret_pp']), dcol('R_CA'), dcol('Fixed_initial'), dcol('Fixed_NoMix')))
    L += ['', '## C. Aggregate comparisons (A -> B; Δ > 0 = B better), job = unit', '', '| A -> B | n | mean Δ pp | SE over jobs | wins B | ties | losses B | deliveries changed | max harm to B | max gain to B | LOO min | LOO max | LOO sign stable |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for k, v in res['aggregate']['comparisons'].items():
        if v['status'] != 'OK':
            L.append('| %s | 0 | INCOMPLETE | | | | | | | | | | |' % k)
            continue
        L.append('| %s | %d | %s | %s | %d | %d | %d | %d | %s | %s | %s | %s | %s |' % (k, v['n_jobs'], fp(v['mean_delta_pp']), f(v['se_over_jobs_pp'], 2), v['wins_B'], v['ties'], v['losses_B'], v['changed_delivery'],
                                                                                          fp(v['max_harm_to_B_pp']), fp(v['max_gain_to_B_pp']), fp(v['loo_min_pp']), fp(v['loo_max_pp']), v['sign_stable_under_loo']))
    L += ['', '## D. Per-job paired seed vectors of the primary comparison R_CA -> R_HistLate (pp of None)', '', '| Job | R_CA | R_HistLate | Δ pp | per seed | SE | t95 |', '|---|---|---|---:|---|---:|---|']
    for r in res['scored']:
        c = r['comparisons']['R_CA->R_HistLate']
        if c['status'] == 'OK':
            L.append('| %s | %s | %s | %s | %s | %s | [%s, %s] |' % (r['job'], r['strategies']['R_CA']['material'], r['strategies']['R_HistLate']['material'], fp(c['delta_pp']), ', '.join(fp(x) for x in c['per_seed_pp']), f(c['se_pp'], 2), fp(c['t95_pp'][0]), fp(c['t95_pp'][1])))
    L += ['', '## E. Per-origin E (mean of seeds, macro over entities) of the delivered material per strategy', '', '| Job | origins | None | ' + ' | '.join(STRATEGIES) + ' |', '|---|---|---|' + '---|' * len(STRATEGIES)]
    for r in res['scored']:
        cell = lambda st: ('/'.join(f(x, 3) for x in r['strategies'][st]['e_per_origin']) if r['strategies'][st]['status'] == 'OK' else '—')
        L.append('| %s | %s | %s | %s |' % (r['job'], '/'.join(str(o) for o in r['e_origins']), '/'.join(f(x, 3) for x in r['e_materials']['None']['per_origin_mean']), ' | '.join(cell(st) for st in STRATEGIES)))
    L += ['', '## F. Historical feedback used by the selectors: ratio to None per history job (early = C_A+C_B merged, late = E)', '']
    for s in res['selections']:
        for st in ('R_HistNear', 'R_HistLate'):
            sel = s[st]
            if sel['status'] != 'OK':
                L.append('- %s %s: INCOMPLETE (%s)' % (s['job'], st, sel.get('reason')))
                continue
            L.append('- %s %s -> %s; J = %s; per history: %s' % (s['job'], st, sel['material'], ', '.join('%s %.4f' % (m, sel['criterion'][m]) for m in PUBLIC),
                                                                 '; '.join('%s [%s]' % (h.replace('RD02_', ''), ', '.join('%s %.3f' % (m[:5], sel['per_history'][h][m]['ratio_to_none']) for m in PUBLIC[1:])) for h in sel['history'])))
    (OUT / 'tables.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def write_report(res: dict) -> None:
    A, S = res['aggregate'], res['scored']
    c = A['comparisons']
    main, fi, fn, near = c['R_CA->R_HistLate'], c['Fixed_initial->R_HistLate'], c['Fixed_NoMix->R_HistLate'], c['R_HistNear->R_HistLate']
    ps = A['per_strategy']
    b = res['budget'] or {}
    hl = res['historical_late'] or {}
    deliveries = {st: [r['strategies'][st]['material'] for r in S] for st in STRATEGIES}
    q1 = ('R_HistLate 相对 R_CA：七批等权配对差 %s pp（SE %s，胜/平/负 %d/%d/%d，交付改变 %d/7 批，最大伤害 %s pp，最大收益 %s pp；留一 Job 均值区间 [%s, %s]，符号%s）。逐批：%s。原始 E（None 均值）逐批：%s。'
          % (fp(main.get('mean_delta_pp')), f(main.get('se_over_jobs_pp'), 2), main.get('wins_B', 0), main.get('ties', 0), main.get('losses_B', 0), main.get('changed_delivery', 0), fp(main.get('max_harm_to_B_pp')), fp(main.get('max_gain_to_B_pp')),
             fp(main.get('loo_min_pp')), fp(main.get('loo_max_pp')), '稳定' if main.get('sign_stable_under_loo') else '不稳定',
             '; '.join('%s %s→%s %s' % (r['job'].replace('RD02_', ''), r['strategies']['R_CA']['material'], r['strategies']['R_HistLate']['material'], fp(r['comparisons']['R_CA->R_HistLate'].get('delta_pp'))) for r in S),
             '; '.join('%s %.4f' % (r['job'].replace('RD02_', ''), r['e_none_mean']) for r in S))) if main['status'] == 'OK' else 'INCOMPLETE'
    same_fi = fi['status'] == 'OK' and fi['changed_delivery'] == 0
    same_fn = fn['status'] == 'OK' and fn['changed_delivery'] == 0
    q2 = ('对 Fixed_initial（%s）：%s；对 Fixed_NoMix：%s。' % (res['fixed_initial'].get('material'),
                                                   ('七批全部同交付，增量为 0' if same_fi else ('等权 %s pp，胜/平/负 %d/%d/%d，交付不同 %d 批，最大伤害 %s pp' % (fp(fi.get('mean_delta_pp')), fi.get('wins_B', 0), fi.get('ties', 0), fi.get('losses_B', 0), fi.get('changed_delivery', 0), fp(fi.get('max_harm_to_B_pp'))))) if fi['status'] == 'OK' else 'INCOMPLETE',
                                                   ('七批全部同交付，增量为 0' if same_fn else ('等权 %s pp，胜/平/负 %d/%d/%d，交付不同 %d 批，最大伤害 %s pp' % (fp(fn.get('mean_delta_pp')), fn.get('wins_B', 0), fn.get('ties', 0), fn.get('losses_B', 0), fn.get('changed_delivery', 0), fp(fn.get('max_harm_to_B_pp'))))) if fn['status'] == 'OK' else 'INCOMPLETE'))
    q3 = ('R_HistLate 对 R_HistNear：%s。两者交付序列 Near=%s / Late=%s。' % (('七批全部同交付，Δ=0，不能说匹配后期时段被证明必要' if near['changed_delivery'] == 0 else '等权 %s pp，胜/平/负 %d/%d/%d，交付不同 %d 批' % (fp(near.get('mean_delta_pp')), near.get('wins_B', 0), near.get('ties', 0), near.get('losses_B', 0), near.get('changed_delivery', 0))) if near['status'] == 'OK' else 'INCOMPLETE',
                                                            ','.join(str(x) for x in deliveries['R_HistNear']), ','.join(str(x) for x in deliveries['R_HistLate'])))
    ranks = lambda st: ','.join(str(r['strategies'][st]['e_rank']) for r in S)
    ca_wrong = [r['job'].replace('RD02_', '') for r in S if r['strategies']['R_CA']['e_rank'] != 1]
    improved = [r['job'].replace('RD02_', '') for r in S if r['strategies']['R_CA']['e_rank'] != 1 and r['strategies']['R_HistLate']['e_rank'] < r['strategies']['R_CA']['e_rank']]
    worsened = [r['job'].replace('RD02_', '') for r in S if r['strategies']['R_HistLate']['e_rank'] > r['strategies']['R_CA']['e_rank']]
    q4 = ('同一四候选上的 E 排名（L1..L7）：R_CA %s；R_HistLate %s；R_HistNear %s；Fixed_initial %s；Fixed_NoMix %s。事后 oracle 遗憾均值（pp）：R_CA %s，R_HistLate %s，R_HistNear %s，Fixed_initial %s，Fixed_NoMix %s，UniformRandom %s。R_CA 未选到 E 最优的批次：%s；其中 R_HistLate 排名改善的：%s；R_HistLate 排名比 R_CA 更差（新增错选）的：%s。'
          % (ranks('R_CA'), ranks('R_HistLate'), ranks('R_HistNear'), ranks('Fixed_initial'), ranks('Fixed_NoMix'),
             fp(ps['R_CA']['mean_regret_pp']), fp(ps['R_HistLate']['mean_regret_pp']), fp(ps['R_HistNear']['mean_regret_pp']), fp(ps['Fixed_initial']['mean_regret_pp']), fp(ps['Fixed_NoMix']['mean_regret_pp']), fp(ps['UniformRandom_expectation']['mean_regret_pp']),
             ', '.join(ca_wrong) or '无', ', '.join(improved) or '无', ', '.join(worsened) or '无'))
    # §9 reading rule applied mechanically
    if main['status'] != 'OK':
        verdict = 'INCOMPLETE'
    elif main['changed_delivery'] == 0:
        verdict = 'NO_CHANGE: R_HistLate 与 R_CA 七批同交付'
    elif main['mean_delta_pp'] > 0 and main['losses_B'] == 0 and same_fi and same_fn:
        verdict = 'DEFAULT_RECOVERY_ONLY: 只改善 R_CA，却与固定默认同交付——历史默认减少了近端选优的损失，未证明动态研究增量'
    elif main['mean_delta_pp'] > 0 and fi['status'] == 'OK' and fn['status'] == 'OK' and fi['mean_delta_pp'] > 0 and fn['mean_delta_pp'] > 0 and main['sign_stable_under_loo'] and fi['changed_delivery'] > 0:
        verdict = 'PROMISING_DEVELOPMENT_EVIDENCE: 优于 R_CA 及两个固定对照，且留一 Job 符号稳定（仍不宣称 Skill 有效）'
    elif main['mean_delta_pp'] > 0 and ((fi['status'] == 'OK' and fi['mean_delta_pp'] < 0) or (fn['status'] == 'OK' and fn['mean_delta_pp'] < 0)):
        verdict = 'BETTER_THAN_CA_BUT_NOT_FIXED: 优于 R_CA 但逊于固定对照，保留固定默认为参照，不包装成成功选择器'
    else:
        verdict = 'NO_IMPROVEMENT_OR_MIXED: 不改善或方向混合，本包停止'
    hist_rows = '; '.join('%s rows [%d, %d) origins %s' % (h.replace('RD02_', ''), v['new_exposure_rows'][0], v['new_exposure_rows'][1], v['origins']) for h, v in (hl.get('jobs') or {}).items())
    L = ['# %s — REPORT' % PACKAGE, '', '任务书：`%s`。身份：`%s`（L1–L7 及初始四批结果均已被研究者看过；不是前瞻验证）。完成：%s。' % (TASK, EXPOSURE, res['finished_local']), '',
         '## 0. §8 四问', '', '1. **主效果** — ' + q1, '', '2. **是否超出默认** — ' + q2, '', '3. **时段是否有用** — ' + q3, '', '4. **好材料是否更多交付** — ' + q4, '',
         '**机械判读（§9 规则）**：`%s`。' % verdict, '',
         '## 1. 费用、时间、暴露、缺项', '',
         '- 新模型拟合 0；优化器更新 0；实验 LLM/HTTP 0；新增 SHA 0；git commit 0。模型推理 %s 次（48 份历史模型 E 推理 + 1 份接线复算）。' % b.get('model_inferences'),
         '- 数值子进程墙钟合计 %.1f s（上限 %d s），单 worker；子进程 %d 个，是否因上限停止：%s。' % (b.get('numeric_wall_used_s', 0.0), NUMERIC_WALL_S, len(b.get('subprocesses', [])), b.get('stopped_for_wall')),
         '- 新开历史标签：%s（此前从未作为 E 评分目标打开；作为本次 `historical_late` 开发反馈记录，不宣称全包 0 新标签）。' % (hist_rows or '无'),
         '- 接线：%s，重算 C_A 与缓存差 %s（宏）/ %s（逐实体最大），冻结容差 %g；scaler/合法父窗口与缓存逐位相同：%s。' % (
             (res['smoke'] or {}).get('wiring', {}).get('cell'), ('%.3g' % (res['smoke'] or {}).get('wiring', {}).get('abs_diff_macro', float('nan'))), ('%.3g' % (res['smoke'] or {}).get('wiring', {}).get('max_abs_diff_per_entity', float('nan'))), WIRING_TOL_ABS,
             (res['smoke'] or {}).get('wiring', {}).get('scaler_and_legal_parents_identical_to_cache')),
         '- 缺项：%s。' % ('; '.join('%s %s INCOMPLETE (%s)' % (s['job'], st, s[st].get('reason')) for s in res['selections'] for st in STRATEGIES if s[st]['status'] != 'OK') or '无（六策略在七批上全部有确定选择）'),
         '- 历史访问日志 %d 条，全部满足 t_h+384 <= t_j（见 result.json.history_access_log）。' % len(res['history_access_log']), '',
         '## 2. 交付序列（L1..L7）', '', '| 策略 | ' + ' | '.join(r['job'].replace('RD02_', '') for r in S) + ' | 均 regret pp | rank1 次数 |', '|---|' + '---|' * len(S) + '---:|---:|']
    for st in STRATEGIES:
        L.append('| %s | %s | %s | %s |' % (st, ' | '.join(str(r['strategies'][st]['material'] if st != 'UniformRandom_expectation' else 'E[4]') for r in S), fp(ps[st]['mean_regret_pp']), ps[st]['n_rank1']))
    L += ['| oracle (诊断) | ' + ' | '.join(r['oracle'] for r in S) + ' | 0.00 | 7 |', '',
          '## 3. 汇总比较（A -> B，Δ > 0 = B 更好，单位 pp of None；Job 为单位）', '', '| A -> B | mean Δ | wins/ties/losses (B) | changed | max harm | LOO [min, max] | sign stable |', '|---|---:|---|---:|---:|---|---|']
    for k, v in c.items():
        if v['status'] == 'OK':
            L.append('| %s | %s | %d/%d/%d | %d | %s | [%s, %s] | %s |' % (k, fp(v['mean_delta_pp']), v['wins_B'], v['ties'], v['losses_B'], v['changed_delivery'], fp(v['max_harm_to_B_pp']), fp(v['loo_min_pp']), fp(v['loo_max_pp']), v['sign_stable_under_loo']))
        else:
            L.append('| %s | INCOMPLETE | | | | | |' % k)
    L += ['', '## 4. 逐批 E（原始，3 seed 均值；括号内为对 None 的 pp）', '', '| Job | E(None) | ' + ' | '.join(PUBLIC[1:]) + ' | oracle | R_CA | R_HistLate | Δ(R_CA->R_HistLate) [t95] |', '|---|---:|' + '---:|' * len(PUBLIC[1:]) + '---|---|---|---|']
    for r in S:
        cc = r['comparisons']['R_CA->R_HistLate']
        L.append('| %s | %s | %s | %s | %s | %s | %s [%s, %s] |' % (r['job'].replace('RD02_', ''), f(r['e_none_mean']), ' | '.join('%s (%s)' % (f(r['e_materials'][m]['mean']), fp(r['e_materials'][m]['delta_vs_none_pp'])) for m in PUBLIC[1:]), r['oracle'],
                                                                     r['strategies']['R_CA']['material'], r['strategies']['R_HistLate']['material'], fp(cc.get('delta_pp')), fp((cc.get('t95_pp') or [None, None])[0]), fp((cc.get('t95_pp') or [None, None])[1])))
    L += ['', '## 5. 历史反馈的 J 值（越小越好；ratio to None，early = C_A+C_B 合并，late = E）', '']
    for s in res['selections']:
        for st in ('R_HistNear', 'R_HistLate'):
            sel = s[st]
            if sel['status'] == 'OK':
                L.append('- %s %s: H = %s → %s; J = %s' % (s['job'].replace('RD02_', ''), st, ','.join(h.replace('RD02_', '') for h in sel['history']), sel['material'], ', '.join('%s %.4f' % (m, sel['criterion'][m]) for m in PUBLIC)))
            else:
                L.append('- %s %s: INCOMPLETE (%s)' % (s['job'].replace('RD02_', ''), st, sel.get('reason')))
    L += ['', '## 6. 历史后期反馈（新算，初始四批 E；ratio to None）', '']
    fi_sel = res['fixed_initial']
    if fi_sel['status'] == 'OK':
        for h in INITIAL:
            d = fi_sel['per_history'][h]
            L.append('- %s: None E %.4f; %s' % (h.replace('RD02_', ''), d['none_mean'], ', '.join('%s %.4f' % (m, d[m]['ratio_to_none']) for m in PUBLIC[1:])))
    L += ['', '## 7. 说明与边界', '',
          '- 所有 E（L1–L7）只读既有 `e_scores.json`；初始四批的 E 由缓存权重 + 缓存 scaler + Baseline-Linear 服务输入在本包新目录推理并用同一 scorer 评分，旧 cell 未改。',
          '- R_HistNear/R_HistLate 与 R_CA 同时改变历史覆盖与反馈数量；两者之差才隔离"模型使用时段"。三 seed 区间只描述训练随机性。',
          '- 7 批不是 21 个独立样本；留一 Job 均值用于检查是否由单批主导。oracle 只作诊断，不回流选择。',
          '- 本包不授权 Source Episode 或跨 Job 分数银行直接进入 Fast Prompt；未启动后续 Fast/Slow。',
          '', '详细表：`tables.md`；机器可读：`result.json`、`selections.json`、`frozen_config.json`、`historical_late/*/historical_late_scores.json`。']
    (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


# ============================================================================= entry
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--historical', action='store_true')
    ap.add_argument('--replay', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--worker-wiring', metavar='OUT')
    ap.add_argument('--worker-hist-freeze', metavar='JOB')
    ap.add_argument('--worker-hist-score', metavar='JOB')
    a = ap.parse_args()
    if a.worker_wiring:
        worker_wiring(a.worker_wiring)
        return
    if a.worker_hist_freeze:
        worker_hist_freeze(a.worker_hist_freeze)
        return
    if a.worker_hist_score:
        worker_hist_score(a.worker_hist_score)
        return
    assert 'torch' not in sys.modules, 'controller must not import torch'
    if a.preflight or a.run:
        preflight()
    ledger = Ledger()
    ledger.save()
    if a.smoke or a.run:
        smoke(ledger)
    if a.historical or a.run:
        historical(ledger)
    if a.replay or a.run:
        replay()
    assert 'torch' not in sys.modules, 'controller imported torch'


if __name__ == '__main__':
    main()
