"""DEV-AUG-MAIN-COMPARISON (docs/DEV_AUG_MAIN_COMPARISON_TASK_2026-09-23.md): the AutoDA-Timeseries comparison line on the 40 Test cases of
DEV-AUG-OFFLINE-SKILL, plus the main-table / cost-table assembly (0 LLM).

AutoDA-Timeseries (ICLR 2026, https://github.com/NetManAIOps/AutoDA-Timeseries, commit 91dbf70) is called through its OFFICIAL policy / loss / transform
modules (autoaugment/AutoDA_Timeseries.py: AugmentModel with 3 stacked AugmentLayers = feature-conditioned probability module + strength module +
Gumbel choice over the official transform list; CompositeLoss with entropy / diversity terms and optional learnable weights) and the official Catch22
feature extractor (dataloader/tsfeature_extractors.py). Adaptation (BASELINE_ADAPTATION.md): the downstream model is the parent study's shared MLP
192->128->64->48 on univariate windows; the common skeleton AdamW lr 1e-3 / wd 1e-4 / batch 64 / 2000 updates over the parent's batch-index stream
replaces the official RAdam + validation early stopping; the policy parameters (and learnable loss weights / tau) are optimized jointly in the same
optimizer; features are Catch22 of each legal training input window (the released forecasting loader passes time-mark features whose shape does
not match the policy MLP; the paper's feature-conditioned design and the official extractor are used instead); forecasting semantics follow the
official loop: only the input window is augmented, the target is never augmented, serving inputs are never augmented.

Parent caches are read-only; this package writes its own case dirs, features, weights and scores under _scratch/dev_aug_main_comparison/.

  --wiring | --source | --test | --score-test | --tables | --status
  subprocess entries: --worker-fit CASE SEED CONFIG MODE | --worker-score CASE OUT CELL...
"""
from __future__ import annotations

import os

KMP_AT_START = os.environ.get('KMP_DUPLICATE_LIB_OK')

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

from methods.ttha.batch_base import context, data, spec
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_domain_aug_entity_split as ES
from evaluation.main_protocol_p4 import batch_research_domain_aug_decision_priority as DP
from evaluation.main_protocol_p4 import batch_research_aug_task_family_screen as SC

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_aug_main_comparison'
MODULE = 'evaluation.main_protocol_p4.batch_research_aug_main_comparison'
PARENT = REPO / '_scratch' / 'dev_aug_offline_skill'
REF = ROOT / 'ref' / 'AutoDA-Timeseries'
PYLIB = ROOT / 'pylib'
PACKAGE = 'DEV-AUG-MAIN-COMPARISON'
IDENTITY = 'SUPPLEMENTARY_COMPARISON_ON_EXPOSED_TEST_CASES (not a new unexposed validation)'
SEEDS = (20269181, 20269182, 20269183)
DOMAINS = ('D01', 'D02', 'D03')
OFFICIAL = {'repo': 'https://github.com/NetManAIOps/AutoDA-Timeseries', 'commit': '91dbf70b54b255214b7f204d8d9f70d26f9c1fe3', 'commit_date': '2026-06-15',
            'project_page': 'https://netmanaiops.github.io/AutoDA-Timeseries/', 'paper': 'https://openreview.net/forum?id=vTLmHAkoIW (ICLR 2026)',
            'modules_used': ['autoaugment/AutoDA_Timeseries.py: AugmentModel, AugmentLayer, AugProbMLPModule, AugStrengthMLPModule, CompositeLoss (unchanged)',
                             'autoaugment/augments/basic_transforms.py: AVAILABLE_TRANSFORMS = Raw, Scale, Jitter, Downsampling, Resampling, FreqWarp (unchanged)',
                             'dataloader/tsfeature_extractors.py: Catch22FeatureExtractor (unchanged; pycatch22 0.5.0 installed under the package dir)'],
            'not_used': 'downstream/ (sktime-based models; replaced by the parent MLP through a stub module), exp/ top-level train (reads Test every epoch)'}
# pre-registered complete configurations (task §4: at most 3, each with a source or an explicit adaptation basis; frozen before any fit)
CONFIGS = {
    'C1_official_example': {'config': {'l_tau': 1, 'tau': 2.87, 'rob': 0.9, 'minp': 0.14, 'ew': 0.44, 'dw': 0.18, 'dyl': 1},
                            'source': 'examples/AutoDA-Timeseries.classification.json (the only released complete config; a CLASSIFICATION example, used as a sourced '
                                      'candidate, not called the forecasting config; its strength_scale key is read by no released module)'},
    'C2_code_defaults': {'config': {'l_tau': 0, 'tau': 10.0, 'rob': 1.0, 'minp': 0.0, 'ew': 0.0, 'dw': 0.0, 'dyl': 0},
                         'source': 'the defaults of _parse_config in autoaugment/AutoDA_Timeseries.py when no tsa config is given (plain task loss, fixed tau 10)'},
    'C3_example_fixed_weights': {'config': {'l_tau': 1, 'tau': 2.87, 'rob': 0.9, 'minp': 0.14, 'ew': 0.44, 'dw': 0.18, 'dyl': 0},
                                 'source': 'C1 with the composite-loss weights fixed (dyl 0, the CompositeLoss "Fixed weights" branch): adaptation basis = with 2000 '
                                           'updates and no early stopping, learnable weights w enter as loss / w^2 + log(w^2 + 1) and can shrink the task term; the '
                                           'fixed-weight branch keeps the configured ratios'},
}
N_LAYERS = 3                       # AutoDA_Timeseries.Model._initialize: n_layers = 3 ("follow previous works")
FEATURE_DIM = 24                   # Catch22FeatureExtractor default cond_dim (catch24)
POLICY_SEED_OFFSET = 7919
BUDGET = {'fits': 180, 'wiring': 6, 'source': 27, 'test': 120, 'wall_warn_s': 8 * 3600, 'wall_cap_s': 12 * 3600, 'llm': 0}
NUMERIC = 2
FIT_TIMEOUT_S = 3600.


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def split() -> dict:
    return context.read_json(PARENT / 'split_copy.json')


def cases() -> dict:
    return ec.cases_from_split(split())


def frozen_parent() -> dict:
    return context.read_json(PARENT / 'frozen_config.json')


def source_selection_cases() -> dict:
    """task §4.3: per domain, the 8 Source cases sorted by (t, case_id); take the 1st, 4th and 8th."""
    css = cases()
    out = {}
    for d in DOMAINS:
        lst = sorted(frozen_parent()['cases']['source'][d], key=lambda c: (css[c].t, c))
        out[d] = [lst[0], lst[3], lst[7]]
    return out


def test_cases() -> list:
    fp = frozen_parent()
    return [c for d in DOMAINS for c in sorted(fp['cases']['test'][d])]


# ============================================================================= official modules (worker side; torch imported here only)
def _official():
    """Import the official AutoDA modules unchanged: the released `downstream` package imports sktime models that this adaptation never
    uses, so a stub `downstream` module that only provides the two names the policy module imports is registered first."""
    if str(PYLIB) not in sys.path:
        sys.path.insert(0, str(PYLIB))
    if str(REF) not in sys.path:
        sys.path.insert(0, str(REF))
    if 'downstream' not in sys.modules or not getattr(sys.modules['downstream'], '_stub', False):
        import torch.nn as nn
        stub = types.ModuleType('downstream')
        stub._stub = True

        class DownstreamModelBase(nn.Module):
            pass

        def build_downstream_model(*a, **k):
            raise RuntimeError('the official downstream models are not used in this adaptation (parent MLP instead)')
        stub.DownstreamModelBase = DownstreamModelBase
        stub.build_downstream_model = build_downstream_model
        sys.modules['downstream'] = stub
    from utils.GlobalConfig import GlobalConfig
    from autoaugment import AutoDA_Timeseries as A
    # the released dataloader package __init__ imports the classification / anomaly loaders (aeon); the feature-extractor file is loaded by path, unchanged
    if 'autoda_tsfeature_extractors' not in sys.modules:
        import importlib.util
        spec_ = importlib.util.spec_from_file_location('autoda_tsfeature_extractors', str(REF / 'dataloader' / 'tsfeature_extractors.py'))
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        sys.modules['autoda_tsfeature_extractors'] = mod
    FX = sys.modules['autoda_tsfeature_extractors']
    return GlobalConfig, A, FX


def install_config(cfg: dict, n_workers: int = 1):
    """A GlobalConfig singleton carrying the tsa args (bypasses its CLI __init__: no stdout redirection, no GPU probing)."""
    GlobalConfig, A, FX = _official()
    g = GlobalConfig.__new__(GlobalConfig)
    g.args = argparse.Namespace(task='long_term_forecasting', downstream='parent_MLP', num_workers=n_workers, n_channels=1, seq_len=spec.L,
                                n_features=FEATURE_DIM, pred_len=spec.H)
    g.device = 'cpu'
    g.tsa_args = dict(cfg)
    g.downstream_args = {}
    GlobalConfig._instance = g
    return g, A, FX


def open_case_T(cs: ec.CaseSpec):
    """This package's own case view (T rows only): scaler recomputed (must equal the parent's), parents built from T."""
    ctx = ec.open_case(cs, ROOT / 'cases', 'material')
    mine = np.load(ROOT / 'cases' / cs.case_id / 'scaler.npz')
    par = np.load(PARENT / 'common' / cs.case_id / 'scaler.npz')
    if not (np.array_equal(mine['mean'], par['mean']) and np.array_equal(mine['scale'], par['scale'])):
        raise RuntimeError('scaler differs from the parent for %s' % cs.case_id)
    return ctx


def features_for(cs: ec.CaseSpec, ctx, FX, n_workers: int = 1) -> np.ndarray:
    """Catch22 (catch24) features of every legal training input window, official extractor, cached in this package."""
    fp = ROOT / 'cases' / cs.case_id / 'catch22_inputs.npy'
    if fp.exists():
        return np.load(fp)
    X = ctx.parents.X_norm.reshape(-1, spec.L)
    g = install_config(copy.deepcopy(__import__('utils.GlobalConfig', fromlist=['GlobalConfig']).GlobalConfig.get_config().tsa_args), n_workers)[0]
    ext = FX.get_feature_extractor('Catch22')(g, None)
    F = ext(X[:, None, :].astype(np.float64))
    tmp = fp.with_suffix('.tmp%d.npy' % os.getpid())
    np.save(tmp, F.astype(np.float32))
    os.replace(tmp, fp)
    return F.astype(np.float32)


def worker_fit(case: str, seed: int, config_name: str, mode: str, out: str) -> None:
    """mode 'autoda': joint AutoDA training (official policy + loss) with the parent MLP; 'none': the plain parent None fit (wiring W1).
    Reads T rows only; writes the frozen MLP / policy weights and a training record; never scores any block."""
    ES._init_worker()
    import torch
    import torch.nn as nn
    from methods.ttha.batch_base import train
    torch.set_num_threads(ES.FIT_THREADS)                                      # 8, as every parent fit (bitwise-stable Consumer)
    cs = cases()[case]
    cfg = CONFIGS[config_name]['config'] if mode == 'autoda' else {}
    g, A, FX = install_config(cfg, 1)
    ctx = open_case_T(cs)
    P = ctx.parents
    Xp = torch.as_tensor(P.X_norm.reshape(-1, spec.L), dtype=torch.float32)
    Yp = torch.as_tensor(P.y_norm.reshape(-1, spec.H), dtype=torch.float32)
    n_pool = int(Xp.shape[0])
    bidx = train.batch_indices(seed, n_pool=n_pool, n_updates=spec.N_UPDATES)
    torch.manual_seed(seed)                                                    # parent discipline: seed -> MLP construction
    mlp = train.make_model()()
    init_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
    t0 = time.time()
    rec = {'case': case, 'seed': seed, 'config': config_name, 'mode': mode, 'started_local': now(), 'n_pool': n_pool, 'rows_read': list(cs.material_rows)}
    if mode == 'autoda':
        F = torch.as_tensor(features_for(cs, ctx, FX), dtype=torch.float32)          # (n_pool, 1, 24)
        rec['feature_seconds'] = time.time() - t0
        with torch.random.fork_rng():
            torch.manual_seed(seed + POLICY_SEED_OFFSET)                         # the policy's own initialization stream
            aug = A.AugmentModel(g, spec.L, 1, FEATURE_DIM, N_LAYERS, transforms=A.AVAILABLE_TRANSFORMS)
            ew, dw, dyl = float(cfg.get('ew', 0)), float(cfg.get('dw', 0)), bool(cfg.get('dyl', 0))
            if ew > 0 or dw > 0:                                                 # official get_criterion logic
                crit = A.CompositeLoss(nn.MSELoss(), N_LAYERS, learnable_weight=dyl, task_weight=1.0, entropy_weight=ew, diversity_weight=dw)
                aug.set_criterion(crit)
            else:
                crit = nn.MSELoss()
        policy_init = {k: v.detach().clone() for k, v in aug.state_dict().items()}
        params, seen = [], set()
        for p in list(mlp.parameters()) + list(aug.parameters()) + (list(crit.parameters()) if isinstance(crit, nn.Module) else []):
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        opt = torch.optim.AdamW(params, lr=spec.LR, weight_decay=spec.WEIGHT_DECAY)
        torch.manual_seed(seed + 2 * POLICY_SEED_OFFSET)                        # policy sampling / transform noise stream (after the MLP init)
        mlp.train()
        aug.train()
        mask = torch.ones((spec.PARENT_BATCH, spec.L))
        curve, choice_counts = [], np.zeros((N_LAYERS, len(A.AVAILABLE_TRANSFORMS)))
        for step in range(spec.N_UPDATES):
            idx = torch.as_tensor(bidx[step], dtype=torch.long)
            x = Xp[idx].unsqueeze(1)
            y = Yp[idx].unsqueeze(1)
            f = F[idx]
            opt.zero_grad(set_to_none=True)
            prev = None
            ax, ay, am = x, y, mask
            for li, layer in enumerate(aug.layers):                            # == AugmentModel.forward, with the per-layer choice recorded
                ax, ay, am, prev, p = layer(ax, ay, f, am, prev)
                if isinstance(crit, A.CompositeLoss):
                    crit.record_prob(p, li)
                choice_counts[li] += prev.detach().sum(0).numpy()
            pred = mlp(ax.squeeze(1))
            loss = crit(pred, Yp[idx])                                           # target never augmented (official forecasting loop)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss at step %d' % step)
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == spec.N_UPDATES - 1:
                curve.append((step, float(loss.item())))
        pol_delta = float(sum(((v - policy_init[k]) ** 2).sum().item() for k, v in aug.state_dict().items() if v.dtype.is_floating_point) ** 0.5)
        rec.update({'loss_curve': curve, 'policy_param_delta_norm': pol_delta, 'choice_share_by_layer': (choice_counts / choice_counts.sum(1, keepdims=True)).round(4).tolist(),
                    'transforms': [t.__name__ for t in A.AVAILABLE_TRANSFORMS], 'tau': [float(l.tau) for l in aug.layers],
                    'loss_weights': ([float(crit.task_weight), float(crit.entropy_weight), float(crit.diversity_weight)] if isinstance(crit, A.CompositeLoss) else None)})
        torch.save(aug.state_dict(), Path(out) / ('%s__s%d__%s__policy.pt' % (case, seed, config_name)))
    else:
        opt = torch.optim.AdamW(mlp.parameters(), lr=spec.LR, weight_decay=spec.WEIGHT_DECAY)
        for step in range(spec.N_UPDATES):
            idx = torch.as_tensor(bidx[step], dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            loss = ((mlp(Xp[idx]) - Yp[idx]) ** 2).mean(dim=-1).mean()          # the parent's pooled loss (None path)
            loss.backward()
            opt.step()
    mlp_delta = float(sum(((v - init_state[k]) ** 2).sum().item() for k, v in mlp.state_dict().items()) ** 0.5)
    mp = Path(out) / ('%s__s%d__%s__%s__mlp.pt' % (case, seed, config_name if mode == 'autoda' else 'none', mode))
    torch.save(mlp.state_dict(), mp)
    rec.update({'mlp_param_delta_norm': mlp_delta, 'model_path': str(mp), 'train_seconds': time.time() - t0, 'finished_local': now(), 'status': 'OK',
                'consumer': 'parent MLP 192->128->64->48 (train.make_model), torch.manual_seed(seed) before construction; AdamW lr %g wd %g; %d updates; batch %d over '
                            'train.batch_indices(seed)' % (spec.LR, spec.WEIGHT_DECAY, spec.N_UPDATES, spec.PARENT_BATCH)})
    context.write_json(Path(out) / ('%s__s%d__%s__%s.json' % (case, seed, config_name if mode == 'autoda' else 'none', mode)), rec)
    print('FIT_OK', case, seed, config_name, mode, round(rec['train_seconds'], 1), flush=True)


def worker_score(case: str, out: str, cells: list) -> None:
    """Evaluator only: E of frozen MLPs on the case's four E origins (serving inputs never augmented), parent scorer."""
    ES._init_worker()
    import torch
    from methods.ttha.batch_base import train
    cs = cases()[case]
    sc = np.load(ROOT / 'cases' / case / 'scaler.npz')
    scaler = data.Scaler(mean=sc['mean'], std=sc['std'], scale=sc['scale'], floor_hits=0)
    sl = ec.load_case_slice(cs, cs.t, max(cs.e) + ec.H)
    wins = data.build_windows(sl, list(cs.e), scaler, with_truth=True)
    truth = np.stack([w.y_true_raw for w in wins], axis=1)
    res = {'case': case, 'rows_read': [cs.t, max(cs.e) + ec.H], 'scored_local': now(), 'cells': {}}
    for mp in cells:
        model = train.make_model()()
        model.load_state_dict(torch.load(mp, weights_only=True))
        pred = train.predict_windows(model, wins, scaler)
        s = ec.score_with_status(pred, truth, scaler, sc['mase'])
        if s['status'] != 'SCORABLE':
            raise RuntimeError('not scorable %s' % mp)
        res['cells'][Path(mp).name] = {'e': s['normalized_mse_macro'], 'mae_raw_macro': s['mae_raw_macro'], 'e_per_entity': s['per_entity_normalized_mse']}
    context.write_json(Path(out), res)
    print('SCORE_OK', case, len(cells), flush=True)


# ============================================================================= coordinator (never imports torch)
class Ledger:
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.s = context.read_json(path) if path.exists() else {'created_local': now(), 'started_epoch': time.time(), 'fit_attempts': 0, 'fits_ok': 0, 'fits_failed': 0,
                                                                 'fit_seconds': 0.0, 'score_runs': 0, 'events': [], 'caps': BUDGET}
        self._save()

    def _save(self):
        with self.lock:
            context.write_json(self.path, self.s)

    def reserve(self, tag):
        with self.lock:
            if self.s['fit_attempts'] >= BUDGET['fits']:
                raise RuntimeError('fit cap %d reached' % BUDGET['fits'])
            if time.time() - self.s['started_epoch'] > BUDGET['wall_cap_s']:
                raise RuntimeError('wall cap reached')
            self.s['fit_attempts'] += 1
            self.s['events'].append({'kind': 'fit_started', 'tag': tag, 'epoch': time.time()})
        self._save()

    def finish(self, tag, ok, secs):
        with self.lock:
            self.s['fits_ok' if ok else 'fits_failed'] += 1
            self.s['fit_seconds'] += secs
            self.s['events'].append({'kind': 'fit_finished', 'tag': tag, 'ok': ok, 'seconds': round(secs, 1), 'epoch': time.time()})
        self._save()


_SEM = threading.BoundedSemaphore(NUMERIC)


def run_sub(args: list, log: Path, timeout: float) -> tuple:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = ES.worker_env()
    with _SEM:
        t_w = time.time()
        while SC.free_gb() < 1.0:
            if time.time() - t_w > 1800:
                raise RuntimeError('host memory below 1 GB for 30 min')
            time.sleep(5)
        t0 = time.time()
        try:
            with log.open('w', encoding='utf-8') as fh:
                p = subprocess.run([sys.executable, '-B', '-m', MODULE] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, env=env)
            return p.returncode, time.time() - t0
        except subprocess.TimeoutExpired:
            return -999, time.time() - t0


def fit_task(led: Ledger, out: Path, case: str, seed: int, config_name: str, mode: str = 'autoda') -> dict:
    tag = '%s__s%d__%s__%s' % (case, seed, config_name if mode == 'autoda' else 'none', mode)
    rp = out / (tag + '.json')
    if rp.exists() and context.read_json(rp).get('status') == 'OK':
        return context.read_json(rp)
    for attempt in (0, 1):
        led.reserve(tag)
        rc, secs = run_sub(['--worker-fit', case, seed, config_name, mode, out], out / 'logs' / ('%s%s.log' % (tag, '' if attempt == 0 else '.retry')), FIT_TIMEOUT_S)
        ok = rc == 0 and rp.exists()
        led.finish(tag, ok, secs)
        print('FIT', tag, 'OK' if ok else 'FAILED rc=%s' % rc, round(secs, 1), flush=True)
        if ok:
            return context.read_json(rp)
    raise RuntimeError('fit failed twice: %s' % tag)


def parallel(jobs: list, fn) -> list:
    out, errs, lock = [], [], threading.Lock()
    q = list(jobs)

    def w():
        while True:
            with lock:
                if not q:
                    return
                j = q.pop(0)
            try:
                r = fn(*j)
                with lock:
                    out.append((j, r))
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errs.append((j, repr(exc)[:300]))
    ts = [threading.Thread(target=w, daemon=True) for _ in range(NUMERIC)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise RuntimeError('tasks failed: %s' % errs[:3])
    return out


def score(case: str, cells: list, out: Path) -> dict:
    if out.exists():
        r = context.read_json(out)
        if all(Path(c).name in r['cells'] for c in cells):
            return r
    rc, _ = run_sub(['--worker-score', case, out] + list(cells), out.parent / 'logs' / ('score_%s.log' % out.stem), 1800)
    if rc != 0:
        raise RuntimeError('score failed %s' % case)
    return context.read_json(out)


def parent_e(case: str, phys: str, seed: int) -> float:
    """E of a parent physical material (screen cache for Source cases, the parent Test evaluation for Test cases)."""
    if '_SCR_' in case:
        return SC_screen_e(case)[phys][seed]
    ev = context.read_json(PARENT / 'test' / 'evaluation.json')
    return ev['e_by_case_phys_seed'][case][phys][str(seed)]


def SC_screen_e(case: str) -> dict:
    es = context.read_json(SC.case_dir() / case / 'e_scores.json')
    out = {}
    for r in es['cells'].values():
        out.setdefault(r['material_id'], {})[int(r['model_seed'])] = float(r['e'])
    return out


# ----------------------------------------------------------------------------- wiring (<= 6 fits)
def wiring() -> dict:
    wd = ROOT / 'wiring'
    wd.mkdir(parents=True, exist_ok=True)
    led = Ledger(ROOT / 'budget.json')
    checks = []

    def check(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})
        print('WIRING', 'OK ' if ok else 'FAIL', name, info, flush=True)

    # W1: the None path reproduces the parent None cell (data, scaler, init, batch stream, scorer)
    case, seed = 'D01_SCR_A1_G1', SEEDS[0]
    r1 = fit_task(led, wd, case, seed, 'C2_code_defaults', 'none')
    sc = score(case, [r1['model_path']], wd / 'score_w1.json')
    e_mine = sc['cells'][Path(r1['model_path']).name]['e']
    check('W1_none_path_reproduces_parent_E', e_mine == parent_e(case, 'None', seed), mine=e_mine, parent=parent_e(case, 'None', seed))
    # W2: official augment / loss modules vs this glue on the same input and RNG state (0 fits)
    rc, _ = run_sub(['--worker-glue-check', wd / 'glue_check.json'], wd / 'logs' / 'glue_check.log', 900)
    gc = context.read_json(wd / 'glue_check.json') if rc == 0 else {}
    check('W2_glue_equals_official_forward_and_loss', gc.get('ok') is True, **{k: v for k, v in gc.items() if k != 'ok'})
    # W3: one real joint training on a Source case: policy and MLP parameters move, finite loss, E scores; T-only rows read
    r3 = fit_task(led, wd, case, seed, 'C1_official_example', 'autoda')
    sc3 = score(case, [r3['model_path']], wd / 'score_w3.json')
    e3 = sc3['cells'][Path(r3['model_path']).name]['e']
    check('W3_joint_training_updates_and_scores', r3['policy_param_delta_norm'] > 0 and r3['mlp_param_delta_norm'] > 0 and math.isfinite(e3)
          and r3['rows_read'] == list(cases()[case].material_rows), policy_delta=r3['policy_param_delta_norm'], e=e3, seconds=r3['train_seconds'],
          choice_share=r3['choice_share_by_layer'], loss_weights=r3['loss_weights'], tau=r3['tau'])
    out = {'written_local': now(), 'checks': checks, 'passed': sum(c['ok'] for c in checks), 'total': len(checks), 'fits': led.s['fit_attempts']}
    context.write_json(wd / 'wiring_result.json', out)
    print('WIRING_DONE', out['passed'], '/', out['total'], 'fits', out['fits'], flush=True)
    return out


def worker_glue_check(out: str) -> None:
    """Same input + same RNG state: the official AugmentModel.forward and CompositeLoss vs the per-layer glue used in worker_fit."""
    ES._init_worker()
    import torch
    import torch.nn as nn
    cfg = CONFIGS['C1_official_example']['config']
    g, A, FX = install_config(cfg, 1)
    torch.manual_seed(1)
    aug = A.AugmentModel(g, spec.L, 1, FEATURE_DIM, N_LAYERS, transforms=A.AVAILABLE_TRANSFORMS)
    crit_a = A.CompositeLoss(nn.MSELoss(), N_LAYERS, learnable_weight=True, task_weight=1.0, entropy_weight=cfg['ew'], diversity_weight=cfg['dw'])
    crit_b = copy.deepcopy(crit_a)
    aug.set_criterion(crit_a)
    x = torch.randn(64, 1, spec.L)
    y = torch.randn(64, 1, spec.H)
    f = torch.randn(64, 1, FEATURE_DIM)
    m = torch.ones(64, spec.L)
    pred = torch.randn(64, spec.H)
    ok = True
    diffs = []
    for rep in range(2):                                     # two consecutive steps: the diversity term uses the previous step's probabilities
        torch.manual_seed(5 + rep)
        ax1, ay1, am1 = aug(x, y, f, m)                      # official AugmentModel.forward (records into crit_a)
        l1 = crit_a(pred, y.squeeze(1))
        torch.manual_seed(5 + rep)
        prev, ax2, ay2, am2 = None, x, y, m
        for li, layer in enumerate(aug.layers):              # the glue of worker_fit (records into crit_b)
            ax2, ay2, am2, prev, p = layer(ax2, ay2, f, am2, prev)
            crit_b.record_prob(p, li)
        l2 = crit_b(pred, y.squeeze(1))
        ok = ok and torch.equal(ax1, ax2) and torch.equal(l1, l2)
        diffs.append({'max_abs_x_diff': float((ax1 - ax2).abs().max()), 'loss_official': float(l1), 'loss_glue': float(l2)})
    context.write_json(Path(out), {'ok': bool(ok), 'steps': diffs})
    print('GLUE', ok, flush=True)


# ----------------------------------------------------------------------------- Source config selection (<= 27 fits) and Test (120 fits)
def source_selection() -> dict:
    sd = ROOT / 'source'
    sd.mkdir(parents=True, exist_ok=True)
    fz = sd / 'selection.json'
    if fz.exists():
        return context.read_json(fz)
    sel = source_selection_cases()
    context.write_json(sd / 'plan.json', {'written_local': now(), 'rule': 'task §4.3: 3 Source cases per domain (1st, 4th, 8th by (t, case_id)) x %d configs x seed %d; '
                                                                         'select by the domain-equal mean of E(config) / E(None) on these cases' % (len(CONFIGS), SEEDS[0]),
                                          'cases': sel, 'configs': CONFIGS, 'seed': SEEDS[0]})
    led = Ledger(ROOT / 'budget.json')
    jobs = [(led, sd, c, SEEDS[0], k, 'autoda') for d in DOMAINS for c in sel[d] for k in CONFIGS]
    res = dict(parallel(jobs, fit_task))
    ratios = {k: {} for k in CONFIGS}
    for d in DOMAINS:
        for c in sel[d]:
            cells = [context.read_json(sd / ('%s__s%d__%s__autoda.json' % (c, SEEDS[0], k)))['model_path'] for k in CONFIGS]
            s = score(c, cells, sd / 'scores' / ('%s.json' % c))
            den = parent_e(c, 'None', SEEDS[0])
            for k in CONFIGS:
                e = s['cells'][Path(context.read_json(sd / ('%s__s%d__%s__autoda.json' % (c, SEEDS[0], k)))['model_path']).name]['e']
                ratios[k].setdefault(d, []).append(e / den)
    J = {k: statistics.fmean(statistics.fmean(v) for v in ratios[k].values()) for k in CONFIGS}
    best = min(CONFIGS, key=lambda k: (J[k], list(CONFIGS).index(k)))
    out = {'frozen_local': now(), 'J_domain_equal': J, 'ratios': ratios, 'selected': best, 'selected_config': CONFIGS[best]}
    context.write_json(fz, out)
    print('SOURCE_SELECTED', best, json.dumps(J), flush=True)
    return out


def test_stage() -> dict:
    td = ROOT / 'test'
    td.mkdir(parents=True, exist_ok=True)
    sel = context.read_json(ROOT / 'source' / 'selection.json')['selected']
    led = Ledger(ROOT / 'budget.json')
    jobs = [(led, td, c, s, sel, 'autoda') for c in test_cases() for s in SEEDS]
    parallel(jobs, fit_task)
    fz = td / 'frozen.json'
    if not fz.exists():
        context.write_json(fz, {'frozen_local': now(), 'config': sel, 'models': sorted(p.name for p in td.glob('*__mlp.pt'))})
    return score_test()


def score_test() -> dict:
    td = ROOT / 'test'
    fz = context.read_json(td / 'frozen.json')
    sel = fz['config']
    for c in test_cases():
        cells = [context.read_json(td / ('%s__s%d__%s__autoda.json' % (c, s, sel)))['model_path'] for s in SEEDS]
        score(c, cells, td / 'scores' / ('%s.json' % c))
    e = {}
    for c in test_cases():
        s = context.read_json(td / 'scores' / ('%s.json' % c))
        e[c] = {str(sd): s['cells']['%s__s%d__%s__autoda__mlp.pt' % (c, sd, sel)]['e'] for sd in SEEDS}
    out = {'scored_local': now(), 'config': sel, 'e_by_case_seed': e}
    context.write_json(td / 'autoda_test_e.json', out)
    print('TEST_SCORED', len(e), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--source', action='store_true')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--score-test', action='store_true')
    ap.add_argument('--worker-fit', nargs=5)
    ap.add_argument('--worker-score', nargs='+')
    ap.add_argument('--worker-glue-check', nargs=1)
    a = ap.parse_args()
    if a.worker_fit:
        c, s, k, m, o = a.worker_fit
        return worker_fit(c, int(s), k, m, o)
    if a.worker_score:
        c, o, *cells = a.worker_score
        return worker_score(c, o, cells)
    if a.worker_glue_check:
        return worker_glue_check(a.worker_glue_check[0])
    if a.wiring:
        wiring()
    if a.source:
        source_selection()
    if a.test:
        test_stage()
    if a.score_test:
        score_test()


if __name__ == '__main__':
    main()
