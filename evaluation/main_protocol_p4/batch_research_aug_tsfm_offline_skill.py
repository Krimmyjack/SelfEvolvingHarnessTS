"""DEV-AUG-TSFM-OFFLINE-SKILL: the PatchTST-conditioned offline-skill design (batch_research_aug_patchtst_offline_skill) with a pretrained
time-series foundation model as the Consumer: Time-MoE-50M, fine-tuned from the same pretrained weights for every (case, plan, seed).

Stages (one controller, idempotent resume): wiring -> Source None calibration of the fine-tune (lr x batch, checkpoints) -> Source evidence
(12 scripted programs + 24 no-card zero-feedback Fast trajectories that see the Time-MoE description) -> Slow (2 domain cards / domain,
Time-MoE evidence only) -> Select -> Test (f_tsfm learned card, f_mlp frozen MLP card, f0 no card, f_naive naive card regenerated for
Time-MoE; references None / NoMix / the Source-selected fixed program / zero-shot Time-MoE anchor) -> readout.

Differences from the PatchTST package: no model file is kept (a fine-tuned Time-MoE is ~450 MB); the fit worker freezes the E-block
predictions right after training from the E INPUT windows only (serving inputs, never targets); the controller scores them after the stage
barrier. Workers need transformers==4.40.1 and the official Time-MoE source (SEH_TSFM_DIR). Parallelism is set by environment:
SEH_GPUS (e.g. 0 or 0,1), SEH_GPU_LANES_PER_GPU, SEH_HTTP, SEH_CPU_POOL; see server presets in docs / server_export.

  --smoke | --wiring | --run | --readout | --status
  subprocess entry: --fit-batch SPEC
"""
from __future__ import annotations

import os

import argparse
import heapq
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha import batch_zero_feedback as zf
from methods.ttha.batch_base import budget, context, data
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_aug_offline_skill as OS
from evaluation.main_protocol_p4 import batch_research_domain_aug_decision_priority as DP
from evaluation.main_protocol_p4 import batch_research_aug_task_family_screen as SC

REPO = Path(__file__).resolve().parents[2]
# 'native' (user 2026-09-24, option B): the official Time-MoE fine-tune - teacher-forced multi-horizon Huber loss over every position of the
# 240-point window plus the router load-balancing loss, windows normalized by their own 192-point context, AdamW beta2 0.95.
# 'rollout': the first version (48-step autoregressive MSE on T-scaled inputs); kept only to read the earlier partial run.
# 'head' (Planner suggestion 2026-09-24, after the Source zero-shot check): the native loss / normalization / optimizer with the backbone frozen -
# only the four existing forecasting heads (lm_heads) are trained; a separate root, run as the no-LLM Source screen (--source-screen).
RECIPE = os.environ.get('SEH_TSFM_RECIPE', 'native')
if RECIPE not in ('native', 'rollout', 'head'):
    raise RuntimeError('SEH_TSFM_RECIPE must be native, rollout or head')
NATIVE = RECIPE in ('native', 'head')
HEAD_ONLY = RECIPE == 'head'
_DEFAULT_ROOT = {'native': 'dev_aug_tsfm_native_offline_skill', 'rollout': 'dev_aug_tsfm_offline_skill', 'head': 'dev_aug_tsfm_head_source'}[RECIPE]
ROOT = Path(os.environ.get('SEH_TSFM_ROOT') or (REPO / '_scratch' / _DEFAULT_ROOT))
PARENT = OS.ROOT                                          # _scratch/dev_aug_offline_skill (read-only)
MODULE = 'evaluation.main_protocol_p4.batch_research_aug_tsfm_offline_skill'
TASK = 'docs/DEV_AUG_TSFM_OFFLINE_SKILL_TASK_2026-09-24.md'
PACKAGE = {'native': 'DEV-AUG-TSFM-NATIVE-OFFLINE-SKILL', 'rollout': 'DEV-AUG-TSFM-OFFLINE-SKILL', 'head': 'DEV-AUG-TSFM-HEAD-SOURCE'}[RECIPE]
IDENTITY = 'EXPOSED_DEVELOPMENT_REVALIDATION (the 40 Test cases were exposed by DEV-AUG-OFFLINE-SKILL; not an independent validation)'
DOMAINS = OS.DOMAINS
SEEDS = OS.SEEDS
MENU_IDS = list(SC.MENU_IDS)
H, L = ec.H, ec.L

TSFM_DIR = Path(os.environ.get('SEH_TSFM_DIR') or (REPO / '_scratch' / 'dev_data_readiness_ltsv_core_repair_signal' / 'tsfm_env'))
MODEL_DIR, SRC = TSFM_DIR / 'model', TSFM_DIR / 'src'
TSFM_PYTHON = os.environ.get('SEH_TSFM_PYTHON') or sys.executable      # an interpreter with transformers==4.40.1
TRANSFORMERS_REQUIRED = '4.40.1'
if NATIVE:
    OPT = {'betas': (0.9, 0.95), 'eps': 1e-8, 'weight_decay': 0.1, 'clip': 1.0}            # official Time-MoE fine-tune (adam_beta2 0.95, max_grad_norm 1)
    CALIB = {'material': 'None', 'lrs': [1e-5, 5e-5, 1e-4], 'batches': [8, 32], 'marks': [25, 50, 100, 200], 'seed': SEEDS[0]}
    if HEAD_ONLY:                                          # heads are linear read-outs of a frozen backbone: a higher learning-rate range
        CALIB = {'material': 'None', 'lrs': [1e-4, 1e-3, 1e-2], 'batches': [8, 32], 'marks': [25, 50, 100, 200], 'seed': SEEDS[0]}
else:
    OPT = {'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.1, 'clip': 1.0}           # the LTSV-package rollout fine-tune (ltsv_tsfm_helper)
    CALIB = {'material': 'None', 'lrs': [1e-5, 3e-6], 'batches': [4, 16], 'marks': [25, 50, 100], 'seed': SEEDS[0]}
STD_FLOOR = 1e-5                                          # instance normalization floor (a 192-point context is never constant in these cases)
PRED_CHUNK = 64

# Transport caps (HTTP attempts, the bounded second attempts of relay transients) may be raised by the operator; the token / request / fit /
# wall caps may not (amendment 2, 2026-09-24: the relay returned 500 on 40 of 152 attempts at 32 requests in flight).
CAPS = {'tokens': 36_000_000, 'logical_requests': 1450, 'http': int(os.environ.get('SEH_HTTP_CAP', '1650')),
        'extra_transport': int(os.environ.get('SEH_EXTRA_TRANSPORT', '200')), 'fits': 2200, 'fit_retries': 60, 'wall_s': 36 * 3600}
TRANSPORT_CAPS = ('http', 'extra_transport')
STOP_AFTER = os.environ.get('SEH_STOP_AFTER', '')                  # 'source': stop after the Source barrier + the Source zero-shot check
# amendment 3 (transport only): streamed responses, a wait before the one bounded retry, a wait before resuming interrupted trajectories
LLM_STREAM = os.environ.get('SEH_LLM_STREAM', '1') == '1'
RETRY_WAIT_S = float(os.environ.get('SEH_RETRY_WAIT_S', '90'))
RESUME_WAIT_S = float(os.environ.get('SEH_RESUME_WAIT_S', '600'))
STREAM_TOTAL_S = 900.0
HTTP = int(os.environ.get('SEH_HTTP', '16'))
CPU_POOL = int(os.environ.get('SEH_CPU_POOL', '8'))
GPUS = [g.strip() for g in os.environ.get('SEH_GPUS', '0').split(',') if g.strip()]
GPU_LANES_MAX = len(GPUS) * int(os.environ.get('SEH_GPU_LANES_PER_GPU', '3'))
MIN_FREE_GB = float(os.environ.get('SEH_MIN_FREE_GB', '2.0'))
EXTRA_LANE_FREE_GB = float(os.environ.get('SEH_EXTRA_LANE_FREE_GB', '4.0'))   # a Time-MoE worker holds ~3 GB of host memory
MAX_BATCH = 36
MAX_JOB_ATTEMPTS = int(os.environ.get('SEH_MAX_JOB_ATTEMPTS', '4'))            # per (case, plan, seed); OOM retries count too
LAUNCH_SETTLE_S = 90.0                                    # a worker launched this recently has not taken its GPU memory yet
SELECT_ARMS = ('d1', 'd2')
TEST_ARMS = ('f_tsfm', 'f_mlp', 'f0', 'f_naive')
EARLY_ARMS = ('f0', 'f_naive', 'f_mlp')
PRIO = {'wiring': -3, 'calibration': -2, 'source': 0, 'select': 1, 'test': 2}

now = DP.now
write_new = DP.write_new
SafeLedger = DP.SafeLedger


def read(p):
    return context.read_json(Path(p))


def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    context.write_json(Path(p), obj)


def paths() -> dict:
    return {'plan': ROOT / 'plan.json', 'split': ROOT / 'split_copy.json', 'common': ROOT / 'common', 'fits': ROOT / 'fits', 'source': ROOT / 'source',
            'slow': ROOT / 'slow', 'select': ROOT / 'select', 'test': ROOT / 'test', 'naive': ROOT / 'naive', 'freeze': ROOT / 'freeze',
            'ledger': ROOT / 'budget.json', 'incidents': ROOT / 'incidents', 'logs': ROOT / 'logs', 'resources': ROOT / 'resources.json',
            'status': ROOT / 'status.json', 'wiring': ROOT / 'wiring', 'smoke': ROOT / 'smoke', 'calibration': ROOT / 'calibration',
            'consumer': ROOT / 'consumer_frozen.json'}


# ============================================================================= the Consumer as the agents see it
def consumer_frozen() -> dict:
    c = read(paths()['consumer'])
    return {'lr': float(c['lr']), 'steps': int(c['steps']), 'batch': int(c['batch'])}


def consumer_t(cf: dict | None = None) -> dict:
    c = cf or consumer_frozen()
    return {'arch': 'Time-MoE-50M: a pretrained decoder-only mixture-of-experts time-series foundation model (official weights; 113M parameters, about 50M '
                    'active per step). Every plan fine-tunes it from the SAME pretrained weights; it forecasts the 48 steps autoregressively from the 192 '
                    'input points (multi-resolution output heads, the largest head not longer than the remaining horizon)',
            'input': 'univariate: every entity window is one sample; one model shared by the 16 entities of the case',
            'optimizer': 'AdamW (beta2 %g, weight decay 0.1 on all non-bias weights, gradient clipping 1.0)' % OPT['betas'][1], 'lr': c['lr'], 'n_updates': c['steps'],
            'parent_batch': c['batch'],
            'loss': (('the official Time-MoE objective on each 240-point window (192 inputs + 48 targets): teacher-forced Huber loss (delta 2) of every position '
                      'for all four output heads (1, 8, 32, 64 steps ahead) plus the router load-balancing loss (0.02); ') if NATIVE else
                     'MSE of the 48-step autoregressive forecast on normalized targets; ') +
                    '0.5 x parent view + 0.5 x child view (the plan\'s augmented copy of the same parent windows); None trains the parent view only',
            'normalization': ('per-entity mean/std fitted on T (the material scale); the model itself sees every window divided by the mean/std of its own 192 '
                              'input points (official Time-MoE convention, also at serving; forecasts are mapped back)') if NATIVE else
                             'per-entity mean/std fitted on T only; derived pairs are NOT re-normalized; no per-window normalization',
            'metric': ec.CONSUMER['metric'],
            'training': 'fine-tune of the pretrained model, AdamW, %d updates of %d parent windows (budget calibrated once on no-augmentation Source fits), the '
                        'same T scaler and batch index stream for every plan of a case' % (c['steps'], c['batch']),
            'serving': ec.CONSUMER['serving']}


FAST_SYSTEM_T = OS.FAST_SYSTEM_ZF.replace('one shared MLP Consumer', 'one shared Time-MoE Consumer (a pretrained forecasting foundation model fine-tuned on the plan)')
SLOW_EVIDENCE_T = ('The census holds completed Source cases (entity groups of 16 at two periods). Every downstream number in it comes from the Consumer described '
                   'in deployment.consumer. Each case record puts together: (1) T observations (quantiles over the 16 entities); (2) the REAL no-card '
                   'zero-feedback Fast trajectory - what it observed, which plans it built with which stated hypotheses, what its material inspections showed '
                   '(reduced to numbers) and why it committed or abandoned a material; (3) the downstream result of its actual commit (E, three seeds, as pp of '
                   'the case\'s no-augmentation None); (4) SCRIPTED EXPERIMENT RECORDS: twelve fixed uniform programs that an evaluator trained with the same '
                   'Consumer and scored on the same case (three seeds), together with their material-change statistics. The scripted records come from a '
                   'scripted evaluator run; they are not an agent\'s research process and they are not available to a deployed Fast. A plan the Fast built that '
                   'equals a scripted program is marked; other built plans were never trained and have no downstream number.')


def slow_system_t(scope: str) -> str:
    return ' '.join([OS.SLOW_ROLE, SLOW_EVIDENCE_T, OS.SLOW_GUIDE, OS.SLOW_FORMAT % ('', OS.CARD_BODY_LIMIT)])


# ============================================================================= plan
def cases_by_stage() -> dict:
    return read(PARENT / 'frozen_config.json')['cases']


def all_cases(cfg=None) -> list:
    cfg = cfg or cases_by_stage()
    return [c for st in ('source', 'select', 'test') for d in DOMAINS for c in sorted(cfg[st][d])]


def css() -> dict:
    return ec.cases_from_split(read(paths()['split']))


CALIB_CASES = ['%s_SCR_A%d_G1' % (d, a) for d in DOMAINS for a in (1, 2)]


def calib_cases() -> list:
    return list(CALIB_CASES)


def freeze_plan() -> dict:
    P_ = paths()
    if P_['plan'].exists():
        return read(P_['plan'])
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PARENT / 'split_copy.json', P_['split'])
    cfg = cases_by_stage()
    plan = {'package': PACKAGE, 'task': TASK, 'frozen_local': now(), 'identity': IDENTITY, 'exposure_shown_to_fast': OS.EXPOSURE,
            'domains': list(DOMAINS), 'seeds': list(SEEDS), 'cases': cfg, 'counts': {k: {d: len(v) for d, v in s.items()} for k, s in cfg.items()},
            'model': OS.MODEL, 'max_output_tokens': OS.MAX_OUTPUT_TOKENS,
            'fast_limits': {'max_calls': OS.LIMITS.max_calls, 'max_tools': OS.LIMITS.max_tools, 'max_materials': OS.LIMITS.max_materials,
                            'tokens_per_trajectory': OS.TRAJ_TOKEN_CAP, 'max_corrections': OS.MAX_CORRECTIONS, 'commit': 1},
            'consumer': {'model': 'Time-MoE-50M (Maple728/TimeMoE-50M; weights and source under SEH_TSFM_DIR)', 'transformers': TRANSFORMERS_REQUIRED,
                         'dtype': 'float32', 'attention': 'eager', 'optimizer': OPT, 'recipe': RECIPE,
                         'training_loss': ('official TimeMoeForPrediction loss on each 240-point window: input_ids = window[:-1], labels = window[1:], '
                                           'Huber(delta 2) over every position and the four heads + 0.02 router aux loss; 0.5 parent + 0.5 child')
                         if NATIVE else '48-step autoregressive forecast MSE (KV-cache unroll, equal to the official generate), 0.5 parent + 0.5 child',
                         'batch_stream': 'first `batch` columns of train.batch_indices(seed, n_pool, steps)',
                         'inputs': ('each window divided by the mean / std (unbiased, floor %g) of its 192 input points, in training and serving; '
                                    'forecasts mapped back to T-scaler units before scoring' % STD_FLOOR) if NATIVE else 'T-scaler normalized, no per-window normalization',
                         'serving': 'autoregressive KV-cache unroll, bitwise equal to the official generate (wiring check)',
                         'calibration': {**CALIB, 'cases': calib_cases(), 'objective': 'equal-domain mean C_A+C_B nMSE on the six Source cases (None only); '
                                         'exact tie: fewer updates, then smaller batch, then lower lr; zero-shot is reported, never selected'}},
            'fast_system': FAST_SYSTEM_T, 'slow_system_domain': slow_system_t('D01'), 'naive_system': OS.naive_system(), 'caps': CAPS,
            'concurrency': {'http_in_flight': HTTP, 'trajectory_threads_per_stage': HTTP, 'cpu_material_pool': CPU_POOL, 'gpus': GPUS,
                            'gpu_lanes_max': GPU_LANES_MAX, 'min_free_gb': MIN_FREE_GB, 'extra_lane_free_gb': EXTRA_LANE_FREE_GB, 'fit_batch': MAX_BATCH},
            'arms': {'source': ['f0'], 'select': list(SELECT_ARMS), 'test': list(TEST_ARMS),
                     'meaning': {'f_tsfm': 'domain card learned here from Time-MoE evidence, chosen on Select', 'f_mlp': 'the frozen OFFLINE-SKILL domain card',
                                 'f0': 'no card', 'f_naive': 'naive card regenerated with the Time-MoE Consumer description'}},
            'slow': '3 domains x 2 candidate domain cards (no shared card); census = Time-MoE Source evidence only; one format correction; KEEP valid',
            'selection': 'J_d(W) = mean over Select_d of seed-mean E(c, W) / E(c, None); argmin; exact ties -> fewer deployment tokens -> slot order; INCOMPLETE = None',
            'fixed_source_t': 'per domain the scripted program (of 12) with the highest mean pp of None over the 8 Source cases; exact ties -> menu order',
            'refs_test': ['None', 'NoMix (P_NoMixRecipe)', 'Fixed_source_T', 'zero-shot Time-MoE (anchor only; no training, so no augmentation pathway)'],
            'incomplete_rule': 'a trajectory without a legal commit is INCOMPLETE; the Runner never supplies a plan; scored as the no-augmentation delivery',
            'e_barriers': 'the fit worker freezes E predictions from E INPUT windows only (never targets); the controller reads targets and scores a stage only '
                          'after all its commits are frozen and all its fits finished; no score reaches any trajectory; Slow never reads Select / Test results',
            'no_model_files': 'fine-tuned weights are not kept (~450 MB each); the frozen predictions are the scored artefact'}
    want = {'source': {'D01': 8, 'D02': 8, 'D03': 8}, 'select': {'D01': 9, 'D02': 9, 'D03': 6}, 'test': {'D01': 16, 'D02': 16, 'D03': 8}}
    if plan['counts'] != want:
        raise RuntimeError('case counts differ: %s' % plan['counts'])
    write(P_['plan'], plan)
    if not P_['resources'].exists():
        write(P_['resources'], {'gpu_lanes': GPU_LANES_MAX, 'note': 'read before every GPU dispatch; lower it to shed load without a restart'})
    return plan


# ============================================================================= package case cache (copy of the parent view) and frozen input windows
def prepare_common(case: str, cs: ec.CaseSpec) -> None:
    dst = paths()['common'] / case
    if (dst / 'prepared.json').exists():
        return
    src = PARENT / 'common' / case
    dst.mkdir(parents=True, exist_ok=True)
    for name in ('scaler.npz', 'overview.json', 'case_spec.json'):
        shutil.copy2(src / name, dst / name)
    reg = ec.load_registry(src)
    ec.save_registry(dst, {m: {**r, 'inherited_from': str(src.relative_to(REPO))} for m, r in reg.items()})
    ctx = ec.open_case(cs, paths()['common'], 'material')          # T rows only; raises on scaler / binding drift
    np.savez(dst / 'parents.npz', Xp=ctx.parents.X_norm.reshape(-1, L).astype('float32'), yp=ctx.parents.y_norm.reshape(-1, H).astype('float32'))
    sc = ctx.scaler
    # E serving inputs only: rows [min(e) - L, max(e)); targets are never read here
    sl = ec.load_case_slice(cs, min(cs.e) - L, max(cs.e))
    xe = np.stack([w.X_norm for w in data.build_windows(sl, list(cs.e), sc, with_truth=False)], axis=1).astype('float32')
    np.savez(dst / 'e_inputs.npz', X=xe, origins=np.array(cs.e))
    if case in calib_cases():                                     # Source only: C_A + C_B inputs for the calibration checkpoints
        vo = list(cs.c_a) + list(cs.c_b)
        sl = ec.load_case_slice(cs, min(vo) - L, max(vo))
        xv = np.stack([w.X_norm for w in data.build_windows(sl, vo, sc, with_truth=False)], axis=1).astype('float32')
        np.savez(dst / 'val_inputs.npz', X=xv, origins=np.array(vo))
    write(dst / 'prepared.json', {'case': case, 'inherited_materials': sorted(reg), 'prepared_local': now(), 'parent': str(src.relative_to(REPO)),
                                  'e_inputs_rows': [min(cs.e) - L, max(cs.e)]})


def prepare_all(cfg=None) -> None:
    cc = css()
    for c in all_cases(cfg):
        prepare_common(c, cc[c])


def registry(case: str) -> dict:
    return ec.load_registry(paths()['common'] / case)


def truth(case: str, origins: list):
    cs = css()[case]
    with np.load(paths()['common'] / case / 'scaler.npz') as z:
        sc = data.Scaler(mean=z['mean'].copy(), std=z['std'].copy(), scale=z['scale'].copy(), floor_hits=0)
        mase = z['mase'].copy()
    sl = ec.load_case_slice(cs, min(origins) - L, max(origins) + H)
    y = np.stack([w.y_true_raw for w in data.build_windows(sl, list(origins), sc, with_truth=True)], axis=1)
    return y, sc, mase


def score_pred(pred_norm: np.ndarray, y: np.ndarray, sc, mase) -> dict:
    pr = pred_norm.astype(np.float64) * sc.scale[:, None, None] + sc.mean[:, None, None]
    s = ec.score_with_status(pr, y, sc, mase)
    if s['status'] != 'SCORABLE':
        raise RuntimeError('non-scorable prediction')
    return {'nmse': s['normalized_mse_macro'], 'per_entity': s['per_entity_normalized_mse'],
            'per_origin': (((pr - y) / sc.scale[:, None, None]) ** 2).mean(axis=(0, 2)).tolist()}


# ============================================================================= Time-MoE worker (subprocess; torch + transformers 4.40.1 only here)
def _peak_rss_mb():
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        return None


def tsfm_setup():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.pop('KMP_DUPLICATE_LIB_OK', None)
    from methods.ttha.batch_base import readiness as rd
    rd._init_openmp_before_torch()
    os.environ.pop('KMP_DUPLICATE_LIB_OK', None)
    import torch
    import transformers
    if transformers.__version__ != TRANSFORMERS_REQUIRED:
        raise RuntimeError('Time-MoE needs transformers==%s, found %s (set SEH_TSFM_PYTHON)' % (TRANSFORMERS_REQUIRED, transformers.__version__))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device unavailable')
    return torch


def load_model(torch):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from time_moe.models.modeling_time_moe import TimeMoeForPrediction
    model = TimeMoeForPrediction.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32, attn_implementation='eager').to('cuda')
    model.eval()                                                  # attention dropout is 0.0; eval() is not no_grad
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def predict(torch, model, X):
    """(B, 192) normalized -> (B, 48): autoregressive with the KV cache exactly as the official generate (ltsv_tsfm_helper.predict)."""
    outs, left, past, inp = [], H, None, X.unsqueeze(-1)
    while left > 0:
        out = model(input_ids=inp, past_key_values=past, use_cache=True, return_dict=True, max_horizon_length=left)
        step = out.logits[:, -1, :]
        outs.append(step)
        past = out.past_key_values
        inp = step.unsqueeze(-1)
        left -= step.shape[1]
    return torch.cat(outs, dim=1)[:, :H]


def context_stats(torch, ctx):
    """Mean / std (unbiased, floored) of each window's 192 input points: the official Time-MoE instance normalization."""
    return ctx.mean(dim=1, keepdim=True), ctx.std(dim=1, keepdim=True).clamp_min(STD_FLOOR)


def predict_block(torch, model, X: np.ndarray) -> np.ndarray:
    """(E, O, 192) T-scaled numpy -> (E, O, 48) T-scaled float32 forecasts, fixed chunks, no grad (native: each context normalized by
    its own mean / std before the model and the forecast mapped back)."""
    E, O, _ = X.shape
    flat = torch.as_tensor(X.reshape(E * O, L), dtype=torch.float32, device='cuda')
    parts = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], PRED_CHUNK):
            x = flat[i:i + PRED_CHUNK]
            if NATIVE:
                mu, sd = context_stats(torch, x)
                parts.append(predict(torch, model, (x - mu) / sd) * sd + mu)
            else:
                parts.append(predict(torch, model, x))
    return torch.cat(parts).float().cpu().numpy().reshape(E, O, H)


def native_loss(torch, model, W):
    """The official TimeMoeForPrediction training loss on (B, 240) T-scaled windows: normalized by their 192-point context, input_ids =
    window[:-1], labels = window[1:] -> Huber(delta 2) over every position and the four heads + 0.02 router load-balancing loss."""
    mu, sd = context_stats(torch, W[:, :L])
    Z = (W - mu) / sd
    return model(input_ids=Z[:, :-1].unsqueeze(-1), labels=Z[:, 1:].unsqueeze(-1), return_dict=True).loss


def new_optimizer(torch, model, lr: float):
    decay = [p for n, p in model.named_parameters() if 'bias' not in n]
    no_decay = [p for n, p in model.named_parameters() if 'bias' in n]
    return torch.optim.AdamW([{'params': decay, 'weight_decay': OPT['weight_decay']}, {'params': no_decay, 'weight_decay': 0.0}], lr=lr,
                             betas=OPT['betas'], eps=OPT['eps'], weight_decay=OPT['weight_decay'])


def _fit_one(torch, train, model, theta0, j: dict, cache: dict) -> None:
    out = REPO / j['output']
    out.mkdir(parents=True, exist_ok=True)
    c = j['case']
    if c not in cache:
        with np.load(ROOT / 'common' / c / 'parents.npz') as z:
            xp, yp = torch.as_tensor(z['Xp'].copy(), device='cuda'), torch.as_tensor(z['yp'].copy(), device='cuda')
        with np.load(ROOT / 'common' / c / 'e_inputs.npz') as z:
            xe = z['X'].copy()
        xv = None
        if (ROOT / 'common' / c / 'val_inputs.npz').exists():
            with np.load(ROOT / 'common' / c / 'val_inputs.npz') as z:
                xv = z['X'].copy()
        cache.clear()
        cache[c] = (xp, yp, xe, xv)
    xp, yp, xe, xv = cache[c]
    with torch.no_grad():
        for p, s in zip(model.parameters(), theta0):
            p.copy_(s)
    model.zero_grad(set_to_none=True)
    t0 = time.time()
    if j.get('kind') == 'zeroshot':
        pe = predict_block(torch, model, xe)
        np.savez(out / 'pred_e.npz', pred=pe, origins=np.arange(pe.shape[1]))
        if xv is not None:                                        # calibration cases: the zero-shot validation anchor
            np.savez(out / 'pred_val_0.npz', pred=predict_block(torch, model, xv))
        write(out / 'fit.json', {'status': 'OK', 'job': j, 'kind': 'zeroshot', 'seconds': time.time() - t0, 'completed_local': now()})
        print('FIT_OK', j['cid'], 'zeroshot', flush=True)
        return
    seed, lr, steps, batch = int(j['seed']), float(j['lr']), int(j['steps']), int(j['batch'])
    child = None
    if j['material_path']:
        with np.load(j['material_path']) as z:
            if z['Xc'].shape != tuple(xp.shape) or z['yc'].shape != tuple(yp.shape):
                raise RuntimeError('child-parent shape mismatch')
            if not np.isfinite(z['Xc']).all() or not np.isfinite(z['yc']).all():
                raise RuntimeError('nonfinite frozen material')
            child = (torch.as_tensor(z['Xc'].copy(), device='cuda'), torch.as_tensor(z['yc'].copy(), device='cuda'))
    idxs = torch.as_tensor(train.batch_indices(seed, n_pool=len(xp), n_updates=steps)[:, :batch], device='cuda')
    torch.manual_seed(seed)
    for n, p in model.named_parameters():                    # head recipe: frozen backbone, only lm_heads.* train (no gradient reaches the backbone)
        p.requires_grad_(n.startswith('lm_heads.') if HEAD_ONLY else True)
    opt = new_optimizer(torch, model, lr)
    marks = set(j.get('marks') or [])
    curve, norms = [], []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for step in range(1, steps + 1):
        idx = idxs[step - 1]
        if NATIVE:
            loss = native_loss(torch, model, torch.cat([xp[idx], yp[idx]], dim=1))
            if child is not None:
                loss = 0.5 * loss + 0.5 * native_loss(torch, model, torch.cat([child[0][idx], child[1][idx]], dim=1))
        else:
            loss = torch.mean((predict(torch, model, xp[idx]) - yp[idx]) ** 2)
            if child is not None:
                loss = 0.5 * loss + 0.5 * torch.mean((predict(torch, model, child[0][idx]) - child[1][idx]) ** 2)
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), OPT['clip'])))
        opt.step()
        model.zero_grad(set_to_none=True)
        v = float(loss.detach())
        if not np.isfinite(v):
            raise RuntimeError('nonfinite training loss')
        if step == 1 or step % 10 == 0 or step == steps:
            curve.append([step, v])
        if step in marks and xv is not None:
            np.savez(out / ('pred_val_%d.npz' % step), pred=predict_block(torch, model, xv))
    torch.cuda.synchronize()
    train_s = time.time() - start
    pe = predict_block(torch, model, xe)
    del opt
    np.savez(out / 'pred_e.npz', pred=pe, origins=np.arange(pe.shape[1]))
    write(out / 'fit.json', {'status': 'OK', 'job': j, 'train_seconds': train_s, 'seconds': time.time() - t0, 'loss_curve': curve,
                             'grad_norm_before_clip_last': norms[-1], 'peak_cuda_mb': torch.cuda.max_memory_allocated() / 2 ** 20,
                             'peak_rss_mb_process': _peak_rss_mb(), 'torch': str(torch.__version__), 'device': torch.cuda.get_device_name(0),
                             'visible_device': os.environ.get('CUDA_VISIBLE_DEVICES'), 'completed_local': now()})
    print('FIT_OK', j['cid'], 'train_s', round(train_s, 1), flush=True)


def fit_batch(spec_path: str) -> None:
    torch = tsfm_setup()
    from methods.ttha.batch_base import train
    t0 = time.time()
    model = load_model(torch)
    theta0 = [p.detach().clone() for p in model.parameters()]
    load_s = time.time() - t0
    spec = read(spec_path)
    cache = {}
    for j in spec['jobs']:
        fp = REPO / j['output'] / 'fit.json'
        if fp.exists() and read(fp).get('status') == 'OK':
            continue
        try:
            if j.get('kind') == 'generate_check':
                _generate_check(torch, model, theta0, j)
            else:
                _fit_one(torch, train, model, theta0, j, cache)
        except Exception as exc:  # noqa: BLE001
            write(REPO / j['output'] / 'fit_error.json', {'job': j, 'error': repr(exc)[:500], 'traceback': traceback.format_exc()[-4000:], 'local': now()})
            print('FIT_FAIL', j['cid'], repr(exc)[:200], flush=True)
    print('BATCH_DONE model_load_s %.1f' % load_s, flush=True)


def _generate_check(torch, model, theta0, j):
    """Wiring: the training / scoring unroll equals the official generate on the real E inputs of a Source case (theta0)."""
    out = REPO / j['output']
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for p, s in zip(model.parameters(), theta0):
            p.copy_(s)
    with np.load(ROOT / 'common' / j['case'] / 'e_inputs.npz') as z:
        X = torch.as_tensor(z['X'].reshape(-1, L)[:32].copy(), device='cuda')
    if NATIVE:                                                    # the model sees instance-normalized contexts at serving
        mu, sd = context_stats(torch, X)
        X = (X - mu) / sd
    with torch.no_grad():
        mine = predict(torch, model, X)
        off = model.generate(inputs=X.clone(), max_new_tokens=H)[:, -H:]
    d = (mine.double() - off.double()).abs()
    write(out / 'fit.json', {'status': 'OK', 'job': j, 'kind': 'generate_check', 'bitwise_equal': bool(torch.equal(mine, off)), 'max_abs': float(d.max()),
                             'horizon_heads': list(model.config.horizon_lengths), 'n_params': sum(p.numel() for p in model.parameters()),
                             'completed_local': now()})


def worker_env(device=None) -> dict:
    env = dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env.update(PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')     # less fragmentation when several workers share a card
    if device is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(device)
    return env


def run_gpu_process(args: list, log: Path, timeout: float, device=None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open('w', encoding='utf-8') as f:
            p = subprocess.run([TSFM_PYTHON, '-B', '-u', '-m', MODULE] + [str(a) for a in args], cwd=str(REPO), env=worker_env(device), stdout=f,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=timeout)
        return p.returncode
    except subprocess.TimeoutExpired:
        return -999


# ============================================================================= GPU queue (priority, per-case batches, lanes over GPUS)
def gpu_lanes() -> int:
    try:
        return max(1, min(GPU_LANES_MAX, int(read(paths()['resources'])['gpu_lanes'])))
    except Exception:  # noqa: BLE001
        return GPU_LANES_MAX


class GPUQueue:
    """Each (case, phys, seed) is fitted at most once; lane i runs on GPUS[i % len(GPUS)]; a worker process loads Time-MoE once and runs up to
    MAX_BATCH jobs of one case and priority (restoring the pretrained weights before each)."""

    def __init__(self, led: SafeLedger):
        self.led, self.cv = led, threading.Condition()
        self.heap, self.seq = [], 0
        self.state, self.dirs, self.meta, self.errors = {}, {}, {}, {}
        self.stop = False
        self.threads = []
        self.batches = len(list((paths()['fits'] / '_batches').glob('batch_*.json')))
        self.dev_running = {g: 0 for g in GPUS}
        self.dev_launch, self.dev_lock = {g: [] for g in GPUS}, threading.Lock()
        self.attempts, self.peak_mb, self.peak_by = {}, 0.0, {}

    def start(self):
        for i in range(GPU_LANES_MAX):
            t = threading.Thread(target=self._lane, args=(i,), name='gpu-%d' % i, daemon=True)
            t.start()
            self.threads.append(t)

    def _push(self, stage, cid, job):
        self.state[cid] = 'queued'
        self.seq += 1
        heapq.heappush(self.heap, (PRIO[stage], self.seq, job))

    def submit(self, stage: str, case: str, phys: str, *, lr=None, steps=None, batch=None, marks=None, seeds=SEEDS, tag='') -> list:
        """A plan fit under the frozen Consumer (or explicit calibration settings when lr / steps / batch are given)."""
        reg = registry(case)
        key, path = reg[phys]['key'], reg[phys].get('path')
        if lr is None:
            cf = consumer_frozen()
            lr, steps, batch = cf['lr'], cf['steps'], cf['batch']
        cids = []
        with self.cv:
            for s in seeds:
                cid = ec.cell_id(case, phys, s, tag)
                cids.append(cid)
                if cid in self.state and self.state[cid] != 'failed':
                    continue
                out = paths()['fits'] / case / (phys + ('__' + tag if tag else '')) / str(s)
                self.meta[cid] = {'case': case, 'phys': phys, 'seed': s, 'key': key, 'stage': stage}
                fp = out / 'fit.json'
                if fp.exists() and read(fp).get('status') == 'OK' and read(fp)['job']['key'] == key:
                    self.state[cid], self.dirs[cid] = 'ok', str(out)
                    continue
                self._push(stage, cid, {'cid': cid, 'case': case, 'phys': phys, 'seed': s, 'key': key, 'material_path': path, 'stage': stage, 'lr': lr,
                                        'steps': steps, 'batch': batch, 'marks': list(marks or []), 'output': str(out.relative_to(REPO)).replace('\\', '/')})
            self.cv.notify_all()
        return cids

    def submit_special(self, stage: str, case: str, kind: str, name: str) -> str:
        cid = '%s__%s' % (case, name)
        out = paths()['fits'] / case / name
        with self.cv:
            if cid in self.state and self.state[cid] != 'failed':
                return cid
            self.meta[cid] = {'case': case, 'phys': name, 'seed': 0, 'key': kind, 'stage': stage}
            if (out / 'fit.json').exists() and read(out / 'fit.json').get('status') == 'OK':
                self.state[cid], self.dirs[cid] = 'ok', str(out)
            else:
                self._push(stage, cid, {'cid': cid, 'case': case, 'phys': name, 'seed': 0, 'key': kind, 'kind': kind, 'material_path': None, 'stage': stage,
                                        'output': str(out.relative_to(REPO)).replace('\\', '/')})
            self.cv.notify_all()
        return cid

    def wait(self, cids: list) -> None:
        with self.cv:
            while not all(self.state.get(c) in ('ok', 'failed') for c in cids):
                if self.stop:
                    raise RuntimeError('GPU queue stopped')
                self.cv.wait(10)
            bad = [c for c in cids if self.state[c] == 'failed']
        if bad:
            raise RuntimeError('fits failed: %s (%s)' % (bad[:3], [self.errors.get(b) for b in bad[:3]]))

    def _take(self, lane: int):
        with self.cv:
            while True:
                if self.stop:
                    return None
                if lane < gpu_lanes() and self.heap and (lane == 0 or OS.free_gb() >= EXTRA_LANE_FREE_GB):
                    break
                self.cv.wait(5)
            prio, seq, job = heapq.heappop(self.heap)
            batch, keep = [job], []
            while self.heap:
                it = heapq.heappop(self.heap)
                if it[0] == prio and it[2]['case'] == job['case'] and len(batch) < MAX_BATCH:
                    batch.append(it[2])
                else:
                    keep.append(it)
            for it in keep:
                heapq.heappush(self.heap, it)
            for j in batch:
                self.state[j['cid']] = 'running'
            return batch

    def _lane(self, lane: int):
        while True:
            batch = self._take(lane)
            if batch is None:
                return
            t0 = time.time()
            while OS.free_gb() < MIN_FREE_GB and time.time() - t0 < 1800:
                time.sleep(5)
            try:
                self._run(batch, lane)
            except Exception as exc:  # noqa: BLE001  (never lose a lane: its jobs go back to the queue)
                print('LANE_ERROR', lane, repr(exc)[:300], flush=True)
                with self.cv:
                    for j in batch:
                        if self.state.get(j['cid']) == 'running':
                            self._push(j['stage'], j['cid'], j)
                    self.cv.notify_all()
                time.sleep(10)

    def _run(self, batch: list, lane: int):
        ok_reserved = []
        for j in batch:
            try:
                self.led.reserve_fit(j['cid'])
                ok_reserved.append(j)
            except Exception as exc:  # noqa: BLE001  (fit cap / wall cap)
                with self.cv:
                    self.state[j['cid']], self.errors[j['cid']] = 'failed', repr(exc)[:200]
                    self.cv.notify_all()
        if not ok_reserved:
            return
        dev = self._acquire_device(max(self._job_need_mb(j) for j in ok_reserved))
        try:
            with self.cv:
                self.batches += 1
                n = self.batches
            spec = paths()['fits'] / '_batches' / ('batch_%04d.json' % n)
            write(spec, {'jobs': ok_reserved, 'lane': lane, 'device': dev, 'written_local': now()})
            t0 = time.time()
            rc = run_gpu_process(['--fit-batch', spec], paths()['logs'] / 'fits' / ('batch_%04d.log' % n), 600 + 900 * len(ok_reserved), device=dev)
            secs = (time.time() - t0) / max(1, len(ok_reserved))
        finally:
            with self.cv:
                self.dev_running[dev] -= 1
                self.cv.notify_all()
        for j in ok_reserved:
            fp = REPO / j['output'] / 'fit.json'
            ok = fp.exists() and read(fp).get('status') == 'OK' and read(fp)['job']['key'] == j['key']
            reason = ''
            if ok:
                pk = float(read(fp).get('peak_cuda_mb') or 0.0)
                self.peak_mb = max(self.peak_mb, pk)
                if j.get('kind') is None and pk:
                    kk = (int(j.get('batch') or 8), 2 if j.get('material_path') else 1)
                    self.peak_by[kk] = max(self.peak_by.get(kk, 0.0), pk + 600.0)     # + the CUDA context outside the allocator
            else:
                ep = REPO / j['output'] / 'fit_error.json'
                reason = ('batch rc=%s' % rc) + ((' ' + read(ep)['error'][:80]) if ep.exists() else '')
            self.led.finish_fit(j['cid'], ok, secs, reason)
            with self.cv:
                self.attempts[j['cid']] = self.attempts.get(j['cid'], 0) + 1
                if ok:
                    self.state[j['cid']], self.dirs[j['cid']] = 'ok', str(REPO / j['output'])
                elif self.attempts[j['cid']] < MAX_JOB_ATTEMPTS and self.led.can_retry():
                    with self.led.lock:
                        self.led.s['retries_used'] += 1
                        self.led.s['events'].append({'kind': 'fit_retry', 'cell': j['cid'], 'reason': reason, 'attempt': self.attempts[j['cid']], 'epoch': time.time()})
                        self.led._save()
                    self._push(j['stage'], j['cid'], j)
                else:
                    self.state[j['cid']], self.errors[j['cid']] = 'failed', reason
                self.cv.notify_all()
        print('GPU_BATCH', n, 'lane', lane, 'gpu', dev, 'jobs', len(ok_reserved), 'rc', rc, 'sec/fit', round(secs, 1), 'queue', len(self.heap), flush=True)

    # --- device choice: the least-busy card of GPUS that has room for one more Time-MoE worker (nvidia-smi free memory)
    def _vram_free_mb(self, dev: str):
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', str(dev)], capture_output=True, text=True, timeout=30)
            return float(out.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            return None

    def _job_need_mb(self, j: dict) -> float:
        """GPU memory of one job: the observed peak of its (batch, views) kind when known, else the measured model (Time-MoE-50M native,
        RTX 4060: 3.9 GB none/8, 5.9 GB child/8, 9.9 GB none/32, >14.3 GB child/32 -> 2.1 GB + 0.25 GB per window-view); +15 %."""
        if j.get('kind') in ('zeroshot', 'generate_check'):
            return 3000.0
        views = 2 if j.get('material_path') else 1
        key = (int(j.get('batch') or 8), views)
        seen = self.peak_by.get(key)
        return 1.15 * (seen if seen else 2100.0 + 250.0 * key[0] * views)

    def _acquire_device(self, need: float) -> str:
        """Least-busy card whose free memory, minus the needs of workers launched there in the last LAUNCH_SETTLE_S (not yet allocated), still
        fits this worker. The check and the reservation happen under one lock so simultaneous lanes cannot both claim the same room."""
        t0 = time.time()
        while True:
            with self.dev_lock:
                nowt = time.time()
                with self.cv:
                    order = sorted(GPUS, key=lambda g: (self.dev_running[g], GPUS.index(g)))
                for g in order:
                    self.dev_launch[g] = [(t, n) for t, n in self.dev_launch[g] if nowt - t < LAUNCH_SETTLE_S]
                    free = self._vram_free_mb(g)
                    room = None if free is None else free - sum(n for _, n in self.dev_launch[g])
                    if room is None or room >= need or nowt - t0 > 1800:
                        self.dev_launch[g].append((nowt, need))
                        with self.cv:
                            self.dev_running[g] += 1
                        return g
            time.sleep(10)

    def snapshot(self) -> dict:
        with self.cv:
            c = {}
            for v in self.state.values():
                c[v] = c.get(v, 0) + 1
            return {'states': c, 'queued': len(self.heap), 'batches': self.batches, 'lanes': gpu_lanes(), 'gpus': GPUS, 'running_per_gpu': dict(self.dev_running), 'peak_mb': round(self.peak_mb)}


# ============================================================================= scoring (controller side, behind the stage barrier)
def score_stage(stage: str, gpu: GPUQueue, cells: dict) -> None:
    """cells: {case: [cid]}; reads targets only when <stage>/models_frozen.json exists; writes <stage>/scores/<case>.json."""
    if not (paths()[stage] / 'models_frozen.json').exists():
        raise RuntimeError('E barrier: %s predictions not frozen' % stage)
    for case, cids in cells.items():
        out = paths()[stage] / 'scores' / ('%s.json' % case)
        if out.exists() and all(c in read(out)['cells'] for c in cids):
            continue
        cs = css()[case]
        y, sc, mase = truth(case, list(cs.e))
        res = {}
        for cid in cids:
            d = Path(gpu.dirs[cid])
            with np.load(d / 'pred_e.npz') as z:
                pr = z['pred'].copy()
            res[cid] = {'phys': gpu.meta[cid]['phys'], 'seed': int(gpu.meta[cid]['seed']), 'key': gpu.meta[cid]['key'], **score_pred(pr, y, sc, mase)}
        write(out, {'case': case, 'stage': stage, 'origins': list(cs.e), 'cells': res, 'scored_local': now()})


def stage_e(stage: str, case: str) -> dict:
    r = read(paths()[stage] / 'scores' / ('%s.json' % case))
    out = {}
    for cell in r['cells'].values():
        out.setdefault(cell['phys'], {})[int(cell['seed'])] = float(cell['nmse'])
    return out


# ============================================================================= zero-feedback Fast with the Time-MoE description
class ZFAdapterT(OS.ZFAdapter):
    def case_card(self) -> dict:
        card = super().case_card()
        card['consumer'] = consumer_t()
        return card


def run_branch_t(bdir: Path, case: str, knowledge: br.Knowledge, led, client, *, cs: ec.CaseSpec, arm: str, resume: bool = False) -> dict:
    """OS.run_branch with the Time-MoE case card and Fast system text (everything else identical)."""
    common, split_path = paths()['common'], paths()['split']
    kn_p = bdir / 'knowledge.json'
    res_state, spent, ratio = None, 0, None
    if not resume:
        bdir.mkdir(parents=True, exist_ok=False)
    adapter = ZFAdapterT(bdir, case, led, common=common, split_path=split_path, cs=cs)
    if resume:
        prior = read(bdir / 'branch_result.json')
        if prior['status'] != 'INCOMPLETE' or prior['failure_kind'] != 'AGENT_CALL_FAILED' or (bdir / case / 'commit.json').exists():
            raise RuntimeError('branch %s is not resumable' % bdir.name)
        if read(kn_p) != br.json_copy(asdict(knowledge)):
            raise RuntimeError('frozen knowledge differs')
        rows = OS._trace(bdir)
        pend = max(i for i, r in enumerate(rows) if r['event'] == 'fast_request')
        if any(r['event'] in ('fast_response', 'response_rejected') for r in rows[pend:]):
            raise RuntimeError('last request was answered')
        prefix = rows[:pend]
        answered = sum(r['event'] in ('fast_response', 'response_rejected') for r in prefix)
        built = [r['output']['plan_id'] for r in prefix if r['event'] == 'tool_completed' and r['tool'] == 'build_material']
        feats = {}
        for r in prefix:
            if r['event'] == 'tool_completed' and r['tool'] == 'inspect_data':
                feats.update(r['output'].get('batch_features', {}))
        n_res = 1 + len(list(bdir.glob('resume*.json')))
        shutil.move(bdir / 'trace.jsonl', bdir / ('trace_before_resume%d.jsonl' % n_res))
        shutil.move(bdir / 'branch_result.json', bdir / ('branch_result_before_resume%d.json' % n_res))
        sink = OS.event_sink(bdir / 'trace.jsonl')
        for r in prefix:
            sink(r)
        sink({'event_id': case + ':resume%d' % n_res, 'event': 'job_resumed', 'calls': answered, 'epoch': time.time()})
        prefix = prefix + [{'event_id': case + ':resume%d' % n_res, 'event': 'job_resumed'}]
        res_state = {'trace': prefix, 'calls': answered, 'tools_used': sum(r['event'] == 'tool_started' for r in prefix), 'materials': len(built),
                     'corrections': sum(r['event'] in ('tool_rejected', 'response_rejected') for r in prefix), 'built': [adapter.restore_built(m) for m in built],
                     'features': feats}
        write(bdir / ('resume%d.json' % n_res), {'epoch': time.time(), 'reason': 'the next Fast request was never answered (transport fault); continued from the identical prefix',
                                               'prefix_events': len(prefix) - 1, 'calls_before': answered, 'restored_plans': built})
        spent = sum((e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0) for e in led.s['events'] if e.get('kind') == 'llm_finished' and e.get('unit') == bdir.name)
    else:
        write(kn_p, asdict(knowledge))
        sink = OS.event_sink(bdir / 'trace.jsonl')
    tc = OS.TrajectoryClient(client, bdir.name, FAST_SYSTEM_T, case=case, arm=arm, spent=spent, ratio=ratio)
    LIM = OS.LIMITS
    result = zf.run_job_zf(job_id=case, knowledge=knowledge, adapter=adapter, client=tc, public=OS.public_candidates(case), allowed_features=OS.ALLOWED_FEATURES,
                           tool_contracts=OS.CONTRACTS_ZF, entity_count=ec.COHORT_SIZE, limits=zf.Limits(LIM.max_calls, LIM.max_tools, LIM.max_materials, max(60.0, led.remaining())),
                           on_event=sink, guard=led.check_wall, evidence_roundtrip=OS.EVIDENCE_ROUNDTRIP, max_corrections=OS.MAX_CORRECTIONS, token_gate=tc.gate,
                           resume=res_state)
    rec = asdict(result)
    rec.pop('trace')
    rec.update({'arm': arm, 'tokens_spent': tc.spent, 'material_build_seconds': round(adapter.build_seconds, 2), 'finished_local': now()})
    write(bdir / 'branch_result.json', rec)
    if result.status == 'COMPLETE':
        reason = next(r['output']['reason'] for r in reversed(result.trace) if r['event'] == 'tool_completed' and r['tool'] == 'commit')
        cr = adapter.commit_record(result.committed_plan_id, reason)
        cr['constructed_plans'] = [m for m in adapter._reg() if m not in OS.PUBLIC]
        write_new(bdir / case / 'commit.json', cr)
    print('BRANCH', bdir.name, result.status, result.failure_kind or '', 'commit', result.committed_plan_id, 'calls', result.calls, 'tokens', tc.spent, flush=True)
    return rec


def commit_of(sdir: Path, case: str, arm: str):
    p = sdir / 'branches' / ('%s__%s' % (case, arm)) / case / 'commit.json'
    return read(p) if p.exists() else None


def delivered_phys(sdir: Path, case: str, arm: str) -> str:
    cm = commit_of(sdir, case, arm)
    return cm['physical_material'] if cm else 'None'


def run_fast_stage(stage: str, jobs: list, knowledge_of, *, led, client, on_commit, sdir: Path) -> dict:
    cc = css()
    errors, results = [], {}
    stop = threading.Event()
    q = list(jobs)
    qlock = threading.Lock()

    def finish(case, arm, r):
        results[(case, arm)] = r
        on_commit(case, arm, commit_of(sdir, case, arm))

    def worker():
        while True:
            with qlock:
                if not q or stop.is_set():
                    return
                case, arm = q.pop(0)
            if rt.unknown_usage_blocks(led):
                stop.set()
                return
            bdir = sdir / 'branches' / ('%s__%s' % (case, arm))
            try:
                if bdir.exists() and not (bdir / 'branch_result.json').exists():
                    k = 1 + len(list(bdir.parent.glob(bdir.name + '__killed*')))
                    shutil.move(bdir, bdir.parent / ('%s__killed%d' % (bdir.name, k)))
                    led.event(kind='trajectory_restarted_after_kill', case=case, arm=arm, stage=stage)
                if (bdir / 'branch_result.json').exists():
                    r = read(bdir / 'branch_result.json')
                    if r['status'] == 'COMPLETE' or r['failure_kind'] != 'AGENT_CALL_FAILED':
                        finish(case, arm, r)
                        continue
                    finish(case, arm, run_branch_t(bdir, case, knowledge_of(case, arm), led, client, cs=cc[case], arm=arm, resume=True))
                else:
                    finish(case, arm, run_branch_t(bdir, case, knowledge_of(case, arm), led, client, cs=cc[case], arm=arm))
            except Exception as exc:  # noqa: BLE001
                errors.append(('branch', case, arm, repr(exc)[:300]))
                print('BRANCH_ERROR', stage, case, arm, repr(exc)[:300], flush=True)

    threads = [threading.Thread(target=worker, name='%s-%d' % (stage, i), daemon=True) for i in range(HTTP)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    again = [k for k, r in results.items() if r.get('status') == 'INCOMPLETE' and r.get('failure_kind') == 'AGENT_CALL_FAILED']
    if again and RESUME_WAIT_S > 0:                          # amendment 3: let a relay outage pass before the one resume round
        print('TRANSPORT_RESUME_WAIT', stage, len(again), 'trajectories', RESUME_WAIT_S, 's', flush=True)
        time.sleep(RESUME_WAIT_S)
    for case, arm in again:
        if led.s.get('extra_transport_attempts', 0) >= CAPS['extra_transport'] or rt.unknown_usage_blocks(led):
            break
        with led.lock:
            led.s['extra_transport_attempts'] = led.s.get('extra_transport_attempts', 0) + 1
            led._save()
        bdir = sdir / 'branches' / ('%s__%s' % (case, arm))
        try:
            finish(case, arm, run_branch_t(bdir, case, knowledge_of(case, arm), led, client, cs=cc[case], arm=arm, resume=True))
        except Exception as exc:  # noqa: BLE001
            errors.append(('resume', case, arm, repr(exc)[:300]))
    missing = [j for j in jobs if j not in results]
    # a trajectory cut by a technical call failure is not an agent that failed to commit: it must not flow on as a no-augmentation delivery
    # (amendment 2); the stage stops and a resume continues it from its last state
    cut = ['%s__%s' % k for k, r in results.items() if r.get('status') == 'INCOMPLETE' and r.get('failure_kind') == 'AGENT_CALL_FAILED']
    summ = {'stage': stage, 'finished_local': now(), 'n_jobs': len(jobs), 'complete': sum(r.get('status') == 'COMPLETE' for r in results.values()),
            'incomplete': {'%s__%s' % k: [r.get('failure_kind'), r.get('reason')] for k, r in results.items() if r.get('status') != 'COMPLETE'},
            'missing': ['%s__%s' % j for j in missing], 'transport_interrupted': cut, 'errors': errors, 'unknown_usage_blocking': rt.unknown_usage_blocks(led)}
    write(sdir / ('dispatch_summary%s.json' % ('' if stage in ('source', 'select') else '_' + stage)), summ)
    print('STAGE_DISPATCH_DONE', stage, summ['complete'], '/', len(jobs), 'errors', len(errors), 'missing', len(missing), 'cut', len(cut), flush=True)
    if missing or cut or summ['unknown_usage_blocking']:
        raise RuntimeError('%s stage incomplete: missing %s transport-interrupted %s errors %s; resume after the fault is cleared'
                           % (stage, summ['missing'][:5], cut[:5], errors[:3]))
    return summ


class _StreamedResponse:
    """The fields ZFMeteredClient.call reads from a chat.completion (model, usage, choices[0].message.content, model_dump), rebuilt from a stream."""

    def __init__(self, rec: dict, usage):
        self.model, self.usage, self._rec = rec['model'], usage, rec
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=rec['choices'][0]['message']['content']))]

    def model_dump(self, mode='json'):
        return self._rec


class ZFStreamClient(OS.ZFMeteredClient):
    """Amendment 3 (relay recommendation, 2026-09-24): stream=true with include_usage, so bytes flow during the 1-4 minute generations and the
    relay can retry a failure before the first byte (nothing generated) without risking a second generation. A failure inside the stream is
    re-raised as a connection / timeout exception, which the frozen classifier counts as a transient transport fault. The one bounded second
    attempt of a transient waits RETRY_WAIT_S first (the same messages object marks the retry of one logical request). Everything else --
    ledger, caps, receipts, returned-model check -- is the unchanged ZFMeteredClient.call."""
    _tl = threading.local()

    def _transport(self, messages, max_tokens, timeout):
        if getattr(self._tl, 'failed', None) is messages and RETRY_WAIT_S > 0:
            time.sleep(RETRY_WAIT_S)
        self._tl.failed = messages
        r = self._stream(messages, max_tokens, timeout) if LLM_STREAM else super()._transport(messages, max_tokens, timeout)
        self._tl.failed = None
        return r

    def _stream(self, messages, max_tokens, timeout):
        import httpx
        import openai
        t0 = time.time()
        st = self.api.chat.completions.create(model=OS.MODEL['requested'], messages=messages, temperature=OS.MODEL['temperature'], max_tokens=max_tokens,
                                              timeout=timeout, stream=True, stream_options={'include_usage': True})
        text, reason, model, rid, usage, finish, n, first = [], [], None, None, None, None, 0, None
        try:
            for ch in st:
                n += 1
                first = time.time() - t0 if first is None else first
                model, rid = ch.model or model, ch.id or rid
                if ch.usage is not None:
                    usage = ch.usage
                for c in ch.choices or []:
                    d = c.delta
                    if d is not None and d.content:
                        text.append(d.content)
                    rc = getattr(d, 'reasoning_content', None) if d is not None else None
                    if rc:
                        reason.append(rc)
                    finish = c.finish_reason or finish
                if time.time() - t0 > STREAM_TOTAL_S:
                    raise TimeoutError('stream exceeded %.0f s' % STREAM_TOTAL_S)
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise TimeoutError('stream read timeout: %s' % str(exc)[:200]) from None
        except (httpx.HTTPError, openai.APIError) as exc:
            raise ConnectionError('stream interrupted (connection): %s %s' % (type(exc).__name__, str(exc)[:200])) from None
        finally:
            close = getattr(st, 'close', None)
            if close:
                close()
        if model is None:
            raise ConnectionError('stream ended without any chunk (connection)')
        rec = {'id': rid, 'object': 'chat.completion', 'model': model,
               'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''.join(text), 'reasoning_content': ''.join(reason) or None},
                            'finish_reason': finish}],
               'usage': usage.model_dump(mode='json') if usage is not None else None,
               'transport': {'stream': True, 'chunks': n, 'first_chunk_s': round(first or 0.0, 3), 'total_s': round(time.time() - t0, 3)}}
        return _StreamedResponse(rec, usage)


def make_client(led, stage: str):
    return ZFStreamClient(led, ROOT / stage / 'llm', http_cap=CAPS['http'], stage=stage, incidents=paths()['incidents'], extra_cap=CAPS['extra_transport'])


# ============================================================================= Source evidence, census, Slow, naive
def source_cells(case: str) -> list:
    phys = list(MENU_IDS)
    d = delivered_phys(paths()['source'], case, 'f0')
    if d not in phys:
        phys.append(d)
    return phys


def _g_seed(se, phys):
    return [100.0 * (se['None'][s] - se[phys][s]) / statistics.fmean(se['None'][t] for t in SEEDS) for s in SEEDS]


def scripted_records_t(case: str, se: dict) -> dict:
    out = {}
    for m in MENU_IDS:
        g = _g_seed(se, m)
        rec = {'program': OS.MENU_LABEL[m], 'steps': OS.MENU_STEPS[m] if OS.MENU_STEPS[m] is not None else ('none' if m == 'None' else 'FixedMixup (historical w=0.25 same-entity mixup)'),
               'G_vs_None_mean': OS._sig(statistics.fmean(g)), 'G_vs_None_by_seed': OS._sig(g)}
        if m != 'None':
            sp = OS.screen_dir(case) / 'aug_materials' / (m + '__summary.json')
            if sp.exists():
                s = read(sp)
                rec['material_change'] = OS._sig({'rms_X': s.get('rms_change_X'), 'rms_y': s.get('rms_change_y'), 'fraction_points_changed_X': s.get('fraction_points_changed_X'),
                                                  'windows_bitwise_unchanged': s.get('windows_bitwise_unchanged')})
        out[OS.MENU_LABEL[m]] = rec
    return out


def census_case_t(case: str) -> dict:
    P_ = paths()
    bdir = P_['source'] / 'branches' / ('%s__f0' % case)
    se = stage_e('source', case)
    cm = commit_of(P_['source'], case, 'f0')
    phys = cm['physical_material'] if cm else 'None'
    g = _g_seed(se, phys)
    scr = scripted_records_t(case, se)
    gm = statistics.fmean(g)
    rank = 1 + sum(1 for v in scr.values() if v['G_vs_None_mean'] > gm)
    same = OS._menu_key_label(case).get(cm['material_key']) if cm else 'None'
    outcome = {'committed_plan': cm['label'] if cm else 'INCOMPLETE (no legal commit; scored as no augmentation)', 'same_material_as_scripted_program': same,
               'commit_reason': cm['reason'] if cm else None, 'G_vs_None_mean': OS._sig(gm), 'G_vs_None_by_seed': OS._sig(g),
               'rank_among_12_scripted_plus_commit': '%d of 13' % rank}
    return {'case_ref': case, 'domain_id': case[:3], 'observations': OS._obs_summary(P_['common'], css()[case]),
            'fast_trajectory': OS.compact_trajectory(bdir, case), 'fast_outcome': outcome,
            'scripted_records': {'note': 'SCRIPTED experiment records: an evaluator trained each of these 12 fixed uniform programs on this case with the deployment Consumer '
                                         '(3 seeds) and scored E as pp of the case None mean (positive = better than no augmentation). Produced by a scripted run, not by an '
                                         'agent\'s research, and not available to a deployed Fast.', 'programs': scr}}


def build_census_t(scope: str) -> dict:
    cases = sorted(read(paths()['plan'])['cases']['source'][scope])
    return {'scope': scope, 'domains': [scope], 'n_cases': len(cases), 'cases': [census_case_t(c) for c in cases], 'legal_evidence_refs': OS.legal_refs_of(cases),
            'units': 'G = 100 x (E(None) - E(plan)) / E(None) of that case, three-seed mean unless by_seed; E = four 48-hour forecasts at t+192..t+336 of the case'}


def slow_payload_t(cen: dict, scope: str, slot: int) -> dict:
    p = OS.slow_payload_zf(cen, scope, slot)
    p['deployment']['consumer'] = consumer_t()
    return p


def fixed_source_t() -> dict:
    out_p = paths()['freeze'] / 'fixed_source_t.json'
    if out_p.exists():
        return read(out_p)
    plan = read(paths()['plan'])
    rec = {'rule': plan['fixed_source_t'], 'frozen_local': now(), 'domains': {}}
    for d in DOMAINS:
        cases = sorted(plan['cases']['source'][d])
        g = {m: statistics.fmean(statistics.fmean(_g_seed(stage_e('source', c), m)) for c in cases) for m in MENU_IDS}
        best = max(MENU_IDS, key=lambda m: (g[m], -MENU_IDS.index(m)))
        rec['domains'][d] = {'program': best, 'label': OS.MENU_LABEL[best], 'mean_G_vs_None_by_program': g}
    write_new(out_p, rec)
    return rec


def source_zeroshot_check(gpu, cases: list) -> dict:
    """Operator go / no-go read behind the Source barrier (no LLM; amendment 2): does fine-tuning with any scripted plan beat the zero-shot
    Time-MoE on Source E? Zero-shot predictions come from the same E input windows and the same scorer as every Source cell. Written to
    <ROOT>/source_zeroshot_check.json only; Fast, Slow, the census and Test never read it."""
    P_ = paths()
    if not (P_['source'] / 'models_frozen.json').exists():
        raise RuntimeError('E barrier: Source predictions not frozen')
    zc = {c: gpu.submit_special('source', c, 'zeroshot', 'zeroshot') for c in cases}
    gpu.wait(list(zc.values()))
    rows = {}
    for c in cases:
        y, sc, mase = truth(c, list(css()[c].e))
        with np.load(Path(gpu.dirs[zc[c]]) / 'pred_e.npz') as z:
            ez = score_pred(z['pred'].copy(), y, sc, mase)['nmse']
        se = stage_e('source', c)
        e = {m: statistics.fmean(se[m][s] for s in SEEDS) for m in MENU_IDS if m in se}
        f0 = delivered_phys(P_['source'], c, 'f0')
        if f0 in se:
            e['F0'] = statistics.fmean(se[f0][s] for s in SEEDS)
        rows[c] = {'E_zero_shot': ez, 'E': e, 'f0_delivered': f0, 'G_vs_zero_shot': {k: 100.0 * (ez - v) / ez for k, v in e.items()}}
    res = {'written_local': now(), 'units': 'G = 100 x (E(zero-shot) - E(fine-tuned plan)) / E(zero-shot), three-seed mean E; positive = beats zero-shot',
           'cases': rows, **zeroshot_summary(rows)}
    write(ROOT / 'source_zeroshot_check.json', res)
    print_zeroshot_check(res)
    return res


def zeroshot_summary(rows: dict) -> dict:
    keys = [k for k in list(MENU_IDS) + ['F0'] if all(k in r['G_vs_zero_shot'] for r in rows.values())]
    doms = [d for d in DOMAINS if any(c[:3] == d for c in rows)]
    by_dom = {k: {d: statistics.fmean(rows[c]['G_vs_zero_shot'][k] for c in rows if c[:3] == d) for d in doms} for k in keys}
    scripted = [k for k in keys if k != 'F0']
    best = {d: max(scripted, key=lambda k: by_dom[k][d]) for d in doms}
    oracle = {d: statistics.fmean(max(rows[c]['G_vs_zero_shot'][k] for k in scripted) for c in rows if c[:3] == d) for d in doms}
    return {'programs': keys, 'labels': {k: OS.MENU_LABEL.get(k, 'F0 (no-card Fast delivery)') for k in keys},
            'G_by_domain': by_dom, 'G_three_domain': {k: statistics.fmean(v.values()) for k, v in by_dom.items()},
            'wins_vs_zero_shot': {k: sum(r['G_vs_zero_shot'][k] > 0 for r in rows.values()) for k in keys}, 'n_cases': len(rows),
            'best_scripted_by_domain': {d: {'program': best[d], 'G': by_dom[best[d]][d]} for d in doms},
            'per_case_best_scripted_mean_by_domain': oracle}


def print_zeroshot_check(res: dict) -> None:
    doms = list(next(iter(res['G_by_domain'].values())).keys())
    print('SOURCE_ZEROSHOT_CHECK  pp of zero-shot (positive = fine-tuned plan beats zero-shot), %d Source cases' % res['n_cases'])
    print('  %-34s %8s  %s  %s' % ('plan', '3-domain', '  '.join('%7s' % d for d in doms), 'wins'))
    for k in sorted(res['programs'], key=lambda k: -res['G_three_domain'][k]):
        print('  %-34s %+8.2f  %s  %d/%d' % (res['labels'][k][:34], res['G_three_domain'][k], '  '.join('%+7.2f' % res['G_by_domain'][k][d] for d in doms),
                                           res['wins_vs_zero_shot'][k], res['n_cases']))
    print('  per-case best scripted (post hoc):', json.dumps({d: round(v, 2) for d, v in res['per_case_best_scripted_mean_by_domain'].items()}))


CTX_LENGTHS = [192, 336, 672, 1344, 2688]                  # 192 = the task window; then 2, 4, 8, 16 weeks of hourly history
CTX_NORMS = ('instance', 'tscale')


def predict_ctx(torch, model, X: np.ndarray, norm: str) -> np.ndarray:
    """(E, O, Lc) T-scaled context -> (E, O, 48) T-scaled forecasts. 'instance' = the official convention (each context normalized by its own mean /
    std, forecast mapped back); 'tscale' = the T-scaled context fed as is."""
    E, O, Lc = X.shape
    flat = torch.as_tensor(X.reshape(E * O, Lc), dtype=torch.float32, device='cuda')
    chunk = max(4, PRED_CHUNK * L // Lc)
    parts = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], chunk):
            x = flat[i:i + chunk]
            if norm == 'instance':
                mu, sd = context_stats(torch, x)
                parts.append(predict(torch, model, (x - mu) / sd) * sd + mu)
            else:
                parts.append(predict(torch, model, x))
    return torch.cat(parts).float().cpu().numpy().reshape(E, O, H)


def context_choice() -> dict:
    """Frozen from the Source screen before any Select / Test context is scored: per domain the (normalization, length) with the highest Source mean
    pp among the variants every Source case of the domain can run; exact ties -> shorter context, then 'instance'."""
    out = ROOT / 'context_choice_source.json'
    if out.exists():
        return read(out)
    src = read(ROOT / 'context_screen.json')
    cases = src['cases']
    rec = {'frozen_local': now(), 'rule': context_choice.__doc__.strip().replace('\n', ' '), 'domain': {}}
    for d in DOMAINS:
        dc = [c for c in cases if c[:3] == d]
        full = [k for k in src['summary'] if all(k in cases[c]['E'] for c in dc)]
        g = {k: statistics.fmean(100.0 * (cases[c]['E']['instance_%d' % L] - cases[c]['E'][k]) / cases[c]['E']['instance_%d' % L] for c in dc) for k in full}
        best = max(full, key=lambda k: (g[k], -int(k.split('_')[1]), k.startswith('instance')))
        rec['domain'][d] = {'choice': best, 'source_G': g[best], 'source_G_by_variant': g}
    write(out, rec)
    return rec


def context_screen() -> dict:
    """Inference-context preparation for the zero-shot Time-MoE (no training, no LLM): the 24 Source cases, E origins as everywhere, each forecast
    from the last Lc hours before its origin (history before T included; nothing at or after the origin) under two normalizations. The
    ('instance', 192) cell must reproduce the stored zero-shot E of the native root. Writes <ROOT>/context_screen.json."""
    P_ = paths()
    stage = os.environ.get('SEH_CTX_STAGE', 'source')
    plan = read(P_['plan']) if P_['plan'].exists() else freeze_plan()
    choice = context_choice() if stage != 'source' else None            # frozen from Source before this stage is scored
    cases = [c for d in DOMAINS for c in sorted(plan['cases'][stage][d])]
    for c in cases:
        prepare_common(c, css()[c])
    stored = {}
    zp = ROOT / 'source_zeroshot_check.json'
    if zp.exists() and stage == 'source':
        stored = {c: r['E_zero_shot'] for c, r in read(zp)['cases'].items()}
    torch = tsfm_setup()
    model = load_model(torch)
    rows = {}
    for c in cases:
        cs = css()[c]
        origins = list(cs.e)
        y, sc, mase = truth(c, origins)
        first = min(origins)
        res = {}
        for Lc in CTX_LENGTHS:
            if first - Lc < 0:
                continue
            sl = ec.load_case_slice(cs, first - Lc, max(origins))
            X = np.stack([(sl.rows(o - Lc, o).T - sc.mean[:, None]) / sc.scale[:, None] for o in origins], axis=1).astype('float32')
            for norm in CTX_NORMS:
                res['%s_%d' % (norm, Lc)] = score_pred(predict_ctx(torch, model, X, norm), y, sc, mase)['nmse']
        rows[c] = {'E': res, 'available_history_hours': first}
        print('CTX_CASE', c, json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
    base = 'instance_%d' % L
    keys = ['%s_%d' % (n, Lc) for Lc in CTX_LENGTHS for n in CTX_NORMS]
    summ = {}
    for k in keys:
        cs_k = [c for c in rows if k in rows[c]['E']]
        g = {c: 100.0 * (rows[c]['E'][base] - rows[c]['E'][k]) / rows[c]['E'][base] for c in cs_k}
        doms = {d: statistics.fmean(v for c, v in g.items() if c[:3] == d) for d in DOMAINS if any(c[:3] == d for c in g)}
        summ[k] = {'n_cases': len(cs_k), 'G_three_domain': statistics.fmean(doms.values()) if len(doms) == len(DOMAINS) else None, 'G_by_domain': doms,
                   'wins_vs_192': sum(v > 0 for v in g.values()), 'E_by_domain': {d: statistics.fmean(rows[c]['E'][k] for c in cs_k if c[:3] == d) for d in doms}}
    res = {'written_local': now(), 'stage': stage, 'baseline': base, 'units': 'G = 100 x (E(instance, 192) - E(variant)) / E(instance, 192), zero-shot, per case; positive = better',
           'baseline_equals_stored_zero_shot': (all(rows[c]['E'][base] == stored[c] for c in rows) if stored else None),
           'baseline_max_rel_diff_vs_stored': (max(abs(rows[c]['E'][base] - stored[c]) / stored[c] for c in rows) if stored else None),
           'summary': summ, 'cases': rows}
    if choice:
        per = {c: 100.0 * (rows[c]['E'][base] - rows[c]['E'][choice['domain'][c[:3]]['choice']]) / rows[c]['E'][base] for c in rows}
        dom = {d: statistics.fmean(v for c, v in per.items() if c[:3] == d) for d in DOMAINS}
        res['source_frozen_choice'] = {'choice': {d: choice['domain'][d]['choice'] for d in DOMAINS}, 'G_three_domain': statistics.fmean(dom.values()),
                                       'G_by_domain': dom, 'wins_vs_192': sum(v > 0 for v in per.values()), 'losses_vs_192': sum(v < 0 for v in per.values()),
                                       'max_harm': min(per.values()), 'by_case': per}
    write(ROOT / ('context_screen.json' if stage == 'source' else 'context_screen_%s.json' % stage), res)
    print('CONTEXT_SCREEN  stage %s, zero-shot, pp of the (instance, 192) zero-shot; positive = better; baseline vs stored zero-shot: equal %s, max rel diff %s'
          % (stage, res['baseline_equals_stored_zero_shot'], res['baseline_max_rel_diff_vs_stored']))
    if choice:
        fc = res['source_frozen_choice']
        print('  SOURCE-FROZEN CHOICE %s -> three-domain %+.2f by domain %s wins %d losses %d max harm %+.2f' % (json.dumps(fc['choice']), fc['G_three_domain'],
              json.dumps({d: round(v, 2) for d, v in fc['G_by_domain'].items()}), fc['wins_vs_192'], fc['losses_vs_192'], fc['max_harm']))
    for k in keys:
        r = summ[k]
        print('  %-16s n=%2d  three-domain %s  by domain %s  wins %d' % (k, r['n_cases'], ('%+6.2f' % r['G_three_domain']) if r['G_three_domain'] is not None else '   n/a',
                                                                       json.dumps({d: round(v, 2) for d, v in r['G_by_domain'].items()}), r['wins_vs_192']))
    return res


SCREEN_PROGRAMS = ['None', 'P_NoMixRecipe', 'C_censor', 'C_shock', 'C_conv']


def source_screen() -> None:
    """No-LLM Source screen of the current recipe (the head recipe): wiring, None-only calibration on the six Source cases (C_A + C_B), the 24 Source
    cases x SCREEN_PROGRAMS x 3 seeds behind the barrier, Source E scores and the zero-shot comparison. Writes <ROOT>/source_zeroshot_check.json."""
    P_ = paths()
    plan = freeze_plan()
    cases = [c for d in DOMAINS for c in sorted(plan['cases']['source'][d])]
    with DP.PackageLock(ROOT, 'run'):
        write(P_['status'], {'status': 'RUNNING', 'phase': 'prepare', 'recipe': RECIPE, 'pid': os.getpid(), 'updated': now()})
        cc = css()
        for c in cases:
            prepare_common(c, cc[c])
        led = SafeLedger(P_['ledger'], max_fit_attempts=700, max_llm_requests=1, max_llm_tokens=1, max_wall_s=CAPS['wall_s'], max_retries=CAPS['fit_retries'])
        gpu = GPUQueue(led)
        gpu.start()
        try:
            w = wiring(gpu)
            if w['status'] != 'PASS':
                raise RuntimeError('wiring FAIL: %s' % json.dumps({k: v for k, v in w.items() if k != 'fit_records'}))
            write(P_['status'], {'status': 'RUNNING', 'phase': 'calibration', 'recipe': RECIPE, 'pid': os.getpid(), 'updated': now()})
            cf = calibration(gpu)
            write(P_['status'], {'status': 'RUNNING', 'phase': 'source_fits', 'recipe': RECIPE, 'pid': os.getpid(), 'updated': now(),
                                 'consumer': {k: cf[k] for k in ('lr', 'steps', 'batch', 'objective')}, 'zero_shot': cf['zero_shot_anchor']['objective']})
            cells = {c: [cid for m in SCREEN_PROGRAMS for cid in gpu.submit('source', c, m)] for c in cases}
            gpu.wait([x for v in cells.values() for x in v])
            if not (P_['source'] / 'models_frozen.json').exists():
                write(P_['source'] / 'models_frozen.json', {'frozen_local': now(), 'cells': sum(len(v) for v in cells.values()), 'recipe': RECIPE})
            score_stage('source', gpu, cells)
            res = source_zeroshot_check(gpu, cases)
        except Exception as exc:  # noqa: BLE001
            write(P_['status'], {'status': 'STOPPED_ERROR', 'recipe': RECIPE, 'error': repr(exc)[:800], 'updated': now()})
            raise
        finally:
            gpu.stop = True
            with gpu.cv:
                gpu.cv.notify_all()
        write(P_['status'], {'status': 'SOURCE_SCREEN_COMPLETE', 'recipe': RECIPE, 'finished_local': now(), 'consumer': {k: cf[k] for k in ('lr', 'steps', 'batch', 'objective')},
                             'zero_shot': cf['zero_shot_anchor']['objective'], 'G_vs_zero_shot_three_domain': res['G_three_domain'], 'fits_ok': led.s['fits_ok']})
        print('SOURCE_SCREEN_COMPLETE', flush=True)


def blend_check(weights=(0.25, 0.5, 0.75, 0.9)) -> dict:
    """Post hoc, no fit, no LLM: E of w x zero-shot + (1 - w) x fine-tuned forecasts on the Source cases of this root, as pp of zero-shot. Every weight is
    reported and none is selected (a weight chosen on these cases would be in-sample)."""
    P_ = paths()
    if not (P_['source'] / 'models_frozen.json').exists():
        raise RuntimeError('E barrier: Source predictions not frozen')
    plan = read(P_['plan'])
    cases = [c for d in DOMAINS for c in sorted(plan['cases']['source'][d])]
    rows = {}
    for c in cases:
        y, sc, mase = truth(c, list(css()[c].e))
        with np.load(P_['fits'] / c / 'zeroshot' / 'pred_e.npz') as z:
            pz = z['pred'].astype(np.float64)
        ez = score_pred(pz, y, sc, mase)['nmse']
        g = {}
        for m in MENU_IDS:
            if not all((P_['fits'] / c / m / str(s) / 'pred_e.npz').exists() for s in SEEDS):
                continue
            pf = []
            for s in SEEDS:
                with np.load(P_['fits'] / c / m / str(s) / 'pred_e.npz') as z:
                    pf.append(z['pred'].astype(np.float64))
            for w in (0.0,) + tuple(weights):
                e = statistics.fmean(score_pred(w * pz + (1 - w) * f, y, sc, mase)['nmse'] for f in pf)
                g['%s@%.2f' % (m, w)] = 100.0 * (ez - e) / ez
        rows[c] = {'E_zero_shot': ez, 'G_vs_zero_shot': g}
    keys = [k for k in next(iter(rows.values()))['G_vs_zero_shot'] if all(k in r['G_vs_zero_shot'] for r in rows.values())]
    by_dom = {k: {d: statistics.fmean(rows[c]['G_vs_zero_shot'][k] for c in rows if c[:3] == d) for d in DOMAINS} for k in keys}
    res = {'written_local': now(), 'recipe': RECIPE, 'weights_on_zero_shot': [0.0] + list(weights), 'n_cases': len(rows),
           'note': 'weights are listed, not selected; w = 0 is the fine-tuned model alone, w = 1 would be zero-shot (G = 0)',
           'G_three_domain': {k: statistics.fmean(v.values()) for k, v in by_dom.items()}, 'G_by_domain': by_dom,
           'wins_vs_zero_shot': {k: sum(r['G_vs_zero_shot'][k] > 0 for r in rows.values()) for k in keys}, 'cases': rows}
    write(ROOT / 'source_blend_check.json', res)
    progs = sorted({k.split('@')[0] for k in keys}, key=lambda m: -res['G_three_domain']['%s@0.50' % m])
    print('SOURCE_BLEND_CHECK  pp of zero-shot (positive = the blend beats zero-shot), %d Source cases; columns = weight on zero-shot' % len(rows))
    print('  %-30s %s' % ('plan', '  '.join('%12s' % ('w=%.2f' % w) for w in res['weights_on_zero_shot'])))
    for m in progs:
        print('  %-30s %s' % (OS.MENU_LABEL.get(m, m)[:30], '  '.join('%+6.2f (%2d/%d)' % (res['G_three_domain']['%s@%.2f' % (m, w)], res['wins_vs_zero_shot']['%s@%.2f' % (m, w)],
                                                                                        len(rows)) for w in res['weights_on_zero_shot'])))
    return res


def amend_transport_caps(plan: dict) -> None:
    """Only the transport caps may differ from the frozen plan; each change is appended to <ROOT>/amendments/transport_caps.json."""
    old = plan.get('caps', {})
    diff = {k: [old.get(k), v] for k, v in CAPS.items() if old.get(k) != v}
    if not diff:
        return
    if any(k not in TRANSPORT_CAPS for k in diff):
        raise RuntimeError('only the transport caps may change after the plan is frozen: %s' % diff)
    p = ROOT / 'amendments' / 'transport_caps.json'
    rec = read(p) if p.exists() else {'rule': 'token / request / fit / wall caps unchanged; only HTTP attempts and the bounded second attempt of relay transients',
                                       'history': []}
    if not rec['history'] or rec['history'][-1]['caps'] != diff:
        rec['history'].append({'local': now(), 'caps': diff, 'http_in_flight': HTTP,
                               'reason': 'relay 500 / connection transients at 32 requests in flight exhausted the 40 second-attempt cap (amendment 2)'})
        write(p, rec)


def amend_transport_mode() -> None:
    """Amendment 3: the transport mode in force is appended to <ROOT>/amendments/transport_mode.json whenever it changes."""
    p = ROOT / 'amendments' / 'transport_mode.json'
    mode = {'stream': LLM_STREAM, 'retry_wait_s': RETRY_WAIT_S, 'resume_wait_s': RESUME_WAIT_S, 'stream_total_s': STREAM_TOTAL_S, 'http_in_flight': HTTP}
    rec = read(p) if p.exists() else {'rule': 'transport only; the one bounded retry per request, the caps, the ledger and every scientific setting are unchanged',
                                       'history': []}
    if not rec['history'] or rec['history'][-1]['mode'] != mode:
        rec['history'].append({'local': now(), 'mode': mode, 'reason': 'relay recommendation 2026-09-24: stream=true so a failure before the first byte is '
                                                                      'retried by the relay; waits let short upstream outages pass'})
        write(p, rec)


def slow_stage(led, client) -> dict:
    P_ = paths()
    summ_p = P_['slow'] / 'proposals.json'
    if summ_p.exists():
        return read(summ_p)
    cens, legal = {}, {}
    for scope in DOMAINS:
        cp = P_['slow'] / scope / 'census.json'
        if not cp.exists():
            write(cp, build_census_t(scope))
        cens[scope] = read(cp)
        legal[scope] = cens[scope]['legal_evidence_refs']
    sizes = {}
    for scope in DOMAINS:
        for k in OS.SLOTS:
            b = len(json.dumps([{'role': 'system', 'content': slow_system_t(scope)}, {'role': 'user', 'content': json.dumps(slow_payload_t(cens[scope], scope, k), ensure_ascii=False)}],
                               ensure_ascii=False).encode('utf-8'))
            sizes['%s_%d' % (scope, k)] = b
            if b > OS.SLOW_BYTE_LIMIT:
                raise RuntimeError('Slow payload %s slot %d is %d bytes > %d' % (scope, k, b, OS.SLOW_BYTE_LIMIT))
    write(P_['slow'] / 'payload_sizes.json', sizes)
    results, lock = {}, threading.Lock()

    def one(scope, k):
        out_p = P_['slow'] / scope / ('proposal_%d.json' % k)
        if out_p.exists():
            r = read(out_p)
        else:
            call = lambda p: client.call('slow', '%s_%d' % (scope, k), p, slow_system_t(scope), max_tokens=OS.MAX_OUTPUT_TOKENS, meta={'scope': scope, 'slot': k},
                                         request_timeout=OS.SLOW_TIMEOUT_S)
            res = OS.propose_card(slow_payload_t(cens[scope], scope, k), call, scope=scope, slot=k, legal_refs=legal[scope])
            r = {'scope': scope, 'slot': k, 'status': res['status'], 'attempts': res['attempts'], 'receipts': res['receipts'],
                 'skill': res['skill'].to_json() if res.get('skill') else None, 'meta': res.get('meta', {}), 'raw': res.get('raw'), 'written_local': now()}
            if r['status'] != 'PROPOSE_CALL_FAILED':
                write_new(out_p, r)
        with lock:
            results[(scope, k)] = r
        print('SLOW', scope, k, r['status'], flush=True)

    threads = [threading.Thread(target=one, args=(s, k), daemon=True) for s in DOMAINS for k in OS.SLOTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(r['status'] == 'PROPOSE_CALL_FAILED' for r in results.values()):
        raise RuntimeError('a Slow call failed technically; resume after the fault is cleared')
    summ = {'written_local': now(), 'scopes': {}}
    for scope in DOMAINS:
        rows = []
        for k in OS.SLOTS:
            r = results[(scope, k)]
            alias = None
            if r['status'] == 'PROPOSED' and k == 2 and results[(scope, 1)]['status'] == 'PROPOSED' and results[(scope, 1)]['skill']['rendered_body'] == r['skill']['rendered_body']:
                alias = 1
            if r['status'] == 'NO_CARD' and k == 2 and results[(scope, 1)]['status'] == 'NO_CARD':
                alias = 1
            rows.append({'slot': k, 'arm': 'd%d' % k, 'status': r['status'], 'alias_of_slot': alias, 'body_chars': len(r['skill']['rendered_body']) if r['skill'] else 0})
        summ['scopes'][scope] = rows
    write_new(summ_p, summ)
    return summ


def card_t(scope: str, slot: int):
    r = read(paths()['slow'] / scope / ('proposal_%d.json' % slot))
    return dsk.skill_from_json(r['skill']) if r['status'] == 'PROPOSED' else None


def naive_payload_t(domain: str) -> dict:
    plan = read(paths()['plan'])
    cases = sorted(plan['cases']['source'][domain])
    cc = css()
    obs = [{'case_ref': c, 'observations': OS._obs_summary(paths()['common'], cc[c])} for c in cases]
    return {'scope': domain, 'deployment': {'tools': OS.CONTRACTS_ZF, 'actions': OS.ES.action_table(), 'fields': OS.OBS_FIELDS_ZF, 'field_definitions': OS.FIELD_DEFINITIONS,
                                            'consumer': consumer_t(), 'limits': {'requests': OS.LIMITS.max_calls, 'tool_calls': OS.LIMITS.max_tools,
                                                                                 'constructed_materials': OS.LIMITS.max_materials, 'commit': 1},
                                            'visible_domain_identity': 'the case card shows a neutral domain id; this card is loaded by it'},
            'development_observations': {'note': 'T-only observation quantiles of eight development cases of this domain (16 entities each); no downstream result exists here',
                                         'cases': obs},
            'legal_evidence_refs': ['source/%s/observations' % c for c in cases], 'body_limit_characters': OS.CARD_BODY_LIMIT}


def naive_cards_t(led, client) -> dict:
    P_ = paths()
    summ_p = P_['naive'] / 'naive_cards.json'
    if summ_p.exists():
        return read(summ_p)
    out, lock = {}, threading.Lock()

    def one(d):
        p_ = P_['naive'] / ('%s.json' % d)
        if p_.exists():
            r = read(p_)
        else:
            payload = naive_payload_t(d)
            low = json.dumps(payload, ensure_ascii=False)
            for bad in ('G_vs_None', 'fast_trajectory', 'scripted_records', 'fast_outcome', 'e_by_seed', 'normalized_mse', 'nmse'):
                if bad in low:
                    raise PermissionError('naive payload carries downstream evidence: %s' % bad)
            legal = payload['legal_evidence_refs']
            attempts, receipts, res = [], [], None
            for i in range(2):
                try:
                    text, rec = client.call('slow_naive', 'naive_%s' % d, payload, OS.naive_system(), max_tokens=OS.MAX_OUTPUT_TOKENS, meta={'scope': d},
                                            request_timeout=OS.SLOW_TIMEOUT_S)
                    receipts.append(rec)
                except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted, OS.UnknownUsageBlock) as exc:
                    attempts.append({'attempt': i, 'fault_kind': type(exc).__name__})
                    res = {'status': 'PROPOSE_CALL_FAILED'}
                    break
                try:
                    raw = json.loads(text) if isinstance(text, str) else None
                    pr = OS.parse_naive(raw, domain=d, legal_refs=legal)
                    attempts.append({'attempt': i, 'ok': True})
                    res = {'status': 'NO_CARD' if pr['decision'] == 'KEEP' else 'PROPOSED', 'skill': pr['skill'].to_json() if pr['skill'] else None, 'meta': pr['meta'], 'raw': raw}
                    break
                except ValueError as exc:
                    attempts.append({'attempt': i, 'error': str(exc)[:400]})
                    if i == 0:
                        payload = {**payload, 'correction': {'previous_output_excerpt': (text or '')[:4000], 'error': attempts[-1]['error'],
                                                             'instruction': 'Correct this format / contract error only; KEEP is allowed; do not seek a different outcome.'}}
            if res is None:
                res = {'status': 'PROPOSE_PARSE_OR_VALIDATION_FAILED'}
            r = {'domain': d, **res, 'attempts': attempts, 'receipts': receipts, 'written_local': now(), 'payload_bytes': len(json.dumps(payload, ensure_ascii=False).encode('utf-8'))}
            if r['status'] != 'PROPOSE_CALL_FAILED':
                write_new(p_, r)
        with lock:
            out[d] = r
        print('NAIVE_CARD', d, r['status'], flush=True)

    ts = [threading.Thread(target=one, args=(d,), daemon=True) for d in DOMAINS]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if any(out[d]['status'] == 'PROPOSE_CALL_FAILED' for d in DOMAINS):
        raise RuntimeError('a naive card call failed technically; resume after the fault is cleared')
    summ = {'written_local': now(), 'cards': {d: {'status': out[d]['status'], 'attempts': len(out[d]['attempts']),
                                                  'body_chars': len(out[d]['skill']['rendered_body']) if out[d].get('skill') else 0} for d in DOMAINS}}
    write_new(summ_p, summ)
    return summ


# ============================================================================= knowledge and jobs
def knowledge_select(case: str, arm: str) -> br.Knowledge:
    return OS.knowledge_of_card(card_t(case[:3], int(arm[1])))


def knowledge_test(case: str, arm: str) -> br.Knowledge:
    d = case[:3]
    if arm == 'f0':
        return OS.no_card()
    if arm == 'f_mlp':
        sk = read(PARENT / 'freeze' / 'cards_frozen.json')['domain'][d]['skill']
        return OS.knowledge_of_card(dsk.skill_from_json(sk) if sk else None)
    if arm == 'f_naive':
        r = read(paths()['naive'] / ('%s.json' % d))
        return OS.knowledge_of_card(dsk.skill_from_json(r['skill']) if r.get('skill') else None)
    if arm == 'f_tsfm':
        sk = read(paths()['freeze'] / 'cards_frozen.json')['domain'][d]['skill']
        return OS.knowledge_of_card(dsk.skill_from_json(sk) if sk else None)
    raise ValueError(arm)


def interleave(by: dict) -> list:
    out = []
    while any(by.values()):
        for d in DOMAINS:
            if by[d]:
                out.append(by[d].pop(0))
    return out


def source_jobs(plan) -> list:
    return interleave({d: [(c, 'f0') for c in sorted(plan['cases']['source'][d])] for d in DOMAINS})


def select_jobs(plan) -> list:
    summ = read(paths()['slow'] / 'proposals.json')
    by = {}
    for d in DOMAINS:
        arms = [r['arm'] for r in summ['scopes'][d] if r['status'] in ('PROPOSED', 'NO_CARD') and r['alias_of_slot'] is None]
        lst = []
        for i, c in enumerate(sorted(plan['cases']['select'][d])):
            k = i % len(arms) if arms else 0
            lst += [(c, a) for a in arms[k:] + arms[:k]]
        by[d] = lst
    return interleave(by)


def test_jobs(plan, arms) -> list:
    by = {}
    for d in DOMAINS:
        lst = []
        for i, c in enumerate(sorted(plan['cases']['test'][d])):
            k = i % len(arms)
            lst += [(c, a) for a in (arms[k:] + arms[:k])]
        by[d] = lst
    return interleave(by)


def selection(plan) -> dict:
    P_ = paths()
    out_p = P_['freeze'] / 'cards_frozen.json'
    if out_p.exists():
        return read(out_p)
    led = read(P_['ledger'])
    jobs = select_jobs(plan)
    J, tok = {}, {}
    for case, arm in jobs:
        e = stage_e('select', case)
        ratio = statistics.fmean(e[delivered_phys(P_['select'], case, arm)].values()) / statistics.fmean(e['None'].values())
        J.setdefault(arm, {}).setdefault(case[:3], []).append(ratio)
        tok.setdefault(arm, {}).setdefault(case[:3], []).append(OS._arm_tokens(led, 'select', '%s__%s' % (case, arm)))
    Jd = {a: {d: statistics.fmean(v) for d, v in dd.items()} for a, dd in J.items()}
    rec = {'frozen_local': now(), 'rule': plan['selection'], 'J_by_arm_domain': Jd, 'tokens_by_arm_domain': {a: {d: statistics.fmean(v) for d, v in dd.items()} for a, dd in tok.items()},
           'domain': {}, 'incomplete': {'%s__%s' % j: True for j in jobs if commit_of(P_['select'], *j) is None}, 'proposals': read(P_['slow'] / 'proposals.json')}
    for d in DOMAINS:
        cands = [a for a in SELECT_ARMS if a in Jd and d in Jd[a]]
        if not cands:
            rec['domain'][d] = {'arm': None, 'slot': None, 'skill': None, 'note': 'no usable domain card; f_tsfm runs without a card'}
            continue
        best = min(cands, key=lambda a: (Jd[a][d], statistics.fmean(tok[a][d]), SELECT_ARMS.index(a)))
        sk = card_t(d, int(best[1]))
        rec['domain'][d] = {'arm': best, 'slot': int(best[1]), 'J': {a: Jd[a][d] for a in cands},
                            'tie_break': 'tokens' if len(cands) == 2 and Jd[cands[0]][d] == Jd[cands[1]][d] else None, 'skill': sk.to_json() if sk else None}
    write_new(out_p, rec)
    return rec


def fixed_ref_phys(case: str, program: str, led) -> str:
    if program in ('None', 'FixedMixup', 'P_AmpResample', 'P_NoMixRecipe'):
        return program
    ctx = ec.open_case(css()[case], paths()['common'], 'material')
    comp = ec.compile_plan_full(ec.uniform_policy(OS.MENU_STEPS[program], 'fixed reference %s' % OS.MENU_LABEL[program]), ctx.overview()['entities'])
    return OS.ensure_material_zf(paths()['common'], paths()['split'], case, comp['assignment'], ledger=led)['material_id']


# ============================================================================= wiring and calibration
def wiring(gpu: GPUQueue) -> dict:
    """Real checks before any paid request: the unroll equals the official generate; NoMix twice (second job of a batch after another fit)
    gives bitwise identical predictions; NoMix differs from None (the child path trains); with 2+ GPUs the same fit on the second GPU is
    bitwise identical to the first. 10 updates, batch 4, lr 1e-5, first seed, one Source case."""
    P_ = paths()
    out_p = P_['wiring'] / 'wiring_result.json'
    if out_p.exists() and read(out_p)['status'] == 'PASS':
        return read(out_p)
    case = calib_cases()[0]
    kw = dict(lr=1e-5, steps=10, batch=4, seeds=(SEEDS[0],))
    g = gpu.submit_special('wiring', case, 'generate_check', 'wiring_generate_check')
    a = gpu.submit('wiring', case, 'None', tag='wiring_none', **kw)
    b = gpu.submit('wiring', case, 'P_NoMixRecipe', tag='wiring_nomix_a', **kw)
    c = gpu.submit('wiring', case, 'P_NoMixRecipe', tag='wiring_nomix_b', **kw)
    gpu.wait([g] + a + b + c)
    ld = lambda cid: np.load(Path(gpu.dirs[cid]) / 'pred_e.npz')['pred']
    gen = read(Path(gpu.dirs[g]) / 'fit.json')
    res = {'written_local': now(), 'case': case, 'generate_equal_unroll': gen['bitwise_equal'], 'generate_max_abs': gen['max_abs'],
           'nomix_repeat_bitwise_equal': bool(np.array_equal(ld(b[0]), ld(c[0]))), 'nomix_differs_from_none': bool(not np.array_equal(ld(a[0]), ld(b[0]))),
           'fit_records': {cid: {k: read(Path(gpu.dirs[cid]) / 'fit.json').get(k) for k in ('train_seconds', 'seconds', 'peak_cuda_mb', 'peak_rss_mb_process', 'device', 'visible_device')}
                           for cid in a + b + c}}
    if len(GPUS) > 1:                     # the same fit forced onto the second GPU through a dedicated worker process
        rel = P_['fits'] / case / 'P_NoMixRecipe__wiring_nomix_gpu2' / str(SEEDS[0])
        job = {**read(Path(gpu.dirs[b[0]]) / 'fit.json')['job'], 'cid': 'wiring_nomix_gpu2', 'output': str(rel.relative_to(REPO)).replace('\\', '/')}
        spec = P_['wiring'] / 'gpu2_batch.json'
        write(spec, {'jobs': [job]})
        if not (rel / 'fit.json').exists():
            gpu.led.reserve_fit('wiring_nomix_gpu2')
            rc = run_gpu_process(['--fit-batch', spec], P_['logs'] / 'wiring_gpu2.log', 1800, device=GPUS[1])
            gpu.led.finish_fit('wiring_nomix_gpu2', (rel / 'fit.json').exists(), 0.0, '' if rc == 0 else 'rc=%s' % rc)
        res['second_gpu_bitwise_equal'] = bool((rel / 'pred_e.npz').exists() and np.array_equal(np.load(rel / 'pred_e.npz')['pred'], ld(b[0])))
    need = ['generate_equal_unroll', 'nomix_repeat_bitwise_equal', 'nomix_differs_from_none'] + (['second_gpu_bitwise_equal'] if len(GPUS) > 1 else [])
    res['status'] = 'PASS' if all(res[k] for k in need) else 'FAIL'
    write(out_p, res)
    print('WIRING', res['status'], json.dumps({k: res[k] for k in need}), flush=True)
    return res


def calibration(gpu: GPUQueue) -> dict:
    """None only, first seed, six Source cases; C_A + C_B validation at the checkpoints (Source rows may be read); never Test."""
    P_ = paths()
    if P_['consumer'].exists():
        return read(P_['consumer'])
    cids = {}
    for c in calib_cases():
        for lr in CALIB['lrs']:
            for b in CALIB['batches']:
                cids[(c, lr, b)] = gpu.submit('calibration', c, 'None', lr=lr, steps=max(CALIB['marks']), batch=b, marks=CALIB['marks'],
                                              seeds=(CALIB['seed'],), tag='calib_lr%g_b%d' % (lr, b))[0]
        gpu.submit_special('calibration', c, 'zeroshot', 'calib_zeroshot')
    gpu.wait(list(cids.values()) + ['%s__calib_zeroshot' % c for c in calib_cases()])
    cc = css()
    val = {}
    for c in calib_cases():
        cs = cc[c]
        vo = list(cs.c_a) + list(cs.c_b)
        y, sc, mase = truth(c, vo)
        val[c] = {'y': y, 'sc': sc, 'mase': mase}
        with np.load(Path(gpu.dirs['%s__calib_zeroshot' % c]) / 'pred_val_0.npz') as z:
            val[c]['zeroshot'] = score_pred(z['pred'], y, sc, mase)['nmse']
    options = []
    for lr in CALIB['lrs']:
        for b in CALIB['batches']:
            for m in CALIB['marks']:
                dom = {}
                for d in [d for d in DOMAINS if any(x.startswith(d) for x in calib_cases())]:
                    vs = []
                    for c in [x for x in calib_cases() if x.startswith(d)]:
                        with np.load(Path(gpu.dirs[cids[(c, lr, b)]]) / ('pred_val_%d.npz' % m)) as z:
                            vs.append(score_pred(z['pred'], val[c]['y'], val[c]['sc'], val[c]['mase'])['nmse'])
                    dom[d] = statistics.fmean(vs)
                options.append({'lr': lr, 'batch': b, 'steps': m, 'objective': statistics.fmean(dom.values()), 'domains': dom})
    chosen = min(options, key=lambda r: (r['objective'], r['steps'], r['batch'], r['lr']))
    rec = {'lr': chosen['lr'], 'steps': chosen['steps'], 'batch': chosen['batch'], 'objective': chosen['objective'], 'domains': chosen['domains'],
           'all_options': options, 'grid': CALIB, 'cases': calib_cases(), 'frozen_local': now(),
           'note': 'None only, Source C_A + C_B only, never Test; zero-shot E predictions of the six cases are kept as anchors, not candidates',
           'fit_seconds': {cid: read(Path(gpu.dirs[cid]) / 'fit.json').get('train_seconds') for cid in cids.values()},
           'zero_shot_anchor': {'objective': statistics.fmean(statistics.fmean(val[c]['zeroshot'] for c in calib_cases() if c.startswith(d))
                                                              for d in DOMAINS if any(x.startswith(d) for x in calib_cases())),
                                'by_case': {c: val[c]['zeroshot'] for c in calib_cases()}}}
    rec['consumer_description_shown_to_agents'] = consumer_t({'lr': rec['lr'], 'steps': rec['steps'], 'batch': rec['batch']})
    write_new(P_['consumer'], rec)
    print('CALIBRATED', json.dumps({k: rec[k] for k in ('lr', 'steps', 'batch', 'objective')}), flush=True)
    return rec


# ============================================================================= controller
class Status:
    def __init__(self):
        self.lock, self.phase = threading.Lock(), {}

    def set(self, part: str, value: str):
        with self.lock:
            self.phase[part] = {'phase': value, 'local': now()}
        print('PHASE', part, value, flush=True)


def run() -> None:
    P_ = paths()
    plan = freeze_plan()
    amend_transport_caps(plan)
    amend_transport_mode()
    DP.set_pools(CPU_POOL, HTTP)
    rt.FIT_RETRY = True
    with DP.PackageLock(ROOT, 'run'):
        write(P_['status'], {'status': 'RUNNING', 'phase': 'prepare', 'pid': os.getpid(), 'updated': now()})
        prepare_all(plan['cases'])
        led = SafeLedger(P_['ledger'], max_fit_attempts=CAPS['fits'], max_llm_requests=CAPS['logical_requests'], max_llm_tokens=CAPS['tokens'],
                         max_wall_s=CAPS['wall_s'], max_retries=CAPS['fit_retries'])
        gpu = GPUQueue(led)
        gpu.start()
        st = Status()
        errors = []
        stop_hb = threading.Event()

        def heartbeat():
            while not stop_hb.wait(60):
                try:
                    write(P_['status'], {'status': 'RUNNING', 'pid': os.getpid(), 'updated': now(), 'phases': st.phase, 'gpu': gpu.snapshot(),
                                         'ledger': {k: led.s.get(k) for k in ('llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown',
                                                                           'unknown_usage_accepted', 'extra_transport_attempts', 'fit_attempts', 'fits_ok', 'fits_failed',
                                                                           'retries_used')},
                                         'free_gb': round(OS.free_gb(), 2), 'errors': errors[-5:]})
                except Exception:  # noqa: BLE001
                    pass
        threading.Thread(target=heartbeat, daemon=True).start()
        try:
            st.set('wiring', 'running')
            w = wiring(gpu)
            if w['status'] != 'PASS':
                raise RuntimeError('wiring FAIL: %s' % json.dumps({k: v for k, v in w.items() if k != 'fit_records'}))
            st.set('calibration', 'running')
            calibration(gpu)
            st.set('calibration', 'done')
        except Exception as exc:  # noqa: BLE001
            stop_hb.set()
            write(P_['status'], {'status': 'STOPPED_ERROR', 'errors': [repr(exc)[:800]], 'pid': os.getpid(), 'updated': now(), 'phases': st.phase})
            raise
        if 'paid_clock_start' not in led.s:
            with led.lock:
                led.s['paid_clock_start'], led.s['paid_clock_start_local'] = time.time(), now()
                led._save()
        for c in [c for d in DOMAINS for c in sorted(plan['cases']['source'][d])]:
            for m in MENU_IDS:
                gpu.submit('source', c, m)
        test_refs_ready = threading.Event()

        def on_test_commit(case, arm, cm):
            gpu.submit('test', case, cm['physical_material'] if cm else 'None')

        def source_pipeline():
            try:
                st.set('source', 'fast')
                run_fast_stage('source', source_jobs(plan), lambda c, a: OS.no_card(), led=led, client=make_client(led, 'source'),
                               on_commit=lambda c, a, cm: gpu.submit('source', c, cm['physical_material'] if cm else 'None'), sdir=P_['source'])
                st.set('source', 'fits')
                scells = {c: [cid for p in source_cells(c) for cid in gpu.submit('source', c, p)] for c in [j[0] for j in source_jobs(plan)]}
                gpu.wait([x for v in scells.values() for x in v])
                if not (P_['source'] / 'models_frozen.json').exists():
                    write(P_['source'] / 'models_frozen.json', {'frozen_local': now(), 'cells': sum(len(v) for v in scells.values())})
                st.set('source', 'score')
                score_stage('source', gpu, scells)
                fx = fixed_source_t()
                test_refs_ready.set()
                print('FIXED_SOURCE_T', json.dumps({d: v['label'] for d, v in fx['domains'].items()}), flush=True)
                if STOP_AFTER == 'source':
                    st.set('source', 'zeroshot_check')
                    source_zeroshot_check(gpu, [c for d in DOMAINS for c in sorted(plan['cases']['source'][d])])
                    st.set('source', 'done (SEH_STOP_AFTER=source)')
                    return
                st.set('slow', 'running')
                summ = slow_stage(led, make_client(led, 'slow'))
                print('SLOW_DONE', json.dumps({s: [r['status'] for r in rows] for s, rows in summ['scopes'].items()}), flush=True)
                st.set('select', 'fast')
                sj = select_jobs(plan)
                sel_cases = sorted({c for c, _ in sj})
                for c in sel_cases:
                    gpu.submit('select', c, 'None')
                run_fast_stage('select', sj, knowledge_select, led=led, client=make_client(led, 'select'),
                               on_commit=lambda c, a, cm: gpu.submit('select', c, cm['physical_material'] if cm else 'None'), sdir=P_['select'])
                st.set('select', 'fits')
                vcells = {c: sorted({cid for a in [a for cc, a in sj if cc == c] for cid in gpu.submit('select', c, delivered_phys(P_['select'], c, a))}
                                    | set(gpu.submit('select', c, 'None'))) for c in sel_cases}
                gpu.wait([x for v in vcells.values() for x in v])
                if not (P_['select'] / 'models_frozen.json').exists():
                    write(P_['select'] / 'models_frozen.json', {'frozen_local': now(), 'cells': sum(len(v) for v in vcells.values())})
                st.set('select', 'score')
                score_stage('select', gpu, vcells)
                fr = selection(plan)
                print('SELECTED', json.dumps({d: fr['domain'][d]['arm'] for d in DOMAINS}), flush=True)
                st.set('test_tsfm', 'fast')
                run_fast_stage('test_tsfm', test_jobs(plan, ('f_tsfm',)), knowledge_test, led=led, client=make_client(led, 'test'), on_commit=on_test_commit, sdir=P_['test'])
                st.set('test_tsfm', 'done')
            except Exception as exc:  # noqa: BLE001
                errors.append(('source_pipeline', repr(exc)[:500]))
                st.set('source', 'STOPPED: %s' % repr(exc)[:200])
                traceback.print_exc()

        def test_early_pipeline():
            if STOP_AFTER == 'source':
                return
            try:
                st.set('naive', 'cards')
                ns = naive_cards_t(led, make_client(led, 'naive'))
                print('NAIVE_DONE', json.dumps({d: v['status'] for d, v in ns['cards'].items()}), flush=True)
                st.set('test_early', 'fast')
                for c in sorted({c for c, _ in test_jobs(plan, EARLY_ARMS)}):
                    for ph in ('None', 'P_NoMixRecipe'):
                        gpu.submit('test', c, ph)
                    gpu.submit_special('test', c, 'zeroshot', 'zeroshot')
                run_fast_stage('test_early', test_jobs(plan, EARLY_ARMS), knowledge_test, led=led, client=make_client(led, 'test'), on_commit=on_test_commit,
                               sdir=P_['test'])
                st.set('test_early', 'done')
            except Exception as exc:  # noqa: BLE001
                errors.append(('test_early_pipeline', repr(exc)[:500]))
                st.set('test_early', 'STOPPED: %s' % repr(exc)[:200])
                traceback.print_exc()

        def fixed_ref_pipeline():
            if STOP_AFTER == 'source':
                return
            try:
                while not test_refs_ready.wait(30):
                    if errors:
                        return
                fx = read(P_['freeze'] / 'fixed_source_t.json')
                refs = {}
                for c in [c for d in DOMAINS for c in sorted(plan['cases']['test'][d])]:
                    refs[c] = fixed_ref_phys(c, fx['domains'][c[:3]]['program'], led)
                    gpu.submit('test', c, refs[c])
                if not (P_['test'] / 'fixed_ref_phys.json').exists():
                    write_new(P_['test'] / 'fixed_ref_phys.json', {'frozen_local': now(), 'program_by_domain': {d: v['program'] for d, v in fx['domains'].items()}, 'phys': refs})
            except Exception as exc:  # noqa: BLE001
                errors.append(('fixed_ref_pipeline', repr(exc)[:500]))
                traceback.print_exc()

        ths = [threading.Thread(target=f, daemon=True, name=f.__name__) for f in (source_pipeline, test_early_pipeline, fixed_ref_pipeline)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        if errors:
            stop_hb.set()
            write(P_['status'], {'status': 'STOPPED_ERROR', 'errors': errors, 'pid': os.getpid(), 'updated': now(), 'phases': st.phase, 'gpu': gpu.snapshot()})
            gpu.stop = True
            raise RuntimeError('pipeline errors: %s' % errors)
        if STOP_AFTER == 'source':
            stop_hb.set()
            gpu.stop = True
            with gpu.cv:
                gpu.cv.notify_all()
            write(P_['status'], {'status': 'STOPPED_AT_SOURCE', 'updated': now(), 'pid': os.getpid(), 'phases': st.phase, 'gpu': gpu.snapshot(),
                                 'ledger': {k: led.s.get(k) for k in ('llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown',
                                                                   'extra_transport_attempts', 'fit_attempts', 'fits_ok', 'fits_failed')},
                                 'next': 'read source_zeroshot_check.json; resume the full run with the same command without SEH_STOP_AFTER'})
            print('STOPPED_AT_SOURCE', flush=True)
            return
        st.set('test', 'fits')
        allc = {}
        for c in [c for d in DOMAINS for c in sorted(plan['cases']['test'][d])]:
            ids = set()
            for a in TEST_ARMS:
                ids.update(gpu.submit('test', c, delivered_phys(P_['test'], c, a)))
            for ph in ('None', 'P_NoMixRecipe', read(P_['test'] / 'fixed_ref_phys.json')['phys'][c]):
                ids.update(gpu.submit('test', c, ph))
            ids.add(gpu.submit_special('test', c, 'zeroshot', 'zeroshot'))
            allc[c] = sorted(ids)
        gpu.wait([x for v in allc.values() for x in v])
        if not (P_['test'] / 'models_frozen.json').exists():
            write(P_['test'] / 'models_frozen.json', {'frozen_local': now(), 'cells': sum(len(v) for v in allc.values()), 'commits_frozen': 160})
        st.set('test', 'score')
        score_stage('test', gpu, allc)
        with led.lock:
            led.s['paid_clock_stop'], led.s['paid_clock_stop_local'] = time.time(), now()
            led._save()
        gpu.stop = True
        with gpu.cv:
            gpu.cv.notify_all()
        stop_hb.set()
        res = readout()
        write(P_['status'], {'status': 'COMPLETE', 'finished_local': now(), 'pid': os.getpid(), 'phases': st.phase, 'gpu': gpu.snapshot(),
                             'main': {k: round(v['three_domain'], 3) for k, v in res['main'].items()}})
        print('COMPLETE', flush=True)


# ============================================================================= readout
PAIRS = (('tsfm_minus_mlp', 'f_tsfm', 'f_mlp'), ('tsfm_minus_f0', 'f_tsfm', 'f0'), ('tsfm_minus_naive', 'f_tsfm', 'f_naive'),
         ('mlp_minus_f0', 'f_mlp', 'f0'), ('naive_minus_f0', 'f_naive', 'f0'))
REFS = ('None', 'NoMix', 'Fixed_source_T', 'ZeroShot')


def readout() -> dict:
    P_ = paths()
    plan = read(P_['plan'])
    led = read(P_['ledger'])
    fixed = read(P_['test'] / 'fixed_ref_phys.json')
    fr = read(P_['freeze'] / 'cards_frozen.json')
    per_case = {}
    for case in [c for d in DOMAINS for c in sorted(plan['cases']['test'][d])]:
        d = case[:3]
        e = stage_e('test', case)
        em = {p: statistics.fmean(v.values()) for p, v in e.items()}
        dp = {a: delivered_phys(P_['test'], case, a) for a in TEST_ARMS}
        rp = {'None': 'None', 'NoMix': 'P_NoMixRecipe', 'Fixed_source_T': fixed['phys'][case], 'ZeroShot': 'zeroshot'}
        den = em['None']
        rec = {'domain': d, 'period': OS._period(case), 'group': OS._group(case), 'delivered_phys': dp, 'ref_phys': rp,
               'delivered': {a: (commit_of(P_['test'], case, a) or {}).get('label', 'INCOMPLETE') for a in TEST_ARMS},
               'E': {a: em[dp[a]] for a in TEST_ARMS}, 'E_refs': {k: em[p] for k, p in rp.items()},
               'G_vs_None': {a: OS.G2(em[dp[a]], den, den) for a in TEST_ARMS}, 'G_refs_vs_None': {k: OS.G2(em[p], den, den) for k, p in rp.items()},
               'tokens': {a: OS._arm_tokens(led, 'test', '%s__%s' % (case, a)) for a in TEST_ARMS}}
        for key, a, b in PAIRS:
            rec[key] = OS.G2(em[dp[a]], em[dp[b]], den)
            rec['seed_' + key] = [100 * (e[dp[b]][s] - e[dp[a]][s]) / den for s in SEEDS]
        for a in TEST_ARMS:
            for k, p in rp.items():
                rec['%s_minus_%s' % (a, k)] = OS.G2(em[dp[a]], em[p], den)
        per_case[case] = rec

    def agg(key, cases=None):
        cs = cases or list(per_case)
        dm = {d: statistics.fmean(per_case[c][key] for c in cs if per_case[c]['domain'] == d) for d in DOMAINS if any(per_case[c]['domain'] == d for c in cs)}
        return {'three_domain': statistics.fmean(dm.values()), 'by_domain': dm}

    def wtl(key, a, bref, ref=False):
        w = t = l = 0
        for r in per_case.values():
            same = r['delivered_phys'][a] == (r['ref_phys'][bref] if ref else r['delivered_phys'][bref])
            if same or abs(r[key]) <= 1e-9:
                t += 1
            elif r[key] > 0:
                w += 1
            else:
                l += 1
        return [w, t, l]

    main = {}
    for key, a, b in PAIRS:
        main[key] = {**agg(key), 'wins_same_losses': wtl(key, a, b), 'max_harm': min(r[key] for r in per_case.values()), 'max_gain': max(r[key] for r in per_case.values()),
                     'cluster_bootstrap_95': OS.boot_ci(per_case, key),
                     'leave_one_period_out': {p: agg(key, [c for c in per_case if per_case[c]['period'] != p])['three_domain'] for p in ('Q1', 'Q2', 'Q3', 'Q4')},
                     'seed_level': [statistics.fmean(statistics.fmean(per_case[c]['seed_' + key][i] for c in per_case if per_case[c]['domain'] == d) for d in DOMAINS) for i in range(3)]}
    vs_refs = {}
    for a in TEST_ARMS:
        for k in REFS:
            key = '%s_minus_%s' % (a, k)
            vs_refs[key] = {**agg(key), 'wins_same_losses': wtl(key, a, k, ref=True), 'cluster_bootstrap_95': OS.boot_ci(per_case, key)}
    arms_vs_none = {a: {'three_domain': statistics.fmean(statistics.fmean(r['G_vs_None'][a] for r in per_case.values() if r['domain'] == d) for d in DOMAINS),
                        'by_domain': {d: statistics.fmean(r['G_vs_None'][a] for r in per_case.values() if r['domain'] == d) for d in DOMAINS}} for a in TEST_ARMS}
    refs_vs_none = {k: {'three_domain': statistics.fmean(statistics.fmean(r['G_refs_vs_None'][k] for r in per_case.values() if r['domain'] == d) for d in DOMAINS),
                        'by_domain': {d: statistics.fmean(r['G_refs_vs_None'][k] for r in per_case.values() if r['domain'] == d) for d in DOMAINS}} for k in REFS}
    nmse = {a: statistics.fmean(statistics.fmean(r['E'][a] for r in per_case.values() if r['domain'] == d) for d in DOMAINS) for a in TEST_ARMS}
    nmse.update({k: statistics.fmean(statistics.fmean(r['E_refs'][k] for r in per_case.values() if r['domain'] == d) for d in DOMAINS) for k in REFS})
    deliv = {a: {d: dict(OS.collections_counter(r['delivered'][a] for r in per_case.values() if r['domain'] == d)) for d in DOMAINS} for a in TEST_ARMS}
    beh = {a: OS.test_behaviour_stage(ROOT, P_['test'], [(c, a) for c in per_case]) for a in TEST_ARMS}
    res = {'package': PACKAGE, 'identity': IDENTITY, 'written_local': now(), 'consumer': consumer_frozen(), 'cards': {d: fr['domain'][d]['arm'] for d in DOMAINS},
           'selection_J': fr['J_by_arm_domain'], 'fixed_source_t': fixed['program_by_domain'], 'main': main, 'vs_refs': vs_refs, 'arms_vs_none': arms_vs_none,
           'refs_vs_none': refs_vs_none, 'nmse_three_domain': nmse, 'delivered_counts': deliv, 'behaviour': beh, 'per_case': per_case,
           'source_evidence': source_evidence_summary(), 'costs': costs()}
    write(ROOT / 'result.json', res)
    (ROOT / 'REPORT.md').write_text(report_md(res), encoding='utf-8')
    return res


def source_evidence_summary() -> dict:
    plan = read(paths()['plan'])
    out = {}
    for d in DOMAINS:
        cases = sorted(plan['cases']['source'][d])
        g = {m: statistics.fmean(statistics.fmean(_g_seed(stage_e('source', c), m)) for c in cases) for m in MENU_IDS}
        commits = [(commit_of(paths()['source'], c, 'f0') or {}).get('label', 'INCOMPLETE') for c in cases]
        out[d] = {'tsfm_G_by_program': g, 'tsfm_best': max(g, key=g.get), 'no_card_commits': dict(OS.collections_counter(commits))}
    return out


def costs() -> dict:
    led = read(paths()['ledger'])
    by_stage = {}
    for e in led['events']:
        if e.get('kind') == 'llm_finished':
            r = by_stage.setdefault(e.get('stage', '?'), {'requests': 0, 'tokens': 0})
            r['requests'] += 1
            r['tokens'] += (e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0)
    return {'tokens': led['llm_tokens_in'] + led['llm_tokens_out'], 'requests': led['llm_requests'], 'http': led['llm_http_attempts'],
            'extra_transport': led.get('extra_transport_attempts', 0), 'unknown_usage': led['llm_tokens_unknown'], 'unknown_accepted': led.get('unknown_usage_accepted', 0),
            'fits': led['fit_attempts'], 'fits_ok': led['fits_ok'], 'fits_failed': led['fits_failed'], 'retries': led['retries_used'], 'llm_by_stage': by_stage,
            'fit_wall_seconds': led['fit_wall_seconds'], 'paid_hours': ((led.get('paid_clock_stop') or time.time()) - led.get('paid_clock_start', led['started_epoch'])) / 3600}


def report_md(res: dict) -> str:
    f = lambda x: '%+.2f' % x
    L_ = ['# Time-MoE 条件化离线学卡（%s）' % PACKAGE, '', '身份：%s。' % res['identity'],
          'Consumer：Time-MoE-50M 预训练权重逐方案微调，lr %g、%d 步、每步 %d 个父窗（Source 不增强校准）。' % (res['consumer']['lr'], res['consumer']['steps'], res['consumer']['batch']),
          '', '| 臂 / 参照 | nMSE（三域等权） | pp of None | 电力 | 交通 | 太阳能 |', '|---|---:|---:|---:|---:|---:|']
    for a in TEST_ARMS:
        r = res['arms_vs_none'][a]
        L_.append('| %s | %.4f | %s | %s | %s | %s |' % (a, res['nmse_three_domain'][a], f(r['three_domain']), *[f(r['by_domain'][d]) for d in DOMAINS]))
    for k in REFS:
        r = res['refs_vs_none'][k]
        L_.append('| %s | %.4f | %s | %s | %s | %s |' % (k, res['nmse_three_domain'][k], f(r['three_domain']), *[f(r['by_domain'][d]) for d in DOMAINS]))
    L_ += ['', '成对差（pp of None，时期 × 实体组聚类 95% 区间，胜/同/负）：', '']
    for k, v in res['main'].items():
        ci = v['cluster_bootstrap_95']
        L_.append('- %s：%s [%s, %s]；%s；最大伤害 %s' % (k, f(v['three_domain']), f(ci['lo95']), f(ci['hi95']), v['wins_same_losses'], f(v['max_harm'])))
    for k in ('f_tsfm_minus_NoMix', 'f_tsfm_minus_Fixed_source_T', 'f_tsfm_minus_ZeroShot'):
        v = res['vs_refs'][k]
        ci = v['cluster_bootstrap_95']
        L_.append('- %s：%s [%s, %s]；%s' % (k, f(v['three_domain']), f(ci['lo95']), f(ci['hi95']), v['wins_same_losses']))
    L_ += ['', '选卡：%s；Source 固定方案：%s。' % (json.dumps(res['cards']), json.dumps(res['fixed_source_t'])),
           '成本：%.2fM token、%d 请求、%d 拟合（失败 %d）；付费墙钟 %.1f h。' % (res['costs']['tokens'] / 1e6, res['costs']['requests'], res['costs']['fits'],
                                                                   res['costs']['fits_failed'], res['costs']['paid_hours'])]
    return '\n'.join(L_) + '\n'


# ============================================================================= smoke (scripted: no LLM client object, no fit)
def smoke() -> dict:
    sm = paths()['smoke']
    if sm.exists():
        shutil.rmtree(sm)
    sm.mkdir(parents=True)
    checks = []

    def check(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})
        print('SMOKE', 'OK ' if ok else 'FAIL', name, info if not ok else '', flush=True)

    plan = freeze_plan()
    cf = {'lr': 1e-5, 'steps': 50, 'batch': 16}
    check('fast_text_tsfm', 'Time-MoE' in FAST_SYSTEM_T and 'MLP' not in FAST_SYSTEM_T)
    check('slow_text_no_mlp', 'MLP' not in slow_system_t('D01'))
    ct = consumer_t(cf)
    check('consumer_desc', 'Time-MoE' in ct['arch'] and ct['n_updates'] == 50 and 'MLP' not in json.dumps(ct))
    prepare_all(plan['cases'])
    bad = [c for c in all_cases(plan['cases']) if not (paths()['common'] / c / 'e_inputs.npz').exists()]
    check('prepared_88_with_e_inputs', not bad and len(all_cases(plan['cases'])) == 88, bad=bad[:3])
    check('val_inputs_on_calibration_cases_only', all((paths()['common'] / c / 'val_inputs.npz').exists() for c in calib_cases())
          and not any((paths()['common'] / c / 'val_inputs.npz').exists() for c in plan['cases']['test']['D01']))
    c0 = calib_cases()[0]
    cs = css()[c0]
    with np.load(paths()['common'] / c0 / 'e_inputs.npz') as z:
        xe = z['X']
    ctx = ec.open_case(cs, paths()['common'], 'material')
    check('e_inputs_shape_and_T_scaler', xe.shape == (16, 4, L) and np.isfinite(xe).all())
    check('model_and_source_present', (MODEL_DIR / 'model.safetensors').exists() and (SRC / 'time_moe').exists(), model=str(MODEL_DIR))
    check('gpu_config', len(GPUS) >= 1 and GPU_LANES_MAX >= 1, gpus=GPUS, lanes=GPU_LANES_MAX)
    res = {'written_local': now(), 'checks': checks, 'passed': sum(c['ok'] for c in checks), 'total': len(checks)}
    write(sm / 'smoke_result.json', res)
    print('SMOKE_DONE', res['passed'], '/', res['total'], flush=True)
    return res


def status() -> None:
    p = paths()['status']
    print(json.dumps(read(p), indent=1, ensure_ascii=False) if p.exists() else '{}')


# ============================================================================= selftest (no LLM; a separate root; a few short Time-MoE fits)
def selftest() -> dict:
    """Everything the paid run relies on before its first request, on this host: the Time-MoE worker (generate equality, repeat and
    second-GPU bitwise equality, the child path), a two-case calibration, Source fits scored only behind the barrier, the zero-feedback
    Fast plumbing with a scripted client (the case card carries the Time-MoE description), the Slow census of one case and a Test
    zero-shot anchor scored behind the Test barrier. Result: <ROOT>/selftest_result.json (read by server/run_tsfm.sh)."""
    global ROOT, CALIB, CALIB_CASES
    base, calib0, cases0 = ROOT, dict(CALIB), list(CALIB_CASES)
    st_root = base.parent / (base.name + '_selftest')
    if st_root.exists():
        shutil.rmtree(st_root)
    ROOT = st_root
    os.environ['SEH_TSFM_ROOT'] = str(st_root)                 # the fit workers resolve the same root
    CALIB = {**CALIB, 'lrs': [CALIB['lrs'][0]], 'batches': [int(os.environ.get('SEH_SELFTEST_BATCH', CALIB['batches'][0]))], 'marks': [5, 10]}
    CALIB_CASES = ['D01_SCR_A1_G1', 'D03_SCR_A1_G1']
    checks, gpu, t0 = [], None, time.time()

    def check(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})
        print('SELFTEST', 'OK ' if ok else 'FAIL', name, json.dumps(info, default=str)[:300] if info else '', flush=True)

    old_proxy = DP.proxy_reachable
    try:
        plan = freeze_plan()
        cc = css()
        src, tst = 'D01_SCR_A1_G1', sorted(plan['cases']['test']['D02'])[0]
        for c in sorted(set(CALIB_CASES + [src, tst])):
            prepare_common(c, cc[c])
        led = SafeLedger(paths()['ledger'], max_fit_attempts=60, max_llm_requests=5, max_llm_tokens=10 ** 6, max_wall_s=3600, max_retries=6)
        gpu = GPUQueue(led)
        gpu.start()
        w = wiring(gpu)
        check('wiring', w['status'] == 'PASS', **{k: v for k, v in w.items() if k not in ('fit_records', 'written_local', 'case')})
        recs = w.get('fit_records', {})
        check('wiring_resources', True, per_fit={k.split('__')[-1]: {'train_s': round(v['train_seconds'] or 0, 1), 'peak_cuda_mb': round(v['peak_cuda_mb'] or 0),
                                                                     'rss_mb': v['peak_rss_mb_process']} for k, v in recs.items()})
        cf = calibration(gpu)
        check('calibration', np.isfinite(cf['objective']) and np.isfinite(cf['zero_shot_anchor']['objective']) and cf['steps'] in CALIB['marks'],
              chosen={k: cf[k] for k in ('lr', 'steps', 'batch', 'objective')}, zero_shot=cf['zero_shot_anchor']['objective'])
        cids = gpu.submit('source', src, 'None') + gpu.submit('source', src, 'P_NoMixRecipe')
        gpu.wait(cids)
        try:
            score_stage('source', gpu, {src: cids})
            refused = False
        except RuntimeError:
            refused = True
        check('source_barrier_refuses_before_freeze', refused)
        write(paths()['source'] / 'models_frozen.json', {'frozen_local': now(), 'selftest': True})
        score_stage('source', gpu, {src: cids})
        se = stage_e('source', src)
        check('source_scores', all(np.isfinite(v) for p in se.values() for v in p.values()) and set(se) == {'None', 'P_NoMixRecipe'},
              nmse={p: [round(v, 4) for v in d.values()] for p, d in se.items()})
        seen = {}

        class FC:
            def call(self, role, unit, payload, system, *, max_tokens, meta):
                seen['system'], seen['payload'] = system, payload
                return json.dumps({'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'P_NoMixRecipe', 'reason': 'selftest'}}]}), \
                    {'request': 0, 'prompt_tokens': 100, 'completion_tokens': 10, 'message_bytes': 1000}
        DP.proxy_reachable = lambda: True
        got = []
        run_fast_stage('source', [(src, 'f0')], lambda c, a: OS.no_card(), led=led, client=FC(), on_commit=lambda c, a, cm: got.append(cm), sdir=paths()['source'])
        txt = json.dumps(seen.get('payload'), ensure_ascii=False)
        check('fast_plumbing', len(got) == 1 and got[0]['physical_material'] == 'P_NoMixRecipe' and seen.get('system') == FAST_SYSTEM_T
              and 'Time-MoE' in txt and 'MLP 192' not in txt)
        sp = paths()['source'] / 'scores' / ('%s.json' % src)
        r = read(sp)
        nones = [v for v in r['cells'].values() if v['phys'] == 'None']
        for m in MENU_IDS:                                        # selftest only: stand-in rows so one census record can be built
            if m not in se:
                for v in nones:
                    r['cells']['%s__%s__s%d' % (src, m, v['seed'])] = {**v, 'phys': m, 'selftest_fill': True}
        write(sp, r)
        cen = census_case_t(src)
        payload = slow_payload_t({'scope': 'D01', 'domains': ['D01'], 'n_cases': 1, 'cases': [cen], 'legal_evidence_refs': OS.legal_refs_of([src]),
                                  'units': 'selftest'}, 'D01', 1)
        ptxt = json.dumps(payload, ensure_ascii=False)
        check('slow_census_payload', 'Time-MoE' in json.dumps(payload['deployment']['consumer']) and 'fast_trajectory' in ptxt and 'scripted_records' in ptxt,
              bytes=len(ptxt.encode('utf-8')))
        z = gpu.submit_special('test', tst, 'zeroshot', 'zeroshot')
        gpu.wait([z])
        try:
            score_stage('test', gpu, {tst: [z]})
            refused = False
        except RuntimeError:
            refused = True
        write(paths()['test'] / 'models_frozen.json', {'frozen_local': now(), 'selftest': True})
        score_stage('test', gpu, {tst: [z]})
        check('test_barrier_and_zero_shot', refused and np.isfinite(stage_e('test', tst)['zeroshot'][0]), nmse=stage_e('test', tst)['zeroshot'][0])
    except Exception as exc:  # noqa: BLE001
        check('exception', False, error=repr(exc)[:500], traceback=traceback.format_exc()[-1500:])
    finally:
        DP.proxy_reachable = old_proxy
        if gpu is not None:
            gpu.stop = True
            with gpu.cv:
                gpu.cv.notify_all()
        ROOT, CALIB, CALIB_CASES = base, calib0, cases0
        os.environ.pop('SEH_TSFM_ROOT', None)
    res = {'written_local': now(), 'recipe': RECIPE, 'gpus': GPUS, 'tsfm_python': TSFM_PYTHON, 'seconds': round(time.time() - t0, 1),
           'passed': sum(c['ok'] for c in checks), 'total': len(checks), 'status': 'PASS' if checks and all(c['ok'] for c in checks) else 'FAIL', 'checks': checks,
           'selftest_root': str(st_root)}
    write(base / 'selftest_result.json', res)
    print('SELFTEST_DONE', res['status'], '%d/%d' % (res['passed'], res['total']), '%.0fs' % res['seconds'], flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--where', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--readout', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--source-check', action='store_true', help='print <ROOT>/source_zeroshot_check.json')
    ap.add_argument('--source-screen', action='store_true', help='no-LLM Source screen of the current recipe (head recipe pilot)')
    ap.add_argument('--blend-check', action='store_true', help='post-hoc zero-shot / fine-tuned forecast blends on the Source cases of this root')
    ap.add_argument('--context-screen', action='store_true', help='zero-shot with prepared inference context (length x normalization) on the Source cases')
    ap.add_argument('--fit-batch')
    a = ap.parse_args()
    if a.source_check:
        p = ROOT / 'source_zeroshot_check.json'
        return print_zeroshot_check(read(p)) if p.exists() else print('no source_zeroshot_check.json yet')
    if a.fit_batch:
        return fit_batch(a.fit_batch)
    if a.source_screen:
        return source_screen()
    if a.blend_check:
        return blend_check()
    if a.context_screen:
        return context_screen()
    if a.where:
        print(ROOT)
        return
    if a.smoke:
        smoke()
    if a.selftest:
        if selftest()['status'] != 'PASS':
            sys.exit(2)
    if a.wiring:
        freeze_plan()
        prepare_all()
        led = SafeLedger(paths()['ledger'], max_fit_attempts=CAPS['fits'], max_llm_requests=CAPS['logical_requests'], max_llm_tokens=CAPS['tokens'],
                         max_wall_s=CAPS['wall_s'], max_retries=CAPS['fit_retries'])
        gpu = GPUQueue(led)
        gpu.start()
        wiring(gpu)
        gpu.stop = True
    if a.run:
        run()
    if a.readout:
        readout()
    if a.status:
        status()


if __name__ == '__main__':
    main()
