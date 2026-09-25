"""DEV-AUG-DLINEAR-OFFLINE-SKILL (docs/DEV_AUG_DLINEAR_OFFLINE_SKILL_TASK_2026-09-24.md): the OFFLINE-SKILL design re-run with the official
DLinear as the Consumer, on this Windows machine (CPU fit workers). Source evidence (12 scripted programs + real no-card zero-feedback Fast
trajectories that see the DLinear Consumer description) is trained and scored with DLinear; Slow forms two candidate domain cards per domain from
that evidence only; Select picks one per domain by the zero-feedback J; Test compares the DLinear-learned card, the frozen MLP card, no card and a
naive card regenerated for DLinear, with None / NoMix / the DLinear-Source fixed program as references.

Derived from batch_research_aug_patchtst_offline_skill (same Fast / Slow / Select / Test protocol, barriers and readout). Consumer-specific parts:
the official DLinear (cure-lab/LTSF-Linear 0c11366) with the CPU training loop of batch_research_aug_dlinear_source (frozen lr / steps from its
None-only Source calibration); its 288 Source fits are reused by (case, material key, seed). Transport: streamed responses with the one bounded
retry after a wait (batch_research_aug_tsfm_offline_skill amendment 3); a stage with transport-interrupted trajectories stops and a resume
continues them. The parent packages are never written.

  --smoke | --wiring | --run | --readout | --status
  subprocess entries: --fit-batch SPEC | --score-cells SPEC
"""
from __future__ import annotations

import os

import argparse
import heapq
import json
import math
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha import batch_zero_feedback as zf
from methods.ttha.batch_base import budget, context, data
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_aug_offline_skill as OS
from evaluation.main_protocol_p4 import batch_research_aug_patchtst_offline_skill as PTS
from evaluation.main_protocol_p4 import batch_research_aug_dlinear_source as DS
from evaluation.main_protocol_p4 import batch_research_aug_tsfm_offline_skill as TS
from evaluation.main_protocol_p4 import batch_research_domain_aug_decision_priority as DP
from evaluation.main_protocol_p4 import batch_research_aug_task_family_screen as SC

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_aug_dlinear_offline_skill'
PARENT = OS.ROOT                                          # _scratch/dev_aug_offline_skill (read-only)
TRANSFER = DS.ROOT                                        # _scratch/dev_aug_dlinear_source (read-only): frozen Consumer + 288 Source fits
MODULE = 'evaluation.main_protocol_p4.batch_research_aug_dlinear_offline_skill'
TASK = 'docs/DEV_AUG_DLINEAR_OFFLINE_SKILL_TASK_2026-09-24.md'
PACKAGE = 'DEV-AUG-DLINEAR-OFFLINE-SKILL'
IDENTITY = 'EXPOSED_DEVELOPMENT_REVALIDATION (the 40 Test cases were exposed by DEV-AUG-OFFLINE-SKILL and the PatchTST packages; not an independent validation)'
DOMAINS = OS.DOMAINS
SEEDS = OS.SEEDS
MENU_IDS = list(SC.MENU_IDS)
CAPS = {'tokens': 36_000_000, 'logical_requests': 1450, 'http': 1520, 'extra_transport': 40, 'fits': 1800, 'fit_retries': 40, 'wall_s': 24 * 3600}
HTTP = int(os.environ.get('SEH_HTTP', '12'))                # user 2026-09-24: API parallelism 8-16
CPU_POOL = 2                                              # material builds (torch 1 thread, CPU)
GPU_LANES_DEFAULT = 2                                     # CPU fit worker processes (DLinear worker peak ~0.7 GB); resources.json may change it while running
FIT_LANES_MAX = 6
MIN_FREE_GB = 0.8
EXTRA_LANE_FREE_GB = 1.2                                  # a second or later fit lane takes a batch only above this free memory
MAX_BATCH = 36                                            # fits of one case per worker process (12 programs x 3 seeds)
SELECT_ARMS = ('d1', 'd2')
TEST_ARMS = ('f_dlin', 'f_mlp', 'f0', 'f_naive')
EARLY_ARMS = ('f0', 'f_naive', 'f_mlp')                   # need nothing learned in this package; start at launch
PRIO = {'score': -1, 'source': 0, 'select': 1, 'test': 2, 'wiring': -2}

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
            'status': ROOT / 'status.json', 'wiring': ROOT / 'wiring', 'smoke': ROOT / 'smoke'}


# ============================================================================= the Consumer as the agents see it (task item 2: Fast sees DLinear)
def consumer_frozen() -> dict:
    c = read(TRANSFER / 'consumer_frozen.json')
    return {'lr': float(c['lr']), 'steps': int(c['steps'])}


def consumer_p() -> dict:
    c = consumer_frozen()
    return {'arch': 'DLinear (official LTSF-Linear model): each 192-point input window is split by a moving average (kernel 25, edge-padded) into a trend '
                    'part and a remainder part; one linear layer maps each part from the 192 inputs to the 48 outputs and the two outputs are added; no '
                    'normalization layer, no dropout, no hidden layer',
            'input': 'univariate: every entity window is one sample of one channel; one model (shared linear layers) for the 16 entities of the case',
            'optimizer': 'AdamW', 'lr': c['lr'], 'weight_decay': 0.0001, 'n_updates': c['steps'], 'parent_batch': 64,
            'loss': ec.CONSUMER['loss'],
            'normalization': 'per-entity mean/std fitted on T only; derived pairs are NOT re-normalized; the model has no instance normalization of its own',
            'metric': ec.CONSUMER['metric'],
            'training': 'shared DLinear, AdamW, %d updates (training budget calibrated once on no-augmentation Source fits), parent batch 64 drawn over the '
                        'actual pool (entities x 433), the same T scaler and batch index stream for every plan of a case' % c['steps'],
            'serving': ec.CONSUMER['serving']}


FAST_SYSTEM_P = OS.FAST_SYSTEM_ZF.replace('one shared MLP Consumer', 'one shared DLinear Consumer')
SLOW_EVIDENCE_P = ('The census holds completed Source cases (entity groups of 16 at two periods). Every downstream number in it comes from the Consumer described '
                   'in deployment.consumer. Each case record puts together: (1) T observations (quantiles over the 16 entities); (2) the REAL no-card '
                   'zero-feedback Fast trajectory - what it observed, which plans it built with which stated hypotheses, what its material inspections showed '
                   '(reduced to numbers) and why it committed or abandoned a material; (3) the downstream result of its actual commit (E, three seeds, as pp of '
                   'the case\'s no-augmentation None); (4) SCRIPTED EXPERIMENT RECORDS: twelve fixed uniform programs that an evaluator trained with the same '
                   'Consumer and scored on the same case (three seeds), together with their material-change statistics. The scripted records come from a '
                   'scripted evaluator run; they are not an agent\'s research process and they are not available to a deployed Fast. A plan the Fast built that '
                   'equals a scripted program is marked; other built plans were never trained and have no downstream number.')


def slow_system_p(scope: str) -> str:
    return ' '.join([OS.SLOW_ROLE, SLOW_EVIDENCE_P, OS.SLOW_GUIDE, OS.SLOW_FORMAT % ('', OS.CARD_BODY_LIMIT)])


# ============================================================================= plan (frozen before any fit or request)
def cases_by_stage() -> dict:
    return read(PARENT / 'frozen_config.json')['cases']


def all_cases(cfg=None) -> list:
    cfg = cfg or cases_by_stage()
    return [c for st in ('source', 'select', 'test') for d in DOMAINS for c in sorted(cfg[st][d])]


def css() -> dict:
    return ec.cases_from_split(read(paths()['split']))


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
            'consumer': {**consumer_frozen(), 'architecture': DS.ARCH, 'kernel_size': 25, 'reference': 'cure-lab/LTSF-Linear', 'official_commit': DS.REF_COMMIT,
                         'device': 'CPU, one torch thread per fit worker process, deterministic algorithms',
                         'source': '_scratch/dev_aug_dlinear_source/consumer_frozen.json (None-only Source C_A+C_B calibration; not re-tuned; Test never used)'},
            'consumer_description_shown_to_agents': consumer_p(), 'fast_system': FAST_SYSTEM_P, 'slow_system_domain': slow_system_p('D01'),
            'naive_system': OS.naive_system(), 'caps': CAPS,
            'transport': {'stream': TS.LLM_STREAM, 'retry_wait_s': TS.RETRY_WAIT_S, 'resume_wait_s': TS.RESUME_WAIT_S,
                          'rule': 'at most one bounded retry per request; a stage whose trajectories stay transport-interrupted stops, a resume continues them'},
            'concurrency': {'http_in_flight': HTTP, 'trajectory_threads_per_stage': HTTP, 'cpu_material_pool': CPU_POOL, 'fit_lanes': 'resources.json (default %d, CPU workers)' % GPU_LANES_DEFAULT,
                            'min_free_gb': MIN_FREE_GB, 'fit_batch': MAX_BATCH},
            'arms': {'source': ['f0'], 'select': list(SELECT_ARMS), 'test': list(TEST_ARMS),
                     'meaning': {'f_dlin': 'domain card learned here from DLinear evidence, chosen on Select', 'f_mlp': 'the frozen OFFLINE-SKILL domain card (learned with MLP evidence)',
                                 'f0': 'no card', 'f_naive': 'naive card regenerated with the DLinear Consumer description (one call + at most one format correction, no selection)'},
                     'same_for_all_agent_arms': 'zero-feedback loop, tools, fields, per-trajectory caps, Fast system text and DLinear Consumer description'},
            'slow': {'calls': '3 domains x 2 candidate domain cards (no shared card); census = DLinear Source evidence only; MLP / PatchTST cards and scores are not shown; '
                              'one format / contract correction; KEEP valid'},
            'selection': 'J_d(W) = mean over Select_d of seed-mean E(c, W) / E(c, None) on DLinear; argmin; exact ties -> fewer deployment tokens -> slot order; '
                         'INCOMPLETE = None delivery',
            'fixed_source_p': 'per domain the scripted program (of 12) with the highest mean pp of None over the 8 Source cases on DLinear; exact ties -> menu order',
            'refs_test': ['None', 'NoMix (P_NoMixRecipe)', 'Fixed_source_D'],
            'incomplete_rule': 'a trajectory without a legal commit is INCOMPLETE; the Runner never supplies a plan; scored as the no-augmentation delivery and counted separately',
            'e_barriers': 'Source E opens after the 24 Source commits are frozen and every Source model is trained; Select E after the 48 Select commits and models; '
                          'Test E after all 160 Test commits are frozen and every Test model (arms and references) is trained. Fits start as soon as a commit is frozen '
                          '(user decision 2026-09-24); no score reaches any trajectory, Slow never reads Select or Test results.',
            'cache': 'materials by assignment key from the parent case caches (read-only); the 288 DLinear Source fits of DEV-AUG-DLINEAR-SOURCE (24 cases x '
                     'None / NoMix / Comp[censor] / Comp[shock] x 3 seeds, same frozen lr / steps, same CPU loop) reused by (case, material key, seed); wiring '
                     're-fits one cached cell and must be bitwise equal',
            'user_decisions_2026_09_24': ['DLinear conditioned offline card learning, full pipeline, on this machine; reuse the existing Source fits; no other '
                                          'new Consumer', 'budget caps as the PatchTST package (API use authorized by the standing instruction)',
                                          'naive card regenerated with the DLinear description', 'the 40 exposed Test cases are a development re-validation',
                                          'Test fits start when each commit freezes; E only after all']}
    want = {'source': {'D01': 8, 'D02': 8, 'D03': 8}, 'select': {'D01': 9, 'D02': 9, 'D03': 6}, 'test': {'D01': 16, 'D02': 16, 'D03': 8}}
    if plan['counts'] != want:
        raise RuntimeError('case counts differ: %s' % plan['counts'])
    write(P_['plan'], plan)
    if not P_['resources'].exists():
        write(P_['resources'], {'gpu_lanes': GPU_LANES_DEFAULT, 'note': 'CPU fit worker processes; read by the running controller before every dispatch'})
    return plan


# ============================================================================= package case cache (copy of the parent view; the parent is never written)
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
    ctx = ec.open_case(cs, paths()['common'], 'material')          # T rows only; raises on scaler / binding drift against the copied parent scaler
    np.savez(dst / 'parents.npz', Xp=ctx.parents.X_norm.reshape(-1, ec.L).astype('float32'), yp=ctx.parents.y_norm.reshape(-1, ec.H).astype('float32'))
    write(dst / 'prepared.json', {'case': case, 'inherited_materials': sorted(reg), 'prepared_local': now(), 'parent': str(src.relative_to(REPO))})


def prepare_all(cfg=None) -> None:
    cc = css()
    for c in all_cases(cfg):
        prepare_common(c, cc[c])


def registry(case: str) -> dict:
    return ec.load_registry(paths()['common'] / case)


def material_path(case: str, phys: str):
    r = registry(case)[phys]
    if r.get('alias_of'):
        raise RuntimeError('alias material %s %s' % (case, phys))
    return r['path']


# ============================================================================= Source-fit cache (the DLinear Source package: same Consumer, same CPU loop)
_TCACHE = None


def transfer_cache() -> dict:
    global _TCACHE
    if _TCACHE is None:
        cf = consumer_frozen()
        out = {}
        for c in read(TRANSFER / 'plan.json')['cases']:
            reg = registry(c)
            for ph in DS.PROGRAMS:
                for s_ in SEEDS:
                    d = TRANSFER / 'fits' / c / ph / str(s_)
                    if (d / 'fit.json').exists() and (d / 'model.pt').exists():
                        r = read(d / 'fit.json')
                        j = r['job']
                        if r['status'] == 'OK' and j['phys'] == ph and int(j['seed']) == s_ and float(j['lr']) == cf['lr'] and int(j['steps']) == cf['steps']:
                            out[(c, reg[ph]['key'], s_)] = str(d)
        _TCACHE = out
    return _TCACHE


# ============================================================================= GPU workers (subprocess; torch only here)
def _peak_rss_mb():
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


def _fit_one(torch, train, j: dict, cfg: dict) -> None:
    """The DLinear Source package's CPU loop (DS.fit_one) with this package's paths; same model / optimizer / batch stream / loss."""
    c, seed, lr, steps = j['case'], int(j['seed']), float(cfg['lr']), int(cfg['steps'])
    out = REPO / j['output']
    out.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / 'common' / c / 'parents.npz') as z:
        xp, yp = torch.as_tensor(z['Xp'].copy()), torch.as_tensor(z['yp'].copy())
    child = None
    if j['material_path']:
        with np.load(j['material_path']) as z:
            if z['Xc'].shape != tuple(xp.shape) or z['yc'].shape != tuple(yp.shape):
                raise RuntimeError('child-parent shape mismatch')
            if not np.isfinite(z['Xc']).all() or not np.isfinite(z['yc']).all():
                raise RuntimeError('nonfinite frozen material')
            child = (torch.as_tensor(z['Xc'].copy()), torch.as_tensor(z['yc'].copy()))
    idxs = torch.as_tensor(train.batch_indices(seed, n_pool=len(xp), n_updates=steps))
    model = DS.model_new(torch, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=DS.WEIGHT_DECAY)
    curve = []
    start = time.time()
    for step in range(1, steps + 1):
        model.train()
        idx = idxs[step - 1]
        opt.zero_grad(set_to_none=True)
        loss_p = ((model(xp[idx, :, None]).squeeze(-1) - yp[idx]) ** 2).mean()
        loss = loss_p if child is None else 0.5 * loss_p + 0.5 * ((model(child[0][idx, :, None]).squeeze(-1) - child[1][idx]) ** 2).mean()
        loss.backward()
        opt.step()
        if step == 1 or step % 250 == 0 or step == steps:
            value = float(loss.detach())
            if not np.isfinite(value):
                raise RuntimeError('nonfinite training loss')
            curve.append([step, value])
    elapsed = time.time() - start
    tmp = out / 'model.tmp.pt'
    torch.save({k: v.detach().clone() for k, v in model.state_dict().items()}, tmp)
    os.replace(tmp, out / 'model.pt')
    write(out / 'fit.json', {'status': 'OK', 'job': j, 'consumer': cfg, 'parameters': sum(p.numel() for p in model.parameters()), 'train_seconds': elapsed,
                             'peak_rss_mb_process': _peak_rss_mb(), 'loss_curve': curve, 'torch': str(torch.__version__), 'device': 'cpu', 'completed_local': now()})
    print('FIT_OK', j['cid'], 'seconds', round(elapsed, 1), flush=True)


def fit_batch(spec_path: str) -> None:
    torch = DS.setup()
    from methods.ttha.batch_base import train
    spec = read(spec_path)
    for j in spec['jobs']:
        fp = REPO / j['output'] / 'fit.json'
        if fp.exists() and read(fp).get('status') == 'OK':
            continue
        try:
            _fit_one(torch, train, j, spec['consumer'])
        except Exception as exc:  # noqa: BLE001
            write(REPO / j['output'] / 'fit_error.json', {'job': j, 'error': repr(exc)[:500], 'traceback': traceback.format_exc()[-4000:], 'local': now()})
            print('FIT_FAIL', j['cid'], repr(exc)[:200], flush=True)


def e_inputs(case: str):
    cs = css()[case]
    with np.load(ROOT / 'common' / case / 'scaler.npz') as z:
        sc = data.Scaler(mean=z['mean'].copy(), std=z['std'].copy(), scale=z['scale'].copy(), floor_hits=0)
        mase = z['mase'].copy()
    origins = list(cs.e)
    sl = ec.load_case_slice(cs, min(origins) - ec.L, max(origins) + ec.H)
    wins = data.build_windows(sl, origins, sc, with_truth=True)
    return (np.stack([w.X_norm for w in wins], axis=1).astype('float32'), np.stack([w.y_true_raw for w in wins], axis=1), sc, mase, origins)


def score_cells(spec_path: str) -> None:
    """Evaluator only: E of the listed cells behind the stage barrier (the transfer package's scoring: raw-scale predictions, ec scorer)."""
    spec = read(spec_path)
    if not (ROOT / spec['stage'] / 'models_frozen.json').exists() and spec['stage'] != 'wiring':
        raise RuntimeError('E barrier: %s models not frozen' % spec['stage'])
    torch = DS.setup()
    x, y, sc, mase, origins = e_inputs(spec['case'])
    out = {}
    for cell in spec['cells']:
        model = DS.model_new(torch, int(cell['seed']))
        model.load_state_dict(torch.load(Path(cell['dir']) / 'model.pt', map_location='cpu', weights_only=True))
        pr = DS.predict(torch, model, x) * sc.scale[:, None, None] + sc.mean[:, None, None]
        s = ec.score_with_status(pr, y, sc, mase)
        if s['status'] != 'SCORABLE':
            raise RuntimeError('non-scorable result %s' % cell['cid'])
        out[cell['cid']] = {'phys': cell['phys'], 'seed': int(cell['seed']), 'key': cell['key'], 'nmse': s['normalized_mse_macro'],
                            'per_entity': s['per_entity_normalized_mse'], 'per_origin': (((pr - y) / sc.scale[:, None, None]) ** 2).mean(axis=(0, 2)).tolist(),
                            'model_dir': cell['dir'], 'cached_from_transfer': cell.get('cached', False)}
        del model
    write(spec['out'], {'case': spec['case'], 'stage': spec['stage'], 'origins': origins, 'cells': out, 'scored_local': now()})
    print('SCORE_OK', spec['case'], len(out), flush=True)


def worker_env() -> dict:
    env = dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env.update(PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env


def run_gpu_process(args: list, log: Path, timeout: float) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open('w', encoding='utf-8') as f:
            p = subprocess.run([sys.executable, '-B', '-u', '-m', MODULE] + [str(a) for a in args], cwd=str(REPO), env=worker_env(), stdout=f,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=timeout)
        return p.returncode
    except subprocess.TimeoutExpired:
        return -999


# ============================================================================= GPU queue (controller side; priority, per-case batches, dynamic lanes)
def gpu_lanes() -> int:
    try:
        return max(1, min(FIT_LANES_MAX, int(read(paths()['resources'])['gpu_lanes'])))
    except Exception:  # noqa: BLE001
        return GPU_LANES_DEFAULT


class GPUQueue:
    """Fits (case, phys, seed) at most once (new fit, earlier fit of this package, or the transfer cache by material key) and E scoring tasks;
    lane threads take the highest priority, batch fits of one case and priority into one worker process, and read the lane count every dispatch."""

    def __init__(self, led: SafeLedger):
        self.led, self.cv = led, threading.Condition()
        self.heap, self.seq = [], 0
        self.state, self.dirs, self.meta, self.errors = {}, {}, {}, {}
        self.score_done, self.stop = {}, False
        self.cfg = consumer_frozen()
        self.threads = []
        self.batches = 0

    def start(self, lanes_max: int = FIT_LANES_MAX):
        for i in range(lanes_max):
            t = threading.Thread(target=self._lane, args=(i,), name='gpu-%d' % i, daemon=True)
            t.start()
            self.threads.append(t)

    def cid(self, case, phys, seed):
        return ec.cell_id(case, phys, seed)

    def submit(self, stage: str, case: str, phys: str) -> list:
        reg = registry(case)
        key, path = reg[phys]['key'], reg[phys].get('path')
        cids = []
        with self.cv:
            for s in SEEDS:
                cid = self.cid(case, phys, s)
                cids.append(cid)
                if cid in self.state and self.state[cid] != 'failed':
                    continue
                out = paths()['fits'] / case / phys / str(s)
                self.meta[cid] = {'case': case, 'phys': phys, 'seed': s, 'key': key, 'stage': stage}
                if (out / 'fit.json').exists() and read(out / 'fit.json').get('status') == 'OK' and read(out / 'fit.json')['job']['key'] == key:
                    self.state[cid], self.dirs[cid] = 'ok', str(out)
                    continue
                tc = transfer_cache().get((case, key, s))
                if tc:
                    self.state[cid], self.dirs[cid] = 'cached', tc
                    self.meta[cid]['cached'] = True
                    self.led.note_cache(cid)
                    continue
                job = {'cid': cid, 'case': case, 'phys': phys, 'seed': s, 'key': key, 'material_path': path, 'stage': stage,
                       'output': str(out.relative_to(REPO)).replace('\\', '/')}
                self.state[cid] = 'queued'
                self.seq += 1
                heapq.heappush(self.heap, (PRIO[stage], self.seq, 'fit', job))
            self.cv.notify_all()
        return cids

    def wait(self, cids: list) -> None:
        with self.cv:
            while not all(self.state.get(c) in ('ok', 'cached', 'failed') for c in cids):
                if self.stop:
                    raise RuntimeError('GPU queue stopped')
                self.cv.wait(10)
            bad = [c for c in cids if self.state[c] == 'failed']
        if bad:
            raise RuntimeError('fits failed: %s (%s)' % (bad[:3], [self.errors.get(b) for b in bad[:3]]))

    def cells(self, cids: list) -> list:
        return [{'cid': c, 'dir': self.dirs[c], **{k: self.meta[c][k] for k in ('phys', 'seed', 'key')}, 'cached': self.state[c] == 'cached'} for c in cids]

    def score(self, stage: str, case: str, cids: list, out: Path) -> dict:
        if out.exists():
            r = read(out)
            if all(c in r['cells'] for c in cids):
                return r
        spec = {'stage': stage, 'case': case, 'cells': self.cells(cids), 'out': str(out)}
        sp = out.with_suffix('.spec.json')
        write(sp, spec)
        ev = threading.Event()
        with self.cv:
            self.seq += 1
            heapq.heappush(self.heap, (PRIO['score'], self.seq, 'score', {'spec': str(sp), 'case': case, 'event': id(ev)}))
            self.score_done[id(ev)] = (ev, None)
            self.cv.notify_all()
        ev.wait()
        err = self.score_done.pop(id(ev))[1]
        if err:
            raise RuntimeError(err)
        return read(out)

    def _take(self, lane: int):
        with self.cv:
            while True:
                if self.stop:
                    return None
                if lane < gpu_lanes() and self.heap and (lane == 0 or OS.free_gb() >= EXTRA_LANE_FREE_GB):
                    break
                self.cv.wait(5)
            prio, seq, kind, job = heapq.heappop(self.heap)
            if kind == 'score':
                return ('score', [job])
            batch, keep = [job], []
            while self.heap:
                it = heapq.heappop(self.heap)
                if it[2] == 'fit' and it[0] == prio and it[3]['case'] == job['case'] and len(batch) < MAX_BATCH:
                    batch.append(it[3])
                else:
                    keep.append(it)
            for it in keep:
                heapq.heappush(self.heap, it)
            for j in batch:
                self.state[j['cid']] = 'running'
            return ('fit', batch)

    def _lane(self, lane: int):
        while True:
            got = self._take(lane)
            if got is None:
                return
            kind, items = got
            t0 = time.time()
            while OS.free_gb() < MIN_FREE_GB:
                if time.time() - t0 > 1800:
                    break
                time.sleep(5)
            if kind == 'score':
                job = items[0]
                rc = run_gpu_process(['--score-cells', job['spec']], paths()['logs'] / ('score_%s.log' % Path(job['spec']).stem), 1800)
                ev, _ = self.score_done[job['event']]
                self.score_done[job['event']] = (ev, None if rc == 0 else 'score worker rc=%s for %s' % (rc, job['case']))
                ev.set()
                continue
            self._run_fits(items)

    def _run_fits(self, batch: list):
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
        with self.cv:
            self.batches += 1
            n = self.batches
        spec = paths()['fits'] / '_batches' / ('batch_%04d.json' % n)
        write(spec, {'consumer': self.cfg, 'jobs': ok_reserved, 'written_local': now()})
        t0 = time.time()
        rc = run_gpu_process(['--fit-batch', spec], paths()['logs'] / 'fits' / ('batch_%04d.log' % n), 180 + 120 * len(ok_reserved))
        secs = (time.time() - t0) / max(1, len(ok_reserved))
        for j in ok_reserved:
            fp = REPO / j['output'] / 'fit.json'
            ok = fp.exists() and read(fp).get('status') == 'OK' and read(fp)['job']['key'] == j['key']
            reason = '' if ok else ('batch rc=%s' % rc)
            self.led.finish_fit(j['cid'], ok, secs, reason)
            with self.cv:
                if ok:
                    self.state[j['cid']], self.dirs[j['cid']] = 'ok', str(REPO / j['output'])
                elif self.led.can_retry():
                    with self.led.lock:
                        self.led.s['retries_used'] += 1
                        self.led.s['events'].append({'kind': 'fit_retry', 'cell': j['cid'], 'reason': reason, 'epoch': time.time()})
                        self.led._save()
                    self.state[j['cid']] = 'queued'
                    self.seq += 1
                    heapq.heappush(self.heap, (PRIO[j['stage']], self.seq, 'fit', j))
                else:
                    self.state[j['cid']], self.errors[j['cid']] = 'failed', reason
                self.cv.notify_all()
        print('GPU_BATCH', n, 'jobs', len(ok_reserved), 'rc', rc, 'sec/fit', round(secs, 1), 'queue', len(self.heap), flush=True)

    def snapshot(self) -> dict:
        with self.cv:
            c = {}
            for v in self.state.values():
                c[v] = c.get(v, 0) + 1
            return {'states': c, 'queued': len(self.heap), 'batches': self.batches, 'lanes': gpu_lanes()}


# ============================================================================= zero-feedback Fast with the DLinear description
class ZFAdapterP(OS.ZFAdapter):
    def case_card(self) -> dict:
        card = super().case_card()
        card['consumer'] = consumer_p()
        return card


def run_branch_p(bdir: Path, case: str, knowledge: br.Knowledge, led, client, *, cs: ec.CaseSpec, arm: str, resume: bool = False) -> dict:
    """OS.run_branch with the DLinear case card and Fast system text (everything else identical)."""
    common, split_path = paths()['common'], paths()['split']
    kn_p = bdir / 'knowledge.json'
    res_state, spent, ratio = None, 0, None
    if not resume:
        bdir.mkdir(parents=True, exist_ok=False)
    adapter = ZFAdapterP(bdir, case, led, common=common, split_path=split_path, cs=cs)
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
    tc = OS.TrajectoryClient(client, bdir.name, FAST_SYSTEM_P, case=case, arm=arm, spent=spent, ratio=ratio)
    L = OS.LIMITS
    result = zf.run_job_zf(job_id=case, knowledge=knowledge, adapter=adapter, client=tc, public=OS.public_candidates(case), allowed_features=OS.ALLOWED_FEATURES,
                           tool_contracts=OS.CONTRACTS_ZF, entity_count=ec.COHORT_SIZE, limits=zf.Limits(L.max_calls, L.max_tools, L.max_materials, max(60.0, led.remaining())),
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


def run_fast_stage(stage: str, jobs: list, knowledge_of, *, led, client, on_commit, sdir: Path) -> dict:
    """OS.run_fast_stage without MLP prefits: every case is already prepared; a frozen commit is handed to on_commit (the GPU queue) at once."""
    cc = css()
    errors, results = [], {}
    stop = threading.Event()
    q = list(jobs)
    qlock = threading.Lock()

    def finish(case, arm, r):
        results[(case, arm)] = r
        cm = commit_of(sdir, case, arm)
        on_commit(case, arm, cm)

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
                    finish(case, arm, run_branch_p(bdir, case, knowledge_of(case, arm), led, client, cs=cc[case], arm=arm, resume=True))
                else:
                    finish(case, arm, run_branch_p(bdir, case, knowledge_of(case, arm), led, client, cs=cc[case], arm=arm))
            except Exception as exc:  # noqa: BLE001
                errors.append(('branch', case, arm, repr(exc)[:300]))
                print('BRANCH_ERROR', stage, case, arm, repr(exc)[:300], flush=True)

    threads = [threading.Thread(target=worker, name='%s-%d' % (stage, i), daemon=True) for i in range(HTTP)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    again = [k for k, r in results.items() if r.get('status') == 'INCOMPLETE' and r.get('failure_kind') == 'AGENT_CALL_FAILED']
    if again and TS.RESUME_WAIT_S > 0:                       # let a relay outage pass before the one resume round
        print('TRANSPORT_RESUME_WAIT', stage, len(again), 'trajectories', TS.RESUME_WAIT_S, 's', flush=True)
        time.sleep(TS.RESUME_WAIT_S)
    for case, arm in again:
        if led.s.get('extra_transport_attempts', 0) >= CAPS['extra_transport'] or rt.unknown_usage_blocks(led):
            break
        with led.lock:
            led.s['extra_transport_attempts'] = led.s.get('extra_transport_attempts', 0) + 1
            led._save()
        bdir = sdir / 'branches' / ('%s__%s' % (case, arm))
        try:
            finish(case, arm, run_branch_p(bdir, case, knowledge_of(case, arm), led, client, cs=cc[case], arm=arm, resume=True))
        except Exception as exc:  # noqa: BLE001
            errors.append(('resume', case, arm, repr(exc)[:300]))
    missing = [j for j in jobs if j not in results]
    # a trajectory cut by a technical call failure is not an agent that failed to commit: it must not flow on as a no-augmentation delivery
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


def delivered_phys(sdir: Path, case: str, arm: str) -> str:
    cm = commit_of(sdir, case, arm)
    return cm['physical_material'] if cm else 'None'           # INCOMPLETE = no-augmentation delivery (counted separately)


def make_client(led, stage: str):
    return TS.ZFStreamClient(led, ROOT / stage / 'llm', http_cap=CAPS['http'], stage=stage, incidents=paths()['incidents'], extra_cap=CAPS['extra_transport'])


# ============================================================================= Source evidence on DLinear
def source_cells(case: str) -> list:
    phys = list(MENU_IDS)
    d = delivered_phys(paths()['source'], case, 'f0')
    if d not in phys:
        phys.append(d)
    return phys


def source_e(case: str) -> dict:
    r = read(paths()['source'] / 'scores' / ('%s.json' % case))
    out = {}
    for cell in r['cells'].values():
        out.setdefault(cell['phys'], {})[int(cell['seed'])] = float(cell['nmse'])
    return out


def _g_seed(se, phys):
    return [100.0 * (se['None'][s] - se[phys][s]) / statistics.fmean(se['None'][t] for t in SEEDS) for s in SEEDS]


def scripted_records_p(case: str, se: dict) -> dict:
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


def census_case_p(case: str) -> dict:
    P_ = paths()
    bdir = P_['source'] / 'branches' / ('%s__f0' % case)
    se = source_e(case)
    cm = commit_of(P_['source'], case, 'f0')
    phys = cm['physical_material'] if cm else 'None'
    g = _g_seed(se, phys)
    scr = scripted_records_p(case, se)
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


def build_census_p(scope: str) -> dict:
    cases = sorted(read(paths()['plan'])['cases']['source'][scope])
    return {'scope': scope, 'domains': [scope], 'n_cases': len(cases), 'cases': [census_case_p(c) for c in cases], 'legal_evidence_refs': OS.legal_refs_of(cases),
            'units': 'G = 100 x (E(None) - E(plan)) / E(None) of that case, three-seed mean unless by_seed; E = four 48-hour forecasts at t+192..t+336 of the case'}


def slow_payload_p(cen: dict, scope: str, slot: int) -> dict:
    p = OS.slow_payload_zf(cen, scope, slot)
    p['deployment']['consumer'] = consumer_p()
    return p


def fixed_source_p() -> dict:
    out_p = paths()['freeze'] / 'fixed_source_p.json'
    if out_p.exists():
        return read(out_p)
    plan = read(paths()['plan'])
    rec = {'rule': plan['fixed_source_p'], 'frozen_local': now(), 'domains': {}}
    for d in DOMAINS:
        cases = sorted(plan['cases']['source'][d])
        g = {m: statistics.fmean(statistics.fmean(_g_seed(source_e(c), m)) for c in cases) for m in MENU_IDS}
        best = max(MENU_IDS, key=lambda m: (g[m], -MENU_IDS.index(m)))
        rec['domains'][d] = {'program': best, 'label': OS.MENU_LABEL[best], 'mean_G_vs_None_by_program': g}
    write_new(out_p, rec)
    return rec


# ============================================================================= Slow (6 calls) and naive cards (3 calls)
def slow_stage(led, client) -> dict:
    P_ = paths()
    summ_p = P_['slow'] / 'proposals.json'
    if summ_p.exists():
        return read(summ_p)
    cens, legal = {}, {}
    for scope in DOMAINS:
        cp = P_['slow'] / scope / 'census.json'
        if not cp.exists():
            write(cp, build_census_p(scope))
        cens[scope] = read(cp)
        legal[scope] = cens[scope]['legal_evidence_refs']
    sizes = {}
    for scope in DOMAINS:
        for k in OS.SLOTS:
            b = len(json.dumps([{'role': 'system', 'content': slow_system_p(scope)}, {'role': 'user', 'content': json.dumps(slow_payload_p(cens[scope], scope, k), ensure_ascii=False)}],
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
            call = lambda p: client.call('slow', '%s_%d' % (scope, k), p, slow_system_p(scope), max_tokens=OS.MAX_OUTPUT_TOKENS, meta={'scope': scope, 'slot': k},
                                         request_timeout=OS.SLOW_TIMEOUT_S)
            res = OS.propose_card(slow_payload_p(cens[scope], scope, k), call, scope=scope, slot=k, legal_refs=legal[scope])
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


def card_p(scope: str, slot: int):
    r = read(paths()['slow'] / scope / ('proposal_%d.json' % slot))
    return dsk.skill_from_json(r['skill']) if r['status'] == 'PROPOSED' else None


def naive_payload_p(domain: str) -> dict:
    plan = read(paths()['plan'])
    cases = sorted(plan['cases']['source'][domain])
    cc = css()
    obs = [{'case_ref': c, 'observations': OS._obs_summary(paths()['common'], cc[c])} for c in cases]
    return {'scope': domain, 'deployment': {'tools': OS.CONTRACTS_ZF, 'actions': OS.ES.action_table(), 'fields': OS.OBS_FIELDS_ZF, 'field_definitions': OS.FIELD_DEFINITIONS,
                                            'consumer': consumer_p(), 'limits': {'requests': OS.LIMITS.max_calls, 'tool_calls': OS.LIMITS.max_tools,
                                                                                 'constructed_materials': OS.LIMITS.max_materials, 'commit': 1},
                                            'visible_domain_identity': 'the case card shows a neutral domain id; this card is loaded by it'},
            'development_observations': {'note': 'T-only observation quantiles of eight development cases of this domain (16 entities each); no downstream result exists here',
                                         'cases': obs},
            'legal_evidence_refs': ['source/%s/observations' % c for c in cases], 'body_limit_characters': OS.CARD_BODY_LIMIT}


def naive_cards_p(led, client) -> dict:
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
            payload = naive_payload_p(d)
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


# ============================================================================= knowledge per arm
def knowledge_select(case: str, arm: str) -> br.Knowledge:
    return OS.knowledge_of_card(card_p(case[:3], int(arm[1])))


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
    if arm == 'f_dlin':
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
        e = select_case_e(case)
        ratio = statistics.fmean(e[delivered_phys(P_['select'], case, arm)].values()) / statistics.fmean(e['None'].values())
        J.setdefault(arm, {}).setdefault(case[:3], []).append(ratio)
        tok.setdefault(arm, {}).setdefault(case[:3], []).append(OS._arm_tokens(led, 'select', '%s__%s' % (case, arm)))
    Jd = {a: {d: statistics.fmean(v) for d, v in dd.items()} for a, dd in J.items()}
    rec = {'frozen_local': now(), 'rule': plan['selection'], 'J_by_arm_domain': Jd, 'tokens_by_arm_domain': {a: {d: statistics.fmean(v) for d, v in dd.items()} for a, dd in tok.items()},
           'domain': {}, 'incomplete': {'%s__%s' % j: True for j in jobs if commit_of(P_['select'], *j) is None}, 'proposals': read(P_['slow'] / 'proposals.json')}
    for d in DOMAINS:
        cands = [a for a in SELECT_ARMS if a in Jd and d in Jd[a]]
        if not cands:
            rec['domain'][d] = {'arm': None, 'slot': None, 'skill': None, 'note': 'no usable domain card; f_dlin runs without a card'}
            continue
        best = min(cands, key=lambda a: (Jd[a][d], statistics.fmean(tok[a][d]), SELECT_ARMS.index(a)))
        sk = card_p(d, int(best[1]))
        rec['domain'][d] = {'arm': best, 'slot': int(best[1]), 'J': {a: Jd[a][d] for a in cands},
                            'tie_break': 'tokens' if len(cands) == 2 and Jd[cands[0]][d] == Jd[cands[1]][d] else None, 'skill': sk.to_json() if sk else None}
    write_new(out_p, rec)
    return rec


def select_case_e(case: str) -> dict:
    r = read(paths()['select'] / 'scores' / ('%s.json' % case))
    out = {}
    for cell in r['cells'].values():
        out.setdefault(cell['phys'], {})[int(cell['seed'])] = float(cell['nmse'])
    return out


# ============================================================================= the controller
class Status:
    def __init__(self):
        self.lock, self.phase = threading.Lock(), {}

    def set(self, part: str, value: str):
        with self.lock:
            self.phase[part] = {'phase': value, 'local': now()}
        print('PHASE', part, value, flush=True)


def score_all(gpu, stage: str, cells: dict) -> None:
    """E scoring of every case of a stage behind its barrier; any failure stops the stage (no partial readout)."""
    errs = []

    def one(c, v):
        try:
            gpu.score(stage, c, v, paths()[stage] / 'scores' / ('%s.json' % c))
        except Exception as exc:  # noqa: BLE001
            errs.append((c, repr(exc)[:300]))
    ths = [threading.Thread(target=one, args=(c, v), daemon=True) for c, v in cells.items()]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:
        raise RuntimeError('%s scoring failed: %s' % (stage, errs[:3]))


def fixed_ref_phys(case: str, program: str, led) -> str:
    if program in ('None', 'FixedMixup', 'P_AmpResample', 'P_NoMixRecipe'):
        return program
    cs = css()[case]
    ctx = ec.open_case(cs, paths()['common'], 'material')
    comp = ec.compile_plan_full(ec.uniform_policy(OS.MENU_STEPS[program], 'fixed reference %s' % OS.MENU_LABEL[program]), ctx.overview()['entities'])
    rec = OS.ensure_material_zf(paths()['common'], paths()['split'], case, comp['assignment'], ledger=led)
    return rec['material_id']


def run() -> None:
    P_ = paths()
    plan = freeze_plan()
    DP.set_pools(CPU_POOL, HTTP)
    rt.FIT_RETRY = True
    with DP.PackageLock(ROOT, 'run'):
        write(P_['status'], {'status': 'RUNNING', 'phase': 'prepare', 'pid': os.getpid(), 'updated': now()})
        prepare_all(plan['cases'])
        led = SafeLedger(P_['ledger'], max_fit_attempts=CAPS['fits'], max_llm_requests=CAPS['logical_requests'], max_llm_tokens=CAPS['tokens'],
                         max_wall_s=CAPS['wall_s'], max_retries=CAPS['fit_retries'])
        if 'paid_clock_start' not in led.s:
            with led.lock:
                led.s['paid_clock_start'], led.s['paid_clock_start_local'] = time.time(), now()
                led._save()
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
                                                                           'retries_used', 'cache_hits')},
                                         'free_gb': round(OS.free_gb(), 2), 'errors': errors[-5:]})
                except Exception:  # noqa: BLE001
                    pass
        threading.Thread(target=heartbeat, daemon=True).start()

        # Source scripted programs: all 864 fits queued at once (critical path)
        for c in [c for d in DOMAINS for c in sorted(plan['cases']['source'][d])]:
            for m in MENU_IDS:
                gpu.submit('source', c, m)
        test_refs_ready = threading.Event()
        test_cids, test_lock = {}, threading.Lock()

        def on_test_commit(case, arm, cm):
            phys = cm['physical_material'] if cm else 'None'
            ids = gpu.submit('test', case, phys)
            with test_lock:
                test_cids.setdefault(case, set()).update(ids)

        def source_pipeline():
            try:
                st.set('source', 'fast')
                run_fast_stage('source', source_jobs(plan), lambda c, a: OS.no_card(), led=led, client=make_client(led, 'source'),
                               on_commit=lambda c, a, cm: gpu.submit('source', c, cm['physical_material'] if cm else 'None'), sdir=P_['source'])
                st.set('source', 'fits')
                scells = {c: [cid for p in source_cells(c) for cid in gpu.submit('source', c, p)] for c in [j[0] for j in source_jobs(plan)]}
                gpu.wait([x for v in scells.values() for x in v])
                if not (P_['source'] / 'models_frozen.json').exists():
                    write(P_['source'] / 'models_frozen.json', {'frozen_local': now(), 'cells': sum(len(v) for v in scells.values()), 'commits_frozen': True})
                st.set('source', 'score')
                score_all(gpu, 'source', scells)
                fx = fixed_source_p()
                test_refs_ready.set()
                print('FIXED_SOURCE_D', json.dumps({d: v['label'] for d, v in fx['domains'].items()}), flush=True)
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
                    write(P_['select'] / 'models_frozen.json', {'frozen_local': now(), 'cells': sum(len(v) for v in vcells.values()), 'commits_frozen': True})
                st.set('select', 'score')
                score_all(gpu, 'select', vcells)
                fr = selection(plan)
                print('SELECTED', json.dumps({d: fr['domain'][d]['arm'] for d in DOMAINS}), flush=True)
                st.set('test_dlin', 'fast')
                run_fast_stage('test_dlin', test_jobs(plan, ('f_dlin',)), knowledge_test, led=led, client=make_client(led, 'test'), on_commit=on_test_commit, sdir=P_['test'])
                st.set('test_dlin', 'done')
            except Exception as exc:  # noqa: BLE001
                errors.append(('source_pipeline', repr(exc)[:500]))
                st.set('source', 'STOPPED: %s' % repr(exc)[:200])
                traceback.print_exc()

        def test_early_pipeline():
            try:
                st.set('naive', 'cards')
                ns = naive_cards_p(led, make_client(led, 'naive'))
                print('NAIVE_DONE', json.dumps({d: v['status'] for d, v in ns['cards'].items()}), flush=True)
                st.set('test_early', 'fast')
                for c in sorted({c for c, _ in test_jobs(plan, EARLY_ARMS)}):
                    for ph in ('None', 'P_NoMixRecipe'):
                        on_test_commit(c, 'ref', {'physical_material': ph})
                run_fast_stage('test_early', test_jobs(plan, EARLY_ARMS), knowledge_test, led=led, client=make_client(led, 'test'), on_commit=on_test_commit,
                               sdir=P_['test'])
                st.set('test_early', 'done')
            except Exception as exc:  # noqa: BLE001
                errors.append(('test_early_pipeline', repr(exc)[:500]))
                st.set('test_early', 'STOPPED: %s' % repr(exc)[:200])
                traceback.print_exc()

        def fixed_ref_pipeline():
            try:
                while not test_refs_ready.wait(30):
                    if errors:
                        return
                fx = read(P_['freeze'] / 'fixed_source_p.json')
                refs = {}
                for c in [c for d in DOMAINS for c in sorted(plan['cases']['test'][d])]:
                    refs[c] = fixed_ref_phys(c, fx['domains'][c[:3]]['program'], led)
                    on_test_commit(c, 'Fixed_source_D', {'physical_material': refs[c]})
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
        # Test barrier: every commit frozen (both dispatches returned) and every needed model trained
        st.set('test', 'fits')
        allc = {}
        for c in [c for d in DOMAINS for c in sorted(plan['cases']['test'][d])]:
            ids = set()
            for a in TEST_ARMS:
                ids.update(gpu.submit('test', c, delivered_phys(P_['test'], c, a)))
            for ph in ('None', 'P_NoMixRecipe', read(P_['test'] / 'fixed_ref_phys.json')['phys'][c]):
                ids.update(gpu.submit('test', c, ph))
            allc[c] = sorted(ids)
        gpu.wait([x for v in allc.values() for x in v])
        if not (P_['test'] / 'models_frozen.json').exists():
            write(P_['test'] / 'models_frozen.json', {'frozen_local': now(), 'cells': sum(len(v) for v in allc.values()), 'commits_frozen': 160})
        st.set('test', 'score')
        score_all(gpu, 'test', allc)
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
def test_case_e(case: str) -> dict:
    r = read(paths()['test'] / 'scores' / ('%s.json' % case))
    out = {}
    for cell in r['cells'].values():
        out.setdefault(cell['phys'], {})[int(cell['seed'])] = float(cell['nmse'])
    return out


PAIRS = (('dlin_minus_mlp', 'f_dlin', 'f_mlp'), ('dlin_minus_f0', 'f_dlin', 'f0'), ('dlin_minus_naive', 'f_dlin', 'f_naive'),
         ('mlp_minus_f0', 'f_mlp', 'f0'), ('naive_minus_f0', 'f_naive', 'f0'))
REFS = ('None', 'NoMix', 'Fixed_source_D')


def readout() -> dict:
    P_ = paths()
    plan = read(P_['plan'])
    led = read(P_['ledger'])
    fixed = read(P_['test'] / 'fixed_ref_phys.json')
    fr = read(P_['freeze'] / 'cards_frozen.json')
    per_case = {}
    for case in [c for d in DOMAINS for c in sorted(plan['cases']['test'][d])]:
        d = case[:3]
        e = test_case_e(case)
        em = {p: statistics.fmean(v[s] for s in SEEDS) for p, v in e.items()}
        dp = {a: delivered_phys(P_['test'], case, a) for a in TEST_ARMS}
        rp = {'None': 'None', 'NoMix': 'P_NoMixRecipe', 'Fixed_source_D': fixed['phys'][case]}
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
    res = {'package': PACKAGE, 'identity': IDENTITY, 'written_local': now(), 'cards': {d: fr['domain'][d]['arm'] for d in DOMAINS},
           'selection_J': fr['J_by_arm_domain'], 'fixed_source_p': fixed['program_by_domain'], 'main': main, 'vs_refs': vs_refs, 'arms_vs_none': arms_vs_none,
           'refs_vs_none': refs_vs_none, 'nmse_three_domain': nmse, 'delivered_counts': deliv, 'behaviour': beh, 'per_case': per_case,
           'source_evidence': source_evidence_summary(), 'costs': costs()}
    write(ROOT / 'result.json', res)
    return res


def source_evidence_summary() -> dict:
    """DLinear vs MLP (screen) vs PatchTST (its offline-skill Source stage) scripted-program means per domain on the same 8 Source cases, and the no-card
    Source commits under each description."""
    plan = read(paths()['plan'])
    out = {}
    for d in DOMAINS:
        cases = sorted(plan['cases']['source'][d])
        pt = {m: statistics.fmean(statistics.fmean(_g_seed(source_e(c), m)) for c in cases) for m in MENU_IDS}
        pp = {m: statistics.fmean(statistics.fmean(_g_seed(PTS.source_e(c), m)) for c in cases) for m in MENU_IDS}
        ml = {}
        for m in MENU_IDS:
            vals = []
            for c in cases:
                se = OS.screen_e(c)
                den = statistics.fmean(se['None'][s] for s in SEEDS)
                vals.append(statistics.fmean(100 * (se['None'][s] - se[m][s]) / den for s in SEEDS))
            ml[m] = statistics.fmean(vals)
        commits_p = [(commit_of(paths()['source'], c, 'f0') or {}).get('label', 'INCOMPLETE') for c in cases]
        commits_m = [(read(PARENT / 'stage_a' / 'evaluation' / 'freeze.json')['commits'].get(c) or {}).get('label', 'INCOMPLETE') for c in cases]
        f0p = statistics.fmean(statistics.fmean(_g_seed(source_e(c), delivered_phys(paths()['source'], c, 'f0'))) for c in cases)
        out[d] = {'dlinear_G_by_program': pt, 'mlp_G_by_program': ml, 'patchtst_G_by_program': pp, 'dlinear_best': max(pt, key=pt.get), 'mlp_best': max(ml, key=ml.get),
                  'patchtst_best': max(pp, key=pp.get),
                  'no_card_commits_dlinear_description': dict(OS.collections_counter(commits_p)), 'no_card_commits_mlp_description': dict(OS.collections_counter(commits_m)),
                  'no_card_G_dlinear': f0p}
    return out


def costs() -> dict:
    led = read(paths()['ledger'])
    by_stage = {}
    for e in led['events']:
        if e.get('kind') == 'llm_finished':
            r = by_stage.setdefault(e.get('stage', '?'), {'requests': 0, 'tokens': 0})
            r['requests'] += 1
            r['tokens'] += (e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0)
    fits = {}
    for e in led['events']:
        if e.get('kind') == 'fit_started':
            k = 'source' if '_SCR_' in e['cell'] else ('select' if '_V_' in e['cell'] else 'test')
            fits[k] = fits.get(k, 0) + 1
    return {'tokens': led['llm_tokens_in'] + led['llm_tokens_out'], 'requests': led['llm_requests'], 'http': led['llm_http_attempts'],
            'extra_transport': led.get('extra_transport_attempts', 0), 'unknown_usage': led['llm_tokens_unknown'], 'unknown_accepted': led.get('unknown_usage_accepted', 0),
            'fits': led['fit_attempts'], 'fits_ok': led['fits_ok'], 'fits_failed': led['fits_failed'], 'retries': led['retries_used'], 'cache_hits': led['cache_hits'],
            'fits_by_stage': fits, 'llm_by_stage': by_stage, 'fit_wall_seconds': led['fit_wall_seconds'],
            'paid_hours': ((led.get('paid_clock_stop') or time.time()) - led.get('paid_clock_start', led['started_epoch'])) / 3600}


# ============================================================================= smoke (scripted: no LLM client object, no fit, no score)
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
    check('fast_text_dlinear', 'DLinear' in FAST_SYSTEM_P and 'MLP' not in FAST_SYSTEM_P and 'PatchTST' not in FAST_SYSTEM_P)
    check('slow_text_no_mlp', 'MLP' not in slow_system_p('D01') and 'screen' not in SLOW_EVIDENCE_P.lower())
    cp = consumer_p()
    check('consumer_desc', 'DLinear' in cp['arch'] and cp['n_updates'] == consumer_frozen()['steps'] and 'MLP' not in json.dumps(cp) and 'PatchTST' not in json.dumps(cp))
    prepare_all(plan['cases'])
    bad = []
    for c in all_cases(plan['cases']):
        a, b = np.load(paths()['common'] / c / 'scaler.npz'), np.load(PARENT / 'common' / c / 'scaler.npz')
        if not (np.array_equal(a['mean'], b['mean']) and np.array_equal(a['scale'], b['scale'])):
            bad.append(c)
    check('scalers_equal_parent_88', not bad and len(all_cases(plan['cases'])) == 88, bad=bad[:3])
    diff = []
    for c in all_cases(plan['cases']):
        with np.load(PTS.ROOT / 'common' / c / 'parents.npz') as x, np.load(paths()['common'] / c / 'parents.npz') as y:
            if not (np.array_equal(x['Xp'], y['Xp']) and np.array_equal(x['yp'], y['yp'])):
                diff.append(c)
    check('parents_equal_patchtst_package_88', not diff, diff=diff[:3])
    check('source_fit_cache_loaded', len(transfer_cache()) == 288, n=len(transfer_cache()))
    case = sorted(plan['cases']['source']['D01'])[0]
    led = SafeLedger(sm / 'budget.json', max_fit_attempts=0, max_llm_requests=5, max_llm_tokens=10 ** 6, max_wall_s=3600, max_retries=0)
    DP.set_pools(1, 1)
    seen = {}

    class FC:
        def call(self, role, unit, payload, system, *, max_tokens, meta):
            seen['system'], seen['card'] = system, payload.get('case') or payload
            return json.dumps({'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'P_NoMixRecipe', 'reason': 'smoke'}}]}), \
                {'request': 0, 'prompt_tokens': 100, 'completion_tokens': 10, 'message_bytes': 1000}
    old = DP.proxy_reachable
    DP.proxy_reachable = lambda: True
    got = []
    try:
        run_fast_stage('smoke', [(case, 'f0')], lambda c, a: OS.no_card(), led=led, client=FC(), on_commit=lambda c, a, cm: got.append(cm), sdir=sm)
    finally:
        DP.proxy_reachable = old
    txt = json.dumps(seen.get('card'), ensure_ascii=False)
    check('fast_saw_dlinear', seen.get('system') == FAST_SYSTEM_P and 'DLinear' in txt and 'MLP 192' not in txt and 'PatchTST' not in txt)
    check('commit_hook', len(got) == 1 and got[0]['physical_material'] == 'P_NoMixRecipe' and got[0]['status'] == 'COMMITTED_UNFITTED')
    check('no_fit_in_smoke', led.s['fit_attempts'] == 0)
    check('parent_untouched', not (PARENT / 'common' / case / 'aug_materials' / 'index.json').read_text(encoding='utf-8').count('inherited_from'))
    kmlp = knowledge_test('D01_Q_Q1_G1', 'f_mlp')
    check('mlp_card_loads', len(kmlp.entries) >= 1 and len(knowledge_test('D01_Q_Q1_G1', 'f0').entries) == 0)
    check('stream_client', make_client.__code__.co_names.count('ZFStreamClient') == 1 and TS.LLM_STREAM)
    res = {'written_local': now(), 'checks': checks, 'passed': sum(c['ok'] for c in checks), 'total': len(checks)}
    write(sm / 'smoke_result.json', res)
    print('SMOKE_DONE', res['passed'], '/', res['total'], flush=True)
    return res


# ============================================================================= wiring (2 real fits, 1 scoring; before any paid request)
def wiring() -> dict:
    """Fit 1: a real Source cell the run needs (Comp[amplitude], not cached). Fit 2: re-trains a cached DLinear Source cell (Comp[shock]) in the same
    worker process after fit 1 -- it must equal the cached model bitwise, and both must score the cached E exactly."""
    P_ = paths()
    plan = freeze_plan()
    prepare_all(plan['cases'])
    wd = P_['wiring']
    wd.mkdir(parents=True, exist_ok=True)
    led = SafeLedger(P_['ledger'], max_fit_attempts=CAPS['fits'], max_llm_requests=CAPS['logical_requests'], max_llm_tokens=CAPS['tokens'],
                     max_wall_s=CAPS['wall_s'], max_retries=CAPS['fit_retries'])
    src = sorted(plan['cases']['source']['D01'])[0]
    reg = registry(src)
    seed = SEEDS[0]
    j1 = {'cid': ec.cell_id(src, 'C_amp', seed), 'case': src, 'phys': 'C_amp', 'seed': seed, 'key': reg['C_amp']['key'], 'material_path': reg['C_amp']['path'],
          'stage': 'source', 'output': str((P_['fits'] / src / 'C_amp' / str(seed)).relative_to(REPO)).replace('\\', '/')}
    j2 = {'cid': 'wiring_refit_' + ec.cell_id(src, 'C_shock', seed), 'case': src, 'phys': 'C_shock', 'seed': seed, 'key': reg['C_shock']['key'],
          'material_path': reg['C_shock']['path'], 'stage': 'wiring', 'output': str((wd / 'refit').relative_to(REPO)).replace('\\', '/')}
    for j in (j1, j2):
        if not (REPO / j['output'] / 'fit.json').exists():
            led.reserve_fit(j['cid'])
    spec = wd / 'batch.json'
    write(spec, {'consumer': consumer_frozen(), 'jobs': [j1, j2], 'written_local': now()})
    t0 = time.time()
    rc = run_gpu_process(['--fit-batch', spec], P_['logs'] / 'wiring_fit.log', 900)
    secs = time.time() - t0
    for j in (j1, j2):
        ok = (REPO / j['output'] / 'fit.json').exists()
        if not any(e.get('kind') == 'fit_finished' and e.get('cell') == j['cid'] for e in led.s['events']):
            led.finish_fit(j['cid'], ok, secs / 2, '' if ok else 'wiring rc=%s' % rc)
    torch = DS.setup()
    a = torch.load(REPO / j2['output'] / 'model.pt', weights_only=True, map_location='cpu')
    tdir = transfer_cache()[(src, reg['C_shock']['key'], seed)]
    b = torch.load(Path(tdir) / 'model.pt', weights_only=True, map_location='cpu')
    bitwise = all(torch.equal(a[k], b[k]) for k in b) and set(a) == set(b)
    sspec = wd / 'score.spec.json'
    write(sspec, {'stage': 'wiring', 'case': src, 'cells': [{'cid': 'refit', 'dir': str(REPO / j2['output']), 'phys': 'C_shock', 'seed': seed, 'key': reg['C_shock']['key']},
                                                          {'cid': 'cached', 'dir': tdir, 'phys': 'C_shock', 'seed': seed, 'key': reg['C_shock']['key']}], 'out': str(wd / 'score.json')})
    rc2 = run_gpu_process(['--score-cells', sspec], P_['logs'] / 'wiring_score.log', 600)
    sc = read(wd / 'score.json')['cells']
    stored = read(TRANSFER / 'scores' / (src + '.json'))['cells']['C_shock__%d' % seed]['nmse']
    f1, f2 = read(REPO / j1['output'] / 'fit.json'), read(REPO / j2['output'] / 'fit.json')
    res = {'written_local': now(), 'fit_rc': rc, 'score_rc': rc2, 'second_fit_in_same_process_bitwise_equal_cached_model': bitwise,
           'e_refit': sc['refit']['nmse'], 'e_cached_model_rescored': sc['cached']['nmse'], 'e_cached_stored': stored,
           'e_equal': sc['refit']['nmse'] == stored == sc['cached']['nmse'], 'train_seconds': [f1['train_seconds'], f2['train_seconds']],
           'batch_wall_seconds': round(secs, 1), 'peak_rss_mb_process': f2.get('peak_rss_mb_process'), 'free_gb_now': round(OS.free_gb(), 2), 'fits': 2,
           'note': 'fit 1 is a real Source cell reused by the run; fit 2 re-trains a cached DLinear Source cell'}
    res['status'] = 'PASS' if (rc == 0 and rc2 == 0 and bitwise and res['e_equal']) else 'FAIL'
    write(wd / 'wiring_result.json', res)
    print('WIRING', res['status'], json.dumps({k: res[k] for k in ('second_fit_in_same_process_bitwise_equal_cached_model', 'e_equal', 'train_seconds', 'batch_wall_seconds',
                                                                   'peak_rss_mb_process', 'free_gb_now')}), flush=True)
    return res


def status() -> None:
    p = paths()['status']
    print(json.dumps(read(p), indent=1, ensure_ascii=False) if p.exists() else '{}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--readout', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--fit-batch')
    ap.add_argument('--score-cells')
    a = ap.parse_args()
    if a.fit_batch:
        return fit_batch(a.fit_batch)
    if a.score_cells:
        return score_cells(a.score_cells)
    if a.smoke:
        smoke()
    if a.wiring:
        wiring()
    if a.run:
        run()
    if a.readout:
        readout()
    if a.status:
        status()


if __name__ == '__main__':
    main()
