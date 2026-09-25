"""TSFM side of DEV-DATA-READINESS-LTSV-CORE-REPAIR-SIGNAL (runs only in the isolated TSFM venv).

Imports numpy / torch / transformers 4.40.1 and the Time-MoE source pinned by the LTSV gitlink; never the readiness package.
Every array it reads was written by the study controller in the base environment; it writes normalized float32 predictions and
update logs, and the controller scores the predictions with the existing readiness.score_block.

  predict(model, X)   48-step autoregressive forecast from 192 normalized inputs with the official head choice (largest head <=
                      remaining: 32, 8, 8), no KV cache, gradient kept when enabled; shared by updates and scoring
  one_step(...)       restore theta0 -> fresh AdamW -> one clipped update on one parent window
  --checks / --blocks / --confirm / --later   stage workers (exit 3 = update budget, 4 = wall clock)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')      # deterministic cuBLAS (set before torch initializes CUDA)

import numpy as np

T_MODULE = time.time()
REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_data_readiness_ltsv_core_repair_signal'
ENV = ROOT / 'tsfm_env'
MODEL_DIR = ENV / 'model'
SRC = ENV / 'src'
H, L = 48, 192
LR, BETAS, EPS, WD, CLIP = 1e-5, (0.9, 0.999), 1e-8, 0.1, 1.0
REF_CHUNK = 64                              # fixed reference batch composition (every evaluation uses the same chunks)
MIDS = ('P0_Linear', 'P1_Seasonal', 'P2_Linear_Hampel')
CONFIRM_SEEDS = (20261201, 20261202, 20261203)
CONFIRM_UPDATES = 100


def torch_setup():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def load_model(device: str = 'cuda'):
    torch = torch_setup()
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from time_moe.models.modeling_time_moe import TimeMoeForPrediction
    model = TimeMoeForPrediction.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32, attn_implementation='eager').to(device)
    model.eval()                                                  # attention_dropout is 0.0; eval() is not no_grad
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def param_groups(model):
    """HF Trainer (4.40.1) grouping used by the LTSV code path: decay everything except parameters named *bias*."""
    decay = [p for n, p in model.named_parameters() if 'bias' not in n]
    no_decay = [p for n, p in model.named_parameters() if 'bias' in n]
    return [{'params': decay, 'weight_decay': WD}, {'params': no_decay, 'weight_decay': 0.0}]


def new_optimizer(model):
    import torch
    return torch.optim.AdamW(param_groups(model), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)


def horizon_plan(model, total: int = H) -> list:
    """The official TSGenerationMixin/TimeMoeForPrediction choice: at each step the largest head <= remaining horizon."""
    heads, plan, left = list(model.config.horizon_lengths), [], total
    while left > 0:
        h = heads[0]
        for x in heads[1:]:
            if x > left:
                break
            h = x
        h = min(h, left)
        plan.append(h)
        left -= h
    return plan


def predict(model, X):
    """X: (B, 192) normalized float32 tensor on the model device -> (B, 48). Autoregressive, own predictions fed back, with the
    KV cache exactly as the official generate runs it (amendment 1: the no-cache unroll differed from generate by up to 3e-3 on the
    real C_A inputs; the cached unroll is bitwise equal, checks/probe_cache_vs_generate.json). Gradients flow through the cache."""
    import torch
    outs, left, past, inp = [], H, None, X.unsqueeze(-1)
    while left > 0:
        out = model(input_ids=inp, past_key_values=past, use_cache=True, return_dict=True, max_horizon_length=left)
        step = out.logits[:, -1, :]                               # (B, horizon)
        outs.append(step)
        past = out.past_key_values
        inp = step.unsqueeze(-1)
        left -= step.shape[1]
    return torch.cat(outs, dim=1)[:, :H]


def official_generate(model, X):
    import torch
    with torch.no_grad():
        out = model.generate(inputs=X.clone(), max_new_tokens=H)
    return out[:, -H:]


def predict_reference(model, Xref):
    """Xref: (N, O, 192) numpy -> (N, O, 48) float32 numpy, fixed chunks, no grad."""
    import torch
    N, O, _ = Xref.shape
    flat = torch.as_tensor(Xref.reshape(N * O, L), dtype=torch.float32, device=next(model.parameters()).device)
    parts = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], REF_CHUNK):
            parts.append(predict(model, flat[i:i + REF_CHUNK]))
    return torch.cat(parts).float().cpu().numpy().reshape(N, O, H)


def snapshot_params(model):
    return [p.detach().clone() for p in model.parameters()]


def restore_params(model, snap):
    import torch
    with torch.no_grad():
        for p, s in zip(model.parameters(), snap):
            p.copy_(s)
    model.zero_grad(set_to_none=True)


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def one_step(model, theta0, x, y, seed: int = 0, count_grads: bool = False):
    """Restore theta0, fresh AdamW, one clipped update on (x (192,), y (48,)); the model is left in the updated state."""
    import torch
    t0 = _sync()
    restore_params(model, theta0)
    torch.manual_seed(seed)
    t1 = _sync()
    opt = new_optimizer(model)
    dev = next(model.parameters()).device
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev).unsqueeze(0)
    t2 = _sync()
    pred = predict(model, xt)
    loss = torch.mean((pred - yt) ** 2)
    t3 = _sync()
    loss.backward()
    t4 = _sync()
    rec = {}
    if count_grads:
        rec['tensors_with_nonzero_grad'] = sum(1 for p in model.parameters() if p.grad is not None and bool(p.grad.abs().sum() > 0))
        rec['tensors_total'] = sum(1 for _ in model.parameters())
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
    t5 = _sync()
    opt.step()
    t6 = _sync()
    del opt
    model.zero_grad(set_to_none=True)
    rec.update({'train_loss': float(loss.item()), 'grad_norm_before_clip': grad_norm,
                'seconds': {'restore': t1 - t0, 'optimizer_new': t2 - t1, 'forward': t3 - t2, 'backward': t4 - t3, 'clip': t5 - t4, 'step': t6 - t5, 'total': t6 - t0}})
    return rec


def param_delta_norm(model, theta0) -> float:
    import torch
    with torch.no_grad():
        return float(torch.sqrt(sum(((p.detach() - s) ** 2).sum() for p, s in zip(model.parameters(), theta0))).item())


def peak_gpu_mb():
    import torch
    return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else None


def job_dir(job: str) -> Path:
    return ROOT / 'jobs' / job


def load_job(job: str):
    with np.load(job_dir(job) / 'train_arrays.npz') as z:
        X = {m: z['X_' + m].astype(np.float32) for m in MIDS}
        y = z['y'].astype(np.float32)
        idx = {s: z['confirm_idx_%d' % s].copy() for s in CONFIRM_SEEDS}
    meta = json.loads((job_dir(job) / 'blocks.json').read_text(encoding='utf-8'))
    with np.load(job_dir(job) / 'ca_inputs.npz') as z:
        ref = z['X_ref'].copy()
    return X, y, idx, meta, ref


def log_update(path: Path, rec: dict) -> None:
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec) + '\n')


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1), encoding='utf-8')
    os.replace(tmp, path)


def _mean(v):
    return float(np.mean(v)) if len(v) else None


# ----------------------------------------------------------------------------------------------------------- probe (no update)
def probe(out_path: str):
    torch = torch_setup()
    import transformers
    model = load_model()
    t_load = _sync() - T_MODULE
    rng = np.random.RandomState(0)
    X = torch.as_tensor(rng.randn(64, L), dtype=torch.float32, device='cuda')
    t1 = _sync()
    with torch.no_grad():
        mine = predict(model, X)
    t_pred = _sync() - t1
    t1 = _sync()
    off = official_generate(model, X)
    t_gen = _sync() - t1
    a, b = mine.double(), off.double()
    diff = (a - b).abs()
    rec = {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__,
           'device': torch.cuda.get_device_name(0), 'model_dir': str(MODEL_DIR), 'dtype': str(next(model.parameters()).dtype),
           'n_params_total': sum(p.numel() for p in model.parameters()), 'n_params_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
           'horizon_plan_48': horizon_plan(model), 'load_seconds': t_load, 'predict_64_seconds': t_pred, 'generate_64_seconds': t_gen,
           'generate_vs_predict': {'max_abs': float(diff.max()), 'max_rel': float((diff / b.abs().clamp_min(1e-12)).max()),
                                   'within_atol1e-5_rtol1e-4': bool(torch.all(diff <= 1e-5 + 1e-4 * b.abs()))},
           'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
           'peak_gpu_mb': peak_gpu_mb(), 'kmp_duplicate_lib_ok': os.environ.get('KMP_DUPLICATE_LIB_OK')}
    Path(out_path).write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(json.dumps(rec, indent=1))


# ----------------------------------------------------------------------------------------------------------- stage workers
def run_checks(job: str, max_updates: int) -> int:
    kmp = os.environ.get('KMP_DUPLICATE_LIB_OK')
    torch = torch_setup()
    model = load_model()
    cold = _sync() - T_MODULE
    X, y, idx, meta, ref = load_job(job)
    theta0 = snapshot_params(model)
    log = ROOT / 'checks' / 'extra_updates.jsonl'
    used = 0
    t = _sync()
    pred0 = predict_reference(model, ref)
    first_ref = _sync() - t
    t = _sync()
    pred0b = predict_reference(model, ref)
    ref_s = _sync() - t
    flat = torch.as_tensor(ref.reshape(-1, L)[:REF_CHUNK], dtype=torch.float32, device='cuda')
    with torch.no_grad():
        mine = predict(model, flat).double()
    off = official_generate(model, flat).double()
    diff = (mine - off).abs()
    gen = {'windows': int(flat.shape[0]), 'max_abs': float(diff.max()), 'max_rel': float((diff / off.abs().clamp_min(1e-12)).max()),
           'within_atol1e-5_rtol1e-4': bool(torch.all(diff <= 1e-5 + 1e-4 * off.abs()))}
    same = next((b for b in meta['blocks'] if not b['x_differs_from_p0']['P1_Seasonal']), None)
    diffb = next(b for b in meta['blocks'] if b['x_differs_from_p0']['P2_Linear_Hampel'])
    anchor = same or diffb
    seq = [('P0_Linear', anchor), ('P2_Linear_Hampel', diffb)] + ([('P1_Seasonal', same)] if same else []) + [('P0_Linear', anchor)]
    recs, preds, block_secs = [], [], []
    for mid, b in seq:
        if used >= max_updates:
            return 3
        r = one_step(model, theta0, X[mid][b['parent']], y[b['parent']], count_grads=True)
        t = _sync()
        r['param_delta_norm'] = param_delta_norm(model, theta0)
        preds.append(predict_reference(model, ref))
        r['seconds']['delta_and_reference'] = _sync() - t
        used += 1
        log_update(log, {'stage': 'extra', 'job': job, 'material': mid, 'block': b['block'], 'epoch': time.time()})
        recs.append({'material': mid, 'block': b['block'], **r})
        block_secs.append(r['seconds']['total'] + r['seconds']['delta_and_reference'])
    checks = {'generate_matches_predict_on_real_c_a_inputs': gen['within_atol1e-5_rtol1e-4'],
              'reference_prediction_repeatable': bool(np.array_equal(pred0, pred0b)),
              'gradient_reaches_parameters': recs[0]['tensors_with_nonzero_grad'] > 0 and recs[0]['grad_norm_before_clip'] > 0,
              'one_step_changes_parameters': all(r['param_delta_norm'] > 0 for r in recs),
              'same_block_same_material_different_order_bitwise_equal': bool(np.array_equal(preds[0], preds[-1])),
              'repair_changes_predictions': not bool(np.array_equal(preds[0], preds[1]))}
    if same:
        checks['identical_x_without_reuse_bitwise_equal'] = bool(np.array_equal(preds[0], preds[2]))
    restore_params(model, theta0)
    checks['theta0_restore_bitwise_equal'] = bool(np.array_equal(predict_reference(model, ref), pred0))
    # confirmation-cost timing: 2 batch-4 updates on P0 from theta0, then reference prediction, state save and load
    restore_params(model, theta0)
    torch.manual_seed(CONFIRM_SEEDS[0])
    opt = new_optimizer(model)
    upd = []
    for u in range(2):
        if used >= max_updates:
            return 3
        t = _sync()
        rows = idx[CONFIRM_SEEDS[0]][u]
        xb = torch.as_tensor(X['P0_Linear'][rows], device='cuda')
        yb = torch.as_tensor(y[rows], device='cuda')
        loss = torch.mean((predict(model, xb) - yb) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        model.zero_grad(set_to_none=True)
        upd.append(_sync() - t)
        used += 1
        log_update(log, {'stage': 'extra', 'job': job, 'material': 'P0_Linear', 'confirm_timing_update': u, 'epoch': time.time()})
    del opt
    t = _sync()
    predict_reference(model, ref)
    ref_adapted = _sync() - t
    tmp = ROOT / 'checks' / 'timing_state.pt'
    t = _sync()
    torch.save(model.state_dict(), tmp)
    save_s = _sync() - t
    t = _sync()
    model.load_state_dict(torch.load(tmp, map_location='cuda'))
    load_s = _sync() - t
    size = tmp.stat().st_size
    tmp.unlink()
    out = {'job': job, 'checks': checks, 'generate': gen, 'sequence': recs, 'extra_updates_used': used, 'kmp_duplicate_lib_ok': kmp,
           'blocks_used': {'identical_p1_block': same['block'] if same else None, 'changed_p2_block': diffb['block']},
           'timing': {'cold_start_seconds': cold, 'first_reference_predict_seconds': first_ref, 'reference_predict_seconds': max(ref_s, ref_adapted),
                      'reference_windows': int(ref.shape[0] * ref.shape[1]), 'block_update_with_reference_seconds': float(np.mean(block_secs[1:] or block_secs)),
                      'block_update_seconds_all': block_secs, 'confirm_update_seconds': float(np.max(upd)), 'confirm_update_seconds_all': upd,
                      'save_model_seconds': save_s, 'load_model_state_seconds': load_s, 'state_bytes': size}, 'peak_gpu_mb': peak_gpu_mb()}
    write_json(ROOT / 'checks' / 'tsfm_checks.json', out)
    return 0


def run_blocks(job: str, max_updates: int, deadline: float) -> int:
    started, kmp = time.time(), os.environ.get('KMP_DUPLICATE_LIB_OK')
    torch_setup()
    model = load_model()
    cold = _sync() - T_MODULE
    X, y, idx, meta, ref = load_job(job)
    theta0 = snapshot_params(model)
    bdir = job_dir(job) / 'blocks'
    bdir.mkdir(parents=True, exist_ok=True)
    log = job_dir(job) / 'block_updates.jsonl'
    p0 = bdir / 'theta0_ca.npz'
    if p0.exists():
        with np.load(p0) as z:
            pred0 = z['pred'].copy()
    else:
        pred0 = predict_reference(model, ref)
        np.savez(p0, pred=pred0)
    N = ref.shape[0]
    used, reused, secs = 0, 0, {'update': [], 'reference': [], 'block_wall': []}
    for b in meta['blocks']:
        f = bdir / ('b%03d.npz' % b['block'])
        if f.exists():
            continue
        tb = time.time()
        preds, recs = np.zeros((3, N, ref.shape[1], H), np.float32), []
        for mi, mid in enumerate(MIDS):
            if mi > 0 and not b['x_differs_from_p0'][mid]:
                preds[mi] = preds[0]
                reused += 1
                recs.append({'material': mid, 'executed': False, 'reused_from': 'P0_Linear'})
                continue
            if used >= max_updates:
                return 3
            if time.time() > deadline:
                return 4
            r = one_step(model, theta0, X[mid][b['parent']], y[b['parent']])
            t = _sync()
            r['param_delta_norm'] = param_delta_norm(model, theta0)
            preds[mi] = predict_reference(model, ref)
            r['seconds']['delta_and_reference'] = _sync() - t
            used += 1
            log_update(log, {'stage': 'block', 'job': job, 'block': b['block'], 'material': mid, 'epoch': time.time()})
            secs['update'].append(r['seconds']['total'])
            secs['reference'].append(r['seconds']['delta_and_reference'])
            recs.append({'material': mid, 'executed': True, **r})
        np.savez(f, pred=preds)
        secs['block_wall'].append(time.time() - tb)
        log_update(bdir / 'records.jsonl', {'block': b['block'], 'records': recs})
    restore_params(model, theta0)
    exact = bool(np.array_equal(predict_reference(model, ref), pred0))
    write_json(bdir / 'done.json', {'job_id': job, 'started_epoch': started, 'finished_epoch': time.time(), 'theta0_restore_exact': exact,
                                    'executed_updates_this_attempt': used, 'reused_this_attempt': reused, 'kmp_duplicate_lib_ok': kmp,
                                    'timing': {'cold_start_seconds': cold, 'update_mean': _mean(secs['update']), 'reference_mean': _mean(secs['reference']),
                                               'block_wall_mean': _mean(secs['block_wall']), 'update_sum': float(np.sum(secs['update'])),
                                               'reference_sum': float(np.sum(secs['reference'])), 'body_seconds': time.time() - started},
                                    'peak_gpu_mb': peak_gpu_mb()})
    return 0


def run_confirm(job: str, max_updates: int, deadline: float) -> int:
    started, kmp = time.time(), os.environ.get('KMP_DUPLICATE_LIB_OK')
    torch = torch_setup()
    model = load_model()
    cold = _sync() - T_MODULE
    X, y, idx, meta, ref = load_job(job)
    theta0 = snapshot_params(model)
    cdir = job_dir(job) / 'confirm'
    cdir.mkdir(parents=True, exist_ok=True)
    log = job_dir(job) / 'confirm_updates.jsonl'
    used, runs = 0, []
    for mid in MIDS:
        for s in CONFIRM_SEEDS:
            name = '%s__s%d' % (mid, s)
            if (cdir / (name + '.json')).exists():
                continue
            if used + CONFIRM_UPDATES > max_updates:
                return 3
            t0 = _sync()
            restore_params(model, theta0)
            torch.manual_seed(s)
            opt = new_optimizer(model)
            losses, norms = [], []
            for u in range(CONFIRM_UPDATES):
                if time.time() > deadline:
                    return 4
                rows = idx[s][u]
                xb = torch.as_tensor(X[mid][rows], device='cuda')
                yb = torch.as_tensor(y[rows], device='cuda')
                loss = torch.mean((predict(model, xb) - yb) ** 2)
                loss.backward()
                norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)))
                opt.step()
                model.zero_grad(set_to_none=True)
                losses.append(float(loss.item()))
                used += 1
                log_update(log, {'stage': 'confirm', 'job': job, 'run': name, 'update': u, 'epoch': time.time()})
            del opt
            t1 = _sync()
            pred = predict_reference(model, ref)
            np.savez(cdir / (name + '_ca.npz'), pred=pred)
            t2 = _sync()
            torch.save(model.state_dict(), cdir / (name + '.pt'))
            t3 = _sync()
            rec = {'run': name, 'material': mid, 'seed': s, 'updates': CONFIRM_UPDATES, 'batch': 4, 'batch_losses': losses, 'grad_norms_before_clip': norms,
                   'param_delta_norm': param_delta_norm(model, theta0), 'model_path': str(cdir / (name + '.pt')),
                   'seconds': {'train': t1 - t0, 'reference': t2 - t1, 'save': t3 - t2}}
            write_json(cdir / (name + '.json'), rec)
            runs.append(rec['seconds'])
    write_json(cdir / 'done.json', {'job_id': job, 'started_epoch': started, 'finished_epoch': time.time(), 'executed_updates_this_attempt': used,
                                    'kmp_duplicate_lib_ok': kmp, 'peak_gpu_mb': peak_gpu_mb(),
                                    'timing': {'cold_start_seconds': cold, 'train_mean': _mean([r['train'] for r in runs]),
                                               'reference_mean': _mean([r['reference'] for r in runs]), 'save_mean': _mean([r['save'] for r in runs]),
                                               'body_seconds': time.time() - started}})
    return 0


def run_later(job: str, deadline: float) -> int:
    if not (ROOT / 'selections.json').exists():
        raise PermissionError('selections are not frozen')
    started, kmp = time.time(), os.environ.get('KMP_DUPLICATE_LIB_OK')
    torch = torch_setup()
    model = load_model()
    cold = _sync() - T_MODULE
    with np.load(job_dir(job) / 'later_inputs.npz') as z:
        Xcb, Xe, ocb, oe = z['X_c_b'].copy(), z['X_e'].copy(), z['c_b_origins'].copy(), z['e_origins'].copy()
    preds, names, per = {}, ['theta0'], []
    t = _sync()
    preds['c_b__theta0'], preds['e__theta0'] = predict_reference(model, Xcb), predict_reference(model, Xe)
    per.append(_sync() - t)
    cdir = job_dir(job) / 'confirm'
    for mid in MIDS:
        for s in CONFIRM_SEEDS:
            if time.time() > deadline:
                return 4
            name = '%s__s%d' % (mid, s)
            t = _sync()
            model.load_state_dict(torch.load(cdir / (name + '.pt'), map_location='cuda'))
            preds['c_b__' + name], preds['e__' + name] = predict_reference(model, Xcb), predict_reference(model, Xe)
            per.append(_sync() - t)
            names.append(name)
    np.savez(job_dir(job) / 'later_predictions.npz', c_b_origins=ocb, e_origins=oe, **preds)
    write_json(job_dir(job) / 'later_frozen.json', {'job_id': job, 'started_epoch': started, 'frozen_epoch': time.time(),
                                                    'frozen_local': time.strftime('%Y-%m-%d %H:%M:%S'), 'models': names, 'kmp_duplicate_lib_ok': kmp,
                                                    'serving': 'P0 Linear inputs',
                                                    'timing': {'cold_start_seconds': cold, 'per_model_load_and_predict': per, 'body_seconds': time.time() - started}})
    return 0


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe')
    ap.add_argument('--checks')
    ap.add_argument('--blocks')
    ap.add_argument('--confirm')
    ap.add_argument('--later')
    ap.add_argument('--max-updates', type=int, default=0)
    ap.add_argument('--deadline', type=float, default=float('inf'))
    a = ap.parse_args()
    if a.probe:
        probe(a.probe)
    elif a.checks:
        sys.exit(run_checks(a.checks, a.max_updates))
    elif a.blocks:
        sys.exit(run_blocks(a.blocks, a.max_updates, a.deadline))
    elif a.confirm:
        sys.exit(run_confirm(a.confirm, a.max_updates, a.deadline))
    elif a.later:
        sys.exit(run_later(a.later, a.deadline))
