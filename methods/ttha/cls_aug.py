"""Classification augmentation wiring (DEV-CLS-AUG-WIRING, 2026-09-25): UCR TRAIN parser, normalization, fit/feedback split, a Torch
re-implementation of FCN, and a thin wrapper around the OFFICIAL AutoDA-Timeseries transforms.

  data         only the <Name>_TRAIN.ts member of a UCR zip is ever opened (TEST members stay unread); values are returned RAW.
  normalize    a separate step: per-series z-normalization (numpy std, ddof 0; std 0 -> 1), as in dl-4-tsc utils for the UCR archive.
  split        per class, seeded permutation, feedback = n_c // 3, fit = the rest (about 2:1); the gate (fit >= 20, feedback >= 10 per class)
               is reported, not enforced here.
  consumer     FCN of Wang et al. 2017 as released in hfawaz/dl-4-tsc classifiers/fcn.py (Conv1D 128/8, 256/5, 128/3 'same' + BN + ReLU, GAP,
               Dense softmax; Adam defaults; 2000 epochs; mini-batch int(min(n/10, 16)); ReduceLROnPlateau(loss, 0.5, 50, min_lr 1e-4);
               keep the weights of the lowest epoch TRAINING loss). This is a RE-IMPLEMENTATION in Torch aligned to Keras semantics
               (TF 'same' padding left (k-1)//2 / right rest; BN momentum 0.99 -> torch 0.01, eps 1e-3; Glorot-uniform kernels, zero biases;
               Adam eps 1e-7; epoch loss = sample-weighted mean of batch losses; LR plateau with absolute min_delta 1e-4). Not bit-identical
               to the Keras code; the loss is computed from logits (Torch cross-entropy) rather than clipped softmax probabilities.
  augmentation training batches only: each sample of a batch is replaced by its augmented copy with probability p; clean and augmented
               samples go through ONE forward pass; the number of samples and updates is the same for every arm; labels are never touched;
               evaluation inputs are never augmented. Transforms are the official classes of AutoDA-Timeseries
               autoaugment/augments/basic_transforms.py (commit 91dbf70), loaded read-only from the local copy; strength is passed as the
               official per-sample strength tensor.
No LLM, no hashing, no I/O beyond reading the given zip.
"""
from __future__ import annotations

import copy
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
AUTODA_REF = ROOT / "_scratch" / "dev_aug_main_comparison" / "ref" / "AutoDA-Timeseries"
AUTODA_TRANSFORMS = AUTODA_REF / "autoaugment" / "augments" / "basic_transforms.py"
AUTODA_COMMIT = "91dbf70b54b255214b7f204d8d9f70d26f9c1fe3"
FCN_SOURCE = "https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py"
EPOCHS = 2000


# ----------------------------------------------------------------------------------------------------------------------------- data
_dspec = importlib.util.spec_from_file_location("cls_aug_data", str(Path(__file__).with_name("cls_aug_data.py")))
_data = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(_data)
train_member, read_ucr_train, znorm = _data.train_member, _data.read_ucr_train, _data.znorm
split_fit_feedback, metrics = _data.split_fit_feedback, _data.metrics


# ----------------------------------------------------------------------------------------------------------------------------- FCN
class _ConvBlock(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.pad = ((k - 1) // 2, k - 1 - (k - 1) // 2)       # TF/Keras 'same': the extra pad goes to the right
        self.conv = nn.Conv1d(cin, cout, k)
        self.bn = nn.BatchNorm1d(cout, eps=1e-3, momentum=0.01)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return F.relu(self.bn(self.conv(F.pad(x, self.pad))))


class FCN(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 1):
        super().__init__()
        self.b1, self.b2, self.b3 = _ConvBlock(in_ch, 128, 8), _ConvBlock(128, 256, 5), _ConvBlock(256, 128, 3)
        self.fc = nn.Linear(128, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):                                      # x: (B, C, T)
        return self.fc(self.b3(self.b2(self.b1(x))).mean(dim=-1))


class _KerasPlateau:
    """keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=1e-4), min_delta 1e-4 absolute, cooldown 0."""

    def __init__(self, opt, factor=0.5, patience=50, min_lr=1e-4, min_delta=1e-4):
        self.opt, self.factor, self.patience, self.min_lr, self.min_delta = opt, factor, patience, min_lr, min_delta
        self.best, self.wait, self.changes = math.inf, 0, []

    def step(self, epoch, loss):
        if loss < self.best - self.min_delta:
            self.best, self.wait = loss, 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            old = self.opt.param_groups[0]["lr"]
            if old > np.float32(self.min_lr):
                new = max(old * self.factor, self.min_lr)
                for g in self.opt.param_groups:
                    g["lr"] = new
                self.changes.append((epoch, new))
            self.wait = 0


def set_determinism():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_fcn(X: np.ndarray, y: np.ndarray, n_classes: int, *, seed: int, device, aug=None, p_aug: float = 0.0, epochs: int = EPOCHS,
            batch_size: int | None = None) -> dict:
    """Train FCN on (already normalized) X (n, T); returns the best-training-loss state and a training record."""
    set_determinism()
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    n = X.shape[0]
    bs = batch_size or max(1, int(min(n / 10, 16)))
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device).unsqueeze(1)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    y_before = yt.clone()
    model = FCN(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-7)
    plateau = _KerasPlateau(opt)
    best_loss, best_epoch, best_state = math.inf, -1, None
    curve, t0 = [], time.time()
    aug_n, aug_sq, aug_changed = (torch.zeros((), device=device, dtype=torch.float64) for _ in range(3))   # device-side: no sync
    aug_pts, seen = 0, 0
    steps = 0
    for ep in range(epochs):
        model.train()
        perm = torch.as_tensor(np_rng.permutation(n), device=device)
        tot = torch.zeros((), device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], yt[idx]
            if aug is not None and p_aug > 0:
                mask = torch.rand(xb.shape[0], device=device) < p_aug
                if bool(mask.any()):
                    xs = xb[mask]
                    xa = aug(xs)
                    d = (xa - xs)
                    aug_n += mask.sum()
                    aug_sq += (d.double() ** 2).sum()
                    aug_pts += d.numel()
                    aug_changed += (d.abs() > 1e-7).sum()
                    xb = xb.clone()
                    xb[mask] = xa
            seen += xb.shape[0]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.detach() * xb.shape[0]
            steps += 1
        ep_loss = float(tot) / n
        curve.append(ep_loss)
        if ep_loss < best_loss:
            best_loss, best_epoch = ep_loss, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        plateau.step(ep, ep_loss)
    secs = time.time() - t0
    if not torch.equal(yt, y_before):
        raise AssertionError("labels changed during training")
    return {"state": best_state, "best_epoch": best_epoch, "best_train_loss": best_loss, "final_train_loss": curve[-1],
            "loss_curve": curve, "lr_changes": plateau.changes, "final_lr": opt.param_groups[0]["lr"], "batch_size": bs,
            "epochs": epochs, "steps": steps, "seconds": secs, "samples_seen": seen,
            "aug": {"p": p_aug, "augmented_samples": int(aug_n), "augmented_fraction": float(aug_n) / max(1, seen),
                    "rms_change": math.sqrt(float(aug_sq) / aug_pts) if aug_pts else 0.0,
                    "fraction_points_changed": float(aug_changed) / aug_pts if aug_pts else 0.0}}


@torch.no_grad()
def predict(state: dict, n_classes: int, X: np.ndarray, device) -> np.ndarray:
    model = FCN(n_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device).unsqueeze(1)
    out = [model(xt[i:i + 256]).argmax(dim=1).cpu() for i in range(0, xt.shape[0], 256)]
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------------------------------------------------- augmentation
_MOD = None


def autoda_transforms():
    """The official basic_transforms module, loaded from the read-only copy without importing the rest of the package."""
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location("autoda_basic_transforms", str(AUTODA_TRANSFORMS))
        _MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MOD)
    return _MOD


class AutoDAOp:
    """One official AutoDA transform at a fixed strength, applied to a (B, 1, T) batch."""

    def __init__(self, name: str, strength: float, seq_len: int, device):
        self.name, self.strength = name, float(strength)
        self.t = getattr(autoda_transforms(), name)(cond_dim=8, n_channels=1, seq_len=seq_len, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.full((x.shape[0], x.shape[1], 1), self.strength, dtype=x.dtype, device=x.device)
        out, _, _ = self.t(x, None, None, None, s)
        if out.shape != x.shape:
            raise AssertionError("%s changed the shape %s -> %s" % (self.name, tuple(x.shape), tuple(out.shape)))
        return out.to(x.dtype)

    def describe(self) -> dict:
        return {"op": self.name, "strength": self.strength, "source": "AutoDA-Timeseries basic_transforms.py @ " + AUTODA_COMMIT[:7]}


# ------------------------------------------------------------------------------------------------ frozen classification program space
# DEV-CLS-AUG-OFFLINE-SKILL (2026-09-25): four operators at frozen strengths; a program is None, one operator, or two different operators
# executed in the frozen canonical order amplitude (Scaling, MagWarp) -> time (WSlice) -> noise (Jitter); 11 programs in total.
#   Jitter   official AutoDA Jitter, s = 0.03 (x + N(0, 0.03^2) per point; identical to Iwana & Uchida jitter(sigma=0.03))
#   Scaling  Iwana & Uchida 2021 scaling(sigma=0.1): one factor N(1, 0.1) per series (Torch re-implementation)
#   MagWarp  Iwana & Uchida 2021 magnitude_warp(sigma=0.2, knot=4): x * CubicSpline(linspace(0, T-1, 6), N(1, 0.2))(0..T-1); scipy's
#            not-a-knot spline is linear in the knot values, so the curve is (warps @ B^T) with B precomputed from scipy (Torch re-implementation)
#   WSlice   official AutoDA WindowSliceWarp, s = 0.2 (random-start 90% window stretched back to T; = Iwana window_slice(0.9) semantics)
OP_ORDER = ("Scaling", "MagWarp", "WSlice", "Jitter")
OP_SOURCE = {"Jitter": "AutoDA-Timeseries basic_transforms.Jitter @ 91dbf70, strength 0.03",
             "Scaling": "Iwana & Uchida 2021 utils/augmentation.py scaling(sigma=0.1), Torch re-implementation",
             "MagWarp": "Iwana & Uchida 2021 utils/augmentation.py magnitude_warp(sigma=0.2, knot=4), Torch re-implementation",
             "WSlice": "AutoDA-Timeseries basic_transforms.WindowSliceWarp @ 91dbf70, strength 0.2"}
PROGRAMS = ("None",) + OP_ORDER + tuple("%s+%s" % (a, b) for i, a in enumerate(OP_ORDER) for b in OP_ORDER[i + 1:])
P_PROGRAM = 0.5


def spline_basis(T: int, knot: int = 4) -> np.ndarray:
    from scipy.interpolate import CubicSpline
    xs = np.linspace(0, T - 1.0, num=knot + 2)
    eye = np.eye(knot + 2)
    return np.stack([CubicSpline(xs, eye[j])(np.arange(T)) for j in range(knot + 2)], axis=1)      # (T, knot+2)


class _Scaling:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, x):
        return x * (1.0 + self.sigma * torch.randn(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype))


class _MagWarp:
    def __init__(self, T, device, sigma=0.2, knot=4):
        self.sigma, self.knot = sigma, knot
        self.B = torch.as_tensor(spline_basis(T, knot), dtype=torch.float32, device=device)       # (T, K)

    def __call__(self, x):
        w = 1.0 + self.sigma * torch.randn(x.shape[0] * x.shape[1], self.knot + 2, device=x.device, dtype=x.dtype)
        return x * (w @ self.B.T).reshape(x.shape)


class Program:
    def __init__(self, name: str, T: int, device):
        self.name = name
        ops = [] if name == "None" else name.split("+")
        if any(o not in OP_ORDER for o in ops) or ops != sorted(ops, key=OP_ORDER.index) or len(set(ops)) != len(ops):
            raise ValueError("illegal program %r" % name)
        make = {"Jitter": lambda: AutoDAOp("Jitter", 0.03, T, device), "WSlice": lambda: AutoDAOp("WindowSliceWarp", 0.2, T, device),
                "Scaling": lambda: _Scaling(), "MagWarp": lambda: _MagWarp(T, device)}
        self.ops = [make[o]() for o in ops]

    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x

    @property
    def is_identity(self):
        return not self.ops

    def describe(self) -> dict:
        ops = [] if self.name == "None" else self.name.split("+")
        return {"program": self.name, "ops": [{"op": o, "source": OP_SOURCE[o]} for o in ops], "p": 0.0 if not ops else P_PROGRAM}


def stratified_folds(classes: np.ndarray, k: int = 3, seed: int = 2026092501) -> np.ndarray:
    """Fold id per TRAIN row: per class, seeded permutation, round-robin over k folds (frozen, shared by every arm and seed)."""
    rng = np.random.default_rng(seed)
    fold = np.empty(classes.size, dtype=np.int64)
    for c in sorted(set(classes.tolist())):
        idx = np.flatnonzero(classes == c)
        idx = idx[rng.permutation(idx.size)]
        fold[idx] = np.arange(idx.size) % k
    return fold
