"""Torch-free part of the classification-augmentation wiring (DEV-CLS-AUG-WIRING): UCR TRAIN parsers (raw values), per-series
z-normalization, seeded per-class 2:1 fit/feedback split, metrics (numpy; identical to sklearn macro-F1 / accuracy / recall with
zero_division=0). Only TRAIN members are ever opened:
  - a per-dataset zip from timeseriesclassification.com: <Name>_TRAIN.ts
  - the official UCR archive UCRArchive_2018.zip (www.cs.ucr.edu/~eamonn/time_series_data_2018, password published on the archive
    page): UCRArchive_2018/<Name>/<Name>_TRAIN.tsv (label first, tab-separated; NaN pads variable-length rows)."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

SPLIT_SEED = 2026092500
UCR_ARCHIVE_PASSWORD = b"someone"


def train_member(archive: Path, name: str) -> str:
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
    if Path(archive).name.startswith("UCRArchive_2018"):
        want = "UCRArchive_2018/%s/%s_TRAIN.tsv" % (name, name)
        hits = [n for n in names if n == want]
    else:
        want = (name + "_TRAIN.ts").lower()
        hits = [n for n in names if n.lower().endswith("/" + want) or n.lower() == want]
    if len(hits) != 1:
        raise ValueError("%s: expected exactly one %s member, found %s" % (Path(archive).name, want, hits))
    return hits[0]


def _finish(name, archive, member, rows, labels, multi) -> dict:
    lengths = sorted({r.size for r in rows})
    equal = len(lengths) == 1
    values = np.vstack(rows) if equal else None
    names_ = tuple(sorted(set(labels), key=lambda t: (len(t), t)))
    classes = np.array([names_.index(t) for t in labels], dtype=np.int64)
    counts = np.bincount(classes, minlength=len(names_))
    has_nan = bool(values is not None and np.isnan(values).any())
    trailing_nan_rows = int(sum(1 for r in rows if r.size and np.isnan(r[-1])))
    return {"name": name, "archive": str(archive), "member": member, "values": values, "labels": labels, "label_names": names_,
            "classes": classes, "n": len(rows), "length": lengths[0] if equal else lengths, "equal_length": equal and trailing_nan_rows == 0,
            "univariate": multi == 0, "has_nan": has_nan, "trailing_nan_rows": trailing_nan_rows,
            "class_counts": counts.tolist(), "min_class": int(counts.min())}


def read_ucr_train(archive: Path, name: str) -> dict:
    """Parse the univariate UCR TRAIN member only. Returns raw values (NaN where missing), string labels and diagnostics."""
    member = train_member(archive, name)
    if not (member.endswith("_TRAIN.tsv") or member.lower().endswith("_train.ts")):
        raise AssertionError("refusing to open a non-TRAIN member")
    tsv = member.endswith(".tsv")
    with zipfile.ZipFile(archive) as z:
        raw = z.read(member, pwd=UCR_ARCHIVE_PASSWORD if tsv else None)
    rows, labels, multi = [], [], 0
    text = raw.decode("utf-8-sig", "replace").splitlines()
    if tsv:
        for line in text:
            s = line.strip()
            if not s:
                continue
            f = s.split("\t")
            labels.append(f[0].strip())
            rows.append(np.array([float("nan") if t.strip() in ("NaN", "nan", "") else float(t) for t in f[1:]], dtype=np.float64))
        return _finish(name, archive, member, rows, labels, 0)
    in_data = False
    for line in text:
        s = line.strip()
        if not in_data:
            in_data = s.lower() == "@data"
            continue
        if not s or s.startswith("#"):
            continue
        parts = s.split(":")
        if len(parts) > 2:
            multi += 1
        vals, lab = ":".join(parts[:-1]), parts[-1].strip()
        rows.append(np.array([float("nan") if t.strip() in ("?", "NaN", "nan", "") else float(t) for t in vals.split(",")], dtype=np.float64))
        labels.append(lab)
    return _finish(name, archive, member, rows, labels, multi)


def znorm(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    sd = v.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (v - v.mean(axis=1, keepdims=True)) / sd


def split_fit_feedback(classes: np.ndarray, seed: int = SPLIT_SEED) -> dict:
    rng = np.random.default_rng(seed)
    fit, fb, per = [], [], {}
    for c in sorted(set(classes.tolist())):
        idx = np.flatnonzero(classes == c)
        idx = idx[rng.permutation(idx.size)]
        n_fb = idx.size // 3
        fb.extend(idx[:n_fb].tolist())
        fit.extend(idx[n_fb:].tolist())
        per[int(c)] = {"fit": int(idx.size - n_fb), "feedback": int(n_fb)}
    gate = all(v["fit"] >= 20 and v["feedback"] >= 10 for v in per.values())
    return {"fit": np.array(sorted(fit)), "feedback": np.array(sorted(fb)), "per_class": per, "gate_fit20_fb10": gate, "seed": seed}


def metrics(y_true, y_pred, n_classes: int) -> dict:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    f1, rec = [], []
    for c in range(n_classes):
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        rec.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return {"macro_f1": float(np.mean(f1)), "accuracy": float((yt == yp).mean()), "worst_class_recall": float(np.min(rec)),
            "per_class_recall": [float(r) for r in rec]}
