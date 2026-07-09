"""data/load_ecg5000.py — ECG5000 真实分类锚（UCR 档案，非 Monash）。

ECG5000 = 5000 条心拍（每条长 140），5 类（不同心拍形态）。形态类靠**尖峰/形状**（含高频
判别特征）区分 → 与 forecast 的"平滑压噪"最优**真冲突**：清洗强度过高会抹掉判别尖峰。这正是
classify 给出"第二条跨任务 C1 证据"所需的真实信号（不同于合成 classify 的类=周期低频鲁棒特征）。

两个来源（按可得性自动选择，接口不变）：
  • **UCR 5-class**（优先）：`timeseriesclassification.com/aeon-toolkit/ECG5000.zip`，zip 内
    `ECG5000_{TRAIN,TEST}.txt` 每行空白分隔：首列=类标签(1..5)，其余 140=序列。类极不均衡
    （合并 ≈ 类1:2919 / 类2:1767 / 类3:96 / 类4:194 / 类5:24）→ `load_ecg5000` 取 top-K 高频类
    + 每类 cap 平衡，防 CV 折退化。
  • **TF 二分类回退**：`storage.googleapis.com/.../ecg.csv`（4998×141，末列=label，1=正常/0=异常，
    即 ECG5000 把 5 类折叠成 正常 vs 异常）。当 UCR 服务器不可达时自动启用；balanced（2919/2079）、
    morphology 判别同样成立 C1（异常心拍靠形状偏移区分 → 过度平滑抹掉判别特征）。
下载后合并、缓存为本地 npz（`_artifacts/ecg5000.npz`，含 `source` 标签），后续离线复用。
"""
from __future__ import annotations

import io
import pathlib
import urllib.request
import zipfile
from typing import Optional, Tuple

import numpy as np

_ARTIFACTS = pathlib.Path(__file__).resolve().parent / "_artifacts"
DEFAULT_CACHE = _ARTIFACTS / "ecg5000.npz"
_UCR_URLS = (
    "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip",
    "http://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip",
)
_TF_CSV = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"


# ════════════════════════════ 来源 1：UCR 5-class zip ════════════════════════════
def _parse_txt(raw: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """UCR .txt：每行首列=label，其余=series。返回 (X (n,L), y (n,))。"""
    arr = np.loadtxt(io.StringIO(raw.decode("utf-8")))
    return arr[:, 1:].astype(float), arr[:, 0].astype(float)


def _fetch_ucr() -> Tuple[np.ndarray, np.ndarray, str]:
    last = None
    for url in _UCR_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            zb = urllib.request.urlopen(req, timeout=60).read()
            Xs, ys = [], []
            with zipfile.ZipFile(io.BytesIO(zb)) as zf:
                for split in ("TRAIN", "TEST"):
                    name = next((n for n in zf.namelist()
                                 if n.upper().endswith(f"ECG5000_{split}.TXT")), None)
                    if name is None:
                        raise RuntimeError(f"zip 内缺 ECG5000_{split}.txt（含 {zf.namelist()}）")
                    X, y = _parse_txt(zf.read(name))
                    Xs.append(X); ys.append(y)
            return np.concatenate(Xs, 0), np.concatenate(ys, 0).astype(int), "ucr_5class"
        except Exception as exc:  # noqa: BLE001 — 试下一个 URL / 最终回退
            last = exc
    raise RuntimeError(f"UCR ECG5000 不可达：{type(last).__name__}: {last}")


# ════════════════════════════ 来源 2：TF 二分类 csv（回退）════════════════════════════
def _fetch_tf() -> Tuple[np.ndarray, np.ndarray, str]:
    raw = urllib.request.urlopen(_TF_CSV, timeout=60).read()
    arr = np.loadtxt(io.StringIO(raw.decode("utf-8")), delimiter=",")
    return arr[:, :-1].astype(float), arr[:, -1].astype(int), "tf_binary"


def build_cache(cache_path: Optional[str] = None) -> str:
    """获取 ECG5000（优先 UCR 5-class，宕机回退 TF 二分类）→ 合并存 npz。返回缓存路径。"""
    cache = pathlib.Path(cache_path) if cache_path else DEFAULT_CACHE
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        X, y, source = _fetch_ucr()
    except Exception as exc:  # noqa: BLE001
        print(f"[ecg5000] UCR 源失败 → 回退 TF 二分类 csv（{type(exc).__name__}）")
        X, y, source = _fetch_tf()
    np.savez(cache, X=X.astype(np.float32), y=y.astype(int), source=source)
    print(f"[ecg5000] source={source}  X={X.shape}")
    return str(cache)


# ════════════════════════════ 加载（带类选择/平衡）════════════════════════════
def _zscore_rows(X: np.ndarray) -> np.ndarray:
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    return (X - m) / np.where(s > 1e-9, s, 1.0)


def load_ecg5000(cache_path: Optional[str] = None, *, top_k: int = 3,
                 cap_per_class: Optional[int] = None, standardize: bool = True,
                 seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (X (n,140), y0 (n,))，y0 为 0-index 重映射后的类标签。

    top_k        : 仅保留计数最高的 K 个原始类（防 CV 折退化；默认 3）。
    cap_per_class: 每类最多取多少条（None → 取所选类的最小计数 → 均衡）。
    standardize  : 逐序列 z-score（与 load_real 真实锚一致；冻结/随机模型尺度对齐所需）。
    """
    cache = pathlib.Path(cache_path) if cache_path else DEFAULT_CACHE
    if not cache.exists():
        build_cache(str(cache))
    with np.load(cache) as d:
        X, y = d["X"].astype(float), d["y"].astype(int)
    if standardize:
        X = _zscore_rows(X)

    classes, counts = np.unique(y, return_counts=True)
    keep = classes[np.argsort(counts)[::-1][:top_k]]            # top-K 高频类
    cap = cap_per_class if cap_per_class is not None else int(counts[np.isin(classes, keep)].min())

    rng = np.random.default_rng(seed)
    sel_idx, remap = [], {c: i for i, c in enumerate(sorted(keep.tolist()))}
    for c in keep:
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        sel_idx.append(ci[:cap])
    sel = np.concatenate(sel_idx)
    rng.shuffle(sel)
    return X[sel], np.array([remap[c] for c in y[sel]], dtype=int)


if __name__ == "__main__":
    import collections
    path = build_cache()
    X, y = load_ecg5000()
    print(f"cached → {path}")
    print(f"loaded X={X.shape} y classes={dict(collections.Counter(y.tolist()))}")
