"""run_calibrate_dstruct.py — 校准 BIN_DSTRUCT_TAU（方向 A 软结构门）。

对真实 Monash domain 流：按 (domain, cell_id) 聚 struct_feats，算
  • intra：同 (domain,cell) 内各序列到本 cell centroid 的 d_struct 分布（τ 须 > 此，别误拦合法复用）；
  • inter：**同 pattern_bin** 跨 domain 的 cell-centroid 两两 d_struct（τ 须 < 此，才能拦负迁移）。
打印两者分位，给出建议 τ。仅诊断，不改状态。
"""
from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.harness import HarnessState
from SelfEvolvingHarnessTS.fast_path.perceive import perceive
from SelfEvolvingHarnessTS.conditioning.distance import d_struct
from SelfEvolvingHarnessTS.conditioning.key import STRUCT_FEAT_NAMES
from SelfEvolvingHarnessTS.run_stream_s1 import real_domains
from SelfEvolvingHarnessTS.slow_path.batch_builder import cell_sample_from_raw_series


def _centroid(feats_list):
    return {k: float(np.median([f.get(k, 0.0) for f in feats_list])) for k in STRUCT_FEAT_NAMES}


def main():
    import sys
    npz = sys.argv[1] if len(sys.argv) > 1 else ""
    h = HarnessState.from_minimal()
    domains = real_domains(npz, n_per_signal=4, min_signals=5)
    print(f"domains (K={len(domains)}): {[d.name for d in domains]}\n")

    # (domain, cell_id) -> [struct_feats]
    groups: dict = {}
    for d in domains:
        for rs in d.corpus:
            cs = cell_sample_from_raw_series(rs)
            key = perceive(cs.raw, cs.task_type, h)
            groups.setdefault((d.name, key["cell_id"]), []).append(key["pattern"]["struct_feats"])

    centroids = {gk: _centroid(fl) for gk, fl in groups.items()}

    # intra: 各序列到本 cell centroid 的距离
    intra = []
    for gk, fl in groups.items():
        c = centroids[gk]
        intra += [d_struct(f, c) for f in fl]
    intra = np.array(intra)

    # inter: 同 pattern_bin 跨 domain 的 centroid 两两距离
    by_bin: dict = {}
    for (dom, cell_id), c in centroids.items():
        pb = cell_id.split("|", 1)[1]
        by_bin.setdefault(pb, []).append((dom, c))
    inter = []
    print("=== inter-domain centroid d_struct（同 pattern_bin）===")
    for pb, items in sorted(by_bin.items()):
        if len(items) < 2:
            continue
        print(f"[{pb}]")
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                dij = d_struct(items[i][1], items[j][1])
                inter.append(dij)
                print(f"    {items[i][0]:<16} ↔ {items[j][0]:<16} d_struct={dij:.3f}")
    inter = np.array(inter)

    def q(a, ps): return {p: round(float(np.quantile(a, p)), 3) for p in ps} if len(a) else {}
    print("\n=== 分布 ===")
    print(f"intra (n={len(intra)}): {q(intra,[0.5,0.9,0.95,1.0])}")
    print(f"inter (n={len(inter)}): min={inter.min():.3f} {q(inter,[0.1,0.25,0.5])}" if len(inter) else "inter: none")
    if len(intra) and len(inter):
        lo, hi = float(np.quantile(intra, 0.95)), float(inter.min())
        print(f"\n建议 τ ∈ ({lo:.3f}, {hi:.3f})  →  取中点 {0.5*(lo+hi):.3f}"
              if lo < hi else f"\n⚠ 重叠：intra p95={lo:.3f} ≥ inter min={hi:.3f}（结构门难两全，需更细特征/分位锚）")


if __name__ == "__main__":
    main()
