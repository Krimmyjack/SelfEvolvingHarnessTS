"""run_batch_scan.py — Track A **exploratory 结构扫描（描述性）**（§13.1 判据族②预览）。

**边界（读前必看）**：本扫描在**发现集** records 上做**描述性**批键分析——**不是转正证据、非门控、
不消费一次性资源**。用户第三十七轮想要的**新 namespace（S2R1_scan_20260707）扫描**须先为新语料
重建 L_test（confirmatory 级 compute）= 早上 turnkey 项；本文件是其廉价前瞻，回答"P1b 是否比
legacy/P0 更好的批键"的方向（response 同质性），供早上锁 confirmatory 参考。**结果不改判据**。

批键（把序列分组成 batch）三臂：
  legacy   四格 cell（task×snr-bin×miss-bin）——现任 E-3.2 分组
  P0       [snr,miss_rate,X_p] 10 维上 KMeans(K)
  P1b      featurized DataView 19 维上 KMeans(K)（bplus_features 缓存）
批语义 = "处理响应相似区域"（§13.1，非 family purity）→ 主指标 = **oracle-action 批内一致率**
（批内多少序列共享该批众数 oracle 动作）+ 批内 action-response 方差；family purity 只作旁证。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_batch_scan
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans

from .run_updater2 import REC_PATH

OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "BatchScan"
SEED = 20260707
K = 8                                              # 固定 = 族数（描述性，非调参）


def _homogeneity(groups: Dict[str, List[dict]], actions: List[str]) -> dict:
    """→ oracle 一致率（批内众数 oracle 占比，n 加权）+ 批内 response 方差 + family purity（旁证）。"""
    agree_num = agree_den = 0
    resp_var, purity_num = [], 0
    for g, rows in groups.items():
        if not rows:
            continue
        oracle = [min(r["L_test"], key=r["L_test"].get) for r in rows]
        modal = Counter(oracle).most_common(1)[0][1]
        agree_num += modal
        agree_den += len(oracle)
        L = np.array([[r["L_test"][a] for a in actions] for r in rows], float)
        resp_var.append(float(L.var(axis=0).mean()))       # 批内各动作损失方差均值
        fam = Counter(r["origin"] for r in rows).most_common(1)[0][1]
        purity_num += fam
    return dict(n_batches=len([g for g, r in groups.items() if r]),
                oracle_agreement=float(agree_num / agree_den) if agree_den else 0.0,
                within_batch_response_var=float(np.mean(resp_var)) if resp_var else 0.0,
                family_purity=float(purity_num / agree_den) if agree_den else 0.0)


def _kmeans_groups(recs: List[dict], feats: np.ndarray, k: int) -> Dict[str, List[dict]]:
    mu, sd = feats.mean(0), feats.std(0)
    sd[sd < 1e-12] = 1.0
    lab = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict((feats - mu) / sd)
    groups: Dict[str, List[dict]] = {}
    for r, c in zip(recs, lab):
        groups.setdefault(f"c{int(c)}", []).append(r)
    return groups


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    actions = list(recs[0]["L_test"].keys())
    p1b_path = OUT.parent / "SkillSliceV2" / "bplus_features.json"
    p1b = json.loads(p1b_path.read_text("utf-8"))

    # legacy：四格 cell
    legacy: Dict[str, List[dict]] = {}
    for r in recs:
        legacy.setdefault(r["cell"], []).append(r)
    # P0 / P1b：KMeans(K)
    X_p0 = np.array([[r["snr"], r["miss_rate"], *r["X_p"]] for r in recs], float)
    X_p1b = np.array([list(p1b[r["uid"]]["d"]) + list(p1b[r["uid"]]["p"]) for r in recs], float)

    keys = {"legacy_cell": legacy,
            "P0_kmeans": _kmeans_groups(recs, X_p0, K),
            "P1b_kmeans": _kmeans_groups(recs, X_p1b, K)}
    res = {name: _homogeneity(groups, actions) for name, groups in keys.items()}

    lines = ["# Track A exploratory 批键扫描（**描述性/发现集/非转正证据**；主指标=oracle 一致率↑）", "",
             "| batch 键 | #batches | oracle 一致率 | 批内 response 方差 | family purity(旁证) |",
             "|---|---|---|---|---|"]
    for name in ("legacy_cell", "P0_kmeans", "P1b_kmeans"):
        s = res[name]
        lines.append(f"| {name} | {s['n_batches']} | {s['oracle_agreement']:.3f} | "
                     f"{s['within_batch_response_var']:.4f} | {s['family_purity']:.3f} |")
    best = max(res, key=lambda n: res[n]["oracle_agreement"])
    lines += ["", f"**方向读数（非门控）**：oracle 一致率最高键 = **{best}**"
              f"（{res[best]['oracle_agreement']:.3f}）。响应同质性越高 = 批越"
              "\"处理响应相似\"。", "",
              "> 边界：发现集描述性扫描，**不锁 confirmatory**；新 namespace S2R1_scan_20260707 扫描"
              "（须重建 L_test）为早上 turnkey 项。K=8 固定=族数（非调参）。"]
    (OUT / "table.md").write_text("\n".join(lines) + "\n", "utf-8")
    (OUT / "report.json").write_text(json.dumps(dict(
        results=res, k=K, seed=SEED, n=len(recs),
        note="描述性/发现集/非转正证据；fresh-namespace scan=morning turnkey（须重建 L_test）"),
        ensure_ascii=False, indent=1), "utf-8")
    print("\n".join(lines), flush=True)
    print(f"\n产物：{OUT}", flush=True)


if __name__ == "__main__":
    main()
