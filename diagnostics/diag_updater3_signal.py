"""diag_updater3_signal.py — updater v3 FAIL 诊断（prereg_updater3.md §6 FAIL 分支执行）。

问题：合法响应签名（史内单切点 48 点退化伪未来）到底携带多少真响应信息？
  ① 每维签名 vs oracle 响应（L_test[probe]−L_test[v_none]，**仅评估用**）的 Spearman/Pearson；
  ② 签名空间族内/族间散布比（操纵检查 0.350 的几何解释）；
  ③ oracle 响应本身的族可分性（上界：若 oracle 响应都分不开族，方向本身死；若 oracle 可分
     而合法估计不可分，死因=估计噪声，即"合法信息预算 < 机制识别所需信噪比"）。

只读既有产物（signatures.json + records_s2.jsonl），不跑新臂、不改阈值。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from SelfEvolvingHarnessTS.run_updater3 import OUT, PROBES, REC_PATH

def loo_1nn_acc(Z: np.ndarray, fam: list) -> float:
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Zz = (Z - mu) / sd
    d2 = ((Zz[:, None, :] - Zz[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(axis=1)
    return float(np.mean([fam[i] == fam[int(j)] for i, j in enumerate(nn)]))


def scatter_ratio(Z: np.ndarray, fam: list) -> float:
    """族内均方散布 / 全局均方散布（越接近 1 → 族不携带几何结构）。"""
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Zz = (Z - mu) / sd
    fams = sorted(set(fam))
    within = np.mean([np.var(Zz[[f == g for f in fam]], axis=0).sum() for g in fams])
    total = np.var(Zz, axis=0).sum()
    return float(within / total)


def main():
    sig = json.loads((OUT / "signatures.json").read_text("utf-8"))["sig"]
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in recs}
    uids = sorted(u for u in by_uid if sig.get(u) is not None)
    fam = [u.split(":")[1] for u in uids]
    Zs = np.array([sig[u] for u in uids], float)
    # oracle 响应（仅评估）：真 L_test 差值，同 3 维
    Zo = np.array([[by_uid[u]["L_test"][a] - by_uid[u]["L_test"]["v_none"]
                    for a in PROBES[1:]] for u in uids], float)
    out = dict(n=len(uids), probes=list(PROBES[1:]))
    out["per_dim"] = {}
    for k, a in enumerate(PROBES[1:]):
        sp = spearmanr(Zs[:, k], Zo[:, k])
        pe = pearsonr(Zs[:, k], Zo[:, k])
        out["per_dim"][a] = dict(spearman=round(float(sp.statistic), 4),
                                 spearman_p=float(sp.pvalue),
                                 pearson=round(float(pe.statistic), 4),
                                 sig_std=round(float(Zs[:, k].std()), 4),
                                 oracle_std=round(float(Zo[:, k].std()), 4))
    # 全响应上界：9 维 oracle Δ（全动作池 vs v_none）——排除"探针子集选错"解释
    acts_all = [a for a in recs[0]["L_test"] if a != "v_none"]
    Zo9 = np.array([[by_uid[u]["L_test"][a] - by_uid[u]["L_test"]["v_none"]
                     for a in acts_all] for u in uids], float)
    out["family_separability"] = dict(
        acc_1nn_signature=round(loo_1nn_acc(Zs, fam), 4),
        acc_1nn_oracle_response=round(loo_1nn_acc(Zo, fam), 4),
        acc_1nn_oracle_response_full9=round(loo_1nn_acc(Zo9, fam), 4),
        scatter_ratio_signature=round(scatter_ratio(Zs, fam), 4),
        scatter_ratio_oracle=round(scatter_ratio(Zo, fam), 4),
        scatter_ratio_oracle_full9=round(scatter_ratio(Zo9, fam), 4),
        note="oracle 响应=评估专用上界；策略输入非法（prereg §0.1）")
    (OUT / "diag_signal.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
