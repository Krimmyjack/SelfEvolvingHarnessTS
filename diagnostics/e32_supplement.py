"""diagnostics/e32_supplement.py — E-3.2 补充统计（评审第十六轮，A-40 前置）。

只读 `results/E3_2/records_{scope}.jsonl`（per-uid 全动作 held-out loss + 各臂 pick），
**零重训**补报最终候选策略 dp_abstain 对三基线的直接 paired CI（原 report 只给了 dp_gbdt 的）。
写 `results/E3_2/supplement_dp_abstain_cis.json`。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.diagnostics.e32_supplement
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..e32_policy import paired_bootstrap_ci

E32 = Path(__file__).resolve().parent.parent / "results" / "E3_2"
SEED = 20260704
COMPARISONS = [("dp_abstain", "global"), ("dp_abstain", "d_lookup"), ("dp_abstain", "d_gbdt"),
               ("dp_abstain", "true_d_gbdt"), ("dp_abstain", "dp_gbdt")]


def regrets(records, arm):
    out = []
    for r in records:
        L = r["L_test"]
        out.append(L[r["arms"][arm]["pick"]] - min(L.values()))
    return np.array(out)


def dose_response(records):
    """评审十六轮共识点 4：固定 cell×origin 的严格 median 剂量反应（修 F0 的跨 cell 混杂——
    F0 的"harm 随选中窗单调"是 cell×SNR×选中窗同变的观察）。零成本：records 每 uid 含全动作 L_test。"""
    doses = ["v_median", "f0_median_w9", "f0_median_w15", "f0_median_w25"]
    by = {}
    for r in records:
        by.setdefault((r["cell"], r["origin"]), []).append(r["L_test"])
    out = {}
    print("\n===== 固定 cell×origin 的 median 剂量反应（mean loss；w5→w9→w15→w25）=====")
    for key in sorted(by):
        rows = by[key]
        means = [float(np.mean([x[d] for x in rows])) for d in doses]
        mono_up = all(means[i] <= means[i + 1] + 1e-9 for i in range(3))    # 单调变差（harm 随剂量升）
        mono_dn = all(means[i] >= means[i + 1] - 1e-9 for i in range(3))    # 单调变好
        tag = "harm↑单调" if mono_up else ("gain↑单调" if mono_dn else "非单调")
        out["|".join(key)] = dict(n=len(rows), mean_loss={d: m for d, m in zip(doses, means)},
                                  monotone_harm=mono_up, monotone_gain=mono_dn)
        print(f"  {key[0]:26s} {key[1]:9s} n={len(rows):3d}  " +
              "  ".join(f"{m:.3f}" for m in means) + f"  [{tag}]")
    return out


def main():
    result = {}
    dr = None
    for scope in ("primary_no_Sar", "all_data"):
        records = [json.loads(l) for l in
                   (E32 / f"records_{scope}.jsonl").read_text("utf-8").splitlines() if l.strip()]
        per_arm = {a: regrets(records, a)
                   for a in ("dp_abstain", "dp_gbdt", "global", "d_lookup", "d_gbdt", "true_d_gbdt")}
        result[scope] = {}
        print(f"===== {scope} (n={len(records)}) — dp_abstain 直接 paired CI（regret 差，负=dp_abstain 优）=====")
        for a, b in COMPARISONS:
            ci = paired_bootstrap_ci(per_arm[a], per_arm[b], n_boot=2000, seed=SEED)
            result[scope][f"{a}_vs_{b}"] = ci
            print(f"  {a} vs {b:12s} mean={ci['mean']:+.4f} CI[{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}] "
                  f"pos%={ci['frac_positive']:.3f}")
        if scope == "all_data":
            dr = dose_response(records)                       # 全语料（含 S_ar）口径的剂量反应
    out = E32 / "supplement_dp_abstain_cis.json"
    out.write_text(json.dumps(dict(
        note=("评审第十六轮：最终候选策略须有自己的主比较 CI（原 report 判据 (i)(ii)(vi) 用 dp_gbdt）。"
              "零重训（records 重算），caveat 同 A-39⑤：条件于已拟合头/router。seed=20260704, B=2000。"),
        cis=result), ensure_ascii=False, indent=1), "utf-8")
    (E32 / "dose_response_fixed_cell.json").write_text(json.dumps(dict(
        note=("评审十六轮共识点4：固定 cell×origin 比较 median@5/9/15/25 的严格剂量反应"
              "（修 F0 '随窗单调' 的跨 cell 混杂）；标签=E-3.2 nested outer-test L_test，all_data 口径。"),
        dose_response=dr), ensure_ascii=False, indent=1), "utf-8")
    print(f"→ {out}\n→ {E32 / 'dose_response_fixed_cell.json'}")


if __name__ == "__main__":
    main()
