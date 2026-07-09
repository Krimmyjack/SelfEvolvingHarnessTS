"""run_mpilot.py — Track C0 M-pilot（§13.3 / §13.6）。

缓存 dev records 上重放**三确定性记忆臂** vs 现任 frozen 路由（dp_abstain），
**复发块 / 首遇块 + in-support / out-of-support 双分列**。离线、零新数据、无外部
API、无一次性资源、无 torch 拟合（frozen 路由推理 + numpy kNN）——探索性 pilot（可
重跑），**非 confirmatory**。

检索键可切换（`--key`）：
  p0    P0 特征 [snr, miss_rate, X_p]（10 维）——首轮结果 results/Stage2/MPilot/
  p1b   P1b featurized DataView（d 2 + p 17 = 19 维，来自 SkillSliceV2/bplus_features.json
        缓存）——判"是检索键的问题还是 episodic 机制本身死了"（§13.3 C1 门前置，用户第
        三十七轮批准 Track C 复跑）；结果 results/Stage2/MPilot_P1b/

预注册预测（§13.3）：memory 优势集中在**复发块 / in-support 序列**。p0 已证伪（in-sup adv
−0.010）；p1b 若仍无优势 → episodic 机制本身在本语料无增量（强结论）；若 p1b 现优势 → 键是
问题，直接指向 C1 用 P1b 键。

三臂 + 参照 updater_v2 与 --key p0 版一致；只换特征函数。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_mpilot [--key p0|p1b]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .policy import FrozenArmRouterPolicy
from .run_updater2 import (DELTA_SAFE, OUT as OUT2, REC_PATH, SEED, Served,
                           half_of, locked_permutations)

RESULTS = Path(__file__).resolve().parent / "results" / "Stage2"
K_VALUES = (3, 5, 10)
K_HEADLINE = 5
ARMS = ("frozen", "mem_knn", "router_fusion", "failure_veto")


# ════════════════════════════ 检索键（可切换特征函数）════════════════════════════
def _feat_p0(r: dict) -> np.ndarray:
    return np.array([r["snr"], r["miss_rate"], *r["X_p"]], float)          # 10 维


def make_featfn(key: str) -> Tuple[Callable[[dict], np.ndarray], str]:
    if key == "p0":
        return _feat_p0, "P0 [snr,miss_rate,X_p] 10d"
    if key == "p1b":
        path = RESULTS / "SkillSliceV2" / "bplus_features.json"
        if not path.exists():
            raise FileNotFoundError(f"缺 P1b 特征缓存 {path}（先跑 run_skill_slice_v2 或其 feat 步）")
        fmap = json.loads(path.read_text("utf-8"))
        cache: Dict[str, np.ndarray] = {u: np.array(list(v["d"]) + list(v["p"]), float)
                                        for u, v in fmap.items()}                # d2 + p17 = 19d

        def _feat_p1b(r: dict) -> np.ndarray:
            return cache[r["uid"]]
        return _feat_p1b, "P1b featurized DataView (d2+p17) 19d"
    raise ValueError(f"未知 key={key!r}")


# ── 支持域（同 updater 配方，特征函数化）──
def build_support_g(rows: List[dict], featfn) -> Optional[dict]:
    if len(rows) < 2:
        return None
    Z_raw = np.array([featfn(r) for r in rows], float)
    mu, sd = Z_raw.mean(axis=0), Z_raw.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (Z_raw - mu) / sd
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return dict(mu=mu, sd=sd, Z=Z, thr=float(np.percentile(np.sqrt(d2.min(axis=1)), 95)))


def in_support_g(sup: Optional[dict], r: dict, featfn) -> bool:
    if sup is None:
        return False
    z = (featfn(r) - sup["mu"]) / sup["sd"]
    return float(np.sqrt(((sup["Z"] - z) ** 2).sum(axis=1).min())) <= sup["thr"]


# ════════════════════════════ 情景记忆 ════════════════════════════
class Memory:
    def __init__(self, featfn):
        self.featfn = featfn
        self.rows: List[dict] = []
        self._F: List[np.ndarray] = []
        self._mu = self._sd = self._Z = None

    def add(self, rows: List[dict]) -> None:
        for r in rows:
            self.rows.append(r)
            self._F.append(self.featfn(r))

    def snapshot(self) -> None:
        if not self._F:
            return
        A = np.array(self._F, float)
        self._mu, self._sd = A.mean(axis=0), A.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        self._Z = (A - self._mu) / self._sd

    def episodic_pick(self, r: dict, actions: List[str], k: int) -> str:
        z = (self.featfn(r) - self._mu) / self._sd
        d = np.sqrt(((self._Z - z) ** 2).sum(axis=1))
        idx = np.argsort(d)[:k]
        M = np.array([[self.rows[i]["L_test"][a] for a in actions] for i in idx], float)
        return actions[int(M.mean(axis=0).argmin())]

    def ready(self, k: int) -> bool:
        return len(self.rows) >= k


# ════════════════════════════ 单臂单排列流 ════════════════════════════
def run_stream_mem(arm: str, k: int, featfn, order, by_uid, blocks_uids, actions, frozen_srv) -> dict:
    mem = Memory(featfn)
    fail_rows: List[dict] = []
    ledger: List[dict] = []
    buckets = {"in": [], "out": []}
    first_seen: set = set()
    for bi, (fam, hh) in enumerate(order):
        rows = [by_uid[u] for u in blocks_uids[(fam, hh)]]
        oracle = np.array([min(r["L_test"].values()) for r in rows])
        fr_picks = frozen_srv.picks(rows)
        sup = build_support_g(mem.rows, featfn)
        sup_fail = build_support_g(fail_rows, featfn)
        mem.snapshot()
        picks: List[str] = []
        for r, fp in zip(rows, fr_picks):
            in_sup = in_support_g(sup, r, featfn)
            ep = mem.episodic_pick(r, actions, k) if mem.ready(k) else fp
            if arm == "mem_knn":
                pick = ep
            elif arm == "router_fusion":
                pick = ep if (in_sup and mem.ready(k)) else fp
            elif arm == "failure_veto":
                near_fail = in_support_g(sup_fail, r, featfn)
                pick = ep if (near_fail and mem.ready(k)) else fp
            else:
                pick = fp
            picks.append(pick)
            buckets["in" if in_sup else "out"].append(
                (float(r["L_test"][fp] - r["L_test"][pick]), fam not in first_seen))
        reg = np.array([r["L_test"][a] for r, a in zip(rows, picks)]) - oracle
        fr_reg = np.array([r["L_test"][a] for r, a in zip(rows, fr_picks)]) - oracle
        ledger.append(dict(block=bi, family=fam, half=hh, first=fam not in first_seen,
                           regret=float(reg.mean()), frozen_regret=float(fr_reg.mean()),
                           n_override=int(sum(p != f for p, f in zip(picks, fr_picks))), n=len(rows)))
        first_seen.add(fam)
        for r, fp in zip(rows, fr_picks):
            if (r["L_test"][fp] - min(r["L_test"].values())) > DELTA_SAFE:
                fail_rows.append(r)
        mem.add(rows)
    return dict(ledger=ledger, buckets=buckets)


# ════════════════════════════ 汇总 / 渲染 ════════════════════════════
def _metrics_from_ledgers(runs: List[dict]) -> dict:
    cum, rec_gain, fu_harm, ov = [], [], [], []
    for run in runs:
        led = run["ledger"]
        cum.append(float(np.mean([b["regret"] for b in led])))
        rec = [b["frozen_regret"] - b["regret"] for b in led if not b["first"]]
        rec_gain.append(float(np.mean(rec)) if rec else 0.0)
        fu = [b["regret"] - b["frozen_regret"] for b in led if b["first"]]
        fu_harm.append(float(np.max(fu)) if fu else 0.0)
        ov.append(float(np.mean([b.get("n_override", 0) / max(1, b.get("n", 1)) for b in led])))
    return dict(cumulative_regret_mean=float(np.mean(cum)),
                cumulative_regret_range=[float(min(cum)), float(max(cum))],
                recurrence_gain_mean=float(np.mean(rec_gain)),
                first_unseen_harm_max_mean=float(np.mean(fu_harm)),
                override_frac_mean=float(np.mean(ov)))


def _bucket_stats(runs: List[dict]) -> dict:
    agg = {"in": [], "out": []}
    for run in runs:
        for key, items in run["buckets"].items():
            for adv, _ in items:
                agg[key].append(adv)
    return {k: dict(n=len(v), mean_adv=float(np.mean(v)) if v else 0.0) for k, v in agg.items()}


def summarize(all_runs, key_name: str) -> dict:
    out: dict = {"headline_k": K_HEADLINE, "retrieval_key": key_name, "arms": {}, "sensitivity_k": {}}
    for arm in ARMS:
        out["arms"][arm] = _metrics_from_ledgers(all_runs[arm][K_HEADLINE])
        out["arms"][arm]["buckets"] = _bucket_stats(all_runs[arm][K_HEADLINE])
    for k in K_VALUES:
        out["sensitivity_k"][k] = {a: _metrics_from_ledgers(all_runs[a][k])["cumulative_regret_mean"]
                                   for a in ARMS}
    v2 = [json.loads((OUT2 / "ckpt" / f"perm{pi}_updater_v2.json").read_text("utf-8"))
          for pi in range(len(locked_permutations()))
          if (OUT2 / "ckpt" / f"perm{pi}_updater_v2.json").exists()]
    if v2:
        out["reference_updater_v2"] = _metrics_from_ledgers(v2)
    best = max((a for a in ARMS if a != "frozen"),
               key=lambda a: out["arms"][a]["recurrence_gain_mean"])
    ba = out["arms"][best]["buckets"]
    out["pilot_readout"] = dict(
        best_memory_arm=best, recurrence_gain=out["arms"][best]["recurrence_gain_mean"],
        prediction_recurrence_positive=bool(out["arms"][best]["recurrence_gain_mean"] > 0.0),
        prediction_in_support_beats_out=bool(ba["in"]["mean_adv"] > ba["out"]["mean_adv"]),
        first_unseen_harm_controlled=bool(out["arms"][best]["first_unseen_harm_max_mean"] < DELTA_SAFE),
        in_support_adv=ba["in"]["mean_adv"], out_support_adv=ba["out"]["mean_adv"])
    return out


def render(res: dict) -> str:
    L = [f"# M-pilot（key={res['retrieval_key']}）：三记忆臂 vs frozen（16 半块×5 排列；headline k={res['headline_k']}）",
         "", "| arm | cum regret [min,max] | 复发增益 | 首遇 harm(max) | override% | in-sup adv | out-sup adv |",
         "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        s = res["arms"][a]
        b = s["buckets"]
        L.append(f"| {a} | {s['cumulative_regret_mean']:.4f} "
                 f"[{s['cumulative_regret_range'][0]:.3f},{s['cumulative_regret_range'][1]:.3f}] | "
                 f"{s['recurrence_gain_mean']:+.4f} | {s['first_unseen_harm_max_mean']:+.3f} | "
                 f"{s['override_frac_mean']:.2f} | {b['in']['mean_adv']:+.4f}(n{b['in']['n']}) | "
                 f"{b['out']['mean_adv']:+.4f}(n{b['out']['n']}) |")
    if "reference_updater_v2" in res:
        r = res["reference_updater_v2"]
        L.append(f"| _ref_ updater_v2 | {r['cumulative_regret_mean']:.4f} | "
                 f"{r['recurrence_gain_mean']:+.4f} | {r['first_unseen_harm_max_mean']:+.3f} | — | — | — |")
    pr = res["pilot_readout"]
    L += ["", f"**k 敏感性（cum regret）**：" + json.dumps(res["sensitivity_k"], ensure_ascii=False), "",
          f"**pilot 判读**（best={pr['best_memory_arm']}，非门控）：复发增益={pr['recurrence_gain']:+.4f} "
          f"→ 预测①(复发>0) **{pr['prediction_recurrence_positive']}**；"
          f"in-sup {pr['in_support_adv']:+.4f} vs out-sup {pr['out_support_adv']:+.4f} "
          f"→ 预测②(in>out) **{pr['prediction_in_support_beats_out']}**；"
          f"首遇 harm 受控 **{pr['first_unseen_harm_controlled']}**"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", choices=["p0", "p1b"], default="p0")
    args = ap.parse_args()
    t0 = time.time()
    featfn, key_name = make_featfn(args.key)
    OUT = RESULTS / ("MPilot" if args.key == "p0" else "MPilot_P1b")
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in recs}
    actions = list(recs[0]["L_test"].keys())
    blocks_uids: Dict[Tuple[str, int], List[str]] = {}
    for r in recs:
        blocks_uids.setdefault((r["origin"], half_of(r["uid"])), []).append(r["uid"])
    for kk in blocks_uids:
        blocks_uids[kk] = sorted(blocks_uids[kk])
    frozen_pol = FrozenArmRouterPolicy.load_frozen("dp_abstain")
    frozen_srv = Served("frozen", frozen_pol.arm, frozen_pol.actions)
    perms = locked_permutations()

    all_runs = {a: {k: [] for k in K_VALUES} for a in ARMS}
    for arm in ARMS:
        for k in K_VALUES:
            for order in perms:
                all_runs[arm][k].append(
                    run_stream_mem(arm, k, featfn, order, by_uid, blocks_uids, actions, frozen_srv))
        print(f"  [{arm}] 完成 {len(K_VALUES)}k × {len(perms)} 排列 [{time.time()-t0:.0f}s]", flush=True)

    res = summarize(all_runs, key_name)
    res["config"] = dict(seed=SEED, delta_safe=DELTA_SAFE, k_values=list(K_VALUES),
                         k_headline=K_HEADLINE, retrieval_key=key_name,
                         prereg="Component Plan §13.3", note="探索性 pilot，非 confirmatory；"
                         "族标签只用于评估分组；seen=严格过去块，current L_test 只事后计 regret")
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    table = render(res)
    (OUT / "table.md").write_text(table, "utf-8")
    print("\n" + table, flush=True)
    print(f"产物：{OUT}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
