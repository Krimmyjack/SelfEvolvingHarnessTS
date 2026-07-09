"""run_updater2.py — Stage 2.5：updater v2 三臂（prereg_l5_updater2.md §3，第四张设计决策表）。

问题：OOD-aware 更新能否**既避免首遇伤害，又吃到真实适应 headroom**（S2-适应策略 ≈0.276 vs
冻结 ≈0.370——v1 因首块伤害没吃到的那 ~0.09）。

三臂同流同账本（16 半块 × 5 预锁排列；机制层已由切片验收 → 全臂账本重放，声明）：
  frozen      P0+abstain，永不更新
  updater_v1  朴素：验证门过即部署（失败机制对照，规则=切片版）
  updater_v2  canary 影子块（吸收冷启动）+ per-uid 支持域混合服务（in→候选，out→frozen）
              + 同验证门/回滚
族真标签只用于评估分组（first-encounter / recurrence），禁入任何策略输入。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_updater2
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .e32_policy import GBDTArm, PolicyData
from .policy import FrozenArmRouterPolicy
from .run_harness_slice import _eval_rows, _is_val, propose_update
from .s2_corpus import S2_FAMILIES

STAGE2 = Path(__file__).resolve().parent / "results" / "Stage2"
REC_PATH = STAGE2 / "S2_replication" / "records_s2.jsonl"
OUT = STAGE2 / "Updater2"
SEED = 20260705
N_PERMS = 5
DELTA_SAFE = 0.05
ARMS3 = ("frozen", "updater_v1", "updater_v2")


# ════════════════════════════ 流构造（16 半块 × 5 预锁排列）════════════════════════════
def half_of(uid: str) -> int:
    return int(hashlib.sha256((uid + "|half").encode()).hexdigest()[:8], 16) % 2


def locked_permutations() -> List[List[Tuple[str, int]]]:
    blocks = [(f, h) for f in S2_FAMILIES for h in (0, 1)]
    perms: List[List[Tuple[str, int]]] = []
    k = 0
    while len(perms) < N_PERMS:
        rng = np.random.default_rng(20260705 + k)
        k += 1
        order = [blocks[i] for i in rng.permutation(len(blocks))]
        pos: Dict[str, List[int]] = {}
        for i, (f, _) in enumerate(order):
            pos.setdefault(f, []).append(i)
        n_spread = sum(1 for f in pos if abs(pos[f][0] - pos[f][1]) >= 3)
        if n_spread >= 4:                                    # ≥4 族两半块间距 ≥3（复现测得到）
            perms.append(order)
    return perms


# ════════════════════════════ 策略包装（账本口径）════════════════════════════
class Served:
    """统一"当前服务策略"：picks(rows) → 每 uid 的动作 id。"""

    def __init__(self, kind: str, arm, actions: List[str], fit_uids: Optional[List[str]] = None,
                 support: Optional[dict] = None):
        self.kind, self.arm, self.actions = kind, arm, list(actions)
        self.fit_uids, self.support = fit_uids or [], support

    def picks(self, rows: List[dict]) -> List[str]:
        data = PolicyData(uids=[r["uid"] for r in rows], actions=self.actions,
                          L=np.zeros((len(rows), len(self.actions))),
                          X_d=np.array([[r["snr"], r["miss_rate"]] for r in rows]),
                          X_p=np.array([r["X_p"] for r in rows]),
                          cell=np.array([r["cell"] for r in rows]),
                          origin=np.array(["?"] * len(rows)))       # 族标签禁入
        p, _ = self.arm.picks(data, np.arange(data.n))
        return [self.actions[int(i)] for i in p]


def build_support(fit_rows: List[dict]) -> dict:
    Z_raw = np.array([[r["snr"], r["miss_rate"], *r["X_p"]] for r in fit_rows], float)
    mu, sd = Z_raw.mean(axis=0), Z_raw.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (Z_raw - mu) / sd
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return dict(mu=mu, sd=sd, Z=Z, thr=float(np.percentile(np.sqrt(d2.min(axis=1)), 95)))


def in_support(sup: dict, r: dict) -> bool:
    z = (np.array([r["snr"], r["miss_rate"], *r["X_p"]]) - sup["mu"]) / sup["sd"]
    return float(np.sqrt(((sup["Z"] - z) ** 2).sum(axis=1).min())) <= sup["thr"]


# ════════════════════════════ 单排列单臂流 ════════════════════════════
def run_stream(arm_name: str, order: List[Tuple[str, int]], by_uid: Dict[str, dict],
               blocks_uids: Dict[Tuple[str, int], List[str]], actions: List[str],
               frozen_srv: Served, verbose: bool = False) -> dict:
    srv: Served = frozen_srv                                  # 当前服务策略
    prev_srv: Served = frozen_srv
    shadow: Optional[dict] = None                             # {"srv":…, "fit_uids":…}
    seen: List[dict] = []
    ledger, events = [], []
    first_seen: set = set()
    for bi, (fam, hh) in enumerate(order):
        rows = [by_uid[u] for u in blocks_uids[(fam, hh)]]
        oracle = np.array([min(r["L_test"].values()) for r in rows])
        fr_picks = frozen_srv.picks(rows)
        # —— 当前臂服务 ——
        if arm_name == "frozen":
            picks = fr_picks
        elif arm_name == "updater_v1":
            picks = srv.picks(rows)
        else:                                                 # v2：支持域混合
            if srv.kind == "frozen":
                picks = fr_picks
            else:
                cand = srv.picks(rows)
                mask = [in_support(srv.support, r) for r in rows]
                picks = [c if m else f for c, f, m in zip(cand, fr_picks, mask)]
                events.append(dict(type="serve_mix", block=bi, in_support=int(sum(mask)), n=len(rows)))
        reg = np.array([r["L_test"][a] for r, a in zip(rows, picks)]) - oracle
        fr_reg = np.array([r["L_test"][a] for r, a in zip(rows, fr_picks)]) - oracle
        ledger.append(dict(block=bi, family=fam, half=hh,
                           first=fam not in first_seen,
                           regret=float(reg.mean()), frozen_regret=float(fr_reg.mean()),
                           served=srv.kind))
        first_seen.add(fam)
        # —— shadow 判定（v2）：整块反事实计账 ——
        if arm_name == "updater_v2" and shadow is not None:
            sh_picks = shadow["srv"].picks(rows)
            sh_reg = float(np.mean([r["L_test"][a] for r, a in zip(rows, sh_picks)]) - oracle.mean())
            if sh_reg <= ledger[-1]["regret"] + DELTA_SAFE:   # 参照=当前服务策略（prereg "incumbent"）
                prev_srv, srv = srv, shadow["srv"]
                events.append(dict(type="canary_activate", block=bi, version=shadow["srv"].kind))
            else:
                events.append(dict(type="canary_reject", block=bi,
                                   harm=round(sh_reg - float(fr_reg.mean()), 4)))
            shadow = None
        # —— 回滚（激活后的服务策略对 frozen 超预算）——
        if arm_name != "frozen" and srv.kind != "frozen" \
                and ledger[-1]["regret"] - ledger[-1]["frozen_regret"] > DELTA_SAFE:
            events.append(dict(type="rollback", block=bi))
            srv = prev_srv
        # —— 累积 + 提案 ——
        if arm_name == "frozen":
            continue
        seen.extend(rows)
        inc_arm = srv.arm
        prop = propose_update(seen, actions, inc_arm)
        if prop is None:
            continue
        events.append(dict(type="proposal", block=bi, accept=prop["accept"], kappa=prop["kappa"]))
        if prop["accept"]:
            fit_uids = [r["uid"] for r in seen if not _is_val(r["uid"])]
            new = Served(f"v@blk{bi}", prop["arm"], actions, fit_uids,
                         build_support([r for r in seen if not _is_val(r["uid"])]))
            if arm_name == "updater_v1":
                prev_srv, srv = srv, new                      # 朴素：立即部署
            else:
                shadow = dict(srv=new)                        # v2：先影子一块
        if verbose:
            print(f"      blk{bi:2d} {fam:14s}h{hh} 完成", flush=True)
    return dict(ledger=ledger, events=events)


# ════════════════════════════ 汇总 ════════════════════════════
def summarize(all_runs: Dict[str, List[dict]]) -> dict:
    out: dict = {"arms": {}, "criteria": {}}
    for arm in ARMS3:
        runs = all_runs[arm]
        cum, fu_harm, rec_gain, ttr = [], [], [], []
        fa = rb = cr = prop_acc = 0
        cov_num = cov_den = 0
        for run in runs:
            led = run["ledger"]
            w = np.array([1.0] * len(led))
            cum.append(float(np.mean([b["regret"] for b in led])))
            fu = [b["regret"] - b["frozen_regret"] for b in led if b["first"]]
            fu_harm.append(float(np.max(fu)) if fu else 0.0)
            rec = [b["frozen_regret"] - b["regret"] for b in led if not b["first"]]
            rec_gain.append(float(np.mean(rec)) if rec else 0.0)
            better = [b["regret"] <= b["frozen_regret"] + 1e-12 for b in led]
            t = next((i for i in range(len(led))
                      if all(better[i:])), len(led))
            ttr.append(t)
            rb += sum(1 for e in run["events"] if e["type"] == "rollback")
            cr += sum(1 for e in run["events"] if e["type"] == "canary_reject")
            prop_acc += sum(1 for e in run["events"] if e["type"] == "proposal" and e["accept"])
            for e in run["events"]:
                if e["type"] == "serve_mix":
                    cov_num += e["in_support"]
                    cov_den += e["n"]
            # false accept：v1=激活后下一块即回滚；v2=canary_reject 等价于拦截在影子期
            acts = [e["block"] for e in run["events"]
                    if e["type"] in ("proposal",) and e["accept"]] if arm == "updater_v1" else []
            rbs = {e["block"] for e in run["events"] if e["type"] == "rollback"}
            fa += sum(1 for b in acts if (b + 1) in rbs)
        served_frac = (cov_num / cov_den) if cov_den else (1.0 if arm == "updater_v1" else 0.0)
        out["arms"][arm] = dict(
            cumulative_regret_mean=float(np.mean(cum)), cumulative_regret_range=[float(min(cum)), float(max(cum))],
            first_unseen_harm_max_mean=float(np.mean(fu_harm)),
            recurrence_gain_mean=float(np.mean(rec_gain)),
            time_to_readiness_mean=float(np.mean(ttr)),
            accepts=prop_acc, rollbacks=rb, canary_rejects=cr, false_accepts_next_block=fa,
            update_coverage=float(served_frac))
    f, v1, v2 = (out["arms"][a] for a in ARMS3)
    out["criteria"] = dict(
        c1_cum_not_worse_than_frozen=bool(v2["cumulative_regret_mean"] <= f["cumulative_regret_mean"] + 1e-9),
        c2_first_unseen_harm_controlled=bool(v2["first_unseen_harm_max_mean"] < DELTA_SAFE),
        c3_fewer_failures_than_v1=bool(v2["rollbacks"] + v2["false_accepts_next_block"]
                                       < v1["rollbacks"] + v1["false_accepts_next_block"]),
        c4_nonzero_coverage=bool(v2["update_coverage"] > 0.0),
        c5_recurrence_beats_frozen=bool(v2["recurrence_gain_mean"] > 0.0),
        c6_ttr_shorter_than_v1=bool(v2["time_to_readiness_mean"] <= v1["time_to_readiness_mean"]))
    passed = sum(out["criteria"].values())
    out["criteria"]["decision"] = (
        "PASS：v2 成为确定性对照，LLM proposer(v3) 上场资格解锁" if passed == 6 else
        f"PARTIAL({passed}/6)：按分支细则判" if passed >= 4 else
        "FAIL：转 response-aware support/episodic memory（不调阈值）")
    return out


def render(res: dict, perms) -> str:
    lines = ["# updater v2 三臂（16 半块 × 5 预锁排列；prereg §3）", "",
             "| arm | cum regret (5 排列均值 [min,max]) | 首遇 harm(max 均值) | 复现增益 | TTR | "
             "accepts | canary-rej | rollbacks | false-acc | coverage |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS3:
        s = res["arms"][a]
        lines.append(f"| {a} | {s['cumulative_regret_mean']:.4f} "
                     f"[{s['cumulative_regret_range'][0]:.3f},{s['cumulative_regret_range'][1]:.3f}] | "
                     f"{s['first_unseen_harm_max_mean']:+.3f} | {s['recurrence_gain_mean']:+.4f} | "
                     f"{s['time_to_readiness_mean']:.1f} | {s['accepts']} | {s['canary_rejects']} | "
                     f"{s['rollbacks']} | {s['false_accepts_next_block']} | {s['update_coverage']:.2f} |")
    lines += ["", "判据：" + json.dumps({k: v for k, v in res["criteria"].items() if k != "decision"},
                                        ensure_ascii=False),
              f"**判决**：{res['criteria']['decision']}"]
    return "\n".join(lines) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true", help="只补一个缺失 (perm,arm) checkpoint 后退出")
    args = ap.parse_args()
    t0 = time.time()
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in recs}
    actions = list(recs[0]["L_test"].keys())
    blocks_uids = {}
    for r in recs:
        blocks_uids.setdefault((r["origin"], half_of(r["uid"])), []).append(r["uid"])
    for k in blocks_uids:
        blocks_uids[k] = sorted(blocks_uids[k])
    frozen_pol = FrozenArmRouterPolicy.load_frozen("dp_abstain")
    frozen_srv = Served("frozen", frozen_pol.arm, frozen_pol.actions)
    perms = locked_permutations()
    all_runs = {a: [] for a in ARMS3}
    ckdir = OUT / "ckpt"
    ckdir.mkdir(parents=True, exist_ok=True)
    done_one = False
    for pi, order in enumerate(perms):
        for arm in ARMS3:
            ck = ckdir / f"perm{pi}_{arm}.json"
            if ck.exists():
                all_runs[arm].append(json.loads(ck.read_text("utf-8")))
                continue
            if args.one and done_one:
                print(f"--one：本次已补 1 个，退出（缺 perm{pi}_{arm} 起）", flush=True)
                return
            print(f"  [perm {pi} × {arm}] 开跑…", flush=True)
            run = run_stream(arm, order, by_uid, blocks_uids, actions, frozen_srv, verbose=True)
            tmp = ck.with_suffix(".tmp")
            tmp.write_text(json.dumps(run, ensure_ascii=False), "utf-8")
            tmp.replace(ck)
            all_runs[arm].append(run)
            done_one = True
            print(f"  [perm {pi} × {arm}] checkpoint 落盘 [{time.time()-t0:.0f}s]", flush=True)
        print(f"  [perm {pi}] 三臂就绪 [{time.time()-t0:.0f}s]", flush=True)
    res = summarize(all_runs)
    res["perms"] = [[f"{f}:{h}" for f, h in p] for p in perms]
    res["config"] = dict(seed=SEED, delta_safe=DELTA_SAFE, n_perms=N_PERMS,
                         prereg="results/Stage2/prereg_l5_updater2.md §3",
                         note="全臂账本重放（机制层已由切片验收）；族标签只用于评估分组")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    table = render(res, perms)
    (OUT / "table.md").write_text(table, "utf-8")
    print("\n" + table, flush=True)
    print(f"产物：{OUT}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
