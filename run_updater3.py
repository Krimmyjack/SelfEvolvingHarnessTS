"""run_updater3.py — Stage 2.5：updater v3 = response-aware support（prereg_updater3.md，
第五张设计决策表）。

问题：把支持域从 P0 特征空间换成**响应签名空间**（观测史内 rolling-origin 轻探针回测，
ΔnMAE vs v_none），能否修掉 v2 的 aliasing 首遇伤害（0.198 > δ_safe）并首次过 c1
（"更新优于不更新"）——单变量归因：v2 规则栈唯一改动 = 支持域空间。

边界（prereg §0）：签名 = 纯 `history[:CUT]` 函数（API 断言拒绝更长输入——L_test /
clean future 物理不可达）；探针硬帽 4 动作 × 1 切点 × 3 维；张量行不作签名。

三臂：frozen / updater_v2 = **逐字复用 Updater2/ckpt**（同流同账本，声明）；
updater_v3 = 本模块 run_stream_v3。守卫 G-A：v3 流实现换回 P0 空间须 bit 级复现
Updater2 perm0 的 v2 账本。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_updater3 [--one]
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .evaluators.base import H_FORECAST, L_WIN
from .evaluators.frozen_probe import FrozenProbe
from .evaluators.grounded_forecast import _build_windows_full
from .fast_path.pipeline import process as fast_process
from .policy import FrozenArmRouterPolicy
from .run_e32 import _variant_map
from .run_harness_slice import _is_val, propose_update
from .run_updater2 import (DELTA_SAFE, OUT as OUT2, REC_PATH, SEED, Served,
                           build_support, half_of, in_support, locked_permutations)
from .run_variance_decomp import CUT
from .s2_corpus import build_s2_dev

OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "Updater3"
PROBES = ("v_none", "v_median", "f0_median_w25", "v_stl")   # prereg §1：机制多样 4 探针，硬帽
SIG_DIM = len(PROBES) - 1                                    # 3（Δ vs v_none）
CUT_SIG = CUT - H_FORECAST                                   # 416：探针训练段
MIN_PF_OBS = 8                                               # 伪未来有效观测下限
COVERAGE_FLOOR = 0.30                                        # prereg §5 g4 预锁下限
ARMS = ("frozen", "updater_v2", "updater_v3")


# ════════════════════════════ 响应签名（prereg §1）════════════════════════════
def probe_signature(hist: np.ndarray, variants: dict, fp: FrozenProbe) -> Optional[List[float]]:
    """观测史内 rolling-origin 轻探针签名。返回 3 维 ΔnMAE 或 None（失败→保守 out-of-support）。"""
    hist = np.asarray(hist, float).ravel()
    assert hist.size == CUT, \
        f"G-B 泄漏守卫：签名只接受判官口径观测史（CUT={CUT}），got {hist.size}"
    tr, pf = hist[:CUT_SIG], hist[CUT_SIG:]
    m = np.isfinite(pf)
    if int(m.sum()) < MIN_PF_OBS:
        return None
    obs = tr[np.isfinite(tr)]
    scale = float(np.std(obs)) if obs.size else 0.0
    if scale < 1e-9:
        return None
    losses: Dict[str, float] = {}
    for a in PROBES:
        try:
            ready = fast_process(tr, "forecast", variants[a], store=None)[1]
            hh = np.asarray(ready, float).ravel()
            if not np.all(np.isfinite(hh)) or hh.size < L_WIN + H_FORECAST:
                return None
            X, Y, _ = _build_windows_full([hh], [24])
            if X is None or len(X) < 6:
                return None
            fp.fit(X, Y)
            yhat = np.asarray(fp.predict(hh[-L_WIN:].reshape(1, -1))[0], float)
            losses[a] = float(np.mean(np.abs(yhat[m] - pf[m])) / scale)
        except AssertionError:
            raise
        except Exception:
            return None
    return [losses[a] - losses["v_none"] for a in PROBES[1:]]


def compute_signatures(verbose: bool = True) -> Dict[str, Optional[List[float]]]:
    """全 dev 语料签名（确定性 → 一次缓存跨臂/排列复用；partial checkpoint 断点续）。"""
    path = OUT / "signatures.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))["sig"]
    part_path = OUT / "signatures_partial.json"
    sig: Dict[str, Optional[List[float]]] = (
        json.loads(part_path.read_text("utf-8")) if part_path.exists() else {})
    corpus = build_s2_dev()
    variants = _variant_map(list(PROBES))
    fp = FrozenProbe()
    t0 = time.time()
    for i, rs in enumerate(corpus):
        if rs.series_uid in sig:
            continue
        sig[rs.series_uid] = probe_signature(rs.history, variants, fp)
        if (i + 1) % 50 == 0:
            part_path.write_text(json.dumps(sig, ensure_ascii=False), "utf-8")
            if verbose:
                print(f"  [sig] {i+1}/{len(corpus)} [{time.time()-t0:.0f}s]", flush=True)
    n_bad = sum(1 for v in sig.values() if v is None)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(
        sig=sig, config=dict(probes=list(PROBES), cut_sig=CUT_SIG, min_pf_obs=MIN_PF_OBS,
                             metric="masked nMAE / std(observed train seg)",
                             prereg="results/Stage2/prereg_updater3.md §1"),
        n=len(sig), n_invalid=n_bad), ensure_ascii=False), "utf-8")
    if part_path.exists():
        part_path.unlink()
    if verbose:
        print(f"  [sig] 完成 {len(sig)} uid（无效 {n_bad}）[{time.time()-t0:.0f}s]", flush=True)
    return sig


# ════════════════════════════ 签名空间支持域（同配方，换空间）════════════════════════════
def build_support_sig(fit_rows: List[dict], sig: Dict[str, Optional[List[float]]]) -> Optional[dict]:
    Z_raw = np.array([sig[r["uid"]] for r in fit_rows if sig.get(r["uid"]) is not None], float)
    if len(Z_raw) < 2:
        return None                                          # 保守：支持域不可建 → 全 out
    mu, sd = Z_raw.mean(axis=0), Z_raw.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (Z_raw - mu) / sd
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return dict(mu=mu, sd=sd, Z=Z, thr=float(np.percentile(np.sqrt(d2.min(axis=1)), 95)))


def in_support_sig(sup: Optional[dict], r: dict, sig: Dict[str, Optional[List[float]]]) -> bool:
    s = sig.get(r["uid"])
    if sup is None or s is None:
        return False                                         # 签名无效 → 保守 out（prereg §1）
    z = (np.array(s, float) - sup["mu"]) / sup["sd"]
    return float(np.sqrt(((sup["Z"] - z) ** 2).sum(axis=1).min())) <= sup["thr"]


# ════════════════════════════ v3 流（镜像 v2 逻辑，唯一改动=支持域空间）════════════════════════════
def run_stream_v3(order: List[Tuple[str, int]], by_uid: Dict[str, dict],
                  blocks_uids: Dict[Tuple[str, int], List[str]], actions: List[str],
                  frozen_srv: Served, sig: Optional[dict] = None,
                  use_p0_space: bool = False, verbose: bool = False) -> dict:
    """use_p0_space=True → 守卫 G-A 模式（须 bit 级复现 Updater2 v2 账本）。"""
    assert use_p0_space or sig is not None
    srv: Served = frozen_srv
    prev_srv: Served = frozen_srv
    shadow: Optional[dict] = None
    seen: List[dict] = []
    ledger, events, diag, episodes = [], [], [], []
    first_seen: set = set()
    for bi, (fam, hh) in enumerate(order):
        rows = [by_uid[u] for u in blocks_uids[(fam, hh)]]
        oracle = np.array([min(r["L_test"].values()) for r in rows])
        fr_picks = frozen_srv.picks(rows)
        if srv.kind == "frozen":
            picks = fr_picks
        else:
            cand = srv.picks(rows)
            if use_p0_space:
                mask = [in_support(srv.support, r) for r in rows]
            else:
                mask = [in_support_sig(srv.support, r, sig) for r in rows]
            picks = [c if m else f for c, f, m in zip(cand, fr_picks, mask)]
            events.append(dict(type="serve_mix", block=bi, in_support=int(sum(mask)), n=len(rows)))
            diag.append(dict(block=bi, family=fam, first=fam not in first_seen,
                             in_support=int(sum(mask)), n=len(rows)))
        reg = np.array([r["L_test"][a] for r, a in zip(rows, picks)]) - oracle
        fr_reg = np.array([r["L_test"][a] for r, a in zip(rows, fr_picks)]) - oracle
        ledger.append(dict(block=bi, family=fam, half=hh,
                           first=fam not in first_seen,
                           regret=float(reg.mean()), frozen_regret=float(fr_reg.mean()),
                           served=srv.kind))
        first_seen.add(fam)
        if shadow is not None:
            sh_picks = shadow["srv"].picks(rows)
            sh_reg = float(np.mean([r["L_test"][a] for r, a in zip(rows, sh_picks)]) - oracle.mean())
            if sh_reg <= ledger[-1]["regret"] + DELTA_SAFE:
                prev_srv, srv = srv, shadow["srv"]
                events.append(dict(type="canary_activate", block=bi, version=shadow["srv"].kind))
                if not use_p0_space:
                    episodes.append(dict(activate_block=bi, version=shadow["srv"].kind,
                                         n_fit=len(shadow["srv"].fit_uids),
                                         sig_centroid=([float(x) for x in shadow["srv"].support["mu"]]
                                                       if shadow["srv"].support else None),
                                         thr=(float(shadow["srv"].support["thr"])
                                              if shadow["srv"].support else None),
                                         rollback_blocks=[]))
            else:
                events.append(dict(type="canary_reject", block=bi,
                                   harm=round(sh_reg - float(fr_reg.mean()), 4)))
            shadow = None
        if srv.kind != "frozen" \
                and ledger[-1]["regret"] - ledger[-1]["frozen_regret"] > DELTA_SAFE:
            events.append(dict(type="rollback", block=bi))
            if episodes:
                episodes[-1]["rollback_blocks"].append(bi)
            srv = prev_srv
        seen.extend(rows)
        inc_arm = srv.arm
        prop = propose_update(seen, actions, inc_arm)
        if prop is None:
            continue
        events.append(dict(type="proposal", block=bi, accept=prop["accept"], kappa=prop["kappa"]))
        if prop["accept"]:
            fit_rows = [r for r in seen if not _is_val(r["uid"])]
            sup = build_support(fit_rows) if use_p0_space else build_support_sig(fit_rows, sig)
            new = Served(f"v@blk{bi}", prop["arm"], actions, [r["uid"] for r in fit_rows], sup)
            shadow = dict(srv=new)
        if verbose:
            print(f"      blk{bi:2d} {fam:14s}h{hh} 完成", flush=True)
    return dict(ledger=ledger, events=events, diag=diag, episodes=episodes)


# ════════════════════════════ 守卫 G-A：结构重放 ════════════════════════════
def guard_replay_v2(order, by_uid, blocks_uids, actions, frozen_srv) -> None:
    gpath = OUT / "ckpt" / "guard_perm0.json"
    if gpath.exists():
        return
    ref = json.loads((OUT2 / "ckpt" / "perm0_updater_v2.json").read_text("utf-8"))
    print("  [G-A] v3 流 ×P0 空间重放 perm0 …", flush=True)
    t0 = time.time()
    run = run_stream_v3(order, by_uid, blocks_uids, actions, frozen_srv, use_p0_space=True)
    got = json.loads(json.dumps(dict(ledger=run["ledger"], events=run["events"])))
    want = dict(ledger=ref["ledger"], events=ref["events"])
    assert got == want, "守卫 G-A 失败：v3 流实现 ×P0 空间 ≢ Updater2 v2 账本——流逻辑漂移，禁止出表"
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(json.dumps(dict(passed=True, blocks=len(run["ledger"]),
                                     seconds=round(time.time() - t0, 1)), ensure_ascii=False), "utf-8")
    print(f"  [G-A] 过：bit 级一致（{len(run['ledger'])} 块）[{time.time()-t0:.0f}s]", flush=True)


# ════════════════════════════ 操纵检查（非门控，prereg §4）════════════════════════════
def manipulation_check(sig: Dict[str, Optional[List[float]]], by_uid: Dict[str, dict]) -> dict:
    def loo_1nn_acc(Z: np.ndarray, fam: List[str]) -> float:
        mu, sd = Z.mean(axis=0), Z.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Zz = (Z - mu) / sd
        d2 = ((Zz[:, None, :] - Zz[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = d2.argmin(axis=1)
        return float(np.mean([fam[i] == fam[int(j)] for i, j in enumerate(nn)]))

    uids = sorted(u for u in by_uid if sig.get(u) is not None)
    fam = [u.split(":")[1] for u in uids]
    Zs = np.array([sig[u] for u in uids], float)
    Zp = np.array([[by_uid[u]["snr"], by_uid[u]["miss_rate"], *by_uid[u]["X_p"]] for u in uids], float)
    return dict(n=len(uids), n_invalid=sum(1 for u in by_uid if sig.get(u) is None),
                acc_1nn_signature=loo_1nn_acc(Zs, fam), acc_1nn_p0=loo_1nn_acc(Zp, fam),
                note="族标签仅评估用；非门控描述（prereg §4）")


# ════════════════════════════ 汇总（三臂；判据 prereg §5）════════════════════════════
def _arm_metrics(runs: List[dict], arm: str) -> dict:
    cum, fu_harm, rec_gain, ttr = [], [], [], []
    rb = cr = prop_acc = 0
    cov_num = cov_den = 0
    cov_first_num = cov_first_den = cov_rec_num = cov_rec_den = 0
    for run in runs:
        led = run["ledger"]
        cum.append(float(np.mean([b["regret"] for b in led])))
        fu = [b["regret"] - b["frozen_regret"] for b in led if b["first"]]
        fu_harm.append(float(np.max(fu)) if fu else 0.0)
        rec = [b["frozen_regret"] - b["regret"] for b in led if not b["first"]]
        rec_gain.append(float(np.mean(rec)) if rec else 0.0)
        better = [b["regret"] <= b["frozen_regret"] + 1e-12 for b in led]
        ttr.append(next((i for i in range(len(led)) if all(better[i:])), len(led)))
        rb += sum(1 for e in run["events"] if e["type"] == "rollback")
        cr += sum(1 for e in run["events"] if e["type"] == "canary_reject")
        prop_acc += sum(1 for e in run["events"] if e["type"] == "proposal" and e["accept"])
        for e in run["events"]:
            if e["type"] == "serve_mix":
                cov_num += e["in_support"]
                cov_den += e["n"]
        for d in run.get("diag", []):
            if d["first"]:
                cov_first_num += d["in_support"]
                cov_first_den += d["n"]
            else:
                cov_rec_num += d["in_support"]
                cov_rec_den += d["n"]
    out = dict(cumulative_regret_mean=float(np.mean(cum)),
               cumulative_regret_range=[float(min(cum)), float(max(cum))],
               first_unseen_harm_max_mean=float(np.mean(fu_harm)),
               recurrence_gain_mean=float(np.mean(rec_gain)),
               time_to_readiness_mean=float(np.mean(ttr)),
               accepts=prop_acc, rollbacks=rb, canary_rejects=cr,
               false_accepts_next_block=0,                  # v2/v3 计账口径同 Updater2（canary 吸收）
               update_coverage=float(cov_num / cov_den) if cov_den else 0.0)
    if cov_first_den or cov_rec_den:
        out["coverage_first_encounter"] = float(cov_first_num / cov_first_den) if cov_first_den else 0.0
        out["coverage_recurrence"] = float(cov_rec_num / cov_rec_den) if cov_rec_den else 0.0
    return out


def summarize(all_runs: Dict[str, List[dict]]) -> dict:
    out: dict = {"arms": {a: _arm_metrics(all_runs[a], a) for a in ARMS}, "criteria": {}}
    f, v2, v3 = (out["arms"][a] for a in ARMS)
    out["criteria"] = dict(
        g1_cum_not_worse_than_frozen=bool(v3["cumulative_regret_mean"]
                                          <= f["cumulative_regret_mean"] + 1e-9),
        g2_first_unseen_harm_controlled=bool(v3["first_unseen_harm_max_mean"] < DELTA_SAFE),
        g3_recurrence_beats_frozen=bool(v3["recurrence_gain_mean"] > 0.0),
        g4_coverage_floor=bool(v3["update_coverage"] >= COVERAGE_FLOOR),
        g5_fewer_failures_than_v2=bool(v3["rollbacks"] + v3["false_accepts_next_block"]
                                       < v2["rollbacks"] + v2["false_accepts_next_block"]),
        g6_probe_budget_respected=True)                     # tests/test_updater3.py 钉死，表内申报
    passed = sum(out["criteria"].values())
    out["criteria"]["decision"] = (
        "PASS(6/6)：v3=确定性对照封顶，LLM proposer（阶梯第 4 级）资格解锁（仍须独立预注册）"
        if passed == 6 else
        f"PARTIAL({passed}/6)：按 prereg §6 分支判" if passed >= 4 else
        "FAIL：响应前提存疑，回操纵检查证据分析（不调阈值）")
    return out


def render(res: dict) -> str:
    lines = ["# updater v3 = response-aware support（16 半块 × 5 预锁排列；prereg_updater3.md）", "",
             "| arm | cum regret (5 排列均值 [min,max]) | 首遇 harm(max 均值) | 复现增益 | TTR | "
             "accepts | canary-rej | rollbacks | coverage | cov首遇/复现 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        s = res["arms"][a]
        cov_split = (f"{s['coverage_first_encounter']:.2f}/{s['coverage_recurrence']:.2f}"
                     if "coverage_first_encounter" in s else "—")
        lines.append(f"| {a} | {s['cumulative_regret_mean']:.4f} "
                     f"[{s['cumulative_regret_range'][0]:.3f},{s['cumulative_regret_range'][1]:.3f}] | "
                     f"{s['first_unseen_harm_max_mean']:+.3f} | {s['recurrence_gain_mean']:+.4f} | "
                     f"{s['time_to_readiness_mean']:.1f} | {s['accepts']} | {s['canary_rejects']} | "
                     f"{s['rollbacks']} | {s['update_coverage']:.2f} | {cov_split} |")
    mc = res.get("manipulation_check", {})
    lines += ["", "判据：" + json.dumps({k: v for k, v in res["criteria"].items() if k != "decision"},
                                        ensure_ascii=False),
              f"**判决**：{res['criteria']['decision']}", "",
              f"操纵检查（非门控）：1-NN 族分类 签名空间 {mc.get('acc_1nn_signature', float('nan')):.3f} "
              f"vs P0 空间 {mc.get('acc_1nn_p0', float('nan')):.3f}（n={mc.get('n')}，"
              f"无效签名 {mc.get('n_invalid')}）",
              "探针预算（独立预算行，prereg §0.3）：4 动作 × 1 切点 × 3 维签名/uid；"
              "无 regret 单位折算（目标函数无成本项，声明）。"]
    return "\n".join(lines) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true", help="只补一个缺失单元（签名/守卫/一个 perm）后退出")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in recs}
    actions = list(recs[0]["L_test"].keys())
    blocks_uids: Dict[Tuple[str, int], List[str]] = {}
    for r in recs:
        blocks_uids.setdefault((r["origin"], half_of(r["uid"])), []).append(r["uid"])
    for k in blocks_uids:
        blocks_uids[k] = sorted(blocks_uids[k])
    frozen_pol = FrozenArmRouterPolicy.load_frozen("dp_abstain")
    frozen_srv = Served("frozen", frozen_pol.arm, frozen_pol.actions)
    perms = locked_permutations()

    sig_done = (OUT / "signatures.json").exists()
    sig = compute_signatures()
    if args.one and not sig_done:
        print("--one：本次已补签名缓存，退出", flush=True)
        return
    guard_done = (OUT / "ckpt" / "guard_perm0.json").exists()
    guard_replay_v2(perms[0], by_uid, blocks_uids, actions, frozen_srv)
    if args.one and not guard_done:
        print("--one：本次已补守卫 G-A，退出", flush=True)
        return

    all_runs: Dict[str, List[dict]] = {a: [] for a in ARMS}
    ckdir = OUT / "ckpt"
    done_one = False
    for pi, order in enumerate(perms):
        for arm in ("frozen", "updater_v2"):                 # 逐字复用 Updater2 ckpt（声明）
            all_runs[arm].append(json.loads(
                (OUT2 / "ckpt" / f"perm{pi}_{arm}.json").read_text("utf-8")))
        ck = ckdir / f"perm{pi}_updater_v3.json"
        if ck.exists():
            all_runs["updater_v3"].append(json.loads(ck.read_text("utf-8")))
            continue
        if args.one and done_one:
            print(f"--one：本次已补 1 个，退出（缺 perm{pi} 起）", flush=True)
            return
        print(f"  [perm {pi} × updater_v3] 开跑…", flush=True)
        run = run_stream_v3(order, by_uid, blocks_uids, actions, frozen_srv, sig=sig, verbose=True)
        tmp = ck.with_suffix(".tmp")
        tmp.write_text(json.dumps(run, ensure_ascii=False), "utf-8")
        tmp.replace(ck)
        all_runs["updater_v3"].append(run)
        done_one = True
        print(f"  [perm {pi} × updater_v3] checkpoint 落盘 [{time.time()-t0:.0f}s]", flush=True)

    res = summarize(all_runs)
    res["manipulation_check"] = manipulation_check(sig, by_uid)
    res["perms"] = [[f"{f}:{h}" for f, h in p] for p in perms]
    res["config"] = dict(seed=SEED, delta_safe=DELTA_SAFE, coverage_floor=COVERAGE_FLOOR,
                         probes=list(PROBES), cut_sig=CUT_SIG,
                         prereg="results/Stage2/prereg_updater3.md",
                         note="frozen/updater_v2 逐字复用 Updater2/ckpt（同流同账本）；"
                              "族标签只用于评估分组；签名=纯观测史函数（G-B）")
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    table = render(res)
    (OUT / "table.md").write_text(table, "utf-8")
    print("\n" + table, flush=True)
    print(f"产物：{OUT}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
