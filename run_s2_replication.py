"""run_s2_replication.py — S2 dev 唯一复制（prereg_s2_replication.md §1 锁定，事后不加组合）。

Phase B（标签）：S2 dev 语料（协议 v3，672 uid）→ collect cells（FrozenProbe+Ridge 头，判官
口径与 E-3.2 一致）→ 分层 5 折 nested 标签（e32_nested.run_policy_folds，含 P0 anchor 臂，
per-fold checkpoint）。seed=20260705（dev 专用；seeds 20–39 永不复用）。
Phase C（复制臂重放，冻结折+冻结标签只换特征——与 P1a/Router-1 同一管线）：
  p0 / p0_abstain      P0 特征 anchor（守卫：重放 picks ≡ Phase B records 内 anchor 臂）
  fixpc / fixpc_abstain 主候选 = P0-D2 + P1a-P9 + C3
  ddec_snr             D 分解：[P1a-SNR, missing_rate] + P1a-P + C（归因：新 SNR 估计单独效应）
  ddec_full            D 分解：P1a-D4(含 gap 拓扑) + P1a-P + C（S2 有 miss-topology 轴 → gap 可评估）
  sq                   Router-1 胜者形态：shared-Q(P,D,C,a)+φ（超参预注册沿用，不调参）

判决（prereg §1，转正判据）：fixpc ΔRegret vs p0 的 paired CI 不跨 0 **且** fixpc worst-group
LCB 不劣于 p0_abstain → 冻结为 P1/Router v1；均值过安全不过 → 转 sq_rank 方向；复制失败 →
回退最简单可复制组合，不做第三轮特征 finetuning。

不触碰：S2 holdout（未物化）、confirmatory、seeds 20–39。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_s2_replication
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .conditioning.p1a import p1a_vectors
from .e32_nested import run_policy_folds, stratified_folds
from .e32_policy import (FALLBACK_ACTION, PRUNED_POOL_CORE, GBDTArm, LookupArm, PolicyData,
                         _subgroup_stats, key_cell, paired_bootstrap_ci)
from .evaluators.frozen_probe import FrozenProbe
from .fast_path.perceive import perceive
from .harness import HarnessState
from .run_e32 import _variant_map
from .run_router1 import SharedQ, action_meta, pa_preds, phi_vector, picks_from_preds, uid_phi_basis
from .run_variance_decomp import assign_cells, build_cell_cache
from .s2_corpus import S2_DEG_GRID, build_s2_dev

OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "S2_replication"
SEED = 20260705                       # dev 专用（≠E-3.2 的 20260704；20–39 永不复用）
OUTER_K, INNER_K = 5, 4
BOOT_B = 2000
SPECS = ("p0", "fixpc", "ddec_snr", "ddec_full")
EVAL_ARMS = ("p0", "p0_abstain", "fixpc", "fixpc_abstain", "ddec_snr", "ddec_full", "sq")


def collect_cells_data_s2(corpus, actions: List[str], verbose: bool = True) -> Dict[str, dict]:
    """S2 版 collect（true_d 直接由 uid 解析 S2_DEG_GRID；其余同 run_e32.collect_cells_data）。"""
    h = HarnessState.from_minimal()
    fp = FrozenProbe()
    variants = _variant_map(actions)
    cells, _ = assign_cells(corpus)
    out: Dict[str, dict] = {}
    for cid in sorted(cells):
        series = cells[cid]
        t0 = time.time()
        action_caches, common_uids, origin_of = build_cell_cache(fp, series, variants)
        if not common_uids:
            continue
        cu = set(common_uids)
        feats_of = {rs.series_uid: perceive(rs.history, "forecast", h)["pattern"]["struct_feats"]
                    for rs in series if rs.series_uid in cu}
        true_d = {}
        for u in common_uids:
            dname = u.split(":")[2]                          # "S2:{family}:{dname}:{j}"
            dp = S2_DEG_GRID[dname]
            true_d[u] = (float(dp["noise"]), float(dp["miss"]))
        out[cid] = dict(action_caches=action_caches, uids=list(common_uids),
                        origin_of={u: origin_of[u] for u in common_uids},
                        feats_of=feats_of, true_d=true_d)
        if verbose:
            print(f"  [cache] {cid:26s} n={len(common_uids):3d} [{time.time()-t0:.0f}s]", flush=True)
    return out


def phase_b(cells_data, actions: List[str]) -> dict:
    """nested 标签 + P0 anchor 臂（per-fold checkpoint → 断点续 bit 级一致）。"""
    all_uids = [u for cd in cells_data.values() for u in cd["uids"]]
    strat_of = {u: f"{cid}|{cd['origin_of'][u]}"
                for cid, cd in cells_data.items() for u in cd["uids"]}
    fold_of = stratified_folds(all_uids, strat_of, OUTER_K, SEED)
    folds = [(f"armin{f}", [u for u in all_uids if fold_of[u] != f],
              [u for u in all_uids if fold_of[u] == f]) for f in range(OUTER_K)]
    arms = {"global": lambda: LookupArm(None), "d_lookup": lambda: LookupArm(key_cell),
            "dp_gbdt": lambda: GBDTArm(("d", "p"), seed=SEED),
            "dp_abstain": lambda: GBDTArm(("d", "p"), abstain=True, seed=SEED)}
    return run_policy_folds(cells_data, actions, arms, folds, inner_k=INNER_K, seed=SEED,
                            ckpt_dir=OUT / "ckpt_armin")


def phase_c(res_b: dict, actions: List[str], hist: Dict[str, np.ndarray],
            meta_r: Dict[str, dict]) -> dict:
    """复制臂重放（冻结折+冻结 L_train 只换特征；p0 重放须 ≡ Phase B anchor = 管线守卫）。"""
    fb = actions.index(FALLBACK_ACTION)
    metas, families = action_meta(actions)
    X: Dict[str, Dict[str, dict]] = {}
    period_uid, basis = {}, {}
    for u, r in meta_r.items():
        v = p1a_vectors(hist[u])
        d0 = np.array([r["snr"], r["miss_rate"]], float)
        d_p1a_snr = np.array([v["d"][0], r["miss_rate"]], float)
        pc = np.concatenate([v["p"], v["c"]])
        X[u] = {"p0": {"d": d0, "p": np.array(r["X_p"], float)},
                "fixpc": {"d": d0, "p": pc},
                "ddec_snr": {"d": d_p1a_snr, "p": pc},
                "ddec_full": {"d": v["d"], "p": pc}}
        period_uid[u] = float(v["p"][0])
        basis[u] = uid_phi_basis(hist[u])

    def phi_of(u: str, a: str) -> np.ndarray:
        fam, w = metas[a]
        return phi_vector(fam, w, period_uid[u], basis[u], families)

    picks: Dict[str, Dict[str, dict]] = {n: {} for n in EVAL_ARMS}
    for det in res_b["fold_details"]:
        t0 = time.time()
        tr, te = sorted(det["train_uids"]), sorted(det["test_uids"])
        order = tr + te
        L_of = det["L_train"]
        L = np.full((len(order), len(actions)), np.nan)
        for i, u in enumerate(order):
            if u in L_of:
                for j, a in enumerate(actions):
                    if a in L_of[u]:
                        L[i, j] = L_of[u][a]
        tr_idx, te_idx = np.arange(len(tr)), np.arange(len(tr), len(order))
        for spec in SPECS:
            data = PolicyData(uids=order, actions=actions, L=L,
                              X_d=np.array([X[u][spec]["d"] for u in order]),
                              X_p=np.array([X[u][spec]["p"] for u in order]),
                              cell=np.array([meta_r[u]["cell"] for u in order]),
                              origin=np.array([meta_r[u]["origin"] for u in order]))
            plain = GBDTArm(("d", "p"), seed=SEED).fit(data, tr_idx)
            p, _ = plain.picks(data, te_idx)
            for i, u in enumerate(te):
                picks[spec][u] = dict(pick=actions[int(p[i])], abstain=False)
            if spec in ("p0", "fixpc"):
                ab_arm = GBDTArm(("d", "p"), abstain=True, seed=SEED).fit(data, tr_idx)
                pr = pa_preds(ab_arm, data, te_idx)
                pk, ab = picks_from_preds(pr, fb, kappa=1.0)
                for i, u in enumerate(te):
                    picks[f"{spec}_abstain"][u] = dict(pick=actions[int(pk[i])], abstain=bool(ab[i]))
        sq = SharedQ(actions, seed=SEED).fit({u: np.concatenate([X[u]["fixpc"]["d"],
                                                                 X[u]["fixpc"]["p"]]) for u in order},
                                             phi_of, L_of, tr)
        pr = sq.preds({u: np.concatenate([X[u]["fixpc"]["d"], X[u]["fixpc"]["p"]]) for u in order},
                      phi_of, te)
        pk, _ = picks_from_preds(pr, fb)
        for i, u in enumerate(te):
            picks["sq"][u] = dict(pick=actions[int(pk[i])], abstain=False)
        print(f"  [C:{det['name']}] 7 臂重放完成 [{time.time()-t0:.0f}s]", flush=True)
    return picks


def guard_p0_matches_anchor(recs: List[dict], picks) -> None:
    bad = 0
    for r in recs:
        if picks["p0"][r["uid"]]["pick"] != r["arms"]["dp_gbdt"]["pick"]:
            bad += 1
        a = picks["p0_abstain"][r["uid"]]
        w = r["arms"]["dp_abstain"]
        if a["pick"] != w["pick"] or a["abstain"] != bool(w["abstain"]):
            bad += 1
    assert bad == 0, f"守卫失败：p0 重放与 Phase B anchor 不一致 {bad} 处——管线漂移，禁止出表"


def evaluate(recs, picks, actions: List[str]) -> dict:
    order = [r["uid"] for r in recs]
    L = np.array([[r["L_test"][a] for a in actions] for r in recs])
    oracle = L.min(axis=1)
    cell = np.array([r["cell"] for r in recs])
    origin = np.array([r["origin"] for r in recs])
    data_eval = PolicyData(uids=order, actions=actions, L=L,
                           X_d=np.zeros((len(order), 2)), X_p=np.zeros((len(order), 1)),
                           cell=cell, origin=origin)
    res: dict = {"arms": {}, "comparisons": {}, "verdict": {}}
    reg: Dict[str, np.ndarray] = {}
    for name in EVAL_ARMS:
        pk = np.array([actions.index(picks[name][u]["pick"]) for u in order])
        ab = np.array([picks[name][u]["abstain"] for u in order], bool)
        loss = L[np.arange(len(order)), pk]
        reg[name] = loss - oracle
        sub = _subgroup_stats(data_eval, loss)
        res["arms"][name] = dict(
            mean_regret=float(reg[name].mean()), abstain_rate=float(ab.mean()),
            worst_group_lcb=float(min(v["lcb"] for v in sub.values())),
            regret_by_origin={o: float(reg[name][origin == o].mean()) for o in sorted(set(origin))},
            regret_by_topo={t: float(reg[name][[t in c for c in
                            np.array([r['uid'].split(':')[2] for r in recs])]].mean())
                            for t in ("rand", "block", "burst", "full")})
    for name in EVAL_ARMS:
        if name != "p0":
            res["comparisons"][f"{name}_vs_p0"] = paired_bootstrap_ci(
                reg[name], reg["p0"], n_boot=BOOT_B, seed=SEED)
    # —— prereg §1 判决 ——
    ci = res["comparisons"]["fixpc_vs_p0"]
    mean_gate = bool(ci["ci_hi"] < 0)
    safety_gate = bool(res["arms"]["fixpc"]["worst_group_lcb"]
                       >= res["arms"]["p0_abstain"]["worst_group_lcb"] - 1e-12)
    res["verdict"] = dict(
        fixpc_mean_ci_excludes_zero=mean_gate,
        fixpc_worst_group_not_inferior_to_p0_abstain=safety_gate,
        decision=("PASS：冻结为 P1/Router v1 候选" if (mean_gate and safety_gate) else
                  "MEAN-ONLY：转 sq_rank 安全方向" if mean_gate else
                  "FAIL：回退最简单可复制组合（不做第三轮特征 finetuning）"))
    return res


def render(res: dict) -> str:
    lines = ["# S2 dev 唯一复制（prereg §1 锁定臂；独立于发现集的第一张表）", "",
             "| arm | mean regret | Δ vs p0 [95% CI] | worst-group LCB | abstain |",
             "|---|---|---|---|---|"]
    for name in EVAL_ARMS:
        a = res["arms"][name]
        ci = res["comparisons"].get(f"{name}_vs_p0")
        ci_s = (f"{ci['mean']:+.4f} [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]" if ci else "—（对照）")
        lines.append(f"| {name} | {a['mean_regret']:.4f} | {ci_s} | "
                     f"{a['worst_group_lcb']:+.4f} | {a['abstain_rate']:.2f} |")
    v = res["verdict"]
    lines += ["", f"**判决（prereg §1）**：均值门={'✅' if v['fixpc_mean_ci_excludes_zero'] else '❌'} "
                  f"安全门={'✅' if v['fixpc_worst_group_not_inferior_to_p0_abstain'] else '❌'} "
                  f"→ {v['decision']}", ""]
    lines.append("| arm | " + " | ".join(sorted(res["arms"]["p0"]["regret_by_origin"])) + " |")
    lines.append("|---|" + "---|" * len(res["arms"]["p0"]["regret_by_origin"]))
    for name in EVAL_ARMS:
        og = res["arms"][name]["regret_by_origin"]
        lines.append(f"| {name} | " + " | ".join(f"{og[o]:.3f}" for o in sorted(og)) + " |")
    return "\n".join(lines) + "\n"


def main():
    t0 = time.time()
    actions = list(PRUNED_POOL_CORE)
    corpus = build_s2_dev()
    hist = {rs.series_uid: rs.history for rs in corpus}
    print(f"S2 dev：{len(corpus)} uid × {len(actions)} 动作（seed={SEED}）", flush=True)
    cells_data = collect_cells_data_s2(corpus, actions)
    res_b = phase_b(cells_data, actions)
    recs = res_b["records"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records_s2.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), "utf-8")
    print(f"Phase B 完成：{len(recs)} uid 标签 [{time.time()-t0:.0f}s]", flush=True)
    meta_r = {r["uid"]: r for r in recs}
    picks = phase_c(res_b, actions, hist, meta_r)
    guard_p0_matches_anchor(recs, picks)
    print("守卫过：p0 重放 ≡ Phase B anchor", flush=True)
    res = evaluate(recs, picks, actions)
    res["config"] = dict(seed=SEED, outer_k=OUTER_K, inner_k=INNER_K, boot_b=BOOT_B,
                         n_uids=len(recs), actions=actions,
                         prereg="results/Stage2/prereg_s2_replication.md §1",
                         protocol="tensor_protocol_v3.json（frozen_full）")
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    table = render(res)
    (OUT / "table.md").write_text(table, "utf-8")
    print("\n" + table, flush=True)
    print(f"产物：{OUT}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
