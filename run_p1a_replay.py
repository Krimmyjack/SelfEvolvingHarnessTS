"""run_p1a_replay.py — Stage 2.1 P1a 第一张表：P0 / P1a 特征在冻结折与冻结标签上的离线重放。

回答的问题：**修复 Pattern（D1 period 劫持 + D2 缺失压缩轴）后，Router 到底有没有变好？**

设计（评审第二十五轮定案）：
  冻结不动：outer folds（ckpt fold_armin*.json 的 uid 划分）、router 训练标签 L_train
  （inner-OOF）、评估标签 L_test（outer-train 头）、动作池/GBDT 超参/κ/E/seed——全部取自
  E-3.2 冻结产物；**只换特征矩阵**重拟合 dp_gbdt/dp_abstain。
  臂（specs）：
    p0        对照（X_d=[SNR,missing_rate]，X_p=P0 8 维——须逐 uid 复现冻结 picks，守卫②）
    p1a_fixp  诊断列：P0 的 D + P1a 的 P（隔离"D1+D2 修复"单独值多少）
    p1a_pd    P1a D(4) + P1a P(9)
    p1a_pdc   P1a D(4) + P1a P(9) + C(3)
  有效性守卫（任一失败即中止，不出表）：
    ① 语料重建确定性：重建 dev 语料 → 重算 P0 特征，与 records 的 X_p/snr/miss **bit 级相等**；
    ② 重放管线保真：p0 spec 重放的 dp_gbdt/dp_abstain picks+abstain ≡ 冻结 records。

⚠ 发现集乐观性 caveat（预注册进报告）：本表跑在发现 D1/D3 的同一批 dev records 上——修复
方案是盯着这批数据的失败设计出来的，改进幅度**天然乐观**。本表只做设计级判定（哪组特征
进 Router 轮，错了可逆）；引用级数字须等 S2 dev 语料生成后复制。

不触碰：confirmatory records（只读历史）、Stage-2 holdout（未生成）、seeds 20–39。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_p1a_replay
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .augment_corpus import build_augmented_corpus
from .conditioning.p1a import P1A_C_FEATS, p1a_vectors
from .e32_policy import (D_FEATS, P_FEATS, FALLBACK_ACTION, GBDTArm, PolicyData,
                         _subgroup_stats, paired_bootstrap_ci)
from .fast_path.perceive import perceive
from .harness import HarnessState

E32 = Path(__file__).resolve().parent / "results" / "E3_2"
OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "P1a"
# p1a_fixpc = **follow-up 机制隔离臂（非预注册）**：首轮显示 pd 比 fixp 差的唯一差异是 D 块
# （gap 拓扑在本语料=6% 均匀随机缺失 → 噪声维）、+C 两臂皆改善 → 补"P0-D + P1a-P + C"隔离验证。
SPECS = ("p0", "p1a_fixp", "p1a_pd", "p1a_pdc", "p1a_fixpc")
ARMS = ("dp_gbdt", "dp_abstain")
BOOT_B = 2000

CAVEAT = ("发现集乐观性：本表跑在发现 D1/D3 的同一批 dev records 上（修复方案盯着这批数据的"
          "失败设计）——幅度天然乐观。仅作设计级判定（P1a 特征集是否进 Router 轮）；"
          "引用级数字须在 S2 dev 语料上复制。")


# ════════════════════════════ 冻结资产装载 ════════════════════════════
def load_frozen():
    freeze = json.loads((E32 / "freeze.json").read_text("utf-8"))
    recs = [json.loads(l) for l in
            (E32 / "records_primary_no_Sar.jsonl").read_text("utf-8").splitlines() if l.strip()]
    details = []
    for f in range(freeze["outer_k"]):
        doc = json.loads((E32 / "ckpt_primary_no_Sar" / f"fold_armin{f}.json").read_text("utf-8"))
        details.append(doc["detail"])
    return freeze, recs, details


def rebuild_histories(uids: set) -> Dict[str, np.ndarray]:
    corpus = build_augmented_corpus(20)
    hist = {rs.series_uid: rs.history for rs in corpus if rs.series_uid in uids}
    missing = uids - set(hist)
    assert not missing, f"语料重建缺 {len(missing)} uid（示例 {sorted(missing)[:3]}）——确定性破裂"
    return hist


def guard1_p0_bit_identical(recs: List[dict], hist: Dict[str, np.ndarray]) -> None:
    """守卫①：重算 P0 特征 == records（bit 级）。失败即中止。"""
    h = HarnessState.from_minimal()
    bad = []
    for r in recs:
        f = perceive(hist[r["uid"]], "forecast", h)["pattern"]["struct_feats"]
        ok = ([float(f[k]) for k in P_FEATS] == [float(x) for x in r["X_p"]]
              and float(f["SNR"]) == float(r["snr"])
              and float(f["missing_rate"]) == float(r["miss_rate"]))
        if not ok:
            bad.append(r["uid"])
    assert not bad, f"守卫①失败：{len(bad)}/{len(recs)} uid P0 特征与 records 不一致（示例 {bad[:3]}）"


# ════════════════════════════ 特征矩阵（per spec）════════════════════════════
def build_features(recs: List[dict], hist: Dict[str, np.ndarray]) -> Dict[str, Dict[str, dict]]:
    """uid → spec → {"d": vec, "p": vec}；P1a 向量现算，P0 取 records（守卫①已证等价）。"""
    out: Dict[str, Dict[str, dict]] = {}
    for r in recs:
        u = r["uid"]
        v = p1a_vectors(hist[u])
        d0 = np.array([r["snr"], r["miss_rate"]], float)
        p0 = np.array(r["X_p"], float)
        out[u] = {
            "p0":        {"d": d0, "p": p0},
            "p1a_fixp":  {"d": d0, "p": v["p"]},
            "p1a_pd":    {"d": v["d"], "p": v["p"]},
            "p1a_pdc":   {"d": v["d"], "p": np.concatenate([v["p"], v["c"]])},
            "p1a_fixpc": {"d": d0, "p": np.concatenate([v["p"], v["c"]])},   # follow-up（见 SPECS 注）
        }
    return out


def _policy_data_for(order: List[str], feats, spec: str, meta: Dict[str, dict],
                     actions: List[str], L_of: Dict[str, Dict[str, float]]) -> PolicyData:
    L = np.full((len(order), len(actions)), np.nan)
    for i, u in enumerate(order):
        if u in L_of:
            for j, a in enumerate(actions):
                if a in L_of[u]:
                    L[i, j] = L_of[u][a]
    return PolicyData(
        uids=list(order), actions=list(actions), L=L,
        X_d=np.array([feats[u][spec]["d"] for u in order]),
        X_p=np.array([feats[u][spec]["p"] for u in order]),
        cell=np.array([meta[u]["cell"] for u in order]),
        origin=np.array([meta[u]["origin"] for u in order]))


# ════════════════════════════ 重放主体 ════════════════════════════
def replay(freeze, recs, details, feats) -> dict:
    actions = list(freeze["actions"])
    seed = int(freeze["seed"])
    meta = {r["uid"]: r for r in recs}
    picks: Dict[str, Dict[str, Dict[str, dict]]] = {s: {a: {} for a in ARMS} for s in SPECS}
    mu_of: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in SPECS}   # uid → dp_gbdt 预测行
    for det in details:
        tr, te = det["train_uids"], det["test_uids"]
        order = sorted(tr) + sorted(te)
        te_order = order[len(tr):]
        tr_idx, te_idx = np.arange(len(tr)), np.arange(len(tr), len(order))
        L_of = {u: det["L_train"][u] for u in det["L_train"]}
        for spec in SPECS:
            data = _policy_data_for(order, feats, spec, meta, actions, L_of)
            for arm_name in ARMS:
                arm = GBDTArm(("d", "p"), abstain=(arm_name == "dp_abstain"), seed=seed)
                arm.fit(data, tr_idx)
                p, ab = arm.picks(data, te_idx)
                if arm_name == "dp_gbdt":                     # response 预测质量用平面臂的 mu
                    Xte = data.X(arm.feats)[te_idx]
                    mu = np.stack([arm.models[a][0].predict(Xte) for a in range(len(actions))])
                    for i, u in enumerate(te_order):
                        mu_of[spec][u] = mu[:, i]
                for i, u in enumerate(te_order):
                    picks[spec][arm_name][u] = dict(pick=actions[int(p[i])], abstain=bool(ab[i]))
        print(f"  [fold {det['name']}] 重放完成（4 specs × 2 arms）", flush=True)
    return dict(picks=picks, mu_of=mu_of, actions=actions)


def guard2_p0_replay_matches(recs: List[dict], picks) -> None:
    bad = []
    for r in recs:
        for arm in ARMS:
            got = picks["p0"][arm][r["uid"]]
            want = r["arms"][arm]
            if got["pick"] != want["pick"] or got["abstain"] != bool(want["abstain"]):
                bad.append((r["uid"], arm, got, want))
    assert not bad, (f"守卫②失败：p0 重放与冻结 records 不一致 {len(bad)} 处（示例 {bad[:2]}）"
                     "——重放管线保真破裂，禁止出表")


# ════════════════════════════ 评估与表 ════════════════════════════
def evaluate(recs, picks, mu_of, actions, feats, seed: int) -> dict:
    order = [r["uid"] for r in recs]
    L = np.array([[r["L_test"][a] for a in actions] for r in recs])
    oracle = L.min(axis=1)
    inc = L[:, actions.index(FALLBACK_ACTION)]
    cell = np.array([r["cell"] for r in recs])
    origin = np.array([r["origin"] for r in recs])
    data_eval = PolicyData(uids=order, actions=actions, L=L,
                           X_d=np.zeros((len(order), 2)), X_p=np.zeros((len(order), 1)),
                           cell=cell, origin=origin)
    res: dict = {"specs": {}, "comparisons": {}, "response_quality": {}, "c_diagnostic": {}}
    regret_of: Dict[str, Dict[str, np.ndarray]] = {}
    for spec in SPECS:
        res["specs"][spec] = {}
        regret_of[spec] = {}
        for arm in ARMS:
            pk = np.array([actions.index(picks[spec][arm][u]["pick"]) for u in order])
            ab = np.array([picks[spec][arm][u]["abstain"] for u in order], bool)
            loss = L[np.arange(len(order)), pk]
            regret = loss - oracle
            regret_of[spec][arm] = regret
            sub = _subgroup_stats(data_eval, loss)
            by_origin = {o: float(regret[origin == o].mean()) for o in sorted(set(origin))}
            miss_mask = np.array(["miss" in c for c in cell])
            res["specs"][spec][arm] = dict(
                mean_loss=float(loss.mean()), mean_regret=float(regret.mean()),
                abstain_rate=float(ab.mean()),
                regret_by_origin=by_origin,
                regret_miss_cells=float(regret[miss_mask].mean()),
                regret_full_cells=float(regret[~miss_mask].mean()),
                worst_group_mean=float(min(v["mean"] for v in sub.values())),
                worst_group_lcb=float(min(v["lcb"] for v in sub.values())),
                delta_vs_incumbent=float((inc - loss).mean()))
    # —— paired ΔRegret CI（spec vs p0，同臂）：负 = P1a regret 更低 = 更好 ——
    for arm in ARMS:
        for spec in SPECS[1:]:
            res["comparisons"][f"{spec}_vs_p0[{arm}]"] = paired_bootstrap_ci(
                regret_of[spec][arm], regret_of["p0"][arm], n_boot=BOOT_B, seed=seed)
    # —— response 预测质量（dp_gbdt 的 mu vs L_test）——
    from scipy.stats import spearmanr
    for spec in SPECS:
        rhos, sse, sst = [], 0.0, 0.0
        for i, u in enumerate(order):
            mu = mu_of[spec][u]
            rho = spearmanr(mu, L[i])[0]
            if np.isfinite(rho):
                rhos.append(float(rho))
            sse += float(((mu - L[i]) ** 2).sum())
            sst += float(((L[i] - L.mean()) ** 2).sum())
        res["response_quality"][spec] = dict(mean_per_uid_rank_corr=float(np.mean(rhos)),
                                             pooled_r2=float(1.0 - sse / sst))
    # —— C 诊断：C 特征与 per-uid regret（p1a_pd 臂）的相关性 + 低置信象限 ——
    c_mat = np.array([feats[u]["p1a_pdc"]["p"][-len(P1A_C_FEATS):] for u in order])
    reg = regret_of["p1a_pd"]["dp_gbdt"]
    for j, name in enumerate(P1A_C_FEATS):
        rho = spearmanr(c_mat[:, j], reg)[0]
        lo_q = c_mat[:, j] <= np.quantile(c_mat[:, j], 0.25)
        res["c_diagnostic"][name] = dict(
            spearman_vs_regret=float(rho) if np.isfinite(rho) else None,
            regret_lowest_quartile=float(reg[lo_q].mean()),
            regret_rest=float(reg[~lo_q].mean()))
    return res


def render_table(res: dict) -> str:
    lines = ["# P1a 第一张表（P0 / P1a 特征离线重放，冻结折+冻结标签）", "",
             f"> ⚠ {CAVEAT}", "",
             "> 注：p1a_fixpc（P0-D+P1a-P+C）为首轮判读后的 follow-up 机制隔离臂，非预注册——"
             "只用于定 Router 轮特征集，不作 headline。", "",
             "| spec | arm | mean regret | ΔRegret vs P0 [95% CI] | S_both | miss cells | "
             "worst-group LCB(Δinc) | abstain |",
             "|---|---|---|---|---|---|---|---|"]
    for spec in SPECS:
        for arm in ARMS:
            s = res["specs"][spec][arm]
            key = f"{spec}_vs_p0[{arm}]"
            ci = res["comparisons"].get(key)
            ci_s = (f"{ci['mean']:+.4f} [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]"
                    if ci else "—（对照）")
            lines.append(
                f"| {spec} | {arm} | {s['mean_regret']:.4f} | {ci_s} | "
                f"{s['regret_by_origin'].get('S_both', float('nan')):.4f} | "
                f"{s['regret_miss_cells']:.4f} | {s['worst_group_lcb']:+.4f} | "
                f"{s['abstain_rate']:.2f} |")
    lines += ["", "## response 预测质量（dp_gbdt μ vs L_test）",
              "| spec | per-uid rank ρ | pooled R² |", "|---|---|---|"]
    for spec in SPECS:
        q = res["response_quality"][spec]
        lines.append(f"| {spec} | {q['mean_per_uid_rank_corr']:.3f} | {q['pooled_r2']:.3f} |")
    lines += ["", "## C 通道诊断（p1a_pd 臂 per-uid regret）",
              "| C 特征 | Spearman vs regret | 最低四分位 regret | 其余 regret |", "|---|---|---|---|"]
    for name, d in res["c_diagnostic"].items():
        rho = d["spearman_vs_regret"]
        lines.append(f"| {name} | {rho:+.3f} | {d['regret_lowest_quartile']:.4f} | "
                     f"{d['regret_rest']:.4f} |" if rho is not None else
                     f"| {name} | n/a | {d['regret_lowest_quartile']:.4f} | {d['regret_rest']:.4f} |")
    return "\n".join(lines) + "\n"


def main():
    t0 = time.time()
    freeze, recs, details = load_frozen()
    uids = {r["uid"] for r in recs}
    print(f"冻结资产：{len(recs)} uid × {len(freeze['actions'])} 动作 × {len(details)} 折"
          f"（seed={freeze['seed']}）", flush=True)
    hist = rebuild_histories(uids)
    print(f"语料重建 OK（{len(hist)} uid）[{time.time()-t0:.0f}s]", flush=True)
    guard1_p0_bit_identical(recs, hist)
    print(f"守卫① 过：P0 特征 480 uid bit 级复现 [{time.time()-t0:.0f}s]", flush=True)
    feats = build_features(recs, hist)
    rep = replay(freeze, recs, details, feats)
    guard2_p0_replay_matches(recs, rep["picks"])
    print(f"守卫② 过：p0 重放 picks ≡ 冻结 records [{time.time()-t0:.0f}s]", flush=True)
    res = evaluate(recs, rep["picks"], rep["mu_of"], rep["actions"], feats, int(freeze["seed"]))
    res["caveat"] = CAVEAT
    res["frozen_inputs"] = dict(freeze_config_sha=freeze.get("config_sha"),
                                n_uids=len(recs), folds=[d["name"] for d in details],
                                boot_b=BOOT_B, specs=list(SPECS))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    table = render_table(res)
    (OUT / "table.md").write_text(table, "utf-8")
    print("\n" + table, flush=True)
    print(f"产物：{OUT}\\report.json / table.md  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
