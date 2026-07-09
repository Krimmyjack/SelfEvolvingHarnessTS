"""run_router1.py — Router 第一轮：策略形态比较（Stage 2.2，第二张设计决策表）。

预注册：results/Stage2/prereg_s2_replication.md §3（臂/超参/触发器/判据全部先落字，不调参）。
特征集 = fixpc（P0-D 2 + P1a-P 9 + C 3，P1a 第一张表定案）；折/L_train/L_test/seed 冻结同 P1a。

八臂：pa_gbdt（对照，须复现 P1a fixpc 数字=守卫③）/ pa_abstain_std / pa_abstain_cgate /
      sq（shared-Q(P,D,C,a)+φ 现算）/ sq_rank（uid 内秩标签，尺度免疫）/ sq_abstain_std /
      sq_abstain_kcv（κ 由 outer-train 内 4 折 CV 选）/ sq_abstain_cgate。
unseen dosage：训练剔除 f0_median_w15 全部标签 → sq 经 φ 元数据仍可预测 w15 列；
      pa_low15（9 动作菜单）= 结构性不可能的对照。
φ(P,D,a) 现算不入 spec（v1.1d）：family one-hot / w/25 / w÷period / smoothable_energy(w)。
C-gated 触发：legacy(κ=1) ∧ (c_peak_sig<q25 ∨ c_acf_confirm<q25)，分位数逐折取自 outer-train；
      c_obs_coverage 不进触发器（dev 上方向反转=cell 代理）。

⚠ 双 caveat 同 P1a：发现集乐观性（设计级判定，引用级等 S2 复制）；8 臂多重比较——
"胜者"只获得进入 S2 复制集的资格。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_router1
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingRegressor

from .conditioning.p1a import P1A_C_FEATS, _interp_perception, p1a_vectors
from .e32_policy import FALLBACK_ACTION, GBDTArm, PolicyData, _subgroup_stats, paired_bootstrap_ci
from .nested_supply import make_folds
from .run_p1a_replay import CAVEAT, guard1_p0_bit_identical, load_frozen, rebuild_histories

OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "Router1"
P1A_REPORT = Path(__file__).resolve().parent / "results" / "Stage2" / "P1a" / "report.json"
HELD_OUT_ACTION = "f0_median_w15"                 # 留一剂量（预注册）
KAPPA_GRID = (0.5, 1.0, 1.5, 2.0)
SQ_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.7)  # 预注册，不调参
E_SQ = 5
BOOT_B = 2000
ARM_ORDER = ("pa_gbdt", "pa_abstain_std", "pa_abstain_cgate",
             "sq", "sq_rank", "sq_abstain_std", "sq_abstain_kcv", "sq_abstain_cgate")


# ════════════════════════════ 动作元数据 + φ ════════════════════════════
def action_meta(actions: List[str]) -> Tuple[Dict[str, Tuple[str, int]], List[str]]:
    """{aid: (family, window)}（单一真源=action_menu_v1 resolved params）+ family 全集。"""
    from .policy import action_menu_v1
    menu = action_menu_v1()
    metas: Dict[str, Tuple[str, int]] = {}
    for aid in actions:
        spec = menu.actions[aid]
        ops = [s.op for s in spec.steps if s.op != "impute_linear"]
        fam = "+".join(ops) if ops else "none"
        w = 0
        for s in spec.steps:
            if "window" in dict(s.params):
                w = int(dict(s.params)["window"])
        metas[aid] = (fam, w)
    return metas, sorted({m[0] for m in metas.values()})


def uid_phi_basis(hist: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """φ 的 uid 侧原料：去趋势感知插值序列的 (freqs, power, total_power)。"""
    raw = np.asarray(hist, float).ravel()
    mask = ~np.isnan(raw)
    xi = _interp_perception(raw, mask)
    t = np.arange(raw.size, dtype=float)
    obs = raw[mask]
    if float(np.var(obs)) <= 1e-12:
        detr = xi - float(obs.mean())
    else:
        coef = np.polyfit(t[mask], obs, 1)
        detr = xi - np.polyval(coef, t)
    p = np.abs(np.fft.rfft(detr - detr.mean())) ** 2
    return np.fft.rfftfreq(raw.size), p, float(p[1:].sum())


def phi_vector(fam: str, w: int, period: float, spec_basis, families: List[str]) -> np.ndarray:
    freqs, power, tot = spec_basis
    onehot = np.zeros(len(families))
    onehot[families.index(fam)] = 1.0
    w_over_p = (w / period) if (w > 0 and period > 0) else 0.0
    sme = float(power[freqs > 1.0 / w].sum() / tot) if (w >= 2 and tot > 0) else 0.0
    return np.concatenate([onehot, [w / 25.0, w_over_p, sme]])


# ════════════════════════════ shared-Q 臂 ════════════════════════════
class SharedQ:
    """单 GBDT Q(x_uid, meta_a, φ)；E 成员 paired ensemble（同 GBDTArm 语义）。"""

    def __init__(self, actions: List[str], rank_labels: bool = False, seed: int = 0,
                 exclude_action: Optional[str] = None):
        self.actions, self.rank_labels, self.seed = list(actions), rank_labels, seed
        self.exclude = exclude_action

    def _gbr(self, e: int) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            **SQ_PARAMS, random_state=(200000 + self.seed * 1000 + e * 11) % 4294967295)

    def fit(self, X_uid: Dict[str, np.ndarray], phi_of, L_of: Dict[str, Dict[str, float]],
            tr_uids: List[str]) -> "SharedQ":
        rows, ys = [], []
        for u in tr_uids:
            labels = L_of.get(u, {})
            avail = [a for a in self.actions
                     if a in labels and a != self.exclude and np.isfinite(labels[a])]
            if not avail:
                continue
            vals = np.array([labels[a] for a in avail])
            if self.rank_labels:                              # uid 内秩（尺度免疫，D10）
                r = rankdata(vals)
                vals = (r - 1) / max(len(r) - 1, 1)
            for a, y in zip(avail, vals):
                rows.append(np.concatenate([X_uid[u], phi_of(u, a)]))
                ys.append(float(y))
        X, y = np.array(rows), np.array(ys)
        self.models = [self._gbr(e).fit(X, y) for e in range(E_SQ)]
        return self

    def preds(self, X_uid, phi_of, te_uids: List[str]) -> np.ndarray:
        """→ (A, E, n_te) 预测（含被 exclude 的动作列——元数据外推正是留一剂量的测点）。"""
        out = np.empty((len(self.actions), E_SQ, len(te_uids)))
        for j, a in enumerate(self.actions):
            Xte = np.array([np.concatenate([X_uid[u], phi_of(u, a)]) for u in te_uids])
            for e, m in enumerate(self.models):
                out[j, e] = m.predict(Xte)
        return out


def picks_from_preds(preds: np.ndarray, fallback_idx: int, kappa: Optional[float] = None,
                     cgate_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(A,E,n) 预测 → (pick_idx, abstain_mask)。kappa=None → 无 abstain；
    cgate_mask 给出时 abstain 须同时落在低 C 区（预注册 §3）。"""
    mu = preds.mean(axis=1)
    pick = np.argmin(mu, axis=0)
    n = preds.shape[2]
    abst = np.zeros(n, bool)
    if kappa is not None:
        rows = np.arange(n)
        adv_e = preds[fallback_idx] - preds[pick, :, rows].T          # (E, n) paired advantage
        abst = adv_e.mean(axis=0) < kappa * adv_e.std(axis=0)
        if cgate_mask is not None:
            abst = abst & cgate_mask
        pick = np.where(abst, fallback_idx, pick)
    return pick.astype(int), abst


def pa_preds(arm: GBDTArm, data: PolicyData, te_idx: np.ndarray) -> np.ndarray:
    """per-action GBDTArm 的 (A,E,n_te) 预测（复用其 models，不平行实现数学）。"""
    Xte = data.X(arm.feats)[te_idx]
    return np.stack([[m.predict(Xte) for m in ens] for ens in arm.models])


# ════════════════════════════ 主流程 ════════════════════════════
def run(freeze, recs, details, hist) -> dict:
    actions = list(freeze["actions"])
    seed = int(freeze["seed"])
    fb = actions.index(FALLBACK_ACTION)
    metas, families = action_meta(actions)
    meta_r = {r["uid"]: r for r in recs}

    # uid 侧特征（fixpc）+ φ 原料
    X_uid: Dict[str, np.ndarray] = {}
    C_uid: Dict[str, np.ndarray] = {}
    period_uid: Dict[str, float] = {}
    basis: Dict[str, tuple] = {}
    for r in recs:
        u = r["uid"]
        v = p1a_vectors(hist[u])
        X_uid[u] = np.concatenate([[r["snr"], r["miss_rate"]], v["p"], v["c"]])   # fixpc 14 维
        C_uid[u] = v["c"]
        period_uid[u] = float(v["p"][0])
        basis[u] = uid_phi_basis(hist[u])

    def phi_of(u: str, a: str) -> np.ndarray:
        fam, w = metas[a]
        return phi_vector(fam, w, period_uid[u], basis[u], families)

    picks: Dict[str, Dict[str, dict]] = {n: {} for n in ARM_ORDER}
    picks_lo15: Dict[str, Dict[str, dict]] = {"sq_low15": {}, "pa_low15": {}}
    kcv_log: List[dict] = []
    actions_lo = [a for a in actions if a != HELD_OUT_ACTION]

    for det in details:
        t0 = time.time()
        tr, te = sorted(det["train_uids"]), sorted(det["test_uids"])
        order = tr + te
        L_of = det["L_train"]
        # —— per-action 基座（fixpc）——
        L = np.full((len(order), len(actions)), np.nan)
        for i, u in enumerate(order):
            if u in L_of:
                for j, a in enumerate(actions):
                    if a in L_of[u]:
                        L[i, j] = L_of[u][a]
        data = PolicyData(uids=order, actions=actions, L=L,
                          X_d=np.array([X_uid[u][:2] for u in order]),
                          X_p=np.array([X_uid[u][2:] for u in order]),
                          cell=np.array([meta_r[u]["cell"] for u in order]),
                          origin=np.array([meta_r[u]["origin"] for u in order]))
        tr_idx, te_idx = np.arange(len(tr)), np.arange(len(tr), len(order))
        # plain=E1（P1a dp_gbdt 语义，守卫③逐位复现）；abstain 基座=E5（GBDTArm 同款 paired ensemble）
        pa1 = GBDTArm(("d", "p"), seed=seed).fit(data, tr_idx)
        preds_pa1 = pa_preds(pa1, data, te_idx)
        pa5 = GBDTArm(("d", "p"), abstain=True, seed=seed).fit(data, tr_idx)
        preds_pa5 = pa_preds(pa5, data, te_idx)
        # —— C 门限（逐折 outer-train，无泄漏；coverage 不进触发器）——
        c_tr = np.array([C_uid[u] for u in tr])
        q_ps, q_ac = np.quantile(c_tr[:, 0], 0.25), np.quantile(c_tr[:, 1], 0.25)
        c_te = np.array([C_uid[u] for u in te])
        cgate = (c_te[:, 0] < q_ps) | (c_te[:, 1] < q_ac)
        # —— shared-Q 三形态 ——
        sq = SharedQ(actions, seed=seed).fit(X_uid, phi_of, L_of, tr)
        preds_sq = sq.preds(X_uid, phi_of, te)
        sq_rank = SharedQ(actions, rank_labels=True, seed=seed).fit(X_uid, phi_of, L_of, tr)
        preds_sqr = sq_rank.preds(X_uid, phi_of, te)
        sq_lo = SharedQ(actions, seed=seed, exclude_action=HELD_OUT_ACTION).fit(X_uid, phi_of, L_of, tr)
        preds_sqlo = sq_lo.preds(X_uid, phi_of, te)
        # —— κ 内层 4 折 CV（sq 基座；标签=L_train，无 outer-test 泄漏）——
        inner = make_folds(tr, 4, seed + 4242)
        kappa_regret = {k: [] for k in KAPPA_GRID}
        for f in range(4):
            itr = [u for u in tr if inner[u] != f]
            ite = [u for u in tr if inner[u] == f]
            if not itr or not ite:
                continue
            m = SharedQ(actions, seed=seed).fit(X_uid, phi_of, L_of, itr)
            pr = m.preds(X_uid, phi_of, ite)
            Li = np.array([[L_of[u].get(a, np.nan) for a in actions] for u in ite])
            orc = np.nanmin(Li, axis=1)
            for k in KAPPA_GRID:
                p_k, _ = picks_from_preds(pr, fb, kappa=k)
                kappa_regret[k].append(float(np.nanmean(Li[np.arange(len(ite)), p_k] - orc)))
        kappa_star = min(KAPPA_GRID, key=lambda k: float(np.mean(kappa_regret[k])))
        kcv_log.append(dict(fold=det["name"], kappa_star=kappa_star,
                            grid={str(k): float(np.mean(v)) for k, v in kappa_regret.items()}))
        # —— 八臂 picks ——
        arm_preds = dict(
            pa_gbdt=(preds_pa1, None, None), pa_abstain_std=(preds_pa5, 1.0, None),
            pa_abstain_cgate=(preds_pa5, 1.0, cgate),
            sq=(preds_sq, None, None), sq_rank=(preds_sqr, None, None),
            sq_abstain_std=(preds_sq, 1.0, None), sq_abstain_kcv=(preds_sq, kappa_star, None),
            sq_abstain_cgate=(preds_sq, 1.0, cgate))
        for name, (pr, k, cg) in arm_preds.items():
            p, ab = picks_from_preds(pr, fb, kappa=k, cgate_mask=cg)
            for i, u in enumerate(te):
                picks[name][u] = dict(pick=actions[int(p[i])], abstain=bool(ab[i]))
        # —— 留一剂量面板 ——
        p_lo, _ = picks_from_preds(preds_sqlo, fb)
        for i, u in enumerate(te):
            picks_lo15["sq_low15"][u] = dict(pick=actions[int(p_lo[i])], abstain=False)
        L9 = np.delete(L, actions.index(HELD_OUT_ACTION), axis=1)
        data9 = PolicyData(uids=order, actions=actions_lo, L=L9,
                          X_d=data.X_d, X_p=data.X_p, cell=data.cell, origin=data.origin)
        pa9 = GBDTArm(("d", "p"), seed=seed).fit(data9, tr_idx)
        p9, _ = pa9.picks(data9, te_idx)
        for i, u in enumerate(te):
            picks_lo15["pa_low15"][u] = dict(pick=actions_lo[int(p9[i])], abstain=False)
        print(f"  [fold {det['name']}] 8 臂+留一剂量 完成（κ*={kappa_star}）"
              f"[{time.time()-t0:.0f}s]", flush=True)
    return dict(picks=picks, picks_lo15=picks_lo15, kcv_log=kcv_log)


# ════════════════════════════ 评估 ════════════════════════════
def evaluate(recs, run_out, actions: List[str], seed: int) -> dict:
    order = [r["uid"] for r in recs]
    L = np.array([[r["L_test"][a] for a in actions] for r in recs])
    oracle = L.min(axis=1)
    cell = np.array([r["cell"] for r in recs])
    origin = np.array([r["origin"] for r in recs])
    data_eval = PolicyData(uids=order, actions=actions, L=L,
                           X_d=np.zeros((len(order), 2)), X_p=np.zeros((len(order), 1)),
                           cell=cell, origin=origin)
    res: dict = {"arms": {}, "comparisons": {}, "unseen_dosage": {}, "kcv_log": run_out["kcv_log"]}
    regret_of: Dict[str, np.ndarray] = {}
    for name in ARM_ORDER:
        pk = np.array([actions.index(run_out["picks"][name][u]["pick"]) for u in order])
        ab = np.array([run_out["picks"][name][u]["abstain"] for u in order], bool)
        loss = L[np.arange(len(order)), pk]
        regret_of[name] = loss - oracle
        sub = _subgroup_stats(data_eval, loss)
        res["arms"][name] = dict(
            mean_regret=float(regret_of[name].mean()), abstain_rate=float(ab.mean()),
            worst_group_lcb=float(min(v["lcb"] for v in sub.values())),
            worst_group_mean=float(min(v["mean"] for v in sub.values())),
            regret_by_origin={o: float(regret_of[name][origin == o].mean())
                              for o in sorted(set(origin))})
    for name in ARM_ORDER[1:]:
        res["comparisons"][f"{name}_vs_pa_gbdt"] = paired_bootstrap_ci(
            regret_of[name], regret_of["pa_gbdt"], n_boot=BOOT_B, seed=seed)
    # —— 留一剂量面板：w15 为 oracle 的子集 ——
    j15 = actions.index(HELD_OUT_ACTION)
    is15 = L.argmin(axis=1) == j15
    for name in ("sq_low15", "pa_low15"):
        pk = np.array([actions.index(run_out["picks_lo15"][name][u]["pick"]) for u in order])
        loss = L[np.arange(len(order)), pk]
        reg = loss - oracle
        res["unseen_dosage"][name] = dict(
            mean_regret=float(reg.mean()),
            regret_w15_oracle_subset=float(reg[is15].mean()) if is15.any() else None,
            picked_w15_rate_on_subset=float((pk[is15] == j15).mean()) if is15.any() else None,
            n_w15_oracle=int(is15.sum()))
    res["unseen_dosage"]["sq_vs_pa_low15_ci"] = paired_bootstrap_ci(
        L[np.arange(len(order)),
          np.array([actions.index(run_out["picks_lo15"]["sq_low15"][u]["pick"]) for u in order])] - oracle,
        L[np.arange(len(order)),
          np.array([actions.index(run_out["picks_lo15"]["pa_low15"][u]["pick"]) for u in order])] - oracle,
        n_boot=BOOT_B, seed=seed)
    return res


def render(res: dict) -> str:
    lines = ["# Router 第一轮（fixpc 特征，冻结折重放；预注册=prereg_s2_replication.md §3）", "",
             f"> ⚠ {CAVEAT}", "",
             "> 8 臂多重比较：全部设计级；胜者只获得进入 S2 复制集的资格。", "",
             "| arm | mean regret | Δ vs pa_gbdt [95% CI] | worst-group LCB | abstain | S_trend |",
             "|---|---|---|---|---|---|"]
    for name in ARM_ORDER:
        a = res["arms"][name]
        ci = res["comparisons"].get(f"{name}_vs_pa_gbdt")
        ci_s = (f"{ci['mean']:+.4f} [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]" if ci else "—（对照）")
        lines.append(f"| {name} | {a['mean_regret']:.4f} | {ci_s} | {a['worst_group_lcb']:+.4f} | "
                     f"{a['abstain_rate']:.2f} | {a['regret_by_origin'].get('S_trend', float('nan')):.3f} |")
    ud = res["unseen_dosage"]
    lines += ["", "## unseen dosage（训练留出 f0_median_w15）",
              "| arm | mean regret | w15-oracle 子集 regret | w15 选中率(子集) |", "|---|---|---|---|"]
    for name in ("sq_low15", "pa_low15"):
        d = ud[name]
        pr = d["picked_w15_rate_on_subset"]
        sub = d["regret_w15_oracle_subset"]
        lines.append(f"| {name} | {d['mean_regret']:.4f} | "
                     f"{'—' if sub is None else f'{sub:.4f}'} | "
                     f"{'—' if pr is None else f'{pr:.2f}'} |")
    c = ud["sq_vs_pa_low15_ci"]
    lines.append(f"\nsq_low15 − pa_low15 paired ΔRegret：{c['mean']:+.4f} "
                 f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]（n_w15_oracle={ud['sq_low15']['n_w15_oracle']}）")
    lines.append("\nκ* 逐折：" + ", ".join(f"{k['fold']}→{k['kappa_star']}" for k in res["kcv_log"]))
    return "\n".join(lines) + "\n"


def main():
    t0 = time.time()
    freeze, recs, details = load_frozen()
    hist = rebuild_histories({r["uid"] for r in recs})
    guard1_p0_bit_identical(recs, hist)
    print(f"守卫① 过（语料 bit 级复现）[{time.time()-t0:.0f}s]", flush=True)
    run_out = run(freeze, recs, details, hist)
    res = evaluate(recs, run_out, list(freeze["actions"]), int(freeze["seed"]))
    # 守卫③：对照臂逐位复现 P1a fixpc（同特征同标签同 seed → 同数字）
    p1a = json.loads(P1A_REPORT.read_text("utf-8"))
    for arm, ref in (("pa_gbdt", "dp_gbdt"), ("pa_abstain_std", "dp_abstain")):
        got = res["arms"][arm]["mean_regret"]
        want = p1a["specs"]["p1a_fixpc"][ref]["mean_regret"]
        assert abs(got - want) < 1e-9, f"守卫③失败：{arm} {got} ≠ P1a fixpc {want}——管线漂移，禁止出表"
    print("守卫③ 过：对照臂复现 P1a fixpc 数字", flush=True)
    res["caveat"] = CAVEAT
    res["prereg"] = "results/Stage2/prereg_s2_replication.md §3"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    table = render(res)
    (OUT / "table.md").write_text(table, "utf-8")
    print("\n" + table, flush=True)
    print(f"产物：{OUT}\\report.json / table.md  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
