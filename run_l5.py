"""run_l5.py — L5 六臂：Action×Model 联合策略学习（prereg_l5_updater2.md §1，第三张设计决策表）。

数据 = 张量 240 槽（tensor_cache，8 族 × core10 × 3 模型 per-series nRMSE）× P0 特征
（records_s2，P1a 未转正）。评估 = leave-one-family-out（策略只见 7 族标签）。
六臂/同预算/分支规则见预注册——本文件不引入任何预注册之外的自由度。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_l5
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .e32_policy import P_FEATS, paired_bootstrap_ci
from .s2_corpus import S2_FAMILIES

STAGE2 = Path(__file__).resolve().parent / "results" / "Stage2"
CACHE = STAGE2 / "tensor_cache"
REC_PATH = STAGE2 / "S2_replication" / "records_s2.jsonl"
OUT = STAGE2 / "L5"
SEED = 20260705
GBDT = dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.7)   # 预注册，不调参
MODELS_MAIN = ("seasonal_naive", "dlinear_pooled", "chronos_bolt_small")
MODELS_SUB = ("dlinear_pooled", "chronos_bolt_small")
ARMS = ("global_pair", "model_only", "action_only", "sequential", "joint")
BOOT_B = 2000


def _gbr(bump: int = 0) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(**GBDT, random_state=(300000 + SEED * 7 + bump) % 4294967295)


def load_tensor(models: Tuple[str, ...]) -> Tuple[Dict[Tuple[str, str, str], float], List[str]]:
    slot: Dict[Tuple[str, str, str], float] = {}
    actions_set = set()
    for f in sorted(CACHE.glob("*__*.json")):
        doc = json.loads(f.read_text("utf-8"))
        if doc["model"] not in models:
            continue
        actions_set.add(doc["action"])
        for u, v in doc["per_series"].items():
            slot[(u, doc["action"], doc["model"])] = float(v)
    return slot, sorted(actions_set)


def run_menu(models: Tuple[str, ...], recs: List[dict]) -> dict:
    slot, actions = load_tensor(models)
    uids = [r["uid"] for r in recs]
    fam_of = {r["uid"]: r["origin"] for r in recs}
    X = {r["uid"]: np.array([r["snr"], r["miss_rate"], *r["X_p"]], float) for r in recs}
    full = [u for u in uids if all((u, a, m) in slot for a in actions for m in models)]
    Y = {u: np.array([[slot[(u, a, m)] for m in models] for a in actions]) for u in full}   # (A,M)

    picks: Dict[str, Dict[str, Tuple[int, int]]] = {n: {} for n in ARMS}
    for held in S2_FAMILIES:                                    # LODO：held-out 一族
        tr = [u for u in full if fam_of[u] != held]
        te = [u for u in full if fam_of[u] == held]
        if not te:
            continue
        Ytr = np.stack([Y[u] for u in tr])                      # (n,A,M)
        Xtr = np.stack([X[u] for u in tr])
        Xte = np.stack([X[u] for u in te])
        mean_am = Ytr.mean(axis=0)                              # (A,M) train 均值
        gi = np.unravel_index(np.argmin(mean_am), mean_am.shape)
        a_star = int(np.argmin(mean_am.mean(axis=1)))           # 边际均值最优（预注册定义）
        m_star = int(np.argmin(mean_am.mean(axis=0)))
        # 1 global_pair
        for u in te:
            picks["global_pair"][u] = (int(gi[0]), int(gi[1]))
        # 2 model_only：动作固定 a*，per-model 头
        pm = [ _gbr(100 + mi).fit(Xtr, Ytr[:, a_star, mi]) for mi in range(len(models)) ]
        mo = np.argmin(np.stack([m.predict(Xte) for m in pm]), axis=0)
        for i, u in enumerate(te):
            picks["model_only"][u] = (a_star, int(mo[i]))
        # 3 action_only：模型固定 m*，per-action 头
        pa = [ _gbr(200 + ai).fit(Xtr, Ytr[:, ai, m_star]) for ai in range(len(actions)) ]
        ao = np.argmin(np.stack([m.predict(Xte) for m in pa]), axis=0)
        for i, u in enumerate(te):
            picks["action_only"][u] = (int(ao[i]), m_star)
        # 5 joint：单 Q(x, onehot(a), onehot(m))
        rows, ys = [], []
        for j, u in enumerate(tr):
            for ai in range(len(actions)):
                for mi in range(len(models)):
                    oa = np.zeros(len(actions)); oa[ai] = 1.0
                    om = np.zeros(len(models)); om[mi] = 1.0
                    rows.append(np.concatenate([Xtr[j], oa, om]))
                    ys.append(Ytr[j, ai, mi])
        q = _gbr(300).fit(np.array(rows), np.array(ys))
        qte = np.empty((len(te), len(actions), len(models)))
        for ai in range(len(actions)):
            for mi in range(len(models)):
                oa = np.zeros(len(actions)); oa[ai] = 1.0
                om = np.zeros(len(models)); om[mi] = 1.0
                qte[:, ai, mi] = q.predict(np.hstack([Xte, np.tile(np.concatenate([oa, om]), (len(te), 1))]))
        for i, u in enumerate(te):
            ai, mi = np.unravel_index(int(np.argmin(qte[i])), qte[i].shape)
            picks["joint"][u] = (int(ai), int(mi))
            # 4 sequential：stage1=action_only 的 â；stage2=同一 Q 限制在 (â,·)
            a_hat = int(ao[i])
            picks["sequential"][u] = (a_hat, int(np.argmin(qte[i, a_hat])))

    # —— 汇总 ——
    from collections import Counter
    oracle = {u: float(Y[u].min()) for u in full}
    res: dict = {"n": len(full), "actions": actions, "models": list(models),
                 "arms": {}, "comparisons": {}, "selection": {}}
    reg: Dict[str, np.ndarray] = {}
    fams = sorted({fam_of[u] for u in full})
    for n in ARMS:
        r = np.array([Y[u][picks[n][u]] - oracle[u] for u in full])
        reg[n] = r
        by_fam = {f: float(r[[fam_of[u] == f for u in full]].mean()) for f in fams}
        res["arms"][n] = dict(mean_regret=float(r.mean()), regret_by_family=by_fam,
                              worst_family=max(by_fam, key=by_fam.get),
                              worst_family_regret=float(max(by_fam.values())))
        res["selection"][n] = dict(
            model=dict(Counter(models[picks[n][u][1]] for u in full)),
            action_top=dict(Counter(actions[picks[n][u][0]] for u in full).most_common(4)))
    res["arms"]["oracle_pair"] = dict(mean_regret=0.0, note="按定义 0（诊断锚）",
                                      mean_loss=float(np.mean([oracle[u] for u in full])))
    for pair in (("joint", "action_only"), ("sequential", "action_only"), ("joint", "sequential"),
                 ("model_only", "action_only"), ("joint", "global_pair")):
        res["comparisons"][f"{pair[0]}_vs_{pair[1]}"] = paired_bootstrap_ci(
            reg[pair[0]], reg[pair[1]], n_boot=BOOT_B, seed=SEED)
    return res


def verdict(main: dict) -> dict:
    j_vs_a = main["comparisons"]["joint_vs_action_only"]
    j_vs_s = main["comparisons"]["joint_vs_sequential"]
    s_vs_a = main["comparisons"]["sequential_vs_action_only"]
    joint_wins = (j_vs_a["ci_hi"] < 0
                  and main["arms"]["joint"]["worst_family_regret"]
                  <= main["arms"]["action_only"]["worst_family_regret"] + 1e-9)
    seq_close = j_vs_s["ci_lo"] <= 0 <= j_vs_s["ci_hi"]
    seq_wins = s_vs_a["ci_hi"] < 0
    if joint_wins and seq_close and seq_wins:
        decision = "SEQUENTIAL：与 joint 无显著差且胜 action_only → 取更简单形态接入 overlay"
    elif joint_wins:
        decision = "JOINT：均值+安全双赢 → model_id 接入现有 overlay（L5 面）"
    elif seq_wins:
        decision = "SEQUENTIAL：joint 未过但 sequential 胜 → sequential 接入"
    else:
        decision = "KEEP-ACTION-ONLY：张量交互存在但 P0 特征下不可实现——保留 action-only Harness"
    return dict(joint_wins_mean_and_safety=bool(joint_wins), sequential_close_to_joint=bool(seq_close),
                sequential_beats_action_only=bool(seq_wins), decision=decision)


def render(main: dict, sub: dict, vd: dict) -> str:
    lines = ["# L5 六臂（张量效用 × P0 特征，leave-one-family-out；prereg_l5_updater2.md §1）", "",
             "> 主结果=三模型菜单；副报 DLinear+Chronos 子菜单（敏感性：双模型 share=0.143——"
             "L5 价值可能部分由弱 seasonal_naive 基线驱动）。", ""]
    for tag, res in (("三模型（主）", main), ("DLinear+Chronos（副）", sub)):
        lines += [f"## {tag}  n={res['n']}",
                  "| arm | mean regret | worst family (regret) | 模型选择分布 |", "|---|---|---|---|"]
        for n in ARMS:
            a = res["arms"][n]
            sel = res["selection"][n]["model"]
            lines.append(f"| {n} | {a['mean_regret']:.4f} | {a['worst_family']} "
                         f"({a['worst_family_regret']:.3f}) | {sel} |")
        lines.append(f"| oracle_pair | 0（锚；mean loss={res['arms']['oracle_pair']['mean_loss']:.3f}） | — | — |")
        lines += ["", "关键比较（paired ΔRegret，负=前者更好）："]
        for k, v in res["comparisons"].items():
            lines.append(f"- {k}: {v['mean']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")
        lines.append("")
    lines += [f"**分支判决（预注册规则）**：{vd['decision']}",
              f"（joint 均值+安全={vd['joint_wins_mean_and_safety']}，seq≈joint={vd['sequential_close_to_joint']}，"
              f"seq>action_only={vd['sequential_beats_action_only']}）"]
    return "\n".join(lines) + "\n"


def main():
    t0 = time.time()
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    res_main = run_menu(MODELS_MAIN, recs)
    res_sub = run_menu(MODELS_SUB, recs)
    vd = verdict(res_main)
    out = dict(main=res_main, submenu=res_sub, verdict=vd,
               prereg="results/Stage2/prereg_l5_updater2.md §1", seed=SEED, gbdt=GBDT)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    table = render(res_main, res_sub, vd)
    (OUT / "table.md").write_text(table, "utf-8")
    print(table, flush=True)
    print(f"产物：{OUT}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
