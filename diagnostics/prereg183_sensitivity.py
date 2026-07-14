# -*- coding: utf-8 -*-
"""prereg:183 预注册**非裁决** sensitivity（结果后报告；不改 verdict；不写任何决策字段）。

基于两 cycle 已落盘 discovery 产物离线重算（**只复刻 signature 步骤 1-3 的 S1/S2**，无 gate、
无 miner、无 torch、无台账写入；judge=闭式 ridge，唯一可变=cfg.lam）：
  ① ε∈{1%,2%,5%}·J_raw 下重判 S1/S2 点条件（post-hoc counterfactual，如实标注）；
  ② CI95 版 LCB 重算（同 cluster bootstrap 协议，分位 0.05→0.025）；
  ③ λ=1e-2 闭式判官在 D1/D2 上重解 → regret/gap 是否同向。
产物=单表 results/Stage2/prereg183_sensitivity.{json,md}。不写 verdict/terminal/activated 等决策字段。

复刻忠实性由 lam=0.001 臂对拍已落盘 s1/s2（regret_mean/lcb90/ceiling_gap/mean_classes）验证。
"""
from __future__ import annotations
import io, json, os, sys
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
ROOT = "C:/Users/辉/Desktop/Agent"; PKG = os.path.join(ROOT, "SelfEvolvingHarnessTS")
if ROOT not in sys.path: sys.path.insert(0, ROOT)
os.chdir(PKG)

from SelfEvolvingHarnessTS.p6.split_manifest import load_manifest
from SelfEvolvingHarnessTS.p6.c0_runner import build_episodes, make_real_degrade_fn, _resolve_judge_cfg
from SelfEvolvingHarnessTS.p6.harness_state import default_state
from SelfEvolvingHarnessTS.p6.cycle_runner import (
    _run_one, _views_for_arm, _fit_domains, _group_by_domain, bootstrap_seed_for)
from SelfEvolvingHarnessTS.p6.fast_path import det_ladder, prepared_artifact, merge_preset_fingerprints
from SelfEvolvingHarnessTS.p6.c0_runner import judge_ingest
from SelfEvolvingHarnessTS.p6.judge_closed_form import SeriesView, series_stats, series_rmse
from SelfEvolvingHarnessTS.p6.metrics import (
    s1_selector, s2_supply, gain, regret, effect_classes, cluster_bootstrap_means)
from SelfEvolvingHarnessTS.p6.materializer import P6TechnicalAbort

def log(*a): print(*a, flush=True)

manifest = load_manifest("results/Stage2/C0Run_A1/selection_rule_manifest.json")
assert manifest.manifest_sha.startswith("5c768155")
c0 = json.load(open("results/Stage2/C0Run_A1/C0_FREEZE.json", encoding="utf-8"))
EPS_FROZEN = float(c0["epsilon"]); DELTA = float(c0["delta_safe"])
J_RAW = EPS_FROZEN / 0.02          # epsilon_rule = 0.02 * J_raw_C0（反解 J_raw）
B = 2000
STATE = default_state(); K = int(STATE.sampler.expected_total)

# ── D1/D2 episodes（与 cycle harness 同构重建）──────────────────────────────
npz = np.load("data/_artifacts/monash_clean.npz", allow_pickle=True)["clean"]
meta = [json.loads(l) for l in open("data/_artifacts/monash_clean.meta.jsonl", encoding="utf-8") if l.strip()]
by_uid = {f"{x['config']}:{x['item_id']}": (x["config"], x["item_id"], npz[i]) for i, x in enumerate(meta)}
EPISODES = {c: build_episodes([by_uid[u] for u in manifest.block(f"D{c}")], make_real_degrade_fn()) for c in (1, 2)}
for c in (1, 2): assert len(EPISODES[c]) == 128, len(EPISODES[c])

# ── 复刻 signature 步骤 1-3 的 S1/S2（judge cfg 可变；无 gate/miner/torch）────
def compute_s1s2(episodes, cfg, boot_seed, eps):
    n_ep = len(episodes)
    clusters = [ep.series_uid for ep in episodes]
    fps_run = merge_preset_fingerprints(episodes, None)
    runs = {ep.uid: _run_one(ep.uid, ep.history, STATE, K, None, fps_run) for ep in episodes}
    chosen = {uid: r.chosen for uid, r in runs.items()}
    views_h = _views_for_arm(episodes, chosen)
    loss_00, u_00, fits_h, _ = _fit_domains(episodes, views_h, cfg, None, None)
    union = {}
    for ep in episodes:
        for c in runs[ep.uid].kept: union.setdefault(c.sha, c)
    det_progs = det_ladder(); det_shas = {c.sha for c in det_progs}
    all_progs = dict(union)
    for c in det_progs: all_progs.setdefault(c.sha, c)
    ell = {}
    for i, ep in enumerate(episodes):
        ch = chosen[ep.uid]
        for sha in sorted(all_progs):
            if ch is not None and sha == ch.sha:
                ell[(i, sha)] = float(loss_00[i]); continue
            art = prepared_artifact(all_progs[sha], ep.history)
            if art is None:
                if sha in det_shas:
                    raise P6TechnicalAbort(f"det prog {sha} fail on {ep.uid}")
                continue
            st = series_stats(SeriesView(uid=ep.uid, history=judge_ingest(art), future=ep.future),
                              stride=cfg["stride"], window_cap=cfg["window_cap"])
            ell[(i, sha)] = float(series_rmse(fits_h[ep.config].W, st))
    per_ep_s1 = []; class_counts = []; pool_min = np.empty(n_ep); det_min = np.empty(n_ep)
    for i, ep in enumerate(episodes):
        losses_i = {c.sha: ell[(i, c.sha)] for c in runs[ep.uid].kept if (i, c.sha) in ell}
        pmin = min(losses_i.values()) if losses_i else float(loss_00[i])
        pool_min[i] = min(pmin, float(loss_00[i]))
        per_ep_s1.append({"loss_chosen": float(loss_00[i]), "loss_pool_min": float(pool_min[i])})
        class_counts.append(len(effect_classes(losses_i)) if losses_i else 1)
        det_min[i] = min(ell[(i, sha)] for sha in det_shas)
    s1 = dict(s1_selector(per_ep_s1, clusters, eps, B, seed=boot_seed))
    s2 = dict(s2_supply(class_counts, gain(u_00, float(pool_min.mean())),
                        gain(u_00, float(det_min.mean())), eps))
    regrets = [regret(x["loss_chosen"], x["loss_pool_min"]) for x in per_ep_s1]
    return s1, s2, regrets, clusters

PERSISTED = {c: json.load(open(f"results/Stage2/Cycle{c}/cycle{c}_deliverable.json", encoding="utf-8"))["signature"]
             for c in (1, 2)}

# ── lam=0.001（冻结）臂 + 对拍验证 ──────────────────────────────────────────
cfg001 = _resolve_judge_cfg({}); cfg01 = _resolve_judge_cfg({"lam": 0.01})
assert cfg001["lam"] == 0.001 and cfg01["lam"] == 0.01, (cfg001, cfg01)
base = {}; lam01 = {}; validation = {}
for c in (1, 2):
    bs = bootstrap_seed_for(c)
    log(f"[compute] cycle{c} lam=0.001 boot_seed={bs} ...")
    s1, s2, regrets, clusters = compute_s1s2(EPISODES[c], cfg001, bs, EPS_FROZEN)
    base[c] = {"s1": s1, "s2": s2, "regrets": regrets, "clusters": clusters, "boot_seed": bs}
    p = PERSISTED[c]
    validation[c] = {
        "regret_mean": {"replay": s1["regret_mean"], "persisted": p["s1"]["regret_mean"],
                        "match": bool(abs(s1["regret_mean"] - p["s1"]["regret_mean"]) < 1e-12)},
        "s1_lcb90": {"replay": s1["lcb90"], "persisted": p["s1"]["lcb90"],
                     "match": bool(abs(s1["lcb90"] - p["s1"]["lcb90"]) < 1e-12)},
        "ceiling_gap": {"replay": s2["ceiling_gap"], "persisted": p["s2"]["ceiling_gap"],
                        "match": bool(abs(s2["ceiling_gap"] - p["s2"]["ceiling_gap"]) < 1e-12)},
        "mean_classes": {"replay": s2["mean_classes"], "persisted": p["s2"]["mean_classes"],
                         "match": bool(abs(s2["mean_classes"] - p["s2"]["mean_classes"]) < 1e-12)},
    }
    log(f"  regret_mean replay={s1['regret_mean']:.12f} persisted={p['s1']['regret_mean']:.12f} "
        f"match={validation[c]['regret_mean']['match']}")
    log(f"  ceiling_gap replay={s2['ceiling_gap']:.12f} persisted={p['s2']['ceiling_gap']:.12f} "
        f"match={validation[c]['ceiling_gap']['match']}")
    log(f"[compute] cycle{c} lam=0.01 ...")
    s1b, s2b, _, _ = compute_s1s2(EPISODES[c], cfg01, bs, EPS_FROZEN)
    lam01[c] = {"s1": s1b, "s2": s2b}
    log(f"  lam0.01 regret_mean={s1b['regret_mean']:.12f} lcb90={s1b['lcb90']:.12f} "
        f"ceiling_gap={s2b['ceiling_gap']:.12f} mean_classes={s2b['mean_classes']}")

VALID_ALL = all(v[k]["match"] for c in (1, 2) for v in [validation[c]] for k in v)
log(f"[validation] all lam=0.001 replay==persisted (bit): {VALID_ALL}")

# ── ① ε 灵敏度：重判 S1/S2 点条件（post-hoc counterfactual）──────────────────
EPS_GRID = {"1pct": 0.01 * J_RAW, "2pct_frozen": 0.02 * J_RAW, "5pct": 0.05 * J_RAW}
eps_table = {}
for c in (1, 2):
    s1 = base[c]["s1"]; s2 = base[c]["s2"]
    row = {}
    for name, e in EPS_GRID.items():
        # S1 点火（冻结定义）：regret_mean ≥ ε 且 lcb90 > 0
        s1_fire = bool(s1["regret_mean"] >= e and s1["lcb90"] > 0.0)
        # S2 点火（冻结定义）：mean_classes < 2 或 ceiling_gap < −ε
        s2_fire = bool(s2["mean_classes"] < 2.0 or s2["ceiling_gap"] < -e)
        row[name] = {"eps": e, "S1_regret_mean": s1["regret_mean"], "S1_lcb90": s1["lcb90"],
                     "S1_point_fire": s1_fire, "S2_ceiling_gap": s2["ceiling_gap"],
                     "S2_mean_classes": s2["mean_classes"], "S2_point_fire": s2_fire}
    eps_table[c] = row

# ── ② CI95 LCB（分位 0.05→0.025）──────────────────────────────────────────
ci_table = {}
for c in (1, 2):
    means = cluster_bootstrap_means(base[c]["regrets"], base[c]["clusters"], B, seed=base[c]["boot_seed"])
    lcb90 = float(np.quantile(means, 0.05, method="linear"))    # 冻结 = 双侧90% 下端
    lcb95 = float(np.quantile(means, 0.025, method="linear"))   # CI95 = 双侧95% 下端
    ci_table[c] = {"S1_regret_lcb90_q05": lcb90, "S1_regret_lcb95_q025": lcb95,
                   "matches_persisted_lcb90": bool(abs(lcb90 - PERSISTED[c]["s1"]["lcb90"]) < 1e-12),
                   "S3_note": "S3 harm_lcb90=0（全 cohort chosen-vs-raw harm≡0）→ 任意分位退化为 0，CI95 亦 0"}

# ── ③ λ=1e-2 同向性（regret/gap 方向）───────────────────────────────────────
lam_table = {}
for c in (1, 2):
    r001 = base[c]["s1"]["regret_mean"]; r01 = lam01[c]["s1"]["regret_mean"]
    g001 = base[c]["s2"]["ceiling_gap"]; g01 = lam01[c]["s2"]["ceiling_gap"]
    lam_table[c] = {
        "regret_mean_lam001": r001, "regret_mean_lam01": r01,
        "regret_both_subeps_frozen": bool(r001 < EPS_FROZEN and r01 < EPS_FROZEN),
        "regret_sign_same": bool(np.sign(r001) == np.sign(r01)),
        "ceiling_gap_lam001": g001, "ceiling_gap_lam01": g01,
        "gap_sign_same": bool(np.sign(g001) == np.sign(g01)),
        "S1_firing_verdict_preserved": bool((r001 >= EPS_FROZEN) == (r01 >= EPS_FROZEN)),
        "S2_firing_verdict_preserved": bool((g001 < -EPS_FROZEN) == (g01 < -EPS_FROZEN)),
    }

# ── 单表落盘（无决策字段）────────────────────────────────────────────────────
report = {
    "schema": "p6-prereg183-sensitivity/1",
    "status": "prereg183_non_adjudicatory_sensitivity_post_hoc",
    "disclaimer": ("prereg:183 预注册非裁决 sensitivity。结果后报告，**不改 verdict、不写任何决策字段**。"
                   "S1/S2 在反事实 ε/λ 下的点火为 post-hoc counterfactual，非任何 promote/activate 决策。"),
    "inputs": {"epsilon_frozen_2pct": EPS_FROZEN, "delta_safe": DELTA, "J_raw_C0": J_RAW,
               "bootstrap_b": B, "boot_seed_cycle1": bootstrap_seed_for(1), "boot_seed_cycle2": bootstrap_seed_for(2),
               "state": "H0(default_state)", "K": K,
               "replay_scope": "signature steps 1-3 (S1/S2 only); no gate/miner/torch/ledger"},
    "replay_faithfulness_vs_persisted": {"all_match_bit_level": VALID_ALL, "per_cycle": validation},
    "analysis_1_eps_sensitivity_point_conditions": eps_table,
    "analysis_2_ci95_lcb_requantile": ci_table,
    "analysis_3_lambda_1e-2_direction": lam_table,
    "notes": {
        "S1_fire_def": "regret_mean ≥ ε 且 cluster-LCB90 > 0（冻结 s1_selector）",
        "S2_fire_def": "mean_classes < 2 或 ceiling_gap < −ε（冻结 s2_supply）",
        "expected_hit": "ε=1% 下 cycle2 S1 点条件成立（regret_mean 0.00798 ≥ ε1% 0.00794）——post-hoc counterfactual",
    },
}
os.makedirs("results/Stage2", exist_ok=True)
json.dump(report, open("results/Stage2/prereg183_sensitivity.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# markdown 单表
def f6(x): return f"{x:.6f}"
lines = ["# prereg:183 非裁决 sensitivity（post-hoc；不改 verdict）", "",
         f"status: `{report['status']}`  | ε_frozen(2%)={f6(EPS_FROZEN)} δ_safe={f6(DELTA)} J_raw={f6(J_RAW)} B={B}",
         f"replay 忠实性（lam=0.001 vs 已落盘, bit 级）: **{VALID_ALL}**", "",
         "## ① ε 灵敏度：S1/S2 点条件（S1 fire = regret≥ε ∧ lcb90>0; S2 fire = classes<2 ∨ gap<−ε）", "",
         "| cycle | ε档 | ε | S1 regret_mean | S1 lcb90 | **S1 fire** | S2 gap | S2 classes | **S2 fire** |",
         "|---|---|---|---|---|---|---|---|---|"]
for c in (1, 2):
    for name, r in eps_table[c].items():
        lines.append(f"| C{c} | {name} | {f6(r['eps'])} | {f6(r['S1_regret_mean'])} | {f6(r['S1_lcb90'])} "
                     f"| **{r['S1_point_fire']}** | {f6(r['S2_ceiling_gap'])} | {r['S2_mean_classes']} "
                     f"| **{r['S2_point_fire']}** |")
lines += ["", "## ② CI95 LCB 重算（cluster bootstrap 分位 0.05→0.025）", "",
          "| cycle | S1 regret LCB90(q05) | S1 regret LCB95(q025) | ==persisted LCB90 |",
          "|---|---|---|---|"]
for c in (1, 2):
    t = ci_table[c]
    lines.append(f"| C{c} | {f6(t['S1_regret_lcb90_q05'])} | {f6(t['S1_regret_lcb95_q025'])} | {t['matches_persisted_lcb90']} |")
lines += ["", "S3: harm_lcb90≡0（全 cohort harm=0）→ 任意分位退化 0，CI95 亦 0。", "",
          "## ③ λ=1e-2 判官重解：regret/gap 同向性", "",
          "| cycle | regret λ.001 | regret λ.01 | gap λ.001 | gap λ.01 | S1 verdict 保持 | S2 verdict 保持 |",
          "|---|---|---|---|---|---|---|"]
for c in (1, 2):
    t = lam_table[c]
    lines.append(f"| C{c} | {f6(t['regret_mean_lam001'])} | {f6(t['regret_mean_lam01'])} "
                 f"| {f6(t['ceiling_gap_lam001'])} | {f6(t['ceiling_gap_lam01'])} "
                 f"| {t['S1_firing_verdict_preserved']} | {t['S2_firing_verdict_preserved']} |")
lines += ["", "> 非裁决：以上反事实点火/方向仅供 robustness 披露，不构成任何 verdict/activate 决策。"]
open("results/Stage2/prereg183_sensitivity.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
log("[done] results/Stage2/prereg183_sensitivity.{json,md}")
log("[DONE-SENSITIVITY]")
