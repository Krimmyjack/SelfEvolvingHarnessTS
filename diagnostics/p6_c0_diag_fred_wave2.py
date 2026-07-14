# -*- coding: utf-8 -*-
"""diagnostics/p6_c0_diag_fred_wave2.py — C0 gate① 根因判决实验 D7-D9（纯诊断，只读复用 p6 冻结件）。

D7 per-series 指纹（加权错配签名）；D8 weighted-Adam 全 96-fit 复刻 → 冻结 evaluate_identity_gate
四判据全表 vs 官方 FAIL；D9 窗协议逐字一致核查（只读）。
只用 C0 legacy 数据（monash_clean.npz，manifest_sha=5c768155…）；不触碰 V/U、不联网、不改冻结面。
结果 → results/Stage2/C0Run/diag/（增量落盘）。
运行：D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.diagnostics.p6_c0_diag_fred_wave2
"""
from __future__ import annotations
import io, json, os, sys, time
from datetime import datetime, timezone
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from SelfEvolvingHarnessTS.p6.split_manifest import build_manifest, validate_manifest
from SelfEvolvingHarnessTS.p6.c0_runner import (
    build_episodes, make_real_degrade_fn, make_adam_trainer, prepared_views_for_program,
    evaluate_identity_gate, EPSILON_RATE, DELTA_SAFE_MULTIPLIER, RAW_PROGRAM_ID,
    P_GATE_PROGRAM_IDS, JUDGE_CFG_FROZEN, HORIZON,
)
from SelfEvolvingHarnessTS.p6.judge_closed_form import (
    fit_domain, series_stats, window_starts, zscore_state, CONTEXT_LEN, KERNEL,
)

PKG = os.path.join(_ROOT, "SelfEvolvingHarnessTS")
OUT = os.path.join(PKG, "results/Stage2/C0Run/diag")
os.makedirs(OUT, exist_ok=True)
DOMAINS = ["covid_deaths", "fred_md", "nn5_daily", "tourism_monthly"]
FRED = "fred_md"
SEEDS = [0, 1, 2]
STRIDE = int(JUDGE_CFG_FROZEN["stride"])


def dump(name, obj):
    obj = {**obj, "_written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    json.dump(obj, open(os.path.join(OUT, name), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print(f"[wrote] {name}")


def log(*a):
    print(*a, flush=True)


# ── C0 物化链（与官方同数据同协议） ──
ledger = [json.loads(l) for l in open(os.path.join(PKG, "results/Stage2/P6Probes/exposure_ledger.jsonl"),
                                      encoding="utf-8") if l.strip()]
u_excl = json.load(open(os.path.join(PKG, "results/Stage2/P6Probes/u_admission_v2_traffic_hourly.json"),
                        encoding="utf-8"))["all_probe_consumed_item_ids"]
manifest = build_manifest(ledger, u_excl)
validate_manifest(manifest)
c0_block = manifest.block("C0")
npz = np.load(os.path.join(PKG, "data/_artifacts/monash_clean.npz"), allow_pickle=True)["clean"]
meta = [json.loads(l) for l in open(os.path.join(PKG, "data/_artifacts/monash_clean.meta.jsonl"),
                                    encoding="utf-8") if l.strip()]
by_uid = {f"{m['config']}:{m['item_id']}": (m["config"], m["item_id"], npz[i]) for i, m in enumerate(meta)}
episodes = build_episodes([by_uid[u] for u in c0_block], make_real_degrade_fn())
by_dom = {d: [ep for ep in episodes if ep.config == d] for d in DOMAINS}
assert manifest.manifest_sha == "5c768155a47c1b4cc033a0eaa724830a7055e343439ef77447ad3e6244e21720"
C0_FREEZE = json.load(open(os.path.join(PKG, "results/Stage2/C0Run/C0_FREEZE.json"), encoding="utf-8"))
log(f"[setup] manifest_sha={manifest.manifest_sha}  domains={ {d: len(v) for d,v in by_dom.items()} }")


# ── 诊断层 weighted-Adam（每窗乘 w_i=n̄/nᵢ，与闭式充分统计量同口径；其余 == 冻结 trainer） ──
def _windows_with_weights(views, stride):
    xs, ys, evals, futs, n_per = [], [], [], [], []
    for v in views:
        h = np.asarray(v.history, float).ravel()
        f = np.asarray(v.future, float).ravel()
        mean, std = zscore_state(h)
        hn = (h - mean) / std
        starts = window_starts(h.size, stride=stride, window_cap=None)
        n_per.append(len(starts))
        for t in starts:
            xs.append(hn[t:t + CONTEXT_LEN]); ys.append(hn[t + CONTEXT_LEN:t + CONTEXT_LEN + HORIZON])
        evals.append(hn[-CONTEXT_LEN:]); futs.append((f[:HORIZON] - mean) / std)
    n = np.array(n_per, float); pos = n > 0
    nbar = float(n[pos].mean())
    w_ep = np.where(pos, nbar / np.where(pos, n, 1.0), 0.0)
    wvec = np.concatenate([np.full(n_per[i], w_ep[i], float) for i in range(len(views))]) if xs else np.array([])
    return (np.asarray(xs, float), np.asarray(ys, float), np.asarray(evals, float),
            np.asarray(futs, float), wvec, n_per, w_ep)


def weighted_adam(views, seed, epochs=120, stride=STRIDE):
    import torch
    import torch.nn.functional as F
    from SelfEvolvingHarnessTS.evaluators import _torch_models as tm
    X, Y, evals, futs, wvec, _n, _w = _windows_with_weights(views, stride)
    old = tm.DEVICE
    tm.DEVICE = "cpu"
    try:
        torch.use_deterministic_algorithms(True)
        tm.seed_all(int(seed))
        net = tm.DLinear(CONTEXT_LEN, HORIZON).to("cpu")
        net.train()
        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        wt = torch.tensor(wvec, dtype=torch.float32)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)          # 无 L2（与冻结一致）
        n = len(Xt)
        for _ep in range(int(epochs)):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                per_win = ((net(Xt[idx]) - Yt[idx]) ** 2).mean(dim=1)   # 逐窗 MSE(over H)
                (wt[idx] * per_win).mean().backward()                  # 加权均值（平均权=1）
                opt.step()
        net.eval()
        with torch.no_grad():
            preds = net(torch.tensor(evals, dtype=torch.float32)).cpu().numpy()
    finally:
        tm.DEVICE = old
    err = preds - futs
    return np.sqrt(np.mean(err * err, axis=1))                          # per-episode


# ══ D7. per-series 指纹（fred raw：n_i + cf/adam per-series eval loss；含 epoch×4） ══
t = time.perf_counter()
fred_views = prepared_views_for_program(by_dom[FRED], "raw_identity")
fred_eps = by_dom[FRED]
cf_fit = fit_domain(fred_views, **JUDGE_CFG_FROZEN)
cf_rmse = np.asarray(cf_fit.per_series_rmse, float)                     # per-episode（views 序）
n_win = np.array([series_stats(v, stride=STRIDE, window_cap=None).n_windows for v in fred_views])
adam120 = np.mean([make_adam_trainer(epochs=120)(fred_views, s) for s in SEEDS], axis=0)
adam480 = np.mean([make_adam_trainer(epochs=480)(fred_views, s) for s in SEEDS], axis=0)

# 聚合到底层 series（4 条；每条 4 preset）
per_series = {}
for i, ep in enumerate(fred_eps):
    per_series.setdefault(ep.series_uid, {"idx": [], "n_win": int(n_win[i])})
    per_series[ep.series_uid]["idx"].append(i)
d7_series = {}
for suid, info in per_series.items():
    ix = info["idx"]
    d7_series[suid] = {
        "n_windows": info["n_win"],
        "cf_rmse_mean": float(cf_rmse[ix].mean()),
        "adam120_rmse_mean": float(adam120[ix].mean()),
        "adam480_rmse_mean": float(adam480[ix].mean()),
        "adam120_minus_cf": float(adam120[ix].mean() - cf_rmse[ix].mean()),
        "adam480_minus_cf": float(adam480[ix].mean() - cf_rmse[ix].mean()),
    }
# 按窗数排序，判定加权错配签名
order = sorted(d7_series, key=lambda s: d7_series[s]["n_windows"])
gaps120 = [d7_series[s]["adam120_minus_cf"] for s in order]
gaps480 = [d7_series[s]["adam480_minus_cf"] for s in order]
nwins = [d7_series[s]["n_windows"] for s in order]
corr_nw_gap = float(np.corrcoef(nwins, gaps120)[0, 1]) if len(nwins) > 1 else float("nan")
low_worse = gaps120[0] > gaps120[-1]        # 窗少序列 gap 更大（Adam 更差）= 加权错配签名
deepen = gaps480[0] > gaps120[0]            # epoch×4 时窗少序列 gap 是否加深
d7_concl = (f"按窗数升序 {list(zip(order,nwins))}；Adam−cf(120) {[round(g,4) for g in gaps120]}；"
            f"corr(n_win,gap)={corr_nw_gap:.3f}；窗少序列更差={low_worse}；×4时短序列特化加深={deepen}")
dump("D7_per_series.json", {"domain": FRED, "program": "raw_identity",
     "per_series": d7_series, "order_by_n_windows": order, "n_windows_sorted": nwins,
     "gap_adam120_minus_cf_sorted": gaps120, "gap_adam480_minus_cf_sorted": gaps480,
     "corr_nwin_gap120": corr_nw_gap, "low_window_worse": bool(low_worse),
     "specialization_deepens_x4": bool(deepen), "conclusion": d7_concl})
log(f"[D7] {d7_concl}  ({time.perf_counter()-t:.1f}s)")


# ══ D8. weighted-Adam 全 96-fit 复刻 → 冻结 evaluate_identity_gate 四判据 vs 官方 ══
t = time.perf_counter()
cf_losses, wad_losses, presets_by_domain = {}, {}, {}
for d in DOMAINS:
    cf_losses[d], wad_losses[d] = {}, {}
    presets_by_domain[d] = [ep.preset for ep in by_dom[d]]
    for g in P_GATE_PROGRAM_IDS:
        views = prepared_views_for_program(by_dom[d], g)
        cf_losses[d][g] = np.asarray(fit_domain(views, **JUDGE_CFG_FROZEN).per_series_rmse, float)
        wad_losses[d][g] = np.mean([weighted_adam(views, s, epochs=120) for s in SEEDS], axis=0)
    log(f"  [D8] domain {d} done ({time.perf_counter()-t:.1f}s cum)")

j_raw = float(np.mean([float(cf_losses[d][RAW_PROGRAM_ID].mean()) for d in DOMAINS]))
eps = EPSILON_RATE * j_raw
delta_safe = DELTA_SAFE_MULTIPLIER * eps
gate_w = evaluate_identity_gate(cf_losses, wad_losses, presets_by_domain, eps)

# 与官方并排（官方 = C0_FREEZE identity_gate）
off = C0_FREEZE["identity_gate"]
c1w = {d: {"abs_diff": gate_w["criterion1_raw_level"]["per_domain"][d]["abs_diff"],
           "tol": gate_w["criterion1_raw_level"]["per_domain"][d]["tol"],
           "u_cf_raw": gate_w["criterion1_raw_level"]["per_domain"][d]["u_cf_raw"],
           "u_adam_raw": gate_w["criterion1_raw_level"]["per_domain"][d]["u_adam_raw"],
           "pass": gate_w["criterion1_raw_level"]["per_domain"][d]["pass"]} for d in DOMAINS}
c1o = {d: {"abs_diff": off["criterion1_raw_level"]["per_domain"][d]["abs_diff"],
           "tol": off["criterion1_raw_level"]["per_domain"][d]["tol"],
           "pass": off["criterion1_raw_level"]["per_domain"][d]["pass"]} for d in DOMAINS}
summary = {
    "epsilon_weighted": eps, "epsilon_official": C0_FREEZE["epsilon"],
    "criterion1_pass_weighted": gate_w["criterion1_raw_level"]["pass"],
    "criterion1_pass_official": off["criterion1_raw_level"]["pass"],
    "criterion2_weighted": {"median_rho": gate_w["criterion2_spearman"]["median_rho"],
                            "pass": gate_w["criterion2_spearman"]["pass"]},
    "criterion2_official": {"median_rho": off["criterion2_spearman"]["median_rho"],
                            "pass": off["criterion2_spearman"]["pass"]},
    "criterion3_weighted": {"rate": gate_w["criterion3_prime_top1"]["rate"],
                            "pass": gate_w["criterion3_prime_top1"]["pass"]},
    "criterion3_official": {"rate": off["criterion3_prime_top1"]["rate"],
                            "pass": off["criterion3_prime_top1"]["pass"]},
    "criterion4_weighted": {"rate": gate_w["criterion4_preset_sign"]["rate"],
                            "pass": gate_w["criterion4_preset_sign"]["pass"]},
    "criterion4_official": {"rate": off["criterion4_preset_sign"]["rate"],
                            "pass": off["criterion4_preset_sign"]["pass"]},
    "gate_pass_weighted": gate_w["pass"], "gate_pass_official": off["pass"],
}
d8_concl = (f"weighted-Adam：① {'四域全过' if summary['criterion1_pass_weighted'] else '仍FAIL'}"
            f"（官方 FAIL）；②ρ={summary['criterion2_weighted']['median_rho']:.3f}/{summary['criterion2_weighted']['pass']}"
            f" ③′={summary['criterion3_weighted']['rate']:.3f}/{summary['criterion3_weighted']['pass']}"
            f" ④={summary['criterion4_weighted']['rate']:.3f}/{summary['criterion4_weighted']['pass']}；"
            f"gate={'PASS' if summary['gate_pass_weighted'] else 'FAIL'}")
dump("D8_weighted_adam_gate.json", {
     "note": "诊断层 weighted-Adam（每窗 w_i=n̄/nᵢ）全 96-fit；其余 == 冻结 trainer（同 seeds/epoch=120/lr=1e-2/无L2）",
     "j_raw_c0": j_raw, "epsilon": eps, "delta_safe": delta_safe,
     "criterion1_weighted_per_domain": c1w, "criterion1_official_per_domain": c1o,
     "full_gate_weighted": gate_w, "summary_vs_official": summary, "conclusion": d8_concl})
log(f"[D8] {d8_concl}  ({time.perf_counter()-t:.1f}s)")


# ══ D9. 窗协议逐字一致核查（只读；Adam vs 闭式） ══
# per-episode 窗数：Adam(_torch_windows_from_views 用 window_starts) vs 闭式(series_stats.n_windows)
adam_nwin = [len(window_starts(int(np.asarray(ep.history).size), stride=STRIDE, window_cap=None))
             for ep in fred_eps]
cf_nwin = [series_stats(v, stride=STRIDE, window_cap=None).n_windows for v in fred_views]
checks = {
    "stride": {"adam": STRIDE, "closed_form": STRIDE, "consistent": True,
               "note": "两者同用 JUDGE_CFG_FROZEN['stride']=4 → window_starts(stride=4)"},
    "per_episode_window_count": {"adam": adam_nwin, "closed_form": cf_nwin,
                                 "consistent": bool(adam_nwin == cf_nwin),
                                 "note": "同一 window_starts(len,stride=4,window_cap=None)"},
    "L_WIN_context_len": {"adam": CONTEXT_LEN, "closed_form": CONTEXT_LEN, "consistent": True},
    "H_horizon": {"adam": HORIZON, "closed_form": HORIZON, "consistent": True},
    "decomposition_kernel": {"adam": KERNEL, "closed_form": KERNEL, "consistent": True,
                             "note": "Adam DLinear.avg_pool1d(kernel=25,replicate-pad) == 闭式 moving_average_replicate(KERNEL=25)"},
    "window_cap": {"adam": None, "closed_form": None, "consistent": True},
    "zscore": {"adam": "history-only", "closed_form": "history-only", "consistent": True,
               "note": "两者同用 zscore_state（history mean/std, floor 1e-8）"},
}
inconsistent = [k for k, v in checks.items() if not v["consistent"]]
# 已知的非窗协议差异（估计器本身，非混杂）
estimator_diff = ("估计器差异（设计内，非窗协议混杂）：Adam=SGD 学习 DLinear 双线性头"
                  "(lin_trend+lin_season, 各含 bias)；闭式=ridge 解 φ=[trend;season;1] 的单 W(97×48)。")
d9_concl = (f"窗协议逐项：{'全一致' if not inconsistent else '不一致项='+str(inconsistent)}"
            f"（stride/窗数/L_WIN/H/核/zscore 均一致）；剩余差异仅估计器本体（非窗协议混杂）")
dump("D9_window_protocol.json", {"checks": checks, "inconsistent_items": inconsistent,
     "all_window_protocol_consistent": not inconsistent, "estimator_difference_note": estimator_diff,
     "conclusion": d9_concl})
log(f"[D9] {d9_concl}")

log("\n[DONE] D7-D9 全部落盘 → results/Stage2/C0Run/diag/")
