# -*- coding: utf-8 -*-
"""diagnostics/p6_c0_diag_fred.py — C0 identity gate ① fred_md 超差根因诊断（纯诊断，只读复用 p6 冻结件）。

复用 p6 冻结组件（import 只读，不修改）：C0 物化链（build_manifest→block C0→monash_clean→
build_episodes）、闭式判官 fit（fit_domain/series_stats）、Adam-DLinear（make_adam_trainer /
frozen DLinear + _torch_windows_from_views）。只在 Adam 训练配置上做诊断变体（L2 / epoch）。

只用 C0 legacy 数据（monash_clean.npz 的 C0 block 16 序列）；不触碰 V/U、不联网。
六项产出 → results/Stage2/C0Run/diag/（每项 JSON + 一行结论，增量落盘）。
运行：D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.diagnostics.p6_c0_diag_fred
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
    build_episodes, make_real_degrade_fn, make_adam_trainer,
    prepared_views_for_program, _torch_windows_from_views,
    C0_SEEDS, P_GATE_PROGRAM_IDS, JUDGE_CFG_FROZEN, HORIZON,
)
from SelfEvolvingHarnessTS.p6.judge_closed_form import (
    fit_domain, series_stats, ridge_matrix, CONTEXT_LEN, DEFAULT_LAM,
)

PKG = os.path.join(_ROOT, "SelfEvolvingHarnessTS")
OUT = os.path.join(PKG, "results/Stage2/C0Run/diag")
os.makedirs(OUT, exist_ok=True)
DOMAINS = ["covid_deaths", "fred_md", "nn5_daily", "tourism_monthly"]
FRED = "fred_md"
SEEDS = list(C0_SEEDS)


def dump(name, obj):
    p = os.path.join(OUT, name)
    obj = {**obj, "_written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"[wrote] {name}")


def log(*a):
    print(*a, flush=True)


# ── C0 物化链（与正式运行同一数据同一协议） ──
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
log(f"[setup] manifest_sha={manifest.manifest_sha}  C0 domains={ {d: len(v) for d,v in by_dom.items()} }")

C0_FREEZE = json.load(open(os.path.join(PKG, "results/Stage2/C0Run/C0_FREEZE.json"), encoding="utf-8"))


# ── faithful Adam replica（复用冻结 DLinear + 窗协议 + seed；只加 per-epoch 捕获 + 可选 L2） ──
def adam_replica(views, seed, epochs, weight_decay=0.0, stride=None, capture=False):
    import torch
    import torch.nn.functional as F
    from SelfEvolvingHarnessTS.evaluators import _torch_models as tm
    eff_stride = int(JUDGE_CFG_FROZEN["stride"] if stride is None else stride)
    X, Y, evals, futs = _torch_windows_from_views(views, eff_stride, None)
    old = tm.DEVICE
    tm.DEVICE = "cpu"
    try:
        torch.use_deterministic_algorithms(True)
        tm.seed_all(int(seed))
        net = tm.DLinear(CONTEXT_LEN, HORIZON).to("cpu")
        net.train()
        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=float(weight_decay))
        n = len(Xt)
        curve = []
        for _ep in range(int(epochs)):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                F.mse_loss(net(Xt[idx]), Yt[idx]).backward()
                opt.step()
            if capture:
                with torch.no_grad():                       # 前向不动 RNG → 训练轨迹与冻结一致
                    curve.append(float(F.mse_loss(net(Xt), Yt).item()))
        net.eval()
        with torch.no_grad():
            preds = net(torch.tensor(evals, dtype=torch.float32)).cpu().numpy()
    finally:
        tm.DEVICE = old
    err = preds - futs
    losses = np.sqrt(np.mean(err * err, axis=1))
    return float(losses.mean()), curve


def u_adam_frozen(views, seed, epochs):
    """冻结 make_adam_trainer 的域均 utility（等权 per-episode RMSE 均值）。"""
    tr = make_adam_trainer(epochs=int(epochs))
    return float(np.mean(tr(views, int(seed))))


fred_raw = prepared_views_for_program(by_dom[FRED], "raw_identity")
U_CF_FRED = float(C0_FREEZE["identity_gate"]["criterion1_raw_level"]["per_domain"][FRED]["u_cf_raw"])
U_ADAM_FRED = float(C0_FREEZE["identity_gate"]["criterion1_raw_level"]["per_domain"][FRED]["u_adam_raw"])
log(f"[ref] fred_md raw: U_cf={U_CF_FRED:.6f}  U_adam(frozen)={U_ADAM_FRED:.6f}")


# ══ D1. 训练曲线（fred raw 3 seeds，120 epoch，末 10% 斜率） ══
t = time.perf_counter()
d1_curves, d1_final = {}, {}
for s in SEEDS:
    u_mean, curve = adam_replica(fred_raw, s, 120, capture=True)
    d1_curves[str(s)] = curve
    d1_final[str(s)] = u_mean
tail_slopes = {}
for s, curve in d1_curves.items():
    c = np.array(curve)
    k = max(2, int(round(0.1 * len(c))))
    tail = c[-k:]
    xs = np.arange(k)
    slope = float(np.polyfit(xs, tail, 1)[0])          # 末 k epoch 训练 MSE 线性斜率（/epoch）
    tail_slopes[s] = {"tail_epochs": k, "slope_per_epoch": slope,
                      "tail_start_mse": float(tail[0]), "tail_end_mse": float(tail[-1]),
                      "rel_drop_over_tail": float((tail[0] - tail[-1]) / max(tail[0], 1e-12))}
repl_ok = abs(np.mean(list(d1_final.values())) - U_ADAM_FRED) < 0.02
median_tail_slope = float(np.median([v["slope_per_epoch"] for v in tail_slopes.values()]))
d1_concl = (f"末10%epoch训练MSE中位斜率={median_tail_slope:.3e}/epoch（{'仍明显下降→欠收敛' if median_tail_slope < -1e-4 else '已近平台'}）；"
            f"replica 复现冻结 U_adam={'是' if repl_ok else '否'}（replica均值{np.mean(list(d1_final.values())):.4f} vs 冻结{U_ADAM_FRED:.4f}）")
dump("D1_training_curves.json", {"program": "raw_identity", "domain": FRED, "epochs": 120, "seeds": SEEDS,
     "per_epoch_train_mse": d1_curves, "replica_final_u_adam": d1_final, "tail_slope": tail_slopes,
     "median_tail_slope_per_epoch": median_tail_slope, "replica_reproduces_frozen": repl_ok,
     "conclusion": d1_concl})
log(f"[D1] {d1_concl}  ({time.perf_counter()-t:.1f}s)")


# ══ D2. 匹配正则（Adam + L2 weight_decay=1e-3，fred raw 3 seeds）；是否落 U_cf±10% ══
t = time.perf_counter()
d2 = {}
for s in SEEDS:
    u_l2, _ = adam_replica(fred_raw, s, 120, weight_decay=DEFAULT_LAM)
    d2[str(s)] = u_l2
u_l2_mean = float(np.mean(list(d2.values())))
band_lo, band_hi = U_CF_FRED * 0.9, U_CF_FRED * 1.1
in_band = bool(band_lo <= u_l2_mean <= band_hi)
d2_concl = (f"Adam+L2(λ=1e-3) U_adam_L2={u_l2_mean:.6f}；U_cf±10%带=[{band_lo:.4f},{band_hi:.4f}]；"
            f"{'落入带内→正则可解释' if in_band else '仍不在带内→正则不足以完全解释'}（vs 无L2冻结{U_ADAM_FRED:.4f}）")
dump("D2_matching_reg.json", {"program": "raw_identity", "domain": FRED, "weight_decay": DEFAULT_LAM,
     "u_adam_L2_per_seed": d2, "u_adam_L2_mean": u_l2_mean, "u_cf": U_CF_FRED,
     "u_cf_band_10pct": [band_lo, band_hi], "in_band": in_band, "u_adam_noL2_frozen": U_ADAM_FRED,
     "conclusion": d2_concl})
log(f"[D2] {d2_concl}  ({time.perf_counter()-t:.1f}s)")


# ══ D3. 延长预算（epoch ×2、×4，无 L2，fred raw 3 seeds）；gap 收窄 or 变宽 ══
t = time.perf_counter()
d3 = {}
for mult in (2, 4):
    ep = 120 * mult
    vals = [u_adam_frozen(fred_raw, s, ep) for s in SEEDS]
    d3[f"x{mult}_epochs{ep}"] = {"per_seed": {str(s): v for s, v in zip(SEEDS, vals)},
                                 "mean": float(np.mean(vals)),
                                 "gap_to_cf": float(np.mean(vals) - U_CF_FRED)}
gap0 = U_ADAM_FRED - U_CF_FRED
narrows = d3["x4_epochs480"]["gap_to_cf"] < d3["x2_epochs240"]["gap_to_cf"] < gap0
d3_concl = (f"gap(U_adam−U_cf): x1={gap0:.4f} → x2={d3['x2_epochs240']['gap_to_cf']:.4f} → "
            f"x4={d3['x4_epochs480']['gap_to_cf']:.4f}；{'单调收窄→欠收敛佐证' if narrows else '未单调收窄→非纯欠收敛'}")
dump("D3_extended_budget.json", {"program": "raw_identity", "domain": FRED, "baseline_gap_x1": gap0,
     "u_cf": U_CF_FRED, "u_adam_x1_frozen": U_ADAM_FRED, "extended": d3,
     "monotone_narrowing": bool(narrows), "conclusion": d3_concl})
log(f"[D3] {d3_concl}  ({time.perf_counter()-t:.1f}s)")


# ══ D4. 逐程序分解（fred_md 8 程序 U_cf vs U_adam 带符号 Δ；读 C0_FREEZE，官方数据） ══
cf_f = C0_FREEZE["per_domain_utilities"]["closed_form"][FRED]
ad_f = C0_FREEZE["per_domain_utilities"]["adam_mean_seed"][FRED]
d4_rows = {}
for prog in sorted(cf_f):
    delta = float(cf_f[prog] - ad_f[prog])
    d4_rows[prog] = {"u_cf": float(cf_f[prog]), "u_adam": float(ad_f[prog]),
                     "delta_cf_minus_adam": delta,
                     "rel_delta": float(delta / max(abs(ad_f[prog]), 1e-12))}
deltas = np.array([v["delta_cf_minus_adam"] for v in d4_rows.values()])
signs = set(np.sign(np.round(deltas, 6)))
spread = float(deltas.max() - deltas.min())
uniform = spread < 0.02 and signs <= {-1.0, 0.0}
d4_concl = (f"fred 8程序 Δ(cf−adam): 全≤0={signs <= {-1.0, 0.0}}，范围[{deltas.min():.4f},{deltas.max():.4f}]，"
            f"极差={spread:.4f}；{'近均匀水平位移（正则性质）' if uniform else '程序相关（非纯均匀位移）'}")
dump("D4_per_program.json", {"domain": FRED, "source": "C0_FREEZE per_domain_utilities",
     "per_program": d4_rows, "delta_range": [float(deltas.min()), float(deltas.max())],
     "delta_spread": spread, "uniform_level_shift": bool(uniform), "conclusion": d4_concl})
log(f"[D4] {d4_concl}")


# ══ D5. 四域带符号 Δ（raw 程序 U_cf−U_adam 符号 + 相对量；读 C0_FREEZE） ══
per_dom = C0_FREEZE["identity_gate"]["criterion1_raw_level"]["per_domain"]
d5_rows = {}
for d in DOMAINS:
    ucf, uad = float(per_dom[d]["u_cf_raw"]), float(per_dom[d]["u_adam_raw"])
    d5_rows[d] = {"u_cf_raw": ucf, "u_adam_raw": uad, "delta_cf_minus_adam": float(ucf - uad),
                  "rel_delta": float((ucf - uad) / max(abs(uad), 1e-12)),
                  "cf_better": bool(ucf < uad), "gate1_pass": bool(per_dom[d]["pass"])}
n_cf_better = sum(1 for v in d5_rows.values() if v["cf_better"])
rels = {d: d5_rows[d]["rel_delta"] for d in DOMAINS}
d5_concl = (f"四域 raw: 闭式更优(U_cf<U_adam)域数={n_cf_better}/4；rel Δ={ {d: round(r,4) for d,r in rels.items()} }；"
            f"仅 fred 越 ±10% 线（{rels['fred_md']:.4f}），其余|rel|≤{max(abs(r) for d,r in rels.items() if d!='fred_md'):.4f}")
dump("D5_four_domain_delta.json", {"source": "C0_FREEZE criterion1_raw_level", "per_domain": d5_rows,
     "n_cf_better": n_cf_better, "rel_tol": 0.10, "conclusion": d5_concl})
log(f"[D5] {d5_concl}")


# ══ D6. 病态性证据（四域 raw 程序等权 pooled Gram 97×97 条件数） ══
t = time.perf_counter()
d6 = {}
R = ridge_matrix()
for d in DOMAINS:
    views = prepared_views_for_program(by_dom[d], "raw_identity")
    stats = [series_stats(v, stride=int(JUDGE_CFG_FROZEN["stride"]), window_cap=None) for v in views]
    nw = np.array([s.n_windows for s in stats], float)
    pos = nw > 0
    nbar = float(nw[pos].mean())
    w = np.where(pos, nbar / np.where(pos, nw, 1.0), 0.0)      # series_weight="equal"
    G_data = np.zeros_like(R)
    for wi, s in zip(w, stats):
        G_data = G_data + wi * s.G
    G_reg = G_data + DEFAULT_LAM * R
    d6[d] = {
        "n_series_with_windows": int(pos.sum()), "total_windows": int(nw.sum()),
        "cond_data_gram": float(np.linalg.cond(G_data)),
        "cond_regularized_gram": float(np.linalg.cond(G_reg)),
        "min_eig_data": float(np.linalg.eigvalsh((G_data + G_data.T) / 2).min()),
    }
cond_data = {d: d6[d]["cond_data_gram"] for d in DOMAINS}
worst = max(DOMAINS, key=lambda d: cond_data[d])
fred_rank = sorted(DOMAINS, key=lambda d: cond_data[d]).index(FRED) + 1
d6_concl = (f"data-Gram 条件数: { {d: f'{cond_data[d]:.2e}' for d in DOMAINS} }；最病态域={worst}；"
            f"fred 排名 {fred_rank}/4（{'fred 最病态' if worst==FRED else 'fred 非最病态'}）")
dump("D6_gram_condition.json", {"program": "raw_identity", "lam": DEFAULT_LAM,
     "per_domain": d6, "worst_conditioned": worst, "fred_rank": fred_rank, "conclusion": d6_concl})
log(f"[D6] {d6_concl}  ({time.perf_counter()-t:.1f}s)")


# ══ 附带：cycle 期 P_FEATS 指纹只读核对 ══
from SelfEvolvingHarnessTS.p6.harness_state import P_FEATS_FROZEN, P0_FEATURE_ALLOWLIST
from SelfEvolvingHarnessTS.p6 import fast_path as fp
fp_probe = fp.toy_fingerprint(np.asarray(by_dom[FRED][0].history, float))
fpc = {
    "p6_fingerprint_functions": ["fast_path.toy_fingerprint"],
    "toy_fingerprint_keys": sorted(fp_probe),
    "P_FEATS_required_by_allowlist": list(P_FEATS_FROZEN),
    "p0_allowlist": sorted(P0_FEATURE_ALLOWLIST),
    "frozen_p6_fn_providing_P_FEATS": None,
    "conclusion": ("p6 内唯一冻结指纹 = fast_path.toy_fingerprint，仅发 {snr, missing_rate}；"
                   "无任何 p6 冻结函数计算 struct_feats 之 P_FEATS（period/trend_strength/seasonal_strength/"
                   "acf1/stationarity_adf/spectral_entropy/lumpiness/outlier_density）——P_FEATS 仅作为"
                   " harness_state.P_FEATS_FROZEN 的 allowlist 名字存在，计算器不在 p6 冻结面内。"),
}
dump("fingerprint_check.json", fpc)
log(f"[FP-check] {fpc['conclusion']}")

log("\n[DONE] 六项 + 附带 全部落盘 → results/Stage2/C0Run/diag/")
