# -*- coding: utf-8 -*-
"""diagnostics/p6_c0_diag_fred_wave3.py — D10 前置门：Adam 极限收敛核查（纯诊断，只读复用 p6 冻结件）。

假说：闭式 ridge-DLinear 与 Adam-DLinear 是同一函数族（DLinear 双线性头 = φ=[trend;season;1] 的
线性头），差异纯属优化路径。判决：fred raw，Adam + L2(λ=1e-3) + 延长预算 ×8(960 epoch) +
lr 余弦衰减，3 seeds → U 是否收敛到 U_cf=0.2011（≤±3%，[0.19507,0.20714]）。
过 → 两臂在共享目标极限一致（gap=优化路径效应，非隐藏 bug）；不过 → 停止报告、不实施 A1。
只用 C0 legacy 数据；不触碰 V/U、不联网、不改冻结面。结果 → diag/D10_limit_check.json。
运行：D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.diagnostics.p6_c0_diag_fred_wave3
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
    build_episodes, make_real_degrade_fn, prepared_views_for_program,
    _torch_windows_from_views, JUDGE_CFG_FROZEN, HORIZON,
)
from SelfEvolvingHarnessTS.p6.judge_closed_form import fit_domain, CONTEXT_LEN

PKG = os.path.join(_ROOT, "SelfEvolvingHarnessTS")
OUT = os.path.join(PKG, "results/Stage2/C0Run/diag")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
STRIDE = int(JUDGE_CFG_FROZEN["stride"])
EPOCHS = 960          # ×8（冻结 baseline 120）
WD = 1e-3             # 与判官 ridge λ 同值
LR0 = 1e-2            # 冻结起始 lr，余弦衰减到 0
TARGET = 0.20110639714410383   # U_cf(fred raw)（官方 C0_FREEZE）
TOL = 0.03            # ±3%


def log(*a):
    print(*a, flush=True)


# ── C0 物化链（与官方同数据同协议） ──
ledger = [json.loads(l) for l in open(os.path.join(PKG, "results/Stage2/P6Probes/exposure_ledger.jsonl"),
                                      encoding="utf-8") if l.strip()]
u_excl = json.load(open(os.path.join(PKG, "results/Stage2/P6Probes/u_admission_v2_traffic_hourly.json"),
                        encoding="utf-8"))["all_probe_consumed_item_ids"]
manifest = build_manifest(ledger, u_excl)
validate_manifest(manifest)
assert manifest.manifest_sha == "5c768155a47c1b4cc033a0eaa724830a7055e343439ef77447ad3e6244e21720"
c0_block = manifest.block("C0")
npz = np.load(os.path.join(PKG, "data/_artifacts/monash_clean.npz"), allow_pickle=True)["clean"]
meta = [json.loads(l) for l in open(os.path.join(PKG, "data/_artifacts/monash_clean.meta.jsonl"),
                                    encoding="utf-8") if l.strip()]
by_uid = {f"{m['config']}:{m['item_id']}": (m["config"], m["item_id"], npz[i]) for i, m in enumerate(meta)}
episodes = build_episodes([by_uid[u] for u in c0_block], make_real_degrade_fn())
fred_views = prepared_views_for_program([ep for ep in episodes if ep.config == "fred_md"], "raw_identity")
U_cf = float(fit_domain(fred_views, **JUDGE_CFG_FROZEN).utility)
log(f"[setup] manifest_sha={manifest.manifest_sha}  U_cf(fred raw)={U_cf:.6f} (target={TARGET:.6f})")


def adam_l2_cosine(views, seed, epochs=EPOCHS, wd=WD, lr0=LR0, stride=STRIDE, capture_every=120):
    """诊断变体：复用冻结 DLinear + 窗协议 + seed；加 L2(weight_decay) + 余弦 lr 衰减 + 延长预算。"""
    import torch
    import torch.nn.functional as F
    from SelfEvolvingHarnessTS.evaluators import _torch_models as tm
    X, Y, evals, futs = _torch_windows_from_views(views, stride, None)
    old = tm.DEVICE
    tm.DEVICE = "cpu"
    try:
        torch.use_deterministic_algorithms(True)
        tm.seed_all(int(seed))
        net = tm.DLinear(CONTEXT_LEN, HORIZON).to("cpu")
        net.train()
        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        opt = torch.optim.Adam(net.parameters(), lr=lr0, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs))
        n = len(Xt)
        curve = []
        for ep in range(int(epochs)):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                F.mse_loss(net(Xt[idx]), Yt[idx]).backward()
                opt.step()
            sched.step()
            if capture_every and (ep + 1) % capture_every == 0:
                net.eval()
                with torch.no_grad():
                    preds = net(torch.tensor(evals, dtype=torch.float32)).cpu().numpy()
                curve.append({"epoch": ep + 1,
                              "u": float(np.sqrt(np.mean((preds - futs) ** 2, axis=1)).mean())})
                net.train()
        net.eval()
        with torch.no_grad():
            preds = net(torch.tensor(evals, dtype=torch.float32)).cpu().numpy()
    finally:
        tm.DEVICE = old
    u = float(np.sqrt(np.mean((preds - futs) ** 2, axis=1)).mean())
    return u, curve


t = time.perf_counter()
per_seed, curves = {}, {}
for s in SEEDS:
    u, curve = adam_l2_cosine(fred_views, s)
    per_seed[str(s)] = u
    curves[str(s)] = curve
    log(f"  [D10] seed={s} U={u:.6f}  ({time.perf_counter()-t:.1f}s cum)")

u_mean = float(np.mean(list(per_seed.values())))
rel = float((u_mean - TARGET) / TARGET)
converged = bool(abs(rel) <= TOL)
concl = (f"Adam+L2(1e-3)+×8(960ep)+cosine → U_mean={u_mean:.6f}；U_cf={TARGET:.6f}；"
         f"rel offset={rel*100:+.3f}%（阈 ±{TOL*100:.0f}%）→ {'收敛（共享目标极限一致，gap=优化路径）' if converged else '未收敛（不满足前置门）'}")

rec = {
    "purpose": "证明闭式 ridge 与 Adam-DLinear 在共享目标极限一致（残差 gap=优化路径效应，非隐藏 bug）",
    "domain": "fred_md", "program": "raw_identity",
    "config": {"epochs": EPOCHS, "epoch_multiple_of_frozen": EPOCHS // 120, "weight_decay": WD,
               "lr0": LR0, "lr_schedule": "CosineAnnealingLR(T_max=epochs)", "seeds": SEEDS,
               "stride": STRIDE, "reuse": "冻结 DLinear + _torch_windows_from_views + seed_all"},
    "u_cf_target": TARGET, "tol_rel": TOL, "noninf_band": [TARGET * (1 - TOL), TARGET * (1 + TOL)],
    "u_adam_per_seed": per_seed, "u_adam_mean": u_mean,
    "signed_relative_offset": rel, "converged_within_tol": converged,
    "u_trajectory_per_seed": curves,
    "reference_frozen": {"u_adam_baseline_noL2_120ep": 0.230857623954379,
                         "u_adam_L2_120ep_D2": 0.229367, "u_adam_x4_480ep_D3": 0.2392},
    "conclusion": concl,
    "verdict": "GO_for_A1" if converged else "STOP_no_A1",
    "_written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
json.dump(rec, open(os.path.join(OUT, "D10_limit_check.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=float)
log(f"[D10] {concl}")
log(f"[D10] verdict={rec['verdict']}  → wrote D10_limit_check.json")
