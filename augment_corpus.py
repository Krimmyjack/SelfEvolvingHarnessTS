"""augment_corpus.py — A-31e 平衡补样（A-38 协议，评审第十四轮，2026-07-04）。

目标（生成前在 protocol.json 锁定，禁 optional stopping）：每个可行 cell×origin 槽位补至
N_TARGET=40 uid（只增不删）；snrHigh×S_ar 结构性不可达（AR 能量被 SNR 估计器计入残差）→
记录 infeasible。接受判据 = perceive 实测 cell 命中（**不看任何 loss** → 无自适应偏差）。

种子命名空间：`sd = _det_seed(struct, "A31e", cell, k) % 2M`、uid=`{struct}:A31e:{cell}:{k}`
—— 与 dev（j 0–19）/confirmatory（j 20–39，续封存）namespace 不交；与现有 corpus 的 sd 值
哈希碰撞显式跳过（护 A-34 cell 独立性）。

连续噪声：noise ~ loguniform[0.03, 2.0]（per-attempt 确定性 RNG）；miss ∈ {0, 0.06} 按目标 cell。
SNR 分布 best-effort 匹配：cell 内以现有语料实测 SNR 中位数分 2 带、缺额对半入带；
attempt 超 2/3 上限后滚存另一带（rolled 记录）。

产物 `results/A31e/`：protocol.json（先于生成落盘）+ manifest.json（重建确定性）+ audit.md。
补样语料 = E-3.2 development 语料；confirmatory 仍 = seeds 20–39 原 namespace。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.augment_corpus --generate
"""
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .harness import HarnessState
from .data.synthetic_gen import RawSeries
from .fast_path.perceive import perceive
from .run_variance_decomp import (CUT, OUT_RATE, STRUCTS, _clean_signal, _degrade,
                                  _det_seed, assign_cells, build_corpus)

RESULTS_A31E = Path(__file__).resolve().parent / "results" / "A31e"
CELLS = ("forecast|snrHigh|full", "forecast|snrHigh|miss",
         "forecast|snrLow|full", "forecast|snrLow|miss")
N_TARGET = 40
NOISE_LO, NOISE_HI = 0.03, 2.0
MAX_ATTEMPTS = 2000                    # per slot；infeasible 槽位以 0/2000 为证据
MISS_OF = {"full": 0.0, "miss": 0.06}
BAND_PHASE_FRAC = 2 / 3                # 前 2/3 attempt 按带配额，之后允许滚存


def _make_series(struct: str, sd: int, noise: float, miss: float, uid: str) -> RawSeries:
    """与 build_corpus 同构地构造一条补样序列（确定性，由 (struct, sd, noise, miss) 完全决定）。"""
    clean = _clean_signal(struct, sd)
    degraded = _degrade(clean, noise, miss, OUT_RATE, sd)
    return RawSeries(pattern=struct, task="forecast", seed=sd, period=24,
                     obs_scale=float(np.std(clean[CUT:])) or 1.0,
                     clean=clean, degraded=degraded,
                     history=degraded[:CUT].copy(), clean_history=clean[:CUT].copy(),
                     future=clean[CUT:].copy(),
                     origin=struct, series_uid=uid)


def _draw_noise(sd: int) -> float:
    rng = np.random.default_rng(sd + 777)
    return float(np.exp(rng.uniform(math.log(NOISE_LO), math.log(NOISE_HI))))


def slot_counts(base_corpus) -> Tuple[Dict[Tuple[str, str], int], Dict[str, float], Dict[str, Dict[str, List[float]]]]:
    """现状统计：(cell,origin)→uid 数、cell→SNR 中位数（带界）、cell→origin→SNR 列表。"""
    cells, snr_of = assign_cells(base_corpus)
    counts: Dict[Tuple[str, str], int] = {(c, s): 0 for c in CELLS for s in STRUCTS}
    snr_by: Dict[str, Dict[str, List[float]]] = {c: {s: [] for s in STRUCTS} for c in CELLS}
    band_split: Dict[str, float] = {}
    for cid, series in cells.items():
        if cid not in counts and (cid, STRUCTS[0]) not in counts:
            continue
        vals = []
        for rs in series:
            counts[(cid, rs.origin)] = counts.get((cid, rs.origin), 0) + 1
            v = snr_of[rs.series_uid]
            vals.append(v)
            snr_by.setdefault(cid, {s: [] for s in STRUCTS})[rs.origin].append(v)
        band_split[cid] = float(np.median(vals))
    return counts, band_split, snr_by


def generate(n_target: int = N_TARGET, max_attempts: int = MAX_ATTEMPTS,
             out_dir: Path = RESULTS_A31E, verbose: bool = True) -> dict:
    """A-38 生成主流程：protocol 先落盘 → 逐槽位接受采样 → manifest + audit。全程确定性。"""
    base = build_corpus(20)
    h = HarnessState.from_minimal()
    counts, band_split, snr_by = slot_counts(base)
    existing_sds = {rs.seed for rs in base}

    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        amendment="A-38", date="2026-07-04", n_target=n_target,
        noise_range=[NOISE_LO, NOISE_HI], miss_of=MISS_OF, max_attempts_per_slot=max_attempts,
        band_phase_frac=BAND_PHASE_FRAC,
        namespace='sd=_det_seed(struct,"A31e",cell,k)%2_000_000; uid="{struct}:A31e:{cell}:{k}"',
        band_split_snr_db={c: band_split.get(c) for c in CELLS},
        counts_before={f"{c}|{s}": counts[(c, s)] for c in CELLS for s in STRUCTS},
        note=("目标先锁定（本文件先于生成写出）；接受只看 perceive cell 命中，不看任何 loss；"
              "seeds 20–39 confirmatory namespace 未触碰"))
    (out_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), "utf-8")

    manifest: List[dict] = []
    slot_log: "OrderedDict[str, dict]" = OrderedDict()
    used_sds: set = set()
    band_cap = int(max_attempts * BAND_PHASE_FRAC)

    for cid in CELLS:
        miss = MISS_OF[cid.rsplit("|", 1)[1]]
        split = band_split.get(cid)
        for struct in STRUCTS:
            before = counts[(cid, struct)]
            deficit = max(0, n_target - before)
            key = f"{cid}|{struct}"
            if deficit == 0:
                slot_log[key] = dict(before=before, deficit=0, accepted=0, rolled=0,
                                     attempts=0, infeasible=False)
                continue
            need = {"lo": (deficit + 1) // 2, "hi": deficit // 2}
            accepted: List[dict] = []
            rolled = 0
            attempts = 0
            for k in range(max_attempts):
                if need["lo"] + need["hi"] == 0:
                    break
                attempts += 1
                sd = _det_seed(struct, "A31e", cid, k) % 2_000_000
                if sd in existing_sds or sd in used_sds:
                    continue                                   # 哈希碰撞显式跳过（A-34 独立性）
                noise = _draw_noise(sd)
                rs = _make_series(struct, sd, noise, miss, uid=f"{struct}:A31e:{cid}:{k}")
                key_p = perceive(rs.history, "forecast", h)
                if key_p["cell_id"] != cid:
                    continue
                snr = float(key_p["pattern"]["struct_feats"].get("SNR", 0.0))
                band = "lo" if (split is not None and snr < split) else "hi"
                if need[band] > 0:
                    need[band] -= 1
                elif k >= band_cap:
                    other = "hi" if band == "lo" else "lo"
                    if need[other] <= 0:
                        continue
                    need[other] -= 1
                    rolled += 1
                else:
                    continue
                used_sds.add(sd)
                accepted.append(dict(uid=rs.series_uid, struct=struct, cell=cid, k=k,
                                     sd=int(sd), noise=noise, miss=miss,
                                     snr_measured=snr, band=band))
            manifest.extend(accepted)
            slot_log[key] = dict(before=before, deficit=deficit, accepted=len(accepted),
                                 rolled=rolled, attempts=attempts,
                                 infeasible=(len(accepted) == 0))
            if verbose:
                tag = "INFEASIBLE" if slot_log[key]["infeasible"] else f"+{len(accepted)}"
                print(f"  {key:38s} before={before:2d} deficit={deficit:2d} -> {tag:10s} "
                      f"(attempts={attempts}, rolled={rolled})", flush=True)

    (out_dir / "manifest.json").write_text(
        json.dumps(dict(protocol="protocol.json", n_aug=len(manifest), entries=manifest),
                   ensure_ascii=False, indent=1), "utf-8")
    audit = _audit(base, manifest, slot_log, band_split, snr_by, n_target)
    (out_dir / "audit.md").write_text(audit, "utf-8")
    if verbose:
        print(f"\n生成完成：+{len(manifest)} 条补样 → {out_dir}")
    return dict(manifest=manifest, slot_log=slot_log)


def _audit(base, manifest, slot_log, band_split, snr_by_before, n_target) -> str:
    """审计：per cell×origin 前后计数 + SNR 分位 + 结构间 overlap（[p10,p90] 区间交并比）。"""
    aug_snr: Dict[str, Dict[str, List[float]]] = {c: {s: [] for s in STRUCTS} for c in CELLS}
    for e in manifest:
        aug_snr[e["cell"]][e["struct"]].append(e["snr_measured"])
    lines = ["# A-31e 补样审计（A-38 协议）", "",
             f"日期：2026-07-04　目标 N={n_target}/槽位　补样总数 +{len(manifest)}　"
             f"接受判据=perceive cell 命中（零 loss 参与）", "",
             "| cell | origin | before | after | SNR p10/p50/p90 (dB, after) | rolled | 状态 |",
             "|---|---|--:|--:|---|--:|---|"]
    for key, lg in slot_log.items():
        cid = "|".join(key.split("|")[:3])              # key = "forecast|snrX|Y|S_struct"
        struct = key.split("|")[3]
        vals = snr_by_before.get(cid, {}).get(struct, []) + aug_snr.get(cid, {}).get(struct, [])
        if vals:
            q = np.percentile(vals, [10, 50, 90])
            qs = f"{q[0]:+.1f} / {q[1]:+.1f} / {q[2]:+.1f}"
        else:
            qs = "—"
        status = ("结构性 infeasible" if lg["infeasible"] and lg["deficit"] > 0
                  else ("已满" if lg["deficit"] == 0 else
                        ("部分" if lg["accepted"] < lg["deficit"] else "补齐")))
        lines.append(f"| {cid} | {struct} | {lg['before']} | {lg['before'] + lg['accepted']} "
                     f"| {qs} | {lg['rolled']} | {status} |")
    lines += ["", "## 结构间 SNR overlap（cell 内，[p10,p90] 区间交/并）", ""]
    for cid in CELLS:
        spans = {}
        for s in STRUCTS:
            vals = snr_by_before.get(cid, {}).get(s, []) + aug_snr.get(cid, {}).get(s, [])
            if len(vals) >= 3:
                spans[s] = tuple(np.percentile(vals, [10, 90]))
        if len(spans) >= 2:
            lo = max(v[0] for v in spans.values())
            hi = min(v[1] for v in spans.values())
            u_lo = min(v[0] for v in spans.values())
            u_hi = max(v[1] for v in spans.values())
            ov = max(0.0, hi - lo) / max(1e-9, u_hi - u_lo)
            lines.append(f"- {cid}: overlap={ov:.2f}　带界(existing median)={band_split.get(cid, float('nan')):+.2f}dB　"
                         + "　".join(f"{s}[{v[0]:+.1f},{v[1]:+.1f}]" for s, v in spans.items()))
    lines += ["", "解释边界：overlap<1 = 结构与连续 SNR 部分纠缠不可完全解开（S_ar 实测 SNR 有结构性上限）",
              "→ E-3.2 判据 (vi) 连续 SNR residualization + SNR 分层置换仍为必要控制（A-37③）。", ""]
    return "\n".join(lines)


def load_augmented(manifest_path: Path = RESULTS_A31E / "manifest.json") -> List[RawSeries]:
    """从 manifest 确定性重建补样序列（不需 perceive）。"""
    doc = json.loads(Path(manifest_path).read_text("utf-8"))
    return [_make_series(e["struct"], e["sd"], e["noise"], e["miss"], e["uid"])
            for e in doc["entries"]]


def build_augmented_corpus(n_seeds: int = 20,
                           manifest_path: Path = RESULTS_A31E / "manifest.json") -> List[RawSeries]:
    """E-3.2 development 语料 = dev corpus（seeds 0..n_seeds-1）+ A-31e 补样。confirmatory 语料另行。"""
    return build_corpus(n_seeds) + load_augmented(manifest_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="按 A-38 协议生成补样（写 results/A31e/）")
    ap.add_argument("--n-target", type=int, default=N_TARGET)
    ap.add_argument("--audit-only", action="store_true", help="只打印现状 cell×origin 计数")
    args = ap.parse_args()
    if args.audit_only or not args.generate:
        counts, band_split, _ = slot_counts(build_corpus(20))
        print("现状 cell×origin uid 计数（目标 N=%d）：" % args.n_target)
        for c in CELLS:
            row = "  ".join(f"{s}={counts[(c, s)]:2d}" for s in STRUCTS)
            print(f"  {c:26s} {row}   带界={band_split.get(c, float('nan')):+.2f}dB")
        return
    generate(n_target=args.n_target)


if __name__ == "__main__":
    main()
