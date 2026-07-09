"""confirmatory_corpus.py — seeds 20–39 holdout 基底 + A38C 同构补齐（A-40④/A-41③⑥）。

门禁（A-41⑥守卫③，构造性）：任何会读取 seeds 20–39 的入口（`build_confirmatory_base` /
`generate_a38c` / `build_confirmatory_corpus`）都先 `_require_freeze()`——confirmatory_freeze.json
不存在即拒绝。带界 = freeze 中锁定的 A31e dev 实测值（**不看 confirmatory SNR 分布再定**，
A-41③）；接受判据 = perceive 实测 cell 命中（**零 loss 参与**——本模块不 import 任何评估头，
守卫④为结构性保证）。

namespace：`sd=_det_seed(struct,"A38C",cell,k)%2M`、uid=`{struct}:A38C:{cell}:{k}`——与
dev(j 0–19)/A31e/confirmatory 基底(j 20–39)全不交；sd 哈希碰撞对三者并集显式跳过（A-34 独立性）。

产物 `results/A38C/`：protocol.json（先于生成落盘）+ manifest.json（确定性重建）+ audit.md。

运行（freeze 落盘且守卫全过后）：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> \
        D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.confirmatory_corpus --generate
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .augment_corpus import (CELLS, MAX_ATTEMPTS, MISS_OF, BAND_PHASE_FRAC, _audit, _draw_noise,
                             _make_series, slot_counts, RESULTS_A31E)
from .confirmatory_freeze import FREEZE_PATH, HOLDOUT_J, N_TARGET_CONF
from .data.synthetic_gen import RawSeries
from .fast_path.perceive import perceive
from .harness import HarnessState
from .run_variance_decomp import (CUT, DEG_GRID, OUT_RATE, STRUCTS, _clean_signal, _degrade,
                                  _det_seed, build_corpus)

RESULTS_A38C = Path(__file__).resolve().parent / "results" / "A38C"


def _require_freeze() -> dict:
    """A-41 门禁：freeze 先于 holdout（守卫③被测对象）。"""
    if not FREEZE_PATH.exists():
        raise SystemExit("A-41 门禁：confirmatory_freeze.json 未落盘，禁止读取 seeds 20–39 / 生成 A38C。")
    return json.loads(FREEZE_PATH.read_text("utf-8"))


def build_corpus_range(j_lo: int, j_hi: int) -> List[RawSeries]:
    """与 run_variance_decomp.build_corpus 逐字段同构，仅 j 范围参数化（等价性由守卫测试
    build_corpus_range(0,20)==build_corpus(20) 确认——该测试不触碰 holdout）。"""
    out = []
    for struct in STRUCTS:
        for dname, dp in DEG_GRID.items():
            for j in range(j_lo, j_hi):
                sd = _det_seed(struct, dname, j) % 2_000_000
                clean = _clean_signal(struct, sd)
                degraded = _degrade(clean, dp["noise"], dp["miss"], OUT_RATE, sd)
                out.append(RawSeries(
                    pattern=struct, task="forecast", seed=sd, period=24,
                    obs_scale=float(np.std(clean[CUT:])) or 1.0,
                    clean=clean, degraded=degraded,
                    history=degraded[:CUT].copy(), clean_history=clean[:CUT].copy(),
                    future=clean[CUT:].copy(),
                    origin=struct, series_uid=f"{struct}:{dname}:{j}"))
    return out


def build_confirmatory_base() -> List[RawSeries]:
    """holdout 基底 = 原 namespace seeds 20–39。freeze 门禁。"""
    _require_freeze()
    return build_corpus_range(HOLDOUT_J[0], HOLDOUT_J[1] + 1)


def generate_a38c(n_target: int = N_TARGET_CONF, max_attempts: int = MAX_ATTEMPTS,
                  out_dir: Path = RESULTS_A38C, verbose: bool = True) -> dict:
    """A38C 生成：与 A-38 逐规则同构，仅 ①基底=seeds 20–39 ②带界=freeze 预锁 dev 值
    ③namespace 标签="A38C"。protocol 先落盘；接受只看 perceive cell 命中。"""
    freeze = _require_freeze()
    band_split: Dict[str, float] = {c: freeze["a38c"]["band_split_snr_db"][c] for c in CELLS}
    base = build_confirmatory_base()
    h = HarnessState.from_minimal()
    # slot_counts 也会算基底自身的带中位数——**只取 counts/snr_by（审计用），带界一律用冻结值**（A-41③）
    counts, _base_split_UNUSED, snr_by = slot_counts(base)
    dev_sds = {rs.seed for rs in build_corpus(20)}
    a31e_sds = {e["sd"] for e in json.loads((RESULTS_A31E / "manifest.json").read_text("utf-8"))["entries"]}
    existing_sds = dev_sds | a31e_sds | {rs.seed for rs in base}

    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        amendment="A-40④+A-41③", date="2026-07-04", n_target=n_target,
        noise_range=list(freeze["a38c"]["noise_range"]), miss_of=MISS_OF,
        max_attempts_per_slot=max_attempts, band_phase_frac=BAND_PHASE_FRAC,
        namespace=freeze["a38c"]["namespace"],
        band_split_snr_db=band_split,
        band_split_source=freeze["a38c"]["band_split_source"],
        freeze_config_sha=freeze["config_sha"],
        counts_before={f"{c}|{s}": counts.get((c, s), 0) for c in CELLS for s in STRUCTS},
        note=("confirmatory 补齐：目标/带界/namespace 全部先锁（freeze）；接受只看 perceive cell "
              "命中，不看任何 loss；sd 碰撞对 dev∪A31e∪基底显式跳过"))
    (out_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), "utf-8")

    manifest: List[dict] = []
    slot_log: "OrderedDict[str, dict]" = OrderedDict()
    used_sds: set = set()
    band_cap = int(max_attempts * BAND_PHASE_FRAC)

    for cid in CELLS:
        miss = MISS_OF[cid.rsplit("|", 1)[1]]
        split = band_split.get(cid)
        for struct in STRUCTS:
            before = counts.get((cid, struct), 0)
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
                sd = _det_seed(struct, "A38C", cid, k) % 2_000_000
                if sd in existing_sds or sd in used_sds:
                    continue                                   # 哈希碰撞显式跳过（A-34 独立性）
                noise = _draw_noise(sd)
                rs = _make_series(struct, sd, noise, miss, uid=f"{struct}:A38C:{cid}:{k}")
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
    (out_dir / "audit.md").write_text(audit.replace("A-31e 补样审计（A-38 协议）",
                                                    "A38C confirmatory 补样审计（A-40④/A-41③，带界=冻结 dev 值）"),
                                      "utf-8")
    if verbose:
        print(f"\nA38C 生成完成：+{len(manifest)} 条 → {out_dir}")
    return dict(manifest=manifest, slot_log=slot_log)


def load_a38c(manifest_path: Path = RESULTS_A38C / "manifest.json") -> List[RawSeries]:
    doc = json.loads(Path(manifest_path).read_text("utf-8"))
    return [_make_series(e["struct"], e["sd"], e["noise"], e["miss"], e["uid"])
            for e in doc["entries"]]


def manifest_by_uid_a38c(manifest_path: Path = RESULTS_A38C / "manifest.json") -> Dict[str, dict]:
    doc = json.loads(Path(manifest_path).read_text("utf-8"))
    return {e["uid"]: e for e in doc["entries"]}


def build_confirmatory_corpus() -> List[RawSeries]:
    """confirmatory 语料 = 基底(seeds 20–39) + A38C 补齐。freeze 门禁 + manifest 硬前置。"""
    _require_freeze()
    if not (RESULTS_A38C / "manifest.json").exists():
        raise SystemExit("A38C manifest 缺失：先跑 confirmatory_corpus --generate（A-40④ 数据构造预锁）。")
    return build_confirmatory_base() + load_a38c()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="生成 A38C 补样（一次性；读取 seeds 20–39）")
    args = ap.parse_args()
    if not args.generate:
        raise SystemExit("显式加 --generate（该动作读取 holdout 基底，freeze 门禁生效）。")
    if (RESULTS_A38C / "manifest.json").exists():
        raise SystemExit("A38C manifest 已存在；一次性产物不覆盖。")
    generate_a38c()


if __name__ == "__main__":
    main()
