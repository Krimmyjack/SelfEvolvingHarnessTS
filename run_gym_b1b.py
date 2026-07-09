"""run_gym_b1b.py — B1b-mini gym：开放程序 proposer 竞技场（压缩计划 §2，正式擂台）。

**项目身份主判据**：LLM(gpt-5.4-mini)+memory 能否在**同一可执行开放程序空间、同候选预算、同评估
口径**下，提出枚举/搜索找不到、可迁移的 harness 结构编辑（1–3 步新算子链，改顺序/参数/适用 cell）。

与 B1a（run_proposer.py）本质不同：候选是**新程序**，损失不在缓存 L_test → **须真实执行**
（executor：ProgramSpec→ActionSpec→ActionCompiler→fast_process→FrozenProbe OOF nRMSE）。
**红线**：frozen 基线与新程序**同走 executor**（同尺度）；缓存 L_test(nested DLinear) 仅供 report 提示
与 frozen 路由选动作（尺度无关），**决策损失一律 executor**。

架构：
  executor（贵，每不同**算子链**一次）：chain → {uid: oof_loss} 全语料，按 chain_sha 落盘（三臂/重跑免费）。
  gym（便宜）：LOFO select-on-selection→eval-on-heldout，utility=frozen_regret−program_regret（oracle 抵消）。

三臂：random（负对照）/ det_search（确定性预算搜索，强基线）/ llm_mem（gpt-5.4-mini+已验证编辑记忆）。
天花板=gpt-5.4-mini（**败非决定性**→触发升档；**胜=强正**）。dev-gate 先在预锁 challenge 族，过再扩。

运行：
  --dry-run     random+det，小 budget，2 dev 族，子集语料，**无 LLM 外呼**（验证 gym 管线）。
  --formal      三臂全 budget 全语料（**待 prereg 审后开**）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .evaluators.frozen_probe import FrozenProbe
from .policy import FrozenArmRouterPolicy
from .policy.action_spec import action_menu_v1
from .policy.program_edit import (DENOISERS, IMPUTERS, OUTLIERS, WINDOW_GRID, WINDOWED,
                                  ProgramSpec, is_novel, to_action_spec, validate)
from .e32_policy import PRUNED_POOL_CORE
from .run_proposer_b1b import execute_action_losses, grouped_folds
from .run_updater2 import REC_PATH, Served
from .s2_corpus import S2_FAMILIES, build_s2_dev

OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "ProposerB1b"
DELTA_SAFE = 0.05
CHALLENGE = ("S_both", "S_regime", "S_trend", "S_multiseason")   # dev-gate 预锁族（先过再扩）


# ════════════════════════════ executor（按 chain 落盘缓存）════════════════════════════
def _exec_dir(tag: str) -> Path:
    d = OUT / "exec_cache" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def exec_cached(key: str, action_spec, series, fp, fold_of, tag: str) -> Dict[str, float]:
    """执行一条链（池动作或新程序）→ {uid: oof_loss}，按 key 落盘（命中即免执行）。"""
    safe = key.replace(":", "_")                                 # Windows 文件名禁 ':'
    path = _exec_dir(tag) / f"{safe}.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    losses = execute_action_losses(action_spec, series, fp, fold_of)
    path.write_text(json.dumps(losses, ensure_ascii=False), "utf-8")
    return losses


def pool_baseline(series, fp, fold_of, tag: str, verbose=True) -> Dict[str, Dict[str, float]]:
    """10 池动作各执行一次（executor 尺度）→ {action_id: {uid: loss}}。frozen 基线之源。"""
    menu = action_menu_v1()
    out: Dict[str, Dict[str, float]] = {}
    t0 = time.time()
    for i, aid in enumerate(PRUNED_POOL_CORE):
        out[aid] = exec_cached(f"pool::{aid}", menu.actions[aid], series, fp, fold_of, tag)
        if verbose:
            print(f"  [pool {i+1}/10] {aid:16s} n={len(out[aid])} [{time.time()-t0:.0f}s]", flush=True)
    return out


def program_losses(spec: ProgramSpec, series, fp, fold_of, tag: str) -> Dict[str, float]:
    return exec_cached(f"prog::{spec.chain_sha()}", to_action_spec(spec), series, fp, fold_of, tag)


# ════════════════════════════ frozen 路由 pick（尺度无关，→ 池动作 id）════════════════════════════
def frozen_picks(by_uid: Dict[str, dict], uids: List[str]) -> Dict[str, str]:
    pol = FrozenArmRouterPolicy.load_frozen("dp_abstain")
    srv = Served("frozen", pol.arm, pol.actions)
    picks = srv.picks([by_uid[u] for u in uids])
    return {u: picks[i] for i, u in enumerate(uids)}


def frozen_loss_of(uids: List[str], fp_pick: Dict[str, str],
                   pool_loss: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """frozen 基线损失 = 路由所选池动作在 **executor 尺度**下的损失（尺度与新程序一致）。"""
    fl: Dict[str, float] = {}
    for u in uids:
        a = fp_pick.get(u)
        if a in pool_loss and u in pool_loss[a]:
            fl[u] = pool_loss[a][u]
    return fl


# ════════════════════════════ mining 报告（executor 尺度，per 族×cell，禁跨族聚合）════════════════════════════
def build_report(sel_uids: List[str], by_uid, pool_loss, frozen_loss) -> dict:
    """每 (族, cell)：frozen regret（executor 尺度，room-for-improvement）+ 该 cell 最优池链（提示 near-good 链）。
    **只在 selection 族上建**（held-out 族不入 → LOFO 无泄漏）。"""
    menu = action_menu_v1()
    chain_of = {aid: [s.op for s in menu.actions[aid].steps] for aid in PRUNED_POOL_CORE}
    agg: Dict[Tuple[str, str], dict] = {}
    for u in sel_uids:
        r = by_uid[u]
        if u not in frozen_loss:
            continue
        key = (r["origin"], r["cell"])
        d = agg.setdefault(key, dict(n=0, fro=0.0, best_a=None, best_l=1e9))
        d["n"] += 1
        # frozen regret 相对该 cell 内最优池动作（executor 尺度）
        cell_pool = [(a, pool_loss[a][u]) for a in PRUNED_POOL_CORE if u in pool_loss[a]]
        if cell_pool:
            ba, bl = min(cell_pool, key=lambda t: t[1])
            d["fro"] += frozen_loss[u] - bl                       # >0 = frozen 在该 cell 有改进空间
            if bl < d["best_l"]:
                d["best_l"], d["best_a"] = bl, ba
    fams: Dict[str, list] = {}
    for (fam, cell), d in agg.items():
        if d["n"] < 3:
            continue
        fams.setdefault(fam, []).append(dict(
            cell=cell, n=d["n"], frozen_gap=round(d["fro"] / d["n"], 4),
            best_pool=d["best_a"], best_pool_chain=chain_of.get(d["best_a"], [])))
    for fam in fams:
        fams[fam].sort(key=lambda s: -s["frozen_gap"])
    return dict(families=fams)


def report_text(report: dict) -> str:
    L = ["Per-family mining signals (frozen router's mean improvement-room vs the best pooled "
         "cleaning chain, on the SAME frozen-probe scale; higher frozen_gap = more room to beat "
         "frozen in that observable cell):"]
    for fam, sig in report["families"].items():
        L.append(f"\n[family {fam}]")
        for s in sig:
            L.append(f"  cell {s['cell']} (n={s['n']}): frozen_gap={s['frozen_gap']:+.3f} | "
                     f"best pooled chain so far = {s['best_pool_chain']}")
    return "\n".join(L)


# ════════════════════════════ 程序空间辅助 ════════════════════════════
def _cells_from(report: dict) -> List[str]:
    return sorted({s["cell"] for sig in report["families"].values() for s in sig})


def _mk(steps, scope) -> Optional[ProgramSpec]:
    spec = ProgramSpec(steps=tuple(steps), scope=tuple(scope), provenance={})
    ok, _ = validate(spec)
    return spec if (ok and is_novel(spec)) else None


# ════════════════════════════ 三臂 proposer（同 budget，同空间）════════════════════════════
class RandomProgramProposer:
    name = "random"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def propose(self, report, budget, cells, memory=None):
        cands, tries = [], 0
        while len(cands) < budget and tries < budget * 40:
            tries += 1
            steps = [(str(self.rng.choice(IMPUTERS)), ())]
            for _ in range(int(self.rng.integers(0, 3))):          # 0..2 后续步
                op = str(self.rng.choice(OUTLIERS + DENOISERS))
                params = (("window", int(self.rng.choice(WINDOW_GRID))),) if op in WINDOWED else ()
                steps.append((op, params))
            k = int(self.rng.integers(1, len(cells) + 1))
            scope = sorted(self.rng.choice(cells, size=k, replace=False).tolist())
            spec = _mk(steps, scope)
            if spec and spec.chain_sha() not in {c.chain_sha() for c in cands}:
                cands.append(spec)
        return cands


# 确定性候选链日程（canonical 顺序；覆盖 impute×body 空间的强基线枚举）
_DET_BODIES: List[list] = [
    [],
    [("winsorize", ())],
    [("denoise_median", (("window", 9),))],
    [("denoise_median", (("window", 25),))],
    [("denoise_stl", ())],
    [("denoise_savgol", ())],
    [("smooth_ma", (("window", 5),))],
    [("winsorize", ()), ("denoise_median", (("window", 9),))],
    [("winsorize", ()), ("denoise_stl", ())],
    [("winsorize", ()), ("smooth_ma", (("window", 5),))],
    [("outlier_mad", ()), ("denoise_median", (("window", 9),))],
    [("winsorize", ()), ("denoise_median", (("window", 25),))],
]


def _det_schedule() -> List[Tuple[str, list]]:
    return [(imp, body) for imp in IMPUTERS for body in _DET_BODIES]


class DetSearchProgramProposer:
    """确定性预算搜索（强基线）：canonical (imputer×body) 日程枚举出所有 novel 合法链，按 frozen_gap
    降序取高 room 的 cell，第 i 候选 = 第 i 条 novel 链 scope={第 i 高 room cell}——确定性、覆盖空间、
    定向 frozen 弱处。同 budget、同 executor 评估 → 对 LLM 的公平强基线。"""
    name = "det_search"

    def propose(self, report, budget, cells, memory=None):
        ranked_cells = [s["cell"] for sig in report["families"].values() for s in sig]
        ranked_cells = sorted(set(ranked_cells),
                              key=lambda c: -max((s["frozen_gap"] for sig in report["families"].values()
                                                  for s in sig if s["cell"] == c), default=0.0))
        if not ranked_cells:
            ranked_cells = list(cells)
        chains: List[ProgramSpec] = []
        seen = set()
        for imp, body in _det_schedule():
            cell = ranked_cells[len(chains) % len(ranked_cells)]
            spec = _mk([(imp, ())] + list(body), [cell])
            if spec and spec.chain_sha() not in seen:
                seen.add(spec.chain_sha())
                chains.append(spec)
            if len(chains) >= budget:
                break
        return chains[:budget]


_LLM_SYS = ("You are a careful time-series data-readiness policy editor. You compose SHORT "
            "preprocessing PROGRAMS (1-3 ordered operators) that a frozen forecast judge will "
            "score. Output JSON only.")

_LLM_INSTR = (
    "You edit a FROZEN preprocessing router by proposing NOVEL cleaning PROGRAMS for observable "
    "cells (task|snr-bin|miss-bin), each program a 1-3 step ordered operator chain.\n"
    "GRAMMAR (stay strictly inside):\n"
    f"  step1 (required) = imputer ∈ {list(IMPUTERS)}  (handles missing values first)\n"
    f"  step2/3 (optional) = outlier ∈ {list(OUTLIERS)} | denoiser ∈ {list(DENOISERS)}\n"
    f"  windowed denoisers {sorted(WINDOWED)} need a window ∈ {list(WINDOW_GRID)}\n"
    "  ORDER MATTERS: denoise∘winsorize ≠ winsorize∘denoise; no adjacent duplicate ops.\n"
    "TIME-SERIES REASONING you should use:\n"
    "  - High frozen_gap cell = frozen leaves room; propose a chain that plausibly cleans that "
    "structure (e.g. heavy median window can wipe seasonality — AVOID window≈period on seasonal "
    "families; light winsorize+savgol preserves shape; STL removes trend+season).\n"
    "  - Keep programs SCOPED to the specific cells where the signal is; do NOT over-generalize a "
    "family-specific harm to all cells.\n"
    'Return JSON ONLY: {"programs": [ {"steps": [ {"op": "impute_linear"}, '
    '{"op": "denoise_median", "window": 9} ], "scope": ["forecast|snrLow|full", ...]} ]}\n'
    "Give EXACTLY N distinct programs (distinct operator chains).")


class LLMProgramProposer:
    name = "llm_mem"

    def __init__(self, llm, with_memory: bool = True):
        self.llm = llm
        self.with_memory = with_memory
        self.name = "llm_mem" if with_memory else "llm_nomem"

    def propose(self, report, budget, cells, memory=None):
        mem_txt = ""
        if self.with_memory and memory:
            mem_txt = ("\n\nPast VERIFIED-successful programs (retrieved experience, chain → where "
                       "it helped):\n" + "\n".join(f"  {m}" for m in memory[:6]))
        user = (f"{report_text(report)}{mem_txt}\n\nObservable cells available: {cells}\n"
                f"N = {budget}.\n{_LLM_INSTR}")
        from .llm.client import extract_json
        raw = self.llm(_LLM_SYS, user, nonce=0)
        spec = extract_json(raw)
        cands: List[ProgramSpec] = []
        seen = set()
        for p in (spec or {}).get("programs", [])[:budget * 2]:
            steps = []
            for st in (p.get("steps") or [])[:3]:
                op = st.get("op")
                params = (("window", int(st["window"])),) if (op in WINDOWED and "window" in st) else ()
                if op:
                    steps.append((op, params))
            scope = [c for c in (p.get("scope") or []) if c in cells]
            cand = _mk(steps, scope) if (steps and scope) else None
            if cand and cand.chain_sha() not in seen:
                seen.add(cand.chain_sha())
                cands.append(cand)
            if len(cands) >= budget:
                break
        return cands, raw


def _menu_chains_text() -> str:
    """现有 15 menu 动作的 compiled (op,window) 链——明示给天花板 LLM（消除信息不对称，评审第 40 轮）。"""
    lines = []
    for aid, spec in action_menu_v1().actions.items():
        chain = "->".join(s.op + (f"(w{s.params['window']})" if s.params.get("window") else "")
                          for s in spec.steps)
        lines.append(f"  {chain}")
    return "\n".join(lines)


_LLM_INSTR_CEILING = _LLM_INSTR.replace(
    "Give EXACTLY N distinct programs (distinct operator chains).",
    "These chains ALREADY EXIST in the frozen menu — a program equal to any of them (after filling "
    "default windows: savgol→w11, median→w5) is NOT novel and will be DISCARDED. Propose genuinely "
    "DIFFERENT chains (new op combinations, new imputers, new window doses, new orders):\n"
    "{menu}\n"
    "Propose up to N distinct NOVEL programs (we evaluate the first 10 that pass the novelty gate).")


class LLMCeilingProposer:
    """B1b-ceiling 臂（评审第 40 轮公平性修复）：**明示 menu 链**（消除信息不对称）+ 提议至多 20 →
    gate/novelty/dedup 后取前 `eval_budget` 条评估（评估预算 = det）。捕获每条拒因（写盘审计）。
    只改 prompt（示 menu）+ 候选上限，**不改 memory/judge/评分**。"""
    name = "llm_ceiling"

    def __init__(self, llm, propose_budget: int = 20, eval_budget: int = 10):
        self.llm = llm
        self.propose_budget = propose_budget
        self.eval_budget = eval_budget
        self.rejections: List[list] = []                     # [[chain_label, reason], ...]（每折覆盖）

    def propose(self, report, budget, cells, memory=None):
        from .llm.client import extract_json
        mem_txt = ("\n\nPast VERIFIED-successful programs (retrieved experience, chain → where it "
                   "helped):\n" + "\n".join(f"  {m}" for m in memory[:6])) if memory else ""
        instr = _LLM_INSTR_CEILING.replace("{menu}", _menu_chains_text())
        user = (f"{report_text(report)}{mem_txt}\n\nObservable cells available: {cells}\n"
                f"N = {self.propose_budget}.\n{instr}")
        raw = self.llm(_LLM_SYS, user, nonce=0)
        spec = extract_json(raw)
        cands, seen, rej = [], set(), []
        for p in (spec or {}).get("programs", [])[:self.propose_budget]:
            steps = []
            for st in (p.get("steps") or [])[:3]:
                op = st.get("op")
                params = (("window", int(st["window"])),) if (op in WINDOWED and "window" in st) else ()
                if op:
                    steps.append((op, params))
            scope = [c for c in (p.get("scope") or []) if c in cells]
            label = "->".join(s[0] for s in steps) if steps else "(empty)"
            if not steps or not scope:
                rej.append([label, "syntax_or_scope"]); continue
            cand = ProgramSpec(steps=tuple(steps), scope=tuple(scope))
            ok, why = validate(cand)
            if not ok:
                rej.append([label, f"gate:{why}"]); continue
            if not is_novel(cand):
                rej.append([label, "non_novel"]); continue
            if cand.chain_sha() in seen:
                rej.append([label, "duplicate"]); continue
            seen.add(cand.chain_sha())
            if len(cands) < self.eval_budget:                # 只评估前 eval_budget 条（=det 预算）
                cands.append(cand)
            else:
                rej.append([label, "over_eval_budget"])
        self.rejections = rej
        return cands, raw


# ════════════════════════════ gym：utility / arm / LOFO ════════════════════════════
def program_utility(spec: ProgramSpec, pl: Dict[str, float], frozen_loss: Dict[str, float],
                    by_uid, eval_uids: List[str]) -> dict:
    """util = mean_{u∈eval}( Δ_u )，Δ_u = frozen_loss − program_loss（>0=改善）若 cell∈scope 否则 0
    （oracle 在差分中抵消）。worst_family = 分族 mean Δ 最小者（族仅评估期审计用，决策不见族）。"""
    scope = set(spec.scope)
    per_fam: Dict[str, List[float]] = {}
    n_fire = 0
    deltas: Dict[str, float] = {}
    for u in eval_uids:
        if u not in frozen_loss:
            continue
        r = by_uid[u]
        if r["cell"] in scope and u in pl:
            d = frozen_loss[u] - pl[u]
            n_fire += 1
        else:
            d = 0.0
        deltas[u] = d
        per_fam.setdefault(r["origin"], []).append(d)
    alld = [d for v in per_fam.values() for d in v]
    util = float(np.mean(alld)) if alld else 0.0
    worst = min((float(np.mean(v)) for v in per_fam.values()), default=0.0)
    return dict(util=util, worst_family=worst, n_fire=n_fire, deltas=deltas)


def group_bootstrap_ci(deltas: Dict[str, float], by_uid, B: int = 2000,
                       seed: int = 20260707) -> Tuple[float, float]:
    """held-out utility 的组自助 CI（按 uid 重采样；winners-only 才算）。"""
    uids = list(deltas)
    if not uids:
        return (0.0, 0.0)
    arr = np.array([deltas[u] for u in uids])
    rng = np.random.default_rng(seed)
    means = [float(arr[rng.integers(0, len(arr), len(arr))].mean()) for _ in range(B)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_bootstrap_ci(dA: Dict[str, float], dB: Dict[str, float], B: int = 2000,
                        seed: int = 20260708) -> Optional[dict]:
    """**主判据**：U_A − U_B 的配对组自助 CI（同 held-out uid 逐序列配对，按 uid 重采样）。
    dA/dB = 两臂胜者的 per-uid held-out Δ（frozen−program，同 cell 条件化面）。CI 下界>0 = A 显著胜 B。"""
    common = sorted(set(dA) & set(dB))
    if not common:
        return None
    diff = np.array([dA[u] - dB[u] for u in common])
    rng = np.random.default_rng(seed)
    means = [float(diff[rng.integers(0, len(diff), len(diff))].mean()) for _ in range(B)]
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return dict(point=round(float(diff.mean()), 4), ci=[round(lo, 4), round(hi, 4)],
                n=len(common), A_beats_B=bool(lo > 0))


def run_arm(proposer, report, budget, cells, sel_uids, ho_uids,
            series, fp, fold_of, frozen_loss, by_uid, tag, memory=None) -> dict:
    t0 = time.time()
    out = proposer.propose(report, budget, cells, memory)
    cands = out[0] if isinstance(out, tuple) else out
    raw = out[1] if isinstance(out, tuple) else None
    rej = getattr(proposer, "rejections", None)                              # ceiling 臂捕获拒因
    cands = [c for c in cands if validate(c)[0] and is_novel(c)][:budget]     # 机械 Gate（再确认）
    if not cands:
        return dict(arm=proposer.name, n_candidates=0, note="无有效候选",
                    rejections=rej, seconds=round(time.time() - t0, 1))
    # executor：每候选执行一次（chain 缓存 → 跨臂/跨 fold 免重算）
    losses = [program_losses(c, series, fp, fold_of, tag) for c in cands]
    # 单一 grounded judge：select-on-selection
    sel = [program_utility(c, pl, frozen_loss, by_uid, sel_uids)["util"]
           for c, pl in zip(cands, losses)]
    best_i = int(np.argmax(sel))
    best, best_pl = cands[best_i], losses[best_i]
    ho = program_utility(best, best_pl, frozen_loss, by_uid, ho_uids)         # eval-on-heldout
    lo, hi = group_bootstrap_ci(ho["deltas"], by_uid)                         # winner-only CI vs frozen
    return dict(arm=proposer.name, n_candidates=len(cands),
                selected=dict(chain=[[s.op, dict(s.params)] for s in to_action_spec(best).steps],
                              scope=list(best.scope), sha=best.chain_sha()),
                selection_util=round(sel[best_i], 4),
                heldout_util=round(ho["util"], 4), heldout_worst=round(ho["worst_family"], 4),
                heldout_ci=[round(lo, 4), round(hi, 4)], heldout_n_fire=ho["n_fire"],
                safe=bool(ho["worst_family"] > -DELTA_SAFE),
                deltas={u: round(d, 6) for u, d in ho["deltas"].items()},     # 配对 CI 用（写盘前剥离）
                rejections=rej,
                raw_len=(len(raw) if raw else 0), seconds=round(time.time() - t0, 1))


def lofo(dev_families, arms_factory, budget: int, n_series: Optional[int], tag: str,
         mem_by_fam=None) -> dict:
    """dev-gate：在预锁 challenge 族上做 LOFO（held-out 一族，其余 dev 族做 selection）。"""
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in recs}
    # ★只物化 **dev_families** 语料 → 另 4 族 untouched 与 holdout **绝不进执行器**（评审条件）
    corpus_full = [rs for rs in build_s2_dev() if rs.origin in dev_families]
    # 语料子集（dry-run 提速：每族均匀取前 n_series//families）
    by_fam: Dict[str, list] = {}
    for rs in corpus_full:
        by_fam.setdefault(rs.origin, []).append(rs)
    if n_series:
        per = max(4, n_series // max(1, len(by_fam)))
        series = [rs for fam in by_fam.values()
                  for rs in sorted(fam, key=lambda r: r.series_uid)[:per]]
    else:
        series = corpus_full
    fp = FrozenProbe()
    fold_of = grouped_folds([rs.series_uid for rs in series])
    corpus_uids = {rs.series_uid for rs in series}
    print(f"[gym] tag={tag} 语料={len(series)} series/{len(by_fam)}族 budget={budget} "
          f"dev={list(dev_families)} [{time.time()-t0:.0f}s]", flush=True)

    pool_loss = pool_baseline(series, fp, fold_of, tag)
    fp_pick_all = frozen_picks(by_uid, [u for u in sorted(corpus_uids) if u in by_uid])
    frozen_loss = frozen_loss_of(list(fp_pick_all), fp_pick_all, pool_loss)
    print(f"[gym] pool baseline 完成；frozen_loss n={len(frozen_loss)} [{time.time()-t0:.0f}s]", flush=True)

    folds = []
    arm_deltas: Dict[str, Dict[str, float]] = {}                  # arm → {uid: Δ}（★ITT：未覆盖/空候选折 no-op 0）
    for held in dev_families:
        sel_uids = [u for u in frozen_loss if by_uid[u]["origin"] in dev_families
                    and by_uid[u]["origin"] != held]
        ho_uids = [u for u in frozen_loss if by_uid[u]["origin"] == held]
        if not sel_uids or not ho_uids:
            continue
        report = build_report(sel_uids, by_uid, pool_loss, frozen_loss)
        cells = _cells_from(report)
        # ★LOFO 无泄漏：LLM 只检索 **selection 族**（≠held）的经验；held 族专属经验禁入 prompt
        #   （每条 seeded 经验仅指涉其自身 src 族结构 → 排除 held 的条目即杜绝"给答案"）。
        sel_families = [f for f in dev_families if f != held]
        mem = [line for f in sel_families for line in (mem_by_fam or {}).get(f, [])]
        held_lines = set((mem_by_fam or {}).get(held, []))
        assert not (held_lines & set(mem)), f"held-out {held} 经验泄漏进 memory"   # 硬守卫
        arm_rows = []
        for proposer in arms_factory():
            r = run_arm(proposer, report, budget, cells, sel_uids, ho_uids,
                        series, fp, fold_of, frozen_loss, by_uid, tag,
                        memory=(mem or None) if proposer.name.startswith("llm") else None)
            arm_rows.append(r)
            print(f"  [{held:14s} | {r['arm']:10s}] cands={r.get('n_candidates')} "
                  f"sel={r.get('selection_util')} ho={r.get('heldout_util')} "
                  f"worst={r.get('heldout_worst')} ci={r.get('heldout_ci')} "
                  f"fire={r.get('heldout_n_fire')} [{r.get('seconds')}s]", flush=True)
        folds.append(dict(held_out=held, n_sel=len(sel_uids), n_ho=len(ho_uids), arms=arm_rows))
        for a in arm_rows:                                       # ★ITT：每臂对每 held-out uid 填 Δ（空候选→0=no-op）
            dmap = a.get("deltas", {})
            arm_deltas.setdefault(a["arm"], {})
            for u in ho_uids:
                arm_deltas[a["arm"]][u] = dmap.get(u, 0.0)

    # 跨 fold 汇总（每臂 held-out util 均值 + worst-family 最差）
    summary: Dict[str, dict] = {}
    for f in folds:
        for a in f["arms"]:
            s = summary.setdefault(a["arm"], dict(ho=[], worst=[], fires=[]))
            if "heldout_util" in a:
                s["ho"].append(a["heldout_util"])
                s["worst"].append(a["heldout_worst"])
                s["fires"].append(a["heldout_n_fire"])
    agg = {arm: dict(mean_heldout_util=round(float(np.mean(v["ho"])), 4) if v["ho"] else None,
                     min_worst_family=round(float(np.min(v["worst"])), 4) if v["worst"] else None,
                     total_fire=int(np.sum(v["fires"])) if v["fires"] else 0)
           for arm, v in summary.items()}
    # ★主判据：配对 U_A − U_B **ITT** 组自助 CI（关键 = LLM − det_search；空候选折已 no-op 计入 → n=full）
    PAIRS = [("llm_mem", "det_search"), ("llm_ceiling", "det_search"), ("llm_nomem", "det_search"),
             ("llm_mem", "random"), ("llm_ceiling", "random"), ("det_search", "random")]
    paired = {}
    for A, B in PAIRS:
        if A in arm_deltas and B in arm_deltas:
            r = paired_bootstrap_ci(arm_deltas[A], arm_deltas[B])
            if r is not None:
                paired[f"{A}_minus_{B}"] = r
    for f in folds:                                              # 写盘前剥离逐 uid Δ（保持 JSON 紧凑）
        for a in f["arms"]:
            a.pop("deltas", None)
    payload = dict(mode=tag, budget=budget, n_series=len(series),
                   dev_gate_families=list(dev_families),
                   scope_note="预锁 dev-gate（非全 8 族 LOFO）；另族 untouched，仅 dev-gate 明确成功后确认",
                   folds=folds, summary=agg, paired_vs=paired, seconds=round(time.time() - t0, 1))
    (OUT / f"gym_{tag}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), "utf-8")
    print(f"\n[gym] 汇总（dev-gate {list(dev_families)}，非全 8 族 LOFO）：", flush=True)
    for arm, a in agg.items():
        print(f"  {arm:10s} mean_heldout_util={a['mean_heldout_util']} "
              f"min_worst_family={a['min_worst_family']} total_fire={a['total_fire']}", flush=True)
    print(f"[gym] 主判据 配对 CI：", flush=True)
    for k, r in paired.items():
        print(f"  {k:26s} point={r['point']:+.4f} CI={r['ci']} A_beats_B={r['A_beats_B']} (n={r['n']})",
              flush=True)
    print(f"产物：{OUT / f'gym_{tag}.json'}  [{time.time()-t0:.0f}s]", flush=True)
    return payload


# ════════════════════════════ 零外呼审计（评审第 40 轮）════════════════════════════
def _menu_reverse() -> Dict[tuple, str]:
    """resolved (op,window) 签名 → menu action_id（拒因归因"撞已有 menu"用）。"""
    from .policy.program_edit import resolved_sig
    m = {}
    for aid, spec in action_menu_v1().actions.items():
        m[tuple((s.op, s.params.get("window")) for s in spec.steps)] = aid
    return m


def _classify_llm_raw(raw: Optional[str], cells: List[str]) -> dict:
    """离线解析一折 mini 原始 JSON → 每候选拒因（syntax/scope/non_novel撞menu/duplicate/ACCEPTED）。
    **关键**：区分"语法非法"与"撞已有 menu 被 novelty 过滤"（后者是信息不对称，非模型能力缺陷）。"""
    from collections import Counter
    from .llm.client import extract_json
    from .policy.program_edit import resolved_sig
    menu_rev = _menu_reverse()
    spec = extract_json(raw or "")
    progs = (spec or {}).get("programs", []) if isinstance(spec, dict) else []
    seen, rows = set(), []
    for p in progs:
        steps = []
        for st in (p.get("steps") or [])[:3]:
            op = st.get("op")
            params = (("window", int(st["window"])),) if (op in WINDOWED and "window" in st) else ()
            if op:
                steps.append((op, params))
        scope = [c for c in (p.get("scope") or []) if c in cells]
        label = "->".join(s[0] for s in steps) if steps else "(empty)"
        if not steps:
            rows.append([label, "syntax:no_steps"]); continue
        if not scope:
            rows.append([label, "scope:empty_or_unknown_cell"]); continue
        cand = ProgramSpec(steps=tuple(steps), scope=tuple(scope))
        ok, why = validate(cand)
        if not ok:
            rows.append([label, f"syntax_gate:{why}"]); continue
        rsig = resolved_sig(cand)
        if rsig in menu_rev:
            rows.append([label, f"non_novel:={menu_rev[rsig]}"]); continue
        if cand.chain_sha() in seen:
            rows.append([label, "duplicate"]); continue
        seen.add(cand.chain_sha())
        rows.append([label, "ACCEPTED"])
    bucket = Counter(("ACCEPTED" if r == "ACCEPTED" else r.split(":")[0]) for _, r in rows)
    return dict(n_raw=len(progs), n_accepted=int(bucket.get("ACCEPTED", 0)),
                reason_counts=dict(bucket), rows=rows)


def _winner_deltas_from_cache(selected: Optional[dict], by_uid, frozen_loss,
                              ho_uids: List[str], tag: str) -> Dict[str, float]:
    """从 executor 落盘缓存重建胜者的 held-out per-uid Δ（selected=None → no-op 全 0，ITT 用）。"""
    if selected is None:
        return {u: 0.0 for u in ho_uids if u in frozen_loss}
    path = _exec_dir(tag) / f"prog__{selected['sha']}.json"
    losses = json.loads(path.read_text("utf-8")) if path.exists() else {}
    scope = set(selected["scope"])
    d = {}
    for u in ho_uids:
        if u not in frozen_loss:
            continue
        cell = by_uid[u]["cell"]
        d[u] = (frozen_loss[u] - losses[u]) if (cell in scope and u in losses) else 0.0
    return d


def audit(tag: str = "formal"):
    """★零外呼审计（评审第 40 轮）：不重跑 mini/不覆盖注册结果。产 gym_formal_audit.json：
    ①每折候选拒因分解（syntax/scope/non_novel/dup）；②四折 intention-to-treat LLM−det CI（空候选折=no-op）；
    ③四折族级 LLM−det 差（唯 4 个独立 proposer 决策，非 336）；④安全门（无臂过 δ_safe）。"""
    t0 = time.time()
    formal_json = json.loads((OUT / f"gym_{tag}.json").read_text("utf-8"))
    dev_families = formal_json["dev_gate_families"]
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in recs}
    corpus = [rs for rs in build_s2_dev() if rs.origin in dev_families]
    fp = FrozenProbe()
    fold_of = grouped_folds([rs.series_uid for rs in corpus])
    pool_loss = pool_baseline(corpus, fp, fold_of, tag, verbose=False)          # cache 命中 → 秒级
    corpus_uids = sorted({rs.series_uid for rs in corpus} & set(by_uid))
    fp_pick = frozen_picks(by_uid, corpus_uids)
    frozen_loss = frozen_loss_of(corpus_uids, fp_pick, pool_loss)
    # LLM 缓存（离线读）
    llm_cache = json.loads((Path(__file__).resolve().parent / "llm" / "_cache"
                            / "proposer_mini.json").read_text("utf-8"))

    from collections import Counter
    per_fold, llm_itt, det_deltas, fam_diff = [], {}, {}, []
    for fold in formal_json["folds"]:
        held = fold["held_out"]
        sel_uids = [u for u in frozen_loss if by_uid[u]["origin"] in dev_families
                    and by_uid[u]["origin"] != held]
        ho_uids = [u for u in frozen_loss if by_uid[u]["origin"] == held]
        report = build_report(sel_uids, by_uid, pool_loss, frozen_loss)
        cells = _cells_from(report)
        arms = {a["arm"]: a for a in fold["arms"]}
        llm_arm = arms.get("llm_ceiling") or arms.get("llm_mem") or {}
        stored_rej = llm_arm.get("rejections")
        if stored_rej is not None:                              # ceiling：拒因已直接落盘 → 免重建 prompt
            bucket = Counter(r[1].split(":")[0] for r in stored_rej)
            n_acc = llm_arm.get("n_candidates", 0)
            cls = dict(n_raw=n_acc + len(stored_rej), n_accepted=n_acc,
                       reason_counts=dict(bucket), rows=stored_rej, cache_hit=True)
        else:                                                   # formal(mini)：离线重建 prompt 命中 LLM 缓存
            sel_families = [f for f in dev_families if f != held]
            mem = [line for f in sel_families for line in SEEDED_MEM.get(f, [])]
            mem_txt = ("\n\nPast VERIFIED-successful programs (retrieved experience, chain → where it "
                       "helped):\n" + "\n".join(f"  {m}" for m in mem[:6])) if mem else ""
            user = (f"{report_text(report)}{mem_txt}\n\nObservable cells available: {cells}\n"
                    f"N = {10}.\n{_LLM_INSTR}")
            key = hashlib.sha1(f"gpt-5.4-mini|0.0|0|{_LLM_SYS}|{user}".encode("utf-8")).hexdigest()
            raw = llm_cache.get(key)
            cls = _classify_llm_raw(raw, cells)
            cls["cache_hit"] = raw is not None
        cls["held_out"] = held
        per_fold.append(cls)
        # ITT deltas（LLM 空候选 → no-op；det 用注册胜者）
        llm_itt.update(_winner_deltas_from_cache(llm_arm.get("selected"), by_uid, frozen_loss, ho_uids, tag))
        det_deltas.update(_winner_deltas_from_cache(arms.get("det_search", {}).get("selected"),
                                                    by_uid, frozen_loss, ho_uids, tag))
        llm_ho = llm_arm.get("heldout_util")
        det_ho = arms.get("det_search", {}).get("heldout_util")
        fam_diff.append(dict(held=held, llm_ho=llm_ho, det_ho=det_ho,
                             llm_minus_det=(round((llm_ho if llm_ho is not None else 0.0) - det_ho, 4))))

    itt = paired_bootstrap_ci(llm_itt, det_deltas)                              # n=336（含 no-op 折）
    safety = {arm: v.get("min_worst_family") for arm, v in formal_json["summary"].items()}
    diffs = [d["llm_minus_det"] for d in fam_diff]
    payload = dict(
        audit_of=f"gym_{tag}.json", note="零外呼审计；不覆盖注册结果",
        per_fold_rejections=per_fold,
        itt_llm_minus_det=dict(**(itt or {}), n_uids=len(set(llm_itt) & set(det_deltas)),
                               interpretation="ITT：空候选折按 no-op(=frozen) 计入 → 完整四折协议 CI"),
        family_level_llm_minus_det=dict(
            per_fold=fam_diff, n_independent_decisions=len(fam_diff),
            llm_wins=int(sum(1 for d in diffs if d > 0)), mean=round(float(np.mean(diffs)), 4),
            caveat="仅 4 个独立 proposer 决策；UID bootstrap 衡量固定程序下的序列不确定性，非跨 family 泛化"),
        safety_gate=dict(delta_safe=DELTA_SAFE, arm_min_worst_family=safety,
                         any_arm_passes=any((v or -1.0) > -DELTA_SAFE for v in safety.values()),
                         note="det 相对赢但 S_regime worst −0.099 触发 unsafe；无臂过完整安全门"),
        seconds=round(time.time() - t0, 1))
    (OUT / f"gym_{tag}_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), "utf-8")
    # 打印
    print(f"\n══ B1b-{tag} 零外呼审计 [{time.time()-t0:.0f}s] ══", flush=True)
    print("── 每折候选拒因（raw→accepted，为何被拒）──", flush=True)
    for pf in per_fold:
        print(f"  [{pf['held_out']:14s}] raw={pf['n_raw']:2d} → accepted={pf['n_accepted']} | "
              f"{pf['reason_counts']}  (cache_hit={pf['cache_hit']})", flush=True)
    print(f"── ★主判据 ITT（完整四折，空折=no-op）LLM−det ──", flush=True)
    if itt:
        print(f"  point={itt['point']:+.4f} CI={itt['ci']} A_beats_B={itt['A_beats_B']} n={itt['n']}", flush=True)
    print(f"── 族级 LLM−det（4 个独立决策，非 {itt['n'] if itt else '?'} uid）──", flush=True)
    for d in fam_diff:
        print(f"  {d['held']:14s} llm_ho={d['llm_ho']} det_ho={d['det_ho']} diff={d['llm_minus_det']:+.4f}",
              flush=True)
    print(f"  → LLM 胜 {payload['family_level_llm_minus_det']['llm_wins']}/4 折，"
          f"mean={payload['family_level_llm_minus_det']['mean']:+.4f}", flush=True)
    print(f"── 安全门（δ_safe={DELTA_SAFE}）──", flush=True)
    print(f"  worst-family: {safety} → 任何臂过安全门={payload['safety_gate']['any_arm_passes']}", flush=True)
    print(f"产物：{OUT / f'gym_{tag}_audit.json'}", flush=True)
    return payload


# ════════════════════════════ 模式 ════════════════════════════
def dry_run(budget: int = 3, n_series: int = 120):
    """random+det 两臂，2 dev 族，子集语料，**无 LLM**——验证 executor→report→proposer→gym 全管线。"""
    def arms():
        return [RandomProgramProposer(seed=20260707), DetSearchProgramProposer()]
    return lofo(("S_both", "S_trend"), arms, budget, n_series, tag="dry")


# 每条经验**仅指涉其自身 src 族结构**（不跨族命名）→ lofo 排除 held 条目即杜绝答案泄漏（硬守卫）。
SEEDED_MEM = {
    "S_both": ["impute_linear→winsorize→denoise_savgol helped high-noise cells (shape-preserving)"],
    "S_regime": ["impute_linear→winsorize helped heavy-outlier cells"],
    "S_trend": ["impute_linear→denoise_stl helped trend cells (removes trend before judge)"],
    "S_multiseason": ["AVOID heavy median window≈period on seasonal cells (wipes seasonality)"],
}


def formal(budget: int = 10):
    """三臂**预锁 4 族 dev-gate**（非全 8 族 LOFO；另 4 族 untouched，仅 dev-gate 成功后确认）。
    **待 prereg 审后开**；含 gpt-5.4-mini 外呼（每折 1 次，共 4 次）。memory 每折仅取 selection 族（无泄漏）。"""
    from .run_proposer import make_mini_client
    mini = make_mini_client()

    def arms():
        return [RandomProgramProposer(seed=20260707), DetSearchProgramProposer(),
                LLMProgramProposer(mini, with_memory=True)]
    return lofo(CHALLENGE, arms, budget, None, tag="formal", mem_by_fam=SEEDED_MEM)


def ceiling(model: str, propose_budget: int = 20, eval_budget: int = 10):
    """★B1b-ceiling 强模型公平性修复实验（评审第 40 轮；**须单独注册 + 用户 go**；含强模型外呼）。
    修复=**明示 menu 链**（消信息不对称）+ 提议至多 propose_budget → gate 后评估前 eval_budget(=det)；
    空候选 no-op 计入 ITT。仍 4 族/4 次调用/不改 memory/judge/评分。保存原始输出 + 每条拒因。
    model: 'pro'(deepseek-v4-pro) | 'agicto:<name>'（经 agicto 端点，如 agicto:gpt-5.4 / agicto:claude-opus-4-8）。"""
    from .llm.client import LLMClient, get_client
    if model.startswith("agicto:"):
        from .run_proposer import AGICTO_URL, _agicto_key
        name = model.split(":", 1)[1]
        temp = None if ("claude" in name or "opus" in name) else 0.0   # claude-opus 拒 temperature
        strong = LLMClient(model=name, temperature=temp, cache_name=f"ceiling_{name.replace('/','_')}",
                           url=AGICTO_URL, key=_agicto_key())
    else:
        strong = get_client(model, temperature=0.0, cache_name=f"ceiling_{model}")

    def arms():
        return [RandomProgramProposer(seed=20260707), DetSearchProgramProposer(),
                LLMCeilingProposer(strong, propose_budget, eval_budget)]
    return lofo(CHALLENGE, arms, eval_budget, None, tag="ceiling", mem_by_fam=SEEDED_MEM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--model", default=None, help="ceiling 强模型：pro | agicto:<name>")
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--n-series", type=int, default=120)
    args = ap.parse_args()
    if args.dry_run:
        dry_run(args.budget, args.n_series)
    elif args.formal:
        formal(args.budget)
    elif args.audit:
        audit(args.model or "formal")
    elif args.ceiling:
        assert args.model, "--ceiling 须指定 --model（pro | agicto:<name>）"
        ceiling(args.model)
        audit("ceiling")                                         # 跑完自动出 ITT+拒因审计
    else:
        print("用 --dry-run / --formal / --audit / --ceiling --model <m>")


if __name__ == "__main__":
    main()
