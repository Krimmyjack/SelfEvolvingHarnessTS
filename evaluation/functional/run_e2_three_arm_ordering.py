"""E2 —— 三臂 STATIC / A3 / A5：Domain Ordering Card 在**真实在线循环**里连续生效。

E1 只证明「一张卡能改变一轮的探测顺序」。E2 问的是工程问题：
**这张卡能不能在一条 prequential 流上连续被重建、连续生效，并且三臂之间唯一的
差别就是 Source Episode 是否用于初始化排序。**

================================ 冻结协议（跑目标流之前写定） ================================

臂（唯一变量 = Source 是否初始化排序）
  STATIC  Source 先验建卡一次，**永不重建**
  A3      Source 为空；只用 Target Episode 重建
  A5      Source 先验初始化，之后采用与 A3 **完全相同**的 Target 更新规则

三臂完全相同的东西（评审冻结条件）
  · 同候选供给：每轮 Fast 供应的候选集合必须逐格相同——运行时**硬断言**
  · 同 lambda：在 Source 上预先固定，三臂同值
  · 同 Support 预算：budget = 3（= fast_propose_v1.maxItems）
  · 同 Memory：三臂都以**空 Episode 列表**构造 TTHAMethod。理由——
    Reference 1/2/3 降级是**第二条学习通道**，会与卡的效果混淆。
    本实验里卡是唯一的学习通道。

算子集合（机械约束，非实验者选择）
  ["outlier_mad", "winsorize", "outlier_iqr"] —— multiop 冻结顺序的前三个。
  第四个 hampel_filter 被 `methods/ttha/schemas/fast_propose_v1.json` 的
  `candidates.maxItems = 3` **机械挡在供给层之外**：一次 propose 最多 3 个候选，
  第 4 个候选会让载荷 schema 校验失败（AgentProtocolError: payload.candidates
  has too many items）。这条约束早于 E2 存在，取前三个是与 E2 结果无关的规则。
  denoise 家族同样被机械排除（targeting_mode=global → modified_fraction > 0.35）。

lambda 的固定方式（只用 Source，不看任何 Target 结果）
  在 Source bank 内部做一次 dress rehearsal：series 前一半建先验、后一半当
  prequential 流，重演同一组 frozen-vs-learning 对比。
  lambda = argmax gain/probe  s.t.  harm(learning) <= harm(frozen)。
  （无约束最大 gain/probe 会选出 lambda=0，即零风险厌恶——这正是上一版
  learning 臂 harm 上升的原因。）

Source / Target 计数分开存（评审第 7 条）
  Source 只以 SOURCE_PSEUDO_COUNT=10 条**伪观测**入场（保留均值、丢弃样本量），
  Target 以真实计数累加。两块原始计数分别写进卡的
  `risk_guards.evidence_blocks.{source,target}`，永不合池。

prequential 纪律
  流按 origin 分批。整批期间 Memory/卡冻结；该批所有 Episode 在**批结束后**
  才写进证据并重建卡（revision += 1）。

数据
  Source = GRID0_dev pool，Target = PATTERN_conf pool，序列零重叠（硬断言）。
  两个 pool 都已 development exposed；**零 virgin series**。
  Source 与 Target 是**同一数据集的不同序列** → 域内 Source 先验，不是跨域迁移。

测量（不进入任何决策）
  累计 gain / probe / harm；以及
  N_first_effective = 该臂**首次出现有效 Target-local 行为**时已消耗的 Support 反馈数。
    「有效」定义（跑之前冻结）：该轮 (a) 本臂当前卡的 order 已不同于本臂
    revision-1 的 order，且 (b) 在**同一格**上，用 revision-1 的 order 反事实
    重放，first-positive 所需 probe 数严格更多（或 revision-1 找不到正向而
    当前 order 找得到）。反事实重放用 multiop 冻结效用表——**只用于测量**，
    不进入任何一臂的策略。STATIC 按构造恒为 null。

零 LLM。只追加不改写。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT.parent, ROOT, ROOT / "evaluation" / "functional",
           ROOT / "methods" / "ttha"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_grid0_census as gc  # noqa: E402
import run_grid0_utility as gu  # noqa: E402
import run_v1_guidance_evolution as runner  # noqa: E402
import run_v1_kdd2018_natural_slow_update as nsu  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha import ordering_card as oc  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    TTHAAgentCore)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore)
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    run_online_round)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway, extract_public_features)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    ScopeExecutor)
from SelfEvolvingHarnessTS.methods.ttha.signed_radius import (  # noqa: E402
    MATERIAL_THRESHOLD)

E2_DIR = ROOT / "artifacts" / "functional" / "e2"
MULTIOP = E2_DIR / "multiop_checkpoint.json"
PATTERN = E2_DIR / "pattern_checkpoint.json"
OUT = E2_DIR / "e2_three_arm_checkpoint.json"

OPS: tuple[str, ...] = ("outlier_mad", "winsorize", "outlier_iqr")
ARMS: tuple[str, ...] = ("STATIC", "A3", "A5")
ORIGINS: tuple[int, ...] = (600, 672, 744, 816, 888, 960)
SOURCE_PSEUDO_COUNT = 10
LAM_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
BUDGET = 3
CARD_ID = "ordering_outlier_forecast"
CONSUMER = "ridge"
FAMILY = "outlier"
M = MATERIAL_THRESHOLD


# ---------------------------------------------------------------- 冻结效用表

def _load_tables() -> tuple[dict, dict]:
    """(source_rows, target_rows) 按 cohort 分组。零重叠硬断言。"""
    doc = json.loads(MULTIOP.read_text(encoding="utf-8"))
    rows = doc["rows"]
    src = {c: [r for r in rows if r["cohort"] == c and r["pool"] == "GRID0_dev"]
           for c in "AB"}
    tgt = {c: [r for r in rows if r["cohort"] == c
               and r["pool"] == "PATTERN_conf"] for c in "AB"}
    for c in "AB":
        overlap = {r["series"] for r in src[c]} & {r["series"] for r in tgt[c]}
        assert not overlap, f"SOURCE/TARGET SERIES OVERLAP in cohort {c}: {overlap}"
    return src, tgt


def _gains(row: Mapping[str, Any]) -> dict[str, float | None]:
    """None = 该格该算子在冻结表里被 verifier 拒绝/仪器失败（不是 0 收益）。"""
    return {op: (None if row.get(op) is None else float(row[op])) for op in OPS}


# --------------------------------------------------- lambda：只在 Source 上固定

def _probe_table(order: Sequence[str], g: Mapping[str, float | None]
                 ) -> tuple[int, float, int, list[tuple[str, float]]]:
    """表上重放 stop-on-first-positive。None（被拒）不消耗 Support 预算。"""
    probes = 0
    seen: list[tuple[str, float]] = []
    for op in order:
        val = g.get(op)
        if val is None:
            continue
        probes += 1
        seen.append((op, val))
        if val >= M:
            return probes, val, sum(1 for _, v in seen if v < -M), seen
    return probes, 0.0, sum(1 for _, v in seen if v < -M), seen


def _evidence_from_rows(rows: Sequence[Mapping[str, Any]]
                        ) -> dict[str, dict[str, float]]:
    """全信息 Source 证据：每格每算子一条观测（被拒的记 legal_opportunity）。"""
    ev = oc.empty_evidence(OPS)
    for row in rows:
        g = _gains(row)
        oc.accumulate(ev, [(op, g[op]) for op in OPS])
    return ev


def _source_dress_rehearsal(rows: Sequence[Mapping[str, Any]], lam: float
                            ) -> tuple[float, float, float]:
    """Source bank 内部的 frozen-vs-learning 对比（不看任何 Target 数据）。"""
    series = sorted({r["series"] for r in rows})
    half = len(series) // 2
    prior_set, stream_set = set(series[:half]), set(series[half:])
    prior_rows = [r for r in rows if r["series"] in prior_set]
    stream = [r for r in rows if r["series"] in stream_set]
    prior_ev = _evidence_from_rows(prior_rows)

    frozen_ev = oc.merge_evidence(prior_ev, oc.empty_evidence(OPS),
                                  source_pseudo_count=SOURCE_PSEUDO_COUNT)
    frozen_order = oc.rank_operators(frozen_ev, lam=lam, tie_break=OPS)
    learn_tgt = oc.empty_evidence(OPS)

    st = {"g": 0.0, "p": 0, "h": 0}
    fr = {"g": 0.0, "p": 0, "h": 0}
    for origin in ORIGINS:
        batch = [r for r in stream if r["origin"] == origin]
        pending: list[list[tuple[str, float]]] = []
        merged = oc.merge_evidence(prior_ev, learn_tgt,
                                   source_pseudo_count=SOURCE_PSEUDO_COUNT)
        learn_order = oc.rank_operators(merged, lam=lam, tie_break=OPS)
        for row in batch:
            g = _gains(row)
            p, gain, h, seen = _probe_table(learn_order, g)
            st["g"] += gain; st["p"] += p; st["h"] += h
            pending.append(seen)
            p, gain, h, _ = _probe_table(frozen_order, g)
            fr["g"] += gain; fr["p"] += p; fr["h"] += h
        for seen in pending:
            oc.accumulate(learn_tgt, seen)
    n = max(len(stream), 1)
    return st["g"] / max(st["p"], 1), st["h"] / n, fr["h"] / n


def _fit_lambda(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    feasible, fallback = [], []
    sweep = []
    for lam in LAM_GRID:
        gpp, h_learn, h_frozen = _source_dress_rehearsal(rows, lam)
        sweep.append({"lambda": lam, "gain_per_probe": gpp,
                      "harm_learning": h_learn, "harm_frozen": h_frozen,
                      "feasible": bool(h_learn <= h_frozen + 1e-12)})
        rec = (gpp, -(h_learn - h_frozen), -lam, lam, h_learn, h_frozen)
        (feasible if h_learn <= h_frozen + 1e-12 else fallback).append(rec)
    if feasible:
        best = max(feasible, key=lambda t: (round(t[0], 6), t[1], t[2]))
        crit = "harm(learning) <= harm(frozen)"
    else:
        best = max(fallback, key=lambda t: (t[1], round(t[0], 6), t[2]))
        crit = "NO feasible lambda -> min harm excess"
    return {"lambda": best[3], "criterion": crit, "sweep": sweep}


# ------------------------------------------------------------------ 卡的安装

def _install(store: SnapshotStore, parent_snapshot: Any,
             card_doc: Mapping[str, Any] | None, *, edit_id: str) -> Any:
    """在 fork 上写/删卡并编译；**不修改 h0**。"""
    parent = store.materialize(parent_snapshot)
    fork = store.fork(parent, edit_id=edit_id)
    learned = fork / "skills" / "learned"
    learned.mkdir(parents=True, exist_ok=True)
    target = learned / f"{CARD_ID}.json"
    if card_doc is None:
        target.unlink(missing_ok=True)
    else:
        (learned / ".gitkeep").unlink(missing_ok=True)
        target.write_text(json.dumps(card_doc, ensure_ascii=False, indent=1,
                                     sort_keys=True), encoding="utf-8")
    snap = compile_snapshot(fork, verify_lock=False)
    store.materialize(snap, parent_sha=parent_snapshot.runtime_bundle_sha)
    return snap


def _card_for(scope: Mapping[str, str], source_ev, target_ev, *,
              lam: float, revision: int) -> dict[str, Any]:
    merged = oc.merge_evidence(source_ev, target_ev,
                               source_pseudo_count=SOURCE_PSEUDO_COUNT)
    return oc.build_ordering_card(
        skill_id=CARD_ID, scope=scope, evidence=merged, lam=lam,
        tie_break=OPS, revision=revision,
        evidence_blocks={"source": source_ev or oc.empty_evidence(OPS),
                         "target": target_ev})


# -------------------------------------------------------------------- 一轮在线

def _one_round(snapshot: Any, executor: Any, series0: np.ndarray,
               values: Mapping[str, Any], *, origin: int, dataset: str,
               round_name: str) -> Any:
    backend = sealed.SealedProbeBackend(
        explore=True, operators=OPS, max_propose_candidates=BUDGET,
        force_pool=True)
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(series0[:origin], task_kind="forecast"))
    # 空 Episode 列表——卡是唯一学习通道（见 docstring）
    method = TTHAMethod(TTHAFastAgent(core), snapshot, ())
    request = runner._a5_request(series0, values, origin, dataset)
    feats = dict(extract_public_features(series0[:origin], task_kind="forecast"))
    return run_online_round(
        method, executor, request, values, origin=origin,
        slow_agent=None, controller=None, store=None,
        card_builder=runner._a5v2_card, round_name=round_name,
        budget=BUDGET, allow_slow=False, domain=dataset, period=24,
        fast_features=feats, ordering_program_family=FAMILY)


def _ops_of(result: Any, candidate_ids: Sequence[str]) -> list[str]:
    steps = dict(result._method.last_trace.candidate_program_steps or {})
    return [oc._leading_op(steps.get(c)) for c in candidate_ids]


# ------------------------------------------------------------------------ 主流程

def main() -> int:
    t0 = time.perf_counter()
    src_tbl, tgt_tbl = _load_tables()

    census = json.loads(gc.CHECKPOINT_REL.read_text(encoding="utf-8"))
    pattern = json.loads(PATTERN.read_text(encoding="utf-8"))
    kdd = {s["entity_id"]: s["raw"] for s in gc._load_kdd_series()}
    reg, _ = gc._load_registry_series()
    ds_b = census["census"]["cohort_B"]["dataset_selected"]
    regb = {s["entity_id"]: s for s in reg if s["dataset_id"] == ds_b}

    conf_cells = {c: [o for o in pattern["observations_conf"] if o["cohort"] == c]
                  for c in "AB"}

    declared = {
        "protocol": "E2_THREE_ARM_ONLINE_ORDERING_CARD",
        "frozen_before_target_stream": True,
        "arms": list(ARMS),
        "only_variable": "whether Source Episodes initialise the ordering",
        "identical_across_arms": [
            "candidate supply (asserted per cell)", "lambda",
            "Target Support budget", "empty episodic Memory "
            "(card is the sole learning channel)"],
        "operator_set": list(OPS),
        "operator_set_rule": "first three of the frozen multiop declared order; "
                             "the 4th (hampel_filter) is excluded MECHANICALLY by "
                             "fast_propose_v1.json candidates.maxItems=3 "
                             "(a 4th candidate fails payload schema validation). "
                             "denoise family excluded mechanically as before.",
        "budget_per_round": BUDGET,
        "source_pseudo_count": SOURCE_PSEUDO_COUNT,
        "lambda_fit": "Source bank dress rehearsal only; harm-constrained; "
                      "frozen before the Target stream; identical in all arms",
        "counts": "Source and Target evidence stored SEPARATELY in "
                  "risk_guards.evidence_blocks; never pooled",
        "prequential": "origin batches; card frozen for the whole batch; "
                       "episodes folded in only after the batch completes",
        "pools": {"source": "GRID0_dev", "target": "PATTERN_conf",
                  "series_overlap": 0, "virgin_series_consumed": 0},
        "scope_note": "Source and Target are different series of the SAME dataset "
                      "-> within-domain source prior, not cross-domain transfer",
        "N_first_effective": "cumulative Target Support receipts consumed up to and "
                             "including the first round where (a) the arm's order "
                             "differs from its revision-1 order and (b) replaying the "
                             "same cell under the revision-1 order costs strictly more "
                             "probes to first positive (or finds none). Counterfactual "
                             "replay uses the frozen multiop table for MEASUREMENT "
                             "ONLY -- it never enters any arm's policy.",
        "llm_calls": 0,
    }

    lam_fit = {c: _fit_lambda(src_tbl[c]) for c in "AB"}
    declared["lambda"] = {c: lam_fit[c]["lambda"] for c in "AB"}
    for c in "AB":
        print(f"[lambda] cohort {c}: lambda={lam_fit[c]['lambda']}  "
              f"[{lam_fit[c]['criterion']}]  -> FROZEN", flush=True)

    h0 = runner._h0_snapshot()
    out: dict[str, Any] = {"declared_before_target_stream": declared,
                           "lambda_fit": lam_fit, "cohorts": {}}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                   encoding="utf-8")   # 协议先落盘，再跑目标流

    # snapshot store 放在 scratchpad：卡的每个 revision 已完整记录在
    # checkpoint 的 arms[*].revisions 里，编译产物无需入库。
    import os as _os, tempfile as _tf  # noqa: PLC0415
    _sd = _os.environ.get("E2_STORE_DIR") or _tf.mkdtemp(prefix="e2_store_")
    store = SnapshotStore(Path(_sd))

    for cohort in "AB":
        lam = float(lam_fit[cohort]["lambda"])
        dataset = "kdd2018" if cohort == "A" else ds_b
        scope = {"task": "forecast", "domain": dataset,
                 "downstream_model_class": CONSUMER, "program_family": FAMILY}
        source_ev = _evidence_from_rows(src_tbl[cohort])
        table = {(r["series"], r["origin"]): _gains(r) for r in tgt_tbl[cohort]}

        streamed: list[tuple[str, int]] = []
        state: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            src = source_ev if arm in ("STATIC", "A5") else None
            card = _card_for(scope, src, oc.empty_evidence(OPS), lam=lam,
                             revision=1)
            snap = _install(store, h0, card, edit_id=f"{cohort}_{arm}_rev1")
            state[arm] = {
                "source_ev": src, "target_ev": oc.empty_evidence(OPS),
                "snapshot": snap, "revision": 1, "order_0": list(
                    card["risk_guards"]["order"]),
                "cum": {"gain": 0.0, "probes": 0, "harm_count": 0,
                        "harm_magnitude": 0.0, "positive_rounds": 0},
                "history": [], "revisions": [
                    {"revision": 1, "order": list(card["risk_guards"]["order"])}],
                "n_first_effective": None, "first_effective_round": None}

        cells_done = 0
        supply_checks = 0
        instrument_mismatch: list[dict[str, Any]] = []
        for origin in ORIGINS:
            batch = [c for c in conf_cells[cohort] if int(c["origin"]) == origin]
            pending = {a: [] for a in ARMS}
            bstat = {a: {"gain": 0.0, "probes": 0, "harm_count": 0,
                         "harm_magnitude": 0.0, "positive_rounds": 0}
                     for a in ARMS}
            for cell in batch:
                uid = cell["series"]
                raw = (kdd[uid] if cell["dataset"] == "kdd2018"
                       else regb[uid]["raw"])
                series0 = np.asarray(raw, dtype=np.float64)
                values = {uid: raw}
                roster = [{"series_uid": uid, "role": "train"},
                          {"series_uid": uid, "role": "eval"}]
                executor = ScopeExecutor(
                    roster, values, gu._config(cell["dataset"], origin),
                    evaluate_fn=nsu._evaluate_kdd)

                supplied_ref: list[str] | None = None
                for arm in ARMS:
                    res = _one_round(
                        state[arm]["snapshot"], executor, series0, values,
                        origin=origin, dataset=cell["dataset"],
                        round_name=f"e2_{cohort}_{arm}_{uid}_{origin}")
                    supplied = sorted(res.probe_order_before_card)
                    if supplied_ref is None:
                        supplied_ref = supplied
                    else:
                        assert supplied == supplied_ref, (
                            "CANDIDATE SUPPLY DIFFERS ACROSS ARMS at "
                            f"{uid}@{origin}: {supplied} vs {supplied_ref}")
                        supply_checks += 1
                    assert res.ordering_card_id == CARD_ID, (
                        f"card not applied at {uid}@{origin} arm={arm}")

                    steps_map = dict(
                        res._method.last_trace.candidate_program_steps or {})
                    op_of = {c: oc._leading_op(steps_map.get(c))
                             for c in res.probe_order_after_card}
                    probed = [p for p in res.actual_probed_programs
                              if p["kind"] == "probe"]
                    # winner = stop-on-first-positive 的最后一条 probe
                    gain = (float(probed[-1]["gain"])
                            if (probed and res.winner_program is not None)
                            else 0.0)
                    bstat[arm]["gain"] += gain
                    bstat[arm]["probes"] += res.target_support_receipts_used
                    bstat[arm]["harm_count"] += res.harm_count
                    bstat[arm]["harm_magnitude"] += res.harm_magnitude
                    bstat[arm]["positive_rounds"] += int(
                        res.winner_program is not None)
                    # 证据按 candidate_id 配对（verifier_rejected 条目会让
                    # 位置对齐错位——不能 zip 顺序表与结果表）；
                    # 供应但未探测的算子记 (op, None) = UNKNOWN 机会。
                    obs: list[tuple[str, float | None]] = []
                    probed_ids = set()
                    for p in probed:
                        op = op_of.get(p["candidate_id"])
                        if op:
                            obs.append((op, float(p["gain"])))
                            probed_ids.add(p["candidate_id"])
                    for cand in res.probe_order_after_card:
                        if cand not in probed_ids and op_of.get(cand):
                            obs.append((op_of[cand], None))
                    pending[arm].append(obs)

                    # ---- 仪器一致性 + N_first_effective（只测量，不入策略）----
                    g = table.get((uid, origin))
                    if g is not None:
                        supplied_ops = {o for o in _ops_of(
                            res, res.probe_order_before_card) if o}
                        g_sup = {o: v for o, v in g.items() if o in supplied_ops}
                        order_now = list(oc.rank_operators(
                            oc.merge_evidence(
                                state[arm]["source_ev"],
                                state[arm]["target_ev"],
                                source_pseudo_count=SOURCE_PSEUDO_COUNT),
                            lam=lam, tie_break=OPS))
                        p_now, _, _, _ = _probe_table(order_now, g_sup)
                        if p_now != res.target_support_receipts_used:
                            instrument_mismatch.append(
                                {"cell": f"{uid}@{origin}", "arm": arm,
                                 "table_probes": p_now,
                                 "live_probes": res.target_support_receipts_used})
                        if (state[arm]["n_first_effective"] is None
                                and order_now != state[arm]["order_0"]):
                            p0, g0, _, _ = _probe_table(
                                state[arm]["order_0"], g_sup)
                            _, gn, _, _ = _probe_table(order_now, g_sup)
                            better = (p_now < p0) or (gn >= M > g0)
                            if better:
                                state[arm]["n_first_effective"] = (
                                    state[arm]["cum"]["probes"]
                                    + bstat[arm]["probes"])
                                state[arm]["first_effective_round"] = (
                                    f"{uid}@{origin}")
                streamed.append((uid, origin))
                cells_done += 1
                if cells_done % 25 == 0:
                    print(f"  cohort {cohort}: {cells_done} cells  "
                          f"{time.perf_counter() - t0:.0f}s", flush=True)

            # ---- 批结束：折入 Episode，重建卡（STATIC 不动）----
            for arm in ARMS:
                cum = state[arm]["cum"]
                for k in cum:
                    cum[k] += bstat[arm][k]
                state[arm]["history"].append(
                    {"origin": origin, "cells": len(batch),
                     **{k: bstat[arm][k] for k in bstat[arm]},
                     **{f"cum_{k}": cum[k] for k in cum}})
                if arm == "STATIC":
                    continue
                for seen in pending[arm]:
                    oc.accumulate(state[arm]["target_ev"], seen)
                state[arm]["revision"] += 1
                card = _card_for(scope, state[arm]["source_ev"],
                                 state[arm]["target_ev"], lam=lam,
                                 revision=state[arm]["revision"])
                state[arm]["snapshot"] = _install(
                    store, state[arm]["snapshot"], card,
                    edit_id=f"{cohort}_{arm}_rev{state[arm]['revision']}")
                state[arm]["revisions"].append(
                    {"revision": state[arm]["revision"],
                     "order": list(card["risk_guards"]["order"]),
                     "after_origin": origin})

        # ---- 参照：随机顺序 / oracle（表上重放，只用于刻度）----
        rng = np.random.default_rng(20260816)
        rand = {"gain": 0.0, "probes": 0, "harm": 0}
        orac = {"gain": 0.0, "probes": 0, "harm": 0}
        n_rep = 200
        for _key in streamed:
            g = table[_key]
            live = [o for o in OPS if g[o] is not None]
            for _ in range(n_rep):
                order = list(rng.permutation(list(OPS)))
                p, gg, h, _ = _probe_table(order, g)
                rand["gain"] += gg / n_rep
                rand["probes"] += p / n_rep
                rand["harm"] += h / n_rep
            best = sorted(live, key=lambda o: -g[o])
            p, gg, h, _ = _probe_table(best, g)
            orac["gain"] += gg; orac["probes"] += p; orac["harm"] += h

        out["cohorts"][cohort] = {
            "dataset": dataset, "lambda": lam, "cells": len(streamed),
            "supply_identity_checks": supply_checks,
            "instrument_mismatch": instrument_mismatch,
            "reference": {"random": rand, "oracle": orac},
            "arms": {a: {k: state[a][k] for k in
                         ("cum", "history", "revisions", "order_0",
                          "n_first_effective", "first_effective_round")}
                     for a in ARMS}}
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                       encoding="utf-8")

        print(f"\n{'=' * 96}\ncohort {cohort}  dataset={dataset}  lambda={lam}  "
              f"{len(table)} target cells / {len(ORIGINS)} prequential batches\n"
              f"{'=' * 96}")
        print("   {:<10}{:>11}{:>10}{:>10}{:>12}{:>13}{:>12}".format(
            "arm", "cum gain", "probes", "harm n", "harm mag", "gain/probe",
            "N_first_eff"))
        for a in ARMS:
            c = state[a]["cum"]
            print("   {:<10}{:>+11.4f}{:>10}{:>10}{:>12.4f}{:>+13.4f}{:>12}".format(
                a, c["gain"], c["probes"], c["harm_count"], c["harm_magnitude"],
                c["gain"] / max(c["probes"], 1),
                str(state[a]["n_first_effective"])))
        print("   {:<10}{:>+11.4f}{:>10.1f}{:>10.1f}{:>12}{:>+13.4f}".format(
            "[random]", rand["gain"], rand["probes"], rand["harm"], "-",
            rand["gain"] / max(rand["probes"], 1)))
        print("   {:<10}{:>+11.4f}{:>10}{:>10}{:>12}{:>+13.4f}".format(
            "[oracle]", orac["gain"], orac["probes"], orac["harm"], "-",
            orac["gain"] / max(orac["probes"], 1)))
        print("\n   per-batch gain/probe")
        print("   {:<9}{:>12}{:>12}{:>12}{:>13}{:>13}".format(
            "origin", "STATIC", "A3", "A5", "A3-STATIC", "A5-STATIC"))
        for i, origin in enumerate(ORIGINS):
            v = {a: state[a]["history"][i]["gain"]
                 / max(state[a]["history"][i]["probes"], 1) for a in ARMS}
            print("   {:<9}{:>+12.4f}{:>+12.4f}{:>+12.4f}{:>+13.4f}{:>+13.4f}".format(
                origin, v["STATIC"], v["A3"], v["A5"],
                v["A3"] - v["STATIC"], v["A5"] - v["STATIC"]))
        print("\n   card revisions (order after each batch)")
        for a in ARMS:
            print(f"   {a:<8}", [r["order"] for r in state[a]["revisions"]])
        if instrument_mismatch:
            print(f"\n   !! instrument mismatch on {len(instrument_mismatch)} "
                  f"arm-cells (table vs live probe count)")

    out["elapsed_seconds"] = time.perf_counter() - t0
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n-> {OUT.name}  ({out['elapsed_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
