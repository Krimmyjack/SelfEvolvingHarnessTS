"""slow_path/deploy_stream.py — ★v4 S1：流式 domain 持续适应控制器（reset-free）+ 三 bootstrap 消融。

承 Refactor_Continual_TaskReadiness_v4 §1 + S1_Implementation_Plan §B。把现有 per-corpus 进化
（Evolver.run）外包成 domain 序列 D₁→D_K 的持续学习：**一个持久 Evolver，按 domain 换 batch_builder**，
进新 domain 不清 harness/evidence/signatures（reset-free），只重热预算+解冻+warm-start 重验（已落
Evolver.run(domain_idx=...)）。

三 bootstrap（Continual Harness）：
  • scratch (A)   : 每 domain fresh harness+memory → 进化（基线）
  • frozen  (B)   : 载入 D_{k-1} checkpoint，**关进化**，只 eval（B−A=累积记忆价值）；evidence 写
                    临时 store 绝不污染 carried memory
  • updating(C)   : 载入 + 继续进化（C−B=继续更新价值）；每 domain 末 deepcopy checkpoint 供 B 复用

前向迁移日志（喂 S2）：per-(domain k, cell, mode) 落 readiness/time-to-readiness。
readiness=(J_raw−J_cur)/(J_raw−J_min_ref)，单位=round（≈该 cell 参与的 epoch；每 round 1 次 propose）。
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config.thresholds import N_MIN, READINESS_THRESHOLD
from ..harness.state import HarnessState
from ..evaluators import get_evaluator, readiness_score
from ..evaluators.chronos_probe import _fillna
from .batch_builder import BatchBuilder, make_eval_sample
from .evolve import Evolver
from . import mining

SCRATCH, FROZEN, UPDATING = "scratch", "frozen", "updating"


# ════════════════════════════ 数据结构 ════════════════════════════
@dataclass
class DomainSpec:
    """一个 domain 的语料。corpus 元素 = RawSeries（synthetic_gen.py:36；经 fast_path→batch_builder 流转）。"""
    name: str
    corpus: List[Any]                          # List[RawSeries]
    task_types: Tuple[str, ...] = ("forecast",)


@dataclass
class DomainResult:
    domain_idx: int
    name: str
    mode: str
    harness_version: int
    n_reval_demote: int
    cell_logs: List[dict] = field(default_factory=list)
    checkpoint: Any = None                     # mode=updating 末 deepcopy(harness)，供 frozen 复用


@dataclass
class StreamResult:
    mode: str
    domains: List[DomainResult] = field(default_factory=list)

    def checkpoints(self) -> List[Any]:
        return [d.checkpoint for d in self.domains]

    def flat_logs(self) -> List[dict]:
        return [r for d in self.domains for r in d.cell_logs]


# ════════════════════════════ grounded 评估辅助 ════════════════════════════
def _grounded(eb, task: str) -> float:
    """J judge grounded val_loss（越低越好）；frozen+probe 确定性 → 单次。"""
    if not eb:
        return float("nan")
    try:
        return float(get_evaluator(task).evaluate(eb, layer="grounded"))
    except Exception:
        return float("nan")


def _raw_eval_batch(samples):
    """raw（不处理，identity）→ eval batch。raw 可能含 NaN（missing）→ fillna 才可窗化/训练。"""
    return [make_eval_sample(_fillna(np.asarray(s.raw, float)), s) for s in samples]


def _processed_eval_batch(samples, harness):
    """harness 处理后 → eval batch（复用 mining 的一次 fast_path）。"""
    return mining._ready_eval(samples, harness)[1]


def _eval_samples(bb: BatchBuilder, cell_id: str):
    """readiness 测量集 = held_out_a（未被 proposer 直接拟合，诚实）；不足退 held_in。"""
    held_in, held_out_a, _ = bb.splits(cell_id)
    return held_out_a if held_out_a else held_in


@dataclass
class _Anchor:
    task: str
    j_raw: float
    j_min_ref: float


def _compute_anchor(bb: BatchBuilder, cell_id: str, minimal_ref: HarnessState) -> _Anchor:
    """每 domain 每 cell 固定锚（J_raw, J_min_ref）算一次。"""
    samples = _eval_samples(bb, cell_id)
    task = samples[0].task_type
    j_raw = _grounded(_raw_eval_batch(samples), task)
    j_min = _grounded(_processed_eval_batch(samples, minimal_ref), task)
    return _Anchor(task, j_raw, j_min)


def _cur_readiness(bb: BatchBuilder, cell_id: str, harness: HarnessState, anchor: _Anchor,
                   threshold: float) -> Tuple[float, float, bool]:
    """当前 harness 在该 cell 的 (j_cur, readiness, ready?)。"""
    samples = _eval_samples(bb, cell_id)
    j_cur = _grounded(_processed_eval_batch(samples, harness), anchor.task)
    r = readiness_score(anchor.j_raw, j_cur, anchor.j_min_ref)
    return j_cur, r, bool(np.isfinite(r) and r >= threshold)


# ════════════════════════════ 主控 ════════════════════════════
def _ingest(bb: BatchBuilder, corpus: Sequence) -> None:
    for rs in corpus:
        bb.add_raw_series(rs)


def _make_cell_log(k: int, name: str, mode: str, cell: str, anchor: _Anchor,
                   j_cur: float, readiness: float, ready: bool,
                   ttr: Optional[int], harness_version: int) -> dict:
    return {
        "k": k, "domain": name, "mode": mode, "cell": cell, "task": anchor.task,
        "time_to_readiness_rounds": ttr,        # 单位=round（该 cell 参与的 epoch）；未达=None
        "llm_calls_to_readiness": ttr,          # 每 round 1 次 propose → 与上等价（保留双字段供泛化）
        "readiness_at_budget": readiness,
        "ready": ready,
        "j_raw": anchor.j_raw, "j_cur": j_cur, "j_min_ref": anchor.j_min_ref,
        "harness_version": harness_version,
    }


def _evolve_one_domain(evolver: Evolver, bb: BatchBuilder, k: int, dom: DomainSpec,
                       minimal_ref: HarnessState, *, n_epochs: int, min_batches: int,
                       reset_free: bool, threshold: float) -> Tuple[List[dict], int]:
    """scratch/updating 共用：进化 D_k + 记 time-to-readiness。reset_free=True→进化前 enter_new_domain+重验。"""
    evolver.bb = bb
    cells = bb.triggerable_cells(min_batches)
    anchors = {c: _compute_anchor(bb, c, minimal_ref) for c in cells}
    ttr: Dict[str, Optional[int]] = {c: None for c in cells}

    def on_epoch_end(epoch: int, cells_: List[str]) -> None:
        for c in cells_:
            if ttr.get(c) is not None or c not in anchors:
                continue
            _jc, _r, ready = _cur_readiness(bb, c, evolver.h, anchors[c], threshold)
            if ready:
                ttr[c] = epoch + 1               # rounds ≈ 参与的 epoch 数

    run_res = evolver.run(n_epochs, min_batches,
                          domain_idx=(k if reset_free else None), on_epoch_end=on_epoch_end)
    logs = []
    for c in cells:
        j_cur, r, ready = _cur_readiness(bb, c, evolver.h, anchors[c], threshold)
        logs.append(_make_cell_log(k, dom.name, evolver.mode_tag, c, anchors[c],
                                    j_cur, r, ready, ttr[c], evolver.h.version))
    return logs, int(run_res.get("n_reval_demote", 0))


def deploy_stream(domains: List[DomainSpec], *, mode: str,
                  make_harness: Callable[[], HarnessState],
                  make_proposer: Optional[Callable[[], Any]] = None,
                  n_epochs_per_domain: int = 3, min_batches: int = 2, n_min: int = N_MIN,
                  bootstrap_checkpoints: Optional[List[Any]] = None,
                  readiness_threshold: float = READINESS_THRESHOLD,
                  candidate_logger: Optional[Any] = None,
                  log_path: Optional[str] = None) -> StreamResult:
    """流式跑 domain 序列。mode ∈ {scratch, frozen, updating}。frozen 需 bootstrap_checkpoints（来自 updating 的产物）。"""
    if mode not in (SCRATCH, FROZEN, UPDATING):
        raise ValueError(f"mode ∈ scratch|frozen|updating, got {mode!r}")
    if mode in (SCRATCH, UPDATING) and make_proposer is None:
        raise ValueError(f"mode={mode!r} 要进化，必须提供 make_proposer（None 会让 evolve_cell 调 proposer.propose 崩 AttributeError）")
    minimal_ref = HarnessState.from_minimal()
    out = StreamResult(mode)
    persistent: Optional[Tuple[HarnessState, Any, Evolver]] = None

    for k, dom in enumerate(domains):
        if mode == FROZEN:
            base = bootstrap_checkpoints[k - 1] if (bootstrap_checkpoints and k > 0) else make_harness()
            harness = copy.deepcopy(base)
            harness.l3.evidence_store = None             # ★ B 不写 carried store（不污染 memory）
            bb = BatchBuilder(harness, n_min=n_min); _ingest(bb, dom.corpus)
            logs = []
            for c in bb.triggerable_cells(min_batches):
                anchor = _compute_anchor(bb, c, minimal_ref)
                j_cur, r, ready = _cur_readiness(bb, c, harness, anchor, readiness_threshold)
                ttr = 0 if ready else None               # frozen 无进化：载入即就绪→0，否则永不达标
                logs.append(_make_cell_log(k, dom.name, FROZEN, c, anchor, j_cur, r, ready, ttr, harness.version))
            out.domains.append(DomainResult(k, dom.name, FROZEN, harness.version, 0, logs, None))
            continue

        # scratch / updating
        if mode == SCRATCH or persistent is None:
            harness = make_harness()
            store = harness.l3.evidence_store
            proposer = make_proposer() if make_proposer else None
            # S0.5/S0.6b：候选级日志只挂 updating（持续路径，E-6.1/E-4.2/E-7.3 的数据源）
            evolver = Evolver(harness, BatchBuilder(harness, n_min=n_min), proposer, evidence_store=store,
                              candidate_logger=(candidate_logger if mode == UPDATING else None))
            evolver.mode_tag = mode
            if mode == UPDATING:
                persistent = (harness, store, evolver)
        else:
            harness, store, evolver = persistent

        bb = BatchBuilder(harness, n_min=n_min); _ingest(bb, dom.corpus)
        logs, n_demote = _evolve_one_domain(
            evolver, bb, k, dom, minimal_ref,
            n_epochs=n_epochs_per_domain, min_batches=min_batches,
            reset_free=(mode == UPDATING), threshold=readiness_threshold)
        ckpt = copy.deepcopy(harness) if mode == UPDATING else None
        out.domains.append(DomainResult(k, dom.name, mode, harness.version, n_demote, logs, ckpt))

    if log_path:
        write_jsonl(out, log_path)
    return out


def write_jsonl(result: StreamResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for d in result.domains:
            for rec in d.cell_logs:
                f.write(json.dumps({**rec, "n_reval_demote_domain": d.n_reval_demote},
                                   ensure_ascii=False) + "\n")
