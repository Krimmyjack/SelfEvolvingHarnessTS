"""slow_path/evolve.py — 慢路径进化主控（plan.md §6.1，R8 冷启动）。

严格串行 round-robin（#5，无锁，可复现）。每 cell 一轮：
  mine_weakness → proposer 出 K 候选 → 按 rank 逐候选「对当前 harness 重验」→ 过门即合（≤ L_t）。
合入后按 scope 标记冻结 cell 待重检（#4）。epoch 末 consolidate（Strength→受保护区）。
proposer 可注入（测试用 stub，免 LLM）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import math

import numpy as np

from ..fast_path.perceive import perceive
from .batch_builder import BatchBuilder
from .schedule import CellSchedule, ACTIVE, FROZEN, RECHECK
from .validator import Validator
from .merger import Merger
from .attribution import AttributionStore
from . import mining


@dataclass
class RoundResult:
    cell_id: str
    epoch: int
    round_idx: int
    budget: int
    n_proposed: int
    n_accepted: int
    accepted_paths: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


def _task_of(cell_id: str) -> str:
    return cell_id.split("|", 1)[0]


class Evolver:
    def __init__(self, harness, batch_builder: BatchBuilder, proposer,
                 validator: Optional[Validator] = None, merger: Optional[Merger] = None,
                 evidence_store=None, candidate_logger=None):
        self.h = harness
        self.bb = batch_builder
        self.proposer = proposer
        self.validator = validator or Validator()
        self.merger = merger or Merger()
        self.store = evidence_store
        self.candidate_logger = candidate_logger      # S0.5：逐候选 JSONL 日志器（可空；只观测不裁决）
        self.attribution = AttributionStore()
        self.schedules: Dict[str, CellSchedule] = {}
        self.rejection_log: Dict[str, list] = {}
        self.history: List[RoundResult] = []
        self._struct_ref_cache: Dict[str, Dict[str, float]] = {}   # 方向 A：cell → 代表结构锚（懒算缓存）

    def _sched(self, cell_id: str) -> CellSchedule:
        if cell_id not in self.schedules:
            self.schedules[cell_id] = CellSchedule(cell_id)
            self.rejection_log[cell_id] = []
        return self.schedules[cell_id]

    def _attribute(self, cell_id: str, cand, outcome) -> None:
        """记录候选的 outcome-calibrated 信用：用 held_in ∧ held_out(a) 的保守 delta（min）→ 抑制过拟合算子。"""
        d_in = outcome.val_in_cur - outcome.val_in_cand
        d_a = outcome.val_a_cur - outcome.val_a_cand
        delta = min(d_in, d_a) if (math.isfinite(d_in) and math.isfinite(d_a)) else d_in
        self.attribution.record(cell_id, cand, delta)

    def _cell_struct_ref(self, cell_id: str, samples) -> Optional[Dict[str, float]]:
        """cell 代表性结构特征 = held_in 批各序列 struct_feats 的逐维中位（供模板软门 struct_ref）。
        按 cell_id 缓存（同 cell 多次 accept 不重算 FFT/ADF）。"""
        if cell_id in self._struct_ref_cache:
            return self._struct_ref_cache[cell_id]
        feats = []
        for cs in samples:
            try:
                feats.append(perceive(cs.raw, cs.task_type, self.h)["pattern"]["struct_feats"])
            except Exception:
                continue
        if not feats:
            return None
        ref = {k: float(np.median([f.get(k, 0.0) for f in feats])) for k in feats[0]}
        self._struct_ref_cache[cell_id] = ref
        return ref

    def _tag_template_struct_ref(self, cand, cell_id: str, samples) -> None:
        """方向 A（AME-TS 软先验）：accepted 的 **cell-scoped** 模板补打 struct_ref（创建时的结构锚），
        供部署期 d_struct 软门复用——只对带 pattern_bin 的真 cell 模板生效，不动全局模板。"""
        path = cand.path or ""
        if "task_templates" not in path:
            return
        name = path.split("::", 1)[1] if "::" in path else getattr(cand.value, "name", None)
        tmpl = self.h.l2.task_templates.get(name)   # 按 path selector 取（named_object 寻址键，非 value.name）
        if tmpl is None:
            return
        pc = tmpl.applies_to.get("pattern_conditions")
        if not isinstance(pc, dict) or not pc.get("pattern_bin") or pc.get("struct_ref"):
            return                               # 全局模板 / 无 cell 标识 / 已标注 → 跳过
        ref = self._cell_struct_ref(cell_id, samples)
        if ref:
            pc["struct_ref"] = ref

    def _mark_recheck_after_merge(self, scope: str, source_cell: str) -> None:
        """#4 解冻触发：global-scope 编辑 → 所有冻结 cell 标 recheck；cell-scope → 同 task 姊妹。"""
        task = _task_of(source_cell)
        for cid, sc in self.schedules.items():
            if cid == source_cell:
                continue
            if scope == "global" or (scope == "cell" and _task_of(cid) == task):
                sc.mark_recheck()

    def evolve_cell(self, cell_id: str, epoch: int, preserve=None) -> RoundResult:
        sched = self._sched(cell_id)
        held_in, held_out_a, held_out_b = self.bb.splits(cell_id)
        weakness = mining.mine_weakness(cell_id, held_in, self.h, self.store)
        weakness.op_attribution = self.attribution.summary(cell_id)   # 历史 per-op 价值喂 proposer（OPD 公式11）
        # preserve（Strength 约束，advisory）由 run() 每 epoch 算一次传入；standalone 调用时为空（避免 O(cells²)）
        candidates = self.proposer.propose(self.h, weakness, preserve or [], self.rejection_log[cell_id])
        budget = sched.current_budget()
        accepted, applied_paths, reasons = 0, [], []

        for cand in candidates:                       # 已按 proposal_rank 排序
            if accepted >= budget:
                break
            if cand.path in applied_paths:            # 本轮同 path 冲突 → 跳过
                continue
            outcome = self.validator.validate(cand, self.h, cell_id,
                                              (held_in, held_out_a, held_out_b))
            reasons.append(f"{cand.path}:{outcome.reason}")
            if self.candidate_logger is not None:     # S0.5：完整候选（含被拒）落盘，供 E-6.1/E-4.2/E-7.3
                self.candidate_logger.log(cell_id, epoch, sched.round_idx, self.h, cand, outcome,
                                          (held_in, held_out_a, held_out_b), task=_task_of(cell_id))
            self._attribute(cell_id, cand, outcome)   # 记 outcome-calibrated 信用（accept/reject 都记）
            if outcome.accept:
                self.merger.apply(self.h, cand, cached_val_loss=outcome.val_in_cand, store=self.store)
                self._tag_template_struct_ref(cand, cell_id, held_in)   # 方向 A：模板补结构锚
                accepted += 1
                applied_paths.append(cand.path)
                self._mark_recheck_after_merge(outcome.resolved_scope or "global", cell_id)
            else:
                self.rejection_log[cell_id].append(({"path": cand.path}, outcome.reason))

        rr = RoundResult(cell_id, epoch, sched.round_idx, budget,
                         len(candidates), accepted, applied_paths, reasons)
        sched.record_round(accepted, epoch)
        self.history.append(rr)
        return rr

    def revalidate_strength(self, cells: List[str]) -> int:
        """★v4 S1（warm-start 重验，防负迁移）：进入新 domain 时，对载入的受保护 strength
        在**新域数据**上重验——不再带正边际 → 降级 advisory（must_preserve=False，不删，留 Q5 压缩裁）。

        baseline = 去掉该片段的反事实 harness；S1 用 mine_strength 的 floor-margin 作代理（能捕到
        负迁移：新域上 margin≤0 即降级）。lazy：仅对有数据的 cell；frozen+probe ms 级 ≈零额外成本。
        """
        sigs = self.h.l3.strength_signatures
        if not sigs:
            return 0
        demoted = 0
        for cell_id in cells:
            pool = self.bb.pools.get(cell_id, [])
            if not pool:
                continue
            report = mining.mine_strength(cell_id, pool[: self.bb.n_min], self.h)
            if report and report.must_preserve:
                continue                               # 新域上仍带正边际 → 保留
            for st in sigs.values():
                if st.cell_id == cell_id and st.must_preserve:
                    st.must_preserve = False           # 负迁移 → 降级 advisory
                    demoted += 1
        return demoted

    def revalidate_templates(self, cells: List[str]) -> int:
        """S0.4（warm-start 迁移重验，防 cell-scoped 模板负迁移；Critical 3.1 技术债）：进入新 domain 时，
        对导入的 **cell-scoped** 模板做**配对对照**重验——H_with_template vs H_same_but_without_template
        （同一迁移后 harness，仅摘除该模板，不与 minimal 比，避免其他已迁移组件混淆）。

        评估用**独立 transfer-gate split**（第 4 段，不消耗 held_out(a)，A-16）；不足则回退 held_in。
        摘除后显著更好（v_without < v_with − ε）→ 模板对该新域是负迁移 → demote 为 advisory（不删，留 Q5 裁）。
        """
        from .validator import grounded_val_loss
        from ..fast_path.compose import matching_templates
        tmpls = self.h.l2.task_templates
        if not any(isinstance(t.applies_to.get("pattern_conditions"), dict) for t in tmpls.values()):
            return 0
        eps = self.validator.eps
        demoted: set = set()
        for cell_id in cells:
            samples = self.bb.transfer_gate_split(cell_id) or self.bb.splits(cell_id)[0]
            if not samples:
                continue
            task = _task_of(cell_id)
            feats = self._cell_struct_ref(cell_id, samples)
            pb = ""
            try:
                pb = perceive(samples[0].raw, task, self.h).get("pattern_bin", "")
            except Exception:
                pass
            cell_tmpls = [t for t in matching_templates(self.h, task, pb, feats)
                          if isinstance(t.applies_to.get("pattern_conditions"), dict)
                          and t.applies_to["pattern_conditions"].get("pattern_bin")
                          and t.name not in demoted]
            for tmpl in cell_tmpls:
                name = tmpl.name
                v_with = grounded_val_loss(self.h, samples)
                snap = self.h.snapshot()
                try:
                    self.h.l2.task_templates.pop(name, None)
                    v_without = grounded_val_loss(self.h, samples)
                finally:
                    self.h.restore(snap)              # 恢复的是 deepcopy → 须在恢复后的对象上打标
                if (np.isfinite(v_with) and np.isfinite(v_without)
                        and v_without < v_with - eps and name in self.h.l2.task_templates):
                    self.h.l2.task_templates[name].applies_to["advisory"] = True
                    demoted.add(name)
        return len(demoted)

    def run(self, n_epochs: int = 3, min_batches: int = 2, *,
            domain_idx: Optional[int] = None, on_epoch_end=None) -> Dict:
        cells = self.bb.triggerable_cells(min_batches)
        for c in cells:
            self._sched(c)
        n_reval_demote = 0
        if domain_idx is not None:           # ★v4 流式：进新 domain → 重热预算+解冻 + warm-start 重验
            for c in cells:
                self._sched(c).enter_new_domain(domain_idx)
            n_reval_demote = self.revalidate_strength(cells)
            n_reval_demote += self.revalidate_templates(cells)   # S0.4：模板迁移重验
        total_accepts = 0
        for epoch in range(n_epochs):
            # Strength 每 epoch 算一次（O(cells)，非 O(cells²)）：既作 proposer 的 preserve 约束，也供 epoch 末整固
            strength = [s for s in (mining.mine_strength(c, self.bb.pools[c][: self.bb.n_min], self.h)
                                    for c in cells) if s]
            preserve = [s for s in strength if s.must_preserve]
            for cell_id in list(cells):
                sched = self._sched(cell_id)
                if not sched.is_schedulable(epoch):
                    continue
                rr = self.evolve_cell(cell_id, epoch, preserve)
                total_accepts += rr.n_accepted
            self.merger.consolidate(self.h, strength, epoch)     # 第三时间尺度
            if on_epoch_end is not None:                          # ★v4 S1：供 deploy_stream 记 time-to-readiness
                on_epoch_end(epoch, cells)
        return {"epochs": n_epochs, "cells": cells, "total_accepts": total_accepts,
                "final_version": self.h.version, "n_reval_demote": n_reval_demote,
                "frozen": [c for c, s in self.schedules.items() if s.status == FROZEN]}
