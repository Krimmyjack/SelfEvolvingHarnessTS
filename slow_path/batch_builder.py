"""slow_path/batch_builder.py — holding pool → (pattern_bin,task) 分组 + 三 split 供给（plan.md §4.2/R2）。

接收原始 (raw_input, task, 真值) 样本，按 perceive→binning 的 cell_id 归入各 cell 的 holding pool；
某 cell 攒够 ≥2 batch 即可触发进化。splits() 给 validator 提供：
  held_in（动机 batch，proposer 见过）/ held_out(a)（同 cell 未见 batch）/ held_out(b)（其他 cell，跨 cell Pareto）。
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config.thresholds import N_MIN
from ..evaluators import ForecastSample, AnomalySample, ClassifySample
from ..fast_path.perceive import perceive

# S0.2：层内可用 series_uid 少于此 → 退化为纯分组无分层，cell 标记低置信（A-1/风险表）
MIN_UIDS_FOR_STRATIFY = 6


@dataclass
class CellSample:
    """慢路径的原始样本：raw 喂 fast_path，真值用于 grounded 评估。"""
    raw: np.ndarray
    task_type: str
    future: Optional[np.ndarray] = None      # forecast
    obs_scale: float = 1.0
    period: int = 24
    positions: Optional[List[int]] = None    # anomaly
    label: Optional[int] = None              # classify
    origin: str = ""                         # 数据集/网格预设名（S0.2 分层）
    series_uid: str = ""                     # 基底序列身份（S0.2 分组；同基底副本共享）


def cell_sample_from_raw_series(rs) -> CellSample:
    """适配 data.RawSeries（按 task 取正确的 fast_path 输入字段）。"""
    origin = getattr(rs, "origin", "") or ""
    uid = getattr(rs, "series_uid", "") or ""
    if rs.task == "forecast":
        return CellSample(rs.history, "forecast", future=rs.future,
                          obs_scale=rs.obs_scale, period=rs.period, origin=origin, series_uid=uid)
    if rs.task == "anomaly_detection":
        return CellSample(rs.anomaly_input, "anomaly_detection", positions=list(rs.anomaly_positions),
                          origin=origin, series_uid=uid)
    return CellSample(rs.degraded, "classification", label=rs.label, period=rs.period,
                      origin=origin, series_uid=uid)


def make_eval_sample(ready: np.ndarray, cs: CellSample):
    """(ready_artifact, CellSample) → 对应任务的 Eval*Sample。"""
    if cs.task_type == "forecast":
        return ForecastSample(ready, cs.future, cs.obs_scale, cs.period)
    if cs.task_type == "anomaly_detection":
        return AnomalySample(ready, cs.positions or [])
    return ClassifySample(ready, int(cs.label))


Splits = Tuple[List[CellSample], List[CellSample], Dict[str, List[CellSample]]]


class BatchBuilder:
    def __init__(self, harness, n_min: int = N_MIN):
        self.harness = harness
        self.n_min = n_min
        self.pools: Dict[str, List[CellSample]] = defaultdict(list)

    def add(self, cs: CellSample) -> str:
        key = perceive(cs.raw, cs.task_type, self.harness)   # cell_id 仅依赖 raw 的 struct_feats
        cid = key["cell_id"]
        self.pools[cid].append(cs)
        return cid

    def add_raw_series(self, rs) -> str:
        return self.add(cell_sample_from_raw_series(rs))

    def triggerable_cells(self, min_batches: int = 2) -> List[str]:
        need = min_batches * self.n_min
        return [c for c, p in self.pools.items() if len(p) >= need]

    # ── S0.2/F2：series_uid 分组 + origin 分层的确定性四段划分 ──────────────
    _N_SEG = 4   # held_in / held_out_a / final_test / transfer_gate（S0.4 用第 4 段）

    def _partition(self, cell_id: str, bs: int) -> List[List[CellSample]]:
        """把一个 cell 的 pool 划成 4 段：held_in / held_out_a / final_test / transfer_gate。

        不变量：①同一 series_uid 的所有退化副本进同一段（防基底泄漏）；②组列表按 origin 分层、
        SHA256(cell_id) 确定性洗牌后顺序装填（各段目标 bs，整组原子装入 → 不跨段切割）；
        ③uid 数 < MIN_UIDS_FOR_STRATIFY 或单一 origin → 纯分组无分层（低置信）。
        前三段填法与旧 pool[:3bs] 语义一致；第 4 段（transfer_gate）吸收原被丢弃的余量、群组不相交。
        """
        pool = self.pools.get(cell_id, [])
        if not pool:
            return [[] for _ in range(self._N_SEG)]
        groups: "OrderedDict[str, List[CellSample]]" = OrderedDict()
        for i, s in enumerate(pool):
            uid = s.series_uid or f"__pos{i}"          # 缺 uid → 按位置视作独立组（不误并）
            groups.setdefault(uid, []).append(s)

        strata: "OrderedDict[str, List[str]]" = OrderedDict()
        for uid, members in groups.items():
            strata.setdefault(members[0].origin or "", []).append(uid)

        seed = int.from_bytes(hashlib.sha256(cell_id.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(seed)             # str hash 跨进程随机 → 必须 SHA256（A-1）
        stratify = len(groups) >= MIN_UIDS_FOR_STRATIFY and len(strata) >= 2
        if stratify:                                  # 各层内洗牌 → 跨层 round-robin 交错
            lists = []
            for origin in sorted(strata.keys()):
                lst = strata[origin]
                lists.append([lst[j] for j in rng.permutation(len(lst))])
            ordered_uids: List[str] = []
            for j in range(max(len(l) for l in lists)):
                for l in lists:
                    if j < len(l):
                        ordered_uids.append(l[j])
        else:
            uids = list(groups.keys())
            ordered_uids = [uids[j] for j in rng.permutation(len(uids))]

        seg: List[List[CellSample]] = [[] for _ in range(self._N_SEG)]
        si = 0
        for uid in ordered_uids:
            while si < self._N_SEG and len(seg[si]) >= bs:
                si += 1
            if si >= self._N_SEG:
                break                                 # 超过 N_SEG·bs 的余量丢弃
            seg[si].extend(groups[uid])
        return seg

    def series_uid_count(self, cell_id: str) -> int:
        return len({s.series_uid or f"__pos{i}" for i, s in enumerate(self.pools.get(cell_id, []))})

    def is_low_confidence(self, cell_id: str) -> bool:
        """S0.2：可用 series_uid < MIN_UIDS_FOR_STRATIFY → 纯分组无分层，进 E-1.1 白名单排除名单。"""
        return self.series_uid_count(cell_id) < MIN_UIDS_FOR_STRATIFY

    def splits(self, cell_id: str, batch_size: Optional[int] = None) -> Splits:
        bs = batch_size or self.n_min
        seg = self._partition(cell_id, bs)
        held_in, held_out_a = seg[0], seg[1]
        held_out_b: Dict[str, List[CellSample]] = {}
        for c in self.pools:
            if c == cell_id:
                continue
            hi_c = self._partition(c, bs)[0]          # 其他 cell 的分组 held_in 作跨 cell Pareto 组
            if len(hi_c) >= max(2, bs // 2):
                held_out_b[c] = hi_c
        return held_in, held_out_a, held_out_b

    def final_test(self, cell_id: str, batch_size: Optional[int] = None) -> List[CellSample]:
        """final-test split —— **进化期从不触碰**，仅主表 ΔPerf 报告期用（Experiment_Design §★.5）。

        与 held_in/held_out_a 同源自 `_partition`：按 series_uid 分组，故与二者 **基底不相交**
        （S0.2 消除了旧 pool[2bs:3bs] 位置切割可能造成的同基底副本跨 split 泄漏）。
        """
        bs = batch_size or self.n_min
        return self._partition(cell_id, bs)[2]

    def transfer_gate_split(self, cell_id: str, batch_size: Optional[int] = None) -> List[CellSample]:
        """S0.4：第 4 段 —— warm-start 时对导入 cell-scoped 模板做迁移重验用；
        与 held_in/held_out_a/final_test 基底不相交（不消耗 held_out(a)，A-16）。可能为空（池小）。
        """
        bs = batch_size or self.n_min
        return self._partition(cell_id, bs)[3]
