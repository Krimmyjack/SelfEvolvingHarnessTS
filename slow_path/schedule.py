"""slow_path/schedule.py — 编辑预算调度（B.2 #6）+ per-cell 冻结状态机（B.2 #4）。

借 SkillOpt 的 LR 调度形态，但"一个 edit"= 一个完整 EditPatch（非原子文本行），故 L_t 的语义是
**每轮 merger 最多接受几个过门候选**（不是从原子编辑池 clip）。cosine 退火：早期 L_t=max_lr（探索，
接受所有过 grounded 门的候选）→ 后期 L_t=min_lr（固化，只接受 proposal_rank 最高的，防非可加交互）。

冻结状态机（#4）：连续 N_FREEZE 轮全拒 → 冻结；任一相邻/全局 harness 变更 → 标记 recheck；
冻结 ≥ FREEZE_RECHECK_EPOCHS epoch → 强制重评（安全网，无永久冻结）。round_idx 跨冻结/解冻持续累加。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..config.thresholds import (
    EDIT_BUDGET_MAX, EDIT_BUDGET_MIN, EDIT_BUDGET_TOTAL_STEPS, EDIT_BUDGET_MODE,
    N_FREEZE, FREEZE_RECHECK_EPOCHS, DOMAIN_BUDGET_CEILING_DECAY,
)

# cell 进化状态
ACTIVE, FROZEN, RECHECK = "active", "frozen", "recheck"


def edit_budget(round_idx: int, *, max_lr: int = EDIT_BUDGET_MAX, min_lr: int = EDIT_BUDGET_MIN,
                total_steps: int = EDIT_BUDGET_TOTAL_STEPS, mode: str = EDIT_BUDGET_MODE) -> int:
    """第 round_idx 轮（0-based）的编辑预算 L_t ∈ [min_lr, max_lr]（整数）。

    cosine：t=0→max_lr，t=T/2→中点，t≥T→min_lr。constant→max_lr。linear→线性退火。
    """
    if round_idx >= total_steps or max_lr <= min_lr:
        return min_lr
    span = max_lr - min_lr
    if mode == "constant":
        lr = max_lr
    elif mode == "linear":
        lr = max_lr - span * (round_idx / total_steps)
    else:  # cosine
        lr = min_lr + 0.5 * span * (1.0 + math.cos(math.pi * round_idx / total_steps))
    return max(min_lr, int(round(lr)))


@dataclass
class CellSchedule:
    """单个 cell 的进化调度状态。evolve.py 每 cell 持有一个。"""
    cell_id: str
    round_idx: int = 0
    consecutive_rejects: int = 0
    status: str = ACTIVE
    frozen_at_epoch: int = -1
    domain_idx: int = 0                       # ★v4 S1：当前所处 domain 序号（meta-退火用）

    def _eff_max_lr(self) -> int:
        """★v4 meta-退火：逐 domain 降预算天花板（记忆越成熟、需新编辑越少）。"""
        return max(EDIT_BUDGET_MIN, EDIT_BUDGET_MAX - self.domain_idx * DOMAIN_BUDGET_CEILING_DECAY)

    def current_budget(self) -> int:
        return edit_budget(self.round_idx, max_lr=self._eff_max_lr())

    def enter_new_domain(self, domain_idx: int) -> None:
        """★v4 S1（reset-free 流式）：进入新 domain 时**重热预算 + 解冻**。

        cell 是 domain-agnostic 的 → 新域新数据可改变该 cell 的最优 harness，故 freeze 是
        domain-scoped：解冻给它重新适配的机会。budget round_idx 归零回到 cosine 起点（重热），
        天花板按 domain_idx meta-退火降低。**跨域"拒绝信息"不在此**——在 Evolver.rejection_log
        （Q4 跨域携带 + staleness 衰减）；本对象只管 budget/freeze。
        """
        self.domain_idx = domain_idx
        self.round_idx = 0                    # 重热：budget 回 cosine 起点（eff_max）
        self.status = ACTIVE                  # ★解冻（freeze domain-scoped）
        self.consecutive_rejects = 0
        self.frozen_at_epoch = -1

    def record_round(self, n_accepted: int, epoch: int) -> None:
        """一轮 mining→propose→validate→merge 结束后调用。n_accepted=本轮合入的候选数。"""
        self.round_idx += 1
        if n_accepted > 0:
            self.consecutive_rejects = 0
            self.status = ACTIVE
            self.frozen_at_epoch = -1
        else:
            self.consecutive_rejects += 1
            if self.consecutive_rejects >= N_FREEZE:
                self.status = FROZEN
                self.frozen_at_epoch = epoch

    def mark_recheck(self) -> bool:
        """相邻/全局 harness 变更触发：冻结的 cell → recheck（下一 epoch 强制跑 1 轮）。返回是否转换。"""
        if self.status == FROZEN:
            self.status = RECHECK
            self.consecutive_rejects = 0     # 给重检一个干净的拒绝计数起点
            return True
        return False

    def force_recheck_due(self, epoch: int) -> bool:
        """周期性安全网：冻结 ≥ FREEZE_RECHECK_EPOCHS epoch → 该 epoch 无条件强制重评。"""
        return self.status == FROZEN and self.frozen_at_epoch >= 0 \
            and (epoch - self.frozen_at_epoch) >= FREEZE_RECHECK_EPOCHS

    def is_schedulable(self, epoch: int) -> bool:
        """本 epoch 是否参与 round-robin：active/recheck 跑；frozen 仅在到达强检周期时跑。"""
        return self.status in (ACTIVE, RECHECK) or self.force_recheck_due(epoch)
