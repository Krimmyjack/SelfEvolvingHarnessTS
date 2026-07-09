"""B.2 #4/#6 验证：edit_budget 退火曲线 + CellSchedule 冻结/解冻/强检状态机。

运行：  python -m SelfEvolvingHarnessTS.tests.test_schedule   （cwd=Agent）
"""
from __future__ import annotations

from SelfEvolvingHarnessTS.slow_path.schedule import edit_budget, CellSchedule, ACTIVE, FROZEN, RECHECK
from SelfEvolvingHarnessTS.config import thresholds as TH


# ── #6 edit_budget cosine 退火 ───────────────────────────────────────────
def test_edit_budget_cosine():
    assert edit_budget(0) == TH.EDIT_BUDGET_MAX                 # t=0 → max(=K=3)
    assert edit_budget(TH.EDIT_BUDGET_TOTAL_STEPS // 2) == 2    # t=T/2 → 中点
    assert edit_budget(TH.EDIT_BUDGET_TOTAL_STEPS) == TH.EDIT_BUDGET_MIN     # t=T → min
    assert edit_budget(99) == TH.EDIT_BUDGET_MIN               # 超 horizon → clamp min
    # 单调不增
    seq = [edit_budget(t) for t in range(TH.EDIT_BUDGET_TOTAL_STEPS + 1)]
    assert all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))


def test_edit_budget_modes():
    assert edit_budget(5, mode="constant") == TH.EDIT_BUDGET_MAX
    assert edit_budget(0, mode="linear") == TH.EDIT_BUDGET_MAX
    assert edit_budget(TH.EDIT_BUDGET_TOTAL_STEPS, mode="linear") == TH.EDIT_BUDGET_MIN


# ── #4 冻结：连续 N_FREEZE 轮全拒 ────────────────────────────────────────
def test_freeze_after_consecutive_rejects():
    cs = CellSchedule("forecast|snrLow|miss")
    for r in range(TH.N_FREEZE - 1):
        cs.record_round(0, epoch=0)
        assert cs.status == ACTIVE                              # 还没到门槛
    cs.record_round(0, epoch=0)
    assert cs.status == FROZEN and cs.frozen_at_epoch == 0
    assert cs.round_idx == TH.N_FREEZE


# ── #4 接受重置 ──────────────────────────────────────────────────────────
def test_accept_resets():
    cs = CellSchedule("c")
    cs.record_round(0, 0); cs.record_round(0, 0)
    assert cs.consecutive_rejects == 2
    cs.record_round(1, 0)                                       # 接受 → 重置
    assert cs.consecutive_rejects == 0 and cs.status == ACTIVE


# ── #4 解冻触发（相邻/全局 harness 变更）+ 周期强检 ─────────────────────
def test_unfreeze_and_force_recheck():
    cs = CellSchedule("c")
    for _ in range(TH.N_FREEZE):
        cs.record_round(0, epoch=2)
    assert cs.status == FROZEN
    # 周期强检：冻结 ≥ M epoch
    assert not cs.force_recheck_due(epoch=2 + TH.FREEZE_RECHECK_EPOCHS - 1)
    assert cs.force_recheck_due(epoch=2 + TH.FREEZE_RECHECK_EPOCHS)
    # 相邻/全局变更 → mark_recheck
    assert cs.mark_recheck() and cs.status == RECHECK
    assert cs.is_schedulable(epoch=99)                          # recheck 必跑
    # active cell mark_recheck 无效（只对 frozen）
    assert not CellSchedule("d").mark_recheck()


# ── round_idx 跨冻结持续累加（budget 不因冻结重置）────────────────────────
def test_round_idx_persists():
    cs = CellSchedule("c")
    for _ in range(TH.N_FREEZE):
        cs.record_round(0, 0)
    b_before = cs.current_budget()
    cs.mark_recheck()
    cs.record_round(1, 0)                                       # 解冻后继续
    assert cs.round_idx == TH.N_FREEZE + 1                      # 持续累加，未重置
    assert cs.current_budget() <= b_before                     # 预算继续退火（不回升）


def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
