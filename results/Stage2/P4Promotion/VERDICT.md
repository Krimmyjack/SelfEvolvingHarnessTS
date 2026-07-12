# P4 VERDICT：慢路径完整晋升周期（机制验收）

> 范围：mechanism acceptance on synthetic substrate; NOT a performance claim; the term self-evolving stays locked until P6；seed=20260709，n=60，held-in/out=30/30（按 cell 分层）。

## 周期

1. **挖掘**（仅 held-in）：现任 v0（F0 剂量启发式）在 snrLow 服务 f0_median_w25，held-in 证据 = {"anomaly|snrHigh|full": {"mean_true_delta": 0.7061198512675966, "n": 15}, "anomaly|snrLow|miss": {"mean_true_delta": 0.7202855238430891, "n": 15}}
2. **提案**：枚举 proposer 出 3 条 scoped RiskRule（好坏都提，裁决在判官）
3. **晋升**：`mined_ban_f0_median_w25_snr_low` 过 true 判官双段验证——held-in +0.2375、**held-out +0.2641 ≥ ε=0.02**、per-cell {"anomaly|snrHigh|full": 0.0, "anomaly|snrLow|miss": 0.52818865911113}、非目标行 bit 级不变
4. **版本**：bundle_v0.e1（sha a5f26f50e42d8b3a，parent 链落盘 bundles/）
5. **回归重放**：非目标行一致=True，anomaly 面零扰动=True，目标行均值增益 +0.5016（30 行）
6. **rollback**：head→v0 后 serving 与原 v0 bit 级一致（verified=True），随后恢复晋升头 bundle_v0.e1；事件流见 bundles/chain.json
7. **拒绝缓冲**：2 条被 true 判官拒绝（rejected_edits.jsonl 留痕拒因）

## Memory 阶梯（prereg §4 条件线）

- live in deployment gate since P1 (escalation._risk_memory_blocks)
- utility/contrast 阶梯：conditional_pending——prereg §4：utility/contrast memory 须 ①胜 static 学习器 ②first-unseen harm ≤ 阈值 ③in-support 显著优于 out-support——需自身预注册 run，本周期不解锁

## 出口判据

`{"typed_edit_op": true, "promotion_gate": true, "true_judge_held_in_out_validation": true, "version_bump_persisted": true, "regression_replay": true, "rollback_demo": true, "rejected_buffer": true, "cycle_complete": true}`

**cycle_complete = true**：typed EditOp → PromotionGate → true 判官双段验证 → 版本升级落盘 →
回归重放 → rollback 演示 → rejected buffer，七环闭合。不宣称 self-evolving（P6 解锁）。
