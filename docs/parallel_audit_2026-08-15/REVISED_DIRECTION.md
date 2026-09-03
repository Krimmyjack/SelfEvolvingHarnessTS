# 修订方向（2026-08-15 夜）

## 为什么改
本地 Pattern 实验只测试了“Pattern 直接预测 gain”，没有测试
“相似 Context 检索”。因此正确分支不是：

```text
Pattern 不能预测 gain → 无 trigger → 不跑 MEMO-B → 换 Program/收束
```

而是：

```text
Pattern 不能直接授权执行
  → Pattern 只用于相似 Experience 检索
  → 当前 Target Support 决定收益
```

## 审计结论处置
1. `open_delayed` 2–3 次重复评估：真问题；正式 delayed 实验前做最小修复
   （评估一次，A/B/C 复用，只加一个集成测试）。
2. 中性 delayed 语义冲突：属实，但当前不修。最终语义：
   delayed ≥ M → LOCAL_ACTIVE；−M ≤ delayed < M → LOCAL_DRAFT（仍需 Support）；
   delayed < −M → 撤销/限制。
3. 无 Scope 收缩：属实，但暂不建设。整卡撤销当前安全；只有真实出现
   “部分 Context 有效、部分有害且存在可表达边界”才做 SPLIT/RESTRICT。
4. Fault Family 映射：只保留文档，不写代码，不建 Router。
5. BSE 可复用：正确，但不是下一实验。Pattern 相似检索不是
   `feature ≥ τ`，不能硬塞进 BSE matcher；未来有真正 Scope trigger 再复用。

## 修订后的实验顺序
- P0：冻结研究问题——Pattern-conditioned retrieval 是否比无 Pattern 检索
  更快召回有效 Workflow。
- P1：Program headroom 预检（已暴露 development 数据，零新 Outcome）。
- P2：Pattern 检索离线纵向切片（确定性 replay，不调 Slow，不建 Rule Card）。
  特征 = 现有 29 维 G/R/D/X Pattern（正式冻结时以本地实验产物清单为准）。
- P3：只有 P2 正向才进真实 Fast Path（runtime_prior_slot）。
- P4：生命周期与 delayed 修复（P3 改变 Fast 行为后）。
- P5：最终 A5 vs A3。

## 当前不做什么
- 不把线上空壳协议填成 Scope Rule；
- 不继续训练 Pattern→gain 模型；
- 不实施完整六类 Fault Router；
- 不建设 Scope SPLIT 平台；
- 不把当前结果收束成“Pattern/Memory 无效”。

## 准备包状态
等待的是“Pattern-conditioned retrieval 结果”，不是
“Pattern 是否产生直接 Utility trigger”。
