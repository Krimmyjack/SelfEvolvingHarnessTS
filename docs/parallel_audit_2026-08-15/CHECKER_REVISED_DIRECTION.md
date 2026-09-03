# Checker 复核：修订方向包

## 确认正确
1. 分支逻辑已按用户原话改为“Pattern 只用于相似 Experience 检索，
   Target Support 决定收益”，未再写“Pattern 直接授权”。
2. P1/P2 空壳均使用 TO_BE_DECIDED，未预选 dataset、workflow、阈值。
3. 明确禁止调用 Slow、不建 Rule Card、不消耗 fresh Outcome。
4. 所有审计结论处置与用户给定五点一致。
5. 未执行任何实验命令；本轮只增加/修改文档。

## 修正项（已执行）
- `SKELETON_SLICE_PROTOCOL.md` 原先仍带 Scope Rule 暗示；已加状态头，
  标记为历史占位并被修订方向接管。

## 残留提示（不阻塞）
- `REVISED_DIRECTION.md` 写的是“G/R/D/X Pattern”；P2 用户原话是
  “现有 29 维 G/R/D/X Pattern”。正式冻结 P2 时应把维度数写死并注明
  特征清单来自本地实验产物，不能只写字母族。
- `RETRIEVAL_SLICE_PREREG_SKELETON.md` 的 verdict rules 尚未设计；
  这是空壳预期，但冻结前必须补：随机基线怎么匹配动作率、如何算
  “更快召回”、如何控制 harm 比较。
