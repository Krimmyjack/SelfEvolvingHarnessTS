# Reviewer 审核：修订方向包

## 裁决
通过。准备包已与用户修订分支一致：
- 等待对象 = Pattern-conditioned retrieval 结果；
- 第一修复面 = MEMORY/检索，不是 SCOPE_RISK Rule Card；
- P1/P2 只留空壳，不运行；
- 审计发现中的 delayed 重复评估、中性 delayed 语义、Scope 收缩
  按优先级延后，不进入本轮。

## 后续允许动作
1. 等待本地 Pattern 检索结果；
2. 结果有效且 P1 三判据由已暴露数据确认后，才允许把
   `RETRIEVAL_SLICE_PREREG_SKELETON.md` 升级为冻结协议；
3. 升级时必须补 checker 提示的两项：
   - 29 维特征清单与归一化公式写死；
   - verdict rules（动作率匹配随机基线、更快召回定义、harm 比较）写死。

## 后续禁止
- 不得在 P1 前运行 P2；
- 不得用 P2 数据训练 Pattern→gain 模型；
- 不得把 Pattern 直接作为执行授权；
- 不得新建 Memory 平台或改 Skill 正文；
- 不得在用户回来前消耗任何 fresh Outcome 或调用 Slow。

## 状态
准备包封版，等待本地 Agent 的 Pattern-conditioned retrieval 结果。
