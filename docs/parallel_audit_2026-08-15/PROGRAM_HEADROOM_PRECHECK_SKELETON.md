# P1 Program Headroom 预检（空壳预注册，不运行）

目标：在已暴露 development 数据上确认“检索相似 Workflow”这一实验有可检索
的东西。零新 Outcome、零 LLM。

## 判据（全部满足才进入 P2）
1. 候选 Program 池中至少存在两个合法且 effect-distinct 的 Workflow；
2. 不同 Context 的最佳 Workflow 确实不同（不能所有 Context 都是同一个赢家）；
3. 至少部分 Context 存在正向 Support headroom（gain ≥ M 的 Workflow 存在）。

## 输入（只读已有数据）
- 数据集：TO_BE_DECIDED（从已暴露 development context 中选；不消耗 fresh）
- Context 数：TO_BE_DECIDED
- Workflow 池：TO_BE_DECIDED（先列合法 operator，不做 outcome 挑选）
- M：沿用 MATERIAL_THRESHOLD

## 输出
- 每个 Context × Workflow 的 support gain；
- 每个 Context 的 best workflow；
- 满足/不满足三条判据的机械结论。

## 分支
- 三判据全过 → P2 可做；
- 只有一个固定 winner 或无正向 headroom → 不进入 Memory 检索实验，
  先换 Program family。

## 禁止
- 禁止在预检里看 delayed 或未来数据；
- 禁止根据 gain 后验删 Context；
- 禁止调用 Slow。
