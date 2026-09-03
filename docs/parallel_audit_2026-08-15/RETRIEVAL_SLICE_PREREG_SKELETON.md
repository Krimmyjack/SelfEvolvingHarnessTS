# P2 Pattern-Conditioned Retrieval 离线切片（空壳预注册，不运行）

## 研究问题（P0 冻结版）
相同 Source Episode、相同候选预算、相同 Target Support 预算下，
Pattern-conditioned retrieval 是否比不使用 Pattern 的检索更快召回有效
Workflow。Surface = MEMORY，不是 SCOPE_RISK Rule Card。

## 两臂
- H0：相同 Source 正/负/冲突 Episodes，不按 Pattern 排序。
- H1：使用现有 Pattern（G/R/D/X，Source-bank z-score + L2 检索），
  分别返回最近的正向、负向、冲突案例。

## 两臂共同不变
- 先做合法性过滤；
- identity + 最多两个非 identity 候选；
- F4 同时用于两臂，只过滤无材料性候选；
- Pattern 只改变候选来源和顺序，不授权执行；
- 当前 Target Support 决定收益；
- 确定性 replay，不调用 Slow，不建 Rule Card；
- 不消耗 fresh Outcome。

## 指标
- Top-2 中包含正向 Workflow 的比例；
- 找到首个正向 Workflow 所需 Support 次数；
- 每次 probe 的平均 gain；
- harm/probe 与累计 harm；
- abstention。

## 关键对照纪律
选择能力必须与“动作率匹配的随机排序”比较，避免把“少行动”误判为
“检索准确”。

## 参数槽
- source episodes = TO_BE_DECIDED
- development contexts = TO_BE_DECIDED
- pattern feature set = 本地 Pattern 实验输出（G/R/D/X 或等价）
- retrieval distance = Source-bank z-score + L2（具体归一化以本地输出为准）
- probe budget = TO_BE_DECIDED
- verdict rules = TO_BE_DECIDED（不得在运行后调整）

## 冻结前置
1. P1 三判据全过；
2. 本地 Pattern 实验明确给出可复用的检索特征；
3. 两臂候选来源和顺序规则在运行前逐字冻结。

## 禁止
- 禁止 Pattern 直接决定执行；
- 禁止把 retrieval score 当作 gain；
- 禁止在结果出现后改指标或换 Context。
