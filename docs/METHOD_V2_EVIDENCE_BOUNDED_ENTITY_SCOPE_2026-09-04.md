# 方法 v2 设计:证据有界的实体 Scope + 两决策部署(主线,2026-09-04)

授权与定向:用户转述导师意见——不要"保底论文",要做出成果、实现项目目标;重心放在设计方法克服已发现的问题。据此
**设计阶段重新打开**,但**只针对四个已量化的约束**,不做开放式设计。治理层(预注册 / 信息墙 / 权威门 / 仪器-科学分离)
**不撤**——它是结果可信的唯一原因。本稿是设计,不是合同;先经 §5 的 0-LLM 可行性重放,再决定是否冻结。

## 1. 约束 → 杠杆(每条设计选择对应一个已测量的失败原因)

| 约束(已测量) | 证据 | 杠杆 |
| --- | --- | --- |
| C1 部署期无安全预测子 | D1:四信号 AUC 0.45–0.65,`NO_OUTCOME_FREE_SEPARATOR` | 不用特征**预测**谁会被伤,用每条序列自己的**已验证历史**决定可否部署 |
| C2 尾部来自新进入者;持续成员大体稳定 | D1/D5:严重伤害新进入者 8/10;D6:持续成员保持受益 7/11 | Fast-only 只部署到有证据的序列;新进入者只探针不部署 |
| C3 收益与伤害同走模型路由,交互 ≈ 0 | D5 `ROUTE_DOMINANT` 73% | 拆两决策:哪些序列的训练行被准备(建模型)vs 哪些序列换到 program model |
| C4 treatment 稀疏 | HEC-1 62/69 平局;5-call ≈ 1 次提议;10/11 Draft FLAGGED、0 修订 | 语义预算;记忆供给候选零成本;修订下沉到实体粒度,每单元都会发生 |
| 目标 headroom | §5.1:per-series oracle +0.61 vs 最佳固定 +0.26 vs 特征 Router +0.19 | v2 的目标 = **安全地逼近逐序列上界**,而非按"像不像"圈序列 |

## 2. Skill v2 定义

```
Skill = Program(typed, ≤2 步)
      + PatternScope(部署可见特征谓词)        → 探针资格:谁可以被 Support 探针
      + EntityEvidence[consumer][entity]       → 部署资格:谁可以被 Fast-only 路由到 program model
          {positives, harms, last_window, quarantined}
```

- **探针资格**由 PatternScope 决定(与 v1 相同,Slow 可收窄)。
- **部署资格**由 EntityEvidence 决定:`positives ≥ k`(拟 k=2:Support + delayed 各一次即够,或两窗口)∧ `not quarantined`。
- **未评估过的实体永远不部署**——C2 直接构造性消除;阈值 0.005 / 0.20 / 0.30 一个不改。
- Consumer 进键:同一实体在 pooled 与 per-channel 下的证据分开(D1/D5 已证形状不同)。

## 3. 两决策部署(C3)

| 决策 | 选项 A | 选项 B | 如何定 |
| --- | --- | --- | --- |
| 训练侧:program model 用谁的准备后训练行 | 全部 PatternScope 匹配序列 | 仅 EntityEvidence 合格序列 | 在 HEC-1 已记录数据上 0-LLM 离线比较(§5),按 harm ≤ 前提下的增益择一并冻结 |
| 服务侧:谁路由到 program model | — | 仅 EntityEvidence 合格序列 | 固定(C2) |

D5 说明"谁换模型"是收益与伤害的主载体,且逐序列可分——服务侧按证据逐序列路由,正是逼近 per-series oracle 的机制。

## 4. 生命周期 v2(治理不变,粒度下沉)

- **held-in 每单元**:PatternScope 匹配 → 供给/探针(Support)→ 权威门(四线不变)→ 通过:该单元治疗集内每条序列 `positives += 1`;
  逐序列受害(< −material):该序列 `harms += 1`,达阈(拟 1 次严重或 2 次实质)→ `quarantined`。
- **delayed / 评价面**:同规则更新证据(评价面 Outcome 仍不进 bank——只作计分;证据只来自 Support/delayed)。
- **Fast-only(held-out)**:冻结的 EntityEvidence 决定路由;无证据 → raw;因此 held-out 上**新进入者伤害 = 0 by construction**,
  代价是覆盖只及已证据化实体(如实报告覆盖率)。
- **修订**:实体级加入/隔离每单元发生(C4);PatternScope 收窄只针对"匹配却反复未过探针"的新进入者族;程序级 ADD / 撤销照 v1.1。
  三态机保留于程序级;**"持续成员受伤"不再 FLAG 整张卡,而是隔离该实体**——P2 类事件从 0 变为常态。
- **外环**照 v1.1(普查 → Slow 语义 → 工具校准 → replay 筛 → 新单元验证),对象增加"证据账本的一致性普查"(同程序跨实体的
  正/害分布)。

## 5. 0-LLM 可行性重放(先做,两天,决定是否冻结)

- **材料**:HEC-1 三顺序 26 单元 × 4 臂在 Support / delayed / +144 三面的**逐序列增益记录**(scoring ledger + cells),
  BSG 与 validation-search 的候选逐序列读数。
- **重放**:按顺序遍历单元,用记录的 Support/delayed 逐序列结果更新 EntityEvidence(k=2、隔离规则如上),在评价面按 v2 部署
  规则(仅证据合格实体路由到该程序)用记录的逐序列增益计分;训练侧 A/B 两选项各算一遍(B 需少量重拟合,≤ 数百 fits)。
- **读数**:v2 曲线(相对 Static 累计)vs v1 四臂;**逼近 per-series oracle 的比例**;harm 事件数;覆盖率;treatment 密度
  (逐单元证据更新数、部署差异数);三顺序一致性。
- **决策规则(预写)**:v2 在三顺序上 harm ≤ v1 online 且累计增益 ≥ v1 online + 物质线,并收回 oracle 上界 ≥ 30% → 冻结 v2
  进 live;否则回到本表重看杠杆,不进 live。**这是 development 级证据**(已曝光数据、事后规则),只决定要不要花 live 预算。

## 6. 正典修订请求(呈 sol)

现行:`serving_scope` 存部署可见特征谓词,不存 UID;数据集/域名不得作 Scope 理由。原意 = 防数据集身份泄入**跨域**迁移。
请求改为两层:**跨数据集/跨域的 Scope 与 Skill 不得含实体身份**(不变);**同域 Target-local Memory 可含实体级验证证据**
(部署可见:运维者知道这是哪个传感器;证据只由本域 held-in 反馈产生;不随 Skill 跨数据集迁移——迁移的是程序 + PatternScope +
证据阈值规则,实体账本在新域从空开始)。这把"当初要防的"与"现在需要的"分开。

## 7. 与实验计划的衔接

- §5 通过 → 冻结 **HEC-2'**:KDD 上 live,方法 v2 + 语义预算(≥1 完整决策、≤2、物理顶 10)+ 前瞻 discordance 门 + 四臂
  (Static / A5-frozen / A5-online / A3-online)+ validation-search 0-LLM;预期 treatment 稠密、可识别。
- 之后 Dataset/Domain 主实验以 v2 为方法:跨数据集迁移程序 + PatternScope,实体账本在每个新数据集的 held-in 里重建——
  A5 的价值 = 更快建起证据(先验告诉它先探什么)+ 更少伤害(先验告诉它先别动什么)。
- Consumer 轴:v2 天然按 Consumer 分证据;per-channel / TSFM 在同一框架内只换键。

## 8. 诚实的边界

- v2 不保证正结果;它保证每条设计对应一个已测量的失败原因。
- 它把"泛化"部分收窄为"程序 + 模式先验跨域、实体证据同域"——论文措辞相应:自进化 = 程序库 + 模式 Scope + 实体证据三层的
  反馈驱动更新;跨域主张只落在前两层。
- 覆盖率会低于 v1(只部署到已证据化实体);报告覆盖与安全并列,不隐藏。
- 治理与仪器纪律一字不改。
