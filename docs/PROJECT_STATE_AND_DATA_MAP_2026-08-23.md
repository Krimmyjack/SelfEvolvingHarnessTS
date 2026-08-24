# 项目状态、证据与数据使用图（2026-08-23）

本页解决三个容易混淆的问题：项目主体是什么、现有证据到底支持什么、为什么有时
仍需要一个新数据域。它是给人阅读的导航页，不是 Manifest、Evidence Ledger 或
新的冻结/哈希体系。数字来自已提交工件；若本页与原始工件冲突，以工件为准。

长期方法边界以仓库 `AGENTS.md` 为准；当前排期和逐轮裁定分别见
`ROADMAP_POST_V1_2026-08-22.md` 与
`STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md`。

## 1. 项目主体与当前证据位置

项目主体是一个会持续积累并在新域继续校准的 Data Readiness Harness。跨域积累、
transfer 与 Target held-in 适应是同一个时间循环，不是“主线 + 可选增强”的三套
能力：

```text
历史 Domain 的 Action–Response
→ Context-conditioned Skill/Capability 累积
→ 带着累积知识进入新 Target
→ held-in 多轮下游反馈驱动复用、校准、否决、Harness Update 与 Fast replay
→ 冻结
→ held-out Fast-only 部署
→ 外部一次性计分
→ 已打开证据只进入下一代知识，不追溯修改本次结果
```

过去文档使用 L1–L4 只是为了分开核验证据，不代表重要性排序。更准确的读法是四个
缺一不可的证据切面：

| 切面 | 要回答的问题 | 已有证据 | 当前缺口 |
|---|---|---|---|
| Target 校准 | 累积知识进入新域后，held-in 反馈能否把 Workflow/Scope/Risk 调整到本域，并在 held-out 保持 | Forecasting 已有 development 纵向闭环；AD 生命周期与单入口已接通 | AD/新域的严格 held-in→held-out 组件验收，即当前 #42g 的 `Static vs A3` |
| 累积知识贡献 | 完整 Harness 是否比删除历史知识的同构系统更好 | Forecasting pooled 有界正例：首正成本 69 vs 123；终态与 harm 打平 | AD v3 行为不可归因，只关闭该候选；未来仍欠合格 A5 的完整同场验收 |
| Pattern/Context 机制 | 复用与校准是否来自可观察 Pattern×Program×Consumer，而非 Dataset 记忆 | 有多条 development 线索和反例 | 尚无独立、跨 cohort 的共同 Pattern 机制证据；#42e2 只考一个候选线索 |
| Task/Consumer 条件化 | 质量标准反号时，同一 Harness 是否随载体改变行为 | T1b/T3 为注入正控；多 Task 基础设施已接通 | M0 模型轴正控 + M1 同入口能力验收 |

产品形态对应 `A5 = 累积知识 + Target held-in 校准`。`A3` 是删除累积知识的消融，
`Static` 是删除适应的消融。科学实验可以把三者分开以归因，但方法叙事和最终系统
不能把 A3 当主产品、把 A5 降成可有可无的附加项。当前 #42g 因 AD v3 未过行为
验收而只能先做 A3 组件门；它通过仍不等于完整系统收口。

## 2. Held-in / held-out 的唯一语义

- **held-in** 是可多轮使用的反馈适应环境，不是一次性 batch。冻结前可在预先
  规定的总反馈预算和停止规则内反复执行 `Fast → Support/delayed → Episode →
  Slow/审计 → Fast replay`；前轮形成的 Target-local Skill、Risk 与 Harness Patch
  可供后轮继续校准。重复读取同一 Outcome 不算新的独立证据，每次 Consumer
  评估/重训仍计预算。
- **held-out** 是零反馈部署区。进入前冻结 Harness 状态；运行时只走 Fast Path，
  不能调用 `open_delayed`、Slow、Skill 更新或按结果重试。所有臂输出冻结后，
  外部 evaluator 才一次性打开 outcome 计分，且结果不得回流本次 Harness。

历史 `FRESH_A5_DELIVERS` 的 NOAA 2025 是在 fresh 区域上的**反馈消耗式适应**，
不是上述 Fast-only held-out。它支持“累积知识降低适应成本”的有界读数，但不能
替代严格冻结的 held-out 终态验收。

## 3. 当前数据使用与曝光

下表覆盖当前仍影响 Claim 或排期的主要数据面，不试图把仓库历史上的每个临时
fixture 都重新登记一遍。

| 数据面 | 具体规模 | 已怎么使用 | 当前曝光与后续用途 |
|---|---:|---|---|
| Forecasting 配方三 cohort：electricity / T233 / traffic | 每 cohort 12 train + 8 eval，共 3 cohort × 2 Consumer = 6 个主 cell；electricity/T233 为 3 Support + 3 delayed origins，traffic 为 2 + 1 | 配方、掩码、Consumer 条件化、窗口复核、经验冷暖对照 | 全部是 outcome 已曝光的 development 数据；可回归/消融，不再作 fresh Target |
| NOAA global hourly | 20 个站物化并体检，最终锁定 12 train + 4 eval；每站 2024 开发区 8760 槽，2025 确认区索引 `[8760,17520)` | Forecasting 的 `FRESH_A5_DELIVERS` 一次性确认 | 2024 与 2025 均已 `EXPOSED`；只能 replay/development。`beyond_17520` 仍 sealed，但未经新书不得读取 |
| NAB Source：AWS / known_cause / realTraffic / realTweets | 8 + 6 + 7 + 10 = **31 个文件**；4 cohort × 2 轮 × 5 程序 = **40 个 Episode** | AD Source Action–Response census、v1/v2/v3 Source Skill 整合 | Context `INSTANCE_SEEN`，outcome 全部已作为 Source 打开；Source family 已封顶，不再加第 5 cohort |
| NAB AdExchange Target：CPC / CPM | CPC 3 + CPM 3 = **6 个文件** | #42 正式运行及 #42d/#42e0/#42e1 development 重放 | 6 个 Target outcome 均已打开；只能 development/replay，不能再称 fresh 或用于 #42g held-out Claim |
| SMD | **28 台 machine × 38 channel**；official train 合计 **708,405 行**；test header 记录 **708,420 行** | 做过结构/可读性勘察 | train 属 development；official test/labels 保持 sealed。当前单变量协议下属于多变量实体形态不匹配，不是 Harness 方法失败 |
| #42f 候选 Target（Yahoo S5 优先、UCR-AD 备选） | **尚未冻结 roster，不能预报文件数** | 计划用于新域 held-in/held-out Target 校准考场 | 在冻结前先盘点本地已有副本和曝光状态；只有不存在合格 sealed Target 时才下载，并须按任务书/用户授权执行 |

### 为什么“数据已经很多”仍可能需要一个新域

缺的不是开发数据总量。当前已有至少 31 个 NAB Source 文件、6 个 AdExchange
Target 文件、三套 Forecasting cohort、NOAA 和 SMD。真正稀缺的是：

1. 与目标 Consumer/任务语义匹配；
2. 能预先冻结同序列时间 held-in/held-out；
3. held-in outcome 可用于适应；
4. held-out outcome 尚未被任何开发或挑选看过。

现有 NAB Target 和 NOAA 2025 已经打开，不能重复冒充 fresh held-out。#42f 若使用
新域，是为了获得一次干净的最终评价，不是为了继续堆 Source Memory。获取顺序应是
“先盘点本地与曝光 → 再决定是否下载”，不得为了扩大数据量而自动下载。

## 4. Pattern 能力已经探索到哪里

共同 Pattern 不是从 #42e2 才第一次出现，但也尚未得到跨域机制证明。

| 历史切片 | 观察到什么 | 不能推出什么 |
|---|---|---|
| Batch recipe：3 cohort × 2 Consumer | 6/6 cell 有延迟非负方案；冠军程序、掩码和是否处理会随 cohort/Consumer 改变 | 不能推出某个粗 Pattern 已解释这些变化；全部为 development |
| M0a/M0b geometry / mask 诊断 | 加入几何观察会改变可见 Context 和探索；程序半稳定而 mask 高度窗口局部 | 既有几何字段不能稳定预测应排除的 series，不能直接升为 Scope |
| `post_shift_support_sufficient`（pss） | 在两 cohort 时完美区分 AWS/KC，扩到四 cohort 后仍是 known_cause 单 cohort 指示器 | 属 cohort 代理，永久禁止作为跨域 Scope；它的失败只否定这个粗特征 |
| T1b / T3 | 同一训练字节在 Forecasting/AD 上反号，TaskSpec 能改变提案 | 属注入 `POSITIVE_CONTROL`，不等于自然数据 held-out 能力 |
| T233 / Source Skill 线 | 建立了 UNGUIDED/conditioned 溯源、Slow 整合和 Skill 审计；也暴露了生 Episode 直入 Fast 与自确认风险 | 不能把重复/conditioned Episode 当独立共同 Pattern 证据 |
| #42e2 | 冻结一个 `isolated_dominant × winsorize` 候选，做方向、代理和 LOCO 诊断 | 正例目前只来自 AWS 一个 cohort；即使分离也至多是 `PATTERN_CANDIDATE_CLUE`，不能声称跨 cohort 共同 Pattern 已证 |

#42e2 的执行解释固定为：每条 series 独立计算 extreme runs，再用
`sum(isolated runs) / sum(all runs)` 聚合到 episode；不得拼接 series 制造连续段。
MAD 退化沿现役 public-feature 的 finite/MAD-floor 语义处理，不另造 epsilon。承重
响应只读 delayed：`POSITIVE` 为正向，`NEGATIVE/CONFLICT` 为不利，
`NEUTRAL/ABSTAIN` 只描述。合法跨 cohort 结论要求正、负两侧都保有至少两个
cohort，且每个 LOCO fold 都仍有两类；当前正例单 cohort 因而天然限制 claim 上限。
dataset-ID router、标签置换和 mad 附录仅是描述参照，不得改变主判。

## 5. 当前排期与最终系统出口

```text
#42e2  一个 Pattern 候选的非阻塞诊断
→ #42f 冻结新域 roster、时间 held-in/held-out 与信息墙
→ #42g Target 校准组件：Static vs A3，held-in 多轮适应后 frozen held-out 验收
→ #43 M0：只变 Consumer/模型归纳偏置的反号正控
→ #44 M1：同一 Harness 读取 Consumer Context 后自主改变 Workflow
→ #45 轻量 forecasting 回归
→ 在合格累积 Skill 与 sealed Target 同时具备后，完成 Static/A3/A5 三臂同场验收
→ #46 单入口系统整合与历史工程债收口
```

M0 必须保持 Task、数据/split、窗口、Program 作用字节、预算、Metric 和最终评价
目标不变，只改变 Consumer/模型归纳偏置。M1 的 Runner 不得按模型名派答案。

#46 的退出条件不是再写一批实验脚本，而是同一个 operational Harness 入口能：

1. 接收不同 `TaskSpec` 和 Consumer adapter；
2. 共用 Workspace、Typed Workflow、Memory、Risk 与 Skill 生命周期；
3. 在 held-in 真实反馈下适应并冻结；
4. 在 held-out 只走 Fast Path；
5. 不靠 Dataset/Consumer 名称硬编码 Workflow，也不把 raw Source Episode 直塞 Fast；
6. 在自然数据上完成 `Static/A3/A5` 同场验收，分别报告端到端、Target 校准和
   累积知识的贡献；
7. 完成 Consumer 条件化的 M1 承重验收。

某一 Source Skill 或 Pattern family 的负结果会限定可复用知识的边界，但不会把
“跨域积累”从完整 Harness 中删除，也不能把项目重新收缩成 Router、固定清洗器或
实验仪器集合。下一轮应回到最早阻塞的 Observation、Program headroom、Scope/Risk
或反馈分辨率，而不是把完整系统永久改成 A3-only。

## #42i（2026-08-24）—— 契约 wiring + AD-native 程序注册（纯代码）

**CODE_LANDED（纯代码书，不跑行为实验）**

- **Part A — `anomaly_background_model_quality_contract_v1`**
  - 在 `contracts/task.py` 增加 `anomaly_background_model_quality_contract_v1()` 与
    `anomaly_task_context_v1()`；`PRESERVATION_VOCABULARY` 新增 `normal_boundary_fidelity`；
    `HARM_VOCABULARY` 新增 `normal_boundary_shrinkage`、`false_alarm_amplification`；
    `anomaly_events` 词汇常量的注释已固定为“指保护 downstream event discrimination
    所需证据，不表示训练区内任何疑似异常点都禁止删除”。
  - `contracts/schemas/task_quality_contract_v1.json` 与
    `contracts/schemas/task_context_v1.json` 同步三个新词到 enum。
  - **红线**：契约 `to_dict()` 中零 Pattern→Program 字段。
- **Part B（r1 修订）—— `contamination_mask_refit_v1` 不注册为 Operator**
  - 实现为 `aegists_iforest_v1` 的 Consumer-conditioned fit policy：
    `evaluation/functional/consumers/aegists_iforest_v1.py` 新增
    `fit_series_with_contamination_mask` 与 `consumer_id_for(program)`；
    `MASK_REFIT_FRACTION=0.01`、`CONTAMINATION_MASK_REFIT_ID` 已固定。
  - 遮罩单位固定为训练窗口；初次 fit → 排除 anomaly score 最高的 ≤1% 窗口 →
    refit；v1 不做点级反投影、邻域延伸、NaN 或删行；原始数据与 Query 不改；
    一次执行计 2 fits；标准化常数跨两次 fit 字节一致。
  - `operators/registry.py` 未触碰；Fast Agent 菜单未扩展。
- **Part C —— 六程序 census 仪器**
  - `evaluation/functional/run_e2_t6_natural_a5_a3.py` 新增
    `PROGRAMS_V2 = PROGRAMS + ("contamination_mask_refit_v1",)` 与
    `NON_IDENTITY_V2`；`menu_headroom_v1(menu_size=5)` 与
    `feedback_unit_v1(menu_size=5)` 仪器参数化（U0 锚定 `feedback_unit_v1`
    锁在 `menu_size=5`，拒绝其它 size）；六程序模式仅 fixture 烟测，
    Yahoo 六程序仍由 #42j 守门。
  - `methods/`（Fast/Slow/Skill 生命周期）零改动。

**门**：#42j 证明六程序 Yahoo 安全 headroom 后方可正式注册为 Operator 或纳入 Fast 菜单；
失败不得改称弃权能力正结果。

**交付**：代码 + 测试已 commit（本书例外于“交付不 commit”）。
