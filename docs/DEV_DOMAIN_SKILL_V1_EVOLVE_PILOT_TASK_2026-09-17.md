# DEV-DOMAIN-SKILL-V1-EVOLVE-PILOT

日期：2026-09-17。Planner：Astra；开发与执行负责人：Opus。
状态：执行规格已定稿，供用户转交执行；本文件写入时未开跑。
前包：DEV-DOMAIN-SKILL-V1-BUILD，BUILD_COMPLETE；不重复验收 A 包，不覆盖其工件。
输出根：`_scratch/dev_domain_skill_v1_evolve_pilot/`，其下按 source / formation / target 分阶段。

## 1. 本包决定及终点

采用最短可解释路线：**已知 domain → 离线生成竞争 Workflow → 真实选择 → 冻结复用**。
本包只回答：同域历史形成并选出的指导，能否改善后续批次 Fast 的材料构造、交付或完整成本？
先不把模型特征匹配同时放进主比较。known-domain 是可用域标签下的参照设置，不是特征泛化证据。

- D01 = Electricity；D02 = Traffic。运行时配置存源名，Skill 正文不存源名、实体 ID 或历史作业答案。
- 内部角色只有 Slow / Skill Optimizer 与 Fast / Target Agent；Opus 是开发执行者，不是方法内的第三个 Agent。
- Skill 是可执行指导：观察什么、提出什么假设、怎样构造与比较、如何决定下一步和停止；不是固定完整配方的改名。
- 沿用共享 MLP、三算子及现有参数/条件化 DSL、7 个工具、归一化 MSE 和 C_A/commit/C_B/E 边界。
- 不追加算子、Consumer、噪声估计平台、逐实体价值标签或新 Router；不再另开只读诊断包。
- 按本包连续完成必要实现、真实形成、选优、四个后续作业、分析。不得按正负结果中途扩预算或换作业。

设计参考 [Eval-Skill §3](https://arxiv.org/html/2606.07040v2)：复用其每域 Skill、竞争 Workflow 与离线选择的思路。
本包是时序训练材料任务上的小规模迁移试点；不是原论文完整实验规模的复现，也不声称模型画像 Router 来自该论文主方法。
Principles 可随首次候选产生，但本包不另做第二轮增补/交叉/演化，避免增加一次独立选优课程。

## 2. 数据、人口与时间冻结

两域均使用 `spec.DATASETS[dataset]['roster']` 的现役默认 32 列，顺序原样保留，不改成数值排序。
L=192，H=48，T 长 672。以 `context.resolve_job` / `spec.job_spec` 为执行真源。
所有材料均标 `EXPOSED_DEVELOPMENT`；Electricity 多个窗口已被历史使用，不把这次新 Skill 运行包装成 fresh 测试。
以下批次按统一时间序列选择，未按本次未知收益筛组。

| 阶段 | D01 | D02 | 权限与作用 |
|---|---|---|---|
| Source | a55、a65：复用指定历史分支 | a55、a65：各补 1 条 no_skill Fast 轨迹 | T、C_A、commit 后 C_B；不读/生成本包 Source E |
| Select | a75，1 个完整批次 | a75，1 个完整批次 | NO_SKILL 与最多 W1/W2 各跑 Fast；以各自交付的 C_B 选 Skill |
| Target | a85、a95 | a85、a95 | 冻结同一张域卡，独立 Fast；C_A 可见；四作业全部收口后开 C_B/E |

| 域 | a55 t | a65 t | a75 t | a85 t | a95 t |
|---|---:|---:|---:|---:|---:|
| D01 | 14448 | 17088 | 19728 | 22344 | 24984 |
| D02 | 9648 | 11400 | 13152 | 14904 | 16656 |

各作业 T=[t−672,t)，C_A 真值=[t,t+96)，C_B 真值=[t+96,t+192)，E 真值=[t+192,t+384)。
冻结配置保存实际 origins、roster 和 dataset，不新增 hash。Source 最晚工作量边界为 D01 17472、D02 11784。
Source、Select、Target 各阶段时间段不重叠；具体几何须在首个付费调用前由现役函数核对。
预检只读冻结作业的 T、表头和现有语义记录；不得先跑 C_A 筛哪批好做。
T 资格不通过时只记该作业 ELIGIBILITY_FAILED，不换人口/截止点；不受影响的域继续。
两个 Target 批次的 T 可供运行时资格检查，但不得进入 Source census 或 Slow 形成输入。

本包新拟合统一种子：`[20260922, 20260923, 20260924]`。
同一作业各臂对齐模型初始化、父 batch 和既有增强随机流；重复时不重新生成配方。
D01 历史 Source 原种子和原预算不重写，明确它们不是本包同配置重复。

## 3. Source 与候选 Workflow

D01 的唯一白名单（不读取整个历史目录的结果报告）：

```text
(dev_batch_research_workflow_v1, a55_H0, a55)
(dev_batch_research_workflow_v1, a65_parent, a65)
```

这两条是已有无 Skill 过程轨迹。只使用 T 观察、实际工具/材料/候选、C_A、commit、成本与 post-commit C_B。
原目录中的 E 文件保持隔离，Source 文字不引用它们。其他 package、a65_candidate 和旧算子效果包不进入本包 census。

D02 在本包 source 子目录实际运行 a55、a65，各一个 no_skill 臂：
- 同样的工具、2 个新候选槽、3 个训练种子，None/Fixed 为公共基线。
- 两条 commit 都冻结后再读取其 C_B，形成真实过程材料；不运行 Source E 阶段。
- 两条均完整才为 D02 形成 Skill；技术失败不得伪装成 KEEP/空成功轨迹。
- 不用 Electricity 轨迹改名补齐 Traffic，也不把历史算子效果当作过程轨迹。

每域一次 Slow 形成，允许输出 1–2 个竞争 Workflow 或 KEEP；仅一次既有契约纠错，不按内容“不够好”重抽。
复用 A 包 `propose` 与 ≤1200 字符载体；各候选沿用其真实 scope，`const:true` 合法，null 不合法。
两个研究方式不同不要求两种不同算子，不要求激进、多样或强制观察。
额外证据摘要只允许确定性 census；Fast 不读取 raw Source、完整 census、另域轨迹或选择集成绩。

## 4. 真实选优及冻结

每域仅在 a75 跑 `NO_SKILL / W1 / W2`，各自独立会话、同等预算、共同 None/Fixed 基线。
候选不足两个不补抽；KEEP 则直接记无候选，不为了填表再启动空选择课程。
候选测试用 `CANDIDATE_TEST_ONLY` 的已有选择入口；不能提前伪造 FROZEN_SELECTED 才让它运行。

**选择块固定 C_B。** Fast 在 C_A 上自主研究和 commit；先完成该域所有选项的 commit，再给执行器读取 C_B。
现有 `selection_runner` 在单选项返回前直接开 C_B 的做法改为同一阶段先运行、后评分；不新建另一套 Runner。
Slow 不读取 a75 分数后修改候选；执行者也不看分数后修改提示词。Select 不打开 E。

选择规则沿用 A 包：各选项“自己真正交付”的 C_B 三种子均值最小；完全平局按 NO_SKILL、W1、W2 顺序。
不以该臂集合中事后 C_B 最好的材料代替其实际 commit。
无 Skill 参照未完成 → SELECTION_INCOMPLETE；未完成候选不参选。
NO_SKILL 最优 → NO_EFFECTIVE_CANDIDATE / NOT_PROMOTED；不偷偷保留第二名当合格卡。
候选最优 → 本包 FROZEN_SELECTED，准确含义仅为“开发选择胜出”，不是统计显著或通用能力获证。

两个域均完成形成/选择或明确失败状态后，再创建 Target 冻结目录。
B 包建立自己的 catalog，只保存本包真实选择状态及 CONTROL；不得将 A 包 fixture 改状态后复用。各域参照画像由 a55/a65 的真实 T 计算，沿用现有字段与汇总形式，供后续匹配包使用；a75/a85/a95 不混入参照画像。此处不新增特征或 Router 调用。
- 卡、Generic 正文、Target 顺序与配置在第一次 Target 拟合前冻结。
- 同域 a85/a95 共用同一卡，不做 a85→a95 更新，不把 a85 的 C_B/E 回流。
- 无选中卡的域，known_domain 记 NO_TREATMENT、不伪造一个 no_skill 重跑充当 Skill 效果；其他对照仍完成。
- 若两域都无卡，明确形成/选择尚未成功，剩余对照只算 Fast 基线结果；不追加候选或重跑选优。

## 5. Target 对照和顺序

四个 Target Job，每个包含以下分支，None/Fixed 共用真实基线模型。

| 分支 | 实际做什么 | 主要回答 |
|---|---|---|
| no_skill | 公共起点、完整 Fast、无指导卡 | 同一个 Agent 不给 Skill 的表现 |
| generic | 同 Fast + 已有冻结 GENERIC_TEXT | 特定历史形成的内容是否超过普通研究建议 |
| known_domain | 同 Fast + 该域选中的冻结 Workflow | 本包主要方法分支 |
| random | 同 DSL 随机提出两个完整方案，真实评估后按 C_A 选交付 | 等拟合预算的零 LLM 搜索对照 |
| None / FixedMixup | 直接使用公共基线 | 处理收益与研究增量分开 |

Generic 精确复用 `batch_research_source_process.GENERIC_TEXT`，首个 Source 调用前保存正文。
它是手工冻结的 CONTROL，不是学得/选优成功的域卡；可直接复用原 Generic Knowledge 构造，不伪造选择记录。
所有 Fast 均：`max_calls=16, max_tools=24, max_new_evaluations=2`；既有工具契约纠错上限 2，调用照常计费。
`evidence_roundtrip=true`，使用 A 包现役公共工具说明；三个 Fast 臂只有指导不同，不附加硬配额、强制串行或提交门。
研究 Agent 可直接复用、继续探索或停止；其提交不被 C_B argmin 覆写。

顺序冻结：

| Job | 分支执行顺序 | Random 两个 policy seed |
|---|---|---|
| a85_D01 | no_skill, generic, known_domain, random | 2026091701, 2026091702 |
| a85_D02 | generic, known_domain, no_skill, random | 2026091703, 2026091704 |
| a95_D01 | known_domain, no_skill, generic, random | 2026091705, 2026091706 |
| a95_D02 | no_skill, known_domain, generic, random | 2026091707, 2026091708 |

Random 复用 `policy.random_policy` 和现有 Random 分支语义，不按 LLM 结果重抽；生成两个逻辑候选，别名如实记录，不无限去重。
Random 的 C_A 最小值平局顺序沿用原实现：None、FixedMixup、p1、p2。
公共基线跨臂共享；私有候选/分数不得泄露给其他臂。完全同材料按既有缓存规则复用时记录真实拟合数与逻辑数。
两域 Fast 可见中性 Job ID，且同一 Job 各臂一致；本包不声称隐域匹配，无须为此另建身份隐藏系统。

## 6. 本包必要实现范围

在既有 domain_skill study 和 roundtrip 上增量实现，不复制新的实验框架。
1. 为实际使用的 Random 路径贯通 `dataset`、roster、种子、commit 和标签阶段；旧默认仍为 electricity。Menu 未使用，不顺手改全库。
2. domain_skill 的 study 臂枚举/调度容纳 Random；Random 不进入 Skill 路由枚举，也不调用 Router。
3. Generic 使用明确 CONTROL 来源；域卡继续只接受真实选择状态，fixture 永不进入 live。
4. Source 与 Select 可只运行到 C_B；不为了通过 live preflight 而给这两个阶段 E 权限。
5. Select 的 C_B 评分晚于该域所有候选 commit；Target 全部 commit 晚于两域卡冻结，且早于任何本包 Target C_B/E。
6. 形成入口目前为 propose/select 分别建账本，必须对**全包**落实 §7 总限额。可复用同一 RuntimeLedger，或用总额不超帽的冻结分账；不得每入口获得一次完整总预算。

只补一组针对这些差集的集成 smoke，并运行已有 controller tests 与 domain_skill 控制 smoke。
已知 measured_start 历史 fault-smoke 失败不在本包修复范围；不反复全矩阵测试或覆盖 A 包报告。
配置/模型身份/数据绑定验证通过后连续开跑，无需每个函数修补再请用户确认。

## 7. 预算和成本

公共 None/Fixed：每 Job 2×3=6 拟合。每个有两个新方案的分支：最多 2×3=6 新拟合。
预算按“所有合法候选与分支都跑满”计算，不依赖别名、提前停止或失败节省。

| 阶段 | 拟合计划上限 | LLM 逻辑请求上限 | token 分配上限 |
|---|---:|---:|---:|
| Traffic Source：2×(6+6) | 24 | 32 | 800,000 |
| 两域形成+Select：2×(6+3×6) | 48 | 100（含两域 Slow 各最多2次） | 2,000,000 |
| 四个 Target：4×(6+3×6+6) | 120 | 192 | 3,200,000 |
| 合计 | **192** | **324** | **6,000,000** |

- 物理拟合尝试硬帽 **194**；全包仅 2 次同配置非科学失败重试，不能用于新种子/候选。
- HTTP 尝试 ≤648；每逻辑请求沿用既有最多一次合规传输重传，不另加隐藏探活请求。
- 全包活跃运行墙钟 **28,800 s（8 h）**；阶段重启不得重置起点或总余额。
- 有效 token 含输入与输出；分账合计不得超 6M。未知 usage 另列，不能把预留估计写成已知账单或保证上界。
- 无法在余额内启动完整分支时记 BUDGET_INCOMPLETE，不缩种子/悄改每臂预算补齐表格。
- Source/Slow/Select 为一次性形成成本；Target Fast、拟合、后处理分别计费。不能只报 Fast 省的钱。
- 模型固定沿用客户端：请求 `cpa-grok-4.6`、返回必须 `grok-4.6-build`、temperature=0；使用现有本地中转和安全配置，凭据不写文档/日志。
- 当前按历史 22–36 s/拟合估计，192 次约 70–115 分钟纯拟合；加调用、接线与收口预计约 4–6 小时，非完成时长保证。

尽量单一账本。若使用阶段分账，冻结后不得因结果不好转移额度；故障重试两格仍由全包计数共同约束。
阶段挂起前的已花成本保留；不得通过另建相同实验目录使未知 usage 或超帽状态失效。

## 8. 信息墙、失败与恢复

- Source/Select 不触及自己的 E；Target 全部计划分支完成或被明确判定终止后，统一 C_B → 冻结全部 E 预测 → 统一 E 评分。
- 不因 a85 完成便提前开标签；不把中断的自动报告当 finalize 权限。
- 复用已修复 labels-withheld/resume 边界；任何本阶段禁止标签已开，禁止恢复其自适应调用。
- 传输/账户/预算故障不改写成模型 KEEP、ABSTAIN 或有效基线交付；模型真实的 KEEP 独立记录。
- 未知 usage 仍按冻结规则暂停后续收费调用，只有用户明确接受未知费用且标签边界仍合法才恢复原请求。本文不预授权未知费用。
- 模型身份不符、权限泄漏、人口或训练几何冲突、需要超总预算时停止受影响阶段，说明实质原因；其余普通实现自行处理。
- 0 新 SHA/Hash，0 git commit；不覆盖旧实验目录，不修改其他 worktree、参考仓库或 AGENTS.md。

## 9. 读数和结论

每 Target Job 报真实交付的 3 个 seed E 宏 NMSE、配对差与 SE；不集成、不挑 seed。
主比较 `Δ_skill = E_no_skill − E_known_domain`，正值表示 Skill 更好。
同时报 Generic−Skill、Random−Skill、Fixed−Skill、None−Skill。
四个 Job 逐项展示；同域两个批次单列，不把 3 个 seed 当三个独立域。
若需单一摘要，预定为四个完整配对 Job 的相对改善等权均值；缺配对时标明实际分母，不能偷偷只留有卡/正向的域。
小样本 SE 只描述固定数据与材料下的训练随机性；不宣称控制了未来错选率。

行为证据至少回答：
1. Slow 两个候选的研究方式具体不同在哪里？实际选择了哪张，还是 NO_SKILL？
2. 卡实际加载次数；观察、实验顺序、候选材料、commit 哪些变化？只改说法不能算策略改善。
3. 交付是否改善；与 Generic、随机搜索相比是否仍有增量？
4. 增量能否抵消形成与使用成本？只减调用不自动等于质量提高。

固定判读：
- 只胜 None/Fixed → 处理或搜索收益，不自动是 Skill 收益。
- 与无 Skill 同材料同交付 → 本设置无可见 Skill 增量；不是所有 Skill 无效。
- 胜无 Skill但不胜 Generic/Random → 局部组件读数，不能声称域知识或 LLM 搜索独占价值。
- 多个后续批次出现一致、具有实际幅度的改善 → 有开发级候选信号，下一包在未参与设计的批次检验；不立即封为通用能力。
- 只有一域/一批好 → 如实限定范围，不把另一域删掉。
- 无卡/技术失败/无加载/加载后无效分开，不强求正号。

## 10. 交付与后续

一个主 REPORT.md，加现役机器可读结果、冻结配置、census/候选/选择/卡、真实请求/轨迹/材料/模型/预测及成本。
首页给：是否形成可用卡；每域选择；四 Job 配对结果；完整成本；执行完成与协议合规分别判定。
报告补明：D01 历史 Source 与 D02 新 Source 形成成本不对称；每域仅两个源作业、一个选择作业，远小于论文实验规模。
任务书末尾追加实际收口，不写入尚未发生的结果。

完成本包即停止，不自动启动匹配或再次演化。后续安排：
- 至少一域卡有可信行为/效用线索：冻结本包卡，下一包比较 known-domain 与 profile-match（并保留 no_skill），使用不同批次；单独计匹配损失与卡本身效用。
- 卡未形成或效用未改善：优先根据真实失败选择一个 Workflow 改进，不增加 Router 来掩盖卡质量；不重开全面噪声/算子普查。
- 两域卡正文相同且行为相同：报告“形成了通用指导”，不包装成域特化；后续才检验是否需要多个域卡。

方法方向已在本任务书选定。执行者遇普通接口问题可按上述范围修复并继续，不再把五个设计选项退回用户逐项选择。

## 11. 实际收口（2026-09-17，执行者 Opus 追加）

主报告：`_scratch/dev_domain_skill_v1_evolve_pilot/REPORT.md`；机器可读结果：`pilot_result.json`。

- **执行完成**：COMPLETE。Source、Select、Target 三阶段均为 FINISHED，Target 16/16 分支 COMPLETE。
- **协议合规**：合规。Source/Select 无 E 文件；Target 按 commit → C_B → E 冻结 → E 评分顺序执行；卡、Generic、catalog 与配置均在第一次 Target 拟合前冻结；返回模型全部为 grok-4.6-build；0 未知 usage、0 重发、0 拟合失败、0 重试；0 新 SHA，0 git 提交。
- **成本**：177 次拟合（上限 194）、108 次请求（上限 324）、4.42M token（上限 6M）、10,442 s（上限 28,800 s）。
  - Source：24 次拟合、11 次请求、0.44M token；形成与 Select：45 次拟合、33 次请求、1.39M token；Target：108 次拟合、64 次请求、2.59M token。
- **形成与选择**：每域 Slow 一次合法输出两个候选。两域均选中 W1（a75 交付 C_B：D01 为 .5341，对 NO_SKILL .5402；D02 为 .5066，对 NO_SKILL .5145），冻结为 FROZEN_SELECTED。
  - D01 卡：均匀 TimeMixup 权重搜索；D02 卡：单次 FreqMask 探测，未过门槛则回退 FixedMixup。
- **Target 主比较** Δ_skill = E(no_skill) − E(known_domain)：a85_D01 +0.0895（SE .0549）、a95_D01 +0.0045（SE .0204）、a85_D02 +0.0013（SE .0059）、a95_D02 0（交付同一材料）。等权相对改善 +2.44%，由 a85_D01 主导。
  - Generic − Skill：−.0249、+.0013、+.0310、−.0066；Random − Skill：+.1227、0、+.0310、0。
- **固定判读**：
  - D01：胜无 Skill、Random、Fixed，对 Generic 不一致 → 局部组件读数；
  - D02：与无 Skill 同交付或差距在 SE 内，并输给 Generic → 本设置无可见 Skill 效用增量，只降低使用成本；
  - 多批一致、有实际幅度的改善未成立。
- 本包完成即停止，未启动匹配包或再次演化。
