# DEV-TEMPO-AUG-WORKFLOW-SKILL：同域增强 Workflow 学习与后续批次检验

日期：2026-09-18
状态：**Planner 执行规格已定，尚未启动。用户转发本任务书并要求执行后，Opus 在下列边界内连续完成实现、检查、真实实验与统一报告；普通实现细节无需逐项请示。**
文件沿用执行者草案的原路径（文件名中的 DRAFT 保留以免断开既有链接）；本正文替代该草案的待裁定项。本任务书不宣称付费运行已经获执行者接收或已经启动。

## 0. 目标、方法身份与本包终点

**目标：在 RD02 开发研究经历上形成域级 Workflow，冻结后交给同域后续批次的 Fast，检验其相对无 Skill、同预算随机搜索及简单固定策略的预测收益与完整成本。**

方法参考为 Eval-Skill（arXiv:2606.07040v2，§3–4、附录 B）：
https://arxiv.org/html/2606.07040v2
借鉴域内开发/测试、竞争工作流、按实际任务表现选优、冻结后复用。原文还有先 Workflow 后 Principles、多轮局部提炼和合并；**本包只实现其缩小规模的 Workflow 生成—选择部分，不称完整复现或持续进化通过**。TempoPFN 只提供增强原语，不是 Skill 学习方法。

- 单个研究案例 = 一批 12 个站点的数据 + 一次完整 Fast 研究轨迹；不是单个训练窗口。
- Workflow 应指导观察、假设、材料构造、实验比较、停止与提交；允许简单有效的统一处理，不能以多步、多样性或更多观察证明成功。
- **不增加“无卡 Fast 先显著胜出”或“新算子先显著胜过 Fixed”的开跑门。** 已有工具与开发效应足以开展本方法检验。
- Opus 是项目开发执行者；实验内 Fast/Slow 是实际 API 调用，不得以 Opus 手写的卡或决策替代。
- 本包只做已知域加载，不做画像 Router、跨域选卡、Principles 修订、TSFM、Consumer 调参或新增算子扫描。
- 完成下列一包并停止。结果可正、负、混合或不确定，不因未见正号追加实验。
- 项目 AGENTS.md（尤其 §5.9、§7–9）仍生效；本包是当前用户指定的域内组件实验，不替代完整项目的最终系统验收。

## 1. 唯一运行目录与复用边界

新运行目录：`_scratch/dev_tempo_aug_workflow_skill/`。
建议唯一 study 入口：`evaluation/main_protocol_p4/batch_research_tempo_aug_workflow_skill.py`，复用既有 roundtrip/runtime，不建第二套框架。

读取：
- `methods/ttha/batch_base/tempo_aug.py`、`tempo_source.py`；
- `evaluation/main_protocol_p4/batch_research_tempo_aug_source_alignment.py`；
- `methods/ttha/domain_skill.py` 与现有 Source/Select/Target、恢复及账本实现；
- `_scratch/dev_tempo_aug_source_alignment/METHOD.md`；
- `_scratch/dev_tempo_aug_source_alignment/data_availability/availability.md` 及同目录 JSON。

只修改本任务的新入口与不可避免的显式 opt-in 接线；保留旧默认路径。历史运行包和参考仓库只读。
不改 AGENTS.md，不重跑历史研究，不写他线文件，不 git commit，新增 SHA/Hash 预算 0。

## 2. 数据与两层划分

数据：RD02，`readiness.DATASETS["RD02"]` 的 PM2.5，既有排序的全部 12 个站点，不换人口。
每 Job 的 t 是 T 的右端点，不是比例：
- T = [t-672, t)；
- C_A 的预测 origins = t、t+48，目标 [t,t+96)；
- C_B origins = t+96、t+144，目标 [t+96,t+192)；
- E origins = t+192、t+240、t+288、t+336，目标 [t+192,t+384)。
每 origin 用其之前 192 小时作预测输入，预测 48 小时；推理时读取已过去的输入不等于把未来目标给 Fast。

| 阶段 | Job | t | 材料 RNG 的 job_index |
|---|---|---:|---:|
| Source | RD02_S1 | 3360 | 1 |
| Source | RD02_S2 | 4560 | 2 |
| Source | RD02_V1 | 5760 | 3 |
| Source | RD02_T2 | 8160 | 4 |
| Select | RD02_T4 | 10560 | 5 |
| Select | RD02_Q2 | 12960 | 6 |
| Target | RD02_L1 | 15360 | 7 |
| Target | RD02_L2 | 16560 | 8 |
| Target | RD02_L3 | 18960 | 9 |
| Target | RD02_L4 | 20160 | 10 |

- 接线验收单独用已曝光 RD02_T1，t=6960，job_index=0。
- Source/Select 均为已曝光 development；本包不读取其 E，不把以前已有 E 搬进 formation。
- 四个 Target 按已有曝光清单，其 E 没有已知评分；统一称“未评分的后续开发批次”，不是最终密封验收。不以 T 曾被使用就单独判定未来 E 泄漏，也不宣称已证明任务独立。
- t=21360、22560、24480 保留，本包不读。RD01B 的 t=10512 不纳入，不做 E-only 补充。
- PRSA 2016 年起（row>=24864）、其他 Final 均不开放。2016 年是否作为最终验收，留后续任务另定。
- 不扩展曝光普查。复核既有元数据和本表绑定即可；如发现具体冲突，停受影响 Job，不默换备用。
- T 资格沿用现役：每实体至少 336 个有限值、std>1e-6、至少 32 个合法父对。父 X 至少 32 个有限点，父 y 全部有限。C_A/C_B/E 沿用每实体每块 25% 观测覆盖要求。未来资格只可按既有缺失掩码接口读取计数，不向 Agent 输出真值。
- 若发生不可评分或技术未完成，保留状态；不删实体、改阈值、换时间或以基线伪装成功。

## 3. Consumer、动作空间与材料定义

### 3.1 全臂共同冻结

沿用 source-alignment 的实际环境与 Consumer：共享 MLP、AdamW、2000 次更新、batch=64、同一合法父对池、同一 T scaler、同一评分。
每候选汇总全部实体材料共同训练；不得拼接不同候选模型的逐实体预测。

- 训练 seed：所有十个正式 Job 均为 **[2026091801, 2026091802, 2026091803]**；同 Job 全臂配对。接线使用 RD02_T1 的历史 [20261001,20261002,20261003]，用于 None 复现。
- 父窗口为 [192 点 Linear 补全 X；48 点原始观测 y]，长度 240，T 内冻结归一化。
- 新增强作用于整段父窗口，生成一个子视图；父样本不变，父/子损失各 0.5。全空方案必须走原 None 路径。
- 服务输入统一 Linear，评分真值始终原始观测；不对 C_A/C_B/E 真值增强。
- 保留缺失感知 normalized MSE 原实现；不能改聚合或删极端 origin。
- 这不是新学的样本权重：不开放 downweight/drop、采样比例、训练时长或 loss 修改。

### 3.2 公共动作空间

7 个已对齐原语：tp_regime、tp_shock、tp_calendar、tp_amplitude、tp_resample、tp_censor、tp_random_conv；参数分布、源码特殊行为沿用 METHOD.md。
显式组合最多 3 步、不得重复、固定执行顺序与互斥规则不变。P_NoMixRecipe 单独构成程序。
共 53 个合法程序：空程序 + 51 个显式组合 + P_NoMixRecipe。
Fast 可按现有公开 T 特征谓词构造默认处理和实体例外，最多 8 条规则；允许全批统一。
不开放新 Mixup、donor、原语源码编辑、任意 Python 或自定参数分布。旧 FixedMixup 仅作公共参照。

四个公共已评估参照：
1. None；
2. FixedMixup（旧 R donor、w=0.25 定义）；
3. P_AmpResample（tp_amplitude → tp_resample）；
4. P_NoMixRecipe（已冻结适配配方）。

四者同样给所有 Fast/Random 提供 C_A，可直接提交，不占四个新方案额度，但其真实拟合成本必须计入。

### 3.3 跨臂材料一致性：仅本包启用新 RNG

消除按候选构造次序分配随机种子的绑定。
程序编号表在首次真实材料生成前冻结：空程序为 0；按长度 1/2/3、PRIMITIVES 原顺序枚举合法组合并规范化，得到 1..51；配方为 52。
每实体每程序的随机种子：
`int(np.random.SeedSequence([2026091800, job_index, program_index, entity_index]).generate_state(1, dtype=np.uint32)[0])`。
实体编号取冻结 roster 内的位置；每个实体内部父窗口顺序固定。Streams 的 NumPy/torch 状态隔离保持。

- 同 Job、同实体、同程序，跨臂、跨构造顺序的子窗口必须逐位相同。
- 只在首次用到 (实体,程序) 时构造；不提前生成全 53 个程序的材料。
- 相同完整 assignment 是一个材料，改名字或文字说明不能重抽。
- 新 RNG 是新材料版本，例如 `tempo_source_v2_entity_program`；缓存绑定必须区分旧版本、roster、T/scaler/父对、assignment、训练 seed/配置及服务几何，不加哈希。
- **旧 AmpResample/NoMixRecipe 模型不能因 Job 或名字相同就复用**，即便旧 Job 也一样。None/FixedMixup 只有语义与数组/环境绑定确认相同时才能复用。
- 原样保留旧随机定义供旧包重放，不覆盖旧材料或结果。

## 4. 接线与一次最小验收

须实际接通：build 新组合 → evaluate 真实共同训练 → compare → commit → 动态 C_B/E 评分，不能只回放已训练的两份预设。

1. 去除新路径中包级 RUN / 旧 P.cell_path 硬绑定；adapter、worker、评分绑定本分支目录。跨臂共享物理缓存可用包内独立目录，分支只持有自己可见的引用。
2. `allow_fit=True` 接入真实 worker、候选额度和包级账本。先预留预算再拟合；失败计费。
3. 动态收集实际已评估材料及提交，不再仅遍历 NEW 两预设。未评估候选不能提交或送进主效果表。
4. 四个公共基线进入实际 Fast 初始请求。overview 的 loss/训练说明改为父子视图，不能残留 readiness 的“无子视图”描述。
5. 缓存复用只返回本臂请求的候选；不暴露其他臂的候选、轨迹、C_A 或提交。跨臂命中一个新候选仍消耗本臂一个逻辑评估槽。
6. 使用一个包账本贯穿阶段；恢复不重置、不重新调用已完成 Slow、不重新抽随机材料。

合成 smoke 只覆盖本次变化：目录/新候选评分、跨臂跨顺序 RNG、逻辑预算、全局标签屏障、空 Skill 与实际注入。复用既有 controller tests 及最相关原 study smoke，不扩测试矩阵。

真实接线验收仅 RD02_T1，0 LLM，最多 5 次拟合：
- 一个非公共基线的新组合真实训练三 seed，经七工具 compare/commit；
- 第二分支按不同构造顺序生成同组合，检查材料逐位一致并复用拟合；
- 在两个脚本分支都结束后打开 C_B、冻结全部 E 预测、评分，确认新组合被收录；
- 重训该组合一个 seed 核对权重；重训 None 一个历史 seed 与历史缓存核对。
这 5 次均计预算，不把脚本行为记作 Agent 行为。

环境沿用上包实际依赖指纹，不为匹配环境名称重装。worker 继续显式处理 OMP 初始化顺序；不使用 KMP_DUPLICATE_LIB_OK 掩盖冲突。接线通过即进入 Source，不追加效果预检。

## 5. Source → Workflow 形成 → Select

### 5.1 Source

四个 Source Job 各跑一条无 Skill Fast：四公共参照 + 最多四个新完整方案，工具/提示/预算结构与 Target 一致。
全部 Source 分支完成或按协议记失败后，统一打开其 C_B；不产生 Source E。
形成输入只来自本包 Source，不从旧修复卡或旧增强 E 总结搬答案。

deterministic census 保留：T 全批特征与实际检查、工具顺序、完整方案/assignment、材料诊断、逐 seed/origin C_A/C_B、commit 理由、状态、成本与可定位 evidence_refs。
允许压缩重复长数组和叙述，不能丢行动、候选、失败、符号或数值。压缩规则在首次 Slow 调用前确定并保存，不按期望结论剪裁。
Source 某分支未完成不伪造经历；没有任何完整可用经历则形成记 INCOMPLETE，不反复抽 Source。

### 5.2 Slow：本包 Workflow-only

一次科学形成调用，合法输出 PROPOSE（1–2 张 W1/W2）或 KEEP。只允许一次格式/合同纠正，不因建议保守、相似、不喜欢或效果不好重问。
沿用 domain_skill 的 experiment_guidance 载体和 1200 字符正文上限，principles=null；scope 固定 const:true，适用情境用 Workflow 内的可观察条件表达。本包不额外引入路由拒绝这一变量。

输入明确：
- 目标是改善同域新任务中整批预测收益与研究成本；
- 学习如何观察、构造对照、利用反馈和停止；两个候选可采用不同调查顺序或实验分配；
- 可以提出有证据边界的算子倾向，也可以保留基线。**不禁止模型自然学到简单统一策略**，但须如实记录它是否只形成固定菜单；
- 不把“未分辨”写成“某算子普遍更差”；不将逐实体预测差写成其材料因果贡献；
- 不写数据源名、Job/实体 ID、行号答案表，不直接交付历史 assignment；
- 不修改 Consumer、权限、工具、预算、RNG 或评分。

前包 RD02_STP1 的 E、峰值诊断、“混合类优先”等研究者事后结论均不进入提示。
可提供共同的中性提醒：不同评价时间块的排名不保证一致；不得偷偷指定哪个原语应赢。

### 5.3 Select：真实比较无 Skill、W1、W2

两个 Select Job（T4、Q2）都跑相同选项：无 Skill 与实际生成的每张候选。候选只有一张就不补造第二张。
每分支四公共基线、四新候选额度；原生 commit，不由 argmin 改写。
顺序：T4 = 无卡,W1,W2；Q2 = W2,W1,无卡，缺项跳过。
全部计划 Select 分支结束后才统一开 C_B；不产生 E，不在中途改卡。

选项 v 的选优值：
`J(v) = mean_over_two_jobs( mean_seed C_B(committed_v) / mean_seed C_B(None) )`。
只允许在两个 Job 均完整可评分的选项入选；分母非正或不可评分时记 SELECTION_INCOMPLETE，不另换评分口径。
J 越小越好；差 <=1e-12 的数值平局顺序：无 Skill、W1、W2。不额外要求显著性门，不看 E 或“故事更合理”人工选卡。

- 无 Skill 赢、Slow KEEP 或无合法候选：冻结空 Skill，标 NO_TREATMENT，不加载 Generic 代替。
- 无完整选优基础：冻结空 Skill，标 SELECTION_INCOMPLETE，不能宣称成功学卡。
- 若空 Skill 与无卡的全部实际请求条件一致，Target 复用无卡轨迹和交付，标 EMPTY_SKILL_ALIAS；不浪费第二次 LLM，不冒充独立重复。
- 保存候选原文、实际注入正文、各开发交付与选优值；选中不是晋升为已验证 Skill。

## 6. Target 比较与随机控制

每个 Target Job 都有：
- F_noSkill：空知识；
- F_domainSkill：已选中的同一冻结 Workflow；
- RandomSearch_B4：0 LLM，同样四个新候选额度；
- 四个固定参照；
- Menu_CA：四公共参照中按 C_A 三 seed 均值 argmin，0 额外拟合/LLM。

所有研究臂最多评估 4 个不同的新完整 assignment；公共基线及本臂已评估别名不再消耗新槽。允许提前提交。每个 Fast 最多 16 次模型调用、24 次工具调用，400,000 总 token。build/inspect 的 wall 与工具调用仍计费；不给无限草案生成。
F_noSkill/F_domainSkill 的基础系统提示、工具、数据、额度及模型参数完全一致，唯一知识差异是冻结正文。不规定必须复杂、不强制全用额度、不强制至少使用某原语。
Fast 可选择任何本臂已评估候选（含四基线），无需机械 argmin；必须保留真实理由。正常输入格式错误可纠正，不能把请求不存在/越界的 sub_range 当科学失败或泄漏成功。

顺序：
- L1：F_noSkill → F_domainSkill → Random；
- L2：F_domainSkill → Random → F_noSkill；
- L3：Random → F_noSkill → F_domainSkill；
- L4：F_domainSkill → F_noSkill → Random。
固定参照在每 Job 研究臂开始前生成/训练。不同臂不共享当前对话。

RandomSearch_B4 的冻结分布：
- 两份全批统一方案：从 53 个程序的统一赋值中，排除公共基线的 assignment，均匀不放回抽。
- 两份条件化方案：从公开 rd.FIELDS 中均匀抽字段，分位阈值从 {0.25,0.5,0.75} 均匀抽，谓词为 >=；默认/命中处理从 53 程序中等概率抽两个不同程序。UNKNOWN 按现役语义走默认。
- 只按 T 检查分组至少各有一个实体，并按完整 assignment 去重公共基线及本臂先前方案；不能用 C_A/C_B/E 决定保留或重抽。
- 每个条件化槽最多尝试 256 次；仍不成立则按同一 RNG 补一份尚未抽取的统一方案，记录 fallback。
- Random 的 RNG 用 `SeedSequence([2026091801,job_index])` 的独立流；四候选在该 Job 第一次拟合前固定，隐藏于 Fast。
- 与四基线一起按 C_A 三 seed 均值选交付；数值平局顺序为四公共基线的上述顺序，再按抽取顺序。
- 这是一种公开的混合随机分布，不声称均匀覆盖全部多规则策略空间。

## 7. 信息边界、缓存和运行故障

开始 Source 前写 frozen_config：所有 Job/roster/窗口、Consumer、环境、模型参数、系统提示、程序枚举、种子规则、额度、顺序、选择和聚合公式。
冻结 domain Skill 必须早于首次 Target 拟合。Target T 的特征不得提前放进 Slow/Select。
每个 Target 的合法 C_A 只给本臂，禁止读取别臂候选/行为或标签；共同公共基线除外。

**Target 全包屏障**：
全部四个 Job 的所有计划研究臂尝试结束、所有正常 commit/失败状态和 Menu_CA 决定冻结
→ 统一 C_B
→ 全部所需模型的 E 预测冻结
→ 全局 E 屏障
→ 统一打开 E 真值计分。
不得按 Job 提前开 E，不能读完 L1 的效果再运行 L2。Source/Select 的屏障按各自整个阶段执行，均不读 E。

缓存只节省物理计算，不增加某臂的候选额度。别臂已训练的材料，必须由本臂自主提出并花一个逻辑槽才能拿到 C_A。不提前展示随机候选。
共享缓存写入要串行或使用已有互斥，避免模型文件覆盖；只做当前实验需要的绑定，不建设哈希平台。

故障：
- 同配置拟合最多重试一次；Source/Select/Target 各最多 2 次额外拟合尝试，全包 6 次。只针对技术失败，不因分数不佳重训。
- HTTP 同一请求最多一次原样重传，全包最多 4 次额外 HTTP；记录所有尝试，不凭 5xx 认定费用为零。
- usage 一旦未知，保留 UNKNOWN，暂停后续收费调用并请求用户接受未知费用；记账预留只是估计，不是保证上界。
- 只在标签仍扣留且既有 resume 边界检查通过时允许恢复；接受未知费用不能解除标签边界。
- 恢复同一未答请求/同一候选，不重抽材料、不抹掉失败、不追溯改冻结预算。已建未评估方案须能恢复，沿用已修逻辑。
- 阶段中断只写 labels_withheld，不能自动开后续标签。可继续的无关工作继续。
- 代码 bug 在不改方法/数据/预算/反馈且未开受影响标签时可修；报告修订。方法或边界必须改时才升级，不自行换 Consumer 或样本。
- 技术未完成、执行完成、原协议合规分别记录；未知费用经批准恢复不自动等于“原协议无偏差”。

## 8. 资源封顶（用户转发执行时采用）

本次不另外启动付费健康探针；首次真实 Source 返回同时核验模型身份。沿用可工作的本地代理与既有凭据读取方式，不把 key 写入报告/提示。
实验请求模型 cpa-grok-4.6、期望返回 grok-4.6-build、temperature=0；其余解码参数沿用最近 readiness 成功配置并在开跑前明列。不同模型不静默替换。Fast/Slow 各 12,000 输出 token 上限，计入总 token 预算。

| 阶段 | 最大常规拟合 | Fast 分支/逻辑调用 | 阶段 token 上限 |
|---|---:|---:|---:|
| 接线 | 5 | 0 | 0 |
| Source | 4 × (4+4) × 3 = 96 | 4 / 64 | 1,600,000 |
| Slow 形成 | 0 | 0；1 次形成 + 至多 1 次合同纠正 | 400,000 |
| Select | 2 × (4+3×4) × 3 = 96 | 6 / 96 | 2,400,000 |
| Target | 4 × (4+3×4) × 3 = 192 | 8 / 128；Random 无 LLM | 3,200,000 |
| 合计 | **389** | **Fast<=288，Slow<=2** | **7,600,000** |

- 最多 6 次技术重试，**物理拟合尝试总帽 395**；接线 5 次内不再加检查拟合。已知 token 总帽 7.6M。额外 HTTP 仍受所属分支/阶段 token 帽约束，不能借其名另加预算；HTTP 总帽 294。
- 阶段结余不自动转移；分支预算相同、预算不足则标 INCOMPLETE，不读部分结果临时扩额。
- 实际跨臂语义相同可复用，物理拟合通常少于上限；实际账本与各方法独立部署成本分别报告。
- 数值活跃墙钟 <=4 小时；整包执行活跃墙钟 <=10 小时（从接线真实拟合开始；实现开发另记，明确人工等待）。外部竞争不自动延长。
- 以上是上限，不是必须花满。不得承诺 90 分钟能完成含 LLM 的整包。

## 9. 读数、成功范围与成本

先回答三个问题：
1. **Skill 是否改变了研究和交付，并胜过无 Skill？**
2. **完整方法是否胜过同预算 Random 和简单强参照？**
3. **质量相近时，包含形成成本是否仍有价值？**

每个 Job、每个臂交付保留 C_A/C_B/E 逐 seed 值。主配对差：
`d[j,s] = E(F_noSkill,j,s) - E(F_domainSkill,j,s)`，正 = Skill 好。
归一化主汇总：
`G = mean_j(100 * mean_s d[j,s] / mean_s E(None,j,s))`，四 Target 等权。
单位是该 Job None 平均损失的百分点；不把它误称“相对 Fixed 的百分比”。同样计算相对 Random、Menu_CA、AmpResample、NoMixRecipe、FixedMixup、None 的差。

- 每 Job 报配对 SE 和 n=3 双侧 95% t 区间（df=2），不拿单臂 SE 代替配对 SE。逐 seed 值不能省略。
- 全部 origin 保留；附每 origin 的贡献，不能事后用删峰值后的分数作为主读数。
- 同时列四个 Job 的效应、均值、范围和符号；seed 重复不是 12 个独立任务。三个 seed 只重复 Consumer 训练，不是三个独立 Fast/Slow 决策；本包不估计 LLM 决策重复的方差。只有 4 个同域批次，不声称稳定跨域统计结论。
- 未完成配对不填 0，不悄悄改主分母；主结果标完整性。可另列完整子集描述。
- 事后各已评估候选的 E 可用于“没探索/评估了未交付”诊断，明确为 oracle 描述，不改交付、不加实验。

行为记录：实际卡正文、加载证据、观察顺序、构造/派生方案、候选材料差异、是否利用前次反馈、commit 与理由；均匀方案完全合法，实际退化为固定菜单则如实写。
不能因为更多观察、更保守、调用减少或换算子就判学习有效。

成本：
- 一次性形成 = Source + Slow + Select；
- 部署 = 基线准备 + 当前 Fast/Random 研究 + 所选 Consumer/推理；
- 同时给总物理成本、每方法独立运行需要的逻辑拟合/LLM token/耗时；共享缓存不能制造后运行方法更便宜的假象；
- 若只省部署成本，报抵消一次性形成成本所需的批次数（可定义时），不称这四批已经抵消。

固定读法：
- 有卡胜无卡，但不胜 Menu/Random：只支持指导的局部增量，未支持完整方法优于简单策略。
- 胜 None/Fixed 而不胜 AmpResample：优先归于公共增强收益。
- 只有行为变化、同材料同交付或更贵：无净效用证据。
- 无卡入选：记录本次未形成可用 Skill；不能用 Generic 冒充。
- 均值正但区间跨 0：报告正向开发线索与不确定性，不强行二分成“显著有效/完全无效”。
- 本包结果不自动晋升 Skill，不证明连续自进化或全项目完成。

## 10. 交付与停止

交付复用同一目录：
- REPORT.md：首屏三问、四批主结果、成本、最大不确定性与唯一下一项；
- METHOD.md：来源适配、RNG/训练/信息边界、与 Eval-Skill 的对应及本包省略项；
- frozen_config.json、census/Slow 原始响应与候选、selected_skill.json；
- result.json 与 tables.md：公式、原始逐 seed 值、配对结果、状态与成本可重算；
- 必要原始轨迹、材料/模型/冻结预测及一个检查记录。不新建审计平台。
- 本任务书末尾追加“实际收口”；计划文档只追加简短事实回执，历史数值不改。

不因工程接通或固定算子有效就报方法成功。不得自动开始第二轮修卡、增加 seed、开放备用或最终数据。
收口时提出一项基于此次实际失败或正向线索的后续实验；由下一次任务决定是否执行。

## 11. 原草案待裁定项的处理

- 域：只 RD02；RD01B E-only 不做。
- Source 四批、Select 两批、Target 四批按表固定，不随机挑“有油水”的批次。
- 四固定参照全部可见，增加 0 成本 Menu_CA。
- Random 用两统一 + 两条件化，公开分布与退化处理。
- 同意本包显式采用 job×entity×program RNG，旧包材料版本保留。
- 最终验收区不在本包裁定，也不开放。
- Select 纳入无 Skill；KEEP/无卡胜出就是空 Skill，不替换为 Generic。
- 正式 Target 使用全包标签屏障，替代原草案“按 Job 各自开放”的表述。
- 预算按这些变更重算；原草案约 350 拟合/224 Fast 调用数字不再适用。

## 12. 实际收口（2026-09-19 00:15，执行者追加；历史数值不改）

- **执行 COMPLETE，出口 NO_TREATMENT。** 报告 `_scratch/dev_tempo_aug_workflow_skill/REPORT.md`，方法 `METHOD.md`，数值 `result.json` / `tables.md`，冻结 `frozen_config.json` + `frozen_config_amendment_1.json`，候选与选优 `formation/`、`selected_skill.json`，账本 `budget.json`。入口 `evaluation/main_protocol_p4/batch_research_tempo_aug_workflow_skill.py`；共享代码唯一改动为 `readiness.EXTRA_JOB_T["RD02"]` 登记 L1–L4。
- 接线 PASS 10/10（5 拟合）；smoke 19/19 + controller tests 26。
- Source：S1→AmpCensor、S2→u_regime、V1→P_NoMixRecipe、T2→P_AmpResample（4 条 COMPLETE，C_B 已开）。Slow 1 次：W1「public-first mild ablation」、W2「shift vs diurnal uniforms」（principles=null、const:true，首次合规）。Select：T4 无卡→P_NoMixRecipe（C_B 2.25）、W1/W2→regime_censor（2.85）；Q2 三者→P_AmpResample（1.05）。J(无卡)=0.801 < J(W1)=J(W2)=0.900 → **无卡入选，冻结空 Skill**，Target 的 F_domainSkill 记 EMPTY_SKILL_ALIAS。
- Target（L1–L4，E 以各批 None 均值百分点计，正=Fast 好）：F_noSkill 对 Random −3.2、对 Menu_CA −5.1、对 P_NoMixRecipe −14.8、对 P_AmpResample +8.7、对 FixedMixup +0.9、对 None +21.3；四批配对完整，G(Skill−无卡)=0。Fast 自造交付两次（L2 ResOnly、L4 amp_u）均为该臂 E 最差/次差；8 个臂的 C_A argmin 与 E argmin 0/8 一致。判读：本次未形成可用 Skill；收益归公共增强参照；无净效用证据。
- 成本：拟合尝试 284/395（含 2 次 seed 溢出失败、10 次工具故障作废）、LLM 93 请求/93 HTTP（0 重传）、已知 token 2.64M 入 / 0.138M 出、0 未知 usage；数值活跃 73 min，整包 2 h 47 min。
- 修订与故障：①训练 seed 改为 8 位编码 [20269181..83]（任务书数值溢出 Consumer 批次流的 2^32 上限，修订 1，任何正式拟合成功前）；②本机代理在 WSL，Windows 端口 8318 被 Hyper-V 保留段覆盖，用户管理员释放后继续，模型入口未改；③第一次 Slow 调用被冻结客户端字节预留拒绝（0 请求），驱动误启 Target 跑了 L1 10 次参照拟合后停止（无 Fast、无随机、无标签），目录作废为 `target__aborted_instrument_1/`；普查压缩加第三档只压尺寸规则后一次通过；驱动改为形成失败即停包。
- 唯一后续建议（未执行，交 Planner）：0-LLM/0-拟合诊断——用本包 10 个 Job 的全部已冻结分数测四公共参照间 C_A→C_B、C_A→E、C_B→E 成对排序一致率与可分辨差距；若 C_A 对 E 不高于随机，先改反馈块设计再谈 Workflow 学习。
