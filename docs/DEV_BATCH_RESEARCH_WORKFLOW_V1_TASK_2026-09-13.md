# DEV-BATCH-RESEARCH-WORKFLOW-V1 实施任务书

日期：2026-09-13
状态：COMPLETE，W首包三个分支真实完成；treatment_present=true，utility_supported=false（正效用未获支持）。实际收口见§10.3，先前实施过程按原时点保留。
当前分工：Astra 实施主目录 W 控制层并整合；Opus 的独立 P 线负责固定流程与公共数值底座。不新增串行审批角色。
来源：用户确认批级完整 Workflow，并要求落实文档及安排后续任务。
启动语义：用户转发本任务书或明确指示执行后，负责人在下述范围内连续完成，不逐步等待确认。当前文档更新不等于实验已启动。

## 0. 要交付什么

完成一个正式可复用的批级研究入口：
整批 T → Fast 自主观察/材料构造/真实比较 → Fast commit → C_B 延迟检查
→ Slow 一次候选修改 → 在变化后的下一 Job 上用新旧 H 重新执行。

M1 验收“完整可运行”，不强制正收益、非 identity、非空 Skill 或程序多样性。自然 KEEP、未命中、无差异/负收益都允许报告；不能伪造修改或重抽直到有利。没有真实更新时，不把这条运行称为已验证 Skill 改善。
只在已曝光 Electricity development 上运行，不启动完整 A3/A5 课程、R50/U50 网格、TSFM/新 Consumer 或其他数据集。

框架：[批级 Workflow 规格](BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md)；架构优先级见 ../AGENTS.md §5.9。

## 1. 修改范围与复用

允许：
- 增加/适配 methods/ttha 中的批级 Fast loop、观察/构造/实验/commit 工具和现有 Skill 解释入口；
- 在 evaluation/main_protocol_p4 增加本包数据/Consumer adapter 及一个 CLI；
- 为复用抽取旧 scratch 中已核验的数值函数，不改变算法数值语义；
- 一份必要 smoke、一个逻辑 runner、一个主报告。

旧 _scratch 包、冻结预测、原始结果、参考仓库、主表和 DECISIONS.md 不写。主项目其他未提交改动保留，不 reset、不清理、不 git commit/push。
General/Specific 沿用既有存储和匹配算法；扩展本包词表/命名空间，不覆盖旧清洗共享词表。
不建设通用 DAG 平台、哈希平台、全量近邻库、独立调度集群；0 新 hash。

参考资产：
- _scratch/ts_aug_source_grounding_pilot/：aug_ops、model_mlp、score_h1、后端和失败记录；
- _scratch/ts_aug_donor_pattern_probe/：R/U donor、完整材料、scaler/父窗口及配对训练；
- _scratch/ts_aug_feedback_stabilization/：配对反馈表达；
- methods/ttha/retrieval.py、既有 applicability/knowledge 实现；
- _scratch/ts_augmentation_preflight/data_spec.json：数据身份与 roster。
确认实际定义后复用；不能依赖 import config 的偶然搜索路径串到另一旧包。

## 2. 数据、作业与时间权限

使用 data_spec.json 的本地 TSLib electricity.csv，不称 Monash。按该文件冻结的字符串字典序32实体，保持完整人口。不要按收益、缺失或效果更换实体。
L=192、H=48、T历史=672小时，stride=1；每实体433父对，共13,856父对。

只用两个预先指定的已曝光 development 作业，顺序 a55 → a65：

| Job | 训练 T | C_A origins | C_B origins | E origins |
|---|---|---|---|---|
| a55 | [13776,14448) | 14448,14496 | 14544,14592 | 14640,14688,14736,14784 |
| a65 | [16416,17088) | 17088,17136 | 17184,17232 | 17280,17328,17376,17424 |

每个 origin 的历史输入是 [origin−192,origin)，目标是 [origin,origin+48)。训练/目标边界以此表和原 adapter 核验；资格异常停止受影响 Job，不换作业。
两作业及原文件已曝光，只能称开发贯通及后续重执行，不称独立未见验证。

- 构造/观察工具只访问当前 T；历史 y 仅指 T 内训练父对目标。
- Fast 通过 evaluate 获得当前 C_A；不取得 C_B/E。
- 各 Job 全部计划分支 commit 并冻结后才打开 C_B。
- a55 的 T/过程/C_A/C_B 可供本次 Slow；不含 a55 E、a65 观察和任何未来分数。
- a65 两个分支状态独立，不能互看工具结果、候选或反馈；公共数据/基线按相同权限提供。
- 所有分支、Slow 候选、选择及分析规则冻结后才生成两 Job 全部 E 预测，再统一读 E 真值。
- 后一个预测 origin 的合法历史包含先前时段已发生数据时，只作预测输入，不回传为标签反馈。

## 3. 固定 Consumer 与材料合约

共享 MLP：192→128→64→48，ReLU，无 dropout/BatchNorm；AdamW，lr=1e−3，weight_decay=1e−4，2000更新，每批64父对。每实体 scaler 仅用原始 T 的 mean/std，floor=1e−6。原始/派生均在同一训练坐标系，不对派生重新归一化。
主指标保持原 normalized MSE：48步均值、全32实体宏均值、块内origin等权；原始 MAE 作辅助，不能换成主指标追正号。

每父对保留原始与一个派生视图，各0.5权重；identity 的派生视图等于原始，复用已核实的 None/Copy 数学等价路径。总质量权重、训练更新和父 batch 不变。两个组合步骤仍只产一个最终子视图，不多加训练质量。

第一版动作边界：
- identity，TimeMixup，FreqMask，FreqMix；
- 每实体0–2步，可以全体相同，也可按合法观察分组；
- TimeMixup donor weight ∈ {0.10,0.25,0.50}；
- FreqMask/FreqMix rate ∈ {0.05,0.10,0.20}；
- 需要 donor 的动作可选 R 随机错排、U 同钟表小时错排；同实体原始父对、一一映射、无自配，沿用已核验实现；
- Scope/组条件使用已实现观察字段与既有三值表达式，支持按当前分布取得量值；不为第一版强建聚类器；
- 不改 scaler、训练历史长度、人口权重、模型、损失或预测侧。
这些范围是小型实现边界，不是预言某参数有效；不按旧 E 为 Agent 预填正确分组/参数。当前负的 donor 配置不作为全家族禁令写入 H0。

## 4. 实际 Fast 与 Workflow

采用同一个 run_job(job, knowledge, budget, mode) 入口。每次请求面向整批数据，不能内部退回32个孤立 LLM 配方调用；数值工具逐实体循环正常。
初始化提供全32实体的简明特征表和 Consumer/权重说明，禁止只用第一条代表全批或只给均值。
可按需调用：
overview、inspect_data、build_material、inspect_material、evaluate、compare、commit。
组合工具请求允许批量读取，提高调用利用率。Workflow 为实际 tool-call/control-flow 轨迹及构造规范；不是执行完后再写的散文。

H0 只包含公共任务/工具语义、操作权限及诚实反馈说明，不含旧 E 排名、人工“正确”Pattern或残差平均结论。Fixed-Mixup 作为已命名 incumbent/基线向全部分支提供，Agent 可参考、检验或超越。
本包提供四种 Skill hook：observation_guidance / construction_guidance / experiment_guidance / decision_guidance。每次 Slow 只改其中一个主要行为面，或 KEEP。

实际记录：
- 每次调用的工具、参数、输入范围及输出引用；
- 使用的知识版本、实际渲染正文、适用性检查；
- 完整材料规范、展开后的32实体赋值和完整 donor/mask 索引；
- 模型/seed/公共训练配置、完整反馈、最终 commit；
- 简短假设、证据引用与决定理由，不要求长篇隐式思维链。

第一版 evaluate 每个新材料固定跑3个配对 seed；不做自适应追加到显著。Fast 可以先看材料、检验一个方案、据反馈提下一方案，或提前停止。新增观察次数不能自动转成方法收益。

## 5. 三次 Fast 执行与一次 Slow

### A. a55 / H0：形成一条真实研究 Episode
先拟合/读取本 Job 的 None 和 Fixed-Mixup 共同基线；Fast 最多再提两个完整候选。根据 C_A 自主研究并 commit。
保存三 seed 全部模型，交付 seed 预先固定为第一个，不能按 C_A 挑 seed。
commit 后开放 C_B，对所有已完成候选评分；不改当前交付。

### B. a55 边界：Slow 一次自然修改
Slow 读取 a55 完整研究 Episode、当前 H0 和本包四个可编辑面，提出一个有依据的指导修改或 KEEP。
公开证据由工具写入，原始分数不可修改。非空正文需显式 scope，主动无条件用 const true；null/缺条件为 INCOMPLETE。修改结果同时明确 hook、procedure 与证据限制。
最多允许一次针对明确 JSON/契约错误的纠正请求，提供原输出和具体错误；不是因 KEEP、窄 scope、建议保守或效果不佳而重抽。保留全部原文和失败状态。
本包候选标为 CANDIDATE_TEST_ONLY，供隔离开发重执行，不写 canonical Skill Store，不称已晋升 Shared Capability。

### C. a65：新旧 H 重新执行
两分支：parent-H0 与 candidate-H，均从同一当前 Job 空 scratch 开始，预算与工具相同，共用冻结基线权限。
各自最多两个新材料候选，允许不同探索，分别 commit 后统一打开 C_B。
未产生合法修改时只跑 parent，候选结果记 NO_TREATMENT，不强造第二臂。候选无命中时可先通过所需 T 观察补查；仍无暴露则报告处理为空，不把同 H 重跑当 Skill 效果。
不做第二次自然 Slow 修改，不使用 a65 结果修好原候选再续跑。

本包是一次 Source 形成及后续新旧指导开发对照，不标 A3/A5：没有完整 Target 多批次适应，也没有多形成轨迹。
自然失败不阻止无依赖分支和外部报告收口；最终报告区分方法完成、处理是否实际存在、效果是否支持。

## 6. 反馈、commit 与统计

正差统一定义为 loss(parent/reference)−loss(candidate)。保存全部 seed 值及块均值、SD、SE；n=1时不补 SE。
三seed同号、近似区间或单Job正差不作为自动科学通过门。本包不新增 MATERIAL、2σ 或收益率门。

Fast 可用 C_A 的整体配对证据决定 commit；程序只校验其引用的是本分支已完整拟合候选，不能替它按 C_B 重选。
若预算耗尽时没有有效 commit，记 FAILED/INCOMPLETE；若服务输出需要 incumbent 兜底，另列 operational_fallback，不将其填成 Agent 主成绩。
延迟 C_B 和 E 分列；主效果不删实体、不改指标、不用旧矩阵拼得分。

固定首个 seed 的 E 是单模型交付读数；全部三seed的 E 配对均值另报，用于理解训练随机性，不构成 ensemble。一个 Job 的32实体/多origin不算32次独立方法试验。
不能仅因结果负就换观察/动作/参数继续当前包。

## 7. 随机性、预算与缓存

训练seed：20260922、20260923、20260924。
初始化与父 batch 各方案按 seed 配对；沿用 batch_seed=800000+100*model_seed+9，父窗口采样形状(2000,64)。
增强流与训练流分开：每实体/步固定流，建议 step_base=900090+100000*step_index，再加实体索引；R/U 各自新建流，不能因前面工具调用消耗随机数改变 donor。参数不同尽量共用底层随机量；保持各动作的边际定义，实际关系写清。材料在三个训练seed间固定。

| 单元 | 基线 | 新候选上限 | 最大逻辑拟合 |
|---|---:|---:|---:|
| a55/H0 | None+Fixed，各3seed | 2×3seed | 12 |
| a65/parent | None+Fixed，各3seed | 2×3seed | 12 |
| a65/candidate | 同一a65公共基线，各3seed | 2×3seed | 12 |
| 合计 | a65基线物理复用、逻辑同权 | | 36 |

最大新物理拟合30次：a55为12，a65为18；外加最多1次原seed复现检查、最多2次瞬态重试，总拟合尝试硬上限33。余量不得用于追加候选/seed/Job。材料别名精确复用时物理成本下降，不拿省下次数扩搜索。
当前两个 Job 的 C_A 被自适应访问，需另报反馈请求与缓存读取；缓存反馈不能冒充新增独立证据。

LLM预算：
- 每次 Fast 最多16个逻辑请求，最多24个工具执行动作；3次 Fast 最多48请求；
- Slow 1次自然请求，另允许最多1次契约纠正，总逻辑请求上限50；
- 每请求最多一次明确瞬态传输重传，总 HTTP 尝试上限100，所有重传计费；
- 整包输入+输出 token上限2,000,000；按后端 usage，未知时记录而不补0；
- 实验模型继承最新有效配置：请求 cpa-grok-4.6，返回须为 grok-4.6-build；若后端身份不符停止实验，不默换模型。不将凭据写入文档/日志。
不额外发无账本“测试API”；首次实际调用即计费和验证身份。

时间：单拟合子进程300秒，单请求300秒；实验活跃墙钟2小时硬上限，含训练/推理/调用/评分；整任务实现+验证+报告最多6小时。计时须在运行中能中止，不能仅在下一格前检查。
单GPU拟合串行，最多等待其他合法占用15分钟，不终止他线。没有GPU或API不可用时完成所有独立实现工作并如实PARTIAL收口。

## 8. 实施顺序与必要检查

1. 建公共数值 adapter 和批级 tool loop，保留旧实现不变。
2. 接上全部人口概览、材料构造、真实配对反馈及 commit。
3. 接上四个 Skill hook、三值匹配和 a55→a65 的边界，模式/权限进入同一入口。
4. 一份集中 smoke 覆盖新关键路径，不复制大型旧测试集：
   - 同一批不同实体观察能进入同一 Fast 请求，组规则实际展开正确；
   - 父/子权重、完整训练、材料别名与原基线对齐；
   - evaluate的输出不含C_B/E；commit后才开C_B，交付不受C_B覆盖；
   - MATCH/NO_MATCH/UNKNOWN、const true与null区分，实际正文可见性一致；
   - 原始Source Episode不进Fast；新旧a65状态不串用；
   - 故障与主动identity/KEEP分开，计数与恢复不重抽已有合法响应。
5. 至多一次原seed原材料重训对齐；无新SHA。无法对齐时查真实原因，不偷偷更换训练器。
6. 连续执行 §5；所有可完成分支冻结后统一E预测→统一E评分→报告。

开发中普通接线/解析/路径错误自行修复；真实执行中若修改会改变工具语义、H0、材料或决策协议，应冻结受影响运行并如实收口，不能混同版本继续记成一包成功。

## 9. 交付与验收

输出 _scratch/dev_batch_research_workflow_v1/：
- 一个可复跑CLI的说明、运行配置和成本；
- trace.jsonl、knowledge快照、材料/模型/预测引用与实际索引；
- result.json（完整/未完成状态、实际commit、C_A/C_B/E和配对差）；
- REPORT.md 一份主报告，说明实现位置、复用资产、行为变化、收益/伤害、成本和仍缺什么。

框架完成标准：
- 一批数据→完整 Workflow→真实共同训练→反馈→commit→边界 Slow 的路径存在并实际执行；
- 后续 Job 可以使用候选知识重新执行；自然KEEP/无处理单列，不假装正向学习链；
- 对固定同一 H 的模式、后续 A3/A5 多批次模式、无反馈部署模式，接口权限一致且后者关闭当前标签反馈工具；本包不启动完整课程。
- 没有退回逐实体孤立LLM、隐藏 C_B argmin、跨模型拼分或用代理损失代替训练价值。

报告分别回答：
1. Fast 是否自主改变观察、构造或比较，哪些只是固定骨架；
2. 完整训练方案相对 None/Fixed 的实际效果；
3. Slow 是否提出实质修改、是否送达、是否改变后续行为；
4. 新旧H在a65的效用/成本读数及不确定性；
5. M2 应优先优化的一个实际环节。

可用状态：COMPLETE / PARTIAL / BLOCKED；另列 treatment_present 与 utility_supported，不能把三个维度揉成一个PASS。
完成即停止；不自动扩到a75、更多Source、更多seed或完整A3/A5。把交付路径与实际预算交回，接收/启动/完成状态由真实收据更新。

## 10. W 线实施进度与公共底座接线（2026-09-13）

当前主目录承担 W 工具循环；P 固定流程线在独立的 `.claude/worktrees/p-line-batch-base/` 开发。
本节是实际进度追加，不修改上文数据、训练预算、评价目标或权限。

已实现：
- [batch_research.py](../methods/ttha/batch_research.py)：整批 Fast 循环、七种操作的调度、显式 commit、候选/Job/seed 绑定、四类指导 hook、三值匹配、一个 hook 的候选修改与 KEEP；
- [test_batch_research.py](../tests/functional/test_batch_research.py)：13 项接口测试，在已有 Windows project 环境通过；
- 复用主项目 retrieval._evaluate；不扩 canonical observable 词表，不从另一个 scratch import 可变 config；
- 当前状态为 CONTROL_LAYER_READY。模型训练、真实 LLM、Slow 自然形成及 a55→a65 全链尚未执行。合成损失和脚本化响应仅验证控制逻辑，不能记为共享 Consumer 训练或效用证据。

公共适配器交接：
1. P 线 batch_base 负责数值原语；W 线只增加薄适配，不修改 P 工作树。待其接口完成后在主目录按已核的源码接入，避免直接依赖他线正在改动的文件。
2. `BatchAdapter.overview()` 返回全32实体表及具有明确统计含义的 `batch_features`；不得把任意一个实体的特征冒充批特征。部署可见字段白名单与 `tool_contracts` 在启动前冻结。
3. `inspect_data`、`build_material`、`inspect_material` 映射到 P 的 context/policy/materials。它们只读 T；控制层的字段拒绝检查不能代替数据层的物理权限隔离。
4. `evaluate` 返回绑定当前 Job、完整材料规范、固定 seed 顺序及模型路径的 Candidate；C_A 损失张量形状为 [seed][origin][entity]。原始/派生权重、父 batch、donor 流仍由公共底座保证。
5. Fast `commit` 的结果交给公共 commit 持久化；同 Job 所有分支冻结后，外层 runner 才开放 C_B。不得另接 C_A/C_B argmin 改写 W 的选择。
6. 无反馈部署的内部拟合必须实际不读取校准标签；当前控制层只验证 `feedback=False` 的契约，不能把“训练函数读过 C_A 后不返回它”当作已经满足物理信息墙。
7. 真实 Client 提供结构化 actions、计费及原始响应；物理 fit/token/HTTP 预算、运行中的硬超时和持久化由已计费的外层 adapter/client 保证。控制层的逻辑计数不代替整包账本。
8. Slow 在边界收到合法 Episode/延迟证据后，再接真实提案调用和至多一次契约纠正；`apply_update` 只处理已收到的提案，不代表已经形成有效知识。

下一项可执行工作：完成公共适配器和真实 Client/Slow 边界接线、基线对齐及阶段权限检查；再按本任务书启动 a55→a65 首包。首包拟合尝试上限仍为33，不能因增加并行开发线扩大为两套预算。
本次没有启动该运行、没有新增哈希、没有写 P 工作树或历史结果。


### 10.1 本轮继续实施收口

- 公共底座已从 P 已交付首版快照接入 `methods/ttha/batch_base/`；未修改 P 工作树。
- 真实薄适配及计费客户端：`evaluation/main_protocol_p4/batch_research_runtime.py`。
- 单入口：`evaluation/main_protocol_p4/run_batch_research_v1.py --run`，需 Windows project Python 以 `-m` 运行。
- 数值训练、三seed完整反馈、Fast commit、a55 Slow一次候选、a65分支、所有E预测冻结后评分已接线；真实 Agent 行为尚未测试。
- 19项检查通过；另1次实际 a55/R/20260918 拟合与历史模型权重及 C_A 逐位相同，子进程56.2秒，独立阶段74.74秒。报告：`_scratch/dev_batch_research_workflow_v1_alignment/REPORT.md`。
- 本次外部实验启动被自动审批拒绝，原因是认为尚缺向具体外部 Grok 目的地发送时序观察和校准反馈的授权。没有绕过拒绝；0实验API。
- 已用1次拟合计入§7上限33；后续 Runner复用既有对齐并计入费用/时间，不重新拟合、不增加余额。
- 当前状态 `LIVE_INTEGRATION_READY__EXTERNAL_SEND_APPROVAL_BLOCKED`；不称COMPLETE、不称M1通过。

公共底座必要本地修复及适配：标签前置检查在读取之前、CSV不解析上界后一行、完整origin×entity反馈张量、仅T的部署拟合、完整频域mask记录、拟合启动前预留及剩余时间硬超时。训练公式、评价语义不变。这些是所有控制线可复用的工程修复，不算W方法优势。


### 10.2 外发授权确认

用户随后明确回复“随便你发送，这个是我自己的中转站”。因此本包 T 观察（含按需历史片段）、当前合法 C_A 及仅供 Slow 的 a55 C_B 可经既定本机中转发给冻结模型；原始跨作业 Episode 不入 Fast，E 不发送，原模型/拟合/请求/token/时间预算不扩大。此前自动审批阻塞已由明确授权解除；开始前仍核查 GPU 与重复进程。

### 10.3 首包实际收口

用户授权后真实执行完成：a55/H0、a65/H0、a65/候选H均COMPLETE；31次拟合（含既有对齐1次）、12次实验LLM、580608 token、26.20分钟，失败/重试0。Slow自然修改decision_guidance，后续3/3 Fast请求MATCH。a65新旧H交付E三seed均值分别0.553900/0.549636；未支持新H正增益。只完成本包最小开发闭环，不晋升M2/M3或完整A3/A5。

收口报告：[_scratch/dev_batch_research_workflow_v1/REPORT.md](../_scratch/dev_batch_research_workflow_v1/REPORT.md)。数值复算通过；新上下文同家族审查未见本次完整性阻断，provisional。后续单独score_e入口全局冻结守卫缺口列入报告；本包实际全局顺序正确。未加种子、作业、课程，历史P4/Natural Final状态不变。
