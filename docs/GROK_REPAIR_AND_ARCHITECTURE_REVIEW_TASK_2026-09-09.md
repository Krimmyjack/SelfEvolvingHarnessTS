# Grok 执行任务：确定缺陷修复与 Skill / Self-Harness 架构调研

日期：2026-09-09。协调及最终复核：Astra。执行模型：WSL `grok-4.6`，`xhigh`。
授权依据：用户本轮要求优先修复已讨论的代码问题，直接交 Grok 执行；允许其并行承担
定向源码/论文调研，最终由 Astra 整理当前架构与结果。**本任务不授权新项目实验或方法重构。**

## 0. 两个有边界的执行角色

启动指令会指定 A 或 B，只执行被分配的一项。不得继续创建子 Agent，不得启动另一个
Grok/Codex/Claude 或调用项目 LLM。你是执行/调研协作者，不是被研究 Harness 内的 Slow。
量大管饱不是修改权限；依据不足须写 UNKNOWN/未查明，不以自信推断补齐。

| 角色 | 任务 | 唯一输出与写入范围 |
| --- | --- | --- |
| A，修复者 | 修已确认的返回状态与证据压缩问题；零成本关键回归；组合能力只诊断 | 下述两个 runner、定向测试、`docs/GROK_CODE_REPAIR_RESULT_2026-09-09.md` |
| B，调研者 | 当前 Skill / Scope / surface / 归因与聚类；Self-Harness 深读，少量近邻比较 | 只写 `docs/SKILL_SURFACES_AND_SELF_HARNESS_AUDIT_2026-09-09.md` |

先完整读取工作区与主项目 `AGENTS.md`；只读参考仓库时先检查其更具体说明。
主项目为当前工作目录。不要把相邻 `SelfEvolvingHarnessTS*`、历史任务书或参考项目
的规则替代当前正典。先读 `docs/SLOW_FEEDBACK_UPDATE_DESIGN_NOTES_2026-09-08.md` §15。

共同禁止：打开 sealed / TARGET_HELD_IN / +144；新 Consumer fits；项目模型调用；
修改风险线、Consumer、Scope 词表、数据/split、评分终点、候选槽位或检索策略；
生成新 SHA、哈希平台；修改/覆盖历史工件；修改全局模型/认证配置；打印凭据；Git
提交/切换/重置/清理。主目录已有未提交改动，先记路径，保留所有他线改动。
本次使用 Grok 的研发调用不是实验 LLM 调用，须区分，不能声称整个工作零模型成本。

只用 apply_patch 编辑文件。定向测试优先现有环境；不可自行安装依赖。WSL 未满足时，
核查 Windows `project` 环境是否可用；不能为跑测试启动大实验或改环境。
主助手已只读确认该解释器为 `/mnt/d/Anaconda_envs/envs/project/python.exe`，
含 pytest 9.1.1、numpy 2.2.6。若 Grok 工作区沙箱内 Windows interop 被系统拒绝，
不要关闭沙箱或扫描整盘；完成白名单补丁和测试文件，交由主助手在该现有环境复核。
普通实现问题自行解决；涉及白名单外核心行为/数据权限则停止受影响修改并报告，调研可继续。

## A. 确定缺陷修复

### A1. 失败状态不得伪装成正常 identity

入口：`evaluation/main_protocol_p4/run_dev_deploy1.py:816,850`；
内部返回：`methods/ttha/fast_agent.py:1326–1352`；调用继承：`run_dev_deploy2.py`。
现 `method.prepare()` 返回的 PreparationResult 被忽略，仅看 trace；内部 FAILED
可以带 trace 正常返回，使空选择被记为 identity/0，faults 仍为空。

要求：

1. 消费 PreparationResult 状态及已有 ExecutionReceipt；在选择/评分前明确分流。
   FAILED、无法解析选择与正常 ABSTAINED/identity 不混为一谈。保留简短的错误类别、
   阶段/可获取状态、请求数和现有 trace；不保存密钥、完整环境或大段敏感请求。
2. 本路径的有效决定不存在时，主机制读数为 UNKNOWN，不因为 trace 存在就给 0。
   正常选择 identity、预定义且实际执行的合法性 raw 回退仍为 0，不删掉这些零。
   若另报实际故障 raw 兜底的系统效用，单列并明确它不是模型的主动选择。
3. 已有整批分母/UNKNOWN 规则继续成立。异常行不丢出人口，不通过“只算可读行”
   生成新的主均值。真正账户/权限错误仍按现有停止语义处理，不成为模型弃权。
4. 检查恢复/汇总及形成证据入口：旧 `EMPTY_SELECTION_FALLBACK_TO_IDENTITY` 行
   不能静默成为新实验的有效成功/失败经验。原工件不动；新视图明确标记原记录语义
   与有效决定缺失。不得自动重调 LLM 补跑历史数据。

授权修改已收口 runner 的这一复用缺陷，仅用于未来执行；以当前 Git HEAD 为基底报告
差异，不声称历史实验当时已经使用修正版本，不重写旧实验数值。

### A2. 恢复 Slow 的真实逐序列程序对照，不增加新证据

`run_dev_deploy2.py:_render_menu` 删除 `per_series_gain`，但提示文字又称其存在。
保留已有合法形成读数，提供紧凑、明确的同 position/UID/Program 成对可读信息。
最小可接受实现是保留原来的 per_series_gain 字段并纠正注释，无须新建复杂对照选择器。

- 不选最好看的对子；不读新的 review/密封数据；不补 fits；不把不同 origin、参数、
  Consumer 或服务人口的成绩拼成一对。尚未测得的程序/序列标 UNKNOWN，不填零。
- 成功、失败、identity 控制均保留。程序身份/参数以现有记录为准，不只凭算子名拼接。
- R/C 拿同样的信息；C-minus 不得从新增字段、赢家标签或收益差偷回新增效果。
- 检查形成材料的 `_render_case` 及其上游是否消费 A1 识别的历史错误行；做显式标记，
  不将历史错误零当作有效效用证据，不在此顺手重新设计材料分组或增设错误类别。
- 不把此次输入修复称为已证明比较式学习有效；历史 C/R 仍是其实际聚合材料上的对照。

### A3. 关键测试与生成空间只读诊断

必须有小型定向测试，至少覆盖：

1. 内部返回 FAILED 且 trace 非空；外部抛异常；都不成为主动 identity。
2. 明确选择 identity 的合法零保留；合法程序路径不变；账户错误停止。
3. 一条缺决定的整个人口主读数 UNKNOWN，可读子集仅诊断。
4. 历史空选择的读取/恢复不绕过修复，不改原 JSON。
5. P/Q 总体相同但逐序列反号的夹具，经过菜单渲染仍能辨别个体差异；真实零与缺测不同。
6. C-minus 没有新增数值效果/赢家泄漏。

另只读检查 Fast 的 1–4 步解析、默认单步 actionability 筛选、Skill/Agent 合池截断。
能用现有编译器做零 LLM 的两步夹具则做；不能需要 Consumer fit。区分“组合可编译”
与“模型自然会生成好组合”。如发现单步默认失败却组合合法的误剪枝，报告最小例子，
**不在本任务改变算子可见菜单、槽位、冻结程序供给或 verifier 规则**。

写入白名单：

- `evaluation/main_protocol_p4/run_dev_deploy1.py`
- `evaluation/main_protocol_p4/run_dev_deploy2.py`
- 新增一个定向测试文件 `tests/main_protocol/test_dev_deploy_feedback_integrity.py`
- `docs/GROK_CODE_REPAIR_RESULT_2026-09-09.md`

不修改 `methods/`、`runtime/`、`contracts/`、AGENTS、DECISIONS、状态页、其他报告。
如必须超出以上路径，先留下原因和建议，Astra 裁定；不要用猴子补丁偷偷实现禁改语义。
报告给出：修改前后语义、精确文件行、测试命令及实际结果、未测项、未决问题、完整
改动清单。不要只列通过数量。原 18/240 与第一包 9 条的历史结论不因修好代码自动变正。

## B. 架构及 Self-Harness 定向调研

目标是让用户理解系统，而不是写一篇通用自进化综述。源码优先，论文辅助；历史设计
和当前实际入口分开。A 同时可能改两个 runner，研究其历史行为时用 Git HEAD 或注明
读取的是修复后工作区；不要把并发变化混成同一版本。Git 版本号可引用，不新算哈希。

### B1. 当前架构必须回答的八个问题

1. General/Specific 在哪定义、存什么？正式数据结构与提示中散文段落分别是什么？
   是否真正有结构化 WHEN/OBSERVE/TRY/RISK/VERIFY/FALLBACK，还是只有正文约定？
2. Skill 现在还有哪些 Scope？逐一分清 `observable_applicability`、legacy
   `serving_scope`、Workflow region 参数、实际评价人口；谁消费、在哪执行、是否阻断
   加载或处理。不要用一个“有 Scope”回答四个问题。
3. Fast 实际读取什么字节、何时读：General/Specific/候选模板各到 inspect/propose/
   select 哪段？追踪一个真实 Skill 的“存储 → 检索 → prompt → 候选 → 交付”。
4. 当前可修改 surface 的目录、操作与权限；六种模板与具体实例数别混。哪些面能
   改观察/候选/选择，哪些只改加载范围？“能影响”与“历史上已测影响”分列。
5. Slow 实际按什么时点触发、看哪些事实、调用几种阶段、输出什么；原因标签是谁
   决定，是否由 surface 反向映射。一个 LLM edit 调用不冒充完整因果诊断算法。
6. 每次更新是否先做错误聚类？分别追 DEV-SEQ 与最新 DEV-DEPLOY-2：规则分组、
   LLM 逐轨迹诊断后聚类、整批池化材料、提示模型自行挑子群必须严格分开。
7. 修改是否真修 Skill 还是换冻结程序；父子如何保留、失效如何处理；编译、暴露、
   实验分支、晋升、独立有效性分别有什么实证。历史 AUTO/KNOW 与逐序列 DEPLOY 不并表。
8. 现有改法可能遗漏了什么：程序对照信息、局部观察、生成剪枝、模板竞争、共享训练
   效应。每项标“已证缺陷/设计选择/待测假说”；不要把所有负结果归结为一处。

入口：AGENTS §1–5.6；`contracts/harness.py`、`contracts/program.py`；
`methods/ttha/{fast_agent,slow_agent,retrieval,method,generative_workflow}.py`；
`methods/ttha/harness/{harness_surfaces.json,h0/}`；
`evaluation/main_protocol_p4/{dev_seq1_knowledge,dev_seq2_knowledge,run_dev_seq1,
run_dev_seq2,run_dev_deploy1,run_dev_deploy2,per_sequence}.py`。
只为解释具体差异再追调用方，不把整个项目重复读一遍。

### B2. Self-Harness 深读：先查它真正怎样提出有效修改

本地只读仓库 `../Self-Harness/`；论文
`../Paper/Self-Harness Harnesses That Improve Themselves.pdf`；公共标识
https://arxiv.org/abs/2606.09498 。可读既有 `../self_harness_extracted.md`，但先核对
它对应哪版论文；本地 PDF/源码版本不一致要披露。不要使用旧调研结论替代本轮源码核对。

必读源码链：diagnosis 的 trace.py / integrated.py / tb2.py → proposer 的
multi_proposer.py / hooks.py / materialize.py → acceptance gate → workflow loop。
抽取以下机制：

- 一条轨迹中如何定位 terminal failure、恢复过的错误、关键行为？是规则还是 LLM？
- 聚类发生在解释之前还是之后？按哪些字段合并？同标签是否就认同因？孤例如何处理？
- 聚类如何变成提案 brief？成功案例如何保护？改哪个 hook、候选多少、怎样允许 no-op？
- 可改面究竟包含 prompt、Skill、tool/runtime 哪些内容？不能把它的整个开放面直接
  推荐为本项目应开放的范围。
- 回归验收如何比较父子、允许怎样的损失、是否反复使用称为 held-out 的集合？
  它的 held-out 不自动等价于本项目最终密封评价。
- 寻找 1–2 个有实际工件支撑的“失败模式 → 修改 → 行为 → 结果”例子；只有最终
  harness 文件或论文总体涨分时，明确不能复原该次因果链，不编故事补齐。

近邻最多再深看两种互补机制：GEPA（反思局部文本更新、外部选择）与 AutoGuide
（成败轨迹差异提炼条件指导）。优先已有本地材料/官方原文；找不到原文则列缺口，
不为了凑数量扩到十几篇。不得说所有其他领域都在测试时看答案，也不能从总涨分断言
错误聚类本身有效，必须查相应消融是否存在。

### B3. 报告形式与禁止推论

报告至少包含：

1. 一页给用户的白话架构图与术语；
2. 我们/ Self-Harness / 两种近邻的事实矩阵，每项标源码文件:行或论文页/节；
3. 当前 surface → Fast 阶段 → 影响通路 → 真实例子/未测的映射；
4. Scope 四种含义及结构化字段 vs 正文指导的区分；
5. 当前错误聚类实际有/没有哪些环节；
6. 最多三个设计选项（最小输入修补、轻量机制分组、结构化 Skill 局部修订），每个
   写清解决什么、不能解决什么、最小可证伪比较。只能建议，不执行，不预设新方法必胜；
7. 核查范围/版本、仍未知项，以及需要 Astra 复核的精确问题。

必须保留：成对证据并非每条 Episode 天然存在；收益受训练—服务管线共同影响；
Scope 条件可判定不等于效用可预测；结构化合法性不等于学到有效知识；代码可达不等于
实际起作用；指导起作用不等于涨分；C-minus 弃权不是有效性因果证明；有事后菜单空间
不等于部署时能识别。近期故障零/删行均值不能被当作干净结果。

只写指定报告，不修改参考仓库、主项目源码、已有 Fable 文档或状态页。完成后返回
摘要、路径和最值得根 Agent 核查的三处证据。研究过程不得泄露本项目私有内容到网页
搜索；公共检索只用公开论文题目/通用关键词，避免 MDPI 来源。

## 交付与停止

以完成指定任务为终点；启动设置有包级时间/轮数上限，触及上限就保存部分进度与
未完成项，不制造“全过”。只要不改变方法和权限，测试/排查不中途逐项求审批。
账号/权限/依赖阻塞按事实报告，不换模型、不无限重试。Astra 复核后才可将结果写成
正式项目结论；“任务书已写”“已启动”“实现完成”“复核通过”是四个不同状态。
