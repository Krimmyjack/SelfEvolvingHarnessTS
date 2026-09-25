# DEV-TEMPO-AUG-TEMPORAL-FEEDBACK-REPLAY

任务书日期：2026-09-19。执行负责人：Opus。用途：可分发的开发实验规格。
状态：允许执行本文限定的实现、缓存回测和补充推理；不调用实验 LLM、不训练模型、不启动后续 Fast/Slow 包。

## 1. 问题与证据边界

检验：用多个较早批次中、与部署模型使用时段相同的反馈选择增强方案，是否比当前批次的单块 C_A 选择更好，并超过固定默认。

当前部署代理任务保持：在 t 前 672 小时历史上训练一次共享 MLP，模型权重不更新；每个预测起点读取此前 192 小时真实可见输入，预测随后 48 小时。E 的四起点为 t+192/240/288/336，目标覆盖训练完成后第 8–16 天；它不是从 t 一次预测 16 天。

本包是固定候选上的历史反馈选择回测，不是 Skill 有效性实验，也不是已实现的新候选实时评分工具。所有 L1–L7 已被研究者看过结果，统一标 EXPOSED_DEVELOPMENT_REPLAY，不宣称前瞻验证或独立泛化。配置须在本包补开历史结果之前冻结。

不要求无卡 Fast 先胜过简单方法才允许研究 Skill；本包只是先检验下一轮学习所用反馈的一种改法。不得把项目永久收缩为四选一 Router。

## 2. 固定项

- 仅 RD02，现有排序的 12 个 PM2.5 站点。RD01B、PRSA row>=24864 及其他封存区均不开放。
- Consumer、2000 更新、scaler、合法父窗口、训练随机流、服务输入、评分与缺失覆盖规则全部沿用缓存身份。
- 仅四个公共完整方案，顺序为 None / FixedMixup / P_AmpResample / P_NoMixRecipe。不新增、修改增强方案，不构造私有候选。
- 种子固定为缓存使用的 20269181、20269182、20269183。不补 seed、不替换模型、不用不同包的旧版材料混充同一方案。
- 读取指定缓存中的模型与分数；旧目录只读。所有新预测、分数、选择与报告写入新根目录。

## 3. 时间安排与缓存

本包会用到的 Job 如下，均为 RD02_ 前缀。t 单位为小时行号。

| Job | t | 角色 |
|---|---:|---|
| V1 | 5760 | 初始历史 |
| T2 | 8160 | 初始历史 |
| T4 | 10560 | 初始历史 |
| Q2 | 12960 | 初始历史 |
| L1 | 15360 | 顺序回测 1，结束后可供后续历史使用 |
| L2 | 16560 | 顺序回测 2 |
| L3 | 18960 | 顺序回测 3 |
| L4 | 20160 | 顺序回测 4 |
| L5 | 21360 | 顺序回测 5 |
| L6 | 22560 | 顺序回测 6 |
| L7 | 24480 | 顺序回测 7；E 末行 24864 exclusive |

每个当前 Job 只使用此前已完成完整反馈的最近四个表内 Job，不进行相似度筛选、近期权重调优或表现筛选：

| 当前 Job | 固定历史 H(j) |
|---|---|
| L1 | V1, T2, T4, Q2 |
| L2 | T2, T4, Q2, L1 |
| L3 | T4, Q2, L1, L2 |
| L4 | Q2, L1, L2, L3 |
| L5 | L1, L2, L3, L4 |
| L6 | L2, L3, L4, L5 |
| L7 | L3, L4, L5, L6 |

强制核对每个历史 h 的 t_h+384 <= t_j。不能借后续 Job 分数、当前 Job C_B/E、报告结论选方案。

缓存分支沿用 field_readout.py 的对应路径：
- `_scratch/dev_tempo_aug_workflow_skill/source/RD02_{V1,T2}_no_skill/`
- `_scratch/dev_tempo_aug_workflow_skill/select/RD02_{T4,Q2}_no_skill/`
- `_scratch/dev_tempo_aug_workflow_skill/target/RD02_L{1..4}_f_no_skill/`
- `_scratch/dev_tempo_aug_decision_learning/test/RD02_L{5..7}_f0/`

任务书编制时只查模型元数据和文件存在性：上述 11 批各有 12 个公共 cell；初始四批共 48 份模型在仓库中存在。cell 的 model_path 为 Windows 绝对路径，在 WSL 只按同一仓库的相对后缀解析；不按文件名猜模型，不改旧 cell。

初始四批暂无对应 E 分数文件。本文明确允许在新目录中、用它们原有模型补算 t_h+192/240/288/336 的预测和分数，作为本次历史开发反馈，命名 `historical_late`。这会新增这些历史目标的曝光，须记录；不声称全包 0 新标签，也不追溯改旧包的 Source/Select 信息墙。

L1–L7 原 E 已评分，只读取既有文件，禁止重开其他时间范围寻找更有利结果。

## 4. 只比较六条确定性策略

所有策略在同一个当前 Job 的四份真实材料/模型中选一份。无需重训。

1. **R_CA（现行对照）**：当前 Job C_A 三 seed 平均损失最低者。
2. **R_HistNear（时间对照）**：最近四个历史 Job 上模型使用早期反馈的等权平均最低者。每个历史 Job 使用四个起点 t_h+0/48/96/144，即历史 C_A+C_B。
3. **R_HistLate（主候选）**：同四个历史 Job 上模型使用后期反馈的等权平均最低者。每个历史 Job 使用四个起点 t_h+192/240/288/336。
4. **Fixed_initial**：只在 V1/T2/T4/Q2 上按 R_HistLate 的同一公式选一次，然后对 L1–L7 始终使用该方案。不能在看完七批后再挑一个最好的固定方案。
5. **Fixed_NoMix**：始终 P_NoMixRecipe，保留已知强参照。
6. **UniformRandom_expectation**：四公共方案均匀随机选取的期望 E，直接对四份真实结果平均；不抽取一次随机数冒充稳定基线。此项不是旧包 RandomSearch_B4。

None 另报为归一化参照，不单独训练。所有有选择的策略均按公共顺序打破绝对差 <=1e-12 的平局。不得引入 SE 门、3/3 门、阈值搜索、当前 C_A 与历史反馈的混合权重或观察字段路由。

R_HistNear 与 R_HistLate 使用完全相同的四段历史、模型、候选和起点数量；主要区别是模型训练后被评价的时段。它们相对 R_CA 同时改变历史覆盖和反馈数量，不能把此对比的全部效果唯一归因于模型使用时段。

## 5. 评分定义

令 L(h,m,s,b) 为既有缺失感知 normalized_mse_macro，b 为 early 或 late；m 为材料，s 为 seed。

- late：沿用该历史 Job 的四 E 起点评分几何。
- early：合并 C_A 和 C_B 的四起点后使用同一 scorer。不能直接把两个块的 macro MSE 各乘 0.5，除非每实体两块观测数相等。
- 可直接从已有 score 的 per_entity_normalized_mse 和 observed_cells_per_entity 恢复每实体 SSE，合并 SSE 与观测数再做宏平均；保留各 origin 的诊断读数。若缓存缺这些字段，先报告，不能补拟合。
- 当前 E、历史 early/late 均保持原 roster、T scaler 与原始观测掩码；候选共用分母和真值，不删极端 origin。

历史策略统一：

`J_b(j,m) = mean over h in H(j) [mean_s L(h,m,s,b) / mean_s L(h,None,s,b)]`。

分母非正、某方案不完整、任何固定历史块不满足原覆盖规则，则对应选择记 INCOMPLETE；不换时窗、删站点、改阈值或只在剩余候选中选。各对比报告实际完整的配对分母，缺项不填 0。

主收益口径：

`Delta_j(A,B) = 100 * (mean_s E_j(A) - mean_s E_j(B)) / mean_s E_j(None)`。

正值表示 B 更好，单位为相对该 Job 的 None 平均 E 的百分点，不称相对 A 的百分比提升。

配对 seed 向量使用同一个 Job 分母：`100*(E_j,s(A)-E_j,s(B))/mean_s E_j(None)`。点估计和 SE 必须基于同一向量；不混用均值之比与比值均值。三 seed 区间只描述训练随机性，不描述跨时间泛化的不确定性。

## 6. 执行顺序与信息隔离

A. 核对指定缓存、当前环境与父包依赖记录，冻结本任务配置和全部选择公式。不导入或启动原包的完整 stage/pilot/Runner，不创建 LLM 客户端。

B. 一次最小 smoke：检查顺序历史名单与未来 Job 拒绝；含缺失的四 origin 合并得分与直接 scorer 一致；选择函数不接收当前 C_B/E。用一份原模型重算已有 C_A 验证推理接线与原缓存一致（容差先冻结，不能看结果后放宽）；0 优化器更新。

C. 在新目录补算初始四批的 historical_late。每批 4 材料 x 3 seed =12 份模型，总计至多48份模型推理。先冻结这些历史预测，再评分，记录新曝光范围。不得调用会把分数写回旧 cell 的原 worker_label。

D. 按 L1→L7 顺序回测。对每批先只提供合法历史和当前 C_A，保存六策略中有确定选择的交付；之后独立 scorer 加载该批现有 E 并计分。该批的 E 只从下一批开始有资格成为历史；对应历史表仍须严格满足时间截止条件。Fixed_initial 在 L1 前冻结且不更新。

E. 一次汇总并停止。不得根据结果换 K、换初始历史、改聚合、增加候选或自动开 LLM 包。

历史后期反馈在本包控制器中的使用不授权原始 Source Episode 或跨 Job 分数银行直接进入 Fast Prompt。若后续研究 Skill，仍遵守 Source→Slow→冻结 Skill 的知识通路；若做候选回测工具，应明确它使用的合法 held-in 范围。

## 7. 预算与工程范围

- 新模型拟合 0；优化器更新 0；实验 LLM/API/HTTP 0；新增 SHA 0；git commit 0。
- 最多48份模型的历史后期推理，外加1份模型的既有 C_A 接线推理；复验仅用缓存评分。
- 数值运行硬上限30分钟，单 worker。超过即保留部分事实停止，不静默扩预算。
- 使用父包兼容环境；保留现有 OpenMP 初始化做法，不设置 KMP_DUPLICATE_LIB_OK 掩盖故障。控制进程不导入 torch。
- 缓存权重实际缺失或身份不匹配，不自动重训。本包不是另一轮训练诊断。
- 一个新研究脚本和一个输出根即可。旧包、AGENTS、Consumer、TempoPFN 参考仓库不改。
- 任何 smoke 不得连到 LLM、启动真实训练 stage；只执行明确列出的评分操作。

建议入口：`evaluation/main_protocol_p4/batch_research_tempo_aug_temporal_feedback_replay.py`。
输出根：`_scratch/dev_tempo_aug_temporal_feedback_replay/`。

## 8. 必须回答的四个问题

1. **主效果**：R_HistLate 相对 R_CA，七批 E 的等权配对差、逐批差、胜/平/负和最大伤害是多少？保留原始 E，不只报归一化数。
2. **是否超出默认**：对 Fixed_initial 和 Fixed_NoMix 是否有增量？如果全部同交付，明确增量为0，不把避免 C_A 错选写成动态选择能力。
3. **时段是否有用**：R_HistLate 对 R_HistNear 是否改善？若两者一样，不能说匹配后期时段被证明必要。
4. **好材料是否更多交付**：仅在同一四候选集合上记录所选材料 E 排名和事后 oracle 遗憾；报告原本 R_CA 错选的批次是否改善，有无新增错选。oracle 仅诊断，不回流选择。

汇总以 Job 为单位；不要把 7批x3seed 当成21个独立时间样本。附逐 origin 结果和留一 Job 汇总，展示是否由单个批次主导，不用本包小样本宣布反馈普遍有信息或无信息。

## 9. 判读与后续

- 若只改善 R_CA、却与固定默认同交付：说明历史默认能减少本次近端选优的损失，未证明动态研究增量。
- 若 R_HistLate 在多个批次改变交付、优于 R_CA 及固定对照，且收益不只由一个批次撑起：记有希望的开发证据，可据此设计下一次小规模 Fast/Skill 对照；不宣称当前已证明 Skill 有效。
- 若优于 R_CA 但逊于固定：保留固定默认为参照，不包装成成功选择器。
- 若不改善或方向混合：本包停止。不得为追正号换窗口、加字段、改提示词、补 seed；下一步需要重新讨论反馈目标或任务条件。

即使出现正向信号，仍欠两个检验：新构造程序能否在合法历史上重建并获得有用反馈；完整 Fast/Skill 能否在未用于开发的后续批次获得增量。这两项不在本包自动授权范围。

## 10. 交付

保留 `REPORT.md`、`result.json`、`selections.json`、`frozen_config.json` 和必要的新历史预测/分数。主报告首页依次回答 §8 四问，写清费用、时间、新开历史标签、缺项与暴露身份。

结果表至少包括：Job、策略、历史Job、选中材料、C_A、历史 early/late J、E逐seed、对R_CA/两个固定对照的差、E排名、遗憾。

只将最终执行回执追加到本任务书，不覆盖任何旧包结果。完成即停止，不启动下一实验。

## 11. 执行回执

待执行。本任务书生成时未运行推理、拟合、LLM，也未读取新增历史目标。

### 11.1 执行回执（Opus，2026-09-19 15:33 完成，追加式）

- 状态：**COMPLETE**。机械判读（§9 规则）：`DEFAULT_RECOVERY_ONLY`——R_HistLate 只改善 R_CA（+9.58 pp，3 胜 4 平 0 负，留一 Job [+6.49, +11.17]），但与 Fixed_initial（=P_NoMixRecipe）、Fixed_NoMix、R_HistNear 七批全部同交付，三项增量均为 0。未证明动态选择能力，未证明后期时段反馈必要。本包停止，未启动任何 Fast/Slow。
- §8 四问：(1) 逐批 Δ(R_CA→R_HistLate)：L1 0 / L2 +12.87 / L3 0 / L4 +26.08 / L5 +28.09 / L6 0 / L7 0 pp；最大伤害 0。(2) 对两个固定对照增量 0。(3) 对 R_HistNear 增量 0。(4) E 排名 R_CA 1,3,1,4,2,1,4 → R_HistLate 1,2,1,1,1,1,4；oracle 遗憾均值 13.54 → 3.97 pp；R_CA 错选的 L2/L4/L5 改善，L7 仍错选（所有策略共同），无新增错选。
- 费用：拟合 0、优化器更新 0、LLM/HTTP 0、新增 SHA 0、git commit 0；模型推理 49 次（48 份初始历史模型 E 推理 + 1 份接线复算）；数值子进程墙钟 18.4 s（上限 1800 s），单 worker，9 个子进程，未触上限。
- 新开历史标签（`historical_late`，开发反馈）：V1 [5952,6144)、T2 [8352,8544)、T4 [10752,10944)、Q2 [13152,13344)，各 4 起点 ×12 模型，全部 SCORABLE；不宣称全包 0 新标签。L1–L7 的 E 只读既有 e_scores.json。
- 接线：RD02_V1__None__s20269181 用重算 scaler/合法父窗口（与缓存逐位相同）复算 C_A，与缓存宏/逐实体差 0.0（冻结容差 1e-5）。smoke：历史表等于规则、60 条历史访问全部满足 t_h+384≤t_j、未来/同批/出窗历史被拒、四 origin 合并得分与直接 scorer 相差 ≤1e-12 且与 0.5/0.5 平均不同、合并后覆盖规则生效、选择函数无当前 C_B/E 入口。
- 缺项：无（六策略七批全部有确定选择）。曝光身份 `EXPOSED_DEVELOPMENT_REPLAY`。
- 入口：`evaluation/main_protocol_p4/batch_research_tempo_aug_temporal_feedback_replay.py`；输出根：`_scratch/dev_tempo_aug_temporal_feedback_replay/`（REPORT.md、tables.md、result.json、selections.json、frozen_config.json、budget.json、smoke/、historical_late/）。
- 独立核对：L2/L4/L5 的 R_CA 选择由原 cell 复算一致；L1–L7 的 NoMix E 比值与旧 field_readout.json 逐位相同；L1 的 J_late 由新分数文件重算一致。
