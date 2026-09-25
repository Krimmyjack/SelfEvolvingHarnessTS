# DEV-DOMAIN-AUG-ENTITY-SPLIT：两域 Skill 学习与新实体复用

日期：2026-09-19  
状态：COMPLETE，已于 2026-09-20 收口并停止；实际结果与偏差见 §10.1，原规格及预算修订记录保留，不自动启动后续实验。  
后续取代：DEV-TEMPO-AUG-DOMAIN-SKILL-OPTIMIZE 的旧 setting；旧包现已完成收口，结果与原执行记录保留。  
负责人：Opus 负责实现和运行；实验内 Fast/Slow 沿用现役模型，二者与开发者角色分开。  
分组定义：[DOMAIN_AUG_ENTITY_SPLIT_V1.json](DOMAIN_AUG_ENTITY_SPLIT_V1.json)。

## 1. 用户需求与这一包要回答的问题

用户确认尽快重新设计 setting：Skill 在同一域的一些序列/实体组上学习，再用于该域其他序列/实体组；Agent 构造有界但完整的数据增强方案，不围绕一个人工配方做开关选择。

本包一次完成：
1. 两域、实体互斥的 Source / Select / MetaTest 接线；
2. 从每域 8 个实际研究案例形成多种 Workflow，开发选优；
3. 在新实体组比较域 Skill、无卡、通用指导、随机搜索与固定增强。

研究对象是“学习并复用准备数据的方法”。NoMix 是共享候选和强固定对照；不设“无卡先赢 NoMix”的门，也不为了出效果删除 NoMix。旧的切窗研究和局部编辑结果保留，不能与本包直接相减解释成学习收益。

本包属于开发级域学习组件，不替代项目完整跨域积累/Target 校准与 Natural Final。形式化多轮进化、画像路由、TSFM、训练协议调参不在本包。第一轮先让新的问题定义与真实实验对应上，不另插诊断包。

## 2. 两层切分：实体是外层边界，窗口是内层样本

外层用于学 Skill，内层用于训练和评价 Consumer：

| 外层集合（每域） | 不同序列数 | Agent 案例数 | 用途 |
|---|---:|---:|---|
| Source | 128 | 8 | 无卡研究经历及后期效果，供 Slow 学习 |
| Select | 32 | 2 | 三张候选卡实际执行并选优 |
| MetaTest | 32 | 2 | 冻结卡用于未参与本轮学习的实体组 |
| 合计 | 192 | 12 | 三集合实体完全互斥 |

每个案例是 16 条序列共同形成的一批训练材料，训练一个共享预测模型；不是把 16 个独立模型的预测拼起来。可按观察对这些序列赋不同程序。128 条 Source 序列对应 8 次研究经历，不能声称 128 次独立 Agent 案例。

具体实体名单、案例 ID、时间切点已写入同目录 JSON。策划阶段只读取 CSV 表头和两个 T 段：
- Electricity 文件 321 个数据列，去除不明确命名的 OT 后，319 列在两个 T 段都合格；
- Traffic 文件 862 个数据列，去除 OT 后，861 列合格；
- 合格条件：两个 T 段各 672 点全部有限、标准差 >1e-6；
- 字符串排序后用固定 Python Random 种子打乱，前 192 列依次分组。D01/D02 的种子为 2026091901/2026091902；运行读已保存名单，不重新随机分组。

两域使用各自的本地 TSLib CSV：
- shared_tsq_datasets/electricity/electricity.csv；
- shared_tsq_datasets/traffic/traffic.csv。
JSON 记录了实读路径。Windows/WSL 路径表示可以转换，数据内容、实体名单不可变。

原始列是当前可用的实体标识；同一列及已知别名不得跨集合，不把多变量的一台设备拆成不同集合来扩大样本数。没有足够物理实体元数据时说明识别限度，不新增长周期审计。

### 2.1 时间设计

每域 S01/S03/S05/S07、V01、Q01 用 t=4032；其余 S02/S04/S06/S08、V02、Q02 用 t=8064。Source、Select、MetaTest 都覆盖这两个时间位置，避免把“新实体”和“新季节”捆成同一个变化。

每个案例仍使用：
- T = [t−672, t)，从中生成 192 输入 + 48 目标的合法训练窗口；
- C_A origins = t、t+48；
- C_B origins = t+96、t+144；
- E origins = t+192、t+240、t+288、t+336。

先确定外层实体名单和时间边界，再切训练窗口。不能先生成重叠窗口再随机拆进 Source/Select/MetaTest。

本包是**离线学 Skill、在新实体任务上进行有预算的本地校准**：MetaTest 中 Fast 可以用该案例的历史 T 和支持块 C_A 做实验，E 始终不见。C_A 可以理解为在提交时已收集的历史验证数据，案例决策时点为 t+96；每个 E 起点的预测输入也只能使用当时已观测过去。

外层 MetaTest 不是 AGENTS 中的 Natural Final/零反馈 held-out：两者名字和权限必须分开。学得 Skill 在 MetaTest 全程冻结，Slow 无法接收 MetaTest 的任何反馈；Consumer 在新实体自身 T 上重训是任务的一部分，不是重新训练 Skill。

### 2.2 暴露身份

统一写 SERIES_DISJOINT_DEVELOPMENT。两个数据源曾参与项目历史开发；本轮实体隔离并不使它们变成从未曝光的独立终验集。

本轮实验模型不读取其他历史包的轨迹、效果表、NoMix 成败总结或人工针对某域写的答案。Source 从这次指定的 16 个案例重新产生。允许复用代码与冻结原语，不复用旧模型冒充新实体模型。

不读取 PRSA/KDD、Natural Final、UCR TEST 或其他封存区，不新增下载。剩余列不追加进本包，不根据效果换人群或切点。

## 3. 动作空间：恢复构造完整程序

公共工具保持 overview、inspect_data、build_material、inspect_material、evaluate、compare、commit。

每个实体可使用：
1. identity：不增强；
2. 七个 TempoPFN 来源原语组成的 1–3 步程序；
3. 公共 NoMixRecipe；
4. NoMixRecipe_Edit：关闭 0–2 个组件，作为局部编辑选项。

七原语为 regime、shock、calendar、amplitude、resample、censor、random_conv。原版参数分布、固定执行次序、互斥对（regime/shock、calendar/amplitude）与不重复规则沿用。预设或编辑是一个完整程序，不与额外原语拼接。默认程序 + 最多 8 条合法观测谓词规则可以形成条件化赋值。

这次**不新增强度档位、概率搜索、任意代码执行、采样权重或新算子**，但不得再把显式 1–3 步构造挡在接口外。NoMix 没有强制起点地位；Fast 可以直接从原语构造，也可以保留公共方案。

每个方案处理历史的 240 点联合窗口，child 的输入和历史训练目标一起增强；parent 不变，损失 parent/child 各 0.5。预测输入与评价真值不增强。所有新材料共同训练同一个 Consumer，不能拼接别的方案模型。

RNG 用显式的 domain、case_index、实体身份和规范化程序语义绑定；不依赖分支名或构造顺序。NoMix 编辑继续共享同实体 NoMix 父随机流，关闭不挪动其余步骤的抽样。复用既有 program 索引和 SeedSequence，不建哈希。新分组是新材料身份，旧缓存不能跨人口复用。

四个公共参照仍是 None、FixedMixup、P_AmpResample、P_NoMixRecipe。FixedMixup 只作既有固定参照，不重新开放 donor/Mixup 搜索。

## 4. Consumer 与共同 Fast 契约

沿用现役 MLP 192→128→64→48、AdamW、2000 updates、parent_batch=64、3 个 seed [20269181,20269182,20269183]、T-only 每实体标准化。实际 n_pool = 16×433；用真实数组长度生成抽样，不沿用旧 32×433 的常数。任意同案例方案的训练随机流配对。

不把拟合长度、正则或模型参数交给 Agent 调整。更换人口后的结果不能与旧人口模型直接比较；本包所有基线与 Agent 共享同一 Consumer。

每条 Fast：
- 最多 4 个新完整方案、16 次 LLM 请求、24 次工具调用；
- 250,000 总 token，单次输出最多 12,000；
- 公共方案的既有 C_A 可见，不额外扣新方案槽；
- 不强制用满名额、分组、三步或偏离 NoMix；
- 可依据反馈组织下一次实验；允许提交任一自身已评估方案或公共参照，Runner 不覆盖为 argmin。

系统提示说明“优化整批后续预测效用；先按数据和处理语义提出假说，检查材料，再比较真实拟合；C_A 是有限支持，不能保证未来排序”。不预写哪个域应选哪个算子，不要求必须显著、必须 3/3。域 ID 只用于正确加载卡，不能查表指定增强。

全臂工具、共同 prompt、预算相同，仅 guidance 不同。原始跨案例 Episode 只给 Slow；Fast 只读本案例合法工具结果和冻结卡。

实验模型仍用现役 cpa-grok-4.6、返回 grok-4.6-build；不静默换模型。开发 Opus 不作为实验模型。环境沿用最近已验证版本，不重新装依赖。

## 5. 基线与主要比较

| 臂 | 行为与证据角色 |
|---|---|
| F0 | 同工具无 Skill 的 Fast；主消融 |
| F_domain | 同域 Source 学得且经 Select 选中的 Workflow+可选 Principles |
| F_generic | 下面冻结的常识性研究指导；控制“仅多一段过程提示” |
| RandomSearch_B4 | 同一动作空间、四个新方案、相同 C_A 反馈，0 LLM |
| Fixed_dev | 该域 Select 选出的统一固定程序，MetaTest 不搜索 |
| None / FixedMixup / AmpResample / NoMix | 固定参照，已在公共评分内；强者保留 |

主读数是 F_domain−F0；另报对 Generic、Random 和强固定方案的差。NoMix 不是唯一成功判据，但不能仅打赢弱对照就声称 Agent 增量。

Generic 正文在首个实验调用前冻结，所有域相同，不看任何 Source/Select 结果：
> 读取任务、Consumer 和全批观察，找可能影响预测的结构；必要时检查代表片段，区分真实变化与处理伪影。用合法原语构造能区分假说的完整方案，检查实际材料变化，在预算内取得下游反馈。根据已有证据决定继续、保留公共方案或提交已评估方案。处理更复杂或改动更大不代表更好；有限支持块的优势不保证后期优势。

Generic 不是跨域学习卡；本包不能仅由 F_domain 胜 Generic 宣称“域划分必不可少”。跨域混合形成的共享 Skill 是后续消融，不混进本包。

### 5.1 随机搜索的固定采样法

首个新拟合前，用当前 T 与独立 policy seed 固定四个完整政策，不看 C_A/C_B/E：
- R1：统一显式 1–3 步合法组合；
- R2：统一 NoMix 编辑（关闭一或两个组件）；
- R3：一条合法观测条件规则，默认/规则分支从全部合法程序中抽；
- R4：第二个条件化方案，抽法同 R3。

组合和关闭集合在规范化合法表内等概率；条件字段从本案例有变化的合法 T 字段中抽，方向等概率，阈值用中位数；规则必须在本案例实际分开至少两个实体。最多 20 次去重尝试，失败则按冻结表序取第一个尚未使用的统一方案。不因效果重新抽样。

与 Fast 相同，实际 evaluate 四个新方案。最后在本臂真实评估集合加公共参照中取 C_A 三 seed 均值最小者；平局依公共顺序、R1…R4 破同。条件化只用已有谓词实现，不写新 Router。

### 5.2 文献对照的后续位置

TSAA（Data Augmentation Policy Search for Long-Term Forecasting，TMLR 2025）有公开实现，可作为下一阶段外部方法对照：
https://arxiv.org/abs/2405.00319
https://github.com/azencot-group/TSAA

本包不安装或简化复现 TSAA，也不把普通贝叶斯搜索标成 TSAA。先完成本项目自己的无卡/随机/固定公平比较；对外方法比较另按其完整训练语义和资源核定。

## 6. 连续执行步骤

### A. 接线及一次 smoke

新增一个薄 study 与必要 adapter，不复制整套训练器。重点复用：
- batch_base/spec.py、data.py、context.py、train.py 的 dense CSV、训练、评分；
- tempo_aug.py / tempo_source.py 的数值原语和组合，recipe_edit_pilot 的编辑语义；
- domain_skill.py、batch_research.py 的 Knowledge、Fast run_job、Slow/选优；
- 已有账本、错误恢复和阶段评分屏障。

需要的真实改动：
1. 显式 CaseSpec(dataset, cohort, t, role)，贯通任意固定 16 实体人口。不要改全局 ROSTER_SIZE 或 RD02 注册去冒充新组。
2. dense CSV 只转换本案例已授权列/行的数值；不要因旧 loader 全列加载把其他集合带进画像、scaler、材料或请求。scaler 绑定 domain/cohort/t；恢复、子进程、cache 和标签阶段同样绑定。
3. 通用构造与编辑都通过同一 adapter；显式组合不能被 EditAdapter 的限定 schema 拒绝。保持旧调用默认不变。
4. 实际 n_pool 和 entity_count 贯通；每份候选是本组共同拟合，不复用旧不同人口权重。
5. Source/Select 的后期分数可在阶段结束后供学习/选卡；MetaTest 分数不回流。

一个 smoke 只覆盖这些新增路径和本轮标签隔离。使用脚本客户端，不调用 LLM；真实接线最多 12 拟合（两域各 None 和一份显式组合，3 seed），使用 Source S01 的数据，独立 wiring 目录、C_A only，健康缓存可在 Source 复用。不因 smoke 触发真实阶段 worker，不重跑全部历史测试。

接线通过后即按以下阶段运行，不先做效果普查。普通技术修复在预算内自行完成；只对实质性数据边界、方法/预算变化或未知费用暂停受影响工作。

### B. Source：两域各 8 个真实案例

每个 Source case 跑一次 F0，无手写域卡。每域各 8 个案例全部 commit 后统一 C_B→E 预测冻结→E 评分；E 为明确授权的开发后期监督，记录所有已评估方案，不能只留提交赢家。

Slow 的每域 census 保留：
- 案例 T 观察及被实际检查的片段/摘要；
- 完整程序、分组条件、实际材料变化；
- 逐 seed C_A/C_B/E 与 commit；
- 不同候选的反例、成本与失败；
- 事后池最优仅标成诊断，不能当部署已知答案。

压缩重复工具文本，保留语义、数字和引用；不把其他域或 MetaTest 的统计拼入该域输入。来源数写“8 个实体组案例”，不能把 16 条实体的局部损失当 16 次独立因果测量。

### C. 每域生成三个候选 Workflow

每域一次 Slow 形成调用，输出最多三张候选 W1/W2/W3（无可用候选则说明理由并记 NO_CARD），每张 <=6000 字符，Workflow 和可选 Principles 一起给出。最多一次格式纠错，不按效果重新采样。

Slow 要学的是：什么观察有助于提出处理假说、怎样构造对照、如何使用当前反馈和历史经验、何时继续或提交；不是强制挑一个常用配方，也不是强制写复杂条件。输出主要规则的 support/counter 与 supported/hypothesis/unresolved，不能继承旧包无依据禁令。

可以提出不同调查顺序或候选优先级；不以文字差异冒充实际决策差异。实体 ID、案例编号、原点到程序的查表禁止进入正文。没有可用候选如实记 NO_CARD；不能用 Generic 补位冒充学习成功。 若某域没有卡，F_domain 明确别名到该域 F0，复用相同轨迹与分数，不额外抽样生成一个“有卡”结果。

### D. Select：每域两个互斥实体组

每张合法候选在该域 V01/V02 上各跑一次完整 Fast。所有候选 commit 后统一打开该域 Select 后期效果。

J(W) = 两个 Select case 等权平均 [mean_seed E(actual_commit_W) / mean_seed E(None)]。

选 J 最低的候选；差 <=1e-12 时按少新评估、少 token、W1/W2/W3 顺序。选卡不使用自身池 oracle；不按“必须胜过 NoMix/显著胜无卡”拒掉完整 MetaTest 比较。无卡不是本阶段新跑的选项；最终与无卡的增量在 MetaTest 测。

Fixed_dev：在两个 Select case 都实际评估过的统一程序交集中，按同一 J 选一份。四公共方案保证有交集。未在某 case 评估的私有方案不能用其他 case 成绩代填；不为选固定对照额外扩拟合。冻结的是程序，不是 Select 模型或实体赋值。

域卡、固定程序、Generic 与 MetaTest 配置全部冻结后再进入 MetaTest。首包只做这轮生成和选优，不额外做无独立选优集的事后 Principles 修订。本轮检验域经验学习与复用；持续多轮进化以后在同一 setting 内扩展。

### E. MetaTest：两域各两个新实体组

每个 Q case 跑 F0、F_domain、F_generic、RandomSearch_B4。三 Fast 用同 prompt/工具/预算；按预先循环顺序平衡臂执行次序。卡整个阶段不更新。

Fixed_dev 按冻结程序应用本案例，若与公共/已评估程序别名则缓存复用；否则每 case 最多 3 拟合。公共固定参照同场评分。

全部四个 MetaTest case 的所有输出冻结后，统一 C_B→E 预测冻结→E 评分。Slow 不再运行，Q01 的结果也不反馈给 Q02。技术失败与合法 None 分开；不看效果加 seed、换人群、扩大算子或启动第二包。

## 7. 冻结资源上限

3 seed 全阶段一致；缓存只在同案例相同材料语义下复用。

| 阶段 | 最多 Fast | 最多物理拟合 | 模型请求帽 | token 帽 |
|---|---:|---:|---:|---:|
| 接线 | 0 | 12 | 0 | 0 |
| Source：16 case，每个 F0 | 16 | 384 | 256 | 4,000,000 |
| 形成：两域，含各一次格式纠错 | 0 | 0 | 4 | 700,000 |
| Select：4 case × 3 卡 | 12 | 192 | 192 | 3,000,000 |
| MetaTest：4 case × 3 Fast + Random + Fixed_dev | 12 | 252 | 192 | 3,000,000 |
| 同配置拟合技术重试预留 | 0 | 6 | 0 | 0 |
| **总上限** | **40** | **846** | **644** | **10,700,000** |

拟合计算：每 case 四公共×3=12；每条 Fast/Random 四新方案×3=12；MetaTest 的 Fixed_dev 额外最多 4×3=12。跨臂相同材料会降低实际拟合，不因此增加搜索槽。

一个抗中断总账本；失败尝试也计费。HTTP 总帽 648，最多 4 次额外传输尝试、每请求最多额外一次；未知 usage 暂停新增付费请求并报告，不能按零费用自动恢复。格式纠错不变成科学修订。无未知费用的普通中断可原状态恢复，不能重生成已冻结卡。

付费运行墙钟上限 8 小时。按既有模型耗时，拟合上限约 2–4 小时，另加模型请求与材料时间；实现目标一个工作时段。以实测报告，不能承诺固定完成时刻。超过预算如实收口，不转移结余扩大阶段或删掉不利臂。

本包比旧的两批局部复验大：这是增加独立实体案例和必要基线的成本，不沿用原 180 次拟合估计。用户转发并要求执行本文即按此完整预算连续完成；本文件产生不代表任何调用已启动。

## 8. 读数：效果、学习贡献和成本

主指标使用本案例 T 标准化后的 E MSE，先实体宏平均，再三 seed 平均；主汇总先域内两个 case 等权，再两域等权。不要让 Traffic 的总列数改变域权重。

Delta_j(A,B) = 100 × [E_j(A)−E_j(B)] / E_j(None)，正号为 B 更好；每个 seed 的差都用同一个该 case None 均值作分母。

首页给出：
1. 每域学到了什么卡，两个域的行为内容是否有实际差异；差异不是成功门。
2. F_domain 对 F0、Generic、Random、Fixed_dev、NoMix 的逐 case 差及域内/总体均值。
3. 新实体中，卡改变了哪些观察、候选和提交；实际 commit 与自身池 C_A argmin 是否相同。
4. 新候选提供的事后机会和选择损失，标 oracle，不等于可实现收益。
5. 一次性形成成本、每 case 部署成本、真实拟合/缓存命中与请求数。

保留逐 seed、逐实体、逐起点分数和普通配对 SE；案例组是重复单位，seed 不是独立任务。小样本只作开发证据，不设置显著性准入门，也不从不显著推断严格相等。

如果域卡只复现域内固定程序，就报告“学得稳定处理偏好”；若动态研究相对静态有增量，再报告对应过程证据。若只赢 None，不归为 Skill 增量。固定策略赢了也照实保留，不因此永久关闭 Agent 学习方向。

## 9. 交付与禁止扩包

建议入口：
evaluation/main_protocol_p4/batch_research_domain_aug_entity_split.py

单一输出根：
_scratch/dev_domain_aug_entity_split/

交付：
- REPORT.md：setting 示意、实体数量、完整实际卡片或可读节选、主结果和具体行为例子；
- result.json / tables.md：真实分数、别名和缺失值；
- frozen_config.json / budget.json 与必要分阶段工件；
- 分组 JSON 的原样拷贝与使用检查；一个必要 smoke。

优先复用现有模型与原语；可增加显式 cohort/CaseSpec 的 opt-in 参数，不覆盖旧实验、默认人口、历史工件。保持参考仓库只读，不 git commit，不新增 SHA/Hash，不建通用路由/数据审计平台。函数真实不兼容时修本轮所需路径，不顺带重构所有旧 runner。

这包以两域学习及四个新实体组的开发复验收口。TSAA、额外域、更多 MetaTest、画像 Router、TSFM、Natural Final 均不自动启动。

## 10. 实际执行回执

待执行。执行者完成后追加真实状态、卡片选择、效果、费用、故障和剩余问题。

### 10.1 执行回执（Opus，2026-09-20 02:40）

- 状态：**COMPLETE**。接线 PASS（12 拟合）→ Source 16 案例全部 COMPLETE → 两域各形成 3 张卡（D02 一次格式纠错）→ Select 12 分支 COMPLETE → 冻结 → MetaTest 20 分支 COMPLETE → E 读数。0 拟合失败、0 重试。报告：`_scratch/dev_domain_aug_entity_split/REPORT.md`；数字：`result.json`、`tables.md`。
- 修订 1（用户指示“预算不卡上限”）：§4 的单条 Fast 250,000 token 帽在 D01_S01_f0 第 8 次请求前阻断（7 请求 295k prompt token、未 commit；请求载荷 76–136 KB），执行者暂停上报后按指示放开：单条 Fast 1,000,000、阶段/总 token 帽 ×3、墙钟 16 h；拟合帽 846、请求帽 644、工具/方案预算、提示词、seed、数据、顺序、选卡规则不变。D01_S01_f0 从未发出的请求处恢复，D01_S02_f0 重启。记录 `amendment_1_budget_not_binding.json`。
- 事故 2：Select 第 161 次请求首个 HTTP 尝试瞬态 500（重试成功）触发未知 usage 停机；执行者接受该 1 次未知 usage 恢复时误起两个进程约 3 分钟（2 个响应文件被覆盖、两条 W3 轨迹重跑；未开任何标签）；账本按磁盘工件重建（`ledger_reconciliation_1.json`）。记录 `operator_incident_2_select_resume.json`。
- 卡片选择：D01 三卡 J 并列 1.0154（V01 同交付 Edit[-random_conv]、V02 同交付 Edit[-shock]）→ 按更少新评估选 W1（“public-NoMix default; skip mild primitives”）；D02 三卡 J 并列 0.9793（V01 FixedMixup、V02 P_NoMixRecipe）→ 选 W2（“C_A-gate publics then one family”）。Fixed_dev 两域均为 P_NoMixRecipe（J 0.821 / 0.952）。D02-W3 正文含案例编号“(S06,S08)”（未选中；文本检查缺口，记为偏差）。
- 效果（E，pp of None）：F_domain − F0 = **+0.74**（D01 +2.24 / D02 −0.75；逐案例 0 / +4.48 / 0 / −1.50，两例与无卡同交付 P_NoMixRecipe）；− Generic +0.74（Generic 与 F0 4/4 同交付）；− Random +1.66；− Fixed_dev = − NoMix **+0.01**（1 胜 1 负 2 并列）；− None +10.88。F0 − None +10.13、Random +9.22、NoMix +10.87。所有 Fast 提交 = 自身池 C_A argmin（12/12）；卡的作用是改变评估集合（新评估 7 vs 14、token 0.81M vs 1.76M），一次有益（D01_Q02 Comp[censor]，+4.5）、一次有害（D02_Q02 Edit[-resample-conv]，−1.5）。判读：INFRASTRUCTURE 完成；学习贡献 INCONCLUSIVE/WEAK_MIXED；固定预设 P_NoMixRecipe 仍是最强单一策略，域卡与其持平。
- 费用：609 拟合（帽 846）、309 LLM 请求（帽 644）、15.10M token（任务书 10.7M 已按修订 1 放开）、paid 墙钟 4 h 17 min（22:19→02:36，含约 6 min 操作者停止）；拟合工人累计 9586 s（每 seed 一子进程、3 并行、固定 8 线程）、材料 2691 s、标签 382 s；形成 3 次 Slow 请求 0.60M token。
- 剩余问题：Select 无法区分三卡（同交付）；D01 卡在 Select 的 J>1；文本检查需补本包案例编号模式；无 git 提交、无新增 SHA；TSAA/多轮进化/更多域未启动。
