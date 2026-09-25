# DEV-DATA-READINESS-DOMAIN-SKILL-V1：直接迁移与最小完整试点

日期：2026-09-17。Planner：Astra；开发与执行负责人：Opus。
状态：交用户转发的执行规格；本文件写入时未启动实现或付费运行。
项目：SelfEvolvingHarnessTS-deepseek-guidance-evolution。
输出根：`_scratch/dev_data_readiness_domain_skill_v1/`，一个包、一份主报告。
前包：`_scratch/dev_domain_skill_v1_evolve_pilot/`，保留原样。

## 0. 决定、授权范围与交付终点

**直接迁移任务环境，保留当前批级 Harness 和 per-domain Skill 生命周期。停止追加增强卡片试验。**

用户当前要求：尽快完成可用项目，以数据就绪为后续主应用；增强只作已完成的小规模方法试点。本包复用批级 Fast、共享 Consumer、7 工具、Slow 形成、真实 Select、freeze、延迟评分和账本，不重建项目，也不退回旧逐序列广播式驱动。

设计沿用 [Eval-Skill](https://arxiv.org/abs/2606.07040) 的离线竞争 Workflow、真实选择和冻结复用思路；
本包是时序数据就绪迁移，不声称复现原论文完整规模。

本包连续完成：
1. 两套本地自然缺失数据接入、批级输入准备工具和可运行训练/评分；
2. 真实无 Skill Source 轨迹；
3. 每个数据域最多两个 Workflow，真实比较并冻结或保留 NO_SKILL；
4. 两个后续批次上的无卡、通用指导、域卡、随机搜索对照；
5. 结果、限制、可复跑入口和下一阶段计划。

实现、必要测试和预算内运行属于同一任务，不在每个小修补后要求用户重新批准。不得只交 BUILD_COMPLETE 或一份调查报告。遇到具体泄漏、实质性超预算、数据身份不符或无法保持本文件干预几何时，暂停受影响部分并说明事实；普通接口问题自行解决。

这是自然数据上的 development MVP。不是正式密封终验，也不是完整 A3/A5 持续更新课程。首包以“离线学出域 Workflow → 后续批次冻结复用”为方法切片；它的完成不替代 AGENTS 的最终系统要求。

## 1. 为什么迁移：自然空间的证据与边界

已核对的本地事实：
- `AGENTS.md §8.1` / `artifacts/main_protocol/p4d_natural_gap_roster.json`：
  KDD **with-missing** 版共 2,942,364 点，其中 503,712 点缺失，约 17.119%；
  旧 `data/kdd2018/series_cache.npz` 是 without-missing 版，NaN 为 0，不能拿来做本包输入。
- `AGENTS.md §5.1` / `docs/P4D_NATURAL_GAP_LINE_CLOSURE_2026-09-01.md`：
  旧 Ridge 设置中，周期插补与异常处理的组合在一个评价窗口出现双面正收益，但仅少数窗口成立，
  不支持稳定条件化或本包 MLP 必然获益。六个旧 origin 共用训练语料的更正也必须保留。
- `docs/DEV_TRAIN3_CROSS_JOB_HARNESS_RESULT_2026-09-11.md`：
  北京空气质量自然数据上曾出现准备收益，主要由一个作业承载；Skill 增量未成立。
- 官方资料也区分含缺失/已填补版本；北京 12 站数据明确用 NA 表示缺失：
  [Monash 数据目录](https://forecastingdata.org/)；
  [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata)。

**判断：自然缺口与局部准备收益确实存在；“空间比增强更大”“Agent 能拿到增量”仍是本包要回答的问题。**
旧正结果的 Consumer、X/y 处理与服务几何也不同，不能直接拿其效应量估计本包收益。
异常峰值可能是真实污染事件，不自带“应删除”的标签。不能把曲线更平滑、缺口填满、改动更多当作下游收益。
本包不注入缺失/噪声制造主结果，不重新用旧六 origin 拟合风险树，也不把老报告的赢家写成正确 Skill。

## 2. 数据、批次与资格：已做只读 T 检查

### 2.1 两个数据域

- RD01：本地 KDD Cup 2018 **with missing values** 原始 ZIP：
  `data/kdd2018/raw/kdd_cup_2018_dataset_with_missing_values.zip`。
  可参考 `preflight_natural_gap_variant.py` 的解析方式，但不能直接调用会把整条未来数值载入内存的 loader；
  本包只数值化当前阶段授权切片。该路径可能是共享数据软链接，保持只读。
- RD02：Beijing Multi-Site 的 **PM2.5**，12 个实际站点：
  `_scratch/train3_preflight/data/prsa/PRSA_Data_20130301-20170228/`。
  路径缺失可从本地 `data/benchmark_v0/raw/beijing_multisite/beijing_multi_site_air_quality.zip`
  在本包目录解压同源文件；不另下新数据。
- RD01 为 32 条序列，RD02 为 12 个站点。让新 readiness adapter 支持实际 N，
  禁止复制站点凑 32。原增强 profile 默认 32 及旧数值路径不变。
- 两域都是空气质量相关数据来源，不将两个 dataset ID 写成跨行业泛化。
  此轮用 known-domain 加载；数据源 ID 不是 Skill 正文的适用理由。
- 全部标为 `EXPOSED_DEVELOPMENT`。不读 Natural Final、UCR TEST 或其他密封材料。

### 2.2 冻结批次

所有索引零起点，左闭右开；L=192、H=48、T=672 小时，不继续用容易误解的 a85/a95 名称。

| 阶段 | Job 后缀 | t | T | C_A 真值 | C_B 真值 | E 真值（仅 Target 评分） |
|---|---|---:|---|---|---|---|
| Source | S1 | 3360 | [2688,3360) | [3360,3456) | [3456,3552) | 不评分 |
| Source | S2 | 4560 | [3888,4560) | [4560,4656) | [4656,4752) | 不评分 |
| Select | V1 | 5760 | [5088,5760) | [5760,5856) | [5856,5952) | 不评分 |
| Target | T1 | 6960 | [6288,6960) | [6960,7056) | [7056,7152) | [7152,7344) |
| Target | T2 | 8160 | [7488,8160) | [8160,8256) | [8256,8352) | [8352,8544) |

CA origins=t,t+48；CB=t+96,t+144；E=t+192,t+240,t+288,t+336。
KDD 为各原始序列内的行索引，保留每条实际起始时间；不得伪造全体相同日历日期。
RD02 使用文件真实时间列，核对小时连续性。Job 为 RD01_S1 等中性标识。
所有计划 T 只允许用于资格/观察；Target T 不进入 Source census、Slow 或 Select。

### 2.3 人口冻结

Planner 本轮已做 0 拟合、0 LLM、仅数值化上述 T 的检查：
初版只用 Source T 选人口会使 RD01_T1 某实体没有完整监督窗口；已在任何本包 C/E 读取前，
改成以下**全计划 T 的结构资格规则**。这是有资格筛选的数据设置，不是未筛选总体。

RD01：
1. 原始 UID 按字符串排序，从下标 80 起的候选池检查；
2. 每个计划 T 至少 336/672 个有限原始值、原始有限值 std>1e-6；
3. 每个计划 T 至少 32 个父窗口同时满足：192 点 X 有至少 32 个有限值、随后 48 点 y 全部有限；
4. 按排序取前 32 个；仅按 T 可训练性选，不按任何模型分数或缺失严重度排名。

只读所得冻结 roster：
`T173,T175,T177,T179,T180,T187,T189,T19,T191,T192,T193,T195,T197,T198,T199,T201,T203,T204,T205,T207,T209,T21,T210,T211,T212,T213,T214,T215,T216,T218,T220,T221`。

RD02 按站名排序使用全部 12 个 PM2.5 站点，不混入其他污染物凑人口。

| t | RD01 T 平均缺失率 | RD01 每实体最少可用父窗 | RD02 T 平均缺失率 | RD02 最少可用父窗 |
|---:|---:|---:|---:|---:|
| 3360 | 11.8769% | 83 | 4.1171% | 171 |
| 4560 | 22.1819% | 40 | 0.6076% | 384 |
| 5760 | 6.3105% | 70 | 1.0789% | 292 |
| 6960 | 3.6923% | 55 | 0.2232% | 286 |
| 8160 | 9.2448% | 95 | 2.2073% | 161 |

上述是当前本地 T 复算，不是未来分数、异常标注或插补可获益比例。没有读取本包 C_A/C_B/E 数值。
Opus 用同源文件复算资格和人口一次；不一致查数据绑定，不静默换 roster/时间。某域结构失败不妨碍另一域继续。

## 3. 本包干预几何：先把一个明确的问题做完

本包准备的是**模型的历史输入**，保持监督值的原始语义：
- Fast 对整个 N 实体批次生成 Workflow；可按 T 特征分组，也可全体相同。
- 每个候选使用同一份原始 T、同一人口、同一合法父窗口集合。
- 原始父窗口从 T 内 stride=1 构造，X=192、y=48；统一仅保留 X 有至少 32 个有限值且 y 全有限的窗口。
  所有臂共用该集合，候选不能挑删训练样本、改权重或补造 y。
- **算子只得到 X，不得到该父窗 y，也不提前整段填好 T 再切 X。**
  单个 X 内的双向插值可用，因为该 192 点在该预测起点全部已发生。
- 训练 y 保持原始有限值；不会把插补出来的目标当作真实监督。
- 部署/评分时，同一冻结的每实体处理程序作用于该 origin 之前的 192 点 X；不读该 origin 的未来 y。
  条件规则在 T 上编译为整批赋值，之后不因 C_B/E 改编。
- 每套完整处理材料共同训练一个 MLP；绝不拼接不同程序模型的逐实体预测。
- 本包同时改变训练输入和预测输入，主结果叫**准备管线效用**。
  另用同一个已拟合模型、公共 Baseline-Linear 预测输入做训练侧影子评分（0 新拟合、0 LLM），
  对比 Baseline-Linear 模型，报告“固定服务输入时的训练材料价值”；它不参与交付/选卡。
  不把完整管线增益全部归因于训练材料。

这是快速迁移的明确切片，不是宣称所有数据就绪问题只有 X 清洗。
历史目标重写、样本权重/筛选、尺度可逆变换、模型选择等以后另议，不在首包同时打开。

### 公共 Consumer 和最低完整性处理

- 复用现役 MLP 192→128→64→48、AdamW、2000 updates、batch=64 及学习率/weight decay。
- readiness profile 用**处理后的 X + 原始 y**训练；不再叠加增强的 parent/child 0.5 混合，
  不在本包调用 Mixup/FreqMask/FreqMix。
- 全部合法父窗按固定次序扁平化，配对 seed 使用同一父窗抽样序列；真实池大小来自本批数据，
  不使用旧写死的 13856。不同实体可用样本数如实列出；候选间绝不变。
- scaler 只由该 Job 原始 T 的有限值计算 mean/std（ddof=0），各臂完全相同，
  不由清洗后的数组重估。零尺度规则仍为 1e-6。
- 空程序指保留全部已观测值，并对 X 中缺口做现有 `impute_linear(strength=1)` 的最低填充。
  全空预测输入仅由固定 T 有限值均值兜底并计数；不能偷读未来或让每臂自选。
- 所有候选最后做有限性/长度检查；残留 NaN 走同一最低填充并记录。Inf、长度改变或失败不能伪装成正常 identity。
- 这个基线叫 **Baseline-Linear**，不叫“原始无处理”，因为普通 MLP 不能直接吃 NaN。

## 4. 动作与观察：复用现有实现，不只给算子名

在 `operators/registry.py` 及实际 callable 上做薄适配，不复制重写算法。
新 study 的公共闭合动作表如下；各臂同权限，最多 3 步，最多 8 条条件规则，允许空程序。
这些是原子能力，完整 Workflow 仍由 Fast 的观察、构造、实验与 commit 组成。

| 类别 | 算子和首版参数 |
|---|---|
| 插补 | impute_linear(strength=1)；impute_ema(alpha=0.1/0.3)；impute_fft(cutoff_ratio=0.1/0.2) |
| 周期插补 | period_complete(period=24/168)；period_median_complete(period=24,cycles=3,min_donors=2) |
| 点式/稳健处理 | hampel_filter(window=5/7,n_sigmas=3/4)；outlier_iqr(k=1.5/3.0)；outlier_mad(k=3.5/5.0) |
| 去噪 | denoise_median(window=3/5,strength=0.5/1.0)；denoise_savgol(window=5/11,order=2) |

period_median 的 168 周期在 192 点输入内凑不出两个历史 donor，本版不提供这个必然退化参数。
不先接分解、对齐、整形、level-shift 或昂贵 SSM；它们不是被判无效，只是本包不承担其额外语义。
底层某算子会自行线性补缺、使用对称边界或发生依赖 fallback，必须在工具说明与执行记录里如实展示。
严格使用现有 verifier 的形状、有限性、时间与参数检查；新 Forecast 不恢复已取消的通用修改点比例硬门。

7 工具仍为 overview / inspect_data / build_material / inspect_material / evaluate / compare / commit。
readiness adapter 替换相关内容：
- overview：所有实体 T 的缺失率、最长连续缺口、头尾缺口、有限值 mean/std、lag24/168 相关、
  持续常值比例、稳健偏离读数、最近/此前水平与波动差；标明公式与 missing-aware 计算规则。
  优先复用已有计算；random 所需 robust_deviation_fraction 固定为有限 T 点中
  |x−median| > 3.5×1.4826×MAD 的比例，MAD=0 时为 null；lag 相关仅用成对有限点，少于32对为 null。
  没有“确诊异常”字段；不可计算为 null，不补成 0 或假相关。
- inspect_data：T 内原始片段及 NaN 掩码、按日/小时摘要；保留绝对时间绑定。
- inspect_material：具体处理前后窗口、填补数量、已观测点改动量、均值/波动改变、fallback、训练 y 未改变的核对。
  这些是材料诊断，不是效用预测器。
- build_material：实际闭合 DSL；支持按公开特征与批内分位数写规则、默认与例外；不靠 UID 匹配。
- evaluate：整批、三配对 seed、只返回 C_A；每个 Fast 最多 2 个新完整方案。
- compare：配对差、每 seed、每 origin、实体贡献、SE 和覆盖数。
- commit：任一已评估方案，包括公共基线；不被 C_B argmin 覆写。

不要只扩 operator name 列表而继续给 Fast 看没有缺失信息的增强旧视图。

## 5. Fast、Slow 与提示词

Opus 是开发执行者，不是模型内 Code Agent。本包不开放任意 Python 发明新算子。

Fast 公共系统提示在首个 Source 调用前冻结，所有臂同一份。必须明确：
> 你的任务是为当前整批数据构造可执行的输入准备流程，使固定共享 Consumer 对原始观测未来的预测更好。
> 阅读全批现象及算子语义，用实际工具证据决定是否补缺、处理异常候选或去噪，组织预算内比较，允许直接保留基线。
> 数据有缺口不等于复杂插补更好；高峰不等于错误；更平滑不等于更有训练价值。
> 只使用 T 和本轨迹 C_A。单实体预测变化不是该实体材料的独立因果贡献。输出真实工具动作，不输出伪执行的说明文字。

在此基础上保留现有 JSON 契约、候选预算、错误处理和禁止修改 Consumer/裁判等条款。
不硬编码某数据域应该用哪个算子，不强制非 identity、观察次数、异质比例或多步方案。

Slow：
- 读本域两个 Source 的真实合法轨迹、原始条件、完整程序、C_A、commit 后 C_B、成本和失败；
  不读本包 Source E、不读 Select/Target 数据，不载入增强卡冒充数据就绪经验。
- 每域一次形成，最多 W1/W2 或 KEEP；只允许现役一次契约纠错，不因“不够聪明/太保守”重抽。
- 沿用 `domain_skill.py` 的 Workflow + 可选 Principles、scope、引用、1200 字符上限。
  允许明确 const:true；缺字段/null 仍不等于无条件。
- 内容解释观察什么、区分什么问题、如何构造及比较、何时停止；
  可以推荐具体动作，但未测不等于有害。不得写历史 UID、Job 或整批答案表。
- 为提速，本包不另开第二轮交叉生成或 Target 中途修卡。形成、选优和复用已经构成首个完整离线学习链。

Generic 精确复用 `batch_research_source_process.GENERIC_TEXT`，它是一般研究建议，未提增强专用动作，
不用重新手写一张刻意强/弱的对照卡。实际请求中的完整 system、卡和工具说明都随现有请求记录保存。

## 6. 一次跑完的课程和对照

### A. Source

两域各 S1/S2，一条 no_skill Fast 轨迹/Job。
每个 Job 公共拟合 Baseline-Linear、Fixed-Seasonal，再给 Fast 两个新方案槽；三配对 seed。
该域两条 commit 冻结后开 C_B；不运行 Source E。随后形成域 W1/W2。

Fixed-Seasonal 是统一 `period_median_complete(period=24,cycles=3,min_donors=2)`，
train/serve 几何同候选。它是事先指定的有力常规插补对照，不是历史最优答案。

### B. Select

每域 V1 上独立跑 NO_SKILL、W1、W2；同样公共基线和每臂两个槽。
**先该域所有选项 commit，再统一开 C_B。**
按各自真正交付的 C_B 三 seed 宏均值选，完全平局顺序 NO_SKILL、W1、W2。
不拿该臂集合中事后最优模型替代实际 commit；不打开 Select E。
NO_SKILL 最优或合法 KEEP → NO_EFFECTIVE_CANDIDATE，不为了两域都有卡而手工选第二名。
被选卡仅标本包 FROZEN_SELECTED，不宣称统计显著。选择失败的域保留其它 Fast/基线结果，
Target known_domain 为 NO_TREATMENT，不伪造一份重跑当 Skill 效果。

### C. Target

两域 T1/T2 共四 Job；同域共用冻结卡，不在 T1 后更新。

| 臂 | 内容 | 比较用途 |
|---|---|---|
| Baseline-Linear | 最低填充、共同模型 | 输入准备的最低完整性基线 |
| Fixed-Seasonal | 固定周期插补 | 简单处理已能获得的收益 |
| no_skill | 公共 Fast，无卡 | Fast 本地研究 |
| generic | 同 Fast + 既有 Generic | 领域内容是否超出通用建议 |
| known_domain | 同 Fast + 选中域卡 | 经验的增量 |
| random | 同 DSL、两个随机完整方案、按 C_A 选择 | 同拟合预算零 LLM 搜索 |

每个主动研究臂两新候选槽，不强制用满。共享基线只算一次物理拟合；私有结果不交叉暴露。
Fast 仍自主 commit，Random 在自己两方案及两个公共基线中取 C_A 均值最小，平局按公共基线在先。

随机供给在首个 Source 拟合前冻结：
- 每个默认/例外程序长度 1/2/3 的概率 0.5/0.35/0.15，原子算子等概率、其公开参数各自等概率；
- 完整方案有 0/1 条例外规则，各概率 0.5；规则字段从 missing_fraction、longest_gap、
  lag24_corr、robust_deviation_fraction 等实际公开可定义字段中等概率取，阈值为本批中位数；
- 规则 null 不命中；两个逻辑候选固定抽两次，别名记录、缓存复用，不按分数重抽。
- policy seeds：RD01_T1=(2026091701,2026091702)、RD02_T1=(2026091703,2026091704)、
  RD01_T2=(2026091705,2026091706)、RD02_T2=(2026091707,2026091708)。

Target 执行顺序：
RD01_T1：no_skill,generic,known_domain,random；
RD02_T1：generic,known_domain,no_skill,random；
RD01_T2：known_domain,no_skill,generic,random；
RD02_T2：no_skill,known_domain,generic,random。
全部 Target 计划臂完成或明确失败之后再统一开 C_B、冻结 E 预测、评分 E。

## 7. 缺失真值、评价对象与读数

训练不伪造监督；评分也不填补未知未来真值：
- C_A/C_B/E 使用原始有限 y 的共同掩码，绝不按模型/候选选择评分点。
- 每实体在一整个块内，对原始可观测的 origin×horizon 单元计算 normalized MSE，
  再对固定人口等权宏平均；完整无缺失时与原口径相同。
- 预声明最低覆盖：每实体块内原始观测数至少为该块预测单元数的 25%。
  未达标记该块 NOT_SCORABLE，主宏分为 null，不能删该实体换分母或把失败填成 0。
  不为获得可评分/正收益回头更换人口、时间或阈值。无关域/块继续。
- 报告按实体/块覆盖、原始 MAE 辅助读数；不把辅助指标换成主目标。
- 显式给出 missing-aware 分数只代表**实际观测到的未来**，不能证明对缺失真值也有效。
- 每 candidate 在 C_A 是三 seed 配对结果；报告均值、每 seed、SE，不以 n=3 的 2SE 当显著性认证。
- Target E 主表列实际 commit 的准备管线效用；另列 §3 的固定服务输入训练侧影子。
- 预测采用现役 rolling-origin 语义：每个 origin 可用自己之前的实际历史，
  不将已到达的早期 E 值用于修改卡、配方或模型，也不把这写成整段 E 无观测的单次长期预测。

必须分别回答：
1. 存在真实缺口，算子是否实际改变合法材料？
2. 任一已测处理相对 Baseline-Linear 有无可辨收益/伤害？
3. Fast 是否胜过固定处理/同预算 Random？
4. 域 Skill 是否改善同 Fast 的交付或含形成成本的效率？
5. 收益来自训练材料还是预测输入，当前影子读数能支持到哪里？

管线收益、Fast 增量、Skill 增量分别列，不能用第一项替代第三项。

## 8. 预算：整个包共用，不能每入口重置

训练种子固定 `[20260922,20260923,20260924]`。
每 Job 两公共基线=6 次拟合；每主动臂两个新方案最多=6 次拟合。

| 阶段 | 计划拟合上限 | LLM 逻辑请求上限 | token 上限 |
|---|---:|---:|---:|
| 两域 Source，各两个 Job | 48 | 64 | 1,200,000 |
| 两域形成 + Select | 48 | 100（含 Slow 与一次契约纠错） | 2,200,000 |
| 四个 Target | 120 | 192 | 3,600,000 |
| 合计 | 216 | 356 | 7,000,000 |

- 单 Fast 沿用 max_calls=16、max_tools=24、max_new_evaluations=2；合法工具纠错最多2，照常计费。
- 额外仅 2 次数值接线检查拟合、2 次同配置非科学失败重试；物理拟合尝试总帽 **220**。
  不使用这些额度追种子、补负结果或重新形成卡。
- HTTP 尝试上限712；不得另加隐形 API 探活花费。模型沿用请求 cpa-grok-4.6、返回 grok-4.6-build、temperature=0。
- 付费实验活跃墙钟上限8小时；开发实现以一个工作日为交付目标，超过时先交真实进度与具体阻塞，
  不无限扩平台。结合上包实际2.9小时，运行估计约3–5小时，工程适配另计，不是时长保证。
- 凭据沿用安全环境配置，不写入任务书、日志或报告；不得默换模型。
- 缓存只在数据身份、人口、原始/处理几何、父窗、scaler、完整赋值、seed均一致时使用；
  不给新材料借旧增强分数，缓存复用不冒充新重复。
- 阶段额度冻结，不按读数转移；全包账本包括失败花费。未知 usage 按现役规则暂停，
  未获用户明确接受不能自动恢复。保留 labels-withheld/resume 检查，不再重建恢复系统。
- Source/Slow/Select 一次性成本、Target 使用成本、程序基线成本分列。节省 Fast 请求不自动等于净节省。

这是给用户转交的建议执行上限；本文件编写不等于本会话已花费/已发起实验。
用户将本任务书交 Opus 执行后，按整包边界连续推进，不逐次拟合请示。

## 9. 代码改动边界与必要检查

优先复用：
- `methods/ttha/batch_research.py`：工具循环、动作轨迹；
- `methods/ttha/domain_skill.py`：census/propose/select/freeze、载体和路由；
- `methods/ttha/batch_base/train.py`：MLP/train_arm；
- `evaluation/main_protocol_p4/batch_research_runtime.py`：Adapter、配对训练、账本；
- `evaluation/main_protocol_p4/run_batch_research_roundtrip.py`：阶段、commit、标签和恢复；
- `operators/registry.py`、现有 executor/verifier：真实算子。
新增一个 readiness profile/adapter 和一个 study 模块即可。共享代码新增 opt-in 参数，
不把新 NaN/人口/评分语义默认施加到旧增强包；不复制 MLP 和另一套 Agent loop。

只做一组覆盖新风险的 smoke：
1. 原始 NaN 到工具/算子仍保留，不先用 without-missing 缓存洗掉；
2. operator 得到 X=192，训练 y、父窗集合、scaler 不被候选改变；
3. train/serve 前后长度和有限性；Baseline-Linear 与显式 linear 为别名；
4. N=12/32 均真正训练同一个共享模型，默认增强接口不变；
5. 缺失评分掩码对所有臂一致，无未来填补；
6. 卡实际进入请求、Source/Select 不开 E、Select 先全 commit；
7. 一条旧增强材料/控制路径仍与保存材料相同，原 controller tests 与最贴近的 domain_skill smoke通过。
不重跑历代实验矩阵、不修无关 measured_start 历史问题。
两次数值拟合检查仅用于实际模型/批次接线，不用它们筛选有利数据。

## 10. 冻结、停止与判读

首个付费调用前冻结真实配置、roster、时间、动作参数、模型、预算、Generic、随机供给和主指标。
只读 T 资格检查已经发生，诚实记录；不伪称配置在接触任何 T 前已冻结。

连续执行 Source→Select→Target，不以 Source 正负作为临时追加/换数据开关。
技术未完成与科学负结果分开；合法 KEEP、NO_EFFECTIVE_CANDIDATE 和主动保留基线保留原义。

固定出口：
- MATERIALS_DIFFER_UTILITY_UNCLEAR：处理确有变化，三 seed/两个后续块仍分不开；
- SIMPLE_PREPARATION_BENEFIT：固定处理/Random 已解释收益；
- FAST_LOCAL_BENEFIT：无卡 Fast 相对固定处理与同预算 Random 有局部收益；
- DOMAIN_SKILL_LOCAL_BENEFIT：域卡相对无卡/Generic 的配对读数有局部优势；
- NO_INCREMENT / HARM / INCOMPLETE：分别说明没有增量、真实伤害或不可比。
这些是证据描述，不用自造 p 值或宽松门自动晋升能力。

两后续批次同向且达到实际幅度才能说“有重复线索”，仍不称广泛泛化。
只在一个作业好、另一作业反向必须并列；训练 seed 不是独立的 Agent 轨迹。
即使没有净收益，也按原表收口，不能又开一个改卡小包追正号。

## 11. 最小交付和后续项目计划

交付：
- 可复跑的 readiness study 入口、同一 Harness 的运行配置；
- `REPORT.md`：第一页放自然缺口、Fast/Skill实际改变、主结果/成本、失败与唯一下一步；
- `result.json`、原始请求/轨迹、冻结卡/配置、材料与模型/预测、现役预算和必要检查；
- 一个短方法说明：应用任务→观察→构造→反馈→形成/选卡→冻结复用，以及训练/服务/真值边界；
- 本任务书末尾追加实际收口，不覆盖历史包或 AGENTS，不 git commit，不新增 SHA/Hash。

后续路线只规划、不自动启动：
1. **本包**完成数据就绪的最小完整版本，先有真实可用系统和主比较；
2. 对本包暴露的一个实际失败修订一次 Workflow，并在预定后续批次验证新旧卡；不要求一定修改；
3. 在卡有可解释用途后接已有 profile_match，对照 known-domain、共享卡/无卡；
4. 扩至另一应用域并冻结正式测试；再做论文完整主表和消融。
不以完成第1步冒充整个科研结论完成，也不把每一步拆成没有真实运行的建设包。

## 12. 实际收口（2026-09-17，Opus）

主报告 `_scratch/dev_data_readiness_domain_skill_v1/REPORT.md`，数值见 `result.json`，方法说明见 `METHOD.md`。

**执行 COMPLETE，协议合规，未自动追加实验。**
- **成本**：119/220 次拟合、78/356 次请求（HTTP 78/712）、已知 2,893,373 token 另有 1 次未知 usage（上限估 72,377）/7M、1.51 h/8 h；0 次拟合失败，0 次重试。
- **结构**：RD01_S1 与 RD01_T2 的 C_A 块按 25% 覆盖规则不可评分（北京站点整段停测），这两个作业不运行任何臂，不换人口、时间或阈值。
- **事件**：RD02_S1 首次 Fast 使用 0 基 sub_range，触发继承的硬权限检查而失败；执行者停止 Source 时 RD01_S2 有 1 次请求在途。用户明确接受这次未知 usage，并决定 RD01 只用 S2 形成。amendment 1 将越界 sub_range 改为可纠错输入错误；两条中断尝试在任何 build、评估或 commit 之前以 `__interrupted_1` 保留并重跑。
- **形成与选卡**：两域 Slow 各给出 W1/W2，均未纠错。RD01 在 V1 上三个选项都交付 Fixed-Seasonal，C_B 平局，判 NO_EFFECTIVE_CANDIDATE（W1 的适用范围未命中，从未加载）。RD02 的 W1/W2 交付 Baseline-Linear，C_B 2.9208 优于无卡的 2.9504，冻结 RD02-W1。
- **Target E**（Δ = E(BL) − E(臂)，正值表示该臂更好）：
  - RD01_T1：Fixed-Seasonal +.046（3/3），Random +.078（2/3），无卡与 Generic 都交付 BL；
  - RD02_T1：三个 Fast 臂都交付 FS −.021，Random +.060（3/3）；
  - RD02_T2：无卡、有卡、Random 交付 BL，Generic 为 FFT +.098（2/3），FS +.139（2/3）。
- **Δ_skill**：RD02_T1 = 0，RD02_T2 = 0（交付完全相同），有卡臂只少用 2 次请求。
- **固定出口**：准备管线 MATERIALS_DIFFER_UTILITY_UNCLEAR，局部 SIMPLE_PREPARATION_BENEFIT；Fast 增量 NO_INCREMENT；域 Skill 增量 NO_INCREMENT。
- **影子读数**：两个异常/去噪类 Random 方案的收益在服务输入固定时基本保留；RD01_T1 的 FS 收益主要经由预测输入。
- **实际失败**：FS 相对 BL 的方向，C_A 与 C_B 只有 2/8 一致，与 E 为 0/3；所有 Fast 臂都跟随 C_A。Fast 从未尝试异常处理或去噪。
- **下一步**只作为路线第 2 步交 Planner 裁定，本包不启动。
