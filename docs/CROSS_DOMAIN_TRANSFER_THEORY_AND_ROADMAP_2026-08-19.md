# 跨域适应能力的理论基础与推进路线(2026-08-19)

状态:方向分析文档 **v3**(经 sol 三审收口)。基于三路文献调研(迁移/不变性/元学习/决策理论、LLM Agent 技能与记忆、数据准备与 AutoML 传统)与项目自身实证发现的整合;v2 并入 sol 五项修正与 grok 故障定位复核,v3 并入串行排序与观察层拆分,本次收口修正四处(见下)。
作者:主 Agent(fabel);调研由三个子代理执行,关键论文均经网络核对,少数标注 [不确定]。

**文档地位(须先读)**:本文件是**方向草案**,尚未进入提交历史,**不是仓库已冻结的执行契约**。可作为主线排序与纪律的依据,但每个 M 的具体配置在执行前仍须以各自冻结的 plan/report artifact 为准。

**v3 收口要点(sol 三审,2026-08-20)**:
1. pss 污染是**语义缺陷,尚未证明解释效用翻转**——"记录 pss 与 union end 的关系"无检验价值(二者按定义等价);M0a 改为计算 `union_pss` vs `level_only_pss` 的分歧及其来源(outlier/missing/两者),只有分歧真实出现且与 RLS 失败分层对应,污染才成为行为解释的候选。
2. 两个覆盖率 ≠ 恢复空间几何:point 口径与 region 口径不对齐,且 fraction 无位置信息、答不了"谁把 union end 推到尾部";M0a 报告扩为五个量(均为 union 一致口径),M0b 的最少字段集在看清"缺覆盖还是缺尾部位置"后才冻结。
3. 删除 Weather 的 pss 符号证据(与 `METRIC_UNREADABLE` 裁定矛盾):Weather 只可检查 Observation 分布,不参与 Source 正负计数,不支撑 pss→utility 翻转,也不需要再修 Weather Metric。
4. M0a 失败只产出 `OUTLIER_LEVEL_MASK_GEOMETRY_CANDIDATE_NOT_INFORMATIVE`——否证本次候选 Observation,不自动宣判 family 终止;不准转去堆 Memory;可再找有机械依据的最小 Observation,无则才 abstain/关闭。

**v3 修正要点(2026-08-19 晚,sol 二审)**:M0 与 M1a 由并行改为**严格串行**(M0 先冻结 Context cell);M0 拆为 M0a 离线普查 / M0b Fast 接入;观察缺口措辞由"分不开两者"改为"缺少两类 mask 的独立空间几何";M0 由"已解释 collapse"降为"最强待验证 first-fault 假设";CRM 类比补 unknown-propensity 限制;≥48 对改为条件化功效情景;GovMem 召回代价与"第一个"降为待验证/未见先例;KDD W3 默认保持 Context freshness;明确 M0 不自动修 Program Binding 并预声明下一 fault 分支;新增 pss 受 union 污染的机械发现。

**v2 修正要点**:理论关系从"精确预言"降格为"定性一致并启发设计约束";机制签名定位为候选 effect modifier(S-admissibility 是需要跨域稳定性证据挣得的属性,不是可先验声称的标签);原 M1 拆成 M1a(单次守门)与 M1b(加宽,缓做);原 M3 从 20–40 域收缩为 4–8 真实基域小面板并明确计数单位;新增 M0(观察可辨性,当前第一刀);provenance 三分类进设计层、缓实施;FCSP 命名推迟。

---

## 0. 结论先行

1. **当前方向可行,不需要换大方向。** 现有骨架(Experience → 跨域 UNGUIDED 授权 → Support 守门 → abstain → 分层执行权)与理论上站得住的跨域迁移设计几乎逐层对应,其中证据层(provenance 防火墙 + 授权审计)已经是文献空位级的贡献。
2. 需要**三个中等修改 + 一个供给层建设**,全部在现有身份内:
   - 语义层:context 从相关性索引升级为**机制签名**(已在启动,T3 改绑);
   - 执行层:先验使用从开环重复提名改为**单次测试守门 + 按证据量加宽不确定度**;
   - 供给层:源证据从"单自然域深挖"改为**机制注入的多样 mini 源域工厂**(域随机化);
   - 治理层:审计规则成文冻结为永久组件(已建议主线执行)。
3. 项目的四个实证发现(D0 原始 episode 锚定、R2 单任务 TRY 锚定、provenance 审计的自我强化、pss 跨域不稳定——其中 Weather 侧符号证据已判不可读,详见 §1.1)**每一个都与一条已确立理论所描述的失效模式定性一致,并由其启发设计约束**。注意声明强度:这些定理的假设(已知因果图、贝叶斯 TS、任务 i.i.d.、有界损失)在 LLM Harness 中不严格成立,论文中只能写"观察与理论失效模式一致、设计受其启发",不能写"理论证明了本系统"。
4. **当前第一阻塞不在理论层**(grok 定位、sol 精化,已逐行核实):公开 mapping **已有两个机制的强度标量**(`local_robust_z_peak`、`level_excursion_score`),缺的是**两类 mask 各自独立的空间覆盖与作用几何**——唯一的空间范围字段 `estimated_region_start/end_fraction` 由 missing|outlier|level 三者的 union 算出(`public_features.py:274,297-298`),`g1.py:3823` 有文档化耦合注释(被 outlier 扩大的 region 却用只由 level 模型估计的 offset 去修复)。
   这是**当前最强、尚待验证的 first-fault 假设**,不是已证明的 proposal-collapse 成因:工具菜单先验、候选供给与采样仍可能共同作用,TimeClaw 的 tool-prior collapse 只是机制类比。因此第一刀是 M0a(廉价离线判定该假设死活),不是任何理论建设,也不是先给 Source Skill 加更多门。

---

## 1. 四层理论框架:什么设计能从理论上支撑"跨域适应能力"

不存在单一定理认证"LLM-agent 技能库可跨域迁移"。可行的论证是四层结构,每层挂一条已确立的理论结果,并且每层恰好对应本项目的一个实证发现。

### 1.1 语义层:可以迁移什么 —— 机制可容许的条件化断言

**理论锚点**:Pearl–Bareinboim transportability(AAAI 2011/2012, Statistical Science 2014)。干预效果可跨域搬运 ⟺ 存在 **S-admissible** 条件集 Z:在干预图中 Z 把"域差异节点 S"与 outcome 隔断,即 z-specific 效果本身跨域不变,变的只是 P(z) 的权重。条件在非 S-admissible 的相关变量上时,符号翻转被理论**允许**。完备性定理同时授权"有原则的 abstain":有些效果在给定观察能力下原则上不可迁移。

**项目对应**:`post_shift_support_sufficient` 是"条件在非机制变量上"的**候选案例**,但证据现状必须准确陈述:它由 union 的 end_fraction 推出(非机制变量,机械污染见 §4 M0);electricity 一侧与 RLS 失败共存的证据成立;此前引用的"Weather 正向"一端已随 `METRIC_UNREADABLE` 裁定作废(Weather 聚合 Utility 的正负结论一律不可用)。因此"跨域符号翻转"降级为**单侧证据 + 待 M0a 检验的污染假设**,不再作为教科书式翻转案例引用。

**设计要求**:
- Skill 的 WHEN 条件必须落在**缺陷生成机制的可观察痕迹**上(extreme deviation 的存在与几何、缺失块拓扑、shift 的 re-anchoring 行为),不是数据集统计摘要;
- **声明强度(v2 修正)**:任何机制签名在获得跨已曝光域的稳定性证据之前,只能称**候选 effect modifier**;S-admissibility 是需要挣得的目标属性,项目目前没有证明任何 Context signature 是 S-admissible;
- 每条 TRY 条款附带显式的**不变性假设声明**("本条款假设哪些机制跨域不变");
- 跨域翻转发生时,第一修复对象是条件集(CONTEXT_GAP),不是推荐强度;
- 公开 Context 无法识别机制时,理论最优动作是 reject/abstain(Cortes, DeSalvo & Mohri, *Learning with Rejection*, ALT 2016),不是编 TRY;
- 补充(域随机化 LMDP 定理,Chen et al. ICLR 2022):从多样源域可学到的最优对象是 **probe-then-act 的闭环条件计划**,不是"某 operator 好"的开环结论——Skill Card 的 OBSERVE→TRY→VERIFY→FALLBACK 结构是定理要求,不是工程装饰。

### 1.2 证据层:何时有权迁移 —— 独立域计数与多样性作为授权硬通货

**理论锚点**:
- Baxter (JAIR 2000) / PAC-Bayes 元学习两段式界 (Pentina & Lampert ICML 2014; Amit & Meir ICML 2018):对新域的泛化差距中,环境复杂度项**只随独立任务/域数 n 收敛**,单域内加样本只压任务内项;
- Rosenfeld et al. (ICLR 2021):环境数 E ≤ 虚假特征维数 d_e 时,不变性目标本身会选中虚假特征;
- EILLS (Fan et al. 2024) 与 2025–2026 少环境识别结果:决定性的是**环境在机制维度上的多样性**,而非数量;两个足够异质的域已具真实证伪力;
- 关键推论:**Skill 在场时诱导的后续尝试不满足任务独立采样假设**,在上述界中不推进环境项——这是 provenance 防火墙("确认性证据不得扩权")的假设层面理由,不只是经验教训;
- 更锋利的对应(grok 补充,sol 加限制):Source Episode 是 **logging policy 下的 bandit 反馈**(Swaminathan & Joachims, *Counterfactual Risk Minimization*, ICML 2015)——GUIDANCE/SKILL_CONDITIONED 正例不能当独立发现,正是 CRM 意义上的 logging 混淆。
  **限制必须同时写出**:CRM 依赖已知 propensity 与方差控制,而本项目的 UNGUIDED 由 h0 LLM 的非均匀策略产生、**动作 propensity 未记录**。所以 UNGUIDED 只是**更干净的证据层**,不是统计意义上的无偏探索数据;provenance 防火墙是**保守的政策依赖隔离**,不是 CRM estimator。**明确不做**:不为此建设 propensity 记录或反事实估计平台。

**项目对应**:授权审计发现 outlier_iqr 的 6/0 里 5 个是卡在场时的自我确认(独立证据 = 1 任务);单域 UNGUIDED 证据饱和律。

**设计要求**(大部分已冻结实施):
- 主动条款授权只数跨域 UNGUIDED 独立任务/域;
- ICP 语义的 Gate 行为:证据不足时输出**弃权**,不是降低门槛;
- 源域选择追求机制维度的张满(Tripuraneni 任务多样性),不是堆同类域。

### 1.3 执行层:迁移错了怎么办 —— 测试守门先验的有界损害

**理论锚点**:
- Simchowitz et al. (NeurIPS 2021):错误先验下 TS 的代价 ≤ Õ(H²ε)(ε = 先验与真相的 TV 距离),且下界匹配——**锚定代价随决策 horizon 平方增长**,反复提名同一家族正是把 H 拉长;
- Bastani et al. (Management Science 2022) prior widening:按证据量加宽先验协方差,meta regret 亚线性;
- Corral (COLT 2017) / ARRoW-CB (ICML 2019):把"信 Skill"与"不信 Skill"同时作为 base learner,最坏情形 "never much worse" 是定理陈述;
- SPIBB (ICML 2019) / conservative bandits:相对 baseline 的高概率不劣保证;
- Explore-then-commit (NeurIPS 2016):测试守门的效率代价只是常数因子(≤2×),不是渐近级牺牲。

**项目对应**:R2 的单任务 TRY 导致 A5 五次首位提名同一家族、首正更晚、harm 更高——开环重复提名 = 长 horizon 下的 misspecified prior 所描述的最坏形态(注意:Simchowitz 界是贝叶斯 TS 下的定理,对 LLM prompt 锚定只是定性对应)。策略迭代母体:新策略必须与 incumbent 混合、直到优势被证明(Kakade & Langford, *Conservative Policy Iteration*, ICML 2002)——A3 作为不可关闭的默认臂正是这个结构。

**设计要求**(本路线图的 M1):
- 先验只改变候选**排序**;
- 每个 context 下先验条款**单次**触发探测,Support 失败即本地静默该条款(写入负 Episode),行为回落到 UNGUIDED 策略;
- 条款不确定度按独立域计数加宽(证据少 → 提名自动恢复多样性);
- 由此最坏情形损害可预算:≤ 条款数 × 单次探测成本,收益侧在机制匹配时跳过冷启动探索。**A5 需要证明的不是"总是正迁移",而是"最坏有界、匹配时更快"。**

### 1.4 供给层:源域应该长什么样 —— 机制多样的合成源 + 少量真实确认

**理论锚点**:
- 域随机化 LMDP 定理(ICLR 2022):随机化源分布覆盖真实机制时,Bayes-最优自适应策略近似 minimax 最优,可零真实样本;反面结果(arXiv:2210.15598):保证质量取决于**机制覆盖密度**,不是随机化越广越好;
- Baxter n-主导 + 饱和律:同域深挖对授权几乎无贡献(赢家家族的后续证据全变 conditioned),**广度供授权、深度供证伪**;
- NTC/NTG (Wang et al. CVPR 2019):A5 vs A3 正是负迁移间隙的预注册操作化。

**项目对应**:T233 十八个任务只产出每家族 1–2 个 UNGUIDED 样本;e31 整体无仪器可读行;自然域供给天然稀缺。

**设计要求**(本路线图的 M3):受控机制注入的 mini 源域工厂。注意与 AGENTS.md"受控注入只是支架"的一致性:注入域用于**建立机制词典与授权证据**,承重 Claim 仍在自然目标域的 A5 vs A3 上,不改变项目终点。

### 1.5 分层执行权的决策论根据(贯穿四层)

各层知识的证据门槛 ∝ 错误条款的最坏损害:
- **调查程序**(怎么看、先探什么):闭环、自纠错,错误代价 = 浪费观察预算 → 门槛最低,研究者可初始化;
- **风险/降权条款**:保守方向,错误代价 = 错过机会,有探索预算上界 → 中等门槛;
- **guarded TRY**:开环断言,错误代价 = 锚定 + harm → 跨域 UNGUIDED ≥2 + 测试守门;
- **无探测直接执行(Shared Capability)**:错误代价无守门 → 最高门槛(多域重复 + 机制签名 + fresh 确认)。

现有三层知识设计(Experience / Target-local / Shared)与此一致;需要补的是把"门槛 ∝ 最坏损害"写成显式原则。

**独立实证支持(MTL, arXiv:2604.14004)**:跨 6 个 coding benchmark 的四种记忆表示对比发现,可迁移价值的 **94.5% 是 meta-knowledge**(验证例程、结构先检查再校验、最小 patch 策略、预判工具链失败),算法策略只占 5.5%;原始轨迹因过度具体诱发负迁移("域不匹配的误导性锚点")。这给 Skill Card 的内容配比以直接指导:**卡的主体应是调查/验证/风险纪律(OBSERVE/VERIFY/RISK/FALLBACK),TRY 是少数派且门槛最高**——与本节的损害比例原则相互印证。

---

## 2. 文献定位:哪些空位是真实的

三路调研的交叉结论(截至 2026-08;技能库方向以第二份更深的调研为准):

1. **provenance 防火墙(证据资格分层)仍是空位,但边界比初判收窄,必须精确划界。** 2026 年 6–8 月有三篇非常近的邻居:
   - **VaG / When Self-Evolution Backfires (arXiv:2608.05810)**:形式化了技能污染链(新技能以当前技能池为上下文蒸馏,缺陷沿血缘传播且事后清理结构性不可逆),用 pre-commit 三重门 + Cold/Warm/Hot 信任分层治理。**它治理的是提案通道(防坏技能进上下文),我们治理的是授权通道(技能在场时产生的确认性证据不得为技能扩权投票);它是一次性准入检查,我们是持续证据资格判定。** 它原文承认"技能被写下时上下文里有哪些技能"这个 provenance 字段在现有系统中不存在——这句话是我们贡献声明的锚点。
   - **GovMem (arXiv:2607.02579)**:"重复的 agent 观察不是独立选票",按依赖结构折算 effective support,依赖类型里已有 same-agent echo。**它的依赖是横向的(多 trace 间相关),我们的是纵向的(证据是否在被评审技能在场时产生);它输出写不写记忆,我们输出给多少执行权。** 它的代价数据是重要警告:严格资格判定使 review burden 0.042→0.692、recall 0.985→0.448——我们"conditioned 证据仍可维护/反驳"的降权而非拒收设计恰好规避此代价,应显式写成相对优势。
   - **Trap of Trajectory (arXiv:2605.09330)**:唯一把反馈闭环写成问题陈述的("被采纳一次的虚假相关会塑造用来证明它自己的证据"),用因果 DAG 语言,但基准是静态构造、方法是表征校准,没做闭环治理。引用其问题陈述,做它没做的机制。
   - 仍未被任何人拿下的四件事:(i) provenance 作为**授权资格**(UNGUIDED 可授权新主动推荐,conditioned 只可维护/反驳);(ii) **执行权与证据来源挂钩**(现有分层全由 critic 分数/执行验证/效用决定);(iii) **跨域独立重复作为晋升硬通货 + 域内证据饱和的显式建模**;(iv) **自我确认污染的定量测量协议**——没有人测"某技能的晋升证据里有多少是它自己诱导的",我们的授权审计(6 正中 5 为 conditioned)是第一个实例。
2. **等预算 A5 vs A3 的时序数据准备实验无直接竞品,但有一个必须对标的已发表近似:HASTE (arXiv:2606.30911)。** ML 工程域的 cold-start vs warm-start:warm 少用 52% refinement iterations、提议保留率 42%→85%——读数形态与我们的"试错数/首正速度"同构。其弱点(单 seed、无硬验证门、无 provenance、无适用条件)正是我们的差异化空间;其 tiered-vs-flat 加载对照(159 技能固定,分层加载 100% vs 平铺 62.5%=等于不加载)独立支持我们的分层检索设计。数据侧最近竞品 DataMaster (arXiv:2605.10906) 与 DataEvolver (arXiv:2606.07001) 固定 consumer、只动数据侧、下游反馈验证,但记忆是平坦存储,无分层执行权、无 provenance、无适用条件。另需核对 MSCE (arXiv:2607.16621) 的三层结构(trace→L2 policy→callable skill,带 evidence links)与我们三层知识设计的重叠度。传统侧:唯一声称清洗政策跨数据集迁移的 L2C2 (2026) 迁的是 TFM 先验对齐 + 9 维统计状态,不是机制条件化经验,且在表格域。

**TS-Agent 侧最近邻(v2 补,来自 sol/grok 复核,必须对标)**:
   - **TimeClaw (arXiv:2605.10038)**:Explore–Compare–Distill–Reinject 循环,并点名 **tool-prior collapse**(早期熟悉工具压死探索熵)——这几乎就是 electricity 上 A3 的 RLS 成瘾(11 探 0 outlier)。它优化 benchmark 分数,不测 probe/harm/abstention 适应轨迹。
   - **TimeClaw (arXiv:2606.05404)**:TS 工具 + 成功轨迹指纹检索;`--k-neighbors 0` 即其 A3,但结局指标不是适应轨迹。
   - **MemoHarness (arXiv:2607.14159)**:六维 harness + 双层经验库,test-time 不花 label/feedback——本项目 A5 恰恰必须花同一笔 Target Support;其作者把统计稳健性与组件归因留作后续,正是本项目承重处。
   - **一句话差异**:它们复用/检索的是**成功执行痕迹或全局配置**;本项目把**带符号的 Action–Response(含失败与冲突)**编成 Target Support 门控的 Skill prior,并在 logging 污染(provenance)或 Observation 不可辨时**拒绝 TRY、退回 A3**。差异不在"有 Memory",而在**负例与冲突有否决权、默认臂不可关**。
   - SiMPL/SPiRL(offline skill prior)可作 related-work 类比:先验是运动技能分布,无执行权边界,不搬其网络。
3. **"清洗效用随任务/模型/机制翻转"不需要我们再证明。** CleanML/REIN/Krishnan 2021(表格)、RobustTSF/2025 缺失预报论文(时序)、ICLR 2026 Chronos vs Chronos-Bolt(同一 outlier 注入下不同模型偏好相反的预处理)已经做实——直接引用,把实验预算花在"Source 经验如何减少 Target 试错"上。
4. **机制词典有现成骨架。** Fox 1972 / Chen & Liu 1993 的 AO/IO/LS/TC/VC + 传感器文献的 missing-block/drift/stuck-at。没有被广泛采用的统一标准——我们可以定义"机制签名 → 修复算子族绑定"的 TS 词典,这本身是贡献。
5. **元特征路由的天花板被广泛默认但少被定理化。** 最硬引用:Feurer 2015 自认只能 warm-start、ELA 的 OOD 选择器 ≤ 均值基线、Rivolli 的可复现性批评。不要声称存在已量化的跨域 R² 上限。
6. **审稿人最低配置 baseline**:A3(等预算 target-only)、meta-feature router(tsfresh/catch22 kNN)、LLM zero-shot 建议;加分:oracle repair 上界、placebo guidance(T4 已冻结)、只成功经验 vs 成功+失败+冲突的 Memory 消融。
7. **没有现成 benchmark 承载本任务**,须自建(多源/目标域、机制注入、冻结 forecasting consumer、delayed 效用/harm/abstention);统计纪律复用 CleanML(多 split、配对检验、三值 flag)。

---

## 3. 方向判定

**判定:骨架保留,三改一建,不换向。**

| 层 | 现状 | 判定 | 动作 |
|---|---|---|---|
| 证据层 (I2) | 授权审计 + UNGUIDED 跨域门槛已冻结实施 | **已达标,是贡献主体** | 规则成文冻结(含卡条件化=conditioned、LOO 口径、反证处理三个待定点) |
| 语义层 (I1) | context 仍是相关性索引(pss) | 缺口 | M2:机制签名词典 v1 |
| 执行层 (I3) | 先验开环重复提名(R2 锚定已证) | 缺口 | M1:test-gated 单次守门 + 证据量加宽 |
| 供给层 (I4) | 单自然域深挖,UNGUIDED 饱和 | 缺口 | M3:机制注入 mini 源域工厂 |

**被否决的换向选项**:
- 端到端 learned router / embedding 检索:撞元特征天花板文献,且违背项目身份(AGENTS.md 明令);
- 纯贝叶斯层次模型(去 Agent 化):丢失 Workspace/Observation/Harness Update 的核心身份,变成 AutoML;
- 纯 foundation-model 先验(问大模型该怎么洗):D0 已经演示了无证据纪律先验的锚定,且不可证伪。

---

## 4. 推进路线 v3(每轮一个主假设;M0 → M1a 严格串行)

### M0 观察层:机制空间几何的可辨性(**当前第一刀**;必须在 M1a 之前完成并冻结)
事实基础(逐行核实):`public_features.py` 分别算出 `outlier_indices` 与 `level_mask`,但公开的唯一空间范围字段由三 mask 的 union 得出(`:274,297-298`);`g1.py:3823` 记录了后果(被 outlier 扩大的 region 却用只由 level 模型估计的 offset 去修复)。两个机制的**强度标量都在**(`local_robust_z_peak`、`level_excursion_score`),**缺的是各自独立的空间覆盖与作用几何**。T233 的 hampel 负任务 task_01 与 outlier 正任务 13–19 共用同一 peak series 与同量级 z-peak/excursion score。

**附带发现(v3 收口后的准确表述)**:`post_shift_support_sufficient` 由 union 的 `end_fraction` 推出(`:299-302`)——这是一个**语义缺陷**:pss 严格说不是"level shift 后还剩多少 Support",而是"三种区域 union 后还剩多少 Support"。它是否真的污染了决策(即 union_pss 与 level_only_pss 是否在真实任务上分歧、分歧是否与 RLS 失败分层对应)是 M0a 要回答的经验问题;在分歧数据出来之前,它只是缺陷,不是 collapse 的已证成因。R2 那条"看 pss"的纪律 patch 建立在该字段上,这一点不依赖分歧数据、现在就成立。

**声明强度**:union 折叠是当前最强、**尚待验证**的 first-fault 假设,不是已证明的 proposal-collapse 成因。

#### M0a:零 LLM 离线普查(不改 Fast contract、不改 `OBSERVABLE_FEATURES`)
- 不把任何字段送给 Fast。直接用 extractor 已有的内部 mask,确定性计算并报告**五个量**(全部与 union 的 region 口径一致,不引入新强度标量、不拟合或修改任何阈值):
  1. `outlier_region_fraction`(用扩张后的 outlier region,与 union 口径对齐;不用原始点计数);
  2. `level_region_fraction`;
  3. `outlier_region_end_fraction`;
  4. `level_region_end_fraction`;
  5. `union_pss` ≠ `level_only_pss` 的比例,及分歧来源分解(outlier / missing / 两者)。
  前四个回答"缺的是覆盖还是尾部位置"(fraction 无位置信息,答不了谁把 union end 推到尾部,所以 end fraction 必须同报);第五个回答"union 是否真的污染了 pss 的决策语义"。
- 数据范围:已曝光 **T233 + electricity** 判定可分性;**Weather 只报字段分布与覆盖率**——其聚合效用已裁定 `METRIC_UNREADABLE`(identity 损失跨通道 24.4×),不得使用任何 Utility 符号。
- **KDD W3 默认不打开**:即使不看 Outcome,查看实例 Context 也会记为 `INSTANCE_SEEN`;仅当已曝光数据不足以判定时才申请开启。
- 报告:字段有限性/非退化性、outlier 与 level 的可分性、mixed 与 ambiguous 比例、pss 分歧率及来源。
- **代表性纪律**:激发该假设的对比就住在 T233,所以 M0a 的可分性结论属**假设生成**级别;字段即使可分,该可分性主张仍须在下一个新域到达时复核一次,不得当作已验证的跨域性质。
- 判定:
  - 有信息量 → 停在 M0b 接入方案,**此时才**依据"缺覆盖还是缺尾部位置"冻结最少字段集(不预设是哪几个);
  - 无信息量 → 归档为 `OUTLIER_LEVEL_MASK_GEOMETRY_CANDIDATE_NOT_INFORMATIVE`:**只否证本次候选 Observation**,不宣判 family 终止;不准转去堆 Memory;可再判断是否存在另一个有机械依据的最小 Observation,确无候选时才 abstain/关闭该 family。

#### M0b:M0a 通过后才接入 Fast
- 只接入 M0a 冻结的最少字段集;不加 embedding、不加检索系统、不动 Source Skill、不改 Judge/Metric/Operator Supply。
- **工程前提(核实所得)**:`OBSERVABLE_FEATURES` 是带断言的封闭词表(`:310-311`),`mapping` 又参与 `feature_context_sha`(`:312-318`)。新增字段必然改动封闭词表并改变该 SHA——处置边界:**只更新当前词表与新运行,不迁移历史 SHA、不重写旧 replay、不升 Schema、不新建哈希体系**。
- 顺序:零 LLM payload 测试 → 2–3 Task 已曝光 micro → 检查 Agent 在 outlier / level / mixed 三种情形下能否产生不同的 grounded hypothesis。
- **明确不做**:不顺手修 `repair_level_shift` 的 union binding(第二个面)。
- **预声明的下一个 first fault 分支**:若 Agent 已正确识别为 outlier,但执行 RLS 时仍因 union binding 修错区域,则下一个 first fault 是 **Program Binding**,不是 Memory,也不是再加 Observation 字段。

### M1a 执行层:misspecified-prior containment(**在 M0 冻结 Context cell 之后**)
**为何不能与 M0 并行**(sol 裁决,已接受):M1a 的守门键是 `skill version × Context cell × Workflow family`,而 M0 正要把粗糙的 union cell 拆成 outlier-dominant / level-dominant / mixed / ambiguous。在旧 cell 上验证守门有两种误判风险:因一种机制的失败而静默掉另一种机制下本来有效的 Workflow;或粗粒度下守门看似有效、拆分 Context 后失去意义。代码面不同不等于实验语义可分。
M0 冻结前只允许:静态代码审阅;以及把确定性守门测试写成**以 cell 身份为不透明键**的形式(cell 重定义不致使测试失效)。不跑 live micro,且这些准备**不计为方法进展**。承重的 3-Task micro 必须等 M0 冻结。
- 用已拒绝的 R2 Skill 作为 exposed electricity 上的 **MISSPECIFIED_PRIOR_CONTROL**(不恢复其授权):Source TRY 只影响排序;每个 skill-version × Context cell × Workflow family 至多一次未确认探测;失败后 Runtime 确定性静默该条款;Target-only 路径始终保留;Support 正向只形成 LOCAL_DRAFT,delayed 双正后才激活。
- **不同时引入 uncertainty widening**(那是 M1b,单独一轮,仅当 M1a 证明单次守门不足时才做)。
- 先 0-LLM 确定性测试,再 3-Task micro;守门未真实击发或单候选不可检验 → 停审,不跑全量。
- 该实验只报告"错误先验的损害能否被结构性限制",不作跨域效用 Claim。
- 可证伪判据:若守门后错误先验仍造成超过"+1 次探测/条款"的损害 → 执行语义有漏,先修 Runtime 再谈任何 Source Skill。

### M2 语义层:机制签名的跨域稳定性(零新 Outcome,依赖 M0 的字段)
- 以 AO/IO/LS/TC/VC + missing-block + drift 为**起点**(缺陷标签≠充分 effect modifier);签名须同时描述:缺陷生成机制与局部几何 + Workflow 作用几何与参数绑定 + Consumer/horizon 敏感性 + Support 覆盖与可识别性。
- **反特征钓鱼纪律**:成功标准是"预先提出的签名使 Workflow→utility 在多个已曝光域中更稳定",不是"反复搜特征直到把 Weather/electricity 分开"。
- 可证伪判据:合法可观察量无法稳定分离 → 记为不可识别,abstain,不硬造特征。

### M3 供给层:小型多域机制面板(先 4–8 个真实基域,不建 20–40 域工厂)
- **计数单位纪律(防伪重复)**:base dataset/site = 独立域;机制参数/注入种子 = 域内实例;window/task = 域内重复。同一 electricity 换 20 个种子 ≠ 20 个独立域,不得共同承担元学习的跨域泛化项。
- 每域浅采(6–8 任务,episode 以 UNGUIDED 为主);产出签名 × program 跨域不变性表;按冻结规则授权(≥2 独立基域、无未解释反证);注入 GT 顺带测签名查准/查全。
- 面板出信号后才考虑扩容;可证伪判据同 v1(签名在注入域间都不稳定 → 回到 Observation)。

### M4 确认层:A5 vs A3 vs placebo(fresh 数据,门控开启)
- 开启条件(全部满足):存在合法 Source-derived Skill;M1a 守门已在活体击发;签名有跨已曝光域稳定性;Source 证据来自多个独立基域;两臂同 Target feedback budget。
- 功效前提(**条件化规划情景,不是普适下界**):按 `calib_power_precheck_v1.md` 的 McNemar 双侧 α=0.05 / power 80% / 无多重校正,**在 Δ≈0.25、ψ≈0.40 的假设参数下约需 48 个有效独立配对**;该文件自身注明"同一 cohort 内连续 Task 不独立,n 视为地板"。因此最终 roster 必须跨**独立重置记忆的多个 Target/Domain** 摊开(该文件给的形状:3 targets × 16–20 frozen tasks),不得把单一 cohort 的连续任务当 iid 凑数。若只认大效应(Δ≥0.40、ψ≈0.50)则约 22 对。9-Task roster 只是 development smoke,不作 confirmation。
- 读数:首正试错数、adaptation regret AUC、cumulative harm(两口径)、delayed 效用、abstention risk–coverage、covered/boundary/unsupported Context 分层、**逐 Target 胜负与 worst-domain negative transfer**(DomainBed 教训:不只报 pooled 平均)。
- 这是 NTG(Wang et al. 2019)的预注册操作化,一次昂贵实验同时回答先验压缩、Context 区分力、负迁移控制、Target-local 学习、Slow 纠错五个预注册问题。

### 明确不做(合并 sol/grok 清单)
- 不把 outlier_iqr 的 5 条 conditioned 写成 TRY;不把注入正控/electricity Outcome/Weather conditioned 正例当 Source prior;
- 不打开 NOAA / sealed G3 query;**KDD W3 的 Context 默认也不打开**(已曝光数据不足时才申请);不改 PSM/SWaT 预注册线凑 Source;
- 不建 propensity 记录或反事实估计平台(CRM 只作类比);不在 M0 里顺手修 RLS 的 union binding;
- 不把 RISK-only A5 包装成主实验(Weather 已证只改顺序、检验力低);不跑 9-Task electricity 当 confirmation;
- 不建 IRM / task embedding / 向量库 / learned retrieval / 大型贝叶斯系统 / 通用安全 Gate / "运输 Context 学习器";
- 暂不实施 PROSPECTIVE_VALIDATION(设计保留,见 §5);暂不启用 FCSP 之类方法命名。

### 治理连续项(不占主要回合)
- 授权审计规则一段成文冻结;Weather 每任务符号证据可用性一次裁决;审计计数纪律并入 T5 术语表。

---

## 5. 论文叙事定位(三段贡献链)

1. **自进化 Harness 的证据治理**:学习飞轮会污染自身授权证据流;provenance 防火墙(UNGUIDED 授权 / conditioned 只可维护反驳)+ 现场拦截两次错误授权的审计实录。**声明强度**:"晋升证据自我诱导比例"的定量测量在**本轮检索中未发现直接先例**(不写"第一个");"我们的降权设计规避了 GovMem 的召回代价"为**待验证**假设,需在本项目自身数据上量出 review burden / recall 才能声称。划界:VaG(提案通道/一次性准入 vs 我们的授权通道/持续资格;它自证所需 provenance 字段不存在)、GovMem(横向 trace 相关 vs 纵向技能在场;写入决策 vs 执行权)、Trap of Trajectory(闭环问题陈述来源)、Echo Gap(分数通胀)、ASG-SI(verifier 独立)。
2. **机制条件化的跨域数据 readiness Skill**:以机制签名(候选 effect modifier,S-admissibility 为目标属性)替代数据集元特征;TS 缺陷机制词典。划界:L2C2(统计先验对齐)、meta-feature router 传统、DataMaster/DataEvolver(平坦记忆、无权利分层)、TimeClaw/MemoHarness(成功痕迹复用,无负例否决权、无等预算适应轨迹)。
3. **等预算 A5 vs A3 的 NTG 协议**:时序预报 consumer 上,Source 正负冲突经验能否更快更安全地形成 Target-local Skill;harm/abstention 作为一等读数。最近对标:HASTE(cold vs warm start,−52% iterations / 42%→85% 保留率,但单 seed、无验证门、无 provenance)——其读数形态可借用为次级指标。

**必读清单(写作与设计前)**:VaG (2608.05810)、GovMem (2607.02579)、HASTE (2606.30911)、MTL (2604.14004)、Trap of Trajectory (2605.09330)、MSCE (2607.16621,须核对三层结构重叠);次优先:Compliance Trap (2607.10608,no-memory 对照的度量模板)、Raw Experience to Skill Consumption (2605.23899,"文本可信度不预测下游效用")、TARL (2608.03699,accepted/pending/rejected 三账本)。

**诚实限制(主动写入)**:四层定理各自的假设(已知图、线性/高斯、任务 i.i.d.、模拟器覆盖)在 LLM-agent Harness 中不严格成立;不存在端到端定理;框架是设计原则的理论对应物,实际证明责任由预注册实验承担。

**Provenance 三分类(设计层预留,当前不实施)**:现行二分(UNGUIDED 可授权 / conditioned 只可维护反驳)在开发阶段安全,但作为最终设计会使 Shared Capability 的 fresh 确认逻辑断裂(fresh 确认必然在 Skill 冻结在场时完成)。最终设计应为:
1. **UNGUIDED_DISCOVERY**——Skill 不在场的独立发现,可提出新主动条款候选;
2. **PROSPECTIVE_VALIDATION**——Skill/Scope/roster 在 Outcome 前冻结后的前瞻检验,只能确认或否决**已预注册**条款,不得事后发明条件或扩大 Scope;
3. **ADAPTIVE_CONDITIONED**——Skill 已影响采样/提案的证据,只能反驳、限制、局部校准。
注意:第 2 类不是新的放权——它就是 AGENTS.md §6 已冻结的 Shared Promotion 语义("Program/Scope/Judge/roster 冻结后一次性打开 Outcome")在 provenance 词汇下的显式化。当前提交 a2fb69a 的保守二分规则保持不动,直到 first fault 闭合;实施推迟到 Shared Promotion 真正临近时。

**方法命名(推迟)**:sol 提议 FCSP(Falsifiable Context-Conditioned Skill Prior)/"Evidence-governed, context-conditioned Skill priors for safe time-series adaptation"。内容方向正确,但在第一张被授权 TRY 被 Target 证伪/证实之前不启用品牌名,防止项目滑向"先写运输理论"。

---

## 6. 与 AGENTS.md 的一致性检查

- 不新增 SHA/Receipt/平台(M0b 改动 `feature_context_sha` 是既有字段的必然后果,不是新建哈希体系);M0a/M0b/M1a/M2/M3 各自是单一面修改,且 **M0 → M1a 串行**:M0a 是 0-LLM 诊断(无行为假设),M0b 是唯一 Observation 面变更,M1a 是唯一 Runtime 执行面变更,三者不同时在场;
- 受控注入(M3)是支架:用于机制词典与授权证据,承重 Claim 在自然域(M4);
- 里程碑(A5 vs A3)不变,只是给了它理论身份(NTG)与达成条件(授权内容 + 守门执行 + 足够功效);
- 可信负结果路径在每个 M 都有定义,不默认引出更多基础设施;M0 切不开即是合法的 family 终止证据。
