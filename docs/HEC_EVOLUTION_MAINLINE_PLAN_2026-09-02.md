# HEC 主线规划稿:Harness 进化曲线(v0,呈 sol 核)

日期:2026-09-02 17:xx。地位:**主线草稿,未冻结**。采 sol 同日四项修正(风险门留 held-in /
第一版只开 Skill 生命周期面 / 臂重命名 / NOAA 2025 非 fresh),加主线一处异议(HEC 不以
Source-v3 成功为前置)与五项补件(指标定义 / 统计单位与检验 / 重遇定义 / 课程构成 / 预算量级)。
sol 核准后:§8 待裁项落定 → 冻结件成文 → 主线写入项目 `AGENTS.md` §5 状态锁(本稿不改正典)。

与既有文件的关系:`MAIN_EXPERIMENT_DESIGN_SKELETON_2026-08-28.md` v1 仍是结构设计(C5 拆层、
三档难度、D1-D5、G1-G5);本稿是其 Phase 2「进化轴」的操作化协议,并吸收 P4 线 2026-09-01 的两项
更正(§5.1 天然缺口线、§5.2 serving-side 几何)。`p4u_main_experiment_contract.json`(v1)与
`P4U-v2` 不改写;本稿对其提出的修订以 **P4U-v3 提案**形式列于 §8,待裁。

---

## 0. 一句话

> 过去两周用 5 步的课程考一台需要 40 步才显形的进化机器。HEC 把迭代次数给够、把 Draft 的
> 增益通路接通、把选择交给风险约束下的目标,让「Harness 随经历改善、冻结的不改善」这条曲线
> 有机会被画出来——并让它画出来之后经得起统计与归因追问。

---

## 1. 目标对账:任务书 → 论文四柱 → 证据现状

任务书六句(用户 2026-08-27 原文)逐句落柱:

| 柱 | 任务书条款 | 已有证据(等级) | 缺口 |
| --- | --- | --- | --- |
| Ⅰ 条件化 | 质量随 Task/Consumer/Pattern 变;固定规则一任务增益一任务破坏 | 同一修复预测 +0.0648~+0.4059 vs AD −0.0455~−0.2808,4/4 同向 0 反例(POSITIVE_CONTROL);pooled/per-channel 结论翻转 2/3;ridge −0.0133 vs kNN −0.1200;KDD 天然缺口 `period_median_complete→outlier_*` 双面为正(NATURAL,dev);全局最优固定程序 +0.242 但受害 35%、单条 −5.19(p4x) | **无**。本柱可直接成章 |
| Ⅱ 反馈验证 Gain/Harm | 真实下游反馈;逐类 harm;abstain | 双门(Support/delayed)、逐序列 harm 记账、CONFLICT 语义、零事故记录;Epilepsy2 密封零害;AD 正确弃权 | **无** |
| Ⅲ 经验积累 | Skill/Memory 迭代优化;提升性能与训练效率 | 效率:NOAA 首正重训 69 vs 123(−44%,FRESH 一次性);分类 regret 0.7710→0.0850(dev,已复现);风险拦截 0/3→3/3 | **最终效用**:带经验最终成绩优于无经验(A5 vs A3)——sealed 上未考 |
| Ⅳ 自进化 | 据 Gain/Harm 迭代 Instruction/Skill/Memory/决策策略 | 卡版本链 v0→v3(dev);活体 Scope 修订链在真 Agent 下跑通 1 次(P4W2 origin 2136:Slow 自加 `estimated_level_offset<=0`,15→6 条,delayed +0.182/受害 5%,单条 −0.92 撞 0.30 线);Stage 3 策略编辑 `S3_EDIT_REJECTED`(v2≡v1) | **曲线**:进化臂随经历优于同起点冻结臂;至少一条 Skill 修订后存活并在重遇中改善 |

**论文头条 = 柱 Ⅳ 的曲线**;柱 Ⅲ 的效用差是曲线上的一个对照;柱 Ⅰ/Ⅱ 现在就能写。

---

## 2. 现状定格(2026-09-02,只列承重事实,工件可核)

- **数据身份**:`EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`,239 条结构可读(`p4s`),
  天然缺失 17.119%、270/270 含缺失(`AGENTS` §8.1)。与 P1–P4c(without 版)**不并表**。
- **评价几何已修正**(`AGENTS` §5.2):旧 evaluator 只在训练语料施 Program,被服务序列走 raw;
  P4 系列负结果**只约束训练侧策展与开环树 Router**。serving-side 双管线已接:Scope 内
  `prepared train → program model → prepared serve context`,Scope 外逐位等于 Static。
  **Scoped Harness 的第一场正式考尚未跑。**
- **尾部预算是当前已观察到的绑定约束**(`p4x`):该数据上每个有实质正增益的策略都过聚合线、
  止于 `harmed_fraction≤0.20` 或 `max_single_series_harm≤0.30`;0 LLM 基线与 9 个 live 探针同命。
  `p4y`:**Support-local 上界**——冻结 Scope 类内 6/7 探针存在可过四线的 Scope;它只说明 Slow 的
  任务在 Support 面上是良定的,**不证明** delayed 可迁移,也不预示 Slow 的成功率。
- **Source-v2**(`p4w2`,5 origin,29 LLM,1153 s):`A5_TREATMENT_EMPTY`。五轮只有 origin 2136
  真正考了方法(差一条序列);1896/2616 死于 manifest 格式;2376 正确弃权;2856 Fast 零候选
  (而该 origin 已知存在 Program headroom,§5.1)。三次 Source 开发累计 98 次 LLM。
- **主实验合同**(`p4u`,FROZEN):Target `readable[80:120]`,Source `readable[160:200]`,held-in
  origins [1896,2136,2376,2616,2856],held-out [4056,4296,4536,4776,5016];`p4t` 判
  `ALL_PROPOSED_HELD_OUT_PAIRS_UNEXPOSED`。**held-out 至今 0 读。**
- **Stage 3**:`S3_EDIT_REJECTED`,Harness v2 ≡ v1(commit 737e9e3);策略参数面暂封。
- **Solar**:已下载、md5 核对、**隔离令生效**(完整性核验外一切分析禁止);F2 资格待单独核。
- **NOAA 2025**:已打开 development(`AGENTS` §5),**不得作 fresh**;2025 之后封存段状态须过台账核验。

---

## 3. 主线定义与判据

**主线名**:HEC(Harness Evolution Curve)。**主张**:

> 同一 Harness 在 held-in 长课程中,随处理单元增加,通过 Skill 形成 / Scope 修订 / 撤销,
> 在后续新单元上的**风险约束下效用**逐渐优于同起点、同反馈、同预算但禁止写回的冻结系统;
> 冻结后该优势在未见 held-out 上仍存在。

**同时成立才可主张"自进化"**(sol 四判据,原文入册):

1. A5-online 相对同起点 A5-frozen 的差随课程增长;
2. 至少一条 Skill 经修订后存活,并在独立重遇中改善;
3. A5-online 相对 A3-online 显示历史知识贡献;
4. 冻结后的优势在未见 held-out 上仍存在。

主线补注:判据 2 可由 **Target-local Skill 在 Phase T 内**完成(`AGENTS` §5"局部冲突→有限改
Scope→独立重验→存活→重遇改善"未限定 Source-derived);判据 3 仅当 Phase S 有存活者时评分。

**方法核心句(sol 措辞,2026-09-02)**:

> 一种双时间尺度、受治理的 Harness 进化:历史反馈用于低成本选择,未来反馈用于独立验证。

**HEC-1 收敛环(sol 草图;§10.1/10.2 裁定后写入 §4.2)**:

```text
单元内 probe → Episode bank → 每 k 个单元外环整合一次 → Restricted Draft
→ 历史 replay 筛选 → 后续新单元 prequential 验证 → 激活 / 重遇 / 归档
```

诊断定性(sol):当前系统更像"单次适应后立刻过门",还不是充分展开的"随经历持续改进"。

---

## 4. 协议骨架:一份协议、两个阶段、一个冻结点

### 4.1 Phase S —— Source-v3(加长版)

目的:自然形成 K0(经审计 Source-derived Skill)。规则(sol 定 + 主线补):

- 数据:source cohort ≥3 块(现 `[160:200]` + 待裁两块,见 §8),各 5 held-in origin,≈15 单元;
  只跑 A5-online(建 K0),不计入任何臂对照。
- 每轮 ≤1 次 Slow;Program、阈值、特征词汇**不变**;Scope 修订**单调收窄**,每张 Draft ≤2 次。
- **Runtime 生成 manifest 骨架**(Program、SHA、applicability 由 Runtime 固定),Slow 只决定新增
  Scope clause——两轮格式死亡归零。
- **确定性候选选择**:全部探针完成后,风险拒绝候选按「过四线**至少需排除多少条序列**」最小者交
  Slow(从 Support-A per-series gain 确定性算,不读 Oracle;2616 例:mad 需排 2 条 < hampel)。
- **restricted Draft(门无关保留;sol 2026-09-02 23:xx 有条件同意:保留 ≠ 始终允许继续收窄)**:
  通过 Support 后在**任一**后续验证面(delayed **或**独立重遇)失败的 Draft 作为 restricted evidence
  **保留、不可部署、不成 Active**;修订 ≤2、验证尝试 ≤3,之后归档。失败类型决定**允许的动作集**
  (机械分类,Slow 在集内选,Runtime 执行边界;优先级 FLAGGED > REVISABLE > WAITING):

  | 状态 | 判定(按失败线 → 该线的归因) | 允许动作 | Source-v3 归位 |
  | --- | --- | --- | --- |
  | `WAITING` | 只有 coverage_floor 败 | 不修订;谓词在后续窗口解析 ≥ 底线时自动获得一次验证(不耗修订);整课程未重遇 → 归档 `PATTERN_NOT_REENCOUNTERED`(非 Skill 失败) | 2376 |
  | `REVISABLE` | 尾部/受害分数线败,受害序列全为 NEW_ENTRANT | Slow 可加子句(读该窗口逐序列反馈;总 ≤2) | 2616 |
  | `FLAGGED` | 任一线的受害/负贡献由 CONTINUING 成员主导(可能是 Observation 或 Program 漂移) | **禁止收窄**;可原样再验一次;再败归档 `EFFECT_NONSTATIONARY`,进外环普查作观察/程序漂移信号 | 1896 |

  原稿只在 delayed 失败时保留,系 v2 时墙在 delayed;Source-v3 三条 Draft 全部止于独立重遇后被丢弃,
  故改为门无关。Source-v3 已完结,不追溯。
- 成功判据:≥1 条 Skill 通过 delayed、存活、并在独立重遇中产生收益 → 冻结为 K0。
- **0 存活不阻塞 Phase T**:如实记 `A5_TREATMENT_EMPTY`,K0 = 空(或既有经审计卡,待裁);
  `AGENTS` §2.1 此时只允许组件验收措辞,判据 3 不评分,判据 1/2/4 照考。

### 4.2 Phase T —— 长课程四臂(HEC-1)

**臂**(sol 命名):

| 臂 | 起点 | 课程内写回 | 对照意义 |
| --- | --- | --- | --- |
| A5-online | K0 | 允许(ADD / 收窄 / 撤销 / Draft 多轮) | 完整系统 |
| A5-frozen | 同 K0 | **禁止**(单元内适应照常,不带走) | vs online = **进化** |
| A3-online | 公共 h0 | 允许 | vs A5-online = **历史积累** |
| Static | 无 Harness | — | vs A3-online = **本地适应** |

各臂同数据、同反馈面、同预算(A5 无例外)。Random-edit 为次要消融(HEC-2);LLM-direct 不占主预算。
**K0 为空时(sol 裁)**:A5-online ≡ A3-online,**不得花钱重跑等价 A5 臂**;A5 累积贡献不可评分(判据 3
不评分);只跑 Static、A3-online 与必要的全局 / ScopeFit 对照。主线注:此时"禁止写回"的对照臂
(= A5-frozen 在空 K0 下的形态,即从 h0 起、逐单元适应、不带走)**不与任何臂等价**,是判据 1 唯一的
frozen 对照,建议以 `A3-frozen` 之名保留——待 sol 在冻结件中确认。全局对照 = 0-LLM Best-Safe-Global
(§5.1);ScopeFit 对照 = shadow(§10.2)。

**进化面(HEC-1 只开一个)**:Skill 生命周期 = ADD(自然正例铸卡)+ Scope 单调收窄(冲突触发)
+ 撤销(delayed harm)+ restricted Draft 多轮(同 4.1 规则)。策略参数(`exploration_policy`)、
General Card 指导文本、ordering card、retrieval 规则**全部封存**,留 HEC-2/3 做"哪个面贡献多少"。
编辑**在何时、由何路径产生**(内环即时 Slow vs 每 k 单元外环整合 + replay 筛选)见 §10.1,待裁;
Scope 子句的阈值来源(Slow 目测 vs 工具校准)见 §10.2,待裁。两者都不新增编辑面。

**风险门位置**(sol 修正 1):bounded_risk_v1(hf ≤0.20、msh ≤0.30、material 0.005)留在 held-in
delayed,决定**执行权**;记忆准入(什么值得保留、继续修)按 §5.1 目标选择。两者分开,与 `AGENTS` §4 一致。

**课程(sol 2026-09-03 收缩裁定:HEC-1 只用 KDD 天然单元)**:
- 单元 = (cohort, origin)。**HEC-1 只用 KDD 天然缺口**(serving-side 管线现成、G1 已过门);Phase S 与
  Phase T **不得共用 40 序列块**(K0 学过的序列不得再进 Target)。块分配:Phase S = `[160:200]`(7 origin)
  + `[200:239]`(待扫描);Phase T = `[0:40]`(≈10)、`[40:80]`(≈8)、`[80:120]` held-in(5,+2 待核)、
  `[120:160]`(3)→ **约 26–28 单元**。**不以注入 electricity/traffic 单元补足 40**(sol:同时改变数据域、
  缺陷机制和适配器,曲线变化不可归因);跨数据扩展留 HEC-2。HEC-1 的曲线如实命名为
  **development mechanism curve**,统计功效按 26–28 单元如实报告。
- 构成三要素在 KDD 内检查并进冻结件:重复模式族(同 cohort 跨 origin 的时间重遇)、族内异质(缺口/尖峰
  混合)、若干模式稀疏单元(沉默/隔离)。
- 三种冻结顺序 Forward / Reverse / Interleaved;第一顺序只作仪器健康观察,不按科学读数决定续跑。
- 已 spent 的 dev 块入课标 `DEVELOPMENT`,各臂**不得**读取其历史 outcome(只读 Context 与本课程内反馈)。

**预算(量级,待用户批)**:p4u 每 cell 6 LLM 为基准;40 单元 × 3 个 LLM 臂 × ≈6 ≈ **720 次/顺序**,
三顺序 ≈ 2200 次;墙钟按 Source-v2 实测 230 s/单元臂 ≈ 8 h/顺序。`llm_cache` 在各臂 prompt 未分叉
前共享调用(合法:同 prompt 确定性复用)。fits 秒级,不构成瓶颈。

### 4.3 Phase F —— 密封终考

- 课程结束冻结 A5* / A5-frozen* / A3* / Static*;Fast-only、零反馈、零写回、一次读。
- **F1(能力终考)**:`p4u` 已冻结 `[80:120]` × held-out origins [4056…5016],`p4t` 已清。
- **context 侧覆盖:只作分层报告,不作筛选**(sol 2026-09-03 裁,取代主线原"预选"):Scope 是部署可见
  谓词,考前只读 Context 可算出冻结 Skill 在每个 held-out origin 上的匹配序列数——该数只用于把终考读数
  **分层报告**(匹配 / 未匹配),**不得据此筛掉或更换已冻结的 held-out origin**。程序:所有臂先生成全部
  输出 → 一次性打开全部 Outcome → 计分。覆盖为零的层只承担安全/隔离读数(Epilepsy2 教训照录)。
- **F2(Solar)**:资格单独核(台账 `AGGREGATE_SEEN`、outcome 未算);模式族与 KDD 缺口 Skill 大概率
  不交,预期承担 F2 安全/隔离读数;隔离令在冻结前持续生效。
- NOAA 2025 之后封存段:须过台账;即使合格也只是 F1 级(同族新实例),非 F2。
- **Phase F 风险预告(Source-v3 独立重遇读数,2026-09-02;措辞依 sol 收窄)**:独立重遇与 held-out
  **共享 Fast-only、零反馈的运行几何,但它仍是 exposed development 上的 +240 步代理,不等同于真正
  held-out**(held-out 距 held-in 1200+ 步,更替只会更大)。三条过 delayed 的 Draft 在重遇上 0/3 通过:
  谓词解析出另一批序列(6→9、5→2、7→15),尾部主要来自未证据化的**新进入者**(2616:受害 3/3 皆新进入,
  持续成员 6/6 非负),1896 另有持续成员条件效应翻号(3/5)。**即便 Skill 过 held-in 三门,Fast-only 到
  新窗口仍可能尾部绑定。** 四候选对策已裁(sol,§10.11):(a) 证据有界 Scope 作候选形式、**不设默认**;
  (b) 部署期 outcome-free 检查**不入 HEC-1**——它是新的 Risk/Scope 决策机制,须先 0-LLM 诊断;
  (c) 模式持续性特征 → HEC-3 新 cohort 前瞻;(d) 覆盖语义:未来合同把**安全准入**与**能力证据充分性**
  分开,HEC-1 只以 `WAITING` 状态承接并报告累计覆盖/重遇次数,不设门值。

---

## 5. 读数与预注册

### 5.1 指标(不做合成分,与 D5 一致)

- **主图**:部署策略相对 Static 的**累计效用**(每单元 mean per-series gain,serving-side 几何)
  随单元序号;四臂同图,A5-online − A5-frozen 的差单独画。
- **副图**:累计 harm 事件数(单元级 hf>0.20 或 msh>0.30)——online 必须 ≤ frozen。
- **Best-Safe-Global baseline 与 advantage**(sol 更名:能被 Scoped policy 超过的东西不是 oracle):
  每单元在固定小菜单(冻结 P1 单算子 + §5.1 已知组合族)上取**过风险预算的最优全局程序**,无则 identity;
  报告各臂相对它的 **advantage**(可为正——"超过最安全的全局程序"正是 §5.2 之后本线要证的)。
  真正的 oracle 须覆盖 Scoped policy(如 UID 级逐序列选择,§5.1 的 +0.6106 口径),**只作离线上界**报告。
  两者事后 0 LLM 离线算,走 `ORACLE_BANNER` 隔离墙,**不进任何臂**。
- **效率**:到首个安全 Skill 的 fits / LLM;每单元 LLM、fits、墙钟;零 LLM 召回部署占比。
- **生命周期**:铸卡数、修订成功率、撤销数、Skill 存活率、**重遇收益**(定义见 5.2)、覆盖率(treated/served)。
- **归因三分账**(逐单元,不需额外运行):A5-online − A5-frozen 拆为 (a) 召回了 frozen 没有的新卡;
  (b) frozen 重复推荐被拦/被害而 online 已收窄或撤销;(c) 探针位释放(SUPPLY_STARVATION 机制)。

### 5.2 统计单位、定义与检验(预注册)

- **重遇**:后续单元中某 Active/Draft Skill 的 Scope 在部署可见特征上匹配,且被供给或部署;
  重遇收益 = 该单元读数 − frozen 臂同单元读数。
- **统计单位 = cohort**;同 cohort 不同 origin 是时间上的重遇,非独立样本(§5.1 anchor 冻结推论)。
- 检验:逐单元配对差(online − frozen)**符号检验,单侧 α=0.05**;三顺序中 ≥2 个终局累计差 >0;
  同时报中位数与 bootstrap CI(按 cohort 重采样)。

### 5.3 预注册预测与 first-fault 映射(`AGENTS` §6)

| 预测 | 内容 | 机制依据 | 若不成立 → first fault |
| --- | --- | --- | --- |
| P1 进化 | online − frozen 累计安全效用随单元数拉开;三顺序 ≥2 个终局 >0;harm online ≤ frozen | 冷启动召回率 ≈29%,召回替代重搜价值大;starvation 释放已在 S2a r2 实测 | 召回部署劣于 frozen 当场搜索 → Scope 过宽(记忆面);铸卡 ≈0 → Fast 候选枯竭(供给面);差为正不显著 → 触发密度不足(课程面) |
| P2 存活链 | Phase T 内 ≥1 条 Draft 经 ≤2 次收窄过 delayed、存活、≥1 次重遇优于 frozen 同单元 | p4y 为 Support-local 上界(6/7),**不证 delayed 可迁移**;2136 链到 delayed 差一条序列 | 收窄后仍撞单条线 → 先按 §10.3 归因(范围进入 vs 成员漂移),再定面;从未到 delayed → Support 再探预算 |
| P3 积累 | 仅 Phase S 有存活者时评分;A5-online − A3-online 累计 >0 | Source 与 Target 同数据族、缺口模式共享 | K0 在 Target 上 Scope 匹配 ≈0 → K0 不可达(只分层报告,不换考场);匹配但无收益 → 跨 cohort 泛化 |

**"2–3 周拿到曲线"是目标,不是证据**(sol 语,入册)。上表三条不成立时各有归属,均**不**读作"进化无效"。

**Phase S 多轨迹的预注册机制假设(依 Source-v3 重遇归因,2026-09-02;待 sol 核)**:

| 假设 | 内容 | 可检验读数(每个验证窗口机械记录) | Source-v3 单轨迹读数 |
| --- | --- | --- | --- |
| H1 成员更替 | 窗口局部谓词在新窗口解析出未证据化的新进入者,尾部伤害集中于其中 | 受害序列中 NEW_ENTRANT 占比;持续成员受害率 | 2616:3/3 新进入;1896:msh 为新进入 T262 |
| H2 效应非平稳 | 同一序列上程序的条件效应跨窗口翻号,谓词区分不了 | 持续成员中 + → − 翻号率 | 1896:3/5;2376:1/1;2616:0/6 |
| H3 覆盖即流行率 | 模式条件化 Skill 的单窗口治疗数随模式流行率波动,与 Skill 质量无关 | 离开者中"因特征退出谓词"的占比;离开者在 delayed 的增益 | 2376:4/5 离开;两最大赢家(T264 +0.905、T262 +2.145)均因尖峰移出窗口离开 |

三条在 Phase S 每条轨迹上机械落账(0 LLM,§10.3 归因字段的直接推广);Phase S 的判词除 Skill 存活外,
须报 H1/H2/H3 的跨轨迹一致性。

### 5.4 证据等级标注

Phase S/T 全部 `DEVELOPMENT`(数据身份已曝光);Phase F 的 F1 为 `FRESH`(同族新实例,Outcome 未见),
F2 为 `FRESH`(新族)但预期只承担安全读数。报告按 `CAPABILITY / MECHANISM / INFRASTRUCTURE /
INSTRUMENT / NEGATIVE / INCONCLUSIVE` 分列(`AGENTS` §8)。

---

## 6. 纪律与护栏(承接既有站规)

- 单假设:HEC-1 只开 Skill 生命周期面;G3 七门(双门 / 容量 / harm 阈 / 越权 / 隔离 / 阶梯 v2 /
  Scope 语义)不可编辑;阈值 0 改动、算子 0 新增。
- 曝光:各臂只读部署可见 Context + 本课程内反馈;历史 outcome、oracle、held-out 全部隔墙;
  仪器故障(BACKEND_UNAVAILABLE 等)单列,不得写成科学判词。
- 反过度工程(`AGENTS` §7):一个 runner package、一个主报告、一个 smoke;SHA 预算 0;不新建 Gate /
  Manifest / 审计平台;Runtime manifest 骨架是**去掉** LLM 不该承担的负担,不是新增层。
- 委派:一层;方法决策、结果整合、裁定由主线负责。

---

## 7. 排期(目标,非承诺)与砍单

**执行序(sol 2026-09-03 最终版,详见 §12)**:D1 → D2–D4 → Phase S → Phase T(KDD 26–28,三顺序)→
课末统一读数 → 冻结末态后开 p4u held-out → HEC-2(pooled/per-channel、Risk 面、跨数据)→ TSFM。
Source-v3 已完整收口(2026-09-02),不再改其协议。

| 周 | 内容 | 产出 |
| --- | --- | --- |
| 1 | D1 路由伤害诊断(≤400 fits,0 LLM);D2–D4:课程供给扫描、HEC-1 合同、一个 0-LLM smoke;四份接线落地 | `p4ab` 诊断件、`hec1_contract`、smoke |
| 2 | Phase S(多 Source cohort 建 K0,0 存活如实继续)→ Phase T Forward(≤500 LLM,仪器健康)→ 依仪器完整性放行 Reverse / Interleaved | 曲线 v0、生命周期表、外环步记录 |
| 3 | 课末统一给出曲线 / 风险 / 谱系 / 成本 / H1–H3 → 冻结全部末态 → 一次打开 p4u held-out;论文柱 Ⅰ/Ⅱ/Ⅲ 成章 | 终考工件、论文初稿 |

**砍掉**(至 HEC-1 收口前不排):Stage 3 复跑、P4b 类门实验、新增审计 / SHA 工件、5–7 单元短课、
策略参数与文本编辑面、LLM-direct 臂、S2b 跨任务迁移。

---

## 8. 待裁定清单

**sol(方法)——2026-09-03 收缩裁定后状态**:
1. 两阶段单协议:Phase T 不以 Phase S 成功为前置——**已裁**("0 存活也如实继续")。K0 为空时的臂
   集见 §4.2(不重跑等价 A5;`A3-frozen` 命名待冻结件确认)。
2. ~~受约束 oracle 的安全 regret~~ → **已裁更名** Best-Safe-Global baseline + advantage(§5.1);
   统计单位 = cohort、§5.2 检验——冻结件中定。
3. ~~context 侧覆盖预选~~ → **已裁**:只分层报告,不筛选、不更换 held-out;F1 = `p4u` held-out;
   Solar 仅 F2 安全读数。
4. restricted Draft 上限 → **已裁**(修订 ≤2 / 验证 ≤3,三态机 §4.1);Source 候选选择 = 最少排除序列数
   ——冻结件中定。
5. **P4U-v3**:Phase T 课程域 = KDD 非终考块 `[0:40]`/`[40:80]`/`[80:120]` held-in/`[120:160]`(≈26–28
   单元,DEVELOPMENT);Phase S = `[160:200]` + `[200:239]`;held-out 不动——**方向已裁**,块表在冻结件中
   定。K0 为空时是否允许既有经审计卡作 K0,或严格取空——**待裁**。
6. ~~注入 cell 补足 40~~ → **已裁否**:不入 HEC-1;跨数据扩展留 HEC-2。
7. 双环分离 → **已裁入 HEC-1 冻结面**;子项 (a) k=5、(b) 内环即时 Slow 关闭、(c) `replay_fits` 独立账本
   ≤ 课程 fit 25% 为主线拟值,**sol 核冻结件时确认**。不可协商:replay 不授部署权。
8. Scope 工具五步链 → **已裁入 HEC-1 冻结面**;(a) 阈值校准 = "过预算最宽阈值、并列取粗箱"、(b) ScopeFit-only
   以 shadow 实现为主线拟值,**sol 核冻结件时确认**。

11. ~~Source-v3 判词措辞~~ **已裁(sol 23:xx,同意)**:`A5_TREATMENT_EMPTY` 维持 + 机制观察("本轨迹
    3/3 delayed-pass Draft 未通过独立重遇;观察到新进入者伤害、持续成员效应翻号和覆盖塌陷");H1–H3 入
    Phase S 预注册;n=3 不升为普遍结论;"Slow 不是瓶颈"改为"Slow 的格式表达、合法收窄和持久化不再是
    瓶颈;其所选 Scope 能否产生稳定条件效应仍未证明"。
12. ~~restricted Draft 门无关保留~~ **已裁(sol,有条件同意)**:保留 ≠ 可继续收窄;三态规则见 §4.1。
13. ~~Phase F 对策候选~~ **已裁(sol)**:(a) 候选形式非默认;(b) **不入 HEC-1**,先 0-LLM 诊断
    (§8-17);(c) HEC-3;(d) 安全准入 / 能力证据充分性分离,HEC-1 只报告不设门。
14. ~~零候选轮仪器补记~~ **已裁(sol,同意)**:记脱敏 Fast 原始决定并区分主动弃权 / 空输出 / 格式
    错误;Fast-only retrieval 与 held-in candidate supply 分列两路径;2136/2856 文案只追加式勘误。
17. ~~路由伤害 0-LLM 诊断~~ → **已完成(2026-09-03,106/400 fits,0 LLM;工件 `p4ab_routing_harm_diagnostic`)**:
    总判 `NO_OUTCOME_FREE_SEPARATOR`(四信号 AUC 0.65/0.51/0.60/0.45–0.40,均未过 0.75;仅预测分歧 CI 下界 >0.5);
    `ROUTING_HARM_NOT_DOMINANT`(严重受害 5 条中 moved≤1 占 0.4;但另 3 条只动 2/6/9 点,改动量 AUC <0.5 →
    伤害与改动量脱钩);反事实 `NO_CLEAR_DIFFERENCE`(per-channel 严重受害清零、msh 0.50→0.08 / 0.88→0.14,
    但轻度受害翻倍、hf 0.20→0.35 / 0.15→0.30)。**裁定**:HEC-2 Risk 面按设计不开;登记 Observation 缺口
    (`AGENTS` §6);HEC-2 Consumer 轴预测**拆为尾部(msh)与基座(hf)两条**(见 §10.11 D1 补记)。
18. ~~HEC-1 冻结面~~ → **已裁批准(sol)**:双环学习 + Scope 阈值工具五步链 + restricted Draft 三态机 +
    measurement runner + 长度 ≤2 组合程序 + 正负先验对称 + 提案缓存与稳定供应;**不加入**新 Risk 面、
    新观察特征,**不修改**风险/覆盖阈值。

**用户(执行)**:
15. 预算——sol 建议**分批授权,不一次批 2200**:D1 ≤400 fits / 0 LLM;Phase T **Forward ≤500 次**;Forward
    运行机制无故障后再依次放行 Reverse、Interleaved;**是否继续只依据仪器完整性,不得依据 Forward 效果正负
    修改合同**。Phase S 预算另列(≈15 单元 × 6 + 外环 ≤2/步)。待用户逐批放行。
16. 密封开启授权在 Phase F 前单独批(F1 一次;F2 视资格)。

---

## 10. 设计诊断与候选优化(未裁定;2026-09-02 主线提出、sol 逐条批注)

**地位**:候选,不冻结。sol 已原则接受 1/3/4/5/6/7,有条件接受 2,后置 8,限定 9;三处纠正已并入
各条。**三句不写入正式规划**(sol,入册防复发):(i) "p4y 的 6/7 意味着 Slow 成功率可接近 6/7"——p4y
只是 Support-local 上界,不证 delayed 可迁移;(ii) "把动作空间变难,经验才能产生质量差"——实验不得
人为制造困难;(iii) "0.30 几乎必然被撞破"——目前只能说它是已观察到的绑定约束。

### 10.0 诊断:八处瓶颈

| # | 瓶颈 | 表现 | 证据 |
| --- | --- | --- | --- |
| A | 学习信号稀疏 | 每单元 ≤3 探针的逐序列增益;Slow 每轮 ≤1 且只在失败时触发 | 进化步数受"失败发生率"限制 |
| B | 选择与验证同源 | 编辑接受/拒绝只看当前单元(n=1) | `S3_EDIT_REJECTED`;Source 0 存活 |
| C | Scope 阈值靠 LLM 目测 | 修订成功 1/5;Support-local 上界 6/7 | `p4w2` vs `p4y` |
| D | 单算子菜单下经验只买成本 | 枚举即可覆盖;质量同、fits 省 | NOAA 69 vs 123 |
| E | 冲突不分类 | Support 正 / delayed 负一律收窄 Scope | 可能是成员漂移而非范围泄漏 |
| F | 先验两种失效 | 正面过宽 → 伤害;负面过宽 → 过度保守 | NAB 20 张负面卡 → 4/4 弃权 |
| G | gate 式运行 | 首个故障全停;判词在开头 | 40 单元课程不可持续 |
| H | 观察面停滞 | 22 维、7 维在旧数据恒定 | §5.1:15 维特征分不开单条伤害 |
| I | Scope 跨窗口不稳定(09-02 新增) | 同一谓词在 +240 步解析另一批序列;尾部来自新进入者;部分持续成员效应翻号 | Source-v3 重遇 0/3(§10.11) |

**B 是根**(sol 同意:系统"单次适应后立刻过门");**I 是 Phase F 的直接风险**。

### 10.1 双环分离(sol:接受,HEC-1 核心变化;§8-7 待裁子项)

- **做法**:内环 = `run_online_round` 逐单元 probe → 双门 → 单元内部署,**只写 Episode bank**(逐序列
  增益、部署可见特征、`serving_scope`、准入结论、§10.3 归因)。外环 = 每 k 单元对 A5-online / A3-online
  各触发一次:(1) 确定性普查(按 `task_consumer_key × 行为指纹` 分组;指纹 = 逐序列增益向量去重,
  §5.1 纪律;沿用 `group_fault` 分组与阶梯 v2 计价)→ 三类候选:重复 POSITIVE 无卡 → ADD;restricted
  Draft 有新证据 → 修订;Skill 重复 CONFLICT/NEGATIVE → 收窄/撤销;(2) Slow 每组 ≤1 次提案
  (Scope 子句走 §10.2);(3) **replay 筛选**:在**本臂本顺序已处理的 (series, origin) 对**上重解析
  Scope、重算逐序列增益(Ridge 秒级,记 `replay_fits`,0 LLM),任一已处理 cell 违反风险预算或聚合
  ≤ material 即淘汰;通过者只成 **restricted Draft**;(4) Draft 以 `requires_target_support` 供给
  后续新单元,过 Support + delayed 方 Active;(5) 连续 2 次新单元失败 → 归档为证据。
- **边界**:replay 只筛选/修订、**不授部署权**;bank 只含本臂已处理对,不含未来单元、held-out、
  他臂;Fast 只见 Skill,不见 Episode(§2.2/§4)。A5-frozen 无外环;两 online 臂外环规则相同。
- **读数**:每外环步的候选数 / replay 淘汰数 / Draft 数 / 新单元激活数 / 归档数;曲线拐点与外环
  步对齐;`outer_llm`、`replay_fits` 单列。

### 10.2 Scope 拟合工具(sol:有条件接受;§8-8 待裁子项)

- **sol 边界**:不是"机器找出所有安全 Scope、LLM 挑一个",而是五步链:**Slow 提出有语义的
  feature + direction → 数值工具只在历史 Episode bank 上校准 threshold → replay 淘汰明显错误候选 →
  后续新 held-in 单元验证 → 通过后才取得执行权**。
- **做法**:Slow 输出 `{feature ∈ 冻结特征词汇, direction ∈ {<=, >=}, rationale}`,**不输出数值**;
  工具在 bank 内该程序的逐序列记录上,候选阈值 = 该特征的**冻结分箱边界**(`observable_numeric_bin`),
  取满足风险预算的最宽阈值(并列取更粗分箱);无可行阈值 → 返回 `NO_FEASIBLE_THRESHOLD`,Slow
  可换 feature/direction(≤2 次)或弃权。随后走既有 `scope_narrowing_preflight`(严格更窄 / 部署可见 /
  子集 / 至少排除一条)→ §10.1 replay → restricted Draft。
- **反 Router 边界**:工具不选程序、不决定是否行动、不选特征;它只把 Slow 指名的语义方向校准到
  分箱边界;rationale 入工件可审。
- **ScopeFit-only 对照(sol:必须有)**:同一冲突上,工具**自行**在词汇 × 方向 × 分箱上按同一目标取最优
  (feature, direction),与 Slow 的选择并列记录;两 Scope 在后续单元上按各自解析集评分(需少量
  `shadow_fits`,0 LLM)。若工具单独即达到同等新单元表现,论文不得把 Scope 修订记为 LLM 贡献。
  形态(shadow vs 第五臂)待裁。
- **读数**:修订成功率(提案 → 激活);Slow 与 ScopeFit 的 (feature, direction) 一致率;二者的重遇读数。

### 10.3 冲突二分归因(sol:接受;先归因,不自动指定修法)

- **做法**:`open_delayed` 记录 Support 治疗集 S_A 与 delayed 重解析集 S_B;每条 delayed 受害序列标
  `NEW_ENTRANT`(∈ S_B∖S_A,"新进入 Scope 后受害")或 `CONTINUING`(∈ S_A∩S_B,"持续成员效果
  漂移");Episode 加 `conflict_attribution = {new_entrant, continuing, dominant}`。故障路由表在
  `RISK_GAP` 下加两个**归因标签**(不加新 cause、不加新 surface),Slow 在上下文中读到归因后自行
  决定:收窄 Scope / 保持 Scope 转 restricted 等重遇 / 弃权。2136 一轮的 T31 系重解析新增,可从
  `p4w2` 工件核对其是否即受害序列。
- **读数**:两类冲突占比;两类之后的修订存活率(本身是论文级机制读数)。

### 10.4 长度 ≤2 组合程序入课(sol:接受结论,理由 = 组合必要性已有实证)

- **做法**:课程程序空间 = 冻结 P1 Common DSL 单算子 + 长度 ≤2 组合(P4 线已枚举 396,窗口校验器
  0.35 不变、算子 0 新增);Fast 在该空间提案。理由**只**是 §5.1 实证:单算子在 2856 全不稳定,
  `period_median_complete → outlier_*` 双面为正、顺序效应 +0.21~+0.28。
- **纪律**:普查前按逐序列增益向量对程序去重(§5.1:396 中大量别名)。

### 10.5 先验对称与受限探测(sol:接受)

- **做法**:负面/风险卡的 Scope 与正面卡走**同一**分箱归纳(无全局风险卡);风险卡权限 =
  `restricts_probe`(降到探测序末位、预算允许仍可探),**永不硬禁**;匹配语境下该程序后续取得
  POSITIVE → 风险卡撤销。Phase T/F 开跑前做 context 侧覆盖预算(只读特征):记录 K0 正/负卡
  匹配率,负卡匹配率过高只**披露**不改。
- **读数**:风险卡匹配单元上 A5-online 与 A3-online 的弃权率差;NAB 式过度保守是否复现。

### 10.6 measurement 式运行(sol:接受)

- **做法**:runner 异常分两类。`UnitFault`(候选失败、校验器拒绝、LLM 输出多次不合法、cell 级
  LLM 预算耗尽)→ 当前单元 identity 弃权、记录、**继续**;`RunFault`(仪器:BACKEND_UNAVAILABLE
  过重试策略;泄漏:G2/oracle 墙;预算:全局 LLM/token/时间;协议/数据错误)→ 全场中止。判词在
  课程末按 §5.2 预注册检验给出。模板 = split-3 的 `LLM_CELL_BUDGET_EXHAUSTED` 原子丢弃语义。

### 10.7 提案稳定化(sol:接受;Source-v3 不为随机性重跑)

- **做法**:`llm_cache` 以规范化 prompt 字节为键——各臂/各顺序遇**相同** prompt 得相同提案,使
  A5-online 与 A5-frozen 的分叉只来自记忆差异而非采样噪声;解码参数固定并记录(中继不支持则
  披露);抽样稳定性(重复抽样一致率)作仪器读数报告。**不**为此改探测策略(策略面封存;Stage 3
  已否 `supply_reserved_probe_slots=1`)。

### 10.8 Observation 面进化(sol:后置 HEC-3;新 cohort 前瞻冻结)

- **做法**:候选特征由机制提出(缺口相对预测起点的位置、缺口簇密度、填补残差尺度等,均部署可见)
  并在评估前成文冻结;只在**新** development cohort 上评估(不得再拟合已看过的六个 origin);
  预注册检验 = 同一 Scope 类下"存在可行 Scope"的探针占比是否提升;通过才入特征词汇(触发
  `dependency_shas`/lock 轮转,作仪器变更披露)。

### 10.9 尾部判据(sol:仅未来合同;0.20/0.30 不动)

- **做法**:Phase F 报告披露 msh 在 X/Y 次拒绝中为绑定约束(观察陈述);下一份 fresh 合同冻结前,
  在已曝光 development 工件上比较候选判据(如受害分数 + 尾部伤害总和/总增益)并预注册其一;
  **不**追溯应用于任何已冻结合同。

### 10.11 Source-v3 收口归因与主线裁定建议(2026-09-02 21:xx;呈 sol,§8-11~14)

**结果**(工件 `p4w3b_source_line_v3_clean_post_fix_replicate_1.json`,执行方口径):5/5 origin,29 次 LLM,
`A5_TREATMENT_EMPTY`,0 Skill 存活;4/4 Slow 调用合法 clause、零格式失败;4/4 preflight 严格更窄
(14→6、15→8、13→7、18→9);delayed 3/4 过;**独立重遇(+240,Fast-only)0/3 过**。所有修复完成后的
第一条完整科学轨迹。累计 Source 开发 172 次 LLM(本跑 29 + 仪器无效 14 + 探测 1 + 此前 128)。

**主线机械归因**(0 LLM;`per_series_gain` 为 support_a 面 20 条字典序位置向量,delayed 非零位与
执行方 `delayed_serving_series` 逐条相符,口径已核):

| 重遇 | 更替(delayed→重遇) | 持续成员 | 新进入者 | 离开者 | 失败线 |
| --- | --- | --- | --- | --- | --- |
| 1896→2136 | 6→9(续 5 / 新 4 / 离 1) | 3/5 翻号(T267 +0.017→−0.147、T269 +0.276→−0.421、T270 +0.286→−0.337) | 4 中 1 受害;**msh=T262(−0.502)为新进入** | T264(+0.905) | 聚合 + 尾部 |
| 2376→2616 | 5→2(续 1 / 新 1 / 离 4) | T33 +0.878→−0.09 | 1,正 | T261/T265/T269/T37 离开 z∈[3,4] 带 | 覆盖率 |
| 2616→2856 | 7→15(续 6 / 新 9 / 离 1) | **6/6 非负**(T36 −0.078→+0.133) | 9 中 3 受害:T260(−0.883,msh)、T263(−0.645)、T267(−0.201),**全部新进入** | T262(+2.145) | 尾部 |

三个失败机制不同:2616 纯"新进入者伤害"(条件效应在持续成员上稳定);1896 混合(聚合败于持续成员漂移、
尾部败于新进入者);2376 覆盖塌陷(带宽 Scope 在新窗口几乎无成员)。两个最大赢家(T264、T262)均因尖峰
移出窗口而**正确地**离开 Scope。

**R1 判词**:`A5_TREATMENT_EMPTY` 维持;不写"随机性差异"——预定口径是对**无信息 null** 的默认,本 null
有信息:三条 Draft 止于同一道门,机制可复算。定性 `MECHANISM` 级观察 + 注册 H1/H2/H3(§5.3)供 Phase S
多轨迹检验。n=3 不确立机制,足以从"随机"改为"有方向的可检验主张"。此即"先归因、不自动指定修法"。

**R2 生命周期**:采执行方诊断,restricted Draft 改门无关保留(§4.1 已改,待裁)。

**R3 Phase F 预演**:独立重遇几何 = held-out 几何。候选对策(均待裁、不改阈值):
- (a) **证据有界 Scope**:归纳默认双侧、只覆盖证据中出现过的分箱;2376 即此形态,代价 = 覆盖(4/5 离开)。
- (b) **部署期 outcome-free 逐序列合法性检查**(`AGENTS` §3 允许"不读取 Outcome 的确定性合法性检查"):
  对解析进 Scope 的每条序列,计算程序对其 serving context 的改动量(修改点比例、改动幅度/稳健尺度),
  超出证据覆盖范围 → 该序列弃权(逐序列,非整卡)。按"程序效应"而非"模式存在"再筛新进入者;held-out
  合法。前置:核 22 维词汇中"探针方向"特征是否已含此量;若无,归观察面(HEC-3,前瞻冻结)。
- (c) **模式持续性特征**(HEC-3 候选):2616 提示"极端偏差在此前窗口是否出现过"(仅用过去数据)可能区分
  可修的反复异常与不该抹掉的新事件;**必须**在新 cohort 上前瞻冻结验证,不得在这 5 个 origin 上拟合。
- (d) **覆盖率底线语义**:对模式条件化 Skill,单窗口 ≥5 治疗数度量的是模式流行率,不是 Skill 质量;下一版
  合同宜改为"跨窗口累计已证据化序列数",单窗口只报告。协议设计问题,非阈值放宽;不追溯。

**R4 零候选轮**:v2@2856、v3@2136,同 cohort 不同 origin;v2 在 2136 有候选、v3 在 2856 有候选 →
现象随采样非随数据,归 §10.7。**仪器缺口**:工件未记 Fast 原始决定——"明确弃权并给理由"(正确行为)与
"空输出/格式失败"(故障)意义相反,后续记脱敏原始回复。另:2136 轮 `retrieved` 含 1896 Draft
`scope_narrowed_942dccd4ee` 而 `resupplied=[]`——重遇门下 Draft 走 Fast-only、不再作候选供给;HEC-1
held-in 供给走 `requires_target_support` 探测路径;冻结件须分列两路径。

**R5 记录更正**:`candidate_supply_instability` 条目沿用 v2 原话,2136 条目误写"at origin 2856";
工件不改(历史不覆盖),代码已修,入册。

**sol 裁定(2026-09-02 23:xx)与主线复议**

- R1 同意;R2 有条件同意(三态机已入 §4.1);R3 分项处理;R4/R5 同意。"有信息的机制性 null,
  不从 n=3 上升为普遍结论"。措辞两处收窄已并入 §4.3/§8-11。
- **关键技术纠正(sol)**:此前已有序列在 serving context **修改点为 0 时仍受严重伤害**——伤害来自
  "被路由到 program model",不是 context 被改。主线复议:这可从 §5.2 几何直接推出——Scope 内序列走
  `prepared train → program model → prepared context`,pooled Ridge 的 program model 在**全部 Scope 内
  序列**的准备后训练行上拟合;自身未被碰的序列一旦划入 Scope 就换了模型,而模型因他序列的行被改而变。
  故"改动量"不是风险载体,**模型路由**才是;主线原 (b) 撤回为对照项。更合理候选(sol):raw model 与
  program model 的**预测分歧**(部署期两份预测均可算,outcome-free);新进入者到历史安全证据区的**距离**;
  Program 对该序列的行为是否**超出历史覆盖**。先在已曝光数据上 0-LLM 诊断,再决定是否作为新 Risk 面冻结
  (HEC-2,不入 HEC-1——否则双环 / Scope 工具 / Risk 面同时改动,归因尽失)。
- **主线推论(供 HEC-2 Consumer 轴预注册)**:路由伤害是 pooled 的跨序列溢出;per-channel Ridge 下每条
  序列只受自身训练行影响,溢出应消失。预测:**同一 Scope 决策下,per-channel 的新进入者伤害显著低于
  pooled**——柱 Ⅰ"标准随 Consumer 变"的一条具体可证伪读数。
- **诊断规格(§8-17)**:材料 = Source-v3 三个重遇窗口 + 四个 delayed 窗口,Scope 内序列 ≈50、
  受害 ≈10;逐序列算三候选信号 + 改动量对照;读数 = 各信号对受害/非受害的分离(AUC 与单阈值命中);
  反事实 = per-channel 重拟同批 cell 的逐序列增益。Ridge fit only,0 LLM;结论只决定是否冻结新 Risk 面。

**D1 结果补记(2026-09-03,执行方 grok;主线核对七窗口 S/E/受害与主线手算逐条一致)**:n=46 Scope 内序列,
`harmed_material` 12、`harmed_severe` 5。四信号对 material 的 AUC:S1 预测分歧 0.65 [0.52, 0.81]、S2 证据区
距离 0.51、S3 行为超覆盖 0.60、S4 改动量 0.45 / 0.40——**全部 `DOES_NOT_SEPARATE`**(预注册 ≥0.75 ∧ CI 下界
>0.5);单阈值命中表为空;严重受害的 div = [0.66, 0.82, 0.90, 1.26, 1.78] 落在非受害分布(中位 0.59、p75 0.88、
最大 2.19)之内。**路由核验** `ROUTING_HARM_NOT_DOMINANT`(moved≤1 占 2/5);但另 3 条仅动 2/6/9 点(1–5%)却伤
−0.42~−0.88,且改动量 AUC <0.5——**伤害与 context 改动量脱钩**,sol 论点的实质成立,严格"零改动"判据未过半。
**反事实**(pooled → per-channel,同数据/Program/Scope/指标):严重受害 3→0、2→0,msh 0.50→0.08、0.88→0.14、
0.09→0.06;轻度受害 4→7、1→1、3→6,hf 0.20→0.35、0.05→0.05、0.15→0.30 → 预注册判 `NO_CLEAR_DIFFERENCE`
(规则把"新进入者受害减半"与"msh 不升"捆在一起,二者反向)。**主线读法**:pooled 死在 msh 线、per-channel 死在
hf 线——**同一 Scope 决策在两种 Consumer 下以不同形状失败**(尾部 vs 基座),是柱 Ⅰ 在 exposed development
上的新观察;HEC-2 Consumer 轴预测改写为两条:P-C1 "per-channel 消除严重尾部(msh)";P-C2 "per-channel 提高
轻度受害基座(hf)"。**裁定**:HEC-2 Risk 面不开(无 outcome-free 分离量);Observation 缺口登记进 HEC-1 合同 §2;
HEC-3 观察面候选(模式持续性)优先级上调。规格矛盾 #2(round 2856 两门口径不一)转为 D4 W4 "单一权威门"验收项。

### 10.12 处置总表

| 条 | 瓶颈 | 阶段 | 裁定状态 |
| --- | --- | --- | --- |
| 10.1 双环分离 | A/B | HEC-1 核心 | 原则接受;子项 §8-7 |
| 10.2 Scope 工具 | C | HEC-1(Source-v3 收口后) | 有条件接受;子项 §8-8 |
| 10.3 冲突归因 | E | HEC-1 | 接受 |
| 10.4 组合入课 | D | HEC-1 | 接受(理由已改) |
| 10.5 先验对称 | F | HEC-1 | 接受 |
| 10.6 measurement 运行 | G | HEC-1 runner | 接受 |
| 10.7 提案稳定化 | — | HEC-1 | 接受 |
| 10.8 观察面 | H | HEC-3 | 后置 |
| 10.9 尾部判据 | — | Phase F 后 | 仅未来合同 |
| 10.11 Source-v3 归因 R1–R5 | I | Phase S 预注册 / HEC-1 三态机 | **已裁**(sol 23:xx;§8-11~14);D1 诊断**已批**(≤400 fits,任务书已出) |
| TSFM Consumer | — | Ridge 曲线成立后 | 用户定序 |

(10.10 号位空缺:原处置总表移至此;10.11 编号已被 §4.3/§8/§10.0 引用,保持不变。)

**TSFM 备注**(不入 HEC-1):TSFM 零样本无训练,数据准备作用于推理 context——只有 §5.2 的
serving-side 几何下才有定义;每次 probe 由"重训"变"推理",评估成本可能下降;Skill 的
`downstream_model_class` 轴已存在。排序不变:先 Ridge 曲线,后换 Consumer,一次只动一个变量。

## 12. 执行路线图(操作序;sol 2026-09-03 收缩版)

**总则**:一条主线、七步、每步一个产出一个门;每次只动一个面;预算分批;判词只在课末。

| 步 | 做什么 | 产出 | 门 / 预算 |
| --- | --- | --- | --- |
| 1 D1 | ~~路由伤害诊断~~ **已完成**(106 fits,0 LLM;`NO_OUTCOME_FREE_SEPARATOR` / `ROUTING_HARM_NOT_DOMINANT` / `NO_CLEAR_DIFFERENCE`) | `p4ab_routing_harm_diagnostic.{json,md}` | 结论已进 §8-17 / §10.11;Risk 面不开 |
| 2 D2–D4 | ~~D2 课程供给扫描~~ **已完成**(0 LLM / 0 fit;Phase S 13、Phase T 26(≤3816)、交集空;`p4ac_hec1_course_supply`);**D3** 骨架 `[D2]` 已填,待 sol 核 `[sol]`;**D4** 四件接线 + 七项 smoke(新增"单一权威门"验收) | `hec1_contract.{json,md}`、smoke 记录、接线 diff | sol 核冻结件(§8-7/8 子项、`[200:239]` 20/19、`[80:120]` held-in 5→7、`A3-frozen` 命名) |
| 3 Phase S | 多 Source cohort(`[160:200]`、`[200:239]`)建 K0;双环开、A5-online 单臂;H1–H3 每验证窗口机械落账 | K0(可能为空)、Phase S 报告 | 0 存活如实继续;LLM ≈15 单元 × 6 + 外环 ≤2/步 |
| 4 Phase T | KDD 天然 26–28 单元;Forward → Reverse → Interleaved;四臂(K0 空则按 §4.2 缩臂) | 逐单元 checkpoint、外环步记录、shadow 记录 | Forward ≤500 LLM;**仅按仪器完整性**放行后两顺序 |
| 5 课末读数 | 曲线(相对 Static)、Best-Safe-Global advantage、harm 事件、谱系、三分账、修订成功率 / 重遇收益 / 存活率、LLM/fits/覆盖、H1–H3 一致性、Slow vs ScopeFit 一致率、预注册检验 | HEC-1 收口报告 | 判词词表预注册;不成立按 §5.3 first-fault 归位 |
| 6 Phase F | 冻结三顺序全部末态臂 → 在 `p4u` held-out 五 origin 上 Fast-only 生成全部输出 → **一次性打开全部 Outcome** → 计分;context 覆盖只分层报告 | 终考工件 | 用户批密封开启;不筛选、不更换 origin |
| 7 HEC-2/3 | pooled vs per-channel(预注册"路由伤害在 per-channel 下消失")→ Risk 面(若 D1 支持)→ 跨数据(注入 electricity/traffic)→ Random-edit / ScopeFit 第五臂(如 shadow 不决)→ 观察面(模式持续性,新 cohort)→ **TSFM** | 消融与扩展 | 每次只动一个面 |

**每单元协议(Phase T,四臂同款)**:Context(20 条 support_a 的 22 维特征)→ 检索(Scope 匹配的 K 以
`requires_target_support` 供给)→ Fast ≤2 提案(程序 ≤2 步;Runtime 按缺陷存在初始化 serving_scope)→
Support-A 探针(p4u 单元预算:7 轮 / 24 probes / 6 LLM)→ bounded_risk 准入 → delayed(+48)→ Active 或三态
restricted → **评价面(+144,只计分、永不回流、不进 bank)** → 写回(online 臂)→ 每 5 单元外环(online 臂)。
内环即时 Slow 关闭。

**分支(预注册)**:Phase S 0 存活 → K0 空 → 臂集缩为 Static / A3-frozen(名待确认)/ A3-online +
Best-Safe-Global + ScopeFit shadow,判据 3 不评分。曲线不拉开 → §5.3 first-fault 映射,不读作"进化无效"。
D1 有信号 → HEC-2 冻 Risk 面;无信号 → 记 `NO_OUTCOME_FREE_SEPARATOR`,新进入者风险只能靠 held-in 探针
与证据有界 Scope 候选形式承担。

**论文对应**:图 1 曲线(四臂 × 三顺序阴影)、图 2 谱系对齐外环步、图 3 held-out 分层、表 = 归因 / 生命周期 /
H1–H3 / 成本;柱 Ⅰ/Ⅱ/Ⅲ(效率)现在即可成章。

### 12.1 Forward 之后(2026-09-03 夜定;sol 授权链见 `OPUS_HANDOFF_BRIEF` §2b)

| 序 | 做什么 | 谁 | 门 / 预算 | 0-LLM 并行件(主线) |
| --- | --- | --- | --- | --- |
| F+0 晨 | **仪器完整性核对**(`OPUS_HANDOFF_BRIEF` §5b 八项),只看仪器不看效果;账本入典 | 主线 | 任一项不过 → 停,定位仪器 first-fault | 起草 `audit_hec1_readout.py` **规格**(分析代码在看到 Reverse/Interleaved 数据前定稿) |
| F+1 | **Reverse**(≤500)→ 仪器核对 → **Interleaved**(≤500)→ 仪器核对 | Opus / 用户逐批放行 | 每顺序独立 store 与 run_label;可 `--resume` | Phase F 协议稿(冻结程序、密封开启程序、分层报告模板);HEC-2 预注册草案(P-C1/P-C2、跨数据适配、Random-edit、ScopeFit 第五臂条件) |
| F+2 | **课末读数**(0 LLM):Best-Safe-Global(冻结菜单、过预算取优)→ 曲线 / harm / advantage / 三分账 / 生命周期 / H1–H3 一致性 / Slow vs ScopeFit / 成本 → 预注册检验 → 判词三选一 | 主线执行 + sol 确认 | 判词只从 §8 词表出;不成立按 §5.3 first-fault 归位;**不因读数改任何合同** | 论文柱 Ⅰ/Ⅱ/Ⅲ 成章;柱 Ⅳ 按判词写 |
| F+3 | **Phase F**:冻结三顺序全部末态臂 → 用户批密封开启 → `[80:120]` × held-out 五 origin Fast-only 0-LLM 生成全部输出 → 一次打开全部 Outcome → 分层报告(context 覆盖只分层不筛) | Opus 执行 / 用户授权 / 主线裁定 | 开启前:所有臂输出已落盘且哈希不需要——路径 + run_label 足够(§7) | HEC-1 收口报告 |
| F+4 | **HEC-2**(每次一个面):① Consumer 轴 pooled vs per-channel(预注册 P-C1 尾部消失 / P-C2 基座上升);② 跨数据(electricity/traffic 注入 cell 的 serving-side 适配);③ Random-edit 对照;④ ScopeFit 第五臂(仅当 shadow 不决);⑤ Risk 面——**D1 已判无 outcome-free 分离量,默认不开**,除非 HEC-3 新特征改变结论 | 主线设计 / sol 裁 / 用户批 | 各自独立冻结件 | — |
| F+5 | **HEC-3** 观察面:模式持续性等新特征,新 cohort 前瞻冻结,检验 Scope 可分性提升 | 同上 | 不得在已看过的 origin 上拟合 | — |
| F+6 | **TSFM** 作为第三 Consumer(Ridge 曲线成立后) | 用户定序 | serving-side 几何已是前提 | — |

**分支(2026-09-03 sol v1.1 裁定后修订)**:Forward 仪器不过 → 修仪器后重跑该顺序(仪器故障重跑不算科学重掷;
**顺序进行中任何代码修改 → 该顺序降 shakedown**,不得 `--resume` 切码);Phase S 0 存活 → 缩臂(Static / A3-frozen /
A3-online),判据 3 不评分,其余照跑,**但 Phase F 不开**——HEC-1 以 A3-online vs A3-frozen 作 Target-local 自进化
组件证据收口;正典要求最终仍有自然数据上的 A5/A3/Static 同场,A3 结果不能替代完整 A5;课末判词 `NOT_SUPPORTED` →
按 first-fault 面(记忆 / 供给 / 课程)进 HEC-2 的单假设,不改 HEC-1 合同;`INCONCLUSIVE`(完成 <0.8·N_T_eff 或三顺序
未齐)→ 补齐缺失顺序,不重跑已完成顺序;**只完成两顺序不得改判据为 2/2,记 INCONCLUSIVE**。Phase F 前置 = 非空 K0 ∧
判词支持完整 A5 主张 ∧ 用户人工开封,三者合取。

**v1.1 事件记录**:首条 Forward(2026-09-03 11:25 发车)因发车后审查抓出接线缺陷 + `MIN_POSITIVE_UNITS_FOR_ADD=2`
与阶梯 v2 冲突 + 发车字节未提交,降为 `FORWARD_SHAKEDOWN`;Phase S-v1(K0 空)标 `superseded`;科学顺序在 v1.1
commit 下重跑。详见 `HEC1_V1_1_AMENDMENT_REQUEST_2026-09-03.md` §3b。

## 11. 工件与文件指针

(§9 预留给冻结后的「变更记录」;sol 批注中引用的编号 §10 保持不变;§12 置于 §11 前系追加顺序所致。)

- 合同与台账:`artifacts/main_protocol/p4u_main_experiment_contract.json`、`p4t_exposure_ledger.json`、
  `p4s_main_experiment_supply.json`、`p4v_main_baselines.json`。
- 瓶颈与可解性:`p4x_admission_regime.json`、`p4y_oracle_scope_bound.json`、`p4z_risk_refusal_routing.json`。
- Source 线:`p4w_source_line.json`(v1)、`p4w2_source_line_v2.json`(v2,含 2136 完整链)、
  `evaluation/main_protocol_p4/run_source_line_v2.py`、`main_experiment_contract_v2.py`、
  `scope_narrowing_preflight.py`。
- Stage 3:`artifacts/functional/e2/s3_pilot_probe_policy.{json,md}`、`s3_harness_v2_freeze.{json,md}`。
- 触发富度与下载:`artifacts/functional/e2/g1_kdd_trigger_census.{json,md}`、`docs/D4_DOWNLOAD_FREEZE_2026-08-29.md`。
- 正典与收口:项目 `AGENTS.md` §5.1/§5.2/§8.1、`docs/P4D_NATURAL_GAP_LINE_CLOSURE_2026-09-01.md`、
  `docs/MAIN_EXPERIMENT_DESIGN_SKELETON_2026-08-28.md`。
- 教师简报:`evaluation/functional/run_teacher_report_viz.py` 生成件(2026-08-31)。
