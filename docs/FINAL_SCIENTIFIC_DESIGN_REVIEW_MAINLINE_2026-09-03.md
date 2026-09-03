# 最终科学设计审查(主线,只读;2026-09-03 夜)

范围:不改代码 / 合同 / 工件,不启动实验,**未读 `FORWARD_SHAKEDOWN` 任何效用读数**(只引其仪器统计口径)。
sol 已裁八项视为固定。标记:**CODE FACT**(读自 `run_hec1.py` / `outer_loop.py` / `restricted_draft.py` /
`hec1_contract.py` 当前磁盘字节)/ **EVIDENCE**(已曝光工件)/ **INFERENCE** / **PROPOSAL**。

---

## 0. 一页执行结论

1. **方法与脚手架的边界是清楚的,而且对审稿人有利**:LLM 决定"提什么程序、在哪个语义方向收窄、要不要动手";Runtime 负责
   验证、准入、生命周期、普查、replay、激活。结果必须写成"**受治理的自进化 Harness**"这一*系统*的结果,不能写成 LLM 的结果;
   证明它不是普通规则系统的消融是 **validation-search(0-LLM 枚举 + 同门)** 与 **ScopeFit-only**,前者是必需 baseline,
   目前缺席。
2. **HEC-1 的正结果有结构性机会,但通道很窄**:26 单元 × k=5 只有 5 个外环步,其中只有前 3 步开出的 Draft 有真正的
   重遇机会;每张 ADD Draft 以根谓词(z≥3,解析 ~13/20 条)进入新单元验证时,面对的正是 p4x 里 0/9、Source-v3 里 0/3 的
   那道尾部门。**所有正曲线路径都经过"至少一张 Draft 在新单元过门"**;它是单点故障。
3. **两处必须在 Phase S-v1.1 前算清的仪器事实**:(a) `MIN_POSITIVE_UNITS_FOR_ADD` 磁盘仍为 2(CODE FACT,与 sol 裁定
   不符,v1.1 未落地);(b) replay 成本:每次筛选对全部已处理 cell 各 3 fits(CODE FACT `FITS_PER_SCORED_FACE=3`),
   第 k 步 = 15k fits;五步累计 225 fits/候选流,而每臂课程 fits 约 156 → **sol 的"100% 自身课程 fits"在最坏情形下仍不够
   覆盖五步**(INFERENCE,§4 有算术);解法是 raw 模型 fit 跨候选缓存(仪器优化,非方法改动)。
4. **统计只能是描述性**(已入合同);建议再加一条**实质效应下限**(用项目既有的 material 0.005,不发明新阈值)防止
   +0.0001 被叫作支持。
5. **Claim ladder 的关键分界**:只有 ADD→召回→探针位释放 = "**经验积累**"(L1),不得称 self-evolving;有一条修订后存活并
   重遇改善 = "**within-dataset self-evolving**"(L2);完整 A5 还差 Source→Target→held-out 整条链(L3)。合同 VERDICTS 已
   把 P1-only 单列为 `HEC1_P1_ONLY__RECALL_ACCUMULATION` 且不开 Phase F(CODE FACT)。
6. **K0 若再空,下一刀是 0-LLM 的穷举供给诊断**(validation-search 当 Phase S 的提案者):若穷举也 0 存活 → 空 K0 是
   数据 × 门的结构事实(Track B 证据);若穷举有存活 → 故障在 Fast 提案面,单假设去修供给。不再多跑 Source。

---

## 1. 未决问题 — 证据 — 推荐裁定

| # | 问题 | 证据 | 推荐 |
| --- | --- | --- | --- |
| Q1 | ADD 阈值 | CODE FACT `outer_loop.py:61 = 2`;sol 裁 1 | v1.1 必改;非作者复核项 |
| Q2 | replay 预算能否覆盖五步 | CODE FACT 3 fits/cell/筛选;INFERENCE 225 vs ~156 | raw fit 缓存 + 合同写最坏算术(§4) |
| Q3 | 实质效应下限 | 合同 STATISTICS 无下限(CODE FACT) | P1 加 `D_o ≥ 0.005·N_eff`(PROPOSAL,sol) |
| Q4 | Support 探针失败是否耗验证次数 | `record_verification` 只在验证面调用(CODE FACT);Support 侧未见计数 | 报告每张 Draft 的 supplied/probed/verified 三计数;不改规则 |
| Q5 | validation-search baseline | 缺席(CODE FACT:无该臂) | 0-LLM 事后臂,同 commit 同单元(§3) |
| Q6 | Phase F 聚合 | 合同要求三顺序;未定聚合 | 全部末态各报 + 均值为主;禁挑最好顺序(§7) |
| Q7 | 重遇归因 | `deployed_via` 五值(CODE FACT L1515) | 再按"同块 / 跨块"分层报告(§1.4) |
| Q8 | 冲突/负结果重复性 | H1–H3 n=3(Source-v3);路由伤害 n=5 | Phase S-v1.1 与 T 机械落账,≥2 轨迹一致才升 MECHANISM |

---

## 2. 合法 claim ladder(§1 答)

### 2.1 六个词的定义(互不替代)

| 术语 | 定义 | 最小证据 |
| --- | --- | --- |
| **Target-local adaptation** | 单元内 probe → 门 → 部署,不带走 | A3-frozen vs Static |
| **Skill accumulation** | 已验证可执行知识跨单元携带并被召回 | Active 形成 + 后续 `deployed_via ∈ {recalled_skill, searched_active_program}` + 该单元 online−frozen |
| **Scope evolution** | 适用范围因冲突证据被修订,修订后存活并在重遇中改善 | P2(合同 `P2_definition`) |
| **Harness evolution** | Harness 依反馈改自身状态,且后续决策因此改变并改善(含形成 + 修订/撤销至少一种) | P1 ∧(P2 或撤销后改善) |
| **cross-cohort transfer** | 在序列集 X 形成的 Skill 在同数据集序列集 Y 上有益 | 跨块召回事件 >0 且有益;Source→Target 为 P3 |
| **cross-domain transfer** | 不同数据生成过程 | 不在 HEC-1;需 F2 级 |

### 2.2 阶梯

| 级 | 观察到什么 | 可以说 | 不可以说 |
| --- | --- | --- | --- |
| L0 | A3-frozen ≥ Static | 反馈门控的数据就绪优于不处理 | 任何"进化""积累" |
| L1 | ADD → Active → 后续召回 → 探针位释放,online−frozen >0(P1) | **feedback-driven Skill-library evolution / Skill acquisition evolution**(sol 更正措辞:P1-only 不是"完全不是自进化");经验积累;成本下降 | Scope-revision evolution;完整 A5;跨域进化;"越改越好" |
| L2 | L1 + ≥1 修订后存活并重遇改善(P2) | **within-dataset Skill-and-Scope evolution**(sol 措辞) | 跨数据集;跨域;通用知识 |
| L3 | K0 非空 ∧ A5−A3 >0 ∧ held-out 保持 | Source→Target **cross-cohort accumulation**(同数据集) | cross-domain |
| L4 | 新族 Outcome 未见 | cross-domain | — |

- **A3-online > A3-frozen 且 K0 空**:L1 或 L2,取决于是否有修订事件;它是 Target-local evolution/accumulation
  (within-course),不是 Source→Target 的 memory accumulation。注意 Phase T 课程本身跨 4 个块,online 臂在块 A 形成、
  块 B 召回的事件是**within-dataset cross-cohort** 复用——应按"同块 / 跨块"分层报告(PROPOSAL)。
- **"self-evolving Harness" 的最小链**:反馈 → 机械记录的状态改变(ADD 或修订)→ 后续决策因它改变(`deployed_via` /
  解析集改变)→ 相对无写回对照改善(P1)→ **至少一次修订事件存活**(P2)。缺 P2 只能叫 "experience-accumulating"。
- **完整 A5 还需**:Phase S 非空 K0(`audit_hec1_k0_freeze` 过)→ Phase T A5-online − A3-online >0(P3)且 K0 卡在
  Target 有 Scope 匹配(按层报)→ Phase F A5* 在 held-out 保持。缺任一环只能 L2。

---

## 3. HEC-1 的结构性机会(§2 答)

### 3.1 事件计数上界(CODE FACT + INFERENCE)

- 外环步:26 // 5 = **5**(单元 5/10/15/20/25);Phase S 13 → 2(CODE FACT `budget_arithmetic`)。
- ADD 候选 ≤ 出 winner 的单元数(去重后);Phase S-v1 基率 3/13 ≈ 23%(EVIDENCE)→ Phase T 期望 ≈ 6。
- 一张 Draft 的时间线:步 1(单元 5)开出 → 单元 6 起供给 → 过门最早单元 6–7 → 召回机会 ≈ 19 单元;步 3(单元 15)开出 →
  召回 ≈ 9;步 4 → ≤ 5;**步 5 开出的 Draft 无重遇机会**。有效 ADD 窗口 = 前 3 步。
- 验证上限 3、修订上限 2、WAITING 耗次(CODE FACT `MAX_VERIFICATION_ATTEMPTS=3`、合同 `consumes_verification_attempt: True`):
  一张 Draft 最多 3 个验证面;若前两次都是 REVISABLE 并各收窄一次,第三次是最后机会。**Draft 不会在有机会前被关**,
  但会在 3 次内被关;D2 显示无稀疏单元(EVIDENCE),根谓词 Draft 的 WAITING 触发概率低,收窄后带宽 Scope 才会 WAITING。

### 3.2 隐藏的结构性零通路

- **单点故障 = 新单元的 bounded_risk 门**。ADD Draft 以根谓词进入(z≥3 解析中位 13.5/20,EVIDENCE D2),等价于 p4x 的
  live 探针情形:9 个实质正探针 0 准入(EVIDENCE);Source-v3 收窄后 delayed 3/4 过、独立重遇 0/3(EVIDENCE)。
  **HEC-1 的"新单元验证"几何 = Source-v3 的独立重遇几何**。因此:根谓词 Draft 大概率首验即 REVISABLE/FLAGGED;
  存活依赖收窄通道;收窄依赖 Slow(≤2 次/步)+ 工具 + replay 预算(§4)。
- **负通路**(必须预注册):online 臂的 resupplied Draft 占候选位可能**饿死**自提(S2a r2 机制,EVIDENCE),使 online < frozen;
  三分账 (c) 须能记负值。
- **不能给正的下界**:已曝光基率为 0/9(根谓词)与 0/3(收窄后新窗口);n 小,单侧 95% 上界分别 ≈ 0.28、≈ 0.63。
  上界:激活数 ≤ 6 × 0.63 ≈ 4;下界 = 0。**不用评价面即无法收窄这个区间**,不偷看。

### 3.3 正曲线来源排序(概率 × 证据)

1. **ADD → Active → 召回 / 探针位释放**(需 ≥1 激活):唯一能产生*质量*差的通道;证据 NOAA −44%(FRESH)、S2a 饥饿机制。
2. **探针位释放本身**:与 1 绑定。
3. **Scope 修订存活(P2)**:新窗口基率 0/3;低。
4. **撤销**:需先有 Active;HEC-1 内几乎不发生。
5. **负向**:resupplied Draft 饥饿自提。

### 3.4 Track A 可信度的最小事件计数(PROPOSAL)

| 事件 | 最小 | 备注 |
| --- | --- | --- |
| 正例铸卡(Draft 开出) | ≥3 | 跨 ≥2 个外环步 |
| 独立激活(新单元过权威门) | ≥2 | 不同程序或不同 cohort |
| 真实重遇(后续单元 `deployed_via` 为召回/起始 Active) | ≥3 单元,跨 ≥2 cohort | 同块/跨块分层 |
| 重遇改善 | 中位 (online−frozen) >0 且 ≥2/3 重遇单元为正 | — |
| harm | online ≤ frozen 每顺序 | — |
| P2(若主张 L2) | ≥1 修订后激活 + ≥1 重遇改善 | 合同定义 |

激活 <2 → 最高只能写"单事件机制观察"。

---

## 4. 方法 vs 脚手架(§3 答)

| 逻辑 | 归属 | 论文如何写 |
| --- | --- | --- |
| candidate census | **方法**(Memory→Skill 的确定性整合,AGENTS §2.2) | 固定治理规则;参数(MIN_POSITIVE)作方法超参报告 |
| minimum-positive 供给 | 方法(阶梯 v2 证据计价) | 同上 |
| Scope 阈值工具 | 方法组件(Slow 使用的工具) | 必配 ScopeFit-only 消融量化 LLM 边际贡献 |
| 三态机 | 方法(生命周期治理) | 人写的固定规则,明示非学习所得 |
| 权威风险门 | 方法(部署问题的固定约束) | 作为问题设定的一部分 |
| replay 选择 | 方法(编辑的低成本评估器) | 主流 harness 进化的评估器角色 |
| Fast/Slow | 方法(Agent 架构) | — |
| 激活/撤销/召回 | 方法(生命周期) | — |
| 臂、顺序、面、预算、comparator、readout | **脚手架** | 实验协议 |

**对尖锐质疑的回答**:LLM 真正决定的是 (i) 每单元提什么程序(含组合与参数)、(ii) 收窄的语义方向(哪个特征、哪个方向、
为什么)、(iii) 是否弃权。Runtime 负责其余一切,**这是设计**(AGENTS §4:LLM 只提案不批准)。因此主张对象是"受治理
的系统",消融要回答两问:**没有 LLM 行不行**(validation-search:0-LLM 枚举菜单、同 Support 门、同预算 → 若打平 Fast,
LLM 提案无增量)与**没有生命周期行不行**(A3-frozen:同 LLM、无写回)。ScopeFit-only 回答 Slow 的语义选择是否优于机械
选择;Random-edit 回答是否优于随机。

| Baseline | 必需性 | 阶段 | 成本 |
| --- | --- | --- | --- |
| A3-frozen(Frozen Harness) | 必需 | HEC-1(有) | 计入 |
| validation-search / Parallel@B | **必需**(0 LLM) | HEC-1 事后,同 commit 同单元 | fits only |
| ScopeFit-only | 必需(shadow) | HEC-1(有) | shadow_fits |
| Best-Safe-Global(offline comparator) | 必需,诚实命名 | HEC-1(有) | ≤1820 fits |
| direct-LLM(无门) | 强烈建议 | HEC-2 | 1 call/单元 |
| Random-edit | 建议 | HEC-2 | 与 Slow 同价 |

---

## 5. 公平性与预算(§4 答)

- **compute-matched frozen**:不需要单独造。frozen 臂**按构造**无法把额外算力换成携带知识;论文限定语:"online 臂每顺序多
  ≤10 次外环 LLM 与 replay fits;报告总算力与到首个安全 Skill 的 fits,不主张算力对等"。validation-search 提供 0-LLM 的
  算力参照。
- **cache / 随机性 / 三顺序**:cache 使两臂分叉只源于记忆(CODE FACT 简报要求);三顺序共享数据与 cache,非独立 seed
  (合同 CODE FACT `orderings_are_not_seeds`)→ 作 development mechanism evidence 足够;确认性需 ≥3 seed × ≥8 cohort。
- **replay 最坏 fits 核算**(CODE FACT `FITS_PER_SCORED_FACE=3`;INFERENCE 其余):第 k 步每候选 3×5k;五步 15+30+45+60+75 =
  **225/候选流**;每 online 臂课程 fits ≈ 26×6 = **156**(由 0.25×312=78 反推,CODE FACT 注释)。
  → 100% 自身 fits 下:1 候选/步可到第 4 步(150),第 5 步不足;2 候选/步第 3 步耗尽。**sol 的 100% 规则在最坏情形不覆盖
  五步**。PROPOSAL(仪器,非方法):raw 模型 fit 按 (cell, face) 缓存并跨候选/跨步复用 → 每候选每 cell 1 fit → 五步 75
  /候选流,2 候选流 150 ≤ 156。发车前在合同写出最坏算术表(0 fit 可算)。
- **两顺序 → INCONCLUSIVE**:确认(CODE FACT `VERDICTS.HEC1_INCONCLUSIVE`);不得改 2/2。

---

## 6. 统计与实际意义(§5 答)

- **实质效应下限**(PROPOSAL,sol):P1 增加 `D_o ≥ material × N_eff`(0.005 × ≈24 ≈ 0.12)于 ≥2/3 顺序,`d_c ≥ 0.005`;
  用既有 material 线,不发明阈值。
- **主指标** = 终点累计差 D_o(合同已定);AUC 与中点差只作辅助。
- **+144 不可评单元**:可评性是单元固有属性(只读 mask),预扫后对所有臂、所有顺序**自动一致**;记 `N_eff`;学习照常。
- **可视化**:三条细曲线(每臂每顺序)不画带;cohort 终点 d_c 点图(4 点 × 3 顺序);逐单元配对差符号条;harm 事件 rug。
- **确认性**:≥8 独立 cohort(或数据集)× ≥3 seed(cache 按 seed 分隔),单侧 Wilcoxon 或 ≥7/8 符号检验,预注册效应下限。

---

## 7. K0 与完整 A5 的补救(§6 答)

- K0 再空时 Phase T 能答:L1/L2(within-dataset、含跨块召回);不能答 L3。
- **非空 K0 的最短、最少事后调参路线**(PROPOSAL):Phase S-v1.1 若空 → **0-LLM 穷举供给诊断**:在同一 Source 块、同门、
  同预算下以 validation-search 替代 Fast 作提案者。穷举也 0 Active → "该数据 × bounded_risk 下无可存活 Skill"(Track B
  结构证据);穷举有 Active → 故障在 Fast 提案面 → 单假设修供给(观察/提示,前瞻冻结)→ Phase S-v1.2。**不再多跑 Source,
  不换数据凑 K0。**
- 跨域 Source:角色 = development Source(Outcome 可读)、与 Target 数据生成过程不同、曝光台账冻结;预期 Scope 匹配低,
  按层报;不是最短路。
- 顺序:先完成 Target-local HEC-1(合同已冻),再穷举诊断,再决定 A5 是否可在 KDD 上存在。

---

## 8. Phase F 资格(§7 答)

清单(CODE FACT 合同 `assert_launchable('phase_f')` 需 SUPPORTED ∧ `seal_released=True`;其余合同/裁定):非空 K0(审计过)
∧ 三顺序齐 ∧ `HEC1_EVOLUTION_SUPPORTED`(P1-only 不合格)∧ 末态冻结(commit id + 快照路径,不建哈希)∧ 覆盖只分层
∧ 0-LLM 机械部署 ∧ 用户开封。
**聚合**:评价**全部**三顺序末态 × 全部臂;主读数 = 三顺序均值的 (A5*−A3*)、(A5*−Static*);各顺序单列;符号不一致 →
`MIXED`。**禁**按 development 挑顺序、按覆盖换 origin。

---

## 9. 最小论文实验矩阵(§8 答)

| 级 | 任务 / 数据 | Consumer | Baseline | 实验 |
| --- | --- | --- | --- | --- |
| **必做** | Forecast / KDD with-missing | pooled Ridge | Static、Best-Safe-Global(offline comparator)、A3-frozen、**validation-search** | E0(§5.1 已有)、E1(flip control 已有)、E3(Phase S 供给记录)、**E4 = HEC-1**、E5 仅当 Phase F 合格 |
| **强烈建议** | Forecast 同数据;AD(Yahoo 已曝光 24,安全/条件化);Classification(已有 dev 生命周期证据作附录) | **per-channel Ridge**(HEC-2 ①) | direct-LLM、Random-edit | E2 作 Track B 语境;D1 路由伤害 |
| **可选** | 跨域 Solar(F2 安全)、注入 electricity/traffic | TSFM ×2 | FFORMS-style、AegisTS-style、ScopeFit 全臂 | — |

最小可投稿组合 = 必做行全部 + per-channel 一刀 + AD/Classification 既有证据作条件化与安全支撑。

---

## 10. TSFM 前置(§9 答)

- Ridge 上至少 **L1 成立且 harm ≤**,并已完成 per-channel 一刀(知道 Consumer 结构如何改变伤害形状)。
- TSFM = **Consumer 泛化实验**,不是新主任务:同 KDD 单元、同顺序、同 commit,只换 Consumer。
- 零样本下 Program 只作用 serving context:context 算子(填补/离群修复/组合 ≤2)可复用;训练侧语义无意义。Specific Skill 以
  Task × Consumer 为键,**不直接复用**,降为 supply-only 候选重验;h0 General 直接复用。
- 归因:一次只换 Consumer;数据、顺序、程序空间、Scope 类不变。
- 回应"只对 Ridge 有效":≥2 个不同家族的 TSFM + per-channel Ridge,同一数据;≥2 数据集只在做跨域主张时才需要。

---

## 11. Track A / B 冻结标准(§10 答)

- **Track A 最小**:P1(含效应下限)+ §3.4 计数表 + harm ≤;若加 P2 → L2 标题"self-evolving (within-dataset)";只有
  ADD/召回 → 标题收窄为"**Governed experience accumulation for data readiness: verified Skill reuse under downstream
  feedback**",贡献 = 条件化 + 门控生命周期 + 积累曲线;"self-evolving" 只进 future work。
- **Track B 必须具备**:跨窗口 Scope 不稳定(H1–H3 ≥2 轨迹一致)、跨 Consumer 伤害形状(D1 + HEC-2 ①)、oracle 上界
  (p4y 可行性、UID 级 +0.61)证明 headroom 存在、§6 first-fault 机械定位、路由伤害机制。
- **已有 n=3 只能叫机制观察**:新进入者伤害、持续成员翻号、覆盖塌陷(Source-v3);路由伤害与改动量脱钩(D1,n=5)。
  需重复:Phase S-v1.1 与 Phase T 的 H1–H3 落账。
- **null 后唯一下一刀**:按 first-fault——供给面 → 穷举供给诊断(0 LLM);记忆面 → Scope 宽度分析(0 LLM,工件);课程面
  → 不适用(触发密度已足)。**禁**同时改门、阈值、观察特征、Consumer。

---

## 12. General 与 Specific(§11 答)

- HEC-1 真正进化的只有 **Specific Skill**(program × scope × evidence);General(h0 bootstrap / 指导文本)冻结(CODE FACT
  合同编辑面只含 Skill 库条目)。
- General 可修改面 = EditController 面③(指导文本)与 bootstrap 条目;HEC-1 关闭,HEC-2/3 才开。
- Specific → Source-derived = **角色变化**(同一内容供给新 cohort),不是抽象;→ Shared/General Capability 需 ≥2 域相似
  可观察 Context 的重复正向 + 风险证据(AGENTS §4)。
- 论文纪律:每张 Skill 标 Target-local / Source-derived 与形成 cohort;所有 within-dataset 限定语不省;唯一"通用"内容是
  手写 h0。

---

## 13. 三周硬路线(§12 答)

| 步 | 输入 → 产物 | 门 | 失败分支 | 执行 | 非作者审 | sol | 用户 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | shakedown 仪器报告 | 八项 | 仅仪器修复 | Opus | — | — | — |
| 2 | v1.1 落地 15 项 + raw fit 缓存 + 最坏算术表 + 效应下限条款 | 测试 + 复核 | 回修 | Opus | **grok(B/C/H + 15 项)** | 效应下限 | — |
| 3 | commit;合同记 commit | `assert_frozen` | — | Opus | — | — | commit 时点 |
| 4 | +144 预扫 → `N_eff` | 0 fit | — | Opus | — | — | — |
| 5 | Phase S-v1.1 → K0 审计 | ≤120;审计脚本 | K0 空 → 缩臂继续;记 superseded | Opus | — | K0 非空时确认 | 已授权 |
| 6 | Forward → Reverse → Interleaved | 八项自动 | disagreement → 该顺序降级 | Opus/grok 监控 | — | — | 已授权 |
| 7 | 课末读数(spec 主线、实现 grok)+ validation-search 0-LLM 臂 | 预注册词表 | first-fault 归位 | grok | 主线 | 判词确认 | validation-search 授权 |
| 8 | 若合格:末态冻结 → Phase F | 三合取 | 不合格 → 不开 | Opus | 主线 | — | **开封** |
| 9 | 若 K0 空:穷举供给诊断(0 LLM) | — | 结构性 vs 供给面 | grok | 主线 | 裁下一刀 | 授权 |
| 10 | 论文柱 Ⅰ–Ⅲ 成章;柱 Ⅳ 按分支 | — | — | 主线 | sol | — | — |

**no-go 清单**:不新建 Gate / SHA / manifest / 平台;不重构 runner;不改任何阈值;不加观察特征;不做 BSG prequential;
不接 TSFM;不换 Source 数据凑 K0;不挑顺序;不按效果重跑;不在顺序进行中改码。

**三分支唯一下一步**:**成立(P1∧P2)** → K0 非空则 Phase F,K0 空则以 L2 收口并做穷举供给诊断决定 A5 是否可存在;
**部分(P1-only)** → 标题收窄为积累,跑 validation-search 臂确认 LLM 增量,再穷举供给诊断;**失败** → 按 first-fault
单刀(供给面穷举 / 记忆面 Scope 宽度),不改合同、不加面。

---

## 15. replay 严格最坏情况核算(2026-09-03 追加;只读;sol 已据此裁定三件不可拆分修复)

**代码事实**:一次 screen 对每个已处理 cell 调 `_policy_reading` = Static `scoped_evaluate`(1 fit)+ 带程序
`scoped_evaluate`(raw + program = 2 fits)= 3 fits/cell(`run_hec1.py:290-313`,`estimated_fits_per_candidate = 3 ×
len(contexts)` `:545`);U_k = 5k;每臂允许额 = share × 3 × 2 × 26 = 156(`:1954`);`outer_llm_per_step = 2`、
`retries_per_candidate = 2`;**program model 与 Scope 无关**(`scoped_serving_evaluator.py:208-212`,Scope 只掩码预测);
ADD 去重只查 `held` = bank 中带 `source_skill_id` 的行(`run_hec1.py:1319-1322`),`open_restricted` 无条件铸壳
(`restricted_draft.py:354-387`);`MIN_POSITIVE_UNITS_FOR_ADD` 现为 1(`outer_loop.py:72`,主线此前读旧字节,更正)。

**Q1 每步候选上限**:ADD = bank 中 POSITIVE 过的不同(程序 × 根 Scope)组数,形式上无上限,只受 replay 守卫;NARROW ≤
被探过的 Active 程序数(adverse ≥2);REVISE ≤ open REVISABLE;需子句者每步实际被筛 ≤ 2(LLM 帽)。每步最坏 fits =
(#ADD + 2) × 3·U_k。

**Q2 同 key 多壳最小反例**:步 1 P 在单元 2 正 → 开 D1;单元 7 D1 过 Support 败 delayed → REVISABLE、非 Active → P ∉ held;
步 2 普查再见 P 正 → 再 ADD → 再 screen(30 fits)→ 开 D2(同 P 同根 Scope,计数器归零)……最多 5 壳;D1(已收窄)与
D2(根谓词)同时供给。变体:刚 Active 未再探的程序亦被重复 ADD。

**Q3 五步 fits 公式**(a_k ADD 筛数、c_k 子句筛数):无缓存 F = Σ 3·U_k·(a_k+c_k)(单流 225;各 1 → 450);raw-only
F = Σ U_k·(a_k+c_k) + R,R ≤ 25(复用内环 Support 面 raw 读数则 0);raw+program(按程序签名 × cell)REVISE/NARROW = 0,
ADD(P) ≤ U_k − p_k,各 1 → ≤ 70。

**Q4 156 内保证五步各 ≥1 screen**:预留规则(第 k 步只在 remaining − cost ≥ Σ_{j>k} m_j 时花)。无缓存 Σm = 225 > 156
→ **任何分配规则都不可能**;raw-only Σm = 75 → 可保证;raw+program 更宽裕。

**Q5 最小修复**:(i) 按完整 census key 对全 lineage 去重 = 正确性修复;(ii) 预测缓存 = 纯仪器(需逐位等价测试);
(iii) 未来步预留 = 记账规则(只改何时筛);(iv) 每步上限/优先序 = 方法变更,**不加**。sol 裁:(i)+(ii)+(iii) 为一个不可
拆分修复,附七组测试;详见 `OPUS_HANDOFF_BRIEF` §2e。

## 14. 需要用户最终决定(≤5)

1. **效应下限条款**入 v1.1(pre-data;sol 同意后)。
2. **raw fit 缓存 + 最坏算术表**作为发车前仪器项(0 LLM,小改动;否则接受"100% 亦可能第 4–5 步耗尽"并如实披露)。
3. **validation-search 0-LLM 臂**纳入 HEC-1 事后读数(fits only,同 commit 同单元)。
4. K0 空时**穷举供给诊断**授权(0 LLM)。
5. Phase F 后首刀 = **HEC-2 ① per-channel**(D1 已给两条可证伪预测)。
