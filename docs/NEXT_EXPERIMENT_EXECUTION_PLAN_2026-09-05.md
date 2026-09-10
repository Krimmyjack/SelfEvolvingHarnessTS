# 下一阶段可执行推进计划:机制测量 → 最小自然闭环 → 跨数据集主实验(主线,2026-09-05)

以下开场为 **2026-09-05 历史状态**：当时是计划文档、只读核实产物，未启动实验、未下载模型、未改代码/合同/阈值、未开密封数据。
当时已批事项只引用现有授权、新增预算与方法变化在 §8 单列待批；**最新授权与任务见下方“当前执行入口”**。协议 M r2(`M_MECHANISM_MEASUREMENT_PROTOCOL_2026-09-05.md`)
**仍未冻结**;本计划把它拆成"核心闭环必需"与"补充解释"两组执行,并按 astra 评审修正四处口径(附录 C)。
目标不变:Agent 驱动的 Data Readiness Harness 自进化,终点为自然数据上的完整 A5 / A3 / Static 验收。

---

## 当前执行入口（2026-09-08；替代下面历史章节的下一步顺序）

用户已同意先推进 **DEV-DEPLOY-1：部署口径对齐**；唯一执行任务书为
[DEV-DEPLOY-1](DEV_DEPLOY1_FAST_ONLY_ALIGNMENT_TASK_2026-09-08.md)。下方 §1–8 及附录
保留为历史讨论/证据，不得再把旧“先修一张卡/先扩 Scope”当当前发车指令。

1. 第一包：历史 Fast 原选择 vs Support 交付的影子审计；开发数据上冻结 Fast-only
   真运行；同样不看当格反馈的真 0-LLM 固定菜单基线。菜单空间与离散/暴露分析只作
   辅助诊断，不等待一套完整环境筛选平台。第一包不让 Slow 写卡。
2. 第二包方向：在第一包定稳的执行条件下，比旧知识与 Slow 自主修订知识对无当格
   Support 的冻结 Fast 是否有帮助。数据、候选、主终点与预算待包末冻结，不自动发车。
3. 最终仍回到自然数据上的 Static/A3/A5 和跨情境积累；不以局部程序改善或执行接线
   代替完整系统，也不因当前 KDD 诊断而默认换 Consumer/数据集。

**2026-09-15 路线检查点**：有可重复价值则固定方法并做独立验证；有空间但现方法
无增量则换明确机制；菜单/环境缺乏空间则明确更改一项实验条件；若仅有故障则承认
未完成并重评投入，不宣称方法失败或自动续出第九个小修包。
当前状态见状态页置顶；授权、执行回执见 `docs/DECISIONS.md`，建议与批准分开。

## 0. 更正插入:决策单位改为逐序列(2026-09-07,追加式,不改本计划的历史数字)

权威记载见项目 `AGENTS.md` §5.3。要点三条,只影响**本计划各条读数所描述的单位**,
不撤销任何已测数字:

1. 本计划(以及 HEC-1 / M-R0 / M-W / Scope 线与 Workflow 线的全部读数)是在
   **cohort 广播式单程序**形状上测得的:每单元一个 `PreparationRequest`,由
   `cell.observation_block`(= 该 block 第一条 eval 序列)构造,一个 compiled
   程序作用于 20 条被服务序列。正确的决策单位是**单条序列及其当前合法窗口**。
2. 因此 §2 的"编辑面"讨论、§7.0 的原路径可达性检查、附录 D 的 M-W 组合账,
   其主语都应读作"cohort 级程序与 cohort 级 Scope",**不能**直接当作逐序列
   方法的可达性或可修复性结论。逐序列读数与组级读数分母不同,不并表、不相减。
3. R4A(2026-09-07 深夜)已裁定停 series 级 **Scope 修订**类实验(M-W / ACC-1
   二层 / 有界选择规则),理由是逐序列增益不是序列的稳定属性(块内 ICC≈0)。
   这与本更正不冲突:决策**发生**在序列上,不等于逐序列增益**可由部署前特征
   预测**;前者是本节要求的形状,后者是 R4A 关闭的路线。

首个按更正后形状执行的开发级验证:DEV-SEQ-1
(`docs/DEV_SEQ1_PER_SEQUENCE_2026-09-07.md`,源
`evaluation/main_protocol_p4/{per_sequence,dev_seq1_knowledge,run_dev_seq1,
smoke_dev_seq1,audit_dev_seq1}.py`)。

---

## 1. 当前状态(一页)

| 类别 | 内容 |
| --- | --- |
| **已完成** | 统一 Harness 运行基础;HEC-1 三顺序长课程(26 单元 × 4 臂,同 commit `d690850`,仪器 9 项全过)已**收口**,冻结判词 `HEC1_EVOLUTION_NOT_SUPPORTED`;0-LLM first-fault 诊断(BSG / validation-search / transfer audit,2013 fits);D1 / D5 / D6 机制诊断;两份独立文献地图;协议 M r2 草案 |
| **已验证(可复算)** | 自然数据上程序有收益(KDD:best-fixed +0.2629;origin 2856 `period_median_complete → outlier_*` 双面 +0.29~+0.65);Harness 能形成并使用知识(Phase S K0 非空:`outlier_mad` @ `z_peak>=3`,delayed +0.360 / hf 0.10 / msh 0.13);跨窗安全不可迁移(Support-safe → +144-safe 10/34);pooled 下伤害主要经模型路由(D5 严重伤害 8/10 路由分量);AD 上同族清洗程序 12/12 程序级宏效用为负(`INVERTED_EFFECT_OBSERVED`) |
| **已测出的否定(HEC-1 冻结数字与判词不变)** | online / frozen 行为分歧稀疏(62/69 平局;7 分歧中 5 预算介导);外环每顺序 10 个臂步、三顺序共 30 步,其中 27 步普查无候选;仅 3 个候选(全为 `REVISE`,来自 REVISABLE Draft),`ADD` 0、`NARROW` 0、外环开 Draft 0、修订 0;11 张 Draft 由 `run_hec1.py:2121`("过 Support 准入 → 部署 → 败 delayed")产生,**其中 8/11 由该 cell 起点仍持有部署权的卡产生(6 张直接来自 K0)**,3/11 来自非 Active 候选;LLM 计费少记 422(物理 1088 上界 / 853 下界 vs 账面 666) |
| **HEC-1 的实现限制(M-R0 审计,2026-09-05;追加解释,不改判词)** | ① **首因 `CENSUS_RELATION_VOCABULARY_MISMATCH`**:`AdmissionVerdict` 无 `relation` 字段 → `run_hec1.py:1849` 恒 `None`、`:1850` 落到 reason 串 → `outer_loop.py:209-211` 原样返回 → `:363-365` 只数 `POSITIVE/CONFLICT/NEGATIVE` ⇒ 30/30 步 `positive_units ≡ 0`、`adverse_units ≡ 0`,**`ADD` 与 `NARROW` 在同一处被接线阻断**;反事实(让 `_relation` 走自身增益定义)= 26 个 `NARROW`(22/30 步)+ 6 个 `ADD`——**只作旧语义下的历史参照**,新代码用真实 relation 且改了已知/活动谱系的去重,重放不必凑回 26/6,差异须解释。② delayed 面从不入 bank(`:1836` 只收 `kind=="probe"`)⇒ **7/11** 条 Active 谱系在 delayed 面已满足 ≥2 次风险线 adverse(M-R0 §1.5 表逐行计数;报告正文的"8/11"与本计划此前的转写系笔误,**勘误**),普查全部不可见——**这是协议事项,不是接线缺陷**(§4.3)。③ `FLAGGED` 禁加子句:10/11 张 Draft。④ 3/3 REVISE:5 次到达阈值工具的提议 `CALIBRATED` 0 次(其中 reverse k5 两次全词表 `NO_FEASIBLE_STUMP`),第 3 次重试被 `outer_llm_per_step=2` 在循环头挡掉、按构造不可达。⑤ **部署权从未丢失**(9 条臂×顺序快照计数零下降、0 撤销建议、`REVOKE` 分支不可达);冻结臂按单元重建 ledger 丢弃 15 张 Draft 系设计行为,不列为缺陷。**含义**:HEC-1 没有测到外环 `ADD/NARROW` 路径;但修好词表**不是充分条件**(U-5:阈值工具 0/5 校准) |
| **仅有候选设计(未实施、未验证)** | 协议 M r2 的测量包与决策表;唯一结构编辑面 = Workflow 组合(优先候选,**未证明是正确修法**);A / B / C 三臂闭环;格级适用性 / 三足迹粒度;方法 v2 已降为 Architecture A 基线 |
| **尚未得到支持** | **自然持续修订链**(冲突 → Agent 修订 → 独立验证 → 后续改善)= 0 条;跨情境重复 = 无;A5 − A3 在新数据集同预算下的优势 = 未测。HEC-1 的准确表述(astra 修正):**知识已进入系统并参与使用(K0 非空,三顺序均记录),但结构修订没有完成,持续修订收益尚未得到支持**——"没有知识""没有修订""没有可靠增益"是三件事,不得混写 |

能力进度只按最后一行衡量;文档、测试与工程完成度不计入。

---

## 2. 最小研究问题与方法边界

**下一阶段唯一优先问题**:旧 Skill 遇到自然冲突后,Agent 能否在**一个**结构编辑面上提出修改,经独立验证后改善后续表现?

**编辑面(先选一个;M-R0 后的准确表述)**:HEC-1 的原有修订路径(`NARROW` / `REVISE`,Scope 子句)**从未被真正测到**——首因是普查词表接线(§1),不是路径本身;因此**在决定开新编辑面之前,先修最小接线、复核、再检查原有路径解锁后的可达性**(§7.0)。Workflow 组合仍是有既往证据(P4d 组合/顺序效应)的**优先候选**,理由是 `FLAGGED`(10/11 张 Draft)在现行代码中没有对应的结构修订入口(`outer_loop.py:474-490`),且 Scope 子句路径在 5 次真实尝试中 `CALIBRATED` 为 0(U-5)。**不预设它一定能修好 FLAGGED**——`FLAGGED` 也可能来自效应漂移、观测不足或读数波动;是否开 M-W,由原路径可达性检查的结果决定(§7.0 第 ③ 步)。

**允许的动作空间(冻结的 P1 Common DSL,`hec1_contract.PROGRAM_SPACE`)**:

- 算子:注册表中 forecast 可用的 s1 算子 20 个——impute 7(`impute_linear/fft/ema/ssm/ar`、`period_complete`、`period_median_complete`)、denoise 5、outlier 4(`winsorize/outlier_iqr/outlier_mad/hampel_filter`,destructive)、structural 1(`repair_level_shift`,destructive)、decompose 3;align / shape 类不进组合。
- 组合长度 ≤ 2;窗口验证器 `MAX_MODIFIED_FRACTION = 0.35` 全有或全无。
- 参数:只有 4 个算子有公开参数且 schema 冻结(`impute_linear.strength`、`period_median_complete.{period,cycles,min_donors}`、`denoise_median.{window,strength}`、`hampel_filter.{window,n_sigmas,global_z_min}`);其余参数为空对象。修订不得引入新算子、新参数键或新特征。
- **合法修改 = 恰好一步**:同类替换一个算子 / 增删一个 impute 前置步 / 交换顺序 / 在 schema 内改一个参数值。按逐序列增益向量去重后计数(P4d:396 程序 → 7 种不同效果)。

**(更正,2026-09-06,6 pro 指出)原文"减法修订不可能创造新收益"在数学上不成立,撤回。** 固定 Program、嵌套 Scope、退出者回 raw 的几何下,
\(U(P,S') - U(P,S) = -\frac{1}{N}\sum_{i\in S\setminus S'} g_i(P)\):被排除集合净负,收窄就提高总体效用;这次收窄未做到,不等于收窄做不到。
同样,"受损数下降"在此几何下是构造性的,不证明选得聪明。仍成立的是:SA-1 与 HEC-1 两种设置下"修订 = 冻结"的同形结果说明**迄今的修订
从未按全人群目标选择过**,不能据此判定哪个编辑面更强。**正确的问题是:现有反馈能否指导系统找到对当时可执行策略真正有价值的修改**(§4.5),
以及"怎样修改"的增量 \(d_e(P\to P') = L_e(P) - L_e(P')\) 是否比程序效应 \(g_e(P)\) 更稳定、更可学习(第一包读数)。C 臂的确定性提议者是
SA-1 机械修订机制的直接后代,B 臂才是新增的东西。

**保持固定**:四线风险(0.005 / 0.20 / 0.30 / MIN_TREATED 5);P4 `_gate` 为唯一执行权威;Scope 初始化冻结表与 12 特征词表;Consumer 由 §3 核心测量决定后固定;语义预算按 sol 裁定;生命周期 v1.1(Draft / 独立验证 / Active / 撤销 / 修订 → 重遇)。

**组件必要性(闭环必需 / 后置)**:

| 组件 | 当前闭环是否必需 | 理由 |
| --- | --- | --- |
| 唯一编辑面上的结构修订 + 谱系 | **必需** | 这是要测的对象 |
| 祖先影子对照(同面同序列) | **必需** | 没有它无法判"后续改善优于旧版" |
| 语义预算(≥1 完整决策) | **必需** | HEC-1 5-call ≈ 1 提议;外环 2/步致 `OUTER_LLM_BUDGET_SPENT` |
| 置信序列(anytime-valid) | 后置 | 只治多重窥视;闭环阶段用固定 Support → delayed 两面即可 |
| 时间衰减 | 后置 | 由 M1 校准后再定;闭环阶段不引入 |
| 实体后验 / 分层先验 | 后置(默认 OFF) | 未证明存在可迁移方差;样本量下会缩回先验 |
| 软混合 / 两决策 ScopeSpec | 后置 | 仅 pooled 相关;由 Consumer 选择决定 |

---

## 3. M 测量的执行清单(0 LLM;字段表见附录 A)

**核心闭环必需(阻塞方法选择)**:

| 编号 | 研究问题 | 一句话规格 | 停止条件 |
| --- | --- | --- | --- |
| **M-R0 修订可达性审计**(**已完成**,Opus,0 fit / 0 LLM;`artifacts/main_protocol/m_r0_reachability.{json,md}`,脚本自证复现 30/30 普查表) | HEC-1 里为何 0 `NARROW`、REVISE 全败、Active Skill 冲突后去了哪 | 结论见 §1"实现限制"行;三处推翻计划预设:部署权从未丢失(`SUSPENDED` 前提不成立)、8/11 Draft 来自持权卡、`key in held` 成立;三个附带仪器缺陷:`by_scope` 只按 Scope 不比程序(1 例跨程序污染已证)、`restrict()` 不写 `census_key`(同键可开第二张壳归零计数)、`retries_per_candidate` 第 3 次按构造不可达;未知 U-1…U-5 | 已止 |
| **M-W Workflow 可修复性**(**暂不发车**;案例与预算已按 M-R0 更新,见附录 D) | 旧程序失败时,是否存在**一步合法修改**,仅凭**当时可见的反馈**即可选中,并在**更晚的单元**改善并过四线 | **时序**:冲突 = 该单元 Support/delayed 面上可见的失败;修改在冲突单元两面上按预写规则选定;验证在顺序上更晚的 ≥2 个可评单元的 Support + delayed 面;+144 只作事后分层。**案例(M-R0 实测)**:按 (cohort × origin × delayed × Program × Scope) 去重 **8 个情境**、41 个实例(33 个为同一确定性评估的重复观测);(情境 × 顺序)= 20,其中 **7 个只差 `coverage_floor`——M-W 不改 Scope、coverage 由谓词对 raw 窗口解析决定,任何 Workflow 编辑都不可能过规则①,预先记 `NOT_REPAIRABLE_BY_WORKFLOW`,不买**;可修 13,再要求后续 ≥2 可评单元 → **11 个组合为分母**。Tier-1/Tier-2 不是两批案例(7/8 情境重叠),Tier-2 唯一额外成本 = 祖先影子对照。结论类别:`RULE_FOUND_NO_FIX` ≠ `NO_LEGAL_EDIT_IN_SET`(post hoc)≠ `SLOW_ABSTAINED` | **充分性规则只此一条**:11 个组合全部评完即止,逐组合报,不做跨情境平均;Gate 0 不适用于 M-W(它是格级前提表的规则);附录 A 原"区间半宽 ≤0.15"作废 |
| **M1 可学习性(持续性)** | 哪个粒度的历史能预测下一窗符号/严重伤害 | 相对无历史基线 B0(多数符号)/ B1(程序级均值符号)/ B2(置换)的**增量**;粒度 = 程序级 / Pattern 层级 / 实体级;Consumer = pooled 与 per-channel(M5 产出) | 主判据:增量 90% bootstrap 下界 > 0 **且** 点估计 ≥ 0.05(准确率)——两者皆满足为"可学习";下界 ≤ 0 且区间半宽 > 0.05 为"证据不足";区间落在 ±0.05 内为"无实质增量" |
| **M3 Pattern 预测增量(Ridge 版)** | 决定编辑面是 Workflow 还是 Scope | 12 词表特征对下一窗符号/严重伤害的 AUC,相对程序级均值的增量;pooled 与 per-channel | `INFORMATIVE`:AUC 增量下界 > 0 且 ≥ 0.05;否则 `WEAK` / `UNINFORMATIVE`;TSFM 列缺失不阻塞 |
| **M4 种植对照(估计器侧)** | 估计器栈能否学回已知规则 | KDD development 副本注入"程序 P 仅在层 S 有益";M1/M3 管线须复原 | Gate 0 条件;学不回 → 先修估计器 |
| **M5 Stage A per-channel 重打分** | 换 Consumer 后收益是否保留、伤害形状是否改变 | 既有规格(收口裁定 §2),≤3000 fits;按实际逐序列拟合计数 | `GAIN_RETAINED / LOST` × `SHAPES / NO_SHAPE`;两读数同报 |

**补充解释与外部有效性(并行,不阻塞;是否阻塞由"是否改变当前方法选择"决定)**:

| 编号 | 内容 | 依赖 |
| --- | --- | --- |
| M0-a 同数据任务—Consumer 设置比较 | Yahoo-24 上 `Forecast/Ridge` vs `AD/IForest`、`AD/PCA`(同时改变 Consumer 与评价目标,**不称 Task 匹配**) | 呈 sol 边界确认;数百 fits |
| M0-b 同任务换数据 | KDD vs NOAA(均 development)验缺口层 × 程序交互同向 | NOAA serving 适配器可用性 |
| M0-c TSFM 探针 | 冻结小模型 context-only headroom | **下载未批,不阻塞任何 Ridge 只读分析** |
| M2 方差分解 | Forecast 内程序 / Consumer / 层 / 数据集 / 实体 / 窗口 | M5、M0-b |

**Gate 0(测量有效,具体口径)**:每个被读格 ≥ 8 个可计分窗口、每面 ≥ 15 条被服务序列;格级均值 Δ 的 90% bootstrap(按窗口块)区间半宽 ≤ 0.10;M4 复原成功。任一不满足 → 该格记"证据不足",不进入方法分支。
**纪律**:事后诊断(M-W 的修改选择、M-R0 的归因)只用于开发选路,不得作为 live Agent 的未来答案进 prompt / 记忆;M 未检出增量时,按上表区分"证据不足"与"无实质增量",不自动推断"没有可迁移结构"。

---

## 4. 最小自然闭环的具体设计

### 4.1 臂(3 个实验臂 + 1 个参照)

| 臂 | 局部适应 | 结构修订 | 识别什么 |
| --- | --- | --- | --- |
| **A** | 照常:Fast(LLM)逐单元提议 → Support → delayed 权威门 → Active / 撤销;外环普查的确定性 `ADD` / `REVOKE` 建议照常 | **关闭**(无 `NARROW` / `REVISE` / Workflow 修订) | 适应的基线 |
| **B** | 与 A 完全相同 | **LLM Slow** 在唯一编辑面上提修订 | B − A = 结构修订的价值 |
| **C** | 与 A 完全相同 | **确定性提议者**,同预算、同编辑面(枚举一步合法修改,按失败窗之前的 bank 证据排序取 top-1) | B − C = LLM 提议的额外价值;C − A = 确定性修订的价值 |
| Static | — | — | 0 成本参照,同 HEC-1 |

三臂共享:起始快照(K0 或 h0,预注册)、证据权限(只读 held-in Support / delayed;评价面只计分不进 bank)、候选空间(§2)、Fast 预算与 fits 预算;Slow 预算只在 B 用 LLM、C 用 0-LLM 等价步数。不再使用 HEC-1 的"A5-frozen(每单元重建快照)"作为第一臂——那会把适应与修订两种贡献混在一个差里。若 K0 非空与 h0 起点都要测,则两套三臂分开跑,不合并读数。

### 4.2 两类链,分开计数

- **Chain-D(修复失败 Draft)**:**非持权**候选过 Support、败 delayed → Draft → 编辑面上修订 → 新版本在后续独立单元过 Support + delayed → Active → 后续重遇改善。证明"候选可以被改好";**不是**持续自进化链。HEC-1 中此类 Draft = 3/11。
- **Chain-S(修订已 Active 的 Skill)**:Skill Active(K0 或课内获得)→ 后续单元 delayed 面 adverse 读数(风险线)达阈 → 修订提出 → 后续独立单元验证 → Active v2 **接替** v1 → 后续重遇:v2 相对 v1 影子改善,伤害不升。**这才是当前设定的自进化链。** HEC-1 中持权卡的 delayed 失败 = **8/11 张 Draft**(Draft 来源;6 张直接来自 K0),**7/11 条谱系**达 ≥2 次风险线 adverse(风险谱系)——**Chain-S 的材料在 HEC-1 里是存在的,只是普查看不见它**(§1 ①②)。

分母:两类链各自一套(进入可修订状态 / 实际提出 / 通过验证 / 重遇改善)。**口径纪律(astra)**:"**7/11** 条谱系达 adverse 阈"(风险谱系计数,按 M-R0 §1.5 表逐行)与"**8/11** 张 Draft 由持权卡产生"(Draft 来源计数)是两个不同的分母,不得混写;本计划此前把前者也写成 8/11,**勘误**。冻结臂被丢弃的 15 张 Draft 不计入任一分母(设计行为)。**修订漏斗必须分层报**(astra):产生候选 → 阈值可行 → 根 Scope / 修订额度合法 → **生命周期能接收** → replay 通过 → 后续独立验证;"候选可行"不等于"修订路径打通"。

### 4.3 祖先 Skill 从哪来、版本怎么接、执行权怎么回来

- **祖先来源(自然)**:(a) K0(Phase S 已有 1 张;HEC-1 三顺序中**始终持权**,此前"K0 丢失"的说法系误读 `active_skill_ids` 只列课内新铸 id,撤回);(b) 课内经 A 路径激活的 Skill(M-R0:forward A3 p05 `outlier_mad`、reverse A5 p06 / A3 p23 `winsorize`、interleaved A3 p18 `outlier_mad` / p25 `outlier_iqr`)。同名 `skill_id` 是确定性命名碰撞,不是 K0 泄漏(`a5_a3_isolation` 通过)。两来源分开报告。
- **冲突判定**:沿用 `MIN_ADVERSE_UNITS_FOR_NARROWING = 2`(同一 Active 谱系 ≥2 个 adverse 单元)。**"撤销先于外环步"的假设已被 M-R0 推翻**(部署权从未丢失、0 撤销、`REVOKE` 不可达),**`SUSPENDED` 暂不立项**。危险 Skill 的部署权取消在任何方案下都不得延迟——这一条与本轮无关,保留为原则。
- **协议事项(单列,呈 sol;不混入本轮接线修复)——delayed 证据是否进入后续普查**:现行 `_bank_rows_from_round`(`run_hec1.py:1836`)只收 Support 探针行,delayed 面是权威门、不是证据;而 `NARROW` 所需的 adverse 计数在本协议里只能来自 bank ⇒ Active 卡的冲突对普查不可见。**边界**:delayed 面是 held-in 反馈(正典允许学习),评价面 +144 永不入 bank(曲线可读性所系)。可选方案:(a) delayed 读数以"谱系级 adverse 事件"进入普查、不进 bank 行(不改 Support 证据的用途);(b) 维持现状、`NARROW` 只由 Support 证据触发(M-R0 反事实 26 个即此类)。这是**协议选择**,须 sol 裁定后才进合同;本轮最小接线修复不含它。
- **版本谱系**:修订产物是同一 `lineage_id` 下的新 `version`,**继承** `revisions` / `verification_attempts` 计数(上限 2 / 3 不变);普查去重键改为 `lineage_id`,使改了 `program_steps` 的版本不会被当作新 `census_key` 开新壳、重置计数(`outer_loop.py:394-401` 的去重意图延伸到 Workflow 面)。
- **执行权**:新版本没有部署权;只能经现有合法路径——后续独立单元 Support → delayed 权威门四线全过——才 Active;Active 后 v1 转 `SUPERSEDED`(只作影子对照,不部署);同一谱系任何时刻至多一个版本 Active。LLM 只提议,不授予任何权。
- **祖先影子对照**:v2 Active 后每个后续可评单元,同面同序列同时计算 v1 的逐序列增益(1 次额外 fit / 面,计费);读 v2 − v1 在目标情境(触发冲突的层/序列组)与非目标序列上的差,后者用于退化检查。
- **预算不重置**:修订由 Slow 预算(外环)承担,不消耗当单元 Fast 预算;同谱系累计验证尝试 ≤3、修订 ≤2;超限记 `REVISION_BUDGET_EXHAUSTED` 并关闭谱系。
- **外环预算**:M-R0 实测——一次尝试 = 恰好 1 次物理调用;3 步到达 Slow、6 次调用、5 次到达阈值工具、**`CALIBRATED` 0 次**;**完整结构提议的成本尚未测得**,不得由"两次耗尽"推定"四次够"。`retries_per_candidate = 2`(最多 3 次尝试)与 `outer_llm_per_step = 2`(每步物理硬顶)**可以同时存在**——前者是候选级上限、不保证用满,后者是每步硬顶;**重试次数不得绕过每步物理硬顶**,若要提高实际可用次数须前瞻批预算,不得以"修循环头"为名移除硬顶(astra 修正;原 E.4(d) 相应改写为"文档化两者关系、不改数值")。物理顶在原路径可达性检查(§7.0 ③)量出成本后再呈批;C 臂用同样的完整提议数上限。

### 4.4 两项语义决定(主线提案,呈 sol 确认后实施;astra 复核 R1 / R2)与一项计费口径(R3)

**R1 · 什么算同一个证据单元(astra 拆分后:本轮只定身份、精确去重、重复不一致处理;聚合/投票策略另列 R1-b,未批)。**
- **R1-a(本轮,仪器规则;Opus 最新补修已实现主体)**:身份 = (census_key = 任务|程序|根 Scope, 单元, 评价面, 精确逐序列读数, relation)。当前 bank 只含 Support 探针行,面固定;若将来 delayed 行进入普查(§4.3),面必须显式进身份。**同身份 → 同一观测,折为一条**,不增加 `unit_count` / `positive_units` / `adverse_units`,不增加阈值搜索权重(bank 保留每一行,去重只在普查视图)。**同单元、同 census_key、读数或 relation 不同 → 两条都保留、单元打标**(`units_with_more_than_one_distinct_observation`,已实现)。
- **R1-a 补一条(仍是仪器规则,不是策略)**:一个单元不能被数成两个单元。被打标的单元在 `unit_count` 中只计一次;在 `positive_units` / `adverse_units` 中**只在其全部不同观测 relation 一致时计一次,不一致则两侧都不计并单列** `units_with_conflicting_relations`。这不选"取最坏/取最后/取部署",只是不让一个单元投两票;当前补修里不一致观测仍各计一次,须补。
- **同身份、读数不同**(评估本应确定性)→ 记 `NONDETERMINISTIC_REPEAT` 仪器异常,不聚合、不投票,先解释来源。
- **R1-b(学习策略,未批,不实施)**:"NARROW 只看该谱系实际部署动作的 relation、探针不投票"与"ADD 取最佳 relation"是**新的学习策略**,astra 裁示不得作为去重规则默认批准;留待 §4.3 协议事项(delayed 证据用途)一并由 sol 裁,裁前普查对不一致单元按上条"不计并单列"处理。

**R2 · 已有 Draft 如何承接 Active Skill 的 `NARROW`(astra:原则认可,**确认后实施**;主线在此确认,交 Opus 实施;复用既有通道,不新开壳、不清零、不改 FLAGGED)。**
- 谱系**无** Draft → `NARROW` 经 `open_restricted` 开该谱系的**第一张**受限 Draft(原设计),计数从 0 起,无部署权。
- 谱系**已有开放 Draft 且 `REVISABLE` 且 `may_add_clause()`** → `NARROW` **归并为一次 `REVISE`** 落到该 Draft(`record_revision`),保留祖先关系与修订/验证计数。
- 谱系已有 Draft 但为 `FLAGGED` / `WAITING` / 已关闭 / 额度耗尽 → **在 propose 阶段就不产出 `NARROW`**,记 `NO_NARROW_TARGET(reason)`,不调用 Slow、不跑 replay(避免付费后在铸壳处被 `LINEAGE_ALREADY_HAS_A_DRAFT` 拦下)。
- 任何路径都不得因候选叫 `NARROW` 而绕过状态限制;`FLAGGED` 仍禁加子句。**预期后果要明写**:HEC-1 中 10/11 张 Draft 为 FLAGGED,故在该几何下多数 `NARROW` 会落入 `NO_NARROW_TARGET(FLAGGED)`——这是原路径的诚实读数,由 §7.0 ③ 的分层漏斗报出,不是要绕开的障碍。
- 守卫一致性(D1)是 R2 的前提——**Opus 最新补修已落地**(`open_restricted` 与 `restrict` 共用 `by_census_key or by_program_and_root` 含已关闭;`root_for_scope(scope, program_steps=)` 经 `root_lookup` 对多根歧义返回 `None` 并让 preflight 拒绝),不重派。
- **实施要点(交 Opus)**:在 `propose_candidates` 生成 `NARROW` 之前查 `ledger.by_program_and_root(program_steps, root_scope)`:无 Draft → 保持 `NARROW`(经 `open_restricted` 开第一张);有开放 Draft 且 `REVISABLE` 且 `may_add_clause()` 且 `verified_since_revision` → 候选改记 `kind="REVISE"`、带 `draft_id`、`provenance.merged_from="NARROW"`,走现有 `consolidate` 的 REVISE 分支(`by_id` + `record_revision`);其余状态 → 不产出候选,写入 `record.rejected`,`outcome="NO_NARROW_TARGET"`,`reason ∈ {FLAGGED, WAITING, CLOSED, REVISION_BUDGET_EXHAUSTED, VERIFICATION_PENDING}`。`consolidate` 里现有的 `LINEAGE_ALREADY_HAS_A_DRAFT` 分支保留为**不应再到达的兜底**并断言计数为 0。测试:三条正例(无 Draft / REVISABLE 归并 / FLAGGED 拒绝于 propose)+ 一条"归并后 revisions 与 verification_attempts 不变"。

**R3 · 共享 Support baseline 的计费(主线意见)。** 是否发生过不可裁定,只有归属可裁定:共享 `_baseline` 的 `_evaluate` 按单元**记一次**进 `Ledgers.baseline_fits`(字段已有、从未累加),归入整场实际成本、不分摊给任一臂;缓存命中不重复计费,两次独立执行的 raw fit 各记;展示分摊方式后定,不阻塞物理计数修复。同理,异常路径的 fits 必须在**执行层**累计、外层结算差额,不能依赖成功返回的 result 列表(astra 补充②)。

### 4.5 修订选择目标的对齐(主线提案,2026-09-06;ACC-1 收口后的唯一方法问题;不是新架构)

**事实(只读核实)**:权威门与课程计分的 `aggregate_gain` / `harmed_fraction` 以**全服务人群**为分母(`run_hec1.py:529-530`,域外序列走 raw、增益 0);阈值工具的 `_reading` 以**被选中行**为分母(`scope_threshold_tool.py:193-194`),`calibrate` 取"过四线的最宽阈值",影子 `best_stump` 取"被选行平均增益最大"。整条修订链没有任何一处把修订与祖先放在同一分母下比较。**推论**:收窄修订的总收益 = 祖先总收益 − 被排除序列收益之和;收窄有益 ⇔ 被排除集合在全人群口径下净负。ACC-1 的 `missing_fraction >= 0.05` 校准读数 0.578 × 覆盖 38/100 ≈ 0.220 < 祖先同期 0.251,即被排除集合在**边界前证据**里已是净正——对齐后的规则会在第一次拟合前拒绝它。m_r0j 后 16 单元(剔除边界前 5 个)复算:修订 +0.106860 vs 祖先 +0.170298,受损 36 vs 51,权威门 4/16 vs 4/16(原报告"净损 2 门"系边界前单元,口径已更正)。收窄下受损数不增加是构造性的(嵌套 Scope、退出回 raw),不构成"选得聪明"的证据。

**提案(astra 二审后改写;呈 sol)**——三件事拆开,不合称"目标对齐":
(a) **效用排序分母** = 全服务人群(与评价一致),这是对齐;
(b) **风险约束保护的人群**(工具现用被处理人群分母,权威门用服务人群分母)——是否统一是**明确的方法变更**,数值阈值不变也会改变通过集合,第一包中作为独立一列分析,不并入 (a);
(c) **比较对象 = 当时真正可执行的策略**,不设两面硬门:若父版本按**当时已到达的合法反馈**仍可部署 → 修订须 `aggregate_served(修订) ≥ aggregate_served(父) + material`,否则 `NO_REVISION`、父版本保持;若父版本已因冲突不可部署 → 实际可执行回退为 raw(收益 0),修订与 raw 比,并报"是否恢复可部署性"。父版本是否合格由生命周期的合法反馈决定,不用未来结果挑。"Support 与 delayed 均不退步"的硬门**撤回**(本次数据不支持:修订在 Support 面全人群收益亦下降,涨的只是描述性过线计数;且会误伤合法的风险修复);两面读数照报。
可行者中取 `aggregate_served` 最高,并列取更宽;"不修订"为一等选项;Slow 只提方向不给数值;不改阈值、不加状态、不上分层模型。

**三个完整工作包(astra 2026-09-06 建议,主线采;每包一次性定比较目标与方法选择,Opus/Kimi 完整执行,包内解决实现问题,astra 只复核改变结论的问题与整包结果)**:

**第一包 · 量清两类编辑各有什么可赢的(开发级,曝光数据)**。两条线可比:**Scope 线**(程序不变、只改适用范围)与 **Workflow 线**(Scope 与 serving 几何不变、只改程序;不得退回 P4 的全局训练语料策展设置);两线同 Consumer、同域外 raw 回退、同父子比较口径(§4.5 c)。列:① 不修订 / 当时可执行策略 ② 原规则 ③ 对齐规则(仅效用分母)④ 对齐 + 风险分母统一(独立列)⑤ 随机保留参照(解析期望)⑥ oracle 三层——L1 逐序列事后 oracle、L2 合法候选空间内事后最好、L3 只看边界前证据的选择器——**三层须交代**:同一风险与覆盖约束、整段固定候选还是逐单元换、含"不修订"与合法回退、同一服务人群与计分;L2−L3 混合未来信息优势、估计误差与跨窗变化,只作定位投入的工具。读数:逐单元约束账(收益 / 伤害线 / 覆盖分列)、排除精度与代价、L2−L3。**只回答三问**:合法空间是否存在比当时可执行策略更好的修改;过去证据能否选出它;其价值是提高收益、恢复可部署性还是仅减少处理量。**时间纪律**:看过 pos 6/9/12/14 后设计的 Workflow 候选只用于测 L2,不得塞回 k1 称自然产生。能精确复用的预测直接复用(`ReplayPredictionCache`),缺读数的如实补最小评价,不为 0 fit 把上界算残(fits **待补**)。包末做路线判断,不再解释单条子句。

**第二包 · 把"研究者找到的改进"变成"系统自己产生的改进"**(前提:第一包找到可学习的改进空间)。臂:A 局部适应结构冻结 / B LLM 从**当时可见的失败档案**提出结构修订 / C 同预算确定性修订;**消融**:B 去掉或打乱失败档案(优势是否来自这段经验)、同预算 validation-search(重新搜索能否追平);共享初始知识、合法编辑能力、反馈条件与预算口径。LLM 职责写清:从失败证据提出有针对性的程序性修改,不猜阈值;若 C 稳定改善而 B ≤ C → 修订机制有价值、LLM 增量未证,据此调整 LLM 职责。修订必须经真实生命周期验证、晋升、后续使用,不以写入 Draft 代替。交付 = 四个行为差异:有失败经验 → 更有针对性的修订;去掉/打乱 → 优势消失;修订知识到后续场景仍有用;同预算重搜不能轻易取得。

**第三包 · 完整项目目标:积累与迁移**。A5 vs Static / A5 vs A3 / A3 vs Static,按三情境摆:情境一(KDD)形成工作流 → 情境二(KDD 其他 block/顺序或 NOAA)发现伤害并修订 → 情境三(Wind Farms,未参与开发)检验冻结知识 / 无积累 / 修订知识三者的适应。同批单元换顺序只算顺序稳健,不代替独立复现。迁移资产并行准备(Wind Farms 下载、NOAA→HEC-1 适配器**现在申请**),但不同时开五条实跑线。Solar 不预写成弃疗集;NOAA 效率信号复查复现,效率不作唯一主终点——固定反馈预算下效用、风险、覆盖与达到目标所需反馈量同报。

**6 pro 增补(2026-09-06,主线采;三包骨架不动)**:
- **第一包加两项读数**:① **修订效应稳定性**——对少量固定差分(Scope 差分经 `ReplayPredictionCache` 0 fit;Workflow 差分计真实 fits),测 \(d_e(P\to P')\) 跨单元符号一致率与幅度,对比 \(g_e(P)\) 的一致率——回答"怎样修改"是否比"哪个程序好"更可学习;② **面向动作的观测** \(z_P=\psi(x,P,\text{Consumer})\),先限三个(修改是否连续成段 / 是否集中于近期 context / 与缺失位置的重叠),只由 context 与合法试执行算得、无未来信息,作为**版本化词表扩展**冻结分箱,测其对 \(d_e\) 的增量预测力,无增量即停。同一谱系上给 (P,S)×(P′,S′) 的 2×2 交叉读数(含交互项 I),只用于解释哪一面值得、是否配合,不绕开单变量比较。
- **第二包加一臂、一检验、一张冻结表**:臂 = **普通 LLM 反思后修订**(同信息量,不做针对性对照与持久 Skill 修订),使四臂为 冻结结构 / 普通反思 / 对照驱动修订 / 同预算确定性;先给两 LLM 臂同样信息比整套机制,再消融"多了什么信息"与"怎样使用"。检验 = **新会话再生成**:形成新 Skill 后,在未参与构造的下一情境开新会话,旧 Skill vs 新 Skill 各让 Fast 重新生成程序,比的是观察、有效候选与达到同等质量所需反馈是否不同(区分:Skill 未被用 / 被用但行为未变 / 行为变了但无助益);Runner 不直供上一轮赢家,跨域原始 Episode 仍留 Slow 侧。冻结表 = 解释类型 ≤5 种(程序不合适 / 适用条件不准 / 规律漂移 / 观测不足 / 仪器问题),每次失败 ≤2 个候选差分(一 Scope 一 Workflow),发车前冻结、之后不加行。**最小接线修复补一项**:Slow 提示只送前 60 行 bank(`run_hec1.py:1665`)而影子用全量 100 行(m_r0e 已核),修好之前任何"LLM vs 确定性"的读数都不成立。
- **第三包加一臂**:累积 Skill **结构冻结、只允许 Target 本地校准**的 A5 变体,把"有历史知识"与"历史知识可被反例修订"的价值分开。
- **纪律三条**:delayed 反馈的时间角色——合法到达后可成为未来决策的历史,一旦用于构造新修改即不得再作该修改的独立验证(§4.3 措辞采此);研究者修正评价分母、泄漏边界、成本计量不是作弊,替系统逐次选答案才是;序列读数次数不是独立样本数,不以后来过门倒推当时可部署。

**这批单元已曝光,第一包结论只算开发比较;方法贡献由第二包的四个行为差异建立,项目目标由第三包的跨情境积累兑现。若第二包中去掉失败经验优势不消失、或同预算搜索能追平 → 方法贡献不足,调整机制或定位,不以审核与日志包装进展。**

### 4.6 提议宽、接受严、停止简单——第一包发车前外环 Slow 的五处改动(用户设计取向 2026-09-06;主线提案,1/3/4/5 含协议或语义变更,呈 sol)

**事实依据**:ACC-1 真实 Slow 弃权发生在复用内环 preparation 角色说明("never infer candidate utility / inspect first / abstain")之下,只换成外环角色说明同证据即两次提议(m_r0f,n=1 对 n=1);提示只送 `rows[:60]`、无单元/序列身份、只含 Support 行、上次失败只回一行"方向不可行"(`run_hec1.py:1649-1665`);12 词表中 6 个分数型特征走默认分箱 (0,1,3,6),有效切点只剩 0.0,96 个三元组仅 8 个可行(m_r0f §4);外环每步 2 次顶使第 3 次重试构造不可达;历史"0 修订"几乎全属运行时提前挡住(词表错配、Draft 关闭、FLAGGED 无出口),不是模型弃权。Self-Harness 亦允许 `decline`,但默认 4 个提议位、无变化/占位无效、弃权须给证据理由(`multi_proposer.py:56-125`)——**默认提议、弃权为需辩护的例外**。

| # | 改动 | 性质 |
| --- | --- | --- |
| 1 | **让它看见**:完整 bank 行(不截断)、带单元/序列身份(仅边界前单元,信息墙不变)、Support 与 delayed 结果均作历史(delayed 一旦用于构造某修改即不得再作其独立验证,§4.3 采此)、受损与获益序列并给、上次失败的具体后果(失败线 / 排除了谁 / 损失多少) | 接线 + §4.3 协议事项 |
| 2 | **换问法与角色**:外环用自己的角色说明;问题改为"根据该 Skill 的成败记录,提出 ≤2 个具体修改(Scope 子句和/或程序差分)并说明预期改变哪些序列;不提须给有证据的理由";默认为提 | 接线(提示) |
| 3 | **修栅格**:6 个分数型特征给与量纲匹配的冻结边缘 | 仪器缺陷,词表栅格版本化 |
| 4 | **归因去否决**:`FLAGGED` 由"禁止收窄且无出口"改为"建议先试程序差分"的顺序提示;三态机其余不动;五类选择题模块继续关闭 | 协议变更 |
| 5 | **预算退到包级**:总 LLM 调用与 fits 上限 + 可中断 + 模型可主动停;删每步 2 次外环顶;成本全记;**臂间比较时在包级对齐** | 预算语义 |

**接受侧不动**:全服务人群分母;比较当时真正可执行的策略;既有四线;独立后续验证;未验证不部署;发车前冻结。**放宽后的真实风险**:曝光开发数据上更多次比较 → 过拟合;防线在接受侧与冻结,不在提议侧加门。**保留的四条**:不偷看未开放答案;模型不改裁判;未验证不部署;发车前冻结。其余规则一律回答"防明确问题,还是不放心模型替它决定"——后者简化或改提示。

**第一包新增"提议者试跑"**:forward 谱系若干边界上,以新输入与问法让模型提,报 提议数 / 可行数 / 过接受数 / 后续有价值数,与确定性对照并排——回答"给合理探索空间,能否学到有效修改"。

---

## 5. 成功、完成与失败(预写)

- **成功标准**:≥1 条**自然 Chain-S**(不是 Chain-D、不是种植副本),且预先安排的重复检验成立——**同数据换顺序**(第二条预注册顺序中同一谱系或另一谱系再现完整链)只算**顺序稳健性**;**新单元 / 新实体组 / 新数据集**上的再现才算**独立复现**。两者分开报告,不互相替代。
- **完成标准**:固定课程(KDD 26 单元,两条预注册顺序)、固定候选空间、固定预算内跑完;报告**全部**修订尝试及失败分母(两类链各自:进入可修订状态 / 提出 / 验证通过 / 重遇改善),连同分歧数、完整决策数、收益、伤害、覆盖、成本。
- **一条链是机制证据,不等于系统有效**(astra):三项同报、分开判——① 是否出现自然 Chain-S;② B − A 的**整段**课程累计安全效用、伤害与成本;③ B − C 的增量。解释矩阵预写:链有而 B − A ≤ 0 → "修订机制可触发但无净收益";C 成功且 B ≤ C → "修订机制有效,LLM 额外价值未证"(两件事不合并判负);B > C > A → 三者皆立。
- **收口规则(任一触发即按规则收口,不延长)**:
  - 预算耗尽 → 已完成单元照常读数,记 `BUDGET_EXHAUSTED_AT_UNIT_n`;
  - 无冲突(无 Active 谱系达 adverse 阈)→ `NO_CONFLICT_OBSERVED`:闭环未被激发,不判成败,报 Active 数与 adverse 分布;
  - 有冲突无合法修改(枚举为空 / Slow 弃权)→ `NO_LEGAL_EDIT`:分开报 LLM 与确定性提议者;
  - 修改提出但验证全败 → `REVISION_NOT_VERIFIED`:报失败线分布;
  - 验证通过但重遇无改善 → `NO_REENCOUNTER_GAIN`:链不成立,记 v2 − v1 分布;
  - 全部单元完成无链 → `CHAIN_NOT_FORMED`(与 HEC-1 判词并列,不覆盖)。
- **禁止**:为凑链、凑分歧或得正号而延长课程、挑顺序、改阈值、改候选空间、把种植副本结果混入自然读数、把 Chain-D 报成 Chain-S。

---

## 6. 数据资产与最终主实验

**角色(数据资产审计已完成,grok,`artifacts/main_protocol/m_data_asset_audit.{json,md}`;下表按该审计更新)**:

| 数据 | 角色(合法) | 不能再当什么 | 接入状态 |
| --- | --- | --- | --- |
| KDD 含缺失(270 条 / 可读 239) | 闭环与机制测量 development;Phase S 的 Source;已跑过的 Target held-in | fresh / Natural Final | **唯一**接上 HEC-1 scoped serving 的数据 |
| KDD 已填补(缓存 270,NaN=0) | 无缺口 outlier 线的 development replay | 与含缺失并表;fresh | — |
| NOAA 2024 / 2025 `[8760,17520)` | 仅 development(`FRESH_A5_DELIVERS` 已开 Outcome) | 再称 fresh | 旧 ForecastCell 存在,**无映射到 HEC-1 几何的工厂 → M0-b 阻塞**,需另申请写适配器 |
| NOAA `beyond_17520` | 保持密封 | 任何读取 | — |
| Solar 10 min(137 × 52560,授权元数据缺失=0) | F2 密封终验候选,隔离令有效 | development;现在开封 | 无 loader(`traffic_or_solar_loader_available=False`);0 缺失只排除缺口依赖 Skill,**不排除 K0 `outlier_mad @ z_peak>=3`**(是否成立 = UNKNOWN,不开数据) |
| Yahoo 字典序前 24 条(real_1, 10–19, 2, 20–30, 3) | AD development / M0-a | fresh;继续拟合这 24 条 | T6/P3 AD 加载器有跑数;P1 主协议 `yahoo_loader_available=False` |
| Yahoo 其余 41 条(含 real_4–9) | 密封一次性终验 | 现在读 Outcome | 故意不进 L1 roster |
| NAB 本地五族 37 条 | **全部已用作 Source 或 AdExchange Target → 只剩 development** | 任一本地族当 virgin Target;第 5 个 Source cohort | 官方未落盘 10 条(AWS 9 + rogue_agent_key_updown)freshness 待裁,不建议当 fresh |
| UCR | 仅各线已开 TRAIN(capstone 另开过 Epilepsy2 TEST) | 本轮不用;不能以"P4d TEST 未读"说全库未见 | — |

**结论**:本地**没有**仍合法的、带天然缺口的预测 fresh Target。主推荐 **Monash Wind Farms Minutely(含缺失版)**:339 条、最短 6345、Missing=Yes、缺口为归档属性非注入,最短长度撑得住 HEC-1 origin 3816+48;与 Solar 同属发电域、过程不同(若 Solar 仍作 F2 需披露域邻近)。次选 London Smart Meters 含缺失版(5560 条),但最短 288 撑不住现行 origin 网格。**下载与适配器均另行申请**(§8)。

**迁移比较组织**:以数据集为 Source / Target 单位;两层检验——同任务新数据集(经验能否跨数据复用)、新域或不同 Consumer(复用边界);每个 Target 内保留合法 held-in 校准与时间隔离的最终评价;窗口作重复观测、同域数据集保留依赖(不当独立样本);A5 / A3 / Static 同 Target 反馈预算、同 Consumer、同程序空间、同风险线、同密封评价(DomainBed 口径)。
**主终点前瞻选定**(冻结前二选一,不得事后改):A5 − A3 的 held-out 累计安全效用,或达到同等安全效用所需的反馈单元数;两者都同时约束伤害与覆盖,并报成本。
**主线建议(呈 sol)**:以"达到同等安全效用所需的反馈单元 / Consumer fits"为主终点、"效用不劣于 A3 且伤害不高于"为约束——依据是项目唯一的自然数据 A5 证据(NOAA)正是效率效应:
菜单小、两臂最终都能找到菜单内最优时,最终质量打平是预期结果,先验买到的是速度与安全。若闭环阶段证明生成型修订能带来 A3 找不到的处理,再考虑效用主终点。
**AD 检查的两种沉默分开解释**:算子 `allowed_tasks` 不含 anomaly_detection 导致的**合法沉默**(注册表把 denoise / outlier / structural / decompose 全部排除在 AD 之外,`operators/registry.py`),与 Agent **从经验学得**的伤害避免,是两件事;后者只能由 impute / align / shape 等 AD 可用算子上的行为证明。

---

## 7. 时间、预算与人员

### 7.0 关键路径(astra 收缩版;所有任务分"阻塞首次闭环"与"可后置"两类)

**第一批核查已完成**(M-R0 + 计费核对 = Opus;数据资产审计 = grok;文档口径已同步)。**下一路径固定(astra,2026-09-05)**:

| 步 | 内容 | 性质 | 可后置(不挡) |
| --- | --- | --- | --- |
| ① 最小接线修复 | **状态(以 Opus 最新工作树补修为准,主线只读核对 2026-09-05 深夜;不重派已修项)**:首因 C1–C7 已闭合;**D1 已修**(`open_restricted` 共用守卫含已关闭;`root_for_scope(program_steps=)` 经 `root_lookup` 歧义拒绝);**D2 已修主体**(同单元同精确读数同 relation 折一条,不一致单元打标;**残留**:不一致观测仍各计一次 → 按 R1-a 补条"单元只计一次、不一致不计并单列");**D4 已修主体**(`_bill_support_fits` 从执行器读增量,异常路径也计;baseline 执行 / 缓存命中 / 失败分列计数,定价留协议;Fast 后端 `_MeteredFastBackend` 逐请求过 `guard`,顺序帽在传输前拒绝);**D3 待实施**(R2 已确认,§4.4 实施要点);**D5 待补**(反例测试:R1-a 不一致单元、R2 三正例、`reserve` 剩余额度边界、except 路径 support_fits;不加长 e2e、不为 NARROW 加 K0 课程)。补丁整体仍 `NEEDS_FIX` 直至 D3/D5 落地并经非作者复核。(d) 重试与硬顶只文档化关系,不改数值 | 仪器修复,新 commit(冻结规则:一条课程一个 commit) | — |
| ② 非作者复核 | 只看该 diff;确认 delayed 是否入 bank **不在**本轮改动内;回归差集;**测试"缺少覆盖"不称"假通过"**——下一轮补具体反例,不以测试总数证完整 | 复核 | — |
| ③ 检查原有修订路径的可达性(0 LLM;修复期间可先备清点) | 在修复后的接线上重放 HEC-1 三顺序 bank 普查 → **分层漏斗,每层标注读数来源**:产生候选(重放,实测)→ 阈值可行(`best_stump` **影子**,全 12 词表,0 fit;标"影子可行",**不是** Slow 提议)→ 根 Scope / 修订额度合法(重放,实测)→ 生命周期能接收(R2 规则,实测)→ **replay 通过 = 未评估** → **真实 Slow 提议 = 未评估** → **后续独立验证 = 未评估**。astra 规则:凡未实际执行、且无可合法复用的完整读数的层,一律标 `未评估`,**不得用影子 stump 可行代替**。**历史清点按各外环步当时状态**(第 k 步时哪些谱系持权、哪些 Draft 存在及其状态、修订/验证计数),**不用课末状态回填**。M-R0 的 26 / 6 只作**旧语义历史参照**,不要求凑回,差异须解释 | 只读重放;0 fit(replay 屏若要跑须另批);不开课程 | M1 / M3(Ridge)/ M2 的 0-fit 估计;M5 Stage A(需授权) |
| ④ 最小真实验收 ACC-1(附录 F)→ 再决定 M-W / 编辑面 | M-R0b 已给出 7 个步骤级机会 / 3 条谱系 / 2 个身份(按当时状态清点),原 Scope 路径**有值得验收的历史候选** → **暂不启动 M-W**。ACC-1 首选 `forward · A5-online · outlier_mad · k1`,资格待 Opus 补表(修复后去重可行性 / 新 Scope 后续覆盖 / 验证→重遇时序);本次只申请**第一层**(真实 Slow → 校准 → preflight → replay → 写入既有 Draft),第二层(历史前向重放的验证→重遇)预注册为可选。结果类别与停止条件预写(F.4);成功或失败都先收口本层再裁定;`ACCEPTED_INTO_DRAFT` 不自动授权后续;Slow 未提出不构成换编辑面的依据 | 实验,待用户批准 LLM ≤2 与 fits(待补) | M0-a、TSFM 探针、NOAA 适配器、AD 扩展、Wind Farms 下载 |
| ⑤ 闭环 → 跨数据集 | 同前 §4–§6 | — | — |

**暂不启动长课程、不扩新方法、M-W 暂不发车**;delayed 证据是否进普查为**协议事项**(§4.3),不混入 ①。

### 7.1 前 10 个工作日

| 日 | 任务 | 负责人 | 输入 | 交付 | 前置 | 可并行 |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | 本计划 + M r2 口径修订呈 sol;数据资产审计任务书(D2/D3 已在候选池)发出 | Fable → sol / 用户 | r2、本计划 | sol 冻结件;审计任务书 | — | 审计与 M-R0 同日起 |
| D1–D2 | **数据资产审计**(只读):曝光台账、serving 适配器、Yahoo 41 / NAB 4 族状态、"有缺口的预测 Target"候选清单 | grok(只读) | `data/`、曝光工件、正典 §4 | 角色表(开发 / Source / Target held-in / 密封) | — | 与 M-R0 并行 |
| D1–D2 | **M-R0 修订可达性审计**(0 fit) | Opus | 三顺序 course / checkpoint 工件 | 谱系表 + REVISE 失败原因 + 0 `NARROW` 归因 | — | — |
| D2 | LLM 计费勘误工件(既定,0 成本) | Opus | `run_hec1.py:1970` | 勘误 JSON | — | — |
| D3–D5 | **M-W Workflow 可修复性**(pooled;若 M5 已批则同时出 per-channel 列) | Opus | HEC-1 失败案例、bank、冻结菜单 | 修复率 / 过线率 / 增益 / 退化表 | M-R0 案例清单 | 与 M1 并行 |
| D3–D5 | **M1 持续性 + M3(Ridge)+ M4 估计器侧** | Opus(grok 复核基线定义) | 逐序列增益台账;KDD 副本 | 可学习性表;Pattern 增量;Gate 0 复原结果 | — | — |
| D3–D6 | **M5 Stage A**(需 fits 授权) | Opus | HEC-1 已记录动作 | `GAIN_*` × `SHAPES/NO_SHAPE` | 用户批 ≤3000 fits | 与上两项并行 |
| D6–D7 | 补充组:M0-a(Yahoo 设置比较)、M0-b(NOAA)、M2;TSFM 探针仅在下载批后 | Opus | 现有程序 / Consumer | 前提表(匹配格) | sol 边界确认;NOAA 适配器 | 不阻塞核心 |
| D8 | 测量读数 → Gate 0 / Gate 1 → 决策表 → **编辑面与 Consumer 选定** | Fable(读数)/ sol(裁定) | 上述工件 | 选路记录 | 核心组齐 | — |
| D9–D10 | 三页最小方法说明(唯一编辑面、谱系、`SUSPENDED`、影子对照、预算)→ sol 冻结 → 接线任务书 | Fable → Opus | 选路记录 | 方法说明;接线任务书;M4 环路验收规格 | D8 | 写作可起草 §1–§2 |

### 7.2 预算(按 单元 × 臂 × 候选/尝试上限 展开;**草案数字,未批**)

**计费口径先说清(只读核实)**:`course_fits` 只在 delayed 面(`run_hec1.py:2032`)与评价面(`:2186`)累计,**不含 Support 探针**;一次"评分面" = `FITS_PER_SCORED_FACE = 3`(Static 参照 + raw + program),经缓存路径 = `CACHE_FITS_PER_CELL = 2`;一次评分**不等于**一次拟合。因此下表按**调用类别**分列:探针(Support)/ 筛选(replay)/ 验证(delayed)/ 后续评分(评价面、事后分层)/ 祖先影子;缓存节省另列,不并入总额。

**M 包(0 LLM)**:M-R0 / M1 / M3 / M2 = 0 fit;**M-W 见附录 D(自然 Tier-1 ≈ 1000–1400 fits,Tier-2 待 M-R0 给出案例数后补报)**——原"300–540"低估,已撤回;M4 ≈ 7 程序 × 26 窗 × 2 面 × 3 ≈ **1100 fits**(缓存后 ≈ 750);M5 ≤ **3000 fits**(既呈);M0-a/b 各 ≈ 7 程序 × 可评窗 × 2 面 × 3(**待审计给出窗数后报数**);M0-c ≤ **4000 次 TSFM 推理**(下载未批)。

**最小闭环(KDD 26 单元 × 2 顺序 × 3 臂 + Static)**:

| 项 | 现行口径(5 call/单元臂) | 语义预算口径(≥1 完整决策、物理顶 10;待批) |
| --- | --- | --- |
| 物理 LLM 请求 / 顺序 | Fast 26 × 3 × 5 = 390;Slow(仅 B)5 步 × 4 = 20 → **≈410** | Fast 最坏 26 × 3 × 10 = 780;Slow 20 → **≤800** |
| 两顺序合计 | ≈820 | ≤1600 |
| 完整决策数 | 实测报告(HEC-1:每格 ≈1) | 目标每格 ≥1,实测报告 |
| Consumer fits(pooled Ridge)/ 顺序,按类别 | **探针**(Support,每完整决策 1 次评分 ≈ 3 fits;HEC-1 未入 `course_fits`,须由计费核对补出实测)≈ 3 臂 × 26 × 3 ≈ 230;**验证**(delayed)+ **后续评分**(评价面)≈ 3 臂 × 26 × 2 面 × 3 ≈ 470(HEC-1 实测 4 臂 `course_fits` 173–186,系缓存命中后的数字);**祖先影子** ≤ 修订谱系数 × 剩余单元 × 2 面 × 1 ≈ ≤100;**筛选**(replay)≤ 各臂课程 fits 的 100%(`REPLAY_FITS_SHARE = 1.0`)→ 合计 **≤ ~1300 名义 / ≈ 700 缓存后**;缓存节省单列不抵扣审批额 | 同 |
| Consumer fits(per-channel Ridge)/ 顺序 | 上表 × 被服务序列数(≈20)→ **≤ ~20000**;须按实际逐序列拟合计数 | 同 |
| 模型推理 | 0(除非 Consumer 选 TSFM) | — |
| 缓存与重放 | 沿用 HEC-1 每臂重放缓存(键:臂 × 格 × 面 × Consumer 配置 × 程序)与未来步预留 | 同 |

**已有授权**:HEC-1 三顺序信封(已用完,不可复用);0-LLM 只读分析无预算限制。**待批**:M5 ≤3000 fits(既呈)、M-W + M4 + M0 ≈ 1200–1500 fits、TSFM 下载与 ≤4000 推理、闭环 LLM 信封(≈820 或 ≤1600)、闭环 fits(pooled ≤ ~2000 两顺序;per-channel 另议)。

### 7.3 闭环成立后的条件式路线

| 条件 | 下一步 |
| --- | --- |
| ≥1 自然 Chain-S + 顺序稳健 | 冻结方法配置 → 跨数据集主实验:同任务新 Target(有缺口的预测集,待审计定)+ Solar(弃疗/无伤害)+ AD 密封面;A5 / A3 / Static + 关键消融(B vs C、编辑面开/关);积累与结构进化分开测 |
| Chain-D 有、Chain-S 无 | 报"候选可修复、持续修订未激发";检查冲突阈与 `SUSPENDED` 时序,**只允许一次**预注册的时序修订再跑,不改编辑面 |
| `NO_LEGAL_EDIT` 主导 | 编辑面切换为 Scope 子句(若 M3 `INFORMATIVE`)或判当前 DSL 无修订空间 |
| `NO_REENCOUNTER_GAIN` 主导 | 修订能过门但无后续价值 → 跨窗可迁移性是绑定约束,进入 Track B 叙事,不再加配置 |

---

## 8. 只列三件事

**可立即安排(0 新实验成本)**:① 最小接线修复五项(Opus;新 commit;§7.0 ①);② 非作者复核(grok,只看 diff);③ 原路径可达性重放(Opus,0 LLM;影子 stump 搜索 0 fit,可行者过 replay 屏的 fits 另计、预计 ≤ 数百);M1 / M3(Ridge)/ M2 的 0-fit 估计并行。

**真正需要裁定(sol)**:① HEC-1 解释追加"实现限制"段(判词与冻结数字不变;措辞:"实际运行系统未达预注册效果,其中 ADD/NARROW 的实现缺陷限制了对完整进化机制的检验");② **delayed 证据是否进入后续普查**(协议事项,方案 a / b,§4.3);③ **R1-a 证据单元与精确去重**(仪器规则,含"不一致单元不计并单列",astra 已认可方向;sol 备案)与 **R1-b 聚合/投票策略**(学习策略,**未批**,与 delayed 证据用途一并裁);**R2 已确认、交 Opus 实施**(§4.4 实施要点);④ **R3 共享 baseline 计费归属**(补修已"计数并分列、定价留协议",sol 定归属口径);⑤ 冻结臂按单元清空 Draft 视为设计行为、不列缺陷;⑥ 三臂定义(A 含局部适应);⑦ M0-a 匹配格是否在正典 §4 边界内;⑧ HEC-2 live 草案暂缓的正式确认。**撤回**:`SUSPENDED` 立项请求;外环 2→4 数字请求(成本未测得)。**分工(astra)**:下一棒 Opus 修 D1–D5;Fable / sol 并行定 R1 / R2 / R3;不再派第三人重复审计。

**真正需要新增授权(用户;本轮均可后置)**:M-W(附录 D,必做 **1056–1188** fits,可选后验穷举 +1056)——**待 §7.0 ③ 结果后再呈**;M5 ≤3000 fits(既呈,可并行);M4 ≈ 1100 fits(闭环接线前);Wind Farms(含缺失)下载;NOAA → HEC-1 serving 适配器编写;TSFM 下载 + ≤4000 推理;闭环 LLM 信封与 fits(编辑面冻结时另呈)。

---

## 附录 A · 核心测量字段表

| 字段 | M-R0 修订可达性 | M-W Workflow 可修复性 | M1 持续性 | M3 Pattern 增量 | M4 种植对照 | M5 Stage A |
| --- | --- | --- | --- | --- | --- | --- |
| 研究问题 | 0 `NARROW` / REVISE 全败 / Active 冲突去向 | 失败后是否存在一步合法修改在后续面改善并过线 | 哪个粒度的历史预测未来 | Pattern 是否有预测增量 | 估计器能否学回已知规则 | 换 Consumer 收益是否保留 |
| 数据与曝光 | HEC-1 三顺序工件(曝光,development) | 同左 + bank | 同左(逐序列增益) | 同左 | KDD development 副本 | HEC-1 已记录动作 |
| 固定候选 | 现有谱系 | 冻结菜单一步修改集(去重) | BSG / validation-search 冻结候选 | 同左 | 注入程序 P × 层 S | HEC-1 实际动作 |
| 作用表面 | 生命周期状态机 | Workflow 组合 | — | Scope 词表 | 层 × 程序 | Consumer |
| 对照 | — | 原程序;identity | B0 多数符号 / B1 程序级均值 / B2 置换 | 程序级均值 | 未注入副本 | pooled 同动作 |
| 可读取反馈 | Support / delayed / 评价面(已记录) | 后续 ≥2 单元的三面 | 下一窗 / +48 / +144 | 同 M1 | 同 M1 | Support / delayed / +144 |
| 统计单位 | 谱系 | 失败案例(Draft 与 Active 分层) | 窗口(块 bootstrap) | 窗口 | 窗口 | 单元 |
| 主要指标 | 可达 / 提出 / 失败原因计数 | 修复率、四线通过率、Δ(修改 − 原)、非目标退化率 | 增量准确率 / AUC / Brier 与 90% 下界 | AUC 增量与下界 | 复原成功与否 | `GAIN_*`、`SHAPES/NO_SHAPE`、10/34 复算 |
| 输出 | `m_r0_reachability.{json,md}` | `m_w_repairability.*` | `m1_persistence.*` | `m3_pattern_increment.*` | `m4_planted_control.*` | 既定 Stage A 工件 |
| 成本 | 0 fit / 0 LLM(已完成) | **1056–1188 fits 必做**(11 组合 × E=10;可选后验穷举 +1056;附录 D.5) | 0 fit | 0 fit | ≈1100 fits(缓存后 ≈750) | ≤3000 fits(按逐序列拟合计) |
| 依赖 | — | §7.0 ③ 原路径可达性结果;U-4 两基线是否数值相同 | M5(per-channel 列) | M5 | — | 用户授权 |
| 停止条件 | 已止 | 11 个组合评完即止(唯一充分性规则) | Gate 0 | Gate 0 | 复原判定 | 既定 |

## 附录 B · 预算与常量出处(只读核实)

`hec1_contract.py`:`PER_UNIT_ARM_BUDGET.llm_calls = 5`(:794)、`OUTER_LLM_PER_STEP = 2`(:803)、`LLM_CAPS` 500/顺序(:825-830)、`REPLAY_FITS_SHARE = 1.0`(:811)、`PROGRAM_SPACE.compositions = "length <= 2"`(:573-588)、`RISK`(:590-603)、`SCOPE_CLASS` 12 词表 ≤2 子句(:605-618)。`outer_loop.py`:`MIN_POSITIVE_UNITS_FOR_ADD = 1`(:72)、`MIN_ADVERSE_UNITS_FOR_NARROWING = 2`(:76)、`OuterBudget.retries_per_candidate = 2`(:91)、`propose_candidates` 三类候选与 FLAGGED 漂移信号(:383-530)。`restricted_draft.py`:`MAX_REVISIONS = 2`(:86)、`classify_failure`(:167-224)。`run_hec1.py`:失败后 Draft 路径(:2121-2139)、`held_lineage_keys = active_lineage_keys`(:1806)。三顺序 course 工件 `hec1_course_v11p0_{forward,reverse,interleaved}_live.json`:`ledgers.llm_fast/llm_outer` = 237/4、203/2、220/0;`course_fits` = 183 / 173 / 186;`outer_steps` 各 10。

## 附录 C · r2 口径修正(astra 评审;本计划采用,r2 文本同步修订)

1. **FLAGGED 分支被阻断 ≠ HEC-1 修订在所有状态下构造性不可能**:`NARROW`(Active 且 ≥2 adverse 单元)与 `REVISE`(`REVISABLE` Draft)两条入口存在;HEC-1 中 `NARROW` 0 次、`REVISE` 3 次全败(1 弃权、2 预算耗尽)。准确表述:**主导失败状态没有对应的结构修订入口;Workflow 是有既往证据支持的优先候选**。
2. **同数据 `Forecast/Ridge` vs `AD/IForest`** 同时改变 Consumer 与评价目标,称"**同数据的任务—Consumer 设置比较**",不称 Task 匹配。
3. **D5 的 0.255 / 0.730 / 0.016** 是**严重伤害样本**中的代数分解份额,不能读成 context 通道贡献了总体收益的 25.5%;context-only 面的收益只能由 D5 四格的逐序列 context 分量另行汇总。
4. **Gate 0 / M1** 口径改为可执行数字(§3);决策表先门后分支,成立行全部报告。
5. **臂数**:r2 §2c 列了 6 项却在 §6 称"五臂";本计划闭环阶段为 **3 个实验臂 + Static 参照**;扩展阶段的臂表在冻结时重新列举并计数。
6. **"treatment 密度构造性提高"** 撤回;分歧数、完整决策数、修订可达性为实测结果。
7. **"A5 treatment 漏斗为空"** 撤回(与 K0 非空矛盾);改为"知识已进入并参与使用,结构修订未完成,持续修订收益未得支持"。
8. **Solar** 改"用途待核实",不因 0 缺失预判增益迁移不可能。
9. **M-W 预算** 原"300–540"撤回;按调用类别与实际案例数重算(附录 D)。
10. **(M-R0 后)"11 张 Draft 无一来自曾 Active 的 Skill"** 撤回:8/11 由持权卡的 delayed 失败产生(6 张直接来自 K0);"K0 丢失"撤回(部署权从未丢失)。
11. **(M-R0 后)"撤销先于外环步"** 假设不成立;`SUSPENDED` 暂不立项;冻结臂丢弃 15 张 Draft 系设计行为。
12. **(M-R0 后)"修订可达性 ≈ 0 是构造性的"** 收窄为:首因是普查 relation 词表接线(`ADD`/`NARROW` 同处阻断),其次 `FLAGGED` 无入口、delayed 不入 bank(协议事项)、阈值工具 0/5 校准;"FLAGGED 主导失败模式无匹配面"仍成立但已不是首因。
13. **(M-R0 后)失败情境** 4 → 8(按 cohort × origin × delayed × Program × Scope);"≤12 实例" → 20 组合 / 13 可修 / 11 有 ≥2 后续单元;`outlier_mad 10 / outlier_iqr 1` 是 Draft 创建程序计数,不是情境计数。
14. **(M-R0 后)"原程序对照多数已记录 → 0"** 不成立(22 个所需后续面仅 3 个已有),全额计入。
15. **(M-R0 后)Tier-1 / Tier-2** 不是两批案例;Tier-2 唯一额外成本 = 祖先影子对照。
16. **(M-R0 后)"与 live 路径同"与"3 fits/面"** 是两种仪器(`ScopeExecutor.evaluate` vs `_policy_reading`),M-W 冻结前必须定死一种并核 U-4(两基线是否数值相同)。

## 附录 D · M-W 小规模执行清单与完整预算(**暂不发车**;按 M-R0 §2 更新;待 §7.0 ③ 结果后再呈批)

### D.1 自然案例(按完整身份去重;M-R0 §2.1,只读核实)

按 (cohort × origin × delayed × Program × Scope) 去重 = **8 个失败情境**,41 个实例(33 个为同一确定性评估按臂/顺序重算的重复观测,不是新证据);全部 Scope 为 `serving_series_predicate[local_robust_z_peak >= 3.0]`。**7/8 情境同时是 Active 卡冲突(Tier-2)**,Tier-1 / Tier-2 是同一批窗口。

| # | cohort | origin → delayed | Program | 当时可见失败线 | treated | 实例 | 顺序 | Workflow 编辑能否动它 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-a | `[0:40]` | 1896 → 1944 | `outlier_mad` | hf + ssh | 13 | 7 | F/R/I | 能 |
| S-b | `[120:160]` | 1176 → 1224 | `outlier_mad` | ssh(最坏 0.485) | 15 | 9 | F/R/I | 能 |
| S-c | `[40:80]` | 2136 → 2184 | `outlier_mad` | coverage_floor | 3 | 5 | F/R/I | **不能**(Scope 不改 ⇒ 覆盖不变) |
| S-d | `[40:80]` | 2136 → 2184 | `winsorize` | coverage_floor | 3 | 2 | F/R | **不能** |
| S-e | `[40:80]` | 2616 → 2664 | `outlier_mad` | hf | 12 | 6 | F/R/I | 能 |
| S-f | `[40:80]` | 2616 → 2664 | `outlier_iqr` | hf + ssh | 12 | 1 | I | 能(唯一非 Tier-2) |
| S-g | `[80:120]` | 1176 → 1224 | `outlier_mad` | coverage_floor | 3 | 5 | F/I | **不能** |
| S-h | `[80:120]` | 2376 → 2424 | `outlier_mad` | ssh | 12 | 6 | F/R/I | 能 |

(情境 × 顺序)= 20 → 剔除只差 `coverage_floor` 的 7 个(`NOT_REPAIRABLE_BY_WORKFLOW`,预先记录、不买)→ 13 → 要求后续 ≥2 可评单元(剔 reverse S-a p24、forward S-b p23)→ **11 个组合 = 分母**;逐格清单在 `m_r0_reachability.json` 的 `situations[*].per_ordering`。
**Tier-3**(validation-search 的 24 个 +144 失败)只作事后分层,不进主表。

### D.2 有限合法修改集(对 1 步 `outlier_mad` / `outlier_iqr` / `winsorize`;拟合前上界)

| 类别 | 候选 | 数量 |
| --- | --- | --- |
| 同类替换 | 其余三个 outlier 算子(`winsorize` / `outlier_iqr` / `outlier_mad` 中非原算子的两个 + `hampel_filter` 默认参数) | 3 |
| 增加前置 impute 步 | `impute_linear(strength=1.0)` / `impute_ema` / `impute_ar` / `impute_ssm` / `impute_fft` / `period_complete` / `period_median_complete(period=24, cycles=3, min_donors=2)` → 各接原 outlier 算子 | 7 |
| 顺序交换 | 1 步程序不适用 | 0 |
| 参数改动 | 观察到的根算子无一是 `hampel_filter`,参数集不展开 | 0 |

**E = 10 为拟合前唯一可用的界**;效果去重(逐序列增益向量)只能在评过之后做,不得用预估去重数(原 E=8)计价。

### D.3 预写选择规则(只用当时可见反馈)

在冲突单元 origin 的 **Support 面与 delayed 面**(两面在冲突被观察到时均已可见)上评每个候选:① 两面均过四线;② 取 aggregate gain(两面均值)最高;③ 并列取改动点数更少者。选 **top-1 与 top-2**(两档分开报)。若无候选过 ①,记 `RULE_FOUND_NO_FIX`。**+144 面不参与选择**。

### D.4 验证与读数

对 11 个组合:在该顺序中冲突单元**之后**的前 2 个可评单元,评 top-1 / top-2 与**原程序**的 Support + delayed 面;读:通过四线否、Δ(修改 − 原)聚合与逐序列、非目标序列(未触发冲突的层)退化率。+144 面事后加评并单独标 post hoc。事后穷举全部 E 个候选在后续面的通过情况,用于区分 `RULE_FOUND_NO_FIX` 与 `NO_LEGAL_EDIT_IN_SET`(post hoc 标记)。
**仪器定死(D-6)**:live Support 面走 `ScopeExecutor.evaluate`(`scoped_evaluate` 2 fits + 按单元缓存的基线),课程评分走 `_policy_reading`(3 fits/面,基线用 `scoped_evaluate`);两者的 Δ 不可混比。M-W 统一用 **`_policy_reading`**(与 HEC-1 曲线同一仪器),并在开工前核 **U-4**(两种基线是否数值相同);若不同,原程序对照与候选都以 `_policy_reading` 重算,不引用 live 收据里的 Support 读数。
**原程序对照全额计入**(所需 22 个 (组合 × 后续单元) 中仅 3 个当时部署了同一程序)。

### D.5 完整拟合预算(名义值 = 每评分面 3 fits;top-2;后续 2 单元;缓存节省不是界)

| 口径 | 选择探针 | 验证 top-2 | 原程序对照 | +144 事后分层 | **必做合计** | 可选后验穷举 |
| --- | --- | --- | --- | --- | --- | --- |
| **推荐:11 组合 × E=10** | 660 | 264 | ≤132 | 132 | **1056–1188** | +1056 |
| 全部 13 可修组合 × E=10 | 780 | 312 | ≤156 | 156 | 1248–1404 | +1248 |

Tier-2 **不另计**候选扫描(同一批窗口);唯一额外项 = 祖先影子对照(闭环阶段才发生,M-W 不含)。0 LLM;不改阈值;数据全为已曝光 development;不读 held-out。**7 个 coverage-only 组合不买**(否则浪费 ≈420 次选择探针)。

### D.6 停止与解释

11 个组合评完即止(唯一充分性规则);逐组合报,不做跨情境平均;`RULE_FOUND_NO_FIX` 说明该提议规则未找到修法,不说明编辑空间无修法;`NO_LEGAL_EDIT_IN_SET` 只能由 post hoc 穷举给出并如此标记。任一情境在 ≥2 条顺序的后续单元上"top-1 过四线且 Δ > 0 且非目标退化 = 0" → 记 `REPAIR_SIGNAL`。**发车条件**:§7.0 ③(原路径可达性)出结果之后,且用户批准。

## 附录 F · 最小真实验收 ACC-1 任务书(草案,2026-09-06;候选与预算待 Opus 补表后填定;**批准前不运行任何真实 LLM / fit**)

### F.0 性质与边界

- 这是**实验**,不是新一轮全面审计;不重新设计方法,不启动长课程,不开 M-W,不开 V2/V3,不开 Phase F。
- 对象是**原 Scope 子句修订路径**在修复后接线上的一次真实走通;不改生产代码、冻结合同、阈值、历史判词;不提交、不新增 SHA。
- 输入:`m_r0b_revision_opportunity.{json,md}`(7 个步骤级机会 / 3 条去重谱系 / 2 个身份;历史清点按各步当时状态)+ Opus 在补的三列:**修复后去重行集上的影子可行性、新 Scope 在后续单元的覆盖、验证 → 重遇时序**。
- **口径纪律**:7 个历史步骤级机会 ≠ 7 个修复后完整闭环机会;M-R0b 的"实测重遇 4 / 3"是**旧程序按旧 Scope 的部署**,不能替代新 Scope 的后续覆盖(该列待补,当前 `UNKNOWN`)。

### F.1 验收对象与选点规则(预写;不按真实收益换点)

**首选**:`forward · A5-online · outlier_mad@local_robust_z_peak>=3 · k1`(边界 p04)。理由:较早的边界(后续可评单元 18,三条谱系中最多);当时 Draft 为 `REVISABLE`(attempts 1)、有 Active 祖先(K0)、生命周期可接收路径 = REVISE 既有 Draft(现有分支,不依赖 R2 归并);as-run 真的到过 Slow(`SLOW_ABSTAINED`),是唯一"机制触及过、止于 Slow 而非生命周期"的边界。
**资格(四条同时成立,由 Opus 补表判定,主线不猜)**:(i) 边界当时 Active 祖先存在;(ii) 修复后规则下生命周期可接收(REVISABLE ∧ `may_add_clause()` ∧ `verified_since_revision`);(iii) **修复后去重行集**上影子可行子句 ≥ 1(影子只用于选点与对照,不给 Slow);(iv) 边界之后 ≥ 2 个可评单元,且**新 Scope**(影子可行子句所定义的谓词族)在其中 ≥ 1 个单元解析出 ≥ `MIN_TREATED = 5` 条序列(待补;为 `UNKNOWN` 则资格记 `PENDING`,不发车)。
**同谱系不算独立复现**:k2(p09)是同一谱系的后续状态,只作 k1 之后的**顺序内后续**读数,不作第二个验收点;**不得把"假设 k1 修订成功后的状态"接到历史 k2 的旧状态上**——若走到 F.2 第二层,后续单元必须从 k1 修订后的状态**向前重放**,历史 k2 记录不复用。
**备选顺位(首选不合格时依次,只看资格不看收益)**:② `reverse · A5-online · winsorize@z>=3 · k2`(p09;NARROW 开第一张 Draft,现有路径;后续 14 单元;实测重遇 0 → (iv) 待补);③ `reverse · A3-online · winsorize@z>=3 · k5`(p24;后续仅 1 单元 → 只能做第一层)。三者皆不合格 → `NO_ELIGIBLE_BOUNDARY`,停止并回报,不扩菜单、不放宽资格。

### F.2 本次验收范围(两层分开;本次申请到第一层,第二层为预注册可选项)

| 层 | 内容 | 成功的含义 | 本次 |
| --- | --- | --- | --- |
| **第一层 · 最小真实接线验收** | 在 k1 边界重建当时状态(该臂检查点:bank、Draft ledger、`active_lineage_keys`)→ 真实 Slow 提出 (feature, direction) → 阈值工具校准 → `validate_narrowing` preflight → replay 屏 → 生命周期接收(`record_revision` 写入既有 Draft,计数保留) | **只证明"修订路径在修复后可走通到写入 Draft"**;写入 Draft **不是**执行权,**不是**进化成功 | **申请** |
| **第二层 · 后续自然修订链(历史前向重放)** | 从修订后状态出发,对边界之后的前 N 个单元按 live 路径重放 Support + delayed(新 Scope 的候选 vs 旧 Scope 祖先影子),看是否独立验证通过 → 存活 → 更晚单元重遇 | 一条"历史数据上的修订链";仍非 live 自然链,且**未测整段课程收益,不得宣称系统有效** | **预注册为可选项**;是否执行由第一层结果 + Opus 补表(iv)决定,预算另批 |

### F.3 信息边界

- Slow 只看 **k1 边界之前的合法证据**:该谱系在 bank 中的 Support 探针行(修复后去重视图)、Draft 的 `root_scope` / `current_scope` / 已记录的 delayed 失败线与归因(REVISE 候选现有字段)、冻结的 12 特征词表。
- **禁止**进入 Slow 视野:影子最佳子句(`best_stump`)、M-R0/M-R0b 的机会清点与任何答案、k2 及之后任何单元的读数、+144 评价面、本任务书。运行时存档**脱敏后的请求内容、字段清单与每个字段的边界来源**(哪一步、哪张 Draft、哪些 bank 行),供非作者复核逐项核对;**不新增任何 SHA/Hash**。
- 影子搜索照常在 Slow 提议**之后**运行,只记 `agree / disagree`,**未经批准不得变成部署答案**;`LLM_THRESHOLD_IGNORED` 纪律不变(Slow 不给数值)。
- 重建状态时核对已知仪器注意事项:resume 路径下 Slow 的 harness view 曾退回 `start_snapshot`;本次必须用**该边界时**的活动快照,并在工件里写明来源。

### F.4 预写结果类别与停止条件

第一层结果**必属其一**,分开计,不合并:`SLOW_ABSTAINED` / `NO_FEASIBLE_DIRECTION`(Slow 提了方向,阈值工具在每次尝试均 `NO_FEASIBLE_THRESHOLD`)/ `PREFLIGHT_REJECTED` / `REPLAY_REJECTED` / `LIFECYCLE_REJECTED`(在资格成立的前提下出现即视为接线缺陷,记仪器项)/ `ACCEPTED_INTO_DRAFT`;**另单列**,不得塞进 `NO_FEASIBLE_DIRECTION`:`OUTER_BUDGET_EXHAUSTED`(每步物理顶用尽而未得结论)/ `TRANSPORT_FAULT` / `TOOL_FAULT`(阈值工具或 preflight 自身异常)。第二层(若执行):`VERIFICATION_FAILED(lines)` / `VERIFIED_NO_REENCOUNTER`(新 Scope 在后续单元覆盖 < 5)/ `VERIFIED_AND_REENCOUNTERED`(并报 vs 祖先影子的 Δ 与非目标退化)。
**停止**:一个边界、一个外环步、物理 Slow 调用不超过现行 `OUTER_LLM_PER_STEP = 2`(第 3 次按构造不可达,如实记)。任何结果类别——**成功或失败——都先收口本层、回报、等待裁定**;`ACCEPTED_INTO_DRAFT` **不自动授权**第二层验证或三臂课程;**不自动增加重试、不放宽任何门、不扩菜单、不换编辑面、不换点**。真实 Slow 若提出与影子最佳子句**不同**的子句,其后续覆盖资格须按该子句自身重算,**不得借用**影子子句在补表中的覆盖资格。每次尝试的工具结论(`NO_FEASIBLE_THRESHOLD` / `SLOW_CLAUSE_UNUSABLE` / `CALIBRATED`)必须持久化(补 U-1)。
**解释纪律**:影子已证明当前 Scope 空间**存在**可行修改,因此 `SLOW_ABSTAINED` / `NO_FEASIBLE_DIRECTION` 只说明**本次提议未实现**,**不能**推断需要换成 Workflow 编辑面;有修订链但整段课程收益未测 → 只能写"修订路径可走通 / 历史前向重放通过",不能写"系统有效"或"自进化成立"。

### F.5 预算与审批(用 Opus 按真实调用路径核出的上界;未知项明列;主线不猜数)

| 项 | 第一层 | 第二层(可选) | 来源 |
| --- | --- | --- | --- |
| 物理 LLM(Slow) | ≤ 2(现行每步顶) | 0 | 合同常量 |
| Consumer fits:replay 屏 | 该臂在 k1 前已处理格 × `CACHE_FITS_PER_CELL = 2`(缓存需在重建时新建 → 建缓存成本)= **待补** | — | Opus 按真实路径核 |
| Consumer fits:阈值工具 / preflight / 影子 | 0(确定性,对 bank 行搜索) | — | 代码事实 |
| Consumer fits:后续验证(新 Scope,Support + delayed) | — | N 单元 × 2 面 × 每面拟合数(仪器定死 `_policy_reading` 3 或 `ScopeExecutor.evaluate` 2+基线;须先核 U-4)= **待补** | Opus |
| 对照成本:祖先影子(旧 Scope 同面) | — | 已部署单元有历史读数(4/18),其余 = **待补** | Opus |
| 共享 baseline | 按单元分列计数,归整场(R3) | 同 | 补修已实现计数 |
| 未知项 | 边界状态能否逐字节重建(检查点在盘:`.hec1_runs/v11p0_forward_live/checkpoints` 104 文件);修复后去重是否改变 k1 可行性;新 Scope 后续覆盖;U-1 / U-4 | | 全部 **待补** |

**须单列、不得包装成接线修复的事项**:① 若希望第 3 次重试可达,须前瞻批"每步物理顶 2 → 3"(协议/预算语义);② delayed 证据入普查(§4.3)——本验收的 k1 走 REVISE 路径不依赖它,**不在本次范围**;③ R1-b 聚合策略——不依赖,不在范围。
**未经用户明确批准,不运行任何真实 LLM / fit,不开 Phase F。**

### F.6 后续衔接

第一层无论成功或失败,**先收口、回报、由用户/sol 裁定下一步**;`ACCEPTED_INTO_DRAFT` 不自动授权第二层或三臂课程。经裁定通过后,才进入 §4.1 的三臂自然课程(A 局部适应结构冻结 / B + LLM 结构修订 / C + 同预算确定性修订),届时同时报告自然修订链、整段 B−A 收益/伤害/成本、B−C 增量。第一层失败类别只作**事实回报**,不预设推断(影子已示 Scope 空间存在可行修改,Slow 未提出不构成换编辑面的依据)。本轮**不展开**多数据集、TSFM、AD、新数据下载的任务书。

## 附录 E · 核查任务书要点

**E.0 状态**:E.1(M-R0)与 E.2(计费核对)已由 Opus 完成(`m_r0_reachability.{json,md}`,含独立计费章;脚本 `audit_m_r0_reachability.py` 复算 30/30 普查表);E.3 已由 grok 完成(`m_data_asset_audit.{json,md}`)。以下 E.1–E.3 保留为历史任务书;**新任务书 = E.4 / E.5**。

**E.4 最小接线修复(Opus;新 commit;非作者复核后方可用于任何重放)——状态以最新工作树为准**:(a) relation 词表 ✔;(b) `by_scope` 按程序 ✔;(c) `restrict` / `open_restricted` 共用 (program, root) 守卫含已关闭、`root_for_scope(program_steps=)` ✔;(d) **改为只文档化** `retries_per_candidate` 与 `outer_llm_per_step` 的关系,**不改数值、不移硬顶**;(e) 计费:`_bill_fast` ✔、执行器级 Support fits ✔、baseline 分列计数 ✔、逐请求计量 ✔。**剩余(本轮派工)**:(f) R1-a 补条——被打标的不一致单元在 `unit_count` 只计一次、`positive_units` / `adverse_units` 不计并单列 `units_with_conflicting_relations`;(g) R2 实施(§4.4 实施要点);(h) D5 反例测试(R1-a 不一致单元;R2 三正例 + 计数不变;`reserve` 剩余额度边界;except 路径 support_fits)。**不得触碰**:阈值、`_gate`、delayed 是否入 bank、Slow 提示词、候选空间、R1-b 聚合策略。
**E.5 原路径可达性重放(Opus,0 LLM / 0 fit;astra 口径)**:在修复后的接线上,用三顺序 course 工件重建 bank,**按第 k 步当时的状态**(持权谱系、既有 Draft 及其状态与计数,取自该步之前的 cell 记录,不用课末回填)重放 30 个外环步的普查与 `propose_candidates`(含 R1-a 去重、R2 的 propose 阶段处理)。输出**分层漏斗表**,每层带来源标签:候选数(实测)→ 影子 stump 可行数(`best_stump`,12 词表、冻结 bin;标"影子可行")→ 合法(根 Scope / 修订额度;实测)→ 生命周期接收(`NARROW` / 归并 REVISE / `NO_NARROW_TARGET(reason)` 分列;实测)→ **replay 通过:未评估** → **真实 Slow 提议:未评估** → **后续独立验证:未评估**。不得把"影子可行"写成"可修订"。与 M-R0 反事实 26 / 6 的差异逐项解释(真实 relation 语义、活动谱系去重、R1-a、R2)。**这决定 §7.0 ④**:生命周期可接收的候选为 0 → M-W 呈批;> 0 → 先呈批一次最小真实验收(replay 屏 + 真实 Slow,预算另批),再谈原路径最小闭环。

**E.1 M-R0 修订可达性审计(Opus;已完成)**:输入三顺序 course 工件 + `.hec1_runs` 检查点;逐 Active 谱系(K0 于 A5 臂;课内激活于 A3/A5 臂)重建:激活单元 → 每个后续单元的 delayed 读数与 relation(POSITIVE / CONFLICT / NEGATIVE)→ 部署权丢失时刻与路径(哪段代码、哪条规则)→ 每个外环步 `held_lineage_keys` 是否含其 `census_key`、`adverse_units` 计数 → `NARROW` 为何未成候选;逐 Draft 重建状态轨迹与三次 REVISE 的失败原因(弃权 / 预算 / `NO_FEASIBLE_THRESHOLD`);量出 Forward k1/k2 每次 Slow 完整提议的物理调用数。输出 `m_r0_reachability.{json,md}`;结论限"被哪个条件挡住",不给方法建议。
**E.2 计费核对(Opus)**:物理 LLM 逐次入账并与后端计数断言相等(`spend(calls=spent−1)` 勘误,耗尽格入账);Support 探针 fits 从 round 收据补出并与 `course_fits` 并列报;输出勘误工件。
**E.3 数据资产审计(grok,只读)**:逐数据集给出 曝光状态(台账依据)/ serving 适配器有无 / 角色(development / Source / Target held-in / 密封)/ 已授权元数据下各现役 Skill 必要条件是否可能满足(Solar 按 D.1 的谓词逐一判,不开数据);NAB 4 族逐族;Yahoo 41 条密封复核;列出 ≥1 个"有天然缺口的预测 Target"候选及其下载来源(不下载)。输出角色表。

