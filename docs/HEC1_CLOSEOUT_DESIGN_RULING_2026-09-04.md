# HEC-1 收口设计裁定(最终、有界;主线 Fable,2026-09-04)

授权:sol 指定"独立方法设计者对主终点可识别性问题做最后一次有限边界的设计裁定"。约束:不重跑 HEC-1、不开 Phase F、
不调风险阈值、不扩算子、不用结果挑配置、不提开放式 HEC-3;不改代码、不启动实验。本稿之后**设计阶段关闭**:sol 做一致性
裁定 → Opus 实现与测试 → 运行。标记 CODE FACT / EVIDENCE / INFERENCE / RULING。

## 0. 一页结论

1. **HEC-1 的判词不变,含义改写**:`HEC1_EVOLUTION_NOT_SUPPORTED` 是冻结机器判词;人印措辞 = "在 KDD × pooled Ridge ×
   5-call 单元预算下,完整 A5 系统未形成足够稠密的 online/frozen 行为差异(69 配对点仅 7 个不同动作,其中 5 个伴随一侧预算
   耗尽),预注册进化曲线**未被识别**;这不证明 Skill 写回或经验复用无效。" 它是一个**treatment-sparse 的低信息 null**。
2. **合同有一个结构缺陷**:INCONCLUSIVE 只看配对点数与顺序完整性,没有"行为分歧太少"的出口,所以机器只能在 SUPPORTED /
   NOT_SUPPORTED 二选一。这是主线在合同里犯的错,HEC-1 只能作事后敏感性披露;**HEC-2 起必须有前瞻的 discordance 门**。
3. **first fault 排序改为**:① 单元预算语义(5 次 agent 调用 ≈ 一次 Fast 决策;20% 格无决策、66% 贴顶——可识别性层面最上游);
   ② 跨窗口安全不可迁移(Support-safe → +144-safe 10/34;失败 22/24 撞 msh——方法层面的绑定约束);③ Fast 候选覆盖不足
   (7–8/17,部分是①的结果);④ pooled 路由放大尾部(D5);⑤ 算子(BSG 14/23 有安全正解,非首因);⑥ 记忆层空转(结果非原因)。
4. **HEC-2 改为机制优先、分级升格**:Stage A(0 LLM)在 HEC-1 已记录的动作上换 Consumer 重算,全窗口足功率地回答伤害形状 ×
   收益保留 2×2;**只有落在 SHAPES ∧ GAIN_RETAINED 才**启动 Stage B live 课程(语义预算 + discordance 门)。
5. **三件 0-LLM 前置**(顺序固定):计费勘误(物理 1088 vs 账面 666)→ `audit_hec1_endpoint_composition` → scope-matched
   `audit_hec1_gate_predictiveness`(≈700 fits)。三者落地前不冻结 HEC-2。
6. **论文**:Track B 骨架现在就立得住(治理协议跑通 + 机制证据链收敛);Track A 只剩 Dataset/Domain 主实验一条路;Phase F
   加"最多开封一次"的自由度约束。

## 1. 七项裁定表

| # | 问题 | RULING |
| --- | --- | --- |
| 1 | HEC-1 claim ceiling 与 first-fault | 判词不变;人印 = treatment-sparse、未识别;上限 = "受治理生命周期可在自然数据上 live 跑完 78 单元臂无仪器故障;写回不增加相对风险;所有自适应臂 > Static;online/frozen 对照未被识别";first-fault 顺序见 §0-3 |
| 2 | HEC-2 机制优先 vs 完整课 | **机制优先 + 分级**:Stage A 0-LLM 重算(§3.1);Stage B live 仅当 SHAPES ∧ GAIN_RETAINED(§3.3) |
| 3 | discordance 门 | **需要,前瞻冻结**:每顺序 online/frozen 动作不同的可计分单元数 ≥ max(6, ⌈0.25·N_eff⌉),否则判 `HEC_UNIDENTIFIED_TREATMENT_SPARSE`;预算介导分歧单独计数;主终点(全部配对点、对称规则)与敏感性终点(剔除预算介导)**同时预注册**(§3.2) |
| 4 | cap | **改为语义预算**:每格保证 ≥1 次完整 Fast 决策,上限 2 次完整决策,物理硬顶 10 次调用;计费 = 物理调用逐次入账并与后端计数断言相等。Consumer 因果可比性由 **Stage A(同动作、同 cap 重算)**保证;Stage B 只做实验内对照,HEC-1↔HEC-2 曲线不直接比;若用户加钱,pooled 在新 cap 下跑一条 Forward 作桥(可选) |
| 5 | gate-predictiveness 审计 | scope-matched:Support-safe 与 Support-unsafe 候选以**同一 Scope** 在 +144 重部署;统计单位 = **单元**(候选行按行为指纹去重后聚类于单元),行级 2×2 只作描述;判词 `GATE_PREDICTIVE / GATE_WEAK / GATE_UNINFORMATIVE`(§3.4) |
| 6 | 2×2 唯一后续 | 见 §3.3 表;每格一个动作,不留分支 |
| 7 | 最终路线 | HEC-2 Stage A → 三件 0-LLM → (Stage B 若够格) → 数据资产审计 → Dataset/Domain 合同(8 集 / 4 域;语义预算;discordance 门;跨窗口安全证据聚合;覆盖相对底线)→ Ridge 五臂 → 同 split 换 TSFM(仅当 Ridge 达 L1 或论文需 Consumer 泛化列)→ 一个密封数据集/域 → 写作。Phase F **最多开封一次**(§5) |

## 2. HEC-1 的准确读法(EVIDENCE,已由 Opus/Kimi/sol 三方复算)

- 主终点 D_o:+0.212 / −0.043 / +0.006;物质线 1/3;非平局 2/3/2,合计 7/69;其中 5 格一侧因 cap 退 identity。剔除后
  −0.011 / −0.060 / 0.000——**敏感性分析**,不替代正式终点(cap 对称、预先冻结;事后只留"两臂都完成"的格是按结果选样本)。
- 234 个 LLM 臂格:47 无决策(A3-online 21 / A5-frozen 9 / A5-online 17)、107 恰贴顶;一次 Fast 决策 ≈ 4–5 次调用 ⇒ 每格
  一次提议机会。"释放探针名额"无名额可释放,62/69 平局由此解释。
- 计费:`spend(calls=spent−1)`(CODE FACT `run_hec1.py:1970`)+ 耗尽格提前返回未入账 ⇒ 账面 241/205/220,物理 363/367/358
  (合计 1088 vs 666);三顺序仍 < 500,效用判词不作废;**成本结论勘误,`llm_total` 不得用于效率主张**。
- 生命周期:11 张 Draft,10 FLAGGED、1 REVISABLE 未修订;revisions = 0;P2 链不存在。**不得为救 P2 调三态机阈值**(标定属性
  ≠ 有效性)。
- P3(A5-online − A3-online):+0.369 / −0.119 / +0.180,均值 +0.143,2/3 为正——**描述性**,同属轨迹依赖,不作安慰奖。
- BSG:14/23 单元存在安全正解,累计 +5.527 ⇒ 菜单不是首因。validation-search 累计 +3.132 但 harm 9 > A5 的 5/4/4 ⇒ 不能说
  "枚举比 A5 好"。Support→+144 安全保留 10/34;失败 22/24 撞 msh、16 撞 hf、4 撞聚合。
- gate 2×2(代理口径):Support-safe → future-safe 4/34(11.8%),unsafe → 15/349(4.3%),lift 2.7×,Fisher p=.077;
  Support 侧 scoped、future 侧无 Scope,口径不一致;383 行聚类于 23 单元;单元级正/平/负 4/10/2 ⇒ **弱描述性线索**,
  待 §3.4 正式审计。
- K0 谱系须明写:Phase S-v1(ADD=2,空)→ v1.1 合法重跑(ADD=1,非空 `fast_winner_forecast_ridge_smase_outlier_mad`)→ K0 审计
  CLEAN → Phase T 三顺序用该 K0;仓内空的泛用 `hec1_k0.json` 与带标签非空 K0 并存,报告只指后者。

## 3. HEC-2 冻结建议

### 3.1 Stage A(0 LLM;先跑;全功率回答 Consumer 问题)

- 材料:HEC-1 三顺序的 scoring ledger 与 cells——每个可计分单元、每臂**已记录的部署动作**(程序 + Scope 解析集)与 Support 探针
  候选;BSG 与 validation-search 的候选集。
- 操作:同一动作、同一 Scope、同一面(Support / delayed / +144),Consumer 由 pooled 换 **per-channel Ridge**(每序列自身
  anchored 训练窗口;同超参、CONTEXT/HORIZON、anchor)。不产生新提案、不调 LLM。
- 读数:P-C1 尾部(msh 分布、严重数)、P-C2 基座(hf)、**P-C6 收益保留**(部署策略聚合增益 pc/pooled)、Support→+144 安全保留率
  pc vs pooled、BSG 安全正解单元数 pc。
- 预注册锚点(冻结前填):SHAPES = 严重(msh>0.30)失败次数下降 ≥ 50% ∧ 四线准入通过率不下降;GAIN_RETAINED = 同动作聚合增益
  pc ≥ 0.7 × pooled(比例作锚点,冻结时定)。
- 成本:≈ 同 D5 量级(每单元每动作 2 fits + per-channel 2×|S|);上限 3000 fits;0 LLM。

### 3.2 前瞻 discordance 门(HEC-2 起所有进化课程通用)

- 定义:discordant = 同一可计分单元上 online 与 frozen 的部署动作(程序或 Scope 解析集)不同。
- 门:每顺序 discordant ≥ max(6, ⌈0.25·N_eff⌉);不满足 → `HEC_UNIDENTIFIED_TREATMENT_SPARSE`(新词,既非 SUPPORTED 也非
  NOT_SUPPORTED),报 treatment funnel。
- 预算介导分歧(一侧因 cap 退 identity)单独计数;主终点保留全部配对点(对称规则),敏感性终点剔除预算介导;二者同时预注册,
  不得事后选择。
- 物质线仍 0.005 × N_eff;另报"逐 discordant 单元均差"。

### 3.3 2×2 唯一后续动作(Stage A 结果决定)

| 伤害形状 | 收益 | 唯一动作 |
| --- | --- | --- |
| SHAPES | RETAINED | per-channel 解耦有价值 → **Stage B live**(per-channel、语义预算、discordance 门;实验内 online/frozen 对照)→ Dataset/Domain 合同以 per-channel 为 Consumer |
| SHAPES | LOST | 路由是收益/伤害同一管道 → 有界负结果;**停止同域配置搜索**;Dataset/Domain 合同以 pooled 为 Consumer,两决策 ScopeSpec 只作候选;论文 Track B |
| NO_SHAPE | RETAINED | 路由非主伤害源 → 只考虑 Scope/Observation 面,且仅当论文需要;默认直接进 Dataset/Domain |
| NO_SHAPE | LOST | per-channel 路线关闭 → Dataset/Domain(pooled);Track B |

P-C5 改写为:"per-channel 预计收缩跨序列路由溢出,但不消除 raw/program 模型切换效应"(D5 对照:|route| 0.404→0.104,非零)。

### 3.4 scope-matched gate-predictiveness 审计(0 LLM,≈700 fits;HEC-2 冻结前)

- 对 23 可计分单元、Support 上探过的全部候选(按行为指纹去重),以**候选自身 Scope**在 +144 重部署;记录 Support 准入(四线)与
  +144 四线及逐序列增益。
- 统计单位 = 单元:每个同时含 safe / unsafe 候选的单元算一个"safe 类 future-safe 率 − unsafe 类 future-safe 率";符号检验
  (描述性)+ 单元聚类 bootstrap;行级 2×2 与 lift 只作描述;按 +144 失败线(msh / hf / aggregate / coverage)分层。
- 判词:`GATE_PREDICTIVE`(单元级差中位 > 0 且 ≥ 2/3 单元非负,行级 lift 聚类 CI 下界 > 1)/ `GATE_WEAK`(方向为正但不满足)
  / `GATE_UNINFORMATIVE`(中位 ≤ 0)。
- 含义(预写):UNINFORMATIVE → 该 horizon 上不存在可迁移的安全结构,任何 harness 配置改动都救不了,应动的是**下一合同的任务定义**
  (评价 horizon / 安全证据跨窗口聚合 / 序列子集);WEAK → 标定问题,多窗口聚合优先;PREDICTIVE 且失败压在 msh → per-channel
  正中要害。同口径再算一遍 per-channel(≈700 fits)作 Stage A 的一部分。

### 3.5 计费与仪器修复(Stage A 前;不重跑 HEC-1)

- `spend(calls=spent−1)` 与耗尽格未入账两处修复;测试:物理调用 = 账本 = 后端计数(每格、每顺序);HEC-1 成本以勘误工件重报
  (`hec1_cost_erratum_v11p0`),不覆盖原工件。
- `audit_hec1_endpoint_composition`:7 个非平局的构成(动作、faults、预算介导标记)+ 234 格 cap 分布,正式工件。
- 以上均为 readout 附录级 0-LLM 脚本,非方法。

## 4. 论文 claim ladder(更新)

| 级 | 需要 | 现状 |
| --- | --- | --- |
| L0 治理与安全 | 生命周期 live 跑通;harm 平价;自适应 > Static;失败机制可复算 | **已有**(HEC-1 三顺序 + D1/D5/D6 + 三诊断) |
| L1 Skill-library evolution | P1 在**可识别**设计下成立 | 未识别(HEC-1);待 Stage B 或 Dataset/Domain |
| L2 Skill-and-Scope evolution | L1 + P2 | 0 修订;Track B 倾向 |
| L3 cross-cohort accumulation | 非空 K0 ∧ A5−A3 ∧ held-out 保持 | K0 首次非空;P3 描述性 +0.143;未考 |
| L4 cross-domain | Dataset/Domain 主实验 | 未开 |

**Track B 叙事(现在即成立)**:"一套预注册、信息墙、仪器/科学分离的协议让受治理的自进化 Harness 在自然数据上完整跑完进化
课程;绑定约束是自然缺口的跨窗口安全不可迁移(10/34;22/24 撞尾部),在 pooled 路由下收益与伤害同管(D5);部署期无安全预测子
(D1);在 5-call 预算下 treatment 稀疏使写回价值不可识别。" 这是方法学 + 机制图 + 诚实 null 的组合,立得住。

## 5. 停止规则与自由度约束

- **Phase F 最多开封一次**:由第一个凭自身预注册拿到 SUPPORTED 的配置进入;若到 Dataset/Domain 实验仍无人够格,本线以有界
  负结果关闭。
- **不做的事**:重跑 HEC-1;调三态机/风险/覆盖阈值;扩算子菜单;按结果挑顺序或配置;在 Stage A 落 GAIN_LOST 后再开同域配置试验;
  开放式 HEC-3。
- **"完成"的定义**:计费勘误 → endpoint composition → gate audit → HEC-2 Stage A → (Stage B 若够格)→ Dataset/Domain 合同与
  运行 → (TSFM 若够格)→ 一次密封 → 写作。此后进入写作,不再开新面。

## 6. 呈 sol 一致性裁定的三处 + 用户授权

- sol:(a) discordance 门阈值 max(6, ⌈0.25·N_eff⌉)与新词 `HEC_UNIDENTIFIED_TREATMENT_SPARSE`;(b) 语义预算(≥1 完整决策、
  ≤2、物理顶 10)与"HEC-1↔HEC-2 不直接比"的声明;(c) Stage A 的 SHAPES / GAIN_RETAINED 数值锚点(50% / 0.7)。
- 用户:三件 0-LLM 前置的 fits(≈ 700 + 700 + Stage A ≤ 3000);Stage B 是否预留 live 预算(仅在 SHAPES ∧ RETAINED 时用);
  Dataset/Domain 数据资产审计的启动。

设计阶段至此关闭。
