# HEC-1 冻结件骨架(D3;待 D2 填数、sol 核后冻结)

日期:2026-09-03。地位:主线起草的**骨架**。标记说明:`[D2]` = 由 `p4ac_hec1_course_supply` 填入;
`[sol]` = 主线拟值,sol 核冻结件时确认或改;`[user]` = 用户执行授权。无标记 = 已裁定,不再议。
冻结形态:`evaluation/main_protocol_p4/hec1_contract.py`(照 `main_experiment_contract_v3.py` 的
`assert_frozen()` 模式)+ `artifacts/main_protocol/hec1_contract.json/.md`。冻结后任何字段不改;
运行中发现的问题只记录,进 HEC-2。

## 1. 身份

- `stage`: `HEC1_CONTRACT`;`version`: HEC-1;`data_version`: `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`。
- 继承:P4U-v1 几何(Target `[80:120]`、Source `[160:200]`、held-out [4056,4296,4536,4776,5016])、
  P4U-v2(RISK_GAP 路由、Skill ADD)、P4U-v3(Runtime manifest 骨架、最少排除序列数选候选、restricted
  Draft ≤2 修订)。**不改写任一**;本件即 P4U-v4 = 课程域扩展 + 双环 + 三态机 + 工具链。
- 主张与判据:`HEC_EVOLUTION_MAINLINE_PLAN` §3(四判据同时成立才主张自进化)。
- 证据等级:Phase S/T `DEVELOPMENT`(development mechanism curve);Phase F `FRESH`(F1 同族新实例)。

## 2. 数据与课程(`[D2]` 已由 `p4ac_hec1_course_supply` 填入,2026-09-03)

- 块与面:`[0:40]` `[40:80]` `[80:120]` `[120:160]` `[160:200]` 各 20/20;**`[200:239]` 39 条,切法默认
  A=20 / B=19**(可用性扫描已在此切法下通过)。**sol 条件(2026-09-03)**:须确认覆盖率与风险分母(served 数、
  `MIN_TREATED` 基数)在全部路径中动态取自当前 served 序列数;若任一处硬编码 20 → 改 **19/19(弃 `T99`)**并记明。
  D4 实现时 grep 核验,结果写入本节。
- **Phase S(13 单元)**:`[160:200]` × {1896, 2136, 2376, 2616, 2856, 3096(均 `SOURCE_V1–V3_READ`), 3576(未读)}
  + `[200:239]` × {1176, 1896, 2136, 2376, 2616, 2856}(全部未读)。已读 origin 入课标 `SOURCE_READ`,臂内记忆全新。
- **Phase T(26 单元,≤3816 保守口径成立:26 ≥ 22)**:`[0:40]` × {1176, 1896, 2136, 2376, 2616, 2856, 3576};
  `[40:80]` × {1176, 1416, 1656, 2136, 2376, 2616, 2856, 3576, 3816};`[80:120]` × {1176, 1416, 1896, 2136,
  2376, 2616, 2856};`[120:160]` × {1176, 1416, 1656}。**P4U-v4 修订项**:Target `[80:120]` held-in 由 p4u 的
  5 origin 扩为 7(新增 1176、1416,均早于 held-in、无 held-out 时间邻接)`[sol 确认]`。全部 usable 口径为 30,不采。
- 三顺序(各 26,序列见工件 `proposals.orderings`):Forward = 块序 `[0:40]→[40:80]→[80:120]→[120:160]`、
  块内 origin 升序;Reverse = 全序反转;Interleaved = 按 origin 轮转各块。
- 构成三要素自检(照录):重复模式族——同块相邻 origin 的 `z_peak>=3` 成员 Jaccard n=37、中位 0.65(0.24–0.95);
  族内异质——12 维分箱向量去重中位 15/20;**模式稀疏单元 = 0**(全部单元 `n_z_peak_ge_3` ≥ 6,中位 13.5)。
  推论:初始化谓词 `z_peak>=3` 在 KDD 上几乎不筛人,收窄全靠 Slow 子句;HEC-1 **不测**"模式缺席时沉默"
  (该项由 Epilepsy2 与 S2a clean cell 已覆盖),如实记入报告限制。
- 曝光:Phase S/T 全部读窗(含 +48 / +144)与 held-out 对集合交集 = ∅(`p4ac` 核);各臂只读部署可见 Context
  与本课程内反馈;评价面 Outcome 不进任何臂的 Episode bank。
- **Observation 缺口登记**(D1 `NO_OUTCOME_FREE_SEPARATOR`):当前 12 维可分箱词汇内无部署期可算量能分离受害
  进入者;按 `AGENTS` §6 只记录缺口、依靠 held-in 反馈与 abstain,不为此新增特征(留 HEC-3 前瞻)。

## 3. Consumer、程序空间、风险

- Consumer:`fixed:pooled-ridge-a1`(同 p4u);Ridge 超参、CONTEXT 192 / HORIZON 48、anchor 列表不变。
- 程序空间:冻结 P1 Common DSL 单算子 + **长度 ≤2 组合**;窗口校验器 `MAX_MODIFIED_FRACTION=0.35` 不变;
  算子 0 新增。普查与计数按逐序列增益向量去重(§5.1 纪律)。
- 风险与门:`bounded_risk_v1`(material 0.005 / hf ≤0.20 / msh ≤0.30)与 `MIN_TREATED=5` **全部不变**;
  失败后的处置由三态机决定(§5),门本身不动。
- Scope 类:现行 `serving_series_predicate`(部署可见特征 × {<=, >=} × 冻结分箱阈值),子句 ≤2;
  证据有界(双侧)为**候选形式**,不设默认。**Scope 工具的特征词汇 = 本仓库 12 个可分箱 numeric observables**
  (D1/D2 核实:公开卡 21 键 + 6 个 X1 结构描述符中只有 12 个有冻结数值分箱;名单以 `p4ac` `numeric_features`
  为准),不扩。
- **门的权威**(D1 规格矛盾 #2 引出):Source-v3 工件 round 2856 出现 `delayed_gate.passes=False`(覆盖底线)与
  online_loop `delayed_event` approved 并存——两道门口径不同。HEC-1 以 P4 `_gate`(含 coverage_floor)为**唯一
  权威**;online_loop 的批准不得单独触发 `activate_approved`;runner 记录 `gate_disagreement` 并断言 Active 集合
  只由权威门产生(D4 W4 验收项)。

## 4. 臂

| 臂 | 起点 | 课程内写回 | 外环 | LLM |
| --- | --- | --- | --- | --- |
| Static | 无 Harness(raw model 全部序列) | — | — | 0 |
| A5-frozen | K0 | **禁止**(每单元后重置到 K0) | 否 | 有 |
| A5-online | K0 | 允许 | 是 | 有 |
| A3-online | h0 | 允许 | 是 | 有 |

- **K0 为空时**(Phase S 0 存活):A5-online 与 A3-online 等价,**不跑 A5-online**;臂集 = Static /
  `A3-frozen`(= 上表 A5-frozen 在空 K0 下的形态,从 h0 起、逐单元适应、不带走)/ A3-online;判据 3 不评分。
  命名 `A3-frozen` `[sol]`。
- K0 为空时是否允许既有经审计卡(如 `s2a_forecast_supply_v0`)作 K0:**默认严格取空** `[sol]`。
- 对照(0 LLM,事后离线):Best-Safe-Global baseline(§7);ScopeFit-only shadow(§6)。
- 各臂同数据、同反馈面、同预算;A5 无预算例外。

## 5. 单元协议与生命周期

- 面:Support-A(origin o)/ delayed(o+48,门)/ **评价面(o+144,只计分、不回流、不进 bank)** `[sol]`。
  +144 与下一 origin(o+240)的 Support context 重叠部分为可见数据,合法;其 Outcome 只进计分账本。
- 单元内:Context(面 A 20 条 22 维特征)→ 检索(K 中 Scope 匹配者以 `requires_target_support` 供给;
  重遇门下的 Fast-only 路径与此分列)→ Fast ≤2 提案(程序 ≤2 步;Runtime 按缺陷存在初始化 serving_scope)
  → Support-A 探针 → bounded_risk 准入 → delayed → Active / restricted。
- **内环即时 Slow:关闭**;全部 Slow 归外环 `[sol]`。
- **外环**(online 臂,每 k=5 单元 `[sol]`):确定性普查(`task_consumer_key × 行为指纹`;ADD / 修订 /
  收窄撤销三类候选)→ Slow ≤2 次/步(§6 工具链)→ replay 筛选(仅本臂本顺序已处理 cell;任一违反风险线或
  聚合 ≤ material 即淘汰;`replay_fits` 独立账本 ≤ 课程 fit 25% `[sol]`)→ restricted Draft →
  后续新单元 Support + delayed 方 Active → 两败归档。**replay 不授部署权。**
- **三态机**(任一后续验证面失败后;修订 ≤2 / 验证 ≤3):`WAITING`(仅 coverage_floor 败;谓词后续解析
  ≥ MIN_TREATED 时自动获一次验证,不耗修订;整课程未重遇 → `PATTERN_NOT_REENCOUNTERED`)/ `REVISABLE`
  (尾部/受害分数败且受害全为 NEW_ENTRANT;可加子句)/ `FLAGGED`(任一线负贡献由 CONTINUING 主导;禁收窄,
  可原样再验一次,再败 → `EFFECT_NONSTATIONARY` 进普查)。优先级 FLAGGED > REVISABLE > WAITING。
- 先验对称:风险卡与正面卡同分箱归纳;权限 `restricts_probe`(探测序末位,不硬禁);匹配语境 POSITIVE →
  风险卡撤销。开跑前 context 侧覆盖率(正/负卡)只披露。

## 6. Scope 工具链(五步,sol 定)

Slow 输出 `{feature ∈ 冻结特征词汇, direction ∈ {<=, >=}, rationale}`,**不输出阈值** → 工具在本臂 bank 内
该程序的逐序列记录上,以该特征的冻结分箱边界为候选阈值,取**过风险预算的最宽阈值**(并列取更粗分箱)
`[sol]`;无可行 → `NO_FEASIBLE_THRESHOLD`,Slow 可换 feature/direction ≤2 次或弃权 → 既有
`scope_narrowing_preflight` → replay 筛选 → 新单元验证 → 执行权。
**ScopeFit-only shadow**:同一冲突上工具自行在词汇 × 方向 × 分箱上按同一目标取最优,与 Slow 选择并列记录;
两 Scope 在后续单元按各自解析集评分(`shadow_fits` 账本,0 LLM)`[sol:shadow 形态]`。

## 7. 预算 `[user]`

- 单元臂:Support-A 轮 7、Support-B 1、probes 24、**LLM ≤5**(p4u 为 6;取 5 使 Forward ≤500 可达:
  **26 × 3 臂 × 5 + 外环 2 臂 × 5 步 × 2 = 410**)`[sol/user]`。若 K0 空(2 个 LLM 臂)则 **≈ 280**。
- 外环:≤2 LLM/步;`replay_fits` ≤ 课程 fit 25%;`shadow_fits` 单列。
- Phase S:**13 × 5 + 外环 2 步 × 2 ≈ 70**(帽 120)。
- Phase T:**Forward ≤500 LLM 硬帽**;Reverse / Interleaved 各 ≤500,分批放行,**只依仪器完整性**。
- 事后 0 LLM:Best-Safe-Global 菜单 = 冻结单算子 + `period_median_complete→outlier_*` 组合族,每单元评价面
  全局评估、过风险预算取最优、无则 identity;真 oracle(UID 级)只作离线上界。

## 8. 读数、统计、预注册

- 主图:相对 Static 的累计增益(评价面)vs 单元序号,四臂 × 三顺序;A5-online − A5-frozen 单列。
- 副图:累计 harm 事件(单元级 hf>0.20 或 msh>0.30);Best-Safe-Global advantage。
- 生命周期:铸卡 / 修订成功率 / 撤销 / 存活率 / 重遇收益 / 覆盖率(treated/served,累计与单窗口分列)。
- 三分账归因(逐单元):新卡召回 / frozen 重复推荐被拦或被害而 online 已收窄撤销 / 探针位释放。
- H1/H2/H3 每验证窗口机械落账(NEW_ENTRANT 受害占比、CONTINUING 翻号率、离开者因特征退出占比)。
- Slow vs ScopeFit 一致率与重遇读数;Fast 原始决定分类(主动弃权 / 空输出 / 格式错误)计数。
- 统计:单位 = cohort;逐单元配对差符号检验(单侧 α=0.05);三顺序 ≥2 终局差 > 0;中位数 + bootstrap CI。
- 预注册预测 P1 进化 / P2 存活链 / P3 积累(仅 K0 非空)与 first-fault 映射:照 `HEC_EVOLUTION_MAINLINE_PLAN` §5.3。
- 判词词表:`HEC1_EVOLUTION_SUPPORTED`(P1 ∧ P2)/ `HEC1_EVOLUTION_NOT_SUPPORTED`(P1 或 P2 不立,
  附 first-fault)/ `HEC1_INCONCLUSIVE`(完成单元 < 0.8·N_T 或三顺序未齐)/ `RUN_BLOCKED_NO_VERDICT`(仪器)。
  H1–H3 一致性另行标注(`CONSISTENT` / `MIXED` / `NOT_OBSERVED`)。

## 9. 停止规则与故障分类

- `UnitFault`(当前单元 identity 弃权、记录、继续):候选失败、校验器拒绝、LLM 输出多次不合法、cell 级 LLM
  预算耗尽、ServingContextDegenerate。
- `RunFault`(全场中止,`RUN_BLOCKED_NO_VERDICT`):BACKEND_UNAVAILABLE 过重试策略;G2/oracle 墙泄漏;
  全局 LLM/token/时间超帽;协议/数据错误;held-out 任何读取。
- 判词只在课末;仪器故障不得写成科学判词;工件不覆盖(新 run_label)。

## 10. Phase F

- 冻结三顺序全部末态臂(K0 空则为 Static / A3-frozen / A3-online)→ 在 `[80:120]` × held-out 五 origin 上
  **Fast-only、0 LLM**(机械召回 Active Skill、Scope 解析、部署;无探针、无写回)生成全部输出 `[sol]` →
  一次性打开全部 Outcome → 计分;context 覆盖只分层报告。开启需 `[user]` 单独授权。

## 11. 冻结程序

D2 落地 → 主线填 `[D2]` → sol 核 `[sol]` 项 → 用户批 `[user]` 项 → 写 `hec1_contract.py` + JSON/MD →
`assert_frozen()` 通过 → 0-LLM smoke 通过 → 发车。冻结后本件只允许追加"勘误"节,不改字段。
