# 预注册骨架：Code-Agent-First P1–P5（SKELETON v0，2026-07-09）

> status: **SKELETON**（P0 产物，Final_Plan_CodeAgentFirst_2026-07-09 §P0 交付件）。
> ε 与保真度阈值为占位符，P3 种子供给校准结束时填入正式值并冻结（新 SHA）；
> 本文件在 P5 confirmatory 开跑前必须冻结为正式版。占位符一律以 `___` 标记。

## 0. 通用规则（对 P1–P5 全部 binding；源 = Final_Plan §1 约束 C1–C6 与 §3 要求 R1–R6）

1. **R1 证据形态**：喂给 code agent 的证据必须是连续特征化 + 结构化 trace（失败原因具体到
   "季节相位偏移 N 步 / 修改率 x% 超 β 预算"），禁二值读数（slice v2 教训：二值必输）。
2. **R2 对照臂**：deterministic 对照臂（budgeted grid + bandit，含 F0 剂量扫描机器）必在场；
   对照臂缺席或弱化（无剂量维）的 LLM 实验一律无效。
3. **R3 harm 台账**：按面分列（router harm / program harm / gate 泄漏）；post-exec 结构检查
   任务感知（forecast 查季节/趋势，classification 查 motif/形状，anomaly 查事件保留）。
4. **R4 保真度前置**：gym proxy 保真度报告（P3）是任何进化/TTA 主张的硬前置（clf proxy
   结构性盲点前科）。
5. **R5 泄漏 lint**：EvidencePacket 与 gym 状态不得含 oracle action / L_test / held-out losses /
   final-test 资源 / raw full series / future labels；阈值只在 dev 上选。
6. **R6 统计**：LLM 臂一律 ITT（invalid/empty 输出 = no-op 计入）；confirmatory 判决统计 =
   grouped / full-refit bootstrap CI；每个 runner 输出 manifest.json + report.json +
   records.jsonl + artifact SHA。
7. **seeds 政策**：dev/discovery seeds 自由但须落 manifest；**P5 confirmatory seeds = 40–59，
   一次性打开**（seeds 20–39 已被 STAGE1 confirmatory 消耗，不复用）；holdout 继续封存，
   仅 P5 按注册协议访问（append-only access log）。
8. **raw 语义**：一律按 `policy/action_semantics.py` 三拆声明参照物——`v_raw_identity`
   （严格恒等）/ `v_impute_linear`（=v_none 语义）/ `v_ledger_baseline`（历史台账口径）；
   报告中 "gain_vs_raw" 必须写明参照物是哪一个。

## 1. P1 gate（接线验收，非性能主张）

- 出口判据：① S2 corpus 子集端到端跑通（DataView → EvidencePacket v2 → CodeAgentComposer →
  Compiler/Sandbox → SafetyGate → execute/fallback → EvidenceStore）；② stub 与 cached-LLM
  双后端同输入同输出（bit 级重放）；③ 对照臂（dp_abstain + SafetyGateLite）完好且同批跑；
  ④ per-surface harm ledger 出现在 records；⑤ 泄漏 lint 单测过。
- 禁止：以本阶段任何数字宣称 Memory/Composer/LLM 收益。

## 2. P2 motivation 表（论文实验 1：readiness 非普适）

- arms（每任务）：`v_raw_identity`、`v_impute_linear`、universal 固定 program（全任务同一条）、
  task-conditioned harness（现任 dp_abstain+gate 或其任务对应物）。
- metrics：ΔPerf per task（各任务判官口径按 TaskSpec.metric）+ 跨任务算子符号翻转计数。
- 通过判据：至少 2 组任务对出现符号翻转（已有：classify C1 平滑翻转 + F0 season harm；
  新增 anomaly 臂预期 winsorize/median 抹 spike → recall 崩）。
- 约束：anomaly rig 一周封顶（合成注入 + 固定检测器 + F1/AUROC 判官），rig 是判官不是研究对象。

## 3. P3 gym + 种子供给校准（原 G1 重定位）——**已执行并冻结（2026-07-09）**

- **保真度（正式冻结）**：主判据 = **within-series 排序保真**（per-series Spearman 的均值；
  pooled 只作诊断——P3 首轮实测 pooled −0.32 而 within +0.35、候选均值层完全单调对齐 =
  跨序列尺度差制造的 Simpson 反向，故 pooled 口径作废）。注册阈值 **ρ_min = 0.70**。
  近简并防护：任一列 distinct 值 < 5 的序列不计入；覆盖率 < 0.5 → insufficient_variance。
- **保真度判决（results/Stage2/P3Gym/，seed=20260709，n=60，rolling-origin K=6 proxy）**：
  - forecast：within 均值 ρ = **0.5699**（p25 0.374）→ **FAIL**（0.57 < 0.70）；
  - anomaly（deployable 面）：**insufficient_variance**（合法面=插补类，指标近常量属预期）；
  - anomaly 违约诊断组：pooled ρ = **0.8504 PASS**（proxy 在空间移动时跟踪 true，机器有牙齿）。
  - **后果（binding）**：两任务标 escalate-only——gym-proxy 只可作搜索中间信号，
    **P4 promotion validator 与 P5 identity gate 的验收一律用 true 判官**（held-out 真实下游
    delta），不得以 proxy 充当验收指标。此即 R4 硬门的设计目的（C5 clf-proxy 前科的重演被拦截）。
- **种子清单（正式冻结 = policy/seed_programs.py SEED_PROGRAMS_V1，8 条，SHA 见
  results/Stage2/P3Gym/skill_bank_v1.json）**：grammar 红线内组合（只用注册算子）——
  forecast 6 条（period/fft/ema 系插补 × {stl, median, ma, savgol} + winsor 复合两剂量）+
  anomaly 2 条（period_complete / impute_fft 单步）。原草案里的 stl_residual_winsorize 等
  不存在算子按红线剔除。全部 menu v1 不可表达（tests 守）。
- **headroom 判决**：本 substrate 上 forecast/anomaly 均为 **0.0000**（0/60 序列 seed 严格获胜）。
  机制已查明：seasonal-naive 判官只读最后两个周期 → 插补轴不可见，种子塌缩到 menu 等价物
  （seed_period_median9 ≡ f0_median_w9 逐值相同）。**此为 substrate 局限性记录，不构成对真实
  corpus 供给价值的否证**（S1 冻结证据：witness v1-STL 回池 ΔL1 +0.046，全序列判官下插补/供给
  轴可见）；真实 corpus 供给判决属 P5 identity gate 本体。
- **ε 正式注册：ε = 0.02**（规则 max(0.02, forecast headroom ci90_lo=0.0000) 触发下限）。
  本节自此冻结，不得改动。

## 4. P4 promotion 演示（机制验收，非性能主张）

- **P3 保真度判决的 binding 后果**：held-in / held-out validator 一律用 **true 判官**
  （冻结下游判官的真实 delta），gym-proxy 只可作 proposer 侧搜索信号，不得进入
  accept/reject 判据（§3 escalate-only）。
- 出口判据：≥1 个完整晋升周期——typed EditOp → PromotionGate（syntax/support）→ held-in +
  held-out validator → PolicyBundle 版本升级 → 回归重放 → rollback 演示 + rejected buffer 留痕。
- Memory M0–M3 阶梯（条件线）：risk-memory veto 先行；utility/contrast memory 晋升须
  ① 赢 static 历史学习器 ② first-unseen harm ≤ `___`（P4 开工时注册）③ in-support 显著优于
  out-support。

## 5.0 P5 冻结批注（2026-07-10，开跑前落盘；此后不得改动）

- **δ_safe = 0.05**（沿 Stage1/P4 worst-group 口径）。
- **有效新颖编辑门槛 K = 3**（计数口径：主 CA 臂 finalize 的程序满足 `is_novel_v1` ∧ 该
  episode true_delta ≥ ε=0.02）。
- **主 CA 臂 = ca_skills**（先验声明，防三臂取最优的多重性膨胀）；ca_plain / ca_skills_memory
  为次要臂只报告不判决。
- **substrate 与预算**：confirmatory slice = seeds 40–59 × n=3 = 60 episodes（从未参与任何
  调参）；task = **forecast only**（anomaly 按 §3 insufficient_variance 排除出 identity gate，
  在 P5-C 只做安全面报告）；每臂每 episode 候选预算 **B=3**（gym proxy_eval，proxy 允许作
  搜索信号，验收只认 true 判官）；CA 臂 = 3 个 nonce 采样（temperature 0.7）按 proxy 选优；
  deterministic 臂 = dev 冻结优先梯 [winsor+savgol, winsor+median9, median9]（P3 dev top-3，
  含剂量机器，R2 反稻草人）按 proxy 选优；random 臂 = 3 个 grammar 均匀采样按 proxy 选优；
  frozen 臂 = abstain（true=0）。invalid/empty LLM 输出 = 消耗候选预算的 no-op（ITT）。
- **统计**：grouped bootstrap，group = 生成 seed（20 组），B=2000，CI90；worst-group = 每
  cell 的 (主 CA 臂 − det 臂) 配对差 LCB ≥ −δ_safe。
- **P5-B 冻结**：pattern 轴 = 退化网格 cell（G_hi/lo × full/miss，E-1.1/F0 谱系的可观测结构
  轴）；domain 轴 = 数据集 config（nn5_daily/tourism_monthly/fred_md，真 Monash 12 基底信号
  × 4 preset = 48 episodes）；动作集 = {v_none, f0_median_w9, f0_median_w15, f0_median_w25,
  v_winsor_savgol}；判官 = seasonal-naive（period=系列自身周期）nRMSE vs **真实未来**
  （H_FORECAST）；recipe = 源组均值最优动作；quadrant regret vs per-series oracle；源组排除
  同 series_uid（防同基底泄漏）；假设 = diff-domain/same-pattern regret < same-domain/
  diff-pattern，配对差 grouped bootstrap（按 series_uid 分组）CI90；诚实注记 = 本轴测的是
  "退化结构 pattern"，内在结构 pattern 需更大语料（future work）。
- **P5-C 冻结**：安全收口在 **confirmatory slice** 上一次性评估（P4 晋升 bundle_v0.e1 vs v0，
  全部阈值/规则 dev 冻结）：coverage/gain/harm rate/worst-group LCB 四联；**S2 记录线的
  sealed holdout 访问显式延期**为独立注册访问（非静默跳过）。
- **API/成本**：cached DeepSeek flash（temperature 0.7，磁盘缓存，max_api_calls 帽 800），
  调用数/墙钟全披露；checkpoint/resume（A-36 教训：后台墙钟被杀可续跑，缓存使重放免费）。

## 5. P5-A code-agent identity gate（正式判决）

- 六臂：frozen/no-op、random valid ProgramSpecV1、deterministic budgeted search
  （grid + bandit，含剂量维）、code agent（no memory）、code agent + skills、
  code agent + skills + memory。
- 同 grammar（ProgramSpecV1）、同候选预算、同验证预算、同 held-out families、同成本台账。
- **LLM headline 判据（全部满足）**：
  ① held-out utility：CA 最优臂 − deterministic 臂 ≥ **ε = 0.02**（§3 已冻结）且 grouped
  bootstrap CI 不跨 0；utility 一律按 **true 判官**计（§3 后果：proxy 不得作验收指标）；
  ② worst-group LCB ≥ −δ_safe（δ_safe 沿用 Stage1 口径，正式值 P5 冻结时确认：`___`）；
  ③ 有效新颖编辑 ≥ `___` 条（`is_novel` + 通过全部 gate）；
  ④ API 调用/缓存命中/成本全披露。
- 任一不满足 → claim 分支 = "self-updating deterministic policy with LLM-optional analysis"
  （合法结局，照发）。

## 6. P5-B pattern-vs-domain 四象限迁移（论文实验 2）

- splits：same-domain/same-pattern、same-domain/diff-pattern、diff-domain/same-pattern、
  diff-domain/diff-pattern（Monash 真实域 + LODO 机器；pattern 区域按 P0 fingerprint 定义，
  split 构造在 dev 冻结）。
- 假设：diff-domain/same-pattern 的迁移 regret < same-domain/diff-pattern。
- 判据：配对差 CI 不跨 0（grouped bootstrap，按 series_uid 分组）。

## 7. P5-C safety 收口

- SafetyGate 阈值 policy 在 dev 冻结后，holdout 一次性评估；报 coverage / harm / gain /
  worst-group LCB 四联；不得回调阈值再评。
