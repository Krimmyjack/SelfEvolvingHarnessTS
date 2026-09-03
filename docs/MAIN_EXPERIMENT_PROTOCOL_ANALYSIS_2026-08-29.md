# 论文级主实验：设计分析（执行者稿，呈 sol / Planner 裁定）

日期：2026-08-29。地位：**分析稿，非裁定**。
上游：`docs/MAIN_EXPERIMENT_DESIGN_SKELETON_2026-08-28.md` v1/v1.1、`docs/STAGE3_PILOT_FREEZE_DRAFT_2026-08-29.md`、
`docs/D4_DOWNLOAD_FREEZE_2026-08-29.md`、`docs/CLS_LINE_FINAL_REPORT_2026-08-28.md`、`AGENTS.md`。
邻域事实来源：`docs/RELATED_WORK_HARNESS_EVOLUTION_2026-08-17.md`（AegisTS 源码级 / TimeClaw 源码级 / RewardHarness 论文级）。
本稿只补 v1 **未定或我判断有结构风险**的部分：外部基线、动作空间、单元生成、主终点口径、功效与统计。

---

## 0. 三个结构性判断

### 判断 1（2026-08-29 修订）：正向效用提升作主终点是**有证据支撑的**，但必须锁对比对

初稿把「有经验 vs 无经验」与「会修订 vs 不会修订」混成一个终点，判断有误。逐工件核对后：

**已做出正向效用提升的对比对 = 带经验 vs 不带经验（A5/K0 vs A3/Static）**

| 读数 | 值 | 工件 | 复现 |
|---|---|---|---|
| 带卡臂 vs 无卡：五 distinct 单元累计 regret | `+0.0850` vs `+0.7710`，gap **`+0.6860`** | `sa1_minimal_r1/r2.md`:98/179 | 两跑逐字节同 |
| 阶梯 v2 供给转化：累计 regret | `0.7710 -> 0.5583`，**`+0.2127`**，零 harm | `l1_ladder_v2_replay_r1.md`:3 | 两跑，判词 `L1_SIGNAL` |
| 逐单元（分类腿） | u1 A3 `+0.4067` -> 带卡 `−0.0667`；u2 A3 `+0.1841` -> 带卡 `−0.0286` | `sa1_minimal_r1.md` per-unit 表 | — |
| forecast held-out pooled | A5 `+0.059` vs A3 `−0.217`，差 **`+0.276`** | `t6_45_frep_b_symmetric_deploy.md` | dev 块 |

**未做出的是「在线修订」的增量（A5-adaptive vs K0-fixed）**：regret 差恰 `+0.0000`，
只省了 2 次 probe / 2 次挨拒（`sa1_minimal_r1.md`:3）；forecast r2 反事实 `−0.1206`。

结论：

```
主终点 = A5 vs {identity, 传统统一清洗, A3-reset, Static} 的 held-out downstream ΔPerf，方向为正
次级   = A5 vs K0-fixed（在线修订增量）：预期为成本侧（省 probe / 避重复挨拒），如实报
gating = harm = 0、越权 = 0、正确弃权率
```

正号的**机制理由**（不是许愿）：dev 已测 3 cohort × 2 Consumer 的 6/6 cell 都有延迟非负方案，
且冠军程序随 cohort/Consumer 改变——**一个固定清洗器不可能在所有配置上当冠军**。
ΔPerf 的正号来自「条件化选择 + 该弃权时弃权」，不是来自「清洗得更干净」。
这与 AegisTS 主表里 Clean4TSDB 在 Handwriting 上 `ΔPerf = −0.0626`（任务无关清洗会伤）同构。

选靶偏倚防线：cell roster 按**缺陷负荷分层预注册**（不按 outcome 选），全部分层都报，
identity 胜出的分层进「正确弃权」列——那是特性不是失败。

### 判断 2：四臂全是内部消融，缺外部比较物 —— 最可能被拒稿的点

Static / A3 / K0 / A5 是同一台 harness 的四个删减版。AegisTS 有 EDITOR 等外部基线 + `cross_dataset.py`；
TimeClaw 直接打三个 benchmark；RewardHarness 打 GPT-5 / GPT-4o——而本仓对 RewardHarness 的批评正是「无消融表」。
反过来，本项目的对称缺陷是**无外部基线**。

最致命的一条是**等预算搜索**：现役菜单只有 5 个算子
（`run_e2_s2a_forecast_oracle.py`:49 `MENU = (identity, outlier_iqr, outlier_mad, hampel_filter, winsorize)`）。
在 20 行 Support 面上，穷举这 5 个只需 5 次 fit——harness 的全部机制在这个预算下会被穷举完爆。

### 判断 3：动作空间必须扩到穷举不可行

`SelfEvolvingHarnessTS/operators/registry.py` 实际有 **19 个算子 / 5 族**（impute 7 / denoise 5 / outlier 4 / structural 2 / align 1），
现役菜单只用了 5 个。建议主实验菜单 = 19 算子 × {全局 / per-series 条件应用} × 至多 2 步组合 ≈ 380 条工作流（再乘 scope 变体）。
此时 20 次 fit 预算下穷举不可行，「用经验缩小搜索空间」才成为可测机制主张。

副作用与对策：菜单扩大 → 冷发现命中率下降（CLS 线实测 29%/位）→ A3 变差 → 易被指「把 baseline 做弱」。
对策 = 同时报 B2 搜索基线在 **1× / 3× / 5×** 预算下的曲线，让「本 harness ≈ 多少倍搜索预算」成为可解释读数（建议为论文主图之一）。

---

## 1. 设计目的

中心问题（沿用 v1 §1 已入典措辞）：

> 在不同 Task、Consumer 和 Pattern 下，同一 Harness 能否利用各任务自己的历史经验，
> 更快、更安全地达到正确的数据就绪决策，并根据后续 Gain/Harm 持续修正自己。

| # | 可证伪问题 | 主张 | 承重腿 |
|---|---|---|---|
| Q1 | 同数据同菜单只换 Consumer，Harness 是否自主产出不同 Workflow | C1 | 预测腿 consumer 轴 |
| Q2 | 累积经验是否降低达到安全有效决策的下游 fit 次数 | C5a/C5b | 预测腿（主承重） |
| Q3 | 无 headroom 时是否正确弃权、零 harm、零越权 | C3 | AD 腿 + 易档 cell |
| Q4 | 以上是否在密封新族复现 | C5c | Solar capstone |
| Q5 | Harness 能否对决策策略面提出并验收一次自修订 | C6 | Stage 3 pilot（次级） |

Q3 是与 AegisTS 类系统的差异化卖点：AegisTS 高层奖励里「数据问题下降」权重 0.5 高于下游性能 0.4
（`RLclean.py`:55, 1364-1373），结构上鼓励「总要清洗」；本项目以下游因果效用为唯一裁定、以正确弃权为一等读数。

---

## 2. Baseline / 臂设计

| 组 | 臂 | 内容 | 归因用途 |
|---|---|---|---|
| 外部 | **B0 identity** | 不处理 | 地板；也是「正确弃权」的正确答案 |
| 外部 | **B1a fixed-heuristic** | 固定 winsorize / 固定 impute_linear | 「总是清洗」的天真自动化 |
| 外部 | **B1b best-single-oracle** | 每 cell 事后最优单算子 | 单步上界 |
| 外部 | **B2 matched-budget search** | 同 fit 预算随机/穷举搜索，报 1×/3×/5× | **最强 no-agent 对照** |
| 外部 | **B3 one-shot LLM planner** | 一次 LLM 读 Context 出 Workflow，不 probe 不修订。**同时落盘每个候选的自声明 gain 方向**（positive/flat/negative），与实测 delayed 反馈比对 | 分离「agentic」与「self-evolving」；**并产出「LLM 直判可靠性」读数**：方向一致率 vs 随机基线（3 类 = 33%）、以及「LLM 说 positive 实测 harm」的假阳性率。这一条直接支撑目标陈述第三句 |
| 外部 | **B4 fingerprint-kNN memory** | TimeClaw 式确定性指纹检索复用历史 Workflow，无 Skill/Scope/双门 | 分离「检索历史」与「Skill 抽象 + 双门验证」 |
| 内部 | Static | 冻结 h0，无适应 | 适应本身的价值 |
| 内部 | A3-reset | 公共 h0 起，仅 Target held-in 适应 | 去掉跨域积累 |
| 内部 | K0-fixed | 带 v0 卡但永不修订 | 去掉在线修订 |
| 内部 | A5-online | 完整系统 | — |
| 上界 | Oracle | `run_e2_s2a_forecast_oracle.py` 已实装 | regret 归一化分母 |

B4 是本项目相对 TimeClaw 的核心差异点的直接对照：TimeClaw 的 A5/A3 就是 `--k-neighbors 0 vs 3`，
记忆 = append-only JSONL + ~20 维确定性指纹两段检索。不设 B4，评审无法判断五轴 Scope / 阶梯 / 双门相对「纯检索」买到了什么。

预注册对比对（6 条）：`A5−Static`、`A5−A3`、`A5−K0`、`A3−Static`、`A5−B2@1×`、`A5−B4`。

### 消融

| 消融 | 内容 | 邻域对位 |
|---|---|---|
| 无归因 | 去掉 first-fault，Skill 随机修订 | SkillAdaptor Succ 33 → 28.6 |
| 无验收门 | 去掉双门 / harm 否决 | SkillAdaptor 方差 ±5.2 → ±8.1 |
| 只蒸馏成功经验 | 丢弃 negative/conflict/abstain Episode | SkillAdaptor「增益消失」 |
| Consumer-blind | 同时切断 Observation 可见性与 Scope 条件化 | G5 已定，归因只到整机，不拆层 |
| 无 Scope | Skill 无条件复用 | dev 锚点：叶 Scope +0.2127 vs 承重五轴 +0.6860 |
| 无 delayed | 只用 Support 单门 | 防自提自批的价值 |

---

## 3. 数据集与数据量

### 3.1 已冻结的池

| 池 | 规模 | 角色 | 实测缺陷富度 |
|---|---|---|---|
| **KDD Cup 2018 含缺失原版**（Zenodo 4656719） | 270 序列 × 9504–10920；503,712 缺失点；270/270 有缺失 | development 触发富池，**永不冒充 fresh** | 每课事件代理 **15.00**（过 ≥2/课）；4 满员 cell + spare 30 |
| **traffic leftover**（TSL traffic 列 480–861） | 382 列 → 6 cell + spare 22 | F1 同族新 Target，承担 A5 vs A3 主结果 | family-aligned **poor**（0 NaN；周窗位移 0–3 条/cell）→ 只能承担**易档 / 正确弃权场** |
| **electricity leftover** | 21 列（300–319 + OT） | spare-only | moderate，不够一个 cell |
| **Solar 10 Minutes**（Zenodo 4656144） | 137 序列 × 52560，0 缺失，md5 已核 | F2 密封 capstone，仅承担 C5c | **隔离中**，除完整性核验外零分析 |

### 3.2 cell 几何（冻结常量，`run_e2_s2a_forecast_oracle.py`:45-52）

```
CELL_WIDTH 60 = N_TRAIN 40 (= Support 20 + delayed 20) + N_HELDOUT 20
最小长度 1848 = ORIGIN_HELDOUT 1800 + H 48；ORIGIN_HELDIN 1104；PERIOD 24
材料线 max(0.005, 1/n_half) -> n_half=20 时 = 0.05；HARM_BAR = 0.005
```

### 3.3 【关键建议】实验单元从「仅序列轴」扩到「序列块 × 时间原点块」

按「序列作行」，Solar 137 序列只切得出 **2 个 cell**（capstone 太薄）；traffic leftover 6 个；KDD 4 个。
这是历史上 `|g/SE| <= 2.27` 的**结构成因**（n 太小），不是方法问题。序列很长，时间轴还有大量未用容量：

| 池 | 序列长度 | 互不重叠 1848 窗数 | 序列块数 | 单元上界（保守取 1/3 时间块） |
|---|---:|---:|---:|---:|
| Solar | 52560 | ~28 | 2 | **18** |
| traffic | 17544 | 9 | 6 | **18** |
| KDD | ~10900 | 5 | 4 | **8** |

建议：**单元 = (序列块 × 时间原点块)**，块划分预注册、互不重叠；统计推断**按序列块 cluster-robust**。
这是本稿最重要的单条改动：把功效问题从「无解」变成「可解」，且不需要新下载、不碰密封件。

### 3.4 规模目标（主实验，不含 pilot）

- 三档难度混编进同一课程（G4 已定），单课程 5–8 单元；每档 >=8 单元；**总单元 >=30**
  （配对设计：n=30 在 80% power 下可检出 d≈0.53；锚点 cls `+0.6860` 属大效应、forecast r2 属零效应，异质性会很大）
- **Consumer 轴**：预测腿 `pooled_ridge_a1` vs `per_channel`（均 ridge × sMASE）；
  **建议再加一个结构上不同的 consumer**（DLinear 或 kNN），否则 Q1 只是「两个线性模型之间的差异」。
  分类腿 ridge/kNN 作困难/敏感性案例；AD 三 consumer 只承担安全读数
- **难度分档来源**：建议由 **KDD 按缺失率/测项自然分层**承担（缺失率 min/med/p90/max = 0.55% / 11.72% / 34.04% / 97.67%；
  孤立 MAD-8 游程 med/p90/max = 12 / 54.2 / 292），而非注入（AGENTS.md §8：受控注入不替代自然数据能力证据）。
  traffic leftover 天然是易档
- **种子**：随机臂（B2、Random-legal-edit）>=3 seed；LLM 臂 T=0 且全课程重复 >=2 跑（CLS 线「两跑同向」标准沿用）
- **Backbone**：主结果单 backbone；**最短课程在第二 backbone 复现**（SkillAdaptor 用了 Kimi/GLM/GPT 三个）

---

## 4. 指标

### 4.1 主读数（口径按 `docs/READOUT_GLOSSARY_2026-08-19.md`）

1. **held-out regret**（相对密封 oracle 的差）——建议作**主终点**，比原始 gain 跨 cell 可比
2. held-out 效用：macro sMASE gain vs identity
3. **harm**：material-harm 计数与累计（`support_gain < -0.005`）+ **worst-series gain**（`HARM_BAR = 0.005`）
4. **错误晋升 / 正确弃权**：无 headroom cell 上部署 identity 的比例（安全腿主读数）

### 4.2 效率读数

5. **到首个安全有效 Skill 的 consumer fits**（成本侧优效终点）
6. time-to-threshold、fit 墙钟
7. LLM 调用数、`real_support_probe_count`——必须用 glossary 口径；`charged_probe_cost` 绝不当 probe 数（0.626 vs 0.759 教训）

禁：任意合成总分。预算记账单位 = **课程级**。

### 4.3 建议增补三条（各有邻域出处，成本极低）

8. **修复保真度**（AegisTS `EvaluationMetrics.py`：precision / recall / f1 / rra / mnad）——**只在受控注入单元上报，
   作机制诊断，不进目标函数**。AegisTS 把它按 0.5 权重写进 reward；我们刻意不这么做。
   「内在质量代理 vs 下游因果效用」何时背离，是一段有价值的 related-work 论述
9. **预测式归因命中率**（AHE decision observability）——每次 Skill 铸造/修订自带 `predicted_affected`，下轮回填实测，
   报 precision/recall vs 随机基线（AHE 报 11.8 / 11.1 vs 随机 5.6 / 5.4）。零代码纯协议
10. **Skill 库轨迹**（RewardHarness 报 0→13→7，明写「收益来自剪枝不是扩张」）——报库规模、收窄 PATCH 次数、
    限制/撤权次数随课程的曲线。本项目有内容 sha 版本链 + 快照血缘，能画得更细

### 4.4 统计

- 配对逐 cell 差值 + **按序列块 cluster 的 paired bootstrap CI** + Wilcoxon signed-rank
  （照 Adaptive Auto-Harness：bootstrap CI + Wilcoxon，且明确报「不显著」）
- 6 个对比对做 Holm 校正，或预注册单一 primary（`A5−Static`）其余为 supporting；非劣 delta 预注册
- 报**逐 cell win/tie/loss 计数与配对差值 CDF**，不只报均值

---

## 5. 评测方式（协议序）

```
Stage 3 pilot（dev 池，G2/G3 防火墙）-> 冻结 Harness v2
-> 预注册书落盘 + hash（数据 roster / split / cell 几何 / 菜单 / consumer / 预算 / 指标定义 /
   判词谱系 / 统计方案 / 超预算裁剪顺序），在打开任何 F1/F2 outcome 之前
-> G1 受测性前置门：修订触发基率 >=2/课（KDD 实测 15.00 已过；F1 池须另测）
-> 主实验：held-in 多轮适应 (r1..rR) -> freeze -> held-out Fast-only
   （禁 open_delayed / Slow / Skill 更新 / 看结果重试）-> 外部 evaluator 一次性开分
-> 消融（同一冻结课程 replay）
-> F2 密封 capstone（Solar）：一次性，开后不改不重跑；A3/A5 同 Target 反馈预算
-> 论文整理
```

分析纪律：**ITT**（进课程的单元全计）；机械退出按 glossary 的 paired-comparable 规则预注册剔除；
重复 replay 不冒充新证据，报告区分「新反馈 / 缓存重放 / 重复观测」。

**RewardHarness 的教训必须写进协议**：他们的 62.5% 是在被 rollback 规则优化了 77 轮的那 40 条 val 上测的，无独立 test split。
我们的对应纪律 = **驱动接受的面（held-in Support/delayed）与最终计分的面（held-out / F2 密封）必须完全不相交**，
且 held-out 结果不得回流本次 Harness。

### 预注册判词谱系

| 结局 | 论文写法 |
|---|---|
| (a) 非劣过 ∧ (b) 成本优效过 | 主张成立：等效用、更低成本、零伤害 |
| (a) 过 ∧ (b) 不过 | 「经验不损害但不加速」；主张退到条件化 + 安全 |
| (a) 不过 | 负结果：A5 效用主张明确否定；论文以 Q1/Q3 为主张 |
| capstone NEUTRAL / 正确弃权 | 有效结局（CLS capstone 先例），不算失败 |

---

## 6. 邻域对照表（related work 底稿）

| 维度 | AegisTS | TimeClaw | RewardHarness | SkillAdaptor | 本工作（建议后） |
|---|---|---|---|---|---|
| 下游任务数 | 3（forecast/cls/clustering） | 3 benchmark | 1（偏好评判） | 3 benchmark × 3 模型 | 3 Task 腿 × >=3 Consumer |
| 结局指标 | final model 提升 | benchmark 分数 | val accuracy | 任务成功率 | held-out regret + 到首个安全 Skill 的 fits |
| 裁定信号 | 连续 perf + 连续 issue rate | 连续 | **二值**（连续 gap 只做诊断） | 二值 Δ>=0 | 连续 sMASE 差 + 材料线 |
| 接受准则 | RL 回报 | 无（评测型） | val 超历史最佳否则回滚 | 确定性重跑 Δ>=0 + frozen 任务回归一票否决 | 双门 + harm 否决 + 阶梯定价 |
| 每步反馈成本 | 便宜 proxy 每步 / 昂贵 final 终局 | 工具调用 | 一次 LLM 判断 | 重跑任务 | 全额重训（0.685 s/次） |
| 外部基线 | EDITOR 等 | benchmark 榜 | GPT-5 / GPT-4o | 三 benchmark 基线 | **B0–B4（本稿新增）** |
| 消融结构 | 3 消融文件 + cross_dataset | k-neighbors 0 vs 3 | **无消融表** | 4 条消融 | 6 条（§2） |
| 检索键 | 不适用 | ~20 维指纹 + NL cosine 两段 | 库内选子集 | embedding + LLM rerank | 五轴 Scope（+ B4 指纹对照） |
| 注入 ground truth 是否进目标 | **是**（权重 0.5） | 不适用 | 不适用 | 不适用 | **否**（只作诊断，§4.3-8） |

值得直接借的四件：

1. **AegisTS 的 proxy/final 两层模型**（`RLclean.py`:893/1240/1266）——便宜 proxy 每步给信号、昂贵 final 只在终局。
   这是固定预算下**提高有效样本量**的现成机制，直接缓解「全额重训 0.685 s/次」的成本约束
2. **AHE 的 decision observability**（§4.3-9）
3. **RewardHarness 的库规模轨迹图**（§4.3-10）
4. **TimeClaw 的单开关消融形态**（B4）

刻意不借：AegisTS 把内在质量指标写进奖励（权重 0.5）；SkillAdaptor 的纯 LLM 软归因
（Localizer/Linker 无核销，正是 P1 `WRONG_REPLACEMENT_SELECTED` 否定过的路径）。

---

## 7. 风险与缺口

1. **中档难度池天然稀缺**：metr_la（207×1024）、nn5_daily（91×714–791）、ETT-small（7 列/文件）、weather Jena（21 列）
   全部过不了「60 序列 × 1848 长度」门。建议中档由 KDD 缺失率分层承担（天然），而非注入
2. **Solar 可能 defect-poor**（0 缺失，光伏出力有强昼夜结构与夜间零段）→ capstone 可能落在「正确弃权」上。
   须预注册接受 NEUTRAL 为有效结局。隔离令：Phase 3 开考前不得做缺陷普查，Solar 难度分档只能用 dev 冻结阈值在开考时机械套用
3. **菜单扩到 19 算子会拉低冷发现率**（实测 29%/位），B2 必须同时报 1×/3×/5×，否则「baseline 被做弱」的质疑无法回应
4. **时间轴扩单元引入相关性**：必须 cluster-robust，块划分预注册，并先在 dev 池测块间相关
5. **第二 backbone 的成本**：预算不足时按 G4 裁剪顺序（消融 > 档数 > 域数，永不裁臂），但 backbone 复现建议优先于第三条消融
6. **AD 腿只承担安全读数**：`#43 M0-C` 已在 Yahoo 24 条 × 三 Consumer × 五程序上关闭正效应（12/12 宏效用为负）
7. **B2 若在扩大菜单后仍打平或胜出**，这是真结论，必须照录：意味着本任务族上「结构化经验」不优于「等预算搜索」，主张须相应收窄

---

## 8. 与现行 v1 设计的差异清单（供 sol 逐条裁）

| # | v1 现状 | 本稿建议 | 理由 |
|---|---|---|---|
| 1 | 四臂全内部 | 加 B0–B4 五类外部基线 | 拒稿风险最高项 |
| 2 | 菜单 5 算子 | 扩到 19 算子 × per-series × <=2 步 | 否则等预算穷举完爆 |
| 3 | 单元 = 序列块 | 单元 = 序列块 × 时间原点块 + cluster 推断 | 解功效结构问题；零新下载 |
| 4 | 主读数并列多项 | co-primary：效用非劣 + 成本优效，harm=0 作 gating | 与已有证据一致，三种结局都能成文 |
| 5 | consumer 轴 = pooled vs per_channel（同为 ridge） | 加一个结构不同的 consumer（DLinear/kNN） | Q1 说服力 |
| 6 | 未定 backbone 轴 | 最短课程第二 backbone 复现 | 挡「模型伪影」质疑 |
| 7 | 未含 proxy/final 两层 | 引入 AegisTS 式 proxy 探测 + final 计分 | 固定预算下提高有效 n |
| 8 | 未含预测式归因读数 | 每次 Skill 编辑附 `predicted_affected`，下轮回填 | 零代码，出图 |

**本稿不裁定任何一条**；执行顺序、是否采纳、预算裁剪一律由 sol / Planner 决定。

---

## 9. AegisTS 主实验 table 的实际结构（可直接照搬的骨架）

来源：`_tmp_aegis_paper.md`（arXiv 2605.04902v2 正文）。逐节核对。

### 9.1 它的四张表

| 表 | 内容 | 行/列结构 |
|---|---|---|
| **Table 2** 数据集统计 | ETTh1 / IDF_OilTemp / Libras / Handwriting；**6 个 dataset-task 配置** | 列 = #Categories, Length, #Samples, #Attrs, Task |
| **Table 3** 主表 | 「Overall comparison on all datasets across different downstream tasks」 | 行 = Task × Method；列 = 数据集 × {**Upstream**: F1↑, NMSE↓, RRA↑ ‖ **Downstream**: Perf↑, **ΔPerf↑**}；最优加粗 |
| **Table 4** 泛化 | Cross-Dataset Transfer（同 task 内 source→target，零重训） | 列 = NMSE↓, ΔPerf↑ |
| **Table 5** 成本 | CPU 运行时 | 行含 **Brute-Force Search > 3 days**、**Single Agent**（去分层的自己，慢 1.9–2.8×） |

之后还有：超参敏感性（μ1–μ4 四条曲线）+ 三个消融（`wo_tasks` / `wo_proxy` / `wo_metrics`）。

### 9.2 它的 Baseline（§6.1.2）—— 全部是「传统统一清洗方法」

- **EDITOR**：多分辨率检测-定位-修复的 MTS 清洗；只能单序列数据集
- **Clean4TSDB**：约束挖掘 + 错误画像，按时序/多变量依赖修复
- **DiffPrep**：可微流水线生成（表格 SOTA，改成时序算子；只适用单序列）
- **Sampling**：随机穷举——从算子池随机采样至多 Lmax 个算子并随机排列
- （Table 5 另有 Brute-Force Search、Single Agent 两条内部对照）

### 9.3 它的指标（§6.1.3）—— 上下游两组

- **上游（清洗保真，依赖注入 ground truth）**：F1（问题检测）、NMSE、RRA
- **下游（任务效用，全部归一化到 [0,1]，越大越好）**：
  预测 = avg(`e^−NRMSE`, 归一化 CC)；分类 = avg(Macro-F1, ROC-AUC)；
  聚类 = avg(归一化 Silhouette, 反比 Davies–Bouldin)
- **ΔPerf = 相对脏数据基线的提升**——这是它的主结论列
- 双层模型（§6.1.4）：proxy = DLinear / MiniRocket / Catch22（每步）；
  final = LSTMForecast / InceptionTime / AEDCNN（终局计分）

### 9.4 它的主结论数字（供对标）

摘要口径「up to 96% 清洗质量提升、27% 下游提升」。ΔPerf：
预测 `+0.0442`(ETTh1) / `+0.1666`(IDF_OilTemp)；分类 `+0.1333`(Libras) / `+0.0660`(Handwriting)，
**同格 Clean4TSDB 为 `−0.0626`**；聚类 `+0.0424` / `+0.1133`；
迁移峰值 `+0.1589`，最差 `−0.0034`。

**可用的论述**：它自己的表就证明了「任务无关的统一清洗会伤下游」（Sampling 在 Handwriting 分类
`−0.1078`、Clean4TSDB `−0.0626`）。我们的「正确弃权」机器正是针对这一点，且我们有 AegisTS 没有的
安全列。

### 9.5 我们的表骨架（照 9.1 改写）

**Table 1 数据集**（列：#序列 / 长度 / 频率 / 自然缺陷画像 / Task / Consumer / 新鲜度层）：
KDD2018 含缺失 270×9504–10920（dev 硬档）；traffic leftover 382×17544（F1 易档）；
electricity leftover 21×26304（spare）；Solar 10-min 137×52560（F2 密封）；
UCR GunPoint 族（分类腿）；Yahoo S5 sealed 41（AD 腿，安全读数）。

**Table 2 主表** —— 行 = 方法组，列 = 每个 (dataset × task × consumer) 配置：

| 组 | 行 |
|---|---|
| 不处理 | identity（ΔPerf 的分母） |
| **传统统一清洗** | IQR/3σ 剔除 + 线性插值；Hampel 滤波；winsorize；SCREEN（速度约束）；Clean4TSDB / Cleanits（若代码可得） |
| 搜索类 | Sampling@1× / @3× / @5×（AegisTS 自己用的那条）；Brute-force（只报时间） |
| agentic SOTA | **AegisTS 本体**（代码在本地 `a-evolve/AegisTS`，含 `environment.yml`）；一次性 LLM 规划器（AutoDCWorkflow 形态） |
| 记忆对照 | fingerprint-kNN（TimeClaw 式，无 Skill/Scope/双门） |
| **Ours** | Static / A3-reset / K0-fixed / **A5（完整）** |

列分三组：**Downstream**（Perf↑, **ΔPerf↑ 主**）‖ **Safety**（harm 事件、worst-series gain、
正确弃权率——**AegisTS 没有这组，是我们的差异化列**）‖ **Cost**（consumer fits、LLM 调用、墙钟）。

**Upstream 列（F1/NMSE/RRA）只在受控注入档报**，作诊断；自然档不报——
因为本项目立场是不把内在质量当目标（AGENTS.md §8 / §4.3-8）。这一不对称须在表注里写明。

**Table 3 消融**（§2 六条）；**Table 4 跨域泛化**（Solar 密封 capstone + 分类腿族外沉默）；
**Table 5 成本**（照 AegisTS Table 5，含 Brute-force 的时间上界——这是「策略引导 ≈ N× 便宜」的出处）。

### 9.6 基线的可行性排序（决定能不能在预算内做完）

| 档 | 基线 | 成本 |
|---|---|---|
| 便宜（0 LLM 纯计算） | identity / IQR+插值 / Hampel / winsorize / Sampling@1×3×5× / fingerprint-kNN | 现有 `_evaluate` 直接跑 |
| 中 | SCREEN（速度约束，约 200 行） | 自实现 |
| 贵但价值最高 | **AegisTS 本体**——需适配「多变量 → 我们的单变量池 + ridge/sMASE Consumer」 | 建议先做一天可行性 spike，成或不成都记录 |
| 建议跳过 | DiffPrep（需可微下游 + 单序列限制）、EDITOR（TCN+GCN 需训练） | 在 related work 里说明未直接对比的理由 |

若 AegisTS 适配不可行：外部基线退为「4 条传统统一清洗 + Sampling 三档」，并在论文中诚实说明
适配障碍（单/多变量形态不匹配），这在评审上是可接受的，前提是**说清楚**。

---

## 10. 目标覆盖审计 + 范围裁定（2026-08-29，用户裁）

对照项目目标陈述逐句核，并记录用户当轮的四条范围裁定。

### 10.1 覆盖表

| 目标陈述 | 落点 | 覆盖 | 裁定 |
|---|---|---|---|
| 质量与**下游任务**相关 | 三腿 forecast / cls / AD；`task_kind` observable 已含三值（`contracts/observables.py`:69-71） | 部分——三腿从未在同一份数据上同时评过 | 见 10.2 补丁 B |
| 质量与**模型结构**相关 | consumer 轴 pooled vs per_channel | 部分——两个都是 ridge | 建议加一个结构不同的 consumer |
| 质量与**时序模式**相关 | 五轴 Scope 的 Pattern family 轴 + observable 词表 | 足（趋势轴缺，见 10.3） | — |
| 固定清洗在预测提升、在 AD 破坏 | 仅 T1b/T3 注入正控；自然侧 `#43 M0-C` 为 12/12 全负 | 缺自然证据 | 见 10.2 补丁 B |
| **直接让 LLM 判断缺乏可靠反馈** | 原无实验 | 已补 | **用户裁：作 Baseline / ablation 顺手做**，落在 B3（§2 已改） |
| 提升下游**模型性能** | Table 2 ΔPerf 主表 | 足 | — |
| 提升**训练效率** | — | — | **用户裁：本意 = 提升数据质量从而提升下游模型能力，不必做训练效率**。收敛为 ΔPerf 单一效用读数；fits/LLM 调用仅作次要 efficiency 报告，不作独立主张 |
| 生成可执行数据准备策略 | typed workflow + verifier + 19 算子 | 足 | — |
| 迭代优化 **Skill** | SA-1 修订环，卡版本链 v0→v3/v4，两跑复现 | 足 | — |
| 迭代优化 **Memory** | Episode 四类 + store + 快照血缘 | 足 | — |
| 迭代优化 **决策策略** | — | — | **用户裁：由 Skill card 的 general / specific 两层体现即可**，不另开可编辑面 pilot。见 10.2 补丁 A |
| 迭代优化 **Instruction** | — | — | **用户裁：本轮不需要实现**。论文列 future work |

### 10.2 两条落地补丁

**补丁 A（裁定落地）：「决策策略」由 Skill card 的 general / specific 两层承载**

仓内已有两个现役机制，不需要新建可编辑面：

- **specific 层** = Target-local Risk Skill（R1）：同域某 Program 家族 ≥2 个 distinct Task 负向且无正向
  → `SkillKind.SAFETY` 条目（`contracts/harness.py`:100-103；使用点 `task_episode_harness/g1.py`、
  `methods/ttha/schema_contracts.py`；回归 `tests/functional/test_target_local_risk_skill.py`）
  → 被召回时对**探测顺序**确定性降权。**改变探测顺序 = 改变决策策略**，这就是这一面的读数
- **general 层** = G1 General Decision Guidance：Context-resolved 负向 Experience → Slow 单 Surface PATCH
  → Fast 消费。机制端到端已通过（`G1_END_TO_END_GUIDANCE_WIRING_PASS`），效果 `INCONCLUSIVE`

**直接后果：Stage 3 pilot 从关键路径撤下**（`STAGE3_PILOT_FREEZE_DRAFT_2026-08-29.md` 降为附录或不做），
主实验四臂结构不变。§0 判断 1 的次级读数相应改为「general/specific card 是否改变了 Fast 的探测顺序与
提案分布」。

**必须同时处理的前置阻塞**：R1 的核心负结果是「**Skill 送达 ≠ 生效**」——纯文本降权在 fast_path 上
降权量恒为零，因为第一次 Support 探测在 select 之前无条件花在 `compiled_rows[0]`；且
`PROPOSAL_DIVERSITY_GAP`（28–50% 臂-Task 只提一个候选）使重排在这些 Task 上**结构性无法动作**。

> **因此：把菜单从 5 个算子扩到 19 个，不只是为了打败 Sampling 基线（§0 判断 3），
> 更是「决策策略」这一面在实验中可测的前置条件。** 一个候选时无序可排，这一面必然测出零。
> 这两条动机现在合流，扩菜单从「建议」升为「必要条件」。

**补丁 B：跨任务符号反转列（motivating claim 的实验化）**

同一份数据、同一个固定清洗规则（`outlier_mad` / `winsorize`），同时跑 forecast 与 AD consumer，报两个
ΔPerf 的**符号**。`#43 M0-C` 的 12/12 全负有利于该主张（固定清洗在 AD 上普遍有害），只需 forecast 侧
同数据为正。障碍是数据不共用（Yahoo=AD、traffic/KDD=forecast）；**更便宜的解法 = 在 Yahoo S5 上接一个
forecast consumer**（单变量长序列可做预测），而非给 traffic 接 AD 适配器。

### 10.3 Observation 覆盖与趋势轴缺口

现役 observable 词表（`contracts/observables.py`:21-44）对目标句四类模式的覆盖：

| 目标句里的模式 | observable 字段 | 状态 |
|---|---|---|
| 缺失 | `missing_fraction`, `longest_missing_run_fraction` | ✓ |
| 异常 | `local_robust_z_peak`, `outlier_region_end_fraction`, `clipping_probe_direction` | ✓ |
| 周期 | `period_change_score`, `period_reliability`, `period_evidence_status`, `period_repair_available` | ✓ |
| （水平位移） | `level_excursion_score`, `estimated_level_offset`, `level_region_fraction` | ✓（目标句未列，实际有） |
| **趋势 / 漂移** | **无对应字段** | **缺** |

二选一：补一个 trend/drift observable，或把「趋势」从目标陈述中删去。
另建议一张零成本描述表：pattern 类型 → observable 字段 → 在最终 Skill Scope 中被引用的频次。

### 10.4 裁定后的论文结构

```
Motivation      LLM 直判可靠性（B3 读数）+ 跨任务符号反转（补丁 B）
Method          同一 harness 入口 × 三 TaskSpec × N Consumer；
                Skill card 两层（general 决策指导 / specific Target-local + SAFETY）
Table 2 主表    ΔPerf（vs 传统统一清洗 + Sampling + AegisTS）+ Safety 列 + Cost 列
Table 3 消融    六条（§2）
Table 4 泛化    Solar 密封 capstone
Table 5 成本    含 brute-force 时间上界
Limitations     Instruction 面（future work）/ 趋势轴 / 注入 vs 自然 / 单 backbone
```

「训练效率」不再作为独立主张；「Instruction 迭代」进 future work；「决策策略」由 Skill card 两层承担，
不另立 Stage 3 可编辑面实验。
