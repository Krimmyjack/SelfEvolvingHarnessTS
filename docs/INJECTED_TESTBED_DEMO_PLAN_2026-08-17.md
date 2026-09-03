# 注入底座 Demo 计划（2026-08-17）

> **历史计划，已停止执行。** 2026-08-17 起，本文件的全部未完成步骤与
> 「修正案 A：反馈单位改为 Episode」均由
> `docs/TASK_EPISODE_HARNESS_EXECUTION_PLAN_2026-08-17.md` 取代。
> 本文件只保留历史设计与审计记录；不得继续从中挑选步骤执行，也不得与新任务书并行维护。

## 0. 本文档的地位

- **取代**：K1 per-origin outlier 支线的全部后续计划（S1a′/S1b/S1c、M0/M1/M2、R1/ICC）。
- **不取代**：已冻结的 verdict、rows、protocol。历史报告一律不动，只追加。
- **历史说明**：本条原为“唯一计划”，现已由页首所指的新任务书取代。

验收标准来自导师意见，**明确低于且不同于"统计上站得住的正结果"**：

> 先跑通整个流程并做出可视化效果，即使效果不好也没关系。先有一个整体的东西。

因此 **第 4 步（端到端跑通）是硬验收线**，第 6 步（调好看）是加分。不要因为效果不好而推迟第 4 步。

---

## 1. 为什么转向（三个已核实的事实）

| 事实 | 数字 | 出处 |
|---|---|---|
| 唯一超过 t=2 的效应 | outlier_mad 汇总 +0.05696，SE 0.0270，**t=2.11**，4/6 origin 为正 | `w1_p4_headroom_2x2_report.json` → `s1b_d3_zero_llm_v3` |
| 想学的标签基本是噪声 | 方差分解：between-origin **6.1%**、between-series 7.5%、残差 **86.4%**；split-half 一致率 **45.7%**（50% = 纯噪声） | 同上 `d5_support_query_sensitivity.per_series_gain_matrix` |
| **根因**：一直在比两个近似替代品 | `outlier_mad` 与 `hampel_filter` 同属离群族，做的事几乎一样；全部实验只用了算子库 14+ 个里的 3 个，全在同一族内 | `operators/s1_outlier.py:30,36,43,53` |

跨族对比完全不同：有缺口的序列 `impute_linear` 能救、`outlier_mad` 无效；有尖峰的序列反过来。**大且符号相反**，这是可构造的。

---

## 2. 已验证可用、不要重建的东西

| 资产 | 位置 |
|---|---|
| 完整算子库（5 族 ~20 个） | `operators/s1_impute.py`（`impute_linear:35` `impute_fft:60` `impute_ema:86` `period_complete:105` `period_median_complete:118` `impute_ssm:183` `impute_ar:285`）、`operators/s1_outlier.py`（`winsorize:30` `outlier_iqr:36` `outlier_mad:43` `hampel_filter:53`）、`operators/s1_denoise.py:41 denoise_median`、`operators/s1_structural.py:228 repair_level_shift`、`operators/s2_align.py` |
| 算子注册表 | `operators/registry.py`：`get_operator:240` `operator_targeting_mode:228` `canonicalize:223` |
| **按训练序列子集施加程序**（本阶段核心，从未启用） | `evaluation/functional/run_e2_autonomous_natural_workflow_generation.py:672,697` 的 `train_series_scope` 参数 |
| 可观察特征 + 5 档冻结分箱 | `contracts/observables.py`：9 个数值特征、`OBSERVABLE_NUMERIC_BIN_LABELS`、`_NUMERIC_BIN_EDGES:48`（`missing_fraction` / `longest_missing_run_fraction` 的边界 `(0.0,0.01,0.05,0.20)` 正是为缺口检测设计的） |
| 25 类归因路由表 | `evaluation/minipipe/feedback/fault_routes.json` + `evaluation/minipipe/feedback/router.py:38` |
| 现成 dashboard | `artifacts/functional/e2/w1_evolution_dashboard.html`（239.6K），生成器 `evaluation/functional/run_v1_guidance_evolution.py` |
| 机械链（已通，不要重验） | 自然失败 → 路由 → 有界 LLM 选择 → 精确 Program replay → 合取门 → Skill 形成/撤销 |

---

## 3. 交付顺序

```
1. 注入器
2. 5 类归因投影
3. novelty gate
4. 端到端跑一遍（数字难看也照跑）   ← 硬验收线
5. dashboard 接四块
6. 调注入强度让效果变好
7. A5 vs A3
```

---

## 4. 各步规格

### 步骤 1 — 注入器

一个模块，签名类似 `inject(series, fault_type, strength, rng) -> (corrupted, ground_truth)`。

三种故障，各自有明确的匹配算子族：

| fault_type | 匹配族 | 代表算子 |
|---|---|---|
| `gap` | 填补 | `impute_linear` |
| `spike` | 离群 | `outlier_mad` |
| `level_shift` | 结构 | `repair_level_shift` |

**两阶段编排**（这是"自进化"叙事的骨架，不要压成一阶段）：

- **Phase 1**：12 条训练序列分 3 组各 4 条 —— `G_gap` / `G_spike` / `G_clean`（不注入，对照）。case 库应学到 2 个 case。
- **Phase 2**：向 `G_clean` 的一部分注入 `level_shift`（Phase 1 时库里没有这一类）。case 库应长出第 3 个 case。

`strength` 必须是参数，第 6 步才调。

### 步骤 2 — 5 类归因投影

**不动 25 类路由表，不删任何 cause code。** 只加一个投影映射，demo 路径上只用这 5 类：

| 类别 | 含义 | ground truth 判定 |
|---|---|---|
| `DIAGNOSIS_WRONG` | 判断的故障族 ≠ 注入的故障族 | 直接比对 |
| `OPERATOR_WRONG` | 诊断对，选的算子族 ≠ 匹配族 | 直接比对 |
| `PARAMETER_WRONG` | 族对，强度/参数导致过度或不足处理 | 同族换参数能过门而当前不能 |
| `SCOPE_WRONG` | 算子和参数对，但作用到了不该处理的 source | per-source 增益符号不一致 |
| `NO_FAULT` | scope 内没有注入任何故障，任何处理都有害 | 正确动作 = 弃权 |

给 LLM 的提示就是**一道五选一的选择题**，选完再在该类内部找具体错误。不要开放式归因。

因为有注入 ground truth，归因准确率是可测的 → 输出 **5×5 混淆矩阵**。

### 步骤 3 — novelty gate

不用 embedding，不用新模型。

- **签名** = 9 个数值可观察量各自的分箱标签，构成一个 9 元组
- **距离** = Hamming 距离（不同位置的个数）
- **入库规则**：新 case 入库当且仅当它与库中**每一个**已有 case 的距离 ≥ k
- **k 在开跑前冻结**，报告里给出完整距离矩阵

LLM 只负责提议 case，**runtime 判定是否入库**（沿用"LLM 不批准自己的 patch"）。实现量约 20 行。

### 步骤 4 — 端到端（硬验收线）

**"跑通"的定义是机械的，与效果无关：**

1. Phase 1 + Phase 2 全部 episode 跑完，无异常中断
2. 每个阶段都产出自己的 artifact
3. 5 类归因对每个失败都给出一个类别
4. novelty gate 对每个候选 case 给出入库/丢弃判定
5. dashboard 能渲染

数字难看**不是**推迟这一步的理由。跑完立刻停下来做步骤 5。

### 步骤 5 — dashboard 四块

复用 `run_v1_guidance_evolution.py` 的生成器，不要从零写。

1. **episode 时间线**：注入故障 → 归因类别 → 选中算子 → 门决策（接受/拒绝/弃权）→ 增益，一行一条
2. **5×5 归因混淆矩阵**（真实准确率，因为有 ground truth）
3. **case/skill 库增长曲线**（Phase 1 长到 2、Phase 2 长到 3）——最直观的"自进化"画面，优先做好这块
4. **A5 vs A3**：到第一个有效 Skill 的 probe 数（步骤 7 之后才有数据，先留位）

### 步骤 6 — 调强度

现在才调 `strength`，目标：

- **对角线**（匹配修复）增益 ≥ 0.3（当前噪声 SE 为 0.027，即 ~10×）
- **非对角线** ≈ 0 或为负
- **split-half 一致率 ≥ 90%**

**调完必须冻结，再重跑上层实验。** 标定底座和跑 harness 实验之间要有明确的冻结线，写进报告。

### 步骤 7 — A5 vs A3

同一 Target feedback 预算下比较：到第一个通过门且经 delayed 确认的 Skill，两臂各花了多少 probe、造成多少 harm、弃权多少次。

**开跑前必须逐字节比对两臂的 Slow 输入，全等则停**（P4 就是栽在这里：两臂输入完全相同，无论结果如何都无法区分 A5 与 A3）。

---

## 5. 冻结约束

**继承（不变）**

- 零 LLM 存在性检查先行：派 LLM 之前先用确定性方法确认答案存在
- LLM 不批准自己的 patch：批准权在确定性 compiler + replay + in-domain feedback
- 配对同场验证：禁止引用历史基线数字当裁判
- 反过度工程：每实验最多一个 runner package + 一个主报告 + 一个必要测试
- 机械层断裂即停：报告，不重跑，不二次修复
- 委派深度 = 1
- 已冻结的 verdict / rows / protocol 只追加不改写

**本阶段新增**

- 不改 `contracts/observables.py` 的 schema，不加新特征，不改分箱边界
- 不删 25 类 cause code，只加投影
- 注入强度只在步骤 6 调，调完冻结
- 合成注入在报告和 demo 里明确标注为**受控测试床**

---

## 6. 明确停止

以下全部进 backlog，本阶段不碰：

- per-origin routing / per-origin Program 选择（K1 n=8 下已可信关闭）
- Scope 谓词 / `observable_applicability` 学习线（三个独立条件信号全为空，方差分解已解释原因）
- R1 / ICC / cell 级重复测量（换底座后不再需要）
- 新聚合器、新 Gate、新 Hash 体系、新 Registry
- 外部项目（TimeClaw / IT'S TIME）scope 集成
- 扩大 cohort / 新序列曝光

**规则**：局部异常只有在直接阻断上面 7 步纵向链时才修，其余一律记入 backlog，不发展成新支线。

---

## 7. 报告格式

一个主报告 `artifacts/functional/e2/w1_injected_testbed_report.json`，至少包含：

```
injection_spec        # fault_type / strength / 分组 / rng seed，phase 1 与 2 分开
frozen_at             # 强度冻结的时间点与 sha
case_library          # 每个 case 的签名、入库时的最小距离、入库/丢弃判定
attribution           # 每个失败的预测类别、真实类别；5x5 混淆矩阵
episodes              # 注入 → 归因 → 算子 → 门决策 → 增益
gate                  # 接受/拒绝/弃权计数
a5_a3                 # 步骤 7 之后
verdict
```

每步完成后追加，不重写。

---

# 修正案 A：反馈单位改为 Episode（2026-08-17 追加）

**落点**：步骤 4 之前。步骤 1–3（注入器、5 类归因、novelty gate）与评估单位无关，**照原样继续，不要返工**。

## A.1 为什么改

同一份已有数据，只换聚合单位：

| 决策单位 | 样本量 | 结果 |
|---|---|---|
| 单个 origin | 8 | SE 0.06–0.23，效应 0.05–0.18，t < 1，split-half 45.7% |
| 6 个 origin 汇总 | 48 | outlier_mad +0.05696，SE 0.0270，**t = 2.11**，可判定 |

per-origin 标签 86.4% 的方差在残差里，学不到；Episode 级标签一直可用。

## A.2 Episode 定义

```
Episode:
  episode_id
  target_domain          # 语料（= 一个注入配方）
  task_spec              # forecast / horizon 48 / ridge / sMASE
  corpus_signature       # 9 特征分箱签名（与 novelty gate 同一个签名函数）
  injection_recipe       # ground truth：哪些源注了什么故障、强度多少
  trajectories: [
    { workflow: [(op, params, scope), ...],
      metric:   mean sMASE over (全部 eval series × K origins),
      n:        len(eval_series) * K,
      outcome:  WIN | LOSS | HARM }
  ]
  winner
  conflicts              # 同一 workflow 在别的语料上输了的记录
```

三条性质必须成立：

1. **同一输入、同一指标、多条轨迹** —— Episode 内部是配对比较，不需要任何迁移
2. 指标聚合到 `len(eval_series) × K` 个样本，赢家明确
3. **迁移只发生在 Episode 之间，且只作为提案先验，不是执行权**

第 3 条是硬要求：Source 经验用于**排候选顺序**，不用于**批准 patch**。批准权仍在确定性 compiler + replay + Target 自己的 Support 反馈。

## A.3 适应性搬到 per-source scope

放弃 per-origin 自适应主张。适应性改为在**一个 Episode 内部**体现：语料同时含 gap 源、spike 源、干净源，正确答案是分作用域的复合 workflow——

```
impute_linear   → gap 源
outlier_mad     → spike 源
(不处理)         → 干净源
```

用 `train_series_scope`（`run_e2_autonomous_natural_workflow_generation.py:672,697`）表达。这是可测的，因为语料成分已知、效应由注入强度控制。

## A.4 四处改动

| # | 改动 | 说明 |
|---|---|---|
| 1 | 多 origin 聚合包装（~15 行） | `_evaluate(..., origin=)` 不动，外面套循环取平均 |
| 2 | **门的合取单位：origin → eval series** | 每条 eval series 在 K 个 origin 上的平均增益都须 ≥ M |
| 3 | Episode 存储接线 | `artifacts/experience/episodes.json` 与 `build_contrast_capsule` 的 positive/negative/conflict 三桶均已存在，是接线不是新建 |
| 4 | Slow 提案接口 | A5 输入 = 检索到的 Source Episode 轨迹（含输家）；A3 = 空 |

改动 2 的理由：每个条件从 1 个样本变成 K 个；且**一条 eval series = 一个真实消费者**，"不能帮 A 伤 B"在这个单位上才有意义，origin 只是时间点。单侧伤害否决语义完全保留。

改动 4 的副作用：两臂输入**天然不同**（差异在检索结果里），P4 那个"两臂逐字节相同"的失效模式在 Episode 架构下不会重演。步骤 7 的逐字节比对断言仍然保留作为保险。

## A.5 Episode 来源与预算

**注入器就是 Episode 生成器**：随机化注入配方 → 每个配方一个语料 → 一个 Episode。目标约 20 个 Source Episode + 10 个 Target Episode = 30 个配方。配方已知，所以 Episode 的正确答案有 ground truth，5 类归因和 novelty gate 都能直接评分。

**预算分配规则**：

> 每个 Episode 的样本量只要够让赢家明确（目标 **t ≥ 3**），剩余预算全部花在**更多 Episode** 上。

注入后效应约 0.3 量级时，`8 series × 3 origins = 24` 个样本即可（SE ≈ 0.06，t ≈ 5）。**不要把 6 个 origin 全塞进一个 Episode**——Episode 数量比单个 Episode 的精度更值钱，学习发生在 Episode 之间。

## A.6 对原文档的具体覆盖

- **步骤 4**：跑通的定义增加一条 —— Episode 记录（含 trajectories 与 winner）落盘
- **步骤 5**：dashboard 第 4 块改为「A5 vs A3：到第一个 WIN trajectory 的试错次数」
- **步骤 6**：冻结判据的对角线/split-half 判据改为在 **Episode 指标**上计算，不在单 origin 上
- **步骤 7**：比较项改为「谁更快找到正向 workflow、少试多少坏方案、最终 delayed harm 如何」
- **报告格式**：`episodes` 字段升级为上面 A.2 的完整结构

其余（冻结约束、停止清单、5 类归因、novelty gate、两阶段注入编排）全部不变。

---

# 修正案 B：执行细则四项决议（2026-08-17 追加）

## B.1 novelty gate 的 k —— 用标定，不用固定值

**实测**（T117 训练窗口 `[408:648]`，三类注入 × 强度 {0.25, 0.5, 1.0}，9 位签名 Hamming 距离）：

```
跨故障类型     最小距离 = 1     ← level_shift@0.25 vs spike@0.5，也 vs clean
同类型跨强度   最大距离 = 4     ← level_shift 0.25↔0.5、spike 0.25↔1.0
```

**双侧约束无解。根因是强度不是 k**：弱注入在可观察量上不可见（`level_shift@0.25` 与 clean 只差 1 位），且同类型跨强度会穿越分箱边界。

**决议 —— 标定规则**：

> 步骤 1 完成后立即跑距离矩阵，取
> **k = (同类型内最大距离) + 1**，并验证 **k ≤ 跨类型最小距离**。
> **不等式不成立时改强度，不改 k。**

**默认取值**：每个 `fault_type` 使用**单一固定强度**（Episode 间的差异来自配方组合，不来自强度）→ 同类型距离恒为 0 → **冻结 k = 3**。

若确需在 Episode 间变化强度并限制在 [0.5, 1.0]：同类型最大 3、跨类型最小 4 → k = 4，余量仅 1，不推荐。

报告必须给出完整距离矩阵。

## B.2 分组与强度

**分组冻结**（与 `w1_kdd2018_frozen_cohort_p41.jsonl` 的 train 顺序一致，已核对）：

```
G_gap   = T117 T118 T119 T12
G_spike = T120 T121 T122 T123
G_clean = T124 T125 T126 T127
Phase 2: T124 T125 注入 level_shift；T126 T127 保持 clean 对照
```

**强度不用统一 0.5**。实测与 clean 的签名距离：gap@0.5 = 3 ✓、**spike@0.5 = 2（偏弱）**、level_shift@0.5 = 3 ✓；且 `spike@0.5` 与 `level_shift@0.25` 距离仅 1。

**决议 —— 签名判据**（零模型运行，步骤 1 内完成）：

> 每个 `fault_type` 的强度取到：**与 clean 的签名距离 ≥ 3**，且**与其他每个 fault_type 的距离 ≥ 4**。

步骤 6 调的是**效应量**；签名判据在**步骤 1** 就必须满足。两者不要混。

## B.3 case 库的实体边界

**确认：demo-local，不晋升 h0。** 不写入 `methods/ttha/harness/h0`、不写入 Memory、不写入 skill 库。novelty gate 判定与 Phase 1→2 的 2→3 增长全部发生在 demo-local store 内。

理由：(1) 合成 case 进 h0 会污染被 sealed verdict 引用的快照；(2) 违反反过度工程约束；(3) demo 需要可反复重置重跑。

追加要求：

- store **确定性播种**且**可重置**，`--reset` 一次从空库重跑
- 报告写明 `store_scope: "demo_local"`，防止日后被误读为真实晋升

## B.4 归因口径

**(a) 矩阵范围**：混淆矩阵**只覆盖失败的 episode**，成功单独计一个标量。矩阵保持 5×5。成功的 episode 没有真实错误类别，硬塞会让准确率失去意义。

**(b) 混合语料的归因粒度**：**先按 training source 逐条归因，再取出现的最高优先级类别作为 episode 标签。** 附带产出 per-source ground truth（dashboard 可加一块）。

**(c) 优先级顺序**（"最早出错环节"规则，保持不变）：

```
DIAGNOSIS_WRONG → OPERATOR_WRONG → PARAMETER_WRONG → SCOPE_WRONG
```

**(d) NO_FAULT 与 SCOPE_WRONG 消歧**：

- **`NO_FAULT`**：**整个语料**未注入任何故障，harness 仍动手（正确动作 = 弃权）
- **`SCOPE_WRONG`**：语料有故障，算子对故障源正确，但作用域包含了 clean 源或漏掉了故障源
