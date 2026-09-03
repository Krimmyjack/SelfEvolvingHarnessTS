# 项目 B 状态报告 / 新 Agent 上手指南（2026-08-16）

> 目标读者：刚接手本仓库的 Agent。读完这份文档应能回答：项目在干什么、哪些已验证、
> 卡在哪里、下一步做什么、哪些事绝对不能做。
> 本文件是**状态快照**，不是设计文档；设计文档见 `docs/` 下各专题，历史 verdict 见
> `artifacts/functional/e2/w1_guidance_evolution_report.json`（下称"主报告"）。

---

## 1. 项目定位与目的

本仓库（`SelfEvolvingHarnessTS-deepseek-guidance-evolution`，包名 `SelfEvolvingHarnessTS`）
是 **SelfEvolvingHarnessTS 总项目的"项目 B"**：Guidance/Skill 文本自进化分支。
（项目 A = `../SelfEvolvingHarnessTS-deepseek`，AdaCTS 主线，Case→Skill；两仓库相邻，
共享数据 cohort 时须互相计入消费账本。）

总项目目标：**借鉴通用 Agent Harness 的 Workspace / Tool / Memory / 反馈归因 / 自进化机制，
构建面向时序数据的数据理解与数据适配 Harness**。核心命题：

> 在相同 Target downstream-feedback 预算下，能读取 Source 成功/失败/冲突 Experience 的
> Harness（**A5**），是否比空 Source Memory 的 Target-only 从头适应（**A3**）**更快且更安全地**
> 形成有效的 Target-local Skill。

知识分三层（前期用一个普通列表/JSON 存，不建三套系统）：

- **Experience Episode**：每次合法 Action–Response 都写入（成功/失败/冲突/abstain），不设正向门；
- **Target-local Skill**：当前 Domain 合法 Support 上形成，由后续 in-domain delayed 更新，只在当前 Domain 用；
- **Shared Capability**：多 Domain 相似可观察 Context 中重复正向证据后才归纳（本项目尚未到此阶段）。

**可迁移知识的基本单位不是算子，而是** `Multi-scale Context × executable Workflow × Action–Response Evidence → Context-conditioned Capability`。

---

## 2. 仓库结构与组件

```text
contracts/       Task/Program/Method 公共契约
conditioning/    时序特征、周期检测、条件路由
operators/       算子实现与唯一 registry（hampel/winsorize/impute_*/outlier_*/repair_level_shift...）
runtime/         executor、candidate_pool、candidate_verification、decision_trace、
                 llm_cache、public_features —— 确定性执行层
methods/ttha/    唯一可执行 Agent 方法（TTHA）
evaluation/      functional/（60+ 实验 runner）+ minipipe/（受控小管线）+ benchmark_v02/（冻结）
artifacts/       functional/e2/ 主报告与各实验落盘；frozen/benchmark_v02/ 冻结 registry
tests/           functional/（各实验的 smoke/integration 测试）+ contracts/ + integration/
docs/            设计文档、预注册、shift report
```

### 2.1 TTHA 方法层（`methods/ttha/`）

Fast/Slow 双路径：

```text
TaskSpec/Target Context
 → Fast: inspect → propose typed Program 候选（含 identity 对照）→ select → 执行
 → Support 反馈 → 立即写 Episode
 → delayed outcome 稍后打开 → 更新 Skill
 → 失败/冲突累积 → Slow: 归因 → 单 Surface 修改提案 → 配对 replay → 版本化 snapshot
```

关键文件：

| 文件 | 作用 |
|---|---|
| `fast_agent.py` / `slow_agent.py` | 两个 LLM Agent |
| `method.py` | `TTHAMethod`；`handle_fast_winner()`（:468）两阶段批准：宽 scope → `requires_target_support` Draft 门，replay/delayed 通过才写 snapshot（**LLM 不批准自己**） |
| `online_loop.py` | 真实在线生命周期：`run_online_round()`（:178，含 `allow_fast_skill`/`allow_slow`/`runtime_prior_slot`）→ `open_delayed()`（:437，只给实际部署的 winner 开 delayed）→ `activate_approved()`（:506，写 active snapshot） |
| `experience_memory.py` | Episode 写入（每次合法 Action–Response 立即写） |
| `retrieval.py` / `fault_cases.py` | 检索与 MATCH/CONFLICT/NEW/ABSTAIN 路由 |
| `group_fault.py` | 失败分组（**当前按 (workflow 指纹, 正负号) 分组，见瓶颈 ③**） |
| `scope_executor.py` | 带 scope 的候选评估 |
| `harness/store.py` | `SnapshotStore`，`set_active()`（:190）原子写 `active.json` |
| `harness/h0/` | 当前基线 snapshot：**rev7**（instruction + skills/bootstrap + skills/learned + candidate_policy/retrieval/verification 合约） |
| `harness/compiler.py` / `harness_surfaces.json` | snapshot 编译与可演化 Surface 清单 |

### 2.2 minipipe（`evaluation/minipipe/`）——受控小管线

- `cycle.py`：minipipe 主循环；`:525` 先例——**probe panel 只跑 `case.to_public_view()`**（信息墙纪律，接线时必须守住）。
- `probes/panel.py`：`M0_PROBE_SPECS`（:79）——每个算子族 × 剂量 0.25/0.50/0.75 的剂量-响应 battery；`_probe_direction` 读三个剂量的符号型态。**目前未接到自然数据线**（episode 里四个 `*_probe_direction` 全为 `unknown`）。
- `valuation/rolling_observed.py`：`RollingObservedValuator`（:52）——只用已观测数据估值，无需 ground truth。
- `feedback/fault_routes.json`：25 个 fault code 的路由表；**17 个落到 `EDITABLE_M0` + 通用文本 `PATCH`**（瓶颈 ③）。`OBSERVABLE_FEATURE_SCHEMA_GAP` 是唯一 `OBSERVATION_CAPABILITY_BACKLOG`（不可自修）。
- `feedback/first_fault.py` / `router.py` / `patterns.py`：归因与路由。

### 2.3 实验层（`evaluation/functional/`）

主文件：`run_v1_guidance_evolution.py`（约 460KB，单一 runner 承载 G/S/TSEM/FULLOP/USEL/N/A5 全部相位，
每个相位 `phase_*()` + 对应 `*_protocol` 冻结节）。其余 `run_v1_*.py` 为独立专项 runner。
报告纪律：**不覆盖**——新实验写新 section，旧 verdict 不追溯修改。

---

## 3. 已实现且已验证的（按时间线）

| 实验 | 结论 | 含义 |
|---|---|---|
| G2 | Runtime-owned binding 的 patch 提案阶段一次通过 | 结构化绑定可行 |
| G3/P4 | 两次自由文本 patch 均"修复部分+引入回归"→ `PATCH_REJECTED` | **自由文本 Guidance 修改 family 已关闭** |
| TSEM | rev6 vs rev7 **同期配对 AB/BA** → `TARGETING_SEMANTICS_CAUSAL_EFFECT` | rev7 的 targeting 语义修正有真实因果正效应；**配对同场是唯一能抗模型漂移的验证方式**（rev6 基线跨时间从 3/4 漂到 0/4） |
| FULLOP2 | 24/24 闭链 | rev7 稳定引导 Fast Agent 生成合法 Workflow（**构造能力已成立**） |
| USEL | 同一 outlier Workflow 跨 Context 正负翻转 | 合法 ≠ 有效；适用性必须 Context 条件化 |
| N4→N4v2 | 字段级暴露审查发现 **registry 滞后 566 支**（certified_virgin 但有消费证据）；traffic 合格池 778→92；T635 排除 | Fresh 纪律现在可信（`exposure_overrides` 追加式入档，冻结 registry 未动） |
| N5v2 | `N5_WIRING_OK` | 增长态预检接线闭环 |
| A5/A3 v1 | `NEGATIVE_TRANSFER` 但 **Runner 手工模拟 Skill 生命周期 + T635 已暴露** → 降级 `EXPOSED_DEVELOPMENT_NEGATIVE_SIGNAL` | 无效实验，不作科学结论 |
| A5v2 | 真生命周期 + sealed uci roster（MT_281/212/161）：A5 唯一形成 reliable Skill（A3 全弃权），但 harm 3>1 → `NEGATIVE_TRANSFER`（安全否决） | 记忆驱动行动，但多踩伤害面 |
| **A5v3（最新，2026-08-16）** | `NEGATIVE_TRANSFER`：A5 增量 harm 5 vs 2 安全否决；**三维信号**：q1 `SOURCE_ACTIONABILITY_POSITIVE`（A5: 3 winners/2 skills，A3: 0）、q3 `NO_DURABLE_SKILL`（两臂终态 reliable 均 0，A5 有 1 张被 removal——**撤销路径已存在**） | Source 让 Agent"敢动"是真的；动得对不对、留不住，是当前缺口 |

另有一套受控侧正向闭环（P0/P1.5 等在项目 A 完成；本仓库的受控批 witness 链见 `run_v1_s2*_nn5_controlled_chain.py`），
证明"信息充分 + Runtime 实测选择"时进化机器本身能走通。

---

## 4. 当前瓶颈（按承重排序）

### ① 主里程碑三次尝试均未形成有效判定
v1 协议无效（暴露+假生命周期）；v2/v3 协议有效但安全维度否决。**注意**：v3 的 verdict 规则是
旧 Gate 语义（任一增量 harm 更严即否）；已定的 Gate 语义改革（见 `docs/` 后续方案：delayed 改单侧
harm-veto + 材料性阈值）下，v3 的部分 harm 是否构成否决需要**新语义校准后**才知道——
但不允许追溯改判，只能在下一版协议里体现。

### ② 测量信号：support 级被 origin 噪声主导，delayed 级疑似有 series 级结构（待确认）
- support 与 delayed 单元级符号一致率仅 47%（26/55）。
- 探针（`../SelfEvolvingHarnessTS-deepseek/evaluation/functional/probe/`，项目 A 侧）证明：
  support 标签下 0 条可用谓词；delayed 标签下唯一候选 `missing_fraction==high` **过不了
  series 级置换检验（p=0.195，基率 0.567）**——是噪声。
- delayed 的 series 级过散检验 p=0.018（support p=0.727）提示**真实的 series 级 propensity
  可能存在**，但该 p 值的零模型（单元级置换）未控制 origin-block 相关——**需 origin 分层
  置换复核 + 跨 seed 稳定性检验**两个廉价前置后才可当真。

### ③ 归因粒度不到"可行动故障面"
- `group_fault.py:74` 按 (workflow 指纹, sign) 分组，不是按 fault family / 可修改 Surface。
- `fault_routes.json` 25 code → 17 个通用文本 PATCH，看似精细实际不约束修改范围。
- 已定方向：归并为 6 个固定 Fault Family（Observation / Program / Control / Scope-Risk /
  Memory / Update Policy）+ 2 个不触发态（INSTRUMENT_BLOCKED / UNIDENTIFIABLE）。

### ④ Observation 缺口是唯一"不可自修"的路由
`OBSERVABLE_FEATURE_SCHEMA_GAP = OBSERVATION_CAPABILITY_BACKLOG`。现有特征词汇表无法描述
"哪类数据适用哪个算子"。唯一带 (context × operator) 交互的候选特征族是
`probe_direction`（minipipe 已现成，未接到自然线）。已定方向：把探针改造成**特征筛选台**
（候选特征 → 与 12 维 series propensity 秩相关 + 置换 p → 确定性准入），使 Observation 面可自修。

### ⑤ Skill 生命周期还有尾巴
a5v3 已有 removal（1 张被撤销），但"delayed 变害 → 自动降级/收缩 scope"的完整 Update Policy
分支尚未系统审计（P0′ 待办）。

---

## 5. 纪律与禁忌（违反会毁掉实验可信度）

1. **反过度工程**（根 AGENTS.md §1）：不建通用 SHA/Hash 体系、Ledger、Registry、大型测试矩阵；
   每实验默认最多一个 runner package + 一个主报告 + 一个必要 smoke/integration 测试。
2. **Fresh 纪律**：`context_exposure` / `outcome_exposure` 语义必须维护；已暴露 context 不得
   再称 SEALED；冻结 verdict 不追溯修改，只追加新 section。
3. **LLM 不批准自己的 patch**：批准权在确定性 compiler + replay + in-domain feedback。
4. **配对同场验证**：patch 验证必须同期 AB/BA；禁止引用历史基线数字当裁判（模型会漂移）。
5. **委派深度 = 1**：只有根 Agent 能建子 Agent；子 Agent 任务必须有界、携带本纪律。
6. **delayed 语义改革方向已定但未校准**：从"第二道收益证明"改为单侧 harm-veto；
   在新 Gate 用旧数据校准通过（TSEM 正控过、已知回归负控拒）之前，不跑新 fresh 实验。
7. 预算/承载指标口径：`feedback_to_reliable_local_skill`（Episode 计数 ≠ Skill 计数）；
   指标截尾规则需预注册。

---

## 6. 下一步计划（已定方向，按序）

```text
P0′ 审计 a5v3 的 delayed→removal/demotion 路径 + 25→6 fault family 映射（只读/追加，零新 outcome）
P0″ overdispersion 的 origin 分层置换复核 + propensity 跨 seed 稳定性（前置，决定数据轴生死）
P1  Batch Draft Gate 量化冻结 + 旧数据校准（正控 TSEM / 负控已知回归）—— go/no-go
P2  最小 Rule Card 切片（普通 JSON 列表 + Runtime matcher + 一种真实故障对应卡型 + 一个 e2e 测试）
P3  自然 Batch Local Evolution → NATURAL_BATCH_LOCAL_EVOLUTION_PASS（第一例自然正向的定义）
P4  fresh A5/A3（需 N4v2 重新冻结 SEALED roster；uci 已暴露；两臂共享新 Gate，唯一变量 Source Pack）
探针线（与 P2 并行候选）：fixed_probe_panel 接自然线（只跑 public view）→ 特征筛选台评分
```

---

## 7. 上手操作

```bash
cd SelfEvolvingHarnessTS-deepseek-guidance-evolution
python -m pytest tests/functional -x -q   # 最近状态：135 过 / 1 已知 f1 失败
```

- 主报告：`artifacts/functional/e2/w1_guidance_evolution_report.json`（按 section 读实验史，
  每相位有对应 `*_protocol` 冻结节，先读 protocol 再读结果）。
- 可视化：`artifacts/functional/e2/w1_evolution_dashboard.html`。
- 主 runner：`evaluation/functional/run_v1_guidance_evolution.py`（相位函数 `phase_*`）。
- 冻结 registry：`artifacts/frozen/benchmark_v02/`（勿改；暴露修正在主报告 `n4_v2.exposure_overrides`）。
- 测试清单即实验清单：`tests/functional/test_{a5,a5v2,a5v3,tsem_targeting,fullop2_verdict,
  n4,n4v2,n5_growth,skill_revocation,usel,...}.py`。

## 8. 一句话给新 Agent

> 机器已建好且安全（会提案、会拒绝、不污染 Memory）；受控正向已证明机器可读强信号；
> 自然侧还没赢过——瓶颈不在"提案质量"，在**测量分辨率、Observation 词汇表和 Gate 语义**
> 三件事上。你的每一个改动都应先问：它是否直接推进这三者之一；不是，就缓。
