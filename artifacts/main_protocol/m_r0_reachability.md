# M-R0 · 修订可达性审计 + 计费核查

**stage** `M_R0_REVISION_REACHABILITY_AUDIT` · **evidence grade** `INSTRUMENT / MECHANISM`
**机器可读工件** `artifacts/main_protocol/m_r0_reachability.json`
**复算脚本** `evaluation/main_protocol_p4/audit_m_r0_reachability.py`

```
python -m evaluation.main_protocol_p4.audit_m_r0_reachability
git diff d690850 -- evaluation/main_protocol_p4/outer_loop.py evaluation/main_protocol_p4/run_hec1.py   # 必须为空
```

**边界**：0 LLM、0 Consumer fit、0 held-out 读、0 密封读、0 阈值改动、0 代码改动、
0 既有工件写入、0 新增 SHA/Hash。只诊断，不修复，未启动 M-W，未开新课程。

## 0. 版本纪律

| 项 | 值 |
| --- | --- |
| 收据声明的运行 commit | `d690850be1cb70d1da5f7034dbb097f42ba42e36`（三顺序 `code_state.code_commit` 一致，`runner_files_clean=true`） |
| 当前 HEAD | `f9f96d7cb8d424a81105ddff3a6bde92c7a2253e` |
| `d690850 → HEAD` 的非工件改动 | 只有 `.gitignore` 与 8 份新增 docs；**无一行运行期代码** |
| 9 个相关运行期文件 | 逐个 `git diff d690850 -- <file>` 为空，且工作树干净 |

因此本报告全部用运行期代码解释历史行为；未切换工作树、未撤销他人改动、未用当前
修复后行为解释历史运行。

---

## 1. 任务一：M-R0 修订可达性

### 1.1 结论一句话

`NARROW` 在 30 个外环步中**一次都不可能触发**，原因既不是撤销、也不是预算，而是
**普查读到的 relation 与它统计的词表不是同一套**：`adverse_units` 和
`positive_units` 在全部 30 步、全部 arm、全部 group 中恒为 0。

### 1.2 阻断链（逐行代码事实）

| # | 位置 | 事实 |
| --- | --- | --- |
| 1 | `methods/ttha/admission_policy.py:69-82` | `AdmissionVerdict` 的字段是 `admitted / rule / reason / aggregate_gain / series_count / harmed_count / harmed_fraction / max_single_series_harm`，**没有 `relation`** |
| 2 | `methods/ttha/online_loop.py:891-894` | 该 verdict 的 `to_dict()` 写入 `probe["admission"]` |
| 3 | `run_hec1.py:1849` | `row["relation"] = probe["admission"]["relation"]` —— 键不存在，**恒为 `None`** |
| 4 | `run_hec1.py:1850` | `row["admission"] = probe["admission"]["reason"]`，取值是 `relation_positive` / `within_risk_budget` / `harmed_fraction_over_budget` / `single_series_harm_over_budget` / `aggregate_below_material_line` |
| 5 | `outer_loop.py:209-211` | `_relation` 返回该 reason 的大写形式，**永不进入** 它自己在 `:212-220` 的按增益回退分支 |
| 6 | `outer_loop.py:363-365` | `positive_units` 只数 `"POSITIVE"`，`adverse_units` 只数 `"CONFLICT" + "NEGATIVE"`；上面五个字符串一个都不是 |
| 7 | `outer_loop.py:416-417` | `ADD` 需要 `positive_units >= 1` → 恒不成立 |
| 8 | `outer_loop.py:445-446` | `NARROW` 需要 `adverse_units >= 2` **且** `key in held` → 前一半恒不成立 |

工件侧的直接证据：30 个外环步、全部 group 的 `positive_units = 0`、
`adverse_units = 0`，而 `relation_counts` 的键正是上面那五个 reason 名
（例：Forward k5 A5-online 的 `outlier_mad` group =
`{AGGREGATE_BELOW_MATERIAL_LINE: 7, HARMED_FRACTION_OVER_BUDGET: 2, RELATION_POSITIVE: 3, SINGLE_SERIES_HARM_OVER_BUDGET: 9, WITHIN_RISK_BUDGET: 5}`）。

**校验**：审计从 cells 里的 probe 行按 `_bank_rows_from_round`
（`run_hec1.py:1830-1855`）重建 bank，再用 `outer_loop.census_key` 重放普查。
30/30 个外环步的 group 集合、`relation_counts`、`unit_count`、`positive_units`、
`adverse_units` **逐项复现**记录值（`census_replay_reproduces_every_recorded_group = true`），
所以下面的反事实是在同一份被验证过的 bank 上做的。

**反事实**（把 relation 字段留空、让 `_relation` 走它自己的增益定义）：

| 量 | 实际记录 | 反事实 |
| --- | --- | --- |
| `ADD` 候选 | 0 | 6 |
| `NARROW` 候选 | 0 | **26** |
| 会提出 `NARROW` 的外环步 | 0 / 30 | **22 / 30** |

### 1.3 **不是**阻断原因的三件事（逐条排除）

| 候选解释 | 判定 | 证据 |
| --- | --- | --- |
| `key in held` 不成立 | **排除** | K0 收据 `hec1_k0_phase_s_v11_live.json` 的 `lineage_keys` 唯一一条是 `forecast\|pooled-ridge-a1\|sMASE\|outlier_mad({})@serving_series_predicate[local_robust_z_peak>=3]`，在课程开始就被 `seed_active_programs`（`run_hec1.py:2513-2515`）注入两个 A5 臂；它与 bank 生成的 census_key **逐字节相同**。held 测试在每一步都会通过 |
| 部署权先丢了，谱系轮不到外环 | **排除** | 9 条 (顺序 × LLM 臂) 的 `snapshot_skill_ids_at_start` 计数序列**没有一次下降**；`revocation_recommendations` 全为 0；`propose_candidates`（`outer_loop.py:406-470`）只产出 ADD / NARROW / REVISE，`consolidate` 里的 `REVOKE` 分支（`:732`）在本协议中**不可达** |
| 外环 LLM 预算 | **仅对已发生的 3 个 REVISE 有效** | `OUTER_LLM_PER_STEP = 2` 只在 REVISABLE-Draft 路径产出候选后才起作用；ADD / NARROW 连候选都没形成，预算从未被它们触及 |

> 计划 §4.3 的预设「若 M-R0 证实『撤销先于外环步』确是 0 NARROW 的原因」——
> **该假设不成立**。本次运行没有任何撤销发生。

### 1.4 第二重、独立于词表的阻断

即使词表被修好，`NARROW` 仍读不到人所说的「冲突」：

- `_bank_rows_from_round`（`run_hec1.py:1836`）**只收 `kind == "probe"` 的行**；
- delayed 面是权威门、不是证据，从不写入 bank（`evaluation_face_enters_bank = 0`，
  `boundary` 同证）。

所以普查看到的是 **Support 探针的准入结果**，而 11 条 Active 谱系上真正的 adverse
读数全部发生在 **delayed 面**。§1.2 的反事实 26 个 `NARROW` 是「Support 证据本会
产生的候选」，**与生命周期记录的 delayed 冲突不是同一个对象**——这一点必须在
M-W / 闭环接线时分开处理。

### 1.5 逐臂 Active 谱系重建（11 条）

「Active」= 该 arm 在 cell 起点持有该程序的卡（`active_program_signatures_at_start`）。
`riskadv` = 持权重部署后 delayed 门因风险线失败的次数；`covadv` = 只失败 coverage_floor
（`classify_failure` 判 `WAITING`，按定义**不是** Skill 失败）。

| 顺序 | 臂 | 程序 | 卡来源 | 激活位置 | 持权重部署 | riskadv | covadv | pass | ≥2 riskadv？ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forward | A5-frozen | `outlier_mad` | K0（每单元重建） | — | 5 | 4 | 1 | 0 | 是 |
| forward | A5-online | `outlier_mad` | K0 | p05 再激活 | 8 | 4 | 2 | 2 | 是 |
| forward | A3-online | `outlier_mad` | 课内激活 | p05 | 4 | 2 | 2 | 0 | 是 |
| reverse | A5-frozen | `outlier_mad` | K0（每单元重建） | — | 6 | 4 | 0 | 2 | 是 |
| reverse | A5-online | `outlier_mad` | K0 | p20 再激活 | 5 | 3 | 0 | 2 | 是 |
| reverse | A5-online | `winsorize` | 课内激活 | p06 | 1 | 0 | 1 | 0 | 否 |
| reverse | A3-online | `winsorize` | 课内激活 | p23 | 0 | 0 | 0 | 0 | 否 |
| interleaved | A5-frozen | `outlier_mad` | K0（每单元重建） | — | 7 | 4 | 1 | 2 | 是 |
| interleaved | A5-online | `outlier_mad` | K0 | — | 7 | 3 | 2 | 2 | 是 |
| interleaved | A3-online | `outlier_mad` | 课内激活 | p18 | 0 | 0 | 0 | 0 | 否 |
| interleaved | A3-online | `outlier_iqr` | 课内激活 | p25 | 0 | 0 | 0 | 0 | 否 |

**8 / 11 条谱系在 delayed 面上达到了 `MIN_ADVERSE_UNITS_FOR_NARROWING = 2` 的实质条件**，
但这些读数按 §1.4 从不进入决定 `NARROW` 的普查。

**部署权丢失时刻**：**没有**。9 条 (顺序 × 臂) 序列全程无下降，0 条撤销建议，
`REVOKE` 分支不可达。A5-frozen 的「重建」是 `begin_unit`（`run_hec1.py:1729-1736`）
对冻结臂的按单元重建，不是撤销。

### 1.6 同名 skill_id ≠ K0 泄漏（按要求单独核）

`_fast_winner_skill_id`（`methods/ttha/method.py:261-279`）是
`(task_type, downstream_model_class, metric, operator)` 的确定性、无哈希函数。HEC-1
把前三项固定，因此**任何**臂学到 `outlier_mad` 都会铸出同一个字符串。

| skill_id | 独立铸造 | 判定 |
| --- | --- | --- |
| `fast_winner_forecast_ridge_smase_outlier_mad` | forward A3-online p05；interleaved A3-online p18；同时也是 K0 卡的 id | **NAMING_COLLISION_NOT_LINEAGE_IDENTITY** |
| `fast_winner_forecast_ridge_smase_winsorize` | reverse A5-online p06；reverse A3-online p23 | **NAMING_COLLISION_NOT_LINEAGE_IDENTITY** |
| `fast_winner_forecast_ridge_smase_outlier_iqr` | interleaved A3-online p25 | 单次铸造 |

独立反证：A3 臂从 h0 起，自身首次激活之前每个 cell 的
`snapshot_skill_ids_at_start` 都是空（forward 到 p05、interleaved 到 p18、
reverse 到 p23）；铸造逐 cell 记在 `skills_minted_this_unit`；
`hec1_k0_freeze_phase_s_v11_live.json` 的 `a5_a3_isolation` 检查以
`leaks_into_non_k0_arms = []` 通过。**未观察到 K0 泄漏。**

### 1.7 11 张 Draft 与 3 次 REVISE 的重建

Draft 由 `run_hec1.py:2121`（`if not gate["passes"] and scope:`）产生，即
「过 Support 准入 → 部署 → 败 delayed」。审计把每个 `history` 事件按
(臂, window, failed_lines) 唯一匹配回具体 cell（33 个事件，**0 个歧义**）。

| 顺序 | 臂 | draft | 创建程序 | 创建于 | 由持权卡产生？ | 终态 | 尝试 | 修订 | 关闭原因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forward | A5-online | d1 | `outlier_mad` | p01 / o1896 | **是**（K0） | FLAGGED | 3 | 0 | EFFECT_NONSTATIONARY |
| forward | A5-online | d2 | `outlier_mad` | p16 / o1176 | **是** | FLAGGED | 3 | 0 | EFFECT_NONSTATIONARY |
| forward | A3-online | d1 | `outlier_mad` | p10 / o2136 | **是**（课内卡） | FLAGGED | 3 | 0 | EFFECT_NONSTATIONARY |
| forward | A3-online | d2 | `outlier_mad` | p23 / o1176 | **是** | FLAGGED | 1 | 0 | EFFECT_NONSTATIONARY |
| reverse | A5-online | d1 | `outlier_mad` | p02 / o1176 | **是**（K0） | FLAGGED | 3 | 0 | EFFECT_NONSTATIONARY |
| reverse | A5-online | d2 | `outlier_mad` | p24 / o1896 | **是**（K0） | REVISABLE | 1 | 0 | REVISION_BUDGET_EXHAUSTED |
| reverse | A3-online | d1 | `outlier_mad` | p02 / o1176 | 否 | FLAGGED | 3 | 0 | EFFECT_NONSTATIONARY |
| interleaved | A5-online | d1 | `outlier_mad` | p02 / o1176 | **是**（K0） | FLAGGED | 3 | 0 | EFFECT_NONSTATIONARY |
| interleaved | A5-online | d2 | `outlier_mad` | p13 / o2136 | **是**（K0） | FLAGGED | 2 | 0 | EFFECT_NONSTATIONARY |
| interleaved | A3-online | d1 | `outlier_mad` | p03 / o1176 | 否 | FLAGGED | 3 | 0 | EFFECT_NONSTATIONARY |
| interleaved | A3-online | d2 | `outlier_iqr` | p19 / o2616 | 否 | FLAGGED | 1 | 0 | EFFECT_NONSTATIONARY |

**对计划 §1「11 张 Draft……无一来自曾 Active 的 Skill」的更正：8 / 11 张 Draft
由该 arm 在该 cell 起点仍持有部署权的卡的 delayed 失败产生**（其中 6 张直接来自
K0 卡）。Draft 与 Skill 卡在记账上分离（`census_key` 全为 `None`），但来源上不分离。
这会改变 §4.2 里 Chain-D 与 Chain-S 的分母划分。

**11 张 Draft 无一被修订**：`revisions = 0`，`root_scope == current_scope`。原因分层：

- **10 张锁死在 `FLAGGED`**：`may_add_clause()`（`restricted_draft.py:271-276`）明令
  FLAGGED 不得加子句，`propose_candidates`（`outer_loop.py:474-490`）把它们记成
  `EFFECT_NONSTATIONARY_CANDIDATE` 漂移信号而非候选。这是**合法弃权/机制拒绝**，
  不是预算或解析失败。
- **1 张 REVISABLE（reverse d2）** 在 p24 创建，课末以 `REVISION_BUDGET_EXHAUSTED`
  关闭——它确实成了 reverse k5 的 REVISE 候选，但该候选在拿到子句前耗尽了外环预算。

三次 REVISE 逐次拆解（**均来自 REVISABLE Draft，无一来自 Active 卡的 `NARROW`**）：

| 顺序 | k | 记录结论 | 物理调用 | 到达阈值工具的提议 | Slow 选的 (feature, direction) | 影子结论 | 达到 `CALIBRATED` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward | 1 | `SLOW_ABSTAINED` | 2 | 1 | `outlier_region_end_fraction >` | `BEST_STUMP`（feasible=8） | 0 |
| forward | 2 | `OUTER_LLM_BUDGET_SPENT` | 2 | 2 | `outlier_region_end_fraction >=`；`missing_fraction <=` | `BEST_STUMP`×2（feasible=1） | 0 |
| reverse | 5 | `OUTER_LLM_BUDGET_SPENT` | 2 | 2 | `level_excursion_score <=`；`level_excursion_score >=` | `NO_FEASIBLE_STUMP`×2 | 0 |

分类（按要求区分，不照抄摘要标签）：

- forward k1 = **1 次提议被阈值工具拒绝 + 1 次 Slow 真正弃权**（第 2 次调用返回
  `None`，故未产生影子记录）。计划把它整体记作「Slow 弃权」是**部分正确**：两次
  物理调用里只有第二次是弃权。
- forward k2 / reverse k5 = **两次提议都被阈值工具拒绝，第 3 次尝试被
  `outer_llm_per_step = 2` 在后端前挡下**。reverse k5 的两次提议连全词表最优
  stump 都无可行阈值（`NO_FEASIBLE_STUMP`），属于「**在冻结 bin 边缘上无可行修改**」，
  与「预算耗尽」是两个叠加原因，摘要标签只显示后者。
- **无一例属于解析/协议失败**（`SLOW_CLAUSE_UNUSABLE` / `MALFORMED` 均未出现）。
- 每次尝试内部的工具结论（`NO_FEASIBLE_THRESHOLD` 还是 `SLOW_CLAUSE_UNUSABLE`）
  **未持久化 → UNKNOWN（U-1）**，工件只保留候选的最终结论与影子。

### 1.8 三个附带的仪器缺陷（只报告，不修）

1. **跨程序污染（1 例，已证实）**：`DraftLedger.by_scope`
   （`restricted_draft.py:547-575`）只按 `current_scope` 匹配、不比对程序。
   reverse A5-online 的 `winsorize` 在 p15 失败后，被记成 `outlier_mad` 的
   `resupplied_draft_1` 的**第 3 次也是最后一次**验证，并把它关闭为
   `EFFECT_NONSTATIONARY`。11 张 Draft 中另有多张共用同一 root Scope，只是恰好
   没有第二个程序落在同一窗口。
2. **`restrict()` 不写 `census_key`**（`restricted_draft.py:331-360`）：11 张 Draft
   的 `census_key` 全为 `None`，因此 `ledger.lineage_keys()` 恒为空集，`held` 只由
   Active 谱系构成；同一 (程序, root Scope) 关闭后可以再开一张新壳并把
   `verification_attempts` 归零——forward A5-online 和 A3-online 各自的 d1/d2 就是
   同一 census key 的两张壳，合计各拿到 4–6 次验证面，超过 `MAX_VERIFICATION_ATTEMPTS = 3`
   的设计意图。
3. **冻结臂的 Draft 被按单元丢弃**：`begin_unit`（`run_hec1.py:1729-1736`）为
   A5-frozen 每单元重建 ledger，`dropped_drafts` 合计 **15**（6 / 4 / 5）。
   生命周期表里的 11 张只覆盖两个 online 臂；A5-frozen 的 15 次同类失败没有留下
   Draft 记录（cell 上的 `restricted_state` 仍在）。

### 1.9 修订被哪些条件挡住 · 实例计数

| 阻断条件 | 影响面 | 实例数 |
| --- | --- | --- |
| `CENSUS_RELATION_VOCABULARY_MISMATCH` | ADD + NARROW，全部外环步 | **30 / 30 步**（反事实下会有 26 个 NARROW、6 个 ADD） |
| `ADVERSE_EVIDENCE_NEVER_ENTERS_THE_BANK` | delayed 冲突对普查不可见 | **8 / 11 条**谱系已满足 ≥2 次风险线 adverse，全部不可见；33 个 Tier-2 实例 |
| `FLAGGED` 禁止加子句 | Draft 侧的 REVISE | **10 / 11 张** Draft |
| 外环 LLM 预算 + 阈值工具无可行边缘 | 已成立的 REVISE 候选 | **3 / 3**（1 弃权+1 拒绝、2×2 拒绝后预算挡） |
| `REVOKE` 不可达 / 无撤销 | 谱系保留窗口 | **0** 次撤销，非阻断因素 |
| 冻结臂 ledger 重建 | Draft 证据供给 | **15** 张被丢弃 |

**已证明**：上表全部六行都有代码行 + 工件双证据。
**仍未知**：见 §5 的 U-1 … U-5，特别是——**即便修好词表也不足以产生一次修订**：
Slow → 阈值工具在 5 次真实尝试里 `CALIBRATED` 次数为 **0**，`NARROW` 走的是同一条
`_clause_for` 路径。

---

## 2. 任务二：案例身份与外环调用成本

### 2.1 失败情境的完整身份

保留 cohort、origin、delayed origin、Program、Scope、臂、顺序后，
**去重后的失败情境是 8 个，不是 4 个**。全部 Scope 均为
`serving_series_predicate[local_robust_z_peak >= 3.0]`。

| # | cohort block | origin → delayed | Program | 失败线 | treated | 实例 | 顺序 | 臂 | Tier | Workflow 编辑能否动它 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-a | `[0:40]` | 1896 → 1944 | `outlier_mad` | hf + ssh | 13 | 7 | F/R/I | A5f, A5o, A3o | Tier-2 | **能** |
| S-b | `[120:160]` | 1176 → 1224 | `outlier_mad` | ssh | 15 | 9 | F/R/I | A5f, A5o, A3o | Tier-2 | **能** |
| S-c | `[40:80]` | 2136 → 2184 | `outlier_mad` | coverage_floor | 3 | 5 | F/R/I | A5o, A3o | Tier-2 | **不能** |
| S-d | `[40:80]` | 2136 → 2184 | `winsorize` | coverage_floor | 3 | 2 | F/R | A5f, A5o | Tier-2 | **不能** |
| S-e | `[40:80]` | 2616 → 2664 | `outlier_mad` | hf | 12 | 6 | F/R/I | A5f, A5o, A3o | Tier-2 | **能** |
| S-f | `[40:80]` | 2616 → 2664 | `outlier_iqr` | hf + ssh | 12 | 1 | I | A3o | Tier-1 | **能** |
| S-g | `[80:120]` | 1176 → 1224 | `outlier_mad` | coverage_floor | 3 | 5 | F/I | A5f, A5o, A3o | Tier-2 | **不能** |
| S-h | `[80:120]` | 2376 → 2424 | `outlier_mad` | ssh | 12 | 6 | F/R/I | A5f, A5o | Tier-2 | **能** |

**与计划附录 D.1 的差异（须交 Fable 更新）**：

- D.1 只按 **origin** 去重，把 origin 1176 的两个**读数完全相反**的 cohort 合成一个
  「S1」：`[120:160]` 是 treated 15、最坏单序列伤害 0.485 的风险线失败；`[80:120]`
  是 treated 3、只差 coverage_floor 的等待态。二者不能当同一个情境买。
- **`[80:120] 2376 → 2424`（S-h，6 个实例、三顺序都有）在 D.1 表中完全缺席。**
- D.1 把 2616 → 2664 记作 `outlier_iqr`；记录里该窗口的 `outlier_mad` 失败有 6 个实例，
  `outlier_iqr` 只有 1 个。
- D.1 的「`outlier_mad` 10 / `outlier_iqr` 1」是 **Draft 创建时的程序**计数，不是失败
  情境计数。

### 2.2 独特情境 / 实际实例 / 重复观测 / 离线反事实复用（分开计）

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| 独特失败情境 | **8** | (cohort, origin, delayed, Program, Scope) |
| 实际发生实例 | **41** | 逐 (顺序, 臂, 位置) 的 delayed 门失败 |
| 重复观测 | **33** | 同一情境的每个实例 delayed 读数**逐位相同**（treated / aggregate / hf / msh 全等），是同一确定性评估按臂、按顺序重算，不是新证据 |
| (情境 × 顺序) 组合 | **20** | 计划假设的「4 × 3 = 12」既非上界也非下界 |
| — 其中只差 coverage_floor | 7 | 结构性不可修（见 §3.3） |
| — 其中 Workflow 编辑可动 | 13 | |
| — 且后续 ≥2 个可评单元 | **11** | M-W 必做预算的推荐分母 |
| 离线反事实复用 | **2013 fits** | BSG 1046 + validation-search 899 + validation-transfer 68；全部是已处理 cell 的重放，不是新的 held-in 反馈 |

### 2.3 Tier-1 / Tier-2 清单

**Tier-2 定义**（本审计采用）：cell 起点该 arm 持有该程序的 Active 卡，且 delayed 门失败。

| 口径 | 实例数 |
| --- | --- |
| Tier-2 实例（全部臂） | **33** |
| Tier-2 实例（仅 online 臂） | **19** |
| Tier-2 实例（online 臂 + 仅风险线，即真正的「冲突」） | **12** |
| Tier-2 情境 | **7 / 8**（唯一非 Tier-2 是 S-f `outlier_iqr`） |

**关键事实：Tier-1 与 Tier-2 不是两批案例。** 8 个情境里 7 个既是 Draft 来源又是
Active 卡冲突。M-W 若按附录 D.5 把 Tier-2 当成「每案 90–100 fits 的追加」，会把
Tier-1 已买的同一批 (cohort, origin, Program, Scope) 面再买一遍。**Tier-2 真正额外
的成本只有祖先影子对照（v1 在同面同序列重算），不是第二轮候选扫描。**

**M-W 可用单元（后续步骤的推荐输入）**：11 个 (情境 × 顺序) 组合 —— 由
S-a、S-b、S-e、S-f、S-h 五个可修情境展开，剔除 reverse 的 S-a（p24，后续只剩 1 个
可评单元）与 forward 的 S-b（p23，同）。逐格清单见 JSON 的
`situations.situations[*].per_ordering`。

### 2.4 外环调用成本（从既有 Slow 记录核对，未新增任何调用）

| 量 | 值 |
| --- | --- |
| 到达 Slow 的外环步 | 3（forward k1、forward k2、reverse k5，均为 A5-online） |
| 物理调用 | **6**（每步 2，等于 `OUTER_LLM_PER_STEP`） |
| 每次尝试的物理成本 | **恰好 1 次**（`outer_loop.py:643-645`） |
| 到达阈值工具的提议 | 5（第 6 次调用是 forward k1 的 Slow 弃权，未产生工具调用） |
| 返回 `CALIBRATED` 的次数 | **0** |
| 所属提议 | 全部 3 个候选均为 `REVISE`、程序 `outlier_mad({})`、来自 REVISABLE Draft |

> **完整提议成本尚未测得。** 整个课程没有任何一次结构提议走到 `CALIBRATED`，因此
> 没有一个「完整提议」可以定价。**不得**由「两次调用后耗尽」推定「四次就够」。
> 已测得的只有：一次尝试 = 一次物理调用；一步在第 2 次后停止。

**外环预算的内部矛盾（新发现）**：`OuterBudget.retries_per_candidate = 2`
（`outer_loop.py:91`）意味着最多 3 次尝试，但同一 dataclass 的
`outer_llm_per_step = 2`（`:87`）在 `_clause_for` 的循环头（`:638`）就把第 3 次挡掉。
**第 3 次重试按构造不可达，写在合同里的重试额度永远花不出去。**

---

## 3. 任务三：计费核查（独立章节）

### 3.1 LLM 调用：按类别分开

| 类别 | 是否记录 | 合计 |
| --- | --- | --- |
| 尝试（`reserve`） | 记录但不计数（`reserve` 不增任何计数器） | — |
| 实际发送并成功返回（Fast，逐 cell 记录） | `llm_calls_this_cell` | **847** |
| 实际发送但整格因 cap 抛错返回（Fast） | **未记录**，由 cap 语义推导 | **235**（47 格 × 5） |
| 实际发送（外环 Slow） | `slow_calls` / `llm_outer` | **6** |
| 后端前拦截 | `budget_guard.blocked_before_backend` | **0**（三顺序均空） |
| 重试 | Fast 的 schema 纠正重试已含在 `llm_calls_this_cell`；外环重试逐次经 `_MeteredOuterBackend` 入账 | — |
| **已记账**（`llm_total`） | | **666** |
| **物理总数** | | **1088** |
| **差额** | | **422** |

逐顺序：

| 顺序 | 账面 | 物理 | 差额 | 其中 `spent−1` | 其中 cap 格 | 顺序上限 | 是否超限 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forward | 241 | 363 | 122 | 67 | 11 × 5 = 55 | 500 | 否 |
| reverse | 205 | 367 | 162 | 57 | 21 × 5 = 105 | 500 | 否 |
| interleaved | 220 | 358 | 138 | 63 | 15 × 5 = 75 | 500 | 否 |
| **合计** | **666** | **1088** | **422** | **187** | **235** | | |

### 3.2 差额的两个来源（完全解释，未凑数）

1. **`spend(calls=spent − 1)`（`run_hec1.py:1970`）= 187 次。**
   注释写「the reserve took one」，但 `BudgetGuard.reserve`（`run_hec1.py:205-217`）
   只做上限检查、**不增加任何计数器**。因此每个「至少发出 1 次调用」的 cell 都少记
   恰好 1 次。少记数 187 **恰好等于**这类 cell 的数量（187），机械吻合。
2. **整格 cap 抛错未入账 = 235 次。**
   47 个 cell 的 fault 为 `AgentCallBudgetExceeded: Agent call budget exhausted at 5`。
   `BudgetedAgentBackend.complete`（`runtime/agent_backend.py:450-455`）在
   `calls >= maximum_calls` 时抛出，即此刻已实际发出 **5** 次；
   `run_hec1.py:1954-1966` 捕获后**在 `:1970` 的 `guard.spend` 之前 return**，
   这 5 次一次都没入账。47 格与收口文档「47 无决策（A3-online 21 / A5-frozen 9 /
   A5-online 17）」的格数一致。

`666 + 187 + 235 = 1088`。与
`docs/HEC1_CLOSEOUT_DESIGN_RULING_2026-09-04.md:43` 的「物理 1088 vs 账面 666、
少记 422」**完全一致**，且本审计给出了此前只给总数的逐项分解。

**边界性说明**：235 是**推导值**而非记录值（U-2）。若那 47 格中有任何一格实际少于
5 次 relay，则 1088 是上界，下界为 **853**（= 666 账面 + 187 次 `spent−1`）。
三顺序物理数在两种口径下均 < 500 的顺序上限，**效用判词不受影响；但
`llm_total` 不得用于任何成本/效率主张**。

### 3.3 Consumer fits：Support 探针 vs delayed/evaluation vs replay vs 影子

| 类别 | 来源 | forward | reverse | interleaved | 合计 |
| --- | --- | --- | --- | --- | --- |
| delayed + evaluation（`course_fits`） | **记录** | 183 | 173 | 186 | **542** |
| Support 探针 | **推导，未记录** | 212 | 207 | 222 | **641** |
| replay 屏（`replay_fits`） | 记录 | 0 | 0 | 0 | **0** |
| 影子对照（`shadow_fits`） | 记录 | 0 | 0 | 0 | **0** |
| baseline（`baseline_fits`） | 记录 | 0 | 0 | 0 | **0** |

- **Support 探针 fits 未被任何账本记录**：`SupportReceipt`
  （`methods/ttha/scope_executor.py`）没有 `consumer_fits` 字段，`ledgers.course_fits`
  只在 `run_hec1.py:2032`（delayed）与 `:2186`（evaluation）累加。
- **推导公式**：`2 × 探针数 + 带探针的单元数`。
  `scoped_evaluate` 对 raw design 记 1 次 fit，Scope 非空使程序管线运行时再记 1 次
  （`scoped_serving_evaluator.py:200-210`）；记录在案的每个探针都解析出非空 Scope，
  故各 2 次。`ScopeExecutor._baseline`（`scope_executor.py:330-337`）按
  (executor, origin) 缓存，而 executor 按单元构造并被四个臂共享
  （`run_hec1.py:343`），故每个有探针的单元再加 1 次。
  探针数 93 / 91 / 98（另有 interleaved 1 次 `verifier_rejected`，0 fit）；
  带探针单元 26 / 25 / 26。
- `shadow_fits = 0` 是结构性的：`best_stump` 是对 bank 行的确定性搜索，不做拟合。
  `replay_fits = 0` 是因为没有任何候选走到 replay 屏。
- **HEC-1 三顺序的 Consumer fit 总量 ≈ 542（记录） + 641（推导） = 1183**，
  另有 **2013** 次离线反事实复用（BSG / validation-search / validation-transfer）。

### 3.4 附录 D 的零拟合预算预检（只检查，不定规则）

**（1）有限候选集与拟合前上界。** 对 1 步 `outlier_*` 程序，按 D.2 自己的表：
同类替换 3 + 前置 impute 7 + 顺序交换 0 + 参数改动 0（**观察到的根算子
`outlier_mad` / `outlier_iqr` / `winsorize` 无一是 `hampel_filter`**）
= **E_上界 = 10**。

**（2）效果去重必须先评价。** D.2 的去重键是逐序列增益向量，只有把候选评过才知道；
D.5 却直接用 **E = 8**（一个去重后的**预估**）计价。**预估去重数量不是成本保证；
拟合前唯一可用的界是枚举集 E = 10。**

**（3）必做 vs 可选后验穷举**（`每评分面 3 fits`、2 面、后续 2 单元、top-2，
不计缓存节省——节省不是界）：

| 口径 | 选择探针 | 验证 top-2 | 原程序对照 | +144 事后分层 | **必做合计** | 可选后验穷举 |
| --- | --- | --- | --- | --- | --- | --- |
| 计划：E=8 × 12 实例 | 576 | 288 | ≤144 | 144 | **1008 – 1152** | 864 |
| 实测分母 11 × E=10 | 660 | 264 | ≤132 | 132 | **1056 – 1188** | 1056 |
| 全部 13 个可修组合 × E=10 | 780 | 312 | ≤156 | 156 | **1248 – 1404** | 1248 |

**（4）计划中残留的预算/口径矛盾（交 Fable 统一更新）**：

| # | 位置 | 矛盾 |
| --- | --- | --- |
| D-1 | 附录 A「成本」行 vs 附录 C.9 vs 附录 D.5 | 附录 A 仍写 M-W「300–540 fits」，而 C.9 已撤回该数、D.5 改为 1050–1150；表未同步 |
| D-2 | 附录 D.2 vs D.5 | D.5 以去重后估计 E=8 计价，D.2 的去重规则却要先花掉拟合才能应用；拟合前的界是 E=10 |
| D-3 | 附录 D.1 情境表 | 仅按 origin 去重 → 4；按记录身份（cohort × origin × delayed × Program × Scope）→ 8。origin 1176 合并了两个读数相反的 cohort；`[80:120] 2376→2424` 完全缺席；2616→2664 还有 6 个 `outlier_mad` 实例 |
| D-4 | 附录 D.1「≤12 个案例实例」 | 实测 (情境 × 顺序) = 20；剔除只差 coverage_floor 的后 = 13；再要求后续 ≥2 个可评单元 = 11 |
| D-5 | §3 Gate 0 vs 附录 A 的 M-W 停止条件 vs 附录 D.4 | Gate 0 要求每被读格 ≥8 个可计分窗口；D.4 每个 (情境 × 顺序) 只有 2 个后续单元；附录 A 又按「区间半宽 ≤0.15」停止。同一测量三套充分性规则 |
| D-6 | 附录 D.4「与 live 路径同」 vs D.5「每评分面 3 fits」 | live Support 面是 `ScopeExecutor.evaluate`（`scoped_evaluate` 2 fits + 按单元缓存的 `forecast_runtime._evaluate` 基线）；「3 fits/面」是 `_policy_reading`（基线改用 `scoped_evaluate`）。两者的 Δ 不可混比，必须先定死用哪一种 |
| D-7 | 附录 D.5 Tier-2 行 | Tier-2 被当成可加成本，但它标注的是 Tier-1 已经买下的同一批窗口（7/8 情境重叠）；真正额外的只有祖先影子对照 |
| D-8 | 正文 §1 状态表 | 「11 张 Draft……无一来自曾 Active 的 Skill」与记录不符：8/11 由持权卡的 delayed 失败产生 |

**（5）本审计新发现、需计划采纳的成本约束**：

- **3 / 8 个情境（对应 7 / 20 个「情境 × 顺序」组合）只差 coverage_floor。**
  coverage_floor 由固定谓词对 raw 窗口的解析决定，M-W 不改 Scope，因此**任何**
  Workflow 编辑都不改变哪些序列入选，D.3 的规则①对所有候选一律不成立，结论在第一次
  拟合之前就已经是 `RULE_FOUND_NO_FIX`。若照买，**浪费 420 次选择探针 fits**。
- **「原程序对照 多数已记录 → 0」不成立。** D.4 需要的是原程序在**后续 2 个验证
  单元**上的读数；实测 22 个所需 (组合 × 后续单元) 中只有 **3** 个当时确实部署了同一
  程序。该行应按 ≤132（E=10、11 实例口径）全额计入，而非 0。

---

## 4. 新增文件与执行边界

| 文件 | 性质 |
| --- | --- |
| `evaluation/main_protocol_p4/audit_m_r0_reachability.py` | 新增只读解析脚本（0 LLM / 0 fit） |
| `artifacts/main_protocol/m_r0_reachability.json` | 新增（此前不存在，已核对） |
| `artifacts/main_protocol/m_r0_reachability.md` | 本文件，新增（此前不存在，已核对） |

**未做**：未运行任何会触发拟合或 API 的测试/runner；未 resume；未读密封数据；未下载；
未改代码、合同、阈值或既有工件；未提交 Git；未新增 SHA/Hash；未搭建审计平台；
未修改主计划、`STATE_ONE_PAGE_2026-09-03.md` 或任何共享账本。发现的缺陷（§1.8、
§2.1 差异、§3.4）**只报告，未顺手修复**。

---

## 5. 仍未知（不补造状态）

| # | 问题 | 状态 |
| --- | --- | --- |
| U-1 | 外环每次尝试内部的工具结论（`NO_FEASIBLE_THRESHOLD` 还是 `SLOW_CLAUSE_UNUSABLE`） | **UNKNOWN**：工件只留候选最终结论与影子 |
| U-2 | 47 个 cap 格各自的真实物理调用数 | **推导非记录**：1088 为上界，853 为下界 |
| U-3 | Support 探针 Consumer fits | **推导非记录**：641 由代码语义算出，无收据字段 |
| U-4 | `ScopeExecutor._baseline` 与 `_policy_reading` 的静态参照是否数值相同 | **UNKNOWN**：无工件比较过；直接影响 M-W 的 Δ 定义 |
| U-5 | 词表修好后普查是否真能产出可用的 `NARROW` | **超出本审计范围**：反事实只证明候选会被提出；Slow 与阈值工具在 5 次真实尝试中 `CALIBRATED` 为 0，因此**修词表不是充分条件** |
