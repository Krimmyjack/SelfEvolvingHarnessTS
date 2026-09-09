# M-R0b · 历史修订机会清点（逐外环步，按当时状态）

**stage** `M_R0B_REVISION_OPPORTUNITY_CENSUS` · **evidence grade** `INSTRUMENT`
**机器可读工件** `artifacts/main_protocol/m_r0b_revision_opportunity.json`
**复算** `python -m evaluation.main_protocol_p4.audit_m_r0b_revision_opportunity`

**边界**：0 LLM、0 Consumer fit、0 held-out 读、0 密封读、0 生产代码改动、0 既有工件覆盖、0 新增 SHA。

## 0. 口径

对 30 个已记录外环步中的每一步、每条 bank 内谱系，**按该步边界当时的状态**回答四问
（绝不用课末状态回填：一张课末 FLAGGED 的 Draft 在早期可能是 REVISABLE）：

| | 问题 | 判定来源 |
| --- | --- | --- |
| **A** | Active 祖先是否存在 | 该步边界前的 K0 注入 + 逐 cell 激活累积 |
| **B** | 生命周期是否可接收 | 由 Draft 的逐事件时间线重放到该边界；无 Draft = 可开新壳 |
| **C** | Scope 影子修改是否可行 | `best_stump` 在**当时 bank 行**上的确定性搜索（冻结词表 + bin 边缘，0 fit） |
| **D** | 后续是否有独立验证与重遇机会 | 该臂该顺序中边界之后的可评单元数；重遇按实际部署读数计 |

报告的交集是 **B ∧ C ∧ D**，并按 A 交叉列出。

**未做的事，逐项标注**：真实 Slow 是否会提出该子句 = `NOT_EVALUATED`；replay 屏 = `NOT_EVALUATED`；
后续验证是否通过 = `NOT_EVALUATED`；未部署单元上该谓词是否重遇 = `UNKNOWN`。
本清点**不**模拟修复后的行为，也**不**产出任何假想效果曲线。

**关系口径**：relation 取自记录的逐序列增益（`_relation` 的增益定义）。理由是本清点问的是
「课程当时**握有**什么证据」，而不是「当时的接线**看得见**什么」——按 M-R0，as-run 普查
ADD/NARROW 各为 0。这不是 26/6 的重述：26/6 数的是候选，本清点还额外要求可接收的生命周期、
可行的子句和后续单元，是更窄的对象。

### 重建可信度（关键前置）

影子搜索需要**逐序列特征**，而课程工件只记录逐序列增益、不记录特征。特征因此通过 runner
自己的 `UnitContext` 在生产路径上重算（读已曝光的 KDD development 缓存，0 fit）。
课程只在 3 个步留下了真实影子读数，它们就是唯一的对照：

| 顺序 | k | 记录 | 重算 | 一致 |
| --- | --- | --- | --- | --- |
| forward | 1 | `BEST_STUMP` `local_robust_z_peak>=6.0`，feasible 8 / 96 | 同 | ✅ |
| forward | 2 | `BEST_STUMP` `longest_missing_run_fraction>=0.2`，feasible 1 / 96 | 同 | ✅ |
| reverse | 5 | `NO_FEASIBLE_STUMP`，96 considered | 同 | ✅ |

**3 / 3 逐字段一致**，搜索行数亦与记录的 `bank_rows` 一致（120 / 220 / 380）。

> 一次中途更正：影子必须搜**as-run 行集**，不能用本轮修复后去重的行集。先用去重行集时
> forward k2 无法复现（得 `local_robust_z_peak>=6.0` / feasible 4，而记录是
> `longest_missing_run_fraction>=0.2` / feasible 1）。改回 as-run 行集后三个全部复现。
> 两种行数在每条记录里并列（`rows_searched` / `rows_after_exact_deduplication`）。

## 1. 总量

| 量 | 值 |
| --- | --- |
| 外环步 | 30 |
| 逐步 × 谱系读数 | 96 |
| **步骤级机会（B ∧ C ∧ D）** | **7** |
| 其中有 Active 祖先 | 7 / 7 |
| 其中达到 `MIN_ADVERSE_UNITS_FOR_NARROWING = 2` | 7 / 7 |
| 其中 as-run 机制**实际触及**该谱系 | **2** / 7 |
| 其中有**实测**重遇（后续单元同程序部署且 treated ≥ 5） | **2** / 7 |
| **去重后谱系（顺序 × 臂 × census_key）** | **3** |
| **去重后谱系身份（census_key）** | **2** |
| 去重后（顺序 × 臂）组合 | 3 |
| 有实测重遇的去重谱系 | **1** |
| 实测之外的重遇 | `UNKNOWN` |

其余 89 条读数不构成机会的原因：

| 原因 | 数 |
| --- | --- |
| 影子搜索无可行子句（`NO_FEASIBLE_STUMP`） | 61 |
| 生命周期不可接收：Draft 当时已因 `EFFECT_NONSTATIONARY` 关闭 | 17 |
| 生命周期不可接收：当时为 `FLAGGED` | 11 |

61 + 17 + 11 + 7 = 96，账目闭合。

## 2. 七个机会，逐条

| # | 顺序 | 臂 | k | 边界位置 | 程序 | A 祖先 | B 接收路径 | 当时 Draft 状态 | C 影子子句 | D 后续可评 / 实测重遇 | as-run 是否触及 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | forward | A5-online | 1 | p04 | `outlier_mad` | 是 | REVISE 既有 Draft | REVISABLE, attempts 1 | `local_robust_z_peak>=6.0`（可行 8） | 18 / **4** | **是** → `SLOW_ABSTAINED` |
| 2 | forward | A5-online | 2 | p09 | `outlier_mad` | 是 | REVISE 既有 Draft | REVISABLE, attempts 1 | `longest_missing_run_fraction>=0.2`（可行 1） | 14 / **3** | **是** → `OUTER_LLM_BUDGET_SPENT` |
| 3 | reverse | A5-online | 2 | p09 | `winsorize` | 是 | NARROW 开新 Draft | 无 Draft | `longest_missing_run_fraction>=0.2`（可行 3） | 14 / 0 | 否 |
| 4 | reverse | A5-online | 3 | p14 | `winsorize` | 是 | NARROW 开新 Draft | 无 Draft | 同上（可行 3） | 10 / 0 | 否 |
| 5 | reverse | A5-online | 4 | p19 | `winsorize` | 是 | NARROW 开新 Draft | 无 Draft | 同上（可行 5） | 5 / 0 | 否 |
| 6 | reverse | A5-online | 5 | p24 | `winsorize` | 是 | NARROW 开新 Draft | 无 Draft | 同上（可行 5） | 1 / 0 | 否 |
| 7 | reverse | A3-online | 5 | p24 | `winsorize` | 是 | NARROW 开新 Draft | 无 Draft | `local_robust_z_peak>=6.0`（可行 3） | 1 / 0 | 否 |

**#3–#6 是同一条谱系在连续四步被反复提供**，不是四次独立机会；去重后 reverse A5-online 的
`winsorize` 只算一条。三条去重谱系为：

1. `forward · A5-online · outlier_mad@local_robust_z_peak>=3`（步 1–2）
2. `reverse · A5-online · winsorize@local_robust_z_peak>=3`（步 2–5）
3. `reverse · A3-online · winsorize@local_robust_z_peak>=3`（步 5）

两条去重身份：`outlier_mad@z>=3` 与 `winsorize@z>=3`。

## 3. 读法

- **只有 2 / 7 被 as-run 机制触及**，且都是 forward 的 `outlier_mad`：它们确实成为了 REVISE 候选，
  然后**止于 Slow / 阈值工具**（一次 `SLOW_ABSTAINED`、一次 `OUTER_LLM_BUDGET_SPENT`），
  不是止于生命周期。其余 5 个从未成为候选（该步 `census produced no candidate`；
  reverse k5 虽有一个 REVISE，但那属于 `outlier_mad` 谱系，与本行的 `winsorize` 不是同一条）。
- **重遇是最稀的一环**。7 个机会里只有 2 个（同一条 forward 谱系）在后续单元上有**实测**重遇；
  `winsorize` 谱系在其后所有单元上实测重遇为 0。其余单元没有该谓词的读数，一律 `UNKNOWN`——
  既不算重遇成功，也不算失败。
- **A 不是瓶颈**：7 / 7 都有 Active 祖先。瓶颈依次是影子无可行子句（61）、生命周期已关闭或
  FLAGGED（28）。
- 本清点**未**回答：Slow 会不会提出这些子句、replay 屏会不会放行、后续验证会不会通过。
  三者在每条记录里都标 `NOT_EVALUATED`。

## 4. 与既有数字的关系（避免误读）

- **不是 26 / 6**。那是「修好关系词表后普查会发出多少候选」；本清点在其上再要求
  可接收的生命周期、可行的子句、后续单元，因此是 7（步级）/ 3（谱系）/ 2（身份）。
- **不是 10/11 终态 FLAGGED**。那是课末终态；本清点按每步当时状态判定，forward 的
  `outlier_mad` 在 k1/k2 当时正是 REVISABLE。
- **不是 462**。462 是 30 个外环步对不断增长的 bank 反复普查的累计单元票，同一单元被
  重复计入多个步；本清点的 96 是「逐步 × 谱系」读数，7 是其中的机会数。

## 5. 新增产物

| 文件 | 性质 |
| --- | --- |
| `evaluation/main_protocol_p4/audit_m_r0b_revision_opportunity.py` | 新增只读清点脚本 |
| `artifacts/main_protocol/m_r0b_revision_opportunity.json` | 新增（此前不存在） |
| `artifacts/main_protocol/m_r0b_revision_opportunity.md` | 本文件，新增 |

未覆盖任何历史工件，未改生产代码，未提交，未发车。
