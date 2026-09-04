# GRID0 执行计划

2026-08-15 · 全程零 LLM · 协议见 `_drafts/grid0_protocol.json`（rev2）

## 背景一句话

Rule Card 机械层 BSE 已经建好并跑通两轮，唯一缺的是 `applicability.feature` 里能填什么。
GRID0 就是找这个字段——找不到就确信地关闭这条 family。

## 关键仪器修正（rev2，动手前必须理解）

`v6._evaluate` 的训练集 = 所有 `anchor + 48 <= origin` 的窗口，anchors 封顶 852。
所以原设计里不同 origin 的训练集是**嵌套累积**的（3→10 个窗口），origin 同时改变训练集和评估点。

**修法：钉住 anchors = `[312, 372, 432, 492, 552]`**（origin=600 时合法的 5 个，对所有更大 origin 都合法）。
训练集从此对所有 cell 恒定，origin 轴变成纯评估点。origin 集 = `{600, 672, 744, 816, 888, 960}`。

---

## 步骤

| # | 步骤 | 干什么 | 产出 |
|---|---|---|---|
| 1 | **P4 落账** | 把 a5v3 的 harm 账目拆成 support/delayed，修正术语（winner ≠ adoption），追加科学状态 `MIXED_TRANSFER_SIGNAL_UNDERPOWERED`。冻结 verdict 不动，只追加 | 主报告 `a5v3.amendment_2026_08_15` + `a5_next_protocol_requirements` |
| 2 | **冻结协议** | 把 rev2 协议写进主报告，之后不许改 | `grid0_protocol`，state=FROZEN |
| 3 | **成本探针** | 在**已暴露**的 T1/T10/T100/T101 上跑 3 格，量单格耗时。不碰任何 virgin series | 中位耗时 `t` |
| 4 | **定规模**（主控 agent） | 套预注册规则：`210*t ≤ 8h` → 25+10 series；否则 15+8 | 规模写进 protocol，仅此一次 |
| 5 | **资格 census** | 零 Outcome 筛 series：窗口完整、算子确实行动、**5 个训练窗口的 modified_fraction 全部 ≤ 0.35**（复用 `ScopeExecutor.verify()`，不另写）。cohort_B 按确定性规则选 metr_la，不合格回退 traffic_hourly | 候选名单 + 每条淘汰原因 |
| 6 | **停下审批**（用户） | 名单交用户批 exposure。批完的 series 永久标 development exposed | 普通 split/roster 登记 |
| 7 | **算 Observation** | 逐 cell 算三组特征。**必须在打开 utility 之前算完**，否则特征会被结果污染 | checkpoint 的 observations 段 |
| 8 | **跑 utility** | 逐 cell 跑 `outlier_mad` vs identity，出 gain。可断点续跑 | checkpoint 的 cells 段 |
| 9 | **P1A 统计** | 按冻结清单算 A1–A8，**只出数不裁定** | checkpoint 的 statistics 段 |
| 10 | **P1B 裁定**（主控 agent） | 判三分支之一，决定下一步 | 主报告 `grid0.verdict` |

## 三组 Observation 各自要回答什么

| 组 | 内容 | 回答什么 |
|---|---|---|
| **F1** 控制组 | 现有 `extract_public_features` | 当前观测面本来就够吗？（试点说不够） |
| **F2** 程序-窗口几何 | 5 个**真实训练窗口**上的修改比例/位置/幅度/拓扑 | 有害与否，能不能由"算子在这个窗口上干了什么"预测？ |
| **F3** cohort 同步 | 同 origin 下 cohort 内其他 series 的聚合（全部 LOO） | 还是说这是整个 cohort 在某个时间点共同变差？ |

## P1B 三个分支

| 分支 | 判据 | 下一步 |
|---|---|---|
| `WINDOW_GEOMETRY_SIGNAL` | 两个跨 cohort 方向都满足 `skill(F1+F2) ≥ 0.15` 且比 `skill(F1)` 高 `≥ 0.05` | F2 做成 Observation 填进 Rule Card，开 P2 Gate 校准 |
| `COHORT_TIME_SIGNAL` | 上条不成立，但 F3 增量在两 cohort 都 `≥ 0.10` | F3 做成 cohort Observation，开 P2 |
| `RESIDUAL_DOMINATED` | 都不成立 | **确信地关闭** outlier_mad Scope family。不调 Gate、不建卡、不跑自然进化 |

三种结果都能决定下一步——这是这批 exposure 值得花的理由。

## 纪律

- **零 LLM**，任何一步需要 LLM 就停下回报
- **顺序不能颠倒**：census → 用户批 → Observation → utility → 统计 → 裁定
- **不看结果调规模**：规模只由第 3 步耗时定，冻结后不扩编，哪怕结果不显著
- **不后验挑阈值**：P1B 之后不许从多个特征里挑最好的
- **只追加不改写**：已冻结的 verdict / rows / protocol 一律不动
- **机械层断裂即停**：报告，不重跑，不二次修复
- 工件就一个 `grid0_checkpoint.json`（四段）+ 主报告两个键。测试只留一个端到端承重 integration test（含信息墙断言），不为每个字段建测试矩阵

## 暂缓，不要做

Rule Card 泛化 · LTSV/TimeInf · P3 Fault Family 编码 · P2 Gate 校准 · P5b headroom 预检 · fresh A5/A3 重跑 · 自然 Batch Local Evolution
——全部等 P1B 结果。

## 分工

- **主控 agent**：第 4 步定规模、第 10 步裁定
- **用户**：第 6 步批 exposure 名单
- **本地 agent**：其余全部

---

## 进度（2026-08-15 更新）

第 1–3 步已完成。**第 4 步已裁定：PRIMARY，cohort_A=25 + cohort_B=10，210 cells，规模就此冻结。**
依据与三条附加记录写在主报告 `grid0_protocol.grid_size_decision_STEP4` 与 `grid0_protocol.rev3_notes_on_record`（只追加，未改任何规则/阈值/公式）。

动手前必读那三条：

- **N1** `evaluate()` 在 verifier 拒绝时提前返回，被拒格耗时远低于正常格。已复核 8 格全部 accepted，成本外推有效。
- **N2** 钉住 anchors 后 **F2 在同一 series 内跨 origin 恒定**（实测 `behavior_point_count` 在 origin 600 与 960 完全相同）。三族观测轴= F1 series×origin / F2 仅 series / F3 cohort×origin。F2 的有效样本量是 35 条 series，不是 210 格。留出单位已是 series/cohort 级，无需改统计。
- **N3** A6 的 **B→A 方向是功效瓶颈**（F2 部分只有 10 条 series 支撑）。若落 RESIDUAL_DOMINATED，verdict 必须写「在本功效下未观测到」，不得写「不存在」。已提前上记录，事后不许改口。
- **N4** **第 5 步 census 只准调 `verify()`，禁止调 `evaluate()`**。三项判据（窗口完整 / 5 窗 modified_fraction 全 ≤0.35 / 算子确实行动）全部可由 verify 得出，全程不算 gain——信息墙落在代码层面。
  「算子确实行动」= 5 个钉住窗口逐窗口修改点数之和 > 0。**这条会真淘汰 series**：已暴露的 T101 就是 0。

**第 6 步照旧停下，不得跳过。** exposure 不可逆，35 条 virgin series 必须先出名单再开跑。
流程：本地 agent 出名单（含每条淘汰原因）→ 主控 agent 对照确定性规则复核 → 一屏交用户点头 → 才进第 7 步。
名单若与确定性规则有任何偏离（cohort_B 触发回退、合格数不足、需要 tie-break），必须显式标出。
