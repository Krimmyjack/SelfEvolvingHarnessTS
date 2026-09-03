# P4b 有限放宽 CONFLICT 门 · 结果与收口（2026-08-31）

> **数据源勘误（2026-09-01 追加，不改动本文任何数字与判词）**
>
> 本文标注的数据集 `KDD Cup 2018 with missing values` **是错的**。实际使用的
> `data/kdd2018/series_cache.npz` 建自 `..._without_missing_values.tsf`，缓存内
> NaN 计数为 0；天然缺口（17.119%，270/270 条序列）已被上游填补消除。
>
> 逐值核验见 `artifacts/main_protocol/p4d_natural_gap_roster.json` 与
> `p4d_natural_gap_preflight.json`：两版本在 2,438,652 个观测位置上最大偏差
> `0.000e+00`。**本文数字在该（without）版本上测得正确，全部保留。**
>
> 但结论范围收窄为**无缺口的 outlier / level / denoise 场景**。identity 自身即
> `_linear_integrity` 线性插补，故全部 imputation 算子在本文数据上退化为恒等，
> 从未真正受考——本文任何负结论**不关闭 imputation 方向**。
>
> 含缺失版本记为独立数据身份 `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`，
> 与本文结果平行、**不并表**。详见 `AGENTS.md` §8.1。


**判词**：`BOUNDED_GATE_STILL_BLOCKING`，`blocking_face = SUPPORT_B`
**工件**：`artifacts/main_protocol/p4b_bounded_risk_forecast_merged.json`
**预注册**：`docs/P4B_BOUNDED_RISK_GATE_PREREGISTRATION_2026-08-31.md`
**不覆盖**：`p4_forecast_performance_b8_llm8_run2_20260830.json`

---

## 1. 收集完整性

48 / 48 held-in cell，2 臂 × 8 origin × 3 replica，`(replica, arm, origin)` 无重复、
无缺失、无意外。三份分片的冻结 contract **逐字段相同**（15 个字段 + transport）。

| 分片 | replica | cells |
| --- | --- | --- |
| `p4b_shard_forward_reverse_20260831.json` | Forward, Reverse | 32 |
| `p4b_shard_interleaved.json` | Interleaved | 16 |

Forward/Reverse 那一轮在 32/48 处被外部中止；Interleaved 用相同的冻结 contract、origin、
臂与预算单独补齐。分片只改执行切分，合并器拒绝任何 contract 不一致的分片。

`WRITEBACK_GATED`（48 cell 全数通过）：没有任何一次 Skill store 变动绕过准入门。

---

## 2. 主结果：门放宽了，但没有一次通过独立确认

| 臂 | cells | Support-A 准入 | **Active Skill** | Support-B 拒绝理由 |
| --- | --- | --- | --- | --- |
| `A5-bounded` | 24 | **6** | **0** | 受害比例越界 ×2、聚合跌破物质线 ×2、单序列越界 ×2 |
| `A5-strict` | 24 | 3 | **0** | 关系非 POSITIVE ×3 |

**bounded 确实放宽了 Support-A**：准入数 6 vs 3，翻倍。这不是噪声——放宽的正是设计要
放宽的那一类（聚合为正、局部有冲突）。

**但 9 次准入没有一次通过 Support-B**，因此 **0 个 Active Skill、0 次 incumbent 变更**。
完整的 Target-local 部署权要求两个面都通过；Support-A 的准入只是临时准入。

### 逐次记录（9 次全部）

| 臂 | replica | origin | 程序 | A: gain/害%/最大 | B: gain/害%/最大 | B 拒绝 |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | Forward | 1176 | `outlier_mad` | +0.5517 / 0.05 / 0.017 | +0.2139 / **0.25** / **0.655** | 受害比例越界 |
| bounded | Interleaved | 1176 | `outlier_mad` | +0.5517 / 0.05 / 0.017 | +0.2139 / **0.25** / **0.655** | 受害比例越界 |
| bounded | Forward | 1896 | `outlier_mad` | +1.1388 | **−0.0213** | 聚合跌破物质线 |
| bounded | Interleaved | 1896 | `outlier_mad` | +1.1388 | **−0.0213** | 聚合跌破物质线 |
| bounded | Forward | 2856 | `outlier_mad` | +0.4930 / 0.15 / 0.131 | +0.1609 / 0.20 / **0.359** | 单序列越界 |
| bounded | Reverse | 2856 | `outlier_mad` | +0.4930 / 0.15 / 0.131 | +0.1609 / 0.20 / **0.359** | 单序列越界 |
| strict | Reverse | 1176 | `outlier_iqr` | +0.4937 | +0.2855 | 关系非 POSITIVE |
| strict | Reverse | 1896 | `outlier_mad` | +1.1388 | **−0.0213** | 关系非 POSITIVE |
| strict | Interleaved | 1896 | `outlier_mad` | +1.1388 | **−0.0213** | 关系非 POSITIVE |

---

## 3. 为什么不通过：风险读数不跨序列组迁移

**先厘清两个面是什么（本节为科学解释更正，不是新实验）。**

Support-A 与 Support-B 是**同一 origin 上两组不同的序列**，各 20 条 eval，交集为 0。
`support_b` 的 `delayed_token = origin + HORIZON` 只是 dispatcher 的路由键，两面读数都在
同一 origin。

而且这两组**既不是随机划分也不是分层划分**：`run_forecast_p1.py:264-266` 把结构可读的
UID 按**字典序**排序后，取 `[:20]` 作 Support-A、`[20:40]` 作 Support-B。因此
A = T1, T101 … T118，B = T119, T12, T120 … T139。

> **更正**：此前把 B 面表述为"相隔 48 步的延迟视界"、把下表的落差归因于"时间漂移"，
> **是错的，予以撤回**。正确表述是：**当前证据首先反映跨序列组不外推，且可能受字典序
> 切分的结构差异影响。**

该结构差异已单独复核，见
`artifacts/main_protocol/p4c_split_and_headroom_check.json`（0 LLM / 0 Outcome）：
7 项部署可见特征中**没有任何一项**在多数 origin 上区分两组，观测到的最大效应量
（rank-biserial 0.22）远低于该样本量的可检出下限（0.36 达显著、0.52 达 80% 功效）。
故字典序切分**未**造成可检出的大结构失衡——但该检查只能排除**大**分离，不能排除中等分离。

对同一个程序在两个面上的风险读数：

| origin | 程序 | Support-A | Support-B | 漂移 |
| --- | --- | --- | --- | --- |
| 1176 | `outlier_mad` | 害 0.05 / 最大 0.017 | 害 **0.25** / 最大 **0.655** | 受害比例 ×5.0，最大单序列损失 **×39.3** |
| 2856 | `outlier_mad` | 害 0.15 / 最大 0.131 | 害 0.20 / 最大 **0.359** | 受害比例 ×1.3，最大单序列损失 ×2.7 |

**Support-A 读数都稳稳落在预算内（0.20 / 0.30），同一程序在 Support-B 的 20 条新序列上
都在预算外。** origin 1896 更直接——聚合从 +1.1388 掉到 −0.0213，程序在另一批序列上根本
没有收益。

这与上游诊断同源：`docs/P4_CONFLICT_PER_SERIES_AUDIT_2026-08-31.md` §5/§8 已证明受害序列
集不稳定、部署可见特征无法提前预测受害（分组 AUC 0.587）。本实验把同一结论推到
**序列划分之间**：在 20 条序列上量到的风险剖面，不能用来约束另外 20 条序列上的风险。

**因此"给伤害设界"这条路径在本数据上不成立**——不是因为界设得太紧，而是因为
**设界所依据的读数本身不外推**。放宽 k/m 只会让更多在 A 面看着安全、在 B 面并不安全的
候选拿到临时准入，仍会在 B 面被挡；调紧则连临时准入都没有。两个方向都不改变
Active Skill = 0。

---

## 4. held-out 未开启（依裁定）

0 个 Active Skill、0 次 incumbent 变更 ⇒ 两臂的 `_frozen_recall` 在每个 held-out origin
上都会落回 identity ⇒ **主对照 `A5-bounded − A5-strict` 按构造恒为 0**。开启终点面只会
把它唯一的一次读数花在一个已定的结论上，故不开启，held-out 八个 origin 保持未曝光。

判词记为 `BOUNDED_GATE_STILL_BLOCKING` 而非 `BOUNDED_GATE_NEUTRAL`：
neutral 需要至少形成一个 Active Skill 且完成 held-out，本轮两个前提都不满足。

---

## 5. 本实验主张什么、不主张什么

**主张：**

- bounded 门在 Support-A 上按设计放宽（准入 6 vs 3），实现正确，回归等价性此前已验
  （`STRICT_EQUIVALENCE_PASS`）
- 安全机制有效：独立 Support-B 确认拦下了全部 9 次临时准入，其中 6 次有明确的风险或
  收益依据；写回全程受门约束（`WRITEBACK_GATED` 48/48）
- Support-A 的逐序列风险读数**不迁移**到 Support-B 的那批不相交序列（×5.0 / ×39.3）

**不主张：**

- 不主张 bounded 门的性能收益或损失——**没有 Skill 形成，held-out 未开启，无性能读数**
- 不主张跨域积累收益——审计 Source 卡在本批 origin 上 Scope 匹配 0/24，treatment 为空
  （见预注册 §0 与 `p4_source_treatment_empty_correction_20260831.json`）
- 不主张旧 P4 负结果被修正或替代
- 不主张 20% / 0.30 是错的阈值——本轮不调阈值，且第 3 节说明调阈值不改变结论

---

## 6. 记录缺口（下一轮必须补）

被 Support-A 拒绝的候选**不记录程序步**：probe 行只有
`candidate_id / kind / aggregate_gain / admission`，`winner_program` 只在形成 winner 时
才有。因此"strict 拒 / bounded 准"这类最有价值的配对无法做 steps 比较，只能比结果。

本轮能核实的部分已核实：origin 1896 上 `A5-bounded` 的 `repair_extreme_deviation` 与
`A5-strict` 的 `outlier_mad_intrinsic` **`winner_program` 逐字段相同**
（`[{"op":"outlier_mad","params":{}}]`），候选名不同只是 Agent 命名。排练中 origin 648
的那一对**无法**这样核实——strict 侧被拒、无 steps 记录，当时的依据只是 20 维逐序列
增益向量逐位相等，属结果同一而非程序同一。

修法：在 probe 行落盘候选的完整 steps + params。本轮未改，因为第三个分片已在用当前
代码运行，中途改记录会让三份分片不一致。

---

## 7. transport

`https://api.nowaterapi.xyz/v1` 的 `gpt-5.6-sol`；旧 P4 为 `api.agicto.cn` 的
`gpt-5.6-luna`。三份分片 transport 一致（合并器已核）。P4b 内部两臂同场同模型，主对照
不受影响；**与旧 P4 的跨场数值不并表**。
