# CAP-1: capstone sampling revision + full exam protocol freeze

protocol: `cap1_capstone_protocol_freeze_v1`  
written: 2026-08-27  
HEAD at write: `010c0d1fc40ab6442891bd9fdfa53c7ad53056c0`  
parent: CAP-0 (`010c0d1`, `SEAL_INTACT` / `MATCH`)

0 LLM / 0 fit / 0 download / **零开封**。不改代码。不写 `docs/STAGE_REPORT`。

## 义务自报

- **D3 零接触**: 本书未打开 zip、未读成员、未计行。TEST 行号 = `random.Random(20260827).sample(range(11420), 476)` 后排序。TRAIN 切分 = `range(80)` 上 `i % 4` 角色拼接。无数值、无标签。
- CAP-0 密封与结构计数仍有效；**仅 mod-24 子集规则作废**。
- 开封前仍禁止 oracle / fit / 标签 / 数值。

## 1. TEST 子集：种子随机（作废 mod-24）

原因：若官方 TEST 按类别或批次排序，`i ≡ 0 (mod 24)` 会系统性偏层。

冻结算法（公开、可复现）：

```text
sorted(random.Random(20260827).sample(range(11420), 476))
```

- CPython `random.Random`（本书生成清单用 3.10.19）。
- seed = **20260827**（写死）。
- 476 行不变；总点 `556 × 178 = 98968` 不变。
- 三臂共用同一清单。

清单摘要：

| 项 | 值 |
|---|---|
| n | 476 |
| min / max | 10 / 11413 |
| first 10 | 10, 14, 19, 20, 48, 142, 144, 157, 191, 195 |
| last 10 | 11307, 11308, 11314, 11331, 11344, 11355, 11367, 11391, 11396, 11413 |
| sum | 2838207 |
| sha256(JSON array) | `7e1c408853a59244dea957dfc323cb3a8bd7dede9c44a1637be8898ebfabf874` |

完整升序行号见同名 JSON 字段 `sampling.test_row_indices_sorted`。

## 2. TRAIN 内 Support / delayed（M-1 对半协议，行号写死）

引用：`m1_margin_gate` = `MARGIN_GATING_CONFIRMED`。活仪器是  
`Support = concat(r1_support, r2_support)`，`delayed = concat(r1_delayed, r2_delayed)`，单轮双门。  
**拒绝** ps0b 同轮 `half_slices`（会塌掉双门）。

活 `_split_fit_support` / `_quarter` 要读标签，开封前不能用。本书用 **全部 80 行 TRAIN** 的行号四分替代；开封后 runner **必须用下列冻结下标，不得按标签重分层**。

| 切片 | 规则 | n | 材料线 |
|---|---|---|---|
| r1_support | `i ≡ 0 (mod 4)` | 20 | — |
| r1_delayed | `i ≡ 1 (mod 4)` | 20 | — |
| r2_support | `i ≡ 2 (mod 4)` | 20 | — |
| r2_delayed | `i ≡ 3 (mod 4)` | 20 | — |
| **Support** | r1s ∪ r2s = 偶数行 | **40** | `max(0.005, 1/40) = 0.025` |
| **delayed** | r1d ∪ r2d = 奇数行 | **40** | `max(0.005, 1/40) = 0.025` |

Consumer 拟合：官方 TRAIN 80 行全用（UCR base-train）。条件对 = `fit_only_artifact`（与课程一致）：注入只在 TRAIN，官方 TEST 子集保持干净，冻结后只开一次。

## 3. 三臂与 K0 同源

| 臂 | 起点 | 写回 |
|---|---|---|
| **Static** | identity；无 Harness / 无 Skill 池 | 否 |
| **A3** | 冷编译 `methods/ttha/harness/h0`；Memory 空；**不带** S1-v2 池 | 仅 Target-local |
| **A5** | 与 S1-v2 **同一 K0 起源**（h0 三张 bootstrap + 惰性 Slow 卡；不含 Target-local 能力、不含课前 PS 双源卡），再装入 **S1-v2 正序终态池**（课程内自产供给卡 + guard） | 是 |

K0 bootstrap：`inspect_and_localize` / `build_contrastive_candidates` / `select_or_identity_and_verify`。  
A5 相对 A3 的唯一额外知识 = S1-v2 正序实际产出的卡。装载第二次成功正序终态池（两份授权卡字节相同则可任取）。

h0 快照（`snapshot.lock.json`）：

- profile `h0-domain-naive`
- `runtime_bundle_sha` = `c3427b4e4b7cdf68322382cbd6354e08f38d584f9a3dd74a5f1987c21b44e539`
- `harness_content_sha` = `53b1c803f4ba38a27e2d1d7621f983997044019d7e073caef2a4436ee900654f`

## 4. Consumer / metric / 菜单

- Consumer = **ridge-raw-plus-difference-v1**（α = 1.0）
- 键 = `classification|ridge-raw-plus-difference-v1|accuracy`
- 指标 = **accuracy + 逐类 recall**
- 契约 = `classification_local_event_task_quality_contract_v1`
- `maximum_candidates` = 3（`1 + SUPPORT_TRIAL_BUDGET`，与现役分类共享 harness 一致）
- `maximum_modified_fraction` = 0.10

现役注册表快照（`operators/registry.py` git blob `8de9545b1af8bed9c7640a0d712c92bfececf9d2`；h0 lock `operator_registry` `fc1b8700…` / `operator_bundle` `336b3718…`）：

canonical、`classification` 可执行、非 `shape_changing`，外加 `identity`：

`identity`, `impute_linear`, `impute_fft`, `impute_ema`, `period_complete`, `period_median_complete`, `impute_ssm`, `impute_ar`, `denoise_savgol`, `denoise_wavelet`, `denoise_median`, `smooth_ma`, `denoise_stl`, `winsorize`, `outlier_iqr`, `outlier_mad`, `hampel_filter`, `repair_level_shift`, `repair_burst_segment`, `stl_decompose`, `fft_decompose`, `smooth_ema`, `resample_uniform`, `znorm`, `minmax_norm`

菜单名列表 sha256 = `48e09ec4d704f278196cbef277edb134c42917ea622fb7a34af6bef5d1b57a29`。  
排除：`sliding_window` / `lag_features` / `spectral_features`；弃用 alias `fill_gaps` / `impute_kalman` / `kalman_filter`。

## 5. 预算、重复、停止

- 每臂 LLM ≤ **15**、fit ≤ **25**；三臂合计墙钟 ≤ **90 min**。
- **单次**：capstone 是一次性验收，不再设评价重复种子。TEST 子集种子仅 20260827。
- `BACKEND_UNAVAILABLE` → 立即停，无科学判词。
- 预算越界 → 停，无科学判词。
- 密封破损或结构不符 → 不开封，回新下载路线。

## 6. 冻结判词

材料线（held-out）：`max(0.005, 1/476) = 0.005`。  
worst-class 非劣容忍：−0.005。  
harm 杠：现役 `HARM_BAR = 0.05`（相对 identity 的逐类 recall；`run_e2_t6_cls_op_shared_harness.py:152`）。

- **`CAPSTONE_POSITIVE`**：A5−A3 accuracy ≥ 0.005，且 worst-class recall 差 ≥ −0.005，且 A5 harm = 0（无类相对 identity 的 recall 跌破 −0.05）。
- **`CAPSTONE_NEGATIVE`**：A5−A3 accuracy ≤ −0.005，或 worst-class 劣于 −0.005，或 A5 harm > 0。
- **`CAPSTONE_NEUTRAL`**：其余。

A5 vs Static、A3 vs Static 仍按 `AGENTS.md` §2.1 报告，不替代本判词。

## 7. 开封条件（缺一不许开封）

引用 `docs/S1V2_DESIGN_DRAFT_2026-08-27.md` 判词 `S1V2_FORWARD_SIGNAL`：A5-online 质量与 harm 非劣于 A3-reset 与 K0-fixed，且累计 regret 或适应成本材料级改善，且优势可归因于课程内自产供给/guard 对后续单元的行为变化。重复计划：正序 ×2（不同注入 seed）+ 信号后反序 ×1。

**必须同时成立：**

1. S1-v2 **正序 ×2** 均出 `S1V2_FORWARD_SIGNAL`；
2. **反序 ×1** 确认（同一归因：课程内知识改变了后续单元）。

届时按**本协议**自动开封 Epilepsy2 冻结子集，无需再授权。缺任一条款不得开封。开封前禁止 oracle / fit / 标签 / 数值。
