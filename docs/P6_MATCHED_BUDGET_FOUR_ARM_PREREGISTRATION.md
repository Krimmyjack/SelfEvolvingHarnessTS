# P6 Matched-Budget 四臂基线 — 完整预注册（2026-08-13）

Batch 机制收益归因：正确性来自 batch evidence 还是预算本身。
执行者：`evaluation/functional/run_v1_matched_budget_four_arm_dev.py`

## 0. 前置与定位

- 用户 P1-P6 自动推进序：P1（block2 闭包，停止条件触发）→ P2 按失败原因分支 → P3 PIA 校准（PIA_SKETCH_SAFE_SCREENING_MARGINAL——分支关闭，不接线）→ P4/P5 前置不满足（无已获批 Skill、无共同 headroom patch）→ **P6 matched-budget 四臂基线**（SHIFT_REPORT NEXT_BRANCH #1：主机制已成立、前置满足）。
- 问题：Batch Context-conditioned Slow（主机制）的正确性是否来自 **batch evidence（capsule/headroom）**，还是单条触发、组 v0 或纯确定性搜索也能在相同预算下达到同等正确性。
- 定位：development exposure——零新 Claim。四臂中 A/B 两臂含真实 LLM 新调用（≤4 次 + 校验重试），C/D 臂引用已暴露报告（零新调用）。

## 1. 证据面（全部已暴露——零新 outcome）

- **E_pos（有矿）**：T117 winsorize 失败组 @888（−0.1426）/ @984（−0.0841）。已暴露：hampel_filter 共同正向（888 +0.034 / 984 +0.015）、组内 replay 全过、holdout @600 +0.0013 压线 ≥−M。正确行为 = 产出 `patch-replace-winsorize-with-hampel_filter` 且过门。
- **E_neg（无矿）**：wave3 development family（winsorize NEGATIVE × 4 series × 6 窗，最负 T100@600 −0.1644）。已暴露：outlier_mad/hampel_filter 均非共同正向（outlier_mad 在 T10@600 −0.047、T1@888 −0.0029 失败；hampel 在 T100@600 −0.081、T100@888 −0.0765、T10@888 −0.1174 失败）。正确行为 = 弃权（零 LLM）或 replay 门拒（选 patch 被门拒绝）。

## 2. 四臂定义（白名单相同 = [outlier_mad, hampel_filter] typed patches）

| 臂 | 机制 | E_pos 输入 | E_neg 输入 | LLM 预算 |
|---|---|---|---|---|
| A 单 Episode Slow | 单条失败 Episode → `handle_feedback_support`（单条路径——无 capsule/headroom/对比） | T117@888 单条 | T100@600 单条（family 最负——确定性） | ≤1/窗口 |
| B Group Fault v0 | 组触发 + **v0 capsule**（per-episode 行 + cohort 统计——**无 view 对齐行、无对比案例、无 replacement_headroom facts**） | 组 [888, 984] | 组（6 窗） | ≤1/组 |
| C Batch Context-conditioned | 当前主机制（全 capsule + headroom） | **已暴露引用**：witness v3（1 调用选 hampel → replay 全过 → pending） | **已暴露引用**：wave4a-r2（1 调用正确弃权） | 0（已花 2） |
| D 等预算 Pipeline Search | 零 LLM 确定性（unique_common_positive + Runtime 编译） | **已暴露引用**：evc dev A 链（hampel → pending） | **已暴露引用**：evc dev B 链 + block2（evidence_abstain 零调用） | 0 |

**预算匹配口径**：候选白名单相同（2 typed patches）；Support 评估 = 窗口 × 候选（A：1×2 per 面；B/C/D：组窗 × 2）；LLM 上限每臂 ≤2。C/D 已在历史中用尽或低于预算（C=2、D=0）——如实记账。

## 3. 正确性定义（预注册）

- E_pos 正确 = 产出 hampel patch 且组内 replay 门全过（到达 pending）；选 outlier_mad = 错误（门拒）；弃权 = 正确性缺口（有 headroom 却不产出——记 `abstain_with_headroom`）。
- E_neg 正确 = 弃权（no_proposal，省预算）或选 patch 被 replay 门拒（正确拒绝但多花评估——两者都记为"正确拒绝"，弃权额外记预算节省）。
- **A 臂 E_neg 特例（预注册口径）**：单条路径设计上不可见组级 no-headroom——T100@600 单窗上 outlier_mad 的 headroom +0.288 存在，单条 Support replay 会过 → 如实记 `single_window_adopt`（组级陷阱暴露：该 patch 在别的 family 窗口 T10@600 −0.047 有害）——这正是 batch 价值主张的反面证据，E_neg 主对比仍以 B/C/D 的组级行为为准。
- 协议失败（>2 LLM 调用/契约失败/格式失败）= PROTOCOL_FAILURE 该臂无效。

## 4. 指标与主对比（预注册）

1. 每臂：E_pos 产出（patch_id / stage）、E_neg 行为（reason_code / stage）、LLM 调用数、Support 评估数。
2. 主对比：
   - **C vs A**：A 弃权或选错而 C 正确 → `BATCH_EVIDENCE_NECESSARY`（单条不足，组级证据是正确性成分）
   - **C vs B**：B 选错而 C 正确 → `HEADROOM_EVIDENCE_NECESSARY`（capsule 证据是弃权/选择依据）
   - **C vs D**：D 与 C 同正确且零 LLM → `DETERMINISTIC_SEARCH_SUFFICES`（正确性来自确定性搜索，LLM 只余编译角色——与 P0 降级设计结论同向）
3. 预算记账：各臂 LLM + Support 评估总数。

## 5. verdict（预注册——组合式如实报告）

- `BATCH_EVIDENCE_CONTRIBUTES`：E_pos 上 A 或 B 至少一臂未产出正确 patch（弃权/选错），而 C 产出
- `BATCH_EVIDENCE_REDUNDANT`：A 与 B 都在 E_pos 产出正确 patch
- `DETERMINISTIC_SEARCH_SUFFICES`：D 与 C 同等正确
- 加上 E_neg 各臂行为表；PROTOCOL_FAILURE 若任一承重臂协议失败

## 6. 纪律

- 一机制一实验：本实验不改任何 Harness 代码——纯装置执行 + 报告引用。
- 真实 LLM 新调用 ≤4 + 校验重试（CountingClient 每臂独立 max 2；协议错误整次重试一次后如实记录）。
- 温度 0（CountingClient 强制）；模型 gpt-5.6-luna / agicto。
- 报告不覆盖：A/B 臂新结果 + C/D 引用来源全部留痕。
