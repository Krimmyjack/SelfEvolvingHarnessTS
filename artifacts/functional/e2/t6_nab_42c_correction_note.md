# #42c 机械分支闭合说明

非新书、非方法轮。接续未提交 #42b/#42c 工作树，未 reset / checkout / 从 HEAD 重建。
四处既有修正只核验，不重写。未运行正式 `--evaluate`。未 commit。

判定：**LIFECYCLE_FIXTURE_CLOSED**（MECHANICAL_FIXTURE，不计方法证据）。

## 本轮实际改动面

同一 runner：`evaluation/functional/run_e2_t6_natural_a5_a3.py`

- `_run_cells(..., support_trial_budget=None)`：默认仍是冻结的 `SUPPORT_TRIAL_BUDGET=2`。
  八-cell smoke / 正式 evaluate 路径的预检与探测预算不变。
- 新入口 `--evaluate-lifecycle-fixture`：同一 `_run_cells` →
  `run_online_round` → `open_delayed` → `activate_approved`。
- 新工件：`artifacts/functional/e2/t6_nab_lifecycle_fixture_v1.json`
- 本说明：`artifacts/functional/e2/t6_nab_42c_correction_note.md`

未建新 runner / Schema / Hash / Gate。未改 Consumer、菜单、roster、
CPC/CPM 分组、窗口、EVALUATE_ORDER、判定格、生命周期或 Observation。
未改 v1/v2 plan 工件。未重跑八-cell smoke。

## 四处修正核验（只读，未重写）

| # | 要求 | 现役位置 | 核验 |
|---|---|---|---|
| 1 | LLM 48 全实验共享总账 | `_run_cells` 顶部 `shared_backend = backend_factory(llm_budget)` 一次；四 cohort×arm 共用；`llm_calls` 读该 backend | `shared_backend` 构造 1 次；`backends.append` 0；`EVALUATE_RUN` 路径未再按臂建 backend |
| 2 | 每轮按当前 `support_origin` 重绑 Gateway | cell 内 `bind_round_data(series0[:support_origin], …)`；`_evaluate_agent` / smoke factory 用 `windows[round_name]` | 显式重绑仍在；构造期不再写死 r1 |
| 3 | delayed 指标只读 delayed response | `evaluated=True` 才记 delayed；无则 None；support 单列 `harmed_series_support_layer` | 混层键已不存在；`final_delayed_macro_f1_gain` 在 `delayed_responses_evaluated==0` 时为 None |
| 4 | 删除 PENDING_ADJUDICATION，十格机械落地 | `_evaluate_verdict` 自上而下首个命中 | `EVALUATE_RUN_COMPLETE_PENDING_ADJUDICATION` 出现 0 次 |

十格顺序未改：

`TARGET_LABEL_WALL_BREACHED` → `INCOMPLETE_LLM_BUDGET` →
`TARGET_FEEDBACK_UNREADABLE` → `NO_ADOPTABLE_PLAN_IN_TARGET` →
`SOURCE_CONTEXT_NOT_RETRIEVED` →
`SOURCE_EXPERIENCE_RETRIEVED_NO_BEHAVIOR_CHANGE` →
`SOURCE_EXPERIENCE_NEGATIVE_TRANSFER` →
`SOURCE_EXPERIENCE_ACCELERATES_TARGET_ADAPTATION_NATURAL` →
`SOURCE_EXPERIENCE_SAFER_NOT_FASTER` →
`NO_SOURCE_EXPERIENCE_ADVANTAGE`。

未活跃臂试验数仍以 `inf` 参与比较。正判仍带 NATURAL / provisional caveat。

## Fixture 完整链路

固定：`source_aws_cloudwatch` / `r2` / 全 8 序列 / `{identity, winsorize}`。
脚本后端，候选仅 `winsorize`；identity 只作 executor baseline。
臂 = A3（空 Memory），只为走生命周期，不测检索。
槽位键仍用冻结 evaluate 槽 `target_cpc`，行数据是 Source，LabelWall
`released=False`。

| 步 | 读数 |
|---|---|
| 序列 | 8 条 `ec2_cpu_utilization_*.csv` |
| 调用 | `run_online_round=1` `open_delayed=1` `activate_approved=1` |
| Support | winsorize gain **+0.026785714285714288**（≥ +0.005） |
| Draft | `fast_skill_event.stage=pending`，`support_relation=POSITIVE` |
| delayed 打开 | adapter `part` 含 support 与 delayed；`delayed_responses_evaluated=1` |
| delayed 分类 | `delayed_relation=POSITIVE`，gain **+0.007804878048780467** |
| Episode | `LOCAL_ACTIVE`，id `target_cpc_target_winsorize_a3_r2_p1` |
| 激活 | `activated=True`，`approved_skill_id=fast_winner_anomaly_detection_aegists_iforest_v1_macro_event_f1_winsorize` |
| 标签墙 | `target_key_requests=[]`，`breached=false`，Target 值未保留 |
| 预算 | AD **16 / 20**，LLM **0 / 0** |

Support / delayed 宏 F1 与 v2 Source bank 同 cell
（identity 0.0357 / 0.1372，winsorize 0.0625 / 0.145）逐字相符。
这是已曝光 Source 读数的机械复现，**不算新增方法证据**。

`support_trial_budget=1` 只在 fixture 入口传入。默认八-cell 预检仍是
`(1+2)*n`。全 8 序列若仍按 2 个非 identity trial 预检，最小需求 24，
会在授权 20 内被 cell 边界预检拦住；收成 1 后最小需求 16，与实测 16 fits
对齐（每序列 identity baseline + winsorize 各 1，× 支持/延迟两个 origin）。

## B1 / B2 / B4 / B5 静态核验（未重跑八-cell）

既有工件未改时间戳：

- `t6_nab_evaluate_smoke_v1.json` — 2026-08-23 12:45:16
- `t6_nab_evaluate_smoke_budget_v1.json` — 2026-08-23 12:51:11

`evaluate_smoke()` 仍调用同一 `_run_cells`，不传 `order_override` /
`support_trial_budget`，因此仍走冻结八格与 `SUPPORT_TRIAL_BUDGET=2`。
本轮只给 `_run_cells` 增加了默认关闭的形参，不改变那条路径的控制流。

| 项 | 旧工件 | 路径核验 |
|---|---|---|
| B1 八 cell 冻结顺序 | OK | `EVALUATE_ORDER` 未改 |
| B1 逐 cell 活调用 ≥1 | OK | fixture 再次证明同一调用点会 +1 |
| B2 Episode 键 | OK | fixture Episode 键仍是 `anomaly_detection\|aegists_iforest_v1\|macro_event_f1` |
| B2 A5 见卡 / A3 不见 | OK | 检索函数未改；本 fixture 用 A3，`source_cards=[]` |
| B4 超限中断 | OK（budget 工件） | cell 边界预检仍在；默认 `trial_budget` 未变 |
| B5 零 Target 键 | OK | fixture LabelWall 再次为空 |
| B3 | 仍 FAIL | 主线已裁定：缩小 stand-in 未触发正向分支，不重跑 |

## 旧 smoke `llm_calls` 口径差

Fix 1 把语义从「四只 backend 计数之和」改成「一只共享总账」。
旧八-cell smoke 已是 0-LLM，`budget_trace.llm_calls=0`，两种口径数字相同，
**不能用该整数区分旧四账与新单账**。

现役语义：

- `run["llm_calls"]` = 唯一 `shared_backend.calls`
- `readings[arm].llm_calls` = 该臂最后完成 cell 看到的**同一只**计数器快照，
  **不是**该臂自己的增量；正式读数求和会重复计算

旧 smoke 工件没有 `llm_ledger` 字段。本 fixture 工件写明
`llm_ledger: one shared backend across every cohort, arm and round`。

## V10 / 释放开关 / Target

- V10：40 个成员均在，本轮未写其中任一文件。
- 正式 `--evaluate`：未调用；`t6_nab_evaluate_v2.json` 不存在。
- Target outcome：fixture / smoke 的 `target_key_requests` 均为 `[]`。

### 必须上报的并发树事实

本会话开始时（约 13:08）读到 `t6_nab_frozen_plan_v2.json` 的
`evaluate_released=false`。13:28:12 该文件被**其他写入者**改成
`evaluate_released=true`（本会话未写 plan 工件，也未把它改回去——
plan 只读）。因此 fixture JSON 里
`evaluate_released_untouched=false` 是对**磁盘现状**的诚实记录，
不是本轮置位。

本轮仍：

- 不调用 `--evaluate`
- 不把开关写回 false 或 true
- fixture 自己的 LabelWall 保持 `released=False`

请主线裁定：是他人误触、二次核验后的合法置位，还是需要主线收回。
在主线二次核验封存之前，本执行方仍不得跑正式 `--evaluate`。

## 歧义（只报不选）

1. 释放开关并发翻转，见上。
2. Fixture 臂取 A3 而非 A5：书面要求是生命周期闭合，不是检索对照；
   A3 空 Memory 避免把 Source 卡影响算进这条机械复现。
3. `open_delayed` 对同一 winner 多次 `evaluate` delayed（Episode 写回 /
   `handle_feedback_delayed` / `delayed_utility`），fits 计 16 而非
   「8 baseline + 8 candidate」的 16 以外再涨，是因为 delayed 侧
   identity/winsorize 模型按 `(uid, signature)` 缓存。未改冻结执行器。
