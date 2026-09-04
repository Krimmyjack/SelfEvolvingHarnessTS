# 域内适应轨迹盘点 v1（只读、0-LLM、描述性）

## 目的

项目的最终目标是：**Harness 在一个域内随经验积累自适应变好**。本文件从已冻结的任务顺序执行数据里读出域内适应轨迹：同一臂在同一 cohort 内按任务顺序推进、且允许跨任务技能积累时，后面的任务是否比前面的任务表现更好。

这是描述性盘点，不是假设检验。样本小、方差大（同配置 A3 重复运行的 `LOCAL_ACTIVE` 计数可从 0 波动到 3）。如实报告不确定性，不把单条好看轨迹写成已成立的适应信号，也不把未形成技能的臂写成适应失败。

## 口径（工件自身字段）

| 读数 | 字段 | 规则 |
| --- | --- | --- |
| 真实探测数 | `metrics.real_support_probe_count`（与 `cost.target_support.real_support_probe_count` 一致） | 不是 `charged_probe_cost` |
| material-positive | `probes[].meets_material_threshold` | `support_gain >= 0.005`（`MATERIAL_THRESHOLD`，`evaluation/functional/task_episode_harness/runner.py`） |
| material-harm | `metrics.harmful_probe_count` / `metrics.cumulative_support_harm` | `support_gain < -0.005`；harm sum 为负增益的绝对值之和 |
| 停止原因 | `stop_reason` | 原样抄录 |
| LOCAL_DRAFT 形成 | `lifecycle.method_event.stage = "pending"` 且 `support_passed = true` | 这些运行的 `winner.local_status` **不写**字面量 `LOCAL_DRAFT`；delayed 拒绝时多为 `RESTRICTED` |
| LOCAL_ACTIVE 激活 | `winner.local_status = "LOCAL_ACTIVE"`；`metrics.task_local_active = 1` | 新形成 vs 复用再确认要分开：后者是 `deployed_existing_skill` |
| 技能复用 | `lifecycle.reused_existing_skill = true` 且 `method_event.stage = "deployed_existing_skill"` | 仅 `active_local_skill_ids_before` / `retrieved_skill_ids` 含 `fast_winner_*` 但未提案/未探测，记为“检索到但未部署” |

前半 / 后半：9 任务按顺序切成 `task_01–04` vs `task_05–09`。技能前 / 技能后：该臂首次 `LOCAL_ACTIVE` 形成任务之前 vs 之后（形成当轮不计入两侧）。

“首选家族命中正向率”：该任务至少有一次真实探测时，是否出现 material-positive（每任务几乎只探 1 次，接近被探家族是否正向）。分母不含 0 探测任务。

“到达首个正向所需探测数”：该任务内第一次 `meets_material_threshold` 的探测序号。这些运行里命中时几乎全是 1；分辨率不足，同时报告臂级“首次 material-positive 的任务序号”。

## 纳入与跳过

**纳入（任务顺序执行，臂内 Memory/Skill 可跨任务累积）：**

| 工件 | cohort | 为何算顺序积累 |
| --- | --- | --- |
| `artifacts/functional/e2/r3_full_ab_electricity.json` | electricity | 指定源。A3/A5 双臂，`arm_execution=parallel`：两臂同任务并发，**下一任务开始前 join**；A5 带 Source Skill，但本盘点读的是臂内 Target-local 轨迹 |
| `artifacts/functional/e2/calib_a3_run01.json` | electricity | 指定源。同配置双 A3（标签仍叫 A3/A5，`source_derived_skill=null`）。A3=COLD，A5=WARM |
| `artifacts/functional/e2/calib_a3_run02.json`、`calib_a3_run03.json` | electricity | 与 run01 同一 driver / 9-task roster，用作方差参照并补全同协议顺序轨迹 |
| `artifacts/functional/e2/g1_agentic_pipeline_report_T233.json` | T233 | 指定源。9 任务顺序；`active_local_skill_ids_before/after` 跨任务传递 |
| `artifacts/functional/e2/g3d1_electricity_skill_only_ab.json` | electricity | 已冻结 9-task 顺序 AB；`calib_a3_variance_report_v1.md` 把它的 A3 算进同配置样本 |

**跳过：** `t233_supply_obs_ab_*`、`t233_independent_*` 等供给类“每任务从 h0 独立开始”的运行。`r1_*` / `r2_*` 为 3-task micro / replay，不作为本盘点主表。`g3_development_electricity.json`、`g4_source_experience_electricity.json`、`g1_agentic_pipeline_report.json`、`g2_close_T233.json` 也是顺序可积累，但协议年代/通道与当前 electricity calib 不一致，只在文末附一句，不混进主判定。

---

## 1. `r3_full_ab_electricity.json`（electricity，A3 vs A5）

`cost_by_arm`：A3 真实探测 8、`task_local_active_count=1`、abstention 6；A5 真实探测 9、`task_local_active_count=2`（含一次复用后再标 ACTIVE）、abstention 5。

### 1.1 A3 任务顺序

| 任务 | 真实探测 | material+ n / 家族 / gain | material-harm n / sum / 家族 | `support_gain`（逐探） | `stop_reason` |
| --- | ---: | --- | --- | --- | --- |
| 01 | 1 | 0 | 1 / 0.0149 / repair_level_shift | −0.0149 | AGENT_ABSTAIN |
| 02 | 1 | 0 | 1 / 0.0548 / repair_level_shift | −0.0548 | REQUEST_OBSERVATION |
| 03 | 1 | 1 / hampel_filter / +0.0094 | 0 | +0.0094 | TRUST_DRAFT_GATE_PASS |
| 04 | 0 | — | 0 | — | AGENT_ABSTAIN |
| 05 | 1 | 0 | 1 / 0.1259 / repair_level_shift | −0.1259 | AGENT_ABSTAIN |
| 06 | 1 | 1 / outlier_mad / +0.0419 | 0 | +0.0419 | TRUST_DRAFT_GATE_PASS |
| 07 | 1 | 1 / outlier_mad / +0.0838 | 0 | +0.0838 | TRUST_DRAFT_GATE_PASS |
| 08 | 1 | 0 | 1 / 0.0221 / hampel_filter | −0.0221 | AGENT_ABSTAIN |
| 09 | 1 | 0 | 1 / 0.0617 / outlier_mad | −0.0617 | AGENT_ABSTAIN |

技能时间线：

- **LOCAL_DRAFT @03**：`method_event.stage=pending`，`edit_id=fast_winner_e1v2_hampel_filter`，`delayed_event.stage=delayed_rejected`（`delayed_gain=-0.0342`），`winner.local_status=RESTRICTED`。`r3_full_ab_electricity.json:3278-3292`
- **LOCAL_ACTIVE @06**：新形成 `fast_winner_e1v2_outlier_mad`，`delayed_event.stage=approved`。`r3_full_ab_electricity.json:6209-6273`
- **复用部署 @07**：`deployed_existing_skill`，`skill_id=fast_winner_e1v2_outlier_mad`，support +0.0838；delayed −0.0275 且 `delayed_ok=false`，winner 回到 `RESTRICTED`。`r3_full_ab_electricity.json:6810-6824`
- **检索到但未部署 @08–09**：`active_local_skill_ids_before` 仍有该 skill；08 提案/探测的是 hampel_filter（harm）；09 提案并探测 outlier_mad（harm）

轨迹：前半家族命中 1/3、harm 任务 2/4；后半 2/5、3/5。技能前命中 1/4、harm 3/5；技能后命中 1/3、harm 2/3。首次 material-positive 在任务 03（1 探）。**后半没有稳定变好。**

### 1.2 A5 任务顺序

| 任务 | 真实探测 | material+ n / 家族 / gain | material-harm n / sum / 家族 | `stop_reason` |
| --- | ---: | --- | --- | --- |
| 01 | 1 | 0 | 1 / 0.0149 / repair_level_shift | REQUEST_OBSERVATION |
| 02 | 1 | 0 | 1 / 0.1128 / fft_decompose | REQUEST_OBSERVATION |
| 03 | 1 | 0 | 1 / 0.1970 / repair_level_shift | AGENT_ABSTAIN |
| 04 | 1 | 0 | 1 / 0.0725 / repair_level_shift | REQUEST_OBSERVATION |
| 05 | 1 | 1 / outlier_mad / +0.1144 | 0 | TRUST_DRAFT_GATE_PASS |
| 06 | 1 | 1 / hampel_filter / +0.0189 | 0 | TRUST_DRAFT_GATE_PASS |
| 07 | 1 | 1 / hampel_filter / +0.0795 | 0 | TRUST_DRAFT_GATE_PASS |
| 08 | 1 | 0 | 1 / 0.1089 / repair_level_shift | AGENT_ABSTAIN |
| 09 | 1 | 1 / hampel_filter / +0.0161 | 0 | TRUST_DRAFT_GATE_PASS |

技能时间线：

- **LOCAL_DRAFT @05**：pending `fast_winner_e1v2_outlier_mad`，`delayed_rejected`，`RESTRICTED`
- **LOCAL_ACTIVE @06**：新形成 `fast_winner_e1v2_hampel_filter`，`delayed=approved`。`r3_full_ab_electricity.json:5682-5750`
- **复用部署 @07**：hampel support +0.0795；delayed −0.0583，`delayed_ok=false`，winner `RESTRICTED`。`r3_full_ab_electricity.json:7250-7264`
- **检索到但未部署 @08**：提案/探测 repair_level_shift（harm），没有部署已有 hampel skill
- **复用部署 @09**：hampel +0.0161，`delayed_ok=true`，`winner.local_status=LOCAL_ACTIVE`。`r3_full_ab_electricity.json:9083-9155`

轨迹：前半命中 0/4、harm 4/4；后半 4/5、1/5。技能前命中 1/5、harm 4/5；技能后 2/3、1/3。首次 material-positive 在任务 05。**后半/技能后看起来更好，但是 3 个后续任务里只有 2 次真正部署了 skill。**

---

## 2. `calib_a3_run01.json`（同配置双 A3；A3=COLD，A5=WARM）

两臂都是当前 A3 配置、`source_derived_skill=null`、9/9 `source_prior_retrieval.matched=false`。这是方差参照，不是 A5−A3。

### 2.1 A3 / COLD

| 任务 | 真实探测 | material+ / 家族 / gain | harm n / sum / 家族 | `stop_reason` |
| --- | ---: | --- | --- | --- |
| 01 | 1 | 0 | 1 / 0.0149 / repair_level_shift | AGENT_ABSTAIN |
| 02 | 1 | 0 | 0（impute_linear +0.000） | REQUEST_OBSERVATION |
| 03 | 1 | 0 | 1 / 0.1970 / repair_level_shift | REQUEST_OBSERVATION |
| 04 | 1 | 0 | 1 / 0.0725 / repair_level_shift | REQUEST_OBSERVATION |
| 05 | 1 | 0 | 1 / 0.1259 / repair_level_shift | REQUEST_OBSERVATION |
| 06 | 0 | — | 0 | AGENT_ABSTAIN |
| 07 | 1 | 1 / hampel_filter / +0.0442 | 0 | REQUEST_OBSERVATION |
| 08 | 1 | 1 / hampel_filter / +0.0395 | 0 | TRUST_DRAFT_GATE_PASS |
| 09 | 1 | 0 | 0（impute_ar +0.000） | REQUEST_OBSERVATION |

技能：仅 **LOCAL_DRAFT @08**（hampel pending，`delayed_rejected`，`RESTRICTED`）。无 LOCAL_ACTIVE，无复用。前半命中 0/4、harm 3/4；后半 2/4、1/5。首次 material-positive 在任务 07。**没有技能，后半仍略好——后半变好可以不经过 LOCAL_ACTIVE。**

### 2.2 A5 / WARM（同配置；`cost_by_arm.A5.task_local_active_count=3`）

| 任务 | 真实探测 | material+ / 家族 / gain | harm n / sum / 家族 | `stop_reason` |
| --- | ---: | --- | --- | --- |
| 01 | 1 | 0 | 1 / 0.0073 / hampel_filter | AGENT_ABSTAIN |
| 02 | 1 | 1 / hampel_filter / +0.0288 | 0 | REQUEST_OBSERVATION |
| 03 | 1 | 1 / hampel_filter / +0.0225 | 0 | REQUEST_OBSERVATION |
| 04 | 1 | 0 | 1 / 0.0725 / repair_level_shift | REQUEST_OBSERVATION |
| 05 | 1 | 0 | 1 / 0.1259 / repair_level_shift | REQUEST_OBSERVATION |
| 06 | 1 | 1 / outlier_mad / +0.0419 | 0 | TRUST_DRAFT_GATE_PASS |
| 07 | 1 | 1 / outlier_mad / +0.0838 | 0 | TRUST_DRAFT_GATE_PASS |
| 08 | 1 | 1 / hampel_filter / +0.0553 | 0 | TRUST_DRAFT_GATE_PASS |
| 09 | 1 | 1 / hampel_filter / +0.0161 | 0 | TRUST_DRAFT_GATE_PASS |

技能时间线：

- **LOCAL_ACTIVE @06**：新 `fast_winner_e1v2_outlier_mad`
- **复用部署 @07**：outlier_mad +0.0838，`deployed_existing_skill`。`calib_a3_run01.json:7671-7684`
- **LOCAL_ACTIVE @08**：在已有 outlier_mad 之上 **新形成** `fast_winner_e1v2_hampel_filter`（`reused_existing_skill=false`）
- **复用部署 @09**：部署 hampel（检索里同时看到两个 skill）

`task_local_active_count=3` = 06 新 + 08 新 + 09 复用后再标 ACTIVE。轨迹：前半命中 2/4、harm 2/4；后半 4/5、1/5。技能后 3/3 命中、0/3 harm。首次 material-positive 在任务 02。这是同配置样本的高位轨迹，不是单独的 A5 效应。

---

## 3. `calib_a3_run02.json` / `calib_a3_run03.json`（同协议重复）

### 3.1 run02 A3 / COLD

无 draft、无 ACTIVE、无复用。任务 01 探了 2 次（denoise_median 0.0 + smooth_ma harm）。01–09 几乎全是 repair_level_shift harm；仅 06 一次 material-positive（+0.0093）后仍 `REQUEST_OBSERVATION`。前半命中 0/4、harm 4/4；后半 1/5、3/5。首次正向任务 06。

### 3.2 run02 A5 / WARM

任务 02 为 `AGENT_PROTOCOL_ERROR`、0 探测（机械退出，不作行为读数；轨迹仍保留）。**DRAFT @05** outlier_mad `delayed_rejected`。**ACTIVE @06** 同家族批准。**复用部署 @07** outlier_mad +0.0838。08 检索到 skill 但探测 smooth_ema（harm −0.3975）。09 检索到但探测 period_complete（0.0）。前半命中 0/3、harm 3/4；后半 3/5、1/5。技能后 1/3 命中、1/3 harm。**复用只成功一轮，随后未再部署。**

### 3.3 run03 A3 / COLD 与 A5 / WARM

两臂都 **0 draft / 0 ACTIVE / 0 复用**。A3 仅任务 06 一次 material-positive（+0.0093），其余多为 repair_level_shift harm。A5 全程无 material-positive（前半命中 0/4，后半 0/5）。这是同配置样本的低位：9 任务走完可以完全不形成可复用技能。

与 `calib_a3_variance_report_v1.md:36-43,53-76` 一致：7 条 A3 配置轨迹的 `LOCAL_ACTIVE` 为 `{0,0,3,0,1,0,0}`，范围 0–3；material-harm n 范围 3–9；首次正向任务序号 2 / 5 / 6 / 7 / 无。

---

## 4. `g1_agentic_pipeline_report_T233.json`（T233，顺序 9 任务）

无 `arm_execution` 字段，但 `rows` 按 `e1v2_task_01…09` 推进，且 `active_local_skill_ids_*` 跨任务传递，属于可积累顺序块。

### 4.1 A3

| 任务 | 真实探测 | material+ / 家族 / gain | harm n / sum / 家族 | `stop_reason` |
| --- | ---: | --- | --- | --- |
| 01 | 2 | 0 | 2 / 0.1383 / hampel + repair_level_shift | AGENT_ABSTAIN |
| 02 | 1 | 0 | 0（repair_level_shift −0.0039，未过 harm 阈） | AGENT_ABSTAIN |
| 03 | 0 | — | 0 | AGENT_ABSTAIN |
| 04 | 1 | 0 | 0（period_complete 0.0） | REQUEST_OBSERVATION |
| 05 | 1 | 0 | 1 / 0.0370 / repair_level_shift | REQUEST_OBSERVATION |
| 06 | 1 | 1 / repair_level_shift / +0.0709 | 0 | TRUST_DRAFT_GATE_PASS |
| 07 | 0 | — | 0 | AGENT_ABSTAIN |
| 08 | 0 | — | 0 | AGENT_ABSTAIN |
| 09 | 0 | — | 0 | AGENT_ABSTAIN |

**LOCAL_ACTIVE @06**：`fast_winner_e1v2_repair_level_shift`。`g1_agentic_pipeline_report_T233.json:7476-7478`。07–09 检索到该 skill（`retrieved_skill_ids` / `active_before`），但 `proposals=[]`、0 探测、`AGENT_ABSTAIN`。技能后无可测家族命中（0/0），也无 harm。**技能形成了，后续任务没有部署它。**

### 4.2 A5

| 任务 | 真实探测 | material+ / 家族 / gain | harm n / sum | `stop_reason` |
| --- | ---: | --- | --- | --- |
| 01 | 2 | 0 | 1 / 0.0609 / repair_level_shift（第二探 period_complete 0.0） | AGENT_ABSTAIN |
| 02 | 0 | — | 0 | AGENT_ABSTAIN |
| 03 | 0 | — | 0 | AGENT_ABSTAIN |
| 04 | 1 | 1 / repair_level_shift / +0.0353 | 0 | TRUST_DRAFT_GATE_PASS |
| 05 | 1 | 1 / outlier_mad / +0.1132 | 0 | TRUST_DRAFT_GATE_PASS |
| 06 | 0 | — | 0 | AGENT_ABSTAIN |
| 07 | 1 | 1 / outlier_mad / +0.1034 | 0 | TRUST_DRAFT_GATE_PASS |
| 08 | 1 | 1 / outlier_mad / +0.0259 | 0 | TRUST_DRAFT_GATE_PASS |
| 09 | 1 | 1 / outlier_mad / +0.0462 | 0 | TRUST_DRAFT_GATE_PASS |

技能时间线：

- **LOCAL_DRAFT @04**：repair_level_shift pending，`delayed_rejected`，`RESTRICTED`
- **LOCAL_ACTIVE @05**：`fast_winner_e1v2_outlier_mad`，delayed +0.105。`g1_agentic_pipeline_report_T233.json:6671-6673`
- **检索未部署 @06**：skill 在 before 里，无提案、0 探测
- **复用部署 @07/08/09**：三次 `deployed_existing_skill`，均为 outlier_mad material-positive；07 的 delayed +0.0628 且 `delayed_ok=true`。`g1_agentic_pipeline_report_T233.json:8298-8351`（07）；08/09 同结构在 `:8907`、`:9995`

轨迹：前半命中 1/2、harm 1/4；后半 4/4、0/5。技能后 3/3 命中、0/4 harm。首次 material-positive 在任务 04。

---

## 5. `g3d1_electricity_skill_only_ab.json`（附加；electricity 顺序 AB）

A3：0 draft / 0 ACTIVE / 0 复用。11 次真实探测、9 次 harm；仅 06 一次 outlier_mad +0.0419。与 calib 低位 COLD 同类。

A5：仅 **LOCAL_DRAFT @05**（先探 repair_level_shift harm −0.1259，再探 outlier_mad +0.1144，2 探才命中正向；`delayed_rejected`）。无 ACTIVE，无复用。后半 harm 5/5。`first+mean=2.0` 来自这一次 2 探才命中——本盘点里少见的“到达首个正向所需探测数 > 1”。

---

## 技能事件计数（纳入的运行 × 臂）

| 运行 × 臂 | LOCAL_DRAFT（pending） | 新 LOCAL_ACTIVE | 复用部署 | 检索到未部署 |
| --- | ---: | ---: | ---: | ---: |
| r3 A3 | 2 | 1 | 1 | 2 |
| r3 A5 | 2 | 1 | 2 | 1 |
| calib01 COLD | 1 | 0 | 0 | 0 |
| calib01 WARM | 2 | 2 | 2 | 0 |
| calib02 COLD | 0 | 0 | 0 | 0 |
| calib02 WARM | 2 | 1 | 1 | 2 |
| calib03 COLD | 0 | 0 | 0 | 0 |
| calib03 WARM | 0 | 0 | 0 | 0 |
| g1 T233 A3 | 1 | 1 | 0 | 3 |
| g1 T233 A5 | 2 | 1 | 3 | 1 |
| g3d1 A3 | 0 | 0 | 0 | 0 |
| g3d1 A5 | 1 | 0 | 0 | 0 |
| **合计** | **13** | **7** | **9** | **9** |

12 条顺序轨迹里，7 条从未形成 LOCAL_ACTIVE。形成了的 5 条（r3 两臂、calib01 WARM、calib02 WARM、g1 两臂）里，复用部署从 0 次到 3 次不等。

---

## 三类轨迹读数（方向，不是检验）

### A. 首选家族命中正向率

| 运行 × 臂 | 前半 → 后半 | 技能前 → 技能后 | 方向 |
| --- | --- | --- | --- |
| r3 A3 | 1/3 → 2/5 | 1/4 → 1/3 | 弱升 / 近平 |
| r3 A5 | 0/4 → 4/5 | 1/5 → 2/3 | 升 |
| calib01 COLD | 0/4 → 2/4 | （无 ACTIVE） | 升，但无技能 |
| calib01 WARM | 2/4 → 4/5 | 2/5 → 3/3 | 升 |
| calib02 COLD | 0/4 → 1/5 | — | 近平 |
| calib02 WARM | 0/3 → 3/5 | 1/4 → 1/3 | 后半升，技能后近平 |
| calib03 COLD | 0/4 → 1/5 | — | 近平 |
| calib03 WARM | 0/4 → 0/5 | — | 无正向 |
| g1 T233 A3 | 0/3 → 1/2 | 0/4 → 0/0 | 技能后无探测 |
| g1 T233 A5 | 1/2 → 4/4 | 1/2 → 3/3 | 升 |
| g3d1 A3 | 0/4 → 1/5 | — | 近平 |
| g3d1 A5 | 0/3 → 1/5 | — | 近平 |

升的轨迹（r3 A5、calib01 WARM、g1 T233 A5）都恰好是形成了 skill 并且后来部署了它的臂。calib01 COLD **没有 skill 后半也升**。多数臂接近不升。

### B. 每任务 material-harm 率

| 运行 × 臂 | 前半 → 后半 | 技能前 → 技能后 | 方向 |
| --- | --- | --- | --- |
| r3 A3 | 2/4 → 3/5 | 3/5 → 2/3 | 不降 |
| r3 A5 | 4/4 → 1/5 | 4/5 → 1/3 | 降 |
| calib01 COLD | 3/4 → 1/5 | — | 降，无技能 |
| calib01 WARM | 2/4 → 1/5 | 3/5 → 0/3 | 降 |
| calib02 COLD | 4/4 → 3/5 | — | 弱降 |
| calib02 WARM | 3/4 → 1/5 | 3/5 → 1/3 | 降，但 08 有一次大 harm |
| calib03 两臂 | 高 → 高 | — | 不降 |
| g1 T233 A3 | 1/4 → 1/5 | 2/5 → 0/3 | 技能后 0 探测，harm 消失不能当成变好 |
| g1 T233 A5 | 1/4 → 0/5 | 1/4 → 0/4 | 降 |
| g3d1 A5 | 3/4 → 5/5 | — | 升（更差） |

同配置 calib 的 harm 任务数本身就在 3–9 之间晃（`calib_a3_variance_report_v1.md:55`）。观察到的“技能后 harm 下降”没有超出这条噪声带。

### C. 到达首个正向所需探测数

有探测且命中 material-positive 时，**探测序号几乎全是 1**（g3d1 A5 任务 05 是唯一清楚的 2）。前半 vs 后半、技能前 vs 技能后都比不出差异。

臂级“第一次 material-positive 出现在第几个任务”：r3 A3=03，r3 A5=05，calib01 COLD=07，calib01 WARM=02，calib02 COLD=06，calib02 WARM=05，calib03 COLD=06，calib03 WARM=无，g1 A3=06，g1 A5=04，g3d1 A3=06，g3d1 A5=05。与方差报告写的 2–7 或 never 重合。**这个指标在 9 任务上排不了名。**

---

## 可指认复用案例

判定规则：后续任务 `deployed_existing_skill`（或提案明确引用 `fast_winner_*`），且该任务 material-positive，并且 support_gain 不低于此前同家族任务。伤害案例：引用/探测了已形成技能的家族，且该任务 material-harm。

### 复用收益（可指认）

1. **最强：g1 T233 A5。** 任务 05 形成 `fast_winner_e1v2_outlier_mad`（support +0.1132，delayed +0.105）。任务 07/08/09 三次部署同一 skill，support +0.1034 / +0.0259 / +0.0462，均 material-positive；07 delayed 仍为正。证据：`g1_agentic_pipeline_report_T233.json:6671-6673`（形成）、`:8298-8351`（07 部署）。任务 08 的 +0.0259 低于形成轮，但仍为正，算复用命中，不算“更好”。07 与形成轮同量级且 delayed 确认，是这条链里最干净的一轮。
2. **r3 A5 任务 07 / 09。** 06 形成 hampel（+0.0189）。07 部署 +0.0795（高于形成轮），09 部署 +0.0161 且 delayed_ok。`r3_full_ab_electricity.json:5682-5750`、`:7250-7264`、`:9083-9155`。07 的 delayed 为负，support 变好没有变成 delayed 变好。
3. **calib01 WARM 任务 07。** 06 形成 outlier_mad（+0.0419），07 部署 +0.0838。`calib_a3_run01.json:7671-7684`。与 r3 A3/A5 任务 07、calib02 WARM 任务 07 是同一家族、同一量级的 support 读数，跨运行重复出现。
4. **r3 A3 任务 07。** 06 形成 outlier_mad（+0.0419），07 部署 +0.0838。`r3_full_ab_electricity.json:6209-6273`、`:6810-6824`。delayed 为负，winner 回到 RESTRICTED。

### 复用伤害 / 未部署伤害

1. **r3 A3 任务 09。** skill `fast_winner_e1v2_outlier_mad` 仍在 `active_before`；本任务提案并探测 outlier_mad，`support_gain=-0.0617`（material-harm）。`r3_full_ab_electricity.json:8482-8488`。这是可指认的同家族复用伤害：不是“没看见 skill”，而是看见了还按该家族探了一枪。
2. **r3 A5 任务 08、calib02 WARM 任务 08。** skill 在 before 里，提案改走 repair_level_shift / smooth_ema，分别 harm −0.1089 / −0.3975。算“有 skill 但没部署，另选家族受伤”，不是 skill 本身致害。
3. **g1 T233 A3 任务 07–09。** skill 被检索，零提案零探测。无收益也无 harm，适应链在这里断开。

没有足够样本断言收益案例比伤害案例“多数成立”。两边都存在，而且收益案例集中在同配置样本已经能长出 1–3 个 LOCAL_ACTIVE 的高位轨迹上。

---

## 汇总判定

**`IN_DOMAIN_ADAPTATION_UNMEASURABLE_AT_CURRENT_SAMPLE`**

理由（对照 calib 波动，而不是把单臂故事说满）：

1. **同配置就能覆盖本盘点看到的全部“适应模样”。** `calib_a3_variance_report_v1.md`：7 条 A3 配置轨迹 `LOCAL_ACTIVE ∈ {0,3}`，pair |Δ| 已到 3；harm n 3–9；首次正向任务 2–7 或没有。r3 A5 / calib01 WARM / g1 T233 A5 的“技能后变好 + 可指认复用”与这条高位轨迹同类；calib03 两臂、g3d1 A3、多数 COLD 则是零技能低位。无法把高低位之差读成“经验积累导致适应”。
2. **形成技能不是多数轨迹的事实。** 纳入的 12 条顺序臂里 7 条从未 LOCAL_ACTIVE。判定条里的“多数顺序运行里技能形成后轨迹改善”前提（技能形成了）在当前样本上就不成立。
3. **形成了也不一律变好。** r3 A3 技能后同家族 harm；g1 T233 A3 检索到但不提案；calib02 WARM 复用一轮后改走别的家族并大 harm。
4. **探测分辨率撑不起“更快找到正向”。** 每任务真实探测几乎是 0 或 1；命中时探测序号几乎总是 1。前半/后半比这个量没有内容。
5. **可指认复用案例存在，但数量少、且与噪声带重叠。** 最干净的一串是 g1 T233 A5 的三次 outlier_mad 部署；它不能单独把判定从“不可测”抬到“信号已在”。

因此：域内“越跑越好”在现有顺序工件上 **看得见单条故事，测不出稳定方向**。下一步若要测，需要的是更多同协议顺序重复（让 LOCAL_ACTIVE 与复用部署的波动带先收敛），而不是再加 Gate 或新摘要格式。

---

## 未列入主表的其他顺序工件（出处）

`g3_development_electricity.json`：A3 在任务 02 即形成 outlier_mad，后续 7 个任务都检索到 skill，家族命中 0/1 → 4/7，但仍有 3/7 harm。`g2_close_T233.json`：两臂都在任务 06 形成 repair_level_shift；A5 任务 09 有一次部署且 material-positive，A3 07–09 检索到但未稳定部署。`g4_source_experience_electricity.json`、`g1_agentic_pipeline_report.json` 同样是 9-task 顺序可积累。它们不改变上面的判定：单臂可以看起来在适应，同协议重复时方向不稳。
