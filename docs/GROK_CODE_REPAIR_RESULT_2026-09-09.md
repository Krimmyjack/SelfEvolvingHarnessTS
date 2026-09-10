# Grok 代码修复回执（A1–A3）

日期：2026-09-09。角色：A，修复者。基底 Git HEAD：`c845b4b87cd9068929ef1d7cb0062684a6e83a65`。  
本包只改未来执行路径。历史 DEV-DEPLOY-1 / DEV-DEPLOY-2 数值、18/240 与第一包 9 条空选择结论不因代码修好而自动变正，也不声称当时实验已经使用修正版本。

Astra 已完成代码复核、补充修复与独立回归，见 §8。以下保留 Grok 的执行过程；工程验收通过不等于学习有效。

## 0. 工作区与禁止项

启动时已有、本任务未改动的他线路径：

- 已修改：`AGENTS.md`，`docs/DECISIONS.md`，`docs/FABLE_FINAL_REVIEW_AND_SUCCESSOR_BRIEF_2026-09-03.md`，`docs/HEC1_INDEPENDENT_REVIEW_CHECKLIST_2026-09-03.md`，`docs/SLOW_FEEDBACK_UPDATE_DESIGN_NOTES_2026-09-08.md`
- 未跟踪（未碰）：`.dev_*_runs/`，若干 `artifacts/main_protocol/hec1_course_pytest_*.json`，`docs/FRAMEWORK_WALKTHROUGH_2026-09-09.md`，`docs/GROK_REPAIR_AND_ARCHITECTURE_REVIEW_TASK_2026-09-09.md`，`idea-stage/`，`refine-logs/`

未做：Git 提交/切换/重置；项目 LLM 调用；Consumer fits；sealed / TARGET_HELD_IN / +144 读取；`methods/`、`runtime/`、`contracts/`、AGENTS、DECISIONS、状态页或其他报告的修改；新 SHA；凭据打印。  
本次 Grok 研发调用不是实验 LLM，不能把整包工作写成零模型成本。测试本身 0 项目 LLM、0 fits。

写入白名单四文件：

- `evaluation/main_protocol_p4/run_dev_deploy1.py`
- `evaluation/main_protocol_p4/run_dev_deploy2.py`
- `tests/main_protocol/test_dev_deploy_feedback_integrity.py`
- `docs/GROK_CODE_REPAIR_RESULT_2026-09-09.md`

## 1. A1 修改前后语义

### 前（HEAD 行为）

`run_dev_deploy1.py` 的 `run_fastonly_decision` 调用 `method.prepare(...)` 后丢弃 `PreparationResult`，只读 `method.last_trace`。`methods/ttha/fast_agent.py:1326–1352` 在 `AgentProtocolError` 等情况下返回 `PreparationStatus.FAILED` 且仍带非空 `DecisionTrace`（`chosen_candidate_id` 可能是残留的 `"identity"` 或空串）。外部 runner 因此：

| 实际内部状态 | 旧 `deploy_status` | 旧主读数 |
| --- | --- | --- |
| `prepare()` 抛异常且 `trace is None` | `FAULT_NO_DECISION` | `UNKNOWN` |
| `FAILED` + 非空 trace，chosen 空 | `EMPTY_SELECTION_FALLBACK_TO_IDENTITY` | `0.0`（冒充 identity） |
| `FAILED` + 非空 trace，chosen=`identity` | `IDENTITY_SELECTED` | `0.0` |
| 明确选择 identity（`ABSTAINED`） | `IDENTITY_SELECTED` | `0.0` |
| chosen 不在候选图 | `UNKNOWN_CANDIDATE_FALLBACK_TO_IDENTITY` | `0.0` |
| 合法程序 + verifier 通过 | `DEPLOYED` | 评分 |
| 合法程序 + verifier 拒绝 | `LEGALITY_FALLBACK_RAW` | `0.0` |
| 账户/权限异常 | `AccountFault`，整线停止 | （保持） |

`faults=[]` 不能证明决策成功。`run_dev_deploy2.py` 原样调用该函数，形成材料把历史 `identity/0` 当效用证据。

### 后（仅未来执行）

消费 `PreparationResult.status` 与 `ExecutionReceipt`，在选择/评分前分流。残留 trace 不是有效决定。

| 状态 | 新 `deploy_status` | 主机制读数 | 是否有效决定 |
| --- | --- | --- | --- |
| `FAILED`（即使 trace 非空） | `PREPARE_FAILED` | `UNKNOWN` | 否 |
| 成功态缺失或 `receipt.ok is False` | `INVALID_PREPARATION_RESULT` | `UNKNOWN` | 否 |
| 外部抛异常 / 无 result | `FAULT_NO_DECISION` | `UNKNOWN` | 否 |
| 空 `chosen_candidate_id` | `EMPTY_SELECTION_NO_VALID_DECISION` | `UNKNOWN` | 否 |
| chosen 不在候选图 | `UNKNOWN_CANDIDATE_NO_VALID_DECISION` | `UNKNOWN` | 否 |
| 明确选择 identity | `IDENTITY_SELECTED` | **保留 0** | 是 |
| 预定义且实际执行的合法性 raw 回退 | `LEGALITY_FALLBACK_RAW` | **保留 0**；`active_model_choice=false` | 是 |
| 合法程序路径 | `DEPLOYED` | 评分（gain 缺失仍为 `UNKNOWN`） | 是 |
| 账户/权限（抛出或 FAILED receipt） | 仍 `AccountFault`，停止，不当弃权 | — | — |

人口规则不变并收紧：缺决定的行留在分母；主均值在任一 `UNKNOWN` 时为 `None`；可读子集只进 `*_diagnostic_only`。不删行换分母。故障路径没有另报“系统 raw 兜底效用=0”，因为该路径并未执行预定义 raw 回退；若将来真执行，应单列且标明不是模型主动选择。`LEGALITY_FALLBACK_RAW` 的 0 已是那种预定义回退，留在主读数。

历史 `EMPTY_SELECTION_FALLBACK_TO_IDENTITY` / `UNKNOWN_CANDIDATE_FALLBACK_TO_IDENTITY`：原 JSON 不改；新视图把机制读数标 `UNKNOWN`，并保留 `original_*` 与 `original_record_semantics`。恢复 checkpoint 只在内存标注，不重调 LLM。

精确入口：

- 分类：`run_dev_deploy1.py:656–1129`（`interpret_prepare_outcome` 760，`apply_legality_and_score` 845，`finalize_fastonly_decision` 904，`annotate_recorded_decision` 964，`summarize_fastonly_unit` / `mechanism_unit_stats` 同文件后部）
- 决策消费：`run_fastonly_decision` 约 1319–1353
- 恢复：`run_fastonly_unit` 1366–1380；含历史空选择的源 checkpoint 不覆盖，旁路写 `*.feedback_integrity.json`
- 汇总：`aggregate_part_b`；CLI `--part fastonly` 在缺决定时退出码 2（`main` 1633–1639）
- 形成材料：`run_dev_deploy2.py:load_pooled_formation` 176–184（无效行不按 identity 重构）
- 部署2 新视图均值：`_arm_review_mean` 1039–1072（重读行、不改原 JSON）；`_paired_diff` 在缺数时为 `UNKNOWN`

## 2. A2 成对程序对照

`_render_menu` 原先删掉 `per_series_gain`，提示文字仍声称该字段存在。现最小实现：R/C（`include_effects=True`）保留已有 `per_series_gain`，按同一 origin / UID / program 对齐；不挑选“最好看”的对子；未测候选保持 `UNAVAILABLE` / `LEGALITY_REJECTED`，不填零。identity 的 0 是真零。C-minus 仍只见 `status`，无增益、无逐序列图、无赢家标签。R 与 C 拿同一份效果菜单。

`_render_case` 对 A1 标出的历史错误行：效果臂把机制增益写成 `UNKNOWN`，另给 `original_recorded_gain_*` 并标明不是有效效用证据；C-minus 不带这些数字。未改材料分组，未新增错误类别体系。

这不是比较式学习有效的证据。历史 C/R 仍是其当时聚合材料上的对照。

精确入口：`run_dev_deploy2.py:_render_case` 519–553，`_render_menu` 556–582，菜单说明 611–620 一带。

## 3. 测试命令与实际结果

WSL 系统 `/usr/bin/python3` 无 numpy。按任务书使用 Windows Conda 环境 `project`：

```text
D:\Anaconda_envs\envs\project\python.exe -m pytest tests/main_protocol/test_dev_deploy_feedback_integrity.py -q --tb=short
```

实际：

```text
.........................                                                [100%]
25 passed in 4.19s
```

解释器：`3.10.19 | conda-forge`，`numpy 2.2.6`，`pytest 9.1.1`。未安装依赖，未改环境。  
前一稿 16 个 helper 测试；随后补了 9 个恢复/CLI/汇总/菜单身份集成测试，合计 25。

逐项（不是只报通过数）：

| 要求 | 测试 | 结果 |
| --- | --- | --- |
| 内部 FAILED 且 trace 非空，不当 identity | `test_failed_prepare_with_nonempty_identity_trace_is_not_identity` | 过。`PREPARE_FAILED`，增益 `UNKNOWN`，`prepare_error.trace_present=True`，残留 chosen=`identity` 不采信 |
| 外部抛异常，不当 identity | `test_external_exception_is_not_identity_zero` | 过。`FAULT_NO_DECISION` / `UNKNOWN` |
| 明确 identity 保留合法零 | `test_explicit_identity_keeps_legal_zero` | 过。`0.0` |
| 合法程序路径不变 | `test_legal_program_path_still_scores_the_chosen_steps` | 过。verifier 通过 → `DEPLOYED` 0.5；拒绝 → `LEGALITY_FALLBACK_RAW` 0.0 |
| 账户错误停止 | `test_account_or_permission_error_still_stops_not_abstains` | 过。402 / `insufficient_user_quota` / FAILED receipt 均 `AccountFault` |
| 空选择与未知候选 | `test_empty_and_unknown_candidate_are_not_identity_fallback` | 过。新状态名，增益 `UNKNOWN` |
| 一条缺决定 → 人口主读数 UNKNOWN | `test_one_missing_decision_makes_population_mean_unknown` | 过。主均值 `None`，诊断子集 0.0，n_total=2 |
| 缺行留在分母 | `test_missing_population_member_stays_in_denominator_as_unknown` | 过 |
| 两个合法 identity 零仍均值 0 | `test_two_legal_identity_zeros_still_mean_zero` | 过 |
| 历史空选择读取不改原 JSON | `test_historical_empty_selection_view_does_not_rewrite_source` | 过。字节前后相等；视图 `UNKNOWN`；`original_*=0.0` |
| P/Q 总体相同、逐序列反号 | `test_menu_render_keeps_per_series_sign_flips_and_true_zeros` | 过。T1 上 P=+1 Q=-1；identity 真零；unmeasured 无零图 |
| C-minus 无效果/赢家泄漏 | `test_cminus_menu_and_case_have_no_effect_or_winner_leak` | 过。R 与 C 菜单/有效案例相等；C-minus 无 gain / per_series_gain / winner |
| 凭据文本不进错误记录 | `test_compact_prepare_error_redacts_credential_text` | 过。错误字段为 `[redacted error detail]`，不含 token/密钥原文 |
| 1–4 步可编译；合池截断 | `test_propose_schema_allows_one_to_four_steps_and_three_candidates`，`test_candidate_pool_truncates_to_total_k_including_identity` | 过。schema maxItems=3、steps 1–4；`total_k=4` 时第 4 个 agent 候选被截 |
| 单步默认 vs 两步夹具 | `test_default_single_step_actionability_vs_two_step_compile` | 过。见 §4 |
| 非成功 receipt / 非法 status | `test_unsuccessful_receipt_and_invalid_status_are_not_valid_choices` | 过。`INVALID_PREPARATION_RESULT` / `UNKNOWN` |
| FAILED 计入 faults 且不留密钥前缀 | `test_internal_failure_is_counted_and_sensitive_prefix_not_retained` | 过 |
| 形成行已标 UNKNOWN 仍保留原 0 注释 | `test_historical_pooled_alias_retains_original_zero_only_as_annotation` | 过 |
| 部分 resume 不覆盖/不重问旧行 | `test_partial_legacy_resume_does_not_overwrite_or_requery_old_row` | 过。源 JSON 字节不变；只对未完成 UID 调用 Fast；旁路 `*.feedback_integrity.json` |
| 账户故障停止后续 unit | `test_account_failure_stops_following_units` | 过。只跑 position 9，状态 `BOUNDARY_ACCOUNT_OR_PERMISSION_FAULT` |
| CLI 不完整批次非 0 退出 | `test_part1_cli_returns_failure_for_incomplete_batch` | 过。退出码 2 |
| 部署2 汇总重读旧行不当旧均值 | `test_part2_aggregation_rechecks_legacy_rows_not_old_mean` | 过。原文件不变；`complete is False` |
| 部分 +48 UNKNOWN 不改用可读子集当主均值 | `test_part2_partial_delayed_does_not_use_readable_subset` | 过。origin 完整均值 0；delayed 为 `None`；差为 `UNKNOWN` |
| 菜单程序身份对 C-minus 中性可见 | `test_menu_typed_parameters_are_neutral_in_all_arms` | 过。`program_steps` 两臂相同；C-minus 仍无 gain |

## 4. 组合空间只读诊断（未改菜单/槽位/verifier）

区分：**组合可编译** ≠ **模型会自然生成好组合**。

源码事实（未改）：

- `methods/ttha/schemas/fast_propose_v1.json`：最多 3 个候选，每案 1–4 步
- `methods/ttha/harness/h0/candidate_policy.json`：`total_k=4`（identity + 3）
- `methods/ttha/exploration_policy.py`：默认 `agent_proposals_kept=1`（Skill 合池时 agent 提案再截到 1）
- `_actionable_operators`（`fast_agent.py:209`）用**默认单步**测 0.35 verifier，失败则该算子不进 propose 合约

零 LLM / 零 Consumer 夹具（100 点：60 个 0 + 40 个 1000；以及同一序列挖 10 个 NaN）：

| 对象 | 默认单步 0.35 | 两步 |
| --- | --- | --- |
| `denoise_stl` | 拒，`MODIFICATION_FRACTION_EXCEEDED`，modified_fraction=0.45 | `impute_linear → denoise_stl` 同样 0.45 拒 |
| `outlier_iqr` / `winsorize` / `impute_linear` / `period_median_complete` | 本夹具可选择（本夹具 IQR/winsorize 实际改动 0） | `period_median_complete → outlier_iqr` 等可选择 |

**未发现**“默认单步失败但含该算子的组合合法”的误剪枝例子。`denoise_stl` 是单步与组合一并被 0.35 门拒绝，属于保守排除，不是“组合本可通过却被单步探针剪掉”。本夹具不能代表训练窗 verifier 或 origin-2856 的 `period_median_complete → outlier_*` 头寸。未改算子可见菜单、槽位、冻结程序供给或 verifier。

## 5. 未测项

- 真实 `TTHAMethod.prepare` / 中转 / 模型；`run_fastonly_decision` 的完整 machinery 构造（分类与评分经抽出的纯函数覆盖）
- `ScopeExecutor.verify` 真执行（测试注入 `.passed`）
- `Scorer.sequence_reading` 真 Consumer（注入固定 gain）
- 对真实 `_scratch/dev_deploy1` 或 `artifacts/main_protocol/dev_deploy*_*.json` 跑 `load_pooled_formation` / `aggregate_part_b` CLI（避免碰原工件与 fits）。标注逻辑用夹具覆盖
- 季节性长序列上的 `denoise_stl`、P4 训练窗 200-window 全有或全无 verifier
- 并行 checkpoint 写入与 `PackageCeiling` 路径
- 修复后的 live Fast-only 重跑

## 6. 未决问题（交 Astra）

1. **Astra 裁定：FAILED 携带 trace 正常返回是合法契约，不是仍未修好的根因。** 诊断 trace 可以保留；缺陷是 caller 忽略 FAILED。本轮在 runner 消费返回状态即可，不需要因此修改 `methods/ttha/fast_agent.py` 或改成一律抛异常。
2. 若有人用新 `aggregate_part_b` 重读旧 `dev_deploy1_part_b__*.json`，含历史空选择的单元主均值会变成 `None`（新视图）。原文件未改。不要把这写成旧实验当时的读数。
3. Astra 已补充恢复路径：遇到含历史空选择的旧检查点，后续工作视图写到相邻 `.feedback_integrity.json`，不写回原件，不重调旧行；下次恢复使用该工作视图。串行部分续跑夹具已检查原件逐字节不变。这是旧数据的新解释视图，不是重做历史实验。
4. 组合误剪枝在本夹具上未证实；若要在默认单步失败、绑定/组合合法的算子上找最小例子，需要另开零 LLM 探针，仍不应顺手改菜单。
5. 本包不证明比较式学习、不重算 18/240 或第一包 9 条。

## 7. 完整改动清单

1. `evaluation/main_protocol_p4/run_dev_deploy1.py` — 消费 `PreparationResult`；FAILED / 非法 receipt / 空选择 / 未知候选 → `UNKNOWN`；identity 与合法性 raw 保留 0；账户错误停止；历史空选择旁路 checkpoint；CLI 缺决定退出码 2；人口缺行占位。
2. `evaluation/main_protocol_p4/run_dev_deploy2.py` — 恢复 `per_series_gain`；菜单保留 typed `program_steps`（非效果）；R/C 同材料；C-minus 无增益泄漏；形成材料与 `_render_case` 标记历史空选择；`_arm_review_mean` 重读行、不改原 JSON。
3. `tests/main_protocol/test_dev_deploy_feedback_integrity.py` — 25 个定向测试（上表）。
4. `docs/GROK_CODE_REPAIR_RESULT_2026-09-09.md` — 本回执。

Grok 初交状态：实现完成，定向测试已跑，等待 Astra。当前最终状态见 §8；不是新实验授权。

## 8. Astra 最终复核（2026-09-09）

Grok A 完成了主补丁与首批 16 项测试；Astra 在复核中补了以下同范围遗漏，
Grok 也保留这些并发改动并复验了最终 25 项定向测试：

- 不成功的 ExecutionReceipt/未知状态不得成为有效选择；内部返回失败也进入 faults，
  不能继续宣称 `faults=[]`；敏感错误信息不保留可能含凭据的冒号前缀。
- 账户错误即使被单元捕获，仍向组级传播停止；第一包 CLI 对缺决定/中止返回 2。
- 历史检查点不覆盖，合法原始零与无决定的 UNKNOWN 仍分开。
- 第二包汇总重读逐行状态，不采信旧的零故障均值；origin 与 +48 各自要求完整人口，
  +48 不能用可读子集填充。缺测差值返回 UNKNOWN。
- 菜单保留已有逐序列收益及生成时已知的完整步骤/参数；C-minus 可以看到中性程序
  参数，不能看到收益、风险数值或赢家标签。未声明原工件后来已经包含这些字段。

Astra 独立命令（已有 Windows `project` 环境）：

```text
D:\Anaconda_envs\envs\project\python.exe -m pytest -q -p no:cacheprovider tests/main_protocol/test_dev_deploy_feedback_integrity.py tests/methods/test_ttha_h0.py
29 passed in 5.78s
```

25 项定向测试 + 4 项既有 h0 检查，零项目 LLM、零 Consumer fits。未做真实中转/模型
重跑，未检验并发中断的所有时序，不能由此称整个 Harness 全部验收。

另外以 `_arm_review_mean` 只读现存 12 个 DEPLOY-2 工件，未写出新实验工件、未重新
拟合或调用模型：

| 组 | F 缺有效决定 | R 缺有效决定 | C 缺有效决定 |
| --- | ---: | ---: | ---: |
| g1 | 2 | 1 | 1 |
| g2 | 13 | 1 | 0 |

按新机制视图，前五个臂均不具完整人群均值；仅 g2/C 完整，origin 均值 0.085740。
所以不能用本修复宣布 C−F 已得到两次干净复现。旧数字按原回退语义保留，未覆盖；
也未删掉这 18 行重算一个冒充整体的新均值。

最终状态：**限定范围代码修复与回归已复核；Skill 改善未来 Workflow 的效用仍待验证。**
没有改 Consumer、风险线、候选槽位、算子菜单、Core/Runtime/Contracts；没有新哈希
平台/清单、没有新项目实验、没有 Git 提交。

后续运行注意：本包没有重构旧 runner 的输出命名；旧默认输出路径不可用于覆盖既有
结果。若另获授权开新 live 实验，任务书须指定新的输出位置与形成/复核边界，不能仅
把这次代码修复理解成可原命令重跑全部旧包。
