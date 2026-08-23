# #42d r1+r2+r3 回报

`evidence_grade = NATURAL / provisional`。未 commit。未重跑全量 fresh confirmation。未另开正式 `--evaluate`。

总判定按阶段：

| 阶段 | 判定 |
|---|---|
| Part 0b | **FORECASTING_COMPAT_RESTORED** |
| Part B | census 成立；合法条件化结构性不可用 |
| Part C | **SOURCE_SKILL_WRITTEN**（risk-only） |
| Part D | **SCOPE_CORRECT_NO_APPLICABLE** |

---

## Part 0 / 0b

两处手写旧 ID 已改调公有 `fast_winner_skill_id(episode)`：

- `evaluation/functional/run_e2_local_skill_recall.py`（原 :411）
- `evaluation/functional/run_e2_fresh_confirmation.py`（原 :1866）

0-LLM 缓存重放 6/6（sol 点名 6，工件实数 6）：

| 来源 | slot | 程序 | 旧 ID | 新 ID | 门 |
|---|---|---|---|---|---|
| LSR | a5_t1 | outlier_iqr | `fast_winner_e1v2_outlier_iqr` | `fast_winner_forecast_ridge_smase_e1v2_outlier_iqr` | pending → approved POSITIVE，检索命中 |
| LSR | a5_t2 | outlier_iqr | 同上 | 同上 | 同上 |
| LSR | a3_t2 | repair_level_shift | `fast_winner_e1v2_repair_level_shift` | `fast_winner_forecast_ridge_smase_e1v2_repair_level_shift` | 同上 |
| FC | a5_pooled | outlier_mad | `fast_winner_e1v2_outlier_mad` | `fast_winner_forecast_ridge_smase_e1v2_outlier_mad` | 同上 |
| FC | a5_per_channel | repair_level_shift | `fast_winner_e1v2_repair_level_shift` | `fast_winner_forecast_ridge_smase_e1v2_repair_level_shift` | 同上 |
| FC | a3_per_channel | repair_level_shift | 同上 | 同上 | 同上 |

残留拼接（本轮只修点名两处，其余入册不动）：

- `evaluation/functional/run_e2_t5_lifecycle_dual_consumer.py:1100` 前缀检查 `fast_winner_%s_`
- `methods/ttha/method.py:270` 公有构造器本体
- `methods/ttha/method.py:1498` docstring 举例
- 0b 重放脚本自己的检索针（不是生产拼接）

成本：LLM 0 / AD 0 / 重训 0。

工件：`artifacts/functional/e2/t6_nab_42d_part0b_forecasting_compat.json`

---

## Part A

删除 `run_e2_t6_natural_a5_a3.py` 原 :1619
`memory = bank_episodes if arm == "A5" else ()`。
双臂构造期 `experience_episodes` 断言为空。A5' 只经快照带 Skill，不经 Memory 带 bank。

---

## Part B census（0 LLM / 0 fit）

输入：v2 `/source_bank/rows` + `episodes_to_dict`，20 条，不重评。

- 证据单元 = `episode_id`
- 承重 = `delayed_relation`
- identity ABSTAIN 不计票
- UNGUIDED 断言成立（bank 工件无 `fast_winner_` / 整合 Skill 字面）

布尔特征 3 个：

| 特征 | 划分 | Scope |
|---|---|---|
| `level_only_post_shift_support_sufficient` | 恒 True | 无分辨力，禁用 |
| `period_repair_available` | 恒 False | 无分辨力，禁用 |
| `post_shift_support_sufficient` | aws 全 False / known_cause 全 True | **完美 cohort 代理，禁用** |

合法可观察条件化：**结构性不可用**。授权改无条件 pool，计票单位 = distinct Source cohort。

已知限制：winsorize 两条 delayed POSITIVE 都是 `source_aws_cloudwatch`（r1/r2 异窗同 cohort）。

授权（跑前预注册，实测相符）：

- TRY = `[]`（winsorize 仅 1 cohort 正，且 known_cause 有负）
- RISK = `['hampel_filter']`（aws r1 NEGATIVE + kc r2 CONFLICT，全 pool 零 POSITIVE）
- 扩展伤害口径 = `{NEGATIVE, CONFLICT}`；严格口径仅 NEGATIVE 时 hampel 只有 1 cohort，过不了门

工件：`artifacts/functional/e2/t6_nab_42d_source_census.json`

---

## Part C Slow（1 / 8 LLM）

`source_skill.py` 最小参数化：`slow_system` / `build_skill_payload` 可选 `skill_id`、`applicability`。默认路径 3/3 字节等同测试通过。

AD 封装：`evaluation/functional/task_episode_harness/agentic/ad_source_skill.py`
`skill_id=source_investigation_ad_v1`，`task_kind==anomaly_detection`。

Slow 一次 ADD：

- TRY = `NO_AUTHORIZED_ACTIVE_RECOMMENDATION`
- RISK 载 `hampel_filter` 降权（非禁止）
- 六段齐全，包含审计全过，无 cohort 泄漏、无数值阈、无发明算子
- 写入 h0s：`c5e5a7346b0201d39f038cb88d90651f5949a40b5a61fcc328409790ff06e595`
- 技能 id 列表含 bootstrap 三件 + `source_investigation_ad_v1`
- 不携冻结程序

工件：`artifacts/functional/e2/t6_nab_42d_source_skill.json`

---

## Part D 同跑配对（LLM 24/32，AD 12/120）

one-shot：`run_id=20260823T090444Z`，锁文件 + 隔离 Store `t6d42_20260823T090444Z`。
A3 = h0 + 空店；A5' = h0s + 空店。构造期 Memory 空。顺序沿冻结反平衡。

启动第一次因 Windows 无 `pgrep` 在验尸处机械失败（未读 Target）。改成本机 WMIC 后只重跑这一次。

| cell | pool → chosen | 非 identity | 部署 / Skill |
|---|---|---|---|
| CPC A3 r1 | 仅 identity | 0 | abstain |
| CPC A5' r1 | 仅 identity | 0 | abstain |
| CPC A3 r2 | 仅 identity | 0 | abstain |
| CPC A5' r2 | 仅 identity | 0 | abstain |
| CPM A5' r1 | 仅 identity | 0 | abstain |
| CPM A3 r1 | 仅 identity | 0 | abstain |
| **CPM A5' r2** | `outlier_mad_local_extreme_deviation` | 1 | Support +0.0593 POSITIVE → delayed +0.0111 POSITIVE → **LOCAL_ACTIVE / activate_approved** |
| **CPM A3 r2** | `localized_outlier_iqr`（池里还有 mad） | 1 | Support +0.1481 POSITIVE → delayed −0.0028 **NEUTRAL** → 拒批，`EPISODE_ONLY`；受害序列 `exchange-4_cpm_results.csv` |

hampel 提案率：A3 = 0，A5' = 0。
非 identity 试验：两臂各 1。
未坍缩成 identity-only。

标签墙：`breached=false`。6 次请求均 `granted`，且全部是 CPM 三文件（CPC 未探测，未要标签）。

主读数（r3）：有边界劝退 vs 全局劝退。本次 **hampel 在两臂都未上场**，风险条款没有适用场合；其余算子仍有探索。故 **SCOPE_CORRECT_NO_APPLICABLE**，不是 UNCHANGED_COLLAPSE。

#42 历史 A3/A5 数字只作描述性附录，判定未引用。

工件：`artifacts/functional/e2/t6_nab_42d_paired_replay.json`

---

## 成本总账

| 段 | LLM | AD fit | 重训 |
|---|---|---|---|
| 0b | 0 | 0 | 0 |
| B | 0 | 0 | 0 |
| C | 1 | 0 | 0 |
| D | 24 | 12 | 0 |
| **合计** | **25 / 40** | **12 / 120** | **0** |

---

## 歧义（只报不选）

1. Part D 自动格写成 `SCOPE_CORRECT_NO_APPLICABLE`，因为预注册主读数是 hampel 提案率，两臂都是 0。同跑里 A5' 激活了 `outlier_mad`、A3 的 `outlier_iqr` 被 delayed NEUTRAL 拒批——这是行为差，但不是 hampel 劝退差。未改判定规则去贴“更快/更安全”。
2. A5' 的 `retrieval_before_round.held=0` / `memory_resolution=no_memory` 是空 Experience 店的预期；Skill 在 snapshot，不在 Memory。本轮没有单独把 harness view 里的 Skill id 打进 evaluate 读数。
3. CPC 六格全部 identity-only：与 #42 竞态样本的“A5 0 次非恒等”结构同类，但本次 A3 在 CPC 同样 0 次，不能用历史数字当因果基线。
4. CONFLICT 计伤害是 r3 唯一新解释点；本 bank 下它决定了 hampel 能否进 RISK。已按扩展口径授权，严格口径并报在 census 里。

---

## 交付（未 commit）

- `evaluation/functional/run_e2_local_skill_recall.py`
- `evaluation/functional/run_e2_fresh_confirmation.py`
- `evaluation/functional/run_e2_t6_forecasting_compat_0b.py`
- `evaluation/functional/run_e2_t6_natural_a5_a3.py`（空店 + `snapshot_for_arm`）
- `evaluation/functional/task_episode_harness/agentic/source_skill.py`（默认兼容参数化）
- `evaluation/functional/task_episode_harness/agentic/ad_source_skill.py`
- `evaluation/functional/run_e2_t6_42d_consolidation.py`
- `tests/functional/test_source_skill_default_bytes.py`
- `artifacts/functional/e2/t6_nab_42d_part0b_forecasting_compat.json`
- `artifacts/functional/e2/t6_nab_42d_source_census.json`
- `artifacts/functional/e2/t6_nab_42d_source_skill.json`
- `artifacts/functional/e2/t6_nab_42d_paired_replay.json`
- `artifacts/functional/e2/t6_nab_42d_report.md`
