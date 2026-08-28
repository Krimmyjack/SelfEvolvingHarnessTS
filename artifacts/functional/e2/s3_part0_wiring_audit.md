# S3 Part 0 接线审计（只读）

日期: 2026-08-29。地位: Stage 3 pilot 可编辑面定位；0 LLM / 0 fit / 0 git。
协议: `docs/STAGE3_PILOT_FREEZE_DRAFT_2026-08-29.md`。种子工件: `artifacts/functional/e2/s2a_g1_run1_r2.json`。

S2a 活路径进口是 `run_online_round`（`SelfEvolvingHarnessTS.methods.ttha.online_loop`），不是 `task_episode_harness.agentic.runner.run_agentic_fast_path`。`runner.py` 在本课只提供 `live_transport`。

---

## 0. 种子复盘（协议叙事 vs 工件）

协议与 STAGE_REPORT 写: 受益单元 4 供给卡入池**被采**，挤掉更优自发现 `robust_mad_outlier_repair`，反事实 −0.1206。

`s2a_g1_run1_r2.json` `rows` 里 position=4 / `A5-online` 的事实是相反的采用关系:

| 项 | A5-online | A3-reset |
| --- | --- | --- |
| pool | `identity`, `robust_mad_outlier_repair`, `cand_skill_s2a_forecast_supply_v0` | `identity`, `local_hampel_outlier_repair` |
| chosen | `robust_mad_outlier_repair` | `identity` |
| 供给 outcome | `not_reached_support_budget_exhausted`（hampel，gain 空） | 无供给卡 |
| winner / deploy | `outlier_mad` / `FROZEN_ACTIVE_SKILL_RECALL` `fast_winner_…_outlier_mad` | `hampel_filter` |
| heldout_gain | +3.786303416048806 | +3.9069250666971445 |

A5 − A3 = **−0.1206216506483385**（协议四位 −0.1206）。K0 同单元也是 +3.9069（hampel），不在 P6 分歧表里。

P3 把单元 4 算进「供给转化」，是因为 `_verdict` 用「强受益 ∧ `candidate_sources.supplied` ∧ deploy ≠ identity」（`run_e2_s2a_forecast_curriculum.py:802-805`），**不是**「winner 的 `source_skill_id` 为供给卡」。该 Episode 的 `source_skill_id` 为 null。

此矛盾列入 **Q1**，不得在未裁定前当作「供给卡胜出」来写 LLM-edit 轨迹。

---

## 1. 入池与分配（file:line）

### 1.1 `runner.py` 在本课的角色

- `evaluation/functional/run_e2_s2a_forecast_curriculum.py:151-155` 从 `agentic.runner` 只取 `live_transport`。
- `evaluation/functional/task_episode_harness/agentic/runner.py:286-304`：`live_transport`（`M0_AGENT_*`）。
- `evaluation/functional/task_episode_harness/agentic/fast_path.py:340-403, 537, 585-718`：另一套 Fast Path（逐候选 LLM select + material 机械门）。**S2a 不调用。**

### 1.2 供给候选注入

1. 卡权限在编译时写死: `supplies_candidates=true`, `grants_execution=false`, `requires_target_support=true`  
   `evaluation/functional/task_episode_harness/agentic/source_skill.py:637-646`。
2. S2a 在 producer 双门 POSITIVE 后编译并只装进 A5:  
   `run_e2_s2a_forecast_curriculum.py:663-673` `_compile_forecast_card`（`pattern_family=None`）；`:1091-1114` 安装；`:1120` `produced_by=ladder_v2_compile_supply_tier`。
3. 检索后的 CAPABILITY 冻步骤打成 `cand_skill_<id>`:  
   `SelfEvolvingHarnessTS/methods/ttha/fast_agent.py:325-370`。
4. 供给档过滤: `risk_guards.authority.supplies_candidates is True`  
   `fast_agent.py:377-407`。
5. 与 Agent 提案合并后 `CandidatePool.build`:  
   `fast_agent.py:1072-1121`, `:1174-1180`；`runtime/candidate_pool.py:41-69`（先 identity，按序截断，同 SHA 去重）。

### 1.3 自主候选注入

- Fast propose: `fast_agent.py:1014-1059`。
- `_compile_candidates`，`source="agent"`；Agent 不得提交 identity: `fast_agent.py:410-438`。
- 上下文确定性 no-op 算子过滤: `fast_agent.py:1066-1071`。
- S2a `runtime_prior_slot=False`（`run_e2_s2a_forecast_curriculum.py:391`），`cand_prior_*` 双槽不启用。

### 1.4 池帽（不是 probe 预算，但决定谁进池）

- h0: `total_k=4`, `identity_slots=1`, `agent_program_slots=3`  
  `SelfEvolvingHarnessTS/methods/ttha/harness/h0/candidate_policy.json:3-5`。
- S2a: `SUPPORT_TRIAL_BUDGET=2`（`:88`）；`maximum_candidates=1+SUPPORT_TRIAL_BUDGET=3`（`:140`）。
- 有效帽: `min(4, 3)=3` = identity + **2** 个 PROGRAM（`fast_agent.py:1174-1180`）。

### 1.5 合并序（供给是否占强制位）

`fast_agent.py:1084-1121`:

- ACTIVE（非 `requires_target_support`、非 signed-risk degraded）: `(*active[:1], *agent[:1], *draft[:1], *degraded[:1])` — ACTIVE **保留**第一 PROGRAM 位。
- 仅 DRAFT: `(*agent[:1], *draft[:1], *degraded[:1])` — **DRAFT 不挤 Agent**。

S2a 供给卡 `requires_target_support=true` → DRAFT 支。单元 4 池为 `[identity, robust_mad, cand_skill_…]`，与「Agent 在前、供给在后、两者都进帽」一致。  
**没有供给强制 probe 位。**

### 1.6 Probe 序与停探

`online_loop.py:382-391` prepare 后:

```
pool = candidate_ids ∩ compiled steps
probe_order = [chosen]（chosen 非空且非 identity）+ 其余 pool
```

`:399-403`。`:415-421` 若有 Domain Ordering Card 则 `reorder_probe_order`（排列、不增删；`ordering_card.py:257-283`）。S2a 未传 `ordering_program_family`（默认 `None`），快照无 ordering card → `_select_ordering_card`（`:296-324`）返回 None。

预算循环 `:424-472`:

- `target_support_receipts_used >= budget` 则停（budget=2）。
- 合法 Support 写 Episode；`relation == POSITIVE` → 记 winner 并 **break**。
- 不比较已探增益；不探完再 argmax。

`POSITIVE` 定义在 `experience_memory.py:411-451`（聚合 ≥ +0.005 且逐序列 ≥ −0.005）。CONFLICT 不得当 winner（`online_loop.py:459-465` 注释 T5 #41 A4）。

### 1.7 Deploy

- Fast winner 若是 `cand_skill_*` → `deployed_existing_skill`（`online_loop.py:670-675`），不再 `handle_fast_winner`。
- 否则 `allow_fast_skill=True` 铸 `fast_winner_…`（`:676-701`）。单元 4 A5 走这条。
- delayed 批准才更新 incumbent: `run_e2_t6_cls_op_shared_harness.py:766-790`；S2a `:399-400`。
- held-out: `_frozen_recall`（`cls` `:1235-1295`）— 适用且带冻步骤的 ACTIVE；`requires_target_support` 且不在 `approved_skill_ids` 则跳过。供给卡因此不能靠召回部署。S2a `_deploy` `:535-570`。

### 1.8 单元 4 A5 实际代码路径

1. DRAFT 合并 → 池内同时有 mad 与供给 hampel（未截断）。
2. select `chosen=robust_mad_outlier_repair`（`fast_agent.py:1277-1297`）。
3. `probe_order` 以 mad 为首；首探 POSITIVE（support +1.0497）→ break。
4. 供给 `outcome=not_reached_support_budget_exhausted`。
5. 铸并召回 `fast_winner_…_outlier_mad`。held-out +3.7863，相对 A3 hampel −0.1206。

---

## 2. 现行 de facto 策略

S2a 每轮最多 **2** 次合法 Support probe；池为 identity + 至多 **2** 个 PROGRAM。DRAFT 供给卡排在保留的 1 个 Agent 提案之后，**不预留 probe 位**。探测序 = Fast select 的非 identity chosen 优先，其后为池序（本课无 ordering card）。**第一个 Support-POSITIVE 即 winner 并停探**，探针之间无增益比较、无并列裁决。held-out 部署是已批准 ACTIVE 的 Fast-only 召回，不是供给卡执行权。

| 项 | 现行值 |
| --- | --- |
| probe 总预算 | 2（`SUPPORT_TRIAL_BUDGET`） |
| 池 PROGRAM 位 | 2（`min(total_k=4, maximum_candidates=3)` − identity） |
| 供给强制 probe 位 | 0 |
| 供给池位（DRAFT） | Agent 之后若有空位则进；单元 4 两格都进了 |
| 排序 | `chosen_first_then_pool` |
| 比较 | `first_positive_in_probe_order` |
| 并列 | 空操作（遇首个 POSITIVE 即停） |

---

## 3. 提议参数化（只提案，不改码）

合法域约束: **不改 `SUPPORT_TRIAL_BUDGET` / `maximum_candidates`**（`:88`, `:140`）。Random-legal-edit 只在下表离散域抽一次。

| 名 | 现值 | 合法域 | 落点 |
| --- | --- | --- | --- |
| `skill_slot_merge_rule` | `draft_does_not_displace_agent` | `draft_does_not_displace_agent` / `supply_then_agent` / `agent_then_supply` / `interleave_one_each` | `fast_agent.py:1112-1121` |
| `supply_reserved_probe_slots` | 0 | `{0, 1}`，且 reserved+其余=2 | `online_loop.py:399-403` 与 `:424-426` |
| `probe_order_rule` | `chosen_first_then_pool` | `chosen_first_then_pool` / `pool_as_built` / `supply_first_then_agent` / `agent_first_then_supply` | `online_loop.py:399-403` |
| `first_positive_stop` | true | `{true, false}`（false 仍 ≤2 探） | `online_loop.py:459-472` |
| `winner_compare_rule` | `first_positive_in_probe_order` | 同上或 `max_support_gain_among_probed_positive` | `online_loop.py:459-472` |
| `tie_break_rule` | `probe_order` | `probe_order` / `prefer_self_proposed` / `prefer_supplied` | 同上（仅比较能打平时） |
| `agent_proposals_kept` | 1 | `{1, 2}`，且 Agent+供给 ≤2 个 PROGRAM | `fast_agent.py:1117-1121` 的 `[:1]` |
| `displacement_margin` | 0.0 | `{0.0, 0.01, 0.05}`，只比**已探 POSITIVE**；**不得**改 0.005 双门线 | `online_loop.py:459-472` |

`first_positive_stop=false` 是否算「不增减总预算」见 **Q6**。

---

## 4. G3 边界（不可编辑）与分离

| 门 | 落点 | 与上表分离 |
| --- | --- | --- |
| **双门** | `experience_memory.py:411-451`；`online_loop.py:239-242`, `:247-278`, `:459-465`；S2a `_supply_row` `:638-641`；`_incumbent_after_delayed` `cls:766-790` | 参数只动已分类 POSITIVE 的序/保留/比较，不改 `classify_relation` / LOCAL_ACTIVE |
| **容量门** | `run_e2_s2a_forecast_oracle.py:45-48`（TRAIN=40, half=20）；`run_e2_s2a_natural_pool.py:4`；电切 `:232-256` | 数据切格，分配参数不读 |
| **harm 阈** | `signed_radius.py:40`；`online_loop.py:61`, `:473-475`；S2a `HARM_BAR=0.005` `:93`, `:530` | `displacement_margin` 域与 0.005 分立，禁接入分类器 |
| **越权守卫** | `source_skill.py:639-646`；`fast_agent.py:422-423`, `:1181-1211`；S2a 菜单 `:124-130`、分数帽 0.35 `:140`、`allow_slow=False` `:388`；`candidate_pool.py:83-95` | 不授执行权、不扩菜单、不开 Slow |
| **隔离守卫** | G2 `_g2_faces` `:326-341`、停跑 `:1076-1083`；`s1` `ORACLE_TOKEN=s1_oracle` `:86-87`, `:108-168`；S2a `PHASE_ARM` `:577`、`_new_state` `:581-585`；`ORACLE_BANNER` `forecast_oracle.py:55` | 分配参数不读 G2/oracle。s2a_oracle 是否被 PHASE_ARM 挡住见 **Q3** |
| **阶梯 v2** | `source_skill.py:319`, `:362-424`, `:695-751`；S2a `_supply_row` / compile `:629-673` | 不改 `SUPPLY_TIER_MIN_DISTINCT_TASKS` 或产卡过滤器 |
| **Scope 匹配** | `retrieval.py:90-96`, `:278-282`；`s1._scope_match_by_skill_id` `:1208-1232`；`five_axis_scope` / `supply_applicability` `source_skill.py:477-576`；S2a `pattern_family=None` `:667` | 不改 AST / 求值器 |

**G3 分离结论:** 上表八个参数只落在 `fast_agent` 合并切片与 `online_loop` 探测序/停探/已探比较；与双门谓词、容量切格、harm 0.005、越权旗、G2/oracle、阶梯计价、Scope AST 无交叉写入。

---

## 5. LLM-edit 种子轨迹字段

文件: `artifacts/functional/e2/s2a_g1_run1_r2.json`（人读对照 `.md` Units 第 4 行与 P3/P6）。

- `protocol`, `seed`
- `course[3]`（`electricity_impulsive_outlier_04`）
- `version_chain[0]`（`s2a_forecast_supply_v0` / sha `b5058018…`）
- `rows` 中 `position==4` 且 `arm==A5-online`:
  - `candidate_sources.{supplied,self_proposed,supplied_ids,self_proposed_ids,dedup_detail}`
  - `scope_match.{classification,forecast}`
  - `rounds[0].{pool,chosen,retrieved_skill_ids,scope_match_by_skill_id,proposals[],probes[],winner_program,episodes[]}`
  - `deployment.{deploy_source,active_skill_id,applied_program,heldout_gain,harm_event}`
- 同 position 的 `A3-reset` / `K0-fixed` 的 `heldout_gain` 与 `applied_ops`（反事实分母）
- `predictions` 中 `id==P3` 与 `id==P6` 的 `observed`

---

## 6. 课程原料（只列不选）

入过 S2a 冻课（`s2a_course_frozen.json`）: `electricity_impulsive_outlier_{03,01,04}`, `traffic_clean_identity_00`, `traffic_gap_00`。

**电 0–299 五格**（`s2a_g0_electricity_sweep.json`; 切法 `run_e2_s2a_electricity_sweep.py:232-256`）:

- 全: `electricity_impulsive_outlier_00` … `_04`
- 已入课: `_01`, `_03`, `_04`
- **未入课可用: `_00`, `_02`**

**traffic recut 0–419 七格**（`run_e2_s2a_forecast_oracle.py:45-114`）:

- 全: `traffic_impulsive_outlier_00` … `_05`, `traffic_gap_00`
- 已入课: `traffic_gap_00`
- **未入课可用: `traffic_impulsive_outlier_00` … `_05`**

`traffic_clean_identity_00` 是 recut 之后 leftover 420–479（冻课 `clean_cell`），**不属于**这七格，但已入课。电 leftover 300–319+OT 未切成 cell。

---

## 7. 开放问题

- **Q1（挡种子叙事）:** 协议「供给被采」与 r2 采用记录相反。须主线改写种子后再让 LLM-edit 读轨迹。
- **Q2:** A5 提 mad、A3/K0 提 hampel 是否由供给在视野引起，工件未鉴定。
- **Q3:** PHASE_ARM 只挡 `/s1_oracle/`；`s2a_oracle/` 无同款机械墙。
- **Q4:** S2a 不调用 `s1._arm_isolation`。
- **Q5:** 编辑必须落 TTHA `online_loop`/`fast_agent`，不是 agentic `fast_path`。
- **Q6:** `first_positive_stop=false` 是否 Random-legal（帽不变、期望花费上升）。
- **Q7:** 未对 `methods/ttha/` 与 `SelfEvolvingHarnessTS/methods/ttha/` 做逐字节比对；S2a import 的是后者。
