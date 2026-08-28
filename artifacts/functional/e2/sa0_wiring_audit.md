# SA-0 接线审计：Skill 修订循环的现役接线（Part A）

`sa0_wiring_audit_v1` · 2026-08-28 · 只读代码审计 · 0 LLM / 0 fit / 0 下载 / 0 git 操作

审计基准 rev **9894a5d**。脏文件表见 JSON `audit_basis`；其中唯一承重的工作树改动是
`evaluation/functional/task_episode_harness/agentic/source_skill.py:319`
（`SUPPLY_TIER_MIN_DISTINCT_TASKS = 1`，阶梯 v2 已落工作树未提交），TRY 档的 LOO 下限
`source_skill.py:250-258` 未动。下文一切涉及供给档的判定均以审计时工作树字节为准。

---

## 四项判定摘要

| 项 | 判定 | 最关键 file:line |
|---|---|---|
| A-1 归因面完整性 | **部分足够**：卡→Support 可机械归因，卡→delayed/部署不可；缺 4 个纯增字段 | `run_e2_s1_curriculum_four_arms.py:1398-1414`（Episode 无 `source_skill_id`，只能按 `workflow_signature` 回接） |
| A-2 修订面 | **`revision` 是静态著作字段，无任何代码自增**；Scope 收窄的 PATCH 面已授权但无写者；回滚只在快照粒度存在 | `contracts/harness.py:111` + `harness_surfaces.json:65-77` |
| A-3 混合反馈现状 | **什么都不做**——正向不累计、Support 拒不可见、delayed 拒只经产线不调用的路径 | `e1.py:1362-1382`（转化只新铸 Target-local 卡，源卡零写回） |
| A-4 撤权钝度 | 两条撤权机制**按构造最钝**（全卡、全域），但对供给卡**均未接通**；当前真实风险是"完全无响应" | `runner.py:947-954` + `e1.py:689-690` |

---

## A-1 归因面：记了什么，够不够机械归因

**已记（足以支撑 Support 侧归因）**

- `round.pool` / `round.proposals[].candidate_id` 形如 `cand_skill_<skill_id>`
 （记录装配 `run_e2_s1_curriculum_four_arms.py:1310-1320`；候选铸造 `methods/ttha/fast_agent.py:365-369`，
 `Candidate.source = "skill:<skill_id>"`）。**供给卡 id 可由字符串解码得到**，但只存在于
 `candidate_id` 里，没有独立字段；`Candidate.source` 携带的 `skill:<id>` 未被记录下来。
- `round.proposals[].outcome / .gain / .verifier_passed / .chosen_by_select / .compiled`
 （`:1302-1320`）——逐候选 Support 侧结局：探过 / verifier 拒 / 无编译步骤被丢 / 预算未及 / 被 select 选中。
- `round.retrieved_skill_ids`（`:1337-1338`；agentic 侧 `agentic/runner.py:1021-1036`）——
 该单元跑时哪些卡在视野，这是"受引导"事后可审的依据。
- `round.fast_features_binned`（`:1362`，编译器 `:401-424`）——**该单元的冻结、分箱、无 oracle
 的 pattern view**。这正是 Scope 排除条件可机械编译的取值来源。
- 受引导标记：`run_e2_s1v2_forward_course.py:1616-1617`（`position > card_installed_after`）→
 `source_skill.py:366`（受引导正例单独入桶、计零）。
- 部署与 held-out：`run_e2_s1v2_forward_course.py:1648-1670`。

**缺口（这是"机械归因到具体卡"卡住的地方）**

1. **`episodes[].source_skill_id` 不存在**（`run_e2_s1_curriculum_four_arms.py:1398-1414`）。
 卡供给的 `hampel_filter` Episode 与 agent 自提的同族 Episode 是同一条记录。v4 正是这个碰撞：
 同单元 K0-fixed 自提 hampel 拿 +0.2690、A5 没提——签名级 join 分不开"卡的功劳"与"采样运气"。
2. **无卡版本戳**（`source_skill_revision` 或内容 sha）。修订后同一 `skill_id` 有两个版本，
 账本若说不出"当时在视野的是哪一版"，就撑不起逐版本 Gain/Harm 记账。
3. **无逐单元 `scope_match`**。"卡没匹上"与"卡匹上了被拒"今天都表现为"池里没有该候选"——
 而这两者的区别正是"Scope 太窄"与"卡是错的"的全部区别。当前只能离线重算 AST，
 L1 的离线门就是这么做的（`l1_ladder_v2_replay_r1.json → t1.scope_table`）。
4. **无逐卡 `guidance_conditioned`**。受引导性由边界处的单元位次派生，对"一张卡装一次"正确，
 对"两张卡在不同边界装入"未定义。

四项都是**纯增字段、零行为改动**。

---

## A-2 修订面

### `SkillEntry.revision` 的语义

声明 `contracts/harness.py:107-115`（`revision: int` 在 `:111`），校验
`contracts/harness.py:81-84` 与 `contracts/schemas/skill_entry_v1.json:12`。

**判定：`revision` 是静态著作字段，不是运行期版本计数器。**

- 每一处铸卡都写字面量 1：`methods/ttha/method.py:658`、`:896`、
 `methods/ttha/ordering_card.py:170/190`、`source_skill.py:484`（`revision: int = 1`）与 `:532`。
- 全仓**没有任何代码自增**既有条目的 `revision`；非 1 的取值只出现在手写的 bootstrap
 procedure（`methods/ttha/harness/h0/skills/bootstrap/build_contrastive_candidates.json:5` = 7；
 `inspect_and_localize.json:5` = 2）。
- 后果的实物证据：已被撤权的卡仍读 `revision:1`——
 `.r2_indep_micro_state/A5/snapshots/<sha>/skills/learned/fast_winner_e1v2_outlier_mad.json`
 同时带着 `restricted_by_target_feedback:true` 与 `"revision":1`。

### `restricted_by_target_feedback` 由谁写、何时触发、作用多大

- **写者**：`evaluation/functional/task_episode_harness/agentic/runner.py:636-687`（`_restrict_skill`），
 对 `skill_library.entries/<id>.risk_guards` 做 **PATCH**，SHA 前置（`:653-666`），
 置 `restricted_by_target_feedback=true` 与 `restriction_reason="delayed_window_disconfirmed_this_skill"`。
- **触发**：`runner.py:947-958`——`delayed_event.stage == "existing_skill_revalidated"`
 且 winner 的 `local_status == "RESTRICTED"`。该 stage 只由**复用路径**产出
 （`e1.py:1334-1343`），而复用路径的第一道过滤是 `_is_local_skill_id`
 （`e1.py:1305 → :689-690`，谓词在 `e1.py:133`），即"本臂自铸的 `fast_winner_*`"。
 状态来源：`methods/ttha/online_loop.py:218-264`——POSITIVE→LOCAL_ACTIVE、
 **CONFLICT→RESTRICTED**、NEGATIVE/NEUTRAL/ABSTAIN→EPISODE_ONLY（`:244-249`）。
- **粒度**：**全卡、全上下文**，无分域/分 Scope 变体。
- **读者**：`methods/ttha/retrieval.py:263-270`——在求值 applicability **之前**跳过该条目，
 **fast 与 slow 两个 role 都跳**；`e1.py:549-557` 在第二条通道上独立执行同一件事。
- **可审计性**：条目连同 guard 与 reason 留在快照里（`runner.py:639-643`），主张与反驳都可读。

### 第二条、更硬的撤权路径

`methods/ttha/online_loop.py:675-736`（`revoke_deployed_skill`）根本不是一次 edit：它 fork
物化树并 **`target.unlink()` 删掉卡的 JSON**（`:711-723`，删除在 `:723`），重编译并改 active。
触发：`online_loop.py:814-817`——winner 来自 `cand_skill_*`（路由在 `:640-645`）
且该轮 `delayed_utility < -M`，`M = 0.005`（`online_loop.py:61`、`signed_radius.py:40`）。
**但在产线不可达**：`online_loop.open_delayed` 从未被
`evaluation/functional/task_episode_harness/**` 调用；本 harness 内唯一调用者是
`t1.py:467-489` 的受控检查。

### `observable_applicability` 能否被 PATCH 收窄

**能——面、AST 词汇、求值器三样今天都齐，只是没有写者。**

- **面已授权**：`methods/ttha/harness/harness_surfaces.json:65-77` —— 模板
 `skill_library.entries/{skill_id}.observable_applicability`，`allowed_operations:["PATCH"]`，
 `precondition:"SHA"`，`target_class:"applicability"`，`derived_outputs:["retrieval_index"]`。
- **AST 支持否定与布尔组合**：校验器 `contracts/observables.py:167-193` 接受
 `all/any/not/const/leaf`；求值器 `methods/ttha/retrieval.py:48-87` 实现之（`not` 在 `:66-68`）。
- **三值逻辑与退化情形**：未知特征返回 `None`（`:70-71`）；`not(None)` 仍是 `None`（`:66-68`）；
 含 `None` 的 `all` 返回 `None`（`:59-62`）；`evaluate_applicability` 只在状态恰为 `True` 时报匹配
 （`:95-96`）。**所以：若被排除单元的叶子在后续单元的特征里缺失，整个 AST 弃权，卡被withheld**
 ——保守方向（宁可少供不多供），但会在本不该排除的单元上静默少供。
- **今天没有写者**：全仓没有任何 EditManifest 指向 `.observable_applicability`；
 每一处 PATCH 都打在 `.risk_guards`（`runner.py:653`、`e1.py:1152`）。
- **授权口子**：唯一授权 `target_class:"applicability"` 的故障因是 **`RETRIEVAL_MISS`**
 （`evaluation/minipipe/feedback/fault_routes.json:14`，执行在 `router.py:61-64`）。
 `RETRIEVAL_MISS` 是为"该检索到却没检索到"（**扩**的方向）命名的；
 **没有任何一个因码的语义是"这个 Scope 伸得太远"**。这是 Q1。

### 版本历史与回滚

- **逐卡版本：无**（见上）。
- **逐快照版本：有，且内容寻址**——每次 materialize 写入不可变树
 `<store>/<runtime_bundle_sha>`，并写血缘记录
 `harness_snapshot_provenance/<sha>/<parent_sha>.json`（`store.py:124-148`、`:150-170`）；
 同 sha 不同字节直接拒绝（`:144-145`、`:160-161`）。
- **回滚：机械上可行**——`store.set_active(旧 sha)` 对任何仍物化的 bundle 成立
 （`store.py:195-197`）；但没有任何方法层 API 把它暴露成"版本回滚"。
- **另有一条"向前恢复"**：`e1.py:1139-1183`（`_lift_restriction`）在同一 `risk_guards` 面上
 PATCH 掉 guard 与 reason 并置 `restored_after_restriction=true`（`:1147-1151`），
 条件是重探一张被限的本地卡（`:1121-1136`）。这是**新的前向 edit，不是回滚到旧版本**。

### Episode 是否不可变

**值不可变，账本槽位是就地覆盖。**

- 值：`methods/ttha/experience_memory.py:113-114` —— `@dataclass(frozen=True) class ExperienceEpisode`。
- 更新方式：`online_loop.py:250-264` 用 `dataclasses.replace` 造新对象
 （`delayed_response` / `relation` / `evidence_level` / `local_status` / `context_summary`）。
- **覆盖点**：`agentic/runner.py:924-927` 按 `episode_id` 找到 `arm_state.episodes` 里的槽位并**替换**，
 于是该 Episode 的"仅 Support"版本不再留存于账本。
- 对设计的意义："Episode 永不改"对冻结记录成立，对账本槽位不成立；要求 append-only 证据链的
 修订机制**不能依赖当前的内存账本**来提供它。

---

## A-3 混合反馈现状：卡在 X/Y 转化、在 Z 被拒，系统对卡做什么

**判定：在产线真正能走到的每一个分支里，对卡都什么都不做。**

| 分支 | 对卡的写回 | 实际发生的 |
|---|---|---|
| **Support 拒**（探了不过材料线 / verifier 拒 / 预算未及） | 无 | 一条 `proposals` 行（`:1302-1320`），若探过则一条 EPISODE_ONLY 的 Episode。卡 Scope 不变，**在此后每个匹配单元继续被供给，每次占一个池位加一个 probe** |
| **delayed 判 CONFLICT**（聚合正、逐序列害） | **无（对供给卡）** | Episode 变 RESTRICTED（`online_loop.py:246-247`）；但撤权 PATCH 要求 `stage==existing_skill_revalidated`（`runner.py:949-954`），只有本地卡复用路径会发（`e1.py:1305 → :689-690/133`）。供给卡 id 不是 `fast_winner_*`，**复用路径对它永不触发**。RESTRICTED 只落在 Episode 与（最多）Target-local 派生卡上，**从不落到源卡** |
| **delayed < −M** | 原理上整卡删除（`online_loop.py:814-817 → :675-736`） | **产线不可达**——`open_delayed` 不被本 harness 调用。**毫无自动响应** |
| **在 X 上转化** | 无 | 通过 `handle_fast_winner` **新铸一张 Target-local `fast_winner_*` 卡**（`e1.py:1362-1382`）并戳上单元名（`:1283-1284`）。源卡的 evidence 块在编译时冻结（`source_skill.py:554-569`），**永不追加** |

一句话：**今天的卡编译一次即冻结——正向不在它身上累计，Support 拒对它不可见，
delayed 拒只能经一条本 harness 不调用的路径触及它。"Skill 是可更新假设"在任何方向上都没有写入路径。**

---

## A-4 撤权钝度：一处被拒是否废掉他处正向

**判定：两条机制按构造都是最钝的（全卡、全域），但对供给卡都没接通；当前真实风险是反向的——完全无响应。**

**机制一 `restricted_by_target_feedback`**
- 精确条件：某一个单元的 delayed 窗口把被复用的本地卡的 Episode 判到 CONFLICT 或更差
 （`runner.py:947-954`；分级 `online_loop.py:244-249`）。
- 作用范围：**全局**——`retrieval.py:269-270` 在求值 applicability 之前就跳过，fast/slow 两 role 都跳；
 `e1.py:549-557` 在第二通道上同样执行。
- 钝度：**一个单元的拒绝在所有地方移除该卡，包括它已经转化成功的每一个上下文。
 全仓没有任何分域/分 Scope 的该 guard 变体。**
- 现有缓解：条目与 `restriction_reason` 留在快照（可审）；重探可向前解除（`e1.py:1139-1183`）。
- 今天实际能打到谁：只有本臂 `fast_winner_*` 本地卡（`_is_local_skill_id` 过滤）。
 而那些卡本已被域戳限定在一个单元上——**所以钝度目前是潜在的、尚未表达出来的**。

**机制二 `revoke_deployed_skill`**
- 精确条件：winner 是 `cand_skill_*` 且该轮 `delayed_utility < −0.005`（`online_loop.py:814-817`）。
- 作用范围：**全局且破坏性**——unlink 卡的 JSON（`:723`），重编译后的快照成为 active。
- 钝度：严格比 guard 更差：**一个单元上一次低于材料线的 delayed 读数就把卡从 active 树上删掉**；
 旧快照保留字节所以历史尚存，但卡的 Scope、evidence 块与任何累计账本从活视图消失。
- 今天实际能打到谁：**没人**——本 harness 不调用。

**最要紧的一条**：台账 2026-08-28 02:3x 条把单例供给卡的事后兜底记为
"Target 反证即 `restricted_by_target_feedback` 收回检索（retrieval.py:143/269 现役）"。
**读端确在产线**；但**写端的触发被 `_is_local_skill_id` 门在本地卡复用路径上**，
对供给卡不可能触发，而替代写者不被本 harness 调用。
所以就 n=1 卡而言，这张网**按当前接线是未上膛的**。这不改变 L1 的判词逻辑
（永不被撤的卡不是被错撤的卡），但它意味着：低价目前由
**verifier + 当前 Target 的 Support/delayed 双门 + harm 否决**背书，
**不由任何事后收回背书**。这是 Q4。

---

## 跨项发现（三条，均入 JSON）

1. **供给卡 Scope 带重复的 `task_kind` 叶**：`source_skill.py:467-474` 先塞 `task_kind`，
 再遍历 `pattern_intersection`（其中已含 `task_kind`）。L1 编出的卡因此是 **18 叶 / 17 个不同特征**。
 匹配上是同义反复无害，但它虚增每一处 `scope_leaves` 读数，并在
 capability 排序用的 applicability 分数里**双计**（`retrieval.py:278-284`、`:290-299`）。→ Q11
2. **四个 pattern 叶因 edit schema 无契约而被静默丢弃**（`source_skill.py:459-475`，
 合法叶集来自 `:434-456`）：L1 卡记录
 `[level_only_post_shift_support_sufficient, level_region_end_fraction, level_region_fraction, outlier_region_end_fraction]`。
 **有效 Scope 比记录的 `scope_v1` 恰好在这四轴上更宽**；同 schema 编译的排除规则会继承同一盲区，
 必须如实声明而不是宣称交集精确。→ Q7
3. **n=1 的交集 Scope 太窄，窄到没有东西可修订**：`l1_ladder_v2_replay_r1.json → t1.scope_table`
 显示尾段 5 单元只 **1 个**机械匹配（预注册预测 4 个）。按 v4 冻结 pattern view 逐叶重算：
 GunPoint 只差 `period_change_score`（high vs zero）、PowerCons 只差 `period_change_score`
 （high vs very_low）、Herring 差 2/17、BirdChicken 差 3/17。
 **一个偶然携带的叶子决定了三次未匹配中的两次。** → Q9，并直接约束 SA-1 的课程设计。

---

## 开放问题

共 **12 条**（Q1–Q12），完整表述见 `sa0_wiring_audit.json` 的 `open_questions_for_sol`，
设计稿 §5 亦逐条复列并标注它阻塞哪一步。

## 义务

三个新文件 + 一个条件件，零覆写；`methods/` `runtime/` `contracts/` `operators/` `evaluation/` 零改；
STAGE_REPORT / AGENTS.md / README / PROJECT_STATE* / SUCCESSOR_BRIEF* / ROADMAP 未碰；
密封件（`s1_oracle/`、Epilepsy2、`data/ucr_conf_downloaded/`）零读；
0 LLM / 0 fit / 0 下载 / 0 git 操作 / 0 子代理；只读脚本仅解析工件，未写 store。
