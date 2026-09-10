# DEV-AUTO-1：自主 Skill 修订开发实验（2026-09-06 收口）

> **⚠️ 本报告的主结论已于同日被 DEV-AUTO-1R 推翻，勿单独引用。**
> 本文把候选与父版本在**各自可读的格**上求均值相减（u16 = 7 格对 15 格），
> 由此得出的「9 个提议超父版本、5 个风险不更差、接受层挡死」并不成立。
> 统一分母后是 **3 个超父、0 个风险不更差**；把绝对屏换成相对参照屏，
> 进入验证的数量仍是 0。更正与新证据见
> [`DEV_AUTO1R_COMPARATOR_AND_ACCEPTANCE_FRAMES_2026-09-06.md`](DEV_AUTO1R_COMPARATOR_AND_ACCEPTANCE_FRAMES_2026-09-06.md) §6。
> 本文其余部分（提议分布、反馈响应性、成本与边界记录）未被更正，原样保留。


**性质：开发级机制比较。** 全部单元已曝光；不冒充正式 A3/A5 主实验，不进入密封终验。
未读 +144 评价面、未读 5 个 `TARGET_HELD_IN` 单元、未读任何密封材料。
未改风险分母、风险阈值、冻结分箱边缘、算子注册表、准入合同或部署状态。

- 主工件 `artifacts/main_protocol/dev_auto1_skill_revision__run2.json`
- 被拒候选事后读数 `artifacts/main_protocol/dev_auto1_blocked_candidates__run2.json`
- 源文件 `evaluation/main_protocol_p4/{dev_auto1_revision.py, run_dev_auto1_skill_revision.py,
  smoke_dev_auto1_skill_revision.py, audit_dev_auto1_blocked_candidates.py}`
- 冒烟 `python -m evaluation.main_protocol_p4.smoke_dev_auto1_skill_revision`（0 fit、0 LLM、九项全过）

---

## 0. 一句话结论

**提议层通了，接受层挡死了。** 放开提议限制后，真实 LLM Slow 在 16 个课程步上自主提出了
25 个合法修改（10 个不同 Workflow、3 个不同 Scope 方向），其中 9 个在全服务人群口径上
超过当时可执行的父版本；但**生产 replay 屏是绝对门**——任一已处理单元越线即淘汰——而
持权卡本身在全部 16 步上都不满足这道门。因此 0 个修改获准、0 次 Skill 更新、两臂行为
无机制性差异。事后测量显示，被这道门拒掉的 `outlier_iqr > winsorize` 在后 16 个单元上
**两面收益都高于祖先、受损序列 51→39、权威门通过数 4/16→7/16**。

---

## 1. 本包实际做了什么

闭环按任务书 §1 实现并真实运行：

```text
读合法历史反馈 → LLM 自主提 Scope 或 Workflow 修改 → 确定性工具评价并返回具体后果
→ （若通过）经现有 Skill 编译/验证/加载接口更新已有 Skill → 后续新会话的 Fast 使用它
```

### 1.1 提议侧放开的五处（任务书 §3A）

| # | 改动 | 落地方式 |
| --- | --- | --- |
| 1 | 专门的"修订已有数据准备知识"角色说明 | `dev_auto1_revision.REVISION_ROLE_INSTRUCTION`，在调用缝替换 `harness_view.instruction`（沿用 M-R0f 手法，**不改快照、不铸 SHA**）。不再复用容易诱发弃权的内环 preparation 指令 |
| 2 | 允许 Workflow 修改或 Scope 修改 | 内联 stage schema `dev-auto1-revision-proposal/1`，两个 `oneOf` 分支。**不写入 `methods/ttha/schemas/`**——那里每个文件都是每份快照 lock 的依赖 SHA，新增会让 K0 的 runtime bundle 旋转 |
| 3 | 不要求先证明唯一失败原因 | 每个提议只需 `what_changes` + `expected_change` 两句话 |
| 4 | 取消每步两次调用的微小限制 | 预算退到包级：外环 backend `maximum_calls = 400`，无每候选帽 |
| 5 | 弃权合法但非终止；粗粒度归因不作否决 | 一次弃权或候选失败不结束课程；`FLAGGED` 由硬否决改为顺序提示（"先试程序差分"），三态机其余不动 |

**探索许可与部署许可分开**：通过屏的 Workflow 修改只开一张 `RestrictedDraft`（无部署权），
经 `resupplied_programs_for_verification` 供给后续单元作候选；只有当后续**独立**单元真的
部署了它且 delayed 权威门通过，才 PATCH Skill 卡。本包无任何候选走到这一步。

### 1.2 反馈侧（任务书 §3B）

送给 Slow 的是这张卡自己的完整记录，**逐单元不截断**：每个已处理单元的
获益序列名单 / 受损序列名单 / 未变化数、全服务人群收益、最坏单序列伤害、覆盖、
两个面（Support 与 delayed）、权威门结果与失败线；外加**每一次早先尝试的具体后果**
（工具测到多少、父版本给多少、差多少、越了哪条线、试过哪些冻结分箱边缘）。

实测：16 步的 `units_sent == units_available`（1→16），`nothing_was_truncated = true`；
携带的早先尝试数 0→23。对照生产路径只送 `rows[:60]`、单面、失败只回一行"方向不可行"
（`run_hec1.py:1649-1665`，M-R0e 已核）。

### 1.3 Skill 更新通道（任务书 §3C）

- **Workflow** → `EditController.apply_to_fork` PATCH `skill_library.entries/<id>.body`，
  cause 用**已注册的** `OUTCOME_GAP`（`fault_routes.json` 声明 `capability` × `PATCH`）。
  冒烟另断言 `RISK_GAP` 对同一编辑被拒——本包用的是存在的那条路，不是方便的那条。
  写入后用 Fast 自己的解析器 `verify_frozen_patch_program` 回读校验。**新增 route 0、surface 0。**
- **Scope** → 生产收窄路径：`calibrate`（冻结分箱）→ `validate_narrowing` →
  `DraftLedger.record_revision`，经 `resupplied_scopes` 供给后续单元。卡的
  `serving_scope` 没有声明编辑面，本包也没有新增一个。
- 后续单元一律新会话：每单元一次 `run_online_round`，不携带上一轮聊天记录；
  Runner 不强制供给上一轮赢家；`experience_episodes=()`，Source 原始 Episode 不进 Fast。
- 每格记录"加载了哪个 Skill、实际做了什么"：`retrieved_skill_ids`、
  `skill_bodies_at_start/at_end`、`candidate_program_steps`、`deployed_via`、
  `resupplied_programs_offered`。

### 1.4 两臂

| 臂 | 单元内 Target 本地探索 | 跨单元 Skill 状态 |
| --- | --- | --- |
| `frozen-skill` | 完整（同模型、同工具、同 5 次/格 LLM 帽、同探针预算） | 每单元从 k1 入口态重建，什么都不跨越单元边界 |
| `revising-skill` | 同上 | 携带；每单元后可提修订、被工具评价、通过则写回 Skill 并在后续使用 |

两臂从同一 M-R0d k1 重建态出发（各持一份 deep copy），同一反馈访问规则，同一搜索额度。

---

## 2. 报告先回答的五个问题

### Q1 系统实际上提出了哪些修改？

**16 次修订调用：14 次提出、2 次弃权（`insufficient_public_evidence`）、0 次传输故障。**
共 **25 个提议**：17 个 Workflow、8 个 Scope。

- **10 个不同 Workflow 程序**：`outlier_iqr>hampel_filter`(×3)、`hampel_filter`(×2)、
  `winsorize`(×2)、`outlier_iqr`(×2)、`outlier_iqr>winsorize`(×2)、
  `hampel_filter>winsorize`(×2)、`denoise_median`、`outlier_mad>winsorize`、
  `repair_level_shift>outlier_mad`、`outlier_mad>outlier_mad`。
- **3 个不同 Scope（特征, 方向）**：`level_excursion_score<=`(×4)、
  `local_robust_z_peak>=`(×3)、`period_reliability>=`。

**真实 LLM 提议与工具确定性校准的分界（严格）**：LLM 只给 `{feature, direction}` 与
`[{op, params}]`；**所有阈值一律由 `scope_threshold_tool.calibrate` 在冻结分箱边缘上确定**，
模型返回的 threshold 被丢弃并记为丢弃。是否接受完全由 `calibrate` /
`validate_narrowing` / replay 屏 / `EditController` 决定，LLM 不批准自己的补丁。

### Q2 失败有没有改变后续尝试？

**有，而且是主流。** 14 个有提议的步骤中，相邻两步提议集合 **12 次改变、1 次相同**
（u23 重复了 u17 的两个程序）。25 个提议里 16 个载荷互不相同、9 个是重复。
**0 次重复弃权**：两次弃权（u5、u7）之后的下一步都提出了新的具体修改。

连续案例（改变的）：

- u6 `hampel_filter` → 屏读数 g=0.017627 vs 父 0.203252、最坏伤害 2.163、受损比 0.7
  → u8 把它与 `winsorize` 并排再提一次，`winsorize` 拿到 0.19242（受损比 0.7→0.3）；
- u12 `outlier_iqr` 已超过父版本（0.1835 vs 0.173453）仍被 `harmed_fraction`+`single_series_harm` 拒
  → u16 改成 `outlier_iqr>winsorize`（0.268127 vs 父 0.16563，两条风险线均**不差于**父版本）；
- u9 `denoise_median` 得到 `NOT_READABLE`（窗口验证器在每个已处理单元都拒）→ u10 换成
  一个 Scope 子句加一个 `outlier_iqr`；`denoise_median` 在 run2 后续不再出现。

连续案例（没有改变的）：u17 与 u23 提交了同一对程序（`outlier_iqr>hampel_filter`、
`hampel_filter>winsorize`）。两步之间只多了一个新单元，父版本读数几乎不变（0.160951 两次相同），
反馈没有提供新的区分信息——这是"同一输入、同一失败原因"下的重复，记录在案。

Scope 侧则是另一种重复：8 个 Scope 提议**全部** `NO_FEASIBLE_THRESHOLD`。原因是仪器而非模型——
6 个分数型特征沿用默认分箱 `(0, 1, 3, 6)`，有效切点只剩 0.0（M-R0f §4 已测）。本包按授权
**没有动栅格**，因此 Scope 出口在这条谱系上仍然是关死的。

### Q3 Skill 是否更新、被加载并改变行为？

**没有更新。** `accepted_for_verification = 0`，`promotions_to_the_skill_card = 0`。
两臂全程各只有一个 Skill 卡状态：
`Frozen program steps: [{"op": "outlier_mad", "params": {}}]`。

因此"更新 → 加载 → 行为改变"这一段**本包没有被检验**，不能记为通过也不能记为失败。

被加载这一半是成立的且有读数：32 格中 **23 格** Fast 检索到了 K0 卡
（`fast_winner_forecast_ridge_smase_outlier_mad`），24 格收到了 Draft 的 resupply
（`resupplied_draft_1 → outlier_mad({})`）。

生命周期确实在动，只是动的是**限制**而不是修订：真实 delayed 失败触发了 3 次
`record_verification`——frozen u10（`coverage_floor`）、revising u10（同）、
revising u12（`harmed_fraction`）——修订臂的 Draft 因此在 u12 由 `REVISABLE` 转入
`FLAGGED`。按 §3A 第 5 条，`FLAGGED` 在本包只作顺序提示（"先试程序差分"），课程继续；
但它也意味着 u12 之后即使有可校准的 Scope 子句也无处落地。实际未被触发：8 个 Scope
提议全部倒在更上游的 `calibrate`。`activated` 全程为 0，可读的 delayed 权威门 3 次全不通过。

### Q4 相对冻结 Skill，对后续收益、风险、覆盖和成本有什么影响？

**没有可归因于机制的影响，因为没有任何修订被应用。** 两臂在后 16 个单元上的读数：

| 臂 | 面 | 全人群收益 | 受损序列 | 处理序列 | 最坏伤害 | 权威门 /16 |
| --- | --- | --- | --- | --- | --- | --- |
| `frozen-skill` | Support | 0.004286 | 0 | 8 | 0.0 | — |
| `frozen-skill` | delayed | 0.001252 | 1 | 3 | 0.034348 | 0 |
| `revising-skill` | Support | 0.028763 | 0 | 19 | 0.0 | — |
| `revising-skill` | delayed | 0.005633 | 8 | 15 | 0.157114 | 0 |

**这个差值是噪声，不是机制。** 决定性证据：本包跑了两次完整课程（`run1` 与 `run2`），
两次的两臂数字**完全对调**——run1 是 frozen 0.028763 / 0.005633、revising 0.004286 / 0.001252，run2 逐位反过来。数值完全由"哪一格恰好部署了 `outlier_mad`"决定（16 格里只有 2 格），
而两臂在无修订时是同一个系统。

**成本**（run2）：

| | LLM 调用 | 物理 Consumer fits |
| --- | --- | --- |
| `frozen-skill` | 79 | 49 |
| `revising-skill` | 91 | 244 |

修订臂多花的 195 个 fits 全部是 replay 屏——为 25 个提议在已处理单元上取真实读数的代价。
不同 Program 数：frozen 探到 2 个、revising 探到 3 个（Fast 侧）；外环另提了 10 个不同程序。

### Q5 目前主要卡在哪一层？

**卡在"选择/接受"层，不是提议层、不是可执行修改层、不是证据层。** 分层漏斗（run2）：

```text
调用 16 → 提出 25 → 可解析且合法 25 → 有读数 23（2 个 NOT_READABLE：窗口验证器全拒）
→ Scope 侧可校准 0 / 8（冻结栅格）
→ 通过生产 replay 屏 0 / 15
→ 获准入 Draft 0 → 后续独立单元验证 0 → PATCH Skill 卡 0
```

被拒的机制是明确的：`outer_loop.screen` 是**绝对门**——任一"适用"的已处理单元越线即淘汰，
不跨单元平均。而**持权卡本身不需要通过它**：K0 卡的部署权来自某个单元的 delayed 权威门，
不是来自全段屏。实测：父版本在**全部 16 步**都不满足这道门
（`harmed_fraction` + `single_series_harm`，u6 起还加 `aggregate_not_material`）。

于是出现结构性不对称：**9 个提议在全服务人群口径上超过父版本 ≥ material，其中 5 个
在两条风险线上都不差于父版本，仍被这道父版本自己过不了的门淘汰。**

第二层卡点（与修订机制无关，但决定了本包的下游读数）：**Fast 极少部署任何东西。**
32 格中 20 格显式选 identity、9 格耗尽 5 次/格 LLM 预算后 abstain，只有 3 格真的部署了程序。
被探测的程序 Support 面读数最高到 +0.407011（frozen 臂 u15，`outlier_mad({})`），仍然选 identity。因此后 16 单元上两臂都远低于
"祖先策略每格都部署"的参照（delayed +0.170298）。这是 `PROPOSAL_DIVERSITY_GAP`
与选择层的老问题，本包只是又量了一次。

---

## 3. 被屏拒掉的东西值多少：事后测量

**这是本包最有信息量的一张表，也是唯一的正向线索。** 对通过"超父版本 ≥ material
且两条风险线不差于父版本"的两个不同程序，在后 16 个单元、两个面、全服务人群上
补齐真实读数（0 LLM、106 fits，与祖先共用 M-R0k 预测缓存）：

| 程序 | 面 | 全人群收益 | vs 祖先 | 受损序列 | 最坏伤害 | 权威门 /16 |
| --- | --- | --- | --- | --- | --- | --- |
| 祖先 `outlier_mad({})` | Support | 0.141389 | — | 69 | 1.180821 | 5 |
| 祖先 `outlier_mad({})` | **delayed** | 0.170298 | — | 51 | 1.115755 | 4 |
| `outlier_iqr>hampel_filter` | Support | 0.125141 | **−0.016248** | 58 | 0.845469 | 4 |
| `outlier_iqr>hampel_filter` | **delayed** | 0.139215 | **−0.031083** | 58 | 1.982855 | 2 |
| `outlier_iqr>winsorize` | Support | 0.170161 | **+0.028772** | 41 | 0.760007 | 6 |
| `outlier_iqr>winsorize` | **delayed** | 0.174513 | **+0.004215** | **39** | 1.142351 | **7** |

`outlier_iqr > winsorize`（Agent 在 u16 与 u24 自主提出，两次都被屏拒）在权威面上
**收益略高、受损序列 51→39、权威门通过数 4/16→7/16**。作为对照，M-R0k 中研究者事后
指定的三个 Workflow 候选没有一个在权威面上超过祖先（最好的 W2 是 +0.148 vs +0.170）。

**必须同时说明的限度：**

1. 这是**事后 L2 读数**（计划的 oracle 第二层），不是"当时应该放行"的判词。程序由
   Agent 只读边界前与已处理证据提出，无未来信息；但**用后 16 个单元给它打分是事后的**。
2. 两个候选里有一个（`outlier_iqr>hampel_filter`）在权威面上**差 −0.031**。放宽屏会同时
   放进这一个。本包没有、也不应据此改屏。
3. 即使修订被接受并写进卡，本包也**没有证明**下游会变好：Fast 在 32 格里只部署过 3 次，
   一个更好的卡未必会被选中和部署。
4. `outlier_iqr>winsorize` 的最坏单序列伤害 1.142 仍高于祖先 1.116，且远高于
   `max_single_series_harm = 0.30`。它在**绝对**风险口径下依然不可部署；它只是在
   **与父版本相对**的口径下不更差。

---

## 4. 成本与边界

| 条目 | 值 |
| --- | --- |
| 物理 Consumer fits（全包，含冒烟、中止尝试、重试与失败调用内花掉的） | **828 / 1000** |
| LLM 调用（全包） | 440 |
| 墙钟（全包实跑合计） | ≈ 17 分钟 / 上限 6 小时 |
| 单次课程 | run1 255 fits / 174 LLM / 411 s；run2 293 fits / 170 LLM / 497 s |
| 被拒候选事后测量 | 106 fits / 0 LLM |
| 中止的第一次尝试（schema 修复） | 90 fits / 62 LLM |

`skills_activated=0`、`deployment_rights_issued=0`、`lifecycle_counters_moved` 仅由
生产 `record_verification` 在真实 delayed 失败时移动、`thresholds_changed=0`、
`operators_added=0`、`routes_added=0`、`surfaces_added=0`、`schema_files_added=0`、
`evaluation_face_reads=0`、`sealed_reads=0`、`target_held_in_reads=0`。
**本包只新增 4 个 Python 文件**（`dev_auto1_revision.py`、`run_dev_auto1_skill_revision.py`、
`smoke_dev_auto1_skill_revision.py`、`audit_dev_auto1_blocked_candidates.py`），
未写入任何既有源文件。核验方式：`find ... -newermt "2026-09-06 17:00"` 只列出这四个。
注意 `run_hec1.py` / `outer_loop.py` / `restricted_draft.py` 在 `git status` 里显示为 modified，
但那是**本包开始前就已存在的未提交改动**（mtime 2026-09-05），与本包无关。
共享的 M-R0k 预测缓存 `_scratch/m_r0k_prediction_store.json` 新增了两个 `CAND_*` 程序的条目
（加法式，按 program 键区分，旧条目逐位未动）。未提交 git。

### 4.1 仪器记录（必须随读数一起读）

1. **模型不是 HEC-1 的模型。** agicto 中继（`gpt-5.6-luna`）现在对超过约 4.4k token 的请求
   一律返回 `HTTP 403 insufficient_quota`——极小探针仍成功，真实提示一律失败（`smoke1`
   工件即该故障的记录）。本包改用本机**已配置**的 DeepSeek 端点
   （`https://api.deepseek.com/v1`, `deepseek-chat`），未新增账户或付费服务。
   两臂与修订提议者用同一个模型，**臂间对比不受影响**；但绝对数字与 HEC-1 不可并表。
2. **第一次发车被我主动中止并重跑。** 初版 proposal schema 允许"只报 kind、不给载荷"，
   于是两个提议被记为 `MALFORMED`——那是在量我的 schema，不是在量提议者。改成
   `oneOf`（每个分支强制携带自己的载荷）后重跑。中止那次的成本已计入总账
   （`_scratch/dev_auto1/aborted_attempt1_schema_fix_progress.json`）。
3. **run1 与 run2 的屏规则不同。** run1 的屏把"覆盖低于底线"的单元也当作违规，
   与生产 `outer_loop._applicable`（这类单元记 NOT_APPLICABLE）不一致。run2 改为
   逐字复用 `outer_loop._applicable` / `_violations`。**run2 是本包的读数**；
   run1 保留为先前的完整运行记录，两者数字不并表。
4. **`relative_to_the_parent_screen` 是独立分析列。** 它不改任何规则、不授予任何权，
   只是把"候选被一道父版本自己过不了的门淘汰"这件事写进回执，否则读者看不到。
5. 两臂在每个单元内共享该单元的 executor，顺序固定为 frozen → revising；
   参照模型的缓存命中按 `_bill_support_fits` 单列，不折进任一臂的 `support_fits`。

---

## 5. 不能从本包推出的结论

- **不能**说"自主修订机制无效"：它从未被完整走通一次（0 次接受 ⇒ 0 次写入 ⇒ 0 次后续使用）。
- **不能**说"LLM 提议不如确定性提议"：本包没有跑确定性提议者对照臂（任务书只要两臂）。
- **不能**说"应该放宽 replay 屏"：两个被拒候选里有一个在权威面显著更差。
- **不能**说"冻结 Skill 更好或更差"：两次运行数字对调，差值是 Fast 选择的运行间方差。
- **不能**把 `outlier_iqr>winsorize` 的读数当成性能主张：开发级、事后、单谱系、单课程、
  单数据身份，且它在绝对风险口径下仍不可部署。

---

## 6. 下一项最有信息量的实验（交 Planner 裁定，本包不自行推进）

按信息量排序，三选一：

1. **接受口径的一次对照（唯一操纵变量 = 屏的参照系）。** 臂 A 用现行绝对屏；
   臂 B 用"相对父版本"屏（收益 ≥ 父 + material，且两条风险线不差于父）。
   终点 = 后续单元权威面的效用/伤害/覆盖/门通过数，以及**被放进来的坏候选**的代价。
   本包已把两边所需的读数都留在工件里，`run2` 的 9 个"超父"候选是现成的输入。
   这是当前唯一能把"卡在接受层"变成可判定命题的实验。
2. **修好分数型特征的冻结栅格（M-R0f #3）。** Scope 出口现在是 8/8 全灭，
   且原因确定是栅格而非模型。修好之前，"Scope vs Workflow 谁值得改"这个问题无法作答。
3. **Fast 的选择层。** 20/32 格显式选 identity、9/32 格耗尽 5 次调用预算，
   探到 +0.39 仍不部署。修订机制即使打通，下游也接不住。

---

## 7. 判词

```text
DEV_AUTO1_PROPOSAL_REACHED__ACCEPTANCE_BLOCKED
```

自主提议、具体后果反馈、反馈驱动的下一次修改：**成立且有连续案例**。
Skill 更新、加载、改变后续行为：**本包未发生，未被检验**。
两臂效用差：**无机制归因，属运行间方差**。
最上游的可动缺口：**接受层的参照系**（绝对屏 vs 相对父版本），其代价已被测量。
