# DEV-AUTO-2 — 修复接线下的 Frame C 自主修订课程

*2026-09-07 · development 级 · 两臂 × 16 个单元完整跑完 · 判词 `CHAIN_CLOSED_TWICE__MOSTLY_VIA_SUPPLY`*

工件 `artifacts/main_protocol/dev_auto2_frame_c__run1.json`（课程）、
`dev_auto2_attribution.json`（归因）、`dev_auto2_frame_c__dryrun.json`（2 单元试跑）
源 `evaluation/main_protocol_p4/{run_dev_auto2_frame_c,smoke_dev_auto2_frame_c,audit_dev_auto2_attribution}.py`
预检 10 项全过（0 LLM / 0 fits）。模型 `deepseek-chat`（agicto 配额 403，沿用 DEV-AUTO-1 的替代通道）。

---

## 0. 一句话结论

**闭环合上了，两次。** 真实 LLM 在 u7 提出的两个程序，经后续独立单元的
Support 与 delayed 权威门验证后，把 Skill 卡的冻结程序体真的改写了两次
（编译、落盘、回读校验、Draft 按生产语义关闭）。**这两个程序都是历史绝对屏
会否决的**——事实上被排队的 15 个候选，绝对屏会一个不剩地全部否决。

但**收益的主要来源不是 Skill 更新，而是候选供给**。两臂在 16 个单元中
**12 个部署完全相同**，全部差异集中在 4 格：供给通道贡献 +0.682973，
更新后的 Skill 卡贡献 +0.141939，新卡下的一次 Fast 搜索贡献 −0.027335。
本包同时打开了这两条通道，因此**不能**把整体差异记在"Skill 会自我更新"账上。

---

## 1. 提出了什么

16 次修订调用：14 次 PROPOSED、1 次 ABSTAINED（u9）、1 次 PROPOSER_FAULT
（u10，模型返回的 `proposals[1]` 少了 `scope_clause` 字段，schema 拒收；
课程按设计继续，没有换方法）。

**27 条提议**：16 条 Workflow、11 条 Scope 子句；**6 个不同程序**。

| 程序 | 次数 |
|---|---|
| `outlier_mad>repair_level_shift` | 4 |
| `repair_level_shift>outlier_mad` | 4 |
| `outlier_mad>winsorize` | 3 |
| `repair_level_shift>winsorize` | 2 |
| `outlier_mad>hampel_filter` | 1 |
| `outlier_mad`（＝父版本） | 1 → `NO_CHANGE` |

**Scope 出口仍然全关**：11 条 Scope 子句全部 `NO_FEASIBLE_THRESHOLD`。
这与 DEV-AUTO-1R §6 的更正一致——失败在风险线，不在冻结栅格；本包未动栅格。

阈值一律由 `scope_threshold_tool.calibrate` 在冻结分箱上定，模型给的数被丢弃。
**没有向模型提供任何已知赢家**：角色指令、反馈、算子表都不含 DEV-AUTO-1R 的
结论、程序名或分数（预检 `no_known_winner_reaches_the_model` 逐条核对）。

---

## 2. 哪些被实际探测、验证、晋升

```
27 条提议
 └─ 11 条 Scope           → NO_FEASIBLE_THRESHOLD（校准，三框架共有的门）
 └─  1 条 Workflow        → NO_CHANGE（与父版本同程序）
 └─ 15 条 Workflow        → QUEUED_FOR_VERIFICATION
      └─  5 个开出 Draft（其余 10 条同身份已有 Draft，按生命周期拒绝重开，
          不清零计数器）
           └─ 2 个被 Fast 经 resupply 选中并部署（u11、u15）
                └─ 2 个通过独立后续单元的 delayed 权威门
                     └─ 2 次晋升，Skill 卡两次被 PATCH
           └─ 3 个 NEVER_VERIFIED（Fast 从未部署它们）
```

**关键对照：15 条排队候选，绝对屏会全部否决**——10 条 `NOT_READABLE`、
5 条 `RISK_LINE_FAILED`。两个最终晋升的程序都在 `RISK_LINE_FAILED` 之列。
换句话说，在框架 A 或 B 下本轮的晋升数会是 **0**，与 DEV-AUTO-1R 的离线复算一致。

历史 replay 与前向验证会双向不一致：`outlier_mad>hampel_filter` 在 u7 的
同分母比较里比父版本**差** −0.00822，它照样在 u15 通过了权威门。

Draft 生命周期（修订臂）：

| Draft | 程序 | 状态 | 验证次数 | 关闭原因 |
|---|---|---|---|---|
| `..._1` | `outlier_mad`（入场态带入） | REVISABLE | 3 | REVISION_BUDGET_EXHAUSTED |
| `..._1` | `outlier_mad>repair_level_shift` | — | 0 | NEVER_VERIFIED |
| `..._2` | `outlier_mad>winsorize` | — | 0 | **PROMOTED_TO_THE_SKILL_CARD** |
| `..._3` | `outlier_mad>hampel_filter` | — | 0 | **PROMOTED_TO_THE_SKILL_CARD** |
| `..._4` | `repair_level_shift>winsorize` | — | 0 | NEVER_VERIFIED |
| `..._5` | `repair_level_shift>outlier_mad` | — | 0 | NEVER_VERIFIED |

对照臂也走了完整生命周期（`outlier_mad` Draft → FLAGGED / EFFECT_NONSTATIONARY，
3 次验证）——这正是 DEV-AUTO-1 的冻结臂被多冻掉的东西。

---

## 3. Skill 是否变化，并被后续 Fast 使用

**变化：是，两次，走的是既有 compile / validate / load 接口。**

| 晋升 | 提出于 | 验证于 | 卡体（前 → 后） | 通道 |
|---|---|---|---|---|
| 1 | u7 | u11 | `[outlier_mad]` → `[outlier_mad, winsorize]` | `OUTCOME_GAP` PATCH，`skill_library.entries/<K0>.body` |
| 2 | u7 | u15 | `[outlier_mad, winsorize]` → `[outlier_mad, hampel_filter]` | 同上 |

两次都 `compiled_and_materialized: true`，并用 Fast consumer 自己的解析器回读
校验（`body_readback`）。第二次的 parent SHA 正是第一次的 candidate SHA——
血缘串起来了，第二个修订是拿第一个当父版本判的。

**对照臂的卡在全部 16 个单元只出现过一个版本**，不变量成立。

**被后续 Fast 使用：一次。**

| 位置 | 修订臂卡体 | 部署 | 路由 | delayed |
|---|---|---|---|---|
| u11 | `[outlier_mad]`（尚未晋升） | `outlier_mad>winsorize` | `resupplied_draft` | +0.384932，门通过 |
| u12 | `[outlier_mad, winsorize]` | `outlier_mad>winsorize` | `searched_this_unit` | +0.042752，门未过 |
| u14 | `[outlier_mad, winsorize]` | `outlier_mad>winsorize` | **`recalled_skill`** | +0.141939，门通过 |
| u15 | `[outlier_mad, winsorize]` | `outlier_mad>hampel_filter` | `resupplied_draft` | +0.298041，门通过 |
| u16 / u23 | `[outlier_mad, hampel_filter]` | `outlier_mad` | `searched_active_program` | 第二张卡**从未被用** |

只有 **u14** 是"更新后的卡被检索并部署"。**晋升不等于被使用**：
第二次晋升之后，Fast 在 u16 和 u23 都自己搜索并选了朴素的父程序，
没有用新卡。Runner 全程没有强制选择或部署任何候选。

---

## 4. 整段效用、风险、覆盖与成本

16 个单元，两臂，全服务人群分母：

| | 冻结臂 | 修订臂 | 差 |
|---|---|---|---|
| Support 面平均收益 | 0.074545 | **0.130218** | +0.055673 |
| **delayed 面平均收益** | 0.064176 | **0.114025** | **+0.049849** |
| delayed 受损序列 | 15 | **19** | +4 |
| delayed 处理序列（覆盖） | 47 | **90** | +43 |
| 最坏单序列伤害 | 0.484687 | 0.484687 | **0** |
| 收益恰为 0 的单元 | 11 | **8** | −3 |
| 权威门通过 | 1/16 | **4/16** | +3 |
| 探测过的不同程序 | 2 | 5 | +3 |
| Consumer fits | 69 | **194** | ×2.8 |
| LLM 调用 | 90 | **112** | ×1.24 |

**归因：12/16 个单元两臂部署完全相同**，整段差异只在 4 格：

| 通道 | 单元 | delayed 差 |
|---|---|---|
| 候选**供给**（`resupplied_draft`） | u11、u15 | **+0.682973** |
| 更新后的 **Skill 卡**（`recalled_skill`） | u14 | +0.141939 |
| 新卡下的一次 Fast 搜索 | u12 | −0.027335 |
| 合计 | | +0.797577（÷16 = +0.049849） |

⚠️ **本包把两条通道同时打开，因此整体差异不能归给"Skill 自我更新"**。
主要贡献来自供给通道，而供给通道在第一次晋升发生**之前**就起了作用（u11）。
要把两者分开，需要第三个臂（只供给、不晋升）——本包未跑，也不自行扩展。

**修订臂并非一致更好。** u12 是一次真实的退步：它部署了
`outlier_mad>winsorize`（0.042752），而对照臂部署朴素 `outlier_mad`
拿到 0.070087；修订臂还多破了一条风险线（`single_series_harm` 0.402 > 0.30，
对照臂只破 `harmed_fraction`）。伤害序列 19 vs 15 也主要来自新增的覆盖。
最坏单序列伤害两臂相同，说明新增伤害是分散的，不是出现了更极端的个案。

**这是 n=1 的两臂对照。** 与 DEV-AUTO-1 不同的是，这次机制确实生效
（2 次晋升、部署路径不同），而且 12/16 格逐位相同把差异锁在了 4 格里——
这比单纯的臂间均值差可归因得多。但一次运行仍然无法估计运行间方差，
不要把 +0.0498 当作效应量的点估计。

---

## 5. 成本

| | fits | LLM | 挂钟 |
|---|---|---|---|
| 2 单元试跑 | 60 | 23 | 105 s |
| 中止的第一次尝试（晋升写回崩溃，见 §6） | 165 | 104 | — |
| 正式课程（16 单元 × 2 臂） | 263 | 202 | 392 s |
| **本包合计** | **488** | **329** | ≈ 10 分钟 |

正式课程用掉 263/1000 fits、202/400 LLM（每臂预分配 200，冻结臂用 90、
修订臂用 112，各自记账）。**五次/格调用帽已不存在**：本轮每格上限是该臂
剩余的包级额度（首格 200，逐格递减），u5 就用了 6 次——旧配置会在第 5 次截断。

---

## 6. 包内解决的三个实现问题

都是"这条路以前没人走到过"暴露出来的，均在包内修好并加了预检，
没有改动任何核心方法或授权边界。

**① `activate_approved` 撞上未落盘的 bundle。**
`Arm._build` 开一个空 `SnapshotStore`，方法拿到的却是从目录编译来的快照，
于是 `store.set_active(snap.runtime_bundle_sha)` 抛
`cannot activate an unmaterialized runtime bundle`。触发点是最普通的路径：
Fast 部署它本来就有的 K0 卡、delayed 面确认。DEV-AUTO-1 从未走到——
strict 准入下几乎没有部署，也就没有 approved 事件。
修法：建臂时把**它自己正在跑的那个快照**落进它自己的 store。
不授予任何权利，被落盘的就是它已经在用的快照。

**② 晋升写回的 manifest 形状是错的。**
`apply_workflow_revision` 用了 `{"kind": "CONTENT_SHA", "sha256": ...}` 和
一个裸字符串 `minimal_patch`。合约要求 `{"kind": "SHA", "sha": ...}`
（`contracts/harness.py`：`PATCH edit requires SHA surface precondition`），
`EditController._apply` 读的是 `minimal_patch["value"]`。
两处都照 `method.py` 的生产写法改正，并把 manifest 构造移进 try——
写不进去的晋升是一条记录在案的拒绝，不是课程终止。
**第一次正式运行就是死在这里**（u14 触发晋升时崩溃），已记为中止尝试并重跑。

**③ Draft 身份号碰撞（只记录，未修）。**
M-R0d 入场态重建把 `resupplied_draft_1` 直接 append 进 ledger 而没有推进
`_minted`（实测 `_minted = 0`），于是课程里第一个 `open_restricted` 又铸出
同名的 `resupplied_draft_1`。本轮两次晋升用的是 `_2`/`_3`，`by_id` 无歧义，
结果不受影响；但 `by_id` 只会返回第一个匹配，这是个潜在隐患。
修它会改动生命周期身份，超出本包授权，交裁定。

---

## 7. 边界

未读 +144 评价面、未读密封材料、未读 5 个 `TARGET_HELD_IN` 单元；
未改风险线（material 0.005 / 0.20 / 0.30 / min_treated 5）、风险分母、
冻结 Scope 栅格、Consumer、DSL、HEC-1 合同（`PER_UNIT_ARM_BUDGET["llm_calls"]`
仍是 5，只是这个 development 配置不对内循环采用它）；未提交 git；
未覆盖任何历史收据。两臂唯一区别是结构修订通道是否开放，修订开销分臂记账。

---

## 8. 下一步（交裁定，未自行推进）

**三臂对照，把供给与晋升分开**：控制臂 / 只供给不晋升臂 / 供给+晋升臂，
其余全同。依据：本包的全部证据表明这两条通道都在起作用，且供给贡献更大
（+0.683 对 +0.142），但一次运行的两臂设计在结构上分不开它们。
这个问题不解决，"Skill 会自我更新并带来收益"这句话就没有干净的证据。

次一级：本轮 11 条 Scope 提议全部死在风险线上（不是栅格），
与 DEV-AUTO-1R 的归因一致；Scope 通道是否值得继续投入，建议单独裁定。

---

## 9. 交付清单

| 文件 | 说明 |
|---|---|
| `evaluation/main_protocol_p4/run_dev_auto2_frame_c.py` | Frame C 课程入口 |
| `evaluation/main_protocol_p4/smoke_dev_auto2_frame_c.py` | 10 项预检，0 LLM / 0 fits |
| `evaluation/main_protocol_p4/audit_dev_auto2_attribution.py` | 通道归因，0 LLM / 0 fits |
| `evaluation/main_protocol_p4/dev_auto1_revision.py` | 修正 `apply_workflow_revision` 的 manifest 形状（§6②） |
| `artifacts/main_protocol/dev_auto2_frame_c__run1.json` | 完整课程收据 |
| `artifacts/main_protocol/dev_auto2_attribution.json` | 归因表 |
| `artifacts/main_protocol/dev_auto2_frame_c__dryrun.json` | 2 单元试跑收据 |
| `_scratch/dev_auto2/aborted_attempt1_manifest_shape_progress.json` | 中止尝试的检查点 |
