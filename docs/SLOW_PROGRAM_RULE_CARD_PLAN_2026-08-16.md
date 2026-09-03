# 主线计划书：结构化 Slow Program Rule Card（rev4，2026-08-16）

> **历史计划，已停止继续执行未来步骤。** 已完成的 P0–P4 机械与实验记录继续有效；
> 2026-08-17 起，后续主线统一由
> `docs/TASK_EPISODE_HARNESS_EXECUTION_PLAN_2026-08-17.md` 管理。
> 不得从本文件恢复 S1/S4/P5 等旧分支，除非新任务书的实测首阻塞明确重新授权。

> **本文件是 rev4。P0–P3 已由本地 agent 实现并通过（74 passed）。**
> **rev4 只改 P4**：收窄为 **ADD-only 自然纵向切片**，代号 `NATURAL_ADD_ONLY_SLOW_PROGRAM_EDIT_SLICE`。
> 本地评审提的五条承重问题**全部经代码核实成立**，已逐条改正（见「修订记录」）。
> **交付对象**：本地 agent 执行。

---

## 主坐标（固定，不随单次实验结果移动）

> 本节是**防漂移装置**。任何提案与本节冲突时，改提案，不改本节。
> 修改本节需要用户显式裁决，并在此留下记录。

**项目目标（工程能力，非现象证明）**
- **C1** Agent 驱动的数据 Readiness：结合任务需求与时序模式，自主生成针对性的数据准备策略
- **C2** 反馈驱动的自适应优化：用下游任务表现验证效果，并利用成功/失败经验持续优化 Harness

**主线**：固定说明书 + Slow Agent 可修改的**结构化 Rule Card**（Program 面）。
**主实验**：A5 vs A3 —— Source Experience 是否让 Harness 更快、更安全地适应新 Target。

**防漂移规则（本轮新增，因已发生多次改题）**

| 规则 | 理由 |
|---|---|
| 单个局部实验结果**不得**直接推出项目转向。转向提案先进「待评估」，不直接改计划 | 已发生：Pattern 关→开→关；Ordering Card 辅助→主线→辅助 |
| family 的开/关必须有显式 verdict；重开需要**新证据 + 显式撤销旧 verdict** | 同上 |
| 引用代码必须带 `file:line`，且**核对 `def` 行**，不得从函数体推断函数名 | 已发生：rev1 引用了不存在的 `_route()` |
| 主线未完成前，任务轴 / 模型轴 / anomaly 一律进 backlog，不进计划 | 已发生：主线未完成就开始讨论任务轴 |
| 区分机制层级：Runtime 自动行为 ≠ Slow 自主决策 | 已发生：把 C5 的 Runtime rebind 表述成「Slow Rule Card 跑通」 |
| **不得为了让某条路径「可运行」或「可判定」而擅自调紧 / 调松判据**；判据只能由架构既有语义决定 | 已发生三次：rev1 的 `delayed ≥ +M`（调紧）、rev2 的 `constrained=False`（调松）、rev3 的「每批恰好一次 Slow Update」（调紧） |

---

## 修订记录（rev1 → rev2）

五条全部代码核实成立，**评审正确，rev1 错误**。

| # | rev1 写的 | 核实结果 | 处置 |
|---|---|---|---|
| 1 | 调用 `first_fault._route()` | **该函数不存在。** 实际是私有 `_supply_failure()`（`first_fault.py:149`）与公开入口 `assess_case(CaseFacts) -> AssessmentResult`（`:608`） | 删除该假设，改为 **P1 适配层**（§4） |
| 2 | 有邻近 Skill 时同时开放 ADD + PATCH，Slow 自选 | **与授权语义冲突。** `fault_routes.json`：`SKILL_LIBRARY_GAP: ops=['ADD']`、`SKILL_CONTENT_GAP: ops=['PATCH']`——一次归因只授权一个操作，Controller 会拒掉另一个 | 第一版收窄为**单 Surface**（§5）；双 Surface 移入 backlog（§5.3） |
| 3 | 未提及 PATCH 落盘与 replay 的一致性 | **真实阻塞。** `edit_controller.py:704` 的 PATCH 是 `replacement = minimal_patch["value"]` 直接 `_pointer_set` 落盘；而 ADD 的 body 由 Runtime 从冻结 steps 重写（`method.py:902`）。**两条路径不对称** | 新增 **P0**，列为第一阻塞（§3） |
| 4 | S2「同候选供给逐格硬断言」 | **成立，且 rev1 自相矛盾**——同一份文档既要求「Reference 1/2/3 三臂全开」（Source 本就该改变供给），又要求「候选集合逐格相同」。两者不能同时成立 | 改为「同**能力边界与预算**」+ 差异作为**实验结果记录**（§7.2） |
| 5 | 「有效 Skill」= 下一入口采用 **且 delayed ≥ +M** | **过严。** Skill 存活由**单侧 harm veto** 决定：`online_loop.py` 的撤销门是 `delayed_utility < -M` | 拆成三个指标（§7.4） |


### rev2 → rev3（第二轮评审，三条全部核实成立）

| # | rev2 写的 | 核实结果 | 处置 |
|---|---|---|---|
| R-6 | P1 只填 `_supply_failure` 读的字段，然后调 `assess_case(CaseFacts)` | **不成立。** `_fold` 返回**第一个** FAIL/UNKNOWN（`first_fault.py:582`），而阶段序是 `ELIGIBILITY → OBSERVATION → LOCALIZATION → MECHANISM → RETRIEVAL_POLICY → CANDIDATE_SUPPLY → …`——**程序供给分支前面还有 5 个阶段**，全部跑在 fixture 默认值上。「不填 oracle 字段」并不能避免伪造 | **P1 改为提炼公开纯函数**，不经过 `assess_case`、不构造 `CaseFacts`（§4） |
| R-7 | `constrained_proposal_succeeds` 固定为 `False` | **与「无证据就 ABSTAIN」自相矛盾**（rev2 自己写了这条纪律）。`_supply_failure` 末路已有 `CANDIDATE_SUPPLY_UNKNOWN → EVIDENCE_BACKLOG → ()`，三态本就可表达 | 改为**三态**（§4.3） |
| R-8 | P4 未说明 Source Experience 如何进入 **Slow** 输入；且 prequential 与即时 Slow 触发冲突 | **成立，且实测出三个更具体的子缺口** | 新增 §7.6 / §7.7 |


### rev3 → rev4（P0–P3 交付后，P4 收窄）

| # | rev3 写的 | 结果 | 处置 |
|---|---|---|---|
| R-9 | P4 直接跑三臂 STATIC/A3/A5 | **照现状跑是空的。** 在线适配器 `build_program_supply_facts` 恒返回 `EXPRESSIBILITY_UNKNOWN` + `constrained_proposal_succeeds=None` → **每个在线案例都 ABSTAIN**，三臂都会产出 0 次 Harness 更新 | P4 收窄为 **ADD-only 切片**；Runtime 必须**挣得** `PROVEN_EXPRESSIBLE`（§7.0） |
| R-10 | §7.7 验收断言「每批**恰好**一次 Slow Update」 | **与「ABSTAIN 是合法结果」冲突**（我自己调紧了判据） | 改为 批内 == 0、批末 **≤1** |
| R-11 | 「E5 必须在 P4 之前，否则 P4 数字要重跑」 | **论据不再成立。** P4 收窄后不产出最终测量值 | E5 移到 **P5（正式 A5/A3）之前** |

### rev4 补丁（本地评审另加四条承重，2026-08-16）

| # | rev4 初稿 | 核实结果 | 处置 |

|---|---|---|---|

| R-12 | E-1 只看 `verify().passed` | `ScopeExecutor.verify()` 零训练窗也可能 `passed`，且原 `WindowVerification` 不暴露 identity-equivalent / 修改窗 / 行为指纹 | `WindowVerification` 增加每窗 behavior hash、modified flag、identity-equivalent flag；E-1 必须 `checked_windows > 0`、至少一窗实际修改、程序整体非 identity-equivalent；两个选项至少要有一窗行为不同。全部只读 verifier 产物，仍零 Outcome、不调 `evaluate()` |

| R-13 | `capability_skill_exists` = 任意 applicability 匹配的 capability skill | 一个无关 imputation Skill 可能挡住 outlier ADD | 改为 program-aware：当前 Context 可用，且**同 program family**（算子序列签名）或与 verified alternative 至少一窗行为等价 |

| R-14 | E-2 = 去掉 `confirmed_cause` 默认值 | 默认值已经去掉，真正的洞是组路径不接收/不验证现有路由结果 | `handle_group_feedback` 接收 `ProgramSupplyDecision` + 单 Surface catalog；仅当 `cause=SKILL_LIBRARY_GAP`、`actionability=EDITABLE_M0`、catalog 恰为一个 capability ADD Surface 才构造 manifest，否则 `route_not_add_only` 且不调 Slow |

| R-15 | 空转看 A3 为 0 | A3=0 而 A5>0 正是 Source Experience 可能让路线首次触发，不是 vacuous | 只有 **A3 与 A5 都为 0** 才判 `P4_VACUOUS_NO_ROUTE_FIRED`；开发数据上的 PASS 标签写 `NATURAL_ADD_ONLY_SLOW_PROGRAM_EDIT_DEV_PASS` |

| R-16 | 现有 Skill 与 alternative 任一窗口输出相同即视为等价 | 「两个程序恰好在干净窗口都不动作」会被集合交集误判为等价 | 同 family，或**全部对齐窗口的 ordered behavior hash tuple 完全相同**；禁止集合交集 |

| R-17 | contrast 数值分类先看 support≥M | support 正 / delayed 负的常见翻转会被错放 positive | **优先 `episode.relation`**：CONFLICT→conflict、POSITIVE→positive、NEGATIVE→negative；缺 relation 才数值回退 |

| R-18 | P4 自动 group 路径仍调恒 Unknown 的 online route；且绑定只认 patch_id | P4 使用该路径会永远 abstain；同名 patch_id 可能 verifier 验证 A、Runtime 执行 B | 新增 `methods/ttha/p4_runner.py`：Card 只构建一次；`run_p4_group_update` 用同一 Card 调 `route_verified_program_supply_fault`；绑定单位为 `(patch_id, exact ordered program_steps)`，同名不同 steps 或重复 patch_id 在调 Slow 前拒绝；Slow 只看到 verified 后的同一 Card；`verified_choice_offered` 显式传入且 P4 中 None/False 均不按数量退化成「真实选择」 |

**代码事实（已核实）**：`handle_group_feedback`（`method.py:649`）**预先构造** manifest——
`operation=EditOperation.ADD`、`surface_precondition={"kind":"ABSENT"}`、`edit_id` 与
`target_surface_id` 均为硬编码模板；Slow 的 `proposed` 只被用来取 `patch_id`。
→ **组路径本来就只能 ADD。** 收窄不是让步，是与代码能力对齐。

---

## 0. 为什么回主线

排序卡对「Harness 自我修改」的贡献 ≈ 0，天花板已实测：

| cohort | 当前 STATIC / A5 | oracle 上界 | 剩余空间 |
|---|---|---|---|
| A · kdd2018 | 0.0577 | 0.0681 | +18.0% |
| B · metr_la | 0.4613 | 0.5077 | +10.1% |

oracle 是逐格开天眼的排序，够不着的上界。叠加 `fast_propose_v1.maxItems=3` 的硬顶，
可学习状态约 1.6 bit / scope。**封存，不再扩展，不建探索槽。**

留下的可用资产：compile → snapshot → retrieval → prequential 冻结 → 撤销 这套机械已压测
（420 次供给同一性断言、630 arm-cell 零仪器偏差、7 revision 连续生效），主线直接复用。

---

## 1. 已经存在什么（动手前必读）

全部已核实。任何「需要新建 X」的提案先对照本表。

| # | 事实 | 位置 |
|---|---|---|
| F-1 | **结构化 manifest 是硬 schema**，13 必填字段，与原设计 surface/trigger/action/scope/risk/evidence 几乎一一对应 | `schemas/slow_edit_v1.json` |
| F-2 | 动作词汇两个：`ADD`（precondition `ABSENT`）、`PATCH`（precondition `SHA`） | 同上，`edit_manifest.oneOf` |
| F-3 | **归因入口是 `assess_case(CaseFacts) -> AssessmentResult`**（`_fold` → `FaultAttribution`）。程序供给分支的判断在私有 `_supply_failure()` | `feedback/first_fault.py:594 / :568 / :153` |
| F-4 | 授权表 `fault-routes/2` 逐 family 给 `{actionability, target_classes, operations}` | `feedback/fault_routes.json` |
| F-5 | surface 模板 `skill_library.entries/{skill_id}` 及 `.body` / `.observable_applicability` / `.risk_guards` **已注册** | `first_fault.py:166/178/404/555`，`fixtures/contract_policy.py:467/513` |
| F-6 | **SHA 由 Runtime 填**，Slow 的猜测一律忽略（避免假性 `StaleEditError`） | `method.py:1073`，`edit_controller.py:423` |
| F-7 | **正向/负向/冲突经验卡已存在** = Reference 1/2/3，作用在 **Program Supply 层** | `experience_memory.py:547-577`，`fast_agent.py:302` |
| F-8 | **REBIND 不是 Slow 的动作**——Runtime 每轮按 `public_parameter_bindings` 自动重绑。C5 证明的是这个 | `fast_agent.py:325` |
| F-9 | **ADD 路径是闭合的**：Runtime 用白名单 steps 重写 body，replay 用**同一** steps | `method.py:902 / :955` |
| F-10 | **归因是「第一个失败阶段」语义**：`_fold` 返回第一个 FAIL/UNKNOWN；程序供给分支之前有 5 个阶段 | `first_fault.py:568 / 170` |

### 1.1 授权表实况（程序面相关行）

```
SKILL_LIBRARY_GAP     act=EDITABLE_M0  ops=['ADD']           classes=['capability']
SKILL_CONTENT_GAP     act=EDITABLE_M0  ops=['PATCH']         classes=['capability']
SCOPED_SELECTION_GAP  act=EDITABLE_M0  ops=['PATCH']         classes=['capability']
MECHANISM_AMBIGUITY   act=EDITABLE_M0  ops=['PATCH']         classes=['capability']
RISK_GAP              act=EDITABLE_M0  ops=['ADD','PATCH']   classes=['safety', …]
EXPRESSIBILITY_UNKNOWN act=EVIDENCE_BACKLOG ops=[]           （= ABSTAIN）
```

**一次归因只授权一个操作**（`RISK_GAP` 是唯一的双操作行——说明双操作路由**是可表达的**，
只是程序能力面当前没有这样一行。这条是 §5.3 backlog 的先例，不是现在动的理由）。

---

## 2. 真正的缺口（四处）

| # | 现状 | 证据 | 后果 |
|---|---|---|---|
| G-1 | **PATCH 落盘内容与 replay 的 Program 无绑定** | `edit_controller.py:704` | 可能 replay 的是 Program A、落盘的是 Slow 文本 B。**修复前任何 `PATCH_PROGRAM_PASS` 不可解释** |
| G-2 | 程序供给的路由逻辑**只存在于私有 `_supply_failure()` 内部**；唯一公开入口 `assess_case` 会先跑 5 个前置阶段 | `first_fault.py:149 / 582 / 183` | 在线要么够不着，要么被前置阶段的 fixture 默认值抢先命中 |
| G-3 | `confirmed_cause = "SKILL_LIBRARY_GAP"` 是**默认参数** | `method.py:299/486/605/830` | 调用方在调 Slow 之前就断定了原因 |
| G-4 | `online_loop.py` 默认 catalog 只有 `allowed_operations:["ADD"]`；而 `slow_agent.py:263` 的 `add_rule` 却写着「PATCH an existing authorized surface instead」 | 同左 | 那条指令指向空集合 |

---

## 3. P0 —— Runtime-owned PATCH body binding（**第一阻塞**）

### 3.1 问题

| 路径 | body 谁写 | replay 用什么 | 一致？ |
|---|---|---|---|
| ADD | **Runtime**：`nv["body"] = "Frozen program steps: " + json(steps)`（`method.py:902`） | 同一个 `steps`（`method.py:955`） | ✅ |
| PATCH | **Slow**：`replacement = minimal_patch["value"]` 直接 `_pointer_set`（`edit_controller.py:704`） | 若取自 `patch_id` 白名单 → **是另一个对象** | ❌ |

### 3.2 最小修复

当 target surface 是 **capability Skill 的 `.body`** 且 manifest 带 `patch_id` 时：

1. Runtime 从 `typed_patch_options` 白名单解析出 `steps`（与 ADD 路径**同一个解析函数**）；
2. Runtime **强制覆写** `minimal_patch["value"] = "Frozen program steps: " + json(steps)`，
   忽略 Slow 提供的任何文本；
3. **落盘后断言**：从 candidate snapshot 读回该 skill 的 body，`_parse_frozen_steps` 出来的
   steps 必须与 replay 用的 `steps` **逐元素相等**；不等 → `apply_failed`，不进 pending。

### 3.3 完成判据

- 一个测试：构造「Slow 给出与 patch_id 不一致的 `minimal_patch.value`」→ 落盘 body 仍等于
  白名单 Program，且断言通过；
- 一个测试：人为破坏一致性 → 断言**必须**抛错并停在 `apply_failed`；
- **不改** `slow_edit_v1` schema、不改 `fault_routes.json`、不改 h0。

---

## 4. P1 —— 提炼公开的 Program Supply 路由纯函数

### 4.1 为什么「部分填充的 CaseFacts + assess_case」是坏方案（rev2 的方案作废）

`_fold`（`first_fault.py:568`）返回**第一个** FAIL/UNKNOWN 的阶段。而 `_build_assessments`
的阶段序是：

```
ELIGIBILITY -> OBSERVATION -> LOCALIZATION -> MECHANISM -> RETRIEVAL_POLICY
  -> CANDIDATE_SUPPLY -> CANDIDATE_SELECTION -> COMPILATION -> EXECUTION -> OUTCOME_RISK
```

**程序供给分支前面还有 5 个阶段**，它们读 `damage_d` / `candidate_utilities` /
`chosen_candidate_id` / `chosen_probe_directions` / `public_evidence_discriminative` /
`localization_iou` / `mechanism_identified` / `mechanism_contradiction` 等字段。

不填这些字段**不等于不伪造**——它们以 fixture 默认值存在，并且可能**抢先命中**一个更早的
first fault。当前默认值恰好让前 5 个阶段通过（例如 `damage_d=0.30` > `critic_damage_min=0.01`），
但这是**巧合，不是设计**：任何一次默认值或 `m0_rules.json` 的改动都会静默改变在线路由。

### 4.2 正确的最小修复

把 `_supply_failure()` 的程序供给逻辑提炼成一个**公开纯函数**：

```python
def route_program_supply_fault(
    *,
    expressibility_status: str,
    expressibility_cause: str | None,
    capability_skill_exists: bool,
    skill_retrieved: bool,
    constrained_proposal_succeeds: bool | None,
) -> tuple[str, str, tuple[str, ...]]:      # (fault_family, actionability, surfaces)
```

- **不经过** `assess_case()`、**不构造** `CaseFacts`、**不触碰** 55 字段里的任何 oracle 字段；
- `_supply_failure()` 内部改为调用它，保证受控 minipipe 与在线走**同一段逻辑**（禁止复制粘贴）；
- 这不是新归因平台，是把已有私有逻辑提成在线可安全调用的最小入口。

### 4.3 `constrained_proposal_succeeds` 必须是三态

rev2 计划把它固定为 `False` 再路由到 `SKILL_CONTENT_GAP`——**这与 rev2 自己写的
「宁可 ABSTAIN，不可猜 cause」直接冲突**。改为：

| 值 | 路由 | 何时允许出现 |
|---|---|---|
| `True` | `PROPOSAL_CONTROL_GAP` | DecisionTrace 或一次**真实**受限提案给出证据 |
| `False` | `SKILL_CONTENT_GAP` | 同上 |
| `None` | `CANDIDATE_SUPPLY_UNKNOWN` -> `EVIDENCE_BACKLOG` -> **ABSTAIN** | **在线默认**：没跑受限提案时 |

`_supply_failure` 末路本来就有 `CANDIDATE_SUPPLY_UNKNOWN, "EVIDENCE_BACKLOG", ()`，
三态是**现成可表达**的，不需要改 `fault_routes.json`。

**不得**为了「让 PATCH 路径可运行」而填 `False`。P3 的受控 PATCH 案例可以**显式提供已知的
`False`**——那是机制正控，不是在线归因。

### 4.4 在线取值来源

| 参数 | 在线来源 | 无证据时 |
|---|---|---|
| `expressibility_status` | 需要证据 | `EXPRESSIBILITY_UNKNOWN` -> ABSTAIN |
| `expressibility_cause` | 需要证据 | `None` |
| `capability_skill_exists` | 当前 view 中是否存在 capability skill | 可直接观测，无默认 |
| `skill_retrieved` | `trace.retrieved_skill_ids` 是否含 capability skill | 可直接观测，无默认 |
| `constrained_proposal_succeeds` | 一次真实受限提案 | `None` -> ABSTAIN |

**纯函数的每个参数都是必填关键字参数（无默认值）**——用签名从机制上杜绝「靠默认值路由」。

### 4.5 完成判据

- `route_program_supply_fault` 是纯函数，**所有参数无默认值**；
- `_supply_failure()` 改为调用它，且**既有 minipipe 测试全部仍然通过**（证明语义未变）；
- 每个分支各一个单测，含 `constrained_proposal_succeeds=None -> ABSTAIN`；
- 在线适配器只产出这 5 个参数，**不构造 `CaseFacts`**；
- **不新建通用归因平台**；不引入 oracle 字段（`clean_u` / `damage_d` / `private_*`）。

---

## 5. P2 —— 单 Surface S1

### 5.1 Claim（**收窄后的真实主张**）—— 阶段代号 `STRUCTURED_SLOW_PROGRAM_EDIT`

> **命名提醒（评审）**：P2 产出的是「结构化 Slow Program Edit -> Executable Capability
> Skill」，**还不是**独立的持久 Rule Card（尚未形成 trigger + action + evidence + risk
> 四件套的规则对象）。阶段 Claim 一律用 `STRUCTURED_SLOW_PROGRAM_EDIT`，
> **不得**称作「Evolvable Rule Card 跑通」。是否固化成完整 Rule Card，等 P4 自然正向后再定。

> Slow 能在 **Runtime 授权的单一 Program Surface** 内，从结构化 Program options 中
> **选择一个修改或 abstain**，并由 Runtime 将**同一个 Program** 应用、重放和核销。

分工固定为：

```
Runtime  选择合法修改面（由 P1 归因 + fault_routes 授权决定，确定性）
Slow     在该 Surface 的 typed Program options 中选择一个，或 abstain
Runtime  绑定（P0）、验证、replay、pending、delayed 批准 / 撤销
```

### 5.2 改动

- catalog 由归因结果生成，**只含被授权的那一个 surface + 那一个 operation**；
- `confirmed_cause` 去掉默认值，改为必填，由 P1 的归因结果传入；
- ABSTAIN 显式化：路由返回空 surface 集 → **不调用 Slow**，记 `abstained_by_route`；
  Slow 被调用但不产生 manifest → 记 `abstained_by_agent`。
  两者都**不**计入 `slow_replay_receipts_used`，**不**算 protocol_error。

### 5.3 明确移入 backlog（不在本轮做）

让 Slow 在 ADD 与 PATCH **两个** Surface 之间自主选择。
这需要新增一条受 Runtime 预授权的多操作路由（`RISK_GAP` 是先例），
属于**修改授权语义**，不是调用侧开关。
**前置条件**：§5.1 的 Claim 先在自然数据上成立一次。

---

## 6. P3 —— 两个受控机制检查

在自然三臂之前，先各跑一个受控案例，证明两条路径都真的闭合：

| 案例 | 路由 | 必须验证 |
|---|---|---|
| ADD 案例 | `SKILL_LIBRARY_GAP` | 真实落盘 → replay → pending → delayed → removal 恢复 |
| PATCH 案例 | `SKILL_CONTENT_GAP` | 同上，**外加 P0 的一致性断言**：落盘 body 的 steps == replay steps |

P3 的 PATCH 案例**可以显式提供已知的 `constrained_proposal_succeeds=False`**——
这是**机制正控**，用于证明 PATCH 链路闭合；它**不是**在线归因，
不得据此表述为「系统判断出内容缺口」。

**P3 判读边界：这两个案例只是受控机械检查，只能声称「ADD/PATCH 两条 Runtime 链路闭合」；不得据此声称自然 Slow Evolution 成立。** 自然进化的结论只能在 P4 的同期三臂上判读。

**审核补充（2026-08-16）**：仅有「delayed 拒绝 → active snapshot 不污染」不构成完整机制链。
受控案例还必须包含 **delayed 通过**：candidate snapshot 激活，且下一正常入口的 view /
候选供给能读到修改后的 Program。拒绝路径 + 通过路径都闭合，P3 才算完成。

任一断裂 → 报告并停，不重跑、不二次修复。

---

## 7. P4 —— ADD-only 自然纵向切片 `NATURAL_ADD_ONLY_SLOW_PROGRAM_EDIT_SLICE`

> **第一版的成功判据是「机制 + 至少一次自然正向」，不是 A5 &gt; A3。**
> A5/A3 的正式比较是 **P5**，不在本节。

### 7.0 先决条件（不满足则 P4 不得启动）

| # | 条件 | 为什么是前提 |
|---|---|---|
| E-1 | Runtime 能**挣得** `PROVEN_EXPRESSIBLE` + `SKILL_LIBRARY_GAP` | 在线适配器现在恒 `EXPRESSIBILITY_UNKNOWN` → 恒 ABSTAIN → **P4 空转** |
| E-2 | `handle_group_feedback` 接收现有 `ProgramSupplyDecision` + 单 Surface catalog；仅当 `cause=SKILL_LIBRARY_GAP`、`actionability=EDITABLE_M0`、catalog 恰为一个 capability ADD Surface 才构造 manifest | 否则路由结果到不了组路径，调用方会继续硬编码 cause |
| E-3 | Capsule 补 **C-1**（`negative` 桶）与 **C-3**（source/target provenance） | 否则 A5 的 Source 负例到不了 Slow，且报不出「用了哪些 Source evidence IDs」 |

**E-1 的实现约束（承重）**：证明「存在合法可执行 typed alternative」**只准调

`ScopeExecutor.verify()`，禁止调 `evaluate()`**。verify 零 Outcome——不消耗 Support 预算、

不把结果信息泄进路由决策。若用 `evaluate()` 证明，路由变成结果依赖，三臂预算也会发散。

（与 GRID0 census 的信息墙纪律 N4 同源。）

`verify().passed` 不是充分条件。一个 verified alternative 必须同时满足：

- `checked_windows > 0`；
- 至少一个窗口实际发生修改（`modified_windows > 0`）；
- 程序整体不得与 identity 等效（至少一个窗口 `effect_equivalent_to_identity == False`）。

「Slow 在两个选项间选择」还要求两个 Program **至少在一个窗口上行为不同**

（比较每窗 prepared-values 的 SHA-256 行为指纹）。这些字段全部来自 verifier 产物。



`capability_skill_exists` 的定义三臂必须**逐字相同**，并且是 **program-aware**：

「当前 Context 可用、且供应**相同 program family**（算子序列签名）或与 verified

alternative **全部对齐窗口的 ordered behavior hash tuple 完全相同**的 capability Skill」。

不是「库里有没有任意

capability skill」——无关 imputation Skill 不得挡住 outlier ADD。随着流推进各臂技能不同，

该值**会**合法地分化——那是被测量的对象，不是协议违反。

### 7.1 三臂（唯一变量 = Source Experience 是否入场）

| 臂 | Memory | Harness Update |
|---|---|---|
| STATIC | 空 | 不允许（rev 固定，Slow 不触发） |
| A3 | 只有 Target Episode | 允许，只由 Target Episode 驱动 |
| A5 | Source 的正向 / 负向 / 冲突 Episode 初始化 | 之后与 A3 **完全相同**的更新规则 |

### 7.2 Slow 的动作面（收窄）

- **只有一个 ADD Surface**；Slow 在其 `typed_patch_options` 中**选择一个 Program 或 abstain**；
- **`evidence_compiler=False`**——`True` 时是确定性 abstain 或收敛到单个白名单 patch，
  **零 LLM 选择**，无法支撑「Slow 选择」这个 Claim；
- **白名单必须 ≥2 个 typed Program options** 才算一次真实选择。只有 1 个时记
  `no_choice_offered`，**排除出「选择分布」指标**——否则会把「只有一条路」报成「Slow 选了它」；
- **不混入** PATCH / E5 / 排序卡 / 新 Gate。

**开发数据 PASS 标签**：`NATURAL_ADD_ONLY_SLOW_PROGRAM_EDIT_DEV_PASS`。PASS 至少要求：

Slow 在两个**行为不同**的合法 Program 中选择 → ADD → replay → delayed 非害 →

下一批真实检索并执行。缺任何一环都不叫 PASS。

### 7.3 什么必须相同、什么允许不同

**必须相同（能力边界与预算）**：Operator contracts、候选数量上限（`maxItems=3`）、
Support 预算、delayed 窗口、verifier 参数、Slow 调用预算、Runtime 授权规则、
路由与适配层、`capability_skill_exists` 的定义、Reference 1/2/3 **三臂一律打开**。

**允许不同，且必须逐格记录——这些差异就是实验结果**：
`candidate_set_delta`、`program_family_delta`、`ADD / ABSTAIN delta`、
`typed_option_set_delta`（白名单本身的差异）。

### 7.4 prequential 纪律与实现约束

`handle_feedback_support` 会在**单个** material failure 后**立即**触发 Slow
（`online_loop.py`），与「整批冻结」冲突。实现必须写成：

```
批内    run_online_round(..., allow_slow=False)      只积累 Episode，不触发 Slow
批末    build_contrast_capsule(group, all_episodes=<A5 含 Source>)
        -> handle_group_feedback(...)                至多一次结构化 Slow Update
下一批  才使用新 snapshot
```

**验收断言（rev4 更正）**：
- 整批期间 Slow 调用次数 **== 0**；
- 批末 Slow Update **≤ 1** 次（**不是**「恰好一次」——ABSTAIN 是合法结果，
  强制每批一次等于禁止弃权）；
- 批内出现任何 snapshot 变更 → 协议违反，停。

### 7.5 Source Experience 必须真的进入 **Slow** 的输入（不只是 Fast）

Reference 1/2/3（`experience_memory.py:547-577`）主要作用在 **Fast 的 Program Supply**。
若不额外接线，即使 A5 与 A3 不同，也**无法区分**三种解释：
① Source 改变了 Fast 候选；② Source 改变了失败轨迹；③ Slow 真正利用 Source 做了不同修改。

通道是 `build_contrast_capsule(group, *, all_episodes=…)`（`group_fault.py:100`）——
A5 的 `all_episodes` 必须包含 Source Episode。**三个子缺口**：

| # | 实况 | 位置 | 必须补 |
|---|---|---|---|
| C-1 | contrast 只有 `{"positive", "conflict"}`，**没有 `negative` 桶** | `group_fault.py:183` | 加 `negative`（`support_gain < -M`）——否则 **Source 负例永远到不了 Slow** |
| C-2 | 只收 `_full_workflow_of(e) == wf` 的 Episode | `group_fault.py:191` | 第一版**不改行为**，但必须记录「因指纹不符被过滤掉的 Source Episode 数」 |
| C-3 | ref 只有 `{episode_id, origin, support_gain}`，**无 source / target 标记** | `group_fault.py:195` | 加 provenance——否则报不出「用了哪些 Source evidence IDs」 |

报告字段建议叫 `source_episode_ids_supplied_to_slow`：它只证明「被放进 Slow 输入」，
**不能**证明 LLM 实际阅读或引用；无需为此修改 Schema。

### 7.6 承重指标

| 指标 | 定义 |
|---|---|
| `feedback_to_local_draft` | 到「当前 Support 正向」所需的反馈数 |
| `feedback_to_local_active` | 到「同域 held-in / 下一正常入口正向」所需的反馈数（对应 `_update_delayed_status` 里真实存在的 `LOCAL_ACTIVE`） |
| `final_surviving_skill` | 流结束时 delayed **未转害**、未被撤销 |
| 路由分布 | 各 fault family 命中次数；ABSTAIN 分 `abstained_by_route` / `abstained_by_agent` |
| 选择分布 | 白名单 ≥2 时 Slow 的选择；`no_choice_offered` 单独计，不进此项 |
| Source 引用 | A5 每次 Slow Update **进入 Slow 输入**的 Source evidence IDs（报告名 `source_episode_ids_supplied_to_slow`）；以及被指纹过滤掉的 Source Episode 数 |
| 无效 Program 探测数 | 被探测但 Support &lt; M 的 Skill 候选数 |
| harm | harm_count 与 harm_magnitude 分开 |
| delayed 数值 | **照常报告**，但不承担第二次收益证明 |
| **真实复用** | 下一批该 Skill 是否被**真实检索、真实供给、真实执行**，以及保留还是撤销——不是「写进 snapshot」就算 |

> **delayed 的职责是单侧 harm veto**：`delayed ≥ -M` → 可保留；`delayed < -M` → 撤销或收缩
> （`online_loop.py` 撤销门实测为 `delayed_utility < -M`）。不得重新要求 delayed 必须显著正。

### 7.7 停止规则（跑之前冻结）

- 整条流跑完 **A3 与 A5 都产出 0 个 route-fired / approved skill** → 判

  `P4_VACUOUS_NO_ROUTE_FIRED`，**停并回报**。A3=0 而 A5>0 **不是** vacuous——

  那正是 Source Experience 可能让路线首次触发（但本切片仍不得输出 A5 优劣结论）。

  **不许**为了让路由触发而放宽 E-1 的证明标准、降低 verifier 门槛或改 `fault_routes.json`。
- 机械层断裂（落盘/replay 不一致、批内出现 snapshot 变更）→ 报告并停，不重跑、不二次修复。

### 7.8 判读纪律

- **冷启动收益与持续收益分开记账。** E2 里 A5 的全部优势集中在第一个 batch。
- **不用 permutation p 值**（本项目已误导三次）。判据是能力匹配的朴素基线。
- 单臂内「后期比前期好」**不**构成进化证据——必须同期 AB 对照。
- 本切片**不得**输出 A5 vs A3 的优劣结论；那是 P5。

---

## 8. 执行顺序

```
[已完成] 权限守卫  contracts/harness.py:333 _reject_mixed_card_authority + 1 测试
    ↓
P0 ✅ Runtime-owned PATCH body binding
    ↓
P1 ✅ 公开路由纯函数 route_program_supply_fault（contracts/program_supply.py）
    ↓
P2 ✅ 单 Surface + 空 catalog → abstained_by_route
    ↓
P3 ✅ 两个受控机制检查（ADD / PATCH，含 delayed 拒绝与 delayed 批准两条链）
    ↓
E-1/E-2/E-3 ✅ P4 的三个先决条件（§7.0，含 rev4 四条补丁）
    ↓
P4   ADD-only 自然纵向切片  NATURAL_ADD_ONLY_SLOW_PROGRAM_EDIT_SLICE
    ↓
E5   F4 接候选队列 —— 一个接线 + 一个集成测试 + 完成即冻结
    ↓
P5   正式 A5 vs A3（F4 全程开启，三臂共享）
```

**E5 的位置（rev4 更正）**：rev3 说「E5 必须在 P4 之前，否则 P4 数字要重跑」——
论据**不再成立**，因为 P4 收窄后不产出最终测量值。E5 应在 **P5 之前**完成并冻结。
一旦它长出新 Gate / 新调参 / 新 Runner，立刻停并回报。

---

## 9. 明确不做

| 项 | 理由 |
|---|---|
| 扩展 Ordering Card / 强制探索槽 | 天花板已测出（+18% / +10%），封存 |
| 第一版让 Slow 同时选 ADD 与 PATCH | 与 `fault-routes/2` 授权语义冲突；移入 §5.3 backlog |
| 新建 `card-entry/1` 或改 `skill-entry/1` / `slow_edit_v1` | 载体够用；改契约会连锁 |
| 新建通用归因平台 | P1 只做程序供给分支的最小纯函数适配 |
| 在适配层伪造 oracle 字段 | `clean_u` / `damage_d` / `private_*` 在线无对应物 |
| 自由文本 Guidance patch | G3 / P4 两次 `PATCH_REJECTED`，family 已关闭 |
| 逐格 Context trigger / Pattern trigger | 七轮实验证明当前不可得，保持暂缓 |
| 任务轴 / 模型轴 / anomaly | 主坐标规则：主线未完成前进 backlog |
| P4 里混入 PATCH / E5 / 排序卡 / 新 Gate | 会让「第一次自然正向」不可归因 |
| 为让路由触发而放宽 E-1 证明标准 | §7.7 停止规则；宁可判 `P4_VACUOUS_NO_ROUTE_FIRED` |
| per-revision receipt / revision ledger / SHA 体系 | 反过度工程 |

---

## 10. 数据、纪律与分工

- **零 virgin series**：P0–P4 全部在已 development exposed 的数据上跑；
  新 series 必须先出名单交用户批 exposure。
- **只追加不改写**；**LLM 不批准自己的 patch**；**配对同场验证**；
  **机械层断裂即停**；**委派深度 = 1**。
- 每实验最多：一个 runner package + 一个主报告 section + 一个承重 integration test。

| 角色 | 职责 |
|---|---|
| 本地 agent | P0–P3 实现与测试；P4 runner 与数据收集；机械层核实 |
| 主控 agent | 协议冻结、判读、verdict 裁定、与既有 verdict 的一致性检查 |
| 用户 | exposure 审批、主坐标修改裁决 |

---

## 附：本计划推翻的既往表述

| # | 曾经说过 | 更正 |
|---|---|---|
| 1 | 「C1 那条线已经通了（C5）」 | **错。** C5 证明的是 Runtime 自动重绑定，不是 Slow 自主决定复用/修改/新增（F-8） |
| 2 | 「结构化 Rule Card 还没建」 | **不准确。** manifest 结构、动作词汇、授权表、surface 模板、PATCH 链路均已存在 |
| 3 | 「经验从未固化」 | **错**（E1 已更正）。`_write_snapshot_tree` 本来就会写 `skills/learned/<id>.json` |
| 4 | rev1「调用 `first_fault._route()`」 | **错。该函数不存在**——从函数体推断了函数名，未核 `def` 行。真实入口见 F-3 |
| 6 | rev2「填几个字段后调 `assess_case`」 | **错。** `_fold` 取第一个失败阶段，程序供给前有 5 个阶段跑在 fixture 默认值上（F-10） |
| 7 | rev2「`constrained_proposal_succeeds` 固定 `False`」 | **与自己写的「无证据就 ABSTAIN」冲突。** 改三态 |
| 8 | rev3「每批**恰好**一次 Slow Update」 | **错，我调紧了判据。** 与「ABSTAIN 是合法结果」冲突，应为 ≤1 |
| 9 | rev3「E5 必须在 P4 之前」 | **论据失效。** P4 收窄后不产出最终测量值，E5 应在 P5 之前 |
| 5 | rev1「三处调用侧开关」 | **低估。** 实际含一处授权/绑定语义修复（P0）+ 一处适配层（P1） |


---

## 附：P0–P3 执行记录（本地 agent，2026-08-16；只追加，不改上方协议）

**状态：**

- `E1_E2_E3_COMPONENT_PASS`
- `P4_EXACT_PROGRAM_BINDING_PASS`
- `P4_RUN_COMPLETE`
- `P4_SELECTED_PROGRAM_REPLAY_REJECTED`
- `P4_INTENDED_CAPABILITY_INCONCLUSIVE_SLOW_EVIDENCE_NOT_BOUND`
- `P4_ARM_DISTINCTION_INERT`
- `P4_HEADROOM_PER_CONTEXT_HEADROOM_EXISTS_NO_COMMON_PROGRAM`
- `S1a_SCOPE_HYPOTHESIS_NOT_EXPRESSIBLE_IN_BINS`
- `FIX_CAPSULE_INTO_CARD_PASS`
- `FIX_CANDIDATE_CONDITIONED_RETRIEVAL_PASS`
- `NATURAL_CONTEXT_SPLIT_REBIND_DEV_BLOCKED_AT_S1a`

S1a 零 LLM 穷举：9 个 numeric observable 在 888/984 的 frozen-bin 标签完全相同，
没有任何单特征 bin 谓词能分离两个 discovery origin。因此不发 LLM、不做 S1b/S1c，
直接进入 S4 方向（新建/组合 Workflow）。Capsule 接入 Card 与 candidate-conditioned
检索两项零 LLM 修复已完成并有测试。

| 项 | 实现 | 测试 |

|---|---|---|

| P0 | `slow_agent.py:111` `bind_frozen_patch_program`（Runtime 覆写 `minimal_patch.value`）；`slow_agent.py:141` `verify_frozen_patch_program`（从 candidate snapshot 读回 body，用 `fast_agent._parse_frozen_steps` 与 replay steps 逐元素比对）；`method.py:419/485` 与 `method.py:995/1056` 在 PATCH apply 前绑定、apply 后读回 | `tests/integration/test_slow_program_rule_card_p0_p3.py:173`（ADD 受控案例）；`:222`（PATCH 受控案例 + Slow 文本被忽略 + readback 一致）；`:400`（人为破坏一致性 → 抛 `FrozenProgramBindingError` → `apply_failed`，不进 pending） |

| P1 | `contracts/program_supply.py:19` `route_program_supply_fault(...)` 公开纯函数，**五个参数全部 keyword-only 必填无默认值**；`first_fault.py:153` `_supply_failure` 只做委托；`CaseFacts.constrained_proposal_succeeds` 改为 `bool | None`（默认 Unknown）。在线适配 `methods/ttha/program_supply.py:76` `build_program_supply_facts(trace, episode, view)` 不构造 `CaseFacts`、不调 `assess_case`；在线 `expressibility_status=EXPRESSIBILITY_UNKNOWN`，`constrained_proposal_succeeds=None` → ABSTAIN | `tests/methods/test_program_supply_route.py:58`（签名无默认值）；`:69`（各分支）；`:136`（`_supply_failure` 只是委托，无私有副本）；`:152`（受控 `assess_case` 与在线同一条路由）；`:210`（适配层字段显式枚举赋值 + 在线 ABSTAIN） |

| P2 | `program_supply.py:130` `build_single_surface_catalog` 只产出一个授权 surface + operation，空集=ABSTAIN；`method.py:336/546/917` `confirmed_cause` 必填；`handle_group_feedback` 改收 `route_decision` + `surface_catalog`（不再有 `confirmed_cause`）；`online_loop.py` material failure 先走 P1 路由，空 catalog 记 `abstained_by_route` 且不调 Slow；方法层 Slow 无 manifest 记 `abstained_by_agent`。`online_loop.py` 空 catalog 分支不再落入未赋值 `sev` 的核销块（审核 bug 已修）。`slow_agent.py:251` 的 Slow 输入 `add_rule` 改为按单 Surface catalog 动态生成（ADD 时明确禁止 PATCH，PATCH 时明确禁止 ADD，空 catalog 要求 abstain），关闭 G-4；历史/开发调用点已显式补传 cause（保持旧默认语义 `SKILL_LIBRARY_GAP`） | `tests/methods/test_program_supply_route.py:289`（单 ADD surface）；`:306`（单 PATCH surface，按 retrieved skill 选一个）；`:328/369`（不可编辑/目标不存在 → 空 catalog）；`:356`（四入口 `confirmed_cause` 必填）；`tests/integration/test_slow_program_rule_card_p0_p3.py:285`（agent abstain 不 pending）；`tests/functional/test_p2_online_route_abstain.py:82`（真实 material failure → Unknown → `abstained_by_route`：Slow 调用 0、Slow/Support 预算不增加、无异常） |

| P3 | 两个受控机制检查都在 `tests/integration/test_slow_program_rule_card_p0_p3.py`：ADD 案例（`SKILL_LIBRARY_GAP`）与 PATCH 案例（`SKILL_CONTENT_GAP`，显式 `False` 仅作机制正控）。两案例均走「真实落盘 → replay → pending → delayed 拒绝 → snapshot 恢复」；PATCH 案例额外验证 body steps == replay steps，并补了 delayed 通过 → candidate snapshot 激活 → 下一入口 view 检索 + `_skill_frozen_candidates` 供给修改后 Program 的成功生命周期（`:323`） | 同一文件，`pytest` 通过 |

| E-1 | `scope_executor.py:44` `WindowVerification` 增加每窗 behavior hash / modified flag / identity-equivalent flag；`program_supply.py:299` `verify_program_supply_alternatives` 只用 `executor.verify()`，按 `checked_windows>0`、`modified_windows>0`、非 identity-equivalent、两选项行为不同四门过滤；`program_supply.py:499` `route_verified_program_supply_fault` 挣得 `PROVEN_EXPRESSIBLE` 后走同一五字段路由 | `tests/methods/test_program_supply_evidence.py:81`（零窗不通过）；`:93`（identity-equivalent 不通过）；`:105`（两选项行为不同才 choice）；`:121`（同行为不是选择）；`:135`（路由 Unknown → SKILL_LIBRARY_GAP）；`:169`（program-aware skill 检查忽略无关 imputation skill）；`:198`（部分窗口哈希重叠 ≠ 行为等价；ordered tuple 全同才等价）；`:234`（真实 ScopeExecutor 受控格，`evaluate` 调用 0） |

| E-2 | `method.py:52` `_group_add_route_error` + `method.py:649` `handle_group_feedback`：接收 `route_decision` + `surface_catalog`，仅 `SKILL_LIBRARY_GAP/EDITABLE_M0/单 capability ADD` 才构造 manifest，否则 `route_not_add_only`；proposed 非 ADD → `proposal_not_add_only`；`evidence_compiler=False`、options≥2、且调用方传入的 E-1 `verified_choice_offered` 为真（未提供时兼容旧受控路径）才记 `choice_offered`，否则记 `no_choice_offered`；controller 用 route_decision.cause_code 授权 | `tests/integration/test_e2_group_route_gate.py:84`（签名无 confirmed_cause，route_decision/surface_catalog 必填）；`:91`（错误路由拒绝且 Slow 调用 0）；`:121`（合法 ADD 路由到达 Slow + choice flags）；`:152`（两选项但 E-1 行为同质 → no_choice_offered）；`:183`（单选项 no_choice_offered） |

| E-3 | `group_fault.py:100` `_provenance_of` + `group_fault.py:116` `build_contrast_capsule(target_domain_namespace=...)`：contrast 增加 `negative` 桶；ref 与 per-episode 行带 source/target/unknown provenance；capsule 输出 `source_provenance.source_episode_ids`、`referenced_source_episode_ids` 与 `filtered_source_episode_ids/count` | `tests/methods/test_group_fault_contrast_provenance.py:40`（negative 桶 + provenance + referenced/filtered Source 计数）；`:96`（support 正 / delayed 负按 relation=CONFLICT 入 conflict，不进 positive）；`:121`（未提供 target domain 时如实 unknown） |
| P4-BINDING | `methods/ttha/p4_runner.py` `run_p4_group_update`：Card 只构建一次 → 同一 Card 交给 `route_verified_program_supply_fault` → `bind_verified_program_options` 按 `(patch_id, exact ordered program_steps)` 绑定并回写 verified steps 到 Card → 过滤后的同一 Card 交给 Slow；同名不同 steps / 重复 patch_id / Slow 选 unverified ID 都在 Slow 前失败；`verified_choice_offered` 默认 false，不按数量退化。`online_loop.py` 自动 group 路径仅保留给旧开发流程，P4 不使用 | `tests/integration/test_p4_runner_binding.py`：Card 构建一次 + Slow 看到 exact verified steps；同 ID 不同 steps → `program_binding_mismatch`（Slow 前）；重复 patch_id → `duplicate_card_patch_id`；Slow 选 unverified ID → `verified_patch_binding_failed`；无 verified alternative → Slow 调用 0；`verified_choice_offered=False` 严格记 `no_choice_offered` |

| P4-RUN | `evaluation/functional/run_p4_natural_add_only_slice.py` 一次完整自然切片：同一 materialized h0 fork STATIC/A3/A5；batch1 {600,792,888,984} 批内 Slow=0/group Slow=0；批末 E-1 → `run_p4_group_update` ≤1/臂；A3/A5 都路由到 `SKILL_LIBRARY_GAP` 且真实 Slow（gpt-5.6-luna）选择 `outlier_mad`；组内 replay 888=+0.1199、984=−0.0608 → `group_replay_rejected`；delayed 未打开；batch2 snapshot 保持 baseline。**窄结论**：机械链跑通 + replay 拒绝正确；Capsule 未进入 Slow Card，不能解释为反馈驱动的 Capability 负结果 | `artifacts/functional/e2/w1_p4_natural_add_only_slice_report.json` |

| P5-PRE | 正式 P5 前必须修正 Memory 检索语义：选择 replacement 时返回各候选 Program 的成功/失败/冲突经验；当前只检索与失败 workflow 相同 Episode，会过滤掉 `outlier_mad` Source 证据 | 暂不实现；记入 P5 前置 |

| P4-HEADROOM | `evaluation/functional/run_p4_headroom_2x2.py` 零 LLM 核查：`outlier_mad`=+0.1199/−0.0608，`hampel_filter`=−0.2294/+0.0556；两者都不能两格 ≥+M。verdict=`PER_CONTEXT_HEADROOM_EXISTS_NO_COMMON_PROGRAM`：每个 Context 各有解，但没有共同 Program；不是整个 Program family 无 headroom | `artifacts/functional/e2/w1_p4_headroom_2x2_report.json` |

| S1a | 同一 P4 headroom Runner 内零 LLM 穷举 frozen-bin 单特征谓词：9 个 numeric observable 在 888/984 的 bin 标签完全一致（`separating_features=[]`）→ `SCOPE_HYPOTHESIS_NOT_EXPRESSIBLE_IN_BINS`。裸浮点阈值被禁止；四个 probe_direction 特征已排除 | 同上报告 `scope_hypothesis_probe` 段 |

| FIX-1 | `run_p4_natural_add_only_slice.py` `_card_for_group` 现把 `contrast_capsule` 写入 `failure_pattern_card.facts`，不再由 `lambda _g, _c: card` 丢弃 Capsule | 代码编译 + 后续 P4v2 前置已就绪 |

| FIX-2 | `group_fault.build_contrast_capsule` 增加 `candidate_workflows`：replacement 决策按「incumbent ∪ candidates」检索，不再只按失败 workflow 过滤；报告含 `retrieval_scope` | `tests/methods/test_group_fault_contrast_provenance.py`：candidate-conditioned Source Episode 进入 `referenced_source_episode_ids`，过滤数为 0 |



**本记录声称的边界**：上表只证明两条 Runtime 机械链路闭合；**不声称自然 Slow Evolution 成立**。

**未改动**：`slow_edit_v1` schema、`fault_routes.json`、`methods/ttha/harness/h0` 内容均未按 P0–P3 修改。

**执行环境注记**：本仓库工作区在开始前已存在未提交的 h0/其他文件改动，导致
`compile_snapshot(..., verify_lock=True)` 基线测试报 `snapshot lock mismatch`；
P0–P3 新测试使用 `verify_lock=False` 编译，不依赖/不改写 h0 lock。
此外 `tests/architecture/test_ttha_dependency_rules.py` 仍会命中一处**开始前已存在**的
`methods/ttha/slow_agent.py:220` 本地导入 `SurfaceRegistry`（methods → evaluation/minipipe）；
本记录的新模块没有新增 methods→evaluation 依赖。

**密钥安全提醒（审核）**：P4 Runner 按既有约定把 `OPENAI_API_KEY` 用于
`https://api.agicto.cn/v1`。若该值实为 OpenAI 官方密钥，它已被发送给第三方端点，
必须立即轮换；第三方端点应只使用独立 `AGICTO_API_KEY`。


---

## 附：S1a′ / S1b 预注册草案（待主控审核，2026-08-16）

**状态定性**：S1a 原结论改名
`PREFIX_OBSERVATION_SCOPE_NOT_EXPRESSIBLE_IN_FROZEN_BINS`。
它只证明全历史前缀 bins 不可分；不证明 Scope 不可表达，也不支持直接进 S4。

### 目标

先用零 LLM 验证一个候选近期 Observation 是否在 **K1 cohort × origin** 尺度上具有
可泛化的 Scope 价值。不验证时，不写任何 Slow Card、不发 LLM、不进 S4。

### 新 Observation（唯一新增，最小变更）

- 字段名：`recent_window_robust_z_peak`
- 语义：取当前可见历史**最后 192 点**，用现有 `runtime/public_features.py` 的同一
  `robust_z` 计算取峰值
- 类型：`number`
- bin：继续复用现有五档 frozen edges；**不新增 bin 体系**
- 不改 `local_robust_z_peak` 的旧语义，不加全局 192 长度断言

候选假设（冻结）：

```text
recent_window_robust_z_peak == high
    -> hampel_filter

recent_window_robust_z_peak in {zero, very_low, low, medium}
    -> outlier_mad
```

### 决策单位与验证网格

- 决策单位 = **K1 cohort × origin**，不是 12 条 train series 各自一个 gain
- `888/984` 只做 discovery，绝不进入验证
- 冻结验证 origins：

```text
1176, 1224, 1272, 1648, 1672, 1720
```

- 选择依据只用了 feature bin 分布，未打开这些 origin 的候选 gain
- 两侧分支必须各命中 ≥2 个验证 origin；不足 → `SCOPE_PREDICATE_DEGENERATE_ON_TARGET`，
  停，不换 roster

### S1b（零 LLM）

比较：

| 策略 | 内容 |
|---|---|
| B1 | 所有验证 origin 一律 `outlier_mad` |
| B2 | 所有验证 origin 一律 `hampel_filter` |
| S | 按上述 recent-observation 规则绑定 |
| B0 | 报告用：一律 `winsorize`，不参与通过判定 |

通过条件：

1. `origin_macro_gain(S) > max(origin_macro_gain(B1), origin_macro_gain(B2))`
2. `harm_origin_count(S) <= harm_origin_count(best_fixed_by_gain)`
3. `harm_magnitude(S) <= harm_magnitude(best_fixed_by_gain)`
4. 每个验证 origin 恰好命中一个分支；两条谓词不重叠、不留空

任一不满足 → `CONTEXT_SCOPE_HYPOTHESIS_NOT_CONFIRMED`，不发 LLM，然后才允许进入 S4。

### S1c（只有 S1b 通过后才做）

1. 先补最小 Runtime-owned `SPLIT_SCOPE` 编译：把 Slow 输出的「一个公开特征 +
   一个 bin 阈值 + 两个 Program 绑定」编译成两条互补 `in` 谓词 SkillEntry。
2. Slow 单次提案，不重试、不调阈值。
3. 机械断言：验证集每格必须恰好匹配一条 Skill；否则
   `SCOPE_TIEBREAK_AMBIGUOUS`，停止。
4. 成功标签：`NATURAL_CONTEXT_SPLIT_REBIND_DEV_PASS`；
   失败标签：`CONTEXT_SCOPE_UNIDENTIFIABLE`。

### 三条机械断言（写入 runner 样板，不写进全局 feature extractor）

1. `recent_window_robust_z_peak` 只能由最后 192 点计算，旧特征语义不动。
2. 任何声称臂有差异的 LLM 前，比较规范化决策承重输入：
   `typed options + 正/负/冲突 evidence + candidate-conditioned Source evidence`。
   全等 → `P4_ARM_DISTINCTION_INERT`，不发 LLM。不比较 provenance/ID 等非决策字段。
3. 两条 Scope Skill 在验证集每格必须 exactly-one-match，禁止 `skill_id` 静默 tie-break。

### 前置与并行状态

- FIX-1 capsule 接 Card：已实现
- FIX-2 candidate-conditioned retrieval：已实现并有测试
- S1b 通过前：不实现 `SPLIT_SCOPE`，不发 Slow LLM
- S1b 失败：进入 S4，但保留 FIX-1/FIX-2
- P5 仍不启动


---

## 附：D0 结果与修正后的 S1b 计划（取代上一版 S1b 草案，2026-08-16）

**上一版 S1b 作废**，原因两个阻塞均成立：

- A：feature 测在 T117（train，不进 eval loss）；当前最近 192 点也不进训练。
- B：验证 origins 的训练 anchors 在 ≥900 后完全饱和，训练因果面被冻结。

### D0 已执行（零 LLM）

在 `origin=984` 上使用与 888 相同的 **9 个 anchors**（排除 852），独立新 executor：

```text
outlier_mad   = -0.048858
hampel_filter = +0.010657
```

翻转仍在，因此：

```text
D0_VERDICT = EVAL_CONTEXT_SUFFICIENT
```

即 eval-context 变化足以造成反对角线。训练集新增窗口不是翻转的必要条件。
（D0 不排除 train×eval 交互，但允许进入 D1。）

### D1：把 Observation 迁到 eval 侧

- 不先改全局 contract。
- Runner 内计算 `eval_recent_high_z_count`：
  - 对 8 条 eval/support/query series，各取最后 192 点；
  - 用现有 `local_robust_z_peak` 计算；
  - `>= 6` 记 high；
  - cohort 聚合规则：`count >= 4 -> hampel_filter`，否则 `outlier_mad`。
- 888/984 discovery 读数：888 = 0/8 high，984 = 5/8 high。

### D2：重新冻结验证 origins

验证网格（已按审核改为时间交错，outcome-blind 冻结）：

```text
1056 -> outlier_mad    (high_z_count=2)
1344 -> hampel_filter  (high_z_count=5)
1776 -> outlier_mad    (high_z_count=2)
2712 -> hampel_filter  (high_z_count=4)
3168 -> outlier_mad    (high_z_count=0)
3408 -> hampel_filter  (high_z_count=7)
```

- 全部 origin % 24 == 0
- `[origin-192, origin+48)` 区间两两不重叠，origin 间隔 ≥ 240
- low/high 时间交错，避免 branch 与“早晚时段”完全混淆
- 两侧各 3 个 origin
- 只按 `eval_recent_high_z_count` 冻结，尚未打开这些 origin 的候选 gain
- 888/984 不进入验证

### D3：S1b 零 LLM

比较：

```text
B1  always outlier_mad
B2  always hampel_filter
S   eval_recent_high_z_count >= 4 -> hampel，否则 outlier
```

同时报告 B0（always winsorize），不参与通过判定。

通过条件：

```text
utility_pass:
  macro_gain(S) >= macro_gain(best_fixed) + M
  harm_origin_count(S) <= harm_origin_count(best_fixed)
  harm_magnitude(S) <= harm_magnitude(best_fixed)

promotion_feasible:
  S 的每个验证 origin 的 selected_gain >= M
  （逐 origin 合取，与现有 Gate 同语义）
```

- 两项都通过 → 允许进入 S1c。
- 只有 `utility_pass` → `SCOPE_UTILITY_PASS_PROMOTION_INCOMPATIBLE`，
  不发 Slow，也不许进 S4。
- 任一失败 → `CONTEXT_SCOPE_HYPOTHESIS_NOT_CONFIRMED`，再考虑 S4。

**D3 v3 已执行结果**：冻结网格 `{1104,1368,1800,2856,3648,3888}`。
Scope policy `macro_gain=-0.01056`、harm origins=4、promotion 4/6 fail；
best fixed = always `outlier_mad`，`macro_gain=+0.05696`、harm origins=2。
`utility_pass=False`、`promotion_feasible=False`。
结论：`eval_recent_high_z_count>=4` 这一候选近期观测在验证网格上不能预测
当前 eval 点的更好 Program；S1b 拒绝，不发 Slow、不建 SPLIT_SCOPE。
该结果只否定这一个冻结的近期聚合规则，不声称所有 Scope 谓词都不可能。

**existing-program oracle（同一 D3 六格）**：逐格选两个现有 Program 的更优者，
`query oracle macro_gain=+0.13674`、`harm=0`、`6/6 >= M`；
best fixed = always `outlier_mad`，`query macro=+0.06827`。
因此**现有 Program 已覆盖全部六格，不存在需要 S4 的 Program headroom 缺口**。

**D4（零 LLM，support-conditioned selection）**：用 4 条 support series 的 gain
选 Program，再用 4 条 query series 评价。结果
`macro_gain=-0.01447`、`harm_origins=2`、`abstain=2`，**显著差于 best fixed**。
support 与 query 在 1368/1800/3648/3888 等格上方向相反，说明当前
support-series 聚合反馈不能可靠替代 query 表现。`S4_NOT_AUTHORIZED`；
首阻塞仍为 selection/feedback resolution。

**D5（零 LLM，已执行）**：
- 8 series × 6 origins × 2 programs per-series gain 矩阵已写入报告。
- 当前 frozen 4/4 划分 rank=31/70；mean policy macro=−0.01447。
- leave-one-support-out：删掉 `T13` 后 macro 从 −0.01447 变为 +0.04486，
  且 abstain 从 2 降为 1；`T13` 是当前唯一高杠杆 support series。
- 70 个有方向 4/4 划分中只有 3 个（4.29%）超过 best fixed；
  所有 top partitions 仍低于其 query best-fixed oracle。
- median / majority 仅作辅助诊断，不部署。
- verdict=`SMALL_SUPPORT_SET_TRANSFER_UNSTABLE`：cohort 级标量 support
  反馈分辨率不足；下一步方向是 series/context-matched feedback，例如
  同 series 早期 Support → 后期 delayed。不得在这批已打开数据上挑
  best partition 并宣称成功。

**主张边界**：D3 能证明的是

```text
observable-conditioned program selection
（公开观测能否预测两个固定训练模型在当前 eval 点谁更好）
```

不是完整的 `context-adaptive repair`。论文口径不得合并。

### S1c（只有 D3 两项通过）

1. 最小 Runtime-owned `SPLIT_SCOPE`：Slow 输出一个公开特征 + 一个 bin 阈值 +
   两个 Program 绑定，编译成两条互补 `in` 谓词 SkillEntry。
2. 单次 Slow 提案，不重试、不调阈值。
3. 每条验证 origin 必须恰好匹配一条 Skill；否则
   `SCOPE_TIEBREAK_AMBIGUOUS`。
4. S1c 只是机制验证：Slow 能否把 D3 已验证的 Scope 写成可安装规则。
   失败按位置记为：
   - `SLOW_SCOPE_PROPOSAL_FAILED`
   - 或 `SPLIT_SCOPE_RUNTIME_BINDING_FAILED`
   D3 的验证 gain 不得再放进 Slow 输入，也不得在同一批已打开 origins 上重复领取效用。
   成功标签：`NATURAL_CONTEXT_SPLIT_REBIND_DEV_PASS`。

### 状态

```text
D0_PASS_EVAL_CONTEXT_SUFFICIENT
D1_COMPLETE_EVAL_SIDE_AGGREGATION_FROZEN
D2_V3_COMPLETE_TIME_MATCHED_OUTCOME_BLIND_GRID_FROZEN
D3_VALID_NEGATIVE_FOR_HIGH_Z_SCOPE
EXISTING_PROGRAM_HEADROOM_PRESENT
D4_VALID_NEGATIVE_FOR_FIXED_SUPPORT_MEAN
D5_COMPLETE_SMALL_SUPPORT_SET_TRANSFER_UNSTABLE
D5_HIGH_LEVERAGE_SUPPORT_SERIES_T13
FIRST_BLOCKER_FEEDBACK_TRANSPORTABILITY_RESOLUTION
S4_NOT_AUTHORIZED
S1c_NOT_AUTHORIZED
P5_BLOCKED
NEXT_SERIES_OR_CONTEXT_MATCHED_FEEDBACK
```

FIX-1 / FIX-2 仍独立有效。P5 不启动。

---

## 附：D5 外部审核修正 + 888/984 匹配 9-anchor 分辨率审计（2026-08-16；只追加）

### D5 审核后的收窄（覆盖上一节 D5 解释，不改 D5 数据）

- `D5_VALID_FIXED_4X4_SUPPORT_TRANSFER_UNSTABLE` 成立，但只否定固定的
  4/4 Support→Query 聚合方案；`−0.0145 vs +0.0683` 同时表明该方案存在系统性偏差，
  不是只有方差。
- `D5_HIGH_LEVERAGE_SUPPORT_SERIES_T13` **不进入状态**。四条 LOSO 的符号变化是
  n=4 且均值贴近零时的算术；T13 按离散度排第 3（T131 更高）。T13 只作诊断记录，
  不得从 roster 删除。
- 同 series carry-forward `+0.0318 <` 固定 `outlier_mad +0.0728`，符号预测
  16/40=40%（n=40，SE≈7.9%）。`NAIVE_SPARSE_ORIGIN_CARRY_FORWARD_NOT_SUPPORTED`：
  关闭「稀疏相邻 origin 赢家直接沿用」；不关闭所有 series/context-matched Memory。
- 六个验证 origin 的全部 4/4 split-half 赢家一致率合计 192/420=45.7%，只有 1368
  可分辨。因此 `PER_ORIGIN_PROGRAM_PREFERENCE_UNRESOLVED_AT_N8`，不写
  「标签是纯噪声」。
- query oracle `+0.13674` 只作为这 4 条 query series 上的事后描述上界；
  `IN_SAMPLE_ORACLE_HEADROOM_DESCRIPTIVE_ONLY`，不再称为可实现/可预测的 routing
  headroom。`EXISTING_PROGRAM_HEADROOM_PRESENT` 从状态表撤回。
- 曝光账：D5 已打开 8 series × 6 origins × 2 programs = 96 个 gain cell。
  `K1_SIX_ORIGIN_EVAL_FULLY_EXPOSED_96_CELLS`；这些格此后只能做 development
  diagnosis，不能再承担 held-out claim。
- `S4_NOT_AUTHORIZED` / `S1c_NOT_AUTHORIZED` 理由改为：当前测量精度下
  「每个 origin 哪个 Program 更好」无法稳定判定，不是「还没试够条件信号」。

### 888/984 匹配 9-anchor 分辨率审计（零 LLM，已执行）

Runner：`evaluation/functional/run_p4_888_984_resolution_audit.py`
Report：`artifacts/functional/e2/w1_p4_888_984_resolution_audit_report.json`

- 两个 origin 共用同一 9-anchor 训练配置 `[312..792 step 60]`（排除 852）。
- 对 8 条 eval series（4 support + 4 query）逐 series 计算两个 Program 的 gain，
  并枚举每个 origin 的全部 C(8,4)=70 个 4/4 split-half 赢家比较。
- 不加特征、不改 Gate、不发 Slow；888/984 原已曝光，32 个 per-series cell
  只作 development diagnosis，不承担 held-out claim。

| origin | cohort om | cohort hampel | cohort 赢家 | om−hampel mean (sd) | \|mean\|/SE | split-half 一致 | 半区赢家计数 | 逐 series 赢家 |
|---|---|---|---|---|---|---|---|---|
| 888 | +0.119861 | −0.229392 | outlier_mad | +0.349254 (0.331747) | 2.98 | 70/70 = 100% | om 140 | om 7 / hampel 1 |
| 984 | −0.048858 | +0.010657 | hampel_filter | −0.059515 (0.218881) | 0.77 | 34/70 = 48.6% | hampel 104 / om 36 | om 5 / hampel 3 |

解释：

- 888 的标签稳定：全部 70 个 4/4 半区的赢家都是 `outlier_mad`。
- 984 的标签在 n=8 下不稳定：两 Program 差距小，70 个 split-half 只有 34 个
  两半一致（48.6% 仅作描述性稳定性；70 个 split-half 高度共享同一批 series，
  不是独立伯努利单位，因此不做 formal binomial 检验）；cohort 级
  `|mean|/SE = 0.77`。
- 984 的 cohort 级方向确实与 888 相反，但「反对角线」要求两个 decision point
  都先有稳定标签。其中一个不稳定 → `INITIAL_ANTI_DIAGONAL_NOT_ESTABLISHED_AT_N8`。

### 分支结果

任一 origin 不稳定 → **关闭当前 K1/n=8/per-origin Program selection 支线**。
不继续换特征、不造 Workflow、不进 S4/S1c/P5。固定 Program 结论的措辞收窄为：
D3 v3 六个 development origins 上 always `outlier_mad` 的
`macro_gain=+0.05696`（≈+0.057）**只表示这六个已打开 origin 上的宏平均为正**；
其中仍有 2 个 harm origin，不能称为已激活的安全 Skill，也不支持 per-origin 选择。

下一步二选一，均需主控决策：

1. 申请扩大 KDD cohort（series_cache.npz 有 270 条，K1 只用 20 条）以提高反馈
   分辨率，但必须先过新序列曝光审批；
2. 按现有证据收口：写「固定 outlier_mad 有正宏增益；在 8 条 eval series 规模下
   per-origin Program 选择不可测量」这一负结果，不申请新曝光。

（上述二选一已被下一节主控建议取代：不扩大 cohort 挽救 per-origin Router，
主线改为多-origin Target-local Skill 估计单位。）

### 状态（D5 审核 + 888/984 审计后）

```text
D5_VALID_FIXED_4X4_SUPPORT_TRANSFER_UNSTABLE
NAIVE_SPARSE_ORIGIN_CARRY_FORWARD_NOT_SUPPORTED
PER_ORIGIN_PROGRAM_PREFERENCE_UNRESOLVED_AT_N8
K1_SIX_ORIGIN_EVAL_FULLY_EXPOSED_96_CELLS
IN_SAMPLE_ORACLE_HEADROOM_DESCRIPTIVE_ONLY
FIXED_OUTLIER_MAD_MACRO_GAIN_POSITIVE_SIX_DEV_ORIGINS_ONLY
S4_NOT_AUTHORIZED
S1c_NOT_AUTHORIZED
P5_BLOCKED
AUDIT_888_984_MATCHED_TRAIN_SPLIT_HALF_COMPLETE
ORIGIN_888_PROGRAM_LABEL_STABLE
ORIGIN_984_PROGRAM_LABEL_UNSTABLE_AT_N8
INITIAL_ANTI_DIAGONAL_NOT_ESTABLISHED_AT_N8
K1_N8_PER_ORIGIN_PROGRAM_SELECTION_CLOSED
NEXT_DECIDE_COHORT_EXPANSION_VS_NEGATIVE_RESULT_CLOSEOUT
```

FIX-1 / FIX-2 仍独立有效。P5 不启动。

---

## 附：主线采纳：多-origin Target-local Skill（M0 → M1 → M2，2026-08-16；只追加）

### 方向决定（主控建议已采纳）

- `K1_N8_PER_ORIGIN_PROGRAM_SELECTION_CLOSED` 正式关闭。
- 不扩大 cohort 挽救 per-origin Router；cohort expansion 仅作为未来
  「interval-level routing 是否可测」的研究支线，当前不作为主线。
- 估计单位改回多-origin Target-local Skill：不再预测「这一格该选哪个」，
  而是学习「这个 Target Domain 当前阶段总体应保留哪个 Workflow」。
- 版本纪律：M0 是新版本估计量。P4、D3–D5、888/984 审计的旧结论
  **不追溯改写**，只在新版本记录中继续。

### M0：多-origin 估计单位承重检查（零 LLM、零新 Outcome）

- 数据：直接复用 D5 已打开的 8 series × 6 origins × 2 programs gain 矩阵，
  不产生新 Outcome cell。
- 估计量（新版本）：

```text
ḡ(s, p) = (1 / K) * Σ_{k in K} gain(s, k, p),  K = {1104, 1368, 1800, 2856, 3648, 3888}
```

- 报告项：
  1. cohort macro gain = `mean_s ḡ(s, p)`；
  2. 8 条 series 中 `ḡ(s, p)` 的正/负数量；
  3. origin 方向一致率：先算
     `d_k = mean_s [gain(s,k,outlier_mad) − gain(s,k,hampel_filter)]`，
     再取 `max(count(d_k>0), count(d_k<0)) / K`；
  4. harm count 与 magnitude（M=0.005 冻结；主口径用 `ḡ(s,p) < −M`，
     附 origin-level `mean_s gain(s,k,p) < −M` 作旧口径对照）；
  5. `outlier_mad` / `hampel_filter` / identity 对比（identity gain=0
     作参照）。
- 回答的问题：在单个窗口标签不稳定时，同一 Target Domain 上跨多个已发生窗口
  累计反馈，能否稳定识别一个值得形成 Target-local Skill 的 Program。
- 分支：
  - 至少一个 Program 在多-origin 单位上稳定为正 → 进入 M1 自然
    Target-local Skill 切片；
  - 两个 Program 都不稳定 → 关闭当前 outlier capability family，换一个
    有更清晰自然 Program headroom 的 family；不为救结果而立即组合新 Workflow。
- 不调用 Slow；不加特征；不改 Gate。

### M1：自然 Target-local Skill 切片（M0 通过后才执行）

- 使用新的时间段：K≥4 Target Support origins → 形成 `LOCAL_DRAFT` →
  独立的 K≥4 later origins → delayed 更新或撤销。
- 保持：Forecasting、Consumer、Program pool 不变；`outlier_mad` /
  `hampel_filter` 至多两个候选；Support 与 delayed origins 不重叠；
  每次合法结果立即写 positive/negative/conflict Episode；Query future 保持 sealed。
- 调用 Slow 前必须零 LLM 确认四项：
  1. Support 上确实存在多-origin Program headroom；
  2. A3/A5 的规范化 Slow 输入确实不同；
  3. A5 的 candidate-conditioned Source evidence 非空；
  4. 两个 verified Program option 均真实可执行。
  任一失败都不调用 LLM。

### M2：恢复核心 A5 vs A3（M1 形成并延迟核销 Target-local Skill 之后）

- A3：空 Source Experience；A5：Source positive/negative/conflict Experience；
  Target Support/feedback 预算完全相同。
- 主指标：首次形成有效 Target-local Skill 所需 probes、累计 harm、abstention、
  delayed macro utility、Source Experience 是否缩短冷启动。

### 暂缓事项

- S1c 暂缓；S4 暂缓；per-origin Scope/Pattern 暂缓；cohort expansion 暂缓。
- P5 等待 M1。

### 状态（主线采纳后；M0 为下一步，尚未执行）

```text
D5_VALID_FIXED_4X4_SUPPORT_TRANSFER_UNSTABLE
NAIVE_SPARSE_ORIGIN_CARRY_FORWARD_NOT_SUPPORTED
PER_ORIGIN_PROGRAM_PREFERENCE_UNRESOLVED_AT_N8
K1_SIX_ORIGIN_EVAL_FULLY_EXPOSED_96_CELLS
IN_SAMPLE_ORACLE_HEADROOM_DESCRIPTIVE_ONLY
FIXED_OUTLIER_MAD_MACRO_GAIN_POSITIVE_SIX_DEV_ORIGINS_ONLY
S4_NOT_AUTHORIZED
S1c_NOT_AUTHORIZED
P5_BLOCKED
AUDIT_888_984_MATCHED_TRAIN_SPLIT_HALF_COMPLETE
ORIGIN_888_PROGRAM_LABEL_STABLE
ORIGIN_984_PROGRAM_LABEL_UNSTABLE_AT_N8
INITIAL_ANTI_DIAGONAL_NOT_ESTABLISHED_AT_N8
K1_N8_PER_ORIGIN_PROGRAM_SELECTION_CLOSED
PER_ORIGIN_COHORT_EXPANSION_DEFERRED
NEXT_MULTI_ORIGIN_TARGET_LOCAL_ESTIMAND_BEARING
```

FIX-1 / FIX-2 仍独立有效。P5 等待 M1，不启动。
