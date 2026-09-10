# DEV-KNOW-1 — 前段积累的 Skill，在相同后续预算下是否让 Agent 做得更好

*2026-09-07 · development 级 · 形成段 u5–u15 × 1 条轨迹 + 后续段 u16/u17/u23/u24/u25 × 2 臂 × 两组，全部跑完 ·
判词 `KNOWLEDGE_WAS_RETAINED_AND_READ__AND_NEVER_SELECTED`*

工件 `artifacts/main_protocol/{dev_know1_two_arms__g1,dev_know1_two_arms__g2,dev_know1_contrasts}.json`
源 `evaluation/main_protocol_p4/{dev_know1_retention,smoke_dev_know1,run_dev_know1_two_arms,audit_dev_know1_contrasts}.py`
预检 12 项全过（含 DEV-AUTO-2 Frame C 预检整体复用），0 LLM / 0 fits。
**本包不 PATCH 任何卡，不改任何核心文件，未提交 git。**

---

## 0. 一句话答案

> **在相同的后续适应预算下，前段积累的合法 Skill 是否让 Agent 更好地选择、修改和验证程序？**

**知识确实形成并被保留下来了；后续段 5/5 格都把它取了出来、其中 2 格真的拿去实测了；一次也没有被选中部署。
两臂交付的策略逐格相同，delayed 与 Support 面差值都恰好 `0.000000`。**

卡片没死、能被检索、能进候选菜单、能被实测——**卡在"选择"这一层**。
所以收口规则落在「知识未被使用」这一条：**去定位使用入口（选择层），不要再加卡片，也不要再改保存方式。**

同时本包顺带量到一件以前没有的东西：**两个机械上完全相同的臂之间的运行间噪声**
（g1 无处理组）= Support **+0.016353**、delayed **+0.000307**。
处理组测到的 0.000000 比这条噪声线还小 ⇒ 本设计以 n=1 个处理组分辨不出小于约 0.016（Support 面）的效应。

> 以上是 **deepseek-chat** 传输下的正式结果，是本报告的主结论。**附录 A** 记录了同一套机制
> 换到 **gpt-5.6-sol** 传输下独立跑的两组结果（不与本节相减、不做跨模型主张）：两组都形成了
> 同一张卡，5/5 格检索到，但这次交付**确实系统性地变了、而且是负的**——同样卡在选择层，
> 只是症状从"从不选"变成了"选了会挤掉旧程序覆盖到的人群"。

---

## 1. 第一段：最小知识保留修补

### 1.1 复用而不是重建

以 DEV-AUTO-3 的 **B 路线**为基础：Fast 本地适应 + 外环提议与候选供给 + 正常 Fast 铸卡生命周期，
**全程不 PATCH 任何卡**。所谓"新程序成为独立卡"这件事，现成的铸卡路径本来就是这样做的：

```
method.handle_fast_winner  →  EditOperation.ADD, surface_precondition={"kind":"ABSENT"}
                           →  两阶段 delayed 批准
                           →  online_loop.activate_approved
```

DEV-AUTO-3 的 B 臂就是走这条路铸出 `fast_winner_forecast_ridge_smase_outlier_iqr` 的。
所以本包**没有新造铸卡路线**，只补了这条路线原本缺的东西。

### 1.2 五条要求，逐条对照

| # | 要求 | 现状 | 本包做了什么 |
|---|---|---|---|
| 1 | 新程序验证通过后成为独立卡，不覆盖父卡 | 已实现（ADD，不是 PATCH） | 复用；并在预检里**实跑**验证（见 1.3） |
| 2 | 子卡失败只影响对应版本，父卡按自身合法状态 | 已实现（`revoke_deployed_skill` 只 unlink 匹配 `skill_id` 的那一个 json） | 复用；预检实跑验证 |
| 3 | 不复活已撤销祖先、不清零验证次数、祖先不绕过当前 Support | 已实现 | 分叉时 Draft **关闭而不是删除**；ledger 整份深拷贝给两臂；`open_restricted` 拒绝重开已关闭血缘；铸出的卡保留 `requires_target_support` |
| 4 | 同一程序×适用范围不重复建卡 | 已实现（id 由任务范围＋winner workflow signature 决定，ADD 的 `ABSENT` 前置挡住同名） | 不新增规则；把它作为**读数**（`duplicate_capability_card`）报出来。**g1 里这条被真实触发**（见 4.1） |
| 5 | 当前卡不存在时如实报告，不拿历史祖先冒充 | **未实现**——`R.current_skill_steps` 在卡缺失时 `return tuple(ANCESTOR_PROGRAM)` | **新增**：`keep.current_card` 如实报告不存在；`revision_step` 包一层，卡不可用就以 `NO_CURRENT_CARD` 结束该步，绝不落到那条回退 |

### 1.3 只做了两个必要行为检查（按任务书），都是实跑

不是读源码字符串，是拿真实 store + 编译器跑出来的：

* **父子可以共存** — 在 K0 真实快照上 ADD 一张子卡：库从 1 张 capability 卡变 2 张，
  父卡 body / serving_scope / risk_guards 逐字段不变，其余卡不变，子卡带 `requires_target_support`。
* **撤销子卡不会误删父卡** — 走 `revoke_deployed_skill` 那套 fork → unlink → `compile_snapshot` → materialize：
  **恰好 1 个 entry 文件被 unlink**，父卡 body 逐字节相同，库回到子卡出现前的成员集合。

第三项（同程序不重复建卡）用真实 controller 跑了一次二次 ADD，被
`AddTargetExistsError: ADD target already exists` 挡下——这是既有机制，不是新加的。

### 1.4 一件试过又撤回的事（边界事实，必须记）

我先在 `methods/ttha/method.py` 的 `handle_fast_winner` 入口加过一道显式"重复卡拒绝"守卫。
**撤回了**：`method_contract` 是 harness 的 dependency sha 之一，改那个文件会改掉**每一个快照的
`runtime_bundle_sha`**，锁定的 h0 立刻失配（`tests/methods/test_ttha_h0.py` 4 项由全过变 3 失败）。
那等于新开一个哈希平台，本包无权这么做。已 `git checkout` 还原，h0 测试恢复全过，
**本包最终没有编辑任何核心文件**。要求 4 本来就由既有机制满足，不需要这道守卫。

---

## 2. 第二段：形成段与后续测试段（运行前写定）

* **形成段**：u5–u15（11 个单元），**每组只跑一条自主积累轨迹**。
* **后续测试段**：u16、u17、u23、u24、u25（5 个单元）。
* u18–u22 由 exposure 名字排除（`TARGET_HELD_IN`），密封数据与 +144 评估面全程不读。

预检机械验证：两段不重叠、并集**恰好等于**边界之后的全部 development 单元
`[5..17, 23, 24, 25]`、被排除的恰好是 `[18..22]`、测试段与被排除集合无交。
切分依据是课程自身的序列组结构，**不按任何程序的成绩挑选**。
这是"后续课程情境"的开发测试，不是全局时间递增证明，也不是独立泛化证明。

---

## 3. 分叉：两臂差在哪一件事上

共同边界 = 形成段结束时的状态。两臂都从这里**新建会话**出发：

* **B 累积知识** — 起点 = 边界快照（含形成段依法铸出的新卡）。
* **A 旧知识** — 起点 = **边界快照减去形成段新增的卡**，用的就是撤销那套 store 往返。
  **不是入场快照**：形成段若合法丢掉过一张卡，两臂一起丢，**不为了让对照好看而复活被撤销的祖先**。
  （本次两组 `lost_during_the_formation_segment` 都是空。）

对称性逐条落实，且都在预检里检查：

| 项 | 处理 |
|---|---|
| 模型 / 工具 / DSL / 编辑能力 / 准入规则 / 接受框架 / 验证规则 | 同一套，`ArmSpec(name,"k0",True,False,True)` 只差名字 |
| 反馈预算 | 每个臂次 `1000//6 = 166` 次物理 LLM；每格帽 = 该臂剩余包级额度，**没有恢复每格小调用帽** |
| **A 不是 identity 对照** | A 保留全部外环：能搜索、能提议、能被实测、能验证、能自己铸卡 |
| 前段未验证 Draft 候选供给池 | **两臂都不带**。分叉时在**两臂共同拷贝的那一份 ledger 上**把所有未关闭 Draft 关闭（`NOT_CARRIED_ACROSS_THE_FORK`），`resupplied_programs_for_verification()` 两边同时为空 |
| 前段原始 Episode | **不进 Fast**。两臂的 method 都由 `TTHAMethod(fast, start_snapshot, ())` 重建，经验列表为空 ⇒ 前段知识到后段的**唯一通道就是 Skill 卡** |
| 历史风险 / 撤销 / 计数记录 | 整份深拷贝给两臂，一条不删。关闭≠删除，计数、状态、风险记录逐字段保留；已关闭血缘两臂都不能重开 |
| 前段验证不授予新情境执行权 | 铸出的卡带 `requires_target_support`，两臂同样要过当前 Support 与同一道 delayed 权威门 |
| 分叉后 | 各自 store / ledger / 缓存 / 历史 / 提议轨迹 / 会话；互不供应发现 |
| 共同前缀的仪器读数 | 两臂**同样地**继承形成段的 replay 缓存（缓存项是"某程序在某格某面的读数"，要拿到必须自己先点名那个程序，因此不能泄漏"存在哪个程序"），计数器归零，后段各付各的 |

**没有同时引入新算子、新归因器或新检索模型。**

---

## 4. 两组结果

### 4.1 g1：`NO_KNOWLEDGE_TREATMENT`（形成段没形成任何新卡）——并且原因已定位

形成段 11 格全部跑完（181 fits / 69 LLM）：外环调用 10 次成功 + 1 次
`PROPOSER_FAULT`（模型输出违反 oneOf 模式：`workflow_program` 那一支缺 `scope_clause`），
评估 19 条提议（11 条入队 / 8 条 `NO_FEASIBLE_THRESHOLD`），**新开 5 张 Draft**
（`hampel_filter` / `outlier_mad>repair_level_shift` / `winsorize` / `outlier_mad>hampel_filter` / `denoise_median`），
另有入场态带入的 1 张（`resupplied_draft_1` = `outlier_mad`）——分叉时共 6 张未关闭，一并关闭。

**但 Fast 在形成段一次都没有部署新程序。** 部署路由是
`identity ×6 / searched_active_program ×2 / resupplied_draft ×2 / recalled_skill ×1`，
落地的程序只有 `outlier_mad({})` 和 identity。两次 delayed 权威门通过（u5、u15）
**部署的都是祖先自己的程序**，于是铸卡撞上 K0 已占用的 id
（`fast_winner_forecast_ridge_smase_outlier_mad`），`ABSENT` 前置拒绝，`activated: []`。

⇒ **不是"保存机制失败"，是"没有任何新程序通过过 delayed 权威门"**，因此没有可保留的新知识。
按任务书记 `NO_KNOWLEDGE_TREATMENT`，**不手工补卡**，也**不把它读成"知识使用无效"**。

**这一组因此变成了最有用的东西：噪声底。** A 与 B 在这一组机械上完全相同
（`formed=[]` ⇒ 移除集合为空 ⇒ 两臂起点逐字节一致），各自跑自己的外环轨迹：

| g1（无处理） | old-knowledge | accumulated-knowledge | 差 |
|---|---|---|---|
| Support 面均值 | 0.151520 | 0.167873 | **+0.016353** |
| delayed 面均值 | 0.196336 | 0.196643 | **+0.000307** |
| 覆盖 treated | 24 | 24 | 0 |
| 受损序列 | 5 | 5 | 0 |
| 权威门通过 | 0 | 0 | 0 |
| 后续段 fits / LLM | 153 / 38 | 111 / 33 | −42 / −5 |

差异全部来自 u17/u24/u25：两臂在后段各自的外环里排了**不同的**候选，
A 部署 `outlier_iqr({})`，B 部署 `hampel_filter({})>outlier_mad({})`。
这就是本设计的臂间运行噪声，**不是任何机制效应**。

### 4.2 g2：`KNOWLEDGE_TREATMENT`（形成段真的铸出一张独立卡）

**知识形成的完整来路**（全部来自轨迹记录，不用模型自述）：

| 环节 | 事实 |
|---|---|
| 提议 | u9 外环提出 `outlier_iqr`，明写是对父程序 `outlier_mad({})` 的替换（"用 IQR 阈值替代 MAD，对驱动单序列伤害的尾部更保守"） |
| 入队 | Frame C 队列，开为 `resupplied_draft_4`，**无部署权** |
| 试验 | u11 Fast 从供给池选中并部署（`resupplied_draft`），Support **+0.320571** |
| 验证 | u11 delayed **+0.308092**，权威门四条全过（**独立后续单元**，不是提出它的那一格） |
| 铸卡 | `activate_approved` → 新增独立卡 `fast_winner_forecast_ridge_smase_outlier_iqr`，Draft 关为 `VERIFIED_AND_MINTED_AS_ITS_OWN_CARD` |
| 父卡 | **原样存活**。边界库 = `[..._outlier_iqr, ..._outlier_mad]` —— 父子在真实课程里共存 |
| 复用 | u12、u15 经 `recalled_skill` 取回；u15 再次过门并再次 `activated`，`minted=[]`（**没有铸出第二张同程序卡**） |
| 其余 Draft | 形成段共新开 3 张（`hampel_filter` / `winsorize` / `outlier_iqr`），加入场态 1 张；除已验证晋级的那张外，3 张在分叉时关闭为 `NOT_CARRIED_ACROSS_THE_FORK` |

它是**新增程序**（库里原本没有 `outlier_iqr`），以"对父程序的修改"的形式被提出，
最终落成**自己的一张卡**，而不是对既有卡的一次修订——这正是本包要的保留形态。

**后续测试段（5 格 × 2 臂）：**

| g2 | u16 | u17 | u23 | u24 | u25 |
|---|---|---|---|---|---|
| A 部署 | `outlier_mad` | identity | `outlier_mad` | identity | identity |
| B 部署 | `outlier_mad` | identity | `outlier_mad` | identity | identity |
| B 是否**检索到**新卡 | ✔ | ✔ | ✔ | ✔ | ✔ |
| B 是否经卡片候选**实测**新卡 | ✘ | ✔ | ✘ | ✔ | ✘ |
| B 是否试了该卡的**程序** | ✘ | ✔ | ✘ | ✔ | ✔（自己搜出来的，非经卡片） |
| Support | 0.197057 / 0.197057 | 0 / 0 | 0.288988 / 0.288988 | 0 / 0 | 0 / 0 |
| delayed | 0.092525 / 0.092525 | — | 0.595428 / 0.595428 | — | — |

| g2（有处理） | old-knowledge | accumulated-knowledge | 差 |
|---|---|---|---|
| Support 面均值 | 0.097209 | 0.097209 | **0.000000** |
| delayed 面均值 | 0.137591 | 0.137591 | **0.000000** |
| 覆盖 treated | 18 | 18 | 0 |
| 受损序列 | 3 | 3 | 0 |
| 权威门通过 | 0 | 0 | 0 |
| 后续段 fits / LLM | 32 / 32 | 56 / 37 | **+24 / +5** |

---

## 5. 三个问题的正式回答

### 5.1 知识形成了吗？

**形成了，两组里有一组形成。**

* 形成的合法 Skill：`fast_winner_forecast_ridge_smase_outlier_iqr`（g2），一张独立卡。
* 来自哪些实际尝试与验证：u9 提议 → 入队为 `resupplied_draft_4` → u11 Fast 部署并拿到
  Support +0.320571 → **同一格 delayed +0.308092、权威门四条全过** → 铸卡。
  提出与验证发生在不同单元（`independent_unit: true`）。
* 是新增程序还是既有程序的修订：**新增程序**（`outlier_iqr` 库中原本没有），
  以对父程序 `outlier_mad` 的修改被提出，落成**自己的卡**；父卡不变、继续可用。
* g1 没有形成任何卡，原因是**没有任何新程序通过过 delayed 权威门**——两次过门部署的都是祖先程序，
  于是铸卡被既有的"同程序不重复建卡"机制挡下。记 `NO_KNOWLEDGE_TREATMENT`。

### 5.2 知识改变行为了吗？

**改变了"考虑什么"和"试什么"，没有改变"交付什么"。**（依据轨迹记录，非模型自述）

* **哪条 Skill 被读取**：`fast_winner_forecast_ridge_smase_outlier_iqr`，
  在后续段 **5/5 格**都出现在 B 的 `retrieved_skill_ids` 里；A 一次也没有。
* **是否改变了提议 / 试验顺序 / 程序修改**：
  * 候选集合 **5/5 格不同** —— B 的候选菜单每格都含 `outlier_iqr({})`，A 的菜单从来没有；
  * 试验序列 **5/5 格不同**；其中 **u17、u24 两格该卡以 `cand_skill_...outlier_iqr` 的身份真的被实测**，
    u25 B 唯一探过的程序也是 `outlier_iqr`（但那一次是它自己搜出来的，不算经卡片供给）；
  * 外环提议 5/5 格不同。
* **是否只被加载而没有行为差异**：**不是"只被加载"**——它进了菜单 5/5 格，还以卡片候选的身份被实测了 2 次
  （u17、u24），第 3 格 u25 Fast 自己又把同一个程序搜了出来实测。
  但**部署层一次都没有选它**：`units_where_a_formed_card_was_deployed = []`，
  两臂逐格部署相同，读数逐位相同。
* 判词：`TRIED_BUT_NEVER_SELECTED`。

### 5.3 行为改变有价值吗？

同一单元集、全服务人群、逐单元配对、UNKNOWN 保留、identity 记其真实回退值 0.0、
比较的是**两臂完整交付**而不是新增卡自身的成绩：

| 指标（g2，有处理组） | 差值（累积 − 旧知识） |
|---|---|
| Support 收益 | **0.000000** |
| delayed 收益 | **0.000000** |
| 覆盖（treated 序列数） | 0 |
| 伤害（受损序列数） | 0 |
| 最坏单序列伤害 | 两臂相同 |
| 权威门通过 | 0（两臂在后段**都是 0/5**） |
| Consumer fits | **+24** |
| 物理 LLM 调用 | **+5** |
| 找到首个合格方案所需反馈量 | **UNKNOWN** —— 两臂在后续段都没有任何一格通过权威门，没有"首个合格方案"可计 |

参照：无处理组 g1 的臂间噪声为 Support **+0.016353** / delayed **+0.000307**。
处理组测到的 0.000000 落在噪声线以内（且恰为零）。

**结论：这次行为改变没有价值——它只多花了 24 次 fits 和 5 次调用，交付一模一样。**

---

## 6. 收口

按任务书的收口规则，本包落在第一条：

> **知识未被使用：定位检索或使用入口，不继续增加卡片数量。**

但要把话说准，因为它决定下一步找哪儿：

* **检索不是卡点**。卡被保留、被检索、5/5 格进了候选菜单，其中 2 格以卡片候选身份被真实测量
  （第 3 格里 Fast 自己又搜出了同一个程序去测）。
* **保存方式不是卡点**。父子共存、子卡失败不误伤父卡、不重复建卡、计数不清零——都成立且被验证。
* **卡点在选择层**：决定"部署哪个候选"的那一步，从来没有选中过这张卡；
  实测之后仍然选了 identity 或祖先程序。这与 DEV-AUTO-1 记下的
  `PROPOSAL_DIVERSITY_GAP` / 选择层问题是同一处。

因此：**不建议再加卡片数量，也不建议再改保存方式**。
下一个该被打开的是"选择"——Fast 拿到一张已验证的卡之后凭什么不用它。
（本包不自行推进到那里：那会动选择规则，超出本次授权。）

同时**不满足**"相同预算下更好 / 达到相近质量更省反馈"，所以**不进入独立情境验证计划**。
也**没有为了凑正号追加运行**——两组各跑一次，配置与执行顺序在运行前写定。

---

## 7. 成本与边界

| 项 | 数字 |
|---|---|
| Consumer fits | **798 / 2000**（首次失败上手 10 + 冒烟试跑 67 + 正式两组 721） |
| 物理 LLM 调用 | **312 / 1000**（2 + 27 + 283） |
| 墙钟 | ≈ 20 分钟 / 6 小时 |
| 每臂次额度 | 166 次调用（6 个臂次），**未恢复每格小调用帽** |
| 每阶段 fits 额度 | 4 个阶段各 500，4×500 = 包级上限；阶段在**花之前**拒绝 |
| 触顶 | 无。`stopped_at` 两组两段全为 `None`；每组 11 个形成格 + 5 个测试格 × 2 臂全部跑完 |

**边界**：评估面(+144) 读取 0；密封数据读取 0；`TARGET_HELD_IN` 读取 0；风险常数、Scope 栅格、
Consumer 均未触碰；未向模型提供任何已知赢家；两测试臂之间无共享缓存；**PATCH 卡片次数 0**；
**编辑核心文件数 0**；未新增哈希平台；未覆盖历史收据；未提交 git。

**传输**：`https://api.deepseek.com/v1` / `deepseek-chat`，与 DEV-AUTO-2、DEV-AUTO-3 逐字相同。
本包新增一道预检：**live transport 必须与上一包收据里的 transport 一致**，
否则预检直接失败（期望值从 DEV-AUTO-3 的收据里读，不在代码里写死主机名）。
这道检查是被真实事件逼出来的——第一次上手时 `M0_AGENT_*` 未设，回退到模块默认的中转站，
对方以 `insufficient_quota` 拒绝，而那是一个 DEV-AUTO-2/3 从未用过的模型。

---

## 8. 这份结果不是什么

* **不是独立泛化证明**。5 个已暴露的 development 单元，切分依据是课程序列组结构；
  旧工件里的 UNREAD 标签不代表现在仍未曝光。
* **不是效应量估计**。有处理的组只有 **1** 个（g2），另一组没有处理可测。n=1，不做显著性主张。
* **两组不冒充独立数据复现**。g1 提供的是噪声底，不是第二次同一测量。
* **不是"知识保留机制失败"的证据**。保留机制该做的都做到了，问题在下游。
* **也不是"知识使用无效"的证据**。只测到"这一张卡、在这五格、被这一套选择规则拒绝了"。

---

## 附录 A：换模型（gpt-5.6-sol）——探查、两次 OOM、然后完整补跑成功

用户 2026-09-07 给了新的中转配置并说"如果是因为 LLM 回答质量问题的话可以用这个再试下 gpt-5.6-sol"，
随后又下了一份完整补跑的任务书。**下面的一切都不与正文合并、不相减**：正文是 deepseek-chat
（与 DEV-AUTO-2/3 同传输、可比），附录是另一个模型的独立两组实验，`comparability` 字段在
`dev_know1_contrasts_sol.json` 里显式声明了这一点。

### A.1 能不能跑通：**能**（探查阶段的判断，已被 A.6 起的完整两组结果验证）

先前 `upstream_error` / 断连是对端上游的临时故障。裸调用、`max_tokens`、`max_completion_tokens`、
`temperature` 都能通（单次 5–50s，抖动很大）。harness 的 `AgictoChatCompletionsBackend.complete()`
只发 `model` + `messages`（不带 `response_format` / `max_tokens`），正是最稳的那个形状。

### A.2 探查阶段观察到的两件事（当时标记为部分记录，现已被完整数据取代或印证）

**① 形成段明显更活跃，且多次尝试互相复现。** 探查阶段两次独立的 sol g1 形成段都在 u11
经 `resupplied_draft` 部署 `outlier_iqr` 并过门、铸出独立卡；完整补跑（A.6 起）里 g1、g2
**两组都在同样的 u11 铸出同一张卡**，验证读数（delayed 0.308092、support 0.320571）在两组之间
**完全相同**——形成段这一步在 sol 上是可靠复现的，不是偶然。deepseek 的 g1 形成段一个新程序都
没部署过，这一点两个模型之间确有差异，但只是观察，不是"sol 比 deepseek 更容易形成知识"的
正式主张（n=2 vs n=1，且不是同一批种子/序）。

**② 探查阶段丢失、现已被正式复核的 u16 读数。** 探查阶段第一次尝试的 g1 后续段 u16 显示
累积臂 delayed 0.132108 高于旧知识臂 0.092525，但那次的 checkpoint 在重跑前被清理步骤删掉，
当时明确"不据此下任何判断"。**完整补跑的 g2 在 u16 独立重现了同一形态**（累积臂 0.132108，
旧知识臂 0.092525，逐位相同）——这不是巧合复原丢失的数字，而是 g2 这次正式、完整、有工件
支持的读数；探查阶段那条记录现在可以归档为"提前看到的同一件事"，不再是孤证。

### A.3 探查阶段为什么没能出结论：两次都被系统内存杀掉（已解决）

* 第一次：跑到 g1 后续段 u16 之后被杀（243 fits / 75 LLM / 3202s）。
* 第二次（只跑 g1、降低峰值）：跑到形成段 u13（9/11 格）被杀（139 fits / 50 LLM / 2487s）。
* 两次剩余物理内存都只有 2.2–2.4 GB / 15.7 GB，占用分散在 Cursor / Chrome / WSL 等进程上，
  没有可安全回收的大户，且两次都是 0 cell fault——环境限制，不是模型或代码故障。

补跑前对 `run_dev_know1_two_arms.py` 做了三处工程修补（都是既有机制的推广，不是新存储/哈希
平台）：**(a)** `run_id` 把 checkpoint 目录、`.dev_know1_runs` 存储根都命名空间化，新尝试不会
覆盖旧 checkpoint 或旧 partial 工件；**(b)** `Budget.to_state/from_state/save/load` 让包级
fits/LLM/wall 账本能跨 OS 进程续接——g1、g2 分别在独立进程里跑，进程退出即释放内存，`main()`
在有 `run=<id>` 时自动从 `package_budget.json` 续接上一进程花掉的额度；**(c)** 形成段结束、
fork 完成两个关键节点各自落一份 `*_formation_done.json` / `*_fork.json`，不必等到整包写完
才留下证据。这次 g1、g2 各自单独起一个 Python 进程，内存在两次低谷（1.2–1.8 GB 空闲）都扛住，
**两组都以 exit code 0 跑完，0 cell fault，`stopped_at` 全部为 `None`**。

### A.4 传输与用量（这次新增的记录能力）

`transport_actual_usage` 字段（新加的读数面，直接读自 `BudgetedAgentBackend` 已有的计数器，
不是新建的计量口径）显示：两组的**请求模型与返回模型完全一致**，都是 `gpt-5.6-sol`（不像
deepseek 探针那次意外发现的 `deepseek-chat`→`deepseek-v4-flash` 别名路由）。用量：g1 约
55.4 万 prompt tokens / 4.4 万 completion tokens，g2 约 54.9 万 / 4.2 万——sol 的多轮工具调用
明显比 deepseek 更"啰嗦"，这也是它慢约 9 倍的一部分原因，不只是网络延迟。

---

### A.6 完整两组结果：形成段

两组**都**在 u11 经 `resupplied_draft` 铸出同一张独立卡
`fast_winner_forecast_ridge_smase_outlier_iqr`（程序 `outlier_iqr({})`，替换的是既有的
`outlier_mad({})`），`treatment=KNOWLEDGE_TREATMENT`。verification 读数在两组间逐位相同
（delayed 0.308092、support 0.320571）——因为验证发生在两组共享的同一课程单元 u11 上。
两组的差别只在**提出时机**：g1 在 u5 提出、g2 在 u7 提出，都是在 u11 被 resupply 后验证。

| | g1 | g2 |
|---|---|---|
| 提出于 | u5 | u7 |
| 验证/铸卡于 | u11 | u11 |
| 形成段 support 均值（治疗人群） | 0.154739（harm 7 / treated 70） | 0.14587（harm 5 / treated 62） |
| 形成段 delayed 均值（治疗人群） | 0.079621（harm 17 / treated 53，门通过 3） | 0.081848（harm 16 / treated 52，门通过 3） |
| 部署来源分布 | recalled_skill 3、identity 5、searched_this_unit 1、searched_active_program 1、resupplied_draft 1 | recalled_skill 2、identity 6、searched_active_program 2、resupplied_draft 1 |
| 排队但从未验证的候选 | 7 个（hampel_filter、winsorize、denoise_median 系列等） | 8 个（同类，含 denoise_wavelet） |

没有卡在形成段丢失（`lost_during_the_formation_segment` 两组均为空），没有卡被 PATCH
（`no_card_was_patched=True`），11 个外层调用两组都全部拿到了模型的正式回复
（g1: 10 PROPOSED + 1 ABSTAINED；g2: 11 PROPOSED，无 abstain）。

### A.7 分叉后：两臂持有的卡

* 旧知识臂：`fast_winner_forecast_ridge_smase_outlier_mad`（唯一）。
* 累积知识臂：`fast_winner_forecast_ridge_smase_outlier_iqr` + `fast_winner_forecast_ridge_smase_outlier_mad`。

与正文 g2 一致的分叉机制：形成段全部开放 Draft 在分叉时统一关闭
（`FORK_CLOSURE`），两臂从同一个 Draft 账本深拷贝，Episode 列表清空重建，
replay 缓存的公共前缀按同一套 `_copy_cache` 规则复制给两臂——这次连同 `run_id` 隔离一起，
`smoke_dev_know1.py` 的全部 12 项预检在 sol 传输下逐条通过（含新加的
`the_transport_is_the_one_the_previous_package_used`，通过 `DEV_KNOW1_TRANSPORT_NOTE`
显式声明"这是一次刻意的模型切换"）。

### A.8 完整两组结果：后续段——七点报告

**①知识形成了什么、从哪来。** 见 A.6：两组都形成了同一个程序的独立卡（`outlier_iqr`
替换 `outlier_mad`），是对库里已有程序的一次修订（不是全新算子），经 11 次外层提案、
两组各自约 17 次候选评估、在同一验证单元 u11 上经 delayed 门槛。

**②新卡什么时候被读取/探测/准入/部署/保留/撤销。**
读取（在库中出现、进入候选清单）：**两组后续段全部 5 个单元，累积臂 100% 都能看到它**
（`cards_actually_retrieved_by_the_accumulated_arm` 每格都非空）。
探测（该卡自己的 skill_id 真的被拿去做候选并测过）：**g1 全程 0 次**；**g2 仅 u17 一次**，
且 u17 那一格两臂最终都摊回 identity（0 覆盖），探测了但没有下文。
部署：**两组全程 0 次**——`the_formed_card_was_deployed` 在 15 个"组×单元"记录里全部是 false。
保留/撤销：两组后续段都没有新的撤销事件，卡保持在库中直到段末。

**③未部署的原因，三种要分开看。**
* *未探测到*：g1 的 5 个后续单元全部属于这一类——形成的卡在库里，但当格的候选搜索
  从未把它自己的 skill_id 纳入候选，无论旧知识臂还是累积知识臂都各走各的搜索路径。
* *探测了但不合格*：g2 的 u17——`fast_winner_forecast_ridge_smase_outlier_iqr` 确实进了
  候选清单并排在探测顺序第一位，但该格两臂最终都是 0 覆盖、0 收益，形成卡没有单独被证明
  不合格，是这一格本身对两臂都没有可服务的目标序列。
* *合格但未被选中*：**没有出现**——本次运行里，"合格但被更弱的候选挤掉"这种情况一次都
  没发生；真正决定结果的，是下一条。

**④两臂完整后续段的全服务人群效用/风险/覆盖/门通过/成本。**

| | g1 support | g1 delayed | g2 support | g2 delayed |
|---|---|---|---|---|
| 累积臂均值 | 0.039411 | 0.018505 | 0.042537 | 0.026422 |
| 旧知识臂均值 | 0.097209 | 0.137591 | 0.097209 | 0.137591 |
| 差值 | **−0.057798** | **−0.119086** | **−0.054672** | **−0.111169** |
| 覆盖（treated 序列） | 7 vs 13 | 3 vs 18 | 7 vs 13 | 3 vs 18 |
| 受害序列 | 1 vs 3 | 1 vs 3 | 3 vs 3 | 0 vs 3 |
| 门通过 | 0 vs 0 | 0 vs 0 | 0 vs 0 | 0 vs 0 |
| 后续段成本（fits/LLM） | 累积 55/28，旧知识 70/29 | | 累积 67/30，旧知识 76/29 | |

两组、两个读数面**方向完全一致：累积知识臂在这五格上系统性地覆盖更少、交付更少**，
门通过两臂都是 0（谁都没有拿到权威 delayed 门的通行），首次合格部署的"用了多少反馈"因此
两臂都是 UNKNOWN（分母不存在，不是没测）。这与正文 deepseek 的"两臂逐位相同、差值 0.000000"
不是同一个结果——sol 上这张卡**确实改变了交付，而且改变的方向是负的**。

**⑤差异来源，定位到具体单元和具体动作，不只按标签。**
两组的差值几乎全部来自同一个单元——**u23**：

* g1 u23：旧知识臂经 `searched_active_program` 找到 `outlier_mad({})` 并部署，覆盖 6（support 面）/
  15（delayed 面）条序列、拿到 0.288988/0.595428 的正收益；累积知识臂当格候选集里**只有**
  `outlier_iqr({})`（一个刚被重新局部化搜索出来的候选，`localized_extreme_deviation_iqr`，
  **不是**已铸卡的 `fast_winner_forecast_ridge_smase_outlier_iqr` 本身——`the_formed_card_was_deployed`
  在这格仍是 false），这个候选没有通过，累积臂**没有像旧知识臂那样退回去找 `outlier_mad`**，
  直接摊回 identity，0 覆盖。
* g2 u23：旧知识臂这次是经 `recalled_skill` 直接调出它自己持有的 `outlier_mad` 卡部署，
  同样覆盖 15 条、收益 0.595428；累积知识臂候选集同样**只有** `outlier_iqr({})`，同样没通过，
  同样摊回 identity。
* g2 u16 是唯一一个累积臂真的部署了 `outlier_iqr({})`（经 `searched_active_program`，仍然
  **不是**recalled 已铸卡本身）并且**赢了**旧知识臂（delayed 0.132108 vs 0.092525，
  +0.039583）的单元——但这一点收益远远盖不过 u23 的损失（delayed 面 −0.595428）。

**归因必须精确**：两臂在这些单元里出现的"outlier_iqr"候选，绝大多数时候是候选搜索**当格
重新发现**的一个局部化变体（`arrived_as: searched_active_program`），而不是 `recalled_skill`
调出形成段铸的那张卡；真正把已铸卡本身当候选测过的，全程只有 g2 u17 一次。也就是说：**这次
sol 上观测到的行为差异，主要不是"积累的卡被复用"造成的，而是"累积臂库里多了 outlier_iqr
这个选项之后，候选搜索在某些格倾向于只围绕它转、不再退回旧程序"这个副作用造成的**——是选择
策略对库内容的敏感性问题，不是卡本身的内容问题。

**⑥sol 内部"累积臂 vs 旧知识臂"是这里的主比较**，两组符号一致（`sign_agrees_across_the_treated_groups=True`），
不拿 sol 的总分和 deepseek 正文的总分相减做任何主张——`dev_know1_contrasts_sol.json` 顶层
`comparability` 字段原文写明"this run used a transport the previous package did not; its
numbers stand on their own and are not to be differenced against DEV-AUTO-2/3 or against the
deepseek-chat run"。

**⑦两组都完整报告，不用"噪声底"或"检测阈值"这类说法。**
本次两组都是 `KNOWLEDGE_TREATMENT`（都形成了知识），所以没有"无处理组"要保留；不存在
需要用噪声底措辞的场景。若把两组当作两次独立重复看，两次的符号、量级、主因单元（u23）
高度一致，这本身是一个值得记录的复现性观察，但仍然只是 n=2 的开发级判断，不做显著性主张。

### A.9 收口：与正文规则一致的判定

`audit_dev_know1_contrasts.build(suffix="sol")` 给出的收口与正文用的是同一套规则：
`closing_rule = KNOWLEDGE_WAS_NOT_USED`，理由原文——"retrieval is not the blocker -- the
card reached the candidate menu, and in the probed case it was even measured -- so the use
entry point to locate is *selection*: what decides which candidate is deployed. Do not add
more cards, and do not change how they are stored."

这与正文 deepseek 结果指向的卡点**是同一层**（选择层），但**症状不同**：deepseek 上选择层
从不挑这张新卡（0 概率被部署，交付逐位相同）；sol 上选择层会围绕新卡转、偶尔部署它的一个
重新发现的变体，代价是丢掉了旧程序原本能覆盖到的那部分人群。两个模型独立指向同一个结构性
问题——"决定部署哪个候选"这一步，而不是卡怎么存、怎么读——这是本附录能负责任地说的最强
一句话，仅此而已；不构成"选择层有 bug"或"应该修选择策略"的工程结论，任务书明确排除了
在本包内改选择策略。

### A.10 累计成本（含全部失败尝试与本次完整补跑）

| 项 | fits | LLM | 墙钟 |
|---|---|---|---|
| deepseek 正式两组（正文） | 798 | 312 | ~0.9h |
| sol 探针（早期，完整跑通的小样本） | 75 | 24 | ~16min |
| sol 第一次完整尝试（被 OOM 杀） | 243 | 75 | ~53min |
| sol 第二次尝试，仅 g1（被 OOM 杀） | 139 | 50 | ~41min |
| **sol 完整补跑，g1+g2（本次，成功）** | **621** | **240** | **~2.77h** |
| **DEV-KNOW-1 全部尝试合计** | **1876 / 预算按包计** | **701 / 预算按包计** | **~5.2h（跨多个包）** |

包级预算（2000 fits / 1000 LLM / 6 小时）按每次运行单独计算，本次成功的完整补跑单包用了
621/2000 fits、240/1000 LLM、9989.3s（约 2.77 小时）——远在预算内；上表"全部尝试合计"是
跨越多次独立包运行的累计记录，不是对某一次包预算的核销。
