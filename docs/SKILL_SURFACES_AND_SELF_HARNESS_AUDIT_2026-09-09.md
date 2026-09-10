# Skill / Scope / Surface 与 Self-Harness 定向审计

日期：2026-09-09。角色：Assignment B（只读调研）。复核：Astra。  
执行模型：WSL `grok-4.6`（研发调用，**不是**项目实验 LLM）。  
本文件是调研底稿，不是项目正式结论。未启动新实验、未改源码、未改参考仓库。

**Astra 重点复核（本轮追加并更正正文）：** Skill 字段、Scope 分工、最新池化入口、
Self-Harness 诊断后聚类的核心源码已核。初稿把旧 online_loop 的 Support 覆盖写成
最新入口的问题，已改正：**DEV-DEPLOY-1/2 已经是 Fast-only，不需要再修一次 select
否决权。** 还更正了 SEQ-3 的“3/3 都翻转”、DEPLOY-2 的全臂 40/40 暴露、GEPA
接受机制与“结构化局部修改尚未开放”的表述。近邻论文总体分数未重算，本地 PDF
与抽出稿的逐页对应未由主助手独立复核；不把底稿全部细节标成已验收事实。

---

## 0. 读本页前先记住的边界

这些不是套话，后面每一节都按它们收口：

- **成对证据不是每条 Episode 自带的。** 比较 P/Q 需要同 UID、同窗口、同 Task/Consumer/计分语义下两程序各自的读数；未测的一侧为 UNKNOWN，不要求 P 与 Q 是同一个程序。
- **收益受训练—服务管线共同影响。** 当前逐序列选择的是「准备—训练—服务」整条管线，不是「只清洗这一条序列、Consumer 固定」。
- **Scope 条件可判定 ≠ 效用可预测。** 检索命中只说明卡被放进 prompt，不说明处理会变好。
- **结构化合法 ≠ 学到有效知识。** 编译通过、暴露检查通过、schema 合法，都不等于后续分数提高。
- **代码可达 ≠ 实际起作用。** 面在目录里、卡在库里、字段在 schema 里，都可能从未进入 Fast 实际读取的字节。
- **指导起作用 ≠ 涨分。** DEV-SEQ-3/4 已出现「select 改了、交付没改」。
- **C-minus 弃权不是有效性因果证明。** 只说明那两次提案会话在缺少效果字段时不写修订。
- **事后菜单空间 ≠ 部署时能识别。** 同口径菜单 oracle 约束的是该菜单内的事后选择，不是整个可生成 Workflow 空间的上界或部署可实现性保证。
- **近期故障零 / 删行均值不能当干净结果。** DEPLOY-1 有空选择回退；DEPLOY-2 的 240 行含 18 条空选择。历史数字不因本审计改写。

历史 runner 行为以 Git HEAD `c845b4b87cd9068929ef1d7cb0062684a6e83a65`（`Checkpoint per-sequence harness development through DEV-DEPLOY-2`）为准。审计时工作区 **未改** `run_dev_deploy1.py` / `run_dev_deploy2.py`。Worker A 可能并行改这两个文件；若读到工作区新字节，必须标明，不得与 HEAD 混成同一版本。

---

## 1. 一页白话：现在这套系统在做什么

### 1.1 术语（当前实现，不是愿望）

| 词 | 现在实际是什么 | 现在实际不是什么 |
| --- | --- | --- |
| **General Skill** | 设计上的语义角色：对每条序列都该读到的观察 / 构造 / 选择原则。实现上主要落在 3 张 bootstrap 卡 + `candidate_policy.proposal_guidance` / `selection_guidance`。 | 不是独立的 `SkillKind`，schema 里没有 `general`。 |
| **Specific Skill** | 设计上的语义角色：有适用条件的情境指导。实现上是 `skill_kind=capability` 的 learned 卡，靠 `observable_applicability` 检索。 | 不是强制的 WHEN/OBSERVE/TRY 结构体。正文多半是散文。 |
| **Program / Workflow** | 编译后的 1–4 步线性算子序列。 | 不是 Agent 的多轮对话步骤，也不是任意 DAG。 |
| **Episode** | 一次合法 Action–Response 记录。 | 不自动获得执行权，也不直接进 Fast prompt。 |
| **Fast** | 逐序列：inspect → propose → verify → select。读冻结知识 + 本序列可见特征。 | 不读当格下游分数，不读跨轨迹 raw Episode bank。 |
| **Slow** | 批次边界（SEQ/DEPLOY）或方法层单条反馈（`handle_feedback`）上的一次 edit 调用。 | 不是完整因果诊断算法。一次 LLM 提案 ≠ 已定位根因。 |
| **Scope** | 至少四种不同对象，见第 5 节。 | 不能用一个「有 Scope」回答四个问题。 |

### 1.2 白话结构图（当前入口，不是 8 月 19 日愿望图）

```text
一条序列、一个合法窗口
        │
        ▼
公开特征（默认整段 values[:origin]，不是服务窗 192 点）
        │
        ▼
resolve_harness_view(role=fast)
  ├─ bootstrap 三张：永远加载
  ├─ capability：applicability 为真，再按 retrieval.top_k=2 截断
  ├─ 冻结程序卡：若 body 含 "Frozen program steps:" → 机械进候选池
  └─ 散文指导卡：只进 prompt，不自动变成候选
        │
        ▼
Fast inspect / propose / select
  （同一份 Resolved Harness JSON 出现在每个阶段的 system）
        │
        ├─ Agent 提案：schema 最多 3 个候选、每案 1–4 步
        ├─ 默认只保留 1 个 Agent 提案进池（agent_proposals_kept=1）
        ├─ 可见算子：先用「单步默认参数」过 verifier 过滤
        └─ select 产出当前选择；它如何变成交付取决于下面的入口
        │
        ▼
当前 DEV-DEPLOY-1/2：prepare 的有效选择经合法性检查直接交付
  无当格 Support；正常 identity 不被其他候选覆盖
  历史缺陷：FAILED 被误计 identity/0，本轮 A 正在修，不能当正常选择
另一个旧入口 online_loop（DEV-SEQ 历史实验）：
  chosen 非 identity 时排到探测序首；Support 首个准入者交付
  （此旧入口的 identity 无否决权；不再当作最新 DEPLOY 的阻塞）
        │
        ▼
本序列自己的分数（训练语料仍是共享的）
        │
        ▼
批次边界 Slow：看形成材料，改一处已授权文本面
        │
        ▼
fork 编译成子快照 → 暴露检查 → 后续单元 Fast-only / 两臂对照
```

用户要记住的一句话：**共享的是知识和 Agent，不是同一个处理程序；但共享训练材料仍会让「这条序列的分数」不等于「只改这条序列」的因果效应。**

---

## 2. 核查范围与版本

### 2.1 主项目

| 项 | 值 |
| --- | --- |
| 仓库 | 当前工作目录 `SelfEvolvingHarnessTS-deepseek-guidance-evolution` |
| Git HEAD | `c845b4b87cd9068929ef1d7cb0062684a6e83a65` |
| 工作区已有未提交改动（审计时） | `AGENTS.md`、`docs/DECISIONS.md`、`docs/SLOW_FEEDBACK_UPDATE_DESIGN_NOTES_2026-09-08.md`、两份 Fable/HEC 文档；大量 `.dev_*_runs/` 未跟踪工件 |
| 两个 deploy runner | 审计时相对 HEAD **干净**；历史行为按 HEAD |
| 正典 | 工作区 `AGENTS.md` / `Agents.md`（内容一致）§1–5.6；设计备忘录 §15 |
| 未读完整遍 | 未把整个项目再读一遍；只为解释差异追到 `online_loop.py`、`scope_executor.py`、`edit_controller.py`、`group_fault.py`、`source_skill.py`、`exploration_policy.py`、`public_features.py` |

### 2.2 Self-Harness 参考材料（版本不一致，必须分开）

| 材料 | 对应哪一版 | 差异 |
| --- | --- | --- |
| 本地仓库 `../Self-Harness/` | 公开代码快照；README 表是 **overall** pass：MiniMax 42.2→53.9，Qwen 18.0→36.7，GLM 46.1→57.0 | 本地只有 Terminal-Bench-2.0 相关代码与 `harnesses/qwen_tb2_final/`；**没有** SWE-bench / AppWorld 实验树 |
| 本地 PDF `../Paper/Self-Harness Harnesses That Improve Themselves.pdf` 与 `../self_harness_extracted.md`、`../Paper/_selfharness.md` | 与 **arXiv 2606.09498v1**（2026-06-08）一致：只报 TB2；held-out MiniMax 40.5→61.9 等 | 抽出稿与 PDF/v1 数字一致 |
| 公共 arXiv **v3**（2026-08-20） | 抽象改为三个基准（TB2 / SWE-bench Verified / AppWorld），九组模型–基准都报 held-in 与 held-out 上升 | **本地 PDF/抽出稿不是 v3**。v3 Table 1 把 v1 的 split 数字与 README 的 overall 数字放进同一张表 |
| 本地 `eval/README.md` | TB2 Clean64：train 43 / heldout 21 | 这是它自己的回归门分母，**不是**本项目密封 held-out |

本轮以 **本地源码 + 本地 PDF/v1** 核对机制；v3 只作公开版本差异披露。不把旧调研笔记当本轮核对结果。

### 2.3 近邻

- **GEPA**：公开论文 arXiv:2507.19457（v2 / ICLR 2026 Oral）；本地另有 `../a-evolve/docs/algorithms/gepa.md`（A-Evolve 封装，不是官方实现正本）。
- **AutoGuide**：公开论文 arXiv:2403.08978v2（NeurIPS 2024）。本机无独立源码树；方法细节来自公开摘要与 HTML 检索片段。arXiv HTML 直取被环境 SSRF 拦截，故 **AutoGuide 的逐节行号标为公开 HTML/摘要，不是本地文件行。**

公共检索只用论文题目与通用关键词，未发送本项目私有路径、数据或凭据。

---

## 3. B1 八问：当前架构（源码优先）

### 3.1 General / Specific 在哪定义、存什么？有没有结构化 WHEN/OBSERVE/TRY/RISK/VERIFY/FALLBACK？

**正式数据结构里没有 General/Specific 这两个 kind。**

`SkillKind` 只有三种：`bootstrap_procedure` / `capability` / `safety`（`contracts/harness.py:101-105`）。  
`SkillEntry` 字段：`skill_id`、`skill_kind`、`revision`、`body`（非空 UTF-8 文本）、`observable_applicability`、`allowed_tools`、`risk_guards`、可选 `serving_scope`（`contracts/harness.py:107-123`；schema `contracts/schemas/skill_entry_v1.json`）。

AGENTS §5.6 / 设计备忘录 §15.1 把 General/Specific 写成**语义角色**，并明确「不急于新增 SkillKind」。当前实现映射是：

| 语义角色 | 当前落点 | 提示里长什么样 |
| --- | --- | --- |
| General | h0 三张 bootstrap：`inspect_and_localize`、`build_contrastive_candidates`、`select_or_identity_and_verify`；外加 `candidate_policy.proposal_guidance` / `selection_guidance` | bootstrap 的 `body` 是长散文；`build_contrastive_candidates` 用 `[FIXED_CONTRACT]` / `[inspect_pattern_guidance]` / `[propose_construction_guidance]` / `[select_guidance]` **段落标题**，不是 schema 字段。guidance 两句很短（`candidate_policy.json:5-6`） |
| Specific | learned `capability` 卡 | 多数是一段 `body` 散文 + 一份 applicability AST。若 body 含 `Frozen program steps:`，Fast 还会把它编成候选 |

**六段 WHEN/OBSERVE/TRY/RISK/VERIFY/FALLBACK 不是 Skill 必填结构。** 它出现在 Source 巩固卡上：

- 定义：`evaluation/functional/task_episode_harness/agentic/source_skill.py:22-51`，`SECTIONS = ("WHEN", "OBSERVE", "TRY", "RISK", "VERIFY", "FALLBACK")`
- 存储：body 里把六段拼成文本，并**再写入** `risk_guards.sections`（同文件约 959–983 行）
- Fast 侧识别：`retrieval.py:158-168` 只在 `risk_guards.sections` 含 `TRY` 时把它当 experience card；无授权 TRY 且无重复 scoped RISK 则对 Fast 隐藏（`retrieval.py:195-238`）
- h0 bootstrap 与 Target-local 冻结程序卡**没有**这六段

结论：结构化六段是 **Source 经验卡的约定 + risk_guards 旁路字段**，不是当前 Skill schema 的一等公民。General/Specific 在提示里主要是散文。`_skill_prompt` 甚至不把 `observable_applicability` 放进模型可见 JSON（`agent_core.py:135-142`）：条件只在检索时生效，Fast 读到的是 `skill_id/kind/body/allowed_tools/risk_guards`。

### 3.2 Skill 现在还有哪些 Scope？四种含义必须分开

| 含义 | 存在哪里 | 谁消费 | 在哪执行 | 是否阻断加载或处理 |
| --- | --- | --- | --- | --- |
| **1. `observable_applicability`** | `SkillEntry` 必填 AST（`const` / `feature-op-value` / `all`/`any`/`not`） | `resolve_harness_view`（`retrieval.py:278-289`） | 检索时，对当前序列公开特征求值 | **阻断 Fast 加载**：不匹配则该 capability 不进 view。bootstrap **不走** 这条过滤（`retrieval.py:249-255`，「always」）。匹配后仍受 `top_k=2` 截断 |
| **2. legacy `serving_scope`** | schema 可选；`scope_type` ∈ `all_serving_series` / `serving_series_predicate` / `none`，谓词是部署可见特征，**禁止存 UID**（`skill_entry_v1.json:54-101`） | 组级 `ScopeExecutor.evaluate` / `online_loop` 双管线 | `scope_executor.py:381-417`：选中序列走 prepared 管线，未选走 raw，未选与 Static 逐位相等；代价是第二次 Consumer fit | 组级路径可改变**哪些序列被处理**。**逐序列路径故意忽略**：`per_sequence.py:287-294`「accepted and deliberately ignored」。SEQ/DEPLOY 的 `edit_preflight` **禁止**指导卡带 `serving_scope`（`dev_seq2_knowledge.py:630-633`） |
| **3. Workflow region 参数** | 算子参数 `region_start_fraction` / `region_end_fraction`；targeting_mode ∈ `global` / `intrinsic` / `external_region`（`operators/registry.py:65-69`） | Fast verify 与执行器 | 作用在**当前这条序列的点区间**，不是人群谓词。intrinsic 算子自己找命中点；external_region 必须绑定 inspect 区域（bootstrap 正文 4e） | 超 `max_modified_fraction=0.35` 或破坏区域外点 → verifier 拒掉该候选，不加载 Skill |
| **4. 实际评价人口** | runner 的 roster / `eval_uids`；SEQ/DEPLOY 写死 20 条 UID | 计分器 `per_sequence.UnitReadings` / `SequenceView` | 分数分母是这批被服务序列 | **不**由 Skill 谓词改小。收窄检索 ≠ 缩小评价人口 |

第三种与第四种经常被混谈。逐序列之后，赋值边界是「这条决策所属的序列」，但评分仍包含共享训练语料经同程序处理的作用；Skill 上的 applicability 决定**知识是否被读到**。

### 3.3 Fast 实际读取什么字节、何时读；一条真实 Skill 的全程

**同一份 view 进三个阶段。** `fast_agent.py:780` `view = resolve_harness_view(snapshot, features, role="fast")`，随后 inspect（946–965）、propose（1018–1045）、select（1302–1320）都传同一个 `harness_view`。

system 字节（`agent_core.py:178-192`）：

1. `instruction.md` 短规则 + 运行时 JSON 信封规则  
2. 整份 `Resolved Harness` JSON：`instruction`、`skills[]`（见上，**无 applicability AST**）、`memories[]`、`controls`  
3. `controls.candidate_policy` 含 `proposal_guidance` / `selection_guidance` / `total_k` / 槽位（h0：`total_k=4`，identity 1，agent_program_slots 3）  
4. `controls.verification` 含 `max_modified_fraction=0.35` 等  

user 侧按阶段不同：inspect 有 features + probe panel；propose 另加 inspection 与 `allowed_operator_contracts`；select 另加已验证候选。

各知识到哪一段：

| 知识 | inspect | propose | select | 机械效果 |
| --- | --- | --- | --- | --- |
| bootstrap 三张 | 在 skills 里 | 在 | 在 | 纯文本。`build_contrastive_candidates` 自称「每个 Fast 阶段都读，只用当前阶段那一节」 |
| 新指导 capability | 仅当 applicability 命中且挤进 top_k | 同 | 同 | 只影响 prompt，除非另有冻结 steps |
| 冻结程序 capability | 同上（先要进 view） | 同 | 同 | `_parse_frozen_steps`（`fast_agent.py:273-299`）→ `_skill_frozen_candidates`（326–371）机械进池 |
| proposal_guidance | controls 里全程 | 同 | 同 | 文本 |
| selection_guidance | 全程都在 system | 同 | **意图上对 select** | DEPLOY 有效选择直接交付；旧 online_loop 则仍有 Support 覆盖 |

**真实例子 A（h0 General，永远在）：**  
存储 `methods/ttha/harness/h0/skills/bootstrap/inspect_and_localize.json` → 检索 `bootstrap: always`（`retrieval.json` + `retrieval.py:249-255`）→ prompt 的 `skills[0..]` → inspect 阶段要求输出最多两个 `pattern_hypotheses`。这张卡自身不含冻结候选；不能据此说整个 Fast 没有候选或交付。

**真实例子 B（DEV-SEQ-3 Specific，有实测）：**  
存储 learned 卡 `missingness_deviation_identity_guard` → 命中 3/60 决策 → 进入提示。
SEQ-3 中 B 的 identity 选择是 3/3、A 为 1/3，**不是三条都翻转**，三对交付相同。
随后 SEQ-4 同起点三重复把提示性效应定位在 u10/T146：A 0/3、B 3/3 选 identity，
六次仍交付 MAD。该对照测试的是原指导；Slow 后来 PATCH 的新正文是另一对象，
不能把此前选择翻转归因于后来那次 PATCH。见 `DEV_SEQ4_GUIDANCE_ADOPTION_RESULT` §2–3。

**真实例子 C（冻结程序卡，K0 常见）：**  
h0 `skills/learned/` **为空**。K0 的冻结程序卡来自更早 AUTO/KNOW 快照（例如 `.dev_auto1_runs/.../fast_winner_forecast_ridge_smase_outlier_mad.json`），body 含 `Frozen program steps:`。进 view 后机械占池位；默认合并规则下 ACTIVE 冻结卡可排在 Agent 提案前，且 `agent_proposals_kept=1`（`exploration_policy.py:47`，`fast_agent.py:1118-1143`）。这是模板挤占生成槽的代码通路。SEQ/DEPLOY 的 Slow **禁止**再往指导面写这个 marker。

### 3.4 可修改 surface 的目录、操作、权限；模板 ≠ 实例

`harness_surfaces.json` 注册 **13 个模板**（`schema_version: harness-surfaces/2`）。  
DEV-SEQ-2/4 与 DEV-DEPLOY-2 **实际开放 6 个模板**（`dev_seq2_knowledge.py:402-409`），不要和实例数混：

| 模板 | 操作 | 改观察 / 候选 / 选择 / 加载范围 | SEQ/DEPLOY 是否供给实例 |
| --- | --- | --- | --- |
| `skill_library.entries/{skill_id}` | ADD | 新指导正文 + 其 applicability（加载范围） | 始终作为 ADD 槽供给 |
| `skill_library.entries/{skill_id}.body` | PATCH | 已有指导正文（观察/候选/选择话术，取决于写什么） | **仅当已有非冻结 capability**。K0 只有冻结程序卡时 **0 实例** |
| `skill_library.entries/{skill_id}.observable_applicability` | PATCH | **只改加载范围** | 同上，K0 常为 0 |
| `bootstrap_skills.entries/{skill_id}.body` | PATCH | 三张 General 程序正文；bootstrap 无 applicability 过滤 ⇒ **每条序列都读到** | h0 有 3 个实例 |
| `candidate_policy.proposal_guidance` | PATCH | 提议阶段话术（controls，全程可见） | 1 个实例 |
| `candidate_policy.selection_guidance` | PATCH | 选择指导；DEPLOY 中有效选择直接影响交付，旧 online_loop 不保证如此 | 1 个实例 |

关闭（`FORBIDDEN_TEMPLATES`，`dev_seq2_knowledge.py:441-451`）：`instruction.core`、`retrieval.capability.top_k`、槽位数、`verification.*`、`risk_guards`、memory ADD。  
`edit_preflight` 对白名单外 fail-closed。

**「能影响」vs「历史上已测影响」**

| 面 | 能影响（代码通路） | 历史上已测影响 |
| --- | --- | --- |
| ADD 新指导卡 | 命中则进 Fast prompt；无冻结 steps 则不进候选池 | SEQ-2：两组都选 ADD，条件过严，**0/36 暴露**，效用未测。SEQ-3 g2：3/60 暴露，select 变、交付不变，全人群 delayed −0.0046 |
| PATCH 已有指导 body | 同检索路径 | SEQ-4 的 Slow 确实 PATCH 了正文；先前同起点选择翻转测试的是原指导，不是这次新正文的效果 |
| PATCH applicability | 只改谁能读到卡 | SEQ-2 为零暴露；SEQ-3 已有 3/60 暴露，不能把两包都归成相同检索失败。没有独立只改条件的效用归因 |
| PATCH bootstrap body | 每条序列的 inspect/propose/select 正文 | 本审计范围内 **未测**（SEQ-2 两组都没选这三张） |
| PATCH proposal_guidance | 候选构造话术 | **未在 SEQ/DEPLOY 实测选中** |
| PATCH selection_guidance | 选择话术 | **DEPLOY-2：g1 R/C 与 g2 R 都 PATCH 此面并暴露 40/40**；C 相对 F origin 面正差，但未过同批 D-new-fixed；且 240 行含空选择，不能称干净复现 |
| 冻结程序卡 / `supplies_candidates` | 机械进池、可挤掉 Agent 提案 | AUTO/KNOW 组级形状测过；**不得与逐序列 DEPLOY 并表**。SEQ 包禁止 Slow 再写冻结程序 |

原因标签：**由模型选中的 surface 机械反推**，不先裁定故障类（`cause_for_surface`，`dev_seq2_knowledge.py:586-600`；`CAUSE_FOR_TEMPLATE` 402-409）。一个 LLM edit 调用被记成 `SKILL_LIBRARY_GAP` / `SELECTION_MISS` 等，**不是**独立因果诊断。

### 3.5 Slow 实际何时触发、看什么、几种阶段、输出什么

当前至少 **三条几何**，不要并成一个 Slow：

**A. 方法层 `TTHAMethod.handle_feedback`（`method.py:426-630`）**  
触发：单条 Episode 的 Support gain < −M **或** `relation==CONFLICT`。  
看：`card_builder(episode)` 的 FailurePatternCard。  
阶段：一次 `slow_agent.propose_edit`（schema `slow_edit_v1`）。  
输出：`EditManifest`。若走 typed patch，Runtime **覆盖 body 为 `Frozen program steps:`**（`method.py:528-531`）。这是「换冻结算子」通路，不是 SEQ 的指导修订。

**B. DEV-SEQ-1/2/4 批次边界（`run_dev_seq2.py:knowledge_boundary` 829 行起）**  
触发：形成单元跑完、知识边界，不在批内。  
看：形成段 Episode → `group_failures`（workflow 指纹 × 符号）→ 最大组的 `evidence_card`（含对照成功、窗口特征、过程字段）。验证单元不在输入里。  
阶段：一次 Slow edit（attempts=2 的 schema 重试，不是多阶段诊断）。  
输出：ADD 或 PATCH 一处白名单面；父快照 fork 成子快照。

**C. DEV-DEPLOY-2（HEAD `run_dev_deploy2.py:539-607, 614-671`）**  
触发：DEPLOY-1 Part B 的 120 条冻结 Fast-only 决策已落盘后。  
看：**整批池化** `formation_cases`（不先按 workflow 分组）；R/C 带 origin/+48 效果；C-minus 去掉增益、赢家、风险结果。菜单参考在 HEAD 删除了 `per_series_gain`，注释却仍提到该字段（`_render_menu:518-534`）。  
阶段：每组三次独立 Slow 会话（R / C / C-minus），每次一处 edit。  
输出：编译候选或弃权。两组 C-minus 均为 `NO_UPDATE`。

Slow 不是「先聚类、再逐轨迹诊断、再提案」的流水线。SEQ 是规则分组后把一组材料交给一次 LLM；DEPLOY-2 是整批材料交给一次 LLM，比较式提示要求**模型自己挑一对子**。

### 3.6 每次更新是否先做错误聚类？四类必须严格分开

| 类型 | 是否存在 | 何处 |
| --- | --- | --- |
| **规则分组** | 有，在 SEQ | `group_fault.group_first_faults`：键 = (完整 workflow 算子序列指纹, NEGATIVE/CONFLICT 符号)，`min_group=2`（`group_fault.py:74-86`）。参数、特征、症状**不是键**，只作记录。孤例留在 `ungrouped_failures` |
| **LLM 逐轨迹诊断后再聚类** | **没有**（本项目当前 SEQ/DEPLOY） | 不存在 Self-Harness 那种 per-trace `terminal_cause` 再按三元组合并 |
| **整批池化材料** | 有，在 DEPLOY-2 | `build_card` 列出全部 formation_cases，「listing them together 不蕴含共同原因」（HEAD `run_dev_deploy2.py:547-554`） |
| **提示模型自行挑子群** | 有，在 DEPLOY-2 的 C | `_WHAT_YOU_MAY_WRITE["C"]` 要求点名一对/小组不同符号或不同交付（HEAD 422-437 行）。R 明确**不要求**比较结构 |

DEV-SEQ 取最大组（`run_dev_seq2.py:860-864`），并写明 Slow 可以改看子组或弃权。  
`fault_cases.py` 的五类选择题（任务理解 / 质量判断 / 供给缺口 / 选择错误 / Scope-风险）是**可选项屏蔽模块**；SEQ-2 把它们降为统计标签，**不对已授权面行使否决权**。HEC-1 等更早包用过三态启发式，与 SEQ 不是同一条。

同标签 ≠ 认同因：规则组只保证「同一算子序列 + 同一失败符号」，不保证同一机制。对照成功被放进 capsule，正是因为组本身不能解释为什么有的窗口同程序却成功。

### 3.7 修改是真修 Skill 还是换冻结程序？父子、失效、实证

| 通路 | 改什么 | 父子 | 失效 |
| --- | --- | --- | --- |
| SEQ/DEPLOY Slow | 指导正文 / applicability / bootstrap body / proposal 或 selection guidance | `EditController.apply_to_fork` 复制父快照再写一处（`edit_controller.py:783-831`）；父 SHA 保留；恢复按 SHA 重编译，不重问 Slow（DEPLOY-2 718-732） | 实验父分支保留。检索限制是另一条既有生命周期能力，不表示 DEPLOY-2 完成过线上晋升/撤销 |
| `handle_feedback` typed patch | Runtime 把 body 写成 Frozen steps | 同样 fork | 宽 applicability 的新卡强制 `requires_target_support`（`method.py:547-550`），进池但不优先 |

实证必须分形状，**AUTO/KNOW 与逐序列 DEPLOY 不并表**：

| 环节 | 逐序列 SEQ/DEPLOY 已有 | 组级 AUTO/KNOW | 未有 |
| --- | --- | --- | --- |
| 编译 | SEQ-2 两组 ADD 都编译出新 SHA | 有 | — |
| 暴露 | SEQ-2：0/36；SEQ-3 g2：3/60；DEPLOY-2 g1 R/C 和 g2 R：40/40，g2 C：8/40 | 组级广播，分母不同 | — |
| 实验分支 | SEQ 两臂 Fast；DEPLOY Fast-only F/R/C | 组级 A3/A5 | — |
| 晋升 / 独立有效性 | SEQ-2 因未暴露 **未测效用**；SEQ-3/4 选择有变、交付无变、分数无稳定增量；DEPLOY-2 C>F 但 < D-new-fixed，且空选择污染 | 旧 P4b：Support-B 全拒 ⇒ 0 Active Skill | 自然 pending→Support-B→versioned revision→独立重遇 **仍 HELD**（AGENTS §5） |

「生成→撤销」只算风险控制。完整持续进化需要局部冲突→有限改 Scope/参数/Workflow→独立重验→存活→后续改善；当前逐序列包还没有这条自然链。

### 3.8 现有改法可能遗漏了什么（每项定性）

| 缺口 | 定性 | 依据 |
| --- | --- | --- |
| 程序对照信息被压掉 | **已证缺陷**（输入层） | HEAD `_render_menu` 删除 `per_series_gain`，提示仍说字段存在（`run_dev_deploy2.py:518-570`）。总体相同、逐序列反号的菜单会被抹平。Worker A 授权修此输入，本调研不修 |
| 空选择记成 identity/0 | **已证缺陷**（读数层） | HEAD `run_dev_deploy1.py` 忽略 `prepare()` 的 `PreparationResult`；`fast_agent.py:1326-1352` FAILED 仍带非空 trace 返回。§15.2 / AGENTS §5.6 已锁定 |
| 局部观察 vs 全历史特征 | **设计选择 + 待测假说** | 特征在 `values[:origin]`（`public_features.py:265+`；`FEATURE_SEMANTICS` `dev_seq2_knowledge.py:1022-1035`）。服务窗是最后 192 点；修改区执行时才有。SEQ-2 条件用全历史分母导致 0 暴露，**未证明**改用局部量就会更好 |
| 生成剪枝（单步 actionability） | **待测假说**（本任务不改菜单） | `_actionable_operators` 用单步默认参数过 verifier（`fast_agent.py:209-259`）。可能剪掉「单步默认非法、组合合法」的算子。本调研未跑夹具；发现须报告最小例子、不得改规则 |
| Skill/Agent 合池截断 | **设计选择，已部分观测** | 默认保留 1 个 Agent 提案；冻结卡可占槽。240 行保留候选均未见多步分隔符（§15.2.5）——这是保留池描述，**不是**原始 LLM 提案审计 |
| 模板竞争（冻结程序 vs 生成） | **设计选择 / 待测假说** | 代码可达；SEQ 包禁止 Slow 再写冻结程序。是否应把模板只作生成参考，§15.3 列为 P1，**未在新入口上测** |
| 共享训练效应 | **已证约束，不是单点缺陷** | `per_sequence.py:39-48`：选程序 P 的序列由「用 P 准备的训练语料」拟合的模型服务。逐序列 loss 不能自动解释成只清洗该序列 |
| select 无交付否决权 | **设计选择，已证行为后果** | SEQ-3/4：指导经 select 起作用时不能阻止部署。12 条「选 identity 仍被改写」delayed 合计 +3.5842，门并非单向有害 |
| C-minus 弃权 | **观测，不是因果证明** | DEPLOY-2 两组缺效果字段则不修订；不能单独证明「有效学习必须有效果反馈」 |

不得把所有负结果收成「Memory 不够」或「只缺聚类」。

---

## 4. B2 Self-Harness 深读：它怎样提出有效修改

### 4.1 机制链（本地源码）

```text
失败 trace
  → tb2.py 抽 verifier 证据（规则：stdout / reward / 缺文件）
  → trace.py: 规则 terminal_failure_kind
       + LLM 逐步分析（terminal_cause, criticality, agent_mechanism, recovered flags）
  → integrated.py: 等 LLM 字段齐了再按三元组精确聚类
  → 渲染 diagnosis brief（含通过用例保护列表）
  → multi_proposer.py: K 个槽（默认 4），每槽改一个 hook；允许 evidence-backed decline
  → hooks.py 把虚拟 hook 打进 baseline 源文件的指定函数
  → materialize.py 落盘候选
  → 跑 train+heldout eval
  → acceptance_gate: 任一 split 平均 pass_rate 下降则拒；至少一 split 上升且无下降才接受
  → 多候选都过则 merge 后再 eval 一次（workflow/scripts/run_self_harness_loop.py:500-588）
```

### 4.2 必答题

**一条轨迹如何定位 terminal failure、恢复过的错误、关键行为？**  
混合。规则：`terminal_failure_kind` 扫 verifier 文本，分桶 `missing_required_artifact` / `missing_dependency` / `agent_timeout` / `verifier_runtime_error` / `verifier_assertion` / `reward_zero` / `unknown`（`trace.py:237-262`）。  
LLM：对每个 stage 产出 `incorrect_step_ids`、`terminal_cause`、`criticality`、`agent_mechanism`、`selected_step_recovered`（`trace.py:278-355`）。提示写明根因是「第一个不可恢复的关键失败」，恢复过的工具摩擦标 `recovered_friction`，**不要**把大 verifier 桶直接当 cause。

**聚类在解释之前还是之后？按哪些字段合并？同标签是否认同因？孤例？**  
**之后。** `build_verifier_causal_clusters` 先 `load_diagnosis`（必须已有 LLM 分析），再按 `(terminal_cause, criticality, agent_mechanism)` 精确相等分组（`integrated.py:41-62, 107-115`）。  
同标签：若 LLM 写出相同三元组，代码就把它们当成同一簇；**这不等于物理同因**，只等于「诊断字段一致」。论文 §3.2 也说同一 verifier 结果（超时、缺产物）可能需要不同 harness 改法，所以才加 agent_mechanism。  
孤例：size=1 的簇仍进入 brief；proposer 被要求「不要把大桶本身当成可修对象」，无把握则 no-op / decline。

**聚类如何变成提案 brief？成功案例如何保护？改哪个 hook、候选多少、怎样允许 no-op？**  
`render_verifier_causal_brief` 列出全部 passing case 作为回归保护（`integrated.py:81-88`）。proposer prompt 要求 `protected_passing_cases`、`expected_affected_cases`，且 `strict_noop=True` 时占位/未改提案非法，只能 `selection_decision=decline` 并写 `decline_reason`（`multi_proposer.py:56-103`）。默认 `route_count=4`，每候选 **恰好一个** hook（`hooks.py:94-95`）。机制族：prompt_instruction / subagent / skill_procedure / tool_configuration / middleware / runtime_control / permission_interrupt。

**可改面包含什么？不能直接推荐为本项目应开放范围。**  
Self-Harness 改的是 DeepAgent 实例化文件里的函数：system prompt、bootstrap/execution/verification/failure-recovery 指令、subagents、skills 源、tools、middleware、runtime 策略、permissions。那是**终端 coding agent 的运行时脚手架**。本项目对应的是时序数据准备的指导正文与适用条件，不是 shell 工具或 subagent。开放面不能照搬。

**回归如何比较父子、允许多大损失、held-out 是否反复用？**  
代码：`delta = candidate_avg_pass_rate - baseline`；`delta<0` ⇒ dropped；接受条件「无 dropped 且至少一 split improved」（`run_acceptance_gate.py:97-111, 131-137`）。**不允许任何 split 平均变差**（相对论文 Algorithm 1 的 Δ≥0 且 max>0）。held-out 每个候选都跑，是**改进环内部的回归门**，任务集合固定、会反复使用。  
**它的 held-out ≠ 本项目最终密封评价。** 本地 `eval/README.md` 的 21 条 heldout（含 `count_dataset_tokens`）在改进过程中对验收门可见分数；本项目 sealed / TARGET_HELD_IN / +144 在打开前不得当反馈。

### 4.3 有工件支撑的失败→修改→行为→结果例子

本地克隆 **没有** 各轮 `diagnosis.md` / `proposal_bundle.json` / `acceptance.json` 轨迹包，不能从本机复原某次迭代的完整因果链。能用的是：

**例子 1（论文 Figure 7 + 分列数字，机制级）：MiniMax × `count-dataset-tokens`。**  
失败：初始 harness 下持续探数据集、超时、不写 `/app/answer.txt`。  
修改：bootstrap 改为尽早创建产物；runtime 限制总 tool messages（论文 §4.3）。  
行为：改为算 token、写文件、再读回校验。  
结果：该任务在 **held-out 名单**（`Self-Harness/eval/README.md:51-75`）。论文用它作定性前后对照。定量上 MiniMax overall 42.2→53.9、held-out 40.5→61.9（v1/本地 PDF §4.2）。**不能**从本机工件核对这一次任务的 acceptance JSON。

**例子 2（论文 Figure 8 + 本地最终文件）：Qwen3.5 × `extract-elf`。**  
失败：写出 extractor 后反复 edit 失败，最后删掉 `/app/extract.js`，verifier 因缺产物失败。  
修改：依赖预检查、禁止原样重试、探索循环打断、缺产物后 1–2 步内创建。  
本地最终文件 `Self-Harness/harnesses/qwen_tb2_final/repo_baseline.py:12-79` 含这些句子（`verify required Python modules`、`Do not perform more than 3 consecutive explore`、`create that artifact within 1-2 steps`、`do not retry the exact same command`）。  
`extract_elf` 在 **train** 名单（同 README:20）。论文说该任务在编辑后恢复产物。  
限制：这是**最终 harness 文件 + 论文叙述**；没有本机逐步 acceptance 记录，不能声称已复原「哪一次提案、哪一个 cluster id」导致这一段 diff。论文还写 subagent/skill 分支后来因不再提升被丢弃（Figure 6）——最终文件里 `build_subagents`/`build_skills` 仍返回空列表，与「分支被弃」一致，但中间稿不在本克隆。

没有逐步工件时，**不编故事补齐**某一 round 的 cluster→hook→delta。

### 4.4 论文是否证明「错误聚类本身有效」？

**本轮核查未找到聚类独立贡献的证据。** Grok 在其读取的本地 PDF / v1 中未见「关掉聚类」消融；主助手核对的 v3 方法与总体结果也不足以隔离聚类贡献，不将这个有限查阅范围写成全论文不存在任何相关实验的断言。因此：**不得从总涨分断言聚类模块有效。** 源码显示聚类发生在 LLM 解释之后，用的是 LLM 自己填的字段。

---

## 5. 近邻：GEPA 与 AutoGuide（只深看这两种）

| 维度 | 本项目当前 | Self-Harness | GEPA | AutoGuide |
| --- | --- | --- | --- | --- |
| 改什么 | 指导正文 / 适用条件 / bootstrap / proposal\|selection 文本；SEQ 禁止冻结算子 | DeepAgent 脚手架函数（prompt、runtime、middleware、skills 源…） | `dict[str,str]` 提示组件（system / skills / memory 等），Pareto 种群 | 条件化自然语言 guideline 集合 + 按当前状态检索 |
| 何时看答案 | Fast 部署路径不看当格 Outcome；Slow 看已打开的形成反馈 | 改进环看 held-in traces；held-out 给验收门看分数 | 优化器在训练/开发任务上跑 rollouts，用分数+文字反馈 | 离线轨迹（含成败）生成 guideline；测试时按**当前状态**选 guideline，不是把测试答案写进提示 |
| 归因 | SEQ：规则组（程序×符号）；DEPLOY-2：池化或模型自挑对 | 规则 terminal 桶 + LLM 机制字段，**之后**精确聚类 | 对单次/小批轨迹做语言反思，不是先聚类 | 对比成功/失败轨迹，抽出「若状态 S 则做 A」 |
| 选择/接受 | 编译器 + 暴露检查 + 冻结 Fast 的 origin 主读数；+48 单列，非线上自动晋升 | 父子 pass_rate，split 不许下降 | 先在反馈 minibatch 上比父版，改善则入候选池并评 Dpareto；Pareto 过滤用于选择后续搜索父版，不是每次入池都要求 Pareto 支配 | 离线形成指导、测试看任务成功；本轮 Grok 未打开 PDF，主助手另核官方 HTML §3；不引第三方模块消融表 |
| 源 | 见第 3 节 | `trace.py` / `integrated.py` / `multi_proposer.py` / `run_acceptance_gate.py`；论文 §3 | arXiv:2507.19457；本地 `a-evolve/docs/algorithms/gepa.md:47-55` | arXiv:2403.08978v2 摘要与公开 HTML 表：ALFWorld/WebShop/WebArena；条件结构 guideline |

补充：

- GEPA 的「反思局部文本 + 外部选择」与本项目 Slow 改一处文本面相近，但它优化的是提示种群，用可教的文字反馈，不是 sMASE 标量一门。若本项目只有标量增益，GEPA 的样本效率主张**不能直接搬**（设计备忘录也写过这点）。
- AutoGuide 的「成败差 → 条件指导 → 按状态选用」最接近 Specific Skill + applicability。它用离线经验，测试时选 guideline 依据的是**当前观察状态**，不是测试标签。不得写成「所有其他领域测试时都在看答案」。
- 两者都不是时序数据准备 harness，都不能替代本项目的 Consumer / 训练几何。

---

## 6. 事实矩阵（每项带出处）

| 问题 | 本项目 | Self-Harness | GEPA | AutoGuide |
| --- | --- | --- | --- | --- |
| 知识结构 | `SkillEntry` + 散文 body；六段只在 Source 卡 `risk_guards.sections`（`source_skill.py:51`，`retrieval.py:158-168`） | 源文件内多个 `build_*` 函数（论文 Figure 3；`qwen_tb2_final/repo_baseline.py`） | 字符串键值提示 | 条件句 guideline |
| 检索 | applicability AST + top_k=2；bootstrap always（`retrieval.py:241-299`，`retrieval.json`） | 无时序特征检索；改完的函数对所有任务生效 | 组件 round-robin 突变 | 按当前状态选相关 guideline（论文摘要） |
| Fast 是否读 raw Episode | 禁止（AGENTS §4；`method.py` 注入的是对比包/签名，不是 bank） | proposer 读的是诊断 brief，不是 raw log 全文（`integrated.py:75-77`） | 反思看压缩轨迹 + 分数 | 生成阶段看离线轨迹 |
| 一次改几处 | 一处 surface（`edit_preflight`） | 一个 hook（`hooks.py:94-95`） | 一次突变一个组件（GEPA 文档 Select-Run-Reflect-Mutate） | 生成多条 guideline，测试时选若干 |
| 失败分组 | SEQ：程序指纹×符号（`group_fault.py:74-86`）；DEPLOY-2：不分组 | LLM 诊断后三元组（`integrated.py:107-115`） | 无强制跨任务聚类 | 按状态/上下文，不是 verifier 签名 |
| 接受/评价规则 | DEPLOY-2 是候选知识分支的冻结评价，origin 主终点、+48 单列；不等于自动晋升 | 无 split 下降且至少一 split 上升（`run_acceptance_gate.py:97-100`） | minibatch 改善后入池，Dpareto 用于后续采样与最终选择（GEPA v2 Algorithm 1–2） | 离线生成后靠测试任务成功率评估，不是父子 harness 门 |
| held-out 语义 | 项目正典：冻结后 Fast-only，打开即 EXPOSED（AGENTS §3） | 改进环内固定 21 任务回归集 | 论文任务 split，不是本项目密封面 | 论文 test split |

---

## 7. Surface → Fast 阶段 → 影响通路 → 例子 / 未测

| Surface | 到达 Fast 哪段 | 通路 | 已测例子 | 未测 |
| --- | --- | --- | --- | --- |
| bootstrap `inspect_and_localize.body` | inspect（也出现在后两段 system） | 改假设怎么写 | 代码可达；SEQ Slow **未选** | 改观察原则是否改变候选族 |
| bootstrap `build_contrastive_candidates.body` | 自称分节用于 inspect/propose/select | 改怎么构造 1–4 步 | 未在 SEQ/DEPLOY 被 Slow 选中 | 组合生成是否因这段而出现 |
| bootstrap `select_or_identity_and_verify.body` | select | 改选择指导 | 本审计所查更新未选中 | 在当前 Fast-only 路径下能否带来有用改变 |
| ADD capability 指导 | 三阶段 prompt，若命中 | 检索命中才读到 | SEQ-2 编译但 0 暴露；SEQ-3 g2 3 条命中，select 变 | 命中且改变交付后的效用 |
| PATCH capability body | 同 | 同 | SEQ-4 写过 PATCH；选择翻转对照发生在 PATCH 之前 | 新正文对后续冻结 Fast 的效用不能用原卡对照代替 |
| PATCH applicability | 不进 prompt，只改谁加载 | 加载范围 | SEQ-2 过严条件导致 0 暴露 | 局部窗条件 vs 全历史条件的外部验证 |
| `proposal_guidance` | controls，全程 | 候选菜单话术 | DEPLOY-2 **未选此面** | — |
| `selection_guidance` | controls，全程；意图在 select | 选择话术 | DEPLOY-2 R/C PATCH 此面，C 相对 F 正差但 < 固定基线；空选择污染 | 干净无故障下的重复 |
| 冻结程序卡 | propose 合池 + select 可见候选 | **执行体**，不是指导 | AUTO/KNOW 组级；SEQ Fast 仍可能读到 K0 冻结卡 | 新入口把模板降为参考后的生成质量 |
| `serving_scope` | 组级执行器 | 广播哪些序列走 prepared | 组级 Scope 实验 | 逐序列包已关闭此面 |

---

## 8. 当前错误聚类实际有 / 没有哪些环节

**有：**

- 失败 Episode 的机械定义（support 或 delayed < −M）
- 按完整算子序列 × 符号的规则组（SEQ）
- 组内对照成功 / conflict（capsule）
- 孤例保留为 Episode，不强迫进组
- 五类故障标签作统计，SEQ-2 起不否决修改面
- DEPLOY-2 比较式提示要求模型自己点名一对子
- 形成材料里的过程字段（候选、选择、交付、动作点数）——SEQ-3 之后才强制对照这些字段

**没有：**

- 先 LLM 逐轨迹写 terminal_cause，再按诊断字段聚类（Self-Harness 有，我们没有）
- 聚类作为提案 brief 的强制输入结构（DEPLOY-2 是平铺 120 行）
- 「同标签即同因」的机器保证
- 聚类模块自己的消融
- 把 Support 正 delayed 负自动写成唯一物理原因（AGENTS：只能证明效果没延续）

---

## 9. 最多三个设计选项（只建议，不执行，不预设必胜）

### 选项 1 — 最小输入修补

**做什么：** 消费 `PreparationResult` 状态，把 FAILED / 空选择 / 主动 identity / 合法 raw 回退分开；主机制缺决定记 UNKNOWN，不删行；菜单恢复同 UID/同窗口 P–Q 成对 `per_series_gain`，未测 UNKNOWN。C-minus 仍看不到效果字段。  
**能解决：** 已证的读数伪装和对照信息被抹平。  
**不能解决：** 条件化是否存在、共享训练几何、模型会不会生成有用的 Workflow。DEPLOY 已对齐选择和交付，无需把旧 online_loop 再修一遍。  
**最小可证伪：** 用夹具：FAILED+非空 trace 不得变 identity；总体相同、逐序列反号的菜单渲染后仍能分开；C-minus 卡中无赢家/增益泄漏。这是仪器测试，不是学习证明。Worker A 的授权范围正是这一项。

### 选项 2 — 轻量机制分组（材料整理，不授予因果）

**做什么（Grok 初稿选项）：** 用事先定义的键整理平铺材料，例如程序、响应方向、指导暴露；成功对照须匹配任务/程序/情境，**不能要求响应符号相同**。这只是规则分组，不应命名为 Self-Harness 式机制聚类。  
**Astra 的另一候选：** 先让 LLM 从轨迹提出可修改的行为假说，再汇总重复机制；保留成功对照和未知项。同程序可有不同机制，不同程序也可共享一次观察或构造缺陷。仅为讨论，不在修复包实现。  
**可能帮助：** 减少平铺材料的阅读负担，或更准确定位应改指导；不能事先断言解决了条件识别。  
**不能解决：** 分组正确 ≠ 修订有用；不能代替局部观察是否信息充分。  
**最小可证伪：** 同一执行器、同一模型、同一形成批次：平铺卡 vs 轻量分组卡，只比后续冻结 Fast 的 origin 效用与风险。预先规定：若行为不变，先查暴露/同义改写，而不是再加检索平台。

### 选项 3 — 结构化 Skill 的局部修订（仍是指导，不是冻结算子）

**做什么：** 不新增存储平台。允许 Slow 只改已有指导卡的一段（观察原则 / 构造建议 / 适用条件）或 ADD 一张 Specific 散文卡。General 继续用 bootstrap + guidance。禁止 `Frozen program steps:` 与 `supplies_candidates`。  
**已有部分与拟改部分：** 有限正文 PATCH、ADD 和条件修改已经开放，不能再次当作新机制。拟讨论的是让更新明确对应旧 Skill 的具体观察/构造条款；仍可复用现有整字段 PATCH，保持无关段落，不急于增加字段级编辑平台。  
**不能解决：** Fast 读不到局部作用事实时，结构化也不会变出信息；观察充分与知识有用仍须实测。  
**最小可证伪：** 同一 Fast-only 入口比较旧知识与一次修订，报告全人口 origin、风险、成本和过程，并列同反馈固定程序。若未改变交付，要区分无暴露、同义建议、合法性回退等；不能一律归为执行器覆盖。

三条都**先假设执行器已对齐**。修执行器的收益不得记成反馈学习。Consumer 若改为「固定模型、只处理本序列」，那是另一套干预几何，须单独命名（备忘录 §15.6）。

---

## 10. 仍未知项，以及需要 Astra 复核的精确问题

### 10.1 UNKNOWN / 未查明

1. Worker A 是否已在本审计之后改两个 runner：审计结束时工作区仍干净；若并行写入，历史数字仍以 HEAD 为准。  
2. DEPLOY-2 每条空选择的根因（中转 / 解析 / 模型 / 账户）：原始逐次错误 receipt 缺失（§15.2）。  
3. Fast 原始 LLM 提案是否写过多步：只审计了保留池与交付，全部单步（§15.2.5）。  
4. 「单步默认失败但两步合法」是否被 `_actionable_operators` 误剪：本任务不跑 Consumer fit，未做零 LLM 夹具。  
5. Self-Harness 某一 round 的 cluster id → 具体 hook → 该次 acceptance JSON：本地无逐步工件。  
6. Self-Harness / GEPA 是否有「去掉聚类/反思」的消融：论文与本地代码中**未见**聚类消融；GEPA 有相对 GRPO/MIPROv2 的总体比较，不是聚类消融。  
7. AutoGuide 官方实现是否与摘要完全一致：无本地源码。  
8. arXiv v3 的 SWE-bench / AppWorld 机制是否与本地 TB2 代码同一套 proposer：本地仓库未见那些基准的 harness 树。  
9. `_skill_prompt` 不展示 applicability AST，Fast 是否因此无法「按条件使用」一张已加载的卡：代码事实已知，行为后果未单独测。

### 10.2 请 Astra 复核的三处证据（也是最值得根 Agent 盯的三处）

1. **`agent_core.py:135-142` + `retrieval.py:278-289`：applicability 对象只用于检索，没有自动进入 Skill JSON。** 正文可能已复述条件，不能断言模型必然不知道；是否显式展示条件和匹配原因，是后续设计选项而非本轮确定缺陷。  
2. **旧 online_loop 的覆盖为真，但不再是最新 DEPLOY 阻塞。** 主助手已核 DEPLOY 直接消费 Fast 选择；本轮要修的是返回状态被忽略，不能再以“先让 select 有否决权”为后续前置任务。  
3. **Self-Harness 本地 PDF/v1 vs 公共 v3 vs README overall 表。** 机制核对应以本地源码 + v1/PDF 为准；v3 多了两个基准。论文 Figure 7 的 `count-dataset-tokens` 落在它自己的 held-out 名单上——那是**他们回归门上的定性例子**，不能当成「未参与改进的密封面」。若要把 Self-Harness 的 held-out 用法当本项目终验类比，这一条必须否决。

### 10.3 额外需裁定、但不阻塞本调研收口

- Consumer 作用对象：继续「每程序一条准备—训练—服务管线」，还是「固定 Consumer、只改本序列」（§15.6）。未决，但不阻止代码审计。  
- General/Specific 是否保持「无新 SkillKind」：源码已按此实现；若改为强制六段 schema，那是新协议，本调研不建议默认采用。

---

## 11. 审计时未跟踪路径（保留，不清理）

主目录已有未提交改动，本调研未碰：

- 已修改：`AGENTS.md`、`docs/DECISIONS.md`、`docs/SLOW_FEEDBACK_UPDATE_DESIGN_NOTES_2026-09-08.md`、两份 2026-09-03 Fable/HEC 文档  
- 未跟踪：`.dev_auto*_runs/`、`.dev_seq*_runs/`、`.dev_deploy2_runs/`、`.dev_know1_runs/`、部分 `artifacts/main_protocol/hec1_*.json`、`docs/FRAMEWORK_WALKTHROUGH_2026-09-09.md`、`docs/GROK_REPAIR_AND_ARCHITECTURE_REVIEW_TASK_2026-09-09.md`、`idea-stage/`、`refine-logs/`

唯一写入：本文件 `docs/SKILL_SURFACES_AND_SELF_HARNESS_AUDIT_2026-09-09.md`。

---

## 12. 交付状态

| 状态 | 是否成立 |
| --- | --- |
| 任务书已写 | 是（用户 / Astra） |
| 本调研已启动并完成指定输出 | 是 |
| 实现完成 | 不适用（B 不改代码） |
| 复核通过 / 正式项目结论 | **否**，待 Astra |

本次含研发模型调用，不能声称整包零模型成本。0 项目实验 LLM、0 新 Consumer fits、0 sealed 打开、0 git 提交。

主助手复核收口：核心架构问题已按本页开头纠偏；上述状态行指 Grok 初交稿，不代表
已经完成新的方法有效性验证。主助手补核原文：
[GEPA v2 §3/Algorithm 1–2](https://arxiv.org/pdf/2507.19457v2)、
[AutoGuide §3/Algorithm 1–2](https://arxiv.org/html/2403.08978v2)。
Grok 初稿对 GEPA “Pareto 改进才保留”的概括已更正；未复算论文实验。
