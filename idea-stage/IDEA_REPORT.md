# Idea Discovery Report

**Direction**: SelfEvolvingHarnessTS-deepseek-guidance-evolution — 在已有初始方向上，针对当前 first-fault 找可发表、可证伪的下一 idea。  
**Date**: 2026-09-07  
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline  
**Run id**: `skill-writeback-add-vs-patch-20260907`  
**Executor**: grok-4.6  
**AUTO_PROCEED**: true  
**Pilots**: 未跑。项目正典禁止未经授权的 LLM/fit；本轮只做文献与机制 idea，不发车任何课程。

## Executive Summary

**论文包装：放弃（mock NeurIPS 3/10）。仓库下一步：DEV-EDIT-TRANSFER-0 有界诊断（7/10），不是 live Agent 课。**

正典仍是 A5 = 经审计积累 + Target 校准；进化主张仍要「修订→存活→重遇」。idea-discovery 不得把该主张写成已证明。

路径压缩：

1. DEV-AUTO-3：供给付钱；PATCH 毁掉卡体。u23 祖先 +0.595 **未过** max-harm 0.30，不能当「被毁掉的安全知识」。
2. DEV-KNOW-1：不 PATCH 后知识被保留并检索（5/5），2 格实测、0 格部署。两张显式卡探测因 harmed_fraction 被拒——**不是**已证明的选择器故障。
3. 查新：ADD-not-PATCH 3/10 放弃独立方法文；Scope 天花板 3/10 放弃理论文；Idea 19 的 \(d_e\) 故事 5/10 谨慎推进，且 \(\arg\max d_e=\arg\max g_e(P')\)。
4. 外部评审（thread `01a079f4-4fdd-7291-a128-07cdaaa39fcc`，请求 gpt-6-astra/xhigh，精确 variant **未验证**）：拒绝当前方法文故事；留下 DEV-EDIT-TRANSFER-0。

本轮 **未授权、因此未跑** 任何 fit/LLM 课程。细化产出在 `refine-logs/`。

## Research Brief（Phase 0，来自本仓库）

权威顺序：`AGENTS.md` > `docs/STATE_ONE_PAGE_2026-09-03.md` > HEC/DEV-AUTO 收口文档。历史 `idea/` 与 8/23 前的 CURRENT_STAGE 不得覆盖。

**问题陈述。** 时序数据的“质量”随 Task / Consumer / 局部 Pattern 变化。Harness 应在 held-in 用真实下游反馈形成 Target-local Skill，冻结后 held-out Fast-only。A5 是产品臂；A3/Static 只做归因。

**已经成立的柱。** 条件化（同一修复对 forecast 与 AD 反号）和反馈验证（Support/delayed 双门、逐序列 harm、abstain）可写。尚未成立的是柱 Ⅳ 的进化曲线，以及柱 Ⅲ 在密封数据上的 A5 vs A3 效用差。

**当前 first-fault（2026-09-07）。** DEV-AUTO-3 三臂 × 16 单元 × 两组：

- B−A 两组同号（+0.039 / +0.032）：收益主要来自 **新增候选供给**。
- C−A 异号。C 的机制已定位：两次写回把 `[outlier_mad]` **就地覆写** 成 `[hampel_filter, winsorize]` → recalled_skill 部署失败 → **撤销整张卡**（祖先程序一起带走）→ 后续单元只能 identity，而 A/B 在同一格用 `outlier_mad` 拿到 +0.595。单格 −0.595 抵掉更新通道全部收益。
- PATCH 是单槽位就地覆写；失败不退回旧卡体。
- B 并非“没有持久化”：B 仍铸新卡并 recalled_skill，只是 **不覆写目标卡**。
- 状态页明确：**不建议在“卡片写回”方向再加实验组**；最该先改的性质是 ADD 而非 PATCH，但这会动编辑路由与生命周期语义，**超出当时授权**。

**约束。** 0 密封 / 0 +144 / 0 TARGET_HELD_IN；不改风险线、栅格、Consumer、DSL；不把 development 读数写成 Capability。Natural Final 仍封。本 idea 轮同样遵守。

**已排除 / 不再作为主 idea。** 单纯提高 LLM 调用、扩大算子菜单、在同一 6 个 origin / 12 折上拟合树 Targeter、把绝对屏换成相对屏（DEV-AUTO-1R 已离线否决）、把 HEC-1 的 `HEC1_EVOLUTION_NOT_SUPPORTED` 外推为「进化普遍无效」。

---

<a id="literature-landscape"></a>
## Literature Landscape

**检索纪律。** 本轮 `— sources: all, gemini` 的外部通道部分失败：arXiv API TLS 超时、Semantic Scholar 429、WebSearch 不可用、gemini CLI 不在 PATH。**实际贡献源**：项目本地调研（8/17、9/05）、OpenAlex DOI 核验（10 篇核心工作 `verified via openalex`）。本地 `papers/`、`literature/` 为空。Zotero/Obsidian MCP 未配置。

`WARN: local contributed nothing — no PDFs found in papers/, literature/, or a configured paper library.`

Sources contributed: **local-docs, openalex**.  
D2 外部 web/arxiv 未贡献；不把失败源伪装成已检索。

### 项目已有调研（不重复发明）

| 文档 | 能用的结论 | 不能外推 |
|---|---|---|
| `docs/RELATED_WORK_HARNESS_EVOLUTION_2026-08-17.md` | AegisTS = 固定算子顺序的层次 RL，奖励里「数据问题下降」权重大于下游性能；TimeClaw = 指纹检索 + 轨迹注入，evolution loop 在源码里就是 train 落盘 + test 检索；RewardHarness = Skill/Tool 的 create/modify/**deprecate**，接受权在 held-out val，**收益来自剪枝不是扩张** | 8/17 时本项目还只有 forecast-ridge 单任务；后续多任务接线已变，但结构对照仍成立 |
| `docs/SelfEvolvingHarnessTS：整体机制与研究定位文献调研（2026-09-05）.md` | 真正缺的是几何一致的 Evidence→Structural Revision；**探索安全 ≠ 部署安全**；优先路线 B（可证伪 procedural Skill）；DGM 证明低分 archive member 可以是 stepping stone | 9/05 时 DEV-AUTO-3 尚未发生；它没有把「PATCH 毁掉有效知识」写成可发表的最小实验 |

### 核验过的外部工作

| Paper | Year | Venue / status | Method | Key result (as claimed) | Relevance | Verification |
|---|---|---|---|---|---|---|
| Darwin Gödel Machine | 2025 | arXiv:2505.22954 | 自改代码 + **open-ended archive** | SWE-bench 20%→50%；低分个体可作 stepping stone | 直接反对「只保留当前最高分/当前卡体」 | ✅ OpenAlex |
| Self-Harness | 2026 | arXiv:2606.09498 | Weakness mining → minimal harness proposal → regression validation | held-out pass 多模型提升 | 结构修补最接近，但 evaluator 是可直接验证的 coding task | ✅ OpenAlex |
| Evo-Harness | 2026 | arXiv:2608.15071 | noisy context → reusable skill harness | 一次执行可编译为持续 harness knowledge | 与「经验编译为可修订 Skill」最近；反馈比 TS Consumer 强 | ✅ OpenAlex |
| AegisTS | 2026 | arXiv:2605.04902 | hierarchical RL for MTS cleaning | 下游感知的清洗 pipeline 优化 | **必须超越**的固定 operator policy learning | ✅ OpenAlex |
| RewardHarness | 2026 | arXiv:2605.08703 | Skill/Tool Markdown 库，create/modify/deprecate，val 超过历史最佳才接受否则整体回滚 | 空库 42.5%→62.5%；库 0→13→7，**剪枝后才继续升** | 写回/deprecate 的接受权不在 LLM；本项目 PATCH 无回滚 | ✅ OpenAlex |
| TTHE | 2026 | arXiv:2607.08124 | test-time population harness evolution，trace 作 proxy | unlabeled trace 可驱动 adaptation | 警告 proxy reliability；与「学完后未来窗口是否仍有效」不是同一 claim | ✅ OpenAlex |
| AFlow | 2024 | arXiv:2410.10762 / ICLR 2025 | 代码化 workflow + MCTS + execution feedback | 六 benchmark 平均 +5.7% | Workflow evolution 最直接；feedback 远比 TS Consumer 便宜 | ✅ OpenAlex |
| SPIBB | 2017 | arXiv:1712.06924 / ICML 2019 | 不确定区域 bootstrap 到 baseline | 安全策略改进 | 支持**部署** fallback，不支持禁止离线探索 | ✅ OpenAlex |
| Progressive Neural Networks | 2016 | arXiv:1606.04671 | 新列、冻旧列，侧向连接 | 顺序任务不覆盖旧权重 | ADD 而非 PATCH 的经典类比 | ✅ OpenAlex |
| Experience Replay for Continual Learning | 2018 | arXiv:1811.11682 | 保留并回放旧经验 | 缓解 catastrophic forgetting | B 臂「祖先卡留在旁边」的学习理论对照 | ✅ OpenAlex |
| Continual lifelong learning (survey) | 2019 | Neural Networks | 综述 | 覆盖是持续学习的核心失败模式 | 把 PATCH 命名为 Skill-level forgetting | ✅ OpenAlex（检索命中） |

### 分组

1. **Agent/Harness 结构进化（ADAS/AFlow/DGM/Self-Harness/TTHE/Evo-Harness）**  
   共同假设：候选可反复、便宜地评价。本项目的 delayed full-Consumer 反馈是最独特也最贵的一环。它们几乎都不处理「一条被接受的结构修改会删除一条仍在赚钱的旧结构」。

2. **时序数据准备（AegisTS、ActiveClean、FFORMA）**  
   优化的是固定算子策略或 numerical router。AGENTS.md 明确禁止项目收缩成这一形态。AegisTS 是必须写出的 closest competitor，不是可抄的答案。

3. **Skill 库的接受与回滚（RewardHarness）**  
   进化对象是人可读 Skill/Tool；接受权在独立 val；失败则**整体回滚**。本项目 C 臂失败后不是回滚到祖先，而是撤销整张卡。这是最硬的机制对照。

4. **持续学习中的覆盖（PNN、Experience Replay、EWC、SPIBB）**  
   新知识用新容量 / 回放 / 不确定区回到 baseline。没有一篇把它表述成「时序 Data Readiness Skill 卡的 ADD vs PATCH」。这是本轮的空隙。

### 结构空隙（给 idea 用）

- 方法已在持续学习/Agent archive 成立、未在 **Consumer-grounded TS Skill 卡** 上做过：ADD-not-PATCH、开放 archive、失败回滚到祖先。
- 文献共识（DGM、RewardHarness）与本项目实现（单槽 PATCH + 连坐撤销）矛盾。
- 人人默认「写回 = 进化」，DEV-AUTO-3 显示写回可以是负的——这个假设几乎没被当成可证伪主实验。
- 16 单元 development 课上供给已付钱；HEC 长度课程上 ADD 是否仍优于 PATCH，是未探索的尺度。
- 几乎没有工作问：**一次 Skill 更新的 credit 有多少来自供给、多少来自卡体、多少来自召回。** DEV-AUTO-2/3 已经把仪器搭好。

📚 Literature survey complete.

AUTO_PROCEED: selected **「Skill 卡 ADD-not-PATCH + 祖先保留」** as the top-ranked direction (aligns with DEV-AUTO-3 next-step and 9/05 Route B). Continuing to Phase 2.

---

<a id="ranked-ideas"></a>
## Ranked Ideas

生成方式：

- Round 1：五透镜 + Consumer-geometry 顺序枚举（执行者 grok-4.6），得到 Idea 1–10。
- Round 2 brainstorm（Codex MCP）：thread `01a079b7-be13-79f3-86ef-de0c3ff90918`（请求 `gpt-5.6-sol`，回报 GPT-5 家族、精确 variant 未暴露）→ 10 条；thread `01a079c5-406f-78a3-819e-164187d908bc`（请求 **`gpt-6-astra`**，回报 “Codex based on GPT-6”，精确 variant / xhigh **未暴露**）→ 9 条。Trace：`.aris/traces/idea-creator/2026-09-07_run02/receipt.json`。
- 机械去重：Astra「effect-diverse proposals」并入 Idea 14；Astra「challenge-window probes」并入 Idea 16。其余保留。生成 ≠  acquittal；跨模型 triage / novelty 另走。

未跑 GPU/fit pilot。

### Idea 1: ADD-not-PATCH Skill lineage — 写回语义线（须另批）

- **一句话**：把 Skill 更新从「就地覆写单槽卡体」改成「新增一张卡，祖先卡保留；新卡独立验证失败则停用新卡，不撤销祖先」。
- **Hypothesis**：C−B 的负差主要来自 **destructive overwrite + cascade revoke**，不是来自「不允许结构更新」。若只改这一条语义、其余供给/门/预算不变，则 C′（供给+ADD）应 ≥ B，且不再出现 u23 那种 −0.595 连坐。
- **MVE**：在 DEV-AUTO-3 同一 16 单元 × 两组几何上重跑三臂：A 控制 / B 供给-only / C′ 供给+ADD。0 密封。只改编辑路由：PATCH→ADD，revoke 不得删除祖先 program。预算与 DEV-AUTO-3 同数量级（约 600 LLM / 1500 fits 量级，须另批）。
- **Contribution**：new method（Skill 生命周期语义）+ diagnostic（若 C′ 仍不优于 B，则「Skill 自我更新」claim 应降级）。
- **Risk**：MEDIUM。正负都有论文价值。实现会动编辑路由，这正是 9/07 状态页说的「超出当时授权」。
- **Effort**：weeks（实现+对照课），不是 months。
- **Pilot**：needs authorized pilot。
- **prior_work**：Progressive Neural Networks（冻旧加新）；RewardHarness（失败整体回滚，不是删库）；DGM（archive 保留非当前最优）；Experience Replay。没有一篇在 TS Data-Readiness Skill 卡上做过 ADD vs PATCH 的等预算三臂。
- **so_what**：这是当前唯一能把「进化曲线」从被 PATCH 污染的仪器里救出来的最小改动。

### Idea 2: Credit-split of supply vs update vs recall — BACKUP / 必做伴随诊断

- **一句话**：任何写回实验必须预注册三通道归因：`resupplied_draft` / `recalled_skill` / Fast 自搜，禁止再用整段效用差宣称「Skill 更新有效」。
- **Hypothesis**：DEV-AUTO-2 已显示 12/16 格两臂逐位相同、供给 +0.68 vs 更新后卡 +0.14。ADD 改革之后，若更新通道仍然 ≈0，则论文主 claim 应是 **candidate supply with persistent ancestors**，不是 self-revision。
- **MVE**：复用 DEV-AUTO-3 的 per-cell 部署日志，不新跑课程即可先出标准表；新课只是把同一张表做成合同字段。
- **Contribution**：diagnostic。
- **Risk**：LOW。
- **Effort**：days。
- **Pilot**：0 extra fit if logs retained；否则 needs authorized replay。
- **prior_work**：无直接竞品；进化论文普遍不报「每次有效修改对应几次 proposal / 来自哪条通道」。
- **so_what**：没有它，Idea 1 的正结果仍可能被说成供给。

### Idea 3: Ancestor-bootstrap under uncertainty (SPIBB for Skill cards)

- **一句话**：新卡在 delayed 门不确定或未过时，部署 **bootstrap 回祖先卡**，而不是 identity，更不是撤销祖先。
- **Hypothesis**：u23 的灾难不是「新程序差」，而是「无卡可用只能 identity」。祖先 fallback 应回收大部分 −0.595。
- **MVE**：在已有 C 臂轨迹上做 0-LLM 反事实：凡 recalled_skill 失败的格子改放祖先 program，重评 delayed。若反事实已接近 B，则连 ADD 都可以先不做，只做 fallback。
- **Contribution**：method / diagnostic。
- **Risk**：LOW–MEDIUM（可能被批成工程修补）。
- **Effort**：days–week。
- **prior_work**：SPIBB；RewardHarness rollback。差异：这里 fallback 的是 **同一谱系祖先 Skill**，不是任意 baseline policy。
- **so_what**：把「安全部署」从「删知识」改成「不确定则用已知卡」。

### Idea 4: Open Skill archive as exploration lineage (DGM 移植到卡，不移植到整份 Harness 源码)

- **一句话**：失败/被拒/未部署的合法程序进入开放 archive，Fast 可以再供给，但只有独立 delayed 过门的才进入可部署集。
- **Hypothesis**：DEV-AUTO-1 被绝对屏杀掉的候选里，有的 delayed 实际优于祖先。开放 archive 提高 future headroom，而不增加部署 harm。
- **MVE**：离线把 DEV-AUTO-1R 框架 C 的 17 条提议当 archive，看后续单元若允许 resupply 这些「未晋升」程序，delayed 是否升。先 0 新 LLM。
- **Contribution**：empirical finding on archive vs single-slot。
- **Risk**：MEDIUM（与 Idea 1 部分重叠；若只做 archive 不做 ADD，可能仍被 PATCH 污染）。
- **Effort**：weeks。
- **prior_work**：DGM archive；ADAS archive。必须写清：不开放整个 Harness 自改源码（路线 A），只开放 **Skill 程序档案**。

### Idea 5: Skill as falsifiable procedure, not overwriteable program list

- **一句话**：卡的本体改成 9/05 的 Skill 记录：WHEN/OBSERVE/HOW/VERIFY/CONTRAINDICATION/EVIDENCE/FALSIFIER。反馈做 Revise/Split/Retire，禁止用新 program 列表覆盖旧证据。
- **Hypothesis**：当前卡体是 program 列表，所以 PATCH 在类型上就只能覆盖。若证据与程序分离，覆盖程序不必删除证据，Retire 也不必连坐。
- **MVE**：先在一张 `outlier_mad` 卡上手工改 schema（development），不接 LLM；用既有 DEV-AUTO 轨迹填 evidence refs，看 Fast 检索/部署是否仍能工作。
- **Contribution**：new method（路线 B 的最小落地）。
- **Risk**：HIGH（schema 一动就转 runtime bundle lock；AGENTS 反过度工程）。
- **Effort**：months if full；weeks for one-card prototype。
- **prior_work**：Evo-Harness；RewardHarness Skill markdown。本项目 9/05 已推荐此路线，但还没有对着 PATCH 失败做过最小实验。

### Idea 6: Serving-side scoped pipeline as the true intervention unit

- **一句话**：在 ADD 之前，先保证 Scope 外序列与 Static 逐位相等（AGENTS §5.2 已裁定、仍是主实验几何）。否则任何 Skill 进化 claim 都可能仍是训练语料策展。
- **Hypothesis**：若 serving-side 未接通，ADD vs PATCH 的读数不能支持「Target-local Scoped Harness 进化」。
- **MVE**：0-LLM：对一条已部署卡，比较 Scope 外序列的 serve context 是否等于 raw。这是仪器，不是新 idea 本身。
- **Contribution**：instrument / necessary premise。
- **Risk**：LOW scientifically, HIGH politically（可能被当成「又在修仪器」）。
- **Effort**：已有机械演示 `p4n_serving_side_gap.json`；缺的是正式主实验，不在本 idea 轮擅自开。
- **so_what**：Idea 1 的论文必须声明 Consumer geometry；否则审稿人会引用 §5.2。

### Idea 7: Equal-budget Target-only vs Source-assisted after ADD works

- **一句话**：跨域 Skill 的最低对照是同一 Target budget 下 Target-only vs Source-assisted（负迁移文献）。现在 A5 常空，是因为写回把知识写坏，不是因为积累无用。
- **Hypothesis**：在 ADD 语义下若能保住 K0，A5−A3 才会第一次变得可测；在 PATCH 下测 A5 是无效实验。
- **MVE**：不作为本轮第一实验。只有 Idea 1 出现存活重遇链之后才立项。
- **Contribution**：the actual A5 claim。
- **Risk**：HIGH until Idea 1 lands。
- **Effort**：the HEC-scale course；months。
- **prior_work**：Wang et al. negative transfer；项目自己的 NOAA 69 vs 123 只是效率，不是最终效用。

### Idea 8: Cheap screen vs expensive delayed — do not promote on Support

- **一句话**：把 Support 只当 screen，把 delayed 当唯一 promotion 面；被 screen 淘汰的候选做抽样 full-eval，检验 proxy 排序。
- **Hypothesis**：HEC-1 诊断已说 Support-safe 的 34 个候选只有 10 个保持安全。若 promotion 仍含 Support 口径，ADD 也会把未来窗口不安全的新卡写进库。
- **MVE**：0-LLM 重放 HEC-1 / DEV-AUTO 候选的 Support vs delayed 列联表。
- **Contribution**：diagnostic，Idea 1 的门设计约束。
- **Risk**：LOW。
- **prior_work**：TTHE proxy reliability；项目 HEC-1 zero-LLM diagnostic。

### Idea 9: Do not treat prune-as-evolution until ADD exists

- **一句话**：RewardHarness 的收益来自 **deprecate/剪枝**。本项目 C 臂的 revoke 看起来像剪枝，其实是 catastrophic delete。先区分「有祖先的 Retire」和「无祖先的 wipe」。
- **Hypothesis**：在 ADD 库里做有界 Retire（只下线被独立否证的那张新卡）会升；在 PATCH 库里做 revoke 会降。同一「剪枝」词，符号相反。
- **MVE**：语言/合同层即可先写进 Idea 1 的预注册，不必单独一课。
- **Contribution**：conceptual clarification。
- **Risk**：LOW。

### Idea 10: HEC-length scaling of ADD vs PATCH

- **一句话**：16 单元课上供给已够用；40+ 单元上祖先冲突累积，ADD 的卡库膨胀可能重新引出 RewardHarness 那种「先膨胀后必须剪枝」。
- **Hypothesis**：存在一个课程长度，超过它 ADD-without-retire 开始负。
- **MVE**：只有 Idea 1 在 16 单元上 C′≥B 之后才值得加长。现在做是浪费。
- **Contribution**：scaling-regime empirical。
- **Risk**：HIGH cost。
- **Effort**：months。

### Round 2 追加（HEC-1 / D5 / M-R0；gpt-6-astra + GPT-5 Codex）

以下编号接 Idea 10。`source` 标明首次提出的线程。Astra 自评 Top3 是 21 / 25 / 13；GPT-5 线程自评 Top3 是 19 / 12 / 15。执行者 **不**在评审前淘汰。

### Idea 11: Executable Evolution Liveness Witness — 0-LLM 首选诊断

- **Method**：用最小合法事件（Positive / delayed-Adverse / FLAGGED / 丢部署权）重放真实实现，看 `ADD/NARROW/REVISE/REVOKE` 能否走到可执行行为差。
- **Hypothesis**：HEC-1 的 0 修订链有一部分是生命周期不可达（缺 `relation`、delayed 不入 bank、FLAGGED 无结构入口、REVOKE 不可达），不是 Agent 不会进化。
- **MVE**：0 LLM / 0 fit 轨迹重放。路径通了 → 剩余问题在提案质量；仍不通 → 语义不可达。
- **Contribution**：diagnostic。**Risk**：LOW。**Effort**：days。
- **first-fault**：M-R0。**dedup_key**：`executable_evolution_liveness_witness`。**source**：GPT-5 thread。
- **so_what**：这是所有「结构修订」idea 的前置。修接线前再开 Agent 课会重复 HEC-1 的假阴性。

### Idea 12: Observation-equivalence actionability bound

- **Method**：按**可执行**观察/谓词签名（Task×Consumer×program geometry，不用 dataset ID）把单元分等价类，在类内算安全 oracle 上界，与 Best-Safe-Global / identity 比较。
- **Hypothesis**：`+5.527` 的 outcome-side headroom 大部分在当前观察语言下不可行动——同类签名里未来安全符号相反。
- **MVE**：exposed logs，0 LLM。上界贴近 BSG → 问题在供给/选择；上界贴近 identity → 先改 Observation，而不是再拟合 Targeter。
- **Contribution**：diagnostic。**Risk**：MEDIUM（划分本身可被作者做粗）。**Effort**：days。
- **dedup_key**：`observation_equivalence_actionability_bound`。**source**：GPT-5；Astra 的 Scope 表达力天花板（Idea 13）是相邻但更窄的语法问题。

### Idea 13: Scope-language expressivity ceiling

- **Method**：在冻结谓词文法上，用已缓存逐序列效应做分析员-only 枚举，比较「该文法能表达的最好安全子集」与无约束 outcome-side 上界。
- **Hypothesis**：有些益/害案例在允许的 Scope 词表下不可区分；`CALIBRATED 0/5` 和 `NO_FEASIBLE_STUMP` 可能是语言天花板，不是校准器坏了。
- **MVE**：days，0 LLM，不装 Scope、不拟合。
- **Contribution**：diagnostic / theory。**source**：Astra。**dedup_key**：`scope-language-expressivity-ceiling`。
- **Reviewer 风险**（Astra 自报）：「受限文法的 oracle 不能约束能读更丰富 context 的 Agent」——必须钉死信息类边界，并给出 constructive collision。

### Idea 14: Effect-space-diverse Fast supply

- **Method**：在固定候选名额内，按**已执行**的逐序列修改足迹去重，而不是按 DSL 拼写去重。
- **Hypothesis**：Fast 漏掉 Support-safe 程序，是因为语义重复提案吃掉名额。
- **MVE**：先 0-LLM 看足迹冗余；若冗余大，再等预算对照「普通提案 vs 足迹多样化」。候选宇宙不变，不加算子。
- **Contribution**：method。**Risk**：MEDIUM。**Effort**：days–weeks。
- **first-fault**：17 个有 Support-safe 程序的单元，A5 只覆盖 8/7/7。
- **dedup_key**：`effect_space_diverse_fast_supply`（含 Astra `observable-effect-diverse-proposals`）。

### Idea 15: Route–context Pareto admission

- **Method**：把 scoped 干预拆成 context-preparation 与 Consumer-route 两个效应，用 mean–tail Pareto 而不是标量 full-pipeline gain 准入。
- **Hypothesis**：D5 显示路由既是 8/10 严重伤害主因（约 73% 幅度）也常是收益管道；标量分数把可分的权衡藏起来。
- **MVE**：复用 D5 四格。identity / context-only / route-only / full，风险预算不变。Pareto 改善 → 因子化准入；失败则「该切片上收益与尾部不可分」是有用负结果。
- **Contribution**：method。**Risk**：MEDIUM。**Effort**：weeks。
- **dedup_key**：`route_context_pareto_admission`。**source**：GPT-5。
- **Reviewer 风险**：`ROUTE_DOMINANT` 可能是 pooled Ridge 制品；把 route 当决策会被说成模型路由而不是 Data Readiness。

### Idea 16: Disagreement-directed / observable-challenge held-in probes

- **Method**：固定反馈预算，把 probe 花在可执行候选 **预测/修改足迹分歧最大** 的合法 held-in 窗，而不是固定近期窗。
- **Hypothesis**：当前 Support 窗没暴露 +144 才出现的候选特异尾部风险。
- **MVE**：顺序重放 exposed 表面，用揭晓前分歧选下一 probe，再打开已有 outcome。等次数、等 fit 比未来安全保留率。
- **Contribution**：method。**Risk**：HIGH（大分歧也可能是大收益）。**Effort**：weeks。
- **dedup_key**：`disagreement_directed_heldin_probes`（含 Astra `observable-challenge-probe-selection`）。
- **与 Idea 8 的差**：Idea 8 改的是 promotion 哪一面贵；这里改的是 **held-in 证据采在哪**。

### Idea 17: Affordance-keyed Skill recall

- **Method**：检索经审计 Skill 用「可观察 Pattern × Consumer geometry × 可执行干预足迹」，不用 provenance / dataset 名 / 表面散文。
- **Hypothesis**：`recalled_skill` 3/1/0 是键不匹配，不是卡真的用不上。
- **MVE**：0-LLM leave-cohort-out，冻结卡、同一 k。覆盖升 → 检索修复；不变 → 卡本身不适用。
- **Contribution**：method。**Risk**：MEDIUM。**Effort**：weeks。
- **dedup_key**：`affordance_keyed_skill_recall`。

### Idea 18: Rolling worst-series transfer certificate

- **Method**：对 34 个 Support-safe 候选，在 +144 之前的滚动子窗上看 worst-series 效应稳定性，用现有 0.30 线预注册证书，不调新阈值。
- **Hypothesis**：聚合 Support 安全掩盖了时间上不稳的单序列效应，这解释 22/24 未来失败撞 `single_series_harm`。
- **MVE**：weeks，重评已有候选。有用的 coverage–safety 前沿 → 证书；失败 → 当前 held-in 历史预报不了主导尾部事件。
- **Contribution**：diagnostic。**dedup_key**：`rolling_worst_series_transfer_certificate`。

### Idea 19: Delta-stable one-edit workflow repair — 进化链科学对象

- **Method**：冲突后只允许 **一步** 合法 Workflow 修改（替换算子 / 增删一个 impute 步 / 换序 / 改一个合法参数）；学习的是 \(d_e(P\to P')\) 的稳定性，不是绝对效用 \(g_e(P)\)。
- **Hypothesis**：局部修改效应比绝对程序效用更跨面、跨重遇稳定，这才能形成 revise→validate→survive→reencounter。
- **MVE**：先 0-LLM 枚举冻结 DSL 的一步邻域，用冲突时可见证据选，下一独立面验证，更晚重遇计分。零 headroom 关闭该编辑面；有则再授权 DEV-AUTO-3 量级 Agent 课。
- **Contribution**：method。**Risk**：HIGH。**Effort**：weeks。
- **first-fault**：FLAGGED 10/11 无结构入口；阈值 CALIBRATED 0/5；修订链 0。
- **dedup_key**：`delta_stable_one_edit_workflow_repair`。**source**：GPT-5 自评 #1。
- **Reviewer 风险**：若不证明 edit delta 能迁到独立重遇且优于等预算新鲜搜索，会被收成受限的 AFlow/AegisTS 局部搜索。
- **纪律**：与 Idea 20 是 **竞争的编辑面**，单假设下 0-LLM 可行性之后最多选一个。

### Idea 20: Counterexample-guided one-clause Scope revision

- **Method**：delayed 冲突后只改 **一条** 可观察 Scope 谓词，排除被牵连的 pattern 区；Workflow 与 Consumer route 不动。
- **Hypothesis**：严重伤害在冻结观察词表里足够局部，一条最小、无 UID 的 Scope 编辑能保住受益者并过独立再验。
- **MVE**：0-LLM 时序可行性：冲突在 t 提供反例，t+1 独立验证，更晚匹配单元测重遇。
- **Contribution**：method。**Risk**：HIGH。**Effort**：weeks。
- **dedup_key**：`counterexample_guided_one_clause_scope_revision`。
- **注**：NEXT_EXPERIMENT 计划把 Workflow 组合当优先候选，是因为 Scope 路径从未真正测到。Idea 11 解锁接线后，应先看原 NARROW 是否可达，再决定开 Idea 19 还是 20。

### Idea 21: Forecast-displacement harm certificate — Astra 自评 #1

- **Method**：用 raw vs prepared 管道的预测位移，给出 **不看未来 outcome** 的单序列 harm 上界。对过去季节尺度 \(q_i>0\)：
  \[
  \bigl|\mathrm{sMASE}_i(P)-\mathrm{sMASE}_i(0)\bigr|
  \le \|p_i^P-p_i^0\|_\infty / q_i.
  \]
- **Hypothesis**：有些有用 Workflow 的预测位移足够小，可以在 Support→未来迁移不稳时仍证尾部门。
- **MVE**：先推导并在保留的预测上检验证书覆盖、保留效用、弃权率。空洞上界 → 关掉这条数值风险指导。
- **Contribution**：method / theory。**Risk**：HIGH（三角不等式 + 过度保守）。**Effort**：weeks。
- **dedup_key**：`forecast-displacement-harm-certificate`。**source**：Astra。
- **Reviewer 风险**（Astra 自报）：「这是包着保守弃权的三角不等式」——必须保住有用 treatment 并促成可验证修补，不等式本身不够。

### Idea 22: Training-target vs training-context inside the route term

- **Method**：固定程序/Scope/pooled Ridge/serve 处理，只拆训练侧 `{raw, prepared context} × {raw, prepared target}`。
- **Hypothesis**：D5 的 route harm 有一部分来自 **准备了训练目标**（评价仍奖 raw outcome），修训练 context 可能保留收益。
- **MVE**：约 40 fits、0 LLM、四训练 cohort × 两面。
- **Contribution**：diagnostic。**Risk**：MEDIUM（可能只是普通预处理问题）。**Effort**：weeks。
- **dedup_key**：`training-target-route-attribution`。**source**：Astra。

### Idea 23: Released delayed feedback necessity for Slow

- **Method**：同一可达接口、同一预算，Slow 输入一组只有 Support+冲突旗，另一组加上 **已经合法释放** 的 delayed 响应。
- **Hypothesis**：冲突旗 + Support 历史不够；真正解释冲突的 delayed 证据从未进入整合。
- **MVE**：days 做输入重放；weeks 做独立验证。改善 → 把合法证据送全；不改善 → 问题在观察或编辑 headroom。
- **Contribution**：empirical。**Risk**：MEDIUM（「更多相关反馈有用」单独不够新）。
- **dedup_key**：`released-delayed-feedback-necessity`。**source**：Astra。
- **与 Idea 11 的差**：11 问路径是否可达；23 问 **可达之后** delayed 内容是否改变 Slow 行为。

### Idea 24: Tail-risk vs treatment-exposure identity

- **Method**：固定逐序列增益向量与风险阈值，用有限总体组合公式把「多处理几条序列」从「每条响应变差」里拆开。
  \(P(\text{no severe harm}\mid k)=\binom{m-h}{k}/\binom{m}{k}\)。
- **Hypothesis**：安全通过率下降有一部分是覆盖变大的机械后果，不是效应更不稳。
- **MVE**：days，0 LLM。暴露能解释大部分下降 → 改覆盖决策并做 coverage-matched 比较；匹配规模后仍恶化 → 真不稳。
- **Contribution**：empirical / diagnostic。**dedup_key**：`tail-risk-treatment-exposure`。**source**：Astra。

### Idea 25: Component-feedback-guided one structural edit — Astra 自评 #2

- **Method**：自然冲突后只许一步结构修改；两组看到 **同一套** 四格损失，一组额外看到显式 D5 分解。
- **Hypothesis**：聚合失败反馈让 Agent 改错机制；分量反馈让选择的修改更有效。
- **MVE**：小冻结冲突集；独立验证 + 更晚重遇。必须把「没有 edit headroom」和「选错编辑」分开。
- **Contribution**：method。**Risk**：HIGH。**Effort**：weeks。
- **dedup_key**：`component-feedback-guided-structural-edit`。
- **Reviewer 风险**：会被说成 prompt 格式，或用额外反事实反馈买优势——两组必须同一损失、同一记账。
- **依赖**：Idea 19 或 20 的编辑面先有 0-LLM headroom。

### Idea 26: Preparation prefix information asymmetry

- **Method**：0-fit 前缀不变性：固定训练 context，只改暴露的 target 后缀，比较准备后的前缀是否依赖 serving 时不可见的信息。
- **Hypothesis**：program-induced route 不稳，有一部分来自训练输入准备时看了 serving 没有的后缀。
- **MVE**：days 筛查；若有超额依赖，再小 fit 只改训练输入准备、保留原 prepared targets。
- **Contribution**：diagnostic。**dedup_key**：`preparation-prefix-information-asymmetry`。**source**：Astra。

### Idea 27: Order-robust two-timescale consolidation

- **Method**：把置换不变的累积能力 与 顺序敏感的 Target-local 适应 分开编译。
- **Hypothesis**：1/3 顺序才有终点物质差、62/69 行为平局，是因为把稀疏顺序触发和可交换证据混在一个状态里。
- **MVE**：同一审计证据多重集走全部已记录顺序，比现行编译 vs 两时间尺度规则。
- **Contribution**：theory。**Risk**：HIGH。**Effort**：weeks。
- **dedup_key**：`order_robust_two_timescale_consolidation`。**source**：GPT-5。

AUTO_PROCEED: 不把 Idea 1 当唯一推荐。0-LLM 先做 **11 → 12/13 → 24/26**；写回语义仍是 **Idea 1**（须另批）；进化链科学对象是 **Idea 19 vs 20** 二选一。Continuing to novelty notes + Codex triage.

## Eliminated / deferred as primary

| Idea | Why not primary now |
|---|---|
| 再跑一组「卡片写回」且仍用 PATCH | 状态页 9/07 明确不建议 |
| 接受口径绝对屏 vs 相对屏 | DEV-AUTO-1R 离线已否决 |
| 换算子菜单 / 第七程序 | first-fault 不在 Program space |
| 开环树 Targeter | AGENTS §5.1 禁止在原 6 origin/12 折上继续拟合 |
| 路线 A：整份 Harness 自改源码 | 9/05：Agent 身份最强但成本最高；当前阻塞在卡语义，不在缺自改代码 |
| 路线 C：只做 set-level 训练池 workflow | 会放弃 Target-local Skill 主张；与项目正典冲突 |
| CATE / R-learner 作为答案 | 9/05：geometry 未对齐前不能当研究答案 |

AUTO_PROCEED: Round-1 曾选 Idea 1。DEV-KNOW-1 + 跨模型查新后，**论文主贡献改为 Idea 19（收窄）**；Idea 1/13 降为诊断。Continuing to research-review.

---

<a id="novelty-verification"></a>
## Novelty Verification

范围：Round 1 的 Idea 1–3，加上 Round 2 头部（11, 12/13, 19, 21, 15）。检索：OpenAlex 标题/DOI；项目内调研。arXiv API 仍超时、S2 429、WebSearch 不可用，故 **concurrent last-3-months 扫描不完整**。OpenAlex 宽检索噪声大，下面只保留机制上说得通的对照。

### Idea 1 — ADD-not-PATCH Skill lineage

| Closest work | Overlap | Differentiation |
|---|---|---|
| Progressive Neural Networks (2016) | 冻旧加新 | 权重列 ≠ Consumer-grounded Skill 卡；无 delayed 四线门 |
| RewardHarness (2026) | 库版本、失败回滚 | 回滚的是整个库快照；任务是图像偏好，不是 TS 准备；无 per-series harm 门 |
| DGM (2025) | 保留非最优个体 | 改的是 agent 源码；evaluator 是 SWE-bench |
| Evo-Harness (2026) | context→skill harness | 编译 Skill，但不是「单槽覆写 vs 多卡并存」的对照实验 |
| Experience Replay (2018) | 保留旧经验 | 回放样本 ≠ 可部署 Skill 卡 |
| 本项目 DEV-AUTO-3 | 已经 **经验性地看到** PATCH 有害 | 还没有把 ADD 实现并做等预算三臂；这正是 idea 而不是已发表结果 |

**Verdict (executor annotation, not acquittal):** 没有找到「在 Agent Data-Readiness Harness 里，Skill 卡 ADD vs PATCH 的等预算三臂」的已发表实验。最接近的是持续学习加列、RewardHarness 回滚、DGM archive。**Concurrent work 风险**：Agent skill-memory 方向 2026 预印本很快，3 个月内可能出现类似「don't overwrite tools」的博客/预印本；本轮未能做完 arXiv 新稿扫描。

### Idea 2 — credit split

未找到把 agent skill 更新的效用拆成 supply/update/recall 三通道的论文。进化论文甚至很少报 proposals-per-accepted-edit。Novelty 足够作为 Idea 1 的强制报告字段，单独成文偏薄。

### Idea 3 — ancestor bootstrap

SPIBB + RewardHarness rollback 是已知机制。Novelty 在 **Skill-card ancestor** 与 **TS delayed gate** 的结合。若只做工程 fallback、没有对照表，审稿人会当工程。应作为 Idea 1 的部署规则，而不是第二篇论文。

**Eliminate?** 无 idea 因「已被发表」淘汰。Idea 7/10 因依赖 Idea 1 而 defer。

### Round 2 头部 — 执行者注释（非正式 acquittal）

| Idea | Closest analogue | Delta | Novelty note |
|---|---|---|---|
| 11 Liveness witness | 软件测试 / 状态机可达性；非 ML 论文对象 | 把 HEC 生命周期当成 **可证伪仪器**，问 ADD/NARROW 是否曾被实例化 | 作为方法论文弱，作为 **先做的诊断** 强。不单独投稿 |
| 12 Observation-equivalence bound | Causal representation / MAXQ 分层；策略类的 observation bottleneck | 把 BSG oracle 改写成 **当前可执行观察语言下的可行动上界** | 找得到「表示不足」文献，找不到 TS Data-Readiness Skill 上的这张表。分区本身可被攻击 |
| 13 Scope expressivity ceiling | 形式语言/谓词文法表达力 | 冻结 12 特征词表 + 四线风险下的 constructive collision | 与 12 相邻。Astra 已警告不要外推到「Agent 能读任意 context」 |
| 15 Route–context Pareto | D5 自己的 2×2；因果分解综述 | 把 route 与 context 变成 **准入决策变量**，不是事后归因 | 机制新；审稿人会说这是模型路由。必须钉在 Consumer adapter |
| 19 One-edit delta-stable repair | AFlow 局部 workflow 搜索；AegisTS 算子 RL；counterfactual recourse | 学 \(d_e(P\to P')\) 而非 \(g_e(P)\)；独立重遇 + 四线门 | **方法新颖度中等**。可发表性取决于「delta 比绝对效应更稳」是否成立，否则是受限搜索 |
| 21 Forecast-displacement certificate | 共形预测（Barber et al., 2023, *AOS*，「beyond exchangeability」）[UNVERIFIED 作为最近共形文献，OpenAlex 命中]；SPIBB | 用 **预测位移 / 季节尺度** 证 sMASE 差，不看未来 y | 不等式本身不新。贡献必须是：在本 evaluator 上 **保住有用 treatment**。Astra 已把这记为最大反对 |
| 23 Delayed feedback to Slow | 任何「给模型看完整标签」 | 隔离 M-R0 的具体缺口：delayed 不入 bank | 经验问题，单独成文薄 |
| 24 Exposure vs instability | 有限总体抽样恒等式 | 拆覆盖机械效应 vs 响应变差 | 诊断，0-LLM 先做 |
| 25 Component-feedback edit | 归因反馈 / 解释性 RL | 两组同一四格损失，只差是否展示分解 | 高风险被说成 prompt。依赖 19/20 的 headroom |
| 26 Prefix information asymmetry | look-ahead bias / imputation leakage 文献（OpenAlex 有 GNSS imputation、LLM zero-shot TS 等，机制弱相关） | 训练窗准备是否偷看 serving 没有的后缀 | 0-fit 筛查便宜；有依赖才值得买 fit |

**Concurrent 风险**：2026 harness/skill 预印本很快（Self-Harness、Evo-Harness、TTHE、RewardHarness）。本轮未能做完 arXiv 新稿扫描。任何「Skill 记忆 / 不要覆盖工具」博客都可能撞 Idea 1。

### Cross-model novelty（Codex GPT-6 家族，thread `01a079e5-7c26-7260-9fdd-5ee2494370b8`）

- **请求**: `gpt-6-astra`。**披露**: “GPT-6-based Codex；精确 variant 与 xhigh **未验证**”。**正式 acquittal: 否。** 不得 `run_state.py accept`。
- Trace: `.aris/traces/novelty-check/2026-09-07_run01/receipt.json`
- 评审者检索到额外近作（OpenAlex 核验，TROVE DOI 404 → `[UNVERIFIED]`）：ACE `2510.04618` ✅；HarnessEvolve `2609.00829` ✅；VCE-Skill `2608.16544` ✅；SkillOpt-Lite `2607.03451` ✅；ADAS `2408.08435` ✅；CORELS `1704.01701` ✅；TROVE `2609.05019` `[UNVERIFIED]`。

| Method | Score | Paper rec | Dominant claim left |
|---|---:|---|---|
| A Idea 1 ADD-not-PATCH | **3/10** | **ABANDON as methods paper**；留作诊断 | 粒度差异不构成新保存原则。B 臂已经铸新卡不覆写祖先。DEV-KNOW-1 已实跑保留+检索+零选择 |
| B Idea 19 delta-stable one-edit | **5/10** | **PROCEED WITH CAUTION as dominant paper** | 跨情境 **Consumer 响应差** \(d_e\) 比绝对效用更稳。**同一候选集上 \(\arg\max d_e=\arg\max g_e\)**，贡献必须在跨情境迁移，不在减法。AFlow/VCE-Skill/TROVE 已记相对改进 |
| C Idea 13 Scope ceiling | **3/10** | **ABANDON as theory paper**；留作诊断 | CORELS 等价点论证的处理选择版。只约束冻结文法 \(\mathcal G\) |

**主贡献只能是 B（收窄后的稳定性假设）。A 与 C 不得并列第一作者贡献。** 「对时序 Agent 应用 X」不够。

**DEV-KNOW-1 对 A 的额外打击：** 保存与检索已不是卡点；选择层才是。Idea 1 的三臂 C′ vs B 若只测「是否保留祖先」，已被 DEV-KNOW-1 部分回答。

---

<a id="external-critical-review"></a>
## External Critical Review

### Round 2 quality triage（Codex GPT-6 家族，**不是** novelty acquittal / 不是 `accept`）

- **threadId**: `01a079d9-e589-72f0-8d25-fb3a53e7f9a0`
- **请求模型**: `gpt-6-astra` / xhigh
- **实际披露**: “GPT-6-based Codex；精确 `gpt-6-astra` serving variant 与 xhigh **未暴露**”
- **性质**: Type-B 排序（下一步该干什么）。0 fit / 0 实验 LLM / 0 密封。身份不完整，**不得** `run_state.py accept`。
- Trace：`.aris/traces/idea-creator/2026-09-07_run02/receipt.json`

**仓库下一步 Top 8（评审者排序）：**

1. **Idea 1 ADD-not-PATCH** — 对准最新失败。须先写清 C′ 与 **已经会铸新卡并 recall 的 B** 还差哪一件。生命周期改动仍未授权。
2. **Idea 13 Scope 表达力天花板** — 决定 Idea 20 有没有空间。
3. **Idea 19 Delta-stable 一步 Workflow 修补** — **最强机制论文假设**。
4. **Idea 21 预测位移 harm 证书** — 不等式成立；HEC replay cache 存 loss 不是预测向量，**不能**从该 cache 零 fit 证明非空。
5. **Idea 26 准备前缀信息不对称**
6. **Idea 14 足迹多样化 Fast 供给** — 仅当可执行行为冗余真的占名额。
7. **Idea 12 观察等价可行动上界**
8. **Idea 20 一句 Scope 修订** — 与 19 竞争。

**机制论文保留：19 主、1（带 2 的归因）次、21 储备。** 防止删除 ≠ 持续进化。

**历史更正（评审者读代码后）：** Idea 6 serving-side 双管线 **已实现**，不要当新 idea 重建。Idea 11 的「从未写回」对后来的 DEV-AUTO 已部分过时——只对**即将改的路径**做针对性可达检查。两更正都不改 `HEC1_EVOLUTION_NOT_SUPPORTED`。

**Fold：** 2/3/9 → 并进 1。**Kill 独立论文包装：** 24（可留 size-matched 诊断）。**互斥：** 19↔20；1↔4/5；14↔16↔17；15↔18/21；22↔26 纠正干预。25 必须跟在已固定的 19 或 20 后面。

评审者核过 DGM / PNN / AFlow / RewardHarness 公开页。重叠要求 **收窄贡献**，不是 blanket 否决。

### 执行者补充（仍不算 acquittal）

1. ADD vs PATCH 必须写成破坏性写回是否为 first-fault，数字来自三臂+逐格归因。
2. Development 数据不能当 sealed A5。
3. 不得把 Idea 1 包装成「HEC-1 其实是正的」。
4. 状态页：改 ADD **超出当时授权**。

### NeurIPS-style research-review（thread `01a079f4-4fdd-7291-a128-07cdaaa39fcc`，xhigh 请求）

- **披露**: 精确 `gpt-6-astra / xhigh` **无法验证**。正式 acquittal: **否**。不得 `run_state.py accept`。
- Trace: `.aris/traces/research-review/2026-09-07_run01/receipt.json`
- **Venue score: 3/10 reject.** **Repo diagnostic: 7/10.** Confidence 4/5 on inspected artifacts.

Fatal / major（评审者核对 JSON，不是执行者转述）：

- **R1** \(d_e=g_e(P')-g_e(P)\) 在每一情境保持对 \(P'\) 的排序；加权平均 delta 仍可能 \(\arg\max g_e(P')\)。没有写出非等价转移规则就没有方法。
- **R2** DEV-KNOW-1 不是选择器故障：u17 Support +0.078 / hf 0.30；u24 +0.292 / hf 0.25；门限 0.20。保留+检索成立；「扔掉了可用改进」不成立。
- **R3** DEV-AUTO-3 u23 +0.595 的 max harm 0.485 > 0.30，delayed 门失败。g1 供给抬高均值也抬高最坏伤害（0.48→1.48）。
- **R4** 无独立 edit-transfer 证据。HEC-1：0 修订链；10/34 安全保留。M-R0：41 实例 / 8 个独特情境。
- **R5** 「更稳」可被近恒等编辑 trivially 满足。
- **R6** 一步句法修改可改训练目标、系数、全部被处理序列。本 checkout **找不到** `p4ad_causal_decomposition.json`，不背书 D5 数字。

Mock NeurIPS: reject。能推向 accept 的是：非等价转移函数、前瞻性优于匹配基线、独立自然情境上的重复收益、且过尾部门。一条存活谱系或一张更低方差的 delta 图不够。

**ABANDON 宽方法文包装。保留 DEV-EDIT-TRANSFER-0。** 若 delta 规则退化成同一 argmax，或转移不能改善后续风险约束决策，Idea 19 作为方法文一并放弃。

---

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md`（**RETHINK** as methods paper; diagnostic READY）
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` — **DEV-EDIT-TRANSFER-0**
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`（三跑均未启动）
- Contract: `idea-stage/docs/research_contract.md`
- Pipeline: `refine-logs/PIPELINE_SUMMARY.md`

**Problem anchor:** After a natural forecasting conflict, can earlier parent–child response evidence guide a useful, safe workflow edit on an independent later encounter?

**Thesis:** Historical edit-response information vs matched absolute-utility history vs equal-budget fresh search. Subtraction is not the method.

**First three runs:** (1) spec+cache audit 0 fit/0 LLM; (2) chronological headroom（缺缓存才申请 fit）; (3) transfer comparison. **A 不在本包。**

## Next Steps

- [ ] 用户若要执行：只授权 **Run 1**（只读规格与缓存审计）。本流水线不启动 `/run-experiment`
- [ ] 写出非等价转移函数，否则不得主张 S
- [ ] Idea 1 不再作为主实验；DEV-KNOW-1 已部分回答保存问题
- [ ] novelty-check / research-review **不得 accept**（评审身份未验证）→ evidence gate 应为 BLOCKED
- [ ] 不打开密封 / +144 / Natural Final 来救不成熟机制

## Search / evidence limitations

- arXiv API timeout；Semantic Scholar 429；WebSearch 失败；gemini 未装。
- 近 3 个月 concurrent preprint 扫描不完整。
- Round 2 已跑 Codex brainstorm：`gpt-5.6-sol` 请求线程与 **`gpt-6-astra`** 请求线程；两者都 **未暴露精确 serving variant**，不得把生成当 acquittal。
- 未跑任何 pilot。
- `research-wiki/` 不存在，跳过 wiki ingest。
- `idea-discovery-evidence` gate 在 novelty-check / research-review 未 accept 前应为 BLOCKED。这是诚实状态，不是完成声明。

<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:START -->
## Evidence Gate
**Status:** BLOCKED

The workflow is not complete. Required stage evidence is missing:
- BLOCKED: novelty-check review evidence missing (status=done)
- BLOCKED: research-review review evidence missing (status=done)
<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:END -->
