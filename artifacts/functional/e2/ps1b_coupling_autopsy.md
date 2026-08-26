# PS-1b — 提案耦合尸检（0 LLM）

protocol: `ps1b_coupling_autopsy_v1`  parent: PS-1 `5bf4d64` / `NO_PROPOSAL_SHIFT`  llm/fit/download: **0/0/0**  code: **untouched**

**TEXT_RUNG_INERT**

scoped 卡的完整 `_skill_prompt`（body + risk_guards）进入了提案那一次 Fast 调用的 system 上下文，位置与 C40 的 Source 卡同槽；提案行为仍不跟随 TRY-HYPOTHESIS。文本档在当前 prompt 与 `returned_model=gpt-5.6-sol` 下天然弱耦合。下一步是权限梯「供一待验证候选」（PS-2），不是修 retrieve 线重跑 PS-1。

> 0 LLM 离线重建。PS-1 store 不存 raw messages；渲染文本由现行 `agent_core._skill_prompt` / `_messages` 与卡 JSON 机械复原。token 数为空白切分近似，不是线上分词器。

## 1. 渲染重建

### 1.1 现行 Fast 消息结构（C40 与 PS-1 同构）

`TTHAFastAgent.prepare` 只调用一次 `resolve_harness_view`（`fast_agent.py:742`），同一 `view` 交给 inspect / propose / select。`run_stage` → `_messages`（`agent_core.py:167-274`）：

**system**（此序）：

1. `harness_view.instruction`（`h0/instruction.md`）
2. Runtime envelope 规则段
3. 字面量 `Resolved Harness: `
4. canonical JSON：`{instruction, skills: [_skill_prompt…], memories, controls}`

**user**：`public-agent-input/1`，含 `features`（observation）、propose 时另加 `inspection` / `allowed_operator_contracts`。技能卡**不**进 user。

`_skill_prompt`（`agent_core.py:135-142`）键集：`skill_id, skill_kind, body, allowed_tools, risk_guards`。body 与 risk_guards 无截断。

Fast 技能序：bootstrap 按 `skill_id` 排序 → 匹配 capability 的 `top_k=2` → safety。实测四元序恒为：

`build_contrastive_candidates` → `inspect_and_localize` → `select_or_identity_and_verify` → **实验卡**

### 1.2 PS-1 scoped 卡在提案 system 中的确切位置

离线复原（h0 + `ps1_source_hypothesis_scoped_v1` + run3 r1 的 `fast_features_binned`）：

| 量 | 值 |
|---|---|
| Fast 是否含卡 | 是（`_is_inert_experience_card=false`：无 `risk_guards.sections.TRY`） |
| 卡下标 | skills[3] |
| 卡前 bootstrap body | 6069 + 673 + 142 字符 |
| 卡 body | 1810 字符 / ~185 word |
| 复原 system | 12799 字符 / ~1300 word |
| 卡 body 起点 | system 字符 9241（约 72%） |
| `hampel_filter` | 2 次：TRY-HYPOTHESIS 散文 + `risk_guards.scope_v1.program_geometry`；首次偏移 10228（约 80%） |
| `Frozen program steps:` | 无 → `_parse_frozen_steps` = None |

提案可见的 TRY 段原文：

> TRY-HYPOTHESIS: prioritise exploring the hampel family (hampel_filter) among the candidates you propose. This is a ranking suggestion only and is not execution authority: it supplies one candidate for the same candidate budget and grants no right to deploy.

observation 在 **user** 消息，整段 system 之后。

### 1.3 C40 `source_investigation_cls_v1`（r2 三臂存卡 + 当时代码）

C40 live git `cb03eb6`（2026-08-25）早于 T1 `03f2c1b`（2026-08-26）。当时无 Fast 惰性过滤；GPA A5 两轮 `retrieved_skill_ids` 均为三 bootstrap + `source_investigation_cls_v1`，池锁 level-shift。

T1 **只改** `retrieval.py`（无授权经验卡不进 Fast）。`_skill_prompt` / `_messages` / `instruction.md` / `build_contrastive_candidates.json` 自 C40 以来未改布局。今日用同一卡走 `resolve_harness_view(fast)`：**不进 Fast**（`_is_inert_experience_card=true`），Slow 仍见。这是 T2 已证的可见性不变量，不是 C40 当时的提案上下文。

强制按 C40 当时「卡在 Fast」复原 system：

| 量 | 值 |
|---|---|
| 卡下标 | skills[3]（与 PS-1 同槽） |
| 卡 body | 906 字符 / ~121 word |
| 复原 system | 10835 字符 / ~1295 word |
| `repair_level_shift` | **2** 次：body RISK + `risk_guards.sections.RISK` 整段复写；首次偏移 9162（约 85%） |

RISK 原文：

> RISK: The census provides no unguided positive support for any listed program, and the lone negative result for repair_level_shift is not repeated; do not infer a transferable preference from this evidence.

**位置结论**：两卡同槽、同在 bootstrap 长文之后、算子 token 都在 system 后部 80–85%。位置解释不了行为分裂。

## 2. 提案指令审计（逐行）

`methods/ttha/harness/h0/instruction.md`：

```
You are the TTHA preparation Agent. Use only public observations, declared tools,
canonical operator contracts, retrieved Harness content, and the typed output schema.
Never request or infer clean references, injection metadata, candidate utility, or
private rankings. Inspect before proposing. Supply at most the configured number of
effect-distinct PROGRAM candidates. Select exactly one candidate, including identity.
Keep modifications local and abstain when public evidence does not justify a repair.
```

- **有**「使用 retrieved Harness content」。
- **无**「仅依据 observation 提案」类措辞。

`build_contrastive_candidates` revision 7，提案相关原句：

1. Goal：「From the current TaskContext, the public Pattern evidence, **and any retrieved Experience**, construct…」
2. Inspect 输入：「only the deployment-visible feature summary and the public tools」
3. Propose 输入：「inspection payload …, the allowed operator contracts, the fixed probe panel, **and any retrieved positive, negative, or conflict Experience**」
4. **4b**：「The Source prior, when admissible, is **injected by the runtime** into a pool slot; **do not propose a copy of it**.」
5. **4f**：「Retrieved Experience may inform the prior slot but **must never remove the exploration slot**.」
6. `propose.rule.effect_distinct`：「**do not propose operators merely because they are allowed.**」
7. `propose.rule.no_legal_binding`：「abstain by returning an **empty candidate set** rather than an unverifiable program.」

`inspect_and_localize` / `select_or_identity_and_verify` 不点名技能卡，也不点名家族。

双重约束：instruction 允许读卡；4b 说先验由 runtime 注入、禁止复制；PS-1 卡自称「supplies one candidate」，runtime 却不注入。模型被授权读文本，同时被程序文本劝阻「不要自己抄一份先验」。

## 3. 布线核查

提案调用链：

`prepare` 单次 `view` → `run_stage(stage="propose", harness_view=view)` → `_messages(..., harness_view)` → system 含完整 `_skill_prompt`。

`DecisionTrace.retrieved_skill_ids = view.skill_ids`（`fast_agent.py:672`）。PS-1 工件无 raw prompt，但 **A5-scoped 四跑八轮** `retrieved_skill_ids` 均为三 bootstrap + `ps1_source_hypothesis_scoped_v1`。其中：

- `ps1_run6` r2：卡在视野，池 `identity + robust_local_outlier_repair`（`outlier_threshold`），3 LLM
- `ps1_run9` r1：卡在视野，池 `identity + local_mad_repair`（`outlier_threshold`），4 LLM

提案 LLM 调用发生过，卡在同一 view，模型提出了**别的家族**。hampel 0/8 轮。这不是「只进了 select/verify」。

机械入池（书内「文本档」的对照面，不是 WIRING_FAULT 定义）：

- `supplies_candidates` 在 `methods/ttha` **只有** `ordering_card.py:216` 默认 `False`。`fast_agent` / `agent_core` / `retrieval` **零读取**。
- PS-1 `card_kind=ps1_source_hypothesis`，`is_ordering_card` 为假，排序卡路径也不碰它。
- `_skill_frozen_candidates` 只解析 `Frozen program steps:`；PS-1 卡故意不含（runner 断言）。
- PS-1 `carried_episodes=()`，`runtime_prior_slot` 不点火。

hampel 仍可合法写出：`allowed_tasks` 含 classification；`fast_propose_v1` 注入全部 `OPERATOR_NAMES`；编译闸是 `_allowed_operators` 不是 contracts 列表。A3 同考场提出 burst / outlier。菜单缺席不是本场零结果。

**布线结论**：卡文本完整进入提案调用。非 `WIRING_FAULT`。

## 4. 卡语言强度

| | C40 | PS-1 scoped |
|---|---|---|
| 点名位置 | RISK（否定句） | TRY-HYPOTHESIS（弱祈使） |
| 句式 | 「lone negative result for **repair_level_shift**」「do not infer」 | 「prioritise exploring」+ 六条 hedge |
| 复写 | body + `risk_guards.sections` 全文再写一遍 | 散文一次 + JSON `program_geometry` |
| WHEN | 短资格句 | 19 叶 Pattern 清单在建议之前 |
| 与冷策略 | 点名模型已会提的 level-shift | 点名 GPOVY 冷启动从不提的 hampel |
| 显著性 | 短卡、独特算子 token、否定失败 | 长卡、hedge 多于算子名 |

C40 能带偏，是「短 RISK 里的具体算子名 + 否定句」在同槽被读成可提案对象，不是因为卡更靠前。PS-1 把建议写成 ranking-only / 非执行权 / n=2 weak / 一探非事实，再叠加 4b「勿抄先验」，文本档没有克服冷策略。

## 5. 裁决（冻结）

**`TEXT_RUNG_INERT`**

- 否决 `WIRING_FAULT`：提案上下文有完整渲染；八轮 retrieved 含卡；两轮还编出了非恒等候选。
- 否决 `MIXED`：不是部分渲染。未读的 `supplies_candidates` 属于下一档权限，本书判给 PS-2。
- 文本档在当前设计与该 returned_model 下天然弱耦合。

### PS-2 设计要点（协议其余不变）

- 权限梯下一档：**供一待验证候选**。
- scoped 卡携带具体 `hampel_filter` 冻结程序（`Frozen program steps:`），经 `supplies_candidates` **机械入池**（接现有 `_skill_frozen_candidates` 或等价单点注入）。执行权仍关；Support / delayed 仍握批准。
- 安慰剂改为携带**合法中性 no-op 程序**的对照卡（测「任何机械入池」vs「正确候选」），不再用空 body。
- 仍是 GPOVY、三臂 ×4、12 跑预算。
- 不要：只修可见性后原样重跑 PS-1；不要打开 `grants_execution`；引导下正例仍不计 Source 跨域授权。

## 6. 模型混淆

**评级：低。**

C40：`gpt-5.6-sol` @ agicto，returned 同名。PS-1：请求 `cpa-gpt-5.6-sol`，returned `gpt-5.6-sol`（CPA 中转首用）。同 returned_model；中转路径不同，已注记。PS-1 三臂共后端且仍不分离，混淆解释不了本对照。

## 7. 书外发现

1. `supplies_candidates` 是死旗：只写不读。
2. PS-1 卡不是 ordering-control，排序路径看不见它。
3. C40 算子名被 `risk_guards.sections` 在 `_skill_prompt` 里双写。
4. A5-scoped 2/4 跑仍提 `outlier_threshold`——卡连竞品家族都未压住。
5. instruction 不是 observation-only。
6. 4b +「supplies one candidate」+ 无注入 = 双重约束。
7. 提案 schema 含 hampel；编译允许 classification。不是菜单缺席。

## 8. 义务自报

0 LLM / 0 fit / 0 下载。未改 `methods/` `runtime/` `contracts/` `operators/`。密钥未入工件。他线脏文件未提交。
