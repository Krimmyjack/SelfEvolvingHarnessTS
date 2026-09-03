# Self-Harness 定向调研笔记（2026-08-20）

调研范围：本地仓库 `c:\Users\辉\Desktop\Agent\Self-Harness`（只读）。  
服务对象：SelfEvolvingHarnessTS 当前阻塞——观察已分化后的**提案层塌缩**（单一 Workflow 家族垄断，类似 tool-prior collapse）。  
纪律：反过度工程；只提取“当前实验能直接用的最小机制”；不做无边界仓库综述。

仓库规模（约）：顶层 `acceptance/ assets/ diagnosis/ eval/ harnesses/ proposer/ workflow/ README.md`，代码文件约 24 个，非空壳。

---

## 1. 整体身份

### 仓库内证据

- `README.md:15-19`：固定模型权重与 evaluator，优化“周围 harness”；每轮评测 → 从执行轨迹挖失败模式 → 同一模型提出 **bounded harness edits** → 仅在 held-in / held-out 回归支持时 promote。
- `README.md:37-45`：论文定位为 *Self-Harness: Harnesses That Improve Themselves*（arXiv:2606.09498）。
- `workflow/scripts/run_self_harness_loop.py:23-24,57-148`：公开闭环入口描述为 `eval/propose/eval/accept`；流程为 baseline eval →（可选）处理 pending candidates → 否则对 active branch 做 diagnosis → build proposer prompt → 外部 LLM 填 response → parse → enqueue → candidate eval → acceptance。
- `harnesses/qwen_tb2_final/repo_baseline.py` / `eval/harness_workspace/repo_baseline.py`：可编辑 surface 是 Deep Agents 风格的 `build_system_prompt` / `build_skills` / `build_tools` / `build_subagents` / middleware / runtime / permissions 等函数，而非时序数据处理 Workflow。
- `eval/README.md:1-74`：仪器是 Terminal-Bench-2.0 的 train(43)/heldout(21) 任务划分，不是时序 Consumer。

### 机制描述

这是一个**通用 coding/agent harness 自改进系统**，不是数据处理 harness。进化对象是 harness **配置面**（prompt 指令、skill 包、工具配置、subagent、middleware、runtime control、permission/interrupt），通过候选 workspace 物化后在 TB2 上重评。闭环数据流：

```text
TB2 eval traces
  → diagnosis（trace LLM 归因 + 跨 case 聚类 brief）
  → multi-proposer（机制族 × hook 提案）
  → materialize candidate surfaces
  → Harbor eval（train + heldout）
  → deterministic acceptance gate
  → branch 晋升 / merge（接受后 supersede 旧 active）
```

### 与我们设计的对照

| Self-Harness | 我们 |
| --- | --- |
| 进化对象：agent harness 配置（prompt/skills/tools…） | 进化对象：TS Observation / Program / Scope / Risk / Memory / Control |
| 任务域：Terminal-Bench 编程代理 | 任务域：时序 Data Readiness + Forecasting Consumer |
| Fast Path 即“带当前 harness 跑任务” | Fast Path：观察 → Typed Workflow → Support 探测 → Experience |
| Slow Path：诊断 brief → LLM 提案 → 评测门 | Slow Path：确定性 first-fault → 单面 patch → compiler/replay/反馈 |

### 是否值得最小借鉴

**身份层不借鉴**（不同问题定义）。可借鉴的是其闭环里与“多候选 + 确定性验收”相关的切片，见第 3、4、6 节。

---

## 2. 诊断 / 归因

### 仓库内证据

- `diagnosis/src/self_harness_diagnosis/trace.py:100-163`：`build_causal_trace_diagnosis` 要求显式 `llm`；`analysis_mode: "llm"`；失败轨迹规范化 steps/stages 后由 LLM 产出 causal analysis。
- `diagnosis/.../trace.py:278-318,342-354`：LLM 被要求标注 `terminal_cause` / `criticality` / `agent_mechanism`；prompt 文字写明 root cause = “first unrecoverable critical failure”，但这是**提示词语义**，不是确定性 first-fault 编译器。
- `diagnosis/.../trace.py:237-262`：`terminal_failure_kind` 对 verifier 文本做启发式分桶（missing artifact / dependency / timeout 等）——确定性预处理，但仍把机制归因交给 LLM。
- `diagnosis/.../integrated.py:41-62,107-115`：跨 case 聚类签名为 `(terminal_cause, criticality, agent_mechanism)`，字段来自 LLM primary analysis item。
- `diagnosis/.../integrated.py:72-84`：brief 明确写“LLM-generated … fields”；“Select one high-confidence terminal-cause cluster”；“prefer no-op”。
- `diagnosis/.../tb2.py:9-39`：从 Harbor trial 抽取 verifier evidence（确定性 I/O），供诊断使用。

### 机制描述

诊断是 **LLM 判读为主 + 确定性证据抽取/聚类为辅**：

1. 从 TB2/Harbor 产物抽 terminal verifier evidence；
2. 规范化 agent trace steps，按 change 边界切 stage；
3. LLM 填 stage-level 因果字段；
4. 按三维签名聚成 failure clusters，渲染成 proposer 只读 brief。

**没有**我们意义上的：确定性 first-fault 归因器、每轮只改一个 Harness Surface 的 Slow Path 纪律（单面纪律在 proposer 的“单 hook”合同里，不在 diagnosis）、失败 Episode 写入与后续检索。

失败证据组织：按 case 的 diagnosis JSON → cluster brief markdown（passing cases 作为 regression 清单）。

### 与我们设计的对照

我们要求 Slow Path **确定性 first-fault + 单面修改 + LLM 不得自批**。Self-Harness 把“找原因”交给 LLM，把“是否采用”交给评测门——与我们归因纪律相反，但与“提案不得自批”部分同向。

### 是否值得最小借鉴

**不值得**把 LLM causal diagnosis 搬进我们 Slow Path（会削弱 first-fault 与可复现归因）。  
可注意其“terminal evidence 优先、recovered friction 不当修理目标”的 brief 措辞（`integrated.py:184-188`），但只作诊断文案参考，不是可执行机制。

---

## 3. 提案多样性（当前阻塞重点）

### 仓库内证据

**机制族菜单与单 hook 合同**

- `proposer/.../hooks.py:56-64`：七个 `mechanism_family` → 允许 hooks 映射（prompt / subagent / skill / tool / middleware / runtime / permission）。
- `proposer/.../hooks.py:94-95`：`mechanism-diverse candidates must change exactly one virtual hook`。
- `proposer/.../multi_proposer.py:298-315`：eval 候选必须恰好一个 `candidate_values` hook，且 hook ∈ 所选 family。

**多路提案与“互异”软硬约束**

- `proposer/.../multi_proposer.py:56,105-106`：默认 `route_count=4`；prompt 要求 “mechanism-diverse multi-proposer”“mutually distinct”。
- `proposer/.../multi_proposer.py:129`：强制字段 `why_distinct`。
- `proposer/.../multi_proposer.py:85-90,147-148`：slot 模式把已生成提案注入 prompt，要求与已有提案 materially distinct。
- `proposer/.../multi_proposer.py:192-229,318-323`：库函数 `generate_multi_proposals` 顺序填 slot，并用签名  
  `(selected_cluster_id, mechanism_family, exact_hook)` 去重；重复则重试失败。

**公开 workflow 路径的缺口（重要）**

- `workflow/.../run_self_harness_loop.py:305-327,359-384` + `proposer/scripts/run_multi_proposer.py:62-71`：公开环是“一次生成整份 prompt → 外部命令写 response → `parse_multi_proposer_response(..., require_one=False)`”。  
  **CLI 解析路径不调用 `generate_multi_proposals`，因此不执行签名去重硬闸。**
- 一次性 prompt（`slot is None`）只靠自然语言要求多样性，没有 familiar-tool 降权、探索熵、强制覆盖未用 family。

**未见机制（全库 rg）**

- 无 `entropy` / temperature sampling 多样性、无 familiar prior 降权、无 “强制备选另一家族” 的硬覆盖约束。
- 签名去重仍允许**同一 `mechanism_family` 多候选**（只要 cluster 或 hook 不同）。

### 机制描述

防塌缩手段分层：

1. **结构合同**：一次只改一个虚拟 hook；hook 必须挂在声明的 mechanism family（防“大 prompt 一锅炖”伪装成多机制）。
2. **软多样性**：多 route、why_distinct、已生成提案上下文。
3. **可选硬多样性**：`(cluster, family, hook)` 路由签名唯一（仅库内 `generate_multi_proposals`）。

这能减轻“所有提案塌成同一种编辑面”，但**不能保证**跨 family 覆盖，也不能对抗“模型先验垄断某一个 family”（例如总选 `prompt_instruction` 或我们场景下的 `repair_level_shift`）。

### 与我们设计的对照

我们当前嫌疑：观察已分出 outlier vs level-shift 几何，但 Agent **首选 Workflow 家族仍被 `repair_level_shift` 垄断**。  
Self-Harness 对应物是：诊断 cluster 可能多样，但提案仍可能集中在一个 mechanism_family。其最接近的解药是 **多 slot + family 标签 + 单面编辑 +（可选）路由签名去重**，不是经验检索，也不是 familiar-family 降权。

### 是否值得最小借鉴

**值得（与阻塞直接相关）**：把“Workflow/Program 家族”做成显式枚举，Fast Path 提案强制产出 `route_count` 条、每条恰好一个家族/一个主算子，并做 `(context_cluster, workflow_family, binding)` 级去重；若公开路径要用，必须像 `generate_multi_proposals` 一样把硬闸接到执行环，而不是只写在 prompt 里。

**不值得**：照搬其 Deep Agents hook 菜单或 TB2 专用 materialize。

---

## 4. 验收门与自批问题

### 仓库内证据

- `README.md:18-19`：promote 需 held-in 与 held-out 支持。
- `acceptance/scripts/run_acceptance_gate.py:10-12,77-115`：确定性门；默认 splits=`train`,`heldout`；规则 `no split drops and at least one split improves`；比较平均 `pass_rate`。
- `proposer/.../multi_proposer.py:106-107`：提案侧明确 “Use only train-side evidence. Do not use or mention holdout…”。
- `workflow/.../run_self_harness_loop.py:472-484,558-587`：候选与 merge 后候选都跑同一 acceptance gate；拒绝则 `rejected` / `accepted_merge_rejected`。
- `workflow/.../run_self_harness_loop.py:995-1015`：接受后旧 `active` → `superseded`，新 branch 成为 active；**不是**显式 rollback/retire 生命周期 API，只是分支指针前移。
- 全库无 `CONDITIONED` / `UNGUIDED` / provenance 分层；无“Skill 诱导证据不得给该 Skill 扩权”的合同。

### 机制描述

批准者是**确定性评测比较器**，不是 LLM 自批。信息墙：proposer 只准看 train diagnosis；heldout 只进 acceptance。  
这解决的是“改完在训练任务上看起来更好就上线”的粗糙自确认，**没有**解决我们更细的 provenance 问题（CONDITIONED 证据 vs 独立 UNGUIDED 授权 TRY）。

生命周期：接受 → 新 branch active、旧 branch superseded；拒绝保留队列状态。未见自动退役/回滚到更早 depth 的策略模块。

### 与我们设计的对照

同向：LLM 不批准自己的 patch；独立评测门。  
我们更强：Experience 写入 ≠ 扩权；Skill 在场诱导的 CONDITIONED 证据不能授权新主动 TRY。Self-Harness 无此分层。

### 是否值得最小借鉴

**部分值得**：train-only propose + held-in/held-out 确定性门作为“Shared/Target 确认”的对照实现很干净。  
对**当前提案塌缩阻塞**不是第一解药；对 Fresh/扩权纪律是弱相关参考。  
**不值得**引入其 branch merge 多候选合并逻辑（`create_merged_candidate`）——超出当前最小面，且与我们“每轮一主要因果假设”冲突风险高。

---

## 5. 经验 / 记忆组织

### 仓库内证据

- `harnesses/qwen_tb2_final/repo_baseline.py:35-51`：`build_memory_sources()` 固定返回 `["/AGENTS.md"]`；`build_skills()` 默认 `[]`——这是 **Deep Agents 的 memory/skills 配置钩子**，不是 Experience Episode 库。
- `proposer/.../hooks.py:146-159,344-359`：提案可物化 `skill_bundle`（生成 `.self_harness_generated_skills`）——仍是 harness surface 编辑，不是检索记忆。
- 全库无 Episode / Target-local Skill / Shared Capability 分层，无三段式检索，无“记录 ≠ 扩权”。

### 机制描述

“Memory/Skill”在此仓库 = **可被自修改的 agent 配置接口**。跨轮状态主要靠 `branch_state` + 接受后的 surface 文件，而不是可检索 Action–Response Experience。

### 与我们设计的对照

我们的核心承重链是 Experience → Target-local Skill →（可选）Shared Capability，且执行权分层。Self-Harness **没有**这条链；它用评测门直接决定 harness 配置是否晋升。

### 是否值得最小借鉴

**不值得**把其 skill_bundle / memory_sources 当作我们的 Memory 层。那会把配置编辑与经验证据混为一谈。

---

## 6. 对我们最有价值的差异

### 我们没有而它有（且与当前阻塞相关）

1. **显式 mechanism_family 菜单 + 单 hook 变更合同**（`hooks.py:56-64,94-95`；`multi_proposer.py:298-315`）——把“提案空间”从自由文本压成可枚举家族，直接针对家族垄断。
2. **多 route 提案 +（库内）`(cluster, family, hook)` 签名去重**（`multi_proposer.py:192-229,318-323`）——逼出备选路线；但需注意公开 CLI 未接线。
3. **train 诊断 / heldout 验收信息墙 + 确定性 pass_rate 门**（`multi_proposer.py:107`；`run_acceptance_gate.py:99-110`）——与 Fresh 纪律同向，但不是提案塌缩主药。

### 它明确没解决（可作为我们论文 / 方法差异点）

1. **Familiar-family / tool-prior collapse**：无对高频家族降权或强制覆盖未用家族；签名去重仍允许同族多提案。
2. **确定性 first-fault Slow Path**：诊断主路径是 LLM。
3. **Experience provenance（CONDITIONED vs UNGUIDED）与执行权分层**：无。
4. **TS-native Context × Workflow × Action–Response Capability**：无；进化对象是通用 agent harness 配置。
5. **同域异构：按 series/channel/interval 选不同 Workflow**：无；TB2 任务级 pass/fail 聚合。
6. **公开环把硬多样性闸留在库函数、未接入默认 loop**：多样性可能只停留在 prompt 愿望。

### 值得最小借鉴的 ≤2 个机制

1. **Mechanism-family 多路提案合同（优先）**  
   映射到我们：为候选 Workflow/Program 建小型显式家族枚举；Fast Path 一次产出 `K` 条互异提案；每条恰好一个主家族/主算子；用 `(观测簇或缺陷机制, workflow_family, 关键几何/参数绑定)` 做硬去重；**必须接到实际 proposer 执行路径**，不要只写 prompt。  
   目的：直接打当前 `repair_level_shift` 垄断。

2. **单面编辑合同（次优先，且与我们 Slow Path 对齐）**  
   映射到我们：不仅 Slow Path 单面，Fast Path 单次试验也限制“一个主 Program 家族”，避免观察分化后仍被复合/默认家族吞掉。  
   Self-Harness 的 “exactly one virtual hook” 是可直接类比的最小结构，不必搬 hook AST 改写器。

### 不值得借鉴的原因（摘要）

| 机制 | 原因 |
| --- | --- |
| LLM trace causal diagnosis 作 Slow Path | 与确定性 first-fault / 可复现归因冲突 |
| Deep Agents skill/memory surface 物化 | 不是 Experience/扩权分层；会偏成配置 AutoML |
| 多候选 merge 成超级 patch | 违反“每轮一个主要因果假设”；过度工程 |
| TB2 Harbor eval 栈 / branch_state 平台 | 仪器不同；建设成本高且不解决提案塌缩 |
| 仅 prompt 声明的 diversity | 他们自己的公开环证明这不够硬 |

---

## 对当前阻塞的可用性判断

**阻塞**：观察已能区分 outlier vs level-shift 几何，但提案仍塌缩到单一 Workflow 家族（`repair_level_shift`）。

**Self-Harness 有没有可直接搬的最小机制？**  
**有，但只有一条半：**

- **可直接搬（最小）**：显式 **workflow/mechanism family 标签 + 多 slot 候选 + 路由签名硬去重 + 每候选单主算子**。这是该仓库里唯一与“tool-prior / family monopoly”同构的防塌缩装置；实现时应对接真实生成循环（对标 `generate_multi_proposals`），并额外考虑一步我们需要而他们没有的约束——例如 **禁止 K 条全落在同一 family**，或对上一轮垄断家族做轻量降权（他们未实现，属我们增量）。
- **不能直接当解药**：其 acceptance 的 train/heldout 门、LLM diagnosis、branch merge、Deep Agents skills——都不解决“首选家族被先验垄断”。
- **论文差异可写**：我们在 TS Data Readiness 上承重 **Experience provenance + 确定性 Slow Path + Context-conditioned Workflow 多样性**；Self-Harness 证明 harness 可自改，但把多样性主要交给机制菜单与评测门，且默认公开环未强制执行最强去重闸。

**建议的下一刀（仍属提案层，单一因果假设）**：在现有观察合同之上，只改 Proposer：强制多家族候选生成与硬去重；暂不动 Memory / Acceptance / SHA 体系。
