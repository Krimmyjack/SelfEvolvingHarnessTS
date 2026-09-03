# 相关工作调研：Harness 自进化 / 时序 Data Readiness

调研人：主控 Agent（执行者角色）
日期：2026-08-17
状态：**事实记录，非计划。本文档不含建议、不含下一步、不含裁决。**

## 0. 证据等级声明（引用本文档前先读这一节）

| 工作 | 我读了什么 | 我**没有**读什么 |
|---|---|---|
| AegisTS | **本地源码**（`a-evolve/AegisTS`），逐行核对 | 论文 PDF 正文 |
| TimeClaw | **本地 README**（`TimeClaw/README.md`） | 源码、论文 PDF |
| RewardHarness | **论文正文**（arXiv HTML，两次定向抽取） | 源码（两个 repo 均未拉取） |
| A-Evolve | 仅 README 开头 | 其余全部 |

凡本文档给出 `file:line` 的，是我实际打开核对过的；凡给出论文数字的，来自 arXiv HTML 正文。
**未标注来源的推断不存在于本文档** —— 如果读起来像推断，那是我的表述问题，请回来质疑。

---

## 1. AegisTS

**位置**：`C:/Users/辉/desktop/agent/a-evolve/AegisTS`
（注意：嵌在 `a-evolve` 仓库目录树里，但**与 A-Evolve 是不同的工作**）

**论文**：AegisTS: A Hierarchical Agent System with Reinforcement Learning for
Multivariate Time Series Data Cleaning — https://arxiv.org/html/2605.04902v2

### 1.1 模块映射（README 自述）

```
Error_Injection/injector.py    488 行   数据质量问题注入
Error_Detection/Detector.py    633 行   论文的 Data Quality Detector
Error_Cleaner/RLclean.py      1952 行   论文的 Cleaning Pipeline Generator
```

算子工具在 `Error_Cleaner/tools/`：`anomaly.py`、`constraints.py`、`missing.py`，
以及子目录 `imputers/`、`outlier_modifiers/`、`constraint_violation_handlers/`。

### 1.2 三个下游任务（关键结构差异）

```
RLclean.py   task_type 取值 = forecast | classification | clustering
分支位置（已核）: 377 466 486 498 528 554 568 600 655 683 761 871 894 896 898
```

task_type 影响**状态构造、模型选择、奖励计算、评估方式**，不是装饰参数。

### 1.3 两层模型：便宜 proxy 每步，昂贵 final 只在终局

```
:893   _create_proxy_model()
         classification -> MiniRocketClassifier(n_kernels=5000, random_state=42)
         forecast       -> DLinear(seq_len=window_size_f, pred_len=horizon_f, ...)
         clustering     -> 带 auto_select_k 的 estimator
:879   train_and_evaluate(task_type, 'proxy', proxy_model, ...)                  初始化
:1240  perf_change, proxy_train_time = train_and_evaluate(task_type, 'proxy',...)  每步
:1266  final_perf, _ = train_and_evaluate(task_type, 'final', final_model, ...)     终局
```

### 1.4 奖励构成

低层（算子级）：

```
:1243  R_Perf_N = np.clip(perf_change, -1.0, 1.0)      <- 来自 proxy 模型
:1249  'perf_component': MU_4 * R_Perf_N
```

高层（每步）：

```
:55    LAMBDA_1, LAMBDA_2, LAMBDA_3 = 0.4, 0.5, 0.1
:1364-1373
       R_H = LAMBDA_1 * low_reward_N     (0.4, 下游性能)
           + LAMBDA_2 * R_Issue_N        (0.5, 数据问题下降)
           - LAMBDA_3 * R_Cost_N         (0.1, 时间成本)
```

**数据问题下降的权重（0.5）高于下游性能（0.4）。**

高层（终局）：

```
:56    LAMBDA_GRAD = 10.0
:1269  R_H = LAMBDA_GRAD * (final_perf - self.initial_final_perf)
```

**代码与注释不一致（警告）**：`:55` 的行内注释写作 `- LAMBDA_1 * R_Cost_N`，
但 `:1365` 实际使用 `-LAMBDA_3 * R_Cost_N`。**以代码为准。**

### 1.5 评估指标：有 ground truth，直接测修复保真度

`Error_Cleaner/EvaluationMetrics.py`，入口
`evaluate_cleaning_effectiveness(d_repair, d_clean, d_dirty, time_cost)`：

```
mse         修复 vs 干净，MinMaxScaler 归一后逐维平均
mnad        同上，用 MAE
rra         1 - mae_rc / (mae_cd + mae_rd)
precision   修复动作里有多少是有益的
recall      注入的错误里有多少被修到
f1_score    上两者调和平均
mae_error   只在未被注入位置上计的误差
time_cost
```

`precision_recall` 的判据（已核）：

```python
bound = (np.max(d_dirty, axis=0) - np.min(d_dirty, axis=0)) / 10000
error_array          = np.abs(d_dirty - d_clean)  > bound   # 哪里有注入错误
repair_array         = np.abs(d_dirty - d_repair) > bound   # 哪里被改动了
benefit_repair_array = (np.abs(d_clean - d_repair) < np.abs(d_dirty - d_clean)) * repair_array
tp = error_array * benefit_repair_array
precision = tp / repair_array      recall = tp / error_array
```

**这套指标完全不依赖下游模型。**

### 1.6 消融：三个独立的近 2000 行文件

```
wo_tasks.py      1944 行   把下游任务信号整体换成内在质量指标
wo_proxy.py      1937 行   去掉便宜的 proxy 反馈
wo_metrics.py    1946 行   去掉内在质量指标
cross_dataset.py           跨数据集泛化
```

`RLclean.py` 与 `wo_tasks.py` 的 diff（已核，节选）：

```
-  cur_test_perf = self.last_test_perf
-  perf_change   = cur_test_perf - self.prev_test_perf
+  cur_issue_severity      = float(np.sum(rates))
+  terminal_quality_reward = np.clip(1.0 - terminal_issue_severity, -1.0, 1.0)
+  R_H = LAMBDA_GRAD * np.clip(terminal_quality_improvement, -1.0, 1.0)
```

即：**用「数据问题严重度下降」替换「下游性能提升」，其余保持不变。**

---

## 2. TimeClaw

**位置**：`C:/Users/辉/desktop/agent/TimeClaw`

**论文**：TimeClaw: Harnessing Generalist Agents for Contextualized Time Series —
https://arxiv.org/abs/2606.05404

**来源等级**：以下全部来自 `README.md`，**未读源码**。

### 2.1 三个 benchmark、三个任务族

- **CiK** —— 带自然语言上下文的预测
- **TSRBench** —— 图表式序列推理
- **TSAIA** —— 金融量化问答

同一个 agent、同一个 CLI（`main.py`），每个任务族内有确定性 train/test 划分。**不做微调。**

### 2.2 三个组件

1. **In-process MCP tool server**（`timeclaw/tools/server.py`，per-worker FastMCP）——
   inspection / forecasting / anomaly / finance-quant 工具。设计目的是**让 agent 永远不用从
   prompt 里读原始数字**，数值计算保持精确。
2. **Memory bank**（`timeclaw/memory/`）—— append-only JSONL 轨迹存储，
   键是**确定性 ~20 维序列指纹**；可选两段检索：先按自然语言上下文 cosine
   （`text-embedding-3-small`）过滤，再按指纹 L2 排序。
3. **capability-evolution loop** —— 记录、验证、复用成功的解题轨迹。

### 2.3 它的 A5/A3 就是一个命令行开关

```
--mode train                      填充 memory bank
--mode test --k-neighbors 0       no-memory 消融
--mode test --k-neighbors 3       完整 TimeClaw
```

**结局指标是 benchmark 任务分数**，不是 probe 数、不是 harm 计数。

模型层 provider-agnostic：OpenAI `gpt-*` / Google `gemini-*` / Anthropic `claude-*`
统一走 LangChain `create_agent`，共用同一个 tool-call 循环。

---

## 3. RewardHarness

**论文**：RewardHarness: Self-Evolving Agentic Post-Training — arXiv **2605.08703**

**代码**：https://github.com/TIGER-AI-Lab/RewardHarness ，
https://github.com/KlingAIResearch/RewardHarness （两个 repo，**均未拉取**）

**项目页**：https://rewardharness.com/

**领域**：图像编辑的偏好评判（reward modeling），**不是时序**。
但它是三者中在结构上与本项目 C2（Harness 自适应进化）最贴近的一个。

### 3.1 进化对象：两类人可读的 Markdown

- **Skill** = 名称 + 一句话描述 + **评分 rubric（把质量拆成可评判准则）** + 示例
- **Tool** = 名称 + 目的 + 期望输入输出 + **调用条件** + 分步执行协议

三种操作：**create / modify / deprecate**。库从空开始，带版本。

### 3.2 循环

```
100 条 preference demo  ->  60 train / 40 val（val 全程 held-out）

每轮：
  Orchestrator 从库里选出相关的工具与技能子集
  冻结的 Sub-Agent 用它们构造推理链，产出偏好判断
  按 ranking 是否一致，把样本切成 correct / error
  自动做 root-cause 分析 -> 提出 create / modify / deprecate
  在 40 条 val 上评估 -> 超过当前最佳则接受，否则整体回滚

共 77 轮，全程复用同一批 100 条，无额外人工标注
最优配置：Gemini-2.0-Flash
```

### 3.3 两个关键设计

**(a) 反馈是二值的。** 判据是 ranking 是否与 ground-truth 偏好一致。
论文明确：**连续的 score gap 只用于诊断分析，不进裁定。**

**(b) 接受权在 held-out 验证集，不在 LLM。** 提案 -> val 评估 -> 超过历史最佳才接受，
否则回滚。论文提到**大量提案被回滚**，且 **Skill 提案的接受率低于 Tool 提案**。

### 3.4 数字

```
空库基线              42.5%   val accuracy
最终库（iter 69）      62.5%   val accuracy   = 3 Skills + 4 Tools
库规模轨迹            0 -> 13（约 iter 50 峰值）-> 7
```

**收益来自剪枝，不是来自扩张**（论文原文）：准确率在库长到 13 条时**停在 52.5%**，
直到约第 50 轮**开始剪枝之后才继续上升**，第 69 轮达到 62.5%。

benchmark（Table 1）：

```
EditReward-Bench 平均    47.4    （K = 2, 3, 4 三个分档）
GenAI-Bench              64.4
GPT-5                    42.1    -> 差 5.3
相对 GPT-4o baseline     +13.9
```

### 3.5 三个方法学弱点（引用本工作前必读）

1. **62.5% 是在被优化的集合上测的。** rollback 规则连续 77 轮都在最大化那 40 条 val 的
   准确率，而 62.5% 就是那 40 条上的数字。**论文未报告 40 条之外的独立 test split。**
2. **空库基线只在 val 上报了，benchmark 上没有。**
   Table 1（EditReward-Bench / GenAI-Bench）**没有对应的 zero-evolution 数字**。
   即：「进化带来提升」这一主张，**只在被优化的集合上被证明过，
   未在 held-out benchmark 上被证明**。
3. **没有消融表。** 无 no-tools / no-skills / fixed-library 对照。

作者承认库「可能过度特化到反复出现的失败模式」，但未加 test split。
另：100 条 calibration 池与两个 benchmark 是否完全不相交，论文未明确确认。

---

## 4. 与本项目的结构对照（纯事实）

本项目一侧的数字全部来自 2026-08-17 session 的实际核对。

| 维度 | AegisTS | TimeClaw | RewardHarness | 本项目 |
|---|---|---|---|---|
| 下游任务数 | 3（forecast / classification / clustering） | 3 benchmark | 1（偏好评判） | **1**（全报告 `task_consumer_key` 唯一值 = forecast-ridge-sMASE） |
| memory / evolution 的结局指标 | final model 提升 | **benchmark 分数** | val accuracy | **probe 数、harm 计数**（`a5a3.aggregate`） |
| 裁定信号类型 | 连续（perf）+ 连续（issue rate） | 连续（benchmark 分数） | **二值**（ranking 对 / 错） | 连续（sMASE 差） |
| 连续数值的用途 | 进奖励 | 进结论 | **只做诊断** | **当裁定** |
| 每步反馈成本 | 便宜 proxy 模型 | 工具调用 | 一次 LLM 判断 | 全额重训（0.685 s/次，4.4 s/task 矩阵） |
| 接受 / 裁定准则 | RL 回报 | 无（评测型） | **val 超过历史最佳则接受，否则回滚** | `gain >= 0.005`，**无可靠性项** |
| 迭代轮数 | RL episodes | train / test 两段 | **77 轮**，复用同 100 条 | 4 个 Target Task |
| 是否直接用注入 ground truth 计分 | **是**（repair precision / recall，权重 0.5） | 不适用 | 不适用 | 否（`affected_indices` 仅用于 scope 打分，任务书 §3.3） |
| 消融结构 | **3 个消融 + cross_dataset** | k-neighbors 0 vs 3 | **无消融表** | 无 |
| 检索键 | 不适用 | **~20 维指纹 + NL cosine 两段检索** | 库内选子集 | 9 特征分箱签名（gap bank 三条**逐字节相同**） |

本项目其他已核事实（file:line 均在本仓库内）：

```
a5a3 候选池 = outlier_mad, hampel_filter, winsorize   三个全部来自 operators/s1_outlier.py
算子库实际规模 = 19 个 / 5 个家族（impute 7, outlier 4, denoise 5, structural 1, align 2）
MATERIAL_THRESHOLD = 0.005                evaluation/functional/task_episode_harness/runner.py:71
gate 判据 above = gain >= MATERIAL_THRESHOLD   evaluation/functional/task_episode_harness/a5a3.py:403
A5/A3 clean replay 全部测量的最大 |g/SE| = 2.27
注入 ground truth 可得（返回 affected_indices）  evaluation/minipipe/corpus/injections.py:37
```

---

## 5. 本文档未覆盖的内容

- **RewardHarness 源码未读**：rollback 的具体实现、root-cause 分析的 prompt 结构、
  Skill / Tool 的实际 Markdown 样例，均未核对。
- **AegisTS 论文正文未读**：论文声称的主张可能与代码结构有出入；
  本文档只描述代码里能看到的东西。
- **TimeClaw 源码未读**：20 维指纹的具体维度、两段检索的阈值，均未核对。
- **A-Evolve**（`a-evolve/`，arXiv 2602.00359，自称 "The PyTorch for Agentic AI"）
  在本地，与 AegisTS 是**不同工作**（AegisTS 只是嵌在它的目录树里），本轮未读。
- 三个工作与本项目任务书 §5 权限边界（Source 不能批准 / 激活 Skill）的关系，未做分析。
- 三个工作的许可证与可复用性，未核。

## 6. 变更纪律

本文档**只追加不改写**。若后续核对推翻了上面任何一条，
**在文末追加「勘误」节**，不修改原文，并注明勘误依据的 file:line 或论文位置。

---

## 7. 补充核对（2026-08-17 第二批：TimeClaw 源码升级 + 三个新收录工作）

调研人：Kimi CLI 会话（应用户要求，结合本会话此前的对比调研补充）。
本批同样只含事实；新增内容全部追加，原文未动。

### 7.0 本批证据等级

| 工作 | 本批读了什么 | 仍未读 |
|---|---|---|
| TimeClaw | **本地源码**：`memory/fingerprint.py`、`memory/store.py`、`memory/text_embed.py`、`memory/summarize.py`、`main.py` 全文，`evaluation/cik.py:660-720` | 论文 PDF；`evaluation/tsrbench.py`、`tsaia.py` 主体；`tools/server.py` |
| A-Evolve | README 全文 + `agent_evolve/` 目录结构 + `engine/versioning.py` 回滚函数签名 | 四个算法目录的实现细节、论文 PDF |
| DataEvolver | README 全文 | 源码、论文 PDF |
| SkillAdaptor | README 全文（本地仓库 `SkillAdaptor/`） | 源码（Localizer/Linker/Reviser 实现未核）、论文 PDF |

### 7.1 TimeClaw 源码核对（§2 的证据等级从 README 升级为源码）

§2 的 README 级描述与源码一致。以下为源码级新增事实：

**指纹恰为 20 维，特征清单确定**（`fingerprint.py:32-54`，`FINGERPRINT_DIM=20`）：

```
log_length  n_channels_log  missing_rate  irregular_ts  mean_z(恒 0)
std_log  iqr_over_std  skewness  kurtosis  trend_slope_z  trend_r2
acf_lag1  acf_lag_sqrtn  acf_lag_n_over_4
fft_top_freq_norm  fft_top_power_frac  spectral_entropy
changepoint_rate  mean_pairwise_corr  outlier_rate
```

纯 numpy+scipy，无 RNG 无 LLM；多通道按通道求均值聚合（`fingerprint.py:1-22`）。

**检索实现**：bank 内 per-feature z-score 后的 L2 距离（`store.py:383-386`）；
numpy 广播实现而非 FAISS（`store.py:23-27`，规模为几千条时 ~1ms/query）。
z-score 的 scaler 从当前 bank 内容重建，不持久化（`store.py:18-21`）。

**两段检索的文本段默认关闭**：`--text-filter-size` 默认 0，帮助文本写明
"0 disables the text filter (pure fingerprint, default)"，建议开启值 20
（`main.py:146-159`）。文本嵌入为 text-embedding-3-small、1536 维
（`text_embed.py:26-27`）。两段路径为：文本 cosine 先筛 top-N_text 候选，
候选内再按指纹 L2 排（`store.py:358-381`）。

**bank 按 (benchmark, model, split_seed, train_ratio, ratio) 五元组隔离**，
docstring 明说 ratio 进键的原因：不同 ratio 的两次运行可能让一方的 test split
与另一方的 train split 重叠，检索会拉到记录自身的 ground truth
（`store.py:61-89`，尤其 :74-80）。**这是 TimeClaw 的曝光/污染隔离机制，
按目录边界实现，不用 hash chain。**

**schema 防护仅两处**：`feature_names_hash`（特征表漂移即拒绝检索，
`store.py:51-54`、`:160-165`）与 task_id 去重（中断重跑幂等，`store.py:216-232`）。

**train 模式把 ground truth 直接给 agent**：`'train' evaluates on the train half
and writes trajectories to the memory bank (ground truth is revealed to the agent)`
（`main.py:99-101`；CiK 侧 `cik.py:669` `gt_for_prompt = record.get("future_time")
if mode == "train" else None`）。即 memory 记录是"看着答案做出来的"轨迹。

**测试时注入的摘要有意剥掉训练者的 GT 与最终答案**。`summarize.py:109-114`
原文："The trainer's GT answer and final-answer text are deliberately OMITTED:
they cause strong answer-anchoring at test time and consistently degrade
accuracy in our ablations." 注入内容只剩分析主干：工具调用名+参数+截断响应，
外加可选的 `context→forecast` 推理块（训练者对"文本上下文如何决定答案形状"的
迁移性解释，置于摘要最前，`summarize.py:73-97`）。截断预算：单条工具响应
200 字符、推理块 500 字符，top-k=3 注入总量 < 2.5KB（`summarize.py:21-23`、
`:122-123`）。

**跨 family 检索是默认路径**：`--retrieve-same-family-only` 是诊断开关、
默认关（`main.py:136-145`；`cik.py:682` `family_filter=... if
retrieve_same_family_only else None`）。dataset/family 名不作默认相似性判据。

**train/test 划分在 --ratio 子采样之后、每个 family 内部做**；
每个 family 保证 ≥1 条 train 记录，test 时必有同 family 近邻可检
（`main.py:107-118`）。

**"capability-evolution loop" 在源码中的全部内容** =
train 轨迹落盘（`cik.py:715-719`）+ test 时检索注入摘要。
**没有任何改写 instruction / skill / 工具配置的代码路径。**
k_neighbors 仅影响 test 模式（`cik.py:679-686`、`tsrbench.py:371`、
`tsaia.py:596`），k=0 时 retrieve 返回空、refs 为空，即 no-memory 消融。

### 7.2 A-Evolve（新收录）

**位置**：`a-evolve/`（注意 §1.1 已述：AegisTS 嵌在其目录树里，两者是不同工作）。

**定位**：自进化 agent 的**基础设施/框架**，非单一算法。
自称 "The PyTorch for Agentic AI"（`README.md:8-9`）；
position paper：*Agentic Evolution is the Path to Evolving LLMs*，arXiv 2602.00359。

**Workspace 即文件系统契约**（`README.md:244-255`）：所有可进化状态是一个标准目录
（`manifest.yaml` / `prompts/system.md` / `skills/` / `tools/` / `memory/`），
进化引擎通过 LLM 文件操作 mutate 任何 agent，无需知道其内部实现。

**五相循环**（`README.md:257-273`）：Solve → Observe → Evolve → **Gate** → Reload。
Gate = 在 holdout 任务上验证 mutation，回退经 git rollback——代码存在：
`agent_evolve/engine/versioning.py:89 rollback()`、`:123 rollback_to_tag()`，
每次接受的 mutation 打 git tag（evo-1, evo-2, …）。
收敛判据为 EGL（Evolutionary Generality Loss）稳定或达到 max_cycles。

**4 个参考算法**（`README.md:298-307`）：`adaptive_evolve`（per-claim 反馈分析
+ meta-learning）、`adaptive_skill`（LLM mutation + bash 工具）、
`skillforge`（EGL gating）、`guided_synth`（memory-first + LLM 引导干预合成）。

**自报数字**（`README.md:30-115`，注明 "Data checked March 2026"）：
单一 Claude Opus-4.6 底座 + 参考算法，10 个 benchmark 对同模型 baseline
提升 +2.2pp ~ +15.2pp（MCP-Atlas 79.4% +3.4pp；SWE-bench Verified 76.8% +2.6pp；
Terminal-Bench 2.0 76.5% +13.0pp；SkillsBench 34.9% +15.2pp 等）。
**均为 README 自报，未独立核对；baseline 是"同底座未进化"，不是第三方方法。**

**MCP-Atlas 进化前后对照**（`README.md:133-174`）：`system.md` 20 行未变；
新增 5 个 SKILL.md + 6 条 episodic memory。原文："5 targeted skills outperformed
10 generic ones." 即其展示的进化产物 = 新增 skill 文件 + 少量 episodic memory，
**不是改写 system prompt**。

**同组相关论文**（`README.md:118-120`）：*Adaptive Auto-Harness*（arXiv 2606.01770，
开放任务流上的持续自改进）；*Harness Updating Is Not Harness Benefit*
（arXiv 2605.30621，7 个 evolver 模型 × 6 个 solver × 3 个 benchmark，
区分"哪个模型产出的 harness 更新好"与"哪些模型从更新中受益"——
更新者与受益者分离）。

### 7.3 DataEvolver（新收录）

**位置**：`DataEvolver/`。**论文**：arXiv 2606.07001（RUC datalab）。

**领域**：LLM 训练数据准备（原始数据 + 少量 seed 样例 → seed 对齐的训练集），
**非时序**。两级自进化（`README.md:36-38`）：

- **算子级**：DAG 编排、结构缺口检测、缺失算子合成；
- **pipeline 级**：trial run → Pilot LLM judge → experience reflow → 下一轮
  understanding/orchestration 更新。

工作流（`README.md:251-256`）：

```
understanding → orchestration → operator_evolution → instantiation
             → trial_run → quality_check → experience → (refine or full run)
```

**验收方式**：quality gates 通过后才允许 full run（`README.md:148`）。

**自报数字**（`README.md:219-229`）：~12% 平均相对增益（vs 较弱数据准备设置）、
7 个 benchmark（instruction / MC-QA / math / SQL）、~40% 平均摊销 token 成本下降。
**README 自报，未核对论文与代码。**

**消融**（`README.md:244-248`）：去算子级进化 → pipeline 可执行性/连贯性下降；
去 pipeline 级进化 → seed 对齐下降。

**两点与本项目审计相关的结构事实**：

- 支持**人工加算子**（`dataevolver op add`，`README.md:173`、`:189-195`）——
  进化回路并非全自动闭环；
- experience reflow 是**规则聚合、确定性**的，不是 LLM 改写
  （`README.md:214`，FAQ 原文）。

### 7.4 SkillAdaptor（新收录，本地仓库）

**位置**：`SkillAdaptor/`。**论文**：arXiv 2606.01311（ZJU + Ant Group）。

**进化对象**：agent 工作区里的 **`SKILL.md` 文件**（`README.md:22`、`:25`），
产物落在 `skills/<id>/SKILL.md` 并同步到目标 agent 的 skill 路径。

**组件链**（`README.md:24`，一级标题原文）：

```
Localizer   失败轨迹上找最早坏步 t★（step-level attribution）
Linker      给"哪条注入的 skill 导致了 t★"打分归因
Reviser     修补已有 skill   /   Generator 新建 skill
Validator   在 held-out 任务上重跑，只在改善时采纳
```

**held-out 纪律**：`input_task/` 的任务约 20% 留作 validation Q′；
另有可选 `test_task/` 额外 held-out（`README.md:99-107`）。
**Retriever-gated inject**：skill 只在相关时才挂载（`README.md:27`）。

**轨迹门槛**：只接受带工具级 action 的轨迹；无合法轨迹时 raise
`TaskExecutionError`，**不编造轨迹**（`README.md:111-127`）。

**可接入真实 harness**：OpenClaw / Claude Code / Codex CLI / Hermes Agent
（`README.md:60-77`）；benchmark 执行器：PinchBench / WebShop / Claw-Eval
（`README.md:29`）。

### 7.5 §4 对照表的补充行（新表，原表未动）

| 维度 | A-Evolve | DataEvolver | SkillAdaptor |
|---|---|---|---|
| 领域 / 任务数 | 通用 agent，10 个 benchmark | LLM 训练数据准备，7 benchmark / 4 类 | agent 技能适配，3 个执行器 |
| 进化对象 | workspace 文件（prompts / skills / memory） | 算子 DAG + pipeline 配置 | `SKILL.md`（patch 或新建） |
| 裁定信号 | benchmark 分数（holdout） | trial 分数 + Pilot LLM judge | held-out 任务改善 |
| 失败归因 | 算法而异（adaptive_evolve = per-claim 反馈分析） | DAG 结构检查 + LLM 评估 | **Localizer 最早坏步 t★ + Linker skill 归因** |
| 验收 / 回滚 | Gate holdout + **git rollback** | quality gates 不过则不能 full run | Validator 改善才采纳 |
| 是否全自动 | 是（但提供人工加算法接口） | **否**（支持人工加算子） | 是（插件进真实 harness） |
| 证据等级 | README + 部分代码 | 仅 README | 仅 README |

### 7.6 本批对 §5「未覆盖」清单的影响

- 「TimeClaw 源码未读」→ **已读**（7.1），§5 该行过时；
- 「A-Evolve 仅 README 开头」→ **README 全文 + 回滚代码签名**（7.2），该行部分过时；
- 新增未覆盖项：SkillAdaptor 与 DataEvolver 均只读了 README（实现未核）；
  RewardHarness 源码仍未读（维持 §5 原状）；A-Evolve 四个算法目录未逐文件核；
  TimeClaw 的 `tsrbench.py` / `tsaia.py` 主体与 `tools/server.py` 未读。
