# 批级构造策略 Harness：框架规格 v0.1（2026-09-13）

状态：**执行者起草（Opus），供 Planner（Astra）复核、用户裁定。** 本文不是已批准协议，不启动实验。

授权状态：
- 用户 2026-09-13 对话中确认三点方向——决策粒度改为批级、Fast 的 Workflow 是"观察→决策→执行"整体、先跑通再进化——并给出收益阶梯：**必须优于 Static，其次优于 Random，再者优于单算子上限（最后一项不强求）**。
- 决策粒度从逐序列（AGENTS §5.3，2026-09-07）改为批级，属于路线变更，按 AGENTS §9.1 归用户；**DECISIONS.md 条目待维护者写入**，写入前本文的路线部分标为"用户口头确认、待记录"。
- 本文引用的代码与数字均已核对（文件路径见 §12）；引用的实验结论限于其原始范围。

---

## 0. 一句话

把 Fast 的决策单位从"一条序列"改成"一批训练数据"，输出一条**可执行的构造策略**（观察 → 每条序列的处理 → 整批共同训练），价值只在**整批共同训练的配对差**上度量；Slow 在批边界修改一个知识面；先把这条链在两份数据上跑完并交出基线阶梯，再优化效果。

---

## 1. 问题定义

### 1.1 对象

给定一个批 B =（数据集 D，截止 t）：32 条实体在训练段 T=[t−672, t) 上各 433 个父对（X 192 → y 48）。一条**构造策略** π 把批内可观察量映射为每条实体的训练材料构造（算子、donor 规则、混合权重、步数），运行时据此生成材料，与原始父对一起训练**一个**共享 Consumer A。

价值定义沿用 `docs/DATA_QUALITY_AND_FEEDBACK_MODEL.md` 的正典：

```
V(π_b ; π_a | A, Q) = E_ω[ R_Q(A(D_{π_a}; ω)) − R_Q(A(D_{π_b}; ω)) ]
```

即两套完整训练材料之间、对训练随机性 ω 取期望、条件于 Consumer 与工作负载 Q 的配对差。正号 = π_b 更好。

### 1.2 为什么是批级（证据）

- 逐实体响应被 seed 噪声主导：实体间 std / seed 内 std ≈ 0.28–0.34，与 13 个观察字段 |ρ| ≤ 0.14（`docs/OPUS_NOTES_FEEDBACK_OBJECT_2026-09-13.md` fact 5）。
- 共享模型下单条序列的材料改动扩散到所有序列的损失：DEV-AUG-DONOR-PATTERN-PROBE 中 a65/a75 最大的 P−S 逐实体差来自材料完全相同的实体（`_scratch/ts_aug_donor_pattern_probe/REPORT.md` 首页 4）。
- 唯一跨块跨作业方向一致的读数是全体统一处理的 N→R。

结论：批是**可测量的最小单位**；决策单位对齐到它。这不否定 Pattern：策略可以是条件化的（§4），但评价永远在批上。

### 1.3 与 TSAA 类方法的界线

批级策略 + 验证损失 = 增强策略搜索。本框架的贡献只能落在：(i) Fast 读观察 + 知识提出策略而非枚举；(ii) Slow 写回可迁移知识；(iii) 知识跨作业/跨数据集复用。**M4/M5（§11）是贡献成立的必要条件，不是可选项。**

---

## 2. 单位与数据

### 2.1 数据集与作业

| 数据集 | 文件 | 总小时 | roster | 已有作业 | 曝光状态 |
|---|---|---|---|---|---|
| electricity | `shared_tsq_datasets/electricity/electricity.csv` | 26304 | 32（字符串排序前 32 列） | a20/a30/a40（preflight）、a50/a60（confirmation）、a55/a65/a75（overnight/probe）、a94（source_grounding） | 全部 development（整文件曾作开发材料） |
| traffic | `shared_tsq_datasets/traffic/traffic.csv` | 17544 | 32 | a20/a30/a40、a50/a60 | 同上 |

作业按 `t = 24·floor(a·T_total/24)` 由时间名单预先指定，**不按效果选**。窗口沿用现役几何：

```
T   = [t−672, t)           训练（父对 433/实体，stride 1）
C_A = {t, t+48}             Support（Fast/Slow/搜索可见）
C_B = {t+96, t+144}         Delayed（held-in 主估值）
E   = {t+192,…,t+336}       Held-out（冻结后一次性打开）
```

scaler 只在 T 上拟合、每实体一套、floor 1e−6；所有臂相同。

### 2.2 形成作业与复用作业

- **形成作业**：Fast/Slow 在其上适应（held-in 多轮）；electricity 首轮固定为 a55 → a65 → a75（已有 N/R/U 缓存）。
- **复用作业**：知识冻结后只跑 Fast，零反馈；electricity 用 a85（t=22344，E 末 22728 < 26304）与 a94（t=24720）；traffic 用 a50/a60 之后的时间名单另定。
- 三段都是 development；报告中不得改称 fresh。

---

## 3. 观察层

Fast 只见 T。观察分两层，全部由运行时确定性计算，**不返回"应该怎么处理"**。

### 3.1 逐实体观察（现役 13 字段 + 本轮新增描述器）

| 来源 | 字段 | 备注 |
|---|---|---|
| `observations.py: entity_observation` | mean, std, missing_count, head168_mean, tail168_mean, standardized_trend_per_100h, last168_std, full_T_std, last168_std_over_full_T_std, lag24_corr, lag168_corr, nondc_spectrum_top5_energy_share, standardized_abs_p95, standardized_abs_max, n_parent_pairs | 现役，不改 |
| `dp_materials.py: compute_pattern` | r_head, r_tail（前后半段去趋势 lag-24 相关） | probe 包 |
| `analyze.py: part2`（mechanism 包） | parent_d1, parent_ydev（T 内固定 5-NN 描述：最近真实输入距离、目标对邻域目标偏离；k=5、排除 ±48h） | 只读 T；参考集与查询窗口重叠 70–75%，报告时注明 |
| `analyze.py: part1` | 对给定 donor 规则：父–donor X/y RMS、水平差、形状差 | 依赖策略中的 donor 规则，按需计算 |

**probe-fit 观察**（可选，计费）：用无增强模型 N 的一次拟合得到"线性插值 MSE/父 MSE"（残差平均比，mechanism 包 `part3.residual_averaging`）。它需要 K 次拟合，计入该批预算；M2 不启用。

### 3.2 批级摘要

对上表每个字段给 32 实体的 min/p25/median/p75/max，加 roster 大小、T 长度、父对数、上批（若有）的策略与配对差摘要。Fast 提示词只放批级摘要 + 逐实体表（32 行 × 选定字段，≤ 4 KB）。

### 3.3 禁止项

不见 C/E 任何行；不见其他批的原始 Episode；不见实体真实名称（用 entity_0…31，保持 §4.3 的"无 ID"检查有意义）。

---

## 4. 策略语法（Policy）

### 4.1 形状

```json
{
  "policy_version": "v0.1",
  "default": {"steps": [{"op": "timemixup", "donor_rule": "R", "w": 0.25}]},
  "rules": [
    {"when": <applicability AST>, "steps": [{"op": "timemixup", "donor_rule": "U", "w": 0.25}]},
    {"when": <applicability AST>, "steps": []}
  ],
  "rationale": "<≤500 字>",
  "observation_fields_used": ["lag24_corr", "r_head", "..."]
}
```

- `steps`：0–2 步；`op ∈ {timemixup, freqmask, freqmix}`；`[]` = identity。
- `timemixup` 参数：`donor_rule ∈ {R, U}`（`_derangement` 随机错排 / 同小时错排，`dp_materials.donors_R/donors_U`），`w ∈ {0.10, 0.25, 0.50}`。
- `freqmask`：`mu ∈ {0.05, 0.10, 0.20}`；`freqmix`：`mu` 同上，donor 规则同 timemixup。
- `when`：复用 `applicability.py` 的 AST（`{"feature","op","value"}` 叶 + all/any/not/const），feature 只能是 §3.1 字段；规则按顺序首个命中生效，无命中用 default。
- 允许的参数取值集合是**运行时常量**（General facts 的一部分），Fast 不能发明新值；扩集合是公共能力变更，所有臂同得。

### 4.2 编译

`policy.compile(policy, observations) → assignment`：32 条 `{entity_index: steps}`。同一 assignment 向量的两条策略是**材料别名**，共享拟合（probe 包 `material_aliases` 逻辑）。

### 4.3 校验（fail-closed）

1. 语法与取值集合；
2. 不含实体名/UID/数据集名/作业名/seed（`knowledge.validate_guidance_text` 的禁词 + `FORBIDDEN_UID_PATTERN`）；
3. 编译后每实体 steps 合法；
4. dry-run：与 incumbent 的 assignment 比较，记录变更实体数（只记录，**不设比例门**，AGENTS §5.7.2）。

### 4.4 随机策略生成器

`policy.random_policy(seed)`：从同一语法均匀抽 default 与 0–2 条规则（规则谓词从字段 × 分位阈值抽）。Random 臂与 Fast 臂共用此语法，保证"同策略空间、同预算"。

---

## 5. 执行与评价（运行时）

沿用 probe 包三阶段，参数不变：

| 步骤 | 实现 | 说明 |
|---|---|---|
| 阶段 1 只读 T | `dp_materials.py` 扩展：按 assignment 生成每实体子对（R/U/w/mu 组合先各算一次再选） | 冻结 `stage1_frozen.json` 后才读 C |
| 拟合 | `dp_train_cell.py`：一个 (job, policy_id, seed) = 一个子进程，共享 MLP 192→128→64→48，AdamW 1e−3/1e−4，2000 更新，批 64，父/子损失 0.5/0.5 | seed 列表固定 `20260918/19/20/21`（K=4）；batch 流 `800000+100r+9` |
| 评分 | `score_h1.score_predictions`，normalized MSE 实体宏均值 | C_A/C_B 在拟合进程内评；E 只在冻结后由独立进程评 |
| 配对差 | `dp_readout._paired`：d = loss(A) − loss(B)，正 = B 好；均值、样本 SD、SE、符号计数、df=3 t 区间；δ_job = 1% × incumbent 的 C_A 均值 | 四种读数：candidate_improvement / candidate_harm / practically_close / uncertain |

**每次策略评价 = K=4 次拟合**（缓存命中则 0）。本机 11–17 s/拟合 → 1 分钟/策略。

---

## 6. Fast Path

### 6.1 一次调用 = 一条策略

输入：General facts（固定：动作空间、参数集合、角色、预算、"你不能改模型/评分/发明动作"）+ 当前 Skill（General 研究方法 + 已加载的 Specific）+ 批级摘要 + 逐实体观察表 + 任务上下文（incumbent 策略、上一轮配对差摘要、本轮实验指令）。
输出：§4.1 的 JSON。解析失败 → 一次重试 → 仍失败记 `FAST_INVALID`，本轮用 incumbent。

### 6.2 批内轮次

每批 R 轮（M2 定 R=2）：

```
r=1: Fast 提出 π₁ → 评价 π₁ vs incumbent（C_A）→ 若 π₁ 在 C_A 上为 candidate_improvement 则 incumbent←π₁，否则保留
r=2: Fast 看到 r=1 的 C_A 配对差摘要 → 提出 π₂ → 同上
批结束：incumbent 冻结为该批交付策略；C_B 打开作 delayed 读数；E 只在整段形成结束后打开
```

incumbent 初值：electricity 首批 = 统一 `timemixup R w=0.25`（有缓存，且是历史强固定方案）。**Fast 决定的是提出什么，交付由 C_A 规则决定**——这与 guidance-evolution 里"select 无权改交付"的问题不同：这里 Fast 的提案本身就是被评价的候选，规则只裁定它是否替换 incumbent。

### 6.3 LLM 传输

沿用 `_scratch/ts_aug_source_grounding_pilot/backend.py`（账户/传输故障分类、有界重试、原子预算记账）。模型与端点按 AGENTS §5.7.1 与用户当前指定（现役：`cpa-grok-4.6` @ `http://127.0.0.1:8318/v1`，返回模型身份须匹配）；不默换模型。

---

## 7. 反馈与 Slow

### 7.1 反馈记录（批级，唯一对象）

```json
{"job": "a65", "round": 1, "policy_a": "<id>", "policy_b": "<id>", "block": "C_A",
 "d_by_seed": [..4..], "mean": .., "sd": .., "se": .., "signs": "3/1/0", "delta_job": .., "reading": "uncertain",
 "assignment_diff_entities": 9, "policy_b_text": {...}}
```

C_B 记录在批结束后追加；E 记录只进外部报告，不进 Slow。逐实体损失向量存盘供报告，**不进 Slow 提示词**（避免逐实体拼分）。

### 7.2 Slow 的编辑面

批边界一次调用，读：累计反馈记录（本形成段）、当前 Skill、各批的批级摘要。可改**一个**面：
- Specific 正文（≤1200 字，`knowledge.validate_guidance_text`）+ 其 applicability AST；或
- General 研究方法正文（"先比什么、证据不足时怎么查"）。
不能改：参数集合、评分、交付规则、Consumer。写回按证据等级绑定可写内容（笔记 §2 权限层）：不确定 → 只许申请复核同一对或弃权；有依据 → 软先验；跨批复现 → 才允许写适用条件。禁令在任何等级都写不出来。

### 7.3 版本

整数版本号 + 全文内联（`GuidanceLineage`），无哈希。

---

## 8. 基线阶梯与臂

### 8.1 阶梯（用户 2026-09-13 定）

| 级 | 臂 | 定义 | 要求 |
|---|---|---|---|
| L0 | **Static** | 无增强（N） | **必须优于** |
| L1 | **Random** | 同策略语法、同轮次 R、同 C_A 替换规则，策略由 `random_policy(seed)` 抽；每作业 3 个随机流取均值 | **必须优于** |
| L2 | **单算子上限** | 事后在统一策略枚举集（{timemixup R/U × w} ∪ {freqmask mu} ∪ {freqmix}，≤ 12 条）中按 C_B 取最优 | 优于则记，**不强求** |
| — | Fixed-incumbent | 统一 timemixup R w=0.25 不动 | 参照，回答"Fast 有没有改坏" |
| — | A3 | Fast+Slow，Skill 从空开始 | 主臂（M2–M4） |
| — | A5 | Fast+Slow，Skill 从上一数据集/上一段形成的冻结 Skill 开始 | M4/M5 |

### 8.2 "正向收益"的判定（预注册）

估值块：形成作业用 **C_B**（Fast/Slow 消费了 C_A，C_A 只作描述）；复用作业用 **E**。
判定：对每个比较（A3 vs Static、A3 vs Random），三个形成作业的配对差（K=4）中 ≥ 2/3 作业均值 > 0 且逐 seed ≥ 3/4 同号，且三作业等权均值 > +δ；否则记"未达"。不做显著性，不按结果改 δ、K、作业或语法。

### 8.3 "跑通"的定义（M2 验收，不要求正号）

流程在指定作业上端到端完成所有臂；Fast 输出有效率 ≥ 90%；阶梯表与四种出口齐全；无残留进程；预算记账闭合。

---

## 9. 评价协议

1. 形成段：作业按时间顺序依次进入；每作业 R 轮；批边界 Slow 一次；段末冻结 Skill 与最终 incumbent。
2. 复用段：冻结 Skill，只跑 Fast（R=1，无替换规则——直接交付 Fast 提案；若无效则 incumbent），零反馈；E 一次性打开。
3. 第二数据集：A5 载入 electricity 冻结 Skill，与 A3（空 Skill）同场；同预算。
4. 所有臂同 K、同 seed、同作业、同 R；只有知识初值不同。
5. 出口：支持 / 伤害 / 不确定 / 不可解释，每个比较单独记；不以追正号为由追加。

---

## 10. 预算与停止

| 项 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|
| 作业 | elec a55/a65/a75 | 同 | + a85/a94 复用 | traffic 3 形成 + 2 复用 |
| 臂 | Static、Random×3 流、A3（无 Slow） | + Slow | + A5（源=M3 冻结 Skill） | Static、Random、A3、A5 |
| 拟合上限 | 3 作业 × (Random 3×8 + A3 8) = 96 | 96 | 复用 2 作业 × 3 臂 × 4 = 24 | ≤ 200 |
| LLM 调用 | Fast 2/批 → 6 | + Slow 2 | Fast 2 | ≤ 40 |
| 墙钟 | ≤ 2 h | ≤ 2 h | ≤ 1 h | ≤ 4 h |

通用：单拟合 300 s 超时、每格 1 次瞬态重试、总重试 ≤ 5%；GPU 被占等待 ≤ 15 min；预算耗尽 PARTIAL 收口；0 新哈希。

---

## 11. 里程碑

| 里程碑 | 内容 | 验收 | 状态 |
|---|---|---|---|
| **M0 headroom** | 零 LLM 固定策略网格确认批级杠杆能动指标 | donor 规则：无 headroom（probe，已完成）；混合权重：待 `_scratch/ts_aug_mechanism_evidence/NEXT_EXPERIMENT.md`（24 拟合） | 半完成 |
| **M1 零 LLM 批级流水线** | 策略语法 + 编译 + 校验 + 随机生成器接入 `dp_*`；阶梯表自动产出 | 用语法表达 probe 的 N/R/U/P/S，编译后材料与 probe 逐位相同；Random 臂能跑 | 待做（估 1 天） |
| **M2 Fast 提案** | 一批一策略，R=2，Static/Random/A3(无 Slow) 三臂 | §8.3 跑通定义；阶梯表 | 待做（估 1 天 + 2 h 运行） |
| **M3 Slow 更新** | 批边界一次编辑；a55→a65→a75 形成 | ≥1 次编辑过校验并改变下一批 Fast 的提案；阶梯表 | 待做 |
| **M4 跨作业复用** | 冻结 → a85/a94 只跑 Fast | A3-frozen vs Static vs Random 在 E 上的阶梯 | 待做 |
| **M5 第二数据集** | traffic：A5（elec Skill）vs A3 vs Static vs Random | 阶梯 + A5 vs A3 | 待做 |

"做完" = M1–M5 全部产出阶梯表与报告；不要求任一级为正。

---

## 12. 与现有代码的映射

| 组件 | 现有文件（已核） | 动作 |
|---|---|---|
| 作业/窗口/Consumer 常量 | `_scratch/ts_aug_donor_pattern_probe/dp_config.py` | 复用；加 traffic 与 a85/a94 作业表、参数集合 |
| 数据切片/父对/scaler | `dp_data.py` | 复用 |
| 材料生成（R/U/pattern/children） | `dp_materials.py`（`donors_R`、`donors_U`、`mix_child`、`compute_pattern`） | 扩展：按 assignment 生成、w/mu 参数化 |
| 增强原语 | `_scratch/ts_aug_source_grounding_pilot/aug_ops.py`（freqmask/freqmix/timemixup） | 复用；w/mu 由调用方传入而非只读 cfg（需一个薄包装，不改原文件） |
| 拟合/评分 | `dp_train_cell.py`、`score_h1.py`、`model_mlp.py` | 复用 |
| 运行器/看门狗/预算 | `dp_run.py`、`watchdog_launch.py` | 复用；加"策略级"格 |
| 配对差与阶梯 | `dp_readout.py` | 扩展：阶梯表、C_B 主估值 |
| 逐实体观察 | `ts_aug_source_grounding_pilot/observations.py` | 复用 |
| 谓词 AST | `applicability.py` | 复用为策略 `when` |
| Skill 校验/版本 | `knowledge.py` | 复用 |
| LLM 传输 | `backend.py` | 复用 |
| Fast/Slow 提示 | `fast_call.py`、`slow_call.py` | **重写**为批级（一批一策略）；保留解析/重试骨架 |
| 新增 | `policy.py`（语法/编译/校验/随机生成）、`batch_observation.py`（摘要 + 表）、`feedback_record.py` | 新写，≤ 3 个文件 |

不做：新平台、哈希链、通用 Schema、Router/聚类器、第二套评价框架。

---

## 13. 风险与明确不做

- **塌成 LLM-AutoML**：若 M4/M5 中 A5 不优于 A3 且 A3 不优于 Random，贡献不成立；如实报告，不改阶梯。
- **噪声底**：δ ≈ 0.004、SE ≈ 0.006；K=4 固定，不追加 seed 追正号；M0 未证明 headroom 的杠杆不进语法（当前：w 待验，donor 规则已无 headroom 但保留在语法中作为消融项）。
- **曝光**：两份数据全为 development；报告不得写 fresh/held-out 泛化。
- **不做**：逐实体价值标签、oracle 上界拼分、按结果换作业/阈值/语法、恢复旧 A5 课程、修改交付规则、开放密封数据。

---

## 14. 判词等级

每份报告用 AGENTS §8 的六级之一：CAPABILITY / MECHANISM / INFRASTRUCTURE / INSTRUMENT / NEGATIVE / INCONCLUSIVE。M1 最高 INFRASTRUCTURE；M2–M3 最高 MECHANISM；只有 M4/M5 的 A5 vs A3/Random 正读数才可能记 CAPABILITY（development 级）。

---

## 附：与旧记录的关系

- AGENTS §5.3/§5.7 的逐序列决策单位：本文提议改为批级；历史读数与判词保留，不并表。
- `docs/OPUS_NOTES_FEEDBACK_OBJECT_2026-09-13.md`：反馈对象定义与权限层沿用。
- probe/mechanism/overnight 三包：作为 M0 与 M1 的直接前置，其结论只约束原配置。
