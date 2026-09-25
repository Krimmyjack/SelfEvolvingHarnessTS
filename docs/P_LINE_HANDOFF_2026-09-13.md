# P 线交接与收尾：批级底座 + 固定流程控制器在两个数据集上的结果

日期：2026-09-13（晚）；收尾修订 2026-09-13（更晚，按 Planner 复核意见：0 新拟合、0 实验 LLM、不改历史数值）
写给：Astra（Planner）。执行者：Opus。
性质：执行结果交接 + 有界收尾。所有数字可由列出的运行目录重算；本文不含任何未跑过的推测数值。P 线自此作为固定流程基线保存，不再追加同类实验。

## 0. 一句话

共享底座与 P 线控制器已经端到端跑通并提交；在 electricity 与 traffic 两个数据集、四个作业上，**框架完整、Slow 修改送达并改变了 Fast 的行为，但本轮已测候选未找到相对 Fixed 的可靠增量**。这个结果支持停止当前"固定流程 + 重复提案"的方式；它**不**足以证明公共动作空间接近上限——搜索覆盖有限（见 §5），且 W 线在 electricity/a95 已有随机候选明显优于 Fixed 的记录（Planner 复核所述，非本文核对）。

## 1. 交付物（分支 `p-line/batch-base`，6 个提交）

| 提交 | 内容 |
|---|---|
| `5a6dfc7` | `methods/ttha/batch_base/`：spec / data / augment / observe / policy / context / materials / train / feedback / commit / budget / llm。`evaluation/main_protocol_p4/batch_base_smoke.py` 28/28（含一次与 probe 逐位相同的重训） |
| `119a267` | `methods/ttha/p_line/`：knowledge / prompts / controller / run / report；两处底座修正（别名格复用、"seed" 不作禁词） |
| `2689805` | `w_headroom.py`（w 0.25→0.5 检验）、`anchor_scan.py`（按效应/噪声选作业） |
| `a4ab6ec` | spec 禁词加 traffic 作业标识 |
| `9928387` | `random_baseline.py`（同语法随机材料对照） |
| `da10e36` + 本次 | 本交接文档 |

W 线可从 `9928387` 起 fork 底座。底座接口见 `methods/ttha/batch_base/README.md`：七个操作（overview / inspect_data / build_material / inspect_material / evaluate / compare / commit）各对应一个函数；行权限按阶段物理限制（`context.STAGE_ROWS`）；材料按 assignment 键别名，同 assignment 不重训。

对照文档 §4 修正项的处理：K 与作业数按任务书公共配置（M1 K=3 两作业；M2 因换数据集改 K=4）；替换门保留为 P 线特征（"mean d > 1%×incumbent C_A 且 ≥n−1 seed 同号"），W 线按规格由 Fast 自行 commit；局部损失诊断存盘并供 Slow 读，标注为非因果；Slow 可编辑面在 P 线只有 construction_guidance 与 experiment_guidance（observation/decision 两面在固定流程中无杠杆，这是 P/W 的刻意差异）。

## 2. 运行汇总

| 运行 | 数据/作业 | 拟合 | LLM | 目录 |
|---|---|---|---|---|
| P 线 M1 | electricity a55→a65，K=3 | 27 | 7（1 次 401） | `_scratch/p_line_m1/` |
| w_headroom | electricity a55/a65/a75，K=4 | 24 | 0 | `_scratch/w_headroom/` |
| 锚点扫描 | traffic a30–a90 七段 + a95，K=4 | 64 | 0 | `_scratch/anchor_scan_traffic{,_a95}/` |
| P 线 M2 | traffic a40→a90，K=4 | 36 | 7 | `_scratch/p_line_m2_traffic/` |
| Random 材料对照 | traffic a40/a90，3 抽样，K=4 | 40 | 0 | `_scratch/random_baseline_traffic/` |

LLM 一律 `cpa-grok-4.6` → 返回 `grok-4.6-build`，无换模型。全部 EXPOSED_DEVELOPMENT。**M1 状态 = COMPLETE with PARTIAL**（a65/H1 第 2 轮 401 认证失败，未回补，回补即事后拟合）。

## 3. 三列判词（M1 与 M2 相同）

| framework_complete | treatment_present | utility_supported |
|---|---|---|
| True | True | 未建立 |

- 交付：M1 三分支、M2 三分支，**全部 = Fixed**（TimeMixup R w=0.25 全体统一）。
- vs Fixed（C_B，预注册）：恒为 0。
- vs None：a55 C_B +0.0090（3/3）、a40 +0.0075（4/0）supported；a65、a90 不确定。归属为 Fixed，不是 Agent。

## 4. 用户阶梯（证据范围按收尾修正）

| 级 | 结论 | 证据 | 归属 / 限定 |
|---|---|---|---|
| 比 Static 好 | 交付成立 | §3 | Fixed |
| 比 Random 好 | **仅"胜过随机抽出的材料"成立**；"胜过公平随机搜索流程"未测 | Fixed→随机材料 C_B 6/6 为负；回顾性重放（§5b）显示带同一替换门的随机控制器在此候选集上同样交付 Fixed | Fixed；两种控制器在已存候选上不可区分 |
| 比单算子上限高 | 本轮未找到 | 10 个真实 Fast 候选 + 4 手工 w/donor 变体 + 6 随机材料：无一在 C_A 与 C_B 同时 >δ 优于 Fixed | 覆盖有限，不外推到动作空间 |

## 5. Fast 候选全表（相对 incumbent Fixed；正 = 候选好）

计数：**10 个真实候选，2 次主动 KEEP（M1 a65/H1 r2 不是 KEEP，是失败；真实 KEEP 只有 M2 a90/H1 r2 一次），1 次认证失败**。下表 12 行 = 10 候选 + 1 KEEP + 1 失败。

| 运行 | 分支/轮 | 提案 | 改动实体 | C_A mean（符号） | C_B mean | 结果 |
|---|---|---|---|---|---|---|
| M1 a55 | H0 r1 | U 默认 + 低周期→FreqMask 0.1 | 32 | −0.0094（1/3） | — | 未过门 |
| M1 a55 | H0 r2 | R 默认 + lag168<0.4→FreqMask 0.1 | 5 | +0.0018（2/3） | — | 未过门 |
| M1 a65 | H0 r1 | U 默认 + r_gap/塌缩→identity | 32 | −0.0012（1/3） | — | 未过门 |
| M1 a65 | H0 r2 | U 默认 + 塌缩/尖峰→identity | 32 | +0.0000（1/3） | — | 未过门 |
| M1 a65 | H1 r1 | R 默认 + lag168 低四分位→w 0.1 | 8 | −0.0094（1/3） | — | 未过门 |
| M1 a65 | H1 r2 | — | — | — | — | **401，PARTIAL** |
| M2 a40 | H0 r1 | U 默认，无规则 | 32 | +0.0026（3/4，0.65δ） | −0.0021（2/2） | 未过门 |
| M2 a40 | H0 r2 | U 默认 + lag168<0.75→FreqMask 0.1 | 32 | +0.0004（3/4） | −0.0057（1/3） | 未过门 |
| M2 a90 | H0 r1 | U 默认 + 2 规则 | 32 | +0.0006（3/4） | −0.0013（1/3） | 未过门 |
| M2 a90 | H0 r2 | R 默认 + 3 规则（含 FreqMix U 0.1） | 26 | −0.0016（1/3） | −0.0022（1/3） | 未过门 |
| M2 a90 | H1 r1 | R 默认 + r_gap 高四分位→identity | 8 | −0.0013（0/4） | +0.0006（2/2） | 未过门 |
| M2 a90 | H1 r2 | KEEP（Fast 自行） | — | — | — | 保留 |

（M1 的 C_B 候选差在 `_scratch/p_line_m1/REPORT_TABLES.md`，范围 −0.010…+0.003，全部 uncertain。）

读法：
- 门没有误判过：最接近过门的 M2 a40/H0 r1 在 C_B 上反转为负。
- 搜索覆盖：默认 10/10 是 TimeMixup；w 从未用 0.5，mu 从未用 0.05/0.2；FreqMix 1 次；观察字段基本只用 lag168_corr / r_gap / last168_std 比。**这是"没找到"而非"没有"的直接原因。**
- Fast 候选相对 Fixed 的 C_B 差均值 −0.0021（M2 五个），随机材料 −0.0048：Fast 的提案系统性比随机"更不坏"。这是 Fast 有信息的证据，不是有增量的证据。

### 5b. 回顾性重放：把 P 的替换门用在随机材料上

**这是重放分析**，用 `_scratch/random_baseline_traffic/result.json` 已存的 C_A 配对差，按 P 线原规则（mean d > 1%×incumbent C_A 且 d>0 于 ≥n−1 seed）判定；无新拟合。Random 包本身**固定交付 Fixed、另行评分三条随机策略**，没有执行随机搜索选优。

| 作业 | 随机策略 | C_A d by seed（vs Fixed） | mean | δ | 同号 | 门 |
|---|---|---|---|---|---|---|
| a40 | rand1 | −0.0089, +0.0042, +0.0028, +0.0051 | +0.0008 | 0.0040 | 3/4 | keep |
| a40 | rand2 | −0.0175, +0.0027, −0.0016, +0.0129 | −0.0009 | 0.0040 | 2/4 | keep |
| a40 | rand3 | −0.0102, +0.0009, −0.0108, −0.0008 | −0.0052 | 0.0040 | 1/4 | keep |
| a90 | rand1 | −0.0064, −0.0073, −0.0039, −0.0030 | −0.0052 | 0.0028 | 0/4 | keep |
| a90 | rand2 | −0.0028, −0.0067, −0.0013, −0.0019 | −0.0032 | 0.0028 | 0/4 | keep |
| a90 | rand3 | −0.0032, −0.0004, +0.0027, −0.0029 | −0.0009 | 0.0028 | 1/4 | keep |

结论：一个同样拥有 incumbent 与替换门、按顺序试这三条随机策略的控制器，在两个作业上都会交付 Fixed。**在已存候选集合上，P 的 Fast 与随机控制器交付相同**；"阶梯第二级"不能理解为 Agent 已经战胜公平的随机搜索流程。要区分两者需要随机搜索与 Fast 同预算、同门的真实对照，本文未做。

## 6. Slow 做了什么，以及成本拆分

两次都 EDIT `construction_guidance`，applicability const true，过校验。M2 正文要点："高 lag24/168 相关被用来把 donor 从 R 换 U，改了全部序列却只换来小而不稳、在 delayed 块上反转的 C_A 优势；加 FreqMask 让 delayed 更差；偏好：保持接近 incumbent 的温和 mixup，不要把周期性统计解读成需要相位对齐 donor 或谱 dropout；检查 delayed 块的 seed 符号。"

送达后 Fast 的行为变化（M2 a90，H1 vs H0）：改动面 32→8，第二轮自行 KEEP（原话引用了 r1 逐 seed 结果与 delayed 反转）。方向与 H0 分支实测一致（大改为负）。

**成本拆分（M2 a90，H0 vs H1，同交付 Fixed）**：

| 口径 | H0（parent） | H1（candidate） | 变化 |
|---|---|---|---|
| 候选边际拟合（轮数 × K） | 8 | 4 | −50% |
| 独立运行拟合（含 None/Fixed 公共基线 8） | 16 | 12 | −25% |
| Fast LLM | 2 请求，7.7k 入 / 6.9k 出 | 2 请求，8.0k 入 / 4.3k 出 | 请求同，出 token −37% |
| Source 形成（Slow，一次性，分摊给 H1） | — | 1 请求，1.8k 入 / 2.1k 出 | H1 额外 |
| M1 同口径（a65） | 6 候选拟合，2 请求 | 3 候选拟合 + 1 失败，1 请求 | 失败导致不可比 |

限定：只有一个作业的样本；"总是保留 Fixed"的控制器成本更低（0 候选拟合、0 LLM）且交付相同。**暂不能据此证明 Slow 学会了有效的预算分配**；它是值得在 W 线 SOURCE-PROCESS 中继续观察的线索。

## 7. 分辨率与锚点

- electricity（K=3/4）：δ≈0.004，单材料 seed 极差 0.02–0.05。w_headroom 证明残差平均的算术成立（P3 0.505–0.524 vs 理论 0.50），训练价值 +0.003 量级不可辨。
- traffic 锚点扫描（seed 18–21）给出 a40/a90 三块 4/0、3–7 SE；换 seed 22–25 后只在 a90 C_A（+0.0044，4/0，SD 0.0017）和 a40 C_B（+0.0075，4/0）复现。**四个 seed 选段会高估稳定性。**
- a90 C_A 的 SE 0.0009 < δ 0.0028：该块分辨率够，候选仍未优于 Fixed。分辨率不是唯一解释，但也不排除更宽搜索能找到（§5 覆盖）。

## 8. P 线作为固定流程基线：与 W 线未来比较所需的配置差异（暂不启动）

P 线保留为"成本较低的固定流程基线"。W 线已在运行多包，本文不再讨论是否开启；下一轮真实实验按 W 的 SOURCE-PROCESS 任务书。若日后做 P/W 同条件比较，需要对齐或明确记录以下差异：

| 项 | P 线现状 | 比较时需要 |
|---|---|---|
| 作业与 seed | traffic a40→a90，seed 20260922–25，K=4（M1：electricity a55→a65，K=3） | 同作业、同 seed、同 K；两线共享 `none`/`fixed` 格（可按材料别名从 `_scratch/p_line_m2_traffic/a*/cells/` 复用，或各自重训后核对逐位相同） |
| 候选预算 | 每分支固定 2 轮，每轮 1 候选 × K 拟合 | 同每分支候选上限（如 ≤2 新候选）与拟合上限；W 的补查调用不计入候选数但计入 LLM |
| 提交者 | 程序门（mean d > 1%×incumbent C_A 且 ≥n−1 seed 同号） | W 由 Fast 依合法 C_A 证据 commit；比较时对 W 的 commit 另做一次 P 门重放，报告两者是否一致 |
| 观察 | 固定观察表（20 字段 + 批摘要），无补查 | W 可 inspect_data/inspect_material；记录 W 实际用到的字段与调用次数 |
| Slow 可编辑面 | construction_guidance、experiment_guidance | W 四面；比较时按面记录 EDIT 位置 |
| 反馈 | C_A 配对 + 实体贡献诊断（非因果）；C_B 事后附加 | 同一 `feedback.paired` 输出 |
| LLM | 7 请求 / ~45k token 每两作业 | W 上限按其任务书；报告按分支拆分 |
| 判词 | 三列 + M1 PARTIAL 标注 | 同三列；PARTIAL 不并入 |

## 9. 未做与已知缺口

- M1 a65/H1 r2 的 401 未回补。
- a95 作为复用作业只扫了 none/fixed（C_A +0.0148 4/0，C_B +0.0021 2/2，E +0.0031 3/1），未在任何控制器运行中使用。
- 未跑任何 held-out；全部 EXPOSED_DEVELOPMENT。
- experiment_guidance 面从未被 Slow 选中，其效果未测。
- 随机搜索控制器（同预算、同门）的真实对照未做；§5b 只是重放。
- 本收尾未追加锚点扫描、改卡、换 Consumer 或扩动作空间。
