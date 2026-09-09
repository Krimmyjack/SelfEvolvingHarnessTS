# R4D · 预算申请：per-channel Stage A 重打分 + 三格作用分解（主线 Fable，2026-09-07；预注册，未批不跑）

地位：呈用户 / sol 批准的**有界、0 LLM** 开发实验申请。它是 M 协议 M5 Stage A 的具体化，不是 HEC-2 ①
全课程重跑（那份草案 `HEC2_PERCHANNEL_PREREGISTRATION_DRAFT_2026-09-03.md` 保持不动，本包是它的前置）。

## 1. 为什么现在需要它

R4A / R4A-b / astra 补算三者合起来给出一个明确的机制事实：在 pooled Ridge 下，**delayed 面 64% 的实例
服务窗口未被程序修改，却承载了 77% 的严重伤害**——效果主要来自"训练语料被准备后训练出另一个模型"这条
共享通道，而不是"这条序列自己的 context 被改了"。这直接解释了为什么所有描述服务窗口的序列级特征都弱
（R4A 全 WEAK），也说明 pooled 下"逐序列 Scope"实际等于"给这条序列换模型"。

两个尚未回答、且决定后续所有 Observation / Scope 设计的问题：

1. **共享模型通道与自身 context 通道各占多少？**（三格作用分解）
2. **拆掉共享通道后，逐序列效应是否变得持久、可由 origin 前可见量识别？**（per-channel 重打分）

两者都需要拟合模型，无法从只存损失的预测库得到；合并在一次运行里采集最省。

## 2. 唯一变量与不变项

- 数据、位置（0–17、23–25）、两面（origin / origin+48）、每面 20 uid、context 192、horizon 48、Ridge 超参、
  训练 anchors、程序实现、评价 metric：**全部与 `_scratch/m_r0k_prediction_store.json` 的生成运行一致**。
- 程序：`ANCESTOR`（outlier_mad）与 `W2_pmc_then_outlier_mad`（必做）；`W1_hampel_filter`（可选，加预算）。
- 变量 A（三格分解，pooled）：对每**块**重拟合一次 pooled raw 模型与 pooled program 模型（R4C 证实块内 42 个面共用同一
  训练语料与模型，见 §3 更正），再对全部 840 个 (context, horizon) 输出三格预测：
  `raw模型/raw context`、`program模型/raw context`、`program模型/program context`；逐序列损失落盘。
  分解：`route_i = L(prog模型/raw ctx) − L(raw/raw)`，`ctx_i = L(prog/prog) − L(prog/raw)`，`total_i = route_i + ctx_i`
  （与 D5 口径一致；D5 只有 11 窗，本包扩到全部 42 面）。
- 变量 B（per-channel）：每条 served 序列用自身训练窗各拟合 raw 与 program 模型；程序只作用于该序列自身的
  训练窗与 serving context；逐序列损失落盘。**此处只换 Consumer 结构，不换任何别的东西。**
- 0 LLM；不读 held-out（`[80:120]` × ≥4056）；不写 Skill/Episode/Store；不改风险线、协议、词表。

## 3. 预算（名义值；缓存节省另列不抵扣）

**R4C 追加更正（2026-09-08）**：R4C 用数据坐实了 AGENTS §5.1 的几何事实——anchors 冻结为 [312…852]、库内最小 origin 1176，
故**同一块内全部 42 个面共用同一训练语料与同一模型**。变量 A 因此只需每块每程序拟合一次，其余全是预测。

**第二次更正（2026-09-08，采认 astra）**：上表 B 行"训练窗随 origin 变化"与本文 §2"训练 anchors 不变"自相矛盾。
固定 anchors [312…852] 下，每条 served 序列**自己的** 10 个训练窗 `raw_i[anchor−192 : anchor+48]` 同样不随 origin 变
（所有 origin ≥ 1176 使全部 anchor 通过），故 per-channel 每条序列只需 3 个模型，之后复用模型预测其全部面。
运行前须逐序列断言训练窗集跨 origin 全等。

| 项 | 计算 | fits |
| --- | --- | --- |
| A 三格分解（pooled，**已完成**，累计 24 含一次落盘精度重跑） | 4 块 × [1 raw + 2 program] | 12（实际累计 24） |
| **B per-channel，ANCESTOR + W2** | **80 条 served 序列 × [1 raw + 2 program]**，每次 `_serve` 一次拟合、批量预测该序列全部面的 raw ctx 与 P ctx | **240** |
| B 可选加 W1 | 80 × 1 | +80 |
| **B 申请上限** | 必做 240；含 W1 320；留 20 次给不可拟合序列的诊断重试 | **≤ 260（不含 W1）** |

逻辑评价次数（每序列每面每格一次预测）另行记录，不计入 fits。

墙钟按 pooled 单 fit 经验值估计 ≤ 2 小时；预算在花之前按阶段拒绝，触顶即停并如实记 `stopped_at`。

## 4. 预注册读数与判词（现在冻结）

**4.1 三格分解（pooled）**
- 每面 route 与 ctx 的代数份额（Σ|route| / (Σ|route|+Σ|ctx|)）；严重伤害实例中 route 为最负分量的比例。
- 判词：`ROUTE_DOMINANT`（份额 ≥ 0.6 且严重伤害中 route 最负 ≥ 0.6）/ `CTX_DOMINANT`（反向）/ `MIXED`。
- 与 D5 的 `ROUTE_DOMINANT`（11 窗，73%，8/10）并列，D5 不覆盖。

**4.2 per-channel 持久性（与 R4A-b 同口径、同脚本逻辑复跑）**
- 同 (position, uid) 两面增益的 Spearman 中位数、按 cohort 的 uid 方差份额与噪声期望、位置间份额。
- 判词沿用 R4A-b 冻结词表：`SERIES_LEVEL_CONDITION` / `COHORT_LEVEL_CONDITION` / `MOSTLY_NOISE` / `MIXED`，
  并与 pooled 的读数并排。**预注册预测**：uid 份额上升、Spearman 中位数上升（方向预测，不预测幅度）。

**4.3 per-channel 可识别性（与 R4A 同特征集、同 LODO、同阈值复跑）**
- 12 词表数值量 + R4A 机制量对 helped / severe_harm 的四折 AUC；判词沿用 `PATTERN_INFORMATIVE / WEAK / UNINFORMATIVE`。
- **预注册预测**：至少一格由 WEAK 升为 INFORMATIVE（方向预测）。若仍全 WEAK，则"观察对象不匹配"不是
  唯一原因，噪声在评价侧（48 步 sMASE）的嫌疑上升。

**4.4 伤害形状（沿用 HEC-2 ① 草案 P-C1 / P-C2 / P-C6 的口径，但只在本 42 面上读）**
- msh 失败占比是否下降、hf 失败占比是否上升、聚合增益是否下降；判词 `CONSUMER_SHAPES_HARM / NO_SHAPE_CHANGE`
  附 `GAIN_RETAINED / GAIN_LOST`。

## 4.5 追加读数（2026-09-08，采认 astra；B 与 R4E 共用）

- **按通道分开的可识别性**：R4A 的 21 个可见量（12 词表数值量 + 9 机制量）与 R4C 的服务侧作用条件，分别对 `route`、`ctx`、`total`
  三个通道的 helped / severe 做四块 LODO AUC 与冻结分箱 stump 的全人群效用（分母全服务人群）。pooled 侧用 R4D-A 的三格库
  （R4E，0 fit）；per-channel 侧用 B 的三格库。
- **"看 Pattern 选处理"对"始终选同一个处理"**：对每个 (program, Consumer)，规则 = 按某可见量在训练折选阈值决定"用 P 还是 identity"，
  测试折读全人群效用，对照 always-P / always-identity / oracle；这是最终裁决口径，AUC 只作辅助。
- **Consumer 基线质量**：per-channel Static 逐序列损失相对 pooled Static 的差（均值、中位、受损比例）。若 per-channel Static 明显更差
  （10 个训练窗 × 192 维的 Ridge 欠定），"该用哪种 Consumer"不在本包结论范围内，但"序列状态是否调节对自身数据准备的响应"仍可读。

## 5. 预写分流（2026-09-08 修订；旧表撤回）

| 结果 | 含义 | 下一步 |
| --- | --- | --- |
| 4.1 `ROUTE_DOMINANT`（已得）∧ 4.5 某通道下某可见量 LODO 每折 ≥ 0.70 且"看 Pattern 选处理"胜过两个固定选择 ≥ 3/4 折 | 存在被总效应掩盖的局部规律 | 该 (通道, 可见量) 进入 Observation 候选；下一包做 LLM 输入消融 |
| B 下 4.2 持久性 / 4.3 可识别性上升，pooled 下无 | 序列状态对**自身数据准备**的响应可读，对**他人数据准备引起的模型变化**的响应不可读 | Scope / Observation 线只在 per-channel 类 Consumer 下重启；HEC-2 ① 草案填锚点 |
| 两种 Consumer 下都无改善 | **这组观察量在这个设置下未表现出足够信息**——不判定噪声、不关闭序列级研究 | 观察对象转向"训练数据 + 程序造成的变化 + 服务窗"三者并描（astra），或搁置该线；写入论文为观察缺口 |
| 4.1 `CTX_DOMINANT`（未出现） | — | — |
| 4.4 `GAIN_LOST` 显著 | per-channel 保住安全但丢掉收益 | 论文诚实报告 Consumer 结构的收益–风险权衡 |

**撤回的表述**："4.2/4.3 无改善 ⇒ 噪声在评价侧、序列级 Scope 关闭"。收益随时间不稳定不等于当时的状态无法预测收益。

## 6. 交付与纪律

一个 runner（复用现有 pooled / per-channel evaluator，禁止新框架）、一个主报告、一个 smoke；工件不覆盖；
逐序列损失落盘为新的预测库（不写入 `_scratch/m_r0k_prediction_store.json`）；结果类别 `MECHANISM /
INSTRUMENT`，不是能力证据；非作者复核（Grok）后才入状态页。**未获用户批准前不运行任何 fit。**
