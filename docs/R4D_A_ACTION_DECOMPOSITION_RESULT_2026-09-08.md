# R4D-A · 三格作用分解：pooled 下"共享模型通道"与"自身 context 通道"各占多少（结果，2026-09-08）

证据类别：**MECHANISM / INSTRUMENT**（development 机制读数，不是能力或泛化证据，不做显著性主张）。
读数与判词由 `docs/R4D_PERCHANNEL_STAGE_A_AND_ACTION_DECOMPOSITION_REQUEST_2026-09-07.md` §4.1 在任何数字之前
冻结，未调整。本包只做**变量 A**（三格分解），变量 B（per-channel 重打分）未跑。

- 脚本：`evaluation/main_protocol_p4/run_r4d_a_action_decomposition.py`
  （`python -m evaluation.main_protocol_p4.run_r4d_a_action_decomposition` 可复跑，单次 12 physical fits，约 45 秒）
- 工件：`artifacts/main_protocol/r4d_a_action_decomposition.json` / `.md`
- 逐序列三格损失：`_scratch/r4d_a_three_cell_store.json`（新库；`_scratch/m_r0k_prediction_store.json` 未写）

## 0. 一句话答案

> **共享模型通道占绝对多数。** 六个 (program, face) 格全部 `ROUTE_DOMINANT`：route 的代数份额
> 0.70–0.80，严重伤害实例中 route 为最负分量的比例 0.65–0.97。并且 `L_rr` 与 `L_pp` 对预测库
> **1680/1680 全部复现**（最大 |Δ| = 1.3e-15），所以这不是一个新口径下的近似读数，而是把库里已有的
> `raw_per_view` / `program_per_view` 从中间劈开。与 D5（11 窗，0.73，8/10，`ROUTE_DOMINANT`）同向，
> 且把它从 11 窗扩到全部 42 面；D5 不覆盖本读数。

## 1. 三格与符号

对每个程序 P ∈ {ANCESTOR = `outlier_mad`，W2 = `period_median_complete → outlier_mad`}，
对每个 served 实例 i（42 面 × 20 uid = 840），在同一真值上取三个损失：

| 格 | 模型 | 服务 context | 等于库里的 |
| --- | --- | --- | --- |
| `L_rr` | raw（未准备的训练语料） | `_linear_integrity(raw[origin−192:origin])` | `raw_per_view`（= Static） |
| `L_pr` | P（准备过的训练语料） | 同上，**未被 P 触碰** | 库里没有，本包新算 |
| `L_pp` | P | `_apply_program(raw[origin−192:origin], compiled_P)` | `program_per_view` |

`route_i = L_pr − L_rr`（只有共享模型换了）、`ctx_i = L_pp − L_pr`（只有这条序列自己的 context 换了）、
`total_i = L_pp − L_rr = route_i + ctx_i`。**符号已核对**：库的 `g = raw − program`，故 `total = −g`；
损失口径下 `total > 0` 是伤害，`total > 0.30` 是严重伤害，对应库口径 `g < −0.30`。
"route 为最负分量"（增益口径）在损失口径就是 `route_i > ctx_i`。

## 2. 边界自检

| 项 | 值 |
| --- | --- |
| 物理 Ridge 拟合（本次运行 / 硬顶 / 目标） | **12 / 24 / 12** |
| 物理拟合（本包累计，含一次口径修正重跑） | **24 / 24**（每次运行 12；见 §6 说明） |
| 拟合计数规则 | 每次 `_serve` 调用 +1，花之前先断言 `3 × 块数 ≤ 24` |
| LLM 调用 | 0 |
| held-out 读 | 0（禁区前沿 4056） |
| 读到的最大时间下标 | **3911**（= max origin 3864 + 48 − 1） |
| 原始窗读取次数 | 1640（4 块 × 200 训练窗 + 840 服务实例） |
| 既有文件编辑 / git 提交 / 新增 SHA / 子 Agent | 0 / 0 / 0 / 0 |
| 预测库写入 | 只写新库 `_scratch/r4d_a_three_cell_store.json`；m_r0k 库未动 |
| 数据身份 / NaN | `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`，NaN = 503712 |
| NOT_EVALUABLE 实例（scale floor） | 0 / 1680 |

## 3. 为什么 12 次拟合够（几何 + 批量的两个理由）

**(a) 每块一套模型。** R4C §4 用数据坐实：anchors 冻结为 [312…852]、保留条件 `anchor + 48 ≤ origin`、
库内最小 origin 1176，故每块的训练语料不随 origin 变。本包对每块**逐值重核**了这一点而不是引用：
每块 200 个训练窗（20 条 train 序列 × 10 个 anchor，10/10 全部通过保留条件），块内所有 origin 得到的窗集
`np.array_equal(..., equal_nan=True)` 全等、train 序列名单全等（[0:40] 14 个 origin、[40:80] 18、[80:120] 4、
[120:160] 6）。任一条不成立即 raise，不降级继续。

**(b) 一次拟合可以批量预测两组 context。** `_serve` 先拟合一次再对传入的全部 context 预测，而
`_exact_weighted_ridge_prediction` 的 `np.linalg.solve` 只读 `x_train / targets / weights`，**从不读 `x_eval`**
（`run_e2_cross_series_curation.py:2826-2831`）。所以把 raw context 与 P-prepared context 放进**同一次**
`_serve` 调用，`L_pr` 与 `L_pp` 出自同一个物理拟合。于是每块 3 次：raw、ANCESTOR、W2 → 4 × 3 = **12**。

程序编译走生产同一条路：`ScopeExecutor(roster, at.values, config, evaluate_fn=representation_view.
forecast_runtime._evaluate, max_modified_fraction=run_forecast_p4_performance.MAX_MODIFIED_FRACTION)._compiled(steps)`
——`run_hec1._executor`（432–437 行）与 `ReplayPredictionCache._build`（612 行）产库时用的就是它。
服务 context 的处理与 `scoped_evaluate`（169–186 行）**逐行一致**：程序只作用于 192 步窗
`raw[origin−192:origin]` 本身，不是"长窗准备后截取"。metric 沿用 `smase` + `seasonal_scale(raw[:origin],
period=24, min_pairs=32)` 与 `isfinite(truth)` 掩码，未自造。

## 4. 对账（决定结果可信度）

| 程序 | n | `L_rr` 复现（\|Δ\|<1e-9） | 逐位全等 | max \|Δ\| | `L_pp` 复现 | 逐位全等 | max \|Δ\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ANCESTOR | 840 | **840 (100%)** | 788 | 1.3e-15 | **840 (100%)** | 788 | 8.9e-16 |
| W2 | 840 | **840 (100%)** | 788 | 1.3e-15 | **840 (100%)** | 791 | 8.9e-16 |

1680 行全部在 1e-9 内复现；逐位（bit-identical）相同的是 `L_rr` 1576/1680、`L_pp` 1579/1680，
剩余 104 / 101 行的偏差量级 1e-16–1e-15，是批量矩阵乘法的浮点末位差（同一个 solve，`z_eval` 行数不同
导致 BLAS 分块不同），不是口径差。
对账通过 ⇒ route / ctx 读数**按预注册报告**，不标 `UNRELIABLE`。

## 5. 读数（§4.1，逐 (program, face)）

| program | face | n | **route 份额** | 严重伤害 n | **严重中 route 最负** | route 均值 | route 中位 | ctx 均值 | ctx 中位 | ρ(route,total) | ρ(ctx,total) | **判词** |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ANCESTOR | support | 420 | **0.768** | 28 | **0.964** (27/28) | −0.1638 | −0.0839 | −0.0817 | 0.0 | 0.902 | 0.214 | **ROUTE_DOMINANT** |
| ANCESTOR | delayed | 420 | **0.804** | 31 | **0.968** (30/31) | −0.1617 | −0.1056 | −0.0630 | 0.0 | 0.937 | 0.245 | **ROUTE_DOMINANT** |
| W2 | support | 420 | **0.714** | 60 | **0.650** (39/60) | −0.2563 | −0.1125 | −0.0160 | 0.0 | 0.868 | 0.292 | **ROUTE_DOMINANT** |
| W2 | delayed | 420 | **0.703** | 75 | **0.787** (59/75) | −0.1745 | −0.1422 | −0.0105 | 0.0 | 0.847 | 0.256 | **ROUTE_DOMINANT** |

份额 = Σ\|route\| / (Σ\|route\| + Σ\|ctx\|)。两面合并：ANCESTOR 0.785（严重 57/59 = 0.966）、W2 0.709（98/135 = 0.726）。
符号一致率：route 与 total 同号 0.94（ANCESTOR）/ 0.87（W2）；ctx 与 total 同号仅 0.27 / 0.51。
两个分量本身多为负（= 程序整体在帮忙）：`total` 均值 −0.235（ANCESTOR）/ −0.229（W2），
严重伤害率 7.0% / 16.1%。

**与 D5 并列**（D5 = 11 窗，route 份额 0.73，严重中 8/10，`ROUTE_DOMINANT`；引用不重算，**D5 不覆盖本读数**）：
本包 42 面上的份额 0.70–0.80 与 D5 的 0.73 同区间，严重伤害口径本包 0.65–0.97 覆盖 D5 的 0.80。

### 5.1 服务窗未被程序修改的实例（astra 2.1 复算）

定义：`_apply_program(context, compiled)` 与 `_linear_integrity(context)` 逐值相同，即 `_prepare` 在 192 步
服务窗上的 moved = 0。

| program | face | 未修改 / 可评价 | 占比 | 其 total 均值 | 中位 | p10 | p90 | 其中严重伤害率 | 全部严重伤害中落在未修改窗的比例 | 未修改窗上 ctx 恰为 0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ANCESTOR | support | 236 / 420 | 0.562 | −0.1563 | −0.0839 | −0.643 | +0.241 | 0.068 | 0.571 (16/28) | 236/236 |
| ANCESTOR | delayed | **269 / 420** | **0.640** | −0.1659 | −0.1065 | −0.696 | +0.281 | 0.089 | **0.774 (24/31)** | 269/269 |
| W2 | support | 75 / 420 | 0.179 | −0.0424 | −0.0463 | −0.377 | +0.533 | 0.133 | 0.167 (10/60) | 75/75 |
| W2 | delayed | 42 / 420 | 0.100 | −0.1608 | −0.1835 | −0.817 | +0.408 | 0.190 | 0.107 (8/75) | 42/42 |

**astra 2.1 的两个数字都精确复现**：ANCESTOR delayed 面 269/420（= 64.0%）服务窗未被修改，而这些实例
承载了该面严重伤害的 **77.4%**（24/31）。本包补上他当时无法给出的那一半：这 269 个实例的 `ctx` **全部恰为 0**
（269/269，`L_pp = L_pr` 逐位相等），它们的 total 均值 −0.166、p10/p90 = −0.696/+0.281——效应完全来自
route，而且并非小效应，双向都很宽。W2 因为周期填补几乎总能在窗内找到供体，未修改比例掉到 10–18%。

## 6. 与前提不符 / 需要更正的事实

1. **"块内 42 面"应读作"全部 42 面分布在 4 个块里"。** 派发口径说"每块全部 42 面 × 20 uid"，实际
   42 个面按块分为 14 / 18 / 4 / 6（[0:40] / [40:80] / [80:120] / [120:160]），共 840 个 (face, uid)。
   R4C §4 的原文说的是"同一块内其全部面共用同一训练语料"，本包按块重核后成立；每块面数不等不影响设计。
2. **本包物理拟合累计 24 次，不是 12 次。** 第一次运行（12 fits）产出的数值与最终工件**完全一致**，
   但两处**序列化**缺陷需要修正才交付：(i) 工件把对账偏差四舍五入到 9 位，1e-15 的偏差被写成 `0.0`；
   (ii) 新预测库的逐序列损失也被四舍五入到 9 位，下游无法在 1e-9 容差上对账。改的是落盘精度、不是计算，
   修改后先用 stub 替换 `_serve` 做了 **0 拟合**的干跑验证，再重跑一次（又 12 fits）。按"每次 `_serve` +1
   逐次累计"的规则如实记为 **24 / 24**，恰好触顶、无余量；单次复跑仍是 12。脚本内的 ledger 只看本次运行，
   这一点写进了工件 `boundary.physical_fits_scope`。
3. **`route` 与 `ctx` 均值为负、不是"伤害"。** 判词读的是**份额**与**严重伤害子集**，不是均值方向；
   两个程序在 42 面上整体都在帮忙（total 均值 ≈ −0.23），route 通道同时是主要收益来源和主要伤害来源。
4. 无其他与前提不符之处：预测库的 `degenerate_uids` 全空、scale floor 触发 0 次，所以 `NOT_EVALUABLE`
   单列在本次运行里是空的（代码路径仍然在位并已验证）。
5. *勘误（非作者复核 Grok，`artifacts/main_protocol/r4d_a_nonauthor_check.md`）*：§5.1 "未修改窗上 ctx **逐位**为 0"在严格
   `ctx == 0.0` 下少 1 条（ANCESTOR delayed T195，ctx = +2.22e-16，非严重伤害；另有 2 行同量级），应表述为 `|ctx| < 1e-12`。
   269/420、24/31 = 0.774、236/420、16/28 等全部数字在该口径下不变；六格读数与判词经独立重算 6/6 一致。

## 7. 正典 §10 五问

**Harness 行为改变了什么。** 没有。0 LLM、0 held-out、不编辑任何既有文件、不提交 git、不新增 SHA、
不写 Skill/Episode/Store、不改风险线/协议/词表/阈值。只新增 1 个脚本、2 个工件、1 个新预测库、本报告。

**数据上观察到了什么。** (i) pooled 下逐序列 Scope 的效应 **70–80% 来自共享模型通道**：给这条序列换成
"训练语料被准备后拟合出来的另一个模型"，而不是改这条序列自己的 context；(ii) 严重伤害更极端地集中在
route（ANCESTOR 96.6%、W2 72.6% 的严重伤害实例里 route 是最大正贡献）；(iii) route 与 total 的 Spearman
0.85–0.94，ctx 与 total 只有 0.21–0.29——total 的排序几乎就是 route 的排序；(iv) ANCESTOR delayed 面
269/420 的服务窗被程序原样放过（ctx 恰为 0），却承载 77.4% 的严重伤害，astra 2.1 的两个数字逐个复现；
(v) 三格里的两个端点对预测库 1680/1680 复现，1576 行逐位相同。

**当前最大方法不确定性。** route 通道占多数**不等于**"拆掉它就会出现序列级信号"。本包只测了分解，
没测拆掉之后（变量 B）。两种可能仍未分开：(a) 序列级条件真实存在，只是被 pooled 的共享模型淹没；
(b) 逐序列效应主要由"固定模型差 × 该 (context, horizon) 的偶然交互"产生，换成 per-channel 后
ctx 通道也同样不可由 origin 前可见量识别。另外 42 面来自 21 位置 × 80 uid 的重复观测，块只有 4 个，
份额的不确定性没有量化（本包不做显著性）。

**是否仍与目标一致。** 一致。按申请 §5 预写分流，`4.1 = ROUTE_DOMINANT` 这一行的下一步是"变量 B
per-channel 重打分"，而不是任何 Scope / Observation 线的重启——重启的前提是 4.2 / 4.3 也改善。
`CTX_DOMINANT`（会与 astra 的服务窗未修改事实冲突、需先查实现）**没有出现**，双管线按说明路由的证据
反而更强了：未修改窗上 ctx 逐位为 0，共 622 个实例（ANCESTOR 505 + W2 117），一个例外都没有。

**下一项最小纵向切片。** 变量 B（per-channel，ANCESTOR + W2，2520 fits）。它是唯一能把上面 (a)/(b)
分开的对照：每条 served 序列用自身训练窗拟合，"块内一对模型"的几何随之消失，route 通道按构造被拆掉。
本包的 `_scratch/r4d_a_three_cell_store.json` 已经把 pooled 侧的三格逐序列损失落盘，B 跑完可直接对齐同一
(position, face, uid) 键做配对读数，无需重跑 A。

## 8. 这份结果不是什么

- 不是能力、泛化或显著性证据；全部 development，held-out 一个都没读（最大下标 3911 < 4056）。
- 不是"逐序列 Scope 无用"的结论：它说的是**在 pooled Consumer 下**逐序列 Scope 的多数效应经由共享模型
  通道传递；per-channel 下同一问题尚未测量（变量 B 未跑）。
- 不是 D5 的复核，也不被 D5 覆盖：D5 的 11 窗与本包 42 面是不同人群，两者并列引用、不合并。
- 不是对 W2 相对 ANCESTOR 的边际读数（那是 R4C 的 `d`）；本包对两个程序各自独立分解，未做 W2 − ANCESTOR。
- 不是"服务侧准备无效"的结论：ctx 份额 0.20–0.30 不是 0，W2 的 ctx 绝对量（Σ\|ctx\| = 157）也不小；
  它只是被 route 压过，且与 total 的秩相关很弱。
- 不涉及 `W1_hampel_filter`（申请里的可选项，未加预算、未跑）。
- 不改变任何已冻结的协议、roster、split、阈值、预算或词表；未新增 SHA、目录或平台层；未提交 git。
