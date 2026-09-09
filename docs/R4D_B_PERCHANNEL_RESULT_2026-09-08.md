# R4D-B · per-channel Consumer 下的三格重打分（结果，2026-09-08 凌晨；主线 Fable 亲自执行）

证据类别：**MECHANISM / INSTRUMENT**（development 机制读数，不是能力、泛化或显著性证据）。
读数口径与判词由 `docs/R4D_PERCHANNEL_STAGE_A_AND_ACTION_DECOMPOSITION_REQUEST_2026-09-07.md` §4.1–4.5、§5（修订版）冻结；
伤害形状的"变化"阈值 `SHAPE_DELTA = 0.05` 由本包脚本在运行前写定（申请文档只给了定性口径），特此披露。

- 脚本：`evaluation/main_protocol_p4/run_r4d_b_perchannel_three_cell.py`
  （`python -m evaluation.main_protocol_p4.run_r4d_b_perchannel_three_cell` 可复跑；**240 physical fits**，硬顶 260，约 100 秒）
- 工件：`artifacts/main_protocol/r4d_b_perchannel_three_cell.json` / `.md`
- 新预测库：`_scratch/r4d_b_perchannel_three_cell_store.json`（与 A 库同键同结构；A 库与 m_r0k 库未写）
- 执行说明：派给执行线的会话在探索阶段停滞（第二次），主线复用 R4D-A 与 R4E 的函数自行实现并运行；
  `run_block` 的唯一改动是把"每块一套模型"换成"每条被服务序列一套模型"。

## 0. 先报基线质量（§4.5 第三点，按申请要求置于首段）

| 面 | n | per-channel Static 均值 | pooled Static 均值 | Δ 均值 | Δ 中位 | pc 更差占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| support | 420 | 0.992580 | 1.462313 | −0.469733 | −0.255677 | 0.280952 |
| delayed | 420 | 1.246216 | 1.527425 | −0.281209 | −0.184633 | 0.380952 |
| 两面 | 840 | 1.119398 | 1.494869 | −0.375471 | −0.223985 | 0.330952 |

**与主线预期相反**：每条序列用自己 10 个训练窗（192 维、10 样本）拟合的 Ridge，作为预测器**明显优于**用块内另外 20 条序列
200 个窗拟合的 pooled Ridge——损失均值低 25%，三分之二的序列更好。|Δ 中位| = 0.224 > 0.10，按冻结口径
`consumer_choice_in_scope = False`，但方向是 pc 更好而不是"欠定到不能用"：本包的"该用哪种 Consumer"结论仍不在范围内
（两套系统的基线水平不同，不是同一把尺），"序列状态是否调节对自身数据准备的响应"照常可读。

## 1. 一句话答案

> **换成 per-channel 之后：(i) 数据准备的收益缩到原来的 1/3–1/10，严重伤害几乎消失；(ii) 效应仍然七成以上来自训练侧
> （route），只是 route 的含义变成了"自己的训练数据被准备后自己的模型变了"；(iii) 逐序列效应仍不持久（跨面 Spearman
> 0.07 / 0.14）；(iv) 序列级可见量仍不能让"看 Pattern 选处理"胜过固定选择（6/6 `FIXED_CHOICE_NOT_BEATEN`）；
> (v) pc 侧出现的几格 `PATTERN_INFORMATIVE` 全部没有全折支撑——严重伤害少到无法识别，本身就是结果。**

## 2. 边界自检

| 项 | 值 |
| --- | --- |
| 物理 Ridge 拟合 | **240 / 260**（80 序列 × [raw, ANCESTOR, W2]，每次 `_serve` +1，花之前断言 3×80 ≤ 260） |
| 不可拟合序列 / 不可评价实例 | **0 / 0**（80 条序列训练 context 均未触 scale floor） |
| LLM / held-out 读 / 新增 SHA / git 提交 / 子 Agent | 0 / 0 / 0 / 0 / 0 |
| 读到的最大时间下标 | 3911（= 3864 + 48 − 1；特征卡 3863） |
| 数据身份 / NaN | `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`，NaN = 503712 |
| 训练窗跨 origin 全等 | 80/80 序列断言通过（每条 10 个窗，anchors 10/10 保留） |
| 既有文件编辑 | 0（状态页由主线另行更新） |

## 3. 五项读数

### 3.1 三格分解：pc 仍然 `ROUTE_DOMINANT`（六格全部）

| Consumer | program | 面 | route 份额 | 严重伤害 n | 严重中 route 最大 | ρ(route,total) | ρ(ctx,total) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| **pc** | ANCESTOR | support / delayed | 0.727 / 0.791 | **3 / 9** | 0.667 / 1.000 | 0.755 / 0.879 | 0.434 / 0.235 |
| **pc** | W2 | support / delayed | 0.733 / 0.733 | **8 / 6** | 0.875 / 0.833 | 0.880 / 0.832 | 0.311 / 0.413 |
| pooled | ANCESTOR | support / delayed | 0.768 / 0.804 | 28 / 31 | 0.964 / 0.968 | 0.902 / 0.937 | 0.214 / 0.245 |
| pooled | W2 | support / delayed | 0.714 / 0.703 | 60 / 75 | 0.650 / 0.787 | 0.868 / 0.847 | 0.292 / 0.256 |

份额几乎不变（0.73–0.79）。**这说明 A 里"共享"不是关键，"训练侧"才是**：无论模型是共享的还是自己的，程序对训练数据
的作用经由模型变化传到预测的那一份，都占七成以上；服务窗自身的准备只占两三成。ρ(ctx,total) 在 pc 下略升（0.23–0.43）。

### 3.2 持久性：仍然接近零

| Consumer | program | 跨面 Spearman 中位 | uid 份额（噪声期望 0.183） | 位置份额 | 判词 |
| --- | --- | ---: | ---: | ---: | --- |
| pc | ANCESTOR | 0.073 | 0.248 / 0.231 | 0.105 / 0.112 | `MOSTLY_NOISE` |
| pc | W2 | 0.144 | **0.317 / 0.308** | 0.138 / 0.119 | `MIXED`（uid 份额过 0.30 线，但跨面相关远低于 0.50） |
| pooled | ANCESTOR | 0.035 | 0.202 / 0.215 | 0.103 / 0.105 | `MOSTLY_NOISE` |
| pooled | W2 | 0.143 | 0.200 / 0.213 | 0.104 / 0.132 | `MOSTLY_NOISE` |

预注册预测"uid 份额与跨面 Spearman 上升"：uid 份额在 W2 上升到 0.31–0.32（方向对，越过 0.30），跨面 Spearman 基本不动。
同一条序列 48 步之后是否受益，在 pc 下依然与现在几乎无关。

### 3.3 可识别性：pc 侧的 INFORMATIVE 全部没有全折支撑

pc 侧有 4 格判 `PATTERN_INFORMATIVE`（ANCESTOR route support `srv_donor_dispersion|severe` 0.838；W2 route delayed
`gap_tail_fraction|severe` 0.816；W2 ctx both `missing_fraction|severe` 0.937；W2 total delayed `gap_longest_run_steps|severe` 0.857），
但**每一格的 `full_support_informative` 都是空列表**：它们的目标是严重伤害，而 pc 下严重伤害只剩 ANCESTOR 12 例、W2 14 例
（pooled 是 59 / 135），四折里必有无正例的折被冻结规则跳过（同 R4E §3.1 的 ANCESTOR ctx 情形）。**这些格不得作为
"pc 下可识别"的证据。** 以 `helped` 为目标的最佳读数在 pc 下全部 WEAK/UNINFORMATIVE（0.58–0.68）。
预注册预测"至少一格由 WEAK 转 INFORMATIVE（全折支撑）"：**未实现**。

### 3.4 "看 Pattern 选处理 vs 始终选同一处理"：6/6 `FIXED_CHOICE_NOT_BEATEN`

pc 下 always-P 的效用只有 +0.02–0.12（pooled 是 +0.23–0.27），identity 因此更接近；最好的规则在 W2 delayed 面上 3/4 折
同时胜过两个固定选择，但相对 always-P 的四折均值只有 +0.000095，远低于 material 线 0.005；其余格 0–2 折。
pooled 侧六格同样 `FIXED_CHOICE_NOT_BEATEN`（与 R4E 一致）。

### 3.5 伤害形状：`CONSUMER_SHAPES_HARM` + `GAIN_LOST`（四格全部）

| program \| 面 | 聚合增益 pc / pooled | 受损占比 pc / pooled | 最坏单序列 pc / pooled | 严重伤害 n pc / pooled |
| --- | ---: | ---: | ---: | ---: |
| ANCESTOR \| support | 0.072 / 0.245 | 0.269 / 0.307 | 0.366 / 1.181 | 3 / 28 |
| ANCESTOR \| delayed | 0.018 / 0.225 | 0.331 / 0.317 | 0.770 / 2.022 | 9 / 31 |
| W2 \| support | 0.117 / 0.272 | 0.360 / 0.360 | 1.186 / 1.826 | 8 / 60 |
| W2 \| delayed | 0.102 / 0.185 | 0.364 / 0.357 | 0.610 / 1.849 | 6 / 75 |

HEC-2 ① 草案的 P-C1（尾部左移）与 P-C6（收益下降）**同时兑现**：最坏单序列伤害降到原来的 1/3–1/2，严重伤害例数降到
1/5–1/10；聚合增益降到 1/3–1/10。受损占比基本不变（P-C2"轻度受害上升"未见）。

## 4. 正典 §10 五问

**Harness 行为改变了什么。** 没有。0 LLM、0 held-out、不编辑既有文件、不写 Skill/Episode、不改协议或阈值。

**数据上观察到了什么。** (i) **Consumer 结构本身是这份数据上最大的效应来源**：per-channel Static 比 pooled Static 好 25%，
而且几乎不需要数据准备就已经拿到了 pooled 下"准备"所买到的大部分收益——pooled 下的 +0.23–0.27 程序增益，很大一部分是在
修一个不匹配的共享模型；(ii) 严重伤害（尾部）在 pc 下基本消失，这直接击中权威门四线里 `max_single_series_harm ≤ 0.30`
这条绑定约束的来源——它主要是 pooled Consumer 的性质；(iii) 无论共享还是自己的模型，训练侧准备都是主通道
（route 0.73–0.79）；(iv) 序列级可见量在两种 Consumer 下都不能做出胜过固定选择的部署决定。

**当前最大方法不确定性。** pc 下几乎没有严重伤害可识别，所以"序列状态能否预判严重伤害"在 pc 下变成了低基率问题，
四块 LODO 分辨不出（正例 3–14）；"序列状态能否预判受益"在两种 Consumer 下都弱。pc Static 更好是这四个 KDD 块、
Ridge、192/48 几何下的读数，不是通用结论；两种 Consumer 的基线水平不同，本包不裁决"该用哪种 Consumer"。

**是否仍与目标一致。** 一致，但方向要再校正一次：项目一直在 pooled Ridge 上研究"数据准备的条件化"，而本包显示
pooled 的收益与伤害形状有很大一部分是 Consumer 不匹配的产物；在更匹配的 Consumer 下，数据准备的空间小得多、风险也小得多。
按 §5 修订分流：pc 下 4.2 部分改善（uid 份额）、4.3 无全折支撑改善 → 落第二行的"否"侧与第三行之间——
**这组序列级观察量在两种 Consumer 下都未表现出足以改变部署决定的信息；不判定噪声。** 同时 4.4 `GAIN_LOST` 显著 →
论文须诚实报告 Consumer 结构的收益–风险权衡。

**下一项最小纵向切片。** 不再在这组 27 个可见量上加实验。两条可选，都不急于本周：
(a) 把"Consumer 结构决定数据准备的价值与风险"写成论文的一个主结果（pooled vs per-channel 的 Static / 增益 / 尾部三张表
已在手，0 新 fits）；(b) 若要继续序列级条件化，观察对象必须转到训练侧作用（astra：训练数据 + 程序造成的模型变化 + 服务窗
三者并描），并在一个严重伤害基率足够的设置下做——pc 不是那个设置。

## 5. 这份结果不是什么

- 不是能力、泛化或显著性证据：全部 development，held-out 一个都没读；80 uid × 21 位置重复观测，4 个块。
- 不是"per-channel 更好"的通用结论：这是 4 个 KDD 块 × Ridge × 192/48 几何下的读数；两种 Consumer 基线水平不同，
  "该用哪种 Consumer"按冻结口径不在范围内。
- 不是"pc 下序列级条件可识别"的证据：pc 侧 4 格 INFORMATIVE 全无全折支撑，靠 3–14 个正例撑起。
- 不是对 pooled 结论的推翻：pooled 六格与 R4D-A、R4E 数字逐位一致（同一库）。
- 不是三格库的对账：pc 库的 `store_*` 字段存的是同实例的 pooled 值，用于配对，不是复现目标。
- 未涉及 W1 / W3 与两个参考程序；未改任何冻结协议、roster、split、阈值、预算或词表；未新增 SHA；未提交 git。

## 6. 非作者复核备注（Grok，`artifacts/main_protocol/r4d_b_nonauthor_check.md`，8/8 一致）

- 新库以 9 位小数落盘，358/1680 行由四舍五入后的 `L_*` 重算通道会与库内 `route/ctx/total` 差约 1e-9（无一超过 1.5e-9）；
  读数一律以库内通道字段为准，六格分解重判 6/6 `ROUTE_DOMINANT` 无变化。
- `ANCESTOR|17|delayed_face|T189` 库内 `total = 2.22e-16`（浮点末位），按 `total > 0` 计为受损会使 delayed 面受损占比多 1/420；不影响判词。
- pc 侧 4 格 `PATTERN_INFORMATIVE` 对应通道的严重正例数分别为 3 / 5 / 3 / 6，坐实 §3.3"无全折支撑、不作证据"。
