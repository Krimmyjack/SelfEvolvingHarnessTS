# DEV-TRAIN-3：跨作业知识复用与 Target 适应——结果报告

完成：2026-09-11 18:56（本机时间），课程 `course_beijing_j1_j2_20260911` 状态 `COMPLETED`，exit 0。
执行者：本会话（Opus）。任务书：[DEV_TRAIN3_CROSS_JOB_HARNESS_TASK_2026-09-11.md](DEV_TRAIN3_CROSS_JOB_HARNESS_TASK_2026-09-11.md)。
本页为执行者自报，同上下文复算，结论暂定，不是独立复审。所有数值来自 `_scratch/dev_train3_cross_job/course_beijing_j1_j2_20260911/result.json`，紧凑工件 `artifacts/main_protocol/dev_train3_cross_job.json`。

## 0. 三个直接回答

| 问题 | 回答 |
| --- | --- |
| **有没有净准备正结果？** | **有，但不来自知识更新。** E 上两作业等权：Source-fixed（`winsorize>fft_decompose`）收益 +0.723951，固定 H 臂 F +0.800619，两者都由 J2 主导（J2 E@3480 Static sMASE 7.30，任何压缩训练材料尖峰的方案都大幅得益；J1 上两者分别为 −0.030672 / +0.047525）。A3 等权 −0.342318（J1 +0.036277、J2 −0.720914），A5 两作业 E 输出与 Static **逐点相同**（收益 0，90/90 决策 identity）。 |
| **A5 是否比 A3 更好或更省？** | **否。** J1：A5−A3 = +0.036277（A5 更差）；J2：−0.720914（A5 更好，但只是因为 A3 采用的分支在 E 上有害而 A5 什么都没做）。两作业不同号，等权 −0.342 不满足“两作业均不负且有净准备收益”。效率线索：A5 的 q2 检查点减 A3 最终质量 J1 +0.278721、J2 −0.720914，不构成“更少反馈达到相近质量”。A5 的历史知识（Source 卡 + J1 卡）在 J2 表现为完全的 identity 门。 |
| **新旧 H 的变化是否比固定 H 搜索更有价值？** | **否。** 三处真实 Skill 修改中：J1-A3 新卡在 C_B 被否决、E 上更差（新−旧 +0.197313）；J1-A5 新卡在 C_B 被采用（1.447 vs 1.528）但 E 上比旧 H 分支差 0.175897；J2-A3 新卡 C_B 被否决、E 上略好于旧 H（−0.288737）但两者都低于 Static。固定 H 臂 F 的两条同 H 分支在 J1 E 上相差 0.130944（C_B 上 0.20），与知识效应同量级；F 的最终 E 收益是所有 LLM 臂中最高的。 |

按任务书 §8.3：**L2 未成立**（有修改，后续新 H 无增益）；**L3 线索未成立**；F/Source-fixed 的正号属于“当前作业与动作组合存在准备空间，且主要在 J2 一个块上”，不是历史积累或调查机制的证据。

## 1. 实际运行了什么

### 1.1 数据与作业（冻结于 `course.json`）

- 六站按名固定：Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng，进入 Fast/Slow 时匿名为 `s01…s06`（站名、数据集名不进入任何 Prompt）。
- 读取端只解析每站 CSV 前 14328 行（2013-03-01 00:00 → 2014-10-18 23:00），时间列连续性通过；2016+ 数值未解析。缺失率 1.0%–2.3%。
- J1 `b=3600`，J2 `b=10800`；局部训练前缀 [0,1800)，anchors 1212…1752，C_A {2040, 2280}，C_B {2520, 2760}，E {3240, 3480}。
- 可得性规则（每站未来 48 点 ≥32 有限点 + 历史季节尺度 `min_pairs=32`）：两作业 6 个 origin 全部合格，无排除。
- Consumer：pooled Ridge α=1、逐窗中位数/MAD 归一化、L=192、H=48、period=24；cap=1.0；sMASE 尺度由作业局部 raw[0:origin] 计算，各臂共用。

### 1.2 仪器修复（所有 LLM 臂共同获得，不计学习）

1. **公共 H0**：canonical h0（`29298dc9…`）隔离副本只改 `verification.max_modified_fraction 0.35→1.0`（运行快照 `191856a3…`，与 TRAIN2 父快照相同）；不载入 TRAIN2 新卡。
2. **能力说明**：`runtime_capabilities` 改为“适应期 Runtime 执行有预算对照、Slow 在批次边界读真实反馈、Fast 不获得当前对象效果、probe unknown≠负证据、部署期无反馈无重训”；七方案菜单以无效果标签形式对所有 LLM 臂可见。
3. **传输**：中继 500/`InternalServerError`（`AgentTransportError`）原不在瞬态标记表中、不重试；本包内归为瞬态并沿用 3 次有界重传，另加**每批一次**对故障行的有界重问（Runner 原有“故障可重问、有效决策不重采样”规则）。
4. **隔离验证（运行中发现后加入，见 §5.1）**：`fast_agent.verify_candidate` 移入一个子进程，60 s 硬墙钟，超时记 `verification_timeouts.jsonl` 并返回封闭词表 `EXECUTION_FAILED`（该算子在该决策不可选）；相同 (程序, 数组, 限制) 走精确缓存。

### 1.3 臂与流程

| 臂 | 起始知识 | J1 | J2 |
| --- | --- | --- | --- |
| Static | identity W/V，一个共享模型 | — | — |
| Source-fixed | Source 完整形成评分选定 `winsorize>fft_decompose` | 每作业重训 | 同左 |
| Source-order | 顺序 [wins>fft, fft, iqr>wins, iqr, wins>hampel, hampel] | q1/q2 C_A → q3 复验 → q4 下一个 | 按自身 J1 读数重排：[iqr>wins, wins>fft, fft, iqr, wins>hampel, hampel] |
| Random | seed=20260911：[wins>fft, wins>hampel, hampel, iqr, iqr>wins, fft] | 同上 | 顺序不变 |
| F | 公共 H0 | q1 → 研究调用 → q2 → 反思调用（只改摘要）→ q3/q4 同 H 两分支 | 同左，重启 |
| A3 | 公共 H0 | q1 → 研究调用 → q2 → Slow 一处修改 → q3 旧 H / q4 新 H | 同左，重启 |
| A5 | **Source-derived H**（H0 + Source 卡） | 同 A3 | **J1 采用的 H**（整合调用弃权） |

Source 形成：Slow 一次 ADD `missingness-unknown-probe-prefer-identity`（条件 forecast ∧ missing_fraction>0 ∧ imputation_probe_direction==unknown），送达 J1 42 个决策视图中的 17 个 → A5-J1 的历史 treatment 非空。J1→J2 整合：Slow `no_proposal`（insufficient_public_evidence）→ A5-J2 从 J1 采用的 H（`5ae35901…` = H0 + Source 卡 + J1 卡）起步。

## 2. 主表（E，两作业分别列出，最后等权）

E 每作业两个 origin × 六站 = 12 个序列窗；收益 = Static sMASE − 本臂 sMASE，正为好；“受损”为收益 < −0.005 的序列窗数。

### J1（Static E = 1.248639；3240: 1.2338，3480: 1.2634）

| 臂 | 主部署分支 | E sMASE | 相对 Static | 受损/12 | 最大单次伤害 |
| --- | --- | ---: | ---: | ---: | ---: |
| Static | — | 1.248639 | 0 | 0 | 0 |
| Source-fixed | wins>fft | 1.279311 | −0.030672 | 5 | 1.617316 |
| Source-order | final = iqr>wins | 1.312443 | −0.063804 | 7 | 0.385671 |
| Random | final = hampel | 1.425586 | −0.176948 | 9 | 0.647483 |
| F | q4（同 H 第二分支） | 1.201113 | +0.047525 | 1 | 0.232559 |
| A3 | q3（旧 H） | 1.212362 | +0.036277 | 5 | 0.152343 |
| A5 | q4（新卡，全 identity） | 1.248639 | 0 | 0 | 0 |

检查点与影子分支（同口径 E）：F q2ckpt +0.005681、q3 **+0.178469**；A3 q2ckpt −0.002542、q4（新 H）−0.161036（最大伤害 1.957）；A5 q2ckpt −0.242444（9 受损，最大伤害 1.915）、q3（旧 H）**+0.175897**；两个搜索臂 q2ckpt = identity。

### J2（Static E = 5.516377；3240: 3.7308，3480: **7.3019**）

| 臂 | 主部署分支 | E sMASE | 相对 Static | 受损/12 | 最大单次伤害 |
| --- | --- | ---: | ---: | ---: | ---: |
| Static | — | 5.516377 | 0 | 0 | 0 |
| Source-fixed | wins>fft | 4.037803 | +1.478574 | 3 | 0.334024 |
| Source-order | final = identity | 5.516377 | 0 | 0 | 0 |
| Random | final = hampel | 6.019582 | −0.503205 | 10 | 1.736380 |
| F | q3 | 3.962664 | **+1.553713** | 4 | 0.464733 |
| A3 | q3（旧 H） | 6.237290 | −0.720914 | 8 | 1.883574 |
| A5 | q3（全 identity） | 5.516377 | 0 | 0 | 0 |

检查点与影子分支：F q2ckpt +1.166000、q4 = q3；A3 q2ckpt **+1.213859**、q4（新 H）−0.432177；A5 三分支均 = Static；搜索臂 q2ckpt = wins>fft +1.478574。

### 等权两作业

| 臂 | E sMASE | 相对 Static |
| --- | ---: | ---: |
| Static | 3.382508 | 0 |
| Source-fixed | 2.658557 | +0.723951 |
| Source-order | 3.414410 | −0.031902 |
| Random | 3.722584 | −0.340076 |
| F | 2.581889 | +0.800619 |
| A3 | 3.724826 | −0.342318 |
| A5 | 3.382508 | 0 |

对照差（sMASE 差，前者更低为好）：A5−A3 = J1 +0.036277 / J2 −0.720914 / 等权 −0.342318；A3−Static = J1 −0.036277 / J2 +0.720914；A5−Source-fixed = J1 −0.030672 / J2 +1.478574；A5−Source-order = J1 −0.063804 / J2 0；A5−Random = J1 −0.176948 / J2 −0.503205；A3−F = J1 +0.011248 / J2 +2.274627；A5−F = J1 +0.047525 / J2 +1.553713。

这是两个作业、同六站、每作业两个 E origin、一次 LLM 课程，不是独立样本；不报显著性。

### 2.1 held-in 读数（各臂各自消费，Static 参照两块各一次单独登记）

| 作业/块 | Static | Source-fixed | Source-order q1→q4 | Random q1→q4 |
| --- | ---: | ---: | --- | --- |
| J1 C_A | 1.342942 | 2.288610 (−0.946, 12/12 受损) | wins>fft 2.2886 → fft 2.8974 | wins>fft 2.2886 → wins>hampel 1.6055 |
| J1 C_B | 1.447172 | 1.513280 (−0.066) | identity 1.4472 → iqr>wins **1.1325 (+0.315)** | identity 1.4472 → hampel 1.3760 (+0.071) |
| J2 C_A | 2.557782 | 1.422347 (**+1.135**) | iqr>wins 1.9103 (+0.648) → wins>fft 1.4223 | wins>fft 1.4223 → wins>hampel 2.2481 |
| J2 C_B | 1.567985 | 1.757512 (−0.190) | wins>fft 1.7575 → fft 1.6060 → 终选 identity | wins>fft 1.7575 → hampel 1.3695 (+0.198) |

| 作业/臂 | q1 (C_A) | q2 (C_A) | q3 旧 H (C_B) | q4 新 H / 第二分支 (C_B) | 采用 |
| --- | ---: | ---: | ---: | ---: | --- |
| J1 F | 1.401240 | 1.416315 | 1.683552 | 1.483893（同 H） | q4 |
| J1 A3 | 1.365909 | 1.336484 | 1.501062 | 1.627427（新卡） | q3 |
| J1 A5 | 1.377924 | 1.362486 | 1.527534 | 1.447172（新卡，=Static） | q4 |
| J2 F | UNKNOWN（2280 一站协议失败） | 2.087564 | 2.022731 | 2.022731（同 H，决策完全相同） | q3 |
| J2 A3 | 2.504530 | 2.087901 | 1.406716 | 1.462120（新卡） | q3 |
| J2 A5 | 2.557782 (=Static) | 2.656168 | 1.567985 (=Static) | 1.567985（同 H，Slow 弃权） | q3 |

## 3. 机制表：实际发生的链条

每条为 `当前观察 → q2 申请 → 真实反馈 → Skill diff → 新旧 H 行为差 → E 差`。

| 单元 | q1 观察（W 非 identity 数/6；V 主要动作） | q2 申请（研究调用输出） | q2 真实反馈 | Skill diff | 新旧 H 行为差（C_B 再求解） | E 差 |
| --- | --- | --- | --- | --- | --- | --- |
| J1 F | W 2/6（outlier_mad, outlier_iqr）；V 含 repair_level_shift/impute_linear | **受控对照**：只改预测侧，V 全 identity，W/模型复用 q1，0 拟合 | 1.401→1.416，更差；检查点保留 q1 | 无（反思调用只改摘要文本，且被过滤器扣留） | 同 H 两分支：q3 W 5/6 用 outlier_*（−0.236，10/12 受损）；q4 W 全 identity（−0.037） | q3 +0.178 vs q4 +0.048：**未采用的分支在 E 上更好** |
| J1 A3 | W 1/6；V 含 repair_level_shift（s02 两窗 +0.61/−0.91） | **受控对照**：只改预测侧 V=identity | 1.366→1.336，+0.029 ≥ 0.005 → 检查点 q2 | ADD `gate-level-repair-on-public-excursion`（条件 forecast，送达 30/30）：无 level 分数时禁 repair_level_shift、宽 robust-z 用 no_actionable_signal | q4 反而出现 impute_linear W 与更多 repair_level_shift V：1.627 vs 1.501 → 否决 | q4 −0.161（最大伤害 1.957）vs q3 +0.036 |
| J1 A5 | W 3/6 hampel（Source 卡影响）；V 含 repair_level_shift | **受控对照**：只改训练侧，W 统一 hampel，V 复用 q1 | 1.378→1.362，+0.015 → 检查点 q2 | ADD `unknown-level-clip-probes-prefer-identity`（条件 clipping/level probe unknown，送达 30/30）：probe unknown 时倾向 identity | q4 W/V 全 identity（=Static 1.447）vs q3 W 5/6 outlier_iqr（1.528）→ 采用 q4 | q3（旧 H）**+0.176** vs q4 0：**C_B 上的采用没有传到 E** |
| J1→J2 整合 | J1 四次查询与卡 | — | — | `no_proposal` | — | — |
| J2 F | q1 一站 PREPARE_FAILED（inspect 两次返回退化区间 [0,0]）→ q1 UNKNOWN | **候选搜索**：两侧都 pin outlier_iqr | 2.088（+0.470，2280 由 4.19 降到 3.31）→ 检查点 q2 | 无（反思文本送达） | 两分支决策**完全相同**（W s01/s02 iqr，V 除 s06@2520 repair_level_shift 外 identity）：2.023（−0.455，12/12 受损） | 该分支 E **+1.554**（全场最高）；q2ckpt +1.166 |
| J2 A3 | W 0/6；V 2040 用 iqr/hampel（0.815 vs Static 0.921） | **受控对照**：只改训练侧，W 统一 outlier_iqr | 2.505→2.088（+0.470）→ 检查点 q2 | ADD `robust-spike-intrinsic-clip`（条件 robust_z≥3，送达 14/30）：高 robust-z 时用一个 intrinsic 尖峰算子，不因未定位而选 identity | q4 W 6/6 hampel vs q3 W 2/6：1.462 vs 1.407 → 否决 | q3 −0.721、q4 −0.432、**q2ckpt +1.214**：C_B 选出的分支在 E@3480 崩溃（8.83） |
| J2 A5 | W 0/6，V 全 identity（=Static） | **受控对照**：只改训练侧，W 统一 hampel | 2.558→2.656（−0.098）→ 检查点 q1 | `no_proposal` | q3=q4 全 identity | 0 |

补充事实：
- 六次 q2 中五次为单因素受控对照（另一侧精确复用），一次候选搜索（J2 F）；六次研究调用全部返回合法计划，无 INVALID_PLAN。
- 给 Fast 的任务内摘要为结构化字段 + 过滤后的 `investigation_purpose`（含数字或效果词即扣留）：J1 F/A3、J2 A3/A5 的文本被扣留，J1 A5 与 J2 F 送达；F 的反思文本 J1 扣留、J2 送达。扣留规则对所有臂相同，原文与送达文本均在 `unit_state.json.summary_filter`。
- 卡片送达/使用是辅助诊断：J1 A5 卡片送达 30/30 且行为按卡片变化（全 identity）；J1 A3 卡片送达 30/30，行为反向（更多 repair_level_shift）。
- 没有任何 LLM 臂构造出七方案以外获益的多步程序；A3/A5/F 的非 identity 动作全是单步 `outlier_iqr / hampel_filter / outlier_mad / repair_level_shift / impute_linear`。
- A5 在 J2 的 90 个 Fast 决策 100% identity；其等权“优于 A3”完全由 A3 在 J2 的伤害解释，不是准备收益。

## 4. 与固定/搜索基线的比较是否解释了全部收益

- **J2 的大收益是块效应**：E@3480 Static 7.30，任何压缩训练材料尖峰的统一方案（Source-fixed 4.41、F 的 s01/s02 iqr 4.08）都有 +1.2～+1.6；这与 held-in 上 C_B 的方向相反（Source-fixed C_B −0.190，F 的同一分支 C_B −0.455）。held-in 反馈在 J2 不能预测 E。
- **Source 程序不跨来源迁移为固定规律**：`winsorize>fft_decompose` 在 J1 C_A −0.946（12/12 受损）、C_B −0.066、E −0.031；在 J2 C_A +1.135、C_B −0.190、E +1.479。KDD 形成段（+0.567）→ Beijing 两个作业四个块，符号 2 正 4 负（含 J1 E）。
- **便宜搜索**：Source-order 在 J1 由 q4 找到 iqr>wins（C_B +0.315），E 上 −0.064；J2 按 J1 读数重排后 q2 命中 wins>fft（C_A +1.135）但 C_B 复验 −0.190 → 终选 identity（E 0）。Random 两作业 E 均为负。历史数值复用（Source-order J2 重排）没有比 Source-fixed 更好。
- **固定 H 搜索**：F 两作业 E 均 ≥ Static（+0.048 / +1.554），高于 A3、A5、Source-order、Random，与 Source-fixed 相当（等权 +0.80 vs +0.72）。F 没有知识变化，其 J1 两条同 H 分支差 0.20（C_B）/0.13（E），说明单次 Fast 再求解的采样差异与本包所有“知识效应”同量级。
- 因此：LLM 臂高于 Static 的部分可由“当前作业内一次统一尖峰压缩 + J2 块效应”解释；没有需要用历史积累或调查技能来解释的剩余收益。

## 5. 故障、未知与仪器事项

### 5.1 J2 运行停滞（已修复、已恢复，见主计划 §6.8）

- 15:18 起请求速率骤降，15:33 后无调用完成；线程 50356 持 GIL。根 Agent 栈采样定位到 `Fast.prepare → _actionable_operators → verify_candidate → impute_ssm → statsmodels Kalman`。执行者独立进程限时复现（`stall_repro.json`）：J2 Changping 1800 点训练前缀（128 缺失、最长 93、默认周期估计 **360**）上 `impute_ssm` >300 s 被杀；J1 同站前缀（周期 0）0.15 s；J2 Guanyuan 192 点预测输入（71 连缺，周期 0）0.06 s。菜单预检查在同一前缀上 16 s 内完成前 10 个算子（denoise_stl 15.7 s），到 `impute_ssm` 不返回。
- 原进程 PID 28024 由执行者 16:00 主动终止（`restart_note.json` event 2）。修复为 §1.2 第 4 条；语义核对：J1 六个前缀与 J2 其余五个前缀，21 个合法算子逐个验证均 < 8.3 s（`prefix_probe_timing.json`），60 s 上限从不触发 → **已冻结 J1 决策与修复后同语义；候选可选性唯一变化是 J2 s02 训练角色下 `impute_ssm` 不可选**（`verification_timeouts.jsonl` 1 条）。未换插补器、未改周期、未改 identity/linear、未按缺失排除 J2。
- 隔离验证在最后一个进程内统计：897 次调用，568 次缓存命中，0 超时，子进程累计 4.0 s。

### 5.2 进程重启与传输故障

| 时间 | 事件 | 处理 |
| --- | --- | --- |
| 12:23 | 启动 | — |
| 12:43 | 执行者停止 PID 43608：中继 500 未被有界重传覆盖，3 条决策 FAULT | 补瞬态分类 + 每批一次故障重问后 `--resume`；故障决策重问，已完成决策未重采样 |
| 16:00 | 执行者停止 PID 28024（§5.1） | 隔离验证后 16:24 `--resume` |
| ~18:30 | Claude Code 宿主因内存不足杀掉启动课程的 shell 任务，课程进程随之退出 | 18:34 以独立进程 `--resume`；18:56 完成 |
| 17:56:09 | 中继瞬时不可达：4 条在途调用及其重传同秒失败 | 批内重问全部恢复 |

暂停时间 2252 s 计入 `paused_seconds`，不重置消耗。恢复段未重问任何已完成的 Slow/研究调用（Source 形成、四次 Slow 边界、整合各 1 次）。

### 5.3 无效决策与 UNKNOWN

- J2 F q1 predict@2280 s02：`PREPARE_FAILED` × 2（模型两次在 inspect 返回 `[0.0, 0.0]` 退化区间，被后验证拒绝；有界重问后仍失败）→ J2 F q1 在 2280 UNKNOWN，q1 均值 UNKNOWN；q2 检查点因此取 q2。其余 515 个最终决策行全部有效（0 fault）。
- J2 F 反思调用与 J2 A5 Slow 各 1 次；J1 A5 研究调用 2 次（第一次协议错误后有界重试）。
- E 输出 30 个分支 × 2 origin 全部冻结后才评分，`missing_frozen_outputs = []`。

### 5.4 其他限定

- `orchestration.json` 与 `unit_cost.fast_session_seconds` 只统计最后一个进程；按决策行汇总的 Fast 会话时间见 §6。
- 摘要过滤器把含数字/效果词的 purpose 文本整段扣留，四个单元被扣留，这削弱了“任务内状态”通道；规则已冻结不回改，原文保留。
- q3/q4 采用规则只比较两个分支，不设 Static 保底（任务书为 LLM 臂未规定保底）；J2 F 因此部署了 C_B 上 −0.455 的分支。数值搜索臂按任务书有 identity 保底。
- 模型缓存按训练赋值指纹跨臂共享（相同赋值 → 同一确定性解），只节省物理拟合，不传递任何读数；逻辑查询按臂计数。
- 站名匿名、Source UID 只对 Slow 可见；Slow 三张卡的适用条件全部为可观察特征，无 UID/数据集名。
- J1-E 按原冻结边界在 J2 全部输出冻结后才打开，与 J2 输入无关。

## 6. 成本

| 项 | 本包实际 | 上限 |
| --- | ---: | ---: |
| 实验 API 尝试（含重试、Source 形成 1、整合 1、研究/反思/Slow 调用） | **2644** | 4400 |
| 输入+输出 token | **22,267,546** | 30,000,000 |
| 物理共享拟合 | **23**（J1 13、J2 10；含 Static 2、菜单/搜索 10、LLM 臂 11；E 阶段 0 新拟合） | 100 |
| 有效运行墙钟 | **5.91 h**（21273 s；另暂停 2252 s） | 12 h |
| Fast 并发峰值 / 拟合写者 | 4 / 1 | 4 / 1 |

| 单元 | Fast 决策 | API | token | 决策墙钟合计（/4 槽） | 逻辑查询 | 物理拟合 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J1 F | 90 | 457 | 3,811,982 | 3.92 h (0.98 h) | 4 | 2 |
| J1 A3 | 78 | 395 | 3,288,819 | 3.17 h (0.79 h) | 4 | 3 |
| J1 A5 | 90 | 457 | 3,891,194 | 3.77 h (0.94 h) | 4 | 2 |
| J2 F | 78 | 404 | 3,286,205 | 2.89 h (0.72 h) | 4 | 2 |
| J2 A3 | 90 | 477 | 3,933,511 | 3.83 h (0.96 h) | 4 | 2 |
| J2 A5 | 90 | 452 | 3,970,428 | 3.37 h (0.84 h) | 4 | 0（全 identity，命中 Static 模型） |

单元上限 650 API / 450 万 token / 2 h（按 4 槽归一）均未触及。Source 形成 1 次调用 55,395 token；整合 1 次 30,012 token；影子分支（未采用的 q2ckpt/q3/q4 之一或二）E 生成约每单元 58–125 次 API（`cost.by_unit` 的 `E-*` 键），已含在上表。数值臂 0 LLM。TRAIN2 的 500 API/10 拟合是历史账，本次只读取其形成材料（压缩后 84,704 字符）。

## 7. 判读与下一步依据（不预写路线）

按任务书 §8.3 逐条：

| 结果 | 本包读数 |
| --- | --- |
| 时间、角色、共享拟合或评分错误 | 未发现；§5.1 是执行成本故障，修复不改变已冻结 J1 语义 |
| Static 之外的固定/搜索也无收益 | 不成立：Source-fixed 等权 +0.724（J2 主导），Source-order/Random ≤ 0 |
| Agent 高于 Static 但不及强固定/搜索 | F 高于 Static（+0.048/+1.554）且略高于 Source-fixed；A3、A5 不高于 Static → 只有固定 H 臂有 L1 局部正号，且由同 H 再求解产生 |
| 有修改但后续新 H 无增益 | **成立**：三处新 H 在 E 上均不优于旧 H 分支（+0.197、+0.176、−0.289 但两者皆负） |
| A5 优于 A3 但不优于 Source-order/F | 等权 A5 > A3 仅因 A3 在 J2 受损；A5 < F、< Source-fixed；J1 上 A5 < A3 |
| A5 两作业均不负、等权 ≥0.005 且有净准备收益 | **不成立**（J1 A5−A3 = +0.036；A5 净准备收益 0） |
| A5 q2 质量不劣于 A3 最终 | J1 不成立（+0.279），J2 成立（−0.721）：混合，不记效率线索 |
| Source 为空、关键臂不完整 | Source treatment 非空且送达；六单元完整；J2 F q1 一个 origin UNKNOWN |

给用户/Planner 的事实依据（不是建议）：
1. 本包在两个真实作业上把“A5 = 历史知识 + Target 适应”跑完整了，结果是历史知识（Source 卡 + J1 卡）在 J2 表现为 100% identity；两次 Slow 弃权、三次 ADD 都朝“更保守/门控”方向，没有一次形成可复用的准备指导。
2. held-in（C_A/C_B）对 E 的预测力在 J2 为负相关级别（Source-fixed、F、A3 的块间符号翻转），J1 上三条影子分支比采用分支好 0.13–0.18；同 H 再求解的采样差异 0.13–0.20。在这种噪声下，0.005 的 MATERIAL 采用门与“一处 Skill 修改”都不能被本包分辨。
3. 本包的正号（Source-fixed、F）都能由“统一尖峰压缩 + J2 E@3480 块效应”解释，不构成历史积累或调查技能的证据。

2026-09-15 路线检查点保留；本包不授权扩实验或改协议。

## 8. 交付与命令

- 入口：`evaluation/main_protocol_p4/dev_train3_cross_job.py`（单一 Runner）；启动/恢复：`python _scratch/dev_train3_cross_job/launch.py [--preflight | --resume] --course-id course_beijing_j1_j2_20260911`；外部评分在课程末尾由 `score_all_E` 统一执行；复算与工件：`python _scratch/dev_train3_cross_job/summarize.py --course-id course_beijing_j1_j2_20260911 --write-artifact`（0 LLM/0 拟合/0 数据读取）。
- 0-LLM 合成接线检查：`python _scratch/dev_train3_cross_job/check_wiring.py`（33/34，唯一 FAIL 为脚本化 Slow 重发相同 PATCH 被判 no-op，属正确拒绝；后已把该结果纳入预期）。
- 故障复现：`_scratch/dev_train3_cross_job/stall_repro.py` → `stall_repro.json`；`prefix_probe_timing.json`。
- 工作目录 `_scratch/dev_train3_cross_job/course_beijing_j1_j2_20260911/`：`course.json`（冻结配置）、`data_receipt.json`、`budget.json`、`tokens.json`、`llm_log/*.jsonl`（全部原始响应）、`source/`（Source 卡、提议、曝光）、`integration/`、`J1|J2/<arm>/`（逐决策检查点、q1–q4 记录、Slow 输入/提议/store、E 冻结预测与评分）、`verification_timeouts.jsonl`、`restart_note.json`、`live.log`。
- 紧凑工件：`artifacts/main_protocol/dev_train3_cross_job.json`。
- 未 commit、未 push，未开放密封数据，未修改 canonical h0、算子、Consumer、评分或历史结果；`AGENTS.md`/`DECISIONS.md` 未动。
