# R4A · 可识别性证据台账（G2；只读整理；2026-09-07）

地位：把项目里已经做过的测量列成一张表。**不重算、不改写原文数字、不主张新结论。**
问题：部署可见 Observation / 特征能否识别处理条件（区分受益/受害、预判伤害、排序候选）。

`CURRENT_STAGE.md` 不在本仓库内，原文在上级目录
`c:\Users\辉\Desktop\Agent\CURRENT_STAGE.md`；下表该文件路径按用户指定书写。

极性：`负` = 未能用该特征/方法识别条件；`正` = 原文记为通过/有用；`上界` = 事后/oracle，明确不可部署。

## 主表

| 测量编号 | 特征/方法 | 数据 | 关键数字 | 项目判词 | 文件路径:行号 | 极性 | 证据类型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W54 | 同 Dataset 不同 Series、192-step 历史 Context、固定 K=5 peer 标准化 contradiction（不读 evaluation future / exact credit / Dataset ID） | Traffic / FRED / NN5 / METR 自然 row×block masking | dataset-macro AUROC=`0.49737`；`2/4` Dataset AUROC>0.5；top-quartile 正 action 比例=`0.51389` vs 全部 `0.53906`（uplift=`-0.02517`）；分 Dataset AUROC `0.42978/0.47844/0.51336/0.56791` | “similar-history peer outcome contradiction”不能解释自然 row×block removal headroom；Observation 与 W23/W24 直接自然 Capability 路线关闭 | `CURRENT_STAGE.md:911` | 负 | 数据模式 · 自然 |
| W61 | Program-conditioned LODO：局部片段形态 + action deformation + cohort prevalence/alignment + Task/Consumer context | 复用 W23/W24 自然 row×block action（0 新 Consumer fit） | Program-conditioned LODO balanced accuracy 约为 `0.5038` | 否定“继续给局部 row×block action 增加普通 TS 特征”；LocalActionEpisode 不再作为可独立晋升的 Capability 单位 | `CURRENT_STAGE.md:802` | 负 | 数据模式 · 自然 |
| W39 | 被选 action 的 support-response 均值（exact singleton response）对 harm 排序；对照 grouped-support scalar；dataset LODO 阈值 | 已曝光受控 E1-TR | 排序 AUROC `0.8833` vs grouped-support scalar `0.6917`；LODO 后两者都仍留 `1` 个 harmful split；response 保留 `85.86%` 正收益，不优于 grouped scalar `88.53%` | 值得成为 Action–Response Context，尚不足以单独形成新 Skill | `CURRENT_STAGE.md:855` | 负（LODO 编译失败；AUROC 本身高但是 Support 响应，不是 origin 前模式） | 历史结果/Action–Response · 受控 |
| P4d-Targeter | 深度 3 决策树、15 维部署可见特征、6 程序菜单；best-fixed / 交叉拟合 Targeter / per-series oracle | KDD without-missing Forecast；跨面 12 折（主）、跨 origin 6 折（次） | best-fixed **+0.2629** / Targeter **+0.1908** / per-series oracle +0.6106；Targeter 胜过 best-fixed 5/12；平均低于 best-fixed 0.072；对 +0.3477 oracle 空间捕获率为负 | 正式判词 `FEATURES_DO_NOT_BEAT_A_FIXED_CHOICE` | `AGENTS.md:328`–`329`；`docs/P4D_NATURAL_GAP_LINE_CLOSURE_2026-09-01.md:145`–`151` | 负 | 数据模式 · 自然（无缺口版） |
| P4-22d | 22 维部署可见特征、leave-one-origin-out 多变量 logistic | 11 个 Support 阶段单元 × 20 序列 → 216 有标签观测（46 受害 / 170 受益） | 分组 AUC **0.587**；最佳单特征 `modified_longest_run` 0.565；7 维缺失类特征在该数据上是常数（grouped AUC 恰为 0.500），故 AUC 0.587 实为 15 维有效特征 | 部署可见特征不能分离伤害；多变量结果跨 origin 不稳定（三个 origin 低于随机） | `AGENTS.md:493`–`494`；`docs/P4_CONFLICT_PER_SERIES_AUDIT_2026-08-31.md:174`；`docs/P4D_NATURAL_GAP_LINE_CLOSURE_2026-09-01.md:354` | 负 | 数据模式 · 自然（无缺口版） |
| D1 | 四信号：S1 预测分歧 / S2 证据区距离 / S3 行为超覆盖 / S4 改动量；对 `harmed_material` 的 AUC | Source-v3 主窗口集 n=46，harmed=12；`EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING` | S1 AUC=0.6495 CI=[0.5177, 0.8079]；S2=0.511；S3=0.5968；S4_mod_fraction=0.451；S4_mod_magnitude=0.4032；全部 `DOES_NOT_SEPARATE` | 总判 `NO_OUTCOME_FREE_SEPARATOR` | 数字不在任务书 `docs/D1_ROUTING_HARM_DIAGNOSTIC_2026-09-03.md`（该书无 AUC 结果）；原文 `artifacts/main_protocol/p4ab_routing_harm_diagnostic.md:19`–`23` | 负 | 数据模式 · 自然 |
| HEC1-BSG | Best-Safe-Global：评价面 Outcome 事后取优 | HEC-1 development；23 个可评单元 | 14/23 有安全 non-identity 程序；累计安全上界 `+5.527089`；1046 fits | 冻结程序菜单存在明显 headroom；**不可部署**的 outcome-side 上界，不是方法 baseline | `docs/HEC1_ZERO_LLM_DIAGNOSTIC_CLOSURE_2026-09-04.md:20` | 上界 | 历史结果/Outcome · 自然 |
| HEC1-VS | validation-search：Support 选、+144 计分 | 26 单元 | 17/26 有至少一个 Support-safe 候选；16 个可评 non-identity 部署中 15 个物质正增益，仅 7 个仍过评价面四线 | 确定性搜索能提高供给和原始收益，但不能可靠保证未来安全 | `docs/HEC1_ZERO_LLM_DIAGNOSTIC_CLOSURE_2026-09-04.md:21` | 负（未来安全） | 历史结果 · 自然 |
| HEC1-TR | 全候选 transfer：Support-safe 候选逐一在 +144 重部署 | `summary`：scoreable_units_with_support_safe_candidates=16；support_safe_candidate_pairs_redeployed=34；candidate_pairs_retaining_evaluation_safety=10；units_where_selected_retains_safety=7；units_rescuable_by_another_support_safe_candidate=1；units_where_no_support_safe_candidate_retains_safety=8 | 34→10 仍安全；选择排序只解释 1/16 个机会 | “选错候选”不是主因；Support→evaluation 的效应/安全迁移不稳 | `docs/HEC1_ZERO_LLM_DIAGNOSTIC_CLOSURE_2026-09-04.md:22`、`:51`、`:65`；`artifacts/main_protocol/hec1_validation_transfer_v11p0.json:10`–`16` | 负 | 历史结果 · 自然 |
| E2.59 | phase-aligned `actual - seasonal-naive` response，对 auxiliary groups 做 cosine/RMSE rank | KDD / Rideshare 等；B={0,1,2} | A5/A3 macro AdaptAUC=`0.28454/0.36176`；`2/4` target 不差；KDD PM2.5 一个 harmful target；3/4 cell 改变了顺序 | `phase_aligned_response_alignment_v1` 按预注册 Gate 关闭 | `CURRENT_STAGE.md:673` | 负 | 历史结果 · 自然 |
| E2.62 | LLM 语义计划 + 随后 `phase-aligned historical channel-binding PolicyEpisode` | 四个 outcome-sealed target roles | 计划四次都改变 top-1；A5/A3=`0.23484/0.61998`，差=`-0.38513`；CO 一次 Support 正、Query 负；Observation 将 A5 提到 `0.45678` 并消 harm，仍低于 A3，`1/4` 不差 | cross-channel binding family 按 Gate 关闭 | `CURRENT_STAGE.md:681` | 负 | 历史结果 · 自然 |
| E2.84 | `pseudo_gap_program_response_order`：历史 192 步内临时遮挡，比较 period-median vs AR 自监督重建误差以排序 Workflow | PRSA O3/PM10（O3 表面先前已开，PM10 首次） | 新 A5=`0.00572`，低于固定 Source order `0.00905` 和 A3 `0.00738`；harm count=`2` | 自监督重建只能做 candidate/proxy evidence，不能作为 Utility 归因/Promotion/Active Memory；数值 FAIL | `CURRENT_STAGE.md:288` | 负 | 数据模式（自监督） · 自然 |
| p4y | 冻结 Scope 类内至多加一条部署可见谓词；**用 Outcome 选子句** | `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`；工件无顶层 `summary` 键 | 顶层 `counts.probes_with_a_feasible_scope_in_class`=6；`probes_where_uid_level_selection_would_clear`=7；`alignment_failures`=0；`verdict`=`FEASIBLE_SCOPE_EXISTS` | 上界、按 Outcome 选择，不是可部署政策；`evidence_grade`=`UPPER_BOUND_SELECTED_ON_OUTCOMES_NOT_A_POLICY` | `artifacts/main_protocol/p4y_oracle_scope_bound.json:2`–`4`、`:885`–`:890` | 上界 | 历史结果/Oracle · 自然 |
| m_r0k-L1L2 | L1 逐序列事后选择；L2 合法空间内事后最优（整段固定 / 逐单元换） | DEV-AUTO-1 / m_r0k 后 16 单元 | Workflow 线上 L1 是整段固定最优的 **2.4–2.9 倍**（0.4486 / 0.1561、0.4114 / 0.1703）；L2 switching 比整段固定多 **+0.039（Support）/ +0.042（delayed）** | L1 要求逐序列挑程序，现有 Scope 类表达不了；剩余空间要逐单元/逐序列判断能力 | `artifacts/main_protocol/m_r0k_scope_workflow_development__run1.md:260` | 上界 | 历史结果/Oracle · 自然 |
| E2.50–E2.56 | 按历史 PolicyEpisode 排序；首次正向 exact Support confirmation 后停止 | 自然 Forecasting/panel：M4 Hourly 新 cohort、Pedestrian、Rideshare、KDD PM2.5 | 三个独立自然 Target 上 dataset-macro A5/A3 AdaptAUC=`0.39019/0.20139`，差值=`+0.18880`，`3/3` 为正且 harmful Target=`0` | `CROSS_DATASET_SELF_EVOLVING_CAPABILITY_SLICE_PASS`（窄 Forecasting/panel Skill） | `CURRENT_STAGE.md:649`、`:653`、`:655`、`:661`、`:663` | 正 | **历史结果** · **自然** |
| W46 | cohort Observer：训练输入+训练标签的类别差异拓扑；Task Context 为 global/coarse 时 bound repair，为 local-event 时 abstain | 本 family 未参与开发的 NN5/METR 自然底座；受控 local-event family | 四节点两 task world precision/recall=`1.0`；NN5 coarse forced gain `+0.5`、METR `+0.0625`；强制 event repair 均为 `-0.5`，task-scoped event harm 均为 `0`；相对 B0 `+0.25/+0.03125` | 首次把 Source evidence 编译成最小 Harness 行为并通过 | `CURRENT_STAGE.md:881` | 正 | 数据模式+标签 · **受控注入** |
| W47 | 同 W46 Observer，接入 `TaskQualityContract`；A3/A4/A5，B={0,1,2} | 未参与 W42–W46 的 GEFCom2014 weather 与 OPSD hourly load | Observer precision/recall `1.0/1.0`；coarse Query gain `+0.5`、event `-0.5`；A3 AdaptAUC `0.875`，A4/A5 `1.0`，`A5-A3=+0.125`；A5 event harm `0`；unscoped event harm `0.5` | 第一条真实运行的等 Target-feedback 适配轨迹 | `CURRENT_STAGE.md:885` | 正 | 数据模式+标签 · **受控注入**（Target 底座自然，任务事件受控） |
| K0 | `outlier_mad` @ `z_peak>=3`（Phase S-v1.1 一张卡） | KDD with-missing Phase S，`[200:239]`×origin 2616 | delayed 四线全过（+0.360 / hf 0.10 / msh 0.13）；评价面 +0.473；K0 非空、审计 `K0_FREEZE_CLEAN` | 项目首张自然数据上经权威门存活的 Skill；这是**过门形成 Skill**，不是“该谓词能跨单元识别受益/受害”的测量 | `docs/STATE_ONE_PAGE_2026-09-03.md:341`–`344`；`docs/NEXT_EXPERIMENT_EXECUTION_PLAN_2026-09-05.md:15` | 正（过门） | **数据模式** · **自然**（单次过门，非识别力检验） |

表内共 **19** 行（含上界 3 行、正向 4 行、负向 12 行）。

## 正向条件化证据的口径

| 条目 | 用的是历史结果还是数据模式 | 自然还是受控注入 | 是否算“部署可见模式识别条件并成功” |
| --- | --- | --- | --- |
| E2.50–E2.56 排序 3/3 | **历史结果**（历史 PolicyEpisode / phase-aligned 历史 response 排顺序；当前 Target 仍要 Support 确认） | **自然** | 否：识别靠的是历史效用记录，不是 origin 前可见模式本身 |
| W46 / W47 | 数据模式 + 训练标签拓扑 | **受控注入**（local-event / fit artifact） | 否：成功发生在受控 family |
| K0 `outlier_mad @ z_peak>=3` | **数据模式**（`local_robust_z_peak` 谓词） | **自然** | 否：一次权威门通过，随后 P4d/D1/HEC-1 等测量并未显示该词汇能稳定区分受益/受害 |

## 原文不一致（不自行修正）

1. **D1 路径**：用户指向 `docs/D1_ROUTING_HARM_DIAGNOSTIC_2026-09-03.md` 的 AUC 0.45–0.65。该文件是任务书，**不含实测 AUC**。实测在 `artifacts/main_protocol/p4ab_routing_harm_diagnostic.md:19–23`：0.6495 / 0.511 / 0.5968 / 0.451 / **0.4032**。0.45–0.65 覆盖 S1 与 S4_mod_fraction 的约数；S4_mod_magnitude=0.4032 **低于 0.45**。
2. **W54 AUROC**：任务书 §0 写 `W54 AUROC 0.497`；`CURRENT_STAGE.md:911` 原文是 `0.49737`。
3. **22 维中常数维数**：`AGENTS.md:493` 写 **7 维**缺失类特征为常数，故 AUC 0.587 为 15 维读数；`docs/P4_CONFLICT_PER_SERIES_AUDIT_2026-08-31.md:182` 点名 **六个**特征恰为 0.500。`P4D` §4 菜单用 **15 维**（:140），与 7 维常数口径一致，与“六个特征”字面不一致。
4. **p4y 顶层无 `summary`**：只有 `counts` + `verdict`。上表用这两处，未发明 `summary`。
5. **E2.51**：`CURRENT_STAGE.md` 从 E2.50 直接到 E2.52，**没有 E2.51 这一编号**。E2.50–E2.56 的 3/3 出现在 `:663`（Pedestrian、Rideshare、KDD）。
6. **HEC-1 `1/16`**：closure `:65` 写“选择排序只解释 1/16 个机会”；transfer `summary` 对应 `units_rescuable_by_another_support_safe_candidate=1` 与 `scoreable_units_with_support_safe_candidates=16`。两处数字一致，语义是“换候选可救 1/16”，不是 1/16 选对。
7. **K0 谓词字面**：`hec1_k0_freeze_phase_s_v11_live.json` 只记 skill_id `fast_winner_forecast_ridge_smase_outlier_mad` 与过门，**文件内无 `z_peak>=3` 字符串**；该谓词出现在 `docs/STATE_ONE_PAGE_2026-09-03.md:341` 与 `docs/NEXT_EXPERIMENT_EXECUTION_PLAN_2026-09-05.md:15`。

## 三行小结

- 负向测量共 **12** 项（W54、W61、W39-LODO、P4d-Targeter、P4-22d AUC 0.587、D1 四信号、HEC1-VS 未来安全、HEC1-TR 34→10、HEC1 1/16、E2.59、E2.62、E2.84）。
- 正向测量共 **4** 项（E2.50–E2.56 的 3/3、W46、W47、K0 过门）。另有 **3** 项 oracle/上界（HEC1-BSG 14/23、p4y、m_r0k L1/L2），不计入正负识别成败。
- 其中真正用“自然数据的部署可见模式”识别处理条件并成功的有 **0** 项。
