# GEFCom 600-936 切片——EXPOSED_DEVELOPMENT_NEGATIVE_TRANSFER_CASE（2026-08-09）

外部审核第六轮裁决：**不按"新自然数据 A5/A3"实现运行**。本切片已被
暴露，准确判定：

> **MECHANICAL_FEASIBILITY_PASS + EXPOSED_DEVELOPMENT_SLICE**
> （不是 fresh/natural A5/A3 确认实验）

## 1. 暴露状态（审核五条承重问题核对——全部属实）

| # | 承重问题 | 核对 |
|---|---|---|
| 1 | **Target Outcome 已打开**：本文件记录了 @792/@888/@936 精确 gain 并据此预测路径与 verdict | 属实。context_exposure = INSTANCE_SEEN；outcome_exposure = EXPOSED。再运行只能验证代码复现预期，不能作为新能力证据 |
| 2 | **"600-936 区间全新未用"不成立**：评价窗口与旧链高度重叠——新 R1 [792,840) 与旧 [784,832)、[832,880) 重叠；R1D [840,888) 与旧 [832,880)、[880,928) 重叠；R2 [888,936) 与旧 [880,928)、[928,976) 重叠；R2D [936,984) 与旧 [928,976)、[976,1024) 重叠；且 @936 已出现在历史 W2 报告 | 属实。原 §1"全新未用"声明撤销 |
| 3 | **Source 答案导向选择**：A5 只放 outlier_iqr，而设计者已知它在 Target @792 为正——Target-outcome-informed curation | 属实。只可作"已知正经验能否通过链路产生行动"的 positive control，不能证明 Harness 在未知 Target 上自然利用 Source Experience |
| 4 | **"更快"未对齐核心反馈预算**：A3 第一 probe 是 verifier REJ（不消耗 Target downstream Support），第一次真实 Support 就由 hampel 命中——按 proposal 次数 A5=1/A3=2，按 Target Support receipt 数两者都可能是 1 | 属实。应分别报告 proposal 数 + first-positive Support receipt index |
| 5 | **Episode 写回不能代理 Skill 形成**：项目明确区分 Experience Episode（一次 Action–Response）vs Target-local Skill（可检索、可执行、受 signed feedback 控制的局部能力） | 属实。"以 Episode 写回代理 Target-local Skill"不成立 |

## 2. 保留的数据事实（development case 记录，非预注册）

| 算子 | @600(S) | @648(S) | @792(R1) | @888(R2) | @936(R2D) |
|---|---|---|---|---|---|
| outlier_iqr | +0.03178 | +0.00941 | +0.03762 | −0.00486 | −0.17205 |
| hampel_filter | −0.00836 | +0.04695 | +0.04344 | −0.04162 | +0.15018 |
| denoise_median | — | — | REJ | REJ | REJ |
| winsorize | −0.02218 | −0.05242 | −0.02731 | −0.01554 | +0.01476 |

另：NOAA 全算子 cohort 级 verifier 拒（3 windows 锚点级固定）不可作 Target；
NN5（legacy 20 支 × 791）四窗口不足。

## 3. 本切片的残余价值（开发回放）

- Source 正经验让 R1 快速命中 → 验证"正经验先引导"；
- 同一算子在 R2/delayed 翻转（@888 微负、@936 强负）→ 验证 **signed
  feedback 是否及时降级过期经验**（防负迁移）；
- 但与已通过的自动写回/优先级反转实验（§7 二十九修正）高度重合，
  边际证据有限——**不新建专用 Runner（不实现 run_v1_new_data_a5_a3.py）**。

## 4. 真正 sealed 的 Target 数据（下一步）

审核第 6 条要求：寻找 outcome-sealed 的新 Target cohort/series/dataset。
registry 盘点发现完全未用候选：

| dataset | 系列数 | 长度 | 状态 |
|---|---|---|---|
| metr_la | 207 | 1024 | 完全未用 |
| uci_electricity_load_diagrams | 370 | 1024 | 完全未用 |
| monash:traffic_hourly | 862 | 1024 | 完全未用 |
| monash:nn5_daily（全量 91 支） | 91 | 714-791 | 部分未用（legacy 20 支已有历史） |

Sealed 流程要求（审核原文）：
1. 只检查长度、可见 Context、动作合法性——**不扫描 Target gain**；
2. 打开 Target 前冻结完整 Source Memory（固定探测计划，不按 Target 结果
   挑正 Episode）；
3. A5/A3 同 Agent、动作空间、真实 Support receipt 预算（proposal 数与
   first-positive Support receipt index 分开报告）；
4. 每次 Support 后立即写回；后续不重叠窗口正常 prepare；
5. 形成正向 Workflow 必须实际写成 LOCAL_DRAFT/LOCAL_ACTIVE Skill，并验证
   下一轮正常入口能执行它；
6. delayed outcome 最后打开，报告 harm、abstention、首次正向 Support 次数、
   Skill delayed utility。
