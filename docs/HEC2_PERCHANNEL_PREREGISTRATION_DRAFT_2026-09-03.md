# HEC-2 ① 预注册草案:per-channel Ridge(只换 Consumer;HEC-1 收口后冻结即发,中间不开设计讨论)

日期:2026-09-03。地位:sol 建议"现在写好、等 HEC-1 收口再跑";主线起草;**HEC-1 读数出来前不冻结、不改**。
接收:Opus(runner 复用 `run_hec1.py`,Consumer 换成 per-channel;不新建框架)。

## 1. 唯一变量

- **变**:Consumer 结构 = per-channel Ridge——每条 served 序列用**自身**历史上的 anchored 训练窗口各拟合一个模型
  (raw 与 program 各一);Program 只作用于该序列自身的训练窗口与 serving context。同 Ridge 超参、CONTEXT 192 /
  HORIZON 48、anchor 列表。
- **不变**:数据(KDD with-missing,`p4ac` 六块)、Phase S/T 块与单元、三顺序、程序空间(单算子 + ≤2 组合)、Scope 类、
  风险线(0.005 / 0.20 / 0.30 / MIN_TREATED 5 / 0.35)、三态机、双环(k=5、外环 LLM ≤2/步、replay 每臂 100%、预留)、
  预算(Phase S ≤120;顺序各 ≤500)、评价面 +144、可评性清单(重算并冻结,允许与 HEC-1 不同——记 `N_T_eff_pc`)。
- **K0**:Skill 以 Task × Consumer 为键 → HEC-1 的 K0 **不可直接复用**;Phase S 在 per-channel 下重跑形成 K0_pc。
  空则同 HEC-1 缩臂规则。
- **代码**:同一 commit 家族,只允许 Consumer 适配 diff;runner 断言 HEAD;`methods/ttha/*` 零改动。

## 2. 预注册预测(源自 D1 反事实与 D5)

| # | 预测 | 读数 | D1/D5 依据 |
| --- | --- | --- | --- |
| P-C1 尾部 | 验证窗口上 `max_single_series_harm` 的分布整体左移;pooled 下因 msh 失败的候选,在 per-channel 下 msh 失败比例显著下降 | 逐窗口 msh;msh 失败占比 pooled vs pc | D1:msh 0.50→0.08、0.88→0.14、0.09→0.06 |
| P-C2 基座 | 轻度受害(< −0.005)序列数与 `harmed_fraction` **上升**;hf 失败占比上升 | 逐窗口 hf | D1:hf 0.20→0.35、0.15→0.30 |
| P-C3 存活 | 过 delayed 并存活的 Draft 数 pc ≥ pooled(方向预测,弱) | 生命周期表 | 尾部线是 pooled 下的绑定约束 |
| P-C4 曲线 | A3-online − A3-frozen 的 D_o:**不预测方向**,只报;与 HEC-1 并列 | 曲线 | — |
| P-C5 路由 | D5 若 `ROUTE_DOMINANT`,pc 下 route 分量 ≈ 0(反事实核) | D5 pc 对照 | 构造上成立 |

| P-C6 收益 | **per-channel 下聚合增益下降**:treatment 退化为 context-only(D5 中位 ≈ +0.1、11 窗 9 正),失去 pooled 下经模型路由获得的大部分增益 | 逐窗口聚合增益 pooled vs pc;`full − ctx` 份额 | D5:route_i 11 窗 9 正且常为增益主体(2376 delayed +0.178 vs ctx −0.012;2616 delayed +0.474 vs +0.010);per-channel 对照 \|route\| 均值 0.404→0.104 |

**机制前提(D5,2026-09-03,`ROUTE_DOMINANT`:严重伤害 8/10 路由为最负分量、代数份额 73%,交互 ≈ 0)**:pooled 下"数据准备
程序"的收益与伤害走**同一条管道**——被服务序列切换到由他序列准备后训练行拟合的模型。per-channel 拆掉的是整条管道,不只是尾部;
D5 的 per-channel 对照还制造了 1 条新严重伤害(T267@2376)。故 HEC-2 ① 的判词须同时读 P-C1/P-C2(伤害形状)与 **P-C6(收益
水平)**,不得只报前两者。

判词在 P-C1/P-C2/P-C6 上给(`CONSUMER_SHAPES_HARM` / `NO_SHAPE_CHANGE`,附 `GAIN_RETAINED` / `GAIN_LOST` 标注);P-C3/P-C4 描述性。

### 2.1 HEC-1 pooled 数值锚点（2026-09-04 收口后填入；不改变上述预测）

- HEC-1 判词：`HEC1_EVOLUTION_NOT_SUPPORTED`；`D_o>=0.115` 为 1/3 顺序，cohort 正向 2/4，harm 条件成立，P2 链为 0。
- Best-Safe-Global outcome-side 上界：14/23 单元存在安全 non-identity，累计 `+5.527089`。
- validation-search：17/26 单元存在 Support-safe 候选；16 个可评 non-identity 部署中 7 个在 +144 仍过四线，累计原始增益
  `+3.131950`、harm 事件 9。
- transfer audit：34 个 Support-safe 候选中 10 个保持评价面安全；16 个可评机会中 7 个所选候选稳定、1 个可由其他候选救回、
  8 个无候选保持安全。HEC-2 应把“future-safe retention”作为机制读数，与 pooled 的 `10/34` 并列；这不是新增成功门。
- 完整口径见 `docs/HEC1_ZERO_LLM_DIAGNOSTIC_CLOSURE_2026-09-04.md`。正式 HEC-2 baseline 的 24 候选顺序须在发车前冻结；
  HEC-1 的事后截断只作诊断。

## 3. A5 treatment funnel(sol 要求;HEC-1 与 HEC-2 读数均须报;post-hoc、只读既有记录、不改 runner)

| 环 | 数 | 来源字段 |
| --- | --- | --- |
| K0 | 存活 Skill 数 | `hec1_k0*.json` |
| Match | Scope 匹配的 Target 单元数 | 逐单元 `retrieved_skill_ids` / 解析集非空 |
| Supply | 真正进入候选池次数 | `resupplied_candidate_ids` / `cand_skill_*` |
| Selection | 被探测/选中次数 | probes 中来源为供给者 |
| Admission | 过 Support / delayed 次数 | 权威门记录 |
| Deployment | 真正部署次数 | `deployed_via ∈ {recalled_skill, resupplied_draft, searched_active_program}` |
| Re-encounter | 后续复用次数(同块 / 跨块分层) | `deployed_via` + 起始 Active 集 |
| Marginal gain | 相对对照臂的效用/成本改变 | 评价面配对差、LLM / fits |

任一环为 0 → 报告必须写"A5 未被实例化/未形成 treatment",**不得**写成"知识无效"。

## 4. 统计与判词

沿用 HEC-1:描述性;D_o ≥ 0.005 × N_T_eff_pc;≥2/3 顺序、≥3/4 cohort、harm online ≤ frozen;三顺序非 seed;
19/23 类门槛按 N_T_eff_pc 重算。词表同 HEC-1 加 `CONSUMER_SHAPES_HARM / NO_SHAPE_CHANGE`。

## 5. 冻结与发车条件

HEC-1 三顺序读数与判词入账 → 本稿按 D5/D6 结果**只填预测的数值锚点**(不改结构)→ sol 核 → 用户批预算 → Consumer 适配
diff + 非作者复核 → commit → Phase S-pc → 三顺序。**不得**在 HEC-1 与 HEC-2 之间插入新的方法设计。
