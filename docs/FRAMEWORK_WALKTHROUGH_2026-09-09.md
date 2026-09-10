# 框架讲解：一条序列从进入到反馈写回（2026-09-09，Fable 按代码整理）

目的：让人不看聊天记录也能明白系统**实际**怎么处理、怎么反馈、知识怎么改。所有陈述以代码为准并给出位置；
与设计文档不一致处单独标出。行号对应 2026-09-09 的工作区。术语只在第一次出现时解释。

## 0. 一页图

```
一条序列 s，决策时点 origin（只见 values[:origin]）
  │
  ├─ ① 公开特征：21 个统计量，在整段可见历史上计算            public_features.py:265-331
  ├─ ② 知识视图：3 张 bootstrap 流程卡 + 谓词命中的学习卡(top_k=2)
  │      + candidate_policy(含 selection/proposal_guidance)     retrieval.py:241-338
  ├─ ③ Fast 三个 LLM 阶段：inspect → propose → select          fast_agent.py:946 / 1018 / 1302
  │      候选 = identity + ≤2 个程序；每个候选先经确定性验证    fast_agent.py:1199-1258
  ├─ ④ 交付
  │      开发期(在线回路)：按顺序用 Support 探测候选，首个准入者部署  online_loop.py:763-780, 884-953
  │      验收期/DEV-DEPLOY：Fast 选什么就交付什么，无 Support        run_dev_deploy1.py:816-853
  ├─ ⑤ 评分：K+1 管线——每个程序 P 一套"准备训练材料→训练 Ridge→服务"  per_sequence.py:22-48
  │      收益 = 基线损失 − 处理后损失 (sMASE)；origin 面 / +48 面
  ├─ ⑥ 反馈写回：每次合法探测立即写一条 Episode                  online_loop.py:862-883
  └─ ⑦ 批次边界：Slow 读证据卡，写一处修改，编译成子快照，父版不动  run_dev_deploy2.py:614-671
```

## 1. Fast 看见什么、能做什么

**输入。** `PreparationRequest{series_uid, values[:origin], task_spec, observed_pattern_spec, task_context}`
（`contracts/method.py:42-49`；截断见 `per_sequence.py:158-167`）。LLM 的 `public_input` 只含 `features`（21 个全历史特征）、
任务绑定、inspect 载荷、算子契约、候选列表（`fast_agent.py:950-954, 1022-1029, 1306-1312`）。

三处需要明确知道的限制：

- **工具不增加信息。** Fast 可调用的工具是 `summarize_series`、`localize_regions`（有探针面板时加 `read_fixed_probe_panel`），
  全部零参数，返回的就是提示里已有的特征（`public_tools.py:193-248`）。能比较历史窗口的 `compare_history_windows` 只在
  `stage=="observe"` 声明，而 Fast 没有 observe 阶段，实际不可达（`public_tools.py:396`）。
- **服务窗特征不进 LLM。** `observed_pattern_spec` 里按最近 192 点算的 `recent.*/change.*` 只用于 no-op 过滤和 signed 半径查询
  （`fast_agent.py:87-102, 860-871`），不在任何 `public_input` 中。四个 `*_probe_direction` 在 `TTHAMethod.prepare` 不传面板的
  正常入口下恒为 `"unknown"`（`method.py:371-377`；`public_tools.py:53-56`）。
- **动作空间。** 27 个 canonical 算子（`operators/registry.py:79-231`），预测任务合法 23 个；`actionable` 模式先用默认参数实跑
  `verify_candidate`，修改比例 > 0.35 的全局变换在进入提示前被剔除（`fast_agent.py:209-259`）。注册表中**没有任何算子声明
  `public_parameter_bindings`**，程序参数全为默认（`registry.py:53-67`）。候选池 `total_k = min(4, maximum_candidates=3)` 且
  identity 占一槽 → **identity + 最多 2 个程序**（`candidate_pool.py:41-69`；`task.py:679-681` 注释与实现不一致）。

**三个阶段。** inspect 输出 ≤2 条结构化假设与区域分数（schema `fast_inspect_v1`）；propose 输出 ≤3 个候选、每个 1–4 步
（`fast_propose_v1`）；每个候选经确定性验证（合法性、修改比例、非 intrinsic 算子不得越出 inspect 区域，`candidate_verification.py`）；
select 只在池中有程序候选时调用，输出 `chosen_candidate_id` 与 `verification_actions`（`fast_agent.py:1290-1325`）。
每阶段 1 次调用 + ≤8 轮工具 + 1 次 schema 重试；每序列 24 次上限来自 runner（`run_dev_seq4.py:101`；`runner.py:119-121`）。

**三种结束状态（读工件时必须分清）。** 池中无程序候选 → `chosen_id="identity"`、`ABSTAINED`（`fast_agent.py:1290-1299, 1377`）；
LLM 显式选 identity → 同为 `ABSTAINED` 但 `compilation_status="ok"`；协议异常（schema 两次不过、非法候选 id 等）→
`chosen_id` 保持初始的 `""`、返回 `FAILED`（`fast_agent.py:918, 1326-1352`）。**`""` 只会来自失败路径。**
DEV-DEPLOY 的包装层丢弃了 `PreparationResult`，把 `""` 记为 `EMPTY_SELECTION_FALLBACK_TO_IDENTITY` 并计 0 收益
（`run_dev_deploy1.py:816, 850-853`）——这是 DECISIONS.md "待裁定四" 的根源。

## 2. 交付：开发期与验收期是两套决策规则

**开发期在线回路**（`online_loop.py:run_online_round`）：`prepare()` 之后，探测顺序 = Fast 所选（非 identity）在前、其余候选按池序
（`:763-780`）；对每个候选 `executor.evaluate(steps, origin)` 得到 Support 收益与逐序列收益（`:834-837`）；
`classify_relation`：聚合 ≥ +0.005 且无逐序列 < −0.005 → POSITIVE，聚合正但有局部伤害 → CONFLICT（`experience_memory.py:401-441`）；
默认准入 `strict_positive_only`：admitted ⟺ POSITIVE；**第一个准入者成为 winner 并停止探测**（`:884-953`）。Fast 选 identity
不会清空候选池，identity 从不取得执行权（`:301-303`）。DEV-SEQ-3 实测：Fast 的选择只决定 53/116 的交付。

**验收期 / DEV-DEPLOY 冻结 Fast-only**：无 Support，`prepare()` 返回的选择直接交付（`run_dev_deploy1.py:850-870`）；
`AGENTS.md:148-157` 规定 held-out 零反馈、无 Slow、不改知识。**因此开发期的 Support 门是一个部署时不存在的兜底。**
第一包影子审计：同一历史下 Fast 原选择 0.0193 vs Support 交付 0.2175（origin 面，116 条）。

## 3. 评分与反馈的确切含义

- **K+1 管线**：每个不同程序 P 对应一套"用 P 准备训练 roster → 训练 Ridge → 用 P 准备服务窗 → 预测"；序列 s 选 P 就走 P 的管线，
  其他序列各走自己的（`per_sequence.py:22-48`）。训练 roster 与评价 roster 不相交；**但训练材料共享**，所以 s 的收益是"这套完整管线
  对 s 的效果"，不是"只改 s 几个点的效果"。R4D-A 测得训练/模型通道占绝对分量的 70–80%。
- **raw/identity 基线**已含 evaluator 的 `_linear_integrity` 基础填补，不是把 NaN 直接喂模型（`scoped_serving_evaluator.py:73`）。
- **收益** = 基线 sMASE − 处理后 sMASE；**origin 面** = 在 origin 提交程序、对 [origin, origin+48) 的预测评分；
  **+48 面** = 同一程序沿用到下一窗口。MATERIAL = 0.005 是效应量阈值，不是显著性阈值（`hec1_contract.py:590-596`）。
- **风险四线**：覆盖 ≥ 5（`min_treated`）、均值 ≥ 0.005、受损比例 ≤ 0.20、最坏单序列伤害 ≤ 0.30。
- **Episode**（`experience_memory.py:113-158`）：每次合法探测立即写一条，含 context_summary、workflow_signature、
  support/delayed 响应、relation。Episode **不属于快照**；Fast 只能在调用方显式传入时以渲染后的文本前缀读到，且渲染刻意不含
  收益数值（`fast_agent.py:804-914`；`signed_radius.py:377-378`）。DEV-DEPLOY 全部 `experience_episodes=()`。

## 4. 知识对象与 Slow 修订

**快照** = 内容寻址目录：`instruction.md`、`skills/bootstrap/{3 张}`、`skills/learned/*`、`memories.jsonl`、`retrieval.json`、
`candidate_policy.json`、`verification.json`（`store.py:106-156`）。两层 SHA：`harness_content_sha`（语义）与 `runtime_bundle_sha`
（语义 + 全部运行时依赖源码 SHA，`compiler.py:207-262, 995-1016`）。**改 `agent_backend.py` 等依赖文件会让所有快照的
`runtime_bundle_sha` 漂移**——这是第二包事故的机制。

**K0（当前父知识，SHA `98dea3b0…`）**只含一张学习卡：`fast_winner_forecast_ridge_smase_outlier_mad`，body 是冻结程序
`[{"op":"outlier_mad"}]`，检索谓词 `task_kind == forecast`（对所有预测序列命中），`serving_scope` 为 `local_robust_z_peak ≥ 3.0`
（执行端）。所以每条决策 Fast 都看到 MAD 供给卡并可直接取为候选（`fast_agent.py:326-371`）。

**检索**：谓词 AST 三值求值，恰为 True 才命中；数值特征可按分箱标签比较——`missing_fraction=="high"` ⟺ ≥ 0.20，
`local_robust_z_peak=="high"` ⟺ ≥ 6.0（默认边界 (0,1,3,6)）（`observables.py:48-65, 95-144`；`retrieval.py:48-96`）。
命中的 capability 卡按 leaf 数排序取 top_k=2。`candidate_policy` 的 `selection_guidance/proposal_guidance` 走 `controls`
无条件进入每条决策的系统提示 → 这类 PATCH 暴露必然 40/40；ADD 卡受谓词门控 → 第二包 g2/C 的卡 8/40。

**Slow 一次修订**（`run_dev_deploy2.py:614-671`）：证据卡（母表 120 行中性事实 + 效果字段 + 菜单聚合读数）+ 可写面目录 →
`slow_edit_v1` manifest（ADD 或 PATCH，恰一处）→ `edit_preflight` 白名单（禁冻结程序、禁供给权、禁 serving_scope、禁改 risk_guards/
top_k/槽数/verification）→ `apply_to_fork`：拷贝父目录、写入、重编译、以新 SHA 落盘，**父目录从不被写**
（`edit_controller.py:783-831`；`store.py:184-219`）→ 零成本暴露检查 → 有暴露才跑对照。只为格式/传输错误重试一次；
`confirmed_cause` 由所选面反推得出，**不是已证实的原因**（`dev_seq2_knowledge.py:586-600`）。

第二包三臂差别只在证据卡（`run_dev_deploy2.py:539-607`）：R/C 多两个效果字段与菜单聚合读数；C 多一段"按对照组织"的写法要求；
C-minus 无效果字段。任务书 §3.3 所说的"基率工具"在代码里**没有交互式实现**，且 `_render_menu` 删除了逐序列读数
（`:518-536`）——比较式修订并未拿到"同一序列上 P 与 Q"的对照材料（DECISIONS.md 待裁定五）。

## 5. 与设计文档不一致、或容易读错的地方

| 现象 | 位置 | 后果 |
|---|---|---|
| 包装层不检查 `PreparationResult.status`，失败记为 identity 0 分 | `run_dev_deploy1.py:816, 850-853` | 第二包 18 条、第一包 9 条应为 UNKNOWN |
| 服务窗特征不进 LLM；全历史特征进 LLM；Consumer 用 192 点 | `fast_agent.py:775-779, 860-871` | 观察窗 ≠ 作用窗 |
| 工具返回与提示重复的特征；历史比较工具不可达 | `public_tools.py:193-248, 396` | "inspect"无新信息 |
| 无算子声明参数绑定；池 = identity + 2 | `registry.py:53-67`；`candidate_pool.py:41-69` | 动作空间接近固定菜单 |
| identity 占槽，但 `task.py:679-681` 注释说不占 | — | 文档误导 |
| Slow 菜单材料删逐序列读数 | `run_dev_deploy2.py:518-536` | 比较式臂无比较材料 |
| 开发期 Support 门在验收期不存在 | `online_loop.py:884` vs `AGENTS.md:148` | 开发对象 ≠ 验收对象 |

## 6. 这些事实对"为什么效果不佳"的意义（一句话版）

系统让一个只能看 21 个全历史统计量、只能从默认参数菜单里挑一两个算子、看不到任何收益数字的 Fast，去做一个连专门的统计
学习器都尚未证明能在 20 序列 × 5 窗口规模上学会的条件化选择；然后让 Slow 从几十个带噪声、跨窗口不稳定、且缺少逐序列对照的
案例里，用一段文字修正它。两个诚实的实验包都显示：拿掉开发期的 Support 兜底后，这样的 Fast 不如一个用同样反馈校准过的固定程序。
（判断与后续取舍见 `DECISIONS.md` 与 `SLOW_FEEDBACK_UPDATE_DESIGN_NOTES_2026-09-08.md` §13–14。）
