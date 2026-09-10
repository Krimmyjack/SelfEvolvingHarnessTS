# D5 + D6 任务书:2×2 因果分解 与 逐序列持续性审计(0 LLM;exposed development;不影响 HEC-1)

日期:2026-09-03。地位:sol 建议、主线起草;**只读工件、只写新文件**;结果只进 HEC-2 预注册与论文机制章,
**不回流 HEC-1**(合同冻结、runner 只读)。接收方:grok 4.6-xhigh。工作目录 = 仓根。先读项目 `AGENTS.md`
§6/§7/§8、`docs/HEC_EVOLUTION_MAINLINE_PLAN_2026-09-02.md` §10.11、`docs/D1_ROUTING_HARM_DIAGNOSTIC_2026-09-03.md`
与工件 `p4ab_routing_harm_diagnostic.json`(复用其窗口、S/E 集与归因字段)。

**硬约束**:0 LLM;Consumer fits **≤300**;只用已曝光 development 窗口;**不得读取** `FORWARD_SHAKEDOWN` 或任何
Phase-T 效用、任何 held-out;不改任何阈值;不改 `scoped_serving_evaluator.py`(交叉项在脚本内组合);一脚本一工件;
不 git 提交。
**运行隔离(sol 执行纪律,2026-09-03)**:HEC-1 十小时科学运行期间**不得修改主工作树**——本任务只能 (a) 在独立
`git worktree` 中运行,或 (b) 在 HEC-1 最终 commit 之后运行且**只写**隔离工件(`p4ad_*` / `p4ae_*`)与本任务的新脚本
文件,不触碰任何被跟踪文件。二者择一并在回报中注明采用了哪种。

---

## D5 · 2×2 因果分解(回答"算子还是路由")

**问题**:pooled Ridge 下进入 Scope 同时意味着 (1) context 被 Program 处理、(2) 该序列改用 program model。
Skill 的伤害与收益来自哪一个?

**材料**:D1 的 7 个主窗口(4 delayed + 3 重遇)+ 4 个 Support-A 探针窗口(origin 1896/2376/2616/2856,根 Scope)。
每窗口:Program P、Scope 解析集 S、served 20 条。

**四格**(逐序列 sMASE;CODE FACT:`scoped_evaluate` 内 raw/program 两套设计与 Scope 无关,`_serve(contexts, view,
x_train, y_train)` 可对任意 contexts 出预测;`serving_mode="train_only"` 即"program model × raw context"):

| 模型 | context | 含义 | 获取 |
| --- | --- | --- | --- |
| raw | raw | Static | `scoped_evaluate(compiled=None)` |
| raw | prepared | **context-only** 效应 | 脚本内:`_serve(prepared_contexts, raw_design)`(复用 D1 的 2-fit 复制路径) |
| program | raw | **model-route** 效应 | `serving_mode="train_only"` |
| program | prepared | 现行完整策略 | `scoped_evaluate(scope=S)` |

每窗口 **2 fits**(raw、program 设计各一),交叉项零额外 fit;11 窗口 ≈ 22 fits。

**逐序列量**(对 S 内序列为主;对全部 served 序列另算一遍作"若被划入"的反事实,单列标注):
`ctx_i = Static − (raw, prepared)`;`route_i = Static − (program, raw)`;`full_i = Static − (program, prepared)`;
`inter_i = full_i − ctx_i − route_i`。

**读数(预注册)**:
- 对 `full_i < −0.30`(严重)与 `< −0.005`(实质)的序列,取三分量中**最负者**为主因;报三分量各自占总严重伤害幅度的份额。
- 判词(占比 ≥ 60% 为主导):`ROUTE_DOMINANT` / `CONTEXT_DOMINANT` / `INTERACTION_DOMINANT` / `MIXED`。
- 新进入者 vs 持续成员分层(用 D1 的 attribution 字段)各报一遍。
- 可选(预算内):per-channel 四格作对照——按 D1 的 per-channel 定义各拟合,预期 route 项趋零;≤ 2×|S| fits/窗口。

**含义(预写,不得事后改)**:`ROUTE_DOMINANT` → HEC-2 ① per-channel 优先,未来合同考虑把"处理 context"与"切换模型"
拆成两个决策;`CONTEXT_DOMINANT` → 算子设计是主因,HEC-3 加 1–2 个缺口优先、位置感知算子;`INTERACTION_DOMINANT` →
单谓词双动作的 ScopeSpec 本身有问题;`MIXED` → 不据此改任何面。

## D6 · 逐序列持续性审计(回答"收益沿序列持续还是跨序列泛化")

**材料**:Source-v3(`p4w3b_…`)与 Source-v2(`p4w2_…`)全部 Support → delayed → 重遇读数;0 fit,只读工件。

**量**:
- P(本窗受益 | 上窗受益, 持续成员)、P(本窗受害 | 上窗受益, 持续成员)——持续成员翻号率;
- P(受益 | 新进入者)、P(受害 | 新进入者);
- 严重受害(< −0.30)中新进入者占比(D1 已有,复核并合并 v2);
- 离开者在离开前一窗的增益分布(H3:最大赢家是否因尖峰移出窗口而离开)。

**读数**:两两对比给点估计与 n;n 极小(3 个重遇)时只作描述,并**明确标注 HEC-1 的 H1/H2/H3 落账将以 n≫3 重测同一
组量**。不得写成"收益沿序列持续"之类结论——Source-v3 三窗口里持续成员稳定只在 1/3(2616 6/6 非负;1896 3/5 翻;
2376 1/1 翻)。

## 输出

`artifacts/main_protocol/p4ad_causal_decomposition.{json,md}`(D5)与 `p4ae_series_persistence.{json,md}`(D6);
字段含 `boundary{llm_calls:0, consumer_fits, held_out_reads:0, shakedown_utility_reads:0, thresholds_changed:0}`。

## 回报

(1) 11 窗口四格表与三分量份额;(2) 严重/实质伤害主因分布,新进入者 vs 持续分层;(3) 判词;(4) D6 六个条件概率与 n;
(5) 实际 fits;(6) 偏离与规格矛盾。
