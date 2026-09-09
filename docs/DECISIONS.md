# 当前决定与交接

只记录影响方向、运行授权和交接的重要决定，不复制报告或建设审批平台。
2026-09-08 由 Astra 随用户要求初始化；后续主维护者为 Fable，其他线提供回执，
避免同时改此页。维护者离线不阻断已授权任务执行。

事实依据是代码与工件；批准依据是用户原话/明确授权，不是任何助手的“建议”。
状态区分：提议、用户批准、执行者收到、运行中、完成/中断。文件出现不等于任务已送达。

| 日期 | 决定 | 提议/整合者 | 批准依据 | 当前状态/执行回执 |
| --- | --- | --- | --- | --- |
| 2026-09-08 | 第一包 DEV-DEPLOY-1：影子审计、冻结 Fast-only、真 0-LLM 基线；不让 Slow 写新卡、不碰密封；详见任务书 | Astra + Fable 讨论，Astra 整合 | 用户：“好的，先继续推进，发第一包，然后同步落实相关的文档” | 用户同意推进；任务书已落盘；**Fable 可行性检查 23:08 通过（附执行前提，见下）**；Opus 接收/启动回执待补 |
| 2026-09-08 | 协作：Astra 方法/任务书/证据整合，Fable 系统/可行性/反方与本日志，Opus 整包执行，用户决定重要边界 | 同上 | 同一本轮推进及文档同步指令 | 已同步 AGENTS §9.1；不新增逐项审批 |
| 2026-09-08 | 2026-09-15 路线检查点；第二包方向为“旧知识 vs 自主修订知识的冻结 Fast-only 价值” | Astra + Fable | 用户同意推进合并计划 | 日期/方向已记录；第二包具体数据、预算与发车仍待第一包结果后决定 |
| 2026-09-08 | 旧 DEV-SEQ-4 A/B 收口选择单独保留；本次新任务不执行旧进程命令、不扩旧包 | Astra 建议在原预算/健康恢复条件下完成既定收口 | 未将新任务指令解释成某条旧恢复命令已获确认 | 当前工件 PARTIAL_MORE_CELLS_TO_RUN；只完成 u12/knowledge-frozen；后续执行回执待补 |
| 2026-09-09 | 外部评审第一部分（A/B/C）收到：`docs/EXTERNAL_REVIEW_GPT6PRO_2026-09-09.md`（网页端 GPT-6 Pro，读 `9.8.zip`）。**属输入，不是裁定。** 发现一处需更正的算术错误（见下"待更正"）及 8 项内部核对事项（I-1…I-8） | 用户发起；Fable 起草简报（`docs/EXTERNAL_BRIEFS_GPT6PRO_2026-09-08.md`） | 用户："感觉可以让他也思考和进行相关的调研" | **Fable 已从工件复算 C-2、B-1、读数 7 三项，全部属实**；转 Astra 裁定：① 是否按下述更正三份文档；② §3.2 口径是否采用其分析协议；③ 文献表抽查。D–G 未完成 |
| 2026-09-08/09 | **Opus 接收/启动/完成回执，补上表 2026-09-08 行"Opus 待补"项**：DEV-DEPLOY-1 三项（影子审计 A、冻结 Fast-only 真实运行 B、0-LLM 基线 C）全部完成，见 `docs/DEV_DEPLOY1_FAST_ONLY_ALIGNMENT_RESULT_2026-09-08.md` | Opus 执行，直接向用户与 Fable 报事实 | 用户任务书 + 用户提供 `terra`（`api.nowaterapi.xyz`，已知 nowater/`gpt-5.6-sol` 端点换新 key）解除 Part B 阻塞 | **完成**。B 段发车前重算 K0 父快照 SHA 与历史 `knowledge-frozen` 逐字节一致，单条决策烟测复现历史 probe 增益逐位相同，一次 billed 预检确认 `returned_models==["gpt-5.6-sol"]`，全部通过后才动用正式预算。120/120 决策 0 故障、0 账户/权限边界；用量 573 LLM / 256 fits（A+B+C 合计）/ 墙钟约 100 分钟，均远低于上限。**未执行 Fable 23:08 前提里"发车前由 Fable 复测内存并按表定并发"这一步**——事前不知道该记录存在，直接用了任务书默认并发 2；实际 0 OOM、0 故障，但流程上是补跑而非按既定前提执行，如实记录不掩盖。已核 Fable 转来的 I-5/I-6：本包代码未调用 `per_sequence.UnitReadings.compose()`（有零收益/缩分母风险的那条路径），未读取 `dev_seq3_guidance_delivery__contrasts.json`，均不适用于本包结果，已用 grep 核实。归因口径已按 Fable 提示补记：F 与历史交付的差距是"无准入门+无同批 Episode 历史"的联合效应，不可整段记到准入门一项；单独隔离准入门的读数仍是影子审计的 0.0193 vs 0.2175。 |

| 2026-09-09 | 第二包 DEV-DEPLOY-2：F / R / C / C-minus 四种知识版本，一次形成边界、两组独立 Slow 更新，冻结 Fast-only 在 u12/u13 复核，D-new-fixed/safe 同信息参照；任务书 `docs/DEV_DEPLOY2_FEEDBACK_TO_FROZEN_FAST_TASK_2026-09-09.md`，设计取舍见备忘录 §14 | Astra 起草；Fable 事前可行性检查**未执行**（任务书与运行在同一时段完成，Fable 事后初核见下） | 用户转发任务书给 Opus 执行（原话未在本日志留存） | **Opus 报告完成**（`docs/DEV_DEPLOY2_FEEDBACK_TO_FROZEN_FAST_RESULT_2026-09-09.md`，13:04）：240/240 决策、六次 Slow（两组 C-minus 弃权）、C−F origin +0.047/+0.030，R−F −0.124/+0.047，**均低于 D-new-fixed 0.284**。中途 nowaterapi 账户耗尽，136 条经本地中转 `127.0.0.1:8318` 补跑，返回身份核为 `gpt-5.6-sol`。**Fable 初核：数字与逐决策工件逐项相符；三项需 Astra 裁定的问题见下节。** |

| 2026-09-09 | **Git：一次选择性提交**（代码 + 文档 + 精选工件；排除 `.dev_*_runs/`、`_scratch/`、压缩包、数据），作为 DEV-DEPLOY-1/2 收口锚点 | Fable 建议（自 9/8 起） | 用户："我给 astra 授权了一次 Git 提交" | **执行者 Astra**；提交范围与 commit hash 回执待补。提交后请把 `.gitignore` 对运行目录的处理一并披露 |

## DEV-DEPLOY-2 Fable 初核（2026-09-09；结论数字属实，附三项待裁定）

**已核实。** 12 份 `dev_deploy2_fastonly__*.json` 共 240 行，`faults` 全空、origin 收益无 UNKNOWN、`returned_models` 全为
`gpt-5.6-sol`；逐单元均值与报告主表逐项一致（g1 F 0.272569/0.005305、C 0.386798/−0.014328、R 0.014116/0.015164；
g2 F 0.114078/−0.003518、C 0.126599/0.044882、R 0.190311/0.014909）。`methods/ttha/agent_backend.py` 与 HEAD 逐字节一致，
无新增核心文件改动。`_corrupted_2026-09-09_account_exhaustion/` 含 14 份备份，与"7 批次 × 检查点+输出"相符；带
`..._of_scored_subset_diagnostic_only` 字段的正是补跑批次（g1/C/u13 与 g2 全部），与报告叙述一致。

**待裁定一：中转与 F 臂空选择的混淆。** 补跑范围 = g1/C/u13 第 5–20 条 + g2 全部，即 g2 整组在中转 B 上运行。
按单元拆开：g1 的 C−F = +0.047 **全部来自 u12（+0.114，两臂均在中转 A，干净）**，换中转的 u13 上 C−F = −0.020；
g2 的 C−F 为 u12 +0.013、u13 +0.048。但**同一个 F 臂（父知识不变）在 g1 空选择回退 2/40，在 g2 为 13/40**
（`chosen_candidate_id=null`、`candidate_programs=[]`、`llm_requests_sent=3`、`faults=[]`），而 g2 的 R、C 分别为 1/40、0/40。
这 13 条按 identity 记 0，直接压低 g2/F，抬高 g2 的 C−F 与 R−F。原始 LLM 响应未见留存，无法 0-LLM 判定是"模型未提候选"
还是"响应解析失败"。建议：① 报告披露中转分布与 g2/F 空选择率；② "C 在两组都跑赢 F"改写为"g1 由 u12 支撑且干净，
g2 受 F 臂空选择率影响，方向一致但第二次重复不干净"；③ 若无法判定原因，按任务书 §4.2 "故障、未完成和未知结果不能默认为
raw"的精神，考虑将 g2/F 主均值降为带注的诊断值。**"均不敌 D-new-fixed 0.284"不受此影响**，在两组、两单元下都成立。

**待裁定二：已收口包的 runner 被改动。** `run_dev_deploy1.py` 修改时间 09-09 12:17，第一包报告为 09-09 01:42；
报告自述把 `_fault_kind` 补丁与 `allow_loopback_http_relay` 猴子补丁加在了该文件。任务书 §6 写明"禁止顺手改旧包"。
因工作区无提交，无法 diff 出第一包收口时的版本——**这正是 9/5 以来未提交的直接代价：现在无法证明跑第一包的代码是什么。**
建议：Opus 把两处新增函数移入 `run_dev_deploy2.py` 或独立新模块并在报告披露；用户授权一次选择性提交作为两包收口锚点。

**待裁定三：状态页未更新。** `docs/STATE_ONE_PAGE_2026-09-03.md` 最后修改 09-08 22:33，两个包的结果都不在置顶；
两份任务书均要求"更新状态页"。`AGENTS.md` 已于 13:05 追加。请 Astra 决定由谁补写。

**给后续包的接线要求（不改本包结果）：** 逐决策工件增加 `transport`/`base_url` 与快照 SHA 字段；保留原始 LLM 响应或其
哈希，使"空选择"可事后分类。

## 待更正（外部评审 C-2，Fable 复算确认，2026-09-09；由 Astra 裁定后由文档所有者修改）

`artifacts/main_protocol/dev_seq3_guidance_delivery__g2.json` 重算：57 对未暴露配对中 46 对为零、**9 对** |Δ| ≥ MATERIAL
（`contrasts.json` 自报 identical 46 / materially_different 9）。剔除 6 对"渲染文本不同"者（5 对为零，唯一非零对
u10/T150 Δ = −0.003990，未达阈值）后，51 对应为 41 零、**9 达 MATERIAL**，而非 6。均值 −0.0093 与"41 对相同"均正确，
只有该计数有误。同一错误出现在三处，需一并更正并保留原文：
`docs/DEV_SEQ3_GUIDANCE_DELIVERY_RESULT_2026-09-08.md:333`、`docs/DEV_SEQ4_GUIDANCE_ADOPTION_RESULT_2026-09-08.md:30`、
`docs/STATE_ONE_PAGE_2026-09-03.md:71`。昨日 Astra 与 Fable 讨论中引用的"6/51"随之作废；简报 `EXTERNAL_BRIEFS` 读数
3/5/7/10 已由 Fable 更正。

另两项与第一包执行直接相关、建议并入 Opus 烟测（评审 I-5、I-6，Fable 未复算）：`per_sequence.py:241–274` 存在"缺
assignment → 零收益 / UnitFault 后对可读子集求均值"的路径，与任务书 §4"缺失记 UNKNOWN、不缩分母"冲突，包装层须拦截；
`contrasts.json` 的 `outcomes.independence` 报 distinct_series=10 / distinct_windows=34，与 20 UID × 3 origin 的实际配对
不符，疑为 formation 口径串入，不得用作独立样本数。

## DEV-DEPLOY-1 执行前提（Fable 可行性检查，2026-09-08 23:08）

输入核对：`artifacts/main_protocol/dev_seq3_guidance_delivery__g2.json`、`_scratch/dev_seq3/pkg3/g2/`
（两臂 validation + followup u10/u11 检查点，共 120 条决策）、`_scratch/dev_seq2/pkg2/g2/`
（形成段 u7/u8 与 boundary）均在盘上。B 段父快照 `98dea3b0…` 可从
`.dev_seq3_runs/pkg3/g2/store/` 恢复；该 SHA 与 `.hec1_runs/*/k0_store/` 中的 K0 快照相同，
即“g2 A 臂父知识”只含 K0 供给卡与 bootstrap skills，不含 DEV-SEQ-2 形成段新知识——请在 plan
字段如实登记，勿读成“带形成段知识的快照”。

内存（唯一阻塞）：检查时宿主空闲 0.49 GB / 15.7 GB，提交量 38.5/53.7 GB；单个 runner 进程常驻约
450 MB，当前不可启动。并发规则（按任务书 §5 由 Fable 定，只看内存与端点，不看效用）：
空闲 < 1.5 GB 不发车；1.5–3 GB 并发 1；3–5 GB 并发 2；≥ 5 GB 并发 4。发车前由 Fable 复测一次
并把实际值写进 runner 配置/命令行。

墙钟与预算：DEV-SEQ-4 实测 20 条决策/格约 18 分钟（并发 2–3），B 段 120 条决策估 2–2.5 小时；
并发 1 时约翻倍，6 小时上限仍可覆盖核心三项，但 §3.1 菜单 oracle 可能被挤掉——按任务书“诊断不足
不阻断核心”处理。fits 估算：C 段校准（菜单 × u7/u8 × 20 条）+ 三个测试单元 + A 段补评 + §3.1
≤ 500，合计约 1000–1300，在 2000 内但余量不大；建议 Opus 在 C 段校准完成后先报一次 fits 用量再进
§3.1。

与旧 DEV-SEQ-4 的关系：其剩余三格与本包不得同时运行（内存）；若用户选 B 续跑，须先于或后于
本包串行，且命令为 `--positions 12 13`（每臂一次），不是单独 `--positions 13`。

Git：工作区自 9/5 起未提交（193 个未跟踪文件 + 核心代码 1112 行改动），本包会继续增加；建议用户
授权一次**选择性**提交（代码 + 文档 + 精选工件，排除 `.dev_*_runs/`、`_scratch/`、压缩包），Fable
不自行提交。

读数提醒（交 Astra 定口径）：B 段 Fast 相对 SEQ-3 历史同时去掉了 Support 兜底和 Episode 历史两项，
F 与历史交付的差不能只归因于前者；A 段影子审计才是单独隔离“兜底”的读数。

当前任务：DEV-DEPLOY-1、DEV-DEPLOY-2 均已完成（见上表）；下一步为 **2026-09-15 路线检查点**，尚无新任务书。
旧 DEV-SEQ-4 剩余三格的 A/B 仍未决。
当前状态：[状态页](STATE_ONE_PAGE_2026-09-03.md)。讨论保存于
[Slow 设计备忘录 §13](SLOW_FEEDBACK_UPDATE_DESIGN_NOTES_2026-09-08.md#13-2026-09-08-合并取舍先对齐部署对象再检验知识修订)。

Git：本次未提交，未批准批量提交/清理既有脏工作区。没有运行、结果或后续阶段被本文
自动标为完成。执行者接收后给出任务路径和运行标识，由维护者补状态即可，无需再贴长文。
