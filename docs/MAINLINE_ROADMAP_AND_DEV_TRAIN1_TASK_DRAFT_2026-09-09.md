# 主线路线图与 DEV-TRAIN-1 任务书

2026-09-11收口：TRAIN-2四臂完整，旧/新/形成段固定的均值收益为−0.164690/0/+0.849895。
新卡确实改变执行，但全部identity，只恢复Static；报告`DEV_TRAIN2_RESULT_2026-09-11.md`。
当前只交Flash执行`FLASH_CHECKPOINT_COMMIT_TASK_2026-09-11.md`，保存并commit已有成果。
`DEV_TRAIN3_GENERAL_GUIDANCE_TASK_2026-09-11.md`仅为未批准讨论草案，后续安排仍与用户讨论，
不得发车。“先取得高于Static的准备价值，再研究超固定/积累”仍为建议，不是已批准实验。
以下“正在运行/未完成”均是对应日期的历史记录。

2026-09-10 22:30更新：DEV-TRAIN-2的Slow已生成候选；修复边界序列化失败后按原候选/原材料恢复，
新旧两臂已开始并行，尚无效果结论。没有重问Slow，形成段两条失败仍UNKNOWN；详见状态页置顶及任务书恢复回执。

2026-09-10 21:33更新：用户批准的下一包DEV-TRAIN-2已发车，10/10、四臂共享cap=1.0，
baseline CPU与Fast重叠、两Skill臂共用4路Fast、fit串行。规格及实际启动回执见
`docs/DEV_TRAIN2_UNCAPPED_WORKFLOW_SKILL_TASK_2026-09-10.md`；下文“不启动新实验”属更早实施记录，已被本次批准及发车更新。

## 最新增量（2026-09-10；下方既有规格与原草案保留）

DEV-TRAIN-1/1B 已有完成结果，见 `docs/DEV_TRAIN1_RESULT_2026-09-09.md` 与
`docs/DEV_TRAIN1B_RESULT_2026-09-10.md`；下文“尚无结果”是原发车时状态，不是现在。
用户现已批准新 Forecast TRAIN 主线取消通用 35% 修改比例硬门（AGENTS §5.7.2）。
代码实施由 Grok 承担、根 Agent 复核；本次不启动新效果实验、不改历史收据。
下一小型对照须让固定基线、旧 Skill、新 Skill 共享取消后的动作权限；保留 Static。
先区分动作空间扩展收益与 Skill 修订收益，不给 Agent 强制提供已知 FFT 答案。
按用户近期缩小开发规模的要求优先沿用 10 train/10 eval 与四路批内并发；完整
20/20 仍是原首包规格，不把不同人口读数相减。其他设计讨论与具体待测问题见
`docs/SLOW_FEEDBACK_UPDATE_DESIGN_NOTES_2026-09-08.md` §18；它不是新包启动回执。

## 执行定稿（2026-09-09，Astra；下方原草案保留为讨论记录）

授权依据：用户确认 20/20、实验模型与入口，并在整窗口主臂及输入侧对照的讨论后回复
“行 那你先这样推进吧”。本节为当前执行规格；下方原草案的五臂、10/10、per-channel
副表、预算估算与待定项不再用于执行。长期边界以 AGENTS.md §5.7 为准。
状态：**方案批准，开始实施；尚无 DEV-TRAIN-1 效果结果**。不代写 Fable 的决策日志。

### A. 首包要回答什么

先恢复用户指定的组织方式，再测一次局部知识修订：每条训练序列由 Fast 生成 Workflow，
汇总准备后的窗口拟合一个共享模型；预测输入也逐条准备。Slow 更新的仍是结构化
General/Specific Skill 或既有观察、构造、选择指导，不是给整组广播固定答案。

四臂：Static、形成段校准的固定 Workflow、旧 Skill 冻结、Slow 一次修订后的 Skill。
共同初始知识使用 `methods/ttha/harness/h0` 的既有 General bootstrap 与指导策略，
保持字节原样；不调用 DEPLOY 的 `parent_snapshot_and_machinery()` 引入历史K0冻结MAD卡。
h0的learned为空是本地一次更新的起点，不删除General/Specific架构；Specific可由Slow
自主形成。本包无程序卡自动供给权、无Runner直供历史赢家，固定Workflow另设基线。
本包不同时建 Source/A5、不换 Consumer、不增加样本权重算子。后续积累仍是项目目标，
不因本包失败删除；先根据失败发生在观察、动作、更新还是使用来安排下一步。

### B. 已定的数据与权限

- 使用既有含缺失 KDD development 身份及冻结 roster，20 train / 20 eval；计分始终全体
  20 eval，不删除故障或弃权对象。训练 anchors 沿用配置；192 输入、48 目标。
- 主臂允许 W 作用于历史训练窗口 [anchor−192, anchor+48)，包括训练输入 X 与目标 y。
  这些训练目标必须在知识形成时已观测。评价真值始终 raw，不改值、不借其做决策。
- 训练 Fast 只读本序列固定训练材料的合法前缀，截止 max(anchor)+48；不得把后续不断
  增长的前缀冒充“固定训练材料”。预测 Fast 只读自己的 origin 前输入及合法工具观察。
- Consumer 结构描述向两种角色和 Slow 提供：pooled linear Ridge、192→48、alpha=1、
  不惩罚截距、逐窗口中心/尺度、平方训练损失、sMASE 评价。训练损失与评价指标分列。
- 使用现有请求/知识接口和实验局部适配器，不为字段新增通用 Schema 或核心哈希依赖。
  若观察工具确实缺少某项必要的当前数据观测，可补最小只读接口，所有臂同等获得，
  在真实调用前冻结；不按观察结果人工指定正确 Workflow。
- 实验 API：model=cpa-grok-4.6，base_url=http://127.0.0.1:8318/v1。复用现有密钥环境，
  不打印密钥，不静默换模型。仅此明确 loopback 地址可用 HTTP；不放宽远程 URL 安全规则。
  Grok CLI 执行代理与此实验 API 是两个配置层，不把 API alias 当作 CLI model 强制替换。

### C. 训练目标对照（严格区分两件事）

**真正输入侧对照：** W 仅接收 X，不得接收 y；目标固定为共同 baseline integrity 的 y0。
不能先 W([X,y]) 再丢弃输出 y，就声称“输入处理未见目标”。预测侧 V、评价人口与拟合
配置在同一批 Workflow 的训练侧反事实比较中固定，不重问 LLM。

**输出分量分解：** 对同一个窗口，B([X,y])=(X0,y0)，W([X,y])=(Xp,yp)，比较
(X0,y0)、(Xp,y0)、(X0,yp)、(Xp,yp)。它测保留哪些已处理分量的效果，允许交互；
不等于独立处理 X/y 的因果比例。B 的 X0 也可能通过插补使用历史 y，须明确记录。
identity 的真实输入侧对照如使用 B(X)，也不声称等于整窗口 B 的 X0。

分解限于形成段合法反馈；不读取后续评价结果再写 General Skill。选择一个已冻结的
合法单步菜单（含 identity），0 新增项目 LLM；真实 fit 均计费。先按冻结材料去重复用，
不预写“必为39 fits”。数值事实可交 Slow，不能由研究者改写成系统学到的结论。
分解用于解释，不要求先发现正收益才能发车。

### D. 时间线与四臂

1. 形成段 u9/u10/u11：旧 Skill 固定，训练侧每序列一份 W，拟合一个共享模型；各窗口
   预测侧逐条生成 V。形成段反馈、执行事实、成功/失败分组与合法配对比较交 Slow。
   不重复把同一测量计为独立证据，不把未测程序记作失败或零。
2. 一个批次边界：Slow 最多产出一处自主修改；标签辅助归因，不预设 ADD、PATCH、
   弃权或必须覆盖多数失败。保留成功对照、事实/假说区别及预期行为。原始 Episode
   不进入 Fast。无合法修改如实记 NO_UPDATE_TREATMENT，不反复催出候选。
3. 分叉后：旧/新 Skill 分别为同一训练材料生成 W，分别汇总拟合共享模型。相同模型、
   工具、材料、资源口径；旧臂不能使用新臂生成的训练赋值，反之亦然。随后冻结各臂
   Skill、W 和模型参数；不把更新后 Skill 配上未声明的旧知识训练结果。
4. 后续 development u12/u13：每条预测输入由冻结 Fast 独立生成 V；无当前 Support、
   无效果兜底、无 Slow、无重新训练。全部决定落盘后外部评分。两窗口都已属于历史曝光
   development，称后续开发检查，不称密封/全新独立泛化。
5. Static 全 identity；固定 Workflow 在形成段按全人群效用校准，同报风险资格，不
   事后强制用已知赢家。固定训练/预测用法须一致并冻结；单步菜单结果只称“固定单步
   基线”，不称全部合法组合的最强上界。允许 Fast 在既有 1–4 步 DSL 内自主生成。

形成训练决策20次、预测60次；最终旧/新各训练20次、预测40次，计划约200次 Fast
决策（非 API 调用数），Slow 一次修改机会；重试和烟测另列实际开销。输入侧反事实
不重调 LLM。包 fit 上限沿草案500；不得用旧170估算作实测。调用/墙钟停止配置在 runner
预检中列明并继承现有包级约束，不设每序列小帽静默丢人口。先完成接线，再核实成本发车。

### E. 实施与必要验收

普通实施主责 Grok；根 Agent 复核训练/冻结/评分差集，确有难点再交 Opus-5。不双人
重复整包；不改旧 runner、核心协议哈希、风险线或历史工件。

- 复用 _evaluate_assignment 的训练窗口汇总与数值算法，但拆出 fit 和 predict，保存
  模型系数等实际状态。旧函数直接 fit+predict 并不等于已有冻结部署接口。
- 不能按程序模型取行。每臂一个模型来自全部异质训练材料；anchors 相同仅说明原始
  材料相同，不保证不同 W 产生的设计矩阵相同。无需新增签名/哈希缓存平台。
- 非恒等且实际改变设计矩阵的夹具用于验证干预；不要求任何程序改动必然改变模型。
- 同窗同序列逐位检查准备结果；全 identity 复现主臂 Static；预测模型落盘/恢复后
  无 fit 且预测一致；真输入侧 W(X) 无 y 访问；共享模型变化时单个 eval identity 不记
  固定收益0。未知、失败、缺测保持 UNKNOWN，全人口不全则总体均值不可读。
- 信息墙检查真实字段和数据流：Fast 无 Outcome；Slow 可读已到达形成反馈，不能读
  u12/u13 评分。不要用“某个数字恰未出现在文本中”代替泄漏测试。
- 可选留一重拟合固定其他训练 W 与预测 V，只报告条件差分、来源和真实 fits；不把
  它叫唯一原因或可加总贡献。不要求先穷尽20个留一才能进入 Slow。
- 手写异质 Workflow 仅用于上述接线夹具，不依据已知收益手写“有效 Skill”充作学习。

### E.1 批内并发（用户追加要求）

Fast 默认4路会话并发：同一训练批或预测批中，各序列独立请求、独立会话，读取同一份
冻结 Skill 和公共配置。不在一个 Fast 会话内部打乱依赖工具调用。全部训练准备完成
后才汇总拟合；形成段全部必要反馈合法到达后才调用 Slow；分叉和知识更新仍在边界。
两臂使用相同并发上限和故障处理，实际排队与完成顺序不决定历史、候选名或知识内容。

Consumer 拟合默认单队列；共享缓存写入、checkpoint 与预算增量由单写者或锁保护，
按稳定决策键合并结果，不用完成顺序分配身份。每项完成即时落盘，恢复只补缺失，
不重问已完成的 Slow、不覆盖既有收据。不得由并发任务读取同批其他任务的 Outcome。

内存不足或端点限流可将并发降至2路，记录调整与吞吐；账户故障停线，不默换模型，
不得因降速删人口或把未完成项记identity。模型调用重试和实际fits照实累加。
必要验收用确定性fake transport检查无串线、无丢决策、无重复fit和记账；真实LLM
串并行不要求输出逐位相同。首个真实批次报告秒/决策、峰值并发与故障数，不承诺线性加速。

### F. 读数与下一步

主读数为后续全人口 origin 效用，新−旧、各臂−Static/固定基线；同时报受损次数、最坏
伤害、既有风险资格、处理量和完整成本。保存生成/呈现/观察/程序/交付变化链，但不以
文本、版本或检索变化替代收益。重复 UID、共享模型和邻近窗口不当独立样本。

- 新知识优于旧知识但不敌固定：有局部更新线索，条件化增量仍未建立。
- 质量相近、伤害明显少：如实报告风险收益取舍；不把大幅降质量换来的保守当成功。
- 没有改变行为：检查知识传递及指导是否可执行，不立刻加卡。
- 行为改变而效果不好：分解训练/预测侧及 X/y，检查观察是否对准被处理材料；决定
  下一包改观察、动作还是更新方式，不沿同一卡重复跑。
- 有效局部修订之后再测 Source 积累/保留与跨情境；未有效也保留有依据的替代路线，
  不用首包否定整个 A5。9/15 做路线检查，不给成功概率或保证。

交付：一个主 runner package、一个必要 smoke、一个主报告及实际工件；Fable 接收
“批准→执行收到→结果”回执后维护 DECISIONS/STATE。当前不自动提交 Git。

---

## 原草案存档（以下不是当前执行指令，冲突项已由上方定稿取代）

日期：2026-09-09。起草：根 Agent（Cursor 助手），依据用户本日确认与两份只读代码审计。
状态：**提议，待 Astra 审核、用户批准**。本文件不是发车授权，不代表执行者已收到；
不改动任何历史工件、运行或判词。长期协议以 `AGENTS.md` 为准，本文件与 §5.7 一致，
与 §5.2/§5.3 冲突处以 §5.7 为准。`DECISIONS.md` 仍由 Fable 单人登记，本文件不写入。

---

## 0. 主线一句话与本文件边界

**研究目标（用户确认）：** 一个在零反馈部署下有正向效果、以 Skill 为中心的 agentic
时序数据准备架构。"数据价值"指**作为下游模型训练数据的质量**；Task 与预计训练的
下游模型以**结构性描述**在处理前作为 Context 给出；Fast **逐条**为训练序列生成
Workflow，Runtime 汇总异质准备结果、拟合**一个**共享 Consumer；预测输入也逐条准备、
由同一模型预测；Slow 在合法反馈到达的批次边界修订 Skill（`AGENTS.md` §5.7）。

本文件三部分：A（§1–2）为什么这样排、实施规格必须写死什么；B（§3）第一包
DEV-TRAIN-1 任务书；C（§4–5）之后的 TRAIN-2、论文级主实验、贡献与未决点。

边界：0 新增 SHA；每包一个 runner、一个 smoke、一个报告；不建 Schema/记账平台；
不改旧 runner；不开密封数据；Runner 不按模型名映射 Workflow；Fast 不读当前 Outcome。

---

## 1. 证据基础：为什么下一包是"训练侧逐序列 + 共享模型"

以下每条只写结论与出处，数字不重算。

| 事实 | 出处 | 对路线的含义 |
| --- | --- | --- |
| 数据准备效应 73% 走训练侧（程序改训练语料→模型变），25% 走服务 context；per-channel 下训练侧仍占 73–79% | D5 2×2 分解；`R4D_B_PERCHANNEL_RESULT_2026-09-08.md` §3.1 | 价值主体在训练数据，与用户定义一致；观察对象必须是被准备的训练序列 |
| 同数据同任务，pooled Ridge 下准备值 +0.23–0.27，per-channel 下 +0.02–0.12 且严重伤害几乎消失 | R4D-B §3.5 | 训练数据质量是 Consumer 相对的（M0 已有数）；主实验选 pooled；Consumer 结构必须进 Context |
| 序列静态特征对"路由决策"无可识别正上限；相邻窗口逐序列增益相关≈0 | `_scratch/headroom_probe/`；R4E；R4D-B §3.4；R4A | 只约束旧的"eval 序列选程序→路由"设置（§5.7 明文）；不验收也不否定新组织方式 |
| held-in Support 门把交付从 0.02 拉到 0.21；固定程序跨窗口不稳（留一选 winsorize −0.16；IQR u12 0.556 / u13 0.013） | DEPLOY-1 影子审计；headroom probe §B/§D | 只靠几个 held-in 窗口选方案会过拟合；这是积累知识应起作用的位置 |
| A5 方向性正证据：Frep 重放 held-out A5 +0.059（伤 1）vs A3 −0.217（伤 4），A5 首正成本 −31.7% | `STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md` §869–871 | development 级、单次配对；机制读法是"Source 先验让 held-in 选择泛化"。**引用前须按 treatment funnel 复核** |
| `run_v1_monash_fresh_a5_a3.py:410–416` 中 `src_eps` 从未传入 `_run_round`，A5≡A3 | 根 Agent 逐行核实；两份审计一致 | 该 runner 的 A5/A3 读数不得作为证据 |
| 代码现状：观察对象是 eval 序列（`per_sequence.py:83–93,148–167`）；程序作用于全部 train 窗口，`scope` 只路由 eval 预测行（`scoped_serving_evaluator.py:128–162,204–212`）；K+1 模型按 eval UID 取行（`per_sequence.py:39–48`）；Consumer 仅 ID（`contracts/task.py:112–119`；`fast_agent.py:638–647`）；dev 线 Slow 从未收到 `task_context`（`run_dev_seq1.py:608`、`run_dev_seq2.py:892`、`run_dev_deploy2.py:691`）；无"一次拟合、后续只预测"路径；Best-Fixed 仅单步（`run_dev_deploy1.py:501–518`） | Grok 审计 `_scratch/grok/runs/code_audit_new_framing_20260909/report.md`；Cursor 子代理审计（同日） | §3.1 的接线清单 |
| **`_evaluate_assignment` 已实现"逐训练序列赋值→一个共享 Ridge"**，含 pooled / per_channel 两变体，服务侧用 raw context | `evaluation/functional/run_batch_composition_headroom.py:242–307`（根 Agent 已核） | 评价器从"新写"降为"接线 + 补预测侧逐条准备" |

---

## 2. 目标几何与实施规格（回应 `AGENTS.md` §5.7 与备忘录 §17.4）

### 2.1 一轮的流程

```text
冻结 General / Specific Skill（同批不变）
→ 对每条 train 序列 t：Fast 读 t 的合法历史 + Consumer 结构 + Skill → Workflow W_t（可 identity）
→ Runtime：用 W_t 准备 t 的全部训练窗口 → 汇总所有 train 序列 → 拟合一个共享 Consumer
→ 对每条 eval 序列 e：Fast 读 e 的 [origin−192, origin) + 同一 Skill → 预测输入 Workflow V_e
→ 同一共享 Consumer 预测 → 外部评分（真值始终 raw）
→ 合法反馈到达：Slow 读训练批次卡 + 预测侧决策 + 逐序列证据 → 一处有限 Skill 修订
→ 下一轮；预算或轮数到 → 冻结 {W_t}、共享模型、Skill → held-out 只做预测侧逐条准备与预测
```

### 2.2 必须在实施规格里写死的项

**（1）训练材料与角色。** 训练材料 = roster 中 role=train 的序列在冻结 anchors
`[312…852]` 上的窗口 `[anchor−192, anchor+48)`。Fast 在训练阶段被明确告知"你在为训练
序列 t 准备训练材料"，可见范围为 t 的 `values[:origin]`（含其全部训练窗口）；不可见：
任何 eval 序列的真值、当前窗口 Outcome、其他序列的 raw Episode。

**（2）训练目标是否可修改（待 Astra 定）。** 现有 `_apply_program(window)` 对整段窗口
（context + 48 步 target）施加程序。两种选项：(a) 沿用整窗口——训练目标中的尖峰/缺口
同样是训练数据质量的一部分；(b) 只改 context、target 保持 raw——避免模型学习"被平滑的
目标"造成部署偏差。**建议：主臂沿用 (a)，另以 0-LLM 对照读数记录 (b)**；不在主臂中途切换。

**（3）预测输入准备。** eval 序列 e 的预测输入 `[origin−192, origin)` 由 Fast 逐条生成
V_e 准备（因果，只用 origin 前），同一共享模型预测；真值 raw。训练/预测变换兼容性：
预测侧允许算子集 ⊆ 训练侧允许算子集，且禁止 shape-changing；Fast 在预测阶段被告知
"模型已由准备后的训练材料拟合"，并附训练侧汇总事实（各 Workflow 计数、改动比例），
不附任何 Outcome。预测侧不得重演 Support 按当格效果选策。

**（4）Consumer 结构描述（用户已同意）。** 冻结字段：`family=linear_ridge`、
`shared_across_series=true|false`、`input_length=192`、`horizon=48`、
`regularization=ridge alpha=1, unpenalized intercept`、`training_windows=N_train×anchors`、
`normalization=per-window center/scale`、`loss=sMASE`。文本种子可复用
`run_e2_recipe_experience_to_skill.py:881–891`。进入 Fast 与 Slow 的 `public_input`；
Runner 禁止按 ID 映射；顺带修正 dev 线 `"ridge"/"fixed:m0"` 与实际 `pooled_ridge_a1` 的不一致。

**（5）"一个共享模型"。** 同一臂同一轮只拟合一个 Consumer。anchors 冻结意味着任何
origin ≥ 900 的训练设计矩阵相同，故 held-in 多轮里训练材料不变、只有反馈与 Skill 变，
冻结后一次拟合服务全部后续窗口。缓存键 = 整批训练赋值签名；**不得按程序取行拼接**。

**（6）归因仪器（可选、计费）。** 留一重拟合：对训练序列 t，把 W_t 换回 identity、其余
不动，重拟合一次，总收益与 eval 伤害分布的变化即 t 的边际贡献 Δ_t。每轮多 N_train 次
fit，按 R4D-B 成本（240 fit ≈ 100 s）可承受。Δ_t 进入 t 的 Episode 作为局部对照；
不预设拥有全部反事实，Δ_t 之和不要求等于总收益。

**（7）知识与编辑面。** General = bootstrap 观察/构造/选择原则（面对此 Consumer 该查
训练材料里的什么、怎样构造带范围/参数/顺序的 Workflow）；Specific = 情境化条款。
Slow 六个编辑面模板复用（`dev_seq2_knowledge.py:479–583`）。Slow 输入必须含
`task_context` + Consumer 结构 + 训练批次卡（逐训练序列 Episode、Δ_t、总收益、eval 伤害
分布、预测侧决策）。不做 C-minus。

**（8）held-in / 冻结 / held-out。** held-in 轮 = u9/u10/u11（评价窗口给反馈；训练材料
同一份）；冻结 = {W_t} + 共享模型 + Skill 版本；held-out = u12/u13：预测侧仍由冻结 Fast
逐条准备（零反馈、无 Slow、无 Skill 变更），同一冻结模型预测。held-out 中 Fast 仍会被
调用，但只在预测侧、只读部署可见输入。

**（9）读数纪律。** 逐 eval 序列损失照常报告，但**不是**某条训练序列处理的独立因果效应
（§5.7）。故障/空选择/FAILED 记 UNKNOWN，不缩分母；identity 全赋值必须与 Static 逐位相等。

---

## 3. DEV-TRAIN-1 任务书：对齐组织方式，并在小规模上取得可归因的正向读数

任务书负责人：Astra 定稿；执行主责：Opus / Kimi（一人主责）；Grok 可承担 0-LLM 夹具
与工件核对。整包派工、连续执行；普通实现问题自行解决，不为一次负数或弃权停工。

### 3.0 唯一主问题

> 在 §5.7 的组织方式下，Skill 指导的逐训练序列准备 + 逐条预测输入准备，冻结后在
> 零反馈 held-out 上，能否**不低于**同一 held-in 反馈校准的最佳单程序且伤害更少；
> 以及 Slow 的有限修订相对初始 Skill 是否带来增量。

本包同时验收五处接线是否真的对齐；任一环不通，就是找到下一处旧逻辑，按 first-fault 报告。

### 3.1 阶段 A：接线（P0，先于一切）

| # | 改什么 | 复用 / 依据 | 验收 |
| --- | --- | --- | --- |
| A1 | 决策人口改为 role=train 序列；请求沿用 `sequence_request()` 的一维单序列契约；新增"训练/预测角色"字段进 `task_context` 或请求 | `per_sequence.py:83–93,148–167` | 训练阶段每条 train 序列各一个 request；eval 序列在训练阶段不产生 request |
| A2 | 评价器：把 `_evaluate_assignment` 接入主链（可复制到 `main_protocol_p4/`），输入 `{train_uid: compiled_or_None}`；补预测侧 `{eval_uid: compiled_or_None}` 的逐条因果准备；返回逐 eval 损失、总收益、伤害、`consumer_fits`；`loo_credit` 开关 | `run_batch_composition_headroom.py:242–307`；`scoped_serving_evaluator._prepare` 的因果准备与 degenerate 检查 | 全赋同一程序 + 预测侧 identity 时与旧 `_evaluate(train_only)` 逐位一致；全 identity 与 Static 逐位一致 |
| A3 | Consumer 结构字段进 `TaskSpec`/`TaskContext` → Fast 与 Slow 的 `public_input`；Slow 调用补传 `task_context` | §2.2(4)；`fast_agent.py:638–647`；`slow_agent.py:417–419` | 渲染字节里出现全部冻结字段；无模型 ID→程序的任何映射代码 |
| A4 | 冻结路径：held-in 结束固定 {W_t}，拟合一次并持有模型对象；held-out 每窗口只做预测侧 Fast + 预测 | §2.2(5)(8) | 断言 held-out 各窗口使用同一模型对象；held-out 零 fit（预测侧准备除外） |
| A5 | Slow 材料改为"训练批次卡"：逐训练序列 Episode（含 Δ_t）、总收益、eval 伤害分布、预测侧决策；六编辑面复用 | `dev_seq2_knowledge.py:479–583,692–704`；`run_dev_deploy2.build_card` | Slow 输入含 `task_context`、Consumer 结构、训练侧改动事实；不含任何 held-out Outcome |
| A6 | 保留既有修补：`prepare()` FAILED→UNKNOWN、窗口绑定、验证去重、逐步落盘预算、恢复不重问 | Grok A 修复回执 `GROK_CODE_REPAIR_RESULT_2026-09-09.md` | 故障注入两路（返回 FAILED / 抛异常）均记 UNKNOWN |

不做：新 Schema 或存储平台；任何新 SHA/manifest；运行时猴子补丁改旧 runner；改 Consumer、
评分、风险线、训练 anchors、候选槽数。旧 `scoped_evaluate` / `ReplayPredictionCache`
保留供历史复现，不再用于本包读数。

### 3.2 阶段 B：零 LLM 夹具（对应备忘录 §17.4 清单）

1. 两条训练序列给不同 Workflow，汇总设计矩阵中各自窗口的改动点数与单独施加时逐位相等。
2. 同一臂同一轮所有 eval 预测来自同一模型对象；换一条训练序列的 W_t 后模型必然改变。
3. 预测侧：每条 eval 序列的输入独立准备，只用 origin 前数据（因果断言）。
4. identity 全赋值（训练侧与预测侧）与 Static 逐位相等。
5. Fast 与 Slow 的实际输入字节不含 `evaluation_truth`、delayed/held-out Outcome（字段名审计 + 数值不出现）。
6. 留一重拟合：Δ_t 可算、计入 `consumer_fits`；记录 ΣΔ_t 与总收益之差，只记录不要求相等。
7. 训练目标处理选项 (a)/(b) 各产生一份读数，主臂固定为 Astra 裁定的那一项。

### 3.3 阶段 C：手写 Skill 能力对照（不计作学习证据）

一张人写的程序性 Specific Skill，例如："对跨序列共享的线性 Consumer，若训练窗口中
缺口边界后紧接高幅残差，先 `period_median_complete` 再离群处理；只对该模式进入训练样本
的序列作用；其余保持 identity"（模式来源：P4d origin 2856 组合正证据、R4F 线索）。
目的只有一个：证明新接口能表达并执行**不同于全量单程序的异质赋值**，且在 u9–u11 上
读数可比。它是接口能力测试与专家上界参考，**不是**系统学会了什么的证据。

### 3.4 阶段 D：小规模五臂

**数据与 Consumer。** `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`，一个 development
块（20 train + 20 eval，与 DEV-SEQ/DEPLOY 同 roster）。held-in = u9/u10/u11，
held-out = u12/u13（**development 级，不是密封**）。主 Consumer pooled Ridge；per-channel
用同一批赋值重算作第二张表（预算内）。

**决策人口。** 默认 train 10 条 + eval 10 条（沿 `DECISION_POPULATION_SIZE=10` 先例，
固定 roster 顺序、开跑前写死）；其余 10 条 eval 以 raw 输入从同一模型预测，只作"非决策
人口"诊断，不入主分母。Astra 可改为 20/20，成本相应加倍。

**五臂。**

| 臂 | 含义 | 读什么 |
| --- | --- | --- |
| Static | 全 identity | 地板 |
| Best-Fixed | 在 u9–u11 反馈上校准的最佳单程序（训练侧全量 + 预测侧同程序），单步菜单，**声明为单步下界**；若 0-LLM 可枚举，另报最佳两步组合 | 最强非 Agent 对照 |
| K0-frozen | 初始 General Skill，无 Slow，直接跑到 held-out | 学习增量的基线 |
| A3 | K0 + 每轮边界一次 Slow 修订 | 本地反馈学习 |
| A5 | K0 + 一张来自**另一个 KDD 块**的 Source Specific Skill（确定性/Slow 整合后审计冻结），held-in 预算与 A3 相同 | 积累知识的边际贡献；如实标为 cohort 迁移 |

**预算（10/10 人口）。** 每 LLM 臂每窗口 10 次训练侧 + 10 次预测侧 Fast；held-in 3 窗 +
held-out 2 窗 → 100 次/臂；三个 LLM 臂 ≈ 300 次 Fast，Slow ≤ 8 次（含重试封顶）。
fit：每臂每 held-in 窗 1 + 10（留一）；held-out 零 fit → 五臂约 170 次，per-channel 副表
再翻一倍，**上限 500**。批内 4 路并行沿用 DEV-SEQ-3 实测（×3.74）。模型身份由 Astra/用户
定并核返回标识，不默认 Flash，不因端点故障静默换模型。

**预写终点。** 主终点：held-out 平均收益（vs Static，origin 面）。共同主终点：受损序列
比例、最坏单序列伤害。次终点：首个正向方案的反馈成本；held-in→held-out 保留率；Δ_t 分布
及其跨窗口相关（顺带回答"新几何下逐序列效应是否持久"）。

**预写判读。**

| 读数 | 结论 |
| --- | --- |
| A3 或 A5 ≥ Best-Fixed 且伤害更少 | 机制在新几何下正向；进入 TRAIN-2 |
| A3 > K0-frozen（质量或伤害） | Slow 有限修订有增量 |
| A5 > A3 | 积累知识有边际贡献（cohort 级） |
| 全部 LLM 臂 < Best-Fixed | 不改口径；按 §6 first-fault 报告卡在供给/观察/选择/执行/归因哪一环 |

**停止规则。** 账户/权限故障不算弃权，记 `ACCOUNT_OR_PERMISSION_FAULT`；A5 的 Specific
卡先做零成本暴露检查，0 命中则 A5 不跑、记 `COMPILED_BUT_NEVER_EXPOSED`；任何
prepare FAILED 记 UNKNOWN 不缩分母；不追加第二组、不为某张卡改条件追正号。

### 3.5 交付

一个 runner 包 `evaluation/main_protocol_p4/run_dev_train1.py`（复用 `per_sequence`、
`dev_seq2_knowledge`、`online_loop` 部件）、一个 smoke、一份结果报告
`docs/DEV_TRAIN1_RESULT_<date>.md`。报告按正典 §10 五问作答，分列 CAPABILITY / MECHANISM /
INSTRUMENT / NEGATIVE。`AGENTS.md` §5.7 的"实施规格已定"一句由 Astra 追加；DECISIONS 由
Fable 登记回执。

### 3.6 顺序与估时

接线 A1–A6 约 2–3 个工作日；夹具 0.5 日；能力对照 0.5 日；五臂运行 1–2 日；报告 0.5 日。
9 月 15 日检查点应决定的是**批准本包与实施规格选项**，不是看结果。

---

## 4. 之后的路线

### 4.1 DEV-TRAIN-2：程序性 Skill 家族与学习增量（TRAIN-1 有方向性正号后）

- 冻结工具与 DSL，**只改 Skill**：同一 Fast + 初始 Skill 对 同一 Fast + Slow 修订 Skill，
  同 held-out、同预算；R 与 C 只在有信号时加对照，不铺四臂矩阵。
- Skill 学的是"检查什么、怎么构造"：Slow 编辑对象限两类——观察过程（加一个既有工具能
  完成的局部检查或改检查顺序）、构造过程（按证据改作用范围、参数、顺序）；不以收窄
  `observable_applicability` 为主要动作。`observable_applicability` 只是入口条件。
- 一次性输入变更（各臂同享、单独登记）：把已计算但不进 LLM 的 `recent.*/change.*`
  近期窗口统计暴露给 Fast；补一个带起止参数的局部窗口查询工具。工具收益与知识收益分列。
- 证据链必须完整：合法经验 → 一处具体修订 → 新情境中观察或程序确实改变 → 无 Support
  代选的交付下下游结果改善。只到第二步是可控性，不是有用性。

### 4.2 论文级主实验（TRAIN-2 有可归因增量后）

- **形状**：Static / Best-Fixed / A3 / A5 同场；held-in 多轮 → 冻结 → held-out 零反馈；
  这是正典 §1 要求的那次自然数据同场验收。
- **数据**：held-out 在打开前冻结为"未见序列块 × 未来窗口"两个维度（KDD 含缺口 roster 239
  条，先按数据地图核曝光记录）；Source 域用一个**真正不同、也有天然缺口**的预测域
  （Monash 带 `with_missing_values` 版本的数据集是候选，需核数据资产清单），否则 A5 只能
  写 cohort 迁移。
- **Consumer**：pooled 主表，per-channel 第二张表，把 M0 现象写成 Harness 行为读数。
- **预注册**：主终点 held-out 平均收益；共同主终点伤害比例与最坏伤害；四假设
  （A5>Static；A5>A3；A5 质量不低于 Best-Fixed 且伤害更少；A5 的 held-in→held-out
  保留率高于 A3）；A5 treatment funnel 全程报告（K0 / Match / Supply / Selection /
  Admission / Deployment / Re-encounter / Marginal gain）。
- **成本**：需要 fresh 数据、更多块、更长 held-in；数周量级；协议冻结前需用户批准数据与预算。

### 4.3 可争取的贡献（主语都是"系统做成了什么"）

1. **Task/Consumer 条件化的训练数据准备生成**：Agent 读结构性 Consumer 描述与训练序列
   的可见结构，逐条生成带范围/参数/顺序的 Workflow，汇总训练一个模型；冻结后在下游任务上
   优于强固定准备与**同工具**的通用 Agent。
2. **成对反馈驱动的程序性 Skill 修订**：以留一边际贡献等局部对照定位需修改的观察/构造
   步骤，修订后的 Skill 在同一 Fast、同工具、同预算下交付质量更高；与普通反思对比说明
   机制有额外价值。
3. **积累的 Consumer 相对知识改善新域适应**：相同 Target 反馈预算下更快形成有效准备、
   冻结后质量—风险更好；机制读法是"跨域先验使 held-in 选择泛化，避免过拟合适应窗口"。

已有负结果（静态特征路由条件化、Skill 卡不改善路由结果、固定程序跨窗口不稳）进
motivation 与消融，解释为什么只记程序编号、只收窄 Scope 不够；不作主体。

### 4.4 检查点

- **9/15**：批准 TRAIN-1 与 §2.2 的规格选项（训练目标 (a)/(b)、人口 10/10 或 20/20、
  模型身份）；不以 headroom 类探测为生死门。
- **TRAIN-1 收口**：五处接线是否对齐；是否有方向性正号；决定 TRAIN-2 或 first-fault 修复。
- **TRAIN-2 收口**：是否有可归因学习增量；决定主实验协议冻结与 Source 域。

---

## 5. 起草人的想法与供 Astra 讨论的未决点

1. **训练目标可否修改**（§2.2(2)）是本包最需要事先裁定的规格项；两种选项都合理，
   建议主臂沿用整窗口、对照读数记录只改 context。
2. **预测侧在 TRAIN-1 是否全量做**：§5.7 要求逐条准备预测输入，成本约占 LLM 调用一半；
   若预算紧，可将预测侧人口降到 10 条，但不建议退回 raw——那会把 TRAIN-1 变成另一种几何。
3. **留一归因的边界**：Δ_t 是局部对照不是因果分解；建议进 Episode、进 Slow 材料，但报告
   里明说 ΣΔ_t ≠ 总收益。
4. **逐序列持久性**在新几何下是否仍≈0，TRAIN-1 顺带测（Δ_t 跨窗口相关）；若仍≈0，
   Skill 的学习对象应更偏向"面对此 Consumer 的通用构造原则"而非"某类序列的条款"。
5. **可选 0-LLM 预检**：在 `_evaluate_assignment` 上从最佳单程序出发做坐标上升（逐条尝试
   换 W_t），留出窗口验证，几百次 fit，回答"异质赋值有没有空间"。不作发车门，只定靶子大小。
6. **Best-Fixed 是否扩到 1–2 步组合**：若 0-LLM 可枚举则扩，否则明写单步下界。
7. **A5 Source 来源**：TRAIN-1 用另一 KDD 块（cohort 迁移）；跨域只在主实验做。
8. **与旧线的关系**：SEQ-4 剩余三格、DEPLOY 报告的更正照旧由原维护者处理；本包不重跑、
   不改旧 runner；旧读数不与新几何读数相减。
9. **8 月正向结果的复核**：Frep/T6 路径的 A5 treatment 需 astra 按 funnel 复核后才可在论文
   中引用；Monash runner 结果不引。

---

## 6. 反过度工程与纪律自检

0 新增 SHA；一个 runner、一个 smoke、一个报告；不建 Schema/记账/哈希平台；不改旧 runner；
不开密封数据；Runner 不按模型 ID 映射 Workflow；Fast/Slow 不读 held-out Outcome；
故障记 UNKNOWN 不缩分母；identity 全赋值 ≡ Static；研发用 Grok 调用与项目实验调用分开计。

---

## 附录 A：关键代码指针（两份审计交叉一致，根 Agent 抽核）

| 位置 | 事实 |
| --- | --- |
| `contracts/method.py:14–20,43–49` | `PreparationRequest.values` 强制一维；适合逐条 Fast，无需改契约 |
| `per_sequence.py:83–93` | 决策人口 = `eval_uids[:N]`（须改为 train） |
| `per_sequence.py:148–167` | `sequence_request()` 已按 UID 重绑 values/observed |
| `per_sequence.py:39–48` | 明文：K+1 模型、按 eval 取行、非独立因果效应 |
| `scoped_serving_evaluator.py:57–70,155–162,204–212` | 训练窗口全量准备；scope 只路由 eval 预测行 |
| `run_batch_composition_headroom.py:242–307` | `_evaluate_assignment`：逐训练序列赋值 → 一个共享 Ridge（pooled/per_channel） |
| `contracts/task.py:112–119`；`fast_agent.py:638–647` | Consumer 仅 `downstream_model_class` 字符串 |
| `run_dev_seq1.py:608`、`run_dev_seq2.py:892`、`run_dev_deploy2.py:691` | Slow 调用不传 `task_context` |
| `run_e2_recipe_experience_to_skill.py:881–891` | 已有 pooled/per_channel 结构文本，可作字段种子 |
| `dev_seq2_knowledge.py:479–583` | 六个编辑面模板 |
| `run_dev_deploy1.py:501–518`；`run_dev_deploy2.py:318–321` | Best-Fixed 单步菜单；选择时排除 identity |
| `run_v1_monash_fresh_a5_a3.py:410–416` | `src_eps` 未传入；A5≡A3 |
| `signed_radius.py:52–73`；`fast_agent.py:93–102,841–847` | `recent.*/change.*` 已计算，只用于 no-op 过滤与 signed 半径，不进 LLM 输入 |

## 附录 B：拟议写入 `AGENTS.md` §5.7 的实施规格补充（供 Astra 定稿，非本文件生效）

> 实施规格（DEV-TRAIN-1 起）：训练材料为 role=train 序列在冻结 anchors 上的窗口；
> Fast 在训练与预测两阶段被明确告知角色；训练目标处理选项按本包裁定并冻结；预测侧允许
> 算子集为训练侧子集且禁止 shape-changing；Consumer 以冻结的结构字段描述、禁止按 ID
> 映射；同一臂同一轮一个共享 Consumer，缓存以整批赋值为键；留一重拟合为可选计费的局部
> 对照；held-out 中 Fast 只在预测侧、只读部署可见输入。


