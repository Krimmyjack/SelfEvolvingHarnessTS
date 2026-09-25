# DEV-BATCH-RESEARCH-FAST-BASELINES：批级 Fast 效用确认

日期：2026-09-13。负责人：当前 W 根执行者。用户在最新状态汇报后要求“继续推进”；本包落实公共工具纠错与下一轮有界开发对照。当前状态：已收口，4/4方法完整，36次拟合、9次模型请求；实际运行状态由包内 launch / execution_finished / result 记录。P/G 分支、历史包和 Natural Final 保持原状态，不提交 git、不新增哈希。

## 1. 问题与范围

检验批级 Fast 在相同候选拟合预算内，能否构造并提交比固定 Mixup、随机搜索更好的完整训练材料。上包 roundtrip 未显示净优势，因此本包不开强制证据分轮；Fast 仍可主动分轮、观察、比较和停止。H0 为空，0 Source/Slow，不检验知识积累。

公共修复：模型提交合法工具名但参数校验失败时，返回确切错误，允许它用普通下一次调用修正。每个 Fast Job 最多返回 2 次纠错机会；第 3 次拒绝终止。失败工具尝试、修正请求均计入原预算；同回复未执行后缀丢弃，必须明确重发，已完成前缀保留。校验器不猜字段、代改配方或自动回退。非法 JSON/动作外壳、越界访问、训练/写盘/模型绑定失败仍终止；这属于公共执行能力，不声称自进化贡献。历史默认 max_tool_corrections=0 保留，已开 E 的 a85_old 不补跑。

## 2. 冻结工作负载

按既定时间顺序采用 Electricity a90、a95，同原字符串序 32 实体。两者均按 EXPOSED_DEVELOPMENT 管理，不能宣称未见或独立终验。选作业不读取当前 C/E 分数；不得替换不利或失败作业。

|Job|T|C_A origins|C_B origins|E origins|
|---|---|---|---|---|
|a90|[22992,23664)|23664,23712|23760,23808|23856,23904,23952,24000|
|a95|[24312,24984)|24984,25032|25080,25128|25176,25224,25272,25320|

同一数值底座：T=672、L=192、H=48、433×32 父对、原/派生各半权重、T 上每实体 scaler、共享 MLP 192→128→64→48、2000 更新、父 batch64，训练/增强随机流和动作域全部复用。训练 seed=20260922/20260923/20260924；第一 seed 为实际单模型交付，三 seed 均值为条件效用诊断，不选幸运 seed、不集成。

Fast 构造只读 T、评估只读合法 C_A；两个 Job 的全部方法决策完成后才开 C_B。C_B 只作延迟审计，无 argmin 覆盖 Fast commit。所有合格分支 E 预测全局冻结后一次性统一评分；结果不回流本包任何 Agent。

## 3. 基线、权限与预算

每 Job 公共 None/Fixed-Mixup 各3次训练；Fast 最多2个新完整候选×3seed；Random 同样2个×3seed。公共基线物理模型共享但私有候选/轨迹隔离。每个方案仍联合训练一个共享 Consumer，不能按实体拼分。

Random 使用原 policy.random_policy；a90 的采样索引为910901/910902，a95为910951/910952；重复、identity、别名自然保留，不重抽。它按 C_A 三seed均值从 None/Fixed/p1/p2 取最小，平局依该顺序；这只是 Random 基线的规则。Fast 自主 commit，不强制选择均值最小者。

顺序固定：a90 Fast→Random；a95 Random→Fast。Random 先跑的作业，Fast 仍只拿公共基线与自己的工具结果。

|资源|全包上限|
|---|---:|
|计划物理拟合|2×(6公共+6Fast+6Random)=36|
|拟合尝试硬帽|38（2格仅预留同配置非科学故障；当前执行器不自动重试）|
|Fast普通调用（包含纠正）|每Job16，全包32|
|HTTP尝试|64；单次传输最多原请求重传1次|
|执行工具尝试|每Fast Job24|
|新候选评估|每方法每Job2|
|可返回工具错误|每Fast Job2，包含在普通调用/工具帽中|
|返回usage token停止边界|1,000,000；未知usage停止后续调用|
|活跃墙钟|5400s（90分钟），单fit子进程最多300s|

使用现有 project 环境、现有授权中转和 grok-4.6-build 返回身份。凭据仅运行时环境；没有额外网络探活收费调用。后端曾忽略 max_tokens，最终在途请求可能跨 token 停止边界，实际全部计账；不能把该边界说成服务端硬token限额。

## 4. 结果表与判断

每 Job 同时报告 None、Fixed、Fast commit、Random commit 的三个 seed E、均值及实际第一 seed。主比较 Random−Fast 与 Fixed−Fast，正为 Fast 改善；同时保留 C_A/C_B 差值、seed SE、逐 origin/实体向量。两个作业分别报告，不用平均数掩盖反向。

分开回答：①实际生成和训练了哪些完整材料；②候选集合是否有较好材料；③Fast最终有没有提交它；④多付出的 LLM/token/拟合成本。集合中最小 E 仅为已运行候选的回顾性描述，不回改 commit，不当完整搜索空间 oracle。三训练seed不是三条独立Agent轨迹。

- Fast 优于 None 但与 Fixed/Random 相当：增强有价值，未支持 Agent 额外价值。
- 有好候选但没有交付：定位选择问题，不能直接手写 E 赢家指导。
- Fast 无好候选且简单基线更好：定位观察/构造/搜索策略，不继续加 Source。
- 不完整分支：主比较 UNKNOWN；不得用公共基线兜底补成完成。
- 实际错误被返回且模型明确修正：只能证明纠错通路，不等于提高材料价值。

不新增显著性采用门、不修改指标或删难例。主张只及当前开发设置；完成后用 aris-result-to-claim 做一次结果审查，不启动新的课程或第三个作业。

## 5. 实施、检查与复跑入口

复用现有 run_batch_research_roundtrip.py 的 --study fast_baselines 配置，不另建训练器。公共 branch 同时向控制器与 W adapter 接入纠错开关。26项控制/费用/信息墙测试通过；新增真实校验器测试证明错误字段不会被自动更名、错误前不构造材料、后缀不执行、各种预算不增加、I/O/权限错误不被吞。薄入口 smoke 验证两个作业、36格与两处接线。

启动：`python -B -m evaluation.main_protocol_p4.run_batch_research_roundtrip --study fast_baselines --run`。

输出：`_scratch/dev_batch_research_fast_baselines/`，含config、预算、原始响应、各分支完整材料/模型/预测、result.json和唯一主REPORT.md。目录存在即拒绝重新启动。重做汇总必须显式传同一个 --study；禁止在历史目录用其他study覆盖。最终收口后不追加种子、拟合或Source。

## 6. 实际收口

21:18:55至21:38:54，活跃1199.46s，36/36拟合成功，0失败0重试。9次请求/9HTTP，400201 token，24次公共基线缓存命中。两个Fast分支均完整，规则送达9/9请求，实际工具纠错触发0；不把受控测试当自然纠错效果。

a90 Fast/Random均交付None，E三seed均值2.891074；Fixed2.527578，Fast/Random反而更差。None相对Fixed的C_A/C_B三seed均正、E三seed均负，当前反转不是只换seed就消失的现象。a95 Fast异质Mixup0.479504、Random0.414051、Fixed0.496833、None0.498221；Random在C_A/C_B/E各块均优于Fast交付。未支持Fast额外净效用，不启动Source。

48份冻结E预测复算差≤5.69e−14，全部检查通过。原worker最后因报告展示引用旧字段KeyError而退出1，所有方法与评分此前完整；仅修报告后全部数值逐项不变。另将旧roundtrip元字段改为明确的fast_executed/evidence_roundtrip_enabled。未重训、未重调模型、未修改commit。原错误日志保留。

完整结果、候选供给/选择分解与工具语义缺口见 [_scratch/dev_batch_research_fast_baselines/REPORT.md](../_scratch/dev_batch_research_fast_baselines/REPORT.md)。本包停止，没有残余实验任务、Source或追加作业。

收尾审查：新上下文同家族核查完成，数值/阶段/费用通过；全部三seed原值已补入主报告。provisional，保留报告退出与donor工具语义warning。
