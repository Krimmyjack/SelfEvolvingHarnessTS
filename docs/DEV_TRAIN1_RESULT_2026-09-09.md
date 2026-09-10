# DEV-TRAIN-1：逐训练序列 Workflow 与共享模型

状态：PROCESS_COMPLETED__EDIT_USED__IDENTITY_ONLY__PAIRED_ENDPOINT_INCOMPLETE。
2026-09-10 10:46运行结束；根Agent完成工件初读，尚非独立审查。
真实Fast/Slow已运行：新指导的60条最终决策全部identity，与Static完全相同；
旧臂u13含1条传输故障，完整两窗口配对效应不可读。详见下节，后续实施记录保留。
执行规格见
`MAINLINE_ROADMAP_AND_DEV_TRAIN1_TASK_DRAFT_2026-09-09.md` 顶部定稿（旧草案已标存档）。

## 结果及机制初读（2026-09-10）

唯一主工件：`_scratch/dev_train1/runs/dev_train1_live_20260910/result.json`。
按同一完整20条评价人口，收益=Static mean_sMASE−arm mean_sMASE：

| 臂 | u12收益 | u13收益 | u12/u13受损次数（各20） |
|---|---:|---:|---|
| Static | 0 | 0 | 0 / 0 |
| 形成段校准固定IQR | +0.556129 | +0.012833 | 5 / 8 |
| 旧Skill | −0.163395 | UNKNOWN | 13 / UNKNOWN |
| 新Skill | 0 | 0 | 0 / 0 |

固定IQR两窗口最坏伤害0.833335/0.792090；旧Skill u12为0.621916。
固定IQR收益更高不代表风险上支配其余臂。不能用u13旧臂的19条决策另换分母。
实际脚本COMPLETED不等于完整方法证据：200个唯一决策键落盘、190有效，形成段9无效，
旧臂u13/T147为AgentTransportError、UNKNOWN。恢复时保留了Slow已经消费的形成证据。
本次课程982次API尝试、57次已完成Consumer fit；含前期开销累计984次API/100次fit。
活动运行约2小时40分，Windows更新重启至恢复的空档另列，预算未清零。

### 为什么新指导全部identity

1. `boundary.json`证明Slow自主PATCH了`build_contrastive_candidates.body`，不是新增程序卡。
   `exposure_check.json`证明修改对20条训练及40条预测视图全部有影响。
2. 新Skill的20训练+40预测全部`IDENTITY_SELECTED`，60/60的`candidate_programs={}`。
   旧臂仅1/60池为空（传输故障），其余59条有候选。新臂两窗逐UID损失与Static完全相等。
3. 四个`*_probe_direction`在全部200条当前记录中均为`unknown`。
   `dev_train1_runner.py:run_fast_decision`不传`fixed_probe_panel`；
   `method.py:bind_round_data`同样创建无panel的LocalPublicToolGateway。
   `public_tools.py:extract_public_features`将空panel变成unknown；
   `LocalPublicToolGateway.schemas_for`只在已有panel非空时提供`read_fixed_probe_panel`。
   因而它是“读取已提供的面板”，不是一个能在本实验中主动产生正方向的工具。
4. 新正文却明确要求imputation/clipping/denoising等方向positive才能提出对应处理，
   unknown时优先identity。这与全空候选、全identity直接一致；不能再称“Fast没使用知识”。
   它不是代码新增硬门，而是Slow写进共享指导、Fast实际遵从的条件。

**支持到哪里：**u12出现了避免旧策略伤害的局部效果，但新知识退回Static，未创造
准备收益；本包不支持条件化优于固定方案，也不支持跨域积累。没有同起点随机重复，
不把所有臂间变化都唯一归因于这一句文本。unknown≠negative；缺少当前效果反馈是
Fast-only设计的边界，不能给Fast偷接当前Support来“修好”这项失败。

### Slow拿到的证据有什么局限

形成段只有u9可完整计分（mean gain −0.143058，12条负、6条正、2条近零）；u10/u11
因故障为UNKNOWN。`build_slow_card`将有正负数值的记录分别收集，并未执行机制聚类，
`matched_successes`实际上没有近邻或条件匹配步骤；名称不能当作匹配成功的证明。
当前输入给了逐行unknown字段，但未明确说明本运行中这些方向始终不可产生正值、
Fast工具仅提供summary/localize。Slow还保留了旧正文对Support与Experience的措辞，
与本包Fast-only边界有歧义。应陈述实际能力，不能要求模型等待不存在的证据。

### 已有读数提供的下一步线索（未发车）

固定预测输入V=identity的形成段训练分解中，baseline平均sMASE为1.354838：
IQR整窗口1.255401、保留Xp/y0为1.442274、保留X0/yp为1.175292；
MAD相应为1.345359、1.519462、1.185733。
这些是`menu_xy_decomposition.json`的已处理分量保留对照，不是独立X/y处理的
可加总因果贡献；它们说明“训练输入、训练目标、预测输入”不能统一理解成清洗同一种材料。
IQR真正输入侧对照不可读，不能将其当作零收益。该表固定V，与固定基线同时处理V的
主表不同，不跨表混算。它可帮助下一次Slow判断该修改哪个角色，而非手写一个赢家。

下一项建议只做一个有界修订检查：把**实际可用观察/工具、不可获得的probe信号及
训练/预测角色**如实交给Slow，保留当前卡和成败材料，让它自主保留或修订一处；
不规定必须少弃权、改IQR或扩大覆盖。先看是否仍依赖不可取得的条件，再决定是否
做新的真实Fast对照；不得覆盖本次结果或把当前曝光窗口称为fresh。
这项新修订机会超出本包“一次Slow”的已执行配置，当前仅登记建议，未调用模型。

本节为根Agent同上下文复核。Grok的有界统计委派返回cancelled、没有完成表格；
不作为独立确认，亦未继续重复派工。未启动额外LLM实验，未修改核心或正典文件。

### 小型能力说明检查：批准与执行规格（2026-09-10，结果前记录）

用户在上述建议后回复“继续”，授权此单一检查。父版本是本包Slow已经形成的指导，
不是重新从h0抽一个赢家。复用u9/u10/u11原形成材料、原训练X/y分解及原修订全文，
补入真实工具/信号可达性、训练/预测角色说明；不送入u12/u13评分或人工替代正文。
Slow仍自行选择一个现有可写面、修改/保留/弃权；不强制少弃权、扩大范围或指定算子。

一次新的修订机会；仅协议错误按既有边界规则至多修复一次，传输按既有有界重试。
新增项目API上限8、Consumer fits=0、15分钟；不改上包收据、不重置已支出成本。
输出独立位于`_scratch/dev_train1/runs/dev_train1_slow_capability_check_20260910/`。
程序由Grok有界提供、根Agent审核执行，不准再派子代理。

判读只问新指导是否继续依赖不可取得的probe，是否给出了实际可执行的观察/构造/
选择建议；“删去positive前提”不自动等于有用，也不作为编译或接受的新门。
这次未设计随机信息消融，不能把文本变化归因于能力说明的独立因果效果。
不自动运行Fast或宣传效用/迁移；若值得测，再单列后续同信息、同人口的实际对照。

### 小型能力说明检查结果（2026-09-10，已完成）

判词：`CAPABILITY_CLARIFICATION_DID_NOT_REMOVE_UNAVAILABLE_PROBE_DEPENDENCY`。
真实Slow调用1次、0 fits、122.27秒；请求`cpa-grok-4.6`，返回`grok-4.6-build`。
Grok实施草稿另计$0.0435268，根Agent复核后简化为复用现有接口的scratch脚本，
未改核心文件。工件在上述独立目录的`input.json`、`boundary.json`、`result.json`；
启动与结束收据为`.aris/runs/dev-train1-slow-capability-check-20260910/launch.json`。

**实际修改不是预想的解除限制。** Slow自主选择
`bootstrap_skills.entries/inspect_and_localize.body`进行PATCH并编译成功。
新正文明确写道：`If imputation_probe_direction is not positive, including unknown, do not emit missingness.`
对clipping/denoising也作同类限制。它预测继续输出no_actionable_signal、空候选和identity。
原`build_contrastive_candidates`文件逐字节不变，故原候选生成和选择阶段的positive前提
仍保留；这次只是把同类限制复制到了另一张观察指导中。只有这一张Skill源条目变化，
父版未覆盖，无部署或晋升。

**输入与读数边界。** input保留20条原训练记录及u9/u10/u11形成材料，实际有数值的
成败例仍只有u9的12负/6正；u10/u11 UNKNOWN没有补成零或重采。能力说明与前次修订
全文进入原`failure_pattern_card`通路；没有送入u12/u13评分或全identity的最终行为统计。
本轮没有Fast调用，因此不能把它的identity预测当作新的60条实测，也不能声称一定全局
无候选（level/period还有其他分支）。编译合法不等于建议能产生效用。

**如何理解。** “只要把实际能力说清，Slow就会自行解除不可达条件”的预期没有在这次
得到支持。它可能是在有限且总体偏负的形成证据下维持保守政策，也可能没有把缺少
效果信号与不值得处理分开；当前输出无法唯一识别这两种解释。一次附加修订也不能
独立估计能力说明的因果效应。此结果不支持继续只补一句提示后马上重跑完整四臂。

**后续建议（未启动，不新增授权）。** 不再对同一输入连续催问或手工替换成赢家。
下一项方法讨论应把“当前观察能否支持具体动作”与“训练/预测角色的价值”放在一起：
训练prefix摘要不等于各240点执行窗的事实，预测角色不能恢复训练侧已失去的结构；
共享训练损失也不能直接归因给某个评价UID。已有X/y分解可作合法形成证据，不能因此
给Fast接当前Support。若开放更贴近实际作用窗的只读观察，须明确是信息接口变化，
先验证观察可用、动作可执行，再测新旧知识效用，不能把接口收益记成Slow学习收益。
同样不能以“少弃权”作为成功目标；Static与形成段固定方案继续作效果和风险参照。

本节为根Agent同上下文语义复核，未另启动独立评审或新实验；DECISIONS/STATE由Fable
按本回执同步。原DEV-TRAIN-1主终点及故障记录不改。

## 当前交付与分工

- 用户已批准整训练窗口 X+y 主臂、输入侧对照、20/20与继续实施，并追加批内并发要求。
- Grok数值线：`7b4e3d4c-aff2-43d3-b833-6efdb984607b`；只写新 evaluator及smoke。
- Grok runner线：`c260e29e-f3e8-4d26-9845-46a4e5573e81`；只写新runner和CLI及fake测试。
- 两线均禁止项目真实API/真实数据实验；根Agent复核后接通本包真实运行，不再请用户
  逐项批准普通修补。数值/编排实现的共同文件不并发写。
- `DECISIONS.md` / `STATE_ONE_PAGE` 由Fable维护；本段供其登记执行收到，不冒写其回执。

### 9/10 恢复回执与数值验收

用户明确确认此前中断为人为操作并批准恢复。原数值/runner客户端均已退出，未假定旧PID
仍在运行。数值会话第二次返回的两文件patch由根Agent检查后应用；runner按原会话
`c260e29e-f3e8-4d26-9845-46a4e5573e81` 续接，新收据位于
`_scratch/grok/runs/dev_train1_runner_20260910_02/`，没有重启已完成实验或重问Slow。

新增 `dev_train1_evaluator.py` / `smoke_dev_train1.py`。根Agent修复：多维输入被ravel
静默展平；B(X)不可计算却误拒合法B([X,y])；分量重组的改点数误包含已舍弃的处理分量。
首次烟测12/13通过，失败来自测试夹具：周期数据中的单个极值不保证改变winsor分位点。
改为明确移动分位点的合成目标块，未放宽数值容差；另加上述回归检查。

运行命令：Windows project Python执行
`python -B -m evaluation.main_protocol_p4.smoke_dev_train1`。
最终 **17/17通过，5次合成fit，0项目LLM，0真实数据读取**；首次烟测另花5次合成fit，
两次合计10次。覆盖异质赋值、旧pooled数值对齐、预测阶段禁止调用solve、模型恢复、
真正W(X)与输出分量重组的区别、identity输入并非全局零收益、缺测保持全人口UNKNOWN。
这是工程验收，不是知识修订有效的证据。runner与真实API尚待接通验收。

## 已核实的实现前提

`_evaluate_assignment` 能汇总异质训练窗口，但直接fit+predict，不保存冻结模型；本包需
拆分该路径。它的 `per_channel` 是按train序列拟合后平均预测，并非另一报告中eval
自身训练的设置；本包不运行这个分支。固定anchors只固定原始窗口，不固定准备后的矩阵。

`_exact_weighted_ridge_prediction` 实际为附加常数列、alpha=1且不惩罚截距的正规方程。
复用时核对模型系数与旧预测，不凭类名替换实现。预测真值只供外部评分。

整窗口输出分量分解与真正W(X)对照分开；单个预测输入identity不能强制计收益0，
因为共享模型仍可能受训练侧准备影响。

初始知识明确使用未修改的 `methods/ttha/harness/h0` General指导；历史DEPLOY的K0
包含Frozen program steps程序卡，会自动占候选池位置，不作本包共同起点。根Agent
在真实调用前检查这一差异，避免候选供给效应混入指导生成路径。

## 环境预检（非方法结果）

已确认既有Windows project Python3.10.19可调用，numpy/pytest/httpx/openai存在；
Windows用户级CPA/OpenAI凭据存在，只核存在性、不记录密钥。进程环境未自动继承用户
级凭据，实际发车须通过已有用户级配置注入进程，不能因此静默选默认远端。
首次只读检查可用内存约0.83 GB，因此尚未加载真实实验；磁盘约497 GB可用。
9/10恢复后的只读复查为0.50/15.7 GB空闲；当前继续代码与轻量验收，不在此余量下启动
真实四路训练/预测批次。未关闭用户程序。Grok续接客户端实际PID180075（父launcher
180058）已查到存活；客户端存活不等于runner已实现，更不等于真实实验已发车。

实验指定入口 `http://127.0.0.1:8318/v1`、请求模型 `cpa-grok-4.6`；用户更新凭据后，
`/models`只读查询已成功且包含所请求的ID。尚未完成生成预检，不能据此声称推理返回身份
已验证。默认4路Fast会话，内存/限流时2路，拟合串行。

## 待实测

| 项目 | 状态 |
| --- | --- |
| 异质训练→共享模型数值验收 | 合成17/17通过；真实数据尚未运行 |
| 四路独立会话/恢复/无反馈信息墙 | runner续接03生成补丁中，待fake检查 |
| 形成段X/y分解与固定单步校准 | 未运行 |
| 真实Slow一次修订 | 未运行 |
| 旧/新Skill冻结后的u12/u13对照 | 未运行 |

真实主读数以全20评价序列的origin效用、风险/伤害、处理量和成本报告；u9–u13
全部仍是曝光development，不报告为密封泛化。后续结果在本报告更新，不另造重复报告。

## 后续恢复与生成预检（9/10追加，更新上方较早状态）

runner续接02以cancelled结束且无文件；用户要求继续按原计划实施，已同会话启动03，
直接返回完整patch，由根Agent审核落盘。没有将准备写代码的口头说明计为实现完成。

生成预检：首次带max_tokens请求500（6.41秒）；model+messages形状的第二次请求成功
（2.73秒，294 tokens），请求`cpa-grok-4.6`，服务器返回`grok-4.6-build`。只确认此
端点报告的别名对应，不断言第一次500的原因。未换服务或模型，未发送研究数据。
2次请求全部计入项目成本；最新内存0.81 GB。收据为
`.aris/runs/dev-train1-20260910/launch.json`，明确记PREFLIGHT_ONLY，真实课程尚未启动。

### runner交付与首次验收

续接03正常end_turn，四个白名单文件完整返回并经AST检查落盘（runner/CLI/fake测试/
委派回执）。首次fake合同测试 **5/11通过、6/11失败**；不作为可运行主线验收。
主要已复现故障：192点预测输入调用了要求两段完整固定窗的旧window_context。
根Agent同时发现条末预算记账、后续评分提前、live入口尚未实现、X/y分解未接等差集，
交同一Grok会话续接04修复；不改历史runner或阈值来绕过。

该次测试实际合成fits为8：两个提前失败的合成课程各3，identity预测组件2；
手写预算测试的1 fit只是计数夹具，不算真实fit。因此截至该次测试，合成fit累计18，
项目API尝试2，真实数据fit仍0。已有失败测试目录保留，后续验收应隔离新测试运行。

难点只读审查由一次有界Opus-5/high调用承担：会话
`3cc6a310-e693-46a4-8b87-64b0ac84121e`，收据
`_scratch/grok/runs/dev_train1_opus_review_20260910_01/`。预算$5/15分钟，仅审当前实现，
0项目API/0fits，不代写代码、不增加方法Gate。此时尚未返回审查结果。

root只读核对原SEQ-3工件元数据：u9/u10/u11分别为1656/2136/2376，block均为[40:80]，
不是新runner注释所写的[0:40]。数据loader须保留该既有roster身份；此核对未加载raw数组。

### 关键只读复核返回（9/10）

Opus调用已正常返回，主模型回执为claude-opus-5（另有CLI辅助Haiku调用），报告成本
$3.520106；0项目实验API、0fits。审查结果保留在上述stdout.json，不以静态分析声称
真实数据触发概率。根Agent将以下差集合入同一runner修复：训练/预测双角色指导暴露、
Slow真实Context及训练特征、训练局部窗口坐标明确、实际执行合法性、完整拟合恢复。
训练侧共享效果仅按批次与形成段分解报告；不把评价UID损失当训练UID独立归因。

不采纳无限扩建：本包不加重复课程、退休平台或强制留一拟合；也不因不合法训练窗口
可能使共享模型不可读而擅自换identity、放宽风险线。相应失败仍记UNKNOWN，并说明
对全臂计分的影响。新几何下即使预测角色没有卡片命中，训练角色的知识修改仍可通过
共享模型影响评价；不得把这类差异机械叫作“未暴露纯噪声”。

Grok续接04工具轮数用尽、没有返回补丁，未改当前runner；05已同会话续接直接输出
修复差分。额外复核任务暂存 `_scratch/grok/tasks/dev_train1_review_followup.txt`，
等待05完成后接续，避免同会话并发写入。此时真实课程仍未启动。

05已正常返回完整修复patch（CLI模型grok-4.6-build，1轮，$0.10904616）。应用时
runner与CLI成功更新，随后测试文件上下文不匹配，故不能把整份patch记作成功验收。
已明确告知执行者实际部分落盘状态；06在同会话acceptEdits权限下接续，限定原四个
新文件并允许合成测试，不读真实数据、不调用实验API。没有重复应用已落盘部分。
最新只读内存检查为0.54GB空闲；真实课程尚未开始，不承诺四路并发吞吐。

对Opus审查的一处精度补充：独立numeric smoke已包含identity/uniform与旧
_evaluate_assignment的一致性、预测禁止solve、真输入侧y敏感对照，已在17项通过中；
该审查没读此文件，不能将这些测试写为项目整体缺失。runner自身的fake恢复/暴露
等测试仍待修好后实际验收。

06最终退出码为0，但正文stopReason=cancelled（15轮，$0.31157498），没有patch、
没有文件变更或测试，不能记作成功。原因未由收据唯一定位，不归因于用户中止。
已停止对该会话追加同类阅读轮；剩余已定位难点交既有Opus-5/high会话有界实施，
回执 `_scratch/grok/runs/dev_train1_opus_fixes_20260910_02/`（$7/25分钟，最多60合成fits，
0真实实验API/0原始数据）。原四个新包文件独占写，Grok无并发写者；不改核心或旧包。
该委派是困难接线收尾，不是换实验模型，实验仍固定cpa-grok-4.6。

### 首次真实发车（2026-09-10）

用户要求停止扩工程、优先真实效果，并明确指定4路。最后一次既有runner合同测试
11/11通过（隔离目录 `_scratch/dev_train1/runs/night_contract_m_enjmns`），本次实际
合成fits13；含此前12次root合同fit，合成累计43，API预检累计2，均计入包成本。
已用现有Windows project环境启动真实课程，PID 9032，run_id=`dev_train1_live_20260910`，
Fast并发4、Consumer拟合串行，20 train/20 eval和四臂不变。启动后已见真实拟合工件
与预算推进；该时点固定菜单校准完成7次fit、课程LLM仍0，不能称已完成Fast或Slow。
实时证据在该run目录 `budget.json`、`live.log` 与逐批检查点；启动收据沿用
`.aris/runs/dev-train1-20260910/launch.json`。尚无效果判词，不自动提交Git。

### Windows更新重启后的恢复（2026-09-10上午）

系统1074/6006事件确认TrustedInstaller发起计划更新重启，旧PID已不存在，本地8318
监听已恢复。重启前已落盘形成训练20条、形成预测60条（其中9条无效：8条传输故障、
1条PREPARE_FAILED），旧臂训练18条（17有效、1传输故障）；不是98条全部成功。
完整u9形成评分可读，u10/u11保留不完整人口UNKNOWN。Slow一次真实调用已PATCH
`bootstrap_skills.entries/build_contrastive_candidates.body`并编译：要求公开probe证据
支持后再构造处理，否则偏向identity；这只是候选指导，不是有效性结论。

原预算515次课程API、55次完成拟合，连同先前开销累计517次API/98次fit，未清零。
恢复只补后续决策；形成段故障已参与Slow证据，固定保留，不在恢复中改成新样本。
旧臂待补2条缺失及1条传输故障，补跑前原检查点与预算存入同run的
`recovery_after_reboot.json`。时间保守计至系统最后启动时刻1788982430.5，再排除
无实验进程的停机/闲置时间；保留原6小时活动运行上限，约已耗1小时44分。
两项0fit/0API恢复检查通过：累计计数与elapsed保持、已消费形成故障不重问。

恢复PID28852，Fast4路/拟合串行。首个检查已见旧臂调用98→105，而形成416、Slow1、
fit55保持，故已实证进入续跑而非重建形成段。实时结果以同run后续工件为准。
