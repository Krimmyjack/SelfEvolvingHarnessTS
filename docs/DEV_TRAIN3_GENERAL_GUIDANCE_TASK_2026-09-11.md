# 未批准讨论草案：DEV-TRAIN-3，General 决策指导的一次效果验证

**2026-09-11授权更正：用户要求交Flash的是Git保存/提交，不是后续实验。**
本文件是Astra误解范围后写出的设计草案，现仅保留供与用户讨论；下文所有实施步骤、
预算和分工均是提议，不得执行。此前“转交即执行”的表述撤回，传阅本草案不等于批准。
实际Flash任务书为`FLASH_CHECKPOINT_COMMIT_TASK_2026-09-11.md`；禁止据此启动TRAIN-3。

## 0. 工作目录、角色与要回答的问题

仓库：`C:\Users\辉\Desktop\Agent\SelfEvolvingHarnessTS-deepseek-guidance-evolution`
（WSL：`/mnt/c/Users/辉/Desktop/Agent/SelfEvolvingHarnessTS-deepseek-guidance-evolution`）。
先读工作区和项目AGENTS，尤其§5.6–5.7、§7、§9.2，以及TRAIN-2结果和本任务书。

Flash是研发执行者；**被实验的Fast/Slow仍是cpa-grok-4.6，返回grok-4.6-build**，
接口`http://127.0.0.1:8318/v1`，现有CPA_API_KEY，不打印，不替换。Windows Python可用
`D:\Anaconda_envs\envs\project\python.exe`。不再派子agent，不改参考仓库。

唯一问题：给Slow同一批合法形成经验，让它只修一处General生成/选择指导，能否让冻结Fast
不依赖当前下游probe，利用历史有效方案产生高于Static的准备收益？
这不是要求Slow必选FFT，也不是要求改动越多越好。稳定条件化和完整A5不是本包终点。

## 1. 先读事实，短核查后直接推进

必读：
- `docs/DEV_TRAIN2_RESULT_2026-09-11.md`及`artifacts/main_protocol/dev_train2_closeout.json`。
- 原运行目录`_scratch/dev_train1/runs/dev_train2_uncapped_workflow_10x10_20260910/`中的
  `slow_input.json`、`fixed_workflow_material.json`、父快照和新卡。
- `dev_train1_runner.py`的`build_slow_card/run_slow_boundary/make_slow_factory`；
  `dev_train2_workflow.py`的材料汇总、并行和评分路径；`dev_train1b_observations.RUNTIME_CAPABILITIES`。
- `docs/DEV_TRAIN1B_RESULT_2026-09-10.md`：此前已经改过General，不能宣称首次开放这一面。

只核四件事，约20分钟内写在最终报告的开头，不另造审计平台：
1. “unknown probe→identity”是否依赖当前不可获得的信息；不是把unknown改成positive。
2. bootstrap旧Support/Experience措辞是软指导、还是代码硬限制；列真正影响生成/选择的路径。
3. 七方案的成功/失败、W/V角色差异与完整人口收益是否实际进入Slow输入，是否有明确截断。
4. 本轮开放的General正文是否确实进入Fast各阶段；不存在的工具/知识不可用提示假装存在。

发现仅是文本偏好，进入下节。若解决必须改instruction.core、算子/DSL、Consumer或信息墙，
只停受影响部分并说明具体限制；不要自行扩大权限。不要重新做文献综述或整个仓库审计。

## 2. 更新设计：只改一处现有General指导

从TRAIN-2的**未学习父快照**开始，不使用那张新identity卡。与旧臂同一个cap=1.0父知识。
允许Slow在当前目录中选择一个实际可写面：
- `bootstrap_skills.entries/build_contrastive_candidates.body`；
- 或现有选择bootstrap的body；
- 或`candidate_policy.proposal_guidance` / `candidate_policy.selection_guidance`。

只一次局部PATCH；不ADD、不同时改多个面。此限制是本包明确的General更新消融，不是永久
禁止Specific。保存实际目录、完整输入和原始可解析响应/失败收据，先落盘再编译，避免TRAIN-2
丢失提议字段。凭据不入日志。已有controller、store、compiler照用，不另建哈希或存储体系。

给Slow的任务说明应表达以下原则，不写具体赢家：
- 目标是指定Task/Consumer下的准备价值，不局限于证明污染后修复。
- 历史真实效果可以支持一个暂定默认Workflow；不要求当前无反馈Fast再次取得positive probe。
- 未知不是测得负值，合法也不是必有收益。允许identity，但不能以不可获得的反馈作为唯一行动条件。
- 根据已测完整W/V策略对照，提出训练/预测配合及观察/构造/选择指导；它们的效果不能相加归因。
- 保留仍有效的路径；没有条件化证据时可以学习通用默认，不编造阈值或强行细分。
- Fast只能用现有部署可见观察、程序语义与合法性试执行；不得读当格效果。

材料复用TRAIN-2形成段，不新跑40次形成决定，不把u12/u13结果或本结果报告传给Slow。
给全部七方案的typed steps、三形成窗的每UID收益、汇总风险和W/V分解；保留旧Agent成败及
两条失败的UNKNOWN。允许去重机械窗口清单、提供紧凑全表；不得按赢家筛选、只保留总分或
把未测项补零。压缩规则在调用前写定；报告保留源字段映射，不能假称与TRAIN-2提示完全相同。

一次内容提议机会；正常no_proposal即保留并收口。协议/传输错误只按原有有界重试，合法但
看起来保守的内容不得反复催写。若又写unknown→identity，记下来；不人工改答案，不把它变成
新拒绝门。合法且送达的候选仍按计划完成效果对照。

## 3. 最小完整比较与资源

沿用TRAIN-2固定的10 train/10 eval人口、anchors、训练prefix900、Ridge、指标、cap=1.0。
形成u9/u10/u11（1656/2136/2376），检查u12/u13（2616/2856）；禁止origin≥3096、sealed、
TARGET_HELD_IN、+144。全部是已曝光development；本包受既有结果启发，不宣称独立确认/新数据泛化。

四臂：Static；形成段校准固定Workflow；旧General；一次Slow改后的General。
固定臂由同一七方案形成材料决定，不能按u12/u13改赢家。Static/固定的冻结模型与评分可在
配置逐字段一致时复用原收据并标cached；历史模型不是本包新fit。
旧/新Agent必须本次同配置重新生成：每臂10条训练W→一个真实共享模型→两窗各10条V。
两臂共60次Fast，不拿TRAIN-2的旧臂随机实现冒充本次同步对照。每次无跨轨迹Episode。

全局4个Fast会话（不是每臂4个），fits和缓存提交串行；checkpoint/预算单写者或锁。
两臂交错调度，原始有效决策不重问。全部新旧输出冻结后再评分；已缓存基线后续分数不入
模型提示、不决定是否继续。恢复按父/候选快照和已有决定，不重问已完成Slow。

上限：1000次实验API尝试、30次新fit、3小时活跃运行（人为停机另记，不能静默归零）。
预计约300次Fast API、一次Slow、两次新臂fit；这是估计，不是实测。不要浪费一次完整试跑
再从头正式跑。普通失败保留；账户/身份/持续传输异常及时停线，不填identity/0。

只做一次必要的0-LLM接线检查：General修改进入真实Fast视图，失败仍UNKNOWN，预算续接、
原两臂父状态相同、待比较输出冻结前无评分回传。复用已有四路/串行fit检查，不重复全仓测试。
送达0或无修改就记utility untested，不强行开始无处理差异的两臂。送达后不因结果正负中途停。

## 4. 必须给出的结果，而不是工程交代

同窗全10条人口的新减旧效用，再对两窗等权；同时报相对Static/固定、受损次数、最坏伤害、
训练W与预测V、模型是否变化、实际改动量、成本。identity零必须区分模型未变与共享模型变化。
基线的风险也报；缺测整窗UNKNOWN，可读子集只作诊断；不以多步率/非identity率代替效用。

按结果作选择：
- 新臂再次等同Static：仍是减害/退出策略，本次General更新未学到正增益，结束本包，不续第3次。
- 新臂高于Static且优于旧、仍输固定：有正准备收益，但条件化优势未成立。
- 接近固定且主要普遍使用同一Workflow：最多支持有效默认策略学习，不包装成逐序列条件化。
- 超过固定且风险可接受：一组开发线索，下一阶段才换训练材料检验，不马上声称稳定/完整A5。
- 无修改、没送达、故障缺测：明确哪一环未测，不当方法负结果也不靠无限补跑追正号。

这个包同时改变Slow的材料呈现/目标说明和开放surface，因此与TRAIN-2的差别是**整个修订
方式的开发变化**，不能把差值全归因于General、压缩或某一句话；同包新旧比较只识别该候选
知识的部署差异。没有普通反思/去反馈臂，不证明特定归因格式或真实反馈的独立因果贡献。

## 5. 修改边界、交付与交接

优先仅新增`evaluation/main_protocol_p4/dev_train3_general.py`和本包launcher，复用已有模块；
必要的跨包共用修复必须说明具体缺陷和最小diff，不能顺手改已收口runner或重写历史工件。
禁止改canonical h0、runtime/、methods/、operators/、contracts/、评分公式、风险阈值、模型或
数据边界；候选知识只能经既有隔离分支controller写入。不修改AGENTS、DECISIONS或Flash地形工件。

工作材料集中`_scratch/dev_train3_general/`，一个结果报告
`docs/DEV_TRAIN3_GENERAL_GUIDANCE_RESULT_2026-09-11.md`，一个紧凑结果工件
`artifacts/main_protocol/dev_train3_general.json`；更新STATE_ONE_PAGE置顶并给Fable事实回执。
报告先给结果和是否解决当前问题，再给实现；没有运行就明确未运行。不提交/push Git，除非
用户另行授权。最终给Astra：四臂数值、实际改的正文、W/V差异、成本、未解决问题与下一选择。

无需每步等待Astra审核；只有信息泄漏/计分不可比、超预算、需要越权修改才暂停相应部分。
