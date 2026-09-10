# DEV-TRAIN-2：扩大合法动作后的 Workflow/Skill 小型效果对照

2026-09-11完成回执：四臂完整，报告`DEV_TRAIN2_RESULT_2026-09-11.md`。
新Skill全部identity，收益0，优于旧Skill的−0.164690但低于固定组合+0.849895。
500次实验API/10fits；最终60条决定有效，形成段两条失败按原证据保留。
下方发车/恢复记录是历史；本包不再续跑。下一任务另见TRAIN-3任务书，尚未发车。

状态：用户已批准继续推进（2026-09-10，“你推进把”），并要求比较 baseline 并行。
根 Agent 定本规格；2026-09-10 21:33（UTC+8）已发车，尚无效果结论。旧包、Flash探索及收据不覆盖。

## 唯一问题与四臂

在取消通用35%修改比例门、真实“异质训练准备→一个共享模型→逐条预测准备”的同一环境下，
Slow一次自主知识修改能否让冻结Fast取得比旧知识更好的完整人口收益/风险？
四臂：Static；形成段选定的固定Workflow；旧Skill；Slow一次修订后的Skill。
优先质量改善，相近质量下风险改善亦有价值。胜固定不是本轮必然预期，也不凭单次胜出自动证明条件化。

## 不变配置

- 复用TRAIN-1B各角色冻结roster的前10条：10 train、10 eval，非按效果筛选。
- 含缺失KDD曝光development；anchors、192输入+48历史目标、prefix900、Ridge、sMASE不变。
- 形成u9/u10/u11（origin1656/2136/2376）；检查u12/u13（2616/2856）；不得读≥3096、密封、+144或TARGET_HELD_IN。
- 每个臂的新训练赋值必须真实拟合一个共享模型。预测输入也逐条生成Workflow；不固定预测为identity，不路由不同程序模型的行。
- API cpa-grok-4.6 @ http://127.0.0.1:8318/v1，要求实际返回grok-4.6-build。使用现有CPA_API_KEY，不打印，不默换模型。
- 继承TRAIN-1B实际窗口观察与能力说明，所有Agent臂一样。General/Specific和当前白名单保留，Fast无当格Outcome或跨轨迹Episode。
- 所有臂明确maximum_modified_fraction=1.0；旧.35数据不复用为新权限决策。canonical h0不改，配置变更与学习修改分开。
- 预算沿现有包上限2000 API attempts、500 CPU fit attempts、6h；预计100个Fast决定、一个Slow修改机会。故障与恢复开销照实计，不逐序列设新小帽。

## 有限固定菜单与Slow材料

固定菜单预先列七个（均用现有算子默认参数，不扩DSL/schema）：identity、outlier_iqr、hampel_filter、
outlier_iqr>winsorize、winsorize>hampel_filter、fft_decompose、winsorize>fft_decompose。
这是受Flash探索启发的有限菜单，不声称全部合法程序的最强基线，也不限制Fast已有1–4步生成空间。
形成段完整人口均值最优者冻结为固定Workflow，同报风险；不按后续窗口选择，不硬指定FFT。

在本次10/10训练材料重新拟合各统一Workflow，用其真实模型测形成段：W=P,V=P；W=P,V=identity；
W=identity,V=P。模型可在这三种读法间合法复用，无逐行跨模型拼接。这些是不同可执行策略的条件比较，
不是可加的训练/服务贡献，也不是规定训练/预测必须相同。固定菜单模型可以与对应读数复用，不重复拟合。
identity参考必须是同UID同窗的Static损失；不完整全人口记UNKNOWN，可读子集只作诊断。

Flash的slow_material_workflow_differences.json只作研究参考，不原封接入：人口不同，部分单步标签引用双步模型，
且“FFT全面占优”等解释不成立。本包按实际声明的typed_steps重新生成可追溯的形成段数值及逐UID读数。
形成材料给Slow：真实各方案效果、原Agent的W/V和成败、执行窗口观察、旧Skill及修改目录；不提供u12/u13结果，
不提供Flash预测/排名结论，不告诉它必须选择FFT。旧/新Fast都只经冻结知识获信息，不读取该材料表。

## 课程与并行

1. Static先拟合并取得形成参考。固定菜单/训练服务差分的CPU工作与旧Skill形成段Fast调用重叠。
2. 形成段Fast各训练序列生成W，汇总拟合；三形成窗口逐条生成V并外部评分。
3. 等形成反馈和预定菜单读数完整后，只给Slow一次自主有限修改机会；不强制ADD、PATCH或弃权。
   编译失败/无修改如实记录，不反复催出候选。合法修改跑既有送达检查；零送达不装成有效处理臂。
4. 边界后旧/新Skill可同时生成各自训练W、各自汇总拟合、各自生成两检查窗口V；所有输出冻结后才统一评分。
5. Static/固定的后续预测可以提前计算冻结，但后续真值评分不得先进入任何Fast/Slow或影响继续策略。

并发规则：整包最多4个活跃Fast会话（不是每臂4个）；每序列内工具调用顺序保留。模型拟合及其缓存提交单队列，
可与网络阻塞的Fast重叠；CPU预测评分不额外启动一堆进程。共享budget/checkpoint按稳定键、锁/单写者合并。
不得按任务完成顺序更新知识或改变roster。普通并发错误当包修复；不启动分布式调度/新缓存/新哈希平台。

## 读数与结论

主读数：每窗完整10人配对的新Skill减旧Skill效用，再对两窗等权；各臂相对Static及固定同时报告。
同人口同权重下，“先逐UID做差再平均”和“各臂均值相减”等价，不把变更算式当新方法。
报告受损次数、最坏伤害、覆盖和成本，以及知识送达、W/V变化、共享模型变化；训练UID不按评价UID损失直接归罪。
两个窗口不是独立样本充足的显著性检验，不采用Flash的MDE0.2–0.35作为可靠功率结论，不宣称稳定显著性。
若新Skill普遍学会一个更好默认Workflow，可记默认策略学习；即使胜该有限固定菜单，也须排除更好固定组合后才谈条件化。
打平/落后也不证明条件结构不存在。已有窗口均曝光，本包不支持密封泛化/完整A5。

停止：账户/模型身份故障、不可恢复错误、总预算/时间上限；保存完成项只补缺项，不重问已完成Slow；
无treatment按既有规则收口。不得根据途中正负停、删人或追加种子到转正。

## 交付

一个局部实验helper、一个launcher、一次必要的并发/信息边界fake检查；根复核后启动。
运行目录_scratch/dev_train1/runs/dev_train2_uncapped_workflow_10x10_20260910；launch receipt沿用.aris/runs/。
最终一个主报告含四臂表、同窗配对、知识/行为、真实成本和范围；给Fable事实回执，不代写DECISIONS。
普通实施Grok；根负责方法、关键复核与发车。不commit，不修改Flash工件，不以检查数代替效果。

## 实施回执（不是实验结果）

2026-09-10：cap已接入RunSpec、实际Fast快照、训练和预测合法性检查及恢复口径。
`_scratch/dev_train1/test_uncapped_contract.py`通过六项真实机械检查（0实验API、1合成fit），
包括canonical h0不变、FFT实际可行动、训练/预测一致及跨cap恢复拒绝；根外只读复核未发现阻断问题。
新launcher必须显式传`maximum_modified_fraction=1.0`。历史Flash脚本仅赋值
`R.MAX_MODIFIED_FRACTION=1.0`已不足以影响RunSpec；历史结果保留，新复算应显式配置，勿静默按旧脚本重跑。
四臂helper/并行入口仍在准备，效果实验尚未启动。Grok早先两次编辑请求被非交互权限取消，
不计作实现成功；根通过正常编辑工具完成cap补丁，未放宽委派权限。

21:33更新：根接收Grok未落盘的helper草稿并完成修正与launcher。合成编排检查通过：
0真实API、4合成fits，实测全局Fast峰值4/fit峰值1、形成齐全后Slow、后续全部冻结后评分、
恢复0额外调用/fit；`_scratch/dev_train1/train2_tests/summary.json`保存收据。
并行helper的额外独立复核因所用模型容量不足未完成，不记为通过；根的本地复核及上述测试已完成。
效果运行Windows PID=3320，启动收据`.aris/runs/dev_train2_uncapped_workflow_10x10_20260910/launch.json`。
启动后核实进程存活、DATA_READY、预算持续前进；Static加六个菜单模型已完成7fits，
固定基线直接复用形成段获胜模型，正与形成段Fast调用重叠。此为发车证据，不是效果结论。

22:30恢复回执：原进程22:09因Slow边界JSON写入遇嵌套mappingproxy退出；子卡此前已真实编译并存储。
局部修补`dev_train1_runner._json_default`支持Mapping，TRAIN-2恢复时直接复用原`slow_input.json`。
`_scratch/dev_train1/prepare_train2_recovery.py`先验证唯一父子编辑为ADD，再归档原失败材料，
由现存子快照/provenance补回边界包装。原始Slow响应、行为预测/尝试日志未保存，明确UNAVAILABLE；
不伪造、不重问。真实恢复路径验证同一子快照、40条形成段均不重问、16份原文件字节不变，0 API/0 fit。
其中u11/T141传输失败、T149准备失败已被Slow作为UNKNOWN读过，不重跑这两条，不改变形成证据。
子卡`missingness-unknown-probe-identity`保持原文与条件，不人为改成FFT指导。
22:29 Windows PID=7576按`launch_train2.py --resume`启动；已核实旧/新两臂各新增2次调用，
形成195/Slow1/fits8未重计。剩60次Fast，沿原4路/预算/终点。归档与测试记录在
运行目录`before_serialization_recovery/`及`serialization_recovery.json`；启动收据保留previous_launch。
Grok只读核对在8轮上限耗尽后未给结论，不能作为独立通过；根恢复实测通过。尚无效果读数。
