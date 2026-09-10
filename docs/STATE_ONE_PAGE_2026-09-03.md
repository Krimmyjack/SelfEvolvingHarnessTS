# 当前状态一页（始建 2026-09-03；最新更新置顶；所有执行线先读此页；账本只作归档）

## 2026-09-11：TRAIN-2 已完成；Flash仅负责Git归档，后续设计待讨论

主报告：`docs/DEV_TRAIN2_RESULT_2026-09-11.md`；可提交证据：
`artifacts/main_protocol/dev_train2_closeout.json`。23:09（9/10）exit0，最终新旧60条决定
全部有效，四臂两窗全10人评分无缺项；不是还在运行。

| 四臂（10 train/10 eval，两窗） | 平均收益对Static | 受损次数/20 |
| --- | ---: | ---: |
| Static | 0 | 0 |
| 形成段校准winsorize→FFT | +0.849895 | 1 |
| 旧Skill | −0.164690 | 14 |
| 新Skill | 0 | 0 |

新臂30/30明确选择identity，模型及预测等同Static。新减旧+0.164690来自退回基线，
不是已经学到高于Static的准备价值。500实验API/10fits，全局4路Fast、fit串行。
形成段2条失败仍UNKNOWN、原始Slow响应缺失及序列化恢复均披露；不能说整包无故障。
判断：骨架按用户目标运行；有效Workflow存在，但本次知识更新走向全面不处理。
新卡的unknown-probe回退规则与本环境probe始终不可得直接冲突；成功方案读数已在Slow材料中，
其注意/解释/采用问题尚未被独立归因。一个组、两个开发窗，不主张稳定条件化或完整A5。

用户澄清：**交给Flash的是保存/commit任务，不是下一轮实验**。当前交接文件为
`docs/FLASH_CHECKPOINT_COMMIT_TASK_2026-09-11.md`。Astra尚未stage或commit，HEAD仍c845b4b。
`DEV_TRAIN3_GENERAL_GUIDANCE_TASK_2026-09-11.md`已降为未批准讨论草案，不得执行；
后续机制、对照和预算由用户与Astra继续讨论。备忘录§19同为建议，非已批准路线。
DECISIONS仍由Fable维护，本次不代写；旧状态段保留为历史。

## 2026-09-10 22:30（UTC+8）：DEV-TRAIN-2 已恢复，新旧 Skill 对照正在运行

原进程在 Slow 已编译新卡后，因 JSON 不支持嵌套 `mappingproxy` 于22:09退出；不是 Slow 弃权。
已修实验层 JSON 序列化，并从现存子快照及父子 provenance 恢复边界包装，未重问 Slow、未改卡正文。
缺失的原始提议响应、行为预测和尝试日志标为 UNAVAILABLE，不从最终卡片反编造。
原失败 result、预算、Slow 输入、40条形成段记录及其评分已归档于运行目录
`before_serialization_recovery/`；`serialization_recovery.json` 记录0 API/0 fit恢复核验。
形成段38条有效、2条失败已被 Slow 消费，失败仍为 UNKNOWN，恢复不重跑或改写这些材料。

22:29恢复进程 Windows PID=7576，沿原运行目录及原包级预算；已核实 DATA_READY 和两臂调用同时前进。
预算由196增至200次尝试（形成195、Slow1、旧臂2、新臂2），fits仍8；Slow计数保持1。
余下为旧/新各10条训练W及两个窗口各10条预测V，共60次Fast决策；全局4路、fits串行。
Static/固定Workflow已完成形成段校准与后续预测冻结；所有后续输出齐全后才统一评分。
形成段胜出的固定组合是 `winsorize>fft_decompose`，仅是校准结果；当前尚无新旧Skill效果结论。
恢复核验为根本地检查通过；Grok只读恢复复核在8轮上限前未交付结论，不计独立复核通过。

## 2026-09-10 21:33（UTC+8）：DEV-TRAIN-2 已发车，四臂同权限效果对照

任务：`docs/DEV_TRAIN2_UNCAPPED_WORKFLOW_SKILL_TASK_2026-09-10.md`。
用户已批准继续推进，并要求baseline并行。10 train/10 eval；逐训练序列W→共享Ridge→逐预测序列V；
Static/形成段校准固定Workflow/旧Skill/一次Slow修订的新Skill四臂；同用cap=1.0。
API cpa-grok-4.6，本地8318入口，逐响应校验grok-4.6-build；全包4路Fast、拟合与模型提交串行。
形成段u9/u10/u11，复核u12/u13；复核全部输出冻结后才评分，不读密封或≥3096。
已核实Windows进程3320及前进中的预算；Static+六个菜单模型7fits已完成，与形成段Fast重叠。
运行目录`_scratch/dev_train1/runs/dev_train2_uncapped_workflow_10x10_20260910/`；
启动收据`.aris/runs/dev_train2_uncapped_workflow_10x10_20260910/launch.json`。
尚无新旧Skill效果结论。上一包结果见`docs/DEV_TRAIN1B_RESULT_2026-09-10.md`；
以下9/8等状态均为历史，不代表当前待办。未commit，不代写Fable的DECISIONS。

## 2026-09-08（当前行动）：DEV-DEPLOY-1 已获同意、任务书就绪，执行回执待补

任务书：`docs/DEV_DEPLOY1_FAST_ONLY_ALIGNMENT_TASK_2026-09-08.md`。
用户本轮指令：“好的，先继续推进，发第一包，然后同步落实相关的文档”。
**本次只落任务书与文档，未启动实验、未调用项目 LLM/Consumer、未提交 Git。**

- 当前是机制开发，不是“只差一个正结果”。逐序列执行、窗口绑定/去重、有限修改与
  指导改变局部 select 已有证据；知识修订改善最终零反馈部署仍未获支持。
- 第一包不让 Slow 写新卡。核心：原选择 vs Support 实际交付的同历史影子；冻结
  Fast-only 真运行；真 0-LLM 的确定性固定菜单基线。全部只用已曝光 KDD development。
- 正式规模：两次重复 × u9/u10/u11 × 20 条 = 120 次 Fast 决策，sol，先冻结所有
  输出再外部评分。主表为 origin 当前预测，+48 保持程序的持续性单列；旧 SEQ 读数
  与旧 delayed 主终点不改。详见任务书，不从本页推导额外授权。
- 菜单 oracle 与效应尺度辅助解释，不设静态特征识别硬门；不把窄菜单上界称为整个
  学习环路上限，不把 MATERIAL 超线频率叫假阳性率。
- **2026-09-15 检查点**：按实际证据决定继续、换明确机制/环境或停止当前机制小修。
  第一包后才冻结第二包；最终完整 Static/A3/A5 不取消。
- 协作已同步 `AGENTS.md` §9.1：Astra 定问题/任务书并整合证据，Fable 做系统整合/
  可行性/反方核查并后续维护决策日志，Opus 整包执行，用户决定路线和重要授权。

**SEQ-4 单独收口，不扩成新卡片实验。** 当前磁盘工件
`artifacts/main_protocol/dev_seq4_revision.json` 是 `PARTIAL_MORE_CELLS_TO_RUN`：仅
u12/knowledge-frozen 完成，缺 u12/slow-candidate 与 u13 两臂，主读数 UNKNOWN。
这不是进程仍在运行的证明。若按另行确认的续跑执行，恢复 u13 必须先承接同臂 u12
历史；直接 `--positions 13` 只播种至 u11 的旧路径有缺口。以下旧状态段作为历史保留。

更正下方历史措辞：“53/116”是选择与交付一致率，不是选择的因果贡献率；窄暴露
意味着难辨别小的全人群增量，不是数学上绝不可能有效。“无 select 否决权”是当前
Support 选策机制的局部性质，不能直接推广到冻结 Fast-only。

## 2026-09-08（最新）:DEV-SEQ-4——**指导确实改变了决策,但 select 改不动交付;Slow 自己改了那张卡;第四步被内存杀掉未完成**

报告 `docs/DEV_SEQ4_GUIDANCE_ADOPTION_RESULT_2026-09-08.md`;工件
`artifacts/main_protocol/dev_seq4_same_start_contrast.json` 与 `_scratch/dev_seq4_revision/pkg4/boundary.json`;
源 `evaluation/main_protocol_p4/{run_dev_seq4,run_dev_seq4_revision,dev_seq4_knowledge,smoke_dev_seq4}.py`。
**⚠️ 本包不是完整实验:第四步(复核对照)被宿主 OOM 连杀三次,未跑完,标为提前终止。**

- **先更正 DEV-SEQ-3 三处**(原报告加第八节,原文保留):①"Fast 完全没使用指导"**不成立**——
  B 臂 select **3/3 选 identity**,A 臂 1/3,而两臂在 57 条未匹配决策上**同为 8/57**;初版只比了
  `deployed_label`/Support/delayed,没比 `chosen_candidate_id`/`candidate_programs`/`tool_calls`。
  ②"全人群负差全是纯采样"当时**没有根据**(历史经 `method.py:374`→`fast_agent.py:804` 进 Fast),
  但**事后实测基本成立**:60 对里 Episode 条数不同 25 对,**渲染进提示的文本**不同仅 6 对,
  暴露决策上 0 对;51 对严格同知识中 **41 对完全相同**。③ SEQ-2 口径是**每组 18/合计 36**,不是 0/60。
- **⚠️ 真正的阻塞点(架构,不是不服从)**:`online_loop` 里 `chosen_candidate_id` **只定探测顺序**,
  **首个通过 Support 准入的候选取得部署权**。120 条实测:选 identity 的 20 条中运行时**照样部署 12**,
  点名候选的 96 条中改部署另一个 18、弃权 33——**agent 的选择只决定 53/116 = 45.7% 的交付**。
  **任何经 select 起作用的指导都不可能阻止一次部署。** 但那 12 条 delayed 合计 **+3.5842**
  (7 正/5 负),**Support 门并非单向有害**,不得据此说运行时该改。
- **同起点小对照(第三节,完整)**:6 情境 × 2 臂 × 3 重复 = 36 次决策,两臂拿**同一份**决策前历史
  (按 DEV-SEQ-3 检查点重建,60 条逐条核对无误)。**163 LLM / 0 fits / 1211s / 0 故障**。
  检索 9/9 对 0/9;**u10/T146 选择 0/3→3/3 干净翻转**(另两条匹配情境 +1/−1 相抵,未匹配基线抖动 ±1,
  所以只认这一条,不认"匹配面 5/9→8/9");**交付 36/36 完全相同**,格内 3 次重复 delayed 也完全一致。
  **u11/T14 无卡时 A 也 3/3 选 identity 却照样被部署**(delayed −0.19238)——覆盖是架构性质。
- **Slow 自我修订(第四节,完整)**:输入 86 条 Episode + 上一版原文与它自己的三条预测 + 过程差异 +
  **交付机制事实**;目录 8 个面(含它自己那张卡的 `.body`/`.observable_applicability`),
  明说保持不变是合法答案。它选了 **PATCH 自己那张卡的正文**(`SKILL_CONTENT_GAP`,一次提议即成),
  **删掉了架构上不可能实现的 `identity_retained` 预测**,把"selection alone is not evidence that the
  repair was prevented"写进正文,并把着力点从**选择**移到**候选提议**,同时明确把准入权留给 Support。
- **第四步未完成**:u12/u13 × 两臂 × 20 = 80 次决策,被宿主 OOM 三连杀(空闲内存 0.45 GB/15.7 GB,
  占用来自用户自己的 WSL/Cursor/浏览器;我方 python 当时 0 MB)。**环境限制,非模型/代码故障**。
  修订的 applicability 未改,在复核窗口只命中 **1/40**(u12 0/20、u13 1/20),
  **主终点在这个宽度下必然没有分辨力**——此判断在开跑前已确定。
- **本包修掉我自己的第七处缺陷**:**预算记账不抗杀**——原只在正常退出时落盘,OOM 不走 `finally`,
  被杀那次的花费整个丢失、下一进程少算总额。已改为每步落盘。**因此两次被杀的 Fast 支出未精确记账,
  报告中如实标注,不用估计值冒充实测。**
- **恢复路径已就位**:`resume_boundary()` 按 SHA 重编译修订并校验哈希,**恢复绝不重调 Slow**;
  `--positions/--arms` 支持一格一进程;预算跨进程承接。腾出内存后四条命令即可续跑。

## 2026-09-08（更晚）:DEV-SEQ-3 收口——**指导第一次真的送到 Fast,Fast 也照做了,但 select 无权改变交付**

判词 `DELIVERED_AND_LOADED__AGENT_COMPLIED__SELECTION_LACKS_DELIVERY_AUTHORITY`
(初版为 `…DELIVERY_UNCHANGED__UTILITY_NOT_SEPARABLE`,逐决策工件复核后更正,见报告第八节)。
报告 `docs/DEV_SEQ3_GUIDANCE_DELIVERY_RESULT_2026-09-08.md`;方法变化写入 `AGENTS.md` §5.4 第八条;
工件 `artifacts/main_protocol/dev_seq3_guidance_delivery__{g1,g2,contrasts}.json` 与
`dev_seq3_exposure_check__g1.json`;源 `evaluation/main_protocol_p4/{run_dev_seq3,smoke_dev_seq3}.py`。
**540 LLM / 60 fits / 2.65h;上限 2000/2000/6h 未触顶;形成段复用 DEV-SEQ-2 SOL 历史,未重跑未重计费;
核心文件编辑 0,未提交 git。**

- **阻塞点推过了"使用",落在权限边界上**:SEQ-1 条件被验证拒;SEQ-2 合法卡**从未被呈现**
  (两组各 0/18,合计 0/36——**不是 0/60**,60 是 SEQ-3 恢复全人群后的分母,两者不可相减);
  **SEQ-3 呈现了(g2 3/60)、也被采纳了,但交付未变(0/3)**。g2 的卡
  `missingness_deviation_identity_guard` 出现在 3 个目标决策的 `retrieved_skill_ids` 里(A 臂 0),
  且 **B 臂 select 阶段 3/3 选 identity**,而两臂在 57 条未匹配决策上的 identity 选择率**同为 8/57**
  ——模型**照做了**。交付没变是因为 **select 对候选池没有否决权**:全部 120 条决策里,agent 选
  identity 的 20 条中运行时**照样部署 12 条**,点名候选的 96 条中改部署另一个 18 条、弃权 33 条,
  **agent 的选择只决定 53/116 = 45.7% 的最终交付**。T146 上 B 选了 identity,运行时仍部署
  技能库供给的 `outlier_mad`(Support +0.0699 准入,delayed −0.0778)。
- **⚠️ 结构约束(优先于下一次卡片实验)**:**任何经 select 阶段起作用的指导都不可能阻止一次部署。**
  但那 12 条覆盖的 delayed 合计 **+3.5842**(7 正 +4.2745 / 5 负 −0.6903),**Support 门并非单向有害**,
  不能据此断言运行时该改。
- **g1 零暴露,失败方式与 SEQ-2 不同**:不是某条子句永远为假,而是 `mf>0.5`(只有 T14)与
  `lmrf>0.05`(只有 T146/T16)**落在互斥序列上**,合取为空。两条是从形成段一起读出的(u7 上 T14 同时满足),
  到臂窗口就分开了。
- **方法论约束(已按复核收紧措辞)**:主终点 −0.004644(u10 −0.0208、u11 +0.0115;u9 −0.0146 是验证段,
  不进主终点)里 3 条暴露决策的**算术**贡献为 0。初版据此断言"全部是纯采样"**当时没有根据**——两臂
  Episode 历史已分叉,历史经 `method.py:374`→`fast_agent.py:804` 进入 Fast 上下文。**事后实测了这条通路**:
  60 对里 Episode 条数不同的有 25 对,但注入 Fast 的**渲染文本**不同的只有 **6 对**,且**3 条暴露决策上
  逐位相同**(渲染 596 字符,非空)。所以暴露读数不受历史混淆;未暴露面 51 对严格同知识(mean −0.0093、
  **41 对完全相同**、6 对达 MATERIAL),被污染的 6 对贡献近乎零(mean −0.000665)。**离散是集中的,不是
  弥散的**——初版"噪声比信号大一个量级"的说法也据此收窄。**全人群终点保留为主读数**(它回答"整套机制对服务人群的净影响",适用面窄
  本身是系统表现的一部分),暴露子集与过程差异**加报**,不事后换口径。**行为链必须按暴露拆开读**:
  未暴露配对本身就有 70% 观察不同、68% 候选池不同、19% 交付不同、9/57 达 MATERIAL。
- **三项修补都接入并有实测**:①特征说明改为公式+窗口、删掉未证明的漂移话术,并区分特征窗(全历史)/
  服务窗(仅 192 点)/修改区(该词汇不测量)——**g1 的卡正文原样引用了这条区分**并据此把结论降级为
  "提示性而非证明",但**仍把全历史量写进条件**;②动作事实由既有 `_prepare` 重放取得,烟测证实训练侧
  与评价器自报 `behavior_point_count` **逐位相等**(T14@1656 `outlier_mad`:服务窗 5/192,训练 778/48000);
  ③批内并行实测 **×3.74**、71s/决策、16s/调用、**0 故障**,读数与串行逐位相同、不多花 fit。
- **采用标记未通过**(配对增益未达 MATERIAL 且 B 未过 u9 权威门);按预定未采用的 B 仍跑完 u10/u11 作
  隔离诊断,不计晋升/部署收益。u11 两臂**都过**权威门(A +0.2644 / B +0.2759,固定 cohort 答案 +0.2567)。
- **本包修掉我自己的六处缺陷**(均为"把可恢复当不可恢复"或"分母静默走样"):403 误判为账户错误;
  臂人口被静默截断成 10 条;形式错误与实质越权混为一谈;`run_or_resume()` 缺参数致进程崩溃;
  中转暂态 400 不在共享重试范围;行为链在全人群上计数会把采样报成行为改变。失败收据全部保留未覆盖。
- **下一项只一个(未启动)**:在**同一条已暴露序列**上,把卡片正文从"建议"改成 Fast 必须显式回应的
  **核查项**(select 阶段须说明是否采纳及理由),其余不变,直接测"呈现→采纳"的断点。不改裁判、
  不改风险线、不扩编辑权限;不启动第三组、消融或密封实验。

## 2026-09-08:DEV-SEQ-2 收口——**Slow 两次都写出了合法修改,两次都把它挡在了自己描述的序列之外**

判词 `LEGAL_EDIT_COMPILED_BUT_NEVER_PRESENTED__UTILITY_UNTESTED`(两组独立复现)。
报告 `docs/DEV_SEQ2_SLOW_UPDATE_TO_FAST_RESULT_2026-09-08.md`;方法边界写入 `AGENTS.md` **§5.4**
(七条,追加式,不改写 §5.3);工件 `artifacts/main_protocol/{dev_seq2_slow_update_to_fast__g1,__g2,
__contrasts,__g1_flash,__g2_flash,__contrasts_flash,dev_seq2_exposure_check__g1,__g2}.json`;源
`evaluation/main_protocol_p4/{dev_seq2_knowledge,run_dev_seq2,smoke_dev_seq2,audit_dev_seq2}.py`。
**主线 206 LLM / 40 fits / 2.55 h(两组独立账本相加);探针线 195 / 36 / 2.26 h(累计账本);
上限 2000/2000/6h 无一触顶。核心文件编辑 0,新增哈希平台 0,历史收据覆盖 0,未提交 git。**

- **主结果**:两组的 Slow 各自从合法反馈里**自己选面、自己写出一处有限修改**,写的是**两张不同的卡**
  (g1 `high_missingness_linear_imputation`;g2 `dense_missingness_outlier_caution`),都经真实 controller
  编译进隔离分支(`98dea3b0…`→`ee99ac2d…` / `1d4b7677…`),父版逐字未动、旧知识一条未删。
  **但两张卡在此后每个预定臂窗口都检索不到:0/18 与 0/18,合计 0/36。**
- **两组卡死在同一个子句**:都把 `longest_missing_run_fraction ≥ 0.05` 写进检索条件,而该特征分母是
  **整段已观测窗口**,窗口随课程单调变长 ⇒ 比值单调衰减(T14:0.069→0.057→0.049→0.038→0.034),
  **该子句两组都 0/18**。其余子句都可满足(g2 的 `local_robust_z_peak≥4.0` 满足 15/18)。
  **卡片写完即自动退休。** 注意:比例下降本身不等于条件写错;已确认的是**尺度错配**——两张卡正文说的
  都是"局部区域",条件用的却是全历史分母的量。**未**证明"改用局部量就会更好"。
- **阻塞点比 DEV-SEQ-1 更靠前**:DEV-SEQ-1 是"条件被预注册验证以覆盖不足拒绝";本包把覆盖门降为诊断、
  改用真实两臂做判据后,暴露的是**"呈现"这一环**——Slow 写得出合法修改,但它没被送到它描述的序列面前。
  因此"可见反馈里是否含可复用信息"这个真问题**本包仍未触及**。
- **事后新增的开发级停止规则(已留痕)**:原任务书未区分 (a) 读到了指导但效果不好(应跑完)与
  (b) 剩余样本全读不到(继续只是重复父知识)。本包两组均属 (b),故新增零成本暴露检查
  (`exposure_check`,在父/候选两快照上渲染 Fast 实际读取的字节,0 fits/0 LLM),命中为 0 则两臂不跑,
  记 `COMPILED_BUT_NEVER_EXPOSED__UTILITY_UNTESTED`。**不是覆盖门**,只问"要花钱测的东西有没有被呈现"。
  **这两组不得作为完整两臂比较呈现,后续也不得只挑指导确实加载了的组展示。**
- **两项工程修复已接入并可核**:窗口绑定 `(series_uid, origin)`(两组各 50 窗口/10 序列/**0 miss**);
  测量去重按预测缓存自身语义键(40→40、34→34),**不删 Episode**、不新增哈希。
  **重算 DEV-SEQ-1 run2 的 L2**:覆盖由"6 条序列中 1 条"变为"7 个失败窗口中 2 个",**L2 仍不过**——
  错绑真实存在但不是当初拒卡的原因,修复**不构成**批准旧卡的理由。
- **可编辑面从 1 个扩到 4 类对象**(新增指导卡 ADD / bootstrap 正文 PATCH / `candidate_policy.
  proposal_guidance` 与 `.selection_guidance` PATCH),六个实例全部呈上,cause 由所选面机械推导,
  五类故障标签退为统计标签不再否决。**两组 Slow 都选了 ADD 新卡**,即使 g2 的内容本质是一条选择层
  建议——写进 `selection_guidance` 本可对每个决策无条件可见,写成情境卡就被条件挡住。如实记下,未追。
- **探针线(flash,单独成线不并表,用户明确要求)**:两组都成组(g2 甚至成 2 组、12 条成功对照),
  但 **Slow 两次都弃权**(`insufficient_public_evidence`),各只问一次未催。同样证据包 sol 写得出卡、
  flash 写不出。另:`deepseek-v4-pro` 在 u7 产出 3 种程序,flash 与 sol 各只 1 种(n=1,仅线索)。
- **模型纪律**:起始 `deepseek-v4-pro`(请求=返回),g1 形成段后账户 **402 Insufficient Balance** ⇒ 该半段
  作废归档不并表,整包改用已授权的 `gpt-5.6-sol`(两组检查点逐条可核)。**402 暴露我自己的缺陷并已修复**:
  原分类器把它当 `SLOW_STAGE_FAULT` 重试并记成 `NO_PROPOSAL`(读起来像"Slow 弃权",是假的),
  现为 `ACCOUNT_OR_PERMISSION_FAULT`:不重试、不记弃权、单列。
- **两次宿主级中断都留痕并恢复**:内存不足杀进程后补上**真正的断点续跑**(Episode 随检查点以可重建形式
  落盘;边界靠重编译本组 fork 恢复,不二次问 Slow;续跑单元不重复计费不重复调用),被杀检查点原样保留。
- **下一项只一个(未启动)**:把**不确定的效果判断**移出 `observable_applicability` 放进正文由 Fast 当场核查,
  条件只留**确定的合法前置**。分寸:"告诉 Slow 特征的计算范围与分母"是**输入完善**(本包已实现,21 个特征
  按漂移性质分类入卡);"软化检索条件"是**待比较的方法变化**,放宽同样可能把指导送错序列,不得预设为正确。
  **本包结束后不再追加当前版本的组**;不启动第三组、消融、密封实验或普通反思第三臂。
- **待办(已评估未启动)**:批内并行。实测进程 CPU 仅约 2%、全程阻塞在网络,4–6 路并发可把一组从数小时压到
  45–60 分钟;阻碍是四处共享可变状态(预测缓存/账本计数/BudgetGuard/分臂计数),风险集中在**成本计数**
  而非实验读数,需先做"串行 vs 并行逐位等值"对拍再用。

## 2026-09-07 深夜（更晚):DEV-SEQ-1 收口——**决策单位改成逐序列并执行正确;知识更新两次都没形成,因为没有可写的条件**

判词 `PER_SEQUENCE_EXECUTED__NO_CONDITION_TO_WRITE`(Opus 主包)。设计更正写入 `AGENTS.md` **§5.3**
(六条,追加式)与执行计划 **§0**;报告 `docs/DEV_SEQ1_PER_SEQUENCE_2026-09-07.md`;工件
`artifacts/main_protocol/{dev_seq1_per_sequence__run1,dev_seq1_contrasts__run1,dev_seq1_per_sequence__run2,
dev_seq1_contrasts__run2,dev_seq1_reference_readings__all}.json`;源
`evaluation/main_protocol_p4/{per_sequence,dev_seq1_knowledge,run_dev_seq1,smoke_dev_seq1,audit_dev_seq1}.py`。
形成段 u7/u8/u9 × 10 条序列 × 2 次独立运行,全部跑完,`stopped_at` 两次都是 null。
**190 fits / 449 LLM / ≈19 分钟(含全部试跑);单包上限 2000 fits / 6h,正式运行各只用 24 与 34 fits。
编辑核心文件 0 个,新增 SHA 0 个,未提交 git。**

- **设计更正(§二)**:决策单位 = **单条序列及其当前合法窗口**,不是 cohort 代表序列。此前全线的 Fast 入口是
  `_a5_request(cell.observation_block, ...)`,而 `observation_block = values[support_a[0]][:origin]`——
  **一条代表序列决定 20 条被服务序列的同一个 compiled 程序**;smoke 实跑核实了这条事实。共享的是
  Agent / 工具 / Skill,不是同一个处理程序;Scope 退回"适用条件 + 作用几何 + 执行边界";旧组级判词全部保留
  但测试范围明确为"cohort 广播式单程序 + 组级知识写回",**与本包读数不并表、不相减**。
- **执行正确性(最强的一条)**:对每条序列,Fast 决策时测到的数必须等于被执行的整体策略交付给它的数——
  **两次运行 60/60 全部相等(<1e-12)**。异质赋值真实发生(u7 有 `outlier_mad` 与 `impute_linear`/`impute_ema`
  并存;u8/u9 一批部署一批 identity)。identity 的 0.0 经"空 Scope 与 Static 逐位相等"实测确认,不是填零。
  **契约本来就承载得了**:`scoped_evaluate` 的两个模型都不依赖 Scope,每条序列的损失只由自己那行预测算出,
  所以异质赋值按程序分组各读一次即精确,**未改 Consumer 训练/服务语义、未建模型路由**。
- **对照 cohort 级固定答案(祖先 `outlier_mad` 发给全体)**:逐序列在 **Support 面六格全胜**
  (均值 **+0.130977 / +0.126873**),但在**权威 delayed 面只有 +0.004461 / +0.010890——量级即噪声**。
  最大一格 u009:cohort 答案在那里**有害**(−0.219691),逐序列靠对 7–8 条弃权把交付拉回微正。
  **"少做"是这一格收益的来源,不是"挑对了程序"。** 覆盖随窗口 10→6→3 / 10→6→2 下降;
  单元权威门(阈值一个未改)两次都只在 u008 过一次。
- **分组**:8 条物料失败 → 1 组 `g_outlier_mad_NEGATIVE`(7 名 / 6 条序列),依据 = 可比性(Task×Consumer×metric
  与 domain 唯一)→ 完整 typed workflow 指纹 × sign → 症状 / 参数变体 / 可见 Pattern 展布 / 机械证据屏蔽后的
  错误类(唯一可选 `SCOPE_MEMORY_RISK_ERROR`);**保留 12 条同指纹成功对照**;1 条孤例留在 Episode 未被塞组;
  未部署候选的 delayed 记 **UNKNOWN 未填零**。
- **⚠️ run1 的弃权是我的证据包缺陷,已定位并修复后重跑**:`build_contrast_capsule` 返回的成功对照只有
  `{episode_id, provenance, origin, support_gain}`——**没有序列身份/程序/delayed/可见特征**,展布又只算失败一侧,
  于是真实 Slow 以 `insufficient_public_evidence` 弃权。修好(两侧同表)后 run2 的 Slow **真的提出**了指导卡
  `guidance_avoid_outlier_mad_high_missing`(散文指导,无冻结程序,无候选供给权),经真实 controller 编译入库
  (`98dea3b0…`→`441d6c1b…`),**但被预注册验证在 L2 拒绝**:条件 `missing_fraction≥0.45 ∧ z_peak≥5.0`
  只覆盖 6 条失败序列中的 **1 条**。**库一个字节没变,没有人工补卡,没有为凑正号追加运行。**
- **为什么写不出条件(机械读数)**:该组失败侧与成功侧放在同一组特征上——**12 个特征无一能分开**,
  4 个两侧恒为常数;**6 条失败序列里 4 条同时出现在成功一侧**(同序列同程序在不同窗口给出相反答案)。
  任何条件要么覆盖几乎没有(L2 拒),要么覆盖所有人(L1 拒)。**这在决策层独立复现了 R4A 的前提。**
- **后续段未开**:没有合格更新 ⇒ 两臂同库 ⇒ 比较失效,按纪律停下受影响部分,记 `NO_UPDATE_TREATMENT`;
  未用单臂冒充后续段。
- **首因上移**:按 `AGENTS.md §6` 阶梯,最早一级是 **"Support 成功而后续窗口不成功 → Scope/过拟合/风险"**,
  第二级是 **更新**。**选择层这次是干净的:"已探到更好候选却没部署" = 0/30(两次都是 0)**——
  与 DEV-KNOW-1 / DEV-AUTO-1 把首因定位在选择层**不同**,因为那两包的决策单位是 cohort 代表序列。
- **下一步(交裁定,未自行推进)**:不再加卡片数量 / 不改保存方式 / 不做 series 级 Scope 收窄;
  值得打开的是**证书粒度**(把决策的证据单位从"单 origin 单序列"换成跨窗口聚合,再看 Support→delayed
  是否还翻号)——与 R4A 建议的 0-fit 重打分同方向,本包提供**带 Agent 的**版本。若要继续测"知识更新是否有用",
  必须先有能过 L1+L2 的条件,而当前 12 维词表下它不存在,所以下一步应先解决**观测面**。
- **仪器勘误(本包内自查)**:`programs_used` 曾以只含算子名的 label 为键,同算子不同参数会塌成一个键;已改为
  带参数的键并加显式碰撞报告。**读数不受影响**(60/60 决策全对得上)。run2 u8 唯一一处碰撞
  (`outlier_mad{}` vs `outlier_mad{z_threshold:3.5}`)按逐序列增益向量去重后**是别名**
  (读数逐位相同,该参数不在 `public_parameter_schema` 里),故不算参数层差异。
- 传输:请求 `deepseek-chat` @ `api.deepseek.com/v1`(`M0_AGENT_*`),**返回两次都是 `deepseek-v4-flash`**
  (同 DEV-KNOW-1 正文的别名路由);用量 ≈69 万 prompt / 1.3 万 completion tokens 每次运行。


## 2026-09-07 深夜（续）:R4 系列更正与合并——**采认 astra 复核；"序列级没有可学条件"收窄为"已测的序列级描述量识别不出"；多数效应来自训练侧共享模型变化；证书粒度重打分撤回**

依据 `idea-stage/observation_mechanism/notes.md`（astra，0 fit / 0 LLM）与 `docs/R4A_B_PERSISTENCE_AND_VARIANCE_RESULT_2026-09-07.md` §6 追加更正。
下一条目（同日深夜 R4A 收口）的数字全部保留；以下只改**读法与下一步**。

- **收窄三条推断**：① 低重测相关（Spearman ≈ 0、ICC ≈ 0）只否定实体记忆策略（"该序列上次受益→下次继续用"），**不否定按 origin 时刻
  可见状态决策**（X_t 独立同分布、g_t = X_t 时相邻相关为 0 而决策可完美）；ANOVA 残差应称"当前分解与已测特征未解释的方差"，不是不可约噪声。
  ② 三个 ORACLE 未来摘要不是信息论上界，`NO_FUTURE_ADVANTAGE_DETECTED` 只否定这三个描述。③ Ridge 为确定性求解，"重测方差"提法不成立；
  对观测增益做置换识别不出"中性程序"的虚警率；改风险门保护目标属协议决定——**"证书粒度重打分"作为下一刀撤回，风险门不动。**
- **新事实（astra 补算，采认；源码公式级，待生产执行器复核）**：delayed 面 **269/420 实例服务窗口未被 MAD 修改，却承载 31 个严重受损中的 24 个**
  ⇒ 多数效果来自训练侧共享模型变化，R4A/R4A-b 全部序列级特征描述的是服务窗，对这些实例**观察对象不匹配**；Hampel 实际命中点 92.6% 全窗 robust-z < 3
  （全局"尖峰"≠局部算子动作）；周期填补面对的缺失点仅 31.6% 有 ≥2 同相位供体，其余退回线性（缺失率未表达程序可用性）；
  `outlier_mad` 自带 `interp_nan`，其后 `period_median_complete` 恒等（W3 ≡ ANCESTOR 42/42 的机制解释）。
  W1 阈值 6.0 规则相对 hampel 全治 +0.081614，但相对 MAD 全治参照 **−0.221642 / −0.124572**：是差程序的禁忌条件，不是有效规律。
- **仍成立**：程序层方向跨单元稳定（R2）；实体记忆无效（DEV-KNOW-1、helped kappa ≈ 0）；已测约 22 个序列级描述量与三个未来摘要均弱；
  台账 19 项测量中"自然数据部署可见模式识别条件成功" 0 项；W61 早已测过局部片段/动作变形（"过去全是全局特征"说法不准确）。
- **R4C 收口（2026-09-08 凌晨，主线亲自执行，0 fit；执行线两次派发分别停滞/撞用量上限）**：
  `docs/R4C_IMPUTATION_DONOR_CONDITIONS_RESULT_2026-09-07.md`；工件 `artifacts/main_protocol/r4c_imputation_donor_conditions.{json,md}`。
  判词 **`ACTION_CONDITION_CONSISTENT`（退化形式）+ `VALUE_CONDITION_UNINFORMATIVE`**。按算子真实逻辑（直接调用
  `period_median_complete` / `interp_nan`，逐点旗标与算子输出 840/840 服务窗、8400/8400 训练窗一致）算出的供体覆盖/一致性/
  两种填补差异/尾部填补点数，对 d = g(W2) − g(ANC) 的方向无识别力（四折 AUC ≤ 0.583），决策规则 0/4 折同时胜过 always-ANC 与
  always-W2。边际效果本身很大（|d| 均值 0.28–0.37，与祖先总增益同量级）却近零均值、正负各半。
  **结构事实**：训练侧五个观察量在块内逐位相同——anchors [312…852] 在所有 origin ≥ 1176 下全部通过，**同一块内 42 个面共用同一
  训练语料与同一模型**（AGENTS §5.1 追加更正的数据坐实）⇒ 训练侧条件在本课程设计上不可能区分单元；astra 的"服务窗未修改却有效应"
  = 固定模型差 × 变化的 context/horizon；R4D 三格分解只需每块每程序拟合一次。
- **R4C 勘误（Grok 非作者复核 `artifacts/main_protocol/r4c_nonauthor_check.md`，25/27 一致）**：结果报告 §3.4 曾把 always-W2 − always-ANC
  四折均值误写为 −0.003883，正确为 **+0.003883**（pooled −0.006376；四折两正两负，块间反号）；已在报告内改正并注明。工件数字无误。
- **R4D-A 收口（2026-09-08 凌晨，Opus，用户口头批准的小额拟合；累计 24 physical fits = 两次 12，第二次为落盘精度修正重跑）**：
  `docs/R4D_A_ACTION_DECOMPOSITION_RESULT_2026-09-08.md`；工件 `artifacts/main_protocol/r4d_a_action_decomposition.{json,md}`；
  新预测库 `_scratch/r4d_a_three_cell_store.json`。三格 = raw 模型/raw ctx（=Static）、P 模型/raw ctx、P 模型/P ctx；两个端点对
  m_r0k 预测库 **1680/1680 复现**（max |Δ| 1.3e-15）。判词 **六格全部 `ROUTE_DOMINANT`**：共享模型通道（route）的代数份额
  ANCESTOR 0.768/0.804、W2 0.714/0.703；严重伤害实例中 route 为最大正贡献 ANCESTOR 96.4%/96.8%、W2 65.0%/78.7%；
  ρ(route,total) 0.85–0.94 对 ρ(ctx,total) 0.21–0.29。astra 2.1 两个数字精确复现：ANCESTOR delayed 269/420 服务窗未被修改
  （ctx 逐位为 0）却承载 77.4% 严重伤害，其 total p10/p90 = −0.696/+0.281。与 D5（11 窗 0.73、8/10）同向、扩到全部 42 面，D5 不覆盖。
  **含义（2026-09-08 按 astra 反馈收窄）**：pooled Ridge 下"准备这条序列自己的窗口"对这条序列结果的贡献只占 20–30%，
  70–80% 来自"训练语料被准备后换了一个共享模型"。**这回答的是效应从哪里产生，不是哪些信息能预测谁受益**：共享模型变化是同一个
  处理，各序列对它的响应仍可能由序列状态调节（处理 × 状态交互），R4A 只对 total 测过（total ≈ route，ρ 0.85–0.94），
  ctx 通道与"按通道分开"均未测。主线此前"序列级可见模式在结构上是错误的观察对象"的表述**收窄**为"序列自身窗口的准备不是效应主要来源"。
  Observation 应同时描述训练数据、程序造成的模型变化与当前服务窗（astra）。
- **R4D 预算与分流更正（采认 astra）**：B 的 2520 fits 与"anchors 固定"自相矛盾——固定 anchors 下每条 served 序列自己的 10 个训练窗
  不随 origin 变，**B = 80 序列 × 3 模型 = 240 fits**（上限 260）。旧分流"B 无改善 ⇒ 噪声在评价侧、关闭序列级"**撤回**，
  改为"这组观察量在该设置下未表现出足够信息，不判定噪声"；新增 §4.5：按 route / ctx / total 三通道分开的可识别性、
  "看 Pattern 选处理 vs 始终选同一处理"的全人群效用裁决、per-channel Static 基线质量。
- **R4E 收口（2026-09-08 凌晨，Opus，0 fit）**：`docs/R4E_CHANNEL_IDENTIFIABILITY_RESULT_2026-09-08.md`；工件
  `artifacts/main_protocol/r4e_channel_identifiability.{json,md}`。用 R4D-A 三格库把 R4A 的 21 个可见量 + R4C 的 6 个作用条件量分别对
  route / ctx / total 三通道做四块 LODO。**答 astra"被总效应掩盖的局部规律存在吗"：存在一条，范围比问题小。** W2 的 **ctx 通道**上
  `local_robust_z_peak → severe` 四折 LODO **0.826536**（delayed 面，四折 0.748–0.969，每折有正例、方向一致，项目首个满支撑
  `PATTERN_INFORMATIVE`）；support 面同特征反向对 helped 0.720834——机制连贯：**有明显尖峰的 context 清洗有益，没尖峰的 context 清洗
  会把正常波动改坏**；R4C 的 `srv_period_filled_points` 首次进入最佳位置（0.731）。四条限定：只在 ctx 通道（占效应 20–30%）；只在 W2
  （ANCESTOR ctx 的"INFORMATIVE"由 3 个正例撑起、两折无正例被规则跳过，不作证据）；只在单面（两面合并掉回 WEAK 0.779）；
  **route 通道（占 70–80%）三格九面全部 WEAK（≤ 0.617）**。最终裁决口径"看 Pattern 选处理 vs 始终选同一处理"六格全部
  `FIXED_CHOICE_NOT_BEATEN`：训练折最优阈值退化为"全都选"（always-P 已为正 +0.23–0.27），oracle 逐序列上界 0.31–0.40 与 always-P
  之间的 0.07–0.13 是可见量拿不到的空间。R4A 对 total 的四格读数逐位复现。按 §5 修订表落第三行：**不判定噪声、不关闭序列级研究**
  （ctx 上信号是真的），也**不**把该特征送进 Observation 候选（部署要为不可识别的 route 一起买单）。
- **R4D-B 收口（2026-09-08 凌晨，主线亲自执行；执行线第二次派发停滞；240 physical fits，0 LLM）**：
  `docs/R4D_B_PERCHANNEL_RESULT_2026-09-08.md`；工件 `artifacts/main_protocol/r4d_b_perchannel_three_cell.{json,md}`；新库
  `_scratch/r4d_b_perchannel_three_cell_store.json`。**五个读数**：(i) **基线**：per-channel Static 比 pooled Static 好 25%
  （损失均值 1.119 对 1.495，Δ 中位 −0.224，仅 33% 序列 pc 更差）——与预期"欠定"相反，按冻结口径 `consumer_choice_in_scope=False`
  （尺子不同），但方向是 pc 更好；(ii) **分解**：pc 下六格仍全部 `ROUTE_DOMINANT`（0.73–0.79）⇒ **"训练侧准备→模型变化"是主通道，
  与是否共享无关**；(iii) **持久性**：跨面 Spearman 0.073 / 0.144，ANCESTOR `MOSTLY_NOISE`，W2 uid 份额升到 0.31–0.32 → `MIXED`；
  (iv) **可识别性**：pc 侧 4 格 INFORMATIVE **全部无全折支撑**（严重伤害只剩 ANCESTOR 12 / W2 14 例，pooled 为 59 / 135），不作证据；
  helped 目标全弱；"看 Pattern 选处理 vs 固定选择" 6/6 `FIXED_CHOICE_NOT_BEATEN`；(v) **伤害形状**：四格 `CONSUMER_SHAPES_HARM`
  + `GAIN_LOST`——聚合增益降到 1/3–1/10（ANCESTOR 0.072/0.018 对 0.245/0.225），最坏单序列伤害降到 1/3–1/2，严重伤害例数降到 1/5–1/10。
  **含义**：pooled 下"数据准备"的收益与尾部伤害有很大一部分是共享 Consumer 不匹配的产物；`max_single_series_harm ≤ 0.30` 这条
  绑定约束主要是 pooled 的性质。序列级观察量在两种 Consumer 下都未能改变部署决定；**不判定噪声，不关闭序列级研究**，但这组
  27 个可见量不再加实验。论文应把"Consumer 结构决定数据准备的价值与风险"作为一个主结果（三张表已在手，0 新 fits）。
  非作者复核（Grok）8/8 一致，三处舍入级备注已追加到报告 §6。
- **R4F 探索性关系图谱（2026-09-08 上午，主线亲自执行，0 fit；执行线第三次派发停滞）**：
  `docs/R4F_RELATIONSHIP_ATLAS_EXPLORATORY_2026-09-08.md`；工件 `artifacts/main_protocol/r4f_relationship_atlas.{json,md}`；
  画廊 `artifacts/main_protocol/r4f_gallery/*.png`（12 张）。**性质 EXPLORATORY / HYPOTHESIS_GENERATING，不设门不判词，任何进入方法的
  关系须冻结后在未读数据重测。** 用户方向：不是证明有效，而是发现"某某与某某相关"。四个因子：① **context 尖峰显著性族（8 个量）↔ 清洗
  自身窗口的收益**，pooled ctx 通道 ρ 0.2–0.46、四块全同号（R4E 那条 0.83 不是孤例）；② per-channel 下 total 开始与可见量相关
  （尖峰族对 ANCESTOR ρ −0.24，尾部缺口被周期填补的点数对 W2 ρ −0.20，四块同号，树根四折一致）；③ 极端尖峰（peak/tail-sd ≥ 15）
  6.5% 实例贡献均值 −0.7、其余 −0.2；④ context ≥ 75% 缺失时准备不再有益（+0.03 对 −0.22），且 `local_robust_z_peak` 在此区间
  数值失效（10⁹–10¹⁰，**仪器缺陷，呈执行线修**）。程序/Consumer 间：同族程序受益者重叠（ANC–W2 0.55、ANC–IQR+winsorize 0.72）；
  route ⟂ ctx；**pooled 与 per-channel 下"谁受益"几乎无关（ρ 0.04 / −0.02，Static 难度却 0.48）——同一序列同一程序换 Consumer 重新洗牌**。
  画廊模式：pooled 最差全是 route 伤害（干净周期序列或 87% 缺失序列）、最好是 T23 巨尖峰跨四个 origin 持久大赢；W2 ctx 伤害两种形状
  （长/尾缺口锯齿填充；MAD 削平真实水平抬升）。发现清单 15 条（10 条新），含对 Scope/供给/Observation 的具体设计含义。
- **对推进的含义**：pooled 下序列级 Scope 修订 / 实体证据继续停；"给 LLM 什么信息"的候选从服务窗形态转向**程序实际作用条件**（R4C 三段拆法：
  作用条件可识别 → 相对价值可识别 → LLM 能利用）；held-out 开封、写作时间线不变。

## 2026-09-07 深夜:R4A 条件可识别性收口——**series 级条件读不出、也不在未来、而且根本不稳定；稳的是 cohort 级程序方向**

判词（Opus 主包）:八个 (program, face) 主格全部 `PATTERN_WEAK`（六格独立,W3≡ANCESTOR 别名）;结构性上界
`NO_FUTURE_ADVANTAGE_DETECTED`。任务书 `docs/R4A_PATTERN_IDENTIFIABILITY_TASKBOOK_2026-09-07.md`(阈值先冻结);
报告 `docs/R4A_PATTERN_IDENTIFIABILITY_RESULT_2026-09-07.md`;工件 `artifacts/main_protocol/r4a_pattern_identifiability.{json,md}`;
主线复核与追加读数 `docs/R4A_MAINLINE_REVIEW_AND_ADDENDUM_2026-09-07.md`;Grok 侧 `r2_revision_effect_stability.{json,md}`、
`docs/R4A_IDENTIFIABILITY_EVIDENCE_LEDGER_2026-09-07.md`。全程 0 fit / 0 LLM / 0 held-out 读(max_time_index 3911 < 4056)。

- **可见模式**:机制导出的 6 个尖峰量 + 4 个缺口量,没有一个胜过现有 12 维词表;最佳四折均值 0.63–0.68,最常胜出的仍是
  `local_robust_z_peak`,且原始 AUC < 0.5(尖峰越不显著越被裁剪重伤)。全治为正的程序,冻结词表挑不出优于全治的子集(+0.000000)。
- **未来窗口**(ORACLE,永不可部署):三个未来量对 severe_harm 的 AUC 最高 0.606145,24 行 margin 全负 ⇒ "条件在未来"这条改写路径无证据。
- **主线追加(决定性)**:同序列同程序 g@origin 对 g@origin+48 的 Spearman 仅 0.017–0.109,符号一致率落在独立期望上
  (0.571 vs 0.568 等);块内按 uid 的 ICC ≈ 0(−0.08…+0.11,7–9 次重复的两个块)。**逐序列增益不是序列的稳定属性。**
- **R2(Grok)**:d(child−ancestor) 方差更小(8/8)但符号不比 g 稳(1/8);`outlier_mad` 单元级方向 86–92% 一致,
  `hampel` 一贯更差。"比较性知识更稳定"假设不成立;稳定对象是 **program × Consumer × cohort 的方向**。
- **唯一模式级线索**:W1(全治 −0.031355)在 `spike_peak_over_tail_sd >= 6.0` 上四折均优于全治 +0.081614——程序级禁忌,非精调 Scope。
- **transport_flip**:Support 受益序列 31.6%–45.1% 在 +48 翻号,无任何 origin 前量可预判(≤0.56,方向不一致)。
- **first-fault 上移**:两条逐序列风险线在单 origin 上度量的是重测相关≈0 的量 ⇒ 34→10 是证书粒度缺陷,不是 Scope 缺陷。
- **呈裁定的修正**:停 series 级 Scope 修订类实验(M-W / ACC-1 二层 / 有界选择规则);改做 0-fit **证书粒度重打分**
  (单 origin / 双 origin 合并 / cohort 聚合+bootstrap 下界 三口径下 Support→+144 保持率);论文条件化主张按粒度改写;
  held-out 开封、Consumer 轴、写作时间线不变(Consumer 轴新增问题:ICC≈0 是 pooled Ridge 的性质还是任务的性质)。
- 证据台账勘误(Grok):D1 四信号实测 0.4032–0.6495(此前写 0.45–0.65);W54 精确 0.49737;p4y 无 `summary` 只有 `counts`;
  K0 卡 JSON 无 `z_peak>=3` 字面。历史数字不改写。

## 2026-09-07 夜:DEV-KNOW-1 收口——**知识保留住了、被读出来了、从未被选中**

判词 `KNOWLEDGE_WAS_RETAINED_AND_READ__AND_NEVER_SELECTED`。
报告 `docs/DEV_KNOW1_ACCUMULATED_SKILL_TWO_ARMS_2026-09-07.md`;工件
`artifacts/main_protocol/{dev_know1_two_arms__g1,dev_know1_two_arms__g2,dev_know1_contrasts}.json`;
源 `evaluation/main_protocol_p4/{dev_know1_retention,smoke_dev_know1,run_dev_know1_two_arms,audit_dev_know1_contrasts}.py`。
形成段 u5–u15(1 条自主轨迹)→ 分叉 → 后续段 u16/u17/u23/u24/u25 × 2 臂(A 旧知识 / B 累积知识)× 两组,全部跑完。

- **保留修补(第一段)**:以 DEV-AUTO-3 的 **B 路线**为基础,**全程不 PATCH 任何卡**。
  新程序验证通过后经既有铸卡路径(`handle_fast_winner` → `ADD` + `ABSENT` 前置 → 两阶段 delayed → `activate_approved`)
  成为**独立卡**。任务书五条里 1–4 既有机制已满足(复用,不重建);**只有第 5 条是新写的**——
  `R.current_skill_steps` 在卡缺失时会 `return tuple(ANCESTOR_PROGRAM)`(拿历史祖先冒充当前卡),
  本包用 `keep.current_card` + `revision_step` 包装层挡在它前面,卡不可用即以 `NO_CURRENT_CARD` 结束该步。
- **两个必要行为检查都是实跑**(真实 store+编译器,0 LLM/0 fits):父子共存(1 张 capability 卡 → 2 张,父卡逐字段不变);
  撤销子卡**恰好 unlink 1 个 entry 文件**,父卡 body 逐字节存活,库回到子卡出现前的成员集合。
- **⚠️ 边界事实**:曾在 `methods/ttha/method.py` 加过显式重复卡守卫,**已撤回**——`method_contract` 是 harness
  dependency sha,改它会改掉每个快照的 `runtime_bundle_sha`,锁定 h0 立刻失配(h0 测试 4 过 → 3 败)。
  **本包最终 0 个核心文件被编辑**;要求 4 本来就由 id 命名空间 + `ABSENT` 前置满足。
- **分叉对称**:A = 边界快照**减去**形成段新增卡(用撤销那套 store 往返),**不是入场快照**(不复活被撤销祖先);
  两臂都不带前段候选供给池(未关闭 Draft 在**两臂共同拷贝的那一份 ledger 上**关闭为 `NOT_CARRIED_ACROSS_THE_FORK`,
  关闭≠删除,计数/状态/风险记录逐字段保留);两臂 method 都以空经验列表重建 ⇒ **前段到后段唯一通道是 Skill 卡**;
  A 保留全部外环(不是 identity 对照);铸出的卡带 `requires_target_support`,不授予新情境执行权。
- **g1 = `NO_KNOWLEDGE_TREATMENT`**,原因已定位:形成段 Fast **一次都没部署新程序**,
  两次过门(u5/u15)部署的都是祖先自己的 `outlier_mad` ⇒ 铸卡撞 K0 已占用 id 被 `ABSENT` 拒绝,`activated: []`。
  **不是保存失败,是没有新程序通过过 delayed 权威门。** 不手工补卡。
  这一组因此给出**臂间运行噪声底**:Support **+0.016353** / delayed **+0.000307**(两臂机械上完全相同)。
- **g2 = `KNOWLEDGE_TREATMENT`**:u9 提议 `outlier_iqr`(对父程序 `outlier_mad` 的替换)→ 入队 `resupplied_draft_4`
  → u11 Fast 部署,Support +0.320571、delayed +0.308092、门四条全过 → 铸出独立卡
  `fast_winner_forecast_ridge_smase_outlier_iqr`,**父卡原样存活**(边界库两张卡并存)→ u12/u15 `recalled_skill` 取回,
  u15 再次过门 `minted=[]`(不重复建卡)。**这是保留修补在真实课程里跑通的证据。**
- **⚠️ 但后续段的答案是零**:B 在 **5/5 格检索到**该卡、**2 格(u17/u24)以卡片候选身份真的实测**、
  **0 格部署**。两臂逐格部署相同、读数逐位相同 ⇒ Support 与 delayed 差值**都恰好 0.000000**,
  覆盖/伤害/门通过差 0,而 B **多花 +24 fits / +5 次调用**。两臂后续段**都是 0/5 权威门通过**,
  故"找到首个合格方案所需反馈量"记 **UNKNOWN**。判词 `TRIED_BUT_NEVER_SELECTED`。
- **收口**:落在「知识未被使用」。但要说准——**检索不是卡点、保存方式不是卡点,卡点在选择层**
  (决定部署哪个候选的那一步,实测之后仍选 identity 或祖先程序;与 DEV-AUTO-1 的 `PROPOSAL_DIVERSITY_GAP` 同处)。
  **不再加卡片数量,也不再改保存方式;不进入独立情境验证计划;不为凑正号追加运行。**
- **成本**:798/2000 fits、312/1000 LLM、≈20 分钟/6 小时(含首次失败上手 10+2、冒烟试跑 67+27、正式两组 721+283)。
  4 个阶段各 500 fits 额度(4×500 = 包级上限),阶段在**花之前**拒绝;两组两段 `stopped_at` 全为 `None`,未触顶。
  0 密封 / 0 +144 / 0 TARGET_HELD_IN;**PATCH 卡片 0 次、编辑核心文件 0 个**;未新增哈希平台;未提交 git。
- **新增预检(被真实事件逼出来)**:live transport 必须与上一包收据里的 transport 一致,否则预检直接失败
  (期望值从 DEV-AUTO-3 收据读,不写死主机名)。第一次上手 `M0_AGENT_*` 未设 → 回退到模块默认中转站
  `api.agicto.cn` / `gpt-5.6-luna`(DEV-AUTO-2/3 从未用过的模型),对方以 `insufficient_quota` 拒绝。
  正式运行的 transport = `https://api.deepseek.com/v1` / `deepseek-chat`,与 DEV-AUTO-2/3 逐字相同。
- **限制**:有处理的组只有 **1** 个(n=1),另一组无处理可测;5 个已暴露 development 单元;两组不冒充独立复现;
  不做显著性主张;本设计以 n=1 处理组分辨不出小于约 0.016(Support 面)的效应。
- **附录:换模型 gpt-5.6-sol,完整补跑已完成(两次 OOM 之后)——两组都形成知识,交付都是负的。**
  探查阶段两次 OOM(2.2–2.4 GB 空闲、0 cell fault,环境限制不是模型/代码故障)之后,做了三处工程
  修补再补跑:`run_id` 隔离 checkpoint/存储根(不覆盖旧 partial/旧 checkpoint)、`Budget.save/load`
  让包级 fits/LLM/wall 账本跨 OS 进程续接(g1、g2 分进程跑、进程退出即释放内存)、形成段结束和
  fork 完成两个节点各落一份 checkpoint。**这次 g1、g2 都以 exit code 0 跑完,0 cell fault,
  `stopped_at` 全部 `None`**,内存两次低谷(1.2–1.8 GB 空闲)都扛住了。
  结果:**两组都在 u11 铸出同一张卡** `fast_winner_forecast_ridge_smase_outlier_iqr`
  (`treatment=KNOWLEDGE_TREATMENT`,验证读数两组逐位相同:delayed 0.308092、support 0.320571)。
  后续段:**5/5 格检索到**;g1 全程 **0 次探测**该卡自身、g2 **1 次探测**(u17,该格两臂皆摊回
  identity);**两组全程 0 次部署**该卡自身的 skill_id。但**交付确实系统性变了,而且是负的**:
  两组 support/delayed 差值符号一致,累积臂均低于旧知识臂(g1: support −0.057798、delayed
  −0.119086;g2: support −0.054672、delayed −0.111169),门通过两臂全 0。**主因定位到同一个
  单元 u23**:两组累积臂当格候选集里都**只有**一个刚被重新局部搜索出来的 `outlier_iqr` 变体
  (不是 recalled 已铸卡本身),没通过就直接摊回 identity、0 覆盖;旧知识臂当格都成功找回/复用
  `outlier_mad` 并覆盖 15 条序列、拿到 +0.595428 的收益。g2 u16 是唯一一格累积臂部署
  `outlier_iqr`(仍是当格重新发现的,非 recalled)且赢了旧知识臂(+0.039583)的单元,但远不够
  抵消 u23 的损失。**归因要精确**:真正把已铸卡本身当候选测过的全程只有 1 次(g2 u17);这次
  观测到的行为差异主要是"累积臂库里多了这个选项之后,候选搜索某些格倾向只围着它转、不退回
  旧程序"的副作用,不是"卡被复用带来的收益"。收口规则与正文同一句(`KNOWLEDGE_WAS_NOT_USED`,
  卡点在选择层)但症状不同:deepseek 是"从不选"(交付逐位相同),sol 是"选了会挤掉旧程序覆盖到
  的人群"(交付系统性变差)——两个独立模型指向同一层,不做更强的工程结论,任务书排除了在本包
  内改选择策略。**不与正文 deepseek 结果相减、不做跨模型主张**(工件 `comparability` 字段原文
  声明二者不可比)。详见报告附录 A `docs/DEV_KNOW1_ACCUMULATED_SKILL_TWO_ARMS_2026-09-07.md`;
  工件 `artifacts/main_protocol/{dev_know1_two_arms__g1_sol,dev_know1_two_arms__g2_sol,dev_know1_contrasts_sol}.json`。
- **传输一致性预检加了明示豁免口**:默认拒绝静默换模型;设 `DEV_KNOW1_TRANSPORT_NOTE=<原因>` 才放行,
  原因写进工件的 `transport_divergence` 字段 ⇒ 换了模型的运行永远自带标签。sol 运行写独立工件后缀 `_sol`,
  不覆盖正文。新增 `transport_actual_usage` 读数面(直接读 `BudgetedAgentBackend` 已有计数器):
  sol 两组请求/返回模型逐字相同(均 `gpt-5.6-sol`,不像 deepseek 那次意外发现的别名路由
  `deepseek-chat`→`deepseek-v4-flash`);用量约 55 万 prompt tokens / 4.3 万 completion tokens
  每组,明显比 deepseek 更"啰嗦",这也是约 9 倍慢的一部分原因。
- **本包累计成本(含全部失败尝试与本次成功补跑)**:**deepseek 798/312;sol 探针 75/24、
  sol 第一次(被杀)243/75、sol 第二次(被杀)139/50、sol 完整补跑(成功)621/240,墙钟
  ≈2.77 小时** ⇒ **DEV-KNOW-1 全部尝试合计 1876 fits / 701 LLM**(包级预算按每次运行单独核销,
  本次成功补跑单包 621/2000 fits、240/1000 LLM,远在预算内)。


## 2026-09-07 晚:DEV-AUTO-3 收口——**供给买到了东西,写回没有;而且写回一次毁掉了仍然有效的知识**

判词 `SUPPLY_PAYS__UPDATE_DOES_NOT__AND_ONCE_DESTROYED_WORKING_KNOWLEDGE`。
报告 `docs/DEV_AUTO3_SUPPLY_VS_UPDATE_2026-09-07.md`;工件
`artifacts/main_protocol/{dev_auto3_three_arms__g1,dev_auto3_three_arms__g2,dev_auto3_contrasts}.json`;
源 `evaluation/main_protocol_p4/{run_dev_auto3_three_arms,smoke_dev_auto3_three_arms,audit_dev_auto3_contrasts}.py`。
三臂(A 控制 / B 供给-only / C 供给+写回)× 16 单元 × 两组,全部跑完。

- **早上的两个答案**:① 收益**主要来自新增候选**,Skill 更新未表现出额外价值;
  ② **B−A 两组一致(+0.039170 / +0.032167),C−A 两组异号(−0.015526 / +0.030023)**。
  C−B:g1 **−0.054696**(2 次写回)、g2 −0.002144(**0 次写回**⇒ 那是臂间噪声,不是机制效应)。
- **C 为什么更差,机制已定位**:u14/u15 两次写回把卡体从 `[outlier_mad]` 覆写成
  `[hampel_filter, winsorize]` → u17 `recalled_skill` 部署,delayed −0.063268、门四条挂三条 →
  **生命周期撤销整张卡**(`snapshot_after: []`)→ u23/u24/u25 C 无卡可用只能 identity,
  而 A/B 在 u23 用 `outlier_mad` 拿到 **+0.595428**。**单 u23 一格 −0.595 就抵掉了更新通道挣的全部。**
  **PATCH 是就地覆写、单槽位;失败不退回旧卡体,撤销把祖先程序一起带走。** 风险线一条没动。
- **B 不是"没有持久化"**:B 保留正常 Fast 生命周期,能铸新卡并 `recalled_skill` 取回(g1 u12),
  只是不覆写目标卡 ⇒ 它把祖先卡留住了,新知识加在旁边。
- **B/C 对称已做实**:验证事件在两臂完全相同、Draft 在**同一时刻**关闭
  (`VERIFIED_BUT_NOT_WRITTEN_BACK` vs `PROMOTED_TO_THE_SKILL_CARD`),消掉了 DEV-AUTO-2 里
  "不晋升的臂永远重复供应"的额外差异。三臂各自维护历史/ledger/缓存,缓存键含臂名,不跨臂共享。
- **控制臂两组逐位相同**(0.064176 / 47 / 15 / 门 1),且与 DEV-AUTO-2 冻结臂逐位相同 ⇒ 噪声不在控制臂。
- **仪器修复(随包)**:`DraftLedger._mint_draft_id()` 跳过已存在 ID。M-R0d 入场态 append Draft 却不推进
  `_minted`(实测 =0)。已有 Draft 的身份/验证次数/修订次数/关闭状态逐字段保留,不重编号、不覆盖历史收据。
  **不能仅凭 DEV-AUTO-2 用了 `_2`/`_3` 就断言当时无影响**——旧结果单列保留,不追溯重算。
- **迟到采纳确实存在**(B 的候选 u5→u11、u9→u14),**且不需要写回卡片**;
  但"第一次修订的反馈催生第二次修订"**仍未被证明**(g1 两次写回同源于 u13 一次调用)。
- **成本**:本包 1554/2000 fits、606/1000 LLM、≈36 分钟/6 小时(试跑 87+37、g1 879+283、g2 588+286)。
  与 DEV-AUTO-2 分账。0 密封 / 0 +144 / 0 TARGET_HELD_IN;未改风险线、栅格、Consumer、DSL;未提交 git。
- **下一步(交裁定,未自行推进)**:不建议在"卡片写回"方向再加实验组。最该先改的是上面那条性质——
  **改为新增卡(ADD)而非就地覆写(PATCH)**,让旧卡在新卡失败时仍可用;这会动编辑路由与生命周期语义,超出授权。


## 2026-09-07:DEV-AUTO-2 收口——**闭环合上了两次,但收益主要来自候选供给**;判词 `CHAIN_CLOSED_TWICE__MOSTLY_VIA_SUPPLY`

报告 `docs/DEV_AUTO2_FRAME_C_LIVE_COURSE_2026-09-07.md`;工件
`artifacts/main_protocol/{dev_auto2_frame_c__run1,dev_auto2_attribution,dev_auto2_frame_c__dryrun}.json`;
源 `evaluation/main_protocol_p4/{run_dev_auto2_frame_c,smoke_dev_auto2_frame_c,audit_dev_auto2_attribution}.py`。
两臂 × 16 个 development 单元**完整跑完**;Frame C 接受规则;模型 deepseek-chat。

- **Skill 卡真的被改写了两次**(DEV-AUTO-1 是 0 次):u7 提出的 `outlier_mad>winsorize` 在 u11 验证、
  `outlier_mad>hampel_filter` 在 u15 验证,均经 `OUTCOME_GAP` PATCH 走既有 compile/validate/load,
  编译落盘 + 回读校验 + Draft 按生产语义关闭为 PROMOTED;第二次的 parent SHA = 第一次的 candidate SHA(血缘成链)。
  **对照臂的卡 16 格只有一个版本**,不变量成立。
- **被排队的 15 个候选,历史绝对屏会全部否决**(10 NOT_READABLE / 5 RISK_LINE_FAILED),两个晋升的都在其中
  ⇒ 框架 A/B 下本轮晋升数会是 0,与 DEV-AUTO-1R 的离线复算一致。
- **整段效果(全服务人群,delayed 面)**:冻结 0.064176 → 修订 **0.114025**;覆盖 47→**90** 处理序列;
  权威门 **1/16 → 4/16**;收益为 0 的单元 11→8;受损序列 15→19;**最坏单序列伤害两臂相同**(0.484687)。
  成本:fits 69→194、LLM 90→112(每臂预分配 200,分臂记账)。
- **⚠️ 归因:12/16 格两臂部署逐位相同**,差异只在 4 格——**供给通道 +0.682973**(u11/u15,`resupplied_draft`)、
  **更新后的卡 +0.141939**(仅 u14,`recalled_skill`)、新卡下一次 Fast 搜索 **−0.027335**(u12,还多破一条风险线)。
  **本包同时开了供给与晋升两条通道,整体差异不能记在"Skill 自我更新"账上;主要来源是供给。**
- **晋升 ≠ 被使用**:第二张卡从未被部署,u16/u23 Fast 自己搜索后选了朴素父程序。
- **Scope 出口仍全关**:11 条 Scope 提议全部 NO_FEASIBLE_THRESHOLD,失败在风险线不在栅格(与 DEV-AUTO-1R 一致)。
- **包内修好三处"以前没走到过"的实现问题**:① 建臂时把自己正在跑的快照落进自己的 store
  (否则 `activate_approved` 抛 `unmaterialized runtime bundle`);② `apply_workflow_revision` 的 manifest 形状
  错了(须 `{"kind":"SHA","sha":...}` + `minimal_patch={"value":...}`),第一次正式运行就死在这里,已记中止尝试并重跑;
  ③ (只记录未修)M-R0d 入场态 append Draft 却没推进 `_minted`,导致 `resupplied_draft_1` 身份号碰撞。
- **成本**:本包合计 488 fits / 329 LLM / ≈10 分钟(试跑 60+23、中止尝试 165+104、正式课程 263+202)。
  0 密封、0 +144、0 TARGET_HELD_IN;未改风险线/分母/栅格/Consumer/DSL/HEC-1 合同;未提交 git。
- **下一项(交裁定,未自行推进)**:**三臂对照**(控制 / 只供给不晋升 / 供给+晋升),把两条通道分开。
  不解决这个,"Skill 会自我更新并带来收益"就没有干净证据。


## 2026-09-06 深夜:DEV-AUTO-1R 收口——**上一条结论被自己的分母推翻**;判词 `ACCEPTANCE_WAS_NOT_THE_BLOCKER`

报告 `docs/DEV_AUTO1R_COMPARATOR_AND_ACCEPTANCE_FRAMES_2026-09-06.md`;工件 `artifacts/main_protocol/dev_auto1r_repair.json`;
源 `evaluation/main_protocol_p4/{dev_auto1_revision(追加比较器),run_dev_auto1r_repair,smoke_dev_auto1r}.py`
+ `run_dev_auto1_skill_revision.py` 三处接线修复。0 LLM、54/100 新增 fits、~4 分钟(上限 2 小时)。

**下面这一条直接更正上一节。** DEV-AUTO-1 用"候选自己可读的格"对"父版本自己可读的格"求均值相减
(u16 = 7 格对 15 格)。统一分母后(READ 用真读数 / 确认不可执行按 raw 回退记 0.0 留在分母 / 未测记 UNKNOWN 扣留判词):

- **"9 个超父、5 个风险不更差" → 3 个超父、0 个风险不更差**(逐单元也是 0)。u16 的
  `outlier_iqr>winsorize` 由 +0.102497 变为 **−0.040504**。父版本逐格读数从预测库重建,
  13 个可核对步骤与 DEV-AUTO-1 记录**逐位一致**,0 fits。
- **三接受框架离线对照:A(现行绝对屏)=0、B(相对参照屏)=0、C(前向验证优先)=17 条提议/10 个程序进队列。**
  换掉绝对屏**什么都不改变**——上一节建议的头号实验(接受口径对照)答案在此,且是负的,不必实跑。
- **C 是唯一打开通路的**:5 个拿到 Support 准入(生产 `admission_policy` 预算,未加 treated≥5 的 Support 门),
  **2 个通过 delayed 权威门**(`outlier_iqr({})` u10 提出、u11 第一个窗口即通过;
  `outlier_mad>winsorize` u11 提出、u14 第 3 次尝试通过)。**晋升仍为 0**:还需 Fast 真的部署。
- **新发现的接线缺陷**:`run_dev_auto1_skill_revision.py` **从未调用 `install_policy`**,
  Fast 侧跑在模块默认的 `strict_positive_only`(伤到任何一条序列即拒)。run2 自己的 39 条 probe 重判:
  **strict 准入 3 条,released bounded 会准入 9 条**;strict 的拒绝理由 `relation_not_positive`
  不在 `RISK_REFUSAL_REASONS` 里 ⇒ **全课程 risk_refusal 恒为 0**,修订通道的上游供料一直是空的。已修接线,**未重跑**。
- **五次调用帽的真实代价**:修订臂 4 格抛 `AgentCallBudgetExceeded`,位置 **u8/u16/u23/u24**,
  正是**永远没进 `processed`** 的那四个单元 ⇒ 反馈证据面被静默削掉(u25 时提议器看到 17 格而非 21 格)。
- **接线修复(仅针对性测试,未重跑课程)**:① 控制臂只冻 Skill(原 `write_back=False` 同时关掉 Draft ledger、
  activate、bank/processed 四样,外加我加的每格重置);② Fast 预算退到包级(每臂 200 对称,
  **HEC-1 合同的 5 一字未动**);③ 装 released 准入策略。`smoke_dev_auto1r.py` 10 项全过。
- **四条更正**(细节见报告 §6):父版本不过屏推不出子版本都不可能过;**Scope 8 次失败 0/8 归因于栅格,8/8 是风险线**
  (`period_reliability` 四条边给出四个不同 treated 数仍全败)⇒ 上一节的②号建议失去依据;
  两次运行数字**逐位对调**说明两臂**可交换**(机制从未生效),不是"差异是噪声",n=2 也估不出方差;
  +0.407 那条 probe HF=0.25 越 0.20 线,**两套规则都拒**,不是被漏掉的好候选。
- **pos23 Support 仍是 UNKNOWN**:0 fits 确定性复现 `FaceNotEvaluable: scale floor`。剔除后
  `outlier_iqr>hampel_filter` 的 Support 由 −0.016248 **变号为 +0.001934**,而 delayed 仍 −0.031083(两面反号,泛化信号)。
  **delayed 面不受影响,审计的 delayed 结论原样成立。**
- **下一项最值得实跑(交裁定,未自行推进)**:在修好的接线上**按框架 C 重跑两臂完整课程一次**,其余不变。
  离线已排除上一节的①②两个候选实验;三个接线缺陷都没在课程级别被观测过。


## 2026-09-06 夜:DEV-AUTO-1 收口——**提议层通了,接受层挡死**;判词 `DEV_AUTO1_PROPOSAL_REACHED__ACCEPTANCE_BLOCKED`

报告 `docs/DEV_AUTO1_AUTONOMOUS_SKILL_REVISION_2026-09-06.md`;工件
`artifacts/main_protocol/dev_auto1_skill_revision__run2.json` 与 `dev_auto1_blocked_candidates__run2.json`;
源 `evaluation/main_protocol_p4/{dev_auto1_revision,run_dev_auto1_skill_revision,smoke_dev_auto1_skill_revision,audit_dev_auto1_blocked_candidates}.py`。
开发级;forward·A5-online·k1 入口态 + 其后 16 个 development 单元 × 2 臂(冻结 Skill / 自主修订)。

- **闭环已真实跑通到接受层**:放开提议后真实 LLM Slow 16 步提出 25 个合法修改(10 个不同 Workflow、3 个 Scope 方向),
  阈值一律由 `calibrate` 在冻结分箱上定,LLM 不批准自己的补丁。**弃权 2 次且都不终止课程**;
  相邻提议 12 改 1 同,失败反馈确实改变了下一次尝试。
- **0 个获准、0 次 Skill 更新**。机制:`outer_loop.screen` 是绝对门(任一适用单元越线即淘汰),
  而**持权卡本身在全部 16 步都过不了这道门**(`harmed_fraction`+`single_series_harm`,u6 起加 `aggregate_not_material`)。
  **9 个提议在全人群口径超父版本 ≥ material,其中 5 个两条风险线都不差于父版本,仍被淘汰。**
- **被拒候选值多少(事后 L2,0 LLM/106 fits)**:`outlier_iqr>winsorize` 在后 16 单元
  Support +0.170161(祖先 +0.141389)、**delayed +0.174513(祖先 +0.170298)、受损 39 vs 51、权威门 7/16 vs 4/16**;
  另一个 `outlier_iqr>hampel_filter` 权威面 **−0.031**。放宽屏会同时放进后者——本包**没有改屏**。
- **两臂效用差 = 运行间方差,不是机制**:两次完整课程的两臂数字逐位对调(frozen/revising 0.028763↔0.004286)。
  没有任何修订被应用 ⇒ 两臂本就是同一个系统。
- **第二层卡点(老问题,又量了一次)**:32 格中 20 格显式选 identity、9 格耗尽 5 次/格 LLM 预算,只有 3 格部署;
  探到 +0.407 仍选 identity。修订即使打通,下游也接不住。
- **Scope 出口仍是关死的**:8 个 Scope 提议全部 `NO_FEASIBLE_THRESHOLD`,原因是 6 个分数型特征的默认分箱
  `(0,1,3,6)` 有效切点只剩 0.0(M-R0f #3)。本包按授权未动栅格。
- **成本**:全包 828/1000 fits、440 LLM、≈17 分钟(上限 6 小时)。0 held-out、0 +144、0 TARGET_HELD_IN、0 密封;
  未改阈值/算子/route/surface/schema 文件/生产 runner 文件;未提交 git。
- **仪器**:agicto 中继现对 >约 4.4k token 的请求一律 403 `insufficient_quota`,本包改用本机已配置的
  DeepSeek(`deepseek-chat`)。两臂同模型 ⇒ 臂间对比不受影响;绝对数字与 HEC-1 **不可并表**。
- **下一项最有信息量(交裁定,未自行推进)**:① 接受口径对照(绝对屏 vs "不差于父版本"相对屏),
  本包已把两边所需读数留在工件里;② 修分数型特征的冻结栅格;③ Fast 的选择层。

## 2026-09-06 晚:ACC-1 事实归位(真实 Slow 弃权;纵向评的是确定性修订);6 pro 分析采认——撤回"减法不能创收益",新增"修订效应稳定性 / 面向动作观测 / 新会话再生成 / 普通反思臂 / A5 结构冻结臂"

- **ACC-1 链路事实**:m_r0d attempt3 真实 Slow 到达后**弃权**(`insufficient_public_evidence`),无子句无状态写入;`run_hec1.py:1665` 只送前 60 行 bank 给 Slow、
  影子用全量 100 行(m_r0e:60 行仍有 9 条可行三元组、最优 stump 同为 `z_peak>=6.0`)→ **Slow 输入截断进最小接线修复清单**,修好前"LLM vs 确定性"读数不成立。
  m_r0j 纵向评价的 `missing_fraction>=0.05` 来自确定性对照 m_r0i(0 LLM),不是 LLM 修订。
- **6 pro 采认(计划 §2 已更正、§7.0 三包已增补)**:① §2"减法修订不可能创造新收益"**数学上不成立,撤回**(收窄有益 ⇔ 被排除集合净负);② 新可学习对象 =
  修订效应 \(d_e(P\to P')\) 是否比程序效应 \(g_e(P)\) 更稳定(第一包读数,Scope 差分 0 fit);③ 面向动作的观测 \(\psi(x,P,\text{Consumer})\) 三个、版本化词表扩展;
  ④ 第二包加**普通 LLM 反思臂**与**新会话再生成检验**(区分 Skill 未用 / 用了行为未变 / 变了无助益),解释类型 ≤5、每失败 ≤2 差分,发车前冻结;
  ⑤ 第三包加 A5 **结构冻结只本地校准**变体;⑥ delayed 时间角色措辞、研究者修仪器 ≠ 作弊、序列次数 ≠ 独立样本。
- 研究中心(astra / 6 pro 收敛):反例到来后系统能否决定保留什么、修改什么并由后续证据检验;不赌 Scope 或 Workflow 谁强。

## 2026-09-06:ACC-1 第一层已走通(真实 Slow → 校准 → preflight → replay → 写入既有 Draft);纵向配对(m_r0j)判**这条收窄不值得**;候选收口;下一包只解决"修订选择目标对齐"

- **链路事实**:forward · A5-online · outlier_mad@z>=3 · k1;修订 = `∧ missing_fraction >= 0.05`;`DRAFT_REVISED`,`deployable=false`,无部署权。m_r0j:21 单元 × 2 面,84 fits,0 LLM。
- **口径更正(astra,主线逐格核对无误)**:后 16 单元(剔除边界前 5 个)delayed 面 修订 +0.106860 vs 祖先 +0.170298;受损 36 vs 51;**权威门 4/16 vs 4/16**;
  原报告"净损 2 门"的两个单元(pos 2、3)在修订前,不算后续退化。9 个行为不同单元 8 负 1 正。**结论:收益下降、伤害减少、过门机会未增**。
- **机制(只读核实)**:权威门/计分分母 = 全服务人群(`run_hec1.py:529-530`);阈值工具分母 = 被选中行(`scope_threshold_tool.py:193-194`);
  链上无一处以同分母比较修订与祖先。收窄总收益 = 祖先 − 被排除集合收益;校准 0.578 × 38/100 ≈ 0.220 < 祖先 0.251 ⇒ 被排除集合边界前已净正,
  对齐规则会在拟合前拒绝。收窄下受损不增是构造性的。Support 面过线 7→10 vs delayed 面 7→5:用 Support 面判收窄会得相反答案。
- **不能推出**:Scope 修订无效 / Harness 无效 / 需换 Workflow 面 / 需上分层模型。已从"能不能修订"推进到"选什么修订才值得"。
- **下一包(计划 §4.5)**:对齐提案呈 sol——同权威门分母过四线 + `aggregate_served(修订) ≥ 祖先 + material`,否则 `NO_REVISION`;五列开发比较
  (不修订 / 原规则 / 对齐规则 / 随机保留参照 / oracle 排除上界)+ 排除精度与代价;候选只用边界前证据;结果只算开发级。不调 missing_fraction。

## 2026-09-06:M-R0b 已出(7 步骤级机会 / 3 谱系 / 2 身份,按当时状态);**最小真实验收 ACC-1 任务书草案已写**(执行计划附录 F);暂不启动 M-W;待 Opus 补表填候选与预算

- **首选验收点** `forward · A5-online · outlier_mad@z>=3 · k1`(p04):当时 REVISABLE(attempts 1)、有 K0 祖先、可接收路径 = REVISE 既有 Draft(现有分支);
  后续可评 18 单元;as-run 曾到 Slow 并 `SLOW_ABSTAINED`。资格四条待 Opus 补表:修复后去重行集影子可行 ≥1、新 Scope 后续覆盖 ≥5 于 ≥1 单元、生命周期可接收、
  祖先存在。k2 同谱系不算独立复现,不得把假设的 k1 修订态接到历史 k2。备选顺位 ② reverse A5-online winsorize k2 → ③ reverse A3-online winsorize k5(仅第一层);
  皆不合格 → `NO_ELIGIBLE_BOUNDARY`,不换点不放宽。
- **本次只申请第一层**(真实 Slow → 校准 → preflight → replay → 写入既有 Draft);写入 Draft ≠ 执行权 ≠ 进化成功。第二层(历史前向重放的验证→重遇)预注册可选、预算另批。
- **信息边界**:Slow 只看 k1 之前的 bank 行、Draft 字段、冻结词表;影子最佳子句 / 机会清点 / 后续读数一律不入 Slow;影子只记 agree/disagree。
- **预算**:物理 Slow ≤2(现行每步顶,第 3 次不可达如实记);replay 屏 fits、后续验证 fits、祖先影子 fits = **待补**(Opus 按真实路径核);U-1 持久化每次尝试工具结论;
  U-4 两种仪器基线先核。"每步顶 2→3"若要须单列前瞻批。**未经用户批准不跑任何真实 LLM/fit,不开 Phase F。**
- 结果类别预写(`SLOW_ABSTAINED` / `NO_FEASIBLE_DIRECTION` / `PREFLIGHT_REJECTED` / `REPLAY_REJECTED` / `LIFECYCLE_REJECTED` / `ACCEPTED_INTO_DRAFT`;
  `OUTER_BUDGET_EXHAUSTED` / `TRANSPORT_FAULT` / `TOOL_FAULT` 单列;第二层三类)。**成功或失败都先收口本层再裁定**;`ACCEPTED_INTO_DRAFT` 不自动授权第二层或三臂;
  Slow 未提出不构成换编辑面的依据(影子已示 Scope 空间有可行修改);真实 Slow 的不同子句不得借用影子子句的覆盖资格;存档用脱敏请求内容 + 字段来源,不新增 SHA/Hash。
  本轮不展开多数据集 / TSFM / AD / 下载。

## 2026-09-06 凌晨:R2 确认交 Opus 实施;R1 拆为 R1-a(仪器,本轮)/ R1-b(学习策略,未批);以 Opus 最新补修为准,D1/D2 主体/D4 主体已修,不重派

- **Opus 最新工作树补修(主线只读核对)**:D1 ✔(`open_restricted` 共用 (program, root) 守卫含已关闭;`root_for_scope(program_steps=)` 歧义拒绝);
  D2 主体 ✔(同单元同精确读数同 relation 折一条;不一致单元打标)——**残留**:不一致观测仍各计一次;D4 主体 ✔(执行器级 Support fits 含异常路径;
  baseline 执行/缓存/失败分列计数,定价留协议;Fast 逐请求计量,顺序帽传输前拒绝);D3 待 R2 实施;D5 反例测试待补。补丁整体仍 `NEEDS_FIX`。
- **R1(astra 拆分)**:R1-a 本轮 = 身份 (census_key, 单元, 面, 精确读数, relation)、精确去重、不一致打标 + **一个单元只计一次:不一致则 positive/adverse
  都不计并单列**;R1-b = "NARROW 只看部署证据""ADD 取最佳"是学习策略,**不作为去重规则默认批准**,与 delayed 证据用途一并由 sol 裁。
- **R2 已确认(计划 §4.4 实施要点)**:propose 阶段查 `by_program_and_root`——无 Draft → NARROW 开第一张;REVISABLE 且可加子句且已验证 → 归并为
  REVISE(`by_id` + `record_revision`,计数不变);FLAGGED/WAITING/已关闭/额度尽 → `NO_NARROW_TARGET(reason)` 入 `rejected`,不调 Slow;
  `LINEAGE_ALREADY_HAS_A_DRAFT` 留作兜底并断言为 0。
- **原路径可达性重放的口径(astra)**:历史清点按**各外环步当时状态**,不用课末回填;分层漏斗每层标来源——影子 stump 可行只是"影子可行",
  **真实 Slow 提议 / replay 通过 / 后续独立验证一律标"未评估"**,不得用影子可行代替;26/6 只作旧语义参照。
- 派工(Opus):R1-a 补条 → R2 实施 → D5 反例测试 → 非作者复核 → E.5 重放(0 fit)。不开长课程、不批新预算。

## 2026-09-05 深夜:Opus 接线补丁 astra 复核 → **首因已修复(C1–C7),补丁整体 `NEEDS_FIX`(D1–D5),尚未具备科学发车条件**;主线出 R1/R2 提案与 R3 意见呈 sol;勘误 7/11

- **已闭合**:Episode relation 进 bank(不再把 admission reason 当关系)、admitted ≠ POSITIVE、`by_scope` 按程序、`restrict` 对 (program, root) 封顶含已关闭、
  Fast 计费与异常路径补记、ADD/NARROW 集合拆开、REVISE 仍改原 Draft。
- **待修(Opus,下一棒)**:D1 `open_restricted` 守卫不完整(`restricted_draft.py:468` 只查 census_key;`root_for_scope` 无 program)→ 与 `restrict`
  共用守卫 + 三条反例测试;D2 普查同单元重复行抬高 `unit_count`/`adverse_units`(`outer_loop.py:339-342`)→ 按 R1 去重;D3 `NARROW` 命中已有 Draft
  在付费后被拦 → 按 R2 在 propose 阶段处理;D4 计费三处(共享 baseline 从未入账、probe 后异常丢 fits、`reserve` 不按剩余额度封顶 ⇒ 499/500 可到 504);
  D5 e2e 是三臂无 K0、修订断言可 skip,不作验收。**不加长 e2e、不为 NARROW 加 K0 课程。**
- **主线两项语义决定(计划 §4.4,呈 sol)**:R1 证据单元 = (谱系, 单元, 面, 动作);完全重复去重;身份同而 fingerprint 异记仪器异常不聚合;单元投票每谱系每面一票——
  NARROW 只看实际部署动作,ADD 取最佳;分面计。R2 已有 Draft 承接 NARROW:无 Draft → 开第一张;REVISABLE 且可加子句 → 归并为 REVISE;FLAGGED/WAITING/
  已关闭/额度尽 → propose 阶段记 `NO_NARROW_TARGET(reason)`、不调 Slow;不新开壳、不清零、FLAGGED 仍禁。R3 共享 baseline 按单元记一次进 `baseline_fits`,
  归整场成本不分臂;异常路径 fits 在执行层累计。
- **三句修正**:M-R0 的 26/6 只作旧语义历史参照,重放不必凑回;重试次数不得绕过每步物理硬顶,提额须前瞻批;**风险谱系 7/11、Draft 来源 8/11**(此前误写 8/11 谱系,勘误)。
- 下一步很短:补 D1 守卫(含 `root_for_scope`)→ 0-fit 原路径可行性(含 D2 去重、D4 剩余额度)→ 单独批准最小真实验收(现有 repair 测试 + 上述反例)→ 再谈自然修订闭环。
  R2 未裁则 `NARROW` 在"Active 已 restrict"几何下仍到不了生命周期接收。

## 2026-09-05 晚:第一批核查完成(M-R0 + 计费 = Opus;数据资产 = grok);HEC-1 追加"实现限制"解释;下一路径固定为 接线修复 → 复核 → 原路径可达性重放 → 再定 M-W

- **HEC-1 判词与冻结数字不变**;追加解释:外环 `ADD` / `NARROW` 在 30/30 步被**普查 relation 词表接线**阻断(`run_hec1.py:1849-1850` →
  `outer_loop.py:209-211/363-365`;反事实 26 `NARROW` / 6 `ADD`);delayed 面不入 bank ⇒ 8/11 Active 谱系的风险线冲突对普查不可见(**协议事项**,
  呈 sol);阈值工具 5 次提议 `CALIBRATED` 0 ⇒ 修好词表不是充分条件。**部署权从未丢失**;"撤销先于外环"假设撤回,`SUSPENDED` 不立项;
  8/11 张 Draft 来自持权卡(6 张 K0);冻结臂丢弃 15 张 Draft 系设计行为。计费差 422 = 187(`spent−1`)+ 235(47 cap 格,推导);
  fits:delayed+评价 542(记录)/ Support 探针 641(推导)。工件 `artifacts/main_protocol/m_r0_reachability.{json,md}`。
- **数据角色(grok,`m_data_asset_audit.{json,md}`)**:仅 KDD 含缺失接上 HEC-1 serving;NOAA 无 HEC-1 几何工厂(M0-b 阻塞);Solar 无 loader、
  0 缺失不排除 K0;NAB 本地五族全部已用 → 只剩 development;Yahoo 24 曝光(字典序)/ 41 密封。本地无合法的带缺口预测 fresh Target,
  推荐 Monash Wind Farms Minutely(含缺失,339 条,最短 6345)——下载与适配器另行申请。
- **M-W 按 M-R0 更新、暂不发车**:8 情境 / 20 组合 / 7 个 coverage-only 不可修(不买)/ 11 个组合为分母;E=10 拟合前上界;必做 1056–1188 fits。
- **下一路径(astra)**:① 五项最小接线修复(relation 词表、`by_scope` 比程序、`restrict()` 写 `census_key`、重试循环头、计费)→ ② 非作者复核
  → ③ 0-LLM 原路径可达性重放(普查 + 影子 stump)→ ④ 可行 `NARROW` 为 0 才呈批 M-W。**不开长课程、不扩方法。** 计划 §7.0 / 附录 E.4–E.5。

## 2026-09-05 更新:设计进入**机制识别阶段**;可执行推进计划已成文 → `docs/NEXT_EXPERIMENT_EXECUTION_PLAN_2026-09-05.md`

- 方法 v2 降为 Architecture A 基线;协议 M r2(`M_MECHANISM_MEASUREMENT_PROTOCOL_2026-09-05.md`)**未冻结**,已按 astra 评审修正口径;
  **自然持续修订链仍为 0 条**——这是下一阶段唯一优先问题。
- 只读核实(course 工件):HEC-1 外环三顺序 `ADD` 0 / `NARROW` 0 / `REVISE` 3(1 弃权、2 预算耗尽)/ 修订 0;11 张 Draft 全来自
  "过 Support、败 delayed"(`run_hec1.py:2121-2139`),无一来自曾 Active 的 Skill。
- 路线:D1–D2 数据资产审计 + M-R0 修订可达性审计(0 fit)→ D3–D7 核心测量(M-W Workflow 可修复性 / M1 / M3 / M4 / M5)→ D8 选路
  → D9–D10 三页方法说明 + sol 冻结 → 最小闭环(A / B / C 三臂 + Static;目标 = ≥1 条自然 Chain-S)→ 跨数据集主实验。
- 待 sol 七项裁定、待用户四项授权见计划 §8;**HEC-2 live 草案暂缓**;TSFM 下载未批不阻塞 Ridge 只读分析。
- **astra 二审后(同日)**:关键路径收缩为 ① M-R0 + 计费核对 + 数据资产审计(0 fit,可立即安排;计划附录 E)→ ② 小规模 M-W(附录 D:
  4 个自然失败情境 × 3 顺序,有限修改集 E≈6–8,只用当时可见的 Support/delayed 选修改;**必做 ≈1050–1150 fits,待用户批**)→ ③ 有
  `REPAIR_SIGNAL` 才冻结编辑面、接线、跑 A/B/C 三臂。口径修正:HEC-1 = "知识已进入并使用,结构修订未完成";Solar 用途待核实;
  `course_fits` 不含 Support 探针,预算按调用类别重列;一条链 ≠ 系统有效(同报 B−A 整段收益与 B−C)。**暂不启动长课程、不扩新方法。**

## 2026-09-04 21:xx 更新：HEC-1 三顺序已收口，判词 `HEC1_EVOLUTION_NOT_SUPPORTED`；0-LLM first-fault 诊断完成

- scientific Forward / Reverse / Interleaved 均 COMPLETE、仪器门全过，同 commit `d690850`；26 个计划单元、23 个可计分。
- P1 未成立：`D_o>=0.115` 为 1/3 顺序，cohort 正向 2/4；harm 条件成立。P2 未成立：修订 Draft=0、存活重遇链=0。
  Phase F **保持关闭**；不得把该判词写成“进化普遍无效”。
- 0-LLM 诊断：Best-Safe-Global 在 14/23 单元有安全 non-identity headroom（累计 `+5.527089`）；validation-search
  在 17/26 单元找到 Support-safe 候选，但 16 个可评 non-identity 部署只有 7 个在 +144 仍过四线；34 个 Support-safe
  候选逐一重部署，仅 10 个保持安全，只有 1/16 个机会可由换候选救回。
- first fault = **Fast 供给不足 + Support→未来窗口的安全/效应不稳定**。算子菜单不是首要空点；单纯提高 LLM 调用或扩大搜索
  不能解决未来尾部。下一项仍为 HEC-2 per-channel 单假设，不重跑 HEC-1。
- 完整边界、原始数字、claim ceiling 与工件指针见 `docs/HEC1_ZERO_LLM_DIAGNOSTIC_CLOSURE_2026-09-04.md`。

## 09-04 21:xx:HEC-1 收口 `HEC1_EVOLUTION_NOT_SUPPORTED`(冻结判词);人印 = **treatment-sparse、未识别**;设计阶段关闭

- 三顺序 26/26、同 commit `d690850`、仪器 9 项全过;D_o +0.212 / −0.043 / +0.006;62/69 平局,7 分歧中 5 预算介导;P2 链 0;
  harm online ≤ frozen;P3 +0.143(描述性)。计费少记 422(物理 1088 vs 账面 666,各顺序仍 <500)。
- 最终有界裁定(`HEC1_CLOSEOUT_DESIGN_RULING_2026-09-04.md`):first-fault ① 预算语义 ② 跨窗口安全不可迁移 ③ Fast 覆盖 ④ 路由
  ⑤ 算子 ⑥ 记忆空转;HEC-2 = Stage A 0-LLM 换 Consumer 重算 → 2×2 → 仅 SHAPES∧RETAINED 开 Stage B;前瞻 discordance 门 +
  新词 `HEC_UNIDENTIFIED_TREATMENT_SPARSE`;cap 改语义预算;scope-matched gate audit;Phase F 最多开封一次。
- **下一步序**:计费勘误 → endpoint composition → gate audit(≈700 fits)→ Stage A(≤3000 fits)→ 2×2 定动作 → 数据资产审计 →
  Dataset/Domain 合同。待 sol 三处一致性裁定、用户三件 fits 授权。

## 22:xx 更新:Forward 首次尝试在第一次外环 Slow 崩(`harness_view={}` 接线错误)→ `RUN_BLOCKED_NO_VERDICT`

- 授权:修 `OuterSlowAgent` 的 `harness_view`(对齐 Source-v3)+ 新增"外环候选经真实 `core.run_stage`"测试 → 聚焦 + 回归 →
  grok 增量复核 → allowlist commit → **Forward 从 0 重跑,不 resume**(一条顺序一个 commit)。
- K0 / Phase S 不重跑(修复路径 Phase S 未执行),合同勘误披露两 commit;崩溃检查点改名 `forward_v11_attempt1_blocked` 留证。
- 35 LLM 记仪器开销单列;Forward 500 信封重起(待用户点头)。
- 观察:A3-online 第 5 单元即有需 Slow 的候选——外环在 Target 上是活的。

## 20:xx 更新:Phase S-v1.1 收口,**K0 非空(1 张卡,`outlier_mad` @ `z_peak>=3`)**,审计 CLEAN;Phase T 四臂 chain 已发车(pid 51112,Forward 0/26)

- 激活链:`[200:239]`×2616,Fast 自提 → Support 过 → delayed 四线全过(+0.360 / hf 0.10 / msh 0.13)→ Active;评价面 +0.473。
  **项目首张自然数据上经权威门存活的 Skill;A5 首次被实例化。**
- 保留:卡来自内环生命周期(非外环 ADD 链,两步外环无候选);同程序的 Draft 曾在 1944 因持续成员受害 FLAGGED;卡在 Phase S 内
  未被复用——重遇与匹配全在 Phase T 考(判据 3 可评分)。
- 待 sol 固定门:确认 K0(审计已过,chain 按预授权自动续跑)。

## 在哪(sol 确认主线方向,2026-09-03 夜;两处状态已更新)

- **HEC-1 v1.1**:三件不可拆分修复 + 六项合同同步已落地;主线只读复核 **PASS(条件式)**。**gate 三分类已裁定并落入代码**:
  风险线分歧或任何状态泄漏 → `AUTHORITY_BYPASSED` 降级;仅覆盖线分歧且状态完全不变 → `AUTHORITY_UPHELD` 只披露;
  `LOST_ACTIVATION` 计数披露。**缓存计数合流与召回归因行为锁已写入工作树**——现在不是等设计,是等测试与非作者增量复核。
- **唯一主链**:聚焦测试 + 全量回归 + smoke → 非作者增量复核 → allowlist commit → Phase S-v1.1(≤120)→ K0 审计 →
  三顺序(各 ≤500,仪器门自动推进,途中禁止修改跟踪文件)→ 统一读数 → 主线判词 → sol 确认 → Phase F(非空 K0 ∧
  SUPPORTED ∧ 用户开封)。
- **两条执行纪律(sol)**:① 最终 commit **必须包含**本次权限修复涉及的 `methods/ttha/method.py`、`online_loop.py`,否则
  提交的不是真实运行闭包(主线注:`method.py` 在 h0 `runtime_bundle_sha` 依赖图内,入 commit 即轮转 lock → 须同步
  `--write-lock` 并作仪器变更披露,否则回归差集会出现 lock mismatch"新失败");② D5/D6 只能在独立 worktree 运行,或
  在最终 commit 后只写隔离工件;十小时科学运行期间不改主工作树。
- **概率说明**:主线给过的"三成 / 四成 / 两成"只是**规划者主观判断**,不入合同、不用于决定是否继续;科学判词只依据
  预注册读数。
- **shakedown Forward**:26/26、165 LLM、3.07 h、无 RunFault;仪器数据保留,不进曲线;N_T_eff = 23 实测。
- **方法设计已关门**;Instruction 面不作要求(用户);此后只有执行与读数。
- **并行 0-LLM 诊断**(不影响 HEC-1):D5 2×2 因果分解 + D6 逐序列持续性(`docs/D5_D6_…`,接收 grok)。
- **HEC-2 ① per-channel 预注册草案已写**(`docs/HEC2_PERCHANNEL_…`),HEC-1 收口后只填数值锚点即冻结发车。

## A5 为什么至今没起作用(sol + 主线共识,一句话)

可迁移 Skill 要串联跨过:稀疏 Program headroom → Fast 提出 → Support 过线 → delayed 过线 → 形成 K0 → Target 匹配供给
→ 独立重遇仍安全;四次空 K0 分别死在不同环(格式/零候选/尾部;新进入者/翻号/覆盖;Support 无准入 + 阈值 2;负先验全局化)。
核心结构问题两个:Scope 识别"缺陷像不像"而非"处理后会不会获益";pooled 下 Scope 把"处理 context"和"切换模型"绑成一个
动作。算子有缺陷但排第三;D1 未证明路由是主因(强嫌疑 + 放大器);尾部门是绑定约束但其跨窗口稳定性从未校准。

## 文件所有权(同一时刻一个写者;违者一律回退)

| 文件 / 目录 | 写者 | 其他线 |
| --- | --- | --- |
| `evaluation/main_protocol_p4/run_hec1.py`、`outer_loop.py`、`restricted_draft.py`、`scope_threshold_tool.py`、`hec1_contract.py`、`hec1_scoreability.py`、`audit_hec1_*.py`、`tests/main_protocol/*hec1*` | **Opus**(commit 后至三顺序完成:**全体只读**) | 只读复核,发现写入报告 |
| `artifacts/main_protocol/hec1_*`、`.aris/runs/hec1*` | Opus 的 runner | 只读 |
| `artifacts/main_protocol/p4ab*/p4ac*/p4ad*/p4ae*` | 各自的 grok 审计脚本(一次性) | 只读 |
| `docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md` | **锚点制**:每线只在自己的锚点行下插入(见账本顶部 `<!-- ANCHOR:… -->`) | 不动他线条目 |
| `docs/HEC_EVOLUTION_MAINLINE_PLAN_*`、`HEC1_*REVIEW*`、`D*_*任务书`、`HEC2_*`、本页 | 主线(Fable) | 只读 |
| `docs/OPUS_HANDOFF_BRIEF_*` | 主线写、Opus 读;Opus 的执行记录写 `MORNING_REPORT_*` / `HEC1_RUN_REPORT_*` | — |
| `docs/FABLE_FINAL_REVIEW_AND_SUCCESSOR_BRIEF_*` | 第二审查线 | 只读 |
| 项目 `AGENTS.md` | sol 裁定后由主线写 §5 状态锁;其余时间只读 | — |
| Phase F 开封 | **用户** | — |

## 到读数前的纪律

commit 后 runner / 合同 / 测试只读;账本只追加**仪器条目**;不出任何新裁定;让机器跑。

## 模式切换(用户裁示,2026-09-03 夜):**关闭三线并行头脑风暴,改单线**

- **执行**:Opus 一条线,按 `OPUS_HANDOFF_BRIEF` §2d/§2e 走完 commit → Phase S-v1.1 → 三顺序 → 停;只报仪器与计数。
- **审计**:grok 只接主线派的只读任务(D5/D6;课末读数实现;非作者复核),不参与设计讨论。
- **主线(Fable)**:唯一的方法整合与裁定建议出口;每日**一份**状态更新(本页),不再逐条入账讨论。
- **sol**:只在固定门出裁定——gate 三分类一句、K0 非空时的 K0 确认、课末判词确认、HEC-2 ① 冻结。
- **用户**:只需在四处按按钮——转 sol 一句、派 grok 一次、看判词、决定 Phase F。
- 第二审查线**停笔**;其文档保留为归档,不再新增。任何新设计想法一律记入 `HEC_EVOLUTION_MAINLINE_PLAN` 的"HEC-2 之后
  候选池",读数前不讨论。
