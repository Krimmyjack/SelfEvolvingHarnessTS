# DEV-TRAIN-2：减害成立，新增准备收益未成立

完成：2026-09-10 23:09（UTC+8），恢复进程exit0，最终四臂评分完整。
判词：`SKILL_CHANGED_EXECUTION__IDENTITY_RECOVERY_ONLY__FIXED_WORKFLOW_SUPERIOR`。
本页为根Agent本地工件复算，review_independence=same-context，结论暂定；不是独立复审。

## 1. 实验和结果

10条训练序列逐条生成W，汇总后每臂训练一个共享Ridge；10条评价序列逐条生成V。
训练整段192输入+48历史目标可准备，评价真值不动；Fast无当前下游反馈。
形成u9/u10/u11，评分u12(origin2616)/u13(origin2856)，全部为已曝光含缺失KDD开发数据。
四臂同一cap=1.0、相同工具/Consumer/人口；旧0.35与Flash20/20不并表。
收益是Static sMASE减本臂sMASE，越大越好；两窗内各10条完整人口，窗间等权。

| 臂 | u12收益 | u13收益 | 两窗平均 | 受损次数/20 | 最坏单次伤害 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static | 0 | 0 | 0 | 0 | 0 |
| 形成段选定winsorize→FFT | +0.670837 | +1.028954 | +0.849895 | 1 | 0.275134 |
| 旧Skill | −0.129358 | −0.200022 | −0.164690 | 14 | 0.639286 |
| 新Skill | 0 | 0 | 0 | 0 | 0 |

新减旧为+0.129358/+0.200022，平均+0.164690，但新臂等同Static。
固定Workflow在形成段按完整人口选定，不由u12/u13结果挑选；其最坏伤害和受损次数仍须同报。
20是10个实体在两个窗口的序列次数，不是20个独立样本。没有种子重复或显著性主张。

## 2. 知识怎样改变行为

Slow一次自主ADD `missingness-unknown-probe-identity`，父版不变；并非人工指定FFT。
新卡条件为forecast且missing_fraction>0，实际视图送达检查10/10训练、20/20预测。

| 决策角色 | 旧Skill | 新Skill |
| --- | --- | --- |
| 训练10条 | 5线性插补、3 MAD、2 identity | 10 identity |
| 预测20条 | 19线性插补、1 MAD | 20 identity |

新臂30条都为有效`IDENTITY_SELECTED`/`explicit_identity`，不是异常伪装成零收益。
其共享模型系数与Static相同，两窗逐UID损失与Static精确相同。
新旧最终60条全部有效、无fault，记录模型均为grok-4.6-build。

## 3. 已知问题与仍待检验的解释

卡正文明确建议“Prefer identity when probes are unknown”。实际四个probe direction字段
全为unknown；`runtime_capabilities`已告诉Slow：本轮没有fixed probe panel，没有工具能取得
positive probe，unknown不代表测得负效用。这里存在一个直接可见的决策规则问题。

形成段七个完整Workflow的训练/服务条件对照确实在`slow_input.json.xy_decomposition.readings`，
不是只给失败而没给成功。输入文件约31.6万JSON字符，含大量重复窗口明细；这说明“有材料”
不等于“模型有效使用材料”，但本轮没有证据证明端点截断或长度是唯一原因。

现有General构造指导仍偏向“缺陷假说→相应机制修复”，含旧Support/prior/Experience措辞。
这可能限制Consumer条件化的平滑/组合准备，尚未通过消融定位。不能把一轮全identity全部
归因于某句提示，也不能说取消35%已经证明自进化有效。

根的设计判断：下一步优先检验General生成/选择知识更新，允许历史合法效用支持候选默认
Workflow，再让Specific表达有依据的局部差别；不由研究者硬写FFT，不强制多步或非identity。
TRAIN-1B已经PATCH过General且未超过Static，故“改General”本身不是新发现或成功保证。
本次拟检验的增量是：把完整方案相对效果写成无当前反馈也能使用的决策知识。

## 4. 故障、恢复与成本

第一次运行在Slow已编译子卡后，边界JSON因嵌套mappingproxy序列化失败而退出。
修实验层JSON，依据已存子快照/provenance恢复边界包装，未重问Slow或改卡。
原始LLM提议响应、行为预测/尝试日志未保存，记UNAVAILABLE；不从卡片反造原响应。
恢复核验0 API/0 fit，16份原文件字节不变，40条形成决策未重问。

形成段有2条失败（u11/T141传输失败、T149准备失败）；该窗总体效用UNKNOWN，已进入
原Slow材料的证据状态保持不变，不在Slow之后补跑或把它们填零。故不能称整包100/100无故障。
最终用于主比较的新旧60条及四臂20个序列窗口评分完整。

整包500次实验API尝试：形成195、Slow1、旧152、新152；10次真实拟合。
其中Static1、六菜单6、形成旧模型1、最终新旧各1；固定臂复用形成段获胜模型。
全局Fast峰值4、fit峰值1；后续评分发生在最后Fast输出冻结之后。
恢复段约40分钟，包预算墙钟5733.19秒（含中断至恢复的间隔），没有重置预算。
研发调用和合成烟测不计入上述实验API/fit；此前Grok恢复审查8轮耗尽未交结论，不记独立通过。

## 5. 状态与证据

骨架已经按用户目标运行；这一轮没有取得高于Static的新Skill收益，更没有胜过固定Workflow。
保留General/Specific、逐序列Workflow、共享模型和Slow边界更新；完整A5/跨域积累尚待验证。
后续想法暂存`DEV_TRAIN3_GENERAL_GUIDANCE_TASK_2026-09-11.md`，属于未批准讨论草案，不能执行。
用户当前只让Flash保存/commit现有进度；下一实验须先与用户讨论。

- 主原始工件：`_scratch/dev_train1/runs/dev_train2_uncapped_workflow_10x10_20260910/result.json`
- 原Slow输入、父子卡、逐决策、错误归档及启动收据仍留原位置；不覆盖。
- 可提交精简证据：`artifacts/main_protocol/dev_train2_closeout.json`（含逐UID数值、60条决定与卡正文）。
- 复算：`python _scratch/dev_train1/audit_train2_result.py`，0 LLM/0 fit/0数据集读取。
- 实现与原规格：`evaluation/main_protocol_p4/dev_train2_workflow.py`、`DEV_TRAIN2_UNCAPPED_WORKFLOW_SKILL_TASK_2026-09-10.md`。

保存Git的是代码、必要入口、文档与精简证据，不是全部本地模型缓存/运行目录的备份；
不能在另一台机器只凭本提交恢复缺失的原始运行目录。未push，未开启密封评价。
