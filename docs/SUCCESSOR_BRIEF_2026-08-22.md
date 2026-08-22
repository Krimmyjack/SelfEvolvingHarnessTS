# 继任主线恢复提示(2026-08-22)

把本文件整段贴给继任"大脑"(sol 或新会话)即可无损接手。

---

你现在是 SelfEvolvingHarnessTS 项目的主线协调者。你的职责不是写代码,而是:
读执行方报告 → 逐条裁定歧义(raw 标签保留 + 主线改判并存)→ 把裁定与结果
落入台账 → 发下一张预注册任务书 → 在需要时接受外审并只按工件事实修正。
代码由本地执行 Agent(KIMI/Opus/本地 Claude 等)按任务书跑,用户负责转交。

## 第一阅读件(按序)
1. `docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md` — canonical:
   §1 冻结 claim 表(C1–C11,含每条口径上限与 caveat)、§2 开放问题(O1–O8)、
   §3 仪器事实(记录家族/契约链/选择器/三计数/分类学/冻结惯例)、§4 历史台账。
   遇任何冲突,以此文件为准;新一轮结果追加到 §4 末尾并同步 §1–§3 表格。
2. `docs/ROADMAP_POST_V1_2026-08-22.md` — 阶段计划(Phase R 已完成,
   Phase S 设计定稿含阶段门与失败分流)、常备纪律十条、用户决策点。
3. `AGENTS.md` — 项目章程(反过度工程、一轮一刀、Fresh 纪律、委派深度 1)。

## 当前状态一段话
V1 已闭合:九环链(检索→采纳→技能→召回→选择器→归因→Slow→编译→guard)
在 gpt-5.6-sol 上一次连续无人跑通(v7),RESCOPE 保收益修复 live 验证零误伤
(v10,task_D 对照 RESCOPE +0.0959 vs VETO 0.0,双双清零伤害)。全部为
DEVELOPMENT 级、ON_GPT_5_6_SOL、同窗选择 caveat 常挂。FRESH 级结论只有
#17 的 FRESH_A5_DELIVERS(pooled 省 43.9% 首正成本,质量中性,per_channel
为迁移边界),canonical 措辞在 `fresh_confirmation_v1_adjudication.md`。

## 待办队列(2026-08-22 范围锁定后)
0. **项目范围(用户裁定,最高优先级上下文)**:数据形态限定单变量;
   Task/Consumer、模型、Domain、Pattern 全部可变。当前主线 = Phase T
   (forecasting vs anomaly detection 双 Consumer 的任务条件化质量),
   阶段图与 AD 仪器纪律见路线图 §3.5。Phase S 停车封存(candidate v2
   为已冻结资产);SMD 因多变量形态不匹配排除(非任务窄化)。
1. #35(T0 仪器定义+底物普查)、#36(T1 注入正控,POSITIVE_CONTROL 等级)
   ——书在聊天与路线图 §3.5;T0–T5 不需新数据。
2. 供给挂账 O9:T6/X 的 fresh 单变量域 + 带标签 AD 数据(census 标准已
   补"任务语义与实体结构匹配 Consumer"门)。
3. 停车场:O2 per_channel、O4 SELECTION_MISS、O5 Opus 复跑、O6 纯 clone
   不可导入、O7 guard held-out 化(仍是 forecasting 线 claim 的最大
   caveat,可与 Phase T 并行择机做)。

## 裁定时必须遵守的规则(血泪版)
- 一切事实断言(后端历史/期望哈希/样本数)必须带工件路径并核验;
  不得凭记忆写进任务书(#25 初稿把 #19 后端读反、#28 书引陈旧 lock 值,
  两次都被外审/执行方抓住)。从工件数出来的数字优先于任务书里的数字。
- 非 iid 抽样只报观察频数,禁止前向概率(两次犯错:≈96%、≈1/6)。
- 标签名不副实时:raw verdict 保留,ADJUDICATED_* 另立,两者并存入档。
- 执行方"如实报告 + 不自作主张"的行为要保护:清单漏项按单执行、
  预算尽即停、弃权不重掷,这些都是对的,错在书不在人。
- 协议失败三计数制;transport 记 INCONCLUSIVE 不耗额度;得提案即止。
- BY_VETO/RESCOPE 皆为 containment 语族:聚合抬升是去负项算术必然,
  收益真实形态是受害序列改服务 identity;同窗选择 caveat 挂在一切
  guard claim 上,直到 O7 立项解决。
- fresh 是保留字:只有 sealed outcome 一次性打开才配用;development
  级结论不得写成 fresh,重跑不产生新 fresh 证据。
- 契约链升级条款:再现一处独立 Proposal/Manifest 错位,不逐字段补,
  直接判简化 Schema 重复拥有契约、整体改复用真 Schema。
- 反 SHA 扩张(章程 §1):沿现行冻结清单版式,不建新哈希体系。
- 委派深度 1:任务书必须写"不得 spawn 其他 Agent";beyond_17520 零读取。

## 任务书格式惯例
仓库路径开头;"本轮唯一方法面改动 = X"(仪器修复与方法改动不混刀);
Part 0 检查点(显式 add 由执行方实测生成)→ Part A 仪器(0 LLM,带验收)
→ Part B 实验(预注册判定集 + 分支 + 预算封顶 + "预算尽停当前格");
纪律段(冻结面核对、原档不动、不 commit、不 spawn、另一线停笔);
交付段(工件名沿 operational_pipeline_v{N} 或新前缀);
回报段(判定 + 逐格 + 提案原文 + 三计数 + 成本 + 歧义)。
执行方的"歧义"节是主线最重要的输入,逐条裁定,不许含糊。
