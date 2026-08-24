# 数据质量与反馈模型(短正典)

建立:2026-08-24(主线,依 sol 审读裁定)。来源:Data_Quality_Disgussion.md(研究讨论档案,非路线权威)+ C24–C32 实验证据链。本文件只保留经裁定的正典内容;修订须经主线,与两份状态文档(STAGE_REPORT / ROADMAP)一致时以后者时间戳新者为准。

## 1. 数据质量的定义(Consumer 相对)

"高质量训练数据"不是看起来更干净,而是**让固定的下游 Consumer 在原始 Query 上表现更好**。
- 动的是训练底物;Query/held-out 原始字节永不处理(协议墙)。
- 下游模型规格冻结,只当判官;Harness 进化的是数据准备 Workflow/Skill。
- identity 也是一种准备:改了反而伤 Consumer 时,正确动作就是不改。
- 不存在对所有任务都更好的一份"优质训练集"。

## 2. 各任务的质量语义

| Consumer 族 | 好训练数据 = | 主要危害 |
|---|---|---|
| Forecasting(ridge/DLinear) | 保留可预测结构;减少影响外推的缺失、噪声、伪尖峰 | 过度平滑伤趋势(相对轻) |
| AD 背景模型(iforest/PCA/一类) | 忠实覆盖正常边界(含合法尾部);污染只在拟合时剔除/降权 | normal_boundary_shrinkage、false_alarm_amplification |
| AD 监督事件 | 异常事件 = 正例证据,必须保留;标签几何与特征窗对齐 | event_erasure(实证形态:清洗致正例行退出拟合,369→184) |
| Classification | 标签可信、类覆盖完整、决策边界信息保真 | 全局粗分类近预测语义;局部事件分类近 AD 语义(双契约在册) |

原则:预测就绪允许甚至奖励"变形";AD 背景族就绪惩罚变形("去污染,不变形")。该原则为 Consumer 族条件化,不是 AD 公理。

## 3. 三层反馈模型

```text
Support(held-in,高频,可用连续代理信号)  → 只能形成 Draft
delayed(held-in,独立事件级 event-F1)     → 才能批准/撤权
held-out(冻结部署,零反馈)                → 外部一次性开 outcome 计分
```
规则(钉):
- 零事件窗只提供误报伤害证据,**不得授权正向采用**;
- 反馈以事件质量计,不以时间份额计;每书报告反馈窗事件数;
- 连续信号(AUPRC/margin/误报率)仅 Support 侧排序/起草;delayed 与最终判官必须使用冻结的 Consumer 真实任务效用,不得由代理信号自批。Yahoo-24 当前实例使用 event-F1 与安全门(宏 > +0.005、受害 ≤2/24、worst ≥ −0.02);这些数字不是跨数据集/跨 Consumer 的永久常数;
- 晋升单位 = 完整 policy 在 cohort 上的宏效用 + worst-case harm;决策/反馈单位 = series/subgroup;
- Memory 必须能表达并保留 **Positive + Negative + Conflict + Abstain 四类**,机制验收应覆盖四类;不得为填满类别而强造自然结果。缺少真实 Positive 证据时不得把负例库包装成完整适应能力,否则系统会退化为劝退器;
- Skill 风险条款必须带 Consumer 族("Memory 跨任务劝退"为已命名失败模式);
- 证据来源定权:独立发现的证据可扩权;Skill 在场自我诱导的确认证据可维护/反驳、不可自我扩权。

## 4. 当前证据支持什么 / 不支持什么

支持(C24–C32,截至 2026-08-24):
- 受控任务条件化正证来自 T1b/C13 的同字节、只换 Task 对照;forecasting 自然获益与 Yahoo AD 三 Consumer 族全害(12/12 宏负,锚复现逐位一致)是跨数据/协议的汇合证据,不能单独归因为 Task 因果翻转;
- AD 内条件化显形于机制与幅度:iforest 边界变形 / 监督训练证据侵蚀 / PCA 重构失真;
- canonical claim-cap:**在 Yahoo 已曝光 24 条 × 现三 Consumer × 现五程序下,无全局安全处理;存在局部 headroom(oracle ≈ +0.04,赢家不相交),但现役 Observation 与反馈无法安全收割**;
- held-in event-F1 反馈单元无安全合格者(#42h);f1_pooled +0.0059 系边缘 development 线索,未授权(#42j)。

不支持(禁止的说法):
- Yahoo 已是完美数据;AD 没有优化空间;Harness 不适合 AD;Slow 未归因好;再堆失败 Experience 即可解决。
- 最终 Judge 已就绪 ≠ AD 的 held-in 学习反馈已验证(后者待 #44a)。

数据病理注记:Yahoo A1 附 Wu & Keogh 四病 caveat(平凡可解/密度失真/错标/run-to-failure 位置偏置——本数据实测 held-out 38 事件 vs held-in 窗 14 即其显形)+ 镜像 provenance caveat。

## 5. 下一项:AD 反馈正控(#44a)

最小纵向切片:固定一个 AD Consumer → 仅向 held-in 训练底物注入一种已知污染 → clean reference / contaminated identity / contaminated exact-or-mask repair → Query 原样。先分别证明污染确实降低 delayed event-F1、repair 确实恢复效用,再查哪种早期 Support 信号(有事件区 AUPRC/事件排序、事件 recall、背景误报率、事件数)能预测该恢复。clean 只称 reference,未经读数不得预称 upper bound;注入位置与 clean reference 仅 evaluator 可见,不得成为 Agent Observation 或 Support 特征。
分流:污染未造成可读伤害 → 当前 contamination/Consumer 不可读;污染有害但 repair 不恢复 → Program 层;repair 有效但 Support 预测不了 → 反馈协议层;Support 可读而 Agent 选不到 → Observation/Scope/selection 层;Agent 找到并过独立 delayed → 才进 Experience/M1b 与 sealed 验收。若使用两个冻结污染率,必须逐率完整报告且禁止择优率授权。
定位:development 正控仪器;结论永不写入自然 Yahoo 能力声明;41 条 sealed 在 replay 验证过的管线就绪前零读取。
