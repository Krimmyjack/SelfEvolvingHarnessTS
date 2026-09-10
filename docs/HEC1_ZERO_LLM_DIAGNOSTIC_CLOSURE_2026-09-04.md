# HEC-1 课末与 0-LLM 诊断收口（2026-09-04）

地位：HEC-1 scientific 三顺序完成后的 **EXPOSED DEVELOPMENT / POST-HOC DIAGNOSTIC** 记录。
本文件不改冻结合同、不重跑课程、不读取 Phase F、不把事后诊断写回 Harness。

## 1. HEC-1 正式读数

- 三顺序均在 commit `d690850be1cb70d1da5f7034dbb097f42ba42e36` 完成，仪器门全过；26 个计划单元中 23 个可计分。
- 冻结 readout 给出 `HEC1_EVOLUTION_NOT_SUPPORTED`：终点物质差 `D_o >= 0.115` 仅 1/3 顺序，cohort 正向仅 2/4；harm 条件三顺序均成立；P2 修订—存活—重遇链为 0。
- `A5-online - A5-frozen` 终点差：Forward `+0.211896`、Reverse `-0.043036`、Interleaved `+0.005719`。69 个逐单元配对中 62 个平局。
- A5-online 的 identity 单元为 18/26、19/26、19/26；事后统一口径的 `recalled_skill` 为 3、1、0 次；三顺序 `revisions_sum=0`。
- 本判词只否定这份 KDD with-missing × pooled Ridge × 当前 Scope/风险合同下的预注册 HEC-1 主张；合同明确禁止写成“进化普遍无效”。Phase F 不开。

主工件：`artifacts/main_protocol/hec1_readout_v11p0.{json,md}`。

## 2. 三项 0-LLM 诊断

| 诊断 | 原始结果 | 可支持的窄结论 |
| --- | --- | --- |
| Best-Safe-Global（评价面 Outcome 事后取优） | 23 个可评单元中 14 个有安全的 non-identity 程序；累计安全上界 `+5.527089`；1046 fits | 冻结程序菜单在该 development 数据上存在明显 headroom；它是不可部署的 outcome-side 上界，不是方法 baseline |
| validation-search（Support 选、+144 计分） | 26 单元中 17 个有至少一个 Support-safe 候选；16 个可评 non-identity 部署中 15 个有物质正增益，但仅 7 个仍过评价面四线；累计原始增益 `+3.131950`、harm 事件 9；899 fits | 确定性搜索能提高供给和原始收益，但不能可靠保证未来安全，不能称为风险约束下胜出 |
| 全候选 transfer audit | 34 个 Support-safe 候选逐一在 +144 重部署，仅 10 个仍安全；16 个可评机会中，所选候选稳定 7 个、换另一个候选可救 1 个、8 个没有任何 Support-safe 候选能保持安全；68 fits | “选错候选”不是主因；Support→evaluation 的效应/安全迁移不稳是更深的约束 |

三份正式诊断共 2013 次 Ridge fit、0 LLM；另做两次单元级预检。held-out 读取 0、Episode bank 读写 0、Skill 创建 0、阈值改动 0。

工件：

- `artifacts/main_protocol/hec1_best_safe_global_v11p0.{json,md}`
- `artifacts/main_protocol/hec1_validation_search_v11p0.{json,md}`
- `artifacts/main_protocol/hec1_validation_transfer_v11p0.{json,md}`

## 3. First-fault 定位

### 3.1 Fast 供给不足是真问题

按 program ID 的宽松上界，把 validation-search 的 Support-safe 程序集与 A5-online 的探针/部署对齐：

| 顺序 | 有 Support-safe 程序的单元 | A5-online 覆盖至少一个 | 全部漏掉 |
| --- | ---: | ---: | ---: |
| Forward | 17 | 8 | 9 |
| Reverse | 17 | 7 | 10 |
| Interleaved | 17 | 7 | 10 |

覆盖计数摘要：`Forward 8 / Reverse 7 / Interleaved 7`。

这是对 A5 有利的宽松口径（不要求 Scope 字节完全相同），仍显示超过一半机会未被 Fast 覆盖。没有观察到“已经探到上述
program-ID 匹配的安全候选却没有部署”的单元，因此 Support 当窗的首要损失在候选供给，而不是 Runtime 把已知安全候选再次挡掉。

### 3.2 但只修 Fast 仍不够

- 34 个 Support-safe 候选只有 10 个在下一评价窗仍安全。
- 24 个未来失败候选中，22 个触发 `single_series_harm`、16 个触发 `harmed_fraction`，仅 4 个触发 `aggregate`（失败线可重叠）。
- 7 个可评单元在 Support 上一个安全候选都没有，但评价面 Best-Safe-Global 在 7/7 都能找到安全 non-identity；反向又有 8 个单元在 Support 有安全候选、评价面却无一保持安全。

因此提高 Fast 召回率会增加 treatment，但不会自动形成稳定 Skill；更深问题是自然时序跨窗口下，当前 Scope 描述的“模式像不像”
不能稳定预测“处理后未来是否受益/安全”。既有 D5 又表明 pooled Ridge 的模型路由承担严重伤害的主要份额，故当前结构约束是
`Scope 条件化 × pooled 模型路由 × 单序列尾部门` 的合取，而非一个坏算子或一次 LLM 随机性。

## 4. Claim ceiling

可以说：

1. 在该 development setting 中，程序空间存在安全 outcome-side headroom。
2. Fast 候选覆盖不足是可复算的供给缺口。
3. 即使使用确定性搜索，Support-safe 候选的未来安全保留率仍低；选择排序只解释 1/16 个机会。
4. HEC-1 的 online/frozen 大量平局主要来自可执行差异稀少，而不是已经充分部署的知识被证明无效。

不可以说：

- Best-Safe-Global 是可部署方法；
- validation-search 在风险约束下优于 A5；
- 当前结果证明算子无效、LLM 无效、自进化普遍无效或跨域迁移失败；
- 这些事后诊断改变冻结 HEC-1 判词或允许打开 Phase F。

## 5. 下一步

不重跑 HEC-1，不开 Phase F。下一项单假设实验仍是 HEC-2 per-channel：保持数据、课程、程序、Scope 与风险线不变，只解除
pooled Ridge 的跨序列模型路由；同时预注册读取：(a) 尾部伤害，(b) 聚合收益是否随路由一并下降，(c) Support-safe 候选的
未来安全保留率能否高于 HEC-1 的 `10/34`，(d) 冻结 24 候选 validation-search 与 Fast 的供给差距。

注意：本轮 validation-search 的“冻结菜单前 24 项”采用已有 Best-Safe-Global 菜单顺序，但具体截断是在 HEC-1 结果之后实现，
因此本轮只作诊断。若作为论文正式 baseline，必须在下一份合同中先冻结候选顺序再运行。
