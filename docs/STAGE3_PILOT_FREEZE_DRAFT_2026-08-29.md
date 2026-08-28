# Stage 3 pilot 冻结稿(草案,呈 sol)

日期:2026-08-29 01:1x。地位:主线起草,依主实验设计 v1 §6(D3 裁定)展开为可冻结协议;
sol 核准后即为 Phase 1 唯一协议。与池决策(D4)解耦:本 pilot 只用已开 development 池。

## 0. 目的与主张边界

验证:Harness 能否依真实 Gain/Harm 轨迹对**决策策略面**提出并验收一次自修订
(C6,论文次级主张)。单轮,不重掷,成败均照录入论文。**不主张**:跨任务迁移、
多面演化、持续策略学习——均出关键路径。

## 1. 种子失败(已发生,不另造)

S2a r2 受益单元 4(`electricity_impulsive_outlier_04`):供给卡 `s2a_forecast_supply_v0`
入池并被采,挤掉更优自发现 `robust_mad_outlier_repair`,反事实 −0.1206
(`artifacts/functional/e2/s2a_g1_run1_r2.json`;裁定入典 2026-08-28 23:3x)。
失败类型命名:**SUPPLY_DISPLACEMENT**(供给挤占)。

## 2. 可编辑 Surface(功能定义 + 接线审计前置)

Surface = **probe 位在「历史供给候选」与「自主探索候选」间的分配与排序策略**
(含:供给候选是否强制占用 probe 位、占用几个、与自主候选的并列评估顺序)。

- **Part 0 接线审计(执行前置,照 SA-0 先例)**:执行方先在 runner 中定位该策略的
  实际落点(候选池装配与 probe 预算分配代码),将其参数化为一组具名策略参数
  (=可编辑面),产出审计工件(file:line 全引证)呈核后方可继续。
- **G3 硬边界(不可编辑,站规)**:双门、容量门、harm 阈、越权守卫、隔离守卫、
  阶梯 v2 门槛、Scope 匹配语义。编辑只触探索/分配,永不触安全与授权。
- 合法编辑空间(Random 臂取样域)= 固定 probe 总预算下,分配权重/顺序/强制位的
  重排;禁止增减总预算,禁止触碰 G3 清单。

## 3. 三臂协议

| 臂 | 内容 |
| --- | --- |
| No-edit | 冻结 v1 策略原样 |
| Random-legal-edit | 合法编辑空间内均匀随机一次编辑(种子预注册) |
| LLM-edit | Slow Agent 读种子失败轨迹(SUPPLY_DISPLACEMENT 工件),提出一次编辑;提案全文入工件 |

三臂共用同一预冻结课程与预算;编辑在课程开始前一次性生效,课程中不再改
(单轮纪律;在线持续策略学习不在本 pilot 范围)。

## 4. 课程与预算(待冻结空位,Part 0 后填)

- 课程:从**已开 development 池**预冻结组成,优先复用已切但未入 S2a 课程的 cell
  (electricity 0-299 余格、traffic recut 余格);须含 ≥2 个「供给可达且存在自主
  候选竞争」的单元(种子失败的结构性复现位),组成与顺序在运行前冻结并入工件。
- 预算:课程级 LLM 调用与 fit 硬帽,数值随课程冻结(参照 S2a 缩形课实测
  LLM 60/fit 88 标定)。
- G2 防火墙:本 pilot 触碰的全部域记录入工件,永久排除出 Phase 3 密封池。

## 5. 接受门(预注册,不可事后放宽)

LLM-edit 臂相对 No-edit 臂,在后续单元上:

1. 累计 held-out regret **非劣** 且 课程级成本(LLM 调用+probe 数)**非劣**;
2. 其中至少一项**严格改善**;
3. harm = 0;越权 = 0;隔离面全零;
4. Random-legal-edit 臂用于归因对照:若 Random 臂同等改善,则不得归因于
   「依轨迹提议」,判词降为 EDIT_NOT_ATTRIBUTABLE。

判词谱系(预注册):`S3_EDIT_ACCEPTED` / `S3_EDIT_REJECTED`(1-3 未过)/
`EDIT_NOT_ATTRIBUTABLE`(4 触发)/ `S3_SEED_UNREPRODUCED`(课程中种子结构未复现,
本轮作废且不重掷,照录)。

## 6. 收尾

无论判词:冻结 **Harness v2**(= v1 + 至多一条被接受的编辑;拒绝则 v2 ≡ v1),
v2 为主实验唯一被试。pilot 全部工件、提案原文、随机种子、触碰域清单入库。
