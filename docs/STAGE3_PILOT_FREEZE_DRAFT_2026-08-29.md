# Stage 3 pilot 冻结稿(草案,呈 sol)

日期:2026-08-29 01:1x。地位:主线起草,依主实验设计 v1 §6(D3 裁定)展开为可冻结协议;
sol 核准后即为 Phase 1 唯一协议。与池决策(D4)解耦:本 pilot 只用已开 development 池。

## 0. 目的与主张边界

验证:Harness 能否依真实 Gain/Harm 轨迹对**决策策略面**提出并验收一次自修订
(C6,论文次级主张)。单轮,不重掷,成败均照录入论文。**不主张**:跨任务迁移、
多面演化、持续策略学习——均出关键路径。

## 1. 种子失败(已发生,不另造;2026-08-29 02:1x 依 Part 0 审计改写)

S2a r2 受益单元 4(`electricity_impulsive_outlier_04`),A5 格一手记录
(`s2a_g1_run1_r2.json` rows/15/rounds/0):池=[identity, 自提
`robust_mad_outlier_repair`, 供给 `cand_skill_s2a_forecast_supply_v0`(hampel)];
Fast select 选自提 mad;首个 Support 阳性即停探;供给卡
`not_reached_support_budget_exhausted`——**到池未到探测**。部署 mad(+3.7863),
而 A3/K0 自发现 hampel(+3.9069)。反事实 −0.1206 的真机制 = **更优的供给候选
从未被评估**(chosen-first 序 + 首正停探饿死供给)。
失败类型命名:**SUPPLY_STARVATION**(供给饥饿;原名 SUPPLY_DISPLACEMENT 作废,
更正入典 02:1x)。可编辑面不受更名影响。

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

## 4. 课程与预算(2026-08-29 02:2x 终冻,依 Part 0 审计原料)

**课程(5 单元,双产例对冲,顺序即冻结顺序)**:

| 位 | 单元 | 角色 |
| --- | --- | --- |
| 1 | `electricity_impulsive_outlier_00` | producer(电族) |
| 2 | `electricity_impulsive_outlier_02` | beneficiary(电族匹配位) |
| 3 | `traffic_impulsive_outlier_00` | producer(traffic 族) |
| 4 | `traffic_impulsive_outlier_01` | beneficiary |
| 5 | `traffic_impulsive_outlier_02` | beneficiary |

设计依据:全部取自 Part 0 审计的未入课原料清单;双产例使「供给可达 + 自主候选
竞争」的复现机会 ≥3(单元 2/4/5);若全程 Scope 未命中或无竞争结构,按 §5
`S3_SEED_UNREPRODUCED` 收口,不重掷。r2 先例:同族相邻 cell 匹配非必然
(elec_03 卡中 _04 不中 _01),故对冲而非单押。

**三臂策略绑定**:No-edit = DEFAULT 策略(= 现行为);Random-legal-edit =
以种子 **20260829** 从八参数合法域(见审计工件 proposed_params)均匀抽一条
非 DEFAULT 单参数改动,运行前落盘;LLM-edit = Slow Agent 读 §5 种子轨迹字段
后输出一组参数赋值(限合法域),提案全文落盘。三臂共用课程、预算、
memory 初始化(冻结 post-S2a 活性态 + 空课程内记忆,同 r2 协议)。

**预算(课程级硬帽)**:每臂 LLM ≤40、fit ≤120;三臂合计 LLM ≤120。
LLM-edit 臂的提案调用单列(≤2 次),不占课程预算。

**判分单元**:beneficiary 位(2/4/5)承担接受门读数;producer 位只供铸卡。

**G2 防火墙**:pilot 触碰 cell = 电 impulsive 00/02 + traffic impulsive
00/01/02,记录入工件,永久排除出 Phase 3 密封池。

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
