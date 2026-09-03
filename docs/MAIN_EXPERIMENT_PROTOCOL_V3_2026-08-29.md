# 论文主实验协议 v3(正向收益定稿稿,呈 sol 裁定)

日期:2026-08-29。地位:**协议合并稿 v3,呈 sol 裁定;未裁定前不动码、不开密封件**。
血统:v0(git 93cb6c1)→ v1/v1.1(`MAIN_EXPERIMENT_DESIGN_SKELETON_2026-08-28.md`)
→ v2 分析稿(`MAIN_EXPERIMENT_PROTOCOL_ANALYSIS_2026-08-29.md`,非裁定)→ **本稿 v3**。

## v2→v3 变更记录(2026-08-29,用户裁定)

1. **主终点转正为正向收益**。用户裁定(本日):"我个人还是追求正向的收益",任务书原文
   = "自适应提升下游模型**性能**与**训练效率**"。v2 分析稿的「效用非劣 + 成本优效
   co-primary」框架**降级为易档期望与附录级内部效度防线**,不再是主终点措辞。
   主终点 = **held-out 下游 ΔPerf 正向优效**(对比对与档位见 §2/§4)。
2. **外部基线入协议**(B0–B4 + 传统统一清洗四条 + AegisTS 适配 spike),回应
   「四臂全内部」的最高拒稿风险(v2 判断 2)。
3. **单元 = 序列块 × 时间原点块**入协议,解 `|g/SE|≤2.27` 的功效结构问题(v2 判断 3.3)。
4. **菜单扩容门控版**:现役 5 算子 → 19 算子/5 族 × {全局, per-series} + identity
   ≈ 39 项;≤2 步组合列为 stretch,以 dev pilot 可行性为门。理由:正向收益的核心机制
   主张是「经验缩小搜索空间」,5 算子在 20 行 Support 面上 5 次 fit 即穷举,
   该主张在旧菜单下**结构性不可测**(v2 判断 2 末段)。
5. **Stage 3 pilot 已执行并判毕**(`S3_EDIT_REJECTED`,8/29 12:55):LLM-edit 未过
   增益非劣门(no_edit +3.3301 vs llm_edit +1.5334),harm 0、越权 0、G2 0。
   **Harness v2 ≡ v1(DEFAULT) 已冻结**(sha 清单 `artifacts/functional/e2/s3_harness_v2_freeze.json`),
   为主实验唯一被试;C6 按预注册作有界负面次级主张收账。exploration_policy 八参数
   参数化留树 fail-closed,不进主实验被试面。
6. G1 已过(KDD 每课事件代理 15.00,保守只计缺口 5.00 ≥ 拟门 2);KDD/Solar 双下载
   已冻结落库(md5 匹配,Solar 隔离令生效)。

## 1. 设计目的

### 1.1 中心问题(v3 措辞)

> 时序数据的质量标准随 Task、Consumer 与局部 Pattern 变化,固定清洗规则与直接 LLM
> 判断都缺乏可靠反馈。同一台 Harness 能否把各任务自己的历史经验,经审计、带 Scope、
> 可被 Target 否决地积累起来,在后续 Target 上**正向提升下游模型性能与训练效率**,
> 并以真实 Gain/Harm 反馈持续修正自己。

### 1.2 与任务书逐句对照

| 任务书句子 | 协议落点 |
| --- | --- |
| 任务与模式感知的 Readiness 优化(趋势/周期/缺失/异常 → 针对性可执行策略) | C1 + Q1(consumer 轴整机闭环)+ 菜单/Observation(§3.4) |
| 反馈驱动的 Harness 自适应进化(分析 Gain/Harm,迭代 Skill/Memory) | C2/C5a/C5b(主承重,§2 内部四臂) |
| …迭代 Instruction 与决策策略 | C6(Stage 3 已判,次级负主张照录;Instruction 面列 future work) |
| 自适应提升下游模型性能与训练效率 | **主终点 ΔPerf(§4.1)+ 效率读数(§4.2)** |

### 1.3 主张表(承接 v1.1,正向化重排)

| # | 主张 | 证据锚点(dev) | 主实验角色 |
| --- | --- | --- | --- |
| C1 | 就绪度 Task/Consumer/Pattern 条件化 | 强:pooled/per-channel 换案 3/3、翻转 2/3;ridge −0.0133 vs kNN −0.1200;AD 12/12 宏效用负 | 动机章 + 主实验 consumer 轴整机闭环(Q1) |
| C2 | Skill 生命周期(形成/供给/修订/收窄/再用) | dev 复证(SA-1 r1/r2,L1) | 分类腿副表 + 预测腿课程内自然复现 |
| C3 | 经验复用条件化且安全 | 密封已证(Epilepsy2:族外卡全沉默、有害提案 −0.2750/−0.0500 全拦、无 headroom 三臂 identity、harm 0) | 安全列主读数;F1 易档与 AD 腿正考 |
| C4 | 同一机制多任务可运行 + 隔离 | 已证缩形(S2a;G2 20/20 零泄漏) | 课程内复现,不单独立项 |
| C5a | 固定经验 > 无经验 | **正**:cls regret gap +0.6860;L1 held-out +0.2127;forecast frep +0.276、首正成本 −43.9%(69 vs 123) | **主终点对比对之一**(A5/K0 vs A3/Static) |
| C5b | 在线修订 > 固定经验 | 未证:修订环 regret 恰 +0.0000(只省 2 probe/2 挨拒);forecast r2 反事实 −0.1206 | A5-online vs K0-fixed,**如实报**;期望主战场在成本侧 |
| C5c | 经验价值在 fresh 新域泛化 | 未证 | F1/F2 密封阶段(§5.3) |
| C6 | 决策策略自修订 | **已判负**(S3_EDIT_REJECTED) | 次级负主张,照录 |

### 1.4 正向收益的机制理由(不是许愿,写进论文逻辑链)

1. 6/6 cell(3 cohort × 2 Consumer)存在延迟非负方案,且**冠军程序随 cohort/Consumer
   改变** → 固定清洗器不可能处处当冠军;ΔPerf 的正号来自「条件化选择 + 该弃权时弃权」,
   与 AegisTS 主表里 Clean4TSDB 在 Handwriting ΔPerf=−0.0626 同构(任务无关清洗会伤)。
2. 冷发现不可靠(CLS 实测 29%/位):A3 会**漏掉**正确 Workflow;经验供给把「发现」
   变「确认」,已知 headroom(如 GPOvY +0.184)从「碰运气」变「按 Scope 兑换」。
   → A5 vs A3 的正向效用差在中/难档有实测先验(+0.6860 / +0.2127 / +0.276)。
3. 训练效率半句的落点:达首个安全有效 Skill 的 consumer fits 与 time-to-threshold
   (dev 锚点 −31.7% / −43.9%)。

**难度三档方向性预测(预注册,禁只取任一档)**:易档 = 非劣 + 零害 + 正确弃权
(经验不得造成负迁移);中档 = 成本节省 + 效用为正;难档 = 效用与成本优势扩大。

## 2. 臂与基线

### 2.1 内部消融臂(同菜单、同 Consumer、同 Target held-in 反馈预算)

| 臂 | 知识起点 | 单元间写回 | 归因用途 |
| --- | --- | --- | --- |
| Static | 无(恒 identity) | 无 | 适应本身的贡献;正确弃权基线 |
| A3-reset | 公共 h0 | 每单元清零 | Target 本地适应边际 |
| K0-fixed | h0 + bootstrap 三卡 + 种子卡 | 禁写回 | 固定先验价值(C5a 对照) |
| A5-online | 同 K0 | Slow 整合 + R1–R4 修订 | 完整系统 |

### 2.2 外部基线(论文级必须)

| 组 | 基线 | 内容 | 成本 |
| --- | --- | --- | --- |
| 地板 | B0 identity | 不处理(= ΔPerf 分母,与 Static 互为校验) | 0 |
| 传统统一清洗 | B1a 系列 | IQR/3σ 剔除 + 线性插值;Hampel 滤波;winsorize;SCREEN(速度约束,自实现约 200 行) | 纯计算 |
| 单步上界 | B1b best-single-oracle | 每单元事后最优单算子(密封 oracle 机制现成) | 纯计算 |
| **最强 no-agent 对照** | **B2 matched-budget search** | 等 fit 预算随机搜索,报 **1×/3×/5×** | 纯计算 |
| 分离 agentic/self-evolving | B3 one-shot LLM planner | 一次 LLM 看 inspection 出 Workflow,不 probe 不修订 | 1 LLM/单元 |
| 分离检索/Skill 抽象 | B4 fingerprint-kNN memory | TimeClaw 式确定性指纹检索复用历史 Workflow,无 Scope/双门 | 纯计算 |
| agentic SOTA | AegisTS 本体适配 | 多变量→单变量池 + ridge/sMASE Consumer;**先做 1 天可行性 spike,成或不成都记录** | 中–贵 |

预注册对比对:**primary = A5−A3(触发富中/难档)与 A5−Static(全课程)**;
supporting = A5−K0、A3−Static、A5−B2@各档、A5−B4、A5−B1a,Holm 校正。
Learn2Clean / BoostClean / CleanAgent / DiffPrep / EDITOR 不实装(表格形态或可微
下游限制),related work 说明理由。

### 2.3 消融(主实验后,同一冻结课程 replay;每条对应一条机制主张)

无归因(随机修订)/ 无验收门(去双门+harm 否决)/ 只蒸馏成功经验 / Consumer-blind
(G5,归因只到整机)/ 无 Scope / 无 delayed。邻域对位见 v2 §2。

## 3. 数据集、单元构造与数据量

### 3.1 池与角色(冻结现状)

| 池 | 规模 | 角色 | 新鲜度 | 档位 |
| --- | --- | --- | --- | --- |
| KDD Cup 2018 含缺失原版(Zenodo 4656719,md5 已核) | 270 序列 ×9504–10920;503,712 缺失点;缺失率 min/med/p90/max = 0.55%/11.72%/34.04%/97.67%;270/270 有缺失 | 触发富池(中/难档 + 修订触发场),**development 级如实披露** | dev | 中/难 |
| traffic leftover(TSL 列 480–861) | 382 列 ×17544;0 缺失;周跳变 0–3 条/cell | **F1**:同族新 Target、Outcome 未见 | F1 | 易 + 正确弃权 |
| electricity leftover | 21 列 | spare-only | dev | — |
| Solar 10 Minutes(Zenodo 4656144,md5 已核) | 137 序列 ×52560,0 缺失 | **F2 密封 capstone**,隔离令生效(完整性核验外零分析) | F2 | C5c 终验 |
| UCR 40(GunPoint 族/PowerCons 等) | — | 分类生命周期副表 | dev | — |
| Yahoo S5 A1 | 41 条 SEALED | AD 安全终考(四臂正确弃权读数) | sealed | 安全 |

### 3.2 单元构造(功效修复的核心)

单元 = **(序列块 × 时间原点块)**,块划分预注册、互不重叠;cell 几何沿用冻结常量
(`run_e2_s2a_forecast_oracle.py`:45-66:CELL_WIDTH 60 = Support 20 + delayed 20 +
held-out 20;最小长度 1848;ORIGIN_HELDIN 1104 / HELDOUT 1800;PERIOD 24;
材料线 max(0.005, 1/n_half);HARM_BAR 0.005)。

| 池 | 互不重叠 1848 窗 | 单元构成 | 单元数 |
| --- | ---: | --- | ---: |
| KDD(4 满员 cell ×60 序列) | ~5 | 4 cell × 2 原点块 | **8**(按单元缺失率分中/难) |
| traffic F1(6 cell) | 9 | 6 cell × 3 原点块 | **18**(易) |
| Solar(capstone) | ~28 | 2 cell × 9 原点块 | **18**(F2,开考时机械套用 dev 冻结阈值分档) |

**预测主课程 = 26 单元**(KDD 8 中/难 + traffic F1 18 易);全系统含分类副表 6 单元
与 AD 安全考 ≥34 单元。功效:n=26 配对在 80% power 下检出 d≈0.55;难档子集 n=8
只承诺检出大效应——主 superiority 对比放全课程/触发富档两级报,不只在难档。
推断按**序列块 cluster-robust**(paired bootstrap CI + Wilcoxon signed-rank)。

### 3.3 数据量与预算估算

- 运行量:主课程 26 单元 × 4 臂 × 2 LLM 重复跑(T=0,「两跑同向」标准)
  ≈ 208 单元运行(Static 近零成本);B2 三档 × 3 seeds ≈ 26×(16+48+80)×3
  ≈ 11,200 fits(纯计算 ~2.1h,0.685s/fit);B0/B1a/B1b/B4 确定性纯计算。
- LLM:适应臂 ~8 调用/单元/臂(扩菜单后 pilot 标定)→ 全程 ~1,200–1,500 调用,
  课程级帽 1,500;fit 帽 3,000;墙钟按 dev pilot 标定后冻结。
- 种子:随机臂(B2、消融随机修订)≥3;LLM 臂 T=0 + 全课程 ×2 重复。
- Backbone:主结果单 backbone;**最短课程(KDD 8 单元)在第二 backbone 复现**
  (预算不足时按 G4 裁剪顺序:消融 > 档数 > 域数,永不裁臂)。

### 3.4 菜单(扩容门控版)

主实验菜单 = **19 算子/5 族(impute 7 / denoise 5 / outlier 4 / structural 2 / align 1)
× {全局, per-series} + identity ≈ 39 项**,注册表契约现成(`operators/registry.py`,
allowed_tasks/destructive/targeting_mode 等逐算子带契约);shape-changing 与依赖
hard_fail 项照旧排除。**非穷举门**:B2@5× fits 须 < 菜单规模(39)不成立即停,
扩 per-series 变体或升级为 stretch 的 ≤2 步组合。现役 5 算子菜单保留为 dev 兼容轨
(全部既有证据可比),不进主实验。
副作用对冲:扩菜单拉低冷发现率(dev 29%/位)是**设计意图**(发现变贵,经验才有
付酬空间);「A3 被做弱」质疑由 B2@1×/3×/5× 曲线回应——「本 Harness ≈ N× 搜索
预算」本身就是论文主图之一。

## 4. 指标

### 4.1 主读数(口径按 `docs/READOUT_GLOSSARY_2026-08-19.md`)

**Gating(任一不过 → 该臂该单元判负,不看效用)**:harm=0(material-harm:
support_gain < −0.005;held-out worst-series gain < −0.005;分类 worst-class recall
Δ < −0.05);越权 0;G2 跨任务泄漏 0;Scope 审计(单调收窄,sha 版本链)。

**Primary(正向收益)**:
1. **held-out ΔPerf**:macro sMASE gain vs identity(预测)/ accuracy + 逐类 recall
   (分类副表)/ 事件 F1(AD 安全考只报弃权正确性);方向为正,材料线
   max(0.005, 1/n)。预注册 superiority 对比:A5−Static(全课程)、A5−A3(触发富档)。
2. **held-out regret**(vs B1b 密封 oracle):跨 cell 可比,作主图横轴读数。

**Co-primary(训练效率,任务书明载)**:
3. **达首个安全有效 Skill 的 consumer fits**(A5 vs A3,paired);
4. time-to-threshold、fit 墙钟;LLM 调用与 `real_support_probe_count` 分项
   (`charged_probe_cost` 绝不当 probe 数)。预算记账单位 = 课程级。

### 4.2 Secondary(机制与差异化列)

5. **正确弃权率 / 错误晋升率**(无 headroom 与 Scope 不匹配单元;安全差异化列,
   AegisTS 无此列);
6. 供给转化漏斗(入池→probe→Support→delayed→部署);
7. 修订账本(R1/R2/R3 计数、Scope 收窄轨迹)与 **Skill 库规模轨迹**(RewardHarness
   式:收益来自剪枝还是扩张);
8. AdaptAUC(单元序→效用曲线下面积);
9. **修复保真度**(precision/recall/F1/RRA,**只在受控注入诊断单元报,不进目标函数**);
10. **预测式归因命中率**(每次 Skill 铸造/修订附 `predicted_affected`,下轮回填,
    报 precision/recall vs 随机基线;零代码纯协议)。

**禁任意合成总分**(D5 铁律)。

### 4.3 统计方案

ITT(进课程全计,TREATMENT_EMPTY 记 A5 负);机械退出按 glossary paired-comparable
规则预注册剔除;paired bootstrap CI(按序列块 cluster)+ Wilcoxon;primary 两条
预注册,其余 Holm;报逐单元 **win/tie/loss 计数与配对差值 CDF**,不只报均值;
报告区分新反馈/缓存重放/重复观测。

## 5. 评测方式

### 5.1 协议序

```
[已完成] Stage 3 pilot(S3_EDIT_REJECTED)→ 冻结 Harness v2 ≡ v1(DEFAULT)
[已完成] G1 受测性前门(KDD 15.00/课过拟门 2);双下载冻结落库;Solar 隔离令
Phase 1  dev 建设:菜单扩容 + B0–B4 基线 + 单元块构造 → KDD dev pilot 标定
        (预算、非穷举门、块间相关、触发率复核)→ 全绿后
        **预注册书落盘 + hash**(roster/split/单元划分/菜单/consumer/预算/指标/
        判词谱系/统计方案/裁剪顺序),此后禁改
Phase 2  主实验:held-in 多轮适应(r1..rR)→ freeze → held-out Fast-only
        → 外部 evaluator 一次性开分(F1 outcome 此刻首开)
Phase 3  消融六条(同一冻结课程 replay)+ 第二 backbone 最短课程复现
Phase 4  F2 Solar 密封 capstone(一次性,开后不改不重跑;A3/A5 同反馈预算)
        + AD Yahoo 41 条密封安全考(四臂正确弃权读数)
Phase 5  论文整理
```

### 5.2 关键纪律

- 驱动接受的面(held-in Support/delayed)与最终计分的面(held-out / F1 / F2 密封)
  **完全不相交**,held-out 结果不回流(RewardHarness 教训成文:62.5% 是在被
  rollback 规则优化了 77 轮的那 40 条 val 上测的)。
- held-out 运行期禁 open_delayed / Slow / Skill 更新 / 看结果重试(正典 §3)。
- A5 只多冻结的累积知识,不多看 Target Outcome(同反馈预算)。

### 5.3 新鲜度三级披露(论文方法论卖点之一)

| 层 | 池 | 承担主张 |
| --- | --- | --- |
| dev(如实标注) | KDD 触发富池 | 正向效用主表(受控基准 + 自然缺陷,非注入) |
| F1(同族新 Target,Outcome 未见) | traffic leftover 18 单元 | A5 vs A3 同族泛化 + 易档无负迁移/正确弃权 |
| F2(新族,Outcome 未见,隔离) | Solar 18 单元 | C5c capstone,一次性 |

### 5.4 预注册判词谱系(正向版)

| 判词 | 条件 | 论文写法 |
| --- | --- | --- |
| **PRIMARY_POSITIVE** | primary superiority 过 ∧ harm=0 ∧ 效率非劣 | 主张完全成立:性能与效率双提升,零伤害 |
| EFFICIENCY_ONLY | 效用非劣但未优效 ∧ 成本优效 | 任务书效率半句成立,性能半句退为非劣 |
| CONDITIONED_SAFETY | 效用全档打平 ∧ harm=0 ∧ 正确弃权率高 | 条件化 + 安全主张(附录级收账) |
| NEGATIVE | harm 事件或主对比显著为负 | 负结果,机制证据关闭具体 family |
| CAPSTONE_POSITIVE / NEUTRAL / NEGATIVE | Solar 终验三态 | NEUTRAL = 正确弃权,有效结局(CLS capstone 先例) |

判词谱系即全部结局的收账位——边界与条件进附录做内部效度防御,**不作论文主章**。

## 6. 风险与缺口(诚实记账)

1. **C5b 是最薄弱承重点**:dev 证据一致指向「A5 赢成本、不赢效用」(SA-1 +0.0000;
   S2a −0.1206;S3 EDIT_REJECTED)。v3 的对策不是换指标,是换考场结构:扩菜单使
   发现变贵(§3.4)+ 触发富池(KDD G1 已过)+ 三档混编——让「经验缩小搜索」的
   机制主张处于可测且有利于兑现的几何下。若仍 EFFICIENCY_ONLY,那是真结论,照录。
2. **Solar 可能 defect-poor**(0 缺失,强昼夜结构)→ capstone 可能落 NEUTRAL;
   预注册接受 NEUTRAL 为有效结局;开考前禁缺陷普查,分档只能开考时机械套用
   dev 冻结阈值。
3. **时间轴扩单元引入块间相关**:dev pilot 先测块间相关,推断 cluster-robust,
   块划分预注册。
4. **中档天然稀缺**:由 KDD 缺失率自然分层承担,不注入(正典 §8:受控注入不替代
   自然能力证据;注入档只承担修复保真度诊断)。
5. **B2 若在扩菜单后仍打平/胜出 A5**,这是真结论,照录——意味着本任务族上
   「结构化经验」不优于「等预算搜索」,主张相应收窄。
6. **AegisTS 适配可能失败**(多变量→单变量形态):spike 限时 1 天,不成则在论文
   诚实说明适配障碍,外部基线退为 B1a 四条 + B2 三档。
7. **AD 腿只承担安全读数**(#43 M0-C 已关闭该组合正效应,12/12 宏效用负);
   分类腿 sealed 族内正迁移 POOL_EXHAUSTED(资源约束,非方法负结果),以 dev
   生命周期副表 + Epilepsy2 密封安全证据入论文。

## 7. 邻域对照与论文表骨架

### 7.1 对照表(related work 底稿,承接 v2 §6)

| 维度 | AegisTS(2605.04902) | TimeClaw(2606.05404) | RewardHarness | SkillAdaptor | Evo-Memory(2511.20857) | 本工作 |
| --- | --- | --- | --- | --- | --- | --- |
| 对象 | 时序清洗 RL 智能体 | 时序 agent harness | 奖励 harness 进化 | 技能库 | agent 记忆基准 | 时序 Data Readiness Harness |
| 记忆/经验 | 无跨域积累 | 原始轨迹 + 指纹 kNN | 库子集 | embedding + rerank | 评测对象 | **经审计五轴 Scope Skill,Target 可否决** |
| 接受准则 | RL 回报(含内在质量 0.5 权重) | 无(评测型) | val 二值 | 确定性重跑 | — | **双门下游因果效用 + harm 否决 + 阶梯定价** |
| 外部基线 | EDITOR/Clean4TSDB 等 | benchmark 榜 | GPT-5/4o | 三 benchmark | 多记忆架构 | B0–B4 + 传统清洗四条 |
| 安全读数 | 无 | 无 | 无 | 部分 | 无 | **harm gating + 正确弃权率(差异化列)** |
| 新鲜度纪律 | 无 | 无 | 无独立 test(教训) | 回归一票否决 | 无 | **三级新鲜度披露 + 一次性开卷** |

定位句:Evo-Memory 评记忆架构、TimeClaw 检索原始轨迹、AegisTS 单域 RL 清洗;
本工作评的是**受治理经验在流式 Target 课程上的下游因果效用与伤害**,
K0-fixed vs A5-online 即「memory reuse vs evolution」轴在本领域的实例化。
自进化 Agent 综述(2507.21046)坐标:what=Skill/Memory 面、when=跨 Target 课程、
how=受治理写回(LLM 不批准自己)、where=时序数据准备。

### 7.2 论文表骨架(照 AegisTS 主表结构改写)

- **Table 1 数据集**:列 = #序列/长度/频率/自然缺陷画像/Task/Consumer/新鲜度层;
- **Table 2 主表**:行 = B0 / B1a 四条 / B2@1×3×5× / B3 / B4 / (AegisTS) / Static /
  A3 / K0 / **A5**;列 = (dataset × consumer)配置 × {**Downstream**: Perf↑, ΔPerf↑ 主
  ‖ **Safety**: harm 事件、worst-series、正确弃权率 ‖ **Cost**: fits、LLM、墙钟};
  Upstream 修复保真列只在注入诊断档报,表注写明不对称理由(不把内在质量当目标);
- **Table 3 消融**(§2.3 六条);**Table 4 泛化**(F1 同族 + Solar F2 + 分类族外沉默);
- **Table 5 成本**(含 Brute-force 时间上界——「策略引导 ≈ N× 便宜」的出处);
- **主图**:B2@1×/3×/5× 效用-预算曲线 vs A5/A3 水平线(「本 Harness ≈ N× 搜索预算」);
  难度三档 × ΔPerf 分组条形图;Skill 库规模与 Scope 收窄轨迹。

## 8. 待裁定清单(呈 sol / 用户,逐条)

1. 主终点正向化(§1/§4.1):primary = ΔPerf 正向优效 + 训练效率 co-primary,
   非劣框架降为易档期望与判词谱系降级位;
2. 菜单扩容门控版(§3.4):39 项 + 非穷举门 + dev 兼容轨;
3. 外部基线集(§2.2):B0–B4 + 传统四条 + AegisTS spike(限时 1 天);
4. 单元块构造与功效口径(§3.2):主课程 26 单元、cluster-robust、难档只承诺大效应;
5. 第二 backbone 最短课程复现(§3.3);
6. 判词谱系(§5.4)与新鲜度三级披露表(§5.3)。

裁定通过前:不动码、不开任何 F1/F2 outcome、不启动 Phase 1 之外的建设。
