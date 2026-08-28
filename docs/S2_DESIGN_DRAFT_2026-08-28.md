# S2 设计稿:机制任务可移植性 + 跨任务条件化守卫(2026-08-28,主线定稿呈 sol)

事实基础:`artifacts/functional/e2/s2_forecast_asset_inventory.json/.md`(提交 `cc8e045`;下文"盘点 N"指其七项)。上游裁定:sol 路线图(收口→v1 冻结→S2/S3→主实验)+ v1 冻结声明("Stage 2 只读 Skill 层")。

## 0. 一句话主张

在预测任务上用 v1 冻结机制**重挣**一遍"经验初始化→低权入场→双门检验→反馈修订"闭环(机制可移植性),同时用分类卡在预测 cell 上的**全面沉默**证明跨任务条件化(task_kind 轴负控);两者合起来即 Stage 2 的承重主张,并为主实验把预测线宿主铺好。

## 1. 盘点事实约束(设计必须服从)

1. **无现成宿主**:S1 四臂 runner 三常数写死 classification(盘点 1/承重段,`run_e2_s1_curriculum_four_arms.py:275-277`、`_scope_v1_admits :1683`、`skill_revision.py:78`);G1 agentic 能跑 forecast 但不承载四臂/修订/五轴。→ S2 须付适配成本。
2. **现役 forecast cell 反馈容量全部不达标**(12+8 / 12+4 vs 新门"对半后每面 ≥20"):须从 `monash:traffic_hourly`(862 序列)/ UCI electricity(370)等重切(盘点 2)。
3. **+31.7% 正账系 v1 前机制年代**(recipe Guidance 卡,盘点 3):只能作历史参照,不得与 S2 读数并写;S2 的一切效应须由 v1 机制在新门下重挣。
4. **无已提交可检索的 forecast Episode 库**(盘点 4):课程内自产从零开始,与分类线 S1 同起点——**冷发现率未知是本设计最大风险**。
5. 注入族用现役 forecast `impulsive_outlier`(分类 impulse 的孪生,盘点 5);Consumer 本阶段单 ridge×sMASE(盘点 6;多 Consumer 留主实验轴);**#31 AD 卡无 task_kind 轴,不能当守卫**(盘点 7)→ 跨任务守卫由分类 supply 卡独任。

## 2. 实验设计(三门串行)

### S2-G0:宿主与考场(0 LLM)

- **适配器**:S1 四臂 runner 三常数参数化;`_scope_v1_admits` task 轴放行 forecast;`skill_revision` 特征提取按 task_kind 分发到 forecast 特征集——**语义零改,仅分发**(见 §3 冻结修订案)。分类侧 146 项回归必须原封全绿(零语义漂移自证)。
- **考场重切**:从 monash traffic_hourly 或 UCI electricity 机械重切 6-8 个 cell:TRAIN ≥40 行(对半每面 ≥20,材料线 ≤0.05)、官方式 held-out、`impulsive_outlier` 注入、`fit_only_artifact` 条件对。
- **双层 oracle 密封 + 资格门**(S1a 纪律):产例单元 held-in LEARNABLE 且对半余量 ≥2×(M-1 门);受益 ≥1 强(≥2×)+1 贴线弱(诚实分层);≥1 identity 场 + ≥1 族外守卫场;oracle 工件隔离。
- **门**:`S2_HOST_READY`(适配测试绿 ∧ cell 资格全过)否则停呈,不改数据凑格。

### S2-G1:机制重挣(live,课程形状照抄 SA-1 已证形状)

- 四臂 Static / A3-reset / K0-fixed / A5-adaptive;课程 = 产例 → 边界产卡(阶梯 v2,1 强正例)→ 强受益(五轴 Scope 转化)→ 贴线冲突(R2 收窄)→ **再遇位**(机制探针,读数单列)→ 族外守卫。
- **主判(ITT)**:课程内自产 forecast 供给卡产生材料级 regret 改善 ∧ harm 0;修订环再遇位行为差。判词三分:`S2_MECHANISM_PORTABLE` / `S2_PARTIAL`(细分:卡成无转化 / 转化无修订素材)/ `TREATMENT_EMPTY`。
- **冷发现风险预置**(分类线教训直接继承):产例选余量最强 cell;`TREATMENT_EMPTY` 允许**一次**预冻结采样重复;两掷仍空即系统性结论呈机制层,禁第三掷、禁改课。

### S2-G2:跨任务条件化守卫(随 G1 免费)

- A5 store 预装**分类 supply 卡 v0**(SA-1 同源):在全部 forecast 单元上 retrieval / scope_match / supply 三面必须全零(task_kind 轴)。三面任一非零 = first fault(冻结轴失效,机制级警报,即停)。
- 这是 capstone"族外沉默"的跨任务版,把条件化主张从"族"抬到"任务"。

### S2-G3(缓议,不入本书)

算子中立程序性迁移(cls Episode → 几何偏好类指导 → forecast 提案分布移动)属原纲领的深水区,机制选项(机械通道 vs prompt 结构)另立设计呈 sol;不与 G0-G2 混跑。

## 3. 所需冻结修订案(唯一一处,呈 sol 核)

v1 冻结清单若含 `skill_revision.py` 等文件,G0 适配器须一份**范围严格受限的修订案**:仅许 (a) runner 三常数参数化 (b) `_scope_v1_admits` task 轴放行 (c) 特征提取 task_kind 分发——**Skill/Memory 语义、阶梯价格、Scope 轴、写回规则零改动**;修订落地后重出冻结清单 v1.1(新 sha),分类回归全绿为生效条件。

## 4. 预注册预测(可证伪)

P1 适配零语义漂移(分类 146 项原封全绿);P2 G2 三面全零;P3 若产例命中,强受益场经供给转化(余量门控已证的延伸,首次跨任务检验);P4 贴线冲突场触发恰一次收窄且再遇位行为差(修订确定性跨任务复现);P5 harm 全零。产例冷发现率不预测(未知,如实报)。

## 5. 预算与止损

G0:0 LLM,墙钟 ≤4h。G1+G2:LLM ≤120 / fit ≤300 / 墙钟 ≤6h;采样重复一次预授权(仅 `TREATMENT_EMPTY` 或 SIGNAL 复合措辞用)。止损:`S2_HOST_READY` 不过即停;两掷空即停;G2 非零即停。执行路由:G0 适配与重切(规格明确)= grok;G1 若涉现场归因判断 = 视复杂度定,默认 grok 跑冻结课程。

## 6. 明确不做

不动多 Consumer(主实验轴);不下载新数据(重切仓内 monash/UCI);不碰密封件;不做 G3;不与 +31.7% 旧账并写;不为凑资格改门。

## 7. 对主实验的输出

合格 forecast cell 家族(密封考宿主)+ forecast 新鲜度普查要求(哪些 monash 集从未被任何线读过,主实验选密封靶用)+ 双 Consumer oracle 扫描选项(分类侧,0 LLM,可与 G0 并行)。
