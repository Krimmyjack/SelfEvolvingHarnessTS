# 接手简报:HEC-1 自主推进(致 Opus 执行线,2026-09-03 夜)

你接手的是一条**已经裁定完毕、只差填数和实现**的主线。你的工作不是重新设计,是把冻结件填完、把四件接线
装好、跑通烟测,然后在授权信封内发车,并把每一步如实记账。**方法争议不由你裁**:sol 是方法裁定方,用户是
执行授权方,主线(Fable)是验收方。你在预授权信封内自主,信封外停下来写清问题等裁。

## 0. 先读(顺序,不可跳)

1. 项目 `AGENTS.md`(全文;§1 目标、§3 held-in/held-out、§4 执行权、§6 单假设、§7 反过度工程、§8 证据纪律)。
2. `docs/HEC_EVOLUTION_MAINLINE_PLAN_2026-09-02.md`(主线规划;§3 判据、§4 协议、§5 读数、§8 待裁状态、
   §10 设计诊断与 sol 裁定、§10.11 Source-v3 归因、§12 路线图)。
3. `docs/HEC1_CONTRACT_SKELETON_2026-09-03.md`(D3 骨架:`[D2]`/`[sol]`/`[user]` 标记就是你的待填清单)。
4. `docs/D4_HEC1_WIRING_SPECS_2026-09-03.md`(四件接线 + smoke 交付门)。
5. `docs/D1_ROUTING_HARM_DIAGNOSTIC_2026-09-03.md`、`docs/D2_HEC1_COURSE_SUPPLY_SCAN_2026-09-03.md`
   (若 D1/D2 已由他人完成,读其工件 `p4ab_*`、`p4ac_*`;未完成则按 `DISPATCH_PACK_D1_D2` 自己做,D2 优先)。
6. 代码:`evaluation/main_protocol_p4/run_source_line_v3.py`、`restricted_draft.py`、`scope_clause_agent.py`、
   `scope_repair_distance.py`、`scoped_serving_evaluator.py`、`main_experiment_contract_v3.py`。
7. 账本 `docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md` 最近 12 条(2026-09-02 起)。

## 1. 目标(一句话)与判据

> 画出「同一 Harness 随经历改善、同起点冻结的不改善」这条曲线,并让它经得起统计与归因追问。

四判据同时成立才主张自进化(§3):online−frozen 差随课程增长;≥1 条 Skill 修订后存活并重遇改善;
A5−A3 显示积累(仅 K0 非空时评分);冻结后优势在 held-out 仍在。**头条是曲线,不是某次 gate。**

## 2. 今晚的序(严格)

| 步 | 动作 | 门 | LLM |
| --- | --- | --- | --- |
| 1 | **D1/D2 已完成、D3 骨架 `[D2]` 已由主线填入**(02:xx)。你的步 1 = 把骨架转成 `hec1_contract.py` + JSON/MD 草案(照 `main_experiment_contract_v3.py` 模式,先不 `assert_frozen`),`[sol]` 项按 §4 默认值填、逐项标"待 sol 确认" | 草案交用户转 sol 核 | 0 |
| 2 | **并行**实现 D4 四件(W1–W4)+ 聚焦测试 + `--smoke` 七项;Source-v3 `--dry-run` 回归 | smoke 全过;`methods/ttha/*` 零改动 | 0 |
| 3 | sol 核回 → 按裁定改草案 → `assert_frozen()` → 账本记冻结 | 冻结 | 0 |
| 4 | Phase S(Source 块;A5-online 单臂;外环开;H1–H3 落账) | 需用户放行 Phase S 预算(≈90 LLM) | ≤120 |
| 5 | K0 冻结(可能为空 → 按合同缩臂,如实记 `A5_TREATMENT_EMPTY`,**不重跑**) | — | 0 |
| 6 | Phase T **Forward**(仅此一顺序) | 需用户放行 Forward 预算(≤500);跑完只做仪器健康核对 | ≤500 |
| 7 | 停。写晨报。Reverse / Interleaved 与 Phase F **不在今晚信封内** | — | 0 |

D1 已完成(`p4ab_routing_harm_diagnostic`,判词 `NO_OUTCOME_FREE_SEPARATOR`):Risk 面不开,Observation 缺口已登记
进合同 §2,你不需要为此做任何事。D2 已完成(`p4ac_hec1_course_supply`):课程数据直接取工件 `proposals`
(Phase S 13 / Phase T ≤3816 口径 26 / 三顺序序列)。另注意 D1 揪出的隐患:Source-v3 round 2856 两道 delayed 门
口径不一——W4 必须以 P4 `_gate` 为唯一激活权威并记 `gate_disagreement`(D4 规格已加验收项)。

## 2b. sol 条件授权(2026-09-03 夜,用户转递并已分发 → 视为执行放行;用户可随时叫停)

1. **立即开展全部 0-LLM 工作**:`hec1_contract.py` 草案、D4 四件、单一权威门、聚焦测试、七项 smoke。
2. **20/19 cohort 分母检查**:必须确认覆盖率(treated/served)与风险分母(harmed_fraction 的 served 数、
   `MIN_TREATED` 的对照基数)在 `_gate` / `admission_policy` / `scoped_evaluate` / 评价面计分 / D4 新模块中
   **全部动态取自当前 served 序列数**;grep 任何字面 `20` / `/ 20` / `range(20)`;若仍有硬编码 → `[200:239]`
   改用 **19/19(弃 T99)**,并在合同与账本记明。
3. **Phase S 放行条件(全部满足才发第一次 LLM 调用)**:sol 核完 12 项 `[sol]` → W4 分歧锁验收通过 → **全量回归
   无新增失败**(基线 = `_scratch/pytest_baseline_tests_tree.txt`:`tests/` 入口 46 failed / 9 errors / 747
   passed,历史预存失败不算新增;比对差集,只看新增)→ `assert_frozen()` 通过。**LLM 硬上限 120。**
4. **Forward 放行条件**:Phase S 完成且 K0 冻结(空亦冻结)→ 仪器完整性检查通过(见 §5b)→ **LLM 硬上限 500**。
5. Forward 完成后**停止并汇报**;不运行 Reverse、Interleaved 或 Phase F held-out。
6. **主线建议的第五放行条件(待用户/sol 认可)**:Phase S 发车前,由**非作者** agent(grok 4.6-xhigh)按
   `docs/HEC1_INDEPENDENT_REVIEW_CHECKLIST_2026-09-03.md` 做独立代码评审 + 对抗性测试(A–G 七组,A 组任一
   FAIL 一票否决),产出 `HEC1_REVIEW_REPORT`。理由:本项目全部错误叙事都来自 runner 接线而非模型,smoke 只测
   作者想到的地方。你(Opus)配合:在 D4 完成后立即通知用户派评审;评审期间不改被评审文件;FAIL 项修后复评。

## 2c. 延长信封:Phase S → Interleaved 仪器核对完成(**待 sol 改令 + 用户预批两个 ≤500 信封后生效**)

sol 现令为"Forward 完成后停止"。若 sol 将其改为"仪器八项**机械**通过则自动续跑"且用户预批 Reverse / Interleaved
各 ≤500,则你可连续自主推进,**硬停点 = Interleaved 仪器核对完成**。条件与纪律:

1. **两份脚本化门(列入独立评审对象,见清单 H 组)**:
   - `audit_hec1_instrument.py`:§5b 八项全部机械断言(计数、账本、集合交集、`resolved_by`、store 相等抽查),输出
     PASS/FAIL 逐项 + 证据;**任一 FAIL → 停**,不得人工改判。
   - `audit_hec1_k0_freeze.py`:K0 非空时逐卡断言——经 P4 权威门的 Support 与 delayed 记录、阈值来自工具(无
     `LLM_THRESHOLD_IGNORED` 之外的 LLM 数值)、无 replay 授予的激活、Scope 子句 ≤2、证据行齐;K0 空时断言缩臂
     配置正确。**任一 FAIL → 停**。
2. **续跑序**:Phase S → `audit_hec1_k0_freeze` PASS → K0 冻结入账 → Forward(≤500)→ `audit_hec1_instrument` PASS
   → Reverse(≤500)→ PASS → Interleaved(≤500)→ PASS → **停**,写晨报/收口报。
3. **运行形态**:每顺序以脱离终端的后台进程运行,独立 store 根与 run_label,每单元臂落检查点并写心跳文件;你的会话
   若中断,进程继续;恢复只用 `--resume`(按 (ordering, position, arm) 去重,不重复计费)。BACKEND_UNAVAILABLE
   过重试策略 → `RUN_BLOCKED_NO_VERDICT` 记录 → 等中继恢复后 `--resume`,**不重跑已完成单元**。
4. **不得**:自己跑课末读数脚本或写判词(读数脚本由非 runner 作者实现,判词由主线出、sol 确认);触碰 Phase F;
   因任何顺序的效果正负改变续跑决定或合同;在 FAIL 后"修一下再跑"——FAIL 即停、记账、等人。
5. **监控换手(建议)**:Forward 发车后,监控与 `--resume` 可交 grok 4.6-xhigh(同一份简报);你留给需要判断的修复。
6. 每顺序结束各写一条账本;三顺序齐后写 `docs/HEC1_RUN_REPORT_<date>.md`(仅仪器、生命周期计数、账本;**无曲线
   结论**)。

## 2d. v1.1 落地指令(sol 正式裁定 2026-09-03;**取代** §2 序与 §2c 的续跑前提;详见 `HEC1_V1_1_AMENDMENT_REQUEST` §3b)

**现状定性**:当前 Forward = `FORWARD_SHAKEDOWN`——可跑完收仪器数据,**中断不得 `--resume`**,无论结果不进曲线;
其仪器报告先于一切入账。Phase S-v1 保留并标 `superseded`。

**落地清单(全部完成、测试通过、非作者复核通过后才允许 commit)**:

| # | 改什么 | 落点 / 验收 |
| --- | --- | --- |
| 1 | `MIN_POSITIVE_UNITS_FOR_ADD` 2 → **1**;单次正例只生成不可部署 restricted Draft | `outer_loop.py`;测试:单正例 → Draft 出现、不 Active;replay 通过仍不 Active |
| 2 | replay 预算:**每个 online 臂独立 = 自身课程 fits 的 100%**;回放**全部可适用**历史单元;低覆盖 cell 记 `NOT_APPLICABLE`(不计违反、不计 aggregate_not_material) | `REPLAY_FITS_SHARE` 语义改为 per-arm 1.0;删除共用额度;删除任何 recency/滑窗 |
| 3 | REVISE **原地** `record_revision`(不开新壳);供给按 `may_verify`;修订 ≤2 / 验证 ≤3 逐候选 | `restricted_draft.py` / `outer_loop.py`(审查线已修,复核锁定) |
| 4 | WAITING 自动再验**仍耗一次验证次数**(维持现状,不改) | 合同文本写明理由 |
| 5 | K0 交接传**快照**(store_root / runtime_bundle_sha),缺失 fail-closed;非空 K0 时 A5 臂 retrieved 含 K0 卡、A3 不含 | 审查线已修;`audit_hec1_k0_freeze.py` **必须以独立脚本存在**并断言 K0 快照、Skill 资格、A5/A3 隔离 |
| 6 | census key = Task × Consumer × 完整 typed Program × **root Scope 谓词**;故障类型只作分层不入键;指纹只折叠别名 | `outer_loop.py` 普查 |
| 7 | 统计:合同 `STATISTICS` 删 α=.05 单侧 sign test;正式标准 = ≥2/3 顺序终点差 >0 ∧ ≥3/4 cohort >0 ∧ online harm ≤ frozen;p 值与 cohort bootstrap CI 只作描述;不画跨顺序置信带 | `hec1_contract.py` / readout |
| 8 | **+144 评价面可评性预扫**:只读 mask、不读效用;不可计分单元预先标记,学习单元保留,所有臂一致;`N_T_eff` 入合同 | 新 0-fit 步,合同字段 |
| 9 | **P4 `_gate` 唯一数值权威**;P4 过而生命周期事件未批准 → `lost_activation` 计数;**科学顺序中任何 gate disagreement → 该顺序降级** | `run_hec1.py` + `audit_hec1_instrument.py`(disagreement 为硬 FAIL) |
| 10 | **19-series 真实行为测试**锁动态分母(不能只靠扫 `/20`);`[200:239]` 20/19 切法据此确认或改 19/19 | 新测试 |
| 11 | 召回归因:`deployed_via` 加机械判"部署程序 ∈ 单元起始 Active 集" | readout / runner 记录 |
| 12 | Phase F 前置改为三者合取:**非空 K0 ∧ HEC-1 判词支持完整 A5 主张 ∧ 用户人工开封**;K0 空 → HEC-1 以 A3-online vs A3-frozen 作 Target-local 自进化组件证据收口,**不开 Phase F** | `hec1_contract.py` Phase F 门 |
| 13 | 只完成两顺序 → 判词 `INCONCLUSIVE`(不得改 2/2) | readout |
| 14 | 命名:Best-Safe-Global = offline in-budget comparator;Phase F = 冻结 Skill 库机械部署;积累 = within-dataset / cross-cohort | 合同与 readout 文案 |
| 15 | 工程:分母扫描不跳过含引号行;冻结合同收据;账本补 09:39–10:04;checkpoint `mode: live|offline` 不匹配 = RunFault | — |

**发车序(sol)**:v1.1 落地 + 测试 + **非作者复核**(grok,清单 B/C/H + 上表 1–11)→ **allowlist commit**,合同记
`code_commit`(不建哈希体系)→ `assert_frozen` → +144 预扫 → **Phase S-v1.1**(≤120)→ `audit_hec1_k0_freeze` PASS
→ 同一 commit 下 **Forward → Reverse → Interleaved**(各 ≤500;只按八项仪器自动续跑,任何 disagreement 降级该顺序)
→ **停**;课末读数与 Track A/B 判词由**非 runner 作者**做;Phase F 按第 12 条三合取。BSG prequential 与 TSFM 暂缓。

**生效前提**:用户明示——「批准 Phase S-v1.1 ≤120 次 LLM,并批准正式 Forward、Reverse、Interleaved 各 ≤500;当前
shakedown 开销单列,不授权 Phase F 开封。」未明示前,只做 0-LLM 工作。

## 2e. v1.1 增补:replay 三件不可拆分修复(sol 最终裁定,2026-09-03;与 §2d 合并,commit 前必须全部闭合)

主线核算(`FINAL_SCIENTIFIC_DESIGN_REVIEW` §15)定位:每次 screen 3 fits/cell、五步 225/候选流 > 每臂 156;同 key 反复
铸壳。sol 批准以下三件为**一个不可拆分的修复**;不加每步候选上限、不加新优先序,沿用已冻结确定性次序。

| # | 修复 | 硬约束 |
| --- | --- | --- |
| A | **按完整 census key 去重**:Task × Consumer × typed Program × root Scope | 范围 = 本课程该 key 的完整 lineage(Active / open / **已关闭**);已关闭 key 不得重开壳绕过"2 修订 / 3 验证";不同 root Scope 才是不同候选;`held` 不再只从 bank 带 `source_skill_id` 的行推 |
| B | **per-arm replay prediction cache**:键 arm × cell × face/origin × Consumer config × typed Program;内容 = raw/program 逐序列预测 + 合法性信息 | 不跨臂;不读未来 cell;不跨 Consumer;Scope 只掩码计分不重拟合;退化 context 仍按原逻辑拒绝;缓存 vs 现算逐位一致;`physical_fits` / `logical_evaluations` / `cache_hits` 分账 |
| C | **未来外环预算预留**:按保守成本为未来每步预留一次 screen(5+10+15+20+25 = 75),当前步只用扣除预留后的余额 | 保证"未来每步有候选则预算不挡第一个",不保证全筛;因预算未筛者完整记录 |

**必补测试**:同 key 连续出现只留一条 lineage;同 Program 不同 root Scope 分开;已关闭 key 不能重开;缓存与现算在
空 Scope / 全集 / 真子集 / 两不同 Scope / 退化 context 五情形逐位一致;两臂缓存隔离;合成五步课程每步有候选时 ≥1 screen;
多候选下预算截断有记录且 ≤156。

**合同同步**:P1 实质效应线 **`D_o ≥ 0.005 × 23 = 0.115`**;有效点门槛 **19/23**;P1-only 判词措辞 = **"feedback-driven
Skill-library evolution / Skill acquisition evolution"**(不得写 Scope-revision evolution、完整 A5、跨域;P1+P2 才是
within-dataset Skill-and-Scope evolution);validation-search 冻结为**必需** 0-LLM、同预算、同风险门 baseline,**不进入
Harness**;Phase S 再空 → 穷举供给诊断,**不得据此生成本轮 K0**;Phase F 三顺序全评、**宏平均主报**、不挑顺序。

**更正记录**:`MIN_POSITIVE_UNITS_FOR_ADD` 现已为 1(`outer_loop.py:72`);主线此前读的是旧字节。

**发车凭据(唯一)**:Opus 单写手整合 → 聚焦测试 + 全量回归(对基线差集)→ 非作者只读复核 → allowlist commit →
同一 commit 启 Phase S-v1.1。**此后不再追加方法设计,直接进入正式实验。**

## 3. 授权信封(用户已批 / 待批,以用户当晚明示为准)

- **已批(sol 方法 + 用户执行)**:D1 ≤400 Ridge fits、0 LLM;D2 0 LLM / 0 fit;D4 实现与 smoke(0 LLM)。
- **待用户当晚放行**:Phase S LLM(建议帽 120);Phase T Forward LLM(帽 500)。**没有明示放行不得发任何 LLM 调用**
  (连通性探测也算)。
- **今晚禁区**:Reverse / Interleaved;Phase F 密封开启;任何 held-out 读取;任何阈值(0.005 / 0.20 / 0.30 /
  MIN_TREATED 5 / 0.35)改动;新增 Risk 面、观察特征、SHA/manifest/Gate;改 `methods/ttha/*`;改 Source-v3
  协议或工件。

## 4. 你可以自己定的(默认值表;sol 若在冻结件核阅时改动,以 sol 为准)

| 项 | 默认 | 备注 |
| --- | --- | --- |
| 外环周期 k | 5 | §8-7a |
| 内环即时 Slow | 关 | §8-7b;Source-v3 不受影响 |
| `replay_fits` 帽 | ≤ 课程 fit 25% | §8-7c |
| 阈值校准 | 过预算最宽阈值,并列取粗箱 | §8-8a |
| ScopeFit-only 对照 | shadow(0 LLM) | §8-8b |
| 单元臂 LLM 帽 | 5 | 使 Forward ≤500;合同写死 |
| Phase T origin 口径 | ≤3816 保守;<22 单元则用全部 usable(`[80:120]` 仍 1896…2856) | 披露时间邻接 |
| 评价面 | o+144,只计分不回流 | 不进 bank |
| K0 空时臂集 | Static / A3-frozen / A3-online | `A3-frozen` 命名待 sol |
| K0 空时既有卡 | 严格取空 | §8-5 |
| Phase F 部署 | Fast-only、0 LLM 机械召回 | 今晚不跑 |
| Best-Safe-Global 菜单 | 冻结单算子 + `period_median_complete→outlier_*` 族 | 事后 0 LLM |

**遇到表外的歧义**:选最保守、最少改动、最可归因的一项,**写进账本"歧义与选择"段**,继续;不要停等,除非涉及
信封外动作或阈值。

## 5. 停止规则(RunFault)与不算故障的事

- 停:BACKEND_UNAVAILABLE 过重试策略(照 S2a:指数退避 5 次、90 s 窗)→ 记 `RUN_BLOCKED_NO_VERDICT`,**不写
  科学判词**,可 `--resume`;G2/oracle 墙泄漏;全局 LLM/token/时间超帽;协议/数据错误;任何 held-out 读取。
- 不停(UnitFault,当前单元 identity 弃权后继续):候选失败、校验器拒绝、LLM 输出多次不合法、cell 级 LLM 耗尽、
  `ServingContextDegenerate`。
- **不得**:因 Forward 效果正负改合同;因某单元难看重跑;因 0 存活补跑 Source;把仪器故障写成 `TREATMENT_EMPTY`
  或 `NOT_SUPPORTED`。

## 5b. 仪器完整性检查(Phase S → Forward 的放行门;Forward → 晨报的核对表;**只看仪器,不看效果正负**)

| 项 | 通过标准 |
| --- | --- |
| 完成度 | 无 `RunFault`;单元完成 = 计划数(或 `--resume` 后补齐) |
| UnitFault | 单元臂 UnitFault 率 ≤ 20%,且分类齐(候选失败 / 校验拒绝 / 输出不合法 / cell 预算耗尽 / 退化) |
| 预算 | 各账本 ≤ 帽;第 N+1 次调用在后端前被阻断且不计费;`replay_fits` ≤ 25% 课程 fit |
| 隔离 | held-out 读取 = 0;评价面 Outcome 不在任何 bank;oracle 墙未触;A5-frozen 每单元后 store 等于 K0(抽查) |
| 门 | `gate_disagreement` 全部 `resolved_by=p4_gate`;Active 集合只由权威门产生 |
| 外环 | 按 k=5 触发次数正确;每步记录候选/淘汰/Draft/Slow 调用/`replay_fits`;不授部署权 |
| 记录 | Fast 原始决定分类齐;H1–H3 字段齐;三态转移有 history;检查点可 `--resume`;cache 命中率入账 |
| 回归 | Source-v3 `--dry-run` 字段一致;全量回归对基线差集无新增失败 |

任一项不过 → 不放行下一顺序,写清项与原因,等主线/sol;**不得**用"效果好/不好"作为放行或停止理由。

## 6. 记账与晨报

- 账本 `docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md`:**只追加**,每完成一步一条,插在主线锚点
  (最新主线条目之上);格式照最近条目(标题带时间与"执行方/Opus");必带:做了什么、0/非 0 LLM 与 fits 数、
  产出工件路径、歧义与选择、未做与原因。
- 工件不覆盖:新 run_label / 新文件名;历史工件一字不改(勘误只追加)。
- 提交:只用 allowlist;不 `git add -A`;不碰另一执行线未提交文件(`AGENTS.md`、`PROJECT_STATE_*`、
  `SUCCESSOR_BRIEF_*`、`methods/ttha/*`)。
- **晨报**(单独文件 `docs/MORNING_REPORT_2026-09-04.md`,按 `AGENTS.md` §10 五问):Harness 行为改变了什么 /
  真实或可控数据上观察到了什么 / 当前最大方法不确定性 / 是否仍与目标一致 / 下一项最小纵向切片。外加:
  Forward 仪器健康表(单元完成数、UnitFault 分布、LLM/fits 账、cache 命中率)与**待 sol / 待用户**清单。
  **不在晨报里写曲线结论**——三顺序未齐前只报仪器与生命周期计数。

## 7. 本项目已经付过学费的反模式(请勿重复)

1. 用 5–7 单元的短课考需要 40 步显形的机器,然后判 REJECTED。
2. 把一次编辑的接受/拒绝压在 n=1 的当前单元上。
3. 首个故障全停、判词写在开头;仪器故障写成科学判词。
4. 把 v2 文案原样贴进 v3 工件(2136 写成 2856)。
5. "Slow 不是瓶颈"式过度陈述——正确措辞:格式表达、合法收窄、持久化不再是瓶颈;所选 Scope 能否产生稳定
   条件效应未证明。
6. 为凑样本量混入另一数据域/缺陷机制/适配器。
7. 把能被超过的基线叫 oracle。
8. 按 context 覆盖挑 held-out 考场。
9. 为一个结果加一个 Gate / SHA / 审计平台;"基础设施、测试、文档、Gate 不算方法进展"(§7)。
10. 以"改动量"当路由伤害的风险信号——伤害来自被路由到 program model,不是 context 被改(§10.11)。

## 8. 模型分层

你(Opus)做 D3 定稿与 D4 实现(难);D1/D2 类纯扫描、审计交 grok 4.6-xhigh。委派深度一层;子任务必须携带
`AGENTS.md` 方向与反过度工程约束。
