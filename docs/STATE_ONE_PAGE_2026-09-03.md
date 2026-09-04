# 当前状态一页（始建 2026-09-03；最新更新置顶；所有执行线先读此页；账本只作归档）

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
