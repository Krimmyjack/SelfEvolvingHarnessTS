# CLS-replay — 受控分类 dynamic-binding 能力于 HEAD 重放

**判定:`REPLAY_REPRODUCED`**
日期 2026-08-25 | HEAD `84aabd1`(父 `098ec40`)| 分支 `g1/general-proposal-guidance`
角色:`INSTRUMENT / MECHANISM` 重放。不是自然缺陷证据,不是 fresh,不产新 Capability。

---

## Part 0 — #45-Frep-b 交付工件提交

提交 `84aabd1`「checkpoint: #45-Frep-b deliverable artifacts + C37 ledger section」,3 文件 / +1852 行:

- `artifacts/functional/e2/t6_45_frep_b_symmetric_deploy.json`
- `artifacts/functional/e2/t6_45_frep_b_symmetric_deploy.md`
- `docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md`(C37 未提交节)

allowlist 纪律:逐路径 `git add`,全程未用 `git add -A`。密钥扫描三文件全过(`sk-*` / `api_key=` / `token=` / `Bearer` / `AKIA` / `PRIVATE KEY` / `password=` 零命中)。
另线在飞四文件 `AGENTS.md` / `README.md` / `PROJECT_STATE_AND_DATA_MAP_2026-08-23.md` / `SUCCESSOR_BRIEF_2026-08-22.md` 保持 modified 且未暂存,零触碰。

---

## Part A — 考古与绑定

### 产出 runner

能力工件 `controlled_classification_dynamic_binding_capability.json` 自报 `supported_report =
source_promoted_binding_capability_transfer_report.json`,该报告的产出者是**专属 runner**:

- `evaluation/functional/run_e2_promoted_binding_capability_transfer.py`
 — plan/evaluate 双入口(`main()` :683-707,sealed-data boundary 用法);
 — 冻结 roster `TARGET_DATASETS` :34-41,六条 UCR;顺序与唯一性在 evaluate :438-443 再校验;
 — 冻结成功门 build_plan :182-192,evaluate :609-620 判定;verdict 子句 :663-667。

上游准入是 W55:`evaluation/functional/run_e2_program_binding_harness_update.py`,由
`_admit_source_report` :77-110 校验四项(verdict `CONTROLLED_PROGRAM_BINDING_HARNESS_UPDATE_PASS`、
H1 全 target 精确绑定、6 条正 query gain、0 条 A5-vs-A3 负例)。转移 runner 复用 W55 的六个函数
(:113-120 / :206-217 / :220-231 / :287-300 / :310-313),没有另起第二套框架。

检索腿在 `run_e2_cross_series_curation.py`:`--phase multiskill-fast-path`(罐装,
`compile_multiskill_llm_fast_path` :5278)与 `--phase multiskill-live-fast-path`(实时,
`run_live_multiskill_llm_fast_path` :5629-5771),菜单是两条能力、五个部署可见 context。

### 运行配置

- **LLM 预算(原书)**:转移 runner 0;罐装 fast-path 0;live fast-path 1(agicto / gpt-5.6-luna)。
- **seed**:全链无 RNG,不需要 seed。注入是固定模板(`SPIKE_AMPLITUDE=16.0`,
 `run_e2_task_context_label_evidence_witness.py` :38 / :88-101),fit/support 按标签有序切分(:73-86),
 Consumer 是固定 alpha 的 `RidgeClassifier`。这是**合成/确定性后端**,故本轮按原样零 LLM 跑主链。
- **受控注入材料**:`condition=fit_only_artifact` —— 脉冲只打进 fit 行,Support 保持干净
 (`_condition_inputs` :223-240)。这正是"fit-only artifact 可执行 / stable event 禁忌"两分的物理来源。
- **判定条款**:六项绑定/范围计数 + A4 零射宏增益 > 0 且正例 ≥ 4 + A5 宏 AUC > A3 + 0 负例 + 事件伤害 ≤ 0。

### 输入材料在盘性

六个 zip 全部在 `data/ucr_task_context/`,mtime 均为 2026-08-03,自采集起未动。
**工件内没有记录任何 SHA**(见书外发现 X3),故无从做"对照工件内记录"。本轮实测并记录:

| 材料 | SHA-256 前 16 |
|---|---|
| BirdChicken.zip | `fada1c7119d797b3` |
| HouseTwenty.zip | `22695841c256be5c` |
| ToeSegmentation1.zip | `bfdb133a828e69fa` |
| PhalangesOutlinesCorrect.zip | `41938528ef09228c` |
| SonyAIBORobotSurface2.zip | `0e76ea43816e5f2e` |
| GunPointAgeSpan.zip | `662f1c4d91a27170` |

更强的内容核验来自重放本身:重生成的 plan 把六条数据的 `official_train_count` /
`series_length` / `fit_count` / `support_count` / observer 节点索引 / 残差强度**逐字节复现**。

### 原判定与关键回执(对照基线)

`CONTROLLED_PROMOTED_BINDING_CAPABILITY_TRANSFER_PASS`,`frozen_gate_pass = true`。
H1 精确绑定 6/6、H0 失配 6/6、事件范围不变 6/6、`A5_policy_event_harm_max = 0.0`、A5-vs-A3 负例 0、
A3/A4/A5 宏 AUC = 0.63071 / 0.72972 / 0.71396、`A5−A3 = +0.08326`、A4 零射宏增益 +0.17631(6/6 为正)、
18 次 Consumer 重训、6 次 TEST 打开。

**入口在 HEAD 可运行**,不触 `REPLAY_ENTRY_BROKEN`,未做任何单点修。

---

## Part B — HEAD 重放

隔离:全部输出写 `artifacts/functional/e2/_cls_replay_20260825/`,零覆盖既有正式工件。
计分 evaluate 只跑一次,零重掷。

### 四腿读数

| 腿 | LLM | 原 canonical SHA-256 | 今 canonical SHA-256 | 差异 |
|---|---|---|---|---|
| 1a plan 重生成 | 0 | `3d97a141…1bf5499d` | `3d97a141…1bf5499d` | **0** |
| 1b evaluate(对冻结 plan) | 0 | `6b2256e9…23720368` | `6b2256e9…23720368` | **0** |
| 2a 罐装 multiskill fast-path | 0 | `4b6ae9d4…c095826f` | `4b6ae9d4…c095826f` | **0** |
| 2b live-LLM multiskill fast-path | 1 | `1c59394a…70812114` | `0e293a91…5e028914` | 2(仅 provider token 计数) |

三条确定性腿在 canonical JSON 层面完全一致。判定复现:
`CONTROLLED_PROMOTED_BINDING_CAPABILITY_TRANSFER_PASS` /
`MULTISKILL_LLM_FAST_PATH_BEHAVIOR_PASS` / `MULTISKILL_LIVE_LLM_FAST_PATH_BEHAVIOR_PASS`。

### 生命周期逐段对照

| 段 | 原 | 今 | 状态 |
|---|---|---|---|
| 检索 | Source 准入四项过;能力编译;罐装与 live planner 均在 `classification_fit_only_artifact` 选中 `controlled_classification_local_event_dynamic_binding_v1` | 同 | REPRODUCED |
| 绑定 | H1 动态绑定精确 6/6(执行位 == observer 节点);H0 固定绑定失配 6/6,且有越界执行位 | 逐节点索引相同 | REPRODUCED |
| 执行 | center-excluded 局部中位数在每条 fit 行上实质修复全部四节点(H1,6/6);H0 打在 {12,36,156,180} 或 {12,36} | runner 内四条断言(:370-386)全过,位点相同 | REPRODUCED |
| 验证/守卫 | `stable_task_event` 6/6 判 `CONTRAINDICATED_ABSTAIN` → `ABSTAIN_KEEP_INCUMBENT`,`policy_event_harm = 0.0` | 同,max 0.0 | REPRODUCED |
| delayed | A5 在 B1/B2 花 Target Support 延迟确认,顺序 H1→H0;5/6 正向,ToeSegmentation1 精确零增益并正确回滚 incumbent | 方向与数值全同 | REPRODUCED |

**`event_erasure_guard` 的实现路径**(书面要求单列):
`contracts/task.py` 的 `classification_local_event_task_quality_contract_v1()`(:517-548)把
`event_erasure` 列为 harm → 经 TaskContext 序列化进 plan 的 `decision_input_text`
(`generic_harms=event_erasure,…`)→ 由 cross-cohort witness 判据落地
(`support_to_fit_strength_ratio ≈ 1.0` 且 `direction_alignment = 1.0` ⇒ 局部标签证据跨 cohort 重复
⇒ `CONTRAINDICATED_ABSTAIN`)。六条数据上守卫全部触发,伤害恒 0,与原书一致。

### 逐数据集回执

| 数据集 | observer 节点 | H1 执行 | H1 精确 | H0 越界 | 守卫 | H1 query gain | A5 选择路径 | delayed support gain (B0:H1 / B1:H0) | A5−A3 AUC |
|---|---|---|---|---|---|---|---|---|---|
| BirdChicken | 41,102,409,470 | 同 | ✓ | 12,36,156,180 | ABSTAIN/0.0 | +0.2500 | H1,H1,H1 | +0.3333 / 0.0 | +0.1250 |
| HouseTwenty | 160,400,1599,1839 | 同 | ✓ | 12,36,156,180 | ABSTAIN/0.0 | +0.0084 | H1,H1,H1 | +0.0833 / 0.0 | +0.0042 |
| ToeSegmentation1 | 22,55,221,254 | 同 | ✓ | 12,36,156,180 | ABSTAIN/0.0 | +0.0526 | H1,**incumbent**,**incumbent** | 0.0 / 0.0 | +0.0175 |
| PhalangesOutlinesCorrect | 6,16,63,73 | 同 | ✓ | 12,36 | ABSTAIN/0.0 | +0.1725 | H1,H1,H1 | +0.2074 / −0.0167 | +0.0862 |
| SonyAIBORobotSurface2 | 5,13,51,59 | 同 | ✓ | 12,36 | ABSTAIN/0.0 | +0.3022 | H1,H1,**H0** | +0.2500 / +0.2500 | +0.1305 |
| GunPointAgeSpan | 12,30,119,137 | 同 | ✓ | 36 | ABSTAIN/0.0 | +0.2722 | H1,H1,H1 | +0.4250 / 0.0 | +0.1361 |

两个非平凡回执也逐字复现,值得单点:

- **ToeSegmentation1 回滚**:H1 的 Support 增益精确为 0,严格改进规则不成立,A5 退回 incumbent,
 由此放弃了真实存在的 +0.0526 query 增益。这是"保守规则代价"的正样本。
- **SonyAIBORobotSurface2 平局倒向 H0**:两候选 Support 精确相等,冻结的
 `equal candidate Support accuracy prefers H0_fixed_binding` 让 B2 选了失配绑定,
 损失 0.178 query 精度。这是 Support/query 背离的第二任务素材。

### LLM 非确定性处理(仅 2b 腿)

- **协议层复现失败:无。** schema 被接受,`contract_violations = []`,`forbidden_plan_fields = []`,
 `unpromoted_or_invented_count = 0`,五个 context 的行为与确定性编译 5/5 匹配,gate 四项全过。
- **采样波动:有,但不承重。** plan 差异 7 处 = 5 处 `reason_codes` 自由文本(条数 6→3 / 6→3 / 4→2 / 2→1,
 及 `scale_invalid`→`scale_context_invalid`)+ 2 处 token 计数。
 **五个 context 的 `decision`、`capability_id`、`workflow_steps` 全部逐字相同**,编译出的可执行行为不变。
- 归类:协议层 REPRODUCED;波动被限制在非承重字段,未触达任何决策。

### 历史工件保护

`run_live_multiskill_llm_fast_path` 把生成的 plan 写死到正式路径(:5747-5752),`--output` 只改 report。
本轮先做字节备份、跑完后字节还原并核 SHA:原 `CF6CF2DD…C1F50F7A`,还原后 `CF6CF2DD…C1F50F7A`,**一致**。
详见书外发现 X1。

---

## TaskContext 携带情况(如实报)

**携带:是,但走的是 runner 级 legacy 合法路径,不是 T6 operational inlet。**

- 构造点:`run_e2_task_risk_action_credit_transfer._helpers()` :318-360 调用
 `SelfEvolvingHarnessTS.contracts.task.classification_task_context_v1(...)`,
 task_spec = `classification_task_spec_v1(downstream_model_class="ridge-raw-plus-difference-v1")`,
 quality_contract = `classification_local_event_task_quality_contract_v1()`。
- 类型是真的:`TaskContext` dataclass,`schema_version = "task-context/1"`,
 contract_id = `classification-local-event-quality-v1`。不是字符串占位。
- 消费点:`run_e2_program_binding_harness_update` :276
 `helpers["decision_text"](task_context, artifact_witness)` —— TaskContext 被序列化进 plan 的
 `decision_input_text` 的 `TASK_CONTEXT` 段(task_type / metric / quality_objective / preserve /
 generic_harms)。
- **未走**:`evaluation/functional/task_episode_harness/e1.py` 的 T6 `task_context` 入口。
 该分类线是独立 plan/evaluate 实验 runner,全链不 import `task_episode_harness`。
- 兼容性观察:#42i 往 harm 词表加了 `normal_boundary_shrinkage` / `false_alarm_amplification`
 两项(`contracts/task.py` :64-72),但分类局部事件合同的 harms 元组未受影响,
 所以该线的 TaskContext 穿过 #42i/#42k/#42l 后语义不变。
- **本轮未做任何接线**,遵书面"不顺手接线"。

---

## 预算记账

| 项 | 本轮 | 原书 / 帽 |
|---|---|---|
| LLM 调用 | 1 | 原 live 腿 1;本书总帽 48 |
| LLM token | prompt 1248 / completion 478 | 原 5628 / 454(见 X5) |
| Consumer 重训 | 18 | 18 |
| TEST split 打开 | 6 | 6(该数据原书已 EXPOSED,本轮属 development 重放) |
| 数据读取 | 仅 `data/ucr_task_context/` 六条 TRAIN + 六条 TEST | — |
| 下载 | 0 | 禁 |
| 子 Agent | 0 | 零 spawn |
| 计分跑 | 1 | 禁重掷 |

---

## 义务自报

- 仓根 `AGENTS.md` 开工第一步完整读毕。
- conda `project`:`D:\Anaconda_envs\envs\project\python.exe`,每次计分运行前打印 `(Get-Command python).Source`。
- **解释器事故(如实报)**:最初一次 `plan` 调用跑在 base Anaconda
 (`D:\New_software\Anaconda\python.exe`)下 —— 先前的 `conda activate project` 落在一个未延续的 shell 会话里。
 从打印出的解释器路径当场发现,该输出已删除作废,`plan` 在 `project` 环境下重跑;
 所有对照与全部计分读数只用 project 环境产物。影响:无 —— 作废那次不产计分结果,也未打开 TEST split。
- 零 spawn;零下载。
- 禁读数据全部零读取:Yahoo 全部 / NOAA 2025 / beyond_17520 / NAB / SMD。本轮只碰 `data/ucr_task_context/`。
- allowlist + 密钥扫描:Part 0 三文件逐路径 add 并扫描通过。
- 另线在飞四文件零触碰,保持 modified 未暂存,与开工时字节一致。
- `methods/` 零改动。
- 正式工件覆盖数 0(live plan 经备份/还原,SHA 核符)。
- 交付工件**不 commit**。临时脚本 `_clsreplay_cmp.py` / `_clsreplay_table.py` 与
 `artifacts/functional/e2/_cls_replay_20260825/` 均不入库。

---

## 书外发现(只报不修)

**X1 — 历史证据可被静默覆盖(证据保全隐患)。**
`run_live_multiskill_llm_fast_path`(:5747-5752)把 plan 写死到正式路径,无输出覆写参数;
`--output` 只管 report。任何人再跑一次 `--phase multiskill-live-fast-path` 就会静默冲掉历史 live-LLM
plan 证据,与 `AGENTS.md` §7「现有历史 SHA、Runner 和工件保留」相抵。本轮以备份/还原规避,代码未动。
最小修建议(未实施,超出本书范围):加一个可选 `--plan-output` 透传到 `plan_output`。

**X2 — 该封口能力零回归保护。**
`tests/` 下没有任何测试引用 `controlled_classification`、`run_e2_promoted_binding_capability_transfer`
或 `run_e2_program_binding_harness_update`。它能穿过 #42i/#42k/#42l 手术完好,靠的是整链确定性
+ 没人碰过,不是靠测试守着。

**X3 — 材料指纹缺口。**
能力工件与冻结 plan 都没记六个 UCR 归档的任何哈希,"对照工件内记录"无从做起。本轮实测哈希已记进交付
JSON 供后续重放引用;**未新建任何哈希基础设施**,遵 §7 反过度工程。

**X4 — A5 未优于 A4(承重口径提醒)。**
`A5_minus_A4_macro_adapt_auc = −0.01576`:在这条受控分类切片上,零反馈的 source-only 臂 A4
宏 adapt AUC **高于**完整 A5 臂,原因正是上面两个回执(ToeSegmentation1 回滚、SonyAIBO 平局倒向 H0)。
冻结门只测 `A5 > A3`,从不测 `A5 > A4`。这是原书就有的性质,不是重放偏离;但它与 #45-Frep-b 在预测线上
暴露的 Support/query 背离同源,故**不应把第二任务读成"A5 占优"**。

**X5 — provider token 计数不可作确定性信号。**
同一份逐字节相同的 prompt,原书报 `prompt_tokens = 5628`,今报 `1248`。属 provider 侧计费/缓存口径变化,
非 Harness 变化。

---

## 结论

`REPLAY_REPRODUCED`。已封口的受控分类 dynamic-binding 能力在当前 HEAD 精确复现:
转移 runner 的 plan 与 report canonical 层面逐字节相同,罐装多技能 fast-path 报告逐字节相同,
live-LLM fast-path 五个决策全同、同门通过,波动只落在自由文本 reason codes 上。
**第二 Task 的正向生命周期——检索 → 绑定 → 执行 → 守卫验证 → delayed——在 #42i/#42k/#42l
多任务手术之后机械上完好。**

本判定不建立:任何自然缺陷证据;任何 fresh 读数;以及"分类任务已有 operational Harness 入口"
——该线至今仍走自己的实验 runner,而非 T6 inlet。
