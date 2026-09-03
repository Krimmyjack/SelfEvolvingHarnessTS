# HEC-1 最终高密度审查与继任正典（Fable，2026-09-03）

角色：主线设计 + 独立审查。0 LLM。所有代码事实以 2026-09-03 10:xx–12:xx 工作树为准（未提交）。
标记：**CODE FACT** 代码/工件直接证明；**EVIDENCE** 真实读数；**INFERENCE** 机制推断；**PROPOSAL** 后续建议。
未找到 `cursor_data_readiness_harness_project.md`（工作区深度 3 内不存在，CODE FACT），本审查不引用它。

---

## A. 执行摘要（≤20 行）

1. HEC-1 **同一 Harness、同一 `run_online_round`、同一权威门**跑四臂与 Phase S；离线 0-LLM 端到端可完成、可 resume、可审计（CODE FACT：`run_hec1.run_course`/`run_unit_arm`；`hec1_course_e2e6_review_fixes.json` 8/8）。
2. 本次对抗审查抓到 **五处会污染主结论的接线缺陷**，全部已修并有回归测试（424/424，含新增 13 项）：
   ① frozen 臂 Draft ledger 跨单元泄漏（压低 online−frozen）；② **K0 交接只传 id 不传快照，非空 K0 时 A5 臂静默从 h0 起跑**（伪造 A5≡A3）；③ 外环 REVISE 开新壳而非原地修订（计数逐壳、旧壳继续供给）；④ replay 把「谓词解析不到覆盖底线」的 cell 判为 `aggregate_not_material`（结构性淘汰一切收窄候选）；⑤ 两条 online 臂共用一个 replay 额度（A5 先跑先占，A5−A3 混入臂序）。
3. **未修、需 sol 裁**：replay 额度 25% 与「逐 cell 线性成本」相乘，使外环在第 2–3 步后必然 `REPLAY_FITS_BUDGET_SPENT`——「课程后半无 Draft」将是仪器伪影（数字见 B/E）。
4. 统计预注册**自相矛盾**：单位=cohort（n=4）却写单侧 sign test α=.05；4/4 同向 p=.0625。HEC-1 只能是**描述性 development mechanism curve**；唯一不改口径的方案见 E。
5. Phase F「Fast-only 0 LLM 机械召回」不是训练期的 Fast Agent；它是**冻结 Skill 库的机械部署**。论文必须如此命名；Agent 的贡献限定在 held-in 形成 Skill。
6. Phase S/T 同为 KDD 不同序列块：最多证明 **跨 cohort、跨时间**积累；「cross-domain」需 HEC-2 的第二数据集 + Solar F2。
7. Best-Safe-Global 现实现**在评价面 Outcome 上事后选优**（CODE FACT：`audit_hec1_best_safe_global.evaluate_unit` 用 `face_origin=+144` 读数选 max）→ 它是 **offline in-budget comparator**，不是 deployable baseline；建议加 prequential 版（在 +48 选、+144 计分，约同样 fits）。
8. Skill 层级最终定义（G）：General（程序性、h0、HEC-1 冻结）≠ Specific（声明式 program×scope×evidence）；二者**不互相晋升**；Specific→Source-derived 是角色变化，→Shared Capability 需 ≥2 域证据。
9. 最可能出正结果的路径：**ADD/召回/探针位释放**机制（A3-online 复用已 Active 卡、省搜索、少踩坑），不是 Scope 修订机制；最可能失败：REVISE 无存活（H1 新进入者 + D1 无分离量），曲线平在 0 附近。
10. Phase S 发车前真正阻塞 ≤3（见末节）：replay 额度裁定；本审查改动的 commit + 一次复核；`audit_hec1_k0_freeze.py` 缺失（Forward 前，非 Phase S 前）。

---

## B. 代码 / 仪器阻塞表

| 级 | 项 | 状态 | 证据 |
|---|---|---|---|
| P0 | frozen 臂 Draft ledger 未重置 | **已修** `Arm.begin_unit`；审计 `frozen_reset` 加断言；测试 | `hec1_course_e2e6.json` A3-frozen pos 2–5 `resupplied=['resupplied_draft_1']`；修后 `*_frozen_ledger_fix` 8/8 |
| P0 | K0 收据无 `store_root/runtime_bundle_sha`，`run_course` 非空 K0 静默回退 h0 | **已修**：`phase_s_k0` 写快照指针；`run_course` fail-closed `K0_SNAPSHOT_UNRESOLVED/MISMATCH`；审计加「frozen 臂每单元起始库 == K0」 | 离线 Phase S 6 单元 → `hec1_k0_k0_smoke_s6.json`（非空）→ Phase T 2 单元四臂，A5 臂 `retrieved` 含 K0 卡，A3 不含；`hec1_instrument_k0_smoke_t2b` 8/8 |
| P0 | 外环 REVISE 用 `open_restricted` 开新壳 | **已修**：原地 `record_revision`，`drafts_revised` 记录；REVISE 需上次修订后有新验证；HEC-1 供给改按 `may_verify`（`_VerifiableLedgerView`，v3 不动）；runner `restrict(revisions=0)` | `outer_loop.consolidate` REVISE 分支；`test_a_revise_candidate_revises_the_existing_draft_in_place` |
| P0 | replay 把不可适用 cell 判失败 | **已修**：`outer_loop._applicable` / `NOT_APPLICABLE`，需 ≥1 可适用 cell | `test_a_replay_cell_below_the_coverage_floor_is_neither_pass_nor_fail` |
| P0 | 两 online 臂共用 replay 剩余额度 | **已修**：额度按 online 臂均分，各臂独立记账 | `replay_fit_allowance.spent_by_arm` |
| **P0 待裁** | replay 额度 25% × 逐 cell 成本 | **未修** | 空 K0：总额 0.25×3×2×26×2=78；每步成本 3×5k：15/30/45/60/75 → 第 3 步起全部 `REPLAY_FITS_BUDGET_SPENT`。满 K0：117/臂 58 → 第 2 步后即枯竭 |
| P1 | `audit_hec1_k0_freeze.py` 不存在（简报 §2c 要求） | 未做 | Forward 放行链缺一道机械门；最小版 ≈80 行 |
| P1 | 激活 = P4 门 ∧ online_loop `approved`（`activate_approved` L1687-1691） | 未改，已知 | P4 通过而 online_loop 无事件时不激活；e2e 中为良性（卡已 Active）；建议审计计 `lost_activation` |
| P1 | `deployed_via` 只按候选 id 判 recall | 未改 | 2856 单元重提已 Active 程序记为 `searched_this_unit` → 三分账 (a) 低估召回；建议加「部署程序 ∈ 单元起始 Active 集」机械判 |
| P1 | WAITING 覆盖失败是否耗验证次数 | 现耗 | 3 个无模式窗口即归档 `PATTERN_NOT_REENCOUNTERED`；与「课末归档」语义二选一，需 sol 一句 |
| P2 | `_hardcoded_denominator_scan` 跳过含引号的行（L972） | 未改 | 动态分母测试已覆盖 |
| P2 | 冻结合同收据 `hec1_contract.json/.md` 未生成（只有 `.draft.*`） | 运行 `python -m evaluation.main_protocol_p4.hec1_contract` 一次 | RATIFICATION 已 `sol_confirmed=True` |
| P2 | 工件覆盖 | **已修**：runner 与 audit 拒绝覆盖，需新 label；`hec1_k0_<label>.json` | erratum 见 `hec1_instrument_e2e6_erratum.*` |
| P2 | 账本缺 09:39–10:04 条目（`run_course` 实现、e2e6、audit） | 请该执行线补记 | 晨报 §7-3 仍写「未实现」 |

---

## C. 对 sol 裁定：接受 / 修改 / 反对

| sol 裁定 | 我的立场 | 理由 |
|---|---|---|
| 1 检查点 commit（allowlist，Fable 审完测绿后） | **接受** | 424/424；allowlist 见 M |
| 2 Fable 一次性 D4 评审，不建平台 | **接受**，已做；B 组新增两项已落审计脚本 | 抓到 5 处；平台 0 |
| 3 13 项确认；Phase S 完整 13 单元；不按效果停 | **接受**，但**追加一项待裁**：replay 25% 份额在当前成本模型下使外环后半死亡，这是 8-7c 默认值本身的问题，不是执行问题 | 数字见 B |
| 4 双轨；Track B 只作预注册失败机制分析 | **接受**；补一句：Track B 的判词词表必须现在冻结（`NOT_SUPPORTED` 的 first-fault 三选一 + H1–H3 一致性），不得事后命名 | 防「重新包装」 |
| 5 e2e6 追加式更正工件 | **已做** `hec1_instrument_e2e6_erratum.{json,md}`；原字节明写「不可恢复」 | 并加了覆盖拒绝 |
| Phase F：SUPPORTED 才开封 / INCONCLUSIVE 补顺序 / NOT_SUPPORTED 封存 | **接受**，一处补充：SUPPORTED 但 P3 不可评分（K0 空）**仍开封**；开封前先出 0-LLM 覆盖分层表 | 见 F |
| Slow 只出 feature/direction，Runtime 校准 | **接受**，附限定：论文须报 Slow-vs-ScopeFit 一致率，若 ≥80% 一致，Scope 修订不得记为 LLM 贡献 | `best_stump` 即为此设 |
| 普查键 Task×Consumer×完整 typed Program | **接受**；补：键还应含 **Scope 根谓词**（同程序不同 initializer 谓词是不同候选）；不含故障类型（那是行为，不是身份） | 见 Q6 |
| Opus 自主至 Interleaved 读数前 | **接受**，前提 = B 表 P0 全清 + `audit_hec1_k0_freeze` 落地 | |
| **最不同意**：统计方案照原文（cohort + sign test α=.05） | **反对** | n=4 不可能过 .05；见 E 唯一方案 |

---

## D. HEC-1 最终协议伪代码与信息流

```text
for ordering in {forward, reverse, interleaved}:            # 三条独立 store/run_label
  arms = Static, A5-frozen(K0), A5-online(K0), A3-online(h0) | 空 K0: Static, A3-frozen(h0), A3-online(h0)
  for position, unit(cohort block, origin o) in ordering:
    ctx = deployment-visible features of served series (data < o)          # 20 或 19 条
    for arm in arms:
      arm.begin_unit()          # frozen: 重建 store+method 自起始快照，丢弃 Draft ledger；online: 带 store
      record.snapshot_skill_ids_at_start
      retrieval: Active Skill 的 Scope 匹配 → requires_target_support 供给；verifiable Drafts → resupplied
      Fast (≤5 LLM/cell, ≤2 提案, 程序 ≤2 步) → Support-A 探针 @o (Consumer fit; Outcome@o 读入决策) 
      → bounded_risk 准入 → winner (program, serving_scope) | identity
      delayed @o+48：P4 权威门四线 (coverage≥5, agg≥.005, hf≤.20, msh≤.30)
         pass ∧ write_back ∧ online_loop approved → Active(store)      # 唯一激活路径
         fail ∧ scope → Draft(by_scope 或 restrict revisions=0).record_verification → WAITING/REVISABLE/FLAGGED
      H1/H2/H3 落账（机械）
      evaluation @o+144：只计分；不进 bank / prompt / replay             # 曲线读数面
      write_back(online only)：bank += Support-A probe rows（不含 delayed，不含 evaluation）
    every k=5 units, for online arms:
      census(bank by task_consumer_key × program signature; 指纹折叠别名)
      → ADD(≥2 POSITIVE 单元且无卡) / REVISE(REVISABLE Draft, 有新验证) / NARROW(Active 卡 ≥2 逆向) / REVOKE(建议)
      → Slow(feature, direction) → tool.calibrate(最宽可行冻结分箱边界) [+ best_stump shadow]
      → narrowing preflight(root 计总子句 ≤2)
      → replay screen on this arm's processed cells @their origins：NOT_APPLICABLE(treated<5) 跳过；任一可适用 cell 违反即淘汰；≥1 可适用
      → restricted Draft（ADD/NARROW 开新 Draft；REVISE 原地修订）；replay 不授执行权
  course end：open Drafts 按 reason 归档；freeze terminal store per arm
readout (0 LLM, 非 runner 作者)：curve / harm / Best-Safe-Global advantage / 三分账 / 生命周期 / H1–H3 / Slow-vs-ScopeFit / cost
Phase F (人批)：三顺序 × 全部臂末态 → held-out 5 origin Fast-only 机械召回 → 全部输出落盘 → 覆盖分层表 → 一次开 Outcome
```

Outcome 回流表：Support@o → 回流（本单元选择）；delayed@o+48 → 回流（激活/三态/H 读数；**不进 bank**）；evaluation@o+144 → **永不回流**；replay 重读已处理单元的 @o 面 → 合法 held-in 复用（计 `replay_fits`）。

---

## E. 四臂、预算、门、统计最终表

| 项 | 值 | 来源 |
|---|---|---|
| Static | 无 Harness，raw train → raw model → raw context | `run_unit_arm` static 分支 |
| A5-frozen | 起点 K0；单元内 Fast/探针/门/部署照常；**不带走**任何东西（store 重建、无 bank、无 Draft、无 Active） | `Arm.begin_unit` |
| A5-online | 起点 K0；写回 bank/Active/Draft；外环 | |
| A3-online | 起点 h0；同 A5-online | |
| 空 K0 | A5-online≡A3-online 不跑；A3-frozen 为唯一 frozen 对照 | `arm_specs` |
| 单元臂 LLM | 5（在后端前阻断）；外环 2/步 | `BudgetGuard.reserve` |
| **预算不对称（须报告）** | online 臂总 LLM = 26×5 + 5×2 = 140/臂；frozen = 130/臂 | 外环 LLM 是进化的成本，不是等预算；HEC-2 可加「frozen +2/5 单元额外搜索」对照 |
| 门 | P4 四线唯一权威；online_loop approved 为必要非充分 | `resolve_gate_disagreement` |
| 统计单位 | cohort（4 个）；origin 为时间重遇 | 合同 STATISTICS |
| **主读数（预注册）** | 每顺序终点累计差 D_o = Σ_u (online−frozen)_u（评价面 aggregate_gain）；每 cohort 差 d_c（该 cohort 单元均值，三顺序平均） | 描述性 |
| **P1 成立判据（定性、无 p 值）** | D_o>0 于 ≥2/3 顺序 ∧ d_c>0 于 ≥3/4 cohort ∧ 每顺序 harm 事件 online ≤ frozen | 不得称显著 |
| 报告项 | 4 个 d_c 符号；精确二项 p（最小 .0625，**标注为描述**）；cohort bootstrap 百分位 CI（n=4，粗）；逐单元符号计数（相关样本，只描述）；AUC 与中点差为副读数 | |
| 顺序 | 三顺序分别画细线；**不画跨顺序置信带**（同数据同 cache，非独立 seed）；cohort bootstrap 带只围绕三顺序均值曲线 | |
| 确认性检验 | 留给 fresh 实验：≥8 个独立 cohort/数据集，单侧 Wilcoxon 符号秩（或 sign：8 个中 ≥7 正 p=.035） | HEC-2 跨数据 + Solar |
| 只能描述的数 | 逐单元增益、逐 origin 曲线、harm 计数、生命周期计数 | cohort 内共享同一训练语料与 program model |

---

## F. Phase F 精确开封协议

1. 前提（机械）：三顺序 `audit_hec1_instrument` 8/8；课末读数判词 = `HEC1_EVOLUTION_SUPPORTED`（sol 确认）；用户签 `seal_released`。
2. 冻结：每（顺序 × 臂）末态 store 的 `active.json` sha 写入 `hec1_phase_f_freeze.json`；Static 一份。
3. 0-LLM 生成（不读 Outcome）：对 held-out 5 origin × `[80:120]`，逐臂：读 origin 之前数据 → 特征卡 → Active Skill Scope 解析 → 治疗集；serving-side 双管线产出全部预测；落盘。
4. 覆盖分层表（开封前写死）：每（origin, 臂）匹配序列数；覆盖 0 的层预登记为 `NOT_REACHED_ON_HELD_OUT`。
5. 一次开封：单次 evaluator 调用同时打开全部 origin × 臂 Outcome → 逐序列 sMASE → 读数；`p4t` 台账记 `EXPOSED`；`held_out_reads` 0→1 事件。
6. 读数：三顺序**全部**末态各报 + 均值；主读数 = 均值 (online−frozen)；分层报告；覆盖 0 → 判据 4 `UNSCOREABLE_NO_COVERAGE`，安全读数不得写成能力。
7. 禁止：按 development 结果选顺序；按覆盖换 origin；开封后任何重跑。

---

## G. General / Specific Skill 正式定义与生命周期

| 类 | 内容 | 形成条件 | 可读信息 | 执行权 | 生命周期 |
|---|---|---|---|---|---|
| General Skill（bootstrap） | 程序性 how-to：`inspect_and_localize`、`build_contrastive_candidates`、`select_or_identity_and_verify`（CODE FACT：每 cell `retrieved_skill_ids`） | h0 冻结；由人/Slow 策略文本面编辑 | Fast 每次读取 | 无部署权（不含 program/scope） | HEC-1 冻结（Stage 3 编辑 REJECTED）；HEC-2/3 才开策略/文本面 |
| Target-local Specific Skill | 声明式 (program, task_consumer_key, Scope 谓词, 证据) | 本域 held-in 经 Support-A + delayed 权威门 | Fast 经 Scope 匹配检索 | 本域匹配 Scope 时部署（held-in 后续单元、held-out） | Draft → WAITING/REVISABLE/FLAGGED → Active → 收窄/撤销 → 课末冻结 |
| Source-derived Skill（K0 成员） | 同一对象，来自先前域/块 | 先前课程 Active 且冻结 | 新域以 `requires_target_support` 供给为探针候选 | **无自动执行权**；过新域 Support+delayed 后成为新域 Target-local Active（保留出处） | 与 Target-local 同 |
| Shared Capability | 跨 ≥2 域相似可观察 Context 下重复正向 + 风险证据 | AGENTS §4 严格门 | 同上 | 可申请零/低探针跨域执行权（需更强 fresh 证据） | 项目内尚无实例（S2 candidate v2 为 `SHARED_CANDIDATE, target_support_required=true`） |

**划分与晋升**：General 是「怎么搜」，Specific 是「什么在哪有效」——**不互相晋升**。Specific→Source-derived 是同一对象随时间的角色变化；Source-derived→Shared Capability 靠跨域证据；Shared 永不变 General。

**完整持续进化的定义**（Q38）：Skill 形成 → 后续单元失败 → 外环修订（子句）→ 修订版在**新**单元过 Support+delayed（Active）→ 再重遇且优于 frozen 同单元；且曲线判据成立。只 ADD/REVOKE = `RISK_CONTROL_ONLY`；Active 无修订无重遇 = 一次适应。

---

## H. 任务 / 数据 / setting / baseline / 中间实验总矩阵

| 层 | 检验什么 | 指标 | 对照 | 数据 | 可支持的 claim |
|---|---|---|---|---|---|
| E0 Natural headroom | 菜单里是否存在过预算的全局/逐序列方案 | Best-Safe-Global in-budget 数、per-UID oracle | identity | Phase T 单元评价面（事后） | 「有东西可学」；无 Agent 主张 |
| E1 Transformation fidelity | 算子是否做到自称的事 | 局部恢复率、附带改动 | finite-median 等 | 受控注入 sidecar（POSITIVE_CONTROL，独立命名空间） | 仪器/算子保真；不进主图 |
| E2 Targeting/Scope 可分性 | 部署可见特征能否分开受害者 | AUC（D1 四信号 0.45–0.65） | 随机 | Source-v3 七窗口（已做） | Observation 边界 |
| E3 Candidate generation | Fast 提案质量 | PROPOSED/ABSTAINED 分布；Support 过预算率 | Random-valid（HEC-2） | Phase T | Fast 贡献 |
| E4 Harness edit/evolution | 外环 → Draft → Active → 重遇 | 修订成功率、存活率、重遇收益、Slow-vs-ScopeFit 一致率 | ScopeFit shadow / Random-edit（HEC-2） | Phase S/T | 进化机制 |
| E5 End-to-end | 曲线 + held-out | D_o、d_c、harm、advantage | frozen / Static / Best-Safe-Global | Phase T → Phase F | 头条 |

任务：Forecast 主任务（唯一有 serving-side 管线 + 自然 headroom 证据）；AD 条件化/安全副任务（C12/C13 翻转、`INVERTED_EFFECT_OBSERVED`、正确弃权）；Classification 可选附录（菜单无 headroom）。
数据：Development = KDD with-missing `[0:239]` 全部课程块 + NOAA 2024/2025 + NAB + Yahoo-24（全部已曝光）；Confirmation = `[80:120]` × held-out 5 origin（F1，sealed）；Fresh Final = Solar（F2，隔离中，资格待核）、Yahoo-41（AD，sealed）、NOAA beyond_17520（sealed，须过台账）。打开过的永远是 development。

Baseline 取舍：**HEC-1 保留** Static、A3/A5-frozen（= 同预算逐单元从零搜索，即 validation-search 的自然形态）、Best-Safe-Global（改 prequential + 保留 outcome-selected 作 offline comparator）、ScopeFit shadow。**延后 HEC-2**：Random-valid search、Random-edit、ScopeFit 第五臂（触发条件见 Q41）、FFORMS-style router（P4 已判 `FEATURES_DO_NOT_BEAT_A_FIXED_CHOICE`，引用不重跑）。**不做**：LLM-direct、AegisTS（需注入真值 + RL）、frozen code agent。

---

## I. 论文 Problem–Gap–Challenge–Method–Claim 链

- **一句话问题**：在没有 clean truth 的自然时序上，数据准备是否「有效」只由下游任务/模型/局部模式决定；一个受治理的 Agent Harness 能否随经历在风险预算内持续变好，而不是每个新窗口从零搜索？
- **核心 insight**：历史反馈只可用于**低成本选择**（普查、replay 淘汰、Scope 校准），执行权只能来自**未来 held-in 反馈**（新单元的 Support + delayed）；两个时间尺度分离才能既积累又不自批。
- **方法**：双时间尺度受治理的 Harness 进化 = 内环（单元内 Fast 提案/探针/权威门）+ 外环（每 k 单元普查 → Slow 语义方向 → 工具校准阈值 → replay 淘汰 → restricted Draft 三态机）+ 冻结/held-out 纪律。
- **贡献（对应证据）**：(1) 条件化现象的自然与受控证据（C12/C13、C18/C19、P4d 组合程序、D1 pooled/per-channel 失败形状）；(2) 逐序列风险预算下的反馈验证生命周期与零事故记录（C9、P4b、Epilepsy2）；(3) 双环治理进化与 online−frozen 曲线（HEC-1，待读）；(4) 失败机制的可复算归因 H1/H2/H3（Source-v3、Phase S/T）。
- **Headline 选择（Q44）**：B（governed self-evolving Harness）为头条，A（conditional natural readiness）为问题设定与风险结构来源。并列会发散；若曲线 NOT_SUPPORTED，按预注册转 Track B（机制分析），头条降为 A + 机制，**不得**再称进化成功。
- **Intro 五段**：① 自然数据无 clean truth，质量随消费者变（举 C12/C13/C18 反号）；② 现有路线：固定清洗/AutoML/Router 开环，harness 进化工作以 benchmark 分数无风险门；③ 挑战：反馈稀疏、尾部风险绑定、跨窗口成员更替、自批风险；④ 方法：双环治理 + 冻结纪律；⑤ 结果与诚实边界：development 曲线 + fresh held-out + 负结果。
- **最相关工作（仓库内可核 PDF；venue 未核）**：*Self-Harness*（harness 自改进，无风险门/无逐序列 harm/无 held-out 冻结）；*Continual Harness*（在线适应 harness，评测型，无「选择 vs 授权」分离）；*Harness Updating Is Not Harness Benefit*（把更新与收益拆开——与我们 frozen 对照同思路，应作为方法学近邻引用）。**绝不写**：「首个自进化时序 harness」「跨域迁移已证」「LLM 发现了 Scope 规则」（除非 shadow 一致率显著低于 100% 且 Slow 不劣）。
- **与 AegisTS / TSPred·FFORMS / TimeClaw 边界（Q48）**：AegisTS 用注入真值与 RL 奖励优化修复保真——我们无真值、只用下游反馈、有治理；TSPred/FFORMS 静态特征→方法开环选择——我们闭环反馈且 P4 已证开环树 router 不胜固定选择；TimeClaw/Self-Harness 以 benchmark 分数为进化目标、A5/A3 为开关——我们有逐序列风险预算、权威门、frozen 对照与 held-out 冻结。
- **读数分层（Q49）**：主表：C1（fresh 效率）、C18/C19（自然跨 cohort 反号 + 首次自然闭环）、P4d 组合程序正证据、D1、Source-v3 H1–H3、HEC-1 曲线；附录：C12–C17（POSITIVE_CONTROL）、P4b/P4c 负结果、批次配方线、W47–W49；撤出项目级结论：KDD without-missing 结果对 imputation 的任何表述、C6/C7 同窗 RESCOPE 作为效用主张、NOAA 2025 作为 fresh held-out。
- **审稿拒稿五因与最低成本修复（Q51）**：① 单数据单 Consumer → HEC-2 per-channel + Solar；② development 曝光曲线 → Phase F + 明确命名；③ n=4 无显著性 → 描述性 + 预注册确认实验；④ LLM 贡献不清 → 报 Slow-vs-ScopeFit 一致率 + Random-valid；⑤ 「数值 Scope 优化器套 LLM」→ 同上 + General 面冻结声明。

---

## J. HEC-2 / HEC-3 / TSFM 路线

1. **HEC-2 ①** Consumer 轴 pooled vs per-channel（P-C1 尾部消失 / P-C2 基座上升）——直击绑定约束，同课程重跑一臂即可。
2. **HEC-2 ②** Random-edit / Random-valid 对照；ScopeFit 第五臂按 Q41 触发。
3. **HEC-2 ③** 跨数据（electricity/traffic 天然单元；Solar F2 安全读数）→ 才可谈 cross-domain。
4. **HEC-2 ④** Scope 解耦（prepare_scope vs model_scope）——仅当 per-channel 未消除尾部时。
5. **HEC-3** 观察面：模式持续性（近 W 窗口内同缺陷出现比例）+ 缺陷相对预测起点位置；新 cohort 前瞻冻结。
6. **TSFM**（Q54–56）：Chronos-bolt-small zero-shot 为第三 Consumer；Program 只作用 serving context → 无 program-model 路由 → pooled 尾部机制按构造消失；数据同 KDD 课程 + Phase F；先只做柱 Ⅰ 条件化（固定程序、无进化），Ridge 曲线成立后再做曲线；Skill 的 Scope/program 可迁移、证据不可迁移（`task_consumer_key` 分列，进入新 Consumer 只作 `requires_target_support`）。Ridge 曲线不成立仍做 TSFM 条件化实验（≤ 200 fits 级，0 LLM），不做 TSFM 曲线。

---

## K. 失败分支决策树（first-fault → 唯一下一步）

| 结果 | first fault | 唯一下一步 |
|---|---|---|
| Phase S 0 Skill | 若无 Draft 过 Support → 供给面；若过 delayed 死于重遇 → Observation（H1/H2） | K0 空缩臂照跑；HEC-2 单假设按主导归档原因选 |
| K0 有卡但 Target 匹配 0 | 覆盖=流行率（H3） | 分层报告；HEC-2 覆盖语义改跨窗累计 |
| online = frozen | 三分账为 0：无 Active → 供给/门；有 Active 未召回 → 检索/Scope；召回无收益 → 记忆面 | 只动分账最大的那一面 |
| online > frozen 但 harm 更高 | 召回把风险带过来 → Risk 面（D1 已判不开） | HEC-2 ① per-channel |
| Draft 多无重遇存活 | H1/H2 | HEC-3 观察面 |
| 三顺序方向不一致 | 课程序敏感 | 报告 ≥2/3 未达；HEC-2 加长课程 |
| Phase F 回落 | +1200 步非平稳（H2） | 如实报；HEC-3 |
| A5 < A3 | K0 负迁移/探针位被占（SUPPLY_STARVATION） | 报积累为负；HEC-2 先验对称/探针位策略单面 |

真正否定方法（Q58）：三顺序 online 均不优于 frozen ∧ 修订存活 0 ∧ HEC-2 per-channel 后仍如此 → 「风险门下的跨单元记忆」在本任务族无收益；或 ScopeFit shadow 处处 ≥ Slow → LLM Slow 非必要。其余只否定课程/Observation/Program/Consumer/协议。

---

## L. 继续做 / 停止做

**继续**：commit（allowlist）→ sol 裁 replay 份额 → 写 `audit_hec1_k0_freeze.py` → Phase S（13 单元）→ K0 → Forward → 仪器 → Reverse → Interleaved → 读数（非作者）→ Phase F（人批）。并行 0-LLM：Best-Safe-Global prequential 版；论文柱 Ⅰ–Ⅲ 骨架。
**停止**：1900 行 runner 重构；O6/仓库清理（Phase S 后）；新增 SHA/Gate/平台；改阈值/算子/观察特征；任何 held-out 读取；用 Forward 效果决定续跑；再讨论 A5 是否主线（已定）；为凑样本混数据域。

---

## M. 继任简报（不依赖我在线）

**先读**：项目 `AGENTS.md` §1/3/4/6/7/8；本文件 B/D/E；`OPUS_HANDOFF_BRIEF` §2c/5b；账本 2026-09-03 10:xx 两条。
**工作树状态**：`evaluation/main_protocol_p4/{run_hec1,outer_loop,restricted_draft,audit_hec1_instrument}.py`、`tests/main_protocol/test_hec1_{wiring,end_to_end}.py` 有本审查改动，未提交；`tests/main_protocol` 424/424；`--smoke` 7/7；离线 e2e 与 K0 冒烟工件在 `artifacts/main_protocol/hec1_*_{e2e6_review_fixes,k0_smoke_s6,k0_smoke_t2,k0_smoke_t2b}.json`。
**commit allowlist**：`evaluation/main_protocol_p4/`（全部 .py）、`tests/main_protocol/test_hec1_*.py`、`artifacts/main_protocol/{hec1_*,p4ab_*,p4ac_*,p4w3b_*,p4x_*,p4y_*,p4z_*,p4u_*,p4v_*,p4t_*}`、`docs/{HEC_EVOLUTION_MAINLINE_PLAN,HEC1_*,D1_*,D2_*,D4_*,DISPATCH_PACK_*,MORNING_REPORT_2026-09-04,OPUS_HANDOFF_BRIEF_2026-09-03,FABLE_FINAL_REVIEW_*}.md`、`docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md`。**排除**：`.hec1_runs/`、`*_store*`、`__pycache__`、`_scratch/`、凭据、`methods/ttha/*`（另一执行线未提交件）、`AGENTS.md`（另一线）。禁 `git add -A`。
**发车序**：(1) sol 裁 replay 份额（建议：每臂额度 = 该臂自身课程 fits 的 100%，或每次 screen 只回放最近 8 个已处理 cell；二选一，只改一处）；(2) `python -m evaluation.main_protocol_p4.hec1_contract` 生成冻结收据；(3) 写并跑 `audit_hec1_k0_freeze.py`（非空：收据 `snapshot_resolved`、每张 Active 卡有 `activated=True` 的 cell 且 `resolved_by=p4_gate`、外环 `wrote_active=False`；空：缩臂配置）；(4) Phase S live：`python -m evaluation.main_protocol_p4.run_hec1 --phase phase_s --run-label phase_s_live_<date>`（脱离终端、心跳、`--resume`）；(5) `--k0 artifacts/main_protocol/hec1_k0_phase_s_live_<date>.json` 跑 Forward；(6) `audit_hec1_instrument <course>` 8/8 才续 Reverse/Interleaved；(7) 停在读数前。
**每步账本一条**；工件永不覆盖（runner/audit 已拒绝）。

---

## 62 问逐条（按组，压缩）

**一、可运行性**
1. 同一 Harness：CODE FACT `run_course`→`Arm._build`→`TTHAMethod(TTHAFastAgent(core), start_snapshot)`，每 cell `online_loop.run_online_round`；Phase S 单臂同路径；离线只换 backend 与外环 Slow。不是 Router：程序由 Fast 提，Scope 由 initializer/工具，执行权由门。
2. 发车前 ≤3：replay 份额裁定；本审查改动 commit+复核；（Forward 前）`audit_hec1_k0_freeze`。
3. 0-LLM e2e 最小覆盖：七项已有（k=5 触发、写回、frozen 重置含 ledger/库、P4 唯一激活、20/19、+144 不进 bank、resume 不重扣）；**缺口**：NARROW/REVISE 活体路径未在 e2e 里被走到（ADD 不需 Slow）——建议一个强制 NARROW 的离线课程或接受单测覆盖。
4. `assert_frozen` 查漂移、`assert_launchable` 查放行；空洞：不检查「e2e 已过」（AUTO_CONTINUE 为散文）、不检查 K0 收据可编译（现由 `run_course` fail-closed 补）。
5. 未退化为数值 Router：工具不选程序/不选特征/不决定行动；但 Scope 修订可能全数值——由 `best_stump` shadow 度量并必须报告。
6. 普查键 = task_consumer_key × 完整 typed program（算子序、顺序、参数）**+ Scope 根谓词**；不含故障类型；指纹只折叠别名（CODE FACT `_program_signature`/`_alias_classes`）。

**二、内部矛盾**
7. 时间线见 D。
8. 第一次外环 5 个 cell；样本少但可选（replay 逐 cell 否决）；**真正问题是额度线性耗尽**（B/E 数字）；建议份额或回放窗口二选一改。
9. replay 只筛选：CODE FACT `screen` 返回 `does_not_grant`，`open_restricted` `revisions=0` 无部署权；激活只在 `run_unit_arm` 经 P4 门。严格做到。
10. 三态充分性：三机制各有归位；REVISE 原地修订后证据保留（`history`）；**未覆盖**：coverage 失败是否耗 attempt（P1 待裁）。
11. 计数从**根谓词**算子句（`validate_narrowing(root=...)` ≤2）；修订/验证按 Draft 计——修复前可换壳（新壳计数归零、旧壳继续供给），修复后不可。
12. 最宽阈值 = 最大覆盖 = 最大新进入者暴露；HEC-1 只记录（合同 8-8a）；HEC-2 候选：证据有界（最紧可行）阈值。
13. 仍可能成功的具体机制：**召回替代重搜**（frozen 每单元烧探针重找同一程序，online 直接部署已 Active 程序并把探针位留给新候选）+ **重复 POSITIVE 铸卡（ADD）**——不依赖新进入者分离；Scope 修订机制大概率不出存活。

**三、统计**
14. 自相矛盾：是（合同 L728-740）。
15–19. 见 E：描述性主读数 D_o/d_c + 定性预注册判据 + cohort bootstrap；确认性留 fresh（n≥8 cohort，Wilcoxon）；不画跨顺序置信带；P1 用终点差（副：AUC/中点差），不用「单调」；可检验的只有 cohort 级聚合，且只作描述。

**四、四臂**
20–21. 见 E；frozen 保留单元内 Support/门/部署，唯一差异 = 跨单元记忆（store/bank/Draft/Active），修复后成立。
22. 空 K0：起点相同故 A5≡A3；A3-frozen 必要且唯一；可支持域内自进化（判据 1/2/4），不支持任何积累主张。
23. 命名：cross-cohort / within-dataset (KDD air-quality blocks) accumulation。
24. 三分账缺 (d)「外环 Draft 被 resupply 并部署」通道与 (e) prompt 分叉采样噪声；预算不对称 +10 LLM/臂（外环）须报告。
25. 最低冻结：模型 id、解码参数、`llm_cache` 规范化 prompt 键跨臂共享、cache 命中率、prompt 相同的 unit-arm 比例、Fast 决定四分类分布；三顺序为唯一重复。

**五、Phase F**（见 F）26. 不是同一个 Fast Agent → 命名「冻结 Skill 库机械部署」；27. 三顺序全部末态；28. 接受（补 K0 空仍开封）；29. 同 cohort 时间泛化；30. 见 F；31. `NOT_REACHED_ON_HELD_OUT` / `UNSCOREABLE_NO_COVERAGE`；32. HEC-2 ③ + Solar。

**六、Scope/Program/路由**
33. 耦合维持到 HEC-1；HEC-2 先 per-channel（若尾部消失则解耦无必要），否则拆 prepare_scope/model_scope。
34. 新进入者 → Scope 可修但 Observation 分不开（D1）；持续成员翻号 → Observation/Program；覆盖塌陷 → 协议语义。
35. HEC-3 特征：模式持续性（近 W 窗口同缺陷比例）、缺陷相对预测起点位置；raw/program 预测分歧作候选（D1 AUC .65 最高）。
36. Consumer-conditioned 分离字段：`task_consumer_key` 含 consumer 结构；风险证据分 msh 与 hf 两线记录。
37. 见 G。38. 见 G 末。

**七、Baseline**
39. 见 H。40. **事后 Outcome 选优** → offline in-budget comparator；1820 fits 值得，但应加 prequential 版（+48 选、+144 分，≈1820+52 fits）。
41. 第五臂触发（预注册）：shadow 不一致率 ≥50% 且 Slow 在重遇读数上不劣 → HEC-2 第五臂；一致 ≥80% → 不加臂，论文如实写；Slow 劣 → 采 shadow 为 HEC-2 变体。
42. 见 H。43. sidecar：KDD 同块注入尖峰/短缺口（有私有逆），指标局部恢复 + 下游增益 vs clean-oracle；`POSITIVE_CONTROL` 独立命名空间，永不进主图。

**八、论文** 44–51 见 I。50. claim 总表（压缩）：条件化 | KDD/NAB/注入 | 固定程序 | 反号计数 | 已有（DEV/PC）；反馈验证零事故 | KDD/Epilepsy2 | 无门 | harm 事件 | 已有；积累效率 | NOAA | A3 | 首正重训 | 已有（FRESH n=1）；进化曲线 | KDD 26 单元 | frozen | D_o/d_c | 待跑；held-out 保持 | `[80:120]`×5 | frozen | 均值差 | 待 Phase F。

**九、任务/数据/TSFM** 52–56 见 H/J。

**十、失败/路线** 57–58 见 K；59. 见 A-9；60. 48h：commit、replay 裁定、k0 审计脚本、合同收据；1 周：Phase S → 三顺序；3 周：读数、Phase F、HEC-2 ① 预注册、论文柱 Ⅰ–Ⅲ；停止清单见 L。61. Phase S 120 / 三顺序各 500 合理；Best-Safe-Global 1820 值得（改 prequential）；replay 份额需改。62. 三个实验：HEC-1 曲线；HEC-2 ① per-channel 同课程；Phase F + Solar F2。

---

## 末节

1. **Phase S 发车前 ≤3 阻塞**：① sol 裁 replay 份额（否则外环后半为仪器伪影）；② 本审查五处修复 commit + 一次非作者复核（可由 grok 按清单 B/C 组）；③ `audit_hec1_k0_freeze.py`（Forward 前）。
2. **最危险的科学设计问题**：外环的学习信号被两道结构性门夹死——D1 无 outcome-free 分离量（新进入者伤害不可预判）+ replay/覆盖成本结构（额度枯竭、覆盖=流行率）——使 Scope 修订通道几乎注定 0 存活；曲线若为正，几乎只能来自召回/探针位机制。论文叙事必须提前接受这一点。
3. **最不同意 sol 的一项**：统计方案照原文（cohort 单位 + sign test α=.05）——数学上不可能成立，须改为描述性主读数 + 定性预注册判据 + fresh 确认实验。
4. **最可能产生正结果的路径**：A3-online vs A3-frozen（空 K0 形态）在 KDD 上通过 ADD（≥2 单元重复 POSITIVE → Draft → 新单元过门 → Active）+ 0-LLM 召回释放探针位，得到小幅正的累计效用差与相等 harm；配合 Phase F 同 cohort 时间保持。
5. **待用户决定 ≤5**：① 是否批准 replay 份额改动（sol 裁后）；② commit 时点；③ Best-Safe-Global 是否加 prequential 版（≈+1870 fits）；④ Phase F 开封（课末）；⑤ HEC-2 ① per-channel 是否作为 Phase F 后第一个实验。
