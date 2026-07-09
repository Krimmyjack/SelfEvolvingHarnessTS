# 预注册草案 — Track B1：慢路径 Proposer（LLM 身份判决）

> **状态：DRAFT，待用户审阅后锁定；锁定前不启动正式外呼**（用户第三十七轮）。
> 实现 Component Plan §13.2。夜间已建：编辑面（B0）、gym scaffold（`run_proposer.py`）、
> 模型接线（gpt-5.4-mini/agicto）、API smoke ✅、单 batch dry-run。**正式 LOFO×full-budget×
> 天花板模型待本 prereg 锁定后开。**

## 0. 核心命题与两阶段诚实边界

- **命题（§13.2）**：LLM proposer 读 **per-family 分解** mining 报告，能否在**同一可执行编辑
  空间、同候选预算**下，提出枚举/搜索找不到的、**可迁移**的 harness 结构编辑？
- **★两阶段边界（用户第三十七轮 readiness bar 的直接推论）**：
  - **B1a（本 prereg，enumerable 空间校准）**：编辑空间 = **有限 scoped risk 规则网格**
    （`run_proposer.SPACE`，210 条单规则）。此空间**可穷举** → det_search/random 原则上能覆盖
    → **LLM 在此不可能因空间大小获得内在优势**（除效率）。∴ B1a **不是**完整身份判决，它验证的是：
    ①harness 正确（同空间/同预算/select→heldout 管线）；②**winner's-curse 守卫**（random 负对照
    不得 ≈ LLM，否则 validator 在当选择器）；③"谁把 per-family 报告转成好编辑"的下界对比。
  - **B1b（真正身份判决，本 prereg 只登记不跑）**：编辑空间 = **开放/大空间**——自由特征提案
    （self-evolving PatternSpec）、算子组合、新 menu 动作——**枚举不可行**处才是 LLM 的生态位。
    B1a 通过（管线可信 + 负对照干净）是 B1b 的前置。
- **不做结果后 prompt/温度/阈值调参**（改动 = 新 prereg 版本）。

## 1. 编辑空间（B1a，四臂共享）

- `SPACE` = {feat ∈ 5 结构特征} × {op ∈ ≥/≤} × {value ∈ 0.2/0.4/0.6} × {group ∈ heavy_median/
  stl/smoothers} × {replacement ∈ v_none/v_median/f0_median_w25}，= 210 条 scoped **ban** 规则。
- 候选 = 1–3 条规则的编辑集；经 B0 `apply_edits`（校验：scope 必带、目标 ∈ 池、规则不重复）→
  `compile_bundle` → `RiskAwareRouterPolicy`（落 overlay 消费面，§13.4 焊接）。
- **同 budget N**：每臂恰 N 个候选（越界/解析失败的 LLM 候选计入"无效"，不补齐）。

## 2. 四臂（同空间 + 同 budget + 同评估）

| 臂 | 提议来源 | 作用 |
|---|---|---|
| **random** | 空间内均匀采样（seeded） | **负对照**——防 winner's curse（validator 当选择器） |
| **det_search** | mining 报告的 `report_estimate` 排序取 top-N + 一个 greedy 多规则 | 确定性搜索基线（§13.2 arm①） |
| **llm_nomem** | LLM 读 per-family 报告 | LLM 无经验 |
| **llm_mem** | LLM 读报告 + 检索到的过去成功编辑 | LLM 带情景经验 |

- **信息面纪律（§13.2；部署侧杀死 LLM 的族盲压缩不得重演）**：mining 报告 **per-family 分解**
  （`build_mining_report`，禁跨族聚合）；LLM 写入规则须 **per-family/worst-group 作用域**
  （空间的 scope 字段强制；越界作用域校验拒绝）。

## 3. 模型（天花板防假阴性）

- **天花板 = Claude Opus 4.8**（§13.2：连最强模型都打不过枚举 → 方法真死；打得过 → 再问便宜
  模型）。正式跑用 **subagent-cache 机制**：orchestrator 用 Claude subagent 按 mining-report
  哈希生成候选、写入 `Proposer/claude_cache/<hash>.json`，gym 的 `ClaudeProposer` 读缓存（保证
  确定性重放 + 可审计）。
- **便宜复制臂 = gpt-5.4-mini**（agicto，已接线 + smoke ✅）。**执行顺序（用户第三十七轮）**：
  gpt-5.4-mini 先（夜间管线/dry-run 已用），Claude Opus 作**决定性天花板**在正式跑。
- **prompt 作受控变量**：`_LLM_INSTR` 版本入 report；天花板/便宜用**同一 prompt**（差异只在模型）。

## 4. 评估与统计（LOFO，防 winner's curse）

- **LOFO**：8 族逐一 held-out；报告与 selection 只用其余 7 族；held-out 族评迁移。
- **select-on-selection → eval-on-heldout**：每臂 N 候选先在 selection 选最优、再在 held-out 评
  （in-sample 选择的 winner's curse 由 held-out + random 负对照双重暴露）。
- **指标**：held-out utility（frozen_regret − edited_regret，>0 改善）/ **worst-family**（作用域
  健康）/ **有效新编辑数 n_fire**（真改变 pick 的行数）/ 跨 family 迁移 / **成本（LLM 调用数=独立轴）**。
- **CI**：full-refit group bootstrap（沿用 STAGE1 A-33c）——正式跑上。

## 5. 判据（B1a）

1. **负对照干净**：random 的 held-out utility **显著低于** llm/det_search（若 random ≈ 之 →
   validator 当选择器，B1a 判无效、回设计，不进 B1b）。
2. **det_search vs LLM**：
   - LLM > det_search（held-out，CI 不跨 0）→ **LLM 在 enumerable 空间已有边际** → 直接进 B1b；
   - LLM ≈ det_search → enumerable 空间无 LLM 特异价值（预期结果之一）→ **进 B1b**（开放空间才是
     生态位），B1a 结论=管线可信 + 负对照干净，不是"LLM 无用"。
   - LLM < det_search → LLM 连有限空间都不如搜索 → 记债，B1b 门槛提高。
3. **天花板优先**：先跑 Claude 天花板；天花板都打不过 det_search+random → B1a 即给 identity 强负
   信号（但仍须 B1b 开放空间复核，因 enumerable 空间本就利于搜索）。

## 6. 分支（§13.2）

- 全程通 + B1b（开放空间）LLM 胜 → 论文形态 **"LLM-driven TS harness evolution with compiled
  deployment"**（比 SkillOpt 完整——彼无便宜部署路径）。
- B1b LLM 败 → **命名诚实收缩**（§13.5："pattern-conditioned adaptive TS preprocessing policy，
  进化层由非-LLM 执行器承担"）——但这是**用户自己实验**得出的答案，非定义维持。

## 7. 当前 scaffold 就绪度（用户第三十七轮清单 → 正式跑前须补）

- ✅ 同空间 + 同 budget + select→heldout 管线（`run_proposer.py`，dry-run 验证）。
- ✅ 模型接线 gpt-5.4-mini + smoke。
- ⏳ **ClaudeProposer（天花板）subagent-cache**：正式跑前实现（§3）。
- ⏳ **llm_mem 的检索经验**：dry-run 用手传列表；正式跑接 B0 `MemoryWrite`→EvidenceStore→检索
  （D7-lite；与 Track C1 共用）。
- ⏳ **det_search 强化**：当前 report_estimate + greedy；正式跑可加束搜索使其成为更强对手（防
  "LLM 赢在 det 太弱"）。
- ⛔ **PatternSpec/menu/算子编辑（B1b 开放空间）**：B0 只记录 PatternSpec 提案（deferred）、menu
  编辑缓办——**B1b 的编辑面须先在 B0 上扩建**（self-evolving PatternSpec 提取器 + menu 编辑 op）。
- LOFO×full-budget×CI×天花板 = 正式跑内容，**待本 prereg 锁定**。

## 8. ★拍板点（用户早上确认）

- **A. B1a 先行**是否 OK（enumerable 空间校准 + 负对照，明知不是完整身份）？
- **B. 天花板模型**：Claude Opus 4.8（subagent-cache）作决定性天花板 + gpt-5.4-mini 便宜臂——
  确认此方案（我无独立 Claude API key，用 subagent；若你有 key 可换直连）。
- **C. B1b 开放空间编辑面**：是否本轮就在 B0 扩建（PatternSpec 提取器 + menu op），还是 B1a 出
  结果后再定 B1b 形态？

---

## 9. ★B1b-mini 定稿（第三十八轮 压缩计划，2026-07-07）——本节为正式跑设计，覆盖上文 B1b 草图

**压缩计划**（额度受限）：跳过正式 B1a 大跑（B1a 保留为集成测试），预算集中于 **B1b-mini**。

### 9.1 开放编辑面（`policy/program_edit.py`，与 B1a 本质不同）
- 编辑 = **提议新程序** = 现有算子组 **1–3 步有序链**（可改顺序/参数/适用条件），非有限规则重路由。
  - step1（必需）imputer∈{impute_linear,impute_fft,impute_ema,period_complete}；step2/3（可选）
    outlier∈{winsorize,outlier_iqr,outlier_mad} | denoise∈{denoise_median,savgol,wavelet,stl,smooth_ma}；
    窗算子（median/ma）window∈{5,9,15,25}；顺序有意义。scope=适用 cell 集。
- 空间 = 数百-数千程序（N=10 预算下**不可枚举** → LLM 生态位真身份检验）。机械 Gate 免费先过
  （步数/算子契约/窗合法/首步 imputer/非空 scope/novelty）。ProgramSpec→ActionSpec→ActionCompiler 直接编译。

### 9.2 执行器（唯一主要开销，`run_proposer_b1b.py`）
- 新链不在缓存 L_test → **须真实执行**：ActionCompiler.to_harness → fast_process → FrozenProbe OOF
  nRMSE（与池动作 L_test **同评估器**，frozen-probe 非全 DLinear）。
- **架构 de-risk 已证**（`--proof`）：novel `impute_linear→winsorize→denoise_median(w9)` 端到端出真损失
  (mean 2.53 vs 池 v_winsor_savgol 2.63，同评估器可比、损失确不同)。
- 每不同程序 → {uid: oof_loss} 全语料，**按程序 SHA 落盘缓存 + 链去重**（三臂/重跑免费）；≈0.3s/序列/程序
  → 全 672 × ~30 程序 ≈ 2h（背景跑）。gym 在已执行损失上 cache-replay（LOFO 便宜）。

### 9.3 三臂 + 模型 + 规模（用户 2026-07-07 拍板）
- 三臂：**random**（负对照/winner's curse 守卫）/ **det budgeted search**（mining 引导，强基线）/
  **LLM(gpt-5.4-mini)+memory**（读 per-family 报告 + 检索过去成功程序）。压缩：删 gpt-mini 复制臂/
  llm-nomem/多档模型；**LLM+mem 若胜再补 llm-nomem** 判 memory 必要性。
- **天花板模型 = gpt-5.4-mini**（用户拍板；非最强）→ **判读非对称：LLM 胜=强正（连便宜模型都赢枚举）；
  LLM 败=非决定性**（弱模型假阴性）→ **触发升级到更强模型**再定论，不即判"LLM 无用"。
- 规模：**预锁 4 族 dev-gate LOFO**（CHALLENGE={S_both,S_regime,S_trend,S_multiseason}），N=10（第 39.5 轮评审据
  额度定案；早前"全 8 族"作废）。**另 4 族 untouched**，仅 dev-gate **明确成功后**才确认扩跑——**4 族结果
  不得写成完整跨族泛化结论**。同空间/同 budget/同 grammar/同 selection/同 validator 次数。

### 9.4 省额度评估结构（压缩计划）
- 所有候选先过**机械 Gate**（免费）；selection 用**单 grounded judge**（执行器 OOF 效用）选每臂最优；
  **独立 reporter 只在 held-out 执行每臂胜者**（非全候选）。LOFO：留一族，selection=其余 7 族选、held-out 族评迁移。
- 主指标：held-out utility（frozen_regret − program_regret，>0 改善）/ worst-family / 有效新程序命中数 /
  跨族迁移 / 成本（LLM 调用数=独立轴）。**CI**：full-refit group bootstrap（沿用 STAGE1 A-33c）。

### 9.5 判据（B1b-mini）
1. **负对照干净**：random held-out utility 显著低于 det/LLM（否则 validator 当选择器 → 判无效回设计）。
2. **det vs LLM（★主判据，必须配对）**：判据 = **配对 Δ_{LLM−det}=U_LLM−U_det 的 group-bootstrap CI**
   （同 held-out uid 逐序列配对），**非**两臂各自 vs frozen 的 CI（"LLM CI>0 且 det CI>0"不能宣称 LLM 胜 det）。
   Δ CI 下界>0 → LLM 承重（补 llm-nomem 判 memory 必要性）；CI 跨 0 → 便宜模型无特异价值（非"LLM 无用"，
   升级天花板复核）；Δ 上界<0 → 记债，升级天花板。
3. **转正安全**：胜者 worst-family LCB 不劣 + first-encounter harm 受控（δ_safe）。

### 9.6 红线
只组现有已注册算子（供给不变）；不改 PatternSpec/config_sha e4f10d11128e943a；不碰 frozen_arms/
seeds20-39/holdout。C1-lite 已关 deployment episodic memory（[[project_c1lite_verdict]]）——与本 proposer-memory
（检索过去成功**编辑**）是不同面，不互相预判。

### 9.7 ★AS-BUILT（第三十九轮，2026-07-07；`run_gym_b1b.py` 建成并 dry-run 验证，覆盖 §9.2/§9.4 细节）
实现比 §9 草图**收紧了三处 rigor**，正式跑锁定以下 as-built 口径：
- **同尺度基线（关键修正）**：缓存 `L_test` 是 nested **DLinear**，执行器是 **FrozenProbe OOF** →**尺度不同**。
  故 frozen 基线与新程序**同走执行器**：`pool_baseline` 把 10 池动作各执行一次（FrozenProbe OOF），
  frozen_loss[u]=pool_exec[路由pick(u)][u]。缓存 L_test **仅**用于 report 提示 + frozen 路由选动作（尺度无关）。
  utility=frozen_loss−program_loss（**oracle 差分抵消**，无需 oracle）。
- **resolved-chain 身份（算子身份修正）**：novelty/dedup/执行器缓存键一律用**编译后** (op, window) 签名
  （`resolved_sig`）——bare `denoise_savgol` 默认窗=11 ≡ v_savgol、`winsorize+savgol` ≡ v_winsor_savgol：
  raw-steps 会误判 novel 并让某臂靠**重提池动作**取胜。已修 `is_novel`/`chain_sha` 用 resolved 身份（8 例守卫过）。
- **scope=可观测 cell 集**（`forecast|snrBin|missBin`，**非族**）→ 无 origin 泄漏；程序 = cell 条件化覆盖，
  与全项目 harness 面一致。det_search=canonical (imputer×body) 日程枚举 novel 链定向高 frozen_gap cell。
- **dev-gate 先行**：`--formal` LOFO 先在预锁 CHALLENGE={S_both,S_regime,S_trend,S_multiseason} 上跑（过再扩
  untouched 4 族）；CI=winner-only group bootstrap（held-out util，按 uid 重采样 2000×）。
- **DRY-RUN 验证（无外呼，96 序列/2 族/budget3，`gym_dry.json`）**：全管线通——执行器（池基线 48s）+链去重
  （S_trend 折复用 S_both 已执行链 0.1s）+ select→heldout + CI 全填。det_search 在 S_both 找到胜 frozen 的
  novel 程序（held-out +0.16，CI[+0.068,+0.244]）、正确捕获 S_trend 非迁移（sel+0.20→ho−0.275）；random 负对照
  两族皆更差（mean ho −0.374 vs det −0.058）→**gym 度量真实增益/伤害、负对照如预期**。
- **待办（正式跑）**：①接 gpt-5.4-mini LLM 臂（`LLMProgramProposer` 已写，TS-grounded prompt）；②全 672 语料；
  ③独立 reporter 复评胜者（当前 held-out util=grounded judge；reporter 用异评估器 = 正式跑补，Stage 1 判官≠报告器）。
  **①②③ 属 §9.6 spend 门后（外呼+~2h），待用户 go。**

### 9.8 ★评审修复（第 39.5 轮，2026-07-07；用户审 as-built 三缺陷 + 两解释边界）——正式跑前置，已落地
用户审出三处会**污染核心结论**的缺陷，全部零成本修复并加测（`tests/test_gym_b1b.py` 5 测过），然后才准正式跑：
1. **held-out family memory 泄漏（最严重）**：原 `formal` 按 held 族喂专属 memory（`mem_by_fam.get(held)`）→ LLM
   收到目标族+答案方向（"S_trend→STL"），**非真 LOFO**。**已修**：每折 memory 只取 **selection 族**
   （`sel_families=[f for f in dev if f!=held]`），held 族条目硬守卫排除（`assert not held_lines & mem`）；每条
   seeded 经验仅指涉自身 src 族结构 → 排除即杜绝答案泄漏。测 `test_memory_no_leak_lofo`。
2. **主判据未实现**：原只有每 winner vs frozen 的 CI（答"是否胜 frozen"），**答不了核心"LLM 是否显著胜
   det_search"**。**已修**：加 `paired_bootstrap_ci`——Δ_{A−B}=U_A−U_B 同 held-out uid 逐序列配对、按 uid 组自助，
   summary 输出 `paired_vs`（关键=`llm_mem_minus_det_search`，CI 下界>0 才宣称 LLM 胜）。测 `test_paired_ci_*`。
3. **非"全 8 族 LOFO"**：代码实为 4 族 CHALLENGE dev-gate。**已锁**：prereg/输出/汇报统一标注"预锁 4 族 dev-gate、
   另 4 族 untouched、成功后才确认"（`payload.scope_note` + §9.3 更正）；4 族结果不写成完整跨族泛化。
**两解释边界（记录，非缺陷；写入 limitations）**：
- (a) **scope=SNR×missing cell** → B1b 测的是"**LLM 能否创造/组合新算子链**"，**非**"LLM 能否进化 PatternSpec"
  （PatternSpec code-gen 是更上一层编辑面，本轮不开——压缩计划只开一个承重编辑面）。
- (b) **frozen router 已在既有合成 DGP 上训练** → 本实验检验**新程序的跨 family 迁移**，不能完全排除 router 对
  合成 DGP 的过拟合；真实域确认留到 §条件式终局（仅 B1b 胜后触发，独立 reporter + 1–2 真实域）。
**不做（用户明确）**：现在不改 N、不扩 8 族、不加 no-memory 臂——先用最低成本答最重要问题；LLM 胜 det 再补
no-memory；LLM 败因仅 mini → 按预注册升级强模型，**不即否定 LLM proposer**。

### 9.9 ★formal 结果 + 公平性缺陷 + B1b-ceiling 注册（第 40 轮评审，2026-07-07）
**formal 已跑（`gym_formal.json`，336 序列/4 CHALLENGE 族，非"全 672"——语料已限于 dev_families，另 4 族+holdout
未触；4 次 mini 外呼）**：主判据 llm_mem−det_search 原始 paired（n=252，因 llm 在 S_multiseason 产 0 候选缺席）
= point −0.0444 CI[−0.236,+0.122] 跨 0。**零外呼审计（`gym_formal_audit.json`，`--audit`，不覆盖注册结果）纠正
三处解读**：
1. **拒因全为 non_novel（信息不对称，非模型能力）**：mini 原始输出 10/6/10/3 条，Gate 后 7/2/1/0；**每一条被拒
   都是撞已有 menu 链（novelty）**，syntax/scope/dup 拒因=0。prompt 要"NOVEL"却**未示 menu 的 compiled 链**，
   而 det/novelty-filter 明知 menu → 强模型沿用同 prompt 会撞同一堵看不见的墙。"模型不能组程序"说法**撤回**。
2. **ITT 完整四折（空折 no-op=frozen）**：llm−det = −0.0484 CI[−0.180,+0.071] n=336（原注册 paired 改为 ITT 口径
   后主判据；lofo 已改 ITT 填 no-op）。仍偏 det、跨 0。
3. **两统计边界**：①UID bootstrap(n=336) 衡量固定程序下**序列**不确定性，**非跨 family 泛化**——真正独立 proposer
   决策只有 **4 折**（族级 llm−det=[+0.033,−0.036,−0.130,−0.061]，LLM 胜 1/4）。②**无臂过安全门**（worst-family
   random −0.187/det −0.099/llm −0.135 全 <−δ_safe）→ 连相对赢家 det 也不可转正部署。
**当前最准确结论**：**新程序供给有价值（det≻random 显著 +0.100 CI[+0.042,+0.164]）；mini 作为供给者未显出超
det 的独特价值**。**非"LLM proposer 已失败"、非"LLM-harness 设想错"**。

**★B1b-ceiling（单独注册，须用户 go；`ceiling(model)`/`LLMCeilingProposer`，代码已就绪未跑）**：
- **唯一改动=修信息不对称**：prompt **明示 15 menu compiled 链**（示"savgol→w11/median→w5 默认窗"）+ 允许提议
  **≤20** 条 → Gate/novelty/dedup 后**只评估前 10**（评估预算=det）；**保存原始输出 + 每条拒因**（run_arm→JSON）。
- **不改**：memory（selection-only 无泄漏）、judge（executor OOF）、评分、scope 面、4 族/4 次调用、ITT 空折 no-op。
- **判据同 §9.5.2**（paired ITT llm_ceiling−det CI）：下界>0 → LLM 承重（补 no-memory）；跨 0 → 强模型仍无独特
  价值（可较有底气写"封闭可机械验证 TS 程序空间中显式搜索比语言推理承重；LLM 于 PatternSpec/开放算子发明/
  自然语言失败反思**仍未被否定**"）；上界<0 → 明确弱于 det。**无论哪种，不据结果临时改 prompt 重跑**。
- **待用户拍板**：强模型选型（`pro`=deepseek-v4-pro | `agicto:<name>` 如 gpt-5.4 / claude-opus-4-8）。

### 9.10 ★B1b-ceiling 已跑（Claude Opus 4.8 via agicto，2026-07-07；`gym_ceiling.json`+`gym_ceiling_audit.json`）
用户选 Claude Opus 4.8（最强天花板）。连通性修复=agicto 的 claude-opus-4-8 拒 `temperature` → 客户端改 None 时省该字段。
4 次 Opus 外呼，det/random 确定性**逐位复现 formal**（一致性守卫过）。
- **公平性修复生效**：示 menu 后 Opus 拒因近零——**10/9/9/10 accepted**（vs mini 7/2/1/0），collision 消失 →
  证实"7/2/1/0 塌缩=未示 menu 伪影"，非模型能力。
- **★主判据 ITT（n=336）llm_ceiling−det = −0.0809 CI[−0.1269,−0.0367]**：**上界<0** → 按 §9.5.2 = **强模型明确
  弱于 deterministic search**（非跨 0；比 mini 平局更干净的负结果）。**仍不证"LLM proposer 无效"**（逃生口：
  PatternSpec/开放算子发明/NL 失败反思未测）。
- **机制（胜者程序）**：Opus **不是**不会提程序（填满 10 条 novel），而是**选择-泛化**差：①**scope 过泛化**
  （S_regime 与 det **同链** stl 但 scope 3 cell vs det 1 cell → sel+0.278 但 ho −0.271，正是 prompt 警告的
  over-generalize）；②**selection-overfit 选链**（S_both 选 impute_ema→median_w5 sel+0.123→ho−0.118，det 稳健
  winsor→stl +0.079）；Opus 仅在收敛到 det 稳健链+紧 scope 时才赢（S_multiseason/S_trend）。
- **族级（4 独立决策）**：Opus 胜 2/4 但败的 2 折巨亏（−0.198/−0.172）→ mean −0.081。**安全门：worst −0.271
  最不安全臂**（激进 scope 致更大 worst-group harm）。det≻random 复现 +0.100 CI[+0.042,+0.164]。
- **裁定级结论（待用户确认）**：**封闭、可机械验证的 TS cell-scoped 程序空间中，显式搜索比语言推理更承重**——
  连 Opus（示 menu、填满预算）也显著败于 det，机制=选择过泛化。**LLM 于 PatternSpec/开放算子发明/NL 失败反思
  仍未被否定**。停在此等人工裁定；**不据结果改 prompt 重跑、不加 no-memory、不扩 8 族**。

---
*依据：Component Plan §13.2/§13.4；用户第三十七轮（同空间同预算 readiness bar）+ 第三十八轮压缩计划
（押 B1b/开放程序/单模型/省额度评估/全 LOFO）；先例 SkillSliceV2（族盲压缩杀 LLM）、classify 阶段 B
（proposer 端到端先例）、E-3.3/F0（winner's curse）。*
