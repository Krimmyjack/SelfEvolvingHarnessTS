# prereg_p6 — Evidence-Driven Harness Evolution（P6 执行冻结协议）

> **状态：FROZEN（2026-07-12 签发，不可逆点 #1；原 DRAFT-2 经四轮外审 GO——详见文末签发块 §10）**。
> 转 FROZEN 的签发门见 §8；签发由 `results/Stage2/P6Freeze/freeze_record.json`（含本文件与全部
> 承重代码/manifest 的 sha256）完成。签发前 C0/D/V/U 一律不得开启。
> 设计依据 `idea/P6_Plan.md`（其 PENDING/数值与本文冲突处，以本文为准；P6_Plan 附 supersession 表）。
> **偏离政策**：签发后任何偏离以 erratum 追加，不得静默修改。
> **technical abort ≠ 科学负结果**：下载不足/hash 漂移/NaN/CUDA 非确定/日志损坏/judge 自检失败
> → `P6-technical-abort`，不计入任何 claim 分支；修复后可按同一协议重新签发新 freeze。

## 0. 范围、口径与 claim 分支

**问题**：固定 task=forecast（单变量）+ 冻结判官协议下，系统能否从 batch downstream feedback
自主提出并晋升 coordinate harness edit（H0→H1→H2），每次通过 held-out 晋升门。

**统一效应口径（全项目唯一合法词汇表；runner 须经 `p6/metrics.py` 的冻结函数产出）**：
- `loss` = 判官 RMSE（越小越好）；`gain(H→e) = loss_H − loss_e`（正=改善）；
- `regret = loss_chosen − min(loss_pool)`（≥0）；`harm = −gain`；
- 判官模块的 `batch_delta = loss_new − loss_old = −gain`（文档/JSON 不再混用 raw delta）。

**claim 分支（冻结措辞）**：
- **B-strong**："two sequential held-out promotions affecting two pre-specified live harness
  surfaces under a frozen ridge-DLinear training-data objective"；
- **B-weak**："two sequential held-out promotions within one pre-specified surface: `<实际 surface 名>`"；
- **B-partial**：恰好 1 次晋升——主成功判据未达，单次晋升单独如实报告；
- **B-null**：0 次晋升——"primary criterion not met"。
- **U 转移限定词**：仅当 U 满足 non-harm 且方向为正，才可追加
  "directional transfer to the pre-admitted traffic_hourly domain"；U 有害时任何分支不得用
  不加限定的 headline。
- **"attribution-exact" 的展开义务**：论文中必须写明 = "exact for the frozen
  leave-one-series ridge replacement estimand"，不得暗示一般因果归因。

**结构性预测（预注册、可证伪、非判据）**：D1 主导 signature = S1；若 H1 修复之，D2 主导 ≠ S1。

**不做**：任务扩充；模型选择进 harness；classification 精确归因；内在 pattern 迁移；
seeds 20–99 复用；legacy 衍生工件（含 `frozen_lstm_real_h64.pt`）进入任何角色；
GrammarMacroPatch（本轮无此 surface，已从配方与 claim 计数中删除）。

## 1. 判官协议（冻结）

**晋升判官 = `dlinear_closed_form_v1`**（`p6/judge_closed_form.py`）：λ=1e-3、stride=4、
window_cap=None、series_weight="equal"、截距不受罚；**per-domain shared** 拟合（域内全部
episode 的窗口 pooled）；评估 = per-episode loss → batch = episode 等权均值；每次正式拟合
双路对拍自检，**判据（冻结；依 U 全宇宙复检实测 7072 窗尺度下 W 级浮点累积 ~1.6e-8）**：
per-episode loss 与 batch utility 的 max|Δ| ≤ 1e-9（主判据）∧ W 的相对差
max|ΔW|/max|W| ≤ 1e-6（辅助）；任一超限 → technical abort。

**主判据口径 = train_gain**（train_effect 的 gain 形式）；**joint/context 全量披露**，且
joint 进入晋升安全门（§4 门⑥）。**Adam co-gate 评估 joint 臂**（部署真实口径）。

**角色分离**：proxy = seasonal-naive 内窗 proxy（P3 语义、escalate-only）；V co-gate =
Adam-DLinear（epochs=120、lr=1e-2、bs=256；**CPU、每次拟合前 `seed_all(seed)` 重播种、
torch deterministic 设置冻结**；paired seeds {0,1,2}）；U reporter roster **签发时固定 =
闭式判官 + Adam(3 seeds) + LSTM-scratch(3 seeds)，本轮不跑 Chronos**；LSTM 禁参与任何选择/否决。

## 2. 数据（冻结；两级不可变工件）

**episode** = 底层序列 × 4 preset（load_real 语义；corruption seed =
sha256("p6deg|{config}|{item}|{preset}")[:8]）；同 series 的 4 preset 同 block。
**3 条单条域仅定性展示：不进入 identity gate、ε/δ、signature、CI、claim 的任何计算。**

**两级 manifest**：
1. **selection_rule_manifest（签发时冻结）**：source=HF `monash_tsf`@PARQUET_REV、config
   inventory digest、eligibility（len≥144、有限值）、hash 选取规则、quota
   （C0 16=4/大域；D1 32=8/域；D2 32=8/域；V1=V2=20=5/大域；U=20 traffic_hourly）、
   曝光规则（legacy 83/83 confirmed_exposed 只进 C0/D；V/U = **project-local certified
   virgin**——只证明未进入本项目选择/实验链）；
2. **sealed_materialization_manifest（C0 阶段生成、链回规则 manifest sha）**：确切 uid、
   content_sha、length、finite check。候选不足/下载失败/hash 漂移/长度不足/重复 uid →
   **technical abort**（不得换 config、降 quota、人工补条）。
3. 后续所有 open 事件绑定 materialization sha。

**U 候选宇宙**：traffic_hourly **全量 862 条** − 全部探针消费 uid = **56 条排除**
（首轮 24 + 全宇宙复检 capability 子样本 32，零重叠；清单与 content-sha 见
`P6Probes/u_admission_v2_traffic_hourly.json:all_probe_consumed_item_ids`）→ 抽取宇宙 806 条。
**复检已完成（2026-07-11）**：全宇宙准入维持 PASS_HEADLINE_U（主导周期 ≤48 占比 0.9954、
168 主导 0%）；canonical 判官口径（history-only z-score）下 judge/sn24=0.847（胜率 0.66）、
judge/sn168=0.884（胜率 0.56）；确定性 PASS。

**运行纪律**：每次运行落盘 uid 级消费 manifest；pytest/runner 一律 `--basetemp`；
环境 = conda `project`；正式 runner 全部 CPU（判官闭式本就无设备依赖）。

## 3. C0 校准（算法冻结；数值以 C0_FREEZE 记录追加）

C0 只用 C0 块。**C0 identity gate FAIL → technical stop：判官身份未确立，D/V/U 不开启，
本 freeze 作废**（不设 Adam fallback 分支——该分支无法机械承载 S1/S3 精确归因与三配方）。

**3.1 identity gate**（vs Adam-DLinear；预算 = 8 程序（raw_identity 已在清单内，不另计）×
4 域 × 3 seeds = **96 次 Adam 拟合**，单 fit 墙钟软超时 15 min = **900.0 s**（正式入口断言此值），
超时 → technical abort；
Windows 无 SIGALRM，超时为事后判定）。**判官摄入填补（冻结）**：非有限值（raw_identity 在
miss preset 上的 NaN 等）在判官摄入前线性插补（首尾钳制），**统一作用于全部程序臂**（否则
J_raw 在 miss preset 不可计算；先例 = report_target 的 raw-with-missing 处理）；全非有限 →
technical abort。Adam 参照无 series 等权（train_forecaster 无权重接口；等权是闭式 ridge
estimand 属性），但评估协议与判官逐字一致以保 ① 可比：
- P_gate（8 程序，冻结）：raw_identity / impute_linear / il+winsorize / il+winsorize+savgol /
  il+median_w9 / il+median_w15 / il+median_w25 / il+smooth_ma_w5；
- 定义：程序效应 = per-domain batch gain vs raw；near-zero 排除带 = |gain| < ε/4 的程序对
  不计入符号一致率分母；top-1 一致 = Adam top-1 ∈ 闭式 top-2 且 loss 差 ≤ ε/2；
  Adam 任一拟合 NaN/发散 → technical abort；
- **PASS = ① ∧ ② ∧ ④ ∧ ③′**：
  ① |U_cf(raw) − mean_seed U_adam(raw)| ≤ 0.10·mean_seed U_adam(raw)（per-domain 全过）；
     **→ 见 §11 Amendment A1**（本判据①已由双侧 equivalence 就地改为一侧 non-inferiority
     U_cf ≤ 1.10·U_adam；原文字保留于此不动，语义修订以 §11 为准）。
  ② per-domain Spearman ρ（8 程序 gain）的 4 域中位数 ≥ 0.7；
  ③′ ε-tolerant top-1 一致率 ≥ 0.6（episode 级）；
  ④ preset 级 gain 符号一致率 ≥ 0.75（near-zero 带外）。

**3.2 阈值**：`J_raw,C0` = C0 全 **64** raw-退化 episode（16 series × 4 preset；单条域
不计）的判官 batch loss（域等权再合并）；
ε = 0.02·J_raw,C0；δ_safe = 2.5·ε。ε 是 practical-effect margin（确定性判官下配对 null
精确为零）；winner's curse 由结构防御（V 全新、一发、候选预算、precommit）承担。

**3.3 P0 cohort 清单（S3 用，冻结）**：{4 preset} ∪ {8 个 P0 粗 bin} =
snr_est 与 missing_rate 各按 **C0 分位数四分**（cutpoints 在 C0_FREEZE 里落盘）。
特征 allowlist（RiskRule scope 同用）= fingerprint 的 {snr, missing_rate, struct_feats 之
P_FEATS}；**禁止** outcome/judge response/series id/domain id 类特征。
signature 扫描为有限清单（12 cohort）上的 worst-case；激活≠主张，晋升门保护有效性
（此 rationale 预注册）。

## 4. H0、signature 与 cycle 协议（冻结）

**H0**（代码字面量，非均分公式）：selector=proxy_rank；sampler=**{det:3, random:5, llm:0}**；
risk_rules=[]。**K=8 是请求 slot 预算**：det 阶梯固定 3 程序；跨 supplier sha 去重后实际
unique pool 可 <8，**不回填**，realized pool size 逐 episode 落账；某 supplier 产出不足即
如实短缺（预算是上限不是配额承诺）。

**failure signature（判官尺度，D 块，`p6/metrics.py` 冻结实现）**：
- 行为等价类：两候选在一 episode 上等价 ⟺ |loss 差| ≤ 1e-9（union-find 成类）；
- **S1**：batch 均值 regret ≥ ε 且 series 聚类 CI90 下界 > 0；
- **S2**：episode 级等价类数的全 episode 均值 < 2.0，**或** 池上限 gain（池内 per-episode
  min-loss 构成的 batch）低于 det 阶梯同口径 − ε；
- **S3**：§3.3 清单内某 cohort（≥5 底层 series）train_harm 的 LCB90 > δ_safe；
- **归一化 headroom = (observed − threshold)/threshold**；激活 headroom 最大的一族；
  并列按 S1>S3>S2；全部不过线 → **abstain**。

**miner = 冻结代码 `p6/miner.py`**（全部常量在码内；无自然语言自由度）：
- SelectorPatch：(a) cross-fit（uid 2-fold，fold = int(sha256(uid),16)%2）ridge（α=1.0，
  特征 z-standardize，标准化常量取自 full-D）回归 特征→train_gain，部署权重 = full-D refit；
  (b) 同 (a) 目标为 episode 内平均秩（ties 取均秩）；(c) **坐标级保守收缩**：
  σ̂_j = |w_j^{fold0} − w_j^{fold1}|/√2，w_c[j] = sign(w_full[j])·max(0, |w_full[j]| − σ̂_j)
  （逐坐标向零软阈值；此为 prereg 首稿"LCB 打分"的唯一冻结释义）。
  候选并列 tie-break = canonical edit sha 升序；
- SamplerPatch：(a) det−1/random+1；(b) random−2/llm+2（**唯一 LLM 解锁**：llm call =
  1 次 HTTP request/episode（一次返回该 episode 全部 llm slot 候选），≤60 requests/cycle）；
  (c) det−2/random+2；
- RiskRulePatch：(a) ban 程序 sha @ 触发 cohort 的 allowlist-P0 scope（scope = 该 cohort
  的定义条件本身：preset 成员或 P0 bin 区间）；(b) ban 算子族@同 scope（算子族映射表冻结在
  miner 码内：denoiser/outlier/imputer 三族）；(c) **同 (a) 但 scope 加一个合取条件**：按
  allowlist 冻结次序取第一个满足"触发 cohort ≥80% episode 落入其 C0 四分位单一 bin"的特征
  （≥80% 以整数形 5·count≥4·n 判定），加该 bin 为第二条件；无此特征 → 配方 (c) 不可用。
  **when 语义（冻结）**：`RiskRuleSpec.when` 原生支持条件列表 = **同规则内 AND**（bin scope
  的 lo/hi 两原子必须同规则合取；禁止拆多规则近似——`apply_risk` 逐规则独立求值会把合取
  退化为并集 ban，bin scope 在并集语义下等价全局 ban）。

**cycle t ∈ {1,2}**：
1. D_t 上以 H_t 跑 fast path（uid 消费 manifest）；
2. 归因（probe ≤16 行为等价类变体）→ signature → 激活一族或 abstain；
3. miner 产 ≤3 候选 edit → D 内 ≤12 次完整配置 paired 重评选优；内部门 train_gain ≥ ε，
   否则 abstain；
4. **precommit**：开 V 前落盘 {candidate edit sha, H-state sha, 完整配置, materialization
   sha, code sha}；
5. **V_t 一发**（次序钉死：precommit → `gate.open_block(V_t)` → **再**加载 V 数据 =
   **先 open 后读取**；一次性 open 是 V 隔离屏障，必须先于任何 V episode 读取），晋升门（全过才 promote）：
   ① train_gain ≥ ε 且 series 聚类 bootstrap CI90 下界 > 0；
   ② preset worst-group train_gain LCB ≥ −δ_safe；
   ③ Adam co-gate（joint 臂，3 paired seeds）均值 gain ≥ −ε；
   ④ scope 类 edit：作用域外 episode 的 **prepared artifact 字节级不变**（end-to-end 校验）；
   ⑤ 台账干净（precommit 在先、V 首开、hash 链完整）；
   ⑥ **joint 安全门**：overall joint_gain LCB ≥ −δ_safe 且 preset worst-group joint_gain
   LCB ≥ −δ_safe；
6. **cycle terminal 事件（不可逆、必记）**：`cycle{t}_closed ∈ {promote, reject, abstain}`；
   V2 开启需 cycle1 terminal + D2 precommit；**U 开启需 cycle2 terminal**（abstain 也是
   terminal——修复 abstain→U 死锁）；
7. **V 隔离**：V_t 详细指标写入 sealed 目录，H_final 冻结前任何 miner/人工不得读取；
   cycle t+1 的输入只有新 H state + terminal verdict；V 拒绝后不得以 bugfix 名义重试
   （预定义 technical-failure 分支除外，且该分支不暴露任何效用数字）。

**bootstrap（冻结）**：按 domain 分层、按底层 series_uid 聚类（4 preset 同抽）、B=2000、
PRNG=numpy default_rng(20260711+cycle)（**U 终评 seed=20260714**）、quantile linear 插值、
**LCB = bootstrap 分布 5% 分位（双侧 90% CI 下端点）**；episode 拟合失败 → technical
abort；exact-tie 按 ≤ 处理。
非裁决 sensitivity（结果后报告、不改 verdict）：ε∈{1%,2%,5%}、CI95、λ∈{1e-3,1e-2}。

**台账（`p6/split_manifest.py` v2，冻结）**：canonical ledger 唯一路径（由 manifest sha 派生）+
独占文件锁（msvcrt 真互斥）+ write-ahead（事件 fsync 落盘后才改内存）+ 事件 hash 链
（genesis 绑 manifest sha）+ **verdict 白名单分层：块级 {promote,reject}（abstain=不开箱，
只存在于 cycle terminal 层 {promote,reject,abstain}，且 terminal 须与已记录块 verdict 一致）** +
每事件绑定 precommit 五元组 {candidate sha, state sha, config digest, materialization sha,
code sha}；崩溃恢复只允许按 precommit 同候选、同字节、同 seeds 幂等续跑（pending_open→resume，
禁重选）。**hash 链固有极限**：末条事件的自洽篡改裸链不可捕获——**运行结束的结果包必须外锚
ledger chain_tip**（写入最终 VERDICT 与 freeze 追加记录）。

**预算/cycle**：discovery 轮 ≤2；edit 候选 ≤6（≤3/族）；probe 变体 ≤16；内部重评 ≤12；
LLM ≤60 HTTP requests（仅 Sampler(b)）；Adam co-gate = 4 域×2 臂×3 seeds=24 拟合；
supplier counterfactual chosen-set ≤3 配置（只在 D 或 H_final 冻结后）。

## 5. U 终评（cycle2 terminal 后一次性）

H0 vs H_final paired 于 U（20 virgin×4 preset）：闭式判官 + Adam(3 seeds) + LSTM-scratch
(3 seeds，from-scratch)。train/context/joint 全披露。**非门**；成功描述子 =
train_gain LCB ≥ −δ_safe（无害）且方向为正 → 允许 §0 的转移限定词。
**一次性 open（次序钉死）**：U runner 在读取任何 U episode 数据之前，自己执行
`gate.open_block("U", bindings)`（bindings 必含 materialization sha；空 bindings 拒绝）——
一次性由台账保证（二次 open 被拒 → 无法重复窥视）；该 open 事件即 U 的不可逆记账。

## 6. 预注册完整结果表

无论结局，必须披露：train/context/joint 三口径 × overall + 4 preset + 4 seen 域方向 +
U；Adam 与 LSTM 数字；promotions∈{0,1,2} 与每次 abstain/reject 的 signature 与原因；
realized pool size 分布；全部成本（拟合数/LLM requests/墙钟）；LODO 仅描述性。

## 7. 冻结清单（签发时 sha256 入 freeze_record.json）

`p6/{judge_closed_form,split_manifest,harness_state,edit_surfaces,fast_path,metrics,miner,
loaders,final_packet}.py`、C0/cycle/V/U runner 与 virgin materializer、`tests/test_p6_*.py`、
`idea/P6_Plan.md`（含 supersession 表）、本文件、selection_rule_manifest、`P6Probes/` 全部探针
产物（含 U 全宇宙复检）。

**正式运行唯一合法入口（冻结；G4/finding 36）**：`run_c0_formal` / `run_cycle_formal` /
`run_u_eval_formal`——三者内部机械断言全部本协议冻结字面量（seeds=(0,1,2)、bootstrap_b=2000、
cycle bootstrap seed=20260711+cycle、U seed=20260714、trainer 超时=900.0、K=8、C0=64 episodes/
4 域/每 series 4 preset/96 fits），并把 entrypoint 名 + 冻结字面量 digest 写入输出记录与台账；
`run_*_unfrozen` 仅供机械单测，正式运行禁止调用。`freeze_record.json` 记录本条款并核验最终
结果包（`p6/final_packet.py`）中的 entrypoint = `run_*_formal`。

## 8. 签发门（全部满足才转 FROZEN）

- [ ] runner/miner/signature/bootstrap/materializer 实现完毕、机械单测通过、入 SHA；
- [ ] H0（3/5/0 字面量）、K slot 语义、det 阶梯 3 程序与代码一致；GrammarMacro 已从协议删除；
- [ ] gain/regret/harm 词汇表落地 `p6/metrics.py`，文档/JSON 无混用；
- [ ] C0 FAIL = technical stop 已写死；
- [ ] cycle terminal / abstain→U 修复 / 原子一次性台账 / V 隔离在代码+测试层实现；
- [ ] U 全宇宙（862）复检探针完成且准入维持；
- [ ] joint 安全门与 claim 限定词入文；
- [ ] 两级 manifest 分层落盘；
- [ ] 正式运行唯一合法入口 = run_*_formal（冻结字面量含 timeout=900）、V/U loader 载体化、
      resume 经 sidecar 不重跑 discovery——代码+测试层实现（G 波 finding 32/34/35/36）；
- [ ] 外部 reviewer 复审通过（第三轮）。

## 9. 修订记录（DRAFT 期，签发前合法；签发后改走 erratum）

- **2026-07-12（codex 二轮复审修复波 finding 31-37）**：
  - §4 步骤 5（F1/finding 31）：明确 V 次序 = precommit → open V_t → **再**加载 V 数据
    （先 open 后读取）；原文对 loader/open 先后有歧义，此处钉死。CycleResult 不再携带 sealed
    目录路径，V 详细指标只落 `sealed_V{t}/v_report.json`（路径由 `sealed_v_dir` 纯函数重算）。
  - §5（F5/finding 35）：U runner 在读取任何 U episode 数据之前自己 `gate.open_block("U",
    bindings)`（bindings 必含 materialization sha；空 bindings 拒绝），一次性由台账保证。
- **2026-07-12（codex 三轮复审收口波 G，finding 32/34/35/36 最小再送审条件）**：
  - §7（G1/finding 32）：V loader 返回值收口为 `BoundVEpisodes`（正式）/ `UnboundEpisodes`
    （测试）载体，删除"loader 无 materialization_sha 则向后兼容跳过"分支——无裸序列静默路径。
  - §4 步骤 6/7（G2/finding 34）：precommit 时同步落盘冻结候选 sidecar
    （`precommit_payload_cycle{t}.json`：EditOp 序列 + config + seeds + state sha），其 sidecar_sha
    入 precommit 事件 hash 链；resume 从 sidecar 恢复、**跳过步骤 1-6（不重跑 discovery/LLM/采样）**。
  - §5（G3/finding 35）：`run_u_eval` 改零参延迟 `u_loader`；次序 = open(U, bindings) → u_loader()
    → manifest-bound 验证（uid∈materialization/content_sha 复算/config-preset/实际 sha==绑定值）
    → 评估；删除 episodes_u 预加载参数。
  - §7/§8（G4/finding 36）：正式运行唯一合法入口 = `run_*_formal`（断言含 timeout=900）；
    `run_*_unfrozen` 仅测试用。entrypoint + 冻结字面量 digest 入输出记录/台账/final packet。
  - §4/§5（G5/backlog 41）：`run_u_eval_formal` 评估后外锚 `p6/final_packet.py`（chain_tip +
    freeze SHA 集 + selection/materialization SHA + claim 分支 + U 转移限定词）。

## 10. 签发块（FROZEN；不可逆点 #1）

**签发日期**：2026-07-12。**签发动作**：本协议转 FROZEN，由
`results/Stage2/P6Freeze/freeze_record.json` 记录本文件与全部承重代码/manifest/探针的 sha256。
签发后 C0/D/V/U 方可按本协议由 `run_*_formal` 正式入口开启。

**外审历程（四轮，终判 GO）**：
- 首轮：24 条审查意见，全部处置（见 DRAFT-2）。
- 二轮（NO-GO）：6 条阻断 → **F 波**修复（finding 31-37：V 隔离次序/物化绑定/U v2 接线/
  crash-resume/U 一次性/formal 断言/preset 成员资格 scope）。
- 三轮（NO-GO）：4 条阻断 → **G 波**收口（finding 32/34/35/36：V loader 载体化无裸序列路径/
  sidecar resume 不重跑 discovery/`run_u_eval` 延迟 loader + manifest-bound 验证/正式入口
  不可绕过 + timeout=900 断言）+ backlog 41（final_packet 接线）。
- 四轮：**GO**——codex session `019f518c-4b99-7501-bbee-ee450f69c07d`，其审定的 prereg
  内容 sha256 = `f74440af73abf9c75dd8f562abfbd982cbcba329d6e14bf334e8e3f382ac71a8`
  （= 盖章前本文件 sha；盖章后 frozen_sha 见 freeze_record.json，二者差异仅为本状态行 + 本 §10）。

**主会话验收**：P6 全套机械单测 fresh 跑 **212/212 通过**（`--basetemp` 独立目录）；全库
792 tests 零导入错误。capability matrix 四件（typed proposal / compiler / serving consumer /
paired validator）三 surface（Selector/Sampler/RiskRule）import 级齐备。

**修订协议（签发后）**：本协议与承重代码的唯一合法修改 = technical-abort 类修复，须在
freeze_record.json 增补 amendment 条目，且**永不发生在任何 V/U 效用数字被读取之后**；
其余偏离一律以 erratum 追加，不得静默修改。

## 11. Amendment A1 — 判据①由双侧 equivalence 改为一侧 non-inferiority（2026-07-12）

**发生时点声明**：本 amendment 提出并实施于 **任何 D/V/U 效用数字被读取之前**（V1/V2/U 仍
sealed 未开箱；仅 C0 legacy 数据被消费）。合法性依 §10 修订协议（签发后 technical-class 修复，
留痕、增补 freeze_record amendment）。

**修订定义**：
- **原判据①（保留于 §3.1，文字不动）**：双侧 raw-level equivalence
  `|U_cf(raw) − U_adam(raw)| ≤ 0.10·U_adam(raw)`（per-domain 全过）——意在"模型身份等价门"。
- **A1 后判据①**：一侧 raw-level **non-inferiority** `U_cf(raw) ≤ 1.10·U_adam(raw)`
  （per-domain 全过）——闭式判官相对 Adam 参照**不显著更差**即过；闭式 loss 更低（更优）恒过，
  仅当闭式比 Adam 更差 >10% 才 FAIL。改称 **`raw-level non-inferiority gate` / estimator
  compatibility gate**；10%、②③′④、trainer、ε/δ 规则、seeds 一律不改。
- **语义收窄（必须随之声明）**：C0 **不再确立**闭式判官与 Adam-DLinear 的**绝对水平等价**。
  ②③′④ 仅支持 **decision concordance**（②排序 / ③′ top-1 决策 / ④符号），**不**证明 CF gain
  与 Adam gain **数值幅度**相同。V 门③（Adam co-gate）只提供 **Adam regime 的 non-harm** 检查
  （防"ridge 上获益、Adam 上明显有害"的 edit 晋升），**不**把 ridge 的 gain 数值解释为 Adam gain。
- **P6 主效应 estimand（随 A1 明确）**：主效应属于 **frozen ridge-DLinear training objective**；
  Adam co-gate 只验证**跨估计器无显著反向伤害（non-harm）**，**不**验证效应幅度可迁移。

**不主张（防越证）**：A1 **不**主张 fred_md 的 CF−Adam 偏移是 **program-invariant** 常数
（D4 证否：range=[−0.02975, +0.00571]，非统一水平位移，做差不自然抵消）；A1 **不**主张
Adam magnitude equivalence。

**出处链（诊断证据 D1–D10，只读，不入 claim 分支）**：
- 起因：C0 首跑 identity gate ① fred_md FAIL——U_cf=0.201106、U_adam=0.230858、
  **signed relative offset = −12.887%**（闭式 loss **更低=更优**，非更差）。
- D1 训练曲线欠收敛（末10%斜率仍降）；D2 匹配 L2(1e-3) 仅 0.2309→0.2294（不入 U_cf±10%）；
  D3 延长×2/×4 gap 反变宽（0.0298→0.0347→0.0381）；D4 CF−Adam **非** program-invariant；
  D5 四域仅 fred 越 ±10%（其余|rel|≤3.33%），闭式系统性 ≤ Adam；D6 fred 非最病态（covid 更病态）；
  D7 fred 四序列窗数全等（147）→ 加权 no-op；D8 weighted-Adam 全 96-fit **①仍 FAIL**（bit 级同）；
  D9 窗协议逐项全一致（stride/窗数/L/H/核/zscore）→ 无窗协议混杂；
  **D10（前置门）Adam+L2(1e-3)+×8(960ep)+cosine lr 3 seeds → U_mean=0.200197（vs U_cf 0.201106，
  rel −0.452%，≤±3%）→ 两估计器在共享目标极限一致**，残差 gap = 优化路径效应（非隐藏 bug）。
- 诊断文件：`results/Stage2/C0Run/diag/{D1..D10,fingerprint_check}.json`（sha 见 freeze_record A1）。

**披露义务（VERDICT/appendix 必须给出）**：
1. 每域 signed offset 与 signed relative offset（不止 abs_diff）；
2. D4 非统一偏移（range=[−0.02975,+0.00571]）——不声称做差抵消；
3. limitations：regime transfer 由门③ non-harm 验证，效应幅度不跨估计器可迁移；
4. 保留原始 FAIL（原 C0 freeze/result、原判据①、fred −12.887%、amendment 提出时点、D/V/U 未开、
   A1 前后代码/prereg/freeze SHA、新旧 C0 record 链式引用），**不得用重跑 PASS 覆盖原 FAIL**。

**实现留痕**：`p6/c0_runner.py::evaluate_identity_gate` ① 单处改一侧判定 + 输出字段增
`signed_offset / signed_relative_offset / upper_bound / upper_noninferiority_pass /
criterion_semantics`；测试 `test_gate_criterion1_boundary` 更新 + 新增
`test_gate_criterion1_noninferiority_a1`（构造"判官更差 11%"→仍 FAIL，证门未整体削弱）。
sha 变更全部登记于 `freeze_record.json` 的 `amendments[A1]`。
