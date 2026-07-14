# P6 VERDICT — 证据驱动 harness 进化（H0→H1→H2）

状态：**FINAL**（2026-07-12 封笔。修订史：DRAFT-1 §3"U 不开箱"被外审推翻→U 完整事件史；DRAFT-2 收 U abort＋E2 replay；DRAFT-3 收 prereg:183 sensitivity；FINAL＝codex 终审 5 项窄域修订全部采纳——①formal 调用计数 4→5 ②LSTM 验证降为推导表述 ③E2 定性为描述性补齐非 §6 兑现 ④"穷尽假说"收窄 ⑤"点火"术语纠正——**均为措辞/计数修正，零数字、零判决变更**。本文档此后任何修改仅经 erratum 通道；文档 SHA 记录于外部（memory＋封笔消息）。）
预注册：`results/Stage2/prereg_p6.md`（frozen f2d13c3e… → post-A1 a5c0c23b…）；冻结记录：`results/Stage2/P6Freeze/freeze_record.json`（25 sha pin + amendments[A1]）

---

## 0. 判决

**Claim branch = B-null："primary criterion not met"。promotions = 0。**

- Cycle 1（D1 开箱）：terminal = **abstain(no_signature)**，V1 未开。
- Cycle 2（D2 开箱）：terminal = **abstain(no_signature)**，V2 未开。
- H_final ≡ H0（state sha `4e7e4ac5b40c941d` 两个 cycle 全程未变，state_changed=False×2）。
- cycle 预算（cycle∈{1,2}）用尽；两 terminal 已入台账，hash 链连续无分叉。
- U 终评：正式运行 = **P6TechnicalAbort**（基础设施故障，U 为非门，不进任何 claim 分支，§3）——**正式 §6 U 终评未完成**；erratum E2 以 out-of-band descriptive replay 补齐描述性披露（闭式/Adam 直接 paired 证实全零，LSTM 由两臂 prepared artifact bit-identical＋单臂确定性复刻推导恒零，§2），不构成 protocol-compliant U result。

B-null 是预注册合法分支（prereg §0），由 promotions=0 单独确定，与 U 结局无关。执行链完整性见 §1，U 事件史见 §3，科学结论见 §5。

## 1. 执行链完整性（停止线事件两起：C0 FAIL、U TechnicalAbort——均按协议停机并经裁决；五类红线零违反）

| 环节 | 事实 | 锚 |
|---|---|---|
| 签发 | prereg FROZEN + 25 文件 sha pin，主会话 25/25 MATCH 复核 | freeze_record.json |
| C0 首跑 | identity gate **FAIL**（①fred_md，U_cf 优于 Adam 12.9%）→ P6TechnicalStop，原始 FAIL 产物一字未改保留 | C0Run/（sha edaedbc7…） |
| 诊断 | D1–D10 系统检验并排除当时识别的主要替代解释（欠收敛/正则/加权/窗协议）→ 剩余差异归于估计器本体 | C0Run/diag/ |
| Amendment A1 | 判据①双侧 equivalence→一侧 non-inferiority；时点=任何 D/V/U 效用读取之前（occurred_before_any_DVU_utility_read=True）；codex 窄审两条必补逐字满足后 GO | prereg §11；freeze_record amendments |
| C0 重跑 | **PASS**（A1 口径四判据全过）；ε=0.015879655776961057、δ_safe=0.039699139442402644、J_raw,C0=0.7939827888480528 | C0Run_A1/（sha 2240635a…） |
| V/U 物化 | 60 certified_virgin（V1/V2/U 各 20）；block digest V1=d726f8db… V2=0a1a4ea7… U=a16def03…；virgin∩legacy=0；legacy ledger（30f9fdcc…）一字未改 | P6Materialization/ |
| Cycle 1 | run_cycle_formal，abstain；V1 四 npz 原封、无 sealed_V1/、无 precommit | Cycle1/（digest 0e4fb965…） |
| Cycle 2 | run_cycle_formal，abstain；V2 四 npz 原封、无 sealed_V2/、无 precommit；D2∩D1=∅、D2∩V2=∅ | Cycle2/（digest 68ecfa25…） |
| U 开箱 | run_u_eval_formal 全冻结默认参；open(U) seq3 提交（bindings：entrypoint／frozen_literals_digest f6227a0c…／materialization_sha 6eb86d7b…）；闭式 2/2＋Adam 6/6 fit 完成后 **LSTM[h0/seed=1] 1189.9s > 冻结 900s 软超时 → P6TechnicalAbort**（早于任何 diff 比较；已算效用随异常丢失、无人观测）；final_packet 未产出；V1/V2 sealed 复核过 | UEval/（error 持久化） |
| U replay | erratum E2 授权 descriptive replay（out-of-band）：12 fit，闭式/Adam paired diff 全零＋LSTM 推导恒零，level 表产出；gate ledger 零写入（前后 sha 不变，仍 3 事件） | UEval/descriptive_replay/ |
| 台账 | genesis fa2ce7a4… → seq1 24f6ca91… → seq2 0298be1a… → seq3 open(U) **e29213816d17ff26610905a102ca38b1adb2309ea4f5f4f93724947d01b11f00**（chain_tip，本文档即外锚；final_packet 永久缺失，其应载信息由本表逐项落锚） | Cycle1/p6_gate_ledger_5c768155….jsonl |

正式入口五次调用（run_c0_formal ×2〔FAIL 首跑＋A1 重跑〕/ run_cycle_formal ×2 / run_u_eval_formal ×1）全部走 formal 路径；崩溃纪律（G2 sidecar resume）零次触发（U 无 sidecar 语义，abort 走停止线裁决，§3）；p6/、tests/、prereg、freeze_record 在两 cycle 与 U 运行期间零改动；U 运行不引用 V1/V2，无 open(V) 事件。

## 2. 预注册完整结果表（prereg §6 披露义务）

**签名证据表**（ε=0.015880、δ_safe=0.039699、B=2000、seed=20260711+cycle）：

| 量 | Cycle 1（D1，128 ep） | Cycle 2（D2，128 ep） | 阈 | fired |
|---|---|---|---|---|
| S1 regret_mean | 0.005100 | 0.007981 | ≥ε | False / False |
| S1 regret_lcb90 | 0.001510 | 0.002175 | >0 | **True / True** |
| S2 mean_effect_classes | 7.4375 | 7.40625 | <2.0 | False / False |
| S2 ceiling_gap | 0.000974 | 0.001959 | ≥ε | False / False |
| S3 worst cohort harm_lcb90 | 0.0（全 11 cohort） | 0.0（全 cohort） | ≤−δ_safe | False / False |
| baseline_utility | 0.655920 | 0.837646 | — | — |
| terminal | abstain(no_signature) | abstain(no_signature) | — | — |

**注（读表义务）**：S1 的 CI 条件在两个 cycle 均满足（lcb90>0——selector regret 统计上非零），但幅度均低于 ε（分别为 ε 的 32% 和 50%）→ 按冻结决策表（S1 = regret≥ε ∧ CI>0 合取）不点火。这不是"未检出"，是**检出了低于预注册实质性阈值的真缺陷并被 ε 正确过滤**。

**反事实池配置**（chosen-set，两 cycle 一致）：det_only{8,0,0} / det_random{3,5,0} / incumbent{3,5,0} → n_distinct_programs 全=1——在此分布上池构成不改变选择结局。

**成本账**：

| 项 | Cycle 1 | Cycle 2 |
|---|---|---|
| probe | 16 | 16 |
| replacement_effect_calls | 2176 | 2176 |
| surrogate_evals | 13184 | 13312 |
| judge_paired_fits | 4 | 4 |
| adam_cogate_fits | 0 | 0 |
| llm_requests | 0 | 0 |
| discovery_rounds | 1 | 1 |
| wall-clock | 301.6 s | 1090.5 s |

promotions∈{0,1,2} 披露：0。每次 abstain 的 signature 与原因：两次均 no_signature（上表）。V 门六条、first-unseen、LODO：**N/A**（V1/V2 未开箱，无晋升候选进入 held-out 评估）。全表原始数据：`Cycle{1,2}/cycle{1,2}_deliverable.json` + `attribution_cycle{1,2}_D.json`。

**U 终评披露（正式 §6 U 终评未完成——正式运行 abort，事件史见 §3；下列数字为 erratum E2 out-of-band descriptive replay 补齐的描述性披露，非 protocol-compliant U result）**：

H_final≡H0 → 三口径全效应恒零：**闭式判官与 Adam 为直接 paired 验证**——闭式 80 对 paired diff（train/context/joint）maxabs 全=0.0，Adam 3 seed×80 对 diff maxabs 全=0.0；**LSTM 为推导**——两臂 prepared artifact bit-identical＋同 seed 单臂重复 fit bit 级同值 → 臂间 diff 按构造≡0（单臂简化已自我声明）。disclosure_s6 全格 {gain 0.0, lcb90 0.0, n 20, direction zero}。

**U(traffic_hourly) 绝对水平首读**（20 series×4 preset，H0 臂）：

| preset | 闭式判官 | Adam-DLinear | LSTM-scratch |
|---|---|---|---|
| G_hi_full | 0.370382 | 0.398214 | 0.371433 |
| G_hi_miss | 0.373664 | 0.396014 | 0.365506 |
| G_lo_full | 0.340084 | 0.364944 | 0.413030 |
| G_lo_miss | 0.355222 | 0.378410 | 0.405622 |
| **overall** | **0.359838** | **0.384396** | **0.388898** |

per-seed overall——Adam {s0 0.387108, s1 0.393317, s2 0.372762}；LSTM {s0 0.396961, s1 0.389391, s2 0.380340}。描述性注记（n=20/格，无 CI，禁作主张）：闭式/Adam 在 G_lo 水平更好而 LSTM 相反——preset 难度序呈报告器依赖，与模型条件化主题（C6）同向，仅作后续实验设计线索。success_descriptor（§5 claim 字段）未产出：正式运行 abort 于其计算之前，且按构造不可达（方向恒非正）。明细：`UEval/descriptive_replay/per_episode_closed_form.json`（80 行 loss_00/10/01/11）。

**非裁决 sensitivity（prereg 冻结义务，"结果后报告、不改 verdict"；产物 `prereg183_sensitivity.{json,md}`，代码在 diagnostics/，零决策字段；复刻忠实性对拍：lam=0.001 臂 vs 两 cycle 持久值 bit 级全 match）**：

| 轴 | 结果 |
|---|---|
| ε∈{1%,2%,5%}·J_raw | 唯一点火＝**C2 S1 @ ε=1%**（regret 0.007981 ≥ ε₁% 0.007940，边际 4e-5；lcb90>0 本就成立）；其余全格不火；S2 任何 ε 档均不火（gap 恒正、classes≈7.4） |
| CI95（q025 重分位，同 cluster bootstrap 协议） | S1 regret LCB95：C1 0.001020 / C2 0.001439——**仍全>0**（CI 条件对分位收紧稳健）；S3 harm≡0，任意分位退化为 0 |
| λ=1e-3→1e-2（闭式判官重解） | regret/gap 变动 ~1e-7 量级（C1 0.00510008→0.00510001；C2 0.00798073→0.00798062），符号与 S1/S2 firing verdict 全保持——测得的 regret 非判官正则化伪影 |

读表义务：ε=1% 档的点火是 post-hoc 反事实，不改任何 verdict，禁作行动依据；边际量 4e-5 说明这是刀锋而非明确超越。三轴合并的边界刻画：**S1 headroom 位于 ~1% practical scale 边界——统计上真实（LCB95>0）、实质上边际（仅 ε 减半时以 4e-5 过线）、对判官正则化 10× 不敏感**。

**结构性预测判定（prereg §0，预注册可证伪项）**："D1 主导 signature = S1" —— **证伪**。D1 上无任何签名过线（S1 幅度不足 ε）。预测的第二半句（H1 修复后 D2 主导≠S1）因前件失败不可评估。如实计入：预注册作者（含本会话）对 H0 缺陷幅度的先验预期过高。

## 3. U 终评事件史与裁决（E1 → 外审改判 → TechnicalAbort → erratum E2）

**最终状态：U 已开箱消费；正式结局＝P6TechnicalAbort（非门，claim 零影响）；正式 §6 U 终评未完成，E2 仅补齐 out-of-band 描述性披露；final_packet 永久缺失。**

**(1) E1：不开箱裁决（已被推翻，留痕）**。DRAFT-1 曾裁 U 不开箱，理由＝估计量退化（H_final≡H0 → 配对差按构造恒零）＋可逆性不对称。外部深评挑战后主会话改判，三个 defeater：①E1 核心论证本身 outcome-conditioned（"简并"依赖双 abstain 这一结局）→不开箱无论如何须记 protocol deviation，不可能是干净豁免；②§6 原文"无论结局，必须披露…+ U；Adam 与 LSTM 数字"——Adam/LSTM 的 level 数字全实验从未产出（adam_cogate_fits=0、LSTM 从未运行），恒等论证只能给 diff=0，给不了 level 表；③U levels＝unseen 域首读非零信息，且"留 U 给未来工作"构成 outcome-dependent holdout management，U 宇宙剩余充足（806−20）使保留价值≈0。

**(2) 正式开箱 → P6TechnicalAbort**。run_u_eval_formal 全冻结默认参派发；open(U) seq3 提交（chain_tip 0298be1a→e2921381）。闭式 2/2＋Adam 6/6 fit 完成后，LSTM[h0/seed=1] 单 fit 1189.9s 超冻结 900s per-fit 软超时（fit 跑完后才查墙钟）→ abort，早于任何 diff 比较；已算的闭式/Adam 效用数值随异常丢失，**无人观测过任何 U 效用读数**。V1/V2 sealed 复核通过。

**(3) 根因（经 replay 证据修订）**：非硬结构不可行——replay 中全部 LSTM fit <900s（349–884s），且 seed1 与其 bit 级重复 fit 墙钟 487.56s vs 862.35s（**同一计算墙钟波动 1.8×**）→ 正式 run 的 1189.9s 是高方差、从未负载测试过的路径在零余量冻结 timeout 下的边界超限。timeout=900 按 C0 Adam 尺度（p50 2.7s）定值，从未对 LSTM×len-1024 试跑＝"声明≠执行"家族第 6 例（流程规则见 §4-f）。派发任务书（主会话）未对该从未执行过的路径要求可行性探针——责任在裁决方，如实记账。

**(4) 无合法正式恢复**：①同参重调用撞一次性语义（can_open("U")=False → P6StateError，U 无 sidecar resume）；②调 timeout 的 amendment 被"任何修改永不得发生在 V/U 效用被读之后"挡死（闭式 fit 已算即为已读，按字面执行，即使数值无人见过）。正式结局定格。

**(5) Erratum E2：descriptive replay（out-of-band）**。授权依据：ⓐ**决策惰性＝权限论证（主论据）**——replay 的任何数值都不具决策权限：不能改变正式 verdict（B-null 由 promotions=0 独立确定）、不能产生正式 success_descriptor、不能触达转移限定词（方向恒非正按构造不可达）；"效用数值无人见过故无 outcome-dependence"为辅助事实。其功能仅为补齐 out-of-band 描述性披露，不追溯修复正式运行；ⓑ先例＝C0 FAIL 后 D1–D10 诊断（已开箱数据＋代码不入 p6/＋不写 gate ledger）；ⓒ闭式/Adam 为同 seed 确定性计算，replay＝重放被 abort handler 丢弃的同一计算。执行边界经主会话验收：gate ledger 前后 sha 不变（仍 3 事件）；冻结字面量全同，唯一偏离＝解除 per-fit timeout（自我声明）；LSTM 单臂简化（自我声明）；产物全部标注 `descriptive_out_of_band_erratum_E2`。结果入 §2。

**(6) 连带披露**：①`final_packet` 永久缺失（唯一合法 writer abort 且不可重跑），其应载信息由 §1 表逐项手工落锚；②virgin ledger 的 20 条 consumed_virgin(U) 为 agent 在 abort 后补记（正式 deliverable 内 consumption_added=0；补记 tied to open event e2921381、时间戳晚约 4 分钟）——已披露的账面修复，消费事实本身由 open 事件＋80 episode 效用计算成立；③§0 转移限定词不 license：success_descriptor 未经正式计算，且按构造不可达。

## 4. 措辞与披露义务（论文/后续引用时强制执行）

- **(a) 主效应 estimand**：全部效应量属于 frozen ridge-DLinear training objective（`dlinear_closed_form_v1`：λ=1e-3、L_WIN=48、H=48、series-equal 加权）。"attribution-exact" 的展开义务 = "exact for the frozen leave-one-series ridge replacement estimand"，不得暗示一般因果归因。
- **(b) A1 语义收窄**：C0 identity gate ① = raw-level **non-inferiority**（单侧），不确立绝对水平等价；②③′④ 支持 decision concordance，不支持幅度等价；晋升门③（Adam co-gate）只验 Adam-regime non-harm，不验证幅度迁移（本轮 adam_cogate_fits=0，该门从未被触达）。**每域带符号 level offset（C0_FREEZE A1）：covid −0.72% / fred −12.89% / nn5 −3.33% / tourism +0.03%**；D4 证实 offset 程序相关非均匀（range [−0.02975, +0.00571]），**禁止任何"差分抵消 offset"论证**。
- **(c) LLM 供给零证据**：llm_supplier=None（开跑前 outcome-independent 裁决，入两份 run provenance）。P6 对 LLM 供给价值**提供零证据**（llm_requests=0×2）；P5 的 LLM 负判决仍是现行结论；LLM rehab 退出本实验 scope。
- **(d) 编辑空间收窄**：p6 冻结面无 struct_feats 计算器（toy_fingerprint 仅 snr+missing）→ RiskRule scope 的 struct_feats 轴本轮实际不可达（limitations）；sampler_b 真 supplier 可评估性张力（2 slot×128 ep=256 调用≫60 预算）为冻结面自身张力，入 backlog——未来 LLM rehab 须冻结 supplier 契约+缓存/子采样评估+独立 prereg。
- **(e) 结构性预测证伪**（§2 末段）必须与 B-null 一并如实报告。
- **(f) U 数字引用纪律＋流程教训**：U 的一切数字为 out-of-band descriptive（erratum E2），正式 U 结局＝P6TechnicalAbort——引用任一方必须并报另一方；不得将 replay 数字当作正式终评产物。流程教训（约束后续 prereg）："声明≠执行"第 6 例＝冻结面包含从未执行过的路径（LSTM reporter 及其 timeout）；**新规则：capability matrix 增列"已在 toy 尺度执行 ≥1 次"，未执行过的路径不得进冻结面；冻结 timeout 一律按实测 p95×安全系数定值**（本例同 fit 墙钟波动实测 1.8×）。**固定引用模板（引用 U 时唯一合法表述，不得省略前半句）**："The formal U evaluation technically aborted; an out-of-band deterministic replay descriptively confirmed zero arm differences under the identical H0/H_final states."

## 5. 科学解读：B-null 说明什么、不说明什么

**说明**：
1. **机器完备性与结果的分离**。进化机器的全链路能力已由构造性证据确立（toy 合成场景：病态 selector→S1 点火→miner 修复→六门晋升 promote / 纯噪声→门① reject；C0 identity gate 全过）；在真实分布上，**配置良好的 H0（det 阶梯 3 + random 供给 5、proxy_rank、K=8）在 ε 实质性尺度上无可开采缺陷**——两轮均检出统计显著但低于 ε 的 selector regret 并正确 abstain（无任何 signature 点火，activated=null×2），两次保住 V 封存，是"没有证据就不编辑"这一设计性质的一发实证。
2. **与 P5 合流的双边界结果**：P5 判定 LLM 供给不优于 random（瓶颈在生成）；P6 判定该供给配比下的 harness 无可测进化 headroom。两侧夹出同一结论——**det+random 基线在此判官口径与语料分布上已近充分**，"自进化"叙事在此设定下无立足空间，这是可发表的边界刻画而非失败。
3. **ε 校准的行为学验证**：两个 cycle 检出统计显著但低于 ε 的 selector regret 并正确弃权——实质性阈值不是摆设，机器不追逐噪声级改进。sensitivity 三轴（§2 末）将该边界定量化：结论不是"零 headroom"，而是 **headroom 低于 2% 实质性阈值、量级停在 ~1% practical scale 的刀锋上**（ε=1% 档 C2 以 4e-5 边际过线、LCB95>0、λ×10 不变）。

**不说明**：
- 不说明进化机制在更弱 H0、更异质语料、或更大 ε headroom 分布下无效（本轮 H0 被刻意配置为强基线）；
- 不说明 LLM 供给无价值（零证据，见 §4-c）；
- 不说明 harness 各坐标（Selector/Sampler/RiskRule）的编辑上界——miner 从未被激活，编辑空间未被探索；
- U 域（traffic_hourly，频率新颖）的**进化转移性**未测——不是因为封存（U 已开箱），而是因为无进化可转移（H_final≡H0，闭式/Adam paired 证实＋LSTM 推导效应恒零）；该域仅获得 H0 绝对水平的描述性首读（§2），任何转移主张仍无证据。

## 6. 工件清单（复核入口）

```
results/Stage2/
  prereg_p6.md                    a5c0c23b…（含 §11 A1）
  P6Freeze/freeze_record.json     25 sha pin + amendments[A1]
  C0Run/                          原始 FAIL（保留）＋ diag/D1–D10
  C0Run_A1/                       C0_FREEZE(A1) 2240635a…
  P6Materialization/              sealed/{V1,V2,U}/ 9 npz＋virgin ledger 693f45b3…
  Cycle1/                         deliverable＋provenance＋consumed manifest（c185ad7d…）＋台账
  Cycle2/                         deliverable＋provenance＋consumed manifest（37ff1d3b…）
  UEval/                          run_provenance＋u_eval_deliverable（正式 abort，error 持久化）
  UEval/descriptive_replay/       replay_report＋per_episode_closed_form（erratum E2，out-of-band）
  prereg183_sensitivity.{json,md} 非裁决 sensitivity（post-hoc，零决策字段）
  P6_VERDICT.md                   本文档（chain_tip e2921381… 外锚，见 §1）
```

签名：主会话（brain），2026-07-12。FINAL 化条件见页首。
