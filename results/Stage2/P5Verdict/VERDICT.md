# P5 STAGE VERDICT（2026-07-10；prereg §5/§5.0 冻结协议，confirmatory seeds 40–59 一次性已消耗）

> 三个正式判决 + claim 分支选定。全部验收以 **true 判官**计（P3 binding），gym-proxy 仅搜索信号。
> 证据目录：`P5IdentityGate/`（504 次真实 DeepSeek 调用，墙钟 1173s）、`P5Quadrant/`（真 Monash 48 episodes）、
> `P5Safety/`（confirmatory slice 一次性）。

## P5-A Code-agent identity gate — **claim 分支 = self-updating deterministic with LLM-optional**

**形式判决（ITT，四判据全败，一发子弹已消耗，立此存照）**：

| 判据 | 结果 |
|---|---|
| ① utility：ca_skills − det ≥ ε=0.02 ∧ CI 不跨 0 | **FAIL**：diff = **−0.799**，CI90 [−0.940, −0.651] |
| ② worst-group LCB ≥ −δ_safe=0.05 | **FAIL**：snrHigh −0.897 / snrLow −1.193 |
| ③ 有效新颖编辑 ≥ K=3 | **FAIL**：0 条 |
| ④ 成本披露 | PASS：504 calls，1173.4s，cached flash |

六臂全景（60 episodes，B=3，forecast，true 判官）：

| 臂 | mean Δ | abstain | served-conditional Δ | 备注 |
|---|---|---|---|---|
| frozen | 0.000 | 100% | — | 参照 |
| random_valid | +0.650 | 0% | +0.650 | grammar 空间随机命中率高 |
| det_search | **+0.799** | 0% | +0.799 | dev 冻结 top-3 梯（R2 反稻草人）|
| ca_plain | 0.000 | 100% | — | **180/180 候选 malformed**（无示例=格式塌缩）|
| ca_skills（主臂） | 0.000 | 100% | — | **179/180 死于 pattern_guard_feature_missing** |
| ca_skills_memory | +0.256 | 70% | **+0.852** | 服务时质量 ≥ det（n=18，观察性）|

**机制归因（与形式判决同等重要）**：主臂之败 **不是生成质量之败，是接口可靠性之败**——
(a) `allowed_grammar.guard_features` 向 agent 宣传了 P_FEATS∪{snr,missing_rate} 可作 guard，
而 gym 观测面只提供 missing_rate → LLM 模仿 skill 卡片提交 guarded 程序（合理行为）被
100% 拒绝 = **harness 契约不一致缺陷**（本轮实测抓获，已修 None-值 fail-safe，契约收窄留待
下一轮）；(b) ca_plain 的全量 malformed = 真实格式合规塌缩（无示例卡片时）；(c) served-
conditional +0.852 ≥ det +0.799 说明**生成质量不是瓶颈**。此结论与项目谱系自洽并精化之：
serving-LLM 六层证伪的 L1 合规塌缩在 proposer 席位重现；check13/14 "瓶颈在选择不在生成"
→ 本轮精化为"瓶颈在合规/接口可靠性而非生成质量"。

**后续路径（须新预注册+新种子，本轮不重跑）**：P5-A.2 retrial 前置 = ①guard 契约一致化
（观测面提供 P0 指纹特征，或 allowed_grammar 只宣传可得特征）②格式合规面（few-shot 示例/
schema 校验重试预算）。在此之前，headline 依 prereg 分支：
**"self-updating deterministic TS data-readiness policy with LLM-optional analysis/proposal support"**。

## P5-B Pattern-vs-domain 四象限（真 Monash） — **冻结轴上假设被推翻（诚实负结果）**

| 象限 | mean regret |
|---|---|
| same-domain / same-pattern | 0.142 |
| same-domain / diff-pattern | **0.170** |
| diff-domain / same-pattern | **0.885** |
| diff-domain / diff-pattern | 0.811 |

配对差（dd/sp − sd/dp）= **+0.715**，CI90 [+0.369, +1.090]，n=48——**反向且 CI 不跨零**：
在冻结的"退化结构 pattern 轴"（SNR×missing preset）上，**domain 迁移显著优于 pattern 迁移**。
机制：domain 携带内在结构（period 7 vs 12、季节性、尺度），而本轴 pattern 只有退化几何——
与 Stage 1 自洽（E-1.1：degradation-only cell 路由不够；E-3.2：P0 结构特征才承重）。
**Claim 纪律**：论文不得写"pattern 胜 domain"；该主张降级为"内在结构 pattern 轴 + 更大语料
的新预注册实验"（不做轴购物）。

## P5-C Safety 收口 — **PASS（一次性评估）**

P4 晋升 `bundle_v0.e1`（阈值/规则全 dev 冻结）在 confirmatory slice（seeds 40–59，n=60）：
coverage 33.3%（规则精确落 snrLow）、gain vs v0 **+0.131**、harm rate **3.3%**、worst-cell
LCB **+0.255**（snrLow）/ 0.0（snrHigh 未触碰）≫ −δ_safe、**anomaly 面 bit 级零扰动**。
S2 记录线的 sealed holdout 访问**显式延期**为独立注册访问（非静默跳过）。

## 汇总：P0–P5 证据链下的论文主张阶梯（更新）

1. **Readiness 任务条件化**（P2 动机表 + frozen classify 引用）：成立。
2. **Safety-gated 条件化 serving 有真实信号且可安全晋升**（P4 周期 + P5-C）：成立（substrate 级）。
3. **慢路径证据驱动进化机制**（P4 七环）：机制成立；"self-evolving" 一词仍锁定（P6 需多周期累积）。
4. **Pattern 胜 domain**：在退化轴上**不成立**（P5-B 负结果）；内在结构轴待新实验。
5. **LLM-driven harness evolution**：**未确立**（P5-A 分支 B）；已定位可修复的接口瓶颈与
   retrial 前置条件；LLM 生成质量非瓶颈的观察性证据在案。

---

# P5-A.2 增补（2026-07-10；prereg_p5a2_retrial.md 冻结协议，seeds 60–79 一次性已消耗）

**形式判决：分支 B 维持**——ca_skills − det = **−0.150**，CI90 [−0.242, −0.077]（判据①②仍败），
判据③**首次达标**。549 次 composer 调用 / 1362.9s。

**接口缺陷归因被验证**（prereg 次级观察量；措辞按外评二审校准）：

| 观察量 | P5-A | P5-A.2 | 结论 |
|---|---|---|---|
| ca_plain malformed | 180/180 | **0/180**（60/60 served，+0.705） | 示例修复格式塌缩 ✓ |
| ca_skills guard 拒绝 | 179/180 feature_missing | 62 候选 **guard_unsatisfied**（55/60 served） | 契约生效：guard 被真实评估；残余=LLM 写 guard 过紧（真实判断差） |
| 主臂 mean Δ | 0.000 | +0.630 | **观察到的 deficit 缩小 81%**（(0.799−0.150)/0.799；跨 seed 批 + 双变更同施 → 强受控归因**非严格 factorial 分解**；det +0.799→+0.780、random +0.650→+0.660 两批稳定支持难度可比） |
| served-conditional | +0.852（n=18） | +0.687（n=55） | 服务质量稳定但 < det +0.780 |

**判据③精确口径（外评二审去重核验）**：novel_effective_edits = **21 次 episode 级有效新颖
选择，对应 11 个 SHA-distinct、menu 不可表达的程序**（两程序重复 7×/5×）；去重后 11 ≥ K=3，
判据③稳健通过。跨臂对照（novel-eff 次数/distinct SHA）：random_valid **41/31**、ca_plain
**23/23**、ca_skills 21/11、ca_skills_memory 9/7、det_search 13/1——正确结论是 **"LLM candidate
supply 存在"，而非"优于随机搜索"**。尾部信号（post-selection，仅观察不作主比较）：ca_skills
novel-eff 条件均值 +1.032 > random +0.892。

**臂序诊断（剩余问题=选择/证据形式，非格式）**：det +0.780 > **ca_plain +0.705** >
ca_skills_memory +0.675 > random +0.660 > **ca_skills +0.630**——格式问题消失后 skill cards
呈**负向锚定**（引导过度 guard、模板化模仿、多样性塌缩：distinct SHA 23→11）；memory 把
+0.630 拉回 +0.675 但仍低于 ca_plain → 强烈支持 packet v2 的必要性：问题不是"给没给知识"，
而是**知识是否以连续、实例相关、可比较的证据形式提供**。残余差距非单 cell 集中
（snrHigh 0.581 vs 0.715；snrLow 0.727 vs 0.909），是广泛的语义选择偏差 + guard 校准不足。

**命名勘误（外评二审）**：本 run `task=forecast_only` 而 cell 名为 `anomaly|*`——后者指
**生成 series family（注入异常形状的合成底座）**，非 anomaly-detection task；P5-A.3 起
records/manifest 改用 `series_family` 字段消歧。

**诚实注记（本 retrial 仍非终审）**：外评核查证实 runner `packet=dict(obs)` 绕过 EvidencePacket
v2——**连续证据（R1）从未喂给 LLM**，而 slice v2 证明连续证据正是能翻盘的杠杆。因此残余
−0.150 差距是"无连续证据条件下的语义/选择差"，终审 = **P5-A.3**（外评采纳路线图⑤）：真实
Monash 数据 + packet v2 契约真源 + ReadinessPlan→deterministic compiler 臂 + 外部基线 +
四指标分解（semantic/compliance/selection/harness-benefit），**seeds 80–99 新预注册**。

**协议碰撞记录（透明留痕）**：并行会话的外评采纳路线图原拟将 seeds 60–79 用于上述更大
retrial；本会话已按先冻结的 prereg_p5a2_retrial.md（窄版两前置）消耗之。两次审判各自协议
有效、并列入档；路线图⑤ 顺延至 seeds 80–99。

**勘误（外评②）**：P5-A 正文"504 次真实调用"及本 run"549 次"均为 **composer 调用次数
（含磁盘缓存命中）**，非纯网络调用数；归档 run 未分离记录。自 P5-A.3 起 runner 落
`client_stats`（n_api/n_hit 分离，机器已接）。冻结正文不改，以本勘误为准。

**安全（外评①，代码侧已执行）**：`llm/client.py` 硬编码 fallback key 已删除，改环境变量
+ 网络调用前 fail-loud（缓存重放不受影响）；**旧 key 在仓库历史中，平台侧轮换作废是
用户动作，截至本增补未完成**。

---

## 终审增补（2026-07-10）：P5-A.3 已执行，B 分支定局

用户完成平台侧 key 轮换 + `DEEPSEEK_API_KEY` 注入后，preflight PASS（n_api/n_hit 分离
核验），**seeds 80–99 一次性消耗**。正式判决见 **`../P5A3Final/VERDICT.md`**：主臂
pv2_plan_compiler − det = **−0.177 CI90[−0.261, −0.097]**（真实 Monash、合规按构造保证、
packet v2 连续证据喂足），判据①②FAIL / ③④PASS →
**claim_branch = self_updating_deterministic_with_llm_novelty_supplier**。四指标分解把
残余差距钉在 **generation（LLM 池上限低于 det 实选，−0.114 CI 不跨 0）**，非合规、非
选择；负向锚定双消融复现；random 文法采样显著胜全信息 LLM（+0.242 CI 不跨 0）。本文件
的"终审 = P5-A.3"预告就此闭环，P5 阶段全部 claim 冻结；上一段安全事项的用户侧动作
（平台 key 轮换）亦已完成。
