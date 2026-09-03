# 晨间报告：P2-v3 → Source Memory 价值链（Wave 0-4 闭包，2026-08-13）

三角色交接格式。

---

## CURRENT_VERDICT

**P2-v3 自然经验 → Context 绑定 → Signed 检索 → A5/A3 全链执行完毕，按停止规则收束**

| Wave | verdict | 关键事实 |
|---|---|---|
| 0 | NATURAL_REPEATED_ACTION_HARM_FOUND + TOP_FAMILY_NO_COMMON_HEADROOM | 216 有效读数、41 material negative、impute_fft 8 窗 × 7 独立 series、5 替代无共同修复 |
| 1 | **P2_CONTEXT_BOUND_EVIDENCE_PASS** | 36 单元（12 series × 3 origins）Context 绑定；信息墙全过（local_pattern 仅 6 公开键、gain 只进 response）；CONFLICT 4 对；分布 POSITIVE 9 / NEGATIVE 8 / NEUTRAL 19 |
| 2 | **EXISTING_OBSERVATION_UNIDENTIFIABLE** | 单特征规则 `region_start>0.074` 命中 6/7 负向 series（1960d9bd 漏检）+ 4 series 正向误标——留一验证失败 |
| 3-B | **NN5_IMPUTE_FFT_UNIDENTIFIABLE_WITH_CURRENT_OBSERVATIONS** | 4 个机制相关候选 Observation（missing_block_phase / seasonal_peak_overlap / missing_block_count / changed_fraction）全部无法区分——按任务书关闭 family，不堆特征 |
| 4 | **ADHERENCE_GAP** | A5 真实检索到 Source（positive=193c26a6@600 / negative=1025b235@600，evidence_sufficient=True）但 Target @712 两臂（A3/A5）都 abstain——行为无差异 |

## BEHAVIOR_CHANGED

无 Harness 行为改变——本班次是证据/检索链验证（Evidence/Memory 修复，非新 Capability）：P2-v3 Episode 的 local_pattern 从 `{"support_gain": gain}`（部署不可见）改为 6 个公开 Pattern 键（missing_fraction / longest_missing_run_fraction / region_start/end / period_reliability / period_change_score）——gain 只进 support_response。

## REAL_EVIDENCE（全部 development 级，零新 Claim）

- 装置修复链：run-1 cohort 聚合缺陷 → run-2 单 series 无 eval 假象（用户核查）→ v3 完整 roster 12+8 + train_series_scope（216 读数全有效）——三版留痕。
- **关键否定结果**：impute_fft 的正负翻转在当前 6 公开特征 + 4 候选 Observation 下不可区分（留一验证）——Observation gap 定位准确（不是 Retrieval/Adherence/Policy 缺口——是"Pattern 看不见"）。
- **A5 检索成功但无行为改变**：同域 held-out origin @712 上 A3/A5 都 abstain（候选池空/Agent 弃权）——Memory 无从影响（无候选可选）。Adherence gap 名义成立，根因是 Target 入口无可用候选。

## FIRST_FAULT（本班次定位的阻塞面）

**OBSERVATION_GAP**（Wave 2/3-B）：impute_fft 正负翻转不可被当前部署可见特征区分——与旧 KDD winsorize 翻转结论同构（翻转原因不在可观察 Context 内）。A5/A3 的价值实验被前置的候选空置阻塞（@712 无合法候选——Target 入口 Supply 问题）。

## NEXT_BRANCH

1. **Target 入口 Supply 修复**（Wave 4 阻塞面）：@712 候选池空——查 fast_agent 的 _actionable_operators 为何无合法候选（verifier 拒绝？契约？）——最小修复后重跑 A5/A3 一次（任务书 Wave 4 未完成——两臂 abstain 未产生可比较数据）。
2. **新 Observation**（Wave 2/3-B 关闭后）：若 OBSERVATION_GAP 定位成立，需机制相关新观察（任务书限定一个）——但 Wave 3-B 已试 4 个候选全失败 → 该 family 关闭，不再堆特征。
3. **跨数据集 Target**：KDD 的 missingness Context overlap 检查（v6.DATASET_CONFIGS 无 KDD 键——需单独装置）——跨数据集 A5/A3（任务书优先项）。

## STOP_CONDITION（本班次命中的停止规则）

- Wave 3-B：新增 Observation 后仍不可区分 → 关闭 family（NN5_IMPUTE_FFT_UNIDENTIFIABLE_WITH_CURRENT_OBSERVATIONS）
- Wave 4：Memory 被检索但正常入口不改变行为 → Adherence gap → 停止并出报告

---

## 附：全部新文件

**Runners**：run_v1_p2_natural_batch_missingness.py（v3 装置）、run_v1_p2_context_binding.py、run_v1_p2_context_separability.py、run_v1_p2_observation_gap.py、run_v1_p2_a5_vs_a3.py

**报告**：w1_p2_natural_batch_missingness_report.json、w1_p2_run1_apparatus_bug_record.txt、w1_p2_context_binding_report.json、w1_p2_context_separability_report.json、w1_p2_observation_gap_report.json、w1_p2_a5_vs_a3_report.json

**预注册**：docs/P2_NATURAL_BATCH_MISSINGNESS_PREREGISTRATION.md

**LLM 成本**：Wave 4 两臂真实 Fast 入口（A3+A5 各一轮——abstain）；本班次其余全零 LLM。
