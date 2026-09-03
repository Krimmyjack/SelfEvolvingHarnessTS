# S2 NN5 Missingness 受控正例完整链 — 预注册（2026-08-13）

受限 Case 框架首次完整链验证：Batch 失败 → 五类选择 → Case 匹配 →
一个 Typed Edit → Support → held-in → delayed → Skill 写入 → 下一
正常入口采用（LLM + 确定性双口径）→ removal 行为恢复。

执行者：`evaluation/functional/run_v1_s2_nn5_controlled_chain.py`

## 0. 定位与用户裁决（硬约束）

- 方案 (c)：真实 LLM 入口与确定性 Runtime 入口都测，**分开判定**：
  - Runtime 采用 + LLM 未采用 → `LOCAL_SKILL_MECHANISM_PASS` +
    `AGENT_ADHERENCE_FAIL`（不算完整闭环）
  - 两者都采用 + removal 后行为恢复 → `CONTROLLED_BATCH_EVOLUTION_CHAIN_PASS`
  - 两者都未采用 → 检索/绑定/执行接线失败
- **不硬编码 impute_ar**：Runtime 只把白名单限制为 ≤2 个合法
  imputation Workflow；Slow Agent 依据 Batch Capsule 自主选择；已知
  headroom 只用于事后核销。
- **两类错误不混淆**：
  - 初始入口未选已证实更优的 impute_ar → `WORKFLOW_DECISION_ERROR`
  - Skill LOCAL_ACTIVE、被检索却未被 Agent 使用 →
    `SCOPE_MEMORY_RISK_ERROR`（新 Action–Response 写入，不得回改
    前一 Case 的错误类型）
- **旧工件恢复冻结，不重扫 gain 选窗口**；旧装置无法提供相互独立
  的 patch/held-in/delayed origins → 如实标 development positive
  control，不伪装 fresh 验证。
- Case 的 NEW/MATCH 由 Runtime 字段比较决定，不预设必须 NEW。

## 1. 冻结装置（全部从旧工件恢复——零重扫）

| 项 | 冻结值 | 来源 |
|---|---|---|
| domain | nn5（monash:nn5_daily, daily_regular, period 7） | v6.DATASET_CONFIGS['nn5'] |
| task_consumer | forecast\|ridge\|sMASE | 旧报告 |
| roster | `v6._fixed_roster(nn5)`——20 series（全 train，hash uid） | 旧装置同款 |
| timeline | src (536, 584) / tgt (632, 680) | core.TIMELINE['nn5'] |
| 评估口径 | `v1.gain_at(roster, values, config, compiled, origin, baseline_cache)` | 旧装置同款 |
| 已暴露失败 | repair_level_shift @632 = −0.0789（delayed −0.1570） | w1_target_local_loop_report.json a5 round1 |
| 已暴露正向替代 | impute_ar @632 = +0.09659 / @680 = +0.02764；impute_ssm @632 = +0.0697 | w1_target_local_loop{,_llm}_report.json |
| 合法 imputation 候选 | impute_linear / impute_fft / impute_ema / impute_ssm / impute_ar | 算子 registry |
| Typed Edit 白名单 | **[impute_ar, impute_ssm]**（5 个中预注册取已暴露可核销的两个） | 预注册 |

## 2. 数据角色与 origins（受控构造——如实 development 标注）

- **D_patch（组内）**：632（已暴露 replay）+ **728（受控新评估**——
  632+96——失败 workflow repair_level_shift 在此 origin 再评估一次以
  形成 ≥2 窗组；若不 material 负则如实记组不成形并走单条路径）
- **D_heldin**：776（受控新评估——728+48——patch 的组外同域验证）
- **D_delayed**：680（已暴露 replay——patch delayed 门）
- **采用/removal**：776（下一正常入口——LLM 入口 + 确定性入口双口径）
- 新评估 6 次（repair@728 / patch@728 / patch@776 / entry@776 ×2 /
  removal@776）——development positive control，**不称 fresh 验证**

## 3. 完整链步骤（每步判定预注册）

1. **组形成**：repair_level_shift @632（已暴露）+ @728（新评估）→
   group_first_faults(min_group=2)。不成组 → PROTOCOL_FAILURE（如实）。
2. **五类选择**（LLM 选择题，S1 接口同款）：taxonomy/allowed 分离；
   allowed = [WORKFLOW_DECISION_ERROR]（机械证据：winner_probed=
   impute_ar +0.0966 已测、agent_chosen=repair_level_shift ≠ winner）；
   Runtime 复核（∈allowed / refs 可解析 / matched ∈ 检索）。
3. **Case reconciliation**（Runtime 确定性）：无 imputation Case →
   NEW_CASE → Runtime 写 CANDIDATE_CASE（case-0004，普通顺序 ID；
   Agent 无写入权）。
4. **一个 Typed Edit**（真实 Slow Agent 1 调用）：白名单
   [impute_ar, impute_ssm]；capsule = 组证据 + 新 Case 摘要；
   no_proposal 通道开放；Agent 自主选择——不告知 gain。
5. **组内 replay 门**：所选 patch @632 ≥M 且 @728 ≥M（全过才
   pending）。
6. **held-in**：@776 ≥ −M。
7. **delayed**：@680 ≥ −M 且 verifier 过 → LOCAL_ACTIVE → Skill
   写入 snapshot（LOCAL_DRAFT → LOCAL_ACTIVE 生命周期照旧）。
8. **采用双口径**（@776 下一正常入口）：
   - 确定性入口：Skill 进检索池 → chosen=cand_skill_* 且执行 →
     `runtime_adopted`
   - 真实 LLM 入口：正常 Fast 入口（1 调用）→ chosen 是否含
     impute_ar skill → `llm_adopted`
   - 事后核销：Agent 所选 patch 是否为已测正向候选（impute_ar/
     impute_ssm 之一）。
9. **removal**：@776 移除 Skill 后行为恢复对照。

## 4. verdict（预注册——组合式）

- `CONTROLLED_BATCH_EVOLUTION_CHAIN_PASS`：LLM 采用 + Runtime 采用 +
  removal 恢复
- `LOCAL_SKILL_MECHANISM_PASS+AGENT_ADHERENCE_FAIL`：Runtime 采用 +
  LLM 未采用
- `RETRIEVAL_BINDING_EXECUTION_FAILURE`：两者都未采用
- `SCOPE_MEMORY_RISK_EVIDENCE`：Skill LOCAL_ACTIVE 被检索但 LLM 未用
  （写新 Action–Response——不回改 case-0004 类型）
- 各门拒绝 → 对应 `_REJECTED` 阶段 verdict + 如实记录
- `PROTOCOL_FAILURE`：组不成形 / 契约失败 / LLM 超预算（每 propose
  CountingClient 硬停 2）

## 5. 纪律

- 零新 Claim——development positive control；报告标注"旧装置无独立
  held-in origins——新 origins 728/776 为受控构造，非 fresh 验证"。
- 一机制一实验：不改任何 Harness 代码；Case 写入走 Runtime 确定性
  路径。
- 真实 LLM 调用预算：5-class 1 + Slow Edit 1（+校验重试）+ 采用入口
  1 = 3-4 次原始调用，全部留痕（temperature 0 / gpt-5.6-luna /
  CountingClient）。
- 报告不覆盖；新 Case 写入 case store（追加 case-0004——不动前三
  case 的字段与类型）。
