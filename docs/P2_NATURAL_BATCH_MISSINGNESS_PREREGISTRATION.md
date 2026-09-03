# P2 自然 Batch Missingness — 预注册（2026-08-13）

真正目标验证：自然多轨迹 Batch 是否形成 Target-local Skill 并在真实
正常入口被采用。执行者：
`evaluation/functional/run_v1_p2_natural_batch_missingness.py`

## 1. 装置（全部预注册——outcome-blind）

- 数据：NN5（monash:nn5_daily，20 series hash uid，长度 791）
- task_consumer：forecast|ridge|sMASE；period 7；评估口径 v1.gain_at
  （旧装置同款）
- **adaptation block**（outcome-blind 积累 Action–Response）：
  series × origins {600, 632, 680}（≤743 可评估上限）——60 决策点；
  每点 H0 确定性 Fast（SealedProbeBackend force_pool——imputation
  family + repair_level_shift 探测序）→ 写 Episode（完整 workflow +
  gain + origin）——零 LLM
- **held-in** origin 712（family 首 series——未参与 census）
- **delayed** origin 728（family 首 series——census 之后）
- 候选 DSL：imputation family（impute_ar/impute_ssm/impute_linear/
  impute_fft/impute_ema）+ repair_level_shift——H0/H1 相同 DSL 相同
  预算

## 2. 链（用户规格逐条编码）

1. **自然 failure family**：group_first_faults（完整 workflow × sign，
   min_group=2）且 **≥2 独立 series**（不只同 series 多 origin）——
   单条失败 → NO_BATCH_FAMILY（不退回单条制造 PASS）
2. **Fault Diagnosis Card**（确定性）：失败数值 + selectable 错误类型
   （fault_cases 机械判据）+ ≤2 合法 Edit Intent（候选 = family 窗
   headroom 最高的两个 imputation 替代——**D_patch 数值开放**，headroom
   用组窗实测）
3. **Agent 选择**（1 次 LLM）：edit_intent/patch_id（A/B/C）
4. **Runtime 编译** Manifest（ADD skill_library.entries/{skill_id}）
5. **Support**（family 全部窗 ≥M）→ **held-in @712**（≥ −M）→
   **delayed @728**（≥ −M）→ approved → Skill 持久化
6. **H0/H1 真实正常入口**（family 首 series @728，真实 LLM fast 各一
   轮——同 DSL 同预算）：H1 检索/选择/执行 learned Skill vs H0 空池
   abstain——效用取**实际执行 winner**，abstain=identity=0
7. **regret 报告**：离线发现的最佳未选择 Workflow（若存在）只报为
   oracle/headroom regret——不作为 H0 实际效用、不判 Scope/Risk
8. 只处理排名第一的 failure family、最多两个候选、一个 Edit

## 3. verdict（预注册）

- `NATURAL_BATCH_LOCAL_SKILL_PASS`：多条自然轨迹成功形成并采用本地
  Skill（H1 真实检索+选择+执行且相对 H0 abstain 增益 ≥0）
- `NO_BATCH_FAMILY`：无 ≥2 独立 series 的重复自然失败
- `NO_COMMON_HEADROOM`：有共同失败但现有 Workflow 无共同修复
- `SAFE_ABSTAIN`：证据不足——Agent 正确不修改（合法弃权）
- `POLICY_NO_GAIN`：Skill 形成但未优于实际 H0 policy
- `PROTOCOL_FAILURE`

## 4. 纪律

- 零新 Claim（development exposure）；不按 outcome 换 roster；
  一机制一实验；报告不覆盖；LLM 调用留痕（选择 1 + H1/H0 入口各一轮
  ≈ 7 次，temperature 0）
- 成功后顺序：等预算 Batch vs single vs deterministic search → A5/A3
