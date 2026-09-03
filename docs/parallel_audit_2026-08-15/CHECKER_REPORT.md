# Checker 复核报告（三角色第二轮）

复核方式：逐条打开执行者引用的函数/文件，不做“信任式验收”。

## 一、执行者稿中“确认正确”的结论

1. **Program 层 delayed 已经是单侧 veto**：确认。
   `method.handle_feedback_delayed` 批准条件为
   `delayed.verification.passed and finite and dg >= -MATERIAL_THRESHOLD`，
   不要求显著正。

2. **Episode 层无中性档**：确认。
   `online_loop._update_delayed_status` 对 support 正、delayed ∈ [−M, M)
   的轨迹判 `CONFLICT/RESTRICTED`。

3. **中性 delayed 可批准并可走向 active**：确认。
   `handle_feedback_delayed` 对 `dg >= -M` 返回 `approved`；
   runner 随后调 `activate_approved` 时只要 stage=approved 就
   `store.set_active`。没有额外的“中性不得扩权”守卫。

4. **已部署 Skill 只有整卡删除，无 Scope 收缩**：确认。
   `revoke_deployed_skill` 只查找并删除 `skills/{learned,bootstrap}/*.json`
   ，
   重编译后 `set_active`。无 RESTRICT/SPLIT 路径。

5. **Shared Promotion 无入口**：确认。
   在 `methods/`、`runtime/`、`evaluation/functional/` 中 grep
   `shared_promotion`、`SHARED_PROMOT`、`cross_domain_promotion`、
   `promote_shared` 均无命中（排除 pycache）。当前所有 active 都是
   Target-local。

6. **BSE 组件确实存在且可复用**：确认。
   `_bse_assemble_rule`、`_bse_rule_fires`、`_bse_parse_slow_choice`、
   `_bse_replay_rows`、`_bse_pass_evaluation` 与执行者稿一致。
   BSE 规则未写入 h0，只存在于 report JSON——执行者已指出。

7. **fault_routes 数量与 actionability 分布**：确认。
   25 个 subtype；17 EDITABLE_M0、5 EVIDENCE_BACKLOG、
   1 INSTRUMENTATION_BACKLOG、1 OBSERVATION_CAPABILITY_BACKLOG、
   1 CAPABILITY_BACKLOG。

## 二、需要修正/补充的点

1. `handle_fast_winner` 当 `support_gain` 由调用方传入时，
   `support_passed=True` 是“复用已通过的 probe”，不重复执行 verifier。
   执行者稿中未写这一层；不影响结论，但状态审计表应补一行
   “support 复用语义”。

2. Fault Family 映射中有两个主观边界：
   - `PROTOCOL_GAP` 暂归 UPDATE_POLICY，但它是运行协议断点，
     更接近 INSTRUMENTATION。建议标 `TENTATIVE`。
   - `EXECUTION_MISMATCH` 暂归 UPDATE_POLICY，但它涉及
     deterministic_verification/safety，也可能归 PROGRAM。
   两个都在映射表中标 `TENTATIVE`，不擅自改。

3. BSE matcher 目前只支持单特征 `ge/le`。未来结构 Pattern 若产出
   多特征 trigger，不能直接塞进现有 `_bse_rule_fires`；应先经过
   reviewer/user 批准扩展，而不是在执行阶段扩展。

4. `SKELETON_SLICE_PROTOCOL.md` 中“Slow 只选择 BSE 式 P1/P2/abstain”
   写得过强。未来结构 Pattern 可能不是二元阈值，Slow 动作空间需要由
   新 trigger 决定；空壳协议应保留动作占位，不应提前把 BSE 的二元
   choice 写死为默认。

## 三、checker 的修正指令

- GATE_STATE_AUDIT：补 support 复用语义。
- FAULT_FAMILY_MAPPING：两处标 TENTATIVE。
- BSE_REUSE_REVIEW：补一句 matcher 单特征限制。
- SKELETON_SLICE_PROTOCOL：把 BSE 式 choice 从“默认流程”改为
  “候选流程之一”。
