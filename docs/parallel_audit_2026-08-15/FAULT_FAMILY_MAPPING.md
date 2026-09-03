# Fault Family 轻量映射（执行者稿）

将 `evaluation/minipipe/feedback/fault_routes.json` 现有 25 个 subtype
映射到六个 fault family。subtype 保留；本表只做普通只读映射，不建 Schema、
不改 fault_routes 执行路径。

## 1. 六个 family
- OBSERVATION：可见信息不足以区分正负情况
- PROGRAM：合法候选、Binding 或步骤缺失
- CONTROL：合法候选存在但供应/选择/预算错误
- SCOPE_RISK：同一 Workflow 跨 Context 效用翻转或执行权过宽
- MEMORY：Source/Target 经验检索或权威使用错误
- UPDATE_POLICY：Skill 创建、复用、撤销生命周期错误

## 2. 映射表

| subtype | family | slow 可修改 Skill？ | 备注 |
|---|---|---|---|
| CRITIC_BLIND | (non-actionable) INSTRUMENTATION | 否 | 仪表问题，先修仪器 |
| OBSERVATION_GAP | OBSERVATION | 否 | 只允许后续 Observation 提案 |
| PROTOCOL_GAP | UPDATE_POLICY (TENTATIVE) | 是 | 也可归 INSTRUMENTATION；运行协议断点，待裁决 |
| OBSERVATION_PROCEDURE_GAP | OBSERVATION | 是 | inspect/localize 程序面 |
| LOCALIZATION_PROCEDURE_GAP | OBSERVATION | 是 | 同上 |
| LOCALIZATION_UNKNOWN | OBSERVATION | 否 | 证据不足 |
| MECHANISM_UNKNOWN | OBSERVATION | 否 | 证据不足 |
| MECHANISM_AMBIGUITY | OBSERVATION | 是 | 需结构化观察澄清 |
| SKILL_LIBRARY_GAP | PROGRAM | 是 | ADD capability |
| SKILL_CONTENT_GAP | PROGRAM | 是 | PATCH capability |
| RETRIEVAL_MISS | MEMORY | 是 | retrieval/applicability |
| PROPOSAL_CONTROL_GAP | CONTROL | 是 | proposal_control |
| SELECTION_MISS | CONTROL | 是 | selection_control |
| SCOPED_SELECTION_GAP | SCOPE_RISK | 是 | capability scope |
| PROBE_SELECTION_CONTRADICTION | CONTROL | 是 | selection_control |
| IMPLEMENTATION_MISMATCH | PROGRAM | 是 | proposal_control |
| EXECUTION_MISMATCH | UPDATE_POLICY (TENTATIVE) | 是 | 涉及 safety/verification，也可归 PROGRAM |
| OUTCOME_GAP | SCOPE_RISK | 是 | utility 与预期不符 |
| RISK_GAP | SCOPE_RISK | 是 | safety/capability risk |
| OBSERVABLE_DERIVATION_PROCEDURE_GAP | OBSERVATION | 是 | inspect_and_localize.body |
| WORKFLOW_GUIDANCE_GAP | PROGRAM | 是 | build_contrastive_candidates.body |
| OBSERVABLE_FEATURE_SCHEMA_GAP | OBSERVATION | 否 | 只允许 Observation 能力提案 |
| OPERATOR_GAP | PROGRAM | 是 | 能力 backlog |
| EXPRESSIBILITY_UNKNOWN | OBSERVATION | 否 | 证据不足 |
| CANDIDATE_SUPPLY_UNKNOWN | PROGRAM | 否 | 证据不足，先做确定性 supply 检查 |

## 3. 概念哨兵（fault_routes 中暂不存在，但必须显式保留）
- `ESTIMATOR_VARIANCE` → 不可由 Slow 修改 Skill；先修测量。
- `INSTRUMENT_BLOCKED` → 不可由 Slow 修改 Skill；先修仪器。

## 4. 明确不可修改集合
以下 family/status 不允许 Slow 修改 Skill：
- INSTRUMENTATION：`CRITIC_BLIND`
- OBSERVATION 但 evidence backlog：`OBSERVATION_GAP`、`LOCALIZATION_UNKNOWN`、
  `MECHANISM_UNKNOWN`、`EXPRESSIBILITY_UNKNOWN`
- CAPABILITY_BACKLOG：`OPERATOR_GAP`（需要 operator 能力，不靠改 Skill）
- OBSERVATION_CAPABILITY_BACKLOG：`OBSERVABLE_FEATURE_SCHEMA_GAP`
- `CANDIDATE_SUPPLY_UNKNOWN`

## 5. 下一步（等待 reviewer）
- 是否需要把 `PROTOCOL_GAP` 从 UPDATE_POLICY 拆到 INSTRUMENTATION？
- `EXECUTION_MISMATCH` 归 UPDATE_POLICY 还是 PROGRAM？
- 该表是否应作为普通 Python 常量放在一个不参与实验主链的文件，还是只保留文档？
