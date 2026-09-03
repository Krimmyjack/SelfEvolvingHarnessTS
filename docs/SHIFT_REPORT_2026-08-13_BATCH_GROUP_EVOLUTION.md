# 晨间报告：Batch Group Evolution 自治班次（2026-08-13 深夜→晨）

三角色交接格式（用户任务书要求）。

---

## CURRENT_VERDICT

**BATCH_GROUP_EVOLUTION_MECHANISM_ESTABLISHED + FIRST_NATURAL_FAMILY_CLOSED_NO_HEADROOM**

主机制链（census → capsule → real Slow → 组内 replay 门 → 补集 → delayed）全部跑通并被真实 LLM 消费；首个自然跨 series failure family 在 development block 中被发现并**如实关闭**（无共同正向替代 + 供给穷举无解 + 翻转不可分 → abstain）。零 PASS 造假、零 Claim 越界。

## BEHAVIOR_CHANGED（今晚实际改了什么 Harness 行为）

1. **Group evidence 主链**（group_fault.py / method.py / online_loop.py）：分组键=完整 workflow 指纹；Capsule 含 per-episode 对齐行 + view 对齐（view_keys）+ 正/冲突对比案例，并**真正进入 Slow Agent 输入**；contracts/TaskContext 透传；跨 series 组内 replay 按 episode 解析 executor（origin 碰撞安全）。
2. **Typed-patch 绑定规则到达 Agent**（slow_agent.py）：typed_patch_binding_rule + 白名单错误反馈 + surface 以 manifest 自身 skill_id 确定性实例化。
3. **abstain 通道修复**（agent_core.py）：SLOW/edit 阶段教学 no_proposal 信封模板 + reason_code 枚举；重试反馈附 no_proposal 模板。修复前真实 LLM 想弃权但通道失效（回退 manifest）；修复后**一次调用正确弃权**。

## REAL_EVIDENCE（真实运行证据，全部 development 级）

| 实验 | verdict | 关键数字 |
|---|---|---|
| Gate1 证据链 | GROUP_EVIDENCE_CHAIN_GATE1_PASS | 8/8 检查；T117 组链到 pending；对齐行 T131 双窗一致恶化/T133 一致改善 |
| Wave2 witness v3 | GROUP_TYPED_PATCH_MECHANISM_PASS | 真实 LLM 1 调用选对 hampel；v1/v2 两次接线缺口留痕 |
| Wave3 census | DEVELOPMENT_FAMILY_FOUND | winsorize NEGATIVE 跨 4 独立 series（T1/T10/T100/T101）6 窗口 min −0.164；exposed 无跨 series family；E31 winsorize=仪器失效 |
| Wave4a-r1 | （修正后）SLOW_AGENT_EVIDENCE_USE_FAILURE | abstain 通道失效证据：edit_id=abstain-* 但回退 manifest → replay 门拒 3/6 窗 |
| Wave4a-r2 | EVIDENCE_GROUNDED_ABSTAIN | 修复后 1 调用正确弃权 insufficient_public_evidence |
| Wave4b supply | SUPPLY_EXHAUSTED | 预注册空间（3 hampel 变体+6 两步组合）54 评估无全过候选 |
| Wave4c-1 翻转可分性 | FLIP_CONTEXT_NOT_SEPARABLE | 当前特征词汇零误差不可分（正 8/负 6 窗口） |
| Wave4c-2 观察探针 | FLIP_PROBE_NOT_SEPARABLE | outlier_density/calendar_phase 两探针均不可分 → 按停止条件 abstain |

## FIRST_FAULT（当前唯一真实 first fault）

winsorize 效用翻转（正负窗口）**不能被当前任何部署可见特征/两个预注册探针区分**——翻转原因不在可观察 Context 内。可解释的下一步不在此数据上。

## NEXT_BRANCH（推荐，待用户拍板）

1. **P6 matched-budget 基线**（主机制已成立，前置满足）：单 Episode Slow / Group Fault v0 / Batch Context-conditioned Slow / 等预算 Pipeline Search 四臂，固定 LLM 调用/候选数/Support 预算，在 T117 exposed + development block 证据上判断收益是否来自可复用 Harness knowledge。**建议新会话完整预注册后执行**（今晚尾段不半途开工）。
2. **PIA 校准**（触发条件接近：今晚 100+ 完整 Consumer 评估已显示成本瓶颈；已有 gold：14 个 winsorize 窗口 + 12 headroom + 54 supply 评估）：Program ΔX/ΔY → first-order Response Sketch vs gold——只验 top-k recall/sign agreement/harmful FP/full-Support 减少量，绝不接批准。
3. **第二个 development block**（需用户拍板预注册）：cache 顺序下 4 个未用 series（T102-T105），同装置——查第二 family 是否也无 headroom（连续两个 family 无 headroom = 停止条件）。

## CODE_ALLOWED / FRESH_DATA_ALLOWED

- 允许：P6 四臂 runner、PIA 校准 runner（只读 gold）、block 2 census runner——均需 docstring 预注册，一机制一实验。
- 禁止：改 Consumer/Metric/split；按 outcome 挑 cohort；扩大门（组内全 ≥M / holdout ≥ −M / delayed 门是 sealed）；SHA/Ledger/向量库/多 Patch 平台；未预冻结 fresh outcome。

## STOP_CONDITION（已达）

连续两个最小 Observation 无法区分 Utility flip → abstain；development block 单一 family 已关闭；P2 方向队列（Supply/Scope-Observation）穷尽；Wave 5（H1）/6-自然 Source/双 Patch 前置不满足。按任务书停止并生成本报告。

---

## 附：今晚全部新文件与改动

**代码改动（均未 commit，工作树增量）**：
- methods/ttha/group_fault.py（重写）、method.py、online_loop.py、slow_agent.py、agent_core.py（abstain 教学）
- 新 runners（evaluation/functional/）：run_v1_group_evidence_chain_dev.py、run_v1_group_witness_real_slow_dev.py、run_v1_batch_census_dev.py、run_v1_group_evolution_top_family_dev.py、run_v1_program_supply_dev.py、run_v1_flip_separability_dev.py、run_v1_flip_observation_probe_dev.py

**新报告（artifacts/functional/e2/）**：w1_group_evidence_chain_gate1_report.json、w1_group_witness_real_slow_report{,_v2,_v3}.json、w1_batch_census_dev_report.json、w1_group_evolution_top_family_report.json(+_corrected,+_r2)、w1_program_supply_dev_report.json、w1_flip_separability_dev_report.json、w1_flip_observation_probe_dev_report.json

**LLM 成本**：真实 Slow 调用共 7 次（witness v1:2/v2:1/v3:1 + wave4a-r1:2 + r2:1），全部留痕（模型 gpt-5.6-luna / temperature=0 / CountingClient 计数在案）。

**Claim 边界**：以上一切为 development exposure（零新 Claim）。可声称：机制级（typed-patch 通道成立、abstain 通道修复生效、组级证据被真实 LLM 消费）；不可声称：任何 holdout/delayed 级性能效应、winsorize 普遍有害、任何已批准 Skill。
