# SelfEvolvingHarnessTS-deepseek 项目推进路线图（修订版）

**当前状态**: 纵向集成 C1-C4/C6-C7 通过，C5 失败  
**当前阻塞**: Skill 在 R2 verify 拒绝，未进入候选池  
**审核日期**: 2026-08-12  
**审核者**: 本地 agent + 外部 AI 联合审核  

---

## 一、当前真实状态总结

### 已完成工作（2026-08-12）

| 阶段 | Verdict | 关键产出 | 日期 |
|------|---------|----------|------|
| **P0-P5** | OPERATIONAL_TARGET_LOCAL_SELF_EVOLUTION_DEV_PASS | online_loop + Draft guard + 8/8 checks | 2026-08-10~11 |
| **E0** | E0_HARDENING_PASS | 7 项评价语义修正 + 9/9 checks | 2026-08-12 |
| **E1** | NO_NATURAL_FAILURE | fresh 自然闭环（合法负档） | 2026-08-12 |
| **E2** | MEMORY_TWO_SLOT_CONTROL_PASS | 双槽机制 + 5/5 checks | 2026-08-12 |
| **Integration** | INTEGRATION_FAILED_C5 | 纵向集成 6/7（C5 失败） | 2026-08-12 |

### 当前能力边界（已验证）

✅ **已经可以 Claim**：
1. Target-local Program evolution 机制成立（P3，8/8）
2. 双槽防止 Source Memory 独占候选供应（E2，5/5）
3. Draft 执行权限门工作（P1）
4. Typed Patch + Runtime 选择 + 外部审批链工作（P1.5）
5. 持久化和重启语义正确（P5）
6. E0 评价语义硬化完成（chosen vs authorized / delayed 只计 winner / 等）
7. 真实 LLM 能因果使用双槽（Integration R1，LLM 选了探索候选而非 prior）

❌ **不能 Claim**：
1. ❌ "完整纵向闭环"（C5 未通过）
2. ❌ "Source Memory 减少 Target 试错"（E3 未运行）
3. ❌ "跨域迁移能力"（需 ≥2 Target 同向）
4. ❌ "完整 Fast Agent 自主生成 Workflow"（backend 是 DSL selector）
5. ❌ "Shared Capability"（需多 Target 稳定复现）
6. ❌ "六个 Surface 都已自然行动化"（只有 Program/Skill 完整）

### 当前第一阻塞（C5）

**现象**：
- R1: Skill 形成并批准 ✅
- R2: Skill 检索成功 ✅
- R2: Skill verify 拒绝 ❌ → 未进入候选池 → 未探测

**根因假设**（PLAUSIBLE，待 P0 确认）：
- **假设 A**: Skill frozen params 应该重新绑定当前 Context 特征值
- **假设 B**: Skill verify 应该用全窗口候选区域（不受 LLM inspect 限制）

**已排除**：
- ✅ 不是参数数值失效（R1 params 在 R2 context 全组合 verify 通过）
- ✅ 不是解析错误（_parse_frozen_steps 正常）

---

## 二、修订后的推进顺序（P0 → P1 → P2 → E3）

```
P0 单因素诊断（1-2 天）
  ↓
P1 应用修复并重跑纵向集成（1-2 天）
  ↓
P2 removal 对照（0.5-1 天）
  ↓
E3 fresh Target 1（5-7 天）
  ↓
E3 fresh Target 2（仅当 Target 1 PASS，5-7 天）
```

**总工期**：2-3 周（P0-P2）+ 1-2 周（E3 Target 1）+ 可选 1-2 周（E3 Target 2）

---

## 三、P0：Skill Binding 单因素诊断（第一优先级）

### 目标
定位 C5 根因：参数重新绑定 vs inspect region 语义

### 装置（零新数据 / 零 LLM）
- 已暴露数据：KDD T117 R2 @888
- 确定性 backend（复现/隔离变量）
- 相同 Request / TaskContext / Skill / Operator / verifier

### 三臂

| Arm | Skill 参数来源 | verify 候选区域 | 期望结果 |
|-----|---------------|----------------|----------|
| **A: frozen** | R1 frozen `[0.0467, 0.1717]` | R2 LLM inspect 声明 | reject（复现失败） |
| **B: rebind** | R2 当前 context 重新绑定 | R2 LLM inspect 声明 | pass（测假设 A） |
| **C: frozen-full** | R1 frozen `[0.0467, 0.1717]` | 全窗口 `[0.0, 1.0]` | pass（测假设 B） |

### 判定矩阵

| 结果 | 根因 | 修复路径 |
|------|------|---------|
| A=reject, B=pass, C=reject | 参数应重新绑定 | retrieval 时机械应用 `public_parameter_bindings` |
| A=reject, B=reject, C=pass | guard 语义问题 | Skill verify 用全窗口候选区域 |
| A=reject, B=pass, C=pass | 两者都有贡献 | 同时修复 A+B |
| A=pass | 归因错误 | 重新诊断 |
| A/B/C 都 reject | 第三个未知因素 | 记录 rejection_reason，单独检查 |

### 验收标准
- [ ] A=reject 确认
- [ ] B 或 C 至少一个 pass
- [ ] 修复路径不新增 Schema / Binding DSL / Controller
- [ ] 回归检查：P3（8/8）+ E2（5/5）不破坏

### 交付物
- `evaluation/functional/run_v1_p0_skill_binding_diagnostic.py`
- `artifacts/functional/p0/skill_binding_diagnostic_report.json`
- `docs/P0_SKILL_BINDING_DIAGNOSTIC.md`（已创建）

---

## 四、P1：应用修复并重跑纵向集成

### 前置条件
P0 确认修复路径（假设 A 或 B 或 A+B）

### 修复方案 A：参数重新绑定

**文件**：`methods/ttha/fast_agent.py` 或 `method.py`（Skill retrieval 路径）

**修改量**：~30 行

**实现**：
```python
def _rebind_skill_params(
    skill: dict,
    public_features: dict[str, float],
    operator_metadata: dict,
) -> dict:
    """Skill retrieval 时重新绑定参数（复用 Registry 元数据）"""
    frozen_steps = skill["program"]["steps"]
    bindings = operator_metadata.get("public_parameter_bindings", {})
    
    if not bindings:
        return frozen_steps  # 无绑定规则，保持 frozen
    
    rebind_steps = []
    for step in frozen_steps:
        params = dict(step["params"])
        for param_name, feature_name in bindings.items():
            if param_name in params and feature_name in public_features:
                params[param_name] = public_features[feature_name]
        rebind_steps.append({"op": step["op"], "params": params})
    
    return rebind_steps
```

**约束**：
- ✅ 复用 Registry 已有 `public_parameter_bindings`
- ✅ 只在 retrieval 时应用，不修改存储格式
- ✅ frozen params 保留（审计/回滚需要）
- ❌ 不新增 Binding DSL / Schema / Controller

### 修复方案 B：Skill verify 用全窗口候选区域

**文件**：`methods/ttha/fast_agent.py`（verifier 调用点）

**修改量**：~20 行

**实现**：
```python
def _verify_skill_candidate(
    skill: dict,
    request: PrepareRequest,
    is_skill: bool = True,
) -> VerifyResult:
    """验证 Skill 候选（已批准 Skill 用全窗口候选区域）"""
    if is_skill:
        candidate_region = [0.0, 1.0]  # 全窗口（不受 LLM inspect 限制）
    else:
        candidate_region = request.inspected_region or [0.0, 1.0]
    
    return verifier.verify(
        program=skill["program"],
        candidate_region=candidate_region,
        preserve_outside=True,
    )
```

**约束**：
- ✅ 只修改 verify 调用时的候选区域参数
- ✅ 不修改 guard 语义本身
- ✅ 不修改 Agent proposal 的 verify 流程
- ❌ 不新增 Schema / Controller

### P1 运行

```bash
# 应用修复后重跑纵向集成
python -m evaluation.functional.run_v1_integration_vertical_loop \
  --tag p1_rebind_fix \
  --model gpt-5.6-luna

# 期望：C1-C7 全过
```

### 验收标准（P1）
- [ ] C1: 双槽填充（A5 池包含 prior + exploration）
- [ ] C2: A3 无 prior
- [ ] C3: Fast winner → Skill 批准
- [ ] C4: guard 工作
- [ ] **C5: skill_retrieved=True, skill_verified_into_pool=True, skill_probed=True**
- [ ] C6: removal 对照（行为变化）
- [ ] C7: 预算一致

### 交付物
- 修复代码（~30-50 行）
- `artifacts/functional/p1/integration_vertical_loop_p1_report.json`
- 回归检查报告（P3 + E2）

---

## 五、P2：Removal 对照实验

### 前置条件
P1 C1-C7 全过

### 目标
确认移除 Skill 后 Program 行为恢复（C6 完整验证）

### 装置
- 相同 R2 Request（KDD T117 @888）
- 两臂：with Skill / without Skill
- 确定性 backend（消除 LLM 方差）

### 观察
- [ ] with Skill: chosen = skill_candidate
- [ ] without Skill: chosen ≠ skill_candidate
- [ ] Program 行为差异（winner params / gain）
- [ ] removal 真正改变行为（非空操作）

### 验收标准
- [ ] 两臂 Program 行为显著不同
- [ ] removal 非空操作（不是 Skill 从未生效）
- [ ] 差异可归因于 Skill 存在性（不是 LLM 方差）

### 交付物
- `evaluation/functional/run_v1_p2_removal_control.py`
- `artifacts/functional/p2/removal_control_report.json`

---

## 六、E3：Fresh Cross-Domain Memory Value（最终目标）

### 前置条件
- P0-P2 全通过
- 纵向集成 C1-C7 全闭合
- removal 对照确认

### E3 Target 1（第一个 fresh Target）

#### 冻结顺序（outcome-blind）

**Step 1**: 冻结 Source Episode pack
```json
{
  "source_id": "integration_vertical_loop_r1",
  "episodes": [
    {"relation": "POSITIVE", "operator": "repair_level_shift", ...},
    // 所有合法 Episode（positive/negative/conflict/abstain）
  ],
  "frozen_at": "2026-08-XX",
  "eligibility": "R1 自然产生的完整 trajectory"
}
```

**约束**：
- ✅ 收纳 R1 全部合法 Episode
- ❌ 不根据 Target 是否匹配筛选
- ❌ 不做 sign-swap
- ❌ 不看 Target outcome 后优化

**Step 2**: 冻结 Target 1
```json
{
  "target_id": "kdd_k3_virgin_984",
  "cohort": ["T160", "T161", ..., "T179"],
  "origins": [984],
  "frozen_at": "2026-08-XX",
  "eligibility": "virgin + verifier + ≥2 replacements + Context overlap"
}
```

**约束**：
- ✅ 新 virgin cohort（未在 E0/E1/E2 消费）
- ✅ verifier 通过
- ✅ Operator 静态前提满足
- ✅ 公开 Context 与 Source 有重叠
- ❌ 不先看 gain 再选 Target

#### 两臂装置

```python
arms = {
    "A5": {
        "source_memory": frozen_source_pack,
        "runtime_prior_slot": True,  # 双槽
        "backend": AgictoChatCompletionsBackend(model="gpt-5.6-luna"),
        "B_total": 3,  # 累计上限（够 Slow 链）
    },
    "A3": {
        "source_memory": None,
        "runtime_prior_slot": False,
        "backend": AgictoChatCompletionsBackend(model="gpt-5.6-luna"),
        "B_total": 3,
    },
}
```

**相同约束**：
- ✅ 相同 Candidate DSL（Operator contracts）
- ✅ 相同 TaskContext
- ✅ 相同 stop rule（第一个正向 winner 或预算耗尽）
- ✅ 相同 Slow 触发规则（第一个 material failure）
- ✅ 相同 delayed opening rule
- ✅ 累计 B_total（不是每轮重置）

#### 主要 Estimand（trajectory-level）

```python
def feedback_to_reliable_local_skill(trajectory: list[RoundResult]) -> int | None:
    """形成可靠 Local Skill 所需的累计 Target Support receipts。
    
    可靠 Local Skill 定义：
    1. delayed 批准（approved_skill_id 非空）
    2. 下一轮检索（next_round_skill_retrieved = True）
    3. verify 进池（next_round_skill_verified = True）
    4. 被探测或申请 Support（next_round_skill_probed = True）
    5. 获得执行权（next_round_authorized_deployment 非空）
    6. removal 对照（单独实验确认）
    
    Returns:
        累计消耗的 Target Support receipts（含 Slow replay）
        或 None（未形成可靠 Skill）
    """
    for i, round in enumerate(trajectory):
        if round.approved_skill_id:
            # 找到批准轮次
            if i + 1 < len(trajectory):
                next_round = trajectory[i + 1]
                if (next_round.next_round_skill_retrieved
                    and next_round.next_round_skill_verified
                    and next_round.next_round_skill_probed
                    and next_round.authorized_deployment):
                    # 计算累计 receipts
                    total_receipts = sum(
                        r.target_support_receipts_used 
                        for r in trajectory[:i+2]
                    )
                    return total_receipts
    return None
```

**两个安全门**：
```python
# 安全门 1：harm 不更差
a5_harm = sum(r.harm_count for r in a5_trajectory)
a3_harm = sum(r.harm_count for r in a3_trajectory)
assert a5_harm <= a3_harm

# 安全门 2：delayed 不显著更差
M = 0.005  # MATERIAL_THRESHOLD
a5_delayed = a5_trajectory[-1].delayed_utility or 0.0
a3_delayed = a3_trajectory[-1].delayed_utility or 0.0
assert a5_delayed >= a3_delayed - M
```

#### 判定标准

| Verdict | 条件 | 可 Claim | 下一步 |
|---------|------|----------|--------|
| **TRANSFER_CASE_PASS** | A5 更少 receipts 形成可靠 Skill<br>且 harm ≤ A3<br>且 delayed ≥ A3 - M | 单 Target 的 mechanism case | 运行 Target 2 |
| **SAFETY_SIGNAL_ONLY** | receipts 相同但 harm 更低 | Memory 降低风险 | 可选 Target 2 |
| **NO_SIGNAL** | Memory 被消费，但指标无材料差异 | 当前 Context resolution 下价值未建立 | 接受负结论 |
| **NEGATIVE_TRANSFER** | harm 更高或 delayed 更差 | Memory 有害 | 转 development 诊断 |
| **CONTENT_INCONCLUSIVE** | Source/Target 无可迁移内容 | 不是方法问题 | 换 Source/Target pair |

#### 验收标准（E3 Target 1）
- [ ] memory_resolution_status: A5="rendered", A3="no_memory"
- [ ] 累计预算 ≤3（两臂相同）
- [ ] delayed 只统计 winner
- [ ] chosen-first 顺序
- [ ] verdict 可判定（非 INCONCLUSIVE）

### E3 Target 2（仅当 Target 1 PASS）

#### 前置条件
Target 1 = TRANSFER_CASE_PASS

#### 装置
- ✅ 不同 Dataset（例如：Target 1 = KDD → Target 2 = Monash）
- ✅ 相同 Source pack（不修改）
- ✅ 相同装置（不修改 model / B_total / stop rule / Context filter）
- ❌ 禁止修改 Prompt / 调 Scope / 换 Operator

#### 综合判定

| 结果组合 | 最终 Claim |
|---------|-----------|
| Target 1 PASS + Target 2 PASS | **跨数据集 Memory 价值（2 案例）** |
| Target 1 PASS + Target 2 NO_SIGNAL | 保守结论：单 Target mechanism case |
| 任一 NEGATIVE | 诊断 first fault，转 development |

---

## 七、关键指标对比（修正版）

### 错误指标（之前版本）

❌ `target_support_receipts_used + slow_replay_receipts_used`（重复计费）  
❌ 单轮 `RoundResult` 函数（Skill 是 trajectory-level）  
❌ B_total=2（不够 Slow 链）  
❌ "至少 3 条 POSITIVE"（容易变成人工挑选）

### 正确指标（修订版）

✅ **主指标**：`feedback_to_reliable_local_skill(trajectory)`（累计 receipts）  
✅ **安全门**：`harm(trajectory)` 和 `delayed_utility(final_round)`  
✅ **B_total=3**：够 Fast-positive + Slow-repair 两条路线  
✅ **Source pack**：收纳自然 trajectory 的全部合法 Episode  

---

## 八、工作量估算

| 阶段 | 工作量 | 关键交付物 |
|------|--------|-----------|
| **P0** | 1-2 天 | 三臂诊断 + 判定报告 |
| **P1** | 1-2 天 | 修复代码（~30-50 行）+ 纵向集成重跑 |
| **P2** | 0.5-1 天 | removal 对照实验 |
| **E3 准备** | 2-3 天 | 冻结 Source + Target 1（outcome-blind）|
| **E3 Target 1** | 5-7 天 | 运行 + 判定 + 报告 |
| **E3 Target 2** | 5-7 天（可选）| 仅当 Target 1 PASS |

**总工期**：
- P0-P2: **2-3 周**
- E3 Target 1: **1-2 周**
- E3 Target 2: **1-2 周**（可选）

---

## 九、实验纪律约束（零违反）

### P0-P2 阶段
1. ✅ **零新数据**：只用已暴露 KDD T117
2. ✅ **单因素实验**：P0 三臂只改一个变量
3. ✅ **不猜答案**：运行 P0 前不预先选择修复路径
4. ✅ **不重跑挑答案**：LLM 方差下不因失败重跑直到成功
5. ✅ **修复最小**：不新增 Schema / Binding DSL / Controller
6. ✅ **回归检查**：P3（8/8）+ E2（5/5）不破坏

### E3 阶段
1. ✅ **Source pack 在看到 Target 之前冻结**
2. ✅ **Target 在看到 outcome 之前冻结**
3. ✅ **两臂使用相同 Candidate DSL**
4. ✅ **累计 B_total（不是每轮重置）**
5. ✅ **单 Target PASS 不 claim cross-domain**
6. ✅ **不在同一 Target 上调装置后重新声称 fresh**
7. ✅ **NO_SIGNAL 是合法负结论**

---

## 十、风险与应对

### 风险 1: P0 诊断 A/B/C 都 reject

**应对**：
- 记录 exact verifier rejection_reason
- 单独检查 Scope / Risk / guard 语义
- 可能存在第三个未知因素（需要新的单因素实验）

### 风险 2: P1 C5 通过但 C6 或 C7 失败

**应对**：
- C6 失败 → removal 对照问题，转 P2 单独诊断
- C7 失败 → 预算统计问题，检查 online_loop 逻辑
- 不因一个 check 失败回退全部修复

### 风险 3: E3 Target 1 = NO_SIGNAL

**应对**：
- 区分三种 NO_SIGNAL：
  1. injection_failed → 装置问题（修复后重跑）
  2. 无可迁移内容 → CONTENT_INCONCLUSIVE（换 Target pair）
  3. 真实无收益 → 接受负结论（不调装置）
- 不在同一 Target 上调 Context filter 后重新声称 fresh

### 风险 4: E3 Target 1 = NEGATIVE_TRANSFER

**应对**：
- 不调整后重跑 fresh
- 转 development 诊断：
  - Source Memory 权限过强？
  - Context 误匹配？
  - Skill Scope 过宽？
- 形成新 method version 后，在新 Target 上验证

---

## 十一、成功后的 Claim 边界

### 如果 P0-P2 全通过
✅ **可以 Claim**：
- "Target-local Skill 完整生命周期成立"
- "Skill retrieval + verify + adoption + removal 闭环工作"
- "Program Binding evolution 机制验证"

❌ **不能 Claim**：
- ❌ "跨域迁移能力"（需 E3）

### 如果 E3 Target 1 PASS
✅ **可以 Claim**：
- "在单个 Target case 上观察到 Source Experience benefit"
- "双槽 + 真实 LLM + Runtime 选择链工作"

❌ **不能 Claim**：
- ❌ "跨数据集迁移能力"（需 Target 2）
- ❌ "通用跨域能力"（只有 2 案例）

### 如果 E3 Target 1+2 都 PASS
✅ **可以 Claim**：
- "在两个预注册的跨数据集 Target case 中观察到 Source Experience benefit"
- "Target-local + 跨数据集 Memory 价值（2 案例）"

❌ **不能 Claim**：
- ❌ "一般性跨域迁移能力"（需更多 Target）
- ❌ "Shared Capability"（需稳定复现）
- ❌ "六个 Surface 都已自然行动化"（只有 Program/Skill）

---

## 十二、立即行动（本周）

### 今天：编写 P0 诊断脚本

```bash
# 创建 P0 实验脚本
touch evaluation/functional/run_v1_p0_skill_binding_diagnostic.py

# 实现三臂逻辑
# - Arm A: frozen + LLM inspect
# - Arm B: rebind + LLM inspect
# - Arm C: frozen + full window
```

### 明天：运行 P0 并判定

```bash
# 运行 P0 三臂诊断（零新数据 / 零 LLM）
python -m evaluation.functional.run_v1_p0_skill_binding_diagnostic \
  --tag p0_diagnostic

# 期望：A=reject, B 或 C 至少一个 pass
```

### 本周末：应用修复并重跑 P1

```bash
# 根据 P0 判定结果应用修复（~30-50 行）
# 重跑纵向集成（真实 LLM）
python -m evaluation.functional.run_v1_integration_vertical_loop \
  --tag p1_rebind_fix \
  --model gpt-5.6-luna

# 期望：C1-C7 全过
```

---

**审核者签字**: ___________________  
**日期**: 2026-08-12  
**版本**: v2.0（修订版，整合本地 agent 审核意见）
