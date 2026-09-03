# E3 Fresh Cross-Domain Memory Value 实验设计

**状态**: E0/E1/E2 已完成，当前在 E3 门口  
**日期**: 2026-08-12  
**前置**: E0_HARDENING_PASS + E2_MEMORY_TWO_SLOT_CONTROL_PASS  

---

## 一、实验目标（单一主问题）

> **在 fresh virgin Target 上，Source Memory（双槽）是否减少形成可靠 Local Skill 所需的 Target Support receipts？**

**不测量**：
- ❌ 完整 Fast Agent 自主生成 Workflow（LLMSelectBackend 是 DSL selector）
- ❌ Shared Capability（需多 Target 稳定复现）
- ❌ 所有 Task/Consumer 迁移（只测 outlier family）
- ❌ 六个 Surface 自然行动化（只测 Program/Skill）

---

## 二、当前已完成工作（E0-E2）

### E0: 评价语义硬化（E0_HARDENING_PASS）

**7 项修正**（`online_loop.py`，零新数据）：
1. ✅ chosen_proposal vs authorized_deployment 分离
2. ✅ first_positive 用合法 Support receipt index
3. ✅ Slow replay 进 actual_probed_programs + harm
4. ✅ delayed gain None 不转 0（保持未评估）
5. ✅ memory_resolution_status 公开（no_memory/rendered/injection_failed）
6. ✅ Slow 调用透传 Operator contracts + TaskContext
7. ✅ current_status 分类修正（restricted/bootstrap/draft/active）

**验收**: 9/9 checks PASS（`run_v1_e0_hardening_check.py`）

### E1: Fresh 自然闭环（NO_NATURAL_FAILURE）

**装置**：
- 预冻结新 KDD cohort E1（20 支 virgin）
- origin 600/792/888
- Fast sealed（force_pool）
- 真实 Slow ≤1 次

**结果**：
- 15 次探测全正向（0.0302/0.2991/0.0661）
- 无 material failure → LLM 0 调用
- **合法负档**：该 cohort 在 600-936 窗口上无失败信号

**含义**：
- 自然失败不是普遍存在
- fresh 闭环需更长轨迹（984/1080）或不同装置
- **不因失败换 cohort**

### E2: Source Memory 双槽（MEMORY_TWO_SLOT_CONTROL_PASS）

**双槽实现**（`SealedProbeBackend`，lines 244-266）：
```python
# propose 阶段：ref1 存在时
candidates = [ref1_op]  # Slot 1: Source prior
if reserve_exploration_slot:
    exploration_op = find_eligible_op(
        skip=[ref1_op, explored, deprioritized]
    )
    if exploration_op:
        candidates.append(exploration_op)  # Slot 2: Target exploration
```

**5/5 验证**（development，已暴露数据）：
- ✅ C1: Source 不能删除探索槽（two 池 3 ops vs hard 池 2 ops）
- ✅ C2: 正例最多优先一个 trial
- ✅ C3: 负例只能降级不能封杀（deprioritized 回退）
- ✅ C4: Target 反馈覆盖 Source 排序（verdict 层：POSITIVE_PRIOR → UNKNOWN）
- ✅ C5: 三臂预算 ≤2 全同

**Caveat**（如实记录）：
- Traffic 臂实际探测系列（T635）因 registry 漂移未被先验暴露
- "零新数据"仅 origin/算子级成立（fresh 求值）
- C4 UNKNOWN 真实机制 = weak→radius 模式切换（非成对中和）

---

## 三、E3 实验设计（修订版）

### 3.1 两臂装置（matched-budget）

```python
arms = {
    "A5-two-slot": {
        "source_memory": frozen_source_pack,
        "backend": LLMSelectBackend(
            reserve_exploration_slot=True,  # 双槽
            model="gpt-5.6-luna",
            temperature=0.0,
        ),
        "B_total": 2,  # 累计上限（不是每轮）
    },
    "A3": {
        "source_memory": None,
        "backend": LLMSelectBackend(
            reserve_exploration_slot=False,  # 无 ref1，无需双槽
            model="gpt-5.6-luna",
            temperature=0.0,
        ),
        "B_total": 2,
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

### 3.2 Source Episode Pack 冻结（outcome-blind）

**选择规则**：
1. ✅ 至少 3 条 POSITIVE Episode
2. ✅ 至少 1 条 Support 通过
3. ✅ 至少 1 条 delayed approved
4. ✅ 所有合法 Episode（positive/negative/conflict/abstain）全部收纳
5. ❌ **不根据 Target 是否匹配筛选**

**候选 Source**：
- KDD outlier_spike（如果有 ≥3 POSITIVE）
- Traffic 正向案例（P3 场景 A）
- GEFCom 正向案例（P3 场景 B）

**冻结格式**：
```json
{
  "source_id": "kdd_outlier_spike_20260812",
  "episodes": [
    {"episode_id": "...", "relation": "POSITIVE", "operator": "outlier_mad", ...},
    {"episode_id": "...", "relation": "NEGATIVE", "operator": "hampel", ...},
    {"episode_id": "...", "relation": "CONFLICT", "operator": "outlier_iqr", ...}
  ],
  "frozen_at": "2026-08-12T10:00:00Z",
  "eligibility": "≥3 POSITIVE, ≥1 Support approved, ≥1 delayed approved",
  "note": "Source pack 在看到 Target 之前冻结，不根据 Target outcome 增删"
}
```

**禁止**：
- ❌ 看到 Target outcome 后筛选 Episode
- ❌ 做 sign-swap（只能用于 development 诊断）
- ❌ 根据 Target Context 优化 Source pack

### 3.3 Target 选择（outcome-blind）

**选择规则**：
1. ✅ 新 virgin cohort（未在 E0/E1/E2 消费）
2. ✅ verifier 通过
3. ✅ Operator 静态前提满足（≥2 legal replacements）
4. ✅ 公开 Context 与 Source 有重叠（例如 outlier family）
5. ✅ 长度 ≥ origin + 2 * horizon

**候选 Target**：
- KDD origin 984/1080 virgin（剩余窗口）
- Monash 新 cohort（weather/rain，outlier family）
- GEFCom 新 cohort（outlier family）

**冻结格式**：
```json
{
  "target_id": "kdd_k3_virgin_20260812",
  "cohort": ["T160", "T161", ..., "T179"],
  "origins": [984, 1080],
  "frozen_at": "2026-08-12T11:00:00Z",
  "eligibility": "virgin + verifier + ≥2 replacements + Context overlap",
  "note": "Target 在看到 outcome 之前冻结"
}
```

**禁止**：
- ❌ 先看 gain 再选 Target
- ❌ 选择"一定会失败"的 cohort
- ❌ 选择"一定会成功"的 cohort
- ❌ 根据 E1 结果选择"有自然失败"的 cohort

### 3.4 主要 Estimand（单一主指标）

```python
def feedback_to_reliable_local_skill(result: RoundResult) -> int | None:
    """形成可靠 Local Skill 所需的 Target Support receipts 数量。
    
    可靠 Local Skill 定义（5 条件）：
    1. delayed 批准（approved_skill_id 非空）
    2. 下一轮检索（next_round_skill_retrieved = True）
    3. 被选择或申请 Support（next_round_skill_chosen = True）
    4. 获得执行权（authorized_deployment 非空）
    5. removal 对照（需要单独实验确认）
    
    Returns:
        累计消耗的 Target Support receipts 数量（含 Slow replay）
        或 None（未形成可靠 Skill）
    """
    if not (
        result.approved_skill_id
        and result.next_round_skill_retrieved
        and result.next_round_skill_chosen
        and result.authorized_deployment
    ):
        return None  # 未形成可靠 Skill
    
    # 返回累计消耗的 Target Support receipts
    return (result.target_support_receipts_used + 
            result.slow_replay_receipts_used)
```

**两个安全门**：
```python
# 安全门 1：harm 不更差
harm_before_recovery = (result.harm_count, result.harm_magnitude)
assert a5_harm <= a3_harm, "A5 harm 更高 → NEGATIVE_TRANSFER"

# 安全门 2：delayed 不显著更差
M = 0.005  # MATERIAL_THRESHOLD
assert a5_delayed >= a3_delayed - M, "A5 delayed 显著更差 → NEGATIVE_TRANSFER"
```

### 3.5 判定标准（单 Target）

| Verdict | 条件 | 可 Claim | 下一步 |
|---------|------|----------|--------|
| **TRANSFER_CASE_PASS** | A5 更少 receipts 形成可靠 Skill<br>且 harm ≤ A3<br>且 delayed ≥ A3 - M | 单 Target 的 mechanism case | 运行 Target 2 |
| **SAFETY_SIGNAL_ONLY** | receipts 相同但 harm 更低 | Memory 降低风险（无效率收益）| 可选 Target 2 |
| **NO_SIGNAL** | Memory 被消费，但指标无材料差异 | 当前 Context resolution 下价值未建立 | 接受负结论 |
| **NEGATIVE_TRANSFER** | harm 更高或 delayed 更差 | Memory 有害 | 转 development 诊断 |
| **CONTENT_INCONCLUSIVE** | Source/Target 无可迁移内容或无 headroom | 不是方法问题 | 换 Source/Target pair |
| **PROTOCOL_INCONCLUSIVE** | 注入/预算/候选池失效 | 装置问题 | 修复后重新运行 |

**关键**：
- 单 Target PASS 不能 claim "cross-domain benefit"
- 至少 **2 个 Target Dataset 同向** 才能 claim 跨域价值

---

## 四、E3 推进时间线

### Week 1, Day 1-2: Gate A 接线检查

**Task 1.1**: 接通 LLMSelectBackend 双槽
```python
# 文件：evaluation/functional/run_v1_e3_gate_a_wiring_fix.py（已创建）
# 修改：LLMSelectBackend.__init__() 暴露 reserve_exploration_slot 参数
# 验证：9/9 checks PASS（干跑，已暴露数据）
```

**Task 1.2**: 确认无回归
```bash
# 重跑 E2 三臂验证（确定性 backend）
python -m evaluation.functional.run_v1_sealed_a5_a3 \
  --tag e3_gate_a_regression \
  --arms "a5_hard,a5_two_slot,a3"

# 期望：5/5 checks 仍 PASS
```

**验收标准**（Gate A）：
- [ ] LLMSelectBackend 支持 reserve_exploration_slot=True
- [ ] 双槽逻辑工作（池大小：A5-two-slot ≥ A5-hard）
- [ ] E2 验证无回归（5/5 仍 PASS）

---

### Week 1, Day 3-5: 冻结 Source + Target 1

**Task 2.1**: 冻结 Source Episode pack（outcome-blind）
```python
# 脚本：evaluation/functional/freeze_source_pack_e3.py
# 输入：已暴露数据的正向案例（KDD/Traffic/GEFCom）
# 输出：artifacts/functional/e3/source_pack_frozen_20260812.json
# 验证：≥3 POSITIVE, ≥1 Support, ≥1 delayed approved
```

**Task 2.2**: 冻结 Target 1（outcome-blind）
```python
# 脚本：evaluation/functional/freeze_target1_e3.py
# 候选：
#   - KDD origin 984 virgin（剩余窗口）
#   - Monash weather/rain outlier family
# 输出：artifacts/functional/e3/target1_frozen_20260812.json
# 验证：virgin + verifier + ≥2 replacements + Context overlap
```

**验收标准**：
- [ ] Source pack 在看到 Target 之前冻结
- [ ] Target 在看到 outcome 之前冻结
- [ ] 所有选择规则可审计（无 outcome 泄漏）

---

### Week 2: 运行 Fresh Target 1

**Task 3.1**: 编写 E3 Target 1 Runner
```python
# 文件：evaluation/functional/run_v1_e3_fresh_target1.py
# 使用：
#   - 统一 online_loop
#   - LLMSelectBackend（reserve_exploration_slot=True/False）
#   - 真实 LLM（gpt-5.6-luna, temp=0）
#   - 累计 B_total=2
```

**Task 3.2**: 运行 E3 Target 1
```bash
python -m evaluation.functional.run_v1_e3_fresh_target1 \
  --source artifacts/functional/e3/source_pack_frozen_20260812.json \
  --target artifacts/functional/e3/target1_frozen_20260812.json \
  --model gpt-5.6-luna \
  --b_total 2

# 输出：artifacts/functional/e3/w2_e3_target1_report.json
```

**Task 3.3**: 判定
```python
# 主指标：feedback_to_reliable_local_skill
a5_receipts = feedback_to_reliable_local_skill(a5_result)
a3_receipts = feedback_to_reliable_local_skill(a3_result)

# 安全门
a5_harm = (a5_result.harm_count, a5_result.harm_magnitude)
a3_harm = (a3_result.harm_count, a3_result.harm_magnitude)
a5_delayed = a5_result.delayed_utility
a3_delayed = a3_result.delayed_utility

# 判定逻辑
if a5_receipts is not None and a3_receipts is not None:
    if a5_receipts < a3_receipts and a5_harm <= a3_harm and a5_delayed >= a3_delayed - M:
        verdict = "TRANSFER_CASE_PASS"
    elif a5_receipts == a3_receipts and a5_harm < a3_harm:
        verdict = "SAFETY_SIGNAL_ONLY"
    else:
        verdict = "NO_SIGNAL"
elif a5_harm > a3_harm or (a5_delayed is not None and a3_delayed is not None and a5_delayed < a3_delayed - M):
    verdict = "NEGATIVE_TRANSFER"
else:
    verdict = "NO_SIGNAL"
```

**验收标准**：
- [ ] memory_resolution_status = "rendered" (A5) / "no_memory" (A3)
- [ ] 累计预算 ≤2（两臂相同）
- [ ] delayed 只统计 winner
- [ ] chosen-first 顺序
- [ ] verdict 可判定（非 INCONCLUSIVE）

---

### Week 3-4: Fresh Target 2（仅当 Target 1 PASS）

**前提**: Target 1 = TRANSFER_CASE_PASS

**Task 4.1**: 冻结 Target 2（不同 Dataset）
```python
# 要求：与 Target 1 不同 Dataset
# 例如：Target 1 = KDD → Target 2 = Monash
# 输出：artifacts/functional/e3/target2_frozen_20260815.json
```

**Task 4.2**: 运行 E3 Target 2
```bash
# 使用：相同 Source pack / 相同装置 / 相同 model / 相同 B_total
# 禁止：修改 Context filter / 修改 Prompt / 修改 stop rule

python -m evaluation.functional.run_v1_e3_fresh_target2 \
  --source artifacts/functional/e3/source_pack_frozen_20260812.json \
  --target artifacts/functional/e3/target2_frozen_20260815.json \
  --model gpt-5.6-luna \
  --b_total 2
```

**Task 4.3**: 综合判定
```
两个 Target 同向（都 PASS）→ 可 claim "跨数据集 Memory 收益（2 案例）"
一个 PASS 一个 NO_SIGNAL → 保守结论"单 Target mechanism case"
任一 NEGATIVE → 诊断 first fault，转 development
```

---

## 五、结果后的分支处理

### 分支 1: Target 1 = TRANSFER_CASE_PASS

```
✅ 运行 Target 2（不同 Dataset）
✅ 不修改装置
✅ Target 2 也 PASS → claim "跨数据集 Memory 价值（2 案例）"
✅ Target 2 = NO_SIGNAL → 保守结论"单 Target mechanism case"
```

### 分支 2: Target 1 = NO_SIGNAL

```
✅ 区分三种 NO_SIGNAL：
   1. LLM 未因果使用 Memory（memory_resolution_status = "injection_failed"）
      → 装置问题，修复后重跑
   2. Source/Target 无可迁移内容（Context 不重叠）
      → CONTENT_INCONCLUSIVE，选择新 Target pair
   3. Memory 被使用、内容重叠、但结果无差异
      → 真实 NO_SIGNAL，接受结论
      
✅ 不在同一 Target 上调 Context filter 后重新声称 fresh
✅ 可以用该 Target 转 development 诊断
```

### 分支 3: Target 1 = NEGATIVE_TRANSFER

```
✅ 不调整后重跑 fresh
✅ 转 development 诊断：
   - Source Memory 权限过强？
   - Context 误匹配？
   - Skill Scope 过宽？
✅ 形成新 method version 后，在新 Target 上验证
```

### 分支 4: Target 无 Program headroom

```
✅ 接受"该 task × defect family × program mechanism 当前无 headroom"
❌ 不继续加 Pattern / 调 Scope / 换 Operator
❌ 不因单 Target 失败放弃方向
```

---

## 六、当前可以/不可以 Claim 的内容

### ✅ 已经可以 Claim（E0-E2 完成）

1. **Target-local Program evolution 机制成立**（P3，8/8）
2. **双槽防止 Source Memory 独占候选供应**（E2，5/5）
3. **Draft 执行权限门工作**（P1，3/3）
4. **Typed Patch + Runtime 选择 + 外部审批链工作**（P1.5，PASS）
5. **持久化和重启语义正确**（P5，PASS）
6. **E0 评价语义硬化完成**

### ❌ 不能 Claim（E3 前）

1. ❌ "Source Memory 减少 Target 试错"（E3 未运行）
2. ❌ "跨域迁移能力"（需 ≥2 Target 同向）
3. ❌ "完整 Fast Agent 自主生成 Workflow"（LLMSelectBackend 是 DSL selector）
4. ❌ "Shared Capability"（需多 Target 稳定复现）
5. ❌ "六个 Surface 都已自然行动化"（只有 Program/Skill 完整）
6. ❌ "fresh 自然闭环已建立"（E1 = NO_NATURAL_FAILURE）

---

## 七、关键约束（零违反）

1. ✅ **Source pack 在看到 Target 之前冻结**
2. ✅ **Target 在看到 outcome 之前冻结**
3. ✅ **两臂使用相同 Candidate DSL**
4. ✅ **累计 B_total（不是每轮重置）**
5. ✅ **delayed 只统计 winner**
6. ✅ **单 Target PASS 不 claim cross-domain**
7. ✅ **不在同一 Target 上调装置后重新声称 fresh**
8. ✅ **NO_SIGNAL 是合法负结论（不强行调出 PASS）**

---

## 八、审查检查清单（Gate A → Target 1 → Target 2）

### Gate A（接线检查）
- [ ] LLMSelectBackend 支持 reserve_exploration_slot
- [ ] 双槽逻辑工作（A5-two-slot 池 ≥ A5-hard 池）
- [ ] E2 验证无回归（5/5 仍 PASS）
- [ ] 9/9 wiring checks PASS

### Target 1（fresh 运行）
- [ ] Source pack outcome-blind 冻结
- [ ] Target outcome-blind 冻结
- [ ] 真实 LLM（gpt-5.6-luna）
- [ ] 累计 B_total=2（不是每轮）
- [ ] memory_resolution_status 正确
- [ ] verdict 可判定

### Target 2（仅当 Target 1 PASS）
- [ ] 不同 Dataset
- [ ] 相同装置（不修改）
- [ ] 综合判定（两 Target 同向 → claim cross-domain）

---

**审核者签字**: ___________________  
**日期**: 2026-08-12
