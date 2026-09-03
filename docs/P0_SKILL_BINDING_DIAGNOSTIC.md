# P0: Skill Parameter Binding vs Inspect Region 诊断

**状态**: 纵向集成 C1-C4/C6-C7 通过，C5 失败  
**日期**: 2026-08-12  
**根因假设**: Skill frozen params 在 R2 verify 拒绝，非因参数失效，而是 inspect region 不匹配  

---

## 一、C5 失败现象（来自 w1_integration_vertical_loop_report.json）

### R1 成功链（6/7 通过）
```json
{
  "A5_r1": {
    "pool": ["identity", "cand_prior_winsorize", "repair_level_shift_local"],
    "chosen": "repair_level_shift_local",  // LLM 自由选择探索候选（非 prior）
    "winner": [{
      "op": "repair_level_shift",
      "params": {
        "estimated_offset": 56.0,
        "region_start_fraction": 0.04671717171717172,  // R1 绑定值
        "region_end_fraction": 0.1717171717171717      // R1 绑定值
      }
    }],
    "support_gain": 0.04733750739266296,  // 正向
    "delayed_utility": 0.09709988099621447,  // 批准
    "approved_skill_id": "fast_winner_repair_level_shift"
  }
}
```

### R2 失败链（C5）
```json
{
  "R2": {
    "retrieved_skill_ids": ["fast_winner_repair_level_shift"],  // ✅ 检索成功
    "skill_retrieved": true,                                    // ✅ view 层找到
    "skill_verified_into_pool": false,                          // ❌ verify 拒绝
    "skill_probed": false,                                      // ❌ 未探测
    "chosen": "level_shift_repair",                             // 选了新 LLM proposal
    "winner": [{
      "op": "repair_level_shift",
      "params": {
        "estimated_offset": 56.0,
        "region_start_fraction": 0.041666666666666664,  // R2 新值（不同于 R1）
        "region_end_fraction": 0.15315315315315314      // R2 新值（不同于 R1）
      }
    }]
  }
}
```

**关键差异**：
- R1 frozen: `[0.0467, 0.1717]`
- R2 fresh:  `[0.0417, 0.1532]`
- Skill 保存了 R1 数值，R2 verify 拒绝

---

## 二、当前诊断结论（V1_SEQUENTIAL_VALIDATION_PLAN.md:740）

用户已确认：

> **归因（如实，经诊断修正）**：C5 失败**不是**绑定参数失效（诊断：R1 参数在 R2 @888 的 verify 全过——preserve/max_modified 全组合 0/108 拒绝；_parse_frozen_steps 解析正常）——最可能机制（PLAUSIBLE）：**R2 真实 LLM 的 inspect 声明区域与 skill 修改区域不匹配 + skill guard `preserve_outside_candidate_region=True` → verify 拒绝**（sealed 装置 inspect 恒 [0,1] 不发生——真实 LLM 的 inspect 输出会变化——确定性装置与真实装置在此行为不同）。

**含义**：
1. ✅ Skill frozen params `[0.0467, 0.1717]` 本身在 R2 context 是合法的
2. ✅ verifier 不是因为参数越界或数值失效拒绝
3. ❌ 但 R2 真实 LLM 的 `inspect` 可能声明了不同的候选区域（例如 `[0.04, 0.15]`）
4. ❌ guard `preserve_outside_candidate_region=True` 要求修改必须在候选区域内
5. ❌ Skill 的修改区域 `[0.0467, 0.1717]` 超出了 R2 LLM 声明的候选区域 → verify 拒绝

---

## 三、两种可能的根因（需要单因素实验区分）

### 假设 A: Skill 应该重新绑定当前 Context 特征值

**机制**：
- Registry 已声明参数绑定规则（`operators/registry.py:164`）：
  ```python
  "public_parameter_bindings": {
      "region_start_fraction": "estimated_region_start_fraction",
      "region_end_fraction": "estimated_region_end_fraction",
      "estimated_offset": "estimated_level_offset",
  }
  ```
- R1 形成 Skill 时，绑定了 R1 context 的特征值
- R2 检索 Skill 时，应该用 R2 context 的当前特征值重新绑定
- 当前实现：直接重放 R1 frozen 数值（`params: {region_start_fraction: 0.0467, ...}`）
- 正确实现：从 R2 public features 读取 `estimated_region_start_fraction` → 重新绑定

**如果这是根因**：
- Skill 学到的不是"在 [0.0467, 0.1717] 修复"
- 而是"在 estimated_region_* 指向的区域修复"
- 修复：retrieval 时机械应用 `public_parameter_bindings`

### 假设 B: Skill verify 应该用全窗口检查区域（而非 LLM inspect 声明）

**机制**：
- R2 LLM 在 `inspect` 阶段声明候选区域（例如 `[0.04, 0.15]`）
- Skill frozen params `[0.0467, 0.1717]` 超出这个声明
- guard `preserve_outside_candidate_region=True` 拒绝
- 但 Skill 是已批准的程序，应该允许在全窗口范围内修复
- 正确语义：Skill verify 用 `[0.0, 1.0]` 作为候选区域，不受 LLM 当前 inspect 限制

**如果这是根因**：
- Skill 绑定参数本身正确
- 问题在于 guard 语义：已批准 Skill 不应被 LLM 当前 inspect 限制
- 修复：Skill verify 时传入全窗口候选区域

---

## 四、P0 单因素诊断实验（三臂）

### 装置（已暴露数据，零 LLM）

**固定**：
- 相同 R2 Request（KDD T117 @888）
- 相同 TaskContext
- 相同 Skill（`fast_winner_repair_level_shift`）
- 相同 Operator（`repair_level_shift`）
- 相同 verifier、Scope、Risk 和 Memory
- 不打开新 Target outcome

**三臂**：

| Arm | Skill 参数来源 | verify 候选区域 |
|-----|---------------|----------------|
| **A: frozen** | R1 frozen 数值 `[0.0467, 0.1717]` | R2 LLM inspect 声明（例如 `[0.04, 0.15]`） |
| **B: rebind** | R2 当前 context 重新绑定（机械应用 `public_parameter_bindings`） | R2 LLM inspect 声明 |
| **C: frozen-full** | R1 frozen 数值 `[0.0467, 0.1717]` | 全窗口 `[0.0, 1.0]`（不受 LLM inspect 限制） |

**观察**：
- exact verifier receipt（pass/reject + rejection_reason）
- skill candidate 是否进入 verified pool
- modified_fraction
- inspected_region vs skill_region
- guard evaluation trace

### 判定矩阵

| 结果组合 | 根因确认 | 修复路径 |
|---------|---------|---------|
| A=reject, B=pass, C=reject | **假设 A 成立**：参数应重新绑定 | retrieval 时机械应用 `public_parameter_bindings` |
| A=reject, B=reject, C=pass | **假设 B 成立**：guard 语义问题 | Skill verify 用全窗口候选区域 |
| A=reject, B=pass, C=pass | **两者都有贡献** | 同时修复 A+B |
| A=pass（不应发生，与现有诊断矛盾） | 当前归因错误 | 重新诊断 |
| A/B/C 都 reject | 还有第三个未知因素 | 记录 verifier rejection_reason，单独检查 |

---

## 五、修复方案（按判定结果）

### 修复 A: Skill retrieval 时重新绑定参数

```python
# 文件：methods/ttha/fast_agent.py 或 method.py 的 Skill retrieval 路径

def _rebind_skill_params(
    skill: dict,
    public_features: dict[str, float],
    operator_metadata: dict,
) -> dict:
    """Skill retrieval 时重新绑定参数（复用 Registry 元数据）。
    
    Args:
        skill: 从 harness view 检索到的 Skill（frozen params）
        public_features: R2 当前 public Context features
        operator_metadata: Operator Registry 的 public_parameter_bindings
    
    Returns:
        重新绑定后的 Skill params
    """
    frozen_steps = skill["program"]["steps"]
    bindings = operator_metadata.get("public_parameter_bindings", {})
    
    if not bindings:
        # 无绑定规则，保持 frozen
        return frozen_steps
    
    rebind_steps = []
    for step in frozen_steps:
        op = step["op"]
        params = dict(step["params"])
        
        # 机械应用绑定规则
        for param_name, feature_name in bindings.items():
            if param_name in params and feature_name in public_features:
                # 用当前 Context 特征值替换 frozen 值
                params[param_name] = public_features[feature_name]
        
        rebind_steps.append({"op": op, "params": params})
    
    return rebind_steps
```

**约束**：
- 不新增 Binding DSL / Schema / Controller
- 复用 Registry 已有 `public_parameter_bindings` 元数据
- 只在 Skill retrieval 时应用，不修改 Skill 存储格式
- frozen params 保留（审计/回滚需要）

### 修复 B: Skill verify 用全窗口候选区域

```python
# 文件：methods/ttha/fast_agent.py 或 verifier 调用点

def _verify_skill_candidate(
    skill: dict,
    request: PrepareRequest,
    context: dict,
    is_skill: bool = True,  # 新增标志
) -> VerifyResult:
    """验证 Skill 候选（已批准 Skill 用全窗口候选区域）。
    
    Args:
        skill: 候选 Skill
        request: prepare request
        context: 当前 Context
        is_skill: True 表示这是已批准 Skill（不是 Agent proposal）
    
    Returns:
        verify 结果（pass/reject + reason）
    """
    if is_skill:
        # 已批准 Skill：用全窗口候选区域（不受 LLM inspect 限制）
        candidate_region = [0.0, 1.0]
    else:
        # Agent proposal：用 LLM inspect 声明的候选区域
        candidate_region = request.inspected_region or [0.0, 1.0]
    
    # 传递给 verifier
    return verifier.verify(
        program=skill["program"],
        candidate_region=candidate_region,
        preserve_outside=True,
    )
```

**约束**：
- 只修改 verify 调用时的候选区域参数
- 不修改 guard 语义本身
- 不修改 Agent proposal 的 verify 流程
- Skill 的 guard 仍然生效（preserve/max_modified）

---

## 六、P0 验收标准（三臂诊断）

### 最小成功条件（任一修复路径通过）

- [ ] A=reject 确认（复现当前失败）
- [ ] B 或 C 至少一个 pass（找到可行修复）
- [ ] 修复路径不新增 Schema / Binding DSL / Controller
- [ ] 修复路径不破坏现有 P0-P5 + E0-E2 验收

### 回归检查

- [ ] 重跑 P3 Operational Self-Evolution Loop（8/8 仍 PASS）
- [ ] 重跑 E2 Memory Two-Slot Control（5/5 仍 PASS）
- [ ] 修复后的 verifier / retrieval 不改变 sealed 装置行为

---

## 七、P0 后的推进顺序

### P0 → P1: 应用修复并重跑纵向集成

```bash
# 假设 P0 确认修复 A（参数重新绑定）

# P1: 应用修复
# 文件：methods/ttha/fast_agent.py 或 method.py
# 修改量：~30 行（_rebind_skill_params + 调用点）

# P1: 重跑纵向集成（已暴露数据，真实 LLM）
python -m evaluation.functional.run_v1_integration_vertical_loop \
  --tag p1_rebind_fix \
  --model gpt-5.6-luna

# 期望：C1-C7 全过
# C5: skill_retrieved=True, skill_verified_into_pool=True, skill_probed=True
```

### P1 → P2: removal 对照（C5 闭合后）

```bash
# 前提：P1 C1-C7 全过

# P2: removal 对照实验
# 确认：移除 Skill 后 Program 行为恢复
# 装置：相同 R2 Request，两臂（with Skill / without Skill）
```

### P2 → E3: fresh Target 1

```bash
# 前提：P2 完整纵向集成 + removal 对照全过

# E3: 冻结 Source + Target 1（outcome-blind）
# 运行：A5 runtime prior + exploration vs A3
# 判定：feedback_to_reliable_local_skill / harm / delayed
```

---

## 八、关键纪律约束

1. ✅ **P0 是单因素诊断**：三臂只改一个变量（params 来源 / 候选区域）
2. ✅ **零新数据**：只用已暴露 KDD T117 R2 @888
3. ✅ **零 LLM**：三臂都用确定性装置（复现/隔离变量）
4. ✅ **不猜答案**：运行 P0 前不预先选择修复路径
5. ✅ **不重跑挑答案**：LLM 方差下不因 C5 失败重跑 P1 直到成功
6. ✅ **如实记录**：A/B/C 三臂结果全部记录，不隐藏意外结果
7. ✅ **修复最小**：不新增 Schema / Binding DSL / Controller / Pattern Graph

---

## 九、P0 实验脚本结构（占位）

```python
# 文件：evaluation/functional/run_v1_p0_skill_binding_diagnostic.py

def run_p0_skill_binding_diagnostic():
    """P0 单因素诊断：Skill frozen params vs inspect region。
    
    三臂：
    A: frozen params + LLM inspect region (当前失败)
    B: rebind params + LLM inspect region (测假设 A)
    C: frozen params + full window region (测假设 B)
    """
    
    # 加载 R2 Request（KDD T117 @888）
    request = load_r2_request()
    
    # 加载 Skill（fast_winner_repair_level_shift）
    skill = load_skill("fast_winner_repair_level_shift")
    
    # 加载 public features（R2 context）
    public_features = {
        "estimated_region_start_fraction": 0.041666666666666664,
        "estimated_region_end_fraction": 0.15315315315315314,
        "estimated_level_offset": 56.0,
    }
    
    # 三臂
    results = {}
    
    # Arm A: frozen + LLM inspect
    results["A_frozen"] = verify_skill(
        skill=skill,
        params_source="frozen",
        candidate_region="llm_inspect",
        request=request,
    )
    
    # Arm B: rebind + LLM inspect
    rebind_params = rebind_skill_params(
        skill=skill,
        public_features=public_features,
        bindings=REGISTRY["repair_level_shift"]["public_parameter_bindings"],
    )
    results["B_rebind"] = verify_skill(
        skill=skill,
        params_source="rebind",
        params=rebind_params,
        candidate_region="llm_inspect",
        request=request,
    )
    
    # Arm C: frozen + full window
    results["C_frozen_full"] = verify_skill(
        skill=skill,
        params_source="frozen",
        candidate_region="full_window",  # [0.0, 1.0]
        request=request,
    )
    
    # 判定
    verdict = judge_p0_results(results)
    
    return {"arms": results, "verdict": verdict}
```

---

**审核者签字**: ___________________  
**日期**: 2026-08-12
