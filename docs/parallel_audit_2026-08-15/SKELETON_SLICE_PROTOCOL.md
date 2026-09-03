> **状态：历史占位，已被 `REVISED_DIRECTION.md` 与 `RETRIEVAL_SLICE_PREREG_SKELETON.md` 接管。**
> 冲突时以修订方向为准；本文件不自动升级为 Scope Rule。

# 参数化纵向切片空壳协议（执行者稿）

本文件只定义“结构 Pattern 实验有结果后，纵向切片怎么填”。不冻结、不运行、
不预选 workflow family / action / trigger。

## 1. 参数槽
```text
trigger            = LOCAL_PATTERN_EXPERIMENT_RESULT
workflow_family    = TO_BE_DECIDED
surface            = TO_BE_DECIDED
action             = TO_BE_DECIDED  # REUSE | MODIFY | ADD | ABSTAIN | REQUEST_OBSERVATION
first_fault_family = TO_BE_DECIDED  # 六 family 之一；UNIDENTIFIABLE/INSTRUMENT_BLOCKED 不得切片
heldout_contexts   = TO_BE_DECIDED  # 必须与 fit 证据 disjoint
budget             = TO_BE_DECIDED
outcome_exposure   = TO_BE_DECIDED  # 首选 ZERO_NEW_OUTCOME；fresh 必须单独批准
```

## 2. 前置条件（全部满足才允许从空壳转为预注册）
1. 本地结构 Pattern 实验给出可观察 trigger 候选，且非 `RESIDUAL_DOMINATED`；
2. 存在至少跨 2 个 series 的 fault-conditioned batch；
3. batch 内存在一个共同 fault family（六类之一）；
4. 存在共同 replacement headroom（零 LLM 检查）；
5. 有 matched positive/conflict 保护组。

## 3. 纵向切片流程（占位，不运行）
```text
结构 Pattern trigger 冻结
  -> 复用 BSE Rule 字段（surface/workflow_signature/applicability/
     unknown_policy/authority/requires_target_support/evidence）
  -> Slow 动作空间 = TO_BE_DECIDED；BSE 式 P1/P2/abstain 只是候选流程之一
  -> held-out 机械 replay（H0 vs H1，pre-probe 固定，post-probe winner）
  -> delayed 单侧 harm veto
  -> removal delta 判据
  -> 通过则 LOCAL_DRAFT；delayed 中性不扩权；delayed < -M 撤销/收缩
```

## 4. 明确禁止
- 禁止在 trigger 未知时填 winsorize / outlier_mad / hampel 等 workflow；
- 禁止预写阈值；
- 禁止新增平台、Schema、Gate、SHA；
- 禁止消耗 fresh Outcome；
- 禁止把 GRID0 已否定的小样本 C1 重新包装为本切片的前置。

## 5. 完成定义
结构 Pattern 结果出现后，本文件只允许被替换为一份**新的冻结协议**，名称格式：
`docs/parallel_audit_2026-08-15/SLICE_FROZEN_<date>_<fault_family>.md`。
