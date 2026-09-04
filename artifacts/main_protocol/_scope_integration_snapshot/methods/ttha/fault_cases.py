"""methods/ttha/fault_cases.py — 受限五类错误选择题 + 轻量 Problem
Case reconciliation（S0，2026-08-13——用户裁决"受限 Case 驱动的 Batch
Harness Evolution"；老师建议：先跑通全流程+可视化；LJJ 建议：固定
5 类选择题，不做任意类型任意范围归因）。

设计纪律（用户裁决硬约束——违反即偏离项目纪律）：
  - 纯确定性模块——无 LLM、无 Hash/SHA、无 Schema/Ledger/向量检索/
    通用平台；
  - case_id = 普通顺序 ID（case-0001...）——两个 Case 是否相同由显式
    字段比较判定，不依赖 ID；
  - LLM（S1 起）只答选择题（fault_type + 差异总结）；MATCH / NEW /
    CONFLICT / ABSTAIN 由 Runtime 按字段差异确定性判定——LLM 不自由
    创建 Case；
  - 选项屏蔽：无机械证据的错误类不可选（防强行归因）；
  - NO_ACTIONABLE_FAULT / INSUFFICIENT_EVIDENCE 不是第六/七类错误，
    是防止强行归因的保底通道。

五类（固定枚举——LLM 选择题的题面）：
  TASK_INTERPRETATION_ERROR    Agent 误解 Task/Consumer/Horizon/质量目标
  QUALITY_DIAGNOSIS_ERROR      质量现象判断与可验证事实矛盾
  WORKFLOW_SUPPLY_GAP          候选空间已测：无共同正向替代 + supply 穷尽
  WORKFLOW_DECISION_ERROR      正向候选已被 probe 证实而 Agent 未提出/未选
  SCOPE_MEMORY_RISK_ERROR      Support 正 + delayed 负（时间/范围风险已测）

每类的机械证据条件（选项屏蔽的依据——全部为已测事实）：
  TASK_INTERPRETATION_ERROR : task_contract_conflict 存在（TaskSpec/
      TaskQualityContract 可验证矛盾——如错误 horizon/目标/保留约束）。
      无机械证据时该选项被屏蔽——不允许仅凭 LLM 判断（用户裁决）。
  QUALITY_DIAGNOSIS_ERROR   : diagnosis_contradiction 存在（诊断结论与
      可验证事实矛盾）。flip 不可分**不是**该类（用户裁决：不可分 →
      INSUFFICIENT_EVIDENCE / NO_ACTIONABLE_FAULT）。
  WORKFLOW_SUPPLY_GAP       : 全部候选 headroom 已测失败 AND
      supply_exhausted（候选空间穷举记录存在）。
  WORKFLOW_DECISION_ERROR   : winner_probed 存在（正向候选已被 probe
      证实 gain ≥ M）AND agent 未提出/未选择该候选。
  SCOPE_MEMORY_RISK_ERROR   : support_positive 已测 AND delayed_negative
      已测。

Reconciliation 规则（确定性——调用方先按 fault_type 硬过滤）：
  决策相关字段 = (task_consumer, workflow_sig, response_class)。
  1. task_consumer 或 workflow_sig 不同 → NEW_CASE（不同机制/任务）；
  2. 相同且 response_class 相反（NEGATIVE vs CONFLICT/POSITIVE）→
     CONFLICT_WITH_EXISTING（相似 Context 相反 Utility——优先判定）；
  3. 相同 → MATCH_ADD_EVIDENCE（headroom/fix 状态差异与 observable
     context 桶差异**不**单独构成 NEW——作为补充证据积累在 Case 内，
    状态升级由生命周期单独管理）；
  4. 证据不足以判定差异是否决策相关 → ABSTAIN（INSUFFICIENT_EVIDENCE）。
  "文本说法不同"永远不构成 NEW——只有字段级差异才算。

用法：S0 阶段由 bootstrap/测试直接调用；S1 起 LLM 选择题的输出由
classify_group 校验（choice 必须属于可选类或 guards）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

FAULT_TYPES = (
    "TASK_INTERPRETATION_ERROR",
    "QUALITY_DIAGNOSIS_ERROR",
    "WORKFLOW_SUPPLY_GAP",
    "WORKFLOW_DECISION_ERROR",
    "SCOPE_MEMORY_RISK_ERROR",
)
GUARDS = ("NO_ACTIONABLE_FAULT", "INSUFFICIENT_EVIDENCE")
CASE_ACTIONS = ("MATCH_ADD_EVIDENCE", "CONFLICT_WITH_EXISTING",
                "NEW_CASE", "ABSTAIN")
# 决策相关字段（差异集判定的唯一依据）
DECISION_FIELDS = ("task_consumer", "workflow_sig", "response_class")


def selectable_fault_types(evidence: Mapping[str, Any]) -> list[str]:
    """按机械证据计算可选错误类（选项屏蔽）。evidence 包字段（全部
    已测事实——由 runner 从报告机械构造，不得编造）：
      task_contract_conflict : TaskSpec/Contract 可验证矛盾 | None
      diagnosis_contradiction: 诊断结论与可验证事实矛盾 | None
      headroom               : {alt: common_positive} | None（未测）
      supply_exhausted       : bool（候选空间穷举记录）
      winner_probed          : {"op": str, "gain": float} | None
      agent_chosen           : op | None（Agent 实际提出/选择）
      support_positive       : bool | None（组级已测）
      delayed_negative       : bool | None（组级已测）
    返回：可选类列表（无机械证据的类被屏蔽）——空列表 = 只剩 guards。"""
    out: list[str] = []
    if evidence.get("task_contract_conflict"):
        out.append("TASK_INTERPRETATION_ERROR")
    if evidence.get("diagnosis_contradiction"):
        out.append("QUALITY_DIAGNOSIS_ERROR")
    headroom = evidence.get("headroom") or {}
    if (isinstance(headroom, Mapping) and headroom
            and all((v is False) for v in headroom.values())
            and evidence.get("supply_exhausted") is True):
        out.append("WORKFLOW_SUPPLY_GAP")
    winner = evidence.get("winner_probed")
    if isinstance(winner, Mapping) and winner.get("op") \
            and evidence.get("agent_chosen") != winner.get("op"):
        out.append("WORKFLOW_DECISION_ERROR")
    if evidence.get("support_positive") is True \
            and evidence.get("delayed_negative") is True:
        out.append("SCOPE_MEMORY_RISK_ERROR")
    return out


def classify_group(evidence: Mapping[str, Any],
                   choice: str) -> tuple[str, str]:
    """校验 LLM 选择题答案：choice 必须属于可选类或 guards。
    返回 (normalized_choice, error_or_empty)。choice 非法 → 返回
    INSUFFICIENT_EVIDENCE + 违规说明（不静默改答案）。"""
    if choice in FAULT_TYPES:
        allowed = selectable_fault_types(evidence)
        if choice in allowed:
            return choice, ""
        return "INSUFFICIENT_EVIDENCE", (
            f"fault_type {choice} 无机械证据（可选类={allowed}）——"
            "已回退 INSUFFICIENT_EVIDENCE")
    if choice in GUARDS:
        return choice, ""
    return "INSUFFICIENT_EVIDENCE", (
        f"unknown fault_type {choice!r}——已回退 INSUFFICIENT_EVIDENCE")


def default_guard(evidence: Mapping[str, Any]) -> str:
    """可选类为空时的保底通道（确定性）：headroom 已测且全失败 →
    NO_ACTIONABLE_FAULT（动作空间已测空）；否则 INSUFFICIENT_EVIDENCE。"""
    headroom = evidence.get("headroom") or {}
    if (isinstance(headroom, Mapping) and headroom
            and all((v is False) for v in headroom.values())):
        return "NO_ACTIONABLE_FAULT"
    return "INSUFFICIENT_EVIDENCE"


def _opposite(response_class: str) -> set[str]:
    """response_class 相反判定（CONFLICT 优先规则）。"""
    if response_class == "NEGATIVE":
        return {"POSITIVE", "CONFLICT"}
    if response_class == "POSITIVE":
        return {"NEGATIVE"}
    return set()


def _case_workflow_sig(case: Mapping[str, Any]) -> str | None:
    """Case 的 workflow 指纹（顶层字段或 workflow_and_effect 内——
    Case 结构把 workflow 放在 workflow_and_effect.workflow_sig）。"""
    sig = case.get("workflow_sig")
    if sig is None:
        wae = case.get("workflow_and_effect") or {}
        sig = wae.get("workflow_sig") if isinstance(wae, Mapping) else None
    return sig


def reconcile_existing(case: Mapping[str, Any],
                       group_fields: Mapping[str, Any]) -> str:
    """确定性 reconciliation（调用方已按 fault_type 硬过滤）。
    case = 已有 Case 记录；group_fields = 症状组的决策字段包
    {task_consumer, workflow_sig, response_class}（其余维度差异不单独
    构成 NEW——记录为补充证据）。返回 CASE_ACTIONS 之一。"""
    for f in DECISION_FIELDS:
        if f not in group_fields or group_fields[f] is None:
            return "ABSTAIN"  # 证据不足以判定差异
    case_wf = _case_workflow_sig(case)
    if (case.get("task_consumer") != group_fields["task_consumer"]
            or case_wf != group_fields["workflow_sig"]):
        return "NEW_CASE"
    if group_fields["response_class"] in _opposite(
            str(case.get("response_class") or "")):
        return "CONFLICT_WITH_EXISTING"
    if case.get("response_class") == group_fields["response_class"]:
        return "MATCH_ADD_EVIDENCE"
    return "ABSTAIN"


def filter_candidates(cases: Sequence[Mapping[str, Any]],
                      fault_type: str,
                      group_fields: Mapping[str, Any]) -> list[Mapping]:
    """硬过滤：fault_type 相同 AND task_consumer 相同的已有 Case
    （返回序即输入序——最多由调用方取 3）。"""
    return [c for c in cases
            if c.get("fault_type") == fault_type
            and c.get("task_consumer") == group_fields.get("task_consumer")]


__all__ = ["FAULT_TYPES", "GUARDS", "CASE_ACTIONS", "DECISION_FIELDS",
           "selectable_fault_types", "classify_group", "default_guard",
           "reconcile_existing", "filter_candidates"]
