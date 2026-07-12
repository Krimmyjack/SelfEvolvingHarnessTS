"""policy/action_semantics.py — raw 语义三拆（P0 卫生包，Final_Plan_CodeAgentFirst_2026-07-09 §P0）。

已确证事实（run_main_table._VARIANT_SPECS + BUILD.md 2026-07-08 review）：
menu v1 的 `v_none` = ["impute_linear"] 单步链 = **impute-linear baseline，不是 strict raw**。
历史台账/结果里所有 "raw"、`gain_vs_raw`、`regret vs raw` 的参照物都是这个 baseline。

三个 canonical 语义名（migration note，一次说清、此后不得再混）：

  v_raw_identity    严格恒等：空程序、缺失不补、观测不动。menu v1 里**不存在**；由本模块
                    的 raw_identity_action_spec() 提供（空 steps → ActionCompiler 编译为
                    空 Program → 执行按构造恒等）。deployment 的"不作为"真值。
  v_impute_linear   `v_none` 的 canonical 语义名（仅线性插补，零平滑）。凡引用历史结果，
                    应称 "impute-linear baseline"。
  v_ledger_baseline 历史台账口径别名：指历史 L_test/OOF 矩阵中记为 `v_none` 的那一列，
                    语义 ≡ v_impute_linear；只用于离线台账对照，不得进入 deployment 主张。

纪律：冻结面（menu v1、frozen_arms、历史 records/L_test）**不迁移、不重命名**——改动作集/
改语义 = 新 SHA。新代码/新实验按本模块语义名声明参照物；escalation 的 raw fallback 目前
落在 `v_none`（= impute-linear baseline），若某任务要求 strict raw fallback，须显式改用
v_raw_identity 并在 manifest 里声明。
"""
from __future__ import annotations

from .action_spec import ActionSpec, _task_constraints

V_RAW_IDENTITY = "v_raw_identity"
V_IMPUTE_LINEAR = "v_impute_linear"
V_LEDGER_BASELINE = "v_ledger_baseline"

# 语义别名表（只解析语义身份，不改任何冻结 artifact）
SEMANTIC_ALIASES = {
    "v_none": V_IMPUTE_LINEAR,
    V_LEDGER_BASELINE: V_IMPUTE_LINEAR,
}

RAW_SEMANTICS_NOTE = (
    "historical ledgers/results record `v_none` = [impute_linear] one-step chain "
    "(impute-linear baseline), NOT strict raw; every historical gain_vs_raw / regret-vs-raw "
    "is measured against that baseline. strict raw = v_raw_identity (empty program, new in P0); "
    "v_ledger_baseline is the documentation alias for the historical `v_none` column and must "
    "not enter deployment claims."
)


def semantic_action_id(action_id: str) -> str:
    """action_id → canonical 语义名（未列入别名表的 ID 语义即自身）。"""
    return SEMANTIC_ALIASES.get(action_id, action_id)


def raw_identity_action_spec() -> ActionSpec:
    """strict raw：空 steps 的 ActionSpec。

    _task_constraints([]) = 全任务交集 = 三任务全允许（不作为在任何任务下都合法）；
    ActionCompiler 对空 steps 编译为空 Program（不走 compose 的 heuristic 路径——那会
    静默合成插补链，正是本模块要消除的语义漂移）。
    """
    return ActionSpec(
        action_id=V_RAW_IDENTITY,
        steps=(),
        task_constraints=_task_constraints([]),
        model_constraints=None,
        provenance={"source": "policy.action_semantics", "semantics": "strict_raw_identity"},
    )
