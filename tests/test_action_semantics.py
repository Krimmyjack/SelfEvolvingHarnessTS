"""P0 契约测试：v_none 语义三拆（policy/action_semantics.py）。

已确证事实（BUILD.md 2026-07-08 review + run_main_table._VARIANT_SPECS）：
`v_none` = ["impute_linear"] 单步链 = impute-linear baseline，**不是 strict raw**。
本模块把这一语义显式化：v_raw_identity（严格恒等，新）/ v_impute_linear（v_none 的
canonical 语义名）/ v_ledger_baseline（历史台账口径别名）。冻结 menu v1 不注入新动作。
"""
import numpy as np

from SelfEvolvingHarnessTS.policy.action_semantics import (
    RAW_SEMANTICS_NOTE,
    V_IMPUTE_LINEAR,
    V_LEDGER_BASELINE,
    V_RAW_IDENTITY,
    raw_identity_action_spec,
    semantic_action_id,
)


def test_v_none_is_impute_linear_not_strict_raw():
    assert semantic_action_id("v_none") == V_IMPUTE_LINEAR
    assert semantic_action_id("v_none") != V_RAW_IDENTITY


def test_ledger_baseline_alias_maps_to_impute_linear():
    assert semantic_action_id(V_LEDGER_BASELINE) == V_IMPUTE_LINEAR


def test_canonical_ids_map_to_themselves():
    assert semantic_action_id(V_RAW_IDENTITY) == V_RAW_IDENTITY
    assert semantic_action_id(V_IMPUTE_LINEAR) == V_IMPUTE_LINEAR
    assert semantic_action_id("v_median") == "v_median"
    assert semantic_action_id("f0_median_w25") == "f0_median_w25"


def test_migration_note_documents_the_gain_vs_raw_caveat():
    # 历史 gain_vs_raw / regret-vs-raw 全部以 impute-linear baseline 为参照，
    # 论文与新实验不得再混称 strict raw。
    assert "impute_linear" in RAW_SEMANTICS_NOTE
    assert "v_raw_identity" in RAW_SEMANTICS_NOTE


def test_raw_identity_spec_is_empty_and_task_universal():
    spec = raw_identity_action_spec()
    assert spec.action_id == V_RAW_IDENTITY
    assert spec.steps == ()
    assert set(spec.task_constraints) == {"forecast", "classification", "anomaly_detection"}


def test_raw_identity_compiles_to_empty_program_and_executes_identity():
    from SelfEvolvingHarnessTS.policy.action_spec import ActionCompiler
    from SelfEvolvingHarnessTS.sandbox.executor import run_pipeline

    spec = raw_identity_action_spec()
    program = ActionCompiler().to_program(
        spec, {"task": {"type": "anomaly_detection"}, "pattern": {}, "cell_id": "c"}
    )
    assert program.op_names() == []

    x = np.array([1.0, np.nan, 3.0, -2.5])
    result = run_pipeline(program.as_pairs(), x)
    assert result.ok
    assert result.artifact.shape == x.shape
    assert np.isnan(result.artifact[1])            # strict raw：缺失不补
    assert result.artifact[0] == 1.0
    assert result.artifact[2] == 3.0
    assert result.artifact[3] == -2.5


def test_frozen_menu_v1_not_mutated_by_semantics_module():
    from SelfEvolvingHarnessTS.policy.action_spec import action_menu_v1

    menu = action_menu_v1()
    assert V_RAW_IDENTITY not in menu              # 冻结 menu 不原地扩（改动作集=新 SHA）
    assert "v_none" in menu
    assert [s.op for s in menu.actions["v_none"].steps] == ["impute_linear"]
