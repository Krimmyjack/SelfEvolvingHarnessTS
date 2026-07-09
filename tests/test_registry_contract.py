"""tests/test_registry_contract.py — S0.7-6 Registry 契约 + alias 回归测试（A-29e）。

守卫内容：①canonical 正名 + 旧 ID alias 解析（fill_gaps/impute_kalman/kalman_filter）；
②旧 artifact/模板按旧名重放不破坏（executor + compose 双路径）且 provenance 同录
requested/canonical；③契约元数据字段齐全；④anomaly 的 usable_ops 物理禁 smoothing/destructive；
⑤harness 配置面只含 canonical（alias 不进 active_operators）。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.operators import _provenance as prov
from SelfEvolvingHarnessTS.operators.registry import (
    ALIASES, OPERATOR_METADATA, OPERATOR_NAMES, TOOL_REGISTRY, canonicalize, get_operator)
from SelfEvolvingHarnessTS.harness import HarnessState
from SelfEvolvingHarnessTS.fast_path import usable_ops
from SelfEvolvingHarnessTS.sandbox import run_pipeline

_CONTRACT_FIELDS = ("allowed_tasks", "destructive", "preserves_observed", "reversible",
                    "changes_target_space", "requires_dependency", "fallback_policy")


# ── ① alias 解析 ───────────────────────────────────────────────────────────
def test_alias_map_frozen():
    assert ALIASES == {"fill_gaps": "impute_linear",
                       "impute_kalman": "impute_ema",
                       "kalman_filter": "smooth_ema"}


@pytest.mark.parametrize("old,canon", sorted(ALIASES.items()))
def test_alias_resolves_to_canonical_fn(old, canon):
    assert canonicalize(old) == canon
    assert get_operator(old) is TOOL_REGISTRY[canon]
    assert old not in OPERATOR_NAMES and canon in OPERATOR_NAMES   # canonical-only 名单


def test_get_operator_records_requested_and_canonical():
    prov.start_recording()
    try:
        get_operator("kalman_filter")
        ledger = prov.get_ledger()
    finally:
        prov.stop_recording()
    assert {"requested": "kalman_filter", "effective": "smooth_ema", "reason": "compat_alias"} in ledger


def test_unknown_operator_raises():
    with pytest.raises(KeyError):
        get_operator("nonexistent_op")


# ── ② 旧 artifact 重放：executor 按旧名执行 + trace 同录 canonical ─────────
def test_executor_replays_old_ids():
    x = np.linspace(0.0, 1.0, 64)
    x[10] = np.nan
    prov.start_recording()
    try:
        res = run_pipeline([("fill_gaps", {}), ("kalman_filter", {"alpha": 0.3})], x)
    finally:
        prov.stop_recording()
    assert res.ok and res.artifact.shape == (64,)
    assert res.trace[0]["op"] == "fill_gaps" and res.trace[0]["canonical"] == "impute_linear"
    assert res.trace[1]["op"] == "kalman_filter" and res.trace[1]["canonical"] == "smooth_ema"
    reasons = [e["reason"] for e in prov.get_ledger()]
    assert reasons.count("compat_alias") >= 2


def test_old_template_preferred_ops_still_compose():
    """旧模板 preferred_ops 用旧 ID → _first_usable canonical 化后仍可用。"""
    from SelfEvolvingHarnessTS.fast_path.compose import _first_usable
    h = HarnessState.from_minimal()
    assert _first_usable(["fill_gaps"], h, banned=set()) == "impute_linear"
    assert _first_usable(["impute_kalman"], h, banned=set()) == "impute_ema"
    # canonical 被 ban 时，旧名同样被拦（ban 按 canonical 归一）
    assert _first_usable(["fill_gaps"], h, banned={"impute_linear"}) == ""


# ── ③ 契约元数据齐全 ──────────────────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(OPERATOR_NAMES))
def test_contract_fields_complete(name):
    meta = OPERATOR_METADATA[name]
    for f in _CONTRACT_FIELDS:
        assert f in meta, f"{name} 缺契约字段 {f}（新算子必须带完整契约入池，E-3.3 前置）"
    assert isinstance(meta["allowed_tasks"], tuple) and len(meta["allowed_tasks"]) >= 1


def test_imputers_declare_preserves_observed():
    for name in ("impute_linear", "impute_fft", "impute_ema", "period_complete"):
        assert OPERATOR_METADATA[name]["preserves_observed"], name


# ── ④ anomaly 物理过滤 ────────────────────────────────────────────────────
def test_anomaly_usable_ops_bans_smoothing_and_destructive():
    h = HarnessState.from_minimal()
    ops = set(usable_ops(h, "anomaly_detection"))
    assert ops, "anomaly usable_ops 不应为空"
    for op in ops:
        meta = OPERATOR_METADATA[op]
        assert not meta["destructive"], f"destructive 算子 {op} 泄入 anomaly"
        assert "smoothing" not in meta["tags"], f"smoothing 算子 {op} 泄入 anomaly"
        assert "anomaly_detection" in meta["allowed_tasks"]
    # 插补类必须保留（anomaly 也要杀 NaN）
    assert "impute_linear" in ops


def test_forecast_usable_ops_keeps_smoothing():
    h = HarnessState.from_minimal()
    ops = set(usable_ops(h, "forecast"))
    assert {"denoise_median", "denoise_savgol", "winsorize"} <= ops   # forecast 不受物理过滤影响


def test_usable_ops_excludes_aliases():
    h = HarnessState.from_minimal()
    for task in ("forecast", "classification", "anomaly_detection"):
        assert not (set(usable_ops(h, task)) & set(ALIASES)), "alias 不应出现在 usable_ops"


# ── ⑤ harness 配置面只含 canonical ─────────────────────────────────────────
def test_harness_registry_canonical_only():
    h = HarnessState.from_minimal()
    assert not (set(h.l2.active_operators) & set(ALIASES))
    assert not (set(h.l2.operator_registry) & set(ALIASES))
    assert {"impute_ema", "smooth_ema"} <= set(h.l2.active_operators)


def test_alias_metadata_marked():
    for old, canon in ALIASES.items():
        m = OPERATOR_METADATA[old]
        assert m["is_alias"] and m["alias_of"] == canon
