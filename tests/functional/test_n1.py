"""N1 ACTION_EVIDENCE（2026-08-15 裁决）纯函数测试——不触碰数据文件。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402

M = runner.M


def test_status_from_counts():
    f = runner._n1_status_from_counts
    assert f({"552": [0, 0], "504": [0], "456": [0, 0]}) == "INERT"
    assert f({"552": [0, 3], "504": [0], "456": [0]}) == "ACTED"
    assert f({"552": None, "504": [0], "456": [0]}) == "UNKNOWN"
    assert f({}) == "UNKNOWN"
    # 空列表不会来自实现（实现以 None 表示不可计算，见 _n1_window_change_counts）


def test_resolve_acted():
    f = runner._n1_resolve_acted
    assert f([0.1, 0.2, 0.05]) == "BENEFICIAL"
    assert f([-0.1, 0.001, -0.02]) == "HARMFUL"
    assert f([0.3, -0.2, 0.0]) == "CONFLICT"
    assert f([0.001, -0.002, 0.0]) == "ACTED_NEUTRAL"
    assert f([0.0, 0.0, 0.0]) == "ACTED_NEUTRAL"
    assert f([M, -M]) == "BENEFICIAL"          # 边界值：≥M 与 <-M 严格
    assert f([-M]) == "ACTED_NEUTRAL"           # -M 不算负（< -M 才算）


def _row(key, series, origin, status, cur):
    return {"key": key, "series": series, "origin": origin,
            "historical_status_prevalence": status,
            "current_changed_total": cur,
            "current_action_active": bool(cur),
            "novel_action": status in ("INERT", "UNKNOWN") and bool(cur)}


def test_select_cases_ok_path():
    rows = [
        _row("T10@600", "T10", 600, "INERT", 145),   # 已消耗发现案例
        _row("T1@600", "T1", 600, "INERT", 43),
        _row("T101@888", "T101", 888, "INERT", 12),
        _row("T100@792", "T100", 792, "ACTED", 98),
    ]
    sel = runner._n1_select_cases(rows)
    assert sel["premise"] == "OK"
    assert sel["dev_case"] == "T1@600"          # 字典序第一个（T10@600 被排除）
    assert sel["validation_case"] == "T101@888"  # 不同 series 的第一个
    assert "T10@600" not in sel["novel_cases"]


def test_select_cases_too_rare_variants():
    assert runner._n1_select_cases([])["premise"] == "NONE"
    rows = [_row("T1@600", "T1", 600, "INERT", 43),
            _row("T1@792", "T1", 792, "INERT", 89)]
    sel = runner._n1_select_cases(rows)
    assert sel["premise"] == "SINGLE_SERIES_ONLY"
    assert sel["verdict"] == "ACTION_EVIDENCE_PREMISE_TOO_RARE"


def test_select_cases_acted_current_inert_not_novel():
    rows = [_row("T1@600", "T1", 600, "ACTED", 0)]   # 当前不激活
    assert runner._n1_select_cases(rows)["premise"] == "NONE"
