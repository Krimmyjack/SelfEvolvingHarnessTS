"""BSE（Batch 自进化 Scope Rule 纵向切片；方案乙 2026-08-15 裁决）测试。

只测纯函数与协议不变量——不触碰 LLM/数据文件/报告。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402

M = runner.M


# ------------------------------------------------------------ 标签口径
def test_stability_label_classes():
    f = runner._bse_stability_label
    assert f(None, False, None) == "NEUTRAL_OR_UNIDENTIFIED"
    assert f(-M - 0.001, False, None) == "SUPPORT_HARM"
    assert f(M + 0.1, True, 0.05) == "STABLE_POSITIVE"
    assert f(M + 0.1, True, -M) == "STABLE_POSITIVE"          # 边界 ≥ −M
    assert f(M + 0.1, True, -M - 0.001) == "SUPPORT_POSITIVE_DELAYED_NEGATIVE"
    assert f(M + 0.1, False, None) == "NEUTRAL_OR_UNIDENTIFIED"  # 缺 delayed
    assert f(M / 2, False, None) == "NEUTRAL_OR_UNIDENTIFIED"    # no-op
    assert f(-M / 2, False, None) == "NEUTRAL_OR_UNIDENTIFIED"


def test_group_map():
    assert runner._BSE_GROUP["STABLE_POSITIVE"] == "A"
    assert runner._BSE_GROUP["SUPPORT_HARM"] == "B"
    assert runner._BSE_GROUP["SUPPORT_POSITIVE_DELAYED_NEGATIVE"] == "C"


# ------------------------------------------------------------ fit 边界
def _pt(key, group, value):
    return {"key": key, "group": group, "value": value}


def test_fit_boundary_exactly_one_gives_midpoint():
    pts = [_pt("b1", "B", -0.2), _pt("b2", "B", -0.148),
           _pt("a1", "A", -0.111), _pt("a2", "A", 0.3)]
    out = runner._bse_fit_boundary(pts)
    assert out["boundary_count"] == 1
    assert abs(out["tau"] - (-0.148 + -0.111) / 2.0) < 1e-12
    assert out["boundary_pair"] == ["b2", "a1"]
    assert [p["key"] for p in out["sorted_fit"]] == ["b1", "b2", "a1", "a2"]


def test_fit_boundary_multiple_blocks_rule():
    pts = [_pt("b1", "B", -0.2), _pt("a1", "A", -0.1),
           _pt("b2", "B", 0.0), _pt("a2", "A", 0.3)]
    out = runner._bse_fit_boundary(pts)
    assert out["boundary_count"] == 3
    assert out["tau"] is None


def test_fit_boundary_zero_blocks_rule():
    out = runner._bse_fit_boundary([_pt("a1", "A", 0.1), _pt("a2", "A", 0.2)])
    assert out["boundary_count"] == 0 and out["tau"] is None


def test_fit_boundary_equal_values_no_tau():
    pts = [_pt("b1", "B", 0.5), _pt("a1", "A", 0.5)]
    out = runner._bse_fit_boundary(pts)
    assert out["boundary_count"] == 1 and out["tau"] is None


def test_fit_boundary_ignores_none_values():
    pts = [_pt("b1", "B", -0.2), _pt("xx", "A", None), _pt("a1", "A", 0.1)]
    out = runner._bse_fit_boundary(pts)
    assert out["boundary_count"] == 1 and out["tau"] is not None


# ---------------------------------------------------------- blind health
def _obs(v, legal=True):
    return {"value": v, "legality": {"ok": legal}}


def test_blind_health_pass_and_fail():
    good = {"k1": _obs(0.1), "k2": _obs(0.2), "k3": _obs(-0.1), "k4": _obs(0.3)}
    assert runner._bse_blind_health(good)["passed"] is True
    constant = {"k1": _obs(0.5), "k2": _obs(0.5), "k3": _obs(0.5), "k4": _obs(0.5)}
    h = runner._bse_blind_health(constant)
    assert h["passed"] is False and h["non_constant"] is False
    sparse = {"k1": _obs(0.1), "k2": _obs(None), "k3": _obs(None)}
    h2 = runner._bse_blind_health(sparse)
    assert h2["passed"] is False and h2["coverage"] == 1
    assert sorted(h2["unknown_contexts"]) == ["k2", "k3"]
    illegal = {"k1": _obs(0.1, False), "k2": _obs(0.2),
               "k3": _obs(-0.1), "k4": _obs(0.3)}
    assert runner._bse_blind_health(illegal)["passed"] is False


# ------------------------------------------------------------- rule 语义
def _rule(op="ge", tau=-0.13):
    return {"rule_id": runner.BSE_RULE_ID, "surface": "scope",
            "workflow_signature": "outlier_mad",
            "applicability": {"feature": runner.BSE_OBS_FEATURE,
                              "operator": op, "threshold": tau},
            "unknown_policy": "no_prior", "authority": "LOCAL_DRAFT",
            "requires_target_support": True, "slow_approved": False}


def test_rule_fires_and_unknown_never_fires():
    r = _rule("ge", -0.13)
    assert runner._bse_rule_fires(-0.015, r) is True
    assert runner._bse_rule_fires(-0.148, r) is False
    assert runner._bse_rule_fires(None, r) is False       # unknown 不放行
    r2 = _rule("le", -0.13)
    assert runner._bse_rule_fires(-0.2, r2) is True
    assert runner._bse_rule_fires(None, r2) is False


def test_assemble_rule_shape():
    labeled = [
        {"key": "T100@600", "group": "A"},
        {"key": "T1@888", "group": "B"},
        {"key": "T1@792", "group": "C"},
    ]
    id_map = {"T100@600": "usel_t100_600_outlier_mad_positive",
              "T1@888": "usel_t1_888_outlier_mad_conflict",
              "T1@792": "batch1_t1_792_outlier_mad_conflict"}
    rule = runner._bse_assemble_rule("P1", runner.BSE_OBS_FEATURE, -0.1295,
                                     labeled, id_map)
    assert rule["applicability"]["operator"] == "ge"
    assert rule["authority"] == "LOCAL_DRAFT"
    assert rule["requires_target_support"] is True
    assert rule["slow_approved"] is False
    assert rule["unknown_policy"] == "no_prior"
    ev = rule["evidence"]
    assert ev["positive_episode_ids"] == ["usel_t100_600_outlier_mad_positive"]
    assert ev["negative_episode_ids"] == ["usel_t1_888_outlier_mad_conflict"]
    assert ev["conflict_episode_ids"] == ["batch1_t1_792_outlier_mad_conflict"]
    rule2 = runner._bse_assemble_rule("P2", runner.BSE_OBS_FEATURE, 0.0,
                                      labeled, id_map)
    assert rule2["applicability"]["operator"] == "le"


# --------------------------------------------------------- Slow 输出解析
def test_parse_slow_choice():
    ok = runner._bse_parse_slow_choice(
        '前置废话 {"choice": "P1", "rationale": "A 组值高于 B 组"} 后置')
    assert ok["choice"] == "P1" and ok["parse"] == "ok"
    assert runner._bse_parse_slow_choice("我不知道")["choice"] == "abstain"
    assert runner._bse_parse_slow_choice('{"choice": "P3"}')["parse"] == "bad_choice"
    assert runner._bse_parse_slow_choice('{"choice":')["parse"] == "no_json"
    assert runner._bse_parse_slow_choice('{"choice": "P1",}')["parse"] == "bad_json"


# ------------------------------------------------------- PASS 判定（纯）
def _row(key, group, h0_prior, h1_prior, h0_gain=0.0, h1_gain=0.0,
         h0_delayed=0.0, h1_delayed=0.0, h0_neg=0, h1_neg=0):
    def arm(prior, gain, delayed, neg):
        return {"prior": prior, "support_receipts": 1 if prior else 0,
                "negative_probes": neg, "winner": "outlier_mad" if gain >= M
                else "identity", "support_gain": gain,
                "delayed_gain": delayed}
    return {"key": key, "group": group,
            "H0": arm(h0_prior, h0_gain, h0_delayed, h0_neg),
            "H1": arm(h1_prior, h1_gain, h1_delayed, h1_neg)}


def test_pass_evaluation_pass_case():
    rows = [
        _row("T10@888", "A", True, True, h0_gain=0.11, h1_gain=0.11,
             h0_delayed=0.148, h1_delayed=0.148),
        _row("T10@600", "B", True, False, h0_gain=-0.011, h0_neg=1),
        _row("T10@792", "C", True, True, h0_gain=0.19, h1_gain=0.19,
             h0_delayed=-0.015, h1_delayed=-0.015),
    ]
    pe = runner._bse_pass_evaluation(rows)
    assert pe["passed"] is True, pe["failed"]
    assert pe["arms"]["H1"]["support_receipts"] < pe["arms"]["H0"]["support_receipts"]


def test_pass_evaluation_harm_unprotected_fails():
    rows = [
        _row("T10@888", "A", True, True, h0_gain=0.11, h1_gain=0.11),
        _row("T10@600", "B", True, True, h0_gain=-0.011, h1_gain=-0.011,
             h0_neg=1, h1_neg=1),
    ]
    pe = runner._bse_pass_evaluation(rows)
    assert pe["passed"] is False
    assert "harm_auto_priority_blocked" in pe["failed"]
    assert "receipts_or_harm_reduced" in pe["failed"]
    assert "removal_delta_real" in pe["failed"]


def test_pass_evaluation_stable_lost_fails():
    rows = [
        _row("T10@888", "A", True, False),          # 稳定正例失去 prior
        _row("T10@600", "B", True, False, h0_gain=-0.011, h0_neg=1),
    ]
    pe = runner._bse_pass_evaluation(rows)
    assert pe["passed"] is False
    assert "stable_prior_recall" in pe["failed"]


def test_pass_evaluation_c_group_not_counted():
    # C 组即使两臂 delayed harm 也不影响 PASS 判定（但必须存在于披露中）
    rows = [
        _row("T10@888", "A", True, True, h0_gain=0.11, h1_gain=0.11),
        _row("T10@600", "B", True, False, h0_gain=-0.011, h0_neg=1),
        _row("T10@792", "C", True, True, h0_gain=0.19, h1_gain=0.19,
             h0_delayed=-0.015, h1_delayed=-0.015),
        _row("T10@984", "C", True, True, h0_gain=0.03, h1_gain=0.03,
             h0_delayed=-0.049, h1_delayed=-0.049),
    ]
    assert runner._bse_pass_evaluation(rows)["passed"] is True


# ----------------------------------------------------------- capsule 匿名
def test_capsule_anonymized():
    view = [{"anon": "f1", "key": "T100@600", "group": "A", "value": 0.31},
            {"anon": "f2", "key": "T1@888", "group": "B", "value": -0.15}]
    prompt = runner._bse_capsule_prompt(view, runner.BSE_OBS_FEATURE, 0.08)
    for forbidden in ("T100@600", "T1@888", "T10@", "T101"):
        assert forbidden not in prompt
    assert "f1" in prompt and "f2" in prompt
    assert "0.31" in prompt and "-0.15" in prompt
    assert "P1" in prompt and "P2" in prompt and "abstain" in prompt
    fb = runner._bse_capsule_prompt(view, runner.BSE_OBS_FEATURE, 0.08,
                                    retry_feedback="未通过判据: x")
    assert "拒绝反馈" in fb


def test_episode_id_map():
    eps = [{"episode_id": "usel_t100_600_outlier_mad_positive",
            "domain_namespace": "kdd2018_dev"},
           {"episode_id": "batch1_t1_792_outlier_mad_conflict",
            "domain_namespace": "kdd2018_dev"},
           {"episode_id": "v6_nn5_support_positive",
            "domain_namespace": "nn5"}]
    m = runner._bse_episode_id_map(eps)
    assert m == {"T100@600": "usel_t100_600_outlier_mad_positive",
                 "T1@792": "batch1_t1_792_outlier_mad_conflict"}


def test_first_check_requires_heldout_validation():
    fit = [{"key": "a1", "group": "A", "features": {"f": 10.0}},
           {"key": "a2", "group": "A", "features": {"f": 9.0}},
           {"key": "b1", "group": "B", "features": {"f": 1.0}},
           {"key": "b2", "group": "B", "features": {"f": 2.0}}]
    held_ok = [{"key": "a3", "group": "A", "features": {"f": 8.0}},
               {"key": "b3", "group": "B", "features": {"f": 1.5}}]
    out = runner._bse_first_check(fit, held_ok)
    assert out["separable"] is True
    held_bad = [{"key": "a3", "group": "A", "features": {"f": 8.0}},
                {"key": "b3", "group": "B", "features": {"f": 9.5}}]
    out2 = runner._bse_first_check(fit, held_bad)
    assert out2["separable"] is False
    assert out2["fit_separable_candidates"]  # fit 可分但 heldout 失败
