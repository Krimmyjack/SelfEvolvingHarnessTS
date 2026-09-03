"""BATCH1（outlier Scope 固定 Batch）零 LLM 裁定函数测试——用户裁决 2026-08-14。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402

M = runner.M


def test_batch1_contexts_grid():
    ctxs = runner._batch1_contexts()
    assert len(ctxs) == 11
    assert len({s for s, _ in ctxs}) == 4  # 覆盖全部 4 条 dev 序列
    keys = {s + "@" + str(o) for s, o in ctxs}
    for used in ("T1@888", "T10@600", "T100@600", "T101@792", "T101@600"):
        assert used not in keys
    assert "T1@600" in keys and "T101@984" in keys


def _lab(key, label, f):
    return {"key": key, "label": label, "features": {"f": f}}


def test_verdict_insufficient():
    labels = [_lab("T100@600", "POSITIVE", 10.0),
              _lab("T1@888", "NEGATIVE", 1.0)]
    v = runner._batch1_verdict([], labels)
    assert v["verdict"] == "INSUFFICIENT_CONTRASTIVE_EVIDENCE"
    assert v["outlier_scope_learning"] == "CLOSED"  # 只有 T100 正例


def test_verdict_only_t100_positive_closes_scope():
    labels = [_lab("T100@600", "POSITIVE", 10.0),
              _lab("T1@888", "NEGATIVE", 1.0),
              _lab("T10@600", "NEGATIVE", 2.0)]
    batch = [{"key": "T1@600", "gain": -M * 2, "features": {"f": 1.5}},
             {"key": "T10@792", "gain": -M * 2, "features": {"f": 2.5}}]
    v = runner._batch1_verdict(batch, labels)
    assert v["verdict"] == "INSUFFICIENT_CONTRASTIVE_EVIDENCE"
    assert v["outlier_scope_learning"] == "CLOSED"
    assert v["pos_count"] == 1 and v["positives"] == ["T100@600"]


def test_verdict_not_separable():
    labels = [_lab(k, "POSITIVE", 5.0) for k in ("T100@600", "A@1", "A@2")] + \
             [_lab(k, "NEGATIVE", 5.0) for k in ("T1@888", "B@1", "B@2")]
    v = runner._batch1_verdict([], labels)
    assert v["verdict"] == "SCOPE_CANDIDATE_ELIGIBLE_NOT_SEPARABLE"


def test_verdict_separable_proposed():
    labels = [_lab(k, "POSITIVE", float(i) + 10.0)
              for i, k in enumerate(("T100@600", "A@1", "A@2"))] + \
             [_lab(k, "NEGATIVE", float(i) + 1.0)
              for i, k in enumerate(("T1@888", "B@1", "B@2"))]
    extra = _lab("C@1", "NEGATIVE", 2.5)  # 验证集外样本（不在 fit 内）
    v = runner._batch1_verdict([], labels + [extra])
    assert v["verdict"] == "SCOPE_CANDIDATE_PROPOSED"
    assert v["scope_candidate"]["direction"] == "positive_above"
    assert v["validation"]["pass"] == ["C@1"] and v["validation"]["fail"] == []
    assert v["validation_ok"] is True


def test_verdict_separable_validation_fail():
    labels = [_lab(k, "POSITIVE", float(i) + 10.0)
              for i, k in enumerate(("T100@600", "A@1", "A@2"))] + \
             [_lab(k, "NEGATIVE", float(i) + 1.0)
              for i, k in enumerate(("T1@888", "B@1", "B@2"))]
    bad = _lab("C@1", "NEGATIVE", 30.0)  # 负例却落在正例区 → 验证失败
    v = runner._batch1_verdict([], labels + [bad])
    # fit 规则在剩余 context 验证失败 = 现有 Observation 整体不可分
    assert v["verdict"] == "SCOPE_CANDIDATE_ELIGIBLE_NOT_SEPARABLE"
    assert v["validation"]["fail"] == ["C@1"]


def test_cobs_feature_zero_llm_real_data():
    # 零 LLM 冒烟：T101（batch 惰性，behavior=0）→ influence=0.0；
    # T100@792（正例）→ 正影响力
    env = runner._load_env()
    values = env["values"]
    v_inert = runner._cobs_feature("T101", 888, values)
    assert v_inert == 0.0
    v_active = runner._cobs_feature("T100", 792, values)
    assert isinstance(v_active, float) and v_active > 0.0


def test_separability_directions():
    items = [_lab("p", "POSITIVE", 1.0), _lab("n1", "NEGATIVE", 10.0),
             _lab("n2", "NEGATIVE", 12.0)]
    seps = runner._batch1_separability(items)
    assert seps and seps[0]["direction"] == "positive_below"
