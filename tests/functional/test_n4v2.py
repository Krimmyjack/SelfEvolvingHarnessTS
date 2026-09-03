"""N4v2 字段级暴露扫描匹配器测试（recall-first）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402

UID = "0414c7e952d0" + "ab" * 26  # 64-hex
SHA = "ff" * 32


def _kinds(uid=UID, entity="T635", sha=SHA):
    return {name: m for name, m in runner._n4v2_series_matchers(uid, entity, sha)}


def _hit(m, text):
    return (m.search(text) is not None) if hasattr(m, "search") else bool(m(text))


def test_full_uid_and_prefix():
    ms = _kinds()
    assert _hit(ms["uid_full"], "series " + UID + " done")
    assert _hit(ms["uid_prefix8"], "（0414c7e9=T635 @792/888）")   # 审核案例原文形态
    assert _hit(ms["uid_prefix8"], "uid 0414c7e952d0ab... rest")
    assert not _hit(ms["uid_prefix8"], "nothing here")


def test_entity_json_quoted():
    ms = _kinds()
    assert _hit(ms["entity_json"], '{"entity_id": "T635", "x": 1}')
    assert not _hit(ms["entity_json"], '{"entity_id": "T6350"}')  # 引号边界防子串
    assert not _hit(ms["entity_json"], "bare T635 prose")          # 不带引号不归此类


def test_entity_at_origin():
    ms = _kinds()
    assert _hit(ms["entity_at_origin"], "T635@792")
    assert _hit(ms["entity_at_origin"], "T635 @792/888")
    assert not _hit(ms["entity_at_origin"], "T635 plain mention")


def test_entity_prose_bare_and_kdd_collision():
    ms = _kinds()
    assert _hit(ms["entity_prose"], "sealed 正例（T635 +0.4045）")
    # KDD 撞名集（T13/T128-134）不产生裸词匹配器
    ms13 = _kinds(uid="ab" * 16, entity="T13", sha="")  # 32 字符 ≠ 64-hex
    assert "entity_prose" not in ms13
    assert "uid_prefix8" not in ms13   # 非 64-hex 无前缀匹配器
    assert "sha_full" not in ms13      # 无 sha 无该匹配器
