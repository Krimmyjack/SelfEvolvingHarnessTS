"""A-31e 补样器（augment_corpus）测试 — A-38 协议守卫。

守卫：①重建确定性（manifest → 序列 bit 级一致）；②namespace 与 dev/confirmatory 不交、
uid 无碰撞；③manifest 与实际 perceive 一致（cell/SNR 可复核）；④补样后槽位计数达标。
manifest 依赖的测试在未生成时 skip（生成是一次性协议动作，不在测试里触发）。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from SelfEvolvingHarnessTS.augment_corpus import (CELLS, N_TARGET, RESULTS_A31E, _draw_noise,
                                                  _make_series, build_augmented_corpus,
                                                  load_augmented, slot_counts)
from SelfEvolvingHarnessTS.harness import HarnessState
from SelfEvolvingHarnessTS.fast_path.perceive import perceive
from SelfEvolvingHarnessTS.run_variance_decomp import STRUCTS, _det_seed, build_corpus

MANIFEST = RESULTS_A31E / "manifest.json"


def test_make_series_deterministic():
    a = _make_series("S_season", 12345, 0.42, 0.06, "S_season:A31e:test:0")
    b = _make_series("S_season", 12345, 0.42, 0.06, "S_season:A31e:test:0")
    assert np.array_equal(a.degraded, b.degraded, equal_nan=True)
    assert np.array_equal(a.clean, b.clean)
    assert a.series_uid == b.series_uid and a.seed == 12345


def test_noise_draw_deterministic_and_in_range():
    for sd in (1, 999, 1_999_999):
        n1, n2 = _draw_noise(sd), _draw_noise(sd)
        assert n1 == n2
        assert 0.03 <= n1 <= 2.0


def test_manifest_json_roundtrip():
    """noise float 经 JSON 往返后重建序列必须 bit 级一致（A-38 重建确定性）。"""
    noise = _draw_noise(777_001)
    a = _make_series("S_ar", 777_001, noise, 0.0, "u")
    noise_rt = json.loads(json.dumps({"noise": noise}))["noise"]
    b = _make_series("S_ar", 777_001, noise_rt, 0.0, "u")
    assert np.array_equal(a.degraded, b.degraded, equal_nan=True)


def test_namespace_disjoint_from_dev_and_confirmatory():
    """A31e 的 sd 命名空间键与原 namespace（dev j 0–19 / confirmatory j 20–39）不重叠：
    同一 (struct, j/k) 下键字符串不同 → 哈希输入不同；实际 sd 值碰撞由生成器显式跳过。"""
    dnames = ("d_hi_full", "d_hi_miss", "d_lo_full", "d_lo_miss")
    orig_keys = {f"{s}|{d}|{j}" for s in STRUCTS for d in dnames for j in range(40)}
    a31e_keys = {f"{s}|A31e|{c}|{k}" for s in STRUCTS for c in CELLS for k in range(100)}
    assert not orig_keys & a31e_keys


@pytest.mark.skipif(not MANIFEST.exists(), reason="A-31e manifest 未生成")
def test_manifest_entries_verifiable():
    """抽样复核：manifest 记录的 (cell, snr_measured) 与重建序列的 perceive 实测一致。"""
    doc = json.loads(MANIFEST.read_text("utf-8"))
    entries = doc["entries"]
    assert doc["n_aug"] == len(entries) > 0
    h = HarnessState.from_minimal()
    rng = np.random.default_rng(0)
    for e in [entries[i] for i in rng.choice(len(entries), size=min(8, len(entries)), replace=False)]:
        rs = _make_series(e["struct"], e["sd"], e["noise"], e["miss"], e["uid"])
        key = perceive(rs.history, "forecast", h)
        assert key["cell_id"] == e["cell"]
        assert float(key["pattern"]["struct_feats"]["SNR"]) == pytest.approx(e["snr_measured"], abs=1e-9)


@pytest.mark.skipif(not MANIFEST.exists(), reason="A-31e manifest 未生成")
def test_augmented_corpus_counts_and_uid_disjoint():
    """补样后：uid 无碰撞、原 320 条不动、manifest 中有收成的槽位达 N_TARGET。"""
    base = build_corpus(20)
    corpus = build_augmented_corpus(20)
    assert len(corpus) == len(base) + json.loads(MANIFEST.read_text("utf-8"))["n_aug"]
    uids = [rs.series_uid for rs in corpus]
    assert len(uids) == len(set(uids))
    assert all(":A31e:" in rs.series_uid for rs in corpus[len(base):])
    assert all(":A31e:" not in rs.series_uid for rs in corpus[:len(base)])

    counts, _, _ = slot_counts(corpus)
    doc = json.loads(MANIFEST.read_text("utf-8"))
    got_by_slot = {}
    for e in doc["entries"]:
        got_by_slot[(e["cell"], e["struct"])] = got_by_slot.get((e["cell"], e["struct"]), 0) + 1
    for slot, got in got_by_slot.items():
        assert counts[slot] >= N_TARGET, f"槽位 {slot} 补样后仍不足：{counts[slot]}"


@pytest.mark.skipif(not MANIFEST.exists(), reason="A-31e manifest 未生成")
def test_augmented_sds_disjoint_from_base():
    """生成器的 sd 碰撞跳过生效：补样 sd 与 dev 语料 sd 无交集（护 A-34 独立性）。"""
    base_sds = {rs.seed for rs in build_corpus(20)}
    aug = load_augmented()
    aug_sds = [rs.seed for rs in aug]
    assert len(aug_sds) == len(set(aug_sds))
    assert not set(aug_sds) & base_sds
