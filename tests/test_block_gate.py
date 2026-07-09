"""tests/test_block_gate.py — 块级新颖性门守卫：确定性 / 无标签 API / 方向正确性。"""
import inspect

import numpy as np

from SelfEvolvingHarnessTS.run_block_gate import ALPHA, N_PERM_NULL, block_novelty_p


def test_deterministic():
    rng = np.random.default_rng(0)
    seen = rng.normal(0, 1, (200, 10))
    blk = rng.normal(0, 1, (40, 10))
    r1 = block_novelty_p(seen, blk, seed=123)
    r2 = block_novelty_p(seen, blk, seed=123)
    assert r1 == r2


def test_no_label_api():
    """检测器签名只收特征矩阵与 seed——族标签物理进不来。"""
    params = list(inspect.signature(block_novelty_p).parameters)
    assert params == ["seen_X", "block_X", "seed"]


def test_direction():
    """同分布块 → 高 p（不触发）；移位块 → p 达最小分辨率（触发）。"""
    rng = np.random.default_rng(1)
    seen = rng.normal(0, 1, (300, 10))
    same = rng.normal(0, 1, (40, 10))
    _, p_same = block_novelty_p(seen, same, seed=7)
    shifted = rng.normal(0, 1, (40, 10)) + 4.0
    _, p_shift = block_novelty_p(seen, shifted, seed=7)
    assert p_same > ALPHA
    assert p_shift == 1.0 / (N_PERM_NULL + 1)


def test_uncalibratable_conservative():
    """已见集太小无法校准 → p=1（不触发，保守语义）。"""
    rng = np.random.default_rng(2)
    seen = rng.normal(0, 1, (30, 10))
    blk = rng.normal(0, 1, (40, 10)) + 9.0
    _, p = block_novelty_p(seen, blk, seed=3)
    assert p == 1.0
