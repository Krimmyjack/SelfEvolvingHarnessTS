"""CLS-4 unit tests for repair_burst_segment."""
from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.operators.registry import (
    OPERATOR_METADATA,
    get_operator,
)
from SelfEvolvingHarnessTS.operators.s1_burst import (
    BURST_MIN_RUN,
    BURST_Z_THRESHOLD,
    detect_burst_segments,
    repair_burst_segment,
)


def test_registry_contract_is_classification_only_intrinsic_destructive():
    meta = OPERATOR_METADATA["repair_burst_segment"]
    assert meta["targeting_mode"] == "intrinsic"
    assert meta["allowed_tasks"] == ("classification",)
    assert meta["destructive"] is True
    assert meta["preserves_observed"] is False
    assert meta.get("is_alias") is None
    assert get_operator("repair_burst_segment") is repair_burst_segment


def test_deterministic_same_input_same_output():
    rng = np.random.RandomState(20260825)
    values = rng.normal(0.0, 1.0, size=96)
    values[30:50] += 8.0
    first = repair_burst_segment(values)
    second = repair_burst_segment(values)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float64
    assert first.shape == values.shape
    assert np.isfinite(first).all()


def test_clean_series_is_byte_identity():
    clean = np.sin(2.0 * np.pi * np.arange(128, dtype=float) / 24.0)
    out = repair_burst_segment(clean)
    np.testing.assert_array_equal(out, clean)
    assert detect_burst_segments(clean) == []


def test_short_run_below_min_length_is_identity():
    values = np.zeros(64, dtype=float)
    values[10:10 + (BURST_MIN_RUN - 1)] = 12.0
    np.testing.assert_array_equal(repair_burst_segment(values), values)
    assert detect_burst_segments(values) == []


def test_detects_synthetic_burst_indices_exactly():
    values = np.zeros(64, dtype=float)
    values[16:32] = 10.0
    assert BURST_Z_THRESHOLD == 3.5
    assert detect_burst_segments(values) == [(16, 32)]


def test_boundary_segment_at_head_and_tail():
    head = np.zeros(64, dtype=float)
    head[0:12] = 9.0
    assert detect_burst_segments(head) == [(0, 12)]
    repaired_head = repair_burst_segment(head)
    np.testing.assert_allclose(repaired_head[0:12], 0.0, atol=1e-12)
    np.testing.assert_array_equal(repaired_head[12:], head[12:])

    tail = np.zeros(64, dtype=float)
    tail[52:64] = 9.0
    assert detect_burst_segments(tail) == [(52, 64)]
    repaired_tail = repair_burst_segment(tail)
    np.testing.assert_allclose(repaired_tail[52:64], 0.0, atol=1e-12)
    np.testing.assert_array_equal(repaired_tail[:52], tail[:52])


def test_repair_mae_beats_corruption_on_synthetic_burst():
    clean = np.sin(2.0 * np.pi * np.arange(128, dtype=float) / 32.0)
    corrupted = clean.copy()
    corrupted[40:60] += 8.0
    repaired = repair_burst_segment(corrupted)
    mae_dirty = float(np.mean(np.abs(corrupted - clean)))
    mae_fixed = float(np.mean(np.abs(repaired - clean)))
    assert detect_burst_segments(corrupted) == [(40, 60)]
    assert mae_dirty > 0.0
    assert mae_fixed < 0.25 * mae_dirty
