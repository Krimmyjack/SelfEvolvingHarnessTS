from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark.ingestion import (
    IngestionInvalid,
    canonical_ingest,
)


def test_ingestion_counts_fill_clamps_endpoints_and_rejects_infinity():
    got = canonical_ingest(np.array([np.nan, 2.0, np.nan, 4.0, np.nan]))
    assert got.values.tolist() == [2.0, 2.0, 3.0, 4.0, 4.0]
    assert got.filled_count == 3
    assert got.fill_rate == pytest.approx(0.6)
    assert got.dependency_flag
    assert not got.values.flags.writeable
    with pytest.raises(IngestionInvalid):
        canonical_ingest(np.array([1.0, np.inf]))


def test_ingestion_is_bit_reproducible_and_does_not_mutate_input():
    raw = np.array([1.0, np.nan, 3.0])
    before = raw.copy()
    a = canonical_ingest(raw)
    b = canonical_ingest(raw)
    assert np.array_equal(raw, before, equal_nan=True)
    assert a.values.tobytes() == b.values.tobytes()
    assert a.fill_rate == b.fill_rate


def test_ingestion_rejects_shape_empty_and_all_nan():
    for values in (np.ones((2, 2)), np.array([]), np.array([np.nan, np.nan])):
        with pytest.raises(IngestionInvalid):
            canonical_ingest(values)


def test_dependency_flag_threshold_is_strictly_above_one_percent():
    one_percent = np.arange(100.0)
    one_percent[5] = np.nan
    assert not canonical_ingest(one_percent).dependency_flag
    above = np.arange(99.0)
    above[5] = np.nan
    assert canonical_ingest(above).dependency_flag
