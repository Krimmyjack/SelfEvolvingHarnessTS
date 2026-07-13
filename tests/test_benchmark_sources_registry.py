from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark.registry import (
    SeriesRecord,
    admit_series,
    import_legacy_inventory,
    read_registry_jsonl,
    write_registry_jsonl,
)
from SelfEvolvingHarnessTS.benchmark.sources import SOURCE_SPECS, SourceSpec
from SelfEvolvingHarnessTS.benchmark.split import SplitRole


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_source_spec_enforces_automatic_and_manual_contracts():
    automatic = SourceSpec(
        source_id="automatic",
        access="automatic",
        official_url="https://example.test/data",
        source_revision="a" * 40,
        license_id="cc-by-4.0",
        expected_frequency="hourly",
        overlap_family="example",
    )
    assert automatic.source_revision == "a" * 40
    assert automatic.incoming_subdir is None

    manual = SourceSpec(
        source_id="manual",
        access="manual",
        official_url="https://example.test/portal",
        source_revision="portal-export-v1",
        license_id="terms-of-use",
        expected_frequency="hourly",
        overlap_family="example",
        incoming_subdir="manual",
    )
    assert manual.incoming_subdir == "manual"

    with pytest.raises(ValueError, match="source_revision"):
        SourceSpec(
            source_id="bad-auto",
            access="automatic",
            official_url="https://example.test/data",
            source_revision="main",
            license_id="cc-by-4.0",
            expected_frequency="hourly",
            overlap_family="example",
        )
    with pytest.raises(ValueError, match="incoming_subdir"):
        SourceSpec(
            source_id="bad-manual",
            access="manual",
            official_url="https://example.test/portal",
            source_revision="portal-export-v1",
            license_id="terms-of-use",
            expected_frequency="hourly",
            overlap_family="example",
        )


def test_approved_official_sources_are_registered_exactly():
    assert set(SOURCE_SPECS) == {
        "monash_hf",
        "metr_la",
        "uci_electricity_load_diagrams",
        "noaa_global_hourly",
        "entsoe_transparency",
        "gefcom2012",
        "gefcom2014",
    }
    assert SOURCE_SPECS["monash_hf"].access == "automatic"
    assert SOURCE_SPECS["monash_hf"].source_revision == "refs/convert/parquet"
    assert "10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX" in SOURCE_SPECS["metr_la"].official_url
    assert "/321/" in SOURCE_SPECS["uci_electricity_load_diagrams"].official_url
    assert "global-hourly" in SOURCE_SPECS["noaa_global_hourly"].official_url.lower()
    assert SOURCE_SPECS["entsoe_transparency"].access == "manual"
    assert "transparency.entsoe.eu" in SOURCE_SPECS["entsoe_transparency"].official_url
    for source_id in ("gefcom2012", "gefcom2014"):
        assert SOURCE_SPECS[source_id].access == "manual"
        assert "kaggle.com" in SOURCE_SPECS[source_id].official_url


def test_registry_keeps_natural_missing_mask_and_round_trips_jsonl(tmp_path):
    values = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    row = SeriesRecord.from_values(
        dataset_id="x",
        entity_id="e",
        values=values,
        source_revision="rev",
        license_id="cc-by-4.0",
        exposure_class="certified_virgin",
        frequency="hourly",
        overlap_group="x:e",
        regime_tag="candidate",
    )
    assert row.natural_missing_rate == pytest.approx(1 / 3)
    assert row.natural_missing_count == 1
    assert row.natural_missing_mask_sha

    changed_mask = SeriesRecord.from_values(
        dataset_id="x",
        entity_id="e2",
        values=np.array([np.nan, 1.0, 3.0]),
        source_revision="rev",
        license_id="cc-by-4.0",
        exposure_class="certified_virgin",
        frequency="hourly",
        overlap_group="x:e2",
        regime_tag="candidate",
    )
    assert changed_mask.natural_missing_mask_sha != row.natural_missing_mask_sha

    path = tmp_path / "registry.jsonl"
    write_registry_jsonl(path, [row, changed_mask])
    loaded = read_registry_jsonl(path)
    assert loaded == [row, changed_mask]
    assert path.read_text("utf-8").endswith("\n")
    row.verify_values(values)
    with pytest.raises(ValueError, match="content_sha"):
        row.verify_values(np.array([1.0, np.nan, 4.0]))


def test_registry_exposure_and_overlap_fail_closed_and_match_split_contract():
    kwargs = dict(
        dataset_id="x",
        entity_id="e",
        values=np.arange(240.0),
        source_revision="rev",
        license_id="cc-by-4.0",
        frequency="hourly",
        regime_tag="candidate",
    )
    with pytest.raises(ValueError, match="exposure_class"):
        SeriesRecord.from_values(
            **kwargs, exposure_class="unknown", overlap_group="x:e"
        )
    with pytest.raises(ValueError, match="overlap_group"):
        SeriesRecord.from_values(
            **kwargs, exposure_class="certified_virgin", overlap_group=""
        )

    exposed = SeriesRecord.from_values(
        **kwargs, exposure_class="confirmed_exposed", overlap_group="x:e"
    )
    assert exposed.roles_allowed == (SplitRole.SUPPORT_A.value,)
    candidate = exposed.to_split_candidate()
    assert candidate.exposure_class == exposed.exposure_class
    assert candidate.overlap_group == exposed.overlap_group


def test_admission_is_explicit_and_noaa_missing_or_irregular_is_not_skipped():
    timestamps = np.array(
        ["2020-01-01T00", "2020-01-01T01", "2020-01-01T03"],
        dtype="datetime64[h]",
    )
    row = SeriesRecord.from_values(
        dataset_id="noaa_global_hourly",
        entity_id="station",
        values=np.array([1.0, np.nan, 3.0]),
        timestamps=timestamps,
        source_revision="2020",
        license_id="us-public-domain",
        exposure_class="certified_virgin",
        frequency="hourly",
        overlap_group="noaa:station",
        regime_tag="candidate_u",
    )
    assert row.natural_missing_count == 1
    assert row.irregular_interval_count == 1
    decision = admit_series(row, min_len=3, allowed_frequencies={"hourly"})
    assert decision.eligible
    assert decision.natural_missing_rate == pytest.approx(1 / 3)
    assert decision.irregular_sampling_rate > 0


def test_existing_legacy_inventory_is_exactly_83_and_support_a_only():
    path = REPO_ROOT / "data" / "_artifacts" / "monash_clean.meta.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    counts = Counter(row["config"] for row in rows)
    assert counts == {
        "nn5_daily": 20,
        "fred_md": 20,
        "tourism_monthly": 20,
        "covid_deaths": 20,
        "us_births": 1,
        "saugeenday": 1,
        "sunspot": 1,
    }
    assert len(rows) == 83

    imported = import_legacy_inventory(path)
    assert len(imported) == 83
    assert {row.exposure_class for row in imported} == {"confirmed_exposed"}
    assert {row.roles_allowed for row in imported} == {(SplitRole.SUPPORT_A.value,)}
