from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import SelfEvolvingHarnessTS.benchmark.sources as sources_module

from SelfEvolvingHarnessTS.benchmark.materialize import write_raw_once
from SelfEvolvingHarnessTS.benchmark.registry import (
    Admission,
    SeriesRecord,
    admit_series,
    import_legacy_inventory,
    read_registry_jsonl,
    write_registry_jsonl,
)
from SelfEvolvingHarnessTS.benchmark.sources import (
    AUTOMATIC_SOURCE_IDS,
    MANUAL_SOURCE_IDS,
    SOURCE_SPECS,
    SourceSpec,
    acquire_all_sources,
    build_acquisition_plan,
    select_noaa_stations,
)
from SelfEvolvingHarnessTS.benchmark.split import SplitRole


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_SHA = "a" * 64
EVIDENCE_SHA = "b" * 64


def _probe_features(
    *, natural_missing_rate: float = 0.0, irregular_sampling_rate: float = 0.0
) -> dict[str, float | int]:
    return {
        "seasonal_strength": 0.1,
        "trend_strength": 0.2,
        "spectral_entropy": 0.3,
        "natural_missing_count": int(natural_missing_rate > 0),
        "natural_missing_rate": natural_missing_rate,
        "irregular_interval_count": int(irregular_sampling_rate > 0),
        "irregular_sampling_rate": irregular_sampling_rate,
    }


def _fresh_record(
    values: np.ndarray,
    *,
    entity_id: str = "e",
    timestamps: np.ndarray | None = None,
    overlap_status: str = "resolved",
) -> SeriesRecord:
    spec = SOURCE_SPECS["noaa_global_hourly"]
    return SeriesRecord.from_values(
        dataset_id="noaa_global_hourly",
        entity_id=entity_id,
        values=values,
        source_id=spec.source_id,
        source_asset_sha256=ASSET_SHA,
        source_revision=spec.source_revision,
        license_id=spec.license_id,
        overlap_family=spec.overlap_family,
        exposure_class="certified_virgin",
        frequency="hourly",
        overlap_group=f"noaa:{entity_id}",
        overlap_status=overlap_status,
        overlap_evidence_sha256=EVIDENCE_SHA,
        timestamps=timestamps,
    )


def test_source_spec_enforces_automatic_and_manual_contracts():
    automatic = SourceSpec(
        source_id="automatic",
        access="automatic",
        official_url="https://example.test/data",
        source_revision="a" * 40,
        revision_kind="git_commit",
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
        revision_kind="manual_export",
        license_id="terms-of-use",
        expected_frequency="hourly",
        overlap_family="example",
        incoming_subdir="manual",
    )
    assert manual.incoming_subdir == "manual"

    pinned_asset = SourceSpec(
        source_id="asset-pinned",
        access="automatic",
        official_url="https://example.test/asset",
        source_revision="object-123",
        revision_kind="object_id",
        license_id="cc-by-4.0",
        expected_frequency="hourly",
        overlap_family="example",
        expected_asset_sha256="c" * 64,
    )
    pinned_asset.validate_asset_sha256("c" * 64)
    with pytest.raises(ValueError, match="expected_asset_sha256"):
        pinned_asset.validate_asset_sha256("d" * 64)

    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        SourceSpec(
            source_id="bad-commit",
            access="automatic",
            official_url="https://example.test/data",
            source_revision="not-a-commit",
            revision_kind="git_commit",
            license_id="cc-by-4.0",
            expected_frequency="hourly",
            overlap_family="example",
        )
    with pytest.raises(ValueError, match="revision_kind"):
        SourceSpec(
            source_id="bad-kind",
            access="automatic",
            official_url="https://example.test/data",
            source_revision="snapshot",
            revision_kind="branch",  # type: ignore[arg-type]
            license_id="cc-by-4.0",
            expected_frequency="hourly",
            overlap_family="example",
        )

    with pytest.raises(ValueError, match="source_revision"):
        SourceSpec(
            source_id="bad-auto",
            access="automatic",
            official_url="https://example.test/data",
            source_revision="main",
            revision_kind="git_commit",
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
            revision_kind="manual_export",
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
    assert SOURCE_SPECS["monash_hf"].source_revision == (
        "7bf79ee8270e340b6c5848b7b56d8e1c35305fb6"
    )
    assert SOURCE_SPECS["monash_hf"].revision_kind == "git_commit"
    assert "10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX" in SOURCE_SPECS["metr_la"].official_url
    assert "/321/" in SOURCE_SPECS["uci_electricity_load_diagrams"].official_url
    assert "global-hourly" in SOURCE_SPECS["noaa_global_hourly"].official_url.lower()
    assert SOURCE_SPECS["entsoe_transparency"].access == "manual"
    assert "transparency.entsoe.eu" in SOURCE_SPECS["entsoe_transparency"].official_url
    for source_id in ("gefcom2012", "gefcom2014"):
        assert SOURCE_SPECS[source_id].access == "manual"
        assert "kaggle.com" in SOURCE_SPECS[source_id].official_url


def test_acquisition_plan_keeps_automatic_and_manual_sources_distinct(tmp_path):
    plan = build_acquisition_plan(tmp_path)
    assert {row.source_id for row in plan if row.access == "automatic"} == set(
        AUTOMATIC_SOURCE_IDS
    )
    manual = {row.source_id: row for row in plan if row.access == "manual"}
    assert set(manual) == set(MANUAL_SOURCE_IDS)
    assert all(row.status == "manual_required" for row in manual.values())
    incoming = tmp_path / "incoming" / "gefcom2012"
    incoming.mkdir(parents=True)
    (incoming / "official.zip").write_bytes(b"x")
    refreshed = {row.source_id: row for row in build_acquisition_plan(tmp_path)}
    assert refreshed["gefcom2012"].status == "manual_ready"


def test_noaa_station_selection_is_content_keyed_and_has_no_refill():
    catalog = pd.DataFrame(
        {
            "USAF": [10000, 10001, 10002, 10003],
            "WBAN": [99999, 99999, 99999, 99999],
            "CTRY": ["US"] * 4,
            "BEGIN": [20200101] * 4,
            "END": [20251231] * 4,
            "LAT": [1.0, 2.0, 3.0, 4.0],
            "LON": [1.0, 2.0, 3.0, 4.0],
        }
    )
    first = select_noaa_stations(catalog, year=2024, count=3)
    second = select_noaa_stations(catalog.iloc[::-1], year=2024, count=3)
    assert first == second
    assert len(first) == 3
    assert all(len(station) == 11 for station in first)


def test_acquisition_status_manifest_is_written_without_network(tmp_path):
    results = acquire_all_sources(tmp_path, automatic=False)
    assert (tmp_path / "acquisition_manifest.json").is_file()
    by_id = {row.source_id: row for row in results}
    assert all(by_id[source_id].status == "automatic_skipped" for source_id in AUTOMATIC_SOURCE_IDS)
    assert all(by_id[source_id].status == "manual_required" for source_id in MANUAL_SOURCE_IDS)


def test_acquisition_status_reports_hash_bound_manual_asset(tmp_path):
    spec = SOURCE_SPECS["gefcom2012"]
    asset = write_raw_once(
        tmp_path / "incoming" / spec.incoming_subdir / "GEFCom2012.zip",
        b"official-archive",
        source_revision=spec.source_revision,
    )

    results = acquire_all_sources(tmp_path, automatic=False)
    row = {item.source_id: item for item in results}["gefcom2012"]

    assert row.status == "manual_bound"
    assert row.asset_sha256 == (asset.sha256,)
    assert row.asset_paths == (asset.path.as_posix(),)


def _preplace_metr_locations(tmp_path, monkeypatch):
    """Bind a fixture coordinate file so acquisition short-circuits its download.

    METR-LA acquisition now also binds the pinned DCRNN sensor coordinates, because the
    spatial blocking that keeps co-located sensors out of opposing roles is a pure
    function of that file.  Already-bound raw assets are never re-fetched, so a
    pre-placed asset means these tests still touch no network.
    """
    body = b"index,sensor_id,latitude,longitude\n0,700000,34.0,-118.0\n"
    asset = write_raw_once(
        tmp_path / "raw" / "metr_la" / "graph_sensor_locations.csv",
        body,
        source_revision=sources_module.METR_LA_SENSOR_GRAPH_COMMIT,
    )
    monkeypatch.setattr(
        sources_module, "METR_LA_SENSOR_LOCATIONS_SHA256", asset.sha256
    )
    return asset


def test_metr_acquisition_uses_google_drive_downloader_after_http_failure(
    tmp_path, monkeypatch
):
    _preplace_metr_locations(tmp_path, monkeypatch)

    class FailingSession:
        def get(self, *args, **kwargs):
            raise RuntimeError("drive direct endpoint failed")

    def fake_drive_download(*, id, output, quiet):
        assert id == SOURCE_SPECS["metr_la"].source_revision
        assert quiet is True
        Path(output).write_bytes(b"\x89HDF\r\n\x1a\nfixture")
        return output

    result = sources_module._acquire_metr_la(
        tmp_path,
        FailingSession(),
        drive_downloader=fake_drive_download,
    )
    assert result.status == "complete"
    # The matrix AND its coordinates: METR-LA is not usable without both.
    assert len(result.asset_sha256) == 2


def test_metr_manual_official_object_is_bound_before_any_network_access(
    tmp_path, monkeypatch
):
    destination = tmp_path / "raw" / "metr_la" / "metr-la.h5"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"\x89HDF\r\n\x1a\nmanual")
    _preplace_metr_locations(tmp_path, monkeypatch)

    class NoNetwork:
        def get(self, *args, **kwargs):
            raise AssertionError("manual import must not touch the network")

    result = sources_module._acquire_metr_la(tmp_path, NoNetwork())
    assert result.status == "complete"
    assert destination.with_name("metr-la.h5.asset.json").is_file()


def test_metr_sensor_locations_are_rejected_when_they_drift_from_the_pinned_digest(
    tmp_path,
):
    write_raw_once(
        tmp_path / "raw" / "metr_la" / "graph_sensor_locations.csv",
        b"index,sensor_id,latitude,longitude\n0,700000,0.0,0.0\n",
        source_revision=sources_module.METR_LA_SENSOR_GRAPH_COMMIT,
    )

    class NoNetwork:
        def get(self, *args, **kwargs):
            raise AssertionError("an already-bound asset must not be re-fetched")

    # If the coordinate file moved, every spatial block silently moves with it.
    with pytest.raises(ValueError, match="differ from the pinned digest"):
        sources_module._acquire_metr_la_sensor_locations(tmp_path, NoNetwork())


def test_registry_keeps_natural_missing_mask_and_round_trips_jsonl(tmp_path):
    values = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    raw = _fresh_record(values)
    row = raw.with_probe_result(
        probe_features=_probe_features(natural_missing_rate=1 / 3),
        regime_tag="candidate",
        admission=Admission(True, (), 1 / 3, 0.0),
    )
    assert row.natural_missing_rate == pytest.approx(1 / 3)
    assert row.natural_missing_count == 1
    assert row.natural_missing_mask_sha

    changed_mask = _fresh_record(
        np.array([np.nan, 1.0, 3.0]), entity_id="e2"
    ).with_probe_result(
        probe_features=_probe_features(natural_missing_rate=1 / 3),
        regime_tag="candidate",
        admission=Admission(True, (), 1 / 3, 0.0),
    )
    assert changed_mask.natural_missing_mask_sha != row.natural_missing_mask_sha

    path = tmp_path / "registry.jsonl"
    write_registry_jsonl(path, [row, changed_mask])
    loaded = read_registry_jsonl(path)
    assert loaded == [row, changed_mask]
    assert dict(loaded[0].probe_features) == dict(row.probe_features)
    with pytest.raises(TypeError):
        loaded[0].probe_features["seasonal_strength"] = 0.9
    assert path.read_text("utf-8").endswith("\n")
    row.verify_values(values)
    with pytest.raises(ValueError, match="content_sha"):
        row.verify_values(np.array([1.0, np.nan, 4.0]))


def test_registry_exposure_and_overlap_fail_closed_and_match_split_contract():
    spec = SOURCE_SPECS["noaa_global_hourly"]
    kwargs = dict(
        dataset_id="noaa_global_hourly",
        entity_id="e",
        values=np.arange(240.0),
        source_id=spec.source_id,
        source_asset_sha256=ASSET_SHA,
        source_revision=spec.source_revision,
        license_id=spec.license_id,
        overlap_family=spec.overlap_family,
        frequency="hourly",
        overlap_status="resolved",
        overlap_evidence_sha256=EVIDENCE_SHA,
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
    exposed = exposed.with_probe_result(
        probe_features=_probe_features(),
        regime_tag="candidate",
        admission=Admission(True, (), 0.0, 0.0),
    )
    candidate = exposed.to_split_candidate()
    assert candidate.exposure_class == exposed.exposure_class
    assert candidate.overlap_group == exposed.overlap_group


def test_admission_is_explicit_and_noaa_missing_or_irregular_is_not_skipped():
    timestamps = np.array(
        ["2020-01-01T00", "2020-01-01T01", "2020-01-01T03"],
        dtype="datetime64[h]",
    )
    row = _fresh_record(
        np.array([1.0, np.nan, 3.0]), entity_id="station", timestamps=timestamps
    )
    assert row.natural_missing_count == 1
    assert row.irregular_interval_count == 1
    decision = admit_series(
        row,
        min_len=3,
        allowed_frequencies={"hourly"},
        max_natural_missing_rate=0.2,
        max_irregular_sampling_rate=0.1,
    )
    assert not decision.eligible
    assert decision.reasons == (
        "excessive_natural_missingness",
        "excessive_irregular_sampling",
    )
    assert decision.natural_missing_rate == pytest.approx(1 / 3)
    assert decision.irregular_sampling_rate > 0
    for field in ("max_natural_missing_rate", "max_irregular_sampling_rate"):
        kwargs = dict(
            min_len=3,
            allowed_frequencies={"hourly"},
            max_natural_missing_rate=0.2,
            max_irregular_sampling_rate=0.1,
        )
        kwargs[field] = 1.1
        with pytest.raises(ValueError, match=field):
            admit_series(row, **kwargs)


def test_registry_rejects_multidimensional_values_and_nonincreasing_timestamps():
    spec = SOURCE_SPECS["noaa_global_hourly"]
    kwargs = dict(
        dataset_id="noaa_global_hourly",
        entity_id="e",
        source_id=spec.source_id,
        source_asset_sha256=ASSET_SHA,
        source_revision=spec.source_revision,
        license_id=spec.license_id,
        overlap_family=spec.overlap_family,
        exposure_class="certified_virgin",
        frequency="hourly",
        overlap_group="x:e",
        overlap_status="resolved",
        overlap_evidence_sha256=EVIDENCE_SHA,
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        SeriesRecord.from_values(values=np.ones((2, 120)), **kwargs)
    for timestamps in (
        np.array(["2020-01-01T00", "2020-01-01T00", "2020-01-01T01"]),
        np.array(["2020-01-01T00", "2019-12-31T23", "2020-01-01T01"]),
    ):
        with pytest.raises(ValueError, match="strictly increasing"):
            SeriesRecord.from_values(
                values=np.arange(3.0), timestamps=timestamps, **kwargs
            )


def test_registered_source_binding_and_fresh_finalization_fail_closed():
    spec = SOURCE_SPECS["noaa_global_hourly"]
    base = dict(
        dataset_id="noaa_global_hourly",
        entity_id="station",
        values=np.arange(240.0),
        source_id=spec.source_id,
        source_asset_sha256=ASSET_SHA,
        source_revision=spec.source_revision,
        license_id=spec.license_id,
        overlap_family=spec.overlap_family,
        exposure_class="certified_virgin",
        frequency="hourly",
        overlap_group="noaa:station",
        overlap_status="resolved",
        overlap_evidence_sha256=EVIDENCE_SHA,
    )
    for field, wrong in (
        ("source_revision", "wrong"),
        ("license_id", "wrong"),
        ("overlap_family", "wrong"),
    ):
        kwargs = dict(base)
        kwargs[field] = wrong
        with pytest.raises(ValueError, match=field):
            SeriesRecord.from_values(**kwargs)

    raw = SeriesRecord.from_values(**base)
    assert raw.probe_features is None
    assert raw.regime_tag is None
    assert raw.admission_reasons is None
    with pytest.raises(ValueError, match="probe/admission"):
        raw.to_split_candidate()
    with pytest.raises(ValueError, match="canonical probe schema"):
        raw.with_probe_result(
            probe_features={"seasonal_strength": 0.1},
            regime_tag="seasonal",
            admission=Admission(True, (), 0.0, 0.0),
        )

    rejected = raw.with_probe_result(
        probe_features=_probe_features(),
        regime_tag="seasonal",
        admission=Admission(False, ("below_min_length",), 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="admission"):
        rejected.to_split_candidate()

    ready = raw.with_probe_result(
        probe_features=_probe_features(),
        regime_tag="seasonal",
        admission=Admission(True, (), 0.0, 0.0),
    )
    assert ready.to_split_candidate().regime_tag == "seasonal"
    with pytest.raises(ValueError, match="already finalized"):
        ready.with_probe_result(
            probe_features=_probe_features(),
            regime_tag="other",
            admission=Admission(True, (), 0.0, 0.0),
        )


def test_unresolved_fresh_overlap_cannot_reach_final_or_split():
    raw = _fresh_record(np.arange(240.0), overlap_status="unresolved")
    assert SplitRole.SUPPORT_B.value not in raw.roles_allowed
    assert SplitRole.FINAL_QUERY.value not in raw.roles_allowed
    assert SplitRole.U.value not in raw.roles_allowed
    ready = raw.with_probe_result(
        probe_features=_probe_features(),
        regime_tag="candidate",
        admission=Admission(True, (), 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="overlap"):
        ready.to_split_candidate()


def test_probe_finalization_must_match_registry_and_fresh_write_is_finalized(tmp_path):
    raw = _fresh_record(np.arange(240.0))
    with pytest.raises(ValueError, match="finalized probe/admission"):
        write_registry_jsonl(tmp_path / "raw.jsonl", [raw])

    wrong = _probe_features(natural_missing_rate=0.5)
    with pytest.raises(ValueError, match="probe feature natural_missing"):
        raw.with_probe_result(
            probe_features=wrong,
            regime_tag="candidate",
            admission=Admission(True, (), 0.0, 0.0),
        )

    with pytest.raises(ValueError, match="finite rate"):
        admit_series(
            raw,
            min_len=3,
            allowed_frequencies={"hourly"},
            max_natural_missing_rate=10**400,
            max_irregular_sampling_rate=0.1,
        )


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
    values_path = path.with_name(path.name.replace(".meta.jsonl", ".npz"))
    metadata_bytes = path.read_bytes()
    values_bytes = values_path.read_bytes()
    expected_asset_sha = hashlib.sha256(
        b"benchmark-legacy-bundle-v1\0"
        + len(metadata_bytes).to_bytes(8, "big")
        + metadata_bytes
        + len(values_bytes).to_bytes(8, "big")
        + values_bytes
    ).hexdigest()
    assert {row.source_id for row in imported} == {"legacy_internal_monash_clean"}
    assert {row.source_asset_sha256 for row in imported} == {expected_asset_sha}
    assert {row.overlap_evidence_sha256 for row in imported} == {expected_asset_sha}
