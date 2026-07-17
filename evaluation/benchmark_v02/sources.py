"""Frozen, verifiable acquisition specifications for benchmark sources."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping
from urllib.parse import urlparse

import pandas as pd

__all__ = [
    "SOURCE_SPECS",
    "AUTOMATIC_SOURCE_IDS",
    "METR_LA_SENSOR_GRAPH_COMMIT",
    "METR_LA_SENSOR_LOCATIONS_SHA256",
    "METR_LA_SENSOR_LOCATIONS_URL",
    "METR_LA_SPATIAL_BLOCKS",
    "NOAA_STATION_COUNT",
    "AcquisitionPlanRow",
    "AcquisitionResult",
    "MANUAL_SOURCE_IDS",
    "RevisionKind",
    "SourceSpec",
    "build_acquisition_plan",
    "acquire_all_sources",
    "get_source_spec",
    "select_noaa_stations",
]


AccessMode = Literal["automatic", "manual"]
RevisionKind = Literal[
    "git_commit", "object_id", "catalog_snapshot", "manual_export"
]
_REVISION_KINDS = frozenset(
    {"git_commit", "object_id", "catalog_snapshot", "manual_export"}
)

# METR-LA ships its sensor coordinates in the DCRNN repository, not in the HDF5
# matrix.  They are pinned to an immutable commit (not `master`) because the spatial
# blocking that keeps co-located sensors out of opposing split roles is a pure
# function of this file: if the file moved, the split would silently move with it.
METR_LA_SENSOR_GRAPH_COMMIT = "82922c830800ca7aeaf53acc412a6d2cf7e56055"
METR_LA_SENSOR_LOCATIONS_SHA256 = (
    "eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f"
)
METR_LA_SENSOR_LOCATIONS_URL = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN/"
    f"{METR_LA_SENSOR_GRAPH_COMMIT}/data/sensor_graph/graph_sensor_locations.csv"
)

# Blocking parameter, frozen with the benchmark: the smallest block count for which
# every outer role still receives at least 20 METR-LA series.  Fewer blocks (12, 16)
# starve Dev-Query; more blocks buy nothing and shrink each block's footprint.
METR_LA_SPATIAL_BLOCKS = 20

# The weather U pool.  select_noaa_stations() orders the eligible universe by a hash
# that does not depend on `count`, so raising the count is append-only: the original
# 12 stations stay exactly where they were and 52 new ones are added behind them.
# The literal "benchmark-v0" salt inside select_noaa_stations is deliberately NOT
# bumped to v0.1 -- bumping it would reshuffle the order and un-select assets that are
# already downloaded and hash-bound.
NOAA_STATION_COUNT = 64


def _canonical_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    value = _canonical_string(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    access: AccessMode
    official_url: str
    source_revision: str
    revision_kind: RevisionKind
    license_id: str
    expected_frequency: str
    overlap_family: str
    incoming_subdir: str | None = None
    expected_asset_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "official_url",
            "source_revision",
            "license_id",
            "expected_frequency",
            "overlap_family",
        ):
            _canonical_string(getattr(self, field_name), field_name)
        parsed = urlparse(self.official_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("official_url must be an absolute HTTPS URL")
        if self.access not in ("automatic", "manual"):
            raise ValueError("access must be 'automatic' or 'manual'")
        if self.revision_kind not in _REVISION_KINDS:
            raise ValueError("revision_kind is not a frozen locator kind")
        if self.revision_kind == "git_commit":
            if len(self.source_revision) != 40 or any(
                character not in "0123456789abcdef"
                for character in self.source_revision
            ):
                raise ValueError(
                    "source_revision must be a 40 lowercase hexadecimal git commit"
                )
        if self.access == "automatic":
            if self.incoming_subdir is not None:
                raise ValueError("automatic sources cannot define incoming_subdir")
            if self.revision_kind == "manual_export":
                raise ValueError("automatic sources cannot use manual_export")
        else:
            if self.revision_kind != "manual_export":
                raise ValueError("manual sources must use manual_export")
            subdir = _canonical_string(self.incoming_subdir, "incoming_subdir")
            if any(
                part in {"", ".", ".."}
                for part in subdir.replace("\\", "/").split("/")
            ):
                raise ValueError("incoming_subdir must be a safe relative directory")
        if self.expected_asset_sha256 is not None:
            _require_sha256(self.expected_asset_sha256, "expected_asset_sha256")

    def validate_asset_sha256(self, actual_sha256: str) -> None:
        """Validate a materialized asset digest against this frozen source."""

        actual = _require_sha256(actual_sha256, "source_asset_sha256")
        if (
            self.expected_asset_sha256 is not None
            and actual != self.expected_asset_sha256
        ):
            raise ValueError("source asset differs from expected_asset_sha256")


_SPECS = (
    SourceSpec(
        source_id="monash_hf",
        access="automatic",
        official_url="https://huggingface.co/datasets/monash_tsf",
        source_revision="7bf79ee8270e340b6c5848b7b56d8e1c35305fb6",
        revision_kind="git_commit",
        license_id="dataset-specific-monash-tsf",
        expected_frequency="mixed",
        overlap_family="monash_tsf",
    ),
    SourceSpec(
        source_id="metr_la",
        access="automatic",
        official_url=(
            "https://drive.google.com/uc?id="
            "10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
        ),
        source_revision="10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX",
        revision_kind="object_id",
        license_id="dcrnn-research-release",
        expected_frequency="5min",
        overlap_family="metr_la",
    ),
    SourceSpec(
        source_id="uci_electricity_load_diagrams",
        access="automatic",
        official_url=(
            "https://archive.ics.uci.edu/dataset/321/"
            "electricityloaddiagrams20112014"
        ),
        source_revision="uci-dataset-321-static-export",
        revision_kind="catalog_snapshot",
        license_id="cc-by-4.0",
        expected_frequency="15min",
        overlap_family="electricity_load_diagrams_2011_2014",
    ),
    SourceSpec(
        source_id="noaa_global_hourly",
        access="automatic",
        official_url="https://www.ncei.noaa.gov/data/global-hourly/access/",
        source_revision="ncei-global-hourly-catalog-2026-07-13",
        revision_kind="catalog_snapshot",
        license_id="us-public-domain",
        expected_frequency="irregular_hourly",
        overlap_family="noaa_isd",
    ),
    SourceSpec(
        source_id="entsoe_transparency",
        access="manual",
        official_url="https://transparency.entsoe.eu/",
        source_revision="actual-total-load-export-v1",
        revision_kind="manual_export",
        license_id="entsoe-transparency-terms",
        expected_frequency="hourly",
        overlap_family="entsoe_actual_total_load",
        incoming_subdir="entsoe_transparency",
    ),
    SourceSpec(
        source_id="gefcom2012",
        access="manual",
        official_url=(
            "https://www.kaggle.com/competitions/"
            "global-energy-forecasting-competition-2012-load-forecasting"
        ),
        source_revision="kaggle-competition-final-files",
        revision_kind="manual_export",
        license_id="kaggle-competition-rules",
        expected_frequency="hourly",
        overlap_family="gefcom_load",
        incoming_subdir="gefcom2012",
    ),
    SourceSpec(
        source_id="gefcom2014",
        access="manual",
        official_url=(
            "https://www.kaggle.com/competitions/"
            "global-energy-forecasting-competition-2014-load-forecasting"
        ),
        source_revision="kaggle-competition-final-files",
        revision_kind="manual_export",
        license_id="kaggle-competition-rules",
        expected_frequency="hourly",
        overlap_family="gefcom_load",
        incoming_subdir="gefcom2014",
    ),
)

SOURCE_SPECS: Mapping[str, SourceSpec] = MappingProxyType(
    {spec.source_id: spec for spec in _SPECS}
)

AUTOMATIC_SOURCE_IDS = tuple(
    spec.source_id for spec in _SPECS if spec.access == "automatic"
)
MANUAL_SOURCE_IDS = tuple(
    spec.source_id for spec in _SPECS if spec.access == "manual"
)


@dataclass(frozen=True)
class AcquisitionPlanRow:
    source_id: str
    access: AccessMode
    status: str
    destination: str
    source_revision: str


@dataclass(frozen=True)
class AcquisitionResult:
    source_id: str
    status: str
    asset_paths: tuple[str, ...]
    asset_sha256: tuple[str, ...]
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "status": self.status,
            "asset_paths": list(self.asset_paths),
            "asset_sha256": list(self.asset_sha256),
            "message": self.message,
        }


def build_acquisition_plan(root: Path | str) -> tuple[AcquisitionPlanRow, ...]:
    """Describe automatic destinations and account-gated incoming status."""
    base = Path(root)
    rows: list[AcquisitionPlanRow] = []
    for spec in _SPECS:
        if spec.access == "automatic":
            status = "pending"
            destination = base / "raw" / spec.source_id
        else:
            assert spec.incoming_subdir is not None
            destination = base / "incoming" / spec.incoming_subdir
            ready = destination.is_dir() and any(path.is_file() for path in destination.rglob("*"))
            status = "manual_ready" if ready else "manual_required"
        rows.append(
            AcquisitionPlanRow(
                source_id=spec.source_id,
                access=spec.access,
                status=status,
                destination=destination.as_posix(),
                source_revision=spec.source_revision,
            )
        )
    return tuple(rows)


def _station_component(value: object, width: int) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit() or len(text) > width:
        raise ValueError("NOAA station identifier is not numeric")
    return text.zfill(width)


def select_noaa_stations(
    catalog: pd.DataFrame,
    *,
    year: int,
    count: int,
    country: str = "US",
) -> tuple[str, ...]:
    """Freeze a catalog-order-invariant station subset; failed downloads are not refilled."""
    required = {"USAF", "WBAN", "CTRY", "BEGIN", "END", "LAT", "LON"}
    if not isinstance(catalog, pd.DataFrame) or not required <= set(catalog.columns):
        raise ValueError("NOAA catalog lacks required station fields")
    if isinstance(year, bool) or not isinstance(year, int) or year < 1900:
        raise ValueError("NOAA year must be a valid integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("NOAA station count must be positive")
    start, end = year * 10000 + 101, year * 10000 + 1231
    eligible: list[str] = []
    for _, row in catalog.iterrows():
        try:
            if str(row["CTRY"]).strip() != country:
                continue
            if int(row["BEGIN"]) > start or int(row["END"]) < end:
                continue
            if pd.isna(row["LAT"]) or pd.isna(row["LON"]):
                continue
            station = _station_component(row["USAF"], 6) + _station_component(row["WBAN"], 5)
        except (TypeError, ValueError, OverflowError):
            continue
        eligible.append(station)
    universe = tuple(sorted(set(eligible)))
    if len(universe) < count:
        raise ValueError(f"NOAA catalog has only {len(universe)} eligible stations; need {count}")
    universe_sha = hashlib.sha256("\n".join(universe).encode("ascii")).hexdigest()
    ordered = sorted(
        universe,
        key=lambda station: hashlib.sha256(
            f"benchmark-v0|noaa|{year}|{universe_sha}|{station}".encode("ascii")
        ).hexdigest(),
    )
    return tuple(ordered[:count])


def _asset_result(source_id: str, status: str, assets: list[object], message: str) -> AcquisitionResult:
    return AcquisitionResult(
        source_id=source_id,
        status=status,
        asset_paths=tuple(Path(getattr(asset, "path")).as_posix() for asset in assets),
        asset_sha256=tuple(str(getattr(asset, "sha256")) for asset in assets),
        message=message,
    )


def _download_http(
    session: object,
    url: str,
    destination: Path,
    *,
    source_revision: str,
    expected_prefix: bytes | None = None,
) -> object:
    from .materialize import promote_download, write_raw_once

    if destination.is_file() and destination.with_name(destination.name + ".asset.json").is_file():
        return write_raw_once(
            destination, destination.read_bytes(), source_revision=source_revision
        )
    partial = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(url, stream=True, timeout=120, allow_redirects=True)
    response.raise_for_status()
    with partial.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    if expected_prefix is not None:
        with partial.open("rb") as handle:
            actual_prefix = handle.read(len(expected_prefix))
        if actual_prefix != expected_prefix:
            raise ValueError(
                f"downloaded bytes from {url!r} do not match the expected file signature"
            )
    return promote_download(
        partial,
        destination,
        source_revision=source_revision,
    )


def _acquire_monash(root: Path) -> AcquisitionResult:
    from huggingface_hub import HfApi, hf_hub_download

    from .materialize import promote_download, write_raw_once

    spec = SOURCE_SPECS["monash_hf"]
    files = set(
        HfApi().list_repo_files(
            "monash_tsf", repo_type="dataset", revision=spec.source_revision
        )
    )
    requested = ("nn5_daily", "covid_deaths", "traffic_hourly", "electricity_hourly")
    assets: list[object] = []
    unavailable: list[str] = []
    for config in requested:
        filename = f"{config}/test/0000.parquet"
        if filename not in files:
            unavailable.append(config)
            continue
        destination = root / "raw" / spec.source_id / filename
        if destination.is_file() and destination.with_name(destination.name + ".asset.json").is_file():
            assets.append(
                write_raw_once(
                    destination,
                    destination.read_bytes(),
                    source_revision=spec.source_revision,
                )
            )
            continue
        cached = Path(
            hf_hub_download(
                "monash_tsf",
                repo_type="dataset",
                revision=spec.source_revision,
                filename=filename,
            )
        )
        partial = destination.with_name(destination.name + ".partial")
        partial.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, partial)
        assets.append(
            promote_download(
                partial,
                destination,
                source_revision=spec.source_revision,
            )
        )
    status = "complete" if not unavailable else "partial_unavailable_at_pinned_revision"
    message = (
        "pinned HF parquet assets acquired"
        if not unavailable
        else f"pinned revision lacks configs: {unavailable}"
    )
    return _asset_result(spec.source_id, status, assets, message)


def _acquire_uci(root: Path, session: object) -> AcquisitionResult:
    spec = SOURCE_SPECS["uci_electricity_load_diagrams"]
    asset = _download_http(
        session,
        "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        root / "raw" / spec.source_id / "electricityloaddiagrams20112014.zip",
        source_revision=spec.source_revision,
        expected_prefix=b"PK",
    )
    return _asset_result(spec.source_id, "complete", [asset], "official UCI dataset-321 ZIP")


def _acquire_metr_la_sensor_locations(root: Path, session: object) -> object:
    """Bind the pinned DCRNN sensor coordinate file that spatial blocking depends on."""
    spec = SOURCE_SPECS["metr_la"]
    asset = _download_http(
        session,
        METR_LA_SENSOR_LOCATIONS_URL,
        root / "raw" / spec.source_id / "graph_sensor_locations.csv",
        source_revision=METR_LA_SENSOR_GRAPH_COMMIT,
    )
    actual = str(getattr(asset, "sha256"))
    if actual != METR_LA_SENSOR_LOCATIONS_SHA256:
        raise ValueError(
            "METR-LA sensor locations differ from the pinned digest "
            f"(expected {METR_LA_SENSOR_LOCATIONS_SHA256}, got {actual})"
        )
    return asset


def _acquire_metr_la(
    root: Path,
    session: object,
    *,
    drive_downloader: object | None = None,
) -> AcquisitionResult:
    from .materialize import promote_download

    spec = SOURCE_SPECS["metr_la"]
    object_id = spec.source_revision
    destination = root / "raw" / spec.source_id / "metr-la.h5"
    sidecar = destination.with_name(destination.name + ".asset.json")
    if destination.is_file() and not sidecar.exists():
        with destination.open("rb") as handle:
            if handle.read(8) != b"\x89HDF\r\n\x1a\n":
                raise ValueError("manually placed METR-LA file is not HDF5")
        partial = destination.with_name(destination.name + ".partial")
        destination.replace(partial)
        asset = promote_download(
            partial,
            destination,
            source_revision=spec.source_revision,
        )
        locations = _acquire_metr_la_sensor_locations(root, session)
        return _asset_result(
            spec.source_id,
            "complete",
            [asset, locations],
            "manually placed DCRNN author Drive object bound immutably; "
            "sensor coordinates pinned to the DCRNN commit",
        )
    try:
        asset = _download_http(
            session,
            f"https://drive.usercontent.google.com/download?id={object_id}&export=download&confirm=t",
            destination,
            source_revision=spec.source_revision,
            expected_prefix=b"\x89HDF\r\n\x1a\n",
        )
    except Exception:
        if drive_downloader is None:
            from gdown import download as drive_downloader

        partial = destination.with_name(destination.name + ".partial")
        partial.parent.mkdir(parents=True, exist_ok=True)
        downloaded = drive_downloader(id=object_id, output=str(partial), quiet=True)
        if downloaded is None or not partial.is_file():
            raise RuntimeError("Google Drive downloader did not materialize METR-LA")
        with partial.open("rb") as handle:
            if handle.read(8) != b"\x89HDF\r\n\x1a\n":
                raise ValueError("Google Drive fallback did not return an HDF5 asset")
        asset = promote_download(
            partial,
            destination,
            source_revision=spec.source_revision,
        )
    locations = _acquire_metr_la_sensor_locations(root, session)
    return _asset_result(
        spec.source_id,
        "complete",
        [asset, locations],
        "DCRNN author Drive object; sensor coordinates pinned to the DCRNN commit",
    )


def _acquire_noaa(
    root: Path,
    session: object,
    *,
    year: int,
    station_count: int,
) -> AcquisitionResult:
    spec = SOURCE_SPECS["noaa_global_hourly"]
    catalog_asset = _download_http(
        session,
        "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv",
        root / "raw" / spec.source_id / "isd-history.csv",
        source_revision=spec.source_revision,
    )
    catalog = pd.read_csv(Path(getattr(catalog_asset, "path")), dtype={"USAF": str, "WBAN": str})
    stations = select_noaa_stations(catalog, year=year, count=station_count)
    assets: list[object] = [catalog_asset]
    failures: list[str] = []
    for station in stations:
        try:
            asset = _download_http(
                session,
                f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv",
                root / "raw" / spec.source_id / str(year) / f"{station}.csv",
                source_revision=spec.source_revision,
            )
            assets.append(asset)
        except Exception as exc:  # no refill: retain the frozen failed station identity
            failures.append(f"{station}:{type(exc).__name__}")
    selection_path = root / "raw" / spec.source_id / f"station_selection_{year}.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_payload = {
        "benchmark_version": "benchmark-v0",
        "catalog_sha256": getattr(catalog_asset, "sha256"),
        "year": year,
        "station_count": station_count,
        "selected_stations": list(stations),
        "download_failures_no_refill": failures,
    }
    from .materialize import write_text_lf

    write_text_lf(
        selection_path,
        json.dumps(selection_payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n",
    )
    status = "complete" if not failures else "partial_download_failed_no_refill"
    return _asset_result(spec.source_id, status, assets, f"NOAA station failures: {failures}")


def acquire_all_sources(
    root: Path | str,
    *,
    automatic: bool = True,
    noaa_year: int = 2024,
    noaa_station_count: int = NOAA_STATION_COUNT,
    http_session: object | None = None,
) -> tuple[AcquisitionResult, ...]:
    """Acquire every approved automatic source and account for every manual source."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    plan = build_acquisition_plan(base)
    results: list[AcquisitionResult] = []
    if automatic:
        if http_session is None:
            import requests

            http_session = requests.Session()
        operations = (
            ("monash_hf", lambda: _acquire_monash(base)),
            ("metr_la", lambda: _acquire_metr_la(base, http_session)),
            (
                "uci_electricity_load_diagrams",
                lambda: _acquire_uci(base, http_session),
            ),
            (
                "noaa_global_hourly",
                lambda: _acquire_noaa(
                    base,
                    http_session,
                    year=noaa_year,
                    station_count=noaa_station_count,
                ),
            ),
        )
        for source_id, operation in operations:
            try:
                results.append(operation())
            except Exception as exc:
                results.append(
                    AcquisitionResult(
                        source_id,
                        "download_failed",
                        (),
                        (),
                        f"{type(exc).__name__}: {exc}",
                    )
                )
    else:
        results.extend(
            AcquisitionResult(source_id, "automatic_skipped", (), (), "automatic acquisition disabled")
            for source_id in AUTOMATIC_SOURCE_IDS
        )
    for row in plan:
        if row.access == "manual":
            spec = SOURCE_SPECS[row.source_id]
            assert spec.incoming_subdir is not None
            incoming = base / "incoming" / spec.incoming_subdir
            files = sorted(
                path for path in incoming.rglob("*")
                if path.is_file() and not path.name.endswith(".asset.json")
            ) if incoming.is_dir() else []
            bound_assets: list[object] = []
            if files:
                from .materialize import RawAsset, verify_raw_asset

                for path in files:
                    sidecar = path.with_name(path.name + ".asset.json")
                    if not sidecar.is_file():
                        bound_assets = []
                        break
                    payload = json.loads(sidecar.read_text("utf-8"))
                    bound_assets.append(
                        verify_raw_asset(
                            RawAsset(
                                path,
                                str(payload["sha256"]),
                                str(payload["source_revision"]),
                                int(payload["size"]),
                            )
                        )
                    )
            if bound_assets and len(bound_assets) == len(files):
                results.append(
                    _asset_result(
                        row.source_id,
                        "manual_bound",
                        bound_assets,
                        "manual source files are immutably hash-bound",
                    )
                )
                continue
            results.append(
                AcquisitionResult(
                    row.source_id,
                    row.status,
                    (),
                    (),
                    f"place untouched files under {row.destination}",
                )
            )
    ordered = tuple(sorted(results, key=lambda row: row.source_id))
    manifest = {
        "schema_version": "benchmark-acquisition/1",
        "benchmark_version": "benchmark-v0",
        "automatic_requested": automatic,
        "results": [row.to_dict() for row in ordered],
    }
    from .materialize import write_text_lf

    write_text_lf(
        base / "acquisition_manifest.json",
        json.dumps(manifest, sort_keys=True, ensure_ascii=True, indent=2) + "\n",
    )
    return ordered


def get_source_spec(source_id: str) -> SourceSpec:
    try:
        return SOURCE_SPECS[source_id]
    except KeyError as exc:
        raise KeyError(f"unregistered benchmark source: {source_id!r}") from exc
