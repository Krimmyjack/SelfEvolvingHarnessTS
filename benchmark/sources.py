"""Frozen, verifiable acquisition specifications for benchmark sources."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping
from urllib.parse import urlparse

__all__ = [
    "SOURCE_SPECS",
    "RevisionKind",
    "SourceSpec",
    "get_source_spec",
]


AccessMode = Literal["automatic", "manual"]
RevisionKind = Literal[
    "git_commit", "object_id", "catalog_snapshot", "manual_export"
]
_REVISION_KINDS = frozenset(
    {"git_commit", "object_id", "catalog_snapshot", "manual_export"}
)


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


def get_source_spec(source_id: str) -> SourceSpec:
    try:
        return SOURCE_SPECS[source_id]
    except KeyError as exc:
        raise KeyError(f"unregistered benchmark source: {source_id!r}") from exc
