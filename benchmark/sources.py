"""Frozen acquisition specifications for the approved benchmark sources.

This module describes where bytes may come from; it deliberately performs no
network or filesystem acquisition.  Automatic sources must name an immutable
object or a frozen project revision, while account-gated portals must declare
the directory into which user-supplied exports are imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from typing import Literal
from urllib.parse import urlparse

__all__ = ["SOURCE_SPECS", "SourceSpec", "get_source_spec"]


AccessMode = Literal["automatic", "manual"]


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    access: AccessMode
    official_url: str
    source_revision: str
    license_id: str
    expected_frequency: str
    overlap_family: str
    incoming_subdir: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "official_url",
            "source_revision",
            "license_id",
            "expected_frequency",
            "overlap_family",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field_name} must be a canonical non-empty string")
        parsed = urlparse(self.official_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("official_url must be an absolute HTTPS URL")
        if self.access not in ("automatic", "manual"):
            raise ValueError("access must be 'automatic' or 'manual'")
        if self.access == "automatic":
            if self.incoming_subdir is not None:
                raise ValueError("automatic sources cannot define incoming_subdir")
            if self.source_revision.lower() in {"main", "master", "head", "latest"}:
                raise ValueError("source_revision must pin an automatic source")
        else:
            if (
                not isinstance(self.incoming_subdir, str)
                or not self.incoming_subdir
                or self.incoming_subdir != self.incoming_subdir.strip()
            ):
                raise ValueError("manual sources require incoming_subdir")
            if any(part in {"", ".", ".."} for part in self.incoming_subdir.replace("\\", "/").split("/")):
                raise ValueError("incoming_subdir must be a safe relative directory")


_SPECS = (
    SourceSpec(
        source_id="monash_hf",
        access="automatic",
        official_url="https://huggingface.co/datasets/monash_tsf",
        source_revision="refs/convert/parquet",
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
        source_revision="drive-object-10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX",
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
        license_id="cc-by-4.0",
        expected_frequency="15min",
        overlap_family="electricity_load_diagrams_2011_2014",
    ),
    SourceSpec(
        source_id="noaa_global_hourly",
        access="automatic",
        official_url="https://www.ncei.noaa.gov/data/global-hourly/access/",
        source_revision="ncei-global-hourly-catalog-2026-07-13",
        license_id="us-public-domain",
        expected_frequency="irregular_hourly",
        overlap_family="noaa_isd",
    ),
    SourceSpec(
        source_id="entsoe_transparency",
        access="manual",
        official_url="https://transparency.entsoe.eu/",
        source_revision="actual-total-load-export-v1",
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
