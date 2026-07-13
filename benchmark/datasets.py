"""Dataset-level properties that the per-series registry cannot express.

`broad_domain`, `network_id`, and `claim_tier` are attributes of a *dataset*, not of a
series, so they hang beside the frozen registry rather than inside it -- adding them as
registry columns would force a schema bump and rewrite bytes that are already hashed
into the benchmark manifest.

Three things live here that the arena needs and the registry cannot say:

`claim_tier`
    Not every dataset carries a headline claim.  GEFCom2012 contributes 20 zones, of
    which only a handful reach Final-Query -- far too few to support an independent
    per-dataset conclusion.  It is marked `supplementary`: it enters the pooled roster
    and the per-cell tables, but a result on GEFCom alone is not reportable as a
    finding.  Marking that *before* results exist is the point; deciding afterwards
    which datasets "count" is how a null becomes a discovery.

`network_id`
    Traffic series are sensors on a road graph, not independent entities.  Two datasets
    sharing a `network_id` cannot be used as independent evidence for each other.
    METR-LA (Los Angeles) and Monash traffic_hourly (San Francisco Bay Area) are
    distinct networks, which is exactly what makes a cross-network claim possible --
    they test each other.

`independence`
    `single_network` says the dataset's own series are spatially coupled, so any
    within-dataset claim is bounded to "unseen entity in THIS network".  METR-LA
    mitigates this with spatial blocking (see `spatial.py`); Monash traffic_hourly
    *cannot*, because the pinned Monash release ships no sensor coordinates.  That
    asymmetry is recorded, not smoothed over.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

__all__ = [
    "CLAIM_TIERS",
    "DATASET_MANIFEST",
    "DatasetProperties",
    "dataset_manifest_document",
    "headline_datasets",
]

CLAIM_TIERS = ("headline", "supplementary", "dev_only")


class DatasetProperties(dict):
    """Frozen dataset-level record; a dict so it serializes without ceremony."""

    def __init__(
        self,
        *,
        broad_domain: str,
        claim_tier: str,
        independence: str,
        network_id: str | None = None,
        note: str = "",
    ) -> None:
        if claim_tier not in CLAIM_TIERS:
            raise ValueError(f"claim_tier must be one of {CLAIM_TIERS!r}")
        if independence not in ("independent_entities", "single_network"):
            raise ValueError("independence must be independent_entities or single_network")
        if independence == "single_network" and not network_id:
            raise ValueError("a single_network dataset must name its network_id")
        super().__init__(
            broad_domain=broad_domain,
            claim_tier=claim_tier,
            independence=independence,
            network_id=network_id,
            note=note,
        )


_MANIFEST: Mapping[str, DatasetProperties] = {
    "metr_la": DatasetProperties(
        broad_domain="traffic",
        claim_tier="headline",
        independence="single_network",
        network_id="la_freeway_2012",
        note=(
            "207 loop detectors on the Los Angeles freeway graph. Split is atomic at "
            "the spatial block (20 blocks over 137 co-location-merged sites), so "
            "adjacent sensors cannot straddle Support and Query. Claim is bounded to "
            "unseen blocks within this network; cross-network evidence must come from "
            "monash:traffic_hourly, which is a different city."
        ),
    ),
    "monash:traffic_hourly": DatasetProperties(
        broad_domain="traffic",
        claim_tier="headline",
        independence="single_network",
        network_id="bay_area_freeway_2015",
        note=(
            "862 sensors on the San Francisco Bay Area freeway graph. The pinned Monash "
            "release ships NO sensor coordinates, so unlike METR-LA this dataset CANNOT "
            "be spatially blocked and its split stays per-series. Adjacent-sensor "
            "leakage across roles is therefore possible and un-quantified; treat "
            "within-dataset numbers as an optimistic bound and rely on METR-LA for the "
            "spatially clean traffic read."
        ),
    ),
    "uci_electricity_load_diagrams": DatasetProperties(
        broad_domain="energy",
        claim_tier="headline",
        independence="independent_entities",
        note="370 independent Portuguese client meters, 15-minute kW resampled to hourly means.",
    ),
    "monash:covid_deaths": DatasetProperties(
        broad_domain="epidemiology",
        claim_tier="headline",
        independence="independent_entities",
        note=(
            "Cumulative national COVID death counts. Near-monotone, tiny seasonal "
            "differences, so the sMASE denominator is small and losses live on a much "
            "larger numeric scale than the rest of the roster. Per-cell tables carry a "
            "scale annotation; do not read its sMASE against other datasets' unscaled."
        ),
    ),
    "monash:nn5_daily": DatasetProperties(
        broad_domain="finance",
        claim_tier="headline",
        independence="independent_entities",
        note="Daily cash withdrawals at UK ATMs.",
    ),
    "gefcom2012_load": DatasetProperties(
        broad_domain="energy",
        claim_tier="supplementary",
        independence="independent_entities",
        note=(
            "20 utility zones from the GEFCom2012 load track. Too few series to reach a "
            "reportable Final-Query sample on its own, so it is supplementary: it joins "
            "the pooled roster and the per-cell tables, but no standalone GEFCom claim "
            "is reportable. Zones within one utility are also plausibly correlated."
        ),
    ),
    "noaa_global_hourly": DatasetProperties(
        broad_domain="weather",
        claim_tier="headline",
        independence="independent_entities",
        note=(
            "NOAA ISD hourly surface temperature, the sealed U (unseen-domain) pool. "
            "Stations are geographically dispersed across the US by a hash-ordered "
            "selection. Carries heavy, genuinely natural missingness -- which is the "
            "point: it is the one source whose defects nobody injected."
        ),
    ),
}

# Legacy Monash series are Support-A-only by exposure, never Query, so they can never
# carry a claim regardless of tier.
for _legacy in (
    "legacy_monash:nn5_daily",
    "legacy_monash:fred_md",
    "legacy_monash:tourism_monthly",
    "legacy_monash:covid_deaths",
    "legacy_monash:us_births",
    "legacy_monash:saugeenday",
    "legacy_monash:sunspot",
):
    _MANIFEST[_legacy] = DatasetProperties(
        broad_domain="legacy_mixed",
        claim_tier="dev_only",
        independence="independent_entities",
        note=(
            "Init Harness / pre-benchmark exposed corpus. Support-A only by exposure "
            "class; never eligible for any Query role."
        ),
    )

DATASET_MANIFEST: Mapping[str, DatasetProperties] = MappingProxyType(dict(_MANIFEST))


def headline_datasets() -> tuple[str, ...]:
    """Datasets whose per-dataset results are reportable as findings."""
    return tuple(
        sorted(
            dataset
            for dataset, properties in DATASET_MANIFEST.items()
            if properties["claim_tier"] == "headline"
        )
    )


def dataset_manifest_document(observed_datasets: Mapping[str, int]) -> dict[str, object]:
    """Bind the dataset manifest to the datasets a probe actually produced."""
    unknown = sorted(set(observed_datasets) - set(DATASET_MANIFEST))
    if unknown:
        raise KeyError(
            "probe produced datasets with no declared claim tier or domain: "
            f"{unknown}. Declare them in benchmark/datasets.py before freezing."
        )
    return {
        "schema_version": "benchmark-dataset-manifest/1",
        "claim_tiers": list(CLAIM_TIERS),
        "headline_datasets": list(headline_datasets()),
        "datasets": {
            dataset: dict(DATASET_MANIFEST[dataset], n_series=count)
            for dataset, count in sorted(observed_datasets.items())
        },
        "network_groups": {
            network: sorted(
                dataset
                for dataset, properties in DATASET_MANIFEST.items()
                if properties["network_id"] == network and dataset in observed_datasets
            )
            for network in sorted(
                {
                    properties["network_id"]
                    for dataset, properties in DATASET_MANIFEST.items()
                    if properties["network_id"] and dataset in observed_datasets
                }
            )
        },
    }
