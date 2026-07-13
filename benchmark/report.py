"""Dev-only discrimination report and frozen saturation labels."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import SATURATION_GAP, SATURATION_GAP_KIND

# A cell mean sMASE above this is almost certainly a denominator artifact rather than a
# forecast collapse; on this roster only cumulative COVID counts reach it.
_SCALE_WARNING_SMASE = 10.0


class ReportProtocolError(ValueError):
    """A report was requested from the wrong split or malformed rows."""


@dataclass(frozen=True)
class DevDiscriminationRow:
    split_role: str
    dataset_id: str
    regime: str
    uid: str
    h_ref_loss: float
    oracle_insample_loss: float

    def __post_init__(self) -> None:
        strings = (self.split_role, self.dataset_id, self.regime, self.uid)
        if any(not isinstance(item, str) or not item or item != item.strip() for item in strings):
            raise ValueError("report identifiers must be canonical non-empty strings")
        if not np.isfinite([self.h_ref_loss, self.oracle_insample_loss]).all():
            raise ValueError("report losses must be finite")


def build_dev_discrimination_report(
    rows: Sequence[DevDiscriminationRow],
    *,
    raw_loss_by_uid: Mapping[str, float] | None = None,
    seasonal_scale_by_uid: Mapping[str, float] | None = None,
    saturation_gap: float = SATURATION_GAP,
    min_uid: int = 12,
) -> dict:
    """Per-cell saturation diagnosis, annotated so the numbers cannot be misread.

    Two annotations exist because both have already misled a reader of this report:

    `seasonal_scale`
        sMASE divides by the mean absolute seasonal difference of the clean inner-train.
        A near-monotone series (cumulative COVID deaths) has a tiny denominator, so its
        sMASE lands in the hundreds while a traffic sensor sits near 1.  That is a scale
        artifact, not a catastrophic forecast, and a cell mean cannot be compared across
        datasets without it.

    `oracle_reverts_to_raw`
        If the oracle's per-cell pick is Raw, then its "gain over H_ref" is precisely the
        harm H_ref did -- refunded, not discovered.  Such a cell is not repair space.
    """
    rows = list(rows)
    if not rows or any(row.split_role != "dev_query" for row in rows):
        raise ReportProtocolError("saturation diagnosis can be created only from Dev-Query")
    if not np.isfinite(saturation_gap) or saturation_gap < 0:
        raise ValueError("saturation gap must be finite and non-negative")
    if isinstance(min_uid, bool) or not isinstance(min_uid, int) or min_uid < 1:
        raise ValueError("minimum uid count must be a positive integer")
    grouped: dict[tuple[str, str], list[DevDiscriminationRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.dataset_id, row.regime)].append(row)
    cells: dict[str, dict] = {}
    for (dataset_id, regime), group in sorted(grouped.items()):
        uids = [row.uid for row in group]
        if len(uids) != len(set(uids)):
            raise ReportProtocolError("Dev discrimination requires one row per uid per cell")
        gain = float(np.mean([row.h_ref_loss - row.oracle_insample_loss for row in group]))
        if len(group) < min_uid:
            tag = "diagnostic_unavailable"
        elif abs(gain) <= saturation_gap:
            tag = "saturated_under_pool_v1"
        else:
            tag = "discriminating_under_pool_v1"
        cell = {
            "dataset_id": dataset_id,
            "regime": regime,
            "n_uid": len(group),
            "oracle_insample_gain_over_h_ref": gain,
            "tag": tag,
            "h_ref_mean_loss": float(np.mean([row.h_ref_loss for row in group])),
            "oracle_insample_mean_loss": float(
                np.mean([row.oracle_insample_loss for row in group])
            ),
        }
        if raw_loss_by_uid is not None:
            raw = [raw_loss_by_uid[row.uid] for row in group if row.uid in raw_loss_by_uid]
            if raw:
                raw_mean = float(np.mean(raw))
                cell["raw_mean_loss"] = raw_mean
                cell["oracle_insample_gain_over_raw"] = (
                    raw_mean - cell["oracle_insample_mean_loss"]
                )
                cell["h_ref_self_harm_vs_raw"] = cell["h_ref_mean_loss"] - raw_mean
        if seasonal_scale_by_uid is not None:
            scales = [
                seasonal_scale_by_uid[row.uid]
                for row in group
                if row.uid in seasonal_scale_by_uid
            ]
            if scales:
                cell["seasonal_scale_median"] = float(np.median(scales))
                cell["seasonal_scale_min"] = float(np.min(scales))
                cell["scale_warning"] = bool(
                    cell["h_ref_mean_loss"] > _SCALE_WARNING_SMASE
                )
        cells[f"{dataset_id}|{regime}"] = cell
    return {
        "split_role": "dev_query",
        "saturation_gap": float(saturation_gap),
        "saturation_gap_kind": SATURATION_GAP_KIND,
        "min_uid": min_uid,
        "annotation_note": (
            "scale_warning marks cells whose sMASE is inflated by a tiny seasonal "
            "denominator, not by a bad forecast; their losses are not comparable across "
            "datasets. oracle_insample_gain_over_raw is the headroom above the No-op "
            "floor -- prefer it over gain_over_h_ref, which is inflated wherever H_ref hurt."
        ),
        "cells": cells,
    }


__all__ = [
    "DevDiscriminationRow",
    "ReportProtocolError",
    "build_dev_discrimination_report",
]
