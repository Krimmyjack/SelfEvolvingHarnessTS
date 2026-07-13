"""Dev-only discrimination report and frozen saturation labels."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import SATURATION_GAP, SATURATION_GAP_KIND


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
    saturation_gap: float = SATURATION_GAP,
    min_uid: int = 12,
) -> dict:
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
        cells[f"{dataset_id}|{regime}"] = {
            "dataset_id": dataset_id,
            "regime": regime,
            "n_uid": len(group),
            "oracle_insample_gain_over_h_ref": gain,
            "tag": tag,
        }
    return {
        "split_role": "dev_query",
        "saturation_gap": float(saturation_gap),
        "saturation_gap_kind": SATURATION_GAP_KIND,
        "min_uid": min_uid,
        "cells": cells,
    }


__all__ = [
    "DevDiscriminationRow",
    "ReportProtocolError",
    "build_dev_discrimination_report",
]
