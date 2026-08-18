"""C0 public Context inlet helper for the natural forward Runner.

Frozen under docs/EXPERIENCE_TO_SKILL_CARD_EVOLUTION_PLAN_2026-08-17.md §3.3:

* every Task Episode computes its deployment-visible observations from the
  public prefix ending at that Episode's first Support origin;
* the scope-selector rule is the existing workflow-generation representative
  rule: high ``local_robust_z_peak`` train series, representative = max
  ``local_robust_z_peak`` within that scope (ties broken by sorted uid);
* task-level ``fast_features`` and Card ``observable_signature`` derive from
  that same deterministic, outcome-blind projection.

The frozen narrowing rule only added one existing numeric public feature from
the closed observation vocabulary.  The feature below is the first feature in
``OBSERVABLE_FEATURES`` code order whose representative-series bin varies
across the already-exposed natural K1 Task Episodes at their own cutoffs; it
was selected for Task Context variation, never for gain.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.observables import observable_numeric_bin
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    extract_public_features,
)

PUBLIC_CONTEXT_TASK_KIND = "forecast"
PUBLIC_CONTEXT_SCOPE_FEATURE = "local_robust_z_peak"
PUBLIC_CONTEXT_SCOPE_BIN = "high"
PUBLIC_CONTEXT_PROJECTION_FEATURE = "estimated_region_start_fraction"

# Frozen after the zero-outcome C0 census over natural_k1_01..04 (2026-08-17).
C0_MATCHING_TASK_ID = "natural_k1_03"
C0_NON_MATCHING_TASK_ID = "natural_k1_04"

def _plain(value: Any) -> Any:
    """Convert extractor/numpy scalars to JSON-native values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(nested) for nested in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_public_prefix(
    values: Mapping[str, Any],
    train_uids: Sequence[str],
    observation_cutoff: int,
) -> dict[str, np.ndarray]:
    if not isinstance(observation_cutoff, int):
        raise TypeError("observation_cutoff must be an integer origin index")
    if observation_cutoff <= 0:
        raise ValueError("observation_cutoff must be a positive origin index")
    if not train_uids:
        raise ValueError("train_uids must not be empty")
    prefixes: dict[str, np.ndarray] = {}
    for uid in train_uids:
        key = str(uid)
        series = np.asarray(values[key], dtype=np.float64)
        if series.ndim != 1 or series.size < observation_cutoff:
            raise ValueError(
                f"series {key!r} is shorter than observation_cutoff "
                f"{observation_cutoff}"
            )
        prefixes[key] = series[:observation_cutoff].copy()
    return prefixes


def build_task_public_context(
    values: Mapping[str, Any],
    train_uids: Sequence[str],
    observation_cutoff: int,
) -> dict[str, Any]:
    """Compute one Task Episode's public Context projection.

    This helper never reads any Support/Query outcome.  It only slices
    ``values[uid][:observation_cutoff]`` and calls the existing public feature
    extractor on those prefixes.
    """
    train_uids = [str(uid) for uid in train_uids]
    prefixes = _require_public_prefix(values, train_uids, observation_cutoff)
    per_series_features: dict[str, dict[str, Any]] = {}
    for uid in train_uids:
        features = dict(
            extract_public_features(
                prefixes[uid], task_kind=PUBLIC_CONTEXT_TASK_KIND
            )
        )
        per_series_features[uid] = _plain(features)

    scope = frozenset(
        uid
        for uid in train_uids
        if observable_numeric_bin(
            PUBLIC_CONTEXT_SCOPE_FEATURE,
            float(per_series_features[uid][PUBLIC_CONTEXT_SCOPE_FEATURE]),
        )
        == PUBLIC_CONTEXT_SCOPE_BIN
    )

    representative_uid: str | None
    representative_features: dict[str, Any]
    task_fast_features: dict[str, Any]
    task_signature: dict[str, Any]
    if scope:
        representative_uid = max(
            sorted(scope),
            key=lambda uid: float(
                per_series_features[uid][PUBLIC_CONTEXT_SCOPE_FEATURE]
            ),
        )
        representative_features = dict(per_series_features[representative_uid])
        projection_bin = observable_numeric_bin(
            PUBLIC_CONTEXT_PROJECTION_FEATURE,
            float(
                representative_features[PUBLIC_CONTEXT_PROJECTION_FEATURE]
            ),
        )
        task_signature = {
            "task_kind": PUBLIC_CONTEXT_TASK_KIND,
            PUBLIC_CONTEXT_PROJECTION_FEATURE: projection_bin,
        }
        task_fast_features = dict(representative_features)
    else:
        representative_uid = None
        representative_features = {}
        task_fast_features = {"task_kind": PUBLIC_CONTEXT_TASK_KIND}
        task_signature = {"task_kind": PUBLIC_CONTEXT_TASK_KIND}

    context = {
        "task_kind": PUBLIC_CONTEXT_TASK_KIND,
        "observation_cutoff": int(observation_cutoff),
        "scope_feature": PUBLIC_CONTEXT_SCOPE_FEATURE,
        "scope_bin": PUBLIC_CONTEXT_SCOPE_BIN,
        "projection_feature": PUBLIC_CONTEXT_PROJECTION_FEATURE,
        "scope_series_uids": sorted(scope),
        "representative_uid": representative_uid,
        "representative_features": representative_features,
        "task_signature": task_signature,
        "task_fast_features": task_fast_features,
        "per_series_features": per_series_features,
    }
    return context


def run_context_census(
    task_contexts: Mapping[str, dict[str, Any]],
    *,
    matching_task_id: str = C0_MATCHING_TASK_ID,
    non_matching_task_id: str = C0_NON_MATCHING_TASK_ID,
) -> dict[str, Any]:
    """Zero-outcome C0 census over already-computed task public contexts."""
    contexts = {str(task_id): context for task_id, context in task_contexts.items()}
    task_signatures = {
        task_id: _plain(context["task_signature"])
        for task_id, context in contexts.items()
    }
    distinct_signatures: list[dict[str, Any]] = []
    for signature in task_signatures.values():
        if signature not in distinct_signatures:
            distinct_signatures.append(dict(signature))
    matching = contexts.get(matching_task_id)
    non_matching = contexts.get(non_matching_task_id)
    matching_signature = (
        _plain(matching["task_signature"]) if matching is not None else None
    )
    non_matching_signature = (
        _plain(non_matching["task_signature"])
        if non_matching is not None
        else None
    )
    matching_usable = bool(
        matching_signature is not None
        and non_matching_signature is not None
        and matching_signature != non_matching_signature
    )
    verdict = (
        "TASK_CONTEXT_INLET_BINDING_PASS"
        if len(distinct_signatures) >= 2 and matching_usable
        else "TASK_CONTEXT_INLET_NOT_DISTINGUISHABLE"
    )
    return {
        "task_ids": sorted(contexts),
        "task_signatures": task_signatures,
        "distinct_signature_count": len(distinct_signatures),
        "distinct_signatures": distinct_signatures,
        "matching_task_id": matching_task_id,
        "non_matching_task_id": non_matching_task_id,
        "matching_signature": matching_signature,
        "non_matching_signature": non_matching_signature,
        "matching_usable": matching_usable,
        "verdict": verdict,
    }
