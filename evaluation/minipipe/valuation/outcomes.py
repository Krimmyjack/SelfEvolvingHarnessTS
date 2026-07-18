from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class OutcomeView:
    clean_u: float
    corrupt_u: float
    prepared_u: float
    damage_d: float
    repair_gain_g: float
    nrr: float | None
    over_restoration: bool
    selection_regret: float
    target_window_gain: float | None
    outside_window_change: float | None
    counterpart_change: float | None
    non_target_collateral: float | None
    agent_decision_status: str
    system_capability_status: str


def _finite(value: float, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def evaluate_candidate_regret(
    candidate_utilities: Mapping[str, float],
    *,
    chosen_candidate_id: str,
    identity_u: float,
) -> float:
    utilities = {
        str(candidate_id): _finite(utility, field=f"utility[{candidate_id}]")
        for candidate_id, utility in candidate_utilities.items()
    }
    identity = _finite(identity_u, field="identity_u")
    if "identity" in utilities and utilities["identity"] != identity:
        raise ValueError("identity candidate utility must equal identity_u")
    utilities["identity"] = identity
    if chosen_candidate_id not in utilities:
        raise ValueError("chosen candidate is absent from the evaluated pool")
    return max(utilities.values()) - utilities[chosen_candidate_id]


def evaluate_outcome(
    *,
    clean_u: float,
    corrupt_u: float,
    prepared_u: float,
    identity_u: float,
    candidate_utilities: Mapping[str, float],
    chosen_candidate_id: str,
    damage_noise_floor: float,
    utility_tolerance: float = 1e-6,
    target_window_gain: float | None = None,
    outside_window_change: float | None = None,
    counterpart_change: float | None = None,
    non_target_collateral: float | None = None,
    agent_decision_status: str = "UNASSESSED",
    system_capability_status: str = "UNASSESSED",
) -> OutcomeView:
    clean = _finite(clean_u, field="clean_u")
    corrupt = _finite(corrupt_u, field="corrupt_u")
    prepared = _finite(prepared_u, field="prepared_u")
    floor = _finite(damage_noise_floor, field="damage_noise_floor")
    tolerance = _finite(utility_tolerance, field="utility_tolerance")
    if floor < 0.0 or tolerance < 0.0:
        raise ValueError("outcome thresholds must be non-negative")
    damage = clean - corrupt
    gain = prepared - corrupt
    nrr = gain / damage if damage > floor else None
    regret = evaluate_candidate_regret(
        candidate_utilities,
        chosen_candidate_id=chosen_candidate_id,
        identity_u=identity_u,
    )

    def optional(value: float | None, field: str) -> float | None:
        return None if value is None else _finite(value, field=field)

    return OutcomeView(
        clean_u=clean,
        corrupt_u=corrupt,
        prepared_u=prepared,
        damage_d=damage,
        repair_gain_g=gain,
        nrr=nrr,
        over_restoration=(prepared - clean) > tolerance,
        selection_regret=regret,
        target_window_gain=optional(target_window_gain, "target_window_gain"),
        outside_window_change=optional(
            outside_window_change,
            "outside_window_change",
        ),
        counterpart_change=optional(counterpart_change, "counterpart_change"),
        non_target_collateral=optional(
            non_target_collateral,
            "non_target_collateral",
        ),
        agent_decision_status=str(agent_decision_status),
        system_capability_status=str(system_capability_status),
    )


__all__ = ["OutcomeView", "evaluate_candidate_regret", "evaluate_outcome"]
