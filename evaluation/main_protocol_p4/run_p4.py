"""Derive the v1.2 P4 split release without changing the P3 receipt.

P4-Performance collects RQ1/RQ2 evidence.  P4-Evolution is the independent
RQ3/H3 claim gate.  P4-AD is restricted to conditioning and safety.  The
historical P3 result remains an input, never an output of this module.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
P3_REPORT = (
    PROJECT_ROOT
    / "artifacts/main_protocol/p3_unified_integration_gate_20260830.json"
)
OUT_JSON = (
    PROJECT_ROOT
    / "artifacts/main_protocol/p4_split_gate_forecast_b8_llm8_20260830.json"
)

PROTOCOL_VERSION = "v1.2.1-Core+p4-split-3-forecast-b8-llm8"
ARMS = ("Static", "A3-reset", "K0-fixed", "A5-online")
REPLICAS = ("Forward", "Reverse", "Interleaved")
EPISODES_PER_TASK = 8
FORECAST_B_MAIN = 8
FORECAST_MAX_SUPPORT_A = 7
FORECAST_MAX_SUPPORT_B = 1
FORECAST_MAX_CHEAP_PROBES = 24
FORECAST_MAX_LLM_CALLS = 8
FORECAST_MAX_TOKENS = 60_000
FORECAST_MAX_UPDATES = 1
FORECAST_MAX_WALL_SECONDS = 45 * 60


class P4GateBlocked(RuntimeError):
    """The historical P3 receipt cannot support any P4 collection."""


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P4GateBlocked("P3 receipt must be a JSON object")
    return value


def _boundary_failures(p3: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if p3.get("p3_integration_complete") is not True:
        failures.append("P3 integration is incomplete")
    errors = dict(p3.get("protocol_errors") or {})
    if not errors or any(int(value or 0) != 0 for value in errors.values()):
        failures.append("P3 protocol-error counters are missing or nonzero")
    if p3.get("live_outcome_release") is not False:
        failures.append("P3 live-outcome boundary is not closed")
    if p3.get("p4_executed") is not False:
        failures.append("historical P3 receipt already records P4 execution")
    claims = dict(p3.get("claim_boundaries") or {})
    if claims.get("natural_final_release") is not False:
        failures.append("Natural Final boundary is not closed")
    return failures


def derive_split_gate(p3: Mapping[str, Any]) -> dict[str, Any]:
    """Apply v1.2 sections 12.2, 13.1 and 14 as independent gates."""
    failures = _boundary_failures(p3)
    base_ready = not failures
    treatment = dict(p3.get("derived_treatment_state") or {})
    forecast_state = dict(treatment.get("forecast") or {}).get("state")
    classification_state = dict(treatment.get("classification") or {}).get(
        "state"
    )
    ad_state = dict(treatment.get("anomaly_detection") or {}).get("state")

    performance = {
        "gate": "P4_PERFORMANCE",
        "status": "RELEASED" if base_ready else "HELD",
        "answers": ["H1", "H2"],
        "release_by_task": {
            "forecast": bool(base_ready),
            "classification": bool(base_ready),
            "anomaly_detection": False,
        },
        "forecast_launch_authorized": bool(base_ready),
        "classification_launch_authorized": bool(base_ready),
        "ad_positive_performance_claim_authorized": False,
        "rq3_not_exercised_is_not_a_blocker": True,
        "failures": failures,
    }
    evolution = {
        "gate": "P4_EVOLUTION",
        "status": "HELD",
        "answers": ["H3"],
        "rq3_status_by_task": {
            "forecast": "RQ3_NOT_EXERCISED",
            "classification": "RQ3_NOT_EXERCISED",
            "anomaly_detection": "RQ3_NOT_EXERCISED",
        },
        "observed_p3_treatment_state": {
            "forecast": forecast_state,
            "classification": classification_state,
            "anomaly_detection": ad_state,
        },
        "required_chain": [
            "production pending update",
            "independent Support-B approval",
            "promotion",
            "versioned revision",
            "independent later re-encounter",
            "material utility or cost improvement over K0-fixed",
        ],
        "does_not_block_h1_h2": True,
    }
    ad = {
        "gate": "P4_AD",
        "status": "RELEASED_CONDITIONING_AND_SAFETY_ONLY" if base_ready else "HELD",
        "allowed_evidence": [
            "Task/Consumer conditioning",
            "signal protection",
            "safe refusal",
            "no-negative-transfer diagnostics",
        ],
        "positive_performance_claim_authorized": False,
        "online_revision_claim_authorized": False,
        "consumer_metric_or_matching_change_authorized": False,
        "main_interpretation": "INVERTED_EFFECT_OBSERVED",
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "P4_SPLIT_RELEASE_GATE",
        "historical_p3_verdict_preserved": p3.get("verdict"),
        "historical_p3_release_p4_preserved": p3.get("release_p4"),
        "p4_performance": performance,
        "p4_evolution": evolution,
        "p4_ad": ad,
        "execution_plan": {
            "episodes_per_task": EPISODES_PER_TASK,
            "replicas": list(REPLICAS),
            "arms": list(ARMS),
            "budget_scope": "FORECAST_P4_PERFORMANCE_ONLY",
            "full_support_budget": FORECAST_B_MAIN,
            "support_a_budget": FORECAST_MAX_SUPPORT_A,
            "support_b_budget": FORECAST_MAX_SUPPORT_B,
            "cheap_probe_budget": FORECAST_MAX_CHEAP_PROBES,
            "llm_call_budget": FORECAST_MAX_LLM_CALLS,
            "token_budget": FORECAST_MAX_TOKENS,
            "accepted_update_budget": FORECAST_MAX_UPDATES,
            "wall_seconds_budget": FORECAST_MAX_WALL_SECONDS,
            "forecast_budget": {
                "operating_point": "B=8",
                "full_support_consumer_evaluations": FORECAST_B_MAIN,
                "support_a_max": FORECAST_MAX_SUPPORT_A,
                "support_b_max": FORECAST_MAX_SUPPORT_B,
                "cheap_probe_max": FORECAST_MAX_CHEAP_PROBES,
                "llm_call_max": FORECAST_MAX_LLM_CALLS,
                "token_max": FORECAST_MAX_TOKENS,
                "accepted_update_max": FORECAST_MAX_UPDATES,
                "wall_seconds_max": FORECAST_MAX_WALL_SECONDS,
            },
            "classification_budget": {
                "operating_point": "B=4",
                "status": "UNCHANGED_BY_FORECAST_SPLIT_2",
            },
            "ad_budget": {
                "operating_point": "B=4",
                "status": "SAFETY_ONLY__NO_POSITIVE_PERFORMANCE_RUN",
            },
            "matched_baseline": "Parallel Best-of-N@8",
            "matched_budget": True,
            "adaptive_arms_share_exact_budget_vector": True,
            "a5_budget_exception": False,
            "cell_llm_budget_exhaustion_action": (
                "ABSTAIN_TO_IDENTITY_AND_CONTINUE"
            ),
            "cell_llm_budget_exhaustion_reason": (
                "LLM_CELL_BUDGET_EXHAUSTED"
            ),
            "partial_cell_state_writeback": False,
            "budget_exhaustion_rate_reported_by_arm": True,
            "k0_a5_same_initial_knowledge": True,
            "a5_only_arm_with_cross_unit_writeback": True,
            "task_local_experience_and_skill_only": True,
        },
        "next_stage_release": False,
        "natural_final_release": False,
        "final_outcome_reads": 0,
        "p3_runner_invocations": 0,
        "p3_report_writes": 0,
        "blocking_failures": failures,
    }


def run() -> dict[str, Any]:
    payload = derive_split_gate(_read_object(P3_REPORT))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(list(argv) if argv is not None else None)
    payload = run()
    print(
        json.dumps(
            {
                "performance": payload["p4_performance"]["status"],
                "evolution": payload["p4_evolution"]["status"],
                "ad": payload["p4_ad"]["status"],
                "forecast_launch_authorized": payload["p4_performance"][
                    "forecast_launch_authorized"
                ],
                "natural_final_release": payload["natural_final_release"],
                "output": OUT_JSON.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["p4_performance"]["status"] == "RELEASED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
