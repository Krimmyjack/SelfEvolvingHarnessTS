from __future__ import annotations

import json

import pytest

from SelfEvolvingHarnessTS.evaluation.minipipe.replay.run_heldout_reuse import (
    load_source_cause,
    resolve_heldout_call_budget,
    select_screened_targets,
)


def test_heldout_screening_takes_first_matches_without_reranking():
    rows = [
        {"seed": 404, "severity": "mild", "applicability_match": False},
        {"seed": 404, "severity": "severe", "applicability_match": True},
        {"seed": 505, "severity": "mild", "applicability_match": True},
        {"seed": 505, "severity": "severe", "applicability_match": False},
        {"seed": 606, "severity": "mild", "applicability_match": True},
    ]

    selected = select_screened_targets(rows, minimum=2)

    assert [(row["seed"], row["severity"]) for row in selected] == [
        (404, "severe"),
        (505, "mild"),
    ]


def test_heldout_screening_does_not_backfill_insufficient_support():
    rows = [{"seed": 404, "severity": "mild", "applicability_match": True}]

    assert len(select_screened_targets(rows, minimum=4)) == 1
    with pytest.raises(ValueError, match="positive"):
        select_screened_targets(rows, minimum=0)


def test_heldout_budget_amendment_changes_only_the_registered_resource_limit():
    preregistration = {
        "preregistration_id": "heldout-1",
        "paired_evaluation": {"hard_uncached_api_call_limit": 180},
    }
    amendment = {
        "preregistration_id": "heldout-1",
        "old_hard_uncached_api_call_limit": 180,
        "new_hard_uncached_api_call_limit": 320,
        "registered_before_any_heldout_outcome_or_h1_result": True,
    }

    assert resolve_heldout_call_budget(
        preregistration,
        requested_call_budget=320,
        budget_amendment=amendment,
    ) == 320
    with pytest.raises(ValueError, match="invalid"):
        resolve_heldout_call_budget(
            preregistration,
            requested_call_budget=320,
            budget_amendment={**amendment, "preregistration_id": "other"},
        )
    with pytest.raises(ValueError, match="exceeds"):
        resolve_heldout_call_budget(
            preregistration,
            requested_call_budget=321,
            budget_amendment=amendment,
        )
    with pytest.raises(ValueError, match="raise the limit"):
        resolve_heldout_call_budget(
            preregistration,
            requested_call_budget=180,
            budget_amendment={**amendment, "new_hard_uncached_api_call_limit": 180},
        )


def test_heldout_replays_the_source_patterns_authorized_cause(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    (public / "failure_patterns.json").write_text(
        json.dumps(
            {
                "patterns": [
                    {
                        "pattern_id": "pattern-scoped",
                        "cause_code": "SCOPED_SELECTION_GAP",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_source_cause(tmp_path, "pattern-scoped") == (
        "SCOPED_SELECTION_GAP"
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_source_cause(tmp_path, "pattern-missing")
