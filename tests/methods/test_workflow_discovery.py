from SelfEvolvingHarnessTS.methods.ttha.workflow_discovery import (
    discover_workflow_supply,
)


CONTEXT = {
    "task": "forecasting",
    "series": {"estimated_period": 24, "missing_fraction": 0.1},
}
WORKFLOWS = [
    {
        "workflow_id": "W_local",
        "description": "bounded local preparation",
        "public_parameter_bindings": {"period": "series.estimated_period"},
    },
    {
        "workflow_id": "W_cohort",
        "description": "bounded cohort preparation",
        "public_parameter_bindings": {
            "missing_fraction": "series.missing_fraction"
        },
    },
]
OBSERVATIONS = ["series_overview", "missing_run_topology"]


def _valid_proposal():
    return {
        "decision": "PROPOSE",
        "selected_workflows": [
            {"workflow_id": "W_local", "bindings": {"period": 24}},
            {
                "workflow_id": "W_cohort",
                "bindings": {"missing_fraction": 0.1},
            },
        ],
        "probe_order": ["W_local", "W_cohort"],
        "requested_observations": ["series_overview"],
        "fallback": "IDENTITY",
    }


def test_legal_catalog_proposal_compiles_to_skill_acquisition_supply():
    result = discover_workflow_supply(
        CONTEXT, WORKFLOWS, OBSERVATIONS, lambda _payload: _valid_proposal()
    )

    assert result["decision"] == "PROPOSE"
    assert result["workflow_supply"] == ["W_local", "W_cohort"]
    assert result["compiled_workflows"][0]["bindings"] == {"period": 24}
    assert result["candidate_status"] == "DISCOVERED_NOT_EVALUATED"


def test_invented_workflow_fails_closed():
    proposal = _valid_proposal()
    proposal["selected_workflows"][0]["workflow_id"] = "W_invented"

    result = discover_workflow_supply(
        CONTEXT, WORKFLOWS, OBSERVATIONS, lambda _payload: proposal
    )

    assert result["decision"] == "ABSTAIN"
    assert result["fallback"] == "IDENTITY"


def test_binding_mismatch_fails_closed():
    proposal = _valid_proposal()
    proposal["selected_workflows"][0]["bindings"]["period"] = 48

    result = discover_workflow_supply(
        CONTEXT, WORKFLOWS, OBSERVATIONS, lambda _payload: proposal
    )

    assert result["decision"] == "ABSTAIN"
    assert result["workflow_supply"] == []


def test_private_or_outcome_field_fails_closed():
    proposal = _valid_proposal()
    proposal["utility"] = 0.7

    result = discover_workflow_supply(
        CONTEXT, WORKFLOWS, OBSERVATIONS, lambda _payload: proposal
    )

    assert result["decision"] == "ABSTAIN"
    assert result["candidate_status"] == "NOT_CREATED"


def test_planner_abstain_preserves_identity():
    result = discover_workflow_supply(
        CONTEXT,
        WORKFLOWS,
        OBSERVATIONS,
        lambda _payload: {"decision": "ABSTAIN", "fallback": "IDENTITY"},
    )

    assert result["decision"] == "ABSTAIN"
    assert result["fallback"] == "IDENTITY"
    assert result["reason_code"] == "PLANNER_ABSTAIN"
