import json
from pathlib import Path

import pytest

from SelfEvolvingHarnessTS.evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as workflow_runner,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_autonomous_natural_workflow_generation import (
    run_autonomous_acquisition_cycle,
)
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import (
    build_public_operator_inventory,
    compile_workflow_proposal,
    resolve_generated_acquisition_lifecycle,
    run_two_round_generation,
)
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_NAMES


PUBLIC_CONTEXT = {
    "task": "forecasting",
    "series": {"length": 96, "missing_fraction": 0.0, "ema_alpha": 0.2},
}


def _proposal(op, observation, *, bindings=None):
    return {
        "decision": "PROPOSE",
        "steps": [{"op": op, "params": {}, "bindings": bindings or {}}],
        "requested_observations": [observation],
        "fallback": "IDENTITY",
    }


def test_feedback_conditioned_revision_creates_candidate_skill_from_trace():
    planner_inputs = []

    def initial(payload):
        planner_inputs.append(payload)
        return _proposal("impute_linear", "missingness_summary")

    def revise(payload):
        planner_inputs.append(payload)
        assert payload["initial_trace"]["support_response"]["accepted"] is False
        assert payload["exploration_required"] is True
        assert payload["exploration_policy"]["maximum_generation_budget"] == 2
        return _proposal(
            "smooth_ema",
            "local_variation_summary",
            bindings={"alpha": "series.ema_alpha"},
        )

    def support(compiled):
        op = compiled.candidate.program.execution_steps()[0][0]
        return {
            "accepted": op == "smooth_ema",
            "support_gain": 0.2 if op == "smooth_ema" else -0.1,
        }

    result = run_two_round_generation(
        PUBLIC_CONTEXT,
        "forecast",
        initial,
        revise,
        support,
        capability_memory=(),
    )

    assert planner_inputs[0]["capability_memory"] == []
    assert "workflow_catalog" not in planner_inputs[0]
    assert {row["name"] for row in planner_inputs[0]["operator_inventory"]} == set(
        OPERATOR_NAMES
    )
    assert result["status"] == "CANDIDATE"
    assert result["final_candidate"].candidate_id == "generated-workflow-2"
    assert result["final_candidate"].program.execution_steps() == [
        ("smooth_ema", {"alpha": 0.2})
    ]
    assert result["skill_draft"]["program"]["steps"] == result[
        "action_response_trace"
    ][1]["action"]["program_steps"]
    assert result["skill_draft"]["requested_observations"] == result[
        "action_response_trace"
    ][1]["action"]["requested_observations"]
    assert result["skill_draft"]["history"] == result["action_response_trace"]
    assert result["skill_draft"]["program_template"]["steps"][0]["bindings"] == {
        "alpha": "series.ema_alpha"
    }
    assert result["skill_draft"]["task_context"] == PUBLIC_CONTEXT
    assert result["skill_draft"]["applicability_seed"]["source_public_context"] == (
        result["action_response_trace"][1]["public_context"]
    )


def test_unavailable_operator_is_visible_but_cannot_compile():
    inventory = build_public_operator_inventory("forecast", PUBLIC_CONTEXT)
    shape_operator = next(row for row in inventory if row["name"] == "sliding_window")

    assert shape_operator["availability"] == "UNAVAILABLE"
    assert "SHAPE_CHANGING_RUNTIME_UNSUPPORTED" in shape_operator["reason"]
    assert "effect" in shape_operator
    assert "runtime_parameters" in shape_operator
    with pytest.raises(ValueError, match="is unavailable"):
        compile_workflow_proposal(
            _proposal("sliding_window", "window_geometry"),
            inventory,
            PUBLIC_CONTEXT,
            generation=1,
        )


def test_public_context_binding_prefix_compiles_to_canonical_relative_path():
    inventory = build_public_operator_inventory("forecast", PUBLIC_CONTEXT)
    compiled = compile_workflow_proposal(
        _proposal(
            "smooth_ema",
            "local_variation_summary",
            bindings={"alpha": "public_context.series.ema_alpha"},
        ),
        inventory,
        PUBLIC_CONTEXT,
        generation=1,
    )

    assert compiled.candidate.program.execution_steps() == [
        ("smooth_ema", {"alpha": 0.2})
    ]
    assert compiled.template_steps[0]["bindings"] == {
        "alpha": "series.ema_alpha"
    }


def test_initial_unavailable_candidate_becomes_revision_feedback():
    def revise(payload):
        feedback = payload["initial_trace"]["support_response"]
        assert feedback["feedback_type"] == "COMPILATION_ERROR"
        assert feedback["error_code"] == "OPERATOR_UNAVAILABLE"
        return _proposal("impute_linear", "missingness_summary")

    result = run_two_round_generation(
        PUBLIC_CONTEXT,
        "forecast",
        lambda _payload: _proposal("sliding_window", "window_geometry"),
        revise,
        lambda _compiled: {"accepted": True, "support_gain": 0.1},
    )

    assert result["status"] == "CANDIDATE"
    assert result["final_candidate"].program.execution_steps() == [
        ("impute_linear", {})
    ]

    invalid = _proposal("impute_linear", "missingness_summary")
    invalid["rationale"] = "extra model field"

    def revise_invalid(payload):
        assert payload["initial_trace"]["support_response"]["error_code"] == (
            "PROPOSAL_INVALID"
        )
        return _proposal("impute_linear", "missingness_summary")

    recovered = run_two_round_generation(
        PUBLIC_CONTEXT,
        "forecast",
        lambda _payload: invalid,
        revise_invalid,
        lambda _compiled: {"accepted": True, "support_gain": 0.1},
    )
    assert recovered["status"] == "CANDIDATE"


def test_no_positive_support_does_not_create_a_skill():
    result = run_two_round_generation(
        PUBLIC_CONTEXT,
        "forecast",
        lambda _payload: _proposal("impute_linear", "missingness_summary"),
        lambda _payload: _proposal("impute_linear", "missingness_summary"),
        lambda _compiled: {"accepted": False, "support_gain": -0.1},
    )

    assert result["status"] == "REJECTED"
    assert result["final_candidate"] is None
    assert result["skill_draft"] is None
    assert result["action_response_trace"][1]["support_response"]["error_code"] == (
        "REVISION_PROGRAM_UNCHANGED"
    )

    initial_abstain = run_two_round_generation(
        PUBLIC_CONTEXT,
        "forecast",
        lambda _payload: {"decision": "ABSTAIN", "reason": "legal risk"},
        lambda _payload: pytest.fail("revision must not run after initial abstention"),
        lambda _compiled: pytest.fail("Support must not run after abstention"),
    )
    assert initial_abstain["action_response_trace"][0]["support_response"][
        "feedback_type"
    ] == "PROPOSER_ABSTAINED"

    revision_abstain = run_two_round_generation(
        PUBLIC_CONTEXT,
        "forecast",
        lambda _payload: _proposal("impute_linear", "missingness_summary"),
        lambda _payload: {"decision": "ABSTAIN", "reason": "legal risk"},
        lambda _compiled: {"accepted": False, "support_gain": -0.1},
    )
    assert revision_abstain["status"] == "ABSTAIN"
    assert revision_abstain["action_response_trace"][1]["support_response"][
        "feedback_type"
    ] == "PROPOSER_ABSTAINED"

    preserved = run_two_round_generation(
        PUBLIC_CONTEXT,
        "forecast",
        lambda _payload: _proposal("impute_linear", "missingness_summary"),
        lambda _payload: pytest.fail("accepted initial candidate must skip revision"),
        lambda _compiled: {"accepted": True, "support_gain": 0.1},
    )
    assert preserved["status"] == "CANDIDATE"
    assert preserved["final_candidate"].candidate_id == "generated-workflow-1"
    assert len(preserved["action_response_trace"]) == 1


def test_cached_noaa_confirmation_rejects_without_memory_write():
    root = Path(__file__).resolve().parents[2]

    def load(name):
        return json.loads(
            (root / "artifacts/functional/e2" / name).read_text(encoding="utf-8")
        )

    sources = [
        load("autonomous_natural_workflow_generation_nn5_training_only_v2_report.json"),
        load("autonomous_natural_workflow_generation_gefcom_training_only_v1_report.json"),
    ]
    cached = load(
        "autonomous_natural_workflow_scope_induction_v2_noaa_confirmation_report.json"
    )
    writes = []
    result = resolve_generated_acquisition_lifecycle(
        sources,
        cached,
        cached["confirmation"],
        memory_writer=writes.append,
    )

    assert result["status"] == "REJECTED_AFTER_CONFIRMATION"
    assert result["memory_write_authorized"] is False
    assert result["memory_write_count"] == 0
    assert result["rejected_capability_version"] == {
        "program_operator": "period_median_complete",
        "typed_scope": [
            {
                "field": "recent.maximum_missing_run_length",
                "op": "<=",
                "value": 5.0,
            }
        ],
    }
    assert result["contextual_episode"]["confirmation_episode"]["relation"] == (
        "CONFLICT"
    )
    assert result["operator_blacklisted"] is False
    assert result["program_family_closed"] is False
    assert writes == []


def test_memory_writer_runs_once_only_after_complete_positive_confirmation():
    def generated_source(candidate_id, stage="INITIAL"):
        return {
            "capability_memory_entry_count": 0,
            "llm": {
                "api_integrated": True,
                "generation_api_call_count": 1,
                "generation_calls": [{"stage": stage}],
            },
            "generation_proposals": [
                {
                    "candidate_id": candidate_id,
                    "stage": stage,
                    "workflow_steps": [
                        {"op": "generated_op", "params": {}, "bindings": {}}
                    ],
                    "compiled_program_steps": [
                        {"op": "generated_op", "params": {}}
                    ],
                    "support_response": {
                        "accepted": True,
                        "support_gain": 0.01,
                    },
                }
            ],
            "candidate_skill_draft": {
                "program": {
                    "source_candidate_id": candidate_id,
                    "steps": [{"op": "generated_op", "params": {}}],
                }
            },
            "selection": {"selection_gain": 0.02},
        }

    slow_path = {
        "scope_proposal": {
            "decision": "RESTRICT_SCOPE",
            "program_op": "generated_op",
            "predicate": {"all": [{"field": "recent.coverage", "op": ">=", "value": 0.9}]},
        },
        "compilation": "VALID",
        "compiled_conditions": [
            {"field": "recent.coverage", "op": ">=", "value": 0.9}
        ],
        "dead_patch": False,
        "scope_dossier_sent_to_proposer": [
            {
                "environment": environment,
                "episodes": [
                    {
                        "support_exact_singleton_response": {
                            "credit_level": "PROPOSAL_ONLY_LOCAL_ACTION_EPISODE"
                        }
                    }
                ],
            }
            for environment in ("A", "B")
        ],
        "evidence_semantics": {
            "local_singleton": "proposal_credit_only",
            "full_scoped_retrain": "policy_evidence",
        },
        "policy_replays": [
            {
                "environment": environment,
                "eligible_count": 1,
                "training_series_count": 2,
                "scoped_program": {
                    "support": {"gain_vs_identity": 0.01},
                    "selection": {
                        "gain_vs_identity": 0.02,
                        "behavior_point_count": 1,
                    },
                },
            }
            for environment in ("A", "B")
        ],
    }
    sources = [generated_source("candidate-a"), generated_source("candidate-b")]
    writes = []
    rejected = resolve_generated_acquisition_lifecycle(
        sources,
        slow_path,
        {
            "passed": False,
            "selection": {"gain_vs_identity": -0.01, "behavior_point_count": 1},
        },
        memory_writer=writes.append,
    )
    promoted = resolve_generated_acquisition_lifecycle(
        sources,
        slow_path,
        {
            "passed": True,
            "selection": {"gain_vs_identity": 0.03, "behavior_point_count": 1},
        },
        memory_writer=writes.append,
    )

    assert rejected["memory_write_count"] == 0
    assert promoted["status"] == "PROMOTED"
    assert promoted["memory_write_authorized"] is True
    assert promoted["memory_write_count"] == 1
    assert len(writes) == 1
    assert writes[0] == promoted["generated_skill_card"]


def test_autonomous_cycle_orders_stages_without_caller_method_inputs():
    events = []

    def generation_runner(_root, **kwargs):
        assert set(kwargs) == {
            "observe_history",
            "dataset_key",
            "model",
            "base_url",
            "write_report",
        }
        assert kwargs["observe_history"] is True
        assert kwargs["write_report"] is False
        dataset_key = kwargs["dataset_key"]
        events.append(f"generate:{dataset_key}")
        return {
            "capability_memory_entry_count": 0,
            "cycle_status": "CANDIDATE",
            "final_status": "SOURCE_CANDIDATE",
            "llm": {
                "api_integrated": True,
                "api_call_count": 3,
                "generation_api_call_count": 1,
                "generation_calls": [{"stage": "INITIAL"}],
            },
            "generation_proposals": [
                {
                    "candidate_id": f"candidate-{dataset_key}",
                    "stage": "INITIAL",
                    "workflow_steps": [
                        {"op": "generated_op", "params": {}, "bindings": {}}
                    ],
                    "compiled_program_steps": [
                        {"op": "generated_op", "params": {}}
                    ],
                    "support_response": {
                        "accepted": True,
                        "support_gain": 0.01,
                    },
                }
            ],
            "candidate_skill_draft": {
                "program": {
                    "source_candidate_id": f"candidate-{dataset_key}",
                    "steps": [{"op": "generated_op", "params": {}}],
                }
            },
            "selection": {"selection_gain": 0.02},
        }

    def scope_runner(_root, **kwargs):
        assert set(kwargs) == {
            "proposer",
            "source_generation_results",
            "portable_workflow_proposal",
            "portable_proposal_path",
            "confirmation_dataset_key",
            "write_report",
        }
        assert len(kwargs["source_generation_results"]) == 2
        assert kwargs["portable_workflow_proposal"] == {
            "decision": "PROPOSE",
            "steps": [{"op": "generated_op", "params": {}, "bindings": {}}],
            "fallback": "IDENTITY",
        }
        assert kwargs["portable_proposal_path"] is None
        assert kwargs["confirmation_dataset_key"] == "noaa"
        assert kwargs["write_report"] is False
        events.append("scope-and-confirm")
        return {
            "common_program_discovered_from_generation_traces": "generated_op",
            "scope_proposal": {
                "decision": "RESTRICT_SCOPE",
                "program_op": "generated_op",
                "predicate": {
                    "all": [
                        {
                            "field": "recent.coverage",
                            "op": ">=",
                            "value": 0.9,
                        }
                    ]
                },
            },
            "compilation": "VALID",
            "compiled_conditions": [
                {"field": "recent.coverage", "op": ">=", "value": 0.9}
            ],
            "dead_patch": False,
            "risk_patch_replay_passed": True,
            "scope_dossier_sent_to_proposer": [
                {
                    "environment": environment,
                    "episodes": [
                        {
                            "support_exact_singleton_response": {
                                "credit_level": (
                                    "PROPOSAL_ONLY_LOCAL_ACTION_EPISODE"
                                )
                            }
                        }
                    ],
                }
                for environment in ("A", "B")
            ],
            "evidence_semantics": {
                "local_singleton": "proposal_credit_only",
                "full_scoped_retrain": "policy_evidence",
            },
            "policy_replays": [
                {
                    "environment": environment,
                    "eligible_count": 1,
                    "training_series_count": 2,
                    "global_program": {
                        "support": {
                            "gain_vs_identity": -0.01,
                            "behavior_point_count": 2,
                        },
                        "selection": {
                            "gain_vs_identity": -0.02,
                            "behavior_point_count": 2,
                        },
                    },
                    "scoped_program": {
                        "support": {
                            "gain_vs_identity": 0.01,
                            "behavior_point_count": 1,
                        },
                        "selection": {
                            "gain_vs_identity": 0.02,
                            "behavior_point_count": 1,
                        },
                    },
                }
                for environment in ("A", "B")
            ],
            "confirmation": {
                "environment": "C",
                "passed": False,
                "selection": {
                    "gain_vs_identity": -0.03,
                    "behavior_point_count": 1,
                },
            },
            "confirmation_status": "OPENED_AFTER_RISK_PATCH_REPLAY",
            "portable_program_source": "CURRENT_GENERATION_TRACES",
            "final_status": "REJECTED_AFTER_CONFIRMATION",
            "llm": {"api_call_count": 1},
        }

    report = run_autonomous_acquisition_cycle(
        Path(__file__).resolve().parents[2],
        generation_runner=generation_runner,
        scope_runner=scope_runner,
        scope_proposer=lambda _payload: {"decision": "ABSTAIN"},
        write_report=False,
    )

    assert events == ["generate:nn5", "generate:gefcom", "scope-and-confirm"]
    assert report["final_status"] == "REJECTED_AFTER_CONFIRMATION"
    assert report["staged_memory_count"] == 0
    assert report["persistent_memory_written"] is False
    assert report["intermediate_reports_written"] is False
    generation = report["stages"]["generation"]
    assert generation[0]["accepted_program_steps"] == [
        {"op": "generated_op", "params": {}}
    ]
    assert generation[0]["support_gain"] == 0.01
    assert generation[0]["selection_gain"] == 0.02
    slow = report["stages"]["scope_and_full_policy_replay"]
    assert slow["common_program"] == "generated_op"
    assert slow["compiled_scope"] == [
        {"field": "recent.coverage", "op": ">=", "value": 0.9}
    ]
    assert slow["policy_replays"][0] == {
        "environment": "A",
        "eligible_count": 1,
        "training_series_count": 2,
        "global": {
            "support": {"gain_vs_identity": -0.01, "behavior_point_count": 2},
            "selection": {"gain_vs_identity": -0.02, "behavior_point_count": 2},
        },
        "scoped": {
            "support": {"gain_vs_identity": 0.01, "behavior_point_count": 1},
            "selection": {"gain_vs_identity": 0.02, "behavior_point_count": 1},
        },
    }
    assert slow["portable_program_source"] == "CURRENT_GENERATION_TRACES"
    assert slow["confirmation_status"] == "OPENED_AFTER_RISK_PATCH_REPLAY"


def test_autonomous_binding_patch_recompiles_both_traces_and_fails_closed(monkeypatch):
    generated: list[dict[str, object]] = []
    confirmation_calls: list[dict[str, object]] = []

    def generation_runner(_root, **kwargs):
        period = 7 if kwargs["dataset_key"] == "nn5" else 24
        candidate_id = f"candidate-{period}"
        report = {
            "capability_memory_entry_count": 0,
            "cycle_status": "CANDIDATE",
            "final_status": "SOURCE_CANDIDATE",
            "public_context_sent_to_llm": {
                "task": {"horizon": 48, "context_length": 192},
                "periodicity": {"calendar_period": period},
                "missingness": {"maximum_run_length": 4},
            },
            "llm": {
                "api_integrated": True,
                "api_call_count": 1,
                "generation_api_call_count": 1,
                "generation_calls": [{"stage": "INITIAL"}],
            },
            "generation_proposals": [
                {
                    "candidate_id": candidate_id,
                    "stage": "INITIAL",
                    "workflow_steps": [
                        {
                            "op": "period_median_complete",
                            "params": {"period": period, "cycles": 3, "min_donors": 2},
                            "bindings": {},
                        }
                    ],
                    "compiled_program_steps": [
                        {
                            "op": "period_median_complete",
                            "params": {"period": period, "cycles": 3, "min_donors": 2},
                        }
                    ],
                    "support_response": {"accepted": True, "support_gain": 0.01},
                }
            ],
            "candidate_skill_draft": {
                "program": {
                    "source_candidate_id": candidate_id,
                    "steps": [
                        {
                            "op": "period_median_complete",
                            "params": {"period": period, "cycles": 3, "min_donors": 2},
                        }
                    ],
                }
            },
            "selection": {"selection_gain": 0.02},
        }
        generated.append(report)
        return report

    def scope_runner(_root, **kwargs):
        assert kwargs["portable_workflow_proposal"] is None
        assert kwargs["portable_proposal_path"] is None
        return {
            "common_program_discovered_from_generation_traces": (
                "period_median_complete"
            ),
            "scope_proposal": {
                "decision": "RESTRICT_SCOPE",
                "program_op": "period_median_complete",
                "predicate": {
                    "all": [
                        {
                            "field": "recent.maximum_missing_run_length",
                            "op": "<=",
                            "value": 5.0,
                        }
                    ]
                },
            },
            "compilation": "VALID",
            "compiled_conditions": [
                {
                    "field": "recent.maximum_missing_run_length",
                    "op": "<=",
                    "value": 5.0,
                }
            ],
            "dead_patch": False,
            "risk_patch_replay_passed": True,
            "scope_dossier_sent_to_proposer": [
                {
                    "environment": environment,
                    "episodes": [
                        {
                            "support_exact_singleton_response": {
                                "credit_level": "PROPOSAL_ONLY_LOCAL_ACTION_EPISODE"
                            }
                        }
                    ],
                }
                for environment in ("A", "B")
            ],
            "evidence_semantics": {
                "local_singleton": "proposal_credit_only",
                "full_scoped_retrain": "policy_evidence",
            },
            "policy_replays": [
                {
                    "environment": "A",
                    "eligible_count": 1,
                    "training_series_count": 2,
                    "scoped_program": {
                        "support": {"gain_vs_identity": 0.01},
                        "selection": {
                            "gain_vs_identity": 0.02,
                            "behavior_point_count": 1,
                        },
                    },
                },
                {
                    "environment": "B",
                    "eligible_count": 0,
                    "training_series_count": 2,
                    "scoped_program": {
                        "support": {"gain_vs_identity": 0.0},
                        "selection": {
                            "gain_vs_identity": 0.0,
                            "behavior_point_count": 0,
                        },
                    },
                },
            ],
            "confirmation": None,
            "confirmation_status": "SKIPPED_NO_CURRENT_PORTABLE_TEMPLATE",
            "portable_program_source": None,
            "final_status": "RISK_PATCH_REPLAY_PASSED",
            "llm": {"api_call_count": 1},
        }

    def confirm(_root, **kwargs):
        confirmation_calls.append(kwargs)
        return {
            "environment": "C",
            "context_sent_to_llm": False,
            "passed": False,
            "selection": {"gain_vs_identity": -0.01, "behavior_point_count": 1},
        }, False

    monkeypatch.setattr(workflow_runner, "_confirm_scoped_portable_program", confirm)
    captured_dossiers = []

    def valid_binding(dossier):
        captured_dossiers.append(dossier)
        serialized = json.dumps(dossier).lower()
        for forbidden in (
            "dataset_id",
            "dataset_key",
            "series_uid",
            "query_outcome",
            "selection_outcome",
            "future_values",
            "raw_values",
        ):
            assert forbidden not in serialized
        return {
            "decision": "PATCH_BINDING",
            "program_op": "period_median_complete",
            "parameter": "period",
            "public_context_field": "periodicity.calendar_period",
        }

    valid = run_autonomous_acquisition_cycle(
        Path(__file__).resolve().parents[2],
        generation_runner=generation_runner,
        scope_runner=scope_runner,
        scope_proposer=lambda _payload: {"decision": "ABSTAIN"},
        binding_proposer=valid_binding,
        write_report=False,
    )
    binding = valid["stages"]["binding"]
    assert captured_dossiers[0]["environments"][0]["environment"] == "A"
    assert binding["status"] == "VALIDATED"
    assert binding["validation"]["source_recompile_equivalent"] == [True, True]
    assert valid["first_fault"]["resolved"] is True
    assert confirmation_calls[0]["portable_proposal"]["steps"] == [
        {
            "op": "period_median_complete",
            "params": {"cycles": 3, "min_donors": 2},
            "bindings": {"period": "periodicity.calendar_period"},
        }
    ]
    assert valid["persistent_memory_written"] is False

    invalid = run_autonomous_acquisition_cycle(
        Path(__file__).resolve().parents[2],
        generation_runner=generation_runner,
        scope_runner=scope_runner,
        scope_proposer=lambda _payload: {"decision": "ABSTAIN"},
        binding_proposer=lambda _dossier: {
            "decision": "PATCH_BINDING",
            "program_op": "impute_linear",
            "parameter": "period",
            "public_context_field": "periodicity.calendar_period",
        },
        write_report=False,
    )
    assert invalid["stages"]["binding"]["status"] == "ABSTAINED"
    assert invalid["stages"]["confirmation"]["status"] == (
        "SKIPPED_BINDING_PATCH_NOT_VALIDATED"
    )
    assert len(confirmation_calls) == 1
    assert invalid["persistent_memory_written"] is False
    assert invalid["operator_blacklisted"] is False


def test_scope_risk_feedback_allows_one_revision_then_enters_binding(monkeypatch):
    source_reports = []

    def generation_runner(_root, **kwargs):
        period = 7 if kwargs["dataset_key"] == "nn5" else 24
        candidate_id = f"candidate-{period}"
        report = {
            "capability_memory_entry_count": 0,
            "cycle_status": "CANDIDATE",
            "final_status": "SOURCE_CANDIDATE",
            "public_context_sent_to_llm": {
                "task": {"horizon": 12, "context_length": 48},
                "periodicity": {"calendar_period": period},
            },
            "llm": {"api_call_count": 1},
            "generation_proposals": [
                {
                    "candidate_id": candidate_id,
                    "stage": "INITIAL",
                    "workflow_steps": [
                        {
                            "op": "period_median_complete",
                            "params": {
                                "period": period,
                                "cycles": 3,
                                "min_donors": 2,
                            },
                            "bindings": {},
                        }
                    ],
                    "compiled_program_steps": [
                        {
                            "op": "period_median_complete",
                            "params": {
                                "period": period,
                                "cycles": 3,
                                "min_donors": 2,
                            },
                        }
                    ],
                    "support_response": {"accepted": True, "support_gain": 0.01},
                }
            ],
            "candidate_skill_draft": {
                "program": {
                    "source_candidate_id": candidate_id,
                    "steps": [
                        {
                            "op": "period_median_complete",
                            "params": {
                                "period": period,
                                "cycles": 3,
                                "min_donors": 2,
                            },
                        }
                    ],
                }
            },
            "selection": {"selection_gain": 0.02},
        }
        source_reports.append(report)
        return report

    dummy_compiled = workflow_runner.CompiledWorkflow(
        candidate=None,
        requested_observations=(),
        template_steps=(),
    )

    def scope_environment(_root, *, environment, dataset_key, bound_step):
        assert dataset_key in {"nn5", "gefcom"}
        assert bound_step["op"] == "period_median_complete"
        prefix = environment.lower()
        summaries = {
            f"{prefix}0": {"recent": {"maximum_missing_run_length": 1.0}},
            f"{prefix}1": {"recent": {"maximum_missing_run_length": 3.0}},
        }
        episodes = [
            {
                "environment": environment,
                "within_environment_ordinal": ordinal,
                "public_history_summary": {
                    "early": {"maximum_missing_run_length": value},
                    "recent": {"maximum_missing_run_length": value},
                    "early_to_recent_change": {
                        "maximum_missing_run_length": 0.0
                    },
                },
                "support_exact_singleton_response": {
                    "credit_level": "PROPOSAL_ONLY_LOCAL_ACTION_EPISODE",
                    "cohort_support_gain": 0.01,
                    "per_view_gain": [0.01],
                    "behavior_point_count": 1,
                },
            }
            for ordinal, value in enumerate((1.0, 3.0))
        ]
        return {"environment": environment, "episodes": episodes}, {
            "roster": [
                {"role": "train", "series_uid": f"{prefix}0"},
                {"role": "train", "series_uid": f"{prefix}1"},
            ],
            "values": {},
            "config": {
                "environment": environment,
                "support_origin": 1,
                "selection_origin": 2,
            },
            "compiled": dummy_compiled,
            "summaries_by_uid": summaries,
        }

    def policy_score(_roster, _values, config, _compiled, *, origin, scope=None):
        environment = config["environment"]
        split = "support" if origin == 1 else "selection"
        global_gain = {
            ("A", "support"): 0.05,
            ("A", "selection"): 0.06144,
            ("B", "support"): -0.10,
            ("B", "selection"): -0.18061,
        }[(environment, split)]
        if scope is None:
            return {"gain_vs_identity": global_gain, "behavior_point_count": 2}
        selected_second = any(str(uid).endswith("1") for uid in scope)
        scoped_gain = {
            (False, "A", "support"): 0.06,
            (False, "A", "selection"): 0.05761,
            (False, "B", "support"): 0.0,
            (False, "B", "selection"): 0.0,
            (True, "A", "support"): 0.08,
            (True, "A", "selection"): 0.08,
            (True, "B", "support"): 0.02,
            (True, "B", "selection"): 0.02,
        }[(selected_second, environment, split)]
        return {"gain_vs_identity": scoped_gain, "behavior_point_count": 1}

    monkeypatch.setattr(workflow_runner, "_scope_induction_environment", scope_environment)
    monkeypatch.setattr(workflow_runner, "_policy_score", policy_score)
    monkeypatch.setattr(
        workflow_runner,
        "_confirm_scoped_portable_program",
        lambda *_args, **_kwargs: (
            {
                "environment": "C",
                "context_sent_to_llm": False,
                "passed": False,
                "selection": {
                    "gain_vs_identity": -0.01,
                    "behavior_point_count": 1,
                },
            },
            False,
        ),
    )

    class ScopeProposer:
        model = "gpt-5.5"
        base_url = "https://api.agicto.cn/v1"

        def __init__(self):
            self.call_count = 0
            self.calls = []
            self.payloads = []

        def __call__(self, payload):
            self.payloads.append(payload)
            self.call_count += 1
            self.calls.append(
                {"stage": "INITIAL" if self.call_count == 1 else "REVISION"}
            )
            if self.call_count == 1:
                return {
                    "decision": "RESTRICT_SCOPE",
                    "program_op": "period_median_complete",
                    "predicate": {
                        "all": [
                            {
                                "field": "recent.maximum_missing_run_length",
                                "op": "<=",
                                "value": 2,
                            }
                        ]
                    },
                }
            assert set(payload) == {
                "original_scope_dossier",
                "original_typed_patch",
                "anonymous_full_policy_replay",
                "frozen_risk_gate_semantics",
                "required_output_json_schema",
            }
            assert payload["required_output_json_schema"] == self.payloads[0][
                "required_output_json_schema"
            ]
            assert payload["anonymous_full_policy_replay"][0][
                "global_selection"
            ] == {"gain_vs_identity": 0.06144, "behavior_point_count": 2}
            assert payload["anonymous_full_policy_replay"][0][
                "scoped_selection"
            ] == {"gain_vs_identity": 0.05761, "behavior_point_count": 1}
            serialized = json.dumps(payload).lower()
            for forbidden in (
                '"dataset_id"',
                '"dataset_key"',
                '"series_uid"',
                '"uid"',
                '"raw"',
                '"path"',
                '"future"',
                '"query"',
                "noaa",
            ):
                assert forbidden not in serialized
            return {
                "decision": "RESTRICT_SCOPE",
                "program_op": "period_median_complete",
                "predicate": {
                    "all": [
                        {
                            "field": "recent.maximum_missing_run_length",
                            "op": ">=",
                            "value": 2,
                        }
                    ]
                },
            }

    scope_proposer = ScopeProposer()
    binding_calls = []

    def binding_proposer(payload):
        binding_calls.append(payload)
        return {
            "decision": "PATCH_BINDING",
            "program_op": "period_median_complete",
            "parameter": "period",
            "public_context_field": "periodicity.calendar_period",
        }

    report = run_autonomous_acquisition_cycle(
        Path(__file__).resolve().parents[2],
        generation_runner=generation_runner,
        scope_proposer=scope_proposer,
        binding_proposer=binding_proposer,
        write_report=False,
    )

    scope_stage = report["stages"]["scope_and_full_policy_replay"]
    assert scope_proposer.call_count == 2
    assert scope_stage["revision_invoked"] is True
    assert len(scope_stage["scope_attempts"]) == 2
    assert scope_stage["scope_attempts"][0]["risk_patch_replay_passed"] is False
    assert scope_stage["scope_attempts"][1]["risk_patch_replay_passed"] is True
    assert scope_stage["compiled_scope"] == [
        {
            "field": "recent.maximum_missing_run_length",
            "op": ">=",
            "value": 2.0,
        }
    ]
    assert len(binding_calls) == 1
    assert report["stages"]["binding"]["status"] == "VALIDATED"

    for terminal_proposal, expected_compilation in (
        ({"decision": "ABSTAIN", "reason": "insufficient evidence"}, "VALID"),
        ({"decision": "RESTRICT_SCOPE"}, "INVALID"),
    ):
        calls = []

        def terminal(payload, proposal=terminal_proposal):
            calls.append(payload)
            return proposal

        stopped = workflow_runner.run_induce_scope(
            Path(__file__).resolve().parents[2],
            proposer=terminal,
            source_generation_results=source_reports,
            portable_proposal_path=None,
            write_report=False,
        )
        assert len(calls) == 1
        assert stopped["revision_invoked"] is False
        assert stopped["compilation"] == expected_compilation


def test_autonomous_cycle_stops_cleanly_at_generation_control_faults():
    calls = []

    def scope_must_not_run(*_args, **_kwargs):
        pytest.fail("Scope must not run after a generation control fault")

    def observation_fault(_root, **kwargs):
        calls.append(kwargs["dataset_key"])
        raise RuntimeError(
            "observation stage must execute compare_history_windows exactly once"
        )

    observation_report = run_autonomous_acquisition_cycle(
        Path(__file__).resolve().parents[2],
        generation_runner=observation_fault,
        scope_runner=scope_must_not_run,
        scope_proposer=lambda _payload: pytest.fail("proposer must not run"),
        write_report=False,
    )

    assert calls == ["nn5"]
    assert observation_report["final_status"] == "ABSTAINED"
    assert observation_report["first_fault"] == {
        "stage": "generation",
        "environment": "A",
        "reason_code": "OBSERVATION_CONTRACT_VIOLATION",
        "message": (
            "observation stage must execute compare_history_windows exactly once"
        ),
    }
    assert observation_report["stages"]["scope_and_full_policy_replay"][
        "final_status"
    ] == "NOT_RUN"
    assert observation_report["persistent_memory_written"] is False

    def generation_abstain(_root, **kwargs):
        calls.append(kwargs["dataset_key"])
        return {
            "cycle_status": "ABSTAIN",
            "cycle_reason_code": "INITIAL_PROPOSER_ABSTAINED",
            "final_status": "REJECTED",
            "generation_proposals": [],
            "candidate_skill_draft": None,
            "selection": None,
            "llm": {"api_call_count": 2},
        }

    abstain_report = run_autonomous_acquisition_cycle(
        Path(__file__).resolve().parents[2],
        generation_runner=generation_abstain,
        scope_runner=scope_must_not_run,
        scope_proposer=lambda _payload: pytest.fail("proposer must not run"),
        write_report=False,
    )

    assert calls == ["nn5", "nn5"]
    assert abstain_report["first_fault"]["reason_code"] == (
        "INITIAL_PROPOSER_ABSTAINED"
    )
    assert abstain_report["staged_memory_count"] == 0
    assert abstain_report["persistent_memory_written"] is False

    def unknown_failure(_root, **_kwargs):
        raise RuntimeError("network transport failed")

    with pytest.raises(RuntimeError, match="network transport failed"):
        run_autonomous_acquisition_cycle(
            Path(__file__).resolve().parents[2],
            generation_runner=unknown_failure,
            scope_runner=scope_must_not_run,
            scope_proposer=lambda _payload: pytest.fail("proposer must not run"),
            write_report=False,
        )
