from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest
from SelfEvolvingHarnessTS.contracts.run_context import RunDependencyBinding
from SelfEvolvingHarnessTS.contracts.task import (
    TaskQualityContract,
    classification_task_spec_v1,
    forecast_neutral_task_quality_contract_v1,
    forecast_task_context_v1,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.contracts.canonical import parse_json_document
from SelfEvolvingHarnessTS.methods.ttha.agent_core import validate_local_schema


ROOT = Path(__file__).resolve().parents[2]


def _run_binding(task_context_sha: str, *, instrument_epoch: str) -> RunDependencyBinding:
    return RunDependencyBinding(
        task_context_sha=task_context_sha,
        evaluator_adapter_id="forecast-chronos-v1",
        instrument_epoch=instrument_epoch,
        corpus_epoch="m0-corpus-v2",
        capability_bundle_sha="1" * 64,
        runtime_sha="2" * 64,
        harness_sha="3" * 64,
        code_commit="4" * 40,
        provider_id="agicto-chat-completions",
        model_id="gpt-5.5",
    )


def test_task_context_identity_is_stable_across_instrument_epochs() -> None:
    context = forecast_task_context_v1(
        task_spec=forecast_task_spec_v1(horizon=12)
    )
    first = _run_binding(context.sha(), instrument_epoch="probe-instrument/3")
    second = _run_binding(context.sha(), instrument_epoch="probe-instrument/4")

    assert len(context.sha()) == 64
    assert first.task_context_sha == second.task_context_sha == context.sha()
    assert first.sha() != second.sha()


def test_preparation_request_rejects_task_context_mismatch_before_agent_call() -> None:
    context = forecast_task_context_v1(
        task_spec=forecast_task_spec_v1(horizon=12)
    )
    with pytest.raises(ValueError, match="task_context.task_spec"):
        PreparationRequest(
            "case-mismatch",
            np.arange(8.0),
            forecast_task_spec_v1(horizon=6),
            task_context=context,
        )

    with pytest.raises(ValueError, match="task_type mismatch"):
        replace(context, task_spec=classification_task_spec_v1())


def test_quality_contract_uses_closed_vocabulary() -> None:
    context = forecast_task_context_v1()
    with pytest.raises(ValueError, match="unknown vocabulary"):
        replace(
            context.quality_contract,
            preserve=("repair_level_shift",),
        )

    with pytest.raises(ValueError, match="unsupported TaskQualityContract revision"):
        replace(
            context.quality_contract,
            schema_version="task-quality-contract/999",
        )


def test_run_dependency_binding_must_name_same_task_context() -> None:
    context = forecast_task_context_v1()
    wrong = _run_binding("f" * 64, instrument_epoch="probe-instrument/3")
    with pytest.raises(ValueError, match="TaskContext SHA mismatch"):
        PreparationRequest(
            "case-run-mismatch",
            np.arange(8.0),
            context.task_spec,
            task_context=context,
            run_dependency_binding=wrong,
        )


def test_contract_has_no_free_text_or_operator_mapping_fields() -> None:
    contract: TaskQualityContract = forecast_task_context_v1().quality_contract
    assert set(contract.to_dict()) == {
        "schema_version",
        "contract_id",
        "task_type",
        "objective",
        "preserve",
        "harms",
        "evidence_expectations",
        "verification_dimensions",
        "abstention_conditions",
    }


def test_default_task_context_conforms_to_declared_closed_schema() -> None:
    schema = parse_json_document(
        (ROOT / "contracts" / "schemas" / "task_context_v1.json").read_bytes()
    )
    validate_local_schema(forecast_task_context_v1().to_dict(), schema)


def test_neutral_forecast_contract_is_distinct_but_keeps_task_identity() -> None:
    task = forecast_task_spec_v1(horizon=12)
    correct = forecast_task_context_v1(task_spec=task)
    neutral = forecast_task_context_v1(
        task_spec=task,
        quality_contract=forecast_neutral_task_quality_contract_v1(),
    )

    assert correct.task_spec == neutral.task_spec
    assert correct.sha() != neutral.sha()
    assert neutral.quality_contract.preserve == ("temporal_order", "series_length")
