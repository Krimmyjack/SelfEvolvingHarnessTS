from pathlib import Path

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    parse_json_document,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_f1_forecast_pilot import (
    audit_stage_cache,
    run_f1_forecast_pilot,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.fixtures.contract_policy import (
    DeterministicContractValuator,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse, ReplayAgentBackend


def _write_record(
    root: Path,
    *,
    name: str,
    stage: str,
    task_sha: str,
    run_sha: str,
    receipt_visible: bool,
) -> None:
    public_input = {"task_context_sha": task_sha}
    if receipt_visible:
        public_input["candidates"] = [
            {
                "candidate_id": "identity",
                "verification_receipt": {"status": "valid"},
            }
        ]
    record = {
        "schema_version": "agent-cache-record/1",
        "key": {"stage": stage},
        "task_context_sha": task_sha,
        "run_context_sha": run_sha,
        "messages": [
            {"role": "system", "content": "public-only"},
            {
                "role": "user",
                "content": canonical_json_bytes(
                    {"public_input": public_input}
                ).decode("utf-8"),
            },
        ],
    }
    (root / f"{name}.json").write_bytes(canonical_json_bytes(record) + b"\n")


def test_stage_cache_audit_enforces_receipt_information_timing(tmp_path: Path) -> None:
    task_sha = "1" * 64
    run_sha = "2" * 64
    for index, stage in enumerate(("inspect", "propose", "select")):
        _write_record(
            tmp_path,
            name=str(index),
            stage=stage,
            task_sha=task_sha,
            run_sha=run_sha,
            receipt_visible=stage == "select",
        )

    audit = audit_stage_cache(
        tmp_path,
        run_sha_by_task_sha={task_sha: run_sha},
    )

    assert audit["status"] == "PASS"
    assert audit["stage_counts"] == {"inspect": 1, "propose": 1, "select": 1}


def test_stage_cache_audit_rejects_premature_receipt_and_private_key(
    tmp_path: Path,
) -> None:
    task_sha = "1" * 64
    run_sha = "2" * 64
    _write_record(
        tmp_path,
        name="bad",
        stage="inspect",
        task_sha=task_sha,
        run_sha=run_sha,
        receipt_visible=True,
    )
    record_path = tmp_path / "bad.json"
    record = parse_json_document(record_path.read_bytes())
    user_payload = parse_json_document(record["messages"][1]["content"].encode("utf-8"))
    user_payload["public_input"]["injection_type"] = "private-label"
    record["messages"][1]["content"] = canonical_json_bytes(user_payload).decode("utf-8")
    record_path.write_bytes(canonical_json_bytes(record) + b"\n")

    audit = audit_stage_cache(
        tmp_path,
        run_sha_by_task_sha={task_sha: run_sha},
    )

    assert audit["status"] == "FAIL"
    assert any("premature_receipt_visibility" in item for item in audit["violations"])
    assert any("private_keys:injection_type" in item for item in audit["violations"])


def _stage(stage: str, payload: dict[str, object]) -> AgentResponse:
    return AgentResponse.valid(
        {
            "schema_version": "agent-envelope/1",
            "kind": "stage_result",
            "stage": stage,
            "payload": payload,
        },
        raw_response={"id": f"offline-{stage}"},
    )


def test_f1_pilot_runs_on_frozen_h2_without_promoting_harness(tmp_path: Path) -> None:
    responses: list[AgentResponse] = []
    for _ in range(8):
        responses.extend(
            [
                _stage(
                    "inspect",
                    {
                        "inspected_region_fractions": [[0.0, 1.0]],
                        "requested_public_tools": [],
                        "uncertainty": "high",
                    },
                ),
                _stage("propose", {"candidates": []}),
                _stage(
                    "select",
                    {
                        "chosen_candidate_id": "identity",
                        "verification_actions": ["identity_retained"],
                    },
                ),
            ]
        )

    report = run_f1_forecast_pilot(
        run_root=tmp_path / "f1-pilot",
        backend=ReplayAgentBackend(responses),
        valuator=DeterministicContractValuator(),
        code_commit="a" * 40,
    )

    assert report["status"] == "PASS"
    assert report["checks"]["h2_content_unchanged"] is True
    assert report["checks"]["select_receipt_coverage"] is True
    assert report["checks"]["stage_cache_information_timing"] is True
    assert (tmp_path / "f1-pilot/public/f1_forecast_pilot_report.json").is_file()
