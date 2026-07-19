from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_document,
)
from SelfEvolvingHarnessTS.contracts.public_boundary import FORBIDDEN_PUBLIC_KEYS
from SelfEvolvingHarnessTS.contracts.run_context import RunDependencyBinding
from SelfEvolvingHarnessTS.contracts.task import (
    TaskContext,
    forecast_neutral_task_quality_contract_v1,
    forecast_task_context_v1,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import build_core_corpus
from SelfEvolvingHarnessTS.evaluation.minipipe.cycle import _CycleCaseRunner
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.panel import PROBE_INSTRUMENT_EPOCH
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import FrozenChronosValuator
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_compatible_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import MaterializedSnapshot
from SelfEvolvingHarnessTS.runtime.agent_backend import (
    DEFAULT_AGENT_BASE_URL,
    DEFAULT_AGENT_MODEL,
    AgictoChatCompletionsBackend,
    BudgetedAgentBackend,
)
from SelfEvolvingHarnessTS.runtime.llm_cache import CachedAgentBackend, EffectiveRequestCache


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RELEASE = _ROOT / "artifacts" / "releases" / "m0-h2" / "harness"
_DEFAULT_RULES = _ROOT / "evaluation" / "minipipe" / "config" / "m0_rules.json"
_DEFAULT_PREREGISTRATION = (
    _ROOT / "artifacts" / "manifests" / "f1_forecast_pilot_preregistration_20260719.json"
)


def _read_object(path: Path) -> dict[str, object]:
    value = parse_json_document(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _forbidden_keys(value: object) -> tuple[str, ...]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key).lower()
            if name in FORBIDDEN_PUBLIC_KEYS:
                found.add(name)
            found.update(_forbidden_keys(nested))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for nested in value:
            found.update(_forbidden_keys(nested))
    return tuple(sorted(found))


def audit_stage_cache(
    cache_root: Path,
    *,
    run_sha_by_task_sha: Mapping[str, str],
) -> dict[str, object]:
    """Check TaskContext binding and candidate-receipt information timing."""

    rows: list[dict[str, object]] = []
    violations: list[str] = []
    for path in sorted(Path(cache_root).glob("*.json"), key=lambda item: item.name):
        record = _read_object(path)
        key = record.get("key")
        if not isinstance(key, Mapping):
            violations.append(f"{path.name}:missing_key")
            continue
        stage = str(key.get("stage", ""))
        task_sha = str(record.get("task_context_sha", ""))
        run_sha = str(record.get("run_context_sha", ""))
        messages = record.get("messages", ())
        if not isinstance(messages, Sequence) or isinstance(
            messages, (str, bytes, bytearray)
        ):
            violations.append(f"{path.name}:invalid_messages")
            continue
        message_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, Mapping)
        )
        receipt_visible = '"verification_receipt"' in message_text
        if task_sha not in run_sha_by_task_sha:
            violations.append(f"{path.name}:unknown_task_context")
        elif run_sha != run_sha_by_task_sha[task_sha]:
            violations.append(f"{path.name}:run_context_mismatch")
        if stage in {"inspect", "propose"} and receipt_visible:
            violations.append(f"{path.name}:premature_receipt_visibility")
        if stage == "select" and not receipt_visible:
            violations.append(f"{path.name}:select_missing_receipt")
        public_forbidden: set[str] = set()
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            try:
                payload = parse_json_document(str(message.get("content", "")).encode("utf-8"))
            except (TypeError, ValueError, UnicodeError):
                violations.append(f"{path.name}:invalid_user_payload")
                continue
            public_forbidden.update(_forbidden_keys(payload))
        if public_forbidden:
            violations.append(
                f"{path.name}:private_keys:{','.join(sorted(public_forbidden))}"
            )
        rows.append(
            {
                "cache_record": path.name,
                "stage": stage,
                "task_context_sha": task_sha,
                "run_context_sha": run_sha,
                "receipt_visible": receipt_visible,
            }
        )
    return {
        "schema_version": "f1-stage-cache-audit/1",
        "record_count": len(rows),
        "stage_counts": {
            stage: sum(row["stage"] == stage for row in rows)
            for stage in ("inspect", "propose", "select")
        },
        "violations": violations,
        "rows": rows,
        "status": "PASS" if rows and not violations else "FAIL",
    }


def _decision_fingerprint(evaluation: object) -> str:
    trace = evaluation.trace
    payload = {
        "inspected_regions": [list(region) for region in trace.inspected_regions],
        "retrieved_skill_ids": list(trace.retrieved_skill_ids),
        "candidate_program_shas": list(trace.candidate_program_shas),
        "chosen_candidate_id": trace.chosen_candidate_id,
        "compilation_status": trace.compilation_status,
        "execution_status": trace.execution_status,
        "verification_actions": list(trace.verification_actions),
        "modified_indices": list(trace.modified_indices),
    }
    return canonical_sha256(payload)


def _binding(
    *,
    context: TaskContext,
    snapshot: object,
    code_commit: str,
    corpus_epoch: str,
    model: str,
) -> RunDependencyBinding:
    dependencies = snapshot.dependency_shas
    return RunDependencyBinding(
        task_context_sha=context.sha(),
        evaluator_adapter_id="forecast-chronos-m0-v1",
        instrument_epoch=PROBE_INSTRUMENT_EPOCH,
        corpus_epoch=corpus_epoch,
        capability_bundle_sha=str(dependencies["operator_bundle"]),
        runtime_sha=snapshot.runtime_bundle_sha,
        harness_sha=snapshot.harness_content_sha,
        code_commit=code_commit,
        provider_id="agicto-chat-completions",
        model_id=model,
    )


def run_f1_forecast_pilot(
    *,
    run_root: Path,
    backend: object,
    valuator: object,
    code_commit: str,
    preregistration_path: Path = _DEFAULT_PREREGISTRATION,
    release_root: Path = _DEFAULT_RELEASE,
    rules_path: Path = _DEFAULT_RULES,
    model: str = DEFAULT_AGENT_MODEL,
    base_url: str = DEFAULT_AGENT_BASE_URL,
    resume: bool = False,
) -> dict[str, object]:
    prereg = _read_object(preregistration_path)
    if prereg.get("schema_version") != "f1-forecast-pilot-preregistration/1":
        raise ValueError("unsupported F1 pilot preregistration")
    if model != prereg["model"] or base_url != prereg["base_url"]:
        raise ValueError("model or relay differs from F1 preregistration")
    if code_commit == str(prereg.get("m0_implementation_commit")):
        raise ValueError("F1 pilot must run on a post-M0 implementation commit")

    run_root = Path(run_root).resolve()
    non_empty = run_root.exists() and any(run_root.iterdir())
    if non_empty and not resume:
        raise FileExistsError("F1 run_root must be absent or empty")
    run_root.mkdir(parents=True, exist_ok=True)

    expected_content_sha = str(prereg["h2_harness_content_sha"])
    snapshot = compile_compatible_snapshot(
        release_root,
        expected_harness_content_sha=expected_content_sha,
    )
    materialized = MaterializedSnapshot(Path(release_root).resolve(), snapshot, None)

    rules = load_m0_rules(rules_path)
    corpus = build_core_corpus(rules)
    cases_by_id = {case.case_id: case for case in corpus.all_cases}
    correct_ids = tuple(str(value) for value in prereg["correct_contract_case_ids"])
    neutral_ids = tuple(str(value) for value in prereg["neutral_contract_case_ids"])
    missing = sorted((set(correct_ids) | set(neutral_ids)) - set(cases_by_id))
    if missing:
        raise ValueError(f"preregistered cases are absent from corpus: {missing}")

    task_spec = forecast_task_spec_v1(
        horizon=int(rules["corpus"]["future_length"]),
        downstream_model_class="frozen_tsfm_m0",
    )
    correct_context = forecast_task_context_v1(task_spec=task_spec)
    neutral_context = forecast_task_context_v1(
        task_spec=task_spec,
        quality_contract=forecast_neutral_task_quality_contract_v1(),
    )
    contexts = {"correct": correct_context, "neutral": neutral_context}
    corpus_epoch = str(prereg["corpus_epoch"])
    bindings = {
        arm: _binding(
            context=context,
            snapshot=snapshot,
            code_commit=code_commit,
            corpus_epoch=corpus_epoch,
            model=model,
        )
        for arm, context in contexts.items()
    }

    budgeted = BudgetedAgentBackend(
        backend,
        maximum_calls=int(prereg["maximum_uncached_api_calls"]),
    )
    cache = EffectiveRequestCache(run_root / "agent_cache")
    cached = CachedAgentBackend(budgeted, cache)
    runners = {
        arm: _CycleCaseRunner(
            backend=cached,
            valuator=valuator,
            rules=rules,
            run_context_sha=bindings[arm].sha(),
            model=model,
            base_url=base_url,
            task_context=context,
            run_dependency_binding=bindings[arm],
        )
        for arm, context in contexts.items()
    }

    evaluations: dict[tuple[str, str], object] = {}
    for arm, case_ids in (("correct", correct_ids), ("neutral", neutral_ids)):
        for case_id in case_ids:
            evaluations[(arm, case_id)] = runners[arm].evaluate(
                materialized,
                cases_by_id[case_id],
            )

    private_rows: list[dict[str, object]] = []
    for (arm, case_id), evaluation in evaluations.items():
        trace = evaluation.trace
        private_rows.append(
            {
                "arm": arm,
                "case_id": case_id,
                "private_family": evaluation.case.private_family,
                "private_severity": evaluation.case.private_severity,
                "task_context_sha": trace.task_context_sha,
                "run_context_sha": trace.run_context_sha,
                "compilation_status": trace.compilation_status,
                "execution_status": trace.execution_status,
                "candidate_ids": list(trace.candidate_ids),
                "candidate_receipt_shas": dict(trace.candidate_receipt_shas),
                "rejection_receipts": [dict(value) for value in trace.rejection_receipts],
                "chosen_candidate_id": trace.chosen_candidate_id,
                "retrieved_skill_ids": list(trace.retrieved_skill_ids),
                "behavior_signature_sha": evaluation.receipt.behavior_signature_sha,
                "decision_fingerprint": _decision_fingerprint(evaluation),
                "utility_u": evaluation.receipt.utility_u,
            }
        )

    expected_task_sha = {arm: context.sha() for arm, context in contexts.items()}
    binding_ok = all(
        row["task_context_sha"] == expected_task_sha[str(row["arm"])]
        and row["run_context_sha"] == bindings[str(row["arm"])].sha()
        for row in private_rows
    )
    completed_rows = [
        row
        for row in private_rows
        if row["compilation_status"] == "ok" and row["execution_status"] == "ok"
    ]
    receipt_coverage_ok = all(
        set(row["candidate_ids"]) == set(row["candidate_receipt_shas"])
        for row in completed_rows
    )
    cache_audit = audit_stage_cache(
        cache.root,
        run_sha_by_task_sha={
            context.sha(): bindings[arm].sha() for arm, context in contexts.items()
        },
    )
    matched_behavior_changes = sum(
        _decision_fingerprint(evaluations[("correct", case_id)])
        != _decision_fingerprint(evaluations[("neutral", case_id)])
        for case_id in neutral_ids
    )
    minimum_completed = int(prereg["acceptance"]["minimum_completed_correct_cases"])
    minimum_completed_neutral = int(
        prereg["acceptance"]["minimum_completed_neutral_cases"]
    )
    completed_correct = sum(
        row["arm"] == "correct"
        and row["compilation_status"] == "ok"
        and row["execution_status"] == "ok"
        for row in private_rows
    )
    completed_neutral = sum(
        row["arm"] == "neutral"
        and row["compilation_status"] == "ok"
        and row["execution_status"] == "ok"
        for row in private_rows
    )
    status = "PASS" if (
        snapshot.harness_content_sha == expected_content_sha
        and binding_ok
        and receipt_coverage_ok
        and cache_audit["status"] == "PASS"
        and completed_correct >= minimum_completed
        and completed_neutral >= minimum_completed_neutral
    ) else "FAIL"

    private_report = {
        "schema_version": "f1-forecast-pilot-private-report/1",
        "status": status,
        "preregistration_sha": canonical_sha256(prereg),
        "implementation_commit": code_commit,
        "h2_harness_content_sha": snapshot.harness_content_sha,
        "f1_runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "task_contexts": {arm: context.to_dict() for arm, context in contexts.items()},
        "task_context_shas": expected_task_sha,
        "run_dependency_bindings": {
            arm: binding.to_dict() for arm, binding in bindings.items()
        },
        "run_context_shas": {arm: binding.sha() for arm, binding in bindings.items()},
        "case_rows": private_rows,
        "cache_audit": cache_audit,
        "checks": {
            "h2_content_unchanged": snapshot.harness_content_sha == expected_content_sha,
            "task_and_run_binding": binding_ok,
            "select_receipt_coverage": receipt_coverage_ok,
            "stage_cache_information_timing": cache_audit["status"] == "PASS",
            "minimum_completed_correct_cases": completed_correct >= minimum_completed,
            "minimum_completed_neutral_cases": (
                completed_neutral >= minimum_completed_neutral
            ),
        },
        "matched_neutral_behavior_change_count": matched_behavior_changes,
        "paid_usage": {
            "calls": budgeted.calls,
            "prompt_tokens": budgeted.prompt_tokens,
            "completion_tokens": budgeted.completion_tokens,
            "returned_models": sorted(budgeted.returned_models),
        },
    }
    private_report["report_sha"] = canonical_sha256(private_report)
    public_report = {
        "schema_version": "f1-forecast-pilot-public-report/1",
        "status": status,
        "scientific_role": "functional_contract_and_receipt_qualification",
        "harness_promotion_enabled": False,
        "h2_harness_content_sha": snapshot.harness_content_sha,
        "f1_runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "correct_contract_case_count": len(correct_ids),
        "neutral_contract_case_count": len(neutral_ids),
        "completed_correct_case_count": completed_correct,
        "completed_neutral_case_count": completed_neutral,
        "select_receipt_coverage_pass": receipt_coverage_ok,
        "task_and_run_binding_pass": binding_ok,
        "stage_cache_information_timing_pass": cache_audit["status"] == "PASS",
        "matched_neutral_behavior_change_count": matched_behavior_changes,
        "paid_usage": private_report["paid_usage"],
        "private_report_sha": private_report["report_sha"],
        "claim_limit": (
            "F1 validates objective/receipt plumbing on frozen forecast H2; it does "
            "not claim a new Harness capability or multi-task improvement."
        ),
    }
    _write_json(run_root / "private" / "f1_forecast_pilot_report.json", private_report)
    _write_json(run_root / "public" / "f1_forecast_pilot_report.json", public_report)
    return private_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen-H2 F1 forecast pilot.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--preregistration", type=Path, default=_DEFAULT_PREREGISTRATION)
    parser.add_argument("--release-root", type=Path, default=_DEFAULT_RELEASE)
    parser.add_argument("--rules", type=Path, default=_DEFAULT_RULES)
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_AGENT_BASE_URL)
    parser.add_argument("--resume", action="store_true")
    return parser


def _main() -> int:
    args = _parser().parse_args()
    api_key = os.environ.get("AGICTO_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("AGICTO_API_KEY is required")
    backend = AgictoChatCompletionsBackend(
        api_key=api_key,
        base_url=args.base_url,
        timeout_seconds=180,
    )
    report = run_f1_forecast_pilot(
        run_root=args.run_root,
        backend=backend,
        valuator=FrozenChronosValuator(),
        code_commit=args.code_commit,
        preregistration_path=args.preregistration,
        release_root=args.release_root,
        rules_path=args.rules,
        model=args.model,
        base_url=args.base_url,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_sha": report["report_sha"],
                "paid_usage": report["paid_usage"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["audit_stage_cache", "run_f1_forecast_pilot"]
