from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_document,
)
from SelfEvolvingHarnessTS.contracts.harness import (
    EditManifest,
    EditOperation,
    load_learned_skill_entry,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import build_core_corpus
from SelfEvolvingHarnessTS.evaluation.minipipe.cycle import _CaseEvaluation, _CycleCaseRunner
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.panel import (
    PROBE_INSTRUMENT_EPOCH,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.paired import (
    CaseRunReceipt,
    PairedReplayRunner,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (
    MaterializedSnapshot,
    SnapshotStore,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (
    DEFAULT_AGENT_BASE_URL,
    DEFAULT_AGENT_MODEL,
    AgentCallBudgetExceeded,
    AgentTransportError,
    AgictoChatCompletionsBackend,
    BudgetedAgentBackend,
)
from SelfEvolvingHarnessTS.runtime.llm_cache import (
    CachedAgentBackend,
    EffectiveRequestCache,
)


_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_H0 = _ROOT / "methods/ttha/harness/h0"
_DEFAULT_RULES = _ROOT / "evaluation/minipipe/config/m0_rules.json"
_DEFAULT_REFERENCE_SKILL = (
    Path(__file__).resolve().parent
    / "reference_skill_entries/closed_level_excursion_repair_v2.json"
)
_DEFAULT_PREREGISTRATION = (
    _ROOT
    / "artifacts/manifests/reference_level_wind_tunnel_preregistration_v2_20260719.json"
)


class _RecordingCaseRunner:
    def __init__(self, delegate: _CycleCaseRunner) -> None:
        self.delegate = delegate
        self.evaluations: dict[tuple[str, str], _CaseEvaluation] = {}

    def run(
        self,
        snapshot: MaterializedSnapshot,
        case: object,
        cache: object,
    ) -> CaseRunReceipt:
        del cache
        evaluation = self.delegate.evaluate(snapshot, case)
        self.evaluations[(snapshot.runtime_bundle_sha, evaluation.case.case_id)] = (
            evaluation
        )
        return evaluation.receipt


def _read_object(path: Path) -> dict[str, Any]:
    value = parse_json_document(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def build_reference_manifest(
    parent: MaterializedSnapshot,
    controller: EditController,
    *,
    reference_skill_path: Path = _DEFAULT_REFERENCE_SKILL,
) -> EditManifest:
    skill_value = _read_object(reference_skill_path)
    skill = load_learned_skill_entry(skill_value)
    if len(skill.allowed_tools) != 1:
        raise ValueError("a reference calibration skill must name exactly one operator")
    expected_operator = skill.allowed_tools[0]
    target_surface = f"skill_library.entries/{skill.skill_id}"
    dependency_keys = controller.surfaces.resolve(
        target_surface
    ).definition.required_dependency_keys
    return EditManifest(
        edit_id=f"calibrate-{skill.skill_id}",
        base_harness_sha=parent.harness_content_sha,
        target_pattern_id=f"pattern-calibrate-{skill.skill_id}",
        target_surface_id=target_surface,
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={
            key: parent.snapshot.dependency_shas[key] for key in dependency_keys
        },
        new_value=skill_value,
        observable_applicability=skill_value["observable_applicability"],
        predicted_agent_behavior_change=(
            f"retrieve_skill:{skill.skill_id}",
            f"supply_operator:{expected_operator}",
            "supply_effect_distinct",
            "choose_candidate_kind:program",
            "identity_retained",
            (
                "scope_modified_fraction<="
                f"{float(skill.risk_guards['max_modified_fraction']):g}"
            ),
            "effective_view_unchanged_out_of_scope",
        ),
        predicted_data_effect=("target utility improves",),
        falsification_condition=(
            "reference skill is not retrieved",
            "reference skill does not change target behavior",
            "a genuine-event risk case changes",
        ),
    )


def _evaluation_row(
    baseline: _CaseEvaluation,
    edited: _CaseEvaluation,
    *,
    reference_skill_id: str,
    expected_operator: str,
    scope_limit: float,
) -> dict[str, object]:
    edited_receipt = edited.receipt
    behavior_changed = (
        baseline.receipt.behavior_signature_sha
        != edited_receipt.behavior_signature_sha
    )
    behavior_flip = (
        baseline.receipt.chosen_candidate_kind == "identity"
        and edited_receipt.chosen_candidate_kind == "program"
        and behavior_changed
    )
    funnel = {
        "retrieved": reference_skill_id in edited_receipt.retrieved_skill_ids,
        "supplied_expected_operator": (
            expected_operator in edited_receipt.supplied_operator_ids
        ),
        "supplied_effect_distinct": edited_receipt.supplied_effect_distinct,
        "selected_program": edited_receipt.chosen_candidate_kind == "program",
        "identity_retained": edited_receipt.identity_retained,
        "scope_within_limit": edited_receipt.modified_fraction <= scope_limit,
    }
    return {
        "case_id": edited.case.case_id,
        "baseline_behavior_signature_sha": baseline.receipt.behavior_signature_sha,
        "edited_behavior_signature_sha": edited_receipt.behavior_signature_sha,
        "behavior_changed": behavior_changed,
        "behavior_flip": behavior_flip,
        "funnel": funnel,
        "baseline_chosen_candidate_kind": baseline.receipt.chosen_candidate_kind,
        "edited_chosen_candidate_kind": edited_receipt.chosen_candidate_kind,
        "edited_retrieved_skill_ids": list(edited_receipt.retrieved_skill_ids),
        "edited_supplied_operator_ids": list(edited_receipt.supplied_operator_ids),
        "expected_operator": expected_operator,
        "baseline_utility_u": baseline.receipt.utility_u,
        "edited_utility_u": edited_receipt.utility_u,
        "paired_improvement_u": (
            edited_receipt.utility_u - baseline.receipt.utility_u
        ),
        "edited_repair_gain_g": edited.feedback.outcome.repair_gain_g,
        "edited_modified_fraction": edited_receipt.modified_fraction,
        "edited_cache_hit_flags": list(edited_receipt.cache_hit_flags),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def run_reference_wind_tunnel(
    *,
    run_root: Path,
    backend: object,
    valuator: object,
    call_budget: int = 70,
    h0_root: Path = _DEFAULT_H0,
    rules_path: Path = _DEFAULT_RULES,
    reference_skill_path: Path = _DEFAULT_REFERENCE_SKILL,
    preregistration_path: Path = _DEFAULT_PREREGISTRATION,
    model: str = DEFAULT_AGENT_MODEL,
    base_url: str = DEFAULT_AGENT_BASE_URL,
    resume: bool = False,
) -> dict[str, object]:
    run_root = Path(run_root).resolve()
    prior_payload: dict[str, Any] | None = None
    if run_root.exists() and any(run_root.iterdir()):
        if not resume:
            raise FileExistsError("wind-tunnel run_root must be absent or empty")
        prior_report = run_root / "private/reference_wind_tunnel_report.json"
        if not prior_report.is_file():
            raise FileNotFoundError("wind-tunnel resume requires a private prior report")
        prior_payload = _read_object(prior_report)
        prior_replay = prior_payload.get("paired_replay_report", {})
        if (
            not isinstance(prior_replay, dict)
            or prior_replay.get("verdict") != "INCONCLUSIVE"
            or prior_replay.get("infrastructure_error") != "AgentTransportError"
        ):
            raise ValueError(
                "wind-tunnel resume is allowed only after AgentTransportError INCONCLUSIVE"
            )
    run_root.mkdir(parents=True, exist_ok=True)
    rules = load_m0_rules(rules_path)
    corpus = build_core_corpus(rules)
    preregistration = _read_object(preregistration_path)
    h0 = compile_snapshot(h0_root)
    store = SnapshotStore(run_root / "harness_snapshots")
    parent = store.materialize(h0)
    controller = EditController(store)
    manifest = build_reference_manifest(
        parent,
        controller,
        reference_skill_path=reference_skill_path,
    )
    applied = controller.apply_to_fork(
        parent,
        manifest,
        confirmed_cause="SKILL_LIBRARY_GAP",
    )
    if call_budget > int(preregistration["primary_call_budget_max"]):
        raise ValueError("call budget exceeds the preregistered primary maximum")
    target_family = str(preregistration.get("target_family", "level_shift"))
    expected_operator = str(
        preregistration.get("expected_operator", "repair_level_shift")
    )
    if expected_operator not in tuple(manifest.new_value["allowed_tools"]):
        raise ValueError("preregistered operator is absent from the reference skill")

    run_context = {
        "schema_version": "reference-wind-tunnel-context/1",
        "scientific_role": "positive_control_not_autonomous_growth_evidence",
        "model": model,
        "base_url": base_url,
        "h0_runtime_bundle_sha": parent.runtime_bundle_sha,
        "candidate_runtime_bundle_sha": applied.candidate_runtime_bundle_sha,
        "rules_sha": rules.rules_sha,
        "probe_instrument_epoch": PROBE_INSTRUMENT_EPOCH,
        "preregistration_sha": canonical_sha256(preregistration),
        "reference_skill_sha": canonical_sha256(manifest.new_value),
        "target_family": target_family,
        "expected_operator": expected_operator,
        "call_budget": call_budget,
    }
    run_context_sha = canonical_sha256(run_context)
    prior_calls = 0
    prior_prompt_tokens = 0
    prior_completion_tokens = 0
    prior_models: set[str] = set()
    prior_resumes = 0
    if prior_payload is not None:
        if prior_payload.get("run_context") != run_context:
            raise ValueError("wind-tunnel resume context differs from prior attempt")
        prior_usage = prior_payload.get("paid_usage", {})
        if not isinstance(prior_usage, dict):
            raise ValueError("prior wind-tunnel usage receipt is invalid")
        prior_calls = int(prior_usage.get("calls", 0))
        prior_prompt_tokens = int(prior_usage.get("prompt_tokens", 0))
        prior_completion_tokens = int(prior_usage.get("completion_tokens", 0))
        prior_models = {
            str(item) for item in prior_usage.get("returned_models", ())
        }
        prior_resumes = int(prior_usage.get("transport_resumes", 0))
        prior_sha = prior_payload.get("report_sha")
        if not isinstance(prior_sha, str) or len(prior_sha) != 64:
            raise ValueError("prior wind-tunnel report SHA is invalid")
        _write_json(
            run_root / "private" / "attempts" / f"{prior_sha}.json",
            prior_payload,
        )
    remaining_calls = call_budget - prior_calls
    if remaining_calls <= 0:
        raise AgentCallBudgetExceeded(
            f"wind-tunnel cumulative call budget exhausted at {call_budget}"
        )
    budgeted = BudgetedAgentBackend(backend, maximum_calls=remaining_calls)
    cache = EffectiveRequestCache(run_root / "agent_cache")
    cached = CachedAgentBackend(budgeted, cache)
    case_delegate = _CycleCaseRunner(
        backend=cached,
        valuator=valuator,
        rules=rules,
        run_context_sha=run_context_sha,
        model=model,
        base_url=base_url,
    )
    recording = _RecordingCaseRunner(case_delegate)
    paired = PairedReplayRunner(recording, rules=rules, cache=cache)
    targets = tuple(
        case for case in corpus.targets if case.private_family == target_family
    )
    risks = tuple(
        case for case in corpus.risks if case.private_family == target_family
    )
    report = paired.run(
        parent=parent,
        candidate=applied.candidate_snapshot,
        applied=applied,
        manifest=manifest,
        target_cases=targets,
        risk_cases=risks,
        out_of_scope_case_ids=tuple(case.case_id for case in risks),
    )

    reference_skill_id = str(manifest.new_value["skill_id"])
    scope_limit = float(manifest.new_value["risk_guards"]["max_modified_fraction"])
    target_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    target_ids = {case.case_id for case in targets}
    for case in (*targets, *risks):
        key_h0 = (parent.runtime_bundle_sha, case.case_id)
        key_edit = (applied.candidate_runtime_bundle_sha, case.case_id)
        if key_h0 not in recording.evaluations or key_edit not in recording.evaluations:
            continue
        row = _evaluation_row(
            recording.evaluations[key_h0],
            recording.evaluations[key_edit],
            reference_skill_id=reference_skill_id,
            expected_operator=expected_operator,
            scope_limit=scope_limit,
        )
        (target_rows if case.case_id in target_ids else risk_rows).append(row)

    flipped = [row for row in target_rows if row["behavior_flip"]]
    flipped_mean_g = (
        sum(float(row["edited_repair_gain_g"]) for row in flipped) / len(flipped)
        if flipped
        else None
    )
    target_gate = (
        len(target_rows)
        == int(
            preregistration["target_gate"].get(
                "target_count",
                preregistration["target_gate"].get("level_target_count", -1),
            )
        )
        and len(flipped)
        >= int(preregistration["target_gate"]["minimum_behavior_flips"])
        and flipped_mean_g is not None
        and flipped_mean_g
        > float(
            preregistration["target_gate"][
                "flipped_case_mean_repair_gain_g_min_exclusive"
            ]
        )
    )
    risk_gate = (
        len(risk_rows) == len(risks)
        and all(
            reference_skill_id not in row["edited_retrieved_skill_ids"]
            and not row["behavior_changed"]
            and abs(float(row["paired_improvement_u"]))
            <= float(rules["utility_tolerance"])
            for row in risk_rows
        )
    )
    status = (
        "INCONCLUSIVE"
        if report.verdict.value == "INCONCLUSIVE"
        else "PASS"
        if target_gate and risk_gate
        else "FAIL"
    )
    payload: dict[str, object] = {
        "schema_version": "reference-wind-tunnel-report/1",
        "scientific_role": "positive_control_not_autonomous_growth_evidence",
        "status": status,
        "run_context": run_context,
        "run_context_sha": run_context_sha,
        "manifest_sha": canonical_sha256(
            {
                "edit_id": manifest.edit_id,
                "target_surface_id": manifest.target_surface_id,
                "new_value": manifest.new_value,
                "predicted_agent_behavior_change": (
                    manifest.predicted_agent_behavior_change
                ),
            }
        ),
        "applied_edit_sha": applied.applied_edit_sha,
        "paired_replay_report": report.to_private_json(),
        "target_cases": target_rows,
        "risk_cases": risk_rows,
        "gates": {
            "target_gate": target_gate,
            "risk_gate": risk_gate,
            "behavior_flip_definition": (
                "baseline identity -> edited program with changed behavior signature"
            ),
            "behavior_flip_count": len(flipped),
            "flipped_case_mean_repair_gain_g": flipped_mean_g,
        },
        "paid_usage": {
            "call_budget": call_budget,
            "calls": prior_calls + budgeted.calls,
            "prompt_tokens": prior_prompt_tokens + budgeted.prompt_tokens,
            "completion_tokens": (
                prior_completion_tokens + budgeted.completion_tokens
            ),
            "returned_models": sorted(prior_models | budgeted.returned_models),
            "transport_resumes": prior_resumes + (1 if resume else 0),
        },
    }
    payload["report_sha"] = canonical_sha256(payload)
    _write_json(run_root / "private/reference_wind_tunnel_report.json", payload)
    public_payload = {
        "schema_version": payload["schema_version"],
        "scientific_role": payload["scientific_role"],
        "status": payload["status"],
        "run_context_sha": run_context_sha,
        "gates": payload["gates"],
        "paid_usage": payload["paid_usage"],
        "report_sha": payload["report_sha"],
    }
    _write_json(run_root / "public/reference_wind_tunnel_report.json", public_payload)
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded GPT reference-skill positive-control wind tunnel."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only an AgentTransportError INCONCLUSIVE run using its cache.",
    )
    parser.add_argument("--call-budget", type=int, default=70)
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_AGENT_BASE_URL)
    parser.add_argument(
        "--reference-skill",
        type=Path,
        default=_DEFAULT_REFERENCE_SKILL,
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=_DEFAULT_PREREGISTRATION,
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Optional short-lived credential file for cross-runtime execution.",
    )
    args = parser.parse_args()
    api_key = os.environ.get("AGICTO_API_KEY", "")
    if args.api_key_file is not None:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("AGICTO_API_KEY is required")
    backend = AgictoChatCompletionsBackend(
        api_key=api_key,
        base_url=args.base_url,
        timeout_seconds=180,
    )
    payload: dict[str, object] | None = None
    for attempt in range(3):
        payload = run_reference_wind_tunnel(
            run_root=args.run_root,
            backend=backend,
            valuator=FrozenChronosValuator(),
            call_budget=args.call_budget,
            reference_skill_path=args.reference_skill,
            preregistration_path=args.preregistration,
            model=args.model,
            base_url=args.base_url,
            resume=args.resume or attempt > 0,
        )
        replay = payload.get("paired_replay_report", {})
        if (
            not isinstance(replay, dict)
            or replay.get("verdict") != "INCONCLUSIVE"
            or replay.get("infrastructure_error") != "AgentTransportError"
        ):
            break
        if attempt < 2:
            print(f"transient_transport_resume={attempt + 1}/2")
    assert payload is not None
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gates": payload["gates"],
                "paid_usage": payload["paid_usage"],
                "report_sha": payload["report_sha"],
            },
            sort_keys=True,
        )
    )
    if payload["status"] == "PASS":
        return 0
    return 3 if payload["status"] == "INCONCLUSIVE" else 2


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "AgentCallBudgetExceeded",
    "build_reference_manifest",
    "run_reference_wind_tunnel",
]
