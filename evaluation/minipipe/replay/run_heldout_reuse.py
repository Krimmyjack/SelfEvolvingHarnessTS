from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_document,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.contracts import PrivateSyntheticCase
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import (
    build_heldout_corpus,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.cycle import (
    _CaseEvaluation,
    _CycleCaseRunner,
    _agent_panel,
    _manifest_from_json,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.paired import (
    CaseRunReceipt,
    EditVerdict,
    PairedReplayRunner,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.panel import (
    M0_PROBE_SPECS,
    PROBE_INSTRUMENT_EPOCH,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (
    MaterializedSnapshot,
    SnapshotStore,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view
from SelfEvolvingHarnessTS.runtime.agent_backend import (
    DEFAULT_AGENT_BASE_URL,
    DEFAULT_AGENT_MODEL,
    AgentCallBudgetExceeded,
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
_DEFAULT_SOURCE_RUN = _ROOT / "runs/minipipe/live-scientific-20260719-run4"
_DEFAULT_PREREGISTRATION = (
    _ROOT
    / "artifacts/manifests/live_m0_run4_heldout_preregistration_20260719.json"
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


def _read_object(path: Path) -> dict[str, object]:
    value = parse_json_document(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = canonical_json_bytes(value) + b"\n"
    if path.exists() and path.read_bytes() == expected:
        return
    path.write_bytes(expected)


def _rules_with_seeds(base: Mapping[str, object], seeds: Sequence[int]) -> dict[str, object]:
    value = copy.deepcopy(dict(base))
    corpus = copy.deepcopy(dict(value["corpus"]))
    corpus["base_seeds"] = [int(seed) for seed in seeds]
    value["corpus"] = corpus
    return value


def _load_source_manifest(source_run: Path, edit_id: str):
    payload = _read_object(source_run / "public/edit_manifest.json")
    edits = payload.get("edits")
    if not isinstance(edits, list):
        raise ValueError("source edit manifest set is invalid")
    matches = [item for item in edits if isinstance(item, dict) and item.get("edit_id") == edit_id]
    if len(matches) != 1:
        raise ValueError("source run does not contain exactly one requested edit")
    return _manifest_from_json(matches[0])


def load_source_cause(source_run: Path, target_pattern_id: str) -> str:
    """Recover the edit authorization cause from the frozen public pattern card."""

    payload = _read_object(Path(source_run) / "public/failure_patterns.json")
    patterns = payload.get("patterns")
    if not isinstance(patterns, list):
        raise ValueError("source failure-pattern set is invalid")
    matches = [
        row
        for row in patterns
        if isinstance(row, dict) and row.get("pattern_id") == target_pattern_id
    ]
    if len(matches) != 1:
        raise ValueError("source run does not contain exactly one target pattern")
    cause = matches[0].get("cause_code")
    if not isinstance(cause, str) or not cause:
        raise ValueError("source target pattern has no cause_code")
    return cause


def resolve_heldout_call_budget(
    preregistration: Mapping[str, object],
    *,
    requested_call_budget: int,
    budget_amendment: Mapping[str, object] | None = None,
) -> int:
    """Validate a resource-only budget amendment against the frozen experiment."""

    paired_contract = preregistration.get("paired_evaluation")
    if not isinstance(paired_contract, Mapping):
        raise ValueError("held-out paired-evaluation contract is invalid")
    registered_budget = int(paired_contract["hard_uncached_api_call_limit"])
    if budget_amendment is not None:
        if (
            budget_amendment.get("preregistration_id")
            != preregistration.get("preregistration_id")
            or int(budget_amendment.get("old_hard_uncached_api_call_limit", -1))
            != registered_budget
            or budget_amendment.get(
                "registered_before_any_heldout_outcome_or_h1_result"
            )
            is not True
        ):
            raise ValueError("held-out resource-budget amendment is invalid")
        amended_budget = int(
            budget_amendment["new_hard_uncached_api_call_limit"]
        )
        if amended_budget <= registered_budget:
            raise ValueError("held-out resource-budget amendment must raise the limit")
        registered_budget = amended_budget
    if requested_call_budget <= 0 or requested_call_budget > registered_budget:
        raise ValueError("call budget exceeds held-out preregistration")
    return registered_budget


def select_screened_targets(
    rows: Sequence[Mapping[str, object]],
    *,
    minimum: int,
) -> tuple[Mapping[str, object], ...]:
    """Take the preregistered first applicability matches without outcome access."""

    if minimum <= 0:
        raise ValueError("held-out target minimum must be positive")
    selected = tuple(row for row in rows if row.get("applicability_match") is True)[:minimum]
    return selected


def _public_features(
    delegate: _CycleCaseRunner,
    case: PrivateSyntheticCase,
) -> dict[str, object]:
    public_panel = delegate._public_panel(case)
    gateway = LocalPublicToolGateway(
        case.corrupt_context,
        task_kind="forecast",
        fixed_probe_panel=_agent_panel(public_panel),
    )
    return dict(gateway.public_features)


def _case_row(
    baseline: _CaseEvaluation,
    edited: _CaseEvaluation,
    *,
    promoted_skill_id: str,
    is_selected_target: bool,
) -> dict[str, object]:
    before = baseline.receipt
    after = edited.receipt
    return {
        "case_id": edited.case.case_id,
        "seed": edited.case.seed,
        "private_family": edited.case.private_family,
        "private_severity": edited.case.private_severity,
        "selected_target": is_selected_target,
        "skill_retrieved": promoted_skill_id in after.retrieved_skill_ids,
        "baseline_behavior_signature_sha": before.behavior_signature_sha,
        "edited_behavior_signature_sha": after.behavior_signature_sha,
        "behavior_changed": before.behavior_signature_sha != after.behavior_signature_sha,
        "baseline_chosen_candidate_kind": before.chosen_candidate_kind,
        "edited_chosen_candidate_kind": after.chosen_candidate_kind,
        "edited_supplied_operator_ids": list(after.supplied_operator_ids),
        "edited_supplied_effect_distinct": after.supplied_effect_distinct,
        "baseline_utility_u": before.utility_u,
        "edited_utility_u": after.utility_u,
        "paired_improvement_u": after.utility_u - before.utility_u,
        "edited_modified_fraction": after.modified_fraction,
        "edited_cache_hit_flags": list(after.cache_hit_flags),
        "effective_view_equal": (
            before.effective_harness_view_sha == after.effective_harness_view_sha
        ),
    }


def run_promoted_edit_heldout(
    *,
    run_root: Path,
    backend: object,
    valuator: object,
    call_budget: int,
    source_run: Path = _DEFAULT_SOURCE_RUN,
    preregistration_path: Path = _DEFAULT_PREREGISTRATION,
    budget_amendment_path: Path | None = None,
    h0_root: Path = _DEFAULT_H0,
    base_rules_path: Path = _DEFAULT_RULES,
    model: str = DEFAULT_AGENT_MODEL,
    base_url: str = DEFAULT_AGENT_BASE_URL,
    resume: bool = False,
) -> dict[str, object]:
    run_root = Path(run_root).resolve()
    private_root = run_root / "private"
    prior: dict[str, object] | None = None
    if run_root.exists() and any(run_root.iterdir()):
        if not resume:
            raise FileExistsError("held-out run_root must be absent or empty")
        report_path = private_root / "heldout_reuse_report.json"
        if not report_path.is_file():
            raise FileNotFoundError("held-out resume requires a prior report")
        prior = _read_object(report_path)
        replay = prior.get("paired_replay_report")
        if not isinstance(replay, dict) or replay.get("verdict") != "INCONCLUSIVE":
            raise ValueError("held-out resume is allowed only after INCONCLUSIVE")
    run_root.mkdir(parents=True, exist_ok=True)

    prereg = _read_object(preregistration_path)
    budget_amendment: dict[str, object] | None = None
    if budget_amendment_path is not None:
        budget_amendment = _read_object(budget_amendment_path)
    resolve_heldout_call_budget(
        prereg,
        requested_call_budget=call_budget,
        budget_amendment=budget_amendment,
    )
    if model != prereg["requested_model"] or base_url != prereg["relay_base_url"]:
        raise ValueError("model or relay differs from held-out preregistration")
    current_probe_sha = canonical_sha256(
        {name: spec.implementation_sha for name, spec in M0_PROBE_SPECS.items()}
    )
    if (
        prereg["probe_instrument_epoch"] != PROBE_INSTRUMENT_EPOCH
        or prereg["probe_specs_sha"] != current_probe_sha
    ):
        raise ValueError("probe instrument differs from held-out preregistration")
    source_run = Path(source_run).resolve()
    edit_id = str(prereg["source_edit_id"])
    promoted_skill_id = str(prereg["promoted_skill_id"])
    manifest = _load_source_manifest(source_run, edit_id)
    source_cause = load_source_cause(source_run, manifest.target_pattern_id)
    source_replay = _read_object(source_run / "private/paired_replay_report.json")
    source_reports = source_replay.get("reports", [])
    supported = [
        row
        for row in source_reports
        if isinstance(row, dict)
        and row.get("edit_id") == edit_id
        and row.get("verdict") == "SUPPORTED_EDIT"
    ]
    if len(supported) != 1 or supported[0].get("report_sha") != prereg[
        "source_paired_replay_report_sha"
    ]:
        raise ValueError("source SUPPORTED_EDIT receipt does not match preregistration")
    if supported[0].get("confirmed_surface") != manifest.target_surface_id:
        raise ValueError("source SUPPORTED_EDIT confirmed another surface")

    store = SnapshotStore(run_root / "harness_snapshots")
    parent = store.materialize(compile_snapshot(h0_root))
    controller = EditController(store)
    applied = controller.apply_to_fork(
        parent,
        manifest,
        confirmed_cause=source_cause,
    )
    if parent.runtime_bundle_sha != prereg["parent_runtime_bundle_sha"]:
        raise ValueError("held-out parent snapshot differs from preregistration")
    if applied.candidate_runtime_bundle_sha != prereg["promoted_runtime_bundle_sha"]:
        raise ValueError("reapplied edit does not reproduce the promoted snapshot")

    base_rules_value = _read_object(base_rules_path)
    screening = prereg["screening"]
    if not isinstance(screening, dict):
        raise ValueError("held-out screening contract is invalid")
    seed_pool = tuple(int(value) for value in screening["candidate_seed_pool_in_order"])
    screen_rules_path = private_root / "screening_rules.json"
    _write_json(screen_rules_path, _rules_with_seeds(base_rules_value, seed_pool))
    screen_rules = load_m0_rules(screen_rules_path)
    screen_corpus = build_heldout_corpus(screen_rules)
    screen_delegate = _CycleCaseRunner(
        backend=backend,
        valuator=valuator,
        rules=screen_rules,
        run_context_sha="heldout-public-screening-no-agent-calls",
        model=model,
        base_url=base_url,
    )
    severity_order = {
        str(value): index for index, value in enumerate(screening["eligible_severities"])
    }
    pool_order = {seed: index for index, seed in enumerate(seed_pool)}
    eligible = sorted(
        (
            case
            for case in screen_corpus.targets
            if case.private_family == screening["eligible_family"]
            and case.private_severity in severity_order
        ),
        key=lambda case: (pool_order[case.seed], severity_order[case.private_severity]),
    )
    screen_rows: list[dict[str, object]] = []
    for case in eligible:
        features = _public_features(screen_delegate, case)
        view = resolve_harness_view(
            applied.candidate_snapshot.snapshot,
            features,
            role="fast",
        )
        screen_rows.append(
            {
                "seed": case.seed,
                "severity": case.private_severity,
                "public_signature_sha": canonical_sha256(features),
                "applicability_match": promoted_skill_id in view.skill_ids,
            }
        )
    minimum = int(screening["minimum_selected_target_cases"])
    selected = select_screened_targets(screen_rows, minimum=minimum)
    _write_json(
        private_root / "outcome_blind_screening_receipt.json",
        {
            "schema_version": "heldout-outcome-blind-screening/1",
            "preregistration_sha": canonical_sha256(prereg),
            "rows": screen_rows,
            "selected": list(selected),
            "forbidden_fields_read": [],
        },
    )
    if len(selected) < minimum:
        payload: dict[str, object] = {
            "schema_version": "promoted-edit-heldout-report/1",
            "scientific_role": prereg["scientific_role"],
            "status": "INCONCLUSIVE_APPLICABILITY_SUPPORT",
            "selected_target_count": len(selected),
            "required_target_count": minimum,
            "paid_usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        }
        payload["report_sha"] = canonical_sha256(payload)
        _write_json(private_root / "heldout_reuse_report.json", payload)
        _write_json(run_root / "public/heldout_reuse_report.json", payload)
        return payload

    selected_keys = tuple((int(row["seed"]), str(row["severity"])) for row in selected)
    selected_seeds = tuple(
        seed for seed in seed_pool if any(seed == selected_seed for selected_seed, _ in selected_keys)
    )
    heldout_rules_path = private_root / "heldout_rules.json"
    _write_json(heldout_rules_path, _rules_with_seeds(base_rules_value, selected_seeds))
    rules = load_m0_rules(heldout_rules_path)
    corpus = build_heldout_corpus(rules)
    targets_by_key = {
        (case.seed, case.private_severity): case
        for case in corpus.targets
        if case.private_family == screening["eligible_family"]
    }
    targets = tuple(targets_by_key[key] for key in selected_keys)
    target_ids = {case.case_id for case in targets}
    risk_and_scope = tuple(case for case in corpus.all_cases if case.case_id not in target_ids)

    prior_usage = prior.get("paid_usage", {}) if prior is not None else {}
    if not isinstance(prior_usage, dict):
        raise ValueError("prior held-out usage is invalid")
    prior_calls = int(prior_usage.get("calls", 0))
    remaining = call_budget - prior_calls
    if remaining <= 0:
        raise AgentCallBudgetExceeded("held-out cumulative call budget is exhausted")
    budgeted = BudgetedAgentBackend(backend, maximum_calls=remaining)
    cache = EffectiveRequestCache(run_root / "agent_cache")
    cached = CachedAgentBackend(budgeted, cache)
    corpus_sha = canonical_sha256(
        {
            "schema_version": "m0-corpus-instance/1",
            "case_private_shas": [case.private_sha for case in corpus.all_cases],
        }
    )
    run_context = {
        "schema_version": "promoted-edit-heldout-context/1",
        "scientific_role": prereg["scientific_role"],
        "model": model,
        "base_url": base_url,
        "parent_runtime_bundle_sha": parent.runtime_bundle_sha,
        "promoted_runtime_bundle_sha": applied.candidate_runtime_bundle_sha,
        "source_report_sha": prereg["source_paired_replay_report_sha"],
        "preregistration_sha": canonical_sha256(prereg),
        "resource_budget_amendment_sha": (
            None
            if budget_amendment is None
            else canonical_sha256(budget_amendment)
        ),
        "rules_sha": rules.rules_sha,
        "corpus_sha": corpus_sha,
        "selected_target_keys": [list(key) for key in selected_keys],
        "selected_seeds": list(selected_seeds),
        "call_budget": call_budget,
    }
    run_context_sha = canonical_sha256(run_context)
    delegate = _CycleCaseRunner(
        backend=cached,
        valuator=valuator,
        rules=rules,
        run_context_sha=run_context_sha,
        model=model,
        base_url=base_url,
    )
    out_of_scope_ids: list[str] = []
    for case in risk_and_scope:
        features = _public_features(delegate, case)
        view = resolve_harness_view(
            applied.candidate_snapshot.snapshot,
            features,
            role="fast",
        )
        if promoted_skill_id not in view.skill_ids:
            out_of_scope_ids.append(case.case_id)
    recording = _RecordingCaseRunner(delegate)
    paired = PairedReplayRunner(recording, rules=rules, cache=cache)
    report = paired.run(
        parent=parent,
        candidate=applied.candidate_snapshot,
        applied=applied,
        manifest=manifest,
        target_cases=targets,
        risk_cases=risk_and_scope,
        out_of_scope_case_ids=tuple(out_of_scope_ids),
    )

    rows: list[dict[str, object]] = []
    for case in (*targets, *risk_and_scope):
        before_key = (parent.runtime_bundle_sha, case.case_id)
        after_key = (applied.candidate_runtime_bundle_sha, case.case_id)
        if before_key not in recording.evaluations or after_key not in recording.evaluations:
            continue
        rows.append(
            _case_row(
                recording.evaluations[before_key],
                recording.evaluations[after_key],
                promoted_skill_id=promoted_skill_id,
                is_selected_target=case.case_id in target_ids,
            )
        )
    status = "PASS" if report.verdict is EditVerdict.SUPPORTED_EDIT else (
        "INCONCLUSIVE" if report.verdict is EditVerdict.INCONCLUSIVE else "FAIL"
    )
    payload = {
        "schema_version": "promoted-edit-heldout-report/1",
        "scientific_role": prereg["scientific_role"],
        "status": status,
        "run_context": run_context,
        "run_context_sha": run_context_sha,
        "screening_receipt_sha": canonical_sha256(
            _read_object(private_root / "outcome_blind_screening_receipt.json")
        ),
        "paired_replay_report": report.to_private_json(),
        "case_rows": rows,
        "paid_usage": {
            "call_budget": call_budget,
            "calls": prior_calls + budgeted.calls,
            "prompt_tokens": int(prior_usage.get("prompt_tokens", 0)) + budgeted.prompt_tokens,
            "completion_tokens": int(prior_usage.get("completion_tokens", 0)) + budgeted.completion_tokens,
            "returned_models": sorted(
                set(str(value) for value in prior_usage.get("returned_models", ()))
                | budgeted.returned_models
            ),
        },
    }
    payload["report_sha"] = canonical_sha256(payload)
    _write_json(private_root / "heldout_reuse_report.json", payload)
    public = {
        "schema_version": payload["schema_version"],
        "scientific_role": payload["scientific_role"],
        "status": payload["status"],
        "run_context_sha": run_context_sha,
        "selected_target_count": len(targets),
        "risk_and_scope_case_count": len(risk_and_scope),
        "verdict": report.verdict.value,
        "target_recovery_fraction": report.facts.target_recovery_fraction,
        "median_target_improvement": report.facts.median_target_improvement,
        "prediction_verified": report.facts.prediction_verified,
        "risk_status": report.facts.risk_status,
        "scope_status": report.facts.scope_status,
        "paid_usage": payload["paid_usage"],
        "report_sha": payload["report_sha"],
    }
    _write_json(run_root / "public/heldout_reuse_report.json", public)
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run an outcome-blind held-out reuse test.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=_DEFAULT_SOURCE_RUN)
    parser.add_argument("--preregistration", type=Path, default=_DEFAULT_PREREGISTRATION)
    parser.add_argument(
        "--parent-snapshot-root",
        type=Path,
        default=_DEFAULT_H0,
        help="Authoring/materialized root of the exact parent snapshot in the comparison.",
    )
    parser.add_argument("--budget-amendment", type=Path)
    parser.add_argument("--call-budget", type=int, default=180)
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_AGENT_BASE_URL)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    api_key = os.environ.get("AGICTO_API_KEY", "").strip()
    if args.api_key_file is not None:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("AGICTO_API_KEY or --api-key-file is required")
    backend = AgictoChatCompletionsBackend(
        api_key=api_key,
        base_url=args.base_url,
        timeout_seconds=180,
    )
    payload = run_promoted_edit_heldout(
        run_root=args.run_root,
        backend=backend,
        valuator=FrozenChronosValuator(),
        call_budget=args.call_budget,
        source_run=args.source_run,
        preregistration_path=args.preregistration,
        budget_amendment_path=args.budget_amendment,
        h0_root=args.parent_snapshot_root,
        model=args.model,
        base_url=args.base_url,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "report_sha": payload["report_sha"],
                "paid_usage": payload["paid_usage"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "load_source_cause",
    "resolve_heldout_call_budget",
    "run_promoted_edit_heldout",
    "select_screened_targets",
]
