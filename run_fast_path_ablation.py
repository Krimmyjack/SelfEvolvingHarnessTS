"""Run a no-API fast-path ablation slice.

This script is intentionally small and deterministic. It exercises the deployed
fast-path interfaces with a stub composer and a fixed synthetic reporter before
any real LLM/API call is introduced.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .fast_path.ablation import (
    FastPathAblationResult,
    standard_fast_path_ablation_arms,
    run_fast_path_ablation,
    write_fast_path_ablation_report,
)
from .memory import EvidenceStore
from .memory.evidence_schema import build_memory_evidence_v2
from .policy.action_spec import action_menu_v1
from .policy.skill_memory_composer import TypedCandidate
from .slow_path.evidence_miner import DeploymentEvidenceMiner, suggest_slow_path_proposals
from .slow_path.promotion import PromotionGate

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results" / "Stage2" / "FastPathAblation"
DEFAULT_LEDGER_RECORDS_PATH = Path(__file__).resolve().parent / "results" / "Stage2" / "S2_replication" / "records_s2.jsonl"
DEFAULT_ORACLE_LEDGER_OUT_DIR = Path(__file__).resolve().parent / "results" / "Stage2" / "FastPathOracleLedgerAblation"


def _record(uid: str, *, snr: float, miss_rate: float, cell: str) -> dict[str, Any]:
    return {
        "uid": uid,
        "cell": cell,
        "snr": snr,
        "miss_rate": miss_rate,
        "X_p": [24.0, 0.05, 0.1, 0.1, 0.0, 0.1, 0.2, 0.05],
    }


def _target_for_index(idx: int) -> np.ndarray:
    return np.sin(np.linspace(0, 2 * np.pi, 64)) + 0.05 * idx


def build_demo_forecast_ablation_inputs(n_records: int = 4) -> tuple[
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, list[Any]],
    dict[str, np.ndarray],
]:
    """Build a deterministic small forecast slice for interface ablation."""
    records: list[dict[str, Any]] = []
    series_by_uid: dict[str, np.ndarray] = {}
    memory_by_uid: dict[str, list[Any]] = {}
    target_by_uid: dict[str, np.ndarray] = {}
    for idx in range(int(n_records)):
        uid = f"demo_{idx}"
        snr = 12.0 if idx % 2 == 0 else -4.0
        miss_rate = 0.0 if idx % 2 == 0 else 0.2
        cell = "forecast|snrHigh|full" if idx % 2 == 0 else "forecast|snrLow|miss"
        target = _target_for_index(idx)
        x = target.copy()
        if miss_rate > 0.0:
            x[10:14] = np.nan
        records.append(_record(uid, snr=snr, miss_rate=miss_rate, cell=cell))
        series_by_uid[uid] = x
        target_by_uid[uid] = target
        memory_by_uid[uid] = [
            build_memory_evidence_v2(
                task="forecast",
                pattern_region=cell,
                memory_type="utility",
                role="recommend",
                skill_id="median_smooth",
                action_id="v_median",
                program={"steps": [{"op": "denoise_median", "params": {"window": 5}}]},
                raw_loss=2.0,
                selected_loss=1.7,
                confidence={"source": "synthetic_demo"},
                support={"n": 3, "n_unique_cases": 3, "slice": "synthetic_forecast_small"},
                evidence_refs=(f"demo:utility:{idx}",),
                provenance={"case_id": f"demo_memory_{idx}"},
            ),
            build_memory_evidence_v2(
                task="forecast",
                pattern_region=cell,
                memory_type="risk",
                role="warn",
                skill_id="median_smooth",
                action_id="v_median",
                program={"steps": [{"op": "denoise_median", "params": {"window": 5}}]},
                raw_loss=1.0,
                selected_loss=1.15 if idx % 2 else 1.05,
                confidence={"source": "synthetic_demo_counter_evidence"},
                support={"n": 1, "n_unique_cases": 1, "slice": "synthetic_forecast_small"},
                failure_signature="synthetic_prior_harm_vs_raw",
                evidence_refs=(f"demo:risk:{idx}",),
                provenance={"case_id": f"demo_risk_memory_{idx}"},
            ),
        ]
    return records, series_by_uid, memory_by_uid, target_by_uid


def _filled_mse(values: Any, target: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    tgt = np.asarray(target, dtype=float).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean((arr - tgt) ** 2))


def _finite_losses(record: Mapping[str, Any]) -> dict[str, float]:
    raw = record.get("L_test")
    if not isinstance(raw, Mapping):
        raise ValueError(f"ledger record {record.get('uid')!r} has no L_test mapping")
    losses: dict[str, float] = {}
    for action, value in raw.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            losses[str(action)] = numeric
    if not losses:
        raise ValueError(f"ledger record {record.get('uid')!r} has no finite L_test values")
    return losses


def _oracle_from_losses(losses: Mapping[str, float]) -> tuple[str, float]:
    action = min(losses, key=losses.get)
    return action, float(losses[action])


def _ledger_dummy_series(record: Mapping[str, Any], length: int = 64) -> np.ndarray:
    miss_rate = float(record.get("miss_rate", record.get("missing_rate", 0.0)) or 0.0)
    arr = np.zeros(int(length), dtype=float)
    if miss_rate > 0.0:
        n_missing = max(1, min(int(length), int(round(float(length) * min(miss_rate, 1.0)))))
        arr[:n_missing] = np.nan
    return arr


def _memories_from_ledger_record(record: Mapping[str, Any], *, raw_action: str) -> list[Any]:
    losses = _finite_losses(record)
    if raw_action not in losses:
        return []
    source_uid = str(record.get("uid") or "")
    cell = str(record.get("cell") or "")
    raw_loss = float(losses[raw_action])
    memories: list[Any] = []

    action, selected_loss = _oracle_from_losses(losses)
    if action != raw_action and raw_loss > selected_loss:
        memories.append(
            build_memory_evidence_v2(
                task="forecast",
                pattern_region=cell,
                memory_type="utility",
                role="recommend",
                skill_id="ledger_prior_best_action",
                action_id=action,
                program={"source": "ledger_prior_best_action", "steps": []},
                raw_loss=raw_loss,
                selected_loss=selected_loss,
                support={"n": 1, "n_unique_cases": 1, "slice": "s2_oracle_ledger_prior"},
                evidence_refs=(f"ledger:utility:{source_uid}:{action}",),
                provenance={"source_uid": source_uid, "source": "prior_same_cell_l_test"},
            )
        )

    for action_id, selected_loss in sorted(losses.items()):
        if action_id == raw_action or selected_loss <= raw_loss:
            continue
        memories.append(
            build_memory_evidence_v2(
                task="forecast",
                pattern_region=cell,
                memory_type="risk",
                role="warn",
                skill_id="ledger_prior_harmful_action",
                action_id=action_id,
                program={"source": "ledger_prior_harmful_action", "steps": []},
                raw_loss=raw_loss,
                selected_loss=selected_loss,
                support={"n": 1, "n_unique_cases": 1, "slice": "s2_oracle_ledger_prior"},
                failure_signature="prior_action_harm_vs_raw",
                evidence_refs=(f"ledger:risk:{source_uid}:{action_id}",),
                provenance={"source_uid": source_uid, "source": "prior_same_cell_l_test"},
            )
        )
    return memories


def _load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, Mapping):
                rows.append(dict(loaded))
    return rows


def build_oracle_ledger_ablation_inputs(
    records_or_path: str | Path | list[Mapping[str, Any]],
    *,
    n_records: int = 8,
    raw_action: str = "v_none",
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, list[Any]], dict[str, dict[str, float]]]:
    """Build a no-API ablation slice from an offline L_test ledger.

    Memory rows are causal within this ordered slice: each record can only see
    prior same-cell ledger outcomes, never its own current L_test values.
    """
    loaded = _load_jsonl_records(records_or_path) if isinstance(records_or_path, (str, Path)) else [dict(r) for r in records_or_path]
    selected = loaded[: int(n_records)]
    records: list[dict[str, Any]] = []
    series_by_uid: dict[str, np.ndarray] = {}
    memory_by_uid: dict[str, list[Any]] = {}
    losses_by_uid: dict[str, dict[str, float]] = {}
    prior_by_cell: dict[str, list[Any]] = {}
    for idx, source in enumerate(selected):
        uid = str(source.get("uid") or f"ledger_{idx}")
        cell = str(source.get("cell") or "")
        record = {
            "uid": uid,
            "cell": cell,
            "origin": str(source.get("origin") or ""),
            "snr": float(source.get("snr", source.get("SNR", 0.0)) or 0.0),
            "miss_rate": float(source.get("miss_rate", source.get("missing_rate", 0.0)) or 0.0),
            "X_p": list(source.get("X_p") or []),
        }
        records.append(record)
        series_by_uid[uid] = _ledger_dummy_series(record)
        memory_by_uid[uid] = list(prior_by_cell.get(cell, []))
        losses_by_uid[uid] = _finite_losses(source)
        memories = _memories_from_ledger_record(source, raw_action=raw_action)
        if memories:
            prior_by_cell.setdefault(cell, []).extend(memories)
    return records, series_by_uid, memory_by_uid, losses_by_uid


def ledger_oracle_validator(
    losses_by_uid: Mapping[str, Mapping[str, float]],
    *,
    raw_action: str = "v_none",
    margin: float = 0.0,
) -> Callable[[Any, Any, Mapping[str, Any]], Mapping[str, Any]]:
    """Return an offline ledger validator using post-hoc L_test losses."""

    def validator(raw: Any, artifact: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        decision = context["decision"]
        uid = str((decision.packet.get("provenance") or {}).get("source_uid") or "")
        losses = dict(losses_by_uid[uid])
        raw_loss = float(losses[raw_action])
        action_id = str(decision.action_id)
        valid = action_id in losses
        selected_loss = float(losses[action_id]) if valid else float("inf")
        oracle_action, oracle_loss = _oracle_from_losses(losses)
        utility = raw_loss - selected_loss if valid else float("-inf")
        harm = max(0.0, selected_loss - raw_loss) if valid else float("inf")
        execution_ok = bool(context["executed"].execution_ok)
        passed = bool(execution_ok and valid and harm <= float(margin))
        return {
            "validator": "ledger_l_test_oracle_v1",
            "passed": passed,
            "valid_action": bool(valid),
            "raw_action": raw_action,
            "raw_loss": raw_loss,
            "selected_loss": selected_loss,
            "oracle_action": oracle_action,
            "oracle_loss": oracle_loss,
            "utility_delta_vs_raw": utility,
            "harm_delta_vs_raw": harm,
            "regret_vs_oracle": selected_loss - oracle_loss if valid else float("inf"),
            "failure_signature": None if passed else "ledger_oracle_harm_or_invalid",
        }

    return validator


def synthetic_oracle_proxy_validator(target_by_uid: Mapping[str, np.ndarray]) -> Callable[[Any, Any, Mapping[str, Any]], Mapping[str, Any]]:
    """Return a deterministic utility/harm reporter for the synthetic slice."""

    def validator(raw: Any, artifact: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        decision = context["decision"]
        uid = str((decision.packet.get("provenance") or {}).get("source_uid") or "")
        target = target_by_uid[uid]
        raw_loss = _filled_mse(raw, target)
        selected_loss = _filled_mse(artifact, target)
        utility = raw_loss - selected_loss
        harm = max(0.0, selected_loss - raw_loss)
        execution_ok = bool(context["executed"].execution_ok)
        passed = execution_ok and harm <= 1e-12
        return {
            "validator": "synthetic_oracle_proxy_v1",
            "passed": passed,
            "raw_loss_proxy": raw_loss,
            "selected_loss_proxy": selected_loss,
            "utility_delta_vs_raw": utility,
            "harm_delta_vs_raw": harm,
            "failure_signature": None if passed else "synthetic_oracle_harm",
        }

    return validator


def _skill_id_supporting_action(skills: list[Mapping[str, Any]], action_id: str) -> str | None:
    for skill in skills:
        name = skill.get("name")
        allowed = {str(value) for value in skill.get("allowed_actions") or []}
        actions = skill.get("actions")
        if isinstance(actions, Mapping):
            allowed.update(str(value) for value in actions.values())
        if name is not None and action_id in allowed:
            return str(name)
    return None


def _memory_mapping_rows(memory: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    return [row for row in memory.get(key) or [] if isinstance(row, Mapping)]


def _row_refs(row: Mapping[str, Any], fallback: str) -> tuple[str, ...]:
    refs = row.get("evidence_refs")
    if isinstance(refs, (list, tuple)) and refs:
        return tuple(str(ref) for ref in refs)
    provenance = row.get("provenance")
    if isinstance(provenance, Mapping):
        for key in ("case_id", "source_uid", "evidence_ref"):
            if provenance.get(key):
                return (str(provenance[key]),)
    return (fallback,)


def _refs_with_stub(refs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*refs, "stub:no_api"]))


def stub_skill_memory_composer(packet: Mapping[str, Any]) -> TypedCandidate:
    """Deterministic typed composer used only for no-API ablations."""
    skills = [row for row in packet.get("skills") or [] if isinstance(row, Mapping)]
    memory = packet.get("memory") or {}
    if not isinstance(memory, Mapping):
        memory = {}
    utility = _memory_mapping_rows(memory, "utility_memory")
    risk = _memory_mapping_rows(memory, "risk_memory")
    prior = _memory_mapping_rows(memory, "prior_fragments")
    action_id = "v_none"
    skill_id: str | None = None
    refs: tuple[str, ...] = ()
    rationale = "deterministic_stub_no_api_composer"

    if utility:
        action_id = str(utility[0].get("action_id") or action_id)
        skill_id = _skill_id_supporting_action(skills, action_id)
        refs = _row_refs(utility[0], "utility_memory:1")
    elif prior:
        action_id = str(prior[0].get("action_id") or action_id)
        skill_id = _skill_id_supporting_action(skills, action_id)
        refs = _row_refs(prior[0], "prior_fragment:1")
    elif risk:
        return TypedCandidate(
            action_id="v_none",
            risk_rule={"source": "risk_memory", "n_risk": len(risk)},
            abstain_to_raw=True,
            rationale="risk_memory_only_abstain",
            evidence_refs=_refs_with_stub(_row_refs(risk[0], "risk_memory:1")),
        )
    elif skills:
        allowed = list(skills[0].get("allowed_actions") or [])
        action_id = str(allowed[0]) if allowed else action_id
        skill_id = _skill_id_supporting_action(skills, action_id)

    return TypedCandidate(
        skill_id=skill_id,
        action_id=action_id,
        rationale=rationale,
        evidence_refs=_refs_with_stub(refs),
    )


def _edit_op_payload(outcome: Any) -> dict[str, Any] | None:
    edit_op = getattr(outcome, "edit_op", None)
    if edit_op is None:
        return None
    return edit_op.to_dict()


def mine_and_write_slow_path_proposals(
    store: EvidenceStore,
    out_dir: str | Path,
    *,
    min_support: int = 2,
) -> dict[str, Any]:
    """Mine ablation evidence and validate proposal promotion candidates."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    miner = DeploymentEvidenceMiner(store)
    gate = PromotionGate(min_support=min_support)
    rows: list[dict[str, Any]] = []
    for cell_id in sorted(store.get_all_cells()):
        summary = miner.summarize_cell(cell_id)
        for proposal in suggest_slow_path_proposals(summary, min_support=min_support):
            outcome = gate.validate(proposal)
            rows.append({
                "cell_id": cell_id,
                "proposal": proposal.to_dict(),
                "accepted": bool(outcome.accepted),
                "reason": outcome.reason,
                "edit_op": _edit_op_payload(outcome),
            })
    with (path / "slow_path_proposals.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    kind_counts = Counter(row["proposal"]["kind"] for row in rows)
    accepted_kind_counts = Counter(row["proposal"]["kind"] for row in rows if row["accepted"])
    return {
        "schema": "fast_path_ablation_slow_path_summary_v1",
        "min_support": int(min_support),
        "n_cells": len(store.get_all_cells()),
        "n_proposals": len(rows),
        "n_promotion_accepted": sum(1 for row in rows if row["accepted"]),
        "proposal_kind_counts": dict(sorted(kind_counts.items())),
        "accepted_kind_counts": dict(sorted(accepted_kind_counts.items())),
    }


def run_demo_forecast_ablation(
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    n_records: int = 4,
) -> dict[str, Any]:
    records, series_by_uid, memory_by_uid, target_by_uid = build_demo_forecast_ablation_inputs(n_records)
    arms = standard_fast_path_ablation_arms(composer=stub_skill_memory_composer)
    store = EvidenceStore()
    results: list[FastPathAblationResult] = run_fast_path_ablation(
        records,
        series_by_uid,
        arms=arms,
        action_menu=action_menu_v1(),
        memory_by_uid=memory_by_uid,
        validator=synthetic_oracle_proxy_validator(target_by_uid),
        store=store,
        harness_version=1,
    )
    report = write_fast_path_ablation_report(
        results,
        out_dir,
        metadata={
            "slice": "synthetic_forecast_small",
            "n_records": int(n_records),
            "composer": "deterministic_stub_no_api_composer",
            "reporter": "synthetic_oracle_proxy_v1",
            "raw_action": "v_none",
            "raw_action_semantics": "v_none_is_impute_linear_baseline_not_strict_raw",
            "memory_protocol": "memory_v2_utility_risk_buckets",
            "ablation_matrix": "phase4_memory_v2",
            "api_calls": 0,
        },
    )
    report["slow_path"] = mine_and_write_slow_path_proposals(store, out_dir, min_support=2)
    report_path = Path(out_dir) / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def run_oracle_ledger_ablation(
    *,
    records_path: str | Path = DEFAULT_LEDGER_RECORDS_PATH,
    out_dir: str | Path = DEFAULT_ORACLE_LEDGER_OUT_DIR,
    n_records: int = 8,
    raw_action: str = "v_none",
) -> dict[str, Any]:
    records, series_by_uid, memory_by_uid, losses_by_uid = build_oracle_ledger_ablation_inputs(
        records_path,
        n_records=n_records,
        raw_action=raw_action,
    )
    arms = standard_fast_path_ablation_arms(composer=stub_skill_memory_composer)
    store = EvidenceStore()
    results: list[FastPathAblationResult] = run_fast_path_ablation(
        records,
        series_by_uid,
        arms=arms,
        action_menu=action_menu_v1(),
        memory_by_uid=memory_by_uid,
        validator=ledger_oracle_validator(losses_by_uid, raw_action=raw_action),
        store=store,
        harness_version=1,
    )
    report = write_fast_path_ablation_report(
        results,
        out_dir,
        metadata={
            "slice": "s2_oracle_ledger_small",
            "n_records": int(n_records),
            "records_path": Path(records_path).as_posix(),
            "composer": "deterministic_stub_no_api_composer",
            "reporter": "ledger_l_test_oracle_v1",
            "raw_action": raw_action,
            "raw_action_semantics": "v_none_is_ledger_baseline_action",
            "memory_protocol": "prior_same_cell_l_test_current_uid_excluded_memory_v2_utility_risk",
            "ablation_matrix": "phase4_memory_v2",
            "api_calls": 0,
        },
    )
    report["slow_path"] = mine_and_write_slow_path_proposals(store, out_dir, min_support=2)
    report_path = Path(out_dir) / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run no-API fast-path ablation slice")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-records", type=int, default=4)
    parser.add_argument("--slice", choices=("synthetic", "oracle-ledger"), default="synthetic")
    parser.add_argument("--records-path", default=str(DEFAULT_LEDGER_RECORDS_PATH))
    args = parser.parse_args(argv)
    if args.slice == "oracle-ledger":
        out_dir = args.out_dir or str(DEFAULT_ORACLE_LEDGER_OUT_DIR)
        report = run_oracle_ledger_ablation(
            records_path=args.records_path,
            out_dir=out_dir,
            n_records=args.n_records,
        )
    else:
        out_dir = args.out_dir or str(DEFAULT_OUT_DIR)
        report = run_demo_forecast_ablation(out_dir=out_dir, n_records=args.n_records)
    print(f"wrote {out_dir}")
    print(
        f"n_results={report['summary']['n_results']} "
        f"api_calls={report['metadata']['api_calls']} "
        f"slow_path_proposals={report['slow_path']['n_proposals']}"
    )


if __name__ == "__main__":
    main()
