"""Mine deployment EvidenceStore rows into slow-path proposal inputs."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..memory import EvidenceRecord, EvidenceStore
from .proposal_schema import SlowProposal


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _selected_action(record: EvidenceRecord) -> str:
    routing = record.routing or {}
    action = routing.get("selected_action")
    if action is None and isinstance(routing.get("candidate"), Mapping):
        action = routing["candidate"].get("action_id")
    return str(action or "")


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "scope"


def _task_from_cell_id(cell_id: str) -> str:
    head = str(cell_id or "").split("|", 1)[0].strip()
    return head or "unknown"


def _task_from_record(record: EvidenceRecord, fallback_cell_id: str) -> str:
    key = record.conditioning_key or {}
    task = key.get("task")
    if isinstance(task, Mapping):
        value = task.get("type") or task.get("name") or task.get("id")
    else:
        value = task
    if value:
        return str(value)
    return _task_from_cell_id(str(record.cell_id or key.get("cell_id") or fallback_cell_id))


def _case_id_from_record(record: EvidenceRecord) -> str:
    routing = record.routing or {}
    for key in ("source_uid", "uid", "case_id"):
        value = routing.get(key)
        if value:
            return str(value)
    packet = routing.get("packet")
    if isinstance(packet, Mapping):
        provenance = packet.get("provenance")
        if isinstance(provenance, Mapping):
            for key in ("source_uid", "uid", "case_id"):
                value = provenance.get(key)
                if value:
                    return str(value)
    key = record.conditioning_key or {}
    for key_name in ("source_uid", "uid", "case_id"):
        value = key.get(key_name)
        if value:
            return str(value)
    return f"record:{record.batch_id}:{record.timestamp}:{id(record)}"


@dataclass
class ActionEvidenceStats:
    action_id: str
    n: int = 0
    n_passed: int = 0
    utility_sum: float = 0.0
    harm_sum: float = 0.0
    harm_count: int = 0
    utility_positive_count: int = 0
    case_ids: set[str] = field(default_factory=set)
    passed_case_ids: set[str] = field(default_factory=set)
    harm_case_ids: set[str] = field(default_factory=set)
    utility_positive_case_ids: set[str] = field(default_factory=set)
    utility_by_case: dict[str, list[float]] = field(default_factory=dict)
    harm_by_case: dict[str, list[float]] = field(default_factory=dict)

    def add(self, *, passed: bool, utility: float, harm: float, case_id: str) -> None:
        self.n += 1
        self.case_ids.add(case_id)
        self.utility_by_case.setdefault(case_id, []).append(utility)
        self.harm_by_case.setdefault(case_id, []).append(harm)
        if passed:
            self.n_passed += 1
            self.passed_case_ids.add(case_id)
        self.utility_sum += utility
        self.harm_sum += harm
        if harm > 0.0:
            self.harm_count += 1
            self.harm_case_ids.add(case_id)
        if utility > 0.0:
            self.utility_positive_count += 1
            self.utility_positive_case_ids.add(case_id)

    @staticmethod
    def _mean_by_case(values_by_case: Mapping[str, list[float]]) -> float:
        if not values_by_case:
            return 0.0
        per_case = [sum(values) / len(values) for values in values_by_case.values() if values]
        return round(sum(per_case) / len(per_case), 12) if per_case else 0.0

    @property
    def mean_utility_delta_vs_raw(self) -> float:
        return round(self.utility_sum / self.n, 12) if self.n else 0.0

    @property
    def mean_harm_delta_vs_raw(self) -> float:
        return round(self.harm_sum / self.n, 12) if self.n else 0.0

    @property
    def mean_case_utility_delta_vs_raw(self) -> float:
        return self._mean_by_case(self.utility_by_case)

    @property
    def mean_case_harm_delta_vs_raw(self) -> float:
        return self._mean_by_case(self.harm_by_case)

    @property
    def n_unique_cases(self) -> int:
        return len(self.case_ids)

    @property
    def n_passed_unique_cases(self) -> int:
        return len(self.passed_case_ids)

    @staticmethod
    def _positive_case_count(values_by_case: Mapping[str, list[float]]) -> int:
        count = 0
        for values in values_by_case.values():
            if values and (sum(values) / len(values)) > 0.0:
                count += 1
        return count

    @property
    def harm_case_count(self) -> int:
        return self._positive_case_count(self.harm_by_case)

    @property
    def utility_positive_case_count(self) -> int:
        return self._positive_case_count(self.utility_by_case)

    def to_support(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_passed": self.n_passed,
            "harm_count": self.harm_count,
            "utility_positive_count": self.utility_positive_count,
            "n_unique_cases": self.n_unique_cases,
            "n_passed_unique_cases": self.n_passed_unique_cases,
            "harm_case_count": self.harm_case_count,
            "utility_positive_case_count": self.utility_positive_case_count,
            "mean_utility_delta_vs_raw": self.mean_utility_delta_vs_raw,
            "mean_harm_delta_vs_raw": self.mean_harm_delta_vs_raw,
            "mean_case_utility_delta_vs_raw": self.mean_case_utility_delta_vs_raw,
            "mean_case_harm_delta_vs_raw": self.mean_case_harm_delta_vs_raw,
        }

@dataclass
class DeploymentEvidenceSummary:
    cell_id: str
    task_type: str = "unknown"
    n_records: int = 0
    n_passed: int = 0
    failure_signatures: Counter = field(default_factory=Counter)
    safety_reasons: Counter = field(default_factory=Counter)
    action_stats: dict[str, ActionEvidenceStats] = field(default_factory=dict)


class DeploymentEvidenceMiner:
    """Summarize fast-path EvidenceRecord rows for slow-path proposal mining."""

    def __init__(self, store: EvidenceStore):
        self.store = store

    def summarize_cell(self, cell: str) -> DeploymentEvidenceSummary:
        summary = DeploymentEvidenceSummary(cell_id=cell)
        for record in self.store.query_by_cell(cell):
            self._add_record(summary, record)
        return summary

    def _add_record(self, summary: DeploymentEvidenceSummary, record: EvidenceRecord) -> None:
        vr = record.verification_result or {}
        downstream = vr.get("downstream") or {}
        routing = record.routing or {}
        safety = routing.get("safety") or {}
        task_type = _task_from_record(record, summary.cell_id)
        if summary.task_type in ("", "unknown"):
            summary.task_type = task_type
        elif task_type and task_type != summary.task_type:
            summary.task_type = "mixed"
        passed = bool(vr.get("passed", False))
        action = _selected_action(record) or "unknown"
        stats = summary.action_stats.setdefault(action, ActionEvidenceStats(action))
        stats.add(
            passed=passed,
            utility=_safe_float(downstream.get("utility_delta_vs_raw"), 0.0),
            harm=_safe_float(downstream.get("harm_delta_vs_raw"), 0.0),
            case_id=_case_id_from_record(record),
        )
        summary.n_records += 1
        if passed:
            summary.n_passed += 1
        failure = vr.get("failure_signature")
        if failure:
            summary.failure_signatures[str(failure)] += 1
        for reason in safety.get("reasons") or []:
            summary.safety_reasons[str(reason)] += 1


def suggest_slow_path_proposals(
    summary: DeploymentEvidenceSummary,
    *,
    min_support: int = 2,
    raw_action: str = "v_none",
) -> list[SlowProposal]:
    """Create deterministic proposal candidates from mined deployment evidence."""
    proposals: list[SlowProposal] = []
    scope = f"cell:{summary.cell_id}"
    for action_id, stats in sorted(summary.action_stats.items()):
        support = stats.to_support()
        base_ref = f"{summary.cell_id}:{action_id}"
        if (
            action_id != raw_action
            and stats.utility_positive_case_count >= min_support
            and stats.mean_case_utility_delta_vs_raw > 0.0
            and stats.harm_case_count == 0
        ):
            mean_utility = stats.mean_case_utility_delta_vs_raw
            proposals.append(
                SlowProposal(
                    kind="MemoryWrite",
                    scope=scope,
                    payload={
                        "task": summary.task_type,
                        "pattern_region": summary.cell_id,
                        "action": action_id,
                        "action_id": action_id,
                        "grounded_utility": mean_utility,
                        "utility_ci": [mean_utility, mean_utility],
                        "harm_delta_vs_raw": stats.mean_case_harm_delta_vs_raw,
                        "scope": scope,
                        "source": "deployment_evidence_miner",
                    },
                    evidence_refs=(f"{base_ref}:utility",),
                    support=support,
                    provenance={"source": "deployment_evidence_miner"},
                )
            )
        if action_id != raw_action and stats.harm_case_count >= min_support:
            proposals.append(
                SlowProposal(
                    kind="ProposeRiskRule",
                    scope=scope,
                    payload={
                        "rule_id": f"auto_ban_{_safe_id(summary.cell_id)}_{_safe_id(action_id)}",
                        "when": {"base_action_in": [action_id]},
                        "then": {"op": "ban", "action": raw_action},
                        "scope": scope,
                    },
                    evidence_refs=(f"{base_ref}:harm",),
                    support=support,
                    provenance={"source": "deployment_evidence_miner"},
                )
            )
    return proposals





