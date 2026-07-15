"""BenchmarkMethod implementation for the deterministic vNext minimum vertical slice."""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from ..benchmark.method_api import MethodSeriesView, PreparedSeries, validate_prepared
from ..operators._provenance import get_ledger, start_recording, stop_recording
from ..operators.registry import canonicalize
from ..policy.program_edit import ProgramSpecV1
from ..policy.task_spec import TaskSpec
from ..sandbox import run_pipeline
from ._canonical import sha256
from .grammar import ActionEligibilityManifestV1, CandidateGrammarV1
from .pattern import PatternCard, build_pattern_card
from .protocol import MethodInputContractV1


class MethodTerminalInvalidInput(ValueError):
    """The benchmark-owned method input failed its frozen precondition."""


OBSERVED_PATTERN_ALLOWLIST = frozenset()


class CandidateSupplier(Protocol):
    kind: str

    def supply(
        self, pattern: PatternCard, task_spec: TaskSpec, grammar: CandidateGrammarV1
    ) -> ProgramSpecV1: ...


@dataclass(frozen=True)
class DeterministicSupplier:
    """One-candidate supplier. The default H0 vertical slice is an explicit no-op."""

    action_id: str = "__noop__"
    kind: str = "deterministic_b1"

    def supply(
        self, pattern: PatternCard, task_spec: TaskSpec, grammar: CandidateGrammarV1
    ) -> ProgramSpecV1:
        del pattern
        if self.action_id == "__noop__":
            return grammar.noop(task_spec)
        return grammar.from_action(self.action_id, task_spec)


@dataclass(frozen=True)
class PreparationAudit:
    series_uid: str
    input_sha: str
    pattern_sha: str
    requested_program_sha: str
    effective_operators: tuple[str, ...]
    fallback_stage: str
    execution_ok: bool
    execution_error: str
    operator_ledger: tuple[tuple[str, str, str], ...]

    @property
    def semantic_payload(self) -> dict[str, object]:
        """Method semantics excluding the audit-only series UID."""
        return {
            "input_sha": self.input_sha,
            "pattern_sha": self.pattern_sha,
            "requested_program_sha": self.requested_program_sha,
            "effective_operators": self.effective_operators,
            "fallback_stage": self.fallback_stage,
            "execution_ok": self.execution_ok,
            "execution_error": self.execution_error,
            "operator_ledger": self.operator_ledger,
        }

    @property
    def semantic_sha(self) -> str:
        return sha256(self.semantic_payload)

    @property
    def sha256(self) -> str:
        return sha256(self)


class VNextBenchmarkMethod:
    """Leakage-safe method surface: visible series -> card -> supplier -> registry."""

    _provenance_lock = threading.Lock()

    def __init__(
        self,
        *,
        method_id: str = "vnext_h0",
        eligibility: ActionEligibilityManifestV1 | None = None,
        supplier: CandidateSupplier | None = None,
        input_contract: MethodInputContractV1 | None = None,
    ) -> None:
        if not method_id or method_id != method_id.strip():
            raise ValueError("method_id must be canonical")
        self.method_id = method_id
        self.eligibility = eligibility or ActionEligibilityManifestV1.conservative()
        self.grammar = CandidateGrammarV1(self.eligibility)
        self.supplier = supplier or DeterministicSupplier()
        self.input_contract = input_contract or MethodInputContractV1()
        self._audit: list[PreparationAudit] = []

    @property
    def audit_records(self) -> tuple[PreparationAudit, ...]:
        return tuple(self._audit)

    @staticmethod
    def _input_sha(values: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(values, dtype="<f8").tobytes())
        return digest.hexdigest()

    @staticmethod
    def _effective_operators(trace, ledger) -> tuple[str, ...]:
        remaining = list(ledger)
        effective: list[str] = []
        for row in trace:
            requested = canonicalize(row["canonical"])
            replacement = requested
            for index, event in enumerate(remaining):
                if canonicalize(event["requested"]) == requested:
                    replacement = canonicalize(event["effective"])
                    remaining.pop(index)
                    break
            effective.append(replacement)
        return tuple(effective)

    def _run(self, spec: ProgramSpecV1, values: np.ndarray):
        if not spec.steps:
            return np.asarray(values, dtype=float).copy(), (), True, "", ()
        with self._provenance_lock:
            start_recording()
            try:
                result = run_pipeline(
                    [(op, dict(params)) for op, params in spec.steps],
                    values,
                    source="vnext_candidate_grammar_v1",
                )
                ledger = get_ledger()
            finally:
                stop_recording()
        operators = self._effective_operators(result.trace, ledger)
        events = tuple(
            (str(row["requested"]), str(row["effective"]), str(row.get("reason", "")))
            for row in ledger
        )
        return result.artifact, operators, result.ok, result.error, events

    def prepare(
        self,
        series_view: MethodSeriesView,
        task_spec: TaskSpec,
        observed_pattern_spec: Mapping[str, float],
    ) -> PreparedSeries:
        if not isinstance(series_view, MethodSeriesView) or not isinstance(task_spec, TaskSpec):
            raise TypeError("prepare requires MethodSeriesView and TaskSpec")
        if not isinstance(observed_pattern_spec, Mapping):
            raise TypeError("observed_pattern_spec must be a mapping")
        unexpected = set(observed_pattern_spec) - OBSERVED_PATTERN_ALLOWLIST
        if unexpected:
            raise ValueError(
                "observed_pattern_spec contains non-whitelisted fields: "
                + ",".join(sorted(map(str, unexpected)))
            )
        visible = np.asarray(series_view.degraded_inner_train, dtype=float)
        input_verdict = self.input_contract.validate(visible)
        if not input_verdict.valid:
            raise MethodTerminalInvalidInput(
                f"{self.input_contract.terminal_code}:{input_verdict.code}"
            )
        raw = visible.copy()
        card = build_pattern_card(raw)
        spec = self.supplier.supply(card, task_spec, self.grammar)
        self.grammar.validate(spec, task_spec)

        artifact, operators, ok, error, ledger = self._run(spec, raw)
        fallback_stage = "selected_program"
        candidate = PreparedSeries(
            series_uid=series_view.series_uid,
            values=raw if artifact is None else artifact,
            operators=operators if artifact is not None else (),
            units="original_units",
        )
        verdict = validate_prepared(candidate, expected_length=len(raw))

        if not ok or not verdict.valid:
            recovery = ProgramSpecV1(
                steps=(("impute_linear", ()),), scope=("global",),
                task_type=task_spec.task_type, risk_budget_beta=1.0,
                max_modified_fraction=1.0,
            )
            artifact, operators, ok, recovery_error, recovery_ledger = self._run(recovery, raw)
            error = error or recovery_error or verdict.code
            ledger = ledger + recovery_ledger
            fallback_stage = "conservative_recovery"
            candidate = PreparedSeries(
                series_uid=series_view.series_uid,
                values=raw if artifact is None else artifact,
                operators=operators if artifact is not None else (),
                units="original_units",
            )
            verdict = validate_prepared(candidate, expected_length=len(raw))

        if not ok or not verdict.valid:
            fallback_stage = "raw_identity"
            candidate = PreparedSeries(
                series_uid=series_view.series_uid, values=raw,
                operators=(), units="original_units",
            )
            verdict = validate_prepared(candidate, expected_length=len(raw))
            if not verdict.valid:
                raise ValueError(f"raw identity cannot satisfy Method contract: {verdict.code}")

        self._audit.append(PreparationAudit(
            series_uid=series_view.series_uid,
            input_sha=self._input_sha(raw), pattern_sha=card.sha256,
            requested_program_sha=self.grammar.canonical_program_sha(spec),
            effective_operators=candidate.operators,
            fallback_stage=fallback_stage, execution_ok=ok,
            execution_error=error, operator_ledger=ledger,
        ))
        return candidate
