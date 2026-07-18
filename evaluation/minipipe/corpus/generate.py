from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes, canonical_sha256
from SelfEvolvingHarnessTS.evaluation.minipipe.config import M0Rules
from SelfEvolvingHarnessTS.evaluation.minipipe.contracts import (
    ArtifactRoots,
    CasePurpose,
    PrivateSyntheticCase,
    PublicCaseView,
)

from .injections import build_risk_series, inject_target


@dataclass(frozen=True)
class CoreCorpus:
    targets: tuple[PrivateSyntheticCase, ...]
    risks: tuple[PrivateSyntheticCase, ...]
    rules_sha: str

    @property
    def all_cases(self) -> tuple[PrivateSyntheticCase, ...]:
        return tuple(sorted((*self.targets, *self.risks), key=lambda case: case.case_id))


@dataclass(frozen=True)
class _CaseSpec:
    purpose_group: str
    seed: int
    family: str
    severity: str

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.purpose_group, self.seed, self.family, self.severity)


def _rules_parts(rules: Mapping[str, object]) -> tuple[Mapping[str, object], str]:
    corpus = rules.get("corpus")
    if not isinstance(corpus, Mapping):
        raise ValueError("rules must contain corpus configuration")
    rules_sha = rules.rules_sha if isinstance(rules, M0Rules) else canonical_sha256(rules)
    return corpus, rules_sha


def build_core_corpus(rules: Mapping[str, object]) -> CoreCorpus:
    corpus_rules, rules_sha = _rules_parts(rules)
    if corpus_rules["context_length"] != 192 or corpus_rules["future_length"] != 48:
        raise ValueError("M0 corpus generator requires 192 context and 48 future values")
    seeds = tuple(int(seed) for seed in corpus_rules["base_seeds"])
    families = tuple(str(family) for family in corpus_rules["target_families"])
    severities = tuple(str(severity) for severity in corpus_rules["severities"])
    specs: list[_CaseSpec] = []
    for seed in seeds:
        for family in families:
            for severity in severities:
                specs.append(_CaseSpec("target", seed, family, severity))
            risk_severity = "clean" if family == "missing" else "genuine"
            specs.append(_CaseSpec("risk", seed, family, risk_severity))
    specs.sort(
        key=lambda spec: (
            0 if spec.purpose_group == "target" else 1,
            spec.seed,
            spec.family,
            spec.severity,
        )
    )
    allocation_order = sorted(
        specs,
        key=lambda spec: canonical_sha256(
            {
                "namespace": "m0-opaque-case-allocation/1",
                "purpose_slot": spec.purpose_group,
                "seed": spec.seed,
                "family_slot": spec.family,
                "severity_slot": spec.severity,
            }
        ),
    )
    ids = {
        spec.key: f"m0-{index:04d}"
        for index, spec in enumerate(allocation_order, start=1)
    }
    cases: list[PrivateSyntheticCase] = []
    for spec in specs:
        case_id = ids[spec.key]
        if spec.purpose_group == "target":
            injected = inject_target(spec.seed, spec.family, spec.severity)
            risk_severity = "clean" if spec.family == "missing" else "genuine"
            counterpart_id = ids[("risk", spec.seed, spec.family, risk_severity)]
            case = PrivateSyntheticCase.create(
                case_id=case_id,
                seed=spec.seed,
                purpose=CasePurpose.TARGET,
                private_family=spec.family,
                private_severity=spec.severity,
                clean_context=injected.clean_context,
                corrupt_context=injected.corrupt_context,
                clean_future=injected.clean_future,
                oracle_affected_indices=injected.affected_indices,
                observable_counterpart_id=counterpart_id,
            )
        else:
            context, future, risk_kind = build_risk_series(spec.seed, spec.family)
            purpose = (
                CasePurpose.RISK_CLEAN
                if risk_kind == "clean"
                else CasePurpose.RISK_GENUINE_EVENT
            )
            case = PrivateSyntheticCase.create(
                case_id=case_id,
                seed=spec.seed,
                purpose=purpose,
                private_family=spec.family,
                private_severity=spec.severity,
                clean_context=context,
                corrupt_context=context,
                clean_future=future,
                oracle_affected_indices=(),
                observable_counterpart_id=None,
            )
        cases.append(case)
    targets = tuple(case for case in cases if case.purpose is CasePurpose.TARGET)
    risks = tuple(case for case in cases if case.purpose is not CasePurpose.TARGET)
    if len(targets) != 24 or len(risks) != 12:
        raise ValueError("M0 core corpus must contain 24 targets and 12 risks")
    return CoreCorpus(targets=targets, risks=risks, rules_sha=rules_sha)


def _write_public_case(view: PublicCaseView, root: Path) -> None:
    if not isinstance(view, PublicCaseView):
        raise TypeError("public artifact writer accepts PublicCaseView only")
    (root / f"{view.case_id}.json").write_bytes(
        canonical_json_bytes(view.to_json()) + b"\n"
    )


def _write_private_case(case: PrivateSyntheticCase, root: Path) -> None:
    if not isinstance(case, PrivateSyntheticCase):
        raise TypeError("private artifact writer accepts PrivateSyntheticCase only")
    (root / f"{case.case_id}.json").write_bytes(
        canonical_json_bytes(case.to_private_json()) + b"\n"
    )


def write_case_artifacts(
    cases: Sequence[PrivateSyntheticCase],
    run_root: Path,
) -> ArtifactRoots:
    roots = ArtifactRoots.create(run_root)
    seen: set[str] = set()
    for case in sorted(cases, key=lambda item: item.case_id):
        if not isinstance(case, PrivateSyntheticCase):
            raise TypeError("case artifact source must be PrivateSyntheticCase")
        if case.case_id in seen:
            raise ValueError("duplicate case ID while writing corpus")
        seen.add(case.case_id)
        _write_public_case(case.to_public_view(), roots.public)
        _write_private_case(case, roots.private)
    return roots


__all__ = ["CoreCorpus", "build_core_corpus", "write_case_artifacts"]
