from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_document_bytes,
    canonical_sha256,
    parse_json_document,
)


_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "utility_tolerance",
        "critic_damage_min",
        "candidate_gain_min",
        "selection_regret_min",
        "risk_epsilon",
        "localization_fail_iou_max",
        "localization_pass_iou_min",
        "probe_gain_min",
        "probe_margin_min",
        "target_recovery_fraction",
        "target_median_gain_min",
        "max_edits_per_cycle",
        "max_promotions_per_cycle",
        "candidate_pool_size",
        "agent_program_slots",
        "infrastructure_retries",
        "probe_betas",
        "public_probe_origins",
        "public_probe_horizon",
        "public_probe_min_finite_targets",
        "public_probe_round_decimals",
        "corpus",
    }
)
_CORPUS_KEYS = frozenset(
    {
        "base_seeds",
        "context_length",
        "future_length",
        "severities",
        "target_families",
    }
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(nested) for key, nested in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(nested) for nested in value)
    return value


@dataclass(frozen=True)
class M0Rules(Mapping[str, object]):
    data: Mapping[str, object]
    rules_sha: str

    def __getitem__(self, key: str) -> object:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


def load_m0_rules(path: Path) -> M0Rules:
    path = Path(path)
    value = parse_json_document(canonical_json_document_bytes(path.read_bytes()))
    if not isinstance(value, dict):
        raise ValueError("M0 rules must be a JSON object")
    missing = _TOP_LEVEL_KEYS - set(value)
    extra = set(value) - _TOP_LEVEL_KEYS
    if missing or extra:
        raise ValueError(
            f"M0 rules top-level keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if value["schema_version"] != "m0-rules/1":
        raise ValueError("M0 rules schema_version must be m0-rules/1")
    corpus = value["corpus"]
    if not isinstance(corpus, dict) or set(corpus) != _CORPUS_KEYS:
        raise ValueError("M0 corpus rules keys mismatch")
    if corpus["context_length"] != 192 or corpus["future_length"] != 48:
        raise ValueError("M0 corpus lengths are frozen at 192 + 48")
    if tuple(corpus["severities"]) != ("mild", "severe"):
        raise ValueError("M0 severities must be mild and severe")
    if value["candidate_pool_size"] != 1 + value["agent_program_slots"]:
        raise ValueError("candidate pool size must include identity plus Agent slots")
    if not 0 < value["target_recovery_fraction"] <= 1:
        raise ValueError("target_recovery_fraction must lie in (0, 1]")
    return M0Rules(_freeze(value), canonical_sha256(value))


__all__ = ["M0Rules", "load_m0_rules"]
