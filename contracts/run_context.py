from __future__ import annotations

import re
from dataclasses import dataclass

from .canonical import canonical_sha256


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CANONICAL = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/+-]*$")


def _identity(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _CANONICAL.fullmatch(value):
        raise ValueError(f"{field_name} must be a canonical non-empty identity")
    return value


def _sha(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class RunDependencyBinding:
    task_context_sha: str
    evaluator_adapter_id: str
    instrument_epoch: str
    corpus_epoch: str
    capability_bundle_sha: str
    runtime_sha: str
    harness_sha: str
    code_commit: str
    provider_id: str
    model_id: str
    schema_version: str = "run-dependency-binding/1"

    def __post_init__(self) -> None:
        if self.schema_version != "run-dependency-binding/1":
            raise ValueError("unsupported RunDependencyBinding revision")
        _sha(self.task_context_sha, "task_context_sha")
        _sha(self.capability_bundle_sha, "capability_bundle_sha")
        _sha(self.runtime_sha, "runtime_sha")
        _sha(self.harness_sha, "harness_sha")
        if not isinstance(self.code_commit, str) or not _COMMIT.fullmatch(self.code_commit):
            raise ValueError("code_commit must be a lowercase 40-character Git SHA")
        for field_name in (
            "evaluator_adapter_id",
            "instrument_epoch",
            "corpus_epoch",
            "provider_id",
            "model_id",
        ):
            _identity(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_context_sha": self.task_context_sha,
            "evaluator_adapter_id": self.evaluator_adapter_id,
            "instrument_epoch": self.instrument_epoch,
            "corpus_epoch": self.corpus_epoch,
            "capability_bundle_sha": self.capability_bundle_sha,
            "runtime_sha": self.runtime_sha,
            "harness_sha": self.harness_sha,
            "code_commit": self.code_commit,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
        }

    def sha(self) -> str:
        return canonical_sha256(self.to_dict())


__all__ = ["RunDependencyBinding"]
