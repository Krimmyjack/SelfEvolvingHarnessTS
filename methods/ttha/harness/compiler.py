from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import (
    CANONICALIZATION_VERSION,
    canonical_json_bytes,
    canonical_json_document_bytes,
    canonical_sha256,
    canonical_text_bytes,
    parse_json_document,
)
from SelfEvolvingHarnessTS.contracts.harness import (
    HarnessSnapshot,
    MemoryEntry,
    SkillEntry,
    SkillKind,
    load_learned_skill_entry,
    load_memory_entry,
    load_skill_entry,
)
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA


COMPILER_VERSION = "ttha-harness-compiler/1"
RETRIEVAL_COMPILER_VERSION = "ttha-retrieval-index/1"
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_REQUIRED_BOOTSTRAP_IDS = frozenset(
    {
        "inspect_and_localize",
        "build_contrastive_candidates",
        "select_or_identity_and_verify",
    }
)


@dataclass(frozen=True)
class _CompilationReceipt:
    snapshot: HarnessSnapshot
    snapshot_profile: str
    operator_bundle_sha: str
    canonicalizer_source_sha: str
    compiler_source_sha: str


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    return value


def skill_entry_to_dict(skill: SkillEntry) -> dict[str, Any]:
    return {
        "schema_version": skill.schema_version,
        "skill_id": skill.skill_id,
        "skill_kind": skill.skill_kind.value,
        "revision": skill.revision,
        "body": skill.body,
        "observable_applicability": _plain(skill.observable_applicability),
        "allowed_tools": list(skill.allowed_tools),
        "risk_guards": _plain(skill.risk_guards),
    }


def memory_entry_to_dict(memory: MemoryEntry) -> dict[str, Any]:
    return {
        "schema_version": memory.schema_version,
        "memory_id": memory.memory_id,
        "revision": memory.revision,
        "body": memory.body,
        "observable_applicability": _plain(memory.observable_applicability),
        "risk_guards": _plain(memory.risk_guards),
    }


def snapshot_to_dict(snapshot: HarnessSnapshot) -> dict[str, Any]:
    return {
        "schema_version": snapshot.schema_version,
        "instruction": snapshot.instruction,
        "skills": [skill_entry_to_dict(skill) for skill in snapshot.skills],
        "memories": [memory_entry_to_dict(memory) for memory in snapshot.memories],
        "retrieval": _plain(snapshot.retrieval),
        "candidate_policy": _plain(snapshot.candidate_policy),
        "verification": _plain(snapshot.verification),
        "dependency_shas": _plain(snapshot.dependency_shas),
        "harness_content_sha": snapshot.harness_content_sha,
        "runtime_bundle_sha": snapshot.runtime_bundle_sha,
    }


def _canonical_file_sha(path: Path, *, kind: str) -> str:
    raw = path.read_bytes()
    if kind == "text":
        canonical = canonical_text_bytes(raw)
    elif kind == "json":
        canonical = canonical_json_document_bytes(raw)
    else:
        raise ValueError(f"unknown canonical file kind: {kind}")
    return hashlib.sha256(canonical).hexdigest()


def _load_json(path: Path) -> Any:
    return parse_json_document(canonical_json_document_bytes(path.read_bytes()))


def _load_lock(root: Path) -> dict[str, Any]:
    lock_path = root / "snapshot.lock.json"
    if not lock_path.is_file():
        return {}
    value = _load_json(lock_path)
    if not isinstance(value, dict):
        raise ValueError("snapshot lock must be a JSON object")
    return value


def _load_skills(root: Path) -> tuple[SkillEntry, ...]:
    bootstrap_root = root / "skills" / "bootstrap"
    learned_root = root / "skills" / "learned"
    bootstrap = [
        load_skill_entry(_load_json(path))
        for path in sorted(bootstrap_root.glob("*.json"), key=lambda item: item.as_posix())
    ]
    learned = [
        load_learned_skill_entry(_load_json(path))
        for path in sorted(learned_root.glob("*.json"), key=lambda item: item.as_posix())
    ]
    bootstrap_ids = {skill.skill_id for skill in bootstrap}
    if bootstrap_ids != _REQUIRED_BOOTSTRAP_IDS:
        raise ValueError(
            "bootstrap skill IDs must be exactly " + ", ".join(sorted(_REQUIRED_BOOTSTRAP_IDS))
        )
    if not all(skill.skill_kind is SkillKind.BOOTSTRAP_PROCEDURE for skill in bootstrap):
        raise ValueError("bootstrap directory may contain bootstrap_procedure skills only")
    skills = tuple(sorted((*bootstrap, *learned), key=lambda skill: skill.skill_id))
    skill_ids = [skill.skill_id for skill in skills]
    if len(skill_ids) != len(set(skill_ids)):
        raise ValueError("duplicate skill_id in Harness authoring")
    return skills


def _load_memories(root: Path) -> tuple[MemoryEntry, ...]:
    path = root / "memories.jsonl"
    if not path.is_file():
        raise ValueError("missing memories.jsonl")
    raw = path.read_bytes()
    rows: list[MemoryEntry] = []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            normalized = parse_json_document(
                canonical_json_document_bytes(line.encode("utf-8"))
            )
            rows.append(load_memory_entry(normalized))
        except ValueError as exc:
            raise ValueError(f"invalid memories.jsonl row {line_number}: {exc}") from exc
    rows.sort(key=lambda memory: memory.memory_id)
    ids = [memory.memory_id for memory in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate memory_id in Harness authoring")
    return tuple(rows)


def _operator_bundle_sha() -> tuple[str, str]:
    operator_root = _PACKAGE_ROOT / "operators"
    sources = [
        {
            "path": path.relative_to(_PACKAGE_ROOT).as_posix(),
            "semantic_text_sha": _canonical_file_sha(path, kind="text"),
        }
        for path in sorted(operator_root.glob("*.py"), key=lambda item: item.as_posix())
    ]
    registry_sha = canonical_sha256(_plain(OPERATOR_METADATA))
    return canonical_sha256({"sources": sources, "operator_registry_sha": registry_sha}), registry_sha


def _dependency_shas() -> tuple[dict[str, str], str, str, str]:
    contracts_root = _PACKAGE_ROOT / "contracts"
    schema_root = contracts_root / "schemas"
    runtime_root = _PACKAGE_ROOT / "runtime"
    ttha_root = _PACKAGE_ROOT / "methods" / "ttha"
    canonicalizer_source_sha = _canonical_file_sha(contracts_root / "canonical.py", kind="text")
    compiler_source_sha = _canonical_file_sha(Path(__file__), kind="text")
    operator_bundle_sha, operator_registry_sha = _operator_bundle_sha()
    dependencies: dict[str, str] = {
        "canonicalizer_source": canonicalizer_source_sha,
        "compiler_source": compiler_source_sha,
        "operator_bundle": operator_bundle_sha,
        "operator_registry": operator_registry_sha,
        "candidate_contract": _canonical_file_sha(contracts_root / "candidate.py", kind="text"),
        "observable_contract": _canonical_file_sha(contracts_root / "observables.py", kind="text"),
        "public_boundary_contract": _canonical_file_sha(
            contracts_root / "public_boundary.py", kind="text"
        ),
        "surface_registry": _canonical_file_sha(Path(__file__).with_name("harness_surfaces.json"), kind="json"),
    }
    for path in sorted(schema_root.glob("*.json"), key=lambda item: item.name):
        dependencies[f"schema:{path.stem}"] = _canonical_file_sha(path, kind="json")
    for filename in (
        "agent_backend.py",
        "candidate_pool.py",
        "decision_trace.py",
        "executor.py",
        "llm_cache.py",
        "public_features.py",
    ):
        dependencies[f"runtime:{Path(filename).stem}"] = _canonical_file_sha(
            runtime_root / filename,
            kind="text",
        )
    for filename in (
        "agent_core.py",
        "fast_agent.py",
        "method.py",
        "public_tools.py",
        "retrieval.py",
        "schema_contracts.py",
        "slow_agent.py",
    ):
        dependencies[f"ttha:{Path(filename).stem}"] = _canonical_file_sha(
            ttha_root / filename,
            kind="text",
        )
    for path in sorted((ttha_root / "schemas").glob("*.json"), key=lambda item: item.name):
        dependencies[f"agent_schema:{path.stem}"] = _canonical_file_sha(path, kind="json")
    return dependencies, operator_bundle_sha, canonicalizer_source_sha, compiler_source_sha


def _validate_authoring_controls(
    retrieval: object,
    candidate_policy: object,
    verification: object,
) -> None:
    if not isinstance(retrieval, dict) or retrieval.get("schema_version") != "retrieval/1":
        raise ValueError("retrieval.json must use retrieval/1")
    capability = retrieval.get("capability")
    if not isinstance(capability, dict) or capability.get("kind") != "rule_lexical":
        raise ValueError("retrieval capability rule must be rule_lexical")
    top_k = capability.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("retrieval top_k must be a non-negative integer")
    if not isinstance(candidate_policy, dict) or candidate_policy.get("schema_version") != "candidate-policy/1":
        raise ValueError("candidate_policy.json must use candidate-policy/1")
    if candidate_policy.get("identity_slots") != 1:
        raise ValueError("candidate policy must reserve exactly one identity slot")
    total = candidate_policy.get("total_k")
    program_slots = candidate_policy.get("agent_program_slots")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or isinstance(program_slots, bool)
        or not isinstance(program_slots, int)
        or total != 1 + program_slots
    ):
        raise ValueError("candidate total_k must equal identity plus Agent program slots")
    if not isinstance(verification, dict) or verification.get("schema_version") != "verification/1":
        raise ValueError("verification.json must use verification/1")
    if verification.get("identity_unfilterable") is not True:
        raise ValueError("identity must be unfilterable")
    if verification.get("require_explicit_choice") is not True:
        raise ValueError("explicit candidate choice must be required")


def _compile(root: Path) -> _CompilationReceipt:
    root = Path(root).resolve()
    lock = _load_lock(root)
    profile = str(lock.get("snapshot_profile", "evolving"))
    instruction_path = root / "instruction.md"
    if not instruction_path.is_file():
        raise ValueError("missing instruction.md")
    instruction = canonical_text_bytes(instruction_path.read_bytes()).decode("utf-8")
    skills = _load_skills(root)
    memories = _load_memories(root)
    if profile == "h0-domain-naive":
        if memories:
            raise ValueError("H0 must have empty memory")
        if any(skill.skill_kind is not SkillKind.BOOTSTRAP_PROCEDURE for skill in skills):
            raise ValueError("H0 capability library must be empty")
    retrieval = _load_json(root / "retrieval.json")
    candidate_policy = _load_json(root / "candidate_policy.json")
    verification = _load_json(root / "verification.json")
    _validate_authoring_controls(retrieval, candidate_policy, verification)
    resolved_retrieval = {
        **retrieval,
        "resolved_skill_index": [
            {
                "skill_id": skill.skill_id,
                "skill_kind": skill.skill_kind.value,
                "revision": skill.revision,
            }
            for skill in skills
        ],
        "resolved_memory_index": [
            {"memory_id": memory.memory_id, "revision": memory.revision}
            for memory in memories
        ],
    }
    content = {
        "schema_version": "harness-content/1",
        "instruction": instruction,
        "skills": [skill_entry_to_dict(skill) for skill in skills],
        "memories": [memory_entry_to_dict(memory) for memory in memories],
        "retrieval": resolved_retrieval,
        "candidate_policy": candidate_policy,
        "verification": verification,
    }
    harness_content_sha = canonical_sha256(content)
    dependencies, operator_bundle_sha, canonicalizer_source_sha, compiler_source_sha = _dependency_shas()
    runtime_bundle_sha = canonical_sha256(
        {
            "schema_version": "runtime-bundle/1",
            "harness_content_sha": harness_content_sha,
            "operator_bundle_sha": operator_bundle_sha,
            "dependency_shas": dependencies,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "compiler_version": COMPILER_VERSION,
            "retrieval_compiler_version": RETRIEVAL_COMPILER_VERSION,
        }
    )
    snapshot = HarnessSnapshot(
        schema_version="harness-snapshot/1",
        instruction=instruction,
        skills=skills,
        memories=memories,
        retrieval=resolved_retrieval,
        candidate_policy=candidate_policy,
        verification=verification,
        dependency_shas=dependencies,
        harness_content_sha=harness_content_sha,
        runtime_bundle_sha=runtime_bundle_sha,
    )
    return _CompilationReceipt(
        snapshot=snapshot,
        snapshot_profile=profile,
        operator_bundle_sha=operator_bundle_sha,
        canonicalizer_source_sha=canonicalizer_source_sha,
        compiler_source_sha=compiler_source_sha,
    )


def _lock_payload(receipt: _CompilationReceipt) -> dict[str, Any]:
    snapshot = receipt.snapshot
    return {
        "schema_version": "snapshot-lock/1",
        "snapshot_profile": receipt.snapshot_profile,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "canonicalizer_source_sha": receipt.canonicalizer_source_sha,
        "compiler_version": COMPILER_VERSION,
        "compiler_source_sha": receipt.compiler_source_sha,
        "retrieval_compiler_version": RETRIEVAL_COMPILER_VERSION,
        "harness_content_sha": snapshot.harness_content_sha,
        "runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "operator_bundle_sha": receipt.operator_bundle_sha,
        "dependency_shas": _plain(snapshot.dependency_shas),
    }


def compile_snapshot(root: Path, verify_lock: bool = True) -> HarnessSnapshot:
    root = Path(root).resolve()
    receipt = _compile(root)
    if verify_lock:
        actual = _load_lock(root)
        expected = _lock_payload(receipt)
        if actual != expected:
            raise ValueError("snapshot lock mismatch; run compiler with --write-lock")
    return receipt.snapshot


def write_lock(root: Path) -> Path:
    root = Path(root).resolve()
    receipt = _compile(root)
    path = root / "snapshot.lock.json"
    path.write_bytes(canonical_json_bytes(_lock_payload(receipt)) + b"\n")
    return path


def _main() -> int:
    parser = argparse.ArgumentParser(description="Compile a TTHA Harness snapshot")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()
    if args.write_lock:
        write_lock(args.root)
    snapshot = compile_snapshot(args.root)
    print(f"harness_content_sha={snapshot.harness_content_sha}")
    print(f"runtime_bundle_sha={snapshot.runtime_bundle_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "COMPILER_VERSION",
    "RETRIEVAL_COMPILER_VERSION",
    "compile_snapshot",
    "memory_entry_to_dict",
    "skill_entry_to_dict",
    "snapshot_to_dict",
    "write_lock",
]
