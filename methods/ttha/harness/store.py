from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import (
    CANONICALIZATION_VERSION,
    canonical_json_bytes,
    canonical_text_bytes,
)
from SelfEvolvingHarnessTS.contracts.harness import HarnessSnapshot, SkillKind

from .compiler import (
    COMPILER_VERSION,
    RETRIEVAL_COMPILER_VERSION,
    memory_entry_to_dict,
    skill_entry_to_dict,
    snapshot_to_dict,
)


@dataclass(frozen=True)
class MaterializedSnapshot:
    root: Path
    snapshot: HarnessSnapshot
    parent_runtime_bundle_sha: str | None

    @property
    def harness_content_sha(self) -> str:
        return self.snapshot.harness_content_sha

    @property
    def runtime_bundle_sha(self) -> str:
        return self.snapshot.runtime_bundle_sha


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    }


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_path = self.root.parent / "active.json"

    def _write_snapshot_tree(self, root: Path, snapshot: HarnessSnapshot) -> None:
        (root / "skills" / "bootstrap").mkdir(parents=True, exist_ok=True)
        (root / "skills" / "learned").mkdir(parents=True, exist_ok=True)
        (root / "instruction.md").write_bytes(
            canonical_text_bytes(snapshot.instruction.encode("utf-8"))
        )
        for skill in snapshot.skills:
            directory = (
                "bootstrap"
                if skill.skill_kind in {SkillKind.BOOTSTRAP_PROCEDURE, SkillKind.SAFETY}
                else "learned"
            )
            _write_json(
                root / "skills" / directory / f"{skill.skill_id}.json",
                skill_entry_to_dict(skill),
            )
        if not any(skill.skill_kind is SkillKind.CAPABILITY for skill in snapshot.skills):
            (root / "skills" / "learned" / ".gitkeep").write_bytes(b"")
        memory_rows = [
            canonical_json_bytes(memory_entry_to_dict(memory)) for memory in snapshot.memories
        ]
        (root / "memories.jsonl").write_bytes(
            b"\n".join(memory_rows) + (b"\n" if memory_rows else b"")
        )
        retrieval = _plain(snapshot.retrieval)
        retrieval.pop("resolved_skill_index", None)
        retrieval.pop("resolved_memory_index", None)
        _write_json(root / "retrieval.json", retrieval)
        _write_json(root / "candidate_policy.json", _plain(snapshot.candidate_policy))
        _write_json(root / "verification.json", _plain(snapshot.verification))
        _write_json(root / "resolved.snapshot.json", snapshot_to_dict(snapshot))
        dependencies = _plain(snapshot.dependency_shas)
        lock = {
            "schema_version": "snapshot-lock/1",
            "snapshot_profile": "materialized",
            "canonicalization_version": CANONICALIZATION_VERSION,
            "canonicalizer_source_sha": dependencies["canonicalizer_source"],
            "compiler_version": COMPILER_VERSION,
            "compiler_source_sha": dependencies["compiler_source"],
            "retrieval_compiler_version": RETRIEVAL_COMPILER_VERSION,
            "harness_content_sha": snapshot.harness_content_sha,
            "runtime_bundle_sha": snapshot.runtime_bundle_sha,
            "operator_bundle_sha": dependencies["operator_bundle"],
            "dependency_shas": dependencies,
        }
        _write_json(root / "snapshot.lock.json", lock)

    def _write_provenance(
        self,
        snapshot: HarnessSnapshot,
        parent_runtime_bundle_sha: str | None,
    ) -> None:
        parent_id = parent_runtime_bundle_sha or "root"
        path = (
            self.root.parent
            / "harness_snapshot_provenance"
            / snapshot.runtime_bundle_sha
            / f"{parent_id}.json"
        )
        payload = {
            "schema_version": "snapshot-provenance/1",
            "runtime_bundle_sha": snapshot.runtime_bundle_sha,
            "harness_content_sha": snapshot.harness_content_sha,
            "parent_runtime_bundle_sha": parent_runtime_bundle_sha,
        }
        expected = canonical_json_bytes(payload) + b"\n"
        if path.exists():
            if path.read_bytes() != expected:
                raise ValueError("snapshot provenance collision with different bytes")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)

    def materialize(
        self,
        snapshot: HarnessSnapshot,
        parent_sha: str | None = None,
    ) -> MaterializedSnapshot:
        destination = self.root / snapshot.runtime_bundle_sha
        temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=self.root))
        try:
            self._write_snapshot_tree(temporary, snapshot)
            if destination.exists():
                if _tree_bytes(destination) != _tree_bytes(temporary):
                    raise ValueError("snapshot directory collision with different bytes")
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, destination)
            self._write_provenance(snapshot, parent_sha)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return MaterializedSnapshot(destination, snapshot, parent_sha)

    def fork(self, parent: MaterializedSnapshot, edit_id: str) -> Path:
        if not isinstance(parent, MaterializedSnapshot):
            raise TypeError("snapshot fork requires a MaterializedSnapshot parent")
        if parent.root.resolve() != (self.root / parent.runtime_bundle_sha).resolve():
            raise ValueError("parent snapshot does not belong to this store")
        forks_root = self.root.parent / ".harness_forks"
        forks_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f"{edit_id}-", dir=forks_root))
        try:
            shutil.copytree(parent.root, temporary, dirs_exist_ok=True)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return temporary

    def discard_fork(self, root: Path) -> None:
        root = Path(root).resolve()
        forks_root = (self.root.parent / ".harness_forks").resolve()
        if not root.is_relative_to(forks_root) or root == forks_root:
            raise ValueError("refusing to discard a path outside the controlled fork root")
        if root.exists():
            shutil.rmtree(root)

    def set_active(self, runtime_bundle_sha: str) -> None:
        if not (self.root / runtime_bundle_sha).is_dir():
            raise ValueError("cannot activate an unmaterialized runtime bundle")
        payload = canonical_json_bytes({"runtime_bundle_sha": runtime_bundle_sha}) + b"\n"
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=".active-", suffix=".json", dir=self.active_path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.active_path)
        finally:
            if temporary.exists():
                temporary.unlink()


__all__ = ["MaterializedSnapshot", "SnapshotStore"]
