from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_document,
)


_EVENT_KINDS = frozenset(
    {"GENESIS", "EDIT_EVALUATED", "PROMOTED", "REJECTED", "PENDING"}
)


@dataclass(frozen=True)
class LineageEvent:
    schema_version: str
    event_index: int
    event_kind: str
    cycle_id: str
    parent_snapshot_sha: str
    candidate_snapshot_sha: str
    active_snapshot_sha: str
    edit_manifest_sha: str
    paired_replay_report_sha: str
    final_core_regression_sha: str
    verdict: str
    scope_kind: str
    previous_event_sha: str
    event_sha: str
    metadata: Mapping[str, object]

    def payload_without_sha(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "event_index": self.event_index,
            "event_kind": self.event_kind,
            "cycle_id": self.cycle_id,
            "parent_snapshot_sha": self.parent_snapshot_sha,
            "candidate_snapshot_sha": self.candidate_snapshot_sha,
            "active_snapshot_sha": self.active_snapshot_sha,
            "edit_manifest_sha": self.edit_manifest_sha,
            "paired_replay_report_sha": self.paired_replay_report_sha,
            "final_core_regression_sha": self.final_core_regression_sha,
            "verdict": self.verdict,
            "scope_kind": self.scope_kind,
            "previous_event_sha": self.previous_event_sha,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> dict[str, object]:
        return {**self.payload_without_sha(), "event_sha": self.event_sha}


def _event_from_json(value: object) -> LineageEvent:
    if not isinstance(value, dict):
        raise ValueError("lineage row must be an object")
    expected = {
        "schema_version",
        "event_index",
        "event_kind",
        "cycle_id",
        "parent_snapshot_sha",
        "candidate_snapshot_sha",
        "active_snapshot_sha",
        "edit_manifest_sha",
        "paired_replay_report_sha",
        "final_core_regression_sha",
        "verdict",
        "scope_kind",
        "previous_event_sha",
        "event_sha",
        "metadata",
    }
    if set(value) != expected:
        raise ValueError("lineage row fields do not match lineage-event/1")
    metadata = value["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("lineage metadata must be an object")
    event = LineageEvent(
        schema_version=str(value["schema_version"]),
        event_index=int(value["event_index"]),
        event_kind=str(value["event_kind"]),
        cycle_id=str(value["cycle_id"]),
        parent_snapshot_sha=str(value["parent_snapshot_sha"]),
        candidate_snapshot_sha=str(value["candidate_snapshot_sha"]),
        active_snapshot_sha=str(value["active_snapshot_sha"]),
        edit_manifest_sha=str(value["edit_manifest_sha"]),
        paired_replay_report_sha=str(value["paired_replay_report_sha"]),
        final_core_regression_sha=str(value["final_core_regression_sha"]),
        verdict=str(value["verdict"]),
        scope_kind=str(value["scope_kind"]),
        previous_event_sha=str(value["previous_event_sha"]),
        event_sha=str(value["event_sha"]),
        metadata=MappingProxyType(dict(metadata)),
    )
    if event.schema_version != "lineage-event/1":
        raise ValueError("lineage schema version mismatch")
    if event.event_kind not in _EVENT_KINDS:
        raise ValueError("unknown lineage event kind")
    return event


class HarnessLineage:
    """Append-only, timestamp-free scientific lineage for Harness promotion."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events = self._read_events()
        if self._events and not self.verify_hash_chain():
            raise ValueError("broken Harness lineage hash chain")

    def _read_events(self) -> tuple[LineageEvent, ...]:
        if not self.path.is_file():
            return ()
        events: list[LineageEvent] = []
        for line_number, raw in enumerate(self.path.read_bytes().splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                events.append(_event_from_json(parse_json_document(raw)))
            except ValueError as exc:
                raise ValueError(f"invalid lineage row {line_number}: {exc}") from exc
        return tuple(events)

    @property
    def events(self) -> tuple[LineageEvent, ...]:
        return self._events

    @property
    def promotions(self) -> tuple[LineageEvent, ...]:
        return tuple(event for event in self._events if event.event_kind == "PROMOTED")

    def verify_hash_chain(self) -> bool:
        previous = ""
        for index, event in enumerate(self._events):
            if event.event_index != index or event.previous_event_sha != previous:
                return False
            if canonical_sha256(event.payload_without_sha()) != event.event_sha:
                return False
            if index == 0 and event.event_kind != "GENESIS":
                return False
            if index > 0 and event.event_kind == "GENESIS":
                return False
            if event.event_kind == "PROMOTED" and not all(
                (
                    event.parent_snapshot_sha,
                    event.edit_manifest_sha,
                    event.paired_replay_report_sha,
                    event.final_core_regression_sha,
                )
            ):
                return False
            previous = event.event_sha
        return True

    def append(
        self,
        *,
        event_kind: str,
        cycle_id: str,
        parent_snapshot_sha: str = "",
        candidate_snapshot_sha: str = "",
        active_snapshot_sha: str = "",
        edit_manifest_sha: str = "",
        paired_replay_report_sha: str = "",
        final_core_regression_sha: str = "",
        verdict: str = "",
        scope_kind: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> LineageEvent:
        if event_kind not in _EVENT_KINDS:
            raise ValueError("unknown lineage event kind")
        if not self._events and event_kind != "GENESIS":
            raise ValueError("lineage must start with GENESIS")
        if self._events and event_kind == "GENESIS":
            raise ValueError("lineage already has a GENESIS event")
        payload = {
            "schema_version": "lineage-event/1",
            "event_index": len(self._events),
            "event_kind": event_kind,
            "cycle_id": cycle_id,
            "parent_snapshot_sha": parent_snapshot_sha,
            "candidate_snapshot_sha": candidate_snapshot_sha,
            "active_snapshot_sha": active_snapshot_sha,
            "edit_manifest_sha": edit_manifest_sha,
            "paired_replay_report_sha": paired_replay_report_sha,
            "final_core_regression_sha": final_core_regression_sha,
            "verdict": verdict,
            "scope_kind": scope_kind,
            "previous_event_sha": self._events[-1].event_sha if self._events else "",
            "metadata": dict(metadata or {}),
        }
        event = _event_from_json(
            {**payload, "event_sha": canonical_sha256(payload)}
        )
        encoded = canonical_json_bytes(event.to_json()) + b"\n"
        with self.path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self._events = (*self._events, event)
        if not self.verify_hash_chain():
            raise AssertionError("new lineage event broke the hash chain")
        return event


__all__ = ["HarnessLineage", "LineageEvent"]

