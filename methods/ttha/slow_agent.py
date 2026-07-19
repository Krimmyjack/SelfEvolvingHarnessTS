from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import PurePath
from typing import Any, Callable

from SelfEvolvingHarnessTS.contracts.harness import (
    EditManifest,
    EditOperation,
    HarnessSnapshot,
)
from SelfEvolvingHarnessTS.contracts.observables import (
    OBSERVABLE_FEATURES,
    validate_applicability,
)

from .agent_core import (
    AgentRole,
    AgentStageResult,
    StagePostValidationError,
    TTHAAgentCore,
)
from .retrieval import resolve_harness_view


_PRIVATE_KEYS = frozenset(
    {
        "clean",
        "clean_values",
        "injection_type",
        "injection_indices",
        "candidate_j",
        "private_receipt",
        "oracle",
        "confirmed_surface",
    }
)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    return value


def _reject_private_or_path(value: object, *, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _PRIVATE_KEYS:
                raise PermissionError(f"private field is forbidden in slow Agent input: {key}")
            _reject_private_or_path(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_private_or_path(nested, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if value.startswith(("/", "\\\\")) or _WINDOWS_ABSOLUTE.match(value):
            raise PermissionError(f"absolute path is forbidden in slow Agent input at {path}")


def _public_features_from_card(card: Mapping[str, object]) -> dict[str, object]:
    candidates = card.get("observable_signature", card.get("public_features", {}))
    if not isinstance(candidates, Mapping):
        return {}
    return {
        key: value
        for key, value in candidates.items()
        if key in OBSERVABLE_FEATURES
    }


class TTHASlowAgent:
    def __init__(self, core: TTHAAgentCore):
        self.core = core
        self.last_no_proposal_reason: str | None = None
        self.last_stage_result: AgentStageResult | None = None

    @staticmethod
    def _manifest_from_payload(payload: Mapping[str, object]) -> EditManifest:
        manifest = payload["edit_manifest"]
        if not isinstance(manifest, Mapping):
            raise ValueError("edit_manifest must be an object")
        manifest_applicability = manifest.get("observable_applicability")
        if manifest_applicability is not None:
            if not isinstance(manifest_applicability, Mapping):
                raise ValueError("manifest observable_applicability must be an object")
            validate_applicability(manifest_applicability)
        return EditManifest(
            edit_id=manifest["edit_id"],
            base_harness_sha=manifest["base_harness_sha"],
            target_pattern_id=manifest["target_pattern_id"],
            target_surface_id=manifest["target_surface_id"],
            operation=EditOperation(manifest["operation"]),
            surface_precondition=manifest["surface_precondition"],
            dependency_precondition_shas=manifest["dependency_precondition_shas"],
            minimal_patch=manifest.get("minimal_patch"),
            new_value=manifest.get("new_value"),
            observable_applicability=manifest_applicability,
            predicted_agent_behavior_change=tuple(
                manifest["predicted_agent_behavior_change"]
            ),
            predicted_data_effect=tuple(manifest["predicted_data_effect"]),
            automatically_selected_risk_cases=tuple(
                manifest.get("automatically_selected_risk_cases", ())
            ),
            falsification_condition=tuple(manifest["falsification_condition"]),
        )

    def propose_edit(
        self,
        card: Mapping[str, object],
        surface_catalog: Mapping[str, object] | Sequence[Mapping[str, object]],
        snapshot: HarnessSnapshot,
        *,
        manifest_preflight: Callable[[EditManifest], None] | None = None,
        allowed_operator_contracts: Sequence[Mapping[str, object]] = (),
        fixed_probe_contracts: Mapping[str, object] | None = None,
    ) -> EditManifest | None:
        self.last_no_proposal_reason = None
        self.last_stage_result = None
        if not isinstance(card, Mapping):
            raise TypeError("FailurePatternCard must be a mapping")
        _reject_private_or_path(card, path="card")
        _reject_private_or_path(surface_catalog, path="surface_catalog")
        _reject_private_or_path(
            allowed_operator_contracts, path="allowed_operator_contracts"
        )
        _reject_private_or_path(
            fixed_probe_contracts or {}, path="fixed_probe_contracts"
        )
        applicability = card.get("observable_applicability")
        if applicability is not None:
            if not isinstance(applicability, Mapping):
                raise ValueError("card observable_applicability must be an object")
            validate_applicability(applicability)
        public_features = _public_features_from_card(card)
        view = resolve_harness_view(
            snapshot,
            public_features,
            role="slow",
        )
        pattern_id = card.get("pattern_id", "pattern-unknown")
        if not isinstance(pattern_id, str):
            raise ValueError("card pattern_id must be a string")
        existing_inventory = [
            {"entry_id": skill.skill_id, "entry_kind": skill.skill_kind.value}
            for skill in snapshot.skills
        ] + [
            {"entry_id": memory.memory_id, "entry_kind": "memory"}
            for memory in snapshot.memories
        ]

        def post_validate(payload: Mapping[str, object]) -> None:
            try:
                proposed = self._manifest_from_payload(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise StagePostValidationError(
                    "MANIFEST_CONSTRUCTION_INVALID",
                    "The schema-valid payload cannot be constructed as one EditManifest.",
                    retryable=True,
                ) from exc
            if manifest_preflight is not None:
                manifest_preflight(proposed)

        stage = self.core.run_stage(
            role=AgentRole.SLOW,
            stage="edit",
            case_id=pattern_id,
            public_input={
                "failure_pattern_card": _plain(card),
                "writable_surface_catalog": _plain(surface_catalog),
                "base_harness_sha": snapshot.harness_content_sha,
                "dependency_precondition_shas": _plain(snapshot.dependency_shas),
                "existing_entry_inventory": existing_inventory,
                "allowed_operator_contracts": _plain(allowed_operator_contracts),
                "fixed_probe_contracts": _plain(fixed_probe_contracts or {}),
                "add_rule": (
                    "ADD creates one new capability or memory entry. Never target an "
                    "entry_id listed in existing_entry_inventory. PATCH an existing "
                    "authorized surface instead."
                ),
            },
            harness_view=view,
            output_schema_name="slow_edit_v1",
            output_schema=self.core.load_stage_schema("slow_edit_v1"),
            source_snapshot_sha=snapshot.runtime_bundle_sha,
            validation_retries=1,
            post_validator=post_validate,
        )
        self.last_stage_result = stage
        if stage.no_proposal_reason is not None:
            self.last_no_proposal_reason = stage.no_proposal_reason
            return None
        return self._manifest_from_payload(stage.payload)


__all__ = ["TTHASlowAgent"]
