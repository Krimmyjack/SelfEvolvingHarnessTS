from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    canonical_text_bytes,
    parse_json_document,
)
from SelfEvolvingHarnessTS.contracts.harness import (
    EditManifest,
    EditOperation,
    MemoryEntry,
    SkillEntry,
    SkillKind,
    load_learned_skill_entry,
    load_memory_entry,
    load_skill_entry,
)
from SelfEvolvingHarnessTS.contracts.public_boundary import assert_public_payload
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import FaultRouter
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
    compile_snapshot,
    memory_entry_to_dict,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (
    MaterializedSnapshot,
    SnapshotStore,
)
from SelfEvolvingHarnessTS.methods.ttha.schema_contracts import (
    LocalSchemaError,
    load_stage_schema,
    validate_local_schema,
)
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES


_SURFACE_PATH = (
    Path(__file__).resolve().parents[3]
    / "methods"
    / "ttha"
    / "harness"
    / "harness_surfaces.json"
)
_CANONICAL_ID = r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*"
_FORBIDDEN_TEXT_TERMS = (
    "clean_future",
    "clean_context",
    "oracle_affected",
    "injection_type",
    "candidate_utilities",
    "selection_regret",
    "loss_j",
    "utility_u",
    "r_private",
    "private_receipt",
)


class EditControllerError(RuntimeError):
    """Base class for deterministic edit-controller failures."""


class StaleEditError(EditControllerError):
    """The edit was replayed against a stale content or dependency precondition."""


class AddTargetExistsError(StaleEditError):
    """A public ADD identifier collided with an existing Harness entry."""


class EditAuthorizationError(EditControllerError):
    """The attributed cause does not authorize the requested Harness surface."""


class EditShapeError(EditControllerError):
    """A proposed edit violates the public, static slow-edit contract."""


class PrivateBoundaryViolation(EditControllerError):
    """A deployable edit attempted to encode private or path-derived evidence."""


class EditContextError(EditControllerError):
    """A public catalog or dependency declaration is inconsistent."""


class NoOpEditError(EditContextError):
    """A PATCH would leave the declared Harness surface unchanged."""


@dataclass(frozen=True)
class SurfaceDefinition:
    surface_template_id: str
    owner: str
    path_template: str | None
    json_pointer: str | None
    target_class: str
    surface_type: str
    allowed_operations: tuple[str, ...]
    precondition: str
    value_schema: str | None
    atomic: bool
    allowed_skill_kinds: tuple[str, ...]
    mutually_exclusive_with: tuple[str, ...]
    derived_outputs: tuple[str, ...]
    required_dependency_keys: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedSurface:
    definition: SurfaceDefinition
    surface_id: str
    parameters: Mapping[str, str]

    def owner_path(self) -> str:
        template = self.definition.path_template or self.definition.owner
        return template.format(**self.parameters)


class SurfaceRegistry:
    def __init__(self, path: Path = _SURFACE_PATH) -> None:
        value = parse_json_document(path.read_bytes())
        if not isinstance(value, dict) or value.get("schema_version") != "harness-surfaces/2":
            raise ValueError("surface registry must use harness-surfaces/2")
        raw_surfaces = value.get("surfaces")
        if not isinstance(raw_surfaces, list):
            raise ValueError("surface registry requires a surface list")
        definitions: list[SurfaceDefinition] = []
        seen_templates: set[str] = set()
        seen_owners: set[tuple[str, str | None]] = set()
        for raw in raw_surfaces:
            if not isinstance(raw, dict):
                raise ValueError("surface definition must be an object")
            template = str(raw["surface_template_id"])
            if template in seen_templates:
                raise ValueError(f"duplicate surface template: {template}")
            seen_templates.add(template)
            definition = SurfaceDefinition(
                surface_template_id=template,
                owner=str(raw["owner"]),
                path_template=(
                    str(raw["path_template"])
                    if raw.get("path_template") is not None
                    else None
                ),
                json_pointer=(
                    str(raw["json_pointer"])
                    if raw.get("json_pointer") is not None
                    else None
                ),
                target_class=str(raw["target_class"]),
                surface_type=str(raw["surface_type"]),
                allowed_operations=tuple(str(item) for item in raw["allowed_operations"]),
                precondition=str(raw["precondition"]),
                value_schema=(
                    str(raw["value_schema"])
                    if raw.get("value_schema") is not None
                    else None
                ),
                atomic=bool(raw["atomic"]),
                allowed_skill_kinds=tuple(
                    str(item) for item in raw.get("allowed_skill_kinds", [])
                ),
                mutually_exclusive_with=tuple(
                    str(item) for item in raw.get("mutually_exclusive_with", [])
                ),
                derived_outputs=tuple(str(item) for item in raw.get("derived_outputs", [])),
                required_dependency_keys=tuple(
                    str(item) for item in raw.get("required_dependency_keys", [])
                ),
            )
            if not definition.atomic:
                raise ValueError("every M0 surface must be atomic")
            if (
                not definition.required_dependency_keys
                or len(definition.required_dependency_keys)
                != len(set(definition.required_dependency_keys))
            ):
                raise ValueError(
                    "every M0 surface must declare unique required dependency keys"
                )
            owner_key = (definition.owner, definition.json_pointer)
            if owner_key in seen_owners:
                raise ValueError(f"overlapping surface owner: {owner_key}")
            seen_owners.add(owner_key)
            definitions.append(definition)
        for definition in definitions:
            for other in definition.mutually_exclusive_with:
                if other not in seen_templates:
                    raise ValueError(f"unknown mutually-exclusive surface: {other}")
                reverse = next(
                    item for item in definitions if item.surface_template_id == other
                )
                if definition.surface_template_id not in reverse.mutually_exclusive_with:
                    raise ValueError("mutually-exclusive surface ownership must be symmetric")
        self.definitions = tuple(definitions)
        self.read_only = tuple(str(item) for item in value.get("read_only", []))

    @staticmethod
    def _pattern(template: str) -> re.Pattern[str]:
        escaped = re.escape(template)
        escaped = escaped.replace(r"\{skill_id\}", rf"(?P<skill_id>{_CANONICAL_ID})")
        escaped = escaped.replace(r"\{memory_id\}", rf"(?P<memory_id>{_CANONICAL_ID})")
        return re.compile(rf"^{escaped}$")

    def resolve(self, surface_id: str) -> ResolvedSurface:
        matches: list[ResolvedSurface] = []
        for definition in self.definitions:
            match = self._pattern(definition.surface_template_id).fullmatch(surface_id)
            if match is not None:
                matches.append(
                    ResolvedSurface(
                        definition=definition,
                        surface_id=surface_id,
                        parameters=MappingProxyType(
                            {key: value for key, value in match.groupdict().items() if value}
                        ),
                    )
                )
        if len(matches) != 1:
            raise ValueError(
                f"target must resolve to exactly one Harness surface; matches={len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True)
class ShapeValidatedEdit:
    manifest: EditManifest
    parsed_entry: SkillEntry | MemoryEntry | None


@dataclass(frozen=True)
class PublicEditValidationFeedback:
    error_code: str
    public_message: str
    retryable: bool


@dataclass(frozen=True)
class ValidatedEdit:
    manifest: EditManifest
    surface: ResolvedSurface
    parsed_entry: SkillEntry | MemoryEntry | None
    skill_kind: str | None


@dataclass(frozen=True)
class AppliedEditReceipt:
    edit_id: str
    target_surface_id: str
    confirmed_cause: str
    parent_harness_content_sha: str
    candidate_harness_content_sha: str
    parent_runtime_bundle_sha: str
    candidate_runtime_bundle_sha: str
    parent_root: Path
    candidate_root: Path
    source_surfaces_changed: tuple[str, ...]
    derived_outputs_changed: tuple[str, ...]
    applied_edit_sha: str
    candidate_snapshot: MaterializedSnapshot


def _json_value(path: Path) -> object:
    return parse_json_document(path.read_bytes())


def _pointer_get(value: object, pointer: str | None) -> object:
    if pointer is None or pointer == "":
        return value
    current = value
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or token not in current:
            raise ValueError(f"JSON pointer does not exist: {pointer}")
        current = current[token]
    return current


def _pointer_set(value: object, pointer: str | None, replacement: object) -> object:
    if pointer is None or pointer == "":
        return replacement
    if not isinstance(value, dict):
        raise ValueError("JSON pointer root must be an object")
    current: dict[str, object] = value
    tokens = pointer.lstrip("/").split("/")
    for raw_token in tokens[:-1]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        nested = current.get(token)
        if not isinstance(nested, dict):
            raise ValueError(f"JSON pointer parent does not exist: {pointer}")
        current = nested
    final = tokens[-1].replace("~1", "/").replace("~0", "~")
    if final not in current:
        raise ValueError(f"JSON pointer target does not exist: {pointer}")
    current[final] = replacement
    return value


def _scan_deployable(value: object) -> None:
    if isinstance(value, Mapping):
        try:
            assert_public_payload(value)
        except ValueError as exc:
            raise ValueError("forbidden deployable field") from exc
        for nested in value.values():
            _scan_deployable(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _scan_deployable(nested)
        return
    if not isinstance(value, str):
        return
    lowered = value.lower()
    if any(term in lowered for term in _FORBIDDEN_TEXT_TERMS):
        raise ValueError("forbidden deployable text references private evidence")
    if re.search(r"\bm0-[0-9]{4}\b", lowered) or re.search(
        r"\bpattern-[0-9a-f]{12}\b", lowered
    ):
        raise ValueError("forbidden deployable text references evaluation provenance")
    if "```" in value or "../" in value or "..\\" in value or re.search(
        r"(?:[a-zA-Z]:\\|/mnt/|/home/)", value
    ):
        raise ValueError("forbidden deployable text contains code or filesystem paths")


def _manifest_shape_payload(manifest: EditManifest) -> dict[str, object]:
    payload: dict[str, object] = {
        "edit_id": manifest.edit_id,
        "base_harness_sha": manifest.base_harness_sha,
        "target_pattern_id": manifest.target_pattern_id,
        "target_surface_id": manifest.target_surface_id,
        "operation": manifest.operation.value,
        "surface_precondition": dict(manifest.surface_precondition),
        "dependency_precondition_shas": dict(
            manifest.dependency_precondition_shas
        ),
        "predicted_agent_behavior_change": list(
            manifest.predicted_agent_behavior_change
        ),
        "predicted_data_effect": list(manifest.predicted_data_effect),
        "automatically_selected_risk_cases": list(
            manifest.automatically_selected_risk_cases
        ),
        "falsification_condition": list(manifest.falsification_condition),
    }
    if manifest.minimal_patch is not None:
        payload["minimal_patch"] = dict(manifest.minimal_patch)
    if manifest.new_value is not None:
        payload["new_value"] = dict(manifest.new_value)
    if manifest.observable_applicability is not None:
        payload["observable_applicability"] = dict(
            manifest.observable_applicability
        )
    return payload


class EditController:
    def __init__(
        self,
        store: SnapshotStore,
        *,
        surfaces: SurfaceRegistry | None = None,
        router: FaultRouter | None = None,
    ) -> None:
        self.store = store
        self.surfaces = surfaces or SurfaceRegistry()
        self.router = router or FaultRouter()

    @staticmethod
    def tree_digest(root: Path) -> str:
        digest = hashlib.sha256()
        root = Path(root).resolve()
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            payload = path.read_bytes()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    def _surface_value(
        self,
        parent: MaterializedSnapshot,
        surface: ResolvedSurface,
    ) -> object:
        definition = surface.definition
        if definition.allowed_operations == ("ADD",):
            target_id = surface.parameters.get("skill_id") or surface.parameters.get("memory_id")
            if target_id is None:
                raise ValueError("dynamic ADD surface has no entry ID")
            if definition.value_schema == "skill-entry/1":
                if any(
                    skill.skill_id == target_id for skill in parent.snapshot.skills
                ):
                    raise AddTargetExistsError(
                        f"ADD target already exists: {surface.surface_id}"
                    )
            elif definition.value_schema == "memory-entry/1":
                if any(memory.memory_id == target_id for memory in parent.snapshot.memories):
                    raise AddTargetExistsError(
                        f"ADD target already exists: {surface.surface_id}"
                    )
            return None
        path = parent.root / surface.owner_path()
        if definition.surface_type == "text" and path.suffix == ".md":
            return canonical_text_bytes(path.read_bytes()).decode("utf-8")
        value = _json_value(path)
        return _pointer_get(value, definition.json_pointer)

    def surface_precondition_sha(
        self,
        parent: MaterializedSnapshot,
        target_surface_id: str,
    ) -> str:
        surface = self.surfaces.resolve(target_surface_id)
        value = self._surface_value(parent, surface)
        if value is None:
            raise ValueError("ABSENT surface has no content SHA")
        return canonical_sha256(value)

    def validate_shape(self, manifest: EditManifest) -> ShapeValidatedEdit:
        """Validate only immutable, public edit syntax and deployable content.

        This is deliberately the same resolved schema shown to the slow Agent.
        Dynamic authorization, dependency, and stale-state checks belong to
        :meth:`validate_context` and are not duplicated here.
        """
        if not isinstance(manifest, EditManifest):
            raise TypeError("edit shape validation requires an EditManifest")
        payload = _manifest_shape_payload(manifest)
        try:
            validate_local_schema(
                {"edit_manifest": payload},
                load_stage_schema("slow_edit_v1"),
                path="slow_edit",
            )
        except LocalSchemaError as exc:
            raise EditShapeError(str(exc)) from exc

        deployable = (
            manifest.new_value
            if manifest.new_value is not None
            else manifest.minimal_patch
        )
        try:
            _scan_deployable(deployable)
        except ValueError as exc:
            raise PrivateBoundaryViolation(
                "deployable edit violates the public boundary"
            ) from exc

        parsed_entry: SkillEntry | MemoryEntry | None = None
        if manifest.new_value is not None:
            schema_version = manifest.new_value.get("schema_version")
            try:
                if schema_version == "skill-entry/1":
                    parsed_entry = load_learned_skill_entry(manifest.new_value)
                    for operator_id in parsed_entry.allowed_tools:
                        metadata = OPERATOR_METADATA.get(operator_id)
                        if (
                            operator_id not in OPERATOR_NAMES
                            or metadata is None
                            or metadata.get("deprecated") is True
                            or "forecast" not in metadata["allowed_tasks"]
                        ):
                            raise ValueError(
                                f"SkillEntry tool is not forecast-compatible: {operator_id}"
                            )
                elif schema_version == "memory-entry/1":
                    parsed_entry = load_memory_entry(manifest.new_value)
                else:
                    raise ValueError("ADD new_value has an unknown deployable schema")
            except ValueError as exc:
                raise EditShapeError(str(exc)) from exc
            if manifest.observable_applicability is None or dict(
                manifest.observable_applicability
            ) != dict(parsed_entry.observable_applicability):
                raise EditShapeError(
                    "manifest and deployable entry applicability must match"
                )
        return ShapeValidatedEdit(manifest=manifest, parsed_entry=parsed_entry)

    def validate_context(
        self,
        parent: MaterializedSnapshot,
        shaped: ShapeValidatedEdit,
        *,
        confirmed_cause: str,
    ) -> ValidatedEdit:
        if not isinstance(parent, MaterializedSnapshot):
            raise TypeError("edit context validation requires a MaterializedSnapshot")
        if not isinstance(shaped, ShapeValidatedEdit):
            raise TypeError("edit context validation requires a ShapeValidatedEdit")
        manifest = shaped.manifest
        if manifest.base_harness_sha != parent.harness_content_sha:
            raise StaleEditError("base_harness_sha does not match the active parent")
        try:
            surface = self.surfaces.resolve(manifest.target_surface_id)
        except ValueError as exc:
            raise EditAuthorizationError(
                "target_surface_id is not an authorized M0 surface"
            ) from exc
        definition = surface.definition
        if manifest.operation.value not in definition.allowed_operations:
            raise EditContextError(
                "operation does not match the public writable-surface catalog"
            )
        if manifest.surface_precondition.get("kind") != definition.precondition:
            raise EditContextError(
                "surface precondition does not match the public catalog"
            )

        parsed_entry = shaped.parsed_entry
        skill_kind: str | None = None
        if definition.value_schema == "skill-entry/1":
            if not isinstance(parsed_entry, SkillEntry):
                raise EditShapeError("SkillEntry surface requires a SkillEntry new_value")
            expected_id = surface.parameters.get("skill_id")
            if parsed_entry.skill_id != expected_id:
                raise EditShapeError("SkillEntry ID does not match the target surface")
            skill_kind = parsed_entry.skill_kind.value
        elif definition.value_schema == "memory-entry/1":
            if not isinstance(parsed_entry, MemoryEntry):
                raise EditShapeError("MemoryEntry surface requires a MemoryEntry new_value")
            expected_id = surface.parameters.get("memory_id")
            if parsed_entry.memory_id != expected_id:
                raise EditShapeError("MemoryEntry ID does not match the target surface")
        elif parsed_entry is not None:
            raise EditShapeError("PATCH surface cannot carry a deployable entry")
        elif definition.allowed_skill_kinds:
            if definition.target_class == "bootstrap_procedure":
                skill_kind = "bootstrap_procedure"
            else:
                path = parent.root / surface.owner_path()
                value = _json_value(path)
                skill = load_skill_entry(value)
                skill_kind = skill.skill_kind.value

        try:
            self.router.authorize(
                confirmed_cause,
                target_class=definition.target_class,
                operation=manifest.operation.value,
                skill_kind=skill_kind,
                target_surface_id=manifest.target_surface_id,
            )
        except (KeyError, ValueError) as exc:
            raise EditAuthorizationError(
                f"{confirmed_cause} does not authorize {manifest.target_surface_id}"
            ) from exc

        declared_dependencies = set(manifest.dependency_precondition_shas)
        required_dependencies = set(definition.required_dependency_keys)
        missing_dependencies = sorted(required_dependencies - declared_dependencies)
        extra_dependencies = sorted(declared_dependencies - required_dependencies)
        if missing_dependencies:
            raise EditContextError(
                "missing required dependency preconditions: "
                + ", ".join(missing_dependencies)
            )
        if extra_dependencies:
            raise EditContextError(
                "unexpected dependency preconditions: "
                + ", ".join(extra_dependencies)
            )
        for name, expected in manifest.dependency_precondition_shas.items():
            actual = parent.snapshot.dependency_shas.get(name)
            if actual is None or actual != expected:
                raise StaleEditError(f"dependency precondition is stale: {name}")
        current = self._surface_value(parent, surface)
        if definition.precondition == "SHA":
            expected_sha = str(manifest.surface_precondition["sha"])
            if current is None or canonical_sha256(current) != expected_sha:
                raise StaleEditError(f"surface precondition is stale: {surface.surface_id}")
            replacement = dict(manifest.minimal_patch or {})["value"]
            if current == replacement:
                raise NoOpEditError(
                    f"PATCH does not change the surface: {surface.surface_id}"
                )
        elif current is not None:
            raise StaleEditError(f"ADD surface is no longer absent: {surface.surface_id}")

        return ValidatedEdit(
            manifest=manifest,
            surface=surface,
            parsed_entry=parsed_entry,
            skill_kind=skill_kind,
        )

    @staticmethod
    def public_feedback_for_error(
        exc: Exception,
    ) -> PublicEditValidationFeedback:
        """Map controller failures to a non-leaking retry contract.

        Messages are deliberately static. Private-wall and authorization failures
        never reveal which hidden field or unavailable surface caused rejection.
        """
        if isinstance(exc, AddTargetExistsError):
            return PublicEditValidationFeedback(
                "ADD_TARGET_EXISTS",
                "The requested new entry ID already exists; choose a fresh canonical ID.",
                True,
            )
        if isinstance(exc, NoOpEditError):
            return PublicEditValidationFeedback(
                "NO_OP_EDIT",
                "The PATCH value is already active; propose a materially different authorized edit or abstain.",
                True,
            )
        if isinstance(exc, EditShapeError):
            return PublicEditValidationFeedback(
                "EDIT_SHAPE_INVALID",
                "The manifest violates a static cross-field edit requirement.",
                True,
            )
        if isinstance(exc, EditContextError):
            return PublicEditValidationFeedback(
                "PUBLIC_CONTEXT_MISMATCH",
                "Copy the operation, precondition, and dependency block exactly from one public catalog entry.",
                True,
            )
        if isinstance(exc, PrivateBoundaryViolation):
            return PublicEditValidationFeedback(
                "PRIVATE_BOUNDARY_VIOLATION",
                "The proposal violates the deployment-observable information boundary.",
                False,
            )
        if isinstance(exc, EditAuthorizationError):
            return PublicEditValidationFeedback(
                "UNAUTHORIZED_SURFACE",
                "The attributed fault does not authorize this edit.",
                False,
            )
        if isinstance(exc, StaleEditError):
            return PublicEditValidationFeedback(
                "STALE_PRECONDITION",
                "The edit precondition is stale and requires a new optimization pass.",
                False,
            )
        return PublicEditValidationFeedback(
            "INFRASTRUCTURE_ERROR",
            "The edit could not be validated because of a non-correctable controller error.",
            False,
        )

    def validate(
        self,
        parent: MaterializedSnapshot,
        manifest: EditManifest,
        *,
        confirmed_cause: str,
    ) -> ValidatedEdit:
        shaped = self.validate_shape(manifest)
        return self.validate_context(
            parent,
            shaped,
            confirmed_cause=confirmed_cause,
        )

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(value) + b"\n")

    def _apply(self, fork_root: Path, validated: ValidatedEdit) -> None:
        manifest = validated.manifest
        surface = validated.surface
        definition = surface.definition
        if manifest.operation is EditOperation.ADD:
            if isinstance(validated.parsed_entry, SkillEntry):
                path = fork_root / surface.owner_path()
                if path.exists():
                    raise StaleEditError("SkillEntry ADD target exists in the fork")
                self._write_json(path, dict(manifest.new_value or {}))
                return
            if isinstance(validated.parsed_entry, MemoryEntry):
                memories = [
                    memory_entry_to_dict(memory)
                    for memory in compile_snapshot(fork_root, verify_lock=False).memories
                ]
                memories.append(memory_entry_to_dict(validated.parsed_entry))
                memories.sort(key=lambda value: str(value["memory_id"]))
                path = fork_root / definition.owner
                path.write_bytes(
                    b"\n".join(canonical_json_bytes(memory) for memory in memories) + b"\n"
                )
                return
            raise ValueError("unsupported ADD entry type")

        replacement = dict(manifest.minimal_patch or {})["value"]
        path = fork_root / surface.owner_path()
        if definition.surface_type == "text" and path.suffix == ".md":
            if not isinstance(replacement, str):
                raise ValueError("text surface replacement must be a string")
            path.write_bytes(canonical_text_bytes(replacement.encode("utf-8")))
            return
        document = _json_value(path)
        if not isinstance(document, dict):
            raise ValueError("editable JSON surface must have an object root")
        updated = _pointer_set(document, definition.json_pointer, replacement)
        if definition.surface_template_id == "candidate_policy.agent_program_slots":
            if isinstance(replacement, bool) or not isinstance(replacement, int):
                raise ValueError("agent_program_slots must be an integer")
            updated["total_k"] = 1 + replacement
        self._write_json(path, updated)

    def apply_to_fork(
        self,
        parent: MaterializedSnapshot,
        manifest: EditManifest,
        *,
        confirmed_cause: str,
    ) -> AppliedEditReceipt:
        validated = self.validate(parent, manifest, confirmed_cause=confirmed_cause)
        fork_root = self.store.fork(parent, manifest.edit_id)
        try:
            self._apply(fork_root, validated)
            candidate = compile_snapshot(fork_root, verify_lock=False)
            if candidate.harness_content_sha == parent.harness_content_sha:
                raise ValueError("edit produced no semantic Harness change")
            materialized = self.store.materialize(
                candidate,
                parent_sha=parent.runtime_bundle_sha,
            )
        finally:
            self.store.discard_fork(fork_root)
        payload = {
            "schema_version": "applied-edit-receipt/1",
            "edit_id": manifest.edit_id,
            "target_surface_id": manifest.target_surface_id,
            "confirmed_cause": confirmed_cause,
            "parent_harness_content_sha": parent.harness_content_sha,
            "candidate_harness_content_sha": materialized.harness_content_sha,
            "parent_runtime_bundle_sha": parent.runtime_bundle_sha,
            "candidate_runtime_bundle_sha": materialized.runtime_bundle_sha,
            "source_surfaces_changed": [manifest.target_surface_id],
            "derived_outputs_changed": list(
                validated.surface.definition.derived_outputs
            ),
        }
        return AppliedEditReceipt(
            edit_id=manifest.edit_id,
            target_surface_id=manifest.target_surface_id,
            confirmed_cause=confirmed_cause,
            parent_harness_content_sha=parent.harness_content_sha,
            candidate_harness_content_sha=materialized.harness_content_sha,
            parent_runtime_bundle_sha=parent.runtime_bundle_sha,
            candidate_runtime_bundle_sha=materialized.runtime_bundle_sha,
            parent_root=parent.root,
            candidate_root=materialized.root,
            source_surfaces_changed=(manifest.target_surface_id,),
            derived_outputs_changed=validated.surface.definition.derived_outputs,
            applied_edit_sha=canonical_sha256(payload),
            candidate_snapshot=materialized,
        )


__all__ = [
    "AppliedEditReceipt",
    "EditAuthorizationError",
    "EditController",
    "NoOpEditError",
    "StaleEditError",
    "SurfaceRegistry",
]
