from __future__ import annotations

import json
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
from SelfEvolvingHarnessTS.contracts.run_context import RunDependencyBinding
from SelfEvolvingHarnessTS.contracts.task import TaskContext

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


def _steps_for_patch_id(
    card: Mapping[str, object],
    patch_id: str | None,
) -> tuple[tuple[str, dict[str, object]], ...]:
    """P3.1-B2 Typed Patch Binding：从 FailurePatternCard 的
    typed_patch_options 白名单按 patch_id 取 Runtime-owned 冻结 steps。
    **禁止**从自然语言 body/skill_id 猜算子；未知 patch_id → 空（调用方
    判 ACTION_UNAVAILABLE）。"""
    if not patch_id:
        return ()
    for opt in (card.get("typed_patch_options") or []):
        if not isinstance(opt, Mapping):
            continue
        if str(opt.get("patch_id")) != patch_id:
            continue
        steps = [(str(s["op"]), dict(s.get("params") or {}))
                 for s in (opt.get("program_steps") or [])
                 if isinstance(s, Mapping) and s.get("op")]
        return tuple(steps) if steps else ()
    return ()


class FrozenProgramBindingError(RuntimeError):
    """P0: applied capability-skill body does not equal the replay Program."""


def _serialize_frozen_program_steps(
    steps: Sequence[tuple[str, Mapping[str, object]]],
) -> str:
    return "Frozen program steps: " + json.dumps(
        [{"op": op, "params": dict(params)} for op, params in steps]
    )


def is_capability_body_surface_id(surface_id: str) -> bool:
    """True only for an instantiated capability Skill ``.body`` PATCH target."""
    return bool(
        re.fullmatch(
            r"skill_library\.entries\/[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\.body",
            surface_id,
        )
    )


def bind_frozen_patch_program(
    manifest: Any,
    steps: Sequence[tuple[str, Mapping[str, object]]],
) -> Any:
    """P0 Runtime-owned PATCH body binding.

    For a capability Skill ``.body`` PATCH, overwrite ``minimal_patch.value``
    with the frozen Program serialized from the typed_patch_options whitelist.
    Any Slow-authored text is ignored.  Non-capability-body PATCHes and ADD
    manifests pass through unchanged.
    """
    import dataclasses

    if getattr(manifest, "operation", None) is not EditOperation.PATCH:
        return manifest
    if not is_capability_body_surface_id(str(manifest.target_surface_id)):
        return manifest
    steps = tuple(
        (str(op), dict(dict(params) if isinstance(params, Mapping) else {}))
        for op, params in steps
    )
    if not steps:
        raise FrozenProgramBindingError(
            "capability body PATCH requires non-empty replay steps"
        )
    minimal_patch = dict(getattr(manifest, "minimal_patch", None) or {})
    minimal_patch["value"] = _serialize_frozen_program_steps(steps)
    return dataclasses.replace(manifest, minimal_patch=minimal_patch)


def verify_frozen_patch_program(
    candidate_snapshot: HarnessSnapshot,
    *,
    target_surface_id: str,
    replay_steps: Sequence[tuple[str, Mapping[str, object]]],
) -> None:
    """P0 post-write readback assertion.

    Read the skill body back from the compiled candidate snapshot, parse it
    with the exact Fast-consumer parser (``fast_agent._parse_frozen_steps``),
    and require element-wise equality with the replay steps.  Any mismatch
    raises ``FrozenProgramBindingError``; callers must stop at ``apply_failed``
    and must not put the candidate into pending.
    """
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import _parse_frozen_steps

    if not isinstance(candidate_snapshot, HarnessSnapshot):
        raise TypeError("candidate_snapshot must be a HarnessSnapshot")
    match = re.fullmatch(
        r"skill_library\.entries\/([a-z][a-z0-9]*(?:[-_][a-z0-9]+)*)\.body",
        str(target_surface_id),
    )
    if match is None:
        raise FrozenProgramBindingError(
            f"not a capability body surface: {target_surface_id}"
        )
    skill_id = match.group(1)
    skill = next(
        (entry for entry in candidate_snapshot.skills if entry.skill_id == skill_id),
        None,
    )
    if skill is None:
        raise FrozenProgramBindingError(
            f"candidate snapshot has no skill {skill_id!r} for readback"
        )
    expected = tuple(
        (str(op), dict(dict(params) if isinstance(params, Mapping) else {}))
        for op, params in replay_steps
    )
    parsed = _parse_frozen_steps(skill.body)
    if parsed != expected:
        raise FrozenProgramBindingError(
            "applied capability body program != replay steps: "
            f"{skill_id} parsed={parsed!r} expected={expected!r}"
        )


def _frozen_program_steps(manifest: Any) -> tuple[tuple[str, dict[str, object]], ...]:
    """从 EditManifest 取冻结 Typed Program steps（P3.1-A2 方法层解析：
    new_value.body 的 "Frozen program steps:" marker 后 JSON——A2 阶段；
    B2 将替换为 Patch ID 绑定）。无法解析 → 空元组。"""
    nv = getattr(manifest, "new_value", None) or {}
    body = str(nv.get("body") or "")
    marker = "Frozen program steps:"
    idx = body.find(marker)
    if idx < 0:
        return ()
    rest = body[idx + len(marker):].strip()
    try:
        arr = json.loads(rest)
    except json.JSONDecodeError:
        return ()
    if not isinstance(arr, list):
        return ()
    steps = [(str(s["op"]), dict(s.get("params") or {}))
             for s in arr if isinstance(s, Mapping) and s.get("op")]
    return tuple(steps) if steps else ()


def _resolve_apply_manifest(manifest: Any, snapshot: HarnessSnapshot) -> Any:
    """apply 前契约修复（P1 同款，方法层内）：surface 模板实例化 +
    dependency_precondition_shas 从 snapshot.dependency_shas 按 surface
    的 required_dependency_keys 补全。

    Wave 2 接线修复（2026-08-13）：skill_library.entries 目标以 manifest
    自身的 new_value.skill_id 为准确定性实例化（真实 LLM 出现 surface
    与 entry ID 不一致 → EditShapeError；harness 只修表面模板/依赖 SHA
    ——不补替换位置/算子/Program steps/Scope-Risk 语义）。"""
    import dataclasses

    from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
        SurfaceRegistry,
    )

    reg = SurfaceRegistry()
    target = str(manifest.target_surface_id)
    nv = getattr(manifest, "new_value", None) or {}
    skill_id = str(nv.get("skill_id") or "")
    if target.startswith("skill_library.entries/") and skill_id:
        target = f"skill_library.entries/{skill_id}"
    resolved = reg.resolve(target)
    snapshot_deps = dict(snapshot.dependency_shas)
    declared_dep = {
        key: snapshot_deps[key]
        for key in resolved.definition.required_dependency_keys
        if key in snapshot_deps}
    return dataclasses.replace(
        manifest,
        target_surface_id=target,
        dependency_precondition_shas=declared_dep)


def _surface_catalog_entries(
    surface_catalog: Mapping[str, object] | Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(surface_catalog, Mapping):
        return (dict(surface_catalog),) if surface_catalog else ()
    return tuple(dict(item) for item in surface_catalog if isinstance(item, Mapping))


def _edit_rule_for_catalog(
    surface_catalog: Mapping[str, object] | Sequence[Mapping[str, object]],
) -> str:
    """P2/G-4: the instruction must describe the one authorized surface.

    The old static ``add_rule`` advertised PATCH even when the runtime only
    authorized ADD (and vice versa)."""
    entries = _surface_catalog_entries(surface_catalog)
    if not entries:
        return (
            "No writable surface is authorized for this fault. Do not emit "
            "edit_manifest; choose the schema no_proposal / abstain path."
        )
    operations: set[str] = set()
    surface_ids: list[str] = []
    for entry in entries:
        operations.update(
            str(value)
            for value in (entry.get("allowed_operations") or [])
            if isinstance(value, str)
        )
        surface_id = entry.get("surface_id")
        if isinstance(surface_id, str):
            surface_ids.append(surface_id)
    surfaces = ", ".join(surface_ids) if surface_ids else "the authorized surface"
    if operations == {"ADD"}:
        return (
            "ADD creates exactly one new entry on the single authorized "
            f"surface: {surfaces}. Never target an entry_id listed in "
            "existing_entry_inventory. PATCH is not authorized; do not PATCH."
        )
    if operations == {"PATCH"}:
        return (
            "PATCH exactly the single authorized surface: "
            f"{surfaces}. Copy its surface_id, operation, surface_precondition "
            "and dependency_precondition_shas exactly from "
            "writable_surface_catalog. ADD is not authorized; do not ADD."
        )
    return (
        "Use only the operations listed in writable_surface_catalog for the "
        f"authorized surface: {surfaces}."
    )


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
            patch_id=manifest.get("patch_id"),
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
        task_context: TaskContext | None = None,
        run_dependency_binding: RunDependencyBinding | None = None,
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
        if run_dependency_binding is not None:
            if task_context is None:
                raise ValueError("slow run dependency binding requires TaskContext")
            if run_dependency_binding.task_context_sha != task_context.sha():
                raise ValueError("slow run dependency TaskContext SHA mismatch")
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

        public_input = {
                "failure_pattern_card": _plain(card),
                "writable_surface_catalog": _plain(surface_catalog),
                "base_harness_sha": snapshot.harness_content_sha,
                "dependency_precondition_shas": _plain(snapshot.dependency_shas),
                "existing_entry_inventory": existing_inventory,
                "allowed_operator_contracts": _plain(allowed_operator_contracts),
                "fixed_probe_contracts": _plain(fixed_probe_contracts or {}),
                "add_rule": _edit_rule_for_catalog(surface_catalog),
            }
        if task_context is not None:
            public_input["task_context"] = task_context.to_dict()
            public_input["task_context_sha"] = task_context.sha()
        # Wave 2 接线修复（2026-08-13）：typed_patch_options 存在时把绑定
        # 规则显式注入 Agent 输入——Runtime 白名单 → edit_manifest.patch_id
        # 必须取自白名单（首次真实 LLM Witness 暴露：模型两次不输出
        # patch_id——通道存在但规则未显式声明）。
        typed_options = card.get("typed_patch_options") or []
        if typed_options:
            public_input["typed_patch_binding_rule"] = {
                "rule": (
                    "failure_pattern_card.typed_patch_options is the runtime "
                    "whitelist of typed patches. You MUST set "
                    "edit_manifest.patch_id to exactly one patch_id value "
                    "from that whitelist. The runtime binds the frozen "
                    "program steps from the whitelist entry — do not invent "
                    "a patch_id and do not describe program steps in prose."
                ),
                "available_patch_ids": [
                    str(o.get("patch_id"))
                    for o in typed_options
                    if isinstance(o, Mapping) and o.get("patch_id")],
            }

        stage = self.core.run_stage(
            role=AgentRole.SLOW,
            stage="edit",
            case_id=pattern_id,
            public_input=public_input,
            harness_view=view,
            output_schema_name="slow_edit_v1",
            output_schema=self.core.load_stage_schema("slow_edit_v1"),
            source_snapshot_sha=snapshot.runtime_bundle_sha,
            task_context_sha=task_context.sha() if task_context is not None else "",
            run_context_sha=(
                run_dependency_binding.sha()
                if run_dependency_binding is not None
                else ""
            ),
            validation_retries=1,
            post_validator=post_validate,
        )
        self.last_stage_result = stage
        if stage.no_proposal_reason is not None:
            self.last_no_proposal_reason = stage.no_proposal_reason
            return None
        return self._manifest_from_payload(stage.payload)


__all__ = [
    "FrozenProgramBindingError",
    "TTHASlowAgent",
    "bind_frozen_patch_program",
    "is_capability_body_surface_id",
    "verify_frozen_patch_program",
]
