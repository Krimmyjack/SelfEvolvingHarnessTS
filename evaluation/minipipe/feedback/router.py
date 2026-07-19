from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from SelfEvolvingHarnessTS.contracts.canonical import parse_json_document


_ROUTES_PATH = Path(__file__).with_name("fault_routes.json")
_FORBIDDEN_TARGET_FRAGMENTS = (
    "observable_feature_v1.json",
    "contracts/observables.py",
    "probes/features.py",
    "public_tools.py",
)


@dataclass(frozen=True)
class RouteAuthorization:
    cause_code: str
    actionability: str
    target_classes: tuple[str, ...]
    allowed_skill_kinds: tuple[str, ...]
    allowed_operations: tuple[str, ...]
    allowed_surface_ids: tuple[str, ...]


class FaultRouter:
    def __init__(self, path: Path = _ROUTES_PATH) -> None:
        value = parse_json_document(path.read_bytes())
        if not isinstance(value, dict) or value.get("schema_version") != "fault-routes/2":
            raise ValueError("invalid fault route table")
        routes = value.get("routes")
        if not isinstance(routes, dict):
            raise ValueError("fault route table has no routes")
        self._routes = routes

    def allowed_targets(self, cause_code: str) -> RouteAuthorization:
        raw = self._routes.get(cause_code)
        if not isinstance(raw, dict):
            raise KeyError(f"unknown fault cause: {cause_code}")
        return RouteAuthorization(
            cause_code=cause_code,
            actionability=str(raw["actionability"]),
            target_classes=tuple(str(value) for value in raw.get("target_classes", [])),
            allowed_skill_kinds=tuple(str(value) for value in raw.get("skill_kinds", [])),
            allowed_operations=tuple(str(value) for value in raw.get("operations", [])),
            allowed_surface_ids=tuple(str(value) for value in raw.get("surface_ids", [])),
        )

    def authorize(
        self,
        cause_code: str,
        *,
        target_class: str,
        operation: str,
        skill_kind: str | None = None,
        target_surface_id: str | None = None,
    ) -> RouteAuthorization:
        route = self.allowed_targets(cause_code)
        if target_class not in route.target_classes:
            raise ValueError("target class is not authorized for the attributed cause")
        if operation not in route.allowed_operations:
            raise ValueError("edit operation is not authorized for the attributed cause")
        if skill_kind is not None and skill_kind not in route.allowed_skill_kinds:
            raise ValueError("skill kind is not authorized for the attributed cause")
        required_kind = {
            "bootstrap_procedure": "bootstrap_procedure",
            "capability": "capability",
            "capability_risk_guard": "capability",
            "safety": "safety",
        }.get(target_class)
        if required_kind is not None and skill_kind != required_kind:
            raise ValueError("target class and skill kind do not form an authorized pair")
        if target_class == "capability_risk_guard" and operation != "PATCH":
            raise ValueError("RISK_GAP may only patch an existing capability risk guard")
        if route.allowed_surface_ids and target_surface_id is None:
            raise ValueError("cause requires an exact declared surface ID")
        if target_surface_id is not None:
            normalized = target_surface_id.replace("\\", "/")
            if any(fragment in normalized for fragment in _FORBIDDEN_TARGET_FRAGMENTS):
                raise ValueError("observable wall substrate is never an M0 edit target")
            if route.allowed_surface_ids and target_surface_id not in route.allowed_surface_ids:
                raise ValueError("cause is restricted to a declared surface ID")
        return route


__all__ = ["FaultRouter", "RouteAuthorization"]
