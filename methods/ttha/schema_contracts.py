from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import parse_json_document
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES
from SelfEvolvingHarnessTS.runtime.errors import ProtocolViolation


_SCHEMA_ROOT = Path(__file__).resolve().with_name("schemas")
_OBSERVABLE_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "contracts"
    / "schemas"
    / "observable_feature_v1.json"
)
_CONTRACT_SCHEMA_ROOT = _OBSERVABLE_SCHEMA.parent
_STAGE_SCHEMA_FILES = {
    "fast_inspect_v1": "fast_inspect_v1.json",
    "fast_propose_v1": "fast_propose_v1.json",
    "fast_select_v1": "fast_select_v1.json",
    "slow_edit_v1": "slow_edit_v1.json",
}


class LocalSchemaError(ProtocolViolation):
    """A local Agent envelope or stage payload violates its schema."""


def load_schema(filename: str) -> dict[str, Any]:
    value = parse_json_document((_SCHEMA_ROOT / filename).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"schema {filename} must be an object")
    return value


def load_stage_schema(name: str) -> dict[str, Any]:
    if name not in _STAGE_SCHEMA_FILES:
        raise ValueError(f"unknown stage schema: {name}")
    schema = load_schema(_STAGE_SCHEMA_FILES[name])
    if name == "fast_propose_v1":
        try:
            steps = schema["properties"]["candidates"]["items"]["properties"][
                "steps"
            ]
            step_schema = steps["items"]
        except (KeyError, TypeError) as exc:
            raise ValueError("fast propose schema has no operator injection point") from exc
        if (
            not isinstance(step_schema, dict)
            or step_schema.get("$comment")
            != "canonical operator step branches are injected at load time"
        ):
            raise ValueError("fast propose operator injection point drifted")
        branches: list[dict[str, object]] = []
        for operator_name in OPERATOR_NAMES:
            metadata = OPERATOR_METADATA[operator_name]
            bindings = metadata.get(
                "public_parameter_bindings", {}
            )
            if not isinstance(bindings, Mapping):
                raise ValueError("operator public_parameter_bindings must be an object")
            declared_parameter_schema = metadata.get("public_parameter_schema")
            if declared_parameter_schema is not None and not isinstance(
                declared_parameter_schema, Mapping
            ):
                raise ValueError("operator public_parameter_schema must be an object")
            if bindings and declared_parameter_schema is not None:
                raise ValueError(
                    "operator cannot declare both bound and free public parameters"
                )
            params_schema: dict[str, object] = (
                dict(declared_parameter_schema)
                if isinstance(declared_parameter_schema, Mapping)
                else {"type": "object"}
            )
            if bindings:
                params_schema.update(
                    {
                        "additionalProperties": False,
                        "required": sorted(str(key) for key in bindings),
                        "properties": {
                            str(key): {"type": "number"} for key in sorted(bindings)
                        },
                    }
                )
            branches.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["op", "params"],
                    "properties": {
                        "op": {"const": operator_name},
                        "params": params_schema,
                    },
                }
            )
        steps["items"] = {"oneOf": branches}
    if name == "slow_edit_v1":
        definitions = schema.get("$defs")
        required_injections = {
            "observable_leaf",
            "behavior_predicate",
            "learned_skill_entry",
            "memory_entry",
        }
        if not isinstance(definitions, dict) or not required_injections.issubset(definitions):
            raise ValueError("slow edit schema injection points drifted")
        observable_leaf = parse_json_document(_OBSERVABLE_SCHEMA.read_bytes())
        if not isinstance(observable_leaf, dict):
            raise ValueError("observable feature schema must be an object")
        definitions["observable_leaf"] = observable_leaf

        behavior = load_schema("behavior_predicate_v1.json")
        behavior_branches = behavior.get("oneOf")
        if not isinstance(behavior_branches, list):
            raise ValueError("behavior predicate schema has no branches")
        injection = next(
            (
                branch
                for branch in behavior_branches
                if isinstance(branch, dict)
                and branch.get("$comment")
                == "canonical forecast operator values are injected at load time"
            ),
            None,
        )
        if not isinstance(injection, dict) or injection.get("enum") != []:
            raise ValueError("behavior predicate operator injection point drifted")
        forecast_operators = tuple(
            sorted(
                name
                for name, metadata in OPERATOR_METADATA.items()
                if metadata.get("deprecated") is not True
                and "forecast" in metadata.get("allowed_tasks", ())
            )
        )
        injection["enum"] = [f"supply_operator:{name}" for name in forecast_operators]
        definitions["behavior_predicate"] = behavior

        learned_skill = parse_json_document(
            (_CONTRACT_SCHEMA_ROOT / "skill_entry_v1.json").read_bytes()
        )
        memory_entry = parse_json_document(
            (_CONTRACT_SCHEMA_ROOT / "memory_entry_v1.json").read_bytes()
        )
        if not isinstance(learned_skill, dict) or not isinstance(memory_entry, dict):
            raise ValueError("deployable entry schemas must be objects")
        learned_properties = learned_skill.get("properties")
        memory_properties = memory_entry.get("properties")
        if not isinstance(learned_properties, dict) or not isinstance(memory_properties, dict):
            raise ValueError("deployable entry schema properties drifted")
        learned_properties["skill_kind"] = {"const": "capability"}
        learned_properties["observable_applicability"] = {
            "$ref": "#/$defs/applicability"
        }
        learned_properties["allowed_tools"] = {
            "type": "array",
            "items": {"enum": list(forecast_operators)},
            "uniqueItems": True,
        }
        memory_properties["observable_applicability"] = {
            "$ref": "#/$defs/applicability"
        }
        definitions["learned_skill_entry"] = learned_skill
        definitions["memory_entry"] = memory_entry
    return schema


def _matches_type(value: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "null":
        return value is None
    return False


def _resolve_local_ref(root: Mapping[str, object], reference: str) -> Mapping[str, object]:
    if not reference.startswith("#/"):
        raise LocalSchemaError(f"unsupported non-local schema reference: {reference}")
    current: object = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise LocalSchemaError(f"unresolved local schema reference: {reference}")
        current = current[part]
    if not isinstance(current, Mapping):
        raise LocalSchemaError(f"schema reference is not an object: {reference}")
    return current


def _validate_local_schema(
    value: object,
    schema: Mapping[str, object],
    *,
    root: Mapping[str, object],
    path: str,
) -> None:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        _validate_local_schema(
            value,
            _resolve_local_ref(root, reference),
            root=root,
            path=path,
        )
        return
    for keyword, required_matches in (("oneOf", 1), ("anyOf", None)):
        branches = schema.get(keyword)
        if isinstance(branches, Sequence) and not isinstance(
            branches, (str, bytes, bytearray)
        ):
            matches = 0
            errors: list[str] = []
            preferred_errors: list[str] = []
            for branch in branches:
                if not isinstance(branch, Mapping):
                    continue
                try:
                    _validate_local_schema(value, branch, root=root, path=path)
                except LocalSchemaError as exc:
                    errors.append(str(exc))
                    if isinstance(value, Mapping):
                        branch_properties = branch.get("properties")
                        if isinstance(branch_properties, Mapping):
                            operation_schema = branch_properties.get("operation")
                            if (
                                isinstance(operation_schema, Mapping)
                                and operation_schema.get("const")
                                == value.get("operation")
                            ):
                                preferred_errors.append(str(exc))
                else:
                    matches += 1
            valid = matches == required_matches if required_matches is not None else matches > 0
            if not valid:
                detail = (
                    preferred_errors[0]
                    if preferred_errors
                    else errors[0]
                    if errors
                    else "no branch matched"
                )
                raise LocalSchemaError(
                    f"{path} does not satisfy {keyword} ({matches} matches): {detail}"
                )
            return
    branches = schema.get("allOf")
    if isinstance(branches, Sequence) and not isinstance(
        branches, (str, bytes, bytearray)
    ):
        for branch in branches:
            if isinstance(branch, Mapping):
                _validate_local_schema(value, branch, root=root, path=path)
    if "const" in schema and value != schema["const"]:
        raise LocalSchemaError(f"{path} does not match const")
    if "enum" in schema and value not in schema["enum"]:
        raise LocalSchemaError(f"{path} is outside enum")
    expected_type = schema.get("type")
    if expected_type is not None:
        allowed = [expected_type] if isinstance(expected_type, str) else list(expected_type)
        if not any(_matches_type(value, item) for item in allowed):
            raise LocalSchemaError(f"{path} has wrong type")
    if isinstance(value, Mapping):
        required = schema.get("required", ())
        missing = set(required) - set(value)
        if missing:
            raise LocalSchemaError(f"{path} missing fields: {sorted(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise LocalSchemaError(f"{path} has unexpected fields: {sorted(extra)}")
        if isinstance(properties, Mapping):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, Mapping):
                    _validate_local_schema(
                        value[key], child_schema, root=root, path=f"{path}.{key}"
                    )
        additional_schema = schema.get("additionalProperties")
        if isinstance(additional_schema, Mapping):
            known = set(properties) if isinstance(properties, Mapping) else set()
            for key in set(value) - known:
                _validate_local_schema(
                    value[key],
                    additional_schema,
                    root=root,
                    path=f"{path}.{key}",
                )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            raise LocalSchemaError(f"{path} has too few items")
        if isinstance(maximum, int) and len(value) > maximum:
            raise LocalSchemaError(f"{path} has too many items")
        if schema.get("uniqueItems") is True:
            from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256

            identities = [canonical_sha256(item) for item in value]
            if len(identities) != len(set(identities)):
                raise LocalSchemaError(f"{path} items must be unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_local_schema(
                    item, item_schema, root=root, path=f"{path}[{index}]"
                )
    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            raise LocalSchemaError(f"{path} is too short")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
            raise LocalSchemaError(f"{path} does not match pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise LocalSchemaError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise LocalSchemaError(f"{path} is above maximum")


def validate_local_schema(
    value: object,
    schema: Mapping[str, object],
    *,
    path: str = "payload",
) -> None:
    _validate_local_schema(value, schema, root=schema, path=path)


__all__ = ["LocalSchemaError", "load_schema", "load_stage_schema", "validate_local_schema"]
