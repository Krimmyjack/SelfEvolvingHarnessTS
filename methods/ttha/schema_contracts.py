from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import parse_json_document
from SelfEvolvingHarnessTS.runtime.errors import ProtocolViolation


_SCHEMA_ROOT = Path(__file__).resolve().with_name("schemas")
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
    return load_schema(_STAGE_SCHEMA_FILES[name])


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


def validate_local_schema(value: object, schema: Mapping[str, object], *, path: str = "payload") -> None:
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
                    validate_local_schema(value[key], child_schema, path=f"{path}.{key}")
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
                validate_local_schema(item, item_schema, path=f"{path}[{index}]")
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


__all__ = ["LocalSchemaError", "load_schema", "load_stage_schema", "validate_local_schema"]
