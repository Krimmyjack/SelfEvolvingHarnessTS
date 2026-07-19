from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


CANONICALIZATION_VERSION = "m0-c14n/1"
_UTF8_BOM = b"\xef\xbb\xbf"


def _decode_utf8(raw: bytes, *, artifact: str) -> str:
    if not isinstance(raw, bytes):
        raise TypeError(f"{artifact} must be bytes")
    payload = raw[len(_UTF8_BOM) :] if raw.startswith(_UTF8_BOM) else raw
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{artifact} must be valid UTF-8") from exc


def canonical_text_bytes(raw: bytes) -> bytes:
    text = _decode_utf8(raw, artifact="text")
    if "\x00" in text:
        raise ValueError("text must not contain NUL")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFC", normalized).rstrip("\n") + "\n"
    return normalized.encode("utf-8")


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number is forbidden")
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_nonfinite(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _reject_nonfinite(nested)


def parse_json_document(raw: bytes) -> Any:
    text = _decode_utf8(raw, artifact="JSON document")
    if "\x00" in text:
        raise ValueError("JSON document must not contain NUL")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON document: {exc.msg}") from exc
    _reject_nonfinite(value)
    return value


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number is forbidden")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        original_keys: dict[str, str] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            canonical_key = unicodedata.normalize("NFC", key)
            if canonical_key in normalized:
                first = original_keys[canonical_key]
                raise ValueError(
                    f"JSON key normalization collision: {first!r} and {key!r}"
                )
            original_keys[canonical_key] = key
            normalized[canonical_key] = _normalize_json(nested)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_normalize_json(nested) for nested in value]
    raise TypeError(f"value is not JSON-native: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_document_bytes(raw: bytes) -> bytes:
    return canonical_json_bytes(parse_json_document(raw))


def canonical_jsonl_bytes(raw: bytes) -> bytes:
    text = _decode_utf8(raw, artifact="JSONL document")
    if "\x00" in text:
        raise ValueError("JSONL document must not contain NUL")
    rows: list[bytes] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(canonical_json_bytes(parse_json_document(line.encode("utf-8"))))
        except ValueError as exc:
            raise ValueError(f"invalid JSONL row {line_number}: {exc}") from exc
    return b"\n".join(rows) + (b"\n" if rows else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "CANONICALIZATION_VERSION",
    "canonical_json_bytes",
    "canonical_json_document_bytes",
    "canonical_jsonl_bytes",
    "canonical_sha256",
    "canonical_text_bytes",
    "parse_json_document",
]
