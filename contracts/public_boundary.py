from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "private_family",
        "private_severity",
        "private_receipt_refs",
        "oracle_affected_indices",
        "clean_context",
        "clean_future",
        "clean_u",
        "corrupt_u",
        "prepared_u",
        "damage_d",
        "repair_gain_g",
        "nrr",
        "candidate_utilities",
        "selection_regret",
        "loss_j",
        "utility_u",
        "r_private",
        "injection_type",
        "confirmed_surface",
    }
)


def assert_public_payload(value: Any, *, path: str = "public_payload") -> None:
    """Reject judge/oracle field names before any Agent backend invocation."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key)
            if name.lower() in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"forbidden private field at {path}.{name}")
            assert_public_payload(nested, path=f"{path}.{name}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            assert_public_payload(nested, path=f"{path}[{index}]")


__all__ = ["FORBIDDEN_PUBLIC_KEYS", "assert_public_payload"]
