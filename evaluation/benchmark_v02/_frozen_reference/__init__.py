"""Private, frozen benchmark-v0.2 reference implementation.

This package exists only to reproduce the retired benchmark arm.  Active
methods and generic runtime code must never import it.
"""

from .config import LegacyReferenceState, default_state
from .fast_path import prepared_artifact, run_legacy_reference_batch

__all__ = [
    "LegacyReferenceState",
    "default_state",
    "prepared_artifact",
    "run_legacy_reference_batch",
]
