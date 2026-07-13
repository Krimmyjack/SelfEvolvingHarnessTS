"""Public, frozen constants for the forecast benchmark protocol.

`benchmark-v0.1` supersedes `benchmark-v0`.  It is a genuinely different arena --
larger roster (METR-LA, GEFCom2012, an expanded NOAA weather pool), spatial
blocking for METR-LA, and a wider corruption grid -- so it carries its own version
string.  The version is an input to both the outer-split group hash and the
corruption seed, which means v0.1 draws its own role assignment and its own
corruption realizations; nothing silently inherits v0's.

`benchmark-v0` artifacts stay readable so the sealed, never-opened v0 Final-Query
remains auditable.  Validators accept any KNOWN_BENCHMARK_VERSIONS entry; builders
only ever emit BENCHMARK_VERSION.
"""
from __future__ import annotations

BENCHMARK_VERSION = "benchmark-v0.1"
KNOWN_BENCHMARK_VERSIONS = ("benchmark-v0", "benchmark-v0.1")

HEADLINE_LOOKBACK = 48
HEADLINE_HORIZON = 48
HEADLINE_MIN_LENGTH = 207

MODEL_SEEDS = (0, 1, 2)
CORRUPTION_REPLICATES = (0, 1)

HARM_THRESHOLD = 0.05
HARM_THRESHOLD_KIND = "conventional"
SATURATION_GAP = 0.02
SATURATION_GAP_KIND = "conventional"

BOOTSTRAP_B = 2000
BOOTSTRAP_MASTER_SEED = 20260713

DESIGN_COMMIT = "9e57da9"
EXTERNAL_ADDENDUM_SHA256 = (
    "468c65fbcb36f48a47a351597f99d9ccebd876fff39d3378923500a8c3ed45ff"
)

__all__ = [
    "BENCHMARK_VERSION",
    "BOOTSTRAP_B",
    "BOOTSTRAP_MASTER_SEED",
    "CORRUPTION_REPLICATES",
    "DESIGN_COMMIT",
    "EXTERNAL_ADDENDUM_SHA256",
    "HARM_THRESHOLD",
    "HARM_THRESHOLD_KIND",
    "HEADLINE_HORIZON",
    "HEADLINE_LOOKBACK",
    "HEADLINE_MIN_LENGTH",
    "KNOWN_BENCHMARK_VERSIONS",
    "MODEL_SEEDS",
    "SATURATION_GAP",
    "SATURATION_GAP_KIND",
]
