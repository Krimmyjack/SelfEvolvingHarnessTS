"""Public, frozen constants for the forecast benchmark protocol.

`benchmark-v0.1` superseded `benchmark-v0` (larger roster, METR-LA spatial blocking,
a wider corruption grid).  `benchmark-v0.2` now supersedes v0.1 and changes three
things that all sit on the frozen face, so they are bumped together rather than
slipped in as revisions:

1. **Downstream training unit.**  The frozen spec fixes what a trainer does -- L, H,
   seeds, ridge lambda, series-equal weighting -- but is silent on *how many models
   are fitted and over which series*.  v0 sliced the pool by `dataset x regime`,
   which leaked the benchmark-private regime tag into the model; v0.1 pooled the
   whole role, which let a program applied to one dataset move the shared weights
   under every other dataset.  v0.2 fills the gap explicitly: one model per
   `(program, scenario, dose, replicate, dataset)`.  `dataset_id` is public, so this
   leaks nothing, and it makes a dataset-conditioned oracle a world that was
   actually trained rather than an untrained counterfactual.

2. **Program pool.**  v0.1's four programs all filled missing values, so five of the
   nine corruption cells could not be acted on at all and every program scored
   identically on them.  v0.2 freezes a mechanism-covering pool (`programs.py`).
   This changes what `best_fixed` means, which is a baseline definition, which is
   frozen.

3. **Support-A sub-split constraint.**  A-validation is now restricted to
   `certified_virgin` series; anything that fed the incumbent harness is confined to
   A-discovery.

The version string is an input to the outer-split group hash, the corruption seed,
and the Support-A partition hash, so v0.2 draws its own role assignment, its own
corruption realizations, and its own sub-split.  Nothing silently inherits v0.1's.
v0.1 Dev numbers are therefore *not* comparable to v0.2 Dev numbers; they were
measured on a different arena and on a different ruler.

Older artifacts stay readable so the sealed, never-opened Final-Query of each prior
version remains auditable.  Validators accept any KNOWN_BENCHMARK_VERSIONS entry;
builders only ever emit BENCHMARK_VERSION.
"""
from __future__ import annotations

BENCHMARK_VERSION = "benchmark-v0.2"
KNOWN_BENCHMARK_VERSIONS = ("benchmark-v0", "benchmark-v0.1", "benchmark-v0.2")

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
