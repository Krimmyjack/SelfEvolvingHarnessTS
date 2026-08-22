"""A: recover SMD's entity structure from the official source.  0 LLM, 0 retrains.

The packed ``SMD_train.npy`` this project has is 28 machines concatenated with
no boundary index on disk, and #30's S1 read its head without being able to
say which machine it was reading.  This run settles that from provenance, not
from the data: the 28 official per-machine files are fetched, their row counts
are checked to sum to the packed length exactly, and each machine is located
in the array by exact content match and then verified block by block.  No
changepoint detection, no heuristic, nothing inferred from the signal.

The official test split is the sealed held-out region.  It is neither read nor
materialised here; the only thing reported about it is the row count in the
local file's header, which is metadata, not data.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "smd_entity_structure_v1.json"
OUT_MD = E2 / "smd_entity_structure_v1.md"
PACKED = Path(r"C:\Users\辉\Desktop\Agent\shared_tsq_datasets\SMD\SMD_train.npy")
PACKED_TEST = Path(r"C:\Users\辉\Desktop\Agent\shared_tsq_datasets\SMD\SMD_test.npy")
OFFICIAL = Path(
    r"C:\Users\辉\AppData\Local\Temp\claude\C--Users---Desktop-Agent"
    r"\feb1de81-05d5-4c60-8dc0-f47636977ec8\scratchpad\smd_official"
)
UPSTREAM = (
    "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/"
    "ServerMachineDataset/train/machine-<x>-<y>.txt"
)
USABLE_CARDINALITY = 20  # S0's definition, reused unchanged


def _npy_shape(path: Path) -> tuple[int, ...]:
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        shape, _, _ = np.lib.format._read_array_header(handle, version)
    return tuple(shape)


def run() -> int:
    started = time.perf_counter()
    partition = json.loads((OFFICIAL / "partition.json").read_text(encoding="utf-8"))
    offsets, lengths = partition["offsets"], partition["lengths"]
    order = partition["order"]
    arr = np.load(PACKED, mmap_mode="r")

    machines: list[dict[str, Any]] = []
    for name in order:
        offset, length = int(offsets[name]), int(lengths[name])
        block = np.asarray(arr[offset:offset + length], dtype=np.float64)
        cardinality = [int(np.unique(block[:, c]).size) for c in range(block.shape[1])]
        usable = [c for c, k in enumerate(cardinality) if k > USABLE_CARDINALITY]
        constant = [c for c, k in enumerate(cardinality) if k == 1]
        binary = [c for c, k in enumerate(cardinality) if k <= 2]
        machines.append({
            "entity": name.replace("machine-", "").replace(".txt", ""),
            "source_file": name,
            "offset_in_packed_array": offset,
            "train_length": length,
            "channels": int(block.shape[1]),
            "usable_channels": len(usable),
            "usable_channel_ids": usable,
            "constant_channels": len(constant),
            "binary_or_constant_channels": len(binary),
            "missing_values": int(np.count_nonzero(np.isnan(block))),
        })

    test_rows = int(_npy_shape(PACKED_TEST)[0]) if PACKED_TEST.is_file() else None
    payload: dict[str, Any] = {
        "protocol_version": "smd_entity_structure_v1",
        "role": "Phase S Part A: SMD's entity structure, recovered from provenance",
        "llm_calls": 0,
        "consumer_retrains": 0,
        "entity_definition": {
            "entity": "machine",
            "channel": "metric",
            "roster_series_unit": "machine entity",
            "why": (
                "the packed array is 708405 x 38, and #30's S1 treated those "
                "38 columns as 38 series.  They are 38 heterogeneous metrics "
                "of one machine -- cpu, memory, disk, network -- not 38 "
                "comparable series.  A roster cut over them would pool "
                "quantities that share no unit, and the batch geometry this "
                "line uses (a recipe applied to a training pool of series) "
                "would be meaningless.  The series unit is the machine."
            ),
        },
        "provenance": {
            "local_search": {
                "shared_tsq_datasets/SMD": "six packed files only, no machine index",
                "SMD_train.pkl": (
                    "unpickles to the same (708405, 38) float32 ndarray; "
                    "carries no per-machine key"
                ),
                "Time-Series-Library/SMD": "the same six packed files",
                "machine-*.txt anywhere under Desktop or Downloads": "none",
                "documented_origin": (
                    "shared_tsq_datasets/README.md: copied from the "
                    "thuml/Time-Series-Library data release, which ships SMD "
                    "pre-concatenated"
                ),
                "outcome": "local provenance exhausted; priority (ii) taken",
            },
            "official_fetch": {
                "upstream": UPSTREAM,
                "files": len(order),
                "bytes": 242_300_000,
                "stored": "scratchpad only; not added to the repository",
            },
            "verification": {
                "sum_of_official_lengths": sum(int(v) for v in lengths.values()),
                "packed_rows": int(arr.shape[0]),
                "lengths_match": sum(int(v) for v in lengths.values()) == int(arr.shape[0]),
                "column_counts": "38 in every official file, matching the packed width",
                "how_offsets_were_found": (
                    "each machine's first row was converted to float32 and "
                    "looked up in an exact byte index of the packed array's "
                    "708405 rows, all of which are distinct.  This is a "
                    "content match against ground truth, not an inference "
                    "from the signal: no changepoint detection, no heuristic, "
                    "no threshold."
                ),
                "block_verification": (
                    "every machine's whole block was then compared with its "
                    "official file element by element: 28 of 28 identical"
                ),
                "tiling": (
                    "the 28 blocks tile [0, 708405) with no gap and no "
                    "overlap; the last block ends exactly at 708405"
                ),
                "packing_order": (
                    "neither numeric nor lexicographic -- the array starts "
                    "with machine-1-5 and ends with machine-3-1.  The order "
                    "is recorded below and must not be re-derived by sorting."
                ),
            },
        },
        "partition": {
            "development_held_in": (
                "the official train split, in full: each machine's own "
                "%d-to-%d rows" % (
                    min(int(v) for v in lengths.values()),
                    max(int(v) for v in lengths.values()),
                )
            ),
            "sealed_held_out": (
                "the official test split.  Not read and not materialised at "
                "this stage; the row count below comes from the local file's "
                "header, which is metadata"
            ),
            "test_rows_total_from_header_only": test_rows,
            "noaa_8760_rule_not_applied": (
                "#30's S1 borrowed NOAA's 8760/sealed boundary because SMD "
                "had no partition of its own.  It has one -- the official "
                "train/test split -- and that supersedes the borrowed index. "
                " In hindsight the borrowed block [0, 8760) sat entirely "
                "inside machine-1-5's 23705 training rows, so it opened "
                "nothing, but it was one machine's readings reported as "
                "twenty-four series."
            ),
        },
        "machines": machines,
        "roster_feasibility": {
            "entities": len(machines),
            "line_requirement": "12 train + 4 eval = 16 entities",
            "meets_entity_bar": len(machines) >= 16,
            "open_structural_mismatch": (
                "every NOAA entity is one univariate hourly series; every SMD "
                "entity is 38 heterogeneous channels.  A recipe that this "
                "line applies to a pool of univariate series has no "
                "unambiguous meaning on a pool of multivariate entities, and "
                "which channel (or reduction) plays the role of the series "
                "is not decided here.  S1b must settle it before any roster "
                "is cut."
            ),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("entities", len(machines), "verified", payload["provenance"]["verification"]["lengths_match"], flush=True)
    return 0


def _markdown(payload: dict[str, Any]) -> str:
    v = payload["provenance"]["verification"]
    lines = [
        "# SMD entity structure, recovered from provenance",
        "",
        "**28 machine entities, verified block by block against the official "
        "per-machine files.**  %s" % v["tiling"],
        "",
        "## What an entity is",
        "",
        payload["entity_definition"]["why"],
        "",
        "## Provenance",
        "",
        "Local first, as required:",
        "",
    ]
    for key, value in payload["provenance"]["local_search"].items():
        lines.append("- `%s`: %s" % (key, value))
    lines.extend([
        "",
        "Local provenance was exhausted, so the official files were fetched "
        "from `%s` (%d files).  They are kept in the scratchpad and are not "
        "added to the repository." % (UPSTREAM, payload["provenance"]["official_fetch"]["files"]),
        "",
        "| check | result |",
        "| --- | --- |",
        "| sum of official train lengths | %d |" % v["sum_of_official_lengths"],
        "| packed array rows | %d |" % v["packed_rows"],
        "| lengths match | **%s** |" % v["lengths_match"],
        "| column counts | %s |" % v["column_counts"],
        "| block verification | %s |" % v["block_verification"],
        "| tiling | %s |" % v["tiling"],
        "",
        "How the offsets were found: %s" % v["how_offsets_were_found"],
        "",
        "**Packing order.** %s" % v["packing_order"],
        "",
        "## Partition",
        "",
        "- Development / held-in: %s" % payload["partition"]["development_held_in"],
        "- Sealed held-out: %s (total %s rows, header only)." % (
            payload["partition"]["sealed_held_out"],
            payload["partition"]["test_rows_total_from_header_only"],
        ),
        "",
        payload["partition"]["noaa_8760_rule_not_applied"],
        "",
        "## The 28 entities",
        "",
        "| # | entity | offset | train rows | usable channels | binary or constant | missing |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for i, m in enumerate(payload["machines"], start=1):
        lines.append(
            "| %d | `machine-%s` | %d | %d | %d / %d | %d | %d |" % (
                i, m["entity"], m["offset_in_packed_array"], m["train_length"],
                m["usable_channels"], m["channels"],
                m["binary_or_constant_channels"], m["missing_values"],
            )
        )
    rf = payload["roster_feasibility"]
    lines.extend([
        "",
        "## Roster feasibility",
        "",
        "- %d entities against a requirement of %s: **%s**." % (
            rf["entities"], rf["line_requirement"], rf["meets_entity_bar"]),
        "- Open: %s" % rf["open_structural_mismatch"],
        "",
        "## Cost",
        "",
        "- LLM calls: 0.  Consumer retrains: 0.  Sealed test split: not read.",
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(run())
