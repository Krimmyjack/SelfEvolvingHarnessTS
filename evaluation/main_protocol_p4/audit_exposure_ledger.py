"""Has any Target series ever been scored at a proposed held-out origin?

The unit of exposure is the pair, not the origin: an old cohort having been read
at origin 4536 says nothing about whether *these* series were.  So the ledger
walks every artifact this data version produced, collects the series each one
names and the origins it scored, and reports the intersection with the proposed
held-out block.

It reads only which cells were evaluated -- never an error, a gain or a utility.
That is the same line the geometry checks already draw: whether the missing-aware
sMASE is defined at all is a mask question; what it evaluates to is an Outcome.

A pair that appears nowhere is unexposed.  A pair that appears anywhere, in any
role, disqualifies that origin from the held-out block.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts/main_protocol"
SUPPLY = ARTIFACTS / "p4s_main_experiment_supply.json"
OUT = ARTIFACTS / "p4t_exposure_ledger.json"

TARGET_SLICE = (80, 120)
PROPOSED_HELD_OUT = (4056, 4296, 4536, 4776, 5016)
#: This audit's own outputs are not evidence about the past.
SELF = {"p4t_exposure_ledger.json"}
#: Naming a pair is not scoring it.  An inventory that fitted no Consumer could
#: not have produced an Outcome for any cell, so it cannot expose one.  The rule
#: is general and read from each artifact's own declared boundary -- not an
#: exception carved out for this line's own supply audits.
OUTCOME_KEYS = ("consumer_fits", "llm_calls")
_UID = re.compile(r"\bT\d{1,4}\b")


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_strings(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for nested in value:
            yield from _walk_strings(nested)
    elif isinstance(value, (int, float)):
        yield str(value)


def _uids_and_origins(payload: Any) -> tuple[set[str], set[int]]:
    uids: set[str] = set()
    origins: set[int] = set()
    for text in _walk_strings(payload):
        uids.update(_UID.findall(text))
        if text.isdigit() and 100 <= int(text) <= 12000:
            origins.add(int(text))
    return uids, origins


def scan_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - an unreadable artifact is a finding
        return {"artifact": path.name, "unreadable": str(exc)[:100]}
    uids, origins = _uids_and_origins(payload)
    return {"artifact": path.name, "series": uids, "origins": origins,
            "consumer_fits": _declared_fits(payload)}


def _declared_fits(payload: Any) -> int | None:
    """Consumer fits the artifact says it spent, wherever it declared them."""
    found: list[int] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in OUTCOME_KEYS and isinstance(value, (int, float)):
                    found.append(int(value))
                walk(value)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for value in node:
                walk(value)

    walk(payload)
    return max(found) if found else None


def scan_npz(path: Path) -> dict[str, Any]:
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        return {"artifact": path.name, "unreadable": str(exc)[:100]}
    uids: set[str] = set()
    origins: set[int] = set()
    for key in data.files:
        try:
            array = data[key]
        except Exception:  # noqa: BLE001
            continue
        if array.dtype == object:
            uids.update(
                str(value) for value in array.ravel()
                if isinstance(value, str) and _UID.fullmatch(str(value))
            )
        elif np.issubdtype(array.dtype, np.integer):
            origins.update(
                int(value) for value in array.ravel() if 100 <= int(value) <= 12000
            )
    # A tensor is a record of scored cells: treat it as outcome-bearing.
    return {"artifact": path.name, "series": uids, "origins": origins,
            "consumer_fits": None}


def build() -> dict[str, Any]:
    supply = json.loads(SUPPLY.read_text(encoding="utf-8"))
    readable = supply["readable_uids"]
    target = readable[TARGET_SLICE[0]:TARGET_SLICE[1]]
    target_set = set(target)

    rows = []
    for path in sorted(ARTIFACTS.rglob("*")):
        if path.is_dir() or path.name in SELF:
            continue
        if path.suffix == ".json":
            entry = scan_json(path)
        elif path.suffix == ".npz":
            entry = scan_npz(path)
        else:
            continue
        if "unreadable" in entry:
            rows.append(entry)
            continue
        touched = sorted(entry["series"] & target_set)
        overlap = sorted(entry["origins"] & set(PROPOSED_HELD_OUT))
        if touched or overlap:
            fits = entry.get("consumer_fits")
            inventory_only = fits == 0
            rows.append({
                "artifact": entry["artifact"],
                "target_series_named": touched,
                "held_out_origins_named": overlap,
                "declared_consumer_fits": fits,
                "inventory_only": inventory_only,
                "both": bool(touched and overlap),
                # Naming both only exposes the pair if the artifact could have
                # scored it.  An audit that fitted nothing read a mask, not an
                # Outcome.
                "exposes_pair": bool(touched and overlap and not inventory_only),
            })

    compromised = sorted({
        origin
        for row in rows
        if row.get("exposes_pair")
        for origin in row.get("held_out_origins_named", ())
    })
    clean = [o for o in PROPOSED_HELD_OUT if o not in compromised]
    unreadable = [row for row in rows if "unreadable" in row]
    return {
        "stage": "P4T_EXPOSURE_LEDGER",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_EXPOSURE_AUDIT",
        "unit_of_exposure": "(series, origin) pair, not the origin alone",
        "reads": "which cells were evaluated; never an error, gain or utility",
        "boundary": {
            "llm_calls": 0, "consumer_fits": 0, "held_out_reads": 0,
            "outcome_values_read": 0,
        },
        "target_slice": "readable[%d:%d]" % TARGET_SLICE,
        "target_series": target,
        "proposed_held_out": list(PROPOSED_HELD_OUT),
        "artifacts_scanned": sum(
            1 for path in ARTIFACTS.rglob("*")
            if path.suffix in {".json", ".npz"} and path.name not in SELF
        ),
        "artifacts_naming_target_or_held_out": rows,
        "artifacts_naming_both": [row for row in rows if row.get("both")],
        "artifacts_exposing_a_pair": [
            row for row in rows if row.get("exposes_pair")],
        "inventory_only_artifacts": [
            row["artifact"] for row in rows if row.get("inventory_only")],
        "unreadable_artifacts": unreadable,
        "origins_disqualified": compromised,
        "origins_clear": clean,
        "verdict": (
            "ALL_PROPOSED_HELD_OUT_PAIRS_UNEXPOSED"
            if not compromised and not unreadable
            else "HELD_OUT_BLOCK_MUST_BE_NARROWED"
        ),
        "releases": "NONE",
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("artifacts scanned            : %d" % report["artifacts_scanned"])
    print("naming a Target series       : %d" % sum(
        1 for row in report["artifacts_naming_target_or_held_out"]
        if row.get("target_series_named")))
    print("naming a held-out origin     : %d" % sum(
        1 for row in report["artifacts_naming_target_or_held_out"]
        if row.get("held_out_origins_named")))
    print("naming BOTH                  : %d (%d are inventory-only)" % (
        len(report["artifacts_naming_both"]),
        sum(1 for row in report["artifacts_naming_both"]
            if row.get("inventory_only"))))
    print("EXPOSING a pair              : %d" % len(
        report["artifacts_exposing_a_pair"]))
    for row in report["artifacts_exposing_a_pair"][:8]:
        print("   %s -> %s @ %s" % (
            row["artifact"], row["target_series_named"][:4],
            row["held_out_origins_named"]))
    print("origins clear                : %s" % report["origins_clear"])
    print("verdict                      : %s" % report["verdict"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
