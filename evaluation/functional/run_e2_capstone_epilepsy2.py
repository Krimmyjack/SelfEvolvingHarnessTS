"""CAP-1 Epilepsy2 capstone runner (PREP-1 skeleton + exam-book unseal).

Every numeric constant is read from ``cap1_capstone_protocol_freeze.json``.
``--smoke-synthetic`` builds TRAIN/TEST in memory, uses the scripted backend,
and never touches D3.

The exam book adds what PREP-1 deliberately left unbuilt, and replaces the two
things CAP-1b replaced:

* **CAP-1 §7 unlock is void.**  Its condition -- S1-v2 forward x2 plus a
  reverse confirmation -- became literally unsatisfiable when S1-v2 retired on
  its third ``TREATMENT_EMPTY``.  ``cap1b_capstone_unlock_amendment.md`` is its
  only legal replacement and this runner reads the authorization chain off the
  artifacts instead: CAP-1b present, and SA-1 r2 recording exit **A** with all
  three mechanism gates.  The old §7 validator is kept and still evaluated, so
  the artifact shows exactly what the replacement changed.
* **CAP-1 §3 A5 pool is replaced.**  The "S1-v2 forward terminal pool" retired
  with the line.  A5 is the same K0 origin plus the SA-1 scope-v2 single-Episode
  supply card (v0 form, supply-only authority) with the R1-R3 revision loop
  open, exactly as SA-1 r1/r2 ran it.

Everything else in CAP-1 is executed to the letter and read from the freeze.

  python evaluation/functional/run_e2_capstone_epilepsy2.py --smoke-synthetic
  python evaluation/functional/run_e2_capstone_epilepsy2.py --seal-check
  python evaluation/functional/run_e2_capstone_epilepsy2.py --exam
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
CAP1_JSON = E2 / "cap1_capstone_protocol_freeze.json"
CAP0_JSON = E2 / "cap0_epilepsy2_capstone_freeze.json"
CAP1B_MD = E2 / "cap1b_capstone_unlock_amendment.md"
SA1_R2_JSON = E2 / "sa1_minimal_r2.json"
SA1_GATES_JSON = E2 / "sa1_minimal_gates.json"
OUT_JSON = E2 / "capstone_epilepsy2_final.json"
OUT_MD = E2 / "capstone_epilepsy2_final.md"
UNSEAL_RECORD = E2 / "capstone_epilepsy2_unseal_record.json"
CHECKPOINT = E2 / "capstone_epilepsy2_final.checkpoint.json"
DEFAULT_UNLOCK = E2 / "cap1_unseal_record.json"
SIGNAL = "S1V2_FORWARD_SIGNAL"
CAP1B_REQUIRED_EXIT = "A"
SA1_CARD_ID = "sa1_supply_scope_v2"
D3_MARKERS = ("D3_reserve", "EpilepticSeizures.zip", "Epilepsy2.zip",
              "EpilepticSeizures.ts")
TRAIN_MEMBER = "EpilepticSeizures/EpilepticSeizures_TRAIN.ts"
TEST_MEMBER = "EpilepticSeizures/EpilepticSeizures_TEST.ts"
EPILEPSY2_UNIT_ID = "Epilepsy2__impulse_v2_capstone"

_D3_TOUCHES: list[str] = []
_UNSEALED = {"open": False}


class Locked(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def load_cap1() -> dict[str, Any]:
    return json.loads(CAP1_JSON.read_text(encoding="utf-8"))


def load_cap0() -> dict[str, Any]:
    return json.loads(CAP0_JSON.read_text(encoding="utf-8"))


def d3_zip_declared_path(cap0: Mapping[str, Any] | None = None) -> Path:
    payload = cap0 or load_cap0()
    declared = None
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            if node.get("zip_path"):
                declared = str(node["zip_path"])
                break
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    if not declared:
        declared = "data/ucr_conf_downloaded/D3_reserve/EpilepticSeizures.zip"
    return PROJECT_ROOT / declared


def _is_d3_path(path: Any) -> bool:
    text = str(path).replace("\\", "/")
    return any(marker in text for marker in D3_MARKERS)


def _record_d3_touch(path: Any) -> None:
    _D3_TOUCHES.append(str(path))


def refuse_d3_open(path: Any) -> None:
    """Refuse any D3 path until an unseal has actually been authorised.

    ``_UNSEALED`` is set in exactly one place -- ``unseal`` -- after the CAP-1b
    chain and the CAP-0 seal record have both been re-checked, so the smoke
    path and every pre-unseal self-test still get the hard refusal they were
    written to assert.
    """
    if _is_d3_path(path):
        _record_d3_touch(path)
        if not _UNSEALED["open"]:
            raise Locked("D3 zip is sealed; runner refused to open %s" % path)


# =========================================================================== #
# CAP-1b: the replacement for CAP-1 section 7
# =========================================================================== #
def validate_unlock_cap1b() -> tuple[bool, str, dict[str, Any]]:
    """Re-derive the unseal authorization from the artifacts it rests on.

    Three facts, each read rather than asserted: the amendment exists, SA-1 r2
    recorded CAP-1b exit A, and its three mechanism gates all held.  Nothing
    here is a judgement call -- if any of them reads otherwise the seal stays
    shut.
    """
    evidence: dict[str, Any] = {
        "amendment": CAP1B_MD.relative_to(PROJECT_ROOT).as_posix(),
        "amendment_present": CAP1B_MD.is_file(),
        "sa1_r2": SA1_R2_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "sa1_r2_present": SA1_R2_JSON.is_file(),
        "required_exit": CAP1B_REQUIRED_EXIT,
    }
    if not evidence["amendment_present"]:
        return False, "CAP-1b amendment file is absent", evidence
    if not evidence["sa1_r2_present"]:
        return False, "SA-1 r2 artifact is absent", evidence
    r2 = json.loads(SA1_R2_JSON.read_text(encoding="utf-8"))
    cap1b = r2.get("cap1b") or {}
    decision = cap1b.get("exit_decision") or {}
    gates = {name: (cap1b.get(name) or {}).get("pass") for name in
             ("G0", "G1", "G2")}
    evidence.update({
        "sa1_r2_exit": decision.get("exit"),
        "sa1_r2_gates": gates,
        "sa1_r2_capstone_shape": decision.get("capstone_shape"),
        "sa1_r2_verdict": (r2.get("verdict") or {}).get("verdict"),
    })
    if str(decision.get("exit")) != CAP1B_REQUIRED_EXIT:
        return False, ("SA-1 r2 recorded exit %s, this book executes exit %s"
                       % (decision.get("exit"), CAP1B_REQUIRED_EXIT)), evidence
    if not all(gates.values()):
        return False, "SA-1 r2 mechanism gates did not all hold: %s" % gates, \
            evidence
    return True, ("CAP-1b amendment present; SA-1 r2 exit A with G0/G1/G2 all "
                  "holding"), evidence


def unlock_readings() -> dict[str, Any]:
    """Both readings side by side: the void section 7 and its replacement."""
    legacy_ok, legacy_reason, _legacy = validate_unlock(DEFAULT_UNLOCK)
    new_ok, new_reason, evidence = validate_unlock_cap1b()
    return {
        "cap1_section7_status": "VOID",
        "cap1_section7_void_because": (
            "its condition was S1-v2 forward x2 plus a reverse confirmation; "
            "S1-v2 retired under its own final-throw cap after a third "
            "TREATMENT_EMPTY, so the condition is literally unsatisfiable and "
            "had to be replaced rather than quietly reinterpreted"),
        "cap1_section7_would_say": {"ok": legacy_ok, "reason": legacy_reason},
        "replacement": "artifacts/functional/e2/"
                       "cap1b_capstone_unlock_amendment.md",
        "replacement_authorised_by": (
            "CAP-1b (frozen before r2 results existed) + the mainline "
            "adjudication entry of 2026-08-28 13:5x + sol's ruling that one "
            "r2 is followed by the capstone regardless of outcome"),
        "replacement_says": {"ok": new_ok, "reason": new_reason},
        "evidence": evidence,
        "unlocked": bool(new_ok),
    }


def validate_unlock(path: Path | None) -> tuple[bool, str, dict[str, Any]]:
    if path is None or not path.is_file():
        return False, "no unseal record file", {}
    record = json.loads(path.read_text(encoding="utf-8"))
    forwards = list(record.get("forward") or record.get("forward_runs") or [])
    reverse = record.get("reverse") or record.get("reverse_run") or {}
    if len(forwards) < 2:
        return False, "unseal record missing S1-v2 forward x2", record
    for index, row in enumerate(forwards[:2], start=1):
        if str(row.get("verdict")) != SIGNAL:
            return False, (
                "forward run %s verdict is %s, need %s"
                % (index, row.get("verdict"), SIGNAL)), record
        if not row.get("artifact"):
            return False, "forward run %s missing artifact path" % index, record
    if str(reverse.get("verdict")) != SIGNAL:
        return False, (
            "reverse confirmation verdict is %s, need %s"
            % (reverse.get("verdict"), SIGNAL)), record
    if not reverse.get("artifact"):
        return False, "reverse confirmation missing artifact path", record
    return True, "unseal record lists forward x2 SIGNAL and reverse SIGNAL", record


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def lock_reason(path: Path | None = None) -> str:
    target = path or DEFAULT_UNLOCK
    ok, reason, _record = validate_unlock(target)
    if ok:
        return "unlocked: %s" % reason
    return (
        "CAPSTONE LOCKED: D3 stays sealed. %s. Required: S1-v2 forward "
        "x2 both %s and reverse confirmation %s, each with an artifact "
        "path, written to %s."
        % (reason, SIGNAL, SIGNAL, _display_path(target))
    )


def require_unseal(path: Path | None = None) -> dict[str, Any]:
    ok, reason, record = validate_unlock(path or DEFAULT_UNLOCK)
    if not ok:
        raise Locked(lock_reason(path or DEFAULT_UNLOCK))
    return record


def load_official_d3(*, unlock_path: Path | None = None
                     ) -> tuple[Any, Any, Any, Any]:
    """PREP-1's placeholder.  The exam book uses ``unseal`` instead.

    Left in place because the smoke self-test asserts that it refuses without
    an unseal record, which is still the behaviour that matters for anyone
    running the pre-unseal drill.
    """
    require_unseal(unlock_path)
    zip_path = d3_zip_declared_path()
    refuse_d3_open(zip_path)
    raise Locked(
        "unseal record is valid but this PREP-1 entry point does not parse D3 "
        "bytes; the exam book's unseal() does"
    )


# =========================================================================== #
# seal re-check and the one authorised open
# =========================================================================== #
def verify_seal() -> dict[str, Any]:
    """Re-check every fact CAP-0 recorded, before anything is parsed.

    Structure only: member names, uncompressed member sizes, raw newline
    counts, and the count of records after the ``@data`` marker.  No token is
    split and no label is read here, which is the same boundary CAP-0 declared.

    One honest gap, stated rather than papered over: **CAP-0 recorded no
    sha256 of the zip.**  Its ``SEAL_INTACT`` rested on the ROSTER byte count,
    the member listing, the member sizes and the row counts, all of which are
    re-checked here and all of which match.  This book computes the digest for
    the first time and records it as the baseline a future book can compare
    against; the comparison itself could not be performed because there was
    nothing recorded to compare to.
    """
    import hashlib
    from zipfile import ZipFile

    cap0 = load_cap0()
    structural = cap0["part2_structural"]
    roster = ((cap0.get("part1_seal_audit") or {}).get("roster") or {})
    declared_bytes = int((roster.get("table_row") or {}).get("bytes") or 0)
    zip_path = d3_zip_declared_path(cap0)
    checks: list[dict[str, Any]] = []

    def _check(name: str, ok: bool, expected: Any, observed: Any,
               note: str | None = None) -> None:
        checks.append({"check": name, "pass": bool(ok), "expected": expected,
                       "observed": observed, "note": note})

    _check("zip file present", zip_path.is_file(), True, zip_path.is_file())
    if not zip_path.is_file():
        return {"verdict": "SEAL_FILE_ABSENT", "pass": False,
                "checks": checks,
                "zip_path": _display_path(zip_path)}
    size = zip_path.stat().st_size
    _check("zip byte count matches the ROSTER record",
           size == declared_bytes, declared_bytes, size)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    with ZipFile(zip_path) as archive:
        names = list(archive.namelist())
        sizes = {info.filename: int(info.file_size)
                 for info in archive.infolist()}
        train_raw = archive.read(TRAIN_MEMBER)
        test_raw = archive.read(TEST_MEMBER)
    _check("member listing matches CAP-0",
           names == list(structural["members_observed"]),
           list(structural["members_observed"]), names)
    for label, member, block in (("TRAIN", TRAIN_MEMBER, structural["train"]),
                                 ("TEST", TEST_MEMBER, structural["test"])):
        _check("%s member uncompressed size matches CAP-0" % label,
               sizes.get(member) == int(block["bytes"]),
               int(block["bytes"]), sizes.get(member))
    _check("val.ts size matches CAP-0",
           sizes.get("EpilepticSeizures/val.ts")
           == int(structural["val_ts"]["file_size"]),
           int(structural["val_ts"]["file_size"]),
           sizes.get("EpilepticSeizures/val.ts"))
    for label, raw, block in (("TRAIN", train_raw, structural["train"]),
                              ("TEST", test_raw, structural["test"])):
        newlines = raw.count(b"\n")
        rows = _count_data_records(raw)
        _check("%s raw newline count matches CAP-0" % label,
               newlines == int(block["file_newlines"]),
               int(block["file_newlines"]), newlines)
        _check("%s data-record count matches CAP-0" % label,
               rows == int(block["data_record_lines"]),
               int(block["data_record_lines"]), rows)
    expected = structural["expected"]
    _check("CAP-0 verdict on record is MATCH",
           str(structural.get("verdict")) == "MATCH", "MATCH",
           structural.get("verdict"))
    passed = all(row["pass"] for row in checks)
    return {
        "verdict": "SEAL_INTACT" if passed else "STRUCTURAL_MISMATCH",
        "pass": passed,
        "checks": checks,
        "zip_path": _display_path(zip_path),
        "zip_bytes": size,
        "zip_sha256_computed_now": digest,
        "zip_sha256_recorded_by_cap0": None,
        "zip_sha256_gap": (
            "CAP-0 recorded the ROSTER byte count, the member listing, the "
            "member sizes and the row counts, but no sha256 of the zip.  That "
            "one comparison therefore could not be made; the digest above is "
            "computed here for the first time and is the baseline for any "
            "future check.  Every fact CAP-0 did record was re-checked and "
            "matches."),
        "expected_shape": {"train_rows": int(expected["train_rows"]),
                           "test_rows": int(expected["test_rows"]),
                           "length": int(expected["length"])},
        "boundary_this_function_respected": (
            "member names, member sizes, raw newline counts, and records after "
            "the @data marker.  No float parse, no label read"),
    }


def _count_data_records(raw: bytes) -> int:
    """Records after the exact ``@data`` marker.  CAP-0's own rule."""
    lines = raw.split(b"\n")
    seen = False
    count = 0
    for line in lines:
        stripped = line.strip()
        if not seen:
            if stripped.lower() == b"@data":
                seen = True
            continue
        if stripped:
            count += 1
    return count


def _parse_ts(raw: bytes, *, length: int) -> tuple[Any, Any]:
    """aeon univariate ``.ts`` -> (values, raw labels).

    One record per line after ``@data``, comma-separated observations then
    ``:label``.  Anything that does not have exactly ``length`` observations
    is a hard error rather than a skip, because a silently dropped row would
    change the frozen row indices underneath the exam.
    """
    body = raw.split(b"\n")
    seen = False
    rows: list[list[float]] = []
    labels: list[str] = []
    for number, line in enumerate(body, start=1):
        stripped = line.strip()
        if not seen:
            if stripped.lower() == b"@data":
                seen = True
            continue
        if not stripped:
            continue
        text = stripped.decode("utf-8")
        if ":" not in text:
            raise Locked("ts record %d carries no ':label'" % number)
        observations, label = text.rsplit(":", 1)
        parts = [item for item in observations.split(",") if item != ""]
        if len(parts) != int(length):
            raise Locked("ts record %d has %d observations, expected %d"
                         % (number, len(parts), int(length)))
        rows.append([float(item) for item in parts])
        labels.append(label.strip())
    if not seen:
        raise Locked("ts member has no @data marker")
    return np.asarray(rows, dtype=np.float64), np.asarray(labels)


def _normalise_like_ucr(values: Any, raw_labels: Any) -> tuple[Any, Any]:
    """The same post-processing every other Target in this line receives.

    ``run_e2_task_context_label_evidence_witness._load_split`` maps the label
    column to ``{0, 1}`` by sorted unique value and z-normalises each row.
    Only the file-format front end differs here (``.ts`` rather than the UCR
    ``.txt`` table), so the arrays the Consumer and the operators see are on
    the same footing as every other unit's.
    """
    label_values = sorted(set(str(value) for value in raw_labels.tolist()))
    if len(label_values) != 2:
        raise Locked("capstone requires a binary Target; found %d classes"
                     % len(label_values))
    labels = np.asarray([label_values.index(str(value))
                         for value in raw_labels.tolist()], dtype=np.int64)
    matrix = np.asarray(values, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise Locked("Epilepsy2 contains a non-finite observation")
    scale = np.std(matrix, axis=1, keepdims=True)
    if bool(np.any(scale <= 1e-8)):
        raise Locked("Epilepsy2 contains a degenerate row")
    normalised = (matrix - np.mean(matrix, axis=1, keepdims=True)) / scale
    return normalised, labels


def unseal() -> dict[str, Any]:
    """The one authorised open.  Records what it did before it did it."""
    import time as _time
    from zipfile import ZipFile

    unlock = unlock_readings()
    if not unlock["unlocked"]:
        raise Locked("unseal refused: %s"
                     % unlock["replacement_says"]["reason"])
    seal = verify_seal()
    if not seal["pass"]:
        raise Locked("unseal refused: seal re-check said %s"
                     % seal["verdict"])
    const = _constants(load_cap1())
    zip_path = d3_zip_declared_path()
    record = {
        "unsealed_at_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          _time.gmtime()),
        "zip_path": _display_path(zip_path),
        "zip_bytes": seal["zip_bytes"],
        "zip_sha256": seal["zip_sha256_computed_now"],
        "authorised_by": unlock["replacement_authorised_by"],
        "first_read_scope": (
            "%s and %s, parsed in full to float arrays; val.ts, "
            "EpilepticSeizures.txt and the .png were not opened"
            % (TRAIN_MEMBER, TEST_MEMBER)),
        "seal_verdict": seal["verdict"],
    }
    _UNSEALED["open"] = True
    with ZipFile(zip_path) as archive:
        train_values, train_raw_labels = _parse_ts(
            archive.read(TRAIN_MEMBER), length=int(const["length"]))
        test_values, test_raw_labels = _parse_ts(
            archive.read(TEST_MEMBER), length=int(const["length"]))
    train, train_labels = _normalise_like_ucr(train_values, train_raw_labels)
    test, test_labels = _normalise_like_ucr(test_values, test_raw_labels)
    if train.shape != (int(const["n_train"]), int(const["length"])):
        raise Locked("TRAIN shape %s is not the frozen %s"
                     % (train.shape,
                        (const["n_train"], const["length"])))
    if test.shape != (int(const["n_test_official"]), int(const["length"])):
        raise Locked("TEST shape %s is not the frozen %s"
                     % (test.shape,
                        (const["n_test_official"], const["length"])))
    record.update({
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "classes_train": sorted(int(v) for v in set(train_labels.tolist())),
        "classes_test": sorted(int(v) for v in set(test_labels.tolist())),
        "class_counts_train": {
            str(label): int((train_labels == label).sum())
            for label in sorted(set(train_labels.tolist()))},
        "normalisation": ("row z-norm and sorted-unique label mapping, the "
                          "same post-processing _load_split applies to every "
                          "other Target in this line"),
    })
    UNSEAL_RECORD.write_text(json.dumps(
        {"unseal_record": record, "unlock": unlock, "seal": seal},
        ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return {"record": record, "unlock": unlock, "seal": seal,
            "train_values": train, "train_labels": train_labels,
            "test_values": test, "test_labels": test_labels}


def load_terminal_pool(path: Path | None) -> list[dict[str, Any]]:
    """Parameterized S1-v2 forward terminal-pool loader."""
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping):
        for key in ("skills", "entries", "terminal_pool", "cards"):
            if isinstance(payload.get(key), list):
                return [dict(item) for item in payload[key]]
    return []


def _constants(cap1: Mapping[str, Any]) -> dict[str, Any]:
    sampling = cap1["sampling"]
    half = cap1["train_half_slice"]
    budget = cap1["budget"]
    verdicts = cap1["verdicts"]
    arms = cap1["arms"]
    return {
        "n_train": int(sampling["frozen_n_train"]),
        "n_test_official": int(sampling["algorithm"]["N"]),
        "n_test_subset": int(sampling["frozen_n_test_subset"]),
        "length": int(sampling["frozen_length"]),
        "seed": int(sampling["algorithm"]["seed"]),
        "test_indices": list(sampling["test_row_indices_sorted"]),
        "support_idx": list(half["support"]["indices"]),
        "delayed_idx": list(half["delayed"]["indices"]),
        "rounds": int(half["rounds"]),
        "llm_per_arm": int(budget["llm_per_arm_max"]),
        "fit_per_arm": int(budget["fit_per_arm_max"]),
        "wall_minutes": int(budget["wall_clock_total_minutes_max"]),
        "material": float(verdicts["MATERIAL"]),
        "heldout_line": float(verdicts["heldout_material_line"]),
        "harm_bar": float(verdicts["HARM_BAR"]),
        "worst_tol": float(verdicts["worst_class_noninferior_tol"]),
        "arm_names": list(arms),
        "consumer": cap1["consumer_metric_menu"],
        "condition": cap1["scope_and_initial_state"]["injection_condition"],
    }


def _synthetic_arrays(const: Mapping[str, Any]
                      ) -> dict[str, Any]:
    """In-memory 80x178 TRAIN + 11420x178 TEST.  Not written.  Not D3."""
    rng = np.random.RandomState(int(const["seed"]))
    n_train = int(const["n_train"])
    n_test = int(const["n_test_official"])
    length = int(const["length"])
    train = rng.normal(0.0, 1.0, size=(n_train, length)).astype(np.float64)
    test = rng.normal(0.0, 1.0, size=(n_test, length)).astype(np.float64)
    train_y = np.asarray([index % 2 for index in range(n_train)], dtype=int)
    test_y = np.asarray([index % 2 for index in range(n_test)], dtype=int)
    # A tiny deterministic pulse so a scripted probe has geometry, not a leak.
    train[0::2, 10:14] += 4.0
    return {
        "train_values": train, "train_labels": train_y,
        "test_values": test, "test_labels": test_y,
        "materialized_in_memory": True,
        "written_to_disk": False,
        "d3_touched": False,
        "shape_train": list(train.shape),
        "shape_test": list(test.shape),
    }


def _cell_from_arrays(const: Mapping[str, Any],
                      arrays: Mapping[str, Any]) -> dict[str, Any]:
    train = np.asarray(arrays["train_values"])
    labels = np.asarray(arrays["train_labels"])
    test = np.asarray(arrays["test_values"])
    test_y = np.asarray(arrays["test_labels"])
    support_idx = [int(i) for i in const["support_idx"]]
    delayed_idx = [int(i) for i in const["delayed_idx"]]
    test_idx = [int(i) for i in const["test_indices"]]
    length = int(const["length"])
    rows_in_window = min(4, int(train.shape[0]))
    return {
        "dataset": "__prep1_synthetic__",
        "condition": const["condition"],
        "data_dir": "__never_a_real_data_dir__",
        "series_length": length,
        "fit_values": train,
        "fit_labels": labels,
        "surfaces": {
            "r1_support": (train[support_idx], labels[support_idx]),
            "r1_delayed": (train[delayed_idx], labels[delayed_idx]),
        },
        "slice_rows": {"r1_support": len(support_idx),
                       "r1_delayed": len(delayed_idx)},
        "heldout_values": test[test_idx],
        "heldout_labels": test_y[test_idx],
        "observation_rows": rows_in_window,
        "observation_block": np.asarray(
            train[:rows_in_window], dtype=np.float64).ravel(),
        "support_reproduces_fit_signal": False,
        "capstone_synthetic": True,
    }


# =========================================================================== #
# the exam cell: CAP-1's frozen splits, the course's injection template
# =========================================================================== #
def _epilepsy2_cell(const: Mapping[str, Any],
                    data: Mapping[str, Any]) -> dict[str, Any]:
    """TRAIN 80 rows fit, mod-4 quarters concatenated per M-1, TEST 476 clean.

    Three frozen things meet here and none of them is re-derived:

    * the quarters are CAP-1 section 2's row-index rule, ``i % 4`` over all 80
      TRAIN rows, **not** the live label-stratified split -- the freeze rejects
      re-stratifying by label at unseal in as many words;
    * Support and delayed are the M-1 one-round role concat, so the dual gate
      survives and each surface carries 40 rows at a 0.025 material line;
    * the injection is the course's own template -- ``cls._inject_v2`` at the
      positions ``helpers["positions"]`` returns -- under condition
      ``fit_only_artifact``, which means the fit cohort carries the artifact
      and the held-in surfaces and the official TEST subset stay clean.
    """
    import run_e2_m1_margin_gate as m1
    import run_e2_t6_cls_op_shared_harness as cls
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _helpers,
    )

    _ctx, helpers = _helpers()
    train = np.asarray(data["train_values"], dtype=np.float64)
    labels = np.asarray(data["train_labels"])
    test = np.asarray(data["test_values"], dtype=np.float64)
    test_labels = np.asarray(data["test_labels"])
    length = int(const["length"])
    positions = tuple(int(p) for p in helpers["positions"](length))

    # fit_only_artifact: the artifact is written into the cohort the Consumer
    # is fitted on, and nowhere else.
    fit_values = cls._inject_v2(np, train, labels, positions)
    quarters = {name: [index for index in range(int(train.shape[0]))
                       if index % 4 == offset]
                for offset, name in enumerate(("r1_support", "r1_delayed",
                                               "r2_support", "r2_delayed"))}
    clean = {name: (train[rows], labels[rows])
             for name, rows in quarters.items()}
    support = m1._concat_surface(clean["r1_support"], clean["r2_support"])
    delayed = m1._concat_surface(clean["r1_delayed"], clean["r2_delayed"])
    frozen_support = [int(i) for i in const["support_idx"]]
    frozen_delayed = [int(i) for i in const["delayed_idx"]]
    got_support = sorted(quarters["r1_support"] + quarters["r2_support"])
    got_delayed = sorted(quarters["r1_delayed"] + quarters["r2_delayed"])
    if got_support != sorted(frozen_support) or got_delayed != sorted(
            frozen_delayed):
        raise Locked("recomputed half-protocol indices do not match the "
                     "frozen CAP-1 table")

    test_idx = [int(i) for i in const["test_indices"]]
    rows_in_window = max(4, -(-int(cls.OBSERVATION_POINTS) // length))
    rows_in_window = min(rows_in_window, int(fit_values.shape[0]))
    return {
        "dataset": "Epilepsy2",
        "condition": str(const["condition"]),
        "data_dir": None,
        "series_length": length,
        "official_train_rows": int(train.shape[0]),
        "fit_rows": int(fit_values.shape[0]),
        "fit_values": fit_values,
        "fit_labels": labels,
        "surfaces": {"r1_support": support, "r1_delayed": delayed},
        "slice_rows": {"r1_support": int(support[0].shape[0]),
                       "r1_delayed": int(delayed[0].shape[0])},
        "quarter_rows": {name: len(rows) for name, rows in quarters.items()},
        "controlled_impulse_positions": list(positions),
        "heldout_values": test[test_idx],
        "heldout_labels": test_labels[test_idx],
        "heldout_rows": len(test_idx),
        "observation_rows": rows_in_window,
        "observation_block": np.asarray(
            fit_values[:rows_in_window], dtype=np.float64).ravel(),
        "injection_template": cls.INJECTION_TEMPLATE_V2,
        "injection_segment_length": cls._v2_segment_length(length),
        "capstone": True,
    }


def _install_exam_heldout(cell: Mapping[str, Any]) -> Any:
    """Serve the frozen 476-row clean TEST subset, and only that."""
    import run_e2_t6_cls_op_shared_harness as cls

    values = np.asarray(cell["heldout_values"])
    labels = np.asarray(cell["heldout_labels"])
    original = cls._heldout_surface

    def _patched(dataset: str, condition: str, data_dir: str | None = None):
        if str(dataset) != str(cell["dataset"]):
            raise Locked("exam heldout patch refused dataset %s" % dataset)
        return values, labels

    cls._heldout_surface = _patched  # type: ignore[method-assign]
    return original


# =========================================================================== #
# CAP-1b's pre-declared dedup instrument (evaluation layer only)
# =========================================================================== #
def dedup_swallowed(record: Mapping[str, Any], *, skill_id: str,
                    card_operators: Sequence[str]) -> dict[str, Any]:
    """Did a matching card's supply get eaten by the agent's own proposal?

    SA-1 r1's P4 found a third path nobody had written down: the card matched,
    the card was in view, and yet no ``cand_skill_`` entry reached the pool,
    because the Fast agent had already named the same frozen program and the
    candidate pool deduplicated the mechanical supply against it.  Without a
    mark for that case, "the card was not supplied" and "the card was refused"
    look identical in the ledger, and supply attribution is unreadable.

    Pure derivation from fields the round record already carries -- the Scope
    verdict, the pool, and each proposal's operators.  Nothing in ``methods/``
    is touched and nothing here changes what is proposed or probed.
    """
    candidate_id = "cand_skill_%s" % skill_id
    pool = [str(item) for item in (record.get("pool") or ())]
    scope_match = bool((record.get("scope_match_by_skill_id") or {})
                       .get(skill_id))
    in_view = skill_id in [str(item) for item
                           in (record.get("retrieved_skill_ids") or ())]
    supplied_in_pool = candidate_id in pool
    wanted = sorted(str(op) for op in card_operators)
    same_program = [
        {"candidate_id": str(row.get("candidate_id")),
         "operators": list(row.get("operators") or ())}
        for row in (record.get("proposals") or ())
        if not str(row.get("candidate_id", "")).startswith("cand_skill_")
        and sorted(str(op) for op in (row.get("operators") or ())) == wanted
    ]
    swallowed = bool(scope_match and in_view and not supplied_in_pool
                     and same_program)
    return {
        "skill_id": skill_id,
        "scope_match": scope_match,
        "card_in_view": in_view,
        "supplied_in_pool": supplied_in_pool,
        "self_proposed_same_program": same_program,
        "dedup_swallowed": swallowed,
        "why": (
            "the card matched and was in view, no cand_skill_ entry reached "
            "the pool, and the agent had proposed the same frozen program "
            "itself" if swallowed else
            "not the dedup case: " + ("Scope did not match" if not scope_match
                                      else "the card was not in view"
                                      if not in_view
                                      else "the supply did reach the pool"
                                      if supplied_in_pool
                                      else "no self-proposed candidate carries "
                                           "the card's program")),
    }


def annotate_candidate_sources(result: Mapping[str, Any], *, skill_id: str,
                               card_operators: Sequence[str]
                               ) -> dict[str, Any]:
    """Per round: supplied / self-proposed / dedup_swallowed, split out."""
    rows = []
    for record in (result.get("rounds") or ()):
        mark = dedup_swallowed(record, skill_id=skill_id,
                               card_operators=card_operators)
        candidate_id = "cand_skill_%s" % skill_id
        proposals = list(record.get("proposals") or ())
        rows.append({
            "round": record.get("round"),
            "pool": [str(item) for item in (record.get("pool") or ())],
            "supplied": sum(1 for row in proposals
                            if str(row.get("candidate_id")) == candidate_id),
            "self_proposed": sum(
                1 for row in proposals
                if not str(row.get("candidate_id", "")).startswith(
                    "cand_skill_")
                and str(row.get("candidate_id")) != "identity"),
            "identity": sum(1 for row in proposals
                            if str(row.get("candidate_id")) == "identity"),
            **mark,
        })
    return {
        "per_round": rows,
        "supplied_total": sum(int(row["supplied"]) for row in rows),
        "self_proposed_total": sum(int(row["self_proposed"]) for row in rows),
        "dedup_swallowed_total": sum(1 for row in rows
                                     if row["dedup_swallowed"]),
        "scope_matched_any_round": any(row["scope_match"] for row in rows),
    }


def _score_capstone(results: Mapping[str, Mapping[str, Any]],
                    const: Mapping[str, Any]) -> dict[str, Any]:
    def _acc(arm: str) -> float:
        dep = (results.get(arm) or {}).get("deployment") or {}
        return float(dep.get("heldout_accuracy")
                     or dep.get("heldout_accuracy_gain") or 0.0)

    def _worst(arm: str) -> float:
        dep = (results.get(arm) or {}).get("deployment") or {}
        deltas = dep.get("heldout_recall_delta_by_class") or {}
        if not deltas:
            return 0.0
        return min(float(v) for v in deltas.values())

    def _harm(arm: str) -> bool:
        return _worst(arm) < -float(const["harm_bar"])

    a5, a3, static = _acc("A5"), _acc("A3"), _acc("Static")
    line = float(const["heldout_line"])
    worst_tol = float(const["worst_tol"])
    delta_a5_a3 = a5 - a3
    worst_delta = _worst("A5") - _worst("A3")
    harm_a5 = _harm("A5")
    positive = (delta_a5_a3 >= line
                and worst_delta >= worst_tol
                and not harm_a5)
    negative = (delta_a5_a3 <= -line
                or worst_delta < worst_tol
                or harm_a5)
    if positive:
        verdict = "CAPSTONE_POSITIVE"
    elif negative:
        verdict = "CAPSTONE_NEGATIVE"
    else:
        verdict = "CAPSTONE_NEUTRAL"
    return {
        "verdict": verdict,
        "acc": {"Static": static, "A3": a3, "A5": a5},
        "delta_a5_minus_a3": delta_a5_a3,
        "worst_class_delta_a5_minus_a3": worst_delta,
        "harm_a5": harm_a5,
        "heldout_material_line": line,
        "attribution": {
            "a5_minus_static": a5 - static,
            "a3_minus_static": a3 - static,
        },
    }


def _install_heldout_patch(cell: Mapping[str, Any]) -> Any:
    import run_e2_t6_cls_op_shared_harness as cls

    values = np.asarray(cell["heldout_values"])
    labels = np.asarray(cell["heldout_labels"])
    original = cls._heldout_surface

    def _patched(dataset: str, condition: str, data_dir: str | None = None):
        refuse_d3_open(dataset)
        refuse_d3_open(data_dir or "")
        if str(dataset) == "__prep1_synthetic__":
            return values, labels
        raise Locked("smoke heldout patch refused non-synthetic dataset %s"
                     % dataset)

    cls._heldout_surface = _patched  # type: ignore[method-assign]
    return original


def _restore_heldout(original: Any) -> None:
    import run_e2_t6_cls_op_shared_harness as cls
    cls._heldout_surface = original  # type: ignore[method-assign]


def _run_arm(*, arm: str, cell: Mapping[str, Any], snapshot: Any,
             const: Mapping[str, Any], store_root: Path) -> dict[str, Any]:
    import run_e2_s1_curriculum_four_arms as s1
    import run_e2_t6_cls_op_shared_harness as cls

    arm_internal = {"Static": s1.ARM_STATIC, "A3": s1.ARM_A3,
                    "A5": s1.ARM_A5}[arm]
    backend = cls._scripted_backend(int(const["llm_per_arm"]))
    unit = {"unit_id": "prep1_synthetic_epilepsy2",
            "dataset": "__prep1_synthetic__",
            "injection": "none"}
    return s1.run_unit(
        unit=unit, cell=cell, arm=arm_internal, base_snapshot=snapshot,
        carried_episodes=(), agent_factory=cls._scripted_agent,
        backend=backend, store_root=store_root / arm,
        rounds=("r1",) if int(const["rounds"]) == 1 else ("r1", "r2"),
        fit_cap=int(const["fit_per_arm"]))


def _checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False,
                               default=str) + "\n", encoding="utf-8")


def _public(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items()
            if not str(key).startswith("_")}


def smoke_synthetic(*, pool_path: Path | None = None,
                    unlock_path: Path | None = None) -> dict[str, Any]:
    """Scripted three-arm end-to-end on in-memory arrays.  Zero CPA.  Zero D3."""
    import run_e2_s1_curriculum_four_arms as s1
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )

    _D3_TOUCHES.clear()
    cap1 = load_cap1()
    const = _constants(cap1)
    started = time.time()
    lock_selftest = _unlock_selftest(unlock_path)
    _D3_TOUCHES.clear()
    arrays = _synthetic_arrays(const)
    cell = _cell_from_arrays(const, arrays)
    original = _install_heldout_patch(cell)
    store_root = Path(tempfile.mkdtemp(prefix="cap1_smoke_"))
    checkpoint = store_root / "cap1_smoke.checkpoint.json"
    results: dict[str, Any] = {}
    resume_skipped: list[str] = []
    try:
        h0 = compile_snapshot(
            PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
            verify_lock=False)
        k0_pack = s1.compile_k0(store_root / "k0")
        pool = load_terminal_pool(pool_path)
        a5_base = k0_pack["k0"]
        a5_applied: list[str] = []
        if pool:
            a5_base, a5_applied = s1._apply_entries(
                k0_pack["k0"], pool, store_root=store_root / "bases",
                tag="a5_s1v2_pool")
        bases = {"Static": h0, "A3": h0, "A5": a5_base}
        # First pass: Static only, then checkpoint.
        results["Static"] = _public(_run_arm(
            arm="Static", cell=cell, snapshot=bases["Static"],
            const=const, store_root=store_root))
        _checkpoint(checkpoint, {
            "completed": ["Static"],
            "results": {"Static": results["Static"]},
        })
        # Resume drill: reload checkpoint and skip Static.
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        done = set(saved.get("completed") or [])
        for arm in ("A3", "A5"):
            if arm in done:
                resume_skipped.append(arm)
                results[arm] = saved["results"][arm]
                continue
            results[arm] = _public(_run_arm(
                arm=arm, cell=cell, snapshot=bases[arm],
                const=const, store_root=store_root))
            done.add(arm)
            _checkpoint(checkpoint, {
                "completed": sorted(done),
                "results": {name: results[name] for name in done},
            })
        score = _score_capstone(results, const)
        return {
            "ok": True,
            "mode": "smoke-synthetic",
            "backend": "scripted 0-CPA",
            "d3_touches": list(_D3_TOUCHES),
            "d3_touched": bool(_D3_TOUCHES),
            "arrays": {
                "shape_train": arrays["shape_train"],
                "shape_test": arrays["shape_test"],
                "written_to_disk": False,
            },
            "unlock_selftest": lock_selftest,
            "pool_path": (str(pool_path) if pool_path else None),
            "pool_applied": a5_applied,
            "k0_sha": k0_pack["k0_sha"],
            "h0_sha": k0_pack["h0_sha"],
            "checkpoint_resume": {
                "wrote_after_static": True,
                "skipped_on_resume": resume_skipped,
                "completed": ["Static", "A3", "A5"],
            },
            "arm_reads": {
                arm: {
                    "llm_calls": results[arm].get("llm_calls"),
                    "consumer_fits": results[arm].get("consumer_fits"),
                    "applied": [
                        step.get("op") for step in
                        ((results[arm].get("deployment") or {}).get(
                            "applied_program") or [])
                    ] or ["identity"],
                    "heldout_accuracy_gain": (
                        (results[arm].get("deployment") or {}).get(
                            "heldout_accuracy_gain")),
                } for arm in ("Static", "A3", "A5")
            },
            "score": score,
            "seconds": round(time.time() - started, 2),
            "constants_from_freeze": {
                "seed": const["seed"],
                "n_train": const["n_train"],
                "n_test_subset": const["n_test_subset"],
                "length": const["length"],
                "llm_per_arm": const["llm_per_arm"],
                "fit_per_arm": const["fit_per_arm"],
            },
        }
    finally:
        _restore_heldout(original)
        shutil.rmtree(store_root, ignore_errors=True)


def _unlock_selftest(unlock_path: Path | None) -> dict[str, Any]:
    missing_ok, missing_reason, _ = validate_unlock(None)
    default_ok, default_reason, _ = validate_unlock(unlock_path or DEFAULT_UNLOCK)
    incomplete = E2 / "_prep1_incomplete_unseal.json"
    incomplete.write_text(json.dumps({
        "forward": [{"verdict": SIGNAL, "artifact": "x.json"}],
        "reverse": {},
    }) + "\n", encoding="utf-8")
    inc_ok, inc_reason, _ = validate_unlock(incomplete)
    incomplete.unlink(missing_ok=True)
    complete = E2 / "_prep1_complete_unseal.json"
    complete.write_text(json.dumps({
        "forward": [
            {"run_id": "1", "verdict": SIGNAL,
             "artifact": "artifacts/functional/e2/s1v2_forward_run1.json"},
            {"run_id": "2", "verdict": SIGNAL,
             "artifact": "artifacts/functional/e2/s1v2_forward_run2.json"},
        ],
        "reverse": {
            "verdict": SIGNAL,
            "artifact": "artifacts/functional/e2/s1v2_reverse_run1.json",
        },
    }) + "\n", encoding="utf-8")
    comp_ok, comp_reason, _ = validate_unlock(complete)
    complete.unlink(missing_ok=True)
    load_refused = None
    try:
        load_official_d3(unlock_path=Path("__no_such_unseal__"))
        load_refused = False
    except Locked as exc:
        load_refused = True
        load_reason = exc.reason
    else:
        load_reason = "unexpectedly returned"
    # Direct zip open must also be refused.
    zip_refused = False
    try:
        refuse_d3_open(d3_zip_declared_path())
    except Locked:
        zip_refused = True
    return {
        "missing_record_locked": (not missing_ok),
        "missing_reason": missing_reason,
        "default_record_locked": (not default_ok),
        "default_reason": default_reason,
        "incomplete_record_locked": (not inc_ok),
        "incomplete_reason": inc_reason,
        "complete_record_accepted": bool(comp_ok),
        "complete_reason": comp_reason,
        "load_official_d3_without_unlock_refused": bool(load_refused),
        "load_official_d3_reason": load_reason,
        "declared_zip_open_refused": zip_refused,
        "d3_touches_during_selftest": list(_D3_TOUCHES),
        "ok": (not missing_ok) and (not inc_ok) and comp_ok
              and bool(load_refused) and zip_refused,
    }


# =========================================================================== #
# Part 0.3 self-checks, and the live exam
# =========================================================================== #
def preflight(const: Mapping[str, Any]) -> dict[str, Any]:
    """Everything checkable before the seal is touched.  Zero oracle, zero fit."""
    import hashlib

    cap1 = load_cap1()
    menu = cap1["consumer_metric_menu"]
    menu_sha = hashlib.sha256(
        json.dumps(list(menu["menu"]), separators=(",", ":")).encode()
    ).hexdigest()
    lock = json.loads((PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"
                       / "snapshot.lock.json").read_text(encoding="utf-8"))
    scope = cap1["scope_and_initial_state"]
    regenerated = sorted(__import__("random").Random(
        int(const["seed"])).sample(range(int(const["n_test_official"])),
                                   int(const["n_test_subset"])))
    index_sha = hashlib.sha256(
        json.dumps(regenerated, separators=(",", ":")).encode()).hexdigest()
    quarters = {name: [i for i in range(int(const["n_train"]))
                       if i % 4 == offset]
                for offset, name in enumerate(("r1_support", "r1_delayed",
                                               "r2_support", "r2_delayed"))}
    checks = [
        {"check": "TEST subset regenerates from seed 20260827 and matches the "
                  "frozen sha",
         "pass": (regenerated == list(const["test_indices"])
                  and index_sha == str(cap1["sampling"]
                                       ["test_row_indices_sha256"])),
         "expected": str(cap1["sampling"]["test_row_indices_sha256"]),
         "observed": index_sha},
        {"check": "menu names sha matches the freeze",
         "pass": menu_sha == str(menu["menu_names_sha256"]),
         "expected": str(menu["menu_names_sha256"]), "observed": menu_sha},
        {"check": "h0 runtime_bundle_sha matches the freeze",
         "pass": (str(lock["runtime_bundle_sha"])
                  == str(scope["h0_runtime_bundle_sha"])),
         "expected": str(scope["h0_runtime_bundle_sha"]),
         "observed": str(lock["runtime_bundle_sha"])},
        {"check": "h0 harness_content_sha matches the freeze",
         "pass": (str(lock["harness_content_sha"])
                  == str(scope["h0_harness_content_sha"])),
         "expected": str(scope["h0_harness_content_sha"]),
         "observed": str(lock["harness_content_sha"])},
        {"check": "mod-4 quarters reproduce the frozen half-protocol indices",
         "pass": (sorted(quarters["r1_support"] + quarters["r2_support"])
                  == sorted(int(i) for i in const["support_idx"])
                  and sorted(quarters["r1_delayed"] + quarters["r2_delayed"])
                  == sorted(int(i) for i in const["delayed_idx"])),
         "expected": {"support": 40, "delayed": 40},
         "observed": {name: len(rows) for name, rows in quarters.items()}},
        {"check": "injection condition is the course's fit_only_artifact",
         "pass": str(const["condition"]) == "fit_only_artifact",
         "expected": "fit_only_artifact", "observed": str(const["condition"])},
    ]
    return {"pass": all(row["pass"] for row in checks), "checks": checks,
            "zero_oracle_zero_fit_zero_label_before_unseal": True}


def _a5_card() -> dict[str, Any]:
    """The SA-1 seed card, byte-identical to the one r1 and r2 ran."""
    from evaluation.functional.task_episode_harness.agentic import (
        source_skill as ss,
    )

    gates = json.loads(SA1_GATES_JSON.read_text(encoding="utf-8"))
    card = gates["card_v0"]
    freeze = json.loads((E2 / "sa1_course_freeze.json").read_text(
        encoding="utf-8"))
    expected = str(freeze["card_seed"]["content_sha_v0"])
    got = ss.skill_content_sha(card)
    if got != expected:
        raise Locked("A5 seed card content sha %s is not the SA-1 seed %s"
                     % (got, expected))
    authority = ((card.get("risk_guards") or {}).get("authority") or {})
    if authority.get("grants_execution") is not False \
            or authority.get("supplies_candidates") is not True:
        raise Locked("A5 seed card authority is not supply-only: %s"
                     % authority)
    return card


def run_exam() -> int:
    import run_e2_ps0c_ps1 as ps0c
    import run_e2_s1_curriculum_four_arms as s1
    import run_e2_sa1_minimal as sa1
    import run_e2_t6_cls_op_shared_harness as cls
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (
        evaluate_applicability,
    )
    from evaluation.functional.task_episode_harness.agentic import (
        source_skill as ss,
    )

    started = time.time()
    cap1 = load_cap1()
    const = _constants(cap1)
    s1._set_phase(s1.PHASE_SETUP)
    payload: dict[str, Any] = {
        "protocol_version": "capstone_epilepsy2_final_v1",
        "evidence_grade": "capability_capstone_single_shot",
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "spec_sources": [
            "artifacts/functional/e2/cap1b_capstone_unlock_amendment.md "
            "(unlock replacement + A5 pool replacement, exit A)",
            "artifacts/functional/e2/cap1_capstone_protocol_freeze.md/.json "
            "(everything else, to the letter)",
            "evaluation/functional/run_e2_capstone_epilepsy2.py (PREP-1 "
            "skeleton, commit 1966fe4 lineage)",
        ],
        "cap1b_exit_executed": CAP1B_REQUIRED_EXIT,
        "single_shot": ("CAP-1 section 5: the capstone is a one-time "
                        "acceptance.  The TEST subset opens once, there is no "
                        "evaluation repeat seed, and no verdict authorises a "
                        "rerun"),
    }
    arms_out: dict[str, Any] = {}
    revisions: list[dict[str, Any]] = []
    chain: list[dict[str, Any]] = []
    original = None
    stopped: str | None = None
    try:
        pre = preflight(const)
        payload["preflight"] = pre
        if not pre["pass"]:
            raise Locked("preflight failed: %s"
                         % [row["check"] for row in pre["checks"]
                            if not row["pass"]])

        probe = ps0c.probe_new_backend()
        payload["backend_probe"] = ps0c.redact(probe)
        print("PROBE ok=%s model=%s" % (probe.get("ok"),
                                        probe.get("returned_model")),
              flush=True)
        if not probe.get("ok"):
            raise Locked("BACKEND_UNAVAILABLE: %s" % probe.get("reason"))

        card = _a5_card()
        card_ops = [str(op) for op in
                    (card["risk_guards"]["scope_v1"]["program_geometry"])]
        payload["a5_pool"] = {
            "replaces": ("CAP-1 section 3's 'S1-v2 forward terminal pool', "
                         "which retired with the S1-v2 line"),
            "skill_id": card["skill_id"],
            "card_content_sha_v0": ss.skill_content_sha(card),
            "authority": card["risk_guards"]["authority"],
            "scope_rule": card["risk_guards"]["scope_v1"]["pattern_axis_kind"],
            "pattern_axis_provenance": card["risk_guards"]["scope_v1"][
                "pattern_axis_provenance"],
            "program_geometry": card_ops,
            "revision_loop": ["R1", "R2", "R3"],
            "same_form_as": "SA-1 r1/r2",
        }

        print("UNSEAL Epilepsy2", flush=True)
        opened = unseal()
        payload["unseal_record"] = opened["record"]
        payload["unlock"] = opened["unlock"]
        payload["seal"] = opened["seal"]
        print("  seal=%s zip_sha=%s train=%s test=%s"
              % (opened["seal"]["verdict"],
                 opened["record"]["zip_sha256"][:12],
                 opened["record"]["train_shape"],
                 opened["record"]["test_shape"]), flush=True)

        cell = _epilepsy2_cell(const, opened)
        payload["cell"] = {
            "dataset": cell["dataset"], "condition": cell["condition"],
            "series_length": cell["series_length"],
            "official_train_rows": cell["official_train_rows"],
            "fit_rows": cell["fit_rows"],
            "slice_rows": cell["slice_rows"],
            "quarter_rows": cell["quarter_rows"],
            "heldout_rows": cell["heldout_rows"],
            "controlled_impulse_positions":
                cell["controlled_impulse_positions"],
            "injection_template": cell["injection_template"],
            "injection_segment_length": cell["injection_segment_length"],
            "heldout_is_clean": True,
            "material_lines": {"support": const["material"],
                               "delayed": const["material"],
                               "heldout": const["heldout_line"]},
        }
        original = _install_exam_heldout(cell)

        features = dict(_public_features(cell))
        scope_match, scope_score = evaluate_applicability(
            card["observable_applicability"], features)
        payload["card_scope_on_epilepsy2"] = {
            "machine_match": bool(scope_match),
            "applicability_score": int(scope_score),
            "binned_pattern_view": s1._binned_contract_leaves(features),
            "missed_leaves": [
                {"feature": leaf["feature"], "card_value": leaf["value"],
                 "unit_value": s1._binned_contract_leaves(features).get(
                     str(leaf["feature"]))}
                for leaf in card["observable_applicability"]["all"]
                if s1._binned_contract_leaves(features).get(
                    str(leaf["feature"])) != leaf["value"]],
            "itt_note": ("a non-match is a legal reading, not a failure: the "
                         "card's claim is conditional and ITT counts the "
                         "condition as the card stated it"),
        }
        print("  card scope match on Epilepsy2: %s" % bool(scope_match),
              flush=True)

        store_root = Path(tempfile.mkdtemp(prefix="capstone_"))
        h0 = compile_snapshot(
            PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
            verify_lock=False)
        k0_pack = s1.compile_k0(store_root / "k0")
        a5_base, applied = s1._apply_entries(
            k0_pack["k0"], [card], store_root=store_root / "bases",
            tag="a5_sa1_card")
        card_sha = ss.skill_content_sha(
            next(s for s in a5_base.skills
                 if str(s.skill_id) == SA1_CARD_ID))
        chain.append({"version": "v0", "card_content_sha": card_sha,
                      "runtime_bundle_sha": a5_base.runtime_bundle_sha})
        payload["snapshots"] = {
            "h0_sha": k0_pack["h0_sha"], "k0_sha": k0_pack["k0_sha"],
            "k0_purity": k0_pack["purity"],
            "a5_base_sha": a5_base.runtime_bundle_sha,
            "a5_applied": applied,
        }
        bases = {"Static": h0, "A3": h0, "A5": a5_base}
        unit = {"unit_id": EPILEPSY2_UNIT_ID, "dataset": cell["dataset"],
                "injection": "impulse_v2"}
        ledger = {"llm": 0, "fit": 0}

        for arm in ("Static", "A3", "A5"):
            elapsed = (time.time() - started) / 60.0
            if elapsed > float(const["wall_minutes"]):
                raise Locked("wall-clock cap %d min exceeded before %s"
                             % (int(const["wall_minutes"]), arm))
            print("ARM %s" % arm, flush=True)
            backend = cls._live_backend(int(const["llm_per_arm"]))
            arm_internal = {"Static": s1.ARM_STATIC, "A3": s1.ARM_A3,
                            "A5": s1.ARM_A5}[arm]
            result = s1.run_unit(
                unit=unit, cell=cell, arm=arm_internal,
                base_snapshot=bases[arm], carried_episodes=(),
                agent_factory=cls._live_agent, backend=backend,
                store_root=store_root, rounds=("r1",),
                fit_cap=int(const["fit_per_arm"]), carried_stamps={})
            public = _public(result)
            ledger["llm"] += int(public.get("llm_calls") or 0)
            ledger["fit"] += int(public.get("consumer_fits") or 0)
            if int(public.get("llm_calls") or 0) > int(const["llm_per_arm"]):
                raise Locked("%s used %s LLM calls, cap is %s"
                             % (arm, public.get("llm_calls"),
                                const["llm_per_arm"]))
            if int(public.get("consumer_fits") or 0) > int(
                    const["fit_per_arm"]):
                raise Locked("%s used %s fits, cap is %s"
                             % (arm, public.get("consumer_fits"),
                                const["fit_per_arm"]))
            public["candidate_sources"] = annotate_candidate_sources(
                public, skill_id=SA1_CARD_ID, card_operators=card_ops)
            arms_out[arm] = public
            deployment = public.get("deployment") or {}
            print("  deploy=%s acc=%.4f identity=%.4f gain=%+.4f llm=%s fit=%s"
                  % (deployment.get("deploy_source"),
                     float(deployment.get("heldout_accuracy") or 0.0),
                     float(deployment.get("heldout_identity_accuracy") or 0.0),
                     float(deployment.get("heldout_accuracy_gain") or 0.0),
                     public.get("llm_calls"), public.get("consumer_fits")),
                  flush=True)
            _checkpoint(CHECKPOINT, {"arms": arms_out, "ledger": ledger,
                                     "chain": chain})

            if arm == "A5":
                scored = {
                    "unit_id": EPILEPSY2_UNIT_ID,
                    "position": 1,
                    "rounds": public.get("rounds") or [],
                    "heldout_utility": float(
                        deployment.get("heldout_accuracy_gain") or 0.0),
                    "harm_event": bool(
                        deployment.get("harmed_classes_over_bar")),
                }
                revision = sa1._revise(
                    result["_end_snapshot"], scored, {"unit_id":
                                                      EPILEPSY2_UNIT_ID,
                                                      "position": 1},
                    store_root=store_root / "revise", card_sha=card_sha)
                if revision["applied"]:
                    chain.append({
                        "version": "v%d" % len(chain),
                        "card_content_sha": revision["card_sha"],
                        "produced_by": "+".join(revision["applied"]),
                        "trigger_unit": EPILEPSY2_UNIT_ID})
                revisions.append({
                    "unit_id": EPILEPSY2_UNIT_ID,
                    "applied": revision["applied"],
                    "readout": revision.get("readout"),
                    "exclusion": revision.get("exclusion"),
                    "card_content_sha_after": revision["card_sha"],
                    "descriptive_only": (
                        "CAP-1b: with one unit there is no later unit for a "
                        "revision to change, so the revision loop's capstone "
                        "readings are a descriptive record and the headline "
                        "rests on the exit-A table, not on them"),
                })
                print("  revision %s -> %s"
                      % ("+".join(revision["applied"]) or "none",
                         revision["card_sha"][:12]), flush=True)
        payload["ledger"] = {
            "llm": ledger["llm"], "llm_per_arm_cap": const["llm_per_arm"],
            "fit": ledger["fit"], "fit_per_arm_cap": const["fit_per_arm"],
            "wall_minutes": round((time.time() - started) / 60.0, 2),
            "wall_minutes_cap": const["wall_minutes"],
            "downloads": 0,
        }
    except Locked as exc:
        stopped = "STOPPED_NO_SCIENTIFIC_VERDICT"
        payload["stop"] = {"verdict": stopped, "reason": exc.reason}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {"verdict": stopped,
                           "reason": "%s: %s" % (type(exc).__name__, exc),
                           "traceback": traceback.format_exc()}
    finally:
        if original is not None:
            _restore_heldout(original)
    payload["arms"] = arms_out
    payload["revisions"] = revisions
    payload["card_version_chain"] = chain
    if stopped is None:
        payload["score"] = _score_capstone(arms_out, const)
        payload["verdict"] = payload["score"]["verdict"]
        payload["governance"] = _governance_readout(arms_out, chain, card)
    else:
        payload["verdict"] = stopped
        payload.setdefault("ledger", {
            "llm": 0, "fit": 0,
            "wall_minutes": round((time.time() - started) / 60.0, 2),
            "downloads": 0})
    payload["predictions"] = _prediction_table(payload)
    payload["obligations"] = _exam_obligations()
    OUT_JSON.write_text(json.dumps(_plainish(payload), ensure_ascii=False,
                                   indent=1, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
    OUT_MD.write_text(_exam_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"],
                      "score": payload.get("score"),
                      "ledger": payload.get("ledger")},
                     ensure_ascii=False, indent=1, default=str), flush=True)
    return 0 if stopped is None else 1


def _public_features(cell: Mapping[str, Any]) -> Mapping[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
        extract_public_features,
    )

    block = np.asarray(cell["observation_block"], dtype=np.float64)
    return dict(extract_public_features(block, task_kind="classification"))


def _plainish(value: Any) -> Any:
    import run_e2_s1_curriculum_four_arms as s1
    return s1._plain(value)


def _governance_readout(arms: Mapping[str, Any], chain: Sequence[Mapping[str, Any]],
                        card: Mapping[str, Any]) -> dict[str, Any]:
    """The G0-shaped safety and over-reach readout, on the capstone."""
    a5 = arms.get("A5") or {}
    deployment = a5.get("deployment") or {}
    added = list(a5.get("skills_added_in_unit") or ())
    tier_moved = [row for row in added
                  if str(row.get("skill_id")) == SA1_CARD_ID]
    v0_authority = dict((card.get("risk_guards") or {}).get("authority") or {})
    return {
        "harmed_classes": list(
            deployment.get("harmed_classes_over_bar") or ()),
        "harm_event": bool(deployment.get("harmed_classes_over_bar")),
        "worst_class_recall_delta": deployment.get(
            "heldout_recall_delta_by_class"),
        "card_authority_unchanged": v0_authority,
        "card_was_not_re_minted": not tier_moved,
        "no_tier_promotion": True,
        "no_scope_widening": (
            "every revision in this book narrows or appends; "
            "narrow_applicability nests the old AST under all(old, not(...)) "
            "and no code path widens it"),
        "guided_positive_counts_zero": (
            "every evidence row the revision appends carries "
            "counts_toward_authorization=false, so a positive earned under "
            "the card buys the card nothing"),
        "version_chain": [row.get("card_content_sha") for row in chain],
    }


def _prediction_table(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    seal = payload.get("seal") or {}
    scope = payload.get("card_scope_on_epilepsy2") or {}
    gov = payload.get("governance") or {}
    sources = ((payload.get("arms") or {}).get("A5") or {}).get(
        "candidate_sources") or {}
    rows = [
        {"id": "P1", "claim": "the seal re-check passes in full",
         "held": bool(seal.get("pass")),
         "observed": "verdict %s; %s of %s checks passed"
                     % (seal.get("verdict"),
                        sum(1 for row in (seal.get("checks") or ())
                            if row.get("pass")),
                        len(seal.get("checks") or ()))},
        {"id": "P2", "claim": "A5 harm = 0 and no over-reach",
         "held": bool(gov and not gov.get("harm_event")
                      and gov.get("card_was_not_re_minted")),
         "observed": "harmed classes %s; card re-minted: %s"
                     % (gov.get("harmed_classes"),
                        not gov.get("card_was_not_re_minted"))},
        {"id": "P3", "claim": "the card's family Scope axis matches "
                              "Epilepsy2 under the course injection family, "
                              "so the card is retrieved and supplies",
         "held": bool(scope.get("machine_match")
                      and int(sources.get("supplied_total") or 0) >= 1),
         "observed": "machine_match %s; supplied into the pool %s; "
                     "dedup_swallowed %s"
                     % (scope.get("machine_match"),
                        sources.get("supplied_total"),
                        sources.get("dedup_swallowed_total"))},
        {"id": "P4", "claim": "the headline A5-A3 reading is the exam question "
                              "itself and was not predicted",
         "held": None,
         "observed": "reported as the three-way verdict, not scored against a "
                     "prediction"},
    ]
    return rows


def _exam_obligations() -> dict[str, Any]:
    return {
        "methods_contracts_runtime_operators_unmodified": True,
        "thresholds_menu_template_prompt_model_unmodified": (
            "every numeric constant is read from "
            "cap1_capstone_protocol_freeze.json; the injection template is "
            "cls._inject_v2 at helpers['positions'], unchanged"),
        "cap1_section7_void_cap1b_substituted": True,
        "cap1_section3_a5_pool_substituted": True,
        "s1_oracle_not_touched": True,
        "downloads": 0,
        "single_shot_no_rerun": (
            "the TEST subset was opened once; no verdict in CAP-1 section 6 "
            "authorises a second pass and none was run"),
        "full_repo_pytest_not_run": True,
        "subagents_spawned": 0,
        "other_lines_files_untouched": True,
        "seal_sha_gap_declared": (
            "CAP-0 recorded no zip sha256, so that one comparison could not be "
            "made; every fact it did record was re-checked and matched, and "
            "the digest is now on record"),
    }


def _exam_markdown(payload: Mapping[str, Any]) -> str:
    score = payload.get("score") or {}
    gov = payload.get("governance") or {}
    acc = score.get("acc") or {}
    sources = ((payload.get("arms") or {}).get("A5") or {}).get(
        "candidate_sources") or {}
    scope = payload.get("card_scope_on_epilepsy2") or {}
    record = payload.get("unseal_record") or {}
    lines = [
        "# CAP-1 capstone: Epilepsy2, single shot",
        "",
        "**CAPSTONE 判词:%s;A5−A3 accuracy = %s;harm = %s**"
        % (payload.get("verdict"),
           ("%+.6f" % float(score["delta_a5_minus_a3"]))
           if "delta_a5_minus_a3" in score else "n/a",
           len(gov.get("harmed_classes") or ()) if gov else "n/a"),
        "",
        "CAP-1b exit **%s** executed.  CAP-1 section 7 is void; section 3's "
        "A5 pool is replaced by the SA-1 scope-v2 supply card with the R1-R3 "
        "revision loop open.  Everything else is the CAP-1 freeze verbatim."
        % payload.get("cap1b_exit_executed"),
        "",
        "## Unseal record",
        "",
    ]
    if record:
        lines += [
            "| field | value |", "|---|---|",
            "| unsealed at (UTC) | %s |" % record.get("unsealed_at_utc"),
            "| zip | `%s` |" % record.get("zip_path"),
            "| zip bytes | %s |" % record.get("zip_bytes"),
            "| zip sha256 | `%s` |" % record.get("zip_sha256"),
            "| seal verdict | %s |" % record.get("seal_verdict"),
            "| first read scope | %s |" % record.get("first_read_scope"),
            "| TRAIN / TEST shape | %s / %s |" % (record.get("train_shape"),
                                                  record.get("test_shape")),
            "| authorised by | %s |" % record.get("authorised_by"),
            "",
        ]
    seal = payload.get("seal") or {}
    if seal:
        lines += ["### Seal re-check", "",
                  "| check | pass | expected | observed |",
                  "|---|---|---|---|"]
        for row in seal.get("checks") or ():
            lines.append("| %s | %s | %s | %s |"
                         % (row.get("check"), row.get("pass"),
                            str(row.get("expected"))[:60],
                            str(row.get("observed"))[:60]))
        lines += ["", "**%s**" % seal.get("zip_sha256_gap"), ""]

    pre = payload.get("preflight") or {}
    if pre:
        lines += ["## Preflight (0 LLM, 0 fit, pre-unseal)", "",
                  "| check | pass |", "|---|---|"]
        for row in pre.get("checks") or ():
            lines.append("| %s | %s |" % (row.get("check"), row.get("pass")))
        lines.append("")

    if score:
        lines += ["## Verdict arithmetic (CAP-1 section 6)", "",
                  "| reading | value |", "|---|---|",
                  "| Static accuracy | %.6f |" % float(acc.get("Static") or 0),
                  "| A3 accuracy | %.6f |" % float(acc.get("A3") or 0),
                  "| A5 accuracy | %.6f |" % float(acc.get("A5") or 0),
                  "| **A5 − A3** | **%+.6f** |"
                  % float(score.get("delta_a5_minus_a3") or 0),
                  "| held-out material line | %.6f |"
                  % float(score.get("heldout_material_line") or 0),
                  "| worst-class delta (A5 − A3) | %+.6f |"
                  % float(score.get("worst_class_delta_a5_minus_a3") or 0),
                  "| A5 harm | %s |" % score.get("harm_a5"),
                  "| A5 − Static | %+.6f |"
                  % float((score.get("attribution") or {}).get(
                      "a5_minus_static") or 0),
                  "| A3 − Static | %+.6f |"
                  % float((score.get("attribution") or {}).get(
                      "a3_minus_static") or 0),
                  "",
                  "A5/A3 against Static are reported per `AGENTS.md` section "
                  "2.1 and do not replace the verdict.", ""]

    lines += ["## Per arm", "",
              "| arm | deploy | applied | accuracy | identity accuracy | gain "
              "| supplied | self-proposed | dedup swallowed | probes | llm | "
              "fit |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("Static", "A3", "A5"):
        row = (payload.get("arms") or {}).get(arm) or {}
        dep = row.get("deployment") or {}
        src = row.get("candidate_sources") or {}
        probes = sum(len(rec.get("probes") or ())
                     for rec in (row.get("rounds") or ()))
        lines.append("| %s | %s | %s | %.6f | %.6f | %+.6f | %s | %s | %s | "
                     "%d | %s | %s |"
                     % (arm, dep.get("deploy_source"),
                        ",".join(str(step.get("op")) for step in
                                 (dep.get("applied_program") or ()))
                        or "identity",
                        float(dep.get("heldout_accuracy") or 0.0),
                        float(dep.get("heldout_identity_accuracy") or 0.0),
                        float(dep.get("heldout_accuracy_gain") or 0.0),
                        src.get("supplied_total", "-"),
                        src.get("self_proposed_total", "-"),
                        src.get("dedup_swallowed_total", "-"),
                        probes, row.get("llm_calls"),
                        row.get("consumer_fits")))
    lines += ["", "Per-class held-out recall:", ""]
    for arm in ("Static", "A3", "A5"):
        dep = ((payload.get("arms") or {}).get(arm) or {}).get(
            "deployment") or {}
        lines.append("- **%s**: recall %s, delta vs identity %s"
                     % (arm, dep.get("heldout_recall_by_class"),
                        dep.get("heldout_recall_delta_by_class")))

    lines += ["", "## The card on Epilepsy2 (ITT)", "",
              "Scope machine match: **%s** (applicability score %s).  %s"
              % (scope.get("machine_match"),
                 scope.get("applicability_score"), scope.get("itt_note")), ""]
    if scope.get("missed_leaves"):
        lines += ["Leaves the Target does not carry at the card's value:", "",
                  "| leaf | card | Epilepsy2 |", "|---|---|---|"]
        for row in scope["missed_leaves"]:
            lines.append("| %s | %s | %s |" % (row["feature"],
                                               row["card_value"],
                                               row["unit_value"]))
        lines.append("")

    if payload.get("card_version_chain"):
        lines += ["## Card version chain", "",
                  " -> ".join("%s `%s`" % (row.get("version"),
                                           str(row.get("card_content_sha"))
                                           [:12])
                              for row in payload["card_version_chain"]), ""]
    for row in payload.get("revisions") or ():
        lines += ["Revision on the exam unit: rules %s, card sha `%s`.  %s"
                  % ("+".join(row.get("applied") or []) or "none",
                     str(row.get("card_content_sha_after"))[:12],
                     row.get("descriptive_only")), ""]

    if gov:
        lines += ["## Governance readout", "", "| item | value |", "|---|---|"]
        for key in ("harm_event", "harmed_classes", "card_authority_unchanged",
                    "card_was_not_re_minted", "no_tier_promotion",
                    "no_scope_widening", "guided_positive_counts_zero"):
            lines.append("| %s | %s |" % (key, gov.get(key)))
        lines.append("")

    lines += ["## Pre-registered predictions", "",
              "| id | claim | held | observed |", "|---|---|---|---|"]
    for row in payload.get("predictions") or ():
        held = row.get("held")
        lines.append("| %s | %s | %s | %s |"
                     % (row["id"], row["claim"],
                        "not predicted" if held is None
                        else ("yes" if held else "**no**"),
                        str(row["observed"]).replace("|", "/")))

    ledger = payload.get("ledger") or {}
    lines += ["", "## Cost", "",
              "LLM %s (cap %s per arm), consumer fits %s (cap %s per arm), "
              "wall %s min (cap %s), downloads %s."
              % (ledger.get("llm"), ledger.get("llm_per_arm_cap"),
                 ledger.get("fit"), ledger.get("fit_per_arm_cap"),
                 ledger.get("wall_minutes"), ledger.get("wall_minutes_cap"),
                 ledger.get("downloads")),
              "", "## Obligations", ""]
    for key, value in sorted((payload.get("obligations") or {}).items()):
        lines.append("- **%s**: %s" % (key, value))
    if payload.get("stop"):
        lines += ["", "## Stop", "", "`%s` -- %s"
                  % (payload["stop"]["verdict"], payload["stop"]["reason"])]
    return "\n".join(lines) + "\n"


def print_locked_and_exit(unlock_path: Path | None = None) -> int:
    print(lock_reason(unlock_path), flush=True)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CAP-1 Epilepsy2 capstone")
    parser.add_argument("--smoke-synthetic", action="store_true")
    parser.add_argument("--seal-check", action="store_true",
                        help="CAP-1b unlock chain + CAP-0 seal re-check, "
                             "structure only, no parse")
    parser.add_argument("--exam", action="store_true",
                        help="the one-shot live exam")
    parser.add_argument("--run", action="store_true",
                        help="PREP-1 legacy entry point; refused")
    parser.add_argument("--unlock", type=Path, default=DEFAULT_UNLOCK)
    parser.add_argument("--s1v2-pool", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    if args.seal_check:
        payload = {"unlock": unlock_readings(),
                   "preflight": preflight(_constants(load_cap1())),
                   "seal": verify_seal()}
        print(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
              flush=True)
        return 0 if (payload["unlock"]["unlocked"]
                     and payload["preflight"]["pass"]
                     and payload["seal"]["pass"]) else 1
    if args.exam:
        return run_exam()
    if args.smoke_synthetic:
        payload = smoke_synthetic(pool_path=args.s1v2_pool,
                                  unlock_path=args.unlock)
        text = json.dumps(payload, indent=1, ensure_ascii=False, default=str)
        if args.json_out:
            args.json_out.write_text(text + "\n", encoding="utf-8")
        print(json.dumps({
            "ok": payload.get("ok"),
            "d3_touched": payload.get("d3_touched"),
            "unlock_selftest_ok": payload.get("unlock_selftest", {}).get("ok"),
            "score": (payload.get("score") or {}).get("verdict"),
            "seconds": payload.get("seconds"),
            "arms": payload.get("arm_reads"),
        }, ensure_ascii=False, indent=1), flush=True)
        return 0 if payload.get("ok") and payload["unlock_selftest"]["ok"] else 1
    if args.run:
        try:
            require_unseal(args.unlock)
        except Locked as exc:
            print(exc.reason, flush=True)
            return 2
        try:
            load_official_d3(unlock_path=args.unlock)
        except Locked as exc:
            print(exc.reason, flush=True)
            return 2
    return print_locked_and_exit(args.unlock)


if __name__ == "__main__":
    raise SystemExit(main())
