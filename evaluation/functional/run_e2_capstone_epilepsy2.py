"""CAP-1 Epilepsy2 capstone runner (PREP-1 prebuild).

Every numeric constant is read from ``cap1_capstone_protocol_freeze.json``.
The official D3 zip is never opened unless a valid unseal record exists.
``--smoke-synthetic`` builds TRAIN/TEST in memory, uses the scripted
backend, and never touches D3.

  python evaluation/functional/run_e2_capstone_epilepsy2.py
  python evaluation/functional/run_e2_capstone_epilepsy2.py --smoke-synthetic
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
DEFAULT_UNLOCK = E2 / "cap1_unseal_record.json"
SIGNAL = "S1V2_FORWARD_SIGNAL"
D3_MARKERS = ("D3_reserve", "EpilepticSeizures.zip", "Epilepsy2.zip",
              "EpilepticSeizures.ts")

_D3_TOUCHES: list[str] = []


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
    if _is_d3_path(path):
        _record_d3_touch(path)
        raise Locked("D3 zip is sealed; runner refused to open %s" % path)


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
    """The only function that may open D3.  Smoke never calls it."""
    require_unseal(unlock_path)
    zip_path = d3_zip_declared_path()
    refuse_d3_open(zip_path)  # still refuse until a later book implements parse
    raise Locked(
        "unseal record is valid but this PREP-1 runner does not parse D3 "
        "bytes; live unseal belongs to the exam book after SIGNAL"
    )


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


def print_locked_and_exit(unlock_path: Path | None = None) -> int:
    print(lock_reason(unlock_path), flush=True)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CAP-1 Epilepsy2 capstone (sealed until S1-v2 SIGNAL)")
    parser.add_argument("--smoke-synthetic", action="store_true")
    parser.add_argument("--run", action="store_true",
                        help="live official D3 exam; refused without unseal")
    parser.add_argument("--unlock", type=Path, default=DEFAULT_UNLOCK)
    parser.add_argument("--s1v2-pool", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
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
