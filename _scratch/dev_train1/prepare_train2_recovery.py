"""Recover this run's already-compiled candidate; never reconstruct an LLM answer.

Zero API, zero fit. --prepare archives failure receipts and writes the missing
boundary wrapper from existing store/provenance; its unavailable fields stay so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.main_protocol_p4 import dev_train1_runner as R

RUN_ID = "dev_train2_uncapped_workflow_10x10_20260910"
RD = ROOT / "_scratch/dev_train1/runs" / RUN_ID
PARENT = "191856a3496bea677d37c15629d62d9949fc16951e54acc1893fd21a6a933155"
CHILD = "8dad8b1798af2546c2edc522ed5aeddd40cd4c8288107017d6b42d02f05e021e"
CARD = "missingness-unknown-probe-identity"
RECEIPT = ROOT / ".aris/runs" / RUN_ID / "launch.json"


def authoring_files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and p.name not in ("resolved.snapshot.json", "snapshot.lock.json", ".gitkeep")}


def validate():
    original = R.read_json(RD / "result.json")
    if original.get("status") != "RUNNER_FAULT" or "mappingproxy" not in original.get("detail", ""):
        raise RuntimeError("not the targeted serialization failure")
    budget = R.read_json(RD / "budget.json")
    assert budget["llm_by_arm"] == {"old_skill_formation": 195, "slow": 1}, budget["llm_by_arm"]
    assert budget["fits_total"] == 8
    assert not (RD / "old_skill").exists() and not (RD / "new_skill").exists()
    for arm in ("static", "fixed_workflow"):
        assert not list((RD / arm).glob("scores_u01[23].json"))
    parent_path, child_path = RD / "store" / PARENT, RD / "store" / CHILD
    assert {p.name for p in (RD / "store").iterdir() if p.is_dir()} == {PARENT, CHILD}
    before_parent, before_child = authoring_files(parent_path), authoring_files(child_path)
    added = before_child.keys() - before_parent.keys()
    assert added == {"skills/learned/" + CARD + ".json"}, added
    assert not before_parent.keys() - before_child.keys()
    assert all(before_parent[k] == before_child[k] for k in before_parent)
    provenance = R.read_json(RD / "harness_snapshot_provenance" / CHILD / (PARENT + ".json"))
    assert provenance["runtime_bundle_sha"] == CHILD and provenance["parent_runtime_bundle_sha"] == PARENT
    machinery = R.load_machinery()
    parent = machinery["compile_snapshot"](parent_path, verify_lock=False)
    child = machinery["compile_snapshot"](child_path, verify_lock=False)
    assert parent.runtime_bundle_sha == PARENT and child.runtime_bundle_sha == CHILD
    entry = next(s for s in child.skills if s.skill_id == CARD)
    # Exact nested frozen applicability shape that broke the original writer.
    payload = {"observable_applicability": dict(entry.observable_applicability)}
    try:
        json.dumps(payload)
    except TypeError as exc:
        assert "mappingproxy" in str(exc), str(exc)
    else:
        raise AssertionError("expected original nested mappingproxy failure")
    encoded = json.loads(json.dumps(payload, default=R._json_default))
    card = R.read_json(child_path / "skills/learned" / (CARD + ".json"))
    assert encoded["observable_applicability"] == card["observable_applicability"]
    assert authoring_files(parent_path) == before_parent
    assert authoring_files(child_path) == before_child
    checkpoints = [RD / "old_skill_formation/train_w.json"] + sorted((RD / "old_skill_formation").glob("eval_v_*.json"))
    assert len(checkpoints) == 4
    failures = []
    for path in checkpoints:
        rows = R.read_json(path)["sequences"]
        assert len(rows) == 10
        failures.extend(row["key"] for row in rows if not row["valid_mechanism_decision"])
    assert len(failures) == 2
    return machinery, parent, child, card, checkpoints, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    machinery, parent, child, card, checkpoints, failures = validate()
    summary = {"status": "CHECK_PASS", "llm_calls": 0, "fits": 0,
               "candidate_recompiled_matches_existing": True,
               "only_semantic_edit": "ADD skill_library.entries/" + CARD,
               "nested_mappingproxy_roundtrip": True, "frozen_failure_keys": failures}
    if not args.prepare:
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    archive = RD / "before_serialization_recovery"
    if archive.exists() or (RD / "boundary.json").exists():
        raise RuntimeError("recovery already prepared or partial archive exists; inspect before repeating")
    saved_material = [RD / n for n in ("result.json", "budget.json", "slow_input.json", "orchestration.json",
                                       "run_contract.json", "preflight.json", "fixed_workflow_material.json")]
    saved_material.extend(checkpoints)
    saved_material.extend((RD / "old_skill_formation").glob("scores_*.json"))
    saved_material.extend((RD / "old_skill_formation").glob("predictions_*.json"))
    preserved = {p.relative_to(RD).as_posix(): p.read_bytes() for p in saved_material}
    for path in saved_material:
        dest = archive / path.relative_to(RD)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    shutil.copy2(RECEIPT, archive / "launch_receipt.json")
    target = "skill_library.entries/" + CARD
    boundary = {
        "maximum_modified_fraction": 1.0,
        "outcome": "CANDIDATE_COMPILED", "treatment": "UPDATE_TREATMENT",
        "proposal": {
            "proposed": True, "target_surface_id": target, "operation": "ADD",
            "skill_id": CARD, "written_text": card["body"],
            "observable_applicability": card["observable_applicability"],
            "source": "recovered final compiled card, not the missing raw LLM response",
            "original_proposal_response": "UNAVAILABLE",
            "original_behavior_predictions_and_attempt_log": "UNAVAILABLE",
        },
        "applied": {"applied": True, "runtime_bundle_sha": CHILD,
                    "parent_runtime_bundle_sha": PARENT, "target_surface_id": target},
        "recovery": {
            "kind": "compiled_snapshot_boundary_wrapper",
            "original_boundary_save_failed": "TypeError: mappingproxy",
            "source_provenance": "harness_snapshot_provenance/" + CHILD + "/" + PARENT + ".json",
            "missing_proposal_fields_not_invented": True, "no_slow_reask": True,
            "formation_evidence_frozen_including_failures": True,
            "original_failure_archive": archive.name,
        },
    }
    R.atomic_write_json(RD / "boundary.json", boundary)
    def forbidden_backend():
        raise AssertionError("recovery reasked a consumed decision")
    spec = R.RunSpec(train_uids=[], eval_uids=[], maximum_modified_fraction=1.0)
    restored = R.run_slow_boundary(spec=spec, machinery=machinery, parent=parent, task_ctx=None,
                                  card={}, budget=None, slow_factory=forbidden_backend,
                                  store=SimpleNamespace(root=RD / "store"), controller=None,
                                  checkpoint_path=RD / "boundary.json")
    assert restored["slow_was_not_reasked"] and restored["candidate_snapshot"].runtime_bundle_sha == CHILD
    for path in checkpoints:
        data = R.read_json(path)
        rows = R.run_batch_fast(spec=spec, bundle=None, snapshot=parent, machinery=machinery,
                               backend_factory=forbidden_backend, uids=[r["series_uid"] for r in data["sequences"]],
                               role=data["role"], arm=data["arm"], position=data["position"],
                               origin=data["origin"], task_ctx=None, budget=None, checkpoint_path=path)
        assert rows == data["sequences"]
    assert all((RD / path).read_bytes() == data for path, data in preserved.items())
    assert RECEIPT.read_bytes() == (archive / "launch_receipt.json").read_bytes()
    summary.update(status="RECOVERY_PREPARED", original_files_unchanged=len(preserved),
                   restored_same_candidate=True, consumed_formation_rows_not_reasked=40,
                   budget_unchanged=True, archive=archive.name)
    R.atomic_write_json(RD / "serialization_recovery.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
