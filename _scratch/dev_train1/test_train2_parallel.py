"""Focused orchestration witness: synthetic data, fake model replies, four real fits.

The injected Slow candidate/exposure tests scheduling only, not knowledge learning.
Compiler and cap behavior are covered by test_uncapped_contract.py.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.main_protocol_p4 import dev_train1_runner as R
from evaluation.main_protocol_p4 import dev_train2_workflow as W
import test_runner_contract as fixtures


def main():
    bundle = fixtures._bundle(n_train=4, n_eval=4)
    spec = fixtures._spec(bundle, maximum_modified_fraction=1.0, max_fits=4)
    machinery = R.load_machinery()
    snapshot = R.load_h0_snapshot(machinery)
    counters = {"slow": 0, "later_scored": 0}
    original_score = R.score_arm
    original_slow = R.run_slow_boundary
    original_fast = R.run_fast_decision
    outdir = ROOT / "_scratch/dev_train1/train2_tests"
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchestration_", dir=outdir) as td:
        rd = Path(td)

        def slow_fixture(**kwargs):
            path = kwargs["checkpoint_path"]
            if path.is_file():
                value = R.read_json(path)
                value["candidate_snapshot"] = kwargs["parent"]
                return value
            assert (rd / "fixed_workflow_material.json").is_file()
            assert (rd / "old_skill_formation/scores_u009.json").is_file()
            assert "chosen_for_fixed_arm_only" not in str(kwargs["card"])
            counters["slow"] += 1
            value = {"maximum_modified_fraction": 1.0, "treatment": "UPDATE_TREATMENT",
                     "outcome": "FIXTURE_CANDIDATE", "fixture_not_a_learning_claim": True}
            R.atomic_write_json(path, value)
            value["candidate_snapshot"] = kwargs["parent"]
            return value

        def checked_score(**kwargs):
            if kwargs["origin"] == 360:
                for arm in ("static", "fixed_workflow", "old_skill", "new_skill"):
                    assert W.eval_predictions_frozen(spec, rd, arm), arm
                counters["later_scored"] += 1
            return original_score(**kwargs)

        def counting_fast(**kwargs):
            return original_fast(**kwargs)

        delays = {uid: 0.015 for uid in bundle.train_uids + bundle.eval_uids}
        factory = fixtures._factory("identity", delay_by_uid=delays)
        with patch.object(R, "run_slow_boundary", slow_fixture), \
             patch.object(R, "score_arm", checked_score), \
             patch.object(R, "run_fast_decision", counting_fast), \
             patch.object(R, "dual_role_exposure", return_value={"outcome": "EXPOSED", "fixture": True}):
            result = W.run(spec=spec, bundle=bundle, run_id="fixture", run_dir=rd,
                           backend_factory=factory, machinery=machinery, snapshot=snapshot,
                           workflow_menu=[("identity", None)])
            assert result["status"] == "COMPLETED", result.get("detail", result["status"])
            assert result["budget"]["fits_total"] == 4, result["budget"]
            assert result["orchestration"]["peak_fast_sessions"] == 4
            assert result["orchestration"]["peak_fits"] == 1
            assert result["orchestration"]["slow_waited_for_formation"]
            assert result["orchestration"]["later_scores_after_fast"]
            assert counters == {"slow": 1, "later_scored": 4}, counters
            for arm in ("old_skill_formation", "old_skill", "new_skill"):
                rows = R.read_json(rd / arm / "train_w.json")["sequences"]
                assert [r["series_uid"] for r in rows] == list(spec.train_uids)
                assert len({r["key"] for r in rows}) == len(rows)
                saved = R.read_json(rd / arm / "frozen_assignment.json")
                assert saved["arm"] == arm
            calls = result["budget"]["llm_total"]
            before = {str(p.relative_to(rd)): p.read_bytes() for p in rd.glob("*/train_w.json")}
            before["boundary.json"] = (rd / "boundary.json").read_bytes()
            resumed = W.run(spec=spec, bundle=bundle, run_id="fixture", run_dir=rd,
                            backend_factory=factory, machinery=machinery, snapshot=snapshot,
                            workflow_menu=[("identity", None)])
            assert resumed["status"] == "COMPLETED", resumed.get("detail")
            assert resumed["budget"]["fits_total"] == 4
            assert resumed["budget"]["llm_total"] == calls
            assert counters["slow"] == 1
            assert all((rd / p).read_bytes() == b for p, b in before.items())

        # Exercise the real boundary restore path: a saved no-update must not call Slow.
        no_update_path = rd / "real_boundary_restore.json"
        R.atomic_write_json(no_update_path, {"maximum_modified_fraction": 1.0,
                                            "outcome": "NO_UPDATE", "treatment": "NO_UPDATE_TREATMENT"})
        def forbidden_factory():
            raise AssertionError("restored Slow was reasked")
        restored = original_slow(spec=spec, machinery=machinery, parent=snapshot,
                                task_ctx=None, card={}, budget=None, slow_factory=forbidden_factory,
                                store=None, controller=None, checkpoint_path=no_update_path)
        assert restored["slow_was_not_reasked"]
        receipt = {"status": "PASS", "real_llm_calls": 0, "synthetic_fits": 4,
                   "fake_backend_attempts": calls,
                   "checks": ["global four Fast across arms", "one fit/commit writer",
                              "Slow waits for both formation branches", "later truth after all outputs freeze",
                              "four real shared model fits; fixed reuses calibrated model",
                              "stable role/UID keys", "resume no extra calls or fits",
                              "real saved Slow boundary does not reask"],
                   "orchestration": result["orchestration"],
                   "scope": "scheduling witness; injected candidate/exposure is not a learning result"}
        R.atomic_write_json(outdir / "summary.json", receipt)
        print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
