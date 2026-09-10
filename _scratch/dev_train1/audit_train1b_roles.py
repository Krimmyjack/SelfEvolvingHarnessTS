"""Post-hoc role recombination of frozen outputs; no LLM and no fit."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.main_protocol_p4 import dev_train1_runner as R

run_dir = R.RUNS_ROOT / "dev_train1b_role_observations_10x10_20260910"
contract = R.read_json(run_dir / "run_contract.json")
spec = R.RunSpec(train_uids=contract["train_uids"], eval_uids=contract["eval_uids"], live_api=True)
full = R.load_kdd_missing_development_bundle()
bundle = R.DataBundle(values={uid: full.values[uid] for uid in spec.train_uids + spec.eval_uids},
                      train_uids=spec.train_uids, eval_uids=spec.eval_uids, period=full.period)
del full
original = R.read_json(run_dir / "result.json")
static = {int(o): s for o, s in original["arms"]["static"]["scores"].items()}
out = {"kind": "POSTHOC_FROZEN_ROLE_RECOMBINATION", "llm_calls": 0, "fits": 0,
       "not_another_learning_run": True,
       "interpretation": "conditional training/prediction differences; not additive independent causal shares or per-training-UID credit",
       "registered_comparisons_unchanged": True, "cells": [], "diagonal_max_loss_difference": 0.0}
for train_owner in ("old_skill", "new_skill"):
    model = R.numeric.FrozenRidge.from_dict(R.read_json(run_dir / train_owner / "frozen_ridge.json"))
    for predict_owner in ("old_skill", "new_skill"):
        for position, origin in spec.eval_units:
            rows = R.read_json(run_dir / predict_owner / ("eval_v_u%03d.json" % position))["sequences"]
            assignment, missing = R.records_to_assignment(spec.eval_uids, rows)
            if assignment is None or missing:
                raise RuntimeError("incomplete frozen prediction assignment")
            predictions = R.predict_arm(spec=spec, bundle=bundle, model=model,
                eval_assignment=assignment, origin=origin)
            assert predictions["consumer_fits"] == 0
            population = R._score_prediction_payload(spec, bundle, predictions, origin, static)
            assert population["n_readable"] == len(spec.eval_uids)
            if train_owner == predict_owner:
                saved = original["arms"][train_owner]["scores"][str(origin)]
                delta = max(abs(population["per_uid_smase"][u] - saved["per_uid_smase"][u]) for u in spec.eval_uids)
                out["diagonal_max_loss_difference"] = max(out["diagonal_max_loss_difference"], delta)
                if delta > 1e-12:
                    raise RuntimeError("diagonal did not reproduce the registered result")
            gain = sum(population["per_uid_utility"].values()) / len(spec.eval_uids)
            out["cells"].append({"train_owner": train_owner, "predict_owner": predict_owner,
                "origin": origin, "mean_gain": gain, "population": population})
R.atomic_write_json(run_dir / "posthoc_role_recombination.json", out)
for a in ("old_skill", "new_skill"):
    for b in ("old_skill", "new_skill"):
        cells = [c for c in out["cells"] if c["train_owner"] == a and c["predict_owner"] == b]
        print(json.dumps({"train":a,"predict":b,"gains":[c["mean_gain"] for c in cells],
            "mean_gain":sum(c["mean_gain"] for c in cells)/len(cells),
            "harmed":sum(c["population"]["risk"]["n_harmed"] for c in cells)}, ensure_ascii=True))
print("DIAGONAL_MAX_LOSS_DIFFERENCE", out["diagonal_max_loss_difference"], "LLM=0 FITS=0")
