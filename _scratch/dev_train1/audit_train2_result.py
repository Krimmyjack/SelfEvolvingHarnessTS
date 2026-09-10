"""Zero-LLM, zero-fit closeout of the existing TRAIN-2 run.

Only reads the named run. Optional output is a compact, Git-trackable evidence
extract, not a replacement for raw receipts. No new hashes or dataset reads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "dev_train2_uncapped_workflow_10x10_20260910"
RD = ROOT / "_scratch/dev_train1/runs" / RUN_ID


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit():
    result = read(RD / "result.json")
    contract = read(RD / "run_contract.json")
    launch = read(ROOT / ".aris/runs" / RUN_ID / "launch.json")
    assert result["status"] == "COMPLETED" and result["exit_code"] == 0
    assert result["incomplete_readouts"] == []
    origins = ("2616", "2856")
    baseline = result["arms"]["static"]["scores"]
    population = set(baseline[origins[0]]["per_uid_smase"])
    assert len(population) == 10
    arms = {}
    for arm in ("static", "fixed_workflow", "old_skill", "new_skill"):
        scores = result["arms"][arm]["scores"]
        gains, per_window = [], {}
        for origin in origins:
            score = scores[origin]
            reference = baseline[origin]["per_uid_smase"]
            measured = score["per_uid_smase"]
            assert set(reference) == set(measured) == population
            assert score["n_readable"] == score["n_eval"] == 10
            assert score["n_unknown"] == 0
            assert abs(statistics.mean(measured.values()) - score["mean_smase"]) < 1e-12
            differences = {uid: reference[uid] - measured[uid] for uid in sorted(population)}
            for uid, value in score.get("per_uid_utility", {}).items():
                assert abs(differences[uid] - value) < 1e-12
            gains.extend(differences.values())
            per_window[origin] = {"mean_gain": statistics.mean(differences.values()),
                                  "per_uid_smase": measured, "per_uid_gain": differences,
                                  "risk": score.get("risk", {"n_harmed": 0, "n": 10})}
        arms[arm] = {"mean_gain": statistics.mean(gains), "windows": per_window,
                     "harmed_count": sum(v["risk"]["n_harmed"] for v in per_window.values()),
                     "series_window_count": len(gains), "worst_harm": max(0.0, -min(gains))}
    decisions = {}
    for arm in ("old_skill", "new_skill"):
        decisions[arm] = {}
        for path in [RD / arm / "train_w.json"] + sorted((RD / arm).glob("eval_v_*.json")):
            rows = read(path)["sequences"]
            assert len(rows) == 10
            assert all(r["valid_mechanism_decision"] and not r["faults"] for r in rows)
            assert all(r["returned_models"] == ["grok-4.6-build"] for r in rows)
            keys = ("key", "series_uid", "role", "origin", "deploy_status", "assignment_policy",
                    "chosen_candidate_id", "deployed_program", "typed_steps", "region_geometry",
                    "returned_models", "llm_requests_sent")
            decisions[arm][path.name] = [{k: r.get(k) for k in keys} for r in rows]
    new_rows = [r for batch in decisions["new_skill"].values() for r in batch]
    assert len(new_rows) == 30
    assert all(r["deploy_status"] == "IDENTITY_SELECTED" for r in new_rows)
    assert all(r["assignment_policy"] == "explicit_identity" for r in new_rows)
    assert result["arms"]["new_skill"]["model"]["coefficients"] == result["arms"]["static"]["model"]["coefficients"]
    assert all(arms["new_skill"]["windows"][o]["per_uid_smase"] == baseline[o]["per_uid_smase"] for o in origins)
    formation_faults = []
    for path in [RD / "old_skill_formation/train_w.json"] + sorted((RD / "old_skill_formation").glob("eval_v_*.json")):
        for row in read(path)["sequences"]:
            if not row["valid_mechanism_decision"]:
                formation_faults.append({k: row.get(k) for k in ("key", "deploy_status", "why_unknown")})
    assert len(formation_faults) == 2
    child = result["boundary"]["applied"]["runtime_bundle_sha"]
    learned = read(RD / "store" / child / "skills/learned/missingness-unknown-probe-identity.json")
    card = read(RD / "slow_input.json")
    return {
        "run_id": RUN_ID, "source": (RD / "result.json").relative_to(ROOT).as_posix(),
        "status": "COMPLETED", "ended_at_utc": launch["ended_at_utc"],
        "population": {"train": 10, "eval": 10, "scored_windows": list(origins)},
        "data_role": contract.get("data_role"), "arms": arms,
        "new_minus_old": {o: arms["new_skill"]["windows"][o]["mean_gain"] - arms["old_skill"]["windows"][o]["mean_gain"] for o in origins},
        "new_minus_old_mean": arms["new_skill"]["mean_gain"] - arms["old_skill"]["mean_gain"],
        "fixed_chosen": result["arms"]["fixed_workflow"]["calibration"]["chosen"],
        "decisions": decisions, "new_model_and_predictions_equal_static": True,
        "learned_card": learned, "runtime_capabilities_given_to_slow": card["runtime_capabilities"],
        "formation_faults_retained_as_unknown": formation_faults,
        "final_decisions_valid": 60, "final_decision_faults": 0,
        "exposure": {k: v for k, v in result["exposure_check"].items() if k != "per_decision"},
        "budget": result["budget"], "orchestration": result["orchestration"],
        "recovery": read(RD / "serialization_recovery.json"),
        "review_independence": "same-context", "acceptance_status": "provisional",
        "claim_ceiling": "One exposed-development run: reduced harm by explicit identity, no positive preparation utility above Static, no advantage over the calibrated fixed workflow.",
        "not_proven": ["unique causal explanation of identity preference", "repeatable learning gain", "conditional superiority", "cross-domain A5"],
        "audit_cost": {"llm": 0, "fits": 0, "dataset_reads": 0},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    summary = audit()
    if args.write:
        target = ROOT / "artifacts/main_protocol/dev_train2_closeout.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": "PASS", "arms": {a: {k: s[k] for k in ("mean_gain", "harmed_count", "worst_harm")} for a, s in summary["arms"].items()},
                      "new_minus_old": summary["new_minus_old_mean"], "final_decisions_valid": 60}, ensure_ascii=False))


if __name__ == "__main__":
    main()
