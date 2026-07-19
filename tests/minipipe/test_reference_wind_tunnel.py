import json
from pathlib import Path

from SelfEvolvingHarnessTS.contracts.harness import load_learned_skill_entry
from SelfEvolvingHarnessTS.evaluation.minipipe.calibration.run_reference_wind_tunnel import (
    build_reference_manifest,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import build_core_corpus
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.features import extract_public_features
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import EditController
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view


ROOT = Path(__file__).resolve().parents[2]
H0_ROOT = ROOT / "methods/ttha/harness/h0"
REFERENCE_PATH = (
    ROOT
    / "evaluation/minipipe/calibration/reference_skill_entries/closed_level_excursion_repair_v2.json"
)
OUTLIER_REFERENCE_PATH = (
    ROOT
    / "evaluation/minipipe/calibration/reference_skill_entries/sparse_public_outlier_repair_v2.json"
)


def test_reference_level_skill_is_public_and_separates_all_targets_from_risks(
    tmp_path,
):
    skill_value = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    skill = load_learned_skill_entry(skill_value)
    rules = load_m0_rules(ROOT / "evaluation/minipipe/config/m0_rules.json")
    corpus = build_core_corpus(rules)
    h0 = compile_snapshot(H0_ROOT)
    store = SnapshotStore(tmp_path / "snapshots")
    parent = store.materialize(h0)
    controller = EditController(store)
    manifest = build_reference_manifest(
        parent,
        controller,
        reference_skill_path=REFERENCE_PATH,
    )
    applied = controller.apply_to_fork(
        parent, manifest, confirmed_cause="SKILL_LIBRARY_GAP"
    )

    level_targets = [
        case for case in corpus.targets if case.private_family == "level_shift"
    ]
    level_risks = [
        case for case in corpus.risks if case.private_family == "level_shift"
    ]
    assert len(level_targets) == 6
    assert len(level_risks) >= 2
    scope_limit = float(skill.risk_guards["max_modified_fraction"])
    assert scope_limit == float(h0.verification["max_modified_fraction"]) == 0.35
    for case in level_targets:
        features = extract_public_features(case.corrupt_context).mapping
        estimated_span = float(features["estimated_region_end_fraction"]) - float(
            features["estimated_region_start_fraction"]
        )
        assert estimated_span <= scope_limit
        view = resolve_harness_view(applied.candidate_snapshot.snapshot, features)
        assert skill.skill_id in view.skill_ids
    for case in level_risks:
        features = extract_public_features(case.corrupt_context).mapping
        view = resolve_harness_view(applied.candidate_snapshot.snapshot, features)
        assert skill.skill_id not in view.skill_ids


def test_reference_outlier_skill_is_public_and_separates_all_targets_from_risks(
    tmp_path,
):
    skill_value = json.loads(OUTLIER_REFERENCE_PATH.read_text(encoding="utf-8"))
    skill = load_learned_skill_entry(skill_value)
    rules = load_m0_rules(ROOT / "evaluation/minipipe/config/m0_rules.json")
    corpus = build_core_corpus(rules)
    h0 = compile_snapshot(H0_ROOT)
    store = SnapshotStore(tmp_path / "snapshots")
    parent = store.materialize(h0)
    controller = EditController(store)
    manifest = build_reference_manifest(
        parent,
        controller,
        reference_skill_path=OUTLIER_REFERENCE_PATH,
    )
    applied = controller.apply_to_fork(
        parent, manifest, confirmed_cause="SKILL_LIBRARY_GAP"
    )

    targets = [
        case for case in corpus.targets if case.private_family == "impulsive_outlier"
    ]
    risks = [
        case for case in corpus.risks if case.private_family == "impulsive_outlier"
    ]
    assert len(targets) == 6
    assert len(risks) >= 2
    for case in targets:
        features = extract_public_features(case.corrupt_context).mapping
        view = resolve_harness_view(applied.candidate_snapshot.snapshot, features)
        assert skill.skill_id in view.skill_ids
    for case in risks:
        features = extract_public_features(case.corrupt_context).mapping
        view = resolve_harness_view(applied.candidate_snapshot.snapshot, features)
        assert skill.skill_id not in view.skill_ids
