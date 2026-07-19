import numpy as np

from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.injections import inject_target
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    extract_public_features as extract_agent_features,
)
from SelfEvolvingHarnessTS.runtime.public_features import extract_public_features


SEEDS = (101, 202, 303)


def _estimated_indices(features, length):
    start = int(np.floor(float(features["estimated_region_start_fraction"]) * length))
    end = int(np.ceil(float(features["estimated_region_end_fraction"]) * length))
    return set(range(max(0, start), min(length, end)))


def _iou(left, right):
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 0.0


def test_severe_family_observable_features_fire_on_the_frozen_corpus():
    for seed in SEEDS:
        missing = inject_target(seed, "missing", "severe")
        missing_features = extract_public_features(missing.corrupt_context).mapping
        assert missing_features["missing_fraction"] >= 0.10
        assert min(missing.affected_indices) / missing.corrupt_context.size >= 0.80
        assert max(missing.affected_indices) < missing.corrupt_context.size

        outlier = inject_target(seed, "impulsive_outlier", "severe")
        outlier_features = extract_public_features(outlier.corrupt_context).mapping
        assert outlier_features["local_robust_z_peak"] >= 4.0

        level = inject_target(seed, "level_shift", "severe")
        level_features = extract_public_features(level.corrupt_context).mapping
        assert level_features["level_excursion_score"] >= 2.5
        assert level_features["period_change_score"] < 0.25
        estimated = _estimated_indices(level_features, level.corrupt_context.size)
        assert _iou(estimated, level.affected_indices) >= 0.30

        period = inject_target(seed, "period_change", "severe")
        period_features = extract_public_features(period.corrupt_context).mapping
        assert period_features["period_evidence_status"] == "OK"
        assert period_features["period_change_score"] >= 0.25


def test_fast_agent_and_probe_panel_share_one_base_feature_extractor():
    case = inject_target(202, "level_shift", "severe")
    shared = dict(extract_public_features(case.corrupt_context).mapping)
    agent = dict(
        extract_agent_features(case.corrupt_context, task_kind="forecast")
    )
    assert {key: agent[key] for key in shared} == shared
    assert agent["level_probe_direction"] == "unknown"
