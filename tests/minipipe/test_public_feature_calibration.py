import numpy as np

from SelfEvolvingHarnessTS.contracts.observables import OBSERVABLE_FEATURES
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.injections import inject_target
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    extract_public_features as extract_agent_features,
)
from SelfEvolvingHarnessTS.runtime.public_features import (
    _DOWNSTREAM_WINDOW_POINTS,
    _POST_SHIFT_SUPPORT_MIN_POINTS,
    _expand,
    extract_public_features,
)


SEEDS = (101, 202, 303)

# The full emitted mapping, frozen as an explicit contract: the M0b wiring adds
# exactly the four split-geometry fields and nothing else may creep in without
# this assertion (and the closed vocabulary) being updated deliberately.
EXPECTED_MAPPING_KEYS = frozenset(
    {
        "task_kind",
        "missing_fraction",
        "longest_missing_run_fraction",
        "local_robust_z_peak",
        "estimated_region_start_fraction",
        "estimated_region_end_fraction",
        "level_region_fraction",
        "level_region_end_fraction",
        "outlier_region_end_fraction",
        "level_only_post_shift_support_sufficient",
        "post_shift_support_sufficient",
        "level_excursion_score",
        "estimated_level_offset",
        "period_change_score",
        "period_reliability",
        "period_evidence_status",
        "period_repair_available",
    }
)

M0B_FIELDS = (
    "level_region_fraction",
    "level_region_end_fraction",
    "outlier_region_end_fraction",
    "level_only_post_shift_support_sufficient",
)


def _end_fraction(mask):
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return 0.0
    return float((int(indices[-1]) + 1) / int(mask.size))


def _pss(end_fraction):
    return bool(
        max(0.0, (1.0 - float(end_fraction)) * _DOWNSTREAM_WINDOW_POINTS)
        >= _POST_SHIFT_SUPPORT_MIN_POINTS
    )


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


def test_m0b_mapping_emits_exactly_the_frozen_key_set():
    for seed in SEEDS:
        for family in ("missing", "impulsive_outlier", "level_shift"):
            case = inject_target(seed, family, "severe")
            mapping = extract_public_features(case.corrupt_context).mapping
            assert set(mapping) == EXPECTED_MAPPING_KEYS
            assert set(mapping) <= set(OBSERVABLE_FEATURES)


def test_m0b_split_geometry_fields_match_their_m0a_definitions():
    """The four wired fields must equal the M0a census definitions exactly:

    level_region_fraction   = mean(level_mask)
    level_region_end_fraction  = (last True index + 1) / n of level_mask, 0 if empty
    outlier_region_end_fraction = same read on the expanded outlier region the
                                  union already uses
    level_only_post_shift_support_sufficient = the frozen pss formula on
                                  level_region_end_fraction
    """
    for seed in SEEDS:
        for family in ("impulsive_outlier", "level_shift", "missing"):
            case = inject_target(seed, family, "severe")
            extraction = extract_public_features(case.corrupt_context)
            mapping = extraction.mapping
            n = case.corrupt_context.size

            level_mask = np.asarray(extraction.level_mask, dtype=bool)
            assert mapping["level_region_fraction"] == float(np.mean(level_mask))
            assert mapping["level_region_end_fraction"] == _end_fraction(level_mask)

            outlier_points = np.zeros(n, dtype=bool)
            if extraction.outlier_indices:
                outlier_points[list(extraction.outlier_indices)] = True
            outlier_region = _expand(outlier_points)
            assert mapping["outlier_region_end_fraction"] == _end_fraction(
                outlier_region
            )

            assert mapping["level_only_post_shift_support_sufficient"] == _pss(
                mapping["level_region_end_fraction"]
            )
            for field in M0B_FIELDS[:3]:
                value = float(mapping[field])
                assert np.isfinite(value) and 0.0 <= value <= 1.0


def test_m0b_union_pss_semantics_are_untouched():
    """post_shift_support_sufficient keeps its union reading; the level-only
    field is a second reading of the same frozen formula, never a replacement.
    On an outlier-tail case the two readings must be allowed to disagree in the
    observed direction (union polluted, level-only clean)."""
    for seed in SEEDS:
        case = inject_target(seed, "impulsive_outlier", "severe")
        mapping = extract_public_features(case.corrupt_context).mapping
        assert mapping["post_shift_support_sufficient"] == _pss(
            mapping["estimated_region_end_fraction"]
        )
        assert isinstance(mapping["post_shift_support_sufficient"], bool)
        assert isinstance(mapping["level_only_post_shift_support_sufficient"], bool)

    # A lone spike in the last 10% of the prefix pushes the union end fraction
    # past the pss boundary while the level mask stays empty: the union reading
    # loses post-shift support, the level-only reading keeps it. This is the
    # OUTLIER-source pss divergence M0a measured on 34.5% of decision points,
    # reproduced deterministically.
    t = np.arange(192, dtype=np.float64)
    tail_spike = np.sin(2.0 * np.pi * t / 24.0)
    tail_spike[187] += 40.0
    mapping = extract_public_features(tail_spike).mapping
    assert mapping["level_region_fraction"] == 0.0
    assert mapping["outlier_region_end_fraction"] > 0.9
    assert mapping["post_shift_support_sufficient"] is False
    assert mapping["level_only_post_shift_support_sufficient"] is True


def test_m0b_missing_branch_forces_level_fields_to_zero():
    """When missing_fraction > 0 the frozen extractor zeroes level_mask; the
    wired fields must reflect that branch, not re-detect a level region."""
    for seed in SEEDS:
        case = inject_target(seed, "missing", "severe")
        mapping = extract_public_features(case.corrupt_context).mapping
        assert float(mapping["missing_fraction"]) > 0.0
        assert mapping["level_region_fraction"] == 0.0
        assert mapping["level_region_end_fraction"] == 0.0
        assert mapping["level_only_post_shift_support_sufficient"] is True
