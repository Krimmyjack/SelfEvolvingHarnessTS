from __future__ import annotations

from pathlib import Path

import numpy as np

from SelfEvolvingHarnessTS.vnext.protocol import (
    DiscoveryFoldManifestV1,
    H0LineageArtifactV2,
    HistoricalExposureManifestV1,
    LLMQualificationArtifactV2,
    MethodInputContractV1,
    VNextDataUsageManifestV1,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_v02_views_are_disjoint_group_atomic_and_counted():
    manifest = VNextDataUsageManifestV1.from_frozen_v02(ROOT)
    assert (len(manifest.init.series_uids), len(manifest.init.overlap_groups)) == (136, 136)
    assert manifest.init.permitted_uses == ("formal_h0_construction",)
    assert (len(manifest.search.series_uids), len(manifest.search.overlap_groups)) == (284, 247)
    assert (len(manifest.sa_validation.series_uids), len(manifest.sa_validation.overlap_groups)) == (135, 110)
    assert not set(manifest.init.overlap_groups) & set(manifest.search.overlap_groups)
    assert manifest.sa_validation.historical_result_exposure.startswith("baseline_outcome_exposed")


def test_historical_exposure_manifest_discloses_all_sa_validation_losses():
    exposure = HistoricalExposureManifestV1.from_frozen_v02(ROOT)
    assert exposure.support_a_uid_count == 555
    assert exposure.sa_validation_program_loss_uid_count == 135
    assert exposure.sa_validation_repeat_loss_uid_count == 135
    assert len(exposure.exposed_program_ids) == 10
    assert len(exposure.repeat_measurements) == 17
    assert not exposure.automatic_search_read_allowed


def test_discovery_fold_views_do_not_change_the_frozen_outer_split():
    folds = DiscoveryFoldManifestV1.from_frozen_v02(ROOT)
    assert len(folds.task_g_folds) == 383
    assert len(folds.search_folds) == 247
    assert {row.fold for row in folds.task_g_folds} <= set(range(5))
    assert len({row.overlap_group for row in folds.search_folds}) == 247


def test_method_input_contract_has_explicit_invalid_terminals():
    contract = MethodInputContractV1()
    assert contract.validate(np.array([1.0, np.nan])).valid
    assert contract.validate(np.array([])).code == "empty_input"
    assert contract.validate(np.array([np.nan])).code == "no_finite_observation"
    assert contract.validate(np.array([np.inf])).code == "infinite_input"
    assert contract.validate(np.ones((2, 2))).code == "input_not_one_dimensional"


def test_h0_lineage_is_unique_and_llm_runtime_is_separate_from_evolution():
    lineage = H0LineageArtifactV2(
        h_base_sha="1" * 64, m3a_prereg_sha="2" * 64, m3a_result_sha="3" * 64,
        supplier_selection_rule_sha="4" * 64, supplier_policy_id="deterministic_b3",
        supplier_policy_sha="5" * 64, h0_harness_sha="6" * 64, h0_method_sha="7" * 64,
    )
    replay = H0LineageArtifactV2(**lineage.__dict__)
    assert lineage.artifact_sha == replay.artifact_sha
    changed = H0LineageArtifactV2(**{
        **lineage.__dict__, "supplier_policy_sha": "8" * 64,
    })
    assert changed.artifact_sha != lineage.artifact_sha

    qualification = LLMQualificationArtifactV2(
        trial_authorization_sha="1" * 64, initial_runtime_efficacy=False,
        evolution_llm_qualified=True, mature_runtime_llm_qualified=False,
        complementarity_qualified=False, m3a_result_sha="2" * 64,
        evolution_result_sha="3" * 64, supplier_swap_sha="4" * 64,
    )
    assert qualification.evolution_llm_qualified
    assert not qualification.runtime_llm_qualified
