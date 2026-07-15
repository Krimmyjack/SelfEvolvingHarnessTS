from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from SelfEvolvingHarnessTS.vnext.gates import (
    M3RuntimeSupplierPreregV3,
    RuntimeSupplierArmAggregateV1,
    select_runtime_supplier,
)
from SelfEvolvingHarnessTS.vnext.init_harness import (
    H0_FORBIDDEN_VIEW_IDS,
    InitCorpusManifestV1,
    InitHarnessArtifactV1,
    InitHarnessPreregV1,
    InitMemoryEntryV1,
    InitOperatorExperienceV1,
    InitProgramTemplateV1,
    InitSeedPolicyV1,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(character: str) -> str:
    return character * 64


def test_init_corpus_is_exactly_80_legacy_plus_56_traffic_and_no_fresh_views():
    corpus = InitCorpusManifestV1.from_frozen_v02(ROOT)
    assert dict(corpus.cohort_counts) == {
        "legacy_core": 80, "probe_consumed_extension": 56,
    }
    domains = dict(corpus.domain_counts)
    assert domains["traffic"] == 56
    assert set(domains) == {
        "public_health", "macroeconomics", "cash_demand", "tourism",
        "demography", "hydrology", "solar", "traffic",
    }
    assert not any(row.exposure_class == "certified_virgin" for row in corpus.members)
    assert corpus.forbidden_view_ids == H0_FORBIDDEN_VIEW_IDS


def test_formal_h0_requires_all_four_nonempty_init_derived_components():
    corpus = InitCorpusManifestV1.from_frozen_v02(ROOT)
    prereg = InitHarnessPreregV1(corpus.artifact_sha)
    experience = InitOperatorExperienceV1(
        "v_none", _sha("1"), 8, 0.0, 0.0, _sha("2"), corpus.artifact_sha,
    )
    policy = InitSeedPolicyV1(
        "seed-noop", _sha("3"), _sha("4"), _sha("5"), corpus.artifact_sha,
    )
    template = InitProgramTemplateV1(
        "noop", _sha("3"), _sha("6"), _sha("7"), _sha("8"),
    )
    memory = InitMemoryEntryV1(
        "risk-noop", _sha("9"), "v_none", "risk", 8, 0.0, 0.0,
        _sha("a"), corpus.artifact_sha,
    )
    payload = dict(
        init_corpus_sha=corpus.artifact_sha, prereg_sha=prereg.artifact_sha,
        pattern_binding_sha=_sha("b"), action_eligibility_sha=_sha("c"),
        candidate_grammar_sha=_sha("d"), base_harness_semantic_sha=_sha("e"),
        init_evaluation_sha=_sha("f"), operator_experience=(experience,),
        seed_policies=(policy,), program_templates=(template,), memory=(memory,),
    )
    h0 = InitHarnessArtifactV1(**payload)
    assert h0.h0_sha == h0.semantic_sha
    with pytest.raises(ValueError, match="cannot omit"):
        InitHarnessArtifactV1(**{**payload, "memory": ()})
    assert "series_uid" not in InitMemoryEntryV1.__dataclass_fields__
    assert "dataset_id" not in InitMemoryEntryV1.__dataclass_fields__


def test_m3_selects_runtime_supplier_without_redefining_h0():
    prereg = M3RuntimeSupplierPreregV3()
    rows = [RuntimeSupplierArmAggregateV1(
        arm_id=arm, delta_vs_deterministic=0.0,
        ci90_low_vs_deterministic=-0.01, delta_vs_frozen_h0=0.0,
        ci90_low_vs_frozen_h0=-0.01,
        supply_ceiling_delta_vs_deterministic=0.0,
        worst_readable_loss_regression=0.0,
        worst_readable_regression_ci90_low=-0.01,
        prepared_valid_fraction=1.0, cost_gate_passed=True,
        replay_gate_passed=True,
    ) for arm in prereg.roster]
    selection = select_runtime_supplier(rows, prereg)
    assert selection.supplier_policy_id == "deterministic_b3"
    assert not selection.changes_h0
