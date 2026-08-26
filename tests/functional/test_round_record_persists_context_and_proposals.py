"""The round record must keep the Context the Fast path saw and every proposal.

PS-1 stopped at its provenance gate because no execution record on the
classification line persisted a deployment-visible Pattern, so a cross-domain
hypothesis card had no machine-evaluable WHEN clause to be built from.  The
same records also lost the top of the behaviour funnel: proposals were tagged
by scanning the agent's free-text candidate id for an operator word, and the
ids it invents ("intrinsic_extreme_deviation_filter") contain none, so every
S1c probe carried an empty operator list.

Two properties are checked here, both at zero LLM and zero Consumer fits:

* the binned projection stored on the record is the contract projection of the
  *same* mapping the round hands to ``run_online_round`` as ``fast_features``,
  and it agrees with the independent pattern-view path the curriculum uses;
* a proposal's family comes from its compiled steps, so it survives a
  candidate id that names no operator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
import run_e2_s1a_curriculum_oracle_audit as s1a  # noqa: E402
from SelfEvolvingHarnessTS.contracts.observables import (  # noqa: E402
    OBSERVABLE_FEATURES,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)

# The two substrates PS-0 re-earns on.  Real cells, no injection scan.
DRY_UNITS = (
    {"unit_id": "GunPointAgeSpan__impulse_v2", "dataset": "GunPointAgeSpan",
     "injection": "impulse_v2", "series_length": 150},
    {"unit_id": "PowerCons__impulse_v2", "dataset": "PowerCons",
     "injection": "impulse_v2", "series_length": 144},
)


@pytest.fixture(scope="module", params=DRY_UNITS,
                ids=[unit["unit_id"] for unit in DRY_UNITS])
def cell(request):
    return s1._build_cell(request.param)


def test_stored_projection_is_the_agent_view_not_a_recomputation(cell):
    """The dry cross-check the book asks for.

    ``_run_round`` computes ``features`` once, hands that mapping to the Fast
    path, and stores ``_binned_contract_leaves(features)``.  Here the same
    mapping is built the same way and projected; it must agree leaf for leaf
    with the curriculum's independent binned-pattern path over the same block.
    """
    block = np.asarray(cell["observation_block"], dtype=np.float64)
    features = dict(extract_public_features(block, task_kind=s1.TASK_KIND))

    stored = s1._binned_contract_leaves(features)
    independent = s1a._binned_public_features(block)

    assert stored == independent
    assert stored, "the projection must not be empty"


def test_every_stored_leaf_is_an_observable_contract_leaf(cell):
    block = np.asarray(cell["observation_block"], dtype=np.float64)
    features = dict(extract_public_features(block, task_kind=s1.TASK_KIND))

    stored = s1._binned_contract_leaves(features)

    assert set(stored) <= set(OBSERVABLE_FEATURES)
    off_contract = set(features) - set(OBSERVABLE_FEATURES)
    assert not (set(stored) & off_contract)


def test_the_projection_carries_pattern_axes_a_scope_can_use(cell):
    """A WHEN clause needs leaves other than the eligibility gate itself."""
    block = np.asarray(cell["observation_block"], dtype=np.float64)
    features = dict(extract_public_features(block, task_kind=s1.TASK_KIND))

    stored = s1._binned_contract_leaves(features)

    assert set(stored) - {"task_kind"}, (
        "task_kind alone is an eligibility gate, not a Pattern Scope")


def test_family_comes_from_steps_not_from_the_candidate_id():
    """The instrument gap S1c hid: ids the agent invents name no operator."""
    invented = "intrinsic_extreme_deviation_filter"
    steps = [{"op": "hampel_filter", "params": {}}]

    assert s1._ops_in(invented) == []
    assert s1._steps_operators(steps) == ["hampel_filter"]
    assert s1._family_of(s1._steps_operators(steps)) == "hampel"


def test_family_map_handles_the_shapes_the_lifecycle_produces():
    assert s1._steps_operators((("outlier_iqr", {}),)) == ["outlier_iqr"]
    assert s1._family_of(["outlier_iqr"]) == "outlier_threshold"
    assert s1._family_of([]) == "identity"
    assert s1._family_of(["hampel_filter", "repair_level_shift"]) == (
        "hampel+level_shift")


def test_the_round_body_stores_both_new_fields():
    """Guard against a later edit dropping the persistence again."""
    import inspect

    source = inspect.getsource(s1._run_round)
    assert '"fast_features_binned": _binned_contract_leaves(features)' in source
    assert '"proposals": proposals' in source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
