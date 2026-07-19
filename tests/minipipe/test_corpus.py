import json
from pathlib import Path

import numpy as np
import pytest

from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import (
    build_core_corpus,
    build_heldout_corpus,
)


RULES = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "minipipe"
    / "config"
    / "m0_rules.json"
)


@pytest.fixture
def m0_rules():
    return load_m0_rules(RULES)


def test_core_corpus_is_24_targets_plus_12_risks_and_reproducible(m0_rules):
    first = build_core_corpus(m0_rules)
    second = build_core_corpus(m0_rules)
    assert len(first.targets) == 24
    assert len(first.risks) == 12
    assert [case.private_sha for case in first.all_cases] == [
        case.private_sha for case in second.all_cases
    ]
    assert {case.private_family for case in first.targets} == {
        "missing",
        "impulsive_outlier",
        "level_shift",
        "period_change",
    }
    target_numbers = {int(case.case_id.split("-")[1]) for case in first.targets}
    assert target_numbers != set(range(1, 25))


def test_heldout_builder_allows_preregistered_size_without_weakening_core_lock():
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    rules["corpus"]["base_seeds"] = [404, 505, 606, 707]

    heldout = build_heldout_corpus(rules)

    assert len(heldout.targets) == 32
    assert len(heldout.risks) == 16
    with pytest.raises(ValueError, match="24 targets and 12 risks"):
        build_core_corpus(rules)


def test_public_view_contains_only_corrupt_context_and_opaque_identity(m0_rules):
    case = build_core_corpus(m0_rules).targets[0]
    public = case.to_public_view()
    assert public.case_id.startswith("m0-")
    assert case.private_family not in json.dumps(public.to_json())
    assert "injection" not in json.dumps(public.to_json()).lower()
    np.testing.assert_array_equal(public.values, case.corrupt_context)
    assert "clean_context" not in public.to_json()
    assert "clean_future" not in public.to_json()


def test_public_and_private_artifacts_are_written_to_disjoint_roots(tmp_path, m0_rules):
    from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import write_case_artifacts

    corpus = build_core_corpus(m0_rules)
    roots = write_case_artifacts(corpus.all_cases, tmp_path)
    assert roots.public.resolve().parent == roots.private.resolve().parent
    assert roots.public.resolve() != roots.private.resolve()
    public_text = "\n".join(
        path.read_text(encoding="utf-8") for path in roots.public.rglob("*.json")
    )
    assert "private_family" not in public_text
    assert "clean_future" not in public_text
    assert len(tuple(roots.public.glob("*.json"))) == 36
    assert len(tuple(roots.private.glob("*.json"))) == 36


def test_target_injections_change_context_only_and_keep_future_clean(m0_rules):
    from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.injections import (
        CONTEXT_LENGTH,
        generate_base_series,
    )

    corpus = build_core_corpus(m0_rules)
    for case in corpus.targets:
        np.testing.assert_array_equal(
            case.clean_future,
            generate_base_series(case.seed)[CONTEXT_LENGTH:],
        )
        assert np.any(
            np.not_equal(case.clean_context, case.corrupt_context)
            & ~(np.isnan(case.clean_context) & np.isnan(case.corrupt_context))
        )
        assert case.oracle_affected_indices
        assert case.observable_counterpart_id is not None


def test_risk_cases_have_no_repair_target_and_are_their_own_clean_data(m0_rules):
    for case in build_core_corpus(m0_rules).risks:
        assert case.oracle_affected_indices == ()
        np.testing.assert_array_equal(case.clean_context, case.corrupt_context)
