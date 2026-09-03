"""E-3/C-1/C-3 tests: negative contrast bucket and source/target provenance."""
from types import SimpleNamespace

from SelfEvolvingHarnessTS.methods.ttha.group_fault import (
    build_contrast_capsule,
)


def _episode(episode_id, *, domain, support_gain, delayed_gain=None,
             workflow=("winsorize", "outlier_mad"), origin=400,
             relation=None):
    return SimpleNamespace(
        episode_id=episode_id,
        domain_namespace=domain,
        workflow_signature="|".join(workflow),
        context_summary={
            "program_geometry": {
                "program_steps": [
                    {"op": op, "params": {}}
                    for op in workflow
                ]
            },
            "support_origin": origin,
            "per_view_gain": [],
        },
        support_response={"gain": support_gain},
        delayed_response={
            "evaluated": delayed_gain is not None,
            "gain": delayed_gain,
        },
        relation=relation,
    )


def _group(episodes):
    workflow = "winsorize|outlier_mad"
    return {"workflow": workflow, "sign": "NEGATIVE", "episodes": episodes}


def test_capsule_has_negative_bucket_and_source_provenance():
    target_neg = _episode("target-neg-1", domain="target", support_gain=-0.1)
    group = _group([target_neg])
    source_neg = _episode(
        "source-neg-1", domain="source", support_gain=-0.2, origin=300
    )
    source_pos = _episode(
        "source-pos-1", domain="source", support_gain=0.1, origin=300
    )
    source_conflict = _episode(
        "source-conflict-1",
        domain="source",
        support_gain=-0.1,
        delayed_gain=0.1,
        origin=300,
    )
    filtered = _episode(
        "source-filtered-1",
        domain="source",
        support_gain=-0.1,
        workflow=("impute_linear",),
        origin=300,
    )
    capsule = build_contrast_capsule(
        group,
        all_episodes=[
            target_neg, source_neg, source_pos, source_conflict, filtered
        ],
        target_domain_namespace="target",
    )
    assert set(capsule["contrast_cases"]) == {
        "positive", "negative", "conflict"
    }
    by_id = {
        ref["episode_id"]: ref
        for bucket in capsule["contrast_cases"].values()
        for ref in bucket
    }
    assert by_id["source-neg-1"]["provenance"] == "source"
    assert by_id["source-neg-1"]["support_gain"] == -0.2
    assert by_id["source-pos-1"]["provenance"] == "source"
    assert by_id["source-conflict-1"]["provenance"] == "source"
    assert "target-neg-1" not in by_id  # group member excluded from contrast

    provenance = capsule["source_provenance"]
    assert provenance["target_domain_namespace"] == "target"
    assert "source-neg-1" in provenance["source_episode_ids"]
    assert "source-pos-1" in provenance["source_episode_ids"]
    assert provenance["referenced_source_episode_ids"] == [
        "source-conflict-1", "source-neg-1", "source-pos-1"
    ]
    assert provenance["filtered_source_episode_ids"] == ["source-filtered-1"]
    assert provenance["filtered_source_episode_count"] == 1
    assert capsule["per_episode_rows"][0]["provenance"] == "target"


def test_candidate_conditioned_retrieval_keeps_source_candidate_episodes():
    target_neg = _episode(
        "target-neg-1", domain="target", support_gain=-0.1,
        workflow=("winsorize",), origin=888,
    )
    source_candidate = _episode(
        "source-outlier-1",
        domain="source",
        support_gain=-0.2,
        workflow=("outlier_mad",),
        origin=300,
    )
    capsule = build_contrast_capsule(
        {"workflow": "winsorize", "sign": "NEGATIVE",
         "episodes": [target_neg]},
        all_episodes=[target_neg, source_candidate],
        target_domain_namespace="target",
        candidate_workflows=("outlier_mad",),
    )
    assert capsule["retrieval_scope"]["incumbent_workflow"] == "winsorize"
    assert capsule["retrieval_scope"]["candidate_workflows"] == ["outlier_mad"]
    assert capsule["source_provenance"][
        "referenced_source_episode_ids"
    ] == ["source-outlier-1"]
    assert capsule["source_provenance"]["filtered_source_episode_ids"] == []


def test_relation_wins_over_numeric_order_for_support_positive_delayed_negative():
    target_neg = _episode("target-neg-1", domain="target", support_gain=-0.1)
    flipped = _episode(
        "source-flip-1",
        domain="source",
        support_gain=0.1,
        delayed_gain=-0.1,
        relation="CONFLICT",
        origin=300,
    )
    capsule = build_contrast_capsule(
        _group([target_neg]),
        all_episodes=[target_neg, flipped],
        target_domain_namespace="target",
    )
    positive_ids = {
        ref["episode_id"] for ref in capsule["contrast_cases"]["positive"]
    }
    conflict_ids = {
        ref["episode_id"] for ref in capsule["contrast_cases"]["conflict"]
    }
    assert "source-flip-1" not in positive_ids
    assert "source-flip-1" in conflict_ids


def test_missing_target_domain_reports_unknown_not_source():
    target_neg = _episode("target-neg-1", domain="target", support_gain=-0.1)
    capsule = build_contrast_capsule(
        _group([target_neg]),
        all_episodes=[target_neg],
    )
    assert set(capsule["contrast_cases"]) == {
        "positive", "negative", "conflict"
    }
    assert capsule["source_provenance"]["target_domain_namespace"] is None
    assert capsule["source_provenance"]["source_episode_ids"] == []
    assert capsule["per_episode_rows"][0]["provenance"] == "unknown"
