import json

import pytest

from SelfEvolvingHarnessTS.evaluators.safety_gate_lite import (
    abstain_to_raw_policy,
    build_safety_gate_report,
    support_gated_policy,
)
from SelfEvolvingHarnessTS.run_safety_gate_lite import (
    conditioning_key_from_record,
    support_scores_from_router,
)


def _records():
    return [
        {
            "uid": "a",
            "origin": "S_a",
            "cell": "forecast|snrLow|miss",
            "L_test": {"v_none": 10.0, "clean": 7.0, "v_median": 11.0},
            "arms": {"dp_abstain": {"pick": "clean", "abstain": False}},
        },
        {
            "uid": "b",
            "origin": "S_b",
            "cell": "forecast|snrHigh|full",
            "L_test": {"v_none": 5.0, "clean": 4.8, "v_median": 6.0},
            "arms": {"dp_abstain": {"pick": "v_median", "abstain": True}},
        },
    ]


def test_abstain_to_raw_policy_only_replaces_abstained_picks():
    picks = {"a": "clean", "b": "v_median", "c": "clean"}
    abstains = {"a": False, "b": True}

    gated = abstain_to_raw_policy(picks, abstains, raw_action="v_none")

    assert gated == {"a": "clean", "b": "v_none", "c": "clean"}


def test_safety_gate_report_scores_abstain_to_raw_variant(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(r) for r in _records()),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    report = build_safety_gate_report(records_path, out_dir, policy_name="dp_abstain")

    base = report["policies"]["dp_abstain"]
    gated = report["policies"]["dp_abstain_abstain_to_raw"]
    assert base["harm_rate"] == pytest.approx(0.5)
    assert gated["harm_rate"] == pytest.approx(0.0)
    assert gated["mean_gain_vs_raw"] > 0
    assert (out_dir / "report.json").exists()
    assert (out_dir / "decision_rows.csv").exists()
    assert (out_dir / "policy_summary.csv").exists()
    assert "dp_abstain_abstain_to_raw" in (out_dir / "table.md").read_text(encoding="utf-8")
    json.loads(
        (out_dir / "report.json").read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def test_support_gated_policy_forces_raw_when_support_is_weak():
    picks = {"a": "clean", "b": "clean", "c": "clean"}
    abstains = {"a": False, "b": False, "c": True}
    support_scores = {"a": 0.1, "b": 0.9, "c": 0.1}

    gated = support_gated_policy(
        picks,
        abstains,
        support_scores,
        max_support_score=0.5,
        raw_action="v_none",
    )

    assert gated == {"a": "clean", "b": "v_none", "c": "v_none"}


def test_safety_gate_report_includes_support_sweep_when_scores_are_available(tmp_path):
    records = _records() + [
        {
            "uid": "c",
            "origin": "S_c",
            "cell": "forecast|snrLow|full",
            "L_test": {"v_none": 3.0, "clean": 4.0, "v_median": 4.5},
            "arms": {"dp_abstain": {"pick": "clean", "abstain": False}},
        }
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(r) for r in records),
        encoding="utf-8",
    )

    report = build_safety_gate_report(
        records_path,
        tmp_path / "out",
        policy_name="dp_abstain",
        support_scores={"a": 0.1, "b": 0.2, "c": 0.9},
        support_quantiles=[0.5],
    )

    strict = report["policies"]["dp_abstain_support_q50"]
    assert strict["harm_rate"] < report["policies"]["dp_abstain_abstain_to_raw"]["harm_rate"]
    assert report["safety_gate"]["support_sweep"][0]["gate_name"] == "support_q50"

def test_conditioning_key_from_record_preserves_router_features():
    record = {
        "uid": "u1",
        "cell": "forecast|snrLow|miss",
        "snr": -3.5,
        "miss_rate": 0.25,
        "X_p": [1.0, 0.2, 0.3, 0.4, 0.0, 0.6, 0.7, 0.8],
    }

    key = conditioning_key_from_record(record)

    assert key["task"] == {"type": "forecast"}
    assert key["cell_id"] == "forecast|snrLow|miss"
    assert key["pattern"]["struct_feats"]["SNR"] == -3.5
    assert key["pattern"]["struct_feats"]["missing_rate"] == 0.25
    assert key["pattern"]["struct_feats"]["period"] == 1.0
    assert key["pattern"]["struct_feats"]["outlier_density"] == 0.8


def test_support_scores_from_router_reads_provenance_distance():
    class FakeRouter:
        def predict(self, conditioning_key, action_menu):
            assert conditioning_key["task"]["type"] == "forecast"
            return type(
                "Decision",
                (),
                {"provenance": {"support": {"available": True, "distance": 0.42}}},
            )()

    records = [
        {
            "uid": "u1",
            "cell": "forecast|snrLow|miss",
            "snr": -3.5,
            "miss_rate": 0.25,
            "X_p": [1.0, 0.2, 0.3, 0.4, 0.0, 0.6, 0.7, 0.8],
        }
    ]

    scores = support_scores_from_router(records, FakeRouter(), action_menu=object())

    assert scores == {"u1": 0.42}

