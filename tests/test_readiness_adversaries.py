import json

import pytest

from SelfEvolvingHarnessTS.evaluators.readiness_adversary import (
    build_adversary_report,
    evaluate_policies,
    policies_from_record_arms,
)


def _records():
    return [
        {
            "uid": "a",
            "origin": "S_a",
            "cell": "forecast|snrLow|miss",
            "L_test": {"v_none": 10.0, "clean": 7.0, "bad": 11.0},
            "arms": {"record_good": {"pick": "clean", "abstain": False}},
        },
        {
            "uid": "b",
            "origin": "S_b",
            "cell": "forecast|snrHigh|full",
            "L_test": {"v_none": 5.0, "clean": 5.1, "bad": 6.0},
            "arms": {"record_good": {"pick": "v_none", "abstain": True}},
        },
        {
            "uid": "c",
            "origin": "S_a",
            "cell": "forecast|snrLow|full",
            "L_test": {"v_none": 4.0, "clean": 3.0, "bad": 8.0},
            "arms": {"record_good": {"pick": "clean", "abstain": False}},
        },
    ]


def test_evaluate_policies_scores_actionable_readiness_against_raw():
    records = _records()
    policies = {
        "raw": {"a": "v_none", "b": "v_none", "c": "v_none"},
        "good": {"a": "clean", "b": "v_none", "c": "clean"},
        "bad": {"a": "bad", "b": "bad", "c": "bad"},
    }

    report = evaluate_policies(records, policies)

    assert report["oracle"]["n_records"] == 3
    assert report["oracle"]["n_actionable"] == 2
    assert report["oracle"]["actionable_rate"] == pytest.approx(2 / 3)

    good = report["policies"]["good"]
    assert good["mean_regret"] == pytest.approx(0.0)
    assert good["mean_gain_vs_raw"] == pytest.approx((3.0 + 0.0 + 1.0) / 3)
    assert good["top1_oracle_rate"] == pytest.approx(1.0)
    assert good["readiness_precision"] == pytest.approx(1.0)
    assert good["readiness_recall"] == pytest.approx(1.0)
    assert good["readiness_f1"] == pytest.approx(1.0)
    assert good["harm_rate"] == pytest.approx(0.0)

    raw = report["policies"]["raw"]
    assert raw["mean_regret"] == pytest.approx((3.0 + 0.0 + 1.0) / 3)
    assert raw["readiness_recall"] == pytest.approx(0.0)
    assert raw["readiness_accuracy"] == pytest.approx(1 / 3)

    bad = report["policies"]["bad"]
    assert bad["harm_rate"] == pytest.approx(1.0)
    assert bad["mean_regret"] == pytest.approx((4.0 + 1.0 + 5.0) / 3)


def test_record_arms_preserve_abstain_metadata():
    policies, abstain = policies_from_record_arms(_records(), ["record_good"])

    assert policies == {"record_good": {"a": "clean", "b": "v_none", "c": "clean"}}
    assert abstain == {"record_good": {"a": False, "b": True, "c": False}}

    report = evaluate_policies(_records(), policies, abstain_by_policy=abstain)
    assert report["policies"]["record_good"]["abstain_rate"] == pytest.approx(1 / 3)


def test_build_adversary_report_writes_json_csv_and_markdown(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(r) for r in _records()),
        encoding="utf-8",
    )
    static_path = tmp_path / "static.json"
    static_path.write_text(json.dumps({"a": "clean", "b": "v_none", "c": "clean"}), encoding="utf-8")

    out_dir = tmp_path / "out"
    report = build_adversary_report(
        records_path,
        out_dir,
        record_arm_names=["record_good"],
        external_pick_paths={"static": static_path},
    )

    assert report["policies"]["static"]["mean_regret"] == pytest.approx(0.0)
    assert (out_dir / "report.json").exists()
    assert (out_dir / "decision_rows.csv").exists()
    assert (out_dir / "policy_summary.csv").exists()
    assert "static" in (out_dir / "table.md").read_text(encoding="utf-8")
    json.loads(
        (out_dir / "report.json").read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )

