"""P5-A.3 机械测试（stub，不触网、不消耗 confirmatory 语义）。"""
import json

from SelfEvolvingHarnessTS.run_p5a3_final import ARMS, PRIMARY_ARM, run_p5a3


def test_p5a3_mechanics_stub(tmp_path):
    report = run_p5a3(seeds=(80, 81), n_per_seed=2, out_dir=tmp_path,
                      backend="stub", bootstrap_b=50)
    assert set(report["arms"]) == set(ARMS)
    for arm in ARMS:
        assert report["arms"][arm]["n_episodes"] == 4
    primary = report["arms"][PRIMARY_ARM]
    assert primary["compliance_valid_rate"] == 1.0                   # 编译器按构造保证合规
    assert primary["selection_mean_regret"] is not None
    assert primary["selection_mean_regret"] >= 0.0
    assert "semantic_guard_drop_rate" in primary                     # 四指标：semantic 台账在
    assert report["claim_branch"] in {"llm_driven_harness_evolution",
                                      "self_updating_deterministic_with_llm_novelty_supplier"}
    assert report["cost_ledger"]["composer_calls"] == 0              # stub 0-API

    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(ARMS) * 4
    row = json.loads(lines[0])
    assert {"series_family", "pattern_preset", "selection_regret"} <= set(row)   # 命名修正字段
    assert "anomaly|" not in row["series_family"]

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prereg"].endswith("prereg_p5a3_final.md")
    assert "LODO" in manifest["continuous_evidence"]


def test_p5a3_resume_and_determinism(tmp_path):
    r1 = run_p5a3(seeds=(80,), n_per_seed=2, out_dir=tmp_path, backend="stub", bootstrap_b=30)
    n1 = len((tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines())
    r2 = run_p5a3(seeds=(80,), n_per_seed=2, out_dir=tmp_path, backend="stub", bootstrap_b=30)
    n2 = len((tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines())
    assert n1 == n2
    assert r2["resumed_episodes"] == len(ARMS) * 2
    assert json.dumps(r1["arms"], sort_keys=True) == json.dumps(r2["arms"], sort_keys=True)
