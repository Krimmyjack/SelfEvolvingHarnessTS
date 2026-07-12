"""P5-B/C 机械测试：四象限迁移（真 Monash）+ safety 收口（confirmatory slice 一次性）。"""
import json

from SelfEvolvingHarnessTS.run_p5_quadrant import QUADRANTS, run_p5_quadrant
from SelfEvolvingHarnessTS.run_p5_safety import run_p5_safety


def test_quadrant_transfer_structure(tmp_path):
    report = run_p5_quadrant(out_dir=tmp_path, bootstrap_b=100)
    assert report["n_episodes"] == 48                                # 12 基底 × 4 preset
    assert set(report["quadrant_mean_regret"]) == set(QUADRANTS)
    for q in QUADRANTS:
        assert report["quadrant_coverage"][q] > 0                    # 每象限都有可评估目标
    hyp = report["hypothesis_dd_sp_vs_sd_dp"]
    assert "paired_diff_mean" in hyp and "ci90" in hyp
    assert hyp["direction_holds"] in (True, False)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pattern_axis"] == "degradation_preset_cell"
    assert manifest["domain_axis"] == "dataset_config"
    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 48


def test_quadrant_deterministic(tmp_path):
    r1 = run_p5_quadrant(out_dir=tmp_path / "a", bootstrap_b=50)
    r2 = run_p5_quadrant(out_dir=tmp_path / "b", bootstrap_b=50)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_safety_closeout_structure(tmp_path):
    report = run_p5_safety(seeds=(40, 41), n_per_seed=2, out_dir=tmp_path, bootstrap_b=100)
    assert report["one_shot"] is True
    fc = report["forecast"]
    assert fc["coverage_fired"] > 0                                  # 晋升规则在 snrLow 触发
    assert "gain_vs_v0_mean" in fc and "harm_rate_vs_v0" in fc
    assert "worst_cell_lcb" in fc
    ad = report["anomaly_detection"]
    assert ad["rows_identical_to_v0"] is True                        # anomaly 面零扰动
    assert report["s2_sealed_holdout"] == "deferred_separate_registered_access"
