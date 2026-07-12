"""P3 出口判据测试：fidelity 报告 + headroom/ε 注册 + skill bank 落盘 + 0-API + 复现。"""
import json

from SelfEvolvingHarnessTS.run_p3_gym import run_p3


def test_run_p3_structure_fidelity_headroom_bank(tmp_path):
    report = run_p3(n_series=8, out_dir=tmp_path, bootstrap_b=100, seed=13)

    # fidelity（R4 硬门材料）：主判据 = within-series 排序保真（决策相关口径），
    # pooled 仅诊断（跨序列尺度差会制造 Simpson 反向——首轮实测 −0.32 教训）
    assert set(report["fidelity"]) == {"forecast", "anomaly_detection"}
    fc = report["fidelity"]["forecast"]
    assert fc["status"] in {"pass", "fail"}
    assert isinstance(fc["within_series_mean_rho"], float)
    assert fc["n_series_scored"] > 0
    assert "pooled_rho_diagnostic" in fc
    ad = report["fidelity"]["anomaly_detection"]
    assert ad["status"] in {"pass", "fail", "insufficient_variance"}
    assert "diagnostic_with_violations" in ad          # 违约诊断组证明 proxy 有牙齿
    assert ad["diagnostic_with_violations"]["n_pairs"] > 0

    # headroom + ε（prereg §3 注册规则：ε = max(0.02, forecast headroom ci90_lo)）
    hr = report["headroom"]
    assert hr["forecast"]["mean"] >= 0.0
    assert hr["anomaly_detection"]["mean"] >= 0.0
    assert report["epsilon_registered"] >= 0.02
    assert report["epsilon_rule"].startswith("max(0.02")

    # skill bank v1 落盘
    bank = json.loads((tmp_path / "skill_bank_v1.json").read_text(encoding="utf-8"))
    assert len(bank["programs"]) == 8
    assert all(entry["novel_vs_menu_v1"] for entry in bank["programs"].values())
    assert bank["bank_version"] == "bank_v1"

    # 0-API + gym 认证
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["api_calls"] == 0
    assert manifest["gym_smoke"]["forecast"]["final_kind"] in {"program", "abstain"}
    assert manifest["gym_smoke"]["anomaly_detection"]["final_kind"] in {"program", "abstain"}
    assert manifest["rho_min"] == 0.7

    # records：8×(15 menu + 6 seed) forecast + 8×(1 menu + 2 seed) anomaly + 8×3 违约诊断
    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8 * 21 + 8 * 3 + 8 * 3
    row = json.loads(lines[0])
    assert {"uid", "task", "source", "name", "true_delta", "proxy_delta", "deployable"} <= set(row)


def test_run_p3_deterministic(tmp_path):
    r1 = run_p3(n_series=4, out_dir=tmp_path / "a", bootstrap_b=50, seed=5)
    r2 = run_p3(n_series=4, out_dir=tmp_path / "b", bootstrap_b=50, seed=5)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
