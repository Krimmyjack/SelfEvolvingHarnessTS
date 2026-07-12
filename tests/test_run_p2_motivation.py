"""P2 出口判据测试：动机表（同一批序列、两任务判官、符号翻转 + 契约列 + 复现）。"""
import json

from SelfEvolvingHarnessTS.run_p2_motivation import run_p2


def test_run_p2_motivation_table_flips_and_contract(tmp_path):
    report = run_p2(n_series=8, out_dir=tmp_path, bootstrap_b=200, seed=11)
    rows = report["table"]
    assert {"v_raw_identity", "v_impute_linear", "median_w9", "winsor",
            "universal_cleaner", "task_conditioned"} <= set(rows)

    uc = rows["universal_cleaner"]
    assert uc["forecast"]["mean_delta"] > 0                      # 清洗助 forecast（去噪去尖峰）
    assert uc["anomaly_detection"]["mean_delta"] < 0             # 同一程序毁 anomaly（抹掉检测目标）
    assert uc["deployable_under_contract"]["forecast"] is True
    assert uc["deployable_under_contract"]["anomaly_detection"] is False   # registry 物理禁

    tc = rows["task_conditioned"]
    assert tc["anomaly_detection"]["mean_delta"] >= rows["universal_cleaner"]["anomaly_detection"]["mean_delta"]

    assert report["fresh_flip_pairs"] >= 1                       # forecast×anomaly 翻转（fresh）
    assert report["exit_criterion_met"] is True                  # fresh 1 对 + frozen classify 引用
    assert report["classify_citation"]["source"].endswith("_clf_maintable.log")

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_spec_shas"]["anomaly_detection"]
    assert manifest["task_spec_shas"]["forecast"]
    assert manifest["epsilon"] == 0.01
    assert manifest["detector"]["frozen"] is True
    assert manifest["reference_action"] == "v_raw_identity"

    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8 * 6 * 2                               # series × programs × tasks


def test_run_p2_deterministic(tmp_path):
    r1 = run_p2(n_series=6, out_dir=tmp_path / "a", bootstrap_b=50, seed=5)
    r2 = run_p2(n_series=6, out_dir=tmp_path / "b", bootstrap_b=50, seed=5)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
