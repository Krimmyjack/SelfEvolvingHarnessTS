"""P1 出口判据测试：runner 三臂报表 + 按面 harm 台账 + stub 双跑可复现。"""
import json

from SelfEvolvingHarnessTS.run_p1_codeagent_first import run_p1


def test_run_p1_report_arms_manifest_and_harm_ledger(tmp_path):
    report = run_p1(n_records=4, out_dir=tmp_path / "a")
    assert set(report["arms"]) == {"raw", "incumbent_control", "code_agent_first_stub"}
    assert report["api_calls"] == 0                                   # no-API 接线验收

    arm = report["arms"]["code_agent_first_stub"]
    assert arm["n_results"] == 4
    assert "program" in arm["harm_ledger_by_surface"]                 # code agent 服务面=program
    assert arm["serve_action_prefix_counts"].get("prog1_", 0) == 4

    control = report["arms"]["incumbent_control"]
    assert control["n_results"] == 4                                  # 对照臂必在场（R2）

    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["action_menu_sha256"]
    assert manifest["packet_schema"] == "skill_memory_evidence_packet_v2"
    assert manifest["task_spec_sha"]
    assert manifest["plan"] == "Final_Plan_CodeAgentFirst_2026-07-09 §P1"

    lines = (tmp_path / "a" / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 12                                           # 3 arms × 4 records
    for line in lines:
        row = json.loads(line)
        assert row["harm_ledger"]["surface"] in {"baseline_raw", "router", "program", "gate_fallback"}
        assert "utility_delta_vs_raw" in row["harm_ledger"]


def test_run_p1_stub_backend_reproducible(tmp_path):
    r1 = run_p1(n_records=4, out_dir=tmp_path / "a")
    r2 = run_p1(n_records=4, out_dir=tmp_path / "b")
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
