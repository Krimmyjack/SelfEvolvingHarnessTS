"""P5-A 机械测试（stub CA，不触网、不烧 confirmatory 语义——正式判决只在官方 run）。"""
import json

from SelfEvolvingHarnessTS.run_p5_identity_gate import ARMS, run_p5_identity


def test_identity_gate_mechanics_with_stub(tmp_path):
    report = run_p5_identity(seeds=(40, 41), n_per_seed=2, out_dir=tmp_path,
                             ca_backend="stub", bootstrap_b=100)
    assert set(report["arms"]) == set(ARMS)
    for arm in ARMS:
        assert report["arms"][arm]["n_episodes"] == 4
        assert isinstance(report["arms"][arm]["mean_true_delta"], float)
    assert report["arms"]["frozen"]["mean_true_delta"] == 0.0

    primary = report["primary_comparison"]
    assert primary["primary_ca_arm"] == "ca_skills"                  # 先验声明的主臂
    assert primary["baseline_arm"] == "det_search"
    assert "diff_mean" in primary and "ci90" in primary

    verdict = report["headline_criteria"]
    assert set(verdict) == {"utility_vs_det", "worst_group", "novel_effective_edits", "cost_disclosed"}
    assert report["claim_branch"] in {
        "llm_driven_harness_evolution",
        "self_updating_deterministic_with_llm_optional",
    }

    # ledger + ITT
    ledger = report["cost_ledger"]
    assert ledger["backend"] == "stub"
    assert ledger["api_calls"] == 0
    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(ARMS) * 4
    row = json.loads(lines[0])
    assert {"arm", "uid", "group_seed", "final_kind", "true_delta", "candidates"} <= set(row)


def test_identity_gate_checkpoint_resume(tmp_path):
    r1 = run_p5_identity(seeds=(40,), n_per_seed=2, out_dir=tmp_path,
                         ca_backend="stub", bootstrap_b=50)
    n_lines = len((tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines())
    r2 = run_p5_identity(seeds=(40,), n_per_seed=2, out_dir=tmp_path,
                         ca_backend="stub", bootstrap_b=50)
    n_lines2 = len((tmp_path / "records.jsonl").read_text(encoding="utf-8").strip().splitlines())
    assert n_lines2 == n_lines                                       # resume：不重复计算
    assert r2["resumed_episodes"] == len(ARMS) * 2
    assert json.dumps(r1["arms"], sort_keys=True) == json.dumps(r2["arms"], sort_keys=True)
