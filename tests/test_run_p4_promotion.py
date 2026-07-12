"""P4 出口判据测试：完整晋升周期（miner→gate→true 判官验证→版本升级→回归重放→rollback）。"""
import json

from SelfEvolvingHarnessTS.run_p4_promotion import run_p4


def test_run_p4_full_promotion_cycle(tmp_path):
    report = run_p4(n_series=16, out_dir=tmp_path, seed=41)

    # 晋升：mined ban w25→w9 @ snrLow 被 true 判官双段接受
    promoted = report["promoted"]
    assert promoted["version"].startswith("bundle_v0.e")
    assert promoted["rule_id"] == "mined_ban_f0_median_w25_snr_low"
    assert promoted["validation"]["accepted"]
    assert promoted["validation"]["held_out"]["mean_gain"] >= 0.02   # P3 冻结 ε 口径
    assert promoted["validation"]["non_targeted_identical"] is True

    # 拒绝缓冲：反向规则 + 伤害规则均被 true 判官拒绝并留痕
    assert report["n_rejected"] == 2
    rejected_lines = (tmp_path / "rejected_edits.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rejected_lines) == 2
    for line in rejected_lines:
        row = json.loads(line)
        assert row["reasons"]

    # 回归重放：非目标行（forecast snrHigh + 全部 anomaly）bit 级不变，目标行有增益
    assert report["regression"]["non_targeted_identical"] is True
    assert report["regression"]["targeted_mean_gain"] > 0
    assert report["regression"]["anomaly_rows_identical"] is True

    # rollback：head 回 v0 后 serving 与原 v0 bit 级一致；随后恢复晋升头；事件全留痕
    assert report["rollback"]["verified"] is True
    chain = json.loads((tmp_path / "bundles" / "chain.json").read_text(encoding="utf-8"))
    events = [e["event"] for e in chain["events"]]
    assert events.count("save") == 2
    assert events.count("rollback") == 2                             # demo 回退 + 恢复头
    assert chain["head"] == promoted["version"]

    # 出口判据聚合
    assert report["exit_criteria"]["cycle_complete"] is True
    # Memory 阶梯按 prereg §4 为条件线：本轮只有 risk-veto 在 gate 活跃，utility 阶梯显式挂起
    assert report["memory_ladder"]["status"] == "conditional_pending"


def test_run_p4_deterministic(tmp_path):
    r1 = run_p4(n_series=12, out_dir=tmp_path / "a", seed=7)
    r2 = run_p4(n_series=12, out_dir=tmp_path / "b", seed=7)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
