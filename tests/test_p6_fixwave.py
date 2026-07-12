"""tests/test_p6_fixwave.py — P6 签发前修复波回归测试（codex 二轮复审 finding 31-37 + 建议 41）。

每个修复至少一条回归测试，测试名含对应编号（test_f1_… ~ test_f7_…、test_f9_…）。
红线：全合成 + 注入 stub；不联网、不读真实效用数据（C0/D/V/U）、不 import torch、无 LLM/git；
文件 IO 只发生在 pytest tmp（--basetemp 指到 scratchpad）。

运行：D:\\Anaconda_envs\\envs\\project\\python.exe -m pytest SelfEvolvingHarnessTS/tests/test_p6_fixwave.py -q
（cwd = C:\\Users\\辉\\Desktop\\Agent，加 --basetemp 指到 scratchpad）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from SelfEvolvingHarnessTS.p6.c0_runner import (
    P6Episode,
    P6FrozenParamError,
    assert_c0_frozen_params,
)
from SelfEvolvingHarnessTS.p6.cycle_runner import assert_cycle_frozen_params
from SelfEvolvingHarnessTS.p6.u_runner import assert_u_frozen_params
from SelfEvolvingHarnessTS.p6.fast_path import Candidate, apply_risk, merge_preset_fingerprints
from SelfEvolvingHarnessTS.p6.harness_state import (
    RiskRuleSpec,
    apply_edit,
    default_state,
)
from SelfEvolvingHarnessTS.p6.edit_surfaces import RiskRulePatch
from SelfEvolvingHarnessTS.p6.miner import mine
from SelfEvolvingHarnessTS.p6.final_packet import (
    CLAIM_BRANCHES,
    load_final_packet,
    write_final_packet,
)

_PKG_ROOT = Path(__file__).resolve().parents[1]
_PREREG = _PKG_ROOT / "results" / "Stage2" / "prereg_p6.md"

BIG4 = ("nn5_daily", "fred_md", "tourism_monthly", "covid_deaths")
PRESETS4 = ("G_hi_full", "G_hi_miss", "G_lo_full", "G_lo_miss")


def _dummy_series():
    return np.zeros(160, dtype=float)


def _c0_episodes(n_domains=4, series_per_domain=4, presets=PRESETS4):
    """64 episode 结构桩（16 series × 4 preset，4 域）：assert 只读 config/series_uid/preset/len。"""
    eps = []
    for d in BIG4[:n_domains]:
        for si in range(series_per_domain):
            for p in presets:
                eps.append(P6Episode(
                    uid=f"{d}:s{si}:{p}", series_uid=f"{d}:s{si}",
                    config=d, preset=p, history=_dummy_series(), future=_dummy_series()[:48],
                ))
    return eps


# ══════════════════════════════════════════════════════════════════════════
# F6（finding 36）：formal 模式冻结字面量断言
# ══════════════════════════════════════════════════════════════════════════
def test_f6_formal_c0_accepts_frozen_and_rejects_drift():
    eps = _c0_episodes()
    assert len(eps) == 64
    assert_c0_frozen_params(eps, (0, 1, 2))                     # 冻结结构 → 不 raise
    with pytest.raises(P6FrozenParamError, match="seeds"):
        assert_c0_frozen_params(eps, (0, 1))                    # 2 seeds
    with pytest.raises(P6FrozenParamError, match="seeds"):
        assert_c0_frozen_params(eps, (0, 1, 5))                 # 非 (0,1,2)
    with pytest.raises(P6FrozenParamError, match="episodes"):
        assert_c0_frozen_params(eps[:-1], (0, 1, 2))            # 63 episode
    with pytest.raises(P6FrozenParamError, match="域数"):
        assert_c0_frozen_params(_c0_episodes(n_domains=2, series_per_domain=8), (0, 1, 2))
    with pytest.raises(P6FrozenParamError, match="preset"):
        assert_c0_frozen_params(_c0_episodes(presets=PRESETS4[:2] + ("x", "y", "z", "w")[:2]) +
                                _c0_episodes(), (0, 1, 2))       # preset 数漂移 + 数量漂移


def test_f6_formal_cycle_rejects_drift():
    assert_cycle_frozen_params((0, 1, 2), 2000, 20260712, 1, 8)   # 冻结值 → 不 raise
    assert_cycle_frozen_params((0, 1, 2), 2000, 20260713, 2, 8)
    with pytest.raises(P6FrozenParamError, match="bootstrap_b"):
        assert_cycle_frozen_params((0, 1, 2), 100, 20260712, 1, 8)
    with pytest.raises(P6FrozenParamError, match="bootstrap seed"):
        assert_cycle_frozen_params((0, 1, 2), 2000, 20260799, 1, 8)
    with pytest.raises(P6FrozenParamError, match="seeds"):
        assert_cycle_frozen_params((0, 1), 2000, 20260712, 1, 8)
    with pytest.raises(P6FrozenParamError, match="K"):
        assert_cycle_frozen_params((0, 1, 2), 2000, 20260712, 1, 5)


def test_f6_formal_u_rejects_drift():
    assert_u_frozen_params((0, 1, 2), 2000, 20260714, 8)          # 冻结值 → 不 raise
    with pytest.raises(P6FrozenParamError, match="bootstrap seed"):
        assert_u_frozen_params((0, 1, 2), 2000, 20260712, 8)      # cycle seed 不是 U seed
    with pytest.raises(P6FrozenParamError, match="bootstrap_b"):
        assert_u_frozen_params((0, 1, 2), 500, 20260714, 8)
    with pytest.raises(P6FrozenParamError, match="K"):
        assert_u_frozen_params((0, 1, 2), 2000, 20260714, 6)


def test_f6_drift_guard_literals_match_prereg_text():
    """drift-guard：formal 断言里的冻结字面量必须与 prereg_p6.md 文本一致（防码/文漂移）。"""
    text = _PREREG.read_text(encoding="utf-8")
    # bootstrap B、cycle/U 两个 seed、K、C0 结构计数、paired seeds —— 逐一在 prereg 文本中出现
    assert "B=2000" in text
    assert "20260711+cycle" in text
    assert "20260714" in text                                    # U 终评 seed
    assert "96 次 Adam 拟合" in text                              # C0 = 8×4×3
    assert "K=8" in text or "K = 8" in text
    assert "{0,1,2}" in text or "{0, 1, 2}" in text
    # 64 = 16 series × 4 preset（§3.2 显式给出 "64"）
    assert "64" in text
    # G4：trainer 超时 900.0 + 正式入口唯一性条款
    assert "900.0" in text
    assert "run_cycle_formal" in text and "run_u_eval_formal" in text and "run_c0_formal" in text


# ══════════════════════════════════════════════════════════════════════════
# F7（finding 37）：RiskRule preset scope = 成员资格，非 C0 中位数半平面近似
# ══════════════════════════════════════════════════════════════════════════
def _ban_target(rule: RiskRuleSpec, fingerprint):
    """apply_risk 该单规则 state 于一个候选（op=denoise_median），返回该候选是否被 ban。"""
    state = apply_edit(default_state(), RiskRulePatch(add_rule=rule))
    cand = Candidate(program_steps=(("denoise_median", {"window": 9}),), source="det")
    _kept, banned = apply_risk([cand], fingerprint, state)
    return len(banned) == 1


def test_f7_preset_scope_is_membership_not_halfplane():
    """构造半平面近似与成员资格给出**不同 ban 集**的场景，断言新实现取成员资格。

    两 episode 的数值 snr 与其 preset **矛盾**（低 snr preset 却测得高 snr，反之亦然）：
      - ep_lo：preset='snr_lo_full'，测得 snr=100（高）；
      - ep_hi：preset='snr_hi_full'，测得 snr=0.1（低）。
    旧半平面把 'snr_lo_full' 翻成 (snr < 中位)：会 ban 错 episode（ep_hi）；
    新成员资格 (preset == 'snr_lo_full')：ban 对的 episode（ep_lo）。两 ban 集不相等。"""
    fp_lo = {"snr": 100.0, "missing_rate": 0.0, "preset": "snr_lo_full"}
    fp_hi = {"snr": 0.1, "missing_rate": 0.0, "preset": "snr_hi_full"}

    # 旧半平面近似规则（snr < 中位 5.0）——重现 finding 37 的错误 scope
    halfplane = RiskRuleSpec(rule_id="hp", when=[{"feature": "snr", "op": "<", "value": 5.0}],
                             then={"action": "ban", "target": "denoise_median"})
    hp_banset = {name for name, fp in (("lo", fp_lo), ("hi", fp_hi)) if _ban_target(halfplane, fp)}
    assert hp_banset == {"hi"}                                  # 半平面 ban 了错的（按数值 snr）

    # 新成员资格规则（preset == 'snr_lo_full'）
    membership = RiskRuleSpec(rule_id="mem",
                              when=[{"feature": "preset", "op": "==", "value": "snr_lo_full"}],
                              then={"action": "ban", "target": "denoise_median"})
    mem_banset = {name for name, fp in (("lo", fp_lo), ("hi", fp_hi)) if _ban_target(membership, fp)}
    assert mem_banset == {"lo"}                                 # 成员资格 ban 对的（按 preset）
    assert mem_banset != hp_banset                             # 两 ban 集确实不同

    # 新实现（miner）从 preset cohort 产出的正是成员资格规则，复现 membership ban 集
    cohort = {"cohort_id": "preset:snr_lo_full", "preset": "snr_lo_full"}
    ev = {"cohort": cohort, "accused_sha": "denoise_median", "accused_ops": ["denoise_median"],
          "fingerprints": []}
    cands = mine("risk", ev, default_state())
    risk_a = next(c for c in cands if c.recipe_id == "risk_a")
    rule_dict = risk_a.proposal_dict["add_rule"]
    assert rule_dict["when"] == [{"feature": "preset", "op": "==", "value": "snr_lo_full"}]
    mined_rule = RiskRuleSpec.from_dict(rule_dict)
    mined_banset = {name for name, fp in (("lo", fp_lo), ("hi", fp_hi))
                    if _ban_target(mined_rule, fp)}
    assert mined_banset == {"lo"} == mem_banset                # 新实现 = 成员资格语义


def test_f7_merge_preset_fingerprints_preserves_numeric_and_adds_preset():
    """merge_preset_fingerprints 让 preset 可见但保留数值特征（与无 preset 口径一致）。"""
    class _Ep:
        def __init__(self, uid, preset, hist):
            self.uid, self.preset, self.history = uid, preset, hist
    eps = [_Ep("nn5:s0:snr_lo_full", "snr_lo_full", np.ones(160)),
           _Ep("nn5:s0:snr_hi_full", "snr_hi_full", np.ones(160))]
    base = {"nn5:s0:snr_lo_full": {"snr": 3.0, "missing_rate": 0.1}}
    merged = merge_preset_fingerprints(eps, base)
    # 提供了 base 的 uid：数值原样保留 + preset 注入
    assert merged["nn5:s0:snr_lo_full"] == {"snr": 3.0, "missing_rate": 0.1, "preset": "snr_lo_full"}
    # 未提供 base 的 uid：toy_fingerprint 数值 + preset
    m2 = merged["nn5:s0:snr_hi_full"]
    assert m2["preset"] == "snr_hi_full" and "snr" in m2 and "missing_rate" in m2


# ══════════════════════════════════════════════════════════════════════════
# F9（建议 41）：最终结果包最小写入器（外锚 ledger chain_tip）
# ══════════════════════════════════════════════════════════════════════════
def test_f9_final_packet_round_trip(tmp_path):
    freeze = {"prereg_p6.md": "a" * 64, "cycle_runner.py": "b" * 64}
    path = tmp_path / "final_packet.json"          # 调用方给路径；本波不产生真实 results 文件
    sha = write_final_packet(
        path, chain_tip="c" * 64, manifest_sha="d" * 64, materialization_sha="e" * 64,
        freeze_shas=freeze, claim_branch="B-weak", u_transfer=True, created_at="2026-07-12",
    )
    assert len(sha) == 64
    loaded = load_final_packet(path)
    assert loaded["ledger_chain_tip"] == "c" * 64
    assert loaded["selection_manifest_sha"] == "d" * 64
    assert loaded["materialization_sha"] == "e" * 64
    assert loaded["freeze_shas"] == freeze
    assert loaded["claim_branch"] == "B-weak"
    assert loaded["u_transfer_qualifier"] is True
    assert "created_at" not in loaded              # 易变字段不入 packet sha、load 时剔除
    # 篡改任一承重字段 → packet_sha256 校验失败
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["claim_branch"] = "B-strong"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="packet_sha256"):
        load_final_packet(path)


def test_f9_final_packet_validation(tmp_path):
    ok = dict(chain_tip="c" * 64, manifest_sha="d" * 64, materialization_sha="e" * 64,
              freeze_shas={"x": "f" * 64}, claim_branch="B-null")
    assert "B-null" in CLAIM_BRANCHES
    with pytest.raises(ValueError, match="claim_branch"):
        write_final_packet(tmp_path / "p1.json", **{**ok, "claim_branch": "B-bogus"})
    with pytest.raises(ValueError, match="chain_tip"):
        write_final_packet(tmp_path / "p2.json", **{**ok, "chain_tip": "short"})
    with pytest.raises(ValueError, match="freeze_shas"):
        write_final_packet(tmp_path / "p3.json", **{**ok, "freeze_shas": {}})
    with pytest.raises(ValueError, match="freeze_shas"):
        write_final_packet(tmp_path / "p4.json", **{**ok, "freeze_shas": {"x": "nothex"}})
