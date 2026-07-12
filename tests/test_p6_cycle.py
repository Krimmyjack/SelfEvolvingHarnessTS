"""tests/test_p6_cycle.py — P6 cycle runner 机械单测（prereg §4 cycle 步骤 1-7）。

运行：D:\\Anaconda_envs\\envs\\project\\python.exe -m pytest SelfEvolvingHarnessTS/tests/test_p6_cycle.py -q
（cwd = C:\\Users\\辉\\Desktop\\Agent，加 --basetemp 指到 scratchpad）

红线：全合成 + 注入 stub trainer/fingerprint；gate 用真 split_manifest 状态机 + tmp
ledger；不联网、不读真实数据、不 import torch、无 LLM/git。文件 IO 只发生在 pytest tmp。

场景物理（经 judge 实测钉死）：
  - "spiky"：正弦 + 稀疏 ±6 尖峰 + 微噪。病态起始 selector（weighted_features 偏好
    modified_fraction）选中重改动程序 → S1 大幅触发；miner 学出轻改动 selector →
    D 内部门过 → V(spiky) 六门全过 promote；
  - "purenoise"：纯噪声 V —— 两臂 train 差 ≈ 0 < ε 且 LCB ≤ 0 → 门① 独立拦截 → reject；
  - 4 底层 series < min_series=5 ⇒ S3 结构性静默（cohort 不承重）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from SelfEvolvingHarnessTS.p6.c0_runner import JUDGE_CFG_FROZEN, P6Episode
from SelfEvolvingHarnessTS.p6.cycle_runner import (
    ABSTAIN_INTERNAL_GATE,
    ABSTAIN_MINER_EMPTY,
    ABSTAIN_NO_SIGNATURE,
    BOOTSTRAP_SEED_BASE,
    COUNTERFACTUAL_CONFIG_BUDGET,
    CountingLlmSupplier,
    CycleBudget,
    DISCOVERY_ROUND_BUDGET,
    INTERNAL_REEVAL_BUDGET,
    LLM_REQUEST_BUDGET,
    P6BudgetError,
    PROBE_BUDGET,
    bootstrap_seed_for,
    build_cohorts,
    evaluate_promotion_gates,
    paired_arm_run,
    precommit_sidecar_path,
    run_cycle_formal,
    run_cycle_unfrozen,
    sealed_v_dir,
    select_probe_variants,
)
from SelfEvolvingHarnessTS.p6.c0_runner import P6FrozenParamError
from SelfEvolvingHarnessTS.p6.loaders import BoundVEpisodes, UnboundEpisodes
from SelfEvolvingHarnessTS.p6 import fast_path as fast_path_mod
from SelfEvolvingHarnessTS.p6.edit_surfaces import RiskRulePatch, SelectorPatch
from SelfEvolvingHarnessTS.p6.fast_path import P6PairingError
from SelfEvolvingHarnessTS.p6.harness_state import (
    P6HarnessState,
    RiskRuleSpec,
    SamplerSpec,
    SelectorSpec,
    apply_edit,
    default_state,
)
from SelfEvolvingHarnessTS.p6.judge_closed_form import fit_domain
from SelfEvolvingHarnessTS.p6.materializer import P6TechnicalAbort
from SelfEvolvingHarnessTS.p6.split_manifest import (
    P6StateError,
    SequentialGate,
    build_manifest,
    ledger_path,
)

# ════════════════════════════ 共用合成 fixtures ════════════════════════════
BIG = ("nn5_daily", "fred_md", "tourism_monthly", "covid_deaths")
SINGLE = ("us_births", "saugeenday", "sunspot")
PRESETS = ("G_hi_full", "G_hi_miss", "G_lo_full", "G_lo_miss")
H, LEN = 48, 240
SHAS = dict(code_sha="c" * 16, materialization_sha="m" * 16, config_digest="d" * 16)
C0F_STD = {
    "epsilon": 0.05, "delta_safe": 0.30,
    "p0_cutpoints": {"snr": [1.0, 5.0, 20.0], "missing_rate": [0.01, 0.05, 0.1]},
}


def _row(config, item):
    return {"config": config, "item_id": item, "series_uid": f"{config}:{item}",
            "exposure_class": "confirmed_exposed"}


def make_ledger():
    rows = [_row(c, f"T{i}") for c in BIG for i in range(1, 21)]
    rows += [_row(c, "T1") for c in SINGLE]
    return rows


@pytest.fixture(scope="module")
def manifest():
    return build_manifest(make_ledger(), [f"T{i}" for i in range(1, 25)])


def mk_episodes(doms, mode, seed0, n_series=2):
    """合成 episodes：底层序列 × 4 preset。mode ∈ {spiky, purenoise}。"""
    eps = []
    for di, d in enumerate(doms):
        for si in range(n_series):
            t = np.arange(LEN, dtype=float)
            clean = (np.sin(2 * np.pi * t / 24.0 + 0.9 * (2 * di + si))
                     + 0.25 * np.sin(2 * np.pi * t / 12.0))
            for k, p in enumerate(PRESETS):
                rng = np.random.default_rng(seed0 + 1000 * (2 * di + si) + k)
                if mode == "spiky":
                    h = clean[:LEN - H] + 0.1 * rng.standard_normal(LEN - H)
                    idx = rng.choice(LEN - H, size=int(0.15 * (LEN - H)), replace=False)
                    h[idx] += 6.0 * rng.choice([-1.0, 1.0], size=idx.size)
                else:  # purenoise
                    h = 1.0 * rng.standard_normal(LEN - H)
                eps.append(P6Episode(uid=f"{d}:s{si}:{p}", series_uid=f"{d}:s{si}",
                                     config=d, preset=p, history=h,
                                     future=clean[LEN - H:].copy()))
    return eps


def adam_mimic(views, seed):
    """stub Adam co-gate trainer：闭式判官自身 losses（joint 口径与判官一致 → 门③过）。"""
    return fit_domain(list(views), **JUDGE_CFG_FROZEN).per_series_rmse


def patho_state():
    """病态起始 H_t：weighted selector 偏好重改动（S1 触发的构造源）。"""
    return P6HarnessState(
        version="v0",
        selector=SelectorSpec("weighted_features", {"modified_fraction": 1.0, "n_steps": 0.1}),
        sampler=SamplerSpec(allocation={"det": 3, "random": 5, "llm": 0}, expected_total=8),
    )


class _VLoader:
    """裸 episodes 的 V loader（测试用）：G1 要求显式 UnboundEpisodes 包装（无裸序列路径）。"""
    def __init__(self, episodes):
        self.episodes = episodes
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return UnboundEpisodes(self.episodes)


def _run(manifest, base, cycle, eps_d, vloader, state, c0f, **kw):
    gate = SequentialGate(manifest, base)
    try:
        res = run_cycle_unfrozen(cycle, eps_d, vloader, state, gate, c0f,
                                 adam_trainer=adam_mimic, fingerprints=None, judge_cfg={},
                                 out_dir=base / "out", bootstrap_b=100, **SHAS, **kw)
    finally:
        gate.close()
    return res


# ══════════════════════════════════════════════════════════════════════════
# A. 冻结常量 / 预算强制（释义 K）
# ══════════════════════════════════════════════════════════════════════════
def test_bootstrap_seed_frozen():
    assert bootstrap_seed_for(1) == BOOTSTRAP_SEED_BASE + 1 == 20260712
    assert bootstrap_seed_for(2) == 20260713
    with pytest.raises(ValueError):
        bootstrap_seed_for(3)
    with pytest.raises(ValueError):
        bootstrap_seed_for(True)


def test_budget_probe_17_raises():
    with pytest.raises(P6BudgetError, match="probe"):
        CycleBudget().charge_probes(PROBE_BUDGET + 1)          # 17 个 → raise
    b = CycleBudget()
    b.charge_probes(PROBE_BUDGET)                              # 恰 16 合法
    with pytest.raises(P6BudgetError):
        b.charge_probes(1)                                     # 第 17 个 → raise


def test_budget_internal_reeval_13_raises():
    b = CycleBudget()
    for _ in range(INTERNAL_REEVAL_BUDGET):
        b.charge_internal_reeval()                             # 12 次合法
    with pytest.raises(P6BudgetError, match="内部重评"):
        b.charge_internal_reeval()                             # 第 13 次 → raise


def test_budget_llm_wrapper_61_raises():
    seen = []

    def supplier(uid, state, n):
        seen.append((uid, int(n)))
        return [[("impute_linear", {})]]

    budget = CycleBudget()
    wrapped = CountingLlmSupplier(supplier, budget)
    st = default_state()
    for i in range(LLM_REQUEST_BUDGET):
        out = wrapped(f"u{i}", st, 2)                          # 60 次合法（1 调用=1 request）
        assert out == [[("impute_linear", {})]]                # 透传 supplier 结果
    assert budget.llm_requests == LLM_REQUEST_BUDGET and len(seen) == LLM_REQUEST_BUDGET
    with pytest.raises(P6BudgetError, match="LLM"):
        wrapped("u60", st, 2)                                  # 第 61 次 → raise
    assert len(seen) == LLM_REQUEST_BUDGET                     # 超限调用未触达 supplier
    with pytest.raises(ValueError):
        CountingLlmSupplier(None, budget)                      # None 不包装


def test_budget_discovery_and_counterfactual():
    b = CycleBudget()
    for _ in range(DISCOVERY_ROUND_BUDGET):
        b.charge_discovery_round()
    with pytest.raises(P6BudgetError, match="discovery"):
        b.charge_discovery_round()
    b2 = CycleBudget()
    for _ in range(COUNTERFACTUAL_CONFIG_BUDGET):
        b2.charge_counterfactual()
    with pytest.raises(P6BudgetError, match="counterfactual"):
        b2.charge_counterfactual()


# ══════════════════════════════════════════════════════════════════════════
# B. probe 选择（释义 A）
# ══════════════════════════════════════════════════════════════════════════
def test_select_probe_variants_17_classes_capped_to_16():
    # 17 个两两分离（差 ≫ tol）的 loss → 17 类 → 取 loss 升序前 16
    losses = {f"sha{i:02d}": 1.0 + 0.1 * i for i in range(17)}
    picked = select_probe_variants(losses)
    assert len(picked) == PROBE_BUDGET == 16
    assert picked == [f"sha{i:02d}" for i in range(16)]        # 类内最优 loss 升序
    assert "sha16" not in picked                               # 最差类被裁


def test_select_probe_variants_class_rep_min_sha_and_tol():
    # b/a 同类（差 0 ≤ tol）→ 代表 = sha 最小者 "a"
    assert select_probe_variants({"b": 1.0, "a": 1.0, "z": 2.0}) == ["a", "z"]
    # 差恰 = tol（0 与 1e-9 的差二进制精确 = 1e-9）→ 仍并入（≤ 语义）；rep = sha 最小者
    assert select_probe_variants({"b": 0.0, "a": 1e-9}) == ["a"]
    # 差 2e-9 > tol → 分类
    assert select_probe_variants({"a": 0.0, "b": 2e-9}) == ["a", "b"]
    assert select_probe_variants({}) == []


# ══════════════════════════════════════════════════════════════════════════
# C. cohort 清单（§3.3 + 释义 D）
# ══════════════════════════════════════════════════════════════════════════
def test_build_cohorts_structure_and_membership():
    eps = mk_episodes(("nn5_daily",), "spiky", 0, n_series=1)   # 4 episode（4 preset）
    fps = [{"snr": 0.5, "missing_rate": 0.0},                   # snr bin0 / miss bin0
           {"snr": 5.0, "missing_rate": 0.02},                  # snr == cut[1] → bin2（左闭右开）
           {"snr": 100.0, "missing_rate": 0.5},                 # bin3 / bin3
           {"missing_rate": 0.06}]                              # 缺 snr → 不入 snr bin
    cohorts = build_cohorts(eps, fps, C0F_STD["p0_cutpoints"])
    by_id = {c["cohort_id"]: c for c in cohorts}
    assert len(by_id) == 4 + 8                                  # 4 preset + 2 特征 × 4 bin
    # preset cohort：成员 = 该 preset 的 episode；miner 形 = 纯 preset 名（成员资格 scope，F7）
    ph = by_id["preset:G_hi_full"]
    assert ph["member_idx"] == [0]
    assert ph["miner_cohort"] == {"cohort_id": "preset:G_hi_full", "preset": "G_hi_full"}
    assert "scope_conditions" not in ph["miner_cohort"]         # 旧半平面近似已删除
    # bin cohort：bisect_right 左闭右开；端 bin lo/hi = None
    assert by_id["bin:snr:0"]["member_idx"] == [0]
    assert by_id["bin:snr:2"]["member_idx"] == [1]              # 5.0 == cut → 上位 bin
    assert by_id["bin:snr:3"]["member_idx"] == [2]
    assert by_id["bin:snr:0"]["miner_cohort"]["bin"] == {"feature": "snr", "lo": None, "hi": 1.0}
    assert by_id["bin:snr:3"]["miner_cohort"]["bin"] == {"feature": "snr", "lo": 20.0, "hi": None}
    assert by_id["bin:missing_rate:1"]["member_idx"] == [1]
    assert by_id["bin:missing_rate:2"]["member_idx"] == [3]     # 0.06 ∈ [0.05, 0.1)


# ══════════════════════════════════════════════════════════════════════════
# D. 晋升门 ①②③⑥ 边界（stub 数值钉死；常数 gain ⇒ bootstrap LCB 精确等于该常数）
# ══════════════════════════════════════════════════════════════════════════
_CL = ["s0", "s0", "s1", "s1", "s2", "s2", "s3", "s3"]
_PR = ["p", "q"] * 4


def _gates(tg, jg, *, eps=0.0625, delta=0.03125, adam=(0.0, 0.0, 0.0),
           risk_ok=True, ledger_ok=True):
    return evaluate_promotion_gates(tg, jg, _CL, _PR, eps=eps, delta_safe=delta,
                                    adam_gains=list(adam), risk_scope_ok=risk_ok,
                                    ledger_ok=ledger_ok, b=200, seed=20260712)


def test_gate1_boundaries():
    out = _gates([0.0625] * 8, [0.0625] * 8)                    # 点恰 = ε（≥ 含边界）
    assert out["gate1_train"]["pass"] is True and out["promote"] is True
    out2 = _gates([0.0625 - 2 ** -10] * 8, [0.0625] * 8)        # 点 < ε → 拒
    assert out2["gate1_train"]["pass"] is False and out2["promote"] is False
    out3 = _gates([0.0] * 8, [0.0] * 8, eps=0.0)                # 点 0 ≥ 0 过；LCB 0 > 0 严格失败
    assert out3["gate1_train"]["lcb90"] == 0.0
    assert out3["gate1_train"]["pass"] is False


def test_gate2_preset_worst_boundary():
    # preset p 恒 −δ（LCB 精确 = −δ）→ ② 过（≥ 含边界）；再降 2⁻¹⁰ → ② 独立失败
    tg_edge = [-0.03125 if p == "p" else 0.5 for p in _PR]
    out = _gates(tg_edge, [0.5] * 8)
    assert out["gate2_preset_train"]["worst_lcb90"] == -0.03125
    assert out["gate2_preset_train"]["pass"] is True
    tg_bad = [-0.03125 - 2 ** -10 if p == "p" else 0.5 for p in _PR]
    out2 = _gates(tg_bad, [0.5] * 8)
    assert out2["gate2_preset_train"]["pass"] is False
    assert out2["gate6_joint_safety"]["pass"] is True           # 只有 ② 拦（joint 全正）
    assert out2["promote"] is False


def test_gate3_adam_cogate_boundary():
    out = _gates([0.5] * 8, [0.5] * 8, adam=(-0.0625, -0.0625, -0.0625))  # 均值恰 = −ε → 过
    assert out["gate3_adam_cogate"]["mean_gain"] == -0.0625
    assert out["gate3_adam_cogate"]["pass"] is True and out["promote"] is True
    out2 = _gates([0.5] * 8, [0.5] * 8, adam=(-0.0625, -0.0625, -0.0625 - 3 * 2 ** -10))
    assert out2["gate3_adam_cogate"]["pass"] is False           # 均值 < −ε → ③ 独立拦截
    assert out2["gate1_train"]["pass"] is True and out2["promote"] is False


def test_gate6_joint_safety_boundary():
    out = _gates([0.5] * 8, [-0.03125] * 8)                     # joint 恒 −δ → 过（≥ 含边界）
    assert out["gate6_joint_safety"]["overall_lcb90"] == -0.03125
    assert out["gate6_joint_safety"]["pass"] is True and out["promote"] is True
    out2 = _gates([0.5] * 8, [-0.03125 - 2 ** -10] * 8)         # overall 越界 → ⑥ 独立拦截
    assert out2["gate6_joint_safety"]["pass"] is False and out2["promote"] is False
    # preset worst-group 子句：overall 达标但 preset p 的 joint 越界 → ⑥ 仍拦
    jg = [-0.03125 - 2 ** -10 if p == "p" else 0.5 for p in _PR]
    out3 = _gates([0.5] * 8, jg)
    assert out3["gate6_joint_safety"]["overall_lcb90"] > -0.03125
    assert out3["gate6_joint_safety"]["pass"] is False


def test_gate4_gate5_passthrough():
    out = _gates([0.5] * 8, [0.5] * 8, risk_ok=False)
    assert out["gate4_scope_bytes"]["pass"] is False and out["promote"] is False
    out2 = _gates([0.5] * 8, [0.5] * 8, ledger_ok=False)
    assert out2["gate5_ledger"]["pass"] is False and out2["promote"] is False


# ══════════════════════════════════════════════════════════════════════════
# E. 门④ risk 族字节校验（paired_arm_run 分派；真 paired_risk_run 路径）
# ══════════════════════════════════════════════════════════════════════════
def _risk_states():
    base = default_state()
    rule = RiskRuleSpec(rule_id="r_never",
                        when=[{"feature": "snr", "op": ">=", "value": 1e9}],
                        then={"action": "ban", "target": "denoise_median"})
    return base, apply_edit(base, RiskRulePatch(add_rule=rule))


def test_paired_arm_run_gate4_non_scope_auto_pass():
    eps = mk_episodes(("nn5_daily",), "spiky", 0, n_series=1)
    a = default_state()
    b = apply_edit(a, SelectorPatch(new_selector=SelectorSpec(
        "weighted_features", {"proxy_score": 1.0})))
    out = paired_arm_run(eps, a, b, "selector_patch", 8, None, None)
    assert out["gate4_ok"] is True and out["out_of_scope_verified"] is None
    assert set(out["chosen_a"]) == {e.uid for e in eps}


def test_paired_arm_run_gate4_risk_bytes_verified():
    eps = mk_episodes(("nn5_daily",), "spiky", 0, n_series=1)
    a, b = _risk_states()
    out = paired_arm_run(eps, a, b, "risk_rule_patch", 8, None, None)
    assert out["gate4_ok"] is True
    assert sorted(out["out_of_scope_verified"]) == sorted(e.uid for e in eps)  # 全 uid 过字节校验


def test_paired_arm_run_gate4_byte_violation(monkeypatch):
    eps = mk_episodes(("nn5_daily",), "spiky", 0, n_series=1)
    a, b = _risk_states()
    counter = [0]

    def tampered(chosen, series):                              # 每次调用产出不同字节
        counter[0] += 1
        return np.arange(16, dtype=float) + counter[0]

    monkeypatch.setattr(fast_path_mod, "prepared_artifact", tampered)
    out = paired_arm_run(eps, a, b, "risk_rule_patch", 8, None, None,
                         on_scope_violation="gate")             # V 语境 → 门④ FAIL（不 raise）
    assert out["gate4_ok"] is False and "门④" in out["gate4_note"]
    with pytest.raises(P6PairingError):                         # D 内部重评语境 → 原样 raise
        paired_arm_run(eps, a, b, "risk_rule_patch", 8, None, None,
                       on_scope_violation="raise")


def test_paired_arm_run_unknown_kind():
    eps = mk_episodes(("nn5_daily",), "spiky", 0, n_series=1)
    with pytest.raises(ValueError, match="edit_kind"):
        paired_arm_run(eps, default_state(), default_state(), "grammar_macro", 8, None, None)


# ══════════════════════════════════════════════════════════════════════════
# F. abstain 三条路（各自记 terminal 且 V 未开、loader 未被调用）
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def abstain_ns(manifest, tmp_path_factory):
    """no_signature abstain：默认 H0 + ε/δ 巨大 → S1/S2/S3 全静默。"""
    base = tmp_path_factory.mktemp("abstain_ns")
    eps_d = mk_episodes(("nn5_daily",), "spiky", 0)
    c0f = {"epsilon": 1e6, "delta_safe": 2.5e6, "p0_cutpoints": C0F_STD["p0_cutpoints"]}
    vloader = _VLoader(mk_episodes(("fred_md",), "spiky", 900))
    res = _run(manifest, base, 1, eps_d, vloader, default_state(), c0f)
    return res, base, vloader, eps_d


def test_abstain_no_signature(abstain_ns, manifest):
    res, base, vloader, _eps = abstain_ns
    assert res.terminal == "abstain" and res.abstain_reason == ABSTAIN_NO_SIGNATURE
    assert res.signature["activated"] is None
    assert not res.signature["s1"]["fired"] and not res.signature["s2"]["fired"]
    assert not res.signature["s3"]["fired"]
    assert res.state_changed is False and res.precommit is None
    assert not hasattr(res, "sealed_dir")                       # F1：CycleResult 不再携带 sealed 路径
    assert vloader.calls == 0                                   # V 数据延迟加载：从未触碰
    with SequentialGate(manifest, base) as g:                   # 重放台账验证状态推进
        assert g.cycle_terminal(1) == "abstain"
        assert g.state("V1") == "sealed" and g.precommit(1) is None
    assert len(res.digest()) == 64


def test_abstain_rerun_rejected_by_state_machine(abstain_ns, manifest):
    res, base, vloader, eps_d = abstain_ns
    with SequentialGate(manifest, base) as g:
        with pytest.raises(P6StateError, match="terminal"):     # cycle1 已 terminal → 拒绝重跑
            run_cycle_unfrozen(1, eps_d, vloader, default_state(), g,
                      {"epsilon": 1e6, "delta_safe": 2.5e6,
                       "p0_cutpoints": C0F_STD["p0_cutpoints"]},
                      adam_trainer=adam_mimic, fingerprints=None, judge_cfg={},
                      out_dir=base / "out2", **SHAS)


def test_uid_consumption_manifest_content(abstain_ns):
    res, base, _vl, eps_d = abstain_ns
    path = base / "out" / "consumed_uids_cycle1_D.json"
    assert path.exists()
    assert res.discovery["consumed_manifest_path"] == str(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema"] == "p6-uid-consumption/1" and doc["block"] == "D1"
    assert doc["cycle"] == 1 and doc["K"] == 8
    assert doc["episode_uids"] == sorted(e.uid for e in eps_d)
    assert doc["series_uids"] == sorted({e.series_uid for e in eps_d})
    assert doc["state_sha"] == default_state().sha()
    assert doc["code_sha"] == SHAS["code_sha"]
    assert doc["materialization_sha"] == SHAS["materialization_sha"]
    assert doc["config_digest"] == SHAS["config_digest"]
    assert len(doc["manifest_sha"]) == 64


def test_abstain_miner_empty(manifest, tmp_path):
    # {det:0, random:1, llm:7} + llm=None：池=1 → S2 触发；三个 sampler 配方全负配额 → []
    st = P6HarnessState(version="v0", selector=SelectorSpec(),
                        sampler=SamplerSpec(allocation={"det": 0, "random": 1, "llm": 7},
                                            expected_total=8))
    eps_d = mk_episodes(("nn5_daily",), "spiky", 0)
    c0f = {"epsilon": 1e6, "delta_safe": 2.5e6, "p0_cutpoints": C0F_STD["p0_cutpoints"]}
    vloader = _VLoader(mk_episodes(("fred_md",), "spiky", 900))
    res = _run(manifest, tmp_path, 1, eps_d, vloader, st, c0f)
    assert res.terminal == "abstain" and res.abstain_reason == ABSTAIN_MINER_EMPTY
    assert res.signature["activated"] == "S2"
    assert res.signature["s2"]["mean_classes"] == 1.0           # 单候选池 → 类数恒 1
    assert res.internal["n_candidates"] == 0
    assert vloader.calls == 0 and not hasattr(res, "sealed_dir")
    with SequentialGate(manifest, tmp_path) as g:
        assert g.cycle_terminal(1) == "abstain" and g.state("V1") == "sealed"


def test_abstain_internal_gate(manifest, tmp_path):
    # {det:1, random:0, llm:7}：池=1 → S2 触发；sampler_a 唯一可用；ε 巨大 → 内部门不过
    st = P6HarnessState(version="v0", selector=SelectorSpec(),
                        sampler=SamplerSpec(allocation={"det": 1, "random": 0, "llm": 7},
                                            expected_total=8))
    eps_d = mk_episodes(("nn5_daily",), "spiky", 0)
    c0f = {"epsilon": 10.0, "delta_safe": 25.0, "p0_cutpoints": C0F_STD["p0_cutpoints"]}
    vloader = _VLoader(mk_episodes(("fred_md",), "spiky", 900))
    res = _run(manifest, tmp_path, 1, eps_d, vloader, st, c0f)
    assert res.terminal == "abstain" and res.abstain_reason == ABSTAIN_INTERNAL_GATE
    assert res.signature["activated"] == "S2"
    assert res.internal["n_candidates"] == 1                    # 只有 sampler_a 可用
    assert res.internal["candidates"][0]["recipe_id"] == "sampler_a"
    assert res.internal["winner"]["train_gain_d"] < 10.0
    assert res.internal["internal_gate_pass"] is False
    assert res.cost["internal_reevals"] == 1
    assert vloader.calls == 0 and res.precommit is None
    with SequentialGate(manifest, tmp_path) as g:
        assert g.cycle_terminal(1) == "abstain" and g.state("V1") == "sealed"
        assert g.precommit(1) is None                           # 内部门不过 → 不 precommit


# ══════════════════════════════════════════════════════════════════════════
# G. promote 全链 + 确定性（两次全新 run digest 恒等）
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def promote_runs(manifest, tmp_path_factory):
    def once(tag):
        base = tmp_path_factory.mktemp(f"promote_{tag}")
        eps_d = mk_episodes(("nn5_daily", "fred_md"), "spiky", 0)
        vloader = _VLoader(mk_episodes(("tourism_monthly", "covid_deaths"), "spiky", 700))
        res = _run(manifest, base, 1, eps_d, vloader, patho_state(), C0F_STD)
        return res, base, vloader
    return once("a"), once("b")


def test_promote_full_chain(promote_runs, manifest):
    (res, base, vloader), _second = promote_runs
    # S1 构造触发（SelectorPatch 明显赢）→ selector 族 → 内部门过 → 六门全过 promote
    assert res.signature["activated"] == "S1"
    assert res.signature["s1"]["fired"] and res.signature["s1"]["lcb90"] > 0
    assert not res.signature["s3"]["fired"]                     # 4 series < min_series=5 → 静默
    assert res.internal["family"] == "S1"
    assert 1 <= res.internal["n_candidates"] <= 3
    assert all(e["edit_kind"] == "selector_patch" for e in res.internal["candidates"])
    assert res.internal["internal_gate_pass"] is True
    assert res.internal["winner"]["train_gain_d"] >= C0F_STD["epsilon"]

    assert res.terminal == "promote" and res.abstain_reason is None
    assert res.state_changed is True
    assert res.new_state.sha() != patho_state().sha()           # 新 state 返回
    assert res.new_state.version == "v0.e1" and len(res.new_state.edit_log) == 1
    assert res.new_state.selector.kind == "weighted_features"
    assert res.precommit["candidate_edit_sha"] == res.internal["winner"]["candidate_sha"]
    assert res.precommit["harness_state_sha"] == patho_state().sha()
    assert vloader.calls == 1                                   # V 只在步骤 7 加载一次

    # gate 状态推进（重放台账）
    with SequentialGate(manifest, base) as g:
        assert g.cycle_terminal(1) == "promote"
        assert g.state("V1") == "verdict_recorded" and g.verdict("V1") == "promote"
        pc = g.precommit(1)
        assert pc["payload"]["candidate_edit_sha"] == res.precommit["candidate_edit_sha"]

    # sealed 目录存在且携带 V 详情 + README 禁读声明（路径由 sealed_v_dir 确定性重算，
    # 不再经 CycleResult 泄漏——F1）
    sealed = sealed_v_dir(base / "out", 1)
    assert sealed == base / "out" / "sealed_V1"
    assert (sealed / "README.md").exists()
    assert "H_final" in (sealed / "README.md").read_text(encoding="utf-8")
    rep = json.loads((sealed / "v_report.json").read_text(encoding="utf-8"))
    assert rep["verdict"] == "promote"
    assert all(rep["gates"][k]["pass"] for k in
               ("gate1_train", "gate2_preset_train", "gate3_adam_cogate",
                "gate4_scope_bytes", "gate5_ledger", "gate6_joint_safety"))
    assert rep["gates"]["gate1_train"]["train_gain"] >= C0F_STD["epsilon"]
    # V 消费 manifest 也已落盘
    vdoc = json.loads((base / "out" / "consumed_uids_cycle1_V.json").read_text(encoding="utf-8"))
    assert vdoc["block"] == "V1"
    assert vdoc["episode_uids"] == sorted(e.uid for e in vloader.episodes)
    # 归因落盘：rows = probe × episode
    att = json.loads((base / "out" / "attribution_cycle1_D.json").read_text(encoding="utf-8"))
    assert len(att["rows"]) == len(att["probe_shas"]) * res.discovery["n_episodes"]

    # discovery 摘要与预算账
    assert res.discovery["counterfactual_chosen_sets"].keys() == {"det_only", "det_random", "incumbent"}
    assert res.cost["probe_variants"] <= 16
    assert res.cost["internal_reevals"] == res.internal["n_candidates"] <= 3
    assert res.cost["llm_requests"] == 0 and res.cost["discovery_rounds"] == 1
    assert res.cost["adam_cogate_fits"] == 3 * 2 * 2            # 3 seeds × 2 臂 × 2 域


def test_promote_result_carries_no_v_numbers(promote_runs):
    (res, _base, _vl), _second = promote_runs
    rep = json.loads((sealed_v_dir(_base / "out", 1) / "v_report.json").read_text(encoding="utf-8"))
    # CycleResult 的 digest payload（其全部可携带内容）不含任何 V 数字
    payload = json.dumps({
        "signature": res.signature, "internal": res.internal,
        "discovery": res.discovery, "precommit": res.precommit, "cost": res.cost,
    }, ensure_ascii=False)
    for v_number in (rep["arms"]["h"]["utility"], rep["arms"]["edit"]["utility"],
                     rep["effects"]["train"]["overall_gain"],
                     rep["gates"]["gate1_train"]["lcb90"]):
        assert repr(float(v_number)) not in payload
    # 字段面固化：CycleResult 没有任何 V 报告承载字段、也无 sealed 路径字段（F1）；
    # entrypoint/frozen_literals_digest 是 G4 正式入口证据（非 V 数字）。
    assert set(res.__dataclass_fields__) == {
        "cycle", "terminal", "abstain_reason", "new_state", "state_changed",
        "signature", "internal", "discovery", "precommit", "cost",
        "entrypoint", "frozen_literals_digest",
    }


def test_f1_cycleresult_no_sealed_path_or_v_detail(promote_runs):
    """F1/finding 31：CycleResult 序列化（asdict/digest）无 sealed 路径、无 per-episode V 明细；
    V 详情只落 sealed_V{t}/v_report.json；precommit → open(V) → loader 次序（loader 一次）。"""
    import dataclasses
    (res, base, vloader), _second = promote_runs
    assert not hasattr(res, "sealed_dir")
    d = dataclasses.asdict(res)
    assert "sealed_dir" not in d
    blob = json.dumps(d, default=str, ensure_ascii=False)
    # sealed 目录路径字节、v_report 名不得出现在结果载体
    assert "sealed_V" not in blob and "v_report" not in blob
    # per-episode V 明细键（仅 v_report.json 独有）不得出现在 CycleResult
    # 注：cost 里的 adam_cogate_fits 是拟合计数（非 V 效用数），不算明细泄漏。
    for k in ("loss_00", "loss_10", "loss_01", "loss_11", "per_episode",
              "loss_edit", "joint_gain"):
        assert k not in blob, f"CycleResult 泄漏 V 明细键 {k!r}"
    # digest payload（discovery 已滤 *_path）同样无 sealed 字节
    assert "sealed_V" not in json.dumps(res.discovery, ensure_ascii=False)
    # V 详情确在 sealed 目录（写入未受影响）；loader 只被调用一次（open 之后）
    assert (sealed_v_dir(base / "out", 1) / "v_report.json").exists()
    assert vloader.calls == 1
    assert len(res.digest()) == 64


class _CrashLoader:
    """precommit + open V 之后、verdict 之前崩溃的 V loader（F4 恢复路径）。"""
    def __init__(self, episodes):
        self.episodes = episodes
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise RuntimeError("simulated crash after open, before verdict")


def test_f4_crash_resume_completes(manifest, tmp_path):
    """F4/finding 34：precommit + open V 后崩溃 → 同参数重启 → precommit 幂等 + 跳过重复 open
    → verdict 落账 → hash 链连续无分叉（无重复 precommit/open 事件）。"""
    eps_d = mk_episodes(("nn5_daily", "fred_md"), "spiky", 0)
    v_eps = mk_episodes(("tourism_monthly", "covid_deaths"), "spiky", 700)
    kw = dict(adam_trainer=adam_mimic, fingerprints=None, judge_cfg={},
              out_dir=tmp_path / "out", bootstrap_b=100, **SHAS)

    crash = _CrashLoader(v_eps)
    g1 = SequentialGate(manifest, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            run_cycle_unfrozen(1, eps_d, crash, patho_state(), g1, C0F_STD, **kw)
    finally:
        g1.close()
    assert crash.calls == 1
    with SequentialGate(manifest, tmp_path) as gi:                # precommit+open durable、无 verdict
        assert gi.precommit(1) is not None
        assert gi.state("V1") == "open" and gi.cycle_terminal(1) is None

    g2 = SequentialGate(manifest, tmp_path)
    try:
        res = run_cycle_unfrozen(1, eps_d, _VLoader(v_eps), patho_state(), g2, C0F_STD, **kw)
    finally:
        g2.close()
    assert res.terminal in ("promote", "reject")
    with SequentialGate(manifest, tmp_path) as g3:                # 恢复后台账完整、可再 replay
        assert g3.cycle_terminal(1) == res.terminal and g3.verdict("V1") == res.terminal
    evs = [json.loads(line) for line in
           ledger_path(manifest, tmp_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [e["event"] for e in evs] == ["precommit", "open", "verdict", "cycle_terminal"]


class _CarrierVLoader:
    """BoundVEpisodes V loader（F2/G1 绑定核验路径）。"""
    def __init__(self, episodes, materialization_sha):
        self.episodes = episodes
        self.materialization_sha = materialization_sha
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return BoundVEpisodes(self.episodes, self.materialization_sha)


def test_f2_run_cycle_v_materialization_binding_mismatch(manifest, tmp_path):
    """F2/finding 32：loader 暴露的 materialization_sha ≠ precommit 绑定值 → P6TechnicalAbort。
    open 已在 loader 之前发生（F1 次序），loader 恰被调用一次。"""
    eps_d = mk_episodes(("nn5_daily", "fred_md"), "spiky", 0)
    v_eps = mk_episodes(("tourism_monthly", "covid_deaths"), "spiky", 700)
    loader = _CarrierVLoader(v_eps, materialization_sha="WRONG" + "0" * 11)   # ≠ SHAS["m..."]
    with pytest.raises(P6TechnicalAbort, match="materialization_sha"):
        _run(manifest, tmp_path, 1, eps_d, loader, patho_state(), C0F_STD)
    assert loader.calls == 1


def test_promote_determinism_two_fresh_runs(promote_runs):
    (r1, _b1, _v1), (r2, _b2, _v2) = promote_runs
    assert r1.terminal == r2.terminal == "promote"
    assert r1.internal["winner"]["candidate_sha"] == r2.internal["winner"]["candidate_sha"]
    assert r1.new_state.sha() == r2.new_state.sha()
    assert r1.digest() == r2.digest()                           # 同输入两次 → 摘要 sha 恒等


# ══════════════════════════════════════════════════════════════════════════
# H. reject 路（V 上构造不过门①）
# ══════════════════════════════════════════════════════════════════════════
def test_reject_gate1_on_v(manifest, tmp_path):
    eps_d = mk_episodes(("nn5_daily", "fred_md"), "spiky", 0)
    vloader = _VLoader(mk_episodes(("tourism_monthly", "covid_deaths"), "purenoise", 700))
    st = patho_state()
    res = _run(manifest, tmp_path, 1, eps_d, vloader, st, C0F_STD)
    assert res.terminal == "reject" and res.abstain_reason is None
    assert res.state_changed is False
    assert res.new_state.sha() == st.sha()                      # state 不变
    assert res.new_state.version == "v0"
    assert vloader.calls == 1
    rep = json.loads((sealed_v_dir(tmp_path / "out", 1) / "v_report.json").read_text(encoding="utf-8"))
    assert rep["verdict"] == "reject"
    assert rep["gates"]["gate1_train"]["pass"] is False         # 门① 独立拦截
    assert rep["gates"]["gate1_train"]["train_gain"] < C0F_STD["epsilon"]
    assert rep["gates"]["promote"] is False
    with SequentialGate(manifest, tmp_path) as g:
        assert g.verdict("V1") == "reject" and g.cycle_terminal(1) == "reject"


# ══════════════════════════════════════════════════════════════════════════
# I. cycle=2 需 cycle1 terminal（状态机拒绝的传播）
# ══════════════════════════════════════════════════════════════════════════
def test_cycle2_requires_cycle1_terminal(manifest, tmp_path):
    eps_d = mk_episodes(("nn5_daily",), "spiky", 0)
    vloader = _VLoader(mk_episodes(("fred_md",), "spiky", 900))
    c0f = {"epsilon": 1e6, "delta_safe": 2.5e6, "p0_cutpoints": C0F_STD["p0_cutpoints"]}
    with SequentialGate(manifest, tmp_path) as g:
        with pytest.raises(P6StateError, match="cycle1 terminal"):
            run_cycle_unfrozen(2, eps_d, vloader, default_state(), g, c0f,
                               adam_trainer=adam_mimic, fingerprints=None, judge_cfg={},
                               out_dir=tmp_path / "out", **SHAS)
        assert g.seq == 0                                       # 拒绝发生在任何台账写入之前
        assert vloader.calls == 0
        # cycle1 terminal 落账后 cycle2 可走（abstain 路）
        g.record_cycle_terminal(1, "abstain")
        res = run_cycle_unfrozen(2, eps_d, vloader, default_state(), g, c0f,
                                 adam_trainer=adam_mimic, fingerprints=None, judge_cfg={},
                                 out_dir=tmp_path / "out", bootstrap_b=100, **SHAS)
        assert res.terminal == "abstain" and res.cycle == 2
        assert g.cycle_terminal(2) == "abstain"
        assert (tmp_path / "out" / "consumed_uids_cycle2_D.json").exists()


# ══════════════════════════════════════════════════════════════════════════
# G 波：正式入口收口（codex 三轮复审 finding 32/34/36 最小再送审条件）
# ══════════════════════════════════════════════════════════════════════════
def test_g1_v_loader_bare_seq_rejected(manifest, tmp_path):
    """G1/finding 32：V loader 返回裸序列（非 BoundV/UnboundEpisodes）→ P6TechnicalAbort。
    杜绝"传普通 list 恰好也能跑"的静默路径。"""
    eps_d = mk_episodes(("nn5_daily", "fred_md"), "spiky", 0)
    v_eps = mk_episodes(("tourism_monthly", "covid_deaths"), "spiky", 700)
    with pytest.raises(P6TechnicalAbort, match="BoundVEpisodes 或 UnboundEpisodes"):
        _run(manifest, tmp_path, 1, eps_d, lambda: list(v_eps), patho_state(), C0F_STD)


def test_g1_unbound_rejected_by_formal(manifest, tmp_path):
    """G1/finding 32：formal 入口拒绝 UnboundEpisodes（必须 manifest-bound）→ P6TechnicalAbort。"""
    eps_d = mk_episodes(("nn5_daily", "fred_md"), "spiky", 0)
    v_eps = mk_episodes(("tourism_monthly", "covid_deaths"), "spiky", 700)
    d = tmp_path / "d"
    d.mkdir()
    with SequentialGate(manifest, d) as g:
        with pytest.raises(P6TechnicalAbort, match="UnboundEpisodes 被拒"):
            run_cycle_formal(1, eps_d, _VLoader(v_eps), patho_state(), g, C0F_STD,
                             adam_trainer=adam_mimic, fingerprints=None, judge_cfg={},
                             out_dir=d / "out", **SHAS)     # bootstrap_b 默认 2000（冻结）


def test_g2_resume_skips_discovery_no_re_mine(manifest, tmp_path, monkeypatch):
    """G2/finding 34：崩溃后 resume 从 sidecar 恢复，**不重跑 discovery/miner**（miner 计数为 0）。"""
    import SelfEvolvingHarnessTS.p6.cycle_runner as cyc_mod
    eps_d = mk_episodes(("nn5_daily", "fred_md"), "spiky", 0)
    v_eps = mk_episodes(("tourism_monthly", "covid_deaths"), "spiky", 700)
    kw = dict(adam_trainer=adam_mimic, fingerprints=None, judge_cfg={},
              out_dir=tmp_path / "out", bootstrap_b=100, **SHAS)
    # 第一次：precommit + open 后崩溃（写 sidecar）；miner 在此被真实调用
    g1 = SequentialGate(manifest, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            run_cycle_unfrozen(1, eps_d, _CrashLoader(v_eps), patho_state(), g1, C0F_STD, **kw)
    finally:
        g1.close()
    assert precommit_sidecar_path(tmp_path / "out", 1).exists()

    # 第二次：resume——监视 miner，断言恢复路径调用次数为 0（不经 discovery）
    mine_calls = {"n": 0}
    orig_mine = cyc_mod.mine

    def _counting_mine(*a, **k):
        mine_calls["n"] += 1
        return orig_mine(*a, **k)

    monkeypatch.setattr(cyc_mod, "mine", _counting_mine)
    g2 = SequentialGate(manifest, tmp_path)
    try:
        res = run_cycle_unfrozen(1, eps_d, _VLoader(v_eps), patho_state(), g2, C0F_STD, **kw)
    finally:
        g2.close()
    assert res.terminal in ("promote", "reject")
    assert res.cost["resumed"] is True                       # 走了恢复分支
    assert mine_calls["n"] == 0                               # miner 未被再调用（discovery 跳过）
    # discovery 摘要来自 sidecar（与冻结候选一致）
    assert res.internal["winner"]["candidate_sha"] == res.precommit["candidate_edit_sha"]


def test_g4_cycle_formal_drift_and_entrypoint(manifest, tmp_path):
    """G4/finding 36：run_cycle_formal 对漂移字面量 raise（台账未写）；冻结下记 entrypoint + digest。"""
    eps_d = mk_episodes(("nn5_daily",), "spiky", 0)
    v_eps = mk_episodes(("fred_md",), "spiky", 900)
    c0f = {"epsilon": 1e6, "delta_safe": 2.5e6, "p0_cutpoints": C0F_STD["p0_cutpoints"]}
    d1 = tmp_path / "d1"
    d1.mkdir()
    with SequentialGate(manifest, d1) as g:
        with pytest.raises(P6FrozenParamError, match="bootstrap_b"):   # 漂移 → 断言在台账写入前
            run_cycle_formal(1, eps_d, _VLoader(v_eps), default_state(), g, c0f,
                             adam_trainer=adam_mimic, fingerprints=None, judge_cfg={},
                             out_dir=d1 / "out", bootstrap_b=100, **SHAS)
        assert g.seq == 0
    # 冻结字面量下 abstain（huge ε）→ 记 entrypoint（loader 未被调用，未触发 bound 检查）
    d2 = tmp_path / "d2"
    d2.mkdir()
    with SequentialGate(manifest, d2) as g:
        res = run_cycle_formal(1, eps_d, _VLoader(v_eps), default_state(), g, c0f,
                               adam_trainer=adam_mimic, fingerprints=None, judge_cfg={},
                               out_dir=d2 / "out", **SHAS)    # bootstrap_b 默认 2000
    assert res.terminal == "abstain"
    assert res.entrypoint == "run_cycle_formal"
    assert len(res.frozen_literals_digest) == 64
