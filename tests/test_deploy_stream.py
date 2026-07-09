"""tests/test_deploy_stream.py — ★v4 S1 流式持续适应 + 三 bootstrap 的单测。

退出判据（S1_Implementation_Plan §D）：
  1. 三 mode 端到端跑通、JSONL 完整。
  2. frozen(B) 不改 H（version 不前进）；scratch(A) 每域从 minimal 起。
  3. lazy 重验触发降级（must_preserve→False），成本=frozen-eval 量级（无 LLM）。
  4. updating(C) 跨域携带（reset-free）。
另测 models/registry 三角色分离、schedule.enter_new_domain、readiness 度量。
"""
import json

import pytest

from SelfEvolvingHarnessTS.slow_path import deploy_stream as ds
from SelfEvolvingHarnessTS.slow_path.batch_builder import BatchBuilder
from SelfEvolvingHarnessTS.slow_path.evolve import Evolver
from SelfEvolvingHarnessTS.slow_path.schedule import CellSchedule, FROZEN, ACTIVE
from SelfEvolvingHarnessTS.harness.state import HarnessState
from SelfEvolvingHarnessTS.harness.edit_patch import EditPatch, Manifest
from SelfEvolvingHarnessTS.harness.layers import StrengthSignatureStats
from SelfEvolvingHarnessTS.data.synthetic_gen import make_forecast_batch
from SelfEvolvingHarnessTS import models
from SelfEvolvingHarnessTS.evaluators import readiness_score, aggregate_time_to_readiness


# ──────────────────────────── fixtures ────────────────────────────
class StubProposer:
    """确定性、免 LLM：对每 cell 提议 winsorize toggle（合法 leaf 编辑）。"""
    def propose(self, harness, weakness, strength=None, rejection_log=None):
        p = EditPatch(edited_layer="L2", op="set", path="l2.active_operators.winsorize", value=True,
                      manifest=Manifest("t", "toggle", "e", "a", "r"), source_type="failure")
        p.cell_id = weakness.cell_id
        return [p]


def _domains():
    return [ds.DomainSpec("P1", make_forecast_batch("P1", n=36, seed0=0), ("forecast",)),
            ds.DomainSpec("P3", make_forecast_batch("P3", n=36, seed0=100), ("forecast",))]


# ──────────────────────────── 1. 三 mode 跑通 + JSONL ────────────────────────────
def test_three_modes_run_and_log(tmp_path):
    doms = _domains()
    c = ds.deploy_stream(doms, mode="updating", make_harness=HarnessState.from_minimal,
                         make_proposer=StubProposer, n_epochs_per_domain=2)
    assert len(c.domains) == 2 and all(d.cell_logs for d in c.domains)

    b = ds.deploy_stream(doms, mode="frozen", make_harness=HarnessState.from_minimal,
                         bootstrap_checkpoints=c.checkpoints(), n_epochs_per_domain=2)
    a = ds.deploy_stream(doms, mode="scratch", make_harness=HarnessState.from_minimal,
                         make_proposer=StubProposer, n_epochs_per_domain=2,
                         log_path=str(tmp_path / "ft.jsonl"))
    assert len(b.domains) == 2 and len(a.domains) == 2

    lines = (tmp_path / "ft.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines
    rec = json.loads(lines[0])
    for key in ("k", "domain", "mode", "cell", "task", "readiness_at_budget",
                "time_to_readiness_rounds", "j_raw", "j_cur", "j_min_ref"):
        assert key in rec


# ──────────────────────────── 2. frozen 不改 H ────────────────────────────
def test_frozen_does_not_mutate_harness():
    doms = _domains()
    c = ds.deploy_stream(doms, mode="updating", make_harness=HarnessState.from_minimal,
                         make_proposer=StubProposer, n_epochs_per_domain=2)
    ckpts = c.checkpoints()
    ver_before = ckpts[0].version
    b = ds.deploy_stream(doms, mode="frozen", make_harness=HarnessState.from_minimal,
                         bootstrap_checkpoints=ckpts, n_epochs_per_domain=2)
    # B 第 k=1 域用 ckpts[0]，frozen 后 version 不前进，且未篡改 checkpoint 对象
    assert b.domains[1].harness_version == ver_before
    assert ckpts[0].version == ver_before
    # B 第 k=0 域无 prior → minimal（version 0）
    assert b.domains[0].harness_version == 0
    # B 不写 carried store
    assert all(d.n_reval_demote == 0 for d in b.domains)


# ──────────────────────────── 3. lazy 重验降级（deterministic via monkeypatch）────────────────────────────
def test_lazy_revalidation_demotes(monkeypatch):
    import SelfEvolvingHarnessTS.slow_path.mining as mining
    from SelfEvolvingHarnessTS.slow_path.mining import StrengthReport

    h = HarnessState.from_minimal()
    bb = BatchBuilder(h, n_min=16)
    for rs in make_forecast_batch("P1", n=36, seed0=0):
        bb.add_raw_series(rs)
    cells = bb.triggerable_cells(2)
    assert cells
    cell = cells[0]
    # 植入一个受保护 strength（模拟 D_{k-1} 沉淀的承重墙）
    h.l3.strength_signatures["sig0"] = StrengthSignatureStats("sig0", cell_id=cell, must_preserve=True)
    ev = Evolver(h, bb, StubProposer())

    # 新域上该片段**不再带正边际** → mine_strength 返回 must_preserve=False → 应降级
    monkeypatch.setattr(mining, "mine_strength",
                        lambda c, s, hh, **k: StrengthReport(c, 1.0, 0.5, -0.5, must_preserve=False))
    n = ev.revalidate_strength([cell])
    assert n == 1
    assert h.l3.strength_signatures["sig0"].must_preserve is False

    # 反例：仍带正边际 → 不降级
    h.l3.strength_signatures["sig1"] = StrengthSignatureStats("sig1", cell_id=cell, must_preserve=True)
    monkeypatch.setattr(mining, "mine_strength",
                        lambda c, s, hh, **k: StrengthReport(c, 0.3, 0.9, 0.6, must_preserve=True))
    assert ev.revalidate_strength([cell]) == 0
    assert h.l3.strength_signatures["sig1"].must_preserve is True


# ──────────────────────────── 4. updating reset-free 跨域携带 ────────────────────────────
def test_updating_is_reset_free():
    doms = _domains()
    c = ds.deploy_stream(doms, mode="updating", make_harness=HarnessState.from_minimal,
                         make_proposer=StubProposer, n_epochs_per_domain=2)
    # 第二域起点 = 第一域末状态（version 单调不回退；harness 对象持久）
    assert c.domains[1].harness_version >= c.domains[0].harness_version
    assert c.domains[1].harness_version >= 1   # 至少一处编辑被合入并跨域保留


# ──────────────────────────── 5. schedule.enter_new_domain 解冻+重热 ────────────────────────────
def test_enter_new_domain_unfreezes_and_rewarms():
    s = CellSchedule("forecast|x")
    for ep in range(3):                        # 连续全拒 → 冻结
        s.record_round(0, ep)
    assert s.status == FROZEN
    s.enter_new_domain(1)
    assert s.status == ACTIVE                   # ★解冻（freeze domain-scoped）
    assert s.round_idx == 0 and s.domain_idx == 1
    assert s.current_budget() >= 1


# ──────────────────────────── 6. models/registry 三角色分离 ────────────────────────────
def test_model_registry_role_separation():
    j_fc = set(models.get_models_for_role("J", "forecast"))
    m_fc = set(models.get_models_for_role("M_deploy", "forecast", include_todo=True))
    assert j_fc and m_fc
    assert j_fc.isdisjoint(m_fc)                # J ∩ M_deploy = ∅（守非循环）
    assert models.model_role_ok("chronos_judge", "J")
    assert not models.model_role_ok("chronos_judge", "M_deploy")


# ──────────────────────────── 7. readiness 度量 ────────────────────────────
def test_readiness_metric():
    # 弱判官（j_min_ref<j_raw，有清洗 headroom）→ 退化为旧式锚，语义不变
    assert readiness_score(2.0, 1.5, 1.5) == pytest.approx(1.0)   # 追平 min_ref（更优参照）
    assert readiness_score(2.0, 1.0, 1.5) == pytest.approx(2.0)   # 超越
    assert readiness_score(2.0, 2.0, 1.5) == pytest.approx(0.0)   # 等于 raw（更差参照）
    import math
    assert math.isnan(readiness_score(2.0, 1.0, 2.0))            # 两参照无差（无可观测尺度）
    agg = aggregate_time_to_readiness([1, 3, None, 2])
    assert agg["median"] == 2.0 and agg["max"] is None and agg["n_ready"] == 3


def test_readiness_judge_aware_fm():
    """基础模型判官（j_min_ref>j_raw：在 RAW 上更准）→ 参照翻向 raw，仍出有限可解释数（非旧式 nan）。"""
    # j_raw=1.0（更优）, j_min_ref=2.0（更差）；readiness 锚 = raw
    assert readiness_score(1.0, 1.0, 2.0) == pytest.approx(1.0)   # 追平 raw（撤销有害清洗）
    assert readiness_score(1.0, 2.0, 2.0) == pytest.approx(0.0)   # 仅及最小清洗（更差参照）
    assert readiness_score(1.0, 0.5, 2.0) == pytest.approx(1.5)   # 超越 raw
    assert readiness_score(1.0, 2.5, 2.0) == pytest.approx(-0.5)  # 比两参照都糟 = 真损伤
