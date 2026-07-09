"""Phase 1 验证：慢路径进化引擎端到端（batch_builder→mining→validator→merger→evolve）。

用 StubProposer（免 LLM、确定性）证明 Phase 1 出口判据：≥1 个 accepted edit 在 held-out 带正增益且
Pareto 安全。场景 = 先把 harness 降级（关掉 forecast 的离群算子），再让 proposer 提议重开 → 应被接受。

运行：  PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.tests.test_slow_path
"""
from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.harness import HarnessState, EditPatch, Manifest
from SelfEvolvingHarnessTS.slow_path import (
    BatchBuilder, Validator, Merger, Evolver, mine_weakness,
)
from SelfEvolvingHarnessTS.slow_path.batch_builder import CellSample
from SelfEvolvingHarnessTS.data import make_forecast_batch, make_anomaly_batch

_OUTLIER_OPS = ["winsorize", "outlier_iqr", "outlier_mad"]
# S0.2 后 splits 走 SHA256 分组洗牌（不再 pool[:bs] 顺序切）→ held_in 是随机子集。
# ε=0.03 在 N_MIN=16（σ_Δ≈0.028）校准；用同规模 batch 才能让"真收益"稳定越过 ε，
# 否则 N=6 的抽样噪声会淹没边际收益（旧 pass 依赖插入顺序的运气）。
NMIN = 16


def _mf(fid="w"):
    return Manifest(fid, "outliers leak into forecast", "lower forecast nRMSE", "revert", "low")


def _degraded_forecast_harness():
    """关掉所有离群算子 → forecast 在含离群的 P1 上次优（离群泄漏进下游训练）。"""
    h = HarnessState.from_minimal()
    for op in _OUTLIER_OPS:
        h.apply_edit(EditPatch("L2", "set", f"l2.active_operators.{op}", False, _mf()))
    return h


def _build_bb(harness):
    bb = BatchBuilder(harness, n_min=NMIN)
    for rs in make_forecast_batch("P1", 2 * NMIN, seed0=0):      # forecast cell（≥2 batch）
        bb.add_raw_series(rs)
    for rs in make_anomaly_batch("P1", NMIN, seed0=300):         # 第二 cell → held_out(b)
        bb.add_raw_series(rs)
    return bb


class StubProposer:
    """确定性：对 forecast cell 提议重开 winsorize（应改善）。"""
    def __init__(self, patches):
        self._patches = patches
    def propose(self, harness, weakness, strength=None, rejection_log=None):
        out = []
        for i, p in enumerate(self._patches):
            q = EditPatch.from_dict(p.to_dict())
            q.cell_id = weakness.cell_id
            q.harness_ver = harness.version
            q.proposal_rank = i
            out.append(q)
        return out


def _forecast_cell(bb):
    return next(c for c in bb.triggerable_cells() if c.startswith("forecast|"))


# ── 1. validator 直测：重开 winsorize 被接受（held-out 正增益 + Pareto 安全）──
def test_validator_accepts_beneficial_edit():
    h = _degraded_forecast_harness()
    bb = _build_bb(h)
    cell = _forecast_cell(bb)
    good = EditPatch("L2", "set", "l2.active_operators.winsorize", True, _mf(), cell_id=cell)
    out = Validator().validate(good, h, cell, bb.splits(cell))
    assert out.accept, f"expected accept, got {out.reason} (Δin={out.val_in_cur - out.val_in_cand:.3f})"
    assert out.val_in_cand < out.val_in_cur          # held-in 改善
    assert out.pareto_safe                           # 跨 cell 安全
    assert h.l2.active_operators["winsorize"] is False   # validator 未污染当前 harness


# ── 2. validator 拒绝有害编辑（关掉插补 → forecast 退化）─────────────────
def test_validator_rejects_harmful_edit():
    h = HarnessState.from_minimal()
    bb = _build_bb(h)
    cell = _forecast_cell(bb)
    bad = EditPatch("L2", "set", "l2.active_operators.impute_linear", False, _mf(), cell_id=cell)
    out = Validator().validate(bad, h, cell, bb.splits(cell))
    # 关插补不会改善 held-in（可能持平或更差）→ 不接受
    assert not out.accept


# ── 3. 出口判据：Evolver 一轮内接受 ≥1 编辑，版本前进 ──────────────────
def test_evolver_accepts_and_bumps_version():
    h = _degraded_forecast_harness()
    bb = _build_bb(h)
    cell = _forecast_cell(bb)
    v0 = h.version
    good = EditPatch("L2", "set", "l2.active_operators.winsorize", True, _mf())
    ev = Evolver(h, bb, StubProposer([good]))
    rr = ev.evolve_cell(cell, epoch=0)
    assert rr.n_accepted >= 1, f"reasons={rr.reasons}"
    assert h.version > v0
    assert h.l2.active_operators["winsorize"] is True     # 真正合入
    assert len(h.patch_log) >= 1


# ── 4. 冻结：连续全拒 → cell 冻结（stub 一直提无效编辑）──────────────────
def test_evolver_freezes_on_repeated_reject():
    from SelfEvolvingHarnessTS.config import thresholds as TH
    h = HarnessState.from_minimal()
    bb = _build_bb(h)
    cell = _forecast_cell(bb)
    # 提议一个合法但无改善的编辑（重开已激活的算子 = 无效果）
    noop = EditPatch("L2", "set", "l2.active_operators.znorm", True, _mf())
    ev = Evolver(h, bb, StubProposer([noop]))
    for e in range(TH.N_FREEZE):
        ev.evolve_cell(cell, epoch=e)
    assert ev.schedules[cell].status == "frozen"


# ── S0.2/F2 ①：同 series_uid 的退化副本必须落同一 split（防基底泄漏）──────
def test_split_groups_replicas_together():
    h = HarnessState.from_minimal()
    bb = BatchBuilder(h, n_min=8)
    cid = "forecast|snrHigh|full"
    pool = []
    for k in range(8):                                   # 8 基底 × 4 退化副本 = 32；2 origin
        origin = "dsA" if k < 4 else "dsB"
        for _r in range(4):
            pool.append(CellSample(np.zeros(96), "forecast", future=np.zeros(4),
                                   origin=origin, series_uid=f"sig{k}"))
    bb.pools[cid] = pool
    hi, ha, ft, tg = bb._partition(cid, 8)
    seg_of = {}
    for name, seg in (("in", hi), ("a", ha), ("ft", ft), ("tg", tg)):
        for s in seg:
            seg_of.setdefault(s.series_uid, set()).add(name)
    for uid, segs in seg_of.items():
        assert len(segs) == 1, f"series_uid {uid} 跨 split 泄漏到 {segs}"
    assert bb.series_uid_count(cid) == 8 and not bb.is_low_confidence(cid)


# ── S0.2/F2 ②：两 origin 各 16 样本 → 三 split 的 origin 比例 ≈ 1:1（分层）──
def test_split_origin_stratified():
    h = HarnessState.from_minimal()
    bb = BatchBuilder(h, n_min=10)
    cid = "forecast|snrHigh|full"
    pool = ([CellSample(np.zeros(96), "forecast", origin="dsA", series_uid=f"A{k}") for k in range(16)]
            + [CellSample(np.zeros(96), "forecast", origin="dsB", series_uid=f"B{k}") for k in range(16)])
    bb.pools[cid] = pool
    for seg in bb._partition(cid, 10)[:3]:            # 检查三大 split（第 4 段 transfer_gate 为余量）
        na = sum(1 for s in seg if s.origin == "dsA")
        nb = sum(1 for s in seg if s.origin == "dsB")
        assert len(seg) >= 8 and abs(na - nb) <= 2, f"origin 不平衡 {na} vs {nb}"


# ── S0.3/F3：mining 按 harness_version 过滤证据（v1 记录 bump 到 v2 后不可见）──
def test_mining_version_filter():
    from SelfEvolvingHarnessTS.memory import EvidenceStore, EvidenceRecord
    h = HarnessState.from_minimal()
    bb = _build_bb(h)
    cell = _forecast_cell(bb)
    store = EvidenceStore()
    ck = {"task": {"type": "forecast"}, "pattern": {}}
    vr = {"passed": False, "failure_signature": "sig_boom", "output_status": "fallback_original"}
    for i in range(3):                                   # ≥ MIN_SUPPORT 条 v1 证据
        store.write(EvidenceRecord(ck, cell, 1, {}, [], dict(vr), batch_id=f"b{i}"))
    h.bump_version(2)                                     # merger 语义：显式版本递进到 v2
    assert h.version == 2
    held_in = bb.splits(cell)[0]
    w_strict = mine_weakness(cell, held_in, h, store, version_arm="strict")
    assert w_strict.failure_signatures == {}                       # v1≠v2 → 严格臂看不见
    assert w_strict.evidence_n_raw == 3 and w_strict.evidence_n_kept == 0
    w_prev = mine_weakness(cell, held_in, h, store, version_arm="prev")
    assert w_prev.failure_signatures.get("sig_boom") == 3          # ≥v-1 臂看得见
    assert w_prev.evidence_n_kept == 3
    w_all = mine_weakness(cell, held_in, h, store, version_arm="all")
    assert w_all.failure_signatures.get("sig_boom") == 3


# ── S0.5：候选级 JSONL 日志器记录全部候选（含被拒）+ 评估明细 ────────────
def test_candidate_logger_records_all(tmp_path):
    import json as _json
    from SelfEvolvingHarnessTS.slow_path import CandidateLogger
    h = _degraded_forecast_harness()
    bb = _build_bb(h)
    cell = _forecast_cell(bb)
    good = EditPatch("L2", "set", "l2.active_operators.winsorize", True, _mf())
    noop = EditPatch("L2", "set", "l2.active_operators.znorm", True, _mf())   # 已激活→无改善→拒
    logpath = tmp_path / "cand.jsonl"
    ev = Evolver(h, bb, StubProposer([good, noop]),
                 candidate_logger=CandidateLogger(str(logpath), run_id="t"))
    ev.evolve_cell(cell, epoch=0)
    lines = [_json.loads(l) for l in logpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2                                       # accepted + rejected 都落
    accepts = [r for r in lines if r["outcome"]["accept"]]
    rejects = [r for r in lines if not r["outcome"]["accept"]]
    assert len(accepts) == 1 and len(rejects) == 1
    r = accepts[0]
    assert r["patch"]["path"] == "l2.active_operators.winsorize" and r["patch"]["op"] == "set"
    for k in ("v_in_cur", "v_in_cand", "v_a_cur", "v_a_cand", "delta_in", "delta_a"):
        assert k in r["outcome"]                                 # 四 v 值 + Δ 都在
    fp = r["split_fingerprint"]
    assert fp["held_in"]["n"] >= 1 and "series_uids" in fp["held_in"] and "origins" in fp["held_in"]
    assert "held_out_b" in fp                                    # 跨 cell Pareto 组指纹
    assert len(r["artifact_key"]) == 8 and len(r["judge_fingerprint"]) == 8   # 确定性重评锚
    assert rejects[0]["patch"]["path"] == "l2.active_operators.znorm"   # 被拒候选完整 patch（模式 B 依赖）


# ── S0.4：warm-start 模板迁移重验 → 负迁移模板 demote 为 advisory（不删）──
def test_revalidate_templates_demotes_negative_transfer():
    """S0.4 降级机制守卫。旧场景（anomaly+savgol）在 D6 后已被 contract 物理拦截于 compose 层
    ——模板表达不出危害 → 无从降级（那是 D6 的正确行为，由 test_d6_contract 守）。
    改用**契约合法但有害**场景：forecast 上强制重剂量 median(w=25)——F0 final 已证
    窗≈周期 24 抹平季节性 → 结构性损伤（S_season 全 4 cell LCB<−δ_safe）。"""
    from SelfEvolvingHarnessTS.harness import PipelineTemplate, StageDef
    from SelfEvolvingHarnessTS.fast_path.perceive import perceive
    h = HarnessState.from_minimal()
    bb = _build_bb(h)
    cell = _forecast_cell(bb)
    pb = perceive(bb.splits(cell)[0][0].raw, "forecast", h)["pattern_bin"]
    bad = PipelineTemplate("force_heavy_median",
                           {"task_type": "forecast", "pattern_conditions": {"pattern_bin": pb}},
                           [StageDef("s1", preferred_ops=["impute_linear"]),
                            StageDef("s2", preferred_ops=["denoise_median"],
                                     params_override={"window": 25})])
    h.apply_edit(EditPatch("L2", "set", "l2.task_templates::force_heavy_median", bad, _mf()))
    assert not h.l2.task_templates["force_heavy_median"].applies_to.get("advisory")
    ev = Evolver(h, bb, StubProposer([]))
    n = ev.revalidate_templates([cell])
    assert n == 1, "有害模板应被迁移重验降级"
    assert h.l2.task_templates["force_heavy_median"].applies_to.get("advisory") is True  # 降级但保留（不删）


# ── 5. mining：weakness_report 给出 baseline + 可改进性 ──────────────────
def test_mining_weakness_report():
    h = _degraded_forecast_harness()
    bb = _build_bb(h)
    cell = _forecast_cell(bb)
    w = mine_weakness(cell, bb.splits(cell)[0], h)
    assert w.task == "forecast" and np.isfinite(w.current_val_loss) and np.isfinite(w.floor)


def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
