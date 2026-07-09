"""Phase 2 验证：检索（distance/kNN/冷启动/signatures）+ LLM compose（stub+live）+ 暖启动集成。

运行：  PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.tests.test_phase2
"""
from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.harness import HarnessState
from SelfEvolvingHarnessTS.memory import EvidenceStore, MemoryIndex, aggregate_failures, retain_top
from SelfEvolvingHarnessTS.fast_path import process, compose_llm, Retriever, usable_ops
from SelfEvolvingHarnessTS.fast_path.compose import Program
from SelfEvolvingHarnessTS.conditioning import build_conditioning_key, similarity, distance
from SelfEvolvingHarnessTS.data import make_forecast_batch


def _key(series, task="forecast"):
    return build_conditioning_key(series, task)


# ── 1. distance/similarity 单调：越像 struct_feats → sim 越高 ─────────────
def test_distance_monotonic():
    t = np.arange(240)
    a = np.sin(2 * np.pi * t / 24) + 0.01 * t
    a2 = np.sin(2 * np.pi * t / 24) + 0.01 * t + np.random.default_rng(0).normal(0, 0.05, t.size)
    far = np.random.default_rng(1).normal(0, 1, t.size)        # 无结构
    ka, ka2, kfar = _key(a), _key(a2), _key(far)
    assert similarity(ka, ka2) > similarity(ka, kfar)
    assert distance(ka, ka) < 1e-6                              # 自距 ≈ 0


# ── 2. kNN 检索成功片段 + 冷启动空 ───────────────────────────────────────
def test_retrieve_success_and_cold_start():
    h = HarnessState.from_minimal()
    store = EvidenceStore()
    fb = make_forecast_batch("P1", 8, seed0=0)
    for rs in fb:                                              # 先跑 fast_path 攒 ready 记录
        process(rs.history, "forecast", h, store=store)
    # 冷启动：空 store → 空
    assert Retriever(None).retrieve(_key(fb[0].history)) == {"prior_fragments": [], "failure_warnings": []}
    # 用一个相似的新 forecast 序列检索 → 应取到成功片段
    probe = make_forecast_batch("P1", 1, seed0=999)[0]
    res = Retriever(store, h.l3.retrieval_config).retrieve(_key(probe.history))
    assert len(res["prior_fragments"]) >= 1
    assert all("program" in f and "sim" in f for f in res["prior_fragments"])


# ── 2b. S0.1/F1：检索按 task 硬过滤（forecast 记录不进 classify 查询）─────
def test_retrieve_task_hard_filter():
    h = HarnessState.from_minimal()
    store = EvidenceStore()
    for rs in make_forecast_batch("P1", 6, seed0=0):
        process(rs.history, "forecast", h, store=store)          # 只攒 forecast 证据
    idx = MemoryIndex(store)
    fc_probe = make_forecast_batch("P1", 1, seed0=999)[0]
    fc_key = _key(fc_probe.history, "forecast")
    cl_key = _key(fc_probe.history, "classification")            # 同结构、task 不同
    # 同 task 查询：仍应命中（过滤不过度阻断）
    assert len(idx.retrieve_success(fc_key, min_sim=0.0)) >= 1
    # 跨 task 查询：forecast 证据必须被硬过滤掉 → 0 命中
    assert idx.retrieve_success(cl_key, min_sim=0.0) == []
    assert idx.retrieve_failures(cl_key) == []


# ── 3. failure signature 聚合 + 保留 ─────────────────────────────────────
def test_failure_aggregation():
    h = HarnessState.from_minimal()
    for op in list(h.l2.active_operators):                     # 全禁 → 必走 fallback_original 留 failure_sig
        from SelfEvolvingHarnessTS.harness import EditPatch, Manifest
        h.apply_edit(EditPatch("L2", "set", f"l2.active_operators.{op}", False, Manifest("x")))
    store = EvidenceStore()
    for rs in make_forecast_batch("P1", 4, seed0=0):
        process(rs.history, "forecast", h, store=store)
    stats = aggregate_failures(store)
    assert stats and all(s.support >= 1 for s in stats.values())
    assert len(retain_top(stats, max_entries=1)) == 1


# ── 4. LLM compose（stub）：spec→Program，banned/未知算子被过滤，空→回退 ──
def test_compose_llm_stub():
    h = HarnessState.from_minimal()
    key = _key(make_forecast_batch("P1", 1)[0].history)

    def stub_ok(sys, user, nonce=0):
        return '{"steps":[{"op":"impute_linear","params":{}},{"op":"denoise_savgol","params":{"window":11}},{"op":"ghost_op","params":{}}]}'
    prog = compose_llm(key, h, [], [], stub_ok)
    assert prog.source == "llm_custom"
    assert prog.op_names() == ["impute_linear", "denoise_savgol"]   # ghost_op 被过滤

    def stub_empty(sys, user, nonce=0):
        return "no json"
    prog2 = compose_llm(key, h, [], [], stub_empty)                 # 解析失败 → 回退 heuristic
    assert prog2.source == "template"


# ── 5. LLM compose 受 L2 active_operators 约束（结构面真正生效）───────────
def test_compose_llm_respects_active_operators():
    h = HarnessState.from_minimal()
    from SelfEvolvingHarnessTS.harness import EditPatch, Manifest
    h.apply_edit(EditPatch("L2", "set", "l2.active_operators.denoise_savgol", False, Manifest("x")))
    key = _key(make_forecast_batch("P1", 1)[0].history)
    assert "denoise_savgol" not in usable_ops(h, "forecast")

    def stub(sys, user, nonce=0):
        return '{"steps":[{"op":"denoise_savgol"},{"op":"impute_linear"}]}'
    prog = compose_llm(key, h, [], [], stub)
    assert "denoise_savgol" not in prog.op_names()                 # 被禁算子被 Skill 物理排除


# ── 6. 集成：暖启动路径端到端产 ready（stub LLM compose + memory）─────────
def test_warmstart_pipeline():
    h = HarnessState.from_minimal()
    store = EvidenceStore()
    fb = make_forecast_batch("P1", 6, seed0=0)
    for rs in fb:
        process(rs.history, "forecast", h, store=store)            # 攒记忆

    def stub_llm(sys, user, nonce=0):
        return '{"steps":[{"op":"impute_linear"},{"op":"winsorize"},{"op":"denoise_savgol"}]}'
    probe = make_forecast_batch("P1", 1, seed0=777)[0]
    rec, art = process(probe.history, "forecast", h, store=store, memory=store, llm=stub_llm)
    assert rec.output_status == "ready" and art.shape == probe.history.shape
    assert rec.program["source"] == "llm_custom"


# ── 7. LLM compose 实测（联网，1 次）：真 LLM 产合法 gated 程序 ───────────
def test_compose_llm_live():
    from SelfEvolvingHarnessTS.llm import get_client
    h = HarnessState.from_minimal()
    store = EvidenceStore()
    for rs in make_forecast_batch("P1", 5, seed0=0):
        process(rs.history, "forecast", h, store=store)
    probe = make_forecast_batch("P1", 1, seed0=42)[0]
    ret = Retriever(store, h.l3.retrieval_config).retrieve(_key(probe.history))
    llm = get_client("flash", temperature=0.3, cache_name="compose_llm")
    rec, art = process(probe.history, "forecast", h, store=None, memory=store, llm=llm)
    print(f"    live LLM program: {[s['op'] for s in rec.program['steps']]} status={rec.output_status} "
          f"(warm-start fragments={len(ret['prior_fragments'])})")
    assert art.shape == probe.history.shape
    assert rec.output_status in ("ready", "fallback_recovery", "fallback_original")
    for s in rec.program["steps"]:                                 # 全部落在可用算子内（Skill 安全）
        assert s["op"] in usable_ops(h, "forecast")


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
