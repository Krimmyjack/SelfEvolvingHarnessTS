"""Phase 0 里程碑验证：快路径端到端（perceive→compose→execute→verify→emit）。

运行：  python -m SelfEvolvingHarnessTS.tests.test_fast_path   （cwd=Agent）
"""
from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.harness import HarnessState, EditPatch, Manifest
from SelfEvolvingHarnessTS.memory import EvidenceStore
from SelfEvolvingHarnessTS.fast_path import process, compose, run_gates, execute, Program, ProgramStep
from SelfEvolvingHarnessTS.fast_path.perceive import perceive


def _degraded(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(240)
    x = (np.sin(2 * np.pi * t / 24) + 0.01 * t + rng.normal(0, 0.5, t.size)).astype(float)
    x[[40, 90, 150, 200]] = 18.0      # 离群
    x[100:104] = np.nan               # 缺失
    return x


def _mf():
    return Manifest("f001", "d", "e", "a", "r")


# ── 1. 里程碑：degraded forecast → ready_artifact + EvidenceRecord，Gate 全过 ──
def test_milestone_forecast_ready():
    h = HarnessState.from_minimal()
    store = EvidenceStore()
    x = _degraded()
    rec, art = process(x, "forecast", h, store=store)

    assert rec.output_status == "ready"
    assert art.shape == x.shape
    assert not np.isnan(art).any() and np.all(np.isfinite(art))
    assert all(g["passed"] for g in rec.verification_result["gate_results"])
    assert {g["name"] for g in rec.verification_result["gate_results"]} == {
        "ast", "skill", "contract", "sandbox", "blowup", "constraint"}   # contract = D6 新 gate
    assert rec.harness_version == 0
    assert len(store) == 1 and store.query_by_cell(rec.cell_id)[0] is rec


# ── 2. C1：同一退化输入，forecast vs anomaly 产出不同程序 ──
def test_c1_task_divergence():
    h = HarnessState.from_minimal()
    x = _degraded()
    fkey, akey = perceive(x, "forecast", h), perceive(x, "anomaly_detection", h)
    fprog, aprog = compose(fkey, h), compose(akey, h)

    f_ops, a_ops = fprog.op_names(), aprog.op_names()
    # forecast 去噪+去离群；anomaly 禁平滑禁删离群
    assert any(o.startswith("denoise") for o in f_ops)
    assert any(o in ("winsorize", "outlier_iqr", "outlier_mad") for o in f_ops)
    assert not any(o.startswith("denoise") for o in a_ops)
    assert not any(o in ("winsorize", "outlier_iqr", "outlier_mad") for o in a_ops)
    # forecast 绝不 standardize
    assert "znorm" not in f_ops and "minmax_norm" not in f_ops
    assert f_ops != a_ops
    # cell_id 按 task 分流
    assert fkey["cell_id"].startswith("forecast|") and akey["cell_id"].startswith("anomaly_detection|")


# ── 3. anomaly 仍产 ready（仅插补，保异常信号）──
def test_anomaly_ready():
    h = HarnessState.from_minimal()
    rec, art = process(_degraded(), "anomaly_detection", h)
    assert rec.output_status == "ready"
    assert not np.isnan(art).any()
    # 离群点应被保留（未删除）—— 检查仍存在大幅值
    assert np.max(np.abs(art)) > 10.0


# ── 4. Skill Gate 物理强制：进化禁用算子后，用它的程序被拦 ──
def test_skill_gate_enforces_disabled_op():
    h = HarnessState.from_minimal()
    # 进化禁用 denoise_savgol
    h.apply_edit(EditPatch("L2", "set", "l2.active_operators.denoise_savgol", False, _mf()))
    bad = Program([ProgramStep("denoise_savgol", {})], source="llm_custom")
    x = _degraded()
    ex = execute(bad, x)
    passed, gates, sig = run_gates(x, ex, bad, h, "forecast")
    assert not passed and sig.startswith("skill:") and "denoise_savgol" in sig


# ── 5. Fallback 阶梯：全算子禁用 + 缺失数据 → 主程序与 recovery 皆空 → fallback_original ──
#    注：每个算子内部都会防御性 interp_nan（避免崩溃），所以只禁 impute 不够——任一下游算子
#    仍会顺手填补 NaN。要确定性触发 fallback，须禁用全部算子（空程序 → NaN 残留 → gate 失败）。
def test_fallback_ladder():
    h = HarnessState.from_minimal()
    for op in list(h.l2.active_operators):
        h.apply_edit(EditPatch("L2", "set", f"l2.active_operators.{op}", False, _mf()))
    rec, art = process(_degraded(), "forecast", h)
    assert rec.output_status == "fallback_original"
    assert rec.verification_result["passed"] is False
    sig = rec.verification_result["failure_signature"]
    assert sig is not None and ("blowup:" in sig or "constraint:" in sig)   # NaN 残留被 gate 拦
    assert np.isnan(art).any()    # identity fallback 如实返回原序列（含 NaN），不掩盖


# ── 6. 干净输入（无缺失/离群/噪声）→ identity 程序仍 ready ──
def test_clean_input_identity_ready():
    h = HarnessState.from_minimal()
    clean = np.linspace(0.0, 8.0, 240)        # 线性：所有探测器判干净（MA 保线 → 高 SNR）
    rec, art = process(clean, "forecast", h)
    assert rec.output_status == "ready"
    assert np.allclose(art, clean)
    assert rec.program["note"].endswith("identity")


# ── 7. EvidenceRecord 可序列化（审计）──
def test_record_serializable():
    import json
    h = HarnessState.from_minimal()
    rec, _ = process(_degraded(), "forecast", h)
    s = json.dumps(rec.to_dict(), ensure_ascii=False, default=float)
    assert "output_status" in s and "conditioning_key" in s


def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
