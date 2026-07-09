"""tests/test_d6_contract.py — D6 修复回归（Stage 2.0-①，Component Plan v1.1b）。

D6（已核实缺陷）：`_first_usable` 签名无 task → recovery 的 winsorize（destructive）可绕过
registry `allowed_tasks` 契约上 anomaly（毁 spike）；`run_gates` 不复查契约。

修复 = 统一资格函数 `is_operator_eligible(op, task, harness, banned)`，由模板 compose /
heuristic compose / LLM compose(usable_ops) / recovery / run_gates 五路共同调用；
run_gates 新增 Contract gate 物理复查。本 suite 守：
  ① recovery 在 anomaly 下不含 destructive/smoothing（曾经的越权路径）；
  ② recovery 在 forecast 下行为不变（仍含 winsorize）；
  ③ 模板 preferred_ops 越权算子在 anomaly 下被过滤；
  ④ run_gates 对越权程序 fail 在 contract gate（recovery/LLM/模板任何来源都拦）；
  ⑤ is_operator_eligible 语义：alias 解析 / task=None 与未知 task 不过滤（向后兼容）；
  ⑥ 端到端 process：anomaly 全链产物程序不含越权算子。
"""
from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.conditioning.key import build_conditioning_key
from SelfEvolvingHarnessTS.fast_path import process
from SelfEvolvingHarnessTS.fast_path.compose import (Program, ProgramStep, compose,
                                                     compose_recovery, is_operator_eligible)
from SelfEvolvingHarnessTS.fast_path.verify import run_gates
from SelfEvolvingHarnessTS.harness import HarnessState
from SelfEvolvingHarnessTS.harness.layers import PipelineTemplate
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA
from SelfEvolvingHarnessTS.sandbox import run_pipeline


def _series(n: int = 128, miss: bool = True) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * np.arange(n) / 24) + 0.1 * rng.standard_normal(n)
    x[5] = 8.0                              # spike（anomaly 的信号，不是脏东西）
    if miss:
        x[40:44] = np.nan
    return x


def _violates_contract(program, task: str) -> list:
    return [s.op for s in program.steps
            if task not in OPERATOR_METADATA.get(s.op, {}).get("allowed_tasks", (task,))]


# ── ① recovery 越权关闭（D6 核心）─────────────────────────────────────────
def test_recovery_anomaly_no_destructive():
    h = HarnessState.from_minimal()
    key = build_conditioning_key(_series(), "anomaly_detection")
    prog = compose_recovery(key, h, "blowup:test")
    assert prog.steps, "anomaly recovery 应至少保留温和插补"
    assert not _violates_contract(prog, "anomaly_detection"), \
        f"destructive/smoothing 算子泄入 anomaly recovery: {prog.op_names()}"
    for s in prog.steps:
        meta = OPERATOR_METADATA[s.op]
        assert not meta["destructive"] and "smoothing" not in meta["tags"], s.op


# ── ② forecast recovery 行为不变 ──────────────────────────────────────────
def test_recovery_forecast_keeps_winsorize():
    h = HarnessState.from_minimal()
    key = build_conditioning_key(_series(), "forecast")
    prog = compose_recovery(key, h, "blowup:test")
    assert "winsorize" in prog.op_names(), "forecast recovery 应保留 winsorize（压爆炸）"


# ── ③ 模板路径同受契约约束 ────────────────────────────────────────────────
def test_template_preferred_ops_filtered_on_anomaly():
    h = HarnessState.from_minimal()
    h.l2.task_templates["bad_anom"] = PipelineTemplate.from_dict({
        "name": "bad_anom",
        "applies_to": {"task_type": "anomaly_detection", "pattern_conditions": None},
        "stages": [
            {"stage": "s1", "preferred_ops": ["winsorize", "impute_linear"],
             "banned_ops": [], "params_override": {}},
        ]})
    key = build_conditioning_key(_series(), "anomaly_detection")
    key["pattern_bin"] = ""
    prog = compose(key, h)
    assert "winsorize" not in prog.op_names(), \
        "模板 preferred_ops 的 destructive 算子不得上 anomaly（应退到同 stage 后备 impute_linear）"
    assert not _violates_contract(prog, "anomaly_detection")


# ── ④ run_gates 契约复查（不信任 compose 来源）────────────────────────────
def test_run_gates_contract_blocks_violation():
    h = HarnessState.from_minimal()
    x = _series()
    prog = Program(steps=[ProgramStep("impute_linear", {}), ProgramStep("winsorize", {})])
    res = run_pipeline(prog.as_pairs(), x)
    passed, gates, sig = run_gates(x, res, prog, h, "anomaly_detection")
    assert not passed and sig.startswith("contract:task_contract"), (passed, sig)
    # 同一程序在 forecast 下应通过 contract（后续 gate 各自独立判）
    passed_f, gates_f, sig_f = run_gates(x, res, prog, h, "forecast")
    assert any(g.name == "contract" and g.passed for g in gates_f), sig_f


def test_run_gates_contract_resolves_alias():
    """旧 artifact 用旧 ID 重放：契约按 canonical 判（kalman_filter≡smooth_ema=smoothing）。"""
    h = HarnessState.from_minimal()
    x = _series(miss=False)
    prog = Program(steps=[ProgramStep("kalman_filter", {"alpha": 0.3})])
    res = run_pipeline(prog.as_pairs(), x)
    passed, _gates, sig = run_gates(x, res, prog, h, "anomaly_detection")
    assert not passed and "kalman_filter" in (sig or ""), (passed, sig)


# ── ⑤ is_operator_eligible 语义 ───────────────────────────────────────────
def test_eligibility_semantics():
    h = HarnessState.from_minimal()
    assert is_operator_eligible("winsorize", "forecast", h, set())
    assert not is_operator_eligible("winsorize", "anomaly_detection", h, set())
    assert is_operator_eligible("impute_linear", "anomaly_detection", h, set())
    # alias 解析后判契约与 active
    assert is_operator_eligible("fill_gaps", "anomaly_detection", h, set())
    assert not is_operator_eligible("fill_gaps", "forecast", h, {"impute_linear"})
    # task=None / 未知 task → 不做契约过滤（兼容旧调用与未知任务旧行为）
    assert is_operator_eligible("winsorize", None, h, set())
    assert is_operator_eligible("winsorize", "some_future_task", h, set())


# ── ⑥ 端到端：anomaly 全链（含可能触发的 recovery）不产越权程序 ─────────────
def test_process_anomaly_end_to_end_contract_clean():
    h = HarnessState.from_minimal()
    rec, artifact = process(_series(), "anomaly_detection", h, store=None)
    ops = [s["op"] for s in rec.program["steps"]]
    bad = [op for op in ops
           if "anomaly_detection" not in OPERATOR_METADATA.get(op, {}).get("allowed_tasks",
                                                                           ("anomaly_detection",))]
    assert not bad, f"anomaly 端到端程序含越权算子: {ops}"
    assert artifact.shape == _series().shape
