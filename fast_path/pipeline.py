"""fast_path/pipeline.py — 快路径编排（plan.md §4.1）。

PERCEIVE → (RETRIEVE 跳过，Phase 2) → COMPOSE → EXECUTE → VERIFY → EMIT。
只产证据、不改 harness（R2）。失败走 recovery→identity 三级 fallback，output_status 如实记录
（不变量 #5：失败即证据，不靠 heuristic 掩盖）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..memory import EvidenceRecord, EvidenceStore
from .perceive import perceive
from .compose import compose, compose_llm, compose_recovery, Program
from .execute import execute
from .retrieve import Retriever
from .verify import run_gates, role_b_score


def process(x, task_type: str, harness, store: Optional[EvidenceStore] = None,
            task_spec: Optional[Dict[str, Any]] = None,
            batch_id: str = "", memory=None, llm=None,
            forced_program=None, routing: Optional[Dict[str, Any]] = None
            ) -> Tuple[EvidenceRecord, np.ndarray]:
    """memory(=L3 EvidenceStore)→RETRIEVE 暖启动；llm→LLM compose。两者皆 None 时退化为 Phase 0/1
    heuristic 行为（慢路径 validator 走此默认 → 确定性、免 LLM）。

    forced_program（2.0-④ overlay，2026-07-05）：conditioned policy 选中的 Program **压过**
    compose/LLM 合成，但 perceive/gates/recovery/emit 全部仍用**当前 harness**——L1 约束、
    L4 gate 定制、L3 证据流原样生效（≠routed_process 旧适配器的 minimal 替换语义）。
    gates 拒绝 forced program 时照常走 recovery/identity（安全网不因策略路由而旁路）。
    routing：写入 EvidenceRecord.routing 的路由证据（2.0-⑤）。"""
    x = np.asarray(x, dtype=float).ravel()

    # PERCEIVE
    key = perceive(x, task_type, harness, task_spec)
    cell_id = key["cell_id"]

    # RETRIEVE（冷启动/无 memory → 空）
    retrieval = Retriever(memory, getattr(harness.l3, "retrieval_config", None)).retrieve(key)

    # COMPOSE → EXECUTE → VERIFY（主程序）：overlay 压过 > LLM > heuristic
    if forced_program is not None:
        program = forced_program
    elif llm is not None:
        program = compose_llm(key, harness, retrieval["prior_fragments"], retrieval["failure_warnings"], llm)
    else:
        program = compose(key, harness)
    pattern_bin = key.get("pattern_bin", "")
    feats = (key.get("pattern") or {}).get("struct_feats")     # 软结构门：与 compose 同源
    exec_res = execute(program, x)
    passed, gates, sig = run_gates(x, exec_res, program, harness, task_type, pattern_bin, feats)

    if passed:
        artifact, status, out_gates, out_sig, out_exec = exec_res.artifact, "ready", gates, None, exec_res
    else:
        # recovery
        rprog = compose_recovery(key, harness, sig)
        rexec = execute(rprog, x)
        rpassed, rgates, _ = run_gates(x, rexec, rprog, harness, task_type, pattern_bin, feats)
        if rpassed:
            artifact, status, out_gates, out_sig, out_exec = rexec.artifact, "fallback_recovery", rgates, sig, rexec
            program = rprog
        else:
            # identity：保留主程序失败证据（sig），产物=原序列
            artifact, status, out_gates, out_sig, out_exec = x.copy(), "fallback_original", gates, sig, exec_res

    rb = role_b_score(x, artifact, task_type)
    record = EvidenceRecord(
        conditioning_key=key,
        cell_id=cell_id,
        harness_version=harness.version,
        program=program.to_dict(),
        execution_trace=out_exec.trace,
        verification_result={
            "passed": status in ("ready", "fallback_recovery"),
            "gate_results": [g.__dict__ for g in out_gates],
            "failure_signature": out_sig,
            "role_b_score": rb,
            "output_status": status,
        },
        batch_id=batch_id,
        routing=routing,
    )
    if store is not None:
        store.write(record)
    return record, np.asarray(artifact, dtype=float).ravel()
