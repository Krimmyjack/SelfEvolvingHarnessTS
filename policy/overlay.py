"""policy/overlay.py — conditioned policy **overlay**（2.0-④，Harness 垂直切片第一段）。

与 deploy.routed_process（旧适配器：minimal+模板 harness **替换**执行）的本质区别：
overlay 把策略选中的 Program 经 `pipeline.process(forced_program=…)` **叠加**到**当前**
harness 上执行——L1 任务约束、L4 gate 定制、L3 memory/证据流、进化后 operator 状态全部
保留；gates 拒绝 forced program 时照常 recovery/identity（策略路由不旁路安全网）。
路由证据（policy 版本/预测效用/support/κ 等）经 EvidenceRecord.routing 落 L3（2.0-⑤ 兑现）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..fast_path.compose import Program, ProgramStep
from ..fast_path.pipeline import process
from ..memory import EvidenceRecord
from .action_spec import ActionMenu
from .router_policy import RouterPolicy, RoutingDecision


def program_from_spec(spec) -> Program:
    """ActionSpec（resolved params）→ Program。与 ActionCompiler.to_program 的编译结果逐步
    等价（等价性由 test_policy_contract 的四层测试锚定——此处直接由 resolved steps 构造，
    不再绕道 minimal harness 的 compose）。"""
    return Program(steps=[ProgramStep(op=s.op, params=dict(s.params)) for s in spec.steps],
                   source="policy_overlay", note=f"overlay:{spec.action_id}")


def routed_process_overlay(x, task_type: str, harness, router: RouterPolicy, menu: ActionMenu,
                           store=None, memory=None, task_spec: Optional[dict] = None,
                           batch_id: str = "", extra_routing: Optional[Dict[str, Any]] = None
                           ) -> Tuple[RoutingDecision, EvidenceRecord, np.ndarray]:
    """perceive（当前 harness）→ Router 决策 → overlay 执行 → 证据（含 routing）。"""
    from ..fast_path.perceive import perceive
    key = perceive(np.asarray(x, dtype=float).ravel(), task_type, harness, task_spec)
    decision = router.predict(key, menu)
    program = program_from_spec(menu.actions[decision.action_id])
    routing = {"selected_action": decision.action_id,
               "abstained": decision.abstained,
               "fallback_action": decision.fallback_action,
               **decision.provenance,
               **(extra_routing or {})}
    record, artifact = process(x, task_type, harness, store=store, task_spec=task_spec,
                               batch_id=batch_id, memory=memory,
                               forced_program=program, routing=routing)
    return decision, record, artifact
