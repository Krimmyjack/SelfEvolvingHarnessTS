"""policy/deploy.py — Fast Path 合流入口（opt-in，Stage 2.0-③）。

routed_process = perceive → RouterPolicy.predict → ActionCompiler.to_harness →
既有 fast_path.process（全 gate 链，含 D6 contract gate）。

设计边界：
  - **默认路径不变**：fast_path.process 签名与行为零改动；合流是新入口，不是替换；
  - routing provenance 由返回的 RoutingDecision 携带——EvidenceRecord 扩展是 2.0-⑤
    （落地期限：任何消费 routed 证据的实验之前，Component Plan v1.1c 写死）；
  - abstain 已在冻结臂内部落到 fallback action（picks 返回的就是回退后的动作），
    此处不再二次处理；
  - **本入口是 P0 策略等价执行适配器，不是完整 Harness 合流**（评审第二十三轮定位）：
    执行用 minimal+模板 harness，base_harness 只供 perceive——当前 harness 的 L1 约束/
    L3 memory/L4 gate 定制/进化后 operator 状态不参与执行。"在当前 Harness 上**叠加**
    conditioned policy 而非替换"= 2.0-④/2.5 收编的设计对象。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..fast_path import process
from ..fast_path.perceive import perceive
from ..harness import HarnessState
from .action_spec import ActionCompiler, ActionMenu
from .router_policy import RouterPolicy, RoutingDecision


def routed_process(x, task_type: str, router: RouterPolicy, menu: ActionMenu,
                   store=None, task_spec: Optional[Dict[str, Any]] = None,
                   batch_id: str = "", base_harness: Optional[HarnessState] = None,
                   compiler: Optional[ActionCompiler] = None
                   ) -> Tuple[RoutingDecision, Any, np.ndarray]:
    """返回 (decision, evidence_record, ready_artifact)。

    base_harness 只用于 perceive（binning 配置）；执行用的 harness 由 ActionCompiler
    从选中的 ActionSpec 编译——保证"线上执行的动作 = Router 选中的动作"。"""
    x = np.asarray(x, dtype=float).ravel()
    base = base_harness if base_harness is not None else HarnessState.from_minimal()
    key = perceive(x, task_type, base, task_spec)
    decision = router.predict(key, menu)
    spec = menu.actions[decision.action_id]
    comp = compiler if compiler is not None else ActionCompiler()
    h = comp.to_harness(spec, task_type)
    record, artifact = process(x, task_type, h, store=store, task_spec=task_spec,
                               batch_id=batch_id)
    return decision, record, artifact
