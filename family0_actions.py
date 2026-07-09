"""family0_actions.py — E-3.3 Family 0 剂量维动作定义（单一真源，A-32/评审第十二轮）。

剂量网格（评审指定）：denoise_median {9,15,25} / denoise_savgol {21,31} / smooth_ma {9,15,25}。
每个动作 ID **含参数**（如 `f0_median_w9`，非只写算子名）。每动作 = minimal harness +
一条 `[impute_linear → <op window=W>]` 模板（与 v_median 等 v2.1 变体同构，仅换算子/窗）。
所有动作在 F0 gate 测（`tests/test_dosage_operators.py`）全过后才允许进入 nested 评估。

供给实验解释边界（评审强调，勿越界）：F0 若成功，只证明 **degradation/SNR 条件化的剂量路由**
有价值——因为 selection 单位仍是 cell=SNR×missing。是否由 **内部 Pattern**（而非 degradation）
预测该剂量维，须留给 E-3.2（P-only/D-only/D+P/cell 内结构对比/LODO）。

已知剂量特性（**非伪影，预期由 nested 选择器处理**）：`f0_savgol_w31`（窗 31 > 语料主周期 24、
polyorder=3）在季节序列端点会多项式过冲（诊断：period-24 正弦上末点 1.0→1.76）。端点正是编码器
输入位置 → 其 forecasting loss 在季节 cell 偏高，选择器应据此拒绝它。这是**剂量质量的真实信号**
（区别于 S0.7-8 的零填充 bug：那是静默、影响未被选中的 incumbent；此处是候选被正确评估）。
保留该点（评审指定网格）以让 F0 同时探测"过量平滑何时开始伤害"。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .harness import HarnessState
from .harness.layers import PipelineTemplate

# (action_id, op, window) —— 动作 ID 含窗参数
F0_DOSAGE_GRID: List[Tuple[str, str, int]] = [
    ("f0_median_w9",  "denoise_median", 9),
    ("f0_median_w15", "denoise_median", 15),
    ("f0_median_w25", "denoise_median", 25),
    ("f0_savgol_w21", "denoise_savgol", 21),
    ("f0_savgol_w31", "denoise_savgol", 31),
    ("f0_ma_w9",      "smooth_ma",      9),
    ("f0_ma_w15",     "smooth_ma",      15),
    ("f0_ma_w25",     "smooth_ma",      25),
]


def dosage_variant(name: str, op: str, window: int, task: str = "forecast") -> HarnessState:
    """构造单个剂量动作 harness（impute_linear → op@window），与 fixed_harness_variants 同构。"""
    h = HarnessState.from_minimal()
    stages = [
        {"stage": "s1", "preferred_ops": ["impute_linear"], "banned_ops": [], "params_override": {}},
        {"stage": "s1", "preferred_ops": [op], "banned_ops": [], "params_override": {"window": int(window)}},
    ]
    h.l2.task_templates[name] = PipelineTemplate.from_dict(
        {"name": name, "applies_to": {"task_type": task, "pattern_conditions": None}, "stages": stages})
    return h


def f0_variants(task: str = "forecast") -> Dict[str, HarnessState]:
    """全部 8 个 F0 剂量动作 → {action_id: harness}。"""
    return {name: dosage_variant(name, op, w, task) for name, op, w in F0_DOSAGE_GRID}
