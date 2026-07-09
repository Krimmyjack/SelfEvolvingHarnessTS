"""models/registry.py — 下游模型库注册表（单一真源，镜像 operators/registry.py）。

三方分离不变量（plan §5 #9 / Refactor_v4 §2.2）：J 判官 / M_deploy 部署 / R 报告器
**互不相交**。本表按角色登记，**靠 API 设计强制**（get_models_for_role 过滤），不靠运行时 assert：
  • L5 harness 选模型 → get_models_for_role("M_deploy", task)，机械不可能选到 J 池条目。
  • 判官/报告器同架构 = **不同实例**（如 chronos_judge 确定性 do_sample=False vs chronos_report 异 seed）→
    name 带角色后缀，避免 report_target.disjoint_targets 的字符串匹配误判。

字段：family（frozen-encoder|TSFM|deep|deep-light|traditional|rule）、task、allowed_roles、
deterministic（J 须 True → 省 seed）、instance_config（同架构不同实例的区分）、status（present|todo）。

S1 仅需 allowed_roles 含 "J"/"R" 的条目（多现成）。allowed_roles 含 "M_deploy" 的为 S4（L5）待建。
"""
from __future__ import annotations

from typing import Dict, List

MODEL_ROLES = ("J", "M_deploy", "R")

# (name, family, task, allowed_roles, deterministic, instance_config, status)
MODEL_SPECS = [
    # ── J 判官（model-agnostic readiness 锚；须确定性 σ_A=0）──────────────────────
    ("frozen_probe",    "frozen-encoder", "forecast",       ("J",),  True,  {},                    "present"),
    ("chronos_judge",   "TSFM",           "forecast",       ("J",),  True,  {"do_sample": False},  "present"),
    ("rocket_judge",    "traditional",    "classification", ("J",),  True,  {},                    "present"),
    ("topk_recall",     "rule",           "anomaly_detection", ("J",), True, {},                   "present"),
    # ── R 报告器（final-test，从不在环内，∉ J）──────────────────────────────────
    ("lstm_scratch",    "deep",           "forecast",       ("R", "M_deploy"), False, {},          "present"),
    ("dlinear_scratch", "deep-light",     "forecast",       ("R", "M_deploy"), False, {},          "present"),
    ("chronos_report",  "TSFM",           "forecast",       ("R",),  False, {"seed_offset": 7},    "present"),
    ("inception",       "deep",           "classification", ("R", "M_deploy"), False, {},          "present"),
    # ── M_deploy 部署池（L5 / S4 待建；TSFM + 传统 GBDT）─────────────────────────
    ("timesfm_zeroshot","TSFM",           "forecast",       ("M_deploy",), False, {},              "todo"),
    ("chronos_lora",    "TSFM-ft",        "forecast",       ("M_deploy",), False, {"finetune": "lora"}, "todo"),
    ("gbdt_lag",        "traditional",    "forecast",       ("M_deploy",), False, {"feats": "lag"},     "todo"),
    ("gbdt_feat",       "traditional",    "classification", ("M_deploy",), False, {"feats": "catch22"}, "todo"),
    ("aedcnn",          "deep",           "anomaly_detection", ("R", "M_deploy"), False, {},        "todo"),
]

MODEL_METADATA: Dict[str, dict] = {
    name: {"name": name, "family": fam, "task": task, "allowed_roles": list(roles),
           "deterministic": det, "instance_config": dict(cfg), "status": st}
    for (name, fam, task, roles, det, cfg, st) in MODEL_SPECS
}


def get_models_for_role(role: str, task: str | None = None, *, include_todo: bool = False) -> List[str]:
    """按角色（+可选 task）取可用模型名。L5 选模型走此函数 → 机械不可能选到 J 池（守三方分离）。"""
    if role not in MODEL_ROLES:
        raise ValueError(f"role ∈ {MODEL_ROLES}, got {role!r}")
    return [n for n, m in MODEL_METADATA.items()
            if role in m["allowed_roles"]
            and (task is None or m["task"] == task)
            and (include_todo or m["status"] == "present")]


def model_role_ok(name: str, role: str) -> bool:
    """name 是否允许充当 role（merger/validator 守 l5.* 编辑时用）。"""
    m = MODEL_METADATA.get(name)
    return bool(m and role in m["allowed_roles"])
