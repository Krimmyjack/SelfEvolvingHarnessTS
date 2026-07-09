"""models/ — 下游模型库注册（三角色化：J 判官 / M_deploy 部署 / R 报告器）。

单一真源 = registry.py（镜像 operators/registry.py）。S1 只用 J/R（多已现成于 evaluators/）；
M_deploy（TSFM + 传统 GBDT）为 S4（L5 task-readiness）待建——本注册表先登记 family 占位，S4 直接挂。
"""
from .registry import MODEL_METADATA, MODEL_ROLES, get_models_for_role, model_role_ok

__all__ = ["MODEL_METADATA", "MODEL_ROLES", "get_models_for_role", "model_role_ok"]
