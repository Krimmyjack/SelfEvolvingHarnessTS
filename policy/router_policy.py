"""policy/router_policy.py — RouterPolicy 统一接口 + P0 冻结臂包装（Stage 2.0-③，v1.1b）。

接口：predict(conditioning_key, action_menu[, model_menu]) → RoutingDecision。

P0 实现 = FrozenArmRouterPolicy：包装 confirmatory 冻结产物 frozen_arms.joblib（默认
dp_abstain 臂，SHA 守卫①同源核验，**永不 fit**）。特征取自 conditioning_key 的
struct_feats，经 PatternSpec.router_features 映射到 (X_d, X_p)——与 e32_nested 训练侧
同一映射（同源性由 tests/test_router_policy.py 的逐 uid picks 等价性测试锚定）。

不做的事（有意）：
  - 不平行实现预测数学——predict 只调用冻结臂自己的 .picks（等价由构造保证）；
  - model_menu 暂 NotImplemented——模型条件化待 2.3 张量 pilot（C6：同动作跨模型可翻号）；
  - 不在此处扩展 EvidenceRecord——那是 2.0-⑤，routing provenance 暂由 RoutingDecision 携带。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..e32_policy import PolicyData
from .action_spec import ActionMenu
from .pattern_spec import PatternSpec, pattern_spec_p0


@dataclass
class RoutingDecision:
    action_id: str
    abstained: bool
    fallback_action: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    # provenance 至少含：router_artifact_sha / arm / pattern_spec {version, config_sha} /
    # action_menu {version, sha256}——四层等价性测试的第 (d) 层。


class RouterPolicy:
    """统一路由接口（2.0-③）。实现者只需覆写 predict。"""

    def predict(self, conditioning_key: Dict[str, Any], action_menu: ActionMenu,
                model_menu: Optional[List[str]] = None) -> RoutingDecision:
        raise NotImplementedError


class FrozenArmRouterPolicy(RouterPolicy):
    """冻结臂包装。arm = e32_policy 的已拟合 LookupArm/GBDTArm（来自 frozen_arms.joblib）。

    Step 1.1 收口（评审第二十三轮）：
      task_scope    冻结臂训练标签 = forecast nested loss → 其效用语义只对 forecast 成立；
                    非 forecast 输入**直接拒绝**（fail-loud），不做无语义的路由；
      版本核验      SHA 守卫护的是文件不是行为——加载时比对 blob 记录的 sklearn/numpy
                    版本与运行时，不匹配即 fail-loud（补 PolicyArtifact"可再生"的运行时半边）；
      OOD 标记      dev 训练支持集（z-score kNN）距离 > 训练自距离 p95 → provenance 标
                    out_of_support=True（**只记录不拦截**；拦截/abstain 是 2.2-⑥ C 通道的事）。
    """

    task_scope = ("forecast",)

    def __init__(self, arm, actions: List[str], arm_name: str, artifact_sha: str,
                 pattern_spec: Optional[PatternSpec] = None, support: Optional[dict] = None,
                 runtime_check: Optional[dict] = None):
        self.arm = arm
        self.actions = list(actions)              # 冻结臂的动作列序（决定 pick 索引语义）
        self.arm_name = arm_name
        self.artifact_sha = artifact_sha
        self.pattern_spec = pattern_spec or pattern_spec_p0()
        self.support = support                    # {"mu","sd","Z","threshold",...} 或 None（不可用）
        # 版本核验结果留存进每条决策 provenance（评审第二十四轮：allow_version_mismatch
        # 放行若不落 provenance，"显式放行须记录"就是空话）
        self.runtime_check = runtime_check or {"checked": False}

    # ── 构造 ──────────────────────────────────────────────────────────────
    @classmethod
    def load_frozen(cls, arm_name: str = "dp_abstain", path: Optional[Path] = None,
                    verify_sha: str = "auto", pattern_spec: Optional[PatternSpec] = None,
                    allow_version_mismatch: bool = False) -> "FrozenArmRouterPolicy":
        """load confirmatory 冻结臂（守卫①：confirmatory_freeze.json 在则按其 SHA 核验；
        Step 1.1-④：运行时 sklearn/numpy 版本与 blob 记录不一致 → fail-loud，显式放行须
        allow_version_mismatch=True 且自担 joblib 反序列化行为漂移风险）。"""
        import hashlib
        from ..confirmatory_freeze import FREEZE_PATH, FROZEN_ARMS, load_frozen_arms
        p = Path(path) if path is not None else FROZEN_ARMS
        sha = None
        if verify_sha == "auto":
            if FREEZE_PATH.exists() and p == FROZEN_ARMS:
                import json
                sha = json.loads(FREEZE_PATH.read_text("utf-8"))["router"]["sha256"]
        elif verify_sha:
            sha = verify_sha
        blob = load_frozen_arms(p, verify_sha=sha)
        rt_check = cls._check_runtime_versions(blob, allow_version_mismatch)
        actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if arm_name not in blob["arms"]:
            raise KeyError(f"frozen_arms 无臂 {arm_name!r}；可用：{sorted(blob['arms'])}")
        return cls(blob["arms"][arm_name], list(blob["actions"]), arm_name, actual_sha,
                   pattern_spec=pattern_spec, support=cls._load_support(),
                   runtime_check=rt_check)

    @staticmethod
    def _check_runtime_versions(blob: dict, allow: bool) -> dict:
        """比对 blob 记录版本与运行时；不匹配且未放行 → fail-loud。返回核验结果 dict
        （进 provenance——放行了什么、在什么版本差异下放行，必须可审计）。"""
        import sklearn
        recorded = {"sklearn": blob.get("sklearn_version"), "numpy": blob.get("numpy_version")}
        runtime = {"sklearn": sklearn.__version__, "numpy": np.__version__}
        mismatch = {k: [recorded[k], runtime[k]] for k in recorded
                    if recorded[k] is not None and recorded[k] != runtime[k]}
        if mismatch and not allow:
            raise RuntimeError(
                f"frozen_arms 运行时版本不匹配（记录 vs 运行时）：{mismatch}——SHA 只护文件不护行为；"
                "确认行为等价后可 allow_version_mismatch=True 显式放行")
        return {"checked": True, "recorded": recorded, "runtime": runtime,
                "mismatch": mismatch, "allowed_mismatch": bool(allow and mismatch)}

    @staticmethod
    def _load_support() -> Optional[dict]:
        """dev 训练支持集（best-effort：records 缺失 → None，provenance 报 unavailable）。
        阈值 = 训练点 leave-one-out 最近邻距离的 p95（z-score 特征空间）。"""
        try:
            from ..confirmatory_freeze import load_dev_records
            recs = load_dev_records("primary_no_Sar")
        except Exception:
            return None
        Z_raw = np.array([[r["snr"], r["miss_rate"], *r["X_p"]] for r in recs], dtype=float)
        mu = Z_raw.mean(axis=0)
        sd = Z_raw.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Z = (Z_raw - mu) / sd
        d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.sqrt(d2.min(axis=1))                      # leave-one-out 最近邻距离
        return {"mu": mu, "sd": sd, "Z": Z, "threshold": float(np.percentile(nn, 95)),
                "n_train": int(Z.shape[0]),
                # 挂账（评审第二十四轮）：支持集当前为加载时从 dev records 动态重建，
                # 未绑定进 PolicyArtifact——S1 流/真实部署前必须随 artifact 一起冻结分发。
                "source": "confirmatory dev records primary_no_Sar（加载时重建，未绑定 artifact）"}

    def _support_check(self, xd: np.ndarray, xp: np.ndarray) -> Dict[str, Any]:
        if self.support is None:
            return {"available": False}
        s = self.support
        z = (np.concatenate([xd, xp]) - s["mu"]) / s["sd"]
        dist = float(np.sqrt(((s["Z"] - z) ** 2).sum(axis=1).min()))
        return {"available": True, "distance": dist, "threshold": s["threshold"],
                "out_of_support": bool(dist > s["threshold"]),
                "n_train": s.get("n_train"), "source": s.get("source")}

    # ── 预测 ──────────────────────────────────────────────────────────────
    def predict(self, conditioning_key: Dict[str, Any], action_menu: ActionMenu,
                model_menu: Optional[List[str]] = None) -> RoutingDecision:
        if model_menu is not None:
            raise NotImplementedError("model_menu 待 2.3 张量 pilot（C6：模型条件化未确立）")
        task = (conditioning_key.get("task") or {}).get("type")
        if task not in self.task_scope:
            raise ValueError(
                f"FrozenArmRouterPolicy 只对 task_scope={self.task_scope} 有效（冻结臂标签="
                f"forecast nested loss），拒绝 task={task!r}——无语义路由比不路由更危险")
        self._check_menu(action_menu)

        struct = conditioning_key["pattern"]["struct_feats"]
        xd, xp = self.pattern_spec.router_features(struct)
        cell = str(conditioning_key.get("cell_id") or "")
        from ..e32_policy import FALLBACK_ACTION
        shim = PolicyData(
            uids=["deploy:0"], actions=self.actions,
            L=np.zeros((1, len(self.actions))),          # picks 不读 L；占位满足契约
            X_d=xd.reshape(1, -1), X_p=xp.reshape(1, -1),
            cell=np.array([cell]), origin=np.array(["deploy"]))
        pick_idx, abstained = self.arm.picks(shim, np.arange(1))
        aid = self.actions[int(pick_idx[0])]
        return RoutingDecision(
            action_id=aid,
            abstained=bool(abstained[0]),
            fallback_action=FALLBACK_ACTION,
            provenance={
                "router_artifact_sha": self.artifact_sha,
                "arm": self.arm_name,
                "pattern_spec": {"version": self.pattern_spec.version,
                                 "config_sha": self.pattern_spec.config_sha(),
                                 "code_sha256": self.pattern_spec.code_sha256},  # 提取器闭包活值
                "action_menu": {"version": action_menu.version, "sha256": action_menu.sha256},
                "support": self._support_check(xd, xp),   # 只记录不拦截（2.2-⑥ 前的廉价哨兵）
                "runtime_check": self.runtime_check,      # 版本核验/放行审计（Step 1.1-④ 补全）
            })

    def _check_menu(self, menu: ActionMenu):
        """兼容性双核验：冻结臂动作集 ⊆ menu；menu 版本在 PatternSpec 兼容清单内。"""
        missing = [a for a in self.actions if a not in menu]
        if missing:
            raise ValueError(f"ActionMenu {menu.version} 缺冻结臂动作：{missing}"
                             "——Router 选得出、菜单编译不出 = 执行契约破裂")
        if menu.version not in self.pattern_spec.compatible_action_menus:
            raise ValueError(f"ActionMenu {menu.version} 不在 PatternSpec "
                             f"{self.pattern_spec.version} 兼容清单 "
                             f"{self.pattern_spec.compatible_action_menus} 内")
