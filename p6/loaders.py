"""p6/loaders.py — P6 延迟 loader 载体类型（G 波 / codex 三轮复审 finding 32/35）。

正式运行的 V/U 数据加载不再接受"裸序列恰好也能跑"的静默路径。loader 返回值必须是本模块
的显式载体之一：

  - `BoundVEpisodes`  ——正式 V loader 载体：已 manifest-bound 验证的 episodes + 绑定的
                        materialization_sha。run_cycle 只需核对 sha == precommit 绑定值。
  - `BoundUEpisodes`  ——正式 U loader 载体：episodes + sealed materialization + 原始序列域
                        （run_u_eval 内部逐条 content_sha 复算 + uid/config/preset 核验）。
  - `UnboundEpisodes` ——**测试专用**裸 episodes 显式包装：formal 入口拒绝（→ P6TechnicalAbort），
                        unfrozen 入口接受。杜绝"传普通 list 恰好能跑"。

红线：stdlib + numpy + 只读 import P6Episode/SealedMaterialization；无 IO/网络/RNG。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from .c0_runner import P6Episode
from .materializer import SealedMaterialization

__all__ = [
    "BoundUEpisodes",
    "BoundVEpisodes",
    "UnboundEpisodes",
    "is_bound_carrier",
]


@dataclass(frozen=True)
class UnboundEpisodes:
    """裸 episodes 的**显式**包装（测试专用）：声明"未经 manifest 绑定"。

    formal 入口（run_cycle_formal / run_u_eval_formal）拒绝此类型；unfrozen 入口接受。
    存在的意义：使"传普通序列"不再是一条静默可跑路径——测试必须显式声明未绑定。"""

    episodes: Tuple[P6Episode, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "episodes", tuple(self.episodes))


@dataclass(frozen=True)
class BoundVEpisodes:
    """正式 V loader 载体：已 manifest-bound 验证的 episodes + 绑定 materialization_sha。

    loader 侧应已用 materializer.load_materialization_bound 完成验证；run_cycle 只核对
    materialization_sha == precommit 绑定值（不一致 → P6TechnicalAbort）。"""

    episodes: Tuple[P6Episode, ...]
    materialization_sha: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "episodes", tuple(self.episodes))
        object.__setattr__(self, "materialization_sha", str(self.materialization_sha))


@dataclass(frozen=True)
class BoundUEpisodes:
    """正式 U loader 载体：episodes + sealed materialization + 原始序列域（content_sha 复算域）。

    run_u_eval 在 open 之后逐条核验：uid(=series_uid) ∈ materialization、content_sha 复算一致、
    config/preset 一致、materialization.materialization_sha == open bindings 绑定值。
    series_by_uid 键 = materialization uid（f"{config}:{item_id}"）→ 原始（NaN 填充后、z-score
    前）序列，即 materialize 时 content_sha 的计算域。"""

    episodes: Tuple[P6Episode, ...]
    materialization: SealedMaterialization
    series_by_uid: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "episodes", tuple(self.episodes))
        object.__setattr__(self, "series_by_uid", dict(self.series_by_uid))


def is_bound_carrier(obj: Any) -> bool:
    """obj 是否为已绑定载体（BoundV/BoundU）。"""
    return isinstance(obj, (BoundVEpisodes, BoundUEpisodes))
