"""P1a-to-Action-Menu-v2 binding without changing the historical P1a declaration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..conditioning import p1a as p1a_module
from ..conditioning.p1a import P1A_C_FEATS, P1A_D_FEATS, P1A_P_FEATS, p1a_feats
from ..policy.action_spec import action_menu_v2
from ._canonical import file_sha256, sha256


@dataclass(frozen=True)
class VNextPatternBindingV1:
    binding_version: str
    extractor_sha: str
    action_menu_sha: str
    feature_names: tuple[str, ...]

    @classmethod
    def current(cls) -> "VNextPatternBindingV1":
        return cls(
            binding_version="vnext-p1a-menu-v2-v1",
            extractor_sha=file_sha256(Path(p1a_module.__file__).resolve()),
            action_menu_sha=action_menu_v2().sha256,
            feature_names=tuple(P1A_D_FEATS + P1A_P_FEATS + P1A_C_FEATS),
        )

    @property
    def sha256(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class PatternCard:
    p: tuple[tuple[str, float], ...]
    d: tuple[tuple[str, float], ...]
    c: tuple[tuple[str, float], ...]
    valid: bool
    summary: str
    source_sha: str

    @property
    def values(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.d + self.p + self.c))

    @property
    def sha256(self) -> str:
        return sha256(self)


def build_pattern_card(values: np.ndarray) -> PatternCard:
    raw = np.asarray(values, dtype=float).ravel()
    feats = p1a_feats(raw)
    binding = VNextPatternBindingV1.current()
    p = tuple((name, float(feats[name])) for name in P1A_P_FEATS)
    d = tuple((name, float(feats[name])) for name in P1A_D_FEATS)
    c = tuple((name, float(feats[name])) for name in P1A_C_FEATS)
    valid = bool(raw.size and np.isfinite(raw).sum() >= 4 and all(np.isfinite(v) for v in feats.values()))
    summary = (
        f"P1a valid={str(valid).lower()}; period={feats['period']:.6g}; "
        f"missing_rate={feats['missing_rate']:.6g}; SNR={feats['SNR']:.6g}; "
        f"outlier_density={feats['outlier_density']:.6g}; "
        f"coverage={feats['c_obs_coverage']:.6g}"
    )
    return PatternCard(p=p, d=d, c=c, valid=valid, summary=summary, source_sha=binding.sha256)
