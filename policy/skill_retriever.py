"""Deterministic SkillSpec retrieval for the Skill+Memory evidence surface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..e32_policy import P_FEATS
from .action_spec import ActionMenu
from .skills import SKILLS_V1, SkillSpec


@dataclass(frozen=True)
class SkillMatch:
    skill: SkillSpec
    score: float
    reasons: tuple[str, ...]
    allowed_actions: list[str]


def _allowed_action_ids(action_menu: Any) -> set[str] | None:
    if action_menu is None:
        return None
    if isinstance(action_menu, ActionMenu):
        return set(action_menu.actions)
    if isinstance(action_menu, Mapping):
        values = action_menu.get("allowed_actions")
        if values is not None:
            return {str(v) for v in values}
        actions = action_menu.get("actions")
        if isinstance(actions, Mapping):
            return {str(k) for k in actions}
        if actions is not None:
            return {str(v) for v in actions}
    return None


def _features(record_or_pattern: Mapping[str, Any]) -> dict[str, float]:
    if "pattern" in record_or_pattern and isinstance(record_or_pattern.get("pattern"), Mapping):
        pattern = record_or_pattern["pattern"]
        struct = dict(pattern.get("struct_feats") or {})
        out = {
            "snr": float(pattern.get("snr", struct.get("SNR", struct.get("snr", 0.0)))),
            "miss_rate": float(pattern.get("missing_rate", struct.get("missing_rate", 0.0))),
        }
        out.update({str(k): float(v) for k, v in struct.items() if _is_number(v)})
        return out
    x_p = list(record_or_pattern.get("X_p") or [])
    out = {
        "snr": float(record_or_pattern.get("snr", record_or_pattern.get("SNR", 0.0))),
        "miss_rate": float(record_or_pattern.get("miss_rate", record_or_pattern.get("missing_rate", 0.0))),
    }
    if len(x_p) == len(P_FEATS):
        out.update({name: float(value) for name, value in zip(P_FEATS, x_p)})
    return out


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _skill_score(skill_name: str, feats: Mapping[str, float]) -> tuple[float, tuple[str, ...]]:
    snr = float(feats.get("snr", feats.get("SNR", 0.0)))
    miss = float(feats.get("miss_rate", feats.get("missing_rate", 0.0)))
    seasonal = float(feats.get("seasonal_strength", 0.0))
    trend = float(feats.get("trend_strength", 0.0))
    outlier = float(feats.get("outlier_density", 0.0))
    lumpiness = float(feats.get("lumpiness", 0.0))
    entropy = float(feats.get("spectral_entropy", 0.0))
    period = float(feats.get("period", 0.0))

    reasons: list[str] = []
    score = 0.1
    low_snr = snr < 2.0
    very_low_snr = snr < -2.0
    has_missing = miss > 0.01
    has_outliers = outlier > 0.04 or lumpiness > 0.6
    has_season = seasonal >= 0.3 and period >= 3.0

    if skill_name == "identity":
        score = 0.2
        if snr >= 8.0:
            score += 0.35
            reasons.append("high_snr")
        if miss <= 0.01:
            score += 0.25
            reasons.append("little_missingness")
        if outlier <= 0.02 and lumpiness <= 0.4:
            score += 0.1
            reasons.append("low_outlier_signal")
        if low_snr or has_missing:
            score -= 0.3
            reasons.append("raw_is_weak_under_noise_or_missingness")
    elif skill_name == "median_smooth":
        score = 0.25
        if low_snr:
            score += 0.3
            reasons.append("low_snr")
        if very_low_snr:
            score += 0.15
            reasons.append("very_low_snr")
        if has_missing:
            score += 0.2
            reasons.append("missingness_present")
        if has_outliers:
            score += 0.15
            reasons.append("outlier_or_lumpy")
        if has_season:
            score -= 0.1
            reasons.append("seasonal_shape_risk")
    elif skill_name == "winsorize":
        score = 0.18
        if has_outliers:
            score += 0.45
            reasons.append("outlier_or_lumpy")
        if has_missing:
            score += 0.05
            reasons.append("missingness_present")
    elif skill_name == "winsor_savgol":
        score = 0.12
        if has_outliers and low_snr:
            score += 0.35
            reasons.append("outlier_plus_noise")
        if has_season:
            score -= 0.15
            reasons.append("shape_change_risk")
    elif skill_name == "stl_deseason":
        score = 0.12
        if has_season:
            score += 0.4
            reasons.append("seasonality_evidence")
        if entropy > 0.75:
            score -= 0.1
            reasons.append("weak_period_reliability")
    elif skill_name == "wavelet_denoise":
        score = 0.15
        if low_snr and entropy >= 0.4:
            score += 0.25
            reasons.append("multi_scale_noise")
        if trend >= 0.4:
            score += 0.1
            reasons.append("nonstationary_structure")
    elif skill_name == "savgol_smooth":
        score = 0.14
        if low_snr and not has_outliers:
            score += 0.2
            reasons.append("smooth_noise_without_outlier_dominance")
        if has_season:
            score -= 0.1
            reasons.append("shape_change_risk")
    else:
        reasons.append("registry_default")
    if not reasons:
        reasons.append("weak_match")
    return max(0.0, round(score, 6)), tuple(reasons)


def retrieve_skills(
    record_or_pattern: Mapping[str, Any],
    *,
    skills: Mapping[str, SkillSpec] | Sequence[SkillSpec] = SKILLS_V1,
    action_menu: Any = None,
    top_k: int = 5,
) -> list[SkillMatch]:
    """Return deterministic top-k SkillSpec matches for a DataView/Pattern record."""

    skill_values = list(skills.values()) if isinstance(skills, Mapping) else list(skills)
    allowed = _allowed_action_ids(action_menu)
    feats = _features(record_or_pattern)
    matches: list[SkillMatch] = []
    for skill in skill_values:
        actions = [aid for aid in skill.actions.values() if allowed is None or aid in allowed]
        if not actions:
            continue
        score, reasons = _skill_score(skill.name, feats)
        matches.append(SkillMatch(skill=skill, score=score, reasons=reasons, allowed_actions=list(actions)))
    matches.sort(key=lambda m: (-m.score, m.skill.name))
    return matches[: max(0, int(top_k))]


def retrieve_skill_cards(
    record_or_pattern: Mapping[str, Any],
    *,
    skills: Mapping[str, SkillSpec] | Sequence[SkillSpec] = SKILLS_V1,
    action_menu: Any = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return packet-ready skill cards with retrieval scores and reasons."""

    cards = []
    for rank, match in enumerate(
        retrieve_skills(record_or_pattern, skills=skills, action_menu=action_menu, top_k=top_k),
        start=1,
    ):
        cards.append(
            {
                "rank": rank,
                "name": match.skill.name,
                "score": match.score,
                "reasons": list(match.reasons),
                "allowed_actions": list(match.allowed_actions),
                "actions": dict(match.skill.actions),
                "applicability": match.skill.applicability,
                "risk": match.skill.risk,
                "fallback": match.skill.fallback,
                "version": match.skill.version,
            }
        )
    return cards