"""policy/seed_programs.py — P3 种子供给：skill bank v1（Final_Plan_CodeAgentFirst §P3，prereg §3）。

8 条手写组合 program，全部 grammar v1 合法且 **冻结 menu v1 不可表达**（resolved 身份 ∉ 15
动作池，tests 守）。menu 的结构性盲区 = imputer 恒为 impute_linear（period_complete /
impute_fft / impute_ema 从未上场）+ 无 winsor→median 组合。种子按此设计：

  forecast（6）：period/fft/ema 系插补 × {stl, median, ma, savgol} + winsor 复合两剂量
  anomaly （2）：period_complete / impute_fft 单步插补（anomaly 物理禁平滑/删改，插补即全部合法面）

用途（P3 出口件）：① headroom 测量的供给臂（vs menu+dose oracle，同判官同 split）；
② P5 identity gate 的 ε 校准输入；③ gym/composer 的检索候选（skill cards）。
本清单在此**冻结**（prereg §3 正式清单）；改动 = bank v2 新身份。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .program_edit import ProgramSpecV1, is_novel_v1, spec_v1_to_dict

BANK_VERSION = "bank_v1"


def _seed(name: str, steps: Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], ...],
          *, task: str = "forecast",
          guard: Tuple[Tuple[str, str, float], ...] = (),
          beta: float = 0.3, applicability: str = "", risk: str = "") -> ProgramSpecV1:
    return ProgramSpecV1(
        steps=steps, scope=("*",), task_type=task, pattern_guard=guard,
        risk_budget_beta=beta, fallback="v_impute_linear",
        provenance={"seed": name, "bank": BANK_VERSION,
                    "applicability": applicability, "risk": risk},
    )


SEED_PROGRAMS_V1: Dict[str, ProgramSpecV1] = {
    # ── forecast（6）─────────────────────────────────────────────────────────
    "seed_period_stl": _seed(
        "seed_period_stl",
        (("period_complete", ()), ("denoise_stl", ())),
        guard=(("seasonal_strength", ">=", 0.5),),
        applicability="强季节 + 块状缺失：周期感知补齐后 STL 去噪（guard 拦非季节结构）",
        risk="季节不稳时周期补齐会造伪周期（guard 承担拦截）",
    ),
    "seed_period_median9": _seed(
        "seed_period_median9",
        (("period_complete", ()), ("denoise_median", (("window", 9),))),
        applicability="季节缺失 + 中噪：周期补齐 + 轻 median",
        risk="窗≈周期时抹季节（w9 ≪ 24 安全）",
    ),
    "seed_fft_ma9": _seed(
        "seed_fft_ma9",
        (("impute_fft", ()), ("smooth_ma", (("window", 9),))),
        applicability="谱结构明显：FFT 补齐 + 移动平均",
        risk="非平稳段 FFT 补齐失真",
    ),
    "seed_winsor_median9": _seed(
        "seed_winsor_median9",
        (("impute_linear", ()), ("winsorize", ()), ("denoise_median", (("window", 9),))),
        applicability="离群 + 噪声并存（universal cleaner；menu 无 winsor→median 组合）",
        risk="anomaly 任务下毁检测目标（P2 动机表实测 −0.85 F1）",
    ),
    "seed_winsor_ma15": _seed(
        "seed_winsor_ma15",
        (("impute_linear", ()), ("winsorize", ()), ("smooth_ma", (("window", 15),))),
        applicability="离群 + 重噪：winsor 后中剂量 MA",
        risk="同上 + 中剂量对短 motif 有损",
    ),
    "seed_ema_savgol": _seed(
        "seed_ema_savgol",
        (("impute_ema", ()), ("denoise_savgol", ())),
        applicability="慢漂移缺失：EMA 前向补齐 + savgol 保形去噪",
        risk="EMA 滞后引入相位偏移",
    ),
    # ── anomaly_detection（2）───────────────────────────────────────────────
    "seed_period_only": _seed(
        "seed_period_only",
        (("period_complete", ()),),
        task="anomaly_detection",
        applicability="anomaly + 季节缺失：周期感知补齐（不动观测点，保 spike）",
        risk="周期估计错误时补入伪周期点（检测器容差 ±2 部分吸收）",
    ),
    "seed_fft_only": _seed(
        "seed_fft_only",
        (("impute_fft", ()),),
        task="anomaly_detection",
        applicability="anomaly + 谱结构缺失：FFT 补齐（保观测）",
        risk="谱泄漏使补齐点带振铃 → 可能新增假告警",
    ),
}


def seed_skill_cards() -> List[Dict[str, Any]]:
    """packet 兼容的 skill cards（composer/gym 的检索候选面）。"""
    cards: List[Dict[str, Any]] = []
    for name, spec in SEED_PROGRAMS_V1.items():
        cards.append({
            "name": name,
            "version": BANK_VERSION,
            "task_scope": spec.task_type,
            "program_spec": spec_v1_to_dict(spec),
            "applicability": spec.provenance.get("applicability", ""),
            "risk": spec.provenance.get("risk", ""),
            "sha": spec.sha(),
            "chain_sha": spec.chain_sha(),
        })
    return cards


def seed_bank_manifest() -> Dict[str, Dict[str, Any]]:
    """bank v1 台账（novelty 逐条核验 + 身份 SHA；P3 runner 落盘为 skill_bank_v1.json）。"""
    return {
        name: {
            "sha": spec.sha(),
            "chain_sha": spec.chain_sha(),
            "task_type": spec.task_type,
            "steps": [[op, dict(p)] for op, p in spec.steps],
            "pattern_guard": [list(g) for g in spec.pattern_guard],
            "novel_vs_menu_v1": is_novel_v1(spec),
        }
        for name, spec in SEED_PROGRAMS_V1.items()
    }
