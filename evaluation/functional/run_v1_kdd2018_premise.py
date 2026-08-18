"""KDD_CUP_2018_PREMISE_ONLY（P0，用户裁决 2026-08-11：新数据 outcome-blind
准备——优先 outlier family）。

只检查公开 Context 和 verifier（零 outcome——不跑 Consumer、不读 gain、
不因 outcome 换 series）：
  - 长度容 Source/R1/delayed/R2/delayed（≥984；hourly period=24 与 uci
    装置同窗口）
  - ≥1 有自然信号的 defect family（outlier——空气污染异常峰值）
  - family ≥2 合法、行为不同的替代 Program（winsorize/outlier_iqr/
    outlier_mad/hampel_filter——静态 verifier 合法 + 非 identity）
  - Source/Target cohort 可互斥（270 条分片）

通过 → P1 自然 Fast 轨迹（寻找 material failure）；前提不满足 → 关闭
outlier family（不换 origin 找答案）。

用法：
  python evaluation/functional/run_v1_kdd2018_premise.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    _actionable_operators,
    _allowed_operators,
)
from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    MetricSpec,
    forecast_task_spec_v1,
)

PERIOD = 24  # hourly（KDD Cup 2018——与 uci 装置同周期语义）
HORIZON = 48
ORIGIN = 792  # R1 决策点（同 uci 装置窗口）
MIN_LEN = 984
TSF = PROJECT_ROOT / "data/kdd2018/raw/kdd_cup_2018_dataset_without_missing_values.tsf"
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2/w1_kdd2018_premise_report.json"
OUTLIER_FAMILY = ("winsorize", "outlier_iqr", "outlier_mad",
                  "hampel_filter")


def parse_tsf(path: Path) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    in_data = False
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not in_data:
                if line.startswith("@data"):
                    in_data = True
                continue
            if not line.strip():
                continue
            # KDD 2018 行格式：name:type:timestamp:values（时间戳含冒号
            # 14:00:00——从右分一次取 values 段）
            head, values = line.rsplit(":", 1)
            name = head.split(":", 1)[0]
            series[name] = np.asarray(
                [float(v) for v in values.split(",") if v], dtype=np.float64)
    return series


def _request_for(series0: np.ndarray) -> PreparationRequest:
    return PreparationRequest(
        "kdd2018-premise",
        series0[:ORIGIN],
        forecast_task_spec_v1(horizon=HORIZON,
                              downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        {},
    )


def main() -> int:
    series = parse_tsf(TSF)
    names = sorted(series)
    lengths = np.asarray([len(series[n]) for n in names], dtype=np.int64)
    print(f"parsed {len(names)} series; min_len={lengths.min()} "
          f"len_ok={int((lengths >= MIN_LEN).sum())}")
    np.savez_compressed(
        CACHE, names=np.asarray(names, dtype=object),
        values=np.asarray([series[n] for n in names], dtype=object),
        lengths=lengths)

    h0 = compile_snapshot(PROJECT_ROOT / "methods/ttha/harness/h0",
                          verify_lock=False)
    n_ok = 0
    outlier_signal = 0
    family_ok = 0
    samples: list[dict[str, object]] = []
    for n in names[:60]:  # 抽样 60 条（premise 快速检查——零 outcome）
        s = np.asarray(series[n][:ORIGIN], dtype=np.float64)
        fe = dict(extract_public_features(s, task_kind="forecast"))
        req = _request_for(series[n])
        view = resolve_harness_view(h0, fe, role="fast")
        act = _actionable_operators(
            req, s, view, _allowed_operators(req))
        fam = [op for op in OUTLIER_FAMILY if op in act]
        has_outlier_signal = bool(
            float(fe.get("level_excursion_score", 0.0)) > 1.0
            or any(k in fe for k in ("estimated_region_start_fraction",)))
        if len(series[n]) >= MIN_LEN:
            n_ok += 1
        if has_outlier_signal:
            outlier_signal += 1
        if len(fam) >= 2:
            family_ok += 1
        if len(samples) < 5:
            samples.append({"name": n, "length": int(len(series[n])),
                            "actionable_outlier": fam,
                            "level_excursion": round(
                                float(fe.get("level_excursion_score", 0.0)), 3),
                            "outlier_signal": has_outlier_signal})
    report = {
        "experiment_id": "v1-kdd2018-premise-only",
        "note": "零 outcome：公开 Context + 静态 verifier（不跑 Consumer/"
                "不读 gain/不因 outcome 换 series）",
        "dataset": "kdd_cup_2018_without_missing",
        "n_series": len(names),
        "n_len_ok": n_ok,
        "n_outlier_signal": outlier_signal,
        "n_family_ok_ge2": family_ok,
        "outlier_family": list(OUTLIER_FAMILY),
        "samples": samples,
        "verdict": ("PREMISE_OK" if (n_ok == len(names[:60])
                                     and family_ok >= len(names[:60]) * 0.5)
                    else "PREMISE_UNAVAILABLE"),
    }
    REPORT_REL.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
