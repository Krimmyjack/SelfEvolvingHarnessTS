"""operators/_provenance.py — 算子身份台账（S0.7 Operator Integrity Gate）。

目的：**"系统声称执行的算子 == 实际执行的算子"**。算子内部广泛用 `except: return other_op(...)`
静默回退 → 声明动作与实际动作不一致 → router 学到错误动作标签 → 所有 Pattern 结论失基。

机制（轻量，进程内）：算子在关键分支调 `record(requested, effective, reason)`。开启记录后，实验/测试
可读台账验证：①无声静默 masquerade（fallback 必留因）；②requested==effective 或 fallback 可见。
另配 `dependency_fingerprint()`（numpy/scipy/statsmodels/pywt 版本）落进产物 provenance。
"""
from __future__ import annotations

from typing import Dict, List

_LEDGER: List[dict] = []
_RECORDING = False


def start_recording() -> None:
    global _RECORDING
    _RECORDING = True
    _LEDGER.clear()


def stop_recording() -> None:
    global _RECORDING
    _RECORDING = False


def record(requested: str, effective: str, reason: str = "") -> None:
    """算子在执行/回退点调用。requested=声明算子，effective=实际执行算子，reason=回退原因（空=按声明执行）。"""
    if _RECORDING:
        _LEDGER.append({"requested": requested, "effective": effective, "reason": reason})


def get_ledger() -> List[dict]:
    return list(_LEDGER)


def fallback_summary() -> Dict[str, Dict[str, int]]:
    """{requested_op: {effective_op: count}}——快速看哪个算子多少次落到别的算子。"""
    out: Dict[str, Dict[str, int]] = {}
    for e in _LEDGER:
        out.setdefault(e["requested"], {}).setdefault(e["effective"], 0)
        out[e["requested"]][e["effective"]] += 1
    return out


def dependency_fingerprint() -> Dict[str, str]:
    """算子实际行为依赖的库版本（落进实验 provenance；wavelet==savgol 就是版本/执行路径不一致的教训）。"""
    fp: Dict[str, str] = {}
    for mod in ("numpy", "scipy", "statsmodels", "pywt", "sklearn"):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            fp[mod] = "MISSING"
    return fp
