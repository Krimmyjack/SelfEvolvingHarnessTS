"""slow_path/candidate_log.py — S0.5：候选级 JSONL 日志器（E-6.1/E-4.2/E-7.3 的地基）。

现状（2026-07-02 核实）：`ev.history` 仅内存、run 脚本只 print；accepted 进 patch_log，**rejected 只剩
reason 字符串**。E-6.1 模式 B 的反事实重放、模式 A 的重打分、E-4.2 拒绝解剖全需要**逐候选完整记录**。

本日志器在 `Evolver.evolve_cell` 的 validate 调用后挂钩（validator 本身不动），对**每个候选（含被拒）**
落一行 JSONL，字段按统一分析语言 **U(a|x,d,t,m)** 组织（A-9，仅日志/分析层命名，非 framing pivot）：
  x/d ≈ cell 指纹（含退化）  a ≈ patch   t ≈ task   m ≈ judge/reporter 指纹   U 的样本 ≈ 四个 v 值与 Δ

评估明细（A-11/A-22）：cell 内 split（held_in/held_out_a）存 per-series `series_uid` 成员 + 逐序列指纹；
held_out(b) 存 per-cell 成员 + 指纹（不存 per-series loss 大向量，但 artifact_key 可确定性恢复成员/
harness/judge → LCB 的 ∀g worst-group 重评可得）。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


def _sha8(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:8]


def _sample_hash(samples: Sequence) -> str:
    """一批样本的确定性指纹（raw 值 + uid）→ 8 hex；用于校验重切/重放取到同一批。"""
    h = hashlib.sha256()
    for s in samples:
        h.update(str(getattr(s, "series_uid", "")).encode("utf-8"))
        arr = getattr(s, "raw", None)
        if arr is not None:
            try:
                h.update(bytes(memoryview(arr.astype("float64").tobytes())))
            except Exception:
                h.update(repr(arr).encode("utf-8"))
    return h.hexdigest()[:8]


def _split_fp(samples: Sequence, with_uids: bool = True) -> Dict[str, Any]:
    fp: Dict[str, Any] = {
        "n": len(samples),
        "origins": dict(Counter(getattr(s, "origin", "") for s in samples)),
        "sample_hash": _sample_hash(samples),
    }
    if with_uids:
        fp["series_uids"] = [getattr(s, "series_uid", "") for s in samples]
    return fp


def judge_fingerprint(harness) -> str:
    """m ≈ judge：grounded evaluator 配置指纹（model/metric/epsilon per task）。"""
    try:
        specs = harness.l4.grounded_evaluators
        parts = [f"{t}:{s.model}:{s.metric}:{s.epsilon}" for t, s in sorted(specs.items())]
        return _sha8("|".join(parts))
    except Exception:
        return ""


class CandidateLogger:
    """逐候选 append-only JSONL 记录器。绝不参与裁决，只观测。"""

    def __init__(self, path: str, run_id: Optional[str] = None):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run_{int(time.time())}"
        self.n_logged = 0

    def log(self, cell_id: str, epoch: int, round_idx: int, harness, patch,
            outcome, splits, *, task: str = "") -> None:
        held_in, held_out_a, held_out_b = splits
        patch_dict = patch.to_dict() if hasattr(patch, "to_dict") else dict(patch)
        d = outcome.deltas() if hasattr(outcome, "deltas") else {}
        jfp = judge_fingerprint(harness)
        # artifact_key：确定性重评锚（split 成员 + harness 序列化 + judge 指纹 + patch path）→ 可复现重评
        member_hash = _sha8(_split_fp(held_in)["sample_hash"], _split_fp(held_out_a)["sample_hash"],
                            *[_sample_hash(s) for s in held_out_b.values()])
        artifact_key = _sha8(member_hash, str(harness.version), jfp, patch_dict.get("path", ""))
        rec = {
            "run_id": self.run_id,
            "ts": time.time(),
            # —— x/d：cell 指纹（含退化坐标） ——
            "cell_id": cell_id,
            # —— t：task ——
            "task": task or cell_id.split("|", 1)[0],
            "epoch": epoch,
            "round_idx": round_idx,
            "harness_version": harness.version,
            # —— a：动作（完整 patch，accepted 与 rejected 都落） ——
            "patch": patch_dict,
            # —— U 的样本：四个 v 值 + Δ + 裁决 ——
            "outcome": {
                "accept": bool(getattr(outcome, "accept", False)),
                "reason": getattr(outcome, "reason", ""),
                "resolved_scope": getattr(outcome, "resolved_scope", None),
                "v_in_cur": _f(getattr(outcome, "val_in_cur", float("nan"))),
                "v_in_cand": _f(getattr(outcome, "val_in_cand", float("nan"))),
                "v_a_cur": _f(getattr(outcome, "val_a_cur", float("nan"))),
                "v_a_cand": _f(getattr(outcome, "val_a_cand", float("nan"))),
                "pareto_safe": bool(getattr(outcome, "pareto_safe", True)),
                "pareto_violator": getattr(outcome, "pareto_violator", ""),
                "delta_in": _f(d.get("held_in", float("nan"))),
                "delta_a": _f(d.get("held_out_a", float("nan"))),
            },
            # —— split 指纹（成员 uid + origin 计数 + 样本 hash 前 8）——
            "split_fingerprint": {
                "held_in": _split_fp(held_in),
                "held_out_a": _split_fp(held_out_a),
                "held_out_b": {c: _split_fp(s) for c, s in held_out_b.items()},
            },
            # —— m：judge 指纹 + 确定性重评锚 ——
            "judge_fingerprint": jfp,
            "artifact_key": artifact_key,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n_logged += 1


def _f(x) -> Optional[float]:
    try:
        xf = float(x)
        return xf if xf == xf else None      # NaN → null（JSON 合法）
    except Exception:
        return None
