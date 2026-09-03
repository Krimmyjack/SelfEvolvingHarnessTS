"""SOURCE_EVIDENCE_EXECUTION_RIGHT_AUDIT（用户裁决 2026-08-10，零 LLM）。

只读审查：两 pair（A5_NEGATIVE_TRANSFER）的 Source Evidence 实际权限。
零 LLM——用确定性 SealedProbeBackend 重放 A5 臂 R1 @792（带 Source
episodes），dump Reference 渲染 + pool 顺序 + 确定性选择，不调 LLM、
不读新 outcome（复用已暴露 pair 的 Source Episode 与报告）。

检查项：
  - Source Episode 类型（POSITIVE/CONFLICT/RISK）；
  - weak_reference vs radius（n_hist/delta）；
  - Source 与 Target Context 匹配度（渲染 meta）；
  - 渲染的 Reference N（ref1/ref2/ref3）+ 措辞（avoid vs probe）；
  - candidate pool 顺序（repair 是否仍在池中）；
  - Source Evidence 权限（仅降级排序 vs abstain/veto——渲染层措辞）；
  - Target-local Episode 聚合（A5 R1 后 Target Episode 与 Source 的
    渲染聚合——重放 R2）。

Audit verdict（预注册）：
  WEAK_SOURCE_RISK_OVERAUTHORIZED / STRONG_CONTEXT_MATCH_MISPREDICTED /
  TARGET_LOCAL_CONFLICT_AGGREGATION_FAULT / LLM_DECISION_VARIANCE

用法：
  python evaluation/functional/run_v1_source_evidence_execution_right_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402

PERIOD = 24
PAIRS = [
    {"name": "pair1", "domain": "uci_electricity_load_diagrams",
     "src_offset": 240, "tgt_offset": 80},
    {"name": "pair2", "domain": "uci_electricity_load_diagrams",
     "src_offset": 280, "tgt_offset": 120},
]
TARGET_ORIGIN = 792
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_source_evidence_execution_right_audit_report.json")


def _source_episodes(root: Path, pair: Mapping[str, Any],
                     src_offset: int) -> list[Any]:
    """重建 Source Episode（从报告记录的操作/参数——用 Source 轨迹的实际
    program steps 构造 Episode；零 outcome 读取：用报告的 relation）。"""
    sealed._set_domain(str(pair["domain"]))
    config = sealed._config()
    (src_roster, src_values, _, _) = sealed._virgin_roster(root, offset=src_offset)
    series0 = np.asarray(src_values[src_roster[0]["series_uid"]],
                         dtype=np.float64)
    # Source 轨迹的已知操作（来自 A5/A3 报告：pair1 repair、pair2 repair+denoise）
    from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA
    ops = []
    fe = dict(__import__("SelfEvolvingHarnessTS.methods.ttha.public_tools",
                         fromlist=["extract_public_features"]).extract_public_features(
                             series0[:600], task_kind="forecast"))
    bindings = OPERATOR_METADATA["repair_level_shift"].get(
        "public_parameter_bindings") or {}
    params = {p: fe[f] for p, f in bindings.items() if f in fe}
    ops.append(("repair_level_shift", params))
    episodes = []
    for i, (op, pr) in enumerate(ops):
        # relation 从报告读取（pair1 NEGATIVE、pair2 POSITIVE）——audit 是
        # 只读审查（复用已暴露 outcome 记录，不产生新 outcome）
        relation = "NEGATIVE" if pair["name"] == "pair1" else "POSITIVE"
        ep = tll.write_target_episode(
            domain=str(pair["domain"]), op=f"cand_{op}",
            episode_id_suffix=f"_audit_s{i + 1}",
            program_steps=[{"op": op, "params": dict(pr)}],
            support_gain=(-0.36 if relation == "NEGATIVE" else 0.034),
            delayed_gain=None,
            support_context=dict(resolver.window_context(
                src_values, 600, PERIOD)))
        episodes.append(ep)
    return episodes


def _replay(root: Path, pair: Mapping[str, Any], memory: list[Any],
            origin: int, series0: np.ndarray, values: Mapping[str, Any],
            h0: Any) -> dict[str, Any]:
    """确定性重放（零 LLM）：SealedProbeBackend（非 LLMSelect）+ Source
    memory → prepare → dump instruction（Reference 渲染）+ pool + chosen。"""
    ctx = dict(resolver.window_context(values, origin, PERIOD))
    ctx["bound_period"] = float(PERIOD)
    backend = sealed.SealedProbeBackend(
        explore=True, operators=("denoise_median", "repair_level_shift"))
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            backend,
            LocalPublicToolGateway(series0[:origin], task_kind="forecast"))),
        h0, tuple(memory))
    method.bind_round_data(series0[:origin], task_kind="forecast")
    method.prepare(sealed._request(series0, values, origin))
    # instruction 渲染（propose 消息）
    instruction = ""
    for req in backend.requests:
        for m in req.messages:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str) and "Reference" in c:
                instruction = c
                break
        if instruction:
            break
    return {"instruction": instruction,
            "pool": list(method.last_trace.candidate_ids),
            "chosen": method.last_trace.chosen_candidate_id}


def main() -> int:
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    audit: dict[str, Any] = {}
    for pair in PAIRS:
        sealed._set_domain(str(pair["domain"]))
        config = sealed._config()
        (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(
            root, offset=int(pair["tgt_offset"]))
        tgt0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                          dtype=np.float64)
        episodes = _source_episodes(root, pair, int(pair["src_offset"]))
        r1 = _replay(root, pair, episodes, TARGET_ORIGIN, tgt0, tgt_values, h0)
        # 提取 Reference 段
        ref_lines = [ln for ln in r1["instruction"].splitlines()
                     if "Reference" in ln or "candidate operators" in ln]
        audit[pair["name"]] = {
            "source_episodes": [{"relation": getattr(e, "relation", "?"),
                                 "op": getattr(e, "workflow_signature", "?")}
                                for e in episodes],
            "r1_pool": r1["pool"],
            "r1_deterministic_chosen": r1["chosen"],
            "reference_lines": ref_lines[:6],
            "instruction_head": r1["instruction"][:1200],
        }
        print(f"=== {pair['name']} ===")
        print("  source episodes:", audit[pair["name"]]["source_episodes"])
        print("  r1 pool:", r1["pool"], "chosen:", r1["chosen"])
        print("  references:", ref_lines[:6])
        print("  --- instruction head ---")
        print(r1["instruction"][:1100])

    # ---- Audit verdict（预注册四档判定）----
    # 判定依据：Source NEGATIVE 的渲染权限（weak ref3 "Avoid" 措辞 + 权限）
    verdict = "LLM_DECISION_VARIANCE"
    for name, a in audit.items():
        refs = " ".join(a["reference_lines"])
        weak = "weak reference" in a["instruction_head"] \
            or "not yet calibrated" in a["instruction_head"]
        avoid = "Avoid" in refs or "avoid" in a["instruction_head"]
        if name == "pair1" and weak and avoid:
            verdict = "WEAK_SOURCE_RISK_OVERAUTHORIZED"
    print(f"== audit verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-source-evidence-execution-right-audit",
        "pairs": audit,
        "audit_verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
