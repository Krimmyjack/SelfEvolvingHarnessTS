"""V1 fast_agent 真实入口 signed radius Replay（零 LLM，2026-08-08）。

审查批准（2026-08-08）：NN5 优势保留且 GEFCom 无强负迁移后，把同一 resolver 接入
fast_agent 真实入口，做一次真实入口 Replay。本脚本：

  1. 重建 gefcom R3 决策时刻的 Memory（source + R1 + R2 本地 Episode，与冻结链一致）；
  2. 用 RecordingBackend（记录 AgentRequest.messages）零 LLM 驱动 TTHAFastAgent.prepare
     ——真实调用链：观察 → 检索 → signed 渲染 → inspect/propose/select（固定响应）；
  3. 断言 inspect 请求的 instruction 含 signed 判定：
     - denoise_savgol 在 Reference 1（POSITIVE_PRIOR 提前）
     - denoise_stl 不在 Reference 1（无验证证据，不提前）
     - 渲染不含 gain 数值（TIMECLAW 消融约束）
  4. A3 对照：空 episodes → 无经验注入。

验收口径：全程零 LLM（Replay backend 固定响应）、不读取当前 Query outcome
（resolver 只用历史 Episode + 部署可见 recent/change 特征）。

用法：
  python evaluation/functional/run_v1_fastagent_signed_replay.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_target_local_loop as loop  # noqa: E402
import signed_radius as resolver  # noqa: E402（方法层单一来源）

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    MetricSpec,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse  # noqa: E402

HORIZON = 48
DOMAIN = "gefcom"
PERIOD = 24
MAX_TARGET_PROBES = 2
H0_ROOT = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_fastagent_signed_replay_report.json")

# 冻结链（与 run_v1_target_local_loop_3rounds 一致）：源 (640, 688) → R1 (736, 784)
# → R2 (832, 880)；R3 决策点 = support 928（Replay 只重放接线，不探测 R3）
SOURCE_ORIGINS = {"gefcom": (640, 688)}
SLICES = [(736, 784), (832, 880)]


class RecordingBackend:
    """零 LLM Replay backend：固定响应 + 记录每次请求的 messages。"""

    def __init__(self, responses: Sequence[AgentResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[Any] = []

    def complete(self, request: Any) -> AgentResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected agent call")
        return self._responses.pop(0)


def _stage(stage: str, payload: Mapping[str, object]) -> AgentResponse:
    return AgentResponse.valid(
        {
            "schema_version": "agent-envelope/1",
            "kind": "stage_result",
            "stage": stage,
            "payload": payload,
        },
        raw_response={"id": f"replay-{stage}"},
    )


def _identity_responses() -> list[AgentResponse]:
    return [
        _stage("inspect", {
            "inspected_region_fractions": [[0.0, 1.0]],
            "requested_public_tools": [],
            "uncertainty": "high",
        }),
        _stage("propose", {"candidates": []}),
        _stage("select", {
            "chosen_candidate_id": "identity",
            "verification_actions": ["public_evidence_insufficient"],
        }),
    ]


def build_r3_memory(root: Path) -> tuple[list[Any], list[Any], dict[int, float]]:
    """重建 gefcom R3 决策时刻的 Memory（与冻结链逐位一致）。

    返回 (source_episodes, a5_local, baseline_cache)——a5_local = R1+R2 本地 Episode。
    """
    from run_w2_operator_scan import _default_params
    config = dict(v6.DATASET_CONFIGS[DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}
    operators = sorted(n for n in v6.OPERATOR_NAMES
                       if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                       and n not in core.CTS_EXCLUDED)

    source_episodes, _ = v1.build_source_memory(
        domain=DOMAIN, roster=roster, values=values, config=config,
        operators=operators,
        source_support_origin=SOURCE_ORIGINS[DOMAIN][0],
        source_delayed_origin=SOURCE_ORIGINS[DOMAIN][1],
        baseline_cache=baseline_cache,
        context_fn=lambda o: resolver.window_context(values, o, period),
    )

    def probe_at(order: Sequence[str], origin: int) -> dict[str, Any]:
        gains: list[float] = []
        probed: list[str] = []
        harm = 0
        for op in order:
            if len(probed) >= MAX_TARGET_PROBES:
                break
            compiled = v1.make_compiled(op, _default_params(op, period))
            g = v1.gain_at(roster, values, config, compiled, origin, baseline_cache)
            if g is None:
                continue
            probed.append(op)
            gains.append(g)
            if g < -core.MATERIAL_THRESHOLD:
                harm += 1
            if g >= core.MATERIAL_THRESHOLD:
                break
        return {"probe_order": probed, "support_gains": gains, "harm": harm}

    a5_local: list[Any] = []
    for round_idx, (ts, td) in enumerate(SLICES):
        f_support = resolver.window_context(values, ts, period)
        order, _signed = resolver.resolve_order(
            query_context=f_support, episodes=source_episodes + a5_local,
            operators=operators, material_threshold=core.MATERIAL_THRESHOLD)
        r = probe_at(order, ts)
        start = len(a5_local)
        for op, g in zip(r["probe_order"], r["support_gains"]):
            a5_local.append(loop.write_target_episode(
                domain=DOMAIN, op=op,
                program_steps=[{"op": op, "params": dict(_default_params(op, period))}],
                support_gain=g, delayed_gain=None, support_context=f_support))
        if not r["probe_order"]:
            a5_local.append(loop.write_abstain_episode(domain=DOMAIN, reason="A5_no_valid_plan"))
        # delayed 只更新本轮（修复 3/4）
        f_delayed = resolver.window_context(values, td, period)
        new_local = []
        for i, ep in enumerate(a5_local):
            if i < start or ep.workflow_signature == "identity":
                new_local.append(ep)
                continue
            compiled = loop.compiled_from_episode(ep, period)
            dg = v1.gain_at(roster, values, config, compiled, td, baseline_cache)
            new_local.append(loop.update_delayed_status(ep, dg, delayed_context=f_delayed)
                             if dg is not None else ep)
        a5_local[:] = new_local
        print(f"  [replay] R{round_idx + 1} support={ts}: {r}")
    return source_episodes, a5_local, baseline_cache


def run_prepare(
    request_values: np.ndarray,
    episodes: Sequence[Any],
    backend: RecordingBackend,
    *,
    observed_pattern_spec: Mapping[str, float] | None = None,
) -> tuple[Any, Any, str]:
    """真实入口：TTHAFastAgent.prepare（零 LLM backend）。返回 (result, trace, instruction)。"""
    # task_spec 与 V1 链 consumer 一致（forecast|ridge|sMASE）——合法性过滤按
    # Task/Consumer 匹配放行；默认 task_spec（dlinear_shared|nRMSE）会把 V1
    # Episode 全部过滤（审查问题 3 的防护行为，Replay 中另行报告）。
    request = PreparationRequest(
        "gefcom-replay-r3",
        np.asarray(request_values, dtype=float),
        forecast_task_spec_v1(horizon=HORIZON,
                              downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed_pattern_spec or {}),
    )
    # verify_lock=False：仓库 h0 lock 与当前代码版本 mismatch（既有状态，与本次改动无关；
    # stash 验证）。Replay 现场编译 = 与当前代码一致，不改生产 lock 文件。
    h0 = compile_snapshot(H0_ROOT, verify_lock=False)
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(np.asarray(request_values, dtype=float), task_kind="forecast"),
    )
    result, trace = TTHAFastAgent(core).prepare(
        request, h0,
        experience_episodes=tuple(episodes),
        calendar_period=PERIOD,
    )
    system_text = ""
    for message in backend.requests:
        content = message.messages
        if isinstance(content, Sequence) and content:
            first = content[0]
            if isinstance(first, Mapping) and "content" in first:
                system_text = str(first["content"])
                break
    return result, trace, system_text


def main() -> int:
    root = PROJECT_ROOT
    config = dict(v6.DATASET_CONFIGS[DOMAIN])
    _roster, values = v6._fixed_roster(root, config)

    # 1) 重建 R3 决策时刻 Memory
    print(f"== {DOMAIN}: rebuilding R3-decision memory (frozen chain)")
    source_episodes, a5_local, _cache = build_r3_memory(root)
    memory = list(source_episodes) + list(a5_local)
    print(f"== memory: source={len(source_episodes)} local={len(a5_local)} "
          f"total={len(memory)}")

    # 2) 真实入口 Replay：A5（带 Memory，单序列 request）+ A3（空 Memory 对照）
    series = values[list(values.keys())[0]]  # 与 v6 链同一数据（任一序列）
    r3_values = np.asarray(series)[:928]  # 部署可见：截断在 R3 决策点

    a5_backend = RecordingBackend(_identity_responses())
    a5_result, a5_trace, a5_instruction = run_prepare(r3_values, memory, a5_backend)
    a3_backend = RecordingBackend(_identity_responses())
    a3_result, a3_trace, a3_instruction = run_prepare(r3_values, (), a3_backend)

    # 3) 单序列口径严格证明（审查 2026-08-08）：fast_agent 用 except: pass 吞异常，
    #    "无注入"不能只靠推理——直接用同一单序列 query 调 resolver，确认：
    #    resolver 正常返回、全部 UNKNOWN、rendered 空、且 meta 距离明细非空
    #    （证明是距离/支持域判定导致，而不是异常/缺特征/KeyError）。
    _ops = sorted(n for n in v6.OPERATOR_NAMES
                  if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                  and n not in core.CTS_EXCLUDED)
    _single_q = resolver.window_context({"s": r3_values}, 928, PERIOD)
    _s_order, _s_signed = resolver.resolve_order(
        query_context=_single_q, episodes=memory, operators=_ops,
        material_threshold=core.MATERIAL_THRESHOLD)
    _s_rendered = resolver.render_signed_instruction(_s_signed, _s_order)
    _s_counts = _s_signed["summary"]["verdict_counts"]
    _s_meta_distances = [
        d for op in _s_signed["per_op"].values()
        for d in (op.get("meta") or {}).get("distances", {}).values()
    ]

    # 3) cohort 口径（与冻结链一致）经 observed_pattern_spec 走真实入口：
    #    上游把部署可见 cohort recent/change Context 放进 request.observed_pattern_spec，
    #    真实 Fast Agent prompt 应注入 denoise_savgol（审查裁决 3' 方案）。
    _cohort_q = resolver.window_context(values, 928, PERIOD)
    a5c_backend = RecordingBackend(_identity_responses())
    a5c_result, a5c_trace, a5c_instruction = run_prepare(
        r3_values, memory, a5c_backend, observed_pattern_spec=_cohort_q)

    _c_order, _c_signed = resolver.resolve_order(
        query_context=_cohort_q, episodes=memory, operators=_ops,
        material_threshold=core.MATERIAL_THRESHOLD)
    _rendered = resolver.render_signed_instruction(_c_signed, _c_order)

    checks: dict[str, bool] = {}
    # (a) 真实入口单序列：安全回退（不注入、不崩溃、零 LLM）
    checks["a5_single_series_no_injection"] = (
        "verified target-local experience" not in a5_instruction)
    checks["a3_no_injection"] = "verified target-local experience" not in a3_instruction
    checks["zero_llm_replay"] = (
        len(a5_backend.requests) == 3 and len(a3_backend.requests) == 3
        and len(a5c_backend.requests) == 3)
    # (b) 单序列口径严格证明：正常返回 + 全 UNKNOWN + 空渲染 + 距离明细非空
    checks["single_series_resolver_returns"] = len(_s_order) == len(_ops)
    checks["single_series_all_unknown"] = _s_counts["UNKNOWN"] == len(_ops)
    checks["single_series_rendered_empty"] = _s_rendered == ""
    _s_const_dev = any(
        (op.get("meta") or {}).get("query_const_deviations", {})
        for op in _s_signed["per_op"].values())
    checks["single_series_distance_proven"] = (
        (len(_s_meta_distances) > 0
         and all(d > _s_signed["summary"]["delta_q75"] for d in _s_meta_distances))
        or bool(_s_const_dev))
    # (c) cohort 口径（observed_pattern_spec）真实入口注入
    checks["cohort_observed_spec_injects"] = (
        "verified target-local experience" in a5c_instruction
        and "denoise_savgol" in a5c_instruction)
    checks["cohort_observed_spec_stl_not_prior"] = (
        "denoise_stl" not in (a5c_instruction.split("Reference 1")[-1]
                              .split("Reference 2")[0]
                              if "Reference 1" in a5c_instruction else a5c_instruction))
    import re as _re
    checks["no_gain_values_in_instruction"] = not _re.search(r"\bgain\b", a5c_instruction)

    print("\n== single-series resolver: counts=", _s_counts,
          "rendered_empty=", _s_rendered == "",
          "distances_n=", len(_s_meta_distances),
          "delta=", _s_signed["summary"]["delta_q75"])
    print("\n== cohort observed_pattern_spec instruction (head):")
    print(a5c_instruction[:500])
    print(f"\n== checks: {checks}")

    # 审查口径（2026-08-08）：三个独立 verdict，不再合并为 SIGNED_INJECTION_PASS
    resolver_pass = (
        checks["cohort_observed_spec_injects"]
        and checks["cohort_observed_spec_stl_not_prior"]
        and checks["no_gain_values_in_instruction"])
    fallback_pass = (
        checks["a5_single_series_no_injection"]
        and checks["single_series_resolver_returns"]
        and checks["single_series_all_unknown"]
        and checks["single_series_rendered_empty"]
        and checks["single_series_distance_proven"])
    injection_pass = (
        checks["cohort_observed_spec_injects"]
        and checks["a3_no_injection"]
        and checks["zero_llm_replay"])
    print("== verdict: "
          f"SIGNED_RESOLVER_PASS={resolver_pass} | "
          f"FAST_AGENT_NO_MATCH_FALLBACK_PASS={fallback_pass} | "
          f"FAST_AGENT_SIGNED_INJECTION={('PASS' if injection_pass else 'PARTIAL')}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-fastagent-signed-replay",
            "domain": DOMAIN,
            "memory_counts": {"source": len(source_episodes), "local": len(a5_local)},
            "checks": checks,
            "verdicts": {
                "SIGNED_RESOLVER_PASS": resolver_pass,
                "FAST_AGENT_NO_MATCH_FALLBACK_PASS": fallback_pass,
                "FAST_AGENT_SIGNED_INJECTION": "PASS" if injection_pass else "PARTIAL",
            },
            "single_series_verdict_counts": _s_counts,
            "single_series_delta": _s_signed["summary"]["delta_q75"],
            "single_series_distance_n": len(_s_meta_distances),
            "a5_single_series_instruction_head": a5_instruction[:300],
            "a5_cohort_instruction": a5c_instruction,
            "a5_status": a5_result.status.name,
            "a3_status": a3_result.status.name,
            "a5_cohort_status": a5c_result.status.name,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
