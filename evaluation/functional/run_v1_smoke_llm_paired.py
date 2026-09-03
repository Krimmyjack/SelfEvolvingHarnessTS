"""V1 2B：真实 LLM A5/A3 paired smoke（单次，2026-08-08）。

审查裁决（模型与预算）：
  - 模型：deepseek-v4-flash（https://api.deepseek.com/v1，env DEEPSEEK_API_KEY——
    与 w1_target_local_loop_llm_report.json 的 deepseek provider 一致，不切换）；
  - 温度：注入包装 client 强制 temperature=0（backend.complete 不传温度参数）；
  - 基础预算 6 次调用（inspect/propose/select × A5/A3），硬上限 8 次；
    达到 8 立即停止，不追加、不重跑、不换模型。

冻结要求：
  - 不改 Prompt/radius/候选顺序/Support 预算；A3/A5 唯一初始差异 = Source Experience；
  - trace 区分 SOURCE 与 TARGET_LOCAL（chosen 算子的 Memory 证据来源标注）；
  - candidate 冻结后才打开 gain；
  - 不因结果不理想重跑；单次只标 smoke。

表述（审查修正）：A5 的 Reference 来自 Target-local 成功链（Source-informed analogy
→ 改变 R1 探索 → Target Support/Delayed 形成 LOCAL_ACTIVE → 后续 Target-local
Memory 生效）——不称"Source 成功经验直接迁移"（NOAA Source 无双正 POSITIVE_PRIOR）。

预注册输出（仅取其一）：
  AGENT_SELECTION_SMOKE_POSITIVE       A5 引用 Memory 且 Support 更好
  AGENT_SOURCE_MEMORY_NOT_ACTIONABLE  引用但计划/行为不变（未遵循）
  AGENT_SELECTION_NEGATIVE_TRANSFER   A5 更差
  INCONCLUSIVE                        API/Schema/编译失败或调用 > 8

用法：
  python evaluation/functional/run_v1_smoke_llm_paired.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402（memory 重建 + contract_params 复用）
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgictoChatCompletionsBackend,
)

HORIZON = 48
PERIOD = 24
TARGET_DOMAIN = "gefcom"
MAX_CALLS = 8
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_smoke_llm_paired_report.json")
R2_SLICE = (832, 880)

# provider 配置（key env 名, base_url, model）——deepseek 为原裁决配置；
# agicto gpt-5.6-luna 为用户提供的备选（退役脚本同款）。
PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-v4-flash"),
    "agicto": (("OPENAI_API_KEY", "AGICTO_API_KEY"), "https://api.agicto.cn/v1", "gpt-5.6-luna"),
}


class ZeroTempCountingClient:
    """温度 0 + 调用计数 + 8 次硬停止包装（审查裁决 2026-08-08 必要最小修复：
    预算边界失效已实际发生——达到 MAX_CALLS 立即抛错停止，不追加、不重试）。"""

    def __init__(self, delegate: Any, *, max_calls: int = MAX_CALLS) -> None:
        self.calls = 0
        self._max_calls = max_calls
        self._delegate = delegate
        self.chat = _Chat(self)

    def _create(self, **kwargs: Any) -> Any:
        if self.calls >= self._max_calls:
            raise RuntimeError(
                f"LLM call budget exceeded (hard stop at {self._max_calls})")
        self.calls += 1
        kwargs.setdefault("temperature", 0)
        return self._delegate.chat.completions.create(**kwargs)


class _Chat:
    def __init__(self, owner: ZeroTempCountingClient) -> None:
        self.completions = _Completions(owner)


class _Completions:
    def __init__(self, owner: ZeroTempCountingClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._create(**kwargs)


def make_backend(api_key: str, base_url: str) -> tuple[AgictoChatCompletionsBackend, Any]:
    """LLM backend + 温度 0/计数包装 client（不改生产代码）。"""
    import openai
    raw = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    counter = ZeroTempCountingClient(raw)
    return AgictoChatCompletionsBackend(client=counter, base_url=base_url), counter


def run_prepare(values: np.ndarray, observed: Mapping[str, float], episodes: Sequence[Any],
                backend: Any, h0: Any, counter: Any, *, model: str,
                base_url: str) -> tuple[Any, Any, str, int]:
    request = PreparationRequest(
        "gefcom-smoke-r2",
        np.asarray(values, dtype=float),
        forecast_task_spec_v1(horizon=HORIZON,
                              downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(np.asarray(values, dtype=float), task_kind="forecast"),
        model=model,
        base_url=base_url,
    )
    try:
        result, trace = TTHAFastAgent(core).prepare(
            request, h0, experience_episodes=tuple(episodes))
    except RuntimeError as exc:  # 硬停止（预算超限）
        if "budget exceeded" in str(exc):
            return None, None, counter.calls
        raise
    return result, trace, counter.calls


def evidence_source_of(op: str, memory: Sequence[Any]) -> list[str]:
    """chosen 算子的 Memory 证据来源标注（SOURCE=noaa / TARGET_LOCAL=gefcom）。"""
    sources: list[str] = []
    for ep in memory:
        if ep.workflow_signature == op:
            src = "TARGET_LOCAL" if getattr(ep, "domain_namespace", "") == TARGET_DOMAIN else "SOURCE"
            if src not in sources:
                sources.append(src)
    return sources


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="V1 2B real-LLM paired smoke")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default="deepseek")
    args = parser.parse_args()
    root = args.root.resolve()
    provider = args.provider
    key_envs, base_url, model = PROVIDERS[provider]
    key_envs = (key_envs,) if isinstance(key_envs, str) else key_envs
    api_key = next((os.environ.get(k, "").strip() for k in key_envs
                    if os.environ.get(k, "").strip()), "")
    if not api_key:
        print(f"== INCONCLUSIVE: provider={provider} 的 key env {key_envs} 未设置")
        return 0

    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}
    series = values[list(values.keys())[0]]
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0", verify_lock=False)

    memory = wiring.build_r2_memory(root)
    ts, td = R2_SLICE
    observed = dict(resolver.window_context(values, ts, period))
    observed["bound_period"] = float(period)
    r_values = np.asarray(series)[:ts]

    # A5：带 Memory（noaa source + gefcom 本地）；A3：空——同 backend/模型/预算
    a5_backend, a5_counter = make_backend(api_key, base_url)
    a5_result, a5_trace, a5_calls = run_prepare(
        r_values, observed, memory, a5_backend, h0, a5_counter,
        model=model, base_url=base_url)
    a3_backend, a3_counter = make_backend(api_key, base_url)
    a3_result, a3_trace, a3_calls = run_prepare(
        r_values, observed, (), a3_backend, h0, a3_counter,
        model=model, base_url=base_url)
    total_calls = a5_calls + a3_calls

    def chosen_operator(trace: Any) -> str | None:
        """从 trace 的 candidate_program_steps 取 chosen 算子（真实 LLM 的
        candidate_id 格式不定，不猜前缀）。"""
        if trace.chosen_candidate_id == "identity":
            return None
        steps = trace.candidate_program_steps.get(trace.chosen_candidate_id)
        if steps:
            return str(steps[0][0])
        return None

    # 候选冻结后才打开 gain（trace None = 预算硬停止）
    chosen_a5 = a5_trace.chosen_candidate_id if a5_trace is not None else "<hard_stop>"
    chosen_a3 = a3_trace.chosen_candidate_id if a3_trace is not None else "<hard_stop>"
    a5_op = chosen_operator(a5_trace) if a5_trace is not None else None
    a3_op = chosen_operator(a3_trace) if a3_trace is not None else None
    a5_gain = a3_gain = None
    if a5_op is not None:
        compiled = v1.make_compiled(a5_op, wiring.contract_params(a5_op, period))
        a5_gain = v1.gain_at(roster, values, config, compiled, ts, baseline_cache)
    if a3_op is not None:
        compiled = v1.make_compiled(a3_op, wiring.contract_params(a3_op, period))
        a3_gain = v1.gain_at(roster, values, config, compiled, ts, baseline_cache)

    # 结果判定（预注册；预算超限如实记录但不断言失败——用户裁决放松；
    # 硬停止修复后超限以 trace=None 显现）
    over_budget = total_calls > MAX_CALLS or a5_trace is None or a3_trace is None
    failure = (a5_trace is None or a3_trace is None
               or a5_trace.compilation_status not in ("ok", "compiled", "not_applicable")
               or a3_trace.compilation_status not in ("ok", "compiled", "not_applicable"))
    if failure:
        verdict = "INCONCLUSIVE"
    elif a5_op is not None and a5_gain is not None and a5_gain >= core.MATERIAL_THRESHOLD:
        verdict = "AGENT_SELECTION_SMOKE_POSITIVE"
    elif a5_op is not None and a5_gain is not None and a5_gain < 0:
        verdict = "AGENT_SELECTION_NEGATIVE_TRANSFER"
    else:
        verdict = "AGENT_SOURCE_MEMORY_NOT_ACTIONABLE"

    sources = evidence_source_of(a5_op, memory) if a5_op is not None else []

    print(f"== calls: A5={a5_calls} A3={a3_calls} total={total_calls} (cap {MAX_CALLS}, "
          f"over_budget={over_budget})")
    print(f"== A5 chosen={chosen_a5} op={a5_op} compile={a5_trace.compilation_status} "
          f"gain={a5_gain if a5_gain is None else round(a5_gain, 4)} "
          f"evidence_sources={sources}")
    if a5_trace.compilation_status != "ok" or a3_trace.compilation_status != "ok":
        for arm_name, trace in (("A5", a5_trace), ("A3", a3_trace)):
            print(f"  [{arm_name}] candidates={trace.candidate_ids} "
                  f"rejections={trace.rejection_receipts}")
    print(f"== A3 chosen={chosen_a3} op={a3_op} compile={a3_trace.compilation_status} "
          f"gain={a3_gain if a3_gain is None else round(a3_gain, 4)}")
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-smoke-llm-paired",
            "provider": provider, "model": model,
            "calls": {"a5": a5_calls, "a3": a3_calls, "total": total_calls,
                      "cap": MAX_CALLS, "over_budget": over_budget},
            "a5": {"chosen_candidate": chosen_a5, "chosen_operator": a5_op,
                   "compilation_status": a5_trace.compilation_status,
                   "support_gain": a5_gain,
                   "evidence_sources": sources},
            "a3": {"chosen_candidate": chosen_a3, "chosen_operator": a3_op,
                   "compilation_status": a3_trace.compilation_status,
                   "support_gain": a3_gain},
            "verdict": verdict,
            "smoke_only": True,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
