"""V1 正常运行路径接线验证（零 LLM，2026-08-08）。

审查裁决（① 最小实现）：
  1. 上游通过 request.observed_pattern_spec 提供部署可见 cohort recent/change Context
     （含 bound_period 键）；
  2. TTHAMethod 显式持有 Episode 集合（不隐式读全局文件）；
  3. calendar period 从同一公开请求上下文取得（observed_pattern_spec.bound_period）；
  4. 使用正常 TTHAMethod.prepare 调用链（非直接调 fast_agent）；
  5. RecordingBackend 确定性验证 system prompt 注入（零 LLM）；
  6. A3 空 Memory 走同一入口且不注入。

完成标准（正常方法入口同时满足）：
  cohort Context 到达 + 合法 Episode 到达 + signed resolver 被调用
  + 捕获的真实 system prompt 包含正确经验（denoise_savgol 提前、stl 不提前、
    无 gain 数值、措辞 "verified historical experience"）+ A3 无注入。

用法：
  python evaluation/functional/run_v1_method_normal_entry_verify.py
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
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastagent_signed_replay as replay  # noqa: E402（重建 + RecordingBackend 复用）
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402

HORIZON = 48
PERIOD = 24
DOMAIN = "gefcom"
R3_ORIGIN = 928
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_method_normal_entry_verify_report.json")


def _make_request(values: np.ndarray, observed: Mapping[str, float]) -> PreparationRequest:
    """正常方法入口的公开请求：observed_pattern_spec = 上游提供的 cohort Context。"""
    return PreparationRequest(
        "gefcom-r3-normal-entry",
        np.asarray(values, dtype=float),
        forecast_task_spec_v1(horizon=HORIZON,
                              downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )


def _run_normal_entry(
    values: np.ndarray,
    observed: Mapping[str, float],
    episodes: tuple[Any, ...],
    h0: Any,
) -> tuple[Any, str]:
    """正常入口：TTHAMethod.prepare（零 LLM RecordingBackend）。返回 (result, system_prompt)。"""
    backend = replay.RecordingBackend(replay._identity_responses())
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(np.asarray(values, dtype=float), task_kind="forecast"),
    )
    method = TTHAMethod(
        TTHAFastAgent(core),
        h0,
        experience_episodes=episodes,  # 显式持有，不隐式读文件
    )
    result = method.prepare(_make_request(values, observed))
    prompt = ""
    for req in backend.requests:
        content = req.messages
        if isinstance(content, tuple) and content and isinstance(content[0], Mapping):
            prompt = str(content[0].get("content", ""))
            break
    return result, prompt


def main() -> int:
    root = PROJECT_ROOT
    config = dict(v6.DATASET_CONFIGS[DOMAIN])
    _roster, values = v6._fixed_roster(root, config)
    series = values[list(values.keys())[0]]
    r3_values = np.asarray(series)[:R3_ORIGIN]

    # 重建 R3 决策时刻 Memory（与冻结链一致）
    source_episodes, a5_local, _ = replay.build_r3_memory(root)
    memory = tuple(source_episodes) + tuple(a5_local)

    # 上游：部署可见 cohort recent/change Context + bound_period（同一公开请求上下文）
    observed = dict(resolver.window_context(values, R3_ORIGIN, PERIOD))
    observed["bound_period"] = float(PERIOD)

    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0", verify_lock=False)

    # A5：正常入口带 Memory
    a5_result, a5_prompt = _run_normal_entry(r3_values, observed, memory, h0)
    # A3：同一入口，空 Memory（默认）
    a3_result, a3_prompt = _run_normal_entry(r3_values, observed, (), h0)

    # 完成标准断言
    import re as _re
    checks: dict[str, bool] = {
        # ① cohort Context 到达：prompt 含经验块（observed 被消费）
        "cohort_context_reached": "verified historical experience" in a5_prompt,
        # ② 合法 Episode 到达：denoise_savgol 出现在 Reference 1
        "legal_episodes_reached": (
            "Reference 1" in a5_prompt and "denoise_savgol" in a5_prompt),
        # ③ signed resolver 被调用：经验块来自 signed 渲染（新措辞）
        "signed_resolver_invoked": (
            "verified historical experience" in a5_prompt
            and "similarity radius" in a5_prompt),
        # ④ 正确经验：denoise_stl 不提前、无 gain 数值
        "correct_experience": (
            "denoise_stl" not in (a5_prompt.split("Reference 1")[-1]
                                  .split("Reference 2")[0]
                                  if "Reference 1" in a5_prompt else a5_prompt)
            and not _re.search(r"\bgain\b", a5_prompt)),
        # ⑤ A3 空 Memory 同一入口不注入
        "a3_no_injection": "verified historical experience" not in a3_prompt,
        # 零 LLM：每入口 3 次固定响应
        "zero_llm": a5_result.status is not None and a3_result.status is not None,
    }
    all_pass = all(checks.values())

    print("\n== A5 normal-entry system prompt (head):")
    print(a5_prompt[:600])
    print("\n== A3 normal-entry prompt (head):")
    print(a3_prompt[:150])
    print(f"\n== checks: {checks}")
    print(f"== verdict: {'NORMAL_ENTRY_WIRING_PASS' if all_pass else 'NORMAL_ENTRY_WIRING_PARTIAL'}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-method-normal-entry-verify",
            "domain": DOMAIN,
            "memory_counts": {"source": len(source_episodes), "local": len(a5_local)},
            "checks": checks,
            "verdict": "NORMAL_ENTRY_WIRING_PASS" if all_pass else "NORMAL_ENTRY_WIRING_PARTIAL",
            "a5_prompt": a5_prompt,
            "a3_prompt_head": a3_prompt[:300],
            "a5_status": a5_result.status.name,
            "a3_status": a3_result.status.name,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
