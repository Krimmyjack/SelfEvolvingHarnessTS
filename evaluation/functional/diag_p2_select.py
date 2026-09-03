"""P2 abstain 诊断：跑单轮（origin=648），dump propose 渲染的候选与 LLM
select 的 prompt/响应，定位为何 LLM 全程选 identity。诊断用，不改 Harness。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402

DOMAIN = "monash:traffic_hourly"
OFFSET = 240
PERIOD = 24
ORIGIN = 648

OPS_ALL = tuple(o for o in (
    "denoise_median", "hampel_filter", "impute_ar", "impute_ema",
    "impute_fft", "impute_linear", "impute_ssm", "outlier_iqr",
    "outlier_mad", "period_complete", "period_median_complete",
    "repair_level_shift", "resample_uniform", "winsorize"))


def main() -> int:
    root = PROJECT_ROOT
    sealed._set_domain(DOMAIN)
    config = sealed._config()
    (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(root, offset=OFFSET)
    series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                         dtype=np.float64)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)

    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=10)

    ctx = dict(resolver.window_context(tgt_values, ORIGIN, PERIOD))
    ctx["bound_period"] = float(PERIOD)
    backend = sealed.LLMSelectBackend(
        explore=True, operators=OPS_ALL, client=counter,
        context_plain=dict(ctx))
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            backend,
            LocalPublicToolGateway(series0[:ORIGIN], task_kind="forecast"))),
        h0, ())
    method.bind_round_data(series0[:ORIGIN], task_kind="forecast")
    result = method.prepare(sealed._request(series0, tgt_values, ORIGIN))
    trace = method.last_trace
    print("== candidate_ids:", list(trace.candidate_ids))
    print("== chosen:", trace.chosen_candidate_id)
    print("== program:", result.program is not None)
    print("== llm_calls:", counter.calls)
    if backend._select_logs:
        log = backend._select_logs[-1]
        print("== select prompt (first 4000 chars) ==")
        print(log["prompt"][:4000])
        print("== raw ==")
        print(log["raw"][:600])
    # propose 阶段的 requests（看 instruction 渲染）
    for req in backend.requests:
        for m in getattr(req, "messages", []):
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str) and "candidates" in content:
                print("== propose message has candidates JSON, len:", len(content))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
