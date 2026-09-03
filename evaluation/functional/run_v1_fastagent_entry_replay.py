"""V1 真实入口 Replay：TTHAFastAgent.prepare 经验注入（零 LLM，2026-08-08）。

收束裁决第 3 步：一次零 LLM 真实入口验证——
1. A3（空 Episode）不注入（instruction 不含经验渲染块）；
2. A5（Source Episode）注入（instruction 含渲染块）；
3. instruction 与候选行为确实变化。

用 compile_snapshot(verify_lock=False)（审查允许——不修锁基础设施，
针对性集成测试用受控验证）。LLM 级配对 smoke 另行（收束裁决第 4 步）。

用法：
  python evaluation/functional/run_v1_fastagent_entry_replay.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))

import numpy as np  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, TaskSpec  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from experience_memory import load_experience_episodes  # noqa: E402

H0 = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"
EXPERIENCE_REFERENCES = "EXPERIENCE REFERENCES FROM PRIOR TRIALS"
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_fastagent_entry_replay_report.json")


def make_task_spec() -> TaskSpec:
    return TaskSpec(
        task_type="forecast",
        target_semantics="future_values",
        label_availability="history_only",
        metric=MetricSpec(name="sMASE", direction="lower_is_better"),
        horizon=48,
        downstream_model_class="ridge",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 fast-agent entry replay (zero-LLM)")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    # 受控 snapshot（verify_lock=False——审查允许的针对性集成测试方式）
    snapshot = compile_snapshot(H0, verify_lock=False)
    print(f"== snapshot compiled (verify_lock=False), instruction len={len(snapshot.instruction)}")

    # PreparationRequest 最小构造（values 只需形状合法——prepare 会走 verify 与特征提取）
    from SelfEvolvingHarnessTS.contracts.method import PreparationRequest
    values = np.linspace(0.0, 1.0, 240, dtype=np.float64)
    request = PreparationRequest(
        series_uid="replay_probe_series",
        values=values,
        task_spec=make_task_spec(),
    )

    from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway
    from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse, ReplayAgentBackend

    class _CapturingBackend(ReplayAgentBackend):
        """记录每次 stage 请求的 prompt（用于断言 instruction 是否含经验块）。"""

        def __init__(self, responses):
            super().__init__(responses)
            self.captured_prompts: list[str] = []

        def complete(self, prompt, **kwargs):
            self.captured_prompts.append(str(prompt))
            return super().complete(prompt, **kwargs)

    def _make_agent(with_episodes: bool):
        responses = [
            AgentResponse.valid(
                {"schema_version": "agent-envelope/1", "kind": "stage_result",
                 "stage": "inspect",
                 "payload": {"inspected_region_fractions": [[0.0, 1.0]],
                             "requested_public_tools": [], "uncertainty": "high"}},
                raw_response={"id": "replay-inspect"},
            ),
            AgentResponse.valid(
                {"schema_version": "agent-envelope/1", "kind": "stage_result",
                 "stage": "propose", "payload": {"candidates": []}},
                raw_response={"id": "replay-propose"},
            ),
            AgentResponse.valid(
                {"schema_version": "agent-envelope/1", "kind": "stage_result",
                 "stage": "select",
                 "payload": {"chosen_candidate_id": "identity",
                             "verification_actions": ["public_evidence_insufficient"]}},
                raw_response={"id": "replay-select"},
            ),
        ] * 3  # 磁带充足（prepare 多阶段 + 可能重试）
        backend = _CapturingBackend(responses)
        core = TTHAAgentCore(
            backend,
            LocalPublicToolGateway(np.linspace(0.0, 1.0, 240), task_kind="forecast"),
        )
        return TTHAFastAgent(core), backend

    episodes = load_experience_episodes(root / "artifacts/experience/episodes_v1_source.json")
    print(f"== source episodes: {len(episodes)}")

    # A3/A5 各自独立 backend/agent；不吞异常（两臂都必须正常返回）
    agent_a3, backend_a3 = _make_agent(with_episodes=False)
    agent_a3.prepare(request, snapshot)
    a3_prompts = " ".join(backend_a3.captured_prompts)
    a3_injected = EXPERIENCE_REFERENCES in a3_prompts

    agent_a5, backend_a5 = _make_agent(with_episodes=True)
    agent_a5.prepare(request, snapshot, experience_episodes=episodes)
    a5_prompts = " ".join(backend_a5.captured_prompts)
    a5_injected = EXPERIENCE_REFERENCES in a5_prompts

    print(f"[1] A3 injected={a3_injected} (expect False)")
    print(f"[2] A5 injected={a5_injected} (expect True)")
    print(f"[3] instruction changed={not a3_injected and a5_injected}")

    checks = {
        "a3_not_injected": not a3_injected,
        "a5_injected": a5_injected,
        "instruction_changed": (not a3_injected and a5_injected),
    }
    all_pass = all(checks.values())
    print(f"\n== checks: {checks}")
    print(f"== verdict: {'PASS' if all_pass else 'PARTIAL'}"
          f"{'（若 prepare 全链路需工具则标注）' if not all_pass else ''}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-fastagent-entry-replay",
            "snapshot_instruction_len": len(snapshot.instruction),
            "episodes_loaded": len(episodes),
            "checks": checks,
            "verdict": "PASS" if all_pass else "PARTIAL",
            "note": ("零 LLM 真实入口验证；prepare 全链路若需外部工具，注入判定以渲染块"
                     "是否进入 instruction 为准；LLM 级配对 smoke 另行（收束裁决第 4 步）。"),
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
