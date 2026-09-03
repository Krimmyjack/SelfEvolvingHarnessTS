"""LEVEL_SHIFT_CANDIDATE_AVAILABILITY_TEST（用户裁决 2026-08-10）。

前提：PUBLIC_LEVEL_EXCURSION_BINDING_PREMISE = BOUND_CANDIDATE_DELAYED_
STABLE_HEADROOM（@744 双正 +0.083/+0.072）。区分 Supply vs Selector：

  Control：  CandidatePool [denoise_median]
  Treatment：CandidatePool [denoise_median, bound repair_level_shift]

保持相同 Context、Prompt、LLM 和预算（@744 决策点；LLMSelectBackend
真实 LLM select）；不把两个候选的 gain 告诉 LLM。bound repair 参数
全部来自公开 Context（extract_public_features @744 的 mapping 值——
post_validator 硬约束：参数必须等于公开特征值）。

结果解释（用户裁决预定义）：
  - Treatment 中 LLM 选择 repair_level_shift → selector 能理解 level
    shift，真正 first fault 是 Program Supply（之后只修复每轮最多两个
    context-relevant candidates 的供应）
  - Treatment 中仍然 abstain → selector 没有消费 level-shift 证据
    （优先让 Runtime 验证两个候选，不先调 Prompt）
  - 两次相同 Treatment 行为不同 → INCONCLUSIVE_LLM_VARIANCE

Verdict（预注册）：
  SELECTOR_SUPPLY_FIRST_FAULT（Treatment 选 repair）
  SELECTOR_NOT_CONSUMING_LEVEL_SHIFT（Treatment 全 abstain）
  INCONCLUSIVE_LLM_VARIANCE（Treatment 两次不一致）
  CONTROL_ABSTAIN_ALSO（Control 也 abstain——参考记录，不单独判档）

development 数据（已暴露），不称 fresh。

用法：
  python evaluation/functional/run_v1_level_shift_candidate_availability_test.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

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
from SelfEvolvingHarnessTS.runtime.public_features import extract_public_features  # noqa: E402

DOMAIN = "uci_electricity_load_diagrams"
OFFSET = 40
PERIOD = 24
ORIGIN = 744  # headroom 稳定决策点（premise 实验 @744 双正）
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_level_shift_candidate_availability_test_report.json")


def _make_method(backend: Any, snapshot: Any, series0: np.ndarray) -> TTHAMethod:
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            backend,
            LocalPublicToolGateway(series0[:ORIGIN], task_kind="forecast"))),
        snapshot, ())
    method.bind_round_data(series0[:ORIGIN], task_kind="forecast")
    return method


def main() -> int:
    root = PROJECT_ROOT
    sealed._set_domain(DOMAIN)
    config = sealed._config()
    (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(root, offset=OFFSET)
    series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                         dtype=np.float64)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)

    # bound 参数（公开 Context，@744；post_validator 要求=特征值）
    fe = extract_public_features(series0[:ORIGIN], task_kind="forecast")
    mapping = dict(fe.mapping)
    bound = {
        "region_start_fraction": float(mapping["estimated_region_start_fraction"]),
        "region_end_fraction": float(mapping["estimated_region_end_fraction"]),
        "estimated_offset": float(mapping["estimated_level_offset"]),
    }
    print(f"== bound repair params (public @{ORIGIN}): {json.dumps(bound)}")

    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — INCONCLUSIVE")
        return 0
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=20)

    ctx = dict(resolver.window_context(tgt_values, ORIGIN, PERIOD))
    ctx["bound_period"] = float(PERIOD)

    def run(operators, *, label: str, max_propose: int = 1,
            force_pool: bool = False) -> dict[str, Any]:
        backend = sealed.LLMSelectBackend(
            explore=True, operators=operators, client=counter,
            context_plain=dict(ctx),
            max_propose_candidates=max_propose,
            bound_params=({"repair_level_shift": bound} if force_pool else None),
            force_pool=force_pool)
        method = _make_method(backend, h0, series0)
        method.prepare(sealed._request(series0, tgt_values, ORIGIN))
        trace = method.last_trace
        last = (backend._select_logs[-1] if backend._select_logs else {})
        return {"label": label, "pool": list(trace.candidate_ids),
                "chosen": trace.chosen_candidate_id,
                "rationale": (last.get("raw") or "")[:400]}

    runs: list[dict[str, Any]] = []
    runs.append(run(("denoise_median",), label="control"))
    runs.append(run(("denoise_median", "repair_level_shift"),
                    label="treatment_1", max_propose=2, force_pool=True))
    runs.append(run(("denoise_median", "repair_level_shift"),
                    label="treatment_2", max_propose=2, force_pool=True))
    for r in runs:
        print(f"== {r['label']}: pool={r['pool']} chosen={r['chosen']}")
        if r["rationale"]:
            print(f"   rationale: {r['rationale'][:200]}")

    # ---- verdict（用户裁决解释规则）----
    t1 = next(r for r in runs if r["label"] == "treatment_1")
    t2 = next(r for r in runs if r["label"] == "treatment_2")
    t1_repair = t1["chosen"] == "cand_repair_level_shift"
    t2_repair = t2["chosen"] == "cand_repair_level_shift"
    t1_abstain = t1["chosen"] == "identity"
    t2_abstain = t2["chosen"] == "identity"
    if t1_repair or t2_repair:
        verdict = "SELECTOR_SUPPLY_FIRST_FAULT"
    elif t1_abstain and t2_abstain:
        verdict = "SELECTOR_NOT_CONSUMING_LEVEL_SHIFT"
    else:
        verdict = "INCONCLUSIVE_LLM_VARIANCE"

    print(f"== verdict: {verdict}")
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-level-shift-candidate-availability-test",
        "dataset": DOMAIN, "cohort_offset": OFFSET, "origin": ORIGIN,
        "bound_params": bound,
        "runs": runs,
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
