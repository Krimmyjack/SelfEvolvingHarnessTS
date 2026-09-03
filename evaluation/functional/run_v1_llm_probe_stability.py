"""第二步：plan-only Reference-1 稳定性诊断（审核 2026-08-09）。

已暴露 metr_la（offset=40）R1 Context、固定候选 [winsorize, outlier_iqr]、
Source Memory（POSITIVE winsorize——真实 Episode）。真实 LLM 同输入调用
2 次（不投票、不评估 gain）。

select prompt 只增加一条**局部动作语义**（不改全局 signed renderer）：

  > 选择候选表示申请一次有预算的 Support probe，不表示最终部署；
  > Reference 1 的当前 Support 尚未确认不是 abstain 理由，Support probe
  > 本身就是确认过程。

判定（预注册）：
  - 两次都选 winsorize      → LLM_REFERENCE1_PROBE_STABLE_PASS
  - 一次选、一次 abstain    → LLM_REFERENCE1_SELECTION_UNSTABLE
  - 两次 abstain            → LLM_REFERENCE1_NOT_ACTIONABLE
  - 接口/格式失败            → LLM_REFERENCE1_DIAG_INCONCLUSIVE

用法：
  python evaluation/functional/run_v1_llm_probe_stability.py
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
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.signed_radius import (  # noqa: E402
    MATERIAL_THRESHOLD, render_signed_instruction, resolve_order)

MODEL = "gpt-5.6-luna"
BASE_URL = "https://api.agicto.cn/v1"
PERIOD = 24
CAND_WINSORIZE = "cand_winsorize"
CAND_OUTLIER = "cand_outlier_iqr"
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_llm_probe_stability_report.json")

PROBE_SEMANTICS = (
    "Choosing a candidate means requesting one budgeted Support probe — it "
    "does NOT mean final deployment. The current Support of a Reference 1 "
    "operator has not yet been confirmed; that is not a reason to abstain — "
    "the Support probe itself IS the confirmation process.")


class CountingClient:
    def __init__(self, delegate: Any, *, max_calls: int = 4) -> None:
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
    def __init__(self, owner: CountingClient) -> None:
        self.completions = _Completions(owner)


class _Completions:
    def __init__(self, owner: CountingClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._create(**kwargs)


def _prompt(candidates: Sequence[Mapping[str, Any]],
            context_plain: Mapping[str, float], memory_rendered: str) -> str:
    lines = [
        "You are the fast-path selector of a time-series preprocessing "
        "harness.",
        "Choose exactly one candidate to probe at the support decision point.",
        "",
        "== Candidate programs (fixed order) ==",
    ]
    for i, cand in enumerate(candidates, start=1):
        lines.append(f"  {i}. {cand['candidate_id']}: {cand['op']} "
                     f"params={json.dumps(cand['params'], sort_keys=True)}")
    lines += [
        "",
        "== Public context at the decision point (deployment-visible) ==",
        json.dumps({k: round(float(v), 6) for k, v in sorted(
            context_plain.items())}, indent=2, sort_keys=True),
        "",
        "== Experience memory (signed, from earlier decision points) ==",
        memory_rendered.strip() if memory_rendered.strip() else
        "  (no applicable signed experience)",
        "",
        "== Probe semantics (important) ==",
        PROBE_SEMANTICS,
        "",
        "== Your task ==",
        f"Choose exactly one candidate ID from "
        f"{[c['candidate_id'] for c in candidates]} or ABSTAIN.",
        'Output JSON only: {"chosen_candidate_id": "<id|ABSTAIN>", '
        '"evidence_refs": ["..."], "rationale": "..."}',
    ]
    return "\n".join(lines)


def main() -> int:
    root = PROJECT_ROOT
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — INCONCLUSIVE")
        out = root / REPORT_OUT_REL
        out.write_text(json.dumps({"experiment_id": "v1-llm-probe-stability",
                                   "verdict": "LLM_REFERENCE1_DIAG_INCONCLUSIVE",
                                   "error": "no api key"},
                                  ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        return 0
    import openai
    client = CountingClient(openai.OpenAI(api_key=api_key,
                                          base_url=BASE_URL), max_calls=4)

    # metr_la offset=40（已暴露）R1 Context + Source Memory（POSITIVE winsorize）
    sealed._set_domain("metr_la")
    config = sealed._config()
    (src_roster, src_values, tgt_roster, tgt_values) = sealed._virgin_roster(
        root, offset=40)
    context_plain = dict(resolver.window_context(tgt_values, 792, PERIOD))
    context_plain["bound_period"] = float(PERIOD)
    candidates = [
        {"candidate_id": CAND_WINSORIZE, "op": "winsorize", "params": {}},
        {"candidate_id": CAND_OUTLIER, "op": "outlier_iqr", "params": {}},
    ]
    ops = ("winsorize", "outlier_iqr")

    # Source Memory（真实 Episode：denoise 0.0 / winsorize POSITIVE，delayed 打开）
    ep1 = tll.write_target_episode(
        domain="metr_la", op="denoise_median",
        program_steps=[{"op": "denoise_median", "params": {"strength": 1.0,
                                                           "window": 1}}],
        support_gain=0.0, delayed_gain=None,
        support_context=resolver.window_context(src_values, 600, PERIOD),
        episode_id_suffix="_stab1")
    ep2 = tll.write_target_episode(
        domain="metr_la", op="winsorize",
        program_steps=[{"op": "winsorize", "params": {}}],
        support_gain=0.8944078152454575, delayed_gain=None,
        support_context=resolver.window_context(src_values, 600, PERIOD),
        episode_id_suffix="_stab2")
    ep1 = tll.update_delayed_status(
        ep1, 0.0, delayed_context=resolver.window_context(src_values, 648,
                                                          PERIOD))
    ep2 = tll.update_delayed_status(
        ep2, 1.2404722987946628,
        delayed_context=resolver.window_context(src_values, 648, PERIOD))
    order, signed = resolve_order(
        query_context={k: float(v) for k, v in context_plain.items()
                       if k.startswith(("recent.", "change."))},
        episodes=(ep1, ep2), operators=ops,
        material_threshold=MATERIAL_THRESHOLD,
        task_consumer_key="forecast|ridge|sMASE", allowed_operators=ops)
    memory_rendered = render_signed_instruction(signed, order,
                                                executable_ops=ops)
    print(f"== rendered: {memory_rendered[:150]!r}")

    prompt = _prompt(candidates, context_plain, memory_rendered)
    # 2 次调用（同输入；不投票、不评估 gain）
    choices: list[dict[str, Any]] = []
    for i in range(2):
        kwargs = {"model": MODEL,
                  "messages": [{"role": "user", "content": prompt}]}
        try:
            resp = client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"})
        except Exception:
            resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        try:
            chosen = str(json.loads(raw).get("chosen_candidate_id", ""))
        except json.JSONDecodeError:
            chosen = ""
        ok = chosen in (CAND_WINSORIZE, CAND_OUTLIER, "ABSTAIN")
        choices.append({"call": i + 1, "chosen": chosen, "ok": ok,
                        "raw": raw})
        print(f"== call {i + 1}: chosen={chosen} ok={ok}")

    picks = [c["chosen"] for c in choices]
    ok_all = all(c["ok"] for c in choices)
    if not ok_all:
        verdict = "LLM_REFERENCE1_DIAG_INCONCLUSIVE"
    elif picks == [CAND_WINSORIZE, CAND_WINSORIZE]:
        verdict = "LLM_REFERENCE1_PROBE_STABLE_PASS"
    elif CAND_WINSORIZE in picks and "ABSTAIN" in picks:
        verdict = "LLM_REFERENCE1_SELECTION_UNSTABLE"
    else:
        verdict = "LLM_REFERENCE1_NOT_ACTIONABLE"
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-llm-probe-stability",
        "setting": "metr_la offset=40（已暴露）R1 @792 Context；候选 "
                   "[winsorize, outlier_iqr]；Source Memory POSITIVE "
                   "winsorize；luna temp=0；2 次调用同输入（不投票）；"
                   "plan-only（不评估 gain）",
        "probe_semantics_added": PROBE_SEMANTICS,
        "candidates": candidates,
        "memory_rendered": memory_rendered,
        "choices": choices,
        "llm_call_count": client.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
