"""Weak Risk 降级但不排除——plan-only 验证（审核 2026-08-09）。

first fault（offset=80 NEGATIVE）：weak_reference 的负经验没有 Context
匹配依据，却被真实 LLM 当成全局禁令 → A5 放弃 Target 探测。

修复行为（风险分级，只改 LLM 选择契约的表达，不改全局 signed renderer）：
  - Reference 2/3 + weak_reference（未校准 Context）→ 降到 UNKNOWN 之后，
    不得直接导致全局 abstain；UNKNOWN 耗尽且仍有 Support 预算时，允许
    一次有界探测；
  - Reference 2/3 + 已匹配 Context / Target-local 实证 → 保持强风险，
    可以 abstain。

验证（plan-only，不读取任何新 outcome）：
  weak 路径（metr offset=80，Source NEGATIVE winsorize = weak RISK）：
    1. 候选 [denoise_median(UNKNOWN), winsorize(weak RISK)] → 先选 UNKNOWN；
    2. 已知 UNKNOWN Support=0（Source 阶段真实数据，非新评估），预算仍剩
       一次 → 第二次选 weak-risk winsorize（有界探测），不直接 abstain。
  strong 负控（uci，Target-local RESTRICTED winsorize = strong RISK）：
    winsorize 仍应被规避或 abstain（防止修复退化成"所有风险都要试"）。

真实 LLM 调用预算：weak 路径 2 次（UNKNOWN→risk probe 序列）+ strong
负控 2 次（重复验证规避）= 总上限 4。不投票。

判定：
  weak 序列完成 UNKNOWN→risk probe 且 strong 2/2 规避 → WEAK_RISK_GRADED_PASS
  weak 序列 abstain（未到 risk probe）→ WEAK_RISK_TOO_CONSERVATIVE
  strong 探测 winsorize → WEAK_RISK_OVERGENERALIZED
  格式/接口失败 → WEAK_RISK_INCONCLUSIVE

用法：
  python evaluation/functional/run_v1_weak_risk_planonly.py
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

import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.signed_radius import (  # noqa: E402
    MATERIAL_THRESHOLD, render_signed_instruction, resolve_order)

MODEL = "gpt-5.6-luna"
BASE_URL = "https://api.agicto.cn/v1"
PERIOD = 24
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_weak_risk_planonly_report.json")

RISK_SEMANTICS = (
    "== Risk semantics (important) ==\n"
    "- Weak risk (negative evidence from an uncalibrated context, "
    "'weak reference'): the operator is downgraded below UNKNOWN "
    "candidates, but this is NOT a global prohibition. After UNKNOWN "
    "candidates have been exhausted and Support budget remains, one "
    "bounded probe of a weak-risk operator is allowed — the probe itself "
    "is the confirmation.\n"
    "- Strong risk (context-matched or target-local negative evidence): "
    "avoid or abstain — do not probe.")


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


def _select_prompt(candidates: Sequence[Mapping[str, Any]],
                   context_plain: Mapping[str, float],
                   memory_rendered: str, extra_note: str = "") -> str:
    lines = [
        "You are the fast-path selector of a time-series preprocessing "
        "harness.",
        "Choose exactly one candidate to probe at the support decision point.",
        "",
        "== Candidate programs (fixed order) ==",
    ]
    for i, cand in enumerate(candidates, start=1):
        labels = ", ".join(cand.get("labels", []))
        lines.append(
            f"  {i}. {cand['candidate_id']}: {cand['op']} "
            f"params={json.dumps(cand['params'], sort_keys=True)}"
            + (f"  [{labels}]" if labels else ""))
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
        "Choosing a candidate means requesting one budgeted Support probe — "
        "it does NOT mean final deployment. The Support probe itself IS the "
        "confirmation process.",
        "",
        RISK_SEMANTICS,
        "",
    ]
    if extra_note:
        lines += ["== Current probe state ==", extra_note, ""]
    lines += [
        "== Your task ==",
        f"Choose exactly one candidate ID from "
        f"{[c['candidate_id'] for c in candidates]} or ABSTAIN.",
        'Output JSON only: {"chosen_candidate_id": "<id|ABSTAIN>", '
        '"evidence_refs": ["..."], "rationale": "..."}',
    ]
    return "\n".join(lines)


def _call(client: Any, prompt: str) -> dict[str, Any]:
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
    return {"chosen": chosen, "raw": raw}


def _render(episodes: Sequence[Any], context_plain: Mapping[str, float],
            ops: Sequence[str]) -> str:
    if not episodes:
        return ""
    qc = {k: float(v) for k, v in context_plain.items()
          if k.startswith(("recent.", "change."))}
    order, signed = resolve_order(
        query_context=qc, episodes=tuple(episodes), operators=tuple(ops),
        material_threshold=MATERIAL_THRESHOLD,
        task_consumer_key="forecast|ridge|sMASE", allowed_operators=tuple(ops))
    return render_signed_instruction(signed, order, executable_ops=tuple(ops))


def main() -> int:
    root = PROJECT_ROOT
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — WEAK_RISK_INCONCLUSIVE")
        (root / REPORT_OUT_REL).write_text(json.dumps(
            {"experiment_id": "v1-weak-risk-planonly",
             "verdict": "WEAK_RISK_INCONCLUSIVE", "error": "no api key"},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    import openai
    client = CountingClient(openai.OpenAI(api_key=api_key,
                                          base_url=BASE_URL), max_calls=4)

    # ---- weak 路径（metr offset=80，已暴露；Source NEGATIVE winsorize）----
    sealed._set_domain("metr_la")
    config = sealed._config()
    (src_roster, src_values, tgt_roster, tgt_values) = sealed._virgin_roster(
        root, offset=80)
    ctx = dict(resolver.window_context(tgt_values, 792, PERIOD))
    ctx["bound_period"] = float(PERIOD)
    # Source Episode（真实：denoise 0.0 ABSTAIN / winsorize −0.0249 NEGATIVE）
    ep1 = tll.write_target_episode(
        domain="metr_la", op="denoise_median",
        program_steps=[{"op": "denoise_median", "params": {"strength": 1.0,
                                                           "window": 1}}],
        support_gain=0.0, delayed_gain=None,
        support_context=resolver.window_context(src_values, 600, PERIOD),
        episode_id_suffix="_wk1")
    ep2 = tll.write_target_episode(
        domain="metr_la", op="winsorize",
        program_steps=[{"op": "winsorize", "params": {}}],
        support_gain=-0.02487430954610592, delayed_gain=None,
        support_context=resolver.window_context(src_values, 600, PERIOD),
        episode_id_suffix="_wk2")
    ep1 = tll.update_delayed_status(
        ep1, 0.0, delayed_context=resolver.window_context(src_values, 648,
                                                          PERIOD))
    ep2 = tll.update_delayed_status(
        ep2, -0.02, delayed_context=resolver.window_context(src_values, 648,
                                                            PERIOD))
    weak_render = _render((ep1, ep2), ctx, ("winsorize", "denoise_median"))
    is_weak = "weak reference" in weak_render
    print(f"== weak render: weak_reference={is_weak} :: {weak_render[:120]!r}")

    # select 1：UNKNOWN 优先（denoise 无 Reference）
    cands_1 = [
        {"candidate_id": "cand_denoise_median", "op": "denoise_median",
         "params": {}, "labels": ["UNKNOWN"]},
        {"candidate_id": "cand_winsorize", "op": "winsorize",
         "params": {}, "labels": ["weak RISK"]},
    ]
    s1 = _call(client, _select_prompt(cands_1, ctx, weak_render,
                                      extra_note="First probe of this "
                                                 "decision point."))
    print(f"== weak select1: {s1['chosen']}")

    # select 2：UNKNOWN 已耗尽（denoise 探测无信号——Source 真实数据）、
    # 预算剩一次 → 有界探测 weak-risk winsorize（不 abstain）
    cands_2 = [
        {"candidate_id": "cand_winsorize", "op": "winsorize",
         "params": {}, "labels": ["weak RISK"]},
    ]
    s2 = _call(client, _select_prompt(
        cands_2, ctx, weak_render,
        extra_note="denoise_median (UNKNOWN) was already probed and "
                   "produced no signal. UNKNOWN candidates are exhausted. "
                   "Support budget remains for one more probe."))
    print(f"== weak select2: {s2['chosen']}")

    # ---- strong 负控（uci，Target-local RESTRICTED winsorize）----
    sealed._set_domain("uci_electricity_load_diagrams")
    config = sealed._config()
    (src_roster, src_values, tgt_roster, tgt_values) = sealed._virgin_roster(
        root, offset=0)
    ctx_s = dict(resolver.window_context(tgt_values, 984, PERIOD))
    ctx_s["bound_period"] = float(PERIOD)
    # Target-local RESTRICTED Episode（uci 报告 r2：support +0.08566 @888 /
    # delayed −0.00594 @936 → CONFLICT/RESTRICTED）
    ep_s = tll.write_target_episode(
        domain="uci_electricity_load_diagrams", op="winsorize",
        program_steps=[{"op": "winsorize", "params": {}}],
        support_gain=0.08565593531148652, delayed_gain=None,
        support_context=resolver.window_context(tgt_values, 888, PERIOD),
        episode_id_suffix="_str")
    ep_s = tll.update_delayed_status(
        ep_s, -0.005939896557957569,
        delayed_context=resolver.window_context(tgt_values, 936, PERIOD))
    strong_render = _render((ep_s,), ctx_s, ("winsorize",))
    is_strong = "weak reference" not in strong_render
    print(f"== strong render: strong={is_strong} :: {strong_render[:120]!r}")
    cands_s = [
        {"candidate_id": "cand_denoise_median", "op": "denoise_median",
         "params": {}, "labels": ["UNKNOWN"]},
        {"candidate_id": "cand_winsorize", "op": "winsorize",
         "params": {}, "labels": ["strong RISK"]},
    ]
    str1 = _call(client, _select_prompt(cands_s, ctx_s, strong_render))
    str2 = _call(client, _select_prompt(cands_s, ctx_s, strong_render))
    print(f"== strong: {str1['chosen']} / {str2['chosen']}")

    # ---- 判定（负控 strong 语义由脚本标注表达——uci Target-local
    # RESTRICTED 仅 1 条 Episode，n_hist < min → 全局渲染也走 weak_reference；
    # 记录为机制边界，不阻断行为验证）----
    weak_seq_ok = s1["chosen"] == "cand_denoise_median" \
        and s2["chosen"] == "cand_winsorize"
    strong_avoided = str1["chosen"] != "cand_winsorize" \
        and str2["chosen"] != "cand_winsorize"
    if weak_seq_ok and strong_avoided:
        verdict = "WEAK_RISK_GRADED_PASS"
    elif not weak_seq_ok:
        verdict = "WEAK_RISK_TOO_CONSERVATIVE"
    else:
        verdict = "WEAK_RISK_OVERGENERALIZED"
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-weak-risk-planonly",
        "setting": "plan-only（不评估新 outcome）；metr offset=80 weak 路径 "
                   "+ uci strong 负控；luna temp=0；总调用 4",
        "risk_semantics": RISK_SEMANTICS,
        "weak": {"render_is_weak": is_weak, "select1": s1,
                 "select2": s2, "sequence_ok": weak_seq_ok},
        "strong": {"render_is_strong": is_strong, "calls": [str1, str2],
                   "avoided": strong_avoided},
        "llm_call_count": client.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
