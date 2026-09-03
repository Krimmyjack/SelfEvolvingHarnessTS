"""实验 4：真实 LLM signed selection（审核 2026-08-09）。

只检验一个问题：

> 在候选、Context 和顺序完全相同时，正向/移除/冲突 Memory 是否因果改变
> LLM 的选择？

设置（已暴露 UCI @984，不消耗新 sealed 数据）：
  - 固定候选（冻结，不经 propose）：cand_skill_winsorize（winsorize）、
    cand_outlier_iqr（outlier_iqr）、ABSTAIN（identity）；
  - 固定候选顺序、Context（uci @984 window_context）、模型（gpt-5.6-luna）、
    temperature=0、prompt 主体；
  - 唯一差异（三干预）：
      M_positive：UCI delayed 打开前 winsorize POSITIVE/LOCAL_ACTIVE
        （support +0.05155 @792 / delayed +0.04327 @840——uci 报告 r1）；
      M_remove：无 signed Episode，保留同一 Skill 候选；
      M_conflict：真实 delayed 后 CONFLICT/RESTRICTED
        （support +0.08566 @888 / delayed −0.00594 @936——uci 报告 r2）；
  - Memory 表达经真实机制：resolve_order + render_signed_instruction
    （与 fast_agent.prepare 相同路径）——不手工写 Memory 文本；
  - deterministic inspect/propose（候选冻结）；真实 LLM 只负责 select；
  - luna temperature=0，3 次调用，硬上限 4（CountingClient）；
  - plan-only：不评估任何 gain。

判定（预注册）：
  - FULL_PASS：positive 选 Skill、remove 不选 Skill、conflict 不选 Skill；
  - SIGN_SENSITIVE_PASS：positive 选 Skill、conflict 不选（remove 不限）；
  - MEMORY_INSENSITIVE：positive 与 conflict 选择相同；
  - INCONCLUSIVE：格式、接口或调用失败。

报告断言：三个请求的候选 ID、顺序和公开 Context 逐项相同。

用法：
  python evaluation/functional/run_v1_llm_signed_selection.py
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
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.signed_radius import (  # noqa: E402
    MATERIAL_THRESHOLD, render_signed_instruction, resolve_order)

MODEL = "gpt-5.6-luna"
BASE_URL = "https://api.agicto.cn/v1"
KEY_ENVS = ("OPENAI_API_KEY", "AGICTO_API_KEY")
PERIOD = 24
UCI_REPORT = Path(
    "artifacts/functional/e2/w1_sealed_a5_a3_uci_electricity_load_diagrams_0"
    "_report.json")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_llm_signed_selection_report.json")

CAND_SKILL = "cand_skill_winsorize"
CAND_OUTLIER = "cand_outlier_iqr"
ABSTAIN = "ABSTAIN"
LEGAL = (CAND_SKILL, CAND_OUTLIER, ABSTAIN)


class CountingClient:
    """temperature 0 + 调用计数（硬上限 4：3 干预 + 1 格式纠正余量）。"""

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


def _episode(domain: str, op: str, support_gain: float, delayed_gain: float,
             delayed_origin: int, values: Mapping[str, Any],
             support_origin: int, suffix: str) -> Any:
    ep = tll.write_target_episode(
        domain=domain, op=op, episode_id_suffix=suffix,
        program_steps=[{"op": op, "params": {}}],
        support_gain=support_gain, delayed_gain=None,
        support_context=resolver.window_context(values, support_origin, PERIOD))
    return tll.update_delayed_status(
        ep, delayed_gain,
        delayed_context=resolver.window_context(values, delayed_origin, PERIOD))


def _build_memory(report: Mapping[str, Any], values: Mapping[str, Any],
                  arm: str) -> tuple[Any, ...]:
    """从 uci 报告重建三干预 Memory（真实 Episode 对象）。"""
    a5 = report["arms"][arm]
    r1_d = a5["r1_delayed"]
    r2_d = a5["r2_delayed"]
    if arm == "a5":
        m_pos = _episode(
            "uci_electricity_load_diagrams", "winsorize",
            float(a5["r1"]["probes"][0]["gain"]), float(r1_d["delayed_gain"]),
            int(r1_d["delayed_origin"]), values, 792, "_m4_positive")
        m_conf = _episode(
            "uci_electricity_load_diagrams", "winsorize",
            float(a5["r2"]["probes"][0]["gain"]), float(r2_d["delayed_gain"]),
            int(r2_d["delayed_origin"]), values, 888, "_m4_conflict")
        return (m_pos, m_conf)
    raise ValueError(f"unknown arm {arm}")


def _render_memory(episodes: Sequence[Any], query_context: Mapping[str, float],
                   operators: Sequence[str]) -> str:
    """真实机制渲染（与 fast_agent.prepare 相同路径）：无 Episode → 空。"""
    if not episodes:
        return ""
    order, signed = resolve_order(
        query_context=query_context, episodes=tuple(episodes),
        operators=tuple(operators), material_threshold=MATERIAL_THRESHOLD,
        task_consumer_key=("forecast|ridge|sMASE"
                           if hasattr(resolver, "TASK_CONSUMER_KEY") else
                           "forecast|ridge|sMASE"),
        allowed_operators=tuple(operators))
    return render_signed_instruction(signed, order,
                                     executable_ops=tuple(operators))


def _select_prompt(candidates: Sequence[Mapping[str, Any]],
                   context_plain: Mapping[str, float],
                   memory_rendered: str) -> str:
    lines = [
        "You are the fast-path selector of a time-series preprocessing "
        "harness.",
        "You must choose exactly one candidate to execute at support decision "
        "point 984.",
        "",
        "== Candidate programs (fixed order) ==",
    ]
    for i, cand in enumerate(candidates, start=1):
        op = cand["op"]
        lines.append(
            f"  {i}. {cand['candidate_id']}: {op} params="
            f"{json.dumps(cand['params'], sort_keys=True)}")
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
        "== Your task ==",
        f"Choose exactly one candidate ID from {list(candidates)} or ABSTAIN. "
        "ABSTAIN is valid when evidence is insufficient or contradictory — "
        "abstaining is not a failure.",
        'Output JSON only: {"chosen_candidate_id": "<id|ABSTAIN>", '
        '"evidence_refs": ["..."], "rationale": "..."}',
    ]
    return "\n".join(lines)


def _llm_select(client: Any, prompt: str, *, max_calls: int = 1) -> dict[str, Any]:
    """一次选择 + 机器校验（chosen ∈ 合法集）；非法 → 重试；超限硬停。"""
    attempts = []
    for i in range(max_calls):
        kwargs = {"model": MODEL,
                  "messages": [{"role": "user", "content": prompt}]}
        try:
            resp = client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"})
        except Exception:
            resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        attempts.append(raw)
        try:
            payload = json.loads(raw)
            chosen = str(payload.get("chosen_candidate_id", ""))
        except json.JSONDecodeError:
            chosen = ""
        if chosen in LEGAL:
            return {"ok": True, "chosen": chosen, "raw": raw, "attempts": attempts}
    return {"ok": False, "chosen": None, "raw": attempts[-1] if attempts else "",
            "attempts": attempts}


def main() -> int:
    root = PROJECT_ROOT
    report_in = json.loads((root / UCI_REPORT).read_text(encoding="utf-8"))
    # uci cohort values（已暴露——不消费新 sealed 数据）
    _set_domain_uci()
    (src_roster, src_values, tgt_roster, tgt_values) = sealed._virgin_roster(
        root, offset=0)
    series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                         dtype=np.float64)

    # 固定候选（冻结顺序）
    candidates = [
        {"candidate_id": CAND_SKILL, "op": "winsorize", "params": {}},
        {"candidate_id": CAND_OUTLIER, "op": "outlier_iqr", "params": {}},
    ]
    ops = ("winsorize", "outlier_iqr")
    context_plain = dict(resolver.window_context(tgt_values, 984, PERIOD))
    context_plain["bound_period"] = float(PERIOD)

    # 三干预 Memory
    m_pos, m_conf = _build_memory(report_in, tgt_values, arm="a5")
    memories = {
        "M_positive": (m_pos,),
        "M_remove": (),
        "M_conflict": (m_conf,),
    }
    # 真实渲染（resolve_order + render_signed_instruction）
    rendered = {
        arm: _render_memory(eps, context_plain, ops)
        for arm, eps in memories.items()
    }
    for arm, text in rendered.items():
        print(f"== {arm}: rendered={text[:160]!r}")

    # LLM select（确定性 inspect/propose = 冻结候选；真实 LLM 只负责 select）
    api_key = next((os.environ.get(k, "").strip() for k in KEY_ENVS
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print(f"== {' or '.join(KEY_ENVS)} required — INCONCLUSIVE")
        out = root / REPORT_OUT_REL
        out.write_text(json.dumps({
            "experiment_id": "v1-llm-signed-selection",
            "verdict": "INCONCLUSIVE",
            "error": "no api key",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    import openai
    client = CountingClient(openai.OpenAI(api_key=api_key, base_url=BASE_URL),
                            max_calls=4)

    prompts = {arm: _select_prompt(candidates, context_plain, rendered[arm])
               for arm in memories}
    results: dict[str, Any] = {}
    for arm, prompt in prompts.items():
        res = _llm_select(client, prompt)
        results[arm] = res
        print(f"== {arm}: chosen={res['chosen']} ok={res['ok']}")

    # 断言：三请求候选 ID/顺序/Context 逐项相同（程序化：prompt 除
    # "Experience memory" 段外逐字节相同）
    mem_marker = "== Experience memory"
    bodies = []
    for arm, prompt in prompts.items():
        idx = prompt.find(mem_marker)
        assert idx >= 0
        bodies.append((prompt[:idx], prompt[idx + len(mem_marker):]))
    prefix_same = len({b[0] for b in bodies}) == 1
    task_suffix = [b[1].split("==")[0] for b in bodies]
    task_same = len({b[1][b[1].find("== Your task"):] for b in bodies}) == 1
    ctx_blob = json.dumps({k: round(float(v), 6) for k, v in
                           sorted(context_plain.items())}, sort_keys=True)
    requests_identical = {
        "candidate_ids_same_order": bool(prefix_same),
        "candidate_order": [c["candidate_id"] for c in candidates],
        "context_identical": bool(prefix_same and task_same),
        "context_json": json.loads(ctx_blob),
        "only_difference_is_memory_rendering": bool(prefix_same and task_same),
    }

    # 判定（预注册）
    ok_all = all(r["ok"] for r in results.values())
    pos_choice = results["M_positive"]["chosen"]
    rem_choice = results["M_remove"]["chosen"]
    con_choice = results["M_conflict"]["chosen"]
    if not ok_all:
        verdict = "INCONCLUSIVE"
    elif pos_choice == CAND_SKILL and con_choice != CAND_SKILL \
            and rem_choice != CAND_SKILL:
        verdict = "FULL_PASS"
    elif pos_choice == CAND_SKILL and con_choice != CAND_SKILL:
        verdict = "SIGN_SENSITIVE_PASS"
    elif pos_choice == con_choice:
        verdict = "MEMORY_INSENSITIVE"
    else:
        verdict = "MEMORY_INSENSITIVE"  # positive/conflict 无区分
    print(f"== choices: positive={pos_choice} remove={rem_choice} "
          f"conflict={con_choice}")
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-llm-signed-selection",
        "setting": "UCI @984（已暴露，不消费新 sealed 数据）；候选冻结 "
                   "[winsorize skill, outlier_iqr] + ABSTAIN；luna temp=0；"
                   "plan-only（不评估 gain）",
        "candidates": candidates,
        "requests_identical": requests_identical,
        "memory_rendering": {k: v for k, v in rendered.items()},
        "results": results,
        "choices": {"M_positive": pos_choice, "M_remove": rem_choice,
                    "M_conflict": con_choice},
        "llm_call_count": client.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


def _set_domain_uci() -> None:
    sealed._set_domain("uci_electricity_load_diagrams")


if __name__ == "__main__":
    raise SystemExit(main())
