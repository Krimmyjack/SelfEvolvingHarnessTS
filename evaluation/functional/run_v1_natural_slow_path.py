"""自然 Slow Path 切片（审核裁决修订版，docs/V1_NATURAL_SLOW_PATH_SLICE.md）。

只检验一个问题：

> LLM 能否根据自然出现的 Program-level Support 失败和确定性反事实证据，
> 生成一个经未来反馈确认、并被下一轮正常入口实际执行的 Harness Update？

固定时间链（R2 不用于寻找第二个失败）：
  R1 @792：自然生成 A→B（确定性组合探测，预算 ≤2）→ Support 负向触发
  → identity/A/B/A→B 反事实 → headroom（max(A,B)>=M 且 max-gain_AB>=M）
  → LLM Typed Patch 冻结（KEEP/REMOVE_A/REMOVE_B/ABSTAIN，信息墙）
  → @840 delayed 验证 → 成立后写 Target-local Skill（KEEP/ABSTAIN 不写）
  → R2 @888 正常入口实际选择并执行 Skill（evaluate chosen，非池首位）
  → @936 delayed 更新

阈值（审核修订）：headroom 公式如上；Patch 有效 = gain_patch >= M；
"池首位"只证明供给。无自然失败 → NO_NATURAL_FAILURE（不换 origin）。

明确不是 A5/A3 比较：无 Source Memory 注入；唯一承重变量 = Slow Path
是否把自然失败转成有效 Typed Program Update。

零 Source Memory；LLM 调用 ≤2（agicto gpt-5.6-luna temp=0）。

用法：
  python evaluation/functional/run_v1_natural_slow_path.py
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
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

DOMAIN = "monash:traffic_hourly"
OFFSET = 80
PERIOD = 24
HORIZON = 48
R1_ORIGIN = 792
R1_DELAYED = 840
R2_ORIGIN = 888
R2_DELAYED = 936
MATERIAL = resolver.MATERIAL_THRESHOLD  # 0.005
MODEL = "gpt-5.6-luna"
BASE_URL = "https://api.agicto.cn/v1"
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_natural_slow_path_report.json")

LEGAL_PATCH = ("KEEP", "REMOVE_A", "REMOVE_B", "ABSTAIN")


class CountingClient:
    def __init__(self, delegate: Any, *, max_calls: int = 2) -> None:
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


def _patch_prompt(a: str, b: str, context_plain: Mapping[str, float],
                 counter: Mapping[str, Any]) -> str:
    lines = [
        "You are the slow attribution path of a time-series preprocessing "
        "harness.",
        "A two-step Workflow was applied at support decision point 792 and "
        "produced a negative Support outcome. You must decide the update "
        "direction by choosing exactly one legal patch ID.",
        "",
        "== Source Workflow (A then B) ==",
        f"  A: {a} params={json.dumps(wiring.contract_params(a, PERIOD), sort_keys=True)}",
        f"  B: {b} params={json.dumps(wiring.contract_params(b, PERIOD), sort_keys=True)}",
        "",
        "== Public context at the decision point (deployment-visible) ==",
        json.dumps({k: round(float(v), 6) for k, v in sorted(
            context_plain.items())}, indent=2, sort_keys=True),
        "",
        "== Counterfactual Support outcomes (training-window cohort, "
        "sMASE gain; positive = better than no preprocessing) ==",
        "  identity (no preprocessing): baseline (gain 0 by definition)",
        f"  A only: {counter['A_only']:+.5f}",
        f"  B only: {counter['B_only']:+.5f}",
        f"  A then B: {counter['A_to_B']:+.5f}",
        "",
        "== Legal patch IDs ==",
        "  KEEP      keep the two-step Workflow A -> B",
        "  REMOVE_A  drop step A, keep only step B",
        "  REMOVE_B  drop step B, keep only step A",
        "  ABSTAIN   no change proposed (evidence insufficient)",
        "",
        "== Your task ==",
        "Choose exactly one legal patch ID. If the stepwise evidence is "
        "insufficient or contradictory, ABSTAIN is valid — abstaining is "
        "not a failure.",
        'Output JSON only: {"patch_id": "<one legal ID>", '
        '"evidence_refs": ["..."], "rationale": "..."}',
    ]
    return "\n".join(lines)


def _llm_patch(client: Any, prompt: str, *, max_calls: int = 2) -> dict[str, Any]:
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
            patch = str(json.loads(raw).get("patch_id", ""))
        except json.JSONDecodeError:
            patch = ""
        if patch in LEGAL_PATCH:
            return {"ok": True, "patch_id": patch, "raw": raw,
                    "attempts": attempts}
    return {"ok": False, "patch_id": None, "raw": attempts[-1] if attempts else "",
            "attempts": attempts}


PREMISE_COHORTS = [
    ("monash:traffic_hourly", 80),
    ("uci_electricity_load_diagrams", 0),
    ("metr_la", 80),
    ("metr_la", 120),
]


def _premise_scan(root: Path, domain: str, offset: int,
                  origin: int = R1_ORIGIN) -> dict[str, Any]:
    """一 cohort 的自然两步 premise 扫描（零 LLM，已暴露数据）：
    自然 proposal 顺序前 ≤2 个两步候选；gain_AB < −M 才反事实；
    headroom 公式同主实验。"""
    sealed._set_domain(domain)
    config = sealed._config()
    (src_roster, src_values, tgt_roster, tgt_values) = sealed._virgin_roster(
        root, offset=offset)
    tgt_series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                             dtype=np.float64)
    executor = ScopeExecutor(tgt_roster, tgt_values, config,
                             evaluate_fn=sealed.v6._evaluate)
    observed = dict(resolver.window_context(tgt_values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    ops = sealed._actionable_ops(root, tgt_series0, origin, observed)
    combos = [(a, b) for i, a in enumerate(ops) for b in ops[i + 1:]]
    probes: list[dict[str, Any]] = []
    headroom_ok = False
    for i, (a, b) in enumerate(combos[:2]):  # 预算 ≤2，预注册顺序不换 pair
        steps = ((a, dict(wiring.contract_params(a, PERIOD))),
                 (b, dict(wiring.contract_params(b, PERIOD))))
        r = executor.evaluate(steps, origin)
        gain_ab = (float(r.gain) if r.gain is not None else None)
        entry = {"proposal": i + 1, "a": a, "b": b, "gain_AB": gain_ab,
                 "passed": r.verification.passed}
        if gain_ab is not None and gain_ab < -MATERIAL:
            # 反事实（只失败才评估 A/B 单步）
            ra = executor.evaluate(((a, dict(wiring.contract_params(a, PERIOD))),), origin)
            rb = executor.evaluate(((b, dict(wiring.contract_params(b, PERIOD))),), origin)
            ga = (float(ra.gain) if ra.gain is not None else None)
            gb = (float(rb.gain) if rb.gain is not None else None)
            best = max(ga, gb) if (ga is not None and gb is not None) else None
            ok = bool(best is not None and best >= MATERIAL
                      and (best - gain_ab) >= MATERIAL)
            entry.update({"trigger": True, "A_only": ga, "B_only": gb,
                          "best_single": best, "headroom_ok": ok})
            headroom_ok = headroom_ok or ok
        probes.append(entry)
    return {"domain": domain, "offset": offset, "origin": origin,
            "actionable": list(ops), "probes": probes,
            "headroom_ok": headroom_ok}


def _premise_main(root: Path) -> int:
    """NATURAL_PROGRAM_INTERACTION_PREMISE：4 已暴露 cohort 有界扫描，
    零 LLM，判定 removable-step headroom 出现率。"""
    results = [_premise_scan(root, d, o) for d, o in PREMISE_COHORTS]
    for res in results:
        rows = [(p["a"], p["b"], round(p["gain_AB"], 4), p.get("trigger"),
                 p.get("headroom_ok")) for p in res["probes"]]
        print(f"== {res['domain']}@{res['offset']}: probes={rows}")
    supported = [r for r in results if r["headroom_ok"]]
    n = len(supported)
    if n >= 2:
        verdict = "PROGRAM_REMOVAL_PREMISE_SUPPORTED"
    elif n == 1:
        verdict = "PROGRAM_REMOVAL_PREMISE_SPARSE"
    else:
        verdict = "PROGRAM_REMOVAL_PREMISE_NOT_SUPPORTED"
    print(f"== cohorts with removable-step headroom: {n}/4 → {verdict}")
    out = root / Path("artifacts/functional/e2/w1_program_interaction_premise_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-natural-program-interaction-premise",
        "setting": "4 已暴露 cohort（traffic 80 / uci 0 / metr 80 / metr "
                   "120）；每 origin ≤2 自然两步候选（预注册顺序不换 "
                   "pair）；gain_AB<-M 才反事实；零 LLM；不消费 virgin",
        "cohorts": results,
        "supported_count": n,
        "verdict": verdict,
        "llm_api_call_count": 0,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> "
          f"artifacts/functional/e2/w1_program_interaction_premise_report.json")
    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="natural Slow Path / premise")
    parser.add_argument("--premise", action="store_true",
                        help="premise 扫描模式（4 已暴露 cohort，零 LLM）")
    args = parser.parse_args()
    root = PROJECT_ROOT
    if args.premise:
        return _premise_main(root)
    sealed._set_domain(DOMAIN)
    config = sealed._config()
    (src_roster, src_values, tgt_roster, tgt_values) = sealed._virgin_roster(
        root, offset=OFFSET)
    tgt_series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                             dtype=np.float64)
    executor = ScopeExecutor(tgt_roster, tgt_values, config,
                             evaluate_fn=sealed.v6._evaluate)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    observed = dict(resolver.window_context(tgt_values, R1_ORIGIN, PERIOD))
    observed["bound_period"] = float(PERIOD)

    # ---- R1：自然两步组合探测（预算 ≤2，预注册顺序不挑失败）----
    ops = sealed._actionable_ops(root, tgt_series0, R1_ORIGIN, observed)
    print(f"== actionable @{R1_ORIGIN}: n={len(ops)} {ops}")
    combos = [(a, b) for i, a in enumerate(ops) for b in ops[i + 1:]]
    print(f"== natural combos (ordered): {combos[:6]} ... total={len(combos)}")

    trigger: dict[str, Any] | None = None
    probes: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(combos[:2]):  # 预算 ≤2
        steps = ((a, dict(wiring.contract_params(a, PERIOD))),
                 (b, dict(wiring.contract_params(b, PERIOD))))
        r = executor.evaluate(steps, R1_ORIGIN)
        gain = (float(r.gain) if r.gain is not None else None)
        entry = {"proposal": i + 1, "a": a, "b": b, "gain": gain,
                 "passed": r.verification.passed}
        probes.append(entry)
        print(f"== probe {i + 1}: {a}->{b} gain={gain} passed={r.verification.passed}")
        if gain is not None and gain < -MATERIAL:
            trigger = {"a": a, "b": b, "gain_AB": gain, "steps": steps,
                       "proposal": i + 1}
            break

    if trigger is None:
        verdict = "NO_NATURAL_FAILURE"
        print(f"== verdict: {verdict}（R1 预算内无自然 Support NEGATIVE——"
              f"可信负结果，不换 origin）")
        out = root / REPORT_OUT_REL
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "experiment_id": "v1-natural-slow-path",
            "dataset": DOMAIN, "cohort_offset": OFFSET,
            "probes": probes, "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"== report -> {out.relative_to(root)}")
        return 0

    a, b = trigger["a"], trigger["b"]
    steps_ab = trigger["steps"]
    steps_a = ((a, dict(wiring.contract_params(a, PERIOD))),)
    steps_b = ((b, dict(wiring.contract_params(b, PERIOD))),)

    # ---- 反事实表（identity/A/B/A→B；确定性评估）----
    r_a = executor.evaluate(steps_a, R1_ORIGIN)
    r_b = executor.evaluate(steps_b, R1_ORIGIN)
    gain_a = (float(r_a.gain) if r_a.gain is not None else None)
    gain_b = (float(r_b.gain) if r_b.gain is not None else None)
    counter = {"identity": 0.0, "A_only": gain_a, "B_only": gain_b,
               "A_to_B": trigger["gain_AB"]}
    print(f"== counterfactuals: {counter}")

    # ---- headroom（审核修订公式）----
    best_single = max(gain_a, gain_b) if (gain_a is not None
                                          and gain_b is not None) else None
    headroom_ok = bool(
        best_single is not None
        and best_single >= MATERIAL
        and (best_single - trigger["gain_AB"]) >= MATERIAL)
    print(f"== headroom: best_single={best_single} ok={headroom_ok}")
    if not headroom_ok:
        verdict = "NO_SINGLE_STEP_HEADROOM"
        print(f"== verdict: {verdict}")
        out = root / REPORT_OUT_REL
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "experiment_id": "v1-natural-slow-path",
            "dataset": DOMAIN, "cohort_offset": OFFSET,
            "trigger": trigger, "counterfactuals": counter,
            "headroom": {"best_single": best_single, "ok": False},
            "verdict": verdict, "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"== report -> {out.relative_to(root)}")
        return 0

    # ---- LLM Typed Patch（信息墙：反事实 Support 表 + Context，无 delayed）----
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        verdict = "INCONCLUSIVE"
        print(f"== verdict: {verdict} (no api key)")
        return 0
    import openai
    client = CountingClient(openai.OpenAI(api_key=api_key,
                                          base_url=BASE_URL), max_calls=2)
    patch = _llm_patch(client, _patch_prompt(a, b, observed, counter))
    print(f"== LLM patch: {patch.get('patch_id')} ok={patch.get('ok')}")

    if not patch["ok"]:
        verdict = "INCONCLUSIVE"
    elif patch["patch_id"] == "ABSTAIN":
        verdict = "ABSTAIN_NO_UPDATE"  # 安全行为，不形成 Skill
    elif patch["patch_id"] == "KEEP":
        verdict = "PATCH_REPLAY_FAILED"  # KEEP 不形成新 Skill（原失败未改善）
    else:
        # REMOVE_A → B-only；REMOVE_B → A-only
        patch_steps = steps_b if patch["patch_id"] == "REMOVE_A" else steps_a
        # ---- verifier + Support replay（Patch 有效 = gain_patch >= M）----
        rp = executor.evaluate(patch_steps, R1_ORIGIN)
        gain_patch = (float(rp.gain) if rp.gain is not None else None)
        patch_ok = bool(rp.verification.passed and gain_patch is not None
                        and gain_patch >= MATERIAL)
        print(f"== replay: gain_patch={gain_patch} passed="
              f"{rp.verification.passed} ok={patch_ok}")
        if not patch_ok:
            verdict = "PATCH_REPLAY_FAILED"
        else:
            # ---- @840 delayed 验证（patch Workflow；Skill 状态依据）----
            rd = executor.evaluate(patch_steps, R1_DELAYED)
            gain_delayed = (float(rd.gain) if rd.gain is not None else None)
            status = ("LOCAL_ACTIVE" if gain_delayed is not None
                      and gain_delayed >= MATERIAL else "RESTRICTED")
            print(f"== delayed @{R1_DELAYED}: gain={gain_delayed} "
                  f"status={status}")

            # ---- Skill 写盘（fork + learned）----
            skill_id = f"{patch_steps[0][0][:8]}-nsp-v1"
            patched, store, fork_root = sealed.write_skill(
                root, h0, patch_steps, skill_id, status,
                rationale=f"natural Slow Path @{R1_ORIGIN}: remove "
                          f"{'A' if patch['patch_id'] == 'REMOVE_A' else 'B'} "
                          f"({a}->{b} failed, gain_AB="
                          f"{trigger['gain_AB']:.5f}, best single "
                          f"{best_single:.5f})")
            print(f"== skill written: {skill_id} status={status}")

            # ---- R2 @888 正常入口实际选择并执行 Skill（非池首位）----
            method = sealed.TTHAMethod(
                sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                    sealed.SealedProbeBackend(
                        explore=True, operators=ops),
                    LocalPublicToolGateway(tgt_series0[:R2_ORIGIN],
                                           task_kind="forecast"))),
                patched, ())
            method.bind_round_data(tgt_series0[:R2_ORIGIN],
                                   task_kind="forecast")
            r2_observed = dict(resolver.window_context(tgt_values, R2_ORIGIN,
                                                       PERIOD))
            r2_observed["bound_period"] = float(PERIOD)
            r2_result = method.prepare(sealed._request(tgt_series0, tgt_values,
                                                       R2_ORIGIN))
            chosen_r2 = method.last_trace.chosen_candidate_id
            skill_cand = f"cand_skill_{skill_id}"
            adopted = chosen_r2 == skill_cand
            r2_gain = None
            if r2_result.program is not None:
                chosen_steps = tuple((op, dict(pr))
                                     for op, pr in
                                     r2_result.program.execution_steps())
                rr2 = executor.evaluate(chosen_steps, R2_ORIGIN)
                r2_gain = (float(rr2.gain) if rr2.gain is not None else None)
            print(f"== R2 @{R2_ORIGIN}: chosen={chosen_r2} adopted={adopted} "
                  f"gain={r2_gain}")

            # ---- @936 delayed 更新（Skill delayed utility）----
            rd2 = executor.evaluate(patch_steps, R2_DELAYED)
            gain_d2 = (float(rd2.gain) if rd2.gain is not None else None)
            print(f"== R2 delayed @{R2_DELAYED}: gain={gain_d2}")

            verdict = ("NATURAL_SLOW_PATH_UPDATE_PASS" if adopted else
                       "NATURAL_SLOW_PATH_UPDATE_PARTIAL")
            store.discard_fork(fork_root)

    print(f"== verdict: {verdict}")
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-natural-slow-path",
        "dataset": DOMAIN, "cohort_offset": OFFSET,
        "probes": probes, "trigger": trigger,
        "counterfactuals": counter,
        "headroom": {"best_single": best_single, "ok": headroom_ok},
        "patch": patch,
        "replay": ({"gain_patch": gain_patch, "passed": rp.verification.passed}
                   if "gain_patch" in dir() else None),
        "delayed": ({"gain": gain_delayed, "status": status}
                    if "gain_delayed" in dir() else None),
        "skill": ({"skill_id": skill_id, "status": status}
                  if "skill_id" in dir() else None),
        "r2": ({"chosen": chosen_r2, "adopted": adopted, "gain": r2_gain}
               if "chosen_r2" in dir() else None),
        "r2_delayed": ({"gain": gain_d2} if "gain_d2" in dir() else None),
        "verdict": verdict,
        "llm_api_call_count": (client.calls if "client" in dir() else 0),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
