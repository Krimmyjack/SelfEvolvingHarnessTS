"""TS-native Slow Agent 开发测试（§7 四十六，用户修订版）。

SLOW_PROGRAM_SUPPLY_TO_EXECUTABLE_SKILL：Slow Agent 输入升级为
  Task/Consumer objective + TS Data Quality Objective（部署可见作用几何/
  窗口观察/Consumer 依赖）+ Grounded Contrast Capsule（成功 vs 失败
  Context 的相同/不同观察）+ 失败 Program + Operator DSL。

Slow Agent 第一判断（诊断分流）：
  - PROGRAM_DEFECT：生成 ≤2 个现有 DSL 内 Typed Program；
  - EXECUTION_LAPSE：保留 Skill 不修改 Program；
  - UNIDENTIFIABLE：请求 1 个定向 Diagnostic Probe 或 abstain；
  - 无动作 → ABSTAIN / ACTION_UNAVAILABLE。

开发测试 case（GEFCom winsorize @928 暴露；**outlier_iqr 正控隐藏**——
prompt 不提及任何"答案"）：
  winsorize @832 +0.14403（成功 Context）/ @928 −0.16360（失败）/
  @976 +0.61000（delayed）。
  正控：outlier_iqr @928 +0.04386 / @976 +0.02719（LLM 不可见）。

验收 6 条件（仅 PROGRAM_DEFECT 路径）：
  ① LLM 自主生成合法候选（Operator DSL 内、verifier 可行动）；
  ② Support 正向（gain@928 >= M）；
  ③ delayed 不翻转（gain@976 >= −M）；
  ④ 写成 Target-local Skill；
  ⑤ 下一轮正常入口实际采用（@976 chosen=cand_skill_* 且执行）；
  ⑥ 移除该 Skill 时行动发生对应变化（H0 prepare @976 chosen != skill）。

verdict（审核第六轮降级）：本 case 的准确口径 =
  TSNATIVE_DIAGNOSTIC_PROMPT_SAFE_ABSTAIN_OBSERVED——证明一次真实 LLM
  调用在当前输入下选择了 UNIDENTIFIABLE + abstain + 不生成 Program；
  不证明三路分流机制正确、不证明 Slow Agent 具备有效归因能力。
  三路分支修复：EXECUTION_LAPSE / UNIDENTIFIABLE（无 programs）/
  PROGRAM_DEFECT（有 programs）各自独立档位；PROGRAM_DEFECT+programs=[]
  单独记 PROGRAM_DEFECT_NO_PROGRAMS（不再误落 UNIDENTIFIABLE）。

LLM 调用 ≤2（agicto gpt-5.6-luna temp=0）。

用法：
  python evaluation/functional/run_v1_slow_agent_tsnative.py
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

PERIOD = 24
HORIZON = 48
MATERIAL = resolver.MATERIAL_THRESHOLD  # 0.005
MODEL = "gpt-5.6-luna"
BASE_URL = "https://api.agicto.cn/v1"
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_slow_agent_tsnative_report.json")

SUCCESS_ORIGIN = 832
FAIL_ORIGIN = 928
DELAYED_ORIGIN = 976


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


def _contrast(values: Mapping[str, Any], ok_origin: int, fail_origin: int,
              period: int) -> dict[str, Any]:
    """Grounded Contrast Capsule：成功 vs 失败 Context 的相同/不同观察
    （确定性提取，不依赖 LLM）。"""
    ok = dict(resolver.window_context(values, ok_origin, period))
    fail = dict(resolver.window_context(values, fail_origin, period))
    same: list[str] = []
    diff: dict[str, tuple[float, float]] = {}
    for k in sorted(set(ok) & set(fail)):
        a, b = float(ok[k]), float(fail[k])
        if abs(a - b) < 1e-3:
            same.append(k)
        elif abs(a - b) >= 1e-3:
            diff[k] = (round(a, 6), round(b, 6))
    return {"ok_origin": ok_origin, "fail_origin": fail_origin,
            "same_observations": same, "different_observations": diff,
            "ok_window": {k: round(float(v), 6) for k, v in sorted(ok.items())},
            "fail_window": {k: round(float(v), 6) for k, v in sorted(fail.items())}}


def _slow_prompt(capsule: Mapping[str, Any], dq: Mapping[str, Any],
                 failed: str, dsl: Sequence[str]) -> str:
    lines = [
        "You are the slow attribution path of a time-series preprocessing "
        "harness.",
        "A Program was applied and produced a negative Support outcome. "
        "Diagnose the failure and, if justified, propose bounded Typed "
        "Programs from the existing operator DSL.",
        "",
        "== Task / Consumer objective ==",
        "  downstream: ridge regression forecast, metric sMASE (lower is "
        "better)",
        "  The Consumer depends on level and seasonal pattern integrity; "
        "outlier contamination inflates sMASE.",
        "",
        "== TS Data Quality Objective (deployment-visible, decision point "
        f"{FAIL_ORIGIN}) ==",
        "  Program applied: " + failed + " (intrinsic, full-window)",
        json.dumps({k: round(float(v), 6) for k, v in sorted(
            dq.items())}, indent=2, sort_keys=True),
        "",
        "== Grounded Contrast Capsule (same Program, two decision points) ==",
        f"  Success Context @{capsule['ok_origin']}: winsorize Support "
        "+0.14403 / delayed +0.51098",
        f"  Failure Context @{capsule['fail_origin']}: winsorize Support "
        "−0.16360 / delayed +0.61000",
        "  Same observations (|Δ| < 1e-3): " + json.dumps(
            capsule["same_observations"]),
        "  Different observations (|Δ| >= 1e-3, ok→fail): " + json.dumps(
            {k: v for k, v in capsule["different_observations"].items()},
            indent=1, sort_keys=True),
        "",
        "== Failed Program ==",
        f"  {failed} (params {{}}) applied at {FAIL_ORIGIN} → Support gain "
        "−0.16360 (NEGATIVE)",
        "",
        "== Operator DSL (all legal programs, 1 step) ==",
        "  " + ", ".join(dsl),
        "",
        "== Your task ==",
        "1. Diagnose: PROGRAM_DEFECT (the program itself is wrong for this "
        "context) / EXECUTION_LAPSE (the program is right but was not "
        "applied as defined — no program change needed) / UNIDENTIFIABLE "
        "(evidence insufficient).",
        "2. If PROGRAM_DEFECT: propose at most 2 replacement Typed Programs "
        "from the Operator DSL (each 1 step). Do not propose the failed "
        "program. Justify from the evidence — do not claim any hidden "
        "answer.",
        "3. If UNIDENTIFIABLE: request 1 diagnostic probe (what evidence "
        "would disambiguate) or abstain.",
        'Output JSON only: {"diagnosis": "<PROGRAM_DEFECT|EXECUTION_LAPSE|'
        'UNIDENTIFIABLE>", "programs": [{"op": "<dsl op>", "params": {}, '
        '"rationale": "..."}], "diagnostic_probe": "<question or empty>", '
        '"abstain": bool}',
    ]
    return "\n".join(lines)


def _llm_diagnose(client: Any, prompt: str, *, max_calls: int = 2) -> dict[str, Any]:
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
        except json.JSONDecodeError:
            payload = {}
        diag = str(payload.get("diagnosis", ""))
        if diag in ("PROGRAM_DEFECT", "EXECUTION_LAPSE", "UNIDENTIFIABLE"):
            return {"ok": True, "payload": payload, "raw": raw,
                    "attempts": attempts}
    return {"ok": False, "payload": {}, "raw": attempts[-1] if attempts else "",
            "attempts": attempts}


def main() -> int:
    root = PROJECT_ROOT
    # GEFCom 开发 case（暴露数据；outlier_iqr 正控隐藏）
    cfg = dict(v6.DATASET_CONFIGS["gefcom"])
    roster, values = v6._fixed_roster(root, cfg)
    executor = ScopeExecutor(roster, values, cfg, evaluate_fn=v6._evaluate)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)

    # 实测开发 case（确认数值；正控仅用于验收判定，不进 prompt）
    r_win_fail = executor.evaluate([("winsorize", {})], FAIL_ORIGIN)
    r_out_pos = executor.evaluate([("outlier_iqr", {})], FAIL_ORIGIN)
    print(f"== case: winsorize@928 gain={r_win_fail.gain} "
          f"(hidden positive: outlier_iqr@928 gain={r_out_pos.gain})")

    # ---- Slow Agent 输入（TS-native）----
    capsule = _contrast(values, SUCCESS_ORIGIN, FAIL_ORIGIN, PERIOD)
    dq = dict(resolver.window_context(values, FAIL_ORIGIN, PERIOD))
    dq["bound_period"] = float(PERIOD)
    dq["decision_point"] = float(FAIL_ORIGIN)
    dq["level_excursion_score"] = dq.get("level_excursion_score", 0.0)
    dsl = ("denoise_median", "hampel_filter", "impute_ar", "impute_ema",
           "impute_fft", "impute_linear", "impute_ssm", "outlier_iqr",
           "outlier_mad", "period_complete", "period_median_complete",
           "repair_level_shift", "resample_uniform", "winsorize")
    print(f"== contrast: same={len(capsule['same_observations'])} "
          f"diff={len(capsule['different_observations'])}")

    # ---- LLM 分流 + 生成（≤2 调用；outlier_iqr 正控不在 prompt 标注）----
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — INCONCLUSIVE")
        return 0
    import openai
    client = CountingClient(openai.OpenAI(api_key=api_key,
                                          base_url=BASE_URL), max_calls=2)
    diag = _llm_diagnose(client, _slow_prompt(capsule, dq, "winsorize", dsl))
    print(f"== LLM: ok={diag['ok']} "
          f"diagnosis={diag['payload'].get('diagnosis')}")
    if diag["ok"] and diag["payload"].get("programs"):
        print(f"== LLM programs: {json.dumps(diag['payload']['programs'],
                                             ensure_ascii=False)}")

    # 机械断言（审核第六轮：verdict 必须由断言承重，不能只从 raw 人工读）
    checks: dict[str, Any] = {}
    diag_name = diag["payload"].get("diagnosis", "")
    prog_list = diag["payload"].get("programs") or []
    checks["diagnosis_matches_payload"] = bool(diag["ok"])
    checks["diagnosis_is_unidentifiable"] = bool(diag_name == "UNIDENTIFIABLE")
    checks["programs_empty"] = bool(len(prog_list) == 0)
    checks["abstain_true"] = bool(diag["payload"].get("abstain") is True)
    checks["diagnostic_probe_nonempty"] = bool(
        str(diag["payload"].get("diagnostic_probe") or "").strip())
    checks["llm_calls_le_2"] = bool(client.calls <= 2)
    # 无越权执行：分流止步时未调用 verifier/写 Skill/打开额外 outcome
    checks["no_verifier_or_skill_when_abstain"] = bool(
        diag_name != "PROGRAM_DEFECT" or not prog_list)

    if not diag["ok"]:
        verdict = "INCONCLUSIVE"
    elif diag_name == "EXECUTION_LAPSE":
        verdict = "EXECUTION_LAPSE_NO_PROGRAM_CHANGE"
    elif diag_name == "UNIDENTIFIABLE":
        # 降级 verdict（审核第六轮）：安全弃权观察，不称分流机制正确
        if prog_list:
            verdict = "UNIDENTIFIABLE_WITH_PROGRAMS"  # 异常形态（如实记录）
        else:
            verdict = "TSNATIVE_DIAGNOSTIC_PROMPT_SAFE_ABSTAIN_OBSERVED"
    elif diag_name == "PROGRAM_DEFECT" and not prog_list:
        verdict = "PROGRAM_DEFECT_NO_PROGRAMS"  # 单独档（修复误分类 bug）
    else:
        # ---- 候选验证链（按 LLM 顺序，≤2 个）----
        adopted = None
        for cand in diag["payload"]["programs"][:2]:
            op = str(cand.get("op", ""))
            if op not in dsl or op == "winsorize":
                continue
            steps = ((op, dict(cand.get("params") or {})),)
            rp = executor.evaluate(steps, FAIL_ORIGIN)
            gain_p = (float(rp.gain) if rp.gain is not None else None)
            checks[f"cand_{op}_verifier"] = rp.verification.passed
            checks[f"cand_{op}_support_positive"] = bool(
                rp.verification.passed and gain_p is not None
                and gain_p >= MATERIAL)
            if not (rp.verification.passed and gain_p is not None
                    and gain_p >= MATERIAL):
                continue
            rd = executor.evaluate(steps, DELAYED_ORIGIN)
            gain_d = (float(rd.gain) if rd.gain is not None else None)
            checks[f"cand_{op}_delayed_no_flip"] = bool(
                gain_d is not None and gain_d >= -MATERIAL)
            if gain_d is not None and gain_d >= -MATERIAL:
                adopted = {"op": op, "steps": steps,
                           "support_gain": gain_p, "delayed_gain": gain_d}
                break
        checks["llm_generated_legal_candidate"] = bool(
            any(c.get("op") in dsl for c in diag["payload"]["programs"][:2]))

        if adopted is None:
            verdict = "PROGRAM_REPLAY_FAILED"
        else:
            # ---- ④ 写 Target-local Skill ----
            skill_id = f"{adopted['op'][:8]}-sa-v1"
            status = ("LOCAL_ACTIVE" if adopted["delayed_gain"] >= MATERIAL
                      else "RESTRICTED")
            patched, store, fork_root = sealed.write_skill(
                root, h0, adopted["steps"], skill_id, status,
                rationale=f"TS-native Slow Agent @{FAIL_ORIGIN}: "
                          f"winsorize failed ({r_win_fail.gain:.5f}), "
                          f"replaced by {adopted['op']} "
                          f"(support {adopted['support_gain']:.5f}, "
                          f"delayed {adopted['delayed_gain']:.5f})")
            checks["skill_written"] = True

            # ---- ⑤ 下一轮正常入口实际采用（@976）----
            ops_all = tuple(o for o in dsl)
            method = sealed.TTHAMethod(
                sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                    sealed.SealedProbeBackend(explore=True,
                                              operators=ops_all),
                    LocalPublicToolGateway(series0[:DELAYED_ORIGIN],
                                           task_kind="forecast"))),
                patched, ())
            method.bind_round_data(series0[:DELAYED_ORIGIN],
                                   task_kind="forecast")
            obs = dict(resolver.window_context(values, DELAYED_ORIGIN, PERIOD))
            obs["bound_period"] = float(PERIOD)
            r2 = method.prepare(sealed._request(series0, values,
                                                DELAYED_ORIGIN))
            chosen = method.last_trace.chosen_candidate_id
            skill_cand = f"cand_skill_{skill_id}"
            adopted_ok = chosen == skill_cand
            r2_gain = None
            if r2.program is not None:
                cs = tuple((op, dict(pr)) for op, pr in
                           r2.program.execution_steps())
                rr2 = executor.evaluate(cs, DELAYED_ORIGIN)
                r2_gain = (float(rr2.gain) if rr2.gain is not None else None)
            checks["next_round_actual_adoption"] = bool(adopted_ok)
            checks["next_round_executed"] = r2_gain is not None
            checks["next_round_gain"] = r2_gain
            print(f"== @976: chosen={chosen} adopted={adopted_ok} "
                  f"gain={r2_gain}")

            # ---- ⑥ 移除 Skill 时行动变化（H0 prepare）----
            method_ctrl = sealed.TTHAMethod(
                sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                    sealed.SealedProbeBackend(explore=True,
                                              operators=ops_all),
                    LocalPublicToolGateway(series0[:DELAYED_ORIGIN],
                                           task_kind="forecast"))),
                h0, ())
            method_ctrl.bind_round_data(series0[:DELAYED_ORIGIN],
                                        task_kind="forecast")
            method_ctrl.prepare(sealed._request(series0, values,
                                                DELAYED_ORIGIN))
            chosen_ctrl = method_ctrl.last_trace.chosen_candidate_id
            checks["removal_changes_action"] = bool(
                chosen_ctrl != skill_cand)
            print(f"== @976 ctrl(no skill): chosen={chosen_ctrl}")

            passed = all(v is True for k, v in checks.items()
                         if k not in ("next_round_gain",))
            verdict = ("SLOW_AGENT_TSNATIVE_PASS" if passed else
                       "SLOW_AGENT_TSNATIVE_PARTIAL")
            store.discard_fork(fork_root)

    print(f"== verdict: {verdict}")
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-slow-agent-tsnative",
        "case": "GEFCom winsorize @928（暴露开发测试；outlier_iqr 正控隐藏）",
        "input": {"task": "forecast|ridge|sMASE",
                  "dq_decision_point": FAIL_ORIGIN,
                  "contrast_same_count": len(capsule["same_observations"]),
                  "contrast_diff_keys": list(
                      capsule["different_observations"].keys())},
        "llm": {"ok": diag["ok"], "payload": diag["payload"],
                "raw": diag["raw"]},
        "checks": checks,
        "verdict": verdict,
        "llm_api_call_count": (client.calls if "client" in dir() else 0),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
