"""Memory 指导预算化反事实探测（主实验；2026-08-09 文献建议定案）。

唯一主假设（文献审查修正后的承重点）：
  > 在相同 Target feedback budget 下，signed Source Experience 能否帮助
  > Agent 更有效地选择"该做哪两个反事实探测"，进而更早生成有效 Typed
  > Patch，而不是在完整反事实答案公布后再选 Patch。

P0 教训（E1/E2 都选对 → 完整反事实表暴露了正确答案，Memory 无边际价值）
→ 本实验**不给完整反事实表**：

  Target public Context + incumbent Workflow（含其总 gain——失败触发）
      → A5/M_normal：最多检索正/冲突各 1 条；A3/M_remove：空 Source Memory
      → 两臂先冻结最多 2 个反事实探测目标（remove one step）
      → Runtime 才打开对应 Target Support 结果（stop-on-first-positive）
      → LLM 输出 1 个 Typed Patch 或 ABSTAIN
      → 确定性 compile + Support replay + delayed replay
      → 写 Target-local Episode（本次实验仅到 replay；Skill 绑定见四格 smoke）

M_swap（因果诊断）：保持 Episode 数量与文本规模，换符号（正↔负/冲突）；
**plan replay**——复用 M_normal 的探测计划，只换 Memory 重选 Patch。
若 M_normal / M_remove / M_swap 的探测目标、Patch 和 abstention 都不变，
不能声称 Agent 使用了 Memory。

承重指标：
  1. 首个正向探测所需探测数（probe index）；
  2. 探测计划（probe_order）与 rationale；
  3. Patch / ABSTAIN 与合法性；
  4. 冻结 Patch 的 Support/delayed replay；
  5. M_normal/remove/swap 的决策敏感性。

判定规则（文献建议）：
  - M_normal 改变探测/Patch 且比 A3 更快或更安全 → Source Experience 有实际价值；
  - 决策改变但效果更差 → Memory 被使用但 applicability/内容有问题；
  - 三种干预决策不变 → Agent 未因果使用 Experience；
  - Support 正、delayed 负 → 只记 LOCAL_DRAFT/CONFLICT；
  - 无局部可执行 Patch → ACTION_UNAVAILABLE/ABSTAIN。

信息墙（审查者检查项）：
  - 探测选择 prompt：**不含任何 ablation 的 gain**（只给 incumbent 总 gain、
    Context、Memory、合法探测集）；
  - Patch 选择 prompt：**只含已打开探测的 gain**（未探测的 ablation 不暴露）；
  - delayed 不进任何 prompt（replay 后才打开）。

LLM：agicto gpt-5.6-luna，temperature 0；单次选择总调用上限 2
（validation_retries=1 语义，与审查裁决 二十 一致）；M_swap 只 1 次。

用法：
  python evaluation/functional/run_v1_memory_guided_probing.py
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
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_slow_path_smoke import CountingClient  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

TARGET_DOMAIN = "gefcom"
PERIOD = 24
HORIZON = 48
MATERIAL = core.MATERIAL_THRESHOLD  # 0.005
MODEL = "gpt-5.6-luna"
BASE_URL = "https://api.agicto.cn/v1"
KEY_ENVS = ("OPENAI_API_KEY", "AGICTO_API_KEY")
ATTR_REPORT = Path("artifacts/functional/e2/w1_counterfactual_attribution_report.json")
SCAN_REPORT = Path("artifacts/functional/e2/w1_counterfactual_scan_report.json")
SCAN_CROSS_REPORT = Path("artifacts/functional/e2/w1_counterfactual_scan_report_880.json")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_memory_guided_probing_report.json")

LEGAL_PROBES = ("probe_remove_step_a", "probe_remove_step_b")
LEGAL_PATCH_IDS = ("KEEP", "REMOVE_STEP_A", "REMOVE_STEP_B", "ABSTAIN")


def _api_kwargs() -> dict[str, Any]:
    api_key = next((os.environ.get(k, "").strip() for k in KEY_ENVS
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        raise RuntimeError(f"{' or '.join(KEY_ENVS)} is required")
    return {"api_key": api_key, "base_url": BASE_URL}


def _memory_lines(arm: str, *, cross_origin: bool = False,
                  domain: str = "gefcom") -> list[str]:
    """A5/M_normal = 真实已暴露 Episode；M_swap = 换符号构造（数量与文本
    规模相同）。M_remove = 空。

    cross_origin=True（决定性测试）：Memory 只含**其他 origin** 的真实
    Episode（时间合法）：
      - gefcom 决策点 880：winsorize @832 POSITIVE（832 < 880）；
      - nn5 决策点 632：impute_ssm @536/584 双正（NN5 slice ScopeExecutor
        重新确认，536 < 632）。
    """
    if cross_origin and domain == "nn5":
        # 728 决策点：两条真实跨 origin Episode（536/584 seed + 632/680 R1，
        # NN5 slice ScopeExecutor 重新确认；都 < 728 时间合法）
        if arm == "A5":
            return [
                "nn5 impute_ssm @536: support +0.01865 delayed +0.02734 → POSITIVE",
                "nn5 impute_ssm @632: support +0.06967 delayed +0.05633 → POSITIVE",
            ]
        if arm == "M_swap":
            return [
                "nn5 impute_ssm @536: support -0.01865 delayed -0.02734 → NEGATIVE (signed relation inverted for causal diagnosis)",
                "nn5 impute_ssm @632: support -0.06967 delayed -0.05633 → NEGATIVE (signed relation inverted for causal diagnosis)",
            ]
        return []
    if cross_origin:
        if arm == "A5":
            return [
                "gefcom winsorize @832: support +0.14403 delayed +0.51098 → POSITIVE",
            ]
        if arm == "M_swap":
            return [
                "gefcom winsorize @832: support -0.14403 delayed -0.51098 → NEGATIVE (signed relation inverted for causal diagnosis)",
            ]
        return []
    if arm == "A5":
        return [
            "gefcom outlier_iqr @928: support +0.04386 delayed +0.02719 → POSITIVE",
            "gefcom impute_fft @928: support +0.01849 delayed -0.03496 → CONFLICT",
        ]
    if arm == "M_swap":
        return [
            "gefcom outlier_iqr @928: support -0.04386 delayed -0.02719 → NEGATIVE (signed relation inverted for causal diagnosis)",
            "gefcom impute_fft @928: support +0.01849 delayed +0.03496 → POSITIVE (signed relation inverted for causal diagnosis)",
        ]
    return []


def _probe_choice_prompt(case: Mapping[str, Any], values: Mapping[str, Any],
                         arm: str, *, cross_origin: bool = False,
                         domain: str = "gefcom") -> str:
    origin = int(case["origin"])
    ctx = resolver.window_context(values, origin, PERIOD)
    ctx_plain = {k: round(float(v), 6) for k, v in sorted(ctx.items())}
    lines = [
        "You are the counterfactual attribution path of a time-series "
        "preprocessing harness.",
        "",
        "A two-step Workflow failed at support decision point "
        f"{origin}. Before proposing a fix, you may run at most two "
        "counterfactual probes: each probe evaluates one leave-one-step-out "
        "version of the Workflow on the SAME support decision point.",
        "",
        "== Incumbent Workflow (A then B) ==",
        f"  A: {case['step_a']}",
        f"  B: {case['step_b']}",
        f"  incumbent full-workflow support gain: {case['support']['A_to_B']:+.5f} "
        "(negative or non-positive = failure trigger)",
        "",
        "== Public context at the decision point ==",
        json.dumps(ctx_plain, indent=2, sort_keys=True),
    ]
    mem = _memory_lines(arm, cross_origin=cross_origin, domain=domain)
    if mem:
        lines += ["", "== Experience memory (signed) ==",
                  *[f"  - {m}" for m in mem]]
    lines += [
        "",
        "== Legal probes (budget: at most 2, each evaluates one ablated "
        "version; the results are NOT known until you run them) ==",
        "  probe_remove_step_a   evaluate B only (drop step A)",
        "  probe_remove_step_b   evaluate A only (drop step B)",
        "",
        "== Your task ==",
        "Choose the probe order that most efficiently localizes the failure "
        "(the first probe you expect to reveal a positive or informative "
        "ablation). You may order both probes or a single one.",
        'Output JSON only: {"probe_order": ["<probe>", ...], '
        '"rationale": "..."}',
    ]
    return "\n".join(lines)


def _patch_choice_prompt(case: Mapping[str, Any], values: Mapping[str, Any],
                         arm: str, probe_results: list[dict[str, Any]],
                         *, cross_origin: bool = False,
                         domain: str = "gefcom") -> str:
    origin = int(case["origin"])
    lines = [
        "You are the counterfactual attribution path of a time-series "
        "preprocessing harness.",
        "",
        "You ran counterfactual probes on the failing two-step Workflow. "
        "Choose the update direction based on the probe results below.",
        "",
        "== Incumbent Workflow (A then B) ==",
        f"  A: {case['step_a']}  B: {case['step_b']}",
        f"  incumbent full-workflow support gain: {case['support']['A_to_B']:+.5f}",
        "",
        "== Opened probe results (support decision point "
        f"{origin}; gain > 0 = better than no preprocessing) ==",
    ]
    for pr in probe_results:
        label = ("B only" if pr["probe"] == "probe_remove_step_a"
                 else "A only")
        lines.append(f"  probe {pr['probe']} ({label}): "
                     f"{pr['gain']:+.5f}")
    mem = _memory_lines(arm, cross_origin=cross_origin, domain=domain)
    if mem:
        lines += ["", "== Experience memory (signed) ==",
                  *[f"  - {m}" for m in mem]]
    lines += [
        "",
        "== Legal patch IDs ==",
        "  KEEP           keep A -> B",
        "  REMOVE_STEP_A  drop step A, keep only step B",
        "  REMOVE_STEP_B  drop step B, keep only step A",
        "  ABSTAIN        no change proposed (evidence insufficient)",
        "",
        "== Your task ==",
        "Choose exactly one legal patch ID. ABSTAIN is valid if the opened "
        "probe results are insufficient.",
        'Output JSON only: {"patch_id": "<one legal ID>", '
        '"evidence_refs": ["..."], "rationale": "..."}',
    ]
    return "\n".join(lines)


def _llm_json(client: Any, prompt: str, *,
              validator: Any, max_calls: int = 2) -> dict[str, Any]:
    """一次 LLM 调用 + 机器校验；非法 → 重试 1 次；超限硬停。"""
    attempts: list[str] = []
    for _ in range(max_calls):
        kwargs = {"model": MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0}
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
            continue
        error = validator(payload)
        if error is None:
            return {"payload": payload, "attempts": attempts}
    return {"payload": None, "attempts": attempts, "invalid": True}


def _validate_probe_order(payload: Mapping[str, Any]) -> str | None:
    order = payload.get("probe_order")
    if not isinstance(order, list) or not order:
        return "probe_order must be a non-empty list"
    if len(order) > 2:
        return "probe_order must contain at most 2 probes"
    if any(p not in LEGAL_PROBES for p in order):
        return f"probe must be in {LEGAL_PROBES}"
    if len(set(order)) != len(order):
        return "probe_order must not contain duplicates"
    return None


def _validate_patch(payload: Mapping[str, Any]) -> str | None:
    pid = payload.get("patch_id")
    if pid not in LEGAL_PATCH_IDS:
        return f"patch_id must be one of {LEGAL_PATCH_IDS}"
    return None


def _patch_to_steps(case: Mapping[str, Any], patch_id: str) -> tuple:
    a = str(case["step_a"])
    b = str(case["step_b"])
    pa = dict(wiring.contract_params(a, PERIOD))
    pb = dict(wiring.contract_params(b, PERIOD))
    if patch_id == "KEEP":
        return ((a, pa), (b, pb))
    if patch_id == "REMOVE_STEP_A":
        return ((b, pb),)
    if patch_id == "REMOVE_STEP_B":
        return ((a, pa),)
    raise ValueError(f"no steps for {patch_id}")


def run_arm(client: Any, case: Mapping[str, Any], values: Mapping[str, Any],
            executor: ScopeExecutor, *, arm: str,
            fixed_probe_order: list[str] | None = None,
            cross_origin: bool = False, domain: str = "gefcom") -> dict[str, Any]:
    """一臂流程。fixed_probe_order 非空 = plan replay（928 实验的 M_swap）；
    cross_origin=True 时 M_swap **重生成探测计划**（决定性测试：测 plan 层
    符号敏感性——不再 plan replay）。"""
    origin = int(case["origin"])
    delayed_origin = origin + HORIZON
    out: dict[str, Any] = {"arm": arm}
    calls_before = client.calls

    # ---- 调用 1：探测选择（plan replay 模式复用计划不调用）----
    if fixed_probe_order is not None and not cross_origin:
        probe_order = list(fixed_probe_order)
        out["probe_plan_replay"] = True
    else:
        probe_prompt = _probe_choice_prompt(case, values, arm,
                                            cross_origin=cross_origin,
                                            domain=domain)
        probe_result = _llm_json(client, probe_prompt,
                                 validator=_validate_probe_order)
        out["probe_prompt"] = probe_prompt
        out["probe_call"] = probe_result
        if probe_result.get("invalid") or probe_result["payload"] is None:
            out["aborted"] = "INVALID_PROBE_ORDER"
            return out
        probe_order = list(probe_result["payload"]["probe_order"])
        if cross_origin and arm == "M_swap":
            out["probe_plan_regenerated"] = True

    # ---- Runtime 按序打开 Support（预算 ≤2；stop-on-first-positive）----
    probe_results: list[dict[str, Any]] = []
    for probe in probe_order:
        if probe == "probe_remove_step_a":
            steps = ((case["step_b"], dict(wiring.contract_params(str(case["step_b"]), PERIOD))),)
            label = "B only"
        else:
            steps = ((case["step_a"], dict(wiring.contract_params(str(case["step_a"]), PERIOD))),)
            label = "A only"
        receipt = executor.evaluate(steps, origin)
        gain = float(receipt.gain) if receipt.gain is not None else None
        probe_results.append({
            "probe": probe, "label": label, "gain": gain,
            "verification_passed": receipt.verification.passed,
            "error": receipt.error,
        })
        out.setdefault("probe_evals", []).append(probe_results[-1])
        if gain is not None and gain >= MATERIAL:
            break  # stop-on-first-positive
    out["probe_order"] = probe_order
    out["first_positive_probe_index"] = next(
        (i + 1 for i, p in enumerate(probe_results)
         if p["gain"] is not None and p["gain"] >= MATERIAL), None)

    # ---- 调用 2：Patch 选择（基于已打开结果）----
    patch_prompt = _patch_choice_prompt(case, values, arm, probe_results,
                                        cross_origin=cross_origin,
                                        domain=domain)
    patch_result = _llm_json(client, patch_prompt, validator=_validate_patch)
    out["patch_prompt"] = patch_prompt
    out["patch_call"] = patch_result
    out["llm_calls"] = client.calls - calls_before
    if patch_result.get("invalid") or patch_result["payload"] is None:
        out["aborted"] = "INVALID_PATCH_ID"
        return out
    pid = patch_result["payload"]["patch_id"]
    out["patch_id"] = pid
    if pid == "ABSTAIN":
        out["abstained"] = True
        return out

    # ---- 确定性 replay（Patch 冻结后）----
    steps = _patch_to_steps(case, pid)
    out["frozen_steps"] = [{"op": op, "params": dict(pr)} for op, pr in steps]
    verification = executor.verify(steps, origin)
    support = executor.evaluate(steps, origin)
    delayed = executor.evaluate(steps, delayed_origin)
    out["replay"] = {
        "compiler": {"passed": verification.passed,
                     "checked_windows": verification.checked_windows,
                     "rejected_windows": verification.rejected_windows},
        "support": {"origin": origin,
                    "gain": (float(support.gain)
                             if support.gain is not None else None)},
        "delayed": {"origin": delayed_origin,
                    "gain": (float(delayed.gain)
                             if delayed.gain is not None else None),
                    "opened_after_patch_freeze": True},
    }
    return out


def main() -> int:
    root = PROJECT_ROOT
    cross_origin = "--cross-origin" in sys.argv
    domain = TARGET_DOMAIN
    origin_override: int | None = None
    for arg in sys.argv[1:]:
        if arg.startswith("--domain="):
            domain = arg.split("=", 1)[1]
        elif arg.startswith("--origin="):
            origin_override = int(arg.split("=", 1)[1])
    if cross_origin:
        # 决定性测试：决策点 880（gefcom，Memory=832 winsorize）或
        # 632/728（nn5，Memory=536/584 [+632/680] impute_ssm）——Memory
        # 只含其他 origin 的真实 Episode（时间合法）
        nn5_origin = origin_override if origin_override is not None else 632
        scan_path = (Path(f"artifacts/functional/e2/w1_counterfactual_scan_report_nn5_{nn5_origin}.json")
                     if domain == "nn5" else SCAN_CROSS_REPORT)
        scan = json.loads((root / scan_path).read_text(encoding="utf-8"))
        case = scan["first_hit"]
        assert int(case["origin"]) == (nn5_origin if domain == "nn5" else 880)
        attr = {"case": case,
                "delayed_runner_side_not_in_feedback": case.get("delayed")}
        report_out = REPORT_OUT_REL.with_name(
            f"{REPORT_OUT_REL.stem}_cross_origin_{domain}"
            f"{REPORT_OUT_REL.suffix}")
        experiment_id = f"v1-memory-guided-probing-cross-origin-{domain}"
    else:
        attr = json.loads((root / ATTR_REPORT).read_text(encoding="utf-8"))
        case = attr["case"]
        report_out = REPORT_OUT_REL
        experiment_id = "v1-memory-guided-probing"
    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    import openai
    # 预算语义：每选择 ≤2 次（validation_retries=1 格式纠正）；
    # cross-origin 6 个选择（A5/A3/M_swap 各 探测+patch）→ 上限 12。
    client = CountingClient(openai.OpenAI(**_api_kwargs()),
                            max_calls=12 if cross_origin else 10)
    print(f"== incumbent: @{case['origin']} {case['step_a']} -> "
          f"{case['step_b']} gain={case['support']['A_to_B']} "
          f"cross_origin={cross_origin} domain={domain}")

    a5 = run_arm(client, case, values, executor, arm="A5",
                 cross_origin=cross_origin, domain=domain)
    print(f"== A5: probes={a5.get('probe_order')} "
          f"first_pos={a5.get('first_positive_probe_index')} "
          f"patch={a5.get('patch_id')} calls={a5.get('llm_calls')}")

    a3 = run_arm(client, case, values, executor, arm="A3",
                 cross_origin=cross_origin, domain=domain)
    print(f"== A3: probes={a3.get('probe_order')} "
          f"first_pos={a3.get('first_positive_probe_index')} "
          f"patch={a3.get('patch_id')} calls={a3.get('llm_calls')}")

    # cross-origin：M_swap **重生成探测计划**（测 plan 层符号敏感性）
    m_swap = run_arm(client, case, values, executor, arm="M_swap",
                     fixed_probe_order=(None if cross_origin
                                        else a5.get("probe_order")),
                     cross_origin=cross_origin, domain=domain)
    replay_note = "plan regenerated" if cross_origin else "plan replay of A5"
    print(f"== M_swap ({replay_note}): probes={m_swap.get('probe_order')} "
          f"patch={m_swap.get('patch_id')} calls={m_swap.get('llm_calls')}")

    # ---- 判定（文献建议规则）----
    checks: dict[str, Any] = {
        "a5_probe_plan_differs_from_a3": (
            a5.get("probe_order") != a3.get("probe_order")),
        "a5_patch_differs_from_a3": (
            a5.get("patch_id") != a3.get("patch_id")),
        "a5_first_positive_earlier_or_equal": _cmp_earlier(
            a5.get("first_positive_probe_index"),
            a3.get("first_positive_probe_index")),
        "a5_no_worse_harm": _harm(a5) <= _harm(a3),
        "swap_changes_decision": (
            m_swap.get("patch_id") != a5.get("patch_id")),
    }
    if cross_origin:
        # plan 层符号敏感性：M_swap 重生成的探测计划 vs A5 的
        checks["swap_changes_probe_plan"] = (
            m_swap.get("probe_order") != a5.get("probe_order"))
    if checks["a5_probe_plan_differs_from_a3"] and checks["a5_first_positive_earlier_or_equal"]:
        outcome = "SOURCE_EXPERIENCE_HAS_ACTUAL_VALUE"
    elif checks["a5_probe_plan_differs_from_a3"] or checks["a5_patch_differs_from_a3"]:
        outcome = "MEMORY_USED_BUT_APPLICABILITY_OR_CONTENT_PROBLEM"
    elif not checks["swap_changes_decision"]:
        outcome = "AGENT_NOT_CAUSALLY_USING_EXPERIENCE"
    else:
        outcome = "MEMORY_INFLUENCES_BUT_NO_FASTER_FIX"
    if cross_origin:
        verdict = f"CROSS_ORIGIN_MEMORY_GUIDED_PROBING_{outcome}"
    else:
        verdict = f"MEMORY_GUIDED_PROBING_{outcome}"
    print(f"== checks: {json.dumps(checks, ensure_ascii=False)}")
    print(f"== verdict: {verdict}")

    # 信息墙 flag：程序断言（从实际 prompt 文本计算）
    wall = _assert_information_wall(attr, {"A5": a5, "A3": a3, "M_swap": m_swap})
    report = {
        "experiment_id": experiment_id,
        "case": case,
        "memory_source": (f"{'impute_ssm @536/632' if domain == 'nn5' else 'winsorize @832'} "
                          "(cross-origin, time-legal)"
                          if cross_origin else "outlier_iqr/impute_fft @928"),
        "arms": {"A5": a5, "A3": a3, "M_swap": m_swap},
        "checks": checks,
        "verdict": verdict,
        "llm_api_call_count": client.calls,
        "information_wall": wall,
    }
    out = root / report_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


def _assert_information_wall(
    attr_report: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """程序断言（审查者 2026-08-09 修订）：从实际 prompt 文本计算信息墙。

    修订（审查裁决）：真正禁止的是"当前 Query 尚未发生的 future"——
    **时间合法的历史 Source Episode outcome 不算泄漏**。裸数值匹配会把
    历史 Episode 行（如 928 实验 memory 行含 outlier_iqr 历史数值）误判为
    泄漏。修正为按**结果行格式**检测（"X only: ±g" / "probe ... gain"），
    历史 Episode 段单独检测"同 origin 同算子"的语义面泄漏（928 场景）。

    - probe_choice_prompt_has_no_ablation_gains：探测 prompt 不得含当前
      决策点的 A_only/B_only **结果行**（incumbent 总 gain 行允许）；
    - patch_prompt_only_opened_probes：patch prompt 的结果行 gain 必须 ∈
      已打开探测的 gain；
    - delayed_not_in_any_prompt：任何 prompt 不得含当前案例 delayed 数值
      作为结果行（历史 Episode 段数值不算）；
    - same_origin_memory_leak_warning：Memory 段含与当前案例同 origin 同
      算子的数值（= 语义面泄漏，928 场景）→ 警告（True=有警告）。
    """
    case = attr_report["case"]
    a_gain = f"{case['support']['A_only']:+.5f}"
    b_gain = f"{case['support']['B_only']:+.5f}"
    case_delayed = attr_report.get("delayed_runner_side_not_in_feedback") or {}
    d_vals = {f"{v:+.5f}" for v in (case_delayed.values())
              if isinstance(v, (int, float))}
    # 结果行格式：出现在 "only:" / "probe" 行内的数值（非 Memory 段）
    result_line_patterns = ("only:", "probe probe_")
    opened: dict[str, list[str]] = {}
    for name, arm in arms.items():
        opened[name] = [f"{p['gain']:+.5f}" for p in arm.get("probe_evals", [])
                        if p.get("gain") is not None]
    probe_leak = False
    delayed_leak = False
    patch_leak = False
    for name, arm in arms.items():
        pp = arm.get("probe_prompt") or ""
        pc = arm.get("patch_prompt") or ""
        for needle in (a_gain, b_gain):
            if needle in pp:
                # 只在结果行上下文里才算泄漏（incumbent 总 gain 行除外）
                idx = pp.find(needle)
                if any(pat in pp[max(0, idx - 60):idx] for pat in result_line_patterns):
                    probe_leak = True
        for v in d_vals:
            for text in (pp, pc):
                idx = text.find(v)
                if idx >= 0 and any(pat in text[max(0, idx - 60):idx]
                                    for pat in result_line_patterns):
                    delayed_leak = True
        opened_vals = set(opened.get(name, []))
        for needle in (a_gain, b_gain):
            if needle in pc:
                idx = pc.find(needle)
                if any(pat in pc[max(0, idx - 60):idx] for pat in result_line_patterns) \
                        and needle not in opened_vals:
                    patch_leak = True
    # 语义面泄漏警告：Memory 段含与当前案例同 origin 同算子的数值
    mem_warning = False
    origin = int(case["origin"])
    for name, arm in arms.items():
        for key in ("probe_prompt", "patch_prompt"):
            text = arm.get(key) or ""
            mem_seg = text.split("== Experience memory")[-1] \
                if "== Experience memory" in text else ""
            if not mem_seg:
                continue
            if any(f"@{origin}" in mem_seg and (f"{case['step_a']}" in mem_seg
                                                or f"{case['step_b']}" in mem_seg)
                   for _ in [0]):
                mem_warning = True
    return {
        "probe_choice_prompt_has_no_ablation_gains": not probe_leak,
        "patch_prompt_only_opened_probes": not patch_leak,
        "delayed_not_in_any_prompt": not delayed_leak,
        "delayed_opened_after_patch_freeze": True,
        "same_origin_memory_leak_warning": mem_warning,
        "note": "修订：结果行格式检测（历史 Episode 段数值不算泄漏）；"
                "same_origin_memory_leak_warning 单独标注 928 语义面泄漏",
    }


def _cmp_earlier(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None:
        return False
    if b is None:
        return True
    return a <= b


def _harm(arm: Mapping[str, Any]) -> float:
    """探测阶段 + replay 阶段的最大负增益幅度（harm）。"""
    gains = [p["gain"] for p in arm.get("probe_evals", [])
             if p.get("gain") is not None and p["gain"] < -MATERIAL]
    replay = arm.get("replay") or {}
    for key in ("support", "delayed"):
        g = replay.get(key, {}).get("gain")
        if g is not None and g < -MATERIAL:
            gains.append(float(g))
    return round(-sum(gains), 6) if gains else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
