"""P0_BATCH_EVIDENCE_CAUSAL_DEV v2（2026-08-13：Batch Evidence 因果使用
决定性实验——唯一一次最小接口修复后的确认运行）。

v1 结果（w1_p0_batch_evidence_causal_report.json）：
  BATCH_EVIDENCE_CAUSAL_USE_DEVELOPMENT_FAIL
  first_fault = BATCH_ALIGNMENT_NOT_USED（protocol 100% / abstain 100% /
  order 83%——但 patch A 臂 16.7%、swap 跟随 16.7%——模型保守弃权，
  读不出对齐证据）
按用户预注册 FAIL 分支执行唯一一次最小接口修复：
  **结构化 Evidence Decision Table**——每个候选：
    rows（episode_id/window/gain 逐行）+ minimum_gain + harm_count
    （gain < −M 的行数）+ common_positive（全部行 gain ≥ M）
  修复集 = 案例 {1, 2, 4}；确认集 = {3, 5, 6}（确认集上不再调任何
  格式/Prompt——不论结果）。

冻结规格（v1 起不变——详见 v1 docstring；本文件保留要点）：
  - 证据规则：唯一候选全部呈现 gain ≥ M → 选其 patch_id；否则 ABSTAIN
    （summary 列由 Runtime 按同一规则从呈现证据派生——swap 翻转
    summary；模型仍需按呈现表选择）
  - Gold 表/案例枚举/选择（registry 序 + 跨 pair 多样性）不变
  - 五臂不变（稳定 patch_id——v1 run-2 起已修）
  - instruction 更新为表格语义（一次性——本次修复的组成部分）
  - PASS 门在**确认集**上评估（同 v1 门）：
    protocol_valid=100% / full-batch 一致性 ≥80% / order ≥80% /
    swap 翻转 ≥75% / patch ≥80% 且 abstain ≥80% / full ≥ mean 且
    full ≥ single
  - 确认集仍失败 → 诚实关闭 LLM_BATCH_EVIDENCE_INTEGRATION_NOT_
    ESTABLISHED → 降级设计（确定性 Runtime 筛 robust candidate/
    abstain，LLM 只编译 Typed Patch——不继续调 Memory radius/解释）

用法：
  python evaluation/functional/run_v1_p0_batch_evidence_causal_dev.py
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
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    deployment_constraints_v1,
    forecast_task_context_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    public_operator_contract,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
)
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgictoChatCompletionsBackend,
)

PERIOD = 24
M = resolver.MATERIAL_THRESHOLD
E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_p0_batch_evidence_causal_report_v2.json"

# 修复集/确认集（冻结——v2 一次性接口修复的边界）
FIX_CASES = {1, 2, 4}
CONFIRM_CASES = {3, 5, 6}

# v2 instruction（一次性更新——表格语义；此后不再调）
INSTRUCTION = (
    "The harness gate accepts a replacement only if its Support gain is "
    "positive (>= 0.005) on ALL presented episodes. Below is the batch "
    "evidence decision table for two candidate replacements. Each row is "
    "one episode's gain; common_positive is true only when ALL of that "
    "candidate's presented gains are >= 0.005. Choose the patch_id of the "
    "candidate whose common_positive is true, if there is exactly one "
    "such candidate. If zero or more than one candidate has "
    "common_positive=true, declare no_proposal with reason_code "
    "insufficient_public_evidence. You do not approve your own edit — a "
    "deterministic replay gate will verify it. Base your choice only on "
    "the presented decision table."
)

ARMS = ("full", "reversed", "swap", "mean_only", "single_episode")
REPS = 2


def _load_series0() -> np.ndarray:
    cache = np.load(PROJECT_ROOT / "data/kdd2018/series_cache.npz",
                    allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    return np.asarray(values[names.index("T117")], dtype=np.float64)


def _candidate_steps(op: str) -> tuple[tuple[str, dict[str, object]], ...]:
    if op.startswith("hampel_V"):
        variant = {"V2": {"window": 7, "n_sigmas": 3.0, "global_z_min": 1.0},
                   "V3": {"window": 3, "n_sigmas": 0.5, "global_z_min": 1.0},
                   "V4": {"window": 5, "n_sigmas": 2.0, "global_z_min": 1.0}}[
            op[len("hampel_"):]]
        return (("hampel_filter", dict(variant)),)
    if "_to_" in op:
        a, b = op.split("_to_")
        return ((a, dict(wiring.contract_params(a, PERIOD))),
                (b, dict(wiring.contract_params(b, PERIOD))))
    return ((op, dict(wiring.contract_params(op, PERIOD))),)


def _load_gold() -> dict[str, dict[str, Any]]:
    gate1 = json.loads((E2 / "w1_group_evidence_chain_gate1_report.json")
                       .read_text(encoding="utf-8"))
    census = json.loads((E2 / "w1_batch_census_dev_report.json")
                        .read_text(encoding="utf-8"))
    supply = json.loads((E2 / "w1_program_supply_dev_report.json")
                        .read_text(encoding="utf-8"))
    t117: dict[str, list[float]] = {}
    for alt, head in (gate1.get("headroom") or {}).items():
        gains = [g["gain"] for g in head["per_episode_gains"]]
        if all(g is not None for g in gains):
            t117[alt] = [float(g) for g in gains]
    dev: dict[str, list[float]] = {}
    fam0 = census["development_families"][0]
    for alt, head in (fam0.get("replacement_headroom") or {}).items():
        gains = [g["gain"] for g in head["per_episode_gains"]]
        if all(g is not None for g in gains):
            dev[alt] = [float(g) for g in gains]
    for c in supply.get("search") or []:
        gains = [w["gain"] for w in c["per_window_gains"]]
        if all(g is not None for g in gains):
            dev[c["label"]] = [float(g) for g in gains]
    seen: list[tuple[float, ...]] = []
    dedup: dict[str, list[float]] = {}
    for label, gains in dev.items():
        key = tuple(round(g, 10) for g in gains)
        if key in seen:
            continue
        seen.append(key)
        dedup[label] = gains
    return {"t117": {"episodes": [888, 984], "candidates": t117},
            "dev": {"episodes": [str(w["series"]) + "@" + str(w["origin"])
                                 for w in fam0["episodes"]],
                    "candidates": dedup}}


def _classify(pair_gains: tuple[list[float], list[float]]) -> str:
    all_pos = [all(g >= M for g in gains) for gains in pair_gains]
    if sum(all_pos) == 1:
        return "patch"
    if sum(all_pos) == 0 and all(any(g >= M for g in gains)
                                 for gains in pair_gains):
        return "abstain"
    return "none"


def _select_distinct_pairs(cases: list[dict], n_slots: int) -> list[dict]:
    chosen: list[dict] = []
    extras: list[dict] = []
    seen: dict[tuple[str, str], int] = {}
    for c in cases:
        key = (c["table"], tuple(c["labels"]))
        seen[key] = seen.get(key, 0) + 1
        (chosen if seen[key] == 1 else extras).append(c)
    return (chosen + extras)[:n_slots]


def _build_cases(gold: dict[str, dict[str, Any]]) -> tuple[list[dict], int]:
    patch_cases: list[dict] = []
    patch_biased: list[dict] = []
    abstain_cases: list[dict] = []
    for table in ("t117", "dev"):
        t = gold[table]
        labels = list(t["candidates"].keys())
        eps = t["episodes"]
        n = len(eps)
        subset_order = [(0, n)] + [(s, l) for s in range(n)
                                   for l in range(2, n - s + 1)
                                   if (s, l) != (0, n)]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                gi, gj = t["candidates"][labels[i]], t["candidates"][labels[j]]
                for start, length in subset_order:
                    subset = eps[start:start + length]
                    gi_s = gi[start:start + length]
                    gj_s = gj[start:start + length]
                    kind = _classify((gi_s, gj_s))
                    if kind == "none":
                        continue
                    winner, loser = None, None
                    if kind == "patch":
                        if all(g >= M for g in gi_s):
                            winner, loser = (labels[i], gi_s), \
                                (labels[j], gj_s)
                        else:
                            winner, loser = (labels[j], gj_s), \
                                (labels[i], gi_s)
                    case = {
                        "table": table,
                        "labels": [labels[i], labels[j]],
                        "subset": list(subset),
                        "gains": {labels[i]: list(gi_s),
                                  labels[j]: list(gj_s)},
                        "kind": kind,
                        "winner": winner[0] if winner else None,
                        "natural": (
                            (table == "t117" and labels[i] == "outlier_mad"
                             and labels[j] == "hampel_filter"
                             and subset == [888, 984])
                            or (table == "dev"
                                and labels[i] == "outlier_mad"
                                and labels[j] == "hampel_filter"
                                and len(subset) == 6)),
                    }
                    if kind == "patch":
                        biased = (loser[1][0] > winner[1][0])
                        case["single_episode_biased"] = biased
                        (patch_biased if biased else patch_cases) \
                            .append(case)
                    else:
                        case["single_episode_biased"] = False
                        abstain_cases.append(case)
    chosen = (_select_distinct_pairs(patch_biased + patch_cases, 3)
              + _select_distinct_pairs(abstain_cases, 3))
    return chosen, len(patch_biased + patch_cases)


def _table_row(pid: str, values: list[float],
               episodes: list[dict] | None) -> dict[str, Any]:
    """v2 结构化 Evidence Decision Table 行（含 summary 列）。"""
    row: dict[str, Any] = {
        "patch_id": pid,
        "minimum_gain": round(min(values), 6),
        "harm_count": sum(1 for v in values if v < -M),
        "common_positive": bool(all(v >= M for v in values)),
    }
    if episodes is not None:
        row["rows"] = [{"episode_id": episodes[k]["episode_id"],
                        "window": episodes[k]["window"],
                        "gain": round(values[k], 6)}
                       for k in range(len(values))]
    else:
        row["rows"] = [{"episode_id": "aggregate",
                        "gain": round(values[0], 6)}]
    return row


def _present(case: dict, arm: str) -> dict[str, Any]:
    """按臂构造呈现证据（v2：结构化 Decision Table + summary 列）。
    稳定 patch_id；reversed 只倒行序；swap 在稳定 id 间换 gain 向量。"""
    labels = list(case["labels"])
    gains = {l: list(case["gains"][l]) for l in labels}
    n = len(case["subset"])
    stable = {l: f"patch-{i}-{l}" for i, l in enumerate(labels)}
    if arm == "swap":
        gains[labels[0]], gains[labels[1]] = \
            gains[labels[1]], gains[labels[0]]
    row_order = list(reversed(labels)) if arm == "reversed" else list(labels)
    episodes = [{"episode_id": f"e{i + 1}", "window": case["subset"][i]}
                for i in range(n)]
    if arm == "mean_only":
        return {"arm": "mean_only",
                "episodes": [{"episode_id": "aggregate",
                              "window": f"mean_over_{n}_episodes"}],
                "decision_table": [
                    _table_row(stable[l], [sum(gains[l]) / n], None)
                    for l in row_order]}
    if arm == "single_episode":
        return {"arm": "single_episode",
                "episodes": [episodes[0]],
                "decision_table": [
                    _table_row(stable[l], [gains[l][0]], [episodes[0]])
                    for l in row_order]}
    return {"arm": arm,
            "episodes": episodes,
            "decision_table": [
                _table_row(stable[l], gains[l], episodes)
                for l in row_order]}


def _expected(case: dict, arm: str) -> str:
    """冻结证据规则作用于呈现证据（v2：从 decision_table rows 取
    数值——与 summary 列同源）。"""
    pres = _present(case, arm)
    passing = []
    for c in pres["decision_table"]:
        vals = [r["gain"] for r in c["rows"]]
        if all(v >= M for v in vals):
            passing.append(c["patch_id"])
    return passing[0] if len(passing) == 1 else "ABSTAIN"


def _run_one_call(card: Mapping[str, object], h0: Any, api_key: str,
                  task_ctx: Any, contracts: tuple) -> dict[str, Any]:
    import openai  # noqa: PLC0415

    from SelfEvolvingHarnessTS.methods.ttha.method import (  # noqa: PLC0415
        _typed_patch_preflight,
    )

    def attempt() -> dict[str, Any]:
        counter = smoke.CountingClient(
            openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL,
                          timeout=120),
            max_calls=2)
        backend = AgictoChatCompletionsBackend(client=counter,
                                               base_url=smoke.BASE_URL)
        core = TTHAAgentCore(
            backend, LocalPublicToolGateway(_load_series0()[:600],
                                            task_kind="forecast"),
            model=smoke.MODEL, base_url=smoke.BASE_URL)
        slow = TTHASlowAgent(core)
        try:
            manifest = slow.propose_edit(
                card,
                [{"surface_id": "skill_library.entries/{skill_id}",
                  "operation": "ADD", "surface_type": "skill",
                  "allowed_operations": ["ADD"]}],
                h0,
                manifest_preflight=lambda m: _typed_patch_preflight(
                    card, m),
                allowed_operator_contracts=contracts,
                task_context=task_ctx)
        except Exception as exc:  # noqa: BLE001
            return {"decision": None, "reason": f"{type(exc).__name__}: "
                                                f"{exc}",
                    "calls": counter.calls, "protocol_error": True}
        if manifest is None:
            reason = slow.last_no_proposal_reason
            if reason in ("insufficient_public_evidence",
                          "no_authorized_minimal_edit", "risk_too_high"):
                return {"decision": "ABSTAIN", "reason": reason,
                        "calls": counter.calls, "protocol_error": False}
            return {"decision": None,
                    "reason": f"no_proposal_invalid_reason:{reason}",
                    "calls": counter.calls, "protocol_error": True}
        pid = getattr(manifest, "patch_id", None)
        if not pid:
            return {"decision": None, "reason": "manifest_no_patch_id",
                    "calls": counter.calls, "protocol_error": True}
        return {"decision": str(pid), "reason": "manifest",
                "calls": counter.calls, "protocol_error": False}

    out = attempt()
    if out["protocol_error"]:
        retry = attempt()
        retry["whole_retried"] = True
        return retry
    out["whole_retried"] = False
    return out


def _metrics(cells: list[dict]) -> dict[str, Any]:
    def acc(subset: list[dict]) -> float:
        valid = [c for c in subset if not c["protocol_error"]]
        if not valid:
            return 0.0
        return sum(1 for c in valid if c["correct"]) / len(valid)

    full_arm = [c for c in cells if c["arm"] == "full"]
    protocol_valid_rate = (
        1 - sum(1 for c in cells if c["protocol_error"]) / len(cells))
    a_acc = acc(full_arm)
    b_arm = [c for c in cells if c["arm"] == "reversed"]
    a_sorted = sorted(full_arm, key=lambda c: (c["case"], c["rep"]))
    b_sorted = sorted(b_arm, key=lambda c: (c["case"], c["rep"]))
    order_pairs = [(a, b) for a, b in zip(a_sorted, b_sorted)
                   if not a["protocol_error"] and not b["protocol_error"]]
    order_consistency = (
        sum(1 for a, b in order_pairs if a["decision"] == b["decision"])
        / max(1, len(order_pairs)))
    patch_cells = [c for c in full_arm if c["kind"] == "patch"]
    swap_cells = [c for c in cells if c["arm"] == "swap"
                  and c["kind"] == "patch"]
    patch_accuracy = acc(patch_cells)
    abstain_cells = [c for c in full_arm if c["kind"] == "abstain"]
    abstain_accuracy = acc(abstain_cells)
    swap_pairs = {}
    for c in sorted(swap_cells, key=lambda c: (c["case"], c["rep"])):
        swap_pairs[(c["case"], c["rep"])] = c
    denom = num = 0
    for (ci_, rep_), sc in swap_pairs.items():
        fa = next(c for c in full_arm if c["case"] == ci_
                  and c["rep"] == rep_)
        if fa["protocol_error"] or sc["protocol_error"]:
            continue
        denom += 1
        if fa["correct"] and sc["correct"]:
            num += 1
    swap_flip = num / max(1, denom)
    d_acc = acc([c for c in cells if c["arm"] == "mean_only"])
    e_acc = acc([c for c in cells if c["arm"] == "single_episode"])
    return {
        "protocol_valid_rate": protocol_valid_rate,
        "full_batch_evidence_consistent_rate": a_acc,
        "candidate_order_consistency": order_consistency,
        "swap_eligible_predicted_flip_rate": swap_flip,
        "patch_choice_accuracy_A": patch_accuracy,
        "abstain_accuracy_A": abstain_accuracy,
        "mean_only_accuracy": d_acc,
        "single_episode_accuracy": e_acc,
        "full_batch_vs_mean_only_delta": a_acc - d_acc,
        "full_batch_vs_single_episode_delta": a_acc - e_acc,
    }


def _gates(m: dict[str, Any]) -> dict[str, bool]:
    return {
        "protocol_valid_100": m["protocol_valid_rate"] >= 1.0,
        "full_batch_ge_80": m["full_batch_evidence_consistent_rate"] >= 0.80,
        "order_ge_80": m["candidate_order_consistency"] >= 0.80,
        "swap_flip_ge_75": m["swap_eligible_predicted_flip_rate"] >= 0.75,
        "patch_ge_80": m["patch_choice_accuracy_A"] >= 0.80,
        "abstain_ge_80": m["abstain_accuracy_A"] >= 0.80,
        "full_not_weaker_than_mean":
            m["full_batch_evidence_consistent_rate"] >= m["mean_only_accuracy"],
        "full_not_weaker_than_single":
            m["full_batch_evidence_consistent_rate"]
            >= m["single_episode_accuracy"],
    }


def _first_fault(m: dict[str, Any], cells: list[dict]) -> str | None:
    if m["candidate_order_consistency"] < 0.80:
        return "POSITION_BIAS"
    if m["swap_eligible_predicted_flip_rate"] < 0.75:
        return "BATCH_ALIGNMENT_NOT_USED"
    if m["abstain_accuracy_A"] < 0.80:
        return "ABSTAIN_NOT_STABLE"
    patch_cells = [c for c in cells if c["arm"] == "full"
                   and c["kind"] == "patch"]
    wrong = [c for c in patch_cells
             if not c["protocol_error"] and not c["correct"]]
    from collections import Counter as _C
    picks = _C(c["decision"] for c in wrong)
    if picks and max(picks.values()) / len(wrong) >= 2 / 3:
        return "SEMANTIC_OPERATOR_PRIOR_DOMINATES"
    return "GENERAL_DECISION_INSTABILITY"


def main() -> int:
    root = PROJECT_ROOT
    api_key = next((os.environ.get(k, "").strip()) for k in
                   ("OPENAI_API_KEY", "AGICTO_API_KEY")
                   if os.environ.get(k, "").strip())
    if not api_key:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no api key"}, indent=1))
        return 0
    gold = _load_gold()
    cases, n_patch_available = _build_cases(gold)
    if len(cases) < 6:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": f"insufficient cases {len(cases)}"},
                         indent=1))
        return 0
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    from SelfEvolvingHarnessTS.contracts.task import (  # noqa: PLC0415
        MetricSpec,
        forecast_task_spec_v1,
    )
    task_ctx = forecast_task_context_v1(
        task_spec=forecast_task_spec_v1(
            horizon=48, downstream_model_class="ridge",
            metric=MetricSpec("sMASE", "lower_is_better")),
        deployment_constraints=deployment_constraints_v1())

    print(f"== cases: {len(cases)} (patch available {n_patch_available})")
    cells: list[dict[str, Any]] = []
    total_calls = 0
    for ci, case in enumerate(cases):
        contracts = tuple(public_operator_contract(op)
                          for lbl in case["labels"]
                          for op, _p in _candidate_steps(lbl))
        for arm in ARMS:
            for rep in range(1, REPS + 1):
                card = {
                    "pattern_id": f"p0-case-{ci + 1}",
                    "failure_family": "workflow_component_negative",
                    "observable_signature": {"task_kind": "forecast"},
                    "context": {},
                    "workflow": {"steps": [{"op": "winsorize",
                                           "params": {}}]},
                    "typed_patch_options": [
                        {"patch_id": f"patch-{idx}-{lbl}",
                         "program_steps": [{"op": o, "params": dict(p)}
                                           for o, p in
                                           _candidate_steps(lbl)]}
                        for idx, lbl in enumerate(case["labels"])],
                    "facts": {"evidence": _present(case, arm)},
                    "instruction": INSTRUCTION,
                }
                out = _run_one_call(card, h0, api_key, task_ctx,
                                    contracts)
                total_calls += out["calls"]
                exp = _expected(case, arm)
                cells.append({
                    "case": ci + 1, "kind": case["kind"], "arm": arm,
                    "rep": rep,
                    "expected": exp,
                    "decision": out["decision"],
                    "correct": out["decision"] == exp,
                    "reason": out["reason"],
                    "calls": out["calls"],
                    "whole_retried": out["whole_retried"],
                    "protocol_error": out["protocol_error"],
                    "set": ("fix" if (ci + 1) in FIX_CASES
                            else "confirm"),
                })
                print(f"== case {ci + 1} [{case['kind']}] arm={arm} "
                      f"rep={rep}: exp={exp} got={out['decision']} "
                      f"correct={out['decision'] == exp} "
                      f"calls={out['calls']}")

    m_all = _metrics(cells)
    m_all["total_llm_calls"] = total_calls
    confirm_cells = [c for c in cells if c["set"] == "confirm"]
    fix_cells = [c for c in cells if c["set"] == "fix"]
    m_confirm = _metrics(confirm_cells)
    m_fix = _metrics(fix_cells)
    gates = _gates(m_confirm)
    verdict = ("BATCH_EVIDENCE_CAUSAL_USE_DEVELOPMENT_PASS" if all(
        gates.values()) else "BATCH_EVIDENCE_CAUSAL_USE_DEVELOPMENT_FAIL")
    first_fault = None if all(gates.values()) else _first_fault(
        m_confirm, confirm_cells)
    report = {
        "experiment_id": "v1-p0-batch-evidence-causal-v2",
        "note": "P0 v2：唯一一次最小接口修复（结构化 Evidence Decision "
                "Table——v1 first_fault=BATCH_ALIGNMENT_NOT_USED）后确认"
                "运行。修复集={1,2,4} 确认集={3,5,6}（确认集不再调任何"
                "格式/Prompt）。development——零新 Claim。",
        "v1_verdict": "BATCH_EVIDENCE_CAUSAL_USE_DEVELOPMENT_FAIL "
                      "(first_fault=BATCH_ALIGNMENT_NOT_USED)",
        "fix_sets": {"fix": sorted(FIX_CASES),
                     "confirm": sorted(CONFIRM_CASES)},
        "cases": [{
            "index": ci + 1,
            "kind": case["kind"], "table": case["table"],
            "labels": case["labels"], "subset": case["subset"],
            "gains": case["gains"], "winner": case["winner"],
            "natural": case["natural"],
            "single_episode_biased": case.get("single_episode_biased"),
            "set": ("fix" if (ci + 1) in FIX_CASES else "confirm"),
            "mark": ("natural" if case["natural"]
                     else "development_control"),
        } for ci, case in enumerate(cases)],
        "cells": cells,
        "metrics_all": m_all,
        "metrics_fix": m_fix,
        "metrics_confirm": m_confirm,
        "gates_confirm": gates,
        "first_fault": first_fault,
        "verdict": verdict,
    }
    print(f"== metrics all: {json.dumps(m_all, ensure_ascii=False)}")
    print(f"== metrics confirm: {json.dumps(m_confirm, ensure_ascii=False)}")
    print(f"== gates(confirm): {json.dumps(gates, ensure_ascii=False)}")
    print(f"== first_fault: {first_fault}")
    print(f"== verdict: {verdict}")
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
