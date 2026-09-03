"""BOUNDED_FAULT_SELECTION_WIRING（S1，2026-08-13：受限五类选择题
接口接线 witness——修订版用户裁决——development exposure——零新
Claim）。

用户裁决三处调整（区别于原草案）：
  1. taxonomy 与 allowed_fault_types 分离——五类定义仅供理解；
     Agent 只能从 Runtime 筛选后的 allowed_fault_types 选择（空时选
     guard）——测的是接口是否正确传递约束，不是复述唯一答案。
  2. proposed_case_action 不控制状态——最终 MATCH/NEW/CONFLICT/
     ABSTAIN 由 Runtime 字段比较决定；Agent 答案仅为提案，不一致时
     不写入不修改 Case。
  3. 两次调用：T117（正向接线：SCOPE_MEMORY_RISK_ERROR → case-0002
     → MATCH）+ T105（证据不可分 → guard NO_ACTIONABLE_FAULT/
     INSUFFICIENT_EVIDENCE——不得强选五类之一）。

Runtime 复核（每次调用）：
  - fault_type ∈ allowed_fault_types（或 allowed 为空时 ∈ GUARDS）；
  - evidence_refs 全部可解析于 Capsule 证据项 ID；
  - matched_case_id ∈ 按 fault_type 硬过滤的检索结果（或 null）；
  - Runtime 自行计算 reconciliation（字段比较）；
  - Agent 输出无写入权限（case store 字节不变——runner 不写）。

通过条件（仅此——verdict 严格限定）：
  2/2 返回合法结构；2/2 未选择屏蔽类别；2/2 evidence_refs 可解析；
  Runtime reconciliation 正常；Agent 输出不拥有写入权限。
  → BOUNDED_FAULT_SELECTION_WIRING_PASS
  不可升级为"LLM 能可靠完成五类归因"或"Case Memory 有收益"。

用法：
  python evaluation/functional/run_v1_bounded_fault_selection_wiring.py
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

import run_v1_slow_path_smoke as smoke  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.fault_cases import (  # noqa: E402
    CASE_ACTIONS,
    FAULT_TYPES,
    GUARDS,
    filter_candidates,
    reconcile_existing,
)

E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_bounded_fault_selection_wiring_report_v2.json"
STORE_REL = E2 / "w1_problem_cases_bootstrap.json"
TASK_CONSUMER = "forecast|ridge|sMASE"
GROUP_FIELDS = {"task_consumer": TASK_CONSUMER,
                "workflow_sig": "winsorize",
                "response_class": "NEGATIVE"}

TAXONOMY_TEXT = """TAXONOMY (definitions for understanding only — you may
NOT choose directly from this taxonomy; your fault_type must come from
allowed_fault_types below):
1. TASK_INTERPRETATION_ERROR — the agent misunderstood the Task / Consumer /
   Horizon / quality target. (Selectable only when a verified TaskSpec /
   Contract conflict is present in the evidence.)
2. QUALITY_DIAGNOSIS_ERROR — a quality-phenomenon diagnosis contradicts
   verifiable facts. (Selectable only when such a contradiction is present.)
3. WORKFLOW_SUPPLY_GAP — every whitelist replacement candidate was measured
   to fail on all in-group windows AND the candidate space was exhaustively
   searched with no full-pass candidate.
4. WORKFLOW_DECISION_ERROR — a positive candidate was measured to exist but
   the agent did not propose or select it.
5. SCOPE_MEMORY_RISK_ERROR — a proposed patch passed Support replay but
   failed delayed validation (temporal / scope risk measured).

MATERIAL THRESHOLD: a candidate "fails" a window when its gain is below
+0.005 (the material threshold M) — a sign-positive gain below +0.005
still counts as FAILED.

GUARDS (choose only when allowed_fault_types is empty — the guard choice
is deterministic, do not improvise):
- NO_ACTIONABLE_FAULT — every replacement candidate in the evidence was
  measured to FAIL (gain < +0.005) on ALL in-group windows.
- INSUFFICIENT_EVIDENCE — such measurements are absent or incomplete, so
  no class can be determined."""

OUTPUT_RULES = """Output exactly one JSON object, nothing else:
{
  "fault_type": "...",
  "proposed_case_action": "MATCH_ADD_EVIDENCE | CONFLICT_WITH_EXISTING | NEW_CASE | ABSTAIN",
  "matched_case_id": "case-XXXX | null",
  "evidence_refs": ["..."],
  "reason": "..."
}
Rules:
- fault_type MUST be one of allowed_fault_types (or, if the list is empty,
  one GUARD).
- proposed_case_action is ADVISORY: the Runtime decides the final case
  action by field comparison and never writes your answer into a case.
- matched_case_id must be one of the case ids listed above, or null.
- every entry of evidence_refs must be an ID that appears under GROUP
  EVIDENCE above."""


def _load(name: str) -> dict[str, Any]:
    return json.loads((E2 / name).read_text(encoding="utf-8"))


def _build_calls() -> list[dict[str, Any]]:
    witness = _load("w1_group_witness_real_slow_report_v3.json")
    b2 = _load("w1_block2_census_ec_dev_report.json")
    store = _load("w1_problem_cases_bootstrap.json")
    cases = store["cases"]

    # ---- T117（正向接线——全证据报告直读）----
    replay = (witness.get("group_feedback_event") or {}).get("group_replay") \
        or []
    delayed = (witness.get("delayed_event") or {}).get("delayed_gain")
    t117_items = [
        {"id": "ep:T117@888", "desc": "winsorize Support gain at origin 888",
         "gain": -0.1426334267351992},
        {"id": "ep:T117@984", "desc": "winsorize Support gain at origin 984",
         "gain": -0.08411687539427182},
    ] + [
        {"id": f"alt:hampel@{e['origin']}",
         "desc": "hampel_filter replacement replay gain",
         "gain": e["gain"]} for e in replay
    ] + [
        {"id": "delayed:T117@1032",
         "desc": "delayed validation of the hampel patch",
         "gain": delayed},
    ]
    t117 = {
        "call_id": "t117",
        "allowed_fault_types": ["SCOPE_MEMORY_RISK_ERROR"],
        "items": t117_items,
        "expected": {"fault_type": "SCOPE_MEMORY_RISK_ERROR",
                     "matched_case_id": "case-0002"},
    }

    # ---- T105（保底通道——block2 报告直读）----
    fam = b2["development_families"][0]
    hr = fam["replacement_headroom"]
    t105_items = [
        {"id": f"ep:T105@{e['origin']}",
         "desc": "winsorize Support gain", "gain": e["gain"]}
        for e in fam["episodes"]
    ] + [
        {"id": f"alt:{alt}@{e['origin']}",
         "desc": f"{alt} replacement replay gain", "gain": e["gain"]}
        for alt, block in hr.items()
        for e in block.get("per_episode_gains") or []
    ]
    t105 = {
        "call_id": "t105",
        "allowed_fault_types": [],
        "items": t105_items,
        "expected": {"fault_type": "NO_ACTIONABLE_FAULT",
                     "matched_case_id": "case-0003"},
    }
    return [t117, t105], cases


def _build_prompt(call: Mapping[str, Any],
                  cases: Sequence[Mapping[str, Any]]) -> str:
    allowed = call["allowed_fault_types"]
    allowed_text = (json.dumps(allowed) if allowed
                    else "[] (empty — choose one GUARD)")
    evidence_lines = "\n".join(
        f"- {it['id']}: {it['desc']} = {it['gain']}"
        for it in call["items"])
    case_lines = "\n".join(
        f"- {c['case_id']} [{c['fault_type']}]: "
        f"{c['failed_behavior']} | status={c['status']}"
        for c in cases)
    return f"""{TAXONOMY_TEXT}

allowed_fault_types (computed by the Runtime from measured facts):
{allowed_text}

GROUP EVIDENCE (capsule — reference items by their IDs):
{evidence_lines}

EXISTING PROBLEM CASES (summaries of all cases in the store):
{case_lines}

{OUTPUT_RULES}"""


def _parse_json(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _ask(counter: Any, prompt: str) -> str:
    resp = counter.chat.completions.create(
        model=smoke.MODEL,
        messages=[{"role": "user", "content": prompt}])
    return str((resp.choices[0].message.content) or "")


def main() -> int:
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no api key"}, indent=1))
        return 0
    import openai  # noqa: PLC0415

    calls, cases = _build_calls()
    store_bytes = STORE_REL.read_bytes()  # 写权限哨兵

    records = []
    for call in calls:
        counter = smoke.CountingClient(
            openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL,
                          timeout=120))
        prompt = _build_prompt(call, cases)
        answer: dict[str, Any] | None = None
        feedback: str | None = None
        for attempt in (1, 2):
            try:
                content = _ask(counter, prompt
                               + (f"\n\nNOTE: your previous output was "
                                  f"invalid ({feedback}). Output the JSON "
                                  "object again."
                                  if feedback else ""))
            except Exception as exc:  # noqa: BLE001
                feedback = f"transport error {type(exc).__name__}"
                continue
            answer = _parse_json(content)
            if answer is None:
                feedback = "could not parse JSON"
                continue
            break

        # ---- Runtime 复核 ----
        checks: dict[str, Any] = {
            "structure_valid": False, "no_masked_selection": False,
            "evidence_refs_resolvable": False,
            "matched_case_in_retrieval": False,
            "runtime_reconciliation": None,
            "proposed_action": None,
        }
        if answer is not None:
            ft = answer.get("fault_type")
            act = answer.get("proposed_case_action")
            refs = answer.get("evidence_refs") or []
            mid = answer.get("matched_case_id")
            allowed = list(call["allowed_fault_types"])
            capsule_ids = {it["id"] for it in call["items"]}
            checks["structure_valid"] = bool(
                isinstance(ft, str) and isinstance(refs, list)
                and all(isinstance(r, str) for r in refs)
                and (mid is None or isinstance(mid, str))
                and act in CASE_ACTIONS)
            checks["no_masked_selection"] = bool(
                (allowed and ft in allowed)
                or (not allowed and ft in GUARDS))
            checks["evidence_refs_resolvable"] = bool(
                refs and set(refs) <= capsule_ids)
            retrieved = filter_candidates(
                cases, ft if isinstance(ft, str) else "",
                GROUP_FIELDS)
            checks["matched_case_in_retrieval"] = bool(
                mid is None
                or any(c["case_id"] == mid for c in retrieved))
            if retrieved:
                checks["runtime_reconciliation"] = reconcile_existing(
                    retrieved[0], GROUP_FIELDS)
            checks["proposed_action"] = act
            checks["agent_answer"] = answer
        records.append({
            "call_id": call["call_id"],
            "allowed_fault_types": call["allowed_fault_types"],
            "expected": call["expected"],
            "llm_calls": counter.calls,
            "checks": checks,
        })
        print(f"== call {call['call_id']}: calls={counter.calls} "
              f"checks={json.dumps(checks, ensure_ascii=False, default=str)}",
              flush=True)

    # ---- 通过条件（用户裁决——仅此）----
    structure_ok = all(r["checks"]["structure_valid"] for r in records)
    masked_ok = all(r["checks"]["no_masked_selection"] for r in records)
    refs_ok = all(r["checks"]["evidence_refs_resolvable"] for r in records)
    reconcile_ok = all(
        r["checks"]["runtime_reconciliation"] == "MATCH_ADD_EVIDENCE"
        for r in records)
    no_write_ok = bool(STORE_REL.read_bytes() == store_bytes)
    verdict = ("BOUNDED_FAULT_SELECTION_WIRING_PASS"
               if (structure_ok and masked_ok and refs_ok and reconcile_ok
                   and no_write_ok)
               else "PROTOCOL_FAILURE")

    report = {
        "experiment_id": "v1-bounded-fault-selection-wiring-v2",
        "note": "S1-v2：v1（w1_bounded_fault_selection_wiring_report.json，"
                "PROTOCOL_FAILURE 保留）暴露接口契约缺口——prompt 未教"
                "material threshold（gain < 0.005 才算 failed——符号正"
                "的 +0.004 仍失败）与 guard 确定性规则 → t105 误选 "
                "INSUFFICIENT_EVIDENCE 且指向跨 guard 类型的 case。"
                "协议修复（只改 prompt 契约文本——不动模型/不动 Case/"
                "不动验证规则）后重跑一次。verdict 仅限 WIRING_PASS——"
                "不证明五类归因可靠、不证明 Case Memory 有收益——"
                "development exposure——零新 Claim",
        "model": smoke.MODEL, "base_url": smoke.BASE_URL,
        "temperature": 0,
        "records": records,
        "pass_conditions": {
            "structure_2of2": structure_ok,
            "no_masked_selection_2of2": masked_ok,
            "evidence_refs_2of2": refs_ok,
            "runtime_reconciliation_normal": reconcile_ok,
            "no_write_permission": no_write_ok,
        },
        "verdict": verdict,
    }
    print("== pass_conditions:", json.dumps(report["pass_conditions"],
                                            ensure_ascii=False))
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
