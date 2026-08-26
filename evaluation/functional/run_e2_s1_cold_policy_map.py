"""S1-diag (arbitration revision): behavior funnel + candidate-cap pairing.

Original Part B (full-menu ranking probe) is void and is not executed.
This runner writes ``artifacts/functional/e2/s1_cold_policy_map.json/.md``.

Part A is 0 LLM: a per-event funnel on already-paid live artifacts.
Part B is propose-only, LLM <= 12, one variable (candidate cap K vs K=5).

Does not modify ``methods/``, ``runtime/``, ``contracts/``, ``operators/``,
or existing runners.  Probe outputs are isolated and must never enter a
future arm prompt or store.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import run_e2_s1a_curriculum_oracle_audit as s1a  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "s1_cold_policy_map.json"
OUT_MD = E2 / "s1_cold_policy_map.md"
PROBE_DIR = E2 / "s1_cold_policy_map_probes"

PROTOCOL = "s1_cold_policy_map_v2_arbitration"
LLM_CAP = 12
ORIGINAL_K = 3
EXPANDED_K = 5
ADAPTIVE_ARMS = ("A3-reset", "K0-fixed", "A5-online")

S1C_JSON = E2 / "s1_course_forward_run1.json"
HISTORICAL = (
    {
        "stratum_id": "t6_cls_op_r2_three_arms",
        "path": E2 / "t6_cls_op_r2_three_arms.json",
        "runner": "evaluation/functional/run_e2_t6_cls_op_shared_harness.py --r2-run",
        "rounds_locator": "part_c.rounds",
    },
    {
        "stratum_id": "t6_cls_op_r2_a5_replay",
        "path": E2 / "t6_cls_op_r2_a5_replay.json",
        "runner": "evaluation/functional/run_e2_t6_cls_op_shared_harness.py --r2-replay-a5",
        "rounds_locator": "part_c.rounds",
    },
    {
        "stratum_id": "t6_cls_conf_dev_ecg200",
        "path": E2 / "t6_cls_conf_dev_ecg200.json",
        "runner": "evaluation/functional/run_e2_t6_cls_op_shared_harness.py --conf-dev-run",
        "rounds_locator": "rounds",
    },
)

# Post-hoc oracle labels from the frozen S1c course.  Never inserted into a
# Part B prompt.  These five are the "含菜单正解" units of the 15 opportunities.
S1C_ORACLE = {
    "MiddlePhalanxOutlineCorrect__impulse_v2": "repair_level_shift",
    "DistalPhalanxOutlineCorrect__burst_cls2": "outlier_iqr",
    "PowerCons__impulse_v2": "hampel_filter",
    "GunPointOldVersusYoung__impulse_v2": "hampel_filter",
    "ECG200__impulse_v2": "repair_burst_segment",
}

PART_B_ORDER = (
    "PowerCons__impulse_v2",
    "GunPointOldVersusYoung__impulse_v2",
    "DistalPhalanxOutlineCorrect__burst_cls2",
    "MiddlePhalanxOutlineCorrect__impulse_v2",
    "ECG200__impulse_v2",
)

RANKING_PROBE_STATUS = (
    "voided_not_run: the full-menu ranking probe was never executed in this "
    "session; there is no exploratory ranking material to archive"
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=False,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _plain(value: Any) -> Any:
    return cls._plain(value)


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(payload: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = payload
    for part in dotted.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


# =========================================================================== #
# candidate-id aliasing (instrument: raw propose payload was not persisted)
# =========================================================================== #
def infer_ops_from_candidate_id(candidate_id: str) -> dict[str, Any]:
    blob = str(candidate_id or "").lower().replace("-", "_")
    hits: list[str] = []
    if "hampel" in blob:
        hits.append("hampel_filter")
    if "winsor" in blob:
        hits.append("winsorize")
    if "iqr" in blob:
        hits.append("outlier_iqr")
    if "mad" in blob:
        hits.append("outlier_mad")
    if "burst" in blob:
        hits.append("repair_burst_segment")
    if any(token in blob for token in (
            "level_shift", "level_excursion", "levelshift")):
        hits.append("repair_level_shift")
    unique = list(dict.fromkeys(hits))
    if len(unique) == 1:
        return {"operators": unique, "id_resolution": "alias_unique"}
    if len(unique) > 1:
        return {"operators": unique, "id_resolution": "alias_ambiguous"}
    if "extreme_deviation" in blob or "outlier" in blob:
        return {"operators": [], "id_resolution": "family_unspecified_outlier"}
    if candidate_id in (None, "", "identity"):
        return {"operators": ["identity"], "id_resolution": "identity"}
    return {"operators": [], "id_resolution": "unresolved"}


def _ops_from_winner(winner: Any) -> list[str]:
    if not winner:
        return []
    if isinstance(winner, Mapping) and winner.get("op"):
        return [str(winner["op"])]
    if isinstance(winner, Sequence) and not isinstance(winner, (str, bytes)):
        return [str(step.get("op")) for step in winner
                if isinstance(step, Mapping) and step.get("op")]
    return []


def _ops_from_episodes(episodes: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(ep.get("workflow_signature")) for ep in episodes
            if ep.get("workflow_signature")]


# =========================================================================== #
# Part A -- behavior funnel
# =========================================================================== #
def _instrument_census(sample_rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = set()
    for record in sample_rounds:
        keys.update(record.keys())
    raw_fields = [
        name for name in (
            "raw_proposals", "propose_payload", "agent_proposals",
            "llm_proposals", "stage_payloads", "candidate_program_steps",
            "rejection_receipts", "inspect_payload", "public_features",
        ) if name in keys
    ]
    return {
        "persisted_round_fields_sampled": sorted(keys),
        "raw_proposal_fields_found": raw_fields,
        "raw_proposals_persisted": bool(raw_fields),
        "earliest_available_layer": (
            "compiled_verified_pool"
            if "pool" in keys else "executed_probes_only"
            if "probes" in keys else "unknown"
        ),
        "gap": (
            "S1c and the listed historical runners persist DecisionTrace."
            "candidate_ids as `pool` after compile + noop filter + "
            "CandidatePool.build(total_k) + verifier selectable filter.  "
            "They do not persist the propose-stage payload (all raw Agent "
            "candidates before truncation/verify).  `proposal_count` is "
            "len(non-identity pool after that filter).  Funnel therefore "
            "starts at compiled_verified_pool, not at the raw LLM list.  "
            "Operator names on unexecuted pool members are recovered from "
            "candidate_id aliases; executed members have workflow_signature."
        ),
    }


def _round_events(
    *,
    stratum: Mapping[str, Any],
    record: Mapping[str, Any],
    unit_id: str,
    oracle: str | None,
    deployment: Mapping[str, Any] | None,
    observation_note: Mapping[str, Any],
) -> dict[str, Any]:
    pool = [str(item) for item in (record.get("pool") or [])]
    chosen = record.get("chosen")
    probes = list(record.get("probes") or [])
    episodes = list(record.get("episodes") or [])
    winner_ops = _ops_from_winner(record.get("winner_program"))
    executed_ops = _ops_from_episodes(episodes)
    inferred_pool = []
    for cid in pool:
        if cid == "identity":
            inferred_pool.append({
                "candidate_id": cid, "operators": ["identity"],
                "id_resolution": "identity"})
            continue
        inferred_pool.append({
            "candidate_id": cid, **infer_ops_from_candidate_id(cid)})

    selected_rejected = [
        probe for probe in probes
        if probe.get("kind") == "verifier_rejected"
    ]
    executed = [
        probe for probe in probes if probe.get("kind") == "probe"
    ]
    support_rows = []
    delayed_rows = []
    for episode in episodes:
        support_rows.append({
            "program": episode.get("workflow_signature"),
            "relation": (episode.get("support_effect") or {}).get(
                "relation") or episode.get("relation"),
            "gain": episode.get("support_gain"),
        })
        delayed_rows.append({
            "program": episode.get("workflow_signature"),
            "relation": (episode.get("delayed_effect") or {}).get(
                "relation"),
            "gain": episode.get("delayed_gain"),
            "evidence_level": episode.get("evidence_level"),
        })
    deployed = None
    if deployment:
        deployed = {
            "source": deployment.get("deploy_source"),
            "program": _ops_from_winner(
                deployment.get("applied_program")
                or deployment.get("scored_program")),
        }

    proposed_ops = []
    for row in inferred_pool:
        proposed_ops.extend(row.get("operators") or [])
    proposed_ops = list(dict.fromkeys(op for op in proposed_ops
                                      if op and op != "identity"))
    # executed signatures override alias guesses
    proposed_ops = list(dict.fromkeys(proposed_ops + executed_ops + winner_ops))

    breakpoint = "no_oracle_on_unit"
    if oracle:
        in_pool = oracle in proposed_ops or any(
            oracle in (row.get("operators") or [])
            or (row.get("id_resolution") == "alias_unique"
                and oracle in (row.get("operators") or []))
            for row in inferred_pool
        )
        # alias_unique on the oracle name, or executed signature
        in_pool = (
            oracle in executed_ops
            or oracle in winner_ops
            or any(oracle in (row.get("operators") or [])
                   and row.get("id_resolution") == "alias_unique"
                   for row in inferred_pool)
        )
        selected_ops = infer_ops_from_candidate_id(str(chosen or "")).get(
            "operators") or []
        if chosen and chosen != "identity":
            selected_ops = list(dict.fromkeys(
                selected_ops + infer_ops_from_candidate_id(str(chosen))[
                    "operators"]))
        oracle_selected = oracle in selected_ops or oracle in winner_ops
        oracle_verifier_rejected = any(
            oracle in infer_ops_from_candidate_id(
                str(probe.get("candidate_id"))).get("operators") or []
            for probe in selected_rejected
        )
        oracle_executed = oracle in executed_ops
        support_pos = any(
            row.get("program") == oracle
            and row.get("relation") == "POSITIVE"
            for row in support_rows
        )
        delayed_pos = any(
            row.get("program") == oracle
            and row.get("relation") == "POSITIVE"
            for row in delayed_rows
        ) or bool(record.get("winner_delayed_approved"))
        deployed_ops = (deployed or {}).get("program") or []
        oracle_deployed = oracle in deployed_ops
        if not in_pool:
            breakpoint = "not_proposed"
        elif not oracle_selected and not oracle_executed:
            breakpoint = "proposed_not_selected"
        elif oracle_verifier_rejected and not oracle_executed:
            breakpoint = "selected_rejected"
        elif oracle_executed and not (support_pos and delayed_pos) and not oracle_deployed:
            breakpoint = "executed_not_passed"
        elif (support_pos or delayed_pos) and not oracle_deployed:
            breakpoint = "passed_not_deployed"
        elif oracle_deployed:
            breakpoint = "deployed"
        else:
            breakpoint = "executed_not_passed"

    return {
        "stratum_id": stratum["stratum_id"],
        "runner": stratum["runner"],
        "protocol_version": stratum.get("protocol_version"),
        "git_head": stratum.get("git_head"),
        "prompt_surface": stratum.get("prompt_surface"),
        "task_context_id": stratum.get("task_context_id"),
        "arm": record.get("arm"),
        "unit_id": unit_id,
        "dataset": record.get("dataset"),
        "round": record.get("round"),
        "menu_oracle_program_posthoc": oracle,
        "agent_visible_observation": observation_note,
        "layers": {
            "menu_available": bool(oracle) if oracle else None,
            "raw_proposals_persisted": False,
            "compiled_verified_pool": pool,
            "compiled_pool_inferred": inferred_pool,
            "proposal_count_persisted": record.get("proposal_count"),
            "selected": chosen,
            "abstained": bool(record.get("abstained")),
            "verifier_rejected": selected_rejected,
            "executed_probes": executed,
            "executed_operators": executed_ops,
            "support": support_rows,
            "delayed": delayed_rows,
            "winner_delayed_approved": record.get("winner_delayed_approved"),
            "frozen_deployed": deployed,
        },
        "breakpoint": breakpoint,
        "memory_resolution": record.get("memory_resolution"),
        "llm_calls_this_round": record.get("llm_calls_this_round"),
    }


def _s1c_stratum(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stratum_id": "s1c_course_forward_run1",
        "runner": (
            "evaluation/functional/run_e2_s1_curriculum_four_arms.py "
            "--run-course --order forward"),
        "protocol_version": payload.get("protocol_version"),
        "git_head": payload.get("git_head"),
        "prompt_surface": (
            "h0 instruction.md + three bootstrap Skills; Fast stages "
            "inspect/propose/select; live gpt-5.6-sol @ agicto; "
            "classification TaskContext maximum_candidates=3; "
            "fast_propose_v1 maxItems=3; h0 agent_program_slots=3; "
            "effective pool min(total_k=4, maximum_candidates=3) = "
            "identity + 2 program slots"),
        "task_context_id": "classification-fixed-consumer-v1",
        "backend": payload.get("backend"),
        "returned_model": (payload.get("backend_probe") or {}).get(
            "returned_model"),
    }


def _observation_note_s1c() -> dict[str, Any]:
    return {
        "what_agent_saw_persisted": False,
        "available": (
            "observation_block is reconstructible from the same UCR TRAIN + "
            "injection used to build the cell; public features are "
            "deterministic from that block.  The live inspect payload "
            "(inspected_region_fractions, pattern_hypotheses) and the "
            "per-round rendered Memory/Skill view were not saved."
        ),
        "posthoc_oracle_label": (
            "menu_oracle_program is a judge-time label from the sealed "
            "oracle / frozen course.  It was not in the Agent prompt."
        ),
    }


def _collect_s1c(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stratum = _s1c_stratum(payload)
    events: list[dict[str, Any]] = []
    by_unit_arm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in payload.get("arm_results") or []:
        unit_id = str(result.get("unit_id") or "")
        deployment = result.get("deployment")
        for record in result.get("rounds") or []:
            event = _round_events(
                stratum=stratum, record=record, unit_id=unit_id,
                oracle=S1C_ORACLE.get(unit_id),
                deployment=deployment,
                observation_note=_observation_note_s1c(),
            )
            events.append(event)
            by_unit_arm[(unit_id, str(result.get("arm")))].append(event)

    opportunities: list[dict[str, Any]] = []
    for unit_id, oracle in S1C_ORACLE.items():
        for arm in ADAPTIVE_ARMS:
            rows = by_unit_arm.get((unit_id, arm), [])
            proposed = any(
                row["breakpoint"] not in ("not_proposed", "no_oracle_on_unit")
                for row in rows)
            executed = any(
                oracle in (row["layers"]["executed_operators"] or [])
                for row in rows)
            deployed = any(
                oracle in ((row["layers"]["frozen_deployed"] or {}).get(
                    "program") or [])
                for row in rows)
            best = "not_proposed"
            order = (
                "deployed", "passed_not_deployed", "executed_not_passed",
                "selected_rejected", "proposed_not_selected", "not_proposed",
            )
            for name in order:
                if any(row["breakpoint"] == name for row in rows):
                    best = name
                    break
            if not rows:
                best = "no_round_recorded"
            opportunities.append({
                "unit_id": unit_id,
                "arm": arm,
                "oracle": oracle,
                "n_rounds": len(rows),
                "oracle_proposed_at_pool_or_executed": proposed,
                "oracle_executed": executed,
                "oracle_deployed": deployed,
                "deepest_breakpoint": best,
            })
    n_hit_deployed = sum(1 for row in opportunities if row["oracle_deployed"])
    n_hit_executed = sum(1 for row in opportunities if row["oracle_executed"])
    n_hit_proposed = sum(
        1 for row in opportunities if row["oracle_proposed_at_pool_or_executed"])
    wording = (
        "在当前课程、当前 Prompt、当前候选预算、这一次随机运行中，"
        "15 个含菜单正解的臂-单元机会命中 %d 次（冻结部署口径）。"
        "同一 15 个机会里，菜单正解被执行 %d 次、在编译池/执行层出现 %d 次。"
        "这是这一次冷提案策略的召回读数，不是数据就绪的固有难度，也不是稳定概率。"
        % (n_hit_deployed, n_hit_executed, n_hit_proposed)
    )
    return events, {
        "n_adaptive_menu_oracle_opportunities": 15,
        "n_deployed_hits": n_hit_deployed,
        "n_executed_hits": n_hit_executed,
        "n_proposed_or_executed_hits": n_hit_proposed,
        "wording_required": wording,
        "opportunities": opportunities,
    }


def _collect_historical() -> list[dict[str, Any]]:
    strata: list[dict[str, Any]] = []
    for spec in HISTORICAL:
        path = spec["path"]
        if not path.exists():
            strata.append({
                "stratum_id": spec["stratum_id"],
                "missing": True, "path": str(path)})
            continue
        payload = _load(path)
        stratum = {
            "stratum_id": spec["stratum_id"],
            "runner": spec["runner"],
            "protocol_version": payload.get("protocol_version"),
            "git_head": payload.get("git_head"),
            "prompt_surface": (
                "CLS-OP shared Fast path; classification TaskContext "
                "maximum_candidates=3; not the S1c course"),
            "task_context_id": "classification-fixed-consumer-v1",
            "run_id": payload.get("run_id"),
            "entry": payload.get("entry"),
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }
        rounds = _nested(payload, spec["rounds_locator"]) or []
        events = []
        for record in rounds:
            unit_id = str(
                record.get("unit_id")
                or record.get("dataset")
                or "")
            events.append(_round_events(
                stratum=stratum, record=record, unit_id=unit_id,
                oracle=None, deployment=None,
                observation_note={
                    "what_agent_saw_persisted": False,
                    "available": (
                        "same instrument gap as S1c: pool/chosen/probes/"
                        "episodes only; inspect payload not saved"),
                },
            ))
        sample = [record for record in rounds[:3]] if rounds else []
        strata.append({
            **stratum,
            "n_rounds": len(events),
            "instrument": _instrument_census(sample),
            "events": events,
            "do_not_mix_into_s1c_hit_rate": True,
        })
    return strata


def _slow_readout(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for item in payload.get("a5_slow_integrations") or []:
        rows.append({
            "after_unit": item.get("after_unit"),
            "into_unit": item.get("into_unit"),
            "slow_llm_calls": item.get("slow_llm_calls"),
            "slow_llm_cap": item.get("slow_llm_cap"),
            "skill_written": item.get("skill_written"),
            "entry_skill_id": item.get("entry_skill_id"),
            "authorized_try_operators": item.get("authorized_try_operators"),
            "risk_authorized_operators": item.get("risk_authorized_operators"),
            "execution_right_granted": item.get("execution_right_granted"),
            "withheld": [
                row.get("withheld_because")
                for row in (item.get("authorization_audit") or [])
                if row.get("withheld_because")
            ],
        })
    used = sum(int(row.get("slow_llm_calls") or 0) for row in rows)
    return {
        "n_boundaries": len(rows),
        "slow_llm_used": used,
        "slow_llm_cap_per_boundary": 6,
        "empty_carry_reason": (
            "each boundary spent 1 of 6 allowed Slow LLM calls and wrote "
            "source_investigation_cls_v1 (already in K0, dropped at the "
            "next-unit wall).  authorized_try and risk_authorized stayed "
            "empty because the census had no unguided positive and no "
            "two-Task same-operator harm.  The empty carry is missing "
            "compilable evidence, not a Slow-budget starve."
        ),
        "boundaries": rows,
    }


def _funnel_summary(opportunities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["deepest_breakpoint"] for row in opportunities)
    return {
        "counts_by_deepest_breakpoint": dict(counts),
        "attribution": {
            "not_proposed": "提案召回问题",
            "proposed_not_selected": "选择或弃权问题",
            "selected_rejected": "合法性问题",
            "executed_not_passed": "反馈或效果问题",
            "passed_not_deployed": "部署规则问题",
            "deployed": "漏斗走通",
            "no_round_recorded": "该臂-单元没有适应轮（例如 A5 在某单元 0 probe）",
        },
    }


def run_part_a() -> dict[str, Any]:
    s1c = _load(S1C_JSON)
    s1c_events, s1c_hits = _collect_s1c(s1c)
    sample = []
    for result in s1c.get("arm_results") or []:
        sample.extend(result.get("rounds") or [])
        if len(sample) >= 6:
            break
    historical = _collect_historical()
    return {
        "instrument_census": _instrument_census(sample),
        "s1c": {
            "stratum": _s1c_stratum(s1c),
            "n_round_events": len(s1c_events),
            "events": s1c_events,
            "menu_oracle_opportunities": s1c_hits,
            "funnel_summary": _funnel_summary(s1c_hits["opportunities"]),
            "slow": _slow_readout(s1c),
        },
        "historical_strata": historical,
        "mixing_rule": (
            "S1c 的 15 机会读数只在 s1c_course_forward_run1 层内计算。"
            "historical_strata 各自成层，不得并入一个总命中率。"
        ),
    }


# =========================================================================== #
# Part B -- candidate-cap pairing, propose only
# =========================================================================== #
def _unit_meta(unit_id: str) -> dict[str, str]:
    dataset, injection = unit_id.split("__", 1)
    return {"unit_id": unit_id, "dataset": dataset, "injection": injection}


def _frozen_inspect(features: Mapping[str, Any]) -> dict[str, Any]:
    """Identical inspect payload for both K arms.  Feature-derived only.

    This is not a replay of S1c's stochastic inspect (that payload was not
    saved).  Both K conditions see these exact bytes so the pair isolates K.
    Operator names are never written into the hypotheses.
    """
    def _frac(name: str) -> float | None:
        value = features.get(name)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    outlier = _frac("outlier_region_fraction")
    level = _frac("level_region_fraction")
    o_start = _frac("outlier_region_start_fraction") or 0.0
    o_end = _frac("outlier_region_end_fraction") or 1.0
    l_start = _frac("level_region_start_fraction") or 0.0
    l_end = _frac("level_region_end_fraction") or 1.0
    regions = [[0.0, 1.0]]
    hyps: list[dict[str, Any]] = []
    if outlier is not None and outlier > 0:
        regions = [[max(0.0, min(o_start, 1.0)), max(0.0, min(o_end, 1.0))]]
        hyps.append({
            "hypothesis_id": "pub_extreme_deviation",
            "pattern_type": "extreme_deviation",
            "region_fractions": regions[0],
            "evidence_features": ["outlier_region_fraction"],
            "confidence": "medium" if outlier >= 0.02 else "low",
        })
    if level is not None and level > 0 and len(hyps) < 2:
        hyps.append({
            "hypothesis_id": "pub_level_excursion",
            "pattern_type": "level_excursion",
            "region_fractions": [
                max(0.0, min(l_start, 1.0)), max(0.0, min(l_end, 1.0))],
            "evidence_features": ["level_region_fraction"],
            "confidence": "medium" if level >= 0.02 else "low",
        })
    if not hyps:
        hyps.append({
            "hypothesis_id": "pub_no_actionable_signal",
            "pattern_type": "no_actionable_signal",
            "region_fractions": [0.0, 1.0],
            "evidence_features": [
                key for key in (
                    "outlier_region_fraction", "level_region_fraction",
                    "missing_fraction")
                if key in features
            ] or ["outlier_region_fraction"],
            "confidence": "low",
        })
    return {
        "inspected_region_fractions": regions,
        "requested_public_tools": [],
        "uncertainty": "medium",
        "pattern_hypotheses": hyps,
    }


def _context_for_k(k: int) -> Any:
    from SelfEvolvingHarnessTS.contracts.task import (
        classification_task_context_v1,
        classification_local_event_task_quality_contract_v1,
        deployment_constraints_v1,
    )
    return classification_task_context_v1(
        task_spec=cls._task_spec(),
        quality_contract=classification_local_event_task_quality_contract_v1(),
        deployment_constraints=deployment_constraints_v1(
            constraint_id="classification-fixed-consumer-v1",
            fixed_downstream_model_id="fixed:classification-consumer-v1",
            maximum_candidates=k,
            maximum_modified_fraction=0.1),
    )


def _view_for_k(view: Any, k: int) -> Any:
    controls = json.loads(json.dumps(cls._plain(view.controls)))
    policy = dict(controls.get("candidate_policy") or {})
    policy["agent_program_slots"] = int(k)
    policy["total_k"] = 1 + int(k)
    controls["candidate_policy"] = policy
    return dataclasses.replace(view, controls=controls)


def _propose_schema(k: int) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.schema_contracts import (
        load_stage_schema,
    )
    schema = json.loads(json.dumps(cls._plain(load_stage_schema("fast_propose_v1"))))
    schema["properties"]["candidates"]["maxItems"] = int(k)
    return schema


def _assert_prompt_clean(public_input: Mapping[str, Any], oracle: str) -> None:
    blob = json.dumps(cls._plain(public_input), ensure_ascii=False)
    forbidden = (
        "oracle", "menu_oracle", "rank all", "full menu ranking",
        "sort every operator", "排名", "全菜单排序",
    )
    hits = [token for token in forbidden if token.lower() in blob.lower()]
    if hits:
        raise RuntimeError("prompt leaked ranking/oracle tokens: %s" % hits)
    # The legal menu lists operator names, including the oracle operator.
    # That is the S1c menu, not an oracle hint.  Refuse only extra pointing.
    if "the correct operator is" in blob.lower():
        raise RuntimeError("prompt pointed at the oracle")


def _extract_ops(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in payload.get("candidates") or []:
        steps = candidate.get("steps") or []
        rows.append({
            "candidate_id": candidate.get("candidate_id"),
            "operators": [str(step.get("op")) for step in steps
                          if isinstance(step, Mapping) and step.get("op")],
            "addresses_hypothesis_id": candidate.get(
                "addresses_hypothesis_id"),
        })
    return rows


def run_part_b() -> dict[str, Any]:
    from SelfEvolvingHarnessTS.contracts.method import PreparationRequest
    from SelfEvolvingHarnessTS.methods.ttha import signed_radius as resolver
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _actionable_operators,
        _allowed_operators,
        _noop_ops_for_context,
        _task_binding,
        public_operator_contract,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
        extract_public_features,
    )
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (
        resolve_harness_view,
    )
    from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA

    started = time.time()
    backend = cls._live_backend(LLM_CAP)
    snapshot = compile_snapshot(
        PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
        verify_lock=False)
    pairs: list[dict[str, Any]] = []
    stop_reason = None
    isolation = {
        "probe_dir": str(PROBE_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "declaration": (
            "Part B outputs are diagnostic isolation material.  They must "
            "not enter any future arm Fast view, Skill store, Memory, or "
            "prompt.  They are not a ranking of the menu and they do not "
            "authorize a program."
        ),
    }
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    for unit_id in PART_B_ORDER:
        oracle = S1C_ORACLE[unit_id]
        meta = _unit_meta(unit_id)
        cell, reason = s1a._r3_build_cell({
            "dataset": meta["dataset"],
            "injection": meta["injection"],
        })
        if cell is None:
            pairs.append({
                "unit_id": unit_id, "oracle_posthoc": oracle,
                "error": "cell_failed:%s" % reason})
            continue
        block = np.asarray(cell["observation_block"], dtype=np.float64)
        features = dict(extract_public_features(
            block, task_kind="classification"))
        inspect_payload = _frozen_inspect(features)
        observed = dict(resolver.window_context(
            {"heldin_observation": block}, int(block.size), cls.PERIOD_HINT))
        observed["bound_period"] = float(cls.PERIOD_HINT)
        agent = cls._live_agent(block, backend.new_arm_backend())
        ctx_orig = _context_for_k(ORIGINAL_K)
        request_orig = PreparationRequest(
            "s1-diag-%s" % unit_id, block, cls._task_spec(), dict(observed),
            task_context=ctx_orig)
        view = resolve_harness_view(snapshot, features, role="fast")
        allowed = _allowed_operators(request_orig)
        actionable = _actionable_operators(
            request_orig, np.asarray(block, dtype=float), view, allowed)
        noop = set(_noop_ops_for_context(request_orig))
        propose_ops = [name for name in actionable if name not in noop]
        bound_ok = [
            name for name in propose_ops
            if (OPERATOR_METADATA[name].get("public_parameter_bindings")
                and all(feat in features for feat in
                        OPERATOR_METADATA[name]["public_parameter_bindings"].values()))
        ]
        propose_ops = bound_ok + [name for name in propose_ops
                                  if name not in bound_ok]
        contracts = [public_operator_contract(name) for name in propose_ops]
        pair_row: dict[str, Any] = {
            "unit_id": unit_id,
            "dataset": meta["dataset"],
            "injection": meta["injection"],
            "oracle_posthoc": oracle,
            "oracle_in_prompt": False,
            "shared": {
                "observation_block_sha": hashlib.sha256(
                    np.asarray(block).tobytes()).hexdigest(),
                "n_observation_points": int(block.size),
                "inspect_payload": inspect_payload,
                "menu_operator_names": list(propose_ops),
                "task_spec": cls._task_spec().to_dict(),
                "system_instruction": snapshot.instruction,
            },
            "conditions": {},
        }

        for k, label in ((EXPANDED_K, "k5"), (ORIGINAL_K, "k_original")):
            used = int(getattr(backend, "calls", 0) or 0)
            if used >= LLM_CAP:
                stop_reason = "LLM_CAP_REACHED_BEFORE_%s_%s" % (unit_id, label)
                break
            ctx = _context_for_k(k)
            request = PreparationRequest(
                "s1-diag-%s" % unit_id, block, cls._task_spec(),
                dict(observed), task_context=ctx)
            view_k = _view_for_k(view, k)
            public_input = {
                **_task_binding(request),
                "features": cls._plain(features),
                "inspection": copy.deepcopy(inspect_payload),
                "fixed_probe_panel": {},
                "allowed_operator_contracts": contracts,
            }
            _assert_prompt_clean(public_input, oracle)
            before = int(getattr(backend, "calls", 0) or 0)
            error = None
            payload: dict[str, Any] = {}
            try:
                result = agent.core.run_stage(
                    role="fast",
                    stage="propose",
                    case_id=request.series_uid,
                    public_input=public_input,
                    harness_view=view_k,
                    output_schema_name="fast_propose_v1",
                    output_schema=_propose_schema(k),
                    source_snapshot_sha=snapshot.runtime_bundle_sha,
                    task_context_sha=request.task_context.sha(),
                    validation_retries=1,
                )
                payload = dict(result.payload or {})
            except Exception as exc:  # noqa: BLE001
                error = "%s: %s" % (type(exc).__name__, exc)
            after = int(getattr(backend, "calls", 0) or 0)
            candidates = _extract_ops(payload) if payload else []
            hit = any(oracle in row["operators"] for row in candidates)
            cond = {
                "k": k,
                "label": label,
                "llm_calls": after - before,
                "llm_calls_cumulative": after,
                "n_candidates": len(candidates),
                "candidates": candidates,
                "oracle_hit_posthoc": hit,
                "error": error,
            }
            pair_row["conditions"][label] = cond
            _dump(PROBE_DIR / ("%s_%s.json" % (unit_id, label)), {
                "isolation": isolation,
                "unit_id": unit_id,
                "k": k,
                "oracle_hit_posthoc": hit,
                "candidates": candidates,
                "error": error,
                "note": (
                    "propose-only; not executed; not a menu ranking; "
                    "oracle judged after output"
                ),
            })
        pairs.append(pair_row)
        if stop_reason:
            break

    reading = _part_b_reading(pairs)
    return {
        "cap": LLM_CAP,
        "original_k": ORIGINAL_K,
        "expanded_k": EXPANDED_K,
        "k_operationalization": (
            "one variable K, realized jointly by fast_propose_v1 maxItems, "
            "harness-view candidate_policy.agent_program_slots/total_k, and "
            "TaskContext.deployment_constraints.maximum_candidates.  "
            "Original K=3 is the S1c live cap (schema maxItems=3 and "
            "TaskContext maximum_candidates=3).  Expanded K=5 is the same "
            "observation, instruction text, menu contracts, and frozen "
            "inspect, with only those three K fields raised.  methods/h0 "
            "files were not written."
        ),
        "inspect_note": (
            "inspect was frozen from public features and shared across the "
            "pair.  S1c's stochastic inspect payload was not persisted, so "
            "a live inspect replay would have added a second random variable "
            "and would have exhausted the 12-call cap (inspect+tools+two "
            "proposes).  This isolates K; it is not a claim that the inspect "
            "equals S1c's inspect."
        ),
        "llm_calls": int(getattr(backend, "calls", 0) or 0),
        "seconds": round(time.time() - started, 2),
        "stop_reason": stop_reason,
        "isolation": isolation,
        "pairs": pairs,
        "reading": reading,
        "ranking_probe": RANKING_PROBE_STATUS,
    }


def _part_b_reading(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_unit = []
    for row in pairs:
        oracle = row.get("oracle_posthoc")
        orig = (row.get("conditions") or {}).get("k_original") or {}
        k5 = (row.get("conditions") or {}).get("k5") or {}
        orig_hit = bool(orig.get("oracle_hit_posthoc"))
        k5_hit = bool(k5.get("oracle_hit_posthoc"))
        if k5 and not k5.get("error") and not k5_hit:
            # Book: K=5 still absent → proposal semantics insufficient.
            # Do not wait for the original-K half if the expanded cap already
            # missed; the original-K column is then only a pairing note.
            label = "proposal_semantics_insufficient"
        elif orig.get("error") and k5.get("error"):
            label = "unreadable"
        elif k5_hit and orig and (not orig_hit) and not orig.get("error"):
            label = "truncation_this_draw"
        elif orig_hit and not k5_hit:
            label = "original_k_only_this_draw"
        elif orig_hit and k5_hit:
            label = "both_proposed"
        else:
            label = "incomplete_pair"
        per_unit.append({
            "unit_id": row.get("unit_id"),
            "oracle_posthoc": oracle,
            "original_k_hit": orig_hit,
            "k5_hit": k5_hit,
            "label": label,
        })
    labels = Counter(row["label"] for row in per_unit)
    if labels.get("proposal_semantics_insufficient", 0) >= 3 and labels.get(
            "truncation_this_draw", 0) == 0:
        frozen = "proposal_semantics_insufficient"
        text = (
            "K=5 在本次已完成的单元抽签中仍未出现菜单正解。"
            "扩大帽也没有把候选数抬到接近 5。按冻结判读只写"
            "「提案语义不足」，不得再拆成 observation 不足或策略偏置。"
            "n=1 对/单元，不是稳定概率。"
        )
    elif labels.get("truncation_this_draw", 0) >= 3:
        frozen = "truncation_this_draw"
        text = (
            "本次配对抽签中，扩大帽 K=5 才出现菜单正解、原帽未出现的单元占多数，"
            "读作截断解释。n=1 对/单元，不得写成稳定概率。"
        )
    elif labels.get("both_proposed", 0) >= 3:
        frozen = "selection_or_abstention_candidate"
        text = (
            "原帽已经提出菜单正解。若 S1c 当时未执行，读作选择/弃权解释，"
            "不是提案召回为零。"
        )
    else:
        frozen = "mixed_or_incomplete"
        text = (
            "五单元配对未形成单一冻结解释。按单元读数，不合成一个总命中率。"
        )
    return {
        "per_unit": per_unit,
        "label_counts": dict(labels),
        "frozen_reading": frozen,
        "frozen_text": text,
        "do_not_split_semantics": (
            "If the reading is proposal_semantics_insufficient, do not "
            "subdivide into observation insufficiency vs policy bias.  "
            "That needs a future observation-expansion ablation.  This book "
            "does not run it."
        ),
    }


def _s1c_original_k_from_funnel(
    opportunities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """S1c A3-reset first-round / unit-level original-K already paid."""
    rows = []
    for row in opportunities:
        if row["arm"] != "A3-reset":
            continue
        rows.append({
            "unit_id": row["unit_id"],
            "oracle": row["oracle"],
            "oracle_proposed_or_executed_in_s1c_a3": row[
                "oracle_proposed_at_pool_or_executed"],
            "oracle_executed_in_s1c_a3": row["oracle_executed"],
            "oracle_deployed_in_s1c_a3": row["oracle_deployed"],
            "deepest_breakpoint": row["deepest_breakpoint"],
        })
    return {
        "note": (
            "S1c A3-reset is the already-paid original-K live propose under "
            "the course Prompt and cap.  It is not byte-paired to Part B's "
            "frozen inspect; it is the historical original-K column."
        ),
        "a3_reset": rows,
    }


def _graded_track_paragraph() -> str:
    return (
        "在这一次 S1c 课程里，15 个含菜单正解的臂-单元机会有 11 个断在"
        "「未提出」、3 个断在「执行后未过」（MiddlePhalanx 三臂都执行了 "
        "repair_level_shift 但 delayed 未批准）、1 个走通部署。"
        "因此「一张 Scope 匹配的 hypothesis 卡提高正确族进入有限帽的概率」"
        "是针对召回断点的合理假设，待因果实验验证，不是本诊断或 S1c 已经"
        "证明的事实。仍须走完：合法独立 Episode → 机器 Scope 匹配 → "
        "专用 prior 槽 → 提案分布实测改变 → Target 自批 Support/delayed → "
        "成本或 regret 改善且 harm 不升。本书封顶，无 diag-r2/r3。"
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    a = payload["part_a"]
    b = payload["part_b"]
    hits = a["s1c"]["menu_oracle_opportunities"]
    funnel = a["s1c"]["funnel_summary"]
    lines = [
        "# S1-diag -- behavior funnel + candidate-cap pairing",
        "",
        "protocol: `%s`  git: `%s`  Part B LLM: %s / %s"
        % (payload["protocol_version"], payload["git_head"],
           b.get("llm_calls"), LLM_CAP),
        "",
        "Original Part B (full-menu ranking) is **void** and was not run.",
        "",
        RANKING_PROBE_STATUS,
        "",
        "## Part A -- behavior funnel (0 LLM)",
        "",
        a["instrument_census"]["gap"],
        "",
        "Earliest available layer: **%s**.  Raw proposals persisted: **%s**."
        % (a["instrument_census"]["earliest_available_layer"],
           a["instrument_census"]["raw_proposals_persisted"]),
        "",
        "### Wording (required)",
        "",
        hits["wording_required"],
        "",
        "### S1c 15 opportunities (do not mix other strata)",
        "",
        "| unit | arm | oracle | proposed/exec layer | executed | deployed | deepest breakpoint |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in hits["opportunities"]:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (row["unit_id"], row["arm"], row["oracle"],
               row["oracle_proposed_at_pool_or_executed"],
               row["oracle_executed"], row["oracle_deployed"],
               row["deepest_breakpoint"]))
    lines += [
        "",
        "Breakpoint counts: `%s`" % json.dumps(
            funnel["counts_by_deepest_breakpoint"], ensure_ascii=False),
        "",
        "Attribution: 未提出=提案召回 / 提出未选=选择或弃权 / 选了被拒=合法性 / "
        "执行后未过=反馈或效果。",
        "",
        "### Slow boundaries (A5)",
        "",
        a["s1c"]["slow"]["empty_carry_reason"],
        "",
        "### Historical strata (not mixed into 1/15)",
        "",
    ]
    for stratum in a["historical_strata"]:
        lines.append(
            "- `%s`  rounds=%s  protocol=`%s`  git=`%s`"
            % (stratum.get("stratum_id"), stratum.get("n_rounds"),
               stratum.get("protocol_version"), stratum.get("git_head")))
    lines += [
        "",
        "## Part B -- candidate-cap pairing (propose only)",
        "",
        b.get("k_operationalization") or "(not run)",
        "",
        b.get("inspect_note") or "",
        "",
        "Isolation: %s" % (
            ((b.get("isolation") or {}).get("declaration"))
            or "Part B not run in this invocation"),
        "",
        "| unit | oracle (posthoc) | K=3 ops / hit | K=5 ops / hit | reading |",
        "|---|---|---|---|---|",
    ]
    conds_by_unit = {
        row.get("unit_id"): row.get("conditions") or {}
        for row in (b.get("pairs") or [])
    }
    for row in (b.get("reading") or {}).get("per_unit") or []:
        conds = conds_by_unit.get(row.get("unit_id")) or {}
        def _ops(label: str) -> str:
            block = conds.get(label) or {}
            if block.get("error"):
                return "error"
            names = []
            for cand in block.get("candidates") or []:
                names.extend(cand.get("operators") or [])
            return ",".join(names) if names else "(empty)"
        lines.append(
            "| %s | %s | %s / %s | %s / %s | %s |"
            % (row.get("unit_id"), row.get("oracle_posthoc"),
               _ops("k_original"), row.get("original_k_hit"),
               _ops("k5"), row.get("k5_hit"),
               row.get("label")))
    lines += [
        "",
        "Frozen reading: **%s**" % ((b.get("reading") or {}).get(
            "frozen_reading") or "n/a"),
        "",
        (b.get("reading") or {}).get("frozen_text") or "",
        "",
        "S1c A3-reset original-K column (already paid, not inspect-paired):",
        "",
    ]
    for row in (payload.get("s1c_original_k_column") or {}).get("a3_reset") or []:
        lines.append(
            "- %s oracle=%s proposed/exec=%s executed=%s deployed=%s breakpoint=%s"
            % (row["unit_id"], row["oracle"],
               row["oracle_proposed_or_executed_in_s1c_a3"],
               row["oracle_executed_in_s1c_a3"],
               row["oracle_deployed_in_s1c_a3"],
               row["deepest_breakpoint"]))
    lines += [
        "",
        "## Graded-hypothesis track",
        "",
        payload["graded_track_paragraph"],
        "",
        "## Obligations",
        "",
    ]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    lines.append("")
    return "\n".join(str(item) if item is not None else "" for item in lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-a-only", action="store_true")
    parser.add_argument(
        "--refresh-from-json", action="store_true",
        help="recompute Part B reading and markdown from the existing JSON "
             "without new LLM calls")
    args = parser.parse_args()
    python = sys.executable
    print("python=", python, flush=True)
    t0 = time.time()
    if args.refresh_from_json:
        existing = _load(OUT_JSON)
        existing["part_b"]["reading"] = _part_b_reading(
            existing["part_b"].get("pairs") or [])
        existing["graded_track_paragraph"] = _graded_track_paragraph()
        existing["git_head"] = existing.get("git_head") or _git(
            "rev-parse", "HEAD")
        _dump(OUT_JSON, existing)
        OUT_MD.write_text(_markdown(existing), encoding="utf-8")
        print("refreshed", OUT_JSON, OUT_MD, flush=True)
        print("part_b_reading",
              existing["part_b"]["reading"].get("frozen_reading"),
              flush=True)
        return 0
    part_a = run_part_a()
    if args.part_a_only:
        part_b = {
            "skipped": True, "llm_calls": 0,
            "ranking_probe": RANKING_PROBE_STATUS,
            "reading": {"per_unit": [], "frozen_reading": "skipped",
                        "frozen_text": "Part B not run"},
        }
    else:
        try:
            part_b = run_part_b()
        except Exception as exc:  # noqa: BLE001
            part_b = {
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
                "llm_calls": None,
                "ranking_probe": RANKING_PROBE_STATUS,
                "reading": {"per_unit": [], "frozen_reading": "error",
                            "frozen_text": str(exc)},
            }
    payload = {
        "protocol_version": PROTOCOL,
        "arbitration": (
            "Part B full-menu ranking voided.  Part A is a behavior funnel.  "
            "Part B is a candidate-cap pairing.  Book is capped; no diag-r2/r3."
        ),
        "git_head": _git("rev-parse", "HEAD"),
        "python": python,
        "seconds": round(time.time() - t0, 2),
        "ranking_probe": RANKING_PROBE_STATUS,
        "part_a": part_a,
        "part_b": part_b,
        "s1c_original_k_column": _s1c_original_k_from_funnel(
            part_a["s1c"]["menu_oracle_opportunities"]["opportunities"]),
        "graded_track_paragraph": _graded_track_paragraph(),
        "obligations": {
            "methods_runtime_contracts_operators_unmodified": True,
            "existing_runners_unmodified": True,
            "course_and_budgets_unmodified": True,
            "downloads": 0,
            "consumer_fits": 0,
            "llm_part_a": 0,
            "llm_part_b": part_b.get("llm_calls"),
            "llm_cap": LLM_CAP,
            "ranking_probe_not_run": True,
            "probe_outputs_isolated": True,
            "sealed_oracles_not_rewritten": True,
            "full_repo_pytest_not_run": True,
            "no_diag_r2_r3": True,
            "wording_1_of_15_is_this_run_only": True,
            "graded_hypothesis_is_hypothesis_not_fact": True,
        },
    }
    _dump(OUT_JSON, payload)
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print("wrote", OUT_JSON, OUT_MD, flush=True)
    print("part_a_hits",
          part_a["s1c"]["menu_oracle_opportunities"]["wording_required"],
          flush=True)
    print("part_b_reading",
          (part_b.get("reading") or {}).get("frozen_reading"),
          "llm", part_b.get("llm_calls"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
