"""AD thin adapter over the frozen forecasting source_skill integrator.

#42d reuses authorization_audit / authorized_try_operators / audit_sections /
build_skill_payload / slow_system / signed_summary.  It does not invent a
second Skill mechanism.  The AD constants live here; forecasting defaults
stay on source_skill.py.

Part S (#42d closeout v2) adds a temporal audit and a v2 Skill id.  The
v1 id remains the historical default so already-written artifacts keep
their name.  source_skill.py is not edited again.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.functional.task_episode_harness.agentic import source_skill as ss

SOURCE_SKILL_ID = "source_investigation_ad_v1"
SOURCE_SKILL_ID_V2 = "source_investigation_ad_v2"
SOURCE_APPLICABILITY: dict[str, Any] = {
    "feature": "task_kind", "op": "==", "value": "anomaly_detection",
}

# Filenames and cohort roots that would leak Source identity into a Skill
# written for a different domain.  audit_sections already owns the check.
SOURCE_COHORT_TOKENS: tuple[str, ...] = (
    "aws", "cloudwatch", "known_cause", "knowncause",
    "exchange", "cpc", "cpm", "nab", "ec2",
    "source_aws_cloudwatch", "source_known_cause",
    "realawscloudwatch", "realknowncause", "realadexchange",
    "ec2_cpu_utilization", "ambient_temperature_system_failure",
    "cpu_utilization_asg_misconfiguration",
    "ec2_request_latency_system_failure",
    "machine_temperature_system_failure",
    "nyc_taxi", "rogue_agent_key_hold",
    "art_daily_jumpsdown",
)

authorization_audit = ss.authorization_audit
authorized_try_operators = ss.authorized_try_operators
audit_sections = ss.audit_sections
signed_summary = ss.signed_summary
SECTIONS = ss.SECTIONS
TRY_ABSTAIN = ss.TRY_ABSTAIN

# Proposal-time public Context only.  Outcome / lifecycle words are future.
_PROPOSAL_FORBIDDEN = (
    "support_relation", "delayed_relation",
    "approval", "approved", "approve",
    "local_active", "local_draft", "episode_only", "restricted",
    "local status", "skill status",
)
_DISTINCT_TASK = (
    "distinct task", "distinct tasks", "distinct_task", "distinct-task",
)
_RISK_STILL_PROBE = (
    "restricted probe", "restricted candidate",
    "still a candidate", "still be proposed", "still may be proposed",
    "may still be proposed", "may still serve as",
    "仍可作为", "仍可作",
)

_TEMPORAL_APPENDIX = (
    " Temporal rules for this AD v2 call, in addition to the frozen "
    "containment audit. OBSERVE and WHEN may name only proposal-time "
    "public Context: task_kind and the census observation-feature names. "
    "They must not name support_relation, delayed_relation, approval, "
    "or Skill-status words. RISK is the Source-census default "
    "deprioritization of hampel_filter: lower its proposal priority, "
    "but you must say that under strong public Pattern evidence it may "
    "still be a restricted probe candidate. RISK is not a hard ban. "
    "VERIFY must state the live two-stage gate in words, with no digits: "
    "current Target Support relation POSITIVE forms a Draft; later "
    "delayed relation POSITIVE approves or keeps Active. Do not require "
    "distinct tasks anywhere. TRY must be exactly "
    "NO_AUTHORIZED_ACTIVE_RECOMMENDATION because the authorized try "
    "list is empty."
)


def slow_system(authorized: Sequence[str],
                skill_id: str | None = None) -> str:
    return ss.slow_system(authorized, skill_id=skill_id or SOURCE_SKILL_ID)


def build_skill_payload(sections: Mapping[str, Any],
                        skill_id: str | None = None) -> dict[str, Any]:
    return ss.build_skill_payload(
        sections,
        skill_id=skill_id or SOURCE_SKILL_ID,
        applicability=SOURCE_APPLICABILITY,
    )


def temporal_audit(sections: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanical timing rules.  Not tunable after the Slow call."""
    when = str(sections.get("WHEN") or "")
    observe = str(sections.get("OBSERVE") or "")
    risk = str(sections.get("RISK") or "")
    verify = str(sections.get("VERIFY") or "")
    whole = " ".join(
        str(sections.get(name) or "") for name in SECTIONS).lower()
    proposal = (when + " " + observe).lower()
    risk_l = risk.lower()
    verify_l = verify.lower()

    forbidden_in_proposal = sorted({
        token for token in _PROPOSAL_FORBIDDEN if token in proposal
    })
    distinct_hits = sorted({
        token for token in _DISTINCT_TASK if token in whole
    })
    still_probe = any(token in risk_l for token in _RISK_STILL_PROBE)
    hard_ban = bool(re.search(
        r"\b(never use|must not use|hard ban|forbidden|禁止|不得使用)\b",
        risk_l,
    )) and not still_probe
    verify_support_draft = (
        "support" in verify_l and "positive" in verify_l
        and "draft" in verify_l
    )
    verify_delayed_active = (
        "delayed" in verify_l and "positive" in verify_l
        and ("approv" in verify_l or "active" in verify_l)
    )
    checks = {
        "observe_when_proposal_time_only": not forbidden_in_proposal,
        "risk_names_hampel": "hampel_filter" in risk_l,
        "risk_is_deprioritization": (
            "depriorit" in risk_l or "lower" in risk_l
            or "降低" in risk or "降权" in risk
        ),
        "risk_allows_restricted_probe": still_probe and not hard_ban,
        "verify_support_positive_forms_draft": verify_support_draft,
        "verify_delayed_positive_approves_active": verify_delayed_active,
        "no_distinct_task_requirement": not distinct_hits,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "forbidden_in_observe_when": forbidden_in_proposal,
        "distinct_task_hits": distinct_hits,
        "risk_hard_ban": hard_ban,
    }


def combined_audit(
    sections: Mapping[str, Any],
    census: Sequence[Mapping[str, Any]],
    *,
    operator_names: Sequence[str],
    observable_features: Sequence[str],
    authorized_try: Sequence[str],
) -> dict[str, Any]:
    contain = audit_sections(
        sections, census,
        operator_names=list(operator_names),
        observable_features=list(observable_features),
        source_cohort_tokens=list(SOURCE_COHORT_TOKENS),
        authorized_try=list(authorized_try),
    )
    timing = temporal_audit(sections)
    return {
        "pass": bool(contain["pass"] and timing["pass"]),
        "containment": contain,
        "temporal": timing,
    }


def _slow_call(messages: list[dict[str, str]]) -> Mapping[str, Any]:
    import openai
    from evaluation.functional.task_episode_harness.e1 import (
        NF_BASE_URL, _parse_json_response,
    )

    api_key = next(
        (os.environ.get(name, "").strip()
         for name in ("OPENAI_API_KEY", "AGICTO_API_KEY")
         if os.environ.get(name, "").strip()),
        None,
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    client = openai.OpenAI(
        api_key=api_key, base_url=NF_BASE_URL, timeout=180)
    completion = client.chat.completions.create(
        model="gpt-5.6-sol", messages=messages)
    return _parse_json_response(
        str(completion.choices[0].message.content or ""))


def issue_v2(*, repo_root: Path, census_path: Path,
             out_json: Path, out_md: Path) -> dict[str, Any]:
    """One or two Slow ADD calls for source_investigation_ad_v2."""
    from SelfEvolvingHarnessTS.contracts.harness import (
        EditManifest, EditOperation,
    )
    from SelfEvolvingHarnessTS.contracts.observables import OBSERVABLE_FEATURES
    from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
        EditController, FaultRouter, SurfaceRegistry,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (
        resolve_harness_view,
    )
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (
        _resolve_apply_manifest,
    )
    from SelfEvolvingHarnessTS.operators.registry import OPERATOR_NAMES
    from evaluation.functional.task_episode_harness.e1 import _FastAgentStub

    census_doc = json.loads(census_path.read_text(encoding="utf-8"))
    authorization = census_doc["authorization"]
    authorized = list(authorization["try_authorized"])
    assert authorized == []
    assert authorization["risk_authorized"] == ["hampel_filter"]

    audit_census = []
    for row in census_doc["signed_summary_by_program"]:
        for relation, key in (
            ("POSITIVE", "positive_cohorts"),
            ("NEGATIVE", "negative_cohorts"),
            ("CONFLICT", "conflict_cohorts"),
            ("NEUTRAL", "immaterial_cohorts"),
        ):
            cohorts = list(row[key])
            if not cohorts:
                continue
            audit_census.append({
                "canonical_program": [row["program"]],
                "support_relation": relation,
                "distinct_task_count": len(cohorts),
                "distinct_task_episode_ids": cohorts,
                "level_only_post_shift_support_sufficient": True,
                "post_shift_support_sufficient": True,
                "period_repair_available": True,
            })

    payload = {
        "authorized_try_operators": authorized,
        "risk_authorized_operators": list(authorization["risk_authorized"]),
        "authorization": authorization,
        "signed_summary": census_doc["signed_summary_by_program"],
        "known_limits": census_doc["known_limits"],
        "required_sections": list(SECTIONS),
        "try_abstain_literal": TRY_ABSTAIN,
        "skill_id": SOURCE_SKILL_ID_V2,
        "applicability": SOURCE_APPLICABILITY,
        "v1_status": "superseded",
        "temporal_rules": (
            "OBSERVE/WHEN proposal-time public Context only; "
            "RISK default deprioritize hampel_filter but keep a restricted "
            "probe exception; VERIFY is the live two-stage Support then "
            "delayed POSITIVE gate; no distinct-task requirement"
        ),
        "target_domain": (
            "a different domain from the census; write what to observe "
            "and what would have to hold, not what happened in a named cohort"
        ),
    }
    system = slow_system(authorized, skill_id=SOURCE_SKILL_ID_V2) + _TEMPORAL_APPENDIX
    attempts: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    for attempt in (1, 2):
        try:
            response = _slow_call([
                {"role": "system", "content": system},
                {"role": "user",
                 "content": json.dumps(payload, ensure_ascii=False)},
            ])
        except (RuntimeError, ValueError) as exc:
            attempts.append({
                "attempt": attempt,
                "error": "%s: %s" % (type(exc).__name__, exc),
            })
            continue
        decision = str(response.get("decision") or "").upper()
        sections = response.get("sections")
        row: dict[str, Any] = {
            "attempt": attempt,
            "decision": decision,
            "slow_response": response,
        }
        if decision != "ADD" or not isinstance(sections, Mapping):
            row["audit"] = {"pass": False, "reason": "malformed"}
            attempts.append(row)
            continue
        audit = combined_audit(
            sections, audit_census,
            operator_names=list(OPERATOR_NAMES),
            observable_features=list(OBSERVABLE_FEATURES) + [
                "level_only_post_shift_support_sufficient",
                "post_shift_support_sufficient",
                "period_repair_available",
            ],
            authorized_try=authorized,
        )
        row["audit"] = audit
        attempts.append(row)
        if audit["pass"]:
            accepted = {"sections": dict(sections), "audit": audit,
                        "slow_response": response, "attempt": attempt}
            break
    result: dict[str, Any] = {
        "protocol_version": "t6_nab_42d_source_skill_v2",
        "skill_id": SOURCE_SKILL_ID_V2,
        "v1_skill_id": SOURCE_SKILL_ID,
        "v1_status": "superseded",
        "v1_not_deleted": True,
        "v1_not_in_h0s_v2": True,
        "authorized_try_operators": authorized,
        "risk_authorized_operators": list(authorization["risk_authorized"]),
        "llm_api_call_count": len(attempts),
        "llm_cap": 6,
        "target_outcome_read": False,
        "counts_as_capability_evidence": False,
        "attempts": attempts,
        "slow_payload": payload,
    }
    if accepted is None:
        result.update({
            "verdict": "SLOW_CONSOLIDATION_UNREADABLE",
            "skill_written": False,
            "reason": "both Slow attempts failed the combined audit",
        })
        out_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        out_md.write_text(
            "# AD Skill v2\n\nSLOW_CONSOLIDATION_UNREADABLE\n",
            encoding="utf-8")
        return result

    entry = build_skill_payload(accepted["sections"],
                                skill_id=SOURCE_SKILL_ID_V2)
    store_root = Path(os.environ.get("TEMP") or "/tmp") / "t6_42d_h0s_v2"
    if store_root.exists():
        shutil.rmtree(store_root)
    h0_root = repo_root / "methods" / "ttha" / "harness" / "h0"
    store = SnapshotStore(store_root / "snapshots")
    base = compile_snapshot(h0_root, verify_lock=False)
    store.materialize(base)
    store.set_active(base.runtime_bundle_sha)
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter())
    manifest = EditManifest(
        edit_id=SOURCE_SKILL_ID_V2,
        base_harness_sha=base.harness_content_sha,
        target_pattern_id="t6-42d-source-derived-ad-skill-v2",
        target_surface_id="skill_library.entries/" + SOURCE_SKILL_ID_V2,
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value=entry,
        observable_applicability=dict(SOURCE_APPLICABILITY),
        predicted_agent_behavior_change=(
            "retrieve_skill:" + SOURCE_SKILL_ID_V2,
            "supply_effect_distinct",
        ),
        predicted_data_effect=("safer_proposal_stage",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",),
        patch_id=None,
    )
    parent = store.materialize(base)
    receipt = controller.apply_to_fork(
        parent,
        _resolve_apply_manifest(manifest, base),
        confirmed_cause="SKILL_LIBRARY_GAP",
    )
    snapshot = receipt.candidate_snapshot.snapshot
    store.set_active(snapshot.runtime_bundle_sha)

    a5_method = TTHAMethod(_FastAgentStub(), snapshot, ())
    a3_method = TTHAMethod(_FastAgentStub(), base, ())
    a5_view = resolve_harness_view(
        snapshot, {"task_kind": "anomaly_detection"}, role="fast")
    a3_view = resolve_harness_view(
        base, {"task_kind": "anomaly_detection"}, role="fast")
    a5_ids = [s.skill_id for s in a5_view.skills]
    a3_ids = [s.skill_id for s in a3_view.skills]
    delivery = {
        "a5_retrieves_v2": SOURCE_SKILL_ID_V2 in a5_ids,
        "a3_does_not_retrieve_v2": SOURCE_SKILL_ID_V2 not in a3_ids,
        "a5_memory_empty": list(
            getattr(a5_method, "experience_episodes", ()) or ()) == [],
        "a3_memory_empty": list(
            getattr(a3_method, "experience_episodes", ()) or ()) == [],
        "a5_view_skill_ids": a5_ids,
        "a3_view_skill_ids": a3_ids,
        "note": (
            "proves legal delivery only; not a behavior gain. "
            "v1 SCOPE_CORRECT_NO_APPLICABLE does not endorse v2"
        ),
    }
    delivery["pass"] = all((
        delivery["a5_retrieves_v2"],
        delivery["a3_does_not_retrieve_v2"],
        delivery["a5_memory_empty"],
        delivery["a3_memory_empty"],
    ))
    result.update({
        "verdict": "SKILL_V2_FROZEN_PENDING_BEHAVIOR_REPLAY",
        "skill_written": True,
        "entry": entry,
        "sections": accepted["sections"],
        "audit": accepted["audit"],
        "accepted_attempt": accepted["attempt"],
        "store_root": str(store_root),
        "h0s_v2_runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "h0_runtime_bundle_sha": base.runtime_bundle_sha,
        "skill_ids": [s.skill_id for s in snapshot.skills],
        "delivery_assert": delivery,
    })
    out_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    out_md.write_text(
        "# AD Skill v2\n\n"
        "verdict: SKILL_V2_FROZEN_PENDING_BEHAVIOR_REPLAY\n\n"
        "v1 `source_investigation_ad_v1` is superseded, not deleted, "
        "and is not in h0s_v2.\n\n"
        "h0s_v2 runtime_bundle_sha: `%s`\n\n"
        "## sections\n\n%s\n\n"
        "## delivery assert\n\n"
        "A5 retrieves v2: %s\n"
        "A3 does not: %s\n"
        "both Memory empty: %s / %s\n"
        "This proves delivery only. Behavior replay is a #42e/#42g "
        "precondition, not this book.\n"
        % (
            snapshot.runtime_bundle_sha,
            "\n".join(
                "### %s\n\n%s\n" % (name, accepted["sections"][name])
                for name in SECTIONS),
            delivery["a5_retrieves_v2"],
            delivery["a3_does_not_retrieve_v2"],
            delivery["a5_memory_empty"],
            delivery["a3_memory_empty"],
        ),
        encoding="utf-8")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[4]
    e2 = root / "artifacts" / "functional" / "e2"
    doc = issue_v2(
        repo_root=root,
        census_path=e2 / "t6_nab_42d_source_census.json",
        out_json=e2 / "t6_nab_42d_source_skill_v2.json",
        out_md=e2 / "t6_nab_42d_source_skill_v2.md",
    )
    print(json.dumps({
        "verdict": doc.get("verdict"),
        "llm": doc.get("llm_api_call_count"),
        "written": doc.get("skill_written"),
        "sha": doc.get("h0s_v2_runtime_bundle_sha"),
        "delivery": (doc.get("delivery_assert") or {}).get("pass"),
    }, ensure_ascii=False, indent=1))
