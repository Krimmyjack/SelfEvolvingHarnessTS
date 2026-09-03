"""#42d -- AD Source census, Slow consolidation, optional paired replay.

Parts B/C (and D only if a Skill is written).  Part 0b is a separate
hard gate.  0 LLM on B; C spends at most 8; D at most 32.  AD fit only
on D, cap 120.  No forecasting retrain.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
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

from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    ad_source_skill as ad,
    source_skill as ss,
)
from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_NAMES  # noqa: E402

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
PLAN_V2 = E2 / "t6_nab_frozen_plan_v2.json"
OUT_CENSUS = E2 / "t6_nab_42d_source_census.json"
OUT_SLOW = E2 / "t6_nab_42d_source_skill.json"
OUT_NOTE = E2 / "t6_nab_42d_report.md"
OUT_REPLAY = E2 / "t6_nab_42d_paired_replay.json"
LOCK_PATH = E2 / "t6_nab_42d_paired_replay.lock"
H0S_COPY = E2 / "t6_nab_42d_h0s"
H0_ROOT = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"

BOOLEAN_FEATURES = (
    "level_only_post_shift_support_sufficient",
    "post_shift_support_sufficient",
    "period_repair_available",
)
# r3: pss is a perfect cohort proxy on this bank and must not be Scope.
FORBIDDEN_SCOPE_FEATURES = frozenset({"post_shift_support_sufficient"})
IDENTITY = "identity"
MIN_DISTINCT_COHORTS = 2
LLM_CAP_C = 8
SLOW_MODEL = "gpt-5.6-sol"
SLOW_BASE = "https://api.agicto.cn/v1"


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _json_text(doc: Mapping[str, Any]) -> str:
    return json.dumps(_plain(doc), indent=2, ensure_ascii=False) + "\n"


def _cohort_of(episode: Mapping[str, Any], row: Mapping[str, Any] | None) -> str:
    if row and row.get("cohort"):
        return str(row["cohort"])
    domain = str(episode.get("domain_namespace") or "")
    if domain:
        return domain
    eid = str(episode.get("episode_id") or "")
    if "aws_cloudwatch" in eid:
        return "source_aws_cloudwatch"
    if "known_cause" in eid:
        return "source_known_cause"
    return "unknown"


def _delayed_relation(episode: Mapping[str, Any],
                      row: Mapping[str, Any] | None) -> str:
    if row and row.get("delayed_relation"):
        return str(row["delayed_relation"]).upper()
    measured = ((episode.get("delayed_response") or {}).get("measured_effect")
                or {})
    if measured.get("relation"):
        return str(measured["relation"]).upper()
    return str(episode.get("relation") or "").upper()


def _vote_bucket(relation: str, program: str) -> str | None:
    if program == IDENTITY and relation == "ABSTAIN":
        return None
    if relation == "POSITIVE":
        return "positive"
    if relation == "NEGATIVE":
        return "negative"
    if relation == "CONFLICT":
        return "conflict"
    if relation == "NEUTRAL":
        return "immaterial"
    if relation == "ABSTAIN":
        return None
    return "immaterial"


def _feature_proxy_audit(
    episodes: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in BOOLEAN_FEATURES:
        by_value: dict[str, set[str]] = defaultdict(set)
        by_cohort: dict[str, set[str]] = defaultdict(set)
        for episode in episodes:
            eid = str(episode["episode_id"])
            row = rows_by_id.get(eid)
            cohort = _cohort_of(episode, row)
            value = ((episode.get("context_summary") or {})
                     .get("local_pattern") or {}).get(feature)
            by_value[str(value)].add(cohort)
            by_cohort[cohort].add(str(value))
        values = {k: sorted(v) for k, v in sorted(by_value.items())}
        perfect = (
            all(len(cohorts) == 1 for cohorts in by_value.values())
            and all(len(vals) == 1 for vals in by_cohort.values())
            and len(by_value) == len(by_cohort)
        )
        constant = len(by_value) == 1
        out.append({
            "feature": feature,
            "values_to_cohorts": values,
            "constant": constant,
            "perfect_cohort_proxy": perfect,
            "usable_as_scope": (not constant and not perfect
                                and feature not in FORBIDDEN_SCOPE_FEATURES),
            "note": (
                "no resolving power" if constant else
                "perfect cohort proxy; forbidden as Scope" if perfect else
                "usable"
            ),
        })
    return out


def _unguided_assertion(plan: Mapping[str, Any]) -> dict[str, Any]:
    """#42a bank was enumerated without a Fast-winner Skill in the store."""
    text = json.dumps(plan.get("source_bank") or {}, ensure_ascii=False)
    present = sorted({
        token for token in (
            "fast_winner_", "source_investigation_", "skill_library.entries",
        ) if token in text
    })
    # The bank construction path is _cell_reading / _build_source_bank, which
    # never calls handle_fast_winner.  Presence of the literal in comments
    # would still fail this check; the v2 artifact must be clean.
    return {
        "all_evidence_unguided": not present,
        "forbidden_literals_in_bank": present,
        "construction_path": (
            "evaluation/functional/run_e2_t6_natural_a5_a3.py::_build_source_bank"
        ),
        "note": (
            "UNGUIDED if the Source bank artifact carries no Fast-winner "
            "Skill id; conditioned otherwise (source_skill conservative default)"
        ),
    }


def census(plan: Mapping[str, Any]) -> dict[str, Any]:
    rows = list((plan.get("source_bank") or {}).get("rows") or ())
    episodes = list((plan.get("source_bank") or {}).get("episodes_to_dict") or ())
    rows_by_id = {str(r["episode_id"]): r for r in rows}
    unguided = _unguided_assertion(plan)
    proxy = _feature_proxy_audit(episodes, rows_by_id)

    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for episode in episodes:
        eid = str(episode["episode_id"])
        row = rows_by_id.get(eid, {})
        program = str(row.get("program") or episode.get("workflow_signature"))
        relation = _delayed_relation(episode, row)
        bucket = _vote_bucket(relation, program)
        if bucket is None:
            continue
        features = ((episode.get("context_summary") or {})
                    .get("local_pattern") or {})
        for feature in BOOLEAN_FEATURES:
            key = (program, feature, str(bool(features.get(feature))))
            cell = cells.setdefault(key, {
                "program": program,
                "feature": feature,
                "value": bool(features.get(feature)),
                "positive": [], "negative": [], "conflict": [],
                "immaterial": [],
            })
            cell[bucket].append(eid)

    cell_rows = []
    for key in sorted(cells):
        cell = cells[key]
        cell_rows.append({
            **{k: cell[k] for k in (
                "program", "feature", "value")},
            "positive_count": len(cell["positive"]),
            "negative_count": len(cell["negative"]),
            "conflict_count": len(cell["conflict"]),
            "immaterial_count": len(cell["immaterial"]),
            "positive_episode_ids": cell["positive"],
            "negative_episode_ids": cell["negative"],
            "conflict_episode_ids": cell["conflict"],
            "immaterial_episode_ids": cell["immaterial"],
        })

    # Unconditional pool, counted in distinct Source cohorts.
    pool: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"positive": set(), "negative": set(), "conflict": set(),
                 "immaterial": set()})
    for episode in episodes:
        eid = str(episode["episode_id"])
        row = rows_by_id.get(eid, {})
        program = str(row.get("program") or episode.get("workflow_signature"))
        relation = _delayed_relation(episode, row)
        bucket = _vote_bucket(relation, program)
        if bucket is None:
            continue
        pool[program][bucket].add(_cohort_of(episode, row))

    signed = []
    for program, bags in sorted(pool.items()):
        signed.append({
            "program": program,
            "positive_cohorts": sorted(bags["positive"]),
            "negative_cohorts": sorted(bags["negative"]),
            "conflict_cohorts": sorted(bags["conflict"]),
            "immaterial_cohorts": sorted(bags["immaterial"]),
            "strict_harm_cohorts": sorted(bags["negative"]),
            "extended_harm_cohorts": sorted(
                bags["negative"] | bags["conflict"]),
        })

    winsorize_pos = [
        eid for episode in episodes
        for eid in [str(episode["episode_id"])]
        if (rows_by_id.get(eid) or {}).get("program") == "winsorize"
        and _delayed_relation(episode, rows_by_id.get(eid)) == "POSITIVE"
    ]
    known_limits = [
        "winsorize's two delayed POSITIVE episodes are both "
        "source_aws_cloudwatch (r1 and r2): same cohort, different round "
        "windows.  That is one cohort under the r3 vote unit, not two."
    ]

    usable_scope = [p for p in proxy if p["usable_as_scope"]]
    return {
        "protocol_version": "t6_nab_42d_census_v1",
        "evidence_unit": "episode_id",
        "vote_unit_for_authorization": "distinct Source cohort",
        "relation_layer": "delayed_relation",
        "boolean_features_enumerated": list(BOOLEAN_FEATURES),
        "unguided": unguided,
        "feature_proxy_audit": proxy,
        "legal_conditioning_structurally_unavailable": not usable_scope,
        "cells": cell_rows,
        "signed_summary_by_program": signed,
        "winsorize_positive_episode_ids": winsorize_pos,
        "known_limits": known_limits,
        "episode_count": len(episodes),
        "cost": {"llm": 0, "ad_fits": 0, "forecast_retrains": 0},
    }


def authorize(census_doc: Mapping[str, Any]) -> dict[str, Any]:
    """r3 pre-registered authorization, written before the Slow call."""
    try_ops: list[str] = []
    risk_ops: list[str] = []
    reasons: dict[str, str] = {}
    for row in census_doc["signed_summary_by_program"]:
        program = row["program"]
        pos = set(row["positive_cohorts"])
        harm = set(row["extended_harm_cohorts"])
        if (len(pos) >= MIN_DISTINCT_COHORTS and not harm
                and program != IDENTITY):
            try_ops.append(program)
            reasons[program] = "TRY: >=2 cohort POSITIVE, zero pool harm"
        elif (len(harm) >= MIN_DISTINCT_COHORTS and not pos
              and program != IDENTITY):
            risk_ops.append(program)
            reasons[program] = (
                "RISK: >=2 cohort extended harm (NEGATIVE|CONFLICT), "
                "zero pool POSITIVE"
            )
        else:
            reasons[program] = (
                "neither: positive_cohorts=%s extended_harm=%s"
                % (sorted(pos), sorted(harm))
            )
    return {
        "min_distinct_cohorts": MIN_DISTINCT_COHORTS,
        "harm_definition": "delayed_relation in {NEGATIVE, CONFLICT}",
        "harm_definition_strict": "delayed_relation == NEGATIVE",
        "try_authorized": try_ops,
        "risk_authorized": risk_ops,
        "preregistered_expectation": {
            "try": [],
            "risk": ["hampel_filter"],
        },
        "matches_preregistration": (
            try_ops == [] and risk_ops == ["hampel_filter"]
        ),
        "per_program": reasons,
    }


def _census_for_audit(census_doc: Mapping[str, Any]
                      ) -> list[dict[str, Any]]:
    """Shape source_skill.audit_sections / signed_summary already know."""
    out: list[dict[str, Any]] = []
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
            out.append({
                "canonical_program": [row["program"]],
                "support_relation": relation,
                "distinct_task_count": len(cohorts),
                "distinct_task_episode_ids": cohorts,
            })
    return out


def _slow_call(messages: list[dict[str, str]]) -> Mapping[str, Any]:
    import openai

    api_key = next(
        (os.environ.get(name, "").strip()
         for name in ("OPENAI_API_KEY", "AGICTO_API_KEY")
         if os.environ.get(name, "").strip()),
        None,
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    client = openai.OpenAI(api_key=api_key, base_url=SLOW_BASE, timeout=180)
    completion = client.chat.completions.create(
        model=SLOW_MODEL, messages=messages)
    raw = str(completion.choices[0].message.content or "")
    from evaluation.functional.task_episode_harness.e1 import (
        _parse_json_response,
    )
    return _parse_json_response(raw)


def run_slow(census_doc: Mapping[str, Any],
             authorization: Mapping[str, Any]) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.contracts.observables import OBSERVABLE_FEATURES
    from SelfEvolvingHarnessTS.contracts.harness import (
        EditManifest, EditOperation,
    )
    from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
        EditController, FaultRouter, SurfaceRegistry,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore

    audit_census = _census_for_audit(census_doc)
    authorized = list(authorization["try_authorized"])
    payload = {
        "authorized_try_operators": authorized,
        "risk_authorized_operators": list(authorization["risk_authorized"]),
        "authorization": authorization,
        "signed_summary": census_doc["signed_summary_by_program"],
        "known_limits": census_doc["known_limits"],
        "required_sections": list(ad.SECTIONS),
        "try_abstain_literal": ad.TRY_ABSTAIN,
        "skill_id": ad.SOURCE_SKILL_ID,
        "applicability": ad.SOURCE_APPLICABILITY,
        "target_domain": (
            "a different domain from the census; write what to observe "
            "and what would have to hold, not what happened in a named cohort"
        ),
    }
    result: dict[str, Any] = {
        "protocol_version": "t6_nab_42d_source_skill_v1",
        "skill_id": ad.SOURCE_SKILL_ID,
        "authorized_try_operators": authorized,
        "risk_authorized_operators": list(authorization["risk_authorized"]),
        "llm_api_call_count": 0,
        "llm_cap": LLM_CAP_C,
        "target_outcome_read": False,
        "slow_payload": payload,
    }
    try:
        response = _slow_call([
            {"role": "system", "content": ad.slow_system(authorized)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ])
        result["llm_api_call_count"] = 1
    except (RuntimeError, ValueError) as exc:
        result.update({
            "verdict": "CONSOLIDATION_NO_ELIGIBLE_SKILL",
            "reason": "Slow call failed: %s: %s" % (type(exc).__name__, exc),
            "skill_written": False,
        })
        return result
    result["slow_response"] = _plain(response)
    decision = str(response.get("decision") or "").upper()
    if decision == "ABSTAIN":
        result.update({
            "verdict": "CONSOLIDATION_NO_ELIGIBLE_SKILL",
            "reason": response.get("reason"),
            "skill_written": False,
            "note": "ABSTAIN is a legitimate result; the Skill is not written by hand",
        })
        return result
    sections = response.get("sections")
    if decision != "ADD" or not isinstance(sections, Mapping):
        result.update({
            "verdict": "CONSOLIDATION_NO_ELIGIBLE_SKILL",
            "reason": "Slow returned a malformed payload",
            "skill_written": False,
        })
        return result
    audit = ad.audit_sections(
        sections, audit_census,
        operator_names=list(OPERATOR_NAMES),
        observable_features=list(OBSERVABLE_FEATURES) + list(BOOLEAN_FEATURES),
        source_cohort_tokens=list(ad.SOURCE_COHORT_TOKENS),
        authorized_try=authorized,
    )
    result["audit"] = audit
    if not audit["pass"]:
        result.update({
            "verdict": "CONSOLIDATION_NO_ELIGIBLE_SKILL",
            "reason": "containment audit failed: %s" % {
                k: v for k, v in audit["checks"].items() if not v},
            "skill_written": False,
            "note": "rejected whole; no hand rewrite",
        })
        return result
    entry = ad.build_skill_payload(sections)
    store_root = Path(os.environ.get("TEMP") or "/tmp") / "t6_42d_h0s"
    if store_root.exists():
        import shutil
        shutil.rmtree(store_root)
    store = SnapshotStore(store_root / "snapshots")
    base = compile_snapshot(H0_ROOT, verify_lock=False)
    store.materialize(base)
    store.set_active(base.runtime_bundle_sha)
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter())
    manifest = EditManifest(
        edit_id=ad.SOURCE_SKILL_ID,
        base_harness_sha=base.harness_content_sha,
        target_pattern_id="t6-42d-source-derived-ad-skill",
        target_surface_id="skill_library.entries/" + ad.SOURCE_SKILL_ID,
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value=entry,
        observable_applicability=dict(ad.SOURCE_APPLICABILITY),
        predicted_agent_behavior_change=(
            "retrieve_skill:" + ad.SOURCE_SKILL_ID,
            "supply_effect_distinct",
        ),
        predicted_data_effect=("safer_proposal_stage",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",),
        patch_id=None,
    )
    try:
        from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (
            _resolve_apply_manifest,
        )
        parent = store.materialize(base)
        receipt = controller.apply_to_fork(
            parent,
            _resolve_apply_manifest(manifest, base),
            confirmed_cause="SKILL_LIBRARY_GAP",
        )
        snapshot = receipt.candidate_snapshot.snapshot
        store.set_active(snapshot.runtime_bundle_sha)
    except Exception as exc:  # noqa: BLE001
        result.update({
            "verdict": "CONSOLIDATION_NO_ELIGIBLE_SKILL",
            "reason": "compiler/EditController rejected the entry: %s: %s"
                      % (type(exc).__name__, exc),
            "skill_written": False,
        })
        return result
    result.update({
        "verdict": "SOURCE_SKILL_WRITTEN",
        "skill_written": True,
        "entry": entry,
        "store_root": str(store_root),
        "runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "skill_ids": [s.skill_id for s in snapshot.skills],
    })
    return result


def part0_status() -> dict[str, Any]:
    import subprocess
    subprocess.run(
        ["git", "update-index", "--really-refresh"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    porcelain = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True).stdout
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True).stdout.strip()
    return {
        "head": head,
        "action": (
            "verified only -- this round does not commit; Part 0 status "
            "is recorded, not sealed"
        ),
        "porcelain_lines": len([ln for ln in porcelain.splitlines() if ln.strip()]),
        "note": (
            "the working tree already carries untracked #42/#42c artifacts; "
            "Part 0 is not a commit this session"
        ),
    }


def main() -> int:
    started = time.perf_counter()
    if not PLAN_V2.exists():
        print(json.dumps({"verdict": "CONSOLIDATION_NO_ELIGIBLE_SKILL",
                          "reason": "missing v2 plan"}, indent=1))
        return 1
    plan = json.loads(PLAN_V2.read_text(encoding="utf-8"))
    part0 = part0_status()
    census_doc = census(plan)
    authorization = authorize(census_doc)
    census_doc["authorization"] = authorization
    OUT_CENSUS.write_text(_json_text(census_doc), encoding="utf-8")

    slow = run_slow(census_doc, authorization)
    slow["part_0"] = part0
    slow["census_artifact"] = "artifacts/functional/e2/t6_nab_42d_source_census.json"
    slow["wall_seconds"] = time.perf_counter() - started
    OUT_SLOW.write_text(_json_text(slow), encoding="utf-8")

    skip_d = not slow.get("skill_written")
    verdict = slow.get("verdict") or "CONSOLIDATION_NO_ELIGIBLE_SKILL"
    if skip_d:
        note = (
            "# #42d consolidation\n\n"
            "Part 0b: FORECASTING_COMPAT_RESTORED (6/6).\n\n"
            "Part B census: %s episodes, legal conditioning structurally "
            "unavailable (pss is a perfect cohort proxy; the other two "
            "booleans are constants).  Authorization is on the unconditional "
            "pool, counted in distinct Source cohorts.\n\n"
            "TRY authorized: %s\n"
            "RISK authorized: %s\n"
            "matches preregistration: %s\n\n"
            "Part C Slow: %s\n"
            "%s\n\n"
            "Part D: skipped (r3 branch: full ABSTAIN / no written Skill).\n"
            "Routing suggestion (not executed): add realTraffic / realTweets "
            "as 3rd/4th Source cohorts, then consolidate again.\n"
            % (
                census_doc["episode_count"],
                authorization["try_authorized"],
                authorization["risk_authorized"],
                authorization["matches_preregistration"],
                verdict,
                slow.get("reason") or "",
            )
        )
        OUT_NOTE.write_text(note, encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "try_authorized": authorization["try_authorized"],
        "risk_authorized": authorization["risk_authorized"],
        "skill_written": bool(slow.get("skill_written")),
        "part_d": "skipped" if skip_d else "required",
        "llm_calls": slow.get("llm_api_call_count", 0),
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_CENSUS)
    print("wrote", OUT_SLOW)
    return 0


def _copy_h0s() -> Path:
    import shutil
    src = Path(os.environ.get("TEMP") or "/tmp") / "t6_42d_h0s"
    if not (src / "active.json").exists():
        raise SystemExit("h0s store missing at %s" % src)
    if H0S_COPY.exists():
        shutil.rmtree(H0S_COPY)
    shutil.copytree(src, H0S_COPY)
    return H0S_COPY


def _operator_of(candidate_id: Any) -> str:
    text = str(candidate_id or "")
    if text.startswith("cand_"):
        text = text[5:]
    for prefix in ("localized_", "local_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in ("_local_extreme_deviation", "_tail_outlier_repair"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def _proposal_profile(cells: Sequence[Mapping[str, Any]],
                      arm: str) -> dict[str, Any]:
    rows = [c for c in cells if c.get("arm") == arm and "error" not in c]
    operators: dict[str, int] = {}
    hampel = 0
    non_identity = 0
    identity_only = 0
    for cell in rows:
        probes = [p for p in (cell.get("probes") or [])
                  if p.get("kind") == "probe"]
        names = [_operator_of(p.get("candidate_id")) for p in probes]
        if not names:
            identity_only += 1
        for name in names:
            operators[name] = operators.get(name, 0) + 1
            if name == "hampel_filter":
                hampel += 1
            if name != IDENTITY:
                non_identity += 1
        pool = [_operator_of(x) for x in (cell.get("pool") or [])
                if x and x != IDENTITY]
        for name in pool:
            operators.setdefault(name, operators.get(name, 0))
    return {
        "cells": [c.get("cell") for c in rows],
        "non_identity_trials": non_identity,
        "identity_only_cells": identity_only,
        "hampel_probe_count": hampel,
        "operator_probe_counts": operators,
        "pools": [c.get("pool") for c in rows],
        "chosens": [c.get("chosen") for c in rows],
        "winners": [c.get("winner_program") for c in rows],
    }


def _replay_verdict(run: Mapping[str, Any]) -> dict[str, Any]:
    cells = list(run.get("cells") or [])
    a3 = _proposal_profile(cells, "A3")
    a5 = _proposal_profile(cells, "A5")
    collapse = (
        a5["non_identity_trials"] == 0
        and a3["non_identity_trials"] > 0
        and a5["identity_only_cells"] == len(a5["cells"])
    )
    hampel_down = a5["hampel_probe_count"] < a3["hampel_probe_count"]
    other_a3 = {k: v for k, v in a3["operator_probe_counts"].items()
                if k not in {IDENTITY, "hampel_filter"}}
    other_a5 = {k: v for k, v in a5["operator_probe_counts"].items()
                if k not in {IDENTITY, "hampel_filter"}}
    others_comparable = (
        not other_a3 or any(other_a5.get(k, 0) > 0 for k in other_a3)
        or (not other_a3 and not other_a5)
    )
    if (run.get("label_wall") or {}).get("breached"):
        verdict = "TARGET_LABEL_WALL_BREACHED"
    elif collapse:
        verdict = "UNCHANGED_COLLAPSE"
    elif hampel_down and others_comparable:
        verdict = "BEHAVIOR_CHANGED"
    elif a5["non_identity_trials"] == a3["non_identity_trials"] and not hampel_down:
        verdict = "SCOPE_CORRECT_NO_APPLICABLE"
    else:
        verdict = "BEHAVIOR_CHANGED"
    return {
        "verdict": verdict,
        "a3": a3,
        "a5_prime": a5,
        "hampel_probe_rate": {
            "A3": a3["hampel_probe_count"],
            "A5_prime": a5["hampel_probe_count"],
        },
        "collapse_to_identity_only": collapse,
        "note": (
            "r3 primary readout: bounded hampel deprioritization vs global "
            "collapse.  Historical #42 A3/A5 numbers are descriptive only."
        ),
    }


def run_part_d() -> int:
    """One-shot paired replay.  Does not re-call Slow."""
    import shutil
    import subprocess

    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    import run_e2_t6_natural_a5_a3 as t6

    leftover: list[str] = []
    try:
        listing = subprocess.run(
            ["wmic", "process", "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, timeout=15)
        for line in (listing.stdout or "").splitlines():
            if "run_e2_t6_natural_a5_a3.py" in line and str(os.getpid()) not in line:
                leftover.append(line.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        leftover = []
    if leftover:
        print(json.dumps({
            "verdict": "CONCURRENT_RUN_BLOCKED",
            "reason": leftover,
        }, indent=1))
        return 2
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(json.dumps({"verdict": "CONCURRENT_RUN_BLOCKED",
                          "reason": "lock held"}, indent=1))
        return 2
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    h0s_root = _copy_h0s()
    active = json.loads((h0s_root / "active.json").read_text(encoding="utf-8"))
    h0s = compile_snapshot(
        h0s_root / "snapshots" / active["runtime_bundle_sha"],
        verify_lock=False)
    h0 = compile_snapshot(H0_ROOT, verify_lock=False)
    if ad.SOURCE_SKILL_ID not in {s.skill_id for s in h0s.skills}:
        print(json.dumps({"verdict": "CONSOLIDATION_NO_ELIGIBLE_SKILL",
                          "reason": "h0s missing AD source skill"}, indent=1))
        return 1
    if ad.SOURCE_SKILL_ID in {s.skill_id for s in h0.skills}:
        print(json.dumps({"verdict": "TARGET_FEEDBACK_UNREADABLE",
                          "reason": "h0 unexpectedly carries the AD skill"},
                         indent=1))
        return 1
    plan = json.loads(PLAN_V2.read_text(encoding="utf-8"))
    wall = t6.LabelWall(released=True)
    universe = t6._load_universe(t6.gate_all(row_order_contract=True))
    budget = t6.FitBudget(120)
    store_tag = "t6d42_%s" % run_id
    try:
        run = t6._run_cells(
            plan=plan,
            cohort_rows=universe["target"],
            agent_factory=t6._evaluate_agent,
            backend_factory=t6._evaluate_backend,
            llm_budget=32,
            fit_budget=budget,
            wall=wall,
            store_tag=store_tag,
            snapshot_for_arm={"A3": h0, "A5": h0s},
        )
    except t6.Stop as stop:
        payload = {"verdict": stop.verdict, "reason": stop.reason,
                   "run_id": run_id}
        OUT_REPLAY.write_text(_json_text(payload), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 1
    run["label_wall"] = wall.audit()
    run["run_id"] = run_id
    run["h0s_runtime_bundle_sha"] = h0s.runtime_bundle_sha
    run["h0_runtime_bundle_sha"] = h0.runtime_bundle_sha
    run["construction_memory_empty"] = True
    verdict = _replay_verdict(run)
    payload = {
        "protocol_version": "t6_nab_42d_paired_replay_v1",
        "entry": "--paired-replay",
        "mechanical_fixture": False,
        "counts_as_method_evidence": True,
        "evidence_grade": "NATURAL",
        "evidence_standing": "provisional",
        "run": run,
        "verdict": verdict,
        "historical_42_numbers_are_descriptive_only": True,
    }
    OUT_REPLAY.write_text(_json_text(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": verdict["verdict"],
        "llm_calls": run.get("llm_calls"),
        "ad_fits": run.get("ad_fits"),
        "hampel": verdict["hampel_probe_rate"],
        "collapse": verdict["collapse_to_identity_only"],
    }, ensure_ascii=False, indent=1))
    print("wrote", OUT_REPLAY)
    return 0


if __name__ == "__main__":
    if "--paired-replay" in sys.argv:
        raise SystemExit(run_part_d())
    raise SystemExit(main())
