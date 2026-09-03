"""S3 pilot: three-arm probe-policy course (freeze 2026-08-29).

Protocol: no_edit / random_edit / llm_edit on a frozen 5-unit dual-family
course. Edits land only on ExplorationPolicy; G3 surfaces stay untouched.
LLM-edit proposal calls are off the course cap and happen only under --run.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
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

import run_e2_ps0c_ps1 as ps0c  # noqa: E402
import run_e2_s2a_forecast_curriculum as s2a  # noqa: E402
import run_e2_s2a_forecast_oracle as traffic  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.exploration_policy import (  # noqa: E402
    DEFAULT,
    LEGAL_DOMAINS,
    ExplorationPolicy,
    install_policy,
    reset_policy,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
R2_JSON = E2 / "s2a_g1_run1_r2.json"
OUT_JSON = E2 / "s3_pilot_probe_policy.json"
OUT_MD = E2 / "s3_pilot_probe_policy.md"
CHECKPOINT = E2 / "s3_pilot_probe_policy.checkpoint.json"

PROTOCOL = "s3_pilot_probe_policy_v1"
ARMS = ("no_edit", "random_edit", "llm_edit")
RANDOM_SEED = 20260829
LLM_CAP_PER_ARM = 40
FIT_CAP_PER_ARM = 120
LLM_CAP_TOTAL = 120
WALL_SECONDS = int(5 * 60 * 60)
TOL = 1e-9
BENEFICIARY_POSITIONS = (2, 4, 5)
SKILL_ELECTRICITY = "s3_supply_electricity_v0"
SKILL_TRAFFIC = "s3_supply_traffic_v0"
SKILL_BY_FAMILY = {
    "electricity": SKILL_ELECTRICITY,
    "traffic": SKILL_TRAFFIC,
}
PATTERN_AXIS_PROVENANCE = (
    "n=1 source intersection of the producer Pattern view; "
    "forecast extractor + observable_numeric_bin + "
    "s1a.PATTERN_KEYS. No S1a forecast cluster exists; "
    "inventing one is forbidden.")
G3_SURFACES = (
    "dual-gate classifier",
    "capacity gate",
    "harm threshold",
    "authority / execution guards",
    "isolation / G2 guards",
    "ladder v2 supply-tier thresholds",
    "Scope match semantics",
)
COURSE = (
    {"position": 1, "unit_id": "electricity_impulsive_outlier_00",
     "role": "producer", "family": "electricity", "source": "injected"},
    {"position": 2, "unit_id": "electricity_impulsive_outlier_02",
     "role": "beneficiary", "family": "electricity", "source": "injected"},
    {"position": 3, "unit_id": "traffic_impulsive_outlier_00",
     "role": "producer", "family": "traffic", "source": "injected"},
    {"position": 4, "unit_id": "traffic_impulsive_outlier_01",
     "role": "beneficiary", "family": "traffic", "source": "injected"},
    {"position": 5, "unit_id": "traffic_impulsive_outlier_02",
     "role": "beneficiary", "family": "traffic", "source": "injected"},
)


class Stop(Exception):
    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__("%s: %s" % (verdict, reason))
        self.verdict = verdict
        self.reason = reason


INSTRUMENT_STOPS = (
    "BACKEND_UNAVAILABLE",
    "COMPUTE_BUDGET_EXCEEDED",
    "INSTRUMENT_UNREADABLE",
    "S3_G2_LEAK",
    "ENVIRONMENT_MISMATCH",
    "IMPORT_PATH_SPLIT",
)


class CellBank(s2a.CellBank):
    def get(self, unit_id: str, freeze: Mapping[str, Any] | None = None
            ) -> dict[str, Any]:
        if unit_id in self._cells:
            return self._cells[unit_id]
        if unit_id.startswith("traffic_impulsive_outlier_"):
            names, pool = self._traf()
            spec = next(c for c in traffic._recut(names)
                        if c["unit_id"] == unit_id)
            values = traffic._inject(spec, pool)
            first = np.asarray(values[str(spec["train"][0])], dtype=np.float64)
            cell = {
                **spec,
                "values": values,
                "observation_block": first[:traffic.ORIGIN_HELDIN].copy(),
                "condition": traffic.CONDITION,
                "source": spec.get("source") or spec.get("family"),
            }
            self._cells[unit_id] = cell
            return cell
        return super().get(unit_id, freeze if freeze is not None else {})


def _compile_card(row: Mapping[str, Any], skill_id: str) -> dict[str, Any]:
    return s2a.ss.compile_supply_tier(
        [row], skill_id=skill_id,
        legal_features=s2a.ss._edit_schema_features(s2a.PROJECT_ROOT),
        pattern_family=None,
        pattern_axis_provenance=PATTERN_AXIS_PROVENANCE)


def _revise(snapshot: Any, scored: Mapping[str, Any], unit: Mapping[str, Any],
            store_root: Path, card_sha: str, skill_id: str) -> dict[str, Any]:
    entry = next((s for s in snapshot.skills
                  if str(s.skill_id) == skill_id), None)
    if entry is None:
        return {"applied": [], "snapshot": snapshot, "card_sha": card_sha}
    rounds = scored.get("rounds") or []
    scope_ok = any(bool((r.get("scope_match_by_skill_id") or {})
                        .get(skill_id)) for r in rounds)
    supplied = [e for r in rounds for e in (r.get("episodes") or [])
                if str(e.get("source_skill_id") or "") == skill_id]
    converted = [e for e in supplied
                 if str(e.get("relation")) == "POSITIVE"
                 and str(e.get("local_status")) == "LOCAL_ACTIVE"]
    refused = [e for e in supplied if e not in converted]
    guards = dict(entry.risk_guards or {})
    ast = json.loads(json.dumps(s2a.s1._plain(entry.observable_applicability)))
    applied: list[str] = []
    narrowed = None
    for episode in converted:
        guards = s2a.sr.append_evidence_row(guards, {
            "rule": "R1_positive",
            "unit_id": str(unit["unit_id"]),
            "support_gain": float(episode.get("support_gain") or 0.0),
            "delayed_gain": float(episode.get("delayed_gain") or 0.0),
            "heldout_gain": float((scored.get("deployment") or {})
                                  .get("heldout_gain") or 0.0),
            "card_content_sha_when_earned": card_sha,
            "counts_toward_authorization": False,
        })
        applied.append("R1")
    harmed = bool((scored.get("deployment") or {}).get("harm_event"))
    if (refused or harmed) and scope_ok:
        source_views = [dict(row.get("pattern_view") or {}) for row in
                        (guards.get("evidence") or {}).get("sources") or []]
        refusing_view = dict((rounds[0] if rounds else {}).get("pattern_view")
                             or {})
        exclusion = s2a.sr.compile_exclusion(
            refusing_view=refusing_view, source_views=source_views,
            axes=s2a.sr.contracted_axes(
                s2a.ss._edit_schema_features(s2a.PROJECT_ROOT)))
        rule = "R3_negative_or_harm" if (
            harmed or any(str(e.get("relation")) == "NEGATIVE"
                          for e in refused)) else "R2_conflict"
        if exclusion["leaves"]:
            narrowed = s2a.sr.narrow_applicability(ast, exclusion["leaves"])
            applied.append("R2" if rule == "R2_conflict" else "R3")
        if rule == "R3_negative_or_harm":
            guards = s2a.sr.append_demotion(guards, {
                "rule": rule, "trigger_unit": str(unit["unit_id"]),
                "harm_event": harmed})
            if "R3" not in applied:
                applied.append("R3")
    if not applied and narrowed is None:
        return {"applied": [], "snapshot": snapshot, "card_sha": card_sha}
    patched = s2a.sr.patch_card(
        snapshot, skill_id=skill_id, store_root=store_root,
        tag="revise_%s" % unit["unit_id"],
        risk_guards=guards, observable_applicability=narrowed,
        predicted_data_effect=tuple(applied) or ("skill_revised",))
    return {"applied": applied, "snapshot": patched["snapshot"],
            "card_sha": patched["card_sha"],
            "receipts": patched.get("receipts")}


def _draw_random_edit() -> dict[str, Any]:
    rng = random.Random(RANDOM_SEED)
    param = rng.choice(sorted(LEGAL_DOMAINS))
    default = DEFAULT.to_dict()[param]
    non_default = [value for value in LEGAL_DOMAINS[param] if value != default]
    if not non_default:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "LEGAL_DOMAINS[%s] has no non-default value" % param)
    value = rng.choice(non_default)
    edit = {param: value}
    policy = {**DEFAULT.to_dict(), **edit}
    ExplorationPolicy(**policy).validate()
    return {
        "seed": RANDOM_SEED,
        "edit": edit,
        "param": param,
        "value": value,
        "policy": policy,
    }


def _legal_domains_jsonable() -> dict[str, list[Any]]:
    return {key: list(domain) for key, domain in LEGAL_DOMAINS.items()}


def _in_domain(value: Any, domain: Sequence[Any]) -> bool:
    for item in domain:
        if type(item) is bool:
            if type(value) is bool and value is item:
                return True
        elif type(item) is float:
            if type(value) is float or type(value) is int:
                if type(value) is not bool and value == item:
                    return True
        elif type(item) is int:
            if type(value) is int and type(value) is not bool and value == item:
                return True
        elif value == item and type(value) is type(item):
            return True
    return False


def _extract_seed_trace() -> dict[str, Any]:
    artifact = json.loads(R2_JSON.read_text(encoding="utf-8"))
    rows = list(artifact.get("rows") or [])
    a5 = next(row for row in rows
              if int(row.get("position") or 0) == 4
              and str(row.get("arm")) == "A5-online")
    counterparts = {}
    for label, arm in (("A3-reset", "A3-reset"), ("K0-fixed", "K0-fixed")):
        peer = next(row for row in rows
                    if int(row.get("position") or 0) == 4
                    and str(row.get("arm")) == arm)
        counterparts[label] = {
            "heldout_gain": peer.get("heldout_gain"),
            "applied_ops": peer.get("applied_ops"),
        }
    round0 = (a5.get("rounds") or [{}])[0]
    deployment = dict(a5.get("deployment") or {})
    predictions = {str(row.get("id")): row.get("observed")
                   for row in (artifact.get("predictions") or [])
                   if str(row.get("id")) in ("P3", "P6")}
    chain = list(artifact.get("version_chain") or [])
    return {
        "protocol": artifact.get("protocol"),
        "seed": artifact.get("seed"),
        "course_3": (artifact.get("course") or [None] * 4)[3],
        "version_chain_0": chain[0] if chain else None,
        "a5_online_position_4": {
            "candidate_sources": a5.get("candidate_sources"),
            "scope_match": a5.get("scope_match"),
            "rounds_0": {
                "pool": round0.get("pool"),
                "chosen": round0.get("chosen"),
                "retrieved_skill_ids": round0.get("retrieved_skill_ids"),
                "scope_match_by_skill_id": round0.get(
                    "scope_match_by_skill_id"),
                "proposals": round0.get("proposals"),
                "probes": round0.get("probes"),
                "winner_program": round0.get("winner_program"),
                "episodes": round0.get("episodes"),
            },
            "deployment": {
                "deploy_source": deployment.get("deploy_source"),
                "active_skill_id": deployment.get("active_skill_id"),
                "applied_program": deployment.get("applied_program"),
                "heldout_gain": deployment.get("heldout_gain"),
                "harm_event": deployment.get("harm_event"),
            },
        },
        "position_4_counterfactual": counterparts,
        "predictions_observed": predictions,
    }


def _build_llm_prompt(seed: Mapping[str, Any],
                      error: str | None = None) -> str:
    a5 = dict(seed.get("a5_online_position_4") or {})
    deploy = dict(a5.get("deployment") or {})
    peers = dict(seed.get("position_4_counterfactual") or {})
    a5_gain = deploy.get("heldout_gain")
    a3_gain = (peers.get("A3-reset") or {}).get("heldout_gain")
    try:
        counterfactual = float(a5_gain) - float(a3_gain)
    except (TypeError, ValueError):
        counterfactual = None
    body = {
        "failure_type": "SUPPLY_STARVATION",
        "failure_narrative": (
            "Supply card reached the candidate pool but was never probed. "
            "A better hampel candidate was never evaluated. "
            "A5 heldout_gain counterfactual versus A3/K0 is -0.1206. "
            "Mechanism: chosen-first probe order plus first-positive stop "
            "exhausted the support budget before the supplied candidate "
            "was assessed."),
        "a5_heldout_gain": a5_gain,
        "a3_heldout_gain": a3_gain,
        "k0_heldout_gain": (peers.get("K0-fixed") or {}).get("heldout_gain"),
        "counterfactual_a5_minus_a3": counterfactual,
        "seed_trace": seed,
        "current_default_policy": DEFAULT.to_dict(),
        "LEGAL_DOMAINS": _legal_domains_jsonable(),
        "hard_constraints": {
            "editable_parameters_only": sorted(LEGAL_DOMAINS),
            "values_must_be_in_LEGAL_DOMAINS": True,
            "must_not_add_or_remove_probe_total_budget": True,
            "g3_surfaces_must_not_be_touched": list(G3_SURFACES),
        },
        "output_contract": (
            "Return exactly one JSON object. Keys are a non-empty subset "
            "of the 8 LEGAL_DOMAINS parameters. Values must be legal "
            "domain members. No markdown, no commentary."),
    }
    prompt = (
        "Propose one legal edit to the Harness exploration/allocation "
        "policy after reading the SUPPLY_STARVATION seed trace.\n\n"
        + json.dumps(body, indent=2, ensure_ascii=False, default=str)
    )
    if error:
        prompt += (
            "\n\nYour previous reply was rejected: %s\n"
            "Try again. Reply with exactly one JSON object."
            % error)
    return prompt


def _first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth = 0
    in_str = False
    escape = False
    for index, char in enumerate(text[start:], start):
        if in_str:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start:index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("JSON root is not an object")
                return parsed
    raise ValueError("unbalanced JSON object in reply")


def _validate_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(proposal, Mapping) or not proposal:
        raise ValueError("proposal must be a non-empty object")
    unknown = sorted(set(proposal) - set(LEGAL_DOMAINS))
    if unknown:
        raise ValueError("unknown keys: %s" % unknown)
    for key, value in proposal.items():
        if not _in_domain(value, LEGAL_DOMAINS[key]):
            raise ValueError(
                "illegal value %s=%r (legal: %r)"
                % (key, value, LEGAL_DOMAINS[key]))
    policy = ExplorationPolicy(**{**DEFAULT.to_dict(), **dict(proposal)})
    policy.validate()
    return policy.to_dict()


def _propose_llm_edit(seed: Mapping[str, Any]) -> dict[str, Any]:
    import openai

    cfg = ps0c._relay_cfg()
    client = openai.OpenAI(
        api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=120)
    raw_replies: list[str] = []
    errors: list[str] = []
    assignment = None
    prompt = _build_llm_prompt(seed)
    for attempt in range(1, 3):
        completion = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}])
        text = str(completion.choices[0].message.content or "")
        raw_replies.append(text)
        try:
            proposal = _first_json_object(text)
            assignment = _validate_proposal(proposal)
            return {
                "attempts": attempt,
                "raw_replies": [ps0c.redact(item) for item in raw_replies],
                "proposal": proposal,
                "assignment": assignment,
                "illegal": False,
                "errors": errors,
            }
        except Exception as exc:  # noqa: BLE001
            err = "%s: %s" % (type(exc).__name__, exc)
            errors.append(err)
            prompt = _build_llm_prompt(seed, error=err)
    return {
        "attempts": 2,
        "raw_replies": [ps0c.redact(item) for item in raw_replies],
        "proposal": None,
        "assignment": None,
        "illegal": True,
        "errors": errors,
    }


def _rewrite_scope_match(scored: Mapping[str, Any]) -> dict[str, bool]:
    rounds = scored.get("rounds") or []
    if not rounds:
        return {"classification": False, "supply_electricity": False,
                "supply_traffic": False}
    scope = dict(rounds[0].get("scope_match_by_skill_id") or {})
    return {
        "classification": bool(scope.get(s2a.CLS_SKILL_ID)),
        "supply_electricity": bool(scope.get(SKILL_ELECTRICITY)),
        "supply_traffic": bool(scope.get(SKILL_TRAFFIC)),
    }


def _minted_skill(family: str, family_sha: Mapping[str, str]) -> str | None:
    return SKILL_BY_FAMILY[family] if family in family_sha else None


def _apply_producer_card(snapshot: Any, scored: Mapping[str, Any],
                         unit: Mapping[str, Any], *, store_root: Path,
                         chain: list[dict[str, Any]], arm: str
                         ) -> tuple[Any, str | None]:
    family = str(unit["family"])
    skill_id = SKILL_BY_FAMILY[family]
    row = s2a._supply_row(scored)
    if not row:
        print("  CARD withheld: no_positive_local_active", flush=True)
        return snapshot, None
    compiled = _compile_card(row, skill_id)
    if compiled.get("card") is None:
        print("  CARD withheld: %s" % compiled.get("withheld_because"),
              flush=True)
        return snapshot, None
    snapshot = s2a._install(
        snapshot, compiled["card"],
        store_root=store_root / "seed",
        tag="%s_%s_v0" % (arm, family))
    sha = s2a.ss.skill_content_sha(
        next(s for s in snapshot.skills if str(s.skill_id) == skill_id))
    chain.append({
        "version": "v%d" % len(chain),
        "skill_id": skill_id,
        "family": family,
        "pilot_arm": arm,
        "card_content_sha": sha,
        "installed_after": unit["unit_id"],
        "produced_by": "ladder_v2_compile_supply_tier",
        "runtime_bundle_sha": snapshot.runtime_bundle_sha,
    })
    print("  CARD %s %s" % (skill_id, sha[:12]), flush=True)
    return snapshot, sha


def _apply_beneficiary_revision(snapshot: Any, scored: Mapping[str, Any],
                                unit: Mapping[str, Any], *, store_root: Path,
                                chain: list[dict[str, Any]], arm: str,
                                card_sha: str) -> tuple[Any, str]:
    family = str(unit["family"])
    skill_id = SKILL_BY_FAMILY[family]
    revision = _revise(
        snapshot, scored, unit, store_root / "revise", card_sha, skill_id)
    snapshot = revision["snapshot"]
    if revision["applied"]:
        chain.append({
            "version": "v%d" % len(chain),
            "skill_id": skill_id,
            "family": family,
            "pilot_arm": arm,
            "card_content_sha": revision["card_sha"],
            "produced_by": "+".join(revision["applied"]),
            "trigger_unit": unit["unit_id"],
            "runtime_bundle_sha": snapshot.runtime_bundle_sha,
        })
        print("    REVISION %s" % "+".join(revision["applied"]), flush=True)
    return snapshot, revision["card_sha"]


def _replay_arm_state(arm: str, rows: Sequence[Mapping[str, Any]],
                      store_root: Path, chain: list[dict[str, Any]]
                      ) -> tuple[Any, dict[str, str]]:
    snapshot = s2a._install(
        s2a._h0(), s2a._cls_card(),
        store_root=store_root / "seed", tag="%s_v0" % arm)
    family_sha: dict[str, str] = {}
    arm_rows = sorted(
        (row for row in rows if str(row.get("pilot_arm")) == arm),
        key=lambda row: int(row["position"]))
    for scored in arm_rows:
        unit = COURSE[int(scored["position"]) - 1]
        family = str(unit["family"])
        if unit["role"] == "producer" and family not in family_sha:
            snapshot, sha = _apply_producer_card(
                snapshot, scored, unit, store_root=store_root,
                chain=chain, arm=arm)
            if sha:
                family_sha[family] = sha
        elif unit["role"] == "beneficiary" and family in family_sha:
            snapshot, sha = _apply_beneficiary_revision(
                snapshot, scored, unit, store_root=store_root,
                chain=chain, arm=arm, card_sha=family_sha[family])
            family_sha[family] = sha
    return snapshot, family_sha


def _empty_ledger() -> dict[str, Any]:
    return {
        "llm": 0,
        "fit": 0,
        "per_arm": {arm: {"llm": 0, "fit": 0} for arm in ARMS},
        "llm_cap_per_arm": LLM_CAP_PER_ARM,
        "fit_cap_per_arm": FIT_CAP_PER_ARM,
        "llm_cap_total": LLM_CAP_TOTAL,
    }


def _support_receipts(row: Mapping[str, Any]) -> int:
    rounds = row.get("rounds") or []
    if not rounds:
        return 0
    return sum(1 for probe in (rounds[0].get("probes") or [])
               if str(probe.get("kind")) == "probe")


def _has_competition(row: Mapping[str, Any]) -> bool:
    sources = row.get("candidate_sources")
    if sources is None:
        rounds = row.get("rounds") or []
        sources = (rounds[0] if rounds else {}).get("candidate_sources")
    sources = sources or {}
    return (int(sources.get("supplied") or 0) >= 1
            and int(sources.get("self_proposed") or 0) >= 1)


def _g2_nonzero(row: Mapping[str, Any]) -> bool:
    faces = row.get("g2") or {}
    return any(int(faces.get(key) or 0)
               for key in ("retrieval", "scope_match", "supply"))


def _ge(left: float, right: float) -> bool:
    return float(left) + TOL >= float(right)


def _le(left: float, right: float) -> bool:
    return float(left) <= float(right) + TOL


def _gt(left: float, right: float) -> bool:
    return float(left) > float(right) + TOL


def _arm_metrics(rows: Sequence[Mapping[str, Any]], arm: str
                 ) -> dict[str, Any]:
    bens = [row for row in rows
            if str(row.get("pilot_arm")) == arm
            and int(row.get("position") or 0) in BENEFICIARY_POSITIONS]
    all_rows = [row for row in rows if str(row.get("pilot_arm")) == arm]
    return {
        "positions": [int(row["position"]) for row in bens],
        "n_beneficiary": len(bens),
        "cum_gain": sum(float(row.get("heldout_gain") or 0.0) for row in bens),
        "llm_calls": sum(int(row.get("llm_calls") or 0) for row in bens),
        "support_receipts": sum(_support_receipts(row) for row in bens),
        "harm_events": sum(1 for row in bens if row.get("harm_event")),
        "g2_nonzero": sum(1 for row in bens if _g2_nonzero(row)),
        "harm_events_all_units": sum(
            1 for row in all_rows if row.get("harm_event")),
        "g2_nonzero_all_units": sum(
            1 for row in all_rows if _g2_nonzero(row)),
        "competition": [
            {"position": int(row["position"]),
             "unit_id": row.get("unit_id"),
             "candidate_sources": row.get("candidate_sources"),
             "reproduced": _has_competition(row)}
            for row in bens],
    }


def _improvement(arm: Mapping[str, Any],
                 control: Mapping[str, Any]) -> dict[str, float]:
    return {
        "gain": float(arm["cum_gain"]) - float(control["cum_gain"]),
        "llm_calls": float(control["llm_calls"]) - float(arm["llm_calls"]),
        "support_receipts": (
            float(control["support_receipts"])
            - float(arm["support_receipts"])),
    }


def _accepted_vs_control(arm: Mapping[str, Any],
                         control: Mapping[str, Any]) -> dict[str, Any]:
    gain_ok = _ge(arm["cum_gain"], control["cum_gain"])
    llm_ok = _le(arm["llm_calls"], control["llm_calls"])
    rec_ok = _le(arm["support_receipts"], control["support_receipts"])
    strict = (
        _gt(arm["cum_gain"], control["cum_gain"])
        or _gt(control["llm_calls"], arm["llm_calls"])
        or _gt(control["support_receipts"], arm["support_receipts"]))
    harm_zero = int(arm["harm_events"]) == 0 and int(control["harm_events"]) == 0
    g2_zero = int(arm["g2_nonzero"]) == 0 and int(control["g2_nonzero"]) == 0
    return {
        "gain_non_inferior": gain_ok,
        "costs_non_inferior": llm_ok and rec_ok,
        "strict_improvement": strict,
        "harm_zero": harm_zero,
        "g2_zero": g2_zero,
        "accepted_vs_control": bool(
            gain_ok and llm_ok and rec_ok and strict and harm_zero and g2_zero),
        "improvement": _improvement(arm, control),
    }


def _instrument_gate(rows: Sequence[Mapping[str, Any]], *,
                     stopped: str, llm_edit_illegal: bool) -> dict[str, Any]:
    return {
        "candidate": None,
        "instrument_stop": stopped,
        "note": "instrument/backend stop; not a scientific verdict",
        "beneficiary_positions": list(BENEFICIARY_POSITIONS),
        "tol": TOL,
        "llm_edit_ran": False,
        "llm_edit_proposal_illegal": bool(llm_edit_illegal),
        "seed_reproduced": False,
        "arms": {arm: _arm_metrics(rows, arm) for arm in ARMS},
        "comparisons": {
            "llm_vs_no_edit": None,
            "random_vs_no_edit": None,
        },
    }


def _judge(rows: Sequence[Mapping[str, Any]], *,
           llm_edit_illegal: bool) -> dict[str, Any]:
    metrics = {arm: _arm_metrics(rows, arm) for arm in ARMS}
    no_edit = metrics["no_edit"]
    seed_reproduced = any(item["reproduced"] for item in no_edit["competition"])
    llm_ran = bool(metrics["llm_edit"]["n_beneficiary"]) and not llm_edit_illegal
    comparisons = {
        "llm_vs_no_edit": _accepted_vs_control(
            metrics["llm_edit"], no_edit) if llm_ran else None,
        "random_vs_no_edit": (
            _accepted_vs_control(metrics["random_edit"], no_edit)
            if metrics["random_edit"]["n_beneficiary"] else None),
    }
    random_ge_llm = None
    if comparisons["llm_vs_no_edit"] and comparisons["random_vs_no_edit"]:
        llm_imp = comparisons["llm_vs_no_edit"]["improvement"]
        rnd_imp = comparisons["random_vs_no_edit"]["improvement"]
        random_ge_llm = all(
            _ge(rnd_imp[key], llm_imp[key])
            for key in ("gain", "llm_calls", "support_receipts"))
        comparisons["random_improvement_ge_llm"] = random_ge_llm
    if not seed_reproduced:
        candidate = "S3_SEED_UNREPRODUCED"
    elif llm_edit_illegal or not llm_ran:
        candidate = "S3_EDIT_REJECTED"
    elif not comparisons["llm_vs_no_edit"]["accepted_vs_control"]:
        candidate = "S3_EDIT_REJECTED"
    elif random_ge_llm:
        candidate = "EDIT_NOT_ATTRIBUTABLE"
    else:
        candidate = "S3_EDIT_ACCEPTED"
    return {
        "candidate": candidate,
        "instrument_stop": None,
        "note": (
            "三臂同课程,菜单 oracle 项对消,Σgain 高 ⟺ 累计 regret 低"),
        "beneficiary_positions": list(BENEFICIARY_POSITIONS),
        "tol": TOL,
        "llm_edit_ran": llm_ran,
        "llm_edit_proposal_illegal": llm_edit_illegal,
        "seed_reproduced": seed_reproduced,
        "arms": metrics,
        "comparisons": comparisons,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return "%+.4f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def _markdown(payload: Mapping[str, Any]) -> str:
    gate = payload.get("gate") or {}
    arms = gate.get("arms") or {}
    lines = [
        "# S3 pilot probe-policy",
        "",
        "protocol: %s" % payload.get("protocol"),
        "gate candidate: %s" % gate.get("candidate"),
        "instrument_stop: %s" % gate.get("instrument_stop"),
        "note: %s" % gate.get("note"),
        "",
        "## Random-legal-edit draw",
        "",
        json.dumps(payload.get("random_edit_draw"), ensure_ascii=False,
                   default=str),
        "",
        "## LLM-edit proposal",
        "",
        "illegal=%s attempts=%s" % (
            (payload.get("llm_edit_proposal") or {}).get("illegal"),
            (payload.get("llm_edit_proposal") or {}).get("attempts")),
        "",
        "## Arm metrics (beneficiary 2/4/5)",
        "",
        "| arm | n | cum_gain | llm_calls | receipts | harm | g2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        row = arms.get(arm) or {}
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (arm, row.get("n_beneficiary"), _fmt(row.get("cum_gain")),
               row.get("llm_calls"), row.get("support_receipts"),
               row.get("harm_events"), row.get("g2_nonzero")))
    lines += ["", "## Units", "",
              "| pos | unit | role | family | no_edit | random_edit | llm_edit |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    by: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        by.setdefault(str(row["unit_id"]), {})[str(row.get("pilot_arm"))] = row
    for unit in payload.get("course") or COURSE:
        pack = by.get(unit["unit_id"]) or {}
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (unit["position"], unit["unit_id"], unit["role"], unit["family"],
               _fmt((pack.get("no_edit") or {}).get("heldout_gain")),
               _fmt((pack.get("random_edit") or {}).get("heldout_gain")),
               _fmt((pack.get("llm_edit") or {}).get("heldout_gain"))))
    if payload.get("stop"):
        lines += ["", "## Stop", "", json.dumps(
            payload["stop"], ensure_ascii=False, default=str)]
    return "\n".join(lines) + "\n"


def _probe_backend() -> dict[str, Any]:
    probe = None
    for wave in range(1, 6):
        probe = ps0c.probe_new_backend()
        print("PROBE wave=%d ok=%s model=%s" % (
            wave, probe.get("ok"), probe.get("returned_model")),
              flush=True)
        if probe.get("ok"):
            break
        reason = str(probe.get("reason") or "")
        transient = any(token in reason for token in (
            "530", "1033", "tunnel_error", "retryable"))
        if not transient or wave == 5:
            break
        print("PROBE transient; sleep 120s before wave %d"
              % (wave + 1), flush=True)
        time.sleep(120)
    return probe or {"ok": False, "reason": "probe returned nothing"}


def _score(unit: Mapping[str, Any], result: Mapping[str, Any], *,
           family_sha: Mapping[str, str], arm: str,
           policy: Mapping[str, Any]) -> dict[str, Any]:
    skill_id = _minted_skill(str(unit["family"]), family_sha)
    scored = s2a._score_unit(unit, result, forecast_skill=skill_id)
    scored["scope_match"] = _rewrite_scope_match(scored)
    scored["pilot_arm"] = arm
    scored["policy"] = dict(policy)
    scored["family"] = unit["family"]
    return scored


def run_course(*, resume: bool = False) -> int:
    started = time.time()
    s2a.s1._set_phase(s2a.s1.PHASE_SETUP)
    s2a.s1.bind_curriculum_identity(
        task_kind="forecast", consumer_id=traffic.CONSUMER_ID,
        metric=traffic.METRIC)
    seed_trace = _extract_seed_trace()
    random_draw = _draw_random_edit()
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "course": list(COURSE),
        "touched_cells": [unit["unit_id"] for unit in COURSE],
        "seed_trace": seed_trace,
        "random_edit_draw": random_draw,
        "llm_edit_proposal": None,
        "llm_edit_proposal_illegal": False,
        "verdict_vocabulary": [
            "S3_EDIT_ACCEPTED", "S3_EDIT_REJECTED",
            "EDIT_NOT_ATTRIBUTABLE", "S3_SEED_UNREPRODUCED",
        ],
    }
    rows: list[dict[str, Any]] = []
    version_chains: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    ledger = _empty_ledger()
    done: set[tuple[str, int]] = set()
    llm_base = 0
    if resume and CHECKPOINT.is_file():
        saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        rows = list(saved.get("rows") or [])
        version_chains = {arm: list((saved.get("version_chains") or {}).get(arm)
                                    or [])
                          for arm in ARMS}
        saved_ledger = saved.get("ledger") or {}
        ledger["llm"] = int(saved_ledger.get("llm") or 0)
        ledger["fit"] = int(saved_ledger.get("fit") or 0)
        for arm in ARMS:
            part = (saved_ledger.get("per_arm") or {}).get(arm) or {}
            ledger["per_arm"][arm] = {
                "llm": int(part.get("llm") or 0),
                "fit": int(part.get("fit") or 0),
            }
        started = time.time() - float(saved.get("wall_seconds") or 0.0)
        llm_base = int(ledger["llm"] or 0)
        done = {(str(row["pilot_arm"]), int(row["position"])) for row in rows}
        if saved.get("random_edit_draw"):
            random_draw = saved["random_edit_draw"]
            payload["random_edit_draw"] = random_draw
        payload["llm_edit_proposal"] = saved.get("llm_edit_proposal")
        payload["llm_edit_proposal_illegal"] = bool(
            saved.get("llm_edit_proposal_illegal"))
        payload["resumed_from_checkpoint"] = sorted(
            "%s/%s" % item for item in done)

    def _save() -> None:
        CHECKPOINT.write_text(json.dumps(ps0c.redact(s2a.s1._plain({
            "rows": rows, "version_chains": version_chains, "ledger": ledger,
            "wall_seconds": round(time.time() - started, 1),
            "llm_edit_proposal": payload.get("llm_edit_proposal"),
            "llm_edit_proposal_illegal": payload.get(
                "llm_edit_proposal_illegal"),
            "random_edit_draw": random_draw,
        })), indent=1, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8")

    stopped: str | None = None
    first_fault: str | None = None
    bank = CellBank()
    assignments = {
        "no_edit": DEFAULT.to_dict(),
        "random_edit": dict(random_draw["policy"]),
    }
    try:
        probe = _probe_backend()
        payload["backend_probe"] = ps0c.redact({
            k: probe.get(k) for k in
            ("ok", "returned_model", "reason", "expected_model")})
        if not probe.get("ok"):
            raise Stop("BACKEND_UNAVAILABLE", str(probe.get("reason")))

        store_root = Path(tempfile.gettempdir()) / "s3_pilot_probe_policy"
        if not resume and store_root.exists():
            shutil.rmtree(store_root)
        backend = cls._live_backend(LLM_CAP_TOTAL)

        for arm in ARMS:
            if arm == "llm_edit":
                record = payload.get("llm_edit_proposal")
                if record is None:
                    record = _propose_llm_edit(seed_trace)
                    payload["llm_edit_proposal"] = record
                    payload["llm_edit_proposal_illegal"] = bool(
                        record.get("illegal"))
                    _save()
                if record.get("illegal") or not record.get("assignment"):
                    payload["llm_edit_proposal_illegal"] = True
                    print("ARM llm_edit skipped: proposal illegal", flush=True)
                    continue
                assignments[arm] = dict(record["assignment"])
            remaining = [unit for unit in COURSE
                         if (arm, int(unit["position"])) not in done]
            if not remaining:
                continue
            if time.time() - started > WALL_SECONDS:
                raise Stop("COMPUTE_BUDGET_EXCEEDED",
                           "wall before arm " + arm)
            print("ARM %s" % arm, flush=True)
            install_policy(ExplorationPolicy(**assignments[arm]))
            try:
                chain = list(version_chains[arm])
                replay_chain: list[dict[str, Any]] = []
                snapshot, family_sha = _replay_arm_state(
                    arm, rows, store_root / arm, replay_chain)
                if not chain:
                    version_chains[arm] = replay_chain
                    chain = version_chains[arm]
                arm_ledger = ledger["per_arm"][arm]
                for unit in remaining:
                    position = int(unit["position"])
                    uid = str(unit["unit_id"])
                    if time.time() - started > WALL_SECONDS:
                        raise Stop("COMPUTE_BUDGET_EXCEEDED",
                                   "wall before " + uid)
                    live_llm = llm_base + int(backend.calls)
                    if (live_llm >= LLM_CAP_TOTAL
                            or int(arm_ledger["llm"]) >= LLM_CAP_PER_ARM
                            or int(arm_ledger["fit"]) >= FIT_CAP_PER_ARM):
                        raise Stop("COMPUTE_BUDGET_EXCEEDED",
                                   "budget before %s/%s" % (arm, uid))
                    print("UNIT %s %d %s (%s)" % (
                        arm, position, uid, unit["role"]), flush=True)
                    cell = bank.get(uid)
                    calls_before = int(backend.calls)
                    fit_before = int(arm_ledger.get("fit") or 0)
                    result = s2a.run_unit(
                        unit=unit, cell=cell, arm="S3-%s" % arm,
                        base_snapshot=snapshot, backend=backend,
                        store_root=store_root / arm,
                        fit_ledger=arm_ledger)
                    arm_ledger["llm"] = (
                        int(arm_ledger.get("llm") or 0)
                        + int(backend.calls) - calls_before)
                    arm_ledger["fit"] = int(arm_ledger.get("fit") or fit_before)
                    ledger["llm"] = llm_base + int(backend.calls)
                    ledger["fit"] = sum(
                        int(part["fit"]) for part in ledger["per_arm"].values())
                    scored = _score(
                        unit, result, family_sha=family_sha, arm=arm,
                        policy=assignments[arm])
                    rows.append(scored)
                    faces = scored.get("g2") or {}
                    if any(int(faces.get(key) or 0) for key in
                           ("retrieval", "scope_match", "supply")):
                        first_fault = (
                            "G2_FIRST_FAULT %s/%s retrieval=%s "
                            "scope_match=%s supply=%s"
                            % (uid, arm, faces.get("retrieval"),
                               faces.get("scope_match"), faces.get("supply")))
                        raise Stop("S3_G2_LEAK", first_fault)
                    if unit["role"] == "producer" and unit["family"] not in family_sha:
                        snapshot, sha = _apply_producer_card(
                            snapshot, scored, unit,
                            store_root=store_root / arm, chain=chain, arm=arm)
                        if sha:
                            family_sha[str(unit["family"])] = sha
                    elif (unit["role"] == "beneficiary"
                          and unit["family"] in family_sha):
                        snapshot, sha = _apply_beneficiary_revision(
                            snapshot, scored, unit,
                            store_root=store_root / arm, chain=chain, arm=arm,
                            card_sha=family_sha[str(unit["family"])])
                        family_sha[str(unit["family"])] = sha
                    version_chains[arm] = chain
                    _save()
                    print("  %-12s deploy=%s gain=%+.4f g2=%s src=%s"
                          % (arm, scored.get("deploy_source"),
                             scored.get("heldout_gain") or 0.0,
                             scored.get("g2"), scored.get("candidate_sources")),
                          flush=True)
            finally:
                reset_policy()
    except (Stop, s2a.Stop, cls.Stop) as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
        if first_fault:
            payload["first_fault"] = first_fault
    except Exception as exc:  # noqa: BLE001
        import traceback
        text = "%s: %s" % (type(exc).__name__, exc)
        transport = (
            type(exc).__name__ in ("AgentTransportError", "InternalServerError")
            or "AgentTransportError" in text
            or "InternalServerError" in text
            or "530" in text
            or "tunnel_error" in text
        )
        stopped = "BACKEND_UNAVAILABLE" if transport else "INSTRUMENT_UNREADABLE"
        payload["stop"] = {
            "verdict": stopped,
            "reason": ps0c.redact(text),
            "traceback": ps0c.redact(traceback.format_exc()),
        }
        if transport:
            first_fault = "BACKEND_UNAVAILABLE — 授权中继隧道不可达"
            payload["first_fault"] = first_fault
    finally:
        reset_policy()
        s2a.s1.bind_curriculum_identity()

    if stopped in INSTRUMENT_STOPS:
        gate = _instrument_gate(
            rows, stopped=stopped,
            llm_edit_illegal=bool(payload.get("llm_edit_proposal_illegal")))
    else:
        gate = _judge(
            rows,
            llm_edit_illegal=bool(payload.get("llm_edit_proposal_illegal")))
    payload.update({
        "rows": rows,
        "version_chains": version_chains,
        "ledger": {**ledger,
                   "wall_seconds": round(time.time() - started, 1)},
        "gate": gate,
    })
    _save()
    if "backend" in locals():
        payload["first_returned_model"] = getattr(
            backend, "first_returned_model", None)
    OUT_JSON.write_text(
        json.dumps(ps0c.redact(s2a.s1._plain(payload)), indent=1,
                   ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    printed = gate.get("instrument_stop") or gate.get("candidate")
    print("GATE %s; llm=%s fit=%s"
          % (printed, ledger["llm"], ledger["fit"]),
          flush=True)
    return 0 if not stopped else 1


def smoke() -> int:
    print("SMOKE %s" % PROTOCOL, flush=True)
    bank = CellBank()
    for unit in COURSE:
        cell = bank.get(unit["unit_id"])
        block = np.asarray(cell["observation_block"])
        print("CELL %s train=%d support=%d delayed=%d heldout=%d "
              "observation_block=%d"
              % (unit["unit_id"],
                 len(cell.get("train") or ()),
                 len(cell.get("support") or ()),
                 len(cell.get("delayed") or ()),
                 len(cell.get("heldout") or ()),
                 int(block.size)),
              flush=True)
    draw = _draw_random_edit()
    print("RANDOM_EDIT param=%s value=%r policy=%s"
          % (draw["param"], draw["value"], draw["policy"]), flush=True)
    seed = _extract_seed_trace()
    missing = [key for key in (
        "protocol", "seed", "course_3", "version_chain_0",
        "a5_online_position_4", "position_4_counterfactual",
        "predictions_observed") if not seed.get(key)]
    a5 = seed.get("a5_online_position_4") or {}
    if not (a5.get("candidate_sources") and a5.get("rounds_0")
            and a5.get("deployment")):
        missing.append("a5_online_position_4.fields")
    if missing:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "seed trace missing %s" % missing)
    print("SEED_TRACE ok protocol=%s seed=%s course_3=%s "
          "version_chain_0.skill_id=%s P3=%s P6=%s"
          % (seed.get("protocol"), seed.get("seed"),
             (seed.get("course_3") or {}).get("unit_id"),
             (seed.get("version_chain_0") or {}).get("skill_id"),
             "P3" in (seed.get("predictions_observed") or {}),
             "P6" in (seed.get("predictions_observed") or {})),
          flush=True)
    prompt = _build_llm_prompt(seed)
    print("LLM_PROMPT chars=%d head=\n%s"
          % (len(prompt), prompt[:500]), flush=True)
    plans = (
        ("no_edit", DEFAULT.to_dict()),
        ("random_edit", draw["policy"]),
        ("llm_edit", DEFAULT.to_dict()),
    )
    for name, assignment in plans:
        try:
            installed = install_policy(ExplorationPolicy(**assignment))
            installed.validate()
            print("POLICY %s ok %s" % (name, installed.to_dict()), flush=True)
        finally:
            reset_policy()
    if reset_policy().to_dict() != DEFAULT.to_dict():
        raise Stop("INSTRUMENT_UNREADABLE", "reset_policy did not restore DEFAULT")
    print("SMOKE_OK 0 LLM 0 fit", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.smoke:
        return smoke()
    if args.run or args.resume:
        return run_course(resume=args.resume)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
