"""Domain Skill component (docs/DEV_DOMAIN_SKILL_V1_BUILD_TASK_2026-09-16.md §4-§6).

Three separate objects:
  Domain         neutral id (D01, D02); its business source is a Runtime binding, never a matching feature
  Batch profile  a program-computed, identity-free summary of ONE job's own T (from observe.overview)
  Domain Skill   a Workflow (+ optional Principles) carried by ONE experiment_guidance entry of the existing
                 Knowledge/Guidance, so Fast loads it through the unchanged render/scope path

Routing: known_domain (diagnostic external label, no inference) and profile_match (one short model call,
SELECTED/ABSTAIN from the supplied ids only). Offline formation: parse Slow proposals (<= 2 Workflows or
KEEP), select by actual Consumer results of complete Fast runs with the no-Skill option kept, freeze.
No I/O, fitting, network or hashing here; callers pass clients and results in.
"""
from __future__ import annotations

import copy
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field, replace

from methods.ttha import batch_research as br
from methods.ttha.batch_base import budget, llm, policy, spec

# ----------------------------------------------------------------------------- profile
PROFILE_FIELDS = ("lag24_corr", "lag168_corr", "nondc_spectrum_top5_energy_share", "standardized_trend_per_100h",
                  "last168_std_over_full_T_std", "standardized_abs_p95", "standardized_abs_max",
                  "r_head", "r_tail", "r_gap", "nn1_input_distance", "nn5_target_deviation")
PROFILE_STATS = ("p25", "median", "p75", "min", "max")
COMPATIBILITY_KEYS = ("task", "consumer", "lookback_L", "horizon_H", "sampling_interval_hours", "T_length", "entities_per_batch")
PROFILE_VERSION = 1
# keys that bind a profile to a concrete source; they exist only in the Runtime binding
IDENTITY_KEYS = frozenset({"dataset", "path", "job_id", "job", "roster", "roster_override", "uid", "entity", "entities", "train_rows",
                           "train_range", "rows_read", "t", "c_a", "c_b", "e", "scores", "e_scores", "external", "source",
                           "first_timestamp", "model_path", "model_refs"})
_JOB_TOKEN = re.compile(r"(?<![A-Za-z0-9])a\d{2}(?:_[A-Za-z0-9]+)?(?![A-Za-z0-9])")
_PATH_TOKEN = re.compile(r"[A-Za-z]:[\\/]|\\|\.(?:csv|json|npz|pt|py)\b|_scratch|(?:^|\s)\.{0,2}/[\w.-]+/")
MISSING_NOTE = ("The loader admits only all-finite T; per-entity missing_count is therefore the constant 0 under this "
                "eligibility contract and is not a missing-data diagnosis. It is excluded from the profile.")


def compatibility() -> dict:
    return {"task": "forecast_point_48_from_192", "consumer": spec.CONSUMER["arch"], "lookback_L": spec.L, "horizon_H": spec.H,
            "sampling_interval_hours": 1, "T_length": spec.TRAIN_SPAN, "entities_per_batch": spec.ROSTER_SIZE}


def batch_profile(overview: dict, job: spec.JobSpec) -> tuple[dict, dict]:
    """(matching view, runtime binding). The view keeps p25/median/p75 (+ existing min/max) and the valid entity
    count of each field; an all-unknown field stays null. Values are copied from overview['summary'] (computed by
    observe.overview from T only); nothing is recomputed or estimated here."""
    summary = overview["summary"]
    fields = {}
    for f in PROFILE_FIELDS:
        s = summary.get(f)
        fields[f] = None if s is None else {**{k: float(s[k]) for k in PROFILE_STATS}, "n_valid": int(s["n"])}
    view = {"profile_version": PROFILE_VERSION, "n_entities": int(overview["n_entities"]), "fields": fields,
            "compatibility": compatibility(), "notes": {"missing": MISSING_NOTE,
                                                        "stats": "p25/median/p75/min/max over the batch's entity rows; n_valid = entities with a defined value"}}
    ds = spec.DATASETS[job.dataset]
    binding = {"dataset": job.dataset, "path": str(ds["path"]), "job_id": job.job_id, "t": job.t, "train_rows": list(job.train_range),
               "roster": list(job.roster), "roster_explicit": job.roster_override is not None, "exposure": spec.EXPOSURE,
               "workload_rows": [job.train_range[0], max(job.e) + spec.H]}
    assert_identity_free(view)
    return view, binding


def assert_identity_free(value, *, extra_forbidden=()) -> None:
    """Fail closed if a matching view carries a binding key, a dataset name, a job-shaped token, a path or a label block."""
    names = tuple(n.lower() for n in spec.DATASETS) + tuple(str(x).lower() for x in extra_forbidden)

    def walk(item, where):
        if isinstance(item, dict):
            for k, v in item.items():
                if str(k).lower() in IDENTITY_KEYS:
                    raise PermissionError("identity/label key %r in matching view at %s" % (k, where))
                walk(v, where + "." + str(k))
        elif isinstance(item, list):
            for i, v in enumerate(item):
                walk(v, "%s[%d]" % (where, i))
        elif isinstance(item, str):
            low = item.lower()
            if any(n in low for n in names):
                raise PermissionError("source name in matching view at %s" % where)
            if _JOB_TOKEN.search(item) or _PATH_TOKEN.search(item) or re.search(r"entity_\d+|series_\d+|\buid\b", item, re.I):
                raise PermissionError("job/path/entity token in matching view at %s: %r" % (where, item[:80]))
    walk(value, "$")


# ----------------------------------------------------------------------------- Skill
HOOK = "experiment_guidance"
BODY_LIMIT = 1200
STATUSES = ("SMOKE_FIXTURE", "CANDIDATE_TEST_ONLY", "FROZEN_SELECTED", "NOT_PROMOTED", "PRINCIPLES_ADDENDUM_CANDIDATE")
ROUTABLE = ("FROZEN_SELECTED",)


@dataclass(frozen=True)
class Skill:
    skill_id: str
    domain_id: str | None            # None = common/general Skill
    revision: int
    workflow: str
    principles: str | None
    applicability_summary: str       # short text shown to the router; not the body
    compatibility_note: str
    observable_applicability: dict | None
    evidence_refs: tuple
    source_stage: str                # fixture | propose | freeze | principles_addendum
    status: str
    rendered_body: str = ""
    derived_from: str | None = None

    @property
    def variant(self) -> str:
        return "workflow+principles" if self.principles else "workflow_only"

    def to_json(self) -> dict:
        d = asdict(self)
        d["evidence_refs"] = list(self.evidence_refs)
        d["variant"] = self.variant
        return d


def render_body(workflow: str, principles: str | None) -> str:
    body = "Workflow:\n" + workflow.strip()
    if principles:
        body += "\n\nPrinciples:\n" + principles.strip()
    return body


def check_reusable_text(text: str, where: str, text_check=None) -> None:
    policy.validate_text(text, allow_entity_ids=False)          # existing identifier/code bans, unchanged
    low = str(text or "").lower()
    if any(n.lower() in low for n in spec.DATASETS) or _JOB_TOKEN.search(text or ""):
        raise ValueError("%s names a source or job" % where)
    if text_check is not None:                                  # a profile's additional source/job identifiers
        text_check(text, where)


def make_skill(*, skill_id: str, domain_id: str | None, revision: int, workflow: str, principles: str | None,
               applicability_summary: str, compatibility_note: str, observable_applicability, evidence_refs,
               legal_evidence_refs, source_stage: str, status: str, derived_from: str | None = None,
               allowed_features=None, text_check=None, body_limit: int | None = None) -> Skill:
    """Validate through the existing apply_update contract (one hook, <=1200 body, explicit scope, resolvable evidence).
    body_limit: a study's explicit opt-in capacity for the rendered body (None = the historical BODY_LIMIT of 1200)."""
    if status not in STATUSES:
        raise ValueError("unknown Skill status")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{1,40}", str(skill_id)):
        raise ValueError("unsafe skill id")
    if not isinstance(workflow, str) or not workflow.strip():
        raise ValueError("a Skill needs a Workflow")
    if principles is not None and (not isinstance(principles, str) or not principles.strip()):
        raise ValueError("principles must be null or nonempty text")
    body = render_body(workflow, principles)
    for text, where in ((body, "body"), (applicability_summary, "applicability_summary"), (compatibility_note, "compatibility_note")):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("%s must be nonempty text" % where)
        check_reusable_text(text, where, text_check)
    if len(applicability_summary) > 400:
        raise ValueError("applicability_summary is a short description (<=400 characters)")
    allowed = frozenset(allowed_features or _allowed_features())
    k = br.apply_update(br.Knowledge(), {"decision": "PROPOSE", "hook": HOOK, "body": body, "observable_applicability": observable_applicability,
                                         "evidence_refs": list(evidence_refs), "rationale": source_stage},
                        allowed_features=allowed, legal_evidence_refs=frozenset(legal_evidence_refs), body_limit=int(body_limit or BODY_LIMIT))
    g = k.entries[0]
    return Skill(skill_id, domain_id, int(revision), workflow.strip(), principles.strip() if principles else None, applicability_summary.strip(),
                 compatibility_note.strip(), g.applicability, tuple(g.evidence_refs), source_stage, status, g.body, derived_from)


def skill_from_json(d: dict) -> Skill:
    return Skill(d["skill_id"], d["domain_id"], int(d["revision"]), d["workflow"], d["principles"], d["applicability_summary"],
                 d["compatibility_note"], d["observable_applicability"], tuple(d["evidence_refs"]), d["source_stage"], d["status"],
                 d["rendered_body"], d.get("derived_from"))


def _allowed_features():
    return frozenset("batch_median_" + f for f in spec.OBS_FIELDS)


def skill_knowledge(skill: Skill | None, base: br.Knowledge | None = None) -> br.Knowledge:
    """The common start plus at most one Skill entry. No Skill = the common start unchanged (same tools, same budgets).
    Adding the entry bumps the knowledge version by one, like the historical frozen Generic control (version 1)."""
    base = copy.deepcopy(base or br.Knowledge())
    if skill is None:
        return base
    if any(e.hook == HOOK for e in base.entries):
        raise ValueError("common start already holds experiment_guidance; a Skill would silently replace it")
    item = br.Guidance(HOOK, skill.rendered_body, copy.deepcopy(skill.observable_applicability), tuple(skill.evidence_refs))
    return br.Knowledge(base.version + 1, tuple(base.entries) + (item,))


def scope_at(skill: Skill | None, batch_features: dict, allowed_features=None) -> dict:
    """allowed_features: a profile's own batch-feature vocabulary (default: the augmentation profile)."""
    if skill is None:
        return {"state": None, "missing_features": []}
    state, missing = br.scope_state(skill.observable_applicability, batch_features, frozenset(allowed_features or _allowed_features()))
    return {"state": state, "missing_features": missing}


# ----------------------------------------------------------------------------- routing
ROUTE_MODES = ("no_skill", "generic", "known_domain", "profile_match")
ROUTER_SYSTEM = '''You route one batch training-data preparation job to at most one frozen domain Skill before any work on the job starts. You receive: the current batch profile (program-computed T-only statistics of the job's own training segment), reference profiles of candidate domains (each a small list of earlier batches, neutral ids only), the compatibility conditions (task, consumer, L/H, sampling interval, T length), and each candidate Skill's id, domain id, compatibility note and short applicability description. You do not see Skill bodies, data, results or losses. Decide whether one candidate is applicable to the current batch. Compare the current profile with the reference profiles and the stated compatibility; a mismatch in compatibility or insufficient/contradictory profile evidence is a reason to abstain. Abstaining is a normal outcome: the job then runs the same public Fast start with all tools. Do not propose recipes, operators, parameters or expected gains, and do not guess a data source. Output exactly one JSON object, no markdown: {"selected_skill_id": "<one supplied skill id>" or null, "status": "SELECTED" or "ABSTAIN", "evidence_fields": ["<supplied profile field or compatibility key>", ...], "rationale": "<= 600 characters: why applicable, or why the information is insufficient"}. SELECTED requires a supplied id and at least one evidence field; ABSTAIN requires null.'''


def router_payload(query_view: dict, domain_refs: dict, skills: list) -> dict:
    """domain_refs: {domain_id: [reference profile views]} (formation batches only); skills: routable candidates."""
    payload = {
        "current_batch_profile": _rounded(query_view),
        "candidate_domains": [{"domain_id": d, "reference_batch_profiles": _rounded(v)} for d, v in sorted(domain_refs.items())],
        "candidate_skills": [{"skill_id": s.skill_id, "domain_id": s.domain_id, "compatibility_note": s.compatibility_note,
                              "applicability_summary": s.applicability_summary} for s in skills],
        "profile_fields": list(PROFILE_FIELDS), "compatibility_keys": list(COMPATIBILITY_KEYS),
        "contract": {"response": {"selected_skill_id": "id or null", "status": "SELECTED|ABSTAIN", "evidence_fields": [], "rationale": ""},
                     "note": "Routing is frozen for the whole job and never revisited after training feedback. Selection grants no new action permission; the Skill's own observable scope is still checked."},
    }
    assert_identity_free(payload)
    return payload


def _rounded(x, nd: int = 4):
    """Presentation copy for the model (4 decimals); the stored profile keeps the program-computed values."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _rounded(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_rounded(v, nd) for v in x]
    return x


def parse_router_response(resp, candidate_ids, *, profile_fields=None, compatibility_keys=None) -> dict:
    """profile_fields / compatibility_keys: a profile's own evidence vocabulary (default: the augmentation profile)."""
    evidence_vocabulary = tuple(profile_fields or PROFILE_FIELDS) + tuple(compatibility_keys or COMPATIBILITY_KEYS)
    if not isinstance(resp, dict) or set(resp) != {"selected_skill_id", "status", "evidence_fields", "rationale"}:
        raise ValueError("router response must hold exactly selected_skill_id, status, evidence_fields, rationale")
    status, sid, ev, why = resp["status"], resp["selected_skill_id"], resp["evidence_fields"], resp["rationale"]
    if status not in ("SELECTED", "ABSTAIN"):
        raise ValueError("router status must be SELECTED or ABSTAIN")
    if not isinstance(ev, list) or any(not isinstance(x, str) or x not in evidence_vocabulary for x in ev):
        raise ValueError("evidence_fields must name supplied profile fields or compatibility keys")
    if not isinstance(why, str) or not why.strip() or len(why) > 600:
        raise ValueError("rationale must be nonempty and <= 600 characters")
    if status == "SELECTED":
        if sid not in candidate_ids:
            raise ValueError("selected id was not supplied")
        if not ev:
            raise ValueError("SELECTED needs evidence fields")
    elif sid is not None:
        raise ValueError("ABSTAIN must carry a null id")
    return {"selected_skill_id": sid, "status": status, "evidence_fields": list(ev), "rationale": why}


def route_known_domain(domain_id: str, skills: list, batch_features: dict) -> dict:
    """Diagnostic reference: the external domain label picks that domain's frozen Skill. No model, no feature classifier."""
    match = [s for s in skills if s.domain_id == domain_id]
    if len(match) > 1:
        raise ValueError("more than one routable Skill for a domain")
    skill = match[0] if match else None
    rec = {"route_mode": "known_domain", "inference": "NONE__EXTERNAL_DOMAIN_LABEL",
           "note": "known-domain routing: the domain label is supplied by the runner; this is not model inference",
           "domain_id": domain_id, "status": "KNOWN_DOMAIN_SELECTED" if skill else "KNOWN_DOMAIN_NO_FROZEN_SKILL",
           "selected_skill_id": skill.skill_id if skill else None, "llm_requests": 0}
    rec["scope_at_route"] = scope_at(skill, batch_features)
    return rec


def route_profile_match(query_view: dict, domain_refs: dict, skills: list, call, batch_features: dict, *, diagnostic_domain_id=None,
                        payload=None, profile_fields=None, compatibility_keys=None, allowed_features=None) -> dict:
    """One short model call via call(payload) -> parsed JSON (the metered client). payload / profile_fields / compatibility_keys /
    allowed_features: another profile's prebuilt identity-free request and vocabularies (defaults: the augmentation profile). Outcome kinds are kept apart:
    NO_CANDIDATE_SKILLS (nothing routable; no call), ROUTER_CALL_FAILED (transport/account/budget; never an abstain),
    ROUTER_PARSE_OR_VALIDATION_FAILED, ABSTAIN, SELECTED (+ scope state; a non-MATCH scope loads nothing)."""
    rec = {"route_mode": "profile_match", "inference": "MODEL_PROFILE_MATCH", "candidate_skill_ids": [s.skill_id for s in skills],
           "candidate_domain_ids": sorted(domain_refs), "llm_requests": 0}
    if not skills:
        rec.update(status="NO_CANDIDATE_SKILLS", selected_skill_id=None, scope_at_route=scope_at(None, batch_features))
        return rec
    payload = router_payload(query_view, domain_refs, skills) if payload is None else payload
    rec["llm_requests"] = 1
    try:
        raw = call(payload)
    except (llm.AccountFault, llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
        rec.update(status="ROUTER_CALL_FAILED", fault_kind=type(exc).__name__, selected_skill_id=None, scope_at_route=scope_at(None, batch_features))
        return rec
    except ValueError as exc:            # response text was not JSON
        rec.update(status="ROUTER_PARSE_OR_VALIDATION_FAILED", error=str(exc)[:300], selected_skill_id=None, scope_at_route=scope_at(None, batch_features))
        return rec
    try:
        parsed = parse_router_response(raw, [s.skill_id for s in skills], profile_fields=profile_fields, compatibility_keys=compatibility_keys)
    except ValueError as exc:
        rec.update(status="ROUTER_PARSE_OR_VALIDATION_FAILED", error=str(exc)[:300], response=br.json_copy(raw) if _jsonable(raw) else None,
                   selected_skill_id=None, scope_at_route=scope_at(None, batch_features))
        return rec
    skill = next((s for s in skills if s.skill_id == parsed["selected_skill_id"]), None)
    rec.update(response=parsed, selected_skill_id=parsed["selected_skill_id"], scope_at_route=scope_at(skill, batch_features, allowed_features))
    if parsed["status"] == "ABSTAIN":
        rec["status"] = "ABSTAIN"
    else:
        rec["status"] = "SELECTED" if rec["scope_at_route"]["state"] == "MATCH" else "SELECTED_SCOPE_NOT_MATCHED"
        if diagnostic_domain_id is not None:
            rec["diagnostic"] = {"selected_domain_equals_runtime_domain": skill.domain_id == diagnostic_domain_id,
                                 "note": "diagnostic only; utility of the selected Skill is tested by real training later"}
    return rec


def _jsonable(x) -> bool:
    try:
        json.dumps(x, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def routed_skill(route: dict, skills: list) -> Skill | None:
    """The Skill a route hands to Fast (loaded or not is then decided by the unchanged scope render)."""
    if route["status"] in ("SELECTED", "SELECTED_SCOPE_NOT_MATCHED", "KNOWN_DOMAIN_SELECTED"):
        return next(s for s in skills if s.skill_id == route["selected_skill_id"])
    return None


# ----------------------------------------------------------------------------- offline formation: propose
SLOW_SYSTEM = '''You are the offline Slow of a batch training-data research Harness. You receive a deterministic census of completed or explicitly failed source branches from ONE neutral domain id: T-only observations, complete material plans, tool results, C_A feedback, commits, costs, failures and post-commit delayed C_B. No E values or rankings exist in this input, and no future-job data. Branches differ in knowledge and control mode and are not same-configuration repeats. Propose at most TWO reusable domain Workflows for future Fast jobs of this domain, or KEEP if the census does not support a reusable Workflow. A Workflow tells Fast how to observe the batch, which hypotheses and complete-plan constructions are worth trying, how to organize comparisons within the fixed budget, and when to stop or reuse a baseline; it may also recommend no augmentation or stopping early. Two candidates must follow genuinely different processing or research approaches; neither needs to change operators, be complex or be aggressive. Optional Principles state short evidence limits or cautions. Do not change model, scoring, tools, permissions or budgets; distinguish observed facts from hypotheses; never write an untested action as harmful; shared-plan loss differences are not causal entity labels. Do not name data sources, job labels, entity ids, and do not hand over one fixed historical assignment as the answer. Output exact JSON, no markdown: {"decision":"KEEP","rationale":"..."} OR {"decision":"PROPOSE","candidates":[{"candidate_id":"W1","research_mode":"<short label>","workflow":"...","principles":"..." or null,"observable_applicability":{"const":true},"applicability_summary":"<=400 characters","evidence_refs":["copied from legal_evidence_refs"],"rationale":"..."}]}. Workflow plus Principles render to at most 1200 characters. const:true is explicit unconditional; null is incomplete. Conditional scopes use only supplied batch feature names with numeric {feature,op,value} leaves. KEEP is valid and will not be resampled; every candidate is CANDIDATE_TEST_ONLY until real selection.'''

CANDIDATE_KEYS = {"candidate_id", "research_mode", "workflow", "principles", "observable_applicability", "applicability_summary", "evidence_refs", "rationale"}


def parse_proposal(resp, *, domain_id: str, legal_evidence_refs, max_candidates: int = 2, allowed_features=None, text_check=None) -> dict:
    """-> {'decision': 'KEEP'} or {'decision': 'PROPOSE', 'skills': [Skill...]} ; raises ValueError on contract errors."""
    if not isinstance(resp, dict) or resp.get("decision") not in ("KEEP", "PROPOSE"):
        raise ValueError("decision must be KEEP or PROPOSE")
    if resp["decision"] == "KEEP":
        if set(resp) - {"decision", "rationale"}:
            raise ValueError("KEEP cannot carry candidates")
        return {"decision": "KEEP", "rationale": str(resp.get("rationale", ""))[:1000], "skills": []}
    if set(resp) != {"decision", "candidates"}:
        raise ValueError("PROPOSE carries exactly decision and candidates")
    cands = resp["candidates"]
    if not isinstance(cands, list) or not 1 <= len(cands) <= max_candidates:
        raise ValueError("PROPOSE needs 1..%d candidates" % max_candidates)
    skills, ids, modes = [], set(), set()
    for c in cands:
        if not isinstance(c, dict) or set(c) != CANDIDATE_KEYS:
            raise ValueError("candidate keys must be exactly %s" % sorted(CANDIDATE_KEYS))
        cid, mode = c["candidate_id"], c["research_mode"]
        if not isinstance(cid, str) or not re.fullmatch(r"W[12]", cid) or cid in ids:
            raise ValueError("candidate ids are W1/W2 and unique")
        if not isinstance(mode, str) or not mode.strip() or len(mode) > 80 or mode.strip().lower() in modes:
            raise ValueError("each candidate needs its own short research_mode")
        ids.add(cid)
        modes.add(mode.strip().lower())
        check_reusable_text(mode, "research_mode", text_check)
        skills.append(make_skill(skill_id="%s-%s-r1" % (domain_id, cid), domain_id=domain_id, revision=1, workflow=c["workflow"],
                                 principles=c["principles"], applicability_summary=c["applicability_summary"],
                                 compatibility_note="Formed for this study's fixed task, Consumer, L/H, sampling interval and T length.",
                                 observable_applicability=c["observable_applicability"], evidence_refs=c["evidence_refs"],
                                 legal_evidence_refs=legal_evidence_refs, source_stage="propose", status="CANDIDATE_TEST_ONLY",
                                 allowed_features=allowed_features, text_check=text_check))
    if len({s.rendered_body for s in skills}) != len(skills):
        raise ValueError("two candidates render the same body")
    return {"decision": "PROPOSE", "skills": skills, "research_modes": {s.skill_id: c["research_mode"] for s, c in zip(skills, cands)},
            "rationales": {s.skill_id: str(c["rationale"])[:1000] for s, c in zip(skills, cands)}}


def propose(census: dict, *, domain_id: str, call, max_candidates: int = 2, allowed_features=None, text_check=None, forbidden_names=()) -> dict:
    """call(payload) -> parsed JSON (metered). One contract correction, no semantic resampling. Transport/account/budget
    faults end as PROPOSE_CALL_FAILED, never as KEEP. allowed_features / text_check / forbidden_names: a profile's own
    batch-feature vocabulary and source identifiers; defaults keep the augmentation profile."""
    refs = census["legal_evidence_refs"]
    payload = {"domain_id": domain_id, "census": census, "legal_evidence_refs": refs, "allowed_batch_features": sorted(allowed_features or _allowed_features()),
               "max_candidates": max_candidates, "status": "CANDIDATE_TEST_ONLY"}
    for n in tuple(spec.DATASETS) + tuple(forbidden_names):
        if n.lower() in json.dumps(payload, ensure_ascii=False).lower():
            raise PermissionError("Slow payload names a data source")
    attempts, prior = [], None
    for i in range(2):
        try:
            raw = call(payload)
        except (llm.AccountFault, llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
            attempts.append({"attempt": i, "fault_kind": type(exc).__name__})
            return {"status": "PROPOSE_CALL_FAILED", "attempts": attempts, "skills": []}
        except ValueError as exc:
            attempts.append({"attempt": i, "error": "response not JSON: %s" % str(exc)[:200]})
            prior = None
        else:
            try:
                out = parse_proposal(raw, domain_id=domain_id, legal_evidence_refs=refs, max_candidates=max_candidates,
                                     allowed_features=allowed_features, text_check=text_check)
                attempts.append({"attempt": i, "ok": True})
                return {"status": "KEEP" if out["decision"] == "KEEP" else "PROPOSED", "attempts": attempts, **out}
            except ValueError as exc:
                attempts.append({"attempt": i, "error": str(exc)[:300]})
                prior = raw if _jsonable(raw) else None
        if i == 0:
            payload = {**payload, "correction": {"previous_output": prior, "error": attempts[-1].get("error"),
                                                 "instruction": "Correct this contract error only. KEEP is allowed. Do not seek a different outcome."}}
    return {"status": "PROPOSE_PARSE_OR_VALIDATION_FAILED", "attempts": attempts, "skills": []}


# ----------------------------------------------------------------------------- offline formation: select / freeze
NO_SKILL = "NO_SKILL"


def select(options: dict, dev_jobs, run_option, *, block: str, score=None) -> dict:
    """options: {option_id: Skill|None}, must include NO_SKILL -> None. run_option(option_id, knowledge, job) runs one complete
    Fast job under the same budget and returns {'status', 'committed_plan_id', 'synthetic': bool, 'cost': {...}} (+ 'loss_by_seed'
    when no separate scorer is given). With score(option_id, job, record) -> {'loss_by_seed': [...]} the delayed block is read
    only after EVERY option of every dev job has committed (two phases). An option is eligible only if COMPLETE with a full
    loss vector on every dev job. The frozen rule is the lowest mean over dev jobs of the committed plan's three-seed `block`
    mean, ties in option order (NO_SKILL first); the Slow never scores. `block` has no default: frozen by the study."""
    if NO_SKILL not in options or options[NO_SKILL] is not None:
        raise ValueError("the no-Skill option must be kept")
    if block not in ("c_a", "c_b"):
        raise ValueError("selection uses a development block (c_a or c_b), never E")
    order = [NO_SKILL] + [o for o in options if o != NO_SKILL]
    runs, eligible = {}, {}
    for oid in order:
        runs[oid] = {}
        for job in dev_jobs:
            r = run_option(oid, skill_knowledge(options[oid]), job)
            runs[oid][job] = br.json_copy(r)
    if score is not None:                      # phase 2: every commit of the selection is frozen before any delayed read
        for oid in order:
            for job in dev_jobs:
                if runs[oid][job]["status"] == "COMPLETE":
                    runs[oid][job].update(br.json_copy(score(oid, job, runs[oid][job])))
    for oid in order:
        ok = all(r["status"] == "COMPLETE" and isinstance(r.get("loss_by_seed"), list) and r["loss_by_seed"] and
                 all(isinstance(x, (int, float)) and math.isfinite(x) for x in r["loss_by_seed"]) for r in runs[oid].values())
        if ok:
            eligible[oid] = statistics.mean(statistics.mean(r["loss_by_seed"]) for r in runs[oid].values())
    synthetic = any(r.get("synthetic") for per in runs.values() for r in per.values())
    rec = {"block": block, "dev_jobs": list(dev_jobs), "option_order": order, "runs": runs, "eligible_scores": eligible,
           "rule": "argmin over eligible options of mean_j(three-seed mean of committed plan loss on block); ties by option order; NO_SKILL kept",
           "synthetic": synthetic}
    if NO_SKILL not in eligible:
        rec.update(status="SELECTION_INCOMPLETE", selected=None, note="the no-Skill reference did not complete; nothing can be promoted")
        return rec
    winner = min(eligible, key=lambda o: (eligible[o], order.index(o)))
    rec["selected"] = winner
    rec["status"] = "NO_EFFECTIVE_CANDIDATE" if winner == NO_SKILL else "SELECTED"
    return rec


def freeze(domain_id: str, selection: dict, options: dict, *, evidence_scope: dict) -> dict:
    """Frozen record for one domain. Synthetic selections never yield FROZEN_SELECTED (smoke only)."""
    base = {"domain_id": domain_id, "selection": {k: selection[k] for k in ("status", "selected", "block", "dev_jobs", "eligible_scores", "rule", "synthetic")},
            "evidence_scope": evidence_scope, "no_skill_option_retained": True}
    if selection["status"] != "SELECTED":
        return {**base, "status": "NOT_PROMOTED", "skill": None}
    s = options[selection["selected"]]
    status = "SMOKE_FIXTURE" if selection["synthetic"] else "FROZEN_SELECTED"
    frozen = replace(s, status=status, source_stage="freeze", derived_from=s.skill_id)
    return {**base, "status": status, "skill": frozen.to_json()}


def principles_addendum(frozen: dict, principles: str, *, evidence_refs, legal_evidence_refs) -> dict:
    """A separate candidate revision with added Principles; the frozen record is not touched and the addendum is not
    routable until it passes its own selection."""
    if frozen.get("status") not in ("FROZEN_SELECTED", "SMOKE_FIXTURE") or not frozen.get("skill"):
        raise ValueError("an addendum needs a frozen Skill")
    s = skill_from_json(frozen["skill"])
    merged = (s.principles + "\n" + principles.strip()) if s.principles else principles
    add = make_skill(skill_id=re.sub(r"-r\d+$", "", s.skill_id) + "-r%d" % (s.revision + 1), domain_id=s.domain_id, revision=s.revision + 1,
                     workflow=s.workflow, principles=merged, applicability_summary=s.applicability_summary, compatibility_note=s.compatibility_note,
                     observable_applicability=s.observable_applicability, evidence_refs=evidence_refs, legal_evidence_refs=legal_evidence_refs,
                     source_stage="principles_addendum", status="PRINCIPLES_ADDENDUM_CANDIDATE", derived_from=s.skill_id)
    return {"status": "PRINCIPLES_ADDENDUM_CANDIDATE", "derived_from": s.skill_id, "frozen_record_unchanged": True, "skill": add.to_json()}


def routable(skills: list, *, allow_fixture: bool = False) -> list:
    ok = ROUTABLE + (("SMOKE_FIXTURE",) if allow_fixture else ())
    return [s for s in skills if s.status in ok]
