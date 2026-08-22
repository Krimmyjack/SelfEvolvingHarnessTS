"""S1b: is SMD a usable Target for S3?  0 LLM, <= 100 Consumer retrains.

Three questions, in order, each a gate on the next:

A  can the candidate's pre-execution conditions be expressed in the Fast
   public view at all, and does every program it recommends have evidence
   from two independent domains?  The answer revises v1 into a frozen v2
   before any SMD outcome is opened.
B  does the machine/channel mapping produce a legal 12+4 roster that holds
   the existing e1v2 window shape?
C  on that roster, is the Judge readable at identity, and does either shared
   program have real headroom?

Nothing here modifies the Harness, the Consumer, the Metric, the menu, the
risk line, the v2 ladder or the frozen v1 candidate.  ``FreshSearch`` already
exists to take a cohort payload instead of looking one up, so the SMD cohort
reaches the frozen evaluation path without a branch being added anywhere.

The official SMD test split and the anomaly labels are never opened.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "evaluation" / "functional")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_e2_fresh_confirmation as FC  # noqa: E402
from task_episode_harness.agentic import g3_sourcing as G3  # noqa: E402
from SelfEvolvingHarnessTS.contracts.observables import (  # noqa: E402
    OBSERVABLE_FEATURES,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
V1 = E2 / "shared_capability_candidate_v1.json"
ENTITY = E2 / "smd_entity_structure_v1.json"
OUT_V2_JSON = E2 / "shared_capability_candidate_v2.json"
OUT_V2_MD = E2 / "shared_capability_candidate_v2.md"
OUT_JSON = E2 / "smd_target_readiness_v1.json"
OUT_MD = E2 / "smd_target_readiness_v1.md"
PACKED = Path(r"C:\Users\辉\Desktop\Agent\shared_tsq_datasets\SMD\SMD_train.npy")

SHARED_PROGRAMS = ("outlier_iqr", "outlier_mad")
CONTRAST_PROGRAMS = ("hampel_filter", "winsorize")
ALL_PROGRAMS = SHARED_PROGRAMS + CONTRAST_PROGRAMS
MATERIAL = 0.005
HARM_LINE = -0.005
RETRAIN_BUDGET = 100
CHANNEL_PREFIX = 1104          # B1: channel choice reads only [0, 1104)
MIN_CARDINALITY = 20           # B2
TRAIN_N, EVAL_N = 12, 4        # B3
EPISODES = ((1, 1104), (2, 1800))  # B4


class Budget:
    def __init__(self, cap: int) -> None:
        self.cap, self.used, self.log = cap, 0, []

    def charge(self, n: int, what: str) -> None:
        self.used += int(n)
        self.log.append({"what": what, "retrains": int(n), "cumulative": self.used})
        if self.used > self.cap:
            raise Budget.Exceeded(
                "retrain budget %d exceeded at %d (%s)" % (self.cap, self.used, what)
            )

    class Exceeded(RuntimeError):
        pass


def window(start: int) -> dict[str, Any]:
    """The e1v2 triple, unchanged: support s/s+48/s+96, delayed s+144/+192/+240."""
    return {
        "start": int(start),
        "support_origins": [start, start + 48, start + 96],
        "delayed_origins": [start + 144, start + 192, start + 240],
        "horizon": 48,
        "farthest_index": start + 240 + 48,
        "syntax": "support s/s+48/s+96, delayed s+144/s+192/s+240, horizon 48",
    }


# =========================================================== Part A: candidate
def part_a() -> dict[str, Any]:
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    rows = v1["evidence_rows"]

    # ---- A1: is every pre-execution condition expressible? -----------------
    audit = []
    for cond in v1["candidate"]["applicability_conditions"]["conditions"]:
        feature = str(cond["feature"]).split(" over ")[0].strip()
        in_view = feature in OBSERVABLE_FEATURES
        if feature == "per-eval-series gain dispersion of the adopted plan":
            verdict, why = "NOT_A_PRE_EXECUTION_CONDITION", (
                "per_eval_series_gain is measured after a plan is adopted.  "
                "It is Risk feedback and it cannot be retrieved before "
                "execution, so it is restated as the rule that attaches the "
                "guard, not as a condition that selects the capability"
            )
        elif in_view:
            verdict, why = "EXPRESSIBLE", (
                "%s is in OBSERVABLE_FEATURES as %s"
                % (feature, OBSERVABLE_FEATURES[feature])
            )
        else:
            verdict, why = "DEMOTED_TO_EVIDENCE", (
                "%s is not in OBSERVABLE_FEATURES, so the Fast view cannot "
                "retrieve it.  The Observation surface is not extended this "
                "round; the reading stays in the card as evidence and is not "
                "an applicability gate" % feature
            )
        audit.append({
            "feature": feature, "in_observable_features": in_view,
            "verdict": verdict, "why": why,
            "original_requirement": cond["requirement"],
        })
    executable = [a for a in audit if a["verdict"] == "EXPRESSIBLE"]

    # ---- A2: per-operator support, by independent domain -------------------
    support: dict[str, Any] = {}
    for program in ALL_PROGRAMS:
        per_domain = {}
        for domain in ("traffic", "noaa"):
            hits = [r for r in rows[domain] if r["program"] == program]
            positive = [r for r in hits if r["delayed_aggregate_gain"] > MATERIAL]
            per_domain[domain] = {
                "rows": len(hits),
                "positive_rows": len(positive),
                "best_delayed_gain": (
                    max((r["delayed_aggregate_gain"] for r in hits), default=None)
                ),
                "episodes": [r["episode"] for r in hits],
            }
        two_domain = all(per_domain[d]["positive_rows"] > 0 for d in per_domain)
        support[program] = {
            "by_domain": per_domain,
            "two_domain_support": two_domain,
            "role": "shared" if two_domain else "contrast_only",
        }
    shared = tuple(p for p in ALL_PROGRAMS if support[p]["two_domain_support"])
    contrast = tuple(p for p in ALL_PROGRAMS if not support[p]["two_domain_support"])

    # ---- A3: v2 --------------------------------------------------------------
    v2 = {
        "protocol_version": "shared_capability_candidate_v2",
        "supersedes": "shared_capability_candidate_v1 (frozen, not modified)",
        "what_changed_from_v1": {
            "shared_programs": {
                "v1": list(v1["candidate"]["capability"]["programs"]),
                "v2": list(shared),
                "why": (
                    "%s have positive delayed evidence in both traffic and "
                    "NOAA; %s have it in traffic only, so they may be offered "
                    "as Source-local contrast or Target exploration but carry "
                    "no shared recommendation right"
                    % (", ".join(shared), ", ".join(contrast))
                ),
            },
            "applicability_conditions": {
                "kept_as_gates": [a["feature"] for a in executable],
                "demoted_to_evidence": [
                    a["feature"] for a in audit
                    if a["verdict"] == "DEMOTED_TO_EVIDENCE"
                ],
                "removed_from_pre_execution": [
                    a["feature"] for a in audit
                    if a["verdict"] == "NOT_A_PRE_EXECUTION_CONDITION"
                ],
                "observation_surface_unchanged": True,
            },
            "unchanged": [
                "the risk guard: min_per_series_gain, delayed window, "
                "threshold -0.005, both VETO_AND_FALL_BACK and "
                "RESCOPE_MASK_HARMED_SERIES",
                "the four fixed fields",
                "the out-of-scope declaration for imputation and level shift",
            ],
        },
        "status": "SHARED_CANDIDATE",
        "authorization": "GUIDANCE",
        "target_support_required": True,
        "grants_confirmation_free_try": False,
        "shared_programs": list(shared),
        "contrast_evidence_only": {
            "programs": list(contrast),
            "rights": (
                "Source-local contrast or Target exploration only; the Shared "
                "Candidate does not recommend them in advance"
            ),
            "evidence": {p: support[p]["by_domain"] for p in contrast},
        },
        "applicability_gates": [
            {
                "feature": a["feature"],
                "requirement": a["original_requirement"],
                "observable_type": OBSERVABLE_FEATURES.get(a["feature"]),
            }
            for a in executable
        ],
        "evidence_only_not_gates": [
            {"feature": a["feature"], "why": a["why"]}
            for a in audit if a["verdict"] != "EXPRESSIBLE"
        ],
        "guard": dict(v1["candidate"]["capability"]["guard"]),
        "out_of_scope": dict(v1["candidate"]["out_of_scope"]),
        "a1_audit": audit,
        "a2_source_support": support,
        "frozen_before_any_smd_outcome_was_opened": True,
    }

    # ---- A4: LODO on v2, shared programs only --------------------------------
    def lodo(compile_from: str, check_on: str) -> dict[str, Any]:
        compiled = sorted({
            r["program"] for r in rows[compile_from]
            if r["program"] in shared and r["delayed_aggregate_gain"] > MATERIAL
        })
        targets = [r for r in rows[check_on] if r["program"] in compiled]
        agree = [r for r in targets if r["delayed_aggregate_gain"] > MATERIAL]
        return {
            "compiled_on": compile_from, "checked_on": check_on,
            "shared_programs_compiled": compiled,
            "target_episodes_non_empty": len(targets),
            "direction_agrees": len(agree),
            "rows": [
                {"episode": r["episode"], "program": r["program"],
                 "delayed_aggregate_gain": r["delayed_aggregate_gain"],
                 "agrees": r["delayed_aggregate_gain"] > MATERIAL}
                for r in targets
            ],
            "verdict": (
                "SUPPORTED" if targets and len(agree) == len(targets)
                else "SOURCE_REPLAY_MISMATCH" if targets else "NO_TARGET_EPISODE"
            ),
        }

    a4 = {
        "traffic_to_noaa": lodo("traffic", "noaa"),
        "noaa_to_traffic": lodo("noaa", "traffic"),
        "same_source_replay": (
            "internal consistency only; not run as transfer evidence"
        ),
        "guard_is_not_a_result": (
            "the harm line that defines a harmed row is the same line the "
            "guard uses, so 'the guard finds the crossing' is true by "
            "construction and is not counted here"
        ),
    }
    v2["a4_lodo"] = a4
    ok = (
        bool(executable)
        and bool(shared)
        and all(support[p]["two_domain_support"] for p in shared)
        and a4["traffic_to_noaa"]["verdict"] == "SUPPORTED"
        and a4["noaa_to_traffic"]["verdict"] == "SUPPORTED"
    )
    return {"v2": v2, "audit": audit, "support": support, "lodo": a4,
            "shared": shared, "contrast": contrast, "passes": ok,
            "executable_gates": executable}


# ================================================== Part B: machine -> series
def part_b(entity: dict[str, Any]) -> dict[str, Any]:
    packed = np.load(PACKED, mmap_mode="r")
    mapping: list[dict[str, Any]] = []
    for machine in entity["machines"]:
        offset, length = int(machine["offset_in_packed_array"]), int(machine["train_length"])
        prefix = np.asarray(packed[offset:offset + CHANNEL_PREFIX], dtype=np.float64)
        cardinality = [int(np.unique(prefix[:, c]).size) for c in range(prefix.shape[1])]
        eligible = [(k, c) for c, k in enumerate(cardinality) if k > MIN_CARDINALITY]
        ranked = sorted(eligible, key=lambda t: (-t[0], t[1]))
        mapping.append({
            "entity": machine["entity"],
            "offset": offset,
            "train_length": length,
            "eligible_channels": len(eligible),
            "primary_channel": ranked[0][1] if ranked else None,
            "primary_cardinality": ranked[0][0] if ranked else None,
            "backup_channel": ranked[1][1] if len(ranked) > 1 else None,
            "backup_cardinality": ranked[1][0] if len(ranked) > 1 else None,
        })
    roster_entities = mapping[:TRAIN_N + EVAL_N]
    train = [m["entity"] for m in roster_entities[:TRAIN_N]]
    evaluation = [m["entity"] for m in roster_entities[TRAIN_N:]]
    farthest = max(window(s)["farthest_index"] for _, s in EPISODES)
    too_short = [m["entity"] for m in roster_entities if m["train_length"] < farthest]
    no_channel = [m["entity"] for m in roster_entities if m["primary_channel"] is None]
    return {
        "entity_unit": "machine",
        "channel_unit": "metric inside a machine",
        "channel_choice_reads": "[0, %d) of each machine's own train block only"
                                % CHANNEL_PREFIX,
        "rule": {
            "eligibility": "cardinality > %d" % MIN_CARDINALITY,
            "primary": "highest cardinality, ties to the lowest channel index",
            "backup": "second highest, ties to the lowest channel index",
            "backup_use": (
                "only once, and only if the primary mapping yields "
                "JUDGE_UNREADABLE; never because a Program's gain is poor"
            ),
        },
        "mapping": mapping,
        "roster": {
            "train": train, "eval": evaluation,
            "taken_from": "the first %d entities of the frozen packing order"
                          % (TRAIN_N + EVAL_N),
            "no_substitution": "entities are not replaced on Outcome",
        },
        "windows": {"episode_%d" % i: window(s) for i, s in EPISODES},
        "farthest_index_required": farthest,
        "entities_too_short": too_short,
        "entities_without_a_usable_channel": no_channel,
        "legal": not too_short and not no_channel and len(roster_entities) == 16,
    }


def build_payload(part: dict[str, Any], which: str) -> dict[str, Any]:
    packed = np.load(PACKED, mmap_mode="r")
    by_entity = {m["entity"]: m for m in part["mapping"]}
    values: dict[str, np.ndarray] = {}
    chosen: dict[str, int] = {}
    for entity in part["roster"]["train"] + part["roster"]["eval"]:
        m = by_entity[entity]
        channel = m["primary_channel"] if which == "primary" else m["backup_channel"]
        chosen[entity] = int(channel)
        values[entity] = np.asarray(
            packed[m["offset"]:m["offset"] + m["train_length"], channel],
            dtype=np.float64,
        )
    payload = FC._cohort_payload(
        part["roster"]["train"], part["roster"]["eval"], values
    )
    payload["name"] = "smd_machine_%s_channel_v1" % which
    payload["exposure"] = (
        "SMD selected train development windows: context = INSTANCE_SEEN, "
        "outcome = EXPOSED.  SMD official test / labels: outcome = SEALED "
        "(never read by this protocol)."
    )
    return {"payload": payload, "chosen_channels": chosen}


# ============================================ Part C: Judge and Program headroom
def part_c(part: dict[str, Any], budget: Budget, which: str) -> dict[str, Any]:
    built = build_payload(part, which)
    payload = built["payload"]
    out: dict[str, Any] = {
        "mapping": which,
        "chosen_channels": built["chosen_channels"],
        "episodes": {},
    }
    searches: dict[str, Any] = {}
    readable = True
    for index, start in EPISODES:
        win = window(start)
        search = FC.FreshSearch(
            payload=payload, consumer_variant="pooled",
            support_origins=win["support_origins"],
            delayed_origins=win["delayed_origins"],
        )
        budget.charge(int(search.retrains), "identity baselines, episode %d" % index)
        searches["episode_%d" % index] = search
        support_read = FC._readability(
            search._identity_support, search.eval_uids, G3.CRITERIA,
            block="support",
        )
        delayed_read = FC._readability(
            search._identity_delayed, search.eval_uids, G3.CRITERIA,
            block="delayed",
        )
        ok = bool(support_read.get("pass")) and bool(delayed_read.get("pass"))
        readable = readable and ok
        out["episodes"]["episode_%d" % index] = {
            "window": win,
            "identity_readability": {"support": support_read, "delayed": delayed_read},
            "readable": ok,
        }
        print("C1 %s ep%d readable=%s" % (which, index, ok), flush=True)
    out["judge_readable"] = readable
    if not readable:
        return out

    programs: dict[str, Any] = {}
    for program in ALL_PROGRAMS:
        per_episode = {}
        for index, _ in EPISODES:
            search = searches["episode_%d" % index]
            support = search.full_batch_support(program)
            budget.charge(len(search.support), "%s support, episode %d" % (program, index))
            delayed = search.delayed_gate(program, [])
            budget.charge(len(search.delayed), "%s delayed, episode %d" % (program, index))
            # "Did the input actually change" is asked of readings that
            # were already paid for, not by reaching into the operator: an
            # inert program leaves every per-view sMASE identical to
            # identity's, so every gain it reports is exactly 0.0.  This is
            # the operationally meaningful sense -- did the change reach
            # the Consumer -- and it needs no new machinery.
            probes = [float(support["aggregate_gain"]),
                      float(delayed["aggregate_gain"])]
            probes += [float(v) for v in
                       (support.get("per_eval_series_gain") or {}).values()]
            probes += [float(v) for v in
                       (delayed.get("per_eval_series_gain") or {}).values()]
            changed = any(value != 0.0 for value in probes)
            vector = {
                str(k): float(v)
                for k, v in (delayed.get("per_eval_series_gain") or {}).items()
            }
            per_episode["episode_%d" % index] = {
                "changed_what_the_consumer_saw": changed,
                "support_aggregate_gain": float(support["aggregate_gain"]),
                "delayed_aggregate_gain": float(delayed["aggregate_gain"]),
                "per_eval_series_delayed_gain": vector,
                "harmed_eval_series": sorted(
                    u for u, g in vector.items() if g < HARM_LINE
                ),
                "min_per_series_gain": min(vector.values()) if vector else None,
            }
            print("C2 %-14s ep%d changed=%-5s sup %+.6f del %+.6f harm %s" % (
                program, index, changed, support["aggregate_gain"],
                delayed["aggregate_gain"],
                per_episode["episode_%d" % index]["harmed_eval_series"] or "none",
            ), flush=True)
        programs[program] = per_episode
    out["programs"] = programs

    def positive(program: str) -> bool:
        rows = programs[program]
        return any(
            r["changed_what_the_consumer_saw"] and r["delayed_aggregate_gain"] > MATERIAL
            for r in rows.values()
        )

    shared_hits = [p for p in SHARED_PROGRAMS if positive(p)]
    contrast_hits = [p for p in CONTRAST_PROGRAMS if positive(p)]
    out["headroom"] = {
        "definition": (
            "at least one shared program (outlier_iqr or outlier_mad) "
            "changes what the Consumer sees and reaches delayed gain "
            "> +%.3f on at least one pre-locked episode" % MATERIAL
        ),
        "shared_programs_with_headroom": shared_hits,
        "contrast_programs_with_headroom": contrast_hits,
        "shared_headroom_available": bool(shared_hits),
        "target_local_headroom_only": bool(contrast_hits and not shared_hits),
        "per_series_harm_does_not_cancel_headroom": (
            "harm is recorded in full because handling it is the mechanism "
            "VETO/RESCOPE exists to test in S3; it does not remove headroom"
        ),
    }
    return out


def run() -> int:
    started = time.perf_counter()
    budget = Budget(RETRAIN_BUDGET)
    entity = json.loads(ENTITY.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "protocol_version": "smd_target_readiness_v1",
        "question": (
            "without modifying the Harness and without the SMD held-out "
            "outcome, does SMD have a correct entity structure, a readable "
            "Forecasting Judge, and real outlier Program headroom against "
            "the Shared Candidate"
        ),
        "llm_calls": 0,
        "retrain_budget": RETRAIN_BUDGET,
        "exposure": {
            "smd_selected_train_development_windows": {
                "context": "INSTANCE_SEEN", "outcome": "EXPOSED",
            },
            "smd_official_test_and_labels": {"outcome": "SEALED"},
            "read_this_round": "the official train split only",
        },
        "nothing_modified": [
            "Harness", "Consumer", "Metric", "menu", "risk line",
            "v2 adoption ladder", "the frozen v1 candidate",
            "the Observation surface",
        ],
    }
    try:
        a = part_a()
        payload["part_a"] = {
            "a1_condition_audit": a["audit"],
            "a2_source_support": a["support"],
            "a4_lodo": a["lodo"],
            "passes": a["passes"],
        }
        OUT_V2_JSON.write_text(
            json.dumps(a["v2"], indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8", newline="\n",
        )
        OUT_V2_MD.write_text(_v2_markdown(a["v2"]), encoding="utf-8", newline="\n")
        print("wrote", OUT_V2_JSON, flush=True)
        if not a["executable_gates"]:
            payload["verdict"] = "CANDIDATE_CONTEXT_NOT_EXECUTABLE"
            payload["verdict_reason"] = (
                "no applicability condition survives as something the Fast "
                "public view can retrieve"
            )
            return _write(payload, started, budget)
        if not a["passes"]:
            payload["verdict"] = "NO_INDEPENDENT_SOURCE_SUPPORT"
            payload["verdict_reason"] = (
                "shared programs %s; lodo %s / %s" % (
                    a["shared"], a["lodo"]["traffic_to_noaa"]["verdict"],
                    a["lodo"]["noaa_to_traffic"]["verdict"],
                )
            )
            return _write(payload, started, budget)

        b = part_b(entity)
        payload["part_b"] = b
        if not b["legal"]:
            payload["verdict"] = "SMD_ENTITY_MAPPING_BLOCKED"
            payload["verdict_reason"] = (
                "too short: %s; no usable channel: %s"
                % (b["entities_too_short"], b["entities_without_a_usable_channel"])
            )
            return _write(payload, started, budget)

        c = part_c(b, budget, "primary")
        payload["part_c"] = {"primary": c}
        if not c["judge_readable"]:
            print("primary mapping unreadable; taking the pre-registered backup",
                  flush=True)
            c = part_c(b, budget, "backup")
            payload["part_c"]["backup"] = c
            payload["backup_mapping_used"] = (
                "the primary mapping produced JUDGE_UNREADABLE, which is the "
                "only condition that releases the backup rule"
            )
        if not c["judge_readable"]:
            payload["verdict"] = "JUDGE_UNREADABLE"
            payload["verdict_reason"] = (
                "identity readings fail the reused NOAA readability bar on "
                "both the primary and the backup mapping"
            )
            return _write(payload, started, budget)
        payload["headroom"] = c["headroom"]
        if not c["headroom"]["shared_headroom_available"]:
            payload["verdict"] = "NO_PROGRAM_HEADROOM"
            payload["verdict_reason"] = (
                "no shared program both changes the input and clears "
                "+%.3f delayed on either episode%s" % (
                    MATERIAL,
                    "; TARGET_LOCAL_HEADROOM_ONLY: %s"
                    % c["headroom"]["contrast_programs_with_headroom"]
                    if c["headroom"]["target_local_headroom_only"] else "",
                )
            )
            return _write(payload, started, budget)
        payload["verdict"] = "SMD_TARGET_READY"
        payload["verdict_reason"] = (
            "the candidate's gates are expressible, both shared programs have "
            "two-domain support, the 12+4 roster is legal, the Judge reads at "
            "identity on both episodes, and %s has headroom"
            % ", ".join(c["headroom"]["shared_programs_with_headroom"])
        )
        payload["what_this_does_not_mean"] = [
            "the candidate has not transferred; S3 has not run",
            "no Shared Capability execution right is granted",
            "no A5-over-A3 conclusion is produced",
        ]
    except Budget.Exceeded as exc:
        payload["verdict"] = "INCOMPLETE_BUDGET"
        payload["verdict_reason"] = str(exc)
    return _write(payload, started, budget)


def _write(payload: dict[str, Any], started: float, budget: Budget) -> int:
    payload["consumer_retrains"] = budget.used
    payload["retrain_log"] = budget.log
    payload["wall_seconds"] = time.perf_counter() - started
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("verdict", payload["verdict"], "| retrains", budget.used, flush=True)
    return 0


def _v2_markdown(v2: dict[str, Any]) -> str:
    ch = v2["what_changed_from_v1"]
    lines = [
        "# Shared Capability candidate v2 (Source-only revision)",
        "",
        "Frozen before any SMD outcome was opened.  v1 is not modified.",
        "",
        "```json",
        json.dumps({
            "status": v2["status"], "authorization": v2["authorization"],
            "target_support_required": v2["target_support_required"],
            "grants_confirmation_free_try": v2["grants_confirmation_free_try"],
            "shared_programs": v2["shared_programs"],
            "guard": v2["guard"],
        }, indent=1, ensure_ascii=False),
        "```",
        "",
        "## What changed",
        "",
        "- Shared programs: `%s` -> `%s`.  %s" % (
            ch["shared_programs"]["v1"], ch["shared_programs"]["v2"],
            ch["shared_programs"]["why"]),
        "- Gates kept: %s." % (ch["applicability_conditions"]["kept_as_gates"] or "none"),
        "- Demoted to evidence: %s." % (
            ch["applicability_conditions"]["demoted_to_evidence"] or "none"),
        "- Removed from pre-execution: %s." % (
            ch["applicability_conditions"]["removed_from_pre_execution"] or "none"),
        "- Unchanged: %s" % "; ".join(ch["unchanged"]),
        "",
        "## Source support, by independent domain",
        "",
        "| program | traffic rows / positive | noaa rows / positive | two-domain | role |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for program, row in v2["a2_source_support"].items():
        t, n = row["by_domain"]["traffic"], row["by_domain"]["noaa"]
        lines.append("| `%s` | %d / %d | %d / %d | %s | %s |" % (
            program, t["rows"], t["positive_rows"], n["rows"], n["positive_rows"],
            row["two_domain_support"], row["role"]))
    lines.extend(["", "## Bidirectional LODO on the shared programs", "",
                  "| direction | compiled | target episodes | agrees | verdict |",
                  "| --- | --- | ---: | ---: | --- |"])
    for key in ("traffic_to_noaa", "noaa_to_traffic"):
        arm = v2["a4_lodo"][key]
        lines.append("| %s | %s | %d | %d | `%s` |" % (
            key, arm["shared_programs_compiled"], arm["target_episodes_non_empty"],
            arm["direction_agrees"], arm["verdict"]))
    lines.extend(["", v2["a4_lodo"]["guard_is_not_a_result"], ""])
    return "\n".join(lines) + "\n"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SMD target readiness (S1b)",
        "",
        "**Verdict: `%s`** -- %s" % (payload["verdict"],
                                     payload.get("verdict_reason", "")),
        "",
        "Exposure: SMD selected train development windows `context = "
        "INSTANCE_SEEN`, `outcome = EXPOSED`; SMD official test and labels "
        "`outcome = SEALED`, never read.",
        "",
    ]
    b = payload.get("part_b")
    if b:
        lines.extend([
            "## Machine -> series mapping",
            "",
            "Entity = machine, channel = a metric inside it.  Channel choice "
            "reads %s.  %s" % (b["channel_choice_reads"], b["rule"]["primary"]),
            "",
            "Roster (first 16 of the frozen packing order, no substitution): "
            "train `%s`; eval `%s`." % (b["roster"]["train"], b["roster"]["eval"]),
            "",
            "| entity | train rows | eligible channels | primary ch | card | backup ch | card |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for m in b["mapping"]:
            lines.append("| `%s` | %d | %d | %s | %s | %s | %s |" % (
                m["entity"], m["train_length"], m["eligible_channels"],
                m["primary_channel"], m["primary_cardinality"],
                m["backup_channel"], m["backup_cardinality"]))
        lines.append("")
    for which in ("primary", "backup"):
        c = (payload.get("part_c") or {}).get(which)
        if not c:
            continue
        lines.extend(["## Judge and headroom -- %s mapping" % which, "",
                      "Judge readable: **%s**." % c["judge_readable"], "",
                      "| episode | support spread | support share | delayed "
                      "spread | delayed share | readable |",
                      "| --- | ---: | ---: | ---: | ---: | --- |"])
        for name, ep in c["episodes"].items():
            s = ep["identity_readability"]["support"]
            d = ep["identity_readability"]["delayed"]
            lines.append("| %s (s=%d) | %.3g | %.3g | %.3g | %.3g | %s |" % (
                name, ep["window"]["start"],
                s.get("eval_loss_spread", float("nan")),
                s.get("largest_single_series_loss_share", float("nan")),
                d.get("eval_loss_spread", float("nan")),
                d.get("largest_single_series_loss_share", float("nan")),
                ep["readable"]))
        if c.get("programs"):
            lines.extend(["", "| program | episode | changed input | support "
                          "gain | delayed gain | harmed | min per-series |",
                          "| --- | --- | --- | ---: | ---: | --- | ---: |"])
            for program, eps in c["programs"].items():
                for name, r in eps.items():
                    lines.append("| `%s` | %s | %s | %+.6f | %+.6f | %s | %s |" % (
                        program, name, r["changed_what_the_consumer_saw"],
                        r["support_aggregate_gain"], r["delayed_aggregate_gain"],
                        ", ".join(r["harmed_eval_series"]) or "none",
                        ("%+.6f" % r["min_per_series_gain"])
                        if r["min_per_series_gain"] is not None else "--"))
            h = c["headroom"]
            lines.extend(["", "**Headroom.** %s" % h["definition"],
                          "", "- Shared programs with headroom: %s."
                          % (h["shared_programs_with_headroom"] or "none"),
                          "- Contrast programs with headroom: %s."
                          % (h["contrast_programs_with_headroom"] or "none"), ""])
    lines.extend([
        "## Cost", "",
        "- LLM calls: 0.  Consumer retrains: %s / %s." % (
            payload.get("consumer_retrains"), payload.get("retrain_budget")),
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
        "",
        "Nothing modified: %s." % ", ".join(payload["nothing_modified"]),
    ])
    if payload.get("what_this_does_not_mean"):
        lines.extend(["", "## What this verdict does not mean", ""])
        for row in payload["what_this_does_not_mean"]:
            lines.append("- %s" % row)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(run())
