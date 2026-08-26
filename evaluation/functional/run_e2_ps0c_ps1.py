"""PS-0c + (conditional) PS-1 -- re-earn PowerCons, then the frozen protocol.

PS-0b confirmed a dual-source *structure* for ``hampel_filter`` at the
unit × oracle-operator surface (GPA 4/4 + PowerCons impulse 3/4).  PS-1
needs live Episodes.  Source A (GPA, ``ps0_srcA_1``) already exists on
the repaired record path.  Source B must be re-earned; the S1c / PS-0
PowerCons episodes stay cancelled.

This runner does not change the frozen PS-1 protocol.  It:

1. switches the live path onto the new relay (``M0_AGENT_*`` / ``CPA_API_KEY``)
   at evaluation-runner level -- no ``methods/`` / ``runtime/`` /
   ``contracts/`` / ``operators/`` edit;
2. re-earns ``PowerCons__impulse_v2`` on A3-reset, isomorphic to PS-0
   (run-ids ``ps0_srcB_3`` / ``ps0_srcB_4``, take-what-comes, no hinting);
3. if earned, re-verifies five-axis Scope from the persisted pattern
   leaves and executes the already-submitted PS-1 functions
   (card compile, 12-run plan, frozen verdict table).

Book caps (hard): LLM ≤180, fit ≤160, wall ≤2.5 h.  Probe is not charged.
The GPA Episode was earned on the previous (agicto) backend; Episode
validity is a consumer reading and does not depend on the relay.
Proposal-behaviour drift is absorbed by the three-arm same-backend
contrast inside PS-1.

Entry::

  python evaluation/functional/run_e2_ps0c_ps1.py --probe-only
  python evaluation/functional/run_e2_ps0c_ps1.py --run
  python evaluation/functional/run_e2_ps0c_ps1.py --ps1-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_e2_ps0_reearn_sources as ps0  # noqa: E402
import run_e2_ps1_arms as ps1  # noqa: E402
import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
PS0_JSON = E2 / "ps0_reearn_sources.json"
OUT_JSON = E2 / "ps0c_reearn_powercons.json"
OUT_MD = E2 / "ps0c_reearn_powercons.md"
DUAL_JSON = E2 / "ps0c_dual_source.json"

PROTOCOL_VERSION = "ps0c_reearn_powercons_v1"
EVIDENCE_GRADE = "development-mechanism"

SOURCE_B_SCENE = {
    "scene": "source_B_prime",
    "unit_id": "PowerCons__impulse_v2",
    "dataset": "PowerCons",
    "injection": "impulse_v2",
    "series_length": 144,
    "family_key": "PowerCons",
    "run_ids": ("ps0_srcB_3", "ps0_srcB_4"),
}

LLM_TOTAL_CAP = 180
FIT_TOTAL_CAP = 160
WALL_SECONDS_CAP = int(2.5 * 60 * 60)

SOURCE_A_BACKEND_NOTE = (
    "source A' (GPA, ps0_srcA_1) was earned on gpt-5.6-sol@agicto.  "
    "Episode validity is the consumer Support / delayed reading and does "
    "not depend on the relay.  Proposal-behaviour differences are absorbed "
    "by the PS-1 three-arm same-backend contrast.  This book does not "
    "fall back to the old relay."
)

_SECRET_RE = re.compile(r"sk-[A-Za-z0-9]{16,}")


# =========================================================================== #
# new-relay install (evaluation layer only)
# =========================================================================== #
def _env(name: str) -> str:
    return str(os.environ.get(name, "") or "").strip()


def _relay_cfg() -> dict[str, str]:
    return {
        "base_url": _env("M0_AGENT_BASE_URL"),
        "model": _env("M0_AGENT_MODEL"),
        "api_key": _env("CPA_API_KEY"),
    }


def _host_of(url: str) -> str:
    try:
        return str(urlsplit(url).hostname or "")
    except Exception:  # noqa: BLE001
        return ""


def install_new_backend() -> dict[str, str]:
    """Point the shared live factories at the new relay.

    ``_live_agent`` re-imports ``e1.NF_BASE_URL`` at call time.
    ``_default_backend_factory`` reads ``runner.NF_BASE_URL`` from its
    module global.  Both must be patched.  The API key is copied into
    ``OPENAI_API_KEY`` because the factory only looks there / at
    ``AGICTO_API_KEY``.  No file under methods/runtime/contracts/operators
    is touched.
    """
    cfg = _relay_cfg()
    if not cfg["base_url"] or not cfg["model"] or not cfg["api_key"]:
        raise s1.Stop(
            "BACKEND_UNAVAILABLE",
            "M0_AGENT_BASE_URL / M0_AGENT_MODEL / CPA_API_KEY are not all set")
    os.environ["OPENAI_API_KEY"] = cfg["api_key"]
    import evaluation.functional.task_episode_harness.e1 as e1
    import evaluation.functional.task_episode_harness.normal_flow as nf
    import evaluation.functional.task_episode_harness.agentic.runner as runner

    for module in (nf, e1, runner):
        module.NF_BASE_URL = cfg["base_url"]
        module.NF_MODEL = cfg["model"]
    cls.SLOW_MODEL = cfg["model"]
    return {"base_url": cfg["base_url"], "model": cfg["model"],
            "host": _host_of(cfg["base_url"])}


def _secret_values() -> list[str]:
    values = []
    for name in ("CPA_API_KEY", "OPENAI_API_KEY", "AGICTO_API_KEY"):
        value = _env(name)
        if value:
            values.append(value)
    return values


def redact(value: Any) -> Any:
    """Strip live keys from anything that will be written to disk."""
    secrets = _secret_values()

    def _scrub(text: str) -> str:
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return _SECRET_RE.sub("[REDACTED]", text)

    if isinstance(value, str):
        return _scrub(value)
    if isinstance(value, Mapping):
        return {str(key): redact(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    return value


def probe_new_backend() -> dict[str, Any]:
    """Identity probe of the new relay.  Not charged.  No old-relay fallback."""
    cfg = _relay_cfg()
    identity = {
        "expected_model": cfg["model"],
        "expected_host": _host_of(cfg["base_url"]),
        "expected_base_url": cfg["base_url"],
        "family": "M0_AGENT_* trycloudflare relay (first book)",
        "charged_to_course_cap": False,
        "old_relay_fallback": False,
    }
    if not cfg["base_url"] or not cfg["model"] or not cfg["api_key"]:
        return {**identity, "ok": False,
                "reason": "new-relay env vars missing",
                "returned_model": None}
    try:
        import openai
        from SelfEvolvingHarnessTS.runtime.agent_backend import (
            _relay_error_payload,
        )

        client = openai.OpenAI(
            api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=60)
        last_reason = "probe failed"
        last_model = None
        # Same relay only.  A single upstream TLS EOF is not "unavailable".
        for attempt in range(1, 4):
            try:
                completion = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "user",
                               "content": "Reply with the single word pong."}])
                relay_error = _relay_error_payload(completion)
                last_model = getattr(completion, "model", None)
                if relay_error:
                    last_reason = "relay error payload: %s" % relay_error
                elif not (getattr(completion, "choices", None) or []):
                    last_reason = "completion returned no choices"
                else:
                    return {
                        **identity,
                        "ok": True,
                        "returned_model": last_model,
                        "completion_id": getattr(completion, "id", None),
                        "probe_attempts": attempt,
                        "probe_charged_to_a5_cap": False,
                    }
            except Exception as exc:  # noqa: BLE001
                last_reason = redact("%s: %s" % (type(exc).__name__, exc))
            print("PROBE attempt %d/3 failed: %s" % (attempt, last_reason),
                  flush=True)
            if attempt < 3:
                time.sleep(8)
        return {**identity, "ok": False, "reason": last_reason,
                "returned_model": last_model, "probe_attempts": 3}
    except Exception as exc:  # noqa: BLE001
        return {**identity, "ok": False,
                "reason": redact("%s: %s" % (type(exc).__name__, exc)),
                "returned_model": None}


# =========================================================================== #
# Part 1 -- PowerCons re-earn
# =========================================================================== #
def _source_a_from_disk() -> dict[str, Any]:
    if not PS0_JSON.is_file():
        raise s1.Stop("INSTRUMENT_UNREADABLE",
                      "source A record missing: %s" % PS0_JSON)
    payload = json.loads(PS0_JSON.read_text(encoding="utf-8"))
    scene = next((row for row in payload.get("scenes") or []
                  if row.get("scene") == "source_A_prime"
                  and row.get("outcome") == "EARNED"), None)
    if scene is None or not (scene.get("earned") or {}).get("earned"):
        raise s1.Stop("INSTRUMENT_UNREADABLE",
                      "ps0_srcA_1 GPA Episode is not EARNED on disk")
    earned = dict(scene["earned"])
    earned.setdefault("run_id", "ps0_srcA_1")
    return {
        "scene": "source_A_prime",
        "unit_id": scene["unit_id"],
        "family_key": scene.get("family_key") or "GunPointFamily",
        "outcome": "EARNED",
        "earned": earned,
        "attempts": scene.get("attempts") or 1,
        "max_attempts": scene.get("max_attempts") or 2,
        "runs": scene.get("runs") or [],
        "backend_era": "agicto",
        "note": SOURCE_A_BACKEND_NOTE,
    }


def _bind_book_caps() -> None:
    for module in (ps0, ps1):
        module.LLM_TOTAL_CAP = LLM_TOTAL_CAP
        module.FIT_TOTAL_CAP = FIT_TOTAL_CAP
        module.WALL_SECONDS_CAP = WALL_SECONDS_CAP


def _part1_markdown(payload: Mapping[str, Any]) -> str:
    scene = (payload.get("scenes") or [{}])[-1] if payload.get("scenes") else {}
    lines = [
        "# PS-0c -- re-earn PowerCons source B",
        "",
        "protocol: `%s`  evidence grade: **%s**  git: `%s`  backend: **%s**"
        % (payload["protocol_version"], payload["evidence_grade"],
           payload["git_head"],
           (payload.get("backend_probe") or {}).get("returned_model")),
        "",
        "**%s**" % payload["verdict"]["verdict"],
        "",
        payload["verdict"]["reason"],
        "",
        SOURCE_A_BACKEND_NOTE,
        "",
        "## Protocol",
        "",
        "- unit: `PowerCons__impulse_v2`",
        "- arm: A3-reset, isomorphic to PS-0 / S1c unit protocol",
        "- run-ids: ps0_srcB_3 / ps0_srcB_4 (≤2, stop on first earn)",
        "- no hinting of prompt, budget, or candidate cap",
        "",
        "## Per-run proposal ledger",
        "",
    ]
    for run in scene.get("runs") or []:
        earned = (run.get("earned") or {}).get("earned")
        lines += [
            "#### `%s` -- %s"
            % (run["run_id"], "earned" if earned else "miss"),
            "",
            "| round | candidate id | operators | family | chosen | "
            "outcome | gain |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in run.get("proposal_ledger") or []:
            gain = row.get("gain")
            lines.append("| %s | `%s` | %s | %s | %s | %s | %s |" % (
                row.get("round"), row.get("candidate_id"),
                ", ".join(row.get("operators") or []) or "-",
                row.get("family"), row.get("chosen_by_select"),
                row.get("outcome"),
                "%.4f" % gain if isinstance(gain, (int, float)) else "-"))
        lines += ["",
                  "- families proposed: %s"
                  % (run.get("proposal_families") or "none"),
                  "- target family proposed: %s"
                  % run.get("target_family_proposed"),
                  "- cost: LLM %s, fits %s, %.1f s"
                  % (run.get("llm_calls"), run.get("consumer_fits"),
                     float(run.get("seconds") or 0.0)),
                  ""]
        lines += ["| round | workflow | relation | Support | delayed |",
                  "|---|---|---|---|---|"]
        for record in run.get("rounds") or []:
            episodes = record.get("episodes") or []
            if not episodes:
                lines.append("| %s | - | - | - | - |" % record.get("round"))
            for episode in episodes:
                lines.append("| %s | `%s` | %s | %s | %s |" % (
                    record.get("round"), episode.get("workflow_signature"),
                    episode.get("relation"), episode.get("support_gain"),
                    episode.get("delayed_gain")))
        lines.append("")
    part0 = payload.get("part0_reverify") or {}
    if part0:
        lines += ["## Part 0 re-verification", "",
                  "- verdict: **%s**" % part0.get("verdict"),
                  "- %s" % part0.get("reason"), ""]
        if part0.get("axes"):
            lines += ["| axis | intersection | agree |", "|---|---|---|"]
            for axis, row in part0["axes"].items():
                value = row.get("intersection")
                if axis == "deployment_visible_pattern_intersection":
                    value = row.get("leaves_beyond_task_kind")
                lines.append("| %s | %s | %s |"
                             % (axis, value, row.get("agree")))
            lines.append("")
    ledger = payload.get("ledger") or {}
    lines += ["## Cost (book so far)", "",
              "- LLM: %s / %s" % (ledger.get("llm"), ledger.get("llm_cap")),
              "- Consumer fits: %s / %s"
              % (ledger.get("fit"), ledger.get("fit_cap")),
              "- wall clock: %s s / %s s"
              % (ledger.get("wall_seconds"), ledger.get("wall_seconds_cap")),
              "- downloads: 0", "", "## Obligations", ""]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    if payload.get("outside_book"):
        lines += ["", "## Outside the book", ""]
        lines += ["- %s" % item for item in payload["outside_book"]]
    return "\n".join(lines) + "\n"


def _write_part1(payload: Mapping[str, Any]) -> None:
    clean = redact(s1._plain(payload))
    s1._dump(OUT_JSON, clean)
    OUT_MD.write_text(_part1_markdown(clean), encoding="utf-8")


# =========================================================================== #
# Part 2 -- frozen PS-1, called as functions (protocol unchanged)
# =========================================================================== #
def _preflight_cards(h0: Any, cards: Mapping[str, Any],
                     store_root: Path) -> None:
    """Fail before the 12-run spend if EditController still rejects a card."""
    for name, card in (("neutral", cards["neutral"]),
                       ("scoped", cards["scoped"])):
        s1._apply_entries(h0, [card], store_root=store_root / "preflight",
                          tag=name)


def _run_ps1(*, part0: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]],
             probe: Mapping[str, Any], h0: Any, store_root: Path,
             ledger: dict[str, int], started: float) -> dict[str, Any]:
    scope = part0["scope_v1"]
    sources = [
        {"unit_id": row["unit_id"], "run_id": row["run_id"],
         "support_gain": next(
             scene["earned"]["support_gain"] for scene in scenes
             if scene["scene"] == row["scene"]),
         "delayed_gain": next(
             scene["earned"]["delayed_gain"] for scene in scenes
             if scene["scene"] == row["scene"])}
        for row in part0["per_source"]]
    scoped = ps1._scoped_card(scope, sources)
    neutral = ps1._neutral_card(scope)
    payload: dict[str, Any] = {
        "protocol_version": ps1.PROTOCOL_VERSION,
        "evidence_grade": ps1.EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "exam_unit": ps1.EXAM_UNIT["unit_id"],
        "ps0_source": DUAL_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "part0_reverify": part0,
        "run_plan": [dict(plan) for plan in ps1.RUN_PLAN],
        "arms": {
            ps1.ARM_A3: "no Source Skill",
            ps1.ARM_NEUTRAL: ("same SkillEntry shape, no operator name, no "
                              "Program family, every authority flag false"),
            ps1.ARM_SCOPED: ("same shape, Scope matched, supplies_candidates "
                             "true and grants_execution false"),
        },
        "experimental_prior_slot": True,
        "prior_slot_implementation": (
            "the runner places the SkillEntry on the arm's own snapshot via "
            "the frozen EditController path, without a Slow authorization "
            "audit.  Everything after that is production."),
        "backend_probe": probe,
        "source_A_backend_note": SOURCE_A_BACKEND_NOTE,
        "cards": {"scoped": scoped, "neutral": neutral},
        "card_audit": ps1._card_audit(scoped, neutral),
    }
    payload["runner_fix"] = {
        "what": ("observable_applicability keeps only leaves legal under "
                 "contracts/schemas/observable_feature_v1.json"),
        "why": ("EditController rejects the four Python-contract leaves "
                "absent from that schema; first apply raised "
                "INSTRUMENT_UNREADABLE"),
        "dropped_from_machine_ast": payload["card_audit"][
            "pattern_leaves_dropped_as_uncontracted_for_edit_schema"],
        "methods_untouched": True,
    }
    ps1.CARD_DIR.mkdir(parents=True, exist_ok=True)
    for name, card in (("scoped", scoped), ("neutral", neutral)):
        (ps1.CARD_DIR / ("ps1_card_%s.json" % name)).write_text(
            json.dumps(redact(s1._plain(card)), indent=1,
                       ensure_ascii=False) + "\n",
            encoding="utf-8")
    stopped: str | None = None
    runs: list[dict[str, Any]] = []
    base_shas: dict[str, str] = {}
    try:
        _preflight_cards(h0, payload["cards"], store_root)
        print("PS1 card preflight ok; dropped_uncontracted=%s"
              % payload["runner_fix"]["dropped_from_machine_ast"], flush=True)
        runs, base_shas = ps1._run_arms(
            cards=payload["cards"], h0=h0, store_root=store_root,
            ledger=ledger, started=started)
    except s1.Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {
            "verdict": stopped,
            "reason": redact("%s: %s" % (type(exc).__name__, exc)),
            "traceback": redact(traceback.format_exc()),
        }
    payload["runs"] = runs
    payload["budget_equality"] = ps1._budget_equality(base_shas)
    payload["aggregate"] = ps1._aggregate(runs)
    payload["verdict"] = ps1._verdict(payload["aggregate"], stopped=stopped)
    payload["ledger"] = {
        "llm": ledger["llm"], "llm_cap": LLM_TOTAL_CAP,
        "fit": ledger["fit"], "fit_cap": FIT_TOTAL_CAP,
        "wall_seconds": round(time.time() - started, 1),
        "wall_seconds_cap": WALL_SECONDS_CAP, "downloads": 0,
        "note": "ledger is the book total (Part 1 re-earn + Part 2 arms)",
    }
    payload["oracle_isolation"] = s1._oracle_isolation_report()
    payload["obligations"] = {
        "methods_package_unmodified": True,
        "runtime_contracts_operators_unmodified": True,
        "production_governance_unmodified": True,
        "no_new_skill_class_or_permission_platform": True,
        "card_is_a_plain_skill_entry": True,
        "experimental_prior_slot": True,
        "budgets_equal_across_arms": payload["budget_equality"]["all_equal"],
        "neither_card_supplies_a_frozen_program": payload["card_audit"][
            "neither_card_supplies_a_frozen_program"],
        "guided_positive_counts_zero_toward_cross_domain_authorization": True,
        "pilot_grade_freezes_no_production_design": True,
        "gray_zone_appends_no_batch": True,
        "arms_run": len(runs),
        "downloads": 0,
        "oracle_isolation_holds": payload["oracle_isolation"]["holds"],
        "stage_report_not_written": True,
        "full_repo_pytest_not_run": True,
        "new_relay_only_no_agicto_fallback": True,
        "secret_key_not_written": True,
    }
    payload["outside_book"] = [
        "runner-level fix: machine applicability drops leaves that "
        "contracts/observables.py lists but observable_feature_v1.json "
        "does not; body and scope_v1 still carry the full intersection",
    ]
    clean = redact(s1._plain(payload))
    s1._dump(ps1.OUT_JSON, clean)
    ps1.OUT_MD.write_text(ps1._markdown(clean), encoding="utf-8")
    return clean


def run_ps1_only() -> int:
    """Resume Part 2 from the already-earned dual-source record."""
    if not DUAL_JSON.is_file() or not OUT_JSON.is_file():
        raise SystemExit("Part 1 artifacts missing; run --run first")
    prior = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    dual = json.loads(DUAL_JSON.read_text(encoding="utf-8"))
    part0 = dual.get("part0_reverify") or prior.get("part0_reverify") or {}
    if not part0.get("pass"):
        raise SystemExit("Part 0 did not pass; refuse to compile a card")
    prior_ledger = prior.get("ledger") or {}
    prior_wall = float(prior_ledger.get("wall_seconds") or 0.0)
    started = time.time() - prior_wall
    s1._set_phase(s1.PHASE_SETUP)
    _bind_book_caps()
    ledger = {
        "llm": int(prior_ledger.get("llm") or 0),
        "fit": int(prior_ledger.get("fit") or 0),
    }
    install = install_new_backend()
    probe = probe_new_backend()
    print("PROBE ok=%s returned_model=%s host=%s (ps1-only resume, "
          "ledger llm=%s fit=%s wall_prior=%.1fs)"
          % (probe.get("ok"), probe.get("returned_model"),
             probe.get("expected_host"), ledger["llm"], ledger["fit"],
             prior_wall), flush=True)
    if not probe.get("ok"):
        prior["ps1_verdict"] = {"verdict": "BACKEND_UNAVAILABLE",
                                "reason": probe.get("reason")}
        _write_part1(prior)
        return 1
    store_ps1 = Path(tempfile.gettempdir()) / "ps1_arms"
    if store_ps1.exists():
        shutil.rmtree(store_ps1)
    k0 = s1.compile_k0(store_ps1 / "k0")
    ps1_payload = _run_ps1(
        part0=part0, scenes=dual.get("scenes") or prior.get("scenes") or [],
        probe=probe, h0=k0["h0"], store_root=store_ps1,
        ledger=ledger, started=started)
    prior["backend_probe"] = probe
    prior["backend_install"] = {
        "host": install["host"], "requested_model": install["model"],
        "resume": "--ps1-only"}
    prior["ledger"] = ps1_payload.get("ledger") or prior.get("ledger")
    prior["ps1_verdict"] = ps1_payload.get("verdict")
    prior["outside_book"] = list(prior.get("outside_book") or []) + [
        "PS-1 resumed via --ps1-only after a runner-level applicability "
        "shape fix; PowerCons was not re-earned"]
    _write_part1(prior)
    print(json.dumps({
        "verdict": prior.get("verdict", {}).get("verdict"),
        "ps1_verdict": (ps1_payload.get("verdict") or {}).get("verdict"),
        "ps1_ran": True,
        "llm": ledger["llm"], "fits": ledger["fit"],
        "seconds": (ps1_payload.get("ledger") or {}).get("wall_seconds"),
        "returned_model": probe.get("returned_model"),
        "artifact": str(ps1.OUT_JSON),
    }, ensure_ascii=False, indent=1), flush=True)
    return 0


def _write_ps1_halt(*, part0: Mapping[str, Any],
                    scenes: Sequence[Mapping[str, Any]],
                    probe: Mapping[str, Any],
                    ledger: Mapping[str, Any],
                    started: float,
                    verdict: Mapping[str, Any]) -> None:
    combined = {
        "scenes": list(scenes),
        "part0_reverify": part0,
    }
    payload = {
        "protocol_version": ps1.PROTOCOL_VERSION,
        "evidence_grade": ps1.EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "exam_unit": ps1.EXAM_UNIT["unit_id"],
        "ps0_source": OUT_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "part0_reverify": part0,
        "backend_probe": probe,
        "source_A_backend_note": SOURCE_A_BACKEND_NOTE,
        "verdict": dict(verdict),
        "miss_analysis": ps1.miss_analysis(combined),
        "ledger": {
            "llm": ledger["llm"], "llm_cap": LLM_TOTAL_CAP,
            "fit": ledger["fit"], "fit_cap": FIT_TOTAL_CAP,
            "wall_seconds": round(time.time() - started, 1),
            "wall_seconds_cap": WALL_SECONDS_CAP, "downloads": 0,
        },
        "obligations": {
            "arms_run": 0,
            "cards_compiled": False,
            "why_no_card": (
                "Part 2 is conditional on both sources being live Episodes "
                "with a usable five-axis intersection"),
            "llm_calls": ledger["llm"],
            "methods_package_unmodified": True,
            "production_governance_unmodified": True,
            "new_relay_only_no_agicto_fallback": True,
            "secret_key_not_written": True,
            "stage_report_not_written": True,
        },
    }
    clean = redact(s1._plain(payload))
    s1._dump(ps1.OUT_JSON, clean)
    ps1.OUT_MD.write_text(ps1._halt_markdown(clean), encoding="utf-8")


# =========================================================================== #
# driver
# =========================================================================== #
def run(*, probe_only: bool = False) -> int:
    started = time.time()
    s1._set_phase(s1.PHASE_SETUP)
    _bind_book_caps()
    ledger = {"llm": 0, "fit": 0}
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "entry": "--probe-only" if probe_only else "--run",
        "source_A_backend_note": SOURCE_A_BACKEND_NOTE,
        "protocol": {
            "arm": ps0.ARM,
            "rounds": list(ps0.ROUNDS),
            "llm_per_run": ps0.LLM_PER_RUN,
            "fit_per_run": ps0.FIT_PER_RUN,
            "max_runs_per_scene": ps0.MAX_RUNS_PER_SCENE,
            "run_ids": list(SOURCE_B_SCENE["run_ids"]),
            "isomorphic_to": "the PS-0 / S1c unit protocol; no hinting",
            "stop_rule": "first earn stops; two misses -> "
                         "PS1_SOURCES_NOT_REEARNED_FINAL",
        },
        "book_caps": {
            "llm": LLM_TOTAL_CAP, "fit": FIT_TOTAL_CAP,
            "wall_seconds": WALL_SECONDS_CAP,
        },
    }
    scenes: list[dict[str, Any]] = []
    stopped: str | None = None
    try:
        install = install_new_backend()
        payload["backend_install"] = {
            "host": install["host"],
            "requested_model": install["model"],
            "patched_modules": (
                "normal_flow.NF_BASE_URL/NF_MODEL, e1.NF_BASE_URL/NF_MODEL, "
                "agentic.runner.NF_BASE_URL/NF_MODEL, cls.SLOW_MODEL"),
            "methods_runtime_untouched": True,
        }
        payload["backend_probe"] = probe_new_backend()
        print("PROBE ok=%s returned_model=%s host=%s"
              % (payload["backend_probe"].get("ok"),
                 payload["backend_probe"].get("returned_model"),
                 payload["backend_probe"].get("expected_host")),
              flush=True)
        if not payload["backend_probe"].get("ok"):
            raise s1.Stop(
                "BACKEND_UNAVAILABLE",
                payload["backend_probe"].get("reason") or "probe failed")
        if probe_only:
            payload["verdict"] = {
                "verdict": "BACKEND_OK",
                "reason": "new relay answered; returned_model=%s"
                          % payload["backend_probe"].get("returned_model")}
            payload["scenes"] = []
            payload["part0_reverify"] = {}
            payload["ledger"] = {
                "llm": 0, "llm_cap": LLM_TOTAL_CAP,
                "fit": 0, "fit_cap": FIT_TOTAL_CAP,
                "wall_seconds": round(time.time() - started, 1),
                "wall_seconds_cap": WALL_SECONDS_CAP, "downloads": 0,
            }
            payload["obligations"] = {
                "probe_only": True,
                "new_relay_only_no_agicto_fallback": True,
                "secret_key_not_written": True,
            }
            _write_part1(payload)
            print(json.dumps({
                "verdict": "BACKEND_OK",
                "returned_model": payload["backend_probe"].get(
                    "returned_model"),
                "host": payload["backend_probe"].get("expected_host"),
            }, ensure_ascii=False, indent=1), flush=True)
            return 0

        source_a = _source_a_from_disk()
        store_root = Path(tempfile.gettempdir()) / "ps0c_reearn"
        if store_root.exists():
            shutil.rmtree(store_root)
        k0 = s1.compile_k0(store_root / "k0")
        payload["h0_runtime_bundle_sha"] = k0["h0_sha"]
        source_b = ps0._run_scene(
            SOURCE_B_SCENE, store_root=store_root, h0=k0["h0"],
            ledger=ledger, started=started)
        scenes = [source_a, source_b]
        payload["scenes"] = scenes
        part0 = ps0.part0_reverify(scenes)
        if source_b["outcome"] != "EARNED":
            part0 = {
                **part0,
                "pass": False,
                "verdict": "PS1_SOURCES_NOT_REEARNED_FINAL",
                "reason": (
                    "PowerCons was not re-earned in ≤2 isomorphic A3-reset "
                    "attempts (ps0_srcB_3/4).  The hampel dual-source "
                    "structure dies at the live Episode layer.  Slice-"
                    "protocol questions are escalated to arbitration."),
            }
        payload["part0_reverify"] = part0
        dual = {
            "protocol_version": PROTOCOL_VERSION,
            "source_A_backend_note": SOURCE_A_BACKEND_NOTE,
            "scenes": scenes,
            "part0_reverify": part0,
            "backend_probe": payload["backend_probe"],
        }
        s1._dump(DUAL_JSON, redact(s1._plain(dual)))

        if part0.get("pass"):
            payload["verdict"] = {
                "verdict": "SOURCES_REEARNED_SCOPE_USABLE",
                "reason": part0["reason"]}
        else:
            payload["verdict"] = {
                "verdict": part0.get("verdict") or "PS1_SOURCES_NOT_REEARNED_FINAL",
                "reason": part0.get("reason") or "Part 0 did not pass"}
    except s1.Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
        payload["verdict"] = {"verdict": stop.verdict, "reason": stop.reason}
        payload["scenes"] = scenes
        payload["part0_reverify"] = payload.get("part0_reverify") or {}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {
            "verdict": stopped,
            "reason": redact("%s: %s" % (type(exc).__name__, exc)),
            "traceback": redact(traceback.format_exc()),
        }
        payload["verdict"] = {"verdict": stopped,
                              "reason": payload["stop"]["reason"]}
        payload["scenes"] = scenes
        payload["part0_reverify"] = payload.get("part0_reverify") or {}

    payload["ledger"] = {
        "llm": ledger["llm"], "llm_cap": LLM_TOTAL_CAP,
        "fit": ledger["fit"], "fit_cap": FIT_TOTAL_CAP,
        "wall_seconds": round(time.time() - started, 1),
        "wall_seconds_cap": WALL_SECONDS_CAP, "downloads": 0,
    }
    payload["oracle_isolation"] = s1._oracle_isolation_report()
    payload["obligations"] = {
        "methods_package_unmodified": True,
        "runtime_contracts_operators_unmodified": True,
        "production_governance_unmodified": True,
        "protocol_isomorphic_to_ps0_s1c_unit": True,
        "no_hinting_of_prompt_budget_or_candidate_cap": True,
        "scenes_stopped_on_first_earn": True,
        "live_backend": (payload.get("backend_probe") or {}).get(
            "returned_model"),
        "new_relay_only_no_agicto_fallback": True,
        "secret_key_not_written": True,
        "downloads": 0,
        "sealed_artifacts_not_read": True,
        "oracle_isolation_holds": payload["oracle_isolation"]["holds"],
        "stage_report_not_written": True,
        "full_repo_pytest_not_run": True,
    }
    payload["outside_book"] = []
    _write_part1(payload)

    part0 = payload.get("part0_reverify") or {}
    ran_ps1 = False
    ps1_verdict = None
    if (not probe_only
            and not stopped
            and part0.get("pass")
            and payload["verdict"]["verdict"]
            == "SOURCES_REEARNED_SCOPE_USABLE"):
        print("PART1 earned; entering frozen PS-1", flush=True)
        store_ps1 = Path(tempfile.gettempdir()) / "ps1_arms"
        if store_ps1.exists():
            shutil.rmtree(store_ps1)
        k0 = s1.compile_k0(store_ps1 / "k0")
        ps1_payload = _run_ps1(
            part0=part0, scenes=payload["scenes"],
            probe=payload["backend_probe"], h0=k0["h0"],
            store_root=store_ps1, ledger=ledger, started=started)
        ran_ps1 = True
        ps1_verdict = (ps1_payload.get("verdict") or {}).get("verdict")
        payload["ledger"] = ps1_payload.get("ledger") or payload["ledger"]
        payload["ps1_verdict"] = ps1_payload.get("verdict")
        _write_part1(payload)
    elif not probe_only:
        _write_ps1_halt(
            part0=part0, scenes=payload.get("scenes") or [],
            probe=payload.get("backend_probe") or {},
            ledger=ledger, started=started,
            verdict=payload["verdict"])

    print(json.dumps({
        "verdict": payload["verdict"]["verdict"],
        "ps1_verdict": ps1_verdict,
        "ps1_ran": ran_ps1,
        "source_B": (payload.get("scenes") or [{}])[-1].get("outcome")
        if payload.get("scenes") else None,
        "llm": ledger["llm"], "fits": ledger["fit"],
        "seconds": payload["ledger"]["wall_seconds"],
        "returned_model": (payload.get("backend_probe") or {}).get(
            "returned_model"),
        "artifact": str(OUT_JSON),
    }, ensure_ascii=False, indent=1), flush=True)
    ok = payload["verdict"]["verdict"] in {
        "SOURCES_REEARNED_SCOPE_USABLE", "BACKEND_OK"}
    if ran_ps1:
        return 0
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--ps1-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.ps1_only:
        return run_ps1_only()
    if args.run or args.probe_only:
        return run(probe_only=bool(args.probe_only and not args.run))
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
