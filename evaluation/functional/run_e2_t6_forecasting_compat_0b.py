"""#42d Part 0b -- 0-LLM forecasting compatibility close.

Replay the six cached forecasting positives through the live
``handle_fast_winner`` -> ``handle_feedback_delayed`` path after the two
hand-written Skill ids were redirected to ``fast_winner_skill_id``.
Reads only already-exposed forecasting artifacts.  Spends no LLM call,
no AD fit and no forecasting retrain.  Does not run the full
``--fresh-confirmation`` exam.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
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

import run_e2_fresh_confirmation as fc  # noqa: E402
import run_e2_local_skill_recall as lsr  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    episode_from_dict,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore,
)
from SelfEvolvingHarnessTS.methods.ttha.method import (  # noqa: E402
    TTHAMethod,
    fast_winner_skill_id,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t6_nab_42d_part0b_forecasting_compat.json"
H0_ROOT = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _retrieve(snapshot: Any, skill_id: str) -> dict[str, Any]:
    view = resolve_harness_view(
        snapshot, {"task_kind": "forecast"}, role="fast")
    hit = any(skill.skill_id == skill_id for skill in view.skills)
    stored = next((s for s in snapshot.skills if s.skill_id == skill_id), None)
    return {
        "hit": bool(hit),
        "in_snapshot": stored is not None,
        "resolved_skill_ids": list(view.skill_ids),
        "body_prefix": (None if stored is None
                        else str(stored.body)[:80]),
        "frozen_marker": (
            None if stored is None
            else "Frozen program steps:" in str(stored.body)
        ),
    }


def _replay_lsr(work: Path) -> list[dict[str, Any]]:
    integration, probe = lsr._load_inputs()
    rows: list[dict[str, Any]] = []
    for (target_id, arm), draft_id in lsr.LOCAL_SOURCE.items():
        source = lsr._probe_source(probe, draft_id)
        plan = lsr.LOCAL_PLAN[(target_id, arm)]
        slot = "%s_%s" % (arm.lower(), target_id.lower())
        root = work / "lsr" / slot / "snapshots"
        store = SnapshotStore(root)
        snapshot = compile_snapshot(H0_ROOT, verify_lock=False)
        store.materialize(snapshot)
        store.set_active(snapshot.runtime_bundle_sha)
        error = None
        local: dict[str, Any] = {}
        try:
            snapshot, local = lsr._persist_local_skill(
                store=store, snapshot=snapshot, plan=plan,
                source=source, slot=slot,
            )
        except Exception as exc:  # noqa: BLE001
            error = "%s: %s" % (type(exc).__name__, exc)
        episode = episode_from_dict(
            dict(source["transition"]["updated_episode"]))
        # persist rewrites the signature to the e1v2 spelling before
        # naming the Skill; reconstruct that same rewrite for the id check
        from evaluation.functional.task_episode_harness import e1 as e1mod
        import dataclasses
        steps = ((str(plan["program"]), {}),)
        named = dataclasses.replace(
            episode,
            workflow_signature=e1mod._v2_workflow_signature(steps),
        )
        expected = fast_winner_skill_id(named)
        retrieval = (_retrieve(snapshot, expected)
                     if error is None else {"hit": False})
        cached = (lsr.json.loads if False else None)
        cached_id = None
        # cached historical id from the already-written artifact
        artifact = json.loads(
            (E2 / "local_skill_recall_v1.json").read_text(encoding="utf-8"))
        cached_skill = ((artifact.get("stores") or {}).get(slot) or {}).get(
            "local_skill") or {}
        cached_id = cached_skill.get("skill_id")
        ok = (
            error is None
            and (local.get("handle_fast_winner") or {}).get("stage") == "pending"
            and (local.get("cached_delayed_approval") or {}).get("stage")
            == "approved"
            and local.get("skill_id") == expected
            and retrieval.get("hit")
            and (local.get("lifecycle") or {}).get("local_status")
            == "LOCAL_ACTIVE"
        )
        rows.append({
            "source_runner": "local_skill_recall",
            "slot": slot,
            "draft_id": draft_id,
            "program": plan["program"],
            "cached_support_gain": float(
                source["transition"]["updated_episode"]
                ["support_response"]["gain"]),
            "cached_delayed_gain": float(
                source["transition"]["updated_episode"]
                ["delayed_response"]["gain"]),
            "old_cached_skill_id": cached_id,
            "new_skill_id": local.get("skill_id") or expected,
            "expected_skill_id": expected,
            "handle_fast_winner": local.get("handle_fast_winner"),
            "delayed_event": local.get("cached_delayed_approval"),
            "retrieval": retrieval,
            "error": error,
            "ok": ok,
        })
    return rows


def _replay_fc(work: Path) -> list[dict[str, Any]]:
    artifact = json.loads(
        (E2 / "fresh_confirmation_v1.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    cells = (artifact.get("stage_2_adaptation") or {}).get("cells") or {}
    for slot_key, cell in cells.items():
        draft_cached = cell.get("draft") or {}
        promo_cached = cell.get("promotion") or {}
        if not draft_cached.get("written"):
            continue
        record = dict(cell["task_A"])
        variant = str(cell["consumer_variant"])
        arm = str(cell["arm"])
        target = fc._target(variant)
        root = work / "fc" / slot_key / "snapshots"
        store = SnapshotStore(root)
        snapshot = compile_snapshot(H0_ROOT, verify_lock=False)
        store.materialize(snapshot)
        store.set_active(snapshot.runtime_bundle_sha)
        slot: dict[str, Any] = {
            "slot": slot_key,
            "_store": store,
            "_snapshot": snapshot,
        }
        error = None
        draft: dict[str, Any] = {}
        promotion: dict[str, Any] = {}
        try:
            draft = fc._persist_draft(
                slot=slot, record=record, target=target, arm=arm)
            probe = {
                "origins": list(
                    (promo_cached.get("lifecycle_fields") or {}).get(
                        "activation_probe_origins")
                    or (promo_cached.get("episode_after") or {}).get(
                        "delayed_block_origins")
                    or []),
                "macro_gain": float(promo_cached["probe_gain"]),
                "se_block": float(
                    (promo_cached.get("lifecycle_fields") or {}).get(
                        "activation_probe_se_block") or 0.0),
                "gain_over_se": (
                    (promo_cached.get("lifecycle_fields") or {}).get(
                        "activation_probe_gain_over_se")),
            }
            promotion = fc._promote(
                slot=slot, probe=probe, draft=draft)
        except Exception as exc:  # noqa: BLE001
            error = "%s: %s" % (type(exc).__name__, exc)
        expected = draft.get("skill_id")
        snapshot = slot.get("_snapshot") or snapshot
        retrieval = (_retrieve(snapshot, expected)
                     if expected and error is None
                     else {"hit": False})
        ok = (
            error is None
            and draft.get("written")
            and (draft.get("handle_fast_winner") or {}).get("stage")
            == "pending"
            and promotion.get("promoted")
            and promotion.get("store_approved")
            and promotion.get("retrievable_skill_id") == expected
            and retrieval.get("hit")
        )
        rows.append({
            "source_runner": "fresh_confirmation",
            "slot": slot_key,
            "program": (record.get("final_plan") or {}).get("program"),
            "cached_support_gain": (record.get("support") or {}).get(
                "aggregate_gain"),
            "cached_delayed_gain": promo_cached.get("probe_gain"),
            "old_cached_skill_id": draft_cached.get("skill_id"),
            "new_skill_id": draft.get("skill_id"),
            "expected_skill_id": expected,
            "handle_fast_winner": draft.get("handle_fast_winner"),
            "delayed_event": promotion.get("store_event"),
            "retrieval": retrieval,
            "promoted": promotion.get("promoted"),
            "error": error,
            "ok": ok,
        })
    return rows


def leftover_id_spellings() -> list[dict[str, str]]:
    """Every remaining concatenation, after the two authorized fixes."""
    needles = (
        'fast_winner_%s',
        'f"fast_winner_{',
        "f'fast_winner_{",
        '"fast_winner_" +',
        "'fast_winner_' +",
    )
    rows: list[dict[str, str]] = []
    for root in (PROJECT_ROOT / "evaluation", PROJECT_ROOT / "methods",
                 PROJECT_ROOT / "tests"):
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            for index, line in enumerate(text.splitlines(), 1):
                if any(n in line for n in needles):
                    rows.append({
                        "path": rel, "line": str(index),
                        "text": line.strip()[:180],
                    })
    return rows


def run() -> int:
    leftover = leftover_id_spellings()
    work = Path(tempfile.mkdtemp(prefix="t6_42d_0b_"))
    try:
        lsr_rows = _replay_lsr(work)
        fc_rows = _replay_fc(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rows = lsr_rows + fc_rows
    all_ok = bool(rows) and all(row["ok"] for row in rows)
    payload = {
        "protocol_version": "t6_nab_42d_part0b_v1",
        "entry": "evaluation/functional/run_e2_t6_forecasting_compat_0b.py",
        "note": (
            "0-LLM replay of cached forecasting positives through the live "
            "Draft -> delayed path after the two handwritten Skill ids were "
            "redirected to fast_winner_skill_id.  Not a fresh confirmation."
        ),
        "positives_replayed": len(rows),
        "expected_count_note": (
            "sol counted 6; actual is the three #11 LOCAL_ACTIVE persist "
            "slots plus the three fresh-confirmation drafts that were written"
        ),
        "rows": rows,
        "leftover_fast_winner_concatenations": leftover,
        "leftover_note": (
            "this round only repaired run_e2_local_skill_recall.py:411 and "
            "run_e2_fresh_confirmation.py:1866; leftovers are logged, not "
            "edited"
        ),
        "cost": {"llm": 0, "ad_fits": 0, "forecast_retrains": 0},
        "verdict": {
            "verdict": ("FORECASTING_COMPAT_RESTORED" if all_ok
                        else "FORECASTING_COMPAT_BROKEN"),
            "reason": (
                "all %d cached positives still Draft -> approved -> ACTIVE "
                "under the live dual gate, and the new task-scoped id is "
                "retrievable" % len(rows)
                if all_ok else
                "first unmet replay: %s" % [
                    row["slot"] for row in rows if not row["ok"]]
            ),
        },
    }
    OUT_JSON.write_text(
        json.dumps(_plain(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=1))
    print("wrote", OUT_JSON, flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
