"""Open the sealed confirmation domain: NOAA fresh cohort, 2024 -> 2025.

This is the one-shot slice.  The frozen version has a single opening; every
hard gate stops the run rather than adapting, and once a 2025 Outcome is read
the instrument may not be repaired and re-run.

Stage 0 asks whether the Judge is readable at all on this cohort (development
region, identity baselines only).  Stage 1 compiles one merged Guidance card
per Consumer from every frozen recipe evidence row and registers it into an
A5 store.  Stage 1.5 checks that the roster's 2025 files exist without
downloading them.  Stage 2 runs the adaptation period inside 2024: a
full-price episode, an out-of-selection promotion probe, and a next-task
recall.  Stage 3 acquires and materializes 2025.  Stage 4 reads the
confirmation task on held-out 2025 with no new Skill formation.

Nothing in ``methods/ttha``, ``evaluation/functional/run_batch_composition_
headroom.py``, the recipe compiler, the prompt templates or the adoption rule
is modified.  Store state lives under ``_scratch/skill_store``; the 2025
extension lives under ``data/benchmark_noaa_fresh_v1``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
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
import run_batch_composition_headroom as bch  # noqa: E402
import run_e2_recipe_experience_to_skill as bridge  # noqa: E402
import run_e2_skill_store_integration as ssi  # noqa: E402
import run_e2_warm_vs_cold_recipe_search as wvc  # noqa: E402
from run_v1_kdd2018_natural_slow_update import _config  # noqa: E402

from evaluation.functional.task_episode_harness import e1 as e1mod  # noqa: E402
from evaluation.functional.task_episode_harness.runner import (  # noqa: E402
    _arm_metrics,
    _mapped_roster,
)
from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    EditManifest,
    EditOperation,
    load_learned_skill_entry,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    EVIDENCE_SUPPORT,
    RELATION_ABSTAIN,
    RELATION_CONFLICT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_LOCAL_ACTIVE,
    STATUS_LOCAL_DRAFT,
    build_episode,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    _parse_frozen_steps,
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
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (  # noqa: E402
    _resolve_apply_manifest,
)

PROTOCOL_VERSION = "fresh_confirmation_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "fresh_confirmation_v1.json"
OUT_MD = E2 / "fresh_confirmation_v1.md"
COHORT_ARTIFACT = E2 / "noaa_fresh_cohort_v2.json"
DATA_DIR = PROJECT_ROOT / "data" / "benchmark_noaa_fresh_v1"
SERIES_DIR = DATA_DIR / "series"
CONFIRMATION_DIR = DATA_DIR / "confirmation_2025"
# The acquisition lands inside the fresh cohort's own namespace rather than
# in data/benchmark_v0/raw/noaa_global_hourly/2025, which belongs to the
# retired NOAA line: this protocol writes only where it was authorized to.
RAW_2025 = DATA_DIR / "raw_2025"
STORE_ROOT = PROJECT_ROOT / "_scratch" / "skill_store" / PROTOCOL_VERSION
H0_ROOT = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"
NAMESPACE = PROTOCOL_VERSION

COHORT_NAME = "noaa_fresh"
GRID_START = datetime(2024, 1, 1)
DEVELOPMENT_HOURS = 8760
CONFIRMATION_END = 17520
NOAA_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/2025/%s.csv"
TMP_MISSING = 9999

HORIZON = int(bch.v6.HORIZON)
CONTEXT_LENGTH = int(bch.v6.CONTEXT_LENGTH)
IDENTITY = bridge.IDENTITY
TREATMENTS = bridge.TREATMENTS
MATERIAL_THRESHOLD = float(bch.MATERIAL_THRESHOLD)
HARM_THRESHOLD = -MATERIAL_THRESHOLD
TIE_COST_WINDOW = 15
LLM_CALL_BUDGET_TOTAL = 40
LLM_CALL_BUDGET_PER_EPISODE = 5

TASK_A_S = 1104
PROBE_ORIGINS = (1440, 1488, 1536)
TASK_B_S = 1800
TASK_C_S = 9864
TASK_D_S = 10560
CONSUMERS = (bch.CONSUMER_POOLED, bch.CONSUMER_PER_CHANNEL)
ARMS = ("A5", "A3")


# --------------------------------------------------------- the frozen surface
# Hashed before the first read and again after the last write.  A drift here
# means somebody else wrote the instrument mid-run and the reading is void.
FROZEN_SURFACE: tuple[str, ...] = (
    "artifacts/functional/e2/noaa_fresh_cohort_v2.json",
    "artifacts/functional/e2/recipe_skill_cards_v1.json",
    "artifacts/functional/e2/batch_recipe_windows_v1.json",
    "artifacts/functional/e2/batch_recipe_v2_all_cells_v1.json",
    "artifacts/functional/e2/batch_recipe_T233_v1.json",
    "artifacts/functional/e2/batch_recipe_electricity_v1.json",
    "artifacts/functional/e2/batch_recipe_traffic_v1.json",
    "evaluation/functional/run_batch_composition_headroom.py",
    "evaluation/functional/run_e2_recipe_experience_to_skill.py",
    "evaluation/functional/run_e2_skill_store_integration.py",
    "evaluation/functional/run_e2_warm_vs_cold_recipe_search.py",
    "evaluation/functional/run_e2_autonomous_natural_workflow_generation.py",
    "evaluation/functional/run_v1_kdd2018_natural_slow_update.py",
    "evaluation/functional/task_episode_harness/e1.py",
    "evaluation/functional/task_episode_harness/runner.py",
    "methods/ttha/method.py",
    "methods/ttha/retrieval.py",
    "methods/ttha/harness/h0/snapshot.lock.json",
    "data/benchmark_noaa_fresh_v1/manifest.json",
)


class ConcurrentWrite(RuntimeError):
    """A frozen file moved while the run was in flight."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze() -> dict[str, str]:
    table: dict[str, str] = {}
    for rel in FROZEN_SURFACE:
        path = PROJECT_ROOT / rel
        if not path.is_file():
            raise SystemExit("frozen surface file is missing: %s" % rel)
        table[rel] = _sha256(path)
    return table


def _verify(before: Mapping[str, str]) -> dict[str, Any]:
    drift = []
    for rel, sha in before.items():
        path = PROJECT_ROOT / rel
        now = _sha256(path) if path.is_file() else None
        if now != sha:
            drift.append({"path": rel, "before": sha, "after": now})
    return {"files": len(before), "drift": drift, "ok": not drift}


def _guard(before: Mapping[str, str], where: str) -> dict[str, Any]:
    row = _verify(before)
    row["checked_at"] = where
    if not row["ok"]:
        raise ConcurrentWrite(
            "frozen surface moved before %s: %s"
            % (where, [item["path"] for item in row["drift"]])
        )
    return row


def _repo_rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


# ----------------------------------------------------------------- the windows
def _window(start: int, label: str, *, note: str = "") -> dict[str, Any]:
    """The e1v2 triple-window syntax, applied to an explicit start.

    ``e1._frozen_task_roster`` builds every task the same way: three support
    origins at s, s+48, s+96 and three delayed origins at s+144, s+192, s+240,
    each read HORIZON ahead.  Verified against the roster at import time.
    """
    support = (start, start + HORIZON, start + 2 * HORIZON)
    delayed = (start + 3 * HORIZON, start + 4 * HORIZON, start + 5 * HORIZON)
    return {
        "window_id": label,
        "start": int(start),
        "support_origins": [int(o) for o in support],
        "delayed_origins": [int(o) for o in delayed],
        "horizon": HORIZON,
        "farthest_index": int(max(delayed) + HORIZON),
        "syntax": "support s/s+48/s+96, delayed s+144/s+192/s+240, horizon 48",
        "note": note,
        # #10's runner records but never exposes these two.  This slice spends
        # no hidden full-menu scan, so identity is the neutral placeholder and
        # both fields are stripped from the delivered rows.
        "reference_plan": {"program": IDENTITY, "excluded_series": []},
        "reference_delayed_aggregate_gain": 0.0,
    }


def _assert_window_syntax() -> dict[str, Any]:
    """The triple-window syntax is read off the frozen roster, not asserted."""
    spec = dict(e1mod._frozen_task_roster()[3])
    support = [int(o) for o in spec["support_origins"]]
    delayed = [int(o) for o in spec["delayed_origins"]]
    start = support[0]
    expected = _window(start, str(spec["task_episode_id"]))
    if (
        expected["support_origins"] != support
        or expected["delayed_origins"] != delayed
        or int(spec["horizon"]) != HORIZON
    ):
        raise SystemExit(
            "the triple-window syntax does not reproduce the frozen roster"
        )
    return {
        "checked_against": "e1._frozen_task_roster()[3]",
        "task_episode_id": str(spec["task_episode_id"]),
        "support_origins": support,
        "delayed_origins": delayed,
        "horizon": int(spec["horizon"]),
        "reproduced": True,
    }


WINDOWS = {
    "task_A": _window(TASK_A_S, "fresh_task_A", note="adaptation, development"),
    "task_B": _window(TASK_B_S, "fresh_task_B", note="next task, development"),
    "task_C": _window(TASK_C_S, "fresh_task_C", note="confirmation, 2025"),
    "task_D": _window(TASK_D_S, "fresh_task_D", note="the one backup, 2025"),
}
PROBE_WINDOW = {
    "window_id": "fresh_probe",
    "origins": [int(o) for o in PROBE_ORIGINS],
    "horizon": HORIZON,
    "farthest_index": int(max(PROBE_ORIGINS) + HORIZON),
    "role": (
        "out-of-selection promotion probe; its reading takes part in no "
        "proposal and no adoption on task_A"
    ),
}


# ------------------------------------------------------ the roster amendment
# Ruled before any Outcome was opened, with no measured value taking part --
# the same class of correction as the fresh-pool floor 24 -> 20 on record in
# noaa_fresh_cohort_v2.  Written here verbatim and reproduced in the artifact.
ROSTER_SPLIT_AMENDMENT: dict[str, Any] = {
    "when": (
        "before stage 0, before any Consumer was fitted on this cohort and "
        "before one 2025 index was read"
    ),
    "zero_measured_value_took_part": True,
    "same_class_as": "the fresh-pool floor correction 24 -> 20 in noaa_fresh_cohort_v2",
    "problem": (
        "the frozen cohort shape is 12 train + 8 eval = 20 series "
        "(v6.TRAIN_SERIES_COUNT / EVAL_SERIES_COUNT); the locked roster from "
        "noaa_fresh_cohort_v2 holds 12 stations, which cannot be split 12 + 8, "
        "and that artifact declares no split"
    ),
    "ruling": "12 train + 4 eval = all 16 PASS stations",
    "train": "the 12 roster stations of noaa_fresh_cohort_v2, verbatim",
    "eval": "the 4 substitute stations of noaa_fresh_cohort_v2, verbatim",
    "why": [
        "the training batch of 12 is the calibrating geometry of every source "
        "evidence row -- the Guidance card, the mask geometry and the 0.005 "
        "material and -0.005 harm lines were all read off a 12-series batch, "
        "so it is a load-bearing dimension for comparability and is not scaled",
        "cutting eval from 8 to 4 adds symmetric noise only; the variance lift "
        "is disclosed rather than hidden",
        "per-station eval share 1/4 = 0.25 and eval fraction 4/16 = 0.25 both "
        "sit inside [0.20, 0.40]",
    ],
    "instrument_check": (
        "TRAIN_SERIES_COUNT / EVAL_SERIES_COUNT appear only in cohort "
        "construction (v6._fixed_roster, run_w2_focus_recheck). The evaluation "
        "path -- bch._evaluate_assignment, bch._evaluate_variant, "
        "bch._gain_rows, wvc.BudgetedSearch -- reads train/eval off the roster "
        "rows and takes eval_uids as a parameter, so 12 + 4 is supplied, not "
        "patched in. No instrument line was changed."
    ),
    "cascade_pre_registered": (
        "a train station with no 2025 file is replaced, in lexicographic "
        "order, by promoting an eval station, and eval shrinks accordingly; an "
        "eval station with no 2025 file simply shrinks eval; eval below 3 "
        "stops the run at INSUFFICIENT_2025_COVERAGE"
    ),
    "min_eval_after_cascade": 3,
    "declared_2025_reads_at_ruling_time": 0,
}

PRE_REGISTERED: dict[str, Any] = {
    "fixed_before_the_first_read": True,
    "one_shot": (
        "the frozen version has one opening. Any hard gate failing stops the "
        "run; while no 2025 Outcome has been opened the version is still "
        "fresh and may be re-run after a repair. Once any 2025 Outcome is "
        "open, the instrument may not be repaired and re-run -- only a "
        "0-evaluation deterministic replay of the readings already opened."
    ),
    "frozen_surface": list(FROZEN_SURFACE),
    "concurrent_write": "a sha256 drift on the frozen surface aborts the run",
    "slow_agent": "off for the whole run; no Slow proposal is requested",
    "stage_0": (
        "identity baselines for both Consumers on task_A (12 Consumer "
        "retrains, 0 LLM). The metric must be finite, non-degenerate and not "
        "dominated by a near-zero denominator, judged by the criteria frozen "
        "in noaa_health_check_v1: eval-loss spread <= 5.0 and single-series "
        "loss share <= 0.40. Failing any of them is JUDGE_UNREADABLE_STOP and "
        "not one 2025 byte is downloaded."
    ),
    "stage_1": (
        "one merged Guidance card per Consumer, compiled by the frozen recipe "
        "compiler from every frozen delayed evidence row across every source "
        "cohort. NOAA contributes nothing, so leave-one-cohort-out drops "
        "nothing. Card bytes are hashed and frozen. A5 = bootstrap + its "
        "Consumer's card; A3 = bootstrap."
    ),
    "stage_1_5": (
        "an existence check over all 16 stations by HTTP HEAD: no body is "
        "downloaded and no value is read. The roster is locked once, before "
        "the adaptation period starts."
    ),
    "stage_2": (
        "adaptation inside 2024. task_A is a full-price episode (shortlist, "
        "mask round, v2 gate); a successful non-identity adoption is written "
        "through handle_fast_winner as a Draft. The probe at 1440/1488/1536 "
        "takes part in no selection: >= +0.005 promotes the Draft to "
        "LOCAL_ACTIVE through e1._update_delayed, anything else does not "
        "promote. task_B recalls an ACTIVE Skill through the Runner's direct "
        "supply and still pays current-window Support confirmation and the "
        "unchanged v2 gate; no ACTIVE Skill means an honest full-price "
        "re-search."
    ),
    "stage_3": (
        "download only the locked roster's 2025 csv; materialize on the same "
        "grid over [8760, 17520) with no health screening of any kind. The "
        "task_C missing gate reads the finiteness mask only."
    ),
    "stage_4": (
        "task_C on held-out 2025 with the Slow Agent off and no new Skill "
        "formed. Each arm runs a standard episode from its own store's final "
        "state. task_D at s=10560 is the only backup and is used only if the "
        "task_C missing gate fails."
    ),
    "first_positive_cost": (
        "cumulative Consumer retrains, in chronological order task_A -> probe "
        "-> task_B -> task_C, up to and including the first episode that "
        "adopts a non-identity plan whose delayed reading is above zero"
    ),
    "cell_verdicts": {
        "A5_WINS": (
            "(first-positive cost A5 < A3 and task_C delayed difference "
            ">= -0.005 and harm_C A5 <= A3) or (task_C delayed difference "
            "> +0.005 and total cost A5 <= A3 and harm_C A5 <= A3)"
        ),
        "A5_TIE_TRANSFER_BOUNDARY": (
            "both arms adopt the same mechanism, |delayed difference| <= 0.005 "
            "and |total cost difference| < 15"
        ),
        "A5_LOSES": "everything else; harm over budget is a loss on its own",
        "CELL_UNREADABLE": "the cell produced no readable task_C number",
    },
    "overall": {
        "FRESH_A5_DELIVERS": "pooled WINS and per_channel in {WINS, TIE}",
        "FRESH_MIXED": "one WINS, the other LOSES or is UNREADABLE",
        "FRESH_TRANSFER_BOUNDARY": "both cells TIE",
        "FRESH_A5_FAILS": "everything else",
    },
    "stop_verdicts": [
        "JUDGE_UNREADABLE_STOP",
        "INSUFFICIENT_2025_COVERAGE",
        "ACQUISITION_BLOCKED",
        "ROSTER_SPLIT_UNSPECIFIED",
        "CONCURRENT_WRITE_ABORT",
        "LLM_BUDGET_EXHAUSTED",
    ],
    "llm_budget": LLM_CALL_BUDGET_TOTAL,
    "material_threshold": MATERIAL_THRESHOLD,
    "harm_threshold": HARM_THRESHOLD,
    "roster_split_amendment": ROSTER_SPLIT_AMENDMENT,
}


# ------------------------------------------------------------- the cohort
def _cohort_artifact() -> dict[str, Any]:
    payload = json.loads(COHORT_ARTIFACT.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != "noaa_fresh_cohort_v2":
        raise SystemExit("unexpected fresh-cohort protocol_version")
    if payload.get("overall_verdict") != "FRESH_COHORT_READY":
        raise SystemExit(
            "the fresh cohort is not READY: %r" % payload.get("overall_verdict")
        )
    return payload


def _readability_criteria(artifact: Mapping[str, Any]) -> dict[str, Any]:
    quoted = artifact["step_2_health_check_v2"]["criteria_quote_from_13"]
    criteria = dict(quoted["pre_registered_criteria"])
    return {
        "source": str(quoted["source"]),
        "max_eval_loss_spread": float(criteria["max_eval_loss_spread"]),
        "max_single_series_loss_share": float(
            criteria["max_single_series_loss_share"]
        ),
        "quoted_verbatim": True,
        "rationale": str(criteria.get("rationale", "")),
    }


def _missing_cap(artifact: Mapping[str, Any]) -> dict[str, Any]:
    cap = dict(artifact["step_2_health_check_v2"]["missing_rate_cap"])
    return {
        "max_missing_rate": float(cap["max_missing_rate"]),
        "min_finite_fraction": 1.0 - float(cap["max_missing_rate"]),
        "derived_from": str(cap["derived_from"]),
        "structural_sub_guards": [
            "each eval context window carries at least 2 finite points, which "
            "is what v6._linear_integrity requires",
            "each eval future window carries at least 1 finite point, which is "
            "what bch._evaluate_assignment requires",
        ],
        "reads": "the finiteness mask only; no value is compared or reported",
    }


def _load_development(stations: Sequence[str]) -> dict[str, Any]:
    values: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for station in stations:
        folder = SERIES_DIR / str(station)
        array = np.load(folder / "values.npy")
        array = np.asarray(array, dtype=np.float64)
        if int(array.size) != DEVELOPMENT_HOURS:
            raise SystemExit(
                "development series %s is %d long, expected %d"
                % (station, int(array.size), DEVELOPMENT_HOURS)
            )
        values[str(station)] = array
        records[str(station)] = json.loads(
            (folder / "record.json").read_text(encoding="utf-8")
        )
    return {"values": values, "records": records}


def _cohort_payload(
    train: Sequence[str], evaluation: Sequence[str],
    values: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    roster = (
        [{"series_uid": str(uid), "role": "train"} for uid in train]
        + [{"series_uid": str(uid), "role": "eval"} for uid in evaluation]
    )
    return {
        "name": COHORT_NAME,
        "roster": roster,
        "mapped_roster": _mapped_roster(roster),
        "values": {str(uid): values[str(uid)] for uid in list(train) + list(evaluation)},
        "train_uids": [str(uid) for uid in train],
        "eval_uids": [str(uid) for uid in evaluation],
        "exposure": (
            "SEALED_CONFIRMATION_COHORT: noaa_global_hourly_fresh_v1. The "
            "development region below index 8760 is opened by this protocol; "
            "[8760, 17520) is the held-out confirmation year"
        ),
    }


class FreshSearch(bridge.BridgeSearch):
    """``BridgeSearch`` with the cohort handed in instead of looked up.

    ``wvc.BudgetedSearch.__init__`` ends its preamble with
    ``loaded = bch.load_cohort(PROJECT_ROOT, self.cohort)``.  ``load_cohort``
    only knows the already-exposed development cohorts, so the fresh cohort
    cannot be reached through it, and adding a branch there would be an edit
    to the frozen instrument.  The body below is that ``__init__`` with that
    one line replaced by the supplied payload; every evaluation, gain, budget
    and retrain count still runs through the parent's own methods.
    """

    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        consumer_variant: str,
        support_origins: Sequence[int],
        delayed_origins: Sequence[int],
    ) -> None:
        self.retrains = 0
        self.retrain_log: list[dict[str, Any]] = []
        self.cohort = str(payload["name"])
        self.consumer_variant = str(consumer_variant)
        self.support = tuple(int(origin) for origin in support_origins)
        self.delayed = tuple(int(origin) for origin in delayed_origins)
        loaded = payload  # <- the only line that differs from the parent
        self.config = dict(_config())
        self.roster = loaded["mapped_roster"]
        self.values = loaded["values"]
        self.train_uids = [str(uid) for uid in loaded["train_uids"]]
        self.eval_uids = [str(uid) for uid in loaded["eval_uids"]]
        self.exposure = str(loaded["exposure"])
        self._compiled: dict[str, Any] = {}
        self._identity_support = bch._evaluate_variant(
            self.roster, self.values, None, self.config, self.support, None,
            self.consumer_variant,
        )
        self._identity_delayed = bch._evaluate_variant(
            self.roster, self.values, None, self.config, self.delayed, None,
            self.consumer_variant,
        )
        self.support_evaluations_charged = 0
        self.internal_evaluations = 0
        self.log: list[dict[str, Any]] = []
        baseline = len(self.support) + len(self.delayed)
        self.retrains += baseline
        self.retrain_log.append({
            "what": "identity baselines", "retrains": baseline,
        })


# ------------------------------------------------------ stage 0: readability
def _readability(
    rows: Sequence[Mapping[str, Any]], eval_uids: Sequence[str],
    criteria: Mapping[str, Any], *, block: str,
) -> dict[str, Any]:
    """Spread and share by g3_sourcing.development_judge_readability's formulas.

    Same two lines, same direction, computed on the identity rows this
    protocol already paid for rather than on a second evaluation.
    """
    losses = [
        float(np.mean([row["per_view_smase"][index] for row in rows]))
        for index in range(len(eval_uids))
    ]
    finite = [value for value in losses if np.isfinite(value)]
    all_finite = len(finite) == len(losses)
    total = float(sum(losses)) if all_finite else float("nan")
    spread = (
        (max(losses) / min(losses))
        if all_finite and min(losses) > 0 else float("inf")
    )
    share = (max(losses) / total) if all_finite and total > 0 else 1.0
    degenerate = bool(
        all_finite and len({round(value, 12) for value in losses}) == 1
    )
    per_view = [
        float(value) for row in rows for value in row["per_view_smase"]
    ]
    checks = {
        "finite": bool(all_finite and all(np.isfinite(per_view))),
        "non_degenerate": not degenerate,
        "spread_within_cap": bool(
            all_finite and spread <= float(criteria["max_eval_loss_spread"])
        ),
        "share_within_cap": bool(
            all_finite
            and share <= float(criteria["max_single_series_loss_share"])
        ),
    }
    return {
        "block": block,
        "per_series_identity_smase": {
            str(uid): (None if not np.isfinite(value) else float(value))
            for uid, value in zip(eval_uids, losses)
        },
        "min": (float(min(losses)) if all_finite else None),
        "max": (float(max(losses)) if all_finite else None),
        "eval_loss_spread": (float(spread) if np.isfinite(spread) else None),
        "largest_single_series_loss_share": float(share),
        "degenerate_single_value": degenerate,
        "checks": checks,
        "pass": all(checks.values()),
        "formula_source": (
            "g3_sourcing.development_judge_readability: per-series mean sMASE "
            "over the block's origins, spread = max/min, share = max/sum"
        ),
    }


def stage_0(
    payload: Mapping[str, Any], criteria: Mapping[str, Any],
) -> dict[str, Any]:
    """Both Consumers, identity only, task_A window.  12 retrains, 0 LLM."""
    window = WINDOWS["task_A"]
    per_consumer: dict[str, Any] = {}
    retrains = 0
    for variant in CONSUMERS:
        started = time.perf_counter()
        try:
            search = FreshSearch(
                payload=payload,
                consumer_variant=variant,
                support_origins=window["support_origins"],
                delayed_origins=window["delayed_origins"],
            )
        except Exception as exc:  # noqa: BLE001
            per_consumer[variant] = {
                "raised": "%s: %s" % (type(exc).__name__, exc),
                "pass": False,
                "wall_seconds": time.perf_counter() - started,
            }
            continue
        retrains += int(search.retrains)
        blocks = {
            "support": _readability(
                search._identity_support, search.eval_uids, criteria,
                block="support",
            ),
            "delayed": _readability(
                search._identity_delayed, search.eval_uids, criteria,
                block="delayed",
            ),
        }
        per_consumer[variant] = {
            "consumer_variant": variant,
            "consumer_retrains": int(search.retrains),
            "identity_absolute_loss_support": bch._identity_absolute_loss(
                search._identity_support
            ),
            "identity_absolute_loss_delayed": bch._identity_absolute_loss(
                search._identity_delayed
            ),
            "blocks": blocks,
            "pass": bool(blocks["support"]["pass"] and blocks["delayed"]["pass"]),
            "wall_seconds": time.perf_counter() - started,
        }
        print(
            "S0 %-11s pass=%s spread(sup/del)=%s/%s share=%s/%s retrains=%d"
            % (
                variant, per_consumer[variant]["pass"],
                blocks["support"]["eval_loss_spread"],
                blocks["delayed"]["eval_loss_spread"],
                round(blocks["support"]["largest_single_series_loss_share"], 4),
                round(blocks["delayed"]["largest_single_series_loss_share"], 4),
                int(search.retrains),
            ),
            flush=True,
        )
    passed = all(row.get("pass") for row in per_consumer.values())
    return {
        "ran": True,
        "window": {
            key: value for key, value in window.items()
            if not key.startswith("reference_")
        },
        "criteria": dict(criteria),
        "llm_calls": 0,
        "consumer_retrains": retrains,
        "per_consumer": per_consumer,
        "pass": bool(passed),
        "verdict": "READABLE" if passed else "JUDGE_UNREADABLE_STOP",
        "on_failure": (
            "not one 2025 byte is downloaded and no Outcome past index 8760 "
            "is opened"
        ),
    }


# ------------------------------------------------- stage 1: cards and stores
SKILL_ID = {
    variant: "fresh_batch_guidance_%s_v1" % variant for variant in CONSUMERS
}


def _sections(target: Mapping[str, Any], card: Mapping[str, Any]) -> dict[str, str]:
    """ssi._sections, with the target looked up from a parameter.

    Every string is the #10 carrier's own; the only change is that the target
    block arrives as an argument instead of through ``ssi.TARGETS``, which has
    no entry for a cohort that did not exist when it was written.
    """
    by_rule: dict[str, list[Mapping[str, Any]]] = {}
    for clause in card["clauses"]:
        by_rule.setdefault(str(clause["clause_id"]).split("-")[0], []).append(
            clause
        )

    def lines(prefix: str) -> str:
        rows = by_rule.get(prefix) or []
        return " ".join(
            "[%s] %s" % (clause["clause_id"], clause["text"]) for clause in rows
        )

    priority = lines("R1")
    risk = lines("R2")
    locality = lines("R3")
    return {
        "WHEN": (
            "A batch of forecast training series processed as one unit under "
            "the %s Consumer structure, where a data-processing program is "
            "chosen for the whole batch before the Consumer is retrained. "
            "Every record behind this card was measured on other cohorts; "
            "everything measured on this one was withheld."
            % target["consumer_variant"]
        ),
        "OBSERVE": (
            "Read the per-series public table already on this Workspace and "
            "the Consumer structure named in the target block. "
            + (locality or "No mask-locality clause was compiled.")
        ),
        "TRY": (
            priority
            or "No priority clause was compiled: the source records did not "
               "put any program above the threshold on this Consumer "
               "structure."
        ),
        "RISK": (
            risk
            or "No risk clause was compiled. The source corpus records a "
               "delayed number only for the plan each cell adopted, so a "
               "program that loses on delayed in two other cohorts almost "
               "never appears; read the absence as missing evidence, not as "
               "safety."
        ),
        "VERIFY": (
            "Believe nothing here until this batch's own Support evaluation "
            "has been spent on it, and the delayed gate has cleared the plan "
            "you name. A clause is guidance and authorizes no execution."
        ),
        "FALLBACK": (
            "If the public observation does not support the clauses, shortlist "
            "on the observation instead, and keep identity: it is always "
            "available and is the incumbent the gate measures against."
        ),
    }


def _card_payload(
    target: Mapping[str, Any], card: Mapping[str, Any], card_text: str,
) -> dict[str, Any]:
    """A ``skill-entry/1`` value.  ssi._card_payload with the same parameter."""
    variant = str(target["consumer_variant"])
    sections = _sections(target, card)
    body = "\n".join(
        "%s: %s" % (name, sections[name].strip())
        for name in ("WHEN", "OBSERVE", "TRY", "RISK", "VERIFY", "FALLBACK")
    )
    return {
        "schema_version": "skill-entry/1",
        "skill_id": SKILL_ID[variant],
        "skill_kind": "capability",
        "revision": 1,
        "body": body,
        "observable_applicability": {
            "feature": "task_kind", "op": "==", "value": "forecast",
        },
        "allowed_tools": [],
        "risk_guards": {
            "carrier": "deterministic_recipe_compilation_card",
            "source_class": ssi.PROVENANCE_POLICY["source_class"],
            "authorization_scope": ssi.PROVENANCE_POLICY["authorization_scope"],
            "advises_the_proposal_stage_only": True,
            "never_supplies_a_candidate": True,
            "requires_target_support": True,
            "grants_confirmation_free_try": False,
            "namespace": NAMESPACE,
            "compiled_for_target": str(target["target_id"]),
            "compiled_cell": "batch:%s|consumer:%s"
            % (target["cohort"], variant),
            "leave_one_cohort_out_withheld": str(target["cohort"]),
            "why_this_source_class": list(
                ssi.PROVENANCE_POLICY["why_this_class"]
            ),
            "clause_ids": [
                str(clause["clause_id"]) for clause in card["clauses"]
            ],
            "clauses": [
                {
                    "clause_id": str(clause["clause_id"]),
                    "rule": str(clause["rule"]),
                    "text": str(clause["text"]),
                }
                for clause in card["clauses"]
            ],
            "compiler_artifact": bridge._repo_relative(ssi.CARDS_JSON),
            "compiled_card_text": card_text,
            "sections": {
                name: sections[name].strip() for name in sorted(sections)
            },
        },
    }


def _build_store(slot: str, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """ssi._build_store against this protocol's own namespace."""
    root = STORE_ROOT / slot / "snapshots"
    store = SnapshotStore(root)
    base = compile_snapshot(H0_ROOT, verify_lock=False)
    parent = store.materialize(base)
    store.set_active(base.runtime_bundle_sha)
    receipt: dict[str, Any] = {
        "slot": slot,
        "store_root": _repo_rel(root),
        "h0_runtime_bundle_sha": base.runtime_bundle_sha,
        "h0_skill_ids": [skill.skill_id for skill in base.skills],
        "card_registered": None,
        "status": None,
    }
    fork = store.fork(parent, "%s-%s" % (NAMESPACE.replace("_", "-"), slot))
    try:
        if payload is not None:
            try:
                entry = load_learned_skill_entry(dict(payload))
            except Exception as exc:  # noqa: BLE001
                receipt.update({
                    "status": "SCHEMA_BLOCKED",
                    "blocked_at_interface": (
                        "contracts.harness.load_learned_skill_entry"
                    ),
                    "blocked_reason": "%s: %s" % (type(exc).__name__, exc),
                })
                return receipt
            path = fork / "skills" / "learned" / ("%s.json" % entry.skill_id)
            path.write_bytes(canonical_json_bytes(dict(payload)) + b"\n")
            receipt["card_registered"] = {
                "skill_id": entry.skill_id,
                "skill_kind": entry.skill_kind.value,
                "revision": int(entry.revision),
                "allowed_tools": list(entry.allowed_tools),
                "entry_sha256": canonical_sha256(dict(payload)),
                "body_sha256": canonical_sha256({"body": str(entry.body)}),
                "body_characters": len(str(entry.body)),
            }
        try:
            snapshot = compile_snapshot(fork, verify_lock=False)
        except Exception as exc:  # noqa: BLE001
            receipt.update({
                "status": "SCHEMA_BLOCKED",
                "blocked_at_interface": "harness.compiler.compile_snapshot",
                "blocked_reason": "%s: %s" % (type(exc).__name__, exc),
            })
            return receipt
        materialized = store.materialize(snapshot, base.runtime_bundle_sha)
        store.set_active(snapshot.runtime_bundle_sha)
    finally:
        store.discard_fork(fork)
    receipt.update({
        "status": "REGISTERED",
        "runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "harness_content_sha": snapshot.harness_content_sha,
        "skill_ids": [skill.skill_id for skill in snapshot.skills],
        "active_pointer": json.loads(
            store.active_path.read_text(encoding="utf-8")
        ),
        "materialized_root": _repo_rel(materialized.root),
        "retrieval_controls": wvc._plain(snapshot.retrieval),
    })
    receipt["_snapshot"] = snapshot
    receipt["_store"] = store
    receipt["_tree"] = ssi._tree(materialized.root)
    return receipt


def stage_1() -> dict[str, Any]:
    """One merged card per Consumer, then the four stores.  0 LLM."""
    if STORE_ROOT.exists():
        shutil.rmtree(STORE_ROOT)
    cards: dict[str, Any] = {}
    stores: dict[str, Any] = {}
    for variant in CONSUMERS:
        target = {
            "target_id": "FRESH_%s" % variant,
            "cohort": COHORT_NAME,
            "consumer_variant": variant,
        }
        card = bridge.compile_skill_card(target)
        card_text = bridge.render_skill_card(card)
        payload = _card_payload(target, card, card_text)
        guarantees = ssi._assert_guidance_only(payload)
        card_bytes = canonical_json_bytes(dict(payload))
        cards[variant] = {
            "target": dict(target),
            "status": str(card["status"]),
            "clause_count": int(card["clause_count"]),
            "clause_ids": [
                str(clause["clause_id"]) for clause in card["clauses"]
            ],
            "clauses": [
                {
                    "clause_id": str(clause["clause_id"]),
                    "rule": str(clause["rule"]),
                    "text": str(clause["text"]),
                }
                for clause in card["clauses"]
            ],
            "rules_that_produced_nothing": list(
                card["rules_that_produced_nothing"]
            ),
            "loco": dict(card["loco"]),
            "loco_note": (
                "the fresh cohort contributes no evidence row, so "
                "leave-one-cohort-out drops nothing and the card is the "
                "all-source merge"
            ),
            "card_text": card_text,
            "card_bytes_sha256": hashlib.sha256(card_bytes).hexdigest(),
            "entry_sha256": canonical_sha256(dict(payload)),
            "carrier_guarantees": guarantees,
            "_payload": payload,
        }
        print(
            "S1 card %-11s %s clauses=%s sha=%s"
            % (
                variant, card["status"], cards[variant]["clause_ids"],
                cards[variant]["card_bytes_sha256"][:12],
            ),
            flush=True,
        )
    for variant in CONSUMERS:
        for arm in ARMS:
            slot = "%s_%s" % (arm.lower(), variant)
            payload = cards[variant]["_payload"] if arm == "A5" else None
            stores[slot] = _build_store(slot, payload)
            stores[slot].update({"arm": arm, "consumer_variant": variant})
            print(
                "S1 store %-16s %s skills=%s"
                % (
                    slot, stores[slot]["status"],
                    stores[slot].get("skill_ids"),
                ),
                flush=True,
            )
    blocked = [row for row in stores.values() if row["status"] != "REGISTERED"]
    parity = {
        variant: ssi._store_parity(
            stores["a3_%s" % variant], stores["a5_%s" % variant]
        )
        for variant in CONSUMERS
    }
    return {
        "ran": True,
        "llm_calls": 0,
        "consumer_retrains": 0,
        "cards": cards,
        "stores": stores,
        "store_parity": parity,
        "provenance_policy": dict(ssi.PROVENANCE_POLICY),
        "blocked": [row["slot"] for row in blocked],
        "verdict": "REGISTERED" if not blocked else "SCHEMA_BLOCKED",
    }


# ------------------------------------------------ stage 1.5: does 2025 exist
def _head(session: Any, station: str, timeout: float = 30.0) -> dict[str, Any]:
    """One HTTP HEAD.  No body is transferred and no value is read."""
    url = NOAA_URL % station
    row: dict[str, Any] = {"station": str(station), "url": url}
    try:
        response = session.head(url, timeout=timeout, allow_redirects=True)
    except Exception as exc:  # noqa: BLE001
        row.update({
            "exists": None,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "method": "HEAD",
        })
        return row
    row.update({
        "method": "HEAD",
        "status_code": int(response.status_code),
        "content_length": response.headers.get("Content-Length"),
        "last_modified": response.headers.get("Last-Modified"),
        "exists": bool(response.status_code == 200),
        "body_bytes_read": 0,
    })
    return row


def stage_1_5(train: Sequence[str], evaluation: Sequence[str]) -> dict[str, Any]:
    """Existence over all 16 stations, then the roster is locked once."""
    import requests

    session = requests.Session()
    listing = [_head(session, station) for station in list(train) + list(evaluation)]
    session.close()
    by_station = {row["station"]: row for row in listing}
    unsupported = [
        row["station"] for row in listing
        if row.get("status_code") in (405, 501)
    ]
    errored = [row["station"] for row in listing if row.get("exists") is None]
    present = {row["station"] for row in listing if row.get("exists")}

    kept_train = [str(uid) for uid in train if str(uid) in present]
    kept_eval = [str(uid) for uid in evaluation if str(uid) in present]
    promoted: list[str] = []
    while len(kept_train) < len(train) and kept_eval:
        promoted.append(kept_eval.pop(0))
        kept_train.append(promoted[-1])
    kept_train = sorted(kept_train)
    kept_eval = sorted(kept_eval)

    enough = (
        len(kept_train) == len(train)
        and len(kept_eval) >= int(
            ROSTER_SPLIT_AMENDMENT["min_eval_after_cascade"]
        )
    )
    return {
        "ran": True,
        "llm_calls": 0,
        "consumer_retrains": 0,
        "bytes_downloaded": 0,
        "method": "HTTP HEAD; no body, no value",
        "checked": len(listing),
        "listing": listing,
        "head_unsupported": unsupported,
        "errored": errored,
        "missing": sorted(
            row["station"] for row in listing if row.get("exists") is False
        ),
        "cascade": {
            "requested_train": [str(uid) for uid in train],
            "requested_eval": [str(uid) for uid in evaluation],
            "promoted_from_eval_to_train": promoted,
            "locked_train": kept_train,
            "locked_eval": kept_eval,
            "rule": ROSTER_SPLIT_AMENDMENT["cascade_pre_registered"],
        },
        "locked_roster": {"train": kept_train, "eval": kept_eval},
        "sufficient": bool(enough),
        "verdict": "ROSTER_LOCKED" if enough else "INSUFFICIENT_2025_COVERAGE",
        "unverified_note": (
            None if not (unsupported or errored) else
            "HEAD did not answer for %s; existence for those stations is "
            "unverified and is settled by the stage 3 download instead"
            % sorted(set(unsupported) | set(errored))
        ),
        "by_station": by_station,
    }


# -------------------------------------------- stage 3: acquire and materialize
def _decode_tmp(raw: str) -> float | None:
    """noaa_fresh_materialize._decode_tmp, unchanged."""
    token = raw.split(",", 1)[0].strip()
    try:
        value = int(token)
    except ValueError:
        return None
    if abs(value) == TMP_MISSING:
        return None
    return value / 10.0


def _hour_index(date_raw: str) -> int | None:
    """noaa_fresh_materialize._parse_hour_index, unchanged."""
    if not date_raw:
        return None
    try:
        timestamp = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)
    timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
    return int((timestamp - GRID_START).total_seconds() // 3600)


def _parse_confirmation(path: Path) -> dict[str, Any]:
    """The 2025 csv onto [8760, 17520) of the same grid.

    Byte for byte the development parser's rules -- missing hours stay NaN,
    first finite value in an hour wins, the 9999 token is missing -- with the
    window moved to the confirmation partition.  No screening, no repair, no
    span selection: whatever the station recorded is what lands.
    """
    span = CONFIRMATION_END - DEVELOPMENT_HOURS
    values = np.full(span, np.nan, dtype=np.float64)
    counters = {
        "n_rows_seen": 0,
        "n_tmp_written": 0,
        "n_outside_window": 0,
        "n_missing_or_unparsed_tmp_in_window": 0,
        "n_hour_collisions_ignored": 0,
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counters["n_rows_seen"] += 1
            index = _hour_index(str(row.get("DATE") or ""))
            if index is None or not (
                DEVELOPMENT_HOURS <= index < CONFIRMATION_END
            ):
                counters["n_outside_window"] += 1
                continue
            decoded = _decode_tmp(str(row.get("TMP") or ""))
            if decoded is None:
                counters["n_missing_or_unparsed_tmp_in_window"] += 1
                continue
            slot = index - DEVELOPMENT_HOURS
            if np.isfinite(values[slot]):
                counters["n_hour_collisions_ignored"] += 1
                continue
            values[slot] = decoded
            counters["n_tmp_written"] += 1
    finite = int(np.isfinite(values).sum())
    counters.update({
        "n_finite_confirmation": finite,
        "n_nan_confirmation": span - finite,
        "missing_rate_confirmation": 1.0 - finite / span,
    })
    return {"values": values, **counters}


def stage_3(stations: Sequence[str], *, write: bool = True) -> dict[str, Any]:
    """Download the locked roster's 2025 csv, then materialize.  0 screening."""
    import requests

    RAW_2025.mkdir(parents=True, exist_ok=True)
    if write:
        CONFIRMATION_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    downloads: list[dict[str, Any]] = []
    failures: list[str] = []
    for station in stations:
        target = RAW_2025 / ("%s.csv" % station)
        row: dict[str, Any] = {
            "station": str(station),
            "url": NOAA_URL % station,
            "path": _repo_rel(target),
            "already_on_disk": target.is_file(),
        }
        if not target.is_file():
            try:
                response = session.get(
                    NOAA_URL % station, timeout=180, stream=True
                )
                response.raise_for_status()
                with target.open("wb") as handle:
                    for block in response.iter_content(1 << 20):
                        handle.write(block)
                row["downloaded"] = True
            except Exception as exc:  # noqa: BLE001
                row.update({
                    "downloaded": False,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                })
                if target.is_file():
                    target.unlink()
                failures.append(str(station))
                downloads.append(row)
                continue
        else:
            row["downloaded"] = False
        row["bytes"] = int(target.stat().st_size)
        row["sha256"] = _sha256(target)
        downloads.append(row)
    session.close()
    if failures:
        return {
            "ran": True,
            "llm_calls": 0,
            "consumer_retrains": 0,
            "downloads": downloads,
            "failures": failures,
            "verdict": "ACQUISITION_BLOCKED",
            "values": {},
            "note": (
                "the 2025 csv could not be acquired for %s; the in-2024 split "
                "alternative is a main-line correction and is not improvised "
                "here" % failures
            ),
        }

    per_station: list[dict[str, Any]] = []
    confirmation: dict[str, np.ndarray] = {}
    for station in stations:
        parsed = _parse_confirmation(RAW_2025 / ("%s.csv" % station))
        series = np.asarray(parsed.pop("values"), dtype=np.float64)
        confirmation[str(station)] = series
        record = {
            "dataset_id": "noaa_global_hourly_fresh_v1",
            "entity_id": str(station),
            "partition": "confirmation_2025",
            "grid_start": GRID_START.isoformat(),
            "index_range": [DEVELOPMENT_HOURS, CONFIRMATION_END],
            "length": int(series.size),
            "parse": {
                "missing_timestamp_is_nan": True,
                "interpolate": False,
                "smooth": False,
                "select_good_span": False,
                "hourly_rule": "first_finite_wins",
                "tmp_missing_code": TMP_MISSING,
                "health_screening": "none; this partition is not screened",
            },
            "source_csv": _repo_rel(RAW_2025 / ("%s.csv" % station)),
            **parsed,
        }
        if write:
            folder = CONFIRMATION_DIR / str(station)
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "values.npy", series)
            (folder / "record.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8", newline="\n",
            )
        per_station.append(record)
        print(
            "S3 %s finite=%d/%d missing=%.4f"
            % (
                station, record["n_finite_confirmation"], int(series.size),
                record["missing_rate_confirmation"],
            ),
            flush=True,
        )
    return {
        "ran": True,
        "llm_calls": 0,
        "consumer_retrains": 0,
        "downloads": downloads,
        "failures": [],
        "index_range": [DEVELOPMENT_HOURS, CONFIRMATION_END],
        "grid_rule": (
            "the same hours-since-2024-01-01 grid as the development slice; "
            "indices 8760..8783 belong to 2024-12-31 and are not in a 2025 "
            "file, so they stay NaN"
        ),
        "screening": "none",
        "per_station": per_station,
        "written_to": _repo_rel(CONFIRMATION_DIR) if write else None,
        "verdict": "MATERIALIZED",
        "values": confirmation,
    }


def _missing_gate(
    values: Mapping[str, np.ndarray], eval_uids: Sequence[str],
    window: Mapping[str, Any], cap: Mapping[str, Any],
) -> dict[str, Any]:
    """The task_C missing gate.  Reads the finiteness mask, never a value."""
    origins = [int(o) for o in window["support_origins"]] + [
        int(o) for o in window["delayed_origins"]
    ]
    lo = min(origins) - CONTEXT_LENGTH
    hi = max(origins) + HORIZON
    rows: list[dict[str, Any]] = []
    for uid in eval_uids:
        mask = np.isfinite(np.asarray(values[str(uid)], dtype=np.float64))
        span = mask[lo:hi]
        contexts = [int(mask[o - CONTEXT_LENGTH:o].sum()) for o in origins]
        futures = [int(mask[o:o + HORIZON].sum()) for o in origins]
        fraction = float(span.sum()) / float(span.size)
        row = {
            "series_uid": str(uid),
            "span": [int(lo), int(hi)],
            "finite_points": int(span.sum()),
            "span_points": int(span.size),
            "finite_fraction": fraction,
            "min_context_finite": min(contexts),
            "min_future_finite": min(futures),
            "rate_ok": bool(fraction >= float(cap["min_finite_fraction"])),
            "context_ok": bool(min(contexts) >= 2),
            "future_ok": bool(min(futures) >= 1),
        }
        row["pass"] = bool(row["rate_ok"] and row["context_ok"] and row["future_ok"])
        rows.append(row)
    passed = all(row["pass"] for row in rows)
    return {
        "window_id": str(window["window_id"]),
        "cap": dict(cap),
        "per_series": rows,
        "failing": [row["series_uid"] for row in rows if not row["pass"]],
        "pass": bool(passed),
        "values_read": 0,
        "note": "computed from np.isfinite only; no value entered any number here",
    }


# ------------------------------------------------------------- the episodes
class _Receipt:
    """The shape method.py's evaluators return.  Carries a number, nothing else."""

    def __init__(self, gain: float) -> None:
        self.gain = float(gain)
        self.verification = type("Verification", (), {"passed": True})()


def _retrieval(
    snapshot: Any, search: Any, expected_card: str | None,
    expected_local: str | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """The real three-stage resolution.  The runner adds nothing to it."""
    context = ssi._public_features(search)
    view = resolve_harness_view(snapshot, dict(context["features"]), role="fast")
    _view, card_row = ssi._retrieve(snapshot, context["features"], expected_card)
    local = next(
        (
            skill for skill in view.skills
            if expected_local and skill.skill_id == expected_local
        ),
        None,
    )
    local_row = None
    if local is not None:
        steps = _parse_frozen_steps(local.body)
        local_row = {
            "skill_id": local.skill_id,
            "body": str(local.body),
            "risk_guards": wvc._plain(local.risk_guards or {}),
            "frozen_steps": (
                None if steps is None else
                [{"op": op, "params": dict(params)} for op, params in steps]
            ),
        }
    record = dict(card_row)
    record.update({
        "expected_local_skill_id": expected_local,
        "local_skill_hit": (bool(local) if expected_local else None),
        "local_skill": local_row,
        "resolved_skill_ids": list(view.skill_ids),
        "context": {
            key: value for key, value in context.items() if key != "per_series"
        },
    })
    return view, record, context


def _relation(support_gain: float, delayed_gain: float, program: str) -> str:
    if str(program) == IDENTITY:
        return RELATION_ABSTAIN
    if support_gain > 0.0 and delayed_gain > 0.0:
        return RELATION_POSITIVE
    if (support_gain > 0.0) != (delayed_gain > 0.0):
        return RELATION_CONFLICT
    return RELATION_NEGATIVE


def _episode(
    *, search: Any, target: Mapping[str, Any], arm: str,
    window: Mapping[str, Any], slot: Mapping[str, Any],
    expected_card: str | None, expected_local: str | None, llm_budget: int,
    tag: str,
) -> dict[str, Any]:
    """A full-price episode: shortlist, optional mask round, v2 gate.

    ssi._run_arm with the search handed in rather than constructed, because
    the fresh cohort is not reachable through ``bch.load_cohort``.  Every
    prompt string, schema, validator, budget and ladder call below is the
    frozen one.
    """
    started = time.perf_counter()
    episode_id = "%s_%s_%s" % (tag, target["consumer_variant"], arm)
    observation = wvc._observation_table(search)
    view, retrieval, context = _retrieval(
        slot["_snapshot"], search, expected_card, expected_local,
    )
    served = retrieval.get("served_card") or {}
    clause_ids = [str(item) for item in served.get("clause_ids", ())]
    base = ssi._base_input(
        target=target, window=window, search=search, observation=observation,
    )
    backend = ssi._default_backend_factory(int(llm_budget))
    gateway = wvc.NoToolGateway({"episode_id": episode_id, "arm": arm})
    core = TTHAAgentCore(
        backend, gateway, model=ssi.NF_MODEL, base_url=ssi.NF_BASE_URL,
    )
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "task": tag,
        "arm": arm,
        "consumer_variant": str(target["consumer_variant"]),
        "window_id": str(window["window_id"]),
        "support_origins": list(search.support),
        "delayed_origins": list(search.delayed),
        "store_slot": slot["slot"],
        "store_runtime_bundle_sha": slot.get("runtime_bundle_sha"),
        "mode": "FULL_PRICE_SEARCH",
        "retrieval": retrieval,
        "task_context": {
            key: value for key, value in context.items() if key != "per_series"
        },
        "skill_clause_ids_available": clause_ids,
        "public_input_sha256": canonical_sha256(wvc._plain(base)),
        "stages": [],
        "shortlist": [],
        "evaluations_used": 0,
        "final_plan": None,
        "support": None,
        "delayed": None,
        "llm_calls": 0,
    }

    def close() -> dict[str, Any]:
        record["llm_calls"] = int(backend.calls)
        record["instrument"] = search.accounting()
        record["consumer_retrains_total"] = int(search.retrains)
        record["wall_seconds"] = time.perf_counter() - started
        return record

    shortlist_payload, shortlist_info = wvc._stage(
        core,
        stage="skill_store_shortlist",
        case_id="FC_%s" % episode_id,
        public_input={**base, "stage_note": ssi.SHORTLIST_NOTE},
        harness_view=view,
        schema_name="skill_bridge_shortlist_v1",
        schema=ssi.SHORTLIST_SCHEMA,
        validator=ssi._make_shortlist_validator(clause_ids),
    )
    record["stages"].append(shortlist_info)
    record["shortlist_payload"] = wvc._plain(shortlist_payload)
    if shortlist_payload is None:
        record["stopped"] = "the shortlist stage produced no payload"
        return close()

    shortlist = [str(item) for item in shortlist_payload["shortlist"]]
    wants_mask = bool(shortlist_payload["request_mask_search"])
    cited = [str(item) for item in shortlist_payload.get("skill_clause_use", ())]
    support_results = {
        program: search.full_batch_support(program) for program in shortlist
    }
    mask_result = None
    if wants_mask:
        best = max(
            shortlist,
            key=lambda program: (
                support_results[program]["aggregate_gain"],
                -shortlist.index(program),
            ),
        )
        mask_result = search.mask_search(best)
    plans, mask_note = bridge._measured_plans(
        shortlist=shortlist, support_results=support_results,
        mask_result=mask_result,
    )
    record.update({
        "shortlist": shortlist,
        "request_mask_search": wants_mask,
        "skill_clause_use": cited,
        "shortlist_reason": str(shortlist_payload.get("reason", "")),
        "support_results": support_results,
        "mask_search": wvc._plain(mask_result),
        "measured_plans": plans,
        "measured_plans_note": mask_note,
        "evaluations_used": int(search.support_evaluations_charged),
    })
    print(
        "FC %-28s hit=%s shortlist=%s mask=%s cited=%s"
        % (episode_id, retrieval.get("hit"), shortlist, wants_mask, cited),
        flush=True,
    )
    adoption_payload, adoption_info = wvc._stage(
        core,
        stage="skill_store_adoption",
        case_id="FC_%s" % episode_id,
        public_input={
            **base,
            "stage_note": ssi.ADOPTION_NOTE,
            "your_shortlist": list(shortlist),
            "measured_plans": [dict(row) for row in plans],
            "measured_plans_note": mask_note,
            "evaluations_spent": int(record["evaluations_used"]),
        },
        harness_view=view,
        schema_name="budgeted_adoption_v1",
        schema=ssi.ADOPTION_SCHEMA,
        validator=wvc._make_adoption_validator(
            shortlist=shortlist, mask_result=record.get("mask_search"),
        ),
    )
    record["stages"].append(adoption_info)
    record["adoption_payload"] = wvc._plain(adoption_payload)
    if adoption_payload is None:
        record["stopped"] = "the adoption stage produced no payload"
        return close()

    named = {
        "program": str(adoption_payload["program"]),
        "excluded_series": sorted(
            str(uid) for uid in adoption_payload.get("excluded_series", ())
        ),
    }
    ladder = ssi._ladder(search, plans=plans, named=named)
    support_gain = float(ladder["support"]["aggregate_gain"])
    delayed_gain = float(ladder["delayed"]["aggregate_gain"])
    record.update({
        "adopted_plan": named,
        "adoption_reason": str(adoption_payload.get("reason", "")),
        "adoption_ladder": {
            key: value for key, value in ladder.items()
            if key not in ("support", "delayed")
        },
        "final_plan": dict(ladder["final_plan"]),
        "support": dict(ladder["support"]),
        "delayed": dict(ladder["delayed"]),
        "relation": _relation(
            support_gain, delayed_gain, ladder["final_plan"]["program"]
        ),
    })
    out = close()
    print(
        "FC %-28s final %s minus %s | support %+.6f delayed %+.6f | %s | "
        "harm %d retrains %d llm %d"
        % (
            episode_id, ladder["final_plan"]["program"],
            ", ".join(ladder["final_plan"]["excluded_series"]) or "nothing",
            support_gain, delayed_gain, ladder["path"],
            int(ladder["delayed"]["harmed_eval_series_count"]),
            int(search.retrains), int(out["llm_calls"]),
        ),
        flush=True,
    )
    return out


def _confirm_support(search: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    """One charged current-window Support confirmation of a recalled plan."""
    program = str(plan["program"])
    excluded = sorted(str(uid) for uid in plan["excluded_series"])
    if not excluded:
        return search.full_batch_support(program)
    search.support_evaluations_charged += 1
    gains = search._gains(search._masked(program, set(excluded), search.support))
    search.log.append({
        "kind": "local_skill_direct_support_confirmation",
        "program": program,
        "excluded_series": excluded,
        "charged": True,
        "aggregate_gain": gains["aggregate_gain"],
    })
    return gains


def _direct_recall(
    *, search: Any, target: Mapping[str, Any], arm: str,
    window: Mapping[str, Any], slot: Mapping[str, Any],
    expected_card: str | None, expected_local: str, tag: str,
) -> dict[str, Any]:
    """Runner direct supply of a naturally retrieved ACTIVE local Skill.

    The #15 shape: the Skill supplies its frozen plan, the plan still pays
    this window's Support confirmation and still walks the unchanged v2
    ladder.  0 LLM.
    """
    started = time.perf_counter()
    episode_id = "%s_%s_%s" % (tag, target["consumer_variant"], arm)
    _view, retrieval, context = _retrieval(
        slot["_snapshot"], search, expected_card, expected_local,
    )
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "task": tag,
        "arm": arm,
        "consumer_variant": str(target["consumer_variant"]),
        "window_id": str(window["window_id"]),
        "support_origins": list(search.support),
        "delayed_origins": list(search.delayed),
        "store_slot": slot["slot"],
        "mode": "DIRECT_RECALL",
        "retrieval": retrieval,
        "task_context": {
            key: value for key, value in context.items() if key != "per_series"
        },
        "llm_calls": 0,
    }
    skill = retrieval.get("local_skill")
    if not skill:
        record.update({
            "recall_payload": None,
            "final_plan": None,
            "support": None,
            "delayed": None,
            "evaluations_used": 0,
            "consumer_retrains_total": int(search.retrains),
            "instrument": search.accounting(),
            "wall_seconds": time.perf_counter() - started,
            "stopped": "RECALL_MISS",
        })
        return record
    guards = dict(skill["risk_guards"])
    frozen = dict(guards.get("frozen_plan") or {})
    steps = list(skill.get("frozen_steps") or [])
    if (
        len(steps) != 1
        or steps[0].get("op") != frozen.get("program")
        or dict(steps[0].get("params") or {})
        or frozen.get("program") not in TREATMENTS
    ):
        raise ValueError("the retrieved local Skill carries no single valid plan")
    plan = {
        "program": str(frozen["program"]),
        "excluded_series": sorted(
            str(uid) for uid in frozen.get("excluded_series") or ()
        ),
    }
    unknown = sorted(set(plan["excluded_series"]) - set(search.train_uids))
    if unknown:
        raise ValueError("the frozen plan excludes unknown training series %s" % unknown)
    support = _confirm_support(search, plan)
    support_gain = float(support["aggregate_gain"])
    material = support_gain >= MATERIAL_THRESHOLD
    record["recall_payload"] = {
        "source": "naturally_retrieved_target_local_skill",
        "skill_id": expected_local,
        "frozen_steps": steps,
        "frozen_plan": plan,
        "lifecycle_at_recall": {
            "local_status": guards.get("local_status"),
            "evidence_level": guards.get("evidence_level"),
            "activation_probe_window": guards.get("activation_probe_window"),
            "activation_probe_gain": guards.get("activation_probe_gain"),
        },
        "current_support_confirmation": {
            "aggregate_gain": support_gain,
            "material_threshold": MATERIAL_THRESHOLD,
            "passed": bool(material),
        },
    }
    if material:
        measured = [{
            "kind": "FULL_BATCH" if not plan["excluded_series"] else "MASKED_PLAN",
            "program": plan["program"],
            "excluded_series": list(plan["excluded_series"]),
            "full_batch": not plan["excluded_series"],
            "support_aggregate_gain": support_gain,
        }]
        ladder = ssi._ladder(search, plans=measured, named=plan)
        final_plan = dict(ladder["final_plan"])
        final_support = dict(ladder["support"])
        delayed = dict(ladder["delayed"])
        adoption = {
            key: value for key, value in ladder.items()
            if key not in ("support", "delayed")
        }
    else:
        final_plan = {"program": IDENTITY, "excluded_series": []}
        final_support = dict(search.support_of_plan(IDENTITY, []))
        delayed = dict(search.delayed_gate(IDENTITY, []))
        adoption = {
            "path": "SUPPORT_CONFIRMATION_FAILED_ABSTAIN",
            "path_text": (
                "the recalled plan missed the +0.005 current-window Support "
                "line; this episode abstained rather than forcing reuse"
            ),
            "bar": 0.0,
            "gate_passed": False,
            "final_plan": dict(final_plan),
        }
    reused = bool(
        final_plan["program"] == plan["program"]
        and sorted(final_plan["excluded_series"]) == sorted(plan["excluded_series"])
    )
    record.update({
        "recall_candidate_plan": plan,
        "reuse_adopted": reused,
        "support_confirmation_passed": bool(material),
        "adoption_ladder": adoption,
        "final_plan": final_plan,
        "support": final_support,
        "delayed": delayed,
        "relation": _relation(
            float(final_support["aggregate_gain"]),
            float(delayed["aggregate_gain"]), final_plan["program"],
        ),
        "evaluations_used": int(search.support_evaluations_charged),
        "consumer_retrains_total": int(search.retrains),
        "instrument": search.accounting(),
        "wall_seconds": time.perf_counter() - started,
    })
    print(
        "FC %-28s recall %s minus %s | support %+.6f delayed %+.6f | harm %d "
        "retrains %d llm 0"
        % (
            episode_id, final_plan["program"],
            ", ".join(final_plan["excluded_series"]) or "nothing",
            float(final_support["aggregate_gain"]),
            float(delayed["aggregate_gain"]),
            int(delayed["harmed_eval_series_count"]), int(search.retrains),
        ),
        flush=True,
    )
    return record


# --------------------------------------------------------- the lifecycle path
EXPERIENCE_PROVENANCE = "fresh_confirmation"
UPDATE_PATH = {
    "draft": "methods/ttha/method.py::TTHAMethod.handle_fast_winner",
    "store_approval": "methods/ttha/method.py::TTHAMethod.handle_feedback_delayed",
    "promotion": "task_episode_harness/e1.py::_update_delayed",
    "probe_metric": "task_episode_harness/runner.py::_arm_metrics",
    "nothing_is_set_by_hand": True,
    "slow_agent_called": False,
    "slow_agent_note": (
        "slow_agent._resolve_apply_manifest is a manifest template resolver "
        "used by the store write; the Slow Agent itself is never asked for a "
        "proposal in this protocol"
    ),
}


def _episode_view(episode: Any) -> dict[str, Any]:
    return {
        "episode_id": str(episode.episode_id),
        "local_status": str(episode.local_status),
        "relation": str(episode.relation),
        "evidence_level": str(episode.evidence_level),
        "workflow_signature": str(episode.workflow_signature),
        "support_gain": (episode.support_response or {}).get("gain"),
        "delayed_gain": (episode.delayed_response or {}).get("gain"),
        "delayed_block_origins": (
            (episode.delayed_response or {}).get("block_origins")
        ),
    }


def _build_local_episode(
    *, record: Mapping[str, Any], target: Mapping[str, Any], arm: str,
    steps: Sequence[tuple[str, Mapping[str, Any]]],
) -> Any:
    plan = dict(record["final_plan"])
    support_gain = float(record["support"]["aggregate_gain"])
    audit = {
        "provenance": EXPERIENCE_PROVENANCE,
        "counts_as_unguided_exploration": False,
    }
    return build_episode(
        episode_id="fc_%s_%s_%s" % (
            str(target["consumer_variant"]), arm.lower(), record["task"],
        ),
        # #42k-b F3: the task hard key is task_type|downstream_model_class|
        # metric, minted by experience_memory.task_consumer_key over this
        # line's live TaskSpec.  It used to carry cell_key's
        # ``batch:<cohort>|consumer:<variant>``, which is a different key with
        # a different arity -- _task_scope_of_episode now rejects it rather
        # than reading it as forecast.  cohort stays where it belongs:
        # domain_namespace plus context_summary.cohort.
        task_consumer_key=ssi._runtime_task_consumer_key(
            str(target["consumer_variant"])),
        domain_namespace=str(target["cohort"]),
        context_summary={
            "task_episode_id": str(record["window_id"]),
            "arm": arm,
            "cohort": {"cohort_name": str(target["cohort"])},
            "local_pattern": {
                "consumer_variant": str(target["consumer_variant"]),
            },
            "program_geometry": {
                "program_steps": [
                    {"op": op, "params": dict(params)} for op, params in steps
                ],
                "frozen_plan_scope": {
                    "excluded_series": sorted(
                        str(uid) for uid in plan["excluded_series"]
                    )
                },
                "consumer_retrains": int(record["consumer_retrains_total"]),
            },
        },
        workflow_signature=e1mod._v2_workflow_signature(steps),
        support_response={
            "gain": support_gain,
            "accepted": support_gain >= MATERIAL_THRESHOLD,
            "block_origins": list(record["support_origins"]),
            "program": str(plan["program"]),
            "excluded_series": list(plan["excluded_series"]),
            **audit,
        },
        delayed_response={
            "evaluated": True,
            "gain": float(record["delayed"]["aggregate_gain"]),
            "se_block": None,
            "gain_over_se": None,
            "block_origins": list(record["delayed_origins"]),
            "took_part_in_selection": True,
            "why_not_promotion_evidence": (
                "this delayed reading set the adoption bar in its own episode, "
                "so it may not also license the promotion"
            ),
            **audit,
        },
        relation=str(record["relation"]),
        evidence_level=EVIDENCE_SUPPORT,
        local_status=STATUS_LOCAL_DRAFT,
        evidence_refs=(EXPERIENCE_PROVENANCE, PROTOCOL_VERSION),
    )


def _persist_draft(
    *, slot: dict[str, Any], record: Mapping[str, Any],
    target: Mapping[str, Any], arm: str,
) -> dict[str, Any]:
    """handle_fast_winner writes the Draft.  Nothing is set by hand."""
    plan = dict(record["final_plan"])
    support_gain = float(record["support"]["aggregate_gain"])
    if str(plan["program"]) == IDENTITY:
        return {
            "written": False,
            "reason": "the episode adopted identity; there is no Skill to form",
        }
    if support_gain < MATERIAL_THRESHOLD:
        return {
            "written": False,
            "reason": (
                "the adopted plan's Support %+.6f is under the %+.6f material "
                "line, which is the Draft gate handle_fast_winner enforces"
                % (support_gain, MATERIAL_THRESHOLD)
            ),
        }
    steps = ((str(plan["program"]), {}),)
    episode = _build_local_episode(
        record=record, target=target, arm=arm, steps=steps,
    )
    method = TTHAMethod(
        e1mod._FastAgentStub(), slot["_snapshot"], experience_episodes=(episode,),
    )
    controller = EditController(
        slot["_store"], surfaces=SurfaceRegistry(), router=FaultRouter(),
    )
    card = {
        "pattern_id": "fresh-confirmation-v1-%s" % slot["slot"],
        "failure_family": "fresh_cohort_batch_program_gap",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {
            "steps": [{"op": op, "params": dict(p)} for op, p in steps]
        },
    }
    event = method.handle_fast_winner(
        episode,
        steps,
        controller=controller,
        store=slot["_store"],
        card=card,
        evaluator=lambda _steps, _mode: _Receipt(support_gain),
        fast_features={"task_kind": "forecast"},
        support_gain=support_gain,
        confirmed_cause="SKILL_LIBRARY_GAP",
    )
    skill_id = fast_winner_skill_id(episode)
    written = str(event.get("stage")) == "pending"
    slot["_method"] = method
    slot["_episode"] = episode
    slot["_steps"] = steps
    slot["_plan"] = plan
    return {
        "written": bool(written),
        "skill_id": skill_id if written else None,
        "handle_fast_winner": wvc._plain(event),
        "call_site": UPDATE_PATH["draft"],
        "episode_before": _episode_view(episode),
        "reason": (
            None if written else
            "handle_fast_winner returned stage=%r" % event.get("stage")
        ),
    }


def _probe(
    *, search: Any, payload: Mapping[str, Any], variant: str,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """One out-of-selection delayed evaluation on the promotion window.

    Its reading takes part in no proposal and no adoption.  It is charged to
    the arm that owns it: identity baseline and candidate are both paid here
    rather than shared, so the two arms stay cost-symmetric.
    """
    origins = tuple(int(o) for o in PROBE_ORIGINS)
    roster = payload["mapped_roster"]
    values = payload["values"]
    config = dict(_config())
    eval_uids = [str(uid) for uid in payload["eval_uids"]]
    identity_rows = bch._evaluate_variant(
        roster, values, None, config, origins, None, variant,
    )
    excluded = {str(uid) for uid in plan["excluded_series"]}
    compiled = search._program(str(plan["program"]))
    if excluded:
        assignment = {
            uid: (None if uid in excluded else compiled)
            for uid in payload["train_uids"]
        }
        candidate_rows = [
            bch._evaluate_assignment(
                roster, values, assignment, config, origin=origin,
                consumer_variant=variant,
            )
            for origin in origins
        ]
    else:
        candidate_rows = bch._evaluate_variant(
            roster, values, compiled, config, origins, None, variant,
        )
    metrics = _arm_metrics(
        list(identity_rows), list(candidate_rows), origins, list(eval_uids),
    )
    gains = bch._gain_rows(identity_rows, candidate_rows, eval_uids)
    return {
        "window": dict(PROBE_WINDOW),
        "origins": [int(o) for o in origins],
        "plan": {
            "program": str(plan["program"]),
            "excluded_series": sorted(str(uid) for uid in plan["excluded_series"]),
        },
        "macro_gain": float(metrics["macro_gain"]),
        "se_block": float(metrics["se_block"]),
        "gain_over_se": metrics["gain_over_se"],
        "harmed_eval_series_count": int(gains["harmed_eval_series_count"]),
        "harmed_eval_series": list(gains["harmed_eval_series"]),
        "harmed_eval_series_total_harm": float(
            gains["harmed_eval_series_total_harm"]
        ),
        "consumer_retrains": 2 * len(origins),
        "took_part_in_selection": False,
        "metric_source": UPDATE_PATH["probe_metric"],
    }


def _promote(
    *, slot: dict[str, Any], probe: Mapping[str, Any], draft: Mapping[str, Any],
) -> dict[str, Any]:
    """The probe drives the store approval and the episode transition."""
    method = slot.get("_method")
    episode = slot.get("_episode")
    if method is None or episode is None:
        return {"promoted": False, "reason": "no Draft was written"}
    gain = float(probe["macro_gain"])
    store_event = method.handle_feedback_delayed(
        lambda _steps, _mode: _Receipt(gain),
        episode_id=episode.episode_id,
    )
    updated = e1mod._update_delayed(
        episode, probe, tuple(int(o) for o in probe["origins"]),
    )
    slot["_episode"] = updated
    promoted = str(updated.local_status) == STATUS_LOCAL_ACTIVE
    approved = str(store_event.get("stage")) == "approved"
    lifecycle: dict[str, Any] | None = None
    if approved:
        snapshot = method._active_snapshot()
        slot["_store"].set_active(snapshot.runtime_bundle_sha)
        slot["_snapshot"] = snapshot
        try:
            snapshot, guards = _patch_lifecycle(
                slot=slot, skill_id=str(draft["skill_id"]),
                plan=dict(slot["_plan"]), probe=probe, episode=updated,
            )
            slot["_snapshot"] = snapshot
            lifecycle = guards
        except Exception as exc:  # noqa: BLE001
            return {
                "promoted": False,
                "store_event": wvc._plain(store_event),
                "episode_before": dict(draft["episode_before"]),
                "episode_after": _episode_view(updated),
                "blocked_at_interface": "EditController patch of risk_guards",
                "reason": "%s: %s" % (type(exc).__name__, exc),
            }
    return {
        "promoted": bool(promoted and approved),
        "store_approved": bool(approved),
        "store_event": wvc._plain(store_event),
        "store_call_site": UPDATE_PATH["store_approval"],
        "promotion_call_site": UPDATE_PATH["promotion"],
        "probe_gain": gain,
        "episode_before": dict(draft["episode_before"]),
        "episode_after": _episode_view(updated),
        "retrievable_skill_id": (
            str(draft["skill_id"]) if promoted and approved else None
        ),
        "lifecycle_fields": lifecycle,
        "reason": (
            None if promoted and approved else
            "e1._update_delayed graded the probe %s at %+.6f, so the Draft was "
            "not promoted" % (str(updated.local_status), gain)
            if approved else
            "handle_feedback_delayed returned stage=%r"
            % store_event.get("stage")
        ),
    }


def _patch_lifecycle(
    *, slot: dict[str, Any], skill_id: str, plan: Mapping[str, Any],
    probe: Mapping[str, Any], episode: Any,
) -> tuple[Any, dict[str, Any]]:
    """Record the lifecycle state on the stored Skill.

    ``skill-entry/1`` has no status field, so the state rides in
    ``risk_guards`` -- the schema's only free-form member -- and is written
    through the EditController rather than by hand.  The excluded-series list
    rides with it because the frozen-steps body carries the program only.
    """
    store = slot["_store"]
    snapshot = slot["_snapshot"]
    skill = next(
        (item for item in snapshot.skills if item.skill_id == skill_id), None
    )
    if skill is None:
        raise ValueError("handle_fast_winner did not write %s" % skill_id)
    guards = dict(skill.risk_guards or {})
    guards.update({
        "local_status": str(episode.local_status),
        "evidence_level": str(episode.evidence_level),
        "evidence_refs": ["artifacts/functional/e2/%s.json" % PROTOCOL_VERSION],
        "source_episode_id": str(episode.episode_id),
        "activation_probe_window": str(PROBE_WINDOW["window_id"]),
        "activation_probe_origins": [int(o) for o in probe["origins"]],
        "activation_probe_gain": float(probe["macro_gain"]),
        "activation_probe_se_block": float(probe["se_block"]),
        "activation_probe_gain_over_se": probe["gain_over_se"],
        "activation_probe_took_part_in_selection": False,
        "frozen_plan": {
            "program": str(plan["program"]),
            "excluded_series": sorted(
                str(uid) for uid in plan["excluded_series"]
            ),
        },
        "provenance": "target_local_skill",
        "current_task_support_confirmation_required": True,
    })
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    parent = store.materialize(snapshot)
    surface = "skill_library.entries/%s.risk_guards" % skill_id
    manifest = EditManifest(
        edit_id="record_lifecycle_%s" % slot["slot"],
        base_harness_sha=snapshot.harness_content_sha,
        target_pattern_id="fresh-confirmation-v1",
        target_surface_id=surface,
        operation=EditOperation.PATCH,
        surface_precondition={
            "kind": "SHA",
            "sha": controller.surface_precondition_sha(parent, surface),
        },
        dependency_precondition_shas={},
        minimal_patch={"value": guards},
        new_value=None,
        observable_applicability=None,
        predicted_agent_behavior_change=("retrieve_skill:%s" % skill_id,),
        predicted_data_effect=("local_improvement",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",),
        patch_id=None,
    )
    receipt = controller.apply_to_fork(
        parent,
        _resolve_apply_manifest(manifest, snapshot),
        confirmed_cause="RISK_GAP",
    )
    updated = receipt.candidate_snapshot.snapshot
    store.set_active(updated.runtime_bundle_sha)
    return updated, guards


# ------------------------------------------------------- stage 2 and stage 4
class Budget:
    def __init__(self, total: int) -> None:
        self.total = int(total)
        self.used = 0

    def take(self, want: int) -> int:
        return max(0, min(int(want), self.total - self.used))

    def spend(self, calls: int) -> None:
        self.used += int(calls)
        if self.used > self.total:
            raise SystemExit("LLM call budget exceeded")

    @property
    def left(self) -> int:
        return self.total - self.used


def _target(variant: str) -> dict[str, Any]:
    return {
        "target_id": "FRESH_%s" % variant,
        "cohort": COHORT_NAME,
        "consumer_variant": variant,
    }


def _adopted_positive(record: Mapping[str, Any]) -> bool:
    plan = record.get("final_plan") or {}
    delayed = record.get("delayed") or {}
    if not plan or str(plan.get("program")) == IDENTITY:
        return False
    gain = delayed.get("aggregate_gain")
    return gain is not None and float(gain) > 0.0


def _trace_row(label: str, record: Mapping[str, Any] | None, retrains: int,
               extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    plan = (record or {}).get("final_plan") or None
    support = (record or {}).get("support") or {}
    delayed = (record or {}).get("delayed") or {}
    row: dict[str, Any] = {
        "step": label,
        "mode": (record or {}).get("mode"),
        "window_id": (record or {}).get("window_id"),
        "adopted_plan": plan,
        "support_aggregate_gain": support.get("aggregate_gain"),
        "delayed_aggregate_gain": delayed.get("aggregate_gain"),
        "harmed_eval_series_count": delayed.get("harmed_eval_series_count"),
        "harmed_eval_series": delayed.get("harmed_eval_series"),
        "harmed_eval_series_total_harm": delayed.get(
            "harmed_eval_series_total_harm"
        ),
        "consumer_retrains": int(retrains),
        "llm_calls": int((record or {}).get("llm_calls") or 0),
        "adopted_delayed_positive": _adopted_positive(record or {}),
    }
    if extra:
        row.update(dict(extra))
    return row


def stage_2(
    payload: Mapping[str, Any], budget: Budget,
) -> dict[str, Any]:
    """The adaptation period, entirely inside 2024."""
    cells: dict[str, Any] = {}
    for variant in CONSUMERS:
        target = _target(variant)
        for arm in ARMS:
            slot_key = "%s_%s" % (arm.lower(), variant)
            slot = STORES[slot_key]
            card_id = SKILL_ID[variant] if arm == "A5" else None
            trace: list[dict[str, Any]] = []

            search_a = FreshSearch(
                payload=payload, consumer_variant=variant,
                support_origins=WINDOWS["task_A"]["support_origins"],
                delayed_origins=WINDOWS["task_A"]["delayed_origins"],
            )
            record_a = _episode(
                search=search_a, target=target, arm=arm,
                window=WINDOWS["task_A"], slot=slot,
                expected_card=card_id, expected_local=None,
                llm_budget=budget.take(LLM_CALL_BUDGET_PER_EPISODE),
                tag="task_A",
            )
            budget.spend(int(record_a.get("llm_calls") or 0))
            trace.append(_trace_row("task_A", record_a, search_a.retrains))
            if not cells and record_a.get("shortlist_payload") is None:
                # The #10 breaker: if the very first episode of the adaptation
                # period returns nothing, the Agent channel is broken and
                # spending the rest of the budget would only bury that.
                cells[slot_key] = {
                    "consumer_variant": variant,
                    "arm": arm,
                    "task_A": record_a,
                    "draft": {"written": False, "reason": "the run stopped here"},
                    "probe": None,
                    "promotion": {"promoted": False, "reason": "the run stopped here"},
                    "task_B": None,
                    "trace": trace,
                    "consumer_retrains": int(search_a.retrains),
                    "llm_calls": int(record_a.get("llm_calls") or 0),
                }
                return {
                    "ran": True,
                    "cells": cells,
                    "update_path": dict(UPDATE_PATH),
                    "stopped_early": (
                        "the first adaptation episode produced no LLM payload"
                    ),
                    "consumer_retrains": int(search_a.retrains),
                    "llm_calls": int(record_a.get("llm_calls") or 0),
                }

            draft = _persist_draft(
                slot=slot, record=record_a, target=target, arm=arm,
            )
            probe: dict[str, Any] | None = None
            promotion: dict[str, Any] = {
                "promoted": False, "reason": draft.get("reason"),
            }
            probe_retrains = 0
            if draft.get("written"):
                probe = _probe(
                    search=search_a, payload=payload, variant=variant,
                    plan=dict(record_a["final_plan"]),
                )
                probe_retrains = int(probe["consumer_retrains"])
                promotion = _promote(slot=slot, probe=probe, draft=draft)
            trace.append(_trace_row(
                "probe", None, probe_retrains,
                extra={
                    "window_id": PROBE_WINDOW["window_id"],
                    "mode": "OUT_OF_SELECTION_PROBE",
                    "probe_macro_gain": (
                        None if probe is None else probe["macro_gain"]
                    ),
                    "harmed_eval_series_count": (
                        None if probe is None
                        else probe["harmed_eval_series_count"]
                    ),
                    "promoted": bool(promotion.get("promoted")),
                },
            ))

            search_b = FreshSearch(
                payload=payload, consumer_variant=variant,
                support_origins=WINDOWS["task_B"]["support_origins"],
                delayed_origins=WINDOWS["task_B"]["delayed_origins"],
            )
            local_id = promotion.get("retrievable_skill_id")
            if local_id:
                record_b = _direct_recall(
                    search=search_b, target=target, arm=arm,
                    window=WINDOWS["task_B"], slot=slot,
                    expected_card=card_id, expected_local=str(local_id),
                    tag="task_B",
                )
                if record_b.get("stopped") == "RECALL_MISS":
                    record_b = _episode(
                        search=search_b, target=target, arm=arm,
                        window=WINDOWS["task_B"], slot=slot,
                        expected_card=card_id, expected_local=str(local_id),
                        llm_budget=budget.take(LLM_CALL_BUDGET_PER_EPISODE),
                        tag="task_B",
                    )
                    record_b["after_recall_miss"] = True
                    budget.spend(int(record_b.get("llm_calls") or 0))
            else:
                record_b = _episode(
                    search=search_b, target=target, arm=arm,
                    window=WINDOWS["task_B"], slot=slot,
                    expected_card=card_id, expected_local=None,
                    llm_budget=budget.take(LLM_CALL_BUDGET_PER_EPISODE),
                    tag="task_B",
                )
                record_b["no_active_local_skill"] = True
                budget.spend(int(record_b.get("llm_calls") or 0))
            trace.append(_trace_row("task_B", record_b, search_b.retrains))

            cells[slot_key] = {
                "consumer_variant": variant,
                "arm": arm,
                "task_A": record_a,
                "draft": draft,
                "probe": probe,
                "promotion": promotion,
                "task_B": record_b,
                "trace": trace,
                "consumer_retrains": int(
                    search_a.retrains + probe_retrains + search_b.retrains
                ),
                "llm_calls": int(
                    (record_a.get("llm_calls") or 0)
                    + (record_b.get("llm_calls") or 0)
                ),
            }
    return {
        "ran": True,
        "cells": cells,
        "update_path": dict(UPDATE_PATH),
        "consumer_retrains": sum(
            int(row["consumer_retrains"]) for row in cells.values()
        ),
        "llm_calls": sum(int(row["llm_calls"]) for row in cells.values()),
    }


def stage_4(
    payload: Mapping[str, Any], adaptation: Mapping[str, Any],
    window: Mapping[str, Any], budget: Budget,
) -> dict[str, Any]:
    """The confirmation task on held-out 2025.  No new Skill is formed."""
    cells: dict[str, Any] = {}
    for variant in CONSUMERS:
        target = _target(variant)
        for arm in ARMS:
            slot_key = "%s_%s" % (arm.lower(), variant)
            slot = STORES[slot_key]
            card_id = SKILL_ID[variant] if arm == "A5" else None
            promotion = adaptation["cells"][slot_key]["promotion"]
            local_id = promotion.get("retrievable_skill_id")
            search = FreshSearch(
                payload=payload, consumer_variant=variant,
                support_origins=window["support_origins"],
                delayed_origins=window["delayed_origins"],
            )
            if local_id:
                record = _direct_recall(
                    search=search, target=target, arm=arm, window=window,
                    slot=slot, expected_card=card_id,
                    expected_local=str(local_id), tag="task_C",
                )
                if record.get("stopped") == "RECALL_MISS":
                    record = _episode(
                        search=search, target=target, arm=arm, window=window,
                        slot=slot, expected_card=card_id,
                        expected_local=str(local_id),
                        llm_budget=budget.take(LLM_CALL_BUDGET_PER_EPISODE),
                        tag="task_C",
                    )
                    record["after_recall_miss"] = True
                    budget.spend(int(record.get("llm_calls") or 0))
            else:
                record = _episode(
                    search=search, target=target, arm=arm, window=window,
                    slot=slot, expected_card=card_id, expected_local=None,
                    llm_budget=budget.take(LLM_CALL_BUDGET_PER_EPISODE),
                    tag="task_C",
                )
                record["no_active_local_skill"] = True
                budget.spend(int(record.get("llm_calls") or 0))
            record["new_skill_formed"] = False
            record["new_skill_note"] = (
                "the confirmation period forms no Skill: handle_fast_winner is "
                "not called here, so this reading cannot feed back into the "
                "store it is measuring"
            )
            cells[slot_key] = {
                "consumer_variant": variant,
                "arm": arm,
                "record": record,
                "consumer_retrains": int(search.retrains),
                "llm_calls": int(record.get("llm_calls") or 0),
                "trace": [_trace_row("task_C", record, search.retrains)],
            }
    return {
        "ran": True,
        "window": {
            key: value for key, value in window.items()
            if not key.startswith("reference_")
        },
        "slow_agent": "off",
        "cells": cells,
        "consumer_retrains": sum(
            int(row["consumer_retrains"]) for row in cells.values()
        ),
        "llm_calls": sum(int(row["llm_calls"]) for row in cells.values()),
    }


# ------------------------------------------------------------- the verdicts
def _first_positive(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cumulative = 0
    for row in trace:
        cumulative += int(row.get("consumer_retrains") or 0)
        if row.get("adopted_delayed_positive"):
            return {
                "reached": True,
                "at_step": str(row["step"]),
                "cumulative_consumer_retrains": cumulative,
                "delayed_aggregate_gain": row.get("delayed_aggregate_gain"),
            }
    return {
        "reached": False,
        "at_step": None,
        "cumulative_consumer_retrains": None,
        "total_spent_without_reaching": cumulative,
    }


def _arm_summary(
    variant: str, arm: str, adaptation: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    key = "%s_%s" % (arm.lower(), variant)
    adapt = adaptation["cells"][key]
    confirm = confirmation["cells"][key]
    trace = list(adapt["trace"]) + list(confirm["trace"])
    record = confirm["record"]
    delayed = record.get("delayed") or {}
    plan = record.get("final_plan")
    readable = bool(
        plan is not None
        and delayed.get("aggregate_gain") is not None
        and np.isfinite(float(delayed["aggregate_gain"]))
    )
    return {
        "consumer_variant": variant,
        "arm": arm,
        "trace": trace,
        "first_positive": _first_positive(trace),
        "total_consumer_retrains": int(
            adapt["consumer_retrains"] + confirm["consumer_retrains"]
        ),
        "adaptation_consumer_retrains": int(adapt["consumer_retrains"]),
        "confirmation_consumer_retrains": int(confirm["consumer_retrains"]),
        "llm_calls": int(adapt["llm_calls"] + confirm["llm_calls"]),
        "task_c_plan": plan,
        "task_c_mode": record.get("mode"),
        "task_c_delayed": (
            None if not readable else float(delayed["aggregate_gain"])
        ),
        "task_c_support": (record.get("support") or {}).get("aggregate_gain"),
        "task_c_harm_count": delayed.get("harmed_eval_series_count"),
        "task_c_harm_series": delayed.get("harmed_eval_series"),
        "task_c_harm_total": delayed.get("harmed_eval_series_total_harm"),
        "task_c_readable": readable,
        "local_skill": adapt["promotion"].get("retrievable_skill_id"),
        "promoted": bool(adapt["promotion"].get("promoted")),
    }


def _cell_verdict(
    variant: str, adaptation: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    a5 = _arm_summary(variant, "A5", adaptation, confirmation)
    a3 = _arm_summary(variant, "A3", adaptation, confirmation)
    if not (a5["task_c_readable"] and a3["task_c_readable"]):
        return {
            "consumer_variant": variant,
            "verdict": "CELL_UNREADABLE",
            "reason": "at least one arm produced no readable task_C delayed number",
            "A5": a5, "A3": a3,
        }
    delta = float(a5["task_c_delayed"]) - float(a3["task_c_delayed"])
    harm5 = int(a5["task_c_harm_count"] or 0)
    harm3 = int(a3["task_c_harm_count"] or 0)
    harm_ok = harm5 <= harm3
    cost5 = a5["total_consumer_retrains"]
    cost3 = a3["total_consumer_retrains"]
    fp5 = a5["first_positive"]["cumulative_consumer_retrains"]
    fp3 = a3["first_positive"]["cumulative_consumer_retrains"]
    fp_cheaper = bool(
        fp5 is not None and (fp3 is None or int(fp5) < int(fp3))
    )
    clause_1 = bool(fp_cheaper and delta >= HARM_THRESHOLD and harm_ok)
    clause_2 = bool(
        delta > MATERIAL_THRESHOLD and cost5 <= cost3 and harm_ok
    )
    same_mechanism = bool(
        a5["task_c_plan"] and a3["task_c_plan"]
        and str(a5["task_c_plan"]["program"]) == str(a3["task_c_plan"]["program"])
    )
    tie = bool(
        same_mechanism
        and abs(delta) <= MATERIAL_THRESHOLD
        and abs(cost5 - cost3) < TIE_COST_WINDOW
    )
    if not harm_ok:
        verdict = "A5_LOSES"
        reason = (
            "A5 harmed %d evaluation series on task_C against A3's %d; harm "
            "over budget is a loss on its own" % (harm5, harm3)
        )
    elif clause_1 or clause_2:
        verdict = "A5_WINS"
        reason = (
            "first-positive cost %s vs %s, task_C delayed difference %+.6f, "
            "harm %d vs %d; clause %s"
            % (
                fp5, fp3, delta, harm5, harm3,
                "1" if clause_1 else "2",
            )
        )
    elif tie:
        verdict = "A5_TIE_TRANSFER_BOUNDARY"
        reason = (
            "both arms adopted %s, |delayed difference| %.6f <= %.3f and "
            "|total cost difference| %d < %d"
            % (
                str(a5["task_c_plan"]["program"]), abs(delta),
                MATERIAL_THRESHOLD, abs(cost5 - cost3), TIE_COST_WINDOW,
            )
        )
    else:
        verdict = "A5_LOSES"
        reason = (
            "neither winning clause held: first-positive cost %s vs %s, "
            "task_C delayed difference %+.6f, total cost %d vs %d, same "
            "mechanism %s" % (fp5, fp3, delta, cost5, cost3, same_mechanism)
        )
    return {
        "consumer_variant": variant,
        "verdict": verdict,
        "reason": reason,
        "task_c_delayed_difference": delta,
        "first_positive_cost": {"A5": fp5, "A3": fp3, "A5_cheaper": fp_cheaper},
        "total_cost": {"A5": cost5, "A3": cost3, "difference": cost5 - cost3},
        "harm": {"A5": harm5, "A3": harm3, "A5_within_A3": harm_ok},
        "same_mechanism": same_mechanism,
        "winning_clause_1": clause_1,
        "winning_clause_2": clause_2,
        "tie_conditions_met": tie,
        "A5": a5,
        "A3": a3,
    }


def _overall(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    pooled = cells[bch.CONSUMER_POOLED]["verdict"]
    channel = cells[bch.CONSUMER_PER_CHANNEL]["verdict"]
    if pooled == "A5_WINS" and channel in ("A5_WINS", "A5_TIE_TRANSFER_BOUNDARY"):
        verdict = "FRESH_A5_DELIVERS"
    elif pooled == "A5_TIE_TRANSFER_BOUNDARY" and channel == "A5_TIE_TRANSFER_BOUNDARY":
        verdict = "FRESH_TRANSFER_BOUNDARY"
    elif "A5_WINS" in (pooled, channel) and (
        "A5_LOSES" in (pooled, channel) or "CELL_UNREADABLE" in (pooled, channel)
    ):
        verdict = "FRESH_MIXED"
    else:
        verdict = "FRESH_A5_FAILS"
    return {
        "verdict": verdict,
        "pooled": pooled,
        "per_channel": channel,
        "primary_cell": bch.CONSUMER_POOLED,
        "reason": (
            "pooled is the primary cell: %s; per_channel: %s" % (pooled, channel)
        ),
    }


def _exposure_ledger(
    *, stage0: Mapping[str, Any], stage3: Mapping[str, Any] | None,
    stage4: Mapping[str, Any] | None, roster: Mapping[str, Any],
) -> dict[str, Any]:
    dev_opened = bool(stage0.get("ran"))
    conf_values = bool(stage3 and stage3.get("verdict") == "MATERIALIZED")
    conf_outcome = bool(stage4 and stage4.get("ran"))
    return {
        "cohort": "noaa_global_hourly_fresh_v1",
        "roster": dict(roster),
        "partitions": {
            "development_2024": {
                "index": [0, DEVELOPMENT_HOURS],
                "instance": "SEEN",
                "outcome": "EXPOSED" if dev_opened else "SEALED",
                "what_opened_it": (
                    "stage 0 identity baselines and the stage 2 adaptation "
                    "episodes fitted Consumers on this partition"
                    if dev_opened else "nothing"
                ),
            },
            "confirmation_2025": {
                "index": [DEVELOPMENT_HOURS, CONFIRMATION_END],
                "instance": "SEEN" if conf_values else "SEALED",
                "outcome": "EXPOSED" if conf_outcome else "SEALED",
                "what_opened_it": (
                    "stage 4 read the confirmation task on this partition"
                    if conf_outcome else
                    "values were materialized but no Consumer was fitted"
                    if conf_values else "nothing"
                ),
            },
            "beyond_17520": {
                "index": [CONFIRMATION_END, None],
                "instance": "SEALED",
                "outcome": "SEALED",
                "what_opened_it": "nothing",
            },
        },
        "family_level_on_record": (
            "AGGREGATE_SEEN: nine old-line NOAA outcome reports and the "
            "40-row frozen registry, quoted from noaa_fresh_cohort_v2"
        ),
        "one_shot": (
            "the confirmation partition is now open. The frozen version has no "
            "second reading: a repaired instrument may not be re-run on it, "
            "only a 0-evaluation deterministic replay of what is already here"
            if conf_outcome else
            "the confirmation partition is still sealed, so the frozen version "
            "remains fresh and may be re-run after a repair"
        ),
    }


# ------------------------------------------------------------------- the run
STORES: dict[str, Any] = {}


def _public(value: Any) -> Any:
    """Strip the live objects the receipts carry so the payload serializes."""
    if isinstance(value, Mapping):
        return {
            str(key): _public(nested)
            for key, nested in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stop(
    payload: dict[str, Any], verdict: str, reason: str, started: float,
) -> dict[str, Any]:
    payload.update({
        "overall_verdict": verdict,
        "overall_verdict_reason": reason,
        "stopped_at": verdict,
        "wall_seconds": time.perf_counter() - started,
    })
    return payload


def run(
    *, stage_0_only: bool = False, offline: bool = False,
) -> int:
    started = time.perf_counter()
    before = _freeze()
    artifact = _cohort_artifact()
    criteria = _readability_criteria(artifact)
    cap = _missing_cap(artifact)
    health = artifact["step_2_health_check_v2"]
    train = [str(uid) for uid in health["confirmation_roster"]]
    evaluation = [str(uid) for uid in health["substitutes"]]

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "open the sealed confirmation domain: compile all-source Guidance "
            "per Consumer, adapt inside 2024, and read the held-out 2025 "
            "confirmation task"
        ),
        "pre_registered": PRE_REGISTERED,
        "roster_split_amendment": ROSTER_SPLIT_AMENDMENT,
        "window_syntax_check": _assert_window_syntax(),
        "windows": {
            name: {
                key: value for key, value in row.items()
                if not key.startswith("reference_")
            }
            for name, row in WINDOWS.items()
        },
        "probe_window": dict(PROBE_WINDOW),
        "frozen_surface_before": before,
        "requested_roster": {"train": train, "eval": evaluation},
        "llm_call_budget": LLM_CALL_BUDGET_TOTAL,
    }

    development = _load_development(train + evaluation)
    dev_payload = _cohort_payload(train, evaluation, development["values"])
    payload["cohort"] = {
        "name": COHORT_NAME,
        "train_uids": list(dev_payload["train_uids"]),
        "eval_uids": list(dev_payload["eval_uids"]),
        "exposure": dev_payload["exposure"],
        "development_records": {
            uid: {
                key: row[key] for key in (
                    "length", "n_finite_development",
                    "missing_rate_development",
                )
            }
            for uid, row in development["records"].items()
        },
    }

    # ---- stage 0 ---------------------------------------------------------
    stage0 = stage_0(dev_payload, criteria)
    payload["stage_0_readability"] = stage0
    payload["guard_after_stage_0"] = _guard(before, "stage 1")
    if not stage0["pass"]:
        payload["exposure_ledger"] = _exposure_ledger(
            stage0=stage0, stage3=None, stage4=None,
            roster=payload["requested_roster"],
        )
        payload["llm_call_count"] = 0
        payload["consumer_retrains_total"] = int(stage0["consumer_retrains"])
        return _write(_stop(
            payload, "JUDGE_UNREADABLE_STOP",
            "the identity Judge is not readable on this cohort; not one 2025 "
            "byte was downloaded",
            started,
        ))
    if stage_0_only:
        payload["llm_call_count"] = 0
        payload["consumer_retrains_total"] = int(stage0["consumer_retrains"])
        return _write(_stop(
            payload, "STAGE_0_ONLY",
            "stage 0 passed; the run was asked to stop here", started,
        ), dry=True)

    # ---- stage 1 ---------------------------------------------------------
    stage1 = stage_1()
    STORES.clear()
    STORES.update(stage1["stores"])
    payload["stage_1_cards_and_stores"] = _public(stage1)
    if stage1["verdict"] != "REGISTERED":
        payload["llm_call_count"] = 0
        payload["consumer_retrains_total"] = int(stage0["consumer_retrains"])
        payload["exposure_ledger"] = _exposure_ledger(
            stage0=stage0, stage3=None, stage4=None,
            roster=payload["requested_roster"],
        )
        return _write(_stop(
            payload, "SCHEMA_BLOCKED",
            "a Guidance card would not fit the store's own schema", started,
        ))

    # ---- stage 1.5 -------------------------------------------------------
    if offline:
        stage15 = {
            "ran": False,
            "verdict": "SKIPPED_OFFLINE",
            "locked_roster": {"train": train, "eval": evaluation},
            "sufficient": True,
        }
    else:
        stage15 = stage_1_5(train, evaluation)
    payload["stage_1_5_existence"] = stage15
    payload["guard_after_stage_1_5"] = _guard(before, "stage 2")
    if not stage15.get("sufficient"):
        payload["llm_call_count"] = 0
        payload["consumer_retrains_total"] = int(stage0["consumer_retrains"])
        payload["exposure_ledger"] = _exposure_ledger(
            stage0=stage0, stage3=None, stage4=None,
            roster=payload["requested_roster"],
        )
        return _write(_stop(
            payload, "INSUFFICIENT_2025_COVERAGE",
            "the locked roster could not be filled from the 2025 listing",
            started,
        ))
    locked = dict(stage15["locked_roster"])
    payload["locked_roster"] = locked
    if (
        sorted(locked["train"]) != sorted(train)
        or sorted(locked["eval"]) != sorted(evaluation)
    ):
        dev_payload = _cohort_payload(
            locked["train"], locked["eval"], development["values"],
        )
        payload["cohort"]["train_uids"] = list(dev_payload["train_uids"])
        payload["cohort"]["eval_uids"] = list(dev_payload["eval_uids"])
        payload["cohort"]["roster_changed_by_cascade"] = True

    # ---- stage 2 ---------------------------------------------------------
    budget = Budget(LLM_CALL_BUDGET_TOTAL)
    stage2 = stage_2(dev_payload, budget)
    payload["stage_2_adaptation"] = _public(stage2)
    payload["guard_after_stage_2"] = _guard(before, "stage 3")
    if stage2.get("stopped_early"):
        payload["llm_call_count"] = budget.used
        payload["consumer_retrains_total"] = int(
            stage0["consumer_retrains"] + stage2["consumer_retrains"]
        )
        payload["exposure_ledger"] = _exposure_ledger(
            stage0=stage0, stage3=None, stage4=None, roster=locked,
        )
        payload["frozen_surface_after"] = _verify(before)
        return _write(_stop(
            payload, "ADAPTATION_STOPPED",
            str(stage2["stopped_early"]) + "; no 2025 byte was downloaded",
            started,
        ))

    # ---- stage 3 ---------------------------------------------------------
    stage3 = stage_3(list(locked["train"]) + list(locked["eval"]))
    confirmation_values = stage3.pop("values")
    payload["stage_3_acquisition"] = _public(stage3)
    if stage3["verdict"] != "MATERIALIZED":
        payload["llm_call_count"] = budget.used
        payload["consumer_retrains_total"] = int(
            stage0["consumer_retrains"] + stage2["consumer_retrains"]
        )
        payload["exposure_ledger"] = _exposure_ledger(
            stage0=stage0, stage3=stage3, stage4=None, roster=locked,
        )
        payload["frozen_surface_after"] = _verify(before)
        return _write(_stop(
            payload, "ACQUISITION_BLOCKED",
            "the 2025 csv could not be acquired for the locked roster; the "
            "in-2024 split alternative is a main-line correction",
            started,
        ))

    extended = {
        uid: np.concatenate([
            development["values"][uid], confirmation_values[uid],
        ])
        for uid in list(locked["train"]) + list(locked["eval"])
    }
    conf_payload = _cohort_payload(locked["train"], locked["eval"], extended)
    gate_c = _missing_gate(extended, conf_payload["eval_uids"], WINDOWS["task_C"], cap)
    window = WINDOWS["task_C"]
    gate_d = None
    if not gate_c["pass"]:
        gate_d = _missing_gate(
            extended, conf_payload["eval_uids"], WINDOWS["task_D"], cap,
        )
        window = WINDOWS["task_D"]
    payload["missing_gate"] = {
        "task_C": gate_c,
        "task_D": gate_d,
        "window_used": str(window["window_id"]),
        "backup_used": bool(gate_d is not None),
        "backup_rule": "task_D is the only backup and is used only here",
    }
    if gate_d is not None and not gate_d["pass"]:
        payload["llm_call_count"] = budget.used
        payload["consumer_retrains_total"] = int(
            stage0["consumer_retrains"] + stage2["consumer_retrains"]
        )
        payload["exposure_ledger"] = _exposure_ledger(
            stage0=stage0, stage3=stage3, stage4=None, roster=locked,
        )
        payload["frozen_surface_after"] = _verify(before)
        return _write(_stop(
            payload, "CONFIRMATION_WINDOW_UNAVAILABLE",
            "neither task_C nor the single backup task_D clears the missing "
            "gate; no 2025 Outcome was opened",
            started,
        ))

    # ---- stage 4 ---------------------------------------------------------
    stage4 = stage_4(conf_payload, stage2, window, budget)
    payload["stage_4_confirmation"] = _public(stage4)
    cells = {
        variant: _cell_verdict(variant, stage2, stage4) for variant in CONSUMERS
    }
    overall = _overall(cells)
    payload.update({
        "cells": _public(cells),
        "overall_verdict": overall["verdict"],
        "overall_verdict_reason": overall["reason"],
        "overall": overall,
        "llm_call_count": budget.used,
        "consumer_retrains_total": int(
            stage0["consumer_retrains"] + stage2["consumer_retrains"]
            + stage4["consumer_retrains"]
        ),
        "exposure_ledger": _exposure_ledger(
            stage0=stage0, stage3=stage3, stage4=stage4, roster=locked,
        ),
        "frozen_surface_after": _verify(before),
        "wall_seconds": time.perf_counter() - started,
    })
    if not payload["frozen_surface_after"]["ok"]:
        payload["overall_verdict"] = "CONCURRENT_WRITE_ABORT"
        payload["overall_verdict_reason"] = (
            "the frozen surface moved during the run; the reading is void"
        )
    return _write(payload)


# --------------------------------------------------------------- the report
def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return "%+.6f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def _plan_label(plan: Mapping[str, Any] | None) -> str:
    if not plan:
        return "--"
    excluded = [str(uid) for uid in plan.get("excluded_series") or ()]
    return "`%s`%s" % (
        plan["program"],
        " full batch" if not excluded else " minus " + ", ".join(excluded),
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Fresh confirmation: NOAA 2024 -> 2025",
        "",
        "**Overall: `%s`** -- %s" % (
            payload.get("overall_verdict"),
            payload.get("overall_verdict_reason", ""),
        ),
        "",
        "One opening of the sealed confirmation domain. Roster, program menu, "
        "recipe compiler, ADOPTION_RULE_V2, the 0.005 material and -0.005 harm "
        "lines, the #10 prompt templates, both Consumers, the Support budget, "
        "the e1v2 triple-window syntax and the lifecycle path were frozen; the "
        "Slow Agent was off throughout.",
        "",
        "## Roster split amendment",
        "",
        "%s. Ruled %s, %s measured value taking part." % (
            payload["roster_split_amendment"]["ruling"],
            payload["roster_split_amendment"]["when"],
            "no" if payload["roster_split_amendment"][
                "zero_measured_value_took_part"
            ] else "a",
        ),
        "",
    ]
    for why in payload["roster_split_amendment"]["why"]:
        lines.append("- %s." % why)
    lines.extend([
        "",
        "%s" % payload["roster_split_amendment"]["instrument_check"],
        "",
        "## Stage 0 -- is the Judge readable",
        "",
        "| Consumer | pass | block | spread | share | min | max |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for variant, row in payload["stage_0_readability"]["per_consumer"].items():
        if "blocks" not in row:
            lines.append(
                "| `%s` | raised | -- | -- | -- | -- | -- |" % variant
            )
            continue
        for name, block in row["blocks"].items():
            lines.append(
                "| `%s` | %s | %s | %s | %.4f | %s | %s |"
                % (
                    variant, block["pass"], name,
                    ("--" if block["eval_loss_spread"] is None
                     else "%.3f" % block["eval_loss_spread"]),
                    block["largest_single_series_loss_share"],
                    ("--" if block["min"] is None else "%.4f" % block["min"]),
                    ("--" if block["max"] is None else "%.4f" % block["max"]),
                )
            )
    criteria = payload["stage_0_readability"]["criteria"]
    lines.extend([
        "",
        "Caps quoted from `%s`: spread <= %.1f, single-series share <= %.2f. "
        "%d Consumer retrains, 0 LLM."
        % (
            criteria["source"], criteria["max_eval_loss_spread"],
            criteria["max_single_series_loss_share"],
            payload["stage_0_readability"]["consumer_retrains"],
        ),
        "",
    ])
    if payload.get("stage_1_cards_and_stores"):
        lines.extend(["## Stage 1 -- the merged Guidance cards", ""])
        for variant, card in payload["stage_1_cards_and_stores"]["cards"].items():
            lines.append(
                "- `%s`: %s, %d clauses %s; bytes sha256 `%s`."
                % (
                    variant, card["status"], card["clause_count"],
                    card["clause_ids"], card["card_bytes_sha256"][:16],
                )
            )
        lines.append("")
        for variant, parity in payload["stage_1_cards_and_stores"][
            "store_parity"
        ].items():
            lines.append(
                "- `%s` store parity outside the skill library: %s."
                % (variant, parity["identical_outside_the_skill_library"])
            )
        lines.append("")
    if payload.get("stage_1_5_existence"):
        row = payload["stage_1_5_existence"]
        lines.extend([
            "## Stage 1.5 -- does 2025 exist",
            "",
            "%s. Checked %s stations, missing %s, %d bytes downloaded."
            % (
                row.get("verdict"), row.get("checked") or 0,
                row.get("missing") if row.get("missing") is not None else "--",
                int(row.get("bytes_downloaded") or 0),
            ),
            "",
        ])
        if row.get("cascade", {}).get("promoted_from_eval_to_train"):
            lines.append(
                "Cascade promoted %s from eval to train."
                % row["cascade"]["promoted_from_eval_to_train"]
            )
            lines.append("")
    if payload.get("cells"):
        lines.extend([
            "## Per cell",
            "",
            "| cell | verdict | first-positive cost A5/A3 | total cost A5/A3 | "
            "task_C delayed A5/A3 | difference | harm A5/A3 |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        ])
        for variant, cell in payload["cells"].items():
            if cell["verdict"] == "CELL_UNREADABLE":
                lines.append(
                    "| `%s` | `CELL_UNREADABLE` | -- | -- | -- | -- | -- |"
                    % variant
                )
                continue
            lines.append(
                "| `%s` | `%s` | %s / %s | %d / %d | %s / %s | %s | %s / %s |"
                % (
                    variant, cell["verdict"],
                    cell["first_positive_cost"]["A5"],
                    cell["first_positive_cost"]["A3"],
                    cell["total_cost"]["A5"], cell["total_cost"]["A3"],
                    _fmt(cell["A5"]["task_c_delayed"]),
                    _fmt(cell["A3"]["task_c_delayed"]),
                    _fmt(cell["task_c_delayed_difference"]),
                    cell["harm"]["A5"], cell["harm"]["A3"],
                )
            )
        lines.extend(["", "## Trajectory", "",
                      "| cell | arm | step | mode | plan | support | delayed | "
                      "harm | retrains | LLM |",
                      "| --- | --- | --- | --- | --- | ---: | ---: | ---: | "
                      "---: | ---: |"])
        for variant, cell in payload["cells"].items():
            for arm in ARMS:
                for row in cell[arm]["trace"]:
                    lines.append(
                        "| `%s` | `%s` | %s | %s | %s | %s | %s | %s | %d | %d |"
                        % (
                            variant, arm, row["step"], row.get("mode") or "--",
                            _plan_label(row.get("adopted_plan")),
                            _fmt(row.get("support_aggregate_gain")),
                            _fmt(
                                row.get("delayed_aggregate_gain")
                                if row.get("probe_macro_gain") is None
                                else row.get("probe_macro_gain")
                            ),
                            ("--" if row.get("harmed_eval_series_count") is None
                             else str(row["harmed_eval_series_count"])),
                            int(row.get("consumer_retrains") or 0),
                            int(row.get("llm_calls") or 0),
                        )
                    )
        lines.append("")
    ledger = payload.get("exposure_ledger")
    if ledger:
        lines.extend([
            "## Exposure ledger",
            "",
            "| partition | index | instance | outcome | opened by |",
            "| --- | --- | --- | --- | --- |",
        ])
        for name, row in ledger["partitions"].items():
            lines.append(
                "| `%s` | %s | `%s` | `%s` | %s |"
                % (
                    name, row["index"], row["instance"], row["outcome"],
                    row["what_opened_it"],
                )
            )
        lines.extend(["", ledger["one_shot"][0].upper() + ledger["one_shot"][1:], ""])
    after = payload.get("frozen_surface_after") or {}
    lines.extend([
        "## Cost and integrity",
        "",
        "- LLM calls: %s / %s." % (
            payload.get("llm_call_count"), payload.get("llm_call_budget")
        ),
        "- Consumer retrains: %s." % payload.get("consumer_retrains_total"),
        "- Frozen surface: %s files, drift %s." % (
            after.get("files"), after.get("drift")
        ),
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


def _write(payload: Mapping[str, Any], *, dry: bool = False) -> int:
    body = _public(payload)
    if dry:
        print(json.dumps(body, indent=2, ensure_ascii=False, default=str))
    else:
        E2.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(
            json.dumps(body, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8", newline="\n",
        )
        OUT_MD.write_text(_markdown(body), encoding="utf-8", newline="\n")
        print("wrote", OUT_JSON, flush=True)
        print("wrote", OUT_MD, flush=True)
    print("overall", body.get("overall_verdict"), flush=True)
    print("llm_calls", body.get("llm_call_count", 0), flush=True)
    print("retrains", body.get("consumer_retrains_total", 0), flush=True)
    return 0 if body.get("overall_verdict") == "FRESH_A5_DELIVERS" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-0-only", action="store_true",
        help="run the readability precheck and stop without writing artifacts",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="skip the 2025 existence check; for rehearsal only",
    )
    args = parser.parse_args(argv)
    try:
        return run(stage_0_only=bool(args.stage_0_only), offline=bool(args.offline))
    except ConcurrentWrite as exc:
        print("CONCURRENT_WRITE_ABORT:", exc, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
