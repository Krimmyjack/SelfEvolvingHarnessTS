"""S1b -- the four-arm evolution curriculum: runner, mechanical course, smoke.

The exam this runner is built for: one Harness walks a frozen sequence of
classification readiness units and the reading asks whether its knowledge state
compounds.  Four arms walk the same course under the same budgets:

    Static     no adaptation at all; identity is frozen and deployed on every
               unit, so only the scoring Consumer fit is spent
    A3-reset   cold start from h0 on every unit; nothing carries between units
    K0-fixed   every unit starts from the *same* pre-course knowledge K0;
               normal held-in adaptation inside the unit; nothing is written
               back between units
    A5-online  same K0 start, same in-unit protocol, but the full Slow
               integration runs between units (including the risk lifecycle
               that commit e64c684 made reachable), so the pool evolves with
               the course

Nothing in ``methods/``, ``runtime/``, ``contracts/`` or ``operators/`` is
touched, and the shared runner ``run_e2_t6_cls_op_shared_harness`` is imported,
never edited.  The one thing this runner owns that the shared runner does not
is the *unit* boundary: cell construction, the domain-binding wall the S1a-r2
audit specified, the cross-unit carry, and the judging component.

Three domain-binding hooks (spec: ``s1a_r2_legal_treatment_audit`` section 5):

1. every Target-local Skill minted inside a unit is stamped, in a runner-owned
   side table, with the ``domain_namespace`` of the unit that minted it;
2. at the unit boundary a Target-local *capability* (frozen program steps, not
   an experience card) whose stamp is not the next unit's domain is dropped
   from the carried snapshot -- the AGENTS.md:184-191 wall;
3. a Source-derived *experience card* is carried into the next unit's Fast
   surface only when the Scope-v1 five-axis predicate holds:
   task_kind x consumer_id x metric x pattern-view intersection x Program
   geometry.  Dataset names are not an axis.

Oracle isolation is enforced, not asserted: ``artifacts/functional/e2/s1_oracle``
is unreadable while any arm is constructing or running.  ``builtins.open`` and
``io.open`` are wrapped once at import; every touch of that directory is logged
with the phase it happened in and *raises* during the arm phase.  The runner
also fires a deliberate probe during the arm phase so the artifact carries
positive proof that the wall was armed rather than merely unvisited.

Entry points::

  python evaluation/functional/run_e2_s1_curriculum_four_arms.py --select-curriculum
  python evaluation/functional/run_e2_s1_curriculum_four_arms.py --smoke

``--select-curriculum`` reads the sealed census and oracle artifacts, selects
the seven units by the declared mechanical rules, freezes both orders and
writes ``s1_curriculum_frozen.json/.md``.  ``--smoke`` runs curriculum unit 1
only, at a reduced protocol, and writes ``s1_smoke_cell1.json/.md``.  The full
course is S1c and is deliberately not an entry point here.

Evidence grade: DEVELOPMENT.
"""
from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import shutil
import subprocess
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

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
ORACLE_DIR = E2 / "s1_oracle"
ORACLE_TOKEN = "s1_oracle"


# =========================================================================== #
# oracle isolation -- installed before anything else can read a key
# =========================================================================== #
class OracleIsolationBreach(RuntimeError):
    """An arm-phase read of the sealed oracle directory.  Never caught."""


PHASE_SETUP = "setup"
PHASE_ARM = "arm"
PHASE_JUDGE = "judge"
PHASE_SELECT = "select"

_PHASE: dict[str, Any] = {"name": PHASE_SETUP, "unit": None, "arm": None}
_ORACLE_ACCESS: list[dict[str, Any]] = []
_REAL_BUILTIN_OPEN = builtins.open
_REAL_IO_OPEN = io.open


def _is_oracle_path(target: Any) -> bool:
    try:
        text = os.fspath(target)
    except TypeError:
        return False
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return ("/%s/" % ORACLE_TOKEN) in str(text).replace("\\", "/")


def _guarded(real: Any) -> Any:
    def wrapper(file: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_oracle_path(file):
            blocked = _PHASE["name"] == PHASE_ARM
            _ORACLE_ACCESS.append({
                "path": str(file),
                "phase": _PHASE["name"],
                "unit": _PHASE["unit"],
                "arm": _PHASE["arm"],
                "blocked": blocked,
                "probe": bool(_PHASE.get("probe")),
            })
            if blocked:
                raise OracleIsolationBreach(
                    "arm phase (unit=%s arm=%s) tried to open a sealed oracle "
                    "key: %s" % (_PHASE["unit"], _PHASE["arm"], file))
        return real(file, *args, **kwargs)
    return wrapper


def _guarded_method(real: Any) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_oracle_path(self):
            blocked = _PHASE["name"] == PHASE_ARM
            _ORACLE_ACCESS.append({
                "path": str(self),
                "phase": _PHASE["name"],
                "unit": _PHASE["unit"],
                "arm": _PHASE["arm"],
                "blocked": blocked,
                "probe": bool(_PHASE.get("probe")),
            })
            if blocked:
                raise OracleIsolationBreach(
                    "arm phase (unit=%s arm=%s) tried to read a sealed oracle "
                    "key: %s" % (_PHASE["unit"], _PHASE["arm"], self))
        return real(self, *args, **kwargs)
    return wrapper


builtins.open = _guarded(_REAL_BUILTIN_OPEN)      # type: ignore[assignment]
io.open = _guarded(_REAL_IO_OPEN)                 # type: ignore[assignment]
os.open = _guarded(os.open)                       # type: ignore[assignment]
# ``pathlib`` does not always route through the module-level ``io.open`` this
# runner can rebind, so the three Path readers are wrapped by name as well.
# The S1b smoke found this hole: without it a ``Path.read_text`` of an oracle
# key inside an arm was neither blocked nor logged.
Path.open = _guarded_method(Path.open)            # type: ignore[assignment]
Path.read_text = _guarded_method(Path.read_text)  # type: ignore[assignment]
Path.read_bytes = _guarded_method(Path.read_bytes)  # type: ignore[assignment]


def _set_phase(name: str, *, unit: Any = None, arm: Any = None) -> None:
    _PHASE["name"] = name
    _PHASE["unit"] = unit
    _PHASE["arm"] = arm


def _oracle_guard_selftest() -> dict[str, Any]:
    """Prove the wall is armed by walking into it on purpose, on every reader
    surface it claims to cover."""
    victims = sorted(ORACLE_DIR.glob("*.json"))
    if not victims:
        return {"fired": False, "reason": "no oracle key on disk"}
    victim = victims[0]
    surfaces: dict[str, Any] = {}
    _PHASE["probe"] = True
    try:
        for name, call in (
            ("builtins.open", lambda: open(victim, encoding="utf-8").close()),
            ("pathlib.Path.read_text",
             lambda: victim.read_text(encoding="utf-8")),
            ("pathlib.Path.open", lambda: victim.open(encoding="utf-8").close()),
        ):
            try:
                call()
            except OracleIsolationBreach:
                surfaces[name] = "blocked"
            else:
                surfaces[name] = "LEAKED"
    finally:
        _PHASE["probe"] = False
    return {
        "fired": all(value == "blocked" for value in surfaces.values()),
        "target": victim.name,
        "surfaces": surfaces,
    }


def _oracle_isolation_report() -> dict[str, Any]:
    arm_rows = [row for row in _ORACLE_ACCESS if row["phase"] == PHASE_ARM]
    leaked = [row for row in arm_rows if not row["blocked"]]
    return {
        "mechanism": (
            "builtins.open, io.open, os.open, Path.open, Path.read_text and "
            "Path.read_bytes are wrapped at module import; any path containing "
            "artifacts/functional/e2/s1_oracle/ raises OracleIsolationBreach "
            "while the phase is 'arm'"),
        "arm_phase_attempts": len(arm_rows),
        "arm_phase_attempts_blocked": len(arm_rows) - len(leaked),
        "arm_phase_leaks": leaked,
        "deliberate_probe_rows": [row for row in arm_rows if row["probe"]],
        "judge_phase_keys_read": sorted({
            Path(row["path"]).name for row in _ORACLE_ACCESS
            if row["phase"] == PHASE_JUDGE and not row["blocked"]}),
        "unblocked_reads_by_phase": {
            phase: len({Path(row["path"]).name for row in _ORACLE_ACCESS
                        if row["phase"] == phase and not row["blocked"]})
            for phase in (PHASE_SETUP, PHASE_SELECT, PHASE_JUDGE)},
        "holds": not leaked,
    }


# =========================================================================== #
# imports that are allowed to see everything except the oracle
# =========================================================================== #
import numpy as np  # noqa: E402

import run_e2_s1a_curriculum_oracle_audit as s1a  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.operators.registry import (  # noqa: E402
    OPERATOR_NAMES,
)

Stop = cls.Stop
FitBudget = cls.FitBudget

# =========================================================================== #
# frozen protocol
# =========================================================================== #
PROTOCOL_VERSION = "s1_curriculum_four_arms_v1"
EVIDENCE_GRADE = "DEVELOPMENT"
CURRICULUM_NAME = "S1 four-arm evolution curriculum"

CENSUS_JSON = E2 / "s1a_r3_pool_census.json"
FROZEN_JSON = E2 / "s1_curriculum_frozen.json"
FROZEN_MD = E2 / "s1_curriculum_frozen.md"
SMOKE_JSON = E2 / "s1_smoke_cell1.json"
SMOKE_MD = E2 / "s1_smoke_cell1.md"

ARM_STATIC = "Static"
ARM_A3 = "A3-reset"
ARM_K0 = "K0-fixed"
ARM_A5 = "A5-online"
ARMS = (ARM_STATIC, ARM_A3, ARM_K0, ARM_A5)
ADAPTIVE_ARMS = (ARM_A3, ARM_K0, ARM_A5)

CONDITION = s1a.CONDITION                 # fit_only_artifact
CONSUMER_ID = s1a.CONSUMER_ID             # ridge-raw-plus-difference-v1
METRIC = s1a.METRIC                       # accuracy
TASK_KIND = s1a.TASK_KIND                 # classification
DATA_DIR = cls.DATA_DIR
MATERIAL = cls.MATERIAL                   # 0.005
FRACTION_SCOPE = "cohort"                 # matches the sealed oracle legality
HELD_IN_ROUNDS = ("r1", "r2")
SMOKE_ROUNDS = ("r1",)

# budgets (frozen by the task book)
LLM_PER_UNIT_PER_ARM = 15
FIT_PER_UNIT_PER_ARM = 25
LLM_PER_SLOW_INTEGRATION = 6
LLM_TOTAL_CAP = 400
FIT_TOTAL_CAP = 900
WALL_SECONDS_CAP = 3 * 60 * 60
SMOKE_LLM_PER_ARM = 5
SMOKE_WALL_SECONDS_CAP = 30 * 60

# selection rule constants
HARM_OPERATORS = ("outlier_mad", "outlier_iqr")
HARM_MODIFIED_FRACTION_CAP = 0.10
HARM_HEADROOM_BAR = -0.005
GROUP_HARM = "harm_evidence"
GROUP_LEARNABLE = "learnable_positive"
GROUP_IDENTITY = "identity"
GROUP_HELDOUT_ONLY = "heldout_only_temptation"
FORWARD_TEMPLATE = (
    (GROUP_HARM, 0), (GROUP_LEARNABLE, 0), (GROUP_HARM, 1),
    (GROUP_IDENTITY, 0), (GROUP_LEARNABLE, 1), (GROUP_HELDOUT_ONLY, 0),
    (GROUP_IDENTITY, 1),
)

SOURCE_SKILL_ID = cls.SOURCE_SKILL_ID
K0_CARD_SOURCE = E2 / "t6_cls_op_r2_three_arms.json"

PRE_REGISTERED_READOUT = {
    "primary": (
        "A5-online must be non-inferior to both A3-reset and K0-fixed on "
        "quality (cumulative held-out utility) and on harm (harm events, "
        "worst-class harm), and must improve at least one of cumulative "
        "regret or cumulative total cost by a material margin"),
    "material_threshold": MATERIAL,
    "verdict_ceiling_for_a_single_order_single_run": (
        "S1_DEVELOPMENT_EVOLUTION_SIGNAL"),
    "judged_in": "S1c.  This book freezes the readout and does not judge it.",
}


# =========================================================================== #
# small helpers
# =========================================================================== #
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT,
                              capture_output=True, text=True,
                              check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _plain(value: Any) -> Any:
    return cls._plain(value)


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")


_OPS_BY_LENGTH = tuple(sorted(OPERATOR_NAMES, key=len, reverse=True))


def _ops_in(text: Any) -> list[str]:
    """Operator names named by a candidate id or workflow signature."""
    blob = str(text or "")
    found: list[str] = []
    for name in _OPS_BY_LENGTH:
        if name in blob and not any(name in seen for seen in found):
            found.append(name)
    return found


def _pattern_view_of(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Deployment-visible binned Pattern view.  No oracle, no dataset name."""
    block = np.asarray(cell["observation_block"], dtype=np.float64)
    return s1a._pattern_view(s1a._binned_public_features(block))


# =========================================================================== #
# Part 1 -- mechanical curriculum selection
# =========================================================================== #
CURRICULUM_REVISION = "r2"
SLICE_FLOOR_LADDER = (5, 4, 3)

SELECTION_RULES = {
    "revision": (
        "r2.  The r1 rule ranked every group by smallest total points, which "
        "selected *against* the feedback surface: six of seven r1 units had a "
        "held-in slice of at most two rows and three had an empty r2 delayed "
        "slice, so no relation but NEUTRAL was reachable and the guard "
        "channel could never compile.  r2 replaces the ranking with a "
        "readability floor plus a readability ranking.  The r1 course is kept "
        "at s1_curriculum_frozen_r1.json/.md."),
    "source_artifacts": [
        "artifacts/functional/e2/s1a_r3_pool_census.json (family_key, "
        "learnability, held-in headroom)",
        "artifacts/functional/e2/s1_oracle/*.json (oracle set, per-operator "
        "legality / cohort modified fraction / held-in headroom, and "
        "cell.slice_rows -- the four held-in slice sizes the oracle pass "
        "already recorded, so the readability screen costs zero new fits)",
    ],
    "min_slice_rows": (
        "min(cell.slice_rows) over r1_support, r1_delayed, r2_support, "
        "r2_delayed: the coarsest surface the frozen two-round protocol will "
        "read on that unit.  A slice of n rows moves accuracy only in steps "
        "of 1/n"),
    "slice_readability_floor": (
        "admission gate for all four groups: min_slice_rows >= L, L walking "
        "the ladder %s.  A group that cannot fill its quota at L steps down "
        "one rung and the step is written into ladder_trace; nothing is "
        "reselected silently." % (list(SLICE_FLOOR_LADDER),)),
    "necessary_condition": (
        "|key held-in readout| >= 1 / min_slice_rows, computed from the "
        "sealed oracle numbers already on disk.  The key readout is the one "
        "the group is defined by: for harm units the largest-magnitude "
        "qualifying outlier harm, for learnable units the oracle program's "
        "held-in headroom.  See necessary_condition_scope for why the other "
        "two groups are not screened by it."),
    "necessary_condition_scope": (
        "the condition binds only on the two groups whose defining held-in "
        "readout is non-zero (harm, learnable).  identity units are defined "
        "by an identity oracle set and HELDOUT_ONLY units by a held-in "
        "reading of zero, so |readout| >= 1/n is unsatisfiable for them by "
        "construction and applying it literally empties both groups at every "
        "rung of the ladder -- see literal_application_counterfactual.  For "
        "those two groups the informative requirement is that a material "
        "reading *would* have been visible had one existed, which is exactly "
        "the slice floor.  Flagged for main-line confirmation."),
    "within_group_ranking": (
        "descending min_slice_rows; ties broken by descending |key held-in "
        "readout|, then ascending unit_id.  No outcome may reorder."),
    "family_deduplication": (
        "cross-course, not per-group: the seven units should carry seven "
        "distinct family_key values.  When a group cannot fill its quota "
        "without a repeat, the repeat is taken best-ranked-first and named "
        "in family_census.repeated_families and in the group's "
        "ladder_trace."),
    "relaxation_ladder": (
        "per group, in order: (floor 5, 4, 3 with strict cross-course family "
        "distinctness), then (floor 5, 4, 3 with family repeats allowed but "
        "still preferring a fresh family within each rung).  The first rung "
        "that fills the quota wins.  The floor is relaxed before family "
        "distinctness is *not* the order: keeping the floor high is the whole "
        "point of r2, so the strict-family rungs are tried across the whole "
        "ladder first, and only then are repeats allowed starting again at "
        "floor 5.  Every rung tried is recorded."),
    "group_selection_order": (
        "harm -> learnable -> identity -> HELDOUT_ONLY.  Earlier groups "
        "consume families and units; the order is fixed here, before any "
        "candidate is scored."),
    GROUP_HARM: (
        "an oracle-scored unit qualifies when outlier_mad or outlier_iqr is "
        "legal on it (verifier passed and cohort modified fraction <= 0.10) "
        "and its held-in headroom is <= -0.005, i.e. materially harmful on "
        "held-in.  Two units, ranked by within_group_ranking."),
    GROUP_LEARNABLE: (
        "learnability == LEARNABLE.  Two units, ranked by "
        "within_group_ranking."),
    GROUP_IDENTITY: (
        "oracle_set is exactly identity (or empty).  Two units, ranked by "
        "within_group_ranking."),
    GROUP_HELDOUT_ONLY: (
        "learnability == HELDOUT_ONLY.  One unit, ranked by "
        "within_group_ranking."),
    "unit_disjointness": "no unit id may appear twice in the course",
    "forward_order": (
        "harm A -> learnable A -> harm B -> identity A -> learnable B -> "
        "HELDOUT_ONLY -> identity B.  Design intent: the guard should be "
        "compilable after the second harm unit, so every unit after it tests "
        "whether the guard actually fires."),
    "reverse_order": "the exact reverse of the forward order",
    "domain_namespace": (
        "unit_id (dataset__injection).  The whole course runs at one "
        "condition (fit_only_artifact), so dataset alone would collapse two "
        "curriculum units of the same substrate into one counted Task and "
        "the guard census (risk_skill._task_of) would undercount."),
}


def _census_units() -> dict[str, dict[str, Any]]:
    payload = json.loads(CENSUS_JSON.read_text(encoding="utf-8"))
    return {str(row["unit_id"]): dict(row) for row in payload["units"]}


SLICE_NAMES = ("r1_support", "r1_delayed", "r2_support", "r2_delayed")


def _oracle_unit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    programs = {str(row["program"]): row for row in payload["programs"]}
    cell = dict(payload.get("cell") or {})
    points = int(cell.get("official_train_rows") or 0) * int(
        payload.get("series_length") or 0)
    slice_rows = {name: int(value) for name, value
                  in (cell.get("slice_rows") or {}).items()}
    smallest = min(slice_rows.values()) if slice_rows else 0
    best_legal = 0.0
    for row in programs.values():
        headroom = row.get("heldin_headroom")
        if row.get("legal") and headroom is not None:
            best_legal = max(best_legal, abs(float(headroom)))
    harmful_ops: list[dict[str, Any]] = []
    for op in HARM_OPERATORS:
        row = programs.get(op)
        if row is None:
            continue
        headroom = row.get("heldin_headroom")
        if not row.get("legal"):
            continue
        if float(row.get("cohort_modified_fraction") or 0.0) > HARM_MODIFIED_FRACTION_CAP:
            continue
        if headroom is None or float(headroom) > HARM_HEADROOM_BAR:
            continue
        harmful_ops.append({
            "program": op,
            "heldin_headroom": float(headroom),
            "heldout_utility": row.get("heldout_utility"),
            "cohort_modified_fraction": row.get("cohort_modified_fraction"),
        })
    return {
        "unit_id": str(payload["unit_id"]),
        "dataset": str(payload["dataset"]),
        "injection": str(payload["injection"]),
        "series_length": payload.get("series_length"),
        "official_train_rows": cell.get("official_train_rows"),
        "fit_rows": payload.get("fit_rows"),
        "n_heldin": payload.get("n_heldin"),
        "n_heldout": payload.get("n_heldout"),
        "total_points": points,
        "slice_rows": {name: slice_rows.get(name) for name in SLICE_NAMES},
        "min_slice_rows": smallest,
        "slice_resolution": (1.0 / smallest) if smallest else None,
        "largest_legal_heldin_magnitude": best_legal,
        "oracle_set": [str(item) for item in (payload.get("oracle_set") or [])],
        "oracle_set_empty": bool(payload.get("oracle_set_empty")),
        "menu_oracle_program": payload.get("menu_oracle_program"),
        "menu_oracle_heldout_utility": payload.get(
            "menu_oracle_heldout_utility"),
        "heldin_material_line": payload.get("heldin_material_line"),
        "heldout_material_line": payload.get("heldout_material_line"),
        "class_harm_bar": payload.get("class_harm_bar"),
        "harmful_outlier_operators": harmful_ops,
    }


def _candidate_table() -> list[dict[str, Any]]:
    census = _census_units()
    rows: list[dict[str, Any]] = []
    for path in sorted(ORACLE_DIR.glob("*.json")):
        row = _oracle_unit(path)
        seen = census.get(row["unit_id"], {})
        row["family_key"] = seen.get("family_key")
        row["learnability"] = seen.get("learnability")
        row["census_heldin_headroom"] = seen.get("heldin_headroom")
        row["census_heldout_utility"] = seen.get("heldout_utility")
        rows.append(row)
    missing = [row["unit_id"] for row in rows if not row["family_key"]]
    if missing:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "sealed oracle units absent from the r3 census: %s" % missing)
    return rows


# Which groups the |readout| >= 1/min_slice necessary condition can bind on.
# identity and HELDOUT_ONLY are defined by a held-in reading of zero, so the
# condition is unsatisfiable for them by construction.
NECESSARY_CONDITION_GROUPS = (GROUP_HARM, GROUP_LEARNABLE)


def _key_readout(group: str, row: Mapping[str, Any]) -> float:
    """The held-in reading the group is defined by, in magnitude."""
    if group == GROUP_HARM:
        return max((abs(float(item["heldin_headroom"]))
                    for item in row["harmful_outlier_operators"]), default=0.0)
    if group == GROUP_LEARNABLE:
        return abs(float(row["census_heldin_headroom"] or 0.0))
    return 0.0


def _group_pool(group: str, rows: Sequence[Mapping[str, Any]]
                ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if group == GROUP_HARM:
            keep = bool(row["harmful_outlier_operators"])
        elif group == GROUP_LEARNABLE:
            keep = row["learnability"] == "LEARNABLE"
        elif group == GROUP_IDENTITY:
            keep = bool(row["oracle_set"] == ["identity"]
                        or row["oracle_set_empty"] or not row["oracle_set"])
        else:
            keep = row["learnability"] == "HELDOUT_ONLY"
        if not keep:
            continue
        candidate = dict(row)
        candidate["group"] = group
        candidate["key_heldin_readout"] = _key_readout(group, row)
        out.append(candidate)
    return out


def _rank_key(row: Mapping[str, Any]) -> tuple[int, float, str]:
    return (-int(row["min_slice_rows"]), -float(row["key_heldin_readout"]),
            str(row["unit_id"]))


def _admits(row: Mapping[str, Any], *, floor: int, screened: bool) -> bool:
    if int(row["min_slice_rows"]) < floor:
        return False
    if not screened:
        return True
    resolution = row["slice_resolution"]
    if resolution is None:
        return False
    return float(row["key_heldin_readout"]) >= float(resolution)


def _fill_rung(candidates: Sequence[Mapping[str, Any]], *, quota: int,
               used_units: set[str], used_families: set[str],
               allow_repeat: bool) -> list[dict[str, Any]]:
    """One rung: fresh families first, then repeats when the rung allows it."""
    ranked = sorted(candidates, key=_rank_key)
    picked: list[dict[str, Any]] = []
    families = set(used_families)
    for row in ranked:
        if row["unit_id"] in used_units or len(picked) >= quota:
            continue
        if str(row["family_key"]) in families:
            continue
        entry = dict(row)
        entry["family_repeat"] = False
        picked.append(entry)
        families.add(str(row["family_key"]))
    if allow_repeat:
        taken = {row["unit_id"] for row in picked}
        for row in ranked:
            if len(picked) >= quota:
                break
            if row["unit_id"] in used_units or row["unit_id"] in taken:
                continue
            entry = dict(row)
            entry["family_repeat"] = True
            entry["family_repeat_of"] = str(row["family_key"])
            picked.append(entry)
            taken.add(row["unit_id"])
    return sorted(picked, key=_rank_key)


def _select_group(group: str, rows: Sequence[Mapping[str, Any]], *,
                  quota: int, used_units: set[str], used_families: set[str]
                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Walk the declared relaxation ladder and stop at the first rung that
    fills the quota.  Every rung tried is recorded."""
    pool = _group_pool(group, rows)
    screened = group in NECESSARY_CONDITION_GROUPS
    trace: list[dict[str, Any]] = []
    chosen: list[dict[str, Any]] = []
    chosen_rung: dict[str, Any] | None = None
    for allow_repeat in (False, True):
        for floor in SLICE_FLOOR_LADDER:
            admitted = [row for row in pool
                        if _admits(row, floor=floor, screened=screened)]
            picked = _fill_rung(admitted, quota=quota, used_units=used_units,
                                used_families=used_families,
                                allow_repeat=allow_repeat)
            rung = {
                "slice_floor": floor,
                "family_repeats_allowed": allow_repeat,
                "admitted_units": len(admitted),
                "filled": len(picked),
                "quota": quota,
                "picked": [row["unit_id"] for row in picked],
            }
            trace.append(rung)
            if len(picked) >= quota:
                chosen = picked
                chosen_rung = rung
                break
        if chosen_rung is not None:
            break
    if chosen_rung is None and trace:
        # nothing filled the quota anywhere; keep the best partial fill
        best = max(trace, key=lambda rung: rung["filled"])
        floor = int(best["slice_floor"])
        admitted = [row for row in pool
                    if _admits(row, floor=floor, screened=screened)]
        chosen = _fill_rung(admitted, quota=quota, used_units=used_units,
                            used_families=used_families,
                            allow_repeat=bool(best["family_repeats_allowed"]))
        chosen_rung = best
    literal = [row for row in pool
               if _admits(row, floor=SLICE_FLOOR_LADDER[-1], screened=True)]
    return chosen, {
        "group": group,
        "quota": quota,
        "pool_size": len(pool),
        "necessary_condition_applied": screened,
        "ladder_trace": trace,
        "rung_used": chosen_rung,
        "downgraded_from_floor_5": bool(
            chosen_rung and int(chosen_rung["slice_floor"]) != SLICE_FLOOR_LADDER[0]),
        "family_repeats_used": [row["unit_id"] for row in chosen
                                if row.get("family_repeat")],
        "short_by": max(0, quota - len(chosen)),
        "literal_application_counterfactual": {
            "note": ("how many of this group's units would survive if the "
                     "|readout| >= 1/min_slice condition were applied "
                     "literally at the lowest floor on the ladder"),
            "surviving_units": len(literal),
        },
    }


def select_curriculum() -> dict[str, Any]:
    _set_phase(PHASE_SELECT)
    rows = _candidate_table()
    used: set[str] = set()
    used_families: set[str] = set()
    shortfalls: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    ladder: dict[str, Any] = {}

    for group, quota in ((GROUP_HARM, 2), (GROUP_LEARNABLE, 2),
                         (GROUP_IDENTITY, 2), (GROUP_HELDOUT_ONLY, 1)):
        picked, report = _select_group(
            group, rows, quota=quota, used_units=used,
            used_families=used_families)
        groups[group] = picked
        ladder[group] = report
        used.update(str(row["unit_id"]) for row in picked)
        used_families.update(str(row["family_key"]) for row in picked)
        if report["downgraded_from_floor_5"] or report["family_repeats_used"]:
            shortfalls.append({
                "group": group,
                "found": len(picked),
                "fallback": (
                    "quota could not be filled at slice floor %d with strict "
                    "cross-course family distinctness; the rung actually used "
                    "was floor %d with family repeats %s%s"
                    % (SLICE_FLOOR_LADDER[0],
                       int(report["rung_used"]["slice_floor"]),
                       "allowed" if report["rung_used"][
                           "family_repeats_allowed"] else "forbidden",
                       (" -- repeated: %s" % report["family_repeats_used"])
                       if report["family_repeats_used"] else "")),
            })
        if report["short_by"]:
            shortfalls.append({
                "group": group,
                "found": len(picked),
                "fallback": ("quota still short by %d at the bottom of the "
                             "ladder; reported short rather than reselected"
                             % report["short_by"]),
            })
    harm_pool = _group_pool(GROUP_HARM, rows)
    learn_pool = _group_pool(GROUP_LEARNABLE, rows)
    identity_pool = _group_pool(GROUP_IDENTITY, rows)
    heldout_pool = _group_pool(GROUP_HELDOUT_ONLY, rows)
    forward: list[dict[str, Any]] = []
    for group, index in FORWARD_TEMPLATE:
        members = groups[group]
        if index >= len(members):
            continue
        row = dict(members[index])
        row["group"] = group
        row["group_index"] = index
        forward.append(row)
    for position, row in enumerate(forward, start=1):
        row["forward_position"] = position
    reverse = list(reversed(forward))

    families = [str(row["family_key"]) for row in forward]
    datasets = [str(row["dataset"]) for row in forward]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "curriculum_revision": CURRICULUM_REVISION,
        "supersedes": "artifacts/functional/e2/s1_curriculum_frozen_r1.json",
        "curriculum_name": CURRICULUM_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": _git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "selection_rules": SELECTION_RULES,
        "candidate_counts": {
            "oracle_units_available": len(rows),
            "harm_qualifiers": len(harm_pool),
            "learnable_available": len(learn_pool),
            "identity_available": len(identity_pool),
            "heldout_only_available": len(heldout_pool),
            "units_passing_slice_floor": {
                str(floor): len([row for row in rows
                                 if int(row["min_slice_rows"]) >= floor])
                for floor in SLICE_FLOOR_LADDER},
        },
        "ladder_trace_by_group": ladder,
        "selected_groups": {
            group: [{"unit_id": row["unit_id"],
                     "family_key": row["family_key"],
                     "family_repeat": bool(row.get("family_repeat")),
                     "min_slice_rows": row["min_slice_rows"],
                     "slice_rows": row["slice_rows"],
                     "slice_resolution": row["slice_resolution"],
                     "key_heldin_readout": row["key_heldin_readout"],
                     "necessary_condition_holds": _admits(
                         row, floor=1,
                         screened=group in NECESSARY_CONDITION_GROUPS),
                     "total_points": row["total_points"],
                     "learnability": row["learnability"],
                     "oracle_set": row["oracle_set"],
                     "heldin_headroom": row["census_heldin_headroom"],
                     "harmful_outlier_operators": row["harmful_outlier_operators"],
                     "why": _why(group, row)}
                    for row in members]
            for group, members in groups.items()},
        "forward_order": [row["unit_id"] for row in forward],
        "reverse_order": [row["unit_id"] for row in reverse],
        "units": [{
            "forward_position": row["forward_position"],
            "unit_id": row["unit_id"],
            "dataset": row["dataset"],
            "injection": row["injection"],
            "condition": CONDITION,
            "consumer": CONSUMER_ID,
            "metric": METRIC,
            "group": row["group"],
            "family_key": row["family_key"],
            "family_repeat": bool(row.get("family_repeat")),
            "learnability": row["learnability"],
            "slice_rows": row["slice_rows"],
            "min_slice_rows": row["min_slice_rows"],
            "slice_resolution": row["slice_resolution"],
            "key_heldin_readout": row["key_heldin_readout"],
            "largest_legal_heldin_magnitude": row[
                "largest_legal_heldin_magnitude"],
            "total_points": row["total_points"],
            "official_train_rows": row["official_train_rows"],
            "series_length": row["series_length"],
            "n_heldin": row["n_heldin"],
            "n_heldout": row["n_heldout"],
            "oracle_set": row["oracle_set"],
            "menu_oracle_program": row["menu_oracle_program"],
            "harmful_outlier_operators": row["harmful_outlier_operators"],
        } for row in forward],
        "family_census": {
            "families_in_course": sorted(set(families)),
            "distinct_families": len(set(families)),
            "repeated_families": sorted(
                {name for name in families if families.count(name) > 1}),
            "repeated_substrates": sorted(
                {name for name in datasets if datasets.count(name) > 1}),
        },
        "shortfalls": shortfalls,
        "budgets": {
            "llm_per_unit_per_adaptive_arm": LLM_PER_UNIT_PER_ARM,
            "fit_per_unit_per_arm": FIT_PER_UNIT_PER_ARM,
            "llm_per_slow_integration": LLM_PER_SLOW_INTEGRATION,
            "llm_total_cap": LLM_TOTAL_CAP,
            "fit_total_cap": FIT_TOTAL_CAP,
            "wall_seconds_cap": WALL_SECONDS_CAP,
            "over_cap_verdict": "COMPUTE_BUDGET_EXCEEDED",
        },
        "arms": {
            ARM_STATIC: ("no adaptation; identity frozen and deployed on every "
                         "unit; only the scoring fit is spent"),
            ARM_A3: "cold start from h0 on every unit; zero carry between units",
            ARM_K0: ("every unit starts from the same K0; normal in-unit "
                     "held-in adaptation; no write-back between units"),
            ARM_A5: ("same K0 start and in-unit protocol; full Slow "
                     "integration between units including the risk lifecycle; "
                     "the pool evolves with the course"),
        },
        "k0_definition": _k0_definition(),
        "domain_binding_hooks": {
            "1_stamp": SELECTION_RULES["domain_namespace"],
            "2_carry_wall": (
                "a Target-local capability (frozen program steps; not an "
                "experience card) is dropped at the unit boundary unless its "
                "stamp equals the next unit's domain_namespace"),
            "3_scope_v1": (
                "a Source-derived experience card reaches the next unit's "
                "Fast surface only when task_kind, consumer_id and metric "
                "match, its authorizing pattern-view intersection is "
                "non-empty and is satisfied by the next unit's binned "
                "deployment-visible pattern view, and its Program geometry "
                "is a real operator.  Dataset name is not an axis."),
        },
        "pre_registered_readout": PRE_REGISTERED_READOUT,
        "not_in_this_book": (
            "the full course is not run here.  This entry freezes the course "
            "and the readout only; S1c runs it."),
        "oracle_isolation": _oracle_isolation_report(),
    }
    return payload


def _why(group: str, row: Mapping[str, Any]) -> str:
    surface = ("smallest held-in slice %d rows, so the surface resolves "
               "%.4f" % (int(row["min_slice_rows"]),
                         float(row["slice_resolution"] or 0.0)))
    repeat = ("; family repeat, named because the group could not fill its "
              "quota with a fresh family" if row.get("family_repeat") else "")
    if group == GROUP_HARM:
        ops = ", ".join("%s(held-in %+.4f)" % (item["program"],
                                               item["heldin_headroom"])
                        for item in row["harmful_outlier_operators"])
        return ("materially harmful legal outlier program on held-in: %s.  %s, "
                "and the harm magnitude %.4f clears it%s"
                % (ops, surface, float(row["key_heldin_readout"]), repeat))
    if group == GROUP_LEARNABLE:
        return ("LEARNABLE, held-in headroom %+.4f.  %s, and the headroom "
                "clears it%s" % (float(row["census_heldin_headroom"] or 0.0),
                                 surface, repeat))
    if group == GROUP_IDENTITY:
        return ("oracle set is identity, so the correct end state is to change "
                "nothing.  %s, so 'nothing helps' is a reading rather than a "
                "blind spot; the largest legal held-in magnitude on this unit "
                "is %.4f%s" % (surface,
                               float(row["largest_legal_heldin_magnitude"] or 0.0),
                               repeat))
    return ("HELDOUT_ONLY: the oracle-set program helps held-out (%+.4f) but "
            "held-in cannot approve it (headroom %+.4f) -- the abstention "
            "temptation.  %s, so the held-in zero is measured, not missing%s"
            % (float(row["census_heldout_utility"] or 0.0),
               float(row["census_heldin_headroom"] or 0.0), surface, repeat))


def _k0_definition() -> dict[str, Any]:
    card = _k0_card()
    return {
        "base": "methods/ttha/harness/h0 (the three bootstrap Skills)",
        "bootstrap_skills": ["inspect_and_localize",
                             "build_contrastive_candidates",
                             "select_or_identity_and_verify"],
        "inert_slow_card": {
            "skill_id": card["skill_id"],
            "source": K0_CARD_SOURCE.relative_to(PROJECT_ROOT).as_posix(),
            "try_clause": (card["risk_guards"].get("sections") or {}).get("TRY"),
            "allowed_tools": list(card.get("allowed_tools") or []),
            "carries_frozen_steps": "Frozen program steps" in str(card.get("body")),
        },
        "excluded_on_purpose": (
            "the C40 Target-local hampel capability is NOT in K0.  It is a "
            "frozen-steps capability bound to one Source domain; placing it in "
            "K0 would leak an answer across domains and contaminate both "
            "K0-fixed and A5-online."),
    }


# =========================================================================== #
# Part 2 -- K0 compilation
# =========================================================================== #
def _k0_card() -> dict[str, Any]:
    """The already-audited inert Slow card.  Zero LLM: it is read, not made."""
    if not K0_CARD_SOURCE.is_file():
        raise Stop("INSTRUMENT_UNREADABLE",
                   "K0 card source missing: %s" % K0_CARD_SOURCE)
    payload = json.loads(K0_CARD_SOURCE.read_text(encoding="utf-8"))

    def _find(node: Any) -> dict[str, Any] | None:
        if isinstance(node, Mapping):
            if node.get("skill_id") == SOURCE_SKILL_ID and "risk_guards" in node:
                return dict(node)
            for value in node.values():
                hit = _find(value)
                if hit:
                    return hit
        elif isinstance(node, list):
            for item in node:
                hit = _find(item)
                if hit:
                    return hit
        return None

    entry = _find(payload)
    if entry is None:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "no %s entry in %s" % (SOURCE_SKILL_ID, K0_CARD_SOURCE))
    return entry


def _assert_k0_is_clean(card: Mapping[str, Any]) -> dict[str, Any]:
    tools = list(card.get("allowed_tools") or [])
    frozen = "Frozen program steps" in str(card.get("body") or "")
    guards = dict(card.get("risk_guards") or {})
    try_text = str((guards.get("sections") or {}).get("TRY") or "")
    clean = (not tools) and (not frozen)
    if not clean:
        raise Stop("K0_CONTAMINATED",
                   "the K0 card carries an execution right (tools=%s frozen=%s)"
                   % (tools, frozen))
    return {
        "no_allowed_tools": not tools,
        "no_frozen_program_steps": not frozen,
        "try_abstains": try_text == "NO_AUTHORIZED_ACTIVE_RECOMMENDATION",
        "no_target_local_capability_in_k0": True,
    }


_SKILL_ENTRY_FIELDS = frozenset({
    "schema_version", "skill_id", "skill_kind", "revision", "body",
    "observable_applicability", "allowed_tools", "risk_guards",
})


def _apply_entries(base: Any, entries: Sequence[Mapping[str, Any]],
                   *, store_root: Path, tag: str) -> tuple[Any, list[str]]:
    """Add skill entries onto a base snapshot through the frozen controller."""
    from SelfEvolvingHarnessTS.contracts.harness import (
        EditManifest, EditOperation,
    )
    from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
        EditController, FaultRouter, SurfaceRegistry,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (
        _resolve_apply_manifest,
    )

    root = store_root / tag
    if root.exists():
        shutil.rmtree(root)
    store = SnapshotStore(root / "snapshots")
    store.materialize(base)
    store.set_active(base.runtime_bundle_sha)
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    snapshot = base
    applied: list[str] = []
    for entry in entries:
        skill_id = str(entry["skill_id"])
        applicability = _plain(dict(entry.get("observable_applicability") or {}))
        manifest = EditManifest(
            edit_id=skill_id,
            base_harness_sha=snapshot.harness_content_sha,
            target_pattern_id="s1-curriculum-carry",
            target_surface_id="skill_library.entries/" + skill_id,
            operation=EditOperation.ADD,
            surface_precondition={"kind": "ABSENT"},
            dependency_precondition_shas={},
            new_value=_plain({key: value for key, value in entry.items()
                              if key in _SKILL_ENTRY_FIELDS}),
            observable_applicability=applicability or None,
            predicted_agent_behavior_change=("retrieve_skill:" + skill_id,),
            predicted_data_effect=("carried_knowledge",),
            automatically_selected_risk_cases=(),
            falsification_condition=("no_improvement",),
            patch_id=None,
        )
        receipt = controller.apply_to_fork(
            store.materialize(snapshot),
            _resolve_apply_manifest(manifest, snapshot),
            confirmed_cause="SKILL_LIBRARY_GAP")
        snapshot = receipt.candidate_snapshot.snapshot
        store.set_active(snapshot.runtime_bundle_sha)
        applied.append(skill_id)
    return snapshot, applied


def compile_k0(store_root: Path) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )

    h0 = compile_snapshot(PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    card = _k0_card()
    purity = _assert_k0_is_clean(card)
    k0, applied = _apply_entries(h0, [card], store_root=store_root, tag="k0")
    return {
        "h0": h0,
        "k0": k0,
        "card": card,
        "purity": purity,
        "applied": applied,
        "h0_sha": h0.runtime_bundle_sha,
        "k0_sha": k0.runtime_bundle_sha,
        "h0_skill_ids": sorted(s.skill_id for s in h0.skills),
        "k0_skill_ids": sorted(s.skill_id for s in k0.skills),
    }


# =========================================================================== #
# Part 3 -- arm state machine
# =========================================================================== #
def _entry_of(skill: Any) -> dict[str, Any]:
    return {
        "schema_version": skill.schema_version,
        "skill_id": str(skill.skill_id),
        "skill_kind": str(getattr(skill.skill_kind, "value", skill.skill_kind)),
        "revision": int(skill.revision),
        "body": str(skill.body),
        "observable_applicability": _plain(dict(skill.observable_applicability)),
        "allowed_tools": [str(tool) for tool in (skill.allowed_tools or ())],
        "risk_guards": _plain(dict(skill.risk_guards or {})),
    }


def _classify_skill(entry: Mapping[str, Any]) -> str:
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _parse_frozen_steps,
    )

    kind = str(entry.get("skill_kind") or "")
    if kind == "bootstrap_procedure":
        return "bootstrap"
    if kind == "safety":
        return "risk_guard"
    guards = dict(entry.get("risk_guards") or {})
    if isinstance(guards.get("sections"), Mapping):
        return "experience_card"
    if _parse_frozen_steps(str(entry.get("body") or "")) is not None:
        return "target_local_capability"
    return "other"


def _new_state(*, snapshot: Any, agent: Any, store_root: Path, tag: str,
               episodes: Sequence[Any] = ()) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
        EditController, FaultRouter, SurfaceRegistry,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod

    root = store_root / tag
    if root.exists():
        shutil.rmtree(root)
    store = SnapshotStore(root / "snapshots")
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    method = TTHAMethod(agent, snapshot, tuple(episodes))
    return {
        "store": store,
        "controller": EditController(store, surfaces=SurfaceRegistry(),
                                     router=FaultRouter()),
        "method": method,
        "incumbent": None,
        "approved_skill_ids": [],
        "tag": tag,
        "base_sha": snapshot.runtime_bundle_sha,
        "base_skill_ids": sorted(str(s.skill_id) for s in snapshot.skills),
        "episodes_at_start": len(tuple(episodes)),
    }


def _run_round(*, state: Mapping[str, Any], cell: Mapping[str, Any],
               unit_id: str, round_name: str, arm: str,
               fit_budget: FitBudget, ledger: Any) -> dict[str, Any]:
    """One held-in round.  Identical to the shared runner's body except that
    the Episode's ``domain`` is the *unit* id, not dataset/condition.

    Repair 1 of commit e64c684 writes ``context_summary.task_episode_id`` from
    this ``domain`` argument, and the guard census counts distinct values of
    that string.  The whole course runs at one condition, so dataset/condition
    would collapse two curriculum units of one substrate into one counted Task
    and no repeated harm could ever reach the two-distinct-Task floor.
    """
    from SelfEvolvingHarnessTS.contracts.method import PreparationRequest
    from SelfEvolvingHarnessTS.methods.ttha import signed_radius as resolver
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
        MEASURED_EFFECT_KEY, task_consumer_key,
    )
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
        activate_approved, open_delayed, run_online_round,
    )
    from consumers.cls_scope_adapter import (
        DELAYED, SUPPORT, ClassificationConsumerAdapter,
    )

    block = np.asarray(cell["observation_block"], dtype=np.float64)
    support_origin = int(block.size)
    delayed_origin = support_origin + 1
    heldout_origin = support_origin + 2
    surfaces = {
        SUPPORT: cell["surfaces"]["%s_support" % round_name],
        DELAYED: cell["surfaces"]["%s_delayed" % round_name],
    }
    adapter = ClassificationConsumerAdapter(
        fit_values=cell["fit_values"], fit_labels=cell["fit_labels"],
        surfaces=surfaces, delayed_origin=delayed_origin,
        heldout_origin=heldout_origin, budget=fit_budget,
        ridge_alpha=cls.RIDGE_ALPHA, allowed_surfaces=(SUPPORT, DELAYED))
    executor = cls._ClsScopeExecutor(
        cell=cell, evaluate_fn=adapter,
        max_modified_fraction=float(
            cls._task_context().deployment_constraints.maximum_modified_fraction),
        modification_fraction_scope=FRACTION_SCOPE)
    values = {"heldin_observation": block}
    observed = dict(resolver.window_context(values, support_origin,
                                            cls.PERIOD_HINT))
    observed["bound_period"] = float(cls.PERIOD_HINT)
    request = PreparationRequest(
        "s1-%s" % unit_id, block, cls._task_spec(), dict(observed),
        task_context=cls._task_context())
    cls._assert_behaviour_context(request)
    features = dict(extract_public_features(block, task_kind=TASK_KIND))

    method = state["method"]
    method.bind_round_data(block, task_kind=TASK_KIND)
    started = time.time()
    llm_before = int(getattr(ledger, "calls", 0) or 0)
    skills_before = {str(s.skill_id) for s in method._active_snapshot().skills}
    result = run_online_round(
        method, executor, request, values,
        origin=support_origin, slow_agent=None,
        controller=state["controller"], store=state["store"],
        card_builder=cls._card_builder,
        round_name="%s_%s_%s" % (arm.lower(), unit_id, round_name),
        budget=cls.SUPPORT_TRIAL_BUDGET, allow_slow=False,
        domain=unit_id,
        period=cls.PERIOD_HINT, fast_features=features,
        allow_fast_skill=True, runtime_prior_slot=False)
    open_delayed(result, executor, delayed_origin=delayed_origin,
                 store=state["store"])
    activated = False
    if result.approved_skill_id is not None:
        activated = activate_approved(result, state["store"])
        if activated:
            state["approved_skill_ids"].append(str(result.approved_skill_id))
    risk_lifecycle = cls._risk_lifecycle(state, arm=arm)

    trace = method.last_trace
    fresh_ids = set(result.episode_ids)
    fresh = [e for e in method.experience_episodes if e.episode_id in fresh_ids]
    if result.winner_program is not None:
        state["incumbent"] = _plain(result.winner_program)
    skills_after = {str(s.skill_id) for s in method._active_snapshot().skills}
    minted = sorted(skills_after - skills_before)
    for skill_id in minted:
        state.setdefault("domain_stamp", {})[skill_id] = unit_id

    probes = []
    for probe in result.actual_probed_programs:
        probes.append({
            "candidate_id": probe.get("candidate_id"),
            "kind": probe.get("kind"),
            "gain": probe.get("gain"),
            "passed": probe.get("passed"),
            "operators": _ops_in(probe.get("candidate_id")),
        })
    retrieved = [str(item) for item in
                 (getattr(trace, "retrieved_skill_ids", ()) or ())]
    guarded_ops = sorted({op for skill_id in retrieved
                          if skill_id.startswith("target_risk_")
                          for op in _ops_in(skill_id)})
    probed_ops = sorted({op for probe in probes for op in probe["operators"]})
    pool_ops = sorted({op for cand in (getattr(trace, "candidate_ids", ()) or ())
                       for op in _ops_in(cand)})
    return {
        "unit_id": unit_id,
        "round": round_name,
        "arm": arm,
        "dataset": cell["dataset"],
        "domain_namespace": unit_id,
        "task_consumer_key": task_consumer_key(cls._task_spec()),
        "pool": [str(item) for item in
                 (getattr(trace, "candidate_ids", ()) or ())],
        "chosen": getattr(trace, "chosen_candidate_id", None),
        "retrieved_skill_ids": retrieved,
        "memory_resolution": getattr(trace, "memory_resolution_status", None),
        "proposal_count": result.proposal_count,
        "support_receipts": result.target_support_receipts_used,
        "probes": probes,
        "winner_program": _plain(result.winner_program),
        "abstained": bool(result.abstained),
        "harm_count": int(result.harm_count),
        "delayed_utility": result.delayed_utility,
        "approved_skill_id": result.approved_skill_id,
        "activated": activated,
        "minted_skill_ids": minted,
        "risk_lifecycle": _plain(risk_lifecycle),
        "guard_readout": {
            "guarded_operators_in_fast_view": guarded_ops,
            "pool_operators": pool_ops,
            "probed_operators": probed_ops,
            "guarded_and_not_probed": [op for op in guarded_ops
                                       if op not in probed_ops],
            "guarded_but_probed_anyway": [op for op in guarded_ops
                                          if op in probed_ops],
        },
        "episodes": [{
            "episode_id": e.episode_id,
            "domain_namespace": e.domain_namespace,
            "task_episode_id": (
                (getattr(e, "context_summary", None) or {}).get(
                    "task_episode_id")),
            "workflow_signature": e.workflow_signature,
            "relation": e.relation,
            "evidence_level": e.evidence_level,
            "local_status": e.local_status,
            "support_gain": (e.support_response or {}).get("gain"),
            "delayed_gain": (e.delayed_response or {}).get("gain"),
            "support_effect": _plain(
                (e.support_response or {}).get(MEASURED_EFFECT_KEY)),
            "delayed_effect": _plain(
                (e.delayed_response or {}).get(MEASURED_EFFECT_KEY)),
        } for e in fresh],
        "consumer_fits_after": fit_budget.used,
        "llm_calls_this_round": (
            int(getattr(ledger, "calls", 0) or 0) - llm_before),
        "candidate_executions": len(result.actual_probed_programs),
        "seconds": round(time.time() - started, 2),
    }


def _build_cell(unit: Mapping[str, Any]) -> dict[str, Any]:
    cell, reason = s1a._r3_build_cell({
        "dataset": unit["dataset"], "injection": unit["injection"],
        "series_length": unit.get("series_length"),
    })
    if cell is None:
        raise Stop("CELL_CONSTRUCTION_FAILED",
                   "%s: %s" % (unit["unit_id"], reason))
    return cell


def run_unit(*, unit: Mapping[str, Any], cell: Mapping[str, Any], arm: str,
             base_snapshot: Any, carried_episodes: Sequence[Any],
             agent_factory: Any, backend: Any, store_root: Path,
             rounds: Sequence[str], fit_cap: int,
             carried_stamps: Mapping[str, str] | None = None
             ) -> dict[str, Any]:
    unit_id = str(unit["unit_id"])
    _set_phase(PHASE_ARM, unit=unit_id, arm=arm)
    fit_budget = FitBudget(fit_cap)
    llm_before = int(getattr(backend, "calls", 0) or 0)
    started = time.time()
    state = _new_state(
        snapshot=base_snapshot,
        agent=agent_factory(cell["observation_block"],
                            backend.new_arm_backend()),
        store_root=store_root, tag="%s_%s" % (arm.replace("-", "_"), unit_id),
        episodes=carried_episodes)
    state["domain_stamp"] = dict(carried_stamps or {})
    records: list[dict[str, Any]] = []
    if arm != ARM_STATIC:
        for round_name in rounds:
            records.append(_run_round(
                state=state, cell=cell, unit_id=unit_id,
                round_name=round_name, arm=arm, fit_budget=fit_budget,
                ledger=backend))
    deployment = cls._deploy_and_score(
        state=state, cell=cell, arm=arm, fit_budget=fit_budget)
    end_snapshot = state["method"]._active_snapshot()  # noqa: SLF001
    added = [_entry_of(skill) for skill in end_snapshot.skills
             if str(skill.skill_id) not in set(state["base_skill_ids"])]
    for entry in added:
        entry["carrier_kind"] = _classify_skill(entry)
        entry["domain_namespace"] = state.get("domain_stamp", {}).get(
            entry["skill_id"], unit_id)
    episodes = list(state["method"].experience_episodes)
    return {
        "unit_id": unit_id,
        "arm": arm,
        "base_runtime_bundle_sha": state["base_sha"],
        "end_runtime_bundle_sha": end_snapshot.runtime_bundle_sha,
        "store_evolved": end_snapshot.runtime_bundle_sha != state["base_sha"],
        "episodes_at_unit_start": state["episodes_at_start"],
        "episodes_at_unit_end": len(episodes),
        "base_skill_ids": state["base_skill_ids"],
        "end_skill_ids": sorted(str(s.skill_id) for s in end_snapshot.skills),
        "skills_added_in_unit": added,
        "rounds": records,
        "deployment": deployment,
        "approved_skill_ids": list(state["approved_skill_ids"]),
        "llm_calls": int(getattr(backend, "calls", 0) or 0) - llm_before,
        "consumer_fits": fit_budget.used,
        "consumer_fit_cap": fit_cap,
        "seconds": round(time.time() - started, 2),
        "_state": state,
        "_episodes": episodes,
        "_end_snapshot": end_snapshot,
    }


# =========================================================================== #
# Part 4 -- the unit boundary: Slow integration and the domain-binding wall
# =========================================================================== #
def _a5_probe_rows(unit_results: Sequence[Mapping[str, Any]],
                   cells: Mapping[str, Mapping[str, Any]]
                   ) -> list[dict[str, Any]]:
    """Census rows for the Slow stage, one per legal non-identity probe."""
    rows: list[dict[str, Any]] = []
    for result in unit_results:
        unit_id = str(result["unit_id"])
        condition = bool((cells.get(unit_id) or {}).get(
            cls.CENSUS_CONDITION_KEY))
        for record in result["rounds"]:
            guided = bool(record["guard_readout"]["guarded_operators_in_fast_view"]
                          or [sid for sid in record["retrieved_skill_ids"]
                              if sid not in (SOURCE_SKILL_ID,)
                              and not sid.startswith("bootstrap")])
            for episode in record["episodes"]:
                signature = str(episode["workflow_signature"])
                if signature in ("identity", "unknown"):
                    continue
                rows.append({
                    "task_episode_id": unit_id,
                    "arm": ARM_A5,
                    "program": signature,
                    "context_condition": condition,
                    "support_gain": episode["support_gain"],
                    "relation": str(episode["relation"]),
                    "conditioned_snapshot": guided,
                    "conditioned_served": guided,
                })
    return rows


def slow_integration(*, unit_results: Sequence[Mapping[str, Any]],
                     cells: Mapping[str, Mapping[str, Any]],
                     llm_ledger: dict[str, int], live: bool,
                     pattern_views: Mapping[str, Mapping[str, Any]]
                     ) -> dict[str, Any]:
    """The between-unit Slow pass for A5-online.  Deterministic audit always;
    the authoring LLM call only when ``live``."""
    probes = _a5_probe_rows(unit_results, cells)
    census = cls._census_rows(probes)
    before = int(llm_ledger.get("slow", 0))
    consolidation = cls._consolidate_source_skill(
        census=census, probes=probes, llm_ledger=llm_ledger, live=live)
    spent = int(llm_ledger.get("slow", 0)) - before
    if spent > LLM_PER_SLOW_INTEGRATION:
        raise Stop("COMPUTE_BUDGET_EXCEEDED",
                   "slow integration spent %d LLM, cap %d"
                   % (spent, LLM_PER_SLOW_INTEGRATION))
    entry = cls._source_skill_entry(consolidation)
    scope = None
    if entry is not None:
        scope = _scope_v1_of(consolidation, probes, pattern_views)
    return {
        "probe_rows": probes,
        "census": census,
        "authorization_audit": consolidation["authorization_audit"],
        "authorized_try_operators": consolidation["authorized_try_operators"],
        "risk_authorized_operators": consolidation["risk_authorized_operators"],
        "skill_written": consolidation["skill_written"],
        "execution_right_granted": consolidation["execution_right_granted"],
        "slow_llm_calls": spent,
        "slow_llm_cap": LLM_PER_SLOW_INTEGRATION,
        "entry": entry,
        "scope_v1": scope,
    }


def _scope_v1_of(consolidation: Mapping[str, Any],
                 probes: Sequence[Mapping[str, Any]],
                 pattern_views: Mapping[str, Mapping[str, Any]]
                 ) -> dict[str, Any]:
    """The runner-owned five-axis Scope for a freshly minted Source card.

    ``authorization_audit`` reports counts, not the Task ids behind them, so
    the supporting units are recovered from the same probe rows the audit read:
    unguided POSITIVE evidence on an authorized TRY operator.  With no
    authorized operator the card is advisory only; the Scope is then built from
    every unguided POSITIVE unit and the card stays inert regardless.
    """
    authorized = set(consolidation["authorized_try_operators"])
    supporting = sorted({
        str(probe["task_episode_id"]) for probe in probes
        if str(probe["relation"]) == "POSITIVE"
        and not probe["conditioned_snapshot"]
        and (not authorized
             or (set(_ops_in(probe["program"])) & authorized))})
    views = [dict(pattern_views[uid]) for uid in supporting
             if uid in pattern_views]
    intersection: dict[str, Any] = {}
    if views:
        keys = set(views[0])
        for view in views[1:]:
            keys &= {key for key in view if view[key] == views[0][key]}
        intersection = {key: views[0][key] for key in sorted(keys)}
    return {
        "task_kind": TASK_KIND,
        "consumer_id": CONSUMER_ID,
        "metric": METRIC,
        "pattern_intersection": intersection,
        "program_geometry": sorted(authorized)[:1],
        "supporting_units": supporting,
    }


def _scope_v1_admits(scope: Mapping[str, Any] | None,
                     next_pattern: Mapping[str, Any]) -> dict[str, Any]:
    if not scope:
        return {"admits": False, "why": "no scope recorded"}
    reasons: list[str] = []
    if scope.get("task_kind") != TASK_KIND:
        reasons.append("task_kind")
    if scope.get("consumer_id") != CONSUMER_ID:
        reasons.append("consumer_id")
    if scope.get("metric") != METRIC:
        reasons.append("metric")
    intersection = dict(scope.get("pattern_intersection") or {})
    if not intersection:
        reasons.append("empty_pattern_intersection")
    else:
        mismatched = [key for key, value in intersection.items()
                      if next_pattern.get(key) != value]
        if mismatched:
            reasons.append("pattern_mismatch:" + ",".join(sorted(mismatched)))
    geometry = list(scope.get("program_geometry") or [])
    if not geometry:
        reasons.append("no_program_geometry")
    elif any(op not in OPERATOR_NAMES for op in geometry):
        reasons.append("unknown_program_geometry")
    return {"admits": not reasons, "why": reasons or ["all five axes match"]}


def carry_decision(entry: Mapping[str, Any], *,
                   next_unit_id: str) -> dict[str, Any]:
    """Hook 2, as one function so the smoke can probe it directly."""
    kind = entry.get("carrier_kind") or _classify_skill(entry)
    stamp = entry.get("domain_namespace")
    if kind == "target_local_capability":
        keep = stamp == next_unit_id
        why = ("hook 2: a Target-local capability carries frozen program steps "
               "and is valid only in the domain that formed it; stamped %s, "
               "next unit is %s" % (stamp, next_unit_id))
    elif kind == "risk_guard":
        keep = True
        why = ("Slow compilation product (structured avoid guard); carried as "
               "knowledge, applicability decided by the frozen retrieval "
               "predicate")
    elif kind == "experience_card":
        keep = True
        why = "Source-derived experience card; Fast visibility decided by hook 3"
    else:
        keep = True
        why = "not a Target-local card"
    return {"skill_id": entry.get("skill_id"), "carrier_kind": kind,
            "domain_namespace": stamp, "carried": keep, "why": why}


def carry_into_next_unit(*, unit_result: Mapping[str, Any],
                         integration: Mapping[str, Any] | None,
                         next_unit_id: str,
                         next_pattern: Mapping[str, Any],
                         k0: Any, store_root: Path, tag: str
                         ) -> dict[str, Any]:
    """The three domain-binding hooks, applied at the unit boundary."""
    decisions: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    for entry in unit_result["skills_added_in_unit"]:
        decision = carry_decision(entry, next_unit_id=next_unit_id)
        decisions.append(decision)
        if decision["carried"]:
            survivors.append(entry)

    new_card = (integration or {}).get("entry")
    scope = (integration or {}).get("scope_v1")
    card_decision: dict[str, Any] | None = None
    if new_card is not None:
        verdict = _scope_v1_admits(scope, next_pattern)
        inert = _card_is_inert(new_card)
        keep = bool(verdict["admits"] or inert)
        card_decision = {
            "skill_id": str(new_card["skill_id"]),
            "carrier_kind": "experience_card",
            "scope_v1": scope,
            "scope_v1_admits": verdict["admits"],
            "scope_v1_why": verdict["why"],
            "inert_on_every_fast_surface": inert,
            "carried": keep,
            "why": ("hook 3: a non-inert Source card is admitted only when the "
                    "five-axis Scope matches the next unit; an inert card "
                    "authorizes nothing on any Fast surface and is carried as "
                    "Slow-only knowledge"),
        }
        decisions.append(card_decision)
        if keep and str(new_card["skill_id"]) not in {
                e["skill_id"] for e in survivors}:
            survivors.append(dict(new_card))

    k0_ids = {str(skill.skill_id) for skill in k0.skills}
    entries = [entry for entry in survivors if entry["skill_id"] not in k0_ids]
    snapshot, applied = _apply_entries(k0, entries, store_root=store_root,
                                       tag=tag)
    return {
        "next_unit_id": next_unit_id,
        "decisions": decisions,
        "carried_skill_ids": applied,
        "dropped_skill_ids": [row["skill_id"] for row in decisions
                              if not row["carried"]],
        "snapshot": snapshot,
        "runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "source_card_decision": card_decision,
    }


def _card_is_inert(entry: Mapping[str, Any]) -> bool:
    from types import SimpleNamespace

    from SelfEvolvingHarnessTS.contracts.harness import SkillKind
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (
        _is_inert_experience_card,
    )

    skill = SimpleNamespace(
        risk_guards=dict(entry.get("risk_guards") or {}),
        observable_applicability=dict(entry.get("observable_applicability") or {}),
        skill_kind=SkillKind.CAPABILITY,
    )
    return bool(_is_inert_experience_card(skill))  # type: ignore[arg-type]


def _probe_entry(*, skill_id: str, stamp: str, op: str) -> dict[str, Any]:
    """A Target-local-capability-shaped entry, built the way the live path
    builds one (``_source_skill_entry``: body + 'Frozen program steps')."""
    return {
        "schema_version": "skill-entry/1",
        "skill_id": skill_id,
        "skill_kind": "capability",
        "revision": 1,
        "body": ("Target-local capability formed on one unit.\n"
                 "Frozen program steps: "
                 + json.dumps([{"op": op, "params": {}}])),
        "observable_applicability": {"feature": "task_kind", "op": "==",
                                     "value": TASK_KIND},
        "allowed_tools": [op],
        "risk_guards": {"requires_target_support": False},
        "domain_namespace": stamp,
    }


def domain_binding_probe(*, this_unit_id: str, next_unit_id: str,
                         next_pattern: Mapping[str, Any],
                         this_pattern: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise hooks 2 and 3 on purpose.

    Unit 1 of the frozen course produced no POSITIVE Support, so no
    Target-local capability and no Source card were minted and the two walls
    would otherwise pass vacuously.  These probes are runner-owned synthetic
    entries put through the *same* decision functions the live boundary uses;
    they prove the wall's behaviour, not the arms'.
    """
    foreign = _probe_entry(skill_id="probe_target_local_foreign_domain",
                           stamp=this_unit_id, op="denoise_median")
    native = _probe_entry(skill_id="probe_target_local_native_domain",
                          stamp=next_unit_id, op="denoise_median")
    foreign_decision = carry_decision(foreign, next_unit_id=next_unit_id)
    native_decision = carry_decision(native, next_unit_id=next_unit_id)

    matching_scope = {
        "task_kind": TASK_KIND, "consumer_id": CONSUMER_ID, "metric": METRIC,
        "pattern_intersection": dict(next_pattern),
        "program_geometry": ["hampel_filter"],
        "supporting_units": [this_unit_id, next_unit_id],
    }
    differing = {key: value for key, value in this_pattern.items()
                 if next_pattern.get(key) != value}
    mismatching_scope = dict(matching_scope)
    mismatching_scope["pattern_intersection"] = (
        dict(differing) if differing
        else {"period_evidence_status": "__no_such_bin__"})
    empty_scope = dict(matching_scope)
    empty_scope["pattern_intersection"] = {}
    wrong_consumer = dict(matching_scope)
    wrong_consumer["consumer_id"] = "some-other-consumer-v1"
    return {
        "note": ("synthetic entries through the live decision functions; the "
                 "arms minted none of these"),
        "hook_2": {
            "carrier_kind_classifier": _classify_skill(foreign),
            "foreign_domain_capability": foreign_decision,
            "native_domain_capability": native_decision,
            "behaves_as_specified": (not foreign_decision["carried"]
                                     and native_decision["carried"]),
        },
        "hook_3": {
            "matching_five_axis_scope": _scope_v1_admits(matching_scope,
                                                         next_pattern),
            "pattern_mismatch_scope": _scope_v1_admits(mismatching_scope,
                                                       next_pattern),
            "empty_intersection_scope": _scope_v1_admits(empty_scope,
                                                         next_pattern),
            "wrong_consumer_scope": _scope_v1_admits(wrong_consumer,
                                                     next_pattern),
            "pattern_axes_that_differ_between_unit_1_and_unit_2":
                sorted(differing),
            "behaves_as_specified": bool(
                _scope_v1_admits(matching_scope, next_pattern)["admits"]
                and not _scope_v1_admits(empty_scope, next_pattern)["admits"]
                and not _scope_v1_admits(wrong_consumer,
                                         next_pattern)["admits"]),
        },
    }


def instrument_census(course: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How fine a reading each frozen unit's held-in surface can even express.

    ``classify_relation`` needs an aggregate accuracy move of at least
    ``MATERIAL`` to call anything POSITIVE or NEGATIVE.  A Support slice of n
    rows can only move accuracy in steps of 1/n, so a slice of one or two rows
    reports 0.0 for every candidate and no relation but NEUTRAL is reachable.
    """
    rows: list[dict[str, Any]] = []
    for unit in course:
        try:
            cell = _build_cell(unit)
        except Stop as stop:
            rows.append({"unit_id": unit["unit_id"], "error": stop.reason})
            continue
        slices = dict(cell["slice_rows"])
        smallest = min(slices.values()) if slices else 0
        empty = sorted(name for name, count in slices.items() if count == 0)
        rows.append({
            "unit_id": unit["unit_id"],
            "forward_position": unit.get("forward_position"),
            "group": unit.get("group"),
            "fit_rows": cell["fit_rows"],
            "support_pool_rows": cell["support_pool_rows"],
            "slice_rows": slices,
            "empty_slices": empty,
            "smallest_slice_rows": smallest,
            "smallest_expressible_gain": (1.0 / smallest) if smallest else None,
            "can_express_a_material_relation": bool(
                smallest and (1.0 / smallest) >= MATERIAL),
            "relation_resolution": (
                "the r2 round has no rows to read at all" if empty else
                "a %d-row slice moves accuracy only in steps of %.3f"
                % (smallest, 1.0 / smallest)),
        })
    coarse = [row for row in rows if row.get("smallest_slice_rows", 0) <= 2]
    empty_units = [row for row in rows if row.get("empty_slices")]
    return {
        "per_unit": rows,
        "units_whose_smallest_slice_is_at_most_two_rows": [
            row["unit_id"] for row in coarse],
        "units_with_an_empty_held_in_slice": [
            {"unit_id": row["unit_id"], "empty_slices": row["empty_slices"]}
            for row in empty_units],
        "two_round_protocol_has_rows_on_every_unit": not empty_units,
        "why_this_matters": (
            "the harm channel needs NEGATIVE Episodes on two distinct units "
            "before risk_skill can compile a guard, and classify_relation "
            "needs an aggregate move of at least %s to call anything but "
            "NEUTRAL.  A slice of n rows moves accuracy only in steps of 1/n, "
            "so a one- or two-row slice reports 0.0 for every candidate and a "
            "zero-row slice leaves the round with no surface to read at all.  "
            "This table is the readability precondition, not a result."
            % MATERIAL),
    }


# =========================================================================== #
# Part 5 -- judging.  Runs once, after every arm is closed.
# =========================================================================== #
def _oracle_for(unit_id: str) -> dict[str, Any]:
    path = ORACLE_DIR / ("%s.json" % unit_id)
    if not path.is_file():
        raise Stop("INSTRUMENT_UNREADABLE", "no sealed oracle for %s" % unit_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _oracle_verdict_on(program_rows: Mapping[str, Mapping[str, Any]],
                       op: str, *, heldout_line: float,
                       class_bar: float) -> dict[str, Any]:
    row = program_rows.get(op)
    if row is None:
        return {"program": op, "known": False}
    utility = row.get("heldout_utility")
    worst = row.get("heldout_worst_class_recall_delta")
    harmful = bool(
        (utility is not None and float(utility) <= -float(heldout_line))
        or (worst is not None and float(worst) <= -float(class_bar)))
    return {
        "program": op,
        "known": True,
        "legal": bool(row.get("legal")),
        "in_oracle_set": bool(row.get("in_oracle_set")),
        "heldout_utility": utility,
        "heldout_worst_class_recall_delta": worst,
        "oracle_harmful": harmful,
    }


def readable_surface_evidence(*, unit: Mapping[str, Any],
                              by_arm: Mapping[str, Mapping[str, Any]]
                              ) -> dict[str, Any]:
    """Did the feedback surface actually say anything, and could it have?

    Live evidence is a Support receipt whose relation is not NEUTRAL.  When
    the proposal stage never sampled an operator the oracle knows is material
    on this unit, that is a sampling miss rather than an unreadable surface,
    and the arithmetic corroboration below says so from the sealed numbers:
    a pooled held-in reading of magnitude m is expressible on a slice of n
    rows when m >= 1/n, and is material when m >= %s.

    Runs in the judging phase, after every arm is closed.
    """ % MATERIAL
    _set_phase(PHASE_JUDGE)
    unit_id = str(unit["unit_id"])
    receipts: list[dict[str, Any]] = []
    for arm in ADAPTIVE_ARMS:
        for record in by_arm[arm]["rounds"]:
            for episode in record["episodes"]:
                receipts.append({
                    "arm": arm,
                    "round": record["round"],
                    "program": episode["workflow_signature"],
                    "relation": episode["relation"],
                    "support_gain": episode["support_gain"],
                    "delayed_gain": episode["delayed_gain"],
                })
    live = [row for row in receipts if str(row["relation"]) != "NEUTRAL"]
    probed = sorted({str(row["program"]) for row in receipts})

    oracle = _oracle_for(unit_id)
    rows = {str(row["program"]): row for row in oracle["programs"]}
    resolution = float(unit.get("slice_resolution") or 0.0)
    watch = sorted(set(HARM_OPERATORS) | {
        str(name) for name in (oracle.get("oracle_set") or [])
        if str(name) != "identity"})
    arithmetic: list[dict[str, Any]] = []
    for op in watch:
        row = rows.get(op)
        if row is None:
            continue
        headroom = row.get("heldin_headroom")
        magnitude = abs(float(headroom)) if headroom is not None else None
        arithmetic.append({
            "program": op,
            "legal": bool(row.get("legal")),
            "pooled_heldin_headroom": headroom,
            "magnitude": magnitude,
            "slice_resolution": resolution,
            "expressible_on_the_smallest_slice": bool(
                magnitude is not None and resolution
                and magnitude >= resolution),
            "material": bool(magnitude is not None and magnitude >= MATERIAL),
            "was_probed_in_the_smoke": op in probed,
        })
    corroborated = [row for row in arithmetic
                    if row["expressible_on_the_smallest_slice"]
                    and row["material"] and row["legal"]]
    mode = ("live" if live else
            "arithmetic_only" if corroborated else "none")
    return {
        "unit_id": unit_id,
        "min_slice_rows": unit.get("min_slice_rows"),
        "slice_resolution": resolution,
        "support_receipts": receipts,
        "programs_probed": probed,
        "live_non_neutral_receipts": live,
        "live_non_neutral_count": len(live),
        "arithmetic_corroboration": arithmetic,
        "arithmetic_corroboration_holds": bool(corroborated),
        "mode": mode,
        "reading": (
            "at least one Support receipt came back non-NEUTRAL: the feedback "
            "surface is readable on live evidence"
            if mode == "live" else
            "every Support receipt was NEUTRAL, but the proposal stage never "
            "sampled a program the sealed oracle knows is material here.  The "
            "arithmetic side-evidence stands in: %s"
            % "; ".join(
                "%s reads %+.4f pooled, %.4f resolvable on a %s-row slice"
                % (row["program"], row["pooled_heldin_headroom"],
                   row["slice_resolution"], unit.get("min_slice_rows"))
                for row in corroborated)
            if mode == "arithmetic_only" else
            "neither a live non-NEUTRAL receipt nor an arithmetic "
            "corroboration: this unit's feedback surface is not readable"),
        "caveat": (
            "the oracle headroom is measured on the pooled held-in surface "
            "(all four slices concatenated) while a round reads one slice, so "
            "the arithmetic is a necessary condition on the surface, not a "
            "guarantee about any single round"),
    }


def judge(*, unit_results: Sequence[Mapping[str, Any]],
          course: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _set_phase(PHASE_JUDGE)
    per_unit: list[dict[str, Any]] = []
    for unit in course:
        unit_id = str(unit["unit_id"])
        oracle = _oracle_for(unit_id)
        rows = {str(row["program"]): row for row in oracle["programs"]}
        heldout_line = float(oracle.get("heldout_material_line") or MATERIAL)
        class_bar = float(oracle.get("class_harm_bar") or MATERIAL)
        menu = float(oracle.get("menu_oracle_heldout_utility") or 0.0)
        for result in unit_results:
            if str(result["unit_id"]) != unit_id:
                continue
            deployment = result["deployment"]
            applied = [str(step["op"]) for step in deployment["applied_program"]]
            verdicts = [_oracle_verdict_on(rows, op, heldout_line=heldout_line,
                                           class_bar=class_bar)
                        for op in applied]
            actual = float(deployment["heldout_accuracy_gain"])
            deltas = deployment["heldout_recall_delta_by_class"] or {}
            worst = min([float(value) for value in deltas.values()] or [0.0])
            wasted = 0
            wasted_detail: list[dict[str, Any]] = []
            for record in result["rounds"]:
                for probe in record["probes"]:
                    ops = probe["operators"] or []
                    reasons = []
                    if probe.get("kind") == "verifier_rejected":
                        reasons.append("verifier_rejected")
                    for op in ops:
                        row = rows.get(op)
                        if row is not None and not row.get("legal"):
                            reasons.append("oracle_illegal:%s" % op)
                        verdict = _oracle_verdict_on(
                            rows, op, heldout_line=heldout_line,
                            class_bar=class_bar)
                        if verdict.get("oracle_harmful"):
                            reasons.append("oracle_harmful:%s" % op)
                    if reasons:
                        wasted += 1
                        wasted_detail.append({
                            "round": record["round"],
                            "candidate_id": probe["candidate_id"],
                            "reasons": sorted(set(reasons))})
            directions: list[dict[str, Any]] = []
            for record in result["rounds"]:
                for episode in record["episodes"]:
                    support = episode.get("support_gain")
                    delayed = episode.get("delayed_gain")
                    if support is None or delayed is None:
                        continue
                    directions.append({
                        "episode_id": episode["episode_id"],
                        "support_gain": support,
                        "delayed_gain": delayed,
                        "same_direction": (float(support) >= 0) == (
                            float(delayed) >= 0)})
            promotions: list[dict[str, Any]] = []
            for record in result["rounds"]:
                if not record.get("activated"):
                    continue
                winner = record.get("winner_program") or []
                for step in winner:
                    verdict = _oracle_verdict_on(
                        rows, str(step.get("op")), heldout_line=heldout_line,
                        class_bar=class_bar)
                    promotions.append({
                        "round": record["round"],
                        "skill_id": record.get("approved_skill_id"),
                        "program": step.get("op"),
                        "wrong": bool(verdict.get("oracle_harmful")
                                      or not verdict.get("in_oracle_set", False)),
                        "oracle": verdict})
            guard_events = []
            for record in result["rounds"]:
                lifecycle = record.get("risk_lifecycle") or {}
                ids = list(lifecycle.get("risk_skill_ids") or [])
                if ids or record["guard_readout"]["guarded_operators_in_fast_view"]:
                    guard_events.append({
                        "round": record["round"],
                        "minted_or_active_guards": ids,
                        **record["guard_readout"]})
            per_unit.append({
                "unit_id": unit_id,
                "forward_position": unit.get("forward_position"),
                "group": unit.get("group"),
                "arm": result["arm"],
                "deploy_source": deployment["deploy_source"],
                "applied_program": deployment["applied_program"],
                "menu_oracle_program": oracle.get("menu_oracle_program"),
                "menu_oracle_heldout_utility": menu,
                "heldout_utility": actual,
                "regret": menu - actual,
                "heldout_accuracy": deployment["heldout_accuracy"],
                "heldout_recall_by_class": deployment["heldout_recall_by_class"],
                "heldout_recall_delta_by_class": deltas,
                "worst_class_harm": worst,
                "harm_event": bool(any(v.get("oracle_harmful") for v in verdicts)),
                "harm_event_detail": verdicts,
                "wrong_promotions": len([p for p in promotions if p["wrong"]]),
                "wrong_promotion_detail": promotions,
                "cost": {
                    "llm": result["llm_calls"],
                    "consumer_fits": result["consumer_fits"],
                    "probes": sum(len(r["probes"]) for r in result["rounds"]),
                },
                "wasted_probes": wasted,
                "wasted_probe_detail": wasted_detail,
                "support_delayed_direction": {
                    "pairs": len(directions),
                    "same_direction": len([d for d in directions
                                           if d["same_direction"]]),
                    "detail": directions},
                "guard_events": guard_events,
                "deploy_purity_breach": deployment["breach"],
            })
    totals: dict[str, dict[str, Any]] = {}
    for row in per_unit:
        bucket = totals.setdefault(row["arm"], {
            "units": 0, "cumulative_regret": 0.0,
            "cumulative_heldout_utility": 0.0, "harm_events": 0,
            "wrong_promotions": 0, "wasted_probes": 0,
            "llm": 0, "consumer_fits": 0, "probes": 0,
            "worst_class_harm_min": 0.0, "curve": []})
        bucket["units"] += 1
        bucket["cumulative_regret"] += float(row["regret"])
        bucket["cumulative_heldout_utility"] += float(row["heldout_utility"])
        bucket["harm_events"] += int(bool(row["harm_event"]))
        bucket["wrong_promotions"] += int(row["wrong_promotions"])
        bucket["wasted_probes"] += int(row["wasted_probes"])
        bucket["llm"] += int(row["cost"]["llm"])
        bucket["consumer_fits"] += int(row["cost"]["consumer_fits"])
        bucket["probes"] += int(row["cost"]["probes"])
        bucket["worst_class_harm_min"] = min(
            bucket["worst_class_harm_min"], float(row["worst_class_harm"]))
        bucket["curve"].append({
            "unit_id": row["unit_id"],
            "forward_position": row["forward_position"],
            "regret": row["regret"],
            "cumulative_regret": bucket["cumulative_regret"],
            "heldout_utility": row["heldout_utility"],
            "llm": row["cost"]["llm"],
            "consumer_fits": row["cost"]["consumer_fits"]})
    return {
        "reads": "artifacts/functional/e2/s1_oracle/*.json",
        "read_when": ("after every arm is closed; the arm phase physically "
                      "cannot open these files"),
        "per_unit_per_arm": per_unit,
        "totals_by_arm": totals,
        "pre_registered_readout": PRE_REGISTERED_READOUT,
    }


# =========================================================================== #
# Part 6 -- the smoke
# =========================================================================== #
def smoke(*, live: bool = False) -> int:
    started = time.time()
    _set_phase(PHASE_SETUP)
    if not FROZEN_JSON.is_file():
        raise Stop("INSTRUMENT_UNREADABLE",
                   "run --select-curriculum first: %s missing" % FROZEN_JSON)
    frozen = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))
    course = list(frozen["units"])
    unit = dict(course[0])
    next_unit = dict(course[1]) if len(course) > 1 else None
    unit_id = str(unit["unit_id"])

    tag = "s1_smoke_live" if live else "s1_smoke"
    store_root = Path(tempfile.gettempdir()) / tag
    if store_root.exists():
        shutil.rmtree(store_root)
    k0 = compile_k0(store_root)
    llm_cap = SMOKE_LLM_PER_ARM * len(ADAPTIVE_ARMS)
    backend = (cls._live_backend(llm_cap) if live
               else cls._scripted_backend(llm_cap))
    agent_factory = cls._live_agent if live else cls._scripted_agent
    llm_ledger = {"fast": 0, "slow": 0}

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "entry": "--smoke" + (" --live" if live else ""),
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": _git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "backend": "live_fast_agent" if live else "scripted_sealed_probe",
        "curriculum_source": FROZEN_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "curriculum": {
            "revision": frozen.get("curriculum_revision", "r1"),
            "forward_order": frozen.get("forward_order"),
            "supersedes": frozen.get("supersedes"),
        },
        "unit_under_test": unit,
        "next_unit_for_the_boundary_test": next_unit,
        "reduced_protocol": {
            "rounds_per_unit": list(SMOKE_ROUNDS),
            "full_protocol_rounds": list(HELD_IN_ROUNDS),
            "llm_cap_per_arm": SMOKE_LLM_PER_ARM,
            "fit_cap_per_unit_per_arm": FIT_PER_UNIT_PER_ARM,
            "wall_seconds_cap": SMOKE_WALL_SECONDS_CAP,
            "note": ("this is a wiring smoke on curriculum unit 1 only.  The "
                     "course is NOT run here."),
        },
        "k0": {
            "h0_runtime_bundle_sha": k0["h0_sha"],
            "k0_runtime_bundle_sha": k0["k0_sha"],
            "h0_skill_ids": k0["h0_skill_ids"],
            "k0_skill_ids": k0["k0_skill_ids"],
            "purity": k0["purity"],
            "definition": _k0_definition(),
        },
    }
    stopped: str | None = None
    try:
        cell = _build_cell(unit)
        pattern_views = {unit_id: _pattern_view_of(cell)}
        next_pattern: dict[str, Any] = {}
        if next_unit is not None:
            next_cell = _build_cell(next_unit)
            next_pattern = _pattern_view_of(next_cell)
            pattern_views[str(next_unit["unit_id"])] = next_pattern
        payload["pattern_views"] = pattern_views

        guard_probe = None
        unit_results: list[dict[str, Any]] = []
        for arm in ARMS:
            base = {ARM_STATIC: k0["h0"], ARM_A3: k0["h0"],
                    ARM_K0: k0["k0"], ARM_A5: k0["k0"]}[arm]
            result = run_unit(
                unit=unit, cell=cell, arm=arm, base_snapshot=base,
                carried_episodes=(), agent_factory=agent_factory,
                backend=backend, store_root=store_root,
                rounds=SMOKE_ROUNDS, fit_cap=FIT_PER_UNIT_PER_ARM)
            if guard_probe is None:
                guard_probe = _oracle_guard_selftest()
            unit_results.append(result)
            print("%-10s %-30s rounds=%d fits=%d llm=%d deploy=%s gain=%+.4f"
                  % (arm, unit_id, len(result["rounds"]),
                     result["consumer_fits"], result["llm_calls"],
                     result["deployment"]["deploy_source"],
                     result["deployment"]["heldout_accuracy_gain"]),
                  flush=True)
            if time.time() - started > SMOKE_WALL_SECONDS_CAP:
                raise Stop("WALL_CLOCK_EXCEEDED",
                           "smoke exceeded %d s" % SMOKE_WALL_SECONDS_CAP)
        llm_ledger["fast"] = int(backend.calls)
        payload["oracle_guard_selftest"] = guard_probe

        by_arm = {str(row["arm"]): row for row in unit_results}
        a5 = by_arm[ARM_A5]
        cells = {unit_id: cell}
        _set_phase(PHASE_ARM, unit=unit_id, arm=ARM_A5)
        integration = slow_integration(
            unit_results=[a5], cells=cells, llm_ledger=llm_ledger, live=live,
            pattern_views=pattern_views)
        carry = None
        if next_unit is not None:
            carry = carry_into_next_unit(
                unit_result=a5, integration=integration,
                next_unit_id=str(next_unit["unit_id"]),
                next_pattern=next_pattern, k0=k0["k0"],
                store_root=store_root, tag="a5_carry")
        payload["a5_slow_integration"] = {
            key: value for key, value in integration.items()
            if key != "entry"} | {"entry_skill_id": (
                (integration.get("entry") or {}).get("skill_id"))}
        payload["a5_carry_into_next_unit"] = (
            {key: value for key, value in carry.items() if key != "snapshot"}
            if carry else None)

        boundary = _boundary_readout(by_arm=by_arm, k0=k0, carry=carry,
                                     next_unit=next_unit)
        payload["unit_boundary_state"] = boundary
        payload["arm_isolation"] = _arm_isolation(by_arm=by_arm, k0=k0,
                                                  boundary=boundary)
        payload["domain_binding"] = _domain_binding_readout(
            by_arm=by_arm, carry=carry, next_unit=next_unit)
        if next_unit is not None:
            payload["domain_binding"]["synthetic_probe"] = domain_binding_probe(
                this_unit_id=unit_id,
                next_unit_id=str(next_unit["unit_id"]),
                next_pattern=next_pattern,
                this_pattern=pattern_views[unit_id])
        _set_phase(PHASE_SETUP)
        payload["instrument_census"] = instrument_census(course)
        payload["instrument_findings"] = _instrument_findings(
            payload["instrument_census"], by_arm=by_arm)
        payload["arm_results"] = [
            {key: value for key, value in row.items()
             if not key.startswith("_")} for row in unit_results]
        payload["judging"] = judge(unit_results=unit_results, course=[unit])
        payload["readable_surface_evidence"] = readable_surface_evidence(
            unit=unit, by_arm=by_arm)
        payload["guard_channel_feasibility"] = guard_channel_feasibility(course)
        payload["deploy_rule_observation"] = deploy_rule_observation(by_arm)
    except Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
    except OracleIsolationBreach as breach:
        stopped = "ORACLE_ISOLATION_BREACH"
        payload["stop"] = {"verdict": stopped, "reason": str(breach)}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {"verdict": stopped,
                           "reason": "%s: %s" % (type(exc).__name__, exc),
                           "traceback": traceback.format_exc()}
    finally:
        _set_phase(PHASE_JUDGE)

    payload["ledger"] = {
        "llm_calls_fast": llm_ledger["fast"],
        "llm_calls_slow": llm_ledger["slow"],
        "llm_calls_total": llm_ledger["fast"] + llm_ledger["slow"],
        "llm_cap": llm_cap,
        "consumer_fits": sum(
            int(row.get("consumer_fits") or 0)
            for row in payload.get("arm_results", [])),
        "wall_seconds": round(time.time() - started, 1),
        "wall_seconds_cap": SMOKE_WALL_SECONDS_CAP,
        "real_llm_spend": (llm_ledger["fast"] + llm_ledger["slow"]) if live else 0,
        "counter_meaning": (
            "live Fast/Slow agent calls" if live else
            "scripted sealed-probe backend calls; real LLM spend is 0"),
    }
    payload["oracle_isolation"] = _oracle_isolation_report()
    payload["verdict"] = _smoke_verdict(payload, stopped=stopped)
    payload["obligations"] = _smoke_obligations(payload, live=live)
    _dump(SMOKE_JSON, payload)
    SMOKE_MD.write_text(_smoke_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "llm": payload["ledger"]["llm_calls_total"],
                      "fits": payload["ledger"]["consumer_fits"],
                      "seconds": payload["ledger"]["wall_seconds"],
                      "artifact": str(SMOKE_JSON)},
                     ensure_ascii=False, indent=1))
    return 0 if payload["verdict"]["verdict"] == "S1B_SMOKE_WIRED" else 1


def _instrument_findings(census: Mapping[str, Any], *,
                         by_arm: Mapping[str, Mapping[str, Any]]
                         ) -> dict[str, Any]:
    relations = sorted({
        str(episode["relation"])
        for arm in ADAPTIVE_ARMS
        for record in by_arm[arm]["rounds"]
        for episode in record["episodes"]})
    coarse = list(census["units_whose_smallest_slice_is_at_most_two_rows"])
    empty = list(census["units_with_an_empty_held_in_slice"])
    harm_written = "NEGATIVE" in relations
    smallest = min([int(row.get("smallest_slice_rows") or 0)
                    for row in census.get("per_unit", [])] or [0])
    blocks = bool(len(coarse) >= 2 or empty or not harm_written)
    if blocks:
        finding = (
            "%d of the seven units have a held-in slice of at most two rows "
            "and %d have an empty slice outright.  On unit 1 the observed "
            "Episode relations were %s.  The wiring is correct; the material "
            "is too coarse to exercise it."
            % (len(coarse), len(empty), relations))
        unblock = (
            "raise the slice readability floor, or stop quartering an already "
            "small support pool.  Both are protocol changes reported for the "
            "main line rather than taken here.")
    else:
        finding = (
            "the r2 readability floor holds across the whole course: every "
            "unit's smallest held-in slice is at least %d rows, none is "
            "empty, and the frozen two-round protocol has a delayed surface "
            "everywhere.  On unit 1 the observed Episode relations were %s, "
            "so a harm Episode was written on live evidence rather than "
            "inferred." % (smallest, relations))
        unblock = (
            "nothing outstanding on the readability axis.  The remaining "
            "question is not whether harm can be *read* but whether the "
            "proposal stage samples the same harmful program on both harm "
            "units -- see guard_channel_feasibility.")
    return {
        "observed_relations_on_unit_1": relations,
        "units_with_an_empty_held_in_slice": empty,
        "two_round_protocol_has_rows_on_every_unit": census[
            "two_round_protocol_has_rows_on_every_unit"],
        "smallest_slice_anywhere_in_the_course": smallest,
        "harm_episode_formed_on_unit_1": harm_written,
        "guard_minted_on_unit_1": any(
            (record.get("risk_lifecycle") or {}).get("risk_skill_ids")
            for arm in ADAPTIVE_ARMS
            for record in by_arm[arm]["rounds"]),
        "coarse_units": coarse,
        "finding": finding,
        "blocks_s1c": blocks,
        "what_would_unblock_it": unblock,
    }


def guard_channel_feasibility(course: Sequence[Mapping[str, Any]]
                              ) -> dict[str, Any]:
    """Can a guard form at all on this course, and on which program?

    ``risk_skill`` compiles a guard when the same Program family carries a
    NEGATIVE Episode on two distinct counted units.  That needs a program
    that is (a) legal on both harm units, (b) materially harmful on held-in
    there, and (c) harmful by more than the coarsest slice can resolve.
    Whether the guard actually forms additionally needs the proposal stage to
    sample that program on both units, which no arithmetic can promise.

    Judging phase: reads the sealed oracle.
    """
    _set_phase(PHASE_JUDGE)
    harm_units = [unit for unit in course if unit.get("group") == GROUP_HARM]
    per_unit: list[dict[str, Any]] = []
    sets: list[set[str]] = []
    for unit in harm_units:
        oracle = _oracle_for(str(unit["unit_id"]))
        resolution = float(unit.get("slice_resolution") or 0.0)
        readable: dict[str, float] = {}
        for row in oracle["programs"]:
            headroom = row.get("heldin_headroom")
            if not row.get("legal") or headroom is None:
                continue
            value = float(headroom)
            if value > -MATERIAL:
                continue
            if resolution and abs(value) < resolution:
                continue
            readable[str(row["program"])] = value
        per_unit.append({
            "unit_id": unit["unit_id"],
            "forward_position": unit.get("forward_position"),
            "min_slice_rows": unit.get("min_slice_rows"),
            "slice_resolution": resolution,
            "readably_harmful_legal_programs": dict(sorted(readable.items())),
        })
        sets.append(set(readable))
    shared = sorted(set.intersection(*sets)) if sets else []
    return {
        "harm_units": per_unit,
        "programs_readably_harmful_on_every_harm_unit": shared,
        "guard_is_formable_in_principle": bool(shared),
        "reading": (
            "a guard can form on %s: legal and readably harmful on both harm "
            "units.  Formation still requires the proposal stage to sample it "
            "on both, which is an agent behaviour and not an arithmetic "
            "guarantee." % shared if shared else
            "no program is legal and readably harmful on both harm units, so "
            "the two-distinct-unit floor cannot be reached on this course"),
        "expected_earliest_guard": (
            "after forward position %s, the second harm unit"
            % max((unit.get("forward_position") for unit in harm_units),
                  default=None)),
    }


def deploy_rule_observation(by_arm: Mapping[str, Mapping[str, Any]]
                            ) -> dict[str, Any]:
    """Did any arm freeze a Workflow that its own delayed feedback rejected?

    Inherited from the shared runner's round body: ``state['incumbent']`` is
    set from the round's Support winner and is not cleared when
    ``handle_feedback_delayed`` refuses to approve.  ``_frozen_recall`` then
    deploys that incumbent.  Recorded, not repaired -- the deploy rule lives
    in the shared runner and changing it is a behaviour change with its own
    slice.
    """
    rows: list[dict[str, Any]] = []
    for arm in ADAPTIVE_ARMS:
        result = by_arm[arm]
        for record in result["rounds"]:
            winner = record.get("winner_program") or []
            if not winner:
                continue
            rejected = [episode for episode in record["episodes"]
                        if str(episode["relation"]) != "POSITIVE"
                        and episode.get("delayed_gain") is not None]
            if not rejected:
                continue
            rows.append({
                "arm": arm,
                "round": record["round"],
                "support_winner": [step.get("op") for step in winner],
                "approved_skill_id": record.get("approved_skill_id"),
                "activated": record.get("activated"),
                "delayed_relations": [episode["relation"]
                                      for episode in rejected],
                "deployed_program": [
                    step.get("op") for step
                    in result["deployment"]["applied_program"]],
                "deploy_source": result["deployment"]["deploy_source"],
            })
    return {
        "rows": rows,
        "any_arm_deployed_a_delayed_rejected_winner": bool(rows),
        "note": (
            "the delayed gate correctly withheld Skill approval, but the "
            "ledger incumbent set by the Support winner survived it and "
            "became the frozen deployment.  Inherited from the shared "
            "runner's round body; recorded here, not repaired."),
    }


def _boundary_readout(*, by_arm: Mapping[str, Mapping[str, Any]], k0: Any,
                      carry: Mapping[str, Any] | None,
                      next_unit: Mapping[str, Any] | None) -> dict[str, Any]:
    """What each arm would start the *next* unit from.  No next unit is run."""
    rows: dict[str, Any] = {}
    for arm in ARMS:
        result = by_arm[arm]
        if arm == ARM_STATIC:
            rows[arm] = {"next_base_sha": k0["h0_sha"],
                         "next_base": "h0 (never adapts, never deploys a "
                                      "learned Workflow)",
                         "episodes_carried": 0, "skills_carried": []}
        elif arm == ARM_A3:
            rows[arm] = {"next_base_sha": k0["h0_sha"],
                         "next_base": "h0 (cold start, zero carry)",
                         "episodes_carried": 0, "skills_carried": []}
        elif arm == ARM_K0:
            rows[arm] = {"next_base_sha": k0["k0_sha"],
                         "next_base": "K0 (reset; no write-back)",
                         "episodes_carried": 0, "skills_carried": []}
        else:
            rows[arm] = {
                "next_base_sha": (carry or {}).get("runtime_bundle_sha"),
                "next_base": "K0 + everything the domain-binding wall admits",
                "episodes_carried": result["episodes_at_unit_end"],
                "skills_carried": list((carry or {}).get("carried_skill_ids")
                                       or []),
                "skills_dropped": list((carry or {}).get("dropped_skill_ids")
                                       or []),
            }
        rows[arm]["end_of_unit_sha"] = result["end_runtime_bundle_sha"]
        rows[arm]["store_evolved_in_unit"] = result["store_evolved"]
        rows[arm]["episodes_at_unit_end"] = result["episodes_at_unit_end"]
    rows["next_unit_id"] = (str(next_unit["unit_id"]) if next_unit else None)
    return rows


def _arm_isolation(*, by_arm: Mapping[str, Mapping[str, Any]], k0: Any,
                   boundary: Mapping[str, Any]) -> dict[str, Any]:
    static = by_arm[ARM_STATIC]
    a3 = by_arm[ARM_A3]
    fixed = by_arm[ARM_K0]
    a5 = by_arm[ARM_A5]
    checks = {
        "static_ran_no_round": len(static["rounds"]) == 0,
        "static_wrote_no_episode": static["episodes_at_unit_end"] == 0,
        "static_store_unchanged": not static["store_evolved"],
        "static_deployed_identity": (
            static["deployment"]["applied_program"] == []),
        "a3_started_from_h0": a3["base_runtime_bundle_sha"] == k0["h0_sha"],
        "a3_started_with_no_episode": a3["episodes_at_unit_start"] == 0,
        "a3_next_base_is_h0": boundary[ARM_A3]["next_base_sha"] == k0["h0_sha"],
        "k0fixed_started_from_k0": (
            fixed["base_runtime_bundle_sha"] == k0["k0_sha"]),
        "k0fixed_next_base_resets_to_k0": (
            boundary[ARM_K0]["next_base_sha"] == k0["k0_sha"]),
        "k0fixed_carries_no_episode": (
            boundary[ARM_K0]["episodes_carried"] == 0),
        "a5_started_from_k0": a5["base_runtime_bundle_sha"] == k0["k0_sha"],
        "a5_carries_episode_memory": (
            boundary[ARM_A5]["episodes_carried"] > 0),
        "a5_and_k0fixed_share_the_same_start": (
            a5["base_runtime_bundle_sha"] == fixed["base_runtime_bundle_sha"]),
        "a5_next_base_differs_from_k0_or_states_why": True,
    }
    checks["a5_next_base_differs_from_k0_or_states_why"] = bool(
        boundary[ARM_A5]["next_base_sha"] != k0["k0_sha"]
        or not boundary[ARM_A5]["skills_carried"])
    # The K0 card lives in the store of both K0-fixed and A5-online, and the
    # frozen T1 predicate must keep it out of every Fast view: TRY abstains and
    # its RISK is scoped only to the eligibility gate, so it authorizes nothing.
    in_store = [arm for arm in (ARM_K0, ARM_A5)
                if SOURCE_SKILL_ID in by_arm[arm]["base_skill_ids"]]
    in_fast = [arm for arm in (ARM_K0, ARM_A5)
               for record in by_arm[arm]["rounds"]
               if SOURCE_SKILL_ID in record["retrieved_skill_ids"]]
    checks["k0_card_in_the_store_of_k0fixed_and_a5"] = (
        sorted(in_store) == sorted([ARM_K0, ARM_A5]))
    checks["k0_card_never_entered_a_fast_view"] = not in_fast
    checks["a3_store_never_held_the_k0_card"] = (
        SOURCE_SKILL_ID not in a3["base_skill_ids"])
    return {"checks": checks, "all_hold": all(checks.values()),
            "k0_card_in_fast_view_of": sorted(set(in_fast))}


def _domain_binding_readout(*, by_arm: Mapping[str, Mapping[str, Any]],
                            carry: Mapping[str, Any] | None,
                            next_unit: Mapping[str, Any] | None
                            ) -> dict[str, Any]:
    stamped = []
    for arm in ADAPTIVE_ARMS:
        for entry in by_arm[arm]["skills_added_in_unit"]:
            stamped.append({"arm": arm, "skill_id": entry["skill_id"],
                            "carrier_kind": entry["carrier_kind"],
                            "domain_namespace": entry["domain_namespace"]})
    decisions = list((carry or {}).get("decisions") or [])
    target_local = [row for row in decisions
                    if row["carrier_kind"] == "target_local_capability"]
    return {
        "hook_1_stamped_skills": stamped,
        "hook_1_every_minted_skill_is_stamped": all(
            row["domain_namespace"] for row in stamped),
        "hook_2_next_unit": (str(next_unit["unit_id"]) if next_unit else None),
        "hook_2_target_local_decisions": target_local,
        "hook_2_all_foreign_target_local_dropped": all(
            not row["carried"] for row in target_local),
        "hook_3_source_card_decision": (carry or {}).get(
            "source_card_decision"),
        "episode_domain_namespaces": sorted({
            str(episode.get("task_episode_id") or episode.get(
                "domain_namespace"))
            for arm in ADAPTIVE_ARMS
            for record in by_arm[arm]["rounds"]
            for episode in record["episodes"]}),
    }


def _smoke_verdict(payload: Mapping[str, Any], *,
                   stopped: str | None) -> dict[str, Any]:
    if stopped:
        return {"verdict": stopped,
                "reason": (payload.get("stop") or {}).get("reason")}
    isolation = payload.get("arm_isolation") or {}
    binding = payload.get("domain_binding") or {}
    oracle = payload.get("oracle_isolation") or {}
    probe = payload.get("oracle_guard_selftest") or {}
    judged = payload.get("judging") or {}
    gates = {
        "four_arm_state_isolation": bool(isolation.get("all_hold")),
        "domain_binding_hooks_live": bool(
            binding.get("hook_1_every_minted_skill_is_stamped")
            and binding.get("hook_2_all_foreign_target_local_dropped")
            and (binding.get("synthetic_probe") or {}).get(
                "hook_2", {}).get("behaves_as_specified")
            and (binding.get("synthetic_probe") or {}).get(
                "hook_3", {}).get("behaves_as_specified")),
        "judging_produced_a_regret_table": bool(
            judged.get("per_unit_per_arm")
            and len(judged["per_unit_per_arm"]) == len(ARMS)),
        "oracle_isolation_holds": bool(oracle.get("holds")),
        "oracle_wall_proved_armed": bool(probe.get("fired")),
        "within_budget": bool(
            payload["ledger"]["llm_calls_total"] <= payload["ledger"]["llm_cap"]
            and payload["ledger"]["wall_seconds"]
            <= payload["ledger"]["wall_seconds_cap"]),
        "deploy_purity_clean": all(
            not row["deploy_purity_breach"]
            for row in judged.get("per_unit_per_arm", [])),
    }
    readable = payload.get("readable_surface_evidence") or {}
    gates["feedback_surface_readable"] = readable.get("mode") in (
        "live", "arithmetic_only")
    return {
        "verdict": "S1B_SMOKE_WIRED" if all(gates.values())
                   else "S1B_SMOKE_INCOMPLETE",
        "gates": gates,
        "feedback_surface_evidence_mode": readable.get("mode"),
        "scope": ("wiring only, on curriculum unit 1, one round per adaptive "
                  "arm.  No Capability claim; the course was not run."),
    }


def _smoke_obligations(payload: Mapping[str, Any], *,
                       live: bool) -> dict[str, Any]:
    return {
        "methods_package_unmodified": True,
        "runtime_contracts_operators_unmodified": True,
        "shared_runner_unmodified": True,
        "full_curriculum_not_run": True,
        "units_run": 1,
        "rounds_per_adaptive_arm": len(SMOKE_ROUNDS),
        "oracle_isolation_mechanism": (payload.get("oracle_isolation") or {}).get(
            "mechanism"),
        "oracle_isolation_holds": (payload.get("oracle_isolation") or {}).get(
            "holds"),
        "oracle_wall_selftest_fired": (
            payload.get("oracle_guard_selftest") or {}).get("fired"),
        "oracle_wall_surfaces_probed": (
            payload.get("oracle_guard_selftest") or {}).get("surfaces"),
        "harm_channel_exercised_on_unit_1": (
            payload.get("instrument_findings") or {}).get(
                "harm_episode_formed_on_unit_1"),
        "instrument_blocker_reported_for_s1c": (
            payload.get("instrument_findings") or {}).get("blocks_s1c"),
        "feedback_surface_evidence_mode": (
            payload.get("readable_surface_evidence") or {}).get("mode"),
        "guard_formable_in_principle_on_this_course": (
            payload.get("guard_channel_feasibility") or {}).get(
                "guard_is_formable_in_principle"),
        "a_delayed_rejected_winner_was_still_deployed": (
            payload.get("deploy_rule_observation") or {}).get(
                "any_arm_deployed_a_delayed_rejected_winner"),
        "curriculum_revision": (
            payload.get("curriculum") or {}).get("revision"),
        "k0_has_no_target_local_capability": (
            payload["k0"]["purity"]["no_target_local_capability_in_k0"]),
        "k0_card_carries_no_frozen_steps": (
            payload["k0"]["purity"]["no_frozen_program_steps"]),
        "live_llm_backend": live,
        "downloads": 0,
        "full_repo_pytest_not_run": True,
        "sealed_artifacts_not_rewritten": True,
    }


# =========================================================================== #
# markdown
# =========================================================================== #
def _frozen_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# S1 four-arm evolution curriculum -- frozen course (%s)"
        % payload["curriculum_revision"],
        "",
        "protocol: `%s`  revision: **%s**  evidence grade: **%s**  git: `%s`"
        % (payload["protocol_version"], payload["curriculum_revision"],
           payload["evidence_grade"], payload["git_head"]),
        "",
        "supersedes `%s` (kept, not discarded)" % payload["supersedes"],
        "",
        "The course below was selected mechanically from the sealed census and "
        "oracle artifacts by the rules in section 1.  No unit was chosen by "
        "hand and no outcome reordered anything after the rules ran.  The "
        "readability screen reads `cell.slice_rows` off the sealed oracle "
        "files, so it costs zero new Consumer fits.",
        "",
        "## 1. Selection rules (declared before scoring)",
        "",
    ]
    for key in ("revision", "min_slice_rows", "slice_readability_floor",
                "necessary_condition", "necessary_condition_scope",
                "within_group_ranking", "family_deduplication",
                "relaxation_ladder", "group_selection_order",
                GROUP_HARM, GROUP_LEARNABLE, GROUP_IDENTITY,
                GROUP_HELDOUT_ONLY, "unit_disjointness", "forward_order",
                "reverse_order", "domain_namespace"):
        lines.append("- **%s**: %s" % (key, payload["selection_rules"][key]))
    lines += ["", "## 2. The seven units, forward order", "",
              "| # | unit | group | family | learnability | oracle set | "
              "slice rows (r1s/r1d/r2s/r2d) | min slice | resolution | key "
              "held-in readout | why |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = payload["selected_groups"]
    why_by_unit = {row["unit_id"]: row["why"]
                   for members in groups.values() for row in members}
    for row in payload["units"]:
        lines.append(
            "| %d | %s | %s | %s%s | %s | %s | %s | %d | %.4f | %.4f | %s |" % (
                row["forward_position"], row["unit_id"], row["group"],
                row["family_key"], " (repeat)" if row["family_repeat"] else "",
                row["learnability"], ",".join(row["oracle_set"]) or "-",
                "/".join(str(value) for value in row["slice_rows"].values()),
                row["min_slice_rows"], row["slice_resolution"] or 0.0,
                row["key_heldin_readout"], why_by_unit.get(row["unit_id"], "")))
    lines += ["", "### Readability ladder actually walked", "",
              "| group | quota | rung used | family repeats | downgraded from "
              "floor 5 | short by | rungs tried |",
              "|---|---|---|---|---|---|---|"]
    for group, report in payload["ladder_trace_by_group"].items():
        rung = report["rung_used"] or {}
        lines.append("| %s | %d | floor %s, repeats %s | %s | %s | %d | %s |" % (
            group, report["quota"], rung.get("slice_floor"),
            "allowed" if rung.get("family_repeats_allowed") else "forbidden",
            report["family_repeats_used"] or "none",
            report["downgraded_from_floor_5"], report["short_by"],
            " ; ".join("floor %s/%s -> %d" % (
                item["slice_floor"],
                "repeat" if item["family_repeats_allowed"] else "strict",
                item["filled"]) for item in report["ladder_trace"])))
    lines += [
        "",
        "forward: `%s`" % payload["forward_order"],
        "",
        "reverse: `%s`" % payload["reverse_order"],
        "",
        "Design intent of the forward order: the guard should become "
        "compilable after the second harm unit, so every unit after it is a "
        "test of whether the guard actually fires.",
        "",
        "## 3. Families and substrates",
        "",
        "- distinct families in course: **%d** (%s)"
        % (payload["family_census"]["distinct_families"],
           ", ".join(payload["family_census"]["families_in_course"])),
        "- repeated families: %s"
        % (payload["family_census"]["repeated_families"] or "none"),
        "- repeated substrates: %s"
        % (payload["family_census"]["repeated_substrates"] or "none"),
        "",
        "## 4. Arms",
        "",
    ]
    for arm, text in payload["arms"].items():
        lines.append("- **%s**: %s" % (arm, text))
    k0 = payload["k0_definition"]
    lines += [
        "",
        "## 5. K0",
        "",
        "- base: %s" % k0["base"],
        "- bootstrap Skills: %s" % ", ".join(k0["bootstrap_skills"]),
        "- inert Slow card: `%s` from `%s`; TRY = `%s`; allowed_tools = %s; "
        "carries frozen steps = %s"
        % (k0["inert_slow_card"]["skill_id"], k0["inert_slow_card"]["source"],
           k0["inert_slow_card"]["try_clause"],
           k0["inert_slow_card"]["allowed_tools"],
           k0["inert_slow_card"]["carries_frozen_steps"]),
        "- **excluded on purpose**: %s" % k0["excluded_on_purpose"],
        "",
        "## 6. Domain-binding hooks",
        "",
    ]
    for key, text in payload["domain_binding_hooks"].items():
        lines.append("- **%s**: %s" % (key, text))
    lines += ["", "## 7. Budgets", ""]
    for key, value in payload["budgets"].items():
        lines.append("- %s: %s" % (key, value))
    lines += ["", "## 8. Pre-registered readout (judged in S1c, not here)", ""]
    for key, value in payload["pre_registered_readout"].items():
        lines.append("- **%s**: %s" % (key, value))
    if payload["shortfalls"]:
        lines += ["", "## 9. Shortfalls and substitutions", ""]
        for row in payload["shortfalls"]:
            lines.append("- %s: found %s -- %s"
                         % (row["group"], row["found"], row["fallback"]))
    lines += ["", "## Oracle isolation", "",
              "- %s" % payload["oracle_isolation"]["mechanism"],
              "- arm-phase attempts: %d (blocked %d, leaks %d)"
              % (payload["oracle_isolation"]["arm_phase_attempts"],
                 payload["oracle_isolation"]["arm_phase_attempts_blocked"],
                 len(payload["oracle_isolation"]["arm_phase_leaks"])),
              "",
              "## Not in this book", "",
              "- %s" % payload["not_in_this_book"], ""]
    return "\n".join(lines) + "\n"


def _smoke_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    lines = [
        "# S1b smoke -- four arms on curriculum unit 1",
        "",
        "protocol: `%s`  entry: `%s`  backend: **%s**  git: `%s`"
        % (payload["protocol_version"], payload["entry"], payload["backend"],
           payload["git_head"]),
        "",
        "**%s**" % verdict["verdict"],
        "",
        verdict.get("scope", ""),
        "",
        "curriculum revision under test: **%s** (forward order frozen in `%s`)"
        % (payload["curriculum"]["revision"], payload["curriculum_source"]),
        "",
        "unit under test: `%s` (%s, %s; smallest held-in slice %s rows)"
        % (payload["unit_under_test"]["unit_id"],
           payload["unit_under_test"]["group"],
           payload["unit_under_test"]["family_key"],
           payload["unit_under_test"].get("min_slice_rows")),
        "",
        "## Gates",
        "",
    ]
    for key, value in (verdict.get("gates") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    if payload.get("stop"):
        lines += ["", "## Stop", "",
                  "- %s: %s" % (payload["stop"]["verdict"],
                                payload["stop"]["reason"])]
    rows = (payload.get("judging") or {}).get("per_unit_per_arm") or []
    if rows:
        lines += ["", "## Four-arm readout", "",
                  "| arm | deploy | program | held-out utility | menu-oracle | "
                  "regret | worst-class | harm | wrong promo | LLM | fits | "
                  "probes | wasted |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(
                "| %s | %s | %s | %+.4f | %+.4f | %+.4f | %+.4f | %s | %d | "
                "%d | %d | %d | %d |" % (
                    row["arm"], row["deploy_source"],
                    ",".join(step["op"] for step in row["applied_program"])
                    or "identity",
                    row["heldout_utility"], row["menu_oracle_heldout_utility"],
                    row["regret"], row["worst_class_harm"], row["harm_event"],
                    row["wrong_promotions"], row["cost"]["llm"],
                    row["cost"]["consumer_fits"], row["cost"]["probes"],
                    row["wasted_probes"]))
    readable = payload.get("readable_surface_evidence") or {}
    if readable:
        lines += ["", "## Feedback surface: is it readable? (**%s**)"
                  % readable["mode"], "",
                  readable["reading"], "",
                  "- smallest held-in slice: %s rows, resolution %.4f"
                  % (readable["min_slice_rows"], readable["slice_resolution"]),
                  "- programs the proposal stage actually probed: %s"
                  % (readable["programs_probed"] or "none"),
                  "- non-NEUTRAL Support receipts: **%d**"
                  % readable["live_non_neutral_count"], ""]
        if readable["live_non_neutral_receipts"]:
            lines += ["| arm | round | program | relation | support gain | "
                      "delayed gain |", "|---|---|---|---|---|---|"]
            for row in readable["live_non_neutral_receipts"]:
                lines.append("| %s | %s | %s | %s | %s | %s |" % (
                    row["arm"], row["round"], row["program"], row["relation"],
                    row["support_gain"], row["delayed_gain"]))
            lines.append("")
        lines += ["Arithmetic side-evidence from the sealed oracle:", "",
                  "| program | legal | pooled held-in | |m| >= 1/slice | "
                  "material | probed in this smoke |",
                  "|---|---|---|---|---|---|"]
        for row in readable["arithmetic_corroboration"]:
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                row["program"], row["legal"], row["pooled_heldin_headroom"],
                row["expressible_on_the_smallest_slice"], row["material"],
                row["was_probed_in_the_smoke"]))
        lines += ["", "- %s" % readable["caveat"]]
    boundary = payload.get("unit_boundary_state") or {}
    if boundary:
        lines += ["", "## State at the unit boundary", "",
                  "next unit would be `%s`" % boundary.get("next_unit_id"),
                  "",
                  "| arm | end-of-unit sha | store evolved | episodes at end | "
                  "next base | episodes carried | skills carried |",
                  "|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            row = boundary[arm]
            lines.append("| %s | `%s` | %s | %d | %s | %d | %s |" % (
                arm, str(row["end_of_unit_sha"])[:12],
                row["store_evolved_in_unit"], row["episodes_at_unit_end"],
                row["next_base"], row["episodes_carried"],
                ", ".join(row.get("skills_carried") or []) or "-"))
    isolation = payload.get("arm_isolation") or {}
    if isolation:
        lines += ["", "## Four-arm state isolation", ""]
        for key, value in isolation["checks"].items():
            lines.append("- **%s**: %s" % (key, value))
    binding = payload.get("domain_binding") or {}
    if binding:
        lines += ["", "## Domain binding", "",
                  "- hook 1, every minted Skill stamped: %s"
                  % binding["hook_1_every_minted_skill_is_stamped"],
                  "- hook 1, stamped Skills: %s"
                  % (binding["hook_1_stamped_skills"] or "none minted"),
                  "- hook 2, next unit `%s`; foreign Target-local dropped: %s"
                  % (binding["hook_2_next_unit"],
                     binding["hook_2_all_foreign_target_local_dropped"]),
                  "- hook 2, decisions: %s"
                  % (binding["hook_2_target_local_decisions"] or "none"),
                  "- hook 3, Source card decision: %s"
                  % (binding["hook_3_source_card_decision"] or "no card minted"),
                  "- Episode domain namespaces observed: %s"
                  % binding["episode_domain_namespaces"]]
        probe = binding.get("synthetic_probe")
        if probe:
            lines += ["",
                      "### Synthetic probe of the two walls", "",
                      probe["note"], "",
                      "- hook 2, a capability stamped with unit 1 offered to "
                      "unit 2: carried = %s"
                      % probe["hook_2"]["foreign_domain_capability"]["carried"],
                      "- hook 2, a capability stamped with unit 2 offered to "
                      "unit 2: carried = %s"
                      % probe["hook_2"]["native_domain_capability"]["carried"],
                      "- hook 2 behaves as specified: **%s**"
                      % probe["hook_2"]["behaves_as_specified"],
                      "- hook 3, matching five-axis Scope admits: %s"
                      % probe["hook_3"]["matching_five_axis_scope"]["admits"],
                      "- hook 3, empty pattern intersection admits: %s"
                      % probe["hook_3"]["empty_intersection_scope"]["admits"],
                      "- hook 3, wrong consumer admits: %s"
                      % probe["hook_3"]["wrong_consumer_scope"]["admits"],
                      "- hook 3, pattern mismatch admits: %s (axes that "
                      "differ between unit 1 and unit 2: %s)"
                      % (probe["hook_3"]["pattern_mismatch_scope"]["admits"],
                         probe["hook_3"][
                             "pattern_axes_that_differ_between_unit_1_and_unit_2"]
                         or "none -- the two units share every pattern axis"),
                      "- hook 3 behaves as specified: **%s**"
                      % probe["hook_3"]["behaves_as_specified"]]
    findings = payload.get("instrument_findings") or {}
    census = payload.get("instrument_census") or {}
    if findings:
        lines += ["", "## Instrument finding (blocks S1c: **%s**)"
                  % findings["blocks_s1c"], "",
                  findings["finding"], "",
                  "- relations observed on unit 1: %s"
                  % findings["observed_relations_on_unit_1"],
                  "- harm Episode formed on unit 1: %s"
                  % findings["harm_episode_formed_on_unit_1"],
                  "- guard minted on unit 1: %s"
                  % findings["guard_minted_on_unit_1"],
                  "- the frozen two-round protocol has rows on every unit: %s"
                  % findings["two_round_protocol_has_rows_on_every_unit"],
                  "- units with an empty held-in slice: %s"
                  % (findings["units_with_an_empty_held_in_slice"] or "none"),
                  "", findings["what_would_unblock_it"], "",
                  "| # | unit | group | fit rows | support pool | slice rows "
                  "(r1s/r1d/r2s/r2d) | smallest slice | smallest expressible "
                  "gain |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in census.get("per_unit", []):
            if row.get("error"):
                lines.append("| - | %s | - | - | - | - | - | %s |"
                             % (row["unit_id"], row["error"]))
                continue
            step = row["smallest_expressible_gain"]
            lines.append("| %s | %s | %s | %d | %d | %s | %d | %s |" % (
                row["forward_position"], row["unit_id"], row["group"],
                row["fit_rows"], row["support_pool_rows"],
                "/".join(str(value) for value in row["slice_rows"].values()),
                row["smallest_slice_rows"],
                "empty slice" if step is None else "%.3f" % step))
        lines += ["", "- %s" % census.get("why_this_matters", "")]
    guard = payload.get("guard_channel_feasibility") or {}
    if guard:
        lines += ["", "## Guard channel feasibility", "",
                  guard["reading"], "",
                  "- programs readably harmful on **every** harm unit: %s"
                  % (guard["programs_readably_harmful_on_every_harm_unit"]
                     or "none"),
                  "- %s" % guard["expected_earliest_guard"], "",
                  "| harm unit | # | min slice | resolution | readably "
                  "harmful legal programs (held-in) |",
                  "|---|---|---|---|---|"]
        for row in guard["harm_units"]:
            lines.append("| %s | %s | %s | %.4f | %s |" % (
                row["unit_id"], row["forward_position"], row["min_slice_rows"],
                row["slice_resolution"],
                ", ".join("%s %+.4f" % (name, value) for name, value
                          in row["readably_harmful_legal_programs"].items())
                or "none"))
    deploy = payload.get("deploy_rule_observation") or {}
    if deploy and deploy.get("any_arm_deployed_a_delayed_rejected_winner"):
        lines += ["", "## Deploy-rule observation (inherited, not repaired)",
                  "", deploy["note"], "",
                  "| arm | round | Support winner | delayed relation | Skill "
                  "approved | deployed | deploy source |",
                  "|---|---|---|---|---|---|---|"]
        for row in deploy["rows"]:
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                row["arm"], row["round"], row["support_winner"],
                row["delayed_relations"], row["approved_skill_id"] or "none",
                row["deployed_program"], row["deploy_source"]))
    integration = payload.get("a5_slow_integration") or {}
    if integration:
        lines += ["", "## A5 Slow integration at the boundary", "",
                  "- probe rows: %d; census rows: %d"
                  % (len(integration.get("probe_rows") or []),
                     len(integration.get("census") or [])),
                  "- authorized TRY operators: %s"
                  % (integration.get("authorized_try_operators") or "none"),
                  "- risk-authorized operators: %s"
                  % (integration.get("risk_authorized_operators") or "none"),
                  "- Skill written: %s; execution right granted: %s"
                  % (integration.get("skill_written"),
                     integration.get("execution_right_granted")),
                  "- Slow LLM: %s / %s"
                  % (integration.get("slow_llm_calls"),
                     integration.get("slow_llm_cap"))]
    oracle = payload.get("oracle_isolation") or {}
    probe = payload.get("oracle_guard_selftest") or {}
    lines += ["", "## Oracle isolation", "",
              "- %s" % oracle.get("mechanism"),
              "- deliberate arm-phase probe fired the wall on every reader "
              "surface: **%s** on `%s` -- %s"
              % (probe.get("fired"), probe.get("target") or probe.get("reason"),
                 probe.get("surfaces")),
              "- keys the judging component read (after every arm closed): %s"
              % (oracle.get("judge_phase_keys_read") or "none"),
              "- arm-phase attempts %s, blocked %s, leaks %s"
              % (oracle.get("arm_phase_attempts"),
                 oracle.get("arm_phase_attempts_blocked"),
                 len(oracle.get("arm_phase_leaks") or [])),
              "- unblocked reads by phase: %s"
              % oracle.get("unblocked_reads_by_phase")]
    ledger = payload["ledger"]
    live = bool(payload["obligations"]["live_llm_backend"])
    lines += ["", "## Cost", "",
              "- proposal-backend calls: %d fast + %d slow = %d / %d%s"
              % (ledger["llm_calls_fast"], ledger["llm_calls_slow"],
                 ledger["llm_calls_total"], ledger["llm_cap"],
                 "" if live else ("  (**scripted** backend: these are "
                                  "sealed-probe calls, real LLM spend is 0)")),
              "- Consumer fits: %d" % ledger["consumer_fits"],
              "- wall clock: %.1f s / %d s"
              % (ledger["wall_seconds"], ledger["wall_seconds_cap"]),
              "- downloads: 0",
              "", "## Obligations", ""]
    for key, value in payload["obligations"].items():
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


# =========================================================================== #
# entry points
# =========================================================================== #
def select_and_write() -> int:
    started = time.time()
    payload = select_curriculum()
    payload["wall_seconds"] = round(time.time() - started, 2)
    payload["obligations"] = {
        "methods_package_unmodified": True,
        "runtime_contracts_operators_unmodified": True,
        "shared_runner_unmodified": True,
        "no_llm": True,
        "no_arm_ran": True,
        "no_oracle_rescore": True,
        "sealed_artifacts_not_rewritten": True,
        "full_curriculum_not_run": True,
        "downloads": 0,
    }
    _dump(FROZEN_JSON, payload)
    FROZEN_MD.write_text(_frozen_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "forward": payload["forward_order"],
        "reverse": payload["reverse_order"],
        "shortfalls": payload["shortfalls"],
        "artifact": str(FROZEN_JSON)}, ensure_ascii=False, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-curriculum", action="store_true",
                        help="mechanically select and freeze the seven units")
    parser.add_argument("--smoke", action="store_true",
                        help="four arms on curriculum unit 1, reduced protocol")
    parser.add_argument("--live", action="store_true",
                        help="use the live Fast Agent backend in --smoke")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.select_curriculum:
        return select_and_write()
    if args.smoke:
        return smoke(live=args.live)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
