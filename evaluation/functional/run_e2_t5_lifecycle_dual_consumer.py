"""T5 / #41 -- one Harness entry, two TaskSpecs, two Consumers, one lifecycle.

The single question this runner exists to answer: can the *same* real Harness
entry point accept a different TaskSpec and a different Consumer, execute
safely under that task's native feedback, write Experience back under the
right key, update a Target-local Skill, and have the next round behave
differently because of it?

Everything here goes through ``run_online_round`` / ``open_delayed``.  There
is no second pipeline in this file: what differs between the two arms is the
TaskSpec, the Consumer adapter (one injected ``evaluate_fn``), and the
task-native readings that come back.  If this file had to hand-pick a
Workflow, or stand up a second Harness, the pre-registered stop
``PHASE_T_NO_HARNESS_CAPABILITY`` is the honest answer and the run says so.

Evidence grade: POSITIVE_CONTROL, permanently.  The substrate is the injected
T1 copy and the already-exposed Qcal/Qf Query surfaces.  A confirmed verdict
here says the lifecycle machine closes along the task axis -- it says nothing
about natural-data generalization.

Usage:
  python evaluation/functional/run_e2_t5_lifecycle_dual_consumer.py --smoke-only
  python evaluation/functional/run_e2_t5_lifecycle_dual_consumer.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_e2_t1_flip_control as t1  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import signed_radius as resolver  # noqa: E402
from consumers import anomaly_detection_trainable_v3 as adt  # noqa: E402
from consumers.ad_scope_adapter import ADConsumerAdapter  # noqa: E402
from run_e2_operational_pipeline import (  # noqa: E402
    FROZEN_SURFACE_V9,
    _freeze,
    _verify,
)

from evaluation.functional.task_episode_harness.agentic.runner import (  # noqa: E402
    _default_backend_factory,
)
from evaluation.functional.task_episode_harness.normal_flow import (  # noqa: E402
    NF_BASE_URL,
)
from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    canonical_sha256,
)
from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    MetricSpec,
    anomaly_task_spec_v1,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    TASK_CONSUMER_KEY_FALLBACK,
    classify_relation,
    task_consumer_key,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent,
    public_operator_contract,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore,
)
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    activate_approved,
    open_delayed,
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import (  # noqa: E402
    OPERATOR_METADATA,
    OPERATOR_NAMES,
)

# --------------------------------------------------------------- constants
PROTOCOL_VERSION = "t5_lifecycle_v1"
EVIDENCE_GRADE = "POSITIVE_CONTROL"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t5_lifecycle_v1.json"
OUT_MD = E2 / "t5_lifecycle_v1.md"
H0_ROOT = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"
# The snapshot store lives outside the repository tree, and short.  Two
# Windows-specific reasons, both of which otherwise surface as a bare
# WinError 5 out of the store's os.replace and read like a lifecycle fault:
# the materialized tree appends a 64-char sha directory plus nested skill
# filenames (which the task-scoped Skill IDs made longer), and the repo sits
# under a synced Desktop whose sync agent holds handles on freshly written
# files.  Nothing about the run's semantics depends on where this sits.
STORE_ROOT = Path(tempfile.gettempdir()) / "t5s"

T1_DIR = t1.T1_DIR
T1_ARTIFACT = E2 / "t1_flip_control_v1.json"
QUERY_ROOT = PROJECT_ROOT / "_scratch" / "phase_t" / "injected" / "t1b_query"

PERIOD = 24
HORIZON = 48
CONTEXT_LENGTH = 192
M = float(resolver.MATERIAL_THRESHOLD)  # 0.005

# The action region: three anchors whose windows tile [120, 840) exactly.
# The AD adapter fits on the concatenation of these very windows, so what is
# fitted is what the executor's verifier checked -- there is no second notion
# of "the region the program acted on" anywhere in this run.
ANCHORS = (312, 552, 792)
BLOCK = (120, 840)
SUPPORT_ORIGIN = 840          # last anchor + HORIZON == 840, so all three qualify
DELAYED_ORIGIN = 888          # SUPPORT_ORIGIN + HORIZON
AD_DELAYED_BOUNDARY = 888     # origin < boundary -> Qcal, else Qf
QCAL_REGION = (2600, 3060)
QF_REGION = (2100, 2560)

# The five-entry menu, frozen since T3.  identity is Runtime-reserved and is
# never a registry entry (it is absent from OPERATOR_METADATA entirely).
EXPERIMENT_PROGRAMS: tuple[str, ...] = (
    "outlier_iqr", "outlier_mad", "hampel_filter", "winsorize",
)
MENU: tuple[str, ...] = ("identity",) + EXPERIMENT_PROGRAMS

BUDGET = 2                    # Target Support receipts per round
LLM_BUDGET = 16               # 4 rounds x inspect/propose/select + 4 retries
FORECASTING_RETRAIN_BUDGET = 120
AD_EVALUATION_BUDGET = 180
EXAM_MODEL = "gpt-5.6-sol"

# The legacy literal A3 removes from online_loop; B5's assertion (a) pins it.
LEGACY_KEY_LITERAL = "forecast|ridge|sMASE"

EXCLUDED_FROM_PART0 = (
    "_scratch/", ".a5a3_", "artifacts/experience/", "artifacts/functional/e1",
    ".harness_forks/", "_tmp_",
)


class Stop(Exception):
    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__("%s: %s" % (verdict, reason))
        self.verdict = verdict
        self.reason = reason


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # the snapshot store deliberately lives outside the repository
        return "<outside-repo>/%s" % resolved.name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_text(doc: Mapping[str, Any]) -> str:
    return json.dumps(_plain(doc), indent=2, ensure_ascii=False,
                      sort_keys=False) + "\n"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(PROJECT_ROOT),
                          capture_output=True, text=True).stdout.strip()


# =========================================================================== #
# Part 0
# =========================================================================== #
def _part0() -> dict[str, Any]:
    porcelain = _git("status", "--porcelain", "-uno")
    rows = [line for line in porcelain.splitlines() if line.strip()]
    tracked = [line[3:].strip() for line in rows]
    return {
        "head_commit": _git("rev-parse", "--short", "HEAD"),
        "head_subject": _git("log", "-1", "--pretty=%s"),
        "tracked_modified": tracked,
        "exclusions": list(EXCLUDED_FROM_PART0),
        "note": (
            "the #40b deliverables and the main-line docs revisions were "
            "committed as this round's Part 0 checkpoint before any wiring "
            "was touched; the ledger's C16 verdict "
            "(TASK_SEPARATION_REGRESSION) is carried forward unedited"
        ),
    }


# =========================================================================== #
# Part A -- the wiring, and the assertions that it is one wiring
# =========================================================================== #
def experiment_menu() -> dict[str, Any]:
    """A1: ONE function, both TaskSpecs.

    The experiment's public menu is the four repair programs plus the
    Runtime-reserved identity.  Everything else in the registry is put in the
    TaskSpec's forbidden set -- the *same* forbidden set for both arms, built
    here once, so the two arms cannot differ in what they are allowed to try.
    This constrains the public exam menu only; it selects nothing and orders
    nothing.
    """
    allowed = set(EXPERIMENT_PROGRAMS)
    forbidden = tuple(sorted(n for n in OPERATOR_NAMES if n not in allowed))
    return {"allowlist": tuple(EXPERIMENT_PROGRAMS), "forbidden": forbidden,
            "identity_reserved_by_runtime": "identity" not in OPERATOR_METADATA}


def f_task_spec() -> Any:
    menu = experiment_menu()
    return forecast_task_spec_v1(
        horizon=HORIZON,
        downstream_model_class="pooled_ridge_a1",
        metric=MetricSpec("sMASE", "lower_is_better"),
        forbidden_modifications=menu["forbidden"],
    )


def ad_task_spec() -> Any:
    menu = experiment_menu()
    return anomaly_task_spec_v1(
        downstream_model_class="ad_ridge_train_v3",
        metric=MetricSpec("macro_event_f1", "higher_is_better"),
        forbidden_modifications=menu["forbidden"],
    )


def _allowed_for(spec: Any) -> tuple[str, ...]:
    """What the live Program Supply filter will leave standing for this spec.

    Mirrors fast_agent._allowed_operators' three mechanical filters exactly;
    it is read here only to assert the two arms agree, never to build a pool.
    """
    out: list[str] = []
    for name in OPERATOR_NAMES:
        metadata = OPERATOR_METADATA[name]
        if spec.task_type not in metadata["allowed_tasks"]:
            continue
        if metadata.get("shape_changing"):
            continue
        if spec.is_op_forbidden(name):
            continue
        out.append(name)
    return tuple(out)


def _part_a_assertions() -> dict[str, Any]:
    menu = experiment_menu()
    f_spec, ad_spec = f_task_spec(), ad_task_spec()
    f_allowed, ad_allowed = _allowed_for(f_spec), _allowed_for(ad_spec)
    f_ids = tuple(sorted(c["name"] for c in
                         (public_operator_contract(n) for n in f_allowed)))
    ad_ids = tuple(sorted(c["name"] for c in
                          (public_operator_contract(n) for n in ad_allowed)))

    # ---- A3 key regression, both halves ---------------------------------
    legacy_spec = forecast_task_spec_v1(
        horizon=HORIZON, downstream_model_class="ridge",
        metric=MetricSpec("sMASE", "lower_is_better"))
    legacy_key = task_consumer_key(legacy_spec)
    f_key = task_consumer_key(f_spec)
    ad_key = task_consumer_key(ad_spec)

    checks = [
        {"id": "A1_ad_supply_non_empty",
         "ok": len(ad_allowed) > 0,
         "detail": "AD candidate supply is %d programs" % len(ad_allowed)},
        {"id": "A1_allowlists_identical",
         "ok": set(f_allowed) == set(ad_allowed) == set(EXPERIMENT_PROGRAMS),
         "detail": "F=%s AD=%s" % (sorted(f_allowed), sorted(ad_allowed))},
        {"id": "A1_contract_id_sets_identical",
         "ok": f_ids == ad_ids,
         "detail": "%s" % (list(f_ids),)},
        {"id": "A1_forbidden_sets_identical",
         "ok": (f_spec.forbidden_modifications
                == ad_spec.forbidden_modifications == menu["forbidden"]),
         "detail": "%d forbidden entries, one function" % len(menu["forbidden"])},
        {"id": "A1_identity_reserved_not_registered",
         "ok": bool(menu["identity_reserved_by_runtime"]),
         "detail": "identity is absent from OPERATOR_METADATA"},
        {"id": "A1_other_ad_bans_untouched",
         "ok": all("anomaly_detection" not in
                   OPERATOR_METADATA[n]["allowed_tasks"]
                   for n in ("smooth_ma", "smooth_ema", "denoise_savgol",
                             "denoise_stl", "stl_decompose")),
         "detail": "smoothing/decompose AD bans left as they were"},
        {"id": "A3_legacy_fixture_key_matches_old_literal",
         "ok": legacy_key == LEGACY_KEY_LITERAL,
         "detail": "task_consumer_key(forecast/ridge/sMASE) == %r" % legacy_key},
        {"id": "A3_actual_f_key_is_minted_not_literal",
         "ok": f_key == task_consumer_key(f_spec) and f_key != LEGACY_KEY_LITERAL,
         "detail": "actual F key %r; old literal %r (differs, as expected)"
                   % (f_key, LEGACY_KEY_LITERAL)},
        {"id": "A3_ad_key_distinct",
         "ok": ad_key not in (f_key, LEGACY_KEY_LITERAL),
         "detail": "AD key %r" % ad_key},
        {"id": "A3_no_hardcoded_key_left_in_online_loop",
         # a quoted occurrence would be the write-back dialect coming back;
         # the surviving mention is the comment that explains why it left
         "ok": ('"%s"' % LEGACY_KEY_LITERAL) not in (
             PROJECT_ROOT / "methods" / "ttha" / "online_loop.py"
         ).read_text(encoding="utf-8"),
         "detail": ("no quoted forecast|ridge|sMASE remains in online_loop; "
                    "the only surviving mention is the comment recording the "
                    "removal")},
        {"id": "A3_fallback_still_available_for_none_spec",
         "ok": task_consumer_key(None) == TASK_CONSUMER_KEY_FALLBACK
               == LEGACY_KEY_LITERAL,
         "detail": "task_spec=None keeps the historical default"},
    ]
    return {
        "menu": {"allowlist": list(menu["allowlist"]),
                 "forbidden_count": len(menu["forbidden"]),
                 "forbidden": list(menu["forbidden"])},
        "task_specs": {"forecasting": f_spec.to_dict(),
                       "anomaly_detection": ad_spec.to_dict()},
        "public_operator_contract_ids": {"forecasting": list(f_ids),
                                         "anomaly_detection": list(ad_ids)},
        "key_migration": {
            "legacy_literal_removed_from_online_loop": LEGACY_KEY_LITERAL,
            "legacy_fixture_key": legacy_key,
            "actual_forecasting_key": f_key,
            "anomaly_detection_key": ad_key,
            "note": (
                "the old literal was itself a dialect: the real forecasting "
                "Consumer on this substrate is the pooled ridge, so the "
                "minted key differs from the literal by design.  A byte-equal "
                "assertion here would only have forced a false green"
            ),
        },
        "checks": checks,
        "all_passed": all(c["ok"] for c in checks),
    }


def _wiring_diff() -> dict[str, Any]:
    # git-visible spellings: SelfEvolvingHarnessTS/ is a symlink back to the
    # repository root, so `git diff -- SelfEvolvingHarnessTS/...` matches
    # nothing even though the import path is real.
    files = ("operators/registry.py",
             "methods/ttha/online_loop.py",
             "methods/ttha/method.py")
    out: dict[str, Any] = {"tracked": {}, "new_files": {}}
    for rel in files:
        out["tracked"][rel] = {
            "diff": _git("diff", "--", rel),
            "stat": _git("diff", "--stat", "--", rel),
            "sha256_now": _sha256(PROJECT_ROOT / rel),
        }
    adapter = "evaluation/functional/consumers/ad_scope_adapter.py"
    out["new_files"][adapter] = {
        "sha256": _sha256(PROJECT_ROOT / adapter),
        "lines": len((PROJECT_ROOT / adapter).read_text(
            encoding="utf-8").splitlines()),
        "note": "new file; its whole content is the diff",
    }
    out["v9_membership"] = {
        rel: (rel in set(FROZEN_SURFACE_V9))
        for rel in list(files) + [adapter]
    }
    out["v9_touched_this_round"] = sorted(
        rel for rel in list(files) + [adapter] if rel in set(FROZEN_SURFACE_V9)
    )
    return out


# =========================================================================== #
# substrate
# =========================================================================== #
# The forecasting arm needs evaluation rows; the T1 injected copy carries
# only the twelve training stations (T1's own four eval series were never
# injected, and T1b's AD arm never needed them).  Rather than reach outside
# the injected copy for four more series, the F arm splits the twelve it
# already has: the first eight are its training rows and the last four its
# evaluation rows.  The AD arm keeps all twelve, because its scored surface
# is the Query region, which is disjoint from the training block entirely.
# This is a device choice, and it is reported as one.
F_TRAIN_COUNT = 8


def _load_substrate() -> dict[str, Any]:
    doc = json.loads(T1_ARTIFACT.read_text(encoding="utf-8"))
    train = [str(u) for u in doc["roster"]["train"]]
    values: dict[str, np.ndarray] = {}
    for uid in train:
        values[uid] = np.asarray(np.load(T1_DIR / ("%s.npy" % uid)),
                                 dtype=np.float64)
    return {"train": train,
            "f_train": train[:F_TRAIN_COUNT],
            "eval": train[F_TRAIN_COUNT:],
            "values": values,
            "unloaded_t1_eval_uids": [str(u) for u in doc["roster"]["eval"]],
            "source": _repo_rel(T1_DIR)}


def _f_config() -> dict[str, object]:
    return {"dataset_id": "t5_injected_block", "sampling": "hourly_regular",
            "period": PERIOD, "anchors": list(ANCHORS),
            "support_origin": SUPPORT_ORIGIN,
            "selection_origin": SUPPORT_ORIGIN}


def _ad_config() -> dict[str, object]:
    return dict(_f_config())


def _f_roster(sub: Mapping[str, Any]) -> list[dict[str, str]]:
    return ([{"series_uid": u, "role": "train"} for u in sub["f_train"]]
            + [{"series_uid": u, "role": "eval"} for u in sub["eval"]])


def _ad_roster(sub: Mapping[str, Any]) -> list[dict[str, str]]:
    # AD scores the twelve stations it trains on; there is no separate eval
    # roster, and the verifier only ever reads train rows.
    return [{"series_uid": u, "role": "train"} for u in sub["train"]]


class _CountingForecastEval:
    """v6._evaluate with a retrain counter and a deterministic memo."""

    def __init__(self, budget: int) -> None:
        self.budget = int(budget)
        self.retrains = 0
        self.calls: list[dict[str, Any]] = []
        self._memo: dict[tuple[str, int], dict[str, Any]] = {}

    def __call__(self, roster, values, compiled, config, *, origin):
        from consumers.ad_scope_adapter import _steps_signature
        key = (_steps_signature(compiled), int(origin))
        if key in self._memo:
            self.calls.append({"signature": key[0], "origin": key[1],
                               "cache": "hit", "retrains": 0})
            return dict(self._memo[key])
        if self.retrains + 1 > self.budget:
            raise RuntimeError("forecasting retrain budget exceeded")
        mapped = [dict(row, role="eval") if str(row["role"]) != "train"
                  else dict(row) for row in roster]
        out = v6._evaluate(mapped, values, compiled, config, origin=origin)
        self.retrains += 1
        self.calls.append({"signature": key[0], "origin": key[1],
                           "cache": "miss", "retrains": 1,
                           "mean_smase": float(out["mean_smase"])})
        self._memo[key] = dict(out)
        return dict(out)


def _build_executors(sub: Mapping[str, Any]) -> dict[str, Any]:
    f_eval = _CountingForecastEval(FORECASTING_RETRAIN_BUDGET)
    f_exec = ScopeExecutor(_f_roster(sub), sub["values"], _f_config(),
                           evaluate_fn=f_eval)
    ad_eval = ADConsumerAdapter(
        adt=adt, apply_program=v6._apply_program, train_uids=sub["train"],
        block=BLOCK, anchors=ANCHORS, query_root=QUERY_ROOT,
        train_ledger_path=T1_DIR / "ledger.json",
        qcal_region=QCAL_REGION, qf_region=QF_REGION,
        delayed_boundary=AD_DELAYED_BOUNDARY,
        evaluation_budget=AD_EVALUATION_BUDGET,
    )
    ad_exec = ScopeExecutor(_ad_roster(sub), sub["values"], _ad_config(),
                            evaluate_fn=ad_eval)
    return {"forecasting": {"executor": f_exec, "reader": f_eval},
            "anomaly_detection": {"executor": ad_exec, "reader": ad_eval}}


def _request_for(arm: str, sub: Mapping[str, Any], origin: int) -> Any:
    spec = f_task_spec() if arm == "forecasting" else ad_task_spec()
    values = sub["values"]
    series0 = values[sub["train"][0]]
    observed = dict(resolver.window_context(values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    return PreparationRequest("t5-%s" % arm, series0[:origin], spec,
                              dict(observed))


def _card_builder_for(arm: str):
    """A5: the Skill's observable applicability claims task_kind isolation and
    nothing else.  consumer identity is not in the Observation vocabulary, so
    claiming same-task cross-Consumer isolation here would be a claim the
    retrieval gate cannot actually enforce."""
    task_kind = "forecast" if arm == "forecasting" else "anomaly_detection"

    def build(_episode: object) -> Mapping[str, object]:
        return {"pattern_id": "t5-%s-block" % arm,
                "observable_signature": {"task_kind": task_kind}}
    return build


def _fresh_store(tag: str) -> dict[str, Any]:
    # A per-process root rather than an rmtree of a fixed path: on Windows a
    # directory that was just removed can still hold the store's os.replace
    # off with WinError 5, and a store that fails to materialize would be
    # read as a lifecycle fault when it is only a filesystem race.
    root = STORE_ROOT / tag
    if root.exists():
        shutil.rmtree(root)
    store = SnapshotStore(root / "snapshots")
    snapshot = compile_snapshot(H0_ROOT, verify_lock=False)
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    return {"store": store, "controller": controller, "snapshot": snapshot,
            "root": _repo_rel(root),
            "runtime_bundle_sha": snapshot.runtime_bundle_sha,
            "harness_content_sha": snapshot.harness_content_sha,
            "skill_ids": [s.skill_id for s in snapshot.skills]}


def _skill_rows(snapshot: Any) -> list[dict[str, Any]]:
    rows = []
    for skill in snapshot.skills:
        guards = dict(skill.risk_guards or {})
        rows.append({
            "skill_id": skill.skill_id,
            "kind": str(getattr(skill.skill_kind, "name", skill.skill_kind)),
            "requires_target_support": bool(
                guards.get("requires_target_support")),
            "restricted": bool(guards.get("restricted")
                               or guards.get("target_disconfirmed")),
            "applicability": _plain(skill.observable_applicability),
        })
    return rows


def _view_skill_ids(snapshot: Any, task_kind: str) -> list[str]:
    view = resolve_harness_view(snapshot, {"task_kind": task_kind})
    ids: list[str] = []
    for attr in ("capability_skills", "skills", "retrieved_skills"):
        entries = getattr(view, attr, None)
        if entries:
            ids.extend(str(getattr(e, "skill_id", e)) for e in entries)
    return sorted(set(ids))


# =========================================================================== #
# Part B -- the zero-LLM vertical smoke tests
# =========================================================================== #
class _ScriptedReadings:
    """A reading table in the executor's own shape.

    Part B is a wiring acceptance, not new evidence: the numbers below are
    fixtures chosen to put each lifecycle cell under test, and they travel
    the real ``run_online_round`` / ``open_delayed`` / verifier path exactly
    as a live reading would.
    """

    def __init__(self, table: Mapping[tuple[str, int], tuple[float, list[float]]],
                 *, views: int) -> None:
        self.table = dict(table)
        self.views = int(views)
        self.seen: list[tuple[str, int]] = []

    @staticmethod
    def _op(compiled: Any) -> str:
        """Key on the operator name alone.  The bound parameters come from
        the live contract and are not something a fixture table should have
        to predict; what the cell under test depends on is which program ran."""
        from consumers.ad_scope_adapter import compiled_steps
        steps = compiled_steps(compiled)
        return str(steps[0][0]) if steps else "identity"

    def __call__(self, roster, values, compiled, config, *, origin):
        sig = self._op(compiled)
        self.seen.append((sig, int(origin)))
        if (sig, int(origin)) not in self.table:
            raise KeyError(
                "scripted readings have no entry for (%s, %d); the fixture "
                "must cover every program the round actually runs"
                % (sig, int(origin)))
        mean, per_view = self.table[(sig, int(origin))]
        return {"mean_smase": float(mean),
                "per_view_smase": [float(v) for v in per_view],
                "behavior_point_count": 1 if sig != "identity" else 0}


def _scripted_method(snapshot: Any, series0: np.ndarray, origin: int,
                     operators: Sequence[str], task_kind: str) -> Any:
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=tuple(operators),
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(series0[:origin], task_kind=task_kind))
    return TTHAMethod(sealed.TTHAFastAgent(core), snapshot, ())


def _b_round(*, arm: str, sub: Mapping[str, Any], readings: _ScriptedReadings,
             operators: Sequence[str], tag: str, store_bundle: Mapping[str, Any],
             method: Any = None) -> dict[str, Any]:
    roster = _f_roster(sub) if arm == "forecasting" else _ad_roster(sub)
    executor = ScopeExecutor(roster, sub["values"],
                             _f_config(), evaluate_fn=readings)
    request = _request_for(arm, sub, SUPPORT_ORIGIN)
    series0 = sub["values"][sub["train"][0]]
    task_kind = "forecast" if arm == "forecasting" else "anomaly_detection"
    if method is None:
        method = _scripted_method(store_bundle["snapshot"], series0,
                                  SUPPORT_ORIGIN, operators, task_kind)
    result = run_online_round(
        method, executor, request, sub["values"],
        origin=SUPPORT_ORIGIN, slow_agent=None,
        controller=store_bundle["controller"], store=store_bundle["store"],
        card_builder=_card_builder_for(arm), round_name=tag, budget=BUDGET,
        allow_slow=False, domain="t5:%s" % arm, period=PERIOD,
        fast_features=dict(extract_public_features(
            series0[:SUPPORT_ORIGIN], task_kind=task_kind)),
        allow_fast_skill=True, runtime_prior_slot=False)
    open_delayed(result, executor, delayed_origin=DELAYED_ORIGIN,
                 store=store_bundle["store"])
    activated = False
    if result.approved_skill_id is not None:
        activated = activate_approved(result, store_bundle["store"])
    episodes = [e for e in method.experience_episodes]
    return {
        "arm": arm, "method": method, "result": result,
        "episodes": episodes,
        "winner_program": _plain(result.winner_program),
        "probes": [{"candidate_id": p["candidate_id"], "gain": p.get("gain")}
                   for p in result.actual_probed_programs],
        "fast_skill_event": _plain(result._fast_skill_event),
        "delayed_event": _plain(result._delayed_event),
        "approved_skill_id": result.approved_skill_id,
        "activated": activated,
        "episode_rows": [{
            "episode_id": e.episode_id,
            "task_consumer_key": e.task_consumer_key,
            "workflow_signature": e.workflow_signature,
            "relation": e.relation,
            "evidence_level": e.evidence_level,
            "local_status": e.local_status,
        } for e in episodes],
        "snapshot_skills": _skill_rows(method._active_snapshot()),
    }


def _part_b(sub: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}

    def check(cid: str, ok: bool, text: str) -> None:
        checks.append({"id": cid, "ok": bool(ok), "detail": text})

    f_views, ad_views = len(sub["eval"]), len(sub["train"])

    # ---- B1 / B5: forecasting, outlier_iqr, clean positive both stages ----
    iqr = "outlier_iqr"
    f_table = {
        ("identity", SUPPORT_ORIGIN): (1.0, [1.0] * f_views),
        ("identity", DELAYED_ORIGIN): (1.0, [1.0] * f_views),
        (iqr, SUPPORT_ORIGIN): (0.90, [0.90] * f_views),
        (iqr, DELAYED_ORIGIN): (0.92, [0.92] * f_views),
    }
    store_f = _fresh_store("b1")
    b1 = _b_round(arm="forecasting", sub=sub,
                  readings=_ScriptedReadings(f_table, views=f_views),
                  operators=("outlier_iqr",), tag="b1", store_bundle=store_f)
    detail["B1"] = {k: v for k, v in b1.items()
                    if k not in ("method", "result", "episodes")}
    ep1 = b1["episode_rows"][0] if b1["episode_rows"] else {}
    check("B1_support_positive_draft",
          any(r["workflow_signature"] == "outlier_iqr" for r in b1["episode_rows"]),
          "outlier_iqr episode written")
    check("B1_winner_formed", b1["winner_program"] is not None,
          "winner=%s" % (b1["winner_program"],))
    check("B1_draft_pending",
          (b1["fast_skill_event"] or {}).get("stage") == "pending",
          "fast winner stage=%s" % ((b1["fast_skill_event"] or {}).get("stage"),))
    check("B1_delayed_positive_active",
          ep1.get("relation") == "POSITIVE"
          and ep1.get("local_status") == "LOCAL_ACTIVE",
          "episode relation=%s status=%s"
          % (ep1.get("relation"), ep1.get("local_status")))
    check("B1_delayed_approved",
          (b1["delayed_event"] or {}).get("stage") == "approved",
          "delayed stage=%s" % ((b1["delayed_event"] or {}).get("stage"),))
    check("B5_forecasting_positive_delayed_still_approves",
          (b1["delayed_event"] or {}).get("stage") == "approved"
          and (b1["delayed_event"] or {}).get("delayed_relation") == "POSITIVE",
          "the tightened gate still approves an unambiguously positive delayed")

    # ---- B2 / B6: AD, hampel_filter, aggregate up but one series harmed ---
    ham = "hampel_filter"
    ad_per_view_good = [-0.50] * ad_views
    ad_per_view_cand = [-0.56] * ad_views
    ad_per_view_cand[3] = -0.44          # this station lost 0.06 of F1
    ad_table = {
        ("identity", SUPPORT_ORIGIN): (-0.50, list(ad_per_view_good)),
        ("identity", DELAYED_ORIGIN): (-0.50, list(ad_per_view_good)),
        (ham, SUPPORT_ORIGIN): (-0.55, list(ad_per_view_cand)),
        (ham, DELAYED_ORIGIN): (-0.55, list(ad_per_view_cand)),
    }
    store_ad = _fresh_store("b2")
    b2 = _b_round(arm="anomaly_detection", sub=sub,
                  readings=_ScriptedReadings(ad_table, views=ad_views),
                  operators=("hampel_filter",), tag="b2",
                  store_bundle=store_ad)
    detail["B2"] = {k: v for k, v in b2.items()
                    if k not in ("method", "result", "episodes")}
    ep2 = b2["episode_rows"][0] if b2["episode_rows"] else {}
    check("B2_ad_candidate_supplied",
          bool(b2["probes"]),
          "AD probed %d candidate(s): %s" % (len(b2["probes"]), b2["probes"]))
    check("B2_conflict_recorded", ep2.get("relation") == "CONFLICT",
          "AD episode relation=%s (aggregate up, one series harmed)"
          % ep2.get("relation"))
    check("B2_conflict_grants_no_execution",
          b2["winner_program"] is None and b2["approved_skill_id"] is None
          and ep2.get("local_status") == "EPISODE_ONLY",
          "winner=%s approved=%s status=%s"
          % (b2["winner_program"], b2["approved_skill_id"],
             ep2.get("local_status")))
    check("B2_no_ad_skill_in_snapshot",
          not any(r["skill_id"].startswith("fast_winner_anomaly")
                  for r in b2["snapshot_skills"]),
          "no AD fast_winner skill was written")

    # ---- B3: keys, and zero cross-task retrieval -------------------------
    f_key = task_consumer_key(f_task_spec())
    ad_key = task_consumer_key(ad_task_spec())
    check("B3_forecasting_key_correct",
          all(r["task_consumer_key"] == f_key for r in b1["episode_rows"]),
          "F episodes keyed %s" % f_key)
    check("B3_ad_key_correct",
          all(r["task_consumer_key"] == ad_key for r in b2["episode_rows"]),
          "AD episodes keyed %s" % ad_key)
    check("B3_zero_cross_task",
          f_key != ad_key
          and not (set(r["task_consumer_key"] for r in b1["episode_rows"])
                   & set(r["task_consumer_key"] for r in b2["episode_rows"])),
          "the two arms' key sets are disjoint")

    # ---- B4: the next F round can retrieve the F Skill; AD cannot --------
    f_snapshot = b1["method"]._active_snapshot()
    f_visible = _view_skill_ids(f_snapshot, "forecast")
    ad_visible = _view_skill_ids(f_snapshot, "anomaly_detection")
    learned = [r["skill_id"] for r in _skill_rows(f_snapshot)
               if r["skill_id"].startswith("fast_winner_")]
    detail["B4"] = {"learned_skills": learned,
                    "visible_to_forecast": f_visible,
                    "visible_to_anomaly_detection": ad_visible,
                    "skill_rows": _skill_rows(f_snapshot)}
    check("B4_f_skill_written", bool(learned),
          "learned skills after B1: %s" % learned)
    check("B4_next_f_round_retrieves",
          bool(learned) and all(s in f_visible for s in learned),
          "forecast view carries %s" % learned)
    check("B4_next_ad_round_cannot_read_f_skill",
          not (set(learned) & set(ad_visible)),
          "anomaly_detection view carries %s" % ad_visible)
    check("B4_skill_id_is_task_scoped",
          all(s.startswith("fast_winner_forecast_") for s in learned),
          "task-scoped, hash-free ids: %s" % learned)

    # ---- B6: the conflict revocation cell, on the forecasting side -------
    # POSITIVE Support -> Draft, then a delayed whose aggregate is up but
    # which harms one evaluation series: CONFLICT, no approval, pending
    # discarded, both raw readings kept.
    mad = "outlier_mad"
    b6_per_view = [0.95] * f_views
    b6_per_view[1] = 1.02                # this series got worse than baseline
    b6_table = {
        ("identity", SUPPORT_ORIGIN): (1.0, [1.0] * f_views),
        ("identity", DELAYED_ORIGIN): (1.0, [1.0] * f_views),
        (mad, SUPPORT_ORIGIN): (0.90, [0.90] * f_views),
        (mad, DELAYED_ORIGIN): (0.94, list(b6_per_view)),
    }
    store_b6 = _fresh_store("b6")
    b6 = _b_round(arm="forecasting", sub=sub,
                  readings=_ScriptedReadings(b6_table, views=f_views),
                  operators=("outlier_mad",), tag="b6", store_bundle=store_b6)
    detail["B6"] = {k: v for k, v in b6.items()
                    if k not in ("method", "result", "episodes")}
    ep6 = b6["episode_rows"][0] if b6["episode_rows"] else {}
    dev6 = b6["delayed_event"] or {}
    ev6 = dev6.get("delayed_evidence") or {}
    check("B6_support_positive_then_draft",
          (b6["fast_skill_event"] or {}).get("stage") == "pending",
          "Support POSITIVE reached pending Draft")
    check("B6_delayed_conflict",
          ep6.get("relation") == "CONFLICT" and dev6.get(
              "delayed_relation") == "CONFLICT",
          "episode=%s gate=%s" % (ep6.get("relation"),
                                  dev6.get("delayed_relation")))
    check("B6_not_approved_pending_discarded",
          dev6.get("stage") == "delayed_rejected"
          and b6["approved_skill_id"] is None
          and b6["method"]._pending_update is None,
          "stage=%s approved=%s pending=%s"
          % (dev6.get("stage"), b6["approved_skill_id"],
             b6["method"]._pending_update))
    check("B6_skill_restricted",
          ep6.get("local_status") == "RESTRICTED",
          "episode local_status=%s" % ep6.get("local_status"))
    check("B6_raw_readings_retained",
          ev6.get("aggregate_gain") is not None
          and int(ev6.get("series_read") or 0) == f_views
          and int(ev6.get("harmed_series_count") or 0) >= 1,
          "aggregate=%s series_read=%s harmed=%s"
          % (ev6.get("aggregate_gain"), ev6.get("series_read"),
             ev6.get("harmed_series_count")))

    return {"checks": checks, "detail": detail,
            "all_passed": all(c["ok"] for c in checks),
            "llm_calls": 0}


# =========================================================================== #
# Part C -- the live interleaved trajectory
# =========================================================================== #
ROUND_ORDER: tuple[tuple[str, str], ...] = (
    ("forecasting", "r1"),
    ("anomaly_detection", "r1"),
    ("forecasting", "r2"),
    ("anomaly_detection", "r2"),
)


def _live_method(snapshot: Any, series0: np.ndarray, origin: int,
                 backend: Any, task_kind: str, memory: tuple) -> Any:
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(series0[:origin], task_kind=task_kind),
        model=EXAM_MODEL, base_url=NF_BASE_URL)
    return TTHAMethod(TTHAFastAgent(core), snapshot, memory)


def _part_c(sub: Mapping[str, Any], executors: Mapping[str, Any]
            ) -> dict[str, Any]:
    """One continuous Method and one fresh empty store across all four rounds.

    Nothing is pre-seeded: whatever the second round of a task sees, that
    task's first round wrote.  The rounds alternate so the single pending
    slot is opened and closed before the other task needs it, and so the AD
    round that directly follows an activated F Skill is a live test of
    cross-task retrieval rather than an assertion about one.
    """
    bundle = _fresh_store("pc")
    backend = _default_backend_factory(LLM_BUDGET)
    series0 = sub["values"][sub["train"][0]]
    method = _live_method(bundle["snapshot"], series0, SUPPORT_ORIGIN,
                          backend, "forecast", ())
    rounds: list[dict[str, Any]] = []
    stopped: str | None = None

    for arm, tag in ROUND_ORDER:
        task_kind = "forecast" if arm == "forecasting" else "anomaly_detection"
        executor = executors[arm]["executor"]
        request = _request_for(arm, sub, SUPPORT_ORIGIN)
        # No manual rebinding here: run_online_round's first act is
        # method.bind_round_data(..., task_kind=request.task_spec.task_type),
        # so the public-tool gateway follows the TaskSpec on its own.  That is
        # the point -- one entry, and the task binding travels through it.
        features = dict(extract_public_features(
            series0[:SUPPORT_ORIGIN], task_kind=task_kind))
        row: dict[str, Any] = {
            "arm": arm, "round": tag,
            "task_spec_sha": request.task_spec.sha(),
            "task_consumer_key": task_consumer_key(request.task_spec),
            "public_features_sha256": canonical_sha256(_plain(features)),
            "llm_calls_before_round": int(getattr(backend, "calls", 0)),
            "snapshot_before": method._active_snapshot().harness_content_sha,
            "skills_before": _skill_rows(method._active_snapshot()),
            "view_skill_ids_before": _view_skill_ids(
                method._active_snapshot(), task_kind),
            "episodes_before": len(list(method.experience_episodes)),
        }
        try:
            result = run_online_round(
                method, executor, request, sub["values"],
                origin=SUPPORT_ORIGIN, slow_agent=None,
                controller=bundle["controller"], store=bundle["store"],
                card_builder=_card_builder_for(arm), round_name="%s_%s" % (
                    arm, tag),
                budget=BUDGET, allow_slow=False, domain="t5:%s" % arm,
                period=PERIOD, fast_features=features,
                allow_fast_skill=True, runtime_prior_slot=False)
        except Exception as exc:  # noqa: BLE001
            row["error"] = "%s: %s" % (type(exc).__name__, exc)
            rounds.append(row)
            stopped = _classify_live_error(exc)
            break
        trace = method.last_trace
        row.update({
            "pool": list(getattr(trace, "candidate_ids", ()) or ()),
            "chosen": getattr(trace, "chosen_candidate_id", None),
            "memory_resolution": getattr(trace, "memory_resolution_status",
                                         None),
            "proposal_count": result.proposal_count,
            "probes": [{"candidate_id": p["candidate_id"],
                        "kind": p.get("kind"), "gain": p.get("gain")}
                       for p in result.actual_probed_programs],
            "winner_program": _plain(result.winner_program),
            "receipts": result.target_support_receipts_used,
            "harm_count": result.harm_count,
            "fast_skill_event": _plain(result._fast_skill_event),
        })
        try:
            open_delayed(result, executor, delayed_origin=DELAYED_ORIGIN,
                         store=bundle["store"])
        except Exception as exc:  # noqa: BLE001
            row["delayed_error"] = "%s: %s" % (type(exc).__name__, exc)
            rounds.append(row)
            stopped = _classify_live_error(exc)
            break
        activated = False
        if result.approved_skill_id is not None:
            activated = activate_approved(result, bundle["store"])
        episodes = list(method.experience_episodes)
        row.update({
            "delayed_event": _plain(result._delayed_event),
            "delayed_utility": result.delayed_utility,
            "approved_skill_id": result.approved_skill_id,
            "activated": activated,
            "episode_rows": [{
                "episode_id": e.episode_id,
                "task_consumer_key": e.task_consumer_key,
                "workflow_signature": e.workflow_signature,
                "relation": e.relation,
                "evidence_level": e.evidence_level,
                "local_status": e.local_status,
                "aggregate_gain": (e.delayed_response or {}).get("gain"),
                "per_series": _plain(
                    ((e.delayed_response or {}).get("measured_effect") or {})
                ) or _plain(
                    ((e.support_response or {}).get("measured_effect") or {})
                ),
            } for e in episodes[row["episodes_before"]:]],
            "episodes_after": len(episodes),
            "snapshot_after": method._active_snapshot().harness_content_sha,
            "skills_after": _skill_rows(method._active_snapshot()),
            "view_skill_ids_after": _view_skill_ids(
                method._active_snapshot(), task_kind),
            "llm_calls_after_round": int(getattr(backend, "calls", 0)),
        })
        rounds.append(row)

    return {
        "store": bundle["root"],
        "h0_runtime_bundle_sha": bundle["runtime_bundle_sha"],
        "pre_seeded_cards": 0,
        "rounds": rounds,
        "stopped": stopped,
        # returned_models is a *set of model names*, not a call counter -- the
        # budgeted backend's own `calls` is the reading.
        "llm_calls": int(getattr(backend, "calls", 0)),
        "llm_budget": LLM_BUDGET,
        "prompt_tokens": int(getattr(backend, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(backend, "completion_tokens", 0)),
        "models_returned": sorted(set(
            getattr(backend, "returned_models", ()) or ())),
    }


def _classify_live_error(exc: Exception) -> str:
    text = "%s: %s" % (type(exc).__name__, exc)
    if "budget" in text.lower():
        return "INCOMPLETE_LLM_BUDGET"
    if "AgentProtocolError" in text or "StagePostValidation" in text:
        return "AGENT_PROTOCOL_UNREADABLE"
    return "PROTOCOL_FAILURE"


# =========================================================================== #
# Part C readings: r1 vs r2, and the attribution assertion
# =========================================================================== #
def _r1_r2_comparison(part_c: Mapping[str, Any]) -> dict[str, Any]:
    by = {(r["arm"], r["round"]): r for r in part_c["rounds"]}
    out: dict[str, Any] = {"arms": {}, "assertions": []}
    for arm in ("forecasting", "anomaly_detection"):
        r1, r2 = by.get((arm, "r1")), by.get((arm, "r2"))
        if not r1 or not r2 or "error" in r1 or "error" in r2:
            out["arms"][arm] = {"incomplete": True}
            continue
        changed = {
            "pool": (r1.get("pool") != r2.get("pool")),
            "chosen": (r1.get("chosen") != r2.get("chosen")),
            "winner": (r1.get("winner_program") != r2.get("winner_program")),
            "probes": ([p["candidate_id"] for p in r1.get("probes", [])]
                       != [p["candidate_id"] for p in r2.get("probes", [])]),
            "memory_resolution": (r1.get("memory_resolution")
                                  != r2.get("memory_resolution")),
            "view_skill_ids": (r1.get("view_skill_ids_before")
                               != r2.get("view_skill_ids_before")),
        }
        out["arms"][arm] = {
            "r1": {k: r1.get(k) for k in (
                "pool", "chosen", "memory_resolution", "winner_program",
                "probes", "view_skill_ids_before", "episodes_before",
                "approved_skill_id", "delayed_utility")},
            "r2": {k: r2.get(k) for k in (
                "pool", "chosen", "memory_resolution", "winner_program",
                "probes", "view_skill_ids_before", "episodes_before",
                "approved_skill_id", "delayed_utility")},
            "changed_fields": changed,
            "behavior_changed": any(changed.values()),
        }
        out["assertions"].append({
            "id": "%s_context_identical_between_rounds" % arm,
            "ok": (r1["task_spec_sha"] == r2["task_spec_sha"]
                   and r1["public_features_sha256"] == r2["public_features_sha256"]
                   and r1["task_consumer_key"] == r2["task_consumer_key"]),
            "detail": (
                "task_spec sha %s, public features sha %s and Consumer key %s "
                "are the same bytes in both rounds, so any behaviour "
                "difference has to come from what the trajectory itself wrote"
                % (r1["task_spec_sha"], r1["public_features_sha256"][:12],
                   r1["task_consumer_key"])),
        })
        out["assertions"].append({
            "id": "%s_memory_or_skill_grew_between_rounds" % arm,
            "ok": (r2.get("episodes_before", 0) > r1.get("episodes_before", 0)
                   or r1.get("view_skill_ids_after")
                   != r2.get("view_skill_ids_before")
                   or bool(r1.get("approved_skill_id"))),
            "detail": "episodes %s -> %s; skills visible before r2: %s"
                      % (r1.get("episodes_before"), r2.get("episodes_before"),
                         r2.get("view_skill_ids_before")),
        })
    return out


def _cross_task_leak(part_c: Mapping[str, Any]) -> dict[str, Any]:
    f_key = task_consumer_key(f_task_spec())
    ad_key = task_consumer_key(ad_task_spec())
    expect = {"forecasting": f_key, "anomaly_detection": ad_key}
    mismatches: list[dict[str, Any]] = []
    leaks: list[dict[str, Any]] = []
    for row in part_c["rounds"]:
        if "error" in row:
            continue
        for ep in row.get("episode_rows", []):
            if ep["task_consumer_key"] != expect[row["arm"]]:
                mismatches.append({"round": row["round"], "arm": row["arm"],
                                   "episode": ep["episode_id"],
                                   "key": ep["task_consumer_key"],
                                   "expected": expect[row["arm"]]})
        wrong_scope = "anomaly_detection" if row["arm"] == "forecasting" \
            else "forecast"
        for sid in row.get("view_skill_ids_before", []):
            if sid.startswith("fast_winner_") and (
                    "_%s_" % wrong_scope) in ("_%s_" % sid):
                leaks.append({"round": row["round"], "arm": row["arm"],
                              "skill_id": sid})
        for sid in row.get("view_skill_ids_before", []):
            if not sid.startswith("fast_winner_"):
                continue
            scope = "forecast" if row["arm"] == "forecasting" \
                else "anomaly_detection"
            if not sid.startswith("fast_winner_%s_" % scope):
                leaks.append({"round": row["round"], "arm": row["arm"],
                              "skill_id": sid, "reason": "out-of-task scope"})
    return {"key_mismatches": mismatches, "skill_leaks": leaks,
            "expected_keys": expect}


# =========================================================================== #
# Part D -- the pre-registered verdict, mechanical blockers first
# =========================================================================== #
def _verdict(*, part_a: Mapping[str, Any], part_b: Mapping[str, Any],
             part_c: Mapping[str, Any] | None,
             comparison: Mapping[str, Any] | None,
             leak: Mapping[str, Any] | None) -> dict[str, Any]:
    def v(name: str, reason: str, **extra: Any) -> dict[str, Any]:
        return {"verdict": name, "reason": reason, **extra}

    # ---- mechanical blockers ------------------------------------------
    a1 = {c["id"]: c for c in part_a["checks"]}
    if not a1["A1_ad_supply_non_empty"]["ok"]:
        return v("AD_PROGRAM_SUPPLY_BLOCKED",
                 a1["A1_ad_supply_non_empty"]["detail"])
    if part_c is None:
        failed = [c["id"] for c in part_b["checks"] if not c["ok"]]
        if any(f.startswith("B2_ad_candidate") for f in failed):
            return v("AD_PROGRAM_SUPPLY_BLOCKED",
                     "the AD arm was supplied no candidate in the smoke round")
        return v("CONSUMER_ADAPTER_UNREADABLE" if failed else "SMOKE_ONLY",
                 "Part B checks failed: %s" % failed if failed
                 else "smoke-only run; Part C not attempted",
                 failed_checks=failed)
    if part_c.get("stopped"):
        return v(part_c["stopped"],
                 "the live trajectory stopped: %s" % part_c["stopped"],
                 completed_rounds=len([r for r in part_c["rounds"]
                                       if "error" not in r]))
    completed = [r for r in part_c["rounds"] if "error" not in r]
    if len(completed) < len(ROUND_ORDER):
        return v("INCOMPLETE_LLM_BUDGET",
                 "only %d of %d rounds completed"
                 % (len(completed), len(ROUND_ORDER)),
                 completed_rounds=len(completed))
    if leak and leak["key_mismatches"]:
        return v("TASK_KEY_WRITEBACK_MISMATCH",
                 "%d episode(s) written under the wrong task key"
                 % len(leak["key_mismatches"]),
                 mismatches=leak["key_mismatches"])
    if leak and leak["skill_leaks"]:
        return v("SKILL_SCOPE_LEAKS_ACROSS_TASKS",
                 "%d Skill(s) visible outside their task"
                 % len(leak["skill_leaks"]), leaks=leak["skill_leaks"])

    # ---- outcome cells -------------------------------------------------
    episodes = [ep for r in completed for ep in r.get("episode_rows", [])]
    relations = [ep["relation"] for ep in episodes]
    adopted = [r for r in completed if r.get("winner_program")]
    if not adopted:
        return v("NO_ADOPTABLE_PLAN_SAMPLE",
                 "no round produced an adoptable plan in this natural sample; "
                 "not re-drawn",
                 relations=relations)
    conflicts = [r for r in completed
                 for ep in r.get("episode_rows", [])
                 if ep["relation"] == "CONFLICT"]
    restricted = [ep for ep in episodes if ep["local_status"] == "RESTRICTED"]
    if conflicts and not restricted and any(
            r.get("approved_skill_id") for r in completed):
        # a CONFLICT was read but nothing was restricted anywhere
        pass
    behaviour = comparison["arms"] if comparison else {}
    changed = [arm for arm, row in behaviour.items()
               if row.get("behavior_changed")]
    retrieved = any(r.get("memory_resolution") not in (None, "no_memory")
                    for r in completed)
    if not conflicts:
        return v("NO_CONFLICT_FEEDBACK_SAMPLE",
                 "the trajectory completed but produced no CONFLICT feedback, "
                 "so the risk branch was never exercised; partial result, not "
                 "a closure",
                 relations=relations, behavior_changed_arms=changed)
    if retrieved and not changed:
        return v("EXPERIENCE_RETRIEVED_BEHAVIOR_UNCHANGED",
                 "experience reached the next round's prompt but no behaviour "
                 "field moved; partial result",
                 relations=relations)
    # delayed-risk restriction must actually bite
    unrestricted_conflict = [
        ep for ep in episodes
        if ep["relation"] == "CONFLICT"
        and ep["local_status"] not in ("RESTRICTED", "EPISODE_ONLY")
    ]
    if unrestricted_conflict:
        return v("DELAYED_RISK_NOT_RESTRICTING_SKILL",
                 "%d CONFLICT episode(s) kept execution rights"
                 % len(unrestricted_conflict),
                 offenders=unrestricted_conflict)
    return v("TASK_CONDITIONED_LIFECYCLE_CLOSES_POSITIVE_CONTROL",
             "one entry, two TaskSpecs and two Consumers: execution safe, "
             "write-back keys correct, CONFLICT recorded as such, delayed "
             "risk restricted execution, zero Skill leakage across tasks, and "
             "the next round's behaviour changed with the trajectory's own "
             "experience as the only thing that moved",
             behavior_changed_arms=changed, relations=relations)



# =========================================================================== #
# what T5 hands back that the verdict field cannot carry (0 LLM)
# =========================================================================== #
def t5_findings(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pure function of the artifact plus git.  Spends nothing.

    The verdict stays where the pre-registered ladder put it: budget
    exhaustion is a mechanical blocker and is read before any outcome cell.
    This section reports the completed segment and the facts the run
    surfaced that the ladder has no field for.
    """
    part_c = payload.get("part_c") or {}
    rounds = [r for r in part_c.get("rounds", []) if "error" not in r]
    failed = [r for r in part_c.get("rounds", []) if "error" in r]
    budget = int((payload.get("budgets") or {}).get("llm") or LLM_BUDGET)
    completed = len(rounds)
    return {
        "llm_budget_arithmetic": {
            "authorized": budget,
            "book_nominal_per_round": 3,
            "book_expected_total": 12,
            "observed_total": budget,
            "observed_rounds_completed": completed,
            "observed_rate_per_round": (
                round(budget / (completed + 1), 2) if completed else None),
            "raw_field_correction": (
                "part_c.llm_calls was read off backend.returned_models, which "
                "is the set of distinct model names, not a call counter. The "
                "true total is the cap itself: the run stopped with "
                "AgentCallBudgetExceeded at %d, which fires only on the call "
                "after the %dth. The runner now reads backend.calls; this "
                "artifact's field is corrected rather than re-measured, "
                "because re-measuring would spend a second round's budget."
                % (budget, budget)),
            "reading": (
                "three rounds and the first stage of a fourth consumed the "
                "whole cap, so the live entry costs about five agent calls "
                "per round, not the three the book's arithmetic assumed. "
                "fast prepare has exactly three stages (inspect / propose / "
                "select), each carrying validation_retries=1, so the four "
                "calls the book set aside as retry headroom were spent as "
                "ordinary traffic rather than left spare. A four-round "
                "trajectory needs roughly 20, not 16. The cap is the book's "
                "and was not raised."),
        },
        "completed_segment": {
            "rounds_completed": completed,
            "round_not_reached": [
                {"arm": r["arm"], "round": r["round"], "error": r["error"]}
                for r in failed],
            "what_the_completed_segment_shows": [
                "F r1: the Agent proposed and probed outlier_mad first "
                "(-0.2166, three of four series harmed -> NEGATIVE), then "
                "hampel_filter (+0.1920, no series harmed -> POSITIVE), which "
                "became the winner and reached a pending Draft Skill.",
                "F r1 delayed: aggregate +0.0450 but one of four series at "
                "-0.0714 -> CONFLICT -> not approved, pending discarded, the "
                "Episode written RESTRICTED. Under the gate this round "
                "replaced (dg >= -0.005) that same reading would have been "
                "APPROVED and written into the active snapshot.",
                "AD r1: the AD arm was supplied candidates at all, which was "
                "the first blocker; it proposed outlier_mad, whose Support "
                "aggregate was +0.2032 with one of twelve series at -0.1905 "
                "-> CONFLICT -> no winner, no Draft, Episode only.",
                "F r2: memory_resolution moved no_memory -> rendered, the "
                "candidate pool changed, and the Agent no longer probed the "
                "harmful outlier_mad at all (1 Support receipt instead of 2, "
                "harm_count 0 instead of 1) -- with TaskSpec, public features "
                "and Consumer key byte-identical to r1.",
            ],
            "what_it_does_not_show": (
                "no Skill was ever approved in the live trajectory, because "
                "every delayed window came back CONFLICT. Target-local Skill "
                "update and live cross-task Skill retrieval were therefore "
                "not exercised; they hold only at the Part B "
                "(scripted-reading) level. The AD arm's second round never "
                "ran."),
        },
        "collateral_on_the_other_line": {
            "what": (
                "the A5 rename to fast_winner_{task}_{model}_{metric}_{op} is "
                "a naming contract five existing functional tests depend on"),
            "failing_tests": [
                "tests/functional/test_skill_revocation.py::"
                "test_delayed_harm_revokes_retrieved_skill",
                "tests/functional/test_skill_evolution_e0.py::"
                "test_e0_add_compile_retrieval_delayed_and_revocation",
                "tests/functional/test_e1_v2_protocol_repair.py::"
                "test_e1_v2_arm_isolation_window_non_overlap_and_local_skill"
                "_reuse",
                "tests/functional/test_g1_proposal_guidance.py::"
                "test_next_task_reuses_instead_of_colliding",
                "tests/functional/test_f1_forecast_pilot.py::"
                "test_f1_pilot_runs_on_frozen_h2_without_promoting_harness",
            ],
            "two_kinds": (
                "three are literal expectations of the old id string. The "
                "other two are not: evaluation/functional/task_episode_harness"
                "/e1.py detects an already-present arm-local Skill by the "
                "prefix _LOCAL_SKILL_PREFIX = 'fast_winner_e1v2_', and the "
                "task scope now sits between 'fast_winner_' and the operator "
                "segment, so the prefix stops matching, the reuse path is not "
                "taken, and the re-ADD collides with the ABSENT precondition "
                "(AddTargetExistsError)."),
            "not_fixed_and_why": (
                "e1.py is a FROZEN_SURFACE_V9 member and this round "
                "authorized method.py only. Reported for the main line to "
                "route; not self-adjudicated, not worked around, and the "
                "book's ID format was implemented as written rather than "
                "bent to keep the prefix alive."),
        },
        "v9_touched_this_round": (payload.get("wiring_diff") or {}).get(
            "v9_touched_this_round", []),
    }


def annotate() -> int:
    """Correct the delivered artifact from evidence already in it.  0 LLM."""
    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    payload["wiring_diff"] = _wiring_diff()
    findings = t5_findings(payload)
    payload["t5_findings"] = findings
    cost = dict(payload.get("cost") or {})
    cost["llm_calls_raw_field"] = cost.get("llm_calls")
    cost["llm_calls"] = int((payload.get("budgets") or {}).get("llm")
                            or LLM_BUDGET)
    cost["llm_calls_derivation"] = findings["llm_budget_arithmetic"][
        "raw_field_correction"]
    payload["cost"] = cost
    payload["annotation_note"] = (
        "t5_findings and the corrected wiring diff are pure functions of this "
        "artifact and of git; adding them made no backend call.")
    OUT_JSON.write_text(_json_text(payload), encoding="utf-8")
    OUT_MD.write_text(_render_md(payload), encoding="utf-8")
    print("annotated", OUT_JSON, flush=True)
    return 0


# =========================================================================== #
# report
# =========================================================================== #
def _ambiguities() -> list[str]:
    return [
        "Part B is a wiring acceptance on scripted readings, not new "
        "evidence: its numbers are fixtures chosen to put each lifecycle "
        "cell under test, and they travel the real entry point unchanged.",
        "The T1 injected copy carries only the twelve training stations, so "
        "the forecasting arm splits them 8 train / 4 eval rather than reach "
        "outside the injected copy for T1's four original eval series. The AD "
        "arm keeps all twelve because its scored surface is the Query region, "
        "which the training block never touches. The two arms therefore see "
        "the same series but not the same roster roles; the book fixes the "
        "Harness, Memory, Agent, DSL and menu as shared, not the roster.",
        "The AD adapter's action region is the concatenation of the three "
        "verified windows ([120,840)), not the full T1 block [120,900). The "
        "44 of 48 injected training events that fall inside it carry the "
        "label signal; the four outside it are not seen by the fit.",
        "The AD reading is reported to the executor as -macro_f1 so the "
        "executor's own gain arithmetic (baseline - candidate) yields "
        "candidate_F1 - baseline_F1. No executor code was changed to do it, "
        "but the negation is a convention this adapter chose and it is "
        "recorded here rather than buried.",
        "Both rounds of a task run at the same origin on purpose (the book "
        "requires byte-identical Context between r1 and r2), so the AD "
        "adapter's memo makes the second round's identical readings free. "
        "The fit and scoring are closed-form and deterministic, so a re-run "
        "returns identical numbers by construction; only cache misses are "
        "counted against the AD evaluation budget.",
        "A4 tightens handle_feedback_delayed for every task, not just AD: "
        "NEUTRAL delayed no longer extends privilege. That is the authorized "
        "behaviour change of this round, and it is what B5 re-checks on the "
        "forecasting side rather than a regression to be worked around.",
        "handle_feedback_support (the Slow path) was NOT given the same "
        "relation gate. Part C runs with allow_slow=False so it is never "
        "reached here; extending the gate there is a second wiring surface "
        "and was left for the main line to route.",
    ]


def _render_md(doc: Mapping[str, Any]) -> str:
    lines: list[str] = [
        "# T5 -- one Harness entry, two Consumers (%s)" % PROTOCOL_VERSION,
        "",
        "Evidence grade: **%s**." % doc["evidence_grade"],
        "",
        "## Verdict",
        "",
        "**%s**" % doc["verdict"]["verdict"],
        "",
        doc["verdict"]["reason"],
        "",
        "## Part 0",
        "",
        "- HEAD `%s` -- %s" % (doc["part_0"]["head_commit"],
                               doc["part_0"]["head_subject"]),
        "- tracked-modified at start of Part A: %s"
        % (doc["part_0"]["tracked_modified"] or "clean"),
        "",
        "## Part A",
        "",
    ]
    for row in doc["part_a"]["checks"]:
        lines.append("- [%s] `%s` -- %s"
                     % ("x" if row["ok"] else " ", row["id"], row["detail"]))
    km = doc["part_a"]["key_migration"]
    lines += [
        "",
        "### Key migration",
        "",
        "| reading | value |",
        "| --- | --- |",
        "| legacy fixture (forecast/ridge/sMASE) | `%s` |"
        % km["legacy_fixture_key"],
        "| actual T5 forecasting key | `%s` |" % km["actual_forecasting_key"],
        "| anomaly detection key | `%s` |" % km["anomaly_detection_key"],
        "",
        km["note"],
        "",
        "## Part B (0 LLM)",
        "",
    ]
    for row in (doc.get("part_b") or {}).get("checks", []):
        lines.append("- [%s] `%s` -- %s"
                     % ("x" if row["ok"] else " ", row["id"], row["detail"]))
    if doc.get("part_c"):
        lines += ["", "## Part C -- live trajectory", "",
                  "| round | arm | chosen | winner | delayed | relation | skill |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in doc["part_c"]["rounds"]:
            if "error" in row:
                lines.append("| %s | %s | ERROR | %s | | | |"
                             % (row["round"], row["arm"], row["error"]))
                continue
            rel = ",".join(ep["relation"] for ep in row.get("episode_rows", []))
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                row["round"], row["arm"], row.get("chosen"),
                (row.get("winner_program") or [{}])[0].get("op")
                if row.get("winner_program") else "none",
                row.get("delayed_utility"), rel or "-",
                row.get("approved_skill_id") or "-"))
        lines += ["", "LLM calls: %d / %d" % (doc["part_c"]["llm_calls"],
                                              LLM_BUDGET), ""]
    findings = doc.get("t5_findings")
    if findings:
        seg = findings["completed_segment"]
        bud = findings["llm_budget_arithmetic"]
        col = findings["collateral_on_the_other_line"]
        lines += ["", "## What the completed segment shows", ""]
        for item in seg["what_the_completed_segment_shows"]:
            lines.append("- %s" % item)
        lines += ["", "> %s" % seg["what_it_does_not_show"], "",
                  "## LLM budget arithmetic", "", bud["reading"], "",
                  "## Collateral on the other line", "",
                  "%s: %s" % (col["what"], col["two_kinds"]), "",
                  "- %s" % col["not_fixed_and_why"], ""]
        for name in col["failing_tests"]:
            lines.append("  - `%s`" % name)
        lines.append("")
    lines += ["", "## Ambiguities (reported, not self-adjudicated)", ""]
    for item in doc["ambiguities"]:
        lines.append("- %s" % item)
    lines.append("")
    return "\n".join(lines)


# =========================================================================== #
# main
# =========================================================================== #
def main() -> int:
    smoke_only = "--smoke-only" in sys.argv[1:]
    before = _freeze()
    part0 = _part0()
    part_a = _part_a_assertions()
    diff = _wiring_diff()
    sub = _load_substrate()

    doc: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "grade_note": (
            "the substrate is the injected T1 copy and the already-exposed "
            "Qcal/Qf surfaces; a confirmed verdict here says the lifecycle "
            "machine closes along the task axis, never that it generalizes "
            "to natural data"
        ),
        "part_0": part0,
        "part_a": part_a,
        "wiring_diff": diff,
        "substrate": {"train": sub["train"],
                      "forecasting_train": sub["f_train"],
                      "forecasting_eval": sub["eval"],
                      "anomaly_detection_train": sub["train"],
                      "t1_eval_uids_not_injected": sub["unloaded_t1_eval_uids"],
                      "source": sub["source"], "block": list(BLOCK),
                      "anchors": list(ANCHORS),
                      "support_origin": SUPPORT_ORIGIN,
                      "delayed_origin": DELAYED_ORIGIN,
                      "qcal_region": list(QCAL_REGION),
                      "qf_region": list(QF_REGION)},
        "budgets": {"llm": LLM_BUDGET,
                    "forecasting_retrains": FORECASTING_RETRAIN_BUDGET,
                    "ad_evaluations": AD_EVALUATION_BUDGET},
        "ambiguities": _ambiguities(),
    }

    if not part_a["all_passed"]:
        doc["verdict"] = _verdict(part_a=part_a,
                                  part_b={"checks": [], "all_passed": False},
                                  part_c=None, comparison=None, leak=None)
        doc["frozen_surface"] = {"before": before, "after": _verify(before)}
        return _write(doc)

    part_b = _part_b(sub)
    doc["part_b"] = part_b
    if not part_b["all_passed"] or smoke_only:
        doc["verdict"] = _verdict(part_a=part_a, part_b=part_b, part_c=None,
                                  comparison=None, leak=None)
        doc["frozen_surface"] = {"before": before, "after": _verify(before)}
        return _write(doc)

    executors = _build_executors(sub)
    part_c = _part_c(sub, executors)
    comparison = _r1_r2_comparison(part_c)
    leak = _cross_task_leak(part_c)
    doc["part_c"] = part_c
    doc["r1_r2_comparison"] = comparison
    doc["cross_task"] = leak
    doc["cost"] = {
        "llm_calls": part_c["llm_calls"],
        "forecasting_retrains": executors["forecasting"]["reader"].retrains,
        "ad_evaluations": executors["anomaly_detection"]["reader"].evaluations_used,
        "ad_reader_calls": executors["anomaly_detection"]["reader"].calls,
        "forecasting_reader_calls": executors["forecasting"]["reader"].calls,
    }
    doc["verdict"] = _verdict(part_a=part_a, part_b=part_b, part_c=part_c,
                              comparison=comparison, leak=leak)
    doc["frozen_surface"] = {"before": before, "after": _verify(before)}
    return _write(doc)


def _write(doc: Mapping[str, Any]) -> int:
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(_json_text(doc), encoding="utf-8")
    OUT_MD.write_text(_render_md(doc), encoding="utf-8")
    print(json.dumps({"verdict": doc["verdict"]["verdict"],
                      "reason": doc["verdict"]["reason"][:200]},
                     ensure_ascii=False, indent=1))
    print("wrote", OUT_JSON, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        annotate() if "--annotate" in sys.argv[1:] else main()
    )
