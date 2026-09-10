"""DEV-TRAIN-2 experiment-local orchestration.  Not a second numerical runner.

Schedules existing ``dev_train1_runner`` Fast/Slow/freeze/predict/score calls
for the uncapped 10/10 workflow/Skill effect package.  Canonical h0 is not
edited; run permission comes from ``configure_snapshot_for_spec``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import json
import statistics
import threading
import time

from evaluation.main_protocol_p4 import dev_train1_evaluator as numeric
from evaluation.main_protocol_p4 import dev_train1_runner as R
from evaluation.main_protocol_p4.dev_train1b_observations import install_observations

PACKAGE = "DEV-TRAIN-2"
OBSERVATION_VERSION = "window_role_summary_v1"
FIXED_WORKFLOW_LABELS = (
    "identity",
    "outlier_iqr",
    "hampel_filter",
    "outlier_iqr>winsorize",
    "winsorize>hampel_filter",
    "fft_decompose",
    "winsorize>fft_decompose",
)
PAIRED_ROLES = (
    ("train_and_serve", "P", "P"),
    ("train_only", "P", "identity"),
    ("serve_only", "identity", "P"),
)
GLOBAL_FAST_SESSIONS = 4
THIS_PACKAGE_PRIOR = {
    "llm": 0,
    "fits": 0,
    "source": "new package; previous package costs remain in original receipts",
}


class IntegrationPending(RuntimeError):
    """Run permission APIs are missing; the cap must not be faked."""


class OrchestrationState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.live_fast = 0
        self.peak_fast = 0
        self.live_fits = 0
        self.peak_fits = 0
        self.last_fast_end = 0.0
        self.formation_complete_at = 0.0
        self.cpu_complete_at = 0.0
        self.slow_started_at = 0.0
        self.later_scores_at = 0.0
        self.account_fault: BaseException | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_fast_sessions": self.peak_fast,
            "peak_fits": self.peak_fits,
            "global_fast_sessions": GLOBAL_FAST_SESSIONS,
            "fit_writers": 1,
            "formation_complete_at": self.formation_complete_at,
            "cpu_complete_at": self.cpu_complete_at,
            "slow_started_at": self.slow_started_at,
            "later_scores_at": self.later_scores_at,
            "last_fast_end": self.last_fast_end,
            "slow_waited_for_formation": (
                self.slow_started_at == 0.0
                or self.slow_started_at >= self.formation_complete_at
            ),
            "later_scores_after_fast": (
                self.later_scores_at == 0.0
                or self.later_scores_at >= self.last_fast_end
            ),
        }


def cap_wiring_available() -> bool:
    fields = getattr(R.RunSpec, "__dataclass_fields__", {})
    return (
        "maximum_modified_fraction" in fields
        and callable(getattr(R, "configure_snapshot_for_spec", None))
        and callable(getattr(R, "effective_modified_cap", None))
    )


def require_cap_wiring() -> None:
    if not cap_wiring_available():
        raise IntegrationPending(
            "RunSpec.maximum_modified_fraction / configure_snapshot_for_spec "
            "are not available; refusing to fake or globally monkeypatch the cap"
        )


def public_cap_facts(spec: R.RunSpec) -> dict[str, Any]:
    if hasattr(R, "effective_modified_cap"):
        cap = float(R.effective_modified_cap(spec))
    else:
        cap = float(getattr(spec, "maximum_modified_fraction", 0.35))
    return {
        "maximum_modified_fraction": cap,
        "generic_modified_fraction_veto_in_force": cap < 1.0 - 1e-12,
        "modified_fraction_is_measured": True,
        "modified_fraction_is_not_a_harm_proxy": True,
        "all_arms_same_permission": True,
        "no_operator_whitelist_exception": True,
        "no_hardcoded_workflow_recommendation": True,
        "fixed_seven_is_not_the_strongest_legal_combination": True,
        "ordinary_scope_and_modification_quantities_still_measured": True,
    }


def make_spec(
    train_uids: Sequence[str],
    eval_uids: Sequence[str],
    **kwargs: Any,
) -> R.RunSpec:
    """Build a spec.  Live callers must pass maximum_modified_fraction=1.0."""
    fields = dict(
        train_uids=list(train_uids),
        eval_uids=list(eval_uids),
        concurrency=int(kwargs.pop("concurrency", GLOBAL_FAST_SESSIONS)),
        max_llm=int(kwargs.pop("max_llm", R.MAX_LLM)),
        max_fits=int(kwargs.pop("max_fits", R.MAX_FITS)),
        max_wall_seconds=int(kwargs.pop("max_wall_seconds", R.MAX_WALL_SECONDS)),
        requested_model=kwargs.pop("requested_model", R.REQUESTED_MODEL),
        base_url=kwargs.pop("base_url", R.LOOPBACK_URL),
    )
    fields.update(kwargs)
    if "maximum_modified_fraction" in fields:
        require_cap_wiring()
    return R.RunSpec(**fields)


def menu_arm(label: str) -> str:
    return "fixed_menu/" + str(label).replace(">", "__")


def workflow_steps(label: str, period: int) -> Any:
    if label == "identity":
        return None
    steps: list[tuple[str, dict]] = []
    for op in str(label).split(">"):
        _name, value = R.onestep_candidate(op, period)
        if value == "UNAVAILABLE":
            return "UNAVAILABLE"
        if value is None:
            continue
        steps.extend(value)
    return tuple(steps) if steps else None


def fixed_workflow_menu(spec: R.RunSpec) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for label in FIXED_WORKFLOW_LABELS:
        value = workflow_steps(label, spec.period)
        if value == "UNAVAILABLE":
            raise RuntimeError("predeclared fixed workflow is unavailable: " + label)
        out.append((label, value))
    return out


def build_run_contract(spec: R.RunSpec) -> dict[str, Any]:
    cap = (
        float(R.effective_modified_cap(spec))
        if hasattr(R, "effective_modified_cap")
        else float(getattr(spec, "maximum_modified_fraction", 0.35))
    )
    return {
        "package": PACKAGE,
        "data_role": "EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING",
        "train_uids": list(spec.train_uids),
        "eval_uids": list(spec.eval_uids),
        "roster_rule": (
            "first ten in each existing span [40, 80] role roster; not outcome-selected"
        ),
        "formation_units": [list(x) for x in spec.formation_units],
        "eval_units": [list(x) for x in spec.eval_units],
        "anchors": list(spec.anchors),
        "period": spec.period,
        "training_prefix_end": spec.training_prefix_end,
        "training_design": spec.training_design,
        "recombination": spec.recombination,
        "consumer": dict(R.CONSUMER_STRUCTURE),
        "model_requested": spec.requested_model,
        "model_returned_required": R.REQUIRED_RETURNED_MODEL,
        "base_url": spec.base_url,
        "fast_concurrency": GLOBAL_FAST_SESSIONS,
        "fit_writers": 1,
        "global_active_fast_sessions": GLOBAL_FAST_SESSIONS,
        "max_llm": spec.max_llm,
        "max_fits": spec.max_fits,
        "max_wall_seconds": spec.max_wall_seconds,
        "observation_version": OBSERVATION_VERSION,
        "same_interface_for_all_agent_arms": True,
        "maximum_modified_fraction": cap,
        "fixed_workflows": [label for label, _steps in fixed_workflow_menu(spec)],
        "paired_roles": [name for name, _w, _v in PAIRED_ROLES],
        "not_from_flash_20x20": True,
        "single_step_labels_use_single_step_models": True,
        "direct_comparison_with_old_20x20_not_allowed": True,
        "no_hardcoded_workflow_recommendation": True,
    }


def write_or_check_contract(run_dir: Path, contract: Mapping[str, Any]) -> None:
    path = run_dir / "run_contract.json"
    if path.is_file():
        if R.read_json(path) != contract:
            raise RuntimeError("existing run contract differs; refusing to reuse checkpoints")
        return
    R.atomic_write_json(path, contract)


@contextmanager
def install_orchestration(spec: R.RunSpec, run_dir: Path):
    """One process-local 4-Fast gate and a single fit writer for this package."""
    state = OrchestrationState()
    fast_gate = threading.Semaphore(GLOBAL_FAST_SESSIONS)
    fit_lock = threading.RLock()
    original_fast = R.run_fast_decision
    original_fit = R.fit_arm
    original_freeze = R.freeze_model
    original_extra = R.public_extra
    original_slow = R.run_slow_boundary
    facts = public_cap_facts(spec)

    def gated_fast(**kwargs: Any) -> Any:
        if state.account_fault is not None:
            raise state.account_fault
        fast_gate.acquire()
        with state.lock:
            state.live_fast += 1
            if state.live_fast > state.peak_fast:
                state.peak_fast = state.live_fast
        try:
            if state.account_fault is not None:
                raise state.account_fault
            return original_fast(**kwargs)
        except R.AccountFault as exc:
            state.account_fault = exc
            raise
        finally:
            with state.lock:
                state.live_fast -= 1
                state.last_fast_end = time.time()
            fast_gate.release()

    def gated_fit(**kwargs: Any) -> Any:
        with fit_lock:
            with state.lock:
                state.live_fits += 1
                if state.live_fits > state.peak_fits:
                    state.peak_fits = state.live_fits
            try:
                return original_fit(**kwargs)
            finally:
                with state.lock:
                    state.live_fits -= 1

    def gated_freeze(**kwargs: Any) -> Any:
        # Keep each model and its receipt commit in the same single queue.
        with fit_lock:
            return original_freeze(**kwargs)

    def extra(role: str, *, observed_length: int, spec: R.RunSpec | None = None) -> dict[str, Any]:
        payload = dict(original_extra(role, observed_length=observed_length, spec=spec))
        payload["cap_facts"] = facts
        payload["no_hardcoded_workflow_recommendation"] = True
        return payload

    def slow_boundary(**kwargs: Any) -> Any:
        with state.lock:
            if state.formation_complete_at == 0.0:
                raise RuntimeError("Slow started before formation material was complete")
            state.slow_started_at = time.time()
        return original_slow(**kwargs)

    R.run_fast_decision = gated_fast
    R.fit_arm = gated_fit
    R.freeze_model = gated_freeze
    R.public_extra = extra
    R.run_slow_boundary = slow_boundary
    try:
        yield state
    finally:
        R.run_fast_decision = original_fast
        R.fit_arm = original_fit
        R.freeze_model = original_freeze
        R.public_extra = original_extra
        R.run_slow_boundary = original_slow
        R.atomic_write_json(Path(run_dir) / "orchestration.json", state.to_dict())


def identity_assignment(uids: Sequence[str]) -> dict[str, None]:
    return {str(uid): None for uid in uids}


def uniform_assignment(uids: Sequence[str], steps: Any) -> dict[str, Any]:
    return {str(uid): steps for uid in uids}


def train_modification_report(
    spec: R.RunSpec, bundle: R.DataBundle, assignment: Mapping[str, Any],
) -> dict[str, Any]:
    design = numeric.build_training_design(
        bundle.roster(), bundle.values, assignment, R.numeric_config(spec),
        origin=int(spec.training_prefix_end),
        training_design=spec.training_design,
        recombination=spec.recombination,
    )
    window_points = int(R.CONTEXT_LENGTH + R.HORIZON)
    windows = []
    for row in design.get("windows") or ():
        moved = int(row.get("moved_points") or 0)
        windows.append({
            "series_uid": row.get("series_uid"),
            "anchor": row.get("anchor"),
            "moved_points": moved,
            "window_points": window_points,
            "modified_fraction": (moved / window_points) if window_points else None,
        })
    return {
        "behavior_point_count": int(design.get("behavior_point_count") or 0),
        "n_windows": len(windows),
        "windows": windows,
        "this_is_action_quantity_not_harm": True,
    }


def _versus_static(
    spec: R.RunSpec,
    pop: Mapping[str, Any],
    static_by_origin: Mapping[int, Mapping[str, Any]],
    origin: int,
) -> dict[str, Any]:
    static_per = dict((static_by_origin.get(int(origin)) or {}).get("per_uid_smase") or {})
    utilities: dict[str, Any] = {}
    readable: list[float] = []
    for uid in spec.eval_uids:
        static_v = static_per.get(uid)
        arm_v = (pop.get("per_uid_smase") or {}).get(uid)
        if isinstance(static_v, (int, float)) and isinstance(arm_v, (int, float)):
            value = float(static_v) - float(arm_v)
            utilities[uid] = value
            readable.append(value)
        else:
            utilities[uid] = "UNKNOWN"
    risk = R.harmed_stats(readable)
    mean_u: Any = "UNKNOWN"
    if len(readable) == len(list(spec.eval_uids)):
        mean_u = float(statistics.fmean(readable))
    return {
        "per_uid_utility_vs_static": utilities,
        "mean_utility_vs_static": mean_u,
        "risk": risk,
        "identity_reference": "same UID same formation window Static sMASE",
    }


def _score_pairing(
    *,
    spec: R.RunSpec,
    bundle: R.DataBundle,
    model: numeric.FrozenRidge,
    eval_assignment: Mapping[str, Any],
    origin: int,
    static_by_origin: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    pred = R.predict_arm(
        spec=spec, bundle=bundle, model=model,
        eval_assignment=eval_assignment, origin=int(origin),
    )
    scored = R.score_arm(
        spec=spec, bundle=bundle, predictions=pred, origin=int(origin),
    )
    pop = R.population_from_scores(spec.eval_uids, scored)
    extra = _versus_static(spec, pop, static_by_origin, origin)
    return {
        **pop,
        **extra,
        "behavior_point_count": pred.get("behavior_point_count"),
        "consumer_fits": pred.get("consumer_fits"),
    }


def calibrate_fixed_workflows(
    *,
    spec: R.RunSpec,
    bundle: R.DataBundle,
    budget: R.Budget,
    static_model: numeric.FrozenRidge,
    static_by_origin: Mapping[int, Mapping[str, Any]],
    run_dir: Path,
    menu: Sequence[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    path = run_dir / "fixed_workflow_material.json"
    context = {
        "menu": [label for label, _steps in (menu or fixed_workflow_menu(spec))],
        "cap": public_cap_facts(spec),
        "train_uids": list(spec.train_uids),
        "eval_uids": list(spec.eval_uids),
        "formation_units": [list(x) for x in spec.formation_units],
        "not_from_flash_20x20": True,
    }
    if path.is_file():
        saved = R.read_json(path)
        if saved.get("context") != context:
            raise RuntimeError("fixed workflow material context mismatch; use a distinct run ID")
        return saved

    items = list(menu) if menu is not None else fixed_workflow_menu(spec)
    models: dict[str, numeric.FrozenRidge] = {}
    readings: list[dict[str, Any]] = []
    for label, steps in items:
        arm = menu_arm(label)
        train_asg = uniform_assignment(spec.train_uids, steps)
        eval_p = uniform_assignment(spec.eval_uids, steps)
        eval_id = identity_assignment(spec.eval_uids)
        try:
            if steps is None:
                model = static_model
                fit_how = "reused_static"
            else:
                model = R.freeze_model(
                    spec=spec, bundle=bundle, assignment=train_asg,
                    budget=budget, arm=arm, run_dir=run_dir,
                )
                fit_how = "fitted"
            models[label] = model
            modification = train_modification_report(spec, bundle, train_asg)
        except (R.PackageCeiling, R.AccountFault):
            raise
        except Exception as exc:  # noqa: BLE001
            readings.append({
                "label": label,
                "typed_steps": R.steps_to_json(steps) if steps is not None else None,
                "model_arm": arm,
                "status": "FIT_FAILED",
                "why": R._sanitize("%s: %s" % (type(exc).__name__, exc)),
                "recommendation": None,
                "not_borrowed_from_another_workflow_model": True,
            })
            continue

        roles: dict[str, Any] = {}
        for role_name, w_tag, v_tag in PAIRED_ROLES:
            if w_tag == "identity":
                used_model = static_model
                used_model_arm = "static"
            else:
                used_model = model
                used_model_arm = arm if steps is not None else "static"
            eval_asg = eval_id if v_tag == "identity" else eval_p
            origin_rows = []
            complete = True
            smases: list[float] = []
            for _position, origin in spec.formation_units:
                try:
                    pop = _score_pairing(
                        spec=spec, bundle=bundle, model=used_model,
                        eval_assignment=eval_asg, origin=int(origin),
                        static_by_origin=static_by_origin,
                    )
                except (R.PackageCeiling, R.AccountFault):
                    raise
                except Exception as exc:  # noqa: BLE001
                    origin_rows.append({
                        "origin": int(origin),
                        "status": "SCORE_FAILED",
                        "why": R._sanitize("%s: %s" % (type(exc).__name__, exc)),
                    })
                    complete = False
                    continue
                origin_rows.append({"origin": int(origin), "population": pop})
                if pop.get("mean_smase") == "UNKNOWN":
                    complete = False
                else:
                    smases.append(float(pop["mean_smase"]))
            mean_smase: Any = (
                float(statistics.fmean(smases)) if complete and smases else "UNKNOWN"
            )
            roles[role_name] = {
                "W": w_tag,
                "V": v_tag,
                "model_arm": used_model_arm,
                "status": "READ" if complete else "INCOMPLETE",
                "mean_formation_smase": mean_smase,
                "origins": origin_rows,
                "u12_u13_not_used": True,
            }

        readings.append({
            "label": label,
            "typed_steps": R.steps_to_json(steps) if steps is not None else None,
            "model_arm": arm if steps is not None else "static",
            "fit": fit_how,
            "status": roles["train_and_serve"]["status"],
            "train_modification": modification,
            "roles": roles,
            "recommendation": None,
            "not_borrowed_from_another_workflow_model": True,
            "single_step_labels_use_single_step_models": True,
        })

    chosen = None
    for row in readings:
        mean = (row.get("roles") or {}).get("train_and_serve", {}).get("mean_formation_smase")
        if row.get("status") == "READ" and isinstance(mean, float):
            if chosen is None or mean < float(
                chosen["roles"]["train_and_serve"]["mean_formation_smase"]
            ):
                chosen = row
    if chosen is None:
        chosen = {
            "label": "identity",
            "typed_steps": None,
            "mean_formation_smase": "UNKNOWN",
            "why": "no complete seven-workflow candidate; identity frozen",
            "recommendation": None,
        }
    payload = {
        "context": context,
        "status": "COMPUTED",
        "source": "this 10/10 run; not Flash 20/20",
        "not_from_flash_slow_material_workflow_differences": True,
        "u12_u13_not_used": True,
        "recommendation": None,
        "no_hardcoded_winner": True,
        "cap_facts": public_cap_facts(spec),
        "menu": [label for label, _steps in items],
        "readings": readings,
        "chosen_for_fixed_arm_only": {
            "label": chosen.get("label"),
            "typed_steps": chosen.get("typed_steps"),
            "mean_formation_smase": (
                (chosen.get("roles") or {}).get("train_and_serve", {}).get("mean_formation_smase")
                if "roles" in chosen else chosen.get("mean_formation_smase")
            ),
            "risk_by_origin": {
                str(row["origin"]): (row.get("population") or {}).get("risk")
                for row in (chosen.get("roles") or {}).get("train_and_serve", {}).get("origins", [])
            },
            "this_is_the_calibrated_fixed_baseline_not_a_slow_instruction": True,
        },
        "identity_reference": "same UID same formation window Static sMASE",
        "models_are_not_jsonable_here": True,
    }
    R.atomic_write_json(path, payload)
    payload["_models"] = models
    payload["_chosen_row"] = chosen
    return payload


def _chosen_steps(material: Mapping[str, Any]) -> Any:
    chosen = material.get("_chosen_row") or material.get("chosen_for_fixed_arm_only") or {}
    steps = chosen.get("typed_steps")
    if steps is None and chosen.get("label") == "identity":
        return None
    if isinstance(steps, list):
        return R.steps_from_json(steps)
    return steps


def run_skill_arm(
    *,
    spec: R.RunSpec,
    bundle: R.DataBundle,
    snapshot: Any,
    machinery: Mapping[str, Any],
    backend_factory: Callable[[], Any],
    arm: str,
    task_ctx: Any,
    budget: R.Budget,
    run_dir: Path,
) -> dict[str, Any]:
    rows, assignment, missing = R.generate_train_w(
        spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
        backend_factory=backend_factory, arm=arm, task_ctx=task_ctx,
        budget=budget, run_dir=run_dir,
    )
    model, why = R._safe_freeze(
        spec=spec, bundle=bundle, assignment=assignment, missing=missing,
        budget=budget, arm=arm, run_dir=run_dir,
    )
    eval_asg: dict[int, dict[str, Any] | None] = {}
    eval_missing: dict[int, list[str]] = {}
    eval_rows: dict[int, list[dict[str, Any]]] = {}
    for position, origin in spec.eval_units:
        erows, easg, emissing = R.generate_eval_v(
            spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
            backend_factory=backend_factory, arm=arm, position=position,
            origin=origin, task_ctx=task_ctx, budget=budget, run_dir=run_dir,
        )
        eval_rows[int(origin)] = erows
        eval_asg[int(origin)] = easg
        eval_missing[int(origin)] = list(emissing)
    if model is not None:
        R.score_units(
            spec=spec, bundle=bundle, model=model,
            assignment_by_origin=eval_asg, missing_by_origin=eval_missing,
            units=spec.eval_units, arm=arm, run_dir=run_dir,
            write_scores=False,
        )
        payload = model.to_dict()
    else:
        payload = None
    return {
        "train_missing": missing,
        "model": payload,
        "n_train_decisions": len(rows),
        "eval_frozen_unscored": True,
        "why_unscored": why,
        "_model": model,
        "_eval_rows": eval_rows,
    }


def eval_predictions_frozen(spec: R.RunSpec, run_dir: Path, arm: str) -> bool:
    for position, origin in spec.eval_units:
        path = run_dir / arm / ("predictions_u%03d.json" % position)
        if not path.is_file():
            return False
        data = R.read_json(path)
        if data.get("predictions_frozen") is not True:
            return False
        if int(data.get("origin") or -1) != int(origin):
            return False
        if data.get("truth_not_included") is not True:
            return False
        del origin
    return True


def incomplete_readouts(result: Mapping[str, Any], spec: R.RunSpec) -> list[list[Any]]:
    incomplete: list[list[Any]] = []
    for name, arm in (result.get("arms") or {}).items():
        if name == "new_skill" and arm.get("treatment") != "UPDATE_TREATMENT":
            continue
        if name == "old_skill_formation":
            continue
        for _position, origin in spec.eval_units:
            score = (arm.get("scores") or {}).get(str(origin), {})
            if score.get("n_readable") != len(spec.eval_uids) or score.get("mean_smase") in (None, "UNKNOWN"):
                incomplete.append([name, origin])
    return incomplete


def _arm_payload(info: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in info.items() if not str(key).startswith("_")}


def run(
    *,
    spec: R.RunSpec,
    bundle: R.DataBundle,
    run_id: str,
    backend_factory: Callable[[], Any],
    machinery: Mapping[str, Any] | None = None,
    snapshot: Any | None = None,
    skip_slow: bool = False,
    require_uncapped: bool = True,
    run_dir: Path | str | None = None,
    workflow_menu: Sequence[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    """One bounded TRAIN-2 course.  Does not launch transport by itself."""
    if require_uncapped:
        require_cap_wiring()
        if float(R.effective_modified_cap(spec)) != 1.0:
            raise IntegrationPending(
                "DEV-TRAIN-2 requires maximum_modified_fraction=1.0 explicitly; "
                "got %s" % spec.maximum_modified_fraction
            )

    run_dir = Path(run_dir) if run_dir is not None else (R.RUNS_ROOT / str(run_id))
    if hasattr(R, "validate_run_cap"):
        R.validate_run_cap(spec, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = build_run_contract(spec)
    write_or_check_contract(run_dir, contract)

    machinery = machinery or R.load_machinery()
    loaded = snapshot or R.load_h0_snapshot(machinery)
    canonical_sha = getattr(loaded, "runtime_bundle_sha", None)
    if cap_wiring_available():
        configured = R.configure_snapshot_for_spec(spec, loaded, machinery)
    elif require_uncapped:
        raise IntegrationPending("configure_snapshot_for_spec is not available")
    else:
        configured = loaded
    snapshot = configured
    if hasattr(R, "effective_modified_cap"):
        R.atomic_write_json(run_dir / "run_cap.json", {
            "maximum_modified_fraction": R.effective_modified_cap(spec),
        })

    task_ctx = R.task_context_for(spec)
    budget = R.Budget.load(
        run_dir / "budget.json",
        max_llm=spec.max_llm, max_fits=spec.max_fits,
        max_wall_seconds=spec.max_wall_seconds,
    )
    budget.path = run_dir / "budget.json"
    if spec.live_api and budget.llm() == 0 and budget.fits() == 0:
        budget.prior_llm = int(THIS_PACKAGE_PRIOR["llm"])
        budget.prior_fits = int(THIS_PACKAGE_PRIOR["fits"])
    budget.concurrency_used = GLOBAL_FAST_SESSIONS
    budget.concurrency_note = (
        "one global %d-active-Fast semaphore across arms; fits serialized"
        % GLOBAL_FAST_SESSIONS
    )
    budget.persist()

    preflight = R.preflight_record(spec, snapshot)
    preflight["package"] = PACKAGE
    preflight["canonical_h0_runtime_bundle_sha"] = canonical_sha
    preflight["run_snapshot_runtime_bundle_sha"] = getattr(snapshot, "runtime_bundle_sha", None)
    preflight["canonical_h0_bytes_not_edited"] = True
    preflight["configured_snapshot_is_not_labelled_canonical_h0"] = (
        getattr(snapshot, "runtime_bundle_sha", None) != canonical_sha
    )
    preflight["observation_version"] = OBSERVATION_VERSION
    preflight["fixed_workflows"] = [label for label, _steps in (workflow_menu or fixed_workflow_menu(spec))]
    preflight["roster"] = {
        "n_train": len(spec.train_uids),
        "n_eval": len(spec.eval_uids),
        "train_uids": list(spec.train_uids),
        "eval_uids": list(spec.eval_uids),
        "rule": contract["roster_rule"],
        "span": [40, 80],
    }
    preflight_path = run_dir / "preflight.json"
    if preflight_path.is_file():
        previous = R.read_json(preflight_path)
        for key in ("canonical_h0_runtime_bundle_sha", "run_snapshot_runtime_bundle_sha"):
            if previous.get(key) != preflight.get(key):
                raise RuntimeError("snapshot changed since launch; refusing checkpoint reuse")
    else:
        R.atomic_write_json(preflight_path, preflight)

    store = machinery["SnapshotStore"](run_dir / "store")
    controller = machinery["EditController"](
        store, surfaces=machinery["SurfaceRegistry"](),
        router=machinery["FaultRouter"](),
    )
    store.materialize(snapshot)

    result: dict[str, Any] = {
        "package": PACKAGE,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": "RUNNING",
        "exit_code": 0,
        "h0_runtime_bundle_sha": canonical_sha,
        "run_snapshot_runtime_bundle_sha": getattr(snapshot, "runtime_bundle_sha", None),
        "cap_facts": public_cap_facts(spec),
        "arms": {},
    }

    try:
        with install_orchestration(spec, run_dir) as orch:
            with install_observations(run_dir):
                _run_course(
                    spec=spec, bundle=bundle, run_id=run_id, run_dir=run_dir,
                    backend_factory=backend_factory, machinery=machinery,
                    snapshot=snapshot, skip_slow=skip_slow,
                    workflow_menu=workflow_menu, task_ctx=task_ctx,
                    budget=budget, store=store, controller=controller,
                    result=result, orch=orch, canonical_sha=canonical_sha,
                )
    except R.AccountFault as exc:
        result["status"] = "ACCOUNT_OR_PERMISSION_FAULT"
        result["detail"] = str(exc)
        result["exit_code"] = 2
        result["budget"] = budget.to_state()
        budget.persist()
    except R.PackageCeiling as exc:
        result["status"] = "PACKAGE_CEILING"
        result["detail"] = str(exc)
        result["exit_code"] = 3
        result["budget"] = budget.to_state()
    except IntegrationPending as exc:
        result["status"] = "INTEGRATION_PENDING"
        result["detail"] = str(exc)
        result["exit_code"] = 2
        result["budget"] = budget.to_state()
    except Exception as exc:  # noqa: BLE001
        result["status"] = "RUNNER_FAULT"
        result["detail"] = "%s: %s" % (type(exc).__name__, R._sanitize(exc))
        result["exit_code"] = 1
        result["budget"] = budget.to_state()

    serialisable = json.loads(json.dumps(result, default=R._json_default))
    R.atomic_write_json(run_dir / "result.json", serialisable)
    return result


def _run_course(
    *,
    spec: R.RunSpec,
    bundle: R.DataBundle,
    run_id: str,
    run_dir: Path,
    backend_factory: Callable[[], Any],
    machinery: Mapping[str, Any],
    snapshot: Any,
    skip_slow: bool,
    workflow_menu: Sequence[tuple[str, Any]] | None,
    task_ctx: Any,
    budget: R.Budget,
    store: Any,
    controller: Any,
    result: dict[str, Any],
    orch: OrchestrationState,
    canonical_sha: Any,
) -> None:
    del run_id, canonical_sha
    static_train = identity_assignment(spec.train_uids)
    static_eval = identity_assignment(spec.eval_uids)
    static_model = R.freeze_model(
        spec=spec, bundle=bundle, assignment=static_train,
        budget=budget, arm="static", run_dir=run_dir,
    )
    static_formation = R.score_units(
        spec=spec, bundle=bundle, model=static_model,
        assignment_by_origin={origin: static_eval for _p, origin in spec.formation_units},
        missing_by_origin={},
        units=spec.formation_units, arm="static", run_dir=run_dir,
        write_scores=True,
    )
    R.score_units(
        spec=spec, bundle=bundle, model=static_model,
        assignment_by_origin={origin: static_eval for _p, origin in spec.eval_units},
        missing_by_origin={},
        units=spec.eval_units, arm="static", run_dir=run_dir,
        write_scores=False,
    )
    result["arms"]["static"] = {
        "train": "identity",
        "model": static_model.to_dict(),
        "formation_scores": {str(key): value for key, value in static_formation.items()},
        "eval_frozen_unscored": True,
    }

    cpu_box: dict[str, Any] = {"error": None, "material": None, "fixed_model": None, "fixed_why": None}

    def cpu_work() -> None:
        try:
            material = calibrate_fixed_workflows(
                spec=spec, bundle=bundle, budget=budget, static_model=static_model,
                static_by_origin=static_formation, run_dir=run_dir, menu=workflow_menu,
            )
            cpu_box["material"] = material
            chosen_steps = _chosen_steps(material)
            fixed_train = uniform_assignment(spec.train_uids, chosen_steps)
            fixed_eval = uniform_assignment(spec.eval_uids, chosen_steps)
            label = material["chosen_for_fixed_arm_only"]["label"]
            source_arm = "static" if label == "identity" else menu_arm(label)
            # Reuse the exact calibrated model. No additional winner fit.
            fixed_model = R.freeze_model(
                spec=spec, bundle=bundle, assignment=fixed_train,
                budget=budget, arm=source_arm, run_dir=run_dir,
            )
            fixed_why = None
            cpu_box["model_source_arm"] = source_arm
            cpu_box["fixed_model"] = fixed_model
            cpu_box["fixed_why"] = fixed_why
            if fixed_model is not None:
                R.score_units(
                    spec=spec, bundle=bundle, model=fixed_model,
                    assignment_by_origin={origin: fixed_eval for _p, origin in spec.eval_units},
                    missing_by_origin={},
                    units=spec.eval_units, arm="fixed_workflow", run_dir=run_dir,
                    write_scores=False,
                )
        except Exception as exc:  # noqa: BLE001
            cpu_box["error"] = exc

    cpu_thread = threading.Thread(target=cpu_work, name="train2-fixed-cpu")
    cpu_thread.start()
    formation_error: BaseException | None = None
    try:
        form_train_rows, form_train_asg, form_train_missing = R.generate_train_w(
            spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
            backend_factory=backend_factory, arm="old_skill_formation",
            task_ctx=task_ctx, budget=budget, run_dir=run_dir,
        )
        form_model, form_why = R._safe_freeze(
            spec=spec, bundle=bundle, assignment=form_train_asg,
            missing=form_train_missing, budget=budget,
            arm="old_skill_formation", run_dir=run_dir,
        )
        formation_eval_rows: dict[int, list[dict[str, Any]]] = {}
        formation_eval_asg: dict[int, dict[str, Any] | None] = {}
        formation_eval_missing: dict[int, list[str]] = {}
        for position, origin in spec.formation_units:
            rows, asg, missing = R.generate_eval_v(
                spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
                backend_factory=backend_factory, arm="old_skill_formation",
                position=position, origin=origin, task_ctx=task_ctx,
                budget=budget, run_dir=run_dir,
            )
            formation_eval_rows[int(origin)] = rows
            formation_eval_asg[int(origin)] = asg
            formation_eval_missing[int(origin)] = missing
        if form_model is None:
            formation_scores = {
                int(origin): R.unknown_population(spec.eval_uids, form_why or "formation fit failed")
                for _position, origin in spec.formation_units
            }
        else:
            formation_scores = R.score_units(
                spec=spec, bundle=bundle, model=form_model,
                assignment_by_origin=formation_eval_asg,
                missing_by_origin=formation_eval_missing,
                units=spec.formation_units, arm="old_skill_formation",
                run_dir=run_dir, static_by_origin=static_formation,
                write_scores=True,
            )
        result["arms"]["old_skill_formation"] = {
            "train_missing": form_train_missing,
            "model": None if form_model is None else form_model.to_dict(),
            "formation_scores": {str(k): v for k, v in formation_scores.items()},
        }
    except BaseException as exc:
        formation_error = exc
    cpu_thread.join()
    orch.cpu_complete_at = time.time()
    if cpu_box["error"] is not None and isinstance(cpu_box["error"], (R.AccountFault, R.PackageCeiling)):
        raise cpu_box["error"]
    if formation_error is not None:
        raise formation_error
    if cpu_box["error"] is not None:
        raise cpu_box["error"]

    material = cpu_box["material"]
    public_material = {
        key: value for key, value in material.items() if not str(key).startswith("_")
    }
    result["fixed_workflow_material"] = public_material
    fixed_model = cpu_box["fixed_model"]
    result["arms"]["fixed_workflow"] = {
        "label": "fixed_workflow",
        "model_source_arm": cpu_box["model_source_arm"],
        "calibration": {
            "chosen": material.get("chosen_for_fixed_arm_only"),
            "menu": material.get("menu"),
        },
        "model": None if fixed_model is None else fixed_model.to_dict(),
        "eval_frozen_unscored": True,
        "why_unscored": cpu_box["fixed_why"],
    }
    orch.formation_complete_at = time.time()

    slow_material = {
        **{k: v for k, v in public_material.items() if k != "chosen_for_fixed_arm_only"},
        "agent_formation_is_on_the_card_members": True,
        "fast_does_not_receive_this_table": True,
    }
    if skip_slow:
        boundary = {"outcome": "SKIPPED", "treatment": "NO_UPDATE_TREATMENT"}
    else:
        extra_slow = R.public_extra("skill_revision", observed_length=0)
        extra_slow["consumer_structure"] = dict(R.CONSUMER_STRUCTURE)
        extra_slow["cap_facts"] = public_cap_facts(spec)
        slow_factory = R.make_slow_factory(
            spec=spec, machinery=machinery, backend_factory=backend_factory,
            extra=extra_slow, budget=budget,
        )
        if (run_dir / "boundary.json").is_file():
            # The saved proposal already consumed this exact material. Do not
            # regenerate or overwrite it when resuming a completed boundary.
            card = R.read_json(run_dir / "slow_input.json")
        else:
            card = R.build_slow_card(
                spec=spec, train_records=form_train_rows,
                eval_records_by_origin=formation_eval_rows,
                formation_scores=formation_scores, xy_decomposition=slow_material,
            )
        boundary = R.run_slow_boundary(
            spec=spec, machinery=machinery, parent=snapshot, task_ctx=task_ctx,
            card=card, budget=budget, slow_factory=slow_factory,
            store=store, controller=controller,
            checkpoint_path=run_dir / "boundary.json",
        )
    result["boundary"] = {k: v for k, v in boundary.items() if k != "candidate_snapshot"}

    run_new = (
        boundary.get("treatment") == "UPDATE_TREATMENT"
        and boundary.get("candidate_snapshot") is not None
    )
    exposure: dict[str, Any] | None = None
    if run_new:
        exposure = R.dual_role_exposure(
            spec=spec, bundle=bundle, parent=snapshot,
            candidate=boundary["candidate_snapshot"], run_dir=run_dir,
        )
        result["exposure_check"] = exposure
        if exposure.get("outcome") != "EXPOSED":
            run_new = False

    arm_errors: list[BaseException] = []
    arm_out: dict[str, Any] = {}

    def one_arm(name: str, snap: Any) -> None:
        try:
            arm_out[name] = run_skill_arm(
                spec=spec, bundle=bundle, snapshot=snap, machinery=machinery,
                backend_factory=backend_factory, arm=name, task_ctx=task_ctx,
                budget=budget, run_dir=run_dir,
            )
        except BaseException as exc:
            arm_errors.append(exc)

    workers = 2 if run_new else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_arm, "old_skill", snapshot)]
        if run_new:
            futures.append(pool.submit(one_arm, "new_skill", boundary["candidate_snapshot"]))
        for fut in as_completed(futures):
            fut.result()
    if arm_errors:
        raise arm_errors[0]

    result["arms"]["old_skill"] = _arm_payload(arm_out["old_skill"])
    if run_new:
        payload = _arm_payload(arm_out["new_skill"])
        payload["treatment"] = "UPDATE_TREATMENT"
        result["arms"]["new_skill"] = payload
    elif (exposure or {}).get("outcome") == "COMPILED_BUT_NEVER_EXPOSED":
        result["arms"]["new_skill"] = {
            "treatment": "COMPILED_BUT_NEVER_EXPOSED__UTILITY_UNTESTED",
            "why": (
                "the compiled candidate changes no rendered Fast view on "
                "any final training or prediction decision, so the paid "
                "new-vs-old course would compare identical knowledge; the "
                "compiled candidate receipt is kept and the arm is not run"
            ),
            "exposure_check": "exposure_check.json",
            "scores": None,
        }
    else:
        result["arms"]["new_skill"] = {
            "treatment": "NO_UPDATE_TREATMENT",
            "why": (
                "Slow produced no compiled update; new_skill is not a "
                "second copy of old_skill and is not scored as a treatment"
            ),
            "scores": None,
        }

    score_arms = ["static"]
    if fixed_model is not None:
        score_arms.append("fixed_workflow")
    if arm_out.get("old_skill", {}).get("_model") is not None:
        score_arms.append("old_skill")
    if run_new and arm_out.get("new_skill", {}).get("_model") is not None:
        score_arms.append("new_skill")
    for arm_name in score_arms:
        if not eval_predictions_frozen(spec, run_dir, arm_name):
            raise RuntimeError("refusing to open later outcomes before %s predictions freeze" % arm_name)
    orch.later_scores_at = time.time()

    static_eval_scores = R.score_frozen_eval_units(
        spec=spec, bundle=bundle, arm="static", run_dir=run_dir,
    )
    result["arms"]["static"]["scores"] = {str(k): v for k, v in static_eval_scores.items()}
    result["arms"]["static"]["eval_frozen_unscored"] = False
    if fixed_model is not None:
        fixed_scores = R.score_frozen_eval_units(
            spec=spec, bundle=bundle, arm="fixed_workflow", run_dir=run_dir,
            static_by_origin=static_eval_scores,
        )
        result["arms"]["fixed_workflow"]["scores"] = {str(k): v for k, v in fixed_scores.items()}
        result["arms"]["fixed_workflow"]["eval_frozen_unscored"] = False
    if arm_out.get("old_skill", {}).get("_model") is not None:
        old_scores = R.score_frozen_eval_units(
            spec=spec, bundle=bundle, arm="old_skill", run_dir=run_dir,
            static_by_origin=static_eval_scores,
        )
        result["arms"]["old_skill"]["scores"] = {str(k): v for k, v in old_scores.items()}
        result["arms"]["old_skill"]["eval_frozen_unscored"] = False
    if run_new and arm_out.get("new_skill", {}).get("_model") is not None:
        new_scores = R.score_frozen_eval_units(
            spec=spec, bundle=bundle, arm="new_skill", run_dir=run_dir,
            static_by_origin=static_eval_scores,
        )
        result["arms"]["new_skill"]["scores"] = {str(k): v for k, v in new_scores.items()}
        result["arms"]["new_skill"]["eval_frozen_unscored"] = False

    result["orchestration"] = orch.to_dict()
    result["budget"] = budget.to_state()
    gaps = incomplete_readouts(result, spec)
    result["incomplete_readouts"] = gaps
    result["readout_status"] = "INCOMPLETE" if gaps else "COMPLETE"
    if gaps:
        result["exit_code"] = 2
        result["status"] = "INCOMPLETE"
    else:
        result["status"] = "COMPLETED"
        result["exit_code"] = 0
