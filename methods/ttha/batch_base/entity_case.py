"""Entity-split case profile of the batch construction base (docs/DEV_DOMAIN_AUG_ENTITY_SPLIT_TASK_2026-09-19.md §2-§4, §6.A).

A *case* is one fixed population of COHORT_SIZE original CSV columns of one dense TSLib file at one cut t, with the geometry of
spec (L=192, H=48, T=672, C_A / C_B / E origins) and the shared MLP / train_arm / seed discipline of train.py:

  data       only the authorized columns and only the rows of the opened stage are converted to numbers (a CaseSlice never holds
             another column of the file); stage row permissions are those of context.STAGE_ROWS
  scaler     per-entity mean/std (ddof=0) of raw T; bound on disk to (dataset, domain, case_id, t, roster)
  parents    433 stride-1 windows per entity inside T; no missing values in these files, so every parent is legal
  window     each parent pair is ONE joint 240-point series [X (192); y (48)] in frozen T-scaler units; the seven TempoPFN-sourced
             primitives, the P_NoMixRecipe preset and its local edit (tempo_aug.py) act on that series and produce the child view
  RNG        one Streams per (domain, case_index, program index, entity column), SeedSequence-derived (no hash); the preset and
             every P_NoMixRecipe_Edit share the preset's stream (program index PRESET_INDEX) so that disabling never redraws
  training   parent view 0.5 + child view 0.5 in ONE shared MLP; the batch stream is drawn over the ACTUAL pool (entities x 433)
  scoring    data.score_predictions (normalized MSE, entity macro, origins equal-weighted) on C_A / C_B / E

No LLM, no selection rule, no hashing. Torch is imported only inside the fit / label functions (subprocess workers).
"""
from __future__ import annotations

import csv
import datetime as _dt
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import augment, commit as _commit, context, data, observe, policy, spec, train
from . import tempo_aug as ta

PROFILE_VERSION = "entity_split_v1"
EXPOSURE = "SERIES_DISJOINT_DEVELOPMENT"
COHORT_SIZE = 16
SEED_ROOT = 2026091900
DOMAIN_INDEX = {"D01": 1, "D02": 2, "D03": 3, "D04": 4, "D05": 5}   # D03-D05 added for DEV-AUG-TASK-FAMILY-SCREEN (additive; D01/D02 seeds unchanged)
L, H, TRAIN_SPAN = spec.L, spec.H, spec.TRAIN_SPAN
L_JOINT = L + H
STAGE_ROWS = context.STAGE_ROWS
CONSUMER = {**spec.CONSUMER,
            "loss": "pooled MSE on normalized targets: 0.5 x parent view (all parent windows of the case) + 0.5 x child view (the plan's augmented "
                    "copy of every parent; identity entities put their parent in the child slot); None trains the parent view only",
            "training": "shared MLP, AdamW, 2000 updates, parent batch 64 drawn over the actual pool (entities x 433), the same T scaler and batch "
                        "index stream for every plan of a case",
            "serving": "prediction inputs are the raw observed 192 points before each origin (never augmented); scoring targets are the raw observed future"}


# --------------------------------------------------------------------------- case specification
@dataclass(frozen=True)
class CaseSpec:
    dataset: str                  # spec.DATASETS key (electricity / traffic) or a synthetic name with csv_path set
    domain: str                   # neutral domain id (D01 / D02)
    case_id: str                  # e.g. D01_S03
    role: str                     # source | select | test
    t: int
    roster: tuple                 # COHORT_SIZE distinct original column names (strings)
    case_index: int
    csv_path: str | None = None   # None = spec.DATASETS[dataset]['path']; a synthetic file for smoke only
    total_hours: int | None = None
    train_range: tuple = field(init=False)
    c_a: tuple = field(init=False)
    c_b: tuple = field(init=False)
    e: tuple = field(init=False)
    material_rows: tuple = field(init=False)
    evaluate_rows: tuple = field(init=False)
    c_b_rows: tuple = field(init=False)
    e_input_rows: tuple = field(init=False)
    e_target_rows: tuple = field(init=False)

    def __post_init__(self):
        t = int(self.t)
        r = tuple(str(x) for x in self.roster)
        if len(r) != COHORT_SIZE or len(set(r)) != COHORT_SIZE:
            raise ValueError("a case roster holds %d distinct column names" % COHORT_SIZE)
        object.__setattr__(self, "roster", r)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "train_range", (t - TRAIN_SPAN, t))
        object.__setattr__(self, "c_a", (t, t + 48))
        object.__setattr__(self, "c_b", (t + 96, t + 144))
        object.__setattr__(self, "e", (t + 192, t + 240, t + 288, t + 336))
        object.__setattr__(self, "material_rows", (t - TRAIN_SPAN, t))
        object.__setattr__(self, "evaluate_rows", (t - TRAIN_SPAN, max(self.c_a) + H))
        object.__setattr__(self, "c_b_rows", (min(self.c_b) - L, max(self.c_b) + H))
        object.__setattr__(self, "e_input_rows", (min(self.e) - L, max(self.e)))
        object.__setattr__(self, "e_target_rows", (min(self.e), max(self.e) + H))
        assert self.evaluate_rows[1] <= min(self.c_b), "evaluate slice must not reach a C_B origin"
        assert self.c_b_rows[1] <= min(self.e), "C_B slice must not reach an E origin"
        if self.t - TRAIN_SPAN < 0:
            raise ValueError("cut too early")
        if self.role not in ("source", "select", "test"):
            raise ValueError("role must be source / select / test")
        if self.max_row() > self.hours():
            raise ValueError("case runs past the file")

    @property
    def job_id(self) -> str:                    # observe.overview / build_windows compatibility
        return self.case_id

    @property
    def roster_override(self) -> tuple:
        return self.roster

    def hours(self) -> int:
        if self.total_hours is not None:
            return int(self.total_hours)
        return int(spec.DATASETS[self.dataset]["total_hours"])

    def max_row(self) -> int:
        return max(self.e) + H

    def path(self) -> Path:
        return Path(self.csv_path) if self.csv_path else Path(spec.DATASETS[self.dataset]["path"])

    def to_json(self) -> dict:
        return {"dataset": self.dataset, "domain": self.domain, "case_id": self.case_id, "role": self.role, "t": self.t, "roster": list(self.roster),
                "case_index": self.case_index, "csv_path": self.csv_path, "total_hours": self.total_hours, "profile": PROFILE_VERSION,
                "train_range": list(self.train_range), "c_a": list(self.c_a), "c_b": list(self.c_b), "e": list(self.e)}


def case_from_json(d: dict) -> CaseSpec:
    return CaseSpec(dataset=d["dataset"], domain=d["domain"], case_id=d["case_id"], role=d["role"], t=int(d["t"]), roster=tuple(d["roster"]),
                    case_index=int(d["case_index"]), csv_path=d.get("csv_path"), total_hours=d.get("total_hours"))


def cases_from_split(split: dict) -> dict:
    """{case_id: CaseSpec} from the frozen partition JSON (docs/DOMAIN_AUG_ENTITY_SPLIT_V1.json layout)."""
    out = {}
    for dom, D in split["domains"].items():
        for g in D["groups"]:
            cs = CaseSpec(dataset=D["dataset"], domain=dom, case_id=g["case_id"], role=g["stage"], t=int(g["t"]), roster=tuple(g["roster"]),
                          case_index=int(g["case_index"]), csv_path=D.get("csv_path"), total_hours=D.get("total_hours"))
            if list(cs.train_range) != list(g["train_rows"]) or list(cs.c_a) != list(g["c_a_origins"]) or list(cs.c_b) != list(g["c_b_origins"]) or list(cs.e) != list(g["e_origins"]):
                raise RuntimeError("frozen split rows disagree with the geometry for %s" % g["case_id"])
            out[cs.case_id] = cs
    return out


def check_split(split: dict) -> dict:
    """Disjointness / size / role checks of the partition (no data row is read)."""
    out = {"domains": {}, "ok": True}
    for dom, D in split["domains"].items():
        groups = D["groups"]
        allr = [c for g in groups for c in g["roster"]]
        roles = {}
        for g in groups:
            roles.setdefault(g["stage"], []).append(g["case_id"])
        excl = set(D.get("excluded_columns", []))
        rec = {"n_groups": len(groups), "n_columns": len(allr), "n_unique": len(set(allr)), "sizes_ok": all(len(g["roster"]) == COHORT_SIZE for g in groups),
               "excluded_in_use": sorted(set(allr) & excl), "roles": {k: len(v) for k, v in roles.items()},
               "case_index_unique": len({g["case_index"] for g in groups}) == len(groups)}
        rec["ok"] = rec["n_unique"] == rec["n_columns"] and rec["sizes_ok"] and not rec["excluded_in_use"] and rec["case_index_unique"] and rec["roles"] == {"source": 8, "select": 2, "test": 2}
        out["domains"][dom] = rec
        out["ok"] = out["ok"] and rec["ok"]
    return out


# --------------------------------------------------------------------------- row-bounded, column-bounded CSV access
@dataclass
class CaseSlice:
    dataset: str
    row_start: int
    row_end: int
    values: np.ndarray          # (rows, COHORT_SIZE) float64: ONLY the authorized columns
    roster: list
    roster_cols: np.ndarray     # arange(COHORT_SIZE): values already hold the roster columns in roster order
    dates: list
    columns_in_file: int

    def rows(self, a: int, b: int) -> np.ndarray:
        if a < self.row_start or b > self.row_end or a >= b:
            raise PermissionError("row access [%d,%d) outside loaded slice [%d,%d)" % (a, b, self.row_start, self.row_end))
        return self.values[a - self.row_start: b - self.row_start]

    def hour_of_row(self, r: int) -> int:
        if r < self.row_start or r >= self.row_end:
            raise PermissionError("timestamp access row %d outside loaded slice" % r)
        return _dt.datetime.strptime(self.dates[r - self.row_start], "%Y-%m-%d %H:%M:%S").hour

    def timestamp_of_row(self, r: int) -> _dt.datetime:
        if r < self.row_start or r >= self.row_end:
            raise PermissionError("timestamp access row %d outside loaded slice" % r)
        return _dt.datetime.strptime(self.dates[r - self.row_start], "%Y-%m-%d %H:%M:%S")


def load_case_slice(cs: CaseSpec, row_start: int, row_end: int) -> CaseSlice:
    """Reads header + rows [row_start, row_end) of the file; converts ONLY the roster cells to float64."""
    if row_start < 0 or row_end > cs.hours() or row_start >= row_end:
        raise ValueError("bad slice [%d,%d) for %s" % (row_start, row_end, cs.case_id))
    dates, rows = [], []
    with open(cs.path(), newline="") as f:
        r = csv.reader(f)
        header = next(r)
        columns_all = header[1:]
        name_to_idx = {name: i + 1 for i, name in enumerate(columns_all)}
        missing = [n for n in cs.roster if n not in name_to_idx]
        if missing:
            raise RuntimeError("roster columns absent from the file: %s" % missing[:4])
        idx = [name_to_idx[n] for n in cs.roster]
        for i, row in enumerate(itertools.islice(r, row_end)):
            if i < row_start:
                continue
            dates.append(row[0])
            rows.append([float(row[j]) for j in idx])
    if len(rows) != row_end - row_start:
        raise RuntimeError("slice [%d,%d): only %d rows available" % (row_start, row_end, len(rows)))
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (row_end - row_start, COHORT_SIZE):
        raise RuntimeError("slice shape drift")
    prev = None
    for i, s in enumerate(dates):
        t = _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        if prev is not None and (t - prev).total_seconds() != 3600.0:
            raise RuntimeError("non-hourly gap inside slice at absolute row %d" % (row_start + i))
        prev = t
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite raw values in roster columns within slice [%d,%d)" % (row_start, row_end))
    return CaseSlice(dataset=cs.dataset, row_start=row_start, row_end=row_end, values=values, roster=list(cs.roster),
                     roster_cols=np.arange(COHORT_SIZE, dtype=np.int64), dates=dates, columns_in_file=len(columns_all))


# --------------------------------------------------------------------------- case context (one case opened at one stage)
@dataclass
class CaseContext:
    job: CaseSpec
    stage: str
    run_dir: Path
    slice: CaseSlice
    scaler: data.Scaler
    parents: data.Parents | None
    mase: np.ndarray | None
    _overview: dict | None = field(default=None, repr=False)

    @property
    def job_dir(self) -> Path:
        d = self.run_dir / self.job.case_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def overview(self) -> dict:
        if self.parents is None:
            raise PermissionError("overview needs the T stage")
        if self._overview is None:
            p = self.job_dir / "overview.json"
            if p.exists():
                self._overview = context.read_json(p)
            else:
                ov = observe.overview(self.slice, self.job, self.scaler, self.parents)
                ov.pop("actions", None)                                   # the study's adapter supplies the action table
                ov.update({"dataset_exposure": EXPOSURE, "case_id": self.job.case_id, "domain_id": self.job.domain, "role": self.job.role,
                           "profile": PROFILE_VERSION, "consumer": CONSUMER,
                           "geometry": {"L": L, "H": H, "T": TRAIN_SPAN, "n_entities": COHORT_SIZE, "n_parents_per_entity": spec.N_PARENTS,
                                        "pool_size": COHORT_SIZE * spec.N_PARENTS, "c_a_origins_relative_to_t": [0, 48], "sampling": "hourly"}})
                self._overview = ov
                context.write_json(p, self._overview)
        return self._overview

    def inspect_data(self, entity_indices, kind: str = "hour_profile", sub_range=None) -> dict:
        return observe.inspect_data(self.slice, self.job, self.scaler, entity_indices, kind, sub_range)


def _binding(cs: CaseSpec) -> dict:
    return {"dataset": cs.dataset, "domain": cs.domain, "case_id": cs.case_id, "t": int(cs.t), "roster": list(cs.roster), "profile": PROFILE_VERSION,
            "file": cs.path().name}


def _check_binding(saved: dict, cs: CaseSpec) -> None:
    want = _binding(cs)
    for k in ("dataset", "domain", "case_id", "t", "roster", "profile", "file"):
        if saved.get(k) != want[k]:
            raise RuntimeError("case binding drift within run for %s: %s" % (cs.case_id, k))


def open_case(cs: CaseSpec, run_dir, stage: str = "material") -> CaseContext:
    lo, hi = getattr(cs, STAGE_ROWS[stage])
    run_dir = Path(run_dir)
    jd = run_dir / cs.case_id
    sc_path, spec_path = jd / "scaler.npz", jd / "case_spec.json"
    if stage in ("material", "evaluate"):
        sl = load_case_slice(cs, lo, hi)
        scaler = data.compute_scaler(sl, cs.t)
        if scaler.floor_hits:
            raise RuntimeError("ELIGIBILITY_FAIL zero-scale entities: %d" % scaler.floor_hits)
        parents = data.build_parents(sl, cs.t, scaler)
        mase = data.mase_denominators(sl, cs.t)
        jd.mkdir(parents=True, exist_ok=True)
        if sc_path.exists():
            old = np.load(sc_path)
            _check_binding(context.read_json(spec_path), cs)
            if not (np.array_equal(old["mean"], scaler.mean) and np.array_equal(old["scale"], scaler.scale)):
                raise RuntimeError("scaler drift within run for %s" % cs.case_id)
        else:
            np.savez(sc_path, mean=scaler.mean, std=scaler.std, scale=scaler.scale, mase=mase, t=cs.t, hours=parents.hours, anchors=parents.anchors,
                     roster=np.array(cs.roster), dataset=np.array(cs.dataset))
            context.write_json(spec_path, {**_binding(cs), "spec": cs.to_json(), "frozen_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                                           "columns_in_file": sl.columns_in_file, "columns_converted": COHORT_SIZE})
        return CaseContext(job=cs, stage=stage, run_dir=run_dir, slice=sl, scaler=scaler, parents=parents, mase=mase)
    if not sc_path.exists() or not spec_path.exists():
        raise RuntimeError("frozen scaler missing for %s; open the material stage first" % cs.case_id)
    _check_binding(context.read_json(spec_path), cs)
    sc = np.load(sc_path)
    if [str(x) for x in sc["roster"]] != list(cs.roster) or int(sc["t"]) != cs.t or str(sc["dataset"]) != cs.dataset:
        raise RuntimeError("scaler binding drift for %s" % cs.case_id)
    scaler = data.Scaler(mean=sc["mean"], std=sc["std"], scale=sc["scale"], floor_hits=0)
    sl = load_case_slice(cs, lo, hi)
    return CaseContext(job=cs, stage=stage, run_dir=run_dir, slice=sl, scaler=scaler, parents=None, mase=sc["mase"])


# --------------------------------------------------------------------------- joint windows and their real timestamps
def joint_parents(ctx: CaseContext) -> tuple:
    """(Z (E*433, 240) float64 normalized [X; y], ent, k) in frozen entity -> window order."""
    if ctx.parents is None:
        raise PermissionError("joint parents need the T stage")
    P = ctx.parents
    E, n = P.X_norm.shape[0], P.n_parents
    Z = np.concatenate([P.X_norm, P.y_norm], axis=-1).reshape(E * n, L_JOINT)
    return Z, np.repeat(np.arange(E), n), np.tile(np.arange(n), E)


def window_starts(ctx: CaseContext) -> list:
    """pandas Timestamps of each parent's first point, checked against the clock hours of the slice."""
    import pandas as pd
    P = ctx.parents
    t0 = ctx.job.train_range[0]
    out = []
    E, n = P.X_norm.shape[0], P.n_parents
    ts_k = []
    for k in range(n):
        ts = ctx.slice.timestamp_of_row(t0 + k)
        if ts.hour != int(P.hours[k]):
            raise RuntimeError("DATE_BINDING_FAILED window %d" % k)
        ts_k.append(pd.Timestamp(ts))
    for e in range(E):
        out.extend(ts_k)
    return out


# --------------------------------------------------------------------------- program tables (frozen enumeration; same order as the earlier packages)
def explicit_table() -> list:
    """0 = empty; 1..51 = legal explicit compositions by length 1, 2, 3 in PRIMITIVES (= execution) order; 52 = P_NoMixRecipe."""
    out = [[]]
    for n in (1, 2, 3):
        for combo in itertools.combinations(ta.PRIMITIVES, n):
            if any(a in combo and b in combo for a, b in ta.EXCLUSIVE):
                continue
            out.append([{"op": x} for x in combo])
    out.append([{"op": ta.RECIPE}])
    if len(out) != 53:
        raise RuntimeError("explicit program table must hold 53 programs")
    return out


def edit_table() -> list:
    """0 = the preset; 1..7 = one component disabled, in PRIMITIVES order; 8..28 = two disabled, in combinations order."""
    out = [[{"op": ta.RECIPE}]]
    for n in ta.PRIMITIVES:
        out.append([ta.edit_step([n])])
    for a, b in itertools.combinations(ta.PRIMITIVES, 2):
        out.append([ta.edit_step([a, b])])
    if len(out) != 29:
        raise RuntimeError("edit program table must hold 29 programs")
    return out


PROGRAMS = explicit_table()
EDIT_PROGRAMS = edit_table()
PRESET_INDEX = 52
_PKEY = lambda steps: json.dumps(steps, sort_keys=True, separators=(",", ":"))
PROGRAM_INDEX = {_PKEY(p): i for i, p in enumerate(PROGRAMS)}
EDIT_INDEX = {_PKEY(p): i for i, p in enumerate(EDIT_PROGRAMS)}
LEGAL_PROGRAMS = PROGRAMS + EDIT_PROGRAMS[1:]          # 53 + 28 = 81 distinct per-entity programs (empty, compositions, preset, edits)


def validate_steps_full(steps) -> list:
    """The complete per-entity program space of this study: [] | 1-3 primitives | [P_NoMixRecipe] | [P_NoMixRecipe_Edit 0-2 disabled]."""
    if isinstance(steps, list) and len(steps) == 1 and isinstance(steps[0], dict) and steps[0].get("op") == ta.RECIPE_EDIT:
        st = steps[0]
        if set(st) - {"op", "disabled_ops"}:
            raise policy.PolicyError("%s takes only disabled_ops; strengths, weights, probabilities and the execution order are not editable" % ta.RECIPE_EDIT)
        try:
            return [ta.edit_step(st.get("disabled_ops", []))]
        except ValueError as exc:                                     # rd.PolicyError is a ValueError
            raise policy.PolicyError(str(exc)) from None
    if isinstance(steps, list) and any(isinstance(s, dict) and s.get("op") == ta.RECIPE_EDIT for s in steps):
        raise policy.PolicyError("%s is a complete program on its own; it cannot be combined with other steps" % ta.RECIPE_EDIT)
    try:
        return ta.validate_steps(steps)
    except ValueError as exc:
        raise policy.PolicyError(str(exc)) from None


def program_seed_index(steps: list) -> int:
    """Seed stream index of a canonical program: explicit compositions by table index; the preset and every edit share PRESET_INDEX."""
    if steps == [{"op": ta.RECIPE}] or ta.is_edit_program(steps):
        return PRESET_INDEX
    return PROGRAM_INDEX[_PKEY(steps)]


def program_table_index(steps: list) -> tuple:
    """('explicit', i) or ('edit', i) for a canonical program (reporting / random draws)."""
    if ta.is_edit_program(steps):
        return ("edit", EDIT_INDEX[_PKEY(steps)])
    return ("explicit", PROGRAM_INDEX[_PKEY(steps)])


def entity_program_seed(domain: str, case_index: int, prog_index: int, entity_column: str) -> int:
    ident = int(entity_column) if str(entity_column).isdigit() else int.from_bytes(str(entity_column).encode("utf-8")[:4].ljust(4, b"\0"), "big")
    return int(np.random.SeedSequence([SEED_ROOT, DOMAIN_INDEX.get(domain, 9), int(case_index), int(prog_index), ident]).generate_state(1, dtype=np.uint32)[0])


def validate_plan_full(pol) -> dict:
    if not isinstance(pol, dict) or "default" not in pol:
        raise policy.PolicyError("policy must be an object with a 'default'")
    if set(pol) - {"default", "rules", "rationale", "observation_fields_used"}:
        raise policy.PolicyError("unknown policy keys %s" % sorted(set(pol) - {"default", "rules", "rationale", "observation_fields_used"}))
    if not isinstance(pol["default"], dict) or set(pol["default"]) != {"steps"}:
        raise policy.PolicyError("default must be {'steps': [...]}")
    clean = {"default": {"steps": validate_steps_full(pol["default"]["steps"])}, "rules": []}
    rules = pol.get("rules", []) or []
    if not isinstance(rules, list) or len(rules) > ta.MAX_RULES:
        raise policy.PolicyError("rules must be a list of at most %d rules" % ta.MAX_RULES)
    for r in rules:
        if not isinstance(r, dict) or set(r) != {"when", "steps"}:
            raise policy.PolicyError("each rule must be {'when': predicate, 'steps': [...]}")
        policy.validate_predicate(r["when"])
        clean["rules"].append({"when": r["when"], "steps": validate_steps_full(r["steps"])})
    rationale = str(pol.get("rationale", ""))[:500]
    policy.validate_text(rationale)
    clean["rationale"] = rationale
    clean["observation_fields_used"] = [f for f in (pol.get("observation_fields_used") or []) if f in spec.OBS_FIELDS]
    return clean


def compile_plan_full(pol: dict, table: list) -> dict:
    clean = validate_plan_full(pol)
    resolved, assignment, rule_index, n_unknown = {}, [], [], 0
    for row in table:
        chosen, ri = clean["default"]["steps"], -1
        for i, r in enumerate(clean["rules"]):
            v = policy._eval(r["when"], row, table, resolved)        # existing three-valued evaluator; UNKNOWN never fires
            if v is None:
                n_unknown += 1
            if v is True:
                chosen, ri = r["steps"], i
                break
        assignment.append([dict(s) for s in chosen])
        rule_index.append(ri)
    return {"policy": clean, "assignment": assignment, "rule_index": rule_index, "resolved_thresholds": resolved, "n_unknown": n_unknown}


def uniform_policy(steps: list, rationale: str) -> dict:
    return {"default": {"steps": steps}, "rules": [], "rationale": rationale, "observation_fields_used": []}


def material_key(assignment: list) -> str:
    return json.dumps({"profile": PROFILE_VERSION, "assignment": assignment}, sort_keys=True, separators=(",", ":"))


def is_identity(assignment: list) -> bool:
    return all(not steps for steps in assignment)


def program_label(steps) -> str:
    if not steps:
        return "none"
    if steps == [{"op": ta.RECIPE}]:
        return "Preset"
    if ta.is_edit_program(steps):
        return "Edit[-%s]" % "-".join(x.replace("tp_", "") for x in steps[0]["disabled_ops"])
    return "Comp[%s]" % "+".join(s["op"].replace("tp_", "") for s in steps)


def assignment_label(assignment, public_id=None) -> str:
    if public_id:
        return public_id
    if assignment == "FixedMixup":
        return "FixedMixup"
    if not isinstance(assignment, list) or not assignment:
        return "unknown"
    if is_identity(assignment):
        return "None"
    labels = [program_label(s) for s in assignment]
    if len(set(labels)) == 1:
        return labels[0]
    groups = sorted(set(labels))
    return "Conditional{%s}" % ", ".join("%s x%d" % (g, labels.count(g)) for g in groups)


def uniform_steps(assignment):
    if isinstance(assignment, list) and assignment and all(s == assignment[0] for s in assignment):
        return assignment[0]
    return None


# --------------------------------------------------------------------------- materials (physical registry per case directory)
PUBLIC = ("None", "FixedMixup", "P_AmpResample", "P_NoMixRecipe")
PUBLIC_STEPS = {"None": [], "P_AmpResample": [{"op": "tp_amplitude"}, {"op": "tp_resample"}], "P_NoMixRecipe": [{"op": ta.RECIPE}]}
FIXED_MIXUP = {"steps": spec.FIXED_MIXUP_STEPS, "seed_rule": "RandomState(spec.aug_seed(entity_index, 0)) = 900090 + entity_index (batch_base augment.py)",
               "definition": "augment.step_timemixup on the joint normalized [X;y] parent windows of each entity: (1-w) z + w z[donor], w = 0.25, donor = derangement "
                             "over the entity's own 433 parents (rule R); the historical fixed reference, not a TempoPFN primitive"}


def aug_dir(job_dir: Path) -> Path:
    d = Path(job_dir) / "aug_materials"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_registry(job_dir: Path) -> dict:
    p = Path(job_dir) / "aug_materials" / "index.json"
    return context.read_json(p) if p.exists() else {}


def save_registry(job_dir: Path, reg: dict) -> None:
    context.write_json(aug_dir(job_dir) / "index.json", reg)


def none_record(case_id: str) -> dict:
    return {"material_id": "None", "job_id": case_id, "key": material_key([[] for _ in range(COHORT_SIZE)]), "assignment": [[] for _ in range(COHORT_SIZE)],
            "profile": PROFILE_VERSION, "alias_of": None, "path": None, "summary_path": None, "material_index": None, "kind": "none"}


def _entity_summary(d: np.ndarray, ent: np.ndarray, assignment) -> list:
    ents = []
    for e in range(COHORT_SIZE):
        rows = np.flatnonzero(ent == e)
        de = d[rows]
        ents.append({"steps": assignment[e] if isinstance(assignment, list) else assignment, "windows": int(rows.size),
                     "rms_change_X": float(np.sqrt((de[:, :L] ** 2).mean())), "rms_change_y": float(np.sqrt((de[:, L:] ** 2).mean())),
                     "fraction_points_changed_X": float((np.abs(de[:, :L]) > 0).mean()), "fraction_points_changed_y": float((np.abs(de[:, L:]) > 0).mean())})
    return ents


def build_material(ctx: CaseContext, assignment: list, phys_id: str) -> dict:
    """Freeze one physical material (child view of every parent) under the per-(domain, case, program, entity) RNG. Torch process only."""
    reg = load_registry(ctx.job_dir)
    if phys_id in reg:
        raise ValueError("physical material id exists")
    if len(assignment) != COHORT_SIZE:
        raise ValueError("assignment must cover every entity")
    assignment = [validate_steps_full(s) for s in assignment]
    key = material_key(assignment)
    if any(r["key"] == key for r in reg.values()):
        raise ValueError("a physical material with this assignment already exists")
    if is_identity(assignment):
        raise ValueError("the all-empty plan is None; no physical material")
    t0 = time.time()
    Z, ent, k = joint_parents(ctx)
    needs_dates = any(st["op"] in ("tp_calendar", ta.RECIPE, ta.RECIPE_EDIT) for steps in assignment for st in steps)
    starts = window_starts(ctx) if needs_dates else None
    C = np.empty(Z.shape, dtype=np.float32)
    per_step = {n: {"windows": 0, "identity": 0} for n in ta.PRIMITIVES}
    shadow = {n: {"drawn": 0, "skipped": 0} for n in ta.PRIMITIVES}
    recipe_sets, shadow_sets, window_identity, no_step, seeds, ent_dis = {}, {}, 0, 0, {}, {}
    cs = ctx.job
    for e in range(COHORT_SIZE):
        rows = np.flatnonzero(ent == e)
        prog = assignment[e]
        if not prog:
            C[rows] = Z[rows].astype(np.float32)
            continue
        edit, preset = ta.is_edit_program(prog), prog == [{"op": ta.RECIPE}]
        pi = program_seed_index(prog)
        disabled = ta.canonical_disabled(prog[0]["disabled_ops"]) if edit else ([] if preset else None)
        seed = entity_program_seed(cs.domain, cs.case_index, pi, cs.roster[e])
        seeds["entity_%d" % e] = {"program_index": pi, "seed": seed, "disabled_ops": disabled}
        ent_dis["entity_%d" % e] = disabled
        streams = ta.Streams(seed)
        for p in rows:                                             # frozen window order inside the entity
            st = starts[p] if starts is not None else None
            if edit:
                out, executed, ident, drawn = ta.run_program_edit(Z[p], prog, streams, st)
            else:
                out, executed, ident = ta.run_program(Z[p], prog, streams, st)
                drawn = executed
            C[p] = out
            if edit or preset:
                lab = "+".join(executed) or "(none)"
                recipe_sets[lab] = recipe_sets.get(lab, 0) + 1
                shadow_sets["+".join(drawn)] = shadow_sets.get("+".join(drawn), 0) + 1
                for n in drawn:
                    shadow[n]["drawn"] += 1
                    shadow[n]["skipped"] += int(n not in executed)
                no_step += int(not executed)
            for n, i in zip(executed, ident):
                per_step[n]["windows"] += 1
                per_step[n]["identity"] += int(i)
            window_identity += int(np.array_equal(out, Z[p].astype(np.float32)))
    secs = time.time() - t0
    d = C.astype(np.float64) - Z
    mdir = aug_dir(ctx.job_dir)
    npz = mdir / (phys_id + ".npz")
    np.savez(npz, Xc=C[:, :L], yc=C[:, L:])
    any_edit = any(ta.is_edit_program(p) for p in assignment)
    summary = {"material_id": phys_id, "job_id": cs.case_id, "profile": PROFILE_VERSION, "domain": cs.domain, "case_index": cs.case_index,
               "entity_program_seeds": seeds,
               "seed_rule": "SeedSequence([%d, domain_index, case_index, program_index, int(entity_column)]).generate_state(1, uint32)[0]; the preset and every "
                            "P_NoMixRecipe_Edit use program_index %d (the preset's stream of that entity)" % (SEED_ROOT, PRESET_INDEX),
               "legal_parents": int(Z.shape[0]), "build_seconds": secs,
               "rms_change_X": float(np.sqrt((d[:, :L] ** 2).mean())), "rms_change_y": float(np.sqrt((d[:, L:] ** 2).mean())),
               "fraction_points_changed_X": float((np.abs(d[:, :L]) > 0).mean()), "fraction_points_changed_y": float((np.abs(d[:, L:]) > 0).mean()),
               "windows_bitwise_unchanged": window_identity, "per_step": per_step, "recipe_step_sets": recipe_sets or None,
               "edit": ({"parent_program_index": PRESET_INDEX, "entity_disabled_ops": ent_dis, "shadow_per_step": shadow, "shadow_step_sets": shadow_sets,
                         "retained_step_sets": recipe_sets, "windows_no_retained_step": no_step,
                         "semantics": "shadow = what the preset drew on that window; skipped = drawn but disabled for that entity; retained steps replayed from the recorded RNG states"}
                        if any_edit else None),
               "units": "normalized by the frozen per-entity T scaler (child - parent)", "entities": _entity_summary(d, ent, assignment)}
    sp = mdir / (phys_id + "__summary.json")
    context.write_json(sp, summary)
    rec = {"material_id": phys_id, "job_id": cs.case_id, "key": key, "assignment": assignment, "profile": PROFILE_VERSION, "alias_of": None,
           "path": str(npz), "summary_path": str(sp), "material_index": int(phys_id[2:]) if phys_id.startswith("TA") else None,
           "kind": "tempo_entity_edit" if any_edit else "tempo_entity", "built_local": time.strftime("%Y-%m-%d %H:%M:%S")}
    reg = load_registry(ctx.job_dir)
    reg[phys_id] = rec
    save_registry(ctx.job_dir, reg)
    return rec


def build_fixed_mixup(ctx: CaseContext) -> dict:
    """The historical fixed reference on the same joint windows (augment.apply_steps with spec.FIXED_MIXUP_STEPS, entity seed rule)."""
    P = ctx.parents
    Z, ent, k = joint_parents(ctx)
    C = np.empty(Z.shape, dtype=np.float64)
    for e in range(COHORT_SIZE):
        rows = np.flatnonzero(ent == e)
        xc, yc, log = augment.apply_steps(P.X_norm[e], P.y_norm[e], P.hours, spec.FIXED_MIXUP_STEPS, e)
        C[rows] = np.concatenate([xc, yc], axis=-1)
    C32 = C.astype(np.float32)
    mdir = aug_dir(ctx.job_dir)
    npz = mdir / "FixedMixup.npz"
    np.savez(npz, Xc=C32[:, :L], yc=C32[:, L:])
    d = C32.astype(np.float64) - Z
    sp = mdir / "FixedMixup__summary.json"
    context.write_json(sp, {"material_id": "FixedMixup", "job_id": ctx.job.case_id, "legal_parents": int(Z.shape[0]), **FIXED_MIXUP,
                            "rms_change_X": float(np.sqrt((d[:, :L] ** 2).mean())), "rms_change_y": float(np.sqrt((d[:, L:] ** 2).mean())),
                            "entities": _entity_summary(d, ent, "FixedMixup")})
    return {"material_id": "FixedMixup", "job_id": ctx.job.case_id, "key": "FixedMixup:R:w0.25:aug_seed", "assignment": None, "profile": "timemixup_batch_base",
            "alias_of": None, "path": str(npz), "summary_path": str(sp), "material_index": None, "kind": "fixed_mixup", "built_local": time.strftime("%Y-%m-%d %H:%M:%S")}


def build_public(ctx: CaseContext) -> list:
    """None record, FixedMixup child and the two public tempo materials of one physical cache (0 fits)."""
    reg = load_registry(ctx.job_dir)
    if "None" not in reg:
        reg["None"] = none_record(ctx.job.case_id)
        save_registry(ctx.job_dir, reg)
    if "FixedMixup" not in load_registry(ctx.job_dir):
        reg = load_registry(ctx.job_dir)
        reg["FixedMixup"] = build_fixed_mixup(ctx)
        save_registry(ctx.job_dir, reg)
    table = ctx.overview()["entities"]
    built = []
    for mid in ("P_AmpResample", "P_NoMixRecipe"):
        if mid in load_registry(ctx.job_dir):
            continue
        compiled = compile_plan_full(uniform_policy(PUBLIC_STEPS[mid], "public reference %s" % mid), table)
        build_material(ctx, compiled["assignment"], mid)
        built.append(mid)
    return built


def inspect(ctx: CaseContext, material_id: str, entity_index=None, window: str = "largest_change") -> dict:
    """What a material changed (executed steps, identity counts, change fraction / RMS for X and y, edit shadow counts, one window on request)."""
    reg = load_registry(ctx.job_dir)
    if material_id not in reg:
        raise KeyError(material_id)
    head = reg[material_id]
    if head.get("kind") == "none" or head.get("alias_of") == "None":
        return {"material_id": material_id, "note": "no augmentation: the parent view only, no child view"}
    if head.get("kind") == "fixed_mixup":
        s = context.read_json(head["summary_path"])
        return {"material_id": material_id, "note": "historical public reference: " + FIXED_MIXUP["definition"], "change_X": {"rms": round(s["rms_change_X"], 4)},
                "change_y": {"rms": round(s["rms_change_y"], 4)}}
    rec = reg[head["alias_of"]] if head.get("alias_of") else head
    s = context.read_json(rec["summary_path"])
    out = {"material_id": material_id, "alias_of": head.get("alias_of"), "legal_windows": s["legal_parents"],
           "steps_executed_windows": {k: v["windows"] for k, v in s["per_step"].items() if v["windows"]},
           "step_identity_windows": {k: v["identity"] for k, v in s["per_step"].items() if v["windows"]},
           "recipe_step_sets": s.get("recipe_step_sets"), "windows_bitwise_unchanged": s["windows_bitwise_unchanged"],
           "change_X": {"rms": round(s["rms_change_X"], 4), "fraction_points_changed": round(s["fraction_points_changed_X"], 4)},
           "change_y": {"rms": round(s["rms_change_y"], 4), "fraction_points_changed": round(s["fraction_points_changed_y"], 4)},
           "entities": {"entity_%d" % e: {"steps": [st["op"] for st in r["steps"]] if isinstance(r["steps"], list) else r["steps"], "rms_X": round(r["rms_change_X"], 4), "rms_y": round(r["rms_change_y"], 4)}
                        for e, r in enumerate(s["entities"])},
           "field_note": "child - parent in frozen T-scaler units; identity = a step returned its input unchanged (e.g. censor mode 0); the child view is trained "
                         "next to the unchanged parent (0.5/0.5); serving inputs and scoring targets never change",
           "interpretation": "Material diagnostics (what changed), not a predictor of downstream utility."}
    ed = s.get("edit")
    if ed:
        legal = s["legal_parents"]
        out["recipe_edit"] = {"entity_disabled_ops": ed["entity_disabled_ops"],
                              "drawn_windows_per_component": {n: v["drawn"] for n, v in ed["shadow_per_step"].items() if v["drawn"]},
                              "skipped_windows_per_component": {n: v["skipped"] for n, v in ed["shadow_per_step"].items() if v["skipped"]},
                              "skip_fraction_of_legal_windows": {n: round(v["skipped"] / legal, 4) for n, v in ed["shadow_per_step"].items() if v["skipped"]},
                              "retained_step_sets": ed["retained_step_sets"], "windows_no_retained_step": ed["windows_no_retained_step"],
                              "note": "drawn = the preset drew that component on the window; skipped = drawn but disabled for that entity; other windows equal the preset"}
    if entity_index is not None:
        e = int(entity_index)
        Z, ent, k = joint_parents(ctx)
        rows = np.flatnonzero(ent == e)
        with np.load(rec["path"]) as z:
            Cc = np.concatenate([z["Xc"][rows], z["yc"][rows]], axis=1).astype(np.float64)
        diff = np.abs(Cc - Z[rows]).max(axis=1)
        j = int(np.argmax(diff)) if window == "largest_change" else (0 if window == "first" else rows.size - 1)
        out["window"] = {"entity": "entity_%d" % e, "choice": window, "absolute_start_row": int(ctx.job.train_range[0] + k[rows[j]]),
                         "parent": [round(float(v), 3) for v in Z[rows[j]]], "child": [round(float(v), 3) for v in Cc[j]],
                         "input_target_split": L, "units": "normalized by the frozen T scaler"}
    return out


# --------------------------------------------------------------------------- fits (torch; subprocess workers)
def cell_id(case_id: str, phys: str, seed: int, tag: str = "") -> str:
    return "%s__%s__s%d%s" % (case_id, phys, seed, ("__" + tag) if tag else "")


def score_with_status(pred, truth, scaler, mase) -> dict:
    s = data.score_predictions(pred, truth, scaler.mean, scaler.scale, mase)
    s["status"] = "SCORABLE" if s["n_nonfinite_predictions"] == 0 and np.isfinite(s["normalized_mse_macro"]) else "NOT_SCORABLE"
    return s


def fit_one(cs: CaseSpec, run_dir, phys: str, seed: int, *, threads: int, timeout_s: float, tag: str = "", n_updates: int | None = None) -> dict:
    """One shared-Consumer fit of a physical material (rows [t-672, t+96) only); C_A scored; model, predictions and cell record written."""
    import torch
    torch.set_num_threads(int(threads))
    t0 = time.time()
    ctx = open_case(cs, run_dir, stage="evaluate")
    P = ctx.parents
    Xp, Yp = P.X_norm.reshape(-1, L), P.y_norm.reshape(-1, H)
    reg = load_registry(ctx.job_dir)
    rec_m = reg[phys]
    if rec_m.get("alias_of"):
        raise RuntimeError("fit a physical material, not an alias")
    child = None
    if rec_m["path"] is not None:
        with np.load(rec_m["path"]) as z:
            child = (z["Xc"].copy(), z["yc"].copy())
        if child[0].shape != Xp.shape or child[1].shape != Yp.shape:
            raise RuntimeError("child does not cover the parent set")
    n_pool = int(Xp.shape[0])
    if n_pool != COHORT_SIZE * spec.N_PARENTS:
        raise RuntimeError("pool size %d is not entities x parents" % n_pool)
    for d in ("runs", "cells", "pred_c_a"):
        (ctx.job_dir / d).mkdir(exist_ok=True)
    steps = spec.N_UPDATES if n_updates is None else int(n_updates)
    bidx = train.batch_indices(seed, n_pool=n_pool, n_updates=steps)
    model, tres = train.train_arm(Xp, Yp, child, bidx, model_seed=seed, timeout_seconds=timeout_s, n_updates=steps)
    wins = data.build_windows(ctx.slice, cs.c_a, ctx.scaler, with_truth=True)
    pred = train.predict_windows(model, wins, ctx.scaler)
    truth = np.stack([w.y_true_raw for w in wins], axis=1)
    c = cell_id(cs.case_id, phys, seed, tag)
    mp = ctx.job_dir / "runs" / (c + ".pt")
    train.save_model(model, mp)
    np.savez(ctx.job_dir / "pred_c_a" / (c + ".npz"), pred_raw=pred, origins=np.array(cs.c_a))
    rec = {"cell_id": c, "status": "OK", "profile": PROFILE_VERSION, "job_id": cs.case_id, "domain": cs.domain, "dataset": cs.dataset, "material_id": phys,
           "material_key": rec_m["key"], "model_seed": seed, "batch_seed": spec.batch_seed(seed), "n_updates": steps, "tag": tag, "physical_fit": True,
           "child_view": child is not None, "pool_size": n_pool, "entity_count": COHORT_SIZE, "roster": list(cs.roster), "roster_explicit": True,
           "batch_stream": "train.batch_indices(seed, n_pool=%d, n_updates=%d)" % (n_pool, steps), "torch_threads": int(threads), "model_path": str(mp),
           "train": {"seconds": tres.seconds, "n_updates_done": tres.n_updates_done, "final_train_loss": tres.final_train_loss, "first_step_loss": tres.first_step_loss,
                     "loss_curve": tres.loss_curve},
           "scores": {"c_a": score_with_status(pred, truth, ctx.scaler, ctx.mase)}, "consumer": CONSUMER, "device": str(train.device()), "torch": torch.__version__,
           "rows_read": list(cs.evaluate_rows), "worker_seconds_total": time.time() - t0}
    context.write_json(ctx.job_dir / "cells" / (c + ".json"), rec)
    return rec


def branch_cells(job_dir: Path) -> dict:
    out = {}
    cd = Path(job_dir) / "cells"
    for p in sorted(cd.glob("*.json")) if cd.exists() else []:
        r = context.read_json(p)
        if r.get("status") == "OK" and not r.get("tag"):
            out[r["cell_id"]] = r
    return out


# --------------------------------------------------------------------------- labels (C_B after commit; E frozen then scored behind the cohort barrier)
def label_prerequisite(stage: str, branch: Path, cs: CaseSpec, stage_root: Path) -> None:
    jd = Path(branch) / cs.case_id
    if stage == "c_b" and not (jd / "commit.json").exists():
        raise PermissionError("C_B refused: commit.json missing")
    if stage == "freeze_e" and not (jd / "c_b_scores.json").exists():
        raise PermissionError("E inputs refused: C_B not scored")
    if stage == "score_e":
        _commit.require_cohort_frozen(branch, cs.case_id, stage_root)       # the existing whole-stage E barrier, unchanged
        if not (jd / "e_frozen.json").exists():
            raise PermissionError("E targets refused: predictions not frozen")


def label_stage(stage: str, branch, cs: CaseSpec, stage_root) -> dict:
    branch, stage_root = Path(branch), Path(stage_root)
    label_prerequisite(stage, branch, cs, stage_root)
    jd = branch / cs.case_id
    cells = branch_cells(jd)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if stage == "c_b":
        ctx = open_case(cs, branch, stage="c_b")
        wins = data.build_windows(ctx.slice, cs.c_b, ctx.scaler, with_truth=True)
        truth = np.stack([w.y_true_raw for w in wins], axis=1)
        out = {"job_id": cs.case_id, "profile": PROFILE_VERSION, "opened_local": now, "rows_read": list(cs.c_b_rows), "cells": {}}
        for c, r in cells.items():
            s = score_with_status(train.predict_windows(train.load_model(r["model_path"]), wins, ctx.scaler), truth, ctx.scaler, ctx.mase)
            out["cells"][c] = {"material_id": r["material_id"], "model_seed": r["model_seed"], "c_b": s}
            r["scores"]["c_b"] = s
            context.write_json(jd / "cells" / (c + ".json"), r)
        context.write_json(jd / "c_b_scores.json", out)
        return out
    if stage == "freeze_e":
        ctx = open_case(cs, branch, stage="e_input")
        wins = data.build_windows(ctx.slice, cs.e, ctx.scaler, with_truth=False)
        (jd / "predictions_e").mkdir(exist_ok=True)
        out = {"job_id": cs.case_id, "profile": PROFILE_VERSION, "started_local": now, "rows_read": list(cs.e_input_rows), "models": {}}
        for c, r in cells.items():
            pred = train.predict_windows(train.load_model(r["model_path"]), wins, ctx.scaler)
            p = jd / "predictions_e" / (c + ".npz")
            np.savez(p, pred_raw=pred, origins=np.array(cs.e))
            out["models"][c] = {"path": str(p), "material_id": r["material_id"], "model_seed": r["model_seed"]}
        out["frozen_at_local"] = out["frozen_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
        context.write_json(jd / "e_frozen.json", out)
        return out
    if stage == "score_e":
        ctx = open_case(cs, branch, stage="e_target")
        fz = context.read_json(jd / "e_frozen.json")
        truth = data.read_targets(ctx.slice, cs.e)
        out = {"job_id": cs.case_id, "profile": PROFILE_VERSION, "scored_local": now, "e_frozen_local": fz["frozen_local"], "rows_read": list(cs.e_target_rows), "cells": {}}
        for c, m in fz["models"].items():
            with np.load(m["path"]) as z:
                if not np.array_equal(z["origins"], np.array(cs.e)):
                    raise RuntimeError("frozen E origins differ")
                pred = z["pred_raw"].copy()
            out["cells"][c] = {"material_id": m["material_id"], "model_seed": m["model_seed"], "e": score_with_status(pred, truth, ctx.scaler, ctx.mase)}
        context.write_json(jd / "e_scores.json", out)
        return out
    raise ValueError(stage)
