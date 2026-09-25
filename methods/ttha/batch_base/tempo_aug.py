"""TempoPFN-sourced augmentation primitives adapted to natural hourly training windows (DEV-TEMPO-AUG-SOURCE-ALIGNMENT §4-§5).

Name of the whole thing: "TempoPFN 来源算子的自然时序适配" -- NOT a reproduction of TempoPFN's full augmentation pipeline.

  primitives   7 source-parameter primitives; the numeric code is the verbatim copy in tempo_source.py (Apache-2.0)
  window       each legal parent pair of the readiness profile is ONE joint 240-point series [X (192, Baseline-Linear filled);
               y (48, raw observed)], normalized by the frozen per-entity T scaler; a primitive sees exactly that series
               (float32 tensor [1, 240, 1], as TempoPFN feeds its series). The child view replaces nothing: parents, targets,
               scaler, legal window set and serving inputs never change.
  composition  <= 3 primitives, no repeats, executed in the fixed order regime/shock -> calendar/amplitude -> resample -> censor
               -> random_conv; regime/shock and calendar/amplitude are mutually exclusive. The frozen public preset P_NoMixRecipe is
               a program of its own (see RECIPE_TEXT). No mixup, no donor, no synthetic source series, no rescaling, no noise.
  randomness   one material seed per (job, material); one Streams object (numpy Generator + torch global + legacy numpy global
               states) is consumed in frozen entity -> window order; global RNG states are swapped in/out around every call, so
               training RNG is untouched. Identical canonical assignments are the same material (alias; built once).
No I/O policy here beyond the job's own aug_materials/ directory; no fitting, no LLM, no hashing.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import math
import time
from pathlib import Path

import numpy as np

from . import context, policy, spec
from . import readiness as rd

L_JOINT = spec.L + spec.H                      # 240
PROFILE_VERSION = "tempo_source_v1"
SEED_BASE = 2026091800
MAX_STEPS = 3
MAX_RULES = rd.MAX_RULES
RECIPE = "P_NoMixRecipe"
PRIMITIVES = ("tp_regime", "tp_shock", "tp_calendar", "tp_amplitude", "tp_resample", "tp_censor", "tp_random_conv")
FIXED_ORDER = {name: i for i, name in enumerate(PRIMITIVES)}           # execution order (regime/shock share rank class 0)
EXCLUSIVE = (("tp_regime", "tp_shock"), ("tp_calendar", "tp_amplitude"))
SOURCE_FILE_OFFLINE = "src/synthetic_generation/augmentations/offline_per_sample_iid_augmentations.py"
SOURCE_FILE_AUG = "src/data/augmentations.py"

# Frozen source facts (read from the code at commit 5969ec6; L = 240 in this package). Shown to Fast and copied into METHOD.md.
CATALOG = {
    "tp_regime": {
        "source": SOURCE_FILE_OFFLINE + ": UnivariateOfflineAugmentor._apply_regime_change (505-556)",
        "semantics": "Piecewise affine edit: 1-3 change points (uniform in [8, 232) at L=240; min segment max(8, L//32)); each segment's "
                     "deviations from its own mean are scaled by U(0.8, 1.25) and the segment is shifted by N(0, 0.15 x std(series)) "
                     "(torch.std, correction=1; std 0/non-finite -> 1). Segments may straddle the input/target boundary.",
        "can_be_identity": "no (continuous draws)"},
    "tp_shock": {
        "source": SOURCE_FILE_OFFLINE + ": UnivariateOfflineAugmentor._apply_shock_recovery (558-589)",
        "semantics": "Adds mag x exp(-clamp(t - t0, min=0) / max(1, half_life)): t0 uniform in [15, 225); |mag| = U(0.5, 2.0) x std(series) "
                     "(torch.std, correction=1), sign +/- 1/2 each; half_life = U(0.03, 0.25) x L = 7.2-60 steps. AS IN THE SOURCE, "
                     "t < t0 is not zeroed: every point before t0 receives the full constant offset mag, and the offset decays after t0 "
                     "(a level shift that recovers), not an impulse on a clean prefix.",
        "can_be_identity": "no"},
    "tp_calendar": {
        "source": SOURCE_FILE_OFFLINE + ": UnivariateOfflineAugmentor._apply_calendar_injections (591-662)",
        "semantics": "Hourly calendar index from the window's REAL first timestamp. Factors start at 1: all Saturday/Sunday hours x one "
                     "U(0.7, 0.95) dip (drawn only if the window contains a weekend); all hours of a month's last day x one U(1.05, 1.3) "
                     "bump (drawn only if present); 1-2 uniformly placed single hours x U(0.8, 1.4). Applied around the window mean: "
                     "(s - mean) x factors + mean.",
        "can_be_identity": "no (1-2 impulses always drawn)"},
    "tp_amplitude": {
        "source": SOURCE_FILE_OFFLINE + ": UnivariateOfflineAugmentor._apply_seasonality_amplitude_modulation (664-684)",
        "semantics": "One contiguous span of width uniform in [15, 120] (max(8, L//16) .. L//2) at a uniform start in [0, L - width): "
                     "deviations from the span's mean are scaled by U(0.5, 1.8).",
        "can_be_identity": "no"},
    "tp_resample": {
        "source": SOURCE_FILE_OFFLINE + ": UnivariateOfflineAugmentor._apply_resample_artifacts (686-732)",
        "semantics": "Keep every factor-th point (factor uniform in 2..7 = max(2, min(8, L//32)); offset uniform in [0, factor)) and "
                     "restore length: linear interpolation (p .5), sample-and-hold (p .2), or linear + 3-point moving average with zero "
                     "padding (np.convolve 'same'; p .3; the first/last point are pulled toward 0 by the zero padding). Outside the "
                     "first/last kept point, linear modes hold the end value.",
        "can_be_identity": "no in practice"},
    "tp_censor": {
        "source": SOURCE_FILE_AUG + ": CensorAugmenter (319-371)",
        "semantics": "Mode uniform in {0, 1, 2}: 0 = unchanged (a legal no-op branch, p 1/3); 1 = clip from above at the sorted value "
                     "at index floor(q_high x (L-1)); 2 = clip from below at index floor(q_low x (L-1)); q_low/q_high = min/max of two "
                     "U(0, 1). Draws from the global torch RNG.",
        "can_be_identity": "yes (mode 0, p 1/3)"},
    "tp_random_conv": {
        "source": SOURCE_FILE_AUG + ": RandomConvAugmenter (1013-1213), called with p_transform=1.0",
        "semantics": "1-3 stacked depthwise 1-D convolutions, padding 'same' with reflect/replicate/circular; per layer odd kernel 3..31, "
                     "dilation 1..8, bias U(-0.5, 0.5), kernel type Gaussian (sigma U(0.5, 5)) / standard normal / quadratic polynomial / "
                     "noisy Sobel (1/4 each), 80% L1-normalized; between layers none/ReLU/tanh (1/3 each); output rescaled to the "
                     "input's min/max (flat output -> input mean). Draws from the global torch and legacy numpy RNGs.",
        "can_be_identity": "no in practice"},
}
RECIPE_WEIGHTS = {"structure": 0.6, "seasonality": 0.5, "artifacts": 0.3, "discrete": 0.6}
RECIPE_CONV_P = 0.3
RECIPE_TEXT = ("P_NoMixRecipe (frozen public preset, per window): draw 2 of the 4 categories structure/seasonality/artifacts/discrete "
               "without replacement with the source category weights 0.6/0.5/0.3/0.6 (normalized); structure = regime or shock "
               "(equiprobable), seasonality = calendar or amplitude (equiprobable), artifacts = resample, discrete = censor; apply "
               "in the fixed order; then with probability 0.3 one random_conv. 2-3 steps per window. An explicitly truncated "
               "adaptation (source: 2-5 of 6 categories incl. invariances/analytic, plus mixup, scaling and noise), not the full "
               "TempoPFN pipeline and not a random search.")


# --------------------------------------------------------------------------- RNG streams (isolated from training RNG)
class Streams:
    """The three random sources the copied code reads: SourceAugmentor.rng (numpy Generator), and the GLOBAL torch and legacy numpy
    states used by CensorAugmenter / RandomConvAugmenter. Seeded like the source augmentor's __init__ (default_rng(seed),
    np.random.seed(seed), torch.manual_seed(seed)) but without touching the caller's global state."""

    def __init__(self, seed: int):
        import torch
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        saved_t, saved_n = torch.get_rng_state(), np.random.get_state()
        try:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            self.torch_state, self.np_state = torch.get_rng_state(), np.random.get_state()
        finally:
            torch.set_rng_state(saved_t)
            np.random.set_state(saved_n)

    @contextlib.contextmanager
    def global_streams(self):
        import torch
        saved_t, saved_n = torch.get_rng_state(), np.random.get_state()
        torch.set_rng_state(self.torch_state)
        np.random.set_state(self.np_state)
        try:
            yield
        finally:
            self.torch_state, self.np_state = torch.get_rng_state(), np.random.get_state()
            torch.set_rng_state(saved_t)
            np.random.set_state(saved_n)

    def snapshot(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "torch": self.torch_state.clone(), "np": self.np_state}

    @classmethod
    def from_snapshot(cls, snap: dict) -> "Streams":
        """A detached Streams positioned at a recorded snapshot (DEV-TEMPO-AUG-RECIPE-EDIT-PILOT §4): consuming it never moves the
        Streams it was taken from. No global RNG is touched here."""
        obj = cls.__new__(cls)
        obj.seed = None
        obj.rng = np.random.default_rng()
        obj.rng.bit_generator.state = snap["rng"]
        obj.torch_state = snap["torch"].clone()
        s = snap["np"]
        obj.np_state = (s[0], np.array(s[1], copy=True), s[2], s[3], s[4])
        return obj


# --------------------------------------------------------------------------- primitive execution
def _src():
    from . import tempo_source
    return tempo_source


def as_series(z) :
    import torch
    return torch.as_tensor(np.asarray(z), dtype=torch.float32).reshape(1, -1, 1).clone()


def apply_primitive(name: str, series, streams: Streams, start=None):
    """series: float32 tensor [1, L, 1]. start: pandas Timestamp of the series' first point (tp_calendar only)."""
    src = _src()
    host = src.SourceAugmentor(streams.rng)
    if name == "tp_regime":
        return host._apply_regime_change(series, p_apply=1.0)
    if name == "tp_shock":
        return host._apply_shock_recovery(series, p_apply=1.0)
    if name == "tp_calendar":
        if start is None:
            raise ValueError("tp_calendar needs the window's real first timestamp")
        return host._apply_calendar_injections(series, [start], ["h"], p_apply=1.0)
    if name == "tp_amplitude":
        return host._apply_seasonality_amplitude_modulation(series, p_apply=1.0)
    if name == "tp_resample":
        return host._apply_resample_artifacts(series, p_apply=1.0)
    if name == "tp_censor":
        with streams.global_streams():
            return src.CensorAugmenter().transform(series)
    if name == "tp_random_conv":
        with streams.global_streams():
            return src.RandomConvAugmenter(p_transform=1.0).transform(series)
    raise ValueError("unknown primitive %r" % name)


def recipe_plan_and_apply(series, streams: Streams, start=None, observer=None) -> tuple:
    """P_NoMixRecipe on one window; draws interleave with the primitives exactly as in the source apply() order.
    Returns (series, executed step names, per-step identity flags). observer (opt-in, RECIPE-EDIT-PILOT): called as
    observer(step_name, streams.snapshot()) immediately before each primitive runs; None = the unchanged public preset."""
    import torch
    rng = streams.rng
    cands = list(RECIPE_WEIGHTS)
    probs = np.array([RECIPE_WEIGHTS[c] for c in cands], dtype=float)
    probs = probs / probs.sum()
    chosen = list(rng.choice(cands, size=2, replace=False, p=probs))
    steps, ident = [], []

    def step(name, s):
        if observer is not None:
            observer(name, streams.snapshot())
        before = s.clone()
        s = apply_primitive(name, s, streams, start)
        steps.append(name)
        ident.append(bool(torch.equal(before, s)))
        return s

    if "structure" in chosen:
        pick = str(rng.choice(["regime", "shock"]))
        series = step("tp_regime" if pick == "regime" else "tp_shock", series)
    if "seasonality" in chosen:
        pick = str(rng.choice(["calendar", "amplitude"]))
        series = step("tp_calendar" if pick == "calendar" else "tp_amplitude", series)
    if "artifacts" in chosen:
        series = step("tp_resample", series)
    if "discrete" in chosen:
        series = step("tp_censor", series)
    if rng.random() < RECIPE_CONV_P:
        series = step("tp_random_conv", series)
    return series, steps, ident


# --------------------------------------------------------------------------- P_NoMixRecipe_Edit (opt-in; DEV-TEMPO-AUG-RECIPE-EDIT-PILOT §3-§4)
RECIPE_EDIT = "P_NoMixRecipe_Edit"
MAX_DISABLED = 2
RECIPE_EDIT_TEXT = ("P_NoMixRecipe_Edit (per window): the preset draws its categories, primitives and random_conv exactly as P_NoMixRecipe "
                    "(same parent random stream, same order, same strengths); then every drawn primitive whose name is in disabled_ops is "
                    "skipped for that window. Nothing is redrawn to replace a skipped step, category weights are not renormalized, and the "
                    "retained primitives use the random draws they would have used; a window may end with one step or unchanged. "
                    "disabled_ops: 0-2 of the seven component names; 0 names = the preset itself. Strengths, weights, counts, the conv "
                    "probability and the execution order are not editable.")


def canonical_disabled(names) -> list:
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
        raise rd.PolicyError("disabled_ops must be a list of component names")
    bad = [n for n in names if n not in PRIMITIVES]
    if bad:
        raise rd.PolicyError("unknown component %r; the components are %s" % (bad[0], list(PRIMITIVES)))
    uniq = sorted(set(names), key=lambda n: FIXED_ORDER[n])
    if len(uniq) > MAX_DISABLED:
        raise rd.PolicyError("at most %d components may be disabled" % MAX_DISABLED)
    return uniq


def edit_step(disabled) -> dict:
    """Canonical one-step program: {'op': P_NoMixRecipe} when nothing is disabled (semantic alias of the preset), else the edit step."""
    d = canonical_disabled(disabled)
    return {"op": RECIPE} if not d else {"op": RECIPE_EDIT, "disabled_ops": d}


def is_edit_program(program) -> bool:
    return isinstance(program, list) and len(program) == 1 and isinstance(program[0], dict) and program[0].get("op") == RECIPE_EDIT


def recipe_edit_plan_and_apply(series, streams: Streams, start, disabled) -> tuple:
    """Shadow path: the unchanged preset runs on a copy of the window and advances `streams` exactly as P_NoMixRecipe would, recording the
    three RNG states at the entry of every step. Edit path: the retained steps are replayed on the real window from those recorded states,
    in the original order; disabled steps are skipped without any redraw. The replay consumes detached copies only, so the shadow's later
    draws (next windows) are untouched. Returns (series, retained step names, identity flags, shadow step names)."""
    import torch
    records = []
    recipe_plan_and_apply(series.clone(), streams, start, observer=lambda name, snap: records.append((name, snap)))
    out, executed, ident = series, [], []
    for name, snap in records:
        if name in disabled:
            continue
        before = out.clone()
        out = apply_primitive(name, out, Streams.from_snapshot(snap), start)
        executed.append(name)
        ident.append(bool(torch.equal(before, out)))
    return out, executed, ident, [name for name, _ in records]


def run_program_edit(z: np.ndarray, program: list, streams: Streams, start=None) -> tuple:
    """One P_NoMixRecipe_Edit step on one joint window. Returns (float32 child (240,), retained steps, identity flags, shadow steps)."""
    if not is_edit_program(program):
        raise ValueError("run_program_edit takes exactly one %s step" % RECIPE_EDIT)
    disabled = canonical_disabled(program[0].get("disabled_ops", []))
    series, executed, identity, shadow = recipe_edit_plan_and_apply(as_series(z), streams, start, disabled)
    out = series.reshape(-1).numpy().astype(np.float32)
    if out.shape != (np.asarray(z).size,) or not np.isfinite(out).all():
        raise rd.ProgramExecutionError("NONFINITE_OR_LENGTH_CHANGED")
    return out, executed, identity, shadow


def run_program(z: np.ndarray, program: list, streams: Streams, start=None) -> tuple:
    """z: one joint window (240,). program: canonical steps. Returns (float32 child (240,), executed step names, identity flags)."""
    import torch
    series = as_series(z)
    executed, identity = [], []
    if program == [{"op": RECIPE}]:
        series, executed, identity = recipe_plan_and_apply(series, streams, start)
    else:
        for st in program:
            before = series.clone()
            series = apply_primitive(st["op"], series, streams, start)
            executed.append(st["op"])
            identity.append(bool(torch.equal(before, series)))
    out = series.reshape(-1).numpy().astype(np.float32)
    if out.shape != (np.asarray(z).size,) or not np.isfinite(out).all():
        raise rd.ProgramExecutionError("NONFINITE_OR_LENGTH_CHANGED")
    return out, executed, identity


# --------------------------------------------------------------------------- plan DSL (existing predicate evaluator)
def validate_steps(steps) -> list:
    if not isinstance(steps, list) or len(steps) > MAX_STEPS:
        raise rd.PolicyError("steps must be a list of at most %d steps" % MAX_STEPS)
    names = []
    for st in steps:
        if not isinstance(st, dict) or "op" not in st or set(st) - {"op", "profile"}:
            raise rd.PolicyError("each step is {'op': <primitive or %s>} (optional 'profile': 'source'); no other parameters exist" % RECIPE)
        if st.get("profile", "source") != "source":
            raise rd.PolicyError("only the 'source' parameter profile exists in this version")
        if st["op"] not in PRIMITIVES and st["op"] != RECIPE:
            raise rd.PolicyError("unknown op %r; legal: %s or the preset %s" % (st["op"], list(PRIMITIVES), RECIPE))
        names.append(st["op"])
    if RECIPE in names and len(names) > 1:
        raise rd.PolicyError("%s is a complete program; it cannot be combined with other steps" % RECIPE)
    if len(set(names)) != len(names):
        raise rd.PolicyError("a program may not repeat a primitive")
    for a, b in EXCLUSIVE:
        if a in names and b in names:
            raise rd.PolicyError("%s and %s are mutually exclusive (same source category)" % (a, b))
    return [{"op": n} for n in sorted(names, key=lambda n: FIXED_ORDER.get(n, -1))]


def validate_plan(pol) -> dict:
    if not isinstance(pol, dict) or "default" not in pol:
        raise rd.PolicyError("policy must be an object with a 'default'")
    if set(pol) - {"default", "rules", "rationale", "observation_fields_used"}:
        raise rd.PolicyError("unknown policy keys %s" % sorted(set(pol) - {"default", "rules", "rationale", "observation_fields_used"}))
    if not isinstance(pol["default"], dict) or set(pol["default"]) != {"steps"}:
        raise rd.PolicyError("default must be {'steps': [...]}")
    clean = {"default": {"steps": validate_steps(pol["default"]["steps"])}, "rules": []}
    rules = pol.get("rules", []) or []
    if not isinstance(rules, list) or len(rules) > MAX_RULES:
        raise rd.PolicyError("rules must be a list of at most %d rules" % MAX_RULES)
    for r in rules:
        if not isinstance(r, dict) or set(r) != {"when", "steps"}:
            raise rd.PolicyError("each rule must be {'when': predicate, 'steps': [...]}")
        rd.validate_predicate(r["when"])
        clean["rules"].append({"when": r["when"], "steps": validate_steps(r["steps"])})
    rationale = str(pol.get("rationale", ""))[:500]
    rd.validate_text(rationale, "rationale")
    clean["rationale"] = rationale
    clean["observation_fields_used"] = [f for f in (pol.get("observation_fields_used") or []) if f in rd.FIELDS]
    return clean


def compile_plan(pol: dict, table: list) -> dict:
    clean = validate_plan(pol)
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


def material_key(assignment: list) -> str:
    return json.dumps({"profile": PROFILE_VERSION, "assignment": assignment}, sort_keys=True, separators=(",", ":"))


def is_identity(assignment: list) -> bool:
    return all(not steps for steps in assignment)


def material_seed(job_index: int, material_index: int, base: int = SEED_BASE) -> int:
    if not (1 <= int(material_index) <= 99 and int(job_index) >= 0):
        raise ValueError("material index must be in 1..99")
    return int(base) + 10000 * int(job_index) + 100 * int(material_index)


# --------------------------------------------------------------------------- real timestamps of each legal window
def row_timestamp(dataset: str, entity: int, abs_row: int) -> _dt.datetime:
    ds = rd.DATASETS[dataset]
    if ds["kind"] == "tsf_zip":
        name = ds["roster"][entity]
        start = _dt.datetime.strptime(rd._tsf_series(ds["path"], ds["roster"])[name]["start"], "%Y-%m-%d %H-%M-%S")
        return start + _dt.timedelta(hours=int(abs_row))
    if ds["kind"] == "prsa_csv":                 # load_slice verified ts == first_time + row hours for every loaded row
        return _dt.datetime(*ds["first_time"]) + _dt.timedelta(hours=int(abs_row))
    raise ValueError("no timestamp metadata for dataset kind %r" % ds["kind"])


def window_starts(ctx) -> list:
    """pandas Timestamps of each legal parent's first point; checked against the slice's clock hours of every window point."""
    import pandas as pd
    lg, t0 = ctx.legal, ctx.job.train_range[0]
    hrs = ctx.slice.hours_of(*ctx.job.train_range)
    out = []
    for p in range(lg.n):
        e, k = int(lg.ent[p]), int(lg.k[p])
        ts = row_timestamp(ctx.job.dataset, e, t0 + k)
        want = hrs[k: k + L_JOINT, e]
        got = np.array([(ts + _dt.timedelta(hours=h)).hour for h in range(L_JOINT)])
        if not np.array_equal(got, want):
            raise RuntimeError("DATE_BINDING_FAILED entity %d window %d" % (e, k))
        out.append(pd.Timestamp(ts))
    return out


# --------------------------------------------------------------------------- materials (per job registry)
def _mdir(job_dir: Path) -> Path:
    d = Path(job_dir) / "aug_materials"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_registry(job_dir: Path) -> dict:
    p = Path(job_dir) / "aug_materials" / "index.json"
    return context.read_json(p) if p.exists() else {}


def resolve(job_dir: Path, material_id: str) -> dict:
    reg = load_registry(job_dir)
    if material_id not in reg:
        raise KeyError("unknown augmentation material %s" % material_id)
    rec = reg[material_id]
    return reg[rec["alias_of"]] if rec.get("alias_of") and rec["alias_of"] in reg else rec


def joint_parents(ctx) -> tuple:
    """(Z (P, 240) float64 normalized [Baseline-Linear X; raw y], ent, k). The Baseline-Linear material P0_Linear is built on demand
    with the unchanged readiness builder."""
    idx = rd.load_index(ctx.job_dir)
    if "P0_Linear" not in idx:
        rd.build(ctx, rd.compile_policy(rd.BASELINE_LINEAR, ctx.overview()["entities"])["assignment"], "P0_Linear")
        idx = rd.load_index(ctx.job_dir)
    X, y = rd.training_arrays(ctx, rd.MaterialRef(**idx["P0_Linear"]))
    return np.concatenate([X, y], axis=1), ctx.legal.ent.copy(), ctx.legal.k.copy()


def build_material(ctx, assignment: list, material_id: str, *, job_index: int, material_index: int | None = None,
                   seed_base: int = SEED_BASE, compiled: dict | None = None) -> dict:
    """Freeze one augmentation material (child view of every legal parent). Aliases: identical canonical assignment -> the existing
    material (no rebuild, no new seed); all-empty -> 'None' (no child view). material_index None = next free index >= 3
    (1 and 2 are the package's frozen P_AmpResample / P_NoMixRecipe)."""
    if len(assignment) != len(ctx.job.roster):
        raise ValueError("assignment must cover every entity")
    reg = load_registry(ctx.job_dir)
    if material_id in reg:
        raise ValueError("material id exists")
    key = material_key(assignment)
    base = {"material_id": material_id, "job_id": ctx.job.job_id, "key": key, "assignment": assignment, "compiled": compiled}
    alias = "None" if is_identity(assignment) else next((m for m, r in reg.items() if r["key"] == key and not r.get("alias_of")), None)
    if alias is not None:
        rec = {**base, "alias_of": alias, "material_index": None, "seed": None}
        reg[material_id] = rec
        context.write_json(_mdir(ctx.job_dir) / "index.json", reg)
        return rec
    used = {r["material_index"] for r in reg.values() if r.get("material_index")}
    if material_index is None:
        material_index = max([2] + sorted(used)) + 1
    if material_index in used:
        raise ValueError("material index %d already used" % material_index)
    seed = material_seed(job_index, material_index, seed_base)
    t0 = time.time()
    Z, ent, k = joint_parents(ctx)
    starts = window_starts(ctx) if any(st["op"] in ("tp_calendar", RECIPE) for steps in assignment for st in steps) else None
    streams = Streams(seed)
    C = np.empty(Z.shape, dtype=np.float32)
    per_step = {n: {"windows": 0, "identity": 0} for n in PRIMITIVES}
    recipe_sets, window_identity = {}, 0
    for e in range(len(ctx.job.roster)):                           # frozen entity -> window order
        for p in np.flatnonzero(ent == e):
            prog = assignment[e]
            if not prog:
                C[p] = Z[p].astype(np.float32)
                continue
            out, executed, ident = run_program(Z[p], prog, streams, starts[p] if starts is not None else None)
            C[p] = out
            if prog == [{"op": RECIPE}]:
                recipe_sets["+".join(executed)] = recipe_sets.get("+".join(executed), 0) + 1
            for n, i in zip(executed, ident):
                per_step[n]["windows"] += 1
                per_step[n]["identity"] += int(i)
            window_identity += int(np.array_equal(out, Z[p].astype(np.float32)))
    secs = time.time() - t0
    d = C.astype(np.float64) - Z
    mdir = _mdir(ctx.job_dir)
    npz = mdir / (material_id + ".npz")
    np.savez(npz, Xc=C[:, :spec.L], yc=C[:, spec.L:])
    ents = []
    for e in range(len(ctx.job.roster)):
        rows = np.flatnonzero(ent == e)
        de = d[rows]
        ents.append({"steps": assignment[e], "windows": int(rows.size),
                     "rms_change_X": float(np.sqrt((de[:, :spec.L] ** 2).mean())), "rms_change_y": float(np.sqrt((de[:, spec.L:] ** 2).mean())),
                     "fraction_points_changed_X": float((np.abs(de[:, :spec.L]) > 0).mean()),
                     "fraction_points_changed_y": float((np.abs(de[:, spec.L:]) > 0).mean())})
    summary = {"material_id": material_id, "job_id": ctx.job.job_id, "seed": seed, "material_index": material_index, "job_index": job_index,
               "profile": PROFILE_VERSION, "legal_parents": int(Z.shape[0]), "build_seconds": secs,
               "rms_change_X": float(np.sqrt((d[:, :spec.L] ** 2).mean())), "rms_change_y": float(np.sqrt((d[:, spec.L:] ** 2).mean())),
               "fraction_points_changed_X": float((np.abs(d[:, :spec.L]) > 0).mean()), "fraction_points_changed_y": float((np.abs(d[:, spec.L:]) > 0).mean()),
               "windows_bitwise_unchanged": window_identity, "per_step": per_step, "recipe_step_sets": recipe_sets,
               "units": "normalized by the frozen per-entity T scaler (child - parent)", "entities": ents}
    sp = mdir / (material_id + "__summary.json")
    context.write_json(sp, summary)
    rec = {**base, "alias_of": None, "material_index": material_index, "seed": seed, "path": str(npz), "summary_path": str(sp)}
    reg[material_id] = rec
    context.write_json(mdir / "index.json", reg)
    return rec


def load_child(job_dir: Path, material_id: str):
    rec = resolve(job_dir, material_id)
    if rec["material_id"] == "None" or rec.get("alias_of") == "None" or (rec.get("alias_of") is None and is_identity(rec["assignment"])):
        return None
    with np.load(rec["path"]) as z:
        return z["Xc"].copy(), z["yc"].copy()


def inspect(ctx, material_id: str, entity_index=None, window: str = "largest_change") -> dict:
    """Actual steps, change fraction / RMS for X and y separately, zero-behaviour counts, and one before/after window on request.
    A description of what changed, not a utility score."""
    reg = load_registry(ctx.job_dir)
    if material_id not in reg:
        raise KeyError(material_id)
    head = reg[material_id]
    if head.get("alias_of") == "None":
        return {"material_id": material_id, "alias_of": "None", "note": "all programs empty: identical to no augmentation (no child view)"}
    rec = resolve(ctx.job_dir, material_id)
    s = context.read_json(rec["summary_path"])
    out = {"material_id": material_id, "alias_of": head.get("alias_of"), "legal_windows": s["legal_parents"],
           "steps_executed_windows": {k: v["windows"] for k, v in s["per_step"].items() if v["windows"]},
           "step_identity_windows": {k: v["identity"] for k, v in s["per_step"].items() if v["windows"]},
           "recipe_step_sets": s["recipe_step_sets"] or None, "windows_bitwise_unchanged": s["windows_bitwise_unchanged"],
           "change_X": {"rms": round(s["rms_change_X"], 4), "fraction_points_changed": round(s["fraction_points_changed_X"], 4)},
           "change_y": {"rms": round(s["rms_change_y"], 4), "fraction_points_changed": round(s["fraction_points_changed_y"], 4)},
           "entities": {"entity_%d" % e: {"steps": [st["op"] for st in r["steps"]], "rms_X": round(r["rms_change_X"], 4), "rms_y": round(r["rms_change_y"], 4)}
                        for e, r in enumerate(s["entities"])},
           "field_note": "child - parent in frozen T-scaler units; identity = a step returned its input unchanged (e.g. censor mode 0); "
                         "the child view is trained next to the unchanged parent (0.5/0.5); targets for scoring never change",
           "interpretation": "Material diagnostics (what changed), not a predictor of downstream utility."}
    if entity_index is not None:
        e = int(entity_index)
        Z, ent, k = joint_parents(ctx)
        rows = np.flatnonzero(ent == e)
        with np.load(rec["path"]) as z:
            C = np.concatenate([z["Xc"][rows], z["yc"][rows]], axis=1).astype(np.float64)
        diff = np.abs(C - Z[rows]).max(axis=1)
        j = int(np.argmax(diff)) if window == "largest_change" else (0 if window == "first" else rows.size - 1)
        out["window"] = {"entity": "entity_%d" % e, "choice": window, "absolute_start_row": int(ctx.job.train_range[0] + k[rows[j]]),
                         "parent": [round(float(v), 3) for v in Z[rows[j]]], "child": [round(float(v), 3) for v in C[j]],
                         "input_target_split": spec.L, "units": "normalized by the frozen T scaler"}
    return out


def action_table() -> dict:
    return {"primitives": {n: {**CATALOG[n], "parameter_profiles": ["source"]} for n in PRIMITIVES},
            "preset": {RECIPE: RECIPE_TEXT},
            "composition": {"max_steps": MAX_STEPS, "fixed_execution_order": list(PRIMITIVES),
                            "mutually_exclusive": [list(x) for x in EXCLUSIVE], "no_repeats": True,
                            "listing_order": "irrelevant: steps are executed in the fixed order; identical sets are the same material"},
            "window": "Each legal training pair is one 240-point joint series [192 Baseline-Linear-filled inputs; 48 observed targets] in "
                      "frozen per-entity T-scaler units; the primitive's output is an extra CHILD pair trained next to the unchanged parent "
                      "(loss 0.5 parent + 0.5 child, same batch indices). Entities with empty steps put their parent in the child slot. "
                      "An all-empty plan is 'None' (no child view). Serving inputs and scoring targets are never augmented.",
            "randomness": "Material randomness is frozen per (job, material) by the Harness; there is no seed or distribution argument. "
                          "Rebuilding an identical plan returns the same material.",
            "not_available": "mixup / donors / synthetic sources, time reversal, sign flip, differencing/integration, quantization, NaN "
                             "injection, extra scaling or noise, change-size filtering."}
