"""Material Plan -> Material: expand a per-entity assignment (from policy.compile_policy) into the
actual derived arrays for the whole batch, on disk, with full step logs (donor indices etc.).
Identity entities get no derived pair (child view == parent view; the fit worker uses the parent
view only for those rows -- the None/Copy equivalence already verified in the confirmation package).
Two assignments with the same key are the same material: build() returns the existing ref as an
alias instead of re-generating anything. inspect_material() reports what the material changed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import augment, context, spec


@dataclass
class MaterialRef:
    material_id: str
    job_id: str
    path: str                 # npz with Xc, Yc (32,433,L/H) and has_child (32,) bool; absent for all-identity
    key: str
    assignment: list
    n_identity: int
    alias_of: str | None
    steps_log_path: str


def _index_path(ctx: context.JobContext) -> Path:
    return ctx.job_dir / "materials" / "index.json"


def _load_index(ctx) -> dict:
    p = _index_path(ctx)
    return context.read_json(p) if p.exists() else {}


def build(ctx: context.JobContext, assignment: list, material_id: str) -> MaterialRef:
    if ctx.parents is None:
        raise PermissionError("build_material needs the T (material) stage")
    if len(assignment) != len(ctx.parents.entity_names):
        raise ValueError("assignment must cover all %d entities" % len(ctx.parents.entity_names))
    key = json.dumps(assignment, sort_keys=True, separators=(",", ":"))
    index = _load_index(ctx)
    for mid, rec in index.items():
        if rec["key"] == key:
            ref = MaterialRef(**rec)
            ref = MaterialRef(material_id=material_id, job_id=ref.job_id, path=ref.path, key=key, assignment=assignment,
                              n_identity=ref.n_identity, alias_of=rec["material_id"] if rec["alias_of"] is None else rec["alias_of"], steps_log_path=ref.steps_log_path)
            index[material_id] = asdict(ref)
            context.write_json(_index_path(ctx), index)
            return ref
    mdir = ctx.job_dir / "materials"
    mdir.mkdir(parents=True, exist_ok=True)
    P = ctx.parents
    E, n = P.X_norm.shape[0], P.n_parents
    Xc = np.empty_like(P.X_norm)
    Yc = np.empty_like(P.y_norm)
    has_child = np.zeros(E, dtype=bool)
    logs = {}
    for e in range(E):
        steps = assignment[e]
        xc, yc, log = augment.apply_steps(P.X_norm[e], P.y_norm[e], P.hours, steps, e)
        if xc is None:
            Xc[e], Yc[e] = P.X_norm[e], P.y_norm[e]         # identity: child view == parent view
        else:
            Xc[e], Yc[e] = xc, yc
            has_child[e] = True
        logs["entity_%d" % e] = {"steps": steps, "log": log}
    npz = mdir / (material_id + ".npz")
    np.savez(npz, Xc=Xc, Yc=Yc, has_child=has_child)
    lp = mdir / (material_id + "__steps.json")
    context.write_json(lp, {"material_id": material_id, "job_id": ctx.job.job_id, "aug_seed_rule": "900090 + i + 100000*step", "entities": logs})
    ref = MaterialRef(material_id=material_id, job_id=ctx.job.job_id, path=str(npz), key=key, assignment=assignment,
                      n_identity=int((~has_child).sum()), alias_of=None, steps_log_path=str(lp))
    index[material_id] = asdict(ref)
    context.write_json(_index_path(ctx), index)
    return ref


def load_child(ref: MaterialRef):
    """Flattened (Xc, Yc) for the fit worker, or None when every entity is identity."""
    z = np.load(ref.path)
    if not z["has_child"].any():
        return None
    Xc, Yc = z["Xc"], z["Yc"]
    return Xc.reshape(-1, Xc.shape[-1]), Yc.reshape(-1, Yc.shape[-1])


def inspect_material(ctx: context.JobContext, ref: MaterialRef) -> dict:
    """Zero-fit description of what the material changed, per entity (T-only; proxies, not utility)."""
    if ctx.parents is None:
        raise PermissionError("inspect_material needs the T stage")
    z = np.load(ref.path)
    Xc, Yc, hc = z["Xc"], z["Yc"], z["has_child"]
    P = ctx.parents
    idx = np.arange(P.n_parents)
    logs = context.read_json(ref.steps_log_path)["entities"]
    out = {"material_id": ref.material_id, "alias_of": ref.alias_of, "n_identity": int((~hc).sum()), "entities": {}}
    for e in range(P.X_norm.shape[0]):
        name = "entity_%d" % e
        rec = {"steps": ref.assignment[e], "identity": bool(not hc[e])}
        if hc[e]:
            rec["child_x_change_rms"] = float(np.sqrt(((Xc[e] - P.X_norm[e]) ** 2).mean(-1)).mean())
            rec["child_y_change_rms"] = float(np.sqrt(((Yc[e] - P.y_norm[e]) ** 2).mean(-1)).mean())
            donor_steps = [s for s in logs[name]["log"] if "donor_idx" in s]
            if donor_steps:
                d = np.asarray(donor_steps[0]["donor_idx"])
                dt = np.abs(d - idx)
                Xd = P.X_norm[e][d]
                rec.update({"donor_rule": donor_steps[0]["donor_rule"], "donor_abs_time_distance_mean_h": float(dt.mean()),
                            "donor_same_hour_ratio": float(((d - idx) % 24 == 0).mean()),
                            "parent_donor_x_rms": float(np.sqrt(((Xd - P.X_norm[e]) ** 2).mean(-1)).mean()),
                            "parent_donor_x_level_diff_abs_mean": float(np.abs(Xd.mean(-1) - P.X_norm[e].mean(-1)).mean())})
        out["entities"][name] = rec
    changed = [v for v in out["entities"].values() if not v["identity"]]
    out["summary"] = {"n_changed_entities": len(changed),
                      "child_x_change_rms_mean": (float(np.mean([v["child_x_change_rms"] for v in changed])) if changed else 0.0),
                      "child_y_change_rms_mean": (float(np.mean([v["child_y_change_rms"] for v in changed])) if changed else 0.0)}
    return out
