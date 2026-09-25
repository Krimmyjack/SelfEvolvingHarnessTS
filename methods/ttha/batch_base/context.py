"""JobContext: one job opened at one stage. The stage decides which rows are physically loaded:

  material  T only                      observe / build_material / inspect_material
  evaluate  T + C_A inputs/targets      fit worker (scores C_A only; C_B rows are not in memory)
  c_b       C_B inputs/targets only     commit.open_c_b (after commit)
  e_input   E inputs only               commit.freeze_e
  e_target  E targets only              commit.score_e

Materials, models and records of one run live under run_dir/<job_id>/.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import data, observe, spec

STAGE_ROWS = {"material": "material_rows", "evaluate": "evaluate_rows", "c_b": "c_b_rows", "e_input": "e_input_rows", "e_target": "e_target_rows"}


@dataclass
class JobContext:
    job: spec.JobSpec
    stage: str
    run_dir: Path
    slice: data.Slice
    scaler: data.Scaler
    parents: data.Parents | None
    mase: np.ndarray | None
    _overview: dict | None = field(default=None, repr=False)

    @property
    def job_dir(self) -> Path:
        d = self.run_dir / self.job.job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def overview(self) -> dict:
        if self.parents is None:
            raise PermissionError("overview needs the T stage")
        if self._overview is None:
            p = self.job_dir / "overview.json"
            if p.exists():
                self._overview = read_json(p)
            else:
                self._overview = observe.overview(self.slice, self.job, self.scaler, self.parents)
                write_json(p, self._overview)
        return self._overview

    def inspect_data(self, entity_indices, kind: str = "hour_profile", sub_range=None) -> dict:
        return observe.inspect_data(self.slice, self.job, self.scaler, entity_indices, kind, sub_range)


_JOB_ID = re.compile(r"^a(\d+)(?:_[A-Za-z0-9]+)?$")


def resolve_job(dataset: str, job: str | float, roster=None) -> spec.JobSpec:
    """'a85' -> a=0.85; 'a97_g1' -> a=0.97 with job_id 'a97_g1' (group suffix only labels the population)."""
    if isinstance(job, str):
        m = _JOB_ID.match(job)
        if not m:
            raise ValueError("job id must look like a85 or a97_g1: %r" % job)
        return spec.job_spec(dataset, int(m.group(1)) / 100.0, job, roster=roster)
    return spec.job_spec(dataset, float(job), roster=roster)


def open_job(dataset: str, job: str | float, run_dir, stage: str = "material", roster=None) -> JobContext:
    js = resolve_job(dataset, job, roster)
    lo, hi = getattr(js, STAGE_ROWS[stage])
    run_dir = Path(run_dir)
    sc_path = run_dir / js.job_id / "scaler.npz"
    if stage in ("material", "evaluate"):
        sl = data.load_slice(dataset, lo, hi, roster=js.roster_override)
        scaler = data.compute_scaler(sl, js.t)
        if scaler.floor_hits:
            raise RuntimeError("ELIGIBILITY_FAIL zero-scale entities: %d" % scaler.floor_hits)
        parents = data.build_parents(sl, js.t, scaler)
        mase = data.mase_denominators(sl, js.t)
        sc_path.parent.mkdir(parents=True, exist_ok=True)
        if sc_path.exists():
            old = np.load(sc_path)
            _check_roster(old, js)
            _check_binding(old, js)
            if not (np.array_equal(old["mean"], scaler.mean) and np.array_equal(old["scale"], scaler.scale)):
                raise RuntimeError("scaler drift within run for %s" % js.job_id)
        else:
            np.savez(sc_path, mean=scaler.mean, std=scaler.std, scale=scaler.scale, mase=mase, t=js.t, hours=parents.hours, anchors=parents.anchors,
                     roster=np.array(js.roster), dataset=np.array(js.dataset))
        return JobContext(job=js, stage=stage, run_dir=run_dir, slice=sl, scaler=scaler, parents=parents, mase=mase)
    # later stages: scaler must already be frozen on disk (no T rows are loaded)
    if not sc_path.exists():
        raise RuntimeError("frozen scaler missing for %s; open the material stage first" % js.job_id)
    sc = np.load(sc_path)
    _check_roster(sc, js)
    _check_binding(sc, js)
    scaler = data.Scaler(mean=sc["mean"], std=sc["std"], scale=sc["scale"], floor_hits=0)
    sl = data.load_slice(dataset, lo, hi, roster=js.roster_override)
    return JobContext(job=js, stage=stage, run_dir=run_dir, slice=sl, scaler=scaler, parents=None, mase=sc["mase"])


def _check_roster(saved, js: spec.JobSpec) -> None:
    """The population is frozen with the scaler; a stage opened with another roster is refused."""
    if "roster" in saved.files:
        if [str(x) for x in saved["roster"]] != js.roster:
            raise RuntimeError("roster drift within run for %s" % js.job_id)
    elif js.roster_override is not None:
        raise RuntimeError("job %s was frozen without an explicit roster; cannot reopen with one" % js.job_id)


def _check_binding(saved, js: spec.JobSpec) -> None:
    """The source and cut are frozen with the scaler: a job directory reopened under another dataset or t is refused.
    Scalers written before the dataset field existed carry only t (all were electricity runs)."""
    if "dataset" in saved.files and str(saved["dataset"]) != js.dataset:
        raise RuntimeError("dataset drift within run for %s" % js.job_id)
    if "dataset" not in saved.files and js.dataset != "electricity":
        raise RuntimeError("job %s was frozen without a dataset binding; cannot reopen as %s" % (js.job_id, js.dataset))
    if "t" in saved.files and int(saved["t"]) != js.t:
        raise RuntimeError("cut drift within run for %s" % js.job_id)


def write_json(path, obj) -> None:
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_default)
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
