"""Frozen specification for the batch-level construction base
(docs/BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md, docs/BATCH_FRAMEWORK_COMPARISON_2026-09-13.md §5,
AGENTS.md §5.9). Shared by the P (fixed-flow policy) and W (tool-loop) controllers; neither line may
change anything here without the change being a public, both-line change.

Numbers are copied from the augmentation line's verified packages
(_scratch/ts_aug_source_grounding_pilot/config.py, _scratch/ts_aug_donor_pattern_probe/dp_config.py)
so that materials and training reproduce those runs bit-for-bit under the same seeds.
No hashing anywhere in this package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- datasets (TSLib CSV development bundles, not Monash)
_ROSTER_32 = ["0", "1", "10", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "11", "110", "111",
              "112", "113", "114", "115", "116", "117", "118", "119", "12", "120", "121", "122", "123", "124", "125", "126"]
DATASETS = {
    "electricity": {"path": Path(r"C:\Users\辉\Desktop\Agent\shared_tsq_datasets\electricity\electricity.csv"),
                    "total_hours": 26304, "first_timestamp": "2016-07-01 02:00:00", "roster": list(_ROSTER_32)},
    "traffic": {"path": Path(r"C:\Users\辉\Desktop\Agent\shared_tsq_datasets\traffic\traffic.csv"),
                "total_hours": 17544, "first_timestamp": "2016-07-01 02:00:00", "roster": list(_ROSTER_32)},
}
ROSTER_SIZE = 32
EXPOSURE = "EXPOSED_DEVELOPMENT"     # both files were used as development material; never fresh/held-out

# --------------------------------------------------------------------------- geometry
L, H, TRAIN_SPAN = 192, 48, 672
N_PARENTS = TRAIN_SPAN - (L + H) + 1          # 433
N_POOL = ROSTER_SIZE * N_PARENTS              # 13856
ZERO_SCALE_FLOOR = 1e-6
MASE_SEASON = 168

# --------------------------------------------------------------------------- consumer (fixed; not editable by any agent)
CONSUMER = {"arch": "MLP 192->128->64->48, ReLU, no dropout/BN", "hidden1": 128, "hidden2": 64,
            "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "n_updates": 2000, "parent_batch": 64,
            "loss": "pooled MSE on normalized targets; parent view 0.5 + child view 0.5 when a child view exists",
            "normalization": "per-entity mean/std fitted on T only; derived pairs are NOT re-normalized",
            "metric": "normalized MSE, 48-step mean, entity macro mean, origins equal-weighted within a block"}
HIDDEN1, HIDDEN2 = CONSUMER["hidden1"], CONSUMER["hidden2"]
LR, WEIGHT_DECAY, N_UPDATES, PARENT_BATCH = CONSUMER["lr"], CONSUMER["weight_decay"], CONSUMER["n_updates"], CONSUMER["parent_batch"]

# --------------------------------------------------------------------------- action space (public; both lines; task doc V1 §3)
ACTIONS = {
    "timemixup": {"donor_rule": ["R", "U"], "w": [0.10, 0.25, 0.50]},
    "freqmask": {"mu": [0.05, 0.10, 0.20]},
    "freqmix": {"donor_rule": ["R", "U"], "mu": [0.05, 0.10, 0.20]},
}
MAX_STEPS = 2
DONOR_RULES = {"R": "random derangement over the entity's own 433 original parents (no self, one-to-one)",
               "U": "same-clock-hour derangement: 24 hour groups by parent start hour, hours ascending, one derangement per group"}
DEFAULT_PARAMS = {"timemixup": {"donor_rule": "R", "w": 0.25}, "freqmask": {"mu": 0.10}, "freqmix": {"donor_rule": "R", "mu": 0.10}}
# the historical strong fixed incumbent (night package plan B == probe plan R)
FIXED_MIXUP_STEPS = [{"op": "timemixup", "donor_rule": "R", "w": 0.25}]

# --------------------------------------------------------------------------- randomness
AUG_SEED_BASE, AUG_STEP_OFFSET = 900_090, 100_000      # step s of entity i: RandomState(900090 + i + 100000*s)
BATCH_SEED_BASE, BATCH_SEED_TERM = 800_000, 9          # parent-batch stream for model seed r: 800000 + 100*r + 9


def aug_seed(entity_index: int, step_index: int) -> int:
    return AUG_SEED_BASE + entity_index + AUG_STEP_OFFSET * step_index


def batch_seed(model_seed: int) -> int:
    return BATCH_SEED_BASE + 100 * model_seed + BATCH_SEED_TERM


# --------------------------------------------------------------------------- jobs and stage row permissions
@dataclass(frozen=True)
class JobSpec:
    dataset: str
    job_id: str
    t: int
    train_range: tuple            # [t-672, t)
    c_a: tuple                    # origins
    c_b: tuple
    e: tuple
    roster_override: tuple | None = None      # explicit 32-entity population (SKILL-REVISION §5.4); None = dataset default
    material_rows: tuple = field(init=False)   # T only
    evaluate_rows: tuple = field(init=False)   # T + C_A inputs/targets; physically excludes C_B
    c_b_rows: tuple = field(init=False)        # C_B inputs/targets only
    e_input_rows: tuple = field(init=False)    # E inputs only (never the last E target block)
    e_target_rows: tuple = field(init=False)

    def __post_init__(self):
        t = self.t
        object.__setattr__(self, "material_rows", (t - TRAIN_SPAN, t))
        object.__setattr__(self, "evaluate_rows", (t - TRAIN_SPAN, max(self.c_a) + H))
        object.__setattr__(self, "c_b_rows", (min(self.c_b) - L, max(self.c_b) + H))
        object.__setattr__(self, "e_input_rows", (min(self.e) - L, max(self.e)))
        object.__setattr__(self, "e_target_rows", (min(self.e), max(self.e) + H))
        assert self.evaluate_rows[1] <= min(self.c_b), "evaluate slice must not reach a C_B origin"
        assert self.e_input_rows[1] <= max(self.e)
        assert max(self.e) + H <= DATASETS[self.dataset]["total_hours"], "job runs past the file"
        if self.roster_override is not None:
            r = tuple(self.roster_override)
            if len(r) != ROSTER_SIZE or len(set(r)) != ROSTER_SIZE or any(not isinstance(x, str) for x in r):
                raise ValueError("roster override must be %d distinct column names" % ROSTER_SIZE)
            object.__setattr__(self, "roster_override", r)

    @property
    def roster(self) -> list:
        if self.roster_override is not None:
            return list(self.roster_override)
        return list(DATASETS[self.dataset]["roster"])


def t_of(dataset: str, a: float) -> int:
    return 24 * math.floor(a * DATASETS[dataset]["total_hours"] / 24.0)


def job_spec(dataset: str, a: float, job_id: str | None = None, roster=None) -> JobSpec:
    t = t_of(dataset, a)
    return JobSpec(dataset=dataset, job_id=job_id or ("a%02d" % round(a * 100)), t=t, train_range=(t - TRAIN_SPAN, t),
                   c_a=(t, t + 48), c_b=(t + 96, t + 144), e=(t + 192, t + 240, t + 288, t + 336),
                   roster_override=None if roster is None else tuple(roster))


# frozen expectations for the electricity jobs already used by the augmentation line (checked at import)
_EXPECTED_ELEC = {"a55": 14448, "a65": 17088, "a75": 19728, "a85": 22344, "a94": 24720}
for _j, _t in _EXPECTED_ELEC.items():
    assert job_spec("electricity", int(_j[1:]) / 100.0).t == _t, ("job table drifted", _j)

# --------------------------------------------------------------------------- observation field list (public; policy predicates may only reference these)
OBS_FIELDS = ["mean", "std", "missing_count", "head168_mean", "tail168_mean", "standardized_trend_per_100h",
              "last168_std", "full_T_std", "last168_std_over_full_T_std", "lag24_corr", "lag168_corr",
              "nondc_spectrum_top5_energy_share", "standardized_abs_p95", "standardized_abs_max", "n_parent_pairs",
              "r_head", "r_tail", "r_gap", "nn1_input_distance", "nn5_target_deviation"]
PREDICATE_OPS = {"<", "<=", ">", ">=", "==", "!="}
FORBIDDEN_TEXT = ("electricity", "traffic", "job_id", "a55", "a65", "a75", "a85", "a94")   # dataset/job identifiers; "seed" as a word is allowed
