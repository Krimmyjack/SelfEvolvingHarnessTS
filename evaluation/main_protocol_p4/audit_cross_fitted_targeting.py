"""Can deployment-visible features pick the right program for a series it has not seen?

The sweep's per-series gain tensor makes this answerable with no new Consumer
fit.  For every (program, origin, face, series) the gain is already on disk, so
a targeter can be trained on one group of series and scored on another purely by
indexing.

Two fold schemes, and the order matters:

* **cross-face is primary.**  The admission ladder recomputed from the tensor
  shows the binding constraint is inside a single origin: at 1176 and 2616 the
  Support-A and Support-B admitted sets are disjoint, and at 1896 and 2376 one
  face admits nothing at all.  Cross-origin generalisation is the question
  after that one, not before it -- only 2 of 6 origins have any stable
  candidate, so leave-one-origin-out has almost nothing to hold out.
* **cross-origin is secondary**, reported for completeness.

Every fold selects its own menu and trains its own targeter on the training side
only; the test side is touched to read gains and nothing else.  ``raw`` is
always in the menu with gain 0 by construction, so the targeter is free to
abstain and abstention can never be scored as a gain.

The tree is a diagnostic baseline, not the proposed method: it answers whether
the *features* carry program-choice signal at all.  A null here indicts the
feature vocabulary; it does not vindicate or condemn targeting as a mechanism.

0 LLM calls, 0 Consumer fits, 0 held-out reads.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TENSOR = PROJECT_ROOT / "artifacts/main_protocol/p4d_gap_per_series_gain.npz"
CORRECTED = PROJECT_ROOT / "artifacts/main_protocol/p4d_param_corrected_gain.npz"
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4g_cross_fitted_targeting.json"

FACES = ("support_a", "support_b")
MATERIAL = 0.005
CONTEXT_LENGTH = contract.CONTEXT_LENGTH
MENU_SIZE = 6
RAW = "raw"


def load_tensor() -> dict[str, Any]:
    """The sweep's gains, with corrected-parameter readings folded in."""
    base = np.load(TENSOR, allow_pickle=True)
    gain = np.array(base["gain"], dtype=np.float64)
    ids = [str(value) for value in base["program_ids"]]
    origins = [int(value) for value in base["origins"]]
    replaced = 0
    if CORRECTED.is_file():
        fixed = np.load(CORRECTED, allow_pickle=True)
        fixed_ids = [str(value) for value in fixed["program_ids"]]
        fixed_origins = [int(value) for value in fixed["origins"]]
        for f_index, program in enumerate(fixed_ids):
            if program not in ids:
                continue
            target = ids.index(program)
            for o_index, origin in enumerate(fixed_origins):
                if origin not in origins:
                    continue
                gain[target, origins.index(origin), :, :] = fixed["gain"][
                    f_index, o_index, :, :
                ]
            replaced += 1
    return {
        "gain": gain,
        "program_ids": ids,
        "origins": origins,
        "support_a": [str(value) for value in base["support_a"]],
        "support_b": [str(value) for value in base["support_b"]],
        "programs_replaced_by_correction": replaced,
    }


def distinct_programs(gain: np.ndarray, ids: Sequence[str]) -> list[int]:
    """One index per distinct effect, keeping only programs readable everywhere.

    Programs that compose an operator with a no-op share an effect vector;
    counting them separately would let one discovery vote several times, and in
    a menu it would waste the budget on aliases.
    """
    keep: dict[bytes, int] = {}
    for index in range(gain.shape[0]):
        block = gain[index]
        if bool(np.isnan(block).any()):
            continue
        signature = np.round(block, 9).tobytes()
        keep.setdefault(signature, index)
    return sorted(keep.values())


def series_features(values: Mapping[str, np.ndarray], uids: Sequence[str],
                    origin: int) -> np.ndarray:
    """Pre-origin, deployment-visible features -- no Outcome, no future."""
    rows = []
    for uid in uids:
        window = np.asarray(
            values[uid][origin - CONTEXT_LENGTH:origin], dtype=np.float64
        )
        card = dict(extract_public_features(window, task_kind=forecast_p4.TASK))
        rows.append(card)
    names = sorted(
        name for name in rows[0]
        if isinstance(rows[0][name], (int, float, bool))
        and not isinstance(rows[0][name], str)
    )
    matrix = np.array(
        [[float(row.get(name) or 0.0) for name in names] for row in rows],
        dtype=np.float64,
    )
    return matrix, names


def _risk(vector: np.ndarray) -> dict[str, Any]:
    worst = float(-vector.min()) if vector.size else 0.0
    return {
        "mean_gain": round(float(vector.mean()), 6) if vector.size else None,
        "harmed_fraction": round(float((vector < -MATERIAL).mean()), 4),
        "max_single_series_harm": round(max(0.0, worst), 6),
        "clears_bounded_budget": bool(
            vector.size
            and float(vector.mean()) >= MATERIAL
            and float((vector < -MATERIAL).mean())
            <= contract.BOUNDED_MAX_HARMED_FRACTION
            and max(0.0, worst) <= contract.BOUNDED_MAX_SINGLE_SERIES_HARM
        ),
    }


def _fold(train: np.ndarray, test: np.ndarray, train_x: np.ndarray,
          test_x: np.ndarray, ids: Sequence[str]) -> dict[str, Any]:
    """One fold.  Menu, best-fixed and targeter all come from the train side."""
    # Menu: the highest mean-gain distinct programs on training series only.
    order = np.argsort(-train.mean(axis=1))[:MENU_SIZE]
    menu_gain_train = np.vstack([train[order], np.zeros((1, train.shape[1]))])
    menu_gain_test = np.vstack([test[order], np.zeros((1, test.shape[1]))])
    menu_names = [ids[index] for index in order] + [RAW]

    best_fixed_index = int(np.argmax(menu_gain_train.mean(axis=1)))
    best_fixed = menu_gain_test[best_fixed_index]

    labels = np.argmax(menu_gain_train, axis=0)
    if len(set(labels.tolist())) < 2:
        targeter = np.full(test.shape[1], best_fixed_index, dtype=int)
        note = "training side had a single argmax; targeter falls back to best-fixed"
    else:
        tree = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=2, random_state=0
        ).fit(train_x, labels)
        targeter = tree.predict(test_x).astype(int)
        note = None
    chosen = menu_gain_test[targeter, np.arange(test.shape[1])]
    oracle = menu_gain_test.max(axis=0)
    return {
        "menu": menu_names,
        "best_fixed_choice": menu_names[best_fixed_index],
        "targeter_choice_histogram": {
            menu_names[index]: int((targeter == index).sum())
            for index in sorted(set(targeter.tolist()))
        },
        "raw": _risk(np.zeros(test.shape[1])),
        "best_fixed": _risk(best_fixed),
        "targeter": _risk(chosen),
        "per_series_oracle": _risk(oracle),
        "targeter_minus_best_fixed": round(
            float(chosen.mean() - best_fixed.mean()), 6),
        "oracle_minus_best_fixed": round(
            float(oracle.mean() - best_fixed.mean()), 6),
        "targeter_share_of_oracle_gap": (
            None if abs(float(oracle.mean() - best_fixed.mean())) < 1e-9
            else round(float((chosen.mean() - best_fixed.mean())
                             / (oracle.mean() - best_fixed.mean())), 4)
        ),
        "note": note,
    }


def build() -> dict[str, Any]:
    data = load_tensor()
    gain, ids, origins = data["gain"], data["program_ids"], data["origins"]
    keep = distinct_programs(gain, ids)
    gain, ids = gain[keep], [ids[index] for index in keep]

    variant = preflight.load_variant()
    uids = {"support_a": data["support_a"], "support_b": data["support_b"]}
    features = {
        (origin, face): series_features(variant, uids[face], origin)
        for origin in origins
        for face in FACES
    }
    feature_names = features[(origins[0], FACES[0])][1]

    cross_face = []
    for o_index, origin in enumerate(origins):
        for train_face, test_face in ((0, 1), (1, 0)):
            fold = _fold(
                gain[:, o_index, train_face, :], gain[:, o_index, test_face, :],
                features[(origin, FACES[train_face])][0],
                features[(origin, FACES[test_face])][0],
                ids,
            )
            fold.update({"origin": int(origin),
                         "train_face": FACES[train_face],
                         "test_face": FACES[test_face]})
            cross_face.append(fold)

    cross_origin = []
    for o_index, origin in enumerate(origins):
        others = [index for index in range(len(origins)) if index != o_index]
        train = np.concatenate(
            [gain[:, index, face, :] for index in others for face in (0, 1)], axis=1
        )
        train_x = np.concatenate(
            [features[(origins[index], FACES[face])][0]
             for index in others for face in (0, 1)], axis=0
        )
        test = np.concatenate([gain[:, o_index, face, :] for face in (0, 1)], axis=1)
        test_x = np.concatenate(
            [features[(origin, FACES[face])][0] for face in (0, 1)], axis=0
        )
        fold = _fold(train, test, train_x, test_x, ids)
        fold["held_out_origin"] = int(origin)
        cross_origin.append(fold)

    def _summary(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "folds": len(folds),
            "mean_best_fixed": round(float(np.mean(
                [fold["best_fixed"]["mean_gain"] for fold in folds])), 6),
            "mean_targeter": round(float(np.mean(
                [fold["targeter"]["mean_gain"] for fold in folds])), 6),
            "mean_per_series_oracle": round(float(np.mean(
                [fold["per_series_oracle"]["mean_gain"] for fold in folds])), 6),
            "folds_where_targeter_beats_best_fixed": sum(
                1 for fold in folds if fold["targeter_minus_best_fixed"] > 0),
            "folds_where_best_fixed_clears_budget": sum(
                1 for fold in folds if fold["best_fixed"]["clears_bounded_budget"]),
            "folds_where_targeter_clears_budget": sum(
                1 for fold in folds if fold["targeter"]["clears_bounded_budget"]),
            "mean_oracle_headroom_over_best_fixed": round(float(np.mean(
                [fold["oracle_minus_best_fixed"] for fold in folds])), 6),
        }

    face_summary = _summary(cross_face)
    origin_summary = _summary(cross_origin)
    return {
        "stage": "P4G_CROSS_FITTED_TARGETING",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_OFFLINE_DIAGNOSTIC",
        "data_version": preflight.DATA_VERSION,
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": 0,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "reads_only": "the persisted per-series gain tensor",
        },
        "design": {
            "programs_in_tensor": len(data["program_ids"]),
            "distinct_effects_readable_everywhere": len(ids),
            "programs_replaced_by_param_correction": data[
                "programs_replaced_by_correction"],
            "menu_size": MENU_SIZE,
            "menu_includes_raw": True,
            "targeter": "DecisionTreeClassifier(max_depth=3, min_samples_leaf=2)",
            "features": feature_names,
            "feature_count": len(feature_names),
            "primary_scheme": "cross_face",
            "why_primary": (
                "the binding constraint is inside one origin: A/B admitted sets "
                "are disjoint at 1176 and 2616 and one face admits nothing at "
                "1896 and 2376, so unseen-series-group is the question that "
                "precedes unseen-origin"
            ),
        },
        "cross_face": {"summary": face_summary, "folds": cross_face},
        "cross_origin": {"summary": origin_summary, "folds": cross_origin},
        "verdict": (
            "FEATURES_CARRY_PROGRAM_CHOICE_SIGNAL"
            if face_summary["folds_where_targeter_beats_best_fixed"]
            > len(cross_face) / 2
            else "FEATURES_DO_NOT_BEAT_A_FIXED_CHOICE"
        ),
        "what_this_does_not_claim": [
            "a null does not condemn targeting as a mechanism; it indicts this "
            "feature vocabulary and this menu",
            "the tree is a diagnostic baseline, not the proposed method",
            "no held-out or fresh data was touched; this is oracle-space "
            "development diagnosis",
        ],
        "releases": "NONE",
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    design = report["design"]
    print("distinct effects readable everywhere: %d of %d (%d corrected)" % (
        design["distinct_effects_readable_everywhere"],
        design["programs_in_tensor"],
        design["programs_replaced_by_param_correction"]))
    for scheme in ("cross_face", "cross_origin"):
        s = report[scheme]["summary"]
        print("--- %s (%d folds)" % (scheme, s["folds"]))
        print("    raw 0.000000 | best-fixed %+.6f | targeter %+.6f | oracle %+.6f"
              % (s["mean_best_fixed"], s["mean_targeter"],
                 s["mean_per_series_oracle"]))
        print("    targeter beats best-fixed in %d/%d folds; budget cleared "
              "best-fixed %d, targeter %d" % (
                  s["folds_where_targeter_beats_best_fixed"], s["folds"],
                  s["folds_where_best_fixed_clears_budget"],
                  s["folds_where_targeter_clears_budget"]))
    print("verdict : %s" % report["verdict"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
