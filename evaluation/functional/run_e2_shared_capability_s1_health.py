"""S1: health check on the S0 primary.  0 LLM, 0 retrains, 0 Outcome.

Reads the development region only and reports frozen aggregate statistics:
missing, the outlier family, the level-shift family, periodicity, and the
channel structure.  It selects nothing and moves no threshold -- every bar
here is one that already existed (``g3_sourcing`` fixed it before any
candidate was opened) and every per-channel verdict is the public feature
extractor's own, not one this file invents.

The development boundary is the line's own: 8760 points of development with
everything at or beyond it sealed, exactly the shape #16 gave NOAA.  The
windows this line would use at 9864 and 10560 therefore sit inside the
sealed region and are not read here.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "evaluation" / "functional")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from task_episode_harness.agentic import g3_sourcing as G3  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
S0 = E2 / "shared_capability_s0_census_v1.json"
OUT_JSON = E2 / "s1_health_v1.json"
OUT_MD = E2 / "s1_health_v1.md"

DEVELOPMENT_HOURS = 8760          # #16's NOAA rule, reused verbatim
SEALED_FROM = DEVELOPMENT_HOURS   # nothing at or past this index is read
USABLE_CARDINALITY = 20           # S0's usable-channel definition
Z_PEAK_BAR = 4.0                  # g3_sourcing.public_phenomenon_census
PREVALENCE_BAR = G3.MIN_SERIES_WITH_PUBLIC_PHENOMENON


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p25": ordered[len(ordered) // 4],
        "median": statistics.median(ordered),
        "p75": ordered[(3 * len(ordered)) // 4],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def run() -> int:
    started = time.perf_counter()
    census = json.loads(S0.read_text(encoding="utf-8"))
    primary = census.get("primary")
    if not primary:
        raise SystemExit("S0 named no primary; S1 has nothing to check")
    entry = next(
        row for row in census["measured_candidates"]
        if row["candidate"] == primary
    )
    path = Path(entry["source_path"])
    array = np.load(path, mmap_mode="r")
    rows_total, channels = int(array.shape[0]), int(array.shape[1])
    block = np.asarray(array[:DEVELOPMENT_HOURS, :], dtype=np.float64)

    per_channel: dict[str, Any] = {}
    for index in range(channels):
        series = block[:, index]
        uid = "channel_%d" % index
        cardinality = int(np.unique(series).size)
        features = dict(extract_public_features(series, task_kind="forecast"))
        per_channel[uid] = {
            "cardinality": cardinality,
            "usable": bool(cardinality > USABLE_CARDINALITY),
            "missing_fraction": float(features.get("missing_fraction", 0.0)),
            "longest_missing_run_fraction": float(
                features.get("longest_missing_run_fraction", 0.0)
            ),
            "local_robust_z_peak": float(
                features.get("local_robust_z_peak", 0.0) or 0.0
            ),
            "level_excursion_score": float(
                features.get("level_excursion_score", 0.0) or 0.0
            ),
            "estimated_level_offset": float(
                features.get("estimated_level_offset", 0.0) or 0.0
            ),
            "level_region_fraction": float(
                features.get("level_region_fraction", 0.0) or 0.0
            ),
            "post_shift_support_sufficient": bool(
                features.get("post_shift_support_sufficient", False)
            ),
            "period_evidence_status": features.get("period_evidence_status"),
            "period_reliability": features.get("period_reliability"),
            "period_repair_available": bool(
                features.get("period_repair_available", False)
            ),
            "probe_directions": {
                key: features.get(key)
                for key in (
                    "imputation_probe_direction", "denoising_probe_direction",
                    "clipping_probe_direction", "level_probe_direction",
                )
            },
        }

    # --- the reference column ------------------------------------------------
    # These numbers mean nothing on their own: "23 of 24 channels have an
    # outlier" is only readable next to a corpus this line has already worked
    # on.  NOAA's development block is exposed and costs nothing to re-read,
    # and it is run through the same extractor with the same block length.
    reference: dict[str, Any] = {}
    noaa_root = PROJECT_ROOT / "data" / "benchmark_noaa_fresh_v1" / "series"
    if noaa_root.is_dir():
        ref_rows: dict[str, Any] = {}
        for station in sorted(noaa_root.iterdir()):
            values_path = station / "values.npy"
            if not values_path.is_file():
                continue
            series = np.asarray(
                np.load(values_path, mmap_mode="r")[:DEVELOPMENT_HOURS],
                dtype=np.float64,
            )
            f = dict(extract_public_features(series, task_kind="forecast"))
            ref_rows[station.name] = {
                "missing_fraction": float(f.get("missing_fraction", 0.0)),
                "local_robust_z_peak": float(
                    f.get("local_robust_z_peak", 0.0) or 0.0
                ),
                "level_excursion_score": float(
                    f.get("level_excursion_score", 0.0) or 0.0
                ),
                "post_shift_support_sufficient": bool(
                    f.get("post_shift_support_sufficient", False)
                ),
                "cardinality": int(np.unique(series[~np.isnan(series)]).size),
            }
        reference = {
            "corpus": "noaa_global_hourly, this line's own development block",
            "why": (
                "already exposed, so re-reading it opens nothing; same "
                "extractor, same 8760-point block, so the columns are "
                "comparable"
            ),
            "series": len(ref_rows),
            "missing_present": sum(
                1 for r in ref_rows.values() if r["missing_fraction"] > 0.0
            ),
            "outlier_family_z_peak_ge_4": sum(
                1 for r in ref_rows.values()
                if r["local_robust_z_peak"] >= Z_PEAK_BAR
            ),
            "level_shift_family": sum(
                1 for r in ref_rows.values() if r["post_shift_support_sufficient"]
            ),
            "distributions": {
                key: _summary([r[key] for r in ref_rows.values()])
                for key in (
                    "missing_fraction", "local_robust_z_peak",
                    "level_excursion_score", "cardinality",
                )
            },
            "per_series": ref_rows,
        }

    usable = [uid for uid, row in per_channel.items() if row["usable"]]
    degenerate = [uid for uid, row in per_channel.items() if not row["usable"]]

    def count(pred, pool=usable) -> list[str]:
        return sorted(uid for uid in pool if pred(per_channel[uid]))

    outlier_family = count(lambda r: r["local_robust_z_peak"] >= Z_PEAK_BAR)
    missing_family = count(lambda r: r["missing_fraction"] > 0.0)
    level_family = count(lambda r: r["post_shift_support_sufficient"])
    period_readable = count(
        lambda r: str(r["period_evidence_status"]) not in ("", "None", "none")
    )
    public_phenomenon = sorted(
        set(outlier_family) | set(missing_family)
    )
    non_neutral_probe = count(
        lambda r: any(
            v not in (None, "", "none", "neutral", "no_action")
            for v in r["probe_directions"].values()
        )
    )

    payload: dict[str, Any] = {
        "protocol_version": "s1_health_v1",
        "role": "Phase S step 1: is the S0 primary healthy enough to proceed",
        "candidate": primary,
        "family": entry["family"],
        "source_path": str(path),
        "source_sha256": _sha256(path),
        "llm_calls": 0,
        "consumer_retrains": 0,
        "reads": {
            "development_rows": DEVELOPMENT_HOURS,
            "sealed_from_index": SEALED_FROM,
            "rows_in_the_file": rows_total,
            "fraction_of_the_file_read": DEVELOPMENT_HOURS / rows_total,
            "opens_no_outcome": (
                "nothing at or past index %d is read.  The windows this line "
                "would use at 9864 and 10560 are inside that sealed region, "
                "so S1 cannot and does not see them." % SEALED_FROM
            ),
            "boundary_rule": (
                "8760 development points with everything beyond sealed, the "
                "same shape #16 fixed for NOAA; reused rather than invented"
            ),
        },
        "bars_reused_not_moved": {
            "usable_channel_cardinality": USABLE_CARDINALITY,
            "z_peak": Z_PEAK_BAR,
            "prevalence": PREVALENCE_BAR,
            "where_they_come_from": (
                "the cardinality rule is S0's own usable-channel definition; "
                "the z peak and the prevalence bar are "
                "g3_sourcing.public_phenomenon_census and "
                "MIN_SERIES_WITH_PUBLIC_PHENOMENON, fixed before any "
                "candidate was opened"
            ),
        },
        "structure": {
            "channels_total": channels,
            "channels_usable": len(usable),
            "channels_degenerate": len(degenerate),
            "usable_channel_ids": usable,
            "degenerate_channel_ids": degenerate,
            "development_length": DEVELOPMENT_HOURS,
            "meets_line_roster_split": len(usable) >= 16,
        },
        "prevalence_over_usable_channels": {
            "missing": {"count": len(missing_family), "which": missing_family},
            "outlier_family_z_peak_ge_4": {
                "count": len(outlier_family), "which": outlier_family,
            },
            "level_shift_family_post_shift_support": {
                "count": len(level_family), "which": level_family,
            },
            "period_evidence_readable": {
                "count": len(period_readable), "which": period_readable,
            },
            "any_non_neutral_probe_direction": {
                "count": len(non_neutral_probe), "which": non_neutral_probe,
            },
            "public_phenomenon_missing_or_z_peak": {
                "count": len(public_phenomenon), "which": public_phenomenon,
            },
        },
        "distributions_over_usable_channels": {
            "missing_fraction": _summary(
                [per_channel[u]["missing_fraction"] for u in usable]
            ),
            "local_robust_z_peak": _summary(
                [per_channel[u]["local_robust_z_peak"] for u in usable]
            ),
            "level_excursion_score": _summary(
                [per_channel[u]["level_excursion_score"] for u in usable]
            ),
            "level_region_fraction": _summary(
                [per_channel[u]["level_region_fraction"] for u in usable]
            ),
            "cardinality": _summary(
                [float(per_channel[u]["cardinality"]) for u in usable]
            ),
        },
        "reference_noaa_development_block": reference,
        "per_channel": per_channel,
        "nothing_was_selected": (
            "no channel was chosen, no threshold was tuned, and no roster was "
            "cut.  Choosing 12 train and 4 eval from these channels is S2's "
            "job and needs the boundary question below settled first."
        ),
        "structural_question_for_s2": (
            "SMD_train.npy is 28 machines concatenated into one array with no "
            "boundary index on disk.  The development block read here is the "
            "head of that array, which at 8760 of %d rows is very likely "
            "inside the first machine but is not verified to be.  If a "
            "machine boundary falls inside the block, the slice mixes two "
            "entities and every per-series reading above is a reading of the "
            "mixture.  S2 must recover the per-machine index before any "
            "roster is cut." % rows_total,
        ),
    }
    # The bar is a count.  A count does not say whether the substrate has the
    # same *shape*, and that is the thing Phase S actually depends on: a
    # capability induced on one corpus can only transfer to another if the
    # operators it leans on have something to act on there.
    inert: list[dict[str, Any]] = []
    if reference:
        if pv_missing := len(missing_family) == 0:
            inert.append({
                "operator_family": "imputation",
                "finding": (
                    "0 of %d usable channels carry any missing value, against "
                    "%d of %d series on NOAA" % (
                        len(usable), reference["missing_present"],
                        reference["series"],
                    )
                ),
                "consequence": (
                    "every imputation operator in the menu is inert on this "
                    "corpus.  A Shared Capability induced on NOAA that leans "
                    "on imputation cannot be tested here at all, and a null "
                    "transfer result would be a property of the substrate "
                    "rather than of the capability"
                ),
            })
        if reference["distributions"]["level_excursion_score"]["max"] == 0.0 and (
            payload_level_median := statistics.median(
                [per_channel[u]["level_excursion_score"] for u in usable]
            )
        ) > 0.0:
            inert.append({
                "operator_family": "level shift",
                "finding": (
                    "level_excursion_score has median %.4g here and is "
                    "identically zero on all %d NOAA series" % (
                        payload_level_median, reference["series"],
                    )
                ),
                "consequence": (
                    "the two corpora differ in kind on this axis, not in "
                    "degree; a level-repair capability has no NOAA evidence "
                    "to be induced from in the first place"
                ),
            })
    payload["substrate_shape_warning"] = {
        "why_this_is_reported_beside_the_verdict": (
            "the prevalence bar counts channels with any public phenomenon "
            "and this corpus clears it.  It does not check that the "
            "phenomena are the same ones the incumbent has, and they are "
            "not."
        ),
        "findings": inert,
        "blocks_the_verdict": False,
        "what_it_blocks": (
            "it does not block S1.  It is a pre-condition on the S2 "
            "candidate: whatever Shared Capability is compiled must lean on "
            "an operator family that both corpora can exercise, or the S3 "
            "comparison measures the substrate instead of the capability."
        ),
    }
    if len(public_phenomenon) >= PREVALENCE_BAR and len(usable) >= 16:
        payload["verdict"] = "PROCEED_UNCHANGED"
        payload["verdict_reason"] = (
            "%d usable channels, %d of them carrying a public phenomenon "
            "against a bar of %d" % (
                len(usable), len(public_phenomenon), PREVALENCE_BAR,
            )
        )
    else:
        payload["verdict"] = "STOP_FOR_LOW_PREVALENCE"
        payload["verdict_reason"] = (
            "%d usable channels and %d with a public phenomenon against bars "
            "of 16 and %d" % (len(usable), len(public_phenomenon), PREVALENCE_BAR)
        )
    payload["wall_seconds"] = time.perf_counter() - started
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("verdict", payload["verdict"], flush=True)
    return 0


def _markdown(payload: dict[str, Any]) -> str:
    st = payload["structure"]
    pv = payload["prevalence_over_usable_channels"]
    lines = [
        "# S1 -- health check on `%s`" % payload["candidate"],
        "",
        "**Verdict: `%s`** -- %s" % (payload["verdict"], payload["verdict_reason"]),
        "",
        payload["reads"]["opens_no_outcome"],
        "",
        "## Structure",
        "",
        "- Channels: %d total, **%d usable**, %d degenerate (cardinality <= %d)."
        % (st["channels_total"], st["channels_usable"], st["channels_degenerate"],
           payload["bars_reused_not_moved"]["usable_channel_cardinality"]),
        "- Development block: %d points, %.1f%% of the %d rows in the file."
        % (payload["reads"]["development_rows"],
           100 * payload["reads"]["fraction_of_the_file_read"],
           payload["reads"]["rows_in_the_file"]),
        "- Meets the 12 train + 4 eval roster split: %s."
        % st["meets_line_roster_split"],
        "",
        "## Prevalence over the usable channels",
        "",
        "| family | count | of usable |",
        "| --- | ---: | ---: |",
    ]
    for key, label in (
        ("missing", "missing present"),
        ("outlier_family_z_peak_ge_4", "outlier family (z peak >= 4)"),
        ("level_shift_family_post_shift_support", "level-shift family"),
        ("period_evidence_readable", "period evidence readable"),
        ("any_non_neutral_probe_direction", "any non-neutral probe direction"),
        ("public_phenomenon_missing_or_z_peak", "public phenomenon (the g3 test)"),
    ):
        lines.append(
            "| %s | %d | %d |" % (label, pv[key]["count"], st["channels_usable"])
        )
    lines.extend(["", "## Distributions over the usable channels", "",
                  "| statistic | min | p25 | median | p75 | max |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for key, row in payload["distributions_over_usable_channels"].items():
        if not row.get("n"):
            continue
        lines.append(
            "| `%s` | %.4g | %.4g | %.4g | %.4g | %.4g |" % (
                key, row["min"], row["p25"], row["median"], row["p75"], row["max"],
            )
        )
    ref = payload.get("reference_noaa_development_block") or {}
    if ref:
        rd = ref["distributions"]
        lines.extend([
            "", "## The same numbers on NOAA, for scale", "",
            "%s (%d series). %s" % (ref["corpus"], ref["series"], ref["why"]),
            "",
            "| family | `%s` | NOAA |" % payload["candidate"],
            "| --- | ---: | ---: |",
            "| missing present | %d / %d | %d / %d |" % (
                pv["missing"]["count"], st["channels_usable"],
                ref["missing_present"], ref["series"]),
            "| outlier family (z peak >= 4) | %d / %d | %d / %d |" % (
                pv["outlier_family_z_peak_ge_4"]["count"], st["channels_usable"],
                ref["outlier_family_z_peak_ge_4"], ref["series"]),
            "| level-shift family | %d / %d | %d / %d |" % (
                pv["level_shift_family_post_shift_support"]["count"],
                st["channels_usable"], ref["level_shift_family"], ref["series"]),
            "",
            "| statistic | median here | median on NOAA | max here | max on NOAA |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for key in ("missing_fraction", "local_robust_z_peak",
                    "level_excursion_score", "cardinality"):
            here = payload["distributions_over_usable_channels"].get(key) or {}
            there = rd.get(key) or {}
            if not here.get("n") or not there.get("n"):
                continue
            lines.append(
                "| `%s` | %.4g | %.4g | %.4g | %.4g |" % (
                    key, here["median"], there["median"],
                    here["max"], there["max"])
            )
        lines.append("")
    warn = payload.get("substrate_shape_warning") or {}
    if warn.get("findings"):
        lines.extend(["", "## Substrate shape warning", "",
                      warn["why_this_is_reported_beside_the_verdict"], ""])
        for row in warn["findings"]:
            lines.append("- **%s**: %s.  %s" % (
                row["operator_family"], row["finding"], row["consequence"]))
        lines.extend(["", warn["what_it_blocks"], ""])
    lines.extend([
        "", "## What this does not do", "",
        "- %s" % payload["nothing_was_selected"],
        "- %s" % payload["structural_question_for_s2"][0],
        "",
        "## Cost", "",
        "- LLM calls: 0.  Consumer retrains: 0.  Outcome opened: none.",
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(run())
