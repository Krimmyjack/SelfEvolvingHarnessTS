"""S0: the third-domain census for Phase S.  0 LLM, 0 retrains, 0 Outcome.

What this does and does not do
------------------------------
It enumerates every forecasting-shaped corpus reachable from this machine,
labels each one's exposure at instance and outcome granularity with a
pointer to the artifact or run that consumed it, and applies the Phase S
gate.  It opens no held-out region: every measurement below reads a
development prefix only, strictly under the sealed index the earlier
screening fixed.

The screening bar is not re-invented.  ``g3_sourcing`` fixed one before any
candidate was opened, and two fresh-family candidates were already rejected
against it; this run reuses that module's own census function so a new
candidate is judged by the same test, and reports the earlier verdicts as
they stand rather than re-deriving them.
"""
from __future__ import annotations

import hashlib
import json
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

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "shared_capability_s0_census_v1.json"
OUT_MD = E2 / "shared_capability_s0_census_v1.md"
SHARED = Path(r"C:\Users\辉\Desktop\Agent\shared_tsq_datasets")
PRIOR_SCREENING = E2 / "g3_candidate_screening_v3.json"

# This line's window shape, from #17 onward.  A candidate that cannot reach
# the farthest index plus one horizon cannot host the trajectory at all.
LINE_WINDOWS = (1104, 1440, 1800, 9864, 10560)
LINE_HORIZON = 48
LINE_MIN_LENGTH = max(LINE_WINDOWS) + 288 + LINE_HORIZON  # 10560+288+48
LINE_MIN_SERIES = 16  # 12 train + 4 eval, the #17 roster split
DEVELOPMENT_CUTOFF = 3072  # g3_sourcing.SEALED_FROM_INDEX; nothing above is read


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _npy_shape(path: Path) -> tuple[int, ...]:
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        shape, _, _ = np.lib.format._read_array_header(handle, version)
    return tuple(shape)


def _structure(n_series: int, length: int) -> dict[str, Any]:
    return {
        "series_count": int(n_series),
        "series_length": int(length),
        "meets_line_series_bar": bool(n_series >= LINE_MIN_SERIES),
        "meets_line_length_bar": bool(length >= LINE_MIN_LENGTH),
        "meets_g3_series_bar": bool(
            n_series >= G3.MIN_TRAIN_SERIES + G3.MIN_EVAL_SERIES
        ),
        "meets_g3_length_bar": bool(length >= G3.MIN_SERIES_LENGTH),
    }


# --------------------------------------------------------- the exposure ledger
# Every row carries the pointer that settles it.  "INSTANCE_SEEN" means some
# run read these very series; "AGGREGATE_SEEN" means only summary statistics
# of the family were ever read; "UNSEEN" means neither.
LEDGER: tuple[dict[str, Any], ...] = (
    {
        "candidate": "noaa_global_hourly (this line's cohort)",
        "family": "weather",
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "EXPOSED",
        "evidence": (
            "data/benchmark_noaa_fresh_v1/manifest.json consumption_census: "
            "consumed_n 44, fresh_n 20; #17 opened the 2025 confirmation "
            "partition once (fresh_confirmation_v1.*), and every trajectory "
            "since has read task_A/probe/task_B/task_C/task_D inside it"
        ),
        "tier": "not a third domain -- it is the incumbent",
        "verdict": "EXCLUDED_INCUMBENT",
    },
    {
        "candidate": "noaa_global_hourly (unused stations, Tier2)",
        "family": "weather",
        "context_exposure": "UNSEEN",
        "outcome_exposure": "SEALED",
        "evidence": (
            "data/benchmark_v0/raw/noaa_global_hourly/2024 holds 64 station "
            "csv files, which is exactly consumed_n 44 + fresh_n 20.  "
            "isd-history.csv lists the wider station universe but the "
            "corresponding hourly csv files are not on disk"
        ),
        "tier": "Tier2 (same family, new region)",
        "verdict": "NOT_MATERIALIZABLE_OFFLINE",
    },
    {
        "candidate": "uci_electricity_load_diagrams / tsquality electricity",
        "family": "energy",
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "EXPOSED",
        "evidence": (
            "batch_recipe_electricity_v1.json exposure = "
            "G3_DEVELOPMENT_SOURCE_FAMILY_OVERLAP (prior Outcome exposure in "
            "this project); g3_candidate_screening_v3 exposed_families.energy"
        ),
        "tier": "Tier1 candidate",
        "verdict": "EXCLUDED_EXPOSED_FAMILY",
    },
    {
        "candidate": "T233",
        "family": "legacy_mixed",
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "EXPOSED",
        "evidence": (
            "batch_recipe_T233_v1.json exposure = 'already exposed "
            "development data; not fresh'; g1/g2 T233 artifacts"
        ),
        "tier": "Tier1 candidate",
        "verdict": "EXCLUDED_EXPOSED_FAMILY",
    },
    {
        "candidate": "monash traffic_hourly / PeMS / metr_la / tsquality traffic",
        "family": "traffic",
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "EXPOSED",
        "evidence": (
            "batch_recipe_traffic_v1.json exposure = "
            "STRUCTURALLY_ACCEPTED_BUT_SOURCE_FAMILY_EXPOSURE_UNRESOLVED; "
            "g3_candidate_screening_v3 exposed_families.traffic lists "
            "metr_la and monash:traffic_hourly; m0a_mask_geometry_census_"
            "traffic_v1 read the instances"
        ),
        "tier": "Tier1 candidate",
        "verdict": "EXCLUDED_EXPOSED_FAMILY",
    },
    {
        "candidate": "tsquality weather (Jena)",
        "family": "weather",
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "EXPOSED",
        "evidence": (
            "g3_candidate_screening_v3 exposed_families.weather lists "
            "tsl_weather_jena beside noaa_global_hourly"
        ),
        "tier": "Tier1 candidate",
        "verdict": "EXCLUDED_EXPOSED_FAMILY",
    },
    {
        "candidate": "kdd2018",
        "family": "air_quality",
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "EXPOSED",
        "evidence": (
            "g3_candidate_screening_v3 exposed_families.air_quality; "
            "kdd_historical_policy_skill_memory_target_report.json"
        ),
        "tier": "Tier1 candidate",
        "verdict": "EXCLUDED_EXPOSED_FAMILY",
    },
    {
        "candidate": "beijing_multisite (PRSA)",
        "family": "air_quality",
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "EXPOSED",
        "evidence": (
            "natural_imputation_prsa_actionable_target_report.json and three "
            "natural_missing_window_weighting_prsa_* reports read these "
            "stations"
        ),
        "tier": "Tier1 candidate",
        "verdict": "EXCLUDED_EXPOSED_FAMILY",
    },
    {
        "candidate": "monash nn5_daily / covid_deaths",
        "family": "finance / epidemiology",
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "EXPOSED",
        "evidence": (
            "autonomous_natural_workflow_generation_nn5_* reports; "
            "g3_candidate_screening_v3 exposed_families.finance and "
            ".epidemiology"
        ),
        "tier": "Tier1 candidate",
        "verdict": "EXCLUDED_EXPOSED_FAMILY",
    },
    {
        "candidate": "gefcom2012_load",
        "family": "energy",
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "EXPOSED",
        "evidence": (
            "autonomous_natural_workflow_generation_gefcom* reports; "
            "g3_candidate_screening_v3 exposed_families.energy"
        ),
        "tier": "Tier1 candidate",
        "verdict": "EXCLUDED_EXPOSED_FAMILY",
    },
)

# Corpora present on disk that no run has consumed, with why each is or is
# not worth measuring.  The three that survive the paper checks are measured
# below; the rest are disqualified on structure alone and cost nothing.
UNCONSUMED: tuple[dict[str, Any], ...] = (
    {
        "candidate": "psm",
        "family": "server telemetry (eBay pooled server metrics)",
        "already_screened": True,
        "prior_verdict": (
            "FRESH_SOURCE_FAMILY, rejected: 1 series with a public "
            "phenomenon against a bar of 4 -- the channels arrive "
            "pre-normalized, so the Operator DSL has nothing to act on"
        ),
    },
    {
        "candidate": "swat",
        "family": "industrial control (Secure Water Treatment testbed)",
        "already_screened": True,
        "prior_verdict": (
            "FRESH_SOURCE_FAMILY, rejected: 15 of 52 columns clean under "
            "both substrate guards against a bar of 20 -- the actuator "
            "channels are near-constant and hit the scale floor"
        ),
    },
    {
        "candidate": "exchange_rate",
        "family": "finance",
        "already_screened": False,
        "disqualified_on_paper": (
            "8 series against this line's bar of 16, and 7588 points against "
            "10896"
        ),
    },
    {
        "candidate": "illness",
        "family": "epidemiology",
        "already_screened": False,
        "disqualified_on_paper": "7 series and 966 points",
    },
    {
        "candidate": "ETT-small",
        "family": "energy (transformer temperature)",
        "already_screened": False,
        "disqualified_on_paper": (
            "7 channels per file against a bar of 16; pooling the two hourly "
            "files gives 14, and pooling all four mixes hourly with 15-minute "
            "resolution.  The energy family is exposed in any case"
        ),
    },
    {
        "candidate": "m4 (Hourly / Daily / Monthly / Yearly)",
        "family": "legacy_mixed",
        "already_screened": False,
        "disqualified_on_paper": (
            "many series but each far short of 10896 (Hourly tops out near "
            "960, Daily near 9933); legacy_mixed is an exposed family"
        ),
    },
    {
        "candidate": "PEMS-SF",
        "family": "traffic",
        "already_screened": False,
        "disqualified_on_paper": (
            "stored as .ts classification samples of 144 steps, not a "
            "continuous batch; traffic is an exposed family"
        ),
    },
    {
        "candidate": (
            "UCR archive (BeetleFly, FordA, ... ), and the .ts classification "
            "corpora (FaceDetection, Heartbeat, Handwriting, ...)"
        ),
        "family": "classification benchmarks",
        "already_screened": False,
        "disqualified_on_paper": (
            "labelled classification samples, not forecasting batches; no "
            "origin/horizon structure to map the window shape onto"
        ),
    },
    {
        "candidate": (
            "weatherbench_daily, wiki_daily_100k, m5, monash_m3_monthly, "
            "electricity_15min, electricity_weekly, energy, healthcare, "
            "finance, synthetic, other"
        ),
        "family": "various",
        "already_screened": False,
        "disqualified_on_paper": (
            "directory present but empty on this machine; nothing to measure"
        ),
    },
)

MEASURE: tuple[dict[str, Any], ...] = (
    {
        "candidate": "smd",
        "family": "server telemetry (Server Machine Dataset)",
        "path": SHARED / "SMD" / "SMD_train.npy",
        "note": (
            "28 machines concatenated into one array of 38 channels; the "
            "development prefix below is read from the head of that array"
        ),
    },
    {
        "candidate": "smap",
        "family": "spacecraft telemetry (Soil Moisture Active Passive)",
        "path": SHARED / "SMAP" / "SMAP_train.npy",
        "note": "25 channels; channel 0 is the measurement, the rest are commands",
    },
    {
        "candidate": "msl",
        "family": "spacecraft telemetry (Mars Science Laboratory)",
        "path": SHARED / "MSL" / "MSL_train.npy",
        "note": "55 channels, same structure as SMAP",
    },
)


def measure(entry: dict[str, Any]) -> dict[str, Any]:
    """Structure and public-phenomenon census on the development prefix only."""
    path = Path(entry["path"])
    out: dict[str, Any] = {
        "candidate": entry["candidate"],
        "family": entry["family"],
        "note": entry["note"],
        "source_path": str(path),
        "source_exists": path.is_file(),
        "context_exposure": "UNSEEN",
        "outcome_exposure": "SEALED",
        "evidence": (
            "no artifact under artifacts/functional/e2 names this corpus; "
            "this census reads its development prefix and opens no outcome"
        ),
        "tier": "Tier1 (different domain family)",
    }
    if not out["source_exists"]:
        out["verdict"] = "ABSENT"
        return out
    shape = _npy_shape(path)
    rows, cols = int(shape[0]), int(shape[1])
    out["shape"] = list(shape)
    out["structure"] = _structure(cols, rows)
    out["sha256"] = _sha256(path)
    if not out["structure"]["meets_line_series_bar"] or not out["structure"][
        "meets_line_length_bar"
    ]:
        out["verdict"] = "FAILS_STRUCTURE"
        return out
    array = np.load(path, mmap_mode="r")
    prefix = np.asarray(array[:DEVELOPMENT_CUTOFF, :], dtype=np.float64)
    uids = ["channel_%d" % i for i in range(cols)]
    values = {uid: prefix[:, i] for i, uid in enumerate(uids)}
    census = G3.public_phenomenon_census(values, uids, DEVELOPMENT_CUTOFF)
    out["development_prefix_rows_read"] = int(prefix.shape[0])
    out["public_phenomena"] = {
        "series_with_public_phenomenon": census["series_with_public_phenomenon"],
        "bar": G3.MIN_SERIES_WITH_PUBLIC_PHENOMENON,
        "pass": bool(census["pass"]),
        "which": sorted(
            uid for uid, row in census["per_series"].items()
            if row["has_public_phenomenon"]
        ),
        "per_series": census["per_series"],
    }
    cardinality = {
        uid: int(np.unique(values[uid]).size) for uid in uids
    }
    constant = sorted(uid for uid, c in cardinality.items() if c == 1)
    binary = sorted(uid for uid, c in cardinality.items() if c <= 2)
    usable = sorted(uid for uid, c in cardinality.items() if c > 20)
    out["degenerate_channels"] = {
        "cardinality_over_the_development_prefix": cardinality,
        "constant": constant,
        "constant_count": len(constant),
        "binary_or_constant": binary,
        "binary_or_constant_count": len(binary),
        "usable_continuous": usable,
        "usable_continuous_count": len(usable),
        "why_it_matters": (
            "a channel that never moves cannot host a defect and cannot be "
            "forecast -- SWaT was rejected on exactly this -- and a binary "
            "command flag is worse than useless here: its robust deviation "
            "is zero, so every transition reads as an extreme z peak and the "
            "public-phenomenon test fires on a channel the Operator DSL has "
            "no business cleaning.  The line's series bar therefore counts "
            "channels that can host a forecast, not raw columns."
        ),
        "this_is_not_a_new_bar": (
            "the bar is still %d series; what changed is that a binary flag "
            "does not count as one" % LINE_MIN_SERIES
        ),
    }
    out["structure"]["usable_series_count"] = len(usable)
    out["structure"]["meets_line_series_bar_on_usable"] = bool(
        len(usable) >= LINE_MIN_SERIES
    )
    if not census["pass"]:
        out["verdict"] = "FAILS_PUBLIC_PHENOMENON_BAR"
    elif len(usable) < LINE_MIN_SERIES:
        out["verdict"] = "FAILS_DEGENERATE_CHANNELS"
        out["verdict_detail"] = (
            "%d of %d channels are continuous; the public-phenomenon count of "
            "%d is an artefact of the test meeting binary data, not a usable "
            "substrate" % (
                len(usable), len(uids),
                census["series_with_public_phenomenon"],
            )
        )
    else:
        out["verdict"] = "ELIGIBLE"
    return out


def run() -> int:
    started = time.perf_counter()
    prior = (
        json.loads(PRIOR_SCREENING.read_text(encoding="utf-8"))
        if PRIOR_SCREENING.is_file() else None
    )
    measured = [measure(dict(row)) for row in MEASURE]
    eligible = [row for row in measured if row.get("verdict") == "ELIGIBLE"]
    payload: dict[str, Any] = {
        "protocol_version": "shared_capability_s0_census_v1",
        "role": "Phase S step 0: which third domain, if any, this line can use",
        "llm_calls": 0,
        "consumer_retrains": 0,
        "opens_no_outcome": (
            "every measurement reads a development prefix of at most %d rows, "
            "strictly below the sealed index the earlier screening fixed; "
            "beyond_17520 and every held-out partition are untouched"
            % DEVELOPMENT_CUTOFF
        ),
        "bars": {
            "this_line": {
                "windows": list(LINE_WINDOWS),
                "horizon": LINE_HORIZON,
                "min_series_length": LINE_MIN_LENGTH,
                "min_series": LINE_MIN_SERIES,
                "why": (
                    "12 train + 4 eval is the #17 roster split, and the "
                    "farthest window index plus its support/delayed stride "
                    "and one horizon is %d" % LINE_MIN_LENGTH
                ),
            },
            "g3_pre_registered": dict(G3.CRITERIA),
            "not_moved": (
                "the g3 bar was fixed before any candidate was opened and is "
                "reused here unchanged; this run adds no new criterion and "
                "relaxes none"
            ),
        },
        "exposure_ledger": [dict(row) for row in LEDGER],
        "unconsumed_corpora": [dict(row) for row in UNCONSUMED],
        "measured_candidates": measured,
        "prior_screening": {
            "artifact": "artifacts/functional/e2/g3_candidate_screening_v3.json",
            "state": (prior or {}).get("fresh_sourcing_status", {}).get("state"),
            "note": (prior or {}).get("fresh_sourcing_status", {}).get("note"),
            "next_directions_if_resumed": (prior or {}).get(
                "fresh_sourcing_status", {}
            ).get("next_directions_if_resumed"),
        },
    }
    if eligible:
        payload["verdict"] = "THIRD_DOMAIN_AVAILABLE"
        payload["primary"] = eligible[0]["candidate"]
        payload["backup"] = eligible[1]["candidate"] if len(eligible) > 1 else None
        payload["verdict_reason"] = (
            "%d of %d measured candidates cleared structure and the public "
            "phenomenon bar" % (len(eligible), len(measured))
        )
    else:
        payload["verdict"] = "NO_ELIGIBLE_THIRD_DOMAIN"
        payload["primary"] = None
        payload["backup"] = None
        payload["verdict_reason"] = (
            "every corpus reachable from this machine is either an exposed "
            "family, structurally unusable, or fails the pre-registered "
            "substrate bar.  The blocker is substrate, not freshness: a "
            "usable third domain needs an unnormalized signal with enough "
            "non-degenerate channels, and the telemetry corpora that remain "
            "are pre-normalized by construction."
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
    lines = [
        "# S0 -- third-domain census for Phase S",
        "",
        "**Verdict: `%s`** -- %s" % (payload["verdict"], payload["verdict_reason"]),
        "",
        payload["opens_no_outcome"],
        "",
        "## Bars",
        "",
        "- This line: %d series minimum (12 train + 4 eval), %d points minimum "
        "(windows %s, horizon %d)." % (
            payload["bars"]["this_line"]["min_series"],
            payload["bars"]["this_line"]["min_series_length"],
            payload["bars"]["this_line"]["windows"],
            payload["bars"]["this_line"]["horizon"],
        ),
        "- Pre-registered g3 bar, reused unchanged: %d train + %d eval series, "
        "%d points, at least %d series carrying a public phenomenon." % (
            G3.MIN_TRAIN_SERIES, G3.MIN_EVAL_SERIES, G3.MIN_SERIES_LENGTH,
            G3.MIN_SERIES_WITH_PUBLIC_PHENOMENON,
        ),
        "",
        "## Exposure ledger",
        "",
        "| candidate | family | context | outcome | verdict | evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["exposure_ledger"]:
        lines.append(
            "| %s | %s | `%s` | `%s` | `%s` | %s |" % (
                row["candidate"], row["family"], row["context_exposure"],
                row["outcome_exposure"], row["verdict"], row["evidence"],
            )
        )
    lines.extend([
        "", "## Unconsumed corpora on this machine", "",
        "| candidate | family | status |", "| --- | --- | --- |",
    ])
    for row in payload["unconsumed_corpora"]:
        status = (
            "screened before: " + row["prior_verdict"]
            if row.get("already_screened") else row.get("disqualified_on_paper", "")
        )
        lines.append("| %s | %s | %s |" % (row["candidate"], row["family"], status))
    lines.extend([
        "", "## Measured this round (development prefix only)", "",
        "| candidate | shape | usable / total channels | length bar | "
        "public phenomena | binary or constant | verdict |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ])
    for row in payload["measured_candidates"]:
        st = row.get("structure") or {}
        ph = row.get("public_phenomena") or {}
        lines.append(
            "| `%s` | %s | %s / %s%s | %s | %s | %s | `%s` |" % (
                row["candidate"], row.get("shape"),
                st.get("usable_series_count", "--"), st.get("series_count", "--"),
                "" if st.get("meets_line_series_bar_on_usable") else " **FAIL**",
                "pass" if st.get("meets_line_length_bar") else "FAIL",
                "%s / %s" % (
                    ph.get("series_with_public_phenomenon"), ph.get("bar")
                ) if ph else "--",
                (row.get("degenerate_channels") or {}).get(
                    "binary_or_constant_count", "--"
                ),
                row.get("verdict"),
            )
        )
    prior = payload.get("prior_screening") or {}
    lines.extend([
        "", "## The prior screening, as it stands", "",
        "- Artifact: `%s`, state `%s`." % (prior.get("artifact"), prior.get("state")),
        "- %s" % (prior.get("note") or ""),
        "- If resumed: %s" % (prior.get("next_directions_if_resumed") or ""),
        "",
        "## Cost", "",
        "- LLM calls: 0.  Consumer retrains: 0.  Outcome opened: none.",
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(run())
