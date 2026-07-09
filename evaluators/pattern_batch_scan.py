"""Pattern-Batch homogeneity scan for Stage2 readiness records.

This module is descriptive: it measures whether a batch key groups records with
similar treatment response. It does not train a deployment policy or tune a gate.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..e32_policy import P_FEATS

JsonRecord = Mapping[str, Any]


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def oracle_action(record: JsonRecord) -> str:
    losses = record["L_test"]
    return str(min(losses, key=losses.get))


def legacy_cell_groups(records: Sequence[JsonRecord]) -> dict[str, str]:
    return {str(r["uid"]): str(r["cell"]) for r in records}


def p0_feature_matrix(records: Sequence[JsonRecord]) -> np.ndarray:
    rows = []
    for record in records:
        x_p = list(record.get("X_p") or [])
        if len(x_p) != len(P_FEATS):
            raise ValueError(f"record {record.get('uid')!r} has X_p length {len(x_p)}, expected {len(P_FEATS)}")
        rows.append([float(record["snr"]), float(record["miss_rate"]), *[float(v) for v in x_p]])
    return np.asarray(rows, dtype=float)


def p1b_feature_matrix(records: Sequence[JsonRecord], features_path: Path | str) -> np.ndarray:
    features = json.loads(Path(features_path).read_text(encoding="utf-8"))
    rows = []
    for record in records:
        uid = str(record["uid"])
        if uid not in features:
            raise KeyError(f"P1b feature file missing uid {uid!r}")
        item = features[uid]
        values = item.get("d") if isinstance(item, Mapping) else item
        rows.append([float(v) for v in values])
    return np.asarray(rows, dtype=float)


def kmeans_groups(records: Sequence[JsonRecord], features: np.ndarray, *, k: int, seed: int = 20260707) -> dict[str, str]:
    if len(records) == 0:
        return {}
    n_clusters = max(1, min(int(k), len(records)))
    z = np.asarray(features, dtype=float)
    mu = z.mean(axis=0)
    sd = z.std(axis=0)
    sd[sd < 1e-12] = 1.0
    z = (z - mu) / sd
    if n_clusters == 1:
        labels = np.zeros(len(records), dtype=int)
    else:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(z)
    return {str(record["uid"]): f"c{int(label)}" for record, label in zip(records, labels)}


def summarize_batch_groups(
    records: Sequence[JsonRecord],
    groups_by_uid: Mapping[str, str],
    *,
    actions: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not records:
        return {
            "n_records": 0,
            "n_batches": 0,
            "oracle_agreement": 0.0,
            "within_batch_response_var": 0.0,
            "family_purity": 0.0,
            "batch_rows": [],
        }
    action_order = list(actions or records[0]["L_test"].keys())
    batches: dict[str, list[JsonRecord]] = defaultdict(list)
    for record in records:
        batches[str(groups_by_uid[str(record["uid"])])].append(record)

    rows: list[dict[str, Any]] = []
    total = len(records)
    oracle_majority_total = 0
    family_majority_total = 0
    response_var_weighted = 0.0
    for batch_key, batch_records in sorted(batches.items()):
        n = len(batch_records)
        oracles = [oracle_action(r) for r in batch_records]
        oracle_label, oracle_count = Counter(oracles).most_common(1)[0]
        origins = [str(r.get("origin", "")) for r in batch_records]
        origin_label, origin_count = Counter(origins).most_common(1)[0]
        losses = np.asarray([[float(r["L_test"][a]) for a in action_order] for r in batch_records], dtype=float)
        response_var = float(losses.var(axis=0).mean()) if n > 1 else 0.0
        oracle_majority_total += oracle_count
        family_majority_total += origin_count
        response_var_weighted += response_var * n
        rows.append(
            {
                "batch_key": batch_key,
                "n_records": int(n),
                "oracle_majority_action": oracle_label,
                "oracle_agreement": oracle_count / n,
                "within_batch_response_var": response_var,
                "origin_majority": origin_label,
                "family_purity": origin_count / n,
            }
        )

    return {
        "n_records": int(total),
        "n_batches": int(len(batches)),
        "oracle_agreement": oracle_majority_total / total,
        "within_batch_response_var": response_var_weighted / total,
        "family_purity": family_majority_total / total,
        "batch_rows": rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def render_table(report: Mapping[str, Any]) -> str:
    lines = [
        "# Pattern-Batch Confirmatory Scan",
        "",
        "Descriptive only: higher oracle agreement and lower response variance mean the batch key is closer to a processing-response-similar region.",
        "",
        "| batch_key | batches | oracle agreement | response variance | family purity |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in report["summaries"].items():
        lines.append(
            f"| {name} | {summary['n_batches']} | {summary['oracle_agreement']:.3f} | "
            f"{summary['within_batch_response_var']:.4f} | {summary['family_purity']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def build_pattern_batch_report(
    records_path: Path | str,
    out_dir: Path | str,
    *,
    k: int = 8,
    p1b_features_path: Path | str | None = None,
    seed: int = 20260707,
) -> dict[str, Any]:
    records = load_jsonl(records_path)
    summaries: dict[str, dict[str, Any]] = {}
    summaries["legacy_cell"] = summarize_batch_groups(records, legacy_cell_groups(records))
    summaries["P0_kmeans"] = summarize_batch_groups(
        records,
        kmeans_groups(records, p0_feature_matrix(records), k=k, seed=seed),
    )
    if p1b_features_path is not None and Path(p1b_features_path).exists():
        summaries["P1b_kmeans"] = summarize_batch_groups(
            records,
            kmeans_groups(records, p1b_feature_matrix(records, p1b_features_path), k=k, seed=seed),
        )

    report = {
        "config": {
            "records_path": str(records_path),
            "k": int(k),
            "seed": int(seed),
            "p1b_features_path": str(p1b_features_path) if p1b_features_path is not None else None,
        },
        "summaries": summaries,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batch_rows = []
    for summary_name, summary in summaries.items():
        for row in summary["batch_rows"]:
            batch_rows.append({"summary": summary_name, **row})
    _write_csv(
        out / "batch_rows.csv",
        batch_rows,
        [
            "summary",
            "batch_key",
            "n_records",
            "oracle_majority_action",
            "oracle_agreement",
            "within_batch_response_var",
            "origin_majority",
            "family_purity",
        ],
    )
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    (out / "table.md").write_text(render_table(report), encoding="utf-8")
    return report
