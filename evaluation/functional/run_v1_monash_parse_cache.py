"""Monash Weather .tsf 解析 + numpy 缓存（feasibility 前置）。

weather_dataset.tsf（212MB，3,010 日序列）→ data/monash_weather_v1/
series_cache.npz（{series_name: float64 values} + meta）。幂等：缓存存在
即跳过解析（避免每次重读 212MB）。

用法：
  python evaluation/functional/run_v1_monash_parse_cache.py [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TSF = PROJECT_ROOT / "data/monash_weather_v1/raw/weather_dataset.tsf"
CACHE = PROJECT_ROOT / "data/monash_weather_v1/series_cache.npz"


def parse_tsf(path: Path, limit: int | None = None) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    in_data = False
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not in_data:
                if line.startswith("@data"):
                    in_data = True
                continue
            if not line.strip():
                continue
            name, _stype, values = line.split(":", 2)
            vals = np.asarray([float(v) for v in values.split(",") if v],
                              dtype=np.float64)
            series[name] = vals
            count += 1
            if limit is not None and count >= limit:
                break
    return series


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if CACHE.exists() and args.limit is None:
        print(f"cache exists: {CACHE}")
        return 0
    print(f"parsing {TSF}...", flush=True)
    series = parse_tsf(TSF, limit=args.limit)
    names = sorted(series)
    values = np.asarray([series[n] for n in names], dtype=object)
    lengths = np.asarray([len(v) for v in values], dtype=np.int64)
    np.savez_compressed(
        CACHE,
        names=np.asarray(names, dtype=object),
        values=values,
        lengths=lengths,
    )
    print(f"parsed {len(names)} series -> {CACHE}")
    print(f"length min={lengths.min()} median={int(np.median(lengths))} "
          f"max={lengths.max()} total_values={int(lengths.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
