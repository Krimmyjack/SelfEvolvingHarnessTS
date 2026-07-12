# Benchmark v0 Implementation Plan Amendment 2

> Completes the production body shown in Amendment 1; this is executable design,
> not a placeholder.

Use this implementation shape for `materialize_corruptions` in Task 4:

```python
def materialize_corruptions(
    values_by_uid: Mapping[str, np.ndarray],
    content_sha_by_uid: Mapping[str, str],
    scenario: str,
    dose: float,
    replicate_idx: int,
    benchmark_version: str,
) -> dict[str, np.ndarray]:
    if set(values_by_uid) != set(content_sha_by_uid):
        raise ValueError("values/content-sha uid sets differ")
    result: dict[str, np.ndarray] = {}
    for uid, values in values_by_uid.items():
        seed = corruption_seed(
            benchmark_version,
            content_sha_by_uid[uid],
            scenario,
            dose,
            replicate_idx,
        )
        result[uid] = apply_corruption(
            np.asarray(values, dtype=np.float64),
            scenario=scenario,
            dose=dose,
            seed=seed,
        )
    return result
```

Iteration order cannot affect any output because each uid obtains its own seed
from content identity and frozen scenario coordinates.
