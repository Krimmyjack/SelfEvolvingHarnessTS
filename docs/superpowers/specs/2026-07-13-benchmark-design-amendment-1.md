# Benchmark Design Amendment 1

> Normative self-review correction to
> `2026-07-13-benchmark-data-metrics-pipeline-design.md` and
> `idea/Benchmark_v0_Forecast_Design_v3_Addendum_2026-07-13.md`.
> Only the two clauses below are replaced; all other approved design remains.

## 1. Repeated-measure folding indices

The complete order is:

```text
within uid x scenario x dose x corruption replicate: average 3 model seeds
-> within uid x scenario x dose: average 2 corruption replicates
-> within uid: equal-average all frozen scenario x dose values
-> cell: series-equal mean
-> regime: dataset macro mean
```

`scenario` and `dose` are separate axes. Bootstrap input must contain exactly
one paired-gain row per uid after both axes have been folded.

## 2. Final campaign roster completeness

The frozen Final roster contains every entry that will read Final-Query:

```text
all candidate methods
+ Raw / best-fixed / H_ref
+ oracle_transfer / oracle_insample
```

The campaign manifest freezes this complete roster, every applicable code SHA,
budgets, seeds, and execution order before `unseal`. No baseline or diagnostic
oracle may be appended after Final-Query has been opened.
