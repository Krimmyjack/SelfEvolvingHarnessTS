# Benchmark v0 data card

Registry rows: 1659; eligible: 1608.

Frozen role counts: `{'dev_query': 301, 'final_query': 515, 'support_a': 469, 'support_b': 317, 'u': 6}`.

NOAA Global Hourly is the frozen weather U pool. Natural NaNs are preserved; hourly bins with partial or absent observations remain missing.

Init Harness (the Support-A-only pre-benchmark exposure subset) is frozen in
`init_harness_manifest.json`: 136 series = 80 `legacy_core` (admitted legacy
Monash bundle) + 56 `probe_consumed_extension` (traffic_hourly series P6's own
U-admission probe recorded as consumed). The remaining 333 `support_a` series
are `certified_virgin` and form the Fresh Support-A harness-update/selection
pool, not Init Harness. See `freeze_init_harness.py` for provenance and
regeneration.
