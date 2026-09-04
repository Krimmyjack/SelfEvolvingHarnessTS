# p4ac HEC-1 course supply scan

0 LLM / 0 Consumer fit / outcome_values_read 0. Inventory only.

## (1) Six blocks × origin, two calibers

| block | all usable | ≤3816 | origins ≤3816 |
| --- | ---: | ---: | --- |
| [0:40] | 9 | 7 | 1176, 1896, 2136, 2376, 2616, 2856, 3576 |
| [40:80] | 11 | 9 | 1176, 1416, 1656, 2136, 2376, 2616, 2856, 3576, 3816 |
| [80:120] | 7 | 7 | 1176, 1416, 1896, 2136, 2376, 2616, 2856 |
| [120:160] | 3 | 3 | 1176, 1416, 1656 |
| [160:200] | 7 | 7 | 1896, 2136, 2376, 2616, 2856, 3096, 3576 |
| [200:239] | 6 | 6 | 1176, 1896, 2136, 2376, 2616, 2856 |

## (2) [200:239] cut

[200:239] has 39 series so a 20/20 cell does not form. Canonical cut for this scan is A=20 / B=19 as the task book states. If a later freeze requires equal faces, the alternative is A=19 / B=19 with leftover ['T99'] excluded; this scan does not freeze either cut.

## (3) Phase S / T unit counts

- Phase S all / ≤3816: **13** / **13**
- Phase T all / ≤3816: **30** / **26**

## (4) Composition three-element check

- Repeat-family Jaccard empty=False n=37 median=0.65
- Heterogeneity unique-bins empty=False median=15.0
- Sparse units (n_z_peak_ge_3<5) empty=True n=0

## (5) Exposure intersection

held-out intersection empty: **True** (n=0). p4t verdict: ALL_PROPOSED_HELD_OUT_PAIRS_UNEXPOSED.

Per-window labels (+0/+48/+144/+240) live in the JSON `exposure_cross_check.per_window_labels`. `[80:120] × 2856 +144 = 3000` does not overlap any held-out origin.

This scan freezes nothing.

## (6) Deviations

- 22-d binned vector: observable numeric vocabulary in this checkout is 12 features; public card has 21 keys. Heterogeneity proxy uses the 12 numeric observables' frozen bins.
- Raw serving-context non-degeneracy: `evaluability()` does not check it; this audit adds a 0-fit `_linear_integrity` + `_center_scale` gate and fails usable when method == `scale_floor_fallback`.

## (7) Spec tensions

- `[200:239]` vs p4s `[200:240]` "no cohort forms": `[200:239]` has 39 series so a 20/20 cell does not form. Canonical cut for this scan is A=20 / B=19. Equal-face alternative is A=19 / B=19 with leftover `T99` excluded; this scan does not freeze either cut.

