# warm vs cold, rotated over all six cells (W3) v1

**Overall: warm wins 4, cold wins 1, ties 1, over 6 targets. Worst target `electricity_per_channel` at -0.005983.**

The earlier two-target run found the warm arm far ahead on quality at equal cost but had no label for it. This rotation runs the same budgeted search on all six cells, on a window that has never been a target, and reads it with a corrected criterion: the paired delayed difference decides, cost is reported beside it and never folded in.

**Engineering effect measurement, not authorization evidence.** no Skill is written, no TRY right is granted, no Episode is promoted, no Fast or Slow path is entered, no snapshot pointer moves.

The two W2 targets of `artifacts/functional/e2/warm_vs_cold_recipe_search_v1.json` are neither re-run nor re-labelled.

## 0. Pre-registered before the first call

- primary readout: per target, the paired delayed difference delta = warm delayed aggregate gain minus cold delayed aggregate gain, both measured by the same instrument on the same window
- labels, first match wins:
  - WARM_WINS_QUALITY: delta > +0.005
  - COLD_WINS_QUALITY: delta < -0.005
  - TIE: otherwise
- cost: charged evaluations and LLM calls are reported per arm and never enter the label. WARM_ALSO_CHEAPER is recorded as a separate flag when the warm arm used strictly fewer charged evaluations and its delayed gain is not below the cold arm's
- overall: the win / loss / tie count over the six targets, plus the worst target by paired delta, named explicitly. A pooled mean on its own is not an acceptable reading of this run
- budget: shortlist capped at 2, so 2 charged full-batch Support evaluations against a menu of 7; the mask round and the delayed gate are free, and the mask still runs on the highest-Support shortlisted program.
- experience isolation: the warm arm's corpus is the 18 frozen records only; every row this rotation produces is written to the artifact and to no episode's visible experience

Changes from the earlier run, and only these:

- targets: all six cells, all on window W3 (origins quoted from batch_recipe_windows_v1; W3 has never been a target)
- budget: shortlist capped at 2, so 2 charged full-batch Support evaluations
- criterion: the paired delayed difference decides the label; cost is reported separately and never folded in
- instrument hardening: the adoption prompt enumerates every measured plan, identically for both arms

## 1. Prompt parity, measured on this run

Per-field digests of the stage-one prompt body actually sent to each arm. All targets pass: **True**. Scope: the stage-one prompt body; the adoption stage differs by construction because each arm measured its own shortlist.

| target | fields compared | fields that differ | only the experience section |
| --- | ---: | --- | --- |
| electricity_pooled | 8 | `prior_batch_experience` | True |
| electricity_per_channel | 8 | `prior_batch_experience` | True |
| T233_pooled | 8 | `prior_batch_experience` | True |
| T233_per_channel | 8 | `prior_batch_experience` | True |
| traffic_pooled | 8 | `prior_batch_experience` | True |
| traffic_per_channel | 8 | `prior_batch_experience` | True |

Experience isolation: the corpus is 18 frozen rows, unchanged across the run (`True`), and **0** rows produced by this rotation were fed back into any arm.

## 2. The twelve-row comparison

| target | arm | shortlist | mask | evals | adopted plan | support | delayed | capture | harmed (s/d) | LLM |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| electricity_pooled | cold | `repair_level_shift`, `hampel_filter` | True | 2 | `hampel_filter` minus 3 | +0.009408 | -0.029759 | -1.848 | 3 / 5 | 2 |
| electricity_pooled | warm | `outlier_iqr`, `winsorize` | True | 2 | `winsorize` minus nothing | +0.057660 | +0.016103 | 1.000 | 2 / 3 | 2 |
| electricity_per_channel | cold | `repair_level_shift`, `hampel_filter` | True | 2 | `repair_level_shift` minus 0, 1, 10, 11, 3 | +0.036267 | +0.033249 | 0.330 | 0 / 2 | 2 |
| electricity_per_channel | warm | `outlier_iqr`, `outlier_mad` | True | 2 | `outlier_iqr` minus 10, 5 | +0.029493 | +0.027266 | 0.271 | 0 / 1 | 2 |
| T233_pooled | cold | `repair_level_shift`, `hampel_filter` | True | 2 | `hampel_filter` minus T241, T254 | +0.380355 | +0.053091 | 0.235 | 2 / 2 | 2 |
| T233_pooled | warm | `outlier_iqr`, `repair_level_shift` | True | 2 | `outlier_iqr` minus T241, T244 | +0.343865 | +0.223271 | 0.989 | 1 / 0 | 2 |
| T233_per_channel | cold | `repair_level_shift`, `hampel_filter` | True | 2 | `repair_level_shift` minus T236, T241, T254, T256 | +0.040564 | -0.009278 | n/a | 0 / 4 | 2 |
| T233_per_channel | warm | `repair_level_shift`, `outlier_iqr` | True | 2 | `repair_level_shift` minus T236, T241, T254, T256 | +0.040564 | -0.009278 | n/a | 0 / 4 | 2 |
| traffic_pooled | cold | `repair_level_shift`, `hampel_filter` | True | 2 | `repair_level_shift` minus nothing | +0.177647 | +0.007260 | 0.007 | 3 / 3 | 2 |
| traffic_pooled | warm | `outlier_iqr`, `winsorize` | True | 2 | `outlier_iqr` minus 3, 5, 7 | +0.949579 | +1.037101 | 1.000 | 0 / 0 | 2 |
| traffic_per_channel | cold | `repair_level_shift`, `hampel_filter` | True | 2 | `hampel_filter` minus 3, 8 | +0.171742 | +0.140359 | 0.363 | 0 / 0 | 2 |
| traffic_per_channel | warm | `outlier_iqr`, `denoise_median` | True | 2 | `outlier_iqr` minus 3 | +0.514752 | +0.386972 | 1.000 | 0 / 0 | 2 |

`capture` is against the frozen full search's own delayed gain on the same cell and W3, quoted from `batch_recipe_windows_v1`. Neither arm ever saw a delayed number while choosing.

## 3. Per-target labels

| target | cold delayed | warm delayed | paired delta | label | cold evals | warm evals | warm also cheaper |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| electricity_pooled | -0.029759 | +0.016103 | +0.045862 | `WARM_WINS_QUALITY` | 2 | 2 | False |
| electricity_per_channel | +0.033249 | +0.027266 | -0.005983 | `COLD_WINS_QUALITY` | 2 | 2 | False |
| T233_pooled | +0.053091 | +0.223271 | +0.170181 | `WARM_WINS_QUALITY` | 2 | 2 | False |
| T233_per_channel | -0.009278 | -0.009278 | +0.000000 | `TIE` | 2 | 2 | False |
| traffic_pooled | +0.007260 | +1.037101 | +1.029841 | `WARM_WINS_QUALITY` | 2 | 2 | False |
| traffic_per_channel | +0.140359 | +0.386972 | +0.246613 | `WARM_WINS_QUALITY` | 2 | 2 | False |

Counts: **warm 4 / cold 1 / tie 1** over 6 readable targets. Worst target: **`electricity_per_channel`** at -0.005983 (`COLD_WINS_QUALITY`). Targets where the warm arm was also cheaper: none.

## 4. Shortlist divergence

### cold arm

1 distinct shortlist(s) over 6 targets; identical across all targets: **True**; largest group: 6.

| shortlist | targets |
| --- | --- |
| `repair_level_shift`, `hampel_filter` | T233_per_channel, T233_pooled, electricity_per_channel, electricity_pooled, traffic_per_channel, traffic_pooled |

Varies within a cohort: {"T233": false, "electricity": false, "traffic": false}. Varies within a consumer variant: {"per_channel": false, "pooled": false}.

### warm arm

5 distinct shortlist(s) over 6 targets; identical across all targets: **False**; largest group: 2.

| shortlist | targets |
| --- | --- |
| `outlier_iqr`, `winsorize` | electricity_pooled, traffic_pooled |
| `outlier_iqr`, `outlier_mad` | electricity_per_channel |
| `outlier_iqr`, `repair_level_shift` | T233_pooled |
| `repair_level_shift`, `outlier_iqr` | T233_per_channel |
| `outlier_iqr`, `denoise_median` | traffic_per_channel |

Varies within a cohort: {"T233": true, "electricity": true, "traffic": true}. Varies within a consumer variant: {"per_channel": true, "pooled": true}.

### Cold against warm, per target

| target | cold shortlist | warm shortlist | identical | overlap |
| --- | --- | --- | --- | --- |
| T233_per_channel | `repair_level_shift`, `hampel_filter` | `repair_level_shift`, `outlier_iqr` | False | `repair_level_shift` |
| T233_pooled | `repair_level_shift`, `hampel_filter` | `outlier_iqr`, `repair_level_shift` | False | `repair_level_shift` |
| electricity_per_channel | `repair_level_shift`, `hampel_filter` | `outlier_iqr`, `outlier_mad` | False | none |
| electricity_pooled | `repair_level_shift`, `hampel_filter` | `outlier_iqr`, `winsorize` | False | none |
| traffic_per_channel | `repair_level_shift`, `hampel_filter` | `outlier_iqr`, `denoise_median` | False | none |
| traffic_pooled | `repair_level_shift`, `hampel_filter` | `outlier_iqr`, `winsorize` | False | none |

## 5. Negative-experience use, warm arm only

| target | losing somewhere | losing everywhere measured | shortlisted anyway | skipped |
| --- | --- | --- | --- | --- |
| electricity_pooled | `denoise_median`, `hampel_filter`, `outlier_mad`, `repair_level_shift`, `smooth_ma` | none | none | none |
| electricity_per_channel | `denoise_median`, `hampel_filter`, `outlier_mad`, `repair_level_shift`, `smooth_ma` | none | none | none |
| T233_pooled | `denoise_median`, `hampel_filter`, `outlier_iqr`, `outlier_mad`, `repair_level_shift`, `smooth_ma` | none | none | none |
| T233_per_channel | `denoise_median`, `hampel_filter`, `outlier_iqr`, `outlier_mad`, `repair_level_shift`, `smooth_ma` | none | none | none |
| traffic_pooled | `denoise_median`, `hampel_filter`, `outlier_iqr`, `outlier_mad`, `repair_level_shift`, `smooth_ma` | `hampel_filter` | none | `hampel_filter` |
| traffic_per_channel | `denoise_median`, `hampel_filter`, `outlier_iqr`, `outlier_mad`, `repair_level_shift`, `smooth_ma` | `hampel_filter` | none | `hampel_filter` |

## 6. Episode reasons

**electricity_pooled_cold** -- shortlist `repair_level_shift`, `hampel_filter` (mask True): The public features show strong, localized level excursions across several channels, including large estimated offsets, making repair_level_shift the most directly targeted program for the pooled model. Hampel filtering is a distinct alternative supported by the pronounced local robust deviations, and the free exclusion-mask round can test whether specific channels drive the shortlisted program's aggregate Support.

  adopted `hampel_filter` minus 3: Adopt the measured Hampel-filter plan with series 3 excluded because it was the only evaluated non-identity plan with positive aggregate Support gain after mask search (0.0094). The unmasked Hampel plan had negative gain, while the measured exclusion improved aggregate performance despite harm on three evaluation series.

  retries: budgeted_shortlist=0, budgeted_adoption=0

**electricity_pooled_warm** -- shortlist `outlier_iqr`, `winsorize` (mask True): The public observations show widespread, strong local deviations and level excursions without missingness, making robust outlier treatment the most directly supported mechanism. In prior pooled evaluations, outlier_iqr and winsorize were consistently beneficial, so they are the most justified two-program shortlist; request a free exclusion-mask round to localize potentially harmful series.

  adopted `winsorize` minus nothing: Adopt the measured winsorize plan because it achieved the higher full-batch Support aggregate gain (0.05765955525139644) versus outlier_iqr (0.014651801418955232). No series were reverted in the mask search, so excluded_series remains empty.

  retries: budgeted_shortlist=0, budgeted_adoption=0

**electricity_per_channel_cold** -- shortlist `repair_level_shift`, `hampel_filter` (mask True): The public features show strong localized level excursions across multiple series, with large offsets and sufficient post-shift support in several cases, making repair_level_shift the primary candidate. Hampel filtering is an effect-distinct alternative for the pronounced local robust-z peaks, and the free exclusion-mask round can identify whether either program is helped by removing problematic training series.

  adopted `repair_level_shift` minus 0, 1, 10, 11, 3: Adopt the measured mask-search plan because repair_level_shift produced a positive aggregate Support gain of 0.0362671488 with no harmed evaluation series after reverting the listed series. This is preferred over the unmasked repair and hampel_filter plans, which had negative aggregate gains.

  retries: budgeted_shortlist=0, budgeted_adoption=0

**electricity_per_channel_warm** -- shortlist `outlier_iqr`, `outlier_mad` (mask True): The public features show strong localized extreme deviations across many series, with no missingness and several substantial level excursions; robust outlier treatments are therefore the most directly supported candidates. A greedy mask-search round can limit harm from series whose deviations are not improved by the selected treatment.

  adopted `outlier_iqr` minus 10, 5: Adopt the measured outlier_iqr plan with the mask-search reverts for series 10 and 5. It achieved the highest measured full-batch Support gain (0.02949), with no harmed evaluation series.

  retries: budgeted_shortlist=0, budgeted_adoption=0

**T233_pooled_cold** -- shortlist `repair_level_shift`, `hampel_filter` (mask True): The public observations show repeated, localized level excursions with large estimated offsets, supporting repair_level_shift as the primary mechanism. Several series also have strong local robust-z peaks, so hampel_filter is an effect-distinct secondary candidate; the free exclusion-mask round can test whether individual training series drive the shortlisted program's aggregate Support.

  adopted `hampel_filter` minus T241, T254: Adopt the measured mask-search plan for hampel_filter because it achieved the highest observed Support gain (0.380354856... ) and the search explicitly reverted T241 and T254. No other exclusions were measured.

  retries: budgeted_shortlist=0, budgeted_adoption=0

**T233_pooled_warm** -- shortlist `outlier_iqr`, `repair_level_shift` (mask True): The public features show widespread, localized level excursions with strong robust-deviation peaks and no missingness, supporting a robust outlier repair and a level-shift repair. Request a free exclusion-mask round for the shortlisted program with the stronger measured Support result.

  adopted `outlier_iqr` minus T241, T244: Adopt the measured outlier_iqr plan with the reverted series T241 and T244, which achieved the highest observed full-batch Support gain among the measured plans. The public evidence shows localized extreme deviations and level excursions, while the mask round improved the aggregate gain over the unmasked plan.

  retries: budgeted_shortlist=0, budgeted_adoption=0

**T233_per_channel_cold** -- shortlist `repair_level_shift`, `hampel_filter` (mask True): The public observations show repeated, large level excursions with sufficient post-shift support in most series, making repair_level_shift the primary candidate. Several series also have pronounced local robust deviations, so hampel_filter is a distinct secondary candidate; the free mask-search round can test whether excluding problematic training series improves support.

  adopted `repair_level_shift` minus T236, T241, T254, T256: Adopt the measured mask-search plan because repair_level_shift achieved the highest observed Support gain (0.04056390893471986) with no harmed evaluation series after reverting these four series. This plan was directly measured and outperformed the unmasked repair and identity baseline.

  retries: budgeted_shortlist=0, budgeted_adoption=0

**T233_per_channel_warm** -- shortlist `repair_level_shift`, `outlier_iqr` (mask True): The public features show repeated level excursions with substantial estimated offsets, alongside elevated local robust deviations across the training series. Shortlist a level-shift repair first, with IQR-based outlier handling as an effect-distinct alternative; request the free exclusion-mask round to localize any harmful series.

  adopted `repair_level_shift` minus T236, T241, T254, T256: This is the strongest measured plan in public support and it harmed no evaluation series. The reverted series show clear level-shift and local-deviation signatures, which matches the chosen repair.

  retries: budgeted_shortlist=0, budgeted_adoption=0

**traffic_pooled_cold** -- shortlist `repair_level_shift`, `hampel_filter` (mask True): No missingness is present, and the strongest public signal is a sustained level offset with extreme robust-z peaks, so `repair_level_shift` is the main bet. `hampel_filter` is the backup for cases where the gain comes from localized spikes, and the free mask pass can prune series that dilute the best program.

  adopted `repair_level_shift` minus nothing: Adopt the measured repair_level_shift plan because it had the highest full-batch Support aggregate gain (0.17764737911086947) among the shortlisted programs. Its public evidence is consistent with strong level excursions and local deviations across the training series.

  retries: budgeted_shortlist=0, budgeted_adoption=0

**traffic_pooled_warm** -- shortlist `outlier_iqr`, `winsorize` (mask True): The public evidence shows strong localized robust deviations across the pooled training series, with no missingness and insufficient post-shift support for most series, favoring robust outlier treatment over level-shift repair. Both shortlisted operators have shown positive pooled support in related public experience, and a free greedy mask-search round can localize harmful series.

  adopted `outlier_iqr` minus 3, 5, 7: Adopt the measured outlier_iqr mask plan because it achieved the highest observed full-batch Support gain (0.9496) with no harmed evaluation series. The reverted series are supported by comparatively weaker public local-deviation evidence than the retained series.

  retries: budgeted_shortlist=0, budgeted_adoption=0

**traffic_per_channel_cold** -- shortlist `repair_level_shift`, `hampel_filter` (mask True): The public features show repeated strong level-excursion signals with localized affected regions, supporting repair_level_shift, and substantial local robust-z peaks without missingness, supporting hampel_filter. A free exclusion-mask round can identify whether either shortlisted program is helped by reverting specific training series.

  adopted `hampel_filter` minus 3, 8: Adopt the measured hampel_filter plan with the reverted series 3 and 8 excluded. It achieved the highest measured Support aggregate gain (0.17174222849667803) with no harmed evaluation series among the measured plans.

  retries: budgeted_shortlist=0, budgeted_adoption=0

**traffic_per_channel_warm** -- shortlist `outlier_iqr`, `denoise_median` (mask True): The public prefixes show strong, localized extreme-deviation signals across most channels, making robust outlier treatment the primary mechanism to test. Denoising is a distinct alternative for the observed short level excursions and period inconsistencies, and the free exclusion-mask round can limit harm from channels where repair is not supported.

  adopted `outlier_iqr` minus 3: Adopt the measured outlier_iqr plan with series 3 excluded: it achieved the highest measured full-batch Support gain (0.5147523618556838) with no harmed evaluation series. The exclusion was accepted by the measured mask-search round.

  retries: budgeted_shortlist=0, budgeted_adoption=0

## 7. Experience entries written

Provenance `budgeted_search_engineering`, `counts_as_unguided_exploration: false`, and **not fed back**: the warm arm's corpus is the 18 frozen records only.

| episode | cell | plan | support | delayed | relation |
| --- | --- | --- | ---: | ---: | --- |
| `electricity_pooled_cold` | `batch:electricity\|consumer:pooled` | `hampel_filter` minus 3 | +0.009408 | -0.029759 | CONFLICT |
| `electricity_pooled_warm` | `batch:electricity\|consumer:pooled` | `winsorize` minus nothing | +0.057660 | +0.016103 | POSITIVE |
| `electricity_per_channel_cold` | `batch:electricity\|consumer:per_channel` | `repair_level_shift` minus 0, 1, 10, 11, 3 | +0.036267 | +0.033249 | POSITIVE |
| `electricity_per_channel_warm` | `batch:electricity\|consumer:per_channel` | `outlier_iqr` minus 10, 5 | +0.029493 | +0.027266 | POSITIVE |
| `T233_pooled_cold` | `batch:T233\|consumer:pooled` | `hampel_filter` minus T241, T254 | +0.380355 | +0.053091 | POSITIVE |
| `T233_pooled_warm` | `batch:T233\|consumer:pooled` | `outlier_iqr` minus T241, T244 | +0.343865 | +0.223271 | POSITIVE |
| `T233_per_channel_cold` | `batch:T233\|consumer:per_channel` | `repair_level_shift` minus T236, T241, T254, T256 | +0.040564 | -0.009278 | CONFLICT |
| `T233_per_channel_warm` | `batch:T233\|consumer:per_channel` | `repair_level_shift` minus T236, T241, T254, T256 | +0.040564 | -0.009278 | CONFLICT |
| `traffic_pooled_cold` | `batch:traffic\|consumer:pooled` | `repair_level_shift` minus nothing | +0.177647 | +0.007260 | POSITIVE |
| `traffic_pooled_warm` | `batch:traffic\|consumer:pooled` | `outlier_iqr` minus 3, 5, 7 | +0.949579 | +1.037101 | POSITIVE |
| `traffic_per_channel_cold` | `batch:traffic\|consumer:per_channel` | `hampel_filter` minus 3, 8 | +0.171742 | +0.140359 | POSITIVE |
| `traffic_per_channel_warm` | `batch:traffic\|consumer:per_channel` | `outlier_iqr` minus 3 | +0.514752 | +0.386972 | POSITIVE |

## 8. What this does not say

- It does not authorize anything and it promotes no plan.
- Six targets, one window, one model, one draw per cell. A per-target label is a comparison of two single runs, not a rate, which is why the count and the worst target are reported rather than a mean.
- The capture denominator is the frozen full search's delayed gain, itself selected on that delayed window. Capture near 1 means 'as good as the unbudgeted search got', not 'optimal'.
- The warm arm's ceiling is what the corpus holds: 18 records from three cohorts, leave-one-cohort-out.
- The mask round still runs on the highest-Support shortlisted program. That frozen rule, not the experience, decides which program gets a mask at all.

## Provenance

- model: `gpt-5.6-luna` at `https://api.agicto.cn/v1`
- instrument, corpus, validators, observation table and Experience writer: imported from `run_e2_warm_vs_cold_recipe_search`, which is not modified
- windows and reference plans: quoted from `artifacts/functional/e2/batch_recipe_windows_v1.json`
- LLM calls: 24 of 60
- wall seconds: 513.5

