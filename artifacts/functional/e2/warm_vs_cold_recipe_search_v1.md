# warm vs cold budgeted recipe search v1

**Overall verdict: `WARM_NO_BENEFIT`**

Two arms search for a batch processing plan under the same evaluation budget. The only difference between them is whether they can see what other batches already produced. This is the A5-vs-A3 question at the recipe layer.

**Engineering effect measurement, not authorization evidence.** no Skill is written, no TRY right is granted, no Episode is promoted, no Fast or Slow path is entered, no snapshot pointer moves.

## 0. The instrument and the budget

| cost class | charged to the arm? |
| --- | --- |
| full-batch Support evaluation, one per shortlisted program | **yes, at most 4** |
| greedy exclusion-mask round on the best shortlisted program | no |
| the mask round's internal per-series evaluations | no (instrument fixed cost) |
| identity baseline | no (every gain is measured against it) |
| delayed-gate evaluation of the adopted plan | no (both arms pay it, revealed only after the plan is fixed) |

The menu holds 7 programs and the budget is 4, so a full scan does not fit and both arms must choose. Everything measurable is the recipe module's `_evaluate_variant` / `_evaluate_assignment` / `_gain_rows` on the same windows and Consumer variant.

## 1. Arms

- **cold**: no other-batch record is shown
- **warm**: the 18 verified records minus every row measured on the target's own cohort (leave-one-cohort-out)
- everything else: same public observation table, same menu, same budget, same instrument, same schemas, same wording; the prompt differs only in the prior_batch_experience section

The corpus is 18 records from `artifacts/functional/e2/batch_recipe_windows_v1.json`, with window-1 menu scans from `artifacts/functional/e2/batch_recipe_v2_all_cells_v1.json`.

## 2. Targets, quoted verbatim

| target | cell | window | support origins | delayed origins | full-search reference plan | reference delayed |
| --- | --- | --- | --- | --- | --- | ---: |
| A | T233 x pooled | W2 | [3360, 3408, 3456] | [3504, 3552, 3600] | `winsorize` minus nothing | +0.270423 |
| B | traffic x per_channel | W2 | [1584, 1848] | [2280] | `outlier_iqr` minus 8 | +0.439241 |

Origins are quoted from `artifacts/functional/e2/batch_recipe_windows_v1.json`; none was newly chosen here.

## 3. The four-row comparison

| target | arm | shortlist | mask asked | evals | adopted plan | support | delayed | capture | harmed (s/d) | LLM |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| A | cold | `repair_level_shift`, `hampel_filter`, `outlier_mad`, `denoise_median` | True | 4 | `outlier_mad` minus nothing | +0.142667 | +0.099344 | 0.367 | 2 / 3 | 3 |
| A | warm | `outlier_iqr`, `outlier_mad`, `winsorize`, `repair_level_shift` | True | 4 | `winsorize` minus T234 | +0.270433 | +0.260068 | 0.962 | 0 / 1 | 2 |
| B | cold | `repair_level_shift`, `hampel_filter`, `outlier_mad`, `denoise_median` | True | 4 | `outlier_mad` minus nothing | +0.264477 | +0.382712 | 0.871 | 0 / 0 | 3 |
| B | warm | `outlier_iqr`, `outlier_mad`, `denoise_median`, `winsorize` | True | 4 | `winsorize` minus nothing | +0.301976 | +0.396081 | 0.902 | 0 / 0 | 3 |

`capture` is the adopted plan's delayed gain over the frozen full search's delayed gain on the same cell and window. The delayed column was never shown to either arm.

## 4. Per-target verdicts

Rules, fixed before the first call, first match wins:

1. WARM_HARMS_QUALITY: the warm arm's capture ratio is below the cold arm's
2. WARM_REDUCES_COST_AT_QUALITY: the warm arm used strictly fewer evaluations and its capture ratio is at least 0.90
3. WARM_NO_BENEFIT: anything else

| target | verdict | why |
| --- | --- | --- |
| A (T233 x pooled) | `WARM_NO_BENEFIT` | warm spent 4 evaluations against cold's 4 and captured 0.962 against cold's 0.367 |
| B (traffic x per_channel) | `WARM_NO_BENEFIT` | warm spent 4 evaluations against cold's 4 and captured 0.902 against cold's 0.871 |

Overall: **`WARM_NO_BENEFIT`**.

## 5. Episode by episode

### A_cold -- target A, cold arm

- experience rows visible: 0 (none)
- shortlist: `repair_level_shift`, `hampel_filter`, `outlier_mad`, `denoise_median`; mask search asked: True
- shortlist reason: The public features show repeated substantial level excursions across the training series, together with concentrated robust local deviations and no missingness. Shortlist level-shift repair first, then robust local outlier and median-based alternatives; request the free exclusion-mask round to localize harmful series effects.
- Support results: repair_level_shift +0.025516; hampel_filter -0.048228; outlier_mad +0.142667; denoise_median -0.430582
- mask search on `outlier_mad`: reverted nothing, Support +0.142667
- adopted: `outlier_mad` minus nothing
- adoption reason: outlier_mad has the strongest positive full-batch Support gain among the shortlisted programs (0.1427), with fewer harmed evaluation series than the other positively supported option. The measured mask-search result reverted no series, so no exclusions are adopted.
- matches the full-search reference plan: False
- retries: budgeted_shortlist=0, budgeted_adoption=1['MASK_NOT_MEASURED']

### A_warm -- target A, warm arm

- experience rows visible: 12 (electricity, traffic)
- shortlist: `outlier_iqr`, `outlier_mad`, `winsorize`, `repair_level_shift`; mask search asked: True
- shortlist reason: The public evidence shows repeated extreme local deviations alongside substantial level excursions, so robust outlier treatments are the safest first checks, with level-shift repair as a targeted alternative. The series-level evidence is heterogeneous, making a greedy exclusion-mask round useful for limiting harm in the pooled consumer.
- Support results: outlier_iqr +0.196288; outlier_mad +0.142667; winsorize +0.250575; repair_level_shift +0.025516
- mask search on `winsorize`: reverted T234, Support +0.270433
- adopted: `winsorize` minus T234
- adoption reason: Winsorize had the highest full-batch Support gain (0.2506) among the shortlisted programs and harmed no evaluation series. The mask search accepted it for the unfiltered set after reverting T234; T246 was rejected and is therefore not excluded.
- matches the full-search reference plan: False
- retries: budgeted_shortlist=0, budgeted_adoption=0
- losing programs visible in the menu scans: `denoise_median`, `hampel_filter`, `outlier_iqr`, `outlier_mad`, `repair_level_shift`, `smooth_ma`
- of those, losing everywhere they were measured: none
- shortlisted anyway: none; skipped: none

### B_cold -- target B, cold arm

- experience rows visible: 0 (none)
- shortlist: `repair_level_shift`, `hampel_filter`, `outlier_mad`, `denoise_median`; mask search asked: True
- shortlist reason: The public features show strong localized level excursions and extreme local robust deviations across many series, with no missingness. These programs target level shifts or isolated/local noise while covering distinct repair mechanisms; exclusion-mask search can address the heterogeneous strength and localization across series.
- Support results: repair_level_shift +0.019408; hampel_filter +0.161141; outlier_mad +0.264477; denoise_median +0.153296
- mask search on `outlier_mad`: reverted nothing, Support +0.264477
- adopted: `outlier_mad` minus nothing
- adoption reason: outlier_mad has the highest public full-batch Support gain among the shortlisted programs and harmed no evaluation series.
- matches the full-search reference plan: False
- retries: budgeted_shortlist=0, budgeted_adoption=1['MASK_NOT_MEASURED']

### B_warm -- target B, warm arm

- experience rows visible: 12 (T233, electricity)
- shortlist: `outlier_iqr`, `outlier_mad`, `denoise_median`, `winsorize`; mask search asked: True
- shortlist reason: The public features show no missingness but widespread localized extreme deviations and level excursions, so robust outlier handling and median denoising are the most directly supported mechanisms. A free exclusion-mask round is worthwhile because the anomaly evidence varies substantially across training series.
- Support results: outlier_iqr +0.288667; outlier_mad +0.264477; denoise_median +0.153296; winsorize +0.301976
- mask search on `winsorize`: reverted nothing, Support +0.301976
- adopted: `winsorize` minus nothing
- adoption reason: Adopt winsorize because it achieved the highest full-batch Support aggregate gain among the shortlisted programs (0.30198) with no harmed evaluation series.
- matches the full-search reference plan: False
- retries: budgeted_shortlist=0, budgeted_adoption=1['MASK_NOT_MEASURED']
- losing programs visible in the menu scans: `denoise_median`, `hampel_filter`, `outlier_iqr`, `outlier_mad`, `repair_level_shift`, `smooth_ma`
- of those, losing everywhere they were measured: `hampel_filter`
- shortlisted anyway: none; skipped: `hampel_filter`

## 6. Experience entries written

Written through the existing episode mechanism, `provenance="budgeted_search_engineering"`, `counts_as_unguided_exploration: false`, and **not fed back into either arm** -- the warm arm's corpus is the frozen 18-record set only.

| episode | cell | plan | support | delayed | relation | provenance |
| --- | --- | --- | ---: | ---: | --- | --- |
| `A_cold` | `batch:T233\|consumer:pooled` | `outlier_mad` minus nothing | +0.142667 | +0.099344 | POSITIVE | `budgeted_search_engineering` |
| `A_warm` | `batch:T233\|consumer:pooled` | `winsorize` minus T234 | +0.270433 | +0.260068 | POSITIVE | `budgeted_search_engineering` |
| `B_cold` | `batch:traffic\|consumer:per_channel` | `outlier_mad` minus nothing | +0.264477 | +0.382712 | POSITIVE | `budgeted_search_engineering` |
| `B_warm` | `batch:traffic\|consumer:per_channel` | `winsorize` minus nothing | +0.301976 | +0.396081 | POSITIVE | `budgeted_search_engineering` |

## 7. What this does not say

- It does not authorize anything and it does not promote any plan.
- Two targets and four episodes on one model is a mechanism reading, not a rate. A per-target verdict is a comparison of two single draws.
- The capture denominator is the frozen full search's own delayed gain, which was selected on that same delayed window. Capture near 1 means 'as good as the unbudgeted search got', not 'optimal'.
- The warm arm's advantage, if any, is bounded by what the corpus contains: 18 records from three cohorts, leave-one-cohort-out.
- Neither arm ever saw a delayed number while choosing.

## Provenance

- model: `gpt-5.6-luna` at `https://api.agicto.cn/v1`
- instrument: `run_batch_composition_headroom._evaluate_variant` / `_evaluate_assignment` / `_gain_rows`, imported and not modified
- windows and reference plans: quoted from `artifacts/functional/e2/batch_recipe_windows_v1.json`
- LLM calls: 11 of 40
- wall seconds: 148.7

