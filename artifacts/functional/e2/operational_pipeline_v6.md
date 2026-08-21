# Banked chain, one pinned Slow backend at a time

**Overall: `BANKED_CHAIN_CLOSES_IN_K_MODELS`** -- 2 of 2 pinned backends carried links 6 to 9 end to end: ['gpt-5.6-sol', 'gpt-5.6-luna']

a closing verdict is suffixed with the model that closed it.  This is a chain-liveness reading on that backend, not an Opus reading, and not a reading about the method.

## Per model

| model | verdict | valid samples | protocol-failed | LLM |
| --- | --- | ---: | ---: | ---: |
| `gpt-5.6-sol` | `BANKED_CHAIN_CLOSES_ON_GPT_5_6_SOL` | 1 | 0 | 1 |
| `gpt-5.6-luna` | `BANKED_CHAIN_CLOSES_ON_GPT_5_6_LUNA` | 1 | 0 | 1 |

### `gpt-5.6-sol` -- `BANKED_CHAIN_CLOSES_ON_GPT_5_6_SOL`

selector -> RISK_GAP at task_A -> one Slow proposal -> compiler accepted -> the banked adoption is contained to identity, ['99999923908'] no longer crosses -0.005, and every unrelated banked episode is unchanged
- draw 1: PROPOSAL

Guard `veto-delayed-single-series-harm`: min_per_series_gain `lt` -0.005000 -> VETO_AND_FALL_BACK on the delayed window, applies to every_adoption.

> Reject an adopted program when any evaluation series loses beyond the declared harm line, even if aggregate delayed gain is positive.

| step | plan before | plan after | delayed before | delayed after | harmed before | harmed after |
| --- | --- | --- | ---: | ---: | --- | --- |
| `task_A` | `outlier_iqr` | `identity` | +0.066941 | +0.000000 | 99999923908 | none |
| `task_B` | `identity` | `identity` | +0.000000 | +0.000000 | none | none |
| `task_C` | `identity` | `identity` | +0.000000 | +0.000000 | none | none |


### `gpt-5.6-luna` -- `BANKED_CHAIN_CLOSES_ON_GPT_5_6_LUNA`

selector -> RISK_GAP at task_A -> one Slow proposal -> compiler accepted -> the banked adoption is contained to identity, ['99999923908'] no longer crosses -0.005, and every unrelated banked episode is unchanged
- draw 1: PROPOSAL

Guard `delayed-single-series-harm`: min_per_series_gain `lt` -0.005000 -> VETO_AND_FALL_BACK on the delayed window, applies to every_adoption.

> Reject any adopted plan whose worst evaluation-series delayed gain is below the harm line, preventing a positive aggregate from concealing material harm to one series.

| step | plan before | plan after | delayed before | delayed after | harmed before | harmed after |
| --- | --- | --- | ---: | ---: | --- | --- |
| `task_A` | `outlier_iqr` | `identity` | +0.066941 | +0.000000 | 99999923908 | none |
| `task_B` | `identity` | `identity` | +0.000000 | +0.000000 | none | none |
| `task_C` | `identity` | `identity` | +0.000000 | +0.000000 | none | none |


## Cost and integrity

- LLM calls: 2 / 12.
- Consumer retrains: 0.
- Frozen surface: 39 files, drift [].
- Wall seconds: 78.2.
