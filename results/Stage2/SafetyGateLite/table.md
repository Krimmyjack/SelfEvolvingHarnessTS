# Readiness Adversary Table

Raw action: `v_none`; margin: 0.0; records: 672; actionable oracle rate: 0.929

| policy | valid | mean regret | gain vs raw | harm rate | top1 | precision | recall | f1 | abstain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dp_abstain | 672/672 | 0.2762 | +0.1243 | 0.391 | 0.174 | 1.000 | 0.604 | 0.753 | 0.342 |
| dp_abstain_abstain_to_raw | 672/672 | 0.3031 | +0.0974 | 0.220 | 0.173 | 1.000 | 0.420 | 0.591 | 0.342 |
| dp_abstain_support_q75 | 672/672 | 0.3042 | +0.0963 | 0.141 | 0.174 | 1.000 | 0.354 | 0.523 | 0.342 |
| dp_abstain_support_q95 | 672/672 | 0.3064 | +0.0941 | 0.201 | 0.177 | 1.000 | 0.412 | 0.583 | 0.342 |
| dp_abstain_support_q50 | 672/672 | 0.3164 | +0.0841 | 0.095 | 0.147 | 1.000 | 0.252 | 0.402 | 0.342 |
| raw | 672/672 | 0.4005 | +0.0000 | 0.000 | 0.071 | nan | 0.000 | nan | 0.000 |

Interpretation: oracle-actionable is positive only when some available action beats raw. A policy is useful when it lowers regret and gain-vs-raw without raising harm.
