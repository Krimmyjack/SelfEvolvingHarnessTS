# Readiness Adversary Table

Raw action: `v_none`; margin: 0.0; records: 672; actionable oracle rate: 0.929

| policy | valid | mean regret | gain vs raw | harm rate | top1 | precision | recall | f1 | abstain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle | 672/672 | 0.0000 | +0.4005 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| dp_abstain | 672/672 | 0.2762 | +0.1243 | 0.391 | 0.174 | 1.000 | 0.604 | 0.753 | 0.342 |
| dp_gbdt | 672/672 | 0.2965 | +0.1040 | 0.329 | 0.195 | 1.000 | 0.599 | 0.749 | 0.000 |
| P1b-static | 672/672 | 0.2965 | +0.1040 | 0.344 | 0.228 | 1.000 | 0.609 | 0.757 | 0.000 |
| raw | 672/672 | 0.4005 | +0.0000 | 0.000 | 0.071 | nan | 0.000 | nan | 0.000 |
| d_lookup | 672/672 | 0.4147 | -0.0142 | 0.467 | 0.073 | 1.000 | 0.479 | 0.648 | 0.000 |
| global | 672/672 | 0.4264 | -0.0259 | 0.317 | 0.088 | 1.000 | 0.304 | 0.467 | 0.000 |

Interpretation: oracle-actionable is positive only when some available action beats raw. A policy is useful when it lowers regret and gain-vs-raw without raising harm.
