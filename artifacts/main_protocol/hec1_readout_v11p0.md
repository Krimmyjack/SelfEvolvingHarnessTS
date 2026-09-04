# HEC-1 course readout

ORACLE_WALL: computed after the course from the scoring ledger only.  No Episode bank is opened and no value here may reach a prompt.

| item | value |
| --- | --- |
| orderings found | 3 of 3 |
| terminal differences above zero | 2 |
| verdict | **HEC1_EVOLUTION_NOT_SUPPORTED** |

P1 did not hold (1/3 orderings, 2/4 cohorts, harm ok=True); see the first-fault map in the contract.

| ordering | ran | paired pts (min 19) | terminal diff | AUC | harm ok | P2 chain |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| forward | 26 | 23 | 0.211896 | 4.105656 | True | False |
| reverse | 26 | 23 | -0.043036 | -0.00823 | True | False |
| interleaved | 26 | 23 | 0.005719 | 0.216044 | True | False |

| cohort | d_c |
| --- | ---: |
| [0:40] | 0.012404 |
| [120:160] | 0.0 |
| [40:80] | 0.000477 |
| [80:120] | -0.002863 |

Cohort sign pattern: 2 positive of 3; exact binomial probability 0.5 (floor at all-positive 0.125) -- descriptive, not a test.

