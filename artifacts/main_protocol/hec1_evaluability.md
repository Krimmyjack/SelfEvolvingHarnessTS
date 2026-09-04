# HEC-1 evaluation-face evaluability (0 fit, 0 LLM)

Reads only whether the metric is **definable** at each scored window. No error, gain or utility participates.

| item | value |
| --- | --- |
| scheduled units (N_T) | 26 |
| **scoreable units (N_T_eff)** | **23** |
| **min paired curve points** (ceil 0.8 x N_T_eff) | **19** |
| orderings agree on N_T_eff | True |
| frozen manifest cross-check | True |
| Phase S N_S / N_S_eff | 13 / 13 |

## Units that contribute no curve point

| block | origin | reason |
| --- | ---: | --- |
| [0:40] | 2856 | horizon contains no observed truth |
| [40:80] | 2856 | horizon contains no observed truth |
| [120:160] | 1656 | horizon contains no observed truth |

These still run and still write Episodes. They are absent from the curve for **every arm identically**.

