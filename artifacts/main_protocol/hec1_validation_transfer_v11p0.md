# HEC-1 validation transfer diagnostic

POST_HOC_OUTCOME_DIAGNOSTIC: evaluation outcomes are used only to attribute failure; no choice here is deployable or enters an arm

| item | value |
| --- | ---: |
| scoreable units with a Support-safe candidate | 16 |
| Support-safe candidate pairs redeployed | 34 |
| candidate pairs retaining safety | 10 |
| selected candidate retains safety | 7 |
| another Support-safe candidate would rescue | 1 |
| no Support-safe candidate retains safety | 8 |
| Consumer fits | 68 |
| LLM calls | 0 |

| unit | Support-safe | evaluation-safe | selected stable | alternative rescue |
| --- | ---: | ---: | --- | --- |
| [0:40] x 1176 | 1 | 0 | False | False |
| [0:40] x 1896 | 1 | 0 | False | False |
| [0:40] x 2136 | 2 | 1 | True | False |
| [0:40] x 2376 | 0 | 0 | False | False |
| [0:40] x 2616 | 1 | 1 | True | False |
| [0:40] x 3576 | 0 | 0 | False | False |
| [40:80] x 1176 | 1 | 1 | True | False |
| [40:80] x 1416 | 0 | 0 | False | False |
| [40:80] x 1656 | 0 | 0 | False | False |
| [40:80] x 2136 | 3 | 3 | True | False |
| [40:80] x 2376 | 2 | 1 | True | False |
| [40:80] x 2616 | 3 | 1 | False | True |
| [40:80] x 3576 | 0 | 0 | False | False |
| [40:80] x 3816 | 1 | 0 | False | False |
| [80:120] x 1176 | 2 | 1 | True | False |
| [80:120] x 1416 | 2 | 0 | False | False |
| [80:120] x 1896 | 1 | 1 | True | False |
| [80:120] x 2136 | 1 | 0 | False | False |
| [80:120] x 2376 | 11 | 0 | False | False |
| [80:120] x 2616 | 0 | 0 | False | False |
| [80:120] x 2856 | 1 | 0 | False | False |
| [120:160] x 1176 | 1 | 0 | False | False |
| [120:160] x 1416 | 0 | 0 | False | False |

