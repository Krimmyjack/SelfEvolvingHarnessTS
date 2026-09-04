# HEC-1 K0 freeze gate

Five mechanical checks. Reads gates, probes and predicates; reads **no** evaluation-face gain.

| check | state | note |
| --- | --- | --- |
| `receipt_matches_course` | PASS |  |
| `snapshot_resolves` | PASS |  |
| `arm_set_for_phase_t` | PASS | Static, A5-frozen, A5-online, A3-online; criterion 3 scored: True |
| `card_provenance` | PASS | 1/1 cards |
| `a5_a3_isolation` | PASS |  |

**K0_FREEZE_CLEAN** (5/5 checks). K0 empty: False.

every card proved its route and K0 is unreachable from A3.

