# HEC-1 instrument check (eight mechanical assertions)

Reads counts, ledgers and set intersections. Reads **no** gain, utility or verdict, which is what makes it safe to self-run.

| check | state | detail |
| --- | --- | --- |
| `completeness` | PASS |  |
| `no_run_fault` | PASS |  |
| `budget` | PASS |  |
| `gate_authority` | PASS |  |
| `exposure` | PASS |  |
| `frozen_reset` | **FAIL** |  |
| `replay` | PASS |  |
| `accounting` | PASS |  |

**7/8 passed. May continue: False.**

