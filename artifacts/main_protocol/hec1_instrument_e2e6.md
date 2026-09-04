# HEC-1 instrument check (eight mechanical assertions)

Reads counts, ledgers and set intersections. Reads **no** gain, utility or verdict, which is what makes it safe to self-run.

| check | state | detail |
| --- | --- | --- |
| `completeness` | PASS |  |
| `no_run_fault` | PASS |  |
| `budget` | PASS |  |
| `gate_authority` | PASS |  |
| `exposure` | PASS |  |
| `frozen_reset` | **FAIL** | problems: [{"arm": "A3-frozen", "why": "the frozen arm was resupplied a restricted Draft opened in an earlier unit", "cells": [{"position": 2, "ids": ["resupplied_draft_1 |
| `replay` | PASS |  |
| `accounting` | PASS |  |

**7/8 passed. May continue: False.**

