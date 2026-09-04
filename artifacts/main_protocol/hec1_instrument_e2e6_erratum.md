# Erratum: `hec1_instrument_e2e6.{json,md}` was regenerated (2026-09-03)

Appended correction, per sol's ruling 5 of 2026-09-03. Nothing here is a scientific reading.

| item | value |
| --- | --- |
| course artifact | `hec1_course_e2e6.json` -- **unchanged**; it still records the leak |
| first instrument reading | 8/8 PASS, `may_continue=true`, written 10:04:00 by the audit **before** `frozen_reset` checked the Draft ledger |
| what changed | `frozen_reset` was tightened: a frozen arm resupplied a restricted Draft from an earlier unit is a FAIL |
| re-run on the same course | the script names its output after the course label, so `hec1_instrument_e2e6.{json,md}` were **overwritten**: now 7/8, `frozen_reset FAIL`, `may_continue=false` |
| original bytes | **not recoverable** (never committed, not copied before the re-run); content quoted from the reviewer's pre-run read in the ledger entry of 10:xx -- not reconstructed here |
| post-fix readings | new paths only: `*_e2e6_frozen_ledger_fix.*` (8/8 after the ledger reset), `*_e2e6_review_fixes.*` (8/8 after the K0 / REVISE / NOT_APPLICABLE fixes) |
| rule from here | never overwrite an instrument or scientific artifact; an audit re-run on an existing course takes a new label; any past overwrite is disclosed by an appended erratum |
