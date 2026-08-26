# PS-1 halted before Part 3

protocol: `ps1_proposal_shift_v2_pilot`  git: `94cf58eb5bef6ad27c3f1392d1d2ae2f22c20b86`

**PS1_SOURCES_NOT_REEARNED**

only 1 of 2 scenes re-earned the target family; a two-source hypothesis needs both

No card was compiled and no arm ran.  Part 3 is conditional on both scenes re-earning; a card compiled from one source would be a single-domain claim, which is exactly what the two-source rule exists to prevent.

## Where each re-earn attempt broke

| scene | run | earned | proposed | selected | verifier | Support | delayed | broke at | families proposed |
|---|---|---|---|---|---|---|---|---|---|
| source_A_prime | `ps0_srcA_1` | True | True | False | True | True | True | **-** | hampel, level_shift, outlier_threshold |
| source_B_prime | `ps0_srcB_1` | False | False | False | False | False | False | **proposed** | level_shift |
| source_B_prime | `ps0_srcB_2` | False | True | False | True | False | False | **support_material_positive** | hampel, level_shift, outlier_threshold |

### The target family's own readings

| run | round | relation | Support | delayed |
|---|---|---|---|---|
| `ps0_srcA_1` | r2 | POSITIVE | 0.4 | 0.4 |
| `ps0_srcB_2` | r2 | NEUTRAL | 0.0 | None |

- never proposed: ['ps0_srcB_1']
- proposed but Support refused: ['ps0_srcB_2']
- source A' earned without the agent ever choosing the family: it chose a level-shift candidate, that candidate was refused by the verifier, and the Support budget reached the second entry in probe_order, which was the hampel one.  Selection is reported but a run is not counted as broken there.

**Reading**: the two misses do not share a bottleneck.  One never named the family at all, which is the discovery failure S1c described.  The other named it, probed it, and the Target's own Support read exactly 0.0 -- a confirmation failure, not a discovery failure.  A hypothesis card can only address the first kind.

## Obligations

- **arms_run**: 0
- **cards_compiled**: False
- **why_no_card**: Part 3 is conditional on both scenes re-earning; compiling a card from one source would make it a single-domain claim, which is the thing the two-source rule exists to prevent
- **llm_calls**: 0
- **methods_package_unmodified**: True
- **production_governance_unmodified**: True
- **stage_report_not_written**: True
