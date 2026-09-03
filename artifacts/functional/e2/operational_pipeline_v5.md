# One continuous operational run

**Overall: `SLOW_ENVELOPE_PROTOCOL_FAILURE_EXHAUSTED`** -- 2 protocol-failed draws on this surface (1 carried in from #24): the Agent has not produced a decision this configuration can read

One un-relayed run of the V1 Harness on noaa_fresh x pooled, arm A5, Slow pinned to `claude-opus-5`. Development level: every window was locked before the run from the #17/#19 registers, nothing beyond index 17520 was read, A5-vs-A3 was not re-estimated and no new method was introduced.

## P2 -- not re-run, precondition verified

- Carried forward: operational_pipeline_v1: 4/4 #19 task_C episodes reproduced digit-for-digit through the promoted enforcement path, 111 retrains, 0 LLM.
- Measurement-side files byte-identical to their post-P1 state: True.
- run_e2_slow_scope_update._open_stores_v2 exits unless the dependency drift is exactly [compiler_source, surface_registry]; A1 adds ttha:schema_contracts.

## Cost and integrity

- LLM calls: 3 / 12.
- Consumer retrains: 0 / 250.
- Frozen surface: 38 files, drift [].
- Wall seconds: 14.5.
