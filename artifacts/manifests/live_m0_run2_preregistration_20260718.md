# Live M0 run 2 preregistration

Run 2 asks whether the evolution loop can open after repairing the contract and
observation instruments. It does not ask H0 to solve every repairable case.

## Primary outcome

Success requires at least one schema-valid slow edit to enter paired replay and
receive a scientific verdict other than `INCONCLUSIVE`. A first
`SUPPORTED_EDIT` and promotion is the stretch outcome, not a prerequisite for
declaring that the loop is operational.

The experiment is invalid if launched before the fresh offline acceptance
receipt is `PASS`.

## Frozen inputs

- Planned run: `live-scientific-20260718-05`, two cycles, 36 cases per cycle
- Model alias: `gpt-5.5`
- Corpus: `m0-corpus/2`
- H0 content SHA: `abf956c90c840b81ed5092df677dfd308b806dba1fb9cd5afeb46191e95ef44b`
- H0 runtime SHA: `0d4d5e483a8a75dc738fa3369782e1f52470c3e91d1ab41c18b917b388b9d380`
- Offline tape self-verification SHA: `8462bdaeb2c3329acb75095242c5c97c7f8e39cd50f94bd41697a52f10d88a95`
- Fixed-program baseline report SHA: `25134240efb32126b4f46fd06311a82ad718bd8c74b0ec6b9e1080c27ce015d4`

## Registered readouts and run-1 controls

| Readout | Run 1 | Run 2 interpretation |
| --- | ---: | --- |
| Valid envelopes | 129/133 | Report exact rate; should not fall below run 1 |
| Schema correction retries | 0 | Count only `stage-validation-error/1`; more than one per slow attempt is invalid |
| Cases with semantic no-op PROGRAM | 12/36 | Expected to fall; directional diagnostic, not primary gate |
| Probe-selection contradictions | approximately 3-5 | Report exact count; corpus-v2 rerouting may change it |
| Slow AST pass | 0/2 | At least one valid manifest or explicit no-proposal must cross the contract |
| Supported edits / promotions | 0 / 0 | First one is the stretch outcome |
| Target supply: missing | 4/6 | Report again |
| Target supply: outlier | 0/6 | May remain limited by localization |
| Target supply: level-shift | 0/6 | May improve immediately after feature/parameter fixes |
| Target supply: period-change | 0/6 | Identity plus `OPERATOR_GAP` is expected |

The final report must also record every paired-replay verdict, H0-to-final
Harness SHA transition, retry usage, and per-family effect-distinct supply.
