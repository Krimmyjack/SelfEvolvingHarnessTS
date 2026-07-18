# M0 corpus v2 fixed-program expressibility calibration

This private family-routed baseline is an evaluator-side ceiling, not a
deployment selector and not Agent-visible evidence.

- Corpus: `m0-corpus/2`
- Evaluator: pinned frozen Chronos with native NaN masking
- Internal report SHA: `25134240efb32126b4f46fd06311a82ad718bd8c74b0ec6b9e1080c27ce015d4`
- Raw report byte SHA-256: `0dbc57021d197fd2102687950ea36c174451c05f38ee534129d5c9ea62d10661`

| Family | Positive damage | Recoverable | Median damage D | Median best gain G |
| --- | ---: | ---: | ---: | ---: |
| missing | 6/6 | 6/6 | 0.166682 | 0.165513 |
| impulsive_outlier | 6/6 | 6/6 | 0.162058 | 0.124479 |
| level_shift | 6/6 | 6/6 | 0.286668 | 0.200122 |
| period_change | 6/6 | 0/6 | 0.998656 | 0.000000 |

The resulting corpus has the intended composition: three expressible repair
families with mechanism-distinct operators and one deliberate, proven
`OPERATOR_GAP` family.
