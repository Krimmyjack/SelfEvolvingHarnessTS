# A38C confirmatory 补样审计（A-40④/A-41③，带界=冻结 dev 值）

日期：2026-07-04　目标 N=40/槽位　补样总数 +240　接受判据=perceive cell 命中（零 loss 参与）

| cell | origin | before | after | SNR p10/p50/p90 (dB, after) | rolled | 状态 |
|---|---|--:|--:|---|--:|---|
| forecast|snrHigh|full | S_season | 20 | 40 | +5.5 / +8.1 / +9.0 | 0 | 补齐 |
| forecast|snrHigh|full | S_trend | 36 | 40 | +4.2 / +11.8 / +12.9 | 0 | 补齐 |
| forecast|snrHigh|full | S_both | 28 | 40 | +4.2 / +11.9 / +13.8 | 0 | 补齐 |
| forecast|snrHigh|full | S_ar | 0 | 0 | — | 0 | 结构性 infeasible |
| forecast|snrHigh|miss | S_season | 19 | 40 | +4.4 / +5.2 / +8.2 | 9 | 补齐 |
| forecast|snrHigh|miss | S_trend | 32 | 40 | +4.5 / +12.2 / +13.7 | 0 | 补齐 |
| forecast|snrHigh|miss | S_both | 31 | 40 | +4.5 / +10.7 / +12.9 | 0 | 补齐 |
| forecast|snrHigh|miss | S_ar | 0 | 0 | — | 0 | 结构性 infeasible |
| forecast|snrLow|full | S_season | 20 | 40 | -6.1 / +2.1 / +3.2 | 0 | 补齐 |
| forecast|snrLow|full | S_trend | 4 | 40 | -6.1 / -4.3 / +3.5 | 0 | 补齐 |
| forecast|snrLow|full | S_both | 12 | 40 | -6.0 / -3.3 / +3.8 | 0 | 补齐 |
| forecast|snrLow|full | S_ar | 40 | 40 | -8.0 / -7.1 / -5.5 | 0 | 已满 |
| forecast|snrLow|miss | S_season | 21 | 40 | -2.9 / +1.6 / +3.6 | 0 | 补齐 |
| forecast|snrLow|miss | S_trend | 8 | 40 | -3.6 / +1.6 / +3.8 | 0 | 补齐 |
| forecast|snrLow|miss | S_both | 9 | 40 | -4.4 / +1.2 / +3.8 | 0 | 补齐 |
| forecast|snrLow|miss | S_ar | 40 | 40 | -8.1 / -6.6 / -5.0 | 0 | 已满 |

## 结构间 SNR overlap（cell 内，[p10,p90] 区间交/并）

- forecast|snrHigh|full: overlap=0.36　带界(existing median)=+8.37dB　S_season[+5.5,+9.0]　S_trend[+4.2,+12.9]　S_both[+4.2,+13.8]
- forecast|snrHigh|miss: overlap=0.40　带界(existing median)=+10.42dB　S_season[+4.4,+8.2]　S_trend[+4.5,+13.7]　S_both[+4.5,+12.9]
- forecast|snrLow|full: overlap=0.04　带界(existing median)=-5.35dB　S_season[-6.1,+3.2]　S_trend[-6.1,+3.5]　S_both[-6.0,+3.8]　S_ar[-8.0,-5.5]
- forecast|snrLow|miss: overlap=0.00　带界(existing median)=+0.41dB　S_season[-2.9,+3.6]　S_trend[-3.6,+3.8]　S_both[-4.4,+3.8]　S_ar[-8.1,-5.0]

解释边界：overlap<1 = 结构与连续 SNR 部分纠缠不可完全解开（S_ar 实测 SNR 有结构性上限）
→ E-3.2 判据 (vi) 连续 SNR residualization + SNR 分层置换仍为必要控制（A-37③）。
