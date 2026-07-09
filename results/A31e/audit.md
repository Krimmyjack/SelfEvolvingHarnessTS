# A-31e 补样审计（A-38 协议）

日期：2026-07-04　目标 N=40/槽位　补样总数 +240　接受判据=perceive cell 命中（零 loss 参与）

| cell | origin | before | after | SNR p10/p50/p90 (dB, after) | rolled | 状态 |
|---|---|--:|--:|---|--:|---|
| forecast|snrHigh|full | S_season | 22 | 40 | +5.5 / +8.0 / +9.0 | 0 | 补齐 |
| forecast|snrHigh|full | S_trend | 35 | 40 | +4.4 / +11.5 / +13.4 | 0 | 补齐 |
| forecast|snrHigh|full | S_both | 31 | 40 | +4.2 / +11.4 / +13.2 | 0 | 补齐 |
| forecast|snrHigh|full | S_ar | 0 | 0 | — | 0 | 结构性 infeasible |
| forecast|snrHigh|miss | S_season | 17 | 40 | +4.6 / +5.4 / +6.7 | 10 | 补齐 |
| forecast|snrHigh|miss | S_trend | 32 | 40 | +4.4 / +11.9 / +13.1 | 0 | 补齐 |
| forecast|snrHigh|miss | S_both | 29 | 40 | +4.4 / +10.8 / +13.1 | 0 | 补齐 |
| forecast|snrHigh|miss | S_ar | 0 | 0 | — | 0 | 结构性 infeasible |
| forecast|snrLow|full | S_season | 18 | 40 | -6.2 / +2.1 / +3.6 | 0 | 补齐 |
| forecast|snrLow|full | S_trend | 5 | 40 | -6.2 / -2.5 / +3.6 | 4 | 补齐 |
| forecast|snrLow|full | S_both | 9 | 40 | -6.3 / -3.3 / +3.4 | 0 | 补齐 |
| forecast|snrLow|full | S_ar | 40 | 40 | -8.2 / -6.7 / -5.4 | 0 | 已满 |
| forecast|snrLow|miss | S_season | 23 | 40 | -2.7 / +2.1 / +3.1 | 0 | 补齐 |
| forecast|snrLow|miss | S_trend | 8 | 40 | -3.6 / +1.5 / +3.8 | 0 | 补齐 |
| forecast|snrLow|miss | S_both | 11 | 40 | -2.4 / +1.8 / +3.7 | 0 | 补齐 |
| forecast|snrLow|miss | S_ar | 40 | 40 | -8.1 / -6.5 / -5.1 | 0 | 已满 |

## 结构间 SNR overlap（cell 内，[p10,p90] 区间交/并）

- forecast|snrHigh|full: overlap=0.37　带界(existing median)=+8.37dB　S_season[+5.5,+9.0]　S_trend[+4.4,+13.4]　S_both[+4.2,+13.2]
- forecast|snrHigh|miss: overlap=0.24　带界(existing median)=+10.42dB　S_season[+4.6,+6.7]　S_trend[+4.4,+13.1]　S_both[+4.4,+13.1]
- forecast|snrLow|full: overlap=0.06　带界(existing median)=-5.35dB　S_season[-6.2,+3.6]　S_trend[-6.2,+3.6]　S_both[-6.3,+3.4]　S_ar[-8.2,-5.4]
- forecast|snrLow|miss: overlap=0.00　带界(existing median)=+0.41dB　S_season[-2.7,+3.1]　S_trend[-3.6,+3.8]　S_both[-2.4,+3.7]　S_ar[-8.1,-5.1]

解释边界：overlap<1 = 结构与连续 SNR 部分纠缠不可完全解开（S_ar 实测 SNR 有结构性上限）
→ E-3.2 判据 (vi) 连续 SNR residualization + SNR 分层置换仍为必要控制（A-37③）。
