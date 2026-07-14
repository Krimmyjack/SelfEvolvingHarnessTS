# prereg:183 非裁决 sensitivity（post-hoc；不改 verdict）

status: `prereg183_non_adjudicatory_sensitivity_post_hoc`  | ε_frozen(2%)=0.015880 δ_safe=0.039699 J_raw=0.793983 B=2000
replay 忠实性（lam=0.001 vs 已落盘, bit 级）: **True**

## ① ε 灵敏度：S1/S2 点条件（S1 fire = regret≥ε ∧ lcb90>0; S2 fire = classes<2 ∨ gap<−ε）

| cycle | ε档 | ε | S1 regret_mean | S1 lcb90 | **S1 fire** | S2 gap | S2 classes | **S2 fire** |
|---|---|---|---|---|---|---|---|---|
| C1 | 1pct | 0.007940 | 0.005100 | 0.001510 | **False** | 0.000974 | 7.4375 | **False** |
| C1 | 2pct_frozen | 0.015880 | 0.005100 | 0.001510 | **False** | 0.000974 | 7.4375 | **False** |
| C1 | 5pct | 0.039699 | 0.005100 | 0.001510 | **False** | 0.000974 | 7.4375 | **False** |
| C2 | 1pct | 0.007940 | 0.007981 | 0.002175 | **True** | 0.001959 | 7.40625 | **False** |
| C2 | 2pct_frozen | 0.015880 | 0.007981 | 0.002175 | **False** | 0.001959 | 7.40625 | **False** |
| C2 | 5pct | 0.039699 | 0.007981 | 0.002175 | **False** | 0.001959 | 7.40625 | **False** |

## ② CI95 LCB 重算（cluster bootstrap 分位 0.05→0.025）

| cycle | S1 regret LCB90(q05) | S1 regret LCB95(q025) | ==persisted LCB90 |
|---|---|---|---|
| C1 | 0.001510 | 0.001020 | True |
| C2 | 0.002175 | 0.001439 | True |

S3: harm_lcb90≡0（全 cohort harm=0）→ 任意分位退化 0，CI95 亦 0。

## ③ λ=1e-2 判官重解：regret/gap 同向性

| cycle | regret λ.001 | regret λ.01 | gap λ.001 | gap λ.01 | S1 verdict 保持 | S2 verdict 保持 |
|---|---|---|---|---|---|---|
| C1 | 0.005100 | 0.005100 | 0.000974 | 0.000974 | True | True |
| C2 | 0.007981 | 0.007981 | 0.001959 | 0.001959 | True | True |

> 非裁决：以上反事实点火/方向仅供 robustness 披露，不构成任何 verdict/activate 决策。
