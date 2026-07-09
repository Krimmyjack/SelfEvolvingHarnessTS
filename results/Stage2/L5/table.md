# L5 六臂（张量效用 × P0 特征，leave-one-family-out；prereg_l5_updater2.md §1）

> 主结果=三模型菜单；副报 DLinear+Chronos 子菜单（敏感性：双模型 share=0.143——L5 价值可能部分由弱 seasonal_naive 基线驱动）。

## 三模型（主）  n=672
| arm | mean regret | worst family (regret) | 模型选择分布 |
|---|---|---|---|
| global_pair | 0.5059 | S_regime (1.922) | {'dlinear_pooled': 504, 'chronos_bolt_small': 168} |
| model_only | 0.5697 | S_regime (1.626) | {'dlinear_pooled': 307, 'chronos_bolt_small': 274, 'seasonal_naive': 91} |
| action_only | 0.6086 | S_regime (1.998) | {'dlinear_pooled': 504, 'chronos_bolt_small': 168} |
| sequential | 0.6911 | S_regime (1.998) | {'chronos_bolt_small': 214, 'dlinear_pooled': 390, 'seasonal_naive': 68} |
| joint | 0.6337 | S_regime (1.888) | {'chronos_bolt_small': 218, 'dlinear_pooled': 393, 'seasonal_naive': 61} |
| oracle_pair | 0（锚；mean loss=0.659） | — | — |

关键比较（paired ΔRegret，负=前者更好）：
- joint_vs_action_only: +0.0251 [-0.0296, +0.0776]
- sequential_vs_action_only: +0.0825 [+0.0601, +0.1052]
- joint_vs_sequential: -0.0574 [-0.1067, -0.0095]
- model_only_vs_action_only: -0.0388 [-0.1006, +0.0174]
- joint_vs_global_pair: +0.1277 [+0.0930, +0.1634]

## DLinear+Chronos（副）  n=672
| arm | mean regret | worst family (regret) | 模型选择分布 |
|---|---|---|---|
| global_pair | 0.4979 | S_regime (1.877) | {'dlinear_pooled': 504, 'chronos_bolt_small': 168} |
| model_only | 0.4634 | S_regime (1.588) | {'dlinear_pooled': 272, 'chronos_bolt_small': 400} |
| action_only | 0.6005 | S_regime (1.953) | {'dlinear_pooled': 504, 'chronos_bolt_small': 168} |
| sequential | 0.6017 | S_regime (1.875) | {'chronos_bolt_small': 310, 'dlinear_pooled': 362} |
| joint | 0.5054 | S_regime (1.829) | {'dlinear_pooled': 317, 'chronos_bolt_small': 355} |
| oracle_pair | 0（锚；mean loss=0.667） | — | — |

关键比较（paired ΔRegret，负=前者更好）：
- joint_vs_action_only: -0.0951 [-0.1469, -0.0483]
- sequential_vs_action_only: +0.0012 [-0.0271, +0.0283]
- joint_vs_sequential: -0.0963 [-0.1539, -0.0461]
- model_only_vs_action_only: -0.1371 [-0.1910, -0.0889]
- joint_vs_global_pair: +0.0075 [-0.0183, +0.0314]

**分支判决（预注册规则）**：KEEP-ACTION-ONLY：张量交互存在但 P0 特征下不可实现——保留 action-only Harness
（joint 均值+安全=False，seq≈joint=False，seq>action_only=False）
