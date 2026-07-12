# P3 VERDICT：gym 认证 + proxy 保真度 + 种子供给 headroom + ε 注册

> 范围：synthetic gym substrate (motivation-grade); real-corpus supply verdict belongs to P5 identity gate；seed=20260709，n=60，B=1000，ρ_min=0.7。

## 保真度（R4 硬门材料；主判据=within-series 排序保真，pooled 仅诊断）

- forecast: within-series 均值 ρ=0.5699079227663222（p25=0.37350014381303226, n_series=60，pooled 诊断=-0.18589241221832217）→ **FAIL**
- anomaly_detection（deployable 面）: status=**INSUFFICIENT_VARIANCE**（within 覆盖率=0.00）
- anomaly 违约诊断组: pooled ρ=0.8503846574372074（n=180）→ proxy 在空间移动时确实跟踪 true
- escalate-only 任务: ['forecast', 'anomaly_detection']

## 供给 headroom（vs 冻结 menu v1 + dose oracle，同判官同 split）

- forecast: mean **+0.0000** [+0.0000,+0.0000]，seed 严格获胜序列 0/60，胜者分布 {}
- anomaly_detection: mean **+0.0000** [+0.0000,+0.0000]，seed 严格获胜序列 0/60，胜者分布 {}

## ε 正式注册：**0.02**（规则 max(0.02, forecast headroom ci90_lo) = max(0.02, 0.0000)）

P5 identity gate 的 utility 判据以此 ε 为效应量门槛；本值落盘后冻结（prereg §3）。
