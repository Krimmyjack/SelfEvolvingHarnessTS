# C1-lite：P1b 表示 vs episodic 记忆机制 因果隔离（16 半块×5 排列；k=5，非 confirmatory）

| arm | cum regret [min,max] | 复发增益 | 首遇 harm(max) | override% | in-sup adv | out-sup adv |
|---|---|---|---|---|---|---|
| frozen | 0.3677 [0.368,0.368] | +0.0000 | +0.000 | 0.00 | +0.0000(n2110) | +0.0000(n1250) |
| P1b-static | 0.2976 [0.298,0.298] | +0.0906 | +0.099 | 0.89 | +0.0934(n2110) | +0.0386(n1250) |
| P1b-memory | 0.3424 [0.322,0.357] | +0.0574 | +0.107 | 0.56 | +0.0379(n2110) | +0.0000(n1250) |
| P1b-random-memory | 0.4109 [0.390,0.433] | -0.0528 | +0.209 | 0.55 | -0.0669(n2110) | +0.0000(n1250) |

**因果判读（点估计，非门控）**：
- ① memory 胜 static(B+)：**False** （Δ=-0.0448；B+ 为 leaky 强上界，偏向零假设）
- ① memory 胜 random-memory：**True** （Δ=+0.0685；leakage-immune 内容承重主检验）
- （旁证）memory 胜 frozen：**True**（Δ=+0.0253）
- ② 收益集中 recurrence/in-support：**True**
- ③ 首遇 harm < δ_safe(0.05)：**False**

**结论**：Memory 线存活（①∧②）= **False**；需 escalation gate（首遇 harm 超 δ_safe）。

> 存活 → 进 B1b-mini 后完整方法；否则关闭 Memory claim（不跑 M0–M3/双底座/cross-domain）。 random-memory 对照 leakage-immune；B+ 对照泄漏偏向零假设。**pilot，非 confirmatory**。
