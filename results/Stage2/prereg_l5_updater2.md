# 预注册：L5 六臂 + updater v2 草案（2026-07-05，实验运行前落字）

> 评审第二十九轮定案。性质同 prereg_s2_replication.md：落盘后不改，变更=追加修正案小节。

## 1. L5 六臂（前台主线；数据=张量 240 槽 + P0 特征——P1a 未转正）

**效用空间声明**：loss = 张量槽 per-series nRMSE（域内 pooled-DLinear / zero-shot Chronos /
seasonal_naive；estimand 见协议 v2/v3 measurement）。与 Harness 切片的 frozen-probe 判官
效用**不同空间**——L5 在张量效用内比较策略形态，胜者接入 overlay 后再以判官口径验收。

**特征**：P0（[SNR, missing_rate] + 8 维 X_p，来自 records_s2——与 S2 复制同源）。
**评估制**：leave-one-family-out（8 折，每折 held-out 一族；策略只见 7 族标签）。
**同预算**：全部学习臂 GBDT(n_estimators=200, max_depth=3, lr=0.1, subsample=0.7, seed=20260705,
E=1)，**不调参**；本轮无 abstain（abstain 属 updater v2/2.2-⑥）。

**六臂定义**（m*=train 全均值最优模型；a*=train 全均值最优动作）：
1. `global_pair`   常数 (a,m) = train 均值 argmin（30 槽）
2. `model_only`    动作固定 a*；per-model 头预测 loss(a*, m) → argmin m
3. `action_only`   模型固定 m*；per-action 头预测 loss(a, m*) → argmin a
4. `sequential`    stage1 = action_only 选 â；stage2 = **joint 的 Q 限制在 (â,·)** 选 m
                   （= 同一 Q、分阶段决策——隔离"联合共优"与"同模型分步"的差异）
5. `joint`         单 GBDT Q(x, onehot(a,10), onehot(m,3)) 全 30 槽 argmin
6. `oracle_pair`   per-uid 30 槽 min（诊断，不进比较）

**菜单**：主结果=三模型全菜单；**副报 DLinear+Chronos 双模型子菜单**（敏感性：双模型
interaction share=0.143——L5 价值可能部分由弱 seasonal_naive 基线驱动，必须并列呈现）。

**指标**：pooled LODO policy regret（vs oracle_pair）+ paired bootstrap CI（B=2000）+
per-family regret（worst-family=安全侧）+ 模型/动作选择分解。

**分支规则（先锁）**：
- joint 赢：joint vs action_only paired CI 不跨 0 **且** worst-family 不更差 → model_id 接入
  现有 overlay（同一 forced_program 机制 + L5 面）；
- sequential ≈ joint（两者 paired CI 跨 0）且 sequential 赢 action_only → 取更简单的 sequential；
- 都不赢 action_only → 张量交互存在但（P0 特征下）不可实现，保留 action-only Harness，
  L5 学习价值降级为"待更好 Pattern"（不回头改特征——backlog 纪律不变）。

## 2. updater v2 草案（并行设计；**不绑定策略形态**，等 L5 结果定 policy 类后再实施）

最小改动链：候选 policy → 估计 applicability/support region → in-support 正常验证 →
out-of-support 自动用 incumbent → 首个新结构只 shadow/canary（观察一块后才正式启用）。

**评估纪律（防"恰好适合某一域顺序"）**：不再单一 family 顺序——预先锁定 **5 个 family
permutation**（rng(20260705) 生成、落盘后不改），报告 first-unseen harm / cumulative regret /
false accept / rollback / update coverage / incumbent fallback rate 的跨排列分布。

**实施时点**：L5 分支判决后（policy 形态确定）；届时补充完整预注册小节（追加，不改本节）。

---

## 3. updater v2 完整预注册（2026-07-05 追加；L5 已裁 KEEP-ACTION-ONLY → 绑定 action-only 形态）

**三臂**（同 S2 dev offline gym、同 L_test 账本、同预算；机制层已由切片验收 → 本轮全臂账本重放，声明）：
`frozen`（P0+abstain，永不更新）/ `updater_v1`（朴素：验证门过即部署——失败机制对照）/
`updater_v2`（下述规则栈）。

**流结构**：每族 84 uid 按 sha(uid|half)%2 切两半块（~42）→ 16 半块；**5 个预锁排列**由
rng(20260705+k) 生成（约束：≥4 族两半块间距 ≥3，不满足则确定性递增重采）；
first-encounter=该族先出现的半块，recurrence=后出现的半块（族标签只用于评估分组，禁入策略）。

**v2 规则栈**：
- **Canary（吸收冷启动）**：任何过验证门的候选先入 **shadow**——下一块整块服务 incumbent、
  候选只反事实计账；shadow 块候选 regret ≤ incumbent + δ_safe → 激活，否则弃（canary-reject）。
- **支持域混合服务**：激活后逐 uid 判 in-support（候选拟合集 P0 特征 z-score kNN，
  阈值=拟合集 LOO NN p95——Step 1.1 support 机制同款数学）：in→候选（带其 κ abstain），
  out→frozen（最终 incumbent）。族真标签禁作输入。
- **验证门/回滚**：同 v1（均值改善∧worst-group 不恶化；激活后块 harm>δ_safe vs frozen→回滚）。

**成功判据（防"永远回退"作弊，全 5 排列聚合）**：①cumulative regret 不劣于 frozen；
②first-unseen harm 受控（每族首遇半块 vs frozen 的 Δ 上界 < δ_safe）；③false accept+rollback
< v1；④update coverage > 0（候选真实服务的 uid 占比非零）；⑤recurrence 半块上优于 frozen；
⑥time-to-readiness 缩短（首个 updating<frozen 且保持的块序）。

**分支（先锁）**：安全且非零适应收益→v2 成为确定性对照，LLM proposer（v3）上场资格解锁；
安全但覆盖≈0→门太保守（非成功）；仍有 OOD false accept→P0 support 不足，转 response-aware
support/episodic memory（不调阈值）。
