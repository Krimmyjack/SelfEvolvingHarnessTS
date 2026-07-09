# 预注册：updater v3 = response-aware support（2026-07-06，实验运行前落字）

> 性质同 prereg_l5_updater2.md：落盘后不改，变更=追加修正案小节。
> 前置判决：updater v2 三臂 PARTIAL(4/6)（results/Stage2/Updater2/），prereg §3 分支
> "仍有 OOD 伤害 → 转 response-aware support/episodic memory（不调阈值）"生效。
> 阶梯定位修正（评审第三十轮采纳）：v2 = **有效中间对照**而非部署候选——已证
> "OOD-aware 优于朴素更新"，未证"更新优于不更新"（c1）。本轮回答后者。

## 0. 边界声明（评审第三十轮核心修正，全部采纳）

1. **动作响应不是免费 Pattern**。签名必须来自**观测史内 rolling-origin 伪未来回测**，
   不得以任何形式读取当前块的 L_test 标签 / clean future——直接用完整 action-response
   行 = oracle 泄漏，v3 将被虚高。API 硬守卫：`probe_signature(hist)` 断言
   `hist.size == CUT(464)`（只接受判官口径的可观测历史）。
2. **张量行不作签名**。三模型张量效用 ≠ frozen-probe action-only 效用（不同空间），
   本轮签名全部在判官效用族内生成。张量行留在 L5 线。
3. **探针预算硬帽**：4 个探针动作 × 1 个切点，签名维度=3。代码+测试双重钉死。
   计算成本以**独立预算行**入账（探针 = 4 次 fast_process+Ridge 闭式头 ≈ 候选 refit 的
   ~10⁻³ 量级；离线 gym 目标函数无延迟/成本项 → 不做任意的 regret 单位折算，声明之）。

## 1. 响应签名定义（锁定）

- **探针集**（从冻结动作池选取，机制多样性覆盖）：
  `v_none`（identity 基准）/ `v_median`（轻 robust 平滑 w5）/ `f0_median_w25`（重 robust）/
  `v_stl`（季节分解族）。
- **切点**：观测史 `hist = degraded[:464]`；探针训练段 = `hist[:416]`（416 = CUT − 48）；
  **伪未来 = `hist[416:464]`**（退化观测值，含 NaN → 掩码；有效观测 < 8 点 → 签名无效）。
  伪未来完全在观测史内部——判官的真未来（`clean[464:]`）与 L_test 全程物理不可达。
- **探针预报器**：FrozenProbe（冻结 LSTM 编码器 + Ridge 闭式头，确定性 σ_A≈0，
  与判官同族）。每 (uid, 探针动作)：`fast_process(hist[:416], forecast, variant)` →
  `_build_windows_full` 滑窗（period=24，与判官同参）→ probe.fit → 由最后窗口直接多步
  预测 48 点。
- **损失**：masked **nMAE** = mean(|ŷ − 伪未来|_observed) / std(hist[:416] 有效观测)
  （尺度只用观测数据；nMAE 而非 nRMSE = 对退化流已知的 ±5 离群注入 robust，事前锁定）。
- **签名** Z_response = [nMAE(v_median) − nMAE(v_none), nMAE(f0_median_w25) − nMAE(v_none),
  nMAE(v_stl) − nMAE(v_none)]（相对值——消除序列难度/尺度共因，评审处方）。
- **失败语义（保守向）**：任何探针步骤失败或伪未来观测不足 → 签名无效 →
  该 uid 服务时判 **out-of-support**（回退 frozen），且**不进**候选支持域拟合集。

## 2. 三臂与单变量归因（锁定）

同 offline gym、同 records_s2.jsonl 账本、同 16 半块 × 同 5 预锁排列（locked_permutations()
逐字复用，seed=20260705）；族真标签只作评估分组。

- `frozen` / `updater_v2`：**逐字复用 Updater2/ckpt 的既有 checkpoint**（同流同账本 →
  合法重用，声明；不重跑）。v1 已充分失败，退出主表（评审处方）。
- `updater_v3`：v2 规则栈**唯一改动 = 支持域空间**——per-uid in-support 判定与候选支持域
  拟合从 P0 特征空间（[SNR, missing_rate, X_p]，z-score kNN LOO p95）替换为 **3 维响应签名
  空间**（**同一阈值配方** z-score kNN LOO p95——换空间不换旋钮，"不调阈值"红线）。
  canary / 验证门 / 回滚 / δ_safe=0.05 / κ 全部不动。

## 3. 守卫（运行前全过，否则不出表）

- **G-A 结构重放**：v3 流实现在支持域空间换回 P0 特征时，必须 bit 级复现
  Updater2/ckpt/perm0_updater_v2.json 的 ledger+events → 任何 v3 差异唯一归因于签名空间。
- **G-B 泄漏 API**：probe_signature 只接受 464 长观测史（超长即 assert 拒绝）；
  测试证明签名值与 L_test / clean future 的任何改动无关（物理不可达）。
- **G-C 确定性**：同 uid 两次计算签名 bit 级一致。
- **G-D 预算**：探针集长度=4、签名维度=3，测试钉死。

## 4. 操纵检查（运行 v3 前计算，**非门控**，纯描述）

672 uid 全签名 → 1-NN 族分类准确率（族标签仅评估用）：签名空间 vs P0 特征空间。
预期签名 > P0（响应机制分离族）；若不成立照跑主实验，结果写入报告作解释材料。

## 5. 成功判据（全 5 排列聚合，先锁）

- **g1** cum regret(v3) ≤ cum(frozen)（v2 未过的承重门——"更新优于不更新"）
- **g2** first-unseen harm（每族首遇半块 vs frozen 的 Δ 上界均值）< δ_safe=0.05（v2=0.198 未过）
- **g3** recurrence 增益 > 0（v2=+0.0033 过，须保持）
- **g4** update coverage ≥ **0.30**（预锁下限；防"签名空间过紧 → 永远回退"的伪安全）
- **g5** rollbacks + false-accepts < v2 的 7（失败少于上一级）
- **g6** 探针预算合规（§0.3 硬帽 + G-B/G-D 守卫过）——由测试执行，表内申报

报告但不门控：TTR；v2 曾受伤的首遇块上 v3 的 in-support 率（签名是否抓住 aliasing——
机制级诊断）；coverage 按首遇/复现分解。

## 6. 分支（先锁）

- **6/6 PASS**：v3 = 确定性对照封顶；**LLM proposer（阶梯第 4 级）上场资格解锁**
  （其自身仍须独立预注册：LLM 只在确定性 updater reject/停滞时提结构变更）。
- **g2 再败**：响应签名单切点不足以在首遇即识别机制 → 转 **episodic retrieval +
  探针集成分歧**（新预注册），仍不调阈值。
- **g4 败**（coverage < 0.30）：签名族内散布过大 → 先做散布诊断；P0∧签名混合支持域
  **不得**未预注册即采纳。
- **仅 g1 败**：防守价值成立、进攻侧不足 → κ/服务积极度的进攻侧设计另开一轮。

不触碰：S2 holdout（未物化）、confirmatory、seeds 20–39。
