# 预注册：LLM-Skill 切片 v2 = 信息面/推理/粒度三方消歧（2026-07-06，运行前落字）

> 落盘后不改，变更=追加修正案。设计源=第三十四轮定案（memory project-llm-reentry）+
> 评审"两个深读"（LLM 理性保守非鲁莽；修复分实例级/类级两层）。
> **前置判决**：切片 v1 G1 败——死因=aliasing 第五层（信息面）：P0 对 S_both 盲读数
> （seasonal_strength med=0.000/period med=434）经 core 视图转述，LLM 信之且不求证
> （15/16 块 1-call、零视图请求），robust 证据躺在抽屉。
> **本轮问题**：把 robust 证据交到决策层手里之后，收益归谁——信息面（任何消费者都行）、
> LLM 推理（B+ 做不到的部分）、还是两者都救不了 S_both？
> **注册预测（对结果负责）**：B+ 吸收大部分 S_both 收益，LLM 独立价值不成立（若成立
> 是意外发现）；global B+ 可能复现 P1a/fixpc 的 S_trend/拓扑伪影失败（S2 复制先例），
> escalation-scoped D_bplus 是受保护变体——这正是升级式作用域的存在理由。

## 1. 舞台

同 v1：records_s2 L_test、per-uid regret vs oracle、16 半块 × 5 预锁排列、族标签仅评估。
触发器沿用 v1 锁定（BlockGate flag ∪ 各流 block0）；**G4 教训兑现：触发率按实例级定义**
（40/80=50%）。

## 2. 六臂

1. **A_frozen**：incumbent（复用，守卫≡账本）。
2. **C_llm_v2**：LLM 全 16 块，DataView v2（§3 双修）。
3. **D_llm_v2**：触发块用 C_llm_v2 决策，其余 ≡A。
4. **C_llm_verify**（强制求证对照）：core=v1 面（structure 带可靠性标注 + mask/skills/
   policy，robust 证据仍在抽屉）+ **协议强制两段**：stage 2 无条件附 period+decomp 再要
   最终决策——隔离"求证行为"与"证据搬运"两个变量。
5. **Bplus_v2**（=P1b 候选）：per-uid 确定性 GBDT on featurized DataView v2，
   Phase-B 冻结折 OOF（fit on L_train / picks on held-out fold——P1a/fixpc/sq 同协议，
   fold ckpt 逐字复用），全域服务。
6. **D_bplus_v2**：触发块用 Bplus_v2 picks，其余 ≡A（零额外计算，补全
   升级层×作用域 2×2）。

**决策共享**：C_llm_v2 与 D_llm_v2 共享 16 个唯一块决策；C_llm_verify 独立 16 个。
LLM 配置同 v1 锁定（flash，t=0，nonce=0，cache=skill_slice；v2/verify 提示词不同→
缓存键自然分离）。**不做结果后 prompt 迭代**。

## 3. DataView v2（双修，全部 history-only，泄漏守卫同 v1）

- **实例级（证据搬运）**：period+decomp 提进 core（core=structure_v2/mask/period/decomp/
  skills/policy；requestable 只剩 window）。
- **类级（可靠性标注+求证规范）**：structure_v2 的季节/周期读数带确定性可靠性标注——
  P0 块中位 seasonal_strength < 0.15 而 ≥2/3 代表序列 robust 检出周期（peak_ratio≥3 ∧
  acf≥0.2）→ 标注 `[低可靠：P0 季节读数与 robust 周期检测冲突，以 period/decomp 为准]`；
  system prompt 加规范行："所有摘要统计是带误差的测量而非真值；冲突时以 robust 证据
  视图为准"。
- 行为日志照旧（views_used/rationale/违规计数）。

## 4. B+ 特征集（featurized DataView v2，per-uid，history-only，缓存）

X_d=[SNR, missing_rate]（P0-D）；X_p=P0-X_p(8) ∪ v2 促升证据(7)：robust_period_diag
{period, cand_period, peak_ratio, acf_at_peak} + decomp{trend_slope, seasonal_amp@P,
resid_std}（+gap{max_run_frac, n_runs_frac} 共 17 维）。GBDT 超参=Phase-C plain 臂
逐字（seed=20260705，不调参）。**声明**：P0 特征超集 → B+ 信息面 ⊇ A；与 fixpc 的差=
无 C 通道、decomp/gap 新增——fixpc 复制失败先例已注册为预测。

## 5. 判据（锁）

- **G1 安全**（C_llm_v2/D_llm_v2/C_llm_verify/Bplus_v2/D_bplus_v2 各自的非 frozen 服务
  块）：mean Δ vs frozen ≤ 0 ∧ max 块级 Δ < δ_safe=0.05。
- **G2 价值 vs A**（D_llm_v2 / Bplus_v2 / D_bplus_v2）：cum 点 < A ∧ uid 级分组
  bootstrap（组=perm×block，B=2000，seed=20260706）CI 上界 < 0。
- **G-main（裁定"LLM 推理独立价值"——本轮主判据）**：触发块上 D_llm_v2 vs D_bplus_v2
  的 uid 级 paired 差，分组 bootstrap CI 不含 0 且 LLM 侧更优 → 独立价值成立；否则
  不成立（=注册预测）。**粒度混杂声明**：LLM=块级 cell-bin 政策、B+=per-uid——若 LLM
  败，粒度是注册的备择解释（本轮不消解，仅在需要时立后续 per-uid LLM 臂）。
- **G3 组合靶**（命名诊断）：S_both 首遇块全臂 vs frozen。
- 行为学（不门控）：v2 下视图请求率变化、verify 违规数、S_both rationale 是否仍称
  "no seasonality"、B+ 的 per-family 分解（S_trend 伪影是否复现）。

## 6. 分支（锁）

- **G-main 成立（LLM 胜 B+）**：LLM deployment 价值 level-1 确立 → 下一轮=真实域开放
  词汇试点 + 慢路径 proposer 并行。
- **B+ 吸收（注册预测）**：修复归属 Pattern——B+ 特征集作为 **P1b 候选**进入下一轮
  复制式预注册（本轮不转正，fixpc 教训）；LLM deployment composer 退出主线，
  **慢路径 proposer 升为 LLM 主擂台**（独立预注册）。
- **两者都救不了 S_both**：证据促升不足 → S_both 修复升级到 proposer 线；
  DataView v2 行为日志照收。
- 任何分支不做结果后调参；慢路径 proposer 实验=独立下一轮（不在本 prereg）。

不触碰：holdout/confirmatory/seeds 20–39。预算：LLM 唯一决策 32（v2 16 + verify 16），
调用 ≤64；B+ 特征 672 uid 一次缓存。
