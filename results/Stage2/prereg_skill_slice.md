# 预注册：LLM-Skill 垂直切片（四臂，2026-07-06，实验运行前落字）

> 性质同既往 prereg：落盘后不改，变更=追加修正案。设计源=Component Plan §12 +
> 第三十三轮四边界/DataView/双重身份决定（memory project-llm-reentry）。
> **声明的系统级问题**："升级式（确定性默认 + 门触发 LLM composer + DataView 信息面）
> 是否胜过确定性-on-P0 的 incumbent"——**非**纯"同信息下 LLM>GBDT"（消歧留给 B+ 后续臂）。

## 1. 舞台与 estimand

同 offline gym：records_s2.jsonl 的 L_test（nested 判官标签）、per-uid regret vs oracle、
16 半块 × 5 预锁排列（run_updater2.locked_permutations 逐字）。族标签仅评估分组。

## 2. 四臂

- **A（incumbent）**：frozen dp_abstain per-uid。守卫：重算块 regret ≡ Updater2 frozen ckpt
  账本 bit 级。
- **B（wrapper 守卫臂）**：A 的 picks 经 Skill 层往返编译（action→skill(param)→action）。
  **设计上 B ≡ A bit 级**（skill 抽象零成本证明）；不占统计比较位。
- **C（LLM 全程）**：每块一个 LLM 块决策（条件小政策，见 §4），所有 16 半块。
- **D（升级式，真候选）**：触发块走 LLM 决策，非触发块 ≡ A。**触发器（锁）**=
  BlockGate flag（results/Stage2/BlockGate/report.json rows 逐字复用，固定 dev 规则，
  边界②兑现：不再称校准）∪ **各流 block 0 强制 LLM**（边界③兑现：门在零已见时不可
  校准=定义级 OOD，显式政策非隐式跳过）。
- **块决策共享（声明）**：块内容只依赖 (family, half)（16 个唯一块），DataView 块局部
  无流历史 → LLM 决策与排列位置无关，C/D 共享同一决策缓存。唯一决策数=16。

## 3. SkillSpec v1（policy/skills.py，registry sha 落盘）

7 skill 恰好覆盖 10 动作（单射入池，无新算子）：identity{v_none}；
median_smooth{w∈5,9,15,25 → v_median/f0_median_w9/w15/w25}；savgol_smooth{v_savgol}；
stl_deseason{v_stl}；wavelet_denoise{v_wavelet}；winsorize{v_winsor}；
winsor_savgol{v_winsor_savgol}。每个 SkillSpec 带 applicability/risk（**只含已发表阶段结论
的蒸馏知识**：F0 season-median 伤害、S0.7 garbage-period STL、classify 平滑符号翻转——
不含任何本 gym 的 grounded outcome）/param 范围/fallback/version。编译=param snap 到
可用剂量 + 校验，非法 → 决策作废（该块回退 frozen，计 llm_failure）。

## 4. LLM 协议（锁）

- 客户端：flash（deepseek-chat），temperature=0.0，nonce=0，cache_name="skill_slice"，
  每块 ≤2 次调用（两段式）。
- **DataView（history-only，tool-mediated，边界=第三十三轮 DataView 决定）**：
  核心视图（stage 1 直给）= structure（块 cell 构成 + P0 特征中位数/散布）/ mask（缺失率
  分布 + 代表序列 gap run 统计）/ skills（7 张 SkillSpec 卡）/ policy（frozen picks 直方图
  + abstain 率）。可请求视图（stage 2 按名提供）= window（3 代表序列末 96 点降采样 z 窗）/
  period（robust_period_diag top 候选 + rfft top-3 谱峰）/ decomp（趋势斜率 + 检出周期上的
  季节振幅 + 残差 std）。**view 构造 API 只收 history（assert size=CUT），future/clean/
  L_test 物理不可达；每块落盘 views_used 日志（=Pattern v2 特征发现仪，双重身份①）。**
- 输出 schema：`{"default":{"skill","param"},"overrides":[{"when":{"snr":"low|high",
  "miss":"none|some"},"skill","param"}],"rationale"}`——条件键=可观察 cell bin
  （snrLow/snrHigh × full/miss），逐 uid 编译。解析失败/非法 → 该块回退 frozen（llm_failure）。

## 5. 判据（锁）

- **G1 安全（承重）**：LLM 服务块上 mean(块 regret − frozen 块 regret) ≤ 0（点）**且**
  max 块级 harm < δ_safe=0.05。C、D 分别判。
- **G2 价值（承重）**：D cum < A cum（点）且 uid 级 paired（D−A）分组 bootstrap
  （组=perm×block，B=2000，seed=20260706）CI 上界 < 0。
- **G3 组合靶（命名方向性诊断，非门）**：S_both 首遇块上 LLM 服务 regret < frozen。
- **G4 成本（机械）**：D 唯一 LLM 决策触发数 ≤ C 的（=16）之 60%；LLM 实际调用数入表。
- **G5 wrapper（bit 级守卫）**：B ≡ A。
- 报告不门控：views 请求分布；C−D 在非触发块上的差（触发器盲区价值）；BlockGate 漏网
  S_both 块上 C 的表现（=更好触发器的 headroom）；llm_failure 计数；per-family 分解。

## 6. 分支（锁）

- **G1∧G2 过**：升级式架构转正为系统默认形态候选；下一轮=①B+ 臂（featurized-DataView
  确定性 router——消歧"LLM 推理 vs 信息面"，≡P1b，双重身份②）②慢路径 proposer 预注册。
- **G1 过、G2 败**：LLM 无害但无增值——view log 收割进 P1b；触发器盲区/信息面分别诊断，
  不做未预注册的 prompt 迭代。
- **G1 败**：deployment LLM composer 在此信息面被拒；确定性 incumbent 不变；view log 仍收割。
- 任何分支：**不做 prompt/温度/触发阈值的结果后调参**；改动=新预注册。

不触碰：S2 holdout（未物化）、confirmatory、seeds 20–39。预算：唯一决策 16、调用 ≤32
（temp 0 + 磁盘缓存，重跑免费）。
