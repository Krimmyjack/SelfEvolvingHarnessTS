# 预注册：P5-A.3 architecture-complete confirmatory evaluation（2026-07-10 冻结）

> 定位（外评二审措辞）：**不是"再次翻案"**。三轮各答各题——P5-A=破损接口下的端到端失败；
> P5-A.2=接口修复的受控归因（deficit 缩小 81%，非严格 factorial）；**P5-A.3=完整信息面
> （packet v2 连续证据）+ ReadinessPlan→deterministic compiler 架构下的最终评估**
> = proposer-LLM 的最后一次审判。两个结局都有论文分支。

## 数据与种子（一次性）

- **真实 Monash**（nn5_daily / tourism_monthly / fred_md，12 基底信号）× FORECAST_PRESETS
  退化网格；**seeds 80–99**（每 seed 3 episodes：(signal, preset) 确定性轮转 + 派生退化
  seed）= 60 episodes；group = seed（grouped bootstrap B=2000, CI90）。seeds 40–59/60–79
  已消耗，永不复用。
- 判官 = seasonal-naive（period=系列自身周期）nRMSE vs **真实观测未来**（true 判官）；
  task = forecast only。
- **命名修正（外评二审）**：records/manifest 用 `series_family`（=数据集 config）与
  `pattern_preset`（=退化 preset）字段；合成时代的 `anomaly|*` cell 命名不再出现。

## 臂（7 臂；主臂先验声明，禁事后挑臂）

| 臂 | 角色 |
|---|---|
| frozen | 地板参照（abstain，true=0） |
| random_valid | 供给随机地板（3 采样按 proxy 选优） |
| det_search | **固定主基线**（dev 冻结梯 [winsor+savgol, winsor+median9, median9]，不变） |
| pv2_direct | packet v2 → 直接 ProgramSpec LLM（隔离"连续证据对 direct 生成"的作用） |
| **pv2_plan_compiler** | **PRIMARY headline**：packet v2 → ReadinessPlan → deterministic compiler |
| plan_compiler_no_ce | 主臂去 continuous_evidence 通道（直接测 R1 增量） |
| plan_compiler_no_skills | 主臂去 skills 通道（机制 ablation，secondary，不判决） |

外部 code-agent/project 基线（AutoTTS 类）**显式延期**（需独立集成，非本 prereg 范围）；
历史 closed-menu serving-LLM 已由 slice v1/v2 判决，不重复设臂（menu 动作 w5 剂量不在
grammar 网格，亦不可经同一 gym 通道公平比较——记录为设计事实而非回避）。

## 协议

- 每臂每 episode 候选预算 **B=3**（LLM 臂=3 nonce 采样，temperature 0.7，repair_retries=1
  计入 api_calls；invalid=消耗预算的 ITT no-op）；proxy（rolling-origin 回测）仅作选择信号，
  验收只认 true 判官。
- **packet v2 = 契约真源**（R1 兑现）：`build_evidence_packet_v2`——pattern 指纹（观测现算）
  + seed skill cards + dev 冻结 memory 摘要 + **continuous_evidence**（从 P5Quadrant 真实
  records 聚合：同 preset、**排除同 series_uid**（LODO 式防泄漏）的 per-action true delta
  mean/q25/q75/n）+ allowed_grammar + TaskSpec。
- 缓存名 `p5a3_final`；checkpoint/resume；`client_stats`（n_api/n_hit 分离）落 manifest。

## 四指标分解（外评二审定义）

1. **Semantic**（plan 臂）：plan 良构率、guard 保留率 vs 丢弃留痕、任务契约过滤留痕
   （compile_info.dropped_*）。
2. **Compliance**：候选有效率、repair 触发/挽回率、malformed 率（预期 plan 臂≈0=架构主张）。
3. **Selection**：决策后**离线**评估同 episode 全部 B 个有效候选的 true delta →
   regret = candidate-oracle − chosen（只入报告，绝不回流选择器）。
4. **Harness benefit**：mean true delta、主对比 CI、worst-group（per series_family ×
   pattern_preset 配对差 LCB ≥ −δ_safe=0.05）、harm、成本。

## Headline 判据（同 §5 谱系；全满足才 LLM-driven）

① pv2_plan_compiler − det_search ≥ ε=0.02 ∧ grouped CI90 不跨 0；② worst-group LCB ≥ −0.05；
③ 有效新颖：**SHA-distinct 去重口径** ≥ 3（外评二审修正）；④ 成本披露。
任一不满足 → headline 定格 **"self-updating deterministic harness with LLM novelty supplier"**
（P5-A.2 已证 supply 存在：11 distinct novel-effective）。

## 开跑硬前置（preflight，未过不得消耗 seeds）

- [ ] 旧 DeepSeek key 已在平台撤销（用户动作）；新 key 经 `DEEPSEEK_API_KEY` 注入；
- [ ] `--preflight` 通过：真实网络 1 次调用 n_api=1；重放 n_hit=1；repair 计数与网络失败
      分类不混淆；client_stats 落盘。
