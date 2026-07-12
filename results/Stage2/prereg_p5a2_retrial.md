# 预注册：P5-A.2 identity gate retrial（2026-07-10 开跑前冻结；此后不得改动）

> 动机：P5-A（seeds 40–59，已消耗）形式判决为分支 B，但机制归因显示主臂之败由**两个 harness
> 接口缺陷**主导而非生成质量（ca_skills 179/180 死于 guard 契约不一致；ca_plain 180/180
> malformed；ca_skills_memory served-conditional +0.852 ≥ det +0.799）。本 retrial 修复接口后
> 用**新种子一次性**重审 LLM 身份。P5-A 原判决不撤销、不重跑，两次审判并列入档。

## 唯一允许的两处变更（受控变量声明）

1. **guard 契约一致化（gym_fingerprint_v2）**：gym 观测面提供完整 P0 指纹（`struct_feats`
   从观测序列现算：8 个 P_FEATS + 估计 SNR + missing_rate，每 episode 缓存一次）；
   `allowed_grammar.guard_features` 的每一项都有观测背书；guard 评估与 observation 同源。
   守卫测试：`test_observation_provides_p0_fingerprint_and_contract_consistency`、
   `test_guarded_program_now_evaluable_end_to_end`。
2. **格式合规面（prompt_v2_exemplar + repair=1）**：system prompt 内嵌一条合法 ProgramSpec v1
   示例（JSON only）；schema 修复重试预算 = **1**（重试是真实调用、计入 api_calls；修复后
   仍无效 → ITT no-op）。守卫测试：`test_repair_retry_*`、`test_default_no_repair_preserves_p5a_semantics`。

**其余全部沿 prereg §5/§5.0 冻结口径不变**：六臂、B=3、主臂=ca_skills（先验）、ε=0.02、
δ_safe=0.05、K_novel=3、true 判官验收、grouped bootstrap（group=生成 seed，B=2000，CI90）、
ITT、成本全披露、det 梯与 random 采样协议、forecast only、n_per_seed=3。

## 种子与一次性

- **confirmatory seeds = 60–79（20 组 × 3 = 60 episodes），一次性打开，不重跑**；
  40–59 已被 P5-A 消耗，永不复用。
- LLM 缓存名 = `p5a2_identity`（与 P5-A 缓存隔离，防跨审判污染）。

## 判据（同 §5，全满足才翻案为 LLM headline）

① ca_skills − det ≥ ε=0.02 ∧ CI90 不跨 0；② per-cell 配对差 LCB ≥ −0.05；
③ 有效新颖编辑 ≥ 3；④ 成本披露。任一不满足 → 分支 B 维持，且**接口瓶颈假设被削弱**
（两缺陷已修仍败 = 败因更深，须如实记录）。

## 预注册的次级观察量（不判决，只记录）

- ca_plain malformed 率（预期因示例大幅下降 → 检验"格式塌缩"归因）；
- ca_skills guard 拒绝率（预期 ≈0 → 检验"契约不一致"归因）；
- served-conditional 质量 vs det（P5-A 的 +0.852 是否复现）；
- 修复重试的触发率与挽回率。
