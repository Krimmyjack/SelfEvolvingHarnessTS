# S0.6b S1 流重锚 — decision（flash 真 Monash + 候选级日志）

日期：2026-07-02
脚本：`run_stream_s1 --llm flash --npz AdaCTS/data/monash_real.npz --min-signals 4 --n-per-signal 8 --epochs 2 --cand-log ...`
domain 序（canonical）：fred_md → nn5_daily → tourism_monthly（K=3，tourism 最后）

## 前向迁移摘要（三 bootstrap）

| mode | fred_md | nn5_daily | tourism_monthly |
|---|---|---|---|
| updating | ver2 readiness=11.6 ttr=1 | ver2 r=0.0 ttr=None | ver4 r=1.0 ttr=1 |
| frozen   | ver0 r=0.50 | ver2 r=0.0 | ver2 r=1.0 |
| scratch  | ver2 r=11.6 ttr=1 | ver1 r=4.7 ttr=2 | ver2 r=1.0 ttr=1 |

- readiness>1（如 11.6）= J_cur ≪ J_min_ref（harness 远优于 minimal 参照）——System Req §13 已警告归一 readiness 可 >1；报告须并列 median/worst，勿单均值当 headline。
- **reval_demote=0**（所有 domain）：本短流未产出对 tourism 负迁移的 cell-scoped 模板（2 epoch、cells 少、多数模板未越 ε 合入）→ S0.4 机制在此 anchor 未被触发（其正确性由单测 `test_revalidate_templates_demotes_negative_transfer` 保证）。**非** S0.4 失效，是数据未制造该 case。

## 候选级日志（S0.5 核心交付，`candidates_flash.jsonl`）

- **41 条候选**（updating 路径），2 accept / 39 reject；4 distinct cell；**37 distinct patch path**（LLM 提议多样，全为 cell-scoped `l2.task_templates::...`）。
- 拒绝原因：`held_in_not_fulfilled` ×36（编辑不越 ε）、`held_out_a_regress` ×3（过拟合被同 cell 另批拦）。
- 每条含：完整 patch（**含被拒**）、四 v 值 + Δ、split 指纹（series_uids/origins/sample_hash，含 held_out_b 跨 cell 组）、artifact_key（确定性重评锚）。
- **这是此前 repo 缺失的 reject 侧数据** → 解锁 E-6.1 模式 B（fixed-proposal-path replay）、E-4.2（拒绝解剖）、E-7.3（跨 cell 复用 replay 的 source patch 池）。

## 关键读数（喂 Stage 1/2）

- accept 率 2/41 ≈ 5%，延续历史 6–11% 量级 → 瓶颈是 **headroom-vs-ε**（多数 cell-scoped 编辑 |Δ|<ε），非 proposer 找不到方向。E-3.2（LLM vs 查找表/枚举）直接测这一点。
- **全部 cell LOWCONF**（真 Monash 仅 4 信号/config，uids=3–4<6）→ E-1.1 白名单须排除全部真数据 cell；per-cell 良定性判定在合成数据上做，真数据仅作聚合旁证（与 anchor_maintable 一致）。

## D-0.1

repo 无修复前 S1 JSONL（glob 确认）→ 本次即 canonical S1 anchor + 首个入库的候选级日志。
