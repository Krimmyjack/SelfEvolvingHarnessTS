# Parallel Audit 2026-08-15

在线 Agent 的并行准备包。与本地 Agent 的 GRID0 / 结构 Pattern 实验互不重复：
本地 Agent 回答“Pattern-conditioned retrieval 是否可用”；本目录准备
“retrieval 结果一旦出现，如何进入现有生命周期”，以及修订后的
P1/P2 空壳预注册。

四个交付物：
1. `GATE_STATE_AUDIT.md` — 当前状态机审计 + 四类历史样本语义 replay
2. `FAULT_FAMILY_MAPPING.md` — 现有 25 个 subtype 到 6 个 fault family 的轻量映射
3. `BSE_REUSE_REVIEW.md` — BSE 已有 Rule/matcher/replay/removal 路径与未来 trigger 注入点
4. `SKELETON_SLICE_PROTOCOL.md` — 参数化纵向切片空壳协议（已被修订方向接管，暂不填 Scope）
5. `REVISED_DIRECTION.md` — 用户修订后的分支逻辑与 P0–P5
6. `PROGRAM_HEADROOM_PRECHECK_SKELETON.md` — P1 空壳
7. `RETRIEVAL_SLICE_PREREG_SKELETON.md` — P2 空壳

纪律：只读审计与草案；不建平台、不建新 Schema、不跑 fresh 实验、
不预选 workflow family 或阈值。冲突时以仓库代码为准。
