# Gate 状态转移审计（执行者稿）

审计对象：Episode → LOCAL_DRAFT → Support 确认 → LOCAL_ACTIVE →
delayed 撤销/收缩 → Shared Promotion。
方法：只读代码；四个历史案例只做语义兼容性 replay，不调阈值、不拟合。

## 1. 状态与真实入口

### 1.1 Episode 状态枚举
`methods/ttha/experience_memory.py`：
- `EPISODE_ONLY`
- `LOCAL_DRAFT`
- `LOCAL_ACTIVE`
- `RESTRICTED`

### 1.2 状态转移函数
| 阶段 | 函数 | 当前行为 |
|---|---|---|
| Episode 形成 | `online_loop._write_target_episode` | support ≥ M → `LOCAL_DRAFT`；否则 `EPISODE_ONLY` |
| delayed 状态更新 | `online_loop._update_delayed_status` | 双正 → POSITIVE/LOCAL_ACTIVE；support 正 delayed < M → CONFLICT/RESTRICTED；双负 → NEGATIVE/EPISODE_ONLY；support 负 delayed 正 → CONFLICT/EPISODE_ONLY |
| Fast winner → Draft Skill | `method.handle_fast_winner` | 宽 Scope → `requires_target_support=true`；support ≥ M → `pending`。调用方传入 support_gain 时复用已通过 probe，不重复 verifier |
| pending → 批准/拒绝 | `method.handle_feedback_delayed` | delayed verifier 通过 + finite + `dg >= -M` → approved；否则 delayed_rejected，snapshot 不变 |
| 已部署 Skill delayed < −M | `online_loop.revoke_deployed_skill` | 从 fork 删除 `skills/{learned,bootstrap}/<id>.json` → 重编译 → materialize → set_active |
| 批准后激活 | `online_loop.activate_approved` | 仅 `_delayed_event.stage == approved` 才 set_active |

## 2. 关键发现（审计意见，暂不修改）

1. **Program 层 delayed 已经是单侧 harm veto**：
   `method.handle_feedback_delayed` 使用 `dg >= -M` 批准，不要求显著正。
   这与既定方向一致。

2. **Episode 层 delayed 没有“中性”档位**：
   `_update_delayed_status` 对 support 正且 delayed ∈ [−M, M) 的轨迹判为
   `CONFLICT/RESTRICTED`。这比 Program 层更严：中性 delayed 在 Episode
   层直接进入 RESTRICTED。两层的语义不一致，未来 Scope 收缩时应以哪个
   为准必须先冻结，不能各自解释。

3. **中性 delayed 目前仍可让 pending Skill 获得 approved 并进入 active**：
   `handle_feedback_delayed` 对 `dg >= -M` 一律 `approved`；`open_delayed`
   随后可 `activate_approved`。这与“中性 delayed 只能维持当前本地 Scope、
   不得扩大执行权”的目标语义冲突。审计意见：批准与扩权应拆成两个条件
   （例如 approved 只更新 snapshot，LOCAL_ACTIVE 需 delayed > 0 或额外
   Scope 证据）。

4. **已部署 Skill 的 delayed 负反馈只有“整卡撤销”，没有“Scope 收缩”**：
   `revoke_deployed_skill` 删除整个 skill 文件。目标语义允许“撤销或收缩
   Scope”；当前代码无收缩路径。未来 Rule Card 应可触发 SPLIT/RESTRICT，
   而不是只能删卡。

5. **Shared Promotion 在当前代码中不存在**：
   所有快照更新和 active 都发生在单个 Target-local 运行内；没有跨数据域
   promotion 入口。这是与“Shared Capability 中性 delayed 不得跨域晋升”
   兼容的现状，不是缺陷，但要在文档中显式声明。

6. **`open_delayed` 对同一 winner 存在两次 delayed 仪器评估**：
   Episode 状态更新时评估一次，计算 `delayed_utility` 时再评估一次。
   当 winner 同时形成 Episode 时构成重复计量。当前只审计，不修改；
   未来 Gate/预算计量修正候选。

## 3. 四类历史样本语义 replay

| 样本 | 当前代码能否表达 | 路径 |
|---|---|---|
| TSEM rev6→rev7（已知正向） | 可表达 | guidance pending → G3 行为核销 + G4 Support/delayed → `activate_pending_guidance`；不属于 Program Skill 生命周期 |
| G3/P4 被拒自由文本 Patch | 可表达 | guidance pending → replay 失败 → 拒绝，snapshot 不变 |
| a5v3 winsorize | 可表达 | `handle_fast_winner` → pending → delayed 批准 → 部署 → delayed < −M → `revoke_deployed_skill` |
| loop1 rev8 | 可表达 | guidance fork → replay `LOOP1_NO_CONTRAST` → 不批准 |

结论：四类样本都能被当前状态机表达，无需改代码做 replay。
但第 2 节第 3/4 条说明“表达”不等于“目标语义已满足”。

## 4. 交给 checker 的重点问题
- 我对 `_update_delayed_status` 中性档位的判断是否准确？
- `handle_feedback_delayed` 的中性 delayed approved → active 路径是否还受
  `activate_approved` 或 `store` 的额外约束？
- `revoke_deployed_skill` 是否只有删除路径，没有 RESTRICT/SPLIT 路径？
- Shared Promotion 是否确实无代码入口，还是存在于 harness/retrieval 层而我遗漏？
