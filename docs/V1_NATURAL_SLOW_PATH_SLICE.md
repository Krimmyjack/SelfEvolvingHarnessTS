# V1 自然 Slow Path 切片——冻结设计（修订版，2026-08-09）

里程碑定稿（§7 四十二）后，第一阻塞转型：

> **自然失败能否被 LLM 转化为可执行且有效的 Harness Update？**

本切片只检验一个问题：

> LLM 能否根据自然出现的 Program-level Support 失败和确定性反事实证据，
> 生成一个经未来反馈确认、并被下一轮正常入口实际执行的 Harness Update？

## 1. 流程（审核裁决修订 5 条，全部落实）

1. **只处理 R1 的自然 Support NEGATIVE**（gain < −M）；delayed CONFLICT
   触发留作独立切片（本切片不让 LLM 看 delayed——归因证据必须完整：
   触发条件与 LLM 可见证据同源，都用 Support）；
2. **固定时间链**（R2 不再用于寻找第二个失败——发现与验证窗口不混淆）：

   ```
   R1 @792：自然生成 A→B，Support 负向触发
   → identity/A/B/A→B 反事实（4 次确定性评估）
   → headroom 检查
   → LLM Typed Patch 冻结（KEEP/REMOVE_A/REMOVE_B/ABSTAIN）
   → @840 delayed 验证（patch Workflow）
   → 成立后写 Target-local Skill（KEEP/ABSTAIN 不形成 Skill）
   → R2 @888 正常入口**实际选择并执行** Skill（evaluate chosen）
   → @936 delayed 更新（Skill 状态）
   ```

3. **阈值 M 排除数值噪声**：
   - headroom：`max(gain_A, gain_B) >= M` **且**
     `max(gain_A, gain_B) − gain_AB >= M`；
   - Patch 有效：`gain_patch >= M`（Support replay）；
   - **"Skill 在池首位"只能证明供给，不能替代实际选择与执行**——R2
     验收 = chosen=cand_skill_* 且实际 evaluate；
   - KEEP 和 ABSTAIN 均不得形成新 Skill。
4. **明确不是新的 A5/A3 比较**：Source Memory 若存在，只能由冻结的正常
   Source 流程生成，不得根据 Target outcome 挑选；本实验唯一承重变量 =
   Slow Path 是否把自然失败转成有效 Typed Program Update。
5. **若 R1 预算内没有自然 Support NEGATIVE** → 直接输出
   `NO_NATURAL_FAILURE`，不换 origin 搜索失败案例（可信负结果足以关闭
   这一 Program-ablation 切片）。

## 2. 数据（新 virgin cohort）

| 候选 | 余量 | 选择 |
|---|---|---|
| **monash:traffic_hourly offset=80** | 806−80 支 virgin | ✓ 首选 |
| uci_electricity offset=40 | 330 支 | 后备 |
| metr_la offset=160 | 47 支 | 后备 |

Source/Target 双互斥 cohort（20+20），certified_virgin（Runner 局部过滤，
冻结 UID）。本切片**无 Source Memory 注入**（非 A5/A3——Slow Path 触发与
Memory 无关；如需 Source 由冻结的正常 Source 流程生成）。

## 3. 探测与两步 Workflow 生成（R1）

- 确定性探测（冻结 Fast Path 机制：no-op 前提过滤 + 生命周期修复）；
- **两步 propose**：候选 = A→B（Program.from_steps 两算子，params 用
  contract_params），生成规则 = actionable 顺序的自然两两组合（预注册，
  不挑失败案例）；探测 1 = 第一组合，探测 2 = 下一组合（预算 ≤2）；
- evaluate(chosen 两步, 792) → gain_AB；
- gain_AB < −M → 触发 Slow Path；预算内无 → NO_NATURAL_FAILURE（停止，
  不换 origin）。

## 4. 反事实与 headroom（Slow Path）

反事实表（4 次确定性评估，origin=792）：
- identity：baseline（ScopeExecutor._baseline，gain=0 定义）；
- A-only、B-only：单步 evaluate；
- A→B：已有 gain_AB。

headroom（审核修订公式）：
```
max(gain_A, gain_B) >= M
且 max(gain_A, gain_B) - gain_AB >= M
```
不满足 → NO_SINGLE_STEP_HEADROOM（如实报告）。

## 5. LLM Typed Patch（冻结）

- agicto gpt-5.6-luna，temperature=0，CountingClient 上限 2（1 + 1
  格式纠正）；
- 合法集 KEEP / REMOVE_A / REMOVE_B / ABSTAIN；
- **信息墙**：LLM 只见反事实 Support 表（identity/A/B/A→B 的 gain）+
  公开 Context；**不见 delayed、不见正确答案、不见 first fault 标注**；
- Patch → Workflow：REMOVE_A → B-only；REMOVE_B → A-only；KEEP → A→B；
  ABSTAIN → 无更新。

## 6. Patch 验证与 Skill 落地

- verifier：evaluate(patch Workflow, 792) passed（可行动）；
- Support replay：gain_patch = evaluate(patch, 792) 的 gain；
  **Patch 有效 = gain_patch >= M**（审核修订）；< M 或 verifier 拒 →
  PATCH_REPLAY_FAILED；
- @840 delayed 验证：evaluate(patch, 840) → 记录（Skill 状态依据）：
  双正 → LOCAL_ACTIVE；冲突 → CONFLICT/RESTRICTED；
- **KEEP/ABSTAIN 不形成 Skill** → ABSTAIN_NO_UPDATE（安全行为）/
  PATCH_REPLAY_FAILED（KEEP 未改善）；
- 成立 → 写 Target-local Skill（fork + learned skill + 生命周期状态
  同步，revision 递增）；
- **R2 @888 采用验证**：本臂 fork 快照 + 正常入口 prepare →
  chosen=cand_skill_* **且实际 evaluate(chosen, 888)**（不是池首位）；
- @936 delayed 更新（Skill 的 delayed utility → 状态保持/降级）。

## 7. 预注册 verdict（六档）

- `NATURAL_SLOW_PATH_UPDATE_PASS`：自然 Support 失败 → headroom →
  LLM Patch（REMOVE_A/B）→ gain_patch >= M → Skill 形成 → R2 实际
  选择并执行 → delayed 结果记录；
- `NO_NATURAL_FAILURE`：R1 预算内无 gain < −M（可信负结果，关闭本切片）；
- `NO_SINGLE_STEP_HEADROOM`：有失败但 headroom 公式不满足；
- `PATCH_REPLAY_FAILED`：Patch 后 verifier 拒或 gain_patch < M（含
  KEEP 未改善）；
- `ABSTAIN_NO_UPDATE`：LLM 合理 ABSTAIN（证据不足——安全行为，不伪装）；
- `INCONCLUSIVE`：接口/格式/调用失败。

## 8. 边界（不称）

- 本切片验证"自然失败 → LLM 归因 → 可执行有效 Harness Update"；
- 不称 Shared Capability / 普遍跨域迁移；不是 A5/A3 比较；
- 不新增 Schema/Gate/Memory 平台；不调 radius/Pattern；不改全局
  renderer；
- delayed CONFLICT 触发留作独立切片（本切片只 Support NEGATIVE）；
- 若 R1 无自然失败，负结果直接关闭本切片（不换 origin/不构造）。

## 9. 运行

```
python evaluation/functional/run_v1_natural_slow_path.py
```

（单一 Runner，一次性运行，如实接受六档 verdict。）
