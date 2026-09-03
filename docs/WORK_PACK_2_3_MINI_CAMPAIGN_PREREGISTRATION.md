# 工作包 2+3：3-Domain Mini-Campaign 预注册（deepseek 副本）

日期：2026-08-06
范围：`SelfEvolvingHarnessTS-deepseek` 副本。
依据：审核稿 §12-14 + 评议（工作包 2/3、通过条件、六个停止条件、预算口径）。
状态：**预注册——以下内容在 Campaign 开始前冻结，跑之前不再修改。**

## 1. 研究问题

两个顺序问题（不同时检验）：
1. **同域效果**：Signed Experience（正/负/冲突对照包）是否让下一批同域数据**更少试错**（Adaptation AUC 更高、首次正向所需 Probe 更少、harm 不增）？
2. **Slow-Path 效果**（问题 1 出现真实失败 cluster 后）：一个 LLM Typed Patch 能否在自然 held-in 数据上被接纳（改变行为、改善效用、不回归）？

## 2. 数据组织（3 Domain × A/B/C 三批）

| Domain 角色 | 选择标准 | A（经验产生） | B（同域验证） | C（delayed 评估） |
|---|---|---|---|---|
| D1 | 已有局部正向迹象（NN5 族） | 产生 Episode | 检索复用 → 形成 LOCAL_ACTIVE 候选 | 未参与选择的后续数据评估 |
| D2 | 已有负向迹象（GEFCom 族） | 产生 Episode | 有害经验不获执行权 | 同上 |
| D3 | 已有 Support-Delayed 冲突（tourism/NOAA 族） | 产生 Episode | CONFLICT 进入对照包 | 同上 |

> **Amendment-001（2026-08-06，预注册后）**：工作包 1 实现时发现 v6 报告只含 A(NN5)/B(GEFCom) 两个 environment（B 的 support_gain=+0.003 为弱正，不是负）；且 v6 报告不含 NOAA。实现采用**真实 RESTRICTED/REJECTED 证据**替代占位描述：
> - NEGATIVE 源 = `historical_policy_episode_workflow_target_local_v2_rejected.json`（status=REJECTED，Target 验证拒绝）——语义上比 GEFCom 弱正更符合"负向"；
> - CONFLICT 源 = `historical_policy_episode_workflow_state_update_e288.json`（v1 曾 ACTIVE → `POSITIVE_SUPPORT_FALSE_CONFIRMED_HARM_ON_FRESH_TARGET` → RESTRICTED）——教科书式"Support 正但 fresh Target 有害"冲突；
> - 三个源报告均已核实存在、字段吻合。Campaign 阶段的 D2/D3 Domain 选择仍按本节原始标准（负向迹象 / 冲突迹象）从数据族中确定，不受本 amendment 影响。

- 三批按**时间顺序**组织（chronological），不随机打散——保留"经验先发生、后续数据再到达"的因果顺序；
- **fresh 对齐**：A3 baseline 与 L± 跑在同一批新 chronological origins 上（不用旧缓存）；
- 不使用 clean repair truth（自然数据无真值），只用可见 Context、Support、delayed downstream feedback。

> **Amendment-002（2026-08-06，预注册后）——评估冻结纪律（用户补充）**：
> - **一轮的定义**：`A（adapt：经验写入 + harness 更新）→ B/C（冻结评估：用 adapt 后的 harness 在同域其他数据上跑一遍，评估效果）`。B/C 段**运行期间 Slow Path 关闭、harness 完全冻结**——不触发任何 slow_path 更新；
> - **B/C 段使用 adapt 后的 harness**：即 B 的检索/执行基于 A 段写入的经验与形成的 Local Skill（不是旧 harness 重跑）；
> - **Slow Path 更新只发生在轮与轮之间**：一轮的 B/C 全部评估完 → 若出现失败 cluster → 才进入工作包 3（一次 Typed Patch）→ 更新后的 harness 进入下一轮 A'；
> - **效果归因**：B/C 段观察到的效果（Adaptation AUC / delayed utility / harm）全部归因于本轮 A 段的 adapt（经验与 harness 更新），与 slow_path 更新严格隔离——这是"边评估边进化导致无法归因"的防护；
> - 通过条件（§6）与停止条件（§7）均按"一轮"为单位判定。

## 3. Arm 定义

| Arm | 含义 | Memory 输入 |
|---|---|---|
| `A3` | Memory off，从头适应 | 无 |
| `L+` | 只读正向 Episode | 仅 POSITIVE 对照包 |
| `L±` | 正/负/冲突全部 | 完整 Signed 对照包 |
| `MISMATCH` | 等数量、等 token，但 Context 错配的 Episode | 错配对照包（混淆控制） |

- `A0/Identity` 只作 Utility reference，不参与适配预算；
- `A4` 暂时为空（无 SHARED_ACTIVE，不凑 Arm）。

## 4. 预算（冻结）

- Probe 预算：`B ∈ {0, 1, 2}`（**不扩到 B=4**）；
- 每轮最多：2 个候选 Workflow + 1 次 revision + 2 次完整 Support policy evaluation；
- **预算口径（分开记账，不对齐混合）**：
  - `llm_token_budget`：LLM 调用 token 数（A3 若用确定性流程记 0）；
  - `consumer_fit_budget`：完整 Consumer fit 次数；
  - `wall_clock_budget`：墙钟时间上限。
  - 报告分别记录三项；"等预算"比较 = 同 `consumer_fit_budget` 下的比较（Primary），llm_token 与 wall_clock 作 Secondary 报告。
- Proxy（LTSV/TimeInf-like）只用于候选定位，不参与效用裁决。

## 5. 主指标

- `Adaptation AUC`（Primary）：适应轨迹曲线下面积；
- Batch C delayed utility；
- 首次正向 Workflow 所需 Probe 数；
- harm count / magnitude（delayed 段）；
- abstention 率（不能靠几乎全 abstain 过关）；
- action coverage（Program/Binding 是否真实改变）；
- Episode 引用率（LLM 最终 Workflow 必须引用并利用检索到的 Episode，而非只复述）。

## 6. 通过条件（功能性门槛，非论文级）

1. `L±` 的 domain-macro Adaptation AUC 优于 `A3`；
2. 至少 2/3 Domain 上 `L±` 不差于 `A3`，且至少一个 Domain 明显改善；
3. delayed harm 不高于 `A3`；
4. 改善不能来自几乎全 abstain；
5. `MISMATCH` 不能取得相同效果（否则不能声称 Context-conditioned Memory 有效）；
6. LLM 的最终 Workflow 引用并利用检索到的 Episode。

**上述全部满足 → 扩展到 6–8 Panel 正式 Campaign。任何一条不满足 → 按第 7 节停止条件处理，绝不直接扩规模。**

## 7. 停止条件（预注册，触发即执行对应动作）

| # | 条件 | 动作 |
|---|---|---|
| S1 | Memory 改变 Plan 但 Utility 不升 | 只修改 Retrieval/summary，不增加 lifecycle 状态 |
| S2 | `L±` 不优于 `L+` | 不扩建 Signed Memory；保留 positive-only 或更简单 contraindication |
| S3 | `MISMATCH` 与正确 Retrieval 同样有效 | 不能声称 Context-conditioned Memory 有效；检查 prompt 变长/同域身份混淆 |
| S4 | LLM Patch 不优于 deterministic template | Slow Path 先用确定性 updater，LLM 提议降级 |
| S5 | 同域 Mini-Campaign 整体无效 | 不增加 Domain、不进入 A5；更换自然任务载体（cross-series curation / forecast-context preparation / group-level historical segment selection） |
| S6 | 同域有效、跨域无效 | v1 成立、Shared Capability 里程碑失败——如实报告，不硬凑跨域 |

## 8. 工作包 3（Slow-Path Update，问题 1 出现失败 cluster 后触发）

### 输入（给 Slow Agent）
- 匹配的成功/失败 Episode（对照包）；
- 已控制相同的 Context；
- Program/Binding 差异；
- Support 与 delayed response；
- rejected bets（CurrentHarnessState）；
- 当前允许修改的一个 Surface。

### LLM 必须输出（六字段）
```yaml
first_fault: ...
changed_surface: ...
proposed_change: ...
predicted_affected_contexts: ...
predicted_behavior_change: ...
falsification_condition: ...
```

### 对照
- no patch；
- 一个简单 deterministic template patch；
- LLM typed patch。

### 允许修改的 Surface（冻结，只三选一）
`RESTRICT_SCOPE` / `PATCH_BINDING` / `PATCH_CONTROL`
（暂不做：ADD_OBSERVATION_TOOL / ADD_OPERATOR / 自由 COMPOSE_WORKFLOW）

### 接纳条件（全部满足才接纳）
1. Patch 确实按预测改变行为（predicted_behavior_change 兑现）；
2. B 或新的 held-in slice 上 Utility 改善；
3. 两个已稳定通过的 sentinel Context 不回归；
4. 不是通过把所有 Context 变成 Identity/abstain；
5. Utility 在噪声范围内时标记 `UNRESOLVED`，不扩大执行权（UNRESOLVED 集群出现 → 触发"检查测量力/feedback 层"，见补充条款）。

## 9. 补充条款（预注册）

1. **UNRESOLVED 集群**：若连续 ≥3 个 patch 均为 UNRESOLVED（Utility 在噪声范围），触发检查 feedback 层测量力（E2-J0 MDE 问题），而非无限 KEEP INCUMBENT；
2. **INSTRUMENT_INVALID**：API timeout / Consumer fit crash / compile failure / metric 无效 → 标 `INSTRUMENT_INVALID`，默认不参与 Skill Retrieval，不计入负向经验；
3. **受控回归面**：合成受控层（四大缺陷族）保留为 regression sentinel——自然主实验不用 clean truth，但受控层继续测算子层对错；
4. **预算口径见 §4**；报告同时输出三项预算与 Primary/Secondary 判据。

## 10. 产出（每阶段 1 runner 1 report）

- runner：`evaluation/functional/run_w2_mini_campaign.py`（含 w3 slow-path 子命令或独立 runner）；
- 报告：`artifacts/functional/e2/w2_mini_campaign_report.json`（Arm 对比 + 指标 + 通过条件判定 + 停止条件触发情况）；
- 预注册冻结时间：本文档写入时刻。Campaign 代码实现完成后、任何数据打开前，用户确认一次本文档无异议。
