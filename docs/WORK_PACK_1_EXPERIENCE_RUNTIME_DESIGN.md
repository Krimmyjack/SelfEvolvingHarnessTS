# 工作包 1：Experience Runtime 最小接线设计（deepseek 副本）

日期：2026-08-06
范围：`SelfEvolvingHarnessTS-deepseek` 副本内实现；原项目（Codex 工作区）不受影响。
依据：审核稿（idea/SelfEvolvingHarnessTS_Framework_Design_and_Experiment_Plan_2026-08-06.md）§5.2/§11 + 评议（最小有效自进化框架）。

## 1. 目标与通过条件（预注册）

本工作包回答一个问题：**v6 已产生的自然 Action–Response（NN5 正 / GEFCom 负 / NOAA 冲突）能否进入下一轮 Fast Path 可检索的运行时经验，而不是停在报告里（staged_memory=[]）？**

通过条件（全部满足才算完成）：
1. NN5 正向、GEFCom 负向、NOAA 冲突都能写入 Episode（含 relation/validity 标注）；
2. Memory-off 与 Memory-on 的 Fast Path 检索结果不同（Plan 输入发生变化）；
3. 检索结果包含**对照包**：最相似的正向 + 最相似的负向 + 冲突提示，而非只给成功案例；
4. GEFCom/NOAA 的有害/冲突经验**不获得执行权**（local_status 不达 LOCAL_ACTIVE）；
5. stale active 状态被 restriction/rejection 覆盖（CurrentHarnessState 的覆盖规则）；
6. Retriever 不使用 outcome 或 Query future（只用公开 Context 特征做匹配）；
7. 私有字段（dataset_id/series_uid/path 等）不进 Episode（复用 v6 `_contrast_episode` 的 forbidden 检查）。

## 2. 数据来源（已暴露轨迹，不产生新科学结论）

| Episode | 来源 | 预期 relation |
|---|---|---|
| A | v6 generation 报告（NN5）`autonomous_natural_acquisition_cycle_v6_report.json` 与 run 轨迹 | POSITIVE（Support 正） |
| B | v6 generation 报告（GEFCom）同源 | NEGATIVE（Support 负） |
| C | NOAA（Support 正、Delayed 负）`historical_policy_episode_workflow_target_local_v2_rejected.json` | CONFLICT |

v6 报告 stages 内含 `dossier.allowed_public_context_scalar_fields` 与 calls 记录；`_contrast_episode`（run_e2_autonomous_natural_workflow_generation.py:995）已实现 episode 构造 + 私有字段检查——直接复用其结构。

## 3. ExperienceEpisode 最小字段

```yaml
experience_episode:
  episode_id: str            # e.g. "nn5_support_pos_v6"
  schema_version: "experience-episode/1"
  task_consumer_key: "forecast|ridge_smase"
  domain_namespace: str      # 仅作局部先验 namespace，不参与跨域匹配
  context_summary:           # 轻量可观察特征（检索键）
    cohort: {...}            # 序列数/覆盖率等聚合
    local_pattern: {...}     # 周期/缺失/异常特征
    program_geometry: {...}  # 作用区间/绑定
  workflow_signature: str    # 算子序列的确定性指纹
  support_response:
    gain: float | null
    accepted: bool | null
  delayed_response:
    gain: float | null
    evaluated: bool
  relation: POSITIVE | NEGATIVE | CONFLICT | ABSTAIN
  evidence_level: SUPPORT | FULL_POLICY | DELAYED
  response_validity: VALID | INSTRUMENT_INVALID   # 仪器故障不算负向经验
  local_status: EPISODE_ONLY | LOCAL_DRAFT | LOCAL_ACTIVE | RESTRICTED
  evidence_refs: [str]       # 完整 trace 引用（不复制全文）
```

近期只启用 4 个 local_status（评议 §一.1）；`SHARED_CANDIDATE/SHARED_ACTIVE` 跨域实验时再启用。

## 4. Signed Retrieval（确定性四步，不训练 Retriever）

1. **硬过滤**：task + consumer + 合法 operator 前置条件（复用 `evaluate_applicability` 的 AST 评估）；
2. **同域分开**：同 domain_namespace 的 Local Skill / Episode 与跨域 Episode 分开检索（同域优先）；
3. **轻量 Context 距离**：cohort summary / local pattern / program geometry 三个维度的简单数值距离（L1 或分箱匹配）；
4. **返回对照包**（最多 3 条）：
   - 1 个最相似 POSITIVE；
   - 1 个最相似 NEGATIVE；
   - 1 个最相似 CONFLICT（若有）；
   - 附带 `evidence_sufficiency` 提示（当前证据是否足够、是否需 abstain）。

Fast Path 得到的是"最相似的成功是什么 / 最相似的失败是什么 / 哪里冲突 / 证据是否够"，不是一堆成功案例。

## 5. CurrentHarnessState（当前视图，不清理历史 artifact）

```yaml
current_state:
  local_skills: [...]
  restrictions: [...]
  rejected_bets: [...]
  latest_status_by_skill: {...}
  schema_version: "current-harness-state/1"
```

覆盖规则：`RESTRICTED/REJECTED 覆盖旧 ACTIVE`。历史文件保持只读；Fast Path 只读 CurrentHarnessState。

## 6. Fast Path 接线点

- 现有：`resolve_harness_view(snapshot, public_features, role)`（methods/ttha/retrieval.py:139）→ `EffectiveHarnessView`（skills/memories/controls）；
- 新增：`resolve_experience_contrast_pack(episodes, public_features, task_consumer_key)`——在 Fast Path LLM prompt 组装处注入对照包（不改变 EffectiveHarnessView 结构，作为附加字段 `experience_contrast_pack`）；
- Fast Path 读取：LLM prompt 的 Context 部分增加 episode refs（引用，不复制全文）。

## 7. 验证方法（机制重放，零新 LLM 调用）

Memory-off vs Memory-on 对照：
- Memory-off：不注入对照包，跑 Fast Path 检索（现有行为基线）；
- Memory-on：注入 NN5/GEFCom/NOAA 三 episode 对照包，跑同一检索；
- 断言：① 检索输出不同；② episode refs 被正确引用；③ 有害/冲突经验 local_status 正确（不达 LOCAL_ACTIVE）；④ 无私有字段泄漏；⑤ 无 outcome/Query future 使用。

## 8. 明确不做（反过度工程纪律）

- 不建向量数据库 / Embedding / Pattern Graph / 通用 Journal 平台；
- 不新增 Operator / Workspace 工具 / 多任务 / Shared Capability 状态机；
- 不做真实 LLM 调用（纯机制重放，LLM API call = 0）；
- 不修改原项目任何文件；全部新代码在副本内。

## 9. 产出

- 新模块：`methods/ttha/experience_memory.py`（ExperienceEpisode + Signed Retrieval + CurrentHarnessState）
- 新 runner：`evaluation/functional/run_w1_experience_runtime_replay.py`（机制重放 + Memory-off/on 对照 + 报告）
- 报告：`artifacts/functional/e2/w1_experience_runtime_replay_report.json`
