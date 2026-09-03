# T2 — TaskSpec / Consumer 观察接线审计 + 一个观察补丁

- 判定: **TASK_CONTEXT_GAP_PATCHED**
- 证据等级: DEVELOPMENT_STATIC_AUDIT(纯静态审计 + 确定性提示词物化,不调用任何 backend)
- 成本: LLM 0/0,forecasting 重训 0/0,AD 评估 0/0
- Part 0 检查点: `be02ab2`(5 文件:#36 T1 交付物 + 台账/路线图修订)

## Part A 审计汇总(A5)

| 面 | 状态 | 指针 |
|---|---|---|
| A1 公开视图 task/consumer/评价语义 | **缺口——本轮已修** | `_base_input` 无 task 字段;`task_kind="forecast"` 硬编码于特征提取(ssi:632, wvc:614)且被 OBSERVATION_FIELDS(wvc:183-195)丢弃;consumer 只有 variant 名+结构描述 |
| A2 卡适用条件 task 维度 | 已接线 | `observable_applicability={task_kind == "forecast"}`(ssi:354-356);词汇域含三任务(observables.py:67-71) |
| A2 卡适用条件 consumer 维度 | 缺口——留 T4 | 闭词汇无 consumer 特征(ssi:350-353 注释明说);只能以 WHEN 文本表达 |
| A3 Episode task/consumer 键 | 键已接线;键构成缺 task 分量——留 T4(书指定) | `task_consumer_key` 必填(experience_memory.py:58);recipe 线写 `batch:<cohort>|consumer:<variant>`(run_e2_fresh_confirmation.py:1765) |
| A4 检索按 task 过滤能力 | 已接线 | skill 层 applicability AST(retrieval.py:177-181);episode 层 `task_consumer_key` 精确匹配(experience_memory.py:214)+ 合法性过滤(fast_agent.py:836-845) |

## Part B 补丁(一个机制,一个文件)

`evaluation/functional/run_e2_skill_store_integration.py`,**+49/-0 纯增量**:

- `_base_input` 新增 `task_spec` 字段,内容恰三项:`task_id` / `consumer_id` / `quality_semantics`;
- 值由 runner 确定性注入(`_task_spec_observation`,按 `consumer_variant` 查表;未知 variant 硬报错),不经 LLM,不带任何 outcome;
- `task_spec_override` 仅供本轮确定性物化构造 anomaly_detection 变体,无 live caller 使用。
- sha256: 前 `37d31cb8…`, 后 `f39c13f3…`。该文件是 FROZEN_SURFACE_V9 成员——本书唯一授权的 Observation 面补丁;注册表 hash 更新留下一检查点(沿 #18/#19 "修复后冻结"先例)。

字段值(forecasting/pooled 实注):

```json
{"task_id": "forecasting", "consumer_id": "pooled_ridge_a1",
 "quality_semantics": "good preparation lowers the sMASE of the evaluation-series forecasts"}
```

## B3 确定性提示词物化(0 LLM)

物化路径 = 真实装配路径:`PublicAgentInput.create → TTHAAgentCore._messages`,真实 h0 snapshot 用该 episode 录档的 context 特征解析。输入帧取自已交付工件 `skill_store_integration_v1.json` 的 `arm_targets[0]`(T233,已曝光开发区,只读)。

| 物化 | user sha256 | 字节数 |
|---|---|---|
| V_old(无字段,等价 #36 前管线) | `1992c196…` | 见 json |
| V_forecasting | `4931dff8…` | 见 json |
| V_anomaly_detection | `8b4c5eef…` | 见 json |

三项验收全过:

1. **向后兼容**:补丁版 `_base_input` 作用于真实 episode 重构输入,剥掉 `task_spec` 后 canonical sha = `eb552e3a…`,与该 episode 录档的修改前 `public_input_sha256` **逐位相等**;added_keys=[task_spec],removed/changed 皆空。
2. **旧→新 diff 恰为新增字段本身**:user 字节唯一差异是插入的 `,"task_spec":{...}` 段(样本入 json)。
3. **forecasting→AD diff 恰为字段值之差**;system 消息三份逐字节相同。

## 冻结面与纪律

- FROZEN_SURFACE_V9(40 原始 / 39 去重):除授权补丁文件外零漂移(`git diff --name-only` 仅 ssi 一文件)。
- NOAA 2025 / beyond_17520 / SMD test+labels 零读取;AD 参数(49/3.5)与注入工件只读未动。
- 交付不 commit(Part 0 除外);未 spawn;另一线停笔。

## 歧义(如实上报)

1. `quality_semantics` 用英文书写(提示词全英文),内容为书定三项,无 outcome。
2. recipe 线 consumer_id 名(pooled_ridge_a1 / per_channel_ridge_a1)由本补丁注册;书只给了 pooled_ridge_a1 与 ad_v1_49_35 示例,per_channel 无书级名称。
3. base 体 `schema_version` 保持 `skill-store-integration-input/1`:提版本号会让 B3"diff 恰为新增字段"验收失败;该新增向后兼容(无消费方拒绝未知键)。
4. anomaly_detection 变体经 `task_spec_override` 物化——当前无 live caller 服务该任务;T3 是否把真实 AD caller 接进此字段是 T3 的决定。
