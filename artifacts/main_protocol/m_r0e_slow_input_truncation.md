# M-R0e · 第 1 步：Slow 实际收到的输入 vs 影子搜索的输入

**stage** `M_R0E_SLOW_INPUT_TRUNCATION` · **成本** 0 LLM、0 Consumer fit
**收据** `artifacts/main_protocol/m_r0e_slow_input_truncation.json`
**入口** `python -m evaluation.main_protocol_p4.audit_m_r0e_slow_input_truncation`

## 零、先更正一处对照口径（非作者复核指出）

`m_r0d_..._attempt3.md` 第五节写「Slow 在看到同样的 100 行后回答
insufficient_public_evidence」。**这句是错的。**

`evaluation/main_protocol_p4/run_hec1.py:1665`：

```python
"evidence_rows": [
    {"features": row.get("features"), "gain": row.get("gain")}
    for row in rows[:60]
],
```

而 `outer_loop._clause_for`（`outer_loop.py:929`）把 `candidate["rows"]` **整份**交给
`tool.clause_from_slow`，其中的 ScopeFit 影子搜索的就是那一整份。**模型读 60 行，影子读
100 行；那次对照的输入并不相同。** 本次审计把截断长度从源码里读出来写进收据，报告因此不会
再与代码脱节。

## 一、这 60 行是什么

去重后的证据是 100 行 = **5 个已处理 unit × 20 条 series**。`rows[:60]` 恰好落在 unit 边界上：

| | 行数 | unit 数 | 哪几个 origin |
| --- | --- | --- | --- |
| 完整 `rows` | 100 | 5 | 1176 / 1896 / 2136 / 2376 / 2616 |
| **模型看到的 `rows[:60]`** | **60** | **3** | 1176 / 1896 / 2136 |
| 被丢掉的 `rows[60:]` | 40 | 2 | 2376 / 2616 |

不是切碎某个 unit，而是**整整两个 unit 从未进入提示**。普查判定的 `adverse_units=5`
里，模型只见到 3 个。

## 二、伤害信号被削弱了多少（可量化）

| | 完整 100 行 | 模型的 60 行 | 丢掉的 40 行 |
| --- | --- | --- | --- |
| 平均 gain | 0.250664 | 0.242122 | 0.263476 |
| 受损行（gain < −0.005） | **13** | **4** | 9 |
| 受损比例 | 13.0% | **6.67%** | 22.5% |
| 最坏单行伤害 | 0.947602 | 0.947602 | 0.553555 |

任务措辞是「damages a few of them past the deployment's risk budget」。模型看到的受损行是
实际的 **4/13**，受损比例减半。最坏的那一行仍在它视野内。

## 三、决定性的一问：那 60 行还找得到可行子句吗

**找得到，而且是同一条。** 在现行 Scope `local_robust_z_peak >= 3.0` 之下，用同一确定性搜索：

| 输入 | Scope 内幸存行 | 枚举三元组 | 可行 | 最优 stump | 目标值 | treated |
| --- | --- | --- | --- | --- | --- | --- |
| 完整 100 行 | 57 | 96 | **8** | `local_robust_z_peak >= 6.0` | 0.823018 | 16 |
| **模型的 60 行** | 27 | 96 | **9** | **`local_robust_z_peak >= 6.0`** | 1.01818 | 9 |
| 丢掉的 40 行 | 30 | 96 | 9 | `estimated_region_start_fraction <= 0.0` | 0.783498 | 6 |

两侧可行特征集合完全相同：`estimated_region_start_fraction`、`local_robust_z_peak`、
`longest_missing_run_fraction`、`missing_fraction`、`period_reliability`。

把完整输入的胜者搬到模型那 60 行上重新校准：**`CALIBRATED`，阈值同样是 6.0**，
treated 9，四条线全过（aggregate 1.01818、harmed 0.0、single-series harm 0.0）。

**判词：`SIGNAL_PRESENT_IN_BOTH__SAME_WINNER`。** 截断削弱了伤害证据的量，但**没有**把可行
信号切掉——模型手里的那份切片，足以推出与全量相同的那条子句。

## 四、这不构成因果

模型**没有**在完整 100 行上被再问一次。本审计只证明「切片里仍有信号」，不能证明
「截断不是弃权的原因」，也不能证明「截断是原因」。要分辨只能真跑第 2 步。

## 五、弃权是模型自己选的，不是运行时兜底

`no_proposal_reason` 取自 `agent_core.AgentStageResult`（`agent_core.py:493`）：Slow 的 edit
阶段被显式告知可以返回 `no_proposal` 信封，`reason_code` 是三选一枚举 ——
`insufficient_public_evidence | no_authorized_minimal_edit | risk_too_high`
（`agent_core.py:205-224`）。`slow_scope_clause_v1` 本身 `required: [scope_clause]`、
`additionalProperties: false`，**没有弃权位**；弃权走的是信封那条路。

所以这不是解析失败被贴了标签：**模型在三个理由里挑了「公开证据不足」。**

## 六、它当时收到的角色说明（K0 快照原文）

`resolve_harness_view(snapshot, {}, role="slow")` 解析出的 `instruction`：

> You are the TTHA **preparation Agent**. Use only public observations, declared tools,
> canonical operator contracts, retrieved Harness content, and the typed output schema.
> Never request or infer clean references, injection metadata, **candidate utility**, or
> private rankings. Inspect before proposing. Supply at most the configured number of
> effect-distinct **PROGRAM** candidates. Select exactly one candidate, including identity.
> Keep modifications local and **abstain when public evidence does not justify a repair**.

与本次外环任务的四处错位（**是待验证的解释，不是已认定的原因**）：

1. 「Never infer **candidate utility**」——而外环给的证据行每一条都带 `gain`，任务就是据此
   把受损 series 排除出去。
2. 「Supply … effect-distinct **PROGRAM** candidates … including **identity**」——外环要的是
   一条 `scope_clause`，不是 program 候选；"identity" 正是弃权形状的那个选项。
3. 「**Inspect before proposing**」——外环只有一次 `stage="edit"` 调用，没有 inspect 阶段可走。
4. 「**abstain when public evidence does not justify a repair**」——观测到的
   `insufficient_public_evidence` 几乎是这句话的逐词回声。

## 七、边界

0 LLM、0 fit：Slow 桩记录候选后返回 `None`，走的正是实况 attempt 3 的
`payload is None → SLOW_ABSTAINED` 分支；replay 屏被换成"一旦被调用就抛异常"的桩，未被触发。
交叉核对通过：候选数 1、`why` 字符串一致、outcome 一致、证据行数 100 —— 与 attempt 3 收据同一步。

未改生产代码，未新增 SHA，未覆盖历史工件，未提交，未触碰后续 unit。
