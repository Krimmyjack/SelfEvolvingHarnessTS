# M-R0d · forward · A5-online · k1 第一层真实验收 —— 完成（attempt 3）

**stage** `M_R0D_FORWARD_K1_OUTER_STEP_LIVE` · **run_label** `attempt3`
**判词** `SLOW_ABSTAINED` · **evidence grade** `INSTRUMENT / MECHANISM`（n=1，非能力判词）
**收据** `artifacts/main_protocol/m_r0d_forward_k1_outer_step_live__attempt3.json`
**传输** `https://api.nowaterapi.xyz/v1` · `gpt-5.6-sol` · `source=M0_AGENT_*`
**用时** 19:50:32 → 19:50:54（22 秒）

## 结果一句话

修好的接线**把候选送到了 Slow**，Slow 被问到后**弃权**：
`no_proposal_reason = "insufficient_public_evidence"`，`returned = null`。
候选结论 `SLOW_ABSTAINED`，无子句、无修订、无状态写入。

## 一、实际提议

**Slow 未提出任何子句。**

| 项 | 值 |
| --- | --- |
| 被问的候选 | `REVISE` → `resupplied_draft_1` |
| 送进去的证据 | 100 条 bank 行（精确去重后），`base_scope = local_robust_z_peak >= 3.0`，冻结 12 词表 |
| Slow 返回 | `null` |
| Slow 给的理由 | `insufficient_public_evidence` |
| 影子记录 | 0 条（`clause_from_slow` 未被调用——payload 为 None 即直接弃权） |

这是一次**合法弃权**，不是解析失败、不是协议错误、不是预算耗尽。按 AGENTS §4，
abstain 是 Agent 的合法动作，照收录、不发执行权。

## 二、通过或失败在哪一步

| 阶段 | 结果 |
| --- | --- |
| 状态重建（bank 7 行 / 5 已处理 cell / Draft / Active 集） | **通过** |
| K0 快照编译 | **通过** |
| 普查 | **通过** —— `outlier_mad` 组 `adverse_units=5`、`unit_votes {CONFLICT:5}`、5 观测（丢弃 1 条精确重复）；`impute_linear` 组 `{NEGATIVE:1}` |
| `propose_candidates` | **通过** —— 发出**唯一**候选 `REVISE` → `resupplied_draft_1`（不新壳） |
| **真实 Slow 调用** | **到达并返回；Slow 弃权** |
| 阈值工具校准 | 未到达（无子句可校准） |
| narrowing preflight | 未到达 |
| replay 屏 | 未到达（0 fits） |
| `record_revision` | 未到达 |

**止步点 = Slow 提议本身**，不是接线、不是生命周期、不是预算、不是传输。

这一点是本次运行最重要的信息：M-R0 认定的历史首因（普查关系词表错配 → 0 候选）**已经不再是
阻断点**；候选确实产生并送达。新的止步点前移到了「Slow 愿不愿意提」。

## 三、状态是否正确写入

**正确：什么都没写，因为什么都没提。**

| 项 | 前 | 后 |
| --- | --- | --- |
| `revisions` | 0 | **0** |
| `current_scope` | `z_peak>=3.0` | **未变** |
| `clauses_added_since_root` | 0 | **0** |
| `verification_attempts` | 1 | **1（未变）** |
| `state` | `REVISABLE` | **REVISABLE（未变）** |
| ledger 内 Draft 数 | 1 | **1（无第二壳）** |
| `revision_history` | [] | **[]** |
| `deployable` | false | **false** |

`skills_activated=0`、`later_units_touched=0`、`held_out_reads=0`、`sealed_reads=0`、
`evaluation_face_reads=0`、`thresholds_changed=0`。**未进入第二层。**

## 四、实际成本

| 项 | 本次 | 授权上限 |
| --- | --- | --- |
| 真实 Slow 物理调用 | **1** | 2（本轮重跑授权） |
| Consumer fits | **0** | 10 |
| 记账 `llm_outer` / `llm_fast` | 1 / 0 | — |
| replay 缓存 physical_fits | 0 | — |
| 后端前被拦截 | 0 | — |
| 在上限内 | **是** | |

弃权使 `_clause_for` 立即返回，所以第 2 次调用没有发生；replay 屏在候选拿到子句后才会跑，
因此 0 fits。**剩余额度：1 次 Slow、10 fits 未动用。**

## 五、一个值得记录的对照（不是判词）

同一批证据上，工具自己的确定性影子搜索（M-R0c，0 fit）找到 **8 个可行 stump**，
最优为 `local_robust_z_peak >= 6.0`；而 Slow 在看到同样的 100 行后回答
`insufficient_public_evidence`。

搜索空间**不空**，模型仍选择不提。这正是影子对照存在的目的所在，但
**n = 1、单模型、单步**，只是一次记录，不构成任何关于「LLM 是否有增量贡献」的结论。

## 六、代码状态

相对 `d690850`：`hec1_contract.py`、`scope_threshold_tool.py`、`scoped_serving_evaluator.py`、
`admission_policy.py` 逐字节相同 —— **阈值、候选空间、准入规则未改**。带补修的 5 个文件即本次
验收路径。本轮生产代码零改动；只加固了本入口脚本的收据写出（`drafts._plain` + `default=str`）
与 `--label` 分文件，这正是 attempt 2 丢失记录的修复。

## 七、三次尝试的账

| 尝试 | 结果 | Slow 调用 | fits | 收据 |
| --- | --- | --- | --- | --- |
| 1 | `TRANSPORT_QUOTA`：403 insufficient_quota | 1 | 0 | 有（`..._live.json`） |
| 2 | `RUN_SPENT_NO_RECEIPT__SERIALISATION_DEFECT`：越过 Slow，写收据时崩，记录尽失 | 1 | UNKNOWN ≤10 | **无**（见 `..._attempt2_receipt_lost.md`） |
| 3 | **`SLOW_ABSTAINED`：完成** | 1 | 0 | 有（本文件配套 JSON） |

attempt 2 的提议内容仍为 `UNKNOWN`，不因本次结果而追认——两次是不同的调用，
attempt 3 的弃权不能当作 attempt 2 也弃权的证据。

## 八、新增产物

| 文件 | 性质 |
| --- | --- |
| `artifacts/main_protocol/m_r0d_forward_k1_outer_step_live__attempt3.json` | 新增运行收据 |
| `artifacts/main_protocol/m_r0d_forward_k1_outer_step_live__attempt3.md` | 本文件 |

未覆盖任何历史工件（attempt 1 的收据与 attempt 2 的损失记录原样保留），未提交，未发车第二层。

---

## 九、更正（追加，不改动上文原句）

非作者复核指出：**第五节的对照口径是错的**。上文原句「Slow 在看到同样的 100 行后回答
`insufficient_public_evidence`」保留在原处以留痕，但它不成立。

`run_hec1.py:1665` 只把 **前 60 行**放进提示（`for row in rows[:60]`），而
`outer_loop.py:929` 把 `candidate["rows"]` **整份 100 行**交给影子搜索。**两侧输入不同，
第五节那一句不能当作「同样证据下模型不愿意提」的记录。**

第 1 步复核（0 LLM、0 fit，`m_r0e_slow_input_truncation.{json,md}`）的结果是：
那 60 行 = 5 个 unit 中的前 3 个，受损行 4/13、受损比例 6.67%（全量 13.0%）；但
确定性搜索在这 60 行上仍找到 **9 条可行三元组**，最优 stump 与全量**同为**
`local_robust_z_peak >= 6.0`，搬过去重校准同样 `CALIBRATED` 于 6.0。

即：**截断削弱了伤害证据的量，但没有切掉可行信号**。这既不证明截断导致弃权，也不证明它无关；
模型未在完整 100 行上被再问一次。本节其余内容（提议、止步点、状态、成本）不受影响。
