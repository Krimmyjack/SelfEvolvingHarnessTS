# M-R0d · forward · A5-online · k1 第一层真实验收 —— 运行收据

**stage** `M_R0D_FORWARD_K1_OUTER_STEP_LIVE` · **判词** `RUN_BLOCKED_NO_VERDICT__TRANSPORT_QUOTA`
**evidence grade** `INSTRUMENT`（仪器记录，**不是**方法读数）
**收据** `artifacts/main_protocol/m_r0d_forward_k1_outer_step_live.json`
**入口** `python -m evaluation.main_protocol_p4.run_m_r0d_forward_k1_outer_step`

## 结果一句话

真实 Slow 请求**已发出**，被中转方以 **HTTP 403 `insufficient_quota`** 拒绝；本次运行止于
**Slow 传输**，没有拿到任何提议。按 `_is_transport_failure` → `RunFault` 的既定纪律记为
传输故障，**不形成科学判词**，与 `v11fix_*` 三次的 `RUN_BLOCKED_NO_VERDICT__TRANSPORT_QUOTA`
同类。

## 一、实际提议

**没有。** Slow 未返回任何 payload，`slow_calls_recorded_by_the_agent` 为空 —— 异常在
`core.run_stage` 内抛出，早于 agent 记录提议的那一行。

已知的只是**将要问什么**：候选为 `REVISE`，作用于 `resupplied_draft_1`，
`base_scope = local_robust_z_peak >= 3.0`，证据 100 行（去重后），
`allowed_features` 为冻结的 12 词表。这些在请求里，答案没回来。

## 二、通过或失败在哪一步

| 阶段 | 结果 |
| --- | --- |
| 状态重建（bank / 已处理 cell / Draft / Active 集） | **通过** —— 7 bank 行、5 个已处理 cell、`resupplied_draft_1` = `REVISABLE`/attempts 1、K0 谱系在 held 内 |
| K0 快照编译 | **通过**（第二次尝试；见下「实现问题」） |
| 普查 + `propose_candidates` | **通过** —— `outlier_mad` 组 `adverse_units=5`、`unit_votes {CONFLICT:5}`；发出**唯一**候选 `REVISE`→`resupplied_draft_1` |
| **真实 Slow 调用** | **失败** —— 403 `insufficient_quota` |
| 阈值工具校准 | 未到达 |
| narrowing preflight | 未到达 |
| replay 屏 | 未到达（0 fits） |
| `record_revision` | 未到达 |

**止步点 = Slow 传输**，不是方法、不是生命周期、不是预算。

## 三、状态是否正确写入

**正确：什么都没写。** 失败发生在写状态之前，Draft 保持原样：

| 项 | 前 | 后 |
| --- | --- | --- |
| `revisions` | 0 | **0** |
| `current_scope` | `z_peak>=3.0` | **未变** |
| `verification_attempts` | 1 | **1（未变）** |
| `state` | `REVISABLE` | **REVISABLE（未变）** |
| ledger 内 Draft 数 | 1 | **1（无第二壳）** |
| `deployable` | false | **false** |

`skills_activated = 0`、`later_units_touched = 0`、`held_out_reads = 0`、
`evaluation_face_reads = 0`。**未进入第二层。**

## 四、实际成本

| 项 | 授权上限 | 实际 |
| --- | --- | --- |
| 真实 Slow 物理调用 | 2 | **1**（已发出，被 403 拒绝，无 completion 返回） |
| Consumer fits | 10 | **0**（replay 屏未到达） |
| 记账 `llm_outer` / `llm_fast` | — | 1 / 0 |
| 后端前被拦截 | — | 0 |
| 在上限内 | — | **是** |

**剩余授权：1 次 Slow 调用、10 次 Consumer fits，未动用。** 我没有拿第二次调用去重试同一个
账户级配额错误——那几乎必然复现同样的 403，只会把剩余额度烧掉；也没有更换密钥或中转方，
那属于实质性边界变更，需另行批准。

## 五、途中的实现问题（自行解决，未越界）

首次运行在 K0 快照编译处抛 `PermissionError`：Phase S 收据里的
`store_root = .hec1_runs/phase_s_v11_live/A5-online/store_online/<sha>` 在本机不可读。
forward 课程工件自己记录的 `k0_snapshot.store_root = .hec1_runs/v11fix_chain/k0_store`
携带**同一个 `runtime_bundle_sha`**（`98dea3b0…63b`）且可读——那也正是 Phase T 当时编译的
那份。入口改为优先用课程工件记录的路径，并在两者 sha 不一致时直接 BLOCKED 拒跑。
该次失败发生在任何调用与拟合之前，**成本 0**。

另在真实发车前先用脚本化 Slow + 零 fit 屏做了一次**排练**（0 LLM、0 fit），确认接线完好：
候选发出 → `DRAFT_REVISED` → `revisions 0→1`、`attempts` 不变、无第二壳，且工具按设计忽略了
Slow 给的阈值 4.25、改用冻结 bin 边缘 6.0。真实运行与排练走同一条代码路径。

## 六、代码状态

相对 `d690850`：`hec1_contract.py`、`scope_threshold_tool.py`、`scoped_serving_evaluator.py`、
`admission_policy.py` **逐字节相同** —— 即**阈值、候选空间、准入规则未改**。
`run_hec1.py`、`outer_loop.py`、`restricted_draft.py`、`online_loop.py`、`scope_executor.py`
带此前已复核的接线补修，这正是本次要验收的路径。本轮**未再改动生产代码**。

## 七、新增产物

| 文件 | 性质 |
| --- | --- |
| `evaluation/main_protocol_p4/run_m_r0d_forward_k1_outer_step.py` | 新增执行入口 |
| `artifacts/main_protocol/m_r0d_forward_k1_outer_step_live.json` | 新增运行收据 |
| `artifacts/main_protocol/m_r0d_forward_k1_outer_step_live.md` | 本文件 |

未覆盖任何历史工件，未提交，未发车第二层。
