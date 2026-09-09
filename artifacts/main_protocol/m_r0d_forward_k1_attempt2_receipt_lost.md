# M-R0d · attempt 2 —— 预算已花，收据被我自己的缺陷毁掉

**判词** `RUN_SPENT_NO_RECEIPT__SERIALISATION_DEFECT` · **evidence grade** `INSTRUMENT`
**没有 JSON 收据** —— 这正是本文件要记录的事实。
**入口** `python -m evaluation.main_protocol_p4.run_m_r0d_forward_k1_outer_step --label attempt2`
**传输** `https://api.nowaterapi.xyz/v1`, `gpt-5.6-sol`（`M0_AGENT_*` 覆盖，`source=M0_AGENT_*`）

## 发生了什么

用户提供新凭据后，第二次运行**越过了 attempt 1 卡住的 Slow 传输**，`build()` 正常返回，
然后在**写收据时**抛 `TypeError: Object of type mappingproxy is not JSON serializable`，
进程退出，**文件从未写出**。预算已经花掉，记录没留下。

这是我的实现缺陷，不是方法问题、不是传输问题、不是预算问题。

## 已知与已失

| 项 | 状态 |
| --- | --- |
| Slow 物理调用 | **2 / 2 已用尽**（attempt 1 一次被 403 拒；attempt 2 一次） |
| attempt 2 的 Slow 是否返回了提议 | **几乎可以确定：是**（见下推断），但**内容已失** |
| 实际提议（feature / direction / rationale） | **UNKNOWN —— 已毁** |
| 阈值是否校准、preflight 是否通过、replay 屏结论 | **UNKNOWN —— 已毁** |
| `record_revision` 是否执行、状态是否写入 | **UNKNOWN —— 已毁** |
| Consumer fits 实际花费 | **UNKNOWN**，上界 10（`OuterBudget` 与回调守卫都在花费前拦，未被触发过） |
| 磁盘上的持久状态 | **未改**：ledger 与 replay cache 都在内存，进程退出即消失；无 Skill 激活、无工件写入 |

### 为什么判断 Slow 确实答复了

`mappingproxy` 只可能来自**实时路径**：用同一 `build()` 配脚本化 Slow 跑一次（0 LLM、0 fit），
报告里 `mappingproxy` 计数为 **0**，`json.dumps` 正常。attempt 1（Slow 被 403 拒、
`stage.payload` 为空）也序列化成功。两相对照，attempt 2 报告里多出来的冻结映射只能是
`slow.calls[*]["returned"]` —— 即 `core.run_stage` 校验后返回的 `scope_clause` payload。
**它存在，说明 Slow 返回了一个 scope_clause。它的内容没能写下来。**

这是推断，不是读数；因此上表把提议内容记为 `UNKNOWN`，不写任何猜测的 feature 或 direction。

## 已修

收据写入路径已加固并用**当初杀死它的那个对象形状**验证过：
`report = drafts._plain(report)` 再 `json.dumps(..., default=str)`。
`drafts._plain` 本来就是为这种冻结映射存在的（其 docstring 写明「Scopes that came back out
of an applied EditManifest are frozen -- mappingproxy inside a tuple」），我先前没有把它用在
收据出口上。验证：原始 `dumps` 仍如实报错，加固后正常输出。

同时 attempt 1 的收据以 `--label` 分文件保留，重跑不会覆盖任何既有记录。

## 为什么停在这里

授权是 **2 次真实 Slow 调用**，已用满。再跑一次就是第 3 次，属于明确的数值越界，
不在「普通实现问题自行解决」的范围内，所以停下报告。

重跑所需极少：该步只要 **1 次 Slow 调用 + ≤10 fits**，入口、凭据、加固后的写入路径都已就位，
命令为

```
M0_AGENT_BASE_URL=https://api.nowaterapi.xyz/v1 M0_AGENT_MODEL=gpt-5.6-sol \
CPA_API_KEY=<key> python -m evaluation.main_protocol_p4.run_m_r0d_forward_k1_outer_step \
  --label attempt3
```

## 边界

未改生产代码（仅本入口脚本的收据写出与 label）；未改阈值、预算上限、候选空间、信息墙；
未覆盖任何历史工件；未提交；未进入第二层。
