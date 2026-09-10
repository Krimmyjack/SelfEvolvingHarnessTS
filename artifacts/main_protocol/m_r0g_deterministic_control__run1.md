# M-R0g · 第 3 步：确定性提议者走完整链（run1）

**stage** `M_R0G_DETERMINISTIC_CONTROL` · **收据** `m_r0g_deterministic_control__run1.json`
**成本** **0 次 LLM、10 次 Consumer fits**（上限 20，剩 10 未动用）· `run_fault: null`
**判词** `NOT_APPLICABLE` —— 校准通过、preflight 通过、**真实 replay 屏否决**

## 一、候选与路径

确定性搜索按既定规则自选一个候选，**未告知任何外部信息**：

| 项 | 值 |
| --- | --- |
| 提议者 | `scope_threshold_tool.best_stump`，0 次 LLM，未构造后端/传输/凭据 |
| 选中 | `local_robust_z_peak >= 6.0`（96 个三元组中 8 个可行里的最优） |
| 路径 | 与模型完全同一条：`_clause_for` → `clause_from_slow` → 冻结 bin 边缘校准 → `validate_narrowing` → 真实 replay 屏 |

唯一的差别是**谁命名了这个特征**。校准规则、冻结边缘、拒绝规则、影子记录都没变。

## 二、逐段结果

| 阶段 | 结果 |
| --- | --- |
| 校准 | **CALIBRATED** `z_peak >= 6.0`，treated **16**，aggregate 0.823018，harmed 0.0，最坏单序列伤害 0.0，四条线全过，`unique_widest` |
| preflight | **accepted**（`structurally_narrower_semantics_unchecked`） |
| 合成 Scope | `z_peak>=3.0 ∧ z_peak>=6.0` |
| **真实 replay 屏** | **passed = false，reason = `NOT_APPLICABLE`** |
| 屏的读数 | 5 个 cell 全部重放，**applicable = 0**，violations = **空**，花费 **10 fits** |
| 后续验证 | **NOT_RUN**（授权明写：失败不换候选追正） |

屏的逐 cell 理由全部是 `predicate resolves below the coverage floor`：
treated 依次为 **4 / 1 / 4 / 4 / 3**。

**注意 violations 是空的。** 这个子句不是"有害"被拒，是"在每个 cell 上都够不到可读性下限"。

## 三、真正的阻断点：覆盖下限用了两个不同的分母

这是本轮最重要的发现，且是**精确对齐**的：

* 校准的 `treated = 16` —— 在**汇总的 100 条 bank 行**上数，16 ≥ `MIN_TREATED = 5`，通过。
* 屏的 `treated` —— 在**每个 unit 各自的 20 条 series** 上数：4 + 1 + 4 + 4 + 3 = **16**。

**同一个 16，同一个下限 5，两个分母。** 汇总口径下它是 3.2 倍余量，逐 unit 口径下它每个 cell
都不够。这不是巧合，两处求和恰好相等。

后果是结构性的：外环校准能通过的子句，只要窄到有意义，屏就会以 `NOT_APPLICABLE` 否决。
要同时满足两边，一个跨 5 个 cell 的窄化实际需要 ≥25 条被治疗的 series 且分布均匀——而下限
名义上只有 5。

**这既不是模型的问题，也不是截断的问题。** M-R0e 已排除截断；M-R0f 显示模型换个角色说明就会
提议；本轮显示即使由确定性搜索给出全局最优子句，链条仍在屏这一步断掉。

## 四、覆盖与祖先行为差异（0 fit 附录）

Scope 解析只读部署可见特征卡，不读任何结果、不产生拟合。下表 **0 fits**：

| pos | origin | exposure | face | served | 祖先 `>=3.0` | 修订 `>=6.0` | 是否达下限 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1176 | SPENT_DEV | support | 20 | 9 | 4 | BELOW |
| 0 | 1176 | SPENT_DEV | delayed | 20 | 4 | 2 | BELOW |
| 1 | 1896 | SPENT_DEV | support | 20 | 10 | 1 | BELOW |
| 1 | 1896 | SPENT_DEV | delayed | 20 | 13 | 0 | BELOW |
| 2 | 2136 | SPENT_DEV | support | 20 | 8 | 4 | BELOW |
| 2 | 2136 | SPENT_DEV | delayed | 20 | 5 | 1 | BELOW |
| 3 | 2376 | SPENT_DEV | support | 20 | 18 | 4 | BELOW |
| 3 | 2376 | SPENT_DEV | delayed | 20 | 10 | 2 | BELOW |
| 4 | 2616 | SPENT_DEV | support | 20 | 12 | 3 | BELOW |
| 4 | 2616 | SPENT_DEV | delayed | 20 | 13 | 6 | OK |
| **6** | **3576** | **UNREAD** | **support** | **20** | **20** | **20** | OK |
| **6** | **3576** | **UNREAD** | **delayed** | **20** | **20** | **20** | OK |

### 目标验证单元上，两个 Scope 行为完全相同

p06 `[0:40]@3576` 的**全部 20 条 series** 同时满足 `z_peak>=3.0` 和 `z_peak>=6.0`。
两个谓词在该 unit 的两个面上解析出**同一个集合**，程序对同一批 series 生效。

**因此，即使屏放行、验证真的跑起来，那次验证也只能得出"行为相同"。**
按预先讲好的口径：**这不能算作修订收益**，`revision_delta` 会是精确的 0，与祖先的差异为 0 条
series。这不是推测——覆盖是解析出来的确定量，与结果无关。

真正有行为差异的是那 5 个已处理 cell（祖先 9/10/8/18/12 → 修订 4/1/4/4/3），而那正是屏判
`NOT_APPLICABLE` 的地方。

## 五、按要求的四项输出

| 要求 | 结果 |
| --- | --- |
| **实际收益** | **未测得。** 屏否决，验证未跑；剩余 10 fits 未动用（授权：失败不换候选追正） |
| **伤害** | 屏的 `violations` **为空**——本候选未在任何 cell 上触发风险线；被拒理由是覆盖，不是伤害 |
| **覆盖** | 见上表。修订后逐 cell 4/1/4/4/3（下限 5），仅 2616 的 delayed 面达标；3576 上 20/20 |
| **与祖先的行为差异** | 5 个已处理 cell 上有差异（严格子集）；**目标验证单元 3576 上为 0 条 series 差异，行为相同** |

## 六、状态与边界

| 项 | 前 | 后 |
| --- | --- | --- |
| revisions | 0 | **0**（屏未放行，`record_revision` 未到达） |
| current_scope | `z_peak>=3.0` | **未变** |
| clauses_added_since_root | 0 | **0** |
| state | REVISABLE | **REVISABLE** |
| Draft 数 | 1 | **1（无第二壳）** |
| deployable | false | **false** |

`skills_activated=0`、`deployment_rights_issued=0`、`stores_written=0`、`snapshots_minted=0`、
`evaluation_face_reads=0`、`held_out_reads=0`、`sealed_reads=0`、`thresholds_changed=0`、
`production_code_changed=0`。未进入长课程。

**一处收据字段更正**：run1 的 JSON 里 `boundary.later_units_touched` 写成 `1`，但验证并未运行，
该字段是无条件写死的。已把源码改为按验证状态派生（`RUN` 才记 1），本收据不覆盖重写。
实际情况：3576 只被构造了 `UnitContext` 并解析了谓词（读部署可见特征卡，**0 fits，未读任何结果、
未读 delayed 真值、未读 +144**），第四节那张表就是这一步的全部产物。

## 七、成本

| 项 | 本包 | 上限 |
| --- | --- | --- |
| LLM 调用 | **0** | 0 |
| Consumer fits 合计 | **10** | 20 |
| ——replay 屏 | 10（5 cell × 2） | |
| ——验证 | **0**（未运行） | |
| 0-fit 附录（覆盖表） | 0 | |
| 在上限内 | **是** | |

---

## 八、更正（追加，不改动上文原句）

非作者复核指出两处推论错误，均成立。上文原文保留留痕。

**1. 第三节「实际需要 ≥25 条」不成立。** 屏的条件是 `outer_loop.py:895`
`"passed": not rejected and applicable > 0` —— **一个** cell 适用即可，不要求五个都达标。
已确立的事实只是：**被试的这一个候选在五个历史 cell 中没有一个适用**；不能推出整个 Scope
空间被封死。第三节标题「覆盖下限用了两个不同的分母」这一观察本身仍成立（16 = 4+1+4+4+3 的
对齐是真的），但由它推出的必要条件是错的。

**2. 第五节「伤害：violations 为空——本候选未在任何 cell 上触发风险线」口径错误。**
五个 cell 都在 `_applicable` 处被跳过，`_violations` 从未运行。全跳过的屏返回空 violations，
含义是 **NOT_MEASURED，不是无伤害**。正确表述：**本轮未取得任何伤害读数。**

**机制的准确表述**（替代第三节的因果说法）：校准器在汇总证据上寻找好候选，却没有提前考虑该
候选能否在实际评价单元上获得有效验证。汇总支持量门与单元覆盖门可以同时正确；未对齐的是
候选选择目标与后续验证资格。

后续的 0 成本全空间核查见 `m_r0h_candidate_eligibility.{json,md}`：5 个候选里 **4 个**
screen/verify/contrast 三列全过，唯一不合格的正是影子按目标函数选中的
`local_robust_z_peak >= 6.0`。
