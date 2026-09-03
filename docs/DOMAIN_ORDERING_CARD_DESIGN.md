# Domain Ordering Card — 设计草案（2026-08-16）

> 定位：本地 agent 裁定的两层结构中的**下层**。
> `Domain Prior Card`（现在可落地） → 未来找到可观察 trigger 后由 `Context Rule Card` 局部覆盖。
> 本卡属于 **Memory/Control Card**，不是 Context-conditioned Scope/Program Card。
> 阶段命名：**Target-local Memory/Control evolution**，不得称作完整 Context-conditioned Skill evolution。

## 1. 为什么这张卡能立，而 Context Rule Card 立不起来

Rule Card 需要填 `applicability.feature`——"什么数据适用什么算子"。本仓库 2026-08-16 的七轮实验
（F1 缺陷特征 / F2 程序几何 / F3 cohort 聚合 / Pattern 多尺度结构 / MEMO 自身历史 /
probe_direction 主动剂量响应；预测框架与排序框架各一遍）证明该字段**当前填不了**。

本卡的 `observable_applicability` 为 `const: true`，**在已限定 scope 内部**恒真，没有条件可以判错。
唯一跨 cohort 稳定的正结果正是它的内容：无上下文全局先验排序相对随机，
gain/probe +26%(kdd2018) / +31%(metr_la)，命中最优算子 55%→70% / 46%→75%。

## 2. `const: true` 的作用域必须显式限定

Bootstrap Skill 可以无条件，因为它只描述工作方法、不推荐具体算子。**算子排序卡不行。**

```yaml
scope:
  task: forecasting
  consumer_family: parametric_forecaster
  domain_authority: target_local
observable_applicability:
  const: true
```

`consumer_family` 这一层有实测依据（本报告 `multiop.consumer_axis_2026_08_16`）：
ridge 与 dlinear_shared 的 gain 相关 r=0.982/0.995、逐格决策迁移 95.8%/97.7%（近乎同一模型），
而 kNN analog 逐格一致率仅 45.7%/67.5%、互相套用逐格选择甚至为负收益（−7.4% / −15.8% headroom）。
→ **真正的边界是「参数化 vs 类比」这类归纳偏置类别，不是 `downstream_model_class` 的模型名字符串。**

Source 卡迁入新 Target 时只能是 `LOCAL_DRAFT` / `source_prior`，**不得直接成为 Target 的 active 恒真卡**。

## 3. 卡结构

```json
{
  "schema_version": "domain-ordering-card/1",
  "card_id": "ordering:<domain>:<task>:<consumer_family>",
  "card_kind": "memory_control",
  "revision": 3,
  "scope": {
    "task": "forecasting",
    "consumer_family": "parametric_forecaster",
    "domain_authority": "target_local",
    "program_family": "outlier"
  },
  "observable_applicability": {"const": true},

  "evidence": {
    "outlier_mad":   {"legal_opportunities": 60, "attempts": 41, "positive": 32,
                      "negative": 7, "conflict": 2,
                      "E_gain": 0.4875, "E_harm_magnitude": 0.0661},
    "winsorize":     {"legal_opportunities": 60, "attempts": 15, "positive": 14,
                      "negative": 1, "conflict": 0,
                      "E_gain": 0.1261, "E_harm_magnitude": 0.0114}
  },

  "ranking_key": {
    "formula": "E_gain - lambda * E_harm_magnitude",
    "lambda": 3.0,
    "lambda_source": "fit on source bank, frozen before any target outcome",
    "rationale": "与 delayed 单侧 harm-veto Gate 对齐；lambda 是声明的风险姿态参数，不得内嵌"
  },
  "order": ["outlier_mad", "winsorize", "outlier_iqr", "hampel_filter"],

  "suppressed": [
    {"operator": "hampel_filter", "action": "downrank",
     "reason": "E_gain<0 且 E_harm 最高", "reinstatement": "任一新 Target Support 为正即重新准入"}
  ],
  "exploration_slot": {"reserved": 1,
                       "policy": "每轮至少一个非先验候选，防止分母自我实现"}
}
```

### 硬约束（违反即失去可信度）

1. **计数分母 = `legal_opportunities`（合法机会），不是 `attempts`。**
   「从没被选中」≠「试过但无效」。用 attempts 当分母会让降权自我实现。
2. **`suppressed` 只允许 `downrank` / `suppress-from-prior`，禁止 `remove`。** 必须记录 `reinstatement` 条件。
3. **保留 `exploration_slot`。** 否则被降权算子的分母永远不再增长。
4. **卡由计数确定性生成，不由 LLM 写。**
   G2（结构化 Runtime-owned binding）一次通过；G3/P4（自由文本 patch）两次「修复部分+引入回归」
   → `PATCH_REJECTED`，自由文本 Guidance family 已关闭。Slow Agent 只能**提议分组**
   （「这批 Episode 可能形成共性」），计数与批准必须确定性，走既有 `requires_target_support` + 配对 replay 门。
5. **卡不能直接执行。** 它只改探测顺序；实际收益仍由 Target Support 判定，去留由 delayed 判定。

## 4. ranking_key 的实测依据（探索性）

`key(op) = E[gain] − λ·E[(−gain)₊]`，两项同源于 `evidence` 计数。

| λ | cohort A gain / harm | cohort B gain / harm |
|---|---|---|
| 0（纯收益） | +0.0690 / 0.532 | +0.5946 / 0.617 |
| **3–5** | **+0.0762 / 0.516** | **+0.5955 / 0.483**（probes 1.60→1.48） |
| 10 | +0.0762 / 0.516 | +0.2655 / 0.350（收益塌） |
| oracle | +0.1008 / 0.365 | +0.6803 / 0.133 |

λ∈[3,5] 在**两个 cohort 上同时**不劣于 λ=0 且伤害更低（B 上 −22%）。

**这正是 a5v2/a5v3 被否决的那条线**：两次都是「A5 敢动 → 增量 harm 更严 → 安全否决」
（v3: 5 vs 2）。上表说明该 harm 是**排序键的函数**，不是 Source 迁移的固有性质。

⚠️ λ=3 是在 query 上扫出来的，**探索性**。落地协议：在 source bank 上拟合 λ、冻结、再套 Target。

## 5. 预先声明的天花板

本卡会收敛到**每域一张**，然后走平。继续上涨需要**拆分**（「本域分成两个子群、各有先验」），
而拆分需要条件特征——七轮证明当前不存在。

→ Scope SPLIT 保持暂缓；**把「平台高度」当作待测量，不当作待修缺陷。**

## 6. 三臂连续实验（本卡的验证）

| 臂 | 先验来源 | 回答 |
|---|---|---|
| STATIC | 冻结，不更新 | 基线 |
| A3 | 只从 Target Episodes 累积 | Target-local 学习是否有效 |
| A5 | Source Episodes 初始化 + **完全相同**的更新规则 | Source Experience 是否加速 |

- 严格 prequential：按 origin 批次冻结 Memory → 整批探测 → 整批写入 → 下批才可用。
  同一 origin 内后处理的 series 不得使用前面 series 刚打开的 Outcome，否则 cohort-time 共模会被误当成学习能力。
- Source bank 与 Target 格必须完全分离。
- 单一排序策略，不得 A 用 Pattern、B 用 origin-only。
- 承重指标：累计 gain / 累计 Support probes / 累计 harm。随机与 oracle 只作上下界。
- 「后期优于前期」不单独构成进化证据——须比较同期 `adaptive − static`。

---

## 附录 A —— 实现修正（2026-08-16，E1/E2 落地后追加，只追加不改写上文）

上文 §2 §3 的两处结构在实现时**被代码事实推翻**，以本附录为准：

| 草案 §2/§3 写的 | 实际落地 | 原因（已代码核实） |
|---|---|---|
| `observable_applicability: {const: true}` | `{feature: "task_kind", op: "==", value: "forecast"}` | `const: true` 会让卡在 **anomaly_detection** 请求上同样被检索到。`task_kind` 是 16 个合法 observable feature 之一，域 `{forecast, anomaly_detection, classification}`——用它做检索层的任务门是零成本的 |
| 顶层 `scope:` 字段 | `risk_guards.scope`（4 键）+ Runtime `card_scope_matches` | `skill-entry/1` 字段集封闭（`_require_exact_fields` + `_reject_forbidden_fields`，8 个字段），顶层新字段会被 loader 直接拒绝 |
| 「写进 `risk_guards` 即可限定 scope」 | **不成立**，必须 Runtime 再查一次 | `retrieval.resolve_harness_view` **不读** `risk_guards`（该字段在 retrieval.py 中只出现在序列化处）。检索层只看 `observable_applicability` |
| `consumer_family: parametric_forecaster` | **精确** `downstream_model_class: "ridge"` | 本地评审：先用精确 consumer，不提前泛化。ridge↔dlinear 的 r=0.982/0.995 是**放宽的依据**，不是现在就放宽的许可 |

### 供给层的硬上限（E2 发现，写进契约）

`methods/ttha/schemas/fast_propose_v1.json` 的 `candidates.maxItems = 3`：
**一次 propose 最多 3 个候选**，第 4 个会让载荷 schema 校验失败
（`AgentProtocolError: payload.candidates has too many items` → `compilation_status=failed`
→ 候选池退化为 `('identity',)`，整轮 abstain）。

这条约束决定了排序卡的**有效作用域上限**：卡只能重排 Fast 已供应的候选，
所以单轮内它最多在 3 个候选之间排序。要让第 4 个算子可达，必须动 **Program Supply**
（Reference 1/2/3 通道或 schema），那是另一条边——排序卡按设计不碰它。
