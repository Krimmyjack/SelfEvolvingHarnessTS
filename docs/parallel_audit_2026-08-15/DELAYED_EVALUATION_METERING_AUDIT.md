# open_delayed 重复仪器评估计量审计（执行者二轮稿）

范围：只读审计 `methods/ttha/online_loop.py::open_delayed`。不修改代码。

## 1. 评估位点
同一个 `d_origin`、同一个 `steps` 可能在一次 `open_delayed` 中被评估最多三次：

| 位点 | 代码 | 用途 |
|---|---|---|
| A | episode 更新循环 `rd = executor.evaluate(tuple(steps), d_origin)` | 更新 winner Episode 的 delayed 状态 |
| B | 三个 pending 分支中的 `handle_feedback_delayed` | 用 delayed 结果批准/拒绝 pending |
| C | `wd = executor.evaluate(result._winner_steps, d_origin)` | 计算 `delayed_utility` |

## 2. 按轮次类型计数

| 轮次 | A | B | C | 实际次数 |
|---|---|---|---|---|
| 普通 Fast winner，allow_fast_skill=true 且形成新 pending Skill | 1 | 1 | 1 | 3 |
| Slow replay 成为 winner 且 slow_event=pending | 1 | 1 | 1 | 3 |
| Group slow pending | 1 | 1 | 1 | 3 |
| 部署已有 Skill（deployed_skill_id，无新 pending） | 1 | 0 | 1 | 2 |
| 无 winner（abstain） | 0 | 0 | 0 | 0 |

## 3. 影响
- 这些调用均未计入 `target_support_receipts_used` 或
  `slow_replay_receipts_used`；所以当前“反馈预算”只计 Support，不计
  delayed 仪器调用。若未来把 delayed 纳入预算，必须先修此处。
- 对确定性 Ridge 评估器，三次调用结果相同，不改变已终裁科学结论；
  但若未来接入有状态/采样型评估器或 TSFM，重复评估会产生不可忽略的
  计量与方差问题。
- 这不是当前第一阻塞，但属于 Gate 计量修正候选。

## 4. 建议（仅记录，不实施）
- 未来最小修复：在 `open_delayed` 开头对 winner 只评估一次
  `winner_delayed = executor.evaluate(winner_steps, d_origin)`；
  A/B/C 三处全部复用该结果。
- 修复前必须加回归测试：三种轮次类型各断言 executor evaluate 调用次数
  ≤1，并确认 delayed_utility 与 episode.delayed_response.gain 逐位一致。
