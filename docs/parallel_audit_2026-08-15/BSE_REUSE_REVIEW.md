# BSE 已有机制复用审查（执行者稿）

结论先行：BSE 已经具备 Rule 表示、Runtime matcher、Slow 结构化选择、
held-out replay、delayed 核销和 removal-delta 判据，未来 trigger 出现后
**不需要新建平台**。但 BSE 当前是 runner 内 replay 装置，尚未接入正常
Fast Path 入口；需要审计的是“从 report 内规则到正常入口”的最小注入点。

## 1. 已有组件入口
| 能力 | 入口 | 位置 |
|---|---|---|
| Rule 表示 | `_bse_assemble_rule` | `evaluation/functional/run_v1_guidance_evolution.py` |
| Runtime matcher | `_bse_rule_fires` | 同上 |
| Slow 结构化选择 | `_bse_parse_slow_choice` | 同上，仅接受 P1/P2/abstain |
| 匿名 Capsule prompt | `_bse_capsule_prompt` | 同上 |
| held-out 机械 replay | `_bse_replay_rows` / `phase_bse_p5` | 同上 |
| PASS/FAIL 判定 | `_bse_pass_evaluation` | 同上，五条判据含 delayed 不劣与 removal delta |
| 协议冻结 | `phase_bse_freeze` | 同上，`bse_protocol` |

BSE 规则结构（现有，可直接作为最小 Rule Card 字段参考）：
```jsonc
{
  "rule_id": "...",
  "surface": "scope",
  "workflow_signature": "outlier_mad",
  "applicability": {"feature": "...", "operator": "ge|le", "threshold": 0.0},
  "unknown_policy": "no_prior",
  "authority": "LOCAL_DRAFT",
  "requires_target_support": true,
  "slow_approved": false,
  "evidence": {"positive_episode_ids": [], "negative_episode_ids": [],
               "conflict_episode_ids": []},
  "semantics": "..."
}
```

## 2. matcher 语义（当前限制：单特征 ge/le）

`_bse_rule_fires(value, rule)`：
- value None → 不放行；
- `operator == "ge"` → value ≥ threshold；
- `operator == "le"` → value < threshold。
未知输入 fail-closed。这是正确的最小语义，未来 trigger 可以复用。
若结构 Pattern 产出多特征 trigger，不能直接塞入现有 matcher；扩展须经
单独批准，不在本准备包内实现。

## 3. replay / removal 现状
- replay：H0 宽 Scope prior 恒在，H1 由规则门控 prior；pre-probe 顺序固定，
  post-probe Runtime winner resolution；delayed 复用已测缓存，零新 Outcome。
- removal：当前 BSE 的 `removal_delta_real` 是 H0/H1 行为差，**不是从
  snapshot 删除规则**。规则只存在于 report JSON，未写入 h0。
- 已落地的真正删除路径是 `online_loop.revoke_deployed_skill`（删除
  skills/** 文件并重编译），这适用于 Program Skill，不适用于 BSE 规则。

## 4. 未来 trigger 的最小注入点
不重建规则平台的前提下，未来 trigger 注入有两个可选点：
1. **Runner replay 注入（BSE 现有模式）**：
   在 `_bse_replay_rows` 的 H1 臂，把 `fires = _bse_rule_fires(value, rule)`
   的 `rule` 换成由结构 Pattern 实验产出的新 trigger。零生产改动。
2. **正常 Fast 入口注入（更接近最终形态）**：
   在 `methods/ttha/fast_agent.py` 的 Slot P 分支（约 1024 行
   `if runtime_prior_slot and _signed is not None:`）之前，增加一个
   纯函数判定 `local_rule_allows_prior(features, rule)`。规则不常驻 h0
   时由调用方传入；常驻 h0 时再从 snapshot 读取。**当前阶段不实现**，
   只记录这是唯一应改的方法层入口。

## 5. 复用结论
- 第一张 Rule Card 的 schema 可以沿用 BSE 规则字段，新增字段须另行批准。
- trigger 只替换 `applicability` 三字段（feature/operator/threshold）或
  未来更丰富的 predicate，不改 matcher 语义。
- 禁止为每一种新 trigger 重写 replay；只替换 `rule` 与 heldout labels。
