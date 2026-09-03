# V1 Sealed-Target A5/A3 切片——冻结设计（2026-08-09）

外部审核第六轮（§7 三十一）裁决：GEFCom 600-936 切片已暴露（
EXPOSED_DEVELOPMENT_NEGATIVE_TRANSFER_CASE，不实现 runner）；下一步 =
**outcome-sealed 的新 Target cohort**。本文件 = sealed 切片冻结设计。

## 1. Sealed 状态与预注册纪律

- **本文件不含任何 Target gain**（未对 Target cohort 运行任何 evaluate；
  只检查长度、可见 Context、动作合法性——审核允许的三项）。
- Target outcome 在冻结 Source Memory 与策略之后才打开（运行时）。
- 预注册指标与 verdict 规则在 §5/§6 固定，运行后按运行时 receipt 计算，
  不事后调整。
- **修正（外部审核第七轮条件批准）**：Source 与 Target = 同一 dataset 的
  **两个互斥 certified-virgin series cohort**——口径为"同 dataset、跨
  series-cohort Experience 复用"，不是跨域，也不是同 cohort 早期反馈。

## 2. 数据身份（修正：互斥 cohort + virgin 过滤）

候选盘点（`series_registry.jsonl`，`frequency: hourly`、length=1024）：

| dataset | 系列数 | certified_virgin | probe_consumed |
|---|---|---|---|
| **monash:traffic_hourly**（首选） | 862 | 806 | 56 |
| uci_electricity_load_diagrams | 370 | 370 | 0 |
| metr_la | 207 | 207 | 0 |

**核对（P0-2）**：现有 `_fixed_roster` 只过滤 dataset+length（
run_e2...py:502），按排序前 20 在 traffic_hourly 上会混入 2 条
probe_consumed——已核实（top20 exposure = 18 virgin / 2 consumed）。

**修正**：
- Runner **局部过滤** `exposure_class == "certified_virgin"`（不改通用
  `_fixed_roster`，不新增 registry hash / Gate / Exposure Ledger）；
- 从 virgin 池（按 series_uid 排序）取**前 20 = Source cohort**、**次 20
  = Target cohort**（互斥，不共享 series）；
- 两组 UID 列表在运行时冻结并写入报告。

Source cohort：@600/@648 探测（窗口 [600,696)）。Target cohort：
@792/@840/@888/@936（保持 outcome sealed 直到 Source 冻结）。

## 3. 时序（修正：补 R1 delayed 生命周期；全部断言计算）

```
Source 阶段（Target cohort 未打开；探测计划固定，不挑正）：
  Source cohort @600 决策点：确定性探测计划 = explore 顺序前 2 个算子
    （proposal 1/2）→ 每个 proposal：evaluate(chosen, 600) → Support
    receipt → 立即写 Episode（正/负/冲突如实记录）
    → 同一 chosen 的 delayed：evaluate(chosen, 648) → update_delayed_status
  Source Episode 集合（≤2 条）在 Target 打开前冻结

Target 阶段（Source 冻结后打开；A5/A3 各自独立 Episode/Skill fork/
  method 实例，禁止跨臂写回）：
  R1 Support @792：两臂同步 prepare → 探测序列（预算 ≤2 proposal/点）
    → 每 Support 立即写本臂 Episode
    → 正向 Workflow 写为 LOCAL_DRAFT Skill（本臂 fork）
  R1 delayed @840：本臂 delayed 打开 → 更新本臂 Episode/Skill 为
    LOCAL_ACTIVE / CONFLICT / RESTRICTED
  R2 prepare @888（= R1 + 2×HORIZON，断言）：bind_round_data(888) →
    正常入口 prepare（本臂 fork 快照：Skill 可检索并执行）→ 探测序列
  R2 delayed @936：最后打开 → Skill delayed utility
```

窗口不重叠断言（程序计算）：R2 − R1 == 2×HORIZON；Source 窗口
[600,696) 末端 ≤ R1=792（决策无未来）；R2 delayed [936,984) ≤ 1024；
Source cohort 与 Target cohort 无共享 series。

## 4. 双臂（唯一初始差异 = A5 的 Source Episode 集合）

| | A5 | A3 |
|---|---|---|
| Memory 初始 | Source 阶段冻结的 Episode 集合（如实） | [] |
| Agent | 同类型 backend（确定性；Reference 1 引导 + explore） | 同（独立实例，状态同步） |
| 动作空间 | 同 actionable inventory（Target cohort） | 同 |
| 预算 | 每决策点 ≤2 proposal（真实 Support receipt 同预算） | 同 |
| 探索状态 | 双臂从 R1 起同步（已验证模式） | 同 |
| 快照/fork | H0 + 本臂 R1 后独立 fork（learned skill） | 同（独立 fork） |
| 写回 | 只写本臂 Episode / Skill | 只写本臂（禁止跨臂） |

## 5. 预注册指标

1. **proposal 数**（每决策点、每臂）：propose 的候选数（含 REJ——REJ 不
   消耗 downstream Support）；
2. **first-positive Support receipt index**：首次 gain ≥ MATERIAL_THRESHOLD
   （0.005）的 Support receipt 序号（REJ 不计——与 proposal 数分开报告）；
3. **harm**：负 gain（< −M）Support receipt 数与幅度和（跨 R1/R2）；
4. **abstention**：无信息/identity/近零 ABSTAIN 探测数——**单独报告，不
   因次数高自动判负**（安全 abstain 可能避免 harm）；
5. **Skill 形成与执行**：R1 正向 Workflow 是否实际写为 LOCAL_DRAFT（
   delayed @840 后 LOCAL_ACTIVE/CONFLICT/RESTRICTED）Skill；R2 正常入口
   是否检索并执行它；
6. **Skill delayed utility**：Skill 在 R2 delayed（@936）的真实增益。

## 6. 预注册 verdict 规则（修正：六档，运行时 receipt 计算，不事后调整）

- `SEALED_A5A3_SOURCE_GUIDANCE_PASS`：A5 的 first-positive Support receipt
  更少 **或** harm 严格更低；其他承重指标不更差；**且**形成的 Skill
  delayed gain ≥ 0.005、正常入口成功执行。
- `SEALED_A5A3_SAME_NO_BENEFIT`：两臂反馈效率和安全性相同（无收益）。
- `SEALED_A5A3_PARTIAL`：A5 初期更快/更安全，但 Skill delayed 失败或未执行。
- `SEALED_A5A3_NEGATIVE`：A5 更慢、harm 更高或最终效用更差（负结果如实报告）。
- `SEALED_A5A3_NO_APPLICABLE_SOURCE_MEMORY`：Source Episode 存在但没有
  合法匹配——不冒充能力失败。
- `SEALED_A5A3_INFEASIBLE_NO_HEADROOM`：合法动作空间没有正向 Workflow
  （如实报告约束，不伪装）。

## 7. 边界（不称）

- 同 cohort 纵向（Source 早期窗口 → Target 后期窗口）——不称跨域迁移；
- 确定性 backend（零 LLM）——真实 LLM 选择质量继续暂缓；
- Source 阶段探测计划固定（explore 顺序），不按 Target 结果挑选——若
  Source 全部为负/冲突，如实记录"此数据上 Source 无正经验"；
- delayed 最后打开——信息墙遵守（决策点 prepare 时 delayed 未发生）。

## 8. 运行

```
python evaluation/functional/run_v1_sealed_a5_a3.py --domain monash:traffic_hourly
```

（runner 在冻结设计批准后实现；批准前不写。）
