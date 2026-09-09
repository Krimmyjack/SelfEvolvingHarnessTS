# R4A 主线复核与追加读数（Fable，2026-09-07；0 fit / 0 LLM / 0 held-out 读）

对象：`docs/R4A_PATTERN_IDENTIFIABILITY_RESULT_2026-09-07.md`（Opus）、
`artifacts/main_protocol/r2_revision_effect_stability.{json,md}` 与
`docs/R4A_IDENTIFIABILITY_EVIDENCE_LEDGER_2026-09-07.md`（Grok）。
证据类别：**MECHANISM / NEGATIVE**，development 读数，不做显著性主张。

## 1. 非作者复核（主线独立重算，只读 `_scratch/m_r0k_prediction_store.json`）

| 项 | Opus 报告 | 主线重算 | 结论 |
| --- | --- | --- | --- |
| W3 与 ANCESTOR 逐值全等 | 42/42 条 | 840/840 行 | 一致；W3 为别名，主表只有 6 格独立 |
| transport_flip（Support 受益中 delayed 受害） | 0.316151 / 0.450777 / 0.334572 | 0.316151 / 0.450777 / 0.334572 | 逐位一致 |
| W1/support mean g | −0.031355 | −0.031355 | 一致 |
| ANCESTOR 两面 helped 率 / severe 率 | 0.692857, 0.066667 / 0.683333, 0.073810 | 0.6929, 0.0667 / 0.6833, 0.0738 | 一致 |
| 边界 | fits 0、LLM 0、max_time_index 3911 | 库内最大 origin 3864 + 48 = 3912 | 一致，未触 4056 |
| verifier 未过的空面 | 剔除 8 条 CAND | 库内 14 条空面全部 `verifier_passed=False`（8 CAND + 6 零星程序） | 处理正确，未记 0 |

判词、阈值、分箱未被调整；四块/三块两版都在工件里。复核 **PASS**。

## 2. 追加读数：逐序列增益是不是序列的稳定属性

Opus 报告把最大不确定性留在"条件存在但表示不够强"与"逐序列增益的方差主要不是序列属性"之间。
预测库里同一 (program, position, uid) 恰有 origin 与 origin+48 两个面，等于一次 48 步间隔的重测；
同一 uid 又在同一块内跨 7–9 个位置重复出现。两者足以做零 fit 的方差分解。

**(a) 48 步重测（同序列、同程序，g@origin 对 g@origin+48；每程序 420 对）**

| program | Pearson | Spearman | 符号一致率 | 独立时的期望一致率 | P(severe@+48 \| severe@o) | severe 基率 |
| --- | --- | --- | --- | --- | --- | --- |
| ANCESTOR (`outlier_mad`) | 0.364 | 0.065 | 0.571 | 0.568 | 0.107 | 0.074 |
| W1 `hampel_filter` | 0.281 | 0.017 | 0.536 | 0.500 | 0.325 | 0.236 |
| W2 `pmc→outlier_mad` | 0.286 | 0.109 | 0.569 | 0.540 | 0.283 | 0.179 |

Spearman 接近 0、符号一致率落在独立假设的期望值上：**同一条序列 48 步之后是否受益，与它现在
是否受益基本无关**。Pearson 略高只来自少数大幅值配对（尖峰同时落在两个 context 里的序列）。
严重伤害的持续性只有基率的 1.4–1.6 倍。

**(b) 块内按 uid 的一元方差分解（序列间方差占比 SSB/SST 与 ICC）**

| program | 块 [0:40]（7 位置/uid） | 块 [40:80]（9 位置/uid） | 零假设下 SSB/SST 期望 |
| --- | --- | --- | --- |
| ANCESTOR | sup 0.126 / ICC −0.012；del 0.176 / 0.047 | sup 0.076 / −0.035；del 0.085 / −0.025 | 0.137 / 0.106 |
| W1 | sup 0.192 / 0.066；del 0.229 / 0.111 | sup 0.041 / −0.077；del 0.096 / −0.011 | 同上 |
| W2 | sup 0.187 / 0.060；del 0.183 / 0.056 | sup 0.040 / −0.077；del 0.079 / −0.032 | 同上 |

（块 [80:120] 仅 2 个位置、[120:160] 仅 3 个位置，SSB/SST 被小样本抬高到零假设水平 0.44–0.61，
不作引用。）在有 7–9 次重复的两个块上，**序列身份解释的方差份额为 0–11%，ICC 在 0 附近**。
若 HEC-1 沿用冻结训练 anchors（同一 program 跨 origin 共用同一个已拟合模型），则跨 origin 的
逐序列差异只来自 serving context 与 horizon 实现；ICC≈0 说明它们不构成任何序列层面的稳定条件。

## 3. 三包合起来的结论

1. **序列层面没有可学的条件。** 可见模式读不出（R4A 最佳 0.63–0.68，机制量不胜现有词表）；
   未来窗口也解释不了（ORACLE ≤ 0.606）；而且它根本不是序列的稳定属性（重测 Spearman ≤ 0.11，
   ICC≈0）。三条独立证据指向同一件事：在 KDD × pooled Ridge × 48 步 horizon 下，
   "这条序列此刻该不该被这个程序处理"这个量，主要是 Consumer/评估侧对单次 horizon 实现的响应，
   不是数据模式的函数。这一个事实同时解释 W54、W61、D1、P4d、HEC-1 的 34→10、
   DEV 系列 32–45% 的翻号，以及为何任何 Scope 修订器都不可能在这上面成立。
2. **单元/cohort 层面的条件是稳的。** R2 显示 `outlier_mad` 在 86–92% 的单元上聚合方向为正，
   `hampel` 一贯更差（W1 d 符号一致率 0.952，全负）；DEV-AUTO-3 的"供给臂减控制臂"两组同号
   也是这一层的知识在起作用。可复用知识的正确粒度是 **program × Consumer × cohort**，
   不是 series/interval。
3. **唯一的模式级线索也在程序层面**：W1 只对 `spike_peak_over_tail_sd >= 6.0` 的序列有效
   （四折均 +0.081614 对全治），本质是"一个平均有害的程序的禁忌条件"，而非"一个平均有益程序
   的精调 Scope"。它是否是留一幸存者需要前瞻验证（≈2 fits/单元），但在 (a)(b) 之后优先级下降。
4. **first-fault 上移到评估/协议层。** 权威门的两条逐序列风险线（harmed_fraction、
   single_series_harm）在单个 origin 上度量的是一个重测相关≈0 的量；Support 面通过对 +48
   几乎没有预测力，是结构性而非 Scope 问题。这正是 §6 梯子的
   "结果可用但无法归因 → instrument / credit assignment"。
5. **对"给 LLM 什么信息"的回答**：不存在可以交给 LLM 的、能预测单序列受益的模式信息；能给的是
   cohort 级程序战绩（K0 卡已经是这个形态）与程序级禁忌（如 W1 的尖峰条件）。信息墙没有饿死
   LLM——它要的那种信息在这份数据上不存在。

## 4. 对推进方案的修正（呈 astra / sol）

- **停止**：任何以 series 级 Scope 修订为对象的实验（含 M-W、ACC-1 第二层、有界选择规则实验）；
  R2 假设"比较性知识更稳定"不成立，不作地基。
- **改做（0 fit，从同一预测库）**：**证书粒度重打分**——把四线门分别在 (i) 单 origin，
  (ii) 两 origin 合并（Support+delayed 逐序列取均值/取劣），(iii) 只用 cohort 级聚合 + 序列
  bootstrap 下界 三种口径下重算 HEC-1 全部候选的通过集合，读每种口径下"Support 通过 → +144 仍安全"
  的保持率。这是把 34→10 从"Scope 缺陷"改写为"证书粒度缺陷"的直接检验，也是唯一能改变
  Harness 行为（门/评估器）的可测设计变化。
- **论文**：条件化主张按粒度改写——自然数据上成立的是 Task/Consumer/cohort 条件化与程序级禁忌，
  series 级 Pattern 条件化在该设置下不可识别且不稳定；把 (a)(b) 两表与 ORACLE 行作为方法学
  核心图之一。它同时是对静态 Router 与对"逐序列风险门"两种设计的负结果。
- **held-out 开封、Consumer 轴、写作时间线不变**；Consumer 轴（per-channel 或 TSFM）现在多了一个
  明确问题：ICC≈0 是 pooled Ridge 的性质，还是任务的性质。

## 5. 这份附录不是什么

- 不是显著性结论（80 个 uid、4 块、重复测量）；不是对 imputation/缺口方向的判决（W2 仅一格独立）；
  不是"未来窗口无关"的物理结论（只测了三个 48 步 ORACLE 量）；不是对 12 维词表的背书。
- 未改任何阈值、协议、roster、预算；未新增 SHA；未提交 git。

## 附：追加读数的计算（可复跑，0 fit）

```python
import json, numpy as np
from collections import defaultdict
d = json.load(open('_scratch/m_r0k_prediction_store.json', encoding='utf-8'))
def rows(prog):
    out = {}
    for v in d.values():
        if v['program'] != prog or not v['raw_per_view']: continue
        deg = set(v.get('degenerate_uids', []))
        for u, r, p in zip(v['eval_uids'], v['raw_per_view'], v['program_per_view']):
            if u not in deg: out[(v['position'], v['face'], u)] = r - p
    return out
block_of = lambda pos: 0 if pos <= 6 else 1 if pos <= 15 else 2 if pos <= 17 else 3
# (a) g@origin vs g@origin+48 on the same (position, uid); (b) one-way ANOVA of g by uid within (face, block)
```
