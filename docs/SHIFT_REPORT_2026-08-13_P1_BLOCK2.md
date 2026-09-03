# 晨间报告：P1 Block 2 闭包 + 停止条件触发（2026-08-13 午后）

三角色交接格式（用户任务书要求）。

---

## CURRENT_VERDICT

**BLOCK2_FAMILY_NO_HEADROOM + STOP_CONDITION_TRIGGERED（连续两个 family 无 headroom）**

P1（用户预注册规格）：第二个自然 Development Block census（T102-T105）+ 确定性 Evidence Compiler 分支。结果：

1. Block 2 **未复现跨 series family**——唯一 failure family = winsorize NEGATIVE × T105 单 series（3 窗口：600/792/984，gain −0.013/−0.061/−0.032）。T102/T103/T104 的 winsorize 全部为正（+0.0002 ~ +0.75）。winsorize 失败是 **series-dependent**，不是跨 series 系统性现象。
2. EC 分支：candidates=(outlier_mad, hampel_filter)，**n_common_positive=0** → 确定性 abstain（`unique_common_positive → None`）。outlier_mad 在三个失败窗口上 gain 全为 0.0（该 series 契约参数下无离群点可修）；hampel_filter 三窗 −0.0077/−0.0002/+0.004 全部 < M。
3. **停止条件（用户 P1 规格）**：Wave 3 family（winsorize × 4 series，6 窗口，无 headroom）+ Block 2 family（无 headroom）**连续两个 family 无 headroom** → 按规格关闭 **winsorize/outlier × forecast|ridge|sMASE × 当前机制**（Group Evidence + 确定性 Evidence Compiler + probe supply）→ 本报告。

零 PASS 造假、零 Claim 越界、零真实 LLM 调用（EC 完整链分支未触发——无唯一共同正向替代，这正是确定性 Evidence Compiler 的设计行为）。

## BEHAVIOR_CHANGED（本阶段改了什么 Harness 行为）

**无**。本阶段是预注册装置的纯执行——未修改任何 Harness 代码。装置层唯一变化（非 Harness 行为）：
1. census 的 **series 级并行**（`parallel_eval.run_parallel`，4 series 并发；单 series 内 4 origin 保持顺序——单轮内探测顺序是自适应语义，不可并行——用户裁决提速）。
2. `evidence_compiler` 模式首次用于真实 development block 分支（此前只在 evc dev 验证链中用过）。

## REAL_EVIDENCE（真实运行证据，全部 development 级）

| 实验 | verdict | 关键数字 |
|---|---|---|
| P1 block2 census | BLOCK2_FAMILY_NO_HEADROOM | 20 probes / 4 series × 4 origins（并行）；唯一 family=T105 winsorize 3 窗；headroom 零共同正向 |
| EC 确定性分支 | 确定性 abstain | outlier_mad 三窗 0.0 / hampel 三窗 <M——`unique_common_positive → None`（正确行为） |
| 停止条件检查 | TRIGGERED | wave3 top family no-headroom ✓（从 census 报告确定性复核）+ block2 no-headroom ✓ |
| 装置一致性 | PASS | 预注册核对：T102-T105 = cache 顺序下 Wave 3 未用的前 4 个 ✓；不在 p41 cohort ✓；长度 10898 ≥ 1032 ✓ |

## FIRST_FAULT（本阶段 first fault）

无新 first fault——装置按预注册执行，结果是科学结论而非装置故障。累计两个 development block（8 series）的 winsorize family 全部无共同 replacement headroom。

## NEXT_BRANCH（按用户 P1-P6 自动推进序）

P1 闭合 → **P2 按失败原因自动分支**：失败原因 = NO_COMMON_PROGRAM_HEADROOM（两 family 同因）——当前机制在 3-op probe 空间内无法为该 family 找到共同正向替代。结合既有证据（今晚 100+ 完整 Consumer 评估 = 成本瓶颈证据），自动分支到 **P3 PIA 校准**：
- gold 已就绪：14 个 winsorize 窗口 + 12 headroom + 54 supply 评估（全部已暴露，零新评估）
- 规格：Program ΔX/ΔY → first-order Response Sketch vs gold——**只验 top-k recall / sign agreement / harmful FP / full-Support 减少量，绝不接批准**
- 定位：评估省钱机制（Program 效果的一阶预筛），不替代 Support 实测

## STOP_CONDITION（已达，按用户 P1 规格）

连续两个 family 无 headroom → 关闭 winsorize/outlier × forecast|ridge|sMASE × 当前机制。生成晨间报告并进入自动推进序的下一项（P2 分支 → PIA 校准）。

---

## 附：本阶段全部新文件与改动

**新 runner**：`evaluation/functional/run_v1_block2_census_ec_dev.py`（P1 预注册：block2 census 并行 + EC 分支完整链——含唯一正向替代时的 LLM 只编译路径、补集检查、delayed 门、Skill adoption 留痕——本 run 未触发完整链，分支代码已备）。

**新报告**：`artifacts/functional/e2/w1_block2_census_ec_dev_report.json`（census rounds 20 probes / families / headroom / EC 分支 / 停止条件检查全记录）。

**LLM 成本**：0 次真实调用（EC 完整链未触发）。P0 以来累计真实调用：72 + 64 + 链验证（全部留痕）。

**Claim 边界**：一切为 development exposure（零新 Claim）。可声称：机制级（确定性 Evidence Compiler 分支按预注册执行、并行 census 装置工作、abstain 正确性）。不可声称：任何性能级效应。关键信息值：winsorize 失败窗口在 8 个 development series 中仅出现于 5 个（wave3 四 series + block2 T105），全部无共同正向替代——该 family 在当前机制下的探针空间已穷尽。
