# P2 动机表 VERDICT（论文实验 1：readiness 非普适）

> 范围：synthetic motivation-grade slice; frozen deterministic judges; NOT confirmatory；ε=0.01；参照=v_raw_identity；seed=20260709，n=40，B=1000。

| program | forecast Δ(nRMSE↓) [90%CI] | anomaly Δ(F1↑) [90%CI] | 部署契约允许(f/a) |
|---|---|---|---|
| v_raw_identity | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | Y/Y |
| v_impute_linear | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | Y/Y |
| median_w9 | +0.9720 [+0.7659,+1.1599] | -0.6416 [-0.6886,-0.5955] | Y/N |
| winsor | +0.6674 [+0.5091,+0.8220] | -0.5859 [-0.6368,-0.5324] | Y/N |
| universal_cleaner | +0.9733 [+0.7848,+1.1571] | -0.8476 [-0.8890,-0.8103] | Y/N |
| task_conditioned | +0.7157 [+0.5385,+0.8904] | +0.0000 [+0.0000,+0.0000] | Y/Y |

**fresh 翻转程序（forecast×anomaly，判据过 CI）**: median_w9, winsor, universal_cleaner
**任务对计数**: fresh=1（forecast×anomaly） + frozen=1（forecast×classification 引用） → 出口判据(≥2) **PASS**

## frozen classify 引用（第二组任务对）

来源: `C:/Users/辉/Desktop/Agent/SelfEvolvingHarnessTS/_clf_maintable.log`

    v_median           inception=+0.249   rocket=+0.134
    v_savgol           inception=+0.046   rocket=-0.054
    v_stl              inception=-0.037   rocket=-0.110

frozen classify C1（ΔPerf vs raw, reporter ⟂ judge, final_test split）: v_median 助 classify（inception +0.249 / rocket +0.134）；v_stl 伤 classify（-0.037 / -0.110）而 stl/savgol 族在 forecast 侧为正收益（E-1.1/F0 冻结结果）→ forecast×classification 符号翻转（frozen 任务对）

## 解读

同一批序列上，universal cleaner（impute→winsor→median9）助 forecast（去噪去尖峰）却毁
anomaly（尖峰正是检测目标）；registry 任务契约（D6）已把该翻转编码为物理禁入（表末列 N）。
task_conditioned 行显示按任务选择处理即可同时保住两侧。这就是 pattern/task 条件化
readiness 的动机证据（motivation-grade；confirmatory 见 P5）。
