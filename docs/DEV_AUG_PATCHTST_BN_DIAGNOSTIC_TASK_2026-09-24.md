# 任务书：DEV-AUG-PATCHTST-BN-DIAGNOSTIC（2026-09-24）

状态：Planner 下发，执行者照做。只做本包，不重新学卡、不重训、不调 LLM、不做优化器更新。入口 `evaluation/main_protocol_p4/batch_research_aug_patchtst_bn_diagnostic.py`，输出 `_scratch/dev_aug_patchtst_bn_diagnostic/`。身份：机制诊断（已曝光的开发 Test）。

## 1. 问题

PatchTST 包里增强模型对 None 的负收益，有多少来自 BatchNorm 运行统计？训练时每步做两次 train 模式前向（父窗一次、子视图一次），BN 的运行均值 / 方差因此混入了子视图的统计量；服务时只看原始窗口。

## 2. 范围与做法

- 电力 16 个已曝光 Test 案例 × 3 seed × 3 个已有模型：None、NoMix（`P_NoMixRecipe`）、Edit[-resample-random_conv]（每案例的物理方案取 `test/fixed_ref_phys.json`）。原包模型只读，诊断输出全部写到新目录。
- 臂 1：原 checkpoint，按原流程评分（`M.e_inputs` + `PT.predict` + `ec.score_with_status`）。结果必须与原包存值逐位相同。
- 臂 2：复制 checkpoint，冻结所有学习参数，只用该案例 T 段父窗输入 X（`common/<case>/parents.npz` 的 `Xp`，16 实体 × 433 = 6928 窗）重估 BN 运行统计，再按原流程评分。
  - 不读 C_A / C_B / E 的输入或任何标签。
  - 冻结的重估协议：批大小 64；按 `parents.npz` 行序；1 遍；`reset_running_stats()` 后 `momentum=None`（按批累计平均）；`model.eval()`（dropout 关）后只把 BN 层设为 `train()`；`torch.no_grad()`。RevIN 用的是逐样本统计，没有缓冲，不受影响。
  - 重估前后 `named_parameters` 逐位相同，否则报错停止。
  - None 模型用同一方式重估。
- 并行：两个 GPU worker（本机 RTX 4060），按案例轮流分配；每个模型约 172 MB，不挤占其他任务。

## 3. 读数

1. NoMix / Edit 相对各自 None 的收益（原对原、重估对重估）：16 例均值、分时期、分案例、分起点。
2. Q4_G1 第二起点的误差和预测偏差（归一化预测 − 真值的均值）在重估后是否明显缩小。
3. 系统读数：按原冻结卡的交付（`result.json` 中 `per_case.delivered_phys.f_patch`；交付 None 的 11 例仍为 None，交付 Edit 的 5 例用 Edit 模型）重算 f_patch 电力对 None 的收益。原口径必须复现原包的 D01 值 −6.875。
4. 原包结果的稳健性补充（原全样本主结果保留，不替代）：中位数、截尾均值、去掉 Q4_G1。截尾比例事先固定为每侧 10%：每个域内从排序后的逐案例值两端各去掉 round(0.10 × n) 例（至少 1 例；电力 16 → 2+2、交通 16 → 2+2、太阳能 8 → 1+1），剩余取均值，三域再等权平均。

## 4. 停止条件

完成后报告并停止，不自动进入重训或改卡。

## 5. 回执

- 2026-09-24 本机 RTX 4060，2 个 worker，144 个模型（16 × 3 × 3），约 10 分钟，0 LLM、0 优化器更新。报告 [_scratch/dev_aug_patchtst_bn_diagnostic/REPORT.md](../_scratch/dev_aug_patchtst_bn_diagnostic/REPORT.md)，逐模型结果 `cells/`，重估后的模型 `models/`。
- 核对全部通过：原流程评分与原包存值逐位相同（144/144）；重估前后学习参数逐位不变；系统原口径复现原包电力 f_patch −6.875；NoMix 原口径复现原包 −9.58。
- **判读：BN 运行统计不是负收益的原因。** 按父窗重估后增强模型没有变好，反而略差：
  - 相对各自 None：NoMix −9.58 → −10.67，Edit −11.19 → −12.62；中位数 −4.98 → −4.22、−3.32 → −3.16；去掉 Q4_G1 −4.16 → −4.31、−5.68 → −6.26；胜例数不变（4、5）；变好的案例 NoMix 8/16、Edit 4/16。
  - Q1–Q3 变化都在 ±1 pp 左右；Q4 全体更差（NoMix −34.5 → −40.6，Edit −41.4 → −46.7），None 自身也变差（Q4 四例 −3.5 到 −7.2 pp，16 例平均 −1.71）。
  - Q4_G1 第二起点：增强模型的误差和负偏差都比 None 大（偏差 −1.30 / −1.37 对 −0.52），重估后没有缩小（−1.42 / −1.49，误差 3.35 → 3.95、3.43 → 4.00）。这个偏差存在于学习参数里，不在 BN 统计里。
  - 系统读数（冻结交付，11 例 None）：−6.88 → −7.76；去掉 Q4_G1 −1.08 → −1.08。
  - 重估带来的 BN 统计相对变化：None 7.2%、NoMix 8.9%、Edit 7.8%。NoMix 的统计确实被子视图带偏得多一点，但纠正它没有带来收益。
- Q4 的负收益不只来自 Q4_G1：去掉 Q4_G1 后 Q4 其余三例 NoMix 仍为 −15.7（重估后 −18.7），Edit 为 −22 到 −31。
- 未检验：两次前向对训练过程本身的影响（父窗批和子视图批分别做批归一化，参数适应了各自的批统计）。检验它需要重训（例如父子拼成一批做一次前向，或换成 LayerNorm），超出本包范围。按任务书停止，没有进入重训或改卡。
