# 任务书：DEV-AUG-DLINEAR-SOURCE（2026-09-24）

状态：Planner 下发，用户转来，要求在本机运行。入口 `evaluation/main_protocol_p4/batch_research_aug_dlinear_source.py`，输出 `_scratch/dev_aug_dlinear_source/`。身份：只用 Source 的开发级 Consumer 筛查，不学卡，不读 Select 或 Test。

## 1. 任务（Planner 原文要点）

- 下一个小型 Consumer 用官方 DLinear，复用现有材料和案例划分。
- 先完成 Source 校准，再做 24 案 × 4 方案（None、NoMix、censor、shock）× 3 seed 的比较，不调用 LLM；独立的拟合按资源并行。
- 汇报逐域、逐时期的收益和成本，然后停止；暂不启动新卡学习或 Test。

## 2. 执行设计

- **Consumer**：官方 DLinear，来自 cure-lab/LTSF-Linear commit 0c11366 的 `models/DLinear.py`，克隆到 `ref/LTSF-Linear`，只读。它与 PatchTST 仓库带的副本只差换行符。配置为单通道（enc_in 1）、共享线性层（individual False）、输入 192、输出 48、滑动平均核 25，共 18,528 个参数。
- **训练循环**：与 PatchTST 包相同，AdamW（wd 1e-4）、每步 64 个父窗、父视图和子视图各占 0.5 损失、`train.batch_indices` 同一随机流、T scaler 输入、float32。在 CPU 上训练，每个进程一个线程，启用确定性算法。这是受控的任务适配，不是官方按轮次加早停的 benchmark 配方。
- **校准**：只用不增强，只看 Source C_A+C_B，用 6 个校准案例（每域 A1_G1、A2_G1），首个 seed。lr 取 {1e-4, 1e-3, 1e-2, 5e-2}，覆盖官方脚本用过的学习率。每个 lr 训练一次 2000 步，在 250 / 500 / 1000 / 2000 步各取一次读数。目标是域内平均再三域等权；平局时取更少步数，再取更低 lr。
- **比较**：24 个 Source 案例（每域 8 个：A1 / A2 两个时期 × 4 个实体组）× {None、P_NoMixRecipe、Comp[censor]、Comp[shock]} × 3 seed，共 288 次拟合。每次拟合结束时只用 E 输入窗冻结预测；全部拟合完成后才读真值评分。
- **接线**：同一个带子视图的拟合跑两次，结果逐位相同；它与 None 不同，说明子视图确实参与了训练。
- **并行**：CPU 进程数按接线实测的峰值内存和当时的空闲内存确定（上限 8），每个案例一个批次。
- **读数**：G 为相对同一 Consumer、同一案例 None 的百分比（seed 均值 E），先在域内平均，再三域等权；分别给出逐域、逐时期（A1 / A2）、逐起点、各 seed 以及胜负案例数。另附同 24 案、同 4 方案在 MLP（任务族筛选包）和 PatchTST（条件化学卡包的 Source 阶段）上的既有读数作对照；这三组数分别来自不同 Consumer 和不同运行，只能并列看，不能合并。
- **成本**：LLM 0 次；拟合 3 + 24 + 288 = 315 次。

## 3. 停止条件

报告写完即停止，不学新卡，不进入 Select 或 Test。

## 4. 回执

- 2026-09-24 18:17–19:06，本机 CPU，COMPLETE。报告 [_scratch/dev_aug_dlinear_source/REPORT.md](../_scratch/dev_aug_dlinear_source/REPORT.md)，机器读数 `result.json`。
- **核对**：smoke PASS（18,528 参数）；接线 PASS（带子视图的拟合重复两次逐位相同，且与 None 不同）；同 24 案的 MLP NoMix 分域读数（+16.16 / −6.17 / +28.87）与任务族筛选包记录的 Preset 对 None 一致。
- **校准**：选中 lr 1e-3、2000 步，目标 0.4174。同 lr 在 500 步时已达 0.4174，之后基本持平；lr 5e-2 发散（0.77–0.80）。
- **读数**（pp of DLinear None，三域等权）：NoMix +7.98（电力 +17.89 / 交通 −12.07 / 太阳能 +18.13）、Comp[censor] **+11.03**（+18.62 / +0.02 / +14.46）、Comp[shock] +10.23（+18.17 / −4.95 / +17.47）。三个 seed 同号。交通上 NoMix 与 shock 都是 0/8 胜，只有 censor 不伤交通（5/3）。
- **时期**：收益集中在 A1（电力 +29 ~ +33、太阳能 +31 ~ +38）；A2 很小（电力 +2.7 ~ +7.2、太阳能 −2.2 ~ +2.1）。时期差距大于方案之间的差距。
- **跨 Consumer**（同 24 案、同 4 方案，各自相对自己的 None）：不增强的 E 由好到差为 PatchTST < DLinear < MLP（电力 0.222 / 0.266 / 0.297，太阳能 0.328 / 0.427 / 0.587）；NoMix 三域收益依次为 PatchTST +0.66、DLinear +7.98、MLP +12.95。方案排序随 Consumer 变化：shock 在 MLP 上 +13.35，在 PatchTST 上 −3.82。
- **成本**：LLM 0 次；拟合 315 次（主比较 288 + 校准 24 + 接线 3），单次 8.3 s（带子视图 9.3 s）。空闲内存约 2 GB，按规则只开 1 个进程，整包墙钟 47.4 min；内存充足时（8 进程）约 6 min。
- **偏差**：第一次启动时，控制器在接线比较处第二次调用 torch 初始化报错（线程数不能重复设置）。已改为只初始化一次后原命令重跑，已完成的 3 次接线拟合直接复用，无拟合丢失，无读数影响。
- 按任务书停止：不学新卡，不进入 Test。
