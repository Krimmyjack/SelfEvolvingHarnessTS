# DEV-CLS-AUG-OFFLINE-SKILL：分类任务上的离线经验迁移（2026-09-25）

来源：Planner 最终修订经用户转来；用户指示“直接用空闲的卡，尽快推进部署”“卡多可以跑三折，卡少就算了”（启动时只有 4、5 号卡空闲 → 一次划分）。
问题：预测线上使用的离线经验学习机制，能否帮助零反馈的分类 Agent 为**未参与学习、来源互斥**的数据集选择更有效的训练增强。
定位：跨任务补充实验；不改变预测主实验，不承担“每种 UCR 类型需要独立域卡”的证明。

## 冻结设计

- **名单**（[CLS_AUG_ROSTER_V1.json](CLS_AUG_ROSTER_V1.json)，方案 a）：24 个独立来源各取一个合格代表（冻结 seed 2026092502）。官方 TEST 从未被任何旧线打开的 12 个来源全部作 Test；曾被打开过的 12 个按 seed 分 8 Source / 4 Select。实验定位：在以二分类为主的开发数据上学习，迁移到来源互斥的多分类数据集；结果不能单独归因于类别数变化。
- **Consumer**：FCN（Torch 重实现，按 dl-4-tsc 与 Keras 语义对齐），2000 epoch，批大小 int(min(n/10, 16))，训练损失最低的权重。
- **程序空间**：Jitter（AutoDA 官方，σ 0.03）、Scaling（Iwana & Uchida 定义，每条 N(1, 0.1)）、MagWarp（Iwana & Uchida 定义，σ 0.2、6 个结点的三次样条）、WSlice（AutoDA 官方 WindowSliceWarp，strength 0.2）；无增强 / 单算子 / 两个不同算子按 Scaling → MagWarp → WSlice → Jitter 执行，共 11 个程序；整条程序以 0.5 概率作用于训练样本。先逐序列 z 标准化，再增强，不重新标准化。
- **反馈**：Source / Select 在官方 TRAIN 内按类 2:1 分拟合/反馈（一次分层划分，seed 2026092500），3 个训练 seed；三个 seed 衡量训练随机性，不算独立任务。
- **拟合范围**：Source 11 程序全网格（264）；Select 只训练 4 张候选卡的交付；Test 只训练六臂的交付（完整官方 TRAIN），相同程序复用。全部 Test 交付冻结后才统一打开官方 TEST（一次）。
- **六臂**：不增强、无卡 Agent、朴素卡 Agent、经验卡 Agent、Source 最佳固定程序（8 个 Source 上相对 None 的平均 Macro-F1 变化最大者）、预先冻结的随机程序（每个 Test 数据集在 11 个程序中均匀抽一次，seed 2026092503，与训练 seed 分开；不称“随机策略的精确期望”）。
- **卡**：经验卡与朴素卡各两张（同样的两种写法），各自在 Select 上用相同机会选一张（J = 4 个 Select 数据集上交付程序的平均反馈 Macro-F1）。
- **指标**：Test 上 Macro-F1 的绝对百分点差为主，辅以 Accuracy、最差类召回、逐数据集胜负、token 与训练成本；独立单位是 Test 数据集（bootstrap 按数据集重采样）。

## 执行者默认（待 Planner 复核）

- **LLM 可见信息**：数据集名称与档案类型全部匿名（S01–S08 / V01–V04 / T01–T12）；Agent 只看 TRAIN 统计（字段定义随卡给出，含 1-NN 留一准确率这一数据难度统计，不是 FCN 结果）、`class_profile`、`inspect_program`（程序作用于 TRAIN 后与类模板的一致性，不训练分类器），最多 5 次请求、3 次检查；没有合法提交时交付 None 并单独计数。
- **调度**：服务器只用空闲卡（卡上出现他人进程就不再派新任务），Source 网格优先（它决定 Slow 的开始时间），Test 拟合随后；每卡 8–12 路。
- **LLM**：本机运行（服务器会话无 API 凭据），模型与传输沿用预测线（流式、有界重试、单一账本实例）。

## 输出

代码 `methods/ttha/cls_aug.py`、`methods/ttha/cls_aug_data.py`、`evaluation/main_protocol_p4/cls_aug_offline_skill.py`（拟合、调度、TEST 一次性打分）、`evaluation/main_protocol_p4/cls_aug_agent.py`（LLM 阶段、冻结、读数）；输出 `_scratch/dev_cls_aug_offline_skill/`。
