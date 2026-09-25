# DEV-CLS-AUG-WIRING：分类增强接线与元数据盘点（2026-09-25）

来源：Planner 意见经用户转来（2026-09-25），直接执行；用户追加：“验证阶段过了可以到服务器上跑，需要多头跑的可以并行开起来验证”。
定位：分类增强跨任务验证的**接线与耗时预检**，不是效果筛查，不启动学卡、AutoDA 联合训练或终评。

## 已定的设计（Planner）

1. **域 = 数据集族**：Source / Select / Test 按数据集分开；首版只做单变量、等长 UCR（UEA 后置）。目标 2 族 × (4 Source + 2 Select + 6 Test) = 24 个数据集；同源变体、共享原始记录的数据集放同一侧；不足时报告清单与缺口，不降门槛、不拆相关样本凑数。独立评价单位是数据集，3 个 seed 不算独立任务。
2. **样本下限**：TRAIN ≥ 100 且每类 ≥ 30；Source/Select 的 TRAIN 按类约 2:1 分拟合/反馈，每类拟合 ≥ 20、反馈 ≥ 10；有分组信息时按组划分。同意为名单补齐下载官方 UCR 数据，先按元数据筛选，本阶段只解析 TRAIN；封存数据保持原状态。
3. **归一化**：原始序列 → 逐序列 z 标准化 → 训练样本按固定概率增强 → 分类器；反馈/测试序列同样 z 标准化、不增强。所有臂同归一化、同样本数与更新预算；增强不增加步数；干净与增强样本同一批次前向。不能按 AutoDA 算子名套常用参数：先做“实际公式—参数含义—启用状态”短表，再冻结强度。
4. **技术预检与效果筛查分开**：GunPoint、PowerCons 的 TRAIN 上 None / Jitter / WindowSliceWarp 各 1 seed，6 次完整拟合，0 LLM；检查训练正常、标签不变、增强生效、评价数据未增强；测单次耗时、峰值内存、显存。FCN 为 Torch 重实现（对齐 padding、BN、初始化），不称逐位复现；2000 epoch 先实测再估预算。
5. **后续 Source 筛查的投入规则（预先写定，本包不执行）**：至少两个非同源 Source 数据集上，候选相对 None 的 Macro-F1 绝对变化 ≥ 1 个百分点且三个 seed 同向，才进入大规模学卡；正负方向都计入；两个接线数据集不计入。

## 本包交付

- 代码：`methods/ttha/cls_aug_data.py`（numpy：TRAIN 解析器、z 标准化、2:1 划分、指标）、`methods/ttha/cls_aug.py`（Torch FCN 重实现、AutoDA 官方算子封装）、`evaluation/main_protocol_p4/cls_aug_wiring.py`（smoke / ops / timing / precheck / 服务器多路 / inventory）。
- 输出：`_scratch/dev_cls_aug_wiring/`（REPORT.md 为主报告）。
- 停止点：接线结果与名单齐后，由 Planner 一次冻结完整实验规模与预算。
