# 任务书：DEV-AUG-DLINEAR-OFFLINE-SKILL（2026-09-24）

状态：用户指示“本地继续跑 DLinear 条件化离线学卡任务，复用已有 Source 拟合，暂不新增其他 Consumer（完整流程，在本地电脑上）”。入口 `evaluation/main_protocol_p4/batch_research_aug_dlinear_offline_skill.py`，输出 `_scratch/dev_aug_dlinear_offline_skill/`。身份 `EXPOSED_DEVELOPMENT_REVALIDATION`：40 个 Test 案例已经曝光，这次是开发复验，不是独立验证。

## 1. 设计

与 PatchTST 条件化学卡包（§5.11.7）完全同构，只把 Consumer 换成 DLinear。

- **Consumer**：官方 DLinear（cure-lab/LTSF-Linear 0c11366）。沿用 DEV-AUG-DLINEAR-SOURCE 的 CPU 训练循环和冻结配置（只用 None 在 Source 上校准：lr 1e-3、2000 步），不重新调参。
- **Fast / Slow / 朴素卡看到的描述**：都换成 DLinear。
- **流程**：
  - Source：24 例 × 12 个固定方案，外加 24 条无卡零反馈轨迹；
  - Slow：每域 2 张卡，只读 DLinear 上的证据；
  - Select：用原来的 J 选卡；
  - Test：四臂同预算，分别是新卡（f_dlin）、原 MLP 卡、无卡、DLinear 朴素卡；参照为 None、NoMix、DLinear Source 选出的固定方案；
  - 屏障与 PatchTST 包相同。
- **复用**：DEV-AUG-DLINEAR-SOURCE 已有 288 次 Source 拟合（24 例 × None / NoMix / censor / shock × 3 seed），按（案例、材料键、seed）直接复用，其余照常新拟合。接线时会在同一个进程里重训其中一个缓存单元，要求与缓存模型逐位相同，E 也完全相同。
- **传输**：流式传输，带 include_usage；每个请求最多重试一次，重试前等 90 s。阶段内如果还有被传输打断的轨迹，先等 600 s 再补跑一轮；仍有被打断的就停止该阶段，续跑时从断点继续，被打断的轨迹不会当作“不增强”交付。
- **并行**：LLM 12 路；拟合用 CPU 工作进程，每个进程一个线程，按空闲内存开 1–2 个。

## 2. 预算（同 PatchTST 包；API 按常设授权）

36M token、1450 个逻辑请求、1800 次拟合、墙钟 24 h。

## 3. 停止条件

读数和报告写完即停止，不自动修订卡片，也不引入新的 Consumer。

## 4. 回执

- 2026-09-24 23:1x 启动前：smoke 12/12；接线 PASS（在同一进程里重训一个缓存单元，与缓存模型逐位相同，E 也相同）。第一次冻结的计划说明里有一处文字错误（“预算同 PatchTST 包”被全局替换成了“同 DLinear 包”），当时还没有任何拟合或请求，删除后重新冻结。
- 2026-09-25 00:xx COMPLETE。报告 [_scratch/dev_aug_dlinear_offline_skill/REPORT.md](../_scratch/dev_aug_dlinear_offline_skill/REPORT.md)。
  - Test（DLinear None 的 pp，三域等权）：新卡 +8.64、原 MLP 卡 +8.80、朴素卡 +8.88、无卡 +1.11；NoMix +8.49，Source 固定方案 +7.64。
  - **新卡 − 无卡 +7.53，区间 [+0.58, +14.23]**（26/1/13，去掉任一时期仍为正）；新卡 − 原 MLP 卡 −0.16、新卡 − 朴素卡 −0.23，区间都跨零。
  - 收益形态与 MLP 包相同：卡把无卡 Agent 纠正回统一固定程序的水平。
  - 22.2M token、1002 次请求、1441 次拟合（0 失败）+ 288 次缓存复用；网络切换导致 10 条轨迹被打断，全部续跑完成。
