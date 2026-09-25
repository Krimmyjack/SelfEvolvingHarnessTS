# 任务书：DEV-AUG-PATCHTST-OFFLINE-SKILL（2026-09-24）

状态：用户转来 Planner 安排，执行者盘点缓存缺口与预算后，用户于 09-24 确认预算和四项决定，随即启动。入口 `evaluation/main_protocol_p4/batch_research_aug_patchtst_offline_skill.py`，输出 `_scratch/dev_aug_patchtst_offline_skill/`。

## 1. Planner 原文（用户转来）

> 目标是检验：使用 PatchTST 自己的离线反馈学习，能否改善零反馈部署，并纠正直接迁移 MLP 经验造成的损失。
>
> 1. 固定已有 Consumer 配置：复用 PatchTST 已冻结的学习率、500 步训练和数据处理方式，不根据已曝光 Test 调参。
> 2. 补齐 Source 经验：沿用三域原 Source 案例，在 PatchTST 下取得公共参照、代表性增强及无卡 Fast 提交方案的真实效果。复用已有材料与同配置缓存；Fast 明确看到 PatchTST 的 Consumer 描述。不要把 MLP 的效果标签当作 PatchTST 监督。
> 3. 形成与选卡：每域生成两张候选卡，在原 Select 上按零反馈部署实测选优。保留完整构造权限，允许不增强；不预写“强增强有害”或指定正确方案。
> 4. 比较对象：PatchTST 专属学习卡、原 MLP 卡、无卡、朴素卡；各 Agent 臂使用相同 Consumer 描述与工具预算。保留 None、NoMix、PatchTST Source 选出的固定程序作参照。原先冻结材料的迁移结果单列，不能冒充本次真实 Fast 部署。
> 5. 评价身份：已有 40 个 Test 已曝光，复用只能记开发复验；先核查是否有可用的未评分实体或时期，再确定是否增加独立验证。任何 Test 结果不得进入 Slow 或选卡。
> 6. 并行和交付：独立 API 调用并行，训练并发按实测内存控制；先根据缓存缺口列出新增拟合、token 和耗时，再冻结整包预算。报告重点回答“重新学习改变了什么、是否改善交付、与简单固定方案差多少”。
>
> TSFM 并行完成接入方案与成本清单，维持准备状态。

## 2. 执行前盘点（0 LLM、0 拟合）

- **缓存**：PatchTST 迁移包的 Source 校准只存了 C_A+C_B 验证分，没有模型和 E，Source 上无可复用拟合；24 Source × 12 固定方案的材料全部在筛选缓存中。Test 上迁移包的 567 个 PatchTST 拟合按（案例、材料键、seed）复用。
- **Consumer 描述**：父包 Fast 系统文本写死 “one shared MLP Consumer”，案例卡带 `ec.CONSUMER`（MLP）；原域卡与朴素卡正文也提到 MLP。新入口只改 Consumer 描述，其余文本、工具、字段、上限不变；旧入口不动。
- **未评分实体 / 时期**：三域 Test 的 E 已到数据末端（电力 26304/26304、交通 17496/17544、太阳能 8712/8760），无更晚时期。Test 四锚点全部合格且从未使用的实体：电力 13、交通 554、太阳能 9；案例固定 16 实体，只有交通能组新案例。Source 实体在未评分 Source 锚点上的案例与 Source 证据同实体，电力 / 交通窗口还与更早一包已评分窗口重叠，不算独立验证。
- **预算估计**：拟合最坏 1768 / 预计约 1460；token 点估计约 24M、保守约 33M；PatchTST 单次拟合约 22–26 s。

## 3. 用户确认（2026-09-24）

1. 按上限冻结并启动：36M token、1450 逻辑请求（HTTP 1520、额外传输 40）、1800 拟合（重试 40）、墙钟 24 h。
2. 朴素卡按 PatchTST 描述重新生成（一次生成加至多一次格式纠错，不选优）。
3. 不加独立验证；40 例 Test 记为开发复验（`EXPOSED_DEVELOPMENT_REVALIDATION`）。
4. Test 每条提交一冻结即训练；全部 160 条提交冻结、全部模型训练完成后才统一打开 E。
5. API 并行 8–16（用户追加），本包取 12 路在途请求，每个 Fast 阶段 12 条轨迹线程。

## 4. 执行规格（入口冻结于 plan.json）

- **Consumer**：官方 PatchTST（204c21e），lr 1e-4、500 更新、batch 64、AdamW wd 1e-4，parent/child 0.5/0.5；与迁移包逐位同训练回路。
- **Source**：24 例 × 12 固定方案 × 3 seed（864 拟合）；24 条无卡零反馈 Fast（PatchTST 描述），提交方案按键去重后三 seed 拟合。24 条提交全部冻结、所有 Source 模型训练完成后才打开 Source E。
- **Slow**：每域 2 张候选域卡（共 6 次调用），证据只含 PatchTST 的 Source 结果；不给 MLP 卡、不给 MLP 分数；证据文字不预设哪种增强好或坏；一次格式纠错；KEEP 合法。无共享卡（比较对象中没有）。
- **Select**：原 24 例 Select，每例跑本域候选卡（别名去重），J_d = 域内均值 E(c,W)/E(c,None)，argmin；精确平局按部署 token、槽位顺序。
- **Test**：40 例 × 4 臂（f_patch 新卡、f_mlp 原 MLP 域卡、f0 无卡、f_naive PatchTST 朴素卡），相同 Consumer 描述与工具预算；参照 None、NoMix、Fixed_source_P（PatchTST Source 8 例平均 pp of None 最高的固定方案）。f0 / f_naive / f_mlp 开跑即派发，f_patch 在选卡后派发。
- **并发与资源**：GPU 按内存开 1–3 路（resources.json 可在运行中调整；第二路起须空闲内存 ≥ 1.6 GB），同一案例的拟合合批进同一进程；CPU 材料构建 2 路；单控制器、包锁。
- **身份**：开发复验。迁移包的冻结材料结果在报告中单列。

## 5. 启动前检查

- smoke 11/11：Fast 文本与案例卡为 PatchTST 描述；88 案例 scaler 与父包逐值相同；父窗与迁移包逐位相同；迁移缓存 567 条；脚本客户端零反馈提交链路；0 拟合；父包未被写入；MLP 卡可加载。
- 接线 PASS：同一进程第二次拟合（重训迁移包一个缓存单元）与迁移包模型逐位相同，E 与迁移包存值相同；单次训练 22 s；拟合进程峰值内存 1.23 GB。两次接线拟合计入本包账本（其中 Source 那次被正式运行复用）。

## 6. 回执

- 2026-09-24 启动：控制器 PID 10144，GPU 两路。实际进度以 `status.json` 为准；完成后在本节补收口。
- 13:28 控制器被 Claude Code 在内存告急时回收（无在途 LLM；38 次在途拟合重训），用户 14:15 独立窗口续跑（PID 23232）。
- **收口（2026-09-24 16:10 COMPLETE）**：报告 `_scratch/dev_aug_patchtst_offline_skill/REPORT.md`。Test（pp of PatchTST None）：f0 +0.38、f_naive −0.32、f_patch −2.09、NoMix −1.68、Fixed_source_P −3.40、f_mlp −5.31。f_patch − f_mlp **+3.22** [−1.10, +8.36]（电力 +13.4，太阳能 −4.45）；f_patch − f0 −2.47 [−11.65, +3.42]（损失集中在电力 Q4：D01_Q_Q4_G1 −93.9，NoMix 同案 −90.8）；f_patch − Fixed_source_P +1.31（27/40 同交付）。22.50M token、1050 请求、1392 拟合尝试（690 缓存）、付费 4.38 h；4 次瞬断自动接受；0 拟合失败。
