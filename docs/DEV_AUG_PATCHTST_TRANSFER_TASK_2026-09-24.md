# DEV-AUG-PATCHTST-TRANSFER

状态：用户已授权今晚运行第二 Consumer；TSFM 仅准备方案。2026-09-24。

目标：将 OFFLINE-SKILL 已冻结的增强交付材料用于 PatchTST，检验离线经验产物是否跨 Consumer 保留收益。这是 frozen-program transfer，不重跑 Fast/Slow、不让它们看新 Consumer 的 Test 结果，也不声称已经完成 Consumer 条件化学习。

设置：Electricity / Traffic / Solar，原 40 Test 案例、实体名单、时间切点及三 seed 20269181–83 均保持。训练历史 672、输入 192、预测 48；同一父窗池、实体 T scaler、增强子窗原字节、parent/child 0.5/0.5；None 只训 parent。服务输入与真值不增强。

模型：调用官方 PatchTST 204c21e 的 supervised Model（Apache-2.0），不改上游代码。3 层、128 维、16 头、FFN256、patch16/stride8、dropout0.2、RevIN 开启，公共扁平头。模型配置来自官方 electricity/traffic 脚本；本任务适配为 pooled univariate enc_in=1、192→48、batch64、AdamW(weight_decay=1e-4)。这是受控 Consumer 替换，不是原论文标准长预测基准复现。

对照六臂：None、NoMix、原 F0 交付、朴素卡交付、共享卡交付、域卡交付。相同 case/material/seed 只训一次，共 567 个物理测试拟合。朴素卡原先两条未完成轨迹继续按原主表规则记 None；不重新询问 Agent。今晚不移植 AutoDA 或扩大数据域。

训练设置先在 Source 确定：每域 A1/A2 的 G1，共六案，None 材料、首 seed；学习率 {1e-4,1e-3}，各训 2000 步，检查点 {250,500,1000,2000}。以 Source C_A+C_B 的四起点 nMSE、域内等权再三域等权选唯一 lr/步数；精确平局优先少步数再低 lr。12 条轨迹、禁止以增强优势作为选择目标。冻结后所有测试臂共用。其余模型参数不搜索，Test 不早停。

运行顺序：合成形状/梯度/种子/标签屏障检查 → T 材料绑定 → Source None/NoMix 100 步真实接线及 NoMix 重跑一次 → Source 校准 → consumer_frozen.json → Test 全部拟合 → 全局模型冻结 → 外部 E 评分与完整报告。测试 E 不参与调参；这些案例原本已曝光，身份仍为开发级补充迁移实验。

并行与恢复：本机 Windows project 环境、RTX4060；一个控制器，最多两个 GPU 拟合进程，CPU 材料准备最多两路。每个拟合独立进程；完成记录校验参数后复用，失败保留并停止续派，不盲目重试或启动第二控制器。单拟合 1800s、整包 16h；567 Test +12 校准+3 接线=582 次拟合上限，0 LLM/HTTP费用、0 SHA、0 commit。不安装全局依赖，不碰其他进程。硬件实测成本记录到 wiring.json/cost_estimate.json。

主读数：PatchTST 自己的 None 作分母，三域等权 nMSE、各域 gain pp、域卡方案对 F0/朴素卡/共享卡/NoMix、时期与实体组聚类区间和逐案例表。对照 MLP 的方向、幅度与失败案例，不混用 MLP 分母；无论正负完成同一份报告，不追加针对 Test 的配置。

输出：`_scratch/dev_aug_patchtst_transfer/`。入口：`evaluation/main_protocol_p4/batch_research_aug_patchtst_transfer.py`。运行完成自动写 REPORT.md/result.json/tables.md，随后停止。


运行期技术记录（09-24 01:29）：Source lr=0.001 首条进程在约1500步后意外退出，无Python异常，原因未确定；0 Test拟合。保存原失败目录，原配置只恢复一次；增加1次失败重做额度，总上限583，科学配置/名单不变。启动后实际并发为1（资源约束）。见 instrument_recovery_1.json；这不是新的科学配置尝试。

第二次技术记录（09-24 01:30）：同一Source任务恢复后再次被控制台中断，退出码0xC000013A，发起者未知。子worker原来未显式使用CREATE_NO_WINDOW；已加无窗口启动和DEVNULL输入，独立无拟合检查确认子进程控制台句柄为0。保存两次失败工件；原配置再恢复1次，总拟合尝试上限584，其中科学拟合仍582。控制器更换为PID 41796，旧控制器已退出，不存在双控制器。后续状态以status.json为准。
