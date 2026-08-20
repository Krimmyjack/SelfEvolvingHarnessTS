# 阶段报告:批次配方线(Batch Recipe Line)

日期:2026-08-21
范围:2026-08-20 ~ 08-21 的配方线工作(组合头寸 → 掩码搜索 → 配方工具 v1/v2 → Consumer 条件化 → 窗口外复核 → Agent 挂载 → 工具对照 → 经验冷暖对照与 LOCO 轮换)。
性质:按 AGENTS.md §11 的阶段性交付。所有数字均出自逐项审计过的冻结工件;代码栈经独立只读审查(`code_review_batch_recipe_stack_v1.md`,裁定 NUMBERS_TRUSTWORTHY)。

## 1. 方法 / Harness 行为发生了什么改变

- 新增一个确定性 0-LLM **批次配方工具**(`run_batch_composition_headroom.py --mode recipe`):菜单扫描 → 掩码贪心剔除搜索 → 冻结采纳规则。规则经一次带版本号的修正(v1 → v2:identity 成为延迟窗口在位者),修正由活体失败案例驱动并归档。
- 配方工具获得 **Consumer 结构条件化**能力(pooled / per_channel 变体,仅存在于实验 runner)。
- Fast Agent 的 Workspace 工具供给新增 **batch_recipe**(`agentic/gateway.py`,唯一 Harness 改动面;binding=None 时与旧行为完全等价)。该工具把延迟窗口数字带进 Fast 上下文,因此**永不绑进产出授权证据的 Task Episode run**(信息墙注记已写入工具描述与工件)。
- Experience 首次装入**内容已验证为正**的条目,并以 provenance 严格隔离:`batch_recipe_tool_v2_engineering` / `agent_hand_rolled_engineering` / `budgeted_search_engineering`,一律 `counts_as_unguided_exploration=false`,不作授权证据、不写 Skill、不授 TRY。
- 新增**预算化配方搜索**仪器(受限评估预算下的 shortlist 决策),用于经验价值的冷暖对照。

## 2. 真实数据上观察到了什么

证据链(全部为已曝光 development 数据;工件在 `artifacts/functional/e2/`):

| # | 结果 | 关键数字 | 工件 |
|---|---|---|---|
| 1 | 每个批次存在可验证正向方案 | 6/6 cell 延迟非负(+0.016 ~ +1.047) | `batch_recipe_v2_all_cells_v1` |
| 2 | 方案随批次变 | 三 cohort 冠军程序互不相同;Agent 历史习惯的 RLS 零次夺冠 | 同上 |
| 3 | 方案随 Consumer 结构变(核心论题) | 同批换 Consumer:程序/掩码/是否处理全翻;稳健 Consumer 吃掉清洗头寸(T233 identity 损失 1.458→0.738,处理增益 +0.117→+0.030) | `consumer_conditioned_recipe_v1` |
| 4 | 池化互作是 Consumer 结构性的 | 换 per-channel 集成后互作 −0.23/−0.25/−0.37 → −0.05/−0.02/−0.07 | 同上 |
| 5 | 采纳方案窗口外成立 | W1 方案在两个新窗口 12/12 延迟非负(v2 规则首次 out-of-selection 验证) | `batch_recipe_windows_v1` |
| 6 | 程序选择半稳、掩码窗口局部 | program_stable 3/6,mask_stable 1/6(且为空集凑数) | 同上 |
| 7 | Agent 闭环行为成立 | 初见调工具 / 复访复用 / Consumer 变化重搜,8 次 LLM 零失败;E3 在看得见旧方案时选择重搜 | `agent_recipe_mount_v1` |
| 8 | 工具对 Agent 的价值 | 同预算 3/3 全胜(延迟差 +1.047/+1.047/+0.225),LLM 省 2.6 倍;徒手臂两次弃权、一次选 hampel 留 +0.22 在桌上 | `agent_recipe_mount_notool_v1` |
| 9 | 经验价值(同成本质量) | #4:目标 A capture 0.367→0.962;#5 轮换:暖 4 / 冷 1 / 平 1,最差 −0.006;暖臂 4/6 逐位或近逐位复现全搜索答案(capture 1.000×3、0.989),评估预算仅 2/7 | `warm_vs_cold_recipe_search_v1`、`warm_vs_cold_rotation_v1` |
| 10 | 无经验 LLM 提案不随上下文条件化 | 冷臂 8/8 个目标给出雷同 shortlist(#4 两目标一张单、#5 六目标一张单),两次采纳延迟为负方案;暖臂 5 张不同单子、全含 outlier_iqr、随 cohort 与 Consumer 变化;hampel 暖臂 0/6 入选 vs 冷臂 6/6 | 同上 |

辅助发现:几何字段无法静态预测掩码剔除(批内反例 + 跨 Consumer 反例双重否定,`m0a_mask_geometry_census_traffic_v1`);traffic 的 union-pss 语义失真达 10/12(全部源于 outlier 区域,继续支持 M0 线的污染判断)。

## 3. 当前最大的方法不确定性

1. **成本维度未测量**:shortlist 填满无代价,"更省预算"结构上不可触发;主张定格为"同成本质量更好"。若要量成本需改读数结构(质量 − λ×评估数)或放宽上限,属下一版仪器设计。
2. **预算化采纳缺 identity 在位者门**(v2 配方规则有、预算化仪器没继承),导致一次两臂双输给 identity 的退化平局。对称不偏,但下一版必须预注册补上。
3. **迁移边界已现形**:批次特异答案(electricity×per_channel 的 denoise)在 LOCO 经验下结构上不可知,暖臂以 0.001 之差落败。跨批次经验的适用范围 = 机制在多批次重复出现之处;批次特异答案必须由目标本地探索补足(对应框架中的 A3 残差通道)。
4. **LLM 行为读数均为单次采样**(每 episode n=1,无重复方差定标);行为结论(路由/复用/单子分化)方向一致且跨 8 个 episode 重复,但单点数字不应过度解读。
5. **受害账是窗口局部性质**:聚合延迟增益窗口外保持(12/12),但逐序列受害数窗口外可增;受害控制依赖本地重搜掩码。
6. Weather 效用不可读(METRIC_UNREADABLE)未解;sealed 库存(NOAA、KDD W3)未动。

## 4. 与用户原始方向是否一致

一致,且是原始两支柱的第一个最小完整实现:

- "任务与模式感知的数据 Readiness 优化" → 配方随批次、随 Consumer 结构条件化(证据 1–4);
- "反馈驱动的 Harness 自适应进化" → 规则 v1→v2 由失败案例驱动;工具挂载源于徒手 Agent 的失败分析;经验使提案分布 context-conditioned(证据 9–10)。

A5-vs-A3 里程碑(相同预算下 Source 经验更快更安全形成 Target 方案)在配方层落地:同成本、质量 4/6 胜、最差损失有界、负迁移受控(暖臂零次灾难性失败,冷臂两次延迟为负)。未滑向 Router:决策由 Agent 读上下文做出,经验只调制提案与风险跳过;identity/弃权始终在位。

## 5. 下一个最有价值的纵向切片

1. **负路径自更新演示**(闭环最后一环):在低头寸 cell(T233×per_channel)的后续窗口上让复用撞延迟门失败,验证 负 Experience 写入 → 下一窗口重搜/弃权 的链条。
2. **收官确认实验**(需用户批准,烧 sealed 数据):方法全冻结后,在 NOAA 或 KDD W3 上一次性对决 完整管线 vs 冷启动。
3. 仪器修缮(并入上述任一步的预注册):预算化采纳补 identity 在位者门;成本读数结构调整。

## 工件索引

配方线:`batch_composition_headroom_v1`、`masked_single_program_v1`、`batch_recipe_{electricity,T233,traffic}_v1`、`consumer_conditioned_recipe_v1`、`batch_recipe_v2_all_cells_v1`、`batch_recipe_windows_v1`
Agent 线:`agent_recipe_mount_v1`、`agent_recipe_mount_notool_v1`、`warm_vs_cold_recipe_search_v1`、`warm_vs_cold_rotation_v1`
审查与普查:`code_review_batch_recipe_stack_v1.md`、`m0a_mask_geometry_census_traffic_v1`
Runner:`run_batch_composition_headroom.py`、`run_e2_m0a_mask_geometry_census{,_traffic}.py`、`run_e2_agent_recipe_mount{,_notool}_micro.py`、`run_e2_batch_recipe_windows.py`、`run_e2_warm_vs_cold_recipe_search.py`、`run_e2_warm_vs_cold_rotation.py`
