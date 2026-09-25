# DOMAIN-SKILL-V1：快速构建计划与第一包任务书

日期：2026-09-16  
状态：可分发；尚未执行。用户转发并要求执行后，首包按本文授权连续完成。  
负责人：一名熟悉 W 线的执行者，建议 Opus；不自动发送、不默认另起并行实现。  
方法方向：借鉴 Eval-Skill 的“每领域一份 Skill、离线探索选优、部署复用”，迁移到现有批级时序训练材料准备 Harness。

## 0. 这次要做出的东西

一条可以实际运行的链：

```text
当前批次合法 T → 程序计算数据画像
→ 模型根据画像选择一份冻结领域 Skill，或使用公共流程
→ 同一个批级 Fast 观察、构造、取得 C_A、比较、commit
→ 全批材料共同训练固定 Consumer
→ commit 后 C_B；全局预测冻结后 E
```

Slow 离线从同领域的多个合法任务及轨迹提炼不同 Workflow，真实比较后选出一份；后续再做一次有边界修订。不要求每个批次即时修改知识，不建复杂检索或自动发现 domain 的平台。

本版是开发级领域 Skill 组件，不冒充完整 A3/A5 系统验收。任务内 Fast 搜索不等于 A3；完整 Target 边界适应与最终零反馈部署仍是后续目标。

## 1. 快速计划：三包，第一包现在可发

| 阶段 | 实际工作 | 交付终点 | 时间安排 |
|---|---|---|---|
| A：本任务 BUILD | 两域数据参数贯通；画像、领域档案、Skill、匹配与 Fast 接线；离线优化入口；真实 T 冒烟 | 能通过同一入口演示“画像→选卡→加载→材料构造”，真实训练/LLM入口可配置，尚未验证效果 | 一个工作时段，目标 4–6 小时 |
| B：EVOLVE-PILOT | 冻结形成/选择/测试作业；补足缺失的合法 Source 轨迹；每域生成 2 个 Workflow，真实选优；有预算才加一次 Principles | 每域一份实测选出的候选 Skill，或明确无候选；完成小规模真实贯通 | A 回执后冻结具体数据和总预算；目标下一个工作时段 |
| C：DOMAIN-TRANSFER | 新批次比较无 Skill、通用 Skill、对应领域 Skill；再检验画像匹配/错配；保留固定处理与简单搜索 | 有效果、成本和适用边界的可解释结果，可据此开始写方法与结果 | B 后按实测费用安排 |

时间是工程排期估计，不是效果保证。A 不以正收益为完成条件；B/C 不以“必须出现正号”为停机条件。A 完成后直接制定有数据与预算的 B 包，不再插入一轮独立审计、分辨率扫描或应用方向调研。

当前先复用 Electricity、Traffic 和已有增强动作空间，避免同时重构 Consumer、算子与评价。算子扩容单列后续，不能把三类增强上的结果外推为完整 Data Readiness 已成立或失败。

## 2. 首包范围与硬边界

首包名称：DEV-DOMAIN-SKILL-V1-BUILD。

本包授权：代码接线、必要配置与提示词、真实 T-only 画像/材料构造、脚本化客户端测试、必要回归。  
本包资源：**0 新 Consumer 拟合、0 实验 LLM/API 调用（含收费探活）、0 新 C_A/C_B/E 标签读取、0 新 SHA/Hash、0 安装依赖。**

0 预算是首包的执行范围，不是把效果问题取消。收费能力要接到真实运行代码；本包以脚本化客户端验证控制流，不得把它报成真实模型实验或 Skill 效用。

一个工作时段内完成授权工作；至 6 小时仍有方法性阻断时，交付可用部分、具体阻断与剩余改动，不自行追加平台。普通实现问题自行解决，不逐项等待 Planner。

不得修改历史工件、P/G 工作树、参考仓库、AGENTS.md、历史判词；不 git commit/push。保留用户与其他执行线的改动。只允许根执行者委派一层、具体有界子任务；子 Agent 不再派工。

## 3. 必读与复用路径

先完整读取工作区和项目 AGENTS.md，批级几何以 §5.9 为准。重点复用：

- methods/ttha/batch_research.py：Knowledge、Guidance、apply_update、scope_state、run_job。
- methods/ttha/batch_base/spec.py、data.py、context.py、observe.py：两域注册、阶段切片、画像。
- methods/ttha/batch_base/policy.py、materials.py、train.py、feedback.py、commit.py：公共执行与评价。
- evaluation/main_protocol_p4/batch_research_runtime.py：Adapter、MeteredClient。
- evaluation/main_protocol_p4/run_batch_research_v1.py：branch、resume_branch。
- evaluation/main_protocol_p4/run_batch_research_roundtrip.py：已修复的失败、扣留标签、恢复边界。
- docs/BATCH_RESEARCH_WORKFLOW_FRAMEWORK_2026-09-13.md。
- 旧 source_process / skill_revision 模块：只参考证据整理与调用方式，不复活旧实验。

已核实的具体接线缺口：底层 spec/context 支持 electricity 与 traffic，但 rt.Adapter.__init__、base.branch 的 commit 以及若干 Runner 路径写死 electricity。不能只改画像入口而让训练、commit、后阶段或恢复仍走电力数据。

建议新增一个小的领域 Skill 模块和一个薄 study/命令入口。允许给公共函数补默认兼容的 dataset 参数；默认 electricity 的旧调用语义保持。复用公共 fit/评分器，不另写一套。执行者可自行决定必要文件拆分，不建设通用注册平台。

唯一新运行产物目录：
_scratch/dev_domain_skill_v1_build/

## 4. 方法合同：domain、画像与 Skill

### 4.1 三个对象分开

- Domain：第一版以 Electricity、Traffic 两个业务来源组织形成证据，采用中性 ID D01/D02。
- Batch profile：某个 Job 自己 T 段的特征摘要。
- Domain Skill：由该域多个合法 Source 任务提炼的 Workflow；附可选 Principles、适用边界和证据引用。

同域一份 Skill 不意味着全实体同一个算子；Fast 仍能按观察生成不同处理，也能统一处理或正常选择 None/Fixed。

本包是用户最新领域 Skill 方向的有界实现。已知 domain 标签可作诊断对照，但不能被当作“跨域相似性已验证”，也不能映射到预写答案。主画像匹配不使用数据集名称、路径、Job 名、真实实体 ID 或结果分数作为特征。不修改现役 policy 的标识符禁令。

### 4.2 画像：一份统一结构，复用已算字段

复用 overview 的实体统计与 batch quantiles，第一版使用以下无量纲描述：

- lag24_corr、lag168_corr；
- nondc_spectrum_top5_energy_share；
- standardized_trend_per_100h；
- last168_std_over_full_T_std；
- standardized_abs_p95、standardized_abs_max；
- r_head、r_tail、r_gap；
- nn1_input_distance、nn5_target_deviation。

每项保留 p25/median/p75、有效实体数；可用现有 min/max，不发明效果阈值。Task、Consumer、L/H、采样间隔、T 长度单列为兼容条件。未知值仍为 null，不置零冒充正常。

当前 loader 要求所选 T 全有限；observe 的 missing_count=0 是此资格合同下的常数，不是完整缺失诊断。不要据此声称已实现自然缺失场景。

内部绑定数据路径、roster、绝对切片等只在 Runtime 保存。对模型的匹配视图剥离这些身份字段。数值由程序计算，LLM 不生成统计值。

Domain profile 由已指定的形成批次画像组成：先保留少量批次的上述摘要，不强迫合成一个平均向量，不训练 embedding。目标批次不得参与其参照画像拟合。

### 4.3 Skill：最小文本载体

复用 Knowledge/Guidance，至少保留以下语义字段，不要求新建全部独立 schema：

- 中性 skill_id / domain_id / revision；
- Workflow 正文；
- 可选 Principles；
- Task/Consumer 兼容说明、显式 observable_applicability；
- evidence_refs、来源阶段、状态；
- rendered_body：实际进入 Fast 的文本。

第一版将 Workflow 与可选 Principles 合并为一个现役 experiment_guidance 条目，沿用正文 1200 字符上限。这是兼容载体，不把内容收缩成固定菜单；可以描述观察、构造、比较与停止。不要拆成多个 hook 绕开上限。

显式 {"const":true} 合法；null/缺字段仍不完整。匹配器选中后，现役 scope 仍需检查；UNKNOWN/NO_MATCH 不加载该条，但 Fast 正常运行。选择 Skill 不授予新动作权限。

手写测试卡只能标 SMOKE_FIXTURE，使用通用过程性内容，不写“电力应 Mixup、交通应 FreqMask”之类效果答案；不得伪装成 Slow 学得的知识或进入正式 Skill 选优。

## 5. 接通两个匹配入口，主实验不混淆

### 5.1 known_domain：诊断参照

按外部已知 domain_id 取对应冻结 Skill。无需 LLM，不调用特征分类器。日志明确这是“已知域路由”，不是模型推断成功。

### 5.2 profile_match：主画像匹配

一次短模型调用接收：当前 Batch profile、候选 Domain profiles、兼容条件、每份 Skill 的短适用描述。输出：

- selected_skill_id，或 null；
- status：SELECTED / ABSTAIN；
- evidence_fields：引用实际存在的字段；
- rationale：为什么认为适用，或为什么信息不足。

不返回候选配方，不调用 Consumer，不读取 C_A/C_B/E，不自行放宽 scope。只从提供的 ID 选择；没有合适项可弃权，进入同一公共 Fast 起点，不能变为强制 identity 或跳过 Fast。

形成失败、匹配传输/解析失败、模型主动 ABSTAIN、合法选中但 scope 未通过分别记录。不得把费用/接口故障改写为正常弃权。首包不加依据 LLM 自报 confidence 的数值门。

主匹配视图使用中性 ID，不暴露业务来源名称。模型知道的是统计与任务，不靠读文件名“猜中”。领域身份预测准确率只作诊断；是否选到有用 Skill 由后续真实效用检验。

路由每 Job 一次且冻结，不能根据该 Job 的 C_A 反复换卡。旧 Agent/现役 General 起点保持相同；未知匹配不扣掉其本来的工具和训练机会。

## 6. 离线生成与选优入口：现在接通，下一包付费运行

按同一入口分成显式阶段，不要强制长链多轮演化：

1. census：读取明确 Source 白名单，整理完整批级 policy、真实轨迹、C_A 与已合法打开的 C_B、成本和失败。Fast 永远不读这些原始跨 Job Episode。
2. propose：Slow 每域最多提出 2 个 Workflow，或 KEEP/无候选；两者需要不同处理/研究方式，不能要求必须换算子、必须复杂或必须激进。
3. select：候选在预先指定的开发 Job 上执行完整 Fast，以同预算的实际 Consumer 结果比较；不由 Slow 自评好坏选卡。
4. freeze：记录选中版本、选择依据及证据范围，保留无 Skill 选项；可选 Principles 增补单独标记，不偷改已冻结版本。

本包只实现构造、调用、结果校验、候选运行与冻结接口，以脚本化 fixtures 验证。禁止从历史分数反推手写最终 Skill，也不在本包按真实 C/E 筛选候选。

来源边界按 dataset + roster + 时间绑定。Traffic 若只有固定算子实验而没有真实 Fast 轨迹，应如实记“效果证据存在/过程轨迹不足”；不能把 Electricity 轨迹改标签充当 Traffic，更不能为凑资料打开未来测试数据。B 包可在预算内补形成轨迹。

A 包在 source_assets.json 只登记现有路径、来源域、证据类型和缺口；不用重新统计所有历史效果，不读取整份含 E 的报告喂模型。当前 Source 输入默认不含 E。

B 包再冻结形成、选择与未来测试批次。选择集属于开发；不能把重复使用的同一数据改名为独立验证。同数据集的新时间/实体批次复用和跨数据集泛化分别表述。

## 7. 两域贯通与信息墙

dataset 参数从 Job 创建贯通到画像、材料、fit 子进程、缓存、commit、C_B、E 和恢复。旧默认参数应逐位保持现有材料数值。

输出至少按 domain_id/job_id/branch 分隔，避免两域都有 a30 时覆盖 scaler、模型或缓存。缓存命中必须核对来源、人口、训练切片、候选和种子；用已有语义字段，不新增哈希。

本包只真开 material 阶段；收费和标签阶段在本包配置下应在实际调用/读取前拒绝。脚本化测试使用显式合成分数，不把它们混进真实读数文件，不生成伪“真实 COMPLETE”实验报告。

未来 live 路径仍复用：全部臂 commit 冻结 → C_B → 全局 E 预测冻结 → E 真值。保留未知 usage 停收费、标签未扣留不可恢复等既有修复，不重新改变失败策略。

## 8. 本包真实 T 冒烟数据：按身份选，不按效果选

两域均使用 spec.DATASETS 现有原始字符串序前 32 列，不替换实体。以下仅是工程冒烟与开发画像，全部标 EXPOSED_DEVELOPMENT；不是未来独立验证集。

| 用途 | Electricity | Traffic |
|---|---|---|
| 领域参照画像 | a30、a40 的各自 T | a30、a40 的各自 T |
| 查询画像/材料构造冒烟 | a50 的 T | a50 的 T |

由现役 job_spec 产生精确窗口，不手工复制另一域的 t；两域总长度不同。同一个 a 不代表相同日历时间。

真数据部分只做：
- 6 个 T 画像以及 2 份领域参照档案；
- 两个 a50 的 None 与 Fixed-Mixup 材料构造；
- 与直接调用现役公共底座的 scaler/父对/材料数组比较；
- 实际 Fast 请求渲染中 Skill 的来源与正文检查。

不训练、不评分、不读 C/E。发现零尺度或非有限只将该真实样例记不合格，不换 roster/切点；继续其他健康样例和合成控制流检查，回报具体缺口。

## 9. 有界检查与完成标准

合并为一个必要 smoke，不新建审计包或大型测试矩阵：

1. 两域各自 T、人口、scaler/材料正确绑定；相同 a 不串数据；旧默认电力材料不变。
2. 去掉身份后的 profile 不含路径、UID、Job、C/E；参照画像不含查询批次。
3. known_domain 与脚本化 profile_match 都能向同一个 run_job 加载指定 fixture；保存实际渲染文本。
4. 未知/不匹配仍有正常 Fast 权限；null scope 不变无条件；{"const":true} 可加载。
5. workflow_only / workflow+principles / KEEP / 失败分别走通；未通过条件不晋升。
6. 0 预算能阻止真实网络/fit/标签读取；未来 dataset 参数到各阶段与恢复的调用绑定有检查。
7. 仅在控制流测试使用显式 synthetic 反馈；真实 T 材料不冒充训练效用。
8. 运行与改动有关的既有控制测试/故障 smoke；通过即停止，不补跑所有历史课程。

BUILD_COMPLETE 的含义：真实两域画像及材料路径可用，领域卡通过实际 Fast 请求路径加载，离线生成/选择/冻结和 live 适配接口已接通。它不等于模型路由准确、Skill有用、完整两域训练已实跑。

若只写了 schema/README、只有 mock 且没有真实 T/材料路径，不能记 BUILD_COMPLETE。真实样例不合格、接口或环境阻断应分别报告，不伪造通过。

## 10. 交付与首包结束

尽量只有以下必要产物，放在同一 scratch 根：
- REPORT.md：做成什么、真实/脚本化各验证什么、未验证什么、实际成本、阻断。
- profiles/ 与必要 domain catalog、fixture Skill、实际请求/路由记录。
- smoke.json：关键通过/失败、0 拟合/0 API/0 新标签读取的实际记录。
- source_assets.json：后续形成资料可用性与缺口，少量路径即可。
- 在 REPORT.md 中给出三个可复跑命令：画像构建、集成 smoke、带显式预算配置的未来 live 调用；标清第三项未执行。
- 在 REPORT.md 中给出 B 包最小配置草案及总成本算式，不自行选新测试批次开跑。

总代价表分开：形成/优化的一次性成本，Router 调用，Fast 调用，真实 Consumer 拟合。后续不能把路由成本藏进公共基线，不能只报选中 Skill 一条分支的花费。

需要用户/Planner定的只留实质项：新增数据权限、模型身份/付费预算、正式形成/选择/测试作业。常规参数贯通、字段组织、提示词格式和测试修复由执行者连续完成。

本包完成即收口；不自动启动 B/C，不重启旧包，不做额外 headroom 或分辨率实验。

## 11. 方法来源与本项目新增部分

- Eval-Skill v2，§3/§4.3/附录B：每域一份技能、Workflow 与 Principles 分离、离线探索与选择。https://arxiv.org/html/2606.07040v2
- 作者代码：预定义 domain/subset 与说明。https://github.com/xing-stellus-yue/Eval-Skill/blob/main/src/eval_skill/data.py

论文优化的是有偏好标签的评判任务；本项目仍需真实共享 Consumer 重训来评价材料效用。画像匹配和训练材料处理是我们的迁移设计，不写成论文已验证结果。当前限定为组件快速实现，不因时间紧缩改成传统数值 Router 或硬编码领域答案。
