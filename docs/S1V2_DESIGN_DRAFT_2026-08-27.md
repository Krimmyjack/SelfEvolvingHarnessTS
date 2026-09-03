# S1-v2 演化课程设计稿(待 sol 终审;参数栏待 M-1 判词回填)

状态:参数已回填(M-1 = MARGIN_GATING_CONFIRMED,提交 55f1d1e),待 sol 终审后发车。

M-1 回填要点:课程全程采用**对半切片协议**(Support = r1s+r2s、delayed = r1d+r2d、单轮双门,
M-1 实证:GPMvF 供给转化 0/4→2/4,+0.1867,harm 0;A3 冷转化亦 3/4——门控主体是确认面);
单元余量一律按对半协议由 ps0b 密封读数算术重算;
**注入可靠性如实设计**:prepare→identity-only 残留使注入约 2/4-3/4(PS-2/G3/M-1 三场一致),
按 sol"不再修 wiring"裁定不追修,S1-v2 的处理组存在性预检与读数按此可靠性设计
(supply 事件按实际注入计,漏注入轮如实记录,不作转化失败计)。

## 目的(任务书对应)

第六句 Skill/Memory 轴主考:同一 Harness 顺序穿过多单元,A5-online 的跨单元写回是否产生
可测复利(regret/成本下降,质量与 harm 非劣),并兑现"训练效率"读数。

## 四臂(冻结)

Static(不适应)/ A3-reset(每单元 H0)/ K0-fixed(与 A5 同一 K0 起点,单元间禁写回)/
A5-online(同 K0,单元间完整 Slow 整合:guard 通道 [B 修] + 供给卡编译 [W-1 通道])。
K0 = bootstrap 三卡 + 惰性 Slow 卡;不含任何 Target-local capability 与既有双源卡
(A5 的供给卡必须课程内自产,这是"演化"与"带资进场"的区别;PS 系列双源卡不入 K0)。

## 课程(参数待 M-1)

- 协议档:**[待 M-1]** 若确认余量门控 → 全课程统一用对半切片协议(所有臂同协议,预注册;
  依据 = M-1 因果证据:可读反馈面是转化前提;协议是仪器选择非调参)。
- 单元池:PS-0b ROBUST/FRAGILE 表按对半协议重算余量后,机械选 6-7 单元:
  ≥2 个高余量(≥2×)可学 + ≥2 个低余量可学(分层对照,预测 A5 优势集中高余量层)+
  2 个 identity + 1 个 HELDOUT_ONLY;全课程家族尽量互异;
  **排除**:GPA/PowerCons(双源)、GPOvY(已用考场)、GPMvF(M-1 用)。
- 顺序:正序一套,预注册;**反序仅在正序出信号后跑**(sol 收紧)。

## 处理组存在性预检(开跑前,0 LLM)

从课程构成算术推演:A5-online 在课程内能否合法形成 ≥1 张供给卡
(2 个未受引导可学正例、Scope 相容;按对半协议余量)与 ≥1 个后续匹配可学单元。
推演空 → 不发车,课程重组(S1a-r2 教训:先证处理组存在)。

## 读数(每单元每臂)

终局部署、regret(配 harm/worst-class 非劣,禁单引)、held-out 与逐类 recall、
错误晋升、**适应/搜索效率**(LLM/probe/fit 计数)、**训练效率**(consumer fit 墙钟合计、
time-to-threshold:达到质量阈值所耗 fit 墙钟)[sol 措辞收紧]、
supply/guard 事件时间线(何单元产生知识、何单元生效、探索槽保留证明)。
聚合:四臂累计曲线;A5-online 的 Slow 开销计入总账。

## 判分口径(sol 收紧,2026-08-27 17:0x 增补)

- **主分析 = ITT**:Scope 合格但注入失败的轮,计入 A5 系统失败;"成功注入后的条件转化率"只作次级读数另报,不入主判。
- **材料门(数值化,课程名单冻结时代入常数)**:
  - regret 门:A5-online 累计 regret 低于 A3-reset 与 K0-fixed 各至少
    Δ_material = max_u(1/n_slice_u)(取课程各单元对半协议下最粗切片材料线的最大值);
  - 成本门:质量与 harm 非劣前提下,累计适应成本(probe 数为主尺)每个可转化单元平均省 ≥1 次完整 probe;
  - 非劣容忍:held-out 质量差 ≥ −0.005,worst-class 差 ≥ −0.005,harm 事件数不高于对照。
- **供给档语义引用 P0**(2 独立未引导正例 → supplies_candidates-only 卡,production 编译;3+LOO 旧档不动)。
- **课程冻结件**(发车前产出):具体单元名单+顺序+transfer graph+预计产卡边界(哪两单元的
  Episode 在哪个 Slow 边界编译出卡)+预计首分叉单元(A5-online 与 K0-fixed 首次可见差异)。
- **重复计划**:正序 ×2(不同注入 seed,预冻)+ 信号后反序 ×1;反序不替代重复。
- **效率措辞**:主指标称"适应/搜索效率"(LLM/probe/fit 计数);"训练计算效率"仅由
  consumer fit 墙钟与 time-to-threshold 两读数支撑,不越界表述。

## 判词(冻结)

- `S1V2_FORWARD_SIGNAL`:A5-online 质量与 harm 非劣于 A3-reset 与 K0-fixed 前提下,
  累计 regret 或累计适应成本材料级改善,且优势可归因于课程内自产知识
  (supply/guard 事件与后续单元行为变化的 trace 链)→ 解锁反序+重复;
- `PRIOR_ONLY` / `NO_TRANSFER` / `NEGATIVE_TRANSFER`(附 first-fault 初诊);
- `TREATMENT_EMPTY`:课程内未产生任何 Fast 可见知识 → **立即停**(sol 收紧),不烧完全程。
- 分层预测(预注册):A5 优势集中高余量单元;低余量单元 A5≈A3(若 M-1 确认)。

## 预算

正序:LLM ≤400 / fit ≤900 / 墙钟 ≤4h(硬,checkpoint+resume);
反序同额,仅信号后发。证据等级 development;单序单跑判词封顶 FORWARD_SIGNAL。

## 与 capstone 的衔接

S1V2_FORWARD_SIGNAL 且反序确认 → 按 CAP-0 冻结协议自动开封 Epilepsy2 子集,
Static/A3/A5 同场终考(A5 携 S1-v2 终态池),无需再次授权。
