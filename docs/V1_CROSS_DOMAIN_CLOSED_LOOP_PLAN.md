# V1 跨域完整链计划（2026-08-08）

> 三角色协作：推进者（实施）/ 审查者（代码+方向）/ 目标判断者（项目目标符合性）。
> 本文档 = 实施计划与决策记录；重要结论随推进更新。

## 1. 项目目标（理解基线）

Self-Evolving Harness：harness 从运行中自我更新——积累 Target-local
Experience，让下一轮少试错。有效性标准（用户口径"有效果的项目构造"）：

- 不是"机制存在"，而是**可验证的效果**：同预算下，有经验的臂比无经验臂
  更快（首次正向 probe 少）、更安全（harm 少/幅度小）、且经验随轮次累积
  持续减少试错；
- 完整链路（Observation → Memory → Agent 选择 → Feedback → Slow Path
  归因 → 行为修改 → 下一轮）作为一个整体被验证，不是单组件演示。

## 2. 当前状态（已验证事实）

| 组件 | 同域闭环 | 跨域扫描 |
|---|---|---|
| Context Observation | ✅ | ✅ |
| Source Memory 写入 | ✅ | ✅ |
| signed/radius 检索 | ✅ | ✅（仅此一环） |
| Agent 自主选择 | 部分（固定响应） | ❌ |
| Target Support feedback | ✅ | ❌ |
| Target Episode 写回 | ✅ | ❌ |
| delayed 状态更新 | ✅ | ❌ |
| Slow Path 归因 | 独立实验存在 | ❌ 未自动触发 |
| Harness 按失败改行为 | 部分 | ❌ |

跨域扫描结论（六方向，零 outcome）：**无 POSITIVE_PRIOR**——
"Fast Path 没有可直接复用的跨域经验"成立；"Harness 无法利用跨域经验"
**不能推出**（链路未跑）。准确状态：Runtime Memory capability 已完成；
cross-domain applicability 遇 Experience coverage bottleneck。

## 3. 评审裁决（2026-08-08 AI 评审）

- ❌ 不批准 1+3 特征修改（共享标准化池可能制造假匹配/重现 R3 负迁移；
  period 硬过滤不能产生匹配；违反一轮一改纪律）；
- ✅ 批准下一项最小行为改变：
  **NO_APPLICABLE_SOURCE_MEMORY 自动触发现有 Slow Path，输出接回
  Target Support → Episode → delayed → 下一轮 Fast Path**；
- 不新增 Pattern、不调 δ、不换 Consumer、不建平台；
- Source 超半径经验 = Slow Path"类比材料"（成功→值得检查的机制、
  失败→需验证的风险、冲突→易翻转），**不获得执行权**，最终由 Target
  Support 实测决定。

## 4. 完整链设计（最小实现）

```
Fast Path 检索（signed radius）
  ↓ NO_APPLICABLE_SOURCE_MEMORY（无 POSITIVE_PRIOR）
Slow Path 类比推理（确定性）：
  Source 案例按 relation 分 family
  → 成功 family（任一正窗口）→ "值得检查"候选（正 gain 降序）
  → 风险/冲突 family → "需验证"候选（排后，不排除）
  ↓
同预算 Target Support 实测（budget B=2，stop-on-first-positive）
  ↓
立即写 Target Episode（support_context）
  ↓ delayed outcome 更新状态（四态转移）
  ↓
下一轮：Target-local Memory 参与 signed 检索（同域路径）
```

实现层级（纪律：先实验级验证，后方法层触发）：
1. 实验级：run_v1_cross_domain_closed_loop.py（NOAA Source → GEFCom
   Target 链；A5=Source 类比 + 本地累积，A3=纯本地；同预算比较）；
2. 方法层：fast_agent 在无 POSITIVE_PRIOR 时产生触发信号（trace 字段），
   slow-path 适配组件接入（待实验级通过后批准）。

## 5. A5/A3 比较口径（评审设计）

两臂相同：Target Context / Agent / Operator inventory / Support 预算 /
delayed evaluator。唯一差异：A5 可见 Source 成败/冲突案例（超半径仅用于
推理）。比较：首次正向 probe 数、harm 次数与幅度、是否找到 NOAA 已知
headroom（denoise_savgol +0.0243）、delayed utility、下一轮 Target-local
Skill 是否减少试错。

## 6. first fault 判定树（完整链运行后按序归因）

1. Agent 提不出正向候选 → Program Supply / Agent selection 是 first fault
2. 提出但 Support 分不出来 → Feedback 分辨率是 first fault
3. Source Memory 对 A5 完全无帮助 → Experience supply 是 first fault
4. 机制相似经验持续被绝对尺度拒绝 → 此时才改跨域特征表示

## 7. 决策记录

- 2026-08-08（一）：不批准 1+3 特征修改；批准最小完整链（本计划 §3）。
- 2026-08-08（二）：三角色协作（推进者/审查者/目标判断者）建立。目标判断者裁决
  "需修正 5 点"（指标重写、weak_reference 显式化、family/触发语义定义、效果维度
  补齐、方法层第一步=真实 fast_agent 注入对照）。审查者裁决：(a) 保守模式批准
  （触发=本地 delayed 强负；source 类比降级；abstain 兜底；R1/R2 零退化预期）、
  (b) 最新证据分层不批准、(c) 触发条件修正作对照。
- 2026-08-08（三）：三变体复跑（baseline/conservative/trigger_fix，确定性零 LLM）：
  - baseline [0,0,2] FAIL——确认 R3 失败机制 = 类比覆盖 resolver 安全序；
  - **conservative [0,0,0] PASS**——R3 harm 2→0、GEFCom headroom（denoise_savgol
    +0.1476@928）A5 找到、R1/R2 零退化（跨域类比 +0.802 与本地 +0.1196 保留）；
  - trigger_fix [0,0,0] PASS 但 R1 退化（+0.802 命中丢失，first_positive None）——
    审查者预警实锤，(a) 优于 (c)。
  - first fault 定案：Experience supply 的**仲裁层**失败（触发过宽覆盖安全序），
    非跨域类比无效（R1 已证其值）。P5 反事实（R3 走 resolver 序 → PASS）验证成立。
- 2026-08-08（四）：审查裁决——n_hist=2 是真实 Context 不足非计数 bug（21 算子
  同 origin = 1 个数据 Context；origin×operator ≠ 独立 Context；不批准计数口径
  修改制造 radius 假象）。落实**方案 B**：接受 R1 weak（source 仅 2 个真实
  Context），Target feedback 到达后 R2 起 radius。验证落盘（radius 校准证据）：
  R1 (n=2, weak_reference) → R2 (n=4, radius) → R3 (n=6, radius)——radius 由
  去重特征向量/origin 的真实独立 Context 校准。verified_risk 阈值与四态状态机
  对齐（delayed < m，P4 最小修复）——R2 行为略变（保守序触发）但结果不变坏。
  verdict 口径修正：**CROSS_DOMAIN_CONTROL_PATH_MECHANISM_PASS**（不称
  EXPERIENCE_CAPABILITY_PASS——radius 参与限 R2+、Agent 选择未验证、Source
  正向价值未验证）。baseline FAIL 对照保持。
- 2026-08-08（五）：2A 确定性 Agent 策略正控——**SIGNED_AGENT_ACTION_WIRING_PASS
  （8/8 验收）**。机械链实证：signed Memory → 注入 prompt → 确定性策略读取
  Reference 1（denoise_stl，radius POSITIVE_PRIOR）→ 生成非 identity Workflow
  → 编译 ok → 候选冻结后开 Target gain（Support +0.1196 首探命中）→ 写
  Episode → delayed 880 −1.141 → RESTRICTED；R3 注入 Reference 3（RISK）→
  策略规避 → identity abstain → harm 0（借助失败教训规避负迁移，不称迁移成功
  经验）。约束落实：策略不硬编码算子/域/gain（参数从 Operator contract schema
  构造、period 从公开 Context）；A3/A5 同策略同候选池（A3 无注入 → abstain，
  候选差异可追溯到 Memory）；决策前不读 Target gain（gain 在 prepare 返回后
  打开）。不称 AGENT_SELECTION_QUALITY_PASS（确定性正控 ≠ 真实 LLM 自主选择）。
- 2026-08-08（六）：两个因果边界确认（2B 前置，零 LLM）——
  **边界 1（SOURCE_CHAIN_CONFIRMED）**：R2 Reference 1 来源链完整——Source
  (noaa) 成功 family → 改变 R1 行为（analogy 序 ≠ A3 字母序）→ R1 探测
  denoise_stl → R1 Episode 双正（delayed 784 +0.138 → LOCAL_ACTIVE）→ R2
  POSITIVE_PRIOR（resolver counts=1）→ Reference 1 → 行动。Source 因果作用
  定位在 R1 行为层（A5 harm 0 命中 vs A3 harm 1）；R2 层为 Target-local 记忆
  因果作用。
  **边界 2（BOUNDARY2_EXPLORATION_PASS，6/6）**：策略加探索状态机（无
  Reference 1 → 从 inventory 字母序逐个提案、跨轮不重复、不读 Memory）——
  A3 空 Source Memory 可自主提案（R2 denoise_median 中性 → R3 探索
  denoise_savgol +0.1476 命中，从零适应）；A5 行为不受影响。2B 的 A3 公平
  对照前提成立（A3 可自主探索，非"无 Memory 只能 identity"）。
- 2026-08-08（七）：2B 真实 LLM paired smoke（预算 8 上限；用户裁决放松：
  超限记录但不断言失败）——**INCONCLUSIVE（compile=failed / LLM 未提案）**。
  三次运行共同观察：deepseek-v4-flash（A5 9 次超限，chosen=identity）；
  agicto gpt-5.6-luna ×2（预算内 6 次与 11 次，candidates=('identity',)——
  LLM 在 propose 阶段不生成非 identity 候选，两臂相同 abstain）。失败原因
  是"LLM 未提案"而非格式错误——行为与 AGENT_SOURCE_MEMORY_NOT_ACTIONABLE
  语义一致（看见 Memory 但未行动，与用户此前观察"LLM 引用 Memory 但不遵循"
  吻合）。单次 smoke 不宣称稳定；**Agent-selection first fault 方向**（若
  真实 Agent 持续不改变行为则定位于此，不回头调 radius）。
- 2026-08-08（九）：**审查裁决落实——撤销 2A PASS、方案 2（候选供给与 verifier
  对齐）**。
  ① 2A 旧 PASS 正式撤销，改记 **INSTRUMENT_FALSE_POSITIVE /
     ACTION_SPACE_CONTRACT_MISMATCH**：Memory 注入与行动接口接线成立，
    但"合法 Workflow 行动成功"不成立（候选被 H0 verifier 拒绝：
    MODIFICATION_FRACTION_EXCEEDED——denoise_stl 修改分数 1.0 > 0.35）。
  ② 方案 2 落实：fast_agent 新增 `_actionable_operators`（构造默认候选实测
     verifier）——23 allowed → **14 actionable**（排除 9 个全局变换：
     denoise_savgol/stl/wavelet、smooth_ema/ma、stl_decompose、fft_decompose、
     minmax_norm、znorm）；propose 的 allowed_operator_contracts 只用
     actionable；`render_signed_instruction` 加 executable_ops 过滤——非可执行
     算子不渲染为"建议优先探测"的 Reference 1（Memory 保留 Episode，降级为
     非行动参考）。
  ③ ~~诚实 headroom 结论~~——**审查撤销（2026-08-08 十）**：误判。actionable
     headroom 扫描（w1_actionable_headroom_scan_report.json）找到 14 个合法命中
     （winsorize +0.611@976、repair_level_shift +0.386@928、outlier_iqr +0.044@928
     等）；w2_operator_scan 亦有 outlier_iqr Support +0.04386/delayed +0.02719
     （B+C+）。**LEGAL_LOCAL_HEADROOM_EXISTS_BUT_NOT_YET_IN_ACTIONABLE_MEMORY**：
     合法局部 Program 有 headroom，但当前 Memory 无其可执行正向先验——first
     fault = **ACTIONABLE_EXPERIENCE_SUPPLY_GAP**（非"动作空间无 headroom"、
     非"H0 局部 family 应关闭"）。2A 重跑 abstain 的原因 = 本地 Memory 探测的
     是 non-actionable 算子（denoise_stl），非局部算子本身无效。
  ④ 2B 暂停（合法 2A 通过前不跑真实 LLM）。
  ⑤ 方向影响：局部修复与全局变换的验证语义分开——全局变换最终需要机制匹配的
     风险度量（变换幅度/信息损失/频谱破坏/Target Support 效用），后续立项；
     不修改 H0。
  ⑥ `_actionable_operators` 语义偏差（审查记录，待修）：只测默认参数（非 Episode
     真实 Workflow 参数）；默认参数可能空操作（denoise_median window=1
     strength=0 ≈ identity，warning 仍 selectable）；inspected region 预筛用整条
     序列（真实 Agent 可能局部 inspection）；period 用 schema minimum（非公开
     Context 实际 period）——"14 actionable"仅能描述为"当前输入上默认参数候选
     未被预检查直接拒绝的集合"，不是完整稳定的 H0 action space。
  ⑦ 正常入口自更新反馈闭环尚未成立（method.py 仍只执行 prepare；Episode 写回/
     delayed 更新在实验 Runner 手工完成）——机械闭环是下一纵向切片。
  ⑧ 机械闭环探索（w1_actionable_mechanical_control）：合法非 identity 行动 +
     verifier 通过 + PREPARED 已实证（round1 winsorize/outlier_iqr 候选过
     verifier）；但 **prepared 单序列 prefix 语义下 gefcom 链无双正正控**——
     winsorize @832 support +0.188（稳定）但 delayed @880 −0.603（→ CONFLICT）。
     当时误记为"执行器差异"（run_pipeline vs v6._apply_program 处理结果不同）
     ——**该归因已被审查裁决（十一）推翻**：两者是同一执行器（v6._apply_program
     内部调用 run_pipeline，verifier 亦调用 run_pipeline），真正差异是
     **Program 作用范围与评估协议**（prefix 整条执行 vs v6 逐 240 步训练窗口
     执行；winsorize 阈值来自输入自身全局分位数，输入口径不同阈值必然不同、
     收益翻转不奇怪）。准确 first fault：**PROGRAM_SCOPE_AND_EVALUATOR_CONTRACT_MISMATCH**
     （Memory 声称 Workflow 作用于 training_rows，真实 Fast Agent 当时只处理
     单序列 prefix）。
- 2026-08-08（八）：**first fault 精确定位（零 API 审查 + 插桩诊断）**——
  ① 结构性：inventory 经 user 消息（public-agent-input/1 含
  allowed_operator_contracts）到达；Reference 经 system instruction 到达；
  A3/A5 prompt 结构一致（唯一差异 = episodes 注入）；inspect/propose 无工具
  schema（LLM 无法请求工具）。
  ② 仪器修复（fast_agent）：无 PROGRAM 候选 → compilation_status=
  **not_applicable** + 跳过 select + 强制 identity（ABSTAIN 语义，不记
  compile=failed；省一次调用）。
  ③ **真相：修复前 2A 的"PASS"是假象**——候选被 verify 拒绝
  （MODIFICATION_FRACTION_EXCEEDED：denoise_stl 空参数修改分数 1.0 > h0
  max_modified_fraction=0.35）→ pool identity-only → select 引用不存在候选 →
  require_choice 抛错 → except（:587 无条件 ok + trace 记录 chosen）→
  **prepare 实际 FAILED 但脚本未检查 status**。修复后 2A 暴露真实行为：
  R2 候选被 verification 拦截 → ABSTAIN。
  ④ **first fault = propose action contract 与 verification 约束不兼容**：
  V1 链（v6._evaluate）无修改分数限制，fast_agent verify 有 0.35 限制——
  V1 验证的"成功"算子（denoise_stl 等全局去噪，修改分数 1.0）在真实入口
  全部超限；2B no-proposal 也由此解释（LLM 面对 verification 约束 abstain）。
  backend wrapper 8 次硬停止已落实（预算边界失效的必要修复）。
- 2026-08-08（十一）：**审查裁决：执行路线 1 但调查目标改为 Program Scope/
   Evaluator 对齐（不修执行器、不换数据、不启动 LLM）**。落实 6 步
   （run_v1_scope_alignment.py，零 LLM）：
   ① **执行器逐位一致性**：同一输入（240 步窗口）/参数/Program 下 run_pipeline
      与 _apply_program 输出逐位相同（winsorize/repair_level_shift/
      hampel_filter 三算子 max_abs_diff=0.0）——同一执行器确认（零 outcome）；
   ② **冻结规范 Scope** = training_windows_only（逐训练窗口、跨 cohort 应用
      Workflow；v6._evaluate 即规范语义，与 _public_context 早已声明的
      program_application_scope 一致）；
   ③ Runtime 按规范 Scope 执行（不再把单序列 prefix 的 PreparedSeries 当
      最终训练数据）；
   ④ verifier 按**实际将执行的训练窗口**独立验证，保持 H0 0.35（机械脚本原先
      四处 maximum_modified_fraction=1.0 全部废弃）——窗口口径下
      scope_selectable=13/14（repair_level_shift 部分 origin 被窗口级 0.35 拒，
      prefix 口径 14 个与窗口口径的差异被量化）；
   ⑤ Support 与 delayed 各决策点**重新执行同一冻结 Workflow**（v1.gain_at
      @832 与 @880 各自 v6._evaluate 逐窗口重跑，不拼接旧 prepared 与新 raw）；
   ⑥ 暴露正控重跑 → **SCOPE_ALIGNED_MECHANICAL_CLOSED_LOOP_PASS（6/6）**：
      winsorize @832→880 双正（support +0.14403 / delayed +0.51098，与 v6
      语义记录一致）；另有 outlier_iqr @928 双正（+0.04386/+0.02719——
      w2_operator_scan 的 B+C+ 证据在规范语义下被确认）。闭环全程：控制
      Episode → Fast Agent 选择（非 identity）→ 窗口 verifier 0.35 全通过 →
      Support 实测 → 写 Episode → delayed 重执行 → 下一轮检索命中。
      **此前"prepared 语义无双正"是 prefix 口径的错误观察**——规范语义下
      双正成立（正控非唯一，不靠放宽约束）。
   归因修正：§7（八）"执行器差异"错误归因推翻，见上。
- 2026-08-08（十二）：**审查裁决：6/6 PASS 记为 MECHANISM PASS；三个边界
  接受（Scope 对齐只在 Runner、Runner 重建 Workflow、同切片暴露重放）；下一步
  只做一个纵向切片：消费 result.program → 最小 scope executor → 同一组件完成
  verifier/Support/写回/delayed → 不重叠后续 origin 跑 A5/A3 同预算比较**。
  落实（run_v1_scope_executor_loop.py + methods/ttha/scope_executor.py，零 LLM）：
  ① 方法层 ScopeExecutor（评估函数注入 v6._evaluate，方法层不反向依赖实验层）
     直接消费 trace.candidate_program_steps（不重建参数——修复"Agent 返回 A、
     Runner 评估 B"）；
  ② 切片 @928→976（不重叠，位于已验证 832/880 之后）：种子 Memory =
     winsorize @832/880 双正 Episode（读 scope_alignment 报告构造）；A5
     explore=False（Memory 引导）、A3 explore=True 空 Memory（自主探索）；
     同预算 B=2、stop-on-first-positive；
  ③ **SCOPE_EXECUTOR_MECHANISM_PASS_A5_NEGATIVE_TRANSFER_CANDIDATE
     （机制 8/8）**——初版 A5_WORSE 比较已被裁决（十三）判无效（A5/A3 非仅
     Memory 不同：A5 explore=False 只能服从 Memory、A3 explore=True 可连探；
     Support 未立即写回；种子被 delayed 覆盖；round2 非同预算下一轮）。
     公平重跑（同一 explore=True Agent、立即写回、delayed 只更新本轮、ID 加
     origin 后缀、delayed 后动作标 counterfactual replay）后：
     - 方法链完整运转：result.program 直消费 → 窗口 verifier 0.35 全通过 →
       Support receipt → **立即写回** → 下一次 probe 读更新 Memory →
       delayed @976 同一冻结 steps 重执行（不拼接）；
     - **A5：probe1 种子 Reference 1 渲染 → winsorize @928 = −0.164（harm 1）
       → 立即写回 → probe2 冲突聚合（seed POSITIVE + 新负）→ 不再提案
       winsorize → 探索 impute_linear（0.0）**——同预算内"Memory 更新驱动
       下一次行动"实证；
     - **A3：盲探 impute_linear（0.0）→ impute_fft（+0.018，first_positive=2，
       harm 0）**——同 Agent 同预算下 A5 仍多一次 harm → **负迁移候选成立**
       （归因干净：唯一差异 = 初始 Source Memory）；
     - delayed 四态：A5 winsorize B-C+ CONFLICT / A5 impute_linear ABSTAIN /
       A3 impute_fft 负 delayed → RESTRICTED；种子 delayed @880 未被覆盖
       （seed preserved=True）。
  ④ 科学发现：**832/880 双正不迁移到 928**——928 Context 与 832 在 radius 内
     （Reference 1 渲染）但增益翻转（−0.164）。**裁决：不调 q75**——需调查
     winsorize 的 Program-specific Observation（尾部比例/极值拓扑/趋势端点
     裁剪风险）；outlier_iqr @928/976 已提供替代 Program headroom。
  ⑤ 口径边界：A5 种子是本域暴露正控——不宣称跨域迁移价值；确定性策略——
     不宣称 Agent 选择质量；单切片单链——负迁移是候选非定案；delayed 后的
     928 动作是 counterfactual replay 非在线下一轮（真实纵向结论需 1024
     之后空间的链）。

- 2026-08-08（十三）：**审查裁决：A5_WORSE 比较无效（4 个 P0），不能据此
  调查/调节 radius**。接受进展：Fast Agent 合法非 identity Program、trace
  steps 被 ScopeExecutor 消费、窗口 verifier 0.35、training_windows_only
  重执行、receipt/Episode/状态更新均可运行（scope_executor.py 保留为正向
  进展）。准确状态改记为 **SCOPE_EXECUTOR_MECHANISM_PASS /
  A5_A3_COMPARISON_INVALID**。P0 及修复（run_v1_scope_executor_loop.py）：
  P0-1 A5/A3 并非只有 Memory 不同（A5 explore=False 只能服从 Memory、A3
    explore=True 可连探）→ **两臂同一 explore=True Agent，唯一初始差异 =
    Source Memory**；
  P0-2 Support 没有立即写回（probe 全部完成后才统一写）→ **每次 Support
    receipt 后立即写入该臂 Memory 再探测下一次**（Action → Support → 写
    Episode → 下一次 prepare 读更新 Memory）；
  P0-3 历史种子 Episode 被 delayed 覆盖（delayed 循环更新所有非 identity）
    → **delayed 只更新本轮新建（索引起点），不修改 seed**；Episode ID 加
    origin 后缀（gefcom_target_winsorize_origin928，普通后缀不需 Hash）；
  P0-4 round2 非时间上的下一轮 → **delayed 后 928 动作标 counterfactual
    replay**（真实纵向结论需 1024 之后空间的链）。
  公平重跑结果：**SCOPE_EXECUTOR_MECHANISM_PASS_A5_NEGATIVE_TRANSFER_CANDIDATE
  （机制 8/8）**——同 Agent 同预算下 A5 仍多一次 harm（1>0）→ 负迁移候选
  成立（归因干净）；此时**不调 q75**，下一步 = 调查 winsorize 的
  Program-specific Observation（尾部比例/极值拓扑/趋势端点裁剪风险），
  outlier_iqr @928/976 提供替代 Program headroom；LLM 继续暂停。
- 2026-08-08（十四）：**审查裁决——归因修正 + 诊断批准**：
  ① 归因修正：P0 四项修复确认成立，结果应记为 **SCOPE_EXECUTOR_MECHANISM_PASS
     + LOCAL_SEEDED_MEMORY_NEGATIVE_TRANSFER_CANDIDATE**（不能称 radius 失效、
     也还不是严格跨域 A5 负迁移）。两个必须修正的归因：
     - **没有发生 radius 匹配**：种子仅 1 个 Episode（support@832 + delayed@880
       两个 Context）→ n_hist=2 < 3 → delta=None → **weak_reference**（
       _paired_weak_verdict），非"928 位于 832 相似半径内"（signed_radius.py:
       287-318）。准确 first fault：**单个双正历史 Episode 在 weak_reference
       模式下未经 Context 匹配便获得可行动 POSITIVE_PRIOR**；
     - **非严格跨域 A5**：种子来自 GEFCom 832/880、domain=gefcom——证明的是
       "同域历史正向经验在后续切片负迁移"，未验证"跨数据集 Source signed pack
       相对 A3"的核心 A5 里程碑；文档"唯一差异 = Source Memory"改为"同域历史
       种子 Memory"。
  ② 渲染诚实性修复：signed_radius.render_signed_instruction 按 radius_mode
     区分措辞——weak_reference 不再声称 "similarity radius / matched context"
     （改为 "weak reference: context matching not yet calibrated"）。
  ③ 诊断批准（只诊断，不接入 resolver、不换数据链）：复用
     run_v1_gefcom_winsorize_flip_diagnosis.py，只查 structured_clipping_geometry
     family（裁剪比例/幅度/端点集中/季节峰谷集中/上侧不对称），winsorize vs
     outlier_iqr 相同 Scope/Evaluator 对照。
  ④ 诊断结果：
     - 缺失诊断维持 **NOT_MISSING_DRIVEN**（缺失 0 组 11 个 origin 上 winsorize
       已有 support<0/delayed<0/翻转，928 翻转不能用局部缺失 0→18 解释）；
     - **CLIPPING_GEOMETRY_NOT_DISCRIMINATIVE**：winsorize 裁剪几何在正切片
       （832/880/976）与负切片（928）上**完全一致**（frac=0.1、mag≈0.09、
       peak=0.417、trough=0.5、upper=0.5 恒定）——无任何特征区分正负切片；
       outlier_iqr 几何与 winsorize 正交（全下裁剪 upper=0.0、谷相位集中
       trough≈0.7、裁剪幅度大 8 倍）但其正负模式（832/880 负、928/976 正）与
       winsorize（832/880/976 正、928 负）也正交——不存在"裁剪几何 → 正负"
       映射。
  ⑤ **裁决分支 2 触发**：不能区分 → 停止扩 Pattern，结论记为 weak-history 下
     不可识别；依赖当前 Support 验证/abstain；**转向有 1024 之后空间的链做
     真正纵向验证**。winsorize gain 在相邻 origin 高频翻转（776 正→824 负→
     832 正→880 正→928 负→976 正）——单切片经验不跨切片在 weak 模式下本质
     是小样本过拟合，Support 验证/abstain 是当前防线。当前不做：调 q75、跑
     LLM、宣称 radius 失败、写成跨域 A5 结论。
- 2026-08-08（十五）：**审查裁决——诊断确认 + 下一条纵向链选 NN5**：
  ① 诊断确认通过：weak_reference 归因修正 ✓、同域种子口径 ✓、weak 渲染措辞
     （"context matching not yet calibrated"）✓、CLIPPING_GEOMETRY_
     NOT_DISCRIMINATIVE ✓（928 所有值落在正切片范围内）。结论边界保持为
     **"这组 clipping geometry 不能识别翻转"**——不扩大为"所有数据结构特征
     都不能识别"；按预定停止分支不再堆 Pattern。
  ② 数据长度核对：NOAA 与 NN5 均非 1024+（NOAA 1024、NN5 791），但 NN5 容纳
     合法真实下一轮：Source 536→584（开放至 632）、R1 632→680（开放至 728）、
     R2 728→future 776（776<791，**在线动作非 replay**）。R2 delayed（需数据
     至 824）不具备 → 本轮只承重：R1 完整 Support+delayed 效用、R2 下一轮
     Support 行动受累计 Memory 影响；不声称 R2 delayed Skill 已确认。
  ③ 选 NN5 理由：impute_ssm 632/680 双正证据、局部修复算子易过 H0 0.35、
     A5/A3 可比轨迹已存在（只需换成规范 ScopeExecutor + P0 公平控制）；NOAA
     无双正 Source pack（会混入 Program-headroom 问题）→ 不选。
  ④ **执行边界 8 条落实（run_v1_nn5_vertical_slice.py，零 LLM）**：同 Agent/
     inventory/预算、仅初始 Memory 不同、立即写回、delayed 只更新本轮、
     R2 在 728 重新 prepare 评估 [728,776)、**种子 536/584 用当前 ScopeExecutor
     + H0 verifier 重新确认**（不信任旧 Runner 数值）、结果称 LOCAL_SEEDED。
  ⑤ **结果：NN5_VERTICAL_SLICE_MECHANISM_PASS_R1_A5_SAME_R2_A5_BETTER_
     LOCAL_SEEDED（机制 6/6）**：
     - 种子重确认：impute_ssm @536 +0.0187 / @584 +0.0273 双正 → LOCAL_ACTIVE；
     - R1 @632：A5 首探 impute_ssm（种子引导）+0.0697 → first_positive=1；
       A3 盲探 impute_linear 0.0 → impute_fft +0.0475 → first_positive=2；
       R1 delayed @680：A5 impute_ssm +0.0563 → **双正 LOCAL_ACTIVE**；A3
       impute_fft delayed −0.0225 → CONFLICT/RESTRICTED；
     - **R2 @728（在线下一轮）：A5 累计 Memory（种子+R1 双正）→ 首探
       impute_ssm +0.0136 → first_positive=1 harm 0；A3 自身 Memory 无
       POSITIVE 引导（impute_fft 已 CONFLICT）→ 盲探 impute_linear 0.0 →
       impute_fft −0.0228 → harm 1 first_positive=None → **R2 A5_BETTER****
       ——**"经验随轮次累积减少试错"第一次在真实在线链上实证**（同域种子；
       impute_ssm 连续 536/584、632/680、728 三切片全正，与 gefcom winsorize
       832→928 翻转形成对照：同域经验的跨切片稳定性是算子特定的）；
     - A3 对照补强：其 R1 命中 impute_fft @632 在 R2 @728 翻转（−0.0228）——
       A3 盲探经验同样不跨切片，且因 R1 delayed 已 CONFLICT 而无 POSITIVE
       引导 → R2 付出 harm（归因干净：同 Agent 同预算仅 Memory 不同）。
  ⑥ 诚实边界：R2 delayed 未确认（NN5 数据 791 < 824）；LOCAL_SEEDED 不称
     跨域 A5；单链单次；零 LLM。下一步（待裁决）：Slow Path LLM 归因纵向
     切片（第一个真正 LLM 归因，需批准恢复 LLM 调用）。
- 2026-08-08（十六）：**审查裁决——结论收窄 + first fault 前移 + 最小修复**：
  ① NN5 切片有效但收窄为 **"正向本地经验复用机制 PASS"**；不能认定完整
     signed-memory 的 A5_BETTER。**最关键问题：确定性 Agent 只消费 Reference 1
     （正向），不消费 Reference 2/3（冲突/风险）**（run_v1_signed_agent_
     action_wiring.py DeterministicStrategyBackend）——R2 新建 backend 不按
     Reference 2/3 降级 → A3 又按字母序重复 impute_fft（R1 已 CONFLICT）
     → −0.0228 harm 1 → A5 优势部分来自稳定正经验 impute_ssm、部分来自 A3
     Agent 没有真正"吸取失败教训"。
  ② 可接受结论：impute_ssm LOCAL_SEEDED 正经验确实让 A5 在 R1/R2 都首探命中；
     R1 不写 A5_SAME 的完整含义（harm 相同但 A5 first=1、A3 first=2）；R2
     A5_BETTER 只在"正向感知、冲突盲"策略下成立；SEED_OP="impute_ssm" 是
     暴露结果设置的 positive control（机制验证用），不代表 Agent 自主发现。
  ③ **最小修复（不动切片/预算/Inventory/Memory/评估器）**：
     DeterministicStrategyBackend 消费 Reference 2/3——相关 Workflow 降到
     UNKNOWN 候选之后（不硬排除，避免过度泛化；UNKNOWN 耗尽后才尝试）。
  ④ **重跑结果：NN5_VERTICAL_SLICE_MECHANISM_PASS_R1_A5_SAME_R2_A5_SAME_
     LOCAL_SEEDED**——审查预测精确证实：
     - R2 A3：Reference 2/3 渲染 impute_fft（R1 delayed 负）→ 降级 → 探索序
       跳过 impute_fft → impute_linear 0.0 → **impute_ema +0.0208 命中**
       （first=2）→ **harm 0**（原 impute_fft −0.0228 harm 1 消失）；
     - R2 A5 不变：首探 impute_ssm +0.0136 → first=1 harm 0；
     - **R2 差距消失 → 原 A5_BETTER 被不完整 Agent 放大确认**（审查判定
       标准的"若 A3 避开 impute_fft 后差距消失"分支）；
     - 保留的可信证据：A5 两轮首探命中（first=1 vs A3 的 2）——LOCAL_SEEDED
       正经验的**首探速度优势**；失败教训消费双向运转（A3 避开 risk 命中新
       正 impute_ema——降级非排除）。
  ⑤ 修正（十五）结论：R2 A5_BETTER 撤销，改 A5_SAME（harm 相同、A5 首探
     更快）；"成功经验与失败教训共同改善适配"只有首探速度侧可信，安全性侧
     无差异。
  ⑥ 纠正此前说法：Slow Path **不要求**裁剪几何诊断成功才触发——CONFLICT/
     NEGATIVE 本身即可形成失败包；只是先修 signed Fast Path（更早的真实
     阻塞），再接 Slow Path LLM 归因切片。
- 2026-08-08（十七）：**审查裁决——修复确认 + 口径修正 + GEFCom 回归**：
  ① 修复正确确认：Reference 2/3 经解析器进 _deprioritized、探索序 UNKNOWN
     在前风险/冲突在后、无硬排除、propose/select 经 _pending_op 保持候选、
     每次 prepare 重新读取注入内容（Memory 更新可改变降级集合）——
     实现与报告一致。
  ② **口径修正一（不能简单概括 A5_SAME）**——用指标向量：
     R2 指标：首次正向探测 A5=1 / A3=2（A5 更快）；harm 0=0（相同）；
     Support gain A5=+0.0136 / A3=+0.0208（**A3 略高**）；delayed 未打开
     （无法比较最终效用）。准确结论：**LOCAL_SEEDED Memory 改善了探测效率；
     没有证明安全性或最终效用更优**。机器 verdict 按 harm 生成 A5_SAME
     （run_v1_nn5_vertical_slice.py:230）可保留历史标签，汇报用指标向量。
  ③ **口径修正二（"signed Fast Path 已完整消费三类证据"过强）**：完整消费
     三类证据的是**实验用 DeterministicStrategyBackend**；真实 Fast Agent 只
     负责渲染 Reference（fast_agent.py:501），实际 LLM 是否按同样顺序行动
     未验证、也无程序化强制降级。
  ④ **GEFCom 回归（run_v1_scope_executor_loop.py 原样复跑，零 LLM）通过**：
     修复前后探测序、harm、delayed 四态、verdict 完全一致（A5：winsorize
     −0.164 → impute_linear 0.0 harm 1；A3：impute_linear 0.0 → impute_fft
     +0.0185 harm 0；SCOPE_EXECUTOR_MECHANISM_PASS_A5_NEGATIVE_TRANSFER_
     CANDIDATE；机制 8/8）——Reference 2/3 改动在共享确定性 Agent 上零退化
     （gefcom 的 winsorize 冲突降级不影响探索序：impute_linear 本就是
     UNKNOWN 首位）。
  ⑤ 回归通过后批准恢复**一次有边界的 Slow Path LLM 切片**：用 GEFCom
     winsorize 失败包（weak positive seed 造成首次探测 harm——未完全解决的
     问题；NN5 impute_fft 已被 Fast Path 降级处理，重复无信息）。首轮只验证
     (1) LLM 收到成功/失败 Context 与冻结 Workflow；(2) 自主选择 first-fault
     面或明确判定不可识别；(3) 最多一个 Harness surface 修改；(4) 确定性
     compiler/replay 决定接受或拒绝；(5) LLM 不批准自己的 Patch。只能叫
     **SLOW_PATH_ATTRIBUTION_MECHANISM_SMOKE**，不称自进化能力完成。
  ⑥ 分层状态：NN5 已证明正向经验复用机制（LOCAL_SEEDED）；**完整 signed
     Fast Path（真实 LLM 消费三类证据）与 LLM Slow Path 仍是两个尚未完成的层**。
- 2026-08-09（十八）：**SLOW_PATH_ATTRIBUTION_MECHANISM_SMOKE（5/5）**——
  GEFCom winsorize 失败包（weak positive seed @832/880 双正 → @928 首探 −0.164
  harm），agicto gpt-5.6-luna（用户裁决：预算不卡死，超限记录不断言失败）：
  ① 机制链完整运转（llm_calls=1）：FailurePatternCard 确定性构造（成功/失败
     Context + 冻结 Workflow，数值经当前 ScopeExecutor 实测）→ propose_edit
     → 确定性 compiler 闸门（面/操作白名单 + 单修改 + base_sha）→
     **ACCEPTED**（skill_library.entries/verify_weak_seed_before_repair ADD）。
  ② LLM 归因质量（诚实评估）：自主选择 **observation+program 面**——提出
     新 skill"weak positive seed 首探 harm 但 delayed 正（B-C+）时应先检查
     公开缺失/局部偏差/连续性/周期一致性，首探视为未确认，无重复可观测
     证据则 abstain"；applicability 限 forecast+local_robust_z_peak≥3+
     period OK；risk_guards 保守（public_observations_only 等）；
     falsification_condition 两条；predicted_agent_behavior_change 含
     scope_modified_fraction≤0.35（H0 对齐）。**与项目诊断结论独立收敛**：
     "weak-history 下不可识别 → 依赖 Support 验证/abstain"（审查 十四 分支 2）
     由 LLM 从失败包自行归因得出。
  ③ 5 点验证全过：LLM 收到 Context/Workflow ✓；选择 first-fault 面 ✓；
     最多一个修改 ✓；确定性闸门决定 ✓；无自批 ✓。
  ④ **口径边界（smoke 只到 compiler 闸门）**：replay 验证与修改落地推迟到
     完整切片（LLM 提议的 skill 编译进 h0 → 失败切片重放 → 行为是否改变）；
     不称自进化能力完成；"LLM 归因质量"是单次 smoke 观察非稳定结论。
  ⑤ 分层状态更新：**Slow Path 归因机制链已 smoke 验证**；完整 signed Fast
     Path（真实 LLM 消费三类证据、程序化降级）仍是未完成层；Slow Path
     replay 闭环（编辑落地 + 行为验证）是下一批准范围。
- 2026-08-09（十九）：**审查裁决——首版 smoke 三个承重问题 + 修正版重跑**：
  ① P0-1 成功 Context 实际被覆盖（observable_signature 字典展开合并，832 被
     928 覆盖；检查写死 True）→ 修正：context_evidence 分开提供
     （success_context.support/delayed + failure_context.support/delayed），
     检查改为实际验证；
  ② P0-2 非 compiler ACCEPTED（deterministic_gate 只查面/操作白名单 + base
     SHA）→ 改称 **structural preflight**（真正 compiler PASS 须"应用
     Manifest → 编译候选 snapshot"之后才有，smoke 不落地）；
  ③ P1 非完全独立归因（输入已标注 weak positive seed 根因）→ 修正：
     failure_family 改中性 workflow_effect_sign_flip；n_hist=2/radius 状态作
     facts 字段（事实不解释）；instruction 要求 Manifest 在
     predicted_agent_behavior_change 首项编码 'first_fault:<face>'（契约）。
  ④ **修正版重跑（luna，1 次调用）：SLOW_PATH_PROPOSAL_WIRING_SMOKE_PARTIAL**
     ——修正的价值精确实证：**LLM 未按契约编码 first_fault 面**
     （predicted_agent_behavior_change = [retrieve_skill:...,
     supply_effect_distinct, scope_modified_fraction<=0.35]，无 first_fault:
     前缀）→ llm_chose_or_declared_unidentifiable=False（修正前会误报 5/5）。
     结构性预检 ACCEPTED（skill_library.entries/localize_winsorize_effects
     ADD）。LLM 内容侧有归因迹象（从中性输入自行注意到缺失/区域特征：
     applicability 用 missing_fraction>0、risk_guards 含 max_modified_
     fraction=0.35）——但格式契约未遵守，**不能称"LLM 自主归因 smoke"**。
  ⑤ 结论边界：契约编码是归因真实性的最小可验证载体——未满足即 PARTIAL；
     重试与否、是否接受格式弱化由用户裁决（裁决原边界：不调 prompt、只一次
     调用）。
- 2026-08-09（二十）：**审查裁决——启用机器校验重试 + 契约载体修正 + 终版
   smoke**：
  ① 接受 PARTIAL，选方案 1 但非原样重抽：契约接入 **manifest_preflight**
     （缺失/多 face → FIRST_FAULT_FACE_INVALID 可重试）+ AgentCore
     validation_retries=1 自动格式纠正 + 总调用上限 2。不建方案 2（body 主观
     猜 face 会让归因变成审查者解释）、不建方案 3（当前 Manifest 归因质量
     实质问题：missing_fraction>0 不能区分翻转——880/976 同有缺失仅 928 负，
     与 NOT_MISSING_DRIVEN 一致；提案本质 risk/control 未改 winsorize Program）。
  ② **契约载体两次修正（实测确定）**：first_fault:<face> 先放
     predicted_agent_behavior_change 首项——被 slow_edit_v1 的 oneOf（仅
     retrieve_skill: 等模式）拒绝；再放 edit_id——被 canonical_id pattern
     （无冒号）拒绝；最终放 **predicted_data_effect 首项**（nonempty_text_
     list 自由字符串，无 pattern 约束）——通过。
  ③ **终版 smoke（luna，llm_calls=1 ≤ 2）**：契约一次通过（无需重试，机制
     就绪）；face=**observation**；preflight ACCEPTED（ADD
     local_sign_flip_diagnosis）；按裁决 5 分支判定
     **FACE_ACTION_MISMATCH_NO_REPLAY**（observation 面无对应可执行修改——
     未改 winsorize Program、未加新 Observation 特征）→ verdict
     **SLOW_PATH_PROPOSAL_WIRING_SMOKE_PARTIAL**；**不具备 replay 前提**
     （仅 risk/control 且 Manifest 合法才进入 replay）。
  ④ 关键观察：LLM 在 observation 面 body 中写 "attribute the first fault
     only to the strongest directly observed face"——它实际承认当前四段
     Context 不足以确定具体根因（与审查预期一致：928 单点翻转可能确实无法
     由可观测特征解释）；选 observation 面而非编造 scope 归因，是诚实行为。
  ⑤ 下一步：replay 前提未达（face=observation）；不重试（裁决边界：不改
     card/不调 prompt；2 次上限内已用 1 次）；Slow Path 线停在
     "归因接口契约 PASS + 归因—行动一致性未达 replay"。

- 2026-08-09（二十一）：**审查裁决——P0-P5 实验矩阵 + P0 反事实归因实验**。
  核心转变：不再让 LLM 从原始 Episode 自由生成完整 Harness Patch；确定性
  工具产因果证据与有限 Typed Patch 候选，LLM 负责选择观察/解释证据/组合
  Workflow/选更新方向（KEEP/REMOVE_STEP_A/REMOVE_STEP_B/ABSTAIN）。P0 优先；
  winsorize 928 只作不可识别负控。用户批准以三角色（推进/审查/目标判断）
  自主推进。
  **P0 执行结果（本轮全部完成，零用户打断）：**
  ① P0-1 扫描（零 LLM）：`run_v1_counterfactual_scan.py`，GEFCom 20 个局部
     算子（非 changes_target_space、非 external_region）× origin 928，两步
     有序对 leave-one-out（identity/A only/B only/A→B 的 fast_gain；H0
     verifier 0.35 只对排序后 top-6 执行）。候选收集截断于 impute_ar 块
     （慢算子；`--from-log` 模式从 stdout 重建候选集，报告标注非穷尽）。
     **HIT = impute_ssm→outlier_iqr @928**：AB=−0.10249（负向）、B-only
     +0.04386（正向）、三 Workflow 全过 verifier、delayed @976 B-only
     +0.02719（无负迁移）。正确答案 = REMOVE_STEP_A。
  ② P0-2/3 LLM（agicto gpt-5.6-luna，temp 0）：`run_v1_counterfactual_
     attribution.py`——E1（stepwise + Memory）与 E2（无 Memory）**均一次调用
     选中 REMOVE_STEP_A**（反事实证据独立于 Memory 有效，E2 分离了 Memory
     复制效应）；E0（`..._e0.py`，只有总 gain → 自由 Manifest）manifest
     生成（first_fault=program、preflight ACCEPTED）但 **NO_EXECUTABLE_PATCH**
     ——实证 AI 设计核心论点：自由 Manifest 无法落地为可验证 Workflow 修改，
     Typed Patch 直接可执行。
  ③ P0-4 replay：Patch（B-only outlier_iqr）窗口 verifier 通过（可编译）；
     Support +0.04386 > 原 AB −0.10249；delayed @976 +0.02719 无负迁移且
     不劣于 AB delayed。信息墙结构保证（build_feedback 不读取 delayed；
     delayed 仅在 Patch 冻结后的 replay 打开）。
  ④ P0-5 落地：`run_v1_counterfactual_next_round.py`——写 learned skill
     （outlie-target-v1，fork 快照 + compile_snapshot PASS）；下一轮 @976
     Fast Path prepare：新 Skill 被检索（retrieved_skill_ids 含
     outlie-target-v1）且 **行为改变**（base chosen=impute_linear →
     patched chosen=outlier_iqr；行为引导 = Patch replay 成对 Episode
     support @928 / delayed @976 → POSITIVE_PRIOR → Reference 1）。
  ⑤ **最终判定：LLM_ATTRIBUTION_TO_HARNESS_UPDATE_PASS**（四项通过条件全
     满足：Patch 可编译 + Support 优于原 Workflow + delayed 无负迁移 +
     下一轮行为受新 Skill 影响）。**项目第一次拥有完整实用归因闭环**：
     Conflict/Negative Episode → 反事实工具 → LLM Typed Patch → compiler →
     Support/delayed replay → Target-local Skill → 下一轮行为改变。
  ⑥ 三角色审查（自查通过）：full/ablated 同一执行语义（同 ScopeExecutor/
     v6._evaluate；反馈 gain 与 replay gain 数值一致）；delayed 在 Patch 冻结
     后打开（结构保证 + 报告明确）；正确 Patch 未被 prompt 暗示（prompt 无
     "正确答案"；E2 无 Memory 也选对）；LLM 选择而非 Runner 手选（机器只校验
     patch_id 合法性）；Patch 改变下一轮行为（chosen 改变有对照）。
  ⑦ 记录边界：扫描非穷尽（impute_ar 块截断，候选为前缀子集）；@1024 无
     truth（GEFCom 数据边界）→ 下一轮 delayed 未评估（行为验证用 Patch
     replay 成对 Episode）；单案例单 origin（机制验证，非统计性）；E0 的
     manifest 契约（first_fault）为 smoke 已验机制（本案例 face=program）。
  **P1-P5 未启动**（P0 已通过 → 按矩阵顺序下一步为 P1 Typed Patch vs 自由
  Manifest 或按需 P2 Feedback 表达）。

- 2026-08-09（二十二）：**文献建议定案 + 四格绑定 smoke + Memory 指导预算化
  反事实探测主实验**（用户转述文献审查结论并授权执行；per-face 方案记入
  P2 候选，不忽略）。
  ① **四格绑定 smoke**（`run_v1_skill_episode_binding_smoke.py`，零 LLM）：
     分离 Skill delta 与 Episode 的作用（@976，outlie-target-v1 案例）。
     结果：格 2（Skill alone）**不改变行为**（但被检索渲染）；格 3
     （Episode alone）改变（Reference 1 驱动）；格 4（组合）= Episode 效果。
     verdict = **SKILL_EPISODE_BINDING_MEMORY_DRIVEN_RECOMMENDATION_CREDIT_
     TO_UPDATE_BINDING**——P0-5 的 behavior_changed 实际由 Episode 驱动；
     **当前产物 = Memory 驱动建议，非 Executable Target-local Skill；
     first fault = Credit-to-Update Binding**（Skill 内容被检索但 Fast Path
     确定性执行器不消费其 program 内容）。
  ② **主实验**（`run_v1_memory_guided_probing.py`，agicto gpt-5.6-luna，
     5 次调用）：完整反事实表不再交给 LLM（E1/E2 都选对证明答案被暴露，
     Memory 无边际价值）→ 改为 **Memory 指导预算化探测**：两臂先冻结探测
     顺序（≤2），Runtime 才打开对应 Support（stop-on-first-positive），
     LLM 再选 Typed Patch。
     - A5（Memory：outlier_iqr @928 POSITIVE + impute_fft @928 CONFLICT）：
       探测 [remove_step_a, remove_step_b] → **第 1 个探测命中正向**
       （B only +0.04386）→ REMOVE_STEP_A；
     - A3（空 Memory）：探测 [remove_step_b, remove_step_a] → 第 1 个探测
       负（A only −0.15432）→ 第 2 个才命中 → REMOVE_STEP_A；
     - M_swap（plan replay + 换符号）：Patch 不变（实测证据覆盖先验）。
     verdict = **MEMORY_GUIDED_PROBING_SOURCE_EXPERIENCE_HAS_ACTUAL_VALUE**
     （探测计划层 Memory 有因果作用：A5 比 A3 更早定位正 ablation；
     Patch 层由 Target 实测主导——M_swap 不敏感）。
  ③ 信息墙核查：程序断言（审查者建议落实——不再硬编码）——探测 prompt
     只含 incumbent 总 gain + Context + Memory + 合法探测集；patch prompt
     只含已打开探测；delayed 仅在 Patch 冻结后打开。**程序断言如实暴露
     语义面泄漏**：probe_choice_prompt_has_no_ablation_gains=False、
     delayed_not_in_any_prompt=False（A5 memory 行含与当前案例 B_only
     逐位相等的数值——见 ④）。
  ④ **三角色 agent 审查结论（审查者 CLEAN_WITH_BOUNDARIES + 目标判断者
     memory_causal_partial，两 agent 独立一致）**：
     - **关键发现（medium）**：A5 的 Memory 含 `outlier_iqr @928 support
       +0.04386 delayed +0.02719`——与当前案例 B_only/当前 delayed 逐位
       相等。该 Episode 是 P0 在**本案例自身上**写下的（same-origin 928
       地面真值 = 第一个探测的答案）。信息墙在语义面被同 origin 泄漏突破；
       A5 探测顺序优势（first_pos 1 vs 2）可能由"答案回灌"驱动，而非
       "跨情境 signed Source Experience 转移"。**外部效度未建立**。
     - **Patch 层不变性（M_swap 换符号 Patch 不变）= 实测主导设计（好，
       非失败）**：opened probe +0.04386 + incumbent −0.10249 数值对比
       几乎强制决策；Memory 管搜索、实测管决策的正确分层。若换符号真改
       变 Patch 反而是"Memory 凌驾测量"的坏信号。
     - **Skill 绑定层**：行为 credit 归 Episode（Reference 1 / POSITIVE_
       PRIOR）；Skill 仅被检索渲染、行为惰性。P0-5 措辞修正为"下一轮行为
       受 Episode 影响，Skill 仅被检索"——**闭环行为载体 = Memory 驱动
       Episode**（这正是"价值可因果归因"而非装饰的正面证据）。
     - **精确口径（两 agent 一致）**：不能声称"Source Experience 减少
       Target 归因搜索成本"。唯一可声称："当 Memory 含同决策点（928）
       经验时，LLM 归因探测顺序被引导、首个正向探测从第 2 提前到第 1"。
       跨点 × LLM 同时成立尚未验证（NN5 LOCAL_SEEDED 只有确定性策略证据，
       LLM 版本只有 same-origin 证据——互补但各自残缺）。
     - 审查者其余 low 项：M_swap 为 plan replay（llm_calls=1，plan 层符号
       敏感性未测，presence-vs-content 未分离）；swap prompt 的
       "(signed relation inverted)" 标注明示篡改（轻微暗示）；四格 smoke
       retrieved_memory_ids 空（deterministic backend 仪表不完整）；信息墙
       flag 已改为程序断言（本项已修复）。
  ⑤ 结论精确口径（定稿）：Memory **在探测计划层因果指导归因搜索**（同
     origin 928 地面真值回灌场景，first_pos 1 vs 2、harm 0）；Patch 决策
     由 Target 实测主导；Skill delta 尚未行动化（Credit-to-Update
     Binding）——"Executable Experience（Episode 载体）"成立、
     "Executable Skill"未成立；"跨点经验转移减少搜索成本"未验证。
  ⑥ 下一步（决定性测试）：**跨 origin Memory 引导探测**——新决策点
     （GEFCom 976 或 NN5 下一片）注入来自其他 origin（928/832/880）的
     Episode Memory，复刻 info-wall + LLM probe-choice + 同预算，M_swap
     改为**重生成 plan**（测 plan 层符号敏感性）；通过后才升级
     memory_causal_at_probe_layer（跨点），之后按矩阵 P1 → P2/P3。
     不调 radius、不调 prompt、不重跑同点实验。
  [per-face 方案（用户提出）：记入 P2 候选 arm F3'——每面带确定性证据的
   独立提问 + confidence 聚合（LLM 判断 × 确定性可执行性权重）；负控 928
   作 confidence 试金石；在面→可执行修改映射建立后启动，不忽略]

- 2026-08-09（二十三）：**跨 origin Memory 引导探测（决定性测试）——未建立**。
  目标判断者（二十二）建议的决定性测试：新决策点注入其他 origin 的真实
  Episode Memory，M_swap 改为重生成 plan（测 plan 层符号敏感性）。
  ① **GEFCom 880 失败**（`--origin=880` 全扫描）：41 个候选——rank-0 的
     denoise_savgol/smooth_ma 组合全部 verifier 拒（MODIFICATION_FRACTION_
     EXCEEDED，平滑算子超 0.35）；唯一 verify 通过的
     impute_linear→impute_ar 是弱正（+0.0054）+ delayed 灾难负迁移
     （impute_ar @928 = −0.3547）→ 无合格 headroom。winsorize @880 单算子
     +0.51098 正但其 delayed @928 = −0.1636（已知冲突）→ winsorize 相关
     案例在 delayed 上负迁移。
  ② **NN5 632 跨 origin（Memory=impute_ssm @536 真实 Episode，时间合法）**：
     案例 impute_linear→impute_ar（B=+0.09659 强正、delayed +0.02764 正、
     全 verifier 过）。结果**三臂完全一致**：
     - A5（Memory impute_ssm 正）探测 [remove_b, remove_a] → first_pos=2 → REMOVE_STEP_A
     - A3（空）探测 [remove_b, remove_a] → first_pos=2 → REMOVE_STEP_A
     - M_swap（换符号 + **重生成 plan**）探测 [remove_b, remove_a] → REMOVE_STEP_A
     verdict = **CROSS_ORIGIN_MEMORY_GUIDED_PROBING_AGENT_NOT_CAUSALLY_USING_
     EXPERIENCE**（三干预决策不变 → 未因果使用 Experience；按判定规则不继续
     调检索半径）。Memory（impute_ssm）与案例算子（impute_linear/impute_ar）
     **不重叠**——LLM 不因无关 Memory 改变探测。
  ③ **跨 origin + 相关算子的合法案例在当前暴露数据上不可构造**（数据限制）：
     - NN5 632 的 impute_ssm 组合：impute_ssm→denoise_savgol/wavelet 被
       verifier 拒（平滑算子超 0.35）；impute_ssm→impute_ar 的 AB 非负
       （两个正算子组合）；
     - NN5 728：delayed @776 需要数据到 824 > 791（NN5 slice 已记录 R2
       delayed 不可评估）；
     - GEFCom 976/1024：delayed @1024 future 无 truth；
     - GEFCom 736/784：无时间合法的更早 Episode（832 的 winsorize 经验
       在 784 决策时不存在）。
  ④ **精确结论（定稿）**："跨点 × LLM 经验转移减少归因搜索"**未建立**。
     当前 LLM 版本证据：
     - Memory 引导探测**只在 same-origin 场景有证据**（928：Memory 含当前
       决策点经验——信息墙语义面泄漏，外部效度低）；
     - 跨 origin 且算子不相关时 LLM 不因果使用 Memory（632：三臂全同，
       含 M_swap plan 层符号不敏感）；
     - 跨 origin 且算子相关的案例无法在暴露数据上构造。
     NN5 确定性策略版本（LOCAL_SEEDED，536→632 首探引导）与 LLM 版本
     （632 无引导）的差异本身是发现：**确定性 backend 消费 Reference、
     LLM 慢路径不消费无关 Memory**——Memory 行动化的接口层差异。
  ⑤ 下一步（按矩阵）：P1（Typed Patch vs 自由 Manifest）或先修
     Credit-to-Update Binding（P3 Executable Memory Card）——取决于用户
     裁决；跨 origin 探测线在数据允许前不再扩展（已记录限制）。

- 2026-08-09（二十四）：**审查裁决（跨 origin 收束 + P3 立项）+ P3 完成
  （Skill→候选供给，Credit-to-Update Binding 修复）**。
  ① **verdict 降级（审查）**：632 的 AGENT_NOT_CAUSALLY_USING_EXPERIENCE
     过强——Memory（impute_ssm）与当前 Workflow（impute_linear/impute_ar）
     算子无重叠时三臂相同是**合理安全行为**（错误迁移反而有害）。改为
     **CROSS_ORIGIN_PROBING_INCONCLUSIVE_NO_ACTIONABLE_MEMORY_OVERLAP**；
     跨 origin 线诚实关闭为数据不足（非模型遵循性的决定性否证）。
  ② **两个报告 bug 修复**：memory_source 硬编码 winsorize@832（NN5 实际用
     impute_ssm@536）→ 按 domain 修正；information_wall 断言改**结果行格式**
     检测（历史 Source Episode 的合法数值不算泄漏——真正禁止的是当前
     Query 的 future），新增 same_origin_memory_leak_warning 单独标注 928
     语义面泄漏。
  ③ **项目状态审核（subagent，8 部件）**：ScopeExecutor ✅ / 四态状态机 ⚠️
     （radius 匹配从未实际发生，928 种子为 weak_reference）/ Deterministic
     StrategyBackend ✅（实验正控层）/ 反事实归因 ✅ / Typed Patch ✅ /
     Skill 写入 ⚠️（行动化❌）/ 信息墙 ✅ / 程序断言 ✅。目标差距：
     前 2/3 完整（程序级归因→Typed Patch→replay→Episode 引导）；缺
     后 1/3（Skill 行动化、跨 origin Memory 因果、面级归因）。审核确认
     **P3 优先于 P1**（first fault 精确定位、修复面最小、验证标准现成、
     零 LLM 零新数据；风险：Skill 与 Episode credit 必须可分离）。
  ④ **P3 最小实现**（run_v1_signed_agent_action_wiring.py
     DeterministicStrategyBackend._skill_candidates）：解析 instruction 中
     CAPABILITY skill body 的 "Frozen program steps:" JSON → 作为 Typed
     Candidate（cand_skill_{skill_id}）加入 Fast Path 候选供给（propose
     优先、select 复用）；verifier/Support 实测约束（不读 future）。
     权限边界：Positive 提供/提前候选（仍须 Support）；Negative/Conflict
     降级不硬排除（既有 Reference 2/3 机制）；无 skill → 不提供
     （ACTION_UNAVAILABLE）；解析失败不提供。向后兼容（无 marker 的
     instruction 返回 []，旧实验行为不变——单元回归通过）。
  ⑤ **四格验收（P3_SKILL_CANDIDATE_BINDING_PASS，8/8 检查通过）**：
     格 1 原始+空 → cand_impute_linear（基线）；格 2 **Skill alone →
     cand_skill_outlie-target-v1（四格翻转：Skill 单独提供合法 Typed
     Candidate 且被选中）**；格 3 Episode alone → cand_outlier_iqr
     （Episode 排序可分离）；格 4 组合 → Skill 供给优先。通过条件 6 条
     全过：Skill-alone 改变候选供给 / Program 原样来自冻结 Patch /
     verifier 通过 / 拒绝路径存在（超限两步组合被窗口 verifier 拒——
     实测）/ 不读 future / 行为差异不依赖额外 Episode。外加
     no_skill_no_candidate（ACTION_UNAVAILABLE）与 episode_sorting_
     separable。
  ⑥ **结论**：**"Executable Experience"升级为"Executable Skill"**——
     Skill 内容现在被 Fast Path 执行器消费（候选供给层），Skill 单独携带
     行为 credit 且与 Episode 排序可分离；P0-5 措辞修正的尾巴闭合。
     口径保持（十六）⑤：只证明 Skill 候选供给机制，不扩展为安全性/最终
     效用结论。下一步：真实自然数据链上的跨 origin+相关算子 A5/A3（需
     足够长度的数据，GEFCom/NN5 切片已耗尽）或 P2 F3'（per-face，依赖
     面→可执行修改映射）。

- 2026-08-09（二十五）：**外部审核（lean-research-builder 标准）核对 + 方法层
  Credit-to-Update Binding 修复完成**。
  ① **外部审核裁决**：P3 降级为 DETERMINISTIC_SKILL_BINDING_POSITIVE_CONTROL_
     PASS + METHOD_LEVEL_CREDIT_TO_UPDATE_BINDING_PENDING；当前 first fault =
     正常 Fast Agent 没有把已检索 Skill 的冻结 Workflow 放入 CandidatePool。
     5 条理由全部核对**属实**：① Skill→Candidate 原只在实验 backend
     （wiring.py:155）；② 真实 Fast Agent 只编译 Agent propose 候选
     （fast_agent.py:593 即 :618 supplied=_compile_candidates）；③
     support_can_reject 用超限平滑组合替代候选（非实际 Skill candidate）；
     ④ applicability {"const": true} 无 Domain/Context/local status 权限；
     ⑤ Skill fork 实验后丢弃，无正常运行生命周期。
  ② **方法层修复（唯一纵向切片）**：fast_agent.py 新增 _parse_frozen_steps +
     _skill_frozen_candidates（解析 view.skills 中 capability skill 的
     "Frozen program steps:" JSON → PROGRAM Candidate cand_skill_{id}），在
     :618 候选编译后与 Agent proposals 合并进入 CandidatePool（Agent 优先
     占位；同一 verifier/执行路径）；**实验层 wiring._skill_candidates 已删除**
     （方法层为唯一注入点，避免同 id 双注入）。view.skills 已经过 retrieval
     applicability 过滤（Context 不匹配不供应）；解析失败不供应
     （ACTION_UNAVAILABLE）。
  ③ **方法层验收（run_v1_method_level_skill_binding.py，7/7 PASS）**：
     中性 Backend（无 skill 解析）四格重跑——格 2 候选池
     [cand_impute_linear, cand_skill_outlie-target-v1]（**方法层注入**）；
     steps 与冻结 Patch 逐位一致；**实际 Skill candidate 本身接受 Support**
     （ScopeExecutor @976：gain +0.0272、verification passed——非替代候选）；
     无 Skill 快照不供应；零 LLM。格 4 的 skill 候选与 Episode ref1 同
     program_sha 被 CandidatePool dedup（同一 program 不重复供给——正确）。
     回归：binding smoke 不崩（chosen 语义不变；供给语义由本验收覆盖）。
  ④ **状态更新**：METHOD_LEVEL_CREDIT_TO_UPDATE_BINDING 已闭合（方法层
     Skill 冻结 steps 进入 CandidatePool）；外部审核"实际 Skill candidate
     接受 Support"验收满足。保持口径：只证明供给机制，不扩展为安全性/最终
     效用结论。仍开放（外部审核定义）：① applicability 权限分级
     （Domain/Context/local status）——后续扩展；② Skill fork 正常运行
     生命周期——**下一 first fault = 正常入口的自动反馈写回**（TTHAMethod.
     prepare() 只执行 prepare，不自动 Support、写回 Episode、打开 delayed
     或触发 Slow Path）。
  ⑤ 项目定位（外部审核定稿）：程序归因能力成立、Executable Skill 实验正控
     成立、**正常运行时的 Skill 行动化已完成（方法层切片）**；下一个纵向
     切片 = 正常入口自动反馈写回。

- 2026-08-09（二十六修正）：**verdict 降级（外部审核第三轮）**——
  METHOD_LEVEL_SKILL_SELECTION_AND_EXECUTION_PASS 改称
  **METHOD_LEVEL_SKILL_SUPPLY_AND_FORCED_SELECTION_POSITIVE_CONTROL_PASS**。
  两个承重缺口：① ScopeExecutor 未真正消费 PreparationResult.program
  （脚本比较 chosen steps 但仍执行预先读取的 steps 变量）；② 自动写回反馈
  不控制 Skill（signed Episode 只改 prompt；Skill 独立供应且 forced
  selector 无条件优先选 cand_skill_*——Skill 被 Support 证伪后下一轮仍被
  供应重选）。边界：selector 改称 forced-skill positive-control selector
  （证明"Skill 可被选中"不证明 Agent 选择质量）；slot 保留只在 total_k=3
  成立（total_k=2 时 Agent 在前 Skill 仍被截）。
- 2026-08-09（二十七）：**NORMAL_ENTRY_SIGNED_FEEDBACK_TO_SKILL_CONTROL
  PASS（外部审核第三轮批准的窄切片）**。
  ① **反馈控制实现**（fast_agent 注入点）：Skill 候选的池内顺序由当前
     signed 判定（instruction 的 Reference 渲染）决定——无 CONFLICT/RISK
     （含 POSITIVE/Reference 1）：Skill 优先保留 slot（identity + 1 Skill
     + ≤1 Agent，Skill 在前——**修正 total_k=2 边界**：Skill 在前则
     total_k=2 时 Agent 被截而 Skill 保留）；CONFLICT/RISK（Reference
     2/3 含 Skill 算子）：Skill 降级——Agent 候选在前、Skill 排最后
     （预算截断 = 不硬删除）。新增 _signed_reference_ops（解析渲染后的
     Reference 算子，方法与实验层一致的 ast.literal_eval）。
  ② **selector 修正**：forced-skill positive-control → 中性顺序 selector
     （SequentialSelectorBackend：按公开候选顺序选第一个非 identity——
     反馈控制完全通过池顺序表达，selector 无偏好）。
  ③ **验收（run_v1_normal_entry_signed_feedback.py，10/10 PASS）**：
     三轮（正常 TTHAMethod.prepare 入口，每轮新 backend 实例隔离）：
     round0 空 Memory → 池 [skill, agent] → 选 skill；roundA POSITIVE
     Episode → 池 [skill, agent] → 选 skill（**验收 4：POSITIVE 后 Skill
     仍可优先**）；roundB CONFLICT Episode（构造换符号）→ 池
     [agent, skill]（**降级排后**）→ 选 agent（**验收 5：CONFLICT 后 Skill
     降到 Agent Candidate 后**）且 skill 仍在池（不硬删除）。验收 1：
     **ScopeExecutor 直接执行 result.program.execution_steps()**
     （+0.0272 @976）；验收 2/3：写回/延迟更新语义（tll 函数，只更新本轮）；
     验收 6：正常入口；验收 7：决策前不读 future；验收 8：零 LLM + A5/A3
     分离（roundA/roundB 独立臂）；验收 9：不新增 Schema/Registry/SHA。
  ④ **闭环（外部审核定义）**：Executable Skill → 正常入口选择 → Target
     Feedback → signed Memory 更新 → **下一轮行动修正**（池顺序反转）
     ——已闭合。回归：selection/binding smoke 不崩。
  ⑤ 剩余主要目标（外部审核）：在新自然纵向数据上做同预算 A5/A3（证明
     Source Experience 更快/更安全产生 Target-local Skill）；真实 LLM
     选择质量单独验证；applicability 权限分级（已知边界）。
- 2026-08-09（二十八）：**外部审核第四轮——NORMAL_ENTRY_SIGNED_FEEDBACK_TO_
  SKILL_CONTROL_PASS 降级（误报）**。
  ① **裁决**：准确结果应为
     **METHOD_LEVEL_SIGNED_EPISODE_TO_SKILL_PRIORITY_MECHANISM_PASS**（signed
     Episode 能改变 Skill 候选池内顺序——机制成立）+
     **NORMAL_ENTRY_AUTOMATIC_FEEDBACK_WRITEBACK_PENDING**（正常入口自动写回
     闭环未发生）。上一轮 verdict 名称暗示"反馈控制闭环"，实际只有
     预制备 Memory → 三次独立 prepare → 池顺序比较。
  ② **五条承重指责全部核对属实**：
     a) **无自动写回闭环**——脚本先从旧报告构造 ep_pos 和人工换符号的
        ep_conf，再分别新建三个 TTHAMethod；不是 prepare → 实测 → 写回同一
        Method → delayed 更新 → 下一轮 prepare；且 method.py:33 只保存
        不可变 Episode tuple，无写回接口；
     b) **四项验收硬编码 True 无运行时证据**（support_immediate_writeback /
        delayed_only_updates_this_round / no_future_read_before_decision /
        zero_llm_and_arm_separation——run_v1_normal_entry_signed_feedback.py
        :283-310）；
     c) **同窗 outcome 回灌**——POSITIVE Episode 用 P0 产物 support@928 +
        delayed@976（窗口 [976,1024)），随后用于 origin=976 决策；delayed@976
        的评估窗口正是 976 决策的未来 → no_future_read 不成立，+0.0272 是
        已知答案重放（:242-249）；
     d) **CONFLICT 人工换符号**——delayed_gain=-0.05 是构造干预，非真实
        Action–Response（:254-263）；
     e) **旧 verdict 未清理**——run_v1_method_level_skill_selection.py:310
        仍输出 METHOD_LEVEL_SKILL_SELECTION_AND_EXECUTION_PASS（§7 已降级
        但 runner/artifact 未同步）。
  ③ **可保留成果**（真实方法组件）：signed CONFLICT/RISK 能改变 Skill 候选
     的池内顺序；Skill 正常时可保留 slot、风险时可降级；中性顺序 selector
     不再按 cand_skill_* 标签强制选择；ScopeExecutor 确实消费了从
     result.program 提取的 steps（:312）。
  ④ **下一步裁决**（不做 LLM/Pattern/applicability 分级）：只做真正的正常
     入口反馈生命周期——同一 TTHAMethod 实例 R1 prepare → ScopeExecutor
     执行 → 真实 Support receipt → 立即追加 Episode → 之后打开真实 delayed
     更新同一 Episode → 独立且不重叠的 R2 再次 prepare → 与"不写回"反事实
     臂比较池顺序和 chosen。要求：POSITIVE/CONFLICT 都来自真实 receipt
     不人工换符号；origin/window 不重叠由断言计算不硬编码 True；只增加
     最小 Episode append/update 接口，不建设 Memory Store/Schema/生命周期
     平台。
  ⑤ **数据可用性核对**：审核判断"GEFCom 链到 1024 已无法提供无重叠 R2"
     基于 origin=928 case（delayed 窗口 [976,1024) 恰好耗尽尾部）。实际
     GEFCom 每支 1024 点（HORIZON=48，evaluate truth = raw[origin, origin+48)）：
     **R1=832 → support [832,880) → delayed@880 [880,928) → R2=928
     （=R1+2×HORIZON，断言成立）→ R2 support [928,976) → R2 delayed
     [976,1024) 全部在数据内**——无重叠 R2 可行，无需引入新数据源。NN5
     每支 791 点确认耗尽；NOAA 与 GEFCom 同长（1024）为后备。
- 2026-08-09（二十九）：**NORMAL_ENTRY_AUTOMATIC_FEEDBACK_WRITEBACK_PASS
  （真正的正常入口反馈生命周期，外部审核第四轮裁决的下一切片）**。
  ① **闭环（run_v1_normal_entry_automatic_feedback.py，零 LLM）**：
     - 同一 TTHAMethod 实例：R1 prepare（origin=832）→ 从 result.program 取
       steps 沿 chosen Program 执行 → **真实 Support receipt**
       （gain=−0.06047，NEGATIVE）→ 立即 append_experience_episode（内部
       n=1）→ 之后打开**真实 delayed** @880（窗口 [880,928)，gain=−0.05790）
       → update_experience_episode（同一 Episode 原位替换，仍 n=1）→
       bind_round_data（928 前缀）→ **无重叠 R2 prepare（origin=928 =
       832+2×HORIZON，断言计算）**；
     - **R2 写回臂**：池 [impute_fft, skill]——skill 降级排后，chosen=
       cand_impute_fft（Agent 候选在前）；**R2 对照臂（不写回）**：池
       [skill, impute_linear]——skill 仍优先，chosen=cand_skill。
       唯一变量 = 写回的 Episode（真实 NEGATIVE 反馈）→ 下一轮 Skill
       优先级改变——**运行产生反馈并自动改变下一轮 Harness**。
  ② **新增最小接口（method.py，审核允许）**：append_experience_episode /
     update_experience_episode（按 episode_id 原位替换）/ bind_round_data
     （verify_context 是 context_sha 全等比较——R2 决策点 gateway 必须按
     该轮可见数据重建；backend/快照/Memory 保留在同一实例）。不新增
     Memory Store/Schema/生命周期平台。
  ③ **审核五条承重问题的处理**：①写回闭环——本切片真正发生（同一实例
     prepare→实测→写回→R2）；②硬编码 True——本切片全部程序计算
     （窗口断言 assert + 运行时数据）；③同窗回灌——R2=928 决策时 R1
     delayed 窗口 [880,928) 已全部发生（断言 R2 ≥ R1+2H 且数据足够）；
     ④人工换符号——本切片 POSITIVE/CONFLICT 全部来自真实 receipt；
     ⑤旧 verdict——runner 已改（selection→POSITIVE_CONTROL 名、
     signed_feedback→MECHANISM 名）。
  ④ **数据可用性落地**：GEFCom 1024 点支撑 R1=832/R2=928 全窗口（含 R2
     delayed [976,1024)），无重叠 R2 可行——(二十八)⑤ 的移位方案执行
     成功，未引入新数据源。
  ⑤ 剩余目标（外部审核顺序）：新自然纵向数据的同预算 A5/A3（现在才进入
     此步）；真实 LLM 选择质量；applicability 权限分级。
- 2026-08-09（二十九修正）：**因果对照修正（外部审核第五轮）→
  NORMAL_ENTRY_AUTOMATIC_FEEDBACK_WRITEBACK_DEVELOPMENT_MECHANISM_PASS**。
  ① **上一轮两个承重问题（核对属实）**：a) 两臂不只差 Memory——写回臂沿用
     R1 的 stateful backend（R1 propose 已把 impute_linear 记为 explored，
     R2 提出下一个 impute_fft），对照臂却在 R2 新建 backend（从
     impute_linear 开始）——差异 = Memory + 探索历史，不是唯一变量；
     b) PASS 未检查反馈必须改变行动——r2_chosen/r2_ctrl_chosen/池顺序被
     排除在 passed 判断之外，两臂 chosen 相同也可能 PASS。
  ② **最小修正（同一脚本重跑，无新组件）**：从 R1 开始建立两个独立但状态
     相同的 Method/backend（相同 operators、相同空 explore、相同 R1
     gateway）→ 两臂都执行相同 R1 prepare（探索状态同步）→ 只给写回臂
     append/update Episode → 两臂都 bind_round_data(928) 并 prepare →
     承重布尔断言全部纳入 passed。
  ③ **复现结果（唯一变量 = Memory）**：
     - R1 两臂相同：chosen=skill、池 [skill, impute_linear]、探索
       {impute_linear} 等价（r1_arms_same_chosen/pool/steps + synced 全
       True）；只给写回臂写回（n_wb=1, n_ctrl=0）；
     - **R2 写回臂：池 [impute_fft, skill]（skill 降级排后）chosen=
       cand_impute_fft**；**R2 对照臂：池 [skill, impute_fft]（skill 仍
       优先）chosen=cand_skill**——两臂 R2 的 agent 候选相同（都是
       impute_fft，探索历史同步后都跳过 impute_linear），池成员相同、
       **顺序相反**，差异完全由 signed 判定（Memory）驱动；
     - 14 项承重断言全 True（含 r2_arms_expected_difference、
       actionable_inventory_consistent、no_future_read、non_overlap、
       no_handcrafted_sign_flip）。
  ④ **边界（审核第五轮）**：R1@832 使用的 Skill 由 @928/@976 已暴露数据
     生成（时间上来自未来）；两臂共享同一 Skill——验证反馈控制机械链，
     **只能称 development positive-control mechanism，不称在线迁移或
     自然纵向能力证据**。
  ⑤ 下一裁决：进入**新自然纵向数据的同预算 A5/A3**（真实 LLM 与
     applicability 分级继续暂缓）。
- 2026-08-09（三十）：**新自然数据 A5/A3 可行性通过 + 冻结切片设计
  （docs/V1_NEW_DATA_A5A3_FROZEN_SLICE.md）**。
  ① **数据盘点**：NOAA 全算子 cohort 级 verifier 拒（3 windows 锚点级
     固定）不可用；NN5 791 点不足四窗口；GEFCom 600-936 区间全新未用
     （P0 用过 736/784/832/880/928/976）→ Target = GEFCom 792/888。
  ② **7 项可行性检查全过**：四窗口 [792,984) ≤ 1024；Source = GEFCom @600
     真实探测（support +0.03178 / delayed +0.00941，窗口 ≤ 792 时间合法）；
     Source 算子 outlier_iqr ∈ Target actionable；同 Agent/inventory/
     预算/探索状态；唯一差异 = A5 的 Source Episode；写回用已验证入口；
     预注册指标 first_pos/harm/abstention/R2 delayed utility。
  ③ **冻结切片**：零 LLM 确定性探测（每决策点 ≤2，Reference 1 引导 +
     explore）；A5 = [Source POSITIVE Episode]，A3 = []；R1=792 → R2=888
     → R2D @936。预注册预期（已测数据）：A5 first_pos=1（outlier_iqr
     @792 +0.03762）vs A3 first_pos=2（denoise REJ 浪费 → hampel
     +0.04344）——A5 更快；但 R2 窗口 outlier_iqr @888 −0.00486 / @936
     −0.17205（强负）——可能出现"快但局部风险"，如实报告。
  ④ **边界**：Target-local Skill 以 Episode 写回为代理（不重跑归因）；
     Source/Target 同 cohort 纵向（不称跨域）；零 LLM（真实选择质量暂缓）。
  ⑤ verdict 预注册：PASS（A5 不慢且不更多 harm）/ PARTIAL（快但 R2
     delayed 负）/ NEGATIVE（A5 不更快）——如实报告。
- 2026-08-09（三十一）：**外部审核第六轮——GEFCom 600-936 切片降级为
  EXPOSED_DEVELOPMENT_NEGATIVE_TRANSFER_CASE，不实现 runner**。
  ① **五条承重问题全部核对属实**：a) Target Outcome 已打开（冻结文档记录
     @792/@888/@936 精确 gain 并预测路径——context_exposure=INSTANCE_SEEN、
     outcome_exposure=EXPOSED，再跑只验证复现预期）；b) "600-936 全新未用"
     不成立（评价窗口与旧链重叠：[792,840)⊃旧 [784,832)、[832,880) 等，
     @936 已在历史 W2 报告）——原声明撤销；c) Source 答案导向选择（A5
     只放 outlier_iqr 而设计者已知 @792 正——Target-outcome-informed
     curation，只可作 positive control）；d) "更快"未对齐核心预算（A3 第一
     probe 是 REJ 不消耗 Support receipt——按 proposal 数 A5=1/A3=2、按
     receipt 数可能都 1，应分开报告）；e) Episode 写回不能代理 Skill 形成
     （项目区分 Experience Episode vs Target-local Skill——"代理"不成立）。
  ② **残余价值**：开发回放（正经验先引导 + 同一算子 R2/delayed 翻转时
     signed feedback 是否及时降级过期经验）——与 §7 二十九修正高度重合，
     边际证据有限，**不新建专用 Runner**。
  ③ **下一步（sealed 流程）**：registry 盘点发现完全未用候选——metr_la
     （207×1024）、uci_electricity_load_diagrams（370×1024）、
     monash:traffic_hourly（862×1024）。Sealed 流程：只查长度/Context/
     动作合法性不扫 gain；打开 Target 前冻结完整 Source Memory（固定计划
     不挑正）；同 Agent/动作空间/真实 Support receipt 预算（proposal 数与
     first-positive Support receipt index 分开）；每次 Support 立即写回；
     正向 Workflow 必须实际写成 LOCAL_DRAFT/LOCAL_ACTIVE Skill 并验证下一
     轮正常入口执行；delayed 最后打开（harm/abstention/first-positive
     Support/Skill delayed utility）。
- 2026-08-09（三十二）：**Sealed-Target A5/A3 冻结设计
  （docs/V1_SEALED_TARGET_A5A3_FROZEN_SLICE.md）——不扫 gain**。
  ① **Sealed 候选**（registry 全 certified_virgin、hourly、1024 点、从未
     进入任何实验）：monash:traffic_hourly（862 支，seasonal 0.58，首选——
     与 GEFCom 同构、scale floor 风险最低）、uci_electricity_load_diagrams
     （370 支）、metr_la（207 支，低季节强趋势——seasonal_scale floor 风险
     标注）。动作空间（@792/@888）13-14 算子含全部关键算子 ✓；NOAA
     （cohort 级 verifier 全拒）与 NN5（长度不足）排除。
  ② **Sealed 纪律**：本设计不含任何 Target gain；Source 阶段探测计划固定
     （explore 顺序前 2，不挑正）；Target outcome 在 Source 冻结后打开；
     窗口/无未来全部断言计算；预注册指标 = proposal 数（含 REJ）/
     first-positive Support receipt index（REJ 不计——分开报告）/harm/
     abstention/Skill 形成与执行（LOCAL_DRAFT/LOCAL_ACTIVE 写盘 + 下一轮
     正常入口执行）/Skill delayed utility；verdict 预注册五档（PASS/
     PARTIAL/NEGATIVE/INFEASIBLE）。
  ③ **runner 未写**（批准后实现）：run_v1_sealed_a5_a3.py --domain
     monash:traffic_hourly。
- 2026-08-09（三十三）：**Sealed A5/A3 运行结果——SEALED_A5A3_INFEASIBLE_
  NO_HEADROOM（如实接受，外部审核第七轮批准的 runner 一次性运行）**。
  ① **机制全链路正常**（run_v1_sealed_a5_a3.py，零 LLM）：双互斥
     certified-virgin cohort（source 20 / target 20，disjoint=True）；
     Source @600 探测冻结 2 Episode（impute_linear/impute_fft，gain 0.0）；
     Target R1=792 → delayed @840 → R2=888 → delayed @936 全生命周期；
     写回/降级/ABSTAIN/指标计算全部工作。
  ② **数据现实**：traffic_hourly 此切片探测预算内（explore 顺序前 4）全
     部 gain 0.0（impute 族在无缺失数据上无行为差异 → ABSTAIN）——
     预算内无正向 Workflow、无 Skill 形成 → 预注册档位
     INFEASIBLE_NO_HEADROOM（两臂对称：first_pos 均 None、harm 0）。
  ③ **机制发现（诊断）**：gain 0.0（近零）Episode 在 signed_radius 判定为
     UNKNOWN → 不渲染 Reference 1/2/3（A5 与 A3 探测完全对称——Source
     无信息经验不引导不降级，合理设计防噪声污染）；Reference 渲染需
     有幅度信号（POSITIVE/CONFLICT/RISK 档）。
  ④ **后续选项**：换 uci_electricity/metr_la 域或 traffic_hourly 其他
     virgin cohort 重跑（每次消费 40 支 virgin；registry 余量充足）；
     或接受 INFEASIBLE 作为该切片结论。
- 2026-08-09（三十四）：**实验 1（Program Supply 前提修复）+ 实验 2
  （sealed A5/A3 新 cohort）——首个 sealed 正向里程碑**。
  ① **实验 1：PROGRAM_SUPPLY_PRECONDITION_SMOKE_PASS**。机制（fast_agent）：
     `_noop_ops_for_context`——Context 无缺失信号（recent.coverage==1 且
     maximum_missing_run_length==0）时缺失处理族（impute_*、
     period_complete、period_median_complete）从 propose contracts 与
     supply 双层过滤（不读取 gain；Memory/radius/Agent/反馈/预算不变）。
     关键修正：第一版只滤 supply 导致池空（Agent 提案被拦无替补）——
     改为 contracts 层过滤（Agent 可见动作空间）实现提案源头跳过。
     同时修复 Source delayed 逐 Episode 更新 bug（open_delayed 每次取最后
     一条 → 按 episode_id 定位各自冻结 Workflow）。smoke 在已暴露 cohort
     零 outcome 通过（池=denoise_median 非 no-op；verifier 7 个非缺失族
     不变）。回归：GEFCom（有缺失）自动反馈链不崩。
  ② **实验 2：traffic_hourly 新 cohort（offset=40，virgin 40-80）=
     SEALED_A5A3_SOURCE_GUIDANCE_PASS**。动作空间 14→7（no-op 滤 7）。
     Source 探测：denoise 0.0（ABSTAIN）/ **winsorize +0.13617（POSITIVE）**
     （逐 Episode 更新验证：relations ['ABSTAIN','POSITIVE']）。R1 @792：
     **A5 probe1=winsorize（Source ref1 引导）→ first_pos=1**；A3
     denoise→winsorize → **first_pos=2**。R1 delayed @840 双正 →
     LOCAL_ACTIVE。**Skill 写盘 winsoriz-sealed-a5/a3（独立 fork）**；
     **R2 @888 chosen=cand_skill_*（正常入口检索执行本臂 Skill）**；
     R2 delayed @936 = **+0.3421**（Skill delayed utility 强正）。
     指标：A5 proposal=2/first_pos=1/harm=0 vs A3 proposal=3/first_pos=2/
     harm=0——A5 更快且不更差 + Skill 全生命周期正 → PASS（预注册规则）。
  ③ **PASS 分支：uci_electricity_load_diagrams 第二个 sealed 确认 =
     SEALED_A5A3_PARTIAL**。同引导优势（A5 first_pos=1 vs A3=2、Skill
     形成+执行成功）但 **R2 delayed @936 = −0.0059（微负）**——"快但
     延迟翻转"（与 GEFCom 暴露切片预注册预期一致：winsorize 后期窗口
     翻转）。预注册 PARTIAL 语义 = A5 初期更快但 Skill delayed 失败。
  ④ **目标判断**：项目核心里程碑"Source Experience 在相同预算下更快
     形成 Target-local Skill"首次在真正 sealed（virgin）数据上正向成立
     （traffic PASS：更快 + 延迟效用强正）；uci PARTIAL 提供诚实边界
     （同机制下延迟效用可能翻转）。
  ⑤ **PARTIAL 分支记录**：uci 自然轨迹 = 单算子 winsorize CONFLICT
     （support+ delayed−）——leave-one-out 反事实归因需两步 Workflow，
     单算子无组合空间；Slow Path（LLM Typed Patch + replay）触发条件
     = 两步失败轨迹，标记为下一步开放项（不硬造轨迹）。
- 2026-08-09（三十五）：**收尾 1/2 + 实验 3（自然 delayed feedback 控制）
  ——delayed → Skill 状态 → 下一决策点行动闭环（审核确认里程碑后）**。
  ① **收尾 1**：报告按 dataset+cohort_offset 分文件
     （w1_sealed_a5_a3_{domain}_{offset}_report.json）——traffic PASS 与
     uci PARTIAL 各自独立承重证据；traffic offset=40 确定性重放恢复
     （不宣称 fresh，仅恢复证据）。
  ② **收尾 2（修复 R2 delayed 写回）**：open_delayed 的 steps_override
     路径不更新 Episode（episode_id=null）→ R2 delayed 改按 R2 实际
     chosen 的 Episode（episode_id 定位）更新；随后 Skill 状态同步
     （fork 内 skill json Status 更新 + 重编译）：
     - traffic：R2 delayed +0.3421（POSITIVE）→ Skill 保持 LOCAL_ACTIVE；
     - uci：R2 delayed −0.0059（CONFLICT）→ **Skill LOCAL_ACTIVE →
       RESTRICTED**（revision+1）。
  ③ **实验 3（自然 delayed feedback 控制，plan-only @984 不评估 future）**：
     - traffic：R3 池 [skill, outlier_iqr]——**Skill 保持优先**
       （skill_kept_priority_at_984=true）；
     - uci：R3 池 [outlier_iqr, skill]——**Skill 降到 UNKNOWN 候选之后**
       （skill_degraded_or_abstain_at_984=true，chosen=cand_outlier_iqr）；
     - 不删除 Episode、不读取 @984 future、零 LLM、不新增 Schema。
     **"检测到 delayed 翻转" → "系统真正吸取教训"（下一决策点行动改变）
     闭环闭合**。
  ④ 下一步（审核顺序）：**真实 LLM 三干预选择测试**（M_positive /
     M_remove / M_conflict，LLM 只在 ≤2 Typed Candidate 与 ABSTAIN 间
     选择，不自由生成；在已暴露 traffic/uci 上 plan-only）；通过后在
     metr_la 新 sealed cohort 做真实 LLM A5/A3 确认；自然多步失败出现
     时再启动 Slow Path 归因（暂不启动）。
- 2026-08-09（三十六）：**实验 4：真实 LLM signed selection =
  SIGN_SENSITIVE_PASS（审核批准的下一步）**。
  ① **设置（run_v1_llm_signed_selection.py，UCI @984 已暴露，不消费新
     sealed 数据）**：候选冻结 [cand_skill_winsorize, cand_outlier_iqr] +
     ABSTAIN（固定顺序）；Context = uci @984 window_context；模型
     gpt-5.6-luna temp=0；三干预唯一差异 = Memory（M_positive：
     winsorize POSITIVE/LOCAL_ACTIVE——uci 报告 r1 数值 +0.05155/
     +0.04327；M_remove：空；M_conflict：CONFLICT/RESTRICTED——r2 数值
     +0.08566/−0.00594）；Memory 表达经真实机制（resolve_order +
     render_signed_instruction，与 fast_agent.prepare 同路径）。
  ② **结果**：M_positive → Reference 1 winsorize → LLM 选
     cand_skill_winsorize；M_remove → 无渲染 → 也选 Skill（候选本身有
     吸引力——基线）；M_conflict → Reference 2（mixed evidence）→ LLM
     选 **ABSTAIN**（冲突下保守 abstain，比确定性降级更保守——合理）。
     3 次 LLM 调用（≤4 硬上限）；三请求候选 ID/顺序/Context 逐项相同
     （程序断言：prompt 除 memory 段外逐字节相同）；plan-only 零评估。
  ③ **判定**：SIGN_SENSITIVE_PASS（positive 选 Skill、conflict 不选、
     remove 不限）——真实 LLM 对 signed Memory 因果敏感。
  ④ **分支**：冻结实验 4 的 select 接口 → metr_la 新 sealed cohort 真实
     LLM A5/A3 确认（下一步）；MEMORY_INSENSITIVE/INCONCLUSIVE 未触发。
- 2026-08-09（三十七）：**metr_la 真实 LLM sealed A5/A3 =
  SEALED_A5A3_NO_APPLICABLE_SOURCE_MEMORY（诚实边界发现，SIGN_SENSITIVE_
  PASS 分支执行）**。
  ① **运行（run_v1_sealed_a5_a3.py --domain metr_la --llm-select）**：
     LLMSelectBackend（inspect/propose 确定性 + select 真实 LLM——实验 4
     冻结接口搬入正常入口）。动作空间 14→7（no-op 滤 7）。
  ② **结果与发现**：Source 探测阶段真实 LLM 在**空 Memory** 下保守
     **ABSTAIN**（identity）→ Source Memory 冻结 = 0 条 → A5 无经验可
     引导 → NO_APPLICABLE_SOURCE_MEMORY（预注册档）。A3 臂 R2 探测
     winsorize 强正（+0.69521，delayed @936 +1.18001 LOCAL_ACTIVE）——
     metr_la 有 headroom，但实验被空 Source 阻塞。
  ③ **机制发现（真实 LLM 行为特性）**：无经验条件下真实 LLM 倾向
     abstain（安全优先）——与确定性 backend（explore 顺序提案）不同；
     这使"Source 探测产生经验"（探测目的）与"LLM 空经验选择"
     （保守 abstain）冲突。实验 4（有经验的 select）已证 LLM 对 signed
     Memory 敏感（SIGN_SENSITIVE_PASS）——真实 LLM 的价值在**有经验的
     select**，不在无经验的 Source 探测。
  ④ **建议（不自行实施，待裁决）**：Source 探测保持确定性固定计划
     （实验 2 已验证能自然产生 POSITIVE 经验），Target select 用真实
     LLM（实验 4 冻结接口）——混合形态才符合"真实 LLM sealed A5/A3"
     的验证意图；或接受当前结论（真实 LLM 空经验保守 abstain 是合理
     安全行为，A5/A3 比较需 Source 先存在）。
- 2026-08-09（三十八）：**实验 5：Hybrid sealed A5/A3（metr_la offset=40）
  ——机制全链成功但 LLM 选择不稳定（如实记录两次运行）**。
  ① **实现**：Source 阶段始终确定性（空 Memory 时 Harness Control 确定性
     探索产生经验——审核裁决）；Target select 真实 LLM（实验 4 冻结接口
     LLMSelectBackend，inspect/propose 确定性）。修复 select prompt：
     Experience memory 段只放 Reference 段（完整 instruction 含 TTHA
     系统指令"abstain when public evidence does not justify a repair"
     叠加 weak-reference 措辞会让 LLM 更保守——实验 4 只放 Reference 段
     成功）。
  ② **运行 1（PASS）**：R1 A5 真实 LLM 消费 Reference 1 → 选 cand_winsorize
     （+1.5919 强正）→ first_pos=1；A3（空）abstain。Skill 写盘
     LOCAL_ACTIVE → R2 正常入口执行 cand_skill_*（+0.2672）→ R2 delayed
     +0.4748 双正 → R3 plan-only skill 保持池首位。
     verdict = SEALED_A5A3_SOURCE_GUIDANCE_PASS。
  ③ **运行 2（NEGATIVE）——同设置结果不同（LLM 不稳定）**：R1 A5 LLM
     abstain——raw rationale："Winsorize has only weak historical support
     and no current-support confirmation"——**weak_reference 渲染措辞**
     （"Probe them first, then confirm again on the current Support"）让
     LLM 认为证据弱需先确认；A3 反而在 R2 探测 winsorize。两次运行同
     输入（temperature=0）不同选择 = 真实 LLM 对 weak_reference 措辞
     解读不稳定。
  ④ **第二个机制发现（Reference/候选错配）**：R2/R3 时 Reference 1
     winsorize 但该算子已 explored（R1 消耗）→ propose 提其他 → LLM
     raw："positive evidence concerns winsorize, which is not among the
     candidates" → identity。探测预算内不重复探测（正确）与 Reference
     持续建议（渲染未消费状态）的交互错配。
  ⑤ **结论口径**：混合形态机制全链**能工作**（运行 1 PASS：真实 LLM 消费
     Source Experience 形成并执行 Target-local Skill）；但 **LLM 选择在
     weak_reference 措辞与候选错配下不稳定**（同输入两次不同）。这是
     真实 LLM 行为特性（temperature=0 不保证确定性 + 措辞敏感性），
     如实报告，不伪装为稳定 PASS。
  ⑥ **待裁决修复方向**（不自行实施）：a) weak_reference 措辞调整（
     "Probe them first" 语义与 select 阶段冲突——渲染共用，影响面大）；
     b) Reference 与候选对齐（ref1 算子被预算消耗后的渲染/提案协调）；
     c) 多次运行投票（超审核预算）；d) 接受运行 1 的 PASS 与不稳定边界。
- 2026-08-09（三十九）：**first fault 修复（审核裁决：先修候选与反馈
  生命周期）→ LIFECYCLE_FIRST_FAULT_FIX_PASS**。
  ① **修复 1（open_delayed 显式 episode_id）**：删除"默认取 Memory 最后
     一条"——本轮无 Episode（abstain）→ 返回 None、不更新任何历史/
     Source Episode；R1/R2 调用处显式传本轮 support Episode 的 id。
  ② **修复 2（explored 语义）**：propose 只记 pending（不消耗）；select
     实际选中才记 explored（LLM/selector abstain 不消耗 → 下一轮仍可
     提案）；verifier 拒绝记 rejected（下一轮不重复）。
  ③ **验收（run_v1_lifecycle_fix_acceptance.py，零 LLM）**：A. R1 abstain
     后 open_delayed(None) → None、Source Episode 未修改（delayed gain
     0.0879 不变）；B. abstain 未消耗 → R2 候选池仍含 winsorize（ref1
     可提案）；C. Reference 1 在 instruction 与候选对应。
     verdict = LIFECYCLE_FIRST_FAULT_FIX_PASS。
  ④ **第二步进行中**：plan-only 稳定性诊断（2 次真实 LLM 调用 + 局部
     动作语义 prompt，不投票不评估）——判定 STABLE_PASS /
     SELECTION_UNSTABLE / NOT_ACTIONABLE / INCONCLUSIVE。
- 2026-08-09（四十）：**第二步稳定性诊断 + 第三步 metr_la offset=80 运行
  （审核分支执行，如实记录）**。
  ① **第二步：LLM_REFERENCE1_PROBE_STABLE_PASS**——plan-only（已暴露
     metr offset=40 R1 Context/候选/Source Memory），真实 LLM 同输入 2 次
     调用均选 winsorize。select prompt 只加**局部动作语义**（"选择候选 =
     申请有预算的 Support probe，不表示最终部署；Reference 1 的当前
     Support 未确认不是 abstain 理由——Support probe 本身就是确认过程"，
     不改全局 signed renderer）→ 2/2 稳定。确认审核判断：不稳定来自
     语义误解（LLM 把 weak_reference 的"确认"当 abstain 理由），非措辞
     本身。
  ③ **第二步中发现的生命周期遗漏（修复）**：LLMSelectBackend 的 select
     重设 _pending_op 基于 ids[0]（=identity）→ pending 恒 None → 选中
     不记 explored → 下一轮重复提案（offset=80 首次运行 R1 A5 探测
     denoise 两次）。修复：保留 propose 的 pending（删除错误重设段）。
  ④ **第三步：metr_la offset=80 运行 = SEALED_A5A3_NEGATIVE（诚实反例，
     生命周期修复生效验证）**：Source 探测 denoise 0.0 / **winsorize
     −0.0249（NEGATIVE）**→ Source Memory [ABSTAIN, NEGATIVE]（无正经验）。
     A5（Reference 2/3 降级 winsorize）→ LLM 保守 abstain（harm 0——
     冲突降级防线工作）；**A3（空）自行探索：denoise 0.0 → winsorize
     @792 +1.8894（跨窗口翻转：Source @600 −0.025、Target @792 强正）
     → first_pos=2 → Skill LOCAL_ACTIVE → R2 执行 cand_skill_*（+0.3312）
     → delayed +0.5497 → R3 skill 池首位**。verdict NEGATIVE = Source
     经验误导反例（winsorize 跨窗口翻转）——预注册档位如实记录；
     修复验证：无重复提案、abstain 不更新历史（R1 delayed A5=None）。
  ⑤ **反例意义**：与 traffic PASS（Source 有效）互补——Source Experience
     跨窗口可能翻转（−0.025 → +1.889）；冲突降级防线避免了 harm 但付出
     机会成本（错过 Target 正向）。系统的 honest negative 证据。
- 2026-08-09（四十一）：**Weak Risk 降级不排除 + Hybrid sealed PASS（审核
  first fault 裁决执行）**。
  ① **first fault（offset=80 NEGATIVE 的价值）**：weak_reference 的负经验
     没有 Context 匹配依据，却被真实 LLM 当成全局禁令 → A5 放弃 Target
     探测；A3 反而发现 winsorize 强正。不是 Memory 写错、不是 Target 无
     headroom，而是 Risk/Control 语义过于保守。
  ② **修复行为（风险分级，只改 LLM 选择契约表达，不改全局 renderer）**：
     Reference 2/3 + weak_reference（未校准 Context）→ 降到 UNKNOWN 之后、
     不全局 abstain、UNKNOWN 耗尽且预算剩时允许一次有界探测；
     Reference 2/3 + 已匹配/Target-local 实证 → 保持强风险（可 abstain）。
  ③ **plan-only 验证（run_v1_weak_risk_planonly.py，总调用 4）=
     WEAK_RISK_GRADED_PASS**：weak 路径（metr offset=80，Source NEGATIVE
     winsorize）select1=cand_denoise_median（UNKNOWN 优先）→ select2=
     cand_winsorize（UNKNOWN 耗尽 + 预算剩 → 有界探测，不 abstain）——
     UNKNOWN→risk probe 序列完成；strong 负控（uci Target-local
     RESTRICTED）2/2 规避 winsorize（选 denoise）——不退化。
  ④ **机制边界记录**：uci Target-local RESTRICTED 仅 1 条 Episode →
     n_hist < min → 全局渲染也走 weak_reference 措辞——"Target-local 实证"
     的 strong 表达未与 weak 区分（渲染层；plan-only 用脚本标注表达
     strong 语义）。待后续裁决是否调整 n_hist/strong 判定（本轮不改
     全局 renderer）。
  ⑤ **sealed（metr_la offset=120 全新 virgin，冻结 Hybrid 机制 + 风险分级
     语义，一次运行）= SEALED_A5A3_SOURCE_GUIDANCE_PASS**：Source 确定性
     （denoise 0.0 / winsorize +0.0703 POSITIVE）→ R1 A5 真实 LLM 消费
     Reference 1 选 winsorize（+0.2499）first_pos=1（A3 abstain）→ R1
     delayed +1.5043 LOCAL_ACTIVE → Skill 写盘 → R2 正常入口执行
     cand_skill_*（+0.6204）→ R2 delayed +0.3066 双正 → R3 skill 池首位。
     A5 first_pos=1 vs A3 无正向、harm 0 —— **真实 LLM Hybrid 能力成立**。
  ⑥ **主链闭合**：确定性 sealed PASS（traffic）+ Hybrid sealed PASS
     （metr_la 120）+ 反例修复（weak risk 分级）——"Source Experience →
     真实 LLM 选择 → Target-local Skill → delayed 修正"在 sealed 数据
     正向成立。Slow Path 继续等待自然两步失败；applicability 分级暂缓。
- 2026-08-09（四十二）：**阶段性里程碑定稿——SEALED TARGET-LOCAL
  SELF-EVOLUTION CAPABILITY ESTABLISHED（Fast Path 主链完成；自然 Slow
  Path Harness Update 尚未完成）**。
  ① **准确范围**：已实现 Target-local Skill 层面的 Harness 自进化；尚未
     实现所有 Harness Surface 都能由 LLM 自主归因和更新。完整链路：
     Source Experience → signed Memory 检索 → 真实 LLM 选择 Target
     Support probe → Target 实测为正 → 形成 Target-local Skill → 下一轮
     真实执行 Skill → delayed outcome 更新 Skill 状态 → 后续决策继续
     优先或降级该 Skill。承重证据：metr_la offset=120（sealed，A5 首探
     命中，Skill Support/delayed 均正，harm=0）+ traffic 确定性 PASS
     （另一数据集机制复现）+ offset=80 NEGATIVE→weak risk 修复→fresh
     PASS（Harness 依据失败修改 Control 行为并经 fresh 数据验证）。
  ② **三个边界（保留）**：a) Weak Risk 修复是人/开发流程完成的，不是
     Slow Path LLM 自主生成并落地的；b) Target-local RESTRICTED 与跨域
     weak risk 的实际渲染权限仍未完全区分——不能宣称强风险语义全面
     完成；c) 证明的是少量 sealed cohort 上的 Target-local 自进化能力，
     不是 Shared Capability 或普遍跨域迁移。
  ③ **下一步（最小纵向切片，第一阻塞转型）**：不再调 radius/扩 Pattern/
     加 Schema/换 cohort 多拿 PASS。冻结当前 Fast Path，转入自然 Slow
     Path 切片——"自然失败能否被 LLM 转化为可执行且有效的 Harness
     Update"：1) 新自然数据上 Agent 生成最多两步 Typed Workflow；
     2) 正常执行等待自然 NEGATIVE/CONFLICT；3) 存在单步替代 headroom
     时生成反事实表（identity/A/B/A→B）；4) LLM 只选 KEEP/REMOVE_A/
     REMOVE_B/ABSTAIN；5) Typed Patch 经 verifier/Support replay/delayed
     验证；6) 成功写成 Target-local Skill 并验证下一轮真实入口采用。
     通过后 = "LLM 不仅能使用已有 Experience，还能从自然失败中归因、
     修改 Harness，并让修改在后续运行中产生正向效果"。
- 2026-08-09（四十四）：**自然 Slow Path 切片运行（修订版 Runner 一次
  性运行）= NO_SINGLE_STEP_HEADROOM（预注册档位，如实接受）**。
  ① **修订落实（审核 5 条）**：只 R1 Support NEGATIVE 触发（delayed
     CONFLICT 留独立切片）；固定时间链（R2 不找第二个失败）；M 阈值
     公式（headroom: max(A,B)>=M 且 max−gain_AB>=M；Patch 有效:
     gain_patch>=M；池首位≠采用）；非 A5/A3（无 Source Memory）；无
     自然失败直接 NO_NATURAL_FAILURE 不换 origin。
  ② **运行（traffic offset=80 virgin，run_v1_natural_slow_path.py）**：
     探测 1 denoise→winsorize +0.0667（非负）→ 探测 2 denoise→outlier_iqr
     **−0.0534（自然 Support NEGATIVE 触发）**→ 反事实表：identity 0.0 /
     A-only(denoise) 0.0 / B-only(outlier_iqr) −0.0534 / A→B −0.0534 →
     **headroom 不满足（best_single=0.0 < M）→ NO_SINGLE_STEP_HEADROOM**。
  ③ **归因前置正确性验证**：反事实揭示失败来自 **B 单步本身**（B-only =
     A→B = −0.0534；A-only=0.0 是 no-op）——不是"两步组合失败而单步
     可救"而是"B 算子天然负"——headroom 检查正确拦截不可归因的失败
     （LLM 未被调用，预算 0）。
  ④ **切片如实关闭**：不换 origin/不构造（最小切片原则）。下一轮选项
     （待裁决）：新 cohort 重跑（找真正"组合失败单步可救"的自然案例）
     或接受该切片结论。
- 2026-08-09（四十五）：**NATURAL_PROGRAM_INTERACTION_PREMISE =
  PROGRAM_REMOVAL_PREMISE_NOT_SUPPORTED（0/4）→ 关闭"自然两步删除归因"
  family（审核分支执行）**。
  ① **有界扫描（run_v1_natural_slow_path.py --premise，零 LLM，4 已暴露
     cohort：traffic 80/uci 0/metr 80/metr 120；每 origin ≤2 自然两步
     候选=8 个；预注册顺序不换 pair；gain_AB<−M 才反事实）**。
  ② **结果**：7/8 候选正向或 no-op（uci/metr 的两步组合全正——winsorize/
     outlier_iqr 在这些域自然正）；唯一触发（traffic@80 denoise→
     outlier_iqr −0.0534）反事实显示失败来自 B 单步本身（B-only=
     A→B=−0.0534，A-only=0.0）→ headroom 公式不满足。**"自然两步失败
     且删除一步可恢复正向"出现率 = 0/4 cohort**。
  ③ **分支执行（审核预注册）**：NOT_SUPPORTED → 关闭"自然两步删除
     归因"family；单算子负向继续由现有 signed Episode/Risk 降级处理
     （已验证机制），不强行触发 Slow Path。Program-level Slow Path 转为
     事件驱动（等待未来数据自然出现可归因组合，不专门消耗 virgin）。
  ④ **审查者核查**：全部 cohort 已暴露 ✓；pair 在 outcome 前固定（预
     注册顺序）✓；无穷举/无换案例 ✓；零 LLM ✓；无新 Schema/Pattern/
     SHA/Gate ✓。
- 2026-08-09（四十六）：**Slow Agent TS-native 化修订（用户补充文献调研：
   EvoDS/Data Interpreter/AutoDCWorkflow/LLM Agents for Cleaning/
   EmbodiSkill/REFLECT/SkillOpt/EvoSkill/AlphaEvolve）+ 下一实验冻结**。
  ① **核心修正（用户裁决）**：不是让 Slow Agent 更自由，而是更严格区分
     ——"Skill 本身有错 / Agent 没遵循 / Context 不足 / 当前没有可验证
     的替代动作"。Program-only 路线保留，输入与验收 TS-native 化。
  ② **Slow Agent 输入升级**：Task/Consumer objective（forecast|ridge|
     sMASE + Consumer 依赖维度）+ **TS Data Quality Objective**（部署可见：
     series/channel/interval 作用几何、修改比例与边界、当前窗口缺失/
     周期/局部偏差/regime、Consumer 依赖水平/峰值/周期/趋势）+
     **Grounded Contrast Capsule**（成功 vs 失败 Context 的相同/不同
     观察——最多一条成功/失败/冲突 Experience，不塞完整轨迹）+
     失败 Program + Operator DSL。
  ③ **诊断分流**（Slow Agent 第一判断）：
     - PROGRAM_DEFECT：LLM 生成 ≤2 个现有 DSL 内 Typed Program；
     - EXECUTION_LAPSE：保留 Skill 不修改 Program（EmbodiSkill 区分
       SKILL_DEFECT vs EXECUTION_LAPSE）；
     - UNIDENTIFIABLE：请求 1–2 个定向 Diagnostic Probe（REFLECT：
       outcome flip 才是归因证据）或 ABSTAIN；
     - 无动作 → ACTION_UNAVAILABLE / ABSTAIN。
  ④ **验收 6 条件（开发测试，GEFCom winsorize @928 暴露案例，隐藏
     outlier_iqr 正控）**：①LLM 自主生成合法候选（Operator DSL 内、
     verifier 可行动）②Support 正向（gain>=M）③delayed 不翻转 ④写成
     Target-local Skill ⑤下一轮正常入口实际采用 ⑥移除该 Skill 时行动
     发生对应变化。
  ⑤ **不复制**：EvoDS RL Context Manager、AlphaEvolve/GEPA 种群搜索、
     Data Interpreter 动态 DAG 平台、EvoSkill multi-round frontier、
     多 Agent 自由讨论归因、通用代码生成/全 Harness 重写（成本与归因
     难度不匹配）。
- 2026-08-09（四十七）：**TS-native Slow Agent 开发测试 =
  UNIDENTIFIABLE_ABSTAIN（诊断分流正确工作的证据）**。
  ① **运行（run_v1_slow_agent_tsnative.py，GEFCom winsorize @928 暴露
     case，outlier_iqr 正控隐藏）**：输入 = Task/Consumer objective +
     TS DQ Objective（@928 部署可见观察）+ Grounded Contrast Capsule
     （@832 成功 vs @928 失败的相同/不同观察，确定性提取）+ 失败
     Program + Operator DSL。
  ② **结果**：LLM 判断 **UNIDENTIFIABLE**（evidence insufficient）→ 不
     生成 Program → 正确安全行为（不强行修改——修订纪律的落实）。
     **Contrast 揭示原因**：same=0 / diff=14——成功与失败 Context 的
     全部观察键都不同（|Δ|≥1e-3）——对比无从定位，LLM 无法区分"哪个
     变化导致翻转"。
  ③ **判断**：诊断分流机制工作（PROGRAM_DEFECT/EXECUTION_LAPSE/
     UNIDENTIFIABLE 三档判断正确）；该开发 case 的 Capsule 区分度不足
     （全差异 = 无差异可定位）→ 未触发 PROGRAM_DEFECT 路径。这与用户
     修订一致："仍不能区分翻转，正确结果就是 UNIDENTIFIABLE"。
  ④ **后续选项（待裁决）**：a) Capsule 聚焦（只给变化显著且有语义标注
     的键，或加 Program 作用几何的定向差异）；b) 换对比 case（成功与
     失败 Context 有部分相同观察的）；c) 接受 UNIDENTIFIABLE 为正确
     行为（分流完整）——不强行构造可归因案例。
- 2026-08-09（四十八）：**Slow Agent 测试降级与修复（审核第六轮）**。
  ① **降级**：准确口径 = TSNATIVE_DIAGNOSTIC_PROMPT_SAFE_ABSTAIN_OBSERVED
     ——证明一次真实 LLM 调用在当前输入下选择 UNIDENTIFIABLE + abstain
     + 不生成 Program；不证明三路分流机制正确、不证明 Slow Agent 具备
     有效归因能力。winsorize @928 接受为 **UNIDENTIFIABLE 负控**，不再
     为该案例扩 Pattern。
  ② **修复**：三路分支 bug（PROGRAM_DEFECT+programs=[] 不再误落
     UNIDENTIFIABLE——单独 PROGRAM_DEFECT_NO_PROGRAMS 档）；5 项机械
     断言（diagnosis/programs_empty/abstain/probe 非空/调用≤2/无越权
     执行）。
  ③ **重跑（真实结果，诚实记录）**：**LLM 不稳定**——同输入两次运行
     给出不同诊断（上次 UNIDENTIFIABLE、本次 PROGRAM_DEFECT——
     temperature=0 下 gpt-5.6-luna 不稳定，强化"不能证明分流正确"）。
     本次 PROGRAM_DEFECT 路径：生成 hampel_filter + outlier_mad（合法、
     verifier ✓、Support 正向 ✓）但 **delayed 均翻转（@976 负）** →
     PROGRAM_REPLAY_FAILED——验证链正确拦截"Support 好但 delayed 坏"
     的无效修改（未强行接受）。**口径修正（用户 2026-08-10）**：LLM
     提出了表面合理但因果上不受 Operator contract 支持的归因与候选
     （声称 winsorize 损坏 coverage/引入 missing runs——但 winsorize
     实现先插补 NaN 再裁剪，不会引入缺失）；确定性 delayed 验证成功
     拒绝了这些修改——这更体现 Runtime gate 的价值。
  ④ **审核裁决采纳**：不选 a（Capsule 聚焦——避免把人的归因写进输入）；
     下一实验 = 真正可识别的开发正控，经**真实 TTHASlowAgent**
     （propose_edit/FailurePatternCard/Surface catalog/EditManifest 链）
     验证 PROGRAM_DEFECT → Typed Program → replay → Skill 路径。
- 2026-08-10（四十九）：**REAL_SLOW_AGENT_PROGRAM_PATCH_POSITIVE_CONTROL
  = REAL_SLOW_AGENT_PATCH_PASS（真实 Slow Agent 全链，用户批准）**。
  ① **正控 case**（GEFCom impute_ssm→outlier_iqr，P0 数值实测确认）：
     identity 0 / A-only −0.15432 / B-only +0.04386 / A→B −0.10249 /
     delayed B-only +0.02719；**不给 LLM REMOVE_A 标签**——只给数值表与
     冻结 Workflow。
  ② **完整链（全部真实组件）**：FailurePatternCard → TTHASlowAgent.
     propose_edit（AgictoChatCompletionsBackend + CountingClient ≤2，1 次
     调用）→ EditManifest → EditController.apply_to_fork（ADD capability
     Skill，confirmed_cause=SKILL_LIBRARY_GAP）→ compile → ScopeExecutor
     Support replay → delayed → 正常 TTHAMethod @976 实际采用 → H0
     remove-skill 对照。
  ③ **结果（验收 6 条件全过）**：LLM 自主归因正确（edit_id=
     add-outlier-iqr-only-forecast-capability——"without preceding
     impute_ssm"，无标签提示）；Support replay +0.04386；delayed
     +0.02719（不翻转）；Skill 写盘；@976 chosen=cand_skill_*（执行
     +0.02719）；remove-skill 对照 chosen=cand_denoise_median（行动
     变化）。verdict = REAL_SLOW_AGENT_PATCH_PASS。
  ④ **确定性契约修复（诚实标注，非归因内容）**：LLM 三次未实例化
     {skill_id} surface 模板 + 未声明 surface 要求的上下文依赖 SHA——
     Harness 层按 surface 定义确定性补齐（模板实例化 + required
     dependency 从 snapshot 依赖表取）。不称 LLM 完全自主（契约格式
     层由 Harness 兜底）。
  ⑤ **意义**：补上"LLM 自主归因 → 可执行 Harness Update 落地"的证据
     （此前边界①：Weak Risk 修复是开发流程而非 Slow Path 自主）。
     development positive control——不宣称自然 Slow Path 能力。
- 2026-08-10（五十）：**里程碑定稿——DEVELOPMENT SLOW UPDATE CAPABILITY
  ESTABLISHED + NATURAL_HARNESS_OPERATION_PILOT 冻结**。
  ① **准确口径（用户）**：在答案明确的开发正控中，真实 Slow Agent 已能
     把失败证据转化为可执行 Skill，经真实反馈验证后影响下一轮行动。
     分工：LLM 自主完成语义归因与 Program 选择；Harness 完成格式实例化/
     依赖绑定/验证/接纳（确定性补齐不替 LLM 选算子）；Runtime 决定修改
     是否有效。
  ② **已打通**：Fast Path（Source→LLM 选择→Skill→delayed）+ 负反馈
     （CONFLICT/NEGATIVE→Skill 降级）+ Slow Path 正控（失败归因→LLM
     修改→executable Skill→下一轮生效）。
  ③ **尚未证明**：自然运行中自动触发 Slow Agent；非预选自然失败上找到
     有效修改；Observation/Scope/Risk/Memory/Control 等其他 Surface 的
     自主修改；Slow Agent 多案例稳定性。
  ④ **Pilot 冻结（不增加开发正控、不再调 Prompt）**：NATURAL_HARNESS_
     OPERATION_PILOT——新 virgin cohort（traffic offset=120）；连续 3
     个真实在线轮次（696/744、792/840、888/936 + 采用轮 936）；当前
     Harness 完全冻结；每轮 Support 预算 ≤2；正常写 Episode 和 delayed；
     运行中不修复；轨迹结束后只定位第一个自然 fault；仅当自然失败同时
     具有"明确 first fault + 可执行 Surface 动作 + 替代 headroom"才调用
     已验证的真实 Slow Agent 更新链。
- 2026-08-10（五十一）：**NATURAL_HARNESS_OPERATION_PILOT 运行 =
  PILOT_TRIGGER_CONDITIONS_UNMET（预注册档位，纪律正确）**。
  ① **3 轮在线轨迹完整运行**（traffic offset=120 virgin；6 个两步组合
     探测预算 ≤2/轮；Episode 写回 + delayed 打开 + Memory 跨轮累积全
     部工作；运行中不修复）。
  ② **第一个自然 fault**：轮 2 denoise→hampel_filter −0.0499（NEGATIVE）；
     反事实 A-only=0.0、B-only=−0.0499=A→B——失败来自 B 单步本身
     （denoise no-op 前缀）→ headroom 不满足（best=0.0<M）→
     **PILOT_TRIGGER_CONDITIONS_UNMET**（三条件缺 headroom，不调用
     Slow 链——纪律正确）。
  ③ **数据模式确认**：与 premise 扫描（0/4）一致——这些域的自然两步
     失败均为"单算子本身负"型，无"组合失败单步可救"案例；Program-level
     Slow Path 持续事件驱动（等真正可归因的自然案例），机制链
     （在线轨迹/fault 定位/三条件判定/Slow 链）全部就绪。
- 2026-08-09（二十六）：**外部审核第二轮 + 方法层 Skill 选择与执行验收 PASS**。
  ① **外部审核裁决（第二轮）**：上一轮只闭合"候选供给"——Skill 进池但未被
     选中（格 2 chosen=cand_impute_linear）；verdict 应为
     METHOD_LEVEL_SKILL_CANDIDATE_SUPPLY_PASS + METHOD_LEVEL_SKILL_SELECTION_
     AND_EXECUTION_PENDING。4 点核对**全部属实**：a) Skill-alone chosen 非
     cand_skill_*（以 JSON 为准）；b) 合并顺序 Agent+Skill 且 CandidatePool
     按 total_k 截断（candidate_pool.py:53）——Agent 双候选会挤掉 Skill；
     c) Support 验收是脚本旁路调用 executor.evaluate（冻结 Patch steps，
     非 chosen Program）；d) applicability {"const": true} 未验 Context
     mismatch（记录为已知边界）。
  ② **实现（7 条修复）**：
     - **Slot 保留**（fast_agent）：skill 候选存在时 Agent proposals 截断
       到 1（identity + 1 Skill + ≤1 Agent）——Skill 不被候选预算挤掉；
     - **中性 selector**（wiring：prefer_skill_in_select，默认 False 旧实验
       不变）：按公开候选 ID 顺序——优先 cand_skill_*，否则第一个非
       identity，否则 identity；不解析 Skill 文本（只匹配 ID 前缀）。
       **关键发现**：select 的候选列表在**第二个 user message 的
       public_input JSON**（不在 instruction 文本——extract_instruction 只
       取第一个 content）→ _select_candidate_ids 改为从 request.messages
       全部内容解析。
  ③ **验收（run_v1_method_level_skill_selection.py，6/6 PASS）**：
     METHOD_LEVEL_SKILL_SELECTION_AND_EXECUTION_PASS——格 2（Skill-alone）
     chosen=cand_skill_outlie-target-v1（**Skill 被选中**）；DoublePropose
     （backend 提交 2 个 Agent 候选）时 Skill 仍保留且被选中（slot 保留
     有效）；chosen Program steps 从 **PreparationResult.program** 取得
     （== 冻结 Patch）；沿 chosen 的 ScopeExecutor Support（+0.0272 @976，
     accept 算链路完成）；格 1 无 skill 不供应；格 3 Episode 引导不变。
  ④ 回归：binding smoke / supply acceptance 不崩（prefer_skill 默认关闭、
     h0 无 skill → 旧实验行为不变）。
  ⑤ **状态**：METHOD_LEVEL_SKILL_SELECTION_AND_EXECUTION 已闭合——Skill
     候选稳定保留（slot 保证）、可被选中（中性 selector 正控）、沿 chosen
     Program 执行且接受 Support。仍开放（外部审核定义）：applicability
     权限分级（Domain/Context/local status）——已知边界；**下一 first
     fault = 正常入口自动反馈写回**：chosen Skill → Support receipt →
     Episode 写回 → delayed 更新 → 下一轮 prepare（TTHAMethod.prepare()
     目前只执行 prepare）。

## 8. 待办与已记录限制

- ✅ 已办：方案 B（R1 weak、R2+ radius 由真实独立 Context 校准，落盘验证）；
  verified_risk 与四态对齐（P4 最小修复，结果不变坏）；verdict 口径
  CONTROL_PATH_MECHANISM；Program Scope/Evaluator 对齐（§7 十一：
  SCOPE_ALIGNED_MECHANICAL_CLOSED_LOOP_PASS——规范语义 = training_windows_only
  逐窗口跨 cohort，verifier 全程 0.35，Support/delayed 各决策点重执行不拼接）。
- 待办（方法层批准前，优先级 1→2→3）：
  ② 真实 fast_agent 注入 vs 空注入对照（基础设施已就绪；验收维度：注入内容
    来自当前合法 signed 判定、Agent 是否引用对应 Experience、生成 Workflow 可
    编译、候选方向与 Memory 证据一致、Target Support 比 A3 更快命中正向、
    不改变计划时由确定性 Support/仲裁安全兜底——"计划改变"本身不是成功）；
  ③ Agent 选择质量维度（注入 vs 空注入时 chosen_candidate_id 是否改变、方向
    是否与记忆一致）；
  ④ 只修会改变仲裁行为的口径缺陷（P7 区分"正负证据"vs"观察过的 Context"——
    0.0 gain 可留校准池但不作证据；P8 distance key collision 报告性暂缓；
    不建设新 ID/Hash 体系）。
- 可选未做：方案 A（Source 加第 3 个真实不重叠 origin——需要新评估；方案 B
  已满足语义，A 仅在需要 R1 即 radius 时启用）。
- 已记录限制：单方向单链（NOAA→GEFCom，3 轮 B=2）——claim ceiling：实验级
  接线 ≠ 方法层 fast_agent trace 触发；"跨域经验有效"结论需方法层验证后升级。
