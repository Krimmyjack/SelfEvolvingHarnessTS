# HEC-1 v1.1 修订请求(pre-data;主线裁定 + 待 sol 六裁 + 待用户三批)

日期:2026-09-03 12:xx。触发:Opus 完成 D3/D4/Phase S 并发车 Forward 后,第二审查线(另一 Fable)在**发车后**抓到
四处主结论级接线缺陷并已修(有回归测试,`tests/main_protocol` 424/424);主线复核又确认 `MIN_POSITIVE_UNITS_FOR_ADD = 2`
与 sol 已批阶梯 v2(供给档证据价 2→1)冲突。Forward 进程(pid 43788,11:25 起)执行的是**发车时刻的未提交字节**,
磁盘已是修复后字节,`--resume` 会切码。全部事实为 CODE FACT。

## 1. 主线裁定(不需再议)

- **R-A Forward 降为 `FORWARD_SHAKEDOWN`**:跑完、不 resume、不进曲线;其仪器数据(UnitFault 分布、时长、cache 命中、
  中继稳定性)全部保留入账;若中继中断则不恢复。降级依据五条:REVISE 开新壳绕过 ≤2 修订上限;replay 将"谓词解析
  不到 MIN_TREATED"的 cell 判 `aggregate_not_material`(结构性淘汰所有收窄候选,把 H3 当否决);replay 预算 25% ×
  "每次回放全部已处理 cell × 3 fits" 在第 2–3 外环步后耗尽;ADD 阈值 2 与阶梯 v2 冲突;发车字节未提交不可复现。
- **R-B 科学顺序 = 同一 commit**:修复 + v1.1 落地 → 非作者复核(grok,清单 B/C/H 组)→ allowlist 提交 →
  合同记 `code_commit`;runner 断言 HEAD 一致且 runner 文件工作树干净;顺序进行中任何修改 → 该顺序降 shakedown。
  五个 runner 文件在三顺序完成前只读。
- **R-C Phase S 在 v1.1 下重跑一次**(≤120 LLM):Phase S-v1(≥2 规则、K0 空)结果**保留入账**,不覆盖;Phase S-v2 在
  阶梯 v2 一致的 ≥1 规则下跑,两者并报。触发原因是**评审前置发现的规则冲突与接线缺陷**,不是结果不好——写明。
- **R-D 预注册叙事(在任何科学顺序前写死)**:HEC-1 最可能的正结果路径 = **ADD(新单元过门)→ Active → 0-LLM 召回
  → 探针位释放**;Scope 修订通道受两道结构性门夹(D1 无 outcome-free 分离量 + replay/覆盖成本),预期贡献小。
  若曲线为正而修订通道为零,叙事为"Skill 形成与复用驱动的进化",**不得**事后改写为"Scope 修订驱动"。
- **R-E 统计方案改为描述性**(主线自认 §5.2 原文自相矛盾:既写单位 = cohort 又写单元级符号检验;n=4 cohort 下
  单侧 α=.05 数学上不可达,最小 p=.0625):主读数 = 每顺序终点累计差 D_o 与每 cohort 差 d_c;**P1 成立 = D_o>0
  于 ≥2/3 顺序 ∧ d_c>0 于 ≥3/4 cohort ∧ harm online ≤ frozen**;报精确二项 p(标"描述性")与 cohort bootstrap CI;
  **不画跨顺序置信带**(同数据同 cache,非独立 seed);P1 用终点差(副:曲线下面积 / 中点差),不用"单调";确认性检验
  留 fresh 实验(≥8 独立 cohort,Wilcoxon)。预算不对称照报:online 臂比 frozen 多外环 LLM。
- **R-F 命名**:Best-Safe-Global = **offline in-budget comparator**(在 +144 Outcome 上事后选优),非可部署 baseline;
  Phase F "Fast-only 0 LLM" = **冻结 Skill 库的机械部署**,非训练期 Fast Agent;Phase S/T 同为 KDD → 积累只可称
  **within-dataset / cross-cohort**;曲线 = development mechanism curve。
- **R-G 评价面可评性前置**:D2 只筛了 origin,+144 面并非总可评。在任何科学顺序开跑前做 0-fit 可评性检查(只读
  "指标是否可定义"),得 `N_T_eff` 并写入 v1.1;`INCONCLUSIVE` 的 0.8 门以 `N_T_eff` 为基。单元不可评 → 所有臂同等
  丢弃并计数。
- **R-H 工程项(v1.1 一并)**:`audit_hec1_k0_freeze.py` 以独立脚本存在(H3 断言;Phase S-v2 若 K0 非空即用);
  `lost_activation` 计数(P4 门过而 online_loop 无 approved 事件);三分账 (a) 加机械判"部署程序 ∈ 起始 Active 集";
  分母扫描不跳过含引号行;冻结合同收据生成;账本补 09:39–10:04 条目;checkpoint `mode: live|offline` 不匹配 = RunFault
  (已做,入合同)。

## 2. 待 sol 裁(六项;主线建议在括号内)

1. **ADD 候选阈值**:`MIN_POSITIVE_UNITS_FOR_ADD` 2 → **1**(与阶梯 v2 供给档一致;restricted Draft 仍须在新单元过
   Support + delayed 才 Active,n=1 由未来验证承接;replay 筛选照旧)。
2. **replay 预算结构**(二选一或合取):每臂 100% 自身课程 fits;**或每次只回放最近 m=8 个 cell**(主线建议:
   **m=8 滑窗 + 每臂独立份额**——成本恒定、筛选仍有效)。
3. **WAITING 的自动再验是否消耗 ≤3 验证次数**(主线建议:**不消耗**,另设 WAITING 再检 ≤3/课程;覆盖是流行率不是质量,
   与 (d) 裁定一致)。
4. **统计方案** R-E(主线与第二审查线一致;请 sol 收回 α=.05 预注册)。
5. **Phase F 补条**:K0 空但判词 SUPPORTED 时仍开封(测的是课程内积累的 held-out 保持);判词 NOT_SUPPORTED /
   INCONCLUSIVE 时**不开封**。
6. **普查键**:算子 + 顺序 + 参数 + **Scope 根谓词**入键,故障类型不入键;指纹只折叠别名。

## 3. 待用户批(三项)

1. **预算**:Phase S-v2 ≤120;科学顺序 Forward-v2 / Reverse / Interleaved 各 ≤500(共 ≤1620;shakedown 已耗 ≤500
   另计)。若只批两条顺序,P1 判据改 "D_o>0 于 2/2",并标注。
2. **commit 时点**:修复 + 非作者复核通过后立即提交(先于 Phase S-v2);之后 runner 文件只读至三顺序完成。
3. **Best-Safe-Global prequential 版(≈+1870 fits)**:主线建议 **HEC-1 不做**,offline comparator + 诚实命名足够;
   留 HEC-2。

## 3b. sol 正式裁定(2026-09-03;主方向批准;Forward 降 `FORWARD_SHAKEDOWN`;批准一次"科学读数前修订";之后重跑正式 Phase S 与三顺序)

| 项 | sol 裁定 | 与主线/审查线建议的差异 |
| --- | --- | --- |
| ADD 门槛 2→1 | **批准**。单次正例只生成不可部署 Draft,须在后续独立单元过 Support + delayed 才激活;replay 永不授部署权 | 同主线 |
| replay 结构 | **不采滑窗**(避免引入 recency 机制)。每个 online 臂**独立拥有自身课程 fits 的 100%** 作 replay 预算,回放**全部可适用**历史单元;低覆盖记 `NOT_APPLICABLE` | 主线建议 m=8 滑窗 → **否**,按 sol |
| WAITING 耗次数 | **仍耗一次验证次数**:它消耗了 fit,也提供 Scope 覆盖不稳定的证据;否则 Draft 可无限等待并持续占探针位 | 主线建议不耗 → **否**,按 sol |
| 统计检验 | **改描述性**。n=4 cohort 下单侧 sign test 最小 p=.0625,不得再写 α=.05 确认性结论。正式标准:≥2/3 顺序终点差为正、≥3/4 cohort 为正、online harm ≤ frozen;p 值与 bootstrap CI 只作描述 | 同主线 R-E |
| K0 空时 Phase F | **不批准开启主 Phase F**。可继续 A3-online vs A3-frozen 作 **Target-local 自进化组件证据**;正典要求最终仍有自然数据上的 A5/A3/Static 同场,**不能拿 A3 结果替代完整 A5** | 审查线建议"K0 空但 SUPPORTED 仍开" → **否**,按 sol |
| census key | **批准加入根 Scope 谓词**:Task × Consumer × 完整 typed Program × root Scope;故障类型只作分层证据不入键;行为指纹只折叠别名 | 同主线 |

**sol 指出 v1.1 尚未落地**:磁盘仍是 `MIN_POSITIVE_UNITS_FOR_ADD=2`、`REPLAY_FITS_SHARE=0.25`、统计项仍写 α=.05,不能直接发正式实验。

**发车前再补四道机械门(sol)**:
1. **预扫全部 +144 评价面可评性**——只读 mask、不读效用;学习单元保留,无法计分的单元预先标记,所有臂一致处理。
2. **完成 `audit_hec1_k0_freeze.py`**——Phase S 后机械确认 K0 快照、Skill 资格及 A5/A3 隔离。
3. **P4 `_gate` 是唯一数值权威**——P4 过而生命周期事件未批准 → 记 `lost_activation`;**科学顺序中出现任何 gate disagreement 即降级该顺序**。
4. **用真实 19-series 行为测试锁动态分母**(不能只靠扫源码 `/20`);**补齐基于单元起始 Active Skill 的召回归因**。

**后续执行顺序(sol)**:(1) 当前 Forward 可跑完收仪器数据;中断则不得 `--resume`;无论结果不进曲线。(2) 落地 v1.1、完成测试与非作者复核。(3) HEC 文件 allowlist commit,合同记录该 commit;不建额外哈希体系。(4) 重跑 Phase S-v1.1,旧 Phase S 保留并标 `superseded`。(5) 同一 commit 下依次跑正式 Forward、Reverse、Interleaved;**只按八项仪器检查自动续跑,不按效果**。(6) 非 runner 作者做课末读数与 Track A/B 判词。(7) 停在 Phase F 密封开启前;Phase F 仍需**非空 K0、HEC-1 支持完整 A5 主张、用户人工开封**三者。
**只跑两顺序 → 不得改 2/2,只能记 `INCONCLUSIVE`**(主线原建议"改 2/2 并标注"→ 否)。Best-Safe-Global prequential 扩展与 TSFM 暂缓。

**授权句(sol 拟,待用户明示)**:
> 批准 Phase S-v1.1 ≤120 次 LLM,并批准正式 Forward、Reverse、Interleaved 各 ≤500;当前 shakedown 开销单列,不授权 Phase F 开封。

## 4. 序(批后)

修复五处 + R-H → grok 复核(B/C/H)→ commit → v1.1 合同 `assert_frozen` → 0-fit 评价面可评性 → Phase S-v2 →
`audit_hec1_k0_freeze` → Forward-v2 → 仪器八项 → Reverse → 八项 → Interleaved → 八项 → 停(读数由非 runner 作者
实现、主线出判词、sol 确认)→ Phase F(用户开封)。Shakedown Forward 的仪器报告先于一切入账。
