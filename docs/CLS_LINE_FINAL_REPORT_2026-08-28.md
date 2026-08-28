# 分类开发线终态报告(2026-08-28 收口)

主线定稿;裁定链 = sol 裁 B(经用户转达)+ 台账 2026-08-28 15:5x 条。本文为分类线论文章节的事实底稿:每一行主张只许引用本文列出的工件与提交,措辞按本文口径。

## 0. 一句话终态

在共享 Harness 的第二任务(分类)上,"经验以最低权限入场、由当前数据检验、按反馈修订"的完整闭环已在开发集重复证明,并在密封处女域证明了条件化与安全;族内密封正迁移因公开资源约束当前不可考,留由论文级主实验(可跨任务)承担。

## 1. 四行证据状态(冻结措辞)

| # | 主张 | 等级与状态 | 关键读数 | 工件/提交 |
|---|---|---|---|---|
| 1 | 带 Scope 经验卡的端到端收益 | **development 已复证**(两跑同向) | 承重五轴卡对无卡 **+0.6860**(distinct 五单元,两跑逐字节复现);叶 Scope 卡(L1)同底 +0.2127 ×2 | `l1_ladder_v2_replay_r1/r2.*`(`ecbe116`);`sa1_minimal_r1/r2.*`(`5ff76b5`/`0f10ec4`) |
| 2 | Skill 随 Gain/Harm 修订 | **development 已复证**(两跑同向;收益目前主为"避免重复错误") | 卡版本链 v0→v3/v4;再遇位:修订臂零供给零挨拒 vs 冻结臂再供给再挨拒;**修订确定性**(两跑前四版内容 sha 逐字节同) | 同上 + CAP-1b 出口 A 判定 |
| 3 | 密封处女域的条件化与安全 | **sealed 已证** | 族外卡全表面沉默(供给 0);双臂有害提案(−0.2750/−0.0500)全被 Support 拦;无 headroom 靶三臂正确 identity;harm 0 | `capstone_epilepsy2_final.*`(`b95a853`/`8568ca8`) |
| 4 | 密封、Scope 匹配场上的正迁移 | **当前 unavailable(资源约束,非方法负结果)** | 公开 UCR 单变量档案"二分类∧等长∧小规模∧全新"格合格 0(反事实自证;metadata 与归档副本同 sha) | `cap2_selection.*`/`cap2_sequential_exam.*`(`c37948f`/`9fa0c79`);终判 `POOL_EXHAUSTED` |

## 2. 定格的方法机制(v1 冻结对象)

1. **权限阶梯 v2**:1 条强正例(双门 POSITIVE)→ supply-only 卡(供应候选,不执行、不重排、不压制);2 条独立未引导正例 → Source 供给档;TRY/RISK/执行/部署门价不变。谱系:C40"权力越大 Scope 越窄"的对偶。
2. **Scope 规则 v2(承重五轴)**:task_kind × consumer × metric × Pattern family × Program 几何;偶然 observation 叶不入初始 Scope;族轴引用先冻家族定义。效应实证:同一张单例卡,叶 Scope 回收 +0.2127,承重轴回收 +0.6860。
3. **三写回(修订环)**:正向→证据账本 append;冲突→由实拒单元 pattern view 机械编译排除条件,PATCH 收窄 Scope;负向→降权/排除。全走冻结 EditController、SHA 前置、内容 sha 版本化、快照血缘可回滚。**单调收窄自主、任何扩张按阶梯证据定价**;供给转化系受引导、计零,阶梯不可自举。
4. **当前 Target 主权**:verifier + Support/delayed 双门 + harm 否决,对一切历史知识不减免;delayed 拒 → identity。
5. **归因仪器**:episodes.source_skill_id / source_skill_revision(内容 sha)/ round.scope_match_by_skill_id / guidance_conditioned_by_skill_id / dedup_swallowed。

## 3. 诚实边界(开放主张,论文不得越线)

- 修订环两跑**只买到成本(避拒/省探),未买到质量**(修订臂与冻结臂 regret 恰 +0.0000);"持续修订带来稳定性能提升"未证,留主实验。
- R3(负向/害撤权)仅离线历史重放背书,无 live 案例(live 课程零 harm,不造 harm 求覆盖)。
- 冷启动发现召回率低(三课七产例位 29%/位)系已量化边界;提案器合法性缺陷(level_shift 连 verifier 都不过)留 Stage 3 答案题。
- 所有正效应均 GunPoint 族内、development 级;卡族覆盖有限系 CAP-2 实证(全档案无第二个可封考场)。
- Q1 授权令牌语义(收窄 PATCH 借用 RETRIEval_MISS 因码)修复中(v1 冻结准备);Q7 四不可达轴声明、Q11 交集编译器重复叶同批。
- py3.12 f-string 测试收集债(`test_skill_revocation.py`)系他线文件,挂账未修。

## 4. 退役线登记

- **S1-v2(课程自举复利)**:三课三空场,系统性结论 = 提案语义 × 供给档价失配;终掷硬帽执行,全停(`9894a5d`)。其管道工程全部由 L1/SA-1 继承。
- **CLS-CONF 独立确认链**:本地池语义耗尽(r1/r2/r3a 三轮停摆)→ 下载批 D1 计算不可行终止;"开发数据可重复用、承重确认需处女数据"原则由此入典。
- **CAP-2 密封批**:池空终局(见第 1 节第 4 行)。

## 5. 对论文级主实验的设计输入

1. **反馈样本量门**:TRAIN 对半后每面 ≥20 行(材料线 ≤0.05)作为选靶硬门——M-1(余量门控)与本线贴线案例的直接教训。
2. **族覆盖预算**:评测池须含 ≥2 个独立 defect family × 各自 ≥2 个 Scope 相容正向场 + ≥1 个族外守卫场;一次性预注册,禁逐个追靶。
3. **密封考选线自由度**:能力级密封正迁移可由数据资源充足的任务线承载(预测线:Monash 资源 + 已有 A5>A3 +31.7% 成本优势账);分类线以本文四行定格贡献。
4. **v1 冻结边界**:Skill/Memory 结构冻结后,Stage 3 许可触碰面 = instruction/决策策略层;Stage 2 只读 Skill 层。

## 6. 工件总索引(承重链,时序)

`s1v2_course_freeze_v4` → `s1v2_v4_forward_run1`(9894a5d)→ 阶梯 v2 裁定 + `l1_ladder_v2_replay_r1/r2`(74978c0/ecbe116)→ `sa0_wiring_audit` + `SA1_SKILL_ADAPTATION_DESIGN`(22256a9)→ `sa1_minimal_r1`(cf2eb12/5ff76b5)→ CAP-1b(c0f6d1e)→ `sa1_minimal_r2`(0f10ec4)→ `capstone_epilepsy2_final`(b95a853/8568ca8)→ CAP-2 冻结(6a604d1)→ `cap2_selection`(c37948f/9fa0c79)→ 本文。
