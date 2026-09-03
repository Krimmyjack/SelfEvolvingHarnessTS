# 晨报 2026-09-04(执行方 / Opus;夜班:D3/D4 → 裁定落地 → 合同冻结 → Phase S 发车)

**一句话**:六条裁定全部落成**可机械检查的强制点**,活体臂循环补齐并由 28 项 0-LLM 端到端测试覆盖,
合同已冻结,Phase S 已发车。过程中被端到端测试逼出**四个真缺陷**,其中两个会让"状态 COMPLETE"
的空跑看起来像成功——这是本轮最有价值的产出。

**本报告不含曲线结论**:三顺序未齐(0/3 在册),按预注册只报仪器与生命周期计数。

---

## 1. Harness 行为改变了什么

**冻结前(本轮补齐)**:活体臂循环。此前只有接线,没有跑课程的东西。现在 `run_course` 按
「单元 × 臂」推进,每 cell 走:Context → 检索 → Fast 提案 → Support-A 探针 → bounded 准入 →
delayed(+48)**权威门** → Active / 三态 restricted → 评价面(+144,只计分)→ 写回(仅 online 臂)
→ 每 5 单元外环。

三处承重设计,理由都是"标志位会被忘记,结构不会":

| 设计 | 不这么做会怎样 |
| --- | --- |
| **frozen 臂每单元用起点 snapshot 重建 store + method** | "online 臂关掉写回"只要有一条路径漏判就会带走知识;重建之后物理上没有东西可带。代价是每单元一次 snapshot 编译,毫秒级 |
| **内环 Slow 关闭,全部 Slow 归外环** | 编辑又在触发它的那个单元上被评判,回到 `S3_EDIT_REJECTED` 的 n=1 几何 |
| **`p4_gate` 是唯一激活权威** | Source-v3 round 2856 那种"lifecycle 说 approved、门说不过"会各自签发执行权 |

**裁定的强制点**(每条都能被 `assert_frozen` 或测试抓住,不只是文字):

- 动态分母:对五个模块做**源码扫描**,`/ 20`、`* 20` 一类写法出现即冻结检查失败;另测
  `[200:239]` 39 条 → 两面 roster eval 为 (20, 19),逐 cell 断言 `coverage == treated/served`。
- Phase F:`assert_launchable("phase_f")` 需**判词 SUPPORTED** 且 `seal_released=True`,后者
  runner 自己传不进来。四种组合由测试固定。
- Slow 数值阈值:由 `clause_from_slow()` 丢弃并记 `LLM_THRESHOLD_IGNORED`。**schema 不改**——
  它在 `methods/ttha/schemas/`,改动会轮转 snapshot lock,与 D4 自身原则冲突。
- 普查 key:`op(json(params))` 按序拼接,含算子、顺序、参数;行为指纹只在**共同出现过的单元**上
  全等才折叠别名。

---

## 2. 真实或可控数据上观察到了什么

### 2.1 四个真缺陷(全部已修,均不涉阈值/合同/特征)

这是本轮最实质的观察,因为其中两个的表现形式是**"状态 COMPLETE"的空跑**:

1. **`base_url` 校验发生在首个 stage 之前**。离线传 `"offline"` → `compilation_status: failed`、
   零 stage、零候选,而课程照样写出 `COMPLETE`。首次 6 单元离线跑就是这样"通过"的。
   改用 `https://offline.invalid/v1`(RFC 2606 保留域,真逃逸也连不上)。
2. **`pattern_id` 必须是 canonical id**(`^[a-z][a-z0-9]*([-_][a-z0-9]+)*$`)。`hec1-A5-frozen`
   带大写 → 整轮在第一个探针前死掉;6 单元里 4 个 0 probe。改 `ArmSpec.slug` 全小写。
3. **`+144` 评价面并非总可评**。D2 只在 **origin** 上筛过 sMASE 是否有定义;+144 再往后 144 步
   可能越过序列已观测段,直接崩掉整个课程。新增 `FaceNotEvaluable`(UnitFault 子类):该单元
   **对所有臂同时**不贡献曲线点,并由仪器第 8 项断言"不得只对部分臂丢失"。**这会压低曲线的
   可用 N**——见 §3。
4. **replay fit 25% 帽可被整屏超掉**。原实现只判"还剩没剩",而一屏 screen 要把候选在**每个**
   已处理 cell 上重打;且单 cell 实际 **3** fits(Static 参照 1 + scoped evaluator 自身 raw/program 2),
   不是 2。改为 screen **前**按 `3 × 已处理 cell 数` 预估,超出即跳过并记
   `REPLAY_FITS_BUDGET_SPENT`;帽改为对**全课程投影** fits 取 25%。

### 2.2 三态机在真窗口上的回放(0 LLM / 0 fit)

与主线 09-02 手算逐条一致(`p4w3b_source_line_v3_clean_post_fix_replicate_1.json`):

| 窗口 | 更替 | 续/新/离 | 持续成员负贡献 | 新进入负贡献 | 败线 | 判定 |
| --- | --- | --- | ---: | ---: | --- | --- |
| 1896 → 2136 | 6 → 9 | 5/4/1 | **0.9049** | 0.5018 | aggregate + msh | `FLAGGED` |
| 2376 → 2616 | 5 → 2 | 1/1/4 | 0.0904 | 0.0 | coverage_floor(仅) | `WAITING` |
| 2616 → 2856 | 7 → 15 | 6/9/1 | **0.0** | 1.7288 | msh | `REVISABLE` |

等级 `INSTRUMENT`:这是机制回放,不是新证据。

### 2.3 Phase S 收口:第四次空 K0,但这次知道空在哪里

13/13 单元、**35 LLM**(帽 120)、course fits 23、held-out 读 0、墙钟约 48 min。判词
**`A5_TREATMENT_EMPTY`**。

| 读数 | 值 |
| --- | --- |
| Fast 提案分类 | 9/9 `PROPOSED`(0 空输出、0 格式错误) |
| 出现风险拒绝的单元 | 4 |
| 产生 winner 的单元 | **3 / 13** |
| 走到 delayed 权威门 | 2 |
| **通过门 / 激活** | **0 / 0** |
| Draft | 1 张,状态 `WAITING`(仅覆盖底线败),整课程未再遇 |
| 外环步 k=1 / k=2 | 两次都是 `the census produced no candidate`;`replay_fits` = 0 |
| `+144` 不可评单元 | **0**(两个 Source 块上全部可评) |

**这是第四次空 K0(v1/v2/v3 + Phase S),但第一次能定量说出空在哪里。** 不是 Slow 失败——
Slow **一次都没被点着**,因为外环找不到候选;不是协议摩擦——9/9 提案合法。空的直接原因是:
**"同一程序在两个以上已处理单元上取得 POSITIVE"这件事没有发生过**。按 `AGENTS` §6 的
first-fault 表,这落在 **Program headroom / 供给面**,不是记忆面。n=13 仍不确立机制,但比
n=5 时的"随机"有方向。

按预注册分支执行:臂集缩为 **Static / A3-frozen / A3-online**,**不跑等价的 A5-online**,
**判据 3 不评分**,判据 1/2/4 照考,**不重跑 Phase S 以制造 treatment**。

两份审计:K0 → `A5_TREATMENT_EMPTY`(空 K0 合法通过);仪器 **8/8**。

---

## 3. 当前最大的方法不确定性

按承重程度:

1. **曲线的可用 N 会低于 26,且低多少现在不知道。** `+144` 不可评的单元不贡献曲线点;
   Forward 前 6 单元里 origin 2856 已命中一次。若命中率是 1/6 量级,26 单元会掉到 ~22,
   统计功效按 cohort 计本就吃紧。**这不是可以靠改课程解决的**(合同已冻),只能如实报。
2. **`FLAGGED` 可能吃掉大多数失败,让修订环几乎不启动。** D2 已测出模式稀疏单元为 0——
   初始化谓词 `z_peak>=3` 在 KDD 上几乎不筛人,治疗集偏大、持续成员多。若 `FLAGGED` 命中率
   高于 Source-v3 的 1/3,判据 2 会因"没有可修的失败"而不成立,而这**不是**进化无效,
   first fault 在 Observation/Program 面。只能由 Phase S/T 的真实分布回答。
3. **replay 帽在 26 单元下会真的绑定。** allowance 78,而 k=5/10/15 三屏累计 15+30+45=90,
   第三屏起被饿死,后期外环筛的候选变少。这是帽在起作用而非故障,但它会影响后段的铸卡率,
   报告时必须和"外环没找到候选"分开。
4. **`llm_cache` 的分叉纪律仍未实测。** 设计上 A5-online 与 A5-frozen 的分叉只应来自记忆差异;
   Phase S 是单臂,验不到这条。Forward 才是第一次真考。

---

## 4. 是否仍与目标一致

是。两处自查:

- 头条仍是**曲线**;读数脚本现在就如实输出 `HEC1_INCONCLUSIVE`(0/3 顺序在册),判词词表在
  看到任何数字之前就冻结了。
- 反过度工程(§7):新增 SHA **0**、Gate **0**、Manifest **0**、算子 **0**、观察特征 **0**、
  阈值改动 **0**。W2 的分箱边界是**读取** `contracts/observables` 而非复制,正是为了不产生
  第二份会漂移的同义数据。两份审计脚本是 sol 明确要求的"脚本按按钮",不是自发加的门。

一个诚实的限定:**本轮交付的仍是仪器**。按 §7 它不算方法进展;方法进展要等 Phase S/T 的读数。
本报告不主张任何 Capability。

---

## 5. 下一项最小纵向切片

**Phase S 收口 → K0 机械审计 → Forward。** 已在自主信封内,不需要再等人。
续跑条件按合同 `AUTO_CONTINUE_CONDITIONS` 八条机械判定,其中放行只看仪器八项,不看效果正负。

再往后的第一个**需要判断**的问题是 §3 第 1 项:等 Forward 跑完,把"可用 N"的实测数交出来,
再决定统计口径是否需要在 HEC-2 里调整(HEC-1 内不动)。

---

## 6. 仪器健康表

| 项 | 值 |
| --- | --- |
| 代码/测试阶段 LLM | **0** |
| **Phase S** | 13/13 单元,**35 LLM**(帽 120),23 fits,48 min,`A5_TREATMENT_EMPTY` |
| **Forward**(进行中) | pid 43788,26 单元 × 3 臂;心跳 `.hec1_runs/forward_live/heartbeat.json`;LLM 投影 ≈270(帽 500) |
| `tests/main_protocol` 全量 | **415/415**(原 386 + 新 29),零回归 |
| HEC-1 测试(重跑于发 Forward 前) | **73/73**(我交付 64 + 另一执行线补 9) |
| 聚焦测试 / 端到端 / smoke | 36 / 28 / 7,全过 |
| 仪器八项(Phase S) | **8/8**(`hec1_instrument_phase_s_live_r2.json`) |
| K0 审计 | `A5_TREATMENT_EMPTY`,空 K0 合法冻结 |
| 仪器八项(离线 6 单元课程) | **8/8**;用**修复前**旧工件跑时第 7 项正确 FAIL |
| K0 审计 | 空 K0 与非空 K0 两种情形均实测通过 |
| 预算算术 | Forward 满 K0 **410**、空 K0 **270**、Phase S **69**,均在帽内 |
| held-out 读 / UCR TEST 读 | **0** / **0** |
| 阈值改动 | **0** |
| `methods/ttha/*` 本执行线改动 | **0**(核法见下注) |
| Best-Safe-Global | 全 26 单元约 **1820 fits**,已获预批,**尚未花** |
| 课末读数 | `HEC1_INCONCLUSIVE`(0/3 顺序)——Phase T 前的预期状态,非负结果 |
| Phase F | 未触;`assert_launchable` 里写死需判词 SUPPORTED + 人给密封授权 |

**`methods/ttha/*` 的核法与勘误**:该目录内 `harness/compiler.py`(09-02 18:36)与
`harness/store.py`(09-02 20:15)有未提交改动,两者写入时间**早于本班次**,系另一执行线的未提交件,
按接手简报 §6 未触碰;其余只有 `__pycache__/*.pyc` 因 import 重新生成。

**后台运行**:Phase S 以脱离终端的进程跑,每单元臂落检查点 + 写心跳;会话若死进程继续,
恢复只靠 `--resume`(已实测 18/18 cell 复盘、0 fit / 0 LLM、读数逐位相同)。

---

## 6b. 并行执行线:同一批文件现在有两个写者(须裁)

本轮进行中,`audit_hec1_instrument.py`、`run_hec1.py`、`outer_loop.py`、`restricted_draft.py`
与两份测试在**我发车之后**被另一执行线修改(时间戳 10:43–11:19)。其中一处是我未撰写的
**工件不覆盖守卫**(`--label` + "refusing to overwrite",并引用一个尚不存在的 erratum 件)。

改动方向正确,且属 sol 已批的独立评审范畴(「若已有评审发现 A 类错误,则必须修复」),所以我
**没有**回退它。我按"不信任、先复核"处理:重跑全部 HEC-1 测试(**73 项**,比我交付时的 64 多 9)
+ smoke 7/7 + 合同漂移 clean,全绿之后才发 Forward。

**需要主线裁一句**:Forward 正在跑的是**发车时的字节**,而后续 `--resume` 会用**当时磁盘上的字节**。
若评审在运行中继续改这几个文件,两者不一定一致,而 resume 是唯一的恢复手段。建议:
**三顺序跑完前,这几个文件冻结给评审只读**;确有 A 类错误则停 → 修 → 从 checkpoint resume,
并在账本记明"修的是仪器、不是科学"。

## 7. 待 sol / 待主线

1. **两份审计脚本请过目**(sol 已列为评审对象):`audit_hec1_instrument.py`(八项)、
   `audit_hec1_k0.py`(每卡三断言)。它们现在是"自动续跑"的按钮,值得被非作者看一眼。
2. **`FaceNotEvaluable` 的处置请确认**:我把它判为 UnitFault 子类——该单元对**所有臂同时**
   不贡献曲线点,而不是停整场(停整场会把可读的其余单元一起丢掉)。若认为应当在课程开跑前
   就把不可评单元标出来,那属于改课程,HEC-1 内我不动。
3. **replay 帽绑定的报告口径**:请确认后段"外环候选变少"要与"外环找不到候选"分列报告——
   我已在工件里分开记(`candidates_starved_by_the_cap`),但判词由主线出。
4. **课末判词仍归主线 + sol**:readout 脚本已在看结果前冻结,我只机械执行一次。

## 8. 待用户

**本轮无待办**——四个信封已批,Phase S 已发车,自主区间到 Interleaved 仪器核对完成为止。

唯一保留给你的:**Phase F 密封开启授权**。它在代码里写死需要「判词 SUPPORTED + 人给授权」两个
条件,runner 无法自己满足其中任何一个。三顺序跑完、readout 出一次之后,我会停在那里等你。
