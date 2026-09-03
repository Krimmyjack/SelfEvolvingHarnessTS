# HEC-1 独立代码评审清单(D4 落地后、Phase S 发车前)

日期:2026-09-03。地位:主线建议的 Phase S **第五放行条件**(sol 四条件之外;待用户/sol 认可)。
评审者 = **非作者** agent(grok 4.6-xhigh);只读代码、只写**新增**对抗性测试文件、不改被评审代码、不 git 提交、
0 LLM 网络调用。产出一份 `docs/HEC1_REVIEW_REPORT_<date>.md`:逐项 PASS / FAIL / N/A + 证据(file:line 或测试名)。
任一 FAIL → 回 Opus 修 → 复评该项。**评审的目的是抓"跑得通但算错了"的接线错误**——这个项目所有错误叙事都来自
runner,不来自模型。

先读:项目 `AGENTS.md` §3/§4/§7/§8;`docs/HEC1_CONTRACT_SKELETON_2026-09-03.md`;`docs/D4_HEC1_WIRING_SPECS_2026-09-03.md`;
`docs/HEC_EVOLUTION_MAINLINE_PLAN_2026-09-02.md` §4/§5/§10.1–10.3;Opus 的实现文件与其测试。

## A. 隔离与曝光(任一 FAIL 即一票否决)

| # | 检查 | 方法 |
| --- | --- | --- |
| A1 | 评价面(o+144)的 Outcome **不进任何臂的 Episode bank**,只进 `scoring_ledger` | 读 W4 写回路径;写测试:跑两单元后断言 bank 内无 `read_origin == o+144` 记录 |
| A2 | 外环 replay 的 bank **只含本臂本顺序已处理单元**,不含未来单元、他臂、held-out | 读 `consolidate` 的 bank 构造;测试:在第 5 单元触发外环时,bank 中无 position>5 的记录 |
| A3 | held-out `[80:120]` × {4056,4296,4536,4776,5016} **任何路径不加载** | grep 数据加载;测试:runner 全程记录读窗集合,与 held-out 对集合交集为空 |
| A4 | 三顺序单元序列 **逐项等于** `p4ac_hec1_course_supply.json` `proposals.orderings` | 直接比对 |
| A5 | Best-Safe-Global / readout 审计脚本 **不被 runner import**;runner 不读其工件 | grep import 图 |
| A6 | Fast 原始决定记录已**脱敏**(无 API key / Authorization 模式) | `ps0c.redact` 或等价;测试:注入含伪 key 的回复,落盘无明文 |

## B. 臂的语义

| # | 检查 | 方法 |
| --- | --- | --- |
| B1 | **A5-frozen 每单元后 store/snapshot 精确回到 K0**(Skill id 集、Scope、Draft ledger 全部) | 测试:跑 3 单元,逐单元后比对与 K0 的 Skill/Scope/ledger 集合相等 |
| B2 | A3-online 起点 = h0,无任何 Source 卡 | 断言初始 Skill 集 |
| B3 | Static = raw model 全部序列,增益恒 0,不发 LLM | 账本 llm=0;评价面 gain 全 0 |
| B4 | K0 为空时臂集缩为 Static / A3-frozen / A3-online,**不跑**与 A3-online 等价的臂 | 读分支;测试:空 K0 → 臂列表 |
| B5 | 各臂同数据、同面、同预算;A5 无任何预算例外 | 读配置;grep 臂名分支 |

## C. 门与执行权

| # | 检查 | 方法 |
| --- | --- | --- |
| C1 | **P4 `_gate`(含 coverage_floor)是唯一激活权威**;online_loop 的 delayed 批准不单独 `activate_approved` | 读调用图;测试:合成 online 批准/P4 拒绝的单元 → Active 集不变,`gate_disagreement.resolved_by=="p4_gate"` |
| C2 | 外环 replay 通过者只成 restricted Draft,**从不**写 Active / 部署 | grep `activate` 调用点;测试:replay 全过的候选在下一单元前仍不可部署 |
| C3 | restricted Draft 以 `requires_target_support` 供给,须过 Support + delayed 才 Active | 读供给路径 |
| C4 | 风险线、material、MIN_TREATED、0.35 校验器 **未被任何新代码重定义或覆盖** | grep 常量来源;与 `admission_policy`/`bounded`/`distance` 同源 |
| C5 | **20/19 分母**:覆盖率与受害分数分母、MIN_TREATED 基数动态取 served 数;无字面 `20` 参与分母 | grep `20`、`/ 20`、`range(20)`;测试:19 序列面上 hf 分母为 19 |

## D. 三态机与归因

| # | 检查 | 方法 |
| --- | --- | --- |
| D1 | `classify_failure` 用 Source-v3 三案例回放:1896→FLAGGED、2376→WAITING、2616→REVISABLE | 跑 Opus 的测试并**独立复算**一遍(从 `p4w3b` 工件重建输入) |
| D2 | NEW_ENTRANT / CONTINUING / LEFT 的序列对齐:`per_series_gain` 位置 ↔ served 字典序 | 用 1896 delayed 非零位还原出 T263/T264/T266/T267/T269/T270 |
| D3 | 修订 ≤2、验证 ≤3 的计数在跨单元、跨外环步时不重置 | 测试:构造第 3 次验证 → 归档 |
| D4 | WAITING 的自动验证不耗修订;FLAGGED 拒绝加子句 | 测试 |
| D5 | 任一状态**不删** Draft(证据保留) | grep 删除路径 |

## E. Scope 工具链

| # | 检查 | 方法 |
| --- | --- | --- |
| E1 | Slow 输出的数值阈值被**忽略**并记 `LLM_THRESHOLD_IGNORED`;阈值只来自工具 | 测试:喂含 threshold 的 Slow 回复 |
| E2 | 工具候选阈值 = 冻结分箱边界(与 Scope 归纳同源),非新造 | 读 bins 来源 |
| E3 | 特征不在 12 词汇内 → 拒绝 | 测试 |
| E4 | `best_stump`(shadow)只记录、不影响 Slow 的选择或任何部署 | grep shadow 输出的消费者 |
| E5 | `NoFeasibleThreshold` 反馈 Slow ≤2 次后弃权,不无限循环 | 测试 |

## F. 预算与故障

| # | 检查 | 方法 |
| --- | --- | --- |
| F1 | 单元臂 LLM 帽、Forward 总帽:第 N+1 次调用**在后端前**被阻断且不计费 | 测试:mock backend 计数 |
| F2 | `llm_fast` / `llm_outer` / `replay_fits` / `shadow_fits` / `course_fits` 分账;`replay_fits` ≤ 25% 有守卫 | 读账本写入点 |
| F3 | UnitFault 列表 → identity 弃权 + 继续;RunFault 列表 → 中止 `RUN_BLOCKED_NO_VERDICT`;**判词不在 RunFault 分支里写成科学词** | 读异常分类;测试:注入 BACKEND_UNAVAILABLE → 判词为仪器词 |
| F4 | 检查点每单元臂落盘;`--resume` 按 (ordering, position, arm) 去重且不重复计费 | 测试:中断后 resume,LLM 计数不重复 |
| F5 | cache 键 = 规范化 prompt 字节;**不含臂名/顺序名**(否则永不共享),**含**检索到的 Skill 内容(否则错误共享) | 读键构造;测试:两臂同 prompt 命中,不同 Skill 上下文不命中 |

## G. 回归与中立

| # | 检查 | 方法 |
| --- | --- | --- |
| G1 | `run_source_line_v3.py --dry-run` 输出字段与 `p4w3_source_line_v3.dry_run.json` 一致 | 跑并 diff 键集 |
| G2 | `methods/ttha/*` 零改动;若有,列明并核 lock 处置 | `git diff --stat` |
| G3 | 全量回归对基线(`_scratch/pytest_baseline_tests_tree.txt`,46F/9E/747P)**差集无新增失败** | 跑 `python -m pytest tests -q`,比对 FAILED/ERROR 行差集 |
| G4 | 七项 smoke 全过且 0 LLM | 跑 `--smoke`,核账本 |

## H. 自动门脚本(若启用 §2c 延长信封,本组为必审;作者自跑这两份脚本的前提是它们经过非作者评审)

| # | 检查 | 方法 |
| --- | --- | --- |
| H1 | `audit_hec1_instrument.py` 八项全部为**机械断言**(无"人工判断"分支、无阈值可调参数、无按效果读数的分支) | 读代码;grep gain/advantage/utility 字样不得出现在判定逻辑 |
| H2 | 任一项 FAIL → 退出码非 0 且 runner 续跑逻辑读该退出码;不存在"WARN 继续"路径 | 测试:注入一项 FAIL → 续跑被阻断 |
| H3 | `audit_hec1_k0_freeze.py`:每张 Active 卡的 Support/delayed 记录来自 P4 权威门;阈值来源字段 = tool;无 replay 激活;子句 ≤2 | 合成一张 replay 激活的卡 → FAIL;合成一张 LLM 数值阈值的卡 → FAIL |
| H4 | 两脚本只读工件与 store,**不读评价面 `scoring_ledger` 的增益**(仪器门不得看效果) | grep 读取路径 |
| H5 | 后台进程 + 检查点 + 心跳:`--resume` 后 LLM 计数、单元集合与中断前一致,无重复计费、无遗漏单元 | 测试:中断于第 k 单元臂后 resume |

## 回报格式

(1) A–G 逐项表(PASS/FAIL/N/A + 证据);(2) 新增的对抗性测试文件清单与通过数;(3) 任何"跑得通但算错"的发现单列并给
最小复现;(4) 未能评审的项与原因;(5) 是否建议放行 Phase S(仅基于仪器与接线,不涉效果)。
