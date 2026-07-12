# P5-A.3 VERDICT — Architecture-Complete Confirmatory（终审）

- **日期**: 2026-07-10（用户完成平台侧 key 轮换 + `DEEPSEEK_API_KEY` 注入后亲自执行）
- **协议**: `results/Stage2/prereg_p5a3_final.md`（先冻结后执行，未改动）
- **种子**: **80–99 一次性消耗**（20 seeds × 3 episodes = 60 episodes × 7 臂 = 420 记录）
- **前置**: `--preflight` PASS（n_api=1 / n_hit=1，网络调用与缓存命中分离核验通过）
- **完整性**: resumed_episodes=0（单次干净跑完）；420 行全在、seeds 80–99 每臂全覆盖（本目录
  records.jsonl 复核）；成本全披露 = **757 次真实 API 调用 / 0 cache hits / 787 次 composer
  调用 / LLM 墙钟 4615.7s（~77 min）/ deepseek-chat**，预算上限 1500 未触及。

## 一、预注册判据（headline，一次性）

| 判据 | 结果 | 数值 |
|---|---|---|
| ① utility_vs_det（主臂 ≥ det + ε=0.02，CI90 下界>0） | **FAIL** | diff = **−0.1768**，CI90 [−0.2608, −0.0974]（grouped by seed，B=2000）——**反向显著** |
| ② worst_group（每 preset LCB ≥ −δ_safe） | **FAIL** | 四 preset LCB 全负：G_hi_full −0.275 / G_hi_miss −0.356 / G_lo_full −0.167 / **G_lo_miss −0.549** |
| ③ novel_effective_distinct（SHA-distinct ≥ K=3） | **PASS** | **4** 个 distinct、menu 不可表达、true Δ≥ε 的程序 |
| ④ cost_disclosed | PASS | 上表 |

**→ claim_branch =`self_updating_deterministic_with_llm_novelty_supplier`（预注册 B 分支）。**

这是终审：P5-A（−0.799）→ P5-A.2 修接口（−0.150）→ **P5-A.3 合规按构造保证 + packet v2
连续证据 + 真实 Monash（−0.177，CI 不跨 0）**。接口层修完、证据层喂足之后，LLM composer
仍稳定输给确定性阶梯。**"LLM-driven harness evolution" 在本框架的 serving 与 proposing 两席
均未确立**；确立的是 B 分支：自更新确定性 harness + LLM 作为新颖性供给者之一。

## 二、四指标分解（本 trial 的架构目的：把 −0.177 归到哪一层）

各臂（60 eps/臂；mean true Δ 相对 raw，越大越好；pool_div = 每 episode 有效候选中 distinct
效果数均值，B=3）：

| 臂 | mean Δ | valid 率 | regret | pool_div | novel-eff SHA |
|---|---|---|---|---|---|
| frozen（弃权锚） | 0.000 | — | — | — | — |
| **random_valid** | **1.306** | 1.000 | 0.093 | **2.73** | **29** |
| **det_search（基线）** | **1.241** | 1.000 | 0.073 | 2.17 | 1 |
| plan_compiler_no_skills | 1.130 | 0.861 | 0.053 | 1.70 | 3 |
| plan_compiler_no_ce | 1.122 | 0.872 | 0.132 | 1.95 | 6 |
| pv2_direct | 1.095 | 0.961 | 0.003 | **1.22** | 8 |
| **pv2_plan_compiler（主臂）** | 1.064 | 0.889 | 0.062 | 1.70 | 4 |

1. **Compliance —— 已解决，不再是瓶颈。** 主臂 invalid 仅 20/180（13 = LLM 调用异常
   ITT 计为 no-op；7 = guard 在当前 fingerprint 上不满足）；**编译器语义丢弃率 = 0**（凡
   解析成功的 plan，任务契约全合规，guard/step 零丢弃）——A.2 的格式塌缩（180/180
   malformed）彻底消失。注意：stub 测试的 valid=1.0 是"可解析 plan ⇒ 必编译合规"的构造
   性质；真实 LLM 的 0.889 全部来自未达编译器的上游失败，非编译器缺口。
2. **Selection —— 不是败因；pv2_direct 的 regret≈0.003 是池塌缩伪影。** 47/60 episode
   里 direct 臂三个候选完全同效（distinct=1）——没有池就没有选择问题。L1 低熵输出在
   proposer 席第三次现身（前两次：slice v1/v2 serving、A.2 skill 模板化 7×/5× 重复）。
3. **Semantic/generation —— 真正的承重败因（本 trial 最锋利的结果）。** supply ceiling
   = 每 episode 取该臂候选池 **oracle 最优**（离线真判官）再与 det 实选比：主臂
   **−0.114 CI90[−0.206, −0.027]**、direct **−0.143 CI90[−0.239, −0.040]**——**即使选择
   开天眼，LLM 的候选池本身仍显著低于确定性阶梯**。差距在生成分布，不在选择、不在合规。

## 三、机制发现（探索性标注，非预注册对比）

- **负向锚定第三、四次复现（双消融）**：no_skills − 主臂 +0.066 CI90[−0.025,+0.162]、
  no_ce − 主臂 +0.057 CI90[−0.030,+0.144]（各自 n.s. 但同向），池上限排序
  no_ce(+0.012) > no_skills(−0.059) > 主臂(−0.114)。与 A.2 臂序诊断跨底座一致：**注入的
  知识/证据买到合规（guard 不满足 7 vs 消融臂 17/19），代价是语义池质量**——证据教会
  LLM 写"能通过的 guard"，同时把提案分布锚进更保守/更差的区域。
- **random 与 LLM 的瓶颈精确对偶**：random 池上限 **+0.158 CI90[+0.055,+0.262]** 高于
  det 实选（38/60 episode 池内含 ≥det 的程序），但 regret 0.093 吃掉大半，实现值
  +0.065 CI90[−0.064,+0.194] n.s.——**random 的瓶颈是选择，LLM 的瓶颈是生成**。（旧线
  check13/14 "瓶颈在选择不在生成"适用于 random/枚举供给，不适用于 LLM 供给。）
- **random − 主臂 = +0.242 CI90[+0.116,+0.368]**：零 API 成本的文法均匀采样在真实数据上
  **显著**胜过全信息 LLM composer。
- 主臂 vs det 逐 episode：胜 6 / 平 28 / 负 26；主臂 60 次从未选中 det 同 SHA 程序
  （平局 = 不同 spec、同执行效果）。

## 四、判据③的诚实口径（供给存在，但不优于 random；卡片模仿双口径核验）

主臂 4 个 novel-effective distinct SHA（10 个 episode）。**卡片模仿核验（post-hoc，
seed bank sha 对账）**：其中 `bf0b78433a16` **= 种子程序 `seed_period_stl`**（skill card
里展示过的程序，1 次有效 +1.224）——预注册口径（novel-vs-**menu**，is_novel_v1）4 ≥ K=3
通过不变；**严判口径（novel-vs-所见，剔除卡片种子）= 3 = K 恰好过线**。两口径均 PASS，
但边际之窄必须入档。主臂 60 次实选中与 seed bank 重合仅 1 次 → A.3 的负向锚定**不是
字面抄卡**（对比 A.2 的模板化模仿嫌疑），是分布收窄。

正面细节：`fe670ade089e` **非种子、非 random novel 集成员**，且**跨三个不同输出接口臂
独立收敛**（pv2_direct 实选 5× / 主臂 13× / no_ce 4×；plan 接口与直出 ProgramSpec 接口
的 prompt 完全不同）——这是 LLM 真实的自有构图偏好，跨 3 个 series family × 全部 4
preset 有效 7 次（fred_md G_lo_full 高达 +5.92）。`2b94e5c8867c` 同样跨臂出现
（no_skills 2× + 主臂 1×）。**LLM 供给真实存在**。但对照：random_valid 29 个 distinct
novel-effective（其中仅 1 个与 seed bank 偶合）。**LLM 供给的边际价值未证优于随机文法
采样**；其未测试的辩护位仅剩零标签冷启动 / 文本元数据语义（部署侧空白，见 LLM 定位判决）。

## 五、结论与后续绑定

1. **论文 headline（预注册 B 分支）**：self-updating deterministic harness；LLM 是
   novelty supplier 之一，不是 driver。P0–P5 的 claim ladder 不变，"LLM-driven" 一词
   从主张中移除。
2. **P6 架构绑定**：suppliers = {det 阶梯, **random 文法采样器（必列，当前最强供给）**,
   LLM} → 统一 Plan/EditOp → compiler+gate → batch 验证 → promotion。P6 的发现引擎若不
   包含 random supplier 基线即不成立。
3. **LLM 必要性（STAGE1_VERDICT 遗留问）在 level-2 真实域的答案**：serving 席被证伪
   （P5-A/A.2），proposing 席被证伪（本 trial）；仅剩冷启动/文本语义位未审。
4. **选择改进方向**（对 random 供给）：regret 0.093 主要由 P3 已判 FAIL 的 forecast proxy
   造成——改进候选选择器（如 B+ 式 featurized 证据消费）预计比改进生成器更值钱。

## 六、局限与留痕

- 单模型（deepseek-chat）、60 episodes、次级对比全部探索性标注。
- **日志缺口**：records 只存 chosen SHA 不存 spec dict → `fe670ade089e` 是什么程序当前
  不可直读；可用缓存重放（0 API、不碰种子语义）补录 spec，P6 前应把候选级 spec 持久化。
- 30 个 `itt_noop:llm_error:RuntimeError` 未区分解析失败 vs 网络异常（均 ITT 计 no-op，
  方向上只对 LLM 臂保守）；下版 runner 应分类落账。
- 平局 28/60 提示 SHA 过度区分（不同 guard/β、同执行效果）——效果等价类是更合理的
  novelty 单位，P6 计数口径待定。
- 安全闭环：旧泄漏 key 已由用户在平台侧轮换作废；本 run 全程 env-var key（外评①闭环）。
