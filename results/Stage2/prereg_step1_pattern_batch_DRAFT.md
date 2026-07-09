# 预注册草案 — Step 1：Pattern-Batch 联合实验（≡ P1b 复制转正）

> **状态：DRAFT，待用户早上拍板后锁定。** 本文件实现 Component Plan §13.1（Track A）。
> 一次两判据：判据族① = P1b 相对 P0 的**复制式**转正；判据族② = batch 键机制。
> 锁定前不消费任何一次性资源；锁定后 = confirmatory（不做结果后 prompt/阈值调参）。
>
> **本草案不预决用户的三处拍板点（见 §7）；其余为按 §13.1 + 项目先例的建议缺省。**

## 0. 动机与边界

- **为什么现在**：Bplus_v2（P0 + robust 周期诊断 4 + 分解 3 + gap 2 = 17 维 featurized DataView →
  per-uid GBDT）在**发现集** SkillSliceV2 上 cum 0.2976 大胜 frozen 0.3677（G2 CI[−0.109,−0.040]）。
  但 fixpc 教训在案：**单轮 dev 胜利 ≠ 复制**。转正须在**新语料**复制。
- **边界**：
  - 语料 = 新 permutation namespace 的 **dev** 语料（**不碰 holdout**；holdout 仍未 materialize）。
  - 结构库已冻结存在（`s2_corpus.py` "结构库 v3"：8 族 + miss-topology 网格 `S2_DEG_GRID`），
    **本实验不改结构库**——只换 split/permutation namespace 使其与发现集不相交。
  - 族真标签仅用于 leave-one-family-out（LOFO）分组与评估，**禁入任何 batch 键 / 策略输入**。
  - batch 语义 = "**处理响应相似区域**"（非 family purity）；本实验用 **oracle 族批**判 batch 键机制，
    可部署 batch builder（软聚类/holding pool）延后 Track D。

## 1. 语料（新 replication namespace）

- 生成器：`s2_corpus.make_series`（bit 级复用结构库 v3）。
- **新 namespace**（用户第三十七轮拍板 A，已定）：exploratory 结构扫描用 namespace
  **`S2R1_scan_20260707`**；**confirmatory 另用一个预先保留、暂不生成的 namespace**（早上锁 prereg
  时命名，与 scan namespace 及发现集三方不相交）。dev split 用此 namespace 种子，使 uid 集与
  `S2_replication/records_s2.jsonl` 发现集**不相交**（记录 SHA + namespace 字符串 + uid 不相交证明入 report）。
- 保留 DEV_FRAC=0.7 的 dev/holdout 切分逻辑；holdout 段**不生成**（首次读取须写
  `holdout_access_log.jsonl`——本实验不触发）。

## 2. 三 batch 键臂（判据族②）

同一 dev 语料、同一 oracle 族批评估口径下，比较三种"把序列分组成 batch"的键：

| 臂 | batch 键 | 说明 |
|---|---|---|
| **K-legacy** | 四格 cell（task×snr-bin×miss-bin） | 现任 E-3.2 分组，基线 |
| **K-P0** | P0 10 维特征上的连续局部聚类 | 连续键但旧特征（对 S_both 盲） |
| **K-P1b** | P1b 17 维特征上的连续局部聚类 | 候选键（robust 周期 + 分解 + gap） |

- **判据族②指标**（batch 内一致性 / 迁移；**非 family purity**）：
  1. batch 内 top-k Skill 一致率（同 batch 内 oracle-best 动作的集中度）；
  2. batch 内 action-response 方差（同 batch 内各 uid 对同一动作的 L 方差，越低越"响应相似"）；
  3. 同一 Harness edit（risk 规则）在 batch 内的**正迁移率**；
  4. 对其他 batch 的**负迁移率**（scope 泄漏度）。
- **胜出 batch 键** = 供 Track C 检索单位与 Track B proposer 分组用（§13.1 产出）。

## 3. P1b 转正臂（判据族①）

- 对比：**P1b 策略**（P0-D + P1b-P 特征的 router/risk 组合）vs **P0-abstain**（现任 level-1 冠军）。
- **转正形态（非裸用；用户第三十七轮拍板 C：强制作用域）**：P1b 必须绑 **escalation/abstain
  作用域**——B+ 自带 harm pocket（S_regime +0.032，max 块 0.118 > δ_safe），全域裸用在 SkillSliceV2
  G1 已败；D_bplus_v2 已示范"G2 过 + 伤害限触发域"。→ 转正臂 = **P1b 覆盖仅在触发块生效，非触发
  块回退 P0**。**裸 P1b（全域覆盖）只作诊断上界臂——报告幅度天花板，无转正资格**。
- **实现挂钩**：P1b 覆盖经 B0 的 `RiskAwareRouterPolicy`/`compile_bundle` 落到 overlay 消费面
  （§13.4 焊接已建），使转正的知识真进部署路径（非被绕过的模板）。

## 4. 分层与统计（复制式，防泄漏）

- **LOFO**：leave-one-family-out——在 7 族上拟合/选择、在留出族上评估，逐族轮换。
- **配对**：P1b 与 P0 在**同 uid** 上配对求 regret 差（paired）。
- **CI**：full-refit group bootstrap（每 replicate 重采样 uid，按 series 身份分 fold 防泄漏，
  完整重跑选择+评估）——沿用 STAGE1 A-33c 口径。
- **报告器 panel**：≥2 个下游报告器同向（C6 制度化：判官≠报告器，防 H*=f(judge) 自证）。

## 5. 判据（四条化，沿用 STAGE1 门）

> **符号约定（避免 loss/gain 号陷阱，用户第三十七轮勘误）**：regret 是**损失**（越小越好）。
> 改善量定义为 **Δ_improve = regret(P0) − regret(P1b)**（>0 = P1b 更好）。所有门按 Δ_improve
> 的**正向**表述；等价地若用 Δ=regret(P1b)−regret(P0)（改善为负），则门 2 为"CI **上界** < 0"。

**判据族①（P1b 转正）四条全过方转正**：
1. 点估计 **Δ_improve > ε**（方向性；ε 沿用 dev 域配对标定值）；
2. **paired CI 下界 > 0**（Δ_improve 的 full-refit group bootstrap CI；等价 Δ 的 CI 上界 < 0）；
3. ≥2 报告器同向；
4. worst-group LCB 非劣于 P0-abstain（不引入新 harm pocket）。

**判据族②（batch 键）**：K-P1b 在指标 1–3 优于 K-legacy 与 K-P0、指标 4（负迁移）不劣。

## 6. 分支（结果后不调参）

- 判据族①**全过** → **P1b 转正**为 level-1 Pattern（绑 escalation 作用域）；成为 Track B/C 的特征键。
- 判据族①**部分过** → 按未过条目定位（如 Δ_improve CI 跨 0 = 幅度未达、worst-group 破 = 作用域
  需收紧），不转正、记债、**不调阈值重跑**。
- 判据族②：胜出键写入 §13.1 产出；若三键无区分度 → batch 语义在本语料不可分，回 oracle 族批。

## 7. 拍板点（用户第三十七轮已定；早上锁 confirmatory 前复核）

- **A. namespace ✅**：scan = `S2R1_scan_20260707`；confirmatory = 预留、暂不生成的独立 namespace
  （早上命名）。三方（scan / confirmatory / 发现集）不相交。
- **B. 两段式 ✅**：先跑**非门控 exploratory 结构扫描**（看 batch 键分离度 + P1b 方向）→ **停**
  → 早上人工审阅后再一次性锁 confirmatory。**绝不看结果后自动改方案继续**（用户强调）。
  confirmatory 的 seeds 另开一次。
- **C. 作用域强度 ✅**：转正**强制**绑 escalation/abstain；裸 P1b 仅诊断上界、无转正资格（见 §3）。

> 早上仅剩：命名 confirmatory namespace + 复核本 prereg → 锁定。exploratory scan 结果**不**改判据。

---
*依据：Component Plan §13.1；先例 STAGE1_VERDICT（A-33c full-refit CI / 四条门 / reporter panel）、
E-3.3（in-sample gate winner's curse）、SkillSliceV2（B+ 发现集胜 + harm pocket）。*
