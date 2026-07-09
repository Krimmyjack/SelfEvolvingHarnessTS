# 预注册：S2 复制 + Router 第一轮（2026-07-05，S2 语料生成前落字）

> 性质：评审第二十六轮双评审共同要求的预注册（Reviewer 1 两项 + Reviewer 2 复制集与解释收紧），
> 非新协议版本（tensor_protocol 链不动）。目的=把 P1a follow-up（fixpc）的乐观性关进笼子、
> 防止 S2 结果出来后临时加组合。本文件落盘后不改；变更=追加"修正案"小节并注明理由。

## 1. S2 dev 复制臂（锁定，S2 结果出来后**不得新增组合**）

| 臂 | 用途 |
|---|---|
| P0 | 对照锚 |
| fixpc（P0-D + P1a-P + C） | **主候选**（P1a 第一张表的 follow-up，在此转正或出局） |
| D 分解三臂：P0-SNR+missing / P1a-SNR+missing / P1a-SNR+missing+gap | 归因收紧（Reviewer 2）：fixp→pd 同时换了 SNR 估计与加 gap 拓扑，dev 表只能支持"P1a-D 整体无效/有害"，gap 单独責任在 S2 上拆 |
| Router-1 胜者（形态+特征） | 第二张表的胜出设计 |

**转正判据（fixpc→正式 P1）**：S2 dev 上 ΔRegret vs P0 的 paired CI 不跨 0，**且** worst-group
LCB 不劣于 P0-abstain。失败 → 回退最简单可复制的组合，不做第三轮特征finetuning。

## 2. S2 生成器硬需求（并行线规格，语料生成前必须满足）

**miss-topology 必须是第一类变异轴**：块状缺失（block）、burst（连续短簇）、rate 梯度
（2%–15%）至少三档，且与 SNR 轴正交采样。理由：dev 语料缺失=6% 均匀随机 → gap 拓扑特征
是常数/噪声维（P1a 第一张表发现 2），不加此轴则 gap 特征在 S2 上第二次白测。

## 3. Router 第一轮（本轮，dev 冻结折重放；设计级，发现集乐观性 caveat 同 P1a）

**特征集**：fixpc（P0-D 2 + P1a-P 9 + C 3）。φ(P,D,a) **现算不入 spec**（v1.1d）：
`{family one-hot(7), w/25, w÷period（period>0 且 w>0 否则 0）, smoothable_energy(w)=去趋势插值
谱中 f>1/w 的能量占比（w≥2 否则 0）}`；w/family 取自 action_menu_v1 resolved params 单一真源。

**臂**（冻结折/L_train/L_test 同 P1a；对照臂须逐位复现 P1a fixpc 数字=管线守卫）：
1. `pa_gbdt` per-action GBDT（对照 = P1a fixpc dp_gbdt）
2. `pa_abstain_std` legacy ensemble-std κ=1（= P1a fixpc dp_abstain）
3. `pa_abstain_cgate` legacy 触发 ∧ 低 C 区（见下）
4. `sq` shared-Q(P,D,C,a)+φ：单 GBDT（**depth=3, n_estimators=200, subsample=0.7, E=5，
   预注册不调参**），行=(uid×action)，argmin 预测
5. `sq_rank` 同 4 但标签=uid 内 loss 秩（归一 [0,1]）——尺度免疫（D10）
6. `sq_abstain_std` κ=1
7. `sq_abstain_kcv` κ ∈ {0.5, 1, 1.5, 2} 由 outer-train 内 4 折 CV 选（最小 mean regret）
8. `sq_abstain_cgate` 同 3 于 sq

**C-gated 触发器（Reviewer 1 成员选择 + Reviewer 2 不设全局硬阈值）**：
abstain ⇔ legacy 触发（adv_mean < κ·adv_std, κ=1）**∧**（c_peak_sig < q25 ∨ c_acf_confirm < q25），
分位数**逐折取自 outer-train**（无泄漏）；**c_obs_coverage 不进触发器**（dev 上方向反转，
是 cell 代理）。C 同时作连续特征进 sq（不设阈值）。

**unseen dosage（留一剂量）**：训练集剔除 `f0_median_w15` 全部标签行；测试时 sq 经 φ 元数据
预测 w15 列（per-action 对照结构上不可能）。报：sq_low15 全动作 regret、w15 为 oracle 时的
选中率与 regret；pa_low15（9 动作菜单）作对照。

**判据（设计级）**：
- sq 采纳线：mean regret paired CI 不劣于 pa_gbdt ∧ worst-group LCB 不更差 ∧ unseen-dosage
  regret 明显优于 pa_low15；
- abstain 选择线：在均值不显著变差前提下 worst-group LCB 最优者胜；全部不改善 →
  Router-2 以 no-abstain 为 incumbent（P1a 发现 1：abstain 曾是特征缺陷补丁）；
- `family selector + 连续剂量回归` 臂本轮**不实现**（显式声明非静默砍：其剂量插值能力被
  sq 的 w 连续元数据覆盖，若 sq 失败再单独立臂）。

**多重比较声明**：8 臂同表，全部设计级；任何"胜者"只获得进入 S2 复制集第 4 行的资格。
