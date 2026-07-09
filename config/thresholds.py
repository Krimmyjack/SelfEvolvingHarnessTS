"""config/thresholds.py — 集中管理 ε/τ/K/S 等待校准 & 已校准数值（plan.md 附录 B.1）。

原则：所有"魔法数字"在此声明，带 provenance（哪个实验校准的）。其它模块只 import，
不内联。✅=已定版/已校准；⏳=有建议值待校准；模块按 STATUS 字典自检。
"""
from __future__ import annotations

# ── 感知 / 索引语言 ────────────────────────────────────────────────────────
STRUCT_FEATS_DIM = 10            # ✅ 已定版（explore 附-A1）。须 == len(conditioning.key.STRUCT_FEAT_NAMES)

# ── Grounded ε 验证门（frozen-foundation + linear probe 底座）──────────────
EPS_NARROW = 0.030               # ✅ E2c 合成校准；**配对重标定确认对 frozen 真实数据也成立**（σ_Δ≈0.028）
EPS_WIDE = 0.060                 # ✅ E2c 合成校准：触发②(可改进性) 的宽 ε + DiD
# ε 配对重标定（run_calibrate_eps 修正版，2026-06-20）：ε 由**同-batch cur vs cand 的配对 Δ** 决定，
#   不是无配对 batch 方差（后者含横截面异质、在 accept 律里抵消 → 过估 ~3×，旧记的 0.25 作废）。
#   跨 cell 中位 σ_Δ(minimal−degraded)：frozen≈0.028（≈原 0.03，E2c 本就对）、chronos≈0.08。
#   仅趋势 cell snrHigh|miss 真高噪（σ_Δ 0.27~0.79）→ 待 per-cell ε / 更大 N_min。
EPS_NARROW_REAL_CHRONOS = 0.08   # ⏳ chronos 判官真实数据稳定 cell 的配对 ε（run_real_longrun --eps 用）
S_SEEDS = 2                      # ⏳ 2–3：每 batch seed 重复，压训练噪声 A

# ── proxy ↔ grounded 校准门 ────────────────────────────────────────────────
TAU_PROXY = 0.4                  # ✅ 已定版 E3/E3c：per-cell Spearman，低于此 proxy 不予采信

# ── 慢路径触发 / 候选 ──────────────────────────────────────────────────────
N_MIN = 16                       # ⏳ 16–32：batch 最小样本（下游训练稳定量）
MIN_SUPPORT = 3                  # ⏳ 结构流 failure_signature 最小支持数
K_CANDIDATES = 3                 # ⏳ 3–5：proposer 并行互异候选数（成本允许取大）
# S_SEEDS 是 substrate-aware：frozen forecast + anomaly 检测器是确定性(σ_A=0) → 实际 S=1；
# 仅随机 grounded（classify InceptionLite / forecast scratch 消融）才需 S≥2 压训练噪声。

# ── 慢路径进化控制（B.2 #4/#5/#6 定版 2026-06-19）──────────────────────────
N_FREEZE = 3                     # ✅ #4 连续全拒轮数 → cell 冻结（K=3 候选×3轮=9 拒，排除 LLM 采样方差）
FREEZE_RECHECK_EPOCHS = 5        # ✅ #4 冻结 ≥M epoch → 无条件强制 1 轮重评（安全网；无永久冻结）
EDIT_BUDGET_MAX = 3              # ✅ #6 max_lr(=K)：早期接受所有过门候选（探索）
EDIT_BUDGET_MIN = 1              # ✅ #6 min_lr：后期只接受 proposal_rank 最高（固化，防非可加交互）
EDIT_BUDGET_TOTAL_STEPS = 12     # ✅ #6 cosine 退火 horizon（每 cell 预期上限；冻结规则通常更早起效）
EDIT_BUDGET_MODE = "cosine"      # ✅ #6 SkillOpt 消融证优于 constant/linear

# ── ★v4 流式持续适应（S1：deploy_stream，Refactor_v4 §1 / S1_Implementation_Plan §B）──
DOMAIN_BUDGET_CEILING_DECAY = 0  # 进第 k 个 domain 时 edit_budget 上限 = max(MIN, MAX − k·DECAY)；
                                 #   meta-退火（记忆越成熟、需新编辑越少）。0 = 不衰减（K=4 短流默认；长流调 1）
READINESS_THRESHOLD = 0.8        # readiness=(J_raw−J_cur)/(J_raw−J_min) 越过此值 → 该 cell 就绪（前向迁移 time-to-readiness 用）

# ── 检索距离 ────────────────────────────────────────────────────────────────
ALPHA_DISTANCE = 0.5             # ⏳ d = α·d_struct + (1-α)·d_quality（Memory 积累后调）

# ── 软结构门：cell-scoped 模板复用（方向 A，AME-TS 软先验，conditioning.distance.d_struct）──
# 旧路径：模板按 pattern_bin(=SNR×missing 2 维)==精确匹配 → 结构迥异 cell（covid 尖刺 / tourism 月季）
# 塌进同 bin 被误套 = 负迁移。软门：模板携带创建时的 struct_ref（10 维中位），仅当当前 cell
# d_struct(struct_ref, feats) < τ 才复用——跨/同 bin 一致按全 10 维结构距离裁。
# d_struct = 归一化欧氏距离/√10；τ 须 > 同 cell 批间散度、< 异 cell 结构墙。
# 校准（run_calibrate_dstruct，K=4 真 Monash）：intra 同-cell p50=0.10/p95=1.26；inter 跨域 centroid:
#   fred_md 是结构离群（≥2.8 离所有域，任何 τ 都拦）；covid↔tourism 反而极近(0.33)、nn5 居中(1.0~1.36)。
#   ⚠ intra p95(1.26) ≥ inter min(0.33)：全 10 维编码也无法干净分离"同 cell 异 batch"与"近域"
#   （covid-tourism 比 ~30% 同 cell 对还近）——这量化坐实"pattern/特征设计欠佳"（不止 bin 粗）。
# τ=1.2 ≈ intra p95，低于 fred 结构墙(2.8)：拦掉 fred 类结构错配的负迁移，~95% 合法同-cell 复用仍过。
BIN_DSTRUCT_TAU = 1.2            # ⏳ run_calibrate_dstruct 校准；特征改进后（方向 B/C）应重标

# ── Gate 阈值（与 layers.GateConfig 默认一致；此处为校准锚）────────────────
BLOWUP_SIGMA = 10.0              # ⏳ 输出超 μ+Nσ → 判爆炸
VIOLATION_TOL = 0.05             # ⏳ 约束违反升高容忍（pp）

# ── 冻结分箱网格（conditioning.binning 用；Phase 0 minimal = SNR×missing×task = 12 cells）──
# 注：当前 SNR struct_feat（MA-11 估计）对周期信号偏低，实测可达范围 ~[1,5.4]dB（非 0–60）。
# 4.0 落在该范围内，把低噪(noise≲0.2,SNR≳4.6)与高噪(noise≳0.5,SNR≲3.6)分开。SNR 估计本身偏弱，待改进。
BIN_SNR_SPLIT_DB = 4.0           # ⏳ SNR(dB) < split → "low" 否则 "high"（2 档）
BIN_MISSING_ANY = 0.0            # ⏳ missing_rate > this → "miss" 否则 "full"（2 档）

# ── quality_profile 探测阈值（conditioning.key 用）────────────────────────
OUTLIER_MAD_K = 3.5              # ⏳ 稳健离群判定：|x-median| > k·MAD
SNR_DB_NOISY = 10.0             # ⏳ SNR(dB) 低于此判 has_noise [待按数据分布校准]
TREND_STRENGTH_DRIFT = 0.3       # ⏳ trend_strength 高于此 → has_drift（配合 ADF）
ADF_NONSTATIONARY_P = 0.05       # ⏳ ADF p 高于此判非平稳（has_drift 辅证）

# 自检：阈值状态登记（便于"动工前必拍板"清单核对）
STATUS = {
    "calibrated": ["STRUCT_FEATS_DIM", "EPS_NARROW", "EPS_WIDE", "TAU_PROXY"],
    "decided": ["N_FREEZE", "FREEZE_RECHECK_EPOCHS", "EDIT_BUDGET_MAX", "EDIT_BUDGET_MIN",
                "EDIT_BUDGET_TOTAL_STEPS", "EDIT_BUDGET_MODE"],   # B.2 #4/#5/#6 设计定版
    "provisional": ["S_SEEDS", "N_MIN", "MIN_SUPPORT", "K_CANDIDATES", "ALPHA_DISTANCE",
                    "BIN_DSTRUCT_TAU", "BLOWUP_SIGMA", "VIOLATION_TOL", "BIN_SNR_SPLIT_DB",
                    "BIN_MISSING_ANY", "OUTLIER_MAD_K", "SNR_DB_NOISY", "TREND_STRENGTH_DRIFT",
                    "ADF_NONSTATIONARY_P"],
}
