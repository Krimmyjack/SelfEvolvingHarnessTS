"""stage2_protocol.py — Stage 2.0-⑥ 新 namespace 注册表 + 张量协议落盘（Component Plan v1.1d）。

当前构建 **v2**（`results/Stage2/tensor_protocol_v2.json`；一次性落盘，存在即拒绝覆盖）。
版本链：v0（config_sha=ff79883f196200c3）→ v1（4cf04acb8d46299c）→ v2；旧版本**原样保留**，
取代关系由 supersedes 字段声明。

v1 → v2 关闭的两个洞（评审第二十四轮）：
  1. **DLinear 训练单位写反了**：v1 声称"per-series 独立 from-scratch，与 STAGE1 reporter
     口径一致"——两处都错。report_target.py 实际是**域内汇总构窗 → 训练一个共享 DLinear →
     逐序列评估**（build_windows(batch) → 单模型）。v2 更正：主口径=域内共享训练（与项目
     "用处理好的数据训练下游模型"的原始设定一致），per-series 独立训练降为廉价诊断臂；
  2. **freeze_gate 自相矛盾**：v1 一边宣布协议不可变，一边要求 draft 族冻结时"更新本文件
     status 并重算 sha"→ v2 改为**版本化冻结流程**：本文件永不改动，结构库冻结=另存 v3；
     张量语料生成只接受 structure_library 全冻结的协议版本（v1/v2 永久不合格）。
另：pattern_spec_ref 的 code_sha256 换为提取器代码闭包（key.py+period.py）；measurement
补 estimand 与 utility/report 关系声明。

v0 → v1 关闭的四个洞（评审第二十三轮，沿袭）：holdout 解锁独立 access log/动作三分归属/
（DLinear 表述——v2 再更正）/dominance worst-group 安全侧；menu SHA 绑 resolved 语义。

其余边界沿 v0：只落盘协议、不生成语料、不读任何新 holdout；draft 结构族生成前须显式冻结；
分支规则数值预注册。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.stage2_protocol
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
OUT_DIR = PKG / "results" / "Stage2"
V0_PATH = OUT_DIR / "tensor_protocol.json"
V1_PATH = OUT_DIR / "tensor_protocol_v1.json"
PROTOCOL_PATH = OUT_DIR / "tensor_protocol_v2.json"
V3_PATH = OUT_DIR / "tensor_protocol_v3.json"
ACCESS_LOG = OUT_DIR / "holdout_access_log.jsonl"

EPS = 0.03                       # 与 dev/confirmatory 同一实用阈值
DELTA_SAFE = 0.05                # worst-group 安全阈（与 Stage-1 δ_safe 同值）
DOMINANCE_COVER = 0.90           # 支配判：单 (model, 轻动作) 对在 ≥90% series 上 ε-最优
INTERACTION_SHARE_MIN = 0.15     # 交互判：交互项方差占比 ≥15% → 激活 L5 联合选择


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def build_protocol() -> dict:
    from .policy import action_menu_v1
    menu = action_menu_v1()

    proto = dict(
        name="stage2_tensor_protocol",
        version="v2",
        date="2026-07-05",
        supersedes="tensor_protocol_v1.json（v1，config_sha=4cf04acb8d46299c；原样保留。v1 的 "
                   "DLinear 训练单位表述有误——声称 per-series 独立且与 STAGE1 reporter 一致，"
                   "实际 reporter 为域内汇总构窗训练共享模型；v2 更正并调整冻结流程为版本化）；"
                   "链上更早：tensor_protocol.json（v0，config_sha=ff79883f196200c3）",
        authority="Component_Optimization_and_Integration_Plan.md v1.1d §2.0-⑥/§5",
        status="frozen_except_structure_library_draft",
        protocol_versioning=dict(
            immutability="本文件落盘后永不改动（不改 status、不追加字段、不重算 sha）",
            freeze_process="draft 结构族参数化落定后**另存新版本协议文件**（v3，"
                           "structure_library.status=frozen_full），本文件永久保持 draft 状态",
            tensor_gate="张量语料生成与三模型效用张量**只接受 structure_library 全冻结的协议版本**"
                        "——v1/v2 永久不合格；据 v2 生成语料/张量 = 协议违规"),
        one_shot_rules=[
            "本协议文件**不可变**：任何改动=另存新版本文件（v3…），不原地编辑、不追加字段",
            "Stage-2 holdout 一次性：首次读取前须向独立 append-only 日志"
            " results/Stage2/holdout_access_log.jsonl 追加一条 {date, actor, reason, protocol_config_sha}"
            "——解锁记录与协议分离（v0 的协议内追加方案与不可变规则矛盾，已废）",
            "seeds 20–39（Stage-1 confirmatory）永不复用",
        ],
        holdout_access=dict(
            log_file="results/Stage2/holdout_access_log.jsonl",
            format='{"date": ..., "actor": ..., "reason": ..., "protocol_config_sha": ...}',
            rule="append-only；空/缺失 = holdout 从未被读；任何读取代码以'日志已有对应记录'为前置断言"),

        # ── namespace 注册表（生成前 SHA 锁定；此处锁定生成规则，语料生成另行执行）──
        namespace=dict(
            scheme="uid = 'S2:{family}:{dname}:{j}'；sd = sha256(uid) % 2_000_000（确定性，免全局 seed）",
            split=dict(
                method="series_uid 分组 + family×domain×cell 分层（F2 纪律：同 uid 不跨 split）",
                dev_frac=0.7, holdout_frac=0.3,
                holdout_rule="holdout 段生成后即封存；读取须显式解锁记录（见 one_shot_rules）"),
            regression_anchor="Stage-1 冻结语料（E-3.2 dev + A31e）保留为回归锚，不混入 Stage-2 分布",
            j_range=[0, 19],
            note="j 与 Stage-1 的 {struct}:{dname}:{j} namespace 无交集（前缀 S2: 隔离）"),

        # ── 结构库 v2：4 冻结族 + 4 draft 提案族（D9：词汇量症状）──
        structure_library_v2=dict(
            frozen_families=["S_trend", "S_season", "S_both", "S_ar"],
            draft_families=dict(
                S_intermittent="间歇/稀疏需求（零膨胀 + 突发脉冲）——覆盖 lumpiness/outlier 轴",
                S_hetero="异方差（GARCH 型波动聚簇）——覆盖 lumpiness 轴、检验剂量对波动结构的破坏",
                S_regime="regime-switching/changepoint（分段均值/斜率）——覆盖非平稳轴",
                S_multiseason="多周期叠加（如 24×168）——直击 D1 周期估计与 S_both 混叠"),
            freeze_gate="draft 族参数化（生成器方程+参数网格）落定后，**另存新版本协议文件**"
                        "（v3，structure_library.status=frozen_full，其 config_sha 独立计算），"
                        "方可生成语料——本文件（v2）永不改动（v1 的'更新本文件并重算 sha'与"
                        "不可变规则矛盾，已废）",
            real_domains=dict(source="monash_clean.npz（83 信号/7 域，HF 直连）",
                              candidates=["nn5_daily", "tourism_monthly", "fred_md",
                                          "covid_deaths", "us_births"],
                              pick="3–5 域，按 series 长度≥MIN_LEN 与域异质性定，生成前冻结")),

        # ── 张量三轴 ──
        tensor_axes=dict(
            series="Stage-2 dev（合成 v2 + 真实 dev 域）",
            action=dict(
                menu_version=menu.version, menu_sha256=menu.sha256, n_actions=len(menu),
                menu_semantics="menu SHA 绑定完整 resolved params（defaults ⊕ override，"
                               "Step 1.1-②）+ meta.operator_defaults_sha",
                source="policy.action_menu_v1()（单一真源：_VARIANT_SPECS+F0_DOSAGE_GRID）",
                roles=dict(
                    core_pool=["v_none", "v_median", "v_savgol", "v_stl", "v_wavelet",
                               "v_winsor", "v_winsor_savgol",
                               "f0_median_w9", "f0_median_w15", "f0_median_w25"],
                    ablation=["f0_ma_w9", "f0_ma_w15", "f0_ma_w25"],
                    dosage_diagnostic=dict(
                        actions=["f0_savgol_w21", "f0_savgol_w31"],
                        note="savgol 剂量维只测不进主比较（F0 已证 w31 季节端点多项式过冲、"
                             "跨 cell 最差自拒）；保留为'过量平滑何时开始伤害'的剂量质量信号")),
                coverage_check="roles 三集合并 = menu 全集（无归属不清动作；tests 守）"),
            model=dict(
                pilot=[
                    dict(id="seasonal_naive", train="无（解析基线）", budget="0"),
                    dict(id="dlinear_scratch",
                         train="**within-domain pooled（域内汇总）from-scratch**：一个 dataset/"
                               "domain 的处理后 train 段全部序列汇总构窗 → 训练**一个共享 "
                               "DLinear** → 在 held-out series/window 上逐序列评估。= STAGE1 "
                               "reporter report_target.py 实际口径（build_windows(batch)→单模型"
                               "→逐序列 nRMSE），也= 项目原始设定'用处理好的数据训练下游模型'。"
                               "【v1 勘误】v1 声称 per-series 独立且与 reporter 一致——两处皆误",
                         budget="epochs=120, seeds S=5（对齐 confirmatory reporter 口径）",
                         coupling_note="series 间经共享模型耦合：某 series 的处理结果影响同域"
                                       "其他 series 的槽值 → 张量 series 轴按'域内联合处理配置'"
                                       "解读，域为独立性单位"),
                    dict(id="chronos_bolt_small", train="zero-shot（无训练）", budget="0",
                         note="确定性 σ_A=0"),
                ],
                gated=dict(id="lstm_scratch",
                           gate="仅当 3 模型 pilot 显示交互份额 ≥ interaction_share_min 才全量训练（最贵菜单项）"),
                diagnostic_only=[
                    "frozen_probe（降诊断列，不进主比较）",
                    "dlinear_per_series（per-series 独立 from-scratch——估计'单序列可学性'效用，"
                    "≠主 estimand；廉价诊断臂，预算另批，不进主比较）"]),
        ),

        # ── 度量与归一 ──
        measurement=dict(
            task_scope="forecast（张量 pilot v0；classify/anomaly 轴不进 v0）",
            estimand="U(域, action, model) = 下游模型在该域**处理后 train 数据**上训练、"
                     "在 held-out series/window 上的 perf——张量测的是'任务条件化数据就绪对"
                     "下游训练效用的影响'，非'单序列可学性'（后者=dlinear_per_series 诊断臂）",
            target_normalization="per-series z-score（参数取自该 series train 段并记录；与 load_real 口径一致）",
            per_model_metric="per-series nRMSE（report_target.py 冻结口径：共享模型逐序列评估）；"
                             "聚合 perf=exp(−mean nRMSE)",
            utility_vs_report="张量=效用测量（判官侧证据，供 Router/L5 训练与分支判读）；"
                              "Stage-2 headline ΔPerf 报告仍须判官↔报告器分离——张量所用模型"
                              "进入某比较的优化环路时，不得同时充当该比较的独立报告器",
            judge_reporter_separation="张量标签生成者 ⟂ 任何用于判读的报告器（disjoint_targets 强制）"),

        # ── 失败/回退（无静默截断）──
        failure_policy=dict(
            non_finite="任何 series 级非有限值 → 显式报错记录（A-41⑥守卫⑤ 风格，无静默回退）",
            retry="per-slot 独立种子重试 ≤2；仍失败 → 该 (series,action,model) 槽记 missing",
            missing_handling="missing 槽从所有对比臂同时剔除并报告计数（no silent caps）",
            runaway="单模型训练墙钟上限 30min/series-batch；超限 kill + checkpoint/resume（A-36 教训）"),

        # ── 分支规则（生成前锁定，2.3 判读依此，不得事后调）──
        branch_rules=dict(
            eps=EPS, delta_safe=DELTA_SAFE,
            interaction=dict(
                stat="two-way 分解 loss ~ action + model + action×model（per-series 标准化后）",
                rule=f"交互项方差份额 ≥ {INTERACTION_SHARE_MIN} → 交互丰富 → 激活 L5 联合 (a,m) 选择"),
            dominance=dict(
                rule=f"存在单一 (model, 轻动作) 对，在 ≥ {DOMINANCE_COVER:.0%} series 上处于 ε-最优集"
                     f"（loss ≤ oracle_pair + {EPS}）",
                worst_group_safety=f"**且** 每 family×cell 子群上该对 Δ vs oracle_pair 的 LCB > "
                                   f"−{DELTA_SAFE}（尾部 10% 不得被结构性伤害——F0 season 教训：聚合"
                                   "支配可掩盖子群 harm）→ 两条都满足才收缩为'安全约束+预算描述'设计"),
            middle="两者皆不满足 → sequential（先 action 后 model）设计",
            comparison_ladder=["global_pair", "action_only", "model_only",
                               "sequential", "joint", "oracle_pair"]),

        # ── runtime pinning（Step 1.1-④；v2 补：核验/放行结果落每条决策 provenance）──
        runtime_pinning=dict(
            rule="任何冻结 artifact（frozen_arms/未来 PolicyArtifact）加载时比对 blob 内记录的 "
                 "sklearn/numpy 版本与运行时，不匹配 fail-loud（显式 allow_version_mismatch 放行须记录）",
            impl="policy.router_policy.FrozenArmRouterPolicy._check_runtime_versions；核验结果"
                 "（recorded/runtime/mismatch/allowed_mismatch）落 RoutingDecision.provenance.runtime_check"),

        # ── 缓存与 provenance ──
        caching=dict(
            cache_key="sha256(uid | action_id | action_menu_sha | model_id | train_config_sha)",
            location="results/Stage2/tensor_cache/",
            provenance_per_cell=["code_fingerprint（confirmatory_freeze._FINGERPRINT_FILES 同构清单）",
                                 "依赖版本", "pattern_spec {version, config_sha}",
                                 "生成时间与 checkpoint 链"]),

        pattern_spec_ref=None,       # 填充见下（P0 语义身份，锚定部署/训练特征同源）
        config_sha=None)

    from .policy import pattern_spec_p0
    p0 = pattern_spec_p0()
    proto["pattern_spec_ref"] = {
        "version": p0.version, "config_sha": p0.config_sha(),
        "code_sha256": p0.code_sha256,
        "code_sha_scope": "提取器代码闭包 conditioning/{key.py, period.py}（落盘时快照，审计用；"
                          "语义身份以 config_sha 为准，张量生成时按 caching.provenance 记活值）"}
    proto["config_sha"] = _sha_obj({k: v for k, v in proto.items() if k != "config_sha"})
    return proto


def build_protocol_v3() -> dict:
    """v3 = v2 逐字副本之上冻结 structure_library（评审第二十七轮指令：落定即生成，不再评审措辞）。
    参数单一真源 = s2_corpus.py 模块常量（本函数直接读入，防协议与生成器漂移）。"""
    from . import s2_corpus as sc
    proto = json.loads(PROTOCOL_PATH.read_text("utf-8"))          # v2 逐字为底
    proto["version"] = "v3"
    proto["date"] = "2026-07-05"
    proto["supersedes"] = ("tensor_protocol_v2.json（v2，config_sha=" + proto["config_sha"]
                           + "；原样保留）；链上更早 v1=4cf04acb8d46299c / v0=ff79883f196200c3")
    proto["status"] = "frozen_full"
    proto["protocol_versioning"] = dict(
        immutability="本文件落盘后永不改动",
        freeze_process="structure_library 已全冻结（本版本即冻结动作的产物）；后续任何改动=另存 v4",
        tensor_gate="张量语料生成与三模型效用张量自 v3 起合格（structure_library=frozen_full）；"
                    "v0–v2 永久不合格")
    dev, hold = sc.s2_split()
    proto["structure_library_v3"] = dict(
        source_of_truth="SelfEvolvingHarnessTS/s2_corpus.py（生成方程+参数网格单一真源；"
                        "协议记录其冻结快照，漂移由 tests/test_s2_corpus.py + audit 守）",
        frozen_families=list(sc.S2_FAMILIES),
        family_grid={k: [dict(p) for p in v] for k, v in sc.S2_FAMILY_GRID.items()},
        variant_rule="variant = sd % len(grid)（确定性）",
        scale_rule="除 S_intermittent（只除 std，保零膨胀）外 _unit（demean+unit std）；"
                   "S_regime 加 0.15 iid 纹理防常数段 SNR 退化",
        multiseason_period_note="周期对 (16,128)：整 bin + 公度比 8（非公度对在 L=512 下被 "
                                "robust_v1 单 lag ACF 确认误拒——已记为 P1b 估计器改进标的）",
        deg_grid={k: dict(v) for k, v in sc.S2_DEG_GRID.items()},
        miss_topology="第一类变异轴（prereg §2 兑现）：random/block/burst × rate {0,3%,6%,12%}；"
                      "缺失流独立 rng(sd+20000)，噪声/离群流 rng(sd+10000) 与 v1 网格同构",
        wave=dict(dev_j=list(sc.DEV_J), reserved_j=list(range(10, 20)),
                  n_dev=len(dev), n_holdout_reserved=len(hold)),
        split_rule="每 (family,dname) 层内 rng(_det_seed('S2split',family,dname)) 置换取前 70%→dev；"
                   "holdout 不物化（确定性可再生），首读须 holdout_access_log.jsonl",
        audit="results/Stage2/s2_corpus_audit.json")
    lib_old = proto.pop("structure_library_v2")
    proto["structure_library_v2_superseded"] = dict(
        note="v2 的 draft 提案（原文保留供审计）", draft_families=lib_old["draft_families"],
        real_domains_candidates=lib_old["real_domains"])
    proto["real_domains_frozen"] = dict(
        rule="预声明规则：monash_real.npz 中可用 ∧ series 长度≥MIN_LEN(144)",
        pick=["nn5_daily", "tourism_monthly", "fred_md"],
        detail={"nn5_daily": "n=4, len=791, period=7", "tourism_monthly": "n=4, len∈[187,264], period=12",
                "fred_md": "n=4, len=728, period=12"},
        unavailable={"covid_deaths": "不在 monash_real.npz", "us_births": "不在 monash_real.npz"},
        role="张量 series 轴的真实域段（复制表主 scope=合成 S2 dev，真实域为张量/level-2 附录）")
    proto["config_sha"] = _sha_obj({k: v for k, v in proto.items() if k != "config_sha"})
    return proto


def main():
    if V3_PATH.exists():
        raise SystemExit(f"{V3_PATH} 已存在；协议不覆盖——改动请另存新版本文件并记录理由。")
    if not PROTOCOL_PATH.exists():
        raise SystemExit("v2 缺失——v3 以 v2 为底构建，链不可跳。")
    proto = build_protocol_v3()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    V3_PATH.write_text(json.dumps(proto, ensure_ascii=False, indent=1), "utf-8")
    print(f"tensor_protocol_v3.json 落盘  config_sha={proto['config_sha']}  status=frozen_full", flush=True)
    print(f"  families={len(proto['structure_library_v3']['frozen_families'])} "
          f"deg_grid={len(proto['structure_library_v3']['deg_grid'])} "
          f"n_dev={proto['structure_library_v3']['wave']['n_dev']}", flush=True)
    print("  张量生成自 v3 起合格；holdout 未解锁。", flush=True)


if __name__ == "__main__":
    sys.exit(main())
