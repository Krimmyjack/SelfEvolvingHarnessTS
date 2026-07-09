"""confirmatory_freeze.py — A-40/A-41 confirmatory 冻结机器（评审第十七轮实现门禁，2026-07-04）。

两件事，且都必须发生在任何 seeds 20–39 读取与 A38C 生成**之前**：

  1) `train_final_arms()` + `serialize_arms()`：用 dev E-3.2 records（primary 口径 nested
     outer-test 标签 = 全语料 OOF，每 uid 恰被 outer-test 一次，A-39① 评估器产物）训练**最终**
     dp_abstain 候选 + 全部冻结对照臂（global/d_lookup/d_gbdt/p_gbdt/dp_gbdt/true_d_gbdt +
     oracle_struct 迁移诊断）→ joblib 序列化 `results/E3_2_confirmatory/frozen_arms.joblib`
     + SHA256。**locked transfer 冻结的是实际模型文件，不是"理论上相同"的训练配方**（A-41②）；
     confirmatory 主结果只 load、永不调 fit。

  2) `write_freeze()`：机器可读 `confirmatory_freeze.json`（A-41①）——动作池/P/D 特征/GBDT/
     κ/fallback/router SHA/holdout seeds/A38C 预锁带界（=A31e dev 实测值，不看 seeds 20–39
     分布，A-41③）/N 目标/**布尔判据表**（A-41④）/统计口径按 reporter 分型（A-41⑤）/
     代码指纹/依赖版本。`confirmatory_corpus._require_freeze` 以本文件存在为读 holdout 的
     构造性前提。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.confirmatory_freeze          # 训练+序列化+写 freeze 一步完成
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .e32_policy import (D_FEATS, DELTA_SAFE, EPS, FALLBACK_ACTION, KAPPA, N_ENSEMBLE, P_FEATS,
                         PRUNED_POOL_CORE, PolicyData)
from .run_e32 import GBDT_PARAMS, make_all_arms

PKG = Path(__file__).resolve().parent
E32 = PKG / "results" / "E3_2"
RESULTS_A31E = PKG / "results" / "A31e"
RESULTS_CONF = PKG / "results" / "E3_2_confirmatory"
FROZEN_ARMS = RESULTS_CONF / "frozen_arms.joblib"
FREEZE_PATH = RESULTS_CONF / "confirmatory_freeze.json"

DEFAULT_SEED = 20260704                 # 与 dev E-3.2 freeze 相同（router 序列化用；holdout 折另有 seed）
HOLDOUT_J = (20, 39)                    # confirmatory 基底 = 原 namespace j ∈ [20, 39]
N_TARGET_CONF = 40                      # A-40④：A38C 补齐每可行槽至 N=40（与 A-38 同构）
OOF_K = 5                               # locked-transfer 测量标签的 confirmatory 内 grouped OOF 折数
REPL_OUTER_K, REPL_INNER_K = 5, 4       # replication（次要 estimand）沿用 dev nested 配置
FULL_REFIT_B = 1000                     # 主判决 CI：grouped full-refit bootstrap（A-33c 机器）
PAIRED_BOOT_B = 2000                    # 次要参照：paired-uid bootstrap / reporter 配对 CI
REPORTER_SEEDS = (0, 1, 2, 3, 4)        # dlinear_scratch S=5 训练种子（A-40⑥）
REPORTER_PANEL = ("dlinear_scratch", "chronos")
REPORTER_POLICIES = ("dp_abstain", "global", "d_lookup")

# 冻结臂全集（A-41②：对照臂同样冻结，否则 "vs 对照" 的 estimand 不闭合）
FROZEN_ARM_NAMES = ("global", "d_lookup", "d_gbdt", "p_gbdt", "dp_gbdt", "dp_abstain",
                    "oracle_struct", "true_d_gbdt")
GATE_COMPARATORS = ("global", "d_lookup", "d_gbdt")            # 布尔判据表的三个门比较
REPORT_COMPARATORS = ("true_d_gbdt", "dp_gbdt")                # 只报告

# 代码指纹清单（A-41①：相关源文件 SHA256 入 freeze，防"理论上相同"的静默改动）
_FINGERPRINT_FILES = (
    "e32_policy.py", "e32_nested.py", "nested_supply.py", "run_e32.py",
    "run_variance_decomp.py", "augment_corpus.py", "family0_actions.py", "run_main_table.py",
    "confirmatory_freeze.py", "confirmatory_corpus.py", "run_confirmatory.py",
    "confirmatory_reporter.py",
    "evaluators/report_target.py", "evaluators/chronos_probe.py", "evaluators/frozen_probe.py",
    "evaluators/_torch_models.py", "evaluators/grounded_forecast.py",
)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def load_dev_records(scope: str = "primary_no_Sar") -> List[dict]:
    path = E32 / f"records_{scope}.jsonl"
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def policy_data_from_records(records: List[dict], actions: List[str]) -> PolicyData:
    """records 的 L_test（每 uid 恰一次 outer-test）→ 全语料 OOF 标签矩阵 = 最终 router 训练集。"""
    order = [r["uid"] for r in records]
    L = np.array([[r["L_test"][a] for a in actions] for r in records])
    return PolicyData(
        uids=order, actions=list(actions), L=L,
        X_d=np.array([[r["snr"], r["miss_rate"]] for r in records]),
        X_p=np.array([r["X_p"] for r in records]),
        cell=np.array([r["cell"] for r in records]),
        origin=np.array([r["origin"] for r in records]),
        X_t=np.array([r.get("X_t", [0.0, 0.0]) for r in records]))


def train_final_arms(seed: int = DEFAULT_SEED, scope: str = "primary_no_Sar"
                     ) -> Tuple[Dict[str, object], PolicyData]:
    """dev records（scope=common-support 主口径）全体行上拟合最终臂（标签=OOF → 无泄漏）。"""
    records = load_dev_records(scope)
    data = policy_data_from_records(records, list(PRUNED_POOL_CORE))
    factory = make_all_arms(seed)
    tr = np.arange(data.n)
    fitted = {name: factory[name]().fit(data, tr) for name in FROZEN_ARM_NAMES}
    return fitted, data


def serialize_arms(fitted: Dict[str, object], data: PolicyData, seed: int,
                   path: Path = FROZEN_ARMS) -> str:
    """joblib 序列化冻结臂 + 训练元数据；返回文件 SHA256（入 freeze，守卫①核验）。"""
    import joblib
    import sklearn
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(
        arms=fitted, arm_names=list(FROZEN_ARM_NAMES), actions=list(data.actions),
        train_scope="primary_no_Sar", train_n=int(data.n), train_seed=int(seed),
        train_labels="dev E-3.2 nested outer-test 标签（全语料 OOF，每 uid 恰一次，A-39①）",
        sklearn_version=sklearn.__version__, numpy_version=np.__version__), path)
    return _sha256_file(path)


def load_frozen_arms(path: Path = FROZEN_ARMS, verify_sha: str = None) -> dict:
    """load 冻结臂；verify_sha 给定时先核验文件 SHA（守卫①）。**调用方永不调 .fit**（A-41②）。"""
    import joblib
    if verify_sha is not None:
        actual = _sha256_file(path)
        if actual != verify_sha:
            raise SystemExit(f"A-41 守卫①失败：frozen_arms SHA 不一致\n  freeze: {verify_sha}\n  actual: {actual}")
    return joblib.load(path)


def _versions() -> dict:
    import joblib
    import scipy
    import sklearn
    out = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
               sklearn=sklearn.__version__, joblib=joblib.__version__)
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:
        out["torch"] = None
    try:
        from importlib.metadata import version
        out["chronos_forecasting"] = version("chronos-forecasting")
    except Exception:
        out["chronos_forecasting"] = None
    return out


def _code_fingerprint() -> Dict[str, str]:
    out = {}
    for rel in _FINGERPRINT_FILES:
        p = PKG / rel
        out[rel] = _sha256_file(p) if p.exists() else "MISSING"
    return out


def write_freeze(router_sha: str, train_n: int, seed: int = DEFAULT_SEED) -> dict:
    """A-41① 机器可读冻结产物。必须先于任何 seeds 20–39 读取落盘。"""
    a31e_protocol = json.loads((RESULTS_A31E / "protocol.json").read_text("utf-8"))
    band_split = a31e_protocol["band_split_snr_db"]        # A-41③：dev 实测带界，预锁不再重算
    dev_records_sha = _sha256_file(E32 / "records_primary_no_Sar.jsonl")

    freeze = dict(
        amendment="A-40+A-41", date="2026-07-04",
        candidate="dp_abstain",
        actions=list(PRUNED_POOL_CORE), p_feats=list(P_FEATS), d_feats=list(D_FEATS),
        gbdt_params=GBDT_PARAMS, kappa=KAPPA, n_ensemble=N_ENSEMBLE, fallback=FALLBACK_ACTION,
        eps=EPS, delta_safe=DELTA_SAFE,
        router=dict(artifact="frozen_arms.joblib", sha256=router_sha,
                    arm_names=list(FROZEN_ARM_NAMES), train_scope="primary_no_Sar",
                    train_n=int(train_n), train_seed=int(seed),
                    train_source="results/E3_2/records_primary_no_Sar.jsonl",
                    train_source_sha256=dev_records_sha,
                    note="locked transfer 只 load 本文件，永不 fit（A-41②）；replication 才重新拟合"),
        holdout=dict(base_namespace="{struct}:{dname}:{j}", j_range=list(HOLDOUT_J),
                     note="seeds 20–39 原 namespace；本 freeze 落盘前不得读取（A-41⑥守卫③）"),
        a38c=dict(namespace='sd=_det_seed(struct,"A38C",cell,k)%2_000_000; uid="{struct}:A38C:{cell}:{k}"',
                  n_target=N_TARGET_CONF, noise_range=[0.03, 2.0], miss_of={"full": 0.0, "miss": 0.06},
                  max_attempts_per_slot=2000, band_phase_frac=2 / 3,
                  band_split_snr_db=band_split,
                  band_split_source="results/A31e/protocol.json（dev 实测；A-41③ 预锁，不看 confirmatory SNR 分布）",
                  acceptance="perceive cell 命中（零 loss 参与）；sd 碰撞对 dev∪A31e∪confirmatory 基底显式跳过"),
        scope=dict(primary="primary_no_Sar（common-support：排除 S_ar，判据口径，A-40①）",
                   annex="all_data（只报告，不判）"),
        estimands=dict(primary="locked_transfer（frozen router→confirmatory uid；测量头=confirmatory 内 grouped OOF）",
                       secondary="replication（confirmatory 内重新 nested cross-fit，检验学习程序可重复）",
                       aggregation_for_gate="original_distribution（uid 加权，与 dev estimand 同口径）",
                       aggregation_report_also="cell_equal（分开写，A-41⑥守卫⑨）"),
        measurement=dict(oof_k=OOF_K, head="Ridge(alpha=1.0) on FrozenProbe 特征（nested_supply._fit_head）",
                         labels="confirmatory 内 cell×origin 分层 grouped OOF（router 冻结→无选择泄漏）",
                         replication_outer_k=REPL_OUTER_K, replication_inner_k=REPL_INNER_K),
        statistics=dict(
            main_judge=f"grouped full-refit bootstrap B={FULL_REFIT_B}（A-33c 机器：每 replicate 组重采样 uid+"
                       "身份分折+sample_weight 重拟合测量头+变折 seed；router picks 冻结不随 replicate 变）"
                       "+ per-replicate 独立种子 + checkpoint/resume",
            paired_uid=f"paired-uid bootstrap B={PAIRED_BOOT_B}（次要参照，条件于已拟合头）",
            dlinear_scratch=f"S={len(REPORTER_SEEDS)} 训练种子逐 series 平均 × uid 配对 bootstrap "
                            f"B={PAIRED_BOOT_B}（B×S 全量重训不可行，A-41⑤；种子间离散另报）",
            chronos=f"零样本无可重拟合头 → grouped paired-uid bootstrap B={PAIRED_BOOT_B}"),
        reporter=dict(panel=list(REPORTER_PANEL), policies=list(REPORTER_POLICIES),
                      dlinear_seeds=list(REPORTER_SEEDS), epochs=120,
                      chronos_model="amazon/chronos-bolt-small",
                      metric="perf=exp(−mean nRMSE)（report_target.py 冻结口径）；配对检验用 per-series nRMSE",
                      nan_policy="任何 series 级非有限值 → 显式报错（无静默回退，A-41⑥守卫⑤）"),
        criteria=dict(
            note="布尔判据表（A-41④）；C1–C5 判于 primary（common-support）+ original_distribution；"
                 "ε=0.03、δ_safe=0.05 与 dev 相同",
            C1_vs_global="mean_regret(dp_abstain)−mean_regret(global) < −ε 且 full-refit CI 上界 < 0",
            C2_vs_dlookup="同 C1，对照=d_lookup（冻结 cell 查表）",
            C3_vs_dgbdt="同 C1，对照=d_gbdt（冻结连续-D GBDT）",
            C4_season_worst_lcb="min over S_season 子群 [full-refit bootstrap 5% 分位(Δ vs incumbent)] > −δ_safe"
                                "（dev 同式 normal-approx LCB 另报作可比参照）",
            C5_trend_retention="trend retention（vs 冻结 d_lookup，locked-transfer 标签点估计）≥ 0.5",
            C6_reporter="两报告器（dlinear_scratch, chronos）各自满足 perf(dp_abstain)>perf(global) 且 "
                        ">perf(d_lookup)（点方向）；且四个配对 per-series nRMSE 差的 bootstrap CI 无显著反向"
                        "（反向=dp 显著更差：diff ci_lo>0）；season/trend 子群方向同报",
            report_only=["overall_worst_group（不作门，且不得写全局安全）", "all_data annex",
                         "cell_equal 聚合", "replication 全部", "oracle_struct 迁移诊断",
                         "true_d_gbdt", "dp_gbdt", "p_gbdt"]),
        comparisons=dict(gate=list(GATE_COMPARATORS), report=list(REPORT_COMPARATORS)),
        one_shot="打开后无论结果如何，不得再调策略并复用本批 holdout（A-41⑦/A-32d）",
        dev_result_pointer=dict(freeze_sha="f1b6c1b75a4e975c", decision="results/E3_2/decision_e32.md",
                                verdict="D-3.2e 7/7 primary"),
        versions=_versions(), code_fingerprint=_code_fingerprint(),
        config_sha=None)
    freeze["config_sha"] = _sha_obj({k: v for k, v in freeze.items() if k != "config_sha"})
    RESULTS_CONF.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(freeze, ensure_ascii=False, indent=1), "utf-8")
    return freeze


def main():
    if FREEZE_PATH.exists():
        raise SystemExit(f"confirmatory_freeze.json 已存在（{FREEZE_PATH}）；"
                         "冻结产物不覆盖——如需重做请人工移走旧文件并记录理由。")
    print("训练最终冻结臂（dev primary records，全语料 OOF 标签）…", flush=True)
    fitted, data = train_final_arms()
    sha = serialize_arms(fitted, data, DEFAULT_SEED)
    print(f"frozen_arms.joblib 落盘  sha256={sha[:16]}…  train_n={data.n}", flush=True)
    freeze = write_freeze(sha, data.n)
    print(f"confirmatory_freeze.json 落盘  config_sha={freeze['config_sha']}", flush=True)
    print("A-41①② 完成：holdout 读取门禁自此解锁（守卫测试全过后方可实际打开）。")


if __name__ == "__main__":
    main()
