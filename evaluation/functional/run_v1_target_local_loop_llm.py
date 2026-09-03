"""【已退役 RETIRED 2026-08-08】逻辑依赖旧闭环（未分离双臂/未修正四类转移）；
Memory 注入可见性已由 3 次真实 LLM 调用验证完成（system 渲染块 + A3 不注入）。
后续闭环以零 LLM 版（run_v1_target_local_loop.py）为准，不再维护本脚本。

原用途：V1 Target-local 闭环：真实 LLM 验证（agicto/deepseek，2026-08-08）。

用户裁决：闭环（零 LLM）通过后，用 2–6 次真实调用同时验证
"Memory 注入、LLM 可见性、下一轮行为"——API 花费对应完整 Harness 自更新能力。

设计（3 次调用，最小可行）：
- R1-A5：LLM 生成（带 Source 对照包渲染）→ Support 实测 → 写 Target Episode
- R1-A3：LLM 生成（无 Memory）→ Support 实测 → 写 Target Episode
- delayed 冻结后打开 → 本地 Episode 状态更新（LOCAL_ACTIVE/RESTRICTED）
- R2-A5：LLM 生成（带 Source + 本地对照包渲染）→ 检查下一轮行为

断言（诚实预期）：
- Memory 注入 + LLM 可见性：确定性（prompt 含渲染块）——应 PASS；
- 下一轮行为：LLM 遵循不可靠是已知事实——如实记录（若忽略，
  闭环的确定性环节（本地 Episode 进检索池/Support 兜底）仍然生效）。

用法：
  python evaluation/functional/run_v1_target_local_loop_llm.py [--provider agicto|deepseek]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_target_local_loop as loop  # noqa: E402
from experience_memory import (  # noqa: E402
    ContrastPack,
    SignedEpisodeRetriever,
    render_experience_pack,
)

REPORT_OUT_REL = Path("artifacts/functional/e2/w1_target_local_loop_llm_report.json")
DOMAIN = "nn5"
EXPERIENCE_REFERENCES = "EXPERIENCE REFERENCES FROM PRIOR TRIALS"

PROVIDERS = {
    "agicto": ("OPENAI_API_KEY", "AGICTO_API_KEY", "https://api.agicto.cn/v1", "gpt-5.6-luna"),
    "deepseek": ("DEEPSEEK_API_KEY", "", "https://api.deepseek.com", "deepseek-v4-flash"),
}


def make_proposer(provider: str, model: str | None = None):
    cfg = PROVIDERS[provider]
    key = next((os.environ.get(k, "").strip() for k in cfg[:2] if os.environ.get(k, "").strip()), "")
    if not key:
        raise SystemExit(f"{cfg[0]} required for provider={provider}")
    return v6.LiveJSONProposer(api_key=key, model=model or cfg[3], base_url=cfg[2])


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 Target-local loop real-LLM verification")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default="agicto")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    root = args.root.resolve()

    ss, sd, ts, td = core.TIMELINE[DOMAIN]
    config = dict(v6.DATASET_CONFIGS[DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    baseline_cache: dict[int, float] = {}

    # Source prior（A5 独有）
    from run_w2_operator_scan import _default_params
    source_episodes, _ = v1.build_source_memory(
        domain=DOMAIN, roster=roster, values=values, config=config,
        operators=sorted(n for n in v6.OPERATOR_NAMES
                         if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                         and n not in core.CTS_EXCLUDED),
        source_support_origin=ss, source_delayed_origin=sd, baseline_cache=baseline_cache,
    )
    print(f"== source episodes: {len(source_episodes)}")

    # 检索器（Source + 本地）
    def make_retriever(extra: list[Any]):
        return SignedEpisodeRetriever(
            source_episodes + extra,
            task_consumer_key="forecast|ridge|sMASE",
        )

    def pack_to_payload(pack: ContrastPack | None) -> Mapping[str, object] | None:
        return pack.to_dict() if pack is not None else None

    # 检索特征（当前 Target 切片）
    target_features = v1.extract_F(values, config, ts)

    def run_llm_round(label: str, pack: ContrastPack | None) -> dict[str, Any]:
        proposer = make_proposer(args.provider, args.model)
        captured_prompts: list[str] = []

        def wrapped(payload: Mapping[str, object]) -> Mapping[str, object]:
            pl = copy.deepcopy(dict(payload))
            if pack is not None:
                pl["experience_contrast_pack"] = pack.to_dict()
            result = proposer(pl)
            # 可见性断言：渲染块在 system prompt（LiveJSONProposer 内部构造）
            captured_prompts.append(getattr(proposer, "last_system_prefix", "") or "")
            return result

        report = v6.run(root, initial_proposer=wrapped, revision_proposer=wrapped,
                        dataset_key=DOMAIN, write_report=False)
        # 提取生成提案 + support
        proposals = report.get("generation_proposals")
        ops = []
        gains = []
        if isinstance(proposals, list):
            for row in proposals:
                if isinstance(row, dict):
                    steps = row.get("compiled_program_steps") or row.get("workflow_steps")
                    if isinstance(steps, list) and steps:
                        ops.append(str(steps[0].get("op", "?")))
                    sr = row.get("support_response")
                    if isinstance(sr, dict) and isinstance(sr.get("support_gain"), (int, float)):
                        gains.append(float(sr["support_gain"]))
        injected = EXPERIENCE_REFERENCES in " ".join(captured_prompts) if captured_prompts else False
        print(f"[{label}] injected={injected} ops={ops} gains={[round(g, 4) for g in gains]} "
              f"status={report.get('final_status')}")
        return {"injected": injected, "ops": ops, "gains": gains,
                "final_status": report.get("final_status")}

    # ---------- R1 ----------
    pack_a5_r1 = make_retriever([]).retrieve(
        {"cohort": {"series_count": 1}, "local_pattern": target_features,
         "program_geometry": {"scope": "training_rows"}}, domain_namespace="")
    print(f"[debug] R1-A5 pack is None: {pack_a5_r1 is None}")
    r1_a5 = run_llm_round("R1-A5", pack_a5_r1)
    r1_a3 = run_llm_round("R1-A3", None)

    # 写 Target Episode（两臂探测过的算子）
    local_episodes = []
    for op, g in zip(r1_a5["ops"], r1_a5["gains"]):
        local_episodes.append(loop.write_target_episode(domain=DOMAIN, op=op, support_gain=g, delayed_gain=None))
    for op, g in zip(r1_a3["ops"], r1_a3["gains"]):
        local_episodes.append(loop.write_target_episode(domain=DOMAIN, op=op, support_gain=g, delayed_gain=None))
    print(f"[R1] target episodes written: {len(local_episodes)}")

    # delayed 冻结后打开 → 状态更新
    updated = []
    for ep in local_episodes:
        compiled = v1.make_compiled(ep.workflow_signature, _default_params(ep.workflow_signature, 7))
        dg = v1.gain_at(roster, values, config, compiled, td, baseline_cache)
        if dg is not None:
            updated.append(loop.update_delayed_status(ep, dg))
    for ep in updated:
        print(f"[R1-delayed] {ep.workflow_signature}: {ep.local_status} (delayed evaluated)")

    # ---------- R2（A5：Source + 本地）----------
    pack_a5_r2 = make_retriever(updated).retrieve(
        {"cohort": {"series_count": 1}, "local_pattern": target_features,
         "program_geometry": {"scope": "training_rows"}}, domain_namespace="")
    print(f"[debug] R2-A5 pack is None: {pack_a5_r2 is None}")
    r2_a5 = run_llm_round("R2-A5", pack_a5_r2)

    # ---------- 断言 ----------
    checks = {
        "memory_injection_visible": r1_a5["injected"] and r2_a5["injected"] and not r1_a3["injected"],
        "r2_retrieval_contains_local": any(
            ep.episode_id in (pack_a5_r2.to_dict() and str(pack_a5_r2.to_dict()))
            for ep in updated
        ) if pack_a5_r2 else False,
        "next_round_behavior_change": r2_a5["ops"] != r1_a5["ops"],
    }
    all_pass = all(checks.values())
    print(f"\n== checks: {checks}")
    print(f"== verdict: {'PASS' if all_pass else 'PARTIAL'}"
          f"（下一轮行为若未变，属 LLM 遵循不可靠的已知边界——闭环确定性环节仍生效）")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-target-local-loop-llm",
            "provider": args.provider,
            "domain": DOMAIN,
            "r1": {"a5": r1_a5, "a3": r1_a3},
            "target_episodes": [e.to_dict() for e in local_episodes],
            "delayed_updated": [e.to_dict() for e in updated],
            "r2": {"a5": r2_a5},
            "checks": checks,
            "verdict": "PASS" if all_pass else "PARTIAL",
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
