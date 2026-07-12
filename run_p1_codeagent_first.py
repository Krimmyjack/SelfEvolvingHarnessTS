"""run_p1_codeagent_first.py — P1 接线验收 runner（Final_Plan_CodeAgentFirst_2026-07-09 §P1）。

no-API 端到端切片：DataView → EvidencePacket v2 → CodeAgentComposer（默认上场）→
Compiler/Sandbox → SafetyGate → execute/fallback → synthetic oracle proxy 判官 → EvidenceStore，
三臂并跑（raw 基线 / incumbent deterministic 对照 / code-agent-first stub），输出
manifest.json + report.json + records.jsonl（含按面 harm 台账：baseline_raw / router /
program / gate_fallback）。

本 runner 只做接线与复现验收（P1 出口判据），**不构成任何性能主张**；synthetic proxy
不是论文判官。真实 LLM 后端（cached DeepSeek）经 --backend llm 挂入，默认不启用。

命令：
  D:\\Anaconda_envs\\envs\\project\\python.exe -m SelfEvolvingHarnessTS.run_p1_codeagent_first --n-records 8
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .fast_path.ablation import FastPathAblationArm, run_fast_path_ablation
from .policy.action_spec import action_menu_v1
from .policy.code_agent_composer import CodeAgentComposer
from .policy.escalation import EscalationConfig
from .policy.evidence_packet import PACKET_SCHEMA_V2
from .policy.task_spec import forecast_task_spec_v1
from .run_fast_path_ablation import (
    build_demo_forecast_ablation_inputs,
    synthetic_oracle_proxy_validator,
)

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "P1CodeAgentFirst"


def build_p1_arms(composer: CodeAgentComposer) -> list[FastPathAblationArm]:
    return [
        FastPathAblationArm.raw(),
        FastPathAblationArm(
            name="incumbent_control",
            use_skills=True,
            use_memory=False,
            use_composer=False,
            use_safety=True,
            config=EscalationConfig(),
        ),
        FastPathAblationArm(
            name="code_agent_first_stub" if composer.backend == "stub" else "code_agent_first_llm",
            use_skills=True,
            use_memory=False,
            use_composer=True,
            use_safety=True,
            composer=composer,
            config=EscalationConfig(composer_first=True),
        ),
    ]


def _surface(decision: Any) -> str:
    """按面 harm 归因（R3）：served artifact 由哪个面产生。"""
    if decision.route == "raw":
        return "baseline_raw"
    if decision.safety.fallback_raw:
        return "gate_fallback"
    if str(decision.action_id).startswith("prog1_"):
        return "program"
    return "router"


def _mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    return (sum(finite) / len(finite)) if finite else None


def run_p1(n_records: int = 8, out_dir: Path | str | None = None,
           backend: str = "stub", composer: CodeAgentComposer | None = None) -> dict[str, Any]:
    records, series_by_uid, _memory_by_uid, target_by_uid = build_demo_forecast_ablation_inputs(n_records)
    validator = synthetic_oracle_proxy_validator(target_by_uid)
    menu = action_menu_v1()
    if composer is None:
        composer = (CodeAgentComposer(backend="stub") if backend == "stub"
                    else CodeAgentComposer.with_deepseek())
    arms = build_p1_arms(composer)

    results = run_fast_path_ablation(
        records, series_by_uid, arms=arms, action_menu=menu, validator=validator,
    )

    rows: list[dict[str, Any]] = []
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for res in results:
        downstream = dict(res.validation.result)
        surface = _surface(res.decision)
        row = {
            "arm": res.arm_name,
            "uid": res.uid,
            "action_id": res.decision.action_id,
            "route": res.decision.route,
            "status": res.executed.status,
            "safety_reasons": list(res.decision.safety.reasons),
            "packet_schema": res.decision.packet.get("schema"),
            "task_spec_sha": (res.decision.packet.get("provenance") or {}).get("task_spec_sha"),
            "utility_delta_vs_raw": downstream.get("utility_delta_vs_raw"),
            "harm_delta_vs_raw": downstream.get("harm_delta_vs_raw"),
            "harm_ledger": {
                "surface": surface,
                "utility_delta_vs_raw": downstream.get("utility_delta_vs_raw"),
                "harm_delta_vs_raw": downstream.get("harm_delta_vs_raw"),
            },
        }
        rows.append(row)
        by_arm[res.arm_name].append(row)

    arms_report: dict[str, Any] = {}
    for arm_name, arm_rows in by_arm.items():
        ledger: dict[str, Any] = {}
        for surface in sorted({r["harm_ledger"]["surface"] for r in arm_rows}):
            surface_rows = [r for r in arm_rows if r["harm_ledger"]["surface"] == surface]
            ledger[surface] = {
                "n": len(surface_rows),
                "mean_utility_delta_vs_raw": _mean([r["utility_delta_vs_raw"] for r in surface_rows]),
                "mean_harm_delta_vs_raw": _mean([r["harm_delta_vs_raw"] for r in surface_rows]),
            }
        prefix_counts = Counter()
        for r in arm_rows:
            aid = str(r["action_id"])
            prefix_counts["prog1_" if aid.startswith("prog1_") else aid] += 1
        arms_report[arm_name] = {
            "n_results": len(arm_rows),
            "mean_utility_delta_vs_raw": _mean([r["utility_delta_vs_raw"] for r in arm_rows]),
            "mean_harm_delta_vs_raw": _mean([r["harm_delta_vs_raw"] for r in arm_rows]),
            "fallback_reason_counts": dict(Counter(
                reason for r in arm_rows for reason in r["safety_reasons"])),
            "serve_action_prefix_counts": dict(prefix_counts),
            "harm_ledger_by_surface": ledger,
        }

    report = {
        "phase": "P1_codeagent_first_wiring",
        "claim_scope": "wiring/reproducibility acceptance only; NOT a performance claim",
        "n_records": int(n_records),
        "backend": composer.backend,
        "api_calls": int(composer.total_api_calls),
        "composer_invocations": int(composer.total_invocations),
        "arms": arms_report,
    }
    manifest = {
        "generated_by": "run_p1_codeagent_first",
        "plan": "Final_Plan_CodeAgentFirst_2026-07-09 §P1",
        "prereg": "results/Stage2/prereg_codeagent_first_P1_P5.md §1",
        "action_menu_version": menu.version,
        "action_menu_sha256": menu.sha256,
        "packet_schema": PACKET_SCHEMA_V2,
        "task_spec_sha": forecast_task_spec_v1().sha(),
        "backend": composer.backend,
        "arms": [arm.name for arm in arms],
        "n_records": int(n_records),
        "validator": "synthetic_oracle_proxy_v1",
        "data": "build_demo_forecast_ablation_inputs (deterministic synthetic slice)",
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 code-agent-first wiring acceptance runner")
    parser.add_argument("--n-records", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--backend", type=str, default="stub", choices=["stub", "llm"])
    args = parser.parse_args()
    report = run_p1(n_records=args.n_records, out_dir=args.out_dir, backend=args.backend)
    print(json.dumps({k: report[k] for k in ("phase", "n_records", "backend", "api_calls")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
