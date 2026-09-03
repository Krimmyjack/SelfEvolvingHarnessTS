"""Necessary P0b adapter/baseline smokes and cost accounting.

This is experiment-local code, not a reusable framework.  It has no loader,
path, Final-outcome, Skill-store, or persistence capability.
"""
from __future__ import annotations

from typing import Any, Mapping


class FitBudget:
    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.used = 0

    def spend(self, count: int = 1) -> None:
        if self.used + int(count) > self.cap:
            raise RuntimeError("P0b Consumer-fit cap exceeded")
        self.used += int(count)


def forecast_adapter_smoke() -> dict[str, Any]:
    import numpy as np

    from evaluation.functional import (
        run_e2_autonomous_natural_workflow_generation as forecast,
    )
    from evaluation.functional.consumers.p0b_scope_adapters import (
        ForecastScopeAdapter,
    )

    t = np.arange(400, dtype=np.float64)
    values = {
        "train_a": np.sin(2 * np.pi * t / 24) + 0.002 * t,
        "train_b": 0.8 * np.sin(2 * np.pi * (t + 3) / 24) + 0.003 * t,
        "train_c": 1.2 * np.sin(2 * np.pi * (t + 7) / 24) - 0.001 * t,
        "eval_a": 0.9 * np.sin(2 * np.pi * (t + 1) / 24) + 0.0025 * t,
        "eval_b": 1.1 * np.sin(2 * np.pi * (t + 5) / 24) + 0.0015 * t,
    }
    roster = [
        {"series_uid": uid, "role": "train"}
        for uid in ("train_a", "train_b", "train_c")
    ] + [
        {"series_uid": uid, "role": "eval"}
        for uid in ("eval_a", "eval_b")
    ]
    budget = FitBudget(2)
    adapter = ForecastScopeAdapter(
        frozen_evaluate=forecast._evaluate,
        fit_budget=budget,
        phase_by_origin={300: "support_a", 324: "support_b"},
    )
    config = {"anchors": (192, 240), "period": 24}
    readings = {
        "support_a": adapter(roster, values, None, config, origin=300),
        "support_b": adapter(roster, values, None, config, origin=324),
    }
    return {
        "task": "forecast",
        "status": "PASS_EXPOSED_SYNTHETIC_CONTRACT",
        "consumer": "existing pooled/shared Ridge; H=48",
        "primary_metric": "sMASE",
        "logical_evaluations": len(adapter.calls),
        "raw_consumer_fits": budget.used,
        "readings": readings,
        "final_outcome_bytes_read": 0,
    }


def anomaly_adapter_smoke() -> dict[str, Any]:
    import numpy as np

    from evaluation.functional.consumers import aegists_iforest_v1 as consumer
    from evaluation.functional.consumers.p0b_scope_adapters import (
        WindowedIForestAdapter,
    )

    t = np.arange(360, dtype=np.float64)
    series = np.sin(2 * np.pi * t / 30) + 0.0005 * t
    series[[202, 203, 204, 302, 303, 304]] += 8.0
    rows = {"series_a": {
        "values": series,
        "windows": {"r1": {
            "train": [0, 180],
            "support_a": [180, 260],
            "support_b": [260, 340],
        }},
    }}
    roster = [{"series_uid": "series_a", "role": "train"}]
    values = {"series_a": series}
    events = {
        ("series_a", 180, 260): [list(range(202, 205))],
        ("series_a", 260, 340): [list(range(302, 305))],
    }
    requests: list[tuple[str, int, int]] = []

    def event_reader(uid: str, lo: int, hi: int) -> list[list[int]]:
        requests.append((uid, lo, hi))
        return events[(uid, lo, hi)]

    budget = FitBudget(1)
    adapter = WindowedIForestAdapter(
        consumer=consumer,
        rows=rows,
        round_name="r1",
        event_reader=event_reader,
        fit_budget=budget,
        phase_by_origin={10: "support_a", 20: "support_b"},
    )
    readings = {
        "support_a": adapter(roster, values, None, {}, origin=10),
        "support_b": adapter(roster, values, None, {}, origin=20),
    }
    return {
        "task": "anomaly_detection",
        "status": "PASS_EXPOSED_SYNTHETIC_CONTRACT",
        "consumer": "existing per-series IsolationForest",
        "primary_metric": "Event-F1",
        "logical_evaluations": len(adapter.calls),
        "raw_consumer_fits": budget.used,
        "model_cache_hits": sum(row["model_cache_hits"] for row in adapter.calls),
        "label_wall_requests": [list(row) for row in requests],
        "readings": readings,
        "final_outcome_bytes_read": 0,
    }


BASELINE_BUDGET = {
    "full_support_evaluations": 4,
    "cheap_probes": 12,
    "llm_calls": 4,
    "tokens": 40_000,
    "accepted_updates": 1,
}


def _baseline_task(program_row: Mapping[str, Any]) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.baselines import (
        ProgramLoss,
        select_best_fixed,
    )

    task = str(program_row["task"])
    programs = ["identity", *list(program_row["smoke_programs"])[:3]]
    if len(programs) != 4:
        raise RuntimeError("%s lacks four programs for baseline smoke" % task)
    losses = dict(zip(programs, (1.00, 0.92, 1.05, 0.97)))
    best = select_best_fixed([
        ProgramLoss("support_a", "p0b_%s_fixture" % task,
                    program, "public_synthetic_unit", loss)
        for program, loss in losses.items()
    ])
    parallel = min(programs, key=lambda value: (losses[value], value))
    sequential = programs[0]
    for candidate in programs[1:]:
        sequential = min((sequential, candidate),
                         key=lambda value: (losses[value], value))

    def row(name: str, selected: str, full: int, *, history="none",
            writeback=False, updates=0, selection: str) -> dict[str, Any]:
        usage = {"full_support_evaluations": full, "cheap_probes": 0,
                 "llm_calls": 0, "tokens": 0, "accepted_updates": updates}
        passed = (selected in programs
                  and all(usage[key] <= BASELINE_BUDGET[key] for key in usage)
                  and updates <= int(writeback))
        return {"method": name, "status": "PASS" if passed else "FAIL",
                "selected_program": selected, "selection": selection,
                "initial_history": history, "writeback": writeback,
                "usage": usage}

    methods = [
        row("Identity", "identity", 0, selection="no adaptation"),
        row("Fixed Heuristics", programs[1], 1, selection="predeclared rule"),
        row("Best Fixed Per-task", best.program_id, 4, selection="Support-A only"),
        row("Parallel Best-of-N@B_main", parallel, 4, selection="independent"),
        row("Sequential Refinement@B_main", sequential, 4, selection="ordered feedback"),
        row("Frozen H0", "identity", 0, selection="frozen initial harness"),
        row("Static", "identity", 0, selection="no history/adaptation"),
        row("A3-reset", programs[1], 1, selection="target-local cold start"),
        row("K0-fixed", programs[1], 1, history="shared_k0_a5_origin",
            selection="historical supply remains Support-gated"),
        row("A5-online", programs[1], 1, history="shared_k0_a5_origin",
            writeback=True, updates=1,
            selection="historical supply remains Support-gated"),
    ]
    return {
        "task": task,
        "status": "PASS" if all(item["status"] == "PASS" for item in methods) else "FAIL",
        "programs": programs,
        "methods": methods,
        "k0_a5_same_initial_history": methods[-2]["initial_history"] == methods[-1]["initial_history"],
    }


def baseline_contract_smoke(program_space: Mapping[str, Any]) -> dict[str, Any]:
    tasks = [_baseline_task(row) for row in program_space["tasks"]]
    return {
        "status": ("PASS_MINIMAL_CONTRACT_SMOKE"
                   if all(row["status"] == "PASS" for row in tasks)
                   else "BLOCKED_MINIMAL_CONTRACT_SMOKE"),
        "common_contract": {
            "input": "same TaskSpec/ConsumerSpec/split/legal-program-list",
            "fixture": "exposed_or_synthetic_no_final_outcome",
            "budget_cap": BASELINE_BUDGET,
            "query_reads": 0,
            "performance_or_headroom_claim": False,
        },
        "tasks": tasks,
        "p1_full_core_smoke": "PENDING",
    }


def cost_accounting(
    *, final_datasets: int, adapter_raw_fits: int, baseline_task_count: int,
) -> dict[str, Any]:
    nonadaptive = ["Identity", "Best Fixed Per-task", "Fixed Linear-impute",
                   "Fixed Hampel", "Fixed Winsor", "Fixed IQR", "Frozen H0", "Static"]
    adaptive = ["Parallel Best-of-N@B_main", "Sequential Refinement@B_main",
                "A3-reset", "K0-fixed", "A5-online"]
    vectors = {2: {"full": 2, "probes": 6, "llm": 2, "tokens": 20_000},
               4: {"full": 4, "probes": 12, "llm": 4, "tokens": 40_000},
               8: {"full": 8, "probes": 24, "llm": 6, "tokens": 60_000}}
    phases = [{"phase": "evolution", "cells": 72, "b": 4},
              {"phase": "validation_b2", "cells": 27, "b": 2},
              {"phase": "validation_b4", "cells": 27, "b": 4},
              {"phase": "validation_b8", "cells": 27, "b": 8},
              {"phase": "natural_final", "cells": final_datasets * 3, "b": 4}]
    totals = {key: 0 for key in (
        "method_cell_runs", "full_support_logical_evaluations_cap",
        "cheap_probes_cap", "llm_calls_cap", "tokens_cap",
        "accepted_a5_updates_cap", "query_extra_logical_evaluations",
        "serial_stochastic_wall_minutes_cap")}
    for phase in phases:
        cells, vector = phase["cells"], vectors[phase["b"]]
        totals["method_cell_runs"] += cells * (len(nonadaptive) + len(adaptive))
        totals["full_support_logical_evaluations_cap"] += cells * len(adaptive) * vector["full"]
        totals["cheap_probes_cap"] += cells * len(adaptive) * vector["probes"]
        totals["llm_calls_cap"] += cells * len(adaptive) * vector["llm"]
        totals["tokens_cap"] += cells * len(adaptive) * vector["tokens"]
        totals["accepted_a5_updates_cap"] += cells
        totals["query_extra_logical_evaluations"] += cells * (len(nonadaptive) + len(adaptive))
        totals["serial_stochastic_wall_minutes_cap"] += cells * len(adaptive) * 45
    return {
        "status": "PASS_COST_ACCOUNTING_FREEZE",
        "judgement": "accounting completeness only; no affordability threshold",
        "method_roster": {"nonadaptive": nonadaptive, "adaptive": adaptive},
        "phase_rows": phases, "resource_vectors": vectors, "totals": totals,
        "replica_rule": "three pre-registered replicas in all phases",
        "consumer_fit_accounting": {
            "logical": "one complete task-native roster evaluation; matched-B unit",
            "raw": "record separately; AD equals evaluated series count",
            "cache": "cache hits have zero raw fits and create no evidence",
            "query": "one extra logical evaluation per method-cell; outside B",
        },
        "p0b_observed_smoke": {"adapter_raw_consumer_fits": adapter_raw_fits,
                               "baseline_contract_task_count": baseline_task_count,
                               "final_outcome_reads": 0},
    }


__all__ = [
    "anomaly_adapter_smoke",
    "baseline_contract_smoke",
    "cost_accounting",
    "forecast_adapter_smoke",
]
