"""Frozen protocol decisions and lineage artifacts for TSharness vNext."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._canonical import file_sha256, require_sha, sha256


PROTOCOL_VERSION = "tsharness-vnext-protocol/3"
DISCOVERY_SEED = 20260713


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@dataclass(frozen=True)
class ProtocolResolutionAddendumV1:
    benchmark_version: str = "benchmark-v0.2"
    task_g_view: str = "support_a_discovery_group_crossfit"
    task_g_may_change_method_semantics: bool = False
    postfreeze_dev_accesses: int = 1
    sa_validation_exposure: str = "baseline_outcome_exposed_vnext_candidate_unqueried"
    sa_validation_comparator: str = "h_candidate_minus_h0"
    support_b_accesses: int = 1
    final_requires_separate_authorization: bool = True
    h0_definition: str = "init_corpus_only_operator_experience_seed_policies_templates_memory"
    initial_llm_efficacy_gates_evolution_trial: bool = False
    bootstrap_seed: int = DISCOVERY_SEED
    bootstrap_replicates: int = 2000
    ci_level: float = 0.90
    epsilon: float = 0.02
    harm_delta: float = 0.05
    schema_version: str = PROTOCOL_VERSION

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class DataViewV1:
    view_id: str
    series_uids: tuple[str, ...]
    overlap_groups: tuple[str, ...]
    permitted_uses: tuple[str, ...]
    historical_result_exposure: str

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.series_uids))) != self.series_uids:
            raise ValueError(f"{self.view_id} series_uids must be sorted and unique")
        if tuple(sorted(set(self.overlap_groups))) != self.overlap_groups:
            raise ValueError(f"{self.view_id} overlap_groups must be sorted and unique")


@dataclass(frozen=True)
class VNextDataUsageManifestV1:
    benchmark_version: str
    split_manifest_sha: str
    support_a_subsplit_sha: str
    init: DataViewV1
    search: DataViewV1
    sa_validation: DataViewV1
    dev: DataViewV1
    support_b: DataViewV1
    final_query: DataViewV1
    u: DataViewV1
    schema_version: str = "vnext-data-usage/1"

    def __post_init__(self) -> None:
        require_sha(self.split_manifest_sha, "split_manifest_sha")
        require_sha(self.support_a_subsplit_sha, "support_a_subsplit_sha")
        if set(self.init.series_uids) & set(self.search.series_uids):
            raise ValueError("Init and SA-D/search must be disjoint")
        if set(self.init.overlap_groups) & set(self.search.overlap_groups):
            raise ValueError("Init and SA-D/search overlap groups must be disjoint")
        if set(self.sa_validation.series_uids) & (
            set(self.init.series_uids) | set(self.search.series_uids)
        ):
            raise ValueError("SA-V must be disjoint from Support-A discovery")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)

    @classmethod
    def from_frozen_v02(cls, project_root: Path | str) -> "VNextDataUsageManifestV1":
        root = Path(project_root)
        result = root / "results" / "Benchmark_v0_2"
        split_path = result / "split_manifest.json"
        subsplit_path = result / "support_a_subsplit.json"
        split = _json(split_path)
        subsplit = _json(subsplit_path)
        rows = split["assignments"]
        by_uid = {row["series_uid"]: row for row in rows}
        discovery = set(subsplit["members"]["support_a_discovery"])
        validation = set(subsplit["members"]["support_a_validation"])

        init_uids = {
            uid for uid in discovery
            if by_uid[uid]["exposure_class"] != "certified_virgin"
        }
        search_uids = discovery - init_uids

        def view(
            view_id: str,
            uids: set[str],
            uses: tuple[str, ...],
            exposure: str,
        ) -> DataViewV1:
            groups = {by_uid[uid]["overlap_group"] for uid in uids}
            return DataViewV1(
                view_id=view_id,
                series_uids=tuple(sorted(uids)),
                overlap_groups=tuple(sorted(groups)),
                permitted_uses=uses,
                historical_result_exposure=exposure,
            )

        role_uids = {
            role: {row["series_uid"] for row in rows if row["role"] == role}
            for role in ("dev_query", "support_b", "final_query", "u")
        }
        manifest = cls(
            benchmark_version="benchmark-v0.2",
            split_manifest_sha=file_sha256(split_path),
            support_a_subsplit_sha=file_sha256(subsplit_path),
            init=view(
                "support_a_discovery_init", init_uids,
                ("formal_h0_construction",),
                "historically_exposed",
            ),
            search=view(
                "support_a_discovery_search", search_uids,
                ("m3_identity", "m3b_evolution", "crossfit", "threshold_fit"),
                "baseline_outcome_exposed_method_unqueried",
            ),
            sa_validation=view(
                "support_a_validation", validation,
                ("one_shot_vnext_candidate_aggregate_gate",),
                "baseline_outcome_exposed_method_unqueried",
            ),
            dev=view(
                "postfreeze_dev", role_uids["dev_query"],
                ("postfreeze_readonly_report",), "historically_exposed_repeatable",
            ),
            support_b=view(
                "support_b", role_uids["support_b"],
                ("one_shot_confirmation",), "sealed",
            ),
            final_query=view(
                "final_query", role_uids["final_query"],
                ("separately_authorized_final_campaign",), "sealed",
            ),
            u=view("u", role_uids["u"], ("directional_only",), "sealed"),
        )
        expected = {
            "init": (136, 136), "search": (284, 247),
            "sa_validation": (135, 110), "dev": (373, 323),
            "support_b": (331, 281), "final_query": (570, 545), "u": (38, 38),
        }
        for name, counts in expected.items():
            item = getattr(manifest, name)
            actual = (len(item.series_uids), len(item.overlap_groups))
            if actual != counts:
                raise ValueError(f"frozen {name} counts drifted: {actual} != {counts}")
        return manifest


@dataclass(frozen=True)
class HistoricalExposureManifestV1:
    benchmark_version: str
    support_a_uid_count: int
    sa_validation_uids: tuple[str, ...]
    sa_validation_program_loss_uid_count: int
    sa_validation_repeat_loss_uid_count: int
    exposed_program_ids: tuple[str, ...]
    repeat_measurements: tuple[tuple[str, float, int], ...]
    program_losses_sha: str
    repeat_losses_sha: str
    exposure_label: str = "baseline_outcome_exposed"
    automatic_search_read_allowed: bool = False
    schema_version: str = "vnext-historical-exposure/1"

    def __post_init__(self) -> None:
        require_sha(self.program_losses_sha, "program_losses_sha")
        require_sha(self.repeat_losses_sha, "repeat_losses_sha")
        if self.automatic_search_read_allowed:
            raise ValueError("historical per-UID losses may not enter automated search")
        if (
            self.sa_validation_program_loss_uid_count != len(self.sa_validation_uids)
            or self.sa_validation_repeat_loss_uid_count != len(self.sa_validation_uids)
        ):
            raise ValueError("SA-V historical result exposure is incomplete or inconsistent")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)

    @classmethod
    def from_frozen_v02(cls, project_root: Path | str) -> "HistoricalExposureManifestV1":
        result = Path(project_root) / "results" / "Benchmark_v0_2"
        subsplit = _json(result / "support_a_subsplit.json")
        validation = set(subsplit["members"]["support_a_validation"])
        program_path = result / "dev_program_losses.jsonl"
        repeat_path = result / "dev_repeat_losses.jsonl"
        program_uids: set[str] = set()
        repeat_uids: set[str] = set()
        support_a_uids: set[str] = set()
        programs: set[str] = set()
        measurements: set[tuple[str, float, int]] = set()
        for row in _jsonl(program_path):
            if row.get("split_role") == "support_a":
                uid = str(row["uid"])
                support_a_uids.add(uid)
                programs.add(str(row["program_id"]))
                if uid in validation:
                    program_uids.add(uid)
        for row in _jsonl(repeat_path):
            if row.get("split_role") == "support_a":
                uid = str(row["uid"])
                if uid in validation:
                    repeat_uids.add(uid)
                    measurements.add((
                        str(row["scenario"]), float(row["dose"]),
                        int(row["corruption_replicate"]),
                    ))
        return cls(
            benchmark_version="benchmark-v0.2",
            support_a_uid_count=len(support_a_uids),
            sa_validation_uids=tuple(sorted(validation)),
            sa_validation_program_loss_uid_count=len(program_uids),
            sa_validation_repeat_loss_uid_count=len(repeat_uids),
            exposed_program_ids=tuple(sorted(programs)),
            repeat_measurements=tuple(sorted(measurements)),
            program_losses_sha=file_sha256(program_path),
            repeat_losses_sha=file_sha256(repeat_path),
        )


@dataclass(frozen=True)
class GroupFoldV1:
    overlap_group: str
    dataset_id: str
    fold: int


@dataclass(frozen=True)
class DiscoveryFoldManifestV1:
    parent_split_sha: str
    support_a_subsplit_sha: str
    task_g_folds: tuple[GroupFoldV1, ...]
    search_folds: tuple[GroupFoldV1, ...]
    fold_count: int = 5
    task_g_salt: str = "vnext-task-g-crossfit-20260715"
    search_salt: str = "vnext-search-crossfit-20260715"
    unit: str = "overlap_group"
    schema_version: str = "vnext-discovery-folds/1"

    def __post_init__(self) -> None:
        require_sha(self.parent_split_sha, "parent_split_sha")
        require_sha(self.support_a_subsplit_sha, "support_a_subsplit_sha")
        for roster in (self.task_g_folds, self.search_folds):
            groups = [row.overlap_group for row in roster]
            if len(groups) != len(set(groups)):
                raise ValueError("an overlap group appears in multiple folds")
            if any(row.fold < 0 or row.fold >= self.fold_count for row in roster):
                raise ValueError("fold index is outside the frozen range")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)

    @staticmethod
    def _assign(rows: list[Mapping[str, Any]], salt: str, folds: int) -> tuple[GroupFoldV1, ...]:
        by_dataset: dict[str, dict[str, None]] = {}
        for row in rows:
            by_dataset.setdefault(str(row["dataset_id"]), {})[str(row["overlap_group"])] = None
        output: list[GroupFoldV1] = []
        for dataset_id, groups in sorted(by_dataset.items()):
            ranked = sorted(
                groups,
                key=lambda group: hashlib.sha256(
                    f"{salt}|{dataset_id}|{group}".encode("utf-8")
                ).hexdigest(),
            )
            for index, group in enumerate(ranked):
                output.append(GroupFoldV1(group, dataset_id, index % folds))
        return tuple(sorted(output, key=lambda row: row.overlap_group))

    @classmethod
    def from_frozen_v02(cls, project_root: Path | str) -> "DiscoveryFoldManifestV1":
        root = Path(project_root)
        result = root / "results" / "Benchmark_v0_2"
        split_path = result / "split_manifest.json"
        sub_path = result / "support_a_subsplit.json"
        split = _json(split_path)
        sub = _json(sub_path)
        discovery = set(sub["members"]["support_a_discovery"])
        rows = [row for row in split["assignments"] if row["series_uid"] in discovery]
        search = [row for row in rows if row["exposure_class"] == "certified_virgin"]
        return cls(
            parent_split_sha=file_sha256(split_path),
            support_a_subsplit_sha=file_sha256(sub_path),
            task_g_folds=cls._assign(rows, "vnext-task-g-crossfit-20260715", 5),
            search_folds=cls._assign(search, "vnext-search-crossfit-20260715", 5),
        )


@dataclass(frozen=True)
class MethodInputVerdictV1:
    valid: bool
    code: str


@dataclass(frozen=True)
class MethodInputContractV1:
    require_one_dimensional: bool = True
    require_non_empty: bool = True
    forbid_inf: bool = True
    require_finite_observation: bool = True
    terminal_code: str = "METHOD_TERMINAL_INVALID_INPUT"
    schema_version: str = "vnext-method-input/1"

    def validate(self, values: Any) -> MethodInputVerdictV1:
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return MethodInputVerdictV1(False, "non_numeric_input")
        if array.ndim != 1:
            return MethodInputVerdictV1(False, "input_not_one_dimensional")
        if not len(array):
            return MethodInputVerdictV1(False, "empty_input")
        if np.isinf(array).any():
            return MethodInputVerdictV1(False, "infinite_input")
        if not np.isfinite(array).any():
            return MethodInputVerdictV1(False, "no_finite_observation")
        return MethodInputVerdictV1(True, "ok")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class HBaseArtifactV1:
    pattern_binding_sha: str
    candidate_grammar_sha: str
    action_eligibility_sha: str
    selector_sha: str
    risk_sha: str
    retrieval_sha: str
    execution_sha: str
    fallback_sha: str
    budget_sha: str
    init_view_sha: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            require_sha(getattr(self, name), name)

    @property
    def semantic_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class H0LineageArtifactV2:
    h_base_sha: str
    m3a_prereg_sha: str
    m3a_result_sha: str
    supplier_selection_rule_sha: str
    supplier_policy_id: str
    supplier_policy_sha: str
    h0_harness_sha: str
    h0_method_sha: str

    def __post_init__(self) -> None:
        if not self.supplier_policy_id or self.supplier_policy_id != self.supplier_policy_id.strip():
            raise ValueError("supplier_policy_id must be canonical")
        for name in self.__dataclass_fields__:
            if name != "supplier_policy_id":
                require_sha(getattr(self, name), name)

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class LLMTrialAuthorizationV1:
    provider_manifest_sha: str
    prompt_sha: str
    decoding_sha: str
    replay_probe_sha: str
    budget_sha: str
    technically_feasible: bool
    reason: str

    def __post_init__(self) -> None:
        for name in (
            "provider_manifest_sha", "prompt_sha", "decoding_sha",
            "replay_probe_sha", "budget_sha",
        ):
            require_sha(getattr(self, name), name)
        if not self.reason:
            raise ValueError("LLM trial authorization requires a reason")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class LLMQualificationArtifactV2:
    trial_authorization_sha: str
    initial_runtime_efficacy: bool
    evolution_llm_qualified: bool
    mature_runtime_llm_qualified: bool
    complementarity_qualified: bool
    m3a_result_sha: str
    evolution_result_sha: str
    supplier_swap_sha: str

    def __post_init__(self) -> None:
        for name in (
            "trial_authorization_sha", "m3a_result_sha", "evolution_result_sha",
            "supplier_swap_sha",
        ):
            require_sha(getattr(self, name), name)

    @property
    def runtime_llm_qualified(self) -> bool:
        return self.initial_runtime_efficacy or self.mature_runtime_llm_qualified

    @property
    def artifact_sha(self) -> str:
        return sha256(self)
