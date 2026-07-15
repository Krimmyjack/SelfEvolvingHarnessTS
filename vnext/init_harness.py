"""Init-Corpus-only construction contract for the formal initial Harness H0.

The corpus manifest can be frozen before scientific evaluation.  The final H0 artifact
cannot be created until real Init-only evidence has populated operator experience, seed
policies, program templates, and aggregate memory.  No empty placeholder may masquerade
as H0.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._canonical import file_sha256, require_sha, sha256
from .protocol import VNextDataUsageManifestV1


LEGACY_DOMAIN_BY_DATASET = {
    "legacy_monash:covid_deaths": "public_health",
    "legacy_monash:fred_md": "macroeconomics",
    "legacy_monash:nn5_daily": "cash_demand",
    "legacy_monash:tourism_monthly": "tourism",
    "legacy_monash:us_births": "demography",
    "legacy_monash:saugeenday": "hydrology",
    "legacy_monash:sunspot": "solar",
}
H0_FORBIDDEN_VIEW_IDS = (
    "support_a_discovery_search",
    "support_a_validation",
    "postfreeze_dev",
    "support_b",
    "final_query",
    "u",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


@dataclass(frozen=True)
class InitCorpusMemberV1:
    series_uid: str
    overlap_group: str
    dataset_id: str
    exposure_class: str
    cohort: str
    domain: str

    def __post_init__(self) -> None:
        if self.cohort not in {"legacy_core", "probe_consumed_extension"}:
            raise ValueError("unknown Init Corpus cohort")
        if self.cohort == "legacy_core" and self.exposure_class != "confirmed_exposed":
            raise ValueError("legacy core must be confirmed_exposed")
        if (
            self.cohort == "probe_consumed_extension"
            and self.exposure_class != "probe_consumed"
        ):
            raise ValueError("probe extension must be probe_consumed")


@dataclass(frozen=True)
class InitCorpusManifestV1:
    benchmark_version: str
    split_manifest_sha: str
    support_a_subsplit_sha: str
    registry_sha: str
    data_usage_manifest_sha: str
    members: tuple[InitCorpusMemberV1, ...]
    forbidden_view_ids: tuple[str, ...] = H0_FORBIDDEN_VIEW_IDS
    schema_version: str = "vnext-init-corpus/1"

    def __post_init__(self) -> None:
        for name in (
            "split_manifest_sha", "support_a_subsplit_sha", "registry_sha",
            "data_usage_manifest_sha",
        ):
            require_sha(getattr(self, name), name)
        if len(self.members) != 136:
            raise ValueError("formal Init Corpus must contain exactly 136 series")
        if len({row.series_uid for row in self.members}) != 136:
            raise ValueError("Init Corpus series must be unique")
        if len({row.overlap_group for row in self.members}) != 136:
            raise ValueError("Init Corpus must contain 136 independent overlap groups")
        cohorts = Counter(row.cohort for row in self.members)
        if cohorts != {"legacy_core": 80, "probe_consumed_extension": 56}:
            raise ValueError("Init Corpus must be 80 legacy core + 56 probe extension")
        if any(row.exposure_class == "certified_virgin" for row in self.members):
            raise ValueError("fresh Support-A cannot enter Init Corpus")
        legacy_domains = {row.domain for row in self.members if row.cohort == "legacy_core"}
        if legacy_domains != set(LEGACY_DOMAIN_BY_DATASET.values()):
            raise ValueError("legacy core domain coverage drifted")
        if self.forbidden_view_ids != H0_FORBIDDEN_VIEW_IDS:
            raise ValueError("H0 forbidden view roster drifted")

    @property
    def cohort_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(Counter(row.cohort for row in self.members).items()))

    @property
    def domain_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(Counter(row.domain for row in self.members).items()))

    @property
    def artifact_sha(self) -> str:
        return sha256(self)

    @classmethod
    def from_frozen_v02(cls, project_root: Path | str) -> "InitCorpusManifestV1":
        root = Path(project_root)
        result = root / "results" / "Benchmark_v0_2"
        split_path = result / "split_manifest.json"
        subsplit_path = result / "support_a_subsplit.json"
        registry_path = result / "series_registry.jsonl"
        usage = VNextDataUsageManifestV1.from_frozen_v02(root)
        split = _json(split_path)
        by_uid = {str(row["series_uid"]): row for row in split["assignments"]}
        members: list[InitCorpusMemberV1] = []
        for uid in usage.init.series_uids:
            row = by_uid[uid]
            dataset_id = str(row["dataset_id"])
            exposure = str(row["exposure_class"])
            if exposure == "probe_consumed":
                if dataset_id != "monash:traffic_hourly":
                    raise ValueError("probe-consumed Init extension must be Traffic")
                cohort, domain = "probe_consumed_extension", "traffic"
            else:
                try:
                    domain = LEGACY_DOMAIN_BY_DATASET[dataset_id]
                except KeyError as exc:
                    raise ValueError(f"unregistered legacy Init domain: {dataset_id}") from exc
                cohort = "legacy_core"
            members.append(InitCorpusMemberV1(
                series_uid=uid, overlap_group=str(row["overlap_group"]),
                dataset_id=dataset_id, exposure_class=exposure,
                cohort=cohort, domain=domain,
            ))
        manifest = cls(
            benchmark_version="benchmark-v0.2",
            split_manifest_sha=file_sha256(split_path),
            support_a_subsplit_sha=file_sha256(subsplit_path),
            registry_sha=file_sha256(registry_path),
            data_usage_manifest_sha=usage.artifact_sha,
            members=tuple(sorted(members, key=lambda row: row.series_uid)),
        )
        forbidden_uids = set().union(*(
            set(getattr(usage, name).series_uids)
            for name in (
                "search", "sa_validation", "dev", "support_b", "final_query", "u",
            )
        ))
        if set(usage.init.series_uids) & forbidden_uids:
            raise ValueError("Init Corpus intersects a forbidden H0 input view")
        return manifest


@dataclass(frozen=True)
class InitHarnessPreregV1:
    init_corpus_sha: str
    required_components: tuple[str, ...] = (
        "operator_experience", "seed_policies", "program_templates", "memory",
    )
    operator_experience_rule: str = "init_only_actual_mixed_corpus_retrained_aggregate"
    seed_policy_rule: str = "init_only_deterministic_typed_programspec_v1"
    program_template_rule: str = "candidate_grammar_v1_active_frozen_presets_only"
    memory_rule: str = "init_only_overlap_group_aggregate_no_uid_or_dataset_identity"
    supplier_gate_role: str = "runtime_supplier_control_not_h0_construction"
    forbidden_view_ids: tuple[str, ...] = H0_FORBIDDEN_VIEW_IDS
    schema_version: str = "vnext-init-harness-prereg/1"

    def __post_init__(self) -> None:
        require_sha(self.init_corpus_sha, "init_corpus_sha")
        if self.required_components != (
            "operator_experience", "seed_policies", "program_templates", "memory",
        ):
            raise ValueError("formal H0 requires all four Init-derived components")
        if self.forbidden_view_ids != H0_FORBIDDEN_VIEW_IDS:
            raise ValueError("H0 prereg forbidden views drifted")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class InitOperatorExperienceV1:
    action_id: str
    effect_signature_sha: str
    support_group_count: int
    mean_utility_gain: float
    worst_material_harm: float
    evidence_sha: str
    init_corpus_sha: str

    def __post_init__(self) -> None:
        for name in ("effect_signature_sha", "evidence_sha", "init_corpus_sha"):
            require_sha(getattr(self, name), name)
        if not self.action_id or self.support_group_count < 1 or self.support_group_count > 136:
            raise ValueError("invalid Init operator experience")


@dataclass(frozen=True)
class InitSeedPolicyV1:
    policy_id: str
    canonical_program_sha: str
    selector_condition_sha: str
    evidence_sha: str
    init_corpus_sha: str

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("seed policy id is required")
        for name in (
            "canonical_program_sha", "selector_condition_sha", "evidence_sha",
            "init_corpus_sha",
        ):
            require_sha(getattr(self, name), name)


@dataclass(frozen=True)
class InitProgramTemplateV1:
    template_id: str
    canonical_program_sha: str
    effect_signature_sha: str
    action_menu_sha: str
    candidate_grammar_sha: str

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("program template id is required")
        for name in (
            "canonical_program_sha", "effect_signature_sha", "action_menu_sha",
            "candidate_grammar_sha",
        ):
            require_sha(getattr(self, name), name)


@dataclass(frozen=True)
class InitMemoryEntryV1:
    memory_id: str
    pattern_region_sha: str
    action_id: str
    role: str
    support_group_count: int
    mean_utility_gain: float
    worst_material_harm: float
    evidence_sha: str
    init_corpus_sha: str

    def __post_init__(self) -> None:
        if self.role not in {"utility", "risk"}:
            raise ValueError("Init memory role must be utility or risk")
        if not self.memory_id or not self.action_id:
            raise ValueError("Init memory requires canonical ids")
        if self.support_group_count < 1 or self.support_group_count > 136:
            raise ValueError("Init memory support is outside the Init Corpus")
        for name in ("pattern_region_sha", "evidence_sha", "init_corpus_sha"):
            require_sha(getattr(self, name), name)


@dataclass(frozen=True)
class InitHarnessArtifactV1:
    """The formal initial Harness H0; never an engineering default or empty shell."""

    init_corpus_sha: str
    prereg_sha: str
    pattern_binding_sha: str
    action_eligibility_sha: str
    candidate_grammar_sha: str
    base_harness_semantic_sha: str
    init_evaluation_sha: str
    operator_experience: tuple[InitOperatorExperienceV1, ...]
    seed_policies: tuple[InitSeedPolicyV1, ...]
    program_templates: tuple[InitProgramTemplateV1, ...]
    memory: tuple[InitMemoryEntryV1, ...]
    schema_version: str = "vnext-formal-h0/1"

    def __post_init__(self) -> None:
        for name in (
            "init_corpus_sha", "prereg_sha", "pattern_binding_sha",
            "action_eligibility_sha", "candidate_grammar_sha",
            "base_harness_semantic_sha", "init_evaluation_sha",
        ):
            require_sha(getattr(self, name), name)
        components = {
            "operator_experience": self.operator_experience,
            "seed_policies": self.seed_policies,
            "program_templates": self.program_templates,
            "memory": self.memory,
        }
        empty = [name for name, values in components.items() if not values]
        if empty:
            raise ValueError(f"formal H0 cannot omit Init components: {','.join(empty)}")
        sourced = (
            tuple(row.init_corpus_sha for row in self.operator_experience)
            + tuple(row.init_corpus_sha for row in self.seed_policies)
            + tuple(row.init_corpus_sha for row in self.memory)
        )
        if any(value != self.init_corpus_sha for value in sourced):
            raise ValueError("H0 contains evidence outside its frozen Init Corpus")

    @property
    def semantic_sha(self) -> str:
        return sha256(self)

    @property
    def h0_sha(self) -> str:
        return self.semantic_sha
