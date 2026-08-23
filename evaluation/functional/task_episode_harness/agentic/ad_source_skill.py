"""AD thin adapter over the frozen forecasting source_skill integrator.

#42d reuses authorization_audit / authorized_try_operators / audit_sections /
build_skill_payload / slow_system / signed_summary.  It does not invent a
second Skill mechanism.  The AD constants live here; forecasting defaults
stay on source_skill.py.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.functional.task_episode_harness.agentic import source_skill as ss

SOURCE_SKILL_ID = "source_investigation_ad_v1"
SOURCE_APPLICABILITY: dict[str, Any] = {
    "feature": "task_kind", "op": "==", "value": "anomaly_detection",
}

# Filenames and cohort roots that would leak Source identity into a Skill
# written for a different domain.  audit_sections already owns the check.
SOURCE_COHORT_TOKENS: tuple[str, ...] = (
    "aws", "cloudwatch", "known_cause", "knowncause",
    "exchange", "cpc", "cpm", "nab", "ec2",
    "source_aws_cloudwatch", "source_known_cause",
    "realawscloudwatch", "realknowncause", "realadexchange",
    "ec2_cpu_utilization", "ambient_temperature_system_failure",
    "cpu_utilization_asg_misconfiguration",
    "ec2_request_latency_system_failure",
    "machine_temperature_system_failure",
    "nyc_taxi", "rogue_agent_key_hold",
    "art_daily_jumpsdown",
)

authorization_audit = ss.authorization_audit
authorized_try_operators = ss.authorized_try_operators
audit_sections = ss.audit_sections
signed_summary = ss.signed_summary
SECTIONS = ss.SECTIONS
TRY_ABSTAIN = ss.TRY_ABSTAIN


def slow_system(authorized: Sequence[str]) -> str:
    return ss.slow_system(authorized, skill_id=SOURCE_SKILL_ID)


def build_skill_payload(sections: Mapping[str, Any]) -> dict[str, Any]:
    return ss.build_skill_payload(
        sections,
        skill_id=SOURCE_SKILL_ID,
        applicability=SOURCE_APPLICABILITY,
    )
