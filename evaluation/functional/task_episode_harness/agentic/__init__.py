"""G1: the one logical Runner package that connects the core Pipeline.

Task book: ``docs/AGENTIC_SKILL_HARNESS_GLOBAL_DESIGN_2026-08-19.md`` §13/G1.

Everything mechanical is reused, not rebuilt: ``TTHAAgentCore``'s tool loop,
the frozen ``fast_inspect`` / ``fast_propose`` / ``fast_select`` stage
contracts, ``compile_workflow_proposal``, the Support / delayed evaluator,
the Episode and Target-local Skill lifecycle, ``EditController`` and
``SnapshotStore``.  What is new here is only the wiring plus the two things
the audit proved were missing: a Workspace tool bound to a *named* scoped
series instead of one Runtime-chosen representative, and a compile-time gate
that refuses to broadcast an external-region parameter across action units.
"""
from .gateway import CohortScopePublicToolGateway
from .dispatch import (
    ParameterOwnershipViolation,
    audit_program_parameter_ownership,
    parameter_owner,
    resolve_action_unit_parameters,
)
from .fast_path import FastPathTrace, run_agentic_fast_path
from .runner import PROTOCOL_VERSION, run_g1_pipeline

__all__ = [
    "PROTOCOL_VERSION",
    "CohortScopePublicToolGateway",
    "FastPathTrace",
    "ParameterOwnershipViolation",
    "audit_program_parameter_ownership",
    "parameter_owner",
    "resolve_action_unit_parameters",
    "run_agentic_fast_path",
    "run_g1_pipeline",
]
