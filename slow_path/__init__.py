"""Slow-path public API."""
from .schedule import edit_budget, CellSchedule
from .batch_builder import BatchBuilder, CellSample, cell_sample_from_raw_series, make_eval_sample
from .validator import Validator, ValidationOutcome, grounded_val_loss
from .mining import mine_weakness, mine_strength, WeaknessReport, StrengthReport
from .proposer import Proposer
from .merger import Merger
from .attribution import AttributionStore, ops_credit
from .candidate_log import CandidateLogger, judge_fingerprint
from .evolve import Evolver, RoundResult
from .evidence_miner import ActionEvidenceStats, DeploymentEvidenceMiner, DeploymentEvidenceSummary, suggest_slow_path_proposals
from .proposal_schema import SlowProposal, validate_slow_proposal
from .promotion import ProposalValidationOutcome, PromotionGate, compile_slow_proposal_to_edit
from .forward_transfer import (
    load_transfer_log, per_domain_points, build_curves, forward_transfer_verdict,
    analyze, DomainPoint,
)

__all__ = [
    "edit_budget", "CellSchedule",
    "BatchBuilder", "CellSample", "cell_sample_from_raw_series", "make_eval_sample",
    "Validator", "ValidationOutcome", "grounded_val_loss",
    "mine_weakness", "mine_strength", "WeaknessReport", "StrengthReport",
    "Proposer", "Merger", "AttributionStore", "ops_credit", "Evolver", "RoundResult",
    "CandidateLogger", "judge_fingerprint",
    "ActionEvidenceStats", "DeploymentEvidenceMiner", "DeploymentEvidenceSummary",
    "suggest_slow_path_proposals", "SlowProposal", "validate_slow_proposal",
    "ProposalValidationOutcome", "PromotionGate", "compile_slow_proposal_to_edit",
    "load_transfer_log", "per_domain_points", "build_curves", "forward_transfer_verdict",
    "analyze", "DomainPoint",
]
