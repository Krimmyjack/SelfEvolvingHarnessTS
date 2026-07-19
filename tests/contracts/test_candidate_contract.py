from SelfEvolvingHarnessTS.contracts.candidate import Candidate, CandidateKind
from SelfEvolvingHarnessTS.contracts.program import Program


def test_identity_is_tagged_and_has_no_program():
    candidate = Candidate.identity()
    assert candidate.candidate_id == "identity"
    assert candidate.kind is CandidateKind.IDENTITY
    assert candidate.program is None


def test_program_candidate_requires_non_empty_program():
    program = Program.from_steps([("impute_linear", {})], source="agent")
    candidate = Candidate.program_candidate("agent-0", program, source="agent")
    assert candidate.kind is CandidateKind.PROGRAM
    assert candidate.program is program
