from SelfEvolvingHarnessTS.policy.skill_memory_composer import (
    TypedCandidate,
    compose_skill_memory_candidate,
    parse_typed_candidate,
)


def _packet():
    return {
        "schema": "skill_memory_evidence_packet_v1",
        "skills": [
            {"name": "median_smooth", "allowed_actions": ["v_median", "f0_median_w9"]},
            {"name": "identity", "allowed_actions": ["v_none"]},
        ],
        "action_menu": {"allowed_actions": ["v_none", "v_median", "f0_median_w9"]},
        "candidate_schema": {"type": "typed_candidate_v1"},
        "risk_constraints": [],
    }


def test_parse_typed_candidate_accepts_allowed_skill_action_candidate():
    raw = '{"skill_id":"median_smooth","action_id":"v_median","rationale":"low snr","evidence_refs":["m1"]}'

    cand = parse_typed_candidate(raw, _packet())

    assert isinstance(cand, TypedCandidate)
    assert cand.skill_id == "median_smooth"
    assert cand.action_id == "v_median"
    assert cand.to_dict()["ProgramSpec"] == {}
    assert cand.to_dict()["evidence_refs"] == ["m1"]


def test_parse_typed_candidate_rejects_unknown_skill_or_action():
    assert parse_typed_candidate('{"skill_id":"unknown","action_id":"v_median"}', _packet()) is None
    assert parse_typed_candidate('{"skill_id":"median_smooth","action_id":"v_stl"}', _packet()) is None
    assert parse_typed_candidate('{"skill_id":"identity","action_id":"v_median"}', _packet()) is None


def test_parse_typed_candidate_accepts_abstain_to_raw_risk_candidate():
    cand = parse_typed_candidate(
        '{"abstain_to_raw":true,"risk_rule":{"rule_id":"weak_support_raw","op":"abstain"}}',
        _packet(),
    )

    assert cand is not None
    assert cand.abstain_to_raw is True
    assert cand.risk_rule["rule_id"] == "weak_support_raw"


def test_compose_skill_memory_candidate_uses_packet_and_parser():
    calls = []

    def stub(system, user, nonce=0):
        calls.append((system, user, nonce))
        return '{"skill_id":"median_smooth","action_id":"f0_median_w9","ProgramSpec":{"steps":[]}}'

    cand = compose_skill_memory_candidate(_packet(), stub)

    assert cand is not None
    assert cand.action_id == "f0_median_w9"
    assert cand.program_spec == {"steps": []}
    assert "skill_memory_evidence_packet_v1" in calls[0][1]