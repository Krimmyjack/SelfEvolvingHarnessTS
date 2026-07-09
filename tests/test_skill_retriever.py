from SelfEvolvingHarnessTS.policy.action_spec import ActionMenu, ActionSpec
from SelfEvolvingHarnessTS.policy.skill_retriever import retrieve_skill_cards, retrieve_skills
from SelfEvolvingHarnessTS.policy.skills import SKILLS_V1


def _record(**overrides):
    record = {
        "uid": "u1",
        "cell": "forecast|snrLow|miss",
        "snr": -6.0,
        "miss_rate": 0.12,
        "X_p": [24.0, 0.2, 0.1, 0.4, 0.0, 0.6, 0.8, 0.05],
    }
    record.update(overrides)
    return record


def test_retrieve_skills_prioritizes_denoising_for_low_snr_missing_record():
    matches = retrieve_skills(_record(), skills=SKILLS_V1, top_k=3)

    names = [m.skill.name for m in matches]
    assert "median_smooth" in names
    assert names.index("median_smooth") < names.index("identity") if "identity" in names else True
    assert matches[0].score >= matches[-1].score
    assert matches[0].reasons


def test_retrieve_skills_prioritizes_identity_for_clean_high_support_record():
    matches = retrieve_skills(
        _record(cell="forecast|snrHigh|full", snr=15.0, miss_rate=0.0, X_p=[24.0, 0.1, 0.05, 0.2, 0.0, 0.2, 0.1, 0.0]),
        skills=SKILLS_V1,
        top_k=2,
    )

    assert matches[0].skill.name == "identity"


def test_retrieve_skills_filters_actions_by_menu():
    menu = ActionMenu(
        "tiny",
        [ActionSpec("v_none", steps=(), task_constraints=("forecast",), model_constraints=None)],
    )

    matches = retrieve_skills(_record(), skills=SKILLS_V1, action_menu=menu, top_k=10)

    assert [m.skill.name for m in matches] == ["identity"]
    assert matches[0].allowed_actions == ["v_none"]


def test_retrieve_skill_cards_are_packet_ready_and_stable():
    cards = retrieve_skill_cards(_record(), skills=SKILLS_V1, top_k=2)

    assert [card["rank"] for card in cards] == [1, 2]
    assert all("name" in card and "score" in card and "allowed_actions" in card for card in cards)
    assert cards == retrieve_skill_cards(_record(), skills=SKILLS_V1, top_k=2)