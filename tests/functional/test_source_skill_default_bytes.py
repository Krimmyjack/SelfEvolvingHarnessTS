"""#42d r3: default source_skill path stays byte-identical to the pre-parameterization freeze."""
from __future__ import annotations

import hashlib
import json

from evaluation.functional.task_episode_harness.agentic import source_skill as ss


# Frozen before the optional-argument patch.  Recomputed only if the
# forecasting default text itself is authorized to move.
_PAYLOAD_SHA = "86603236bb649ac17d3ab6aa4cdf16e8788bcd922ca057bd32cf12d7a1323196"
_SLOW_EMPTY_SHA = "ed6fc16d3842741a1c6fe92b1d6ade934846f74dabd5d81e902e294b5e342cad"
_SLOW_ONE_SHA = "0fbc86db725fb548b6acf0d9500092035a78fd86951349c6907196a9e9394fb6"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fixture_sections() -> dict[str, str]:
    sections = {name: "placeholder sentence about task_kind only."
                for name in ss.SECTIONS}
    sections["TRY"] = ss.TRY_ABSTAIN
    return sections


def test_default_build_skill_payload_bytes_match_pre_patch() -> None:
    payload = ss.build_skill_payload(_fixture_sections())
    assert payload["skill_id"] == "source_investigation_v1"
    assert payload["observable_applicability"] == {
        "feature": "task_kind", "op": "==", "value": "forecast",
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    assert _sha(encoded) == _PAYLOAD_SHA


def test_default_slow_system_bytes_match_pre_patch() -> None:
    assert _sha(ss.slow_system(())) == _SLOW_EMPTY_SHA
    assert _sha(ss.slow_system(["outlier_iqr"])) == _SLOW_ONE_SHA


def test_override_does_not_change_default_constants() -> None:
    payload = ss.build_skill_payload(
        _fixture_sections(),
        skill_id="source_investigation_ad_v1",
        applicability={"feature": "task_kind", "op": "==",
                       "value": "anomaly_detection"},
    )
    assert payload["skill_id"] == "source_investigation_ad_v1"
    assert ss.SOURCE_SKILL_ID == "source_investigation_v1"
    assert ss.SOURCE_APPLICABILITY["value"] == "forecast"
    assert _sha(ss.slow_system(())) == _SLOW_EMPTY_SHA
