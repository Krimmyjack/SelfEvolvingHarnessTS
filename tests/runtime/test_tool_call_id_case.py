"""call_id is a correlation identifier, so its case carries no meaning.

G2 shakedown (T233 rerun): four Task Episodes ended in AGENT_PROTOCOL_ERROR
because the model built a call_id out of the series it was about to inspect --
``inspect_T234_summary`` -- and series uids are uppercase.  The envelope
pattern accepted lowercase only.

Downstream, call_id is used in exactly two ways: exact-equality duplicate
detection within one stage, and verbatim echo into tool-result/1, whose schema
puts no pattern on it at all.  No lookup, no path, no case-folded comparison.
So the case restriction was protecting nothing and cost four Tasks.

tool_name is a different matter -- it indexes the declared tool set -- and is
deliberately left lowercase-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    parse_agent_envelope,
)


def _tool_request(call_id: str, tool_name: str = "summarize_series") -> str:
    return json.dumps(
        {
            "schema_version": "agent-envelope/1",
            "kind": "tool_request",
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": {"series_uid": "T234"},
        }
    )


def _status(call_id: str, tool_name: str = "summarize_series") -> str:
    return parse_agent_envelope(_tool_request(call_id, tool_name))[1]


def test_call_id_may_carry_the_uppercase_series_it_refers_to():
    # The exact ids the live run was rejected for.
    for call_id in (
        "inspect_T234_summary",
        "inspect_summary_T234",
        "inspect_summary_T234_retry",
    ):
        assert _status(call_id) == "VALID_AGENT_ENVELOPE", call_id


def test_relaxation_is_case_only_and_nothing_else():
    assert _status("inspect_summary_1") == "VALID_AGENT_ENVELOPE"
    for rejected in ("9leading_digit", "_leading_underscore", "has space",
                     "has.dot", "has/slash", ""):
        assert _status(rejected) == "INVALID_AGENT_ENVELOPE", rejected


def test_tool_name_stays_lowercase_because_it_indexes_the_tool_set():
    assert _status("ok", tool_name="Summarize_Series") == "INVALID_AGENT_ENVELOPE"
    assert _status("ok", tool_name="summarize_series") == "VALID_AGENT_ENVELOPE"


def test_duplicate_detection_still_distinguishes_ids_that_differ_only_in_case():
    """Exact equality is the rule, so two distinct ids stay distinct."""
    first = parse_agent_envelope(_tool_request("callA"))[0]
    second = parse_agent_envelope(_tool_request("calla"))[0]
    assert first is not None and second is not None
    assert first["call_id"] != second["call_id"]
