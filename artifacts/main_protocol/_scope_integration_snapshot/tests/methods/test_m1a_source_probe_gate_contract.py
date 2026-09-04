"""M1a: the deterministic Source-probe gate contract, frozen before its code.

Three assertions, taken verbatim from the frozen M1a gating semantics, and
nothing else:

1. a second unconfirmed Source probe on the same key is refused;
2. a failed first probe silences the clause for that key;
3. the Target-only candidate channel survives that silencing.

The Context cell identity is an **opaque string** everywhere below.  M0 is
about to re-cut what a cell is, so this module never constructs, parses,
compares or interprets cell semantics -- it only passes the key through.  The
same discipline applies to ``skill_version`` and ``family``: the repository has
no ``skill_version`` field today (see
``artifacts/functional/e2/m1a_static_prep_review.md`` §d.1), so treating it as
opaque is the only reading that survives M0 and whatever binds that identity
later.

The gate does not exist yet and is deliberately not written here.  The module
skips itself until the implementation lands, so the existing suite stays green.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - the skip below is the point
    from evaluation.functional.task_episode_harness.agentic.source_probe_gate import (  # noqa: E501
        SourceProbeGate,
        partition_probe_channel,
    )
except ImportError:  # pragma: no cover
    SourceProbeGate = None
    partition_probe_channel = None

pytestmark = pytest.mark.skipif(
    SourceProbeGate is None or partition_probe_channel is None,
    reason="M1a gate contract frozen ahead of its implementation",
)

# Opaque identities.  Nothing in this module reads structure out of any of them.
SKILL_VERSION = "opaque-skill-version"
CELL_KEY = "opaque-context-cell"
SOURCE_FAMILY = "opaque-source-family"
TARGET_ONLY_FAMILY = "opaque-target-only-family"


def _key() -> dict[str, str]:
    return {
        "skill_version": SKILL_VERSION,
        "cell_key": CELL_KEY,
        "family": SOURCE_FAMILY,
    }


def test_a_second_unconfirmed_source_probe_on_the_same_key_is_refused() -> None:
    """One unconfirmed Source-triggered probe per key, and only one.

    "Unconfirmed" is the state a probed Episode is already in before any
    delayed window runs, so the second probe is refused while the first is
    still awaiting confirmation rather than after it has been graded.
    """
    gate = SourceProbeGate()

    assert gate.may_probe(**_key()) is True
    gate.record_probe_outcome(**_key(), confirmed=False)

    assert gate.may_probe(**_key()) is False


def test_a_failed_first_probe_silences_the_clause_for_that_key() -> None:
    """Silencing is deterministic and scoped to the key, not to the Skill."""
    gate = SourceProbeGate()
    gate.record_probe_outcome(**_key(), confirmed=False)

    assert gate.is_silenced(**_key()) is True


def test_the_target_only_channel_survives_a_silenced_source_clause() -> None:
    """Silencing a clause never removes the candidates that do not need it.

    This is also what keeps the gate compatible with R1, whose contract states
    that a deprioritized candidate "is never dropped, never blocked, and stays
    selectable".
    """
    gate = SourceProbeGate()
    gate.record_probe_outcome(**_key(), confirmed=False)

    compiled_rows = [
        {"candidate_id": "source-attributed", "family": SOURCE_FAMILY},
        {"candidate_id": "target-only", "family": TARGET_ONLY_FAMILY},
    ]
    channels = partition_probe_channel(
        compiled_rows,
        gate=gate,
        skill_version=SKILL_VERSION,
        cell_key=CELL_KEY,
        source_families=frozenset({SOURCE_FAMILY}),
    )

    assert [row["candidate_id"] for row in channels["target_only"]] == [
        "target-only"
    ]
