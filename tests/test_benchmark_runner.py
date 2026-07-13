from __future__ import annotations

import hashlib

import pytest

from SelfEvolvingHarnessTS.benchmark.ledger import (
    CampaignEntry,
    MethodResultStatus,
)
from SelfEvolvingHarnessTS.benchmark.runner import (
    BenchmarkRunner,
    FINAL_ROSTER_REQUIRED,
    RunnerGateError,
    validate_campaign_roster,
)


def _sha(character="a"):
    return character * 64


class _Ledger:
    def __init__(self):
        self.calls = []

    def record_access(self, entry_id, run_id):
        self.calls.append(("access", entry_id, run_id))

    def record_result(self, entry_id, run_id, status, digest, **kwargs):
        self.calls.append(("result", entry_id, run_id, status, digest, kwargs))


class _FinalStore:
    def __init__(self, calls):
        self.calls = calls

    def load(self, entry_id):
        self.calls.append(("read", entry_id))
        return [entry_id]


def _runner():
    calls = []
    ledger = _Ledger()
    runner = BenchmarkRunner(
        final_store=_FinalStore(calls),
        ledger=ledger,
        input_manifest_sha=_sha("1"),
        materialization_sha=_sha("2"),
        runner_code_sha=_sha("3"),
    )
    return runner, ledger, calls


def test_final_loader_commits_access_before_loading():
    runner, ledger, reads = _runner()
    runner.load_final_for_entry("method", "run-1")
    assert ledger.calls[0][:3] == ("access", "method", "run-1")
    assert reads == [("read", "method")]


def _roster(include_custom=False):
    ids = sorted(FINAL_ROSTER_REQUIRED | ({"method"} if include_custom else set()))
    return tuple(
        CampaignEntry(
            entry_id,
            _sha(chr(97 + index)),
            index,
            budget=10,
            dry_run_sha=_sha("d") if entry_id == "method" else None,
            confirmation_sha=_sha("e") if entry_id == "method" else None,
        )
        for index, entry_id in enumerate(ids)
    )


def test_final_requires_dev_report_timeouts_support_gates_and_full_roster(tmp_path):
    runner, _, _ = _runner()
    runner.dev_discrimination_sha = None
    with pytest.raises(RunnerGateError, match="Dev discrimination"):
        runner.freeze_campaign(_roster(), campaign_id="c", ledger_path=tmp_path / "c.jsonl")
    runner.dev_discrimination_sha = _sha("4")
    with pytest.raises(RunnerGateError, match="timeout"):
        runner.freeze_campaign(_roster(), campaign_id="c", ledger_path=tmp_path / "c.jsonl")
    runner.calibrate_timeouts([1.0, 2.0], [3.0, 4.0])
    with pytest.raises(RunnerGateError, match="required entries"):
        validate_campaign_roster(_roster()[1:])
    broken = list(_roster(include_custom=True))
    method = next(index for index, entry in enumerate(broken) if entry.entry_id == "method")
    broken[method] = CampaignEntry("method", broken[method].method_code_sha, method, 10)
    with pytest.raises(RunnerGateError, match="Support"):
        runner.freeze_campaign(tuple(broken), campaign_id="c", ledger_path=tmp_path / "c.jsonl")


def test_timeout_calibration_is_numeric_two_times_p95():
    runner, _, _ = _runner()
    prepare, trainer = runner.calibrate_timeouts([1.0, 2.0, 3.0], [2.0, 4.0, 8.0])
    assert prepare == pytest.approx(2 * 2.9)
    assert trainer == pytest.approx(2 * 7.6)


def test_method_exception_is_terminal_invalid():
    runner, ledger, _ = _runner()
    status = runner.run_final_entry(
        "method",
        "run-1",
        lambda _: (_ for _ in ()).throw(ValueError("bad method output")),
    )
    assert status is MethodResultStatus.INVALID
    result = ledger.calls[-1]
    assert result[0] == "result"
    assert result[3] is MethodResultStatus.INVALID
    assert len(result[4]) == hashlib.sha256().digest_size * 2

