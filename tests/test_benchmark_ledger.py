from __future__ import annotations

import dataclasses
import json

import pytest

from SelfEvolvingHarnessTS.benchmark.ledger import (
    CampaignEntry,
    CampaignLedger,
    CampaignManifest,
    CampaignStateError,
    MethodResultStatus,
    ResumeBinding,
)


def _sha(character: str) -> str:
    return character * 64


def _manifest(roster=("m",)):
    entries = tuple(
        CampaignEntry(entry_id, _sha(chr(97 + i)), i, budget=10)
        for i, entry_id in enumerate(roster)
    )
    return CampaignManifest(
        campaign_id="campaign-1",
        benchmark_version="benchmark-v0",
        input_manifest_sha=_sha("1"),
        materialization_sha=_sha("2"),
        runner_code_sha=_sha("3"),
        entries=entries,
    )


def frozen_ledger(tmp_path, roster=("m",)):
    return CampaignLedger(tmp_path / "campaign.jsonl", _manifest(roster))


def exact_resume_binding(entry_id="m"):
    return ResumeBinding(
        campaign_id="campaign-1",
        run_id="run-1",
        entry_id=entry_id,
        method_code_sha=_sha("a"),
        runner_code_sha=_sha("3"),
        input_manifest_sha=_sha("1"),
        materialization_sha=_sha("2"),
        checkpoint_sha=_sha("4"),
    )


def test_unseal_and_access_are_durable_before_read(tmp_path):
    with frozen_ledger(tmp_path) as ledger:
        ledger.unseal()
        ledger.record_access("m", "run-1")
        assert [event["event"] for event in ledger.events()] == [
            "campaign_freeze",
            "unseal",
            "method_access",
        ]
        disk = [json.loads(line) for line in ledger.path.read_text("utf-8").splitlines()]
        assert [event["event"] for event in disk] == [
            "campaign_freeze",
            "unseal",
            "method_access",
        ]


def test_unrostered_oracle_cannot_access_final(tmp_path):
    with frozen_ledger(tmp_path, roster=("raw", "method")) as ledger:
        ledger.unseal()
        with pytest.raises(CampaignStateError, match="roster"):
            ledger.record_access("oracle_insample", "run-o")


def test_method_invalid_is_terminal(tmp_path):
    with frozen_ledger(tmp_path, roster=("m",)) as ledger:
        ledger.unseal()
        ledger.record_access("m", "run-1")
        ledger.record_result("m", "run-1", MethodResultStatus.INVALID, _sha("5"))
        with pytest.raises(CampaignStateError, match="terminal"):
            ledger.record_access("m", "run-2")


def test_exact_infrastructure_resume_is_idempotent(tmp_path):
    binding = exact_resume_binding()
    with frozen_ledger(tmp_path, roster=(binding.entry_id,)) as ledger:
        ledger.unseal()
        ledger.record_access(binding.entry_id, binding.run_id)
        ledger.record_result(
            binding.entry_id,
            binding.run_id,
            MethodResultStatus.INFRA_INTERRUPTED,
            binding.checkpoint_sha,
            resume_binding=binding,
        )
        assert ledger.resume(binding) == binding.run_id
        assert ledger.resume(binding) == binding.run_id
        changed = dataclasses.replace(binding, runner_code_sha=_sha("9"))
        with pytest.raises(CampaignStateError, match="resume binding"):
            ledger.resume(changed)


def test_replay_detects_tampering(tmp_path):
    path = tmp_path / "campaign.jsonl"
    with CampaignLedger(path, _manifest()) as ledger:
        ledger.unseal()
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    rows[-1]["campaign_id"] = "tampered"
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", "utf-8")
    with pytest.raises(CampaignStateError, match="tamper|hash|campaign"):
        CampaignLedger(path, _manifest())


def test_campaign_closes_only_after_full_roster_terminal(tmp_path):
    with frozen_ledger(tmp_path, roster=("a", "b")) as ledger:
        ledger.unseal()
        ledger.record_access("a", "run-a")
        ledger.record_result("a", "run-a", MethodResultStatus.COMPLETE, _sha("6"))
        with pytest.raises(CampaignStateError, match="roster"):
            ledger.close_campaign()
        ledger.record_access("b", "run-b")
        ledger.record_result("b", "run-b", MethodResultStatus.FAILED_TIMEOUT, _sha("7"))
        ledger.close_campaign()
        with pytest.raises(CampaignStateError, match="closed"):
            ledger.record_access("a", "run-new")

