from SelfEvolvingHarnessTS.memory import EvidenceRecord, EvidenceStore


def _record(cell="forecast|snrLow|miss", batch="b1"):
    return EvidenceRecord(
        conditioning_key={"task": {"type": "forecast"}, "cell_id": cell},
        cell_id=cell,
        harness_version=3,
        program={"note": "tmpl:v_none", "steps": []},
        execution_trace=[{"op": "noop"}],
        verification_result={"passed": True, "output_status": "executed"},
        batch_id=batch,
        routing={"selected_action": "v_none", "safety": {"accepted": True}},
    )


def test_evidence_store_jsonl_persistence_roundtrip(tmp_path):
    path = tmp_path / "evidence.jsonl"
    store = EvidenceStore(persist_path=path)
    rec = _record()

    store.write(rec)
    loaded = EvidenceStore.from_jsonl(path)

    assert len(loaded) == 1
    row = loaded.query_by_cell(rec.cell_id)[0]
    assert row.cell_id == rec.cell_id
    assert row.program["note"] == "tmpl:v_none"
    assert row.routing["selected_action"] == "v_none"
    assert loaded.replay_contract()["schema"] == "evidence_store_replay_v1"
    assert loaded.replay_contract()["n_records"] == 1


def test_evidence_store_save_jsonl_exports_in_memory_records(tmp_path):
    path = tmp_path / "export" / "records.jsonl"
    store = EvidenceStore()
    store.write(_record(cell="forecast|snrHigh|full", batch="b2"))

    store.save_jsonl(path)
    loaded = EvidenceStore.from_jsonl(path)

    assert len(loaded) == 1
    assert loaded.replay_contract()["cells"] == ["forecast|snrHigh|full"]
    assert loaded.replay_contract()["batches"] == ["b2"]
