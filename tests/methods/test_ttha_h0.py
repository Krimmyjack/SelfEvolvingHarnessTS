import json
from pathlib import Path

import pytest

from SelfEvolvingHarnessTS.contracts.harness import SkillKind
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore


H0 = Path(__file__).resolve().parents[2] / "methods" / "ttha" / "harness" / "h0"


def _copy_tree(source_root: Path, target_root: Path) -> None:
    target_root.mkdir()
    for source in source_root.rglob("*"):
        if source.is_file():
            target = target_root / source.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())


def test_h0_is_stable_procedural_and_domain_naive():
    first = compile_snapshot(H0)
    second = compile_snapshot(H0)
    assert first.harness_content_sha == second.harness_content_sha
    assert first.runtime_bundle_sha == second.runtime_bundle_sha
    assert first.memories == ()
    assert [skill.skill_kind for skill in first.skills] == [
        SkillKind.BOOTSTRAP_PROCEDURE,
        SkillKind.BOOTSTRAP_PROCEDURE,
        SkillKind.BOOTSTRAP_PROCEDURE,
    ]
    forbidden = ("missing ->", "outlier ->", "impute_linear", "winsorize")
    resolved = first.instruction + "\n" + "\n".join(skill.body for skill in first.skills)
    assert not any(token in resolved for token in forbidden)


def test_h0_lock_mismatch_fails_loudly(tmp_path):
    root = tmp_path / "h0"
    _copy_tree(H0, root)
    (root / "instruction.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="lock mismatch"):
        compile_snapshot(root)


def test_formatting_variants_keep_semantic_sha_but_semantic_edits_do_not(tmp_path):
    original = compile_snapshot(H0)
    equivalent = tmp_path / "equivalent"
    _copy_tree(H0, equivalent)
    instruction = (equivalent / "instruction.md").read_text(encoding="utf-8")
    (equivalent / "instruction.md").write_bytes(
        b"\xef\xbb\xbf" + instruction.replace("\n", "\r\n").encode("utf-8")
    )
    for path in equivalent.rglob("*.json"):
        if path.name == "snapshot.lock.json":
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = dict(reversed(tuple(value.items())))
        path.write_text(json.dumps(value, indent=3, ensure_ascii=False), encoding="utf-8")
    assert compile_snapshot(equivalent).harness_content_sha == original.harness_content_sha

    changed = tmp_path / "changed"
    _copy_tree(H0, changed)
    (changed / "candidate_policy.json").write_text(
        json.dumps(
                {
                    **json.loads((changed / "candidate_policy.json").read_text(encoding="utf-8")),
                    "proposal_guidance": "Supply one minimal public-evidence candidate.",
                }
        ),
        encoding="utf-8",
    )
    assert compile_snapshot(changed, verify_lock=False).harness_content_sha != original.harness_content_sha


def test_snapshot_store_materializes_without_mutating_h0(tmp_path):
    before = {path.relative_to(H0): path.read_bytes() for path in H0.rglob("*") if path.is_file()}
    snapshot = compile_snapshot(H0)
    store = SnapshotStore(tmp_path / "runs" / "minipipe" / "harness_snapshots")
    materialized = store.materialize(snapshot)
    repeated = store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    assert materialized.root.name == snapshot.runtime_bundle_sha
    assert repeated.root == materialized.root
    assert materialized.parent_runtime_bundle_sha is None
    assert compile_snapshot(materialized.root).runtime_bundle_sha == snapshot.runtime_bundle_sha
    assert json.loads(store.active_path.read_text(encoding="utf-8")) == {
        "runtime_bundle_sha": snapshot.runtime_bundle_sha
    }
    assert before == {
        path.relative_to(H0): path.read_bytes() for path in H0.rglob("*") if path.is_file()
    }
