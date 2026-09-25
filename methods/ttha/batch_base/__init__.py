"""Batch-level construction base shared by the P (fixed-flow policy) and W (tool-loop) controllers.

The seven controller-facing operations of docs/BATCH_RESEARCH_WORKFLOW_FRAMEWORK §4 map onto:

  overview          context.open_job(..., stage="material").overview()
  inspect_data      JobContext.inspect_data(entity_indices, kind, sub_range)
  build_material    policy.compile_policy(policy, overview["entities"]) -> materials.build(ctx, assignment, material_id)
  inspect_material  materials.inspect_material(ctx, ref)
  evaluate          train.evaluate(ctx, ref, seeds, ledger, repo_root)      (C_A only; subprocess per seed)
  compare           feedback.paired(cells_a, cells_b, block)
  commit            commit.commit(...); then commit.open_c_b / freeze_e / score_e in that order

Stage permissions are physical (context.STAGE_ROWS): a worker opened at the `evaluate` stage never
has a C_B or E row in memory. No LLM, no hashing, no selection rule lives here.
"""
from . import augment, budget, commit, context, data, feedback, llm, materials, observe, policy, spec, train  # noqa: F401

__all__ = ["augment", "budget", "commit", "context", "data", "feedback", "llm", "materials", "observe", "policy", "spec", "train"]
