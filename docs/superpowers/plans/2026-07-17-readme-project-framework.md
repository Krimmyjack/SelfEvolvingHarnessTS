# README Project Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concise top-level project framework and active data flow to `README.md`, then prepare the verified branch for the already requested local merge.

**Architecture:** Replace only the existing `Active structure` bullets with a commented directory tree and a compact main-flow diagram. Keep verification commands, historical recovery, runtime code, and tests unchanged.

**Tech Stack:** Markdown, Git, pytest, Bash.

## Global Constraints

- Show top-level directories and responsibilities only; do not enumerate all 114 tracked files.
- Keep the existing verification commands and historical-recovery section unchanged.
- Describe H_ref_v02 as current and TTHA as future work.
- Treat Benchmark-v0.2 as auxiliary evaluation, not the product main line.
- Do not stash, overwrite, stage, or delete the other Agent's main-worktree changes.

---

### Task 1: Document the Converged Project Framework

**Files:**

- Modify: `README.md`

**Interfaces:**

- Consumes: the current top-level directory layout.
- Produces: an onboarding view of directory ownership and the canonical execution flow.

- [ ] **Step 1: Verify every directory intended for the README exists**

Run:

```bash
for directory in contracts conditioning operators runtime methods evaluation artifacts experiments tests docs; do
  test -d "$directory" || exit 1
done
```

Expected: exit code 0 and no output.

- [ ] **Step 2: Replace the `Active structure` section**

Use this exact content between the introductory paragraph and `## Verification`:

````markdown
## Current project framework

```text
SelfEvolvingHarnessTS/
├── contracts/       # Task, Program, and Method public contracts
├── conditioning/    # Time-series features, period detection, and condition routing
├── operators/       # Canonical operator implementations and the sole registry
├── runtime/         # Sole executor, fast path, trace, and error model
├── methods/         # Current H_ref_v02 reference; future TTHA active method line
├── evaluation/      # Auxiliary Benchmark-v0.2 evaluation environment
├── artifacts/       # Frozen evidence and architecture-cleanup manifests
├── experiments/     # Git recovery instructions; no importable historical source
├── tests/           # Functional tests grouped by contracts, components, and integration
└── docs/            # Current architecture designs and implementation records
```

The active execution path is:

```text
TaskSpec -> Method -> Runtime -> Operators
                   ^
              Conditioning
```
````

- [ ] **Step 3: Validate the documentation against the tree**

Run:

```bash
for directory in contracts conditioning operators runtime methods evaluation artifacts experiments tests docs; do
  rg -q "├── ${directory}/|└── ${directory}/" README.md || exit 1
done
if rg -n '├── (benchmark|config|p6|policy|sandbox|results|data)/' README.md; then
  exit 1
fi
git diff --check -- README.md
```

Expected: exit code 0, no retired directory match, and no whitespace error.

- [ ] **Step 4: Run the retained functional suite**

Run from the worktree parent:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_readme_framework
```

Expected: all retained tests pass.

- [ ] **Step 5: Commit the README**

Run:

```bash
git add README.md
git commit -m "docs: show current project framework"
```

Expected: one README-only implementation commit.

## Handoff

After verification, use `superpowers:finishing-a-development-branch`. The requested choice
is local merge to `main`, but stop before merging if the main worktree's existing tracked
changes would be overwritten; do not use automatic stash without new user approval.
