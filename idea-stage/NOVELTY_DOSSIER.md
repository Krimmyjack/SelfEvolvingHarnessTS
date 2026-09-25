# Novelty dossier — SelfEvolvingHarnessTS idea cluster

You are a senior ML reviewer checking **novelty**, not generating new ideas.
Requested model: **gpt-6-astra**, reasoning **xhigh**. If the exact serving
variant is not that, disclose it in the first line.

Executor: grok-4.6 (xAI). This is a cross-family Type-B check. Do **not**
acquit if you cannot name your model. Do **not** invent papers. If you cannot
verify a citation, mark `[UNVERIFIED]`.

Read, then judge:
- `/mnt/c/Users/辉/Desktop/Agent/SelfEvolvingHarnessTS-deepseek-guidance-evolution/idea-stage/IDEA_REPORT.md` (especially Ranked Ideas + Novelty Verification)
- `/mnt/c/Users/辉/Desktop/Agent/SelfEvolvingHarnessTS-deepseek-guidance-evolution/docs/STATE_ONE_PAGE_2026-09-03.md` (DEV-AUTO-3 first-fault)
- `/mnt/c/Users/辉/Desktop/Agent/SelfEvolvingHarnessTS-deepseek-guidance-evolution/docs/HEC1_ZERO_LLM_DIAGNOSTIC_CLOSURE_2026-09-04.md`
- `/mnt/c/Users/辉/Desktop/Agent/SelfEvolvingHarnessTS-deepseek-guidance-evolution/docs/SelfEvolvingHarnessTS：整体机制与研究定位文献调研（2026-09-05）.md` §B if needed

No experiments, fits, sealed reads, or code edits.

## Methods to judge (three, because they are the paper cluster)

### Method A — Idea 1: ADD-not-PATCH Skill lineage
Replace in-place PATCH of a single Skill card with ADD a descendant card,
keep the ancestor, and never cascade-revoke the ancestor when the descendant
fails delayed gates. Equal-budget three-arm: A control / B supply-only (already
mints and recalls new cards without overwriting the target) / C′ supply+ADD.
Development data only.

**Core claims to test for novelty**
1. Destructive in-place Skill overwrite + cascade revoke is a first-fault of
   agent skill evolution under delayed consumer feedback.
2. ADD-not-PATCH (keep ancestor) is a distinct mechanism from RewardHarness
   whole-library rollback, DGM open archive, and Progressive Neural Networks.
3. The finding (positive or negative) is publishable on TS Data-Readiness
   Harness cards, not just “apply continual learning to agents”.

### Method B — Idea 19: Delta-stable one-edit workflow repair
After a natural delayed conflict, allow exactly one legal DSL edit
(swap one operator / add-or-drop one impute step / swap order / one legal
parameter). Learn/select using the **edit effect** \(d_e(P\to P')\), not
absolute program utility \(g_e(P)\). Independent next-face validation, then
later re-encounter. Frozen program menu; no new operators.

**Core claims**
1. Local edit effects are more stable across faces/re-encounters than absolute
   workflow utility, enabling a revise→validate→survive→reencounter chain.
2. This is not just AFlow/AegisTS local search if (and only if) transferred
   edit knowledge beats equal-budget fresh search.
3. One-edit + delayed four-line risk gates on TS consumers is a distinct
   experimental setting, not a new search algorithm.

### Method C — Idea 13: Scope-language expressivity ceiling
Analyst-only enumeration of the **frozen** 12-feature predicate grammar:
best safe subset expressible vs unconstrained outcome-side upper bound,
same four risk lines. Constructive collisions. No fitted Targeter, no new
Scope installed.

**Core claims**
1. Some benefit/harm cases are indistinguishable under the allowed Scope
   vocabulary, imposing a ceiling prompting cannot beat.
2. Calibration failure (0/5 CALIBRATED) is not itself an expressivity proof;
   constructive collisions are required.
3. This is a contract-specific diagnostic, not a general theory of languages.

## Candidate papers already on the table (verify, do not invent)

Must treat these as the closest-work shortlist. Confirm overlap vs delta.

| Paper | ID | Why listed |
|---|---|---|
| Darwin Gödel Machine | arXiv:2505.22954 | open archive of non-best agents |
| Progressive Neural Networks | arXiv:1606.04671 | freeze-old add-new columns |
| Experience Replay for Continual Learning | arXiv:1811.11682 | keep old experience |
| RewardHarness | arXiv:2605.08703 | skill/tool library, val accept or rollback, gains from pruning |
| AFlow | arXiv:2410.10762 | coded workflow + MCTS + execution feedback |
| AegisTS | arXiv:2605.04902 | hierarchical RL for MTS cleaning |
| Self-Harness | arXiv:2606.09498 | harness self-improvement |
| Evo-Harness | arXiv:2608.15071 | context → skill harness compilation |
| TTHE | arXiv:2607.08124 | test-time harness evolution, proxy reliability |
| SPIBB | arXiv:1712.06924 | bootstrap to baseline in uncertain regions |
| ADAS / Meta Agent Search | ICLR 2025 | archive of agent code |
| Conformal prediction beyond exchangeability | Barber et al. 2023 AOS | only if discussing Idea 21; not required for A/B/C |

Also check last 6 months for: “don’t overwrite tools/skills”, “skill memory
catastrophic forgetting”, “agent skill versioning”, “workflow one-edit
repair”, “scope predicate expressivity”. If search is unavailable, say so
explicitly; do not hallucinate concurrent preprints.

## Required output

For **each** of A, B, C:

```
### Method [A/B/C]
- One-sentence restatement
- Core claims: HIGH/MEDIUM/LOW novelty each, closest paper, delta
- Closest prior work table (only verified or [UNVERIFIED])
- Overall novelty 1–10
- Recommendation: PROCEED / PROCEED WITH CAUTION / ABANDON as a *paper*
  (diagnostics can PROCEED as repo work even if ABANDON as paper)
- Key differentiator
- Reviewer-cited prior-work risk
- Suggested positioning (one paragraph)
```

Then:
- Which **one** method should be the paper’s dominant contribution?
- Which must remain supporting diagnostics (not co-first claims)?
- Any method that is already published in all but the application domain?

Be brutal. “Apply X to time series agents” is not enough.
