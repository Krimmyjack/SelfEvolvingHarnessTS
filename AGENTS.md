# Project execution priorities

For this repository, optimize work in this order:

1. Complete the user-visible and research-critical functionality.
2. Converge duplicated branches into clear, reusable components.
3. Improve actual method quality, runtime performance, and iteration speed.
4. Preserve only compatibility boundaries that are still needed by active workflows.

Tests are evidence for functionality, not an end product. Add the smallest useful set of
tests that proves a requested feature works, protects a critical compatibility boundary,
or detects a meaningful performance regression. Do not spend substantial effort building
benchmark infrastructure, exhaustive low-risk edge-case suites, or ceremonial validation
unless the user explicitly asks for that rigor or the change affects frozen scientific
evidence.

When rigor and delivery compete, prefer a complete, measurable functional path with focused
verification. Record deferred hardening clearly, then return effort to method and component
optimization.

## Repository-specific architecture override

The rules in this section override broader workspace instructions and historical plans when they
conflict. This repository implements an **Agentic Skill Harness**, not a raw-episode prompting
system.

### Authoritative knowledge flow

The only mainline knowledge flow is:

```text
Experience Episode
-> deterministic Runtime evidence
-> either (a) in-domain Fast/Runtime lifecycle -> Target-local Skill
   or    (b) first-fault attribution -> Slow consolidation/update -> General or Specific Skill
-> Fast Agent observation, proposal, Support and execution
```

The Fast Agent may receive:

- active bootstrap and General Skills / guidance;
- a Specific or Target-local Skill that is legally applicable in the current Domain;
- current deployment-visible Workspace observations and tool results;
- Target Support that has already occurred in the current adaptation trajectory;
- concise scope, risk, provenance and evidence-strength annotations attached to a Skill.

The Fast Agent must not receive:

- a raw or row-wise Source/Target Experience Episode bank;
- `source_experiences`, `raw_episode_bank`, or an equivalent episode-list prompt field;
- a deterministic aggregation of Episodes as an independent candidate menu that bypasses Skill
  formation;
- an unmatched Source Target-local Card as an executable candidate;
- the current Query future or delayed Outcome.

Raw positive, negative, conflict and abstention Episodes remain available to deterministic Runtime
and the Slow Agent for retrieval, contrast, attribution and Skill formation or revision. Runtime may
construct a bounded signed census for the Slow Agent. Episode evidence alone grants neither proposal
priority nor execution rights in the Fast Path.

### A3 / A5 comparison

For the current milestone, both arms use the same Fast Path, tools, Operator supply, Target Support
budget, Runtime and Judge.

- `A3` starts with the common bootstrap / baseline General Skill and no Source-derived Skill.
- `A5` additionally receives a Source-derived General or Shared Skill frozen before Target Outcome,
  plus any legally applicable Skill formed later from its own Target adaptation.
- The A3/A5 difference must not be implemented by directly injecting Source Episodes into the Fast
  prompt.
- A Target-local Skill remains current-Domain only. It cannot acquire cross-Domain execution rights
  through a coarse applicability bin or a development replay.

The Source-derived Skill may be a soft prior that asks the Agent to inspect, probe, avoid or abstain;
it does not bypass current Target Support unless the stricter Shared-Capability evidence requirement
has independently been met.

### Current route lock

The completed `T233 raw Source Episodes -> electricity Fast Agent` development run is retained as a
credible rejected-alternative result and should be labelled
`RAW_SOURCE_EPISODES_TO_FAST_REJECTED`. It does not test the intended Skill-only A5 route and must
not be repaired by reformatting, reweighting, retrieving or aggregating the same raw Episodes for
direct Fast consumption.

The next eligible Source-transfer slice is:

```text
T233 Episodes
-> deterministic Source evidence census
-> Slow PATCH or ABSTAIN
-> frozen Source-derived Skill
-> A5 Fast Agent receives the Skill only
-> electricity development comparison against A3
```

The Source-derived Skill must be frozen without access to electricity Outcome. Existing Pipeline,
Workspace-tool, Target-local Skill lifecycle, attribution, replay and paired-concurrency work remains
valid and should be reused; do not restart or platformize it.
