# slow_prep_rank_v1 — design record

**Status:** registered API candidate in `methods/ttha/` (schema + `TTHASlowAgent.propose_prep_rank`).  
**Not:** sealed TRAIN3 A5 · EditManifest / `propose_edit` overload · L2/L3 established.  
**Date:** 2026-09-13

## Purpose

Separate **prep-menu ranking** from sealed harness edit proposal:

| Path | Stage / schema | Output |
| --- | --- | --- |
| Prep ranking | `stage=prep_menu_rank` / `slow_prep_rank_v1` | `ordered_prep_keys` (length k, ⊆ `M_large`) |
| Harness edit | `stage=edit` / `slow_edit_v1` | `EditManifest` via `propose_edit` |

Callers that need a fit-budget shortlist must use `propose_prep_rank`, never overload `propose_edit`.

## Non-claims

- **NOT** sealed TRAIN3 A5
- **NOT** `EditManifest` / `slow_edit_v1` ranking
- **NOT** invoking sealed `propose_edit` for prep menus
- **NOT** L2/L3 established (`H1_CLAIM_HOLDS=true` is shaped research evidence, not a sealed product prior)
- Research `form_memory` ≠ sealed harness envelopes (`FailurePatternCard` / `HarnessSnapshot`)
- **NOT** a claim that C_B is a pure “form-only” signal (see below)

## Callers

**No production / TRAIN3 runners are wired yet.** This landing is intentional **API-first**: schema registration, agent method, contract tests. Wiring into Beijing / limited-k / research runners is a follow-up.

## H1 optional control arm

`apply_h1` on `propose_prep_rank` **defaults to `False`**.

H1 (`h1_simple_bias_shortlist`) is an **optional fixed control/repair arm** from Beijing role-swap repair / `H1_CLAIM_HOLDS` research — **not** a sealed product prior. Runners that want Beijing-faithful H1 behavior must pass `apply_h1=True`.

Frozen constants (provenance: Beijing H1 repair / Fixed-div schedule):

| Constant | Value | Role |
| --- | --- | --- |
| `SIMPLE_RESERVE_M` | `2` | Default simplex reserve count |
| `PRIMARY_FIXED_DIV_K5` | identity, outlier_iqr(k=1.0), outlier_mad(k=2.5), winsorize(limits=0.05), fft_decompose | Primary Fixed-div k=5 schedule used when `fixed_div_simplex_order` is omitted |

Rule sketch (no E peek):

1. Clamp `m_eff = min(m, k)`; if `m > k`, set meta `m_clamped=true` (do not silently claim the requested `m`).
2. Force-reserve `m_eff` Fixed-div **simplex** labels.
3. Fill remaining `k − m_eff` slots from Slow rank; demote composites whose pipe-head ∈ Fixed-div simples when room allows.
4. Merge as **`forced + fill`** (reserved simplex **first**).

### `ordered_prep_keys` semantics (after H1 order fix)

- Index **0** = highest explore priority when a consumer walks the shortlist **in order**.
- For bag-of-k **C_B-safe** pick, **membership** still matters; order is now coherent with “guarantee simples are explored early.”
- Pre-H1 LLM order is preserved when `apply_h1=False`.

## C_B information role (vs §6.28.4)

`form_memory.C_B` (and enum token `C_B` in `form_signals_used`) is **held-in delayed-calibration block utility** used by the research **C_B-safe** gate.

It is:

- **Not** deploy / held-out **E**
- **Not** accurately described as a pure “form-only” signal

**Sealed multi-arm feedback accounting is still TBD** — this is the §6.28.4 concern: how (or whether) delayed-calibration utilities enter sealed harness memory / multi-arm ledgers without laundering E. Until that accounting is specified, treat C_B here as a research-stage held-in signal only.

`form_memory` still **hard-rejects E** (`additionalProperties: false` + post-validator).

## Artifacts in tree

| Path | Role |
| --- | --- |
| `methods/ttha/schemas/slow_prep_rank_v1.json` | Stage schema |
| `methods/ttha/slow_agent.py` | `propose_prep_rank`, `h1_simple_bias_shortlist`, Fixed-div constants |
| `methods/ttha/schema_contracts.py` | `_STAGE_SCHEMA_FILES` registration |
| `methods/ttha/agent_core.py` | `prep_menu_rank` response-contract teaching |
| `tests/methods/test_slow_prep_rank_contract.py` | Contract tests |

## Citation

The research pack that pioneered this API lived under `_scratch/research_pack_2026-09-12/p4_slow_prep_rank_v1/` (**gitignored**). This document is the **in-repo** design record; code/schema citations should point here rather than to `_scratch/...`.

— end —
