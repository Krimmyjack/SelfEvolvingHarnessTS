# README Project Framework Design

**Date:** 2026-07-17
**Status:** Approved for implementation

## Objective

Make the converged project structure visible from the repository landing page without
turning the README into a complete file inventory.

## Presentation

Replace the current flat `Active structure` bullets with:

1. a top-level directory tree covering `contracts`, `conditioning`, `operators`,
   `runtime`, `methods`, `evaluation`, `artifacts`, `experiments`, `tests`, and `docs`;
2. one concise responsibility comment for each directory;
3. a compact active-flow line: `TaskSpec -> Method -> Runtime -> Operators`, with
   `Conditioning` feeding the runtime/method decision path.

The tree does not enumerate individual files. Historical source remains described as
Git-recoverable rather than present under `experiments/`.

## Constraints

- Keep the existing verification commands and historical-recovery section unchanged.
- Describe `methods/h_ref_v02` as the current reference and TTHA as future active work.
- Do not imply that benchmark evaluation is the main product line.
- Use plain text/Markdown only; no generated diagram assets.

## Verification

- Check that every displayed top-level directory exists.
- Check that no retired directory appears in the README framework.
- Run the retained test suite after the documentation change before local integration.
