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
