"""Atomic, resumable ledger for one run: physical fits, cache hits, LLM requests/tokens, wall clock.
Logical feedback opportunities and physical cost are recorded separately (comparison doc §5).
Caps are set by the controller's task doc; exceeding one raises BudgetExhausted before the cost is
incurred. No hashing.
"""
from __future__ import annotations

import time
from pathlib import Path

from . import context


class BudgetExhausted(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, *, max_fit_attempts: int, max_llm_requests: int = 0, max_llm_tokens: int = 0, max_wall_s: float = 0.0, max_retries: int = 0):
        self.path = Path(path)
        if self.path.exists():
            self.s = context.read_json(self.path)
        else:
            self.s = {"created_local": time.strftime("%Y-%m-%d %H:%M:%S"), "caps": {}, "fit_attempts": 0, "fits_ok": 0, "fits_failed": 0, "cache_hits": 0,
                      "retries_used": 0, "fit_wall_seconds": 0.0, "llm_requests": 0, "llm_http_attempts": 0, "llm_tokens_in": 0, "llm_tokens_out": 0,
                      "llm_tokens_unknown": 0, "started_epoch": time.time(), "events": []}
        self.s["caps"] = {"max_fit_attempts": max_fit_attempts, "max_llm_requests": max_llm_requests, "max_llm_tokens": max_llm_tokens,
                          "max_wall_s": max_wall_s, "max_retries": max_retries}
        self._save()

    def _save(self) -> None:
        context.write_json(self.path, self.s)

    # --- fits
    def check_fit(self) -> None:
        if self.s["fit_attempts"] >= self.s["caps"]["max_fit_attempts"]:
            raise BudgetExhausted("fit attempts cap %d reached" % self.s["caps"]["max_fit_attempts"])
        self.check_wall()

    def charge_fit(self, cell_id: str, ok: bool, seconds: float, detail: str = "", retry: bool = False) -> None:
        self.s["fit_attempts"] += 1
        self.s["fits_ok" if ok else "fits_failed"] += 1
        if retry:
            self.s["retries_used"] += 1
        self.s["fit_wall_seconds"] += seconds
        self.s["events"].append({"t": time.strftime("%H:%M:%S"), "kind": "fit", "cell": cell_id, "ok": ok, "seconds": round(seconds, 1), "retry": retry, "detail": detail[-300:]})
        self._save()

    def note_cache(self, cell_id: str) -> None:
        self.s["cache_hits"] += 1
        self.s["events"].append({"t": time.strftime("%H:%M:%S"), "kind": "cache", "cell": cell_id})
        self._save()

    def can_retry(self) -> bool:
        return self.s["retries_used"] < self.s["caps"]["max_retries"]

    # --- LLM
    def check_llm(self) -> None:
        caps = self.s["caps"]
        if caps["max_llm_requests"] and self.s["llm_requests"] >= caps["max_llm_requests"]:
            raise BudgetExhausted("LLM request cap %d reached" % caps["max_llm_requests"])
        if caps["max_llm_tokens"] and (self.s["llm_tokens_in"] + self.s["llm_tokens_out"]) >= caps["max_llm_tokens"]:
            raise BudgetExhausted("LLM token cap reached")
        self.check_wall()

    def charge_llm(self, role: str, unit: str, http_attempts: int, tokens_in, tokens_out, ok: bool, detail: str = "") -> None:
        self.s["llm_requests"] += 1
        self.s["llm_http_attempts"] += int(http_attempts)
        if tokens_in is None or tokens_out is None:
            self.s["llm_tokens_unknown"] += 1
        else:
            self.s["llm_tokens_in"] += int(tokens_in)
            self.s["llm_tokens_out"] += int(tokens_out)
        self.s["events"].append({"t": time.strftime("%H:%M:%S"), "kind": "llm", "role": role, "unit": unit, "http": http_attempts, "ok": ok, "detail": detail[-300:]})
        self._save()

    # --- wall
    def check_wall(self) -> None:
        cap = self.s["caps"]["max_wall_s"]
        if cap and (time.time() - self.s["started_epoch"]) > cap:
            raise BudgetExhausted("wall clock cap %.0fs reached" % cap)

    def summary(self) -> dict:
        return {k: v for k, v in self.s.items() if k != "events"}
