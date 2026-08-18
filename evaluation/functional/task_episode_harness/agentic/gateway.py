"""The existing public tools, bound to a series the Agent names.

Frozen design §6.1 requires a Workspace tool to bind an explicit data object
and time window; §5.2 requires the Agent's Context to grow from what it chose
to look at rather than from one Runtime-chosen representative summary.  The
two existing gateways cannot do both: ``LocalPublicToolGateway`` is built
around a single series and rejects every argument, so an Agent reasoning about
a Task whose Scope holds a dozen series can only ever see the one series the
Runtime picked for it.  That is exactly the resolution loss §12 lists as
un-connected.

This gateway keeps the two existing tools (``summarize_series``,
``localize_regions``), the same ``extract_public_features`` extractor and the
same public prefix the Task Context already read.  The only difference is one
bounded argument: ``series_uid``, restricted by enum to the Task's own Scope.
No new tool kind is introduced (§15 ``new TS-native tools in G1: 0``), no free
path, no interval, no code.

Information wall: the gateway is constructed from ``values[uid][:cutoff]``
only.  Nothing at or after the Support origin is reachable through it, so no
tool call can see a Query future or a delayed Outcome.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    PublicToolReceipt,
    extract_public_features,
)

_TOOL_NAMES = ("summarize_series", "localize_regions")
_FAST_STAGES = frozenset({"inspect", "propose", "select"})


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(nested) for nested in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class CohortScopePublicToolGateway:
    """Deployment-visible reads over one Task's scoped series prefixes."""

    def __init__(
        self,
        prefixes: Mapping[str, Any],
        *,
        task_kind: str,
        observation_cutoff: int,
        maximum_calls: int = 6,
    ) -> None:
        if not prefixes:
            raise ValueError("at least one scoped series prefix is required")
        if isinstance(maximum_calls, bool) or not isinstance(maximum_calls, int):
            raise ValueError("maximum_calls must be an integer")
        if maximum_calls < 1:
            raise ValueError("maximum_calls must be positive")
        self._task_kind = str(task_kind)
        self._cutoff = int(observation_cutoff)
        self.maximum_calls = maximum_calls
        self.calls = 0
        self.refused_calls = 0
        self.call_log: list[dict[str, Any]] = []
        self._series: dict[str, np.ndarray] = {}
        for uid, values in prefixes.items():
            array = np.asarray(values, dtype=np.float64).ravel().copy()
            if array.size != self._cutoff:
                raise ValueError(
                    f"series {uid!r} prefix is {array.size} points, "
                    f"expected the Task cutoff {self._cutoff}"
                )
            array.setflags(write=False)
            self._series[str(uid)] = array
        self.scope_series_uids = tuple(sorted(self._series))
        self._features: dict[str, Mapping[str, Any]] = {}
        self._context_sha = canonical_sha256(
            {
                "schema_version": "cohort-scope-public-tool-context/1",
                "task_kind": self._task_kind,
                "observation_cutoff": self._cutoff,
                "series": {
                    uid: [
                        float(value) if math.isfinite(float(value)) else None
                        for value in self._series[uid]
                    ]
                    for uid in self.scope_series_uids
                },
            }
        )

    # -- PublicToolGateway protocol ---------------------------------------
    @property
    def context_sha(self) -> str:
        return self._context_sha

    def schemas_for(
        self,
        *,
        role: Any,
        stage: str,
    ) -> tuple[Mapping[str, Any], ...]:
        if str(role) not in {"fast", "AgentRole.FAST"} or stage not in _FAST_STAGES:
            return ()
        series_argument = {
            "type": "object",
            "additionalProperties": False,
            "required": ["series_uid"],
            "properties": {
                "series_uid": {"enum": list(self.scope_series_uids)},
            },
        }
        return (
            {
                "name": "summarize_series",
                "description": (
                    "Return the deployment-visible feature summary of one "
                    "named series in this Task's Scope, computed on its public "
                    "prefix. Costs one Workspace tool call, never a Support "
                    f"probe. At most {self.maximum_calls} calls per Task."
                ),
                "input_schema": series_argument,
            },
            {
                "name": "localize_regions",
                "description": (
                    "Return the public estimated region fractions of one named "
                    "series in this Task's Scope. Fractions are of that "
                    "series' own public prefix."
                ),
                "input_schema": series_argument,
            },
        )

    def call(self, name: str, arguments: Mapping[str, Any]) -> PublicToolReceipt:
        if name not in _TOOL_NAMES:
            raise PermissionError(f"undeclared public tool: {name}")
        if not isinstance(arguments, Mapping) or set(arguments) != {"series_uid"}:
            raise PermissionError(
                "public Workspace tools accept exactly one series_uid argument"
            )
        uid = str(arguments["series_uid"])
        if uid not in self._series:
            raise PermissionError(
                "series_uid is outside this Task's Scope"
            )
        if self.calls >= self.maximum_calls:
            # A refusal is behaviour, not an instrument failure: the Agent is
            # told the observation budget is spent and decides what to do.
            self.refused_calls += 1
            self.call_log.append(
                {"tool_name": name, "series_uid": uid, "ok": False,
                 "reason": "WORKSPACE_TOOL_BUDGET_EXHAUSTED"}
            )
            return PublicToolReceipt.create(
                tool_name=name,
                arguments=arguments,
                public_result={
                    "refused": "WORKSPACE_TOOL_BUDGET_EXHAUSTED",
                    "calls_used": self.calls,
                    "maximum_calls": self.maximum_calls,
                },
                context_sha=self.context_sha,
                ok=False,
            )
        features = self._features.get(uid)
        if features is None:
            features = _plain(
                extract_public_features(
                    self._series[uid], task_kind=self._task_kind
                )
            )
            self._features[uid] = features
        if name == "summarize_series":
            result = {
                "series_uid": uid,
                "observation_cutoff": self._cutoff,
                "features": dict(features),
            }
        else:
            result = {
                "series_uid": uid,
                "observation_cutoff": self._cutoff,
                "estimated_region_start_fraction": features[
                    "estimated_region_start_fraction"
                ],
                "estimated_region_end_fraction": features[
                    "estimated_region_end_fraction"
                ],
            }
        self.calls += 1
        self.call_log.append(
            {"tool_name": name, "series_uid": uid, "ok": True}
        )
        return PublicToolReceipt.create(
            tool_name=name,
            arguments=arguments,
            public_result=result,
            context_sha=self.context_sha,
        )

    def observed_feature_keys(self) -> set[str]:
        """Public feature names this Agent has actually been shown so far.

        The inspect post-validator refuses a hypothesis that cites a feature
        the Agent never fetched, so the grounding set has to be readable while
        the stage is still running -- the caller's trace is only assembled
        after ``run_stage`` returns.
        """
        keys: set[str] = set()
        for uid, features in self._features.items():
            keys.update(str(key) for key in features)
        return keys

    # -- accounting --------------------------------------------------------
    def accounting(self) -> dict[str, Any]:
        """Workspace tool cost only.  Never merged with LLM or Support cost."""
        return {
            "workspace_tool_calls": self.calls,
            "workspace_tool_calls_refused": self.refused_calls,
            "workspace_tool_call_budget": self.maximum_calls,
            "distinct_series_observed": len(
                {row["series_uid"] for row in self.call_log if row["ok"]}
            ),
            "scope_series_count": len(self.scope_series_uids),
            "call_log": [dict(row) for row in self.call_log],
        }


__all__ = ["CohortScopePublicToolGateway"]
