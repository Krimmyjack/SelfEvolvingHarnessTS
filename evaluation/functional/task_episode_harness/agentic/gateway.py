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

# --------------------------------------------------------------- batch recipe
#
# One optional extra Workspace tool.  The gateway does not know what a batch
# recipe is: a caller that wants the tool passes a *binding* object, and the
# gateway only registers its name, publishes its description and bounded
# argument enums, routes one call to it and renders the result back into the
# Agent's context.  No recipe module is imported here.
#
# ``batch_recipe_binding=None`` -- the default -- leaves this class exactly as
# it was: the same two tools, the same stages, and a ``context_sha`` payload
# with no extra key, so every existing run is byte-identical.
#
# A binding must provide:
#     identity                  JSON-plain mapping, folded into context_sha
#     description               str, shown to the Agent
#     cohort_choices            Sequence[str], the tool's ``cohort`` enum
#     consumer_variant_choices  Sequence[str], the ``consumer_variant`` enum
#     run(cohort=..., consumer_variant=...) -> Mapping[str, Any]
_BATCH_RECIPE_TOOL = "batch_recipe"
_BATCH_RECIPE_STAGES = _FAST_STAGES | frozenset({"batch_plan"})


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
        batch_recipe_binding: Any | None = None,
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
        self._recipe = batch_recipe_binding
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
        context_payload: dict[str, Any] = {
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
        # The extra key exists only when the extra tool exists, so a gateway
        # built without a binding hashes exactly what it hashed before.
        if self._recipe is not None:
            context_payload["batch_recipe"] = _plain(
                getattr(self._recipe, "identity", {})
            )
        self._context_sha = canonical_sha256(context_payload)

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
        allowed_stages = (
            _BATCH_RECIPE_STAGES if self._recipe is not None else _FAST_STAGES
        )
        if str(role) not in {"fast", "AgentRole.FAST"} or stage not in allowed_stages:
            return ()
        series_argument = {
            "type": "object",
            "additionalProperties": False,
            "required": ["series_uid"],
            "properties": {
                "series_uid": {"enum": list(self.scope_series_uids)},
            },
        }
        base: tuple[Mapping[str, Any], ...] = (
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
        if self._recipe is None:
            return base
        return base + (
            {
                "name": _BATCH_RECIPE_TOOL,
                "description": str(getattr(self._recipe, "description", "")),
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["cohort", "consumer_variant"],
                    "properties": {
                        "cohort": {
                            "enum": [
                                str(name) for name in
                                getattr(self._recipe, "cohort_choices", ())
                            ]
                        },
                        "consumer_variant": {
                            "enum": [
                                str(name) for name in
                                getattr(
                                    self._recipe, "consumer_variant_choices", ()
                                )
                            ]
                        },
                    },
                },
            },
        )

    def _refuse(
        self,
        name: str,
        arguments: Mapping[str, Any],
        reason: str,
        detail: Mapping[str, Any],
    ) -> PublicToolReceipt:
        """Tell the Agent no, and let it decide what to do next.

        A refusal is behaviour, not an instrument failure.  Raising here ends
        the whole run from inside the stage loop, which is how one malformed
        argument destroyed a nine-Task run; the Agent never even learns it got
        the call wrong.  Returning an ``ok=False`` receipt keeps the
        information wall exactly where it was -- no data is served -- while
        putting the refusal in the conversation where it belongs.
        """
        self.refused_calls += 1
        self.call_log.append(
            {"tool_name": name, "ok": False, "reason": reason,
             "arguments": _plain(arguments)}
        )
        return PublicToolReceipt.create(
            tool_name=name,
            arguments=arguments if isinstance(arguments, Mapping) else {},
            public_result={"refused": reason, **dict(detail)},
            context_sha=self.context_sha,
            ok=False,
        )

    def _call_batch_recipe(
        self, arguments: Mapping[str, Any]
    ) -> PublicToolReceipt:
        """Route one bounded call to the binding and render its result back.

        The gateway owns the argument wall and the budget; the binding owns
        what the recipe is.  Nothing here interprets, scores or rewrites the
        result -- it is handed to the Agent as the binding returned it.
        """
        name = _BATCH_RECIPE_TOOL
        cohorts = [
            str(item) for item in getattr(self._recipe, "cohort_choices", ())
        ]
        variants = [
            str(item)
            for item in getattr(self._recipe, "consumer_variant_choices", ())
        ]
        if (
            not isinstance(arguments, Mapping)
            or set(arguments) != {"cohort", "consumer_variant"}
        ):
            return self._refuse(
                name, arguments, "INVALID_TOOL_ARGUMENTS",
                {
                    "required_arguments": ["cohort", "consumer_variant"],
                    "received_argument_names": sorted(
                        str(key) for key in arguments
                    ) if isinstance(arguments, Mapping) else [],
                    "allowed_cohorts": cohorts,
                    "allowed_consumer_variants": variants,
                },
            )
        cohort = str(arguments["cohort"])
        variant = str(arguments["consumer_variant"])
        if cohort not in cohorts:
            return self._refuse(
                name, arguments, "COHORT_OUTSIDE_TOOL_SCOPE",
                {"allowed_cohorts": cohorts},
            )
        if variant not in variants:
            return self._refuse(
                name, arguments, "CONSUMER_VARIANT_OUTSIDE_TOOL_SCOPE",
                {"allowed_consumer_variants": variants},
            )
        if self.calls >= self.maximum_calls:
            return self._refuse(
                name, arguments, "WORKSPACE_TOOL_BUDGET_EXHAUSTED",
                {"calls_used": self.calls, "maximum_calls": self.maximum_calls},
            )
        result = _plain(
            self._recipe.run(cohort=cohort, consumer_variant=variant)
        )
        self.calls += 1
        self.call_log.append(
            {
                "tool_name": name,
                "cohort": cohort,
                "consumer_variant": variant,
                "ok": True,
            }
        )
        return PublicToolReceipt.create(
            tool_name=name,
            arguments=arguments,
            public_result=result,
            context_sha=self.context_sha,
        )

    def call(self, name: str, arguments: Mapping[str, Any]) -> PublicToolReceipt:
        if name == _BATCH_RECIPE_TOOL:
            if self._recipe is None:
                raise PermissionError(f"undeclared public tool: {name}")
            return self._call_batch_recipe(arguments)
        if name not in _TOOL_NAMES:
            # Unreachable through the stage loop, which rejects an undeclared
            # tool before calling.  Kept as a hard error for direct misuse.
            raise PermissionError(f"undeclared public tool: {name}")
        if not isinstance(arguments, Mapping) or set(arguments) != {"series_uid"}:
            return self._refuse(
                name, arguments, "INVALID_TOOL_ARGUMENTS",
                {
                    "required_arguments": ["series_uid"],
                    "received_argument_names": sorted(
                        str(key) for key in arguments
                    ) if isinstance(arguments, Mapping) else [],
                    "allowed_series_uids": list(self.scope_series_uids),
                },
            )
        uid = str(arguments["series_uid"])
        if uid not in self._series:
            return self._refuse(
                name, arguments, "SERIES_OUTSIDE_TASK_SCOPE",
                {"allowed_series_uids": list(self.scope_series_uids)},
            )
        if self.calls >= self.maximum_calls:
            return self._refuse(
                name, arguments, "WORKSPACE_TOOL_BUDGET_EXHAUSTED",
                {"calls_used": self.calls,
                 "maximum_calls": self.maximum_calls},
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

    def observed_feature_values(self) -> dict[str, set[str]]:
        """Feature name -> the exact value strings this gateway has served.

        A hypothesis may cite ``key=value``; that is only accepted when the
        value it names is one this Task actually saw.  A set rather than a
        single value because the same feature is served once per inspected
        series.
        """
        served: dict[str, set[str]] = {}
        for features in self._features.values():
            for key, value in features.items():
                served.setdefault(str(key), set()).add(str(value))
        return served

    # -- accounting --------------------------------------------------------
    def accounting(self) -> dict[str, Any]:
        """Workspace tool cost only.  Never merged with LLM or Support cost."""
        return {
            "workspace_tool_calls": self.calls,
            "workspace_tool_calls_refused": self.refused_calls,
            "workspace_tool_call_budget": self.maximum_calls,
            "distinct_series_observed": len(
                {
                    row["series_uid"] for row in self.call_log
                    if row["ok"] and "series_uid" in row
                }
            ),
            "refusal_reasons": sorted(
                {str(row["reason"]) for row in self.call_log if not row["ok"]}
            ),
            "scope_series_count": len(self.scope_series_uids),
            "call_log": [dict(row) for row in self.call_log],
        }


__all__ = ["CohortScopePublicToolGateway"]
