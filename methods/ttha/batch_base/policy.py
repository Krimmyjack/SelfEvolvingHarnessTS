"""Policy grammar, validation, compilation and random generation (the Material Plan language of
docs/BATCH_RESEARCH_WORKFLOW_FRAMEWORK §4.2 / docs/BATCH_POLICY_HARNESS_FRAMEWORK §4).

    {"default": {"steps": [...]},
     "rules": [{"when": <predicate>, "steps": [...]}, ...],      # first matching rule wins; else default
     "rationale": "<short>", "observation_fields_used": [...]}

steps: 0-2 of {"op": "timemixup"|"freqmask"|"freqmix", ...params in spec.ACTIONS}; [] = identity.
predicate: {"feature","op","value"} leaf (feature in spec.OBS_FIELDS; value a number or
{"quantile": q} resolved against the current batch at compile time), or {"all":[...]}, {"any":[...]},
{"not": p}, {"const": true|false}. Compilation expands a policy into 32 per-entity step lists
(the assignment) and records resolved thresholds; the assignment key identifies material aliases.
Validation is fail-closed: unknown ops / values / fields, entity names, dataset or job identifiers
and code-like text are rejected.
"""
from __future__ import annotations

import json
import re

import numpy as np

from . import spec

_UID_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(uid|entity_\d+|series_\d+)(?![A-Za-z0-9_])", re.IGNORECASE)
_CODE_PATTERN = re.compile(r"```|<code>|import |def |exec\(|eval\(")


class PolicyError(ValueError):
    pass


# ----------------------------------------------------------------------------- validation
def validate_steps(steps) -> list:
    if not isinstance(steps, list) or len(steps) > spec.MAX_STEPS:
        raise PolicyError("steps must be a list of at most %d steps" % spec.MAX_STEPS)
    out = []
    for st in steps:
        if not isinstance(st, dict) or "op" not in st:
            raise PolicyError("each step must be an object with an 'op'")
        op = st["op"]
        if op not in spec.ACTIONS:
            raise PolicyError("unknown op %r" % op)
        params = spec.ACTIONS[op]
        clean = {"op": op}
        for k, allowed in params.items():
            if k not in st:
                raise PolicyError("step %s missing parameter %r" % (op, k))
            v = st[k]
            if isinstance(v, float) or isinstance(v, int):
                if not any(abs(float(v) - float(a)) < 1e-12 for a in allowed):
                    raise PolicyError("step %s parameter %s=%r not in %s" % (op, k, v, allowed))
                clean[k] = float([a for a in allowed if abs(float(v) - float(a)) < 1e-12][0])
            else:
                if v not in allowed:
                    raise PolicyError("step %s parameter %s=%r not in %s" % (op, k, v, allowed))
                clean[k] = v
        extra = set(st) - set(params) - {"op"}
        if extra:
            raise PolicyError("step %s has unknown parameters %s" % (op, sorted(extra)))
        out.append(clean)
    return out


def validate_predicate(p) -> None:
    if not isinstance(p, dict):
        raise PolicyError("predicate must be an object")
    if set(p) == {"const"}:
        if not isinstance(p["const"], bool):
            raise PolicyError("const must be a boolean")
        return
    if set(p) == {"not"}:
        validate_predicate(p["not"])
        return
    if set(p) in ({"all"}, {"any"}):
        k = next(iter(p))
        if not isinstance(p[k], list) or not p[k]:
            raise PolicyError("%s must be a non-empty list" % k)
        for q in p[k]:
            validate_predicate(q)
        return
    if set(p) != {"feature", "op", "value"}:
        raise PolicyError("predicate leaf must contain exactly feature, op, value (got %s)" % sorted(p))
    if p["feature"] not in spec.OBS_FIELDS:
        raise PolicyError("unknown feature %r" % p["feature"])
    if p["op"] not in spec.PREDICATE_OPS:
        raise PolicyError("unknown predicate op %r" % p["op"])
    v = p["value"]
    if isinstance(v, dict):
        if set(v) != {"quantile"} or not (0.0 <= float(v["quantile"]) <= 1.0):
            raise PolicyError("value object must be {'quantile': q in [0,1]}")
    elif not isinstance(v, (int, float)) or isinstance(v, bool):
        raise PolicyError("value must be a number or {'quantile': q}")


def validate_text(text: str, allow_entity_ids: bool = True) -> None:
    """Policy rationales may name entity_k rows of the current batch (a Material Plan is job-local);
    reusable Skill text must not (allow_entity_ids=False) -- comparison doc §2 / AGENTS §5.9."""
    lowered = str(text or "").lower()
    for bad in spec.FORBIDDEN_TEXT:
        if bad.lower() in lowered:
            raise PolicyError("text contains forbidden identifier %r" % bad)
    if not allow_entity_ids and _UID_PATTERN.search(text or ""):
        raise PolicyError("text contains an entity/UID-shaped token")
    if _CODE_PATTERN.search(text or ""):
        raise PolicyError("text looks like code")


def validate_policy(policy) -> dict:
    if not isinstance(policy, dict) or "default" not in policy:
        raise PolicyError("policy must be an object with a 'default'")
    default = policy["default"]
    if not isinstance(default, dict) or "steps" not in default:
        raise PolicyError("default must be {'steps': [...]}")
    clean = {"default": {"steps": validate_steps(default["steps"])}, "rules": []}
    for r in policy.get("rules", []) or []:
        if not isinstance(r, dict) or set(r) != {"when", "steps"}:
            raise PolicyError("each rule must be {'when': predicate, 'steps': [...]}")
        validate_predicate(r["when"])
        clean["rules"].append({"when": r["when"], "steps": validate_steps(r["steps"])})
    if len(clean["rules"]) > 8:
        raise PolicyError("at most 8 rules")
    rationale = str(policy.get("rationale", ""))[:500]
    validate_text(rationale)
    clean["rationale"] = rationale
    fields = policy.get("observation_fields_used", []) or []
    clean["observation_fields_used"] = [f for f in fields if f in spec.OBS_FIELDS]
    return clean


# ----------------------------------------------------------------------------- compilation
def _resolve(value, feature: str, table: list):
    if isinstance(value, dict):
        vals = np.array([r[feature] for r in table if r.get(feature) is not None], dtype=np.float64)
        if vals.size == 0:
            return None
        return float(np.quantile(vals, float(value["quantile"])))
    return float(value)


def _eval(p, row: dict, table: list, resolved: dict):
    if "const" in p:
        return bool(p["const"])
    if "not" in p:
        v = _eval(p["not"], row, table, resolved)
        return None if v is None else (not v)
    if "all" in p:
        vals = [_eval(q, row, table, resolved) for q in p["all"]]
        return False if any(v is False for v in vals) else (None if any(v is None for v in vals) else True)
    if "any" in p:
        vals = [_eval(q, row, table, resolved) for q in p["any"]]
        return True if any(v is True for v in vals) else (None if any(v is None for v in vals) else False)
    f, op, raw = p["feature"], p["op"], p["value"]
    key = json.dumps([f, raw], sort_keys=True)
    if key not in resolved:
        resolved[key] = _resolve(raw, f, table)
    thr = resolved[key]
    x = row.get(f)
    if x is None or thr is None:
        return None                                   # UNKNOWN: this rule does not fire
    return {"<": x < thr, "<=": x <= thr, ">": x > thr, ">=": x >= thr, "==": x == thr, "!=": x != thr}[op]


def compile_policy(policy: dict, table: list) -> dict:
    """table: overview()['entities'] (32 rows). Returns {'assignment': [steps]*32, 'rule_index': [...],
    'resolved_thresholds': {...}, 'n_unknown': int, 'key': str}."""
    clean = validate_policy(policy)
    resolved, assignment, rule_index, n_unknown = {}, [], [], 0
    for row in table:
        chosen, ri = clean["default"]["steps"], -1
        for i, r in enumerate(clean["rules"]):
            v = _eval(r["when"], row, table, resolved)
            if v is None:
                n_unknown += 1
            if v is True:
                chosen, ri = r["steps"], i
                break
        assignment.append(chosen)
        rule_index.append(ri)
    return {"policy": clean, "assignment": assignment, "rule_index": rule_index,
            "resolved_thresholds": resolved, "n_unknown": n_unknown, "key": assignment_key(assignment)}


def assignment_key(assignment: list) -> str:
    return json.dumps(assignment, sort_keys=True, separators=(",", ":"))


def uniform_policy(steps: list, rationale: str = "") -> dict:
    return {"default": {"steps": steps}, "rules": [], "rationale": rationale, "observation_fields_used": []}


def identity_policy() -> dict:
    return uniform_policy([], "no augmentation")


def fixed_mixup_policy() -> dict:
    return uniform_policy(list(spec.FIXED_MIXUP_STEPS), "historical fixed incumbent: uniform TimeMixup, random donor, w=0.25")


# ----------------------------------------------------------------------------- random policies (same grammar; the Random arm)
def _random_steps(rng: np.random.RandomState) -> list:
    n = int(rng.choice([0, 1, 1, 2]))
    steps = []
    for _ in range(n):
        op = str(rng.choice(list(spec.ACTIONS)))
        st = {"op": op}
        for k, allowed in spec.ACTIONS[op].items():
            st[k] = allowed[int(rng.randint(len(allowed)))]
        steps.append(st)
    return steps


def random_policy(seed: int) -> dict:
    """Uniform draw over the grammar: 0-2 rules on quantile thresholds of public fields."""
    rng = np.random.RandomState(seed)
    rules = []
    for _ in range(int(rng.choice([0, 1, 1, 2]))):
        f = str(rng.choice([x for x in spec.OBS_FIELDS if x not in ("missing_count", "n_parent_pairs")]))
        op = str(rng.choice(["<", ">="]))
        q = float(rng.choice([0.25, 0.5, 0.75]))
        rules.append({"when": {"feature": f, "op": op, "value": {"quantile": q}}, "steps": _random_steps(rng)})
    return {"default": {"steps": _random_steps(rng)}, "rules": rules, "rationale": "random policy (control arm)", "observation_fields_used": []}
