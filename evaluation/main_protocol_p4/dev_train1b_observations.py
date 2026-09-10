"""DEV-TRAIN-1B process-local execution-window observation adapter."""
import collections
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np

from evaluation.main_protocol_p4 import dev_train1_runner as R

_FEATURE_KEYS = (
    "missing_fraction", "longest_missing_run_fraction", "local_robust_z_peak",
    "level_excursion_score", "estimated_level_offset", "period_change_score",
    "period_reliability",
)
_NOTES = (
    "Descriptors do not establish effect. Each train W is replayed on every "
    "240-point historic window. Fractions are local to each window. Per-series "
    "program geometry is unchanged. Train X/y are legally observed historical data."
)
_current_packet = ContextVar("dev_train1b_packet")
RUNTIME_CAPABILITIES = {
    "fast_has_no_current_support_delayed_downstream_metric_or_cross_trajectory_episode": True,
    "verify_means": "deterministic legality/trial execution, not utility",
    "hypotheses_need_not_have_known_downstream_benefits_before_legal_consideration": True,
    "effects_never_guaranteed": True,
    "fixed_probe_panel": "absent",
    "imputation_probe_direction": "unavailable/unknown",
    "clipping_probe_direction": "unavailable/unknown",
    "denoising_probe_direction": "unavailable/unknown",
    "level_probe_direction": "unavailable/unknown",
    "probe_unknown_is_not_measured_negative": True,
    "available_fast_tools": ["summarize_series", "localize_regions"],
    "no_tool_can_obtain_a_positive_probe_in_this_run": True,
    "training_observed_prefix_points": int(R.TRAINING_PREFIX_END),
    "training_w_replayed_on_each_historic_window_points": int(R.CONTEXT_LENGTH + R.HORIZON),
    "whole_x_and_y_changes_permitted": True,
    "one_shared_model": True,
    "prediction_uses_last_context_points": int(R.CONTEXT_LENGTH),
    "prediction_cannot_refit": True,
    "no_mandated_operator_threshold_program_count_nonidentity_rate_or_recommendation": True,
    "identity_is_legal": True,
}


def _summary(values):
    array = np.asarray(values, dtype=np.float64).ravel()
    mapping = R.extract_public_features(array, task_kind="forecast")
    out = {"n_points": int(array.size)}
    for key in _FEATURE_KEYS:
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        out[key] = value
    return out


def execution_observations(observed, role, spec):
    observed = np.asarray(observed, dtype=np.float64)
    packet = {
        "role": role,
        "notes": _NOTES,
        "descriptors_do_not_establish_effect": True,
        "per_series_program_geometry_unchanged": True,
    }
    if role == "train_workflow":
        if observed.ndim != 1:
            raise ValueError("train_workflow observed must be 1D")
        cutoff = int(spec.training_prefix_end)
        if int(observed.size) != cutoff:
            raise ValueError(
                "train_workflow observed length %d != training_prefix_end %d"
                % (int(observed.size), cutoff)
            )
        need = int(R.CONTEXT_LENGTH + R.HORIZON)
        windows = []
        for raw_anchor in spec.anchors:
            anchor = int(raw_anchor)
            if anchor + R.HORIZON > cutoff:
                continue
            start = anchor - R.CONTEXT_LENGTH
            stop = anchor + R.HORIZON
            if start < 0:
                raise ValueError("training anchor starts before the legal prefix")
            window = observed[start:stop]
            if int(window.size) != need:
                raise ValueError(
                    "training window at anchor %d is not length %d" % (anchor, need)
                )
            windows.append({
                "anchor": anchor,
                "window_start_index": int(start),
                "window_end_index_exclusive": int(stop),
                "local_layout": (
                    "[anchor-%d, anchor+%d) = %d X + %d historic observed y"
                    % (R.CONTEXT_LENGTH, R.HORIZON, R.CONTEXT_LENGTH, R.HORIZON)
                ),
                "whole": _summary(window),
                "X": _summary(window[:R.CONTEXT_LENGTH]),
                "y": _summary(window[R.CONTEXT_LENGTH:]),
            })
        packet.update({
            "observed_length": int(observed.size),
            "windows": windows,
            "one_workflow_replayed_on_every_window": True,
            "fractions_are_local_to_each_window": True,
            "x_and_y_are_historic_observed_labels": True,
        })
        return packet
    if role == "predict_input":
        if observed.ndim != 1 or int(observed.size) != int(R.CONTEXT_LENGTH):
            raise ValueError(
                "predict_input observed must be 1D length %d" % int(R.CONTEXT_LENGTH)
            )
        packet.update({
            "observed_length": int(observed.size),
            "local_layout": "[origin-%d, origin)" % int(R.CONTEXT_LENGTH),
            "context": _summary(observed),
        })
        return packet
    raise ValueError("unknown Fast role %r" % role)


def _bind(index, key, packet):
    if key in index and index[key] != packet:
        raise ValueError("duplicate conflicting observation key %r" % (key,))
    index[key] = packet


def _packet_from(row):
    if "execution_observations" not in row or row.get("execution_observations") is None:
        return "UNKNOWN"
    return row["execution_observations"]


def _wrap_decision(original):
    def run_fast_decision(
        *, spec, bundle, snapshot, machinery, backend_factory, uid, role, arm,
        position, origin, task_ctx,
    ):
        uid = str(uid)
        raw = np.asarray(bundle.values[uid], dtype=np.float64)
        if role == "train_workflow":
            observed = raw[:int(spec.training_prefix_end)]
        elif role == "predict_input":
            origin_i = int(origin)
            start = origin_i - R.CONTEXT_LENGTH
            if start < 0:
                raise ValueError("origin %d is shorter than 192" % origin_i)
            observed = raw[start:origin_i]
        else:
            raise ValueError("unknown Fast role %r" % role)
        packet = execution_observations(observed, role, spec)
        token = _current_packet.set(packet)
        try:
            record = original(
                spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
                backend_factory=backend_factory, uid=uid, role=role, arm=arm,
                position=position, origin=origin, task_ctx=task_ctx,
            )
            record["execution_observations"] = packet
            return record
        finally:
            _current_packet.reset(token)
    return run_fast_decision


def _wrap_extra(original):
    def public_extra(role, *, observed_length, spec=None):
        extra = dict(original(role, observed_length=observed_length, spec=spec))
        extra["runtime_capabilities"] = RUNTIME_CAPABILITIES
        frame = extra.get("observation_frame")
        if isinstance(frame, dict):
            frame = dict(frame)
            if role == "train_workflow":
                frame["feature_window"] = (
                    "existing public tools summarize the supplied training PREFIX "
                    "only; they do not compute per-window whole/X/y summaries. "
                    "Explicit window summaries live in execution_observations when "
                    "supplied and are not a new tool capability."
                )
            elif role == "predict_input":
                frame["feature_window"] = (
                    "existing public tools summarize the supplied 192-point serve "
                    "context only. The explicit context summary lives in "
                    "execution_observations when supplied and is not a new tool "
                    "capability."
                )
            extra["observation_frame"] = frame
        packet = _current_packet.get(None)
        if packet is not None and role in ("train_workflow", "predict_input"):
            extra["execution_observations"] = packet
        return extra
    return public_extra


def _wrap_slow(original, run_dir):
    def build_slow_card(
        *, spec, train_records, eval_records_by_origin, formation_scores,
        xy_decomposition,
    ):
        card = original(
            spec=spec, train_records=train_records,
            eval_records_by_origin=eval_records_by_origin,
            formation_scores=formation_scores, xy_decomposition=xy_decomposition,
        )
        card["runtime_capabilities"] = RUNTIME_CAPABILITIES
        card["observable_signature"]["why_no_representative_vector"] = (
            "Per-member vectors stay separate for the %d training and %d evaluation "
            "series; shared-model loss does not identify a training UID's contribution."
            % (len(spec.train_uids), len(spec.eval_uids))
        )
        train_index = {}
        for row in train_records:
            _bind(train_index, str(row.get("series_uid")), _packet_from(row))
        for member in ((card.get("training_batch") or {}).get("members") or []):
            member["execution_observations"] = train_index.get(
                str(member.get("series_uid")), "UNKNOWN",
            )
        eval_index = {}
        buckets = collections.defaultdict(list)
        for origin, rows in eval_records_by_origin.items():
            origin_i = int(origin)
            per_uid = dict((formation_scores.get(origin_i) or {}).get("per_uid_utility") or {})
            for row in rows:
                uid = str(row.get("series_uid"))
                _bind(eval_index, (uid, origin_i), _packet_from(row))
                if not row.get("valid_mechanism_decision"):
                    continue
                util = per_uid.get(uid, "UNKNOWN")
                if not isinstance(util, (int, float)) or isinstance(util, bool) or not np.isfinite(util):
                    continue
                if util > R.MATERIAL:
                    sign = "POSITIVE"
                elif util < -R.MATERIAL:
                    sign = "NEGATIVE"
                else:
                    sign = "NEUTRAL"
                key = json.dumps(
                    {"typed_steps": row.get("typed_steps"), "sign": sign},
                    sort_keys=True, default=str,
                )
                buckets[key].append({"series_uid": uid, "origin": origin_i})
        for item in list(card.get("failing_eval_decisions") or []):
            item["execution_observations"] = eval_index.get(
                (str(item.get("series_uid")), int(item.get("origin"))), "UNKNOWN",
            )
        for item in list(card.get("matched_successes") or []):
            item["execution_observations"] = eval_index.get(
                (str(item.get("series_uid")), int(item.get("origin"))), "UNKNOWN",
            )
        groups = []
        for key in sorted(buckets):
            payload = json.loads(key)
            groups.append({
                "typed_steps": payload.get("typed_steps"),
                "sign": payload.get("sign"),
                "members": buckets[key],
            })
        card["case_groups"] = {
            "by": "full typed_steps including params, and sign",
            "groups": groups,
            "same_program_and_sign_is_not_common_cause": True,
            "training_contributions": "UNKNOWN",
            "not_clustering": True,
            "not_causal_attribution": True,
            "matched_successes_is_not_real_matching": True,
        }
        R.atomic_write_json(Path(run_dir) / "slow_input.json", card)
        return card
    return build_slow_card


@contextmanager
def install_observations(run_dir: Path):
    original_decision = R.run_fast_decision
    original_extra = R.public_extra
    original_slow = R.build_slow_card
    try:
        R.run_fast_decision = _wrap_decision(original_decision)
        R.public_extra = _wrap_extra(original_extra)
        R.build_slow_card = _wrap_slow(original_slow, Path(run_dir))
        yield
    finally:
        R.run_fast_decision = original_decision
        R.public_extra = original_extra
        R.build_slow_card = original_slow
