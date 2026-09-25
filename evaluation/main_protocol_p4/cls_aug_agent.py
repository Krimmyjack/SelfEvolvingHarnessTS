"""DEV-CLS-AUG-OFFLINE-SKILL, LLM part (docs/DEV_CLS_AUG_OFFLINE_SKILL_TASK_2026-09-25.md): can offline experience make a zero-feedback Agent choose
a better training-augmentation program for an FCN classifier on datasets that took no part in learning?

Action per dataset: one of the 11 frozen programs (cls_aug.PROGRAMS). The Agent sees only TRAIN-derived observations of an anonymised dataset
(no name, no archive type), may call class_profile and inspect_program (what a program does to the TRAIN series, never a classifier score),
then commits. Fits and scores come from the server grid (cls_aug_offline_skill.py); TEST is opened once, after every Test delivery is frozen.

Stages (run locally; the fits run on the server between them):
  --source-f0     no-card trajectories on the 8 Source datasets (real experience for Slow)
  --test-f0       no-card trajectories on the 12 Test datasets (independent of any card; started early)
  --cards         two learned cards (Slow: observations + no-card trajectories + Source grid scores) and two naive cards (observations only)
  --select        4 cards x 4 Select datasets; writes select_jobs.json for the server (fb fits of the deliveries)
  --freeze-cards  J from the Select fits (mean Macro-F1 of the delivery), one learned and one naive card frozen
  --test-cards    learned / naive trajectories on the 12 Test datasets
  --freeze-test   all six arms' Test deliveries frozen -> test_jobs.json (full fits) and freeze/deliveries.json
  --readout       after the server scored TEST once
  --smoke
"""
from __future__ import annotations

import os

import argparse
import importlib.util
import json
import re
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_aug_offline_skill as OS
from evaluation.main_protocol_p4 import batch_research_aug_tsfm_offline_skill as TS
from evaluation.main_protocol_p4 import batch_research_domain_aug_decision_priority as DP

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("CLS_AUG_OUT", str(REPO / "_scratch" / "dev_cls_aug_offline_skill")))
ROSTER = REPO / "docs" / "CLS_AUG_ROSTER_V1.json"
TASK = "docs/DEV_CLS_AUG_OFFLINE_SKILL_TASK_2026-09-25.md"
PACKAGE = "DEV-CLS-AUG-OFFLINE-SKILL"
CAPS = {"tokens": 8_000_000, "logical_requests": 900, "http": 1000, "extra_transport": 200, "wall_s": 12 * 3600}
HTTP = int(os.environ.get("SEH_HTTP", "6"))
FAST_MAX_CALLS = 5
FAST_MAX_INSPECT = 3
FAST_MAX_TOKENS = 2000
CARD_LIMIT = 1500
SLOTS = (1, 2)
SEEDS = (0, 1, 2)
now = DP.now
SafeLedger = DP.SafeLedger


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cad = _load("cls_aug_data", REPO / "methods" / "ttha" / "cls_aug_data.py")
_CA = None


def ca():
    global _CA
    if _CA is None:
        _CA = _load("cls_aug", REPO / "methods" / "ttha" / "cls_aug.py")
    return _CA


PROGRAMS = ("None", "Scaling", "MagWarp", "WSlice", "Jitter", "Scaling+MagWarp", "Scaling+WSlice", "Scaling+Jitter", "MagWarp+WSlice",
            "MagWarp+Jitter", "WSlice+Jitter")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    tmp.replace(p)


def roster():
    return read(ROSTER)["sources"]


def datasets(role):
    return sorted(r["representative"] for r in roster() if r["role"] == role)


def anon(ds):
    """Stable anonymous id: role letter + index in the sorted role list (names and archive types never reach an LLM)."""
    for role, L in (("Source", "S"), ("Select", "V"), ("Test", "T")):
        if ds in datasets(role):
            return "%s%02d" % (L, datasets(role).index(ds) + 1)
    raise KeyError(ds)


def load_train(ds):
    z = np.load(ROOT / "data" / (ds + "_TRAIN.npz"))
    return z["values"], z["classes"]


def g3(x):
    return float("%.3g" % float(x))


# ============================================================================= deployment-visible observations (TRAIN only)
CONSUMER = {"model": "FCN (Wang et al. 2017): three 1-D convolution blocks (128/256/128 filters, kernels 8/5/3, batch norm, ReLU), global average "
                     "pooling, softmax; trained from scratch on this dataset's training series",
            "training": "each series is z-normalized (own mean / std); 2000 epochs over the training set, Adam, batch size min(n/10, 16), learning "
                        "rate halved on training-loss plateaus; the weights with the lowest training loss are kept",
            "augmentation": "during training every sample of a batch is replaced, with probability 0.5, by the output of the committed program "
                            "applied to it (a fresh random draw each time); the label is unchanged; the number of samples and updates is the same "
                            "as without augmentation; test series are never augmented",
            "goal": "macro-F1 on unseen test series of the same dataset"}
MENU = {"programs": list(PROGRAMS),
        "operators": {"Scaling": "multiply the whole series by one factor drawn from N(1, 0.1)",
                      "MagWarp": "multiply the series by a smooth random curve: a cubic spline through 6 evenly spaced knots drawn from N(1, 0.2)",
                      "WSlice": "crop a random window of 90% of the length and stretch it back to the full length (linear interpolation)",
                      "Jitter": "add independent Gaussian noise with standard deviation 0.03 to every point (the series is z-normalized)"},
        "composition": "None, one operator, or two different operators applied in the fixed order Scaling -> MagWarp -> WSlice -> Jitter",
        "default": "None (no augmentation)"}
FIELD_DEFINITIONS = {
    "n_train": "number of training series", "length": "points per series", "n_classes": "number of classes",
    "class_count_min / class_count_max": "smallest / largest class size in the training set",
    "lag1_autocorr_median": "median over series of the lag-1 autocorrelation of the z-normalized series (near 1 = smooth)",
    "highfreq_power_fraction_median": "median over series of the share of spectral power in the upper half of the frequency range (noise-like content)",
    "class_template_r2_median": "median over series of 1 - residual energy / energy after subtracting the series' own class-mean template (how stereotyped series are within a class)",
    "misalignment_median": "median over series of |best-matching shift| / length when aligning the series to its class template (shifts up to 25% of the length)",
    "template_max_between_class_corr": "largest correlation between two different class-mean templates (near 1 = classes look alike on average)",
    "nn1_loo_accuracy": "leave-one-out accuracy of 1-nearest-neighbour (Euclidean, z-normalized) on the training set: a data-difficulty statistic, not the FCN",
    "raw_amplitude_class_eta2": "share of the variance of log per-series standard deviation (before normalization) explained by the class; z-normalization removes this information from the FCN input",
    "raw_level_class_eta2": "same for the per-series mean level before normalization"}


def _templates(Z, y):
    cls = sorted(set(y.tolist()))
    return cls, np.stack([Z[y == c].mean(axis=0) for c in cls])


def _misalignment(Z, y, tmpl, cls):
    n, T = Z.shape
    m = max(1, T // 4)
    F = np.fft.rfft(Z, n=2 * T)
    Ft = np.fft.rfft(tmpl[[cls.index(c) for c in y]], n=2 * T)
    cc = np.fft.irfft(F * np.conj(Ft), n=2 * T)
    lags = np.concatenate([np.arange(0, m + 1), np.arange(-m, 0)])
    sel = np.concatenate([cc[:, :m + 1], cc[:, 2 * T - m:]], axis=1)
    return np.abs(lags[np.argmax(sel, axis=1)]) / T


def _nn1_loo(Z, y, chunk=400):
    sq = (Z ** 2).sum(axis=1)
    correct = 0
    for i in range(0, len(Z), chunk):
        d = sq[i:i + chunk, None] + sq[None, :] - 2 * Z[i:i + chunk] @ Z.T
        for k in range(d.shape[0]):
            d[k, i + k] = np.inf
        correct += int((y[d.argmin(axis=1)] == y[i:i + chunk]).sum())
    return correct / len(Z)


def _eta2(v, y):
    tot = ((v - v.mean()) ** 2).sum()
    if tot <= 0:
        return 0.0
    return float(sum(((v[y == c].mean() - v.mean()) ** 2) * (y == c).sum() for c in set(y.tolist())) / tot)


def observations(ds) -> dict:
    p = ROOT / "obs" / ("%s__obs.json" % ds)
    if p.exists():
        return read(p)
    raw, y = load_train(ds)
    Z = cad.znorm(raw)
    n, T = Z.shape
    cls, tmpl = _templates(Z, y)
    counts = np.bincount(y)
    lag1 = np.array([np.corrcoef(z[:-1], z[1:])[0, 1] if z.std() > 0 else 0.0 for z in Z])
    P = np.abs(np.fft.rfft(Z, axis=1)) ** 2
    K = P.shape[1]
    hf = P[:, K // 2:].sum(axis=1) / np.maximum(P[:, 1:].sum(axis=1), 1e-12)
    res = Z - tmpl[[cls.index(c) for c in y]]
    r2 = 1 - (res ** 2).sum(axis=1) / np.maximum((Z ** 2).sum(axis=1), 1e-12)
    tc = np.corrcoef(tmpl) if len(cls) > 1 else np.ones((1, 1))
    off = tc[~np.eye(len(cls), dtype=bool)] if len(cls) > 1 else np.array([1.0])
    sd, mu = raw.std(axis=1), raw.mean(axis=1)
    out = {"n_train": int(n), "length": int(T), "n_classes": int(len(cls)), "class_count_min": int(counts.min()), "class_count_max": int(counts.max()),
           "lag1_autocorr_median": g3(np.median(lag1)), "highfreq_power_fraction_median": g3(np.median(hf)),
           "class_template_r2_median": g3(np.median(r2)), "misalignment_median": g3(np.median(_misalignment(Z, y, tmpl, cls))),
           "template_max_between_class_corr": g3(off.max()), "nn1_loo_accuracy": g3(_nn1_loo(Z, y)),
           "raw_amplitude_class_eta2": g3(_eta2(np.log(np.maximum(sd, 1e-12)), y)), "raw_level_class_eta2": g3(_eta2(mu, y))}
    write(p, out)
    return out


def class_profile(ds) -> dict:
    p = ROOT / "obs" / ("%s__class_profile.json" % ds)
    if p.exists():
        return read(p)
    raw, y = load_train(ds)
    Z = cad.znorm(raw)
    cls, tmpl = _templates(Z, y)
    res = Z - tmpl[[cls.index(c) for c in y]]
    r2 = 1 - (res ** 2).sum(axis=1) / np.maximum((Z ** 2).sum(axis=1), 1e-12)
    mis = _misalignment(Z, y, tmpl, cls)
    tc = np.corrcoef(tmpl) if len(cls) > 1 else np.ones((1, 1))
    rows = []
    for i, c in enumerate(cls):
        m = y == c
        other = [tc[i, j] for j in range(len(cls)) if j != i]
        rows.append({"class": "c%d" % (i + 1), "count": int(m.sum()), "template_r2_median": g3(np.median(r2[m])),
                     "misalignment_median": g3(np.median(mis[m])), "most_similar_other_template_corr": g3(max(other)) if other else None,
                     "raw_std_median": g3(np.median(raw[m].std(axis=1)))})
    out = {"note": "per class, training series only; classes are anonymised", "classes": rows}
    write(p, out)
    return out


def inspect_program(ds, prog) -> dict:
    """What the program does to the TRAIN series when applied (the training applies it to each sample with probability 0.5). No classifier."""
    p = ROOT / "obs" / ("%s__inspect__%s.json" % (ds, prog))
    if p.exists():
        return read(p)
    import torch
    raw, y = load_train(ds)
    Z = cad.znorm(raw)
    cls, tmpl = _templates(Z, y)
    torch.manual_seed(0)
    P = ca().Program(prog, Z.shape[1], torch.device("cpu"))
    with torch.no_grad():
        A = np.concatenate([P(torch.as_tensor(Z[i:i + 512], dtype=torch.float32).unsqueeze(1)).squeeze(1).numpy() for i in range(0, len(Z), 512)])
    A = A.astype(np.float64)
    d = A - Z

    def nearest(X):
        dd = (X ** 2).sum(1)[:, None] + (tmpl ** 2).sum(1)[None, :] - 2 * X @ tmpl.T
        return np.array(cls)[dd.argmin(axis=1)]
    own0, own1 = nearest(Z) == y, nearest(A) == y
    corr = [np.corrcoef(a, z)[0, 1] if a.std() > 0 and z.std() > 0 else 1.0 for a, z in zip(A, Z)]
    per = [float((own1[y == c]).mean() - (own0[y == c]).mean()) for c in cls]
    out = {"program": prog, "applied_to": "every training series once (seed 0); units = z-normalized values",
           "rms_change": g3(np.sqrt((d ** 2).mean())), "median_corr_with_original": g3(np.median(corr)),
           "nearest_class_template_agreement": {"original": g3(own0.mean()), "augmented": g3(own1.mean())},
           "worst_class_change_in_template_agreement": g3(min(per)),
           "note": "template agreement = share of series whose nearest class-mean template (templates of the original training series) is their own class"}
    write(p, out)
    return out


def case_card(ds) -> dict:
    return {"dataset_id": anon(ds), "task": "univariate time-series classification; choose the training-augmentation program", "consumer": CONSUMER,
            "menu": MENU, "training_set_observations": observations(ds), "field_definitions": FIELD_DEFINITIONS}


# ============================================================================= one zero-feedback trajectory
FAST_SYSTEM = ("You choose the training-augmentation program of a time-series classifier for one dataset. This is a zero-feedback deployment: no "
               "accuracy, F1 or other result of the classifier is available to you now or later. Decide from the dataset card and, if you request "
               "them, the class profile and program inspections (what a program does to the training series; they are not classifier results). "
               "If a knowledge card is given, it holds experience learned offline from other datasets; treat it as guidance and check it against "
               "this dataset's observations. Reply with ONE JSON object and nothing else: {\"tool\": \"class_profile\", \"arguments\": {}} or "
               "{\"tool\": \"inspect_program\", \"arguments\": {\"program\": \"<a program from the menu>\"}} or {\"tool\": \"commit\", \"arguments\": "
               "{\"program\": \"<a program from the menu>\", \"reason\": \"<brief, evidence-based>\"}}. You have at most %d requests and at most %d "
               "inspections; the last request must commit." % (FAST_MAX_CALLS, FAST_MAX_INSPECT))
_SCORE_KEYS = r'"(macro_f1[a-z_]*|accuracy|feedback_[a-z_]*|pp_vs_none|scripted_records[a-z_]*|no_card_agent|worst_class_recall[a-z_]*)"'


def parse_action(text, calls_left, inspections):
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    a = json.loads(t)
    if not isinstance(a, dict) or a.get("tool") not in ("class_profile", "inspect_program", "commit"):
        raise ValueError('reply with one JSON object whose "tool" is "class_profile", "inspect_program" or "commit"')
    arg = a.get("arguments") or {}
    if a["tool"] != "commit" and calls_left <= 1:
        raise ValueError("this is the last request: it must commit")
    if a["tool"] == "inspect_program":
        if inspections >= FAST_MAX_INSPECT:
            raise ValueError("no inspections left; commit or request the class profile")
        if arg.get("program") not in PROGRAMS or arg.get("program") == "None":
            raise ValueError("inspect_program needs a non-None program from the menu")
    if a["tool"] == "commit":
        if arg.get("program") not in PROGRAMS:
            raise ValueError("commit needs a program from the menu: %s" % list(PROGRAMS))
        if not str(arg.get("reason") or "").strip():
            raise ValueError('commit needs a brief "reason"')
    return a


def run_trajectory(bdir, ds, card_body, client, stage, arm):
    rp = bdir / "result.json"
    if rp.exists():
        r = read(rp)
        if r["status"] != "CALL_FAILED":
            return r
    bdir.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": case_card(ds), "knowledge_card": card_body, "transcript": []}
    trace, calls, corrections, tokens, insp = [], 0, 0, 0, 0
    status, choice, reason = "INCOMPLETE", None, None
    while calls < FAST_MAX_CALLS:
        payload["requests_left_including_this"] = FAST_MAX_CALLS - calls
        payload["inspections_left"] = FAST_MAX_INSPECT - insp
        if re.search(_SCORE_KEYS, json.dumps(payload)):
            raise PermissionError("a Fast payload carries a downstream score field")
        try:
            text, rec = client.call("fast", bdir.name, payload, FAST_SYSTEM, max_tokens=FAST_MAX_TOKENS,
                                    meta={"stage": stage, "dataset": anon(ds), "arm": arm, "call": calls + 1}, request_timeout=300.0)
        except (rt.llm.AccountFault, rt.llm.TransportFault, OS.UnknownUsageBlock) as exc:
            status = "CALL_FAILED"
            trace.append({"call": calls + 1, "fault": type(exc).__name__, "detail": str(exc)[:200]})
            break
        calls += 1
        tokens += (rec.get("prompt_tokens") or 0) + (rec.get("completion_tokens") or 0)
        try:
            act = parse_action(text, FAST_MAX_CALLS - calls + 1, insp)
        except (ValueError, json.JSONDecodeError) as exc:
            trace.append({"call": calls, "response_excerpt": (text or "")[:600], "rejected": str(exc)[:300]})
            if corrections < 1:
                corrections += 1
                payload["transcript"].append({"your_previous_reply_rejected": str(exc)[:300]})
            continue
        trace.append({"call": calls, "action": act})
        if act["tool"] == "class_profile":
            payload["transcript"].append({"tool": "class_profile", "output": class_profile(ds)})
            continue
        if act["tool"] == "inspect_program":
            insp += 1
            payload["transcript"].append({"tool": "inspect_program", "output": inspect_program(ds, act["arguments"]["program"])})
            continue
        status, choice, reason = "COMPLETE", act["arguments"]["program"], act["arguments"]["reason"]
        break
    r = {"dataset": ds, "dataset_id": anon(ds), "arm": arm, "stage": stage, "status": status, "choice": choice,
         "delivered": choice if status == "COMPLETE" else "None", "reason": reason, "calls": calls, "corrections": corrections,
         "inspected": [t["action"]["arguments"]["program"] for t in trace if t.get("action", {}).get("tool") == "inspect_program"],
         "used_class_profile": any(t.get("action", {}).get("tool") == "class_profile" for t in trace), "tokens": tokens, "trace": trace,
         "finished_local": now()}
    write(rp, r)
    print("TRAJ", stage, bdir.name, status, choice, "calls", calls, "tokens", tokens, flush=True)
    return r


def run_stage(stage, jobs, card_of, client):
    sdir = ROOT / "agent" / stage
    results, lock = {}, threading.Lock()

    def go(todo):
        q, ql = list(todo), threading.Lock()

        def worker():
            while True:
                with ql:
                    if not q:
                        return
                    ds, arm = q.pop(0)
                r = run_trajectory(sdir / ("%s__%s" % (ds, arm)), ds, card_of(ds, arm), client, stage, arm)
                with lock:
                    results[(ds, arm)] = r
        ts = [threading.Thread(target=worker, daemon=True) for _ in range(HTTP)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    go(jobs)
    failed = [k for k, r in results.items() if r["status"] == "CALL_FAILED"]
    if failed:
        print("TRANSPORT_RESUME_WAIT", stage, len(failed), TS.RESUME_WAIT_S, flush=True)
        time.sleep(TS.RESUME_WAIT_S)
        go(failed)
    failed = ["%s__%s" % k for k, r in results.items() if r["status"] == "CALL_FAILED"]
    summ = {"stage": stage, "finished_local": now(), "jobs": len(jobs), "complete": sum(r["status"] == "COMPLETE" for r in results.values()),
            "incomplete": sum(r["status"] == "INCOMPLETE" for r in results.values()), "call_failed": failed}
    write(sdir / "dispatch_summary.json", summ)
    print("STAGE_DONE", json.dumps(summ), flush=True)
    if failed:
        raise RuntimeError("%s: transport-interrupted trajectories %s; resume after the fault is cleared" % (stage, failed[:5]))
    return results


# ============================================================================= Source grid scores (server fits, pulled)
def fb_scores(ds) -> dict:
    """{program: {seed: metrics}} from the fb fits of one Source / Select dataset."""
    out = {}
    for p in (ROOT / "fb" / ds).glob("*.json"):
        r = read(p)
        out.setdefault(r["prog"], {})[r["seed"]] = r["feedback_metrics"]
    return out


def source_records(ds) -> dict:
    sc = fb_scores(ds)
    base = [sc["None"][s]["macro_f1"] for s in SEEDS]
    rec = {}
    for prog in PROGRAMS:
        if prog == "None":
            continue
        d = [100 * (sc[prog][s]["macro_f1"] - sc["None"][s]["macro_f1"]) for s in SEEDS]
        w = [100 * (sc[prog][s]["worst_class_recall"] - sc["None"][s]["worst_class_recall"]) for s in SEEDS]
        rec[prog] = {"macro_f1_pp_vs_none_mean": g3(statistics.fmean(d)), "macro_f1_pp_vs_none_by_seed": [g3(x) for x in d],
                     "worst_class_recall_pp_vs_none_mean": g3(statistics.fmean(w))}
    return {"none_macro_f1_mean": g3(100 * statistics.fmean(base)), "none_macro_f1_seed_sd": g3(100 * statistics.pstdev(base)),
            "feedback_size": sum(1 for _ in read(next((ROOT / "fb" / ds).glob("None__s0.json")))["feedback_true"]), "programs": rec}


# ============================================================================= Slow (learned) and naive cards
SLOW_SYSTEM = ("You write a knowledge card for a zero-feedback agent that chooses the training-augmentation program (one of 11: None, four "
               "operators, six ordered pairs) of an FCN time-series classifier on a new dataset. The census holds eight development datasets. For "
               "each: its training-set observations, what the no-card agent did (inspections, choice and reason) and SCRIPTED RECORDS: for every "
               "program, the change in feedback macro-F1 against no augmentation (percentage points, mean and per training seed) and in the worst "
               "class recall, from an evaluator run on a held-out part of the training set. The scripted records come from an evaluator, not from "
               "an agent, and a deployed agent never sees any such number. Write guidance a deployed agent can apply from the observations it will "
               "actually see (the same fields, the class profile, program inspections): which program to prefer by default, which observable "
               "conditions change the choice, what to avoid. Seed-to-seed spread is large on small feedback sets: do not build rules on differences "
               "that the seeds do not support. Do not name datasets, do not claim more certainty than the census supports. KEEP (no card) is "
               "allowed. Reply with ONE JSON object: {\"decision\": \"PROPOSE\" or \"KEEP\", \"card\": {\"title\": \"<short>\", \"body\": \"<at most "
               "%d characters>\"}}." % CARD_LIMIT)
NAIVE_SYSTEM = ("You write a knowledge card for a zero-feedback agent that chooses the training-augmentation program (one of 11: None, four "
                "operators, six ordered pairs) of an FCN time-series classifier on a new dataset. No classifier result of any kind exists for you: "
                "you see the deployment description (classifier, menu, the agent's tools) and the training-set observations of eight development "
                "datasets. Using your general knowledge of time-series classification and data augmentation, write guidance the agent can apply "
                "from the observations it will see: the default choice, the observable conditions that change it, what to avoid. Do not name "
                "datasets. KEEP (no card) is allowed. Reply with ONE JSON object: {\"decision\": \"PROPOSE\" or \"KEEP\", \"card\": {\"title\": "
                "\"<short>\", \"body\": \"<at most %d characters>\"}}." % CARD_LIMIT)
SLOT_FRAME = {1: "Write a short, rule-first card: the default, then at most three observable conditions that change it.",
              2: "Write a card that states the decisive observations, the default, the exceptions and the known failure modes."}
DEPLOYMENT = {"consumer": CONSUMER, "menu": MENU, "agent_tools": ["class_profile", "inspect_program (at most %d)" % FAST_MAX_INSPECT, "commit"],
              "field_definitions": FIELD_DEFINITIONS}


def census() -> dict:
    rows = []
    for ds in datasets("Source"):
        r = read(ROOT / "agent" / "source" / ("%s__f0" % ds) / "result.json")
        rows.append({"dataset_id": anon(ds), "training_set_observations": observations(ds),
                     "no_card_agent": {"inspected": r["inspected"], "used_class_profile": r["used_class_profile"], "choice": r["delivered"],
                                       "status": r["status"], "reason": r["reason"]},
                     "scripted_records": source_records(ds)})
    return {"deployment": DEPLOYMENT, "n_datasets": len(rows), "datasets": rows}


def naive_payload() -> dict:
    return {"deployment": DEPLOYMENT, "development_observations": {"note": "observations only; no classifier was scored for you",
                                                                    "datasets": [{"dataset_id": anon(ds), "training_set_observations": observations(ds)}
                                                                                 for ds in datasets("Source")]}}


def validate_card(obj):
    if not isinstance(obj, dict) or obj.get("decision") not in ("PROPOSE", "KEEP"):
        raise ValueError('reply with {"decision": "PROPOSE" or "KEEP", "card": {...}}')
    if obj["decision"] == "KEEP":
        return {"status": "NO_CARD", "body": None, "title": None}
    card = obj.get("card") or {}
    body, title = str(card.get("body") or "").strip(), str(card.get("title") or "").strip()
    if not body or len(body) > CARD_LIMIT:
        raise ValueError("card.body must be non-empty and at most %d characters (got %d)" % (CARD_LIMIT, len(body)))
    # dataset names never enter any LLM input (anonymised ids only), so only the anonymised ids are checked; a real-name substring check
    # produced a false positive on "crop" (the WSlice description) and was removed (instrument correction 1, 2026-09-25)
    if re.search(r"\b[STV]\d\d\b", body):
        raise ValueError("the card must not name datasets")
    return {"status": "PROPOSED", "body": body, "title": title}


def card_calls(client, kind):
    """kind 'learned' (census) or 'naive' (observations only); two slots each, same framings."""
    res, lock = {}, threading.Lock()
    system = SLOW_SYSTEM if kind == "learned" else NAIVE_SYSTEM
    base = census() if kind == "learned" else naive_payload()
    if kind == "naive" and re.search(_SCORE_KEYS, json.dumps(base)):
        raise PermissionError("naive payload carries a downstream score or agent outcome")

    def one(k):
        p = ROOT / "cards" / ("%s_%d.json" % (kind, k))
        if p.exists():
            r = read(p)
        else:
            payload = {"census" if kind == "learned" else "input": base, "framing": SLOT_FRAME[k]}
            r, errs = None, []
            for i in range(2):
                try:
                    text, rec = client.call("slow_" + kind, "%s_%d" % (kind, k), payload, system, max_tokens=4000, meta={"kind": kind, "slot": k},
                                            request_timeout=600.0)
                except (rt.llm.AccountFault, rt.llm.TransportFault, OS.UnknownUsageBlock) as exc:
                    r = {"status": "CALL_FAILED", "fault": type(exc).__name__}
                    break
                try:
                    r = {**validate_card(json.loads(re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip())), "attempts": i + 1,
                         "raw": text}
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    errs.append(str(exc)[:300])
                    payload = {**payload, "correction": {"previous_output_excerpt": (text or "")[:3000], "error": str(exc)[:300],
                                                         "instruction": "Correct this format / contract error only; KEEP is allowed."}}
            if r is None:
                r = {"status": "PARSE_OR_VALIDATION_FAILED", "errors": errs, "body": None}
            r.update({"kind": kind, "slot": k, "written_local": now()})
            if r["status"] != "CALL_FAILED":
                write(p, r)
        with lock:
            res["%s_%d" % (kind, k)] = r
        print("CARD", kind, k, r["status"], flush=True)
    ts = [threading.Thread(target=one, args=(k,), daemon=True) for k in SLOTS]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if any(r["status"] == "CALL_FAILED" for r in res.values()):
        raise RuntimeError("a card call failed technically; resume after the fault is cleared")
    return res


def card_body(kind, slot):
    r = read(ROOT / "cards" / ("%s_%d.json" % (kind, slot)))
    return r.get("body")


# ============================================================================= Select, freezing, Test
SELECT_ARMS = ("l1", "l2", "n1", "n2")


def select_card(ds, arm):
    return card_body("learned" if arm[0] == "l" else "naive", int(arm[1]))


def jobs_file(name, jobs):
    uniq = sorted({(j["kind"], j["ds"], j["prog"], j["seed"]) for j in jobs})
    write(ROOT / name, [{"kind": k, "ds": d, "prog": p, "seed": s} for k, d, p, s in uniq])
    print("JOBS", name, len(uniq), flush=True)


def select_jobs(arms=SELECT_ARMS, name="select_jobs.json"):
    jobs = []
    for ds in datasets("Select"):
        for arm in arms:
            r = read(ROOT / "agent" / "select" / ("%s__%s" % (ds, arm)) / "result.json")
            jobs += [{"kind": "fb", "ds": ds, "prog": r["delivered"], "seed": s} for s in SEEDS]
    jobs_file(name, jobs)


def freeze_cards():
    out = ROOT / "freeze" / "cards_frozen.json"
    if out.exists():
        return read(out)
    J, tok = {}, {}
    for arm in SELECT_ARMS:
        for ds in datasets("Select"):
            r = read(ROOT / "agent" / "select" / ("%s__%s" % (ds, arm)) / "result.json")
            sc = fb_scores(ds)[r["delivered"]]
            J.setdefault(arm, []).append(statistics.fmean(sc[s]["macro_f1"] for s in SEEDS))
            tok.setdefault(arm, []).append(r["tokens"])
    Jm = {a: statistics.fmean(v) for a, v in J.items()}
    mt = {a: statistics.fmean(v) for a, v in tok.items()}
    pick = lambda arms: max(arms, key=lambda a: (round(Jm[a], 12), -mt[a], -int(a[1])))
    L, N = pick(("l1", "l2")), pick(("n1", "n2"))
    rec = {"frozen_local": now(), "rule": "J = mean over the 4 Select datasets of the mean feedback macro-F1 (3 seeds) of the delivered program; "
                                          "learned and naive cards are selected separately among their two slots; ties -> fewer tokens -> lower slot",
           "J": Jm, "mean_tokens": mt, "learned": {"arm": L, "body": select_card(None, L)}, "naive": {"arm": N, "body": select_card(None, N)}}
    write(out, rec)
    return rec


def test_card(ds, arm):
    if arm == "f0":
        return None
    fr = read(ROOT / "freeze" / "cards_frozen.json")
    return fr["learned"]["body"] if arm == "f_learned" else fr["naive"]["body"]


def source_best_fixed() -> dict:
    means = {prog: statistics.fmean(source_records(ds)["programs"][prog]["macro_f1_pp_vs_none_mean"] for ds in datasets("Source"))
             for prog in PROGRAMS if prog != "None"}
    means["None"] = 0.0
    best = max(means, key=lambda p: (means[p], -PROGRAMS.index(p)))
    return {"rule": "argmax over the 11 programs of the mean (over the 8 Source datasets) of the mean feedback macro-F1 change vs None; None counts as 0",
            "means_pp": {k: g3(v) for k, v in means.items()}, "program": best}


def freeze_test():
    out = ROOT / "freeze" / "deliveries.json"
    if out.exists():
        return read(out)
    ra = read(ROOT / "random_arm.json")["draws"]
    fixed = source_best_fixed()
    dl, jobs = {}, []
    for ds in datasets("Test"):
        d = {"none": "None", "fixed_source_best": fixed["program"], "random": ra[ds]}
        for arm in ("f0", "f_naive", "f_learned"):
            r = read(ROOT / "agent" / "test" / ("%s__%s" % (ds, arm)) / "result.json")
            assert r["status"] != "CALL_FAILED"
            d[arm] = r["delivered"]
        dl[ds] = d
        jobs += [{"kind": "full", "ds": ds, "prog": p, "seed": s} for p in set(d.values()) for s in SEEDS]
    rec = {"frozen_local": now(), "arms": ["none", "f0", "f_naive", "f_learned", "fixed_source_best", "random"], "fixed_source_best": fixed,
           "deliveries": dl, "note": "every Test delivery is frozen here, before any TEST member is opened"}
    write(out, rec)
    jobs_file("test_jobs.json", jobs)
    return rec


# ============================================================================= readout (after the one-shot TEST scoring on the server)
def boot(vals, B=4000, seed=2026092504):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    m = [float(v[rng.integers(0, len(v), len(v))].mean()) for _ in range(B)]
    return {"lo95": float(np.percentile(m, 2.5)), "hi95": float(np.percentile(m, 97.5)), "B": B}


def readout() -> dict:
    fr = read(ROOT / "freeze" / "deliveries.json")
    ts = read(ROOT / "test_scores.json")["scores"]
    arms = fr["arms"]
    per = {}
    for ds, d in fr["deliveries"].items():
        m = {a: {k: statistics.fmean(ts[ds][d[a]][str(s)][k] for s in SEEDS) for k in ("macro_f1", "accuracy", "worst_class_recall")} for a in arms}
        per[ds] = {"dataset_id": anon(ds), "delivered": d, "metrics": m}
    pairs = {}
    for a, b in (("f_learned", "f0"), ("f_learned", "f_naive"), ("f_naive", "f0"), ("f_learned", "none"), ("f_learned", "fixed_source_best"),
                 ("f_learned", "random"), ("f0", "none"), ("fixed_source_best", "none"), ("random", "none"), ("f_naive", "none")):
        diffs = [100 * (per[ds]["metrics"][a]["macro_f1"] - per[ds]["metrics"][b]["macro_f1"]) for ds in per]
        pairs["%s_minus_%s" % (a, b)] = {"macro_f1_pp_mean": statistics.fmean(diffs), "wins_same_losses": [sum(x > 1e-9 for x in diffs), sum(abs(x) <= 1e-9 for x in diffs),
                                                                                                       sum(x < -1e-9 for x in diffs)],
                                         "bootstrap_95_over_datasets": boot(diffs),
                                         "accuracy_pp_mean": statistics.fmean(100 * (per[ds]["metrics"][a]["accuracy"] - per[ds]["metrics"][b]["accuracy"]) for ds in per),
                                         "worst_class_recall_pp_mean": statistics.fmean(100 * (per[ds]["metrics"][a]["worst_class_recall"] - per[ds]["metrics"][b]["worst_class_recall"]) for ds in per)}
    arms_abs = {a: {"macro_f1_mean": statistics.fmean(100 * per[ds]["metrics"][a]["macro_f1"] for ds in per)} for a in arms}
    beh = {}
    for st, arm_list in (("source", ["f0"]), ("select", list(SELECT_ARMS)), ("test", ["f0", "f_naive", "f_learned"])):
        for a in arm_list:
            rs = [read(p) for p in (ROOT / "agent" / st).glob("*__%s/result.json" % a)]
            if rs:
                beh["%s:%s" % (st, a)] = {"n": len(rs), "choices": dict(sorted(OS.collections_counter(r["delivered"] for r in rs).items())),
                                          "incomplete": sum(r["status"] != "COMPLETE" for r in rs), "mean_calls": statistics.fmean(r["calls"] for r in rs),
                                          "mean_tokens": statistics.fmean(r["tokens"] for r in rs), "mean_inspections": statistics.fmean(len(r["inspected"]) for r in rs)}
    led = read(ROOT / "budget.json")
    res = {"package": PACKAGE, "written_local": now(), "cards": read(ROOT / "freeze" / "cards_frozen.json"), "fixed_source_best": fr["fixed_source_best"],
           "arms_abs": arms_abs, "pairs": pairs, "behaviour": beh, "per_dataset": per,
           "costs": {k: led.get(k) for k in ("llm_requests", "llm_http_attempts", "llm_tokens_in", "llm_tokens_out", "llm_tokens_unknown")}}
    write(ROOT / "result.json", res)
    (ROOT / "tables.md").write_text(tables_md(res), encoding="utf-8")
    return res


ARM_CN = {"none": "不增强", "f0": "无卡 Agent", "f_naive": "朴素卡 Agent", "f_learned": "经验卡 Agent", "fixed_source_best": "Source 最佳固定程序",
          "random": "冻结随机程序"}


def tables_md(r) -> str:
    L = ["## Test（12 个来源互斥的多分类数据集，官方 TEST 一次打开；Macro-F1，3 个训练 seed 平均）", "",
         "| 臂 | 平均 Macro-F1 (%) |", "|---|---:|"]
    for a, v in r["arms_abs"].items():
        L.append("| %s | %.2f |" % (ARM_CN.get(a, a), v["macro_f1_mean"]))
    L += ["", "| 比较 | Macro-F1 差 (pp) | 胜/同/负 | 95% 区间（按数据集重采样） | Accuracy 差 | 最差类召回差 |", "|---|---:|---|---|---:|---:|"]
    for k, v in r["pairs"].items():
        a, b = k.split("_minus_")
        ci = v["bootstrap_95_over_datasets"]
        L.append("| %s − %s | %+.2f | %s | [%+.2f, %+.2f] | %+.2f | %+.2f |" % (ARM_CN.get(a, a), ARM_CN.get(b, b), v["macro_f1_pp_mean"],
                                                                         "/".join(map(str, v["wins_same_losses"])), ci["lo95"], ci["hi95"],
                                                                         v["accuracy_pp_mean"], v["worst_class_recall_pp_mean"]))
    L += ["", "## 逐数据集（Macro-F1 %，交付程序）", "", "| 数据集 | " + " | ".join(ARM_CN[a] for a in ARM_CN) + " |", "|---|" + "---|" * len(ARM_CN)]
    for ds, p in sorted(r["per_dataset"].items(), key=lambda kv: kv[1]["dataset_id"]):
        L.append("| %s %s | " % (p["dataset_id"], ds) + " | ".join("%.1f (%s)" % (100 * p["metrics"][a]["macro_f1"], p["delivered"][a]) for a in ARM_CN) + " |")
    L += ["", "## 行为", ""]
    for k, v in r["behaviour"].items():
        L.append("- %s：n=%d，未合法提交 %d，平均请求 %.1f，平均检查 %.1f，平均 token %.0f；交付 %s" % (k, v["n"], v["incomplete"], v["mean_calls"], v["mean_inspections"],
                                                                                    v["mean_tokens"], json.dumps(v["choices"], ensure_ascii=False)))
    L += ["", "Source 最佳固定程序：%s（%s）" % (r["fixed_source_best"]["program"], r["fixed_source_best"]["rule"]),
          "", "选中的卡：经验卡 %s，朴素卡 %s；J = %s" % (r["cards"]["learned"]["arm"], r["cards"]["naive"]["arm"], json.dumps({k: round(100 * v, 2) for k, v in r["cards"]["J"].items()})),
          "", "LLM 成本：%s" % json.dumps(r["costs"])]
    return "\n".join(L) + "\n"


# ============================================================================= controller
def make_client(led, stage):
    return TS.ZFStreamClient(led, ROOT / "agent" / stage / "llm", http_cap=CAPS["http"], stage=stage, incidents=ROOT / "incidents",
                             extra_cap=CAPS["extra_transport"])


def ledger():
    return SafeLedger(ROOT / "budget.json", max_fit_attempts=1, max_llm_requests=CAPS["logical_requests"], max_llm_tokens=CAPS["tokens"],
                      max_wall_s=CAPS["wall_s"], max_retries=0)


def freeze_plan():
    p = ROOT / "agent_plan.json"
    if not p.exists():
        write(p, {"package": PACKAGE, "task": TASK, "frozen_local": now(), "programs": list(PROGRAMS), "fast": {"system": FAST_SYSTEM, "max_calls": FAST_MAX_CALLS,
                  "max_inspect": FAST_MAX_INSPECT, "max_tokens": FAST_MAX_TOKENS, "consumer": CONSUMER, "menu": MENU},
                  "slow": {"system": SLOW_SYSTEM, "naive_system": NAIVE_SYSTEM, "slots": SLOT_FRAME, "card_limit": CARD_LIMIT}, "caps": CAPS,
                  "http_in_flight": HTTP, "model": OS.MODEL, "incomplete_rule": "no legal commit -> None delivered and counted separately",
                  "same_for_all_arms": "dataset card, tools, menu, request budget; only the knowledge card differs",
                  "anonymisation": "dataset names and archive types never reach an LLM (ids S01-S08 / V01-V04 / T01-T12)"})


def stage_run(name, jobs, card_of):
    freeze_plan()
    DP.set_pools(2, HTTP)
    with DP.PackageLock(ROOT, "agent_" + name):
        return run_stage(name, jobs, card_of, make_client(ledger(), name))


def smoke():
    freeze_plan()
    ds = datasets("Source")[0]
    seen = []

    class FC:
        def __init__(self, replies):
            self.r = list(replies)

        def call(self, role, unit, payload, system, *, max_tokens, meta, request_timeout):
            seen.append(json.dumps(payload))
            return self.r.pop(0), {"prompt_tokens": 10, "completion_tokens": 5}
    import shutil
    sm = ROOT / "agent" / "smoke"
    if sm.exists():
        shutil.rmtree(sm)
    r1 = run_trajectory(sm / "a", ds, None, FC(['{"tool": "class_profile", "arguments": {}}', '{"tool": "inspect_program", "arguments": {"program": "WSlice+Jitter"}}',
                                                '{"tool": "commit", "arguments": {"program": "Jitter", "reason": "smoke"}}']), "smoke", "f0")
    r2 = run_trajectory(sm / "b", ds, "card text", FC(["no json", '{"tool": "inspect_program", "arguments": {"program": "Scaling"}}'] +
                                                       ['{"tool": "class_profile", "arguments": {}}'] * 4), "smoke", "f_learned")
    checks = {"commit": r1["status"] == "COMPLETE" and r1["delivered"] == "Jitter" and r1["inspected"] == ["WSlice+Jitter"] and r1["used_class_profile"],
              "incomplete_defaults_to_none": r2["status"] == "INCOMPLETE" and r2["delivered"] == "None" and r2["corrections"] == 1,
              "no_scores_in_payloads": not any(re.search(_SCORE_KEYS, s) for s in seen),
              "anonymised": not any(n in s for s in seen for n in [r["representative"] for r in roster()]),
              "card_reaches_fast": any('"knowledge_card": "card text"' in s for s in seen)}
    shutil.rmtree(sm)
    print(json.dumps(checks, indent=1), "\nALL PASS:", all(checks.values()))
    return checks


def main():
    ap = argparse.ArgumentParser()
    for a in ("smoke", "source_f0", "test_f0", "cards", "select", "freeze_cards", "test_cards", "freeze_test", "readout", "obs", "naive_cards"):
        ap.add_argument("--" + a.replace("_", "-"), action="store_true")
    ap.add_argument("--select-arms", default=",".join(SELECT_ARMS))
    a = ap.parse_args()
    if a.naive_cards:
        freeze_plan()
        DP.set_pools(2, HTTP)
        with DP.PackageLock(ROOT, "agent_cards"):
            card_calls(make_client(ledger(), "cards"), "naive")
    if a.obs:
        for r in roster():
            observations(r["representative"])
            class_profile(r["representative"])
        print("observations ready")
    if a.smoke:
        smoke()
    if a.source_f0:
        stage_run("source", [(ds, "f0") for ds in datasets("Source")], lambda d, x: None)
    if a.test_f0:
        stage_run("test", [(ds, "f0") for ds in datasets("Test")], lambda d, x: None)
    if a.cards:
        freeze_plan()
        DP.set_pools(2, HTTP)
        with DP.PackageLock(ROOT, "agent_cards"):
            led = ledger()
            card_calls(make_client(led, "cards"), "learned")
            card_calls(make_client(led, "cards"), "naive")
    if a.select:
        arms = tuple(a.select_arms.split(","))
        stage_run("select", [(ds, arm) for ds in datasets("Select") for arm in arms], select_card)
        select_jobs(arms, "select_jobs_%s.json" % "_".join(arms))
    if a.freeze_cards:
        print(json.dumps({k: v for k, v in freeze_cards().items() if k in ("J", "learned", "naive")}, indent=1, ensure_ascii=False)[:3000])
    if a.test_cards:
        stage_run("test", [(ds, arm) for ds in datasets("Test") for arm in ("f_naive", "f_learned")], test_card)
    if a.freeze_test:
        print(json.dumps(freeze_test()["deliveries"], indent=1))
    if a.readout:
        r = readout()
        print(json.dumps({"arms": r["arms_abs"], "pairs": {k: {kk: v[kk] for kk in ("macro_f1_pp_mean", "wins_same_losses", "bootstrap_95_over_datasets")}
                                                            for k, v in r["pairs"].items()}}, indent=1))


if __name__ == "__main__":
    main()
