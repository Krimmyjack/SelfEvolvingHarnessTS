"""DEV-DOMAIN-AUG-DECISION-PRIORITY (docs/DEV_DOMAIN_AUG_DECISION_PRIORITY_TASK_2026-09-20.md): learn observation / construction /
evaluation priorities from the parent package's 12 completed entity cases per domain (no Source rerun), three independent Slow
proposals per domain, new Select (V03/V04) of old card vs the proposals, freeze W_new / W_old / Fixed_dev / H_deploy, MetaTest on
8 new entity cases (Q03-Q06 x 2 domains): F0 / F_old / F_new / RandomSearch_B4 / Fixed_dev + the four public references.

Reused unchanged (read-only imports): the parent study `batch_research_domain_aug_entity_split` (ES) for the case profile workers
(build / material / fit / label CLI), CaseAdapter tool semantics, RandomSearch_B4 draw, Fixed_dev branch, evidence helpers and the
readout arithmetic; the Fast controller (batch_research.run_job through run_batch_research_v1.branch / resume_branch); the
Skill carrier (domain_skill); the tempo_aug primitives. New here (opt-in): the parallel coordinator (one package lock, one
thread-safe ledger, per-request receipts, one HTTP pool, one numeric pool), the request-side overview dedupe, the common
material-semantics sentence, the decision-point census of the parent's 12 cases, three single-card Slow proposals per domain,
the J / H_deploy freeze and the G readout of task §9.

  --preflight | --smoke | --wiring | --run | --resume-stage STAGE [--accept-unknown-usage] [--restart b1,b2] | --result | --monitor
  subprocess entries: --stage-worker CFG | --lock-probe ROOT
Numerical workers are the parent study's CLI (evaluation.main_protocol_p4.batch_research_domain_aug_entity_split --worker-*); the
coordinator never imports torch.
"""
from __future__ import annotations

import os

KMP_AT_START = os.environ.get('KMP_DUPLICATE_LIB_OK')

import argparse
import copy
import json
import math
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha.batch_base import budget, context, policy, spec, tempo_aug as ta
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_domain_skill as dsks
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import run_batch_research_v1 as base
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_skill as W            # read-only helpers: compression, text check
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_learning_loop as LL    # read-only helpers: review parsing, ref ranges
from evaluation.main_protocol_p4 import batch_research_domain_aug_entity_split as ES              # read-only: parent study (workers, adapter, arms, readout helpers)

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_domain_aug_decision_priority'
MODULE = 'evaluation.main_protocol_p4.batch_research_domain_aug_decision_priority'
WORKER_MODULE = ES.MODULE                                   # the parent's worker CLI (build / material / fit / label)
TASK = 'docs/DEV_DOMAIN_AUG_DECISION_PRIORITY_TASK_2026-09-20.md'
SPLIT_DOC = REPO / 'docs' / 'DOMAIN_AUG_DECISION_PRIORITY_V1.json'
PARENT_SPLIT_DOC = ES.SPLIT_DOC
PARENT_ROOT = ES.ROOT
PARENT_PACKAGE = ES.PACKAGE
PACKAGE = 'DEV-DOMAIN-AUG-DECISION-PRIORITY'
EXPOSURE = ec.EXPOSURE
DOMAINS = ES.DOMAINS
SEEDS = ES.SEEDS
PUBLIC = ec.PUBLIC
PUBLIC_STEPS = ec.PUBLIC_STEPS
LIMITS = dict(ES.LIMITS)                                    # 16 calls / 24 tools / 4 new evaluations per Fast
MAX_OUTPUT_TOKENS = ES.MAX_OUTPUT_TOKENS                    # 12,000
MAX_TOOL_CORRECTIONS = ES.MAX_TOOL_CORRECTIONS
EVIDENCE_ROUNDTRIP = ES.EVIDENCE_ROUNDTRIP
REQUEST_TRACE_DEDUPE = True                                 # task §3: lossless request-side dedupe of the overview copies
MODEL = ES.MODEL
BODY_LIMIT = ES.BODY_LIMIT                                  # 6000 characters per card
TOL = ES.TOL
T975_DF2 = ES.T975_DF2
FIT_TIMEOUT_S = ES.FIT_TIMEOUT_S
FIT_THREADS = ES.FIT_THREADS                                # 8 (unchanged; the workers read the parent's constant)
WIRING_COMPOSITION = ES.WIRING_COMPOSITION
WIRING_CASES = ('D01_S01', 'D02_S01')
PROPOSALS = ('N1', 'N2', 'N3')
SELECT_ARMS = ('w_old', 'n1', 'n2', 'n3')
TEST_FAST_ARMS = ('f0', 'f_old', 'f_new')
CONTROL_ARMS = ('random', 'fixed_dev')
ARM_LABEL = {'f0': 'F0', 'f_old': 'F_old', 'f_new': 'F_new', 'random': 'RandomSearch_B4', 'fixed_dev': 'Fixed_dev', 'w_old': 'W_old', 'n1': 'N1', 'n2': 'N2', 'n3': 'N3'}
NEW_ROLES = {'select': 2, 'test': 4}
NEW_CASE_INDEX = (12, 13, 14, 15, 16, 17)
CONCURRENCY = {'cases': 4, 'http': 4, 'numeric': 3, 'fit_threads': FIT_THREADS}
TOTAL = {'max_fit_attempts': 756, 'max_llm_requests': 652, 'max_llm_tokens': 60_000_000, 'max_wall_s': 48 * 3600, 'max_retries': 6}   # tokens / wall: non-binding guards (warnings below)
HTTP_CAP = 656
TRANSPORT_RETRIES = 4                                       # package-wide, at most one per logical request
ALLOC = {'wiring': {'fits': 6, 'requests': 0}, 'slow': {'fits': 0, 'requests': 12}, 'select': {'fits': 240, 'requests': 256}, 'metatest': {'fits': 504, 'requests': 384}}
TOKEN_WARNINGS = (12_000_000, 20_000_000)
PAID_WALL_WARNINGS_S = (4 * 3600, 8 * 3600)
FAST_TOKEN_MONITOR = (250_000, 500_000, 1_000_000)          # cumulative per-trajectory monitoring only (task §8.1: no hard per-Fast cap)
RANDOM_SEED_ROOT = ES.RANDOM_SEED_ROOT
CENSUS_BYTE_TARGET = 530_000                                # ~205k tokens: the envelope the backend accepted in the parent package (495 KB -> 191k tokens)
CONTRACTS = copy.deepcopy(ES.CONTRACTS)
ALLOWED_FEATURES = ES.ALLOWED_FEATURES
FIELD_DEFINITIONS = ES.FIELD_DEFINITIONS

MATERIAL_SEMANTICS = ('材料检查用来确认处理是否生效、作用位置是否符合假说、数值是否有效。更平滑、峰值降低或改动较大，本身不能证明下游有害；同样也不能证明有益。'
                      '你可以基于语义和有限预算降低某候选的实验优先级，但未取得其下游反馈时应记作未验证，不写成已证伤害。优先让有区分力的假说取得一次真实反馈，再决定下一步；'
                      '必要时继续观察。无需机械地为每份材料检查多个实体，也无需评估所有构造。当前 remaining 已给出调用、工具和评估余量，安排时为提交留出余量。'
                      'C_A 是有限支持证据，历史 Skill 是可修正的先验，两者都不保证未来排序。\n'
                      '(Material inspection confirms that a treatment took effect, acted where the hypothesis says and produced valid numbers. Smoother, lower peaks or a '
                      'larger change do not by themselves show downstream harm - nor benefit. You may lower a candidate\'s experimental priority on semantics and the '
                      'limited budget, but without its downstream feedback it stays UNVERIFIED, never "shown harmful". Prefer giving one discriminating hypothesis a real '
                      'feedback before deciding the next step; keep observing when needed. No need to inspect several entities per material mechanically, nor to evaluate '
                      'every construction. `remaining` already states the call, tool and evaluation margins; leave room for the commit. C_A is limited support evidence and '
                      'a historical Skill is a correctable prior; neither guarantees future rankings.)')
FAST_SYSTEM = ES.FAST_SYSTEM + '\n' + MATERIAL_SEMANTICS


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'configs': root / 'stage_configs', 'logs': root / 'logs', 'split': root / 'split_copy.json',
            'parent_split': root / 'parent_split_copy.json', 'wiring': root / 'wiring', 'smoke': root / 'smoke', 'evidence': root / 'evidence', 'slow': root / 'slow',
            'select': root / 'select', 'freeze': root / 'freeze', 'metatest': root / 'metatest', 'warnings': root / 'warnings', 'lock': root / 'package.lock'}


def ref(stage: str, branch: str, tail: str) -> str:
    return '%s/%s/%s' % (stage, branch, tail)


def write_new(path, obj) -> None:
    """Exclusive create: a numbered receipt / response is never overwritten (task §7.2 item 4)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=context._default)


# ============================================================================= split: verbatim copy, path binding, disjointness against the parent
def resolve_csv_path(dataset: str, given) -> dict:
    """The document names the files by a /mnt/c/... (WSL) path; the numerical environment is the Windows Python of the parent package.
    Binding rule: the given path (or its drive translation) must resolve to the SAME file as spec.DATASETS[dataset]['path']; the cases then
    use csv_path=None (the spec path), exactly like the parent's cases."""
    spec_path = Path(spec.DATASETS[dataset]['path'])
    candidates = []
    if given:
        candidates.append(str(given))
        m = re.match(r'^/mnt/([a-zA-Z])/(.*)$', str(given))
        if m:
            candidates.append('%s:/%s' % (m.group(1).upper(), m.group(2)))
    resolved = next((c for c in candidates if Path(c).exists()), None)
    same = resolved is not None and Path(resolved).resolve() == spec_path.resolve()
    if not same:
        raise RuntimeError('csv path binding failed for %s: given %r, resolved %r, spec %r' % (dataset, given, resolved, str(spec_path)))
    return {'given': given, 'resolved': resolved, 'spec_path': str(spec_path), 'identical_file': True, 'name': spec_path.name, 'bound_as': 'spec.DATASETS path (csv_path=None)'}


def load_split(root: Path) -> dict:
    p = paths(root)['split']
    return context.read_json(p if p.exists() else SPLIT_DOC)


def load_parent_split(root: Path) -> dict:
    p = paths(root)['parent_split']
    return context.read_json(p if p.exists() else PARENT_SPLIT_DOC)


def bound_split(split: dict) -> dict:
    """The split with csv_path/path replaced by the bound spec path (None) after the binding check; the verbatim copy stays on disk."""
    out = copy.deepcopy(split)
    for dom, D in out['domains'].items():
        resolve_csv_path(D['dataset'], D.get('csv_path') or D.get('path'))
        D['csv_path'] = None
        D.pop('path', None)
    return out


def cases_of(root: Path) -> dict:
    return ec.cases_from_split(bound_split(load_split(root)))


def parent_cases_of(root: Path) -> dict:
    return ec.cases_from_split(load_parent_split(root))


def case_ids(root: Path, role: str, domain: str | None = None) -> list:
    return [c for c, cs in cases_of(root).items() if cs.role == role and (domain is None or cs.domain == domain)]


def check_split_dp(split: dict, parent: dict) -> dict:
    """This package's partition check (never the parent's 8/2/2 rule): per domain 2 Select + 4 MetaTest groups of 16, case_index 12..17 unique,
    all columns distinct, disjoint from EVERY parent group of that domain, excluded columns unused, the same two anchors, the same dataset."""
    out = {'domains': {}, 'ok': True}
    for dom, D in split['domains'].items():
        P_ = parent['domains'].get(dom)
        groups = D['groups']
        allr = [c for g in groups for c in g['roster']]
        parent_cols = {c for g in (P_ or {}).get('groups', []) for c in g['roster']}
        roles = {}
        for g in groups:
            roles.setdefault(g['stage'], []).append(g['case_id'])
        excl = set(D.get('excluded_columns', []))
        rec = {'n_groups': len(groups), 'n_columns': len(allr), 'n_unique': len(set(allr)), 'sizes_ok': all(len(g['roster']) == ec.COHORT_SIZE for g in groups),
               'excluded_in_use': sorted(set(allr) & excl), 'roles': {k: len(v) for k, v in roles.items()},
               'case_index': sorted(g['case_index'] for g in groups), 'overlap_with_parent': sorted(set(allr) & parent_cols),
               'parent_columns': len(parent_cols), 'same_dataset_as_parent': bool(P_) and P_['dataset'] == D['dataset'],
               'anchors_ok': sorted(split['anchors']) == sorted(parent['anchors']) and all(g['t'] in split['anchors'] for g in groups),
               'geometry_ok': all(list(g['train_rows']) == [g['t'] - spec.TRAIN_SPAN, g['t']] and list(g['c_a_origins']) == [g['t'], g['t'] + 48] and
                                  list(g['c_b_origins']) == [g['t'] + 96, g['t'] + 144] and list(g['e_origins']) == [g['t'] + 192, g['t'] + 240, g['t'] + 288, g['t'] + 336] for g in groups)}
        rec['ok'] = (rec['n_unique'] == rec['n_columns'] and rec['sizes_ok'] and not rec['excluded_in_use'] and rec['roles'] == NEW_ROLES
                     and rec['case_index'] == list(NEW_CASE_INDEX) and not rec['overlap_with_parent'] and rec['same_dataset_as_parent'] and rec['anchors_ok'] and rec['geometry_ok'])
        out['domains'][dom] = rec
        out['ok'] = out['ok'] and rec['ok']
    out['ok'] = out['ok'] and set(split['domains']) == set(parent['domains'])
    return out


# ============================================================================= package lock (OS exclusive lock; one controller per package directory)
class PackageLock:
    """One byte-range lock on <root>/package.lock, held by the controller process for its whole life (task §7.2 item 3). A second driver /
    resume / stage worker fails immediately; the monitor never takes it. The record inside the file is informational only."""

    def __init__(self, root: Path, role: str):
        self.path, self.role, self.fh = paths(Path(root))['lock'], role, None

    def acquire(self) -> 'PackageLock':
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise RuntimeError('another controller holds the package lock %s' % self.path) from None
        self.fh = fh
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps({'pid': os.getpid(), 'role': self.role, 'local': now()}).encode('utf-8'))
            fh.flush()
        except OSError:
            pass
        return self

    def release(self) -> None:
        if self.fh is None:
            return
        try:
            if os.name == 'nt':
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.fh.close()
            self.fh = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


def lock_probe(root: Path) -> int:
    """Subprocess entry (smoke): 0 if the lock could be taken, 3 if another controller holds it."""
    try:
        PackageLock(root, 'probe').acquire().release()
        return 0
    except RuntimeError:
        return 3


# ============================================================================= thread-safe ledger and pools
class SafeLedger(rt.RuntimeLedger):
    """The package ledger with one re-entrant lock around every mutation / save / check; HTTP waits and fits run outside the lock."""

    def __init__(self, path, **caps):
        self.lock = threading.RLock()
        super().__init__(path, **caps)

    def _save(self):
        with self.lock:
            super()._save()

    def reserve_fit(self, cid):
        with self.lock:
            super().reserve_fit(cid)

    def finish_fit(self, cid, ok, seconds, reason):
        with self.lock:
            super().finish_fit(cid, ok, seconds, reason)

    def note_cache(self, cid):
        with self.lock:
            super().note_cache(cid)

    def check_fit(self):
        with self.lock:
            super().check_fit()

    def check_llm(self):
        with self.lock:
            super().check_llm()

    def check_wall(self):
        with self.lock:
            super().check_wall()

    def remaining(self):
        with self.lock:
            return super().remaining()

    def can_retry(self):
        with self.lock:
            return super().can_retry()

    def add(self, key: str, value: float) -> None:
        with self.lock:
            self.s[key] = self.s.get(key, 0.0) + value
            self._save()

    def event(self, **row) -> None:
        with self.lock:
            self.s['events'].append({**row, 'epoch': time.time()})
            self._save()


class Pools:
    """One numeric pool (material builds, fits, labels) and one HTTP pool per controller; peak / queue-wait accounting for the report."""

    def __init__(self, numeric: int, http: int):
        self.numeric_sem, self.http_sem = threading.BoundedSemaphore(int(numeric)), threading.BoundedSemaphore(int(http))
        self.numeric_limit, self.http_limit = int(numeric), int(http)
        self.lock = threading.Lock()
        self.numeric_active = self.http_active = 0
        self.numeric_peak = self.http_peak = 0
        self.numeric_waits, self.http_waits = [], []

    def _enter(self, kind):
        with self.lock:
            n = getattr(self, kind + '_active') + 1
            setattr(self, kind + '_active', n)
            setattr(self, kind + '_peak', max(getattr(self, kind + '_peak'), n))

    def _leave(self, kind):
        with self.lock:
            setattr(self, kind + '_active', getattr(self, kind + '_active') - 1)

    class _Slot:
        def __init__(self, pools, kind):
            self.pools, self.kind = pools, kind

        def __enter__(self):
            t0 = time.time()
            getattr(self.pools, self.kind + '_sem').acquire()
            with self.pools.lock:
                getattr(self.pools, self.kind + '_waits').append(time.time() - t0)
            self.pools._enter(self.kind)
            return self

        def __exit__(self, *exc):
            self.pools._leave(self.kind)
            getattr(self.pools, self.kind + '_sem').release()

    def numeric(self):
        return Pools._Slot(self, 'numeric')

    def http(self):
        return Pools._Slot(self, 'http')

    def snapshot(self) -> dict:
        with self.lock:
            return {'numeric_limit': self.numeric_limit, 'numeric_active': self.numeric_active, 'numeric_peak': self.numeric_peak,
                    'numeric_waits': len(self.numeric_waits), 'numeric_wait_mean_s': statistics.fmean(self.numeric_waits) if self.numeric_waits else 0.0,
                    'numeric_wait_max_s': max(self.numeric_waits) if self.numeric_waits else 0.0,
                    'http_limit': self.http_limit, 'http_active': self.http_active, 'http_peak': self.http_peak,
                    'http_waits': len(self.http_waits), 'http_wait_mean_s': statistics.fmean(self.http_waits) if self.http_waits else 0.0,
                    'http_wait_max_s': max(self.http_waits) if self.http_waits else 0.0}


POOLS: Pools | None = None      # one per controller process; created once before any concurrent work (never switched per branch)


def pools() -> Pools:
    global POOLS
    if POOLS is None:
        POOLS = Pools(CONCURRENCY['numeric'], CONCURRENCY['http'])
    return POOLS


def set_pools(numeric: int, http: int) -> Pools:
    global POOLS
    if POOLS is not None and (POOLS.numeric_active or POOLS.http_active):
        raise RuntimeError('pools are in use; concurrency is set once before the first concurrent task')
    POOLS = Pools(numeric, http)
    return POOLS


# ============================================================================= controller-side subprocess plumbing (never imports torch)
def worker_env() -> dict:
    return ES.worker_env()


def run_worker(args: list, log: Path, timeout: float) -> tuple:
    """One parent-study worker subprocess inside the numeric pool; (returncode, seconds)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with pools().numeric():
        t0 = time.time()
        try:
            with log.open('w', encoding='utf-8') as fh:
                p = subprocess.run([sys.executable, '-B', '-m', WORKER_MODULE] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=timeout, env=worker_env())
            return p.returncode, time.time() - t0
        except subprocess.TimeoutExpired:
            return -999, time.time() - t0


def proxy_reachable() -> bool:
    return ES.proxy_reachable()


def fit_cells(ledger: SafeLedger, root: Path, split_path: Path, case: str, jobs: list, *, tag: str = '', n_updates=None) -> dict:
    """jobs: [(phys, seed)] of ONE case. Cached OK cells come back with fit_started=False; the others are reserved-before-launch physical
    fits, one worker subprocess each inside the numeric pool (a case's three seeds may run side by side when slots are free), each with at most
    one same-configuration retry drawn from the package retry cap. Returns {cell_id: {'record', 'fit_started', 'attempts'}}; the fit receipt
    (not a global counter difference) tells the caller whether a fit happened (task §7.2 item 2)."""
    out, pending = {}, []
    for phys, seed in jobs:
        cid = ec.cell_id(case, phys, seed, tag)
        cp = Path(root) / case / 'cells' / (cid + '.json')
        if cp.exists() and context.read_json(cp).get('status') == 'OK':
            out[cid] = {'record': context.read_json(cp), 'fit_started': False, 'attempts': 0}
            continue
        pending.append((phys, seed, cid, cp))
    errors = []

    def run_one(phys, seed, cid, cp):
        try:
            for attempt in (0, 1):
                log = Path(root) / case / 'fit_logs' / (cid + ('.log' if attempt == 0 else '.retry.log'))
                log.parent.mkdir(parents=True, exist_ok=True)
                args = ['--worker-fit', root, split_path, case, phys, seed] + (['--tag', tag] if tag else []) + (['--n-updates', n_updates] if n_updates else [])
                with pools().numeric():
                    ledger.reserve_fit(cid)
                    timeout = min(FIT_TIMEOUT_S, ledger.remaining())
                    t0 = time.time()
                    try:
                        with log.open('w', encoding='utf-8') as fh:
                            p = subprocess.run([sys.executable, '-B', '-m', WORKER_MODULE] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                                               timeout=timeout, env=worker_env())
                        rc = p.returncode
                    except subprocess.TimeoutExpired:
                        rc = -999
                    ok = rc == 0 and cp.exists() and context.read_json(cp).get('status') == 'OK'
                    reason = '' if ok else ('worker_timeout' if rc == -999 else 'worker_failed')
                    ledger.finish_fit(cid, ok, time.time() - t0, reason)
                print('FIT', cid, 'OK' if ok else reason, round(time.time() - t0, 1), flush=True)
                if ok:
                    rec = context.read_json(cp)
                    if rec['scores']['c_a']['n_nonfinite_predictions']:
                        raise RuntimeError('nonfinite predictions')
                    out[cid] = {'record': rec, 'fit_started': True, 'attempts': attempt + 1}
                    return
                if attempt == 0 and rt.FIT_RETRY and ledger.can_retry():
                    with ledger.lock:
                        ledger.s['retries_used'] += 1
                        ledger.s['events'].append({'kind': 'fit_retry', 'cell': cid, 'reason': reason, 'epoch': time.time()})
                        ledger._save()
                    continue
                raise RuntimeError(reason)
        except Exception as exc:  # noqa: BLE001
            errors.append((cid, exc))

    threads = [threading.Thread(target=run_one, args=j, name='fit:' + j[2], daemon=True) for j in pending]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError('fit failed: %s (%s)' % (errors[0][0], errors[0][1]))
    return out


def ensure_material(root: Path, split_path: Path, case: str, assignment: list, *, ledger: SafeLedger | None = None) -> dict:
    """The physical record of an assignment in the case cache: None / existing by key / built now in one worker (serial per case; the
    case's thread is the unique registrar of its TA numbers)."""
    reg = ec.load_registry(Path(root) / case)
    key = ec.material_key(assignment)
    for r in reg.values():
        if r['key'] == key:
            return r
    if ec.is_identity(assignment):
        return reg['None']
    used = [int(r['material_index']) for r in reg.values() if r.get('material_index')]
    phys = 'TA%03d' % (max(used + [0]) + 1)
    context.write_json(ec.aug_dir(Path(root) / case) / (phys + '__request.json'), {'assignment': assignment, 'requested_local': now(), 'package': PACKAGE})
    t0 = time.time()
    rc, secs = run_worker(['--worker-material', root, split_path, case, phys], Path(root) / case / 'fit_logs' / ('material_%s.log' % phys), 900)
    if ledger is not None:
        ledger.add('material_wall_seconds', time.time() - t0)
    err = ec.aug_dir(Path(root) / case) / (phys + '__error.json')
    if rc != 0:
        if err.exists():
            raise br.ToolInputError('plan rejected at execution (no fit, no material): %s' % context.read_json(err)['message'], execution_status='PROGRAM_EXECUTION_REJECTED')
        raise RuntimeError('material worker failed for %s %s' % (case, phys))
    return ec.load_registry(Path(root) / case)[phys]


# ============================================================================= metered client with per-request receipts (one HTTP pool)
class MeteredClient:
    """Task §7.2 items 1, 2, 4: request numbers, HTTP attempts and token totals change only inside the ledger lock; the HTTP wait happens
    outside it inside the HTTP pool; every request carries stage / case / arm / call / attempt in its receipt and its own usage comes back
    with the parsed response (never a global counter difference). Failed attempts follow the frozen rule (unknown usage, no automatic
    re-dispatch beyond the one bounded transport retry); an account fault or a returned-model mismatch is fatal."""

    def __init__(self, ledger: SafeLedger, out, *, http_cap: int, stage: str):
        import openai
        key = next((os.environ.get(n, '').strip() for n in ('CPA_API_KEY', 'OPENAI_API_KEY') if os.environ.get(n, '').strip()), '')
        if not key:
            raise rt.llm.AccountFault('API key unavailable')
        self.api = openai.OpenAI(api_key=key, base_url=MODEL['base_url'], max_retries=0, timeout=300)
        self.ledger, self.out, self.stage = ledger, Path(out), stage
        self.out.mkdir(parents=True, exist_ok=True)
        self.fatal = False
        self.http_cap = int(http_cap)
        self.unit_tokens = {}

    def _transport(self, messages, max_tokens, timeout):
        return self.api.chat.completions.create(model=MODEL['requested'], messages=messages, temperature=MODEL['temperature'], max_tokens=max_tokens, timeout=timeout)

    def call(self, role, unit, payload, system, *, max_tokens=MAX_OUTPUT_TOKENS, meta=None, request_timeout=300.):
        led, meta = self.ledger, dict(meta or {})
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        upper = len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 2048 + max_tokens
        with led.lock:
            if self.fatal:
                raise rt.llm.AccountFault('backend identity/authentication previously failed')
            led.check_llm()
            if rt.unknown_usage_blocks(led):
                raise budget.BudgetExhausted('unknown usage prevents further budgeted calls')
            used = led.s['llm_tokens_in'] + led.s['llm_tokens_out'] + led.s.get('token_reserved_failed_upper', 0)
            if used + 2 * upper > led.s['caps']['max_llm_tokens']:
                raise budget.BudgetExhausted('insufficient conservative remaining token budget')
            led.s['llm_requests'] += 1
            number = led.s['llm_requests']
            led.s['events'].append({'kind': 'llm_request', 'request': number, 'role': role, 'unit': unit, 'stage': self.stage, **meta, 'reserved_upper': upper, 'epoch': time.time()})
            led._save()
            prefix = self.out / ('%03d_%s' % (number, role))
            write_new(str(prefix) + '_request.json', {'role': role, 'unit': unit, 'stage': self.stage, **meta, 'request': number, 'messages': messages,
                                                     'requested_model': MODEL['requested'], 'temperature': MODEL['temperature'], 'max_tokens': max_tokens, 'sent_local': now()})
        t_q = time.time()
        with pools().http():
            wait = time.time() - t_q
            for attempt in range(2):
                with led.lock:
                    led.check_wall()
                    if led.s['llm_http_attempts'] >= self.http_cap:
                        raise budget.BudgetExhausted('HTTP cap')
                    led.s['llm_http_attempts'] += 1
                    led._save()
                try:
                    response = self._transport(messages, max_tokens, min(float(request_timeout), led.remaining()))
                except Exception as exc:  # noqa: BLE001
                    kind = rt.llm.classify_fault(exc)
                    write_new(str(prefix) + '_attempt%d.json' % attempt, {'failure_kind': kind, 'exception_type': type(exc).__name__, 'request': number, 'stage': self.stage, **meta, 'epoch': time.time()})
                    with led.lock:
                        led.s['llm_tokens_unknown'] += 1
                        led.s.setdefault('llm_failed_attempts', 0)
                        led.s['llm_failed_attempts'] += 1
                        led.s.setdefault('token_reserved_failed_upper', 0)
                        led.s['token_reserved_failed_upper'] += upper
                        led.s['events'].append({'kind': 'llm_attempt_failed', 'request': number, 'attempt': attempt, 'failure_kind': kind, 'stage': self.stage, **meta, 'epoch': time.time()})
                        led._save()
                        if kind == 'ACCOUNT_OR_PERMISSION_FAULT':
                            self.fatal = True
                    if kind == 'ACCOUNT_OR_PERMISSION_FAULT':
                        raise rt.llm.AccountFault(kind) from None
                    if kind == 'TRANSPORT_TRANSIENT_FAULT' and attempt == 0:
                        continue
                    raise rt.llm.TransportFault(kind) from None
                write_new(str(prefix) + '_response.json', response.model_dump(mode='json'))
                usage = response.usage
                pt, ct = getattr(usage, 'prompt_tokens', None), getattr(usage, 'completion_tokens', None)
                with led.lock:
                    if pt is None or ct is None:
                        led.s['llm_tokens_unknown'] += 1
                    else:
                        led.s['llm_tokens_in'] += pt
                        led.s['llm_tokens_out'] += ct
                        ut = self.unit_tokens.setdefault(unit, {'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0})
                        ut['requests'] += 1
                        ut['prompt_tokens'] += pt
                        ut['completion_tokens'] += ct
                    led.s['events'].append({'kind': 'llm_finished', 'role': role, 'unit': unit, 'request': number, 'stage': self.stage, **meta, 'attempts': attempt + 1,
                                            'prompt_tokens': pt, 'completion_tokens': ct, 'queue_wait_s': round(wait, 3), 'returned_model': response.model, 'epoch': time.time()})
                    if response.model != MODEL['returned_required']:
                        self.fatal = True
                    led._save()
                if response.model != MODEL['returned_required']:
                    raise rt.llm.AccountFault('returned model mismatch')
                text = response.choices[0].message.content if response.choices else ''
                print('LLM', role, unit, 'request', number, 'usage', pt, ct, 'wait %.1fs' % wait, flush=True)
                if not isinstance(text, str):
                    raise ValueError('no response text')
                receipt = {'request': number, 'prompt_tokens': pt, 'completion_tokens': ct, 'attempts': attempt + 1, 'queue_wait_s': wait}
                return json.loads(text), receipt
        raise AssertionError('unreachable')


class FastClient:
    """The Fast-side view of the metered client for one stage: per-trajectory cumulative token monitoring (warnings, no cap), the free proxy
    pre-check, call numbering per unit; the client contract of br.run_job (payload -> parsed JSON)."""

    def __init__(self, client: MeteredClient, system: str = FAST_SYSTEM):
        self.client, self.system, self.spent, self.calls, self.lock = client, system, {}, {}, threading.Lock()
        self.warned = {}

    def fast(self, unit, *, case=None, arm=None):
        def call(payload):
            if not proxy_reachable():          # free TCP pre-check: no request number, no HTTP attempt, no unknown usage; the never-sent call is resumable
                raise rt.llm.TransportFault('PROXY_UNREACHABLE_PRECHECK')
            with self.lock:
                self.calls[unit] = self.calls.get(unit, 0) + 1
                n = self.calls[unit]
            out, rec = self.client.call('fast', unit, payload, self.system, max_tokens=MAX_OUTPUT_TOKENS, meta={'case': case, 'arm': arm, 'call': n})
            with self.lock:
                self.spent[unit] = self.spent.get(unit, 0) + (rec['prompt_tokens'] or 0) + (rec['completion_tokens'] or 0)
                for th in FAST_TOKEN_MONITOR:
                    if self.spent[unit] >= th and th not in self.warned.setdefault(unit, set()):
                        self.warned[unit].add(th)
                        print('FAST_TOKEN_MONITOR', unit, 'cumulative', self.spent[unit], '>=', th, '(monitoring only)', flush=True)
                        self.client.ledger.event(kind='fast_token_monitor', unit=unit, cumulative=self.spent[unit], threshold=th)
            return out
        return call


# ============================================================================= adapter (parent tool semantics; receipts instead of counter differences)
class CaseAdapter(ES.CaseAdapter):
    def __init__(self, run_dir, case, ledger, repo, *, common, split_path, cs, public_ids, tool_error_feedback=True, seeds=SEEDS, fit_fn=None):
        fit_fn = fit_fn or (lambda phys, seeds_: fit_cells(ledger, Path(common), Path(split_path), case, [(phys, s) for s in seeds_]))
        super().__init__(run_dir, case, ledger, repo, common=common, split_path=split_path, cs=cs, public_ids=public_ids, tool_error_feedback=tool_error_feedback, seeds=seeds, fit_fn=fit_fn)

    def build_material(self, arguments, *, remaining_seconds):
        # identical to the parent except that the material worker runs inside the numeric pool (module-level ensure_material of this study)
        if set(arguments) != {'plan_id', 'policy'}:
            self._reject('build needs plan_id and policy')
        mid = arguments['plan_id']
        if not isinstance(mid, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}', mid) or mid in PUBLIC or mid.startswith('TA'):
            self._reject('unsafe or reserved plan id')
        reg = self._reg()
        if mid in reg:
            self._reject('plan id already exists; use it or a new id')
        try:
            compiled = ec.compile_plan_full(arguments['policy'], self.ctx.overview()['entities'])
        except policy.PolicyError as exc:
            self._reject(str(exc), legal_program_forms=['[] (no augmentation)', '1-3 of %s' % list(ta.PRIMITIVES), '[{"op":"%s"}]' % ta.RECIPE,
                                                        '[{"op":"%s","disabled_ops":[0-2 names]}]' % ta.RECIPE_EDIT], valid_observation_fields=list(spec.OBS_FIELDS))
        key = ec.material_key(compiled['assignment'])
        same = next((m for m, r in reg.items() if r.get('key') == key), None)
        if same is not None:
            self._reject('identical complete assignment to existing plan %s (same material): reuse that id; no new material, fit or slot' % same, alias_of=same,
                         execution_status='DUPLICATE_ASSIGNMENT')
        phys = ensure_material(self.common, self.split_path, self.case, compiled['assignment'], ledger=self.ledger)
        rec = {**phys, 'material_id': mid, 'phys_id': phys['material_id'], 'compiled': compiled}
        reg = self._reg()
        reg[mid] = rec
        ec.save_registry(self.ctx.job_dir, reg)
        programs, idx = [], []
        for steps in compiled['assignment']:
            if steps not in programs:
                programs.append(steps)
            idx.append(programs.index(steps))
        ms = {'profile': ec.PROFILE_VERSION, 'policy': compiled['policy'], 'programs': programs, 'entity_program': idx, 'rule_index': compiled['rule_index'],
              'resolved_thresholds': compiled['resolved_thresholds'], 'n_unknown': compiled['n_unknown'], 'alias_of': None, 'label': ec.assignment_label(compiled['assignment']),
              'execution_order': 'fixed: ' + ' -> '.join(ta.PRIMITIVES)}
        self.specs[mid] = ms
        context.write_json(ec.aug_dir(self.ctx.job_dir) / (mid + '__compiled.json'), {'material_spec': ms, 'assignment': compiled['assignment'], 'key': key, 'phys_id': phys['material_id']})
        return br.Candidate(mid, ms, self.case)

    def evaluate(self, candidate, seeds, *, feedback, remaining_seconds):
        rec = self._reg()[candidate.plan_id]
        phys = rec.get('phys_id', rec['material_id'])
        recs, missing = {}, []
        for seed in seeds:
            bc = self.ctx.job_dir / 'cells' / (ec.cell_id(self.case, candidate.plan_id, seed) + '.json')
            if bc.exists():
                recs[seed] = context.read_json(bc)
            else:
                missing.append(seed)
        if missing:
            got = self.fit_fn(phys, missing)
            for seed in missing:
                r = got[ec.cell_id(self.case, phys, seed)]
                pr = r['record']
                if not r['fit_started']:
                    self.ledger.note_cache(ec.cell_id(self.case, candidate.plan_id, seed))
                if pr['job_id'] != self.case or pr['model_seed'] != seed or pr.get('material_key') != rec['key'] or not Path(pr['model_path']).exists():
                    raise RuntimeError('physical cell binding failed')
                mine = {**pr, 'cell_id': ec.cell_id(self.case, candidate.plan_id, seed), 'material_id': candidate.plan_id, 'physical_cell': pr['cell_id'], 'physical_material': phys,
                        'cache_hit': not r['fit_started'], 'fit_attempts_this_request': r['attempts']}
                bc = self.ctx.job_dir / 'cells' / (mine['cell_id'] + '.json')
                bc.parent.mkdir(exist_ok=True)
                context.write_json(bc, mine)
                recs[seed] = mine
        ordered = [recs[s] for s in seeds]
        if any(r['scores']['c_a']['status'] != 'SCORABLE' for r in ordered):
            raise RuntimeError('C_A_NOT_SCORABLE')
        losses = tuple(tuple(tuple(row) for row in r['scores']['c_a']['per_origin_entity_normalized_mse']) for r in ordered) if feedback else ()
        fitted = replace(candidate, model_seeds=tuple(seeds), model_refs=tuple(r['model_path'] for r in ordered), ca_losses=losses)
        self.fitted[candidate.plan_id] = fitted
        return fitted

    def restore(self, plan_ids):
        out = []
        for mid in plan_ids:
            started = []
            orig = self.fit_fn

            def guarded(phys, seeds_):
                got = orig(phys, seeds_)
                started.extend(c for c, r in got.items() if r['fit_started'])
                return got
            self.fit_fn = guarded
            try:
                out.append(self.evaluate(br.Candidate(mid, self._spec_of(mid), self.case), self.seeds, feedback=True, remaining_seconds=self.ledger.remaining()))
            finally:
                self.fit_fn = orig
            if started:
                raise RuntimeError('restore must not fit: %s' % started)
        return out


def adapter_factory(common: Path, split_path: Path, cs: ec.CaseSpec, public_ids, seeds):
    def make(root, case, led, repo, *, tool_error_feedback=True, roster=None, seeds=seeds, dataset=None):
        return CaseAdapter(root, case, led, repo, common=common, split_path=split_path, cs=cs, public_ids=public_ids, tool_error_feedback=tool_error_feedback, seeds=seeds)
    return make


def fast_branch(path, case, knowledge, led, client_call, common, *, split_path, cs, public_ids, seeds):
    """client_call: an object with .fast(unit) -> callable (FastClient bound to case / arm, or a scripted client)."""
    return base.branch(path, case, knowledge, led, client_call, None, evidence_roundtrip=EVIDENCE_ROUNDTRIP, max_tool_corrections=MAX_TOOL_CORRECTIONS,
                       tool_contracts=CONTRACTS, seeds=seeds, adapter_factory=adapter_factory(common, split_path, cs, public_ids, seeds), limits=LIMITS, dataset=cs.dataset,
                       commit_fn=ES.commit_branch, allowed_features=ALLOWED_FEATURES, entity_count=ec.COHORT_SIZE, request_trace_dedupe=REQUEST_TRACE_DEDUPE)


def resume_fast_branch(path, case, knowledge, led, client_call, common, *, split_path, cs, public_ids, seeds):
    return base.resume_branch(path, case, knowledge, led, client_call, max_tool_corrections=MAX_TOOL_CORRECTIONS, tool_contracts=CONTRACTS, seeds=seeds,
                              adapter_factory=adapter_factory(common, split_path, cs, public_ids, seeds), evidence_roundtrip=EVIDENCE_ROUNDTRIP, limits=LIMITS,
                              dataset=cs.dataset, commit_fn=ES.commit_branch, allowed_features=ALLOWED_FEATURES, entity_count=ec.COHORT_SIZE, request_trace_dedupe=REQUEST_TRACE_DEDUPE)


class BoundFast:
    """A FastClient bound to one (case, arm) so that base.branch's client.fast(unit) carries the receipt metadata."""

    def __init__(self, fc: FastClient, case: str, arm: str):
        self.fc, self.case, self.arm = fc, case, arm
        self.ledger = fc.client.ledger

    def fast(self, unit):
        return self.fc.fast(unit, case=self.case, arm=self.arm)


def random_branch(root: Path, case: str, led, common: Path, supply: dict, *, split_path, cs, public_ids, seeds) -> br.RunResult:
    """RandomSearch_B4 of the parent (0 LLM, four frozen plans, argmin three-seed C_A) on this study's adapter."""
    root.mkdir(parents=True, exist_ok=False)
    ad = CaseAdapter(root, case, led, REPO, common=common, split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
    n, trace = ad.n, []
    emit = ES._emit_factory(root, case, trace)
    context.write_json(root / 'knowledge.json', asdict(br.Knowledge()))
    cands = list(ad.baselines())
    emit('job_started', mode='random_search_b4', baseline_ids=[c.plan_id for c in cands], seeds=list(seeds), supply_rng=supply['rng'])
    for pl in supply['plans']:
        c = ad.build_material({'plan_id': pl['slot'], 'policy': pl['policy']}, remaining_seconds=led.remaining())
        emit('random_material', plan_id=c.plan_id, material_spec=c.material_spec, draw={k: v for k, v in pl.items() if k != 'policy'})
        c = ad.evaluate(c, seeds, feedback=True, remaining_seconds=led.remaining())
        cands.append(c)
        emit('random_evaluated', plan_id=c.plan_id, feedback=c.feedback(seeds, 2, n))
    order = [c.plan_id for c in cands]
    chosen = min(cands, key=lambda c: (c.feedback(seeds, 2, n)['mean_loss'], order.index(c.plan_id)))
    reason = 'RandomSearch_B4: minimum three-seed C_A mean among %s; ties in that order.' % ', '.join(order)
    ES.commit_branch(root, cs.dataset, case, chosen.plan_id, reason, seeds[0])
    emit('committed', plan_id=chosen.plan_id, delivery_model_ref=chosen.model_refs[0], reason=reason)
    result = br.RunResult('COMPLETE', case, 0, chosen.plan_id, chosen.model_refs[0], 0, 0, len(supply['plans']), trace)
    context.write_json(root / 'branch_result.json', asdict(result))
    print('RANDOM', case, 'commit', chosen.plan_id, flush=True)
    return result


def fixed_dev_branch(root: Path, case: str, led, common: Path, program: dict, *, split_path, cs, public_ids, seeds) -> br.RunResult:
    """Fixed_dev of the parent (the domain's frozen uniform program; alias of a public reference when the assignment coincides)."""
    root.mkdir(parents=True, exist_ok=False)
    ad = CaseAdapter(root, case, led, REPO, common=common, split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
    trace = []
    emit = ES._emit_factory(root, case, trace)
    context.write_json(root / 'knowledge.json', asdict(br.Knowledge()))
    cands = {c.plan_id: c for c in ad.baselines()}
    emit('job_started', mode='fixed_dev', baseline_ids=list(cands), seeds=list(seeds), frozen_program=program)
    steps = program['steps']
    key = ec.material_key([steps] * ec.COHORT_SIZE) if steps is not None else None
    alias = next((m for m in public_ids if m != 'FixedMixup' and ec.material_key(ec.compile_plan_full(ec.uniform_policy(PUBLIC_STEPS[m], 'p'), ad.ctx.overview()['entities'])['assignment']) == key), None)
    if program.get('public_id') == 'FixedMixup':
        alias = 'FixedMixup'
    if alias is not None:
        chosen = cands[alias]
        emit('fixed_dev_alias', plan_id=alias, note='frozen program coincides with a public reference; cached fits reused')
    else:
        c = ad.build_material({'plan_id': 'FixedDev', 'policy': ec.uniform_policy(steps, 'Fixed_dev: the domain development-selected uniform program (no search)')}, remaining_seconds=led.remaining())
        emit('fixed_dev_material', plan_id=c.plan_id, material_spec=c.material_spec)
        chosen = ad.evaluate(c, seeds, feedback=True, remaining_seconds=led.remaining())
        emit('fixed_dev_evaluated', plan_id=chosen.plan_id, feedback=chosen.feedback(seeds, 2, ad.n))
    reason = 'Fixed_dev: the frozen uniform program %s selected on the domain Select cases; no search on this case.' % program['label']
    ES.commit_branch(root, cs.dataset, case, chosen.plan_id, reason, seeds[0], extra={'fixed_dev': program, 'alias_of_public': alias})
    emit('committed', plan_id=chosen.plan_id, delivery_model_ref=chosen.model_refs[0], reason=reason)
    result = br.RunResult('COMPLETE', case, 0, chosen.plan_id, chosen.model_refs[0], 0, 0, 0 if alias else 1, trace)
    context.write_json(root / 'branch_result.json', asdict(result))
    print('FIXED_DEV', case, 'commit', chosen.plan_id, flush=True)
    return result


# ============================================================================= knowledge carriers
def knowledge_from_json(d: dict) -> br.Knowledge:
    return ES.knowledge_from_json(d)


def old_card(domain: str) -> dict:
    """The parent's frozen card of the domain, read verbatim (freeze/<D>.json): the injected body is the Skill's rendered body."""
    fr = context.read_json(PARENT_ROOT / 'freeze' / ('%s.json' % domain))
    if fr.get('status') != 'FROZEN_SELECTED' or not fr.get('skill'):
        raise RuntimeError('parent freeze of %s is not FROZEN_SELECTED' % domain)
    s = dsk.skill_from_json(fr['skill'])
    if s.rendered_body != fr['injected_body'] or s.domain_id != domain:
        raise RuntimeError('parent injected body differs from the skill record of %s' % domain)
    return {'skill': s, 'injected_body': fr['injected_body'], 'skill_id': s.skill_id, 'parent_selection': fr.get('selection'), 'parent_fixed_dev': fr.get('fixed_dev')}


# ============================================================================= budget (one package ledger, cumulative stage caps)
def stage_caps(root: Path, stage: str) -> dict:
    P_ = paths(root)
    rec = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    if stage in rec:
        return rec[stage]
    led = SafeLedger(P_['ledger'], **TOTAL)
    s, alloc = led.s, ALLOC[stage]
    snap = {k: s.get(k, 0) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts',
                                     'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'llm_failed_attempts', 'fit_wall_seconds')}
    snap['elapsed_s'] = time.time() - s['started_epoch']
    retries_stage = (TOTAL['max_retries'] - s['retries_used']) if alloc['fits'] else 0
    caps = {'max_fit_attempts': min(TOTAL['max_fit_attempts'], s['fit_attempts'] + alloc['fits'] + retries_stage),
            'max_llm_requests': min(TOTAL['max_llm_requests'], s['llm_requests'] + alloc['requests']),
            'max_llm_tokens': TOTAL['max_llm_tokens'], 'max_wall_s': TOTAL['max_wall_s'], 'max_retries': TOTAL['max_retries']}
    rec[stage] = {'epoch_start': time.time(), 'local_start': now(), 'allocation': alloc, 'snapshot_at_start': snap, 'caps': caps,
                  'http_cap': min(HTTP_CAP, s['llm_http_attempts'] + alloc['requests'] + max(0, TRANSPORT_RETRIES - s.get('llm_failed_attempts', 0)))}
    context.write_json(P_['stages'], rec)
    return rec[stage]


def start_paid_clock(root: Path) -> None:
    P_ = paths(root)
    led = SafeLedger(P_['ledger'], **TOTAL)
    if led.s.get('paid_clock_started_epoch') is None:
        if led.s['llm_requests']:
            raise RuntimeError('paid requests exist before the paid clock start')
        led.s['wiring_elapsed_before_paid_clock_s'] = time.time() - led.s['started_epoch']
        led.s['paid_clock_started_epoch'] = time.time()
        led.s['paid_clock_started_local'] = now()
        led._save()


def budget_warnings(root: Path, where: str) -> list:
    """Task §8.1: tokens / paid wall are warnings, never a mechanical stop. Each threshold is written once."""
    P_ = paths(root)
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    tok = led.get('llm_tokens_in', 0) + led.get('llm_tokens_out', 0)
    paid = (time.time() - led['paid_clock_started_epoch']) if led.get('paid_clock_started_epoch') else 0.0
    P_['warnings'].mkdir(parents=True, exist_ok=True)
    out = []
    for th in TOKEN_WARNINGS:
        p = P_['warnings'] / ('tokens_%d.json' % th)
        if tok >= th and not p.exists():
            context.write_json(p, {'threshold': th, 'tokens': tok, 'where': where, 'local': now(), 'action': 'progress / remaining estimate only; no data, arm or request change'})
            out.append(str(p))
    for th in PAID_WALL_WARNINGS_S:
        p = P_['warnings'] / ('paid_wall_%dh.json' % (th // 3600))
        if paid >= th and not p.exists():
            context.write_json(p, {'threshold_s': th, 'paid_elapsed_s': paid, 'where': where, 'local': now(), 'action': 'progress / remaining estimate only'})
            out.append(str(p))
    for w in out:
        print('BUDGET_WARNING', w, flush=True)
    return out


def numeric_seconds(root: Path) -> float:
    led = context.read_json(paths(root)['ledger'])
    return float(led.get('fit_wall_seconds', 0.0)) + float(led.get('label_wall_seconds', 0.0)) + float(led.get('material_wall_seconds', 0.0))


# ============================================================================= stage coordinator (one process, one thread per active case, sequential arms inside a case)
class Progress:
    """Coordinator-written snapshot for the read-only monitor (never read by any arm)."""

    def __init__(self, path: Path):
        self.path, self.lock, self.state = Path(path), threading.Lock(), {'cases': {}, 'started_local': now()}

    def set(self, case: str, **kw) -> None:
        with self.lock:
            self.state['cases'].setdefault(case, {}).update(kw)
            self.state['updated_local'] = now()
            self.state['pools'] = pools().snapshot()
            context.write_json(self.path, self.state)

    def note(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)
            self.state['updated_local'] = now()
            self.state['pools'] = pools().snapshot()
            context.write_json(self.path, self.state)


def prepare_common(root: Path, case: str, led: SafeLedger, cfg: dict) -> Path:
    """Public references of one case: build (0 fits) -> random supply frozen (MetaTest) -> reference fits (numeric pool)."""
    common = root / (case + '_common')
    cases = ec.cases_from_split(context.read_json(cfg['split_path']))
    cs = cases[case]
    seeds, public_ids, split_path = tuple(cfg['seeds']), tuple(cfg['public_ids']), Path(cfg['split_path'])
    if not all(m in ec.load_registry(common / case) for m in public_ids):
        t0 = time.time()
        rc, _ = run_worker(['--worker-build', common, split_path, case], root / 'logs' / ('build_%s.log' % case), 1800)
        led.add('material_wall_seconds', time.time() - t0)
        if rc != 0:
            raise RuntimeError('common build failed for %s' % case)
    if cfg.get('random') and not (common / 'random_supply.json').exists():
        table = context.read_json(common / case / 'overview.json')['entities']
        dsks.write_once(common / 'random_supply.json', ES.draw_random_supply(case, cs.domain, cs.case_index, table))
    got = fit_cells(led, common, split_path, case, [(m, s) for m in public_ids for s in seeds])
    for r in got.values():
        if r['record']['scores']['c_a']['status'] != 'SCORABLE':
            raise RuntimeError('C_A_NOT_SCORABLE: %s' % r['record']['cell_id'])
    return common


def _safe_message(exc) -> str | None:
    return ES._safe_message(exc)


def _run_case(root: Path, cfg: dict, led: SafeLedger, fc, case: str, stop: threading.Event, progress: Progress, resumed: bool) -> tuple:
    """All arms of one case in the configured order; returns (branches, failures) of this case."""
    branches, failures = [], []
    cases = ec.cases_from_split(context.read_json(cfg['split_path']))
    cs = cases[case]
    seeds, public_ids, split_path = tuple(cfg['seeds']), tuple(cfg['public_ids']), Path(cfg['split_path'])
    progress.set(case, status='preparing_common', local=now())
    common = prepare_common(root, case, led, cfg)
    for arm in cfg['order'][case]:
        path = root / ('%s_%s' % (case, arm))
        alias = (cfg.get('aliases') or {}).get(case, {}).get(arm)
        if alias:
            p = root / ('%s_%s_alias.json' % (case, arm))
            if not p.exists():
                context.write_json(p, {'arm': arm, 'alias_of': alias, 'note': (cfg.get('alias_notes') or {}).get(arm, 'same frozen card text: the trajectory of the aliased arm is reused; no second LLM run')})
            continue
        kn = cfg['knowledge'].get(case, {}).get(arm)
        if arm not in CONTROL_ARMS and kn is None:
            raise RuntimeError('no knowledge for arm %s of %s' % (arm, case))
        if arm == 'fixed_dev' and not (cfg.get('fixed_dev') or {}).get(cs.domain):
            p = root / ('%s_%s_treatment.json' % (case, arm))
            if not p.exists():
                context.write_json(p, {'status': 'NO_TREATMENT', 'note': 'no Fixed_dev program frozen for this domain'})
            continue
        if stop.is_set():
            failures.append({'branch': path.name, 'kind': 'NOT_STARTED', 'reason': 'stage stopping (backend fatal / unknown usage); arm left for resume'})
            progress.set(case, status='stopped_before_' + arm, local=now())
            break
        branches.append((path, case))
        progress.set(case, status='running_' + arm, local=now())
        client_call = BoundFast(fc, case, arm) if fc is not None else None
        try:
            if path.exists() and (path / 'branch_result.json').exists():
                prior = context.read_json(path / 'branch_result.json')
                if prior['status'] == 'COMPLETE':
                    result = br.RunResult(**{k: v for k, v in prior.items() if k != 'trace'})
                elif arm in CONTROL_ARMS:
                    raise RuntimeError('control branch cannot be resumed')
                else:
                    result = resume_fast_branch(path, case, knowledge_from_json(kn), led, client_call, common, split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
            elif path.exists():
                raise RuntimeError('branch directory without a result; inspect before any replay')
            elif arm == 'random':
                result = random_branch(path, case, led, common, context.read_json(common / 'random_supply.json'), split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
            elif arm == 'fixed_dev':
                result = fixed_dev_branch(path, case, led, common, cfg['fixed_dev'][cs.domain], split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
            else:
                result = fast_branch(path, case, knowledge_from_json(kn), led, client_call, common, split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
            if result.status != 'COMPLETE':
                failures.append({'branch': path.name, 'kind': result.failure_kind, 'reason': result.reason})
        except Exception as exc:  # noqa: BLE001
            failures.append({'branch': path.name, 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
            print('BRANCH_FAILED', path.name, type(exc).__name__, _safe_message(exc), flush=True)
        progress.set(case, status='done_' + arm, last_arm_local=now())
        if fc is not None and (getattr(fc.client, 'fatal', False) or rt.unknown_usage_blocks(led)):
            stop.set()
            print('STAGE_STOP_REQUESTED', 'backend fatal' if fc.client.fatal else 'unknown usage', flush=True)
    progress.set(case, status='finished', local=now())
    return branches, failures


def _run_cases(root: Path, cfg: dict, led: SafeLedger, client, branches: list, failures: list, *, resumed: bool) -> None:
    """Task §7.1: up to CONCURRENCY['cases'] cases at once in the configured queue order; a free slot takes the next case; arms of a case
    stay sequential; results are merged in the frozen case / arm order (never by completion time)."""
    fc = FastClient(client) if client is not None else None
    progress = Progress(root / 'progress.json')
    stop = threading.Event()
    q = queue.Queue()
    for c in cfg['cases']:
        q.put(c)
    results, errors, lock = {}, [], threading.Lock()

    def worker():
        while True:
            try:
                case = q.get_nowait()
            except queue.Empty:
                return
            try:
                out = _run_case(root, cfg, led, fc, case, stop, progress, resumed)
            except Exception as exc:  # noqa: BLE001
                out = ([], [{'case': case, 'exception_type': type(exc).__name__, 'message': _safe_message(exc)}])
                errors.append((case, exc))
                print('CASE_FAILED', case, type(exc).__name__, _safe_message(exc), flush=True)
            with lock:
                results[case] = out
            q.task_done()
    n = min(int(cfg['concurrency']['cases']), len(cfg['cases']))
    threads = [threading.Thread(target=worker, name='case-worker-%d' % i, daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        budget_warnings(Path(cfg['package_root']), cfg['stage'])
        with led.lock:
            snap = {k: led.s.get(k) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown')}
        progress.note(ledger=snap)
    for t in threads:
        t.join()
    for case in cfg['cases']:
        b, f = results.get(case, ([], [{'case': case, 'kind': 'NOT_RUN'}]))
        branches.extend(b)
        failures.extend(f)
    progress.note(finished_local=now(), pools=pools().snapshot())
    if stop.is_set():
        raise RuntimeError('backend fatal or unknown usage')


def label_one(name: str, branch_dir: Path, case: str, led: SafeLedger, stage_root: Path, split_path: Path) -> None:
    t0 = time.time()
    rc, secs = run_worker(['--worker-label', name, branch_dir, split_path, case, stage_root], stage_root / 'logs' / ('label_%s_%s.log' % (name, branch_dir.name)), max(1.0, led.remaining()))
    led.add('label_wall_seconds', time.time() - t0)
    if rc:
        raise RuntimeError('label stage failed: %s %s' % (name, branch_dir.name))


def _parallel(jobs: list, fn) -> list:
    """Run fn(*job) for every job in threads (each job takes its own numeric slot inside); returns the list of exceptions (empty = all ok)."""
    errs = []

    def run(job):
        try:
            fn(*job)
        except Exception as exc:  # noqa: BLE001
            errs.append((job, exc))
    ts = [threading.Thread(target=run, args=(j,), daemon=True) for j in jobs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return errs


def open_labels(root: Path, cfg: dict, branches, failures, led: SafeLedger, *, withhold_reason=None, resumed=False) -> None:
    """C_B of every committed branch (parallel inside the phase) -> E predictions frozen (parallel) -> whole-stage barrier -> E scored (parallel).
    Any failure inside a phase is recorded and the later phases are not entered."""
    split_path = Path(cfg['split_path'])
    if withhold_reason:
        context.write_json(root / 'labels_withheld.json', {'epoch': time.time(), 'reason': withhold_reason, 'branches': [(str(p), j) for p, j in branches],
                                                          'unknown_usage': led.s['llm_tokens_unknown'], 'next': '--resume-stage (operator decision) or stop'})
        print('LABELS_WITHHELD', withhold_reason, flush=True)
        return
    eligible = [(p, j) for p, j in branches if (p / j / 'commit.json').exists()]
    try:
        errs = _parallel([(p, j) for p, j in eligible if not (p / j / 'c_b_scores.json').exists()], lambda p, j: label_one('c_b', p, j, led, root, split_path))
        if errs:
            raise RuntimeError('C_B failed: %s' % errs[0][0][0].name)
        if not cfg['labels_e']:
            context.write_json(root / 'labels_c_b_only.json', {'epoch': time.time(), 'branches': [str(p) for p, j in eligible], 'resumed': resumed})
        else:
            errs = _parallel([(p, j) for p, j in eligible if not (p / j / 'e_frozen.json').exists()], lambda p, j: label_one('freeze_e', p, j, led, root, split_path))
            if errs:
                raise RuntimeError('freeze_e failed: %s' % errs[0][0][0].name)
            context.write_json(root / 'all_e_predictions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in eligible], 'resumed': resumed})
            errs = _parallel([(p, j) for p, j in eligible if not (p / j / 'e_scores.json').exists()], lambda p, j: label_one('score_e', p, j, led, root, split_path))
            if errs:
                raise RuntimeError('score_e failed: %s' % errs[0][0][0].name)
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'external', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
        print('EXTERNAL_STOP', type(exc).__name__, flush=True)
    context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': resumed,
                                                         'pools': pools().snapshot()})


def stage_worker(cfg_path: Path, *, client=None) -> None:
    cfg = context.read_json(cfg_path)
    root = Path(cfg['output'])
    lock = PackageLock(Path(cfg['package_root']), 'stage_worker:%s' % cfg['stage']).acquire()
    try:
        if (root / 'experiment_started.json').exists():
            raise RuntimeError('stage already started; no paid replay')
        root.mkdir(parents=True, exist_ok=True)
        (root / 'logs').mkdir(exist_ok=True)
        context.write_json(root / 'config.json', cfg)
        set_pools(cfg['concurrency']['numeric'], cfg['concurrency']['http'])
        led = SafeLedger(cfg['ledger_path'], **cfg['caps'])
        if rt.unknown_usage_blocks(led):
            raise RuntimeError('package ledger holds unknown usage; paid stage refused without an operator decision')
        rt.FIT_RETRY = bool(cfg.get('fit_retry'))                       # set once, before any concurrent work
        context.write_json(root / 'experiment_started.json', {'epoch': time.time(), 'pid': os.getpid(), 'stage': cfg['stage'], 'concurrency': cfg['concurrency']})
        branches, failures, stopped = [], [], None
        try:
            if client is None and cfg.get('llm'):
                client = MeteredClient(led, root / 'raw_responses', http_cap=int(cfg['http_cap']), stage=cfg['stage'])     # config only; no unmetered ping
            _run_cases(root, cfg, led, client, branches, failures, resumed=False)
            context.write_json(root / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches]})
        except Exception as exc:  # noqa: BLE001
            failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
            stopped = type(exc).__name__
            print('EXECUTION_STOP', stopped, _safe_message(exc), flush=True)
        finally:
            context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures})
        open_labels(root, cfg, branches, failures, led, withhold_reason=stopped and 'execution stopped before every planned arm was attempted (%s)' % stopped)
    finally:
        lock.release()


def resume_stage(root: Path, stage: str, *, accept_unknown_usage: bool = False, restart=()) -> None:
    """One operator continuation under the package lock: labels still withheld (existing boundary check), completed branches kept, never-sent
    Fast calls continued, missing arms run as planned, then labels. Unknown usage needs an explicit acceptance."""
    from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp
    sroot = paths(root)[stage]
    cfg = context.read_json(sroot / 'config.json')
    with PackageLock(root, 'resume:%s' % stage):
        n_prev = len(list(sroot.glob('resume*.json')))
        if (sroot / 'experiment_started.json').exists() and not (sroot / 'execution_finished.json').exists() and not (sroot / 'labels_withheld.json').exists():
            context.write_json(sroot / 'labels_withheld.json', {'epoch': time.time(), 'reason': 'stage process stopped before finishing (operator stop / crash); no label was opened', 'written_by': 'resume_stage'})
        blocked = rtp.labels_boundary_violations(sroot)
        if blocked:
            raise RuntimeError('resume refused; labels are no longer withheld: ' + '; '.join(blocked))
        set_pools(cfg['concurrency']['numeric'], cfg['concurrency']['http'])
        led = SafeLedger(cfg['ledger_path'], **cfg['caps'])
        if rt.unknown_usage_blocks(led):
            if not accept_unknown_usage:
                raise RuntimeError('ledger holds unknown usage; pass --accept-unknown-usage to record an explicit operator decision')
            led.s['unknown_usage_accepted'] = led.s['llm_tokens_unknown']
            led.event(kind='unknown_usage_accepted', accepted=led.s['llm_tokens_unknown'], stage=stage)
        for name in ('execution_finished.json', 'labels_withheld.json'):
            if (sroot / name).exists():
                shutil.move(sroot / name, sroot / name.replace('.json', '_before_resume%d.json' % (n_prev + 1)))
        moved = []
        for name in restart:
            b = sroot / name
            prior = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
            if not b.is_dir() or (prior and (prior['status'] == 'COMPLETE' or prior['new_evaluations'])) or any(b.rglob('commit.json')):
                raise RuntimeError('only an interrupted branch without commit or new evaluation can be restarted: %s' % name)
            dest = sroot / ('%s__interrupted_%d' % (name, n_prev + 1))
            shutil.move(b, dest)
            moved.append({'branch': name, 'kept_as': dest.name, 'prior_status': prior and prior['status'], 'prior_failure_kind': prior and prior['failure_kind']})
        context.write_json(sroot / ('resume%d.json' % (n_prev + 1)), {'epoch': time.time(), 'local': now(), 'accepted_unknown_usage': bool(accept_unknown_usage), 'restarted_branches': moved})
        rt.FIT_RETRY = bool(cfg.get('fit_retry'))
        branches, failures, stopped = [], [], None
        try:
            client = MeteredClient(led, sroot / 'raw_responses', http_cap=int(cfg['http_cap']), stage=cfg['stage']) if cfg.get('llm') else None
            _run_cases(sroot, cfg, led, client, branches, failures, resumed=True)
            context.write_json(sroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches], 'resumed': True})
        except Exception as exc:  # noqa: BLE001
            failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
            stopped = type(exc).__name__
            print('EXECUTION_STOP', stopped, _safe_message(exc), flush=True)
        finally:
            context.write_json(sroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': True})
        open_labels(sroot, cfg, branches, failures, led, resumed=True, withhold_reason=stopped and 'resumed stage stopped again (%s)' % stopped)


def stage_config(root: Path, stage: str, *, cases, order, knowledge, labels_e: bool, llm: bool, random: bool, aliases=None, alias_notes=None, fixed_dev=None) -> Path:
    P_ = paths(root)
    path = P_['configs'] / ('%s.json' % stage)
    if path.exists():
        return path
    sc = stage_caps(root, stage)
    conc = dict(effective_concurrency(root))                                   # wiring's measured decision (concurrency_effective.json) or the planned values
    cfg = {'study': 'domain_aug_decision_priority', 'status': 'FROZEN', 'stage': stage, 'output': str(P_[stage]), 'package_root': str(root), 'cases': list(cases),
           'split_path': str(root / 'split_bound.json'), 'order': {c: list(order[c]) for c in cases}, 'knowledge': knowledge, 'labels_e': bool(labels_e), 'llm': bool(llm),
           'random': bool(random), 'seeds': list(SEEDS), 'public_ids': list(PUBLIC), 'limits': LIMITS, 'max_tool_corrections': MAX_TOOL_CORRECTIONS,
           'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'request_trace_dedupe': REQUEST_TRACE_DEDUPE, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'fit_threads': FIT_THREADS,
           'concurrency': conc, 'caps': sc['caps'], 'http_cap': sc['http_cap'], 'ledger_path': str(P_['ledger']), 'fit_retry': True,
           'fit_attempts_at_stage_start': context.read_json(P_['ledger'])['fit_attempts'], 'aliases': aliases or {}, 'alias_notes': alias_notes or {},
           'fixed_dev': fixed_dev or {}, 'fast_system': FAST_SYSTEM, 'frozen_local': now()}
    dsks.write_once(path, cfg)
    return path


def run_stage(root: Path, stage: str, cfg_path: Path, driver_lock: PackageLock) -> dict:
    """The driver hands the package lock to the stage worker (one controller at a time) and takes it back when the worker exits."""
    P_ = paths(root)
    sroot = P_[stage]
    if not (sroot / 'experiment_started.json').exists():
        P_['logs'].mkdir(parents=True, exist_ok=True)
        print('STAGE_START', stage, now(), flush=True)
        driver_lock.release()
        try:
            with (P_['logs'] / ('%s.log' % stage)).open('a', encoding='utf-8') as log:
                p = subprocess.Popen([sys.executable, '-B', '-m', MODULE, '--stage-worker', str(cfg_path)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=worker_env())
                context.write_json(root / ('%s_worker_launch.json' % stage), {'epoch': time.time(), 'pid': p.pid, 'driver_pid': os.getpid(), 'local': now()})
                rc = p.wait()
        finally:
            driver_lock.acquire()
        print('STAGE_EXIT', stage, rc, now(), flush=True)
    return dsks.stage_status(sroot)


# ============================================================================= evidence: the parent's 12 cases per domain as decision points (deterministic, 0 LLM)
GENERIC_TEXT = ES.GENERIC_TEXT
PARENT_STAGE_ARMS = {'source': ('f0',), 'select': ('cand_W1', 'cand_W2', 'cand_W3'), 'metatest': ('f0', 'f_domain', 'f_generic', 'random', 'fixed_dev')}
PARENT_CASE_STAGE = {'S': 'source', 'V': 'select', 'Q': 'metatest'}
HARM_WORDS = ('harm', 'worse', 'destruct', 'degrad', 'hurt', 'damag', 'distort', 'flatten', 'break', 'unsafe', 'risky', 'bad', 'reject', 'discard', 'avoid', 'skip')
CASE_TOKEN = re.compile(r'(?<![A-Za-z0-9])(?:D0\d(?:_[A-Za-z0-9]+)?|[SVQ]0?\d{1,2})(?![A-Za-z0-9])')
ROW_TOKEN = re.compile(r'(?<!\d)(?:3360|4032|4080|4128|4176|4224|4272|4320|4368|7392|8064|8112|8160|8208|8256|8304|8352|8400)(?!\d)')
ENTITY_TOKEN = re.compile(r'\b(?:entity|entities|series|column|col|roster)[ _-]?\d+\b', re.I)


def dp_text_check(text: str, where: str) -> None:
    """The parent's checks (data source names, old job tokens) plus this package's case ids (full D01_Q03 and short S06 / V03 / Q3 forms),
    anchor / block row numbers and entity / column numbers (task §4.2)."""
    W.skill_text_check(text, where)
    t = str(text or '')
    for rx, what in ((CASE_TOKEN, 'case id'), (ROW_TOKEN, 'row / cut position'), (ENTITY_TOKEN, 'entity / column number')):
        m = rx.search(t)
        if m:
            raise ValueError('%s carries a %s token %r (deployable text must not index cases, rows or entities)' % (where, what, m.group(0)))


def parent_branch_dirs(domain: str) -> list:
    """Every parent branch of the domain (12 cases): completed arms and the kept interrupted trajectories, in frozen case / arm order."""
    out = []
    for stage, tails in (('source', ('S%02d' % i for i in range(1, 9))), ('select', ('V01', 'V02')), ('metatest', ('Q01', 'Q02'))):
        sroot = PARENT_ROOT / stage
        for tail in tails:
            case = '%s_%s' % (domain, tail)
            for arm in PARENT_STAGE_ARMS[stage]:
                d = sroot / ('%s_%s' % (case, arm))
                if d.exists():
                    out.append({'stage': stage, 'case': case, 'arm': arm, 'dir': d, 'stage_root': sroot, 'kind': 'completed_or_failed', 'branch': d.name})
                for d2 in sorted(sroot.glob('%s_%s__interrupted_*' % (case, arm))):
                    out.append({'stage': stage, 'case': case, 'arm': arm, 'dir': d2, 'stage_root': sroot, 'kind': 'interrupted', 'branch': d2.name})
    return out


def _cards_of_domain(domain: str) -> dict:
    """Card texts that ran in the parent (looked up by rendered body): W_old (the frozen card), the three Select candidates, the generic text."""
    cards = {}
    oc = old_card(domain)
    cards['W_old'] = {'body': oc['injected_body'], 'skill_id': oc['skill_id'], 'role': 'the card frozen by the previous generation (selected on two Select cases); the historical hypothesis this package tests'}
    prop = context.read_json(PARENT_ROOT / 'formation' / domain / 'propose.json')
    for s in prop.get('skills', []):
        cid = s['skill_id'].split('-')[1]
        same = cid if s['rendered_body'] != oc['injected_body'] else 'W_old'
        cards['V_' + cid] = {'body': s['rendered_body'] if same != 'W_old' else 'identical to W_old', 'skill_id': s['skill_id'], 'role': 'candidate card %s tested on the two Select cases' % cid}
    cards['generic'] = {'body': GENERIC_TEXT, 'role': 'frozen generic guidance (no learning), one MetaTest arm'}
    return cards


def _card_ref(kn: dict, cards: dict) -> str | None:
    bodies = [e['body'] for e in kn.get('entries', [])]
    if not bodies:
        return None
    for name, c in cards.items():
        if c['body'] == bodies[0] or (c['body'] == 'identical to W_old' and cards['W_old']['body'] == bodies[0] and name != 'W_old'):
            return 'card:%s' % ('W_old' if c['body'] == 'identical to W_old' else name)
    return 'card:unlisted'


def _policy_summary(pol: dict) -> dict:
    out = {'default': ec.program_label(pol.get('default', {}).get('steps', [])), 'rules': []}
    for r in pol.get('rules', []) or []:
        w = r.get('when', {})
        out['rules'].append({'when': '%s %s %s' % (w.get('feature'), w.get('op'), json.dumps(w.get('value'))) if isinstance(w, dict) and 'feature' in w else json.dumps(w)[:120],
                             'steps': ec.program_label(r.get('steps', []))})
    return out


def _compact_result(tool: str, out, tier: int) -> dict:
    """The numbers of one tool result that a later decision could use; never the whole payload."""
    if not isinstance(out, dict):
        return {'value': str(out)[:200]}
    if tool == 'build_material':
        ms = out.get('material_spec', {})
        idx = ms.get('rule_index', [])
        return {'plan_id': out.get('plan_id'), 'label': ms.get('label'), 'rule_groups': {str(i): idx.count(i) for i in sorted(set(idx))} if idx else None, 'n_unknown': ms.get('n_unknown')}
    if tool == 'inspect_material':
        ident = {k: v for k, v in (out.get('step_identity_windows') or {}).items() if v}
        rss = out.get('recipe_step_sets')
        r = {'plan_id': out.get('material_id'), 'change_X': out.get('change_X'), 'change_y': out.get('change_y'), 'steps_executed_windows': out.get('steps_executed_windows'),
             'step_identity_windows_nonzero': ident or None, 'windows_bitwise_unchanged': out.get('windows_bitwise_unchanged'),
             'recipe_step_sets': (rss if (tier == 1 or not isinstance(rss, dict)) else {'n_distinct_step_sets': len(rss), 'top3': dict(sorted(rss.items(), key=lambda kv: -kv[1])[:3])})}
        if out.get('note'):
            r['note'] = out['note'][:160]
        if out.get('recipe_edit'):
            re_ = out['recipe_edit']
            r['recipe_edit'] = {k: re_.get(k) for k in ('skipped_windows_per_component', 'windows_no_retained_step')}
        w = out.get('window')
        if w and tier <= 2:
            pa, ch = np.asarray(w['parent'], dtype=float), np.asarray(w['child'], dtype=float)
            r['window'] = {'entity': w['entity'], 'rms_diff': round(float(np.sqrt(((ch - pa) ** 2).mean())), 4), 'parent_range': [round(float(pa.min()), 2), round(float(pa.max()), 2)],
                           'child_range': [round(float(ch.min()), 2), round(float(ch.max()), 2)]}
        if tier == 1 and isinstance(out.get('entities'), dict):
            r['entities_rms_X'] = {k: v.get('rms_X') for k, v in out['entities'].items()}
        return r
    if tool == 'inspect_data':
        ents = out.get('entities')
        r = {'kind': out.get('kind'), 'entities': list(ents) if isinstance(ents, dict) else ents, 'units': 'frozen T scaler'}
        if isinstance(ents, dict):
            r['per_entity'] = {}
            for k, v in list(ents.items())[:8]:
                vals = [x for x in (v if isinstance(v, list) else []) if isinstance(x, (int, float)) and not isinstance(x, bool)]
                if not vals:
                    continue
                st = {'n': len(vals), 'min': round(min(vals), 2), 'max': round(max(vals), 2), 'mean': round(statistics.fmean(vals), 2)}
                if tier <= 2 and len(vals) <= 28:
                    st['values_2dp'] = [round(x, 2) for x in vals]
                r['per_entity'][k] = st
        for k in ('absolute_rows', 'sub_range', 'row_range'):
            if k in out:
                r[k] = out[k]
        return r
    if tool == 'evaluate':
        fb = out.get('feedback', {})
        return {'plan_id': out.get('plan_id'), 'c_a_by_seed': [round(x, 5) for x in fb.get('loss_by_seed', [])], 'c_a_mean': round(fb.get('mean_loss'), 5) if fb.get('mean_loss') is not None else None,
                **({'c_a_by_seed_origin': [[round(x, 5) for x in row] for row in fb.get('loss_by_seed_origin', [])]} if tier == 1 else {})}
    if tool == 'compare':
        return {'a': out.get('a'), 'b': out.get('b'), 'mean_delta_a_minus_b_loss': round(out['mean_delta'], 5) if out.get('mean_delta') is not None else None,
                'positive_means': out.get('positive_means'), **({'delta_by_seed': [round(x, 5) for x in out.get('delta_by_seed', [])]} if tier <= 2 else {}),
                'seed_se': round(out['seed_se'], 5) if out.get('seed_se') is not None else None, 'signs': out.get('signs')}
    if tool == 'commit':
        return {'plan_id': out.get('plan_id'), 'reason': str(out.get('reason', ''))[:600]}
    if tool == 'overview':
        return {'note': 'overview read (identical to the case overview)'}
    return {k: (v if isinstance(v, (int, float, str, bool)) or v is None else '...') for k, v in list(out.items())[:8]}


def decision_points(rows: list, plan_to_phys: dict, labels: dict, tier: int) -> tuple:
    """(decision points, summary) of one trajectory: one record per Fast response, in time order, with what was visible then, the actions
    and their stated hypotheses, and the compact results; nothing is back-filled from later blocks."""
    built, evaluated, inspected_m, inspected_d, points = {}, {}, {}, 0, []
    calls = tools = new_evals = 0
    cur = None
    baseline_ids = next((r.get('baseline_ids', []) for r in rows if r.get('event') == 'job_started'), [])
    for r in rows:
        ev = r.get('event')
        if ev == 'fast_response':
            calls += 1
            cur = {'ref_event': r['event_id'], 'call': calls,
                   'visible_then': {'remaining_before': {'calls': LIMITS['max_calls'] - (calls - 1), 'tools': LIMITS['max_tools'] - tools, 'new_evaluations': LIMITS['max_new_evaluations'] - new_evals},
                                    'evaluated_c_a_mean_so_far': {p: v for p, v in evaluated.items()}, 'built_not_evaluated': [p for p in built if p not in evaluated],
                                    'inspections_so_far': {'inspect_data_calls': inspected_d, 'inspect_material_calls': sum(inspected_m.values()),
                                                           **({'inspect_material_by_plan': dict(inspected_m)} if tier == 1 else {})}},
                   'actions': [], 'results': [], 'hypotheses_stated': []}
            seen_inspect = {}
            for a in r.get('response', {}).get('actions', []):
                args = a.get('arguments', {}) if isinstance(a.get('arguments'), dict) else {}
                if a.get('tool') == 'inspect_material' and tier >= 2:
                    seen_inspect.setdefault(args.get('plan_id'), []).append(args.get('entity_index'))
                    continue
                act = {'tool': a.get('tool')}
                if a.get('tool') == 'build_material':
                    act.update(plan_id=args.get('plan_id'), policy=_policy_summary(args.get('policy', {}) if isinstance(args.get('policy'), dict) else {}),
                               observation_fields_used=(args.get('policy') or {}).get('observation_fields_used') if isinstance(args.get('policy'), dict) else None)
                    rat = (args.get('policy') or {}).get('rationale') if isinstance(args.get('policy'), dict) else None
                    if rat:
                        cur['hypotheses_stated'].append({'plan_id': args.get('plan_id'), 'rationale': str(rat)[:500 if tier <= 2 else 240]})
                elif a.get('tool') in ('evaluate', 'inspect_material'):
                    act.update(plan_id=args.get('plan_id'), **({'entity_index': args.get('entity_index'), 'window': args.get('window')} if a.get('tool') == 'inspect_material' else {}))
                elif a.get('tool') == 'inspect_data':
                    act.update(kind=args.get('kind'), entity_indices=args.get('entity_indices'), sub_range=args.get('sub_range'))
                elif a.get('tool') == 'compare':
                    act.update(a=args.get('a'), b=args.get('b'))
                elif a.get('tool') == 'commit':
                    act.update(plan_id=args.get('plan_id'))
                    cur['hypotheses_stated'].append({'commit': args.get('plan_id'), 'reason': str(args.get('reason', ''))[:600 if tier <= 2 else 300]})
                cur['actions'].append(act)
            if seen_inspect:
                cur['actions'].append({'tool': 'inspect_material', 'plans_and_entity_indices': seen_inspect})
            cur['_inspected_in_call'] = set()
            points.append(cur)
            continue
        if cur is None:
            continue
        if ev == 'tool_started':
            tools += 1
            continue
        if ev == 'tool_completed':
            tool, out = r.get('tool'), r.get('output')
            if tool == 'inspect_material' and tier >= 2 and isinstance(out, dict) and out.get('material_id') in cur['_inspected_in_call']:
                w = out.get('window') or {}
                cur['results'].append({'tool': 'inspect_material', 'plan_id': out.get('material_id'), 'same_numbers_as_first_inspection': True, 'window_entity': w.get('entity')})
                inspected_m[out.get('material_id')] = inspected_m.get(out.get('material_id'), 0) + 1
                continue
            res = _compact_result(tool, out, tier)
            res['tool'] = tool
            if tool == 'inspect_material' and isinstance(out, dict):
                cur['_inspected_in_call'].add(out.get('material_id'))
            if tool == 'build_material':
                pid = out.get('plan_id')
                built[pid] = {'phys': plan_to_phys.get(pid), 'label': (out.get('material_spec') or {}).get('label')}
                res['physical_material'] = plan_to_phys.get(pid)
            elif tool == 'evaluate':
                pid = out.get('plan_id')
                if pid not in baseline_ids and pid not in evaluated:
                    new_evals += 1
                evaluated[pid] = res.get('c_a_mean')
                res['physical_material'] = plan_to_phys.get(pid)
            elif tool == 'inspect_material':
                inspected_m[out.get('material_id')] = inspected_m.get(out.get('material_id'), 0) + 1
            elif tool == 'inspect_data':
                inspected_d += 1
            cur['results'].append(res)
        elif ev == 'tool_rejected':
            cur['results'].append({'tool': r.get('tool'), 'rejected': True, 'message': str((r.get('error') or {}).get('message', ''))[:200]})
        elif ev == 'action_batch_deferred':
            cur['results'].append({'deferred_actions': [a.get('tool') for a in r.get('actions', [])], 'reason': 'evidence roundtrip: read new evidence first'})
        elif ev == 'job_incomplete':
            cur['results'].append({'job_incomplete': r.get('failure_kind'), 'reason': str(r.get('reason', ''))[:200]})
    for p_ in points:
        p_.pop('_inspected_in_call', None)
    summary = {'calls': calls, 'tool_calls': tools, 'new_evaluations': new_evals, 'built': list(built), 'evaluated': [p for p in evaluated if p not in baseline_ids],
               'built_never_evaluated': [p for p in built if p not in evaluated], 'inspect_material_calls': sum(inspected_m.values()), 'inspect_data_calls': inspected_d,
               'built_labels': {p: b['label'] for p, b in built.items()}}
    return points, summary


def _overview_view(ov: dict, tier: int) -> dict:
    cols = W._columnar(ov['entities'])
    if tier >= 2:
        const = {k: v[0] for k, v in cols['columns'].items() if len({json.dumps(x) for x in v}) == 1}
        cols = {'n': cols['n'], 'columns': _round_tree({k: v for k, v in cols['columns'].items() if k not in const}, 4), 'constant_columns': const}
        summ = _round_tree({k: v for k, v in ov['summary'].items() if k not in const}, 4)
    else:
        summ = ov['summary']
    return {'n_entities': ov['n_entities'], 'summary': summ, 'entities': cols}


def _round_tree(x, nd):
    if nd is None:
        return x
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round_tree(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round_tree(v, nd) for v in x]
    return x


def _by_seed_origin(cells_or_scores: dict, mid: str, block: str, seeds=SEEDS):
    return ES._by_seed(cells_or_scores, mid, block, seeds)


def case_materials(bdirs: list, case: str, tier: int) -> dict:
    """One materials table per case keyed by physical id, merged over every branch of the case (C_A / C_B / E per seed; per origin at tier 1)."""
    mats = {}
    for b in bdirs:
        d = b['dir']
        reg = ec.load_registry(d / case)
        cells = ec.branch_cells(d / case)
        ca = {c: {'material_id': r['material_id'], 'model_seed': r['model_seed'], 'c_a': r['scores']['c_a']} for c, r in cells.items()}
        cb = context.read_json(d / case / 'c_b_scores.json')['cells'] if (d / case / 'c_b_scores.json').exists() else {}
        es = context.read_json(d / case / 'e_scores.json')['cells'] if (d / case / 'e_scores.json').exists() else {}
        for mid, r in reg.items():
            phys = r.get('phys_id', mid)
            asg = 'FixedMixup' if mid == 'FixedMixup' else (r.get('assignment') if mid not in PUBLIC else [PUBLIC_STEPS[mid]] * ec.COHORT_SIZE)
            uni = ec.uniform_steps(asg) if isinstance(asg, list) else None
            row = mats.setdefault(phys, {'label': ec.assignment_label(asg, mid if mid in PUBLIC else None), 'public_id': mid if mid in PUBLIC else None,
                                         'uniform_program': ec.program_label(uni) if uni is not None else None,
                                         'assignment_by_entity': ([ec.program_label(s) for s in asg] if (tier == 1 and isinstance(asg, list) and uni is None and mid not in PUBLIC) else None),
                                         'c_a_by_seed': None, 'c_b_by_seed': None, 'e_by_seed': None, 'e_status': 'unavailable', 'evaluated_by': []})
            for blk, src in (('c_a', ca), ('c_b', cb), ('e', es)):
                if row[blk + '_by_seed'] is None:
                    v, vo = _by_seed_origin(src, mid, blk)
                    if v is not None:
                        row[blk + '_by_seed'] = [round(x, 5) for x in v]
                        if tier == 1:
                            row[blk + '_by_seed_origin'] = [[round(x, 5) for x in o] for o in vo]
                        if blk == 'e':
                            row['e_status'] = 'development_late'
            if any(c['material_id'] == mid for c in cells.values()) and b['arm'] not in row['evaluated_by']:
                row['evaluated_by'].append(b['arm'] + ('(interrupted)' if b['kind'] == 'interrupted' else ''))
    none_e = statistics.fmean(mats['None']['e_by_seed']) if mats.get('None', {}).get('e_by_seed') else None
    for m in mats.values():
        m['e_ratio_to_none'] = round(statistics.fmean(m['e_by_seed']) / none_e, 4) if (none_e and m.get('e_by_seed')) else None
        m['c_a_mean'] = round(statistics.fmean(m['c_a_by_seed']), 5) if m.get('c_a_by_seed') else None
    return mats


def branch_record(b: dict, mats: dict, cards: dict, tier: int) -> tuple:
    """(record, legal refs) of one parent branch: decision points + unknown-then + offline verification + other-arm evidence pointer."""
    d, case = b['dir'], b['case']
    res = context.read_json(d / 'branch_result.json') if (d / 'branch_result.json').exists() else None
    kn = context.read_json(d / 'knowledge.json') if (d / 'knowledge.json').exists() else {'entries': []}
    reg = ec.load_registry(d / case)
    plan_to_phys = {mid: r.get('phys_id', mid) for mid, r in reg.items()}
    commit = context.read_json(d / case / 'commit.json') if (d / case / 'commit.json').exists() else None
    rows = ES._trace(d)
    if not rows and (d / 'trace_before_resume.jsonl').exists():
        rows = [json.loads(x) for x in (d / 'trace_before_resume.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    labels = {mid: mats.get(plan_to_phys.get(mid), {}).get('label') for mid in reg}
    points, summ = decision_points(rows, plan_to_phys, labels, tier)
    prefix = ref(b['stage'], b['branch'], '')
    for p in points:
        p['ref'] = ref(b['stage'], b['branch'], p.pop('ref_event'))
    refs = [ref(b['stage'], b['branch'], r['event_id']) for r in rows] + [ref(b['stage'], b['branch'], 'overview'), ref(b['stage'], b['branch'], 'labels_after_commit')]
    tokens = dsks._unit_tokens(b['stage_root']).get('fast:%s' % b['branch'].split('__interrupted')[0], {}) if (b['stage_root'] / 'raw_responses').exists() else {}
    committed = res.get('committed_plan_id') if res else None
    committed_phys = plan_to_phys.get(committed) if committed else None
    evaluated_phys = sorted({plan_to_phys[p] for p in summ['evaluated'] if p in plan_to_phys} | {m for m in PUBLIC if m in reg})
    verification = {}
    for phys in evaluated_phys:
        m = mats.get(phys, {})
        if tier == 1:
            verification[phys] = {k: m.get(k) for k in ('label', 'c_a_by_seed', 'c_b_by_seed', 'e_by_seed', 'e_ratio_to_none', 'c_a_by_seed_origin', 'c_b_by_seed_origin', 'e_by_seed_origin') if m.get(k) is not None}
        else:
            verification[phys] = {'label': m.get('label'), 'c_a_mean': m.get('c_a_mean'), 'e_ratio_to_none': m.get('e_ratio_to_none'), 'per_seed_numbers': 'see case.materials[%s]' % phys}
    with_e = {p: statistics.fmean(mats[p]['e_by_seed']) for p in verification if mats.get(p, {}).get('e_by_seed')}
    with_ca = {p: statistics.fmean(mats[p]['c_a_by_seed']) for p in verification if mats.get(p, {}).get('c_a_by_seed')}
    order = list(verification)
    post_hoc = {'note': 'DIAGNOSTIC on the branch\'s own evaluated pool; the E best is not a known deployment answer',
                'c_a_argmin_physical': min(with_ca, key=lambda p: (with_ca[p], order.index(p))) if with_ca else None,
                'e_best_physical': min(with_e, key=lambda p: (with_e[p], order.index(p))) if with_e else None, 'commit_physical': committed_phys,
                'commit_is_c_a_argmin': (min(with_ca, key=lambda p: (with_ca[p], order.index(p))) == committed_phys) if (with_ca and committed_phys) else None}
    interrupted = b['kind'] == 'interrupted'
    rec = {'ref_prefix': prefix, 'case_ref': case, 'arm': b['arm'], 'trajectory_kind': ('interrupted (operator stop / restart; facts only, no commit, cost unknown)' if interrupted else 'completed_or_failed'),
           'knowledge_loaded': _card_ref(kn, cards), 'status': res['status'] if res else ('INTERRUPTED' if interrupted else 'NOT_RUN'),
           'failure_kind': res.get('failure_kind') if res else None,
           'decision_points': points,
           'unknown_then': {'built_never_evaluated': [{'plan_id': p, 'label': summ['built_labels'].get(p), 'downstream_feedback': 'none (never fitted; no C_A / C_B / E exists for it; not an oracle)'} for p in summ['built_never_evaluated']],
                            'blocks_not_open_at_decision_time': 'C_B and E of this case were closed during the whole trajectory; they appear only under offline_verification'},
           'offline_verification': {'supervision_note': 'per-seed C_A / C_B / E of every material this branch evaluated (E = development_late, opened after all commits; given to Slow as supervision only)',
                                    'materials': verification, 'commit_plan': committed, 'commit_physical': committed_phys, 'commit_reason': (commit or {}).get('reason', '')[:600] if commit else None,
                                    'post_hoc': post_hoc},
           'cost': ({'fast_calls': res['calls'], 'tool_calls': res['tool_calls'], 'new_evaluations': res['new_evaluations'], 'tokens': tokens or 'unknown'} if (res and not interrupted)
                    else {'fast_calls': summ['calls'], 'tool_calls': summ['tool_calls'], 'new_evaluations': summ['new_evaluations'], 'tokens': 'unknown (interrupted trajectory shares the unit name of the restarted one)'}),
           'summary': {k: summ[k] for k in ('calls', 'tool_calls', 'new_evaluations', 'built', 'evaluated', 'built_never_evaluated', 'inspect_material_calls', 'inspect_data_calls')}}
    return rec, refs


def census_dp(domain: str, tier: int = 1) -> dict:
    """Twelve parent cases of one domain organised as decision points (task §4.1). Reads only the parent package; never a new-case directory."""
    cards = _cards_of_domain(domain)
    bdirs = parent_branch_dirs(domain)
    by_case = {}
    for b in bdirs:
        by_case.setdefault(b['case'], []).append(b)
    split = context.read_json(PARENT_SPLIT_DOC)
    cases, refs = {}, []
    for case, bl in by_case.items():
        mats = case_materials(bl, case, tier)
        ov_dir = next((b['dir'] for b in bl if (b['dir'] / case / 'overview.json').exists()), None)
        ov = context.read_json(ov_dir / case / 'overview.json') if ov_dir else None
        recs = []
        for b in bl:
            if not (b['dir'] / 'trace.jsonl').exists() and not (b['dir'] / 'trace_before_resume.jsonl').exists():
                recs.append({'ref_prefix': ref(b['stage'], b['branch'], ''), 'arm': b['arm'], 'status': 'NOT_RUN'})
                continue
            rec, r_ = branch_record(b, mats, cards, tier)
            refs += r_
            recs.append(rec)
        for rec in recs:
            mine = set((rec.get('offline_verification') or {}).get('materials', {}))
            others = {}
            for other in recs:
                if other is rec or other.get('status') == 'NOT_RUN':
                    continue
                for phys in (other.get('offline_verification') or {}).get('materials', {}):
                    if phys not in mine:
                        others.setdefault(phys, []).append(other['arm'])
            rec['other_arm_evidence'] = {'note': 'materials evaluated only by OTHER arms of the same case (same entities, same blocks); NOT visible to this trajectory at decision time; numbers in case.materials',
                                         'physical_materials': {p: {'arms': a, 'label': mats.get(p, {}).get('label'), 'e_ratio_to_none': mats.get(p, {}).get('e_ratio_to_none')} for p, a in sorted(others.items())}}
        cases[case] = {'case_kind': PARENT_CASE_STAGE[case.split('_')[1][0]], 'cut_position': 'anchor_%d' % (1 + (ov['train_rows'][1] == split['anchors'][1])) if ov else None,
                       'train_rows': ov['train_rows'] if ov else None,
                       'overview': _overview_view(ov, tier) if ov else None,
                       'materials': mats, 'branches': recs}
    actions = ES.action_table()
    actions['primitives'] = {n: {k: v for k, v in p.items() if k != 'source'} for n, p in actions['primitives'].items()}
    n_completed = sum(1 for c in cases.values() for r in c['branches'] if r.get('status') == 'COMPLETE')
    ev = {'census_status': 'CENSUS_COMPLETE' if len(cases) == 12 and n_completed >= 24 else 'CENSUS_INCOMPLETE', 'domain_id': domain, 'purpose': 'decision_priority_learning',
          'profile': ec.PROFILE_VERSION, 'material_origin': 'the previous generation of this domain: 8 no-card research cases, 2 Select cases (three candidate cards each), '
                                                            '2 later cases (no-card / previous card / generic / random search / fixed program); all completed and labelled',
          'organisation': 'per case: one T-only overview, one materials table (physical id -> label, program, C_A / C_B / E per seed), one record per trajectory organised as DECISION POINTS '
                          '(what was visible then, the actions and stated hypotheses, compact results), what was unknown then, the OFFLINE verification (supervision), and the '
                          'materials only other arms evaluated (not visible then). Every case counts once; a longer trajectory is not more evidence.',
          'cases_in_order': list(cases), 'cases': cases, 'n_branches_completed': n_completed,
          'cards_that_ran': {k: {kk: vv for kk, vv in v.items() if kk != 'body'} for k, v in cards.items()},
          'legal_evidence_refs': sorted(set(refs)), 'compression': {'tier_used': str(tier), 'tier_meaning': {'1': 'per-origin numbers, per-entity assignments, inspect window stats',
                                                                                                         '2': 'per-seed numbers only, inspect window stats, rationale <=500 chars',
                                                                                                         '3': 'per-seed numbers only, no window stats, rationale <=240 chars'},
                                                                   'never_dropped': 'actions, plans, C_A / C_B / E per seed, commits, failures, rejections, hypotheses (truncated at most)'},
          'feedback_roles': {'C_A': 'immediate feedback used by Fast at decision time (origins t, t+48)', 'C_B': 'delayed check of the same case (origins t+96, t+144), opened after commit',
                             'E': 'development_late: the late block (origins t+192..t+336), opened after every commit of that stage; authorized for learning only'},
          'public_semantics': {'field_definitions': FIELD_DEFINITIONS, 'actions': actions, 'tool_contracts': CONTRACTS, 'consumer': ec.CONSUMER,
                               'geometry': {'L': spec.L, 'H': spec.H, 'T': spec.TRAIN_SPAN, 'n_entities': ec.COHORT_SIZE, 'n_parents_per_entity': spec.N_PARENTS},
                               'public_references': {m: ES.public_spec(m) for m in PUBLIC}, 'budget_per_case': LIMITS, 'body_limit_characters': BODY_LIMIT,
                               'metric': 'normalized MSE (frozen T scaler), entity macro; lower is better',
                               'material_semantics_sentence_all_fast_arms': MATERIAL_SEMANTICS, 'case_count': '12 entity-group cases (16 entities each) at two cut positions; not independent time environments'}}
    low = json.dumps(ev, ensure_ascii=False).lower()
    leak = [n for n in tuple(spec.DATASETS) if n in low]
    if leak:
        raise PermissionError('census names a data source: %s' % leak)
    return ev


# ============================================================================= Slow: three independent single-card proposals per domain
FEEDBACK_ROLES_DP = ('Feedback roles, by name: C_A = the immediate feedback Fast legally used at decision time (origins t, t+48); it explains why Fast decided as it did. '
                     'C_B = the delayed check of that case after its commit (origins t+96, t+144). E (development_late) = the late block of that case (origins t+192..t+336), the '
                     'criterion this study cares about, opened only after every commit of that stage and authorized for learning only. When C_A and E disagree, KEEP the conflict: a C_A '
                     'lead is not a later gain, and no rule may a priori ignore all C_A. All blocks are the same 48-hour forecast; they differ in the calendar period scored. The twelve '
                     'cases are different entity groups of ONE domain at two cut positions; the cases this card will serve are NEW entity groups of the same domain. A per-entity '
                     'prediction change is not the independent contribution of that entity\'s material; grouping by observations stays legal and is judged by the complete shared model. '
                     'A material that was built but never evaluated has NO downstream number: it is unverified, not shown harmful, and the post-hoc pool best is a diagnostic, never a '
                     'known answer. Material appearance (smoother, flatter, larger change) is not downstream utility in either direction. Uncertainty can justify an extra check, a '
                     'limited trial or keeping a public reference; one undecided comparison never becomes a permanent ban; an untried action is not harmful.')
GOAL_DP = ('Your goal (task §4.2): 在有限的四次新方案评估内，学习哪些观察值得取得、哪些处理假说值得优先实测、什么反馈会改变下一项实验，以及何时继续或提交。优化实际下游交付；成本是附加维度，'
           '不能通过永远不实验冒充交付进步。请提出相对旧卡可检验的决策差异。不要因原语没被试过就禁止它，不从外观直接推断效用，不把旧卡正文当证据。允许完整组合和条件化；是否用它们由当前证据决定，不强制复杂。 '
           '(Within the four new-plan evaluations per case, learn which observations are worth obtaining, which processing hypotheses deserve a real test first, which feedback would '
           'change the next experiment, and when to continue or commit. Optimise the actual downstream delivery; cost is an additional dimension - never experimenting is not delivery '
           'progress. Propose decision differences relative to the old card that can be TESTED. Do not ban a primitive because it was not tried; do not infer utility from appearance; '
           'do not treat the old card\'s text as evidence. Complete compositions and observation-conditional plans are allowed; whether to use them follows the current evidence; '
           'complexity is not required.) You write ONE card (Workflow + optional Principles) or KEEP. Soft priorities never change tool permissions; a card may not forbid grouping, '
           'three-step compositions or any public tool as a global ban "because untested". Do not require "must dominate" or "3/3". State the COMMIT POLICY explicitly. Say which '
           'conditions the current tools compute and which are learned preferences. Give every main rule its support / counter refs and mark it supported / hypothesis / unresolved; '
           'a kept old rule needs the same. Name the ONE main decision mechanism you change and its deployment-observable trigger, one historical decision point your card would have '
           'changed (expected, not run), one counterexample where the card should hold or may fail, and the conditions under which your card behaves like the old card. KEEP or an '
           'undecided rule is allowed when the evidence is insufficient; the format never forces novelty.')
NO_IDS_DP = ('Do not write data source names, case ids (also short forms like S06 / V03 / Q3), cut positions or row numbers, entity or column numbers, historical assignments or '
             'answer tables into deployable text (workflow, principles, applicability_summary, research_mode); evidence_refs and the two ref fields carry the references. Do not change '
             'model, scoring, tools, permissions, budgets, randomness, targets or legal windows. observable_applicability MUST be exactly {"const":true}; applicability conditions belong '
             'inside the Workflow text as observable conditions.')
FORMAT_DP = ('Output exact JSON, no markdown: {"decision":"KEEP","rationale":"...","evidence_review":{"supports":[...],"contradicts_or_limits":[...],"rule_changes":[...],'
             '"uncertain_and_expected_behavior_change":"..."},"conditions_same_as_old_card":"..."} OR {"decision":"PROPOSE","candidate":{"research_mode":"<short label>",'
             '"workflow":"<executable Workflow incl. observe / construct / experiment / commit-or-stop; the commit policy must be inside it>","principles":"<conditions, actions, reasons, exceptions>" or null,'
             '"observable_applicability":{"const":true},"applicability_summary":"<=400 characters","evidence_refs":["exact refs from the supplied legal ranges"],"rationale":"...",'
             '"evidence_review":{"supports":["observation -> which advice it supports"],"contradicts_or_limits":["which later result (C_B / E) contradicts or limits which judgement"],'
             '"rule_changes":["which rule is kept / dropped / rewritten because of it"],"uncertain_and_expected_behavior_change":"what remains uncertain and which Fast behaviour should change"},'
             '"rule_status":[{"rule":"<one main rule>","status":"supported|hypothesis|unresolved","support":["refs or short facts"],"counter":["refs or short facts"]}],'
             '"commit_policy":"<one paragraph restating the commit rule>","main_mechanism_changed":{"mechanism":"<the one decision mechanism>","deployment_observable_trigger":"<what the tools show that triggers it>"},'
             '"decision_change_vs_old_card":"<the concrete decision difference>","historical_decision_point_expected_to_change":{"ref":"<one legal ref>","expected_change":"..."},'
             '"counterexample_expected":{"ref":"<one legal ref>","should_hold_or_may_fail":"..."},"conditions_same_as_old_card":"..."}}. Workflow plus Principles render to at most %d characters. '
             'KEEP (= the old card stays; recorded as an alias) is valid and is not resampled.')
SLOW_DP = ('You are the offline Slow of a batch training-data augmentation research Harness. You receive a deterministic census of TWELVE completed research cases of ONE neutral '
           'domain (entity groups of 16 at two cut positions): eight researched once by the no-card Fast, two Select cases researched by three candidate cards each, and two later '
           'cases researched by the no-card Fast, the previous-generation card, a generic guidance, a random search and a fixed program. Each trajectory is organised as DECISION '
           'POINTS: what was visible then (T observations, C_A obtained so far, built / evaluated sets, remaining budget), the actions with their stated hypotheses, the compact tool '
           'results, what was unknown then, and - separately, as supervision for you - the OFFLINE verification: per-seed C_A / C_B / E of every material that trajectory evaluated, '
           'its real commit and cost. Materials evaluated only by other arms of the same case are listed apart and were not visible to that trajectory. The previous-generation card '
           '(W_old) is given in its own section as a HISTORICAL HYPOTHESIS UNDER TEST, not as a fact; its own trajectories are in the census. You are one of three independent Slow '
           'proposals formed from the identical evidence; you do not see the other two. ' + FEEDBACK_ROLES_DP + ' ' + GOAL_DP + ' ' + NO_IDS_DP + ' ' + FORMAT_DP % BODY_LIMIT)
CAND_KEYS_DP = {'research_mode', 'workflow', 'principles', 'observable_applicability', 'applicability_summary', 'evidence_refs', 'rationale', 'evidence_review', 'rule_status',
                'commit_policy', 'main_mechanism_changed', 'decision_change_vs_old_card', 'historical_decision_point_expected_to_change', 'counterexample_expected', 'conditions_same_as_old_card'}


def payload_bytes(system: str, payload: dict) -> int:
    return len(json.dumps([{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], ensure_ascii=False).encode('utf-8'))


def slow_payload(cen: dict, domain: str, slot: int) -> dict:
    oc = old_card(domain)
    refs = cen['legal_evidence_refs']
    body = {k: v for k, v in cen.items() if k != 'legal_evidence_refs'}
    return {'domain_id': domain, 'proposal_slot': slot, 'proposal_note': 'proposal %d of three independent proposals formed from the identical census; the others are not visible to you' % slot,
            'census': body,
            'historical_hypothesis_under_test': {'W_old': oc['injected_body'], 'origin': 'the card frozen by the previous generation of this domain after its own Select; it ran as '
                                                                                        'card:W_old on the two later cases (and as one of the Select candidates); those trajectories are facts in the census',
                                                 'status': 'a hypothesis to test against the evidence, not inherited; KEEP is allowed if the evidence does not support a decision difference'},
            'body_limit_characters': BODY_LIMIT, 'legal_evidence_refs': LL.ref_ranges(refs),
            'legal_evidence_refs_format': 'evidence_refs must be exact ids inside these ranges, e.g. "source/<branch>/<case>:<integer>", "select/<branch>/<case>:<integer>", "metatest/<branch>/<case>:<integer>", or one of other_refs',
            'required': {'observable_applicability': {'const': True}}, 'status': 'CANDIDATE_TEST_ONLY'}


def make_card_dp(*, domain, skill_id, workflow, principles, summary, refs, legal_refs, mode) -> dsk.Skill:
    if not isinstance(mode, str) or not mode.strip() or len(mode) > 80:
        raise ValueError('research_mode must be short nonempty text')
    dsk.check_reusable_text(mode, 'research_mode', dp_text_check)
    return dsk.make_skill(skill_id=skill_id, domain_id=domain, revision=1, workflow=workflow, principles=principles, applicability_summary=summary,
                          compatibility_note='Formed for this study\'s fixed task, Consumer, L/H, hourly sampling, T length and 16-entity cases.', observable_applicability={'const': True},
                          evidence_refs=refs, legal_evidence_refs=legal_refs, source_stage='propose', status='CANDIDATE_TEST_ONLY',
                          allowed_features=ALLOWED_FEATURES, text_check=dp_text_check, body_limit=BODY_LIMIT)


def _legal_ref(x, legal: set, where: str) -> str:
    if not isinstance(x, dict) or not isinstance(x.get('ref'), str) or x['ref'] not in legal:
        raise ValueError('%s.ref must be one exact legal evidence ref' % where)
    other = [k for k in x if k != 'ref']
    if len(other) != 1 or not isinstance(x[other[0]], str) or not x[other[0]].strip():
        raise ValueError('%s carries ref and exactly one nonempty text field' % where)
    return x['ref']


def parse_proposal(resp, *, domain: str, slot: int, legal_refs) -> dict:
    """-> {'decision': 'KEEP'|'PROPOSE', 'skill': Skill|None, 'meta': {...}}; ValueError on contract errors."""
    legal = set(legal_refs)
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'PROPOSE'):
        raise ValueError('decision must be KEEP or PROPOSE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale', 'evidence_review', 'conditions_same_as_old_card'}:
            raise ValueError('KEEP carries only rationale, evidence_review and conditions_same_as_old_card')
        return {'decision': 'KEEP', 'skill': None, 'meta': {'rationale': str(resp.get('rationale', ''))[:3000], 'evidence_review': LL.check_review(resp['evidence_review']) if 'evidence_review' in resp else None,
                                                            'conditions_same_as_old_card': str(resp.get('conditions_same_as_old_card', ''))[:1500]}}
    if set(resp) != {'decision', 'candidate'} or not isinstance(resp['candidate'], dict):
        raise ValueError('PROPOSE carries exactly decision and one candidate object')
    c = resp['candidate']
    if set(c) != CAND_KEYS_DP:
        raise ValueError('candidate keys must be exactly %s' % sorted(CAND_KEYS_DP))
    if c['observable_applicability'] != {'const': True}:
        raise ValueError('observable_applicability must be exactly {"const": true}')
    for k in ('commit_policy', 'decision_change_vs_old_card', 'conditions_same_as_old_card'):
        if not isinstance(c[k], str) or not c[k].strip():
            raise ValueError('%s must be nonempty text' % k)
    mm = c['main_mechanism_changed']
    if not isinstance(mm, dict) or set(mm) != {'mechanism', 'deployment_observable_trigger'} or any(not isinstance(mm[k], str) or not mm[k].strip() for k in mm):
        raise ValueError('main_mechanism_changed must be {mechanism, deployment_observable_trigger} with nonempty text')
    hist_ref = _legal_ref(c['historical_decision_point_expected_to_change'], legal, 'historical_decision_point_expected_to_change')
    counter_ref = _legal_ref(c['counterexample_expected'], legal, 'counterexample_expected')
    review = LL.check_review(c['evidence_review'])
    rs = ES.check_rule_status(c['rule_status'])
    s = make_card_dp(domain=domain, skill_id='%s-N%d-r1' % (domain, slot), workflow=c['workflow'], principles=c['principles'], summary=c['applicability_summary'],
                     refs=c['evidence_refs'], legal_refs=legal_refs, mode=c['research_mode'])
    meta = {'research_mode': c['research_mode'], 'rationale': str(c['rationale'])[:3000], 'evidence_review': review, 'rule_status': rs, 'commit_policy': c['commit_policy'][:2000],
            'main_mechanism_changed': {k: mm[k][:800] for k in mm}, 'decision_change_vs_old_card': c['decision_change_vs_old_card'][:2000],
            'historical_decision_point_expected_to_change': {**c['historical_decision_point_expected_to_change'], 'ref': hist_ref},
            'counterexample_expected': {**c['counterexample_expected'], 'ref': counter_ref}, 'conditions_same_as_old_card': c['conditions_same_as_old_card'][:1500]}
    return {'decision': 'PROPOSE', 'skill': s, 'meta': meta}


def propose_one(payload: dict, call, *, domain: str, slot: int, legal_refs) -> dict:
    """One scientific proposal call; one contract correction only; transport / account / budget faults end as PROPOSE_CALL_FAILED."""
    low = json.dumps(payload, ensure_ascii=False).lower()
    if any(n in low for n in tuple(spec.DATASETS)):
        raise PermissionError('Slow payload names a data source')
    attempts, prior, receipts = [], None, []
    for i in range(2):
        try:
            raw, rec = call(payload)
            receipts.append(rec)
        except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
            attempts.append({'attempt': i, 'fault_kind': type(exc).__name__})
            return {'status': 'PROPOSE_CALL_FAILED', 'attempts': attempts, 'receipts': receipts, 'skill': None, 'meta': {}}
        except ValueError as exc:
            attempts.append({'attempt': i, 'error': 'response not JSON: %s' % str(exc)[:200]})
            prior = None
        else:
            try:
                out = parse_proposal(raw, domain=domain, slot=slot, legal_refs=legal_refs)
                attempts.append({'attempt': i, 'ok': True})
                return {'status': 'KEEP' if out['decision'] == 'KEEP' else 'PROPOSED', 'attempts': attempts, 'receipts': receipts, 'raw': raw, **out}
            except ValueError as exc:
                attempts.append({'attempt': i, 'error': str(exc)[:300]})
                prior = raw if dsk._jsonable(raw) else None
        if i == 0:
            payload = {**payload, 'correction': {'previous_output': prior, 'error': attempts[-1].get('error'), 'instruction': 'Correct this contract error only. KEEP is allowed. Do not seek a different outcome.'}}
    return {'status': 'PROPOSE_PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'receipts': receipts, 'skill': None, 'meta': {}}


def build_census(root: Path, domain: str) -> dict:
    """The frozen census of the domain (tier chosen so that the payload fits the byte target) written once under evidence/<D>/."""
    P_ = paths(root)
    out = P_['evidence'] / domain
    out.mkdir(parents=True, exist_ok=True)
    p = out / 'census.json'
    if p.exists():
        return context.read_json(p)
    sizes, chosen = {}, None
    for tier in (1, 2, 3):
        cen = census_dp(domain, tier)
        nb = payload_bytes(SLOW_DP, slow_payload(cen, domain, 1))
        sizes['tier_%d_bytes' % tier] = nb
        chosen = cen
        if nb <= CENSUS_BYTE_TARGET:
            break
    chosen['compression']['payload_bytes_by_tier'] = sizes
    chosen['compression']['byte_target'] = CENSUS_BYTE_TARGET
    chosen['compression']['frozen_local'] = now()
    dsks.write_once(p, chosen)
    return chosen


def slow_stage(root: Path, driver_lock: PackageLock) -> dict:
    """Task §4.2: three independent proposals per domain (six calls through the HTTP pool), each with at most one contract correction; KEEP or a
    byte-identical body is an alias. 0 fits. Writes slow/<D>/proposal_N<k>.json and slow/<D>/proposals.json."""
    P_ = paths(root)
    P_['slow'].mkdir(parents=True, exist_ok=True)
    summary_p = P_['slow'] / 'proposals.json'
    if summary_p.exists():
        return context.read_json(summary_p)
    censuses = {d: build_census(root, d) for d in DOMAINS}
    for d in DOMAINS:
        if censuses[d]['census_status'] != 'CENSUS_COMPLETE':
            raise RuntimeError('census of %s incomplete: %s' % (d, censuses[d]['census_status']))
    payloads = {(d, k): slow_payload(censuses[d], d, k) for d in DOMAINS for k in (1, 2, 3)}
    reservation = {'%s_N%d' % (d, k): {'bytes': payload_bytes(SLOW_DP, p), 'reserved_tokens_upper_2x': 2 * (payload_bytes(SLOW_DP, p) + 2048 + MAX_OUTPUT_TOKENS)} for (d, k), p in payloads.items()}
    if not (P_['slow'] / 'reservation_check.json').exists():
        dsks.write_once(P_['slow'] / 'reservation_check.json', {'checked_local': now(), 'payloads': reservation, 'note': 'no call sent by this check; sizes of the final serialized payloads'})
    sc = stage_caps(root, 'slow')
    if not proxy_reachable():
        raise RuntimeError('LLM proxy not reachable; Slow refused before any paid call')
    start_paid_clock(root)                                            # before the stage's own ledger instance is opened (one writer at a time)
    led = SafeLedger(P_['ledger'], **sc['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage; paid Slow refused')
    client = MeteredClient(led, P_['slow'] / 'raw_responses', http_cap=sc['http_cap'], stage='slow')
    results, lock = {}, threading.Lock()

    def one(d, k):
        outp = P_['slow'] / d / ('proposal_N%d.json' % k)
        if outp.exists():
            prior = context.read_json(outp)
            if prior['status'] in ('PROPOSED', 'KEEP', 'PROPOSE_PARSE_OR_VALIDATION_FAILED'):
                with lock:
                    results[(d, k)] = prior
                return
            n_prev = len(list((P_['slow'] / d).glob('proposal_N%d_failed_*.json' % k)))
            shutil.move(outp, P_['slow'] / d / ('proposal_N%d_failed_%d.json' % (k, n_prev + 1)))    # technical failure: the same frozen payload is called again after the operator decision
        legal = censuses[d]['legal_evidence_refs']
        res = propose_one(payloads[(d, k)], lambda p: client.call('slow_propose', '%s_N%d' % (d, k), p, SLOW_DP, max_tokens=MAX_OUTPUT_TOKENS, meta={'domain': d, 'slot': k}),
                          domain=d, slot=k, legal_refs=legal)
        rec = {**{kk: v for kk, v in res.items() if kk != 'skill'}, 'skill': res['skill'].to_json() if res.get('skill') else None, 'domain_id': d, 'slot': k, 'written_local': now()}
        outp.parent.mkdir(parents=True, exist_ok=True)
        dsks.write_once(outp, rec)
        with lock:
            results[(d, k)] = rec
    errs = _parallel([(d, k) for d in DOMAINS for k in (1, 2, 3)], one)
    if errs:
        raise RuntimeError('slow proposal thread failed: %s' % errs[0][1])
    failed = [(d, k) for (d, k), r in results.items() if r['status'] == 'PROPOSE_CALL_FAILED']
    if client.fatal or rt.unknown_usage_blocks(led) or failed:
        raise RuntimeError('Slow stopped: fatal=%s unknown_usage_blocks=%s call_failed=%s; finished proposals are kept, rerun --run (with --accept-unknown-usage after the operator decision) to call only the unfinished slots'
                           % (client.fatal, rt.unknown_usage_blocks(led), ['%s_N%d' % x for x in failed]))
    summary = {'written_local': now(), 'domains': {}}
    for d in DOMAINS:
        oc = old_card(d)
        rows, bodies = [], {'W_old': oc['injected_body']}
        for k in (1, 2, 3):
            r = results[(d, k)]
            row = {'slot': k, 'arm': 'n%d' % k, 'status': r['status'], 'attempts': r['attempts'], 'requests': [x.get('request') for x in r.get('receipts', [])],
                   'tokens': sum((x.get('prompt_tokens') or 0) + (x.get('completion_tokens') or 0) for x in r.get('receipts', [])), 'alias_of': None}
            if r['status'] == 'KEEP':
                row['alias_of'] = 'w_old'
                row['alias_reason'] = 'KEEP: the old card stays'
            elif r['status'] == 'PROPOSED':
                body = r['skill']['rendered_body']
                same = next((name for name, b in bodies.items() if b == body), None)
                if same is not None:
                    row['alias_of'] = 'w_old' if same == 'W_old' else same
                    row['alias_reason'] = 'byte-identical rendered body to %s' % same
                else:
                    bodies['n%d' % k] = body
                row['research_mode'] = r['meta'].get('research_mode')
                row['main_mechanism_changed'] = r['meta'].get('main_mechanism_changed')
            else:
                row['alias_reason'] = 'technical failure: not a KEEP, not an alias; no card for this slot'
            rows.append(row)
        summary['domains'][d] = {'old_card_skill_id': oc['skill_id'], 'proposals': rows, 'n_distinct_new_cards': sum(1 for r in rows if r['status'] == 'PROPOSED' and r['alias_of'] is None),
                                 'census_tier': censuses[d]['compression']['tier_used'], 'census_bytes': censuses[d]['compression']['payload_bytes_by_tier']}
    dsks.write_once(summary_p, summary)
    return summary


def new_cards(root: Path, domain: str) -> dict:
    """{arm: Skill} of the domain's distinct new cards (aliases excluded)."""
    P_ = paths(root)
    summ = context.read_json(P_['slow'] / 'proposals.json')
    out = {}
    for row in summ['domains'][domain]['proposals']:
        if row['status'] == 'PROPOSED' and row['alias_of'] is None:
            rec = context.read_json(P_['slow'] / domain / ('proposal_N%d.json' % row['slot']))
            out[row['arm']] = dsk.skill_from_json(rec['skill'])
    return out


def select_aliases(root: Path, domain: str) -> dict:
    summ = context.read_json(paths(root)['slow'] / 'proposals.json')
    return {row['arm']: row['alias_of'] for row in summ['domains'][domain]['proposals'] if row['alias_of']}


# ============================================================================= Select (V03 / V04 per domain): old card vs the new proposals; J on E; freeze
def rotate(base: tuple, idx: int) -> list:
    r = int(idx) % len(base)
    return list(base[r:]) + list(base[:r])


def select_stage(root: Path, driver_lock: PackageLock) -> dict:
    cases_all = cases_of(root)
    knowledge, order, aliases, cases = {}, {}, {}, []
    for d in DOMAINS:
        oc = old_card(d)
        cards = new_cards(root, d)
        al = select_aliases(root, d)
        arms = {'w_old': dsk.skill_knowledge(oc['skill'])}
        arms.update({arm: dsk.skill_knowledge(s) for arm, s in cards.items()})
        for tail in ('V03', 'V04'):
            c = '%s_%s' % (d, tail)
            cs = cases_all[c]
            cases.append(c)
            order[c] = [a for a in rotate(SELECT_ARMS, cs.case_index) if a in arms]
            knowledge[c] = {a: asdict(k) for a, k in arms.items()}
            aliases[c] = dict(al)
    notes = {a: 'alias (KEEP or byte-identical body): the aliased arm\'s trajectory and scores are reused; no second LLM run' for a in ('n1', 'n2', 'n3')}
    cfgp = stage_config(root, 'select', cases=cases, order=order, knowledge=br.json_copy(knowledge), labels_e=True, llm=True, random=False, aliases=aliases, alias_notes=notes)
    return run_stage(root, 'select', cfgp, driver_lock)


def _block_vec(bdir: Path, case: str, plan: str, block: str):
    return ES._block_vec(bdir, case, plan, block)


def _committed(bdir: Path, case: str):
    return ES._committed(bdir, case)


def _arm_dir(stage_root: Path, case: str, arm: str, aliases: dict) -> tuple:
    """(branch dir, alias_of) resolving alias chains of one case."""
    seen, a = set(), arm
    while a in aliases.get(case, {}) and a not in seen:
        seen.add(a)
        a = aliases[case][a]
    return stage_root / ('%s_%s' % (case, a)), (a if a != arm else None)


def select_domain(root: Path, domain: str) -> dict:
    """J(W) = mean over V03/V04 of mean_seed E(actual commit of W) / mean_seed E(None); W_new = lowest J among the distinct new cards; W_old's J recorded;
    Fixed_dev = lowest-J uniform program evaluated on BOTH Select cases (any branch, incl. public); H_deploy = lowest J among W_new / W_old / Fixed_dev
    (ties: fixed program first, then fewer evaluations, fewer tokens)."""
    P_ = paths(root)
    cfg = context.read_json(P_['configs'] / 'select.json')
    sel_cases = ['%s_V03' % domain, '%s_V04' % domain]
    arms = ['w_old'] + [a for a in ('n1', 'n2', 'n3') if a not in select_aliases(root, domain)]
    aliases = cfg.get('aliases', {})
    tokens_all = dsks._unit_tokens(P_['select']) if (P_['select'] / 'raw_responses').exists() else {}
    runs, J, incomplete, cost = {}, {}, [], {}
    for a in arms:
        runs[a], ratios = {}, []
        for c in sel_cases:
            b, alias_of = _arm_dir(P_['select'], c, a, aliases)
            plan = _committed(b, c)
            e = _block_vec(b, c, plan, 'e') if plan else None
            none_e = _block_vec(b, c, 'None', 'e') if b.exists() else None
            res = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
            tok = tokens_all.get('fast:%s' % b.name, {})
            row = {'status': res['status'] if res else 'NOT_RUN', 'alias_of': alias_of, 'committed': plan, 'committed_label': ES._label_of(b, c, plan) if plan else None,
                   'failure_kind': res['failure_kind'] if res else None, 'e_by_seed': e, 'none_e_by_seed': none_e, 'c_a_by_seed': _block_vec(b, c, plan, 'c_a') if plan else None,
                   'calls': res['calls'] if res else None, 'new_evaluations': res['new_evaluations'] if res else None,
                   'tokens': (tok.get('prompt_tokens', 0) + tok.get('completion_tokens', 0)) if tok else 0}
            if e is None or none_e is None or statistics.fmean(none_e) <= 0:
                incomplete.append('%s/%s' % (a, c))
                row['ratio'] = None
            else:
                row['ratio'] = statistics.fmean(e) / statistics.fmean(none_e)
                ratios.append(row['ratio'])
            runs[a][c] = row
        if len(ratios) == len(sel_cases):
            J[a] = statistics.fmean(ratios)
            cost[a] = (sum(runs[a][c]['new_evaluations'] or 0 for c in sel_cases), sum(runs[a][c]['tokens'] for c in sel_cases))
    rec = {'domain_id': domain, 'formula': 'J(W) = mean over V03/V04 of mean_seed E(actual commit of W) / mean_seed E(None); lower is better',
           'tie_rule_new': '<= 1e-12 -> fewer new evaluations, fewer tokens, then N1/N2/N3', 'arms': arms, 'aliases': select_aliases(root, domain), 'runs': runs, 'J': J,
           'incomplete': incomplete, 'block': 'e', 'select_cases': sel_cases, 'written_local': now()}
    new_arms = [a for a in arms if a != 'w_old' and a in J]
    if not new_arms:
        rec.update(w_new={'status': 'ALIAS_OLD', 'arm': None, 'alias_of': 'w_old', 'note': 'no distinct complete new card (all KEEP / alias / technical failure): W_new aliases W_old'})
    else:
        best = min(J[a] for a in new_arms)
        tied = [a for a in new_arms if J[a] - best <= TOL]
        winner = min(tied, key=lambda a: (cost[a][0], cost[a][1], arms.index(a)))
        rec.update(w_new={'status': 'SELECTED', 'arm': winner, 'alias_of': None, 'tied': tied, 'J': J[winner], 'margins': {a: J[a] - J[winner] for a in new_arms}})
    rec['w_old'] = {'J': J.get('w_old'), 'complete': 'w_old' in J}
    # Fixed_dev: uniform programs with E on BOTH cases over every branch of the domain's Select cases
    per_case = {}
    for c in sel_cases:
        pool = {}
        for b in sorted(p for p in P_['select'].glob('%s_*' % c) if p.is_dir() and not p.name.endswith('_common') and '__interrupted' not in p.name):
            for key, m in ES._uniform_materials_e(b, c).items():
                pool.setdefault(key, m)
        per_case[c] = pool
    common_keys = set.intersection(*(set(p) for p in per_case.values())) if per_case and all(per_case.values()) else set()
    fixed = {}
    for key in common_keys:
        ratios = []
        for c in sel_cases:
            m, none = per_case[c][key], per_case[c].get(ec._PKEY([]))
            if none is None or statistics.fmean(none['e_by_seed']) <= 0:
                ratios = None
                break
            ratios.append(statistics.fmean(m['e_by_seed']) / statistics.fmean(none['e_by_seed']))
        if ratios:
            m0 = per_case[sel_cases[0]][key]
            fixed[key] = {'label': m0['label'], 'steps': m0['steps'], 'public_id': m0['public_id'], 'J': statistics.fmean(ratios), 'ratios': ratios}
    if fixed:
        def tie(key):
            m = fixed[key]
            pub_rank = PUBLIC.index(m['public_id']) if m['public_id'] in PUBLIC else len(PUBLIC)
            tab = ec.program_table_index(m['steps']) if m['steps'] is not None else ('fixed', 0)
            return (m['J'], pub_rank, 0 if tab[0] == 'explicit' else 1, tab[1])
        kbest = min(fixed, key=tie)
        rec['fixed_dev'] = {**fixed[kbest], 'candidates_in_intersection': len(fixed), 'table': {v['label']: round(v['J'], 6) for v in sorted(fixed.values(), key=lambda v: v['J'])}}
    else:
        rec['fixed_dev'] = None
    # H_deploy among W_new / W_old / Fixed_dev by the same J; ties: fixed first, then fewer evaluations, fewer tokens
    cands = {}
    if rec['fixed_dev']:
        cands['fixed_dev'] = (rec['fixed_dev']['J'], 0, 0, 0)
    if 'w_old' in J:
        cands['w_old'] = (J['w_old'], 1, cost['w_old'][0], cost['w_old'][1])
    if rec['w_new']['status'] == 'SELECTED':
        a = rec['w_new']['arm']
        cands['w_new'] = (J[a], 1, cost[a][0], cost[a][1])
    if cands:
        best = min(v[0] for v in cands.values())
        tied = [k for k, v in cands.items() if v[0] - best <= TOL]
        choice = min(tied, key=lambda k: cands[k][1:])
        rec['h_deploy'] = {'choice': choice, 'J_table': {k: v[0] for k, v in cands.items()}, 'tied': tied, 'rule': 'lowest J; ties <= 1e-12: fixed program first, then fewer evaluations, fewer tokens',
                           'metatest_arm': {'fixed_dev': 'fixed_dev', 'w_old': 'f_old', 'w_new': 'f_new'}[choice], 'note': 'frozen adoption decision; MetaTest reuses the corresponding arm, no new call'}
    else:
        rec['h_deploy'] = None
    return rec


def freeze_domain(root: Path, domain: str) -> dict:
    P_ = paths(root)
    P_['freeze'].mkdir(parents=True, exist_ok=True)
    out = P_['freeze'] / ('%s.json' % domain)
    if out.exists():
        return context.read_json(out)
    sel = select_domain(root, domain)
    dsks.write_once(P_['slow'] / domain / 'selection.json', sel)
    oc = old_card(domain)
    cards = new_cards(root, domain)
    rec = {'domain_id': domain, 'frozen_local': now(), 'selection': {k: sel.get(k) for k in ('J', 'w_new', 'w_old', 'incomplete', 'select_cases', 'aliases', 'h_deploy')},
           'w_old': {'skill': oc['skill'].to_json(), 'injected_body': oc['injected_body'], 'J': sel['w_old']['J']}, 'fixed_dev': sel.get('fixed_dev'), 'h_deploy': sel.get('h_deploy')}
    if sel['w_new']['status'] == 'SELECTED':
        s = cards[sel['w_new']['arm']]
        frozen = replace(s, status='FROZEN_SELECTED', source_stage='freeze', derived_from=s.skill_id)
        rec['w_new'] = {'status': 'FROZEN_SELECTED', 'arm': sel['w_new']['arm'], 'skill': frozen.to_json(), 'injected_body': frozen.rendered_body, 'J': sel['w_new']['J'], 'alias_of': None}
    else:
        rec['w_new'] = {'status': 'ALIAS_OLD', 'arm': None, 'skill': None, 'injected_body': oc['injected_body'], 'J': sel['w_old']['J'], 'alias_of': 'w_old',
                        'note': sel['w_new'].get('note')}
    rec['candidates'] = {a: s.to_json() for a, s in cards.items()}
    rec['note'] = 'selected on the two new Select cases\' E; a selection, not a promotion to a verified Skill; W_new keeps its MetaTest arm whatever H_deploy chose'
    dsks.write_once(out, rec)
    return rec


# ============================================================================= MetaTest (Q03-Q06 x 2 domains): F0 / F_old / F_new / Random / Fixed_dev + publics
def metatest_stage(root: Path, driver_lock: PackageLock) -> dict:
    frozen = {d: freeze_domain(root, d) for d in DOMAINS}
    cases_all = cases_of(root)
    cases = [c for d in DOMAINS for c in ('%s_Q03' % d, '%s_Q04' % d, '%s_Q05' % d, '%s_Q06' % d)]
    knowledge, order, aliases, fixed = {}, {}, {}, {}
    for d in DOMAINS:
        fr = frozen[d]
        old = dsk.skill_from_json(fr['w_old']['skill'])
        if fr['w_new']['status'] == 'FROZEN_SELECTED':
            new = dsk.skill_from_json(fr['w_new']['skill'])
            if new.status != 'FROZEN_SELECTED' or new.domain_id != d:
                raise PermissionError('only the FROZEN_SELECTED card of its domain enters MetaTest')
            kn_new, alias = asdict(dsk.skill_knowledge(new)), None
        else:
            kn_new, alias = None, 'f_old'
        if fr.get('fixed_dev'):
            fixed[d] = {k: fr['fixed_dev'][k] for k in ('label', 'steps', 'public_id', 'J')}
        for c in ('%s_Q03' % d, '%s_Q04' % d, '%s_Q05' % d, '%s_Q06' % d):
            cs = cases_all[c]
            knowledge[c] = {'f0': asdict(br.Knowledge()), 'f_old': asdict(dsk.skill_knowledge(old)), 'f_new': kn_new, 'random': None, 'fixed_dev': None}
            order[c] = rotate(TEST_FAST_ARMS, cs.case_index) + list(CONTROL_ARMS)
            aliases[c] = {'f_new': alias} if alias else {}
    cat = paths(root)['freeze'] / 'metatest_catalog.json'
    if not cat.exists():
        dsks.write_once(cat, {'status': 'METATEST_FROZEN', 'epoch': time.time(), 'frozen_local': now(), 'w_new': {d: frozen[d]['w_new'] for d in DOMAINS},
                              'w_old': {d: frozen[d]['w_old']['skill']['skill_id'] for d in DOMAINS}, 'fixed_dev': fixed, 'h_deploy': {d: frozen[d].get('h_deploy') for d in DOMAINS},
                              'orders': order, 'aliases': aliases, 'frozen_before_first_metatest_fit': True})
    cfgp = stage_config(root, 'metatest', cases=cases, order=order, knowledge=br.json_copy(knowledge), labels_e=True, llm=True, random=True, aliases=aliases,
                        alias_notes={'f_new': 'W_new aliases W_old (KEEP / identical body): the F_old trajectory and scores are reused; no second LLM run'}, fixed_dev=fixed)
    return run_stage(root, 'metatest', cfgp, driver_lock)


# ============================================================================= readout (task §9)
def G(a_e, b_e, den) -> dict | None:
    """G(A over B) = 100 x [E(B) - E(A)] / E(None) per seed; positive = A better; den = the case three-seed None mean."""
    if a_e is None or b_e is None or not den:
        return None
    return ES._stats([100.0 * (y - x) / den for x, y in zip(a_e, b_e)])


def _research_stats(bdir: Path, case: str) -> dict:
    """Task §9 item 4 from the trajectory: built / evaluated / never-evaluated counts, inspection counts, and a heuristic flag for a built-but-
    never-evaluated plan whose id appears with a harm word in a later rationale / commit reason (for reading; not a verdict)."""
    rows = ES._trace(bdir)
    reg = ec.load_registry(bdir / case)
    plan_to_phys = {mid: r.get('phys_id', mid) for mid, r in reg.items()}
    points, summ = decision_points(rows, plan_to_phys, {}, 2)
    flags = []
    for p in summ['built_never_evaluated']:
        for dp in points:
            for h in dp['hypotheses_stated']:
                text = (h.get('reason') or h.get('rationale') or '')
                low = text.lower()
                if h.get('plan_id') != p and p.lower() in low and any(w in low for w in HARM_WORDS):
                    flags.append({'plan_id': p, 'label': summ['built_labels'].get(p), 'call': dp['call'], 'text': text[:300]})
    return {**{k: summ[k] for k in ('calls', 'tool_calls', 'new_evaluations', 'inspect_material_calls', 'inspect_data_calls')}, 'n_built': len(summ['built']), 'n_evaluated_new': len(summ['evaluated']),
            'built_never_evaluated': [{'plan_id': p, 'label': summ['built_labels'].get(p)} for p in summ['built_never_evaluated']],
            'possible_unverified_harm_claims': flags, 'evaluated_labels': {p: summ['built_labels'].get(p) for p in summ['evaluated']}}


def _arm_record(bdir: Path, case: str, *, with_research: bool) -> dict:
    rec = ES._arm_record(bdir, case)
    cells = ec.branch_cells(bdir / case)
    rec['physical_fits_this_branch'] = sum(int(c.get('fit_attempts_this_request') or 0) for c in cells.values() if c.get('cache_hit') is False)
    rec['cache_hits_this_branch'] = sum(1 for c in cells.values() if c.get('cache_hit'))
    tok = dsks._unit_tokens(bdir.parent).get('fast:%s' % bdir.name, {}) if (bdir.parent / 'raw_responses').exists() else {}
    rec['tokens'] = (tok.get('prompt_tokens', 0) + tok.get('completion_tokens', 0)) if tok else 0
    rec['requests'] = tok.get('requests', 0) if tok else 0
    if with_research and (bdir / 'trace.jsonl').exists():
        rec['research'] = _research_stats(bdir, case)
    return rec


def _post_hoc_pool(rec: dict, den, r_id: str = 'P_NoMixRecipe') -> dict | None:
    pool = rec.get('pool') or {}
    with_e = {m: v['e'] for m, v in pool.items() if v.get('e') is not None}
    if not with_e or not den or not rec.get('e_by_seed') or r_id not in with_e:
        return None
    best = min(with_e.values())
    available = 100.0 * (with_e[r_id] - best) / den
    selection_loss = 100.0 * (statistics.fmean(rec['e_by_seed']) - best) / den
    return {'reference': r_id, 'available_pp': available, 'selection_loss_pp': selection_loss, 'net_pp': available - selection_loss, 'pool_size': len(with_e),
            'pool_best_label': pool[min(with_e, key=with_e.get)]['label'], 'note': 'descriptive arithmetic on the branch\'s own evaluated pool; unfitted materials have no E'}


def readout(root: Path = ROOT) -> dict:
    P_ = paths(root)
    cases = cases_of(root)
    split = load_split(root)
    frozen = {d: (context.read_json(P_['freeze'] / ('%s.json' % d)) if (P_['freeze'] / ('%s.json' % d)).exists() else {'status': 'NOT_FROZEN'}) for d in DOMAINS}
    out = {'package': PACKAGE, 'exposure': EXPOSURE, 'written_local': now(), 'frozen': {}, 'slow': {}, 'select': {}, 'metatest': {}, 'summary': {}}
    for d in DOMAINS:
        fr = frozen[d]
        out['frozen'][d] = {'w_new': {k: (fr.get('w_new') or {}).get(k) for k in ('status', 'arm', 'J', 'alias_of', 'injected_body')}, 'w_old': {k: (fr.get('w_old') or {}).get(k) for k in ('J', 'injected_body')},
                            'fixed_dev': {k: v for k, v in (fr.get('fixed_dev') or {}).items() if k != 'table'} if fr.get('fixed_dev') else None, 'h_deploy': fr.get('h_deploy')}
    # ---- Slow
    sp = P_['slow'] / 'proposals.json'
    if sp.exists():
        s = context.read_json(sp)
        for d in DOMAINS:
            rows = []
            for r in s['domains'][d]['proposals']:
                row = dict(r)
                pp = P_['slow'] / d / ('proposal_N%d.json' % r['slot'])
                if pp.exists():
                    rec = context.read_json(pp)
                    row['meta'] = rec.get('meta')
                    row['body'] = (rec.get('skill') or {}).get('rendered_body')
                rows.append(row)
            out['slow'][d] = {'proposals': rows, 'n_distinct_new_cards': s['domains'][d]['n_distinct_new_cards'], 'census_tier': s['domains'][d]['census_tier'], 'census_bytes': s['domains'][d]['census_bytes']}
    # ---- Select
    for d in DOMAINS:
        p = P_['slow'] / d / 'selection.json'
        if p.exists():
            s = context.read_json(p)
            out['select'][d] = {'J': s['J'], 'w_new': s['w_new'], 'w_old': s['w_old'], 'h_deploy': s.get('h_deploy'), 'incomplete': s['incomplete'], 'aliases': s['aliases'],
                                'runs': {a: {c: {k: v.get(k) for k in ('status', 'alias_of', 'committed', 'committed_label', 'ratio', 'new_evaluations', 'calls', 'tokens')} for c, v in rr.items()} for a, rr in s['runs'].items()},
                                'fixed_dev': s.get('fixed_dev')}
    # ---- MetaTest
    arms = ('f0', 'f_old', 'f_new', 'random', 'fixed_dev')
    mt_cfg = context.read_json(P_['configs'] / 'metatest.json') if (P_['configs'] / 'metatest.json').exists() else {'aliases': {}}
    for c in case_ids(root, 'test'):
        cs = cases[c]
        recs = {}
        for a in arms:
            b, alias_of = _arm_dir(P_['metatest'], c, a, mt_cfg.get('aliases', {}))
            if b.exists():
                recs[a] = _arm_record(b, c, with_research=a in TEST_FAST_ARMS)
                if alias_of:
                    recs[a]['alias_of'] = alias_of
            else:
                recs[a] = {'status': 'NOT_RUN'}
        f0 = recs.get('f0', {})
        pub = {m: f0['pool'][m]['e_by_seed'] for m in PUBLIC if m in f0.get('pool', {}) and f0['pool'][m]['e_by_seed']}
        none_e = pub.get('None')
        den = statistics.fmean(none_e) if none_e else None
        e = {a: recs[a].get('e_by_seed') for a in arms}
        h = (frozen[cs.domain].get('h_deploy') or {}).get('metatest_arm')
        e['h_deploy'] = e.get(h) if h else None
        row = {'domain': cs.domain, 'anchor': 'A' if cs.t == split['anchors'][0] else 'B', 'none_e_by_seed': none_e, 'public_e_by_seed': pub, 'h_deploy_arm': h,
               'arms': {a: {k: v for k, v in r.items() if k != 'pool'} for a, r in recs.items()}, 'G': {}}
        row['G'] = {'F_new_over_F_old': G(e['f_new'], e['f_old'], den), 'F_new_over_F0': G(e['f_new'], e['f0'], den), 'F_new_over_Random': G(e['f_new'], e['random'], den),
                    'F_new_over_FixedDev': G(e['f_new'], e['fixed_dev'], den), 'F_new_over_NoMix': G(e['f_new'], pub.get('P_NoMixRecipe'), den), 'F_new_over_None': G(e['f_new'], none_e, den),
                    'F_old_over_F0': G(e['f_old'], e['f0'], den), 'F_old_over_NoMix': G(e['f_old'], pub.get('P_NoMixRecipe'), den), 'F0_over_NoMix': G(e['f0'], pub.get('P_NoMixRecipe'), den),
                    'F0_over_None': G(e['f0'], none_e, den), 'F_old_over_None': G(e['f_old'], none_e, den), 'Random_over_None': G(e['random'], none_e, den),
                    'FixedDev_over_None': G(e['fixed_dev'], none_e, den), 'NoMix_over_None': G(pub.get('P_NoMixRecipe'), none_e, den),
                    'H_deploy_over_F0': G(e['h_deploy'], e['f0'], den), 'H_deploy_over_F_old': G(e['h_deploy'], e['f_old'], den), 'H_deploy_over_F_new': G(e['h_deploy'], e['f_new'], den),
                    'H_deploy_over_FixedDev': G(e['h_deploy'], e['fixed_dev'], den), 'H_deploy_over_NoMix': G(e['h_deploy'], pub.get('P_NoMixRecipe'), den)}
        for a in ('f0', 'f_old', 'f_new', 'random'):
            r = recs.get(a, {})
            if r.get('pool') and den and r.get('e_by_seed'):
                best = r.get('e_best_oracle')
                row['arms'][a]['oracle_gap_pp'] = 100.0 * (statistics.fmean(r['e_by_seed']) - r['pool'][best]['e']) / den if best else None
                row['arms'][a]['oracle_label'] = r['pool'][best]['label'] if best else None
                row['arms'][a]['post_hoc_vs_nomix'] = _post_hoc_pool(r, den)
                row['arms'][a]['independent_deployment_fits'] = 12 + 3 * int(r.get('new_evaluations') or 0)
        row['e_ratio_to_none'] = {a: (statistics.fmean(e[a]) / den if (e.get(a) and den) else None) for a in list(arms) + ['h_deploy']}
        row['e_ratio_to_none'].update({m: (statistics.fmean(v) / den if den else None) for m, v in pub.items()})
        row['same_delivery'] = {'F_new_vs_F_old': (recs['f_new'].get('committed_label') == recs['f_old'].get('committed_label')) if recs['f_new'].get('committed_label') else None,
                                'F_new_vs_F0': (recs['f_new'].get('committed_label') == recs['f0'].get('committed_label')) if recs['f_new'].get('committed_label') else None}
        out['metatest'][c] = row
    # ---- aggregate: domain = equal-weight 4 cases; overall = equal-weight 2 domains; anchors stratified; win / tie / loss counts
    keys = list(next(iter(out['metatest'].values()))['G']) if out['metatest'] else []
    agg = {}
    for k in keys:
        dom_means = {}
        for d in DOMAINS:
            vals = [out['metatest'][c]['G'][k]['mean'] for c in out['metatest'] if out['metatest'][c]['domain'] == d and out['metatest'][c]['G'].get(k)]
            dom_means[d] = statistics.fmean(vals) if vals else None
        ok = [v for v in dom_means.values() if v is not None]
        by_anchor = {}
        for an in ('A', 'B'):
            vals = [out['metatest'][c]['G'][k]['mean'] for c in out['metatest'] if out['metatest'][c]['anchor'] == an and out['metatest'][c]['G'].get(k)]
            by_anchor[an] = statistics.fmean(vals) if vals else None
        per = {c: out['metatest'][c]['G'][k]['mean'] if out['metatest'][c]['G'].get(k) else None for c in out['metatest']}
        agg[k] = {'by_domain': dom_means, 'overall': statistics.fmean(ok) if len(ok) == len(DOMAINS) else None, 'by_anchor': by_anchor, 'per_case': per,
                  'wins': sum(1 for v in per.values() if v is not None and v > 1e-9), 'ties': sum(1 for v in per.values() if v is not None and abs(v) <= 1e-9),
                  'losses': sum(1 for v in per.values() if v is not None and v < -1e-9), 'n_cases': sum(1 for v in per.values() if v is not None)}
    research = {}
    for a in TEST_FAST_ARMS:
        rows = [out['metatest'][c]['arms'][a].get('research') for c in out['metatest'] if out['metatest'][c]['arms'].get(a, {}).get('research')]
        research[a] = {'cases': len(rows), 'calls': sum(r['calls'] for r in rows), 'tool_calls': sum(r['tool_calls'] for r in rows), 'new_evaluations': sum(r['new_evaluations'] for r in rows),
                       'built': sum(r['n_built'] for r in rows), 'built_never_evaluated': sum(len(r['built_never_evaluated']) for r in rows),
                       'inspect_material_calls': sum(r['inspect_material_calls'] for r in rows), 'inspect_data_calls': sum(r['inspect_data_calls'] for r in rows),
                       'possible_unverified_harm_claims': sum(len(r['possible_unverified_harm_claims']) for r in rows),
                       'commit_is_c_a_argmin': sum(1 for c in out['metatest'] if out['metatest'][c]['arms'].get(a, {}).get('commit_is_c_a_argmin')),
                       'tokens': sum(int(out['metatest'][c]['arms'][a].get('tokens') or 0) for c in out['metatest'] if not out['metatest'][c]['arms'][a].get('alias_of')),
                       'requests': sum(int(out['metatest'][c]['arms'][a].get('requests') or 0) for c in out['metatest'] if not out['metatest'][c]['arms'][a].get('alias_of')),
                       'physical_fits': sum(int(out['metatest'][c]['arms'][a].get('physical_fits_this_branch') or 0) for c in out['metatest'] if not out['metatest'][c]['arms'][a].get('alias_of')),
                       'independent_deployment_fits': sum(int(out['metatest'][c]['arms'][a].get('independent_deployment_fits') or 0) for c in out['metatest'])}
    out['summary'] = {'G_pp': agg, 'research': research,
                      'commit_vs_c_a_argmin': {c: {a: out['metatest'][c]['arms'][a].get('commit_is_c_a_argmin') for a in TEST_FAST_ARMS} for c in out['metatest']},
                      'committed_labels': {c: {a: out['metatest'][c]['arms'][a].get('committed_label') for a in arms} for c in out['metatest']},
                      'same_delivery': {c: out['metatest'][c]['same_delivery'] for c in out['metatest']},
                      'post_hoc_vs_nomix': {c: {a: out['metatest'][c]['arms'][a].get('post_hoc_vs_nomix') for a in ('f0', 'f_old', 'f_new', 'random')} for c in out['metatest']}}
    out['cost'] = cost_readout(root)
    context.write_json(root / 'result.json', out)
    (root / 'tables.md').write_text(tables(out), encoding='utf-8')
    return out


def cost_readout(root: Path) -> dict:
    P_ = paths(root)
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    stages = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    per_stage, names = {}, list(stages)
    for i, s in enumerate(names):
        snap = stages[s]['snapshot_at_start']
        nxt = stages[names[i + 1]]['snapshot_at_start'] if i + 1 < len(names) else {k: led.get(k, 0) for k in snap}
        per_stage[s] = {k: nxt.get(k, 0) - snap.get(k, 0) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'fit_wall_seconds')}
        sroot = P_.get(s)
        fin = sroot / 'execution_finished.json' if sroot and (sroot / 'execution_finished.json').exists() else None
        if fin:
            f = context.read_json(fin)
            per_stage[s]['wall_s'] = f['epoch'] - stages[s]['epoch_start']
            per_stage[s]['pools'] = f.get('pools')
    slow_tok = {}
    ev = [e for e in led.get('events', []) if e.get('kind') == 'llm_finished' and e.get('stage') == 'slow']
    slow_tok = {'requests': len(ev), 'tokens': sum((e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0) for e in ev)}
    return {'ledger': {k: v for k, v in led.items() if k != 'events'}, 'per_stage': per_stage, 'slow_calls': slow_tok,
            'paid_elapsed_s': (time.time() - led['paid_clock_started_epoch']) if led.get('paid_clock_started_epoch') else None,
            'numeric_seconds': numeric_seconds(root) if P_['ledger'].exists() else None}


def _f(x, nd=2):
    return '-' if x is None else ('%.*f' % (nd, x))


def tables(res: dict) -> str:
    L_ = ['# %s tables (%s)' % (PACKAGE, res['written_local']), '', 'Exposure: %s. Metric: E normalized MSE (T scaler), entity macro, seed mean. G(A over B) in pp of the case None mean; positive = A better.' % res['exposure'], '']
    cols = ('F_new_over_F_old', 'F_new_over_F0', 'F_new_over_Random', 'F_new_over_FixedDev', 'F_new_over_NoMix', 'F_new_over_None', 'F_old_over_F0', 'F0_over_NoMix')
    L_ += ['## MetaTest: G per case (pp; SE df=2 in parentheses)', '', '| case | dom | anchor | ' + ' | '.join(cols) + ' |', '|---|---|---|' + '---:|' * len(cols)]
    for c, r in res['metatest'].items():
        g = r['G']
        L_.append('| %s | %s | %s | %s |' % (c, r['domain'], r['anchor'], ' | '.join((_f(g[k]['mean']) + (' (%s)' % _f(g[k]['se']) if g[k]['se'] is not None else '')) if g.get(k) else '-' for k in cols)))
    agg = res['summary'].get('G_pp', {})
    L_ += ['', '| aggregate | D01 | D02 | overall | anchor A | anchor B | win/tie/loss |', '|---|---:|---:|---:|---:|---:|---|']
    for k, v in agg.items():
        L_.append('| %s | %s | %s | %s | %s | %s | %d/%d/%d of %d |' % (k, _f(v['by_domain'].get('D01')), _f(v['by_domain'].get('D02')), _f(v['overall']), _f(v['by_anchor'].get('A')), _f(v['by_anchor'].get('B')),
                                                                       v['wins'], v['ties'], v['losses'], v['n_cases']))
    L_ += ['', '## MetaTest: E / None per arm', '', '| case | F0 | F_old | F_new | Random | Fixed_dev | H_deploy | NoMix | FixedMixup | AmpResample |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c, r in res['metatest'].items():
        rr = r['e_ratio_to_none']
        L_.append('| %s | %s |' % (c, ' | '.join(_f(rr.get(k), 4) for k in ('f0', 'f_old', 'f_new', 'random', 'fixed_dev', 'h_deploy', 'P_NoMixRecipe', 'FixedMixup', 'P_AmpResample'))))
    L_ += ['', '## MetaTest: commits and research process', '', '| case | arm | status | committed | C_A argmin? | oracle gap pp | calls | tools | built | new evals | never evaluated | inspect_material | inspect_data | harm-claim flags | tokens |',
           '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c, r in res['metatest'].items():
        for a, v in r['arms'].items():
            rs = v.get('research') or {}
            L_.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                c, ARM_LABEL.get(a, a) + (' (alias %s)' % ARM_LABEL.get(v['alias_of'], v['alias_of']) if v.get('alias_of') else ''), v.get('status'), v.get('committed_label'), v.get('commit_is_c_a_argmin'),
                _f(v.get('oracle_gap_pp')), v.get('calls'), rs.get('tool_calls', v.get('tool_calls')), rs.get('n_built', '-'), v.get('new_evaluations'), len(rs['built_never_evaluated']) if rs else '-',
                rs.get('inspect_material_calls', '-'), rs.get('inspect_data_calls', '-'), len(rs['possible_unverified_harm_claims']) if rs else '-', v.get('tokens')))
    L_ += ['', '## MetaTest: post-hoc pool arithmetic (reference NoMix; pp of None)', '', '| case | arm | available | selection loss | net | pool | pool best |', '|---|---|---:|---:|---:|---:|---|']
    for c, r in res['metatest'].items():
        for a in ('f0', 'f_old', 'f_new', 'random'):
            ph = r['arms'].get(a, {}).get('post_hoc_vs_nomix')
            if ph:
                L_.append('| %s | %s | %s | %s | %s | %d | %s |' % (c, ARM_LABEL[a], _f(ph['available_pp']), _f(ph['selection_loss_pp']), _f(ph['net_pp']), ph['pool_size'], ph['pool_best_label']))
    L_ += ['', '## Select and freeze', '']
    for d, s in res['select'].items():
        L_.append('- %s: J %s; W_new %s; W_old J %s; H_deploy %s; Fixed_dev %s (J %s)' % (d, {k: round(v, 4) for k, v in (s.get('J') or {}).items()}, (s.get('w_new') or {}).get('arm') or (s.get('w_new') or {}).get('status'),
                                                                                       _f((s.get('w_old') or {}).get('J'), 4), (s.get('h_deploy') or {}).get('choice'), (s.get('fixed_dev') or {}).get('label'), _f((s.get('fixed_dev') or {}).get('J'), 4)))
        for a, rr in (s.get('runs') or {}).items():
            L_.append('  - %s: ' % ARM_LABEL.get(a, a) + '; '.join('%s -> %s (E/None %s, evals %s)' % (c, v.get('committed_label'), _f(v.get('ratio'), 4), v.get('new_evaluations')) for c, v in rr.items()))
    L_ += ['', '## Slow proposals', '']
    for d, s in res['slow'].items():
        for r in s['proposals']:
            L_.append('- %s %s: %s%s; mode %s; tokens %s' % (d, r['arm'].upper(), r['status'], (' -> alias %s' % r['alias_of']) if r.get('alias_of') else '', r.get('research_mode'), r.get('tokens')))
    cost = res.get('cost', {})
    L_ += ['', '## Cost', '', '```', json.dumps({k: v for k, v in cost.get('ledger', {}).items() if k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts',
                                                                                                          'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'llm_failed_attempts', 'fit_wall_seconds', 'label_wall_seconds', 'material_wall_seconds')}, indent=1),
           json.dumps(cost.get('per_stage', {}), indent=1, default=str), json.dumps({'slow_calls': cost.get('slow_calls'), 'paid_elapsed_s': cost.get('paid_elapsed_s')}, indent=1), '```']
    return '\n'.join(L_) + '\n'


# ============================================================================= package driver
def accept_unknown_usage(root: Path, where: str) -> None:
    """Operator decision (task §7.2): the failed attempts recorded so far are accepted as unknown usage; their reserved upper bounds stay in the ledger."""
    P_ = paths(root)
    led = SafeLedger(P_['ledger'], **TOTAL)
    if rt.unknown_usage_blocks(led):
        led.s['unknown_usage_accepted'] = led.s['llm_tokens_unknown']
        led.event(kind='unknown_usage_accepted', accepted=led.s['llm_tokens_unknown'], where=where, operator='explicit --accept-unknown-usage')
        print('UNKNOWN_USAGE_ACCEPTED', led.s['llm_tokens_unknown'], where, flush=True)


def package_run(root: Path = ROOT, *, accept_unknown: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    lock = PackageLock(root, 'driver').acquire()
    context.write_json(root / 'driver_launch.json', {'pid': os.getpid(), 'epoch': time.time(), 'local': now()})
    try:
        if accept_unknown:
            accept_unknown_usage(root, 'driver_start')
        status = context.read_json(root / 'package_status.json') if (root / 'package_status.json').exists() else {}

        def stop(where, st):
            status.update({'stopped_at': where, 'stage_status': st, 'epoch': time.time(), 'local': now()})
            context.write_json(root / 'package_status.json', status)
            print('PACKAGE_STOP', where, st.get('status') if isinstance(st, dict) else st, flush=True)
            return status
        preflight(root)
        w = wiring(root)
        status['wiring'] = w['status']
        if w['status'] != 'PASS':
            return stop('wiring', w)
        sl = slow_stage(root, lock)
        status['slow'] = {d: [(r['arm'], r['status'], r['alias_of']) for r in sl['domains'][d]['proposals']] for d in DOMAINS}
        context.write_json(root / 'package_status.json', status)
        st = select_stage(root, lock)
        status['select'] = st
        if st['status'] != 'FINISHED':
            return stop('select', st)
        for d in DOMAINS:
            fr = freeze_domain(root, d)
            status['frozen_%s' % d] = {'w_new': fr['w_new']['status'], 'h_deploy': (fr.get('h_deploy') or {}).get('choice')}
        context.write_json(root / 'package_status.json', status)
        st = metatest_stage(root, lock)
        status['metatest'] = st
        if st['status'] != 'FINISHED':
            return stop('metatest', st)
        status['finished_epoch'] = time.time()
        status['finished_local'] = now()
        context.write_json(root / 'package_status.json', status)
        readout(root)
        print('PACKAGE_FINISHED', flush=True)
        return status
    finally:
        lock.release()


# ============================================================================= preflight (0 cost) and frozen configuration
def effective_concurrency(root: Path) -> dict:
    p = root / 'concurrency_effective.json'
    if p.exists():
        return context.read_json(p)['concurrency']
    return dict(CONCURRENCY)


def preflight(root: Path = ROOT) -> dict:
    """Frozen before the first fit: verbatim partition copies (this package + parent), the binding of the document paths to the numerical files,
    the partition check, cases, environment, prompts, contracts, orders, selection / readout formulas, budgets, concurrency. Reads no data row."""
    root.mkdir(parents=True, exist_ok=True)
    P_ = paths(root)
    if not P_['split'].exists():
        shutil.copy2(SPLIT_DOC, P_['split'])
    if not P_['parent_split'].exists():
        shutil.copy2(PARENT_SPLIT_DOC, P_['parent_split'])
    split, parent = context.read_json(P_['split']), context.read_json(P_['parent_split'])
    if split != context.read_json(SPLIT_DOC) or parent != context.read_json(PARENT_SPLIT_DOC):
        raise RuntimeError('a frozen split copy differs from its document')
    bound = bound_split(split)
    if not (root / 'split_bound.json').exists():
        context.write_json(root / 'split_bound.json', bound)
    elif context.read_json(root / 'split_bound.json') != bound:
        raise RuntimeError('split_bound.json drifted')
    p = root / 'frozen_config.json'
    if p.exists():
        return context.read_json(p)
    chk = check_split_dp(split, parent)
    if not chk['ok'] or set(split['domains']) != set(DOMAINS):
        raise RuntimeError('partition check failed: %s' % chk)
    binding = {d: resolve_csv_path(D['dataset'], D.get('csv_path') or D.get('path')) for d, D in split['domains'].items()}
    cases = ec.cases_from_split(bound)
    for cs in cases.values():
        if not cs.path().exists():
            raise RuntimeError('data file missing for %s: %s' % (cs.case_id, cs.path()))
    pc = ec.cases_from_split(parent)
    for c in WIRING_CASES:
        if c not in pc:
            raise RuntimeError('wiring case %s missing from the parent split' % c)
    cards = {d: old_card(d) for d in DOMAINS}
    cfg = {'package': PACKAGE, 'task': TASK, 'parent_package': PARENT_PACKAGE, 'parent_root': str(PARENT_ROOT), 'split_document': str(SPLIT_DOC), 'split_setting_id': split['setting_id'],
           'parent_split_document': str(PARENT_SPLIT_DOC), 'split_check': chk, 'path_binding': binding, 'exposure': EXPOSURE,
           'domains': {d: {'dataset': split['domains'][d]['dataset'], 'file': binding[d]['name'], 'columns_in_file': split['domains'][d]['columns_in_file'], 'partition_seed': split['domains'][d]['partition_seed'],
                           'shuffle_slice': split['domains'][d].get('shuffle_slice')} for d in DOMAINS},
           'cases': {c: cs.to_json() for c, cs in cases.items()}, 'learning_material': {d: ['%s_S%02d' % (d, i) for i in range(1, 9)] + ['%s_V01' % d, '%s_V02' % d, '%s_Q01' % d, '%s_Q02' % d] for d in DOMAINS},
           'cohort_size': ec.COHORT_SIZE, 'geometry': {'L': spec.L, 'H': spec.H, 'T': spec.TRAIN_SPAN, 'n_parents': spec.N_PARENTS, 'pool': ec.COHORT_SIZE * spec.N_PARENTS, 'anchors': split['anchors']},
           'consumer': ec.CONSUMER, 'seeds': list(SEEDS), 'model': MODEL, 'environment': ES.environment(), 'limits': LIMITS, 'max_output_tokens': MAX_OUTPUT_TOKENS,
           'max_tool_corrections': MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'request_trace_dedupe': REQUEST_TRACE_DEDUPE,
           'fast_token_cap': 'none (cumulative monitoring at %s)' % list(FAST_TOKEN_MONITOR), 'total': TOTAL, 'http_cap': HTTP_CAP, 'transport_retries': TRANSPORT_RETRIES, 'allocation': ALLOC,
           'token_warnings': list(TOKEN_WARNINGS), 'paid_wall_warnings_s': list(PAID_WALL_WARNINGS_S), 'concurrency_planned': CONCURRENCY, 'concurrency_effective': dict(CONCURRENCY),
           'fast_system': FAST_SYSTEM, 'material_semantics_sentence': MATERIAL_SEMANTICS, 'slow_system': SLOW_DP, 'contracts': CONTRACTS,
           'old_cards': {d: {'skill_id': cards[d]['skill_id'], 'injected_body': cards[d]['injected_body']} for d in DOMAINS},
           'public_references': {m: ES.public_spec(m) for m in PUBLIC}, 'program_tables': {'explicit': [ec.program_label(p_) for p_ in ec.PROGRAMS], 'edit': [ec.program_label(p_) for p_ in ec.EDIT_PROGRAMS]},
           'seed_rule': 'SeedSequence([%d, domain_index, case_index, program_index, int(entity_column)]); preset and every edit use program_index %d' % (ec.SEED_ROOT, ec.PRESET_INDEX),
           'random_search': {'seed_root': RANDOM_SEED_ROOT, 'max_tries': ES.RANDOM_MAX_TRIES, 'min_group': ES.RANDOM_MIN_GROUP, 'quantile': 0.5, 'programs_pool': len(ec.LEGAL_PROGRAMS),
                             'commit': 'argmin three-seed C_A mean over public + R1..R4; ties in public order then R1..R4', 'frozen': 'before the case\'s first new fit; never shown to Fast'},
           'orders': {'select': 'rotate(%s, case_index)' % list(SELECT_ARMS), 'metatest': 'rotate(%s, case_index) + %s' % (list(TEST_FAST_ARMS), list(CONTROL_ARMS)),
                      'case_queue': 'select: V03/V04 of D01 then D02; metatest: Q03..Q06 of D01 then D02; a free slot takes the next case'},
           'slow': {'proposals_per_domain': 3, 'same_census': True, 'independent': 'identical payload but proposal_slot; the others never visible', 'contract_correction_per_proposal': 1,
                    'alias_rule': 'KEEP -> alias W_old; byte-identical rendered body -> alias', 'census_byte_target': CENSUS_BYTE_TARGET, 'body_limit': BODY_LIMIT},
           'selection': {'J': 'J(W) = mean over V03/V04 of [mean_seed E(actual commit of W) / mean_seed E(None)]; lowest wins',
                         'w_new': 'lowest J among the distinct new cards; ties <= 1e-12 by fewer new evaluations, fewer tokens, N1/N2/N3; all KEEP -> alias W_old',
                         'fixed_dev': 'among uniform programs actually evaluated on BOTH Select cases (any branch, incl. public), the lowest J; ties in public order then table order',
                         'h_deploy': 'lowest J among W_new / W_old / Fixed_dev; ties: fixed program first, then fewer evaluations, fewer tokens; MetaTest reuses the corresponding arm'},
           'readout': {'metric': 'E normalized MSE (T scaler), entity macro, then seed mean; domain = equal-weight 4 cases; overall = equal-weight 2 domains; anchors stratified',
                       'G': 'G_j(A over B) = 100 x [mean_seed E(B) - mean_seed E(A)] / mean_seed E(None) (paired per seed with the same denominator); positive = A better; pp of None',
                       'post_hoc': 'available = 100 x [E(NoMix) - min_pool E] / E(None); selection_loss = 100 x [E(commit) - min_pool E] / E(None); net = available - selection_loss'},
           'frozen_local': now(), 'no_sha': True}
    dsks.write_once(p, cfg)
    return cfg


# ============================================================================= wiring acceptance (old D01_S01 / D02_S01, None x 3 seeds each, in parallel; <= 6 fits, 0 LLM)
def _sample_memory(stop: threading.Event, out: dict) -> None:
    import psutil
    me = psutil.Process(os.getpid())
    out.update({'peak_worker_rss_mb': 0.0, 'peak_workers_total_rss_mb': 0.0, 'peak_n_workers': 0, 'available_mb_min': None, 'samples': 0})
    while not stop.is_set():
        try:
            kids = [c for c in me.children(recursive=True) if 'python' in c.name().lower()]
            rss = [c.memory_info().rss / 1e6 for c in kids]
            out['peak_worker_rss_mb'] = max([out['peak_worker_rss_mb']] + rss)
            out['peak_workers_total_rss_mb'] = max(out['peak_workers_total_rss_mb'], sum(rss))
            out['peak_n_workers'] = max(out['peak_n_workers'], len(rss))
            av = psutil.virtual_memory().available / 1e6
            out['available_mb_min'] = av if out['available_mb_min'] is None else min(out['available_mb_min'], av)
            out['samples'] += 1
        except Exception:  # noqa: BLE001
            pass
        stop.wait(0.5)


def wiring(root: Path = ROOT) -> dict:
    P_ = paths(root)
    out = P_['wiring'] / 'wiring_result.json'
    if out.exists():
        return context.read_json(out)
    import psutil
    preflight(root)
    sc = stage_caps(root, 'wiring')
    led = SafeLedger(P_['ledger'], **sc['caps'])
    rt.FIT_RETRY = True
    set_pools(CONCURRENCY['numeric'], CONCURRENCY['http'])
    P_['wiring'].mkdir(parents=True, exist_ok=True)
    (P_['wiring'] / 'logs').mkdir(exist_ok=True)
    pcases = parent_cases_of(root)
    res = {'status': 'PASS', 'started_local': now(), 'cases': {}, 'checks': [], 'available_mb_before': psutil.virtual_memory().available / 1e6}
    mem, stop = {}, threading.Event()
    sampler = threading.Thread(target=_sample_memory, args=(stop, mem), daemon=True)
    sampler.start()
    t_all = time.time()

    def one(case):
        cs = pcases[case]
        common = P_['wiring'] / (case + '_common')
        cfg = {'split_path': str(P_['parent_split']), 'seeds': list(SEEDS), 'public_ids': ['None'], 'random': False, 'package_root': str(root)}
        prepare_common(P_['wiring'], case, led, cfg)                      # None: 3 fits (no material)
        reg = ec.load_registry(common / case)
        checks = [{'case': case, 'check': 'public materials built without fits', 'ok': all(m in reg for m in PUBLIC), 'materials': sorted(reg)}]
        with np.load(common / case / 'scaler.npz') as z:
            checks.append({'case': case, 'check': 'scaler bound to the case', 'ok': [str(x) for x in z['roster']] == list(cs.roster) and int(z['t']) == cs.t and str(z['dataset']) == cs.dataset})
        spec_rec = context.read_json(common / case / 'case_spec.json')
        checks.append({'case': case, 'check': 'only the roster columns were converted', 'ok': spec_rec['columns_converted'] == ec.COHORT_SIZE and spec_rec['columns_in_file'] > ec.COHORT_SIZE})
        # the parent's physical None cells of the same case (Source cache): C_A must reproduce exactly under the concurrent scheduler
        mine, theirs, eq = {}, {}, []
        for s in SEEDS:
            cid = ec.cell_id(case, 'None', s)
            a = context.read_json(common / case / 'cells' / (cid + '.json'))
            pb = PARENT_ROOT / 'source' / (case + '_common') / case / 'cells' / (cid + '.json')
            b = context.read_json(pb) if pb.exists() else None
            mine[s] = a['scores']['c_a']['normalized_mse_macro']
            theirs[s] = b['scores']['c_a']['normalized_mse_macro'] if b else None
            eq.append(b is not None and mine[s] == theirs[s] and a['pool_size'] == b['pool_size'] == ec.COHORT_SIZE * spec.N_PARENTS and a['torch_threads'] == b['torch_threads'] == FIT_THREADS)
        checks.append({'case': case, 'check': 'None x 3 seeds: C_A identical to the parent cache (same pool, same threads)', 'ok': all(eq), 'mine': mine, 'parent': theirs})
        # non-empty composition material: built and compared with the parent's wiring material (no fit)
        table = context.read_json(common / case / 'overview.json')['entities']
        comp = ec.compile_plan_full(ec.uniform_policy(WIRING_COMPOSITION, 'wiring: explicit 3-step composition'), table)
        m = ensure_material(common, P_['parent_split'], case, comp['assignment'], ledger=led)
        pm = PARENT_ROOT / 'wiring' / (case + '_common') / case / 'aug_materials'
        preq = context.read_json(pm / 'TA001__request.json') if (pm / 'TA001__request.json').exists() else None
        same_asg = preq is not None and preq['assignment'] == comp['assignment']
        bytes_eq = False
        if same_asg and (pm / 'TA001.npz').exists():
            with np.load(m['path']) as z1, np.load(pm / 'TA001.npz') as z2:
                bytes_eq = np.array_equal(z1['Xc'], z2['Xc']) and np.array_equal(z1['yc'], z2['yc'])
        summ = context.read_json(m['summary_path'])
        checks.append({'case': case, 'check': 'explicit composition material rebuilt bitwise equal to the parent wiring material (0 fits)', 'ok': bool(bytes_eq), 'same_assignment': same_asg,
                       'build_seconds': summ.get('build_seconds'), 'steps': {k: v['windows'] for k, v in summ['per_step'].items() if v['windows']}})
        res['cases'][case] = {'none_c_a_by_seed': mine, 'parent_c_a_by_seed': theirs, 'fit_seconds': [round(context.read_json(common / case / 'cells' / (ec.cell_id(case, 'None', s) + '.json'))['train']['seconds'], 1) for s in SEEDS]}
        return checks
    errs, results = [], {}

    def run(case):
        try:
            results[case] = one(case)
        except Exception as exc:  # noqa: BLE001
            errs.append((case, exc))
    ts = [threading.Thread(target=run, args=(c,), daemon=True) for c in WIRING_CASES]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    stop.set()
    sampler.join(timeout=5)
    for c in WIRING_CASES:
        res['checks'] += results.get(c, [{'case': c, 'check': 'case ran', 'ok': False, 'error': str(errs)[:300]}])
    res['wall_seconds_both_cases'] = time.time() - t_all
    res['fits_total'] = led.s['fit_attempts']
    res['fit_wall_seconds_sum'] = led.s['fit_wall_seconds']
    res['pools'] = pools().snapshot()
    res['memory'] = mem
    res['checks'].append({'check': 'exactly 6 physical fits, all OK, numeric pool actually used in parallel (peak >= 2)', 'ok': led.s['fit_attempts'] == 6 and led.s['fits_ok'] == 6 and res['pools']['numeric_peak'] >= 2})
    no_labels = not any(p_.name in ('c_b_scores.json', 'e_frozen.json', 'e_scores.json') for p_ in P_['wiring'].rglob('*.json'))
    res['checks'].append({'check': 'no label file in wiring', 'ok': no_labels})
    # concurrency decision from the measured worker memory (task §7.1): keep 3 unless three peak workers would not fit into the available memory with a 1.5 GB margin
    peak = mem.get('peak_worker_rss_mb') or 0.0
    avail = res['available_mb_before']
    numeric = CONCURRENCY['numeric'] if 3 * peak + 1500 <= avail else 2
    decision = {'concurrency': {**CONCURRENCY, 'numeric': numeric}, 'peak_worker_rss_mb': peak, 'peak_workers_total_rss_mb': mem.get('peak_workers_total_rss_mb'), 'available_mb_before': avail,
                'available_mb_min_during': mem.get('available_mb_min'), 'rule': '3 x peak worker RSS + 1500 MB margin <= available before start, else 2', 'decided_local': now(),
                'note': 'training threads / seeds / batch / budgets unchanged; the numeric pool only limits how many worker processes run at once'}
    context.write_json(root / 'concurrency_effective.json', decision)
    res['concurrency_decision'] = decision
    res['status'] = 'PASS' if all(c['ok'] for c in res['checks']) and led.s['fit_attempts'] <= ALLOC['wiring']['fits'] else 'FAIL'
    res['finished_local'] = now()
    dsks.write_once(out, res)
    print('WIRING', res['status'], 'fits', res['fits_total'], 'numeric', numeric, flush=True)
    return res


# ============================================================================= smoke (synthetic file + synthetic split; 0 real fits, 0 API; concurrency, receipts, lock, resume, dedupe, census, parser)
class ScriptedClient(ES.ScriptedClient):
    """The Fast tool loop driven by a fixed action list; ledger-free. Optional: capture the request payloads; raise a fault at a given call."""

    def __init__(self, steps: list, *, fail_at: int | None = None, capture: list | None = None):
        super().__init__(steps)
        self.fail_at, self.capture, self.n = fail_at, capture, 0

    def fast(self, unit):
        def call(payload):
            self.n += 1
            if self.capture is not None:
                self.capture.append(copy.deepcopy(payload))
            if self.fail_at is not None and self.n == self.fail_at:
                raise rt.llm.TransportFault('SMOKE_FAULT_NEVER_SENT')
            if not self.steps:
                raise RuntimeError('scripted client exhausted')
            return self.steps.pop(0)
        return call


class _FakeUsage:
    def __init__(self, pt, ct):
        self.prompt_tokens, self.completion_tokens = pt, ct


class _FakeResponse:
    def __init__(self, text, pt, ct, model=MODEL['returned_required']):
        self.model, self.usage = model, _FakeUsage(pt, ct)
        self.choices = [type('C', (), {'message': type('M', (), {'content': text})()})()]

    def model_dump(self, mode='json'):
        return {'model': self.model, 'usage': {'prompt_tokens': self.usage.prompt_tokens, 'completion_tokens': self.usage.completion_tokens}, 'choices': [{'message': {'content': self.choices[0].message.content}}]}


class FakeMeteredClient(MeteredClient):
    """MeteredClient without a network: no API key needed, never falls back to the real transport; the transport is a deterministic function of the payload."""

    def __init__(self, ledger, out, *, http_cap, stage, latency=0.05, fail_requests=()):
        self.api = None
        self.ledger, self.out, self.stage = ledger, Path(out), stage
        self.out.mkdir(parents=True, exist_ok=True)
        self.fatal, self.http_cap, self.unit_tokens = False, int(http_cap), {}
        self.latency, self.fail_requests, self.sent = latency, set(fail_requests), []

    def _transport(self, messages, max_tokens, timeout):
        time.sleep(self.latency)
        n = len(self.sent) + 1
        self.sent.append(n)
        if n in self.fail_requests:
            raise ConnectionError('smoke transient 502 bad gateway')
        pt = len(messages[1]['content']) // 4
        return _FakeResponse(json.dumps({'echo': json.loads(messages[1]['content']).get('k')}), pt, 7)


def _synthetic_splits(out: Path, n_cols: int = 140, hours: int = 1200) -> tuple:
    """(this package's synthetic split, a synthetic parent split): 2 Select + 4 MetaTest groups with case_index 12..17 disjoint from a 2-group parent."""
    import datetime as _dt
    rng = np.random.RandomState(7)
    t0 = _dt.datetime(2016, 7, 1, 2)
    csvp = out / 'synthetic.csv'
    with csvp.open('w', newline='') as f:
        f.write('date,' + ','.join(str(i) for i in range(n_cols)) + ',OT\n')
        base_ = rng.rand(n_cols) * 5 + 1
        for h in range(hours):
            ts = t0 + _dt.timedelta(hours=h)
            vals = base_ * (1 + 0.3 * np.sin(2 * np.pi * (h % 24) / 24 + np.arange(n_cols))) + 0.1 * rng.randn(n_cols)
            f.write(ts.strftime('%Y-%m-%d %H:%M:%S') + ',' + ','.join('%.4f' % v for v in vals) + ',%.3f\n' % rng.rand())
    cols = [str(i) for i in range(n_cols)]
    t = 700

    def group(cid, role, idx, roster):
        cs = ec.CaseSpec(dataset='synthetic', domain='SM', case_id=cid, role=role, t=t, roster=tuple(roster), case_index=idx, csv_path=str(csvp), total_hours=hours)
        return {'case_id': cid, 'stage': role, 't': t, 'roster': list(cs.roster), 'case_index': idx, 'train_rows': list(cs.train_range), 'c_a_origins': list(cs.c_a), 'c_b_origins': list(cs.c_b), 'e_origins': list(cs.e)}
    parent_groups = [group('SM_S01', 'source', 0, cols[0:16]), group('SM_S02', 'source', 1, cols[16:32])]
    new_groups = [group('SM_V03', 'select', 12, cols[32:48]), group('SM_V04', 'select', 13, cols[48:64])] + [group('SM_Q%02d' % (3 + i), 'test', 14 + i, cols[64 + 16 * i: 80 + 16 * i]) for i in range(4)]
    dom = {'dataset': 'synthetic', 'csv_path': str(csvp), 'total_hours': hours, 'columns_in_file': n_cols + 1, 'excluded_columns': ['OT'], 'partition_seed': 0}
    parent = {'setting_id': 'SMOKE_PARENT', 'anchors': [t], 'cohort_size': 16, 'domains': {'SM': {**dom, 'groups': parent_groups}}}
    split = {'setting_id': 'SMOKE', 'anchors': [t], 'cohort_size': 16, 'domains': {'SM': {**dom, 'groups': new_groups}}}
    context.write_json(out / 'split.json', split)
    context.write_json(out / 'parent_split.json', parent)
    return out / 'split.json', out / 'parent_split.json'


def smoke(root: Path = ROOT) -> dict:
    out = paths(root)['smoke']
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    checks = []

    def check(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})
        print('SMOKE', 'ok ' if ok else 'FAIL', name, flush=True)
    set_pools(3, 2)
    # 1 partition document of this package against the parent document (no data row read)
    split, parent = context.read_json(SPLIT_DOC), context.read_json(PARENT_SPLIT_DOC)
    chk = check_split_dp(split, parent)
    check('document partition: 2 x (2 Select + 4 MetaTest) groups of 16, case_index 12..17, disjoint from every parent group, OT unused, same anchors', chk['ok'], detail={d: {k: v for k, v in r.items() if k != 'overlap_with_parent'} for d, r in chk['domains'].items()})
    bad = copy.deepcopy(split)
    bad['domains']['D01']['groups'][0]['roster'][0] = parent['domains']['D01']['groups'][0]['roster'][0]
    check('a roster column shared with the parent is detected', not check_split_dp(bad, parent)['ok'])
    check('csv path binding: the /mnt/c document path resolves to the spec file of each domain', all(resolve_csv_path(split['domains'][d]['dataset'], split['domains'][d]['csv_path'])['identical_file'] for d in DOMAINS))
    ok = True
    try:
        resolve_csv_path('electricity', '/mnt/c/does/not/exist.csv')
        ok = False
    except RuntimeError:
        pass
    check('an unresolvable document path is refused', ok)
    # 2 package lock: a second controller (subprocess) is refused while held; free after release
    lk = PackageLock(out, 'smoke').acquire()
    rc_held = subprocess.run([sys.executable, '-B', '-m', MODULE, '--lock-probe', str(out)], cwd=str(REPO), capture_output=True, env=worker_env()).returncode
    lk.release()
    rc_free = subprocess.run([sys.executable, '-B', '-m', MODULE, '--lock-probe', str(out)], cwd=str(REPO), capture_output=True, env=worker_env()).returncode
    check('package lock: second controller refused (3) while held, admitted (0) after release', rc_held == 3 and rc_free == 0, rc_held=rc_held, rc_free=rc_free)
    # 3 exclusive-create receipts
    write_new(out / 'once.json', {'a': 1})
    ok = False
    try:
        write_new(out / 'once.json', {'a': 2})
    except FileExistsError:
        ok = context.read_json(out / 'once.json') == {'a': 1}
    check('numbered receipt refuses overwrite (exclusive create)', ok)
    # 4 thread-safe ledger + fake metered client: unique contiguous request numbers, per-request receipts, usage attribution, HTTP pool cap, transient fault -> unknown usage
    led = SafeLedger(out / 'ledger_llm.json', max_fit_attempts=0, max_llm_requests=100, max_llm_tokens=10_000_000, max_wall_s=3600, max_retries=0)
    fc = FakeMeteredClient(led, out / 'raw', http_cap=100, stage='smoke')
    fast = FastClient(fc)
    got, errs = [], []

    def worker(case, arm):
        f = fast.fast('%s_%s' % (case, arm), case=case, arm=arm)
        for k in range(5):
            try:
                got.append((case, arm, f({'k': '%s/%s/%d' % (case, arm, k), 'pad': 'x' * 400})))
            except Exception as exc:  # noqa: BLE001
                errs.append((case, arm, k, type(exc).__name__))
    ts = [threading.Thread(target=worker, args=('SM_Q%02d' % (3 + i), 'f0')) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    reqs = sorted(int(p.name[:3]) for p in (out / 'raw').glob('*_request.json'))
    resps = sorted(int(p.name[:3]) for p in (out / 'raw').glob('*_response.json'))
    metas = [context.read_json(p) for p in (out / 'raw').glob('*_request.json')]
    tok_sum = sum(v['prompt_tokens'] + v['completion_tokens'] for v in fc.unit_tokens.values())
    check('20 concurrent Fast calls over 4 cases: request numbers 1..20 unique, every request answered, receipts carry case / arm / call, ledger tokens == sum of receipts, HTTP peak <= 2',
          reqs == list(range(1, 21)) and resps == reqs and all(m.get('case') and m.get('arm') and m.get('call') for m in metas) and led.s['llm_requests'] == 20 and led.s['llm_http_attempts'] == 20
          and tok_sum == led.s['llm_tokens_in'] + led.s['llm_tokens_out'] and pools().http_peak <= 2 and not errs and len(got) == 20 and all(isinstance(g[2], dict) and g[2]['echo'].startswith(g[0]) for g in got),
          http_peak=pools().http_peak, errs=errs[:3], unit_tokens=fc.unit_tokens)
    fc2 = FakeMeteredClient(led, out / 'raw2', http_cap=100, stage='smoke', fail_requests={1})
    r1 = fc2.call('fast', 'u', {'k': 'a'}, 'sys', meta={'case': 'c', 'arm': 'a', 'call': 1})
    blocked = False
    try:
        fc2.call('fast', 'u', {'k': 'b'}, 'sys', meta={'case': 'c', 'arm': 'a', 'call': 2})
    except budget.BudgetExhausted:
        blocked = True
    check('transient transport fault: one bounded retry answers the same logical request (2 HTTP attempts, 1 unknown usage), and the next paid call is blocked until an operator accepts',
          r1[1]['attempts'] == 2 and led.s['llm_tokens_unknown'] == 1 and blocked and (out / 'raw2' / '021_fast_attempt0.json').exists())
    # 5 synthetic cases: concurrent fits of two cases (n_updates=20) through the numeric pool; receipts say cache vs fit; adapter cache notes; restore never fits
    sp, psp = _synthetic_splits(out)
    scases = ec.cases_from_split(context.read_json(sp))
    check('synthetic split passes this package\'s check against its synthetic parent', check_split_dp(context.read_json(sp), context.read_json(psp))['ok'])
    run = out / 'run'
    led2 = SafeLedger(out / 'ledger_fits.json', max_fit_attempts=40, max_llm_requests=0, max_llm_tokens=0, max_wall_s=3600, max_retries=1)
    rt.FIT_RETRY = True
    for case in ('SM_V03', 'SM_V04'):
        rc, _ = run_worker(['--worker-build', run, sp, case], out / ('build_%s.log' % case), 600)
        check('public materials built in a worker for %s' % case, rc == 0 and all(m in ec.load_registry(run / case) for m in PUBLIC))
    got2, e2 = {}, []

    def fits(case):
        try:
            got2[case] = fit_cells(led2, run, sp, case, [('None', s) for s in SEEDS], n_updates=20)
        except Exception as exc:  # noqa: BLE001
            e2.append((case, exc))
    ts = [threading.Thread(target=fits, args=(c,)) for c in ('SM_V03', 'SM_V04')]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    snap = pools().snapshot()
    check('6 short fits of two cases through the numeric pool: ledger charged 6, all fit_started, numeric peak in [2, 3]',
          not e2 and led2.s['fit_attempts'] == 6 and all(r['fit_started'] for c in got2.values() for r in c.values()) and 2 <= snap['numeric_peak'] <= 3, numeric_peak=snap['numeric_peak'], errors=str(e2)[:200])
    again = fit_cells(led2, run, sp, 'SM_V03', [('None', s) for s in SEEDS], n_updates=20)
    check('cached cells come back as fit_started=False without any ledger fit', all(not r['fit_started'] for r in again.values()) and led2.s['fit_attempts'] == 6)
    cs = scases['SM_V03']
    table = ec.load_registry(run / 'SM_V03') and context.read_json(run / 'SM_V03' / 'overview.json')['entities']
    comp = ec.compile_plan_full(ec.uniform_policy(WIRING_COMPOSITION, 'smoke'), table)
    m1 = ensure_material(run, sp, 'SM_V03', comp['assignment'], ledger=led2)
    fit_cells(led2, run, sp, 'SM_V03', [(m1['material_id'], SEEDS[0])], n_updates=20)
    branch = out / 'branch_f0'
    ad = CaseAdapter(branch, 'SM_V03', led2, REPO, common=run, split_path=sp, cs=cs, public_ids=('None',), seeds=SEEDS,
                     fit_fn=lambda phys, seeds_: fit_cells(led2, run, sp, 'SM_V03', [(phys, s) for s in seeds_], n_updates=20))
    bl = ad.baselines()
    check('branch view: None restored from cache (0 new fits, 3 cache notes)', led2.s['fit_attempts'] == 7 and led2.s['cache_hits'] == 3 and bl[0].feedback(SEEDS, 2, 16)['loss_by_seed'] is not None)
    cand = ad.build_material({'plan_id': 'P1', 'policy': ec.uniform_policy(WIRING_COMPOSITION, 'smoke branch plan')}, remaining_seconds=100)
    cand = ad.evaluate(cand, SEEDS, feedback=True, remaining_seconds=100)
    cell = context.read_json(branch / 'SM_V03' / 'cells' / (ec.cell_id('SM_V03', 'P1', SEEDS[0]) + '.json'))
    check('branch evaluate: alias of the existing physical material, 2 new fits, 1 cache hit recorded on the branch cell from the receipt (not a counter difference)',
          cand.model_seeds == SEEDS and led2.s['fit_attempts'] == 9 and led2.s['cache_hits'] == 4 and cell.get('cache_hit') is True)
    before = led2.s['fit_attempts']
    ad2 = CaseAdapter(out / 'branch_restore', 'SM_V03', led2, REPO, common=run, split_path=sp, cs=cs, public_ids=('None',), seeds=SEEDS,
                      fit_fn=lambda phys, seeds_: fit_cells(led2, run, sp, 'SM_V03', [(phys, s) for s in seeds_], n_updates=20))
    ad2.baselines()
    c2 = ad2.build_material({'plan_id': 'P1', 'policy': ec.uniform_policy(WIRING_COMPOSITION, 'restore plan')}, remaining_seconds=100)
    ad2.evaluate(c2, SEEDS, feedback=True, remaining_seconds=100)
    ad3 = CaseAdapter(out / 'branch_restore', 'SM_V03', led2, REPO, common=run, split_path=sp, cs=cs, public_ids=('None',), seeds=SEEDS,
                      fit_fn=lambda phys, seeds_: fit_cells(led2, run, sp, 'SM_V03', [(phys, s) for s in seeds_], n_updates=20))
    ad3.baselines()
    restored = ad3.restore(['P1'])
    check('restore of a fitted private plan: cache only, never a fit', len(restored) == 1 and led2.s['fit_attempts'] == before)
    # 6 Fast loop with a scripted client: request-side overview dedupe (lossless), stop before a never-sent call, resume sends only the remaining calls
    steps = [{'actions': [{'tool': 'overview', 'arguments': {}}]},
             {'actions': [{'tool': 'build_material', 'arguments': {'plan_id': 'WireComp', 'policy': ec.uniform_policy(WIRING_COMPOSITION, 'smoke composition')}}]},
             {'actions': [{'tool': 'evaluate', 'arguments': {'plan_id': 'WireComp'}}]},
             {'actions': [{'tool': 'compare', 'arguments': {'a': 'None', 'b': 'WireComp'}}]},
             {'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'None', 'reason': 'smoke: keep the public reference'}}]}]
    cap = []
    bpath = out / 'SM_V03_scripted'
    sc_ = ScriptedClient(copy.deepcopy(steps), fail_at=4, capture=cap)
    common_fit = adapter_factory(run, sp, cs, ('None',), SEEDS)
    result = base.branch(bpath, 'SM_V03', br.Knowledge(), led2, sc_, None, evidence_roundtrip=EVIDENCE_ROUNDTRIP, max_tool_corrections=MAX_TOOL_CORRECTIONS, tool_contracts=CONTRACTS, seeds=SEEDS,
                         adapter_factory=lambda r_, c_, l_, rp, **kw: CaseAdapter(r_, c_, l_, rp, common=run, split_path=sp, cs=cs, public_ids=('None',), seeds=SEEDS,
                                                                                  fit_fn=lambda phys, seeds_: fit_cells(led2, run, sp, 'SM_V03', [(phys, s) for s in seeds_], n_updates=20)),
                         limits=LIMITS, dataset='synthetic', commit_fn=ES.commit_branch, allowed_features=ALLOWED_FEATURES, entity_count=16, request_trace_dedupe=True)
    rows = ES._trace(bpath)
    req2 = cap[1]
    full_ov = req2['overview']
    tr = req2['current_trace']
    stored_ov = next(r['overview'] for r in rows if r['event'] == 'job_started')
    ov_tool = next((r for r in tr if r.get('event') == 'tool_completed' and r.get('tool') == 'overview'), None)
    same_prefix = [r['event'] for r in tr] == [r['event'] for r in rows[:len(tr)]] and rows[len(tr)]['event'] == 'fast_request' and rows[len(tr)]['number'] == 2
    others_intact = all(a == b for a, b in zip(tr, rows[:len(tr)]) if not (a.get('event') == 'job_started' or (a.get('event') == 'tool_completed' and a.get('tool') == 'overview')))
    check('request-side overview dedupe: the request keeps the full overview once, job_started / overview-tool copies in current_trace are markers, the stored trace holds the full overview, every other event intact',
          isinstance(full_ov, dict) and full_ov.get('n_entities') == 16 and tr[0]['overview'] == br.OVERVIEW_DEDUPE_MARKER and stored_ov == full_ov and ov_tool is not None
          and ov_tool['output'] == br.OVERVIEW_DEDUPE_MARKER and same_prefix and others_intact, n_trace_events_in_request=len(tr), n_stored=len(rows))
    check('scripted stop before the 4th call: INCOMPLETE / AGENT_CALL_FAILED, 3 answered calls, last request unanswered',
          result.status == 'INCOMPLETE' and result.failure_kind == 'AGENT_CALL_FAILED' and result.calls == 4 and rows[-2]['event'] == 'fast_request' and not any(r['event'] == 'fast_response' and r['number'] == 4 for r in rows))
    sc_resume = ScriptedClient(copy.deepcopy(steps[3:]))
    fits_before = led2.s['fit_attempts']
    result2 = base.resume_branch(bpath, 'SM_V03', br.Knowledge(), led2, sc_resume, max_tool_corrections=MAX_TOOL_CORRECTIONS, tool_contracts=CONTRACTS, seeds=SEEDS,
                                 adapter_factory=lambda r_, c_, l_, rp, **kw: CaseAdapter(r_, c_, l_, rp, common=run, split_path=sp, cs=cs, public_ids=('None',), seeds=SEEDS,
                                                                                          fit_fn=lambda phys, seeds_: fit_cells(led2, run, sp, 'SM_V03', [(phys, s) for s in seeds_], n_updates=20)),
                                 evidence_roundtrip=EVIDENCE_ROUNDTRIP, limits=LIMITS, dataset='synthetic', commit_fn=ES.commit_branch, allowed_features=ALLOWED_FEATURES, entity_count=16, request_trace_dedupe=True)
    rows2 = ES._trace(bpath)
    n_prefix = context.read_json(bpath / 'resume.json')['prefix_events']
    check('resume sends only the remaining 2 calls (no completed request repeated), trace prefix kept, 0 new fits, commit None',
          result2.status == 'COMPLETE' and sc_resume.n == 2 and result2.calls == 5 and led2.s['fit_attempts'] == fits_before and (bpath / 'resume.json').exists()
          and rows2[:n_prefix] == rows[:n_prefix] and rows2[n_prefix]['event'] == 'job_resumed' and rows[n_prefix]['event'] == 'fast_request' and result2.committed_plan_id == 'None')
    # 7 labels: C_B, freeze_e, score_e refused before the whole-stage barrier, then scored; physical cache holds no label block
    branch = bpath
    rc1, _ = run_worker(['--worker-label', 'c_b', branch, sp, 'SM_V03', out], out / 'label_cb.log', 600)
    rc2, _ = run_worker(['--worker-label', 'freeze_e', branch, sp, 'SM_V03', out], out / 'label_fe.log', 600)
    rc3, _ = run_worker(['--worker-label', 'score_e', branch, sp, 'SM_V03', out], out / 'label_se_refused.log', 600)
    context.write_json(out / 'execution_finished.json', {'epoch': time.time(), 'branches': [[str(branch), 'SM_V03']], 'failures': []})
    context.write_json(out / 'all_e_predictions_frozen.json', {'epoch': time.time(), 'branches': [str(branch)]})
    rc4, _ = run_worker(['--worker-label', 'score_e', branch, sp, 'SM_V03', out], out / 'label_se.log', 600)
    es = context.read_json(branch / 'SM_V03' / 'e_scores.json') if rc4 == 0 else {}
    check('C_B scored, E frozen, score_e refused before the barrier and scored after it (4 origins x 16 entities); physical cache carries no label block',
          rc1 == 0 and rc2 == 0 and rc3 != 0 and rc4 == 0 and all(v['e']['n_origins'] == 4 and v['e']['n_entities'] == 16 for v in es.get('cells', {}).values())
          and all(set(r['scores']) == {'c_a'} for r in ec.branch_cells(run / 'SM_V03').values()))
    # 8 decision-point census on the real parent artifacts (read-only) and the Slow contract
    cen = census_dp('D01', 3)
    q02 = next(r for r in cen['cases']['D01_Q02']['branches'] if r['arm'] == 'f0')
    never = [x['plan_id'] for x in q02['unknown_then']['built_never_evaluated']]
    refs = set(cen['legal_evidence_refs'])
    dp_refs_ok = all(p['ref'] in refs for c in cen['cases'].values() for r in c['branches'] if r.get('decision_points') for p in r['decision_points'])
    interrupted = [r for c in cen['cases'].values() for r in c['branches'] if 'interrupted' in r.get('trajectory_kind', '')]
    check('census of the parent D01: 12 cases, decision points reference legal refs, no data source name, the interrupted trajectory is kept as facts, the built-but-never-evaluated censor plan of the later case is listed as unverified',
          cen['census_status'] == 'CENSUS_COMPLETE' and len(cen['cases']) == 12 and dp_refs_ok and 'censor_only' in never and interrupted and interrupted[0]['cost']['tokens'].startswith('unknown'),
          never_evaluated_q02_f0=never, n_refs=len(refs), n_interrupted=len(interrupted), bytes_tier3=payload_bytes(SLOW_DP, slow_payload(cen, 'D01', 1)))
    other = q02['other_arm_evidence']['physical_materials']
    check('other-arm evidence is listed apart from the branch\'s own pool', all(p not in q02['offline_verification']['materials'] for p in other))
    legal = sorted(refs)
    fake = {'decision': 'PROPOSE', 'candidate': {'research_mode': 'test-first priority', 'workflow': 'Observe the overview; construct one discriminating plan; evaluate it before any second construction; commit the evaluated plan with the lower C_A unless the gap is within one seed SE.',
                                                 'principles': None, 'observable_applicability': {'const': True}, 'applicability_summary': 'smoke', 'evidence_refs': [legal[0]], 'rationale': 'r',
                                                 'evidence_review': {'supports': [], 'contradicts_or_limits': [], 'rule_changes': [], 'uncertain_and_expected_behavior_change': 'u'},
                                                 'rule_status': [{'rule': 'r1', 'status': 'hypothesis', 'support': [], 'counter': []}], 'commit_policy': 'p',
                                                 'main_mechanism_changed': {'mechanism': 'evaluate before inspecting more', 'deployment_observable_trigger': 'a built plan without C_A while slots remain'},
                                                 'decision_change_vs_old_card': 'd', 'historical_decision_point_expected_to_change': {'ref': legal[0], 'expected_change': 'x'},
                                                 'counterexample_expected': {'ref': legal[1], 'should_hold_or_may_fail': 'y'}, 'conditions_same_as_old_card': 'z'}}
    parsed = parse_proposal(fake, domain='D01', slot=1, legal_refs=legal)
    ok = parsed['decision'] == 'PROPOSE' and parsed['skill'].skill_id == 'D01-N1-r1'
    for badtext in ('build censor as on S06', 'use the D01_Q03 recipe', 'the cut at 4032 hours', 'entity 7 is special', 'the electricity data'):
        try:
            parse_proposal({**fake, 'candidate': {**fake['candidate'], 'workflow': fake['candidate']['workflow'] + ' ' + badtext}}, domain='D01', slot=1, legal_refs=legal)
            ok = False
        except ValueError:
            pass
    try:
        parse_proposal({**fake, 'candidate': {**fake['candidate'], 'historical_decision_point_expected_to_change': {'ref': 'nowhere/x/y:1', 'expected_change': 'x'}}}, domain='D01', slot=1, legal_refs=legal)
        ok = False
    except ValueError:
        pass
    keep = parse_proposal({'decision': 'KEEP', 'rationale': 'r', 'conditions_same_as_old_card': 'all'}, domain='D01', slot=2, legal_refs=legal)
    check('Slow parser: one candidate accepted; case ids (short and full), row numbers, entity numbers, data source names and an illegal ref are rejected; KEEP accepted', ok and keep['decision'] == 'KEEP')
    check('old cards of both domains load verbatim from the parent freeze', all(old_card(d)['injected_body'] == context.read_json(PARENT_ROOT / 'freeze' / ('%s.json' % d))['injected_body'] for d in DOMAINS))
    check('rotation orders by case_index', rotate(SELECT_ARMS, 12) == ['w_old', 'n1', 'n2', 'n3'] and rotate(SELECT_ARMS, 13) == ['n1', 'n2', 'n3', 'w_old'] and rotate(TEST_FAST_ARMS, 14) == ['f_new', 'f0', 'f_old'])
    check('G sign: A better (lower E) is positive', G([1.0, 1.0, 1.0], [1.1, 1.1, 1.1], 1.0)['mean'] > 0)
    res = {'status': 'PASS' if all(c['ok'] for c in checks) else 'FAIL', 'checks': checks, 'real_fits': 0, 'llm_requests': 0, 'finished_local': now()}
    context.write_json(out / 'smoke_result.json', res)
    print('SMOKE_RESULT', res['status'], sum(c['ok'] for c in checks), '/', len(checks), flush=True)
    return res


# ============================================================================= read-only monitor
def monitor(root: Path = ROOT, interval: float = 60.0, once: bool = False) -> None:
    """Prints process liveness, ledger totals, last request / fit completion, stage progress and log tails. Never writes, never takes the lock."""
    import psutil
    P_ = paths(root)
    while True:
        line = ['MONITOR', now()]
        procs = []
        for name in ('driver_launch.json', 'slow_worker_launch.json', 'select_worker_launch.json', 'metatest_worker_launch.json'):
            q = root / name
            if q.exists():
                try:
                    pid = int(context.read_json(q)['pid'])
                    procs.append('%s=%d:%s' % (name.split('_')[0], pid, 'alive' if psutil.pid_exists(pid) else 'exited'))
                except Exception:  # noqa: BLE001
                    pass
        line.append('procs[%s]' % ' '.join(procs))
        held = None
        if P_['lock'].exists():
            try:
                with open(P_['lock'], 'rb') as fh:
                    fh.read(1)
                held = False
            except OSError:
                held = True                                     # a locked byte range cannot be read: some controller holds it (read-only probe)
        line.append('lock=%s' % ({None: 'absent', True: 'held', False: 'free'}[held]))
        if P_['ledger'].exists():
            led = context.read_json(P_['ledger'])
            ev = led.get('events', [])
            last_llm = max((e['epoch'] for e in ev if e.get('kind') == 'llm_finished'), default=None)
            last_fit = max((e['epoch'] for e in ev if e.get('kind') == 'fit_finished'), default=None)
            line.append('fits=%d/%d ok=%d failed=%d cache=%d req=%d http=%d tok=%.2fM unknown=%d' % (
                led['fit_attempts'], led['caps']['max_fit_attempts'], led['fits_ok'], led['fits_failed'], led['cache_hits'], led['llm_requests'], led['llm_http_attempts'],
                (led['llm_tokens_in'] + led['llm_tokens_out']) / 1e6, led['llm_tokens_unknown']))
            line.append('last_llm=%s last_fit=%s' % ('%.0fs ago' % (time.time() - last_llm) if last_llm else '-', '%.0fs ago' % (time.time() - last_fit) if last_fit else '-'))
            if led.get('paid_clock_started_epoch'):
                line.append('paid=%.1fh' % ((time.time() - led['paid_clock_started_epoch']) / 3600))
        for stage in ('select', 'metatest'):
            pp = P_[stage] / 'progress.json'
            if pp.exists():
                pr = context.read_json(pp)
                st = {}
                for c, v in pr.get('cases', {}).items():
                    st[v.get('status', '?')] = st.get(v.get('status', '?'), 0) + 1
                pl = pr.get('pools') or {}
                line.append('%s: %s pools(num %s/%s peak %s, http %s/%s peak %s)' % (stage, st, pl.get('numeric_active'), pl.get('numeric_limit'), pl.get('numeric_peak'), pl.get('http_active'), pl.get('http_limit'), pl.get('http_peak')))
        for name in ('slow', 'select', 'metatest'):
            lg = P_['logs'] / ('%s.log' % name)
            if lg.exists():
                tail = lg.read_text(encoding='utf-8', errors='replace').splitlines()[-2:]
                if tail:
                    line.append('%s.log: %s' % (name, ' | '.join(t[:110] for t in tail)))
        print(' '.join(line), flush=True)
        if once:
            return
        time.sleep(interval)


# ============================================================================= CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--resume-stage')
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--restart', default='')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--monitor', action='store_true')
    ap.add_argument('--interval', type=float, default=60.0)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--stage-worker')
    ap.add_argument('--lock-probe')
    a = ap.parse_args()
    root = Path(a.root)
    if a.lock_probe:
        sys.exit(lock_probe(Path(a.lock_probe)))
    elif a.stage_worker:
        stage_worker(Path(a.stage_worker))
    elif a.preflight:
        print(json.dumps({k: v for k, v in preflight(root).items() if k in ('package', 'split_check', 'path_binding', 'environment', 'total', 'concurrency_effective')}, indent=1, ensure_ascii=False))
    elif a.smoke:
        smoke(root)
    elif a.wiring:
        print(json.dumps(wiring(root), indent=1, ensure_ascii=False, default=str))
    elif a.run:
        package_run(root, accept_unknown=a.accept_unknown_usage)
    elif a.resume_stage:
        resume_stage(root, a.resume_stage, accept_unknown_usage=a.accept_unknown_usage, restart=[x for x in a.restart.split(',') if x])
    elif a.result:
        readout(root)
        print((root / 'tables.md').read_text(encoding='utf-8'))
    elif a.monitor:
        monitor(root, a.interval, a.once)


if __name__ == '__main__':
    main()
