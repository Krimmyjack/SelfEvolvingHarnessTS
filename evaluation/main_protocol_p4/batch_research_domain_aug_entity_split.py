"""DEV-DOMAIN-AUG-ENTITY-SPLIT (docs/DEV_DOMAIN_AUG_ENTITY_SPLIT_TASK_2026-09-19.md): two domains, entity-disjoint Source / Select /
MetaTest cases of 16 original columns each; no-card Fast research on 8 Source cases per domain -> one Slow formation per domain (<= 3
candidate Workflows + optional Principles, late E of the Source cases authorized) -> Select on 2 cases per domain (J on E) -> freeze the
domain card and a Fixed_dev program -> MetaTest on 2 new cases per domain: F0 / F_domain / F_generic / RandomSearch_B4 / Fixed_dev + the
four public references, whole-stage C_B -> E barrier -> E -> readout.

Reused unchanged: the Fast controller (batch_research.run_job via run_batch_research_v1.branch / resume_branch), the metered client and
ledger (batch_research_runtime), the Skill carrier (domain_skill), the tempo_aug primitives / preset / edit, the census compression helpers
of the earlier tempo-aug packages. New here (opt-in only): the entity-split case profile (methods/ttha/batch_base/entity_case.py), its
adapter, the RandomSearch_B4 draw of task §5.1, the per-domain formation with three candidates, the J(W) / Fixed_dev selection and the readout.

  --preflight | --smoke | --wiring | --run | --resume-stage STAGE [--accept-unknown-usage] [--restart b1,b2] | --result
  workers (subprocess only): --stage-worker CFG | --worker-build ROOT SPLIT CASE | --worker-material ROOT SPLIT CASE PHYS |
           --worker-fit ROOT SPLIT CASE PHYS SEED [--tag T] [--n-updates N] | --worker-label STAGE BRANCH SPLIT CASE STAGE_ROOT
The controller never imports torch.
"""
from __future__ import annotations

import os

KMP_AT_START = os.environ.get('KMP_DUPLICATE_LIB_OK')

import argparse
import copy
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
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
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_skill as W          # read-only helpers: prompts, contracts, compression
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_learning_loop as LL  # read-only helpers: review parsing, skeleton tier

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_domain_aug_entity_split'
MODULE = 'evaluation.main_protocol_p4.batch_research_domain_aug_entity_split'
TASK = 'docs/DEV_DOMAIN_AUG_ENTITY_SPLIT_TASK_2026-09-19.md'
SPLIT_DOC = REPO / 'docs' / 'DOMAIN_AUG_ENTITY_SPLIT_V1.json'
PACKAGE = 'DEV-DOMAIN-AUG-ENTITY-SPLIT'
EXPOSURE = ec.EXPOSURE
DOMAINS = ('D01', 'D02')
SEEDS = (20269181, 20269182, 20269183)
WIRING_SEEDS = SEEDS
PUBLIC = ec.PUBLIC
PUBLIC_STEPS = ec.PUBLIC_STEPS
WIRING_COMPOSITION = [{'op': 'tp_regime'}, {'op': 'tp_resample'}, {'op': 'tp_censor'}]        # the wiring's explicit 3-step composition
LIMITS = {'max_calls': 16, 'max_tools': 24, 'max_new_evaluations': 4}
FAST_TOKEN_CAP = 1_000_000                       # amendment 1 (user 2026-09-19 22:35 '预算不卡上限'): the task-book 250,000 blocked D01_S01_f0 before commit; token caps no longer bind
MAX_OUTPUT_TOKENS = 12_000
MAX_TOOL_CORRECTIONS = 2
EVIDENCE_ROUNDTRIP = True
TOTAL = {'max_fit_attempts': 846, 'max_llm_requests': 644, 'max_llm_tokens': 32_000_000, 'max_wall_s': 16 * 3600, 'max_retries': 6}   # amendment 1: tokens / wall non-binding; fits and requests are protocol quantities (unchanged)
HTTP_CAP = 648
STAGE_RETRIES = 2
ALLOC = {'wiring': {'fits': 12, 'requests': 0, 'tokens': 0},
         'source': {'fits': 384, 'requests': 256, 'tokens': 12_000_000},
         'formation': {'fits': 0, 'requests': 4, 'tokens': 2_100_000},
         'select': {'fits': 192, 'requests': 192, 'tokens': 9_000_000},
         'metatest': {'fits': 252, 'requests': 192, 'tokens': 9_000_000}}   # amendment 1: token allocations x3 (non-binding); task-book values 4.0M / 0.7M / 3.0M / 3.0M
FIT_TIMEOUT_S = 300.
FIT_THREADS = 8                                  # fixed per worker (bitwise-stable across serial / parallel launches)
FIT_PARALLEL = 3                                 # at most one worker per seed at a time
MODEL = W.MODEL
BODY_LIMIT = 6000
MAX_CANDIDATES = 3
CAND_IDS = ('W1', 'W2', 'W3')
ALLOWED_FEATURES = rt.BATCH_FIELDS               # batch_median_<spec.OBS_FIELDS>
TOL = 1e-12
T975_DF2 = 4.302652729911275
RANDOM_SEED_ROOT = 2026091901
RANDOM_MAX_TRIES = 20
RANDOM_MIN_GROUP = 2
SELECT_ORDER = {'V01': ('cand_W1', 'cand_W2', 'cand_W3'), 'V02': ('cand_W3', 'cand_W1', 'cand_W2')}
TEST_ORDER = {'D01_Q01': ('f0', 'f_domain', 'f_generic'), 'D01_Q02': ('f_domain', 'f_generic', 'f0'),
              'D02_Q01': ('f_generic', 'f0', 'f_domain'), 'D02_Q02': ('f0', 'f_generic', 'f_domain')}
CONTROL_ARMS = ('random', 'fixed_dev')
ARM_LABEL = {'f0': 'F0', 'f_domain': 'F_domain', 'f_generic': 'F_generic', 'random': 'RandomSearch_B4', 'fixed_dev': 'Fixed_dev'}

FIELD_DEFINITIONS = {
    'mean': 'raw mean of the 672 T values', 'std': 'raw std (ddof=0) of T; also the scaler', 'missing_count': 'always 0 in these dense files',
    'head168_mean': 'raw mean of the first 168 T hours', 'tail168_mean': 'raw mean of the last 168 T hours',
    'standardized_trend_per_100h': 'OLS slope of the standardized T per 100 hours', 'last168_std': 'raw std of the last 168 hours', 'full_T_std': 'raw std of T',
    'last168_std_over_full_T_std': 'ratio of the two', 'lag24_corr': 'Pearson autocorrelation at lag 24 h', 'lag168_corr': 'Pearson autocorrelation at lag 168 h',
    'nondc_spectrum_top5_energy_share': 'share of non-DC spectral energy in the 5 strongest bins', 'standardized_abs_p95': '95th percentile of |standardized T|',
    'standardized_abs_max': 'max of |standardized T|', 'n_parent_pairs': 'training windows per entity (433)',
    'r_head': 'de-trended lag-24 correlation of the first half of T', 'r_tail': 'same for the second half', 'r_gap': '|r_head - r_tail|',
    'nn1_input_distance': 'mean RMS distance from each parent input to its nearest other parent input (|k-l| >= 48)',
    'nn5_target_deviation': 'mean RMS deviation of a parent target from the mean target of its 5 nearest inputs'}

GENERIC_TEXT = ('读取任务、Consumer 和全批观察，找可能影响预测的结构；必要时检查代表片段，区分真实变化与处理伪影。用合法原语构造能区分假说的完整方案，检查实际材料变化，'
                '在预算内取得下游反馈。根据已有证据决定继续、保留公共方案或提交已评估方案。处理更复杂或改动更大不代表更好；有限支持块的优势不保证后期优势。')

PERMISSION_SENTENCE = ('允许按合法 T 观察构造统一或条件化方案；单实体预测变化不能视作该实体材料的独立因果贡献，条件化完整方案仍可由共享训练比较。'
                       ' (Uniform or entity-conditional complete plans may be constructed from legal T observations; a change in one entity\'s prediction is '
                       'not the independent causal contribution of that entity\'s material, yet conditional complete plans can still be compared through the '
                       'shared training.)')
PACKAGE_PARAGRAPH = ('''本包的动作空间：每个实体的程序可以是 (a) 不增强（空 steps）；(b) 七个原语中的 1–3 步显式组合（固定执行次序、regime/shock 与 calendar/amplitude 互斥、不重复）；(c) 公共预设 P_NoMixRecipe；(d) P_NoMixRecipe_Edit（关闭预设的 0–2 个组件；0 个 = 预设本身）。default 与最多 8 条 T 观测谓词规则可以按观察给不同实体不同程序，也可以全体相同。每个方案处理 240 点联合窗口（192 输入 + 48 训练目标一起变换），子视图与不变的父视图 0.5/0.5 共同训练同一个 Consumer；预测输入与评分真值从不增强。目标是优化整批后续预测效用：先按数据现象与处理语义提出假说，检查实际材料变化，再比较真实拟合；C_A 是有限支持块，不能保证未来排序，没有 C_B/E。没有任何原语、预设或组件被预先认定有益或有害；不强制用满 4 个新方案、不强制分组、不强制三步、不强制偏离或保留 NoMix。可以依据前一次实验的结果组织下一次实验。可以提交任一自身已评估方案或公共参照；Runner 不以 C_A argmin 覆盖你的提交。
(Action space of this package: each entity's program is (a) no augmentation (empty steps); (b) an explicit 1-3 step composition of the seven primitives (fixed execution order, regime/shock and calendar/amplitude mutually exclusive, no repeats); (c) the public preset P_NoMixRecipe; (d) P_NoMixRecipe_Edit disabling 0-2 preset components (0 = the preset itself). The default and up to 8 T-observation predicate rules may give different entities different programs, or all the same. A plan transforms each 240-point joint window (192 inputs + 48 child training targets together); the child view is trained next to the unchanged parent view (0.5/0.5) in one shared Consumer; serving inputs and scoring truth are never augmented. Goal: better whole-batch utility on later predictions - form hypotheses from the data phenomena and the processing semantics, inspect the actual material change, then compare real fits; C_A is a limited support block that does not guarantee later rankings; there is no C_B/E. No primitive, preset or component is presumed helpful or harmful; using all 4 slots, grouping, three steps, deviating from or keeping NoMix are not required. You may organize the next experiment from the previous result. You may commit any plan you evaluated or any public reference; the Runner never overrides your commit with a C_A argmin.)''')
FAST_SYSTEM = W.FAST_SYSTEM + '\n' + PERMISSION_SENTENCE + '\n' + PACKAGE_PARAGRAPH

CONTRACTS = copy.deepcopy(W.CONTRACTS)
CONTRACTS['overview'] = {'arguments': {}, 'meaning': 'All N entity rows of T-only observations (formulas in field_definitions), batch summaries, the Consumer (parent 0.5 + child 0.5 training) and the action table: 7 TempoPFN-sourced primitives, the P_NoMixRecipe preset, its local edit, composition rules, window and randomness semantics.'}
CONTRACTS['inspect_data'] = {'arguments': {'entity_indices': '[1..8 integers in 0..N-1]', 'kind': 'hour_profile|daily_means|segment',
                                           'sub_range': 'optional [absolute_start, absolute_end): ABSOLUTE row indices inside overview.train_rows; segment at most 336 rows (default: the last 336 T rows)'},
                             'meaning': 'Raw current-T views normalized by the frozen T scaler: 24 clock-hour means, 28 daily means, or a segment.'}
CONTRACTS['build_material'] = {
    'arguments': {'plan_id': 'new short alphanumeric/underscore id',
                  'policy': {'default': {'steps': [{'op': '<primitive name>'}, {'op': '<another primitive>'}]},
                             'rules': [{'when': {'feature': '<overview field>', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': [{'op': ta.RECIPE_EDIT, 'disabled_ops': ['<component>']}]}],
                             'rationale': '<current hypothesis, the observation behind it, the expected material change; no data source or case name>', 'observation_fields_used': []}},
    'meaning': 'Construct one complete whole-batch plan and freeze its child view; does not train. Each program is [] (no augmentation), 1-3 primitives (no repeats, regime/shock and calendar/amplitude exclusive, executed in the fixed order regardless of listing), [{"op":"P_NoMixRecipe"}] alone, or one {"op":"P_NoMixRecipe_Edit","disabled_ops":[0-2 component names]} step. Up to 8 rules; first matching rule wins, a null field never fires a rule; thresholds are numbers or a current-batch quantile. No seed, strength, weight or probability argument exists; the child of a program on an entity is frozen per (entity, program), and plans with identical complete assignments are the same material and are rejected as duplicates of the existing id (no new material, fit or slot). An all-empty plan is None.'}
CONTRACTS['inspect_material'] = {'arguments': {'plan_id': '<built id>', 'entity_index': 'optional integer: also return one parent/child window', 'window': "optional 'largest_change' (default) | 'first' | 'last'"},
                                 'meaning': 'What the plan changed: executed steps, identity (no-op) counts, edit skip counts, change fraction and RMS for inputs and child targets separately. Not predictive utility.'}
CONTRACTS['evaluate'] = {'arguments': {'plan_id': '<built id>'}, 'meaning': 'Train the complete plan (parent 0.5 + child 0.5, shared MLP, 2000 updates) under the 3 frozen paired seeds and return C_A only (normalized MSE on the two support origins, entity macro). At most 4 new complete plans per case; the four public references are already evaluated.'}

FEEDBACK_ROLES = ('Feedback roles, by name: C_A = the immediate feedback Fast legally used at decision time (origins t, t+48); it explains why Fast decided as it did. '
                  'C_B = the delayed check of that case after its commit (origins t+96, t+144). E (development_late) = the late block of that case (origins t+192..t+336), '
                  'the criterion this study cares about, opened after every Source commit was frozen and authorized for learning only; it tells which guidance deserves '
                  'reuse, revision or should stay a hypothesis. When C_A and E disagree, KEEP the conflict: a C_A lead is not a later gain, and no rule may a priori ignore '
                  'all C_A. All blocks are the same 48-hour forecast; they differ in the calendar period scored. The cases are different entity groups of ONE domain at two '
                  'cut positions; a later case reuses the learned method on entities that never entered this input. A per-entity prediction change is not the independent '
                  'contribution of that entity\'s material; this does not forbid grouping by observations, whose plans are still judged by the complete shared model. '
                  'Uncertainty can justify an extra check, a limited trial or keeping a public reference; one undecided comparison never becomes a permanent ban; a '
                  'historical conclusion supports only the treatments and situations it actually covered; an untried action is not harmful. The post-hoc pool best is a '
                  'diagnostic, never a known deployment answer.')
GOAL = ('Your goal: improve the FINAL downstream prediction (E) of Fast on NEW entity groups of the same domain and save worthless experiments. Learn from the eight '
        'trajectories: which observations helped to form a processing hypothesis, how discriminating complete plans were (or should have been) constructed, how current '
        'C_A feedback and this history should be used, and when to continue, keep a public reference or commit. You are NOT asked to pick one usual recipe and you are NOT '
        'asked to write complex conditions; a simple uniform strategy, a public reference, an explicit composition, a preset edit or an observation-conditional plan are all '
        'legal recommendations when the evidence supports them. State the COMMIT POLICY explicitly (what role the current C_A plays; when a learned preference yields to '
        'current evidence). Do not require "must dominate" or "must win 3/3". Say which conditions the current tools compute and which are learned preferences. For every '
        'main rule give its support and counter evidence and mark it supported / hypothesis / unresolved; do not inherit any ban that this input does not support.')
NO_IDS = LL.NO_IDS
FORMAT_FORM = ('Output exact JSON, no markdown: {"decision":"KEEP","rationale":"...","evidence_review":{...}} OR {"decision":"PROPOSE","candidates":[{"candidate_id":"W1",'
               '"research_mode":"<short label>","workflow":"<executable Workflow incl. observe / construct / experiment / commit-or-stop; the commit policy must be inside it>",'
               '"principles":"<conditions, actions, reasons, exceptions>" or null,"observable_applicability":{"const":true},"applicability_summary":"<=400 characters",'
               '"evidence_refs":["exact refs from the supplied legal ranges"],"rationale":"...","evidence_review":{"supports":["observation -> which advice it supports"],'
               '"contradicts_or_limits":["which later result (C_B / E) contradicts or limits which judgement"],"rule_changes":["which rule is kept / dropped / rewritten because of it"],'
               '"uncertain_and_expected_behavior_change":"what remains uncertain and which Fast behaviour should change"},'
               '"rule_status":[{"rule":"<one main rule>","status":"supported|hypothesis|unresolved","support":["refs or short facts"],"counter":["refs or short facts"]}],'
               '"commit_policy":"<one paragraph restating the commit rule>","decision_change_vs_no_card":"<the one concrete decision this card changes>"}, ...]}. '
               'One to THREE candidates with candidate_id W1, W2, W3 in order; candidates must differ by a stated research reason (a different investigation order, '
               'construction priority, experiment allocation or commit judgement), not rewordings. Workflow plus Principles render to at most %d characters per candidate. '
               'KEEP (= no usable candidate, recorded NO_CARD) is valid and is not resampled.')
SLOW_FORM = ('You are the offline Slow of a batch training-data augmentation research Harness. You receive a deterministic census of eight completed research cases of ONE '
             'neutral domain: eight disjoint groups of 16 entities, each researched once by the no-card Fast with the same tools and budget; every case carries T-only '
             'observations, the actual tool calls and returned results (compressed), complete augmentation plans, material diagnostics, per-seed/per-origin C_A of every '
             'evaluated material, the commit and its reason, post-commit C_B and the late block E of every evaluated material, costs, failures, and a deterministic '
             'post-hoc summary (commit vs C_A argmin vs E best, marked diagnostic). ' + FEEDBACK_ROLES + ' ' + GOAL + ' ' + NO_IDS + ' ' + FORMAT_FORM % BODY_LIMIT)

CAND_KEYS = set(LL.CAND_KEYS) | {'rule_status', 'commit_policy', 'decision_change_vs_no_card'}
COMPRESSION_RULES = {
    'frozen_before_first_slow_call': True,
    'inherits': 'the size-only tier rules 1-3 of the earlier tempo-aug packages (W._compress) and the skeleton tier (LL.compress_trajectory)',
    'this_package': ['evidence refs are "source/<branch>/<event_id>" (+ "/overview", "/labels_after_commit"); legal set = every such id of the domain',
                     'per case one materials table keyed by physical material id: label, assignment (uniform steps or per-entity), C_A / C_B / E per seed and per origin',
                     'tier chosen per domain so that the frozen client reservation 2 x (message bytes + 2048 + max_tokens) leaves room for one correction round inside the formation cap',
                     'the other domain, the Select and MetaTest cases never enter a domain census'],
    'never_dropped': 'actions, plans, C_A / C_B / E numbers, commits, failures, conflicts',
}


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'configs': root / 'stage_configs', 'logs': root / 'logs', 'split': root / 'entity_split_copy.json',
            'wiring': root / 'wiring', 'smoke': root / 'smoke', 'source': root / 'source', 'formation': root / 'formation', 'select': root / 'select',
            'freeze': root / 'freeze', 'metatest': root / 'metatest'}


def ref(stage: str, branch: str, tail: str) -> str:
    return '%s/%s/%s' % (stage, branch, tail)


def load_split(root: Path) -> dict:
    p = paths(root)['split']
    return context.read_json(p if p.exists() else SPLIT_DOC)


def cases_of(root: Path) -> dict:
    return ec.cases_from_split(load_split(root))


def case_ids(root: Path, role: str, domain: str | None = None) -> list:
    return [c for c, cs in cases_of(root).items() if cs.role == role and (domain is None or cs.domain == domain)]


# ============================================================================= workers (subprocess entries; torch allowed)
def _init_worker() -> None:
    from methods.ttha.batch_base import readiness as rd
    rd._init_openmp_before_torch()


def _case(split_path: str, case: str) -> ec.CaseSpec:
    return ec.cases_from_split(context.read_json(split_path))[case]


def worker_build(root: str, split_path: str, case: str) -> None:
    _init_worker()
    cs = _case(split_path, case)
    ctx = ec.open_case(cs, Path(root), stage='material')
    built = ec.build_public(ctx)
    print('BUILD_OK', case, sorted(ec.load_registry(ctx.job_dir)), 'built', built, flush=True)


def worker_material(root: str, split_path: str, case: str, phys: str) -> None:
    _init_worker()
    cs = _case(split_path, case)
    ctx = ec.open_case(cs, Path(root), stage='material')
    req = context.read_json(ec.aug_dir(ctx.job_dir) / (phys + '__request.json'))
    try:
        rec = ec.build_material(ctx, req['assignment'], phys)
    except (ta.rd.ProgramExecutionError, policy.PolicyError) as exc:
        context.write_json(ec.aug_dir(ctx.job_dir) / (phys + '__error.json'), {'kind': 'PROGRAM_EXECUTION_REJECTED', 'message': str(exc)[:300]})
        raise
    print('MATERIAL_OK', case, phys, 'build_s %.1f' % context.read_json(rec['summary_path'])['build_seconds'], flush=True)


def worker_fit(root: str, split_path: str, case: str, phys: str, seed: int, tag: str = '', n_updates=None) -> None:
    _init_worker()
    cs = _case(split_path, case)
    rec = ec.fit_one(cs, Path(root), phys, int(seed), threads=FIT_THREADS, timeout_s=FIT_TIMEOUT_S - 30, tag=tag, n_updates=n_updates)
    print('CELL_OK', rec['cell_id'], round(rec['train']['seconds'], 2), 'c_a %.5f' % rec['scores']['c_a']['normalized_mse_macro'], flush=True)


def worker_label(stage: str, branch: str, split_path: str, case: str, stage_root: str) -> None:
    cs = _case(split_path, case)
    ec.label_prerequisite(stage, Path(branch), cs, Path(stage_root))      # raises before any label row is loaded or torch is imported
    _init_worker()
    ec.label_stage(stage, Path(branch), cs, Path(stage_root))
    print('LABEL_OK', stage, Path(branch).name, case, flush=True)


# ============================================================================= controller-side subprocess plumbing (never imports torch)
def worker_env() -> dict:
    env = dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def run_sub(args: list, log: Path, timeout: float) -> tuple:
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with log.open('w', encoding='utf-8') as fh:
            p = subprocess.run([sys.executable, '-B', '-m', MODULE] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout, env=worker_env())
        return p.returncode, time.time() - t0
    except subprocess.TimeoutExpired:
        return -999, time.time() - t0


def proxy_reachable() -> bool:
    import socket
    from urllib.parse import urlparse
    u = urlparse(MODEL['base_url'])
    try:
        with socket.create_connection((u.hostname, u.port or 80), timeout=3):
            return True
    except OSError:
        return False


def fit_cells(ledger, root: Path, split_path: Path, case: str, jobs: list, *, tag: str = '', n_updates=None) -> dict:
    """jobs: [(phys, seed)]. Reserved-before-launch physical fits into the case cache, up to FIT_PARALLEL workers at a time (one per seed at
    most); each cell gets one same-configuration retry when FIT_RETRY and the ledger allows. Returns {cell_id: record}."""
    out, pending = {}, []
    for phys, seed in jobs:
        cid = ec.cell_id(case, phys, seed, tag)
        cp = Path(root) / case / 'cells' / (cid + '.json')
        if cp.exists() and context.read_json(cp).get('status') == 'OK':
            out[cid] = context.read_json(cp)
            continue
        pending.append((phys, seed, cid, cp, 0))
    running = []
    while pending or running:
        while pending and len(running) < FIT_PARALLEL:
            phys, seed, cid, cp, attempt = pending.pop(0)
            ledger.reserve_fit(cid)
            timeout = min(FIT_TIMEOUT_S, ledger.remaining())
            log = Path(root) / case / 'fit_logs' / (cid + ('.log' if attempt == 0 else '.retry.log'))
            log.parent.mkdir(parents=True, exist_ok=True)
            args = ['--worker-fit', root, split_path, case, phys, seed] + (['--tag', tag] if tag else []) + (['--n-updates', n_updates] if n_updates else [])
            fh = log.open('w', encoding='utf-8')
            p = subprocess.Popen([sys.executable, '-B', '-m', MODULE] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, env=worker_env())
            running.append({'p': p, 'fh': fh, 'phys': phys, 'seed': seed, 'cid': cid, 'cp': cp, 'attempt': attempt, 't0': time.time(), 'timeout': timeout})
        time.sleep(0.5)
        for r in list(running):
            rc = r['p'].poll()
            timed_out = rc is None and (time.time() - r['t0']) > r['timeout']
            if rc is None and not timed_out:
                continue
            if timed_out:
                r['p'].kill()
                r['p'].wait()
                rc = -999
            r['fh'].close()
            running.remove(r)
            ok = rc == 0 and r['cp'].exists() and context.read_json(r['cp']).get('status') == 'OK'
            reason = '' if ok else ('worker_timeout' if rc == -999 else 'worker_failed')
            ledger.finish_fit(r['cid'], ok, time.time() - r['t0'], reason)
            print('FIT', r['cid'], 'OK' if ok else reason, round(time.time() - r['t0'], 1), flush=True)
            if ok:
                rec = context.read_json(r['cp'])
                if rec['scores']['c_a']['n_nonfinite_predictions']:
                    raise RuntimeError('nonfinite predictions')
                out[r['cid']] = rec
            elif r['attempt'] == 0 and rt.FIT_RETRY and ledger.can_retry():
                ledger.s['retries_used'] += 1
                ledger.s['events'].append({'kind': 'fit_retry', 'cell': r['cid'], 'reason': reason, 'epoch': time.time()})
                ledger._save()
                pending.append((r['phys'], r['seed'], r['cid'], r['cp'], 1))
            else:
                for q in running:
                    q['p'].kill()
                    q['fh'].close()
                raise RuntimeError(reason)
    return out


def ensure_material(root: Path, split_path: Path, case: str, assignment: list, *, ledger=None) -> dict:
    """The physical record of an assignment in the case cache: None / existing by key / built now in a worker (serial)."""
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
    rc, secs = run_sub(['--worker-material', root, split_path, case, phys], Path(root) / case / 'fit_logs' / ('material_%s.log' % phys), 900)
    if ledger is not None:
        ledger.s['material_wall_seconds'] = ledger.s.get('material_wall_seconds', 0.0) + (time.time() - t0)
        ledger._save()
    err = ec.aug_dir(Path(root) / case) / (phys + '__error.json')
    if rc != 0:
        if err.exists():
            raise br.ToolInputError('plan rejected at execution (no fit, no material): %s' % context.read_json(err)['message'], execution_status='PROGRAM_EXECUTION_REJECTED')
        raise RuntimeError('material worker failed for %s %s' % (case, phys))
    return ec.load_registry(Path(root) / case)[phys]


# ============================================================================= adapter (BatchAdapter protocol + resume hooks)
def public_spec(mid: str) -> dict:
    if mid == 'FixedMixup':
        return {'kind': 'public_reference', 'augmentation': ec.FIXED_MIXUP['definition']}
    steps = PUBLIC_STEPS[mid]
    return {'kind': 'public_reference', 'profile': ec.PROFILE_VERSION, 'policy': ec.uniform_policy(steps, 'public reference %s' % mid), 'programs': [steps],
            'entity_program': [0] * ec.COHORT_SIZE, 'alias_of': None, 'execution_order': 'fixed: ' + ' -> '.join(ta.PRIMITIVES)}


def action_table() -> dict:
    t = ta.action_table()
    t['recipe_edit'] = {'op': ta.RECIPE_EDIT, 'arguments': {'disabled_ops': '0-2 names from primitives (order irrelevant, duplicates ignored)'}, 'semantics': ta.RECIPE_EDIT_TEXT,
                        'randomness': 'Every edit shares the preset\'s per-(case, entity) random stream: the categories, primitives and conv draw of each window are those of '
                                      'the preset; disabling a component never re-draws anything. The same switch on the same entity is the same material in every plan.',
                        'alias': '%s with an empty disabled_ops is %s itself (the public reference; no new material)' % (ta.RECIPE_EDIT, ta.RECIPE)}
    t['program_space'] = ('Per entity: [] | 1-3 primitives (fixed order, exclusives, no repeats) | [%s] | one %s step. The default and up to %d observation-predicate rules '
                          'may give different programs to different entities.' % (ta.RECIPE, ta.RECIPE_EDIT, ta.MAX_RULES))
    t['public_references'] = {'None': 'no augmentation (parent view only)', 'FixedMixup': ec.FIXED_MIXUP['definition'],
                              'P_AmpResample': 'uniform tp_amplitude -> tp_resample (explicit composition)', 'P_NoMixRecipe': 'uniform ' + ta.RECIPE + ' (the preset)'}
    return t


def init_branch_dir(common: Path, branch_root: Path, case: str, public_ids) -> None:
    """A branch view of the case cache: static T objects copied, the public references registered; no other material, cell or trajectory."""
    src, dst = Path(common) / case, Path(branch_root) / case
    if (dst / 'aug_materials' / 'index.json').exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for name in ('scaler.npz', 'overview.json', 'case_spec.json'):
        if not (dst / name).exists():
            shutil.copy2(src / name, dst / name)
    reg = ec.load_registry(src)
    ec.save_registry(dst, {m: {**reg[m], 'phys_id': m} for m in public_ids})


def commit_branch(root, dataset: str, case: str, material_id: str, reason: str, delivery_seed: int, extra: dict | None = None) -> dict:
    jd = Path(root) / case
    p = jd / 'commit.json'
    if p.exists():
        raise RuntimeError('case %s already committed in this branch' % case)
    cells = ec.branch_cells(jd)
    cid = ec.cell_id(case, material_id, delivery_seed)
    if cid not in cells:
        raise ValueError('commit refused: %s has no OK cell for seed %d in this branch' % (material_id, delivery_seed))
    rec = {'job_id': case, 'profile': ec.PROFILE_VERSION, 'material_id': material_id, 'physical_material': cells[cid].get('physical_material', material_id),
           'material_key': cells[cid].get('material_key'), 'delivery_seed': delivery_seed, 'delivery_cell': cid, 'reason': str(reason)[:1000],
           'committed_at_local': now(), 'fitted_materials_at_commit': sorted({c['material_id'] for c in cells.values()}), 'extra': extra or {}}
    context.write_json(p, rec)
    return rec


class CaseAdapter:
    """BatchAdapter over the entity-split case profile. Physical cache = common/<case>/ (materials by assignment key, fits by physical id);
    this branch sees only its own plan ids, its copies of the cells and the public references."""

    def __init__(self, run_dir, case, ledger, repo, *, common, split_path, cs: ec.CaseSpec, public_ids, tool_error_feedback=True, seeds=SEEDS, fit_fn=None):
        self.root, self.common, self.case, self.split_path = Path(run_dir), Path(common), case, Path(split_path)
        self.cs, self.public_ids = cs, tuple(public_ids)
        self.ledger, self.repo, self.seeds = ledger, repo, tuple(seeds)
        self.fit_fn = fit_fn or (lambda phys, seeds_: fit_cells(self.ledger, self.common, self.split_path, self.case, [(phys, s) for s in seeds_]))
        self.tool_error_feedback = tool_error_feedback
        init_branch_dir(self.common, self.root, case, self.public_ids)
        self.ctx = ec.open_case(cs, self.root, 'material')
        self.n = ec.COHORT_SIZE
        self.specs, self.fitted = {}, {}

    def _reject(self, message, **details):
        if self.tool_error_feedback:
            raise br.ToolInputError(message, **details)
        raise ValueError(message)

    def _reg(self) -> dict:
        return ec.load_registry(self.ctx.job_dir)

    def _spec_of(self, mid: str) -> dict:
        if mid in self.specs:
            return self.specs[mid]
        p = ec.aug_dir(self.ctx.job_dir) / (mid + '__compiled.json')
        self.specs[mid] = context.read_json(p)['material_spec'] if p.exists() else public_spec(mid)
        return self.specs[mid]

    # --- read tools
    def overview(self):
        ov = br.json_copy(self.ctx.overview())
        ov['batch_features'] = {'batch_median_' + k: v['median'] for k, v in ov['summary'].items() if v is not None}
        ov['batch_feature_definition'] = 'Median across all N T-only entity values of the field; a field with no computable value is omitted.'
        ov['field_definitions'] = FIELD_DEFINITIONS
        ov['actions'] = action_table()
        ov['consumer'] = ec.CONSUMER
        return ov

    def inspect_data(self, arguments, *, remaining_seconds):
        if set(arguments) - {'entity_indices', 'kind', 'sub_range'}:
            self._reject('unknown inspection argument')
        idx = arguments.get('entity_indices')
        if not isinstance(idx, list) or not 1 <= len(idx) <= 8 or any(type(i) != int or not 0 <= i < self.n for i in idx):
            self._reject('entity_indices must be 1..8 integers in 0..%d' % (self.n - 1))
        kind = arguments.get('kind', 'hour_profile')
        if kind not in ('hour_profile', 'daily_means', 'segment'):
            self._reject('kind must be hour_profile, daily_means or segment')
        span = arguments.get('sub_range')
        if span is not None and (not isinstance(span, list) or len(span) != 2 or any(type(x) != int for x in span) or span[0] >= span[1]):
            self._reject('sub_range must be two increasing integer boundaries')
        lo, hi = self.ctx.job.train_range
        if span is not None and (span[0] < lo or span[1] > hi):
            self._reject('sub_range must use absolute row indices inside T=[%d,%d) (overview.train_rows), not 0-based offsets' % (lo, hi))
        if kind == 'segment':
            if span is None:
                span = [hi - 336, hi]
            elif span[1] - span[0] > 336:
                self._reject('segment is limited to 336 rows')
        out = self.ctx.inspect_data(idx, kind, span)
        out['units'] = 'normalized by the frozen per-entity T scaler'
        return out

    def build_material(self, arguments, *, remaining_seconds):
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

    def inspect_material(self, plan_id, arguments, *, remaining_seconds):
        if 'plan_id' not in arguments or set(arguments) - {'plan_id', 'entity_index', 'window'}:
            self._reject('inspect_material takes plan_id and optional entity_index / window')
        ei, w = arguments.get('entity_index'), arguments.get('window', 'largest_change')
        if ei is not None and (type(ei) != int or not 0 <= ei < self.n):
            self._reject('entity_index must be an integer in 0..%d' % (self.n - 1))
        if w not in ('largest_change', 'first', 'last'):
            self._reject("window must be 'largest_change', 'first' or 'last'")
        return ec.inspect(self.ctx, plan_id, ei, w)

    # --- training
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
            before = self.ledger.s['fit_attempts']
            got = self.fit_fn(phys, missing)
            for seed in missing:
                pr = got[ec.cell_id(self.case, phys, seed)]
                if self.ledger.s['fit_attempts'] == before:
                    self.ledger.note_cache(ec.cell_id(self.case, candidate.plan_id, seed))
                if pr['job_id'] != self.case or pr['model_seed'] != seed or pr.get('material_key') != rec['key'] or not Path(pr['model_path']).exists():
                    raise RuntimeError('physical cell binding failed')
                mine = {**pr, 'cell_id': ec.cell_id(self.case, candidate.plan_id, seed), 'material_id': candidate.plan_id, 'physical_cell': pr['cell_id'], 'physical_material': phys}
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
            before = self.ledger.s['fit_attempts']
            out.append(self.evaluate(br.Candidate(mid, self._spec_of(mid), self.case), self.seeds, feedback=True, remaining_seconds=self.ledger.remaining()))
            if self.ledger.s['fit_attempts'] != before:
                raise RuntimeError('restore must not fit')
        return out

    def restore_built(self, mid):
        return br.Candidate(mid, self._spec_of(mid), self.case)

    def baselines(self):
        return tuple(self.evaluate(br.Candidate(mid, public_spec(mid), self.case), self.seeds, feedback=True, remaining_seconds=self.ledger.remaining()) for mid in self.public_ids)


def adapter_factory(common: Path, split_path: Path, cs: ec.CaseSpec, public_ids, seeds):
    def make(root, case, led, repo, *, tool_error_feedback=True, roster=None, seeds=seeds, dataset=None):
        return CaseAdapter(root, case, led, repo, common=common, split_path=split_path, cs=cs, public_ids=public_ids, tool_error_feedback=tool_error_feedback, seeds=seeds)
    return make


def fast_branch(path, case, knowledge, led, client, common, *, split_path, cs, public_ids, seeds):
    return base.branch(path, case, knowledge, led, client, None, evidence_roundtrip=EVIDENCE_ROUNDTRIP, max_tool_corrections=MAX_TOOL_CORRECTIONS,
                       tool_contracts=CONTRACTS, seeds=seeds, adapter_factory=adapter_factory(common, split_path, cs, public_ids, seeds), limits=LIMITS, dataset=cs.dataset,
                       commit_fn=commit_branch, allowed_features=ALLOWED_FEATURES, entity_count=ec.COHORT_SIZE)


def resume_fast_branch(path, case, knowledge, led, client, common, *, split_path, cs, public_ids, seeds):
    return base.resume_branch(path, case, knowledge, led, client, max_tool_corrections=MAX_TOOL_CORRECTIONS, tool_contracts=CONTRACTS, seeds=seeds,
                              adapter_factory=adapter_factory(common, split_path, cs, public_ids, seeds), evidence_roundtrip=EVIDENCE_ROUNDTRIP, limits=LIMITS,
                              dataset=cs.dataset, commit_fn=commit_branch, allowed_features=ALLOWED_FEATURES, entity_count=ec.COHORT_SIZE)


# ============================================================================= metered Fast client with the per-branch token cap
class FastClient:
    def __init__(self, client, stage_root: Path, system: str = FAST_SYSTEM):
        self.client, self.spent, self.system = client, {}, system
        for unit, row in dsks._unit_tokens(stage_root).items():
            if unit.startswith('fast:'):
                self.spent[unit[5:]] = row['prompt_tokens'] + row['completion_tokens']

    def fast(self, unit):
        def call(payload):
            if self.spent.get(unit, 0) >= FAST_TOKEN_CAP:
                raise budget.BudgetExhausted('per-branch token cap %d reached' % FAST_TOKEN_CAP)
            if not proxy_reachable():          # free TCP pre-check: no request number, no HTTP attempt, no unknown usage; the never-sent call is resumable
                raise rt.llm.TransportFault('PROXY_UNREACHABLE_PRECHECK')
            led = self.client.ledger
            before = led.s['llm_tokens_in'] + led.s['llm_tokens_out']
            try:
                return self.client.call('fast', unit, payload, self.system, max_tokens=MAX_OUTPUT_TOKENS)
            finally:
                self.spent[unit] = self.spent.get(unit, 0) + (led.s['llm_tokens_in'] + led.s['llm_tokens_out'] - before)
        return call


# ============================================================================= knowledge carriers
def knowledge_from_json(d: dict) -> br.Knowledge:
    return br.Knowledge(int(d['version']), tuple(br.Guidance(e['hook'], e['body'], copy.deepcopy(e['applicability']), tuple(e['evidence_refs'])) for e in d['entries']))


def generic_knowledge() -> br.Knowledge:
    """The frozen common-sense research guidance (task §5): identical for every domain, formed from no Source / Select result."""
    return br.Knowledge(1, (br.Guidance(dsk.HOOK, GENERIC_TEXT, {'const': True}, ('generic:frozen_task_section_5',)),))


# ============================================================================= RandomSearch_B4 (task §5.1): four frozen policies drawn from T and a policy seed
def _table_fields_with_variation(table: list) -> list:
    out = []
    for f in spec.OBS_FIELDS:
        vals = [r.get(f) for r in table if isinstance(r.get(f), (int, float)) and not isinstance(r.get(f), bool)]
        if len(vals) == len(table) and len(set(vals)) > 1:
            out.append(f)
    return out


def draw_random_supply(case: str, domain: str, case_index: int, table: list) -> dict:
    """R1 uniform explicit 1-3 step composition; R2 uniform preset edit (1-2 disabled); R3 / R4 one observation rule (field with variation in T,
    direction >= or <, threshold = median), default and rule programs drawn from all 81 legal programs, both groups >= RANDOM_MIN_GROUP entities;
    complete assignments deduplicated against the public references and earlier slots; 20 tries per slot, then the first unused uniform program
    in frozen table order. Drawn before the case's first new fit; never shown to Fast; never redrawn after an outcome."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([RANDOM_SEED_ROOT, ec.DOMAIN_INDEX.get(domain, 9), int(case_index)])))
    public_keys = {ec.material_key(ec.compile_plan_full(ec.uniform_policy(PUBLIC_STEPS[m], 'public'), table)['assignment']) for m in PUBLIC_STEPS}
    taken, plans = set(public_keys), []
    explicit = [i for i in range(1, 52) if ec.material_key([ec.PROGRAMS[i]] * ec.COHORT_SIZE) not in public_keys]      # 1-3 step compositions minus P_AmpResample
    edits = list(range(1, 29))                                                                                            # 1-2 disabled
    frozen_order = [('explicit', i) for i in range(1, 52)] + [('explicit', 52)] + [('edit', i) for i in range(1, 29)]
    fields = _table_fields_with_variation(table)

    def program_of(kind, i):
        return ec.PROGRAMS[i] if kind == 'explicit' else ec.EDIT_PROGRAMS[i]

    def uniform(kind, i, slot, fallback, tries):
        pol = ec.uniform_policy(program_of(kind, i), 'random uniform program (control arm)')
        comp = ec.compile_plan_full(pol, table)
        key = ec.material_key(comp['assignment'])
        return {'slot': slot, 'kind': 'uniform', 'program': [kind, i], 'label': ec.assignment_label(comp['assignment']), 'policy': pol, 'fallback': fallback, 'tries': tries}, key

    def first_unused_uniform(slot, tries):
        for kind, i in frozen_order:
            pol = ec.uniform_policy(program_of(kind, i), 'random uniform program (frozen-order fallback)')
            key = ec.material_key(ec.compile_plan_full(pol, table)['assignment'])
            if key not in taken:
                rec, key = uniform(kind, i, slot, True, tries)
                return rec, key
        raise RuntimeError('no unused uniform program')

    def draw_slot(slot, draw):
        for t in range(1, RANDOM_MAX_TRIES + 1):
            rec, key = draw(t)
            if rec is not None and key not in taken:
                taken.add(key)
                plans.append(rec)
                return
        rec, key = first_unused_uniform(slot, RANDOM_MAX_TRIES)
        taken.add(key)
        plans.append(rec)

    draw_slot('R1', lambda t: uniform('explicit', int(explicit[int(rng.integers(len(explicit)))]), 'R1', False, t))
    draw_slot('R2', lambda t: uniform('edit', int(edits[int(rng.integers(len(edits)))]), 'R2', False, t))

    def conditional(slot, t):
        if not fields:
            return None, None
        f = fields[int(rng.integers(len(fields)))]
        op = '>=' if int(rng.integers(2)) == 0 else '<'
        a, b = [int(x) for x in rng.choice(len(ec.LEGAL_PROGRAMS), size=2, replace=False)]
        pol = {'default': {'steps': ec.LEGAL_PROGRAMS[a]}, 'rules': [{'when': {'feature': f, 'op': op, 'value': {'quantile': 0.5}}, 'steps': ec.LEGAL_PROGRAMS[b]}],
               'rationale': 'random conditional plan (control arm)', 'observation_fields_used': [f]}
        comp = ec.compile_plan_full(pol, table)
        hit = sum(x == 0 for x in comp['rule_index'])
        if hit < RANDOM_MIN_GROUP or ec.COHORT_SIZE - hit < RANDOM_MIN_GROUP or comp['n_unknown']:
            return None, None
        return {'slot': slot, 'kind': 'conditional', 'field': f, 'op': op, 'quantile': 0.5, 'default_program': ec.program_label(ec.LEGAL_PROGRAMS[a]),
                'hit_program': ec.program_label(ec.LEGAL_PROGRAMS[b]), 'label': ec.assignment_label(comp['assignment']), 'policy': pol, 'fallback': False, 'tries': t,
                'entities_hit': hit, 'resolved_threshold': comp['resolved_thresholds']}, ec.material_key(comp['assignment'])

    draw_slot('R3', lambda t: conditional('R3', t))
    draw_slot('R4', lambda t: conditional('R4', t))
    return {'case': case, 'domain': domain, 'case_index': case_index, 'rng': 'Generator(PCG64(SeedSequence([%d, domain_index, case_index])))' % RANDOM_SEED_ROOT,
            'fields_with_variation': fields, 'drawn_local': now(), 'plans': plans, 'hidden_from_fast': True}


def _emit_factory(root: Path, case: str, trace: list):
    def emit(event, **kw):
        row = br.json_copy({'event_id': '%s:%d' % (case, len(trace)), 'event': event, **kw})
        trace.append(row)
        base.event_sink(root / 'trace.jsonl')(row)
    return emit


def random_branch(root: Path, case: str, led, common: Path, supply: dict, *, split_path, cs, public_ids, seeds) -> br.RunResult:
    """RandomSearch_B4: 0 LLM, four frozen plans evaluated (4 logical slots), delivery = argmin three-seed C_A mean over public + random;
    ties in public order then draw order."""
    root.mkdir(parents=True, exist_ok=False)
    ad = CaseAdapter(root, case, led, REPO, common=common, split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
    n, trace = ad.n, []
    emit = _emit_factory(root, case, trace)
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
    commit_branch(root, cs.dataset, case, chosen.plan_id, reason, seeds[0])
    emit('committed', plan_id=chosen.plan_id, delivery_model_ref=chosen.model_refs[0], reason=reason)
    result = br.RunResult('COMPLETE', case, 0, chosen.plan_id, chosen.model_refs[0], 0, 0, len(supply['plans']), trace)
    context.write_json(root / 'branch_result.json', asdict(result))
    print('RANDOM', case, 'commit', chosen.plan_id, flush=True)
    return result


def fixed_dev_branch(root: Path, case: str, led, common: Path, program: dict, *, split_path, cs, public_ids, seeds) -> br.RunResult:
    """Fixed_dev: the domain's frozen uniform program applied to this case without search (alias of a public reference when the assignment
    coincides; else one material + 3 fits); commit = that plan."""
    root.mkdir(parents=True, exist_ok=False)
    ad = CaseAdapter(root, case, led, REPO, common=common, split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
    trace = []
    emit = _emit_factory(root, case, trace)
    context.write_json(root / 'knowledge.json', asdict(br.Knowledge()))
    cands = {c.plan_id: c for c in ad.baselines()}
    emit('job_started', mode='fixed_dev', baseline_ids=list(cands), seeds=list(seeds), frozen_program=program)
    steps = program['steps']
    key = ec.material_key([steps] * ec.COHORT_SIZE)
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
    commit_branch(root, cs.dataset, case, chosen.plan_id, reason, seeds[0], extra={'fixed_dev': program, 'alias_of_public': alias})
    emit('committed', plan_id=chosen.plan_id, delivery_model_ref=chosen.model_refs[0], reason=reason)
    result = br.RunResult('COMPLETE', case, 0, chosen.plan_id, chosen.model_refs[0], 0, 0, 0 if alias else 1, trace)
    context.write_json(root / 'branch_result.json', asdict(result))
    print('FIXED_DEV', case, 'commit', chosen.plan_id, flush=True)
    return result


# ============================================================================= budget (one package ledger, cumulative stage caps, per-stage retries)
def stage_caps(root: Path, stage: str) -> dict:
    P_ = paths(root)
    rec = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    if stage in rec:
        return rec[stage]
    led = rt.RuntimeLedger(P_['ledger'], **TOTAL)
    s, alloc = led.s, ALLOC[stage]
    snap = {k: s.get(k, 0) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts',
                                     'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'fit_wall_seconds')}
    snap['elapsed_s'] = time.time() - s['started_epoch']
    retries_stage = min(STAGE_RETRIES, TOTAL['max_retries'] - s['retries_used']) if alloc['fits'] else 0
    caps = {'max_fit_attempts': min(TOTAL['max_fit_attempts'], s['fit_attempts'] + alloc['fits'] + retries_stage),
            'max_llm_requests': min(TOTAL['max_llm_requests'], s['llm_requests'] + alloc['requests']),
            'max_llm_tokens': min(TOTAL['max_llm_tokens'], s['llm_tokens_in'] + s['llm_tokens_out'] + alloc['tokens']),
            'max_wall_s': TOTAL['max_wall_s'], 'max_retries': min(TOTAL['max_retries'], s['retries_used'] + retries_stage)}
    rec[stage] = {'epoch_start': time.time(), 'local_start': now(), 'allocation': alloc, 'snapshot_at_start': snap, 'caps': caps,
                  'http_cap': min(HTTP_CAP, s['llm_http_attempts'] + alloc['requests'] + 4)}
    context.write_json(P_['stages'], rec)
    return rec[stage]


def start_paid_clock(root: Path) -> None:
    """The 8 h wall cap counts paid running time: the clock is restarted once when the first paid stage (source) begins, before any LLM request."""
    P_ = paths(root)
    led = rt.RuntimeLedger(P_['ledger'], **TOTAL)
    if led.s.get('paid_clock_started_epoch') is None:
        if led.s['llm_requests']:
            raise RuntimeError('paid requests exist before the paid clock start')
        led.s['wiring_elapsed_before_paid_clock_s'] = time.time() - led.s['started_epoch']
        led.s['started_epoch'] = time.time()
        led.s['paid_clock_started_epoch'] = led.s['started_epoch']
        led.s['paid_clock_started_local'] = now()
        led._save()


def numeric_seconds(root: Path) -> float:
    led = context.read_json(paths(root)['ledger'])
    return float(led.get('fit_wall_seconds', 0.0)) + float(led.get('label_wall_seconds', 0.0)) + float(led.get('material_wall_seconds', 0.0))


# ============================================================================= stage worker (subprocess; labels withheld on any stop)
def label_stage(branch_dir: Path, case: str, name: str, led, stage_root: Path, split_path: Path) -> None:
    t0 = time.time()
    rc, secs = run_sub(['--worker-label', name, branch_dir, split_path, case, stage_root], stage_root / 'logs' / ('label_%s_%s.log' % (name, branch_dir.name)), max(1.0, led.remaining()))
    led.s['label_wall_seconds'] = led.s.get('label_wall_seconds', 0.0) + (time.time() - t0)
    led._save()
    if rc:
        raise RuntimeError('label stage failed: %s %s' % (name, branch_dir.name))


def _safe_message(exc) -> str | None:
    return None if isinstance(exc, (rt.llm.AccountFault, rt.llm.TransportFault)) else str(exc)[:300]


def open_labels(root: Path, cfg: dict, branches, failures, led, *, withhold_reason=None, resumed=False) -> None:
    split_path = Path(cfg['split_path'])
    if withhold_reason:
        context.write_json(root / 'labels_withheld.json', {'epoch': time.time(), 'reason': withhold_reason, 'branches': [(str(p), j) for p, j in branches],
                                                          'unknown_usage': led.s['llm_tokens_unknown'], 'next': '--resume-stage (operator decision) or stop'})
        print('LABELS_WITHHELD', withhold_reason, flush=True)
        return
    eligible = [(p, j) for p, j in branches if (p / j / 'commit.json').exists()]
    try:
        for p, j in eligible:
            if not (p / j / 'c_b_scores.json').exists():
                label_stage(p, j, 'c_b', led, root, split_path)
        if not cfg['labels_e']:
            context.write_json(root / 'labels_c_b_only.json', {'epoch': time.time(), 'branches': [str(p) for p, j in eligible], 'resumed': resumed})
        else:
            for p, j in eligible:
                if not (p / j / 'e_frozen.json').exists():
                    label_stage(p, j, 'freeze_e', led, root, split_path)
            context.write_json(root / 'all_e_predictions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in eligible], 'resumed': resumed})
            for p, j in eligible:
                if not (p / j / 'e_scores.json').exists():
                    label_stage(p, j, 'score_e', led, root, split_path)
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'external', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
        print('EXTERNAL_STOP', type(exc).__name__, flush=True)
    context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': resumed})


def copy_physical_cache(src_job_dir: Path, common: Path, case: str, cs: ec.CaseSpec, *, seeds) -> dict:
    """The wiring's healthy cache of a Source case (T objects, materials, C_A-only physical cells) copied into the Source stage cache after binding
    checks; nothing of the wiring is written; no label or delivery file may exist in a physical cache."""
    dst = common / case
    if (dst / 'aug_materials' / 'index.json').exists():
        return {'copied': False, 'reason': 'cache already present'}
    src_job_dir = Path(src_job_dir)
    if not (src_job_dir / 'aug_materials' / 'index.json').exists():
        return {'copied': False, 'reason': 'no source cache'}
    ec._check_binding(context.read_json(src_job_dir / 'case_spec.json'), cs)
    forbidden = [p.name for p in src_job_dir.rglob('*') if p.name in ('c_b_scores.json', 'e_scores.json', 'e_frozen.json', 'commit.json') or p.name.startswith('predictions_e')]
    if forbidden:
        raise RuntimeError('source cache carries labels or deliveries: %s' % forbidden[:3])
    dst.mkdir(parents=True, exist_ok=True)
    for name in ('scaler.npz', 'overview.json', 'case_spec.json'):
        shutil.copy2(src_job_dir / name, dst / name)
    shutil.copytree(src_job_dir / 'aug_materials', dst / 'aug_materials')
    reg = ec.load_registry(dst)
    for m, r in reg.items():
        for k in ('path', 'summary_path'):
            if r.get(k):
                r[k] = str(dst / 'aug_materials' / Path(r[k]).name)
    ec.save_registry(dst, reg)
    for d in ('runs', 'cells', 'pred_c_a'):
        (dst / d).mkdir(exist_ok=True)
    n = 0
    for cp in sorted((src_job_dir / 'cells').glob('*.json')) if (src_job_dir / 'cells').exists() else []:
        rec = context.read_json(cp)
        if rec.get('status') != 'OK' or rec.get('tag') or rec['model_seed'] not in seeds or rec.get('profile') != ec.PROFILE_VERSION or rec.get('n_updates') != spec.N_UPDATES:
            continue
        if set(rec.get('scores', {})) - {'c_a'}:
            raise RuntimeError('physical cell %s carries a label block' % rec['cell_id'])
        mp = Path(rec['model_path'])
        shutil.copy2(mp, dst / 'runs' / mp.name)
        rec['model_path'] = str(dst / 'runs' / mp.name)
        rec['copied_from_wiring_cache'] = str(cp)
        context.write_json(dst / 'cells' / cp.name, rec)
        pc = src_job_dir / 'pred_c_a' / (rec['cell_id'] + '.npz')
        if pc.exists():
            shutil.copy2(pc, dst / 'pred_c_a' / pc.name)
        n += 1
    out = {'copied': True, 'source': str(src_job_dir), 'materials': sorted(reg), 'cells': n, 'copied_local': now()}
    context.write_json(dst / 'cache_copied_from.json', out)
    return out


def prepare_common(root: Path, case: str, led, cfg: dict) -> Path:
    """Public references of one case: (wiring cache copy) -> build (0 fits) -> random supply frozen (MetaTest) -> reference fits."""
    common = root / (case + '_common')
    cs = ec.cases_from_split(context.read_json(cfg['split_path']))[case]
    seeds, public_ids, split_path = tuple(cfg['seeds']), tuple(cfg['public_ids']), Path(cfg['split_path'])
    src = (cfg.get('cache_from') or {}).get(case)
    if src and not (common / case / 'aug_materials' / 'index.json').exists():
        copy_physical_cache(Path(src), common, case, cs, seeds=seeds)
    if not all(m in ec.load_registry(common / case) for m in public_ids):
        t0 = time.time()
        rc, _ = run_sub(['--worker-build', common, split_path, case], root / 'logs' / ('build_%s.log' % case), 1800)
        led.s['material_wall_seconds'] = led.s.get('material_wall_seconds', 0.0) + (time.time() - t0)
        led._save()
        if rc != 0:
            raise RuntimeError('common build failed for %s' % case)
    if cfg.get('random') and not (common / 'random_supply.json').exists():
        table = context.read_json(common / case / 'overview.json')['entities']
        dsks.write_once(common / 'random_supply.json', draw_random_supply(case, cs.domain, cs.case_index, table))
    got = fit_cells(led, common, split_path, case, [(m, s) for m in public_ids for s in seeds])
    for rec in got.values():
        if rec['scores']['c_a']['status'] != 'SCORABLE':
            raise RuntimeError('C_A_NOT_SCORABLE: %s' % rec['cell_id'])
    return common


def _run_arms(root: Path, cfg: dict, led, client, branches, failures, *, resumed: bool) -> None:
    fc = FastClient(client, root) if client is not None else None
    cases = ec.cases_from_split(context.read_json(cfg['split_path']))
    split_path = Path(cfg['split_path'])
    for case in cfg['cases']:
        cs = cases[case]
        seeds, public_ids = tuple(cfg['seeds']), tuple(cfg['public_ids'])
        common = prepare_common(root, case, led, cfg)
        for arm in cfg['order'][case]:
            path = root / ('%s_%s' % (case, arm))
            kn = cfg['knowledge'].get(case, {}).get(arm)
            if arm not in CONTROL_ARMS and kn is None:
                if not (root / ('%s_%s_treatment.json' % (case, arm))).exists():
                    context.write_json(root / ('%s_%s_treatment.json' % (case, arm)), (cfg.get('treatment_note') or {}).get(arm) or
                                       {'status': 'NO_TREATMENT', 'note': 'no frozen card for this arm; not run, no substitute, no budget transfer'})
                continue
            if arm == 'fixed_dev' and not (cfg.get('fixed_dev') or {}).get(cs.domain):
                if not (root / ('%s_%s_treatment.json' % (case, arm))).exists():
                    context.write_json(root / ('%s_%s_treatment.json' % (case, arm)), {'status': 'NO_TREATMENT', 'note': 'no Fixed_dev program frozen for this domain'})
                continue
            branches.append((path, case))
            try:
                if path.exists() and (path / 'branch_result.json').exists():
                    prior = context.read_json(path / 'branch_result.json')
                    if prior['status'] == 'COMPLETE':
                        result = br.RunResult(**{k: v for k, v in prior.items() if k != 'trace'})
                    elif arm in CONTROL_ARMS:
                        raise RuntimeError('control branch cannot be resumed')
                    else:
                        result = resume_fast_branch(path, case, knowledge_from_json(kn), led, fc, common, split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
                elif path.exists():
                    raise RuntimeError('branch directory without a result; inspect before any replay')
                elif arm == 'random':
                    result = random_branch(path, case, led, common, context.read_json(common / 'random_supply.json'), split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
                elif arm == 'fixed_dev':
                    result = fixed_dev_branch(path, case, led, common, cfg['fixed_dev'][cs.domain], split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
                else:
                    result = fast_branch(path, case, knowledge_from_json(kn), led, fc, common, split_path=split_path, cs=cs, public_ids=public_ids, seeds=seeds)
                if result.status != 'COMPLETE':
                    failures.append({'branch': path.name, 'kind': result.failure_kind, 'reason': result.reason})
            except Exception as exc:  # noqa: BLE001
                failures.append({'branch': path.name, 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
                print('BRANCH_FAILED', path.name, type(exc).__name__, _safe_message(exc), flush=True)
            led.check_wall()
            if client is not None:
                led.check_llm()
                if getattr(client, 'fatal', False) or rt.unknown_usage_blocks(led):
                    raise RuntimeError('backend fatal or unknown usage')
                if arm not in CONTROL_ARMS and not proxy_reachable():
                    raise RuntimeError('LLM proxy unreachable; stage stopped with labels withheld (resume when the proxy is back)')


def stage_worker(cfg_path: Path, *, client=None) -> None:
    cfg = context.read_json(cfg_path)
    root = Path(cfg['output'])
    if (root / 'experiment_started.json').exists():
        raise RuntimeError('stage already started; no paid replay')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    context.write_json(root / 'config.json', cfg)
    led = rt.RuntimeLedger(cfg['ledger_path'], **cfg['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage; paid stage refused without an operator decision')
    rt.FIT_RETRY = bool(cfg.get('fit_retry'))
    context.write_json(root / 'experiment_started.json', {'epoch': time.time(), 'pid': os.getpid(), 'stage': cfg['stage']})
    branches, failures, stopped = [], [], None
    try:
        if client is None and cfg.get('llm'):
            client = rt.MeteredClient(led, root / 'raw_responses', http_cap=int(cfg['http_cap']))     # config only; no unmetered ping
        _run_arms(root, cfg, led, client, branches, failures, resumed=False)
        context.write_json(root / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches]})
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
        stopped = type(exc).__name__
        print('EXECUTION_STOP', stopped, _safe_message(exc), flush=True)
    finally:
        context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures})
    open_labels(root, cfg, branches, failures, led, withhold_reason=stopped and 'execution stopped before every planned arm was attempted (%s)' % stopped)


def resume_stage(root: Path, stage: str, *, accept_unknown_usage: bool = False, restart=()) -> None:
    """One operator continuation: labels still withheld (existing boundary check), completed branches kept, a never-sent Fast call continued,
    missing arms run as planned, then labels. Unknown usage needs an explicit acceptance."""
    from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp
    sroot = paths(root)[stage]
    cfg = context.read_json(sroot / 'config.json')
    n_prev = len(list(sroot.glob('resume*.json')))
    if (sroot / 'experiment_started.json').exists() and not (sroot / 'execution_finished.json').exists() and not (sroot / 'labels_withheld.json').exists():
        # the stage process was stopped (operator / crash) before it could finish: labels were never opened; record the withheld state first
        context.write_json(sroot / 'labels_withheld.json', {'epoch': time.time(), 'reason': 'stage process stopped before finishing (operator stop / crash); no label was opened', 'written_by': 'resume_stage'})
    blocked = rtp.labels_boundary_violations(sroot)
    if blocked:
        raise RuntimeError('resume refused; labels are no longer withheld: ' + '; '.join(blocked))
    led = rt.RuntimeLedger(cfg['ledger_path'], **cfg['caps'])
    if rt.unknown_usage_blocks(led):
        if not accept_unknown_usage:
            raise RuntimeError('ledger holds unknown usage; pass --accept-unknown-usage to record an explicit operator decision')
        led.s['unknown_usage_accepted'] = led.s['llm_tokens_unknown']
        led._save()
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
        client = rt.MeteredClient(led, sroot / 'raw_responses', http_cap=int(cfg['http_cap'])) if cfg.get('llm') else None
        _run_arms(sroot, cfg, led, client, branches, failures, resumed=True)
        context.write_json(sroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches], 'resumed': True})
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
        stopped = type(exc).__name__
        print('EXECUTION_STOP', stopped, _safe_message(exc), flush=True)
    finally:
        context.write_json(sroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': True})
    open_labels(sroot, cfg, branches, failures, led, resumed=True, withhold_reason=stopped and 'resumed stage stopped again (%s)' % stopped)


def stage_config(root: Path, stage: str, *, cases, order, knowledge, labels_e: bool, llm: bool, random: bool, treatment_note=None, cache_from=None, fixed_dev=None) -> Path:
    P_ = paths(root)
    path = P_['configs'] / ('%s.json' % stage)
    if path.exists():
        return path
    sc = stage_caps(root, stage)
    cfg = {'study': 'domain_aug_entity_split', 'status': 'FROZEN', 'stage': stage, 'output': str(P_[stage]), 'cases': list(cases), 'split_path': str(P_['split']),
           'order': {c: list(order[c]) for c in cases}, 'knowledge': knowledge, 'labels_e': bool(labels_e), 'llm': bool(llm), 'random': bool(random),
           'seeds': list(SEEDS), 'public_ids': list(PUBLIC), 'limits': LIMITS, 'max_tool_corrections': MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': EVIDENCE_ROUNDTRIP,
           'fast_token_cap': FAST_TOKEN_CAP, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'fit_threads': FIT_THREADS, 'fit_parallel': FIT_PARALLEL, 'caps': sc['caps'],
           'http_cap': sc['http_cap'], 'ledger_path': str(P_['ledger']), 'fit_retry': True, 'fit_attempts_at_stage_start': context.read_json(P_['ledger'])['fit_attempts'],
           'treatment_note': treatment_note or {}, 'cache_from': {k: str(v) for k, v in (cache_from or {}).items()}, 'fixed_dev': fixed_dev or {}, 'frozen_local': now()}
    dsks.write_once(path, cfg)
    return path


def run_stage(root: Path, stage: str, cfg_path: Path) -> dict:
    P_ = paths(root)
    sroot = P_[stage]
    if not (sroot / 'experiment_started.json').exists():
        led = context.read_json(P_['ledger'])
        remaining = TOTAL['max_wall_s'] - (time.time() - led['started_epoch'])
        if remaining <= 0:
            raise RuntimeError('package wall exhausted')
        P_['logs'].mkdir(parents=True, exist_ok=True)
        print('STAGE_START', stage, now(), flush=True)
        with (P_['logs'] / ('%s.log' % stage)).open('a', encoding='utf-8') as log:
            p = subprocess.Popen([sys.executable, '-B', '-m', MODULE, '--stage-worker', str(cfg_path)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=worker_env())
            try:
                rc = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)
                context.write_json(root / ('%s_supervisor_timeout.json' % stage), {'epoch': time.time(), 'pid': p.pid})
                rc = 124
        print('STAGE_EXIT', stage, rc, now(), flush=True)
    return dsks.stage_status(sroot)


# ============================================================================= preflight (0 cost) and frozen configuration
def environment() -> dict:
    code = 'import sys, torch, numpy, pandas; print(sys.version.split()[0], torch.__version__, numpy.__version__, pandas.__version__, torch.get_num_threads())'
    got = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, env=worker_env()).stdout.split()
    return {'python': got[0], 'torch': got[1], 'numpy': got[2], 'pandas': got[3], 'default_threads': int(got[4]), 'fit_threads': FIT_THREADS, 'fit_parallel': FIT_PARALLEL,
            'executable': sys.executable, 'KMP_DUPLICATE_LIB_OK_at_start': KMP_AT_START, 'workers_env_kmp_removed': True, 'cpu_count': os.cpu_count()}


def preflight(root: Path = ROOT) -> dict:
    """Frozen before the first fit: the partition copy and its check, cases, geometry, Consumer, environment, model, prompts, program tables, seed
    rules, quotas, orders, selection and aggregation formulas. Reads no data row."""
    root.mkdir(parents=True, exist_ok=True)
    P_ = paths(root)
    if not P_['split'].exists():
        shutil.copy2(SPLIT_DOC, P_['split'])
    split = context.read_json(P_['split'])
    if split != context.read_json(SPLIT_DOC):
        raise RuntimeError('the frozen split copy differs from the document')
    p = root / 'frozen_config.json'
    if p.exists():
        return context.read_json(p)
    chk = ec.check_split(split)
    if not chk['ok']:
        raise RuntimeError('partition check failed: %s' % chk)
    cases = ec.cases_from_split(split)
    for cs in cases.values():
        if not cs.path().exists():
            raise RuntimeError('data file missing for %s: %s' % (cs.case_id, cs.path()))
    cfg = {'package': PACKAGE, 'task': TASK, 'split_document': str(SPLIT_DOC), 'split_setting_id': split['setting_id'], 'split_check': chk, 'exposure': EXPOSURE,
           'domains': {d: {'dataset': split['domains'][d]['dataset'], 'file': Path(spec.DATASETS[split['domains'][d]['dataset']]['path']).name,
                           'columns_in_file': split['domains'][d]['columns_in_file'], 'partition_seed': split['domains'][d]['partition_seed']} for d in DOMAINS},
           'cases': {c: cs.to_json() for c, cs in cases.items()}, 'cohort_size': ec.COHORT_SIZE, 'geometry': {'L': spec.L, 'H': spec.H, 'T': spec.TRAIN_SPAN, 'n_parents': spec.N_PARENTS,
                                                                                                           'pool': ec.COHORT_SIZE * spec.N_PARENTS, 'anchors': split['anchors']},
           'consumer': ec.CONSUMER, 'seeds': list(SEEDS), 'model': MODEL, 'environment': environment(), 'limits': LIMITS, 'fast_token_cap': FAST_TOKEN_CAP,
           'max_output_tokens': MAX_OUTPUT_TOKENS, 'max_tool_corrections': MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'total': TOTAL, 'http_cap': HTTP_CAP,
           'allocation': ALLOC, 'stage_retries': STAGE_RETRIES, 'fast_system': FAST_SYSTEM, 'slow_system': SLOW_FORM, 'generic_text': GENERIC_TEXT, 'contracts': CONTRACTS,
           'public_references': {m: public_spec(m) for m in PUBLIC}, 'program_tables': {'explicit': [ec.program_label(p) for p in ec.PROGRAMS], 'edit': [ec.program_label(p) for p in ec.EDIT_PROGRAMS]},
           'seed_rule': 'SeedSequence([%d, domain_index, case_index, program_index, int(entity_column)]); preset and every edit use program_index %d' % (ec.SEED_ROOT, ec.PRESET_INDEX),
           'random_search': {'seed_root': RANDOM_SEED_ROOT, 'max_tries': RANDOM_MAX_TRIES, 'min_group': RANDOM_MIN_GROUP, 'quantile': 0.5, 'programs_pool': len(ec.LEGAL_PROGRAMS),
                             'commit': 'argmin three-seed C_A mean over public + R1..R4; ties in public order then R1..R4'},
           'orders': {'select': SELECT_ORDER, 'metatest': TEST_ORDER, 'controls_after_fast': list(CONTROL_ARMS)},
           'selection': {'J': 'J(W) = mean over the two Select cases of [mean_seed E(actual commit of W) / mean_seed E(None)]; lowest wins; ties <= 1e-12 by fewer new evaluations, fewer tokens, W1/W2/W3',
                         'fixed_dev': 'among uniform programs actually evaluated on BOTH Select cases (any branch, incl. public), the lowest J of the same form; ties in public order then table order',
                         'no_card': 'F_domain is an alias of F0 for that domain (same trajectory and scores)'},
           'readout': {'metric': 'E normalized MSE (T scaler), entity macro, then seed mean; domain = equal-weight 2 cases; overall = equal-weight 2 domains',
                       'delta': 'Delta_j(A,B) = 100 x [E_j(A) - E_j(B)] / E_j(None) per seed, positive = B better; denominator = the case three-seed None mean'},
           'body_limit': BODY_LIMIT, 'max_candidates': MAX_CANDIDATES, 'compression_rules': COMPRESSION_RULES, 'frozen_local': now(), 'no_sha': True}
    dsks.write_once(p, cfg)
    return cfg


# ============================================================================= wiring acceptance (S01 of each domain, scripted client, <= 12 fits, 0 LLM, C_A only)
class ScriptedClient:
    """The Fast tool loop driven by a fixed action list; ledger-free (no LLM request is charged)."""

    def __init__(self, steps: list):
        self.steps = list(steps)
        self.ledger = None

    def fast(self, unit):
        def call(payload):
            if not self.steps:
                raise RuntimeError('scripted client exhausted')
            return self.steps.pop(0)
        return call


def wiring(root: Path = ROOT) -> dict:
    P_ = paths(root)
    out = P_['wiring'] / 'wiring_result.json'
    if out.exists():
        return context.read_json(out)
    preflight(root)
    sc = stage_caps(root, 'wiring')
    led = rt.RuntimeLedger(P_['ledger'], **sc['caps'])
    rt.FIT_RETRY = True
    P_['wiring'].mkdir(parents=True, exist_ok=True)
    (P_['wiring'] / 'logs').mkdir(exist_ok=True)
    cases = cases_of(root)
    res = {'status': 'PASS', 'started_local': now(), 'cases': {}, 'checks': []}
    for dom in DOMAINS:
        case = '%s_S01' % dom
        cs = cases[case]
        common = P_['wiring'] / (case + '_common')
        cfg = {'split_path': str(P_['split']), 'seeds': list(WIRING_SEEDS), 'public_ids': ['None'], 'random': False}
        before = led.s['fit_attempts']
        prepare_common(P_['wiring'], case, led, cfg)                      # None: 3 fits (no material)
        reg = ec.load_registry(common / case)
        res['checks'].append({'case': case, 'check': 'public materials built without fits', 'ok': all(m in reg for m in PUBLIC), 'materials': sorted(reg)})
        with np.load(common / case / 'scaler.npz') as z:
            res['checks'].append({'case': case, 'check': 'scaler bound to the case', 'ok': [str(x) for x in z['roster']] == list(cs.roster) and int(z['t']) == cs.t and str(z['dataset']) == cs.dataset})
        spec_rec = context.read_json(common / case / 'case_spec.json')
        res['checks'].append({'case': case, 'check': 'only the roster columns were converted', 'ok': spec_rec['columns_converted'] == ec.COHORT_SIZE and spec_rec['columns_in_file'] > ec.COHORT_SIZE,
                              'columns_in_file': spec_rec['columns_in_file']})
        # a scripted Fast: overview -> build the explicit composition -> inspect -> evaluate -> compare -> commit
        steps = [{'actions': [{'tool': 'overview', 'arguments': {}}]},
                 {'actions': [{'tool': 'build_material', 'arguments': {'plan_id': 'WireComp', 'policy': ec.uniform_policy(WIRING_COMPOSITION, 'wiring: explicit 3-step composition')}}]},
                 {'actions': [{'tool': 'inspect_material', 'arguments': {'plan_id': 'WireComp', 'entity_index': 0}}]},
                 {'actions': [{'tool': 'evaluate', 'arguments': {'plan_id': 'WireComp'}}]},
                 {'actions': [{'tool': 'compare', 'arguments': {'a': 'None', 'b': 'WireComp'}}]},
                 {'actions': [{'tool': 'commit', 'arguments': {'plan_id': 'None', 'reason': 'wiring acceptance: the public reference is kept'}}]}]
        bpath = P_['wiring'] / ('%s_wiring' % case)
        result = fast_branch(bpath, case, br.Knowledge(), led, ScriptedClient(steps), common, split_path=P_['split'], cs=cs, public_ids=('None',), seeds=WIRING_SEEDS)
        fits = led.s['fit_attempts'] - before
        cells = ec.branch_cells(bpath / case)
        comp = [c for c in cells.values() if c['material_id'] == 'WireComp']
        pool_ok = all(c['pool_size'] == ec.COHORT_SIZE * spec.N_PARENTS and c['entity_count'] == ec.COHORT_SIZE for c in cells.values())
        threads_ok = all(c['torch_threads'] == FIT_THREADS for c in cells.values())
        phys = ec.load_registry(common / case)
        ta_ids = [m for m in phys if m.startswith('TA')]
        summ = context.read_json(phys[ta_ids[0]]['summary_path']) if ta_ids else {}
        rows = [json.loads(x) for x in (bpath / 'trace.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        ov = next(r['overview'] for r in rows if r['event'] == 'job_started')
        leak = [n for n in tuple(spec.DATASETS) if n in json.dumps(ov).lower()]
        no_labels = not any(p.name in ('c_b_scores.json', 'e_frozen.json', 'e_scores.json') for p in (P_['wiring']).rglob('*.json'))
        res['cases'][case] = {'branch_status': result.status, 'committed': result.committed_plan_id, 'fits_charged': fits, 'cells': len(cells),
                              'comp_c_a_by_seed': [c['scores']['c_a']['normalized_mse_macro'] for c in sorted(comp, key=lambda c: c['model_seed'])],
                              'none_c_a_by_seed': [c['scores']['c_a']['normalized_mse_macro'] for c in sorted((c for c in cells.values() if c['material_id'] == 'None'), key=lambda c: c['model_seed'])],
                              'fit_seconds': [round(c['train']['seconds'], 1) for c in cells.values()], 'material_build_seconds': summ.get('build_seconds'),
                              'material_steps_executed': summ.get('per_step'), 'pool_and_entity_count_ok': pool_ok, 'threads_fixed_ok': threads_ok,
                              'overview_names_no_source': not leak, 'no_label_file_in_wiring': no_labels, 'rows_read_by_fits': sorted({tuple(c['rows_read']) for c in cells.values()})}
        ok = result.status == 'COMPLETE' and fits == 6 and len(comp) == 3 and pool_ok and threads_ok and not leak and no_labels and result.committed_plan_id == 'None'
        res['checks'].append({'case': case, 'check': 'scripted Fast loop: 6 fits (None + composition x 3 seeds), commit None, no leak, no label', 'ok': ok})
    res['fits_total'] = led.s['fit_attempts']
    res['fit_wall_seconds'] = led.s['fit_wall_seconds']
    res['status'] = 'PASS' if all(c['ok'] for c in res['checks']) and led.s['fit_attempts'] <= ALLOC['wiring']['fits'] else 'FAIL'
    res['finished_local'] = now()
    dsks.write_once(out, res)
    print('WIRING', res['status'], 'fits', res['fits_total'], flush=True)
    return res


# ============================================================================= formation: per-domain census (deterministic) + ONE Slow call
def _trace(bdir: Path) -> list:
    tr = bdir / 'trace.jsonl'
    return [json.loads(x) for x in tr.read_text(encoding='utf-8').splitlines() if x.strip()] if tr.exists() else []


def _by_seed(cells_or_scores: dict, mid: str, block: str, seeds) -> tuple:
    vals = {}
    for c, v in cells_or_scores.items():
        if v['material_id'] == mid and block in v:
            vals[v['model_seed']] = v[block]
    if set(vals) < set(seeds) or any(vals[s]['status'] != 'SCORABLE' for s in seeds):
        return None, None
    return ([vals[s]['normalized_mse_macro'] for s in seeds],
            [[statistics.fmean(vals[s]['per_origin_entity_normalized_mse'][o]) for o in range(vals[s]['n_origins'])] for s in seeds])


def branch_evidence(bdir: Path, case: str, stage_name: str, tier, *, with_e: bool, seeds=SEEDS) -> tuple:
    """(record, refs, materials keyed by physical id) of one branch; E only when the caller authorizes it (Source census after the stage's E)."""
    res = context.read_json(bdir / 'branch_result.json') if (bdir / 'branch_result.json').exists() else None
    kn = context.read_json(bdir / 'knowledge.json') if (bdir / 'knowledge.json').exists() else {'entries': []}
    reg = ec.load_registry(bdir / case)
    cells = ec.branch_cells(bdir / case)
    ca_view = {c: {'material_id': r['material_id'], 'model_seed': r['model_seed'], 'c_a': r['scores']['c_a']} for c, r in cells.items()}
    cb = context.read_json(bdir / case / 'c_b_scores.json')['cells'] if (bdir / case / 'c_b_scores.json').exists() else {}
    es = context.read_json(bdir / case / 'e_scores.json')['cells'] if (bdir / case / 'e_scores.json').exists() else {}
    if es and not with_e:
        raise PermissionError('E present but not authorized for this census: %s' % bdir.name)
    commit = context.read_json(bdir / case / 'commit.json') if (bdir / case / 'commit.json').exists() else None
    plan_to_phys, materials = {}, {}
    for mid, r in reg.items():
        phys = r.get('phys_id', mid)
        plan_to_phys[mid] = phys
        asg = 'FixedMixup' if mid == 'FixedMixup' else (r.get('assignment') if mid not in PUBLIC else [PUBLIC_STEPS[mid]] * ec.COHORT_SIZE)
        row = materials.setdefault(phys, {'label': ec.assignment_label(asg, mid if mid in PUBLIC else None), 'public_id': mid if mid in PUBLIC else None,
                                          'uniform_steps': ec.uniform_steps(asg) if isinstance(asg, list) else None,
                                          'assignment': None if (mid in PUBLIC or ec.uniform_steps(asg) is not None) else asg,
                                          'c_a_by_seed': None, 'c_b_by_seed': None, 'e_by_seed': None, 'e_status': 'unavailable'})
        for blk, src in (('c_a', ca_view), ('c_b', cb), ('e', es)):
            if row[blk + '_by_seed'] is None:
                v, vo = _by_seed(src, mid, blk, seeds)
                if v is not None:
                    row[blk + '_by_seed'], row[blk + '_by_seed_origin'] = v, vo
                    if blk == 'e':
                        row['e_status'] = 'development_late'
    rows = _trace(bdir)
    refs = [ref(stage_name, bdir.name, r['event_id']) for r in rows] + [ref(stage_name, bdir.name, 'overview'), ref(stage_name, bdir.name, 'labels_after_commit')]
    traj = LL.compress_trajectory(rows, tier)
    for r in traj:
        r['ref'] = ref(stage_name, bdir.name, r['event_id'])
    tokens = dsks._unit_tokens(bdir.parent).get('fast:%s' % bdir.name, {}) if (bdir.parent / 'raw_responses').exists() else {}
    rec = {'ref_prefix': ref(stage_name, bdir.name, ''), 'case_ref': case, 'knowledge_loaded': [e_['body'] for e_ in kn.get('entries', [])] or None,
           'status': res['status'] if res else 'NOT_RUN', 'failure_kind': res.get('failure_kind') if res else None, 'reason': res.get('reason') if res else None,
           'committed_plan_id': res.get('committed_plan_id') if res else None, 'committed_physical_id': plan_to_phys.get(res.get('committed_plan_id')) if res else None,
           'commit_reason': commit['reason'] if commit else None, 'plan_to_physical': plan_to_phys, 'plan_specs': {},
           'cost': {'fast_calls': res['calls'] if res else None, 'tool_calls': res['tool_calls'] if res else None, 'new_evaluations': res['new_evaluations'] if res else None, 'tokens': tokens},
           'trajectory': traj}
    for mid in reg:
        p = ec.aug_dir(bdir / case) / (mid + '__compiled.json')
        if p.exists():
            ms = context.read_json(p)['material_spec']
            rec['plan_specs'][mid] = {k: v for k, v in ms.items() if k not in ('execution_order', 'profile', 'alias_of')}
    return rec, refs, materials


def _post_hoc(materials: dict, committed_phys: str | None, seeds=SEEDS) -> dict:
    """Deterministic diagnostic: C_A argmin over the evaluated pool, E best over the pool, the commit's E ratio to None; never an answer."""
    def mean(v):
        return statistics.fmean(v) if v else None
    with_ca = {p: mean(m['c_a_by_seed']) for p, m in materials.items() if m.get('c_a_by_seed')}
    with_e = {p: mean(m['e_by_seed']) for p, m in materials.items() if m.get('e_by_seed')}
    none_e = with_e.get('None')
    ca_argmin = min(with_ca, key=lambda p: (with_ca[p], list(materials).index(p))) if with_ca else None
    e_best = min(with_e, key=lambda p: (with_e[p], list(materials).index(p))) if with_e else None
    return {'note': 'DIAGNOSTIC: post-hoc pool readings; the E best is not a known deployment answer',
            'c_a_argmin_physical': ca_argmin, 'e_best_physical_diagnostic': e_best, 'commit_physical': committed_phys,
            'commit_is_c_a_argmin': (ca_argmin == committed_phys) if committed_phys else None,
            'e_ratio_to_none': {p: (with_e[p] / none_e if none_e else None) for p in with_e} if none_e else None,
            'labels': {p: m['label'] for p, m in materials.items()}}


def census(root: Path, domain: str, tier, *, purpose: str = 'formation') -> dict:
    P_ = paths(root)
    sroot = P_['source']
    cases, refs, records = {}, [], []
    for case in case_ids(root, 'source', domain):
        bdir = sroot / ('%s_f0' % case)
        if not (bdir / 'branch_result.json').exists():
            records.append({'branch': ref('source', bdir.name, ''), 'status': 'NOT_RUN'})
            continue
        rec, r_, mats = branch_evidence(bdir, case, 'source', tier, with_e=True)
        refs += r_
        ov = context.read_json(bdir / case / 'overview.json')
        cases[case] = {'cut_position': 'anchor_%d' % (1 + (ov['train_rows'][1] == load_split(root)['anchors'][1])), 'train_rows': ov['train_rows'],
                       'overview': {'n_entities': ov['n_entities'], 'summary': ov['summary'], 'entities': W._columnar(ov['entities'])},
                       'materials': mats, 'branch': rec, 'post_hoc': _post_hoc(mats, rec['committed_physical_id'])}
    actions = action_table()
    actions['primitives'] = {n: {k: v for k, v in p.items() if k != 'source'} for n, p in actions['primitives'].items()}
    ev = {'census_status': 'CENSUS_COMPLETE' if len(cases) == 8 else 'CENSUS_INCOMPLETE', 'domain_id': domain, 'purpose': purpose, 'profile': ec.PROFILE_VERSION,
          'cases_in_order': list(cases), 'cases': cases, 'not_run': records, 'legal_evidence_refs': sorted(set(refs)),
          'compression': {**COMPRESSION_RULES, 'tier_used': str(tier)},
          'feedback_roles': {'C_A': 'immediate feedback used by Fast at decision time (origins t, t+48)', 'C_B': 'delayed check of the same case (origins t+96, t+144), opened after commit',
                             'E': 'development_late: the late block (origins t+192..t+336) of these Source cases, opened after every Source commit was frozen; authorized for learning only'},
          'public_semantics': {'field_definitions': FIELD_DEFINITIONS, 'actions': actions, 'tool_contracts': CONTRACTS, 'consumer': ec.CONSUMER,
                               'geometry': {'L': spec.L, 'H': spec.H, 'T': spec.TRAIN_SPAN, 'n_entities': ec.COHORT_SIZE, 'n_parents_per_entity': spec.N_PARENTS},
                               'public_references': {m: public_spec(m) for m in PUBLIC}, 'budget_per_case': LIMITS, 'body_limit_characters': BODY_LIMIT,
                               'metric': 'normalized MSE (frozen T scaler), entity macro; lower is better', 'source_count': '8 entity-group cases (16 entities each); not 128 independent cases'}}
    low = json.dumps(ev, ensure_ascii=False).lower()
    leak = [n for n in tuple(spec.DATASETS) if n in low]
    if leak:
        raise PermissionError('census names a data source: %s' % leak)
    return ev


def payload_bytes(system: str, payload: dict) -> int:
    return len(json.dumps([{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], ensure_ascii=False).encode('utf-8'))


def slow_payload(cen: dict) -> dict:
    refs = cen['legal_evidence_refs']
    compact = cen.get('compression', {}).get('tier_used') in ('3', 'skeleton')
    body = {k: v for k, v in cen.items() if k != 'legal_evidence_refs'} if compact else cen
    return {'domain_id': cen['domain_id'], 'census': body, 'max_candidates': MAX_CANDIDATES, 'candidate_ids': list(CAND_IDS), 'body_limit_characters': BODY_LIMIT,
            **({'legal_evidence_refs': LL.ref_ranges(refs), 'legal_evidence_refs_format': 'evidence_refs must be exact ids inside these ranges, e.g. "source/<branch>/<case>:<integer>", or one of other_refs'}
               if compact else {'legal_evidence_refs': refs}),
            'required': {'observable_applicability': {'const': True}}, 'status': 'CANDIDATE_TEST_ONLY'}


def make_card(*, domain, skill_id, revision, workflow, principles, summary, refs, legal_refs, mode) -> dsk.Skill:
    if not isinstance(mode, str) or not mode.strip() or len(mode) > 80:
        raise ValueError('research_mode must be short nonempty text')
    dsk.check_reusable_text(mode, 'research_mode', W.skill_text_check)
    return dsk.make_skill(skill_id=skill_id, domain_id=domain, revision=revision, workflow=workflow, principles=principles, applicability_summary=summary,
                          compatibility_note='Formed for this study\'s fixed task, Consumer, L/H, hourly sampling, T length and 16-entity cases.', observable_applicability={'const': True},
                          evidence_refs=refs, legal_evidence_refs=legal_refs, source_stage='propose', status='CANDIDATE_TEST_ONLY',
                          allowed_features=ALLOWED_FEATURES, text_check=W.skill_text_check, body_limit=BODY_LIMIT)


def check_rule_status(rs) -> list:
    if not isinstance(rs, list) or not rs:
        raise ValueError('rule_status must be a nonempty list')
    out = []
    for r in rs:
        if not isinstance(r, dict) or set(r) != {'rule', 'status', 'support', 'counter'} or r['status'] not in ('supported', 'hypothesis', 'unresolved'):
            raise ValueError('each rule_status entry is {rule, status in supported|hypothesis|unresolved, support[], counter[]}')
        if not isinstance(r['rule'], str) or not r['rule'].strip() or any(not isinstance(r[k], list) for k in ('support', 'counter')):
            raise ValueError('rule_status entry malformed')
        out.append({'rule': r['rule'][:400], 'status': r['status'], 'support': [str(x)[:300] for x in r['support']][:8], 'counter': [str(x)[:300] for x in r['counter']][:8]})
    return out


def parse_formation(resp, *, domain: str, legal_refs) -> dict:
    """-> {'decision': 'KEEP'|'PROPOSE', 'skills': [Skill...], 'meta': {skill_id: {...}}}; raises ValueError on contract errors."""
    if not isinstance(resp, dict) or resp.get('decision') not in ('KEEP', 'PROPOSE'):
        raise ValueError('decision must be KEEP or PROPOSE')
    if resp['decision'] == 'KEEP':
        if set(resp) - {'decision', 'rationale', 'evidence_review'}:
            raise ValueError('KEEP carries only rationale and evidence_review')
        return {'decision': 'KEEP', 'rationale': str(resp.get('rationale', ''))[:3000], 'evidence_review': LL.check_review(resp['evidence_review']) if 'evidence_review' in resp else None,
                'skills': [], 'meta': {}}
    if set(resp) != {'decision', 'candidates'} or not isinstance(resp['candidates'], list) or not 1 <= len(resp['candidates']) <= MAX_CANDIDATES:
        raise ValueError('PROPOSE carries exactly decision and 1..%d candidates' % MAX_CANDIDATES)
    skills, meta, modes = [], {}, set()
    for i, c in enumerate(resp['candidates']):
        if not isinstance(c, dict) or set(c) != CAND_KEYS:
            raise ValueError('candidate keys must be exactly %s' % sorted(CAND_KEYS))
        if c['candidate_id'] != CAND_IDS[i]:
            raise ValueError('candidate_id must be %s in order' % ', '.join(CAND_IDS[:len(resp['candidates'])]))
        if c['observable_applicability'] != {'const': True}:
            raise ValueError('observable_applicability must be exactly {"const": true}')
        for k in ('commit_policy', 'decision_change_vs_no_card'):
            if not isinstance(c[k], str) or not c[k].strip():
                raise ValueError('%s must be nonempty text' % k)
        mode = c['research_mode']
        if not isinstance(mode, str) or mode.strip().lower() in modes:
            raise ValueError('each candidate needs its own research_mode')
        modes.add(mode.strip().lower())
        review = LL.check_review(c['evidence_review'])
        rs = check_rule_status(c['rule_status'])
        s = make_card(domain=domain, skill_id='%s-%s-r1' % (domain, c['candidate_id']), revision=1, workflow=c['workflow'], principles=c['principles'], summary=c['applicability_summary'],
                      refs=c['evidence_refs'], legal_refs=legal_refs, mode=mode)
        skills.append(s)
        meta[s.skill_id] = {'candidate_id': c['candidate_id'], 'research_mode': mode, 'rationale': str(c['rationale'])[:3000], 'evidence_review': review, 'rule_status': rs,
                            'commit_policy': c['commit_policy'][:2000], 'decision_change_vs_no_card': c['decision_change_vs_no_card'][:2000]}
    if len({s.rendered_body for s in skills}) != len(skills):
        raise ValueError('two candidates render the same body')
    return {'decision': 'PROPOSE', 'skills': skills, 'meta': meta}


def propose_domain(cen: dict, call, domain: str) -> dict:
    """One scientific formation call; one contract correction only. Never a semantic resample; transport / account / budget faults end as PROPOSE_CALL_FAILED."""
    refs = cen['legal_evidence_refs']
    payload = slow_payload(cen)
    low = json.dumps(payload, ensure_ascii=False).lower()
    if any(n in low for n in tuple(spec.DATASETS)):
        raise PermissionError('Slow payload names a data source')
    attempts, prior = [], None
    for i in range(2):
        try:
            raw = call(payload)
        except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
            attempts.append({'attempt': i, 'fault_kind': type(exc).__name__})
            return {'status': 'PROPOSE_CALL_FAILED', 'attempts': attempts, 'skills': [], 'meta': {}}
        except ValueError as exc:
            attempts.append({'attempt': i, 'error': 'response not JSON: %s' % str(exc)[:200]})
            prior = None
        else:
            try:
                out = parse_formation(raw, domain=domain, legal_refs=refs)
                attempts.append({'attempt': i, 'ok': True})
                return {'status': 'NO_CARD' if out['decision'] == 'KEEP' else 'PROPOSED', 'attempts': attempts, 'raw': raw, **out}
            except ValueError as exc:
                attempts.append({'attempt': i, 'error': str(exc)[:300]})
                prior = raw if dsk._jsonable(raw) else None
        if i == 0:
            payload = {**payload, 'correction': {'previous_output': prior, 'error': attempts[-1].get('error'),
                                                 'instruction': 'Correct this contract error only. KEEP is allowed. Do not seek a different outcome.'}}
    return {'status': 'PROPOSE_PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'skills': [], 'meta': {}}


def formation(root: Path, domain: str) -> dict:
    P_ = paths(root)
    out = P_['formation'] / domain
    out.mkdir(parents=True, exist_ok=True)
    if not (P_['formation'] / 'census_compression_rules.json').exists():
        dsks.write_once(P_['formation'] / 'census_compression_rules.json', {**COMPRESSION_RULES, 'frozen_local': now()})
    per_domain_tokens = ALLOC['formation']['tokens'] // len(DOMAINS)
    if not (out / 'census.json').exists():
        sizes, chosen = {}, None
        for tier in (1, 2, 3, 'skeleton'):
            cen = census(root, domain, tier)
            if cen['census_status'] != 'CENSUS_COMPLETE':
                chosen = cen
                break
            nb = payload_bytes(SLOW_FORM, slow_payload(cen))
            sizes['tier_%s_bytes' % tier] = nb
            chosen = cen
            if LL.reservation(nb) + 2 * 8192 <= per_domain_tokens:              # the frozen client reservation of one call must fit this domain's share (correction round: same order)
                break
        chosen['compression']['payload_bytes_by_tier'] = dict(sizes)
        chosen['compression']['per_domain_token_share'] = per_domain_tokens
        dsks.write_once(out / 'census.json', chosen)
    if (out / 'propose.json').exists():
        return context.read_json(out / 'propose.json')
    cen = context.read_json(out / 'census.json')
    if cen['census_status'] != 'CENSUS_COMPLETE':
        rec = {'status': 'NO_PROPOSAL_INPUT', 'census_status': cen['census_status'], 'skills': [], 'meta': {}, 'formation': 'INCOMPLETE'}
        dsks.write_once(out / 'propose.json', rec)
        return rec
    sc = stage_caps(root, 'formation')
    led = rt.RuntimeLedger(P_['ledger'], **sc['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage; paid formation refused')
    if not proxy_reachable():
        raise RuntimeError('LLM proxy not reachable; formation refused before any paid call')
    client = rt.MeteredClient(led, out / 'raw_responses', http_cap=sc['http_cap'])
    res = propose_domain(cen, lambda p: client.call('slow_form', domain, p, SLOW_FORM, max_tokens=MAX_OUTPUT_TOKENS), domain)
    rec = {**{k: v for k, v in res.items() if k != 'skills'}, 'skills': [s.to_json() for s in res['skills']], 'census_legal_ref_count': len(cen['legal_evidence_refs']),
           'census_tier': cen['compression']['tier_used'], 'slow_requests_after': led.s['llm_requests'], 'written_local': now()}
    dsks.write_once(out / 'propose.json', rec)
    if client.fatal or rt.unknown_usage_blocks(led):
        raise RuntimeError('backend fatal or unknown usage during formation; stop')
    return rec


def candidates_of(root: Path, domain: str) -> list:
    p = paths(root)['formation'] / domain / 'propose.json'
    prop = context.read_json(p) if p.exists() else {}
    return [dsk.skill_from_json(s) for s in prop.get('skills', [])] if prop.get('status') == 'PROPOSED' else []


def arm_of(skill: dsk.Skill) -> str:
    return 'cand_' + skill.skill_id.split('-')[1]


# ============================================================================= Select (V01 / V02 per domain) with J on E; freeze card + Fixed_dev
def select_stage(root: Path) -> dict:
    cands = {d: candidates_of(root, d) for d in DOMAINS}
    if not any(cands.values()):
        p = paths(root)['formation'] / 'select_skipped.json'
        if not p.exists():
            dsks.write_once(p, {'status': 'NO_CANDIDATES', 'note': 'no domain produced a candidate: Select not run; F_domain aliases F0 everywhere'})
        return {'status': 'SKIPPED'}
    cases, order, knowledge = [], {}, {}
    for d in DOMAINS:
        arms = {arm_of(s): s for s in cands[d]}
        if not arms:
            continue
        for tail in ('V01', 'V02'):
            c = '%s_%s' % (d, tail)
            cases.append(c)
            order[c] = [a for a in SELECT_ORDER[tail] if a in arms]
            knowledge[c] = {a: asdict(dsk.skill_knowledge(s)) for a, s in arms.items()}
    cfgp = stage_config(root, 'select', cases=cases, order=order, knowledge=br.json_copy(knowledge), labels_e=True, llm=True, random=False)
    return run_stage(root, 'select', cfgp)


def _block_vec(bdir: Path, case: str, plan: str, block: str, seeds=SEEDS):
    jd = bdir / case
    if block == 'c_a':
        src = {c: {'material_id': r['material_id'], 'model_seed': r['model_seed'], 'c_a': r['scores']['c_a']} for c, r in ec.branch_cells(jd).items()}
    else:
        f = jd / ('%s_scores.json' % block)
        if not f.exists():
            return None
        src = context.read_json(f)['cells']
    v, _ = _by_seed(src, plan, block, seeds)
    return v


def _committed(bdir: Path, case: str) -> str | None:
    if not (bdir / 'branch_result.json').exists():
        return None
    res = context.read_json(bdir / 'branch_result.json')
    return res['committed_plan_id'] if res['status'] == 'COMPLETE' and (bdir / case / 'commit.json').exists() else None


def _uniform_materials_e(bdir: Path, case: str) -> dict:
    """{program key: {'steps', 'label', 'public_id', 'e_by_seed'}} of every uniform program with E in this branch."""
    out = {}
    reg = ec.load_registry(bdir / case)
    for mid, r in reg.items():
        if mid == 'FixedMixup':
            steps, key, pub = None, 'FixedMixup', 'FixedMixup'
        else:
            asg = r.get('assignment') if mid not in PUBLIC else [PUBLIC_STEPS[mid]] * ec.COHORT_SIZE
            steps = ec.uniform_steps(asg)
            if steps is None:
                continue
            key, pub = ec._PKEY(steps), (mid if mid in PUBLIC else None)
        e = _block_vec(bdir, case, mid, 'e')
        if e is None:
            continue
        out.setdefault(key, {'steps': steps, 'label': ec.assignment_label([steps] * ec.COHORT_SIZE if steps is not None else 'FixedMixup', pub), 'public_id': pub, 'e_by_seed': e})
    return out


def select_domain(root: Path, domain: str) -> dict:
    """J(W) on the two Select cases' E; Fixed_dev from uniform programs evaluated on both cases."""
    P_ = paths(root)
    skills = candidates_of(root, domain)
    arms = {arm_of(s): s for s in skills}
    sel_cases = ['%s_V01' % domain, '%s_V02' % domain]
    runs, J, incomplete, cost = {}, {}, [], {}
    for a in arms:
        runs[a], ratios = {}, []
        for c in sel_cases:
            b = P_['select'] / ('%s_%s' % (c, a))
            plan = _committed(b, c)
            e = _block_vec(b, c, plan, 'e') if plan else None
            none_e = _block_vec(b, c, 'None', 'e') if b.exists() else None
            res = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
            row = {'status': res['status'] if res else 'NOT_RUN', 'committed': plan, 'committed_label': None, 'failure_kind': res['failure_kind'] if res else None,
                   'e_by_seed': e, 'none_e_by_seed': none_e, 'c_a_by_seed': _block_vec(b, c, plan, 'c_a') if plan else None,
                   'new_evaluations': res['new_evaluations'] if res else None, 'tokens': (dsks._unit_tokens(P_['select']).get('fast:%s' % b.name) or {}) if (P_['select'] / 'raw_responses').exists() else {}}
            if plan:
                reg = ec.load_registry(b / c)
                r = reg.get(plan, {})
                row['committed_label'] = ec.assignment_label(r.get('assignment') if plan not in PUBLIC else [PUBLIC_STEPS.get(plan, [])] * ec.COHORT_SIZE if plan != 'FixedMixup' else 'FixedMixup', plan if plan in PUBLIC else None)
            if e is None or none_e is None or statistics.fmean(none_e) <= 0:
                incomplete.append('%s/%s' % (a, c))
                row['ratio'] = None
            else:
                row['ratio'] = statistics.fmean(e) / statistics.fmean(none_e)
                ratios.append(row['ratio'])
            runs[a][c] = row
        if len(ratios) == len(sel_cases):
            J[a] = statistics.fmean(ratios)
            cost[a] = (sum(runs[a][c]['new_evaluations'] or 0 for c in sel_cases), sum((runs[a][c]['tokens'].get('prompt_tokens', 0) + runs[a][c]['tokens'].get('completion_tokens', 0)) for c in sel_cases))
    rec = {'domain_id': domain, 'formula': 'J(W) = mean over V01/V02 of mean_seed E(actual commit of W) / mean_seed E(None); lower is better',
           'tie_rule': '<= 1e-12 -> fewer new evaluations, fewer tokens, then W1/W2/W3', 'candidates': list(arms), 'runs': runs, 'J': J, 'incomplete': incomplete, 'block': 'e',
           'select_cases': sel_cases, 'written_local': now()}
    if not J:
        rec.update(status='NO_CARD', selected=None, note='no candidate is complete and scorable on both Select cases; F_domain aliases F0')
    else:
        best = min(J.values())
        tied = [a for a in arms if a in J and J[a] - best <= TOL]
        winner = min(tied, key=lambda a: (cost[a][0], cost[a][1], list(arms).index(a)))
        rec.update(status='SELECTED', selected=winner, tied=tied, margins=({a: J[a] - J[winner] for a in J}))
    # Fixed_dev: uniform programs with E on BOTH cases (intersection over every branch of the domain's Select cases)
    per_case = {}
    for c in sel_cases:
        pool = {}
        for b in sorted(P_['select'].glob('%s_cand_*' % c)):
            for key, m in _uniform_materials_e(b, c).items():
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
        rec['fixed_dev'] = {**fixed[kbest], 'candidates_in_intersection': len(fixed), 'table': {v['label']: round(v['J'], 6) for v in fixed.values()}}
    else:
        rec['fixed_dev'] = None
    return rec


def freeze_domain(root: Path, domain: str) -> dict:
    P_ = paths(root)
    P_['freeze'].mkdir(parents=True, exist_ok=True)
    out = P_['freeze'] / ('%s.json' % domain)
    if out.exists():
        return context.read_json(out)
    prop_p = P_['formation'] / domain / 'propose.json'
    prop = context.read_json(prop_p) if prop_p.exists() else {'status': 'NOT_RUN'}
    skills = candidates_of(root, domain)
    if prop.get('status') not in ('PROPOSED', 'NO_CARD'):
        raise RuntimeError('formation status %s of %s is not a scientific outcome; nothing is frozen' % (prop.get('status'), domain))
    if not skills:
        rec = {'domain_id': domain, 'status': 'NO_CARD', 'skill': None, 'propose_status': prop.get('status'), 'selection': None, 'fixed_dev': None,
               'note': 'no candidate card: F_domain aliases F0 (same trajectory and scores); Fixed_dev not defined (no Select run)', 'frozen_local': now()}
    else:
        sel = select_domain(root, domain)
        dsks.write_once(P_['formation'] / domain / 'selection.json', sel)
        if sel['status'] == 'SELECTED':
            s = next(x for x in skills if arm_of(x) == sel['selected'])
            frozen = replace(s, status='FROZEN_SELECTED', source_stage='freeze', derived_from=s.skill_id)
            rec = {'domain_id': domain, 'status': 'FROZEN_SELECTED', 'skill': frozen.to_json(), 'selection': {k: sel.get(k) for k in ('status', 'selected', 'J', 'margins', 'tied', 'incomplete', 'select_cases', 'block')},
                   'candidates': [x.to_json() for x in skills], 'injected_body': frozen.rendered_body, 'fixed_dev': sel.get('fixed_dev'), 'frozen_local': now(),
                   'note': 'selected on the two Select cases\' E; selection is not promotion to a verified Skill'}
        else:
            rec = {'domain_id': domain, 'status': 'NO_CARD', 'skill': None, 'selection': {k: sel.get(k) for k in ('status', 'selected', 'J', 'incomplete', 'select_cases', 'block')},
                   'candidates': [x.to_json() for x in skills], 'fixed_dev': sel.get('fixed_dev'), 'frozen_local': now(), 'note': 'selection basis incomplete: F_domain aliases F0'}
    dsks.write_once(out, rec)
    return rec


# ============================================================================= MetaTest
def metatest_stage(root: Path) -> dict:
    frozen = {d: freeze_domain(root, d) for d in DOMAINS}
    cases = [c for d in DOMAINS for c in ('%s_Q01' % d, '%s_Q02' % d)]
    knowledge, order, note, fixed = {}, {}, {}, {}
    for d in DOMAINS:
        fr = frozen[d]
        if fr['status'] == 'FROZEN_SELECTED':
            s = dsk.skill_from_json(fr['skill'])
            if s.status != 'FROZEN_SELECTED' or s.domain_id != d:
                raise PermissionError('only the FROZEN_SELECTED card of its domain enters MetaTest')
            kn_skill = asdict(dsk.skill_knowledge(s))
        else:
            kn_skill = None
            note[d] = {'f_domain': {'status': 'EMPTY_SKILL_ALIAS', 'alias_of_arm': 'f0', 'frozen_skill_status': fr['status'],
                                    'note': 'no card for this domain: every actual request condition equals the F0 arm; the F0 trajectory and delivery are reused, no second LLM run'}}
        if fr.get('fixed_dev'):
            fixed[d] = {k: fr['fixed_dev'][k] for k in ('label', 'steps', 'public_id', 'J')}
        for c in ('%s_Q01' % d, '%s_Q02' % d):
            knowledge[c] = {'f0': asdict(br.Knowledge()), 'f_domain': kn_skill, 'f_generic': asdict(generic_knowledge()), 'random': None, 'fixed_dev': None}
            order[c] = [a for a in TEST_ORDER[c] if not (a == 'f_domain' and kn_skill is None)] + list(CONTROL_ARMS)
    cat = paths(root)['freeze'] / 'metatest_catalog.json'
    if not cat.exists():
        dsks.write_once(cat, {'status': 'METATEST_FROZEN', 'epoch': time.time(), 'frozen_local': now(), 'skills': {d: frozen[d].get('skill') for d in DOMAINS},
                              'skill_status': {d: frozen[d]['status'] for d in DOMAINS}, 'fixed_dev': fixed, 'generic': GENERIC_TEXT, 'orders': order, 'frozen_before_first_metatest_fit': True})
    tn = {}
    for d in DOMAINS:
        tn.update(note.get(d, {}))
    cfgp = stage_config(root, 'metatest', cases=cases, order=order, knowledge=br.json_copy(knowledge), labels_e=True, llm=True, random=True, treatment_note=tn, fixed_dev=fixed)
    return run_stage(root, 'metatest', cfgp)


# ============================================================================= package driver
def package_run(root: Path = ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    status = context.read_json(root / 'package_status.json') if (root / 'package_status.json').exists() else {}

    def stop(where, st):
        status.update({'stopped_at': where, 'stage_status': st, 'epoch': time.time(), 'local': now()})
        context.write_json(root / 'package_status.json', status)
        print('PACKAGE_STOP', where, st.get('status'), flush=True)
        return status
    preflight(root)
    w = wiring(root)
    status['wiring'] = w['status']
    if w['status'] != 'PASS':
        return stop('wiring', w)
    start_paid_clock(root)
    src_cases = case_ids(root, 'source')
    cache_from = {c: paths(root)['wiring'] / (c + '_common') / c for c in src_cases if (paths(root)['wiring'] / (c + '_common') / c / 'aug_materials' / 'index.json').exists()}
    st = run_stage(root, 'source', stage_config(root, 'source', cases=src_cases, order={c: ['f0'] for c in src_cases}, knowledge={c: {'f0': asdict(br.Knowledge())} for c in src_cases},
                                                labels_e=True, llm=True, random=False, cache_from=cache_from))
    status['source'] = st
    if st['status'] != 'FINISHED':
        return stop('source', st)
    for d in DOMAINS:
        prop = formation(root, d)
        status['formation_%s' % d] = prop.get('status')
        if prop.get('status') not in ('PROPOSED', 'NO_CARD'):
            return stop('formation_%s' % d, {'status': prop.get('status'), 'attempts': prop.get('attempts'), 'note': 'formation did not produce a scientific outcome; operator decision needed'})
    st = select_stage(root)
    status['select'] = st
    if st['status'] not in ('FINISHED', 'SKIPPED'):
        return stop('select', st)
    for d in DOMAINS:
        status['frozen_%s' % d] = freeze_domain(root, d)['status']
    st = metatest_stage(root)
    status['metatest'] = st
    if st['status'] != 'FINISHED':
        return stop('metatest', st)
    status['finished_epoch'] = time.time()
    status['finished_local'] = now()
    context.write_json(root / 'package_status.json', status)
    readout(root)
    print('PACKAGE_FINISHED', flush=True)
    return status


# ============================================================================= readout (task §8)
def _stats(v: list) -> dict:
    m = statistics.fmean(v)
    se = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None
    return {'mean': m, 'se': se, 't95': [m - T975_DF2 * se, m + T975_DF2 * se] if se is not None and len(v) == 3 else None, 'per_seed': v,
            'signs': [int(x > 0) - int(x < 0) for x in v]}


def _delta(a, b, den) -> dict | None:
    """Delta = 100 x (E(A) - E(B)) / den per seed; positive = B better; den = the case three-seed None mean."""
    if a is None or b is None or not den:
        return None
    return _stats([100.0 * (x - y) / den for x, y in zip(a, b)])


def _label_of(bdir: Path, case: str, plan: str) -> str:
    if plan in PUBLIC:
        return plan
    reg = ec.load_registry(bdir / case)
    r = reg.get(plan) or {}
    return ec.assignment_label(r.get('assignment'))


def _arm_record(bdir: Path, case: str) -> dict:
    res = context.read_json(bdir / 'branch_result.json') if (bdir / 'branch_result.json').exists() else None
    plan = _committed(bdir, case)
    rec = {'status': res['status'] if res else 'NOT_RUN', 'failure_kind': res.get('failure_kind') if res else None, 'committed': plan, 'committed_label': _label_of(bdir, case, plan) if plan else None,
           'calls': res['calls'] if res else None, 'tool_calls': res['tool_calls'] if res else None, 'new_evaluations': res['new_evaluations'] if res else None,
           'e_by_seed': _block_vec(bdir, case, plan, 'e') if plan else None, 'c_a_by_seed': _block_vec(bdir, case, plan, 'c_a') if plan else None,
           'c_b_by_seed': _block_vec(bdir, case, plan, 'c_b') if plan else None, 'pool': {}}
    reg = ec.load_registry(bdir / case)
    for mid in reg:
        ca, e = _block_vec(bdir, case, mid, 'c_a'), _block_vec(bdir, case, mid, 'e')
        if ca is not None:
            rec['pool'][mid] = {'label': _label_of(bdir, case, mid), 'c_a': statistics.fmean(ca), 'e': statistics.fmean(e) if e else None, 'e_by_seed': e, 'public': mid in PUBLIC}
    if rec['pool']:
        order = list(rec['pool'])
        rec['c_a_argmin'] = min(order, key=lambda m: (rec['pool'][m]['c_a'], order.index(m)))
        with_e = [m for m in order if rec['pool'][m]['e'] is not None]
        rec['e_best_oracle'] = min(with_e, key=lambda m: (rec['pool'][m]['e'], order.index(m))) if with_e else None
        rec['commit_is_c_a_argmin'] = (rec['c_a_argmin'] == plan) if plan else None
        rec['own_plans_evaluated'] = [m for m in order if m not in PUBLIC]
    tok = dsks._unit_tokens(bdir.parent).get('fast:%s' % bdir.name, {}) if (bdir.parent / 'raw_responses').exists() else {}
    rec['tokens'] = tok.get('prompt_tokens', 0) + tok.get('completion_tokens', 0) if tok else 0
    rec['requests'] = tok.get('requests', 0) if tok else 0
    return rec


def readout(root: Path = ROOT) -> dict:
    P_ = paths(root)
    cases = cases_of(root)
    frozen = {d: (context.read_json(P_['freeze'] / ('%s.json' % d)) if (P_['freeze'] / ('%s.json' % d)).exists() else {'status': 'NOT_FROZEN'}) for d in DOMAINS}
    out = {'package': PACKAGE, 'exposure': EXPOSURE, 'written_local': now(), 'frozen': {d: {k: frozen[d].get(k) for k in ('status', 'selection', 'fixed_dev', 'injected_body')} for d in DOMAINS},
           'metatest': {}, 'source': {}, 'select': {}, 'summary': {}}
    # ---- Source: what happened per case (with E)
    for c in case_ids(root, 'source'):
        b = P_['source'] / ('%s_f0' % c)
        if b.exists():
            r = _arm_record(b, c)
            none_e = r['pool'].get('None', {}).get('e')
            out['source'][c] = {**{k: r[k] for k in ('status', 'committed', 'committed_label', 'calls', 'tool_calls', 'new_evaluations', 'c_a_argmin', 'e_best_oracle', 'commit_is_c_a_argmin', 'tokens', 'requests') if k in r},
                                'commit_e_ratio_to_none': (statistics.fmean(r['e_by_seed']) / none_e) if (r.get('e_by_seed') and none_e) else None,
                                'pool_labels': {m: v['label'] for m, v in r['pool'].items()},
                                'pool_e_ratio_to_none': {m: (v['e'] / none_e if (v['e'] is not None and none_e) else None) for m, v in r['pool'].items()}}
    # ---- Select
    for d in DOMAINS:
        p = P_['formation'] / d / 'selection.json'
        if p.exists():
            s = context.read_json(p)
            out['select'][d] = {'status': s['status'], 'selected': s.get('selected'), 'J': s.get('J'), 'incomplete': s.get('incomplete'),
                                'runs': {a: {c: {k: v.get(k) for k in ('status', 'committed', 'committed_label', 'ratio', 'new_evaluations')} for c, v in rr.items()} for a, rr in s['runs'].items()},
                                'fixed_dev': s.get('fixed_dev')}
    # ---- MetaTest
    arms = ('f0', 'f_domain', 'f_generic', 'random', 'fixed_dev')
    for c in case_ids(root, 'test'):
        cs = cases[c]
        recs = {}
        for a in arms:
            b = P_['metatest'] / ('%s_%s' % (c, a))
            if b.exists():
                recs[a] = _arm_record(b, c)
            elif a == 'f_domain' and (P_['metatest'] / ('%s_f0' % c)).exists():
                recs[a] = {**_arm_record(P_['metatest'] / ('%s_f0' % c), c), 'alias_of': 'f0'}
            else:
                recs[a] = {'status': 'NOT_RUN'}
        f0 = recs.get('f0', {})
        pub = {m: f0['pool'][m]['e_by_seed'] for m in PUBLIC if m in f0.get('pool', {}) and f0['pool'][m]['e_by_seed']}
        none_e = pub.get('None')
        den = statistics.fmean(none_e) if none_e else None
        row = {'domain': cs.domain, 'anchor': 'A' if cs.t == load_split(root)['anchors'][0] else 'B', 'none_e_by_seed': none_e, 'public_e_by_seed': pub,
               'arms': {a: {k: v for k, v in r.items() if k != 'pool'} for a, r in recs.items()}, 'deltas': {}}
        e = {a: recs[a].get('e_by_seed') for a in arms}
        dom = e.get('f_domain')
        row['deltas'] = {'F_domain_vs_F0': _delta(e.get('f0'), dom, den), 'F_domain_vs_Generic': _delta(e.get('f_generic'), dom, den),
                         'F_domain_vs_Random': _delta(e.get('random'), dom, den), 'F_domain_vs_FixedDev': _delta(e.get('fixed_dev'), dom, den),
                         'F_domain_vs_NoMix': _delta(pub.get('P_NoMixRecipe'), dom, den), 'F_domain_vs_None': _delta(none_e, dom, den),
                         'F0_vs_None': _delta(none_e, e.get('f0'), den), 'Generic_vs_None': _delta(none_e, e.get('f_generic'), den),
                         'Random_vs_None': _delta(none_e, e.get('random'), den), 'FixedDev_vs_None': _delta(none_e, e.get('fixed_dev'), den),
                         'NoMix_vs_None': _delta(none_e, pub.get('P_NoMixRecipe'), den), 'F0_vs_NoMix': _delta(pub.get('P_NoMixRecipe'), e.get('f0'), den)}
        for a in ('f0', 'f_domain', 'f_generic', 'random'):
            r = recs.get(a, {})
            if r.get('pool') and den and r.get('e_by_seed'):
                best = r.get('e_best_oracle')
                row['arms'][a]['oracle_gap_pp'] = 100.0 * (statistics.fmean(r['e_by_seed']) - r['pool'][best]['e']) / den if best else None
                row['arms'][a]['oracle_label'] = r['pool'][best]['label'] if best else None
        row['e_ratio_to_none'] = {a: (statistics.fmean(e[a]) / den if (e.get(a) and den) else None) for a in arms}
        row['e_ratio_to_none'].update({m: (statistics.fmean(v) / den if den else None) for m, v in pub.items()})
        out['metatest'][c] = row
    # ---- aggregate: domain = equal-weight 2 cases; overall = equal-weight 2 domains
    keys = list(next(iter(out['metatest'].values()))['deltas']) if out['metatest'] else []
    agg = {}
    for k in keys:
        dom_means = {}
        for d in DOMAINS:
            vals = [out['metatest'][c]['deltas'][k]['mean'] for c in out['metatest'] if out['metatest'][c]['domain'] == d and out['metatest'][c]['deltas'].get(k)]
            dom_means[d] = statistics.fmean(vals) if vals else None
        ok = [v for v in dom_means.values() if v is not None]
        agg[k] = {'by_domain': dom_means, 'overall': statistics.fmean(ok) if len(ok) == len(DOMAINS) else None,
                  'per_case': {c: out['metatest'][c]['deltas'][k]['mean'] if out['metatest'][c]['deltas'].get(k) else None for c in out['metatest']},
                  'positive_cases': sum(1 for c in out['metatest'] if out['metatest'][c]['deltas'].get(k) and out['metatest'][c]['deltas'][k]['mean'] > 0),
                  'n_cases': sum(1 for c in out['metatest'] if out['metatest'][c]['deltas'].get(k))}
    out['summary'] = {'deltas_pp': agg, 'commit_vs_c_a_argmin': {c: {a: out['metatest'][c]['arms'][a].get('commit_is_c_a_argmin') for a in ('f0', 'f_domain', 'f_generic')} for c in out['metatest']},
                      'committed_labels': {c: {a: out['metatest'][c]['arms'][a].get('committed_label') for a in arms} for c in out['metatest']}}
    out['cost'] = cost_readout(root)
    context.write_json(root / 'result.json', out)
    (root / 'tables.md').write_text(tables(out), encoding='utf-8')
    return out


def cost_readout(root: Path) -> dict:
    P_ = paths(root)
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    stages = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    per_stage = {}
    names = list(stages)
    for i, s in enumerate(names):
        snap = stages[s]['snapshot_at_start']
        nxt = stages[names[i + 1]]['snapshot_at_start'] if i + 1 < len(names) else {k: led.get(k, 0) for k in snap}
        per_stage[s] = {k: nxt.get(k, 0) - snap.get(k, 0) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'fit_wall_seconds')}
    return {'ledger': {k: v for k, v in led.items() if k != 'events'}, 'per_stage': per_stage,
            'paid_elapsed_s': (time.time() - led['paid_clock_started_epoch']) if led.get('paid_clock_started_epoch') else None}


def _f(x, nd=2):
    return '-' if x is None else ('%.*f' % (nd, x))


def tables(res: dict) -> str:
    L_ = ['# %s tables (%s)' % (PACKAGE, res['written_local']), '', 'Exposure: %s. Metric: E normalized MSE (T scaler), entity macro, seed mean. Delta in pp of the case None mean; positive = the second arm better.' % res['exposure'], '']
    L_ += ['## MetaTest: F_domain deltas (pp)', '', '| case | dom | vs F0 | vs Generic | vs Random | vs Fixed_dev | vs NoMix | vs None |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for c, r in res['metatest'].items():
        d = r['deltas']
        L_.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (c, r['domain'], *[(_f(d[k]['mean']) + (' (SE %s)' % _f(d[k]['se']) if d[k] and d[k]['se'] is not None else '')) if d.get(k) else '-'
                                                                   for k in ('F_domain_vs_F0', 'F_domain_vs_Generic', 'F_domain_vs_Random', 'F_domain_vs_FixedDev', 'F_domain_vs_NoMix', 'F_domain_vs_None')]))
    agg = res['summary'].get('deltas_pp', {})
    L_ += ['', '| aggregate | D01 | D02 | overall | positive cases |', '|---|---:|---:|---:|---:|']
    for k, v in agg.items():
        L_.append('| %s | %s | %s | %s | %d/%d |' % (k, _f(v['by_domain'].get('D01')), _f(v['by_domain'].get('D02')), _f(v['overall']), v['positive_cases'], v['n_cases']))
    L_ += ['', '## MetaTest: E / None per arm', '', '| case | F0 | F_domain | F_generic | Random | Fixed_dev | NoMix | FixedMixup | AmpResample |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c, r in res['metatest'].items():
        rr = r['e_ratio_to_none']
        L_.append('| %s | %s |' % (c, ' | '.join(_f(rr.get(k), 4) for k in ('f0', 'f_domain', 'f_generic', 'random', 'fixed_dev', 'P_NoMixRecipe', 'FixedMixup', 'P_AmpResample'))))
    L_ += ['', '## MetaTest: commits', '', '| case | arm | status | committed | C_A argmin? | oracle gap pp | calls | new evals | tokens |', '|---|---|---|---|---|---:|---:|---:|---:|']
    for c, r in res['metatest'].items():
        for a, v in r['arms'].items():
            L_.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (c, ARM_LABEL.get(a, a) + (' (alias F0)' if v.get('alias_of') else ''), v.get('status'), v.get('committed_label'),
                                                                        v.get('commit_is_c_a_argmin'), _f(v.get('oracle_gap_pp')), v.get('calls'), v.get('new_evaluations'), v.get('tokens')))
    L_ += ['', '## Select', '']
    for d, s in res['select'].items():
        L_.append('- %s: status %s, selected %s, J %s, fixed_dev %s' % (d, s['status'], s.get('selected'), {k: round(v, 4) for k, v in (s.get('J') or {}).items()},
                                                                         (s.get('fixed_dev') or {}).get('label')))
    L_ += ['', '## Source (F0 per case)', '', '| case | committed | commit E/None | C_A argmin? | oracle | calls | new evals | tokens |', '|---|---|---:|---|---|---:|---:|---:|']
    for c, r in res['source'].items():
        L_.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (c, r.get('committed_label'), _f(r.get('commit_e_ratio_to_none'), 4), r.get('commit_is_c_a_argmin'),
                                                                  (r.get('pool_labels') or {}).get(r.get('e_best_oracle')), r.get('calls'), r.get('new_evaluations'), r.get('tokens')))
    cost = res.get('cost', {})
    L_ += ['', '## Cost', '', '```', json.dumps({k: v for k, v in cost.get('ledger', {}).items() if k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts',
                                                                                                          'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'fit_wall_seconds', 'label_wall_seconds', 'material_wall_seconds')}, indent=1),
           json.dumps(cost.get('per_stage', {}), indent=1), '```']
    return '\n'.join(L_) + '\n'


# ============================================================================= smoke (synthetic file, synthetic split; 0 real fits, 0 API; new paths + label isolation)
def _synthetic_split(out: Path, n_cols: int = 40, hours: int = 1200) -> Path:
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
    groups = []
    t = 700
    for i, (cid, role) in enumerate((('SM_S01', 'source'), ('SM_V01', 'select'))):
        cs = ec.CaseSpec(dataset='synthetic', domain='SM', case_id=cid, role=role, t=t, roster=tuple(cols[16 * i: 16 * (i + 1)]), case_index=i, csv_path=str(csvp), total_hours=hours)
        groups.append({'case_id': cid, 'stage': role, 't': t, 'roster': list(cs.roster), 'case_index': i, 'train_rows': list(cs.train_range), 'c_a_origins': list(cs.c_a),
                       'c_b_origins': list(cs.c_b), 'e_origins': list(cs.e)})
    split = {'setting_id': 'SMOKE', 'anchors': [t], 'cohort_size': 16, 'domains': {'SM': {'dataset': 'synthetic', 'csv_path': str(csvp), 'total_hours': hours, 'columns_in_file': n_cols + 1,
                                                                                         'excluded_columns': ['OT'], 'partition_seed': 0, 'groups': groups}}}
    p = out / 'split.json'
    context.write_json(p, split)
    return p


def smoke(root: Path = ROOT) -> dict:
    out = paths(root)['smoke']
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    checks = []

    def check(name, ok, **info):
        checks.append({'check': name, 'ok': bool(ok), **info})
        print('SMOKE', 'ok ' if ok else 'FAIL', name, flush=True)
    # 1 partition document
    split = context.read_json(SPLIT_DOC)
    chk = ec.check_split(split)
    check('document partition: 2 x 12 groups of 16, disjoint, roles 8/2/2, OT excluded', chk['ok'], detail=chk)
    cases = ec.cases_from_split(split)
    check('document geometry matches CaseSpec rows', len(cases) == 24)
    # 2 program space
    ok = True
    try:
        ec.validate_steps_full([{'op': 'tp_censor'}, {'op': 'tp_regime'}, {'op': 'tp_resample'}])
        ec.validate_steps_full([{'op': ta.RECIPE_EDIT, 'disabled_ops': ['tp_shock', 'tp_calendar']}])
        ec.validate_steps_full([{'op': ta.RECIPE}])
        ec.validate_steps_full([])
    except Exception as exc:  # noqa: BLE001
        ok = False
    for bad in ([{'op': 'tp_regime'}, {'op': 'tp_shock'}], [{'op': 'a'}] * 4, [{'op': ta.RECIPE}, {'op': 'tp_regime'}], [{'op': ta.RECIPE_EDIT, 'disabled_ops': ['x']}],
                [{'op': ta.RECIPE_EDIT, 'disabled_ops': ['tp_regime']}, {'op': 'tp_censor'}], [{'op': ta.RECIPE_EDIT, 'disabled_ops': ['tp_regime', 'tp_shock', 'tp_censor']}]):
        try:
            ec.validate_steps_full(bad)
            ok = False
        except policy.PolicyError:
            pass
    check('program validation: compositions, preset, edits accepted; exclusives / 4 steps / mixed / unknown / 3 disabled rejected', ok)
    check('program tables: 53 explicit + 29 edit, 81 legal; preset seed index shared by every edit',
          len(ec.PROGRAMS) == 53 and len(ec.EDIT_PROGRAMS) == 29 and len(ec.LEGAL_PROGRAMS) == 81 and ec.program_seed_index(ec.EDIT_PROGRAMS[5]) == ec.PRESET_INDEX == ec.program_seed_index([{'op': ta.RECIPE}]))
    check('program tables identical to the earlier packages\' enumeration', ec.PROGRAMS == W.program_table())
    s1 = ec.entity_program_seed('D01', 3, 7, '25')
    check('seed rule: deterministic, distinct across domain / case / program / entity',
          s1 == ec.entity_program_seed('D01', 3, 7, '25') and len({s1, ec.entity_program_seed('D02', 3, 7, '25'), ec.entity_program_seed('D01', 4, 7, '25'),
                                                                   ec.entity_program_seed('D01', 3, 8, '25'), ec.entity_program_seed('D01', 3, 7, '26')}) == 5)
    # 3 synthetic case: loader converts only the roster, stage row permissions, scaler binding
    sp = _synthetic_split(out)
    scases = ec.cases_from_split(context.read_json(sp))
    cs = scases['SM_S01']
    run = out / 'run'
    ctx = ec.open_case(cs, run, 'material')
    check('loader converts only the 16 roster columns (file has 41)', ctx.slice.values.shape[1] == 16 and ctx.slice.columns_in_file == 41)
    ok = True
    try:
        ctx.slice.rows(cs.t, cs.t + 1)
        ok = False
    except PermissionError:
        pass
    check('material stage cannot read row t (C_A) ', ok)
    ev = ec.open_case(cs, run, 'evaluate')
    ok = True
    try:
        ev.slice.rows(cs.c_b[0], cs.c_b[0] + 1)
        ok = False
    except PermissionError:
        pass
    check('evaluate stage cannot read a C_B origin row', ok and ev.slice.row_end == cs.t + 96)
    ok = True
    try:
        ec.open_case(replace(cs, t=cs.t + 24), run, 'material')
        ok = False
    except RuntimeError:
        pass
    try:
        ec.open_case(replace(cs, roster=tuple(scases['SM_V01'].roster)), run, 'material')
        ok = False
    except (RuntimeError, ValueError):
        pass
    check('scaler / spec binding refuses another t or roster in the same case dir', ok)
    check('stage rows: c_b [t-96,t+192), e_input [t,t+336), e_target [t+192,t+384)',
          cs.c_b_rows == (cs.t - 96, cs.t + 192) and cs.e_input_rows == (cs.t, cs.t + 336) and cs.e_target_rows == (cs.t + 192, cs.t + 384))
    ov = ctx.overview()
    check('overview: 16 entity rows, no dataset name, no action table (adapter adds it)', ov['n_entities'] == 16 and 'actions' not in ov and not any(n in json.dumps(ov).lower() for n in spec.DATASETS))
    # 4 materials in a worker: public + composition + edit + conditional; alias; window timestamps
    rc, _ = run_sub(['--worker-build', run, sp, 'SM_S01'], out / 'build.log', 600)
    reg = ec.load_registry(run / 'SM_S01')
    check('public materials built in a worker (None, FixedMixup, P_AmpResample, P_NoMixRecipe)', rc == 0 and all(m in reg for m in PUBLIC), log=(out / 'build.log').read_text(encoding='utf-8')[-300:])
    table = ov['entities']
    comp = ec.compile_plan_full(ec.uniform_policy(WIRING_COMPOSITION, 'smoke'), table)
    m1 = ensure_material(run, sp, 'SM_S01', comp['assignment'])
    cond = ec.compile_plan_full({'default': {'steps': [{'op': ta.RECIPE_EDIT, 'disabled_ops': ['tp_random_conv']}]},
                                 'rules': [{'when': {'feature': 'lag24_corr', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': [{'op': 'tp_calendar'}]}], 'rationale': 'smoke conditional'}, table)
    m2 = ensure_material(run, sp, 'SM_S01', cond['assignment'])
    again = ensure_material(run, sp, 'SM_S01', comp['assignment'])
    s1_ = context.read_json(m1['summary_path'])
    s2_ = context.read_json(m2['summary_path'])
    check('explicit composition material: 3 steps executed on every window; same assignment -> same physical record (no rebuild)',
          m1['material_id'] == 'TA001' and again['material_id'] == 'TA001' and all(s1_['per_step'][n]['windows'] == 16 * 433 for n in ('tp_regime', 'tp_resample', 'tp_censor')))
    check('conditional edit + calendar material: both programs present, edit skip counts recorded, every entity seeded from the preset stream or the calendar index',
          m2['material_id'] == 'TA002' and s2_['edit'] is not None and set(cond['rule_index']) == {-1, 0} and sum(x == 0 for x in cond['rule_index']) >= 2)
    preset_seed = s2_['entity_program_seeds']
    check('edit entities share the preset program index in the seed record', all(v['program_index'] == ec.PRESET_INDEX for v in preset_seed.values() if v['disabled_ops'] is not None))
    check('material key / alias of an identity assignment resolves to None', ensure_material(run, sp, 'SM_S01', [[] for _ in range(16)])['material_id'] == 'None')
    # 5 random supply
    sup = draw_random_supply('SM_S01', 'SM', 0, table)
    sup2 = draw_random_supply('SM_S01', 'SM', 0, table)
    check('random supply: 4 slots, deterministic, R1 explicit / R2 edit / R3-R4 conditional or frozen-order fallback, keys distinct',
          len(sup['plans']) == 4 and sup == sup2 and sup['plans'][0]['kind'] == 'uniform' and sup['plans'][1]['kind'] == 'uniform'
          and len({ec.material_key(ec.compile_plan_full(p['policy'], table)['assignment']) for p in sup['plans']}) == 4, slots=[(p['slot'], p['kind'], p['label']) for p in sup['plans']])
    # 6 short fits (n_updates=20) through the parallel launcher + branch view + labels + barrier
    led = rt.RuntimeLedger(out / 'ledger.json', max_fit_attempts=20, max_llm_requests=0, max_llm_tokens=0, max_wall_s=3600, max_retries=1)
    got = fit_cells(led, run, sp, 'SM_S01', [('None', s) for s in SEEDS] + [('TA001', SEEDS[0])], n_updates=20)
    check('4 short fits through the parallel launcher: pool 16x433, threads fixed, C_A scored, ledger charged 4',
          len(got) == 4 and all(r['pool_size'] == 6928 and r['torch_threads'] == FIT_THREADS and r['scores']['c_a']['status'] == 'SCORABLE' for r in got.values()) and led.s['fit_attempts'] == 4)
    branch = out / 'branch_f0'
    ad = CaseAdapter(branch, 'SM_S01', led, REPO, common=run, split_path=sp, cs=cs, public_ids=('None',), seeds=SEEDS, fit_fn=lambda phys, seeds_: fit_cells(led, run, sp, 'SM_S01', [(phys, s) for s in seeds_], n_updates=20))
    bl = ad.baselines()
    check('branch view: None restored from cache (0 new fits), feedback tensor 3 x 2 x 16', led.s['fit_attempts'] == 4 and bl[0].feedback(SEEDS, 2, 16)['loss_by_seed'] is not None)
    cand = ad.build_material({'plan_id': 'P1', 'policy': ec.uniform_policy(WIRING_COMPOSITION, 'smoke branch plan')}, remaining_seconds=100)
    cand = ad.evaluate(cand, SEEDS, feedback=True, remaining_seconds=100)
    check('branch evaluate: alias of the existing physical material, 2 new fits only (one cached)', cand.model_seeds == SEEDS and led.s['fit_attempts'] == 6 and led.s['cache_hits'] >= 1)
    ok = True
    try:
        ad.build_material({'plan_id': 'P2', 'policy': ec.uniform_policy(WIRING_COMPOSITION, 'dup')}, remaining_seconds=100)
        ok = False
    except br.ToolInputError as exc:
        ok = exc.details.get('execution_status') == 'DUPLICATE_ASSIGNMENT'
    check('duplicate assignment rejected as a tool input error (no slot)', ok)
    commit_branch(branch, 'synthetic', 'SM_S01', 'P1', 'smoke commit', SEEDS[0])
    ok = True
    try:
        ec.label_prerequisite('score_e', branch, cs, out)
        ok = False
    except PermissionError:
        pass
    check('score_e refused before C_B / freeze / barrier', ok)
    rc1, _ = run_sub(['--worker-label', 'c_b', branch, sp, 'SM_S01', out], out / 'label_cb.log', 600)
    rc2, _ = run_sub(['--worker-label', 'freeze_e', branch, sp, 'SM_S01', out], out / 'label_fe.log', 600)
    rc3, _ = run_sub(['--worker-label', 'score_e', branch, sp, 'SM_S01', out], out / 'label_se_refused.log', 600)
    check('C_B scored, E frozen; score_e worker refused without the whole-stage barrier', rc1 == 0 and rc2 == 0 and rc3 != 0 and (branch / 'SM_S01' / 'c_b_scores.json').exists() and (branch / 'SM_S01' / 'e_frozen.json').exists())
    context.write_json(out / 'execution_finished.json', {'epoch': time.time(), 'branches': [[str(branch), 'SM_S01']], 'failures': []})
    context.write_json(out / 'all_e_predictions_frozen.json', {'epoch': time.time(), 'branches': [str(branch)]})
    rc4, _ = run_sub(['--worker-label', 'score_e', branch, sp, 'SM_S01', out], out / 'label_se.log', 600)
    es = context.read_json(branch / 'SM_S01' / 'e_scores.json') if rc4 == 0 else {}
    check('score_e after the barrier: 4 origins x 16 entities per cell', rc4 == 0 and all(v['e']['n_origins'] == 4 and v['e']['n_entities'] == 16 for v in es.get('cells', {}).values()))
    cells = ec.branch_cells(run / 'SM_S01')
    check('physical cache holds no label block after the branch labels', all(set(r['scores']) == {'c_a'} for r in cells.values()))
    # 7 knowledge rendering and Slow parsing contract
    gk = generic_knowledge().render({}, ALLOWED_FEATURES)
    check('generic knowledge renders as one loaded experiment_guidance entry', len(gk['loaded']) == 1 and gk['loaded'][0]['hook'] == dsk.HOOK)
    fake = {'decision': 'PROPOSE', 'candidates': [{'candidate_id': 'W1', 'research_mode': 'mode a', 'workflow': 'Observe lag structure; construct one composition and one edit; compare on C_A; commit the lower unless the gap is within seed SE.',
                                                   'principles': None, 'observable_applicability': {'const': True}, 'applicability_summary': 'smoke', 'evidence_refs': ['source/x/y:1'], 'rationale': 'r',
                                                   'evidence_review': {'supports': [], 'contradicts_or_limits': [], 'rule_changes': [], 'uncertain_and_expected_behavior_change': 'u'},
                                                   'rule_status': [{'rule': 'r1', 'status': 'hypothesis', 'support': [], 'counter': []}], 'commit_policy': 'p', 'decision_change_vs_no_card': 'd'},
                                                  {'candidate_id': 'W2', 'research_mode': 'mode b', 'workflow': 'Keep the preset unless two of three seeds favour an evaluated alternative on C_A.',
                                                   'principles': 'Prefer fewer new plans.', 'observable_applicability': {'const': True}, 'applicability_summary': 'smoke', 'evidence_refs': ['source/x/y:1'], 'rationale': 'r',
                                                   'evidence_review': {'supports': [], 'contradicts_or_limits': [], 'rule_changes': [], 'uncertain_and_expected_behavior_change': 'u'},
                                                   'rule_status': [{'rule': 'r1', 'status': 'supported', 'support': ['a'], 'counter': []}], 'commit_policy': 'p', 'decision_change_vs_no_card': 'd'}]}
    parsed = parse_formation(fake, domain='D01', legal_refs=['source/x/y:1'])
    ok = len(parsed['skills']) == 2 and parsed['skills'][1].principles is not None
    try:
        parse_formation({**fake, 'candidates': [{**fake['candidates'][0], 'workflow': 'use electricity data'}]}, domain='D01', legal_refs=['source/x/y:1'])
        ok = False
    except ValueError:
        pass
    check('formation parser: two candidates with optional principles; a data-source name is rejected', ok)
    res = {'status': 'PASS' if all(c['ok'] for c in checks) else 'FAIL', 'checks': checks, 'real_fits': 0, 'llm_requests': 0, 'finished_local': now()}
    context.write_json(out / 'smoke_result.json', res)
    print('SMOKE_RESULT', res['status'], sum(c['ok'] for c in checks), '/', len(checks), flush=True)
    return res


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
    ap.add_argument('--stage-worker')
    ap.add_argument('--worker-build', nargs=3, metavar=('ROOT', 'SPLIT', 'CASE'))
    ap.add_argument('--worker-material', nargs=4, metavar=('ROOT', 'SPLIT', 'CASE', 'PHYS'))
    ap.add_argument('--worker-fit', nargs=5, metavar=('ROOT', 'SPLIT', 'CASE', 'PHYS', 'SEED'))
    ap.add_argument('--tag', default='')
    ap.add_argument('--n-updates', type=int, default=None)
    ap.add_argument('--worker-label', nargs=5, metavar=('STAGE', 'BRANCH', 'SPLIT', 'CASE', 'STAGE_ROOT'))
    a = ap.parse_args()
    root = Path(a.root)
    if a.worker_build:
        worker_build(*a.worker_build)
    elif a.worker_material:
        worker_material(*a.worker_material)
    elif a.worker_fit:
        worker_fit(*a.worker_fit[:4], int(a.worker_fit[4]), tag=a.tag, n_updates=a.n_updates)
    elif a.worker_label:
        worker_label(*a.worker_label)
    elif a.stage_worker:
        stage_worker(Path(a.stage_worker))
    elif a.preflight:
        print(json.dumps({k: v for k, v in preflight(root).items() if k in ('package', 'split_check', 'environment', 'total')}, indent=1, ensure_ascii=False))
    elif a.smoke:
        smoke(root)
    elif a.wiring:
        print(json.dumps(wiring(root), indent=1, ensure_ascii=False))
    elif a.run:
        package_run(root)
    elif a.resume_stage:
        resume_stage(root, a.resume_stage, accept_unknown_usage=a.accept_unknown_usage, restart=[x for x in a.restart.split(',') if x])
    elif a.result:
        readout(root)
        print((root / 'tables.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
