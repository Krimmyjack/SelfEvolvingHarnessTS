"""DEV-DATA-READINESS-DOMAIN-SKILL-V1 study (docs/DEV_DATA_READINESS_DOMAIN_SKILL_V1_TASK_2026-09-17.md).

One package, one ledger: preflight (0 cost) -> 2 wiring fits -> Source (no-Skill Fast, C_B only) -> formation (census +
one Slow per domain) -> Select (NO_SKILL/W1/W2 on V1, every commit before any C_B) -> freeze -> Target (Baseline-Linear,
Fixed-Seasonal, no_skill, generic, known_domain, random; C_B then E + shadow after every planned arm) -> readout.

Reused unchanged: the Fast loop (batch_research.run_job via run_batch_research_v1.branch/resume_branch), the metered client
and ledger, rt.fit (serial reserved fits, same-configuration retry), the domain Skill carrier / propose / select / freeze,
the Generic control text, the E cohort barrier and the labels-withheld resume boundary. Numerics: batch_base.readiness.

  --smoke                 the new-risk smoke (0 real fits, 0 API, 0 label rows of any planned job)
  --run                   the whole package (stops at the first stage that does not finish; no automatic extension)
  --resume-stage STAGE    one operator continuation of a stage whose labels are still withheld
  --result                readout only
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha.batch_base import context
from methods.ttha.batch_base import readiness as rd
from evaluation.main_protocol_p4 import batch_research_domain_skill as dsks
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_source_process as sp
from evaluation.main_protocol_p4 import run_batch_research_v1 as base

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_data_readiness_domain_skill_v1'
TASK = 'docs/DEV_DATA_READINESS_DOMAIN_SKILL_V1_TASK_2026-09-17.md'
SEEDS = (20260922, 20260923, 20260924)
LIMITS = {'max_calls': 16, 'max_tools': 24, 'max_new_evaluations': 2}
MAX_TOOL_CORRECTIONS = 2
EVIDENCE_ROUNDTRIP = True
TOTAL = {'max_fit_attempts': 220, 'max_llm_requests': 356, 'max_llm_tokens': 7_000_000, 'max_wall_s': 28800, 'max_retries': 2}
HTTP_CAP = 712
ALLOC = {'wiring': {'fits': 2, 'requests': 0, 'tokens': 0},
         'source': {'fits': 48, 'requests': 64, 'tokens': 1_200_000},
         'formation': {'fits': 48, 'requests': 100, 'tokens': 2_200_000},
         'target': {'fits': 120, 'requests': 192, 'tokens': 3_600_000}}
DOMAINS = ('RD01', 'RD02')
SOURCE_JOBS = ('RD01_S1', 'RD02_S1', 'RD01_S2', 'RD02_S2')
SELECT_JOB = {'RD01': 'RD01_V1', 'RD02': 'RD02_V1'}
TARGET_JOBS = ('RD01_T1', 'RD02_T1', 'RD01_T2', 'RD02_T2')
TARGET_ORDER = {'RD01_T1': ('no_skill', 'generic', 'known_domain', 'random'), 'RD02_T1': ('generic', 'known_domain', 'no_skill', 'random'),
                'RD01_T2': ('known_domain', 'no_skill', 'generic', 'random'), 'RD02_T2': ('no_skill', 'known_domain', 'generic', 'random')}
TARGET_RANDOM_SEEDS = {'RD01_T1': (2026091701, 2026091702), 'RD02_T1': (2026091703, 2026091704),
                       'RD01_T2': (2026091705, 2026091706), 'RD02_T2': (2026091707, 2026091708)}
TASK_TABLE = {'RD01': {3360: (11.8769, 83), 4560: (22.1819, 40), 5760: (6.3105, 70), 6960: (3.6923, 55), 8160: (9.2448, 95)},
              'RD02': {3360: (4.1171, 171), 4560: (0.6076, 384), 5760: (1.0789, 292), 6960: (0.2232, 286), 8160: (2.2073, 161)}}
MODEL = {'requested': 'cpa-grok-4.6', 'returned_required': 'grok-4.6-build', 'temperature': 0, 'base_url': 'http://127.0.0.1:8318/v1'}
BASELINES = (('Baseline_Linear', rd.BASELINE_LINEAR), ('Fixed_Seasonal', rd.FIXED_SEASONAL))
ALLOWED_FEATURES = frozenset('batch_median_' + f for f in rd.FIELDS)


def ds_of(job: str) -> str:
    return job.split('_')[0]


def n_of(job: str) -> int:
    return len(rd.DATASETS[ds_of(job)]['roster'])


# ============================================================================= public contracts and prompts (frozen before the first Source call)
CONTRACTS = {
    'overview': {'arguments': {}, 'meaning': 'All N entity rows of T-only, missing-aware observations (formulas in field_definitions), batch summaries, the Consumer, the legal training-window rule and the closed action table with operator semantics.'},
    'inspect_data': {'arguments': {'entity_indices': '[1..8 integers in 0..N-1]', 'kind': 'gaps|segment|daily_means|hour_profile', 'sub_range': 'optional [absolute_start, absolute_end): ABSOLUTE row indices inside overview.train_rows (not 0-based offsets); segment at most 336 rows (default: last 168 T rows)'},
                     'meaning': 'Raw current-T views with missing positions kept (null), normalized by the frozen T scaler, with absolute row indices.'},
    'build_material': {'arguments': {'plan_id': 'new short alphanumeric/underscore id',
                                     'policy': {'default': {'steps': [{'op': '<op from overview.actions.ops>', '<parameter>': '<legal value>'}]},
                                                'rules': [{'when': {'feature': '<overview field>', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': []}],
                                                'rationale': '<current hypothesis; no data source or job name>', 'observation_fields_used': []}},
                       'meaning': 'Construct one complete whole-batch input-preparation plan; does not train. First matching rule wins, no rule = default, a null field never fires a rule. At most 8 rules and 3 steps per program; a parameter with a single legal value may be omitted; empty steps = Baseline-Linear. Thresholds are numbers or a current-batch quantile. The entity program runs on every legal training X window of that entity and later on each of its 192-point prediction inputs; y, the legal window set and the scaler never change. Plans with identical canonical assignments are aliases (no second fit).'},
    'inspect_material': {'arguments': {'plan_id': '<built id>', 'entity_index': 'optional integer: also return one concrete window (raw / Baseline-Linear / prepared)', 'window': "optional 'most_missing' (default) or 'last'"},
                         'meaning': 'T-only diagnostics of what the plan changed: filled points, changed observed values, mean/volatility change against Baseline-Linear, operator fallbacks. Not predictive utility.'},
    'evaluate': {'arguments': {'plan_id': '<built id>'}, 'meaning': 'Train the complete shared plan under 3 frozen paired seeds (same legal windows, targets, scaler and sampling sequence for every plan) and return C_A only: missing-aware loss on observed future cells. At most 2 new complete plans.'},
    'compare': {'arguments': {'a': '<evaluated id>', 'b': '<evaluated id>'}, 'meaning': 'Positive paired delta means b has lower C_A loss; per seed, origin and entity; seed uncertainty, not future certainty.'},
    'commit': {'arguments': {'plan_id': '<evaluated id>', 'reason': '<brief evidence-based decision>'}, 'meaning': 'Final action. Any evaluated plan including Baseline_Linear or Fixed_Seasonal. No automatic argmin, no later selection override.'},
}

FAST_SYSTEM = '''You are the Fast path of a batch data-readiness research Harness. Your unit is one entire batch of N entities (see overview) and one shared MLP Consumer, not separate per-entity models.
你的任务是为当前整批数据构造可执行的输入准备流程，使固定共享 Consumer 对原始观测未来的预测更好。阅读全批现象及算子语义，用实际工具证据决定是否补缺、处理异常候选或去噪，组织预算内比较，允许直接保留基线。数据有缺口不等于复杂插补更好；高峰不等于错误；更平滑不等于更有训练价值。只使用 T 和本轨迹 C_A。单实体预测变化不是该实体材料的独立因果贡献。输出真实工具动作，不输出伪执行的说明文字。
(Task: build an executable input-preparation workflow for the whole current batch so that the fixed shared Consumer predicts the originally observed future better. Read the batch-wide phenomena and the operator semantics; decide from actual tool evidence whether to complete gaps, treat anomaly candidates or denoise; organize comparisons within budget; keeping a baseline directly is allowed. A gap does not mean complex imputation is better; a peak is not an error; smoother is not more valuable for training. Use only T and this trajectory's C_A. A change in one entity's prediction is not the independent causal contribution of that entity's material. Output real tool actions, not text that pretends execution.)
Baseline_Linear and Fixed_Seasonal are fully evaluated common baselines; you may commit either without spending more fits. No C_B or E is available; do not invent future results. Uniform and heterogeneous plans are both legal; non-identity programs, many observations, heterogeneity and multi-step programs are not goals. Guidance is advice and does not prohibit public tools. Return exactly one JSON object {"actions":[{"tool":"listed_name","arguments":{...}}]}, no markdown. Batched actions execute in order and must use valid IDs. Commit must be last. State brief hypotheses in material rationale or commit reason, not hidden chain-of-thought. Do not include data source or job identifiers in rationale. Do not change seeds, model, training budget, scaler, legal windows, targets, scoring or population.'''

SLOW_SYSTEM = '''You are the offline Slow of a batch data-readiness research Harness. You receive a deterministic census of the completed or explicitly failed Source branches of ONE neutral domain id: T-only missing-aware observations, the actual Fast trajectories (tool calls and returned results), complete input-preparation plans, material diagnostics, C_A feedback, commits, post-commit delayed C_B, costs and failures. No E values exist in this input, and no Select or Target data. The branches ran the same no-Skill Fast start on different earlier batches; they are not repeats. Propose at most TWO reusable domain Workflows for future Fast jobs of this domain, or KEEP if the census does not support a reusable Workflow. A Workflow explains what to observe, which problems to distinguish (input gaps, anomaly candidates, noise, level or volatility change), how to construct and compare complete preparation plans within the fixed budget, and when to stop or keep a baseline. It may recommend concrete operators or plans; an untested action is not harmful, a gap does not imply that complex imputation is better, a peak is not an error and smoother is not more valuable. The two candidates must follow genuinely different approaches; neither needs to be complex or aggressive. Optional Principles state short evidence limits or cautions. Do not change model, scoring, tools, permissions, budgets, targets or legal windows; distinguish observed facts from hypotheses; shared-plan loss differences are not causal entity labels. Do not write data source names, job labels, historical entity ids or a whole-batch answer table. Output exact JSON, no markdown: {"decision":"KEEP","rationale":"..."} OR {"decision":"PROPOSE","candidates":[{"candidate_id":"W1","research_mode":"<short label>","workflow":"...","principles":"..." or null,"observable_applicability":{"const":true},"applicability_summary":"<=400 characters","evidence_refs":["copied from legal_evidence_refs"],"rationale":"..."}]}. Workflow plus Principles render to at most 1200 characters. const:true is explicit unconditional; null is incomplete. Conditional scopes use only supplied batch feature names with numeric {feature,op,value} leaves. KEEP is valid and will not be resampled; every candidate is CANDIDATE_TEST_ONLY until real selection.'''


def skill_text_check(text: str, where: str) -> None:
    rd.validate_text(text, where, allow_entity_ids=False)


def generic_control_record() -> dict:
    return dsks.generic_control_record()


def generic_knowledge() -> br.Knowledge:
    return dsks.generic_knowledge(generic_control_record())


def knowledge_from_json(d: dict) -> br.Knowledge:
    return br.Knowledge(int(d['version']), tuple(br.Guidance(e['hook'], e['body'], copy.deepcopy(e['applicability']), tuple(e['evidence_refs'])) for e in d['entries']))


# ============================================================================= adapter (same BatchAdapter protocol and rt.Adapter hooks)
class ReadinessAdapter(rt.Adapter):
    def __init__(self, run_dir, job, ledger, repo, *, tool_error_feedback=False, roster=None, seeds=SEEDS, dataset=None):
        self.ctx = rd.open_job(dataset or ds_of(job), job, run_dir, 'material', roster=roster)
        self.seeds = tuple(seeds)
        self.ledger, self.repo = ledger, repo
        self.refs, self.specs, self.fitted = {}, {}, {}
        self.tool_error_feedback = tool_error_feedback
        self.n = len(self.ctx.job.roster)

    def overview(self):
        ov = br.json_copy(self.ctx.overview())
        ov['batch_features'] = {'batch_median_' + k: v['median'] for k, v in ov['summary'].items() if v is not None}
        ov['batch_feature_definition'] = 'Median across all N T-only entity values of the field; a field with no computable value is omitted.'
        return ov

    def inspect_data(self, arguments, *, remaining_seconds):
        if set(arguments) - {'entity_indices', 'kind', 'sub_range'}:
            self._reject_input('unknown inspection argument')
        idx = arguments.get('entity_indices')
        if not isinstance(idx, list) or not 1 <= len(idx) <= rd.MAX_INSPECT_ENTITIES or any(type(i) != int or not 0 <= i < self.n for i in idx):
            self._reject_input('entity_indices must be 1..%d integers in 0..%d' % (rd.MAX_INSPECT_ENTITIES, self.n - 1))
        kind = arguments.get('kind', 'gaps')
        if kind not in rd.INSPECT_KINDS:
            self._reject_input('kind must be one of %s' % list(rd.INSPECT_KINDS))
        span = arguments.get('sub_range')
        if span is not None and (not isinstance(span, list) or len(span) != 2 or any(type(x) != int for x in span) or span[0] >= span[1]):
            self._reject_input('sub_range must be two increasing integer boundaries')
        lo, hi = self.ctx.job.train_range
        if span is not None and (span[0] < lo or span[1] > hi):
            # amendment 1: rejected before any row access (the module keeps its hard PermissionError as a second guard)
            self._reject_input('sub_range must use absolute row indices inside T=[%d,%d) (overview.train_rows), not 0-based offsets' % (lo, hi))
        if kind == 'segment' and span is not None and span[1] - span[0] > rd.MAX_SEGMENT_ROWS:
            self._reject_input('segment is limited to %d rows' % rd.MAX_SEGMENT_ROWS)
        return self.ctx.inspect_data(idx, kind, span)

    def build_material(self, arguments, *, remaining_seconds):
        if set(arguments) != {'plan_id', 'policy'}:
            self._reject_input('build needs plan_id and policy')
        mid = arguments['plan_id']
        if not isinstance(mid, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}', mid):
            self._reject_input('unsafe plan id')
        if mid in self.refs:
            self._reject_input('plan id already exists; use existing plan or a new id')
        try:
            compiled = rd.compile_policy(arguments['policy'], self.overview()['entities'])
        except rd.PolicyError as exc:
            if not self.tool_error_feedback:
                raise
            raise br.ToolInputError(str(exc), valid_observation_fields=list(rd.FIELDS), legal_ops=sorted(rd.OPS)) from None
        try:
            ref = rd.build(self.ctx, compiled['assignment'], mid)
        except rd.ProgramExecutionError as exc:
            if not self.tool_error_feedback:
                raise
            raise br.ToolInputError('plan rejected at execution (no fit, no material): %s' % exc, execution_status='PROGRAM_EXECUTION_REJECTED') from None
        ms = {**rd.material_spec(compiled), 'alias_of': ref.alias_of}
        self.refs[mid], self.specs[mid] = ref, ms
        context.write_json(self.ctx.job_dir / 'materials' / (mid + '__compiled.json'),
                           {'material_spec': ms, 'assignment': compiled['assignment'], 'requested_assignment': compiled['requested_assignment']})
        return br.Candidate(mid, ms, self.ctx.job.job_id)

    def inspect_material(self, plan_id, arguments, *, remaining_seconds):
        if 'plan_id' not in arguments or set(arguments) - {'plan_id', 'entity_index', 'window'}:
            self._reject_input('inspect_material takes plan_id and optional entity_index / window')
        ei, w = arguments.get('entity_index'), arguments.get('window', 'most_missing')
        if ei is not None and (type(ei) != int or not 0 <= ei < self.n):
            self._reject_input('entity_index must be an integer in 0..%d' % (self.n - 1))
        if w not in ('most_missing', 'last'):
            self._reject_input("window must be 'most_missing' or 'last'")
        return rd.inspect_material(self.ctx, self.refs[plan_id], ei, w)

    def evaluate(self, candidate, seeds, *, feedback, remaining_seconds):
        recs = rt.fit(self.ctx, self.refs[candidate.plan_id], seeds, self.ledger, self.repo, feedback)
        if feedback and any((r['scores'].get('c_a') or {}).get('status') != 'SCORABLE' for r in recs):
            raise RuntimeError('C_A_NOT_SCORABLE')
        losses = tuple(tuple(tuple(row) for row in r['scores']['c_a']['per_origin_entity_normalized_mse']) for r in recs) if feedback else ()
        fitted = replace(candidate, model_seeds=tuple(seeds), model_refs=tuple(r['model_path'] for r in recs), ca_losses=losses)
        self.fitted[candidate.plan_id] = fitted
        return fitted

    def _load_ref(self, mid):
        self.refs[mid] = rd.MaterialRef(**rd.load_index(self.ctx.job_dir)[mid])
        self.specs[mid] = context.read_json(self.ctx.job_dir / 'materials' / (mid + '__compiled.json'))['material_spec']

    def restore(self, plan_ids):
        out = []
        for mid in plan_ids:
            if mid in self.refs:
                raise ValueError('plan already present')
            self._load_ref(mid)
            before = self.ledger.s['fit_attempts']
            out.append(self.evaluate(br.Candidate(mid, self.specs[mid], self.ctx.job.job_id), self.seeds, feedback=True, remaining_seconds=self.ledger.remaining()))
            if self.ledger.s['fit_attempts'] != before:
                raise RuntimeError('restore must not fit')
        return out

    def restore_built(self, mid):
        self._load_ref(mid)
        return br.Candidate(mid, self.specs[mid], self.ctx.job.job_id)

    def baselines(self):
        out = []
        index = rd.load_index(self.ctx.job_dir)
        for mid, pol in BASELINES:
            if mid in index:                      # copied from the job's common directory
                self._load_ref(mid)
                c = br.Candidate(mid, self.specs[mid], self.ctx.job.job_id)
            else:
                c = self.build_material({'plan_id': mid, 'policy': pol}, remaining_seconds=self.ledger.remaining())
            out.append(self.evaluate(c, self.seeds, feedback=True, remaining_seconds=self.ledger.remaining()))
        return tuple(out)


class FastClient:
    """The readiness Fast system prompt on the unchanged metered client."""
    def __init__(self, client):
        self.client = client

    def fast(self, unit):
        return lambda payload: self.client.call('fast', unit, payload, FAST_SYSTEM)


# ============================================================================= arms
def fast_branch(path, job, knowledge, led, client, shared):
    return base.branch(path, job, knowledge, led, FastClient(client), shared, evidence_roundtrip=EVIDENCE_ROUNDTRIP, max_tool_corrections=MAX_TOOL_CORRECTIONS,
                       tool_contracts=CONTRACTS, seeds=SEEDS, adapter_factory=ReadinessAdapter, limits=LIMITS, dataset=ds_of(job),
                       commit_fn=rd.commit, allowed_features=ALLOWED_FEATURES, entity_count=n_of(job))


def resume_fast_branch(path, job, knowledge, led, client):
    return base.resume_branch(path, job, knowledge, led, FastClient(client), max_tool_corrections=MAX_TOOL_CORRECTIONS, tool_contracts=CONTRACTS, seeds=SEEDS,
                              adapter_factory=ReadinessAdapter, evidence_roundtrip=EVIDENCE_ROUNDTRIP, limits=LIMITS, dataset=ds_of(job),
                              commit_fn=rd.commit, allowed_features=ALLOWED_FEATURES, entity_count=n_of(job))


def random_branch(root, job, led, shared, policy_seeds):
    """Same DSL, two frozen random complete plans, 0 LLM; commit the lowest three-seed C_A mean among Baseline_Linear,
    Fixed_Seasonal, p1, p2 (ties in that order)."""
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(shared / job, root / job)
    adapter = ReadinessAdapter(root, job, led, REPO, seeds=SEEDS)
    n = adapter.n
    candidates, trace = list(adapter.baselines()), []

    def emit(event, **kw):
        row = {'event_id': '%s:%d' % (job, len(trace)), 'event': event, **kw}
        trace.append(row)
        base.event_sink(root / 'trace.jsonl')(row)
    context.write_json(root / 'knowledge.json', asdict(br.Knowledge()))
    for i, seed in enumerate(policy_seeds, 1):
        c = adapter.build_material({'plan_id': 'p%d' % i, 'policy': rd.random_policy(int(seed))}, remaining_seconds=led.remaining())
        emit('random_material', sampling_seed=int(seed), plan_id=c.plan_id, material_spec=c.material_spec)
        c = adapter.evaluate(c, SEEDS, feedback=True, remaining_seconds=led.remaining())
        candidates.append(c)
        emit('random_evaluated', plan_id=c.plan_id, feedback=c.feedback(SEEDS, 2, n))
    order = [c.plan_id for c in candidates]
    chosen = min(candidates, key=lambda c: (c.feedback(SEEDS, 2, n)['mean_loss'], order.index(c.plan_id)))
    rd.commit(root, ds_of(job), job, chosen.plan_id, 'Random control: minimum three-seed C_A mean among Baseline_Linear, Fixed_Seasonal, p1, p2; ties in that order.', SEEDS[0])
    emit('committed', plan_id=chosen.plan_id, delivery_model_ref=chosen.model_refs[0])
    result = br.RunResult('COMPLETE', job, 0, chosen.plan_id, chosen.model_refs[0], 0, 0, len(policy_seeds), trace)
    context.write_json(root / 'branch_result.json', asdict(result))
    print('RANDOM', job, 'commit', chosen.plan_id, flush=True)
    return result


# ============================================================================= budget (one package ledger, cumulative stage caps)
def paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'configs': root / 'stage_configs', 'logs': root / 'logs',
            'wiring': root / 'wiring_check', 'source': root / 'source', 'formation': root / 'formation', 'select': root / 'select', 'target': root / 'target'}


def stage_caps(root: Path, stage: str) -> dict:
    return dsks.stage_caps(root, stage, total=TOTAL, alloc=ALLOC, http_cap=HTTP_CAP, stages_path=paths(root)['stages'], ledger_path=paths(root)['ledger'])


# ============================================================================= stage worker (subprocess; labels withheld on any stop)
def label_stage(branch_dir: Path, job: str, name: str, led) -> None:
    cmd = [sys.executable, '-B', '-m', rd.FIT_MODULE, '--stage', name, '--output', str(branch_dir), '--dataset', ds_of(job), '--job', job]
    pr = subprocess.run(cmd, cwd=REPO, timeout=max(1.0, led.remaining()))
    if pr.returncode:
        raise RuntimeError('label stage failed: %s %s' % (name, branch_dir.name))


def _safe_message(exc) -> str | None:
    return None if isinstance(exc, (rt.llm.AccountFault, rt.llm.TransportFault)) else str(exc)[:300]


def open_labels(root: Path, cfg: dict, branches, failures, led, *, withhold_reason=None, resumed=False, runner=label_stage) -> None:
    if withhold_reason:
        context.write_json(root / 'labels_withheld.json', {'epoch': time.time(), 'reason': withhold_reason, 'branches': [(str(p), j) for p, j in branches],
                                                          'unknown_usage': led.s['llm_tokens_unknown'], 'next': '--resume-stage (operator decision) or stop'})
        print('LABELS_WITHHELD', withhold_reason, flush=True)
        return
    eligible = [(p, j) for p, j in branches if (p / j / 'commit.json').exists()]
    try:
        for p, j in eligible:
            if not (p / j / 'c_b_scores.json').exists():
                runner(p, j, 'c_b', led)
        if not cfg['labels_e']:
            context.write_json(root / 'labels_c_b_only.json', {'epoch': time.time(), 'branches': [str(p) for p, j in eligible], 'resumed': resumed})
        else:
            for p, j in eligible:
                if not (p / j / 'e_frozen.json').exists():
                    runner(p, j, 'freeze_e', led)
            context.write_json(root / 'all_e_predictions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in eligible], 'resumed': resumed})
            for p, j in eligible:
                if not (p / j / 'e_scores.json').exists():
                    runner(p, j, 'score_e', led)
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'external', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
        print('EXTERNAL_STOP', type(exc).__name__, flush=True)
    context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': resumed})


def _run_arms(root: Path, cfg: dict, led, client, branches, failures, *, resumed: bool) -> None:
    for job in cfg['jobs']:
        shared = root / (job + '_common')
        if (root / (job + '_not_scorable.json')).exists():
            continue
        shared.mkdir(exist_ok=resumed)
        try:
            ReadinessAdapter(shared, job, led, REPO, seeds=SEEDS).baselines()
        except RuntimeError as exc:
            if str(exc) != 'C_A_NOT_SCORABLE':
                raise
            context.write_json(root / (job + '_not_scorable.json'), {'job': job, 'status': 'C_A_NOT_SCORABLE', 'note': 'common observed-cell mask below the 25% rule; no arm of this job is run'})
            failures.append({'job': job, 'kind': 'C_A_NOT_SCORABLE'})
            continue
        for arm in cfg['order'][job]:
            path = root / ('%s_%s' % (job, arm))
            kn = cfg['knowledge'].get(job, {}).get(arm)
            if arm != 'random' and kn is None:
                if not (root / ('%s_%s_treatment.json' % (job, arm))).exists():
                    context.write_json(root / ('%s_%s_treatment.json' % (job, arm)), {'status': 'NO_TREATMENT', 'note': 'no frozen card for this arm; not run, no substitute, no budget transfer'})
                continue
            branches.append((path, job))
            try:
                if path.exists() and (path / 'branch_result.json').exists():
                    prior = context.read_json(path / 'branch_result.json')
                    if prior['status'] == 'COMPLETE':
                        result = br.RunResult(**{k: v for k, v in prior.items() if k != 'trace'})
                    elif arm == 'random':
                        raise RuntimeError('control branch cannot be resumed')
                    else:
                        result = resume_fast_branch(path, job, knowledge_from_json(kn), led, client)
                elif path.exists():
                    raise RuntimeError('branch directory without a result; inspect before any replay')
                elif arm == 'random':
                    result = random_branch(path, job, led, shared, cfg['random_policy_seeds'][job])
                else:
                    result = fast_branch(path, job, knowledge_from_json(kn), led, client, shared)
                if result.status != 'COMPLETE':
                    failures.append({'branch': path.name, 'kind': result.failure_kind, 'reason': result.reason})
            except Exception as exc:  # noqa: BLE001
                failures.append({'branch': path.name, 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
                print('BRANCH_FAILED', path.name, type(exc).__name__, flush=True)
            led.check_wall()
            led.check_llm()
            if getattr(client, 'fatal', False) or rt.unknown_usage_blocks(led):
                raise RuntimeError('backend fatal or unknown usage')


def stage_worker(cfg_path: Path, *, client=None, runner=label_stage) -> None:
    cfg = context.read_json(cfg_path)
    root = Path(cfg['output'])
    if (root / 'experiment_started.json').exists():
        raise RuntimeError('stage already started; no paid replay')
    root.mkdir(parents=True, exist_ok=True)
    context.write_json(root / 'config.json', cfg)
    led = rt.RuntimeLedger(cfg['ledger_path'], **cfg['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage; paid stage refused without an operator decision')
    rt.FIT_RETRY = bool(cfg.get('fit_retry'))
    context.write_json(root / 'experiment_started.json', {'epoch': time.time(), 'pid': os.getpid(), 'stage': cfg['stage']})
    branches, failures, stopped = [], [], None
    try:
        client = client or rt.MeteredClient(led, root / 'raw_responses', http_cap=int(cfg['http_cap']))
        _run_arms(root, cfg, led, client, branches, failures, resumed=False)
        context.write_json(root / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches]})
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
        stopped = type(exc).__name__
        print('EXECUTION_STOP', stopped, flush=True)
    finally:
        context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures})
    open_labels(root, cfg, branches, failures, led, runner=runner,
                withhold_reason=stopped and 'execution stopped before every planned arm was attempted (%s)' % stopped)


def resume_stage(root: Path, stage: str, *, accept_unknown_usage: bool = False, restart=()) -> None:
    """One operator continuation: labels must still be withheld (existing boundary check); completed branches kept, a
    never-sent Fast call continued, missing arms run as planned, then labels. Unknown usage needs an explicit acceptance."""
    from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp
    sroot = paths(root)[stage]
    cfg = context.read_json(sroot / 'config.json')
    if (sroot / 'resume.json').exists():
        raise RuntimeError('already resumed once; no second replay')
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
            shutil.move(sroot / name, sroot / name.replace('.json', '_before_resume.json'))
    moved = []
    for name in restart:                 # technically interrupted Fast branches with no new evaluation: kept aside, run again from the common start
        b = sroot / name
        prior = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
        if not b.is_dir() or (prior and (prior['status'] == 'COMPLETE' or prior['new_evaluations'])) or any(b.rglob('commit.json')):
            raise RuntimeError('only an interrupted branch without commit or new evaluation can be restarted: %s' % name)
        dest = sroot / ('%s__interrupted_1' % name)
        shutil.move(b, dest)
        moved.append({'branch': name, 'kept_as': dest.name, 'prior_status': prior and prior['status'], 'prior_failure_kind': prior and prior['failure_kind']})
    context.write_json(sroot / 'resume.json', {'epoch': time.time(), 'accepted_unknown_usage': bool(accept_unknown_usage), 'slow_recalled': False,
                                               'restarted_branches': moved})
    rt.FIT_RETRY = bool(cfg.get('fit_retry'))
    branches, failures, stopped = [], [], None
    try:
        client = rt.MeteredClient(led, sroot / 'raw_responses', http_cap=int(cfg['http_cap']))
        _run_arms(sroot, cfg, led, client, branches, failures, resumed=True)
        context.write_json(sroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches], 'resumed': True})
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': _safe_message(exc)})
        stopped = type(exc).__name__
    finally:
        context.write_json(sroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': True})
    open_labels(sroot, cfg, branches, failures, led, resumed=True, withhold_reason=stopped and 'resumed stage stopped again (%s)' % stopped)


def stage_config(root: Path, stage: str, *, jobs, order, knowledge, labels_e: bool, random_seeds=None) -> Path:
    P = paths(root)
    path = P['configs'] / ('%s.json' % stage)
    if path.exists():
        return path
    sc = stage_caps(root, 'formation' if stage == 'select' else stage)
    cfg = {'study': 'data_readiness', 'status': 'FROZEN', 'stage': stage, 'output': str(P[stage]), 'jobs': list(jobs),
           'order': {j: list(order[j]) for j in jobs}, 'knowledge': knowledge, 'random_policy_seeds': {j: list(s) for j, s in (random_seeds or {}).items()},
           'labels_e': bool(labels_e), 'seeds': list(SEEDS), 'limits': LIMITS, 'max_tool_corrections': MAX_TOOL_CORRECTIONS,
           'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'caps': sc['caps'], 'http_cap': sc['http_cap'], 'ledger_path': str(P['ledger']), 'fit_retry': True,
           'fit_attempts_at_stage_start': context.read_json(P['ledger'])['fit_attempts']}
    dsks.write_once(path, cfg)
    return path


def run_stage(root: Path, stage: str, cfg_path: Path) -> dict:
    P = paths(root)
    sroot = P[stage]
    if not sroot.exists():
        led = context.read_json(P['ledger'])
        remaining = TOTAL['max_wall_s'] - (time.time() - led['started_epoch'])
        if remaining <= 0:
            raise RuntimeError('package wall exhausted')
        P['logs'].mkdir(parents=True, exist_ok=True)
        print('STAGE_START', stage, flush=True)
        with (P['logs'] / ('%s.log' % stage)).open('a', encoding='utf-8') as log:
            p = subprocess.Popen([sys.executable, '-B', '-m', 'evaluation.main_protocol_p4.batch_research_data_readiness', '--stage-worker', str(cfg_path)],
                                 cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)
                context.write_json(root / ('%s_supervisor_timeout.json' % stage), {'epoch': time.time(), 'pid': p.pid})
                rc = 124
        print('STAGE_EXIT', stage, rc, flush=True)
    return dsks.stage_status(sroot)


# ============================================================================= preflight (0 cost) and wiring fits
def preflight(root: Path) -> dict:
    import inspect
    from SelfEvolvingHarnessTS.operators import _provenance
    P = paths(root)
    if (root / 'package_config.json').exists():
        return context.read_json(root / 'package_config.json')
    eligibility = recompute_eligibility()
    src = inspect.getsource(rt.MeteredClient)
    if not all(x in src for x in ("model='cpa-grok-4.6'", "'grok-4.6-build'", 'temperature=0', "base_url='http://127.0.0.1:8318/v1'")):
        raise RuntimeError('metered client no longer carries the frozen model identity')
    gc = generic_control_record()
    if gc['body'] != sp.GENERIC_TEXT or generic_knowledge() != br.Knowledge(1, (br.Guidance('experiment_guidance', sp.GENERIC_TEXT, {'const': True}, ('generic:frozen',)),)):
        raise RuntimeError('generic control differs from the historical Generic')
    random_supply = {j: {'seeds': list(s), 'policies': [rd.random_policy(x) for x in s]} for j, s in TARGET_RANDOM_SEEDS.items()}
    cfg = {'package': 'DEV-DATA-READINESS-DOMAIN-SKILL-V1', 'task': TASK, 'exposure': rd.EXPOSURE, 'frozen_epoch': time.time(),
           'freeze_note': 'Frozen before the first paid call. The Planner and this preflight read planned T rows for eligibility only; the configuration was not frozen before any T was touched.',
           'datasets': {d: {k: (str(v) if isinstance(v, Path) else v) for k, v in rd.DATASETS[d].items()} for d in DOMAINS},
           'eligibility_recomputed': eligibility, 'job_t': rd.JOB_T,
           'geometry': {'L': rd.L, 'H': rd.H, 'T': rd.TRAIN_SPAN, 'c_a': 't, t+48', 'c_b': 't+96, t+144', 'e': 't+192..t+336 step 48',
                        'legal_parent': 'X >= 32 finite and y fully finite', 'coverage_min': rd.COVERAGE_MIN},
           'stages': {'source': list(SOURCE_JOBS), 'select': SELECT_JOB, 'target': list(TARGET_JOBS), 'target_order': {j: list(o) for j, o in TARGET_ORDER.items()}},
           'seeds': list(SEEDS), 'limits': LIMITS, 'max_tool_corrections': MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': EVIDENCE_ROUNDTRIP,
           'model': MODEL, 'total_caps': TOTAL, 'http_cap': HTTP_CAP, 'stage_allocations': ALLOC,
           'fit_budget_formula': 'Source 4x(6+6)=48; formation+Select 2x(6+3x6)=48; Target 4x(6+3x6+6)=120; +2 wiring +2 same-configuration retries = 220',
           'actions': rd.action_table(), 'fields': list(rd.FIELDS), 'field_definitions': rd.FIELD_DEFINITIONS, 'consumer': rd.CONSUMER,
           'contracts': CONTRACTS, 'fast_system': FAST_SYSTEM, 'slow_system': SLOW_SYSTEM, 'generic_control': gc,
           'baselines': dict(BASELINES), 'random_supply': {'rule': rd.random_policy.__doc__, 'jobs': random_supply},
           'select_rule': 'per domain on V1: NO_SKILL, W1, W2 each commit first; then C_B of each own commit; argmin three-seed macro; ties NO_SKILL, W1, W2',
           'metric': 'missing-aware normalized MSE per entity over observed origin x horizon cells, equal-weight entity macro; block NOT_SCORABLE if an entity < 25% coverage',
           'shadow': 'same fitted model fed the Baseline-Linear inputs (0 fits, 0 LLM); compared with the Baseline-Linear model; not used for delivery or selection',
           'dependency_fingerprint': _provenance.dependency_fingerprint()}
    root.mkdir(parents=True, exist_ok=True)
    dsks.write_once(root / 'package_config.json', cfg)
    return cfg


def recompute_eligibility() -> dict:
    """T rows only: roster rule, missing rate and minimum legal parents per planned T vs the task table (4 decimals)."""
    import numpy as np
    out = {}
    for d in DOMAINS:
        rows = {}
        for suffix, t in rd.JOB_T.items():
            sl = rd.load_slice(d, t - rd.TRAIN_SPAN, t)
            seg = sl.rows(t - rd.TRAIN_SPAN, t)
            lg = rd.legal_parents(seg)
            miss = float(np.isnan(seg).mean() * 100)
            fin = np.isfinite(seg)
            ok = all(fin[:, e].sum() >= rd.MIN_T_FINITE and seg[fin[:, e], e].std() > 1e-6 and lg.counts[e] >= rd.MIN_LEGAL_PARENTS for e in range(seg.shape[1]))
            want = TASK_TABLE[d][t]
            rows[suffix] = {'t': t, 'missing_rate_pct': round(miss, 4), 'min_legal_parents': int(lg.counts.min()), 'legal_parents_total': lg.n, 'all_entities_eligible': ok,
                            'matches_task_table': round(miss, 4) == want[0] and int(lg.counts.min()) == want[1]}
            if not ok or not rows[suffix]['matches_task_table']:
                raise RuntimeError('eligibility/table mismatch for %s %s: %s' % (d, suffix, rows[suffix]))
        out[d] = {'roster': rd.DATASETS[d]['roster'], 'jobs': rows}
    out['RD01']['roster_rule_recomputed'] = _rd01_roster_rule() == rd.RD01_ROSTER
    if not out['RD01']['roster_rule_recomputed']:
        raise RuntimeError('RD01 roster rule does not reproduce the frozen roster')
    return out


def _rd01_roster_rule() -> list:
    import io, zipfile
    import numpy as np
    path = rd.DATASETS['RD01']['path']
    toks = {}
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as h:
        in_data = False
        for line in io.TextIOWrapper(h, encoding='utf-8'):
            if not in_data:
                in_data = line.strip().lower() == '@data'
                continue
            f = line.rstrip().split(':')
            toks[f[0]] = f[-1].split(',')
    roster = []
    for uid in sorted(toks)[80:]:
        ok = True
        for t in rd.JOB_T.values():
            seg = np.array([np.nan if x == '?' else float(x) for x in toks[uid][t - rd.TRAIN_SPAN:t]])[:, None]
            fin = np.isfinite(seg[:, 0])
            if fin.sum() < rd.MIN_T_FINITE or not seg[fin, 0].std() > 1e-6 or rd.legal_parents(seg).counts[0] < rd.MIN_LEGAL_PARENTS:
                ok = False
                break
        if ok:
            roster.append(uid)
        if len(roster) == 32:
            break
    return roster


def wiring_check(root: Path) -> dict:
    """2 real fits (Baseline-Linear, first seed, no C_A read) on the two populations: the shared model trains on the actual
    pool and predicts finite values. Charged to the package ledger; never reused as a cache."""
    import numpy as np
    P = paths(root)
    rec_path = P['wiring'] / 'wiring.json'
    if rec_path.exists():
        return context.read_json(rec_path)
    sc = stage_caps(root, 'wiring')
    led = rt.RuntimeLedger(P['ledger'], **sc['caps'])
    rt.FIT_RETRY = False
    out = {}
    for d in DOMAINS:
        job = d + '_S1'
        a = ReadinessAdapter(P['wiring'] / d, job, led, REPO, seeds=SEEDS)
        a.build_material({'plan_id': 'Baseline_Linear', 'policy': rd.BASELINE_LINEAR}, remaining_seconds=led.remaining())
        cell = rt.fit(a.ctx, a.refs['Baseline_Linear'], (SEEDS[0],), led, REPO, feedback=False)[0]
        X, _ = rd.training_arrays(a.ctx, a.refs['Baseline_Linear'])
        pred = rd.train.predict(rd.train.load_model(cell['model_path']), X[-64:])
        out[d] = {'job': job, 'entities': a.n, 'pool_size': cell['pool_size'], 'legal_parents': a.ctx.legal.n, 'n_updates_done': cell['train']['n_updates_done'],
                  'train_seconds': round(cell['train']['seconds'], 1), 'final_train_loss': cell['train']['final_train_loss'], 'rows_read': cell['rows_read'],
                  'prediction_shape': list(pred.shape), 'predictions_finite': bool(np.isfinite(pred).all()),
                  'ok': cell['pool_size'] == a.ctx.legal.n and cell['train']['n_updates_done'] == 2000 and bool(np.isfinite(pred).all()) and pred.shape == (64, 48)}
    out['status'] = 'PASS' if all(out[d]['ok'] for d in DOMAINS) else 'FAIL'
    context.write_json(rec_path, out)
    if out['status'] != 'PASS':
        raise RuntimeError('wiring check failed')
    return out


# ============================================================================= formation (census + one Slow per domain), select, freeze
def _branch_complete_with_c_b(bdir: Path, job: str) -> bool:
    return dsks._branch_complete_with_c_b(bdir, job)


def _trace(bdir: Path) -> list:
    tr = bdir / 'trace.jsonl'
    return [json.loads(x) for x in tr.read_text(encoding='utf-8').splitlines() if x.strip()] if tr.exists() else []


def _plan_blocks(bdir: Path, job: str) -> dict:
    out = {}
    for c in sorted((bdir / job / 'cells').glob('*.json')):
        cell = context.read_json(c)
        if cell.get('status') != 'OK':
            continue
        row = out.setdefault(cell['material_id'], {})
        sc = cell.get('scores', {})
        row[cell['model_seed']] = {b: (sc.get(b) or {}).get('normalized_mse_macro') for b in ('c_a', 'c_b')}
        row[cell['model_seed']]['c_b_status'] = (sc.get('c_b') or {}).get('status')
    return out


def census_domain(root: Path, d: str) -> dict:
    P = paths(root)
    jobs = [j for j in SOURCE_JOBS if ds_of(j) == d]
    records = []
    for j in jobs:
        ns = P['source'] / ('%s_not_scorable.json' % j)
        if ns.exists():
            cell = context.read_json(next((P['source'] / ('%s_common' % j) / j / 'cells').glob('*.json')))
            cov = cell['scores']['c_a']['coverage']
            records.append({'job': j, 'status': 'C_A_NOT_SCORABLE',
                            'note': 'common observed-cell mask of C_A below the 25% rule for some entities; baselines fitted, no Fast trajectory',
                            'c_a_observed_cells_per_entity': cov['observed_cells_per_entity'], 'cells_per_entity': cov['cells_per_entity'],
                            'overview_summary': context.read_json(P['source'] / ('%s_common' % j) / j / 'overview.json')['summary']})
    for old in sorted(P['source'].glob('%s_*__interrupted_*' % d)):
        r = old / 'branch_result.json'
        records.append({'branch_attempt': old.name, 'status': 'INTERRUPTED_TECHNICAL',
                        'result': ({k: v for k, v in context.read_json(r).items() if k != 'trace'} if r.exists() else None),
                        'note': 'technical/operator interruption before any new evaluation; restarted under the amended interface; not a scientific outcome'})
    branches = [(P['source'] / ('%s_no_skill' % j), j) for j in jobs if (P['source'] / ('%s_no_skill' % j)).exists()]
    complete = [(b, j) for b, j in branches if _branch_complete_with_c_b(b, j)]
    for b, j in branches:
        if (b, j) not in complete:
            r = b / 'branch_result.json'
            records.append({'job': j, 'status': 'INCOMPLETE', 'result': ({k: v for k, v in context.read_json(r).items() if k != 'trace'} if r.exists() else None)})
    if not complete:
        return {'census_status': 'SOURCE_INCOMPLETE', 'domain_id': d, 'records': records, 'legal_evidence_refs': [],
                'note': 'no COMPLETE Source trajectory with post-commit C_B in this domain; no substitute trajectory'}
    branches = complete
    refs, out = [], []
    for b, j in branches:
        res = context.read_json(b / 'branch_result.json')
        commit_rec = context.read_json(b / j / 'commit.json')
        rows = []
        for r in _trace(b):
            r = copy.deepcopy(r)
            if r['event'] == 'job_started':
                r['overview'] = 'see overview (identical)'
            if r['event'] == 'tool_completed' and r['tool'] == 'overview':
                r['output'] = 'see overview (identical)'
            rows.append(r)
            refs.append(r['event_id'])
        ov = context.read_json(b / j / 'overview.json')
        index = rd.load_index(b / j)
        blocks = _plan_blocks(b, j)
        plans = {}
        for mid, rec in index.items():
            comp = context.read_json(b / j / 'materials' / (mid + '__compiled.json'))
            summ = context.read_json(rec['summary_path'])
            tot = {k: int(sum(e[k] for e in summ['entities'])) for k in ('windows', 'windows_with_missing', 'filled_points', 'observed_points', 'observed_changed_points')}
            sc = blocks.get(mid, {})
            seeds_ok = set(sc) == set(SEEDS)
            plans[mid] = {'material_spec': comp['material_spec'], 'alias_of': rec['alias_of'], 'material_totals': tot,
                          'c_a_by_seed': [sc[s]['c_a'] for s in SEEDS] if seeds_ok else None,
                          'c_b_after_commit_by_seed': [sc[s]['c_b'] for s in SEEDS] if seeds_ok else None,
                          'c_b_status': sorted({sc[s]['c_b_status'] for s in SEEDS}) if seeds_ok else None, 'fitted': seeds_ok}
            for k in ('c_a_by_seed', 'c_b_after_commit_by_seed'):
                v = plans[mid][k]
                plans[mid][k.replace('_by_seed', '_mean')] = statistics.mean(v) if v and all(x is not None for x in v) else None
        refs += ['%s:overview' % j, '%s:c_b_after_commit' % j]
        tokens = dsks._unit_tokens(P['source']).get('fast:%s' % b.name, {})
        out.append({'job': j, 'branch_status': res['status'], 'committed_plan_id': res['committed_plan_id'], 'commit_reason': commit_rec['reason'],
                    'cost': {'fast_calls': res['calls'], 'tool_calls': res['tool_calls'], 'new_evaluations': res['new_evaluations'], 'fast_tokens': tokens},
                    'failure_kind': res['failure_kind'],
                    'overview': {'reference': '%s:overview' % j, 'n_entities': ov['n_entities'], 'summary': ov['summary'], 'entities': ov['entities'],
                                 'geometry': ov['geometry']},
                    'plans': plans, 'delayed_block': {'reference': '%s:c_b_after_commit' % j, 'block': 'C_B (opened after this branch committed)'},
                    'trajectory': rows})
    ev = {'census_status': 'CENSUS_COMPLETE', 'domain_id': d, 'profile': 'data_readiness', 'branches': out, 'structural_and_technical_records': records,
          'legal_evidence_refs': sorted(set(refs)),
          'public_semantics': {'field_definitions': rd.FIELD_DEFINITIONS, 'actions': rd.action_table(), 'tool_contracts': CONTRACTS,
                               'baselines': {'Baseline_Linear': 'empty program: minimum linear completion', 'Fixed_Seasonal': 'uniform period_median_complete(24,3,2)'},
                               'metric': 'missing-aware normalized MSE on observed future cells; lower is better'},
          'purpose': 'Offline Slow input: can the Source processes of this neutral domain be consolidated into at most two reusable data-readiness Workflows?'}
    low = json.dumps(ev, ensure_ascii=False).lower()
    if any(n in low for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')):
        raise PermissionError('census names a data source')
    return ev


def formation_calls(root: Path, client=None) -> None:
    P = paths(root)
    for d in DOMAINS:
        out = P['formation'] / d
        if not (out / 'census.json').exists():
            dsks.write_once(out / 'census.json', census_domain(root, d))
    sc = stage_caps(root, 'formation')
    led = None
    for d in DOMAINS:
        out = P['formation'] / d
        if (out / 'propose.json').exists():
            continue
        census = context.read_json(out / 'census.json')
        if census['census_status'] != 'CENSUS_COMPLETE':
            dsks.write_once(out / 'propose.json', {'status': 'NO_PROPOSAL_INPUT', 'census_status': census['census_status'], 'skills': []})
            continue
        if led is None:
            led = rt.RuntimeLedger(P['ledger'], **sc['caps'])
            if rt.unknown_usage_blocks(led):
                raise RuntimeError('package ledger holds unknown usage; paid formation refused')
            client = client or rt.MeteredClient(led, P['formation'] / 'raw_responses', http_cap=sc['http_cap'])
        res = dsk.propose(census, domain_id=d, call=lambda p, d=d: client.call('slow_domain', d, p, SLOW_SYSTEM, max_tokens=8192),
                          allowed_features=ALLOWED_FEATURES, text_check=skill_text_check, forbidden_names=[n for n in rd.SOURCE_NAMES if n not in ('rd01', 'rd02')])
        dsks.write_once(out / 'propose.json', {**{k: v for k, v in res.items() if k != 'skills'}, 'skills': [s.to_json() for s in res['skills']],
                                               'census_legal_ref_count': len(census['legal_evidence_refs'])})
        if getattr(client, 'fatal', False) or rt.unknown_usage_blocks(led):
            raise RuntimeError('backend fatal or unknown usage during formation; stop')


def candidates_of(root: Path, d: str) -> list:
    prop = context.read_json(paths(root)['formation'] / d / 'propose.json')
    return [dsk.skill_from_json(s) for s in prop.get('skills', [])] if prop.get('status') == 'PROPOSED' else []


def select_stage(root: Path) -> dict:
    jobs, order, knowledge = [], {}, {}
    for d in DOMAINS:
        skills = candidates_of(root, d)
        if not skills:
            continue
        job = SELECT_JOB[d]
        jobs.append(job)
        order[job] = ['no_skill'] + ['cand_' + s.skill_id.split('-')[1] for s in skills]
        knowledge[job] = {'no_skill': asdict(br.Knowledge()), **{'cand_' + s.skill_id.split('-')[1]: asdict(dsk.skill_knowledge(s)) for s in skills}}
    if not jobs:
        dsks.write_once(paths(root)['formation'] / 'select_skipped.json', {'status': 'NO_CANDIDATES_IN_ANY_DOMAIN'})
        return {'status': 'SKIPPED'}
    cfgp = stage_config(root, 'select', jobs=jobs, order=order, knowledge=br.json_copy(knowledge), labels_e=False)
    return run_stage(root, 'select', cfgp)


def c_b_scorer(branch_dir):
    def score(option_id, job, rec):
        jdir = Path(branch_dir(option_id, job)) / job
        if not (jdir / 'c_b_scores.json').exists():
            return {'loss_by_seed': None, 'status': 'BLOCK_INCOMPLETE'}
        by_seed = {}
        for c in (jdir / 'cells').glob('*.json'):
            cell = context.read_json(c)
            cb = cell.get('scores', {}).get('c_b') or {}
            if cell.get('status') == 'OK' and cell['material_id'] == rec['committed_plan_id'] and cb.get('status') == 'SCORABLE':
                by_seed[cell['model_seed']] = cb['normalized_mse_macro']
        if not set(by_seed) >= set(SEEDS):
            return {'loss_by_seed': None, 'status': 'BLOCK_NOT_SCORABLE_OR_INCOMPLETE'}
        return {'loss_by_seed': [by_seed[s] for s in SEEDS]}
    return score


def freeze_domains(root: Path) -> None:
    P = paths(root)
    for d in DOMAINS:
        out = P['formation'] / d
        if (out / 'frozen_skill.json').exists():
            continue
        skills = candidates_of(root, d)
        prop = context.read_json(out / 'propose.json')
        if not skills:
            dsks.write_once(out / 'frozen_skill.json', {'domain_id': d, 'status': 'NO_CANDIDATE', 'skill': None, 'propose_status': prop.get('status'),
                                                        'note': 'KEEP / no proposal input / formation failure: no selection course, no card'})
            continue
        job = SELECT_JOB[d]
        arm_of = {dsk.NO_SKILL: 'no_skill', **{s.skill_id: 'cand_' + s.skill_id.split('-')[1] for s in skills}}
        options = {dsk.NO_SKILL: None, **{s.skill_id: s for s in skills}}
        bdir = lambda oid, j: P['select'] / ('%s_%s' % (j, arm_of[oid]))

        def read_run(oid, knowledge, j):
            b = bdir(oid, j)
            if not (b / 'branch_result.json').exists():
                return {'status': 'NOT_RUN', 'synthetic': False}
            if context.read_json(b / 'knowledge.json') != br.json_copy(asdict(knowledge)):
                raise RuntimeError('selection branch %s ran other knowledge than the option' % b.name)
            res = context.read_json(b / 'branch_result.json')
            return {'status': res['status'], 'committed_plan_id': res['committed_plan_id'], 'failure_kind': res['failure_kind'], 'synthetic': False, 'branch': b.name,
                    'cost': {'fast_calls': res['calls'], 'tool_calls': res['tool_calls'], 'new_evaluations': res['new_evaluations']}}
        sel = dsk.select(options, [job], read_run, block='c_b', score=c_b_scorer(bdir))
        dsks.write_once(out / 'selection.json', sel)
        dsks.write_once(out / 'frozen_skill.json', dsk.freeze(d, sel, options, evidence_scope={'census': 'formation/%s/census.json' % d, 'select_job': job, 'block': 'c_b'}))


def target_catalog(root: Path) -> dict:
    P = paths(root)
    cat = P['formation'] / 'target_catalog.json'
    if cat.exists():
        return context.read_json(cat)
    cards = {}
    for d in DOMAINS:
        fr = context.read_json(P['formation'] / d / 'frozen_skill.json')
        if fr['status'] == 'FROZEN_SELECTED':
            s = dsk.skill_from_json(fr['skill'])
            if s.status != 'FROZEN_SELECTED' or s.domain_id != d:
                raise PermissionError('only a real FROZEN_SELECTED card of its own domain enters the Target catalog')
            cards[d] = s.to_json()
        else:
            cards[d] = None
    rec = {'status': 'TARGET_FROZEN', 'epoch': time.time(), 'cards': cards, 'generic_control': generic_control_record(), 'routing': 'known_domain (runtime domain id)',
           'frozen_before_first_target_fit': True}
    dsks.write_once(cat, rec)
    return rec


def target_knowledge(cat: dict) -> dict:
    out = {}
    for job in TARGET_JOBS:
        card = cat['cards'].get(ds_of(job))
        out[job] = {'no_skill': asdict(br.Knowledge()), 'generic': asdict(generic_knowledge()),
                    'known_domain': asdict(dsk.skill_knowledge(dsk.skill_from_json(card))) if card else None, 'random': None}
    return br.json_copy(out)


# ============================================================================= package driver
def package_run(root: Path = ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    status = {}

    def stop(where, st):
        status.update({'stopped_at': where, 'stage_status': st, 'epoch': time.time()})
        context.write_json(root / 'package_status.json', status)
        print('PACKAGE_STOP', where, st.get('status'), flush=True)
        return status
    preflight(root)
    status['wiring'] = wiring_check(root)['status']
    st = run_stage(root, 'source', stage_config(root, 'source', jobs=SOURCE_JOBS, order={j: ['no_skill'] for j in SOURCE_JOBS},
                                                knowledge={j: {'no_skill': asdict(br.Knowledge())} for j in SOURCE_JOBS}, labels_e=False))
    status['source'] = st
    if st['status'] != 'FINISHED':
        return stop('source', st)
    formation_calls(root)
    st = select_stage(root)
    status['select'] = st
    if st['status'] not in ('FINISHED', 'SKIPPED'):
        return stop('select', st)
    freeze_domains(root)
    cat = target_catalog(root)
    st = run_stage(root, 'target', stage_config(root, 'target', jobs=TARGET_JOBS, order=TARGET_ORDER, knowledge=target_knowledge(cat), labels_e=True,
                                                random_seeds=TARGET_RANDOM_SEEDS))
    status['target'] = st
    if st['status'] != 'FINISHED':
        return stop('target', st)
    status['finished_epoch'] = time.time()
    context.write_json(root / 'package_status.json', status)
    package_result(root)
    print('PACKAGE_FINISHED', flush=True)
    return status


# ============================================================================= readout
def _cells(bdir: Path, job: str) -> dict:
    jdir = bdir / job
    e = context.read_json(jdir / 'e_scores.json')['cells'] if (jdir / 'e_scores.json').exists() else {}
    out = {}
    for c in sorted((jdir / 'cells').glob('*.json')) if (jdir / 'cells').exists() else []:
        cell = context.read_json(c)
        if cell.get('status') != 'OK':
            continue
        sc = cell.get('scores', {})
        ee = e.get(cell['cell_id']) or {}
        out.setdefault(cell['material_id'], {})[cell['model_seed']] = {
            'c_a': (sc.get('c_a') or {}).get('normalized_mse_macro'), 'c_b': (sc.get('c_b') or {}).get('normalized_mse_macro'),
            'c_b_shadow': (sc.get('c_b_shadow_baseline_inputs') or {}).get('normalized_mse_macro'),
            'e': (ee.get('e') or {}).get('normalized_mse_macro'), 'e_shadow': (ee.get('e_shadow_baseline_inputs') or {}).get('normalized_mse_macro'),
            'e_mae': (ee.get('e') or {}).get('mae_raw_macro'), 'e_status': (ee.get('e') or {}).get('status'),
            'e_min_coverage': ((ee.get('e') or {}).get('coverage') or {}).get('min_observed_fraction')}
    return out


def _vec(cells: dict, plan, key: str):
    if plan is None or plan not in cells or set(cells[plan]) != set(SEEDS) or any(cells[plan][s][key] is None for s in SEEDS):
        return None
    return [cells[plan][s][key] for s in SEEDS]


def _paired(a, b) -> dict | None:
    """a - b per seed; positive = b has the lower loss."""
    if a is None or b is None:
        return None
    d = [x - y for x, y in zip(a, b)]
    return {'delta_by_seed': d, 'mean': statistics.mean(d), 'se': statistics.stdev(d) / len(d) ** 0.5,
            'signs': [int(x > 0) - int(x < 0) for x in d], 'relative_to_first': statistics.mean(d) / statistics.mean(a)}


def package_result(root: Path = ROOT) -> dict:
    P = paths(root)
    res = {'package': 'DEV-DATA-READINESS-DOMAIN-SKILL-V1', 'exposure': rd.EXPOSURE, 'source': {}, 'formation': {}, 'select': {}, 'target': {}, 'costs': {}}
    for stage in ('source', 'select', 'target'):
        if (P[stage] / 'execution_finished.json').exists():
            res['%s_status' % stage] = dsks.stage_status(P[stage])
    for j in SOURCE_JOBS:
        b = P['source'] / ('%s_no_skill' % j)
        if (b / 'branch_result.json').exists():
            r = context.read_json(b / 'branch_result.json')
            cells = _cells(b, j)
            res['source'][j] = {'status': r['status'], 'commit': r['committed_plan_id'], 'behavior': dsks._behavior(b),
                                'plans': {m: {'c_a': _vec(cells, m, 'c_a'), 'c_b': _vec(cells, m, 'c_b')} for m in cells}}
    for d in DOMAINS:
        f = P['formation'] / d
        row = {}
        for name in ('propose', 'selection', 'frozen_skill'):
            if (f / (name + '.json')).exists():
                row[name] = context.read_json(f / (name + '.json'))
        res['formation'][d] = row
    for job in TARGET_JOBS:
        arms = {}
        base_cells = None
        for arm in TARGET_ORDER[job]:
            b = P['target'] / ('%s_%s' % (job, arm))
            if not (b / 'branch_result.json').exists():
                t = P['target'] / ('%s_%s_treatment.json' % (job, arm))
                arms[arm] = {'status': context.read_json(t)['status'] if t.exists() else 'NOT_RUN'}
                continue
            r = context.read_json(b / 'branch_result.json')
            cells = _cells(b, job)
            base_cells = base_cells or cells
            plan = r['committed_plan_id']
            beh = dsks._behavior(b)
            comp = context.read_json(b / job / 'materials' / (plan + '__compiled.json')) if plan else None
            summ = context.read_json(rd.load_index(b / job)[plan]['summary_path']) if plan else None
            arms[arm] = {'status': r['status'], 'commit': plan, 'failure_kind': r['failure_kind'], 'fast_calls': r['calls'], 'new_evaluations': r['new_evaluations'],
                         'e': _vec(cells, plan, 'e'), 'e_shadow': _vec(cells, plan, 'e_shadow'), 'c_a': _vec(cells, plan, 'c_a'), 'c_b': _vec(cells, plan, 'c_b'),
                         'e_mae': _vec(cells, plan, 'e_mae'), 'material_spec': comp['material_spec'] if comp else None,
                         'material_totals': {k: int(sum(e[k] for e in summ['entities'])) for k in ('windows', 'filled_points', 'observed_points', 'observed_changed_points')} if summ else None,
                         'behavior': beh, 'fast_tokens': dsks._unit_tokens(P['target']).get('fast:%s' % b.name)}
        row = {'arms': arms}
        if base_cells:
            row['Baseline_Linear'] = {k: _vec(base_cells, 'Baseline_Linear', k) for k in ('e', 'c_a', 'c_b', 'e_mae')}
            row['Fixed_Seasonal'] = {k: _vec(base_cells, 'Fixed_Seasonal', k) for k in ('e', 'e_shadow', 'c_a', 'c_b', 'e_mae')}
            anyb = next(iter(base_cells.values()))
            row['e_status'] = sorted({v['e_status'] for v in anyb.values() if v['e_status']})
            row['e_min_coverage'] = min((v['e_min_coverage'] for v in anyb.values() if v['e_min_coverage'] is not None), default=None)
            bl, fs = row['Baseline_Linear']['e'], row['Fixed_Seasonal']['e']
            e = {a: v.get('e') for a, v in arms.items()}
            row['pipeline_utility_vs_baseline_linear'] = {a: _paired(bl, v) for a, v in {**e, 'Fixed_Seasonal': fs}.items()}
            row['training_material_value_shadow'] = {a: _paired(bl, arms[a].get('e_shadow')) for a in arms if arms[a].get('e_shadow')}
            row['training_material_value_shadow']['Fixed_Seasonal'] = _paired(bl, row['Fixed_Seasonal']['e_shadow'])
            row['fast_increment'] = {'Fixed_Seasonal_minus_no_skill': _paired(fs, e.get('no_skill')), 'random_minus_no_skill': _paired(e.get('random'), e.get('no_skill'))}
            ks = e.get('known_domain')
            row['skill_increment'] = {'delta_skill_no_skill_minus_known_domain': _paired(e.get('no_skill'), ks), 'generic_minus_known_domain': _paired(e.get('generic'), ks),
                                      'random_minus_known_domain': _paired(e.get('random'), ks), 'fixed_seasonal_minus_known_domain': _paired(fs, ks),
                                      'baseline_linear_minus_known_domain': _paired(bl, ks)}
        res['target'][job] = row
    led = context.read_json(P['ledger']) if P['ledger'].exists() else {}
    res['costs'] = {'ledger': {k: v for k, v in led.items() if k != 'events'}, 'stages': context.read_json(P['stages']) if P['stages'].exists() else {},
                    'tokens_by_unit': {s: dsks._unit_tokens(P[s]) for s in ('source', 'select', 'target') if P[s].exists()},
                    'formation_tokens': dsks._unit_tokens(P['formation']) if P['formation'].exists() else {}}
    context.write_json(root / 'result.json', res)
    return res


# ============================================================================= smoke (new risks only; 0 real fits, 0 API, no label row of a planned job)
def _register_synthetic(name: str = 'RDX', n: int = 12, rows: int = 8544, seed: int = 7) -> None:
    import numpy as np
    rng = np.random.RandomState(seed)
    r = np.arange(rows)
    base_ = 50 + 20 * np.sin(2 * np.pi * r / 24)[:, None] + 5 * rng.randn(rows, n) + 10 * np.arange(n)[None, :]
    base_[rng.rand(rows, n) < 0.01] = np.nan
    base_[5000:5040, 3] = np.nan
    base_[rng.rand(rows, n) < 0.003] += 200
    rd.DATASETS[name] = {'kind': 'synthetic', 'values': base_, 'hours': np.tile((r % 24)[:, None], (1, n)), 'roster': ['s%02d' % i for i in range(n)]}


class _ScriptedClient:
    """Fast script (inspect -> build+evaluate -> compare -> commit) per unit; records every request."""
    fatal = False

    def __init__(self):
        self.requests, self.count = [], {}

    def call(self, role, unit, payload, system, max_tokens=4096, request_timeout=300.):
        self.requests.append({'role': role, 'unit': unit, 'payload': payload, 'system': system})
        k = self.count[unit] = self.count.get(unit, 0) + 1
        plan = {'default': {'steps': [{'op': 'hampel_filter', 'window': 7, 'n_sigmas': 3.0}]},
                'rules': [{'when': {'feature': 'missing_fraction', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': [{'op': 'period_median_complete'}]}],
                'rationale': 'smoke hypothesis', 'observation_fields_used': ['missing_fraction']}
        script = {1: [{'tool': 'inspect_data', 'arguments': {'entity_indices': [0], 'kind': 'gaps'}}],
                  2: [{'tool': 'build_material', 'arguments': {'plan_id': 'P1', 'policy': plan}}, {'tool': 'evaluate', 'arguments': {'plan_id': 'P1'}}],
                  3: [{'tool': 'compare', 'arguments': {'a': 'Baseline_Linear', 'b': 'P1'}}],
                  4: [{'tool': 'commit', 'arguments': {'plan_id': 'P1', 'reason': 'smoke commit'}}]}
        return {'actions': copy.deepcopy(script[min(k, 4)])}


def _fake_fit_factory(counter: dict):
    """rt.fit stand-in for synthetic data: the real readiness fit_one in-process with an untrained model (no optimizer step)."""
    from methods.ttha.batch_base import train as _train

    def fake_train_arm(Xp, Yp, child, bidx, model_seed, timeout_seconds=None, record_every=100):
        torch, _ = _train._torch()
        torch.manual_seed(model_seed)
        assert child is None and bidx.max() < Xp.shape[0] and Xp.shape[1] == 192 and Yp.shape[1] == 48
        return _train.make_model()(), _train.TrainResult(loss_curve=[(0, 0.0)], n_updates_done=0)

    def fake_fit(ctx, ref, seeds, ledger, repo, feedback=True):
        orig = _train.train_arm
        _train.train_arm = fake_train_arm
        try:
            out = []
            for seed in seeds:
                cid = _train.cell_id(ctx.job.job_id, ref.material_id, seed)
                cp = ctx.job_dir / 'cells' / (cid + '.json')
                src = cp if cp.exists() else (ctx.job_dir / 'cells' / (_train.cell_id(ctx.job.job_id, ref.alias_of, seed) + '.json') if ref.alias_of else cp)
                if src.exists():
                    rec = context.read_json(src)
                    if src != cp:
                        rec = {**rec, 'cell_id': cid, 'material_id': ref.material_id, 'material_alias_of': ref.alias_of}
                        context.write_json(cp, rec)
                    counter['cache'] = counter.get('cache', 0) + 1
                    out.append(rec)
                    continue
                counter['fits'] = counter.get('fits', 0) + 1
                out.append(rd.fit_one(ctx.job.dataset, ctx.job.job_id, str(ctx.run_dir), ref.material_id, seed, feedback=feedback))
            return out
        finally:
            _train.train_arm = orig
    return fake_fit


def smoke(out: Path) -> dict:
    import tempfile
    import numpy as np
    from SelfEvolvingHarnessTS.runtime import executor as ex
    from methods.ttha.batch_base import data as bdata, spec as bspec, train as btrain
    from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    orig_apply, orig_fit = rd.apply_program, rt.fit
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td = Path(td)
        # 1. natural NaN reaches tools/operators (independent recount from the raw text; not the without-missing cache)
        seen = {'lengths': set(), 'nan_inputs': 0}

        def spy(x, steps, fill_mean):
            seen['lengths'].add(int(np.asarray(x).size))
            seen['nan_inputs'] += int(np.isnan(x).sum())
            return orig_apply(x, steps, fill_mean)
        ctxs, c1 = {}, {}
        for d in DOMAINS:
            job = d + '_S1'
            t = rd.JOB_T['S1']
            ctx = ctxs[d] = rd.open_job(d, job, td / 'real', 'material')
            seg = ctx.slice.rows(t - 672, t)
            if d == 'RD01':
                ser = rd._tsf_series(rd.DATASETS[d]['path'], rd.RD01_ROSTER)
                raw_missing = sum(tok == '?' for name in rd.RD01_ROSTER for tok in ser[name]['tokens'][t - 672:t])
            else:
                import csv as _csv
                raw_missing = 0
                for st in rd.RD02_ROSTER:
                    with open(rd._prsa_dir(rd.DATASETS[d]) / (rd.DATASETS[d]['file'] % st), newline='', encoding='utf-8') as f:
                        rows_ = list(_csv.reader(f))
                    iv = rows_[0].index('PM2.5')
                    raw_missing += sum(r_[iv] == 'NA' for r_ in rows_[1 + t - 672:1 + t])
            ov = ctx.overview()
            gaps = ctx.inspect_data([0], 'gaps')['entities']['entity_0']
            rd.apply_program = spy
            try:
                rd.build(ctx, rd.compile_policy(rd.FIXED_SEASONAL, ov['entities'])['assignment'], 'Fixed_Seasonal')
            finally:
                rd.apply_program = orig_apply
            legal_nan = int(np.isnan(ctx.legal.X_raw).sum())
            c1[d] = {'slice_nan': int(np.isnan(seg).sum()), 'raw_text_missing': int(raw_missing), 'overview_missing_fraction_consistent':
                     all(abs(r['missing_fraction'] - np.isnan(seg[:, e]).mean()) < 1e-4 for e, r in enumerate(ov['entities'])),
                     'inspect_gap_hours_entity0': sum(g[2] for g in gaps['gaps']), 'entity0_nan': int(np.isnan(seg[:, 0]).sum()),
                     'operator_saw_nan_points': seen['nan_inputs'], 'legal_window_nan_points': legal_nan,
                     'source_is_with_missing': str(rd.DATASETS[d]['path']).endswith(('with_missing_values.zip', 'PRSA_Data_20130301-20170228'))}
            seen['nan_inputs'] = 0
        checks['1_natural_nan_reaches_tools_and_operators'] = {'ok': all(v['slice_nan'] == v['raw_text_missing'] > 0 and v['overview_missing_fraction_consistent']
                                                                         and v['inspect_gap_hours_entity0'] == v['entity0_nan'] and v['operator_saw_nan_points'] == v['legal_window_nan_points'] > 0
                                                                         and v['source_is_with_missing'] for v in c1.values()), 'detail': c1}
        # 2. operators get X=192 only; y, legal windows and scaler do not move with the plan
        ctx = ctxs['RD01']
        with np.load(ctx.job_dir / 'scaler.npz') as z:
            before = {k: z[k].copy() for k in ('mean', 'scale', 'legal_ent', 'legal_k')}
        rd.apply_program = spy
        try:
            rbl = rd.build(ctx, rd.compile_policy(rd.BASELINE_LINEAR, ctx.overview()['entities'])['assignment'], 'Baseline_Linear')
            rh = rd.build(ctx, rd.compile_policy(rd.uniform_policy([{'op': 'hampel_filter', 'window': 7, 'n_sigmas': 3.0}], 'smoke'), ctx.overview()['entities'])['assignment'], 'H')
        finally:
            rd.apply_program = orig_apply
        Xb, yb = rd.training_arrays(ctx, rbl)
        Xh, yh = rd.training_arrays(ctx, rh)
        reopened = rd.open_job('RD01', 'RD01_S1', td / 'real', 'material')
        with np.load(ctx.job_dir / 'scaler.npz') as z:
            after = {k: z[k].copy() for k in before}
        lg = ctx.legal
        y_raw_back = yb * ctx.scaler.scale[lg.ent][:, None] + ctx.scaler.mean[lg.ent][:, None]
        checks['2_operator_x_only_targets_windows_scaler_fixed'] = {
            'ok': seen['lengths'] == {192} and np.array_equal(yb, yh) and not np.array_equal(Xb, Xh) and all(np.array_equal(before[k], after[k]) for k in before)
            and np.array_equal(reopened.legal.ent, lg.ent) and np.allclose(y_raw_back, lg.y_raw, rtol=0, atol=1e-9) and np.isfinite(lg.y_raw).all(),
            'detail': {'operator_input_lengths': sorted(seen['lengths']), 'y_identical_across_plans': bool(np.array_equal(yb, yh)), 'x_differs': bool(not np.array_equal(Xb, Xh)),
                       'legal_parents': lg.n}}
        # 3. train/serve length and finiteness; failures never pass as identity; Baseline-Linear == explicit linear alias
        alias = rd.build(ctx, rd.compile_policy(rd.uniform_policy([{'op': 'impute_linear', 'strength': 1}], 'smoke'), ctx.overview()['entities'])['assignment'], 'LinearExplicit')
        allnan, rec_nan = rd.apply_program(np.full(192, np.nan), [{'op': 'hampel_filter', 'window': 5, 'n_sigmas': 3.0}], 42.0)
        faults = {}
        for label, fake in (('INF_OUTPUT', lambda steps, x, source='': ex.ExecutionResult(np.full(192, np.inf), True)),
                            ('LENGTH_CHANGED', lambda steps, x, source='': ex.ExecutionResult(np.zeros(191), True)),
                            ('OPERATOR_FAILED', lambda steps, x, source='': ex.ExecutionResult(None, False, 'boom'))):
            saved = ex.run_pipeline
            ex.run_pipeline = fake
            try:
                rd.apply_program(ctx.legal.X_raw[0], [{'op': 'outlier_iqr', 'k': 1.5}], 0.0)
                faults[label] = 'NOT_RAISED'
            except rd.ProgramExecutionError as exc:
                faults[label] = str(exc).split(':')[0]
            finally:
                ex.run_pipeline = saved
        _register_synthetic()
        sctx = rd.open_job('RDX', 'RDX_V1', td / 'syn', 'material')
        sctx_e = rd.open_job('RDX', 'RDX_V1', td / 'syn', 'evaluate')
        Xs, srec = rd.serve_inputs(sctx_e.slice, sctx_e.job.c_a, [[{'op': 'denoise_savgol', 'window': 11, 'order': 2}]] * 12, sctx_e.scaler)
        checks['3_train_serve_length_finite_and_linear_alias'] = {
            'ok': alias.alias_of == 'Baseline_Linear' and np.array_equal(Xb.shape, (lg.n, 192)) and np.isfinite(Xb).all() and np.isfinite(Xh).all()
            and rec_nan['all_missing_mean_fill'] == 1 and np.all(allnan == 42.0) and faults == {'INF_OUTPUT': 'INF_OUTPUT', 'LENGTH_CHANGED': 'LENGTH_CHANGED', 'OPERATOR_FAILED': 'OPERATOR_FAILED'}
            and Xs.shape == (12, 2, 192) and np.isfinite(Xs).all() and srec['windows'] == 24,
            'detail': {'explicit_linear_alias_of': alias.alias_of, 'fault_codes': faults, 'serve_record': srec}}
        # 4. N=12 and N=32 train one shared model over the actual pool; augmentation defaults unchanged
        c4 = {}
        for d in DOMAINS:
            cx = ctxs[d]
            ref = rd.build(cx, [[] for _ in cx.job.roster], 'BL4') if d == 'RD02' else rbl
            X, y = rd.training_arrays(cx, ref)
            bidx = btrain.batch_indices(SEEDS[0], n_pool=cx.legal.n)
            torch, _ = btrain._torch()
            torch.manual_seed(SEEDS[0])
            pred = btrain.predict(btrain.make_model()(), X[bidx[0]])
            c4[d] = {'entities': len(cx.job.roster), 'pool': cx.legal.n, 'X': list(X.shape), 'y': list(y.shape), 'batch_index_max': int(bidx.max()),
                     'one_model_output': list(pred.shape)}
        legacy = np.random.RandomState(bspec.batch_seed(SEEDS[0])).randint(0, 13856, size=(2000, 64))
        from methods.ttha.batch_base import context as bctx
        checks['4_n12_n32_shared_model_and_aug_defaults'] = {
            'ok': c4['RD01']['entities'] == 32 and c4['RD02']['entities'] == 12 and all(v['X'] == [v['pool'], 192] and v['y'] == [v['pool'], 48] and v['batch_index_max'] < v['pool']
                                                                                         and v['one_model_output'] == [64, 48] for v in c4.values())
            and np.array_equal(btrain.batch_indices(SEEDS[0]), legacy) and bspec.N_POOL == 13856 and not hasattr(bctx.JobContext, 'fit_module')
            and rd.ReadinessContext.fit_module == rd.FIT_MODULE,
            'detail': c4}
        # 5. missing-aware scoring: one common mask, observed cells only, coverage rule, equals the classic scorer when complete
        rng = np.random.RandomState(3)
        sc5 = bdata.Scaler(mean=np.full(4, 10.0), std=np.full(4, 2.0), scale=np.full(4, 2.0), floor_hits=0)
        y5 = 10 + rng.randn(4, 2, 48)
        yn = y5.copy()
        yn[0, 0, :30] = np.nan
        yn[1, :, :] = np.nan
        yn[1, 1, :20] = 10.0
        pa, pb = 10 + rng.randn(4, 2, 48), 10 + 2 * rng.randn(4, 2, 48)
        sa, sb = rd.score_block(pa, yn, sc5), rd.score_block(pb, yn, sc5)
        ok_mask = sa['coverage'] == sb['coverage'] and np.isnan(yn).sum() == 30 + 96 - 20
        manual0 = float(np.nanmean(((pa[0] - yn[0]) / 2.0) ** 2))
        full_new = rd.score_block(pa, y5, sc5)
        full_old = bdata.score_predictions(pa, y5, sc5.mean, sc5.scale, np.ones(4))
        yn2 = yn.copy()
        yn2[1, 1, :] = 10.0
        s_ok = rd.score_block(pa, yn2, sc5)
        checks['5_missing_mask_common_no_future_fill'] = {
            'ok': ok_mask and sa['status'] == 'NOT_SCORABLE' and sa['normalized_mse_macro'] is None and sa['coverage']['not_scorable_entities'] == [1]
            and abs(sa['per_entity_normalized_mse'][0] - manual0) < 1e-12 and abs(statistics.mean(r_[0] for r_ in sa['per_origin_entity_normalized_mse']) - manual0) < 1e-12
            and abs(full_new['normalized_mse_macro'] - full_old['normalized_mse_macro']) < 1e-12 and np.isnan(yn).sum() == 106
            and s_ok['status'] == 'SCORABLE' and s_ok['normalized_mse_macro'] is not None,
            'detail': {'coverage_equal_across_predictions': ok_mask, 'entity0_manual_vs_block': [manual0, sa['per_entity_normalized_mse'][0]],
                       'complete_equals_classic': [full_new['normalized_mse_macro'], full_old['normalized_mse_macro']], 'coverage_50pct_entity_status': s_ok['status']}}
        # 6. card reaches the request; Source/Select never open E; all commits before any C_B; E only after the barrier
        counter = {}
        rt.fit = _fake_fit_factory(counter)
        try:
            proot = td / 'pkg'
            proot.mkdir()
            ledger = proot / 'budget.json'
            caps = {'max_fit_attempts': 999, 'max_llm_requests': 999, 'max_llm_tokens': 10 ** 9, 'max_wall_s': 3600, 'max_retries': 0}
            rt.RuntimeLedger(ledger, **caps)
            refs = ['fixture:smoke']
            card = dsk.make_skill(skill_id='RDX-W1-r1', domain_id='RDX', revision=1, workflow='Inspect gaps first; compare a seasonal completion with the baseline; stop when C_A differences are within seed noise.',
                                  principles=None, applicability_summary='smoke card', compatibility_note='smoke', observable_applicability={'const': True},
                                  evidence_refs=refs, legal_evidence_refs=refs, source_stage='fixture', status='SMOKE_FIXTURE', allowed_features=ALLOWED_FEATURES,
                                  text_check=skill_text_check)
            client = _ScriptedClient()
            inproc = lambda p, j, name, led: {'c_b': rd.open_c_b, 'freeze_e': rd.freeze_e, 'score_e': rd.score_e}[name](p, ds_of(j), j)
            scfg = {'stage': 'select', 'output': str(proot / 'select'), 'jobs': ['RDX_V1'], 'order': {'RDX_V1': ['no_skill', 'cand_W1']},
                    'knowledge': br.json_copy({'RDX_V1': {'no_skill': asdict(br.Knowledge()), 'cand_W1': asdict(dsk.skill_knowledge(card))}}),
                    'random_policy_seeds': {}, 'labels_e': False, 'caps': caps, 'http_cap': 999, 'ledger_path': str(ledger), 'fit_retry': False}
            context.write_json(proot / 'select.json', scfg)
            stage_worker(proot / 'select.json', client=client, runner=inproc)
            sroot = proot / 'select'
            first = {u: next(r_ for r_ in client.requests if r_['unit'] == u) for u in ('RDX_V1_no_skill', 'RDX_V1_cand_W1')}
            loaded = {u: [x['body'] for x in r_['payload']['guidance']['loaded']] for u, r_ in first.items()}
            commits = [sroot / b / 'RDX_V1' / 'commit.json' for b in ('RDX_V1_no_skill', 'RDX_V1_cand_W1')]
            cbs = [sroot / b / 'RDX_V1' / 'c_b_scores.json' for b in ('RDX_V1_no_skill', 'RDX_V1_cand_W1')]
            e_files = [p_ for p_ in sroot.rglob('*') if p_.name in ('e_frozen.json', 'e_scores.json', 'all_e_predictions_frozen.json') or p_.name == 'predictions_e']
            sel = dsk.select({dsk.NO_SKILL: None, card.skill_id: card}, ['RDX_V1'],
                             lambda oid, k, j: {'status': 'COMPLETE', 'committed_plan_id': 'P1', 'synthetic': True},
                             block='c_b', score=c_b_scorer(lambda oid, j: sroot / ('%s_%s' % (j, 'no_skill' if oid == dsk.NO_SKILL else 'cand_W1'))))
            # a Target-like stage with E and the Random control
            tcfg = {**scfg, 'stage': 'target', 'output': str(proot / 'target'), 'jobs': ['RDX_T1'], 'order': {'RDX_T1': ['known_domain', 'random']},
                    'knowledge': br.json_copy({'RDX_T1': {'known_domain': asdict(dsk.skill_knowledge(card)), 'random': None}}),
                    'random_policy_seeds': {'RDX_T1': [2026091701, 2026091702]}, 'labels_e': True}
            context.write_json(proot / 'target.json', tcfg)
            stage_worker(proot / 'target.json', client=client, runner=inproc)
            troot = proot / 'target'
            ef = [troot / b / 'RDX_T1' / 'e_frozen.json' for b in ('RDX_T1_known_domain', 'RDX_T1_random')]
            es = [troot / b / 'RDX_T1' / 'e_scores.json' for b in ('RDX_T1_known_domain', 'RDX_T1_random')]
            barrier = troot / 'all_e_predictions_frozen.json'
            rres = context.read_json(troot / 'RDX_T1_random' / 'branch_result.json')
            esc = context.read_json(es[1])['cells']
            resume_blocked = rtp.labels_boundary_violations(sroot)
            checks['6_card_in_request_no_e_in_select_commits_before_labels'] = {
                'ok': loaded['RDX_V1_no_skill'] == [] and loaded['RDX_V1_cand_W1'] == [card.rendered_body] and all(p_.exists() for p_ in commits + cbs)
                and max(p_.stat().st_mtime for p_ in commits) <= min(p_.stat().st_mtime for p_ in cbs) and not e_files and (sroot / 'labels_c_b_only.json').exists()
                and all(r_['system'] == FAST_SYSTEM for r_ in client.requests) and sel['status'] in ('SELECTED', 'NO_EFFECTIVE_CANDIDATE')
                and all(p_.exists() for p_ in ef + es) and max(p_.stat().st_mtime for p_ in ef) <= barrier.stat().st_mtime <= min(p_.stat().st_mtime for p_ in es)
                and rres['status'] == 'COMPLETE' and all('e_shadow_baseline_inputs' in v for v in esc.values()) and bool(resume_blocked)
                and counter.get('cache', 0) > 0,
                'detail': {'loaded_first_request': {u: len(v) for u, v in loaded.items()}, 'select_status': sel['status'], 'select_scores': sel['eligible_scores'],
                           'e_files_in_select_stage': [str(p_) for p_ in e_files], 'random_commit': rres['committed_plan_id'], 'fake_fits': counter,
                           'resume_refused_after_labels': resume_blocked[:3]}}
        finally:
            rt.fit = orig_fit
            rd.DATASETS.pop('RDX', None)
        # 7. an augmentation material path equals its saved material (0 fits)
        led7 = rt.RuntimeLedger(td / 'aug_budget.json', max_fit_attempts=0)
        a = rt.Adapter(td / 'aug', 'a55', led7, REPO)
        a.build_material({'plan_id': 'R', 'policy': rt.policy.fixed_mixup_policy()}, remaining_seconds=60)
        with np.load(REPO / '_scratch/ts_aug_donor_pattern_probe/materials/a55__children.npz') as old, np.load(a.refs['R'].path) as new:
            same = bool(np.array_equal(old['RX'], new['Xc']) and np.array_equal(old['RY'], new['Yc']))
        checks['7_augmentation_material_unchanged'] = {'ok': same and led7.s['fit_attempts'] == 0, 'detail': {'a55_fixed_mixup_equals_saved': same}}
    res = {'status': 'PASS' if all(c['ok'] for c in checks.values()) else 'FAIL', 'checks': checks, 'real_fits': 0, 'api_calls': 0,
           'label_rows_of_planned_jobs': 0, 'note': 'real T only for RD01_S1/RD02_S1 material stages; label stages exercised on a registered synthetic dataset'}
    res['existing_controls'] = []
    for cmd in (['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/functional/test_batch_research.py'],
                ['-c', 'import json,sys; from evaluation.main_protocol_p4 import batch_research_domain_skill as m; r=m.smoke_pilot(); '
                       'json.dump(r, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str); sys.exit(0 if r["status"]=="PASS" else 1)',
                 str(out / 'domain_skill_smoke_pilot.json')]):
        t0 = time.time()
        pr = subprocess.run([sys.executable, '-B'] + cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
        res['existing_controls'].append({'command': ' '.join(cmd[:4]), 'returncode': pr.returncode, 'seconds': round(time.time() - t0, 1), 'tail': (pr.stdout + pr.stderr)[-400:]})
        if pr.returncode:
            res['status'] = 'FAIL'
    context.write_json(out / 'smoke.json', res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--stage-worker', type=Path)
    ap.add_argument('--resume-stage', choices=['source', 'select', 'target'])
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--restart', default='', help='comma-separated technically interrupted branch names to run again (resume only)')
    ap.add_argument('--continue-package', action='store_true', help='after the resume, continue the package driver')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--root', type=Path, default=ROOT)
    a = ap.parse_args()
    root = a.root.resolve()
    if a.stage_worker:
        stage_worker(a.stage_worker.resolve())
    elif a.resume_stage:
        resume_stage(root, a.resume_stage, accept_unknown_usage=a.accept_unknown_usage, restart=[x for x in a.restart.split(',') if x])
        if a.continue_package:
            package_run(root)
    elif a.run:
        package_run(root)
    elif a.result:
        package_result(root)
    elif a.preflight:
        print(json.dumps(preflight(root)['eligibility_recomputed'], ensure_ascii=False)[:2000])
    elif a.smoke:
        res = smoke(root / 'smoke')
        print('READINESS_SMOKE', res['status'], len(res['checks']), flush=True)
        if res['status'] != 'PASS':
            raise SystemExit(1)
    else:
        ap.error('choose --run, --stage-worker, --resume-stage, --result, --preflight or --smoke')


if __name__ == '__main__':
    from evaluation.main_protocol_p4 import batch_research_data_readiness as _canonical
    _canonical.main()
