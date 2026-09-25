"""DEV-TEMPO-AUG-RECIPE-EDIT-PILOT (docs/DEV_TEMPO_AUG_RECIPE_EDIT_PILOT_TASK_2026-09-19.md): can a bounded local edit of the public preset
P_NoMixRecipe (disable 0-2 of its seven components, everything else and the parent random stream unchanged) build better training material,
and can the Fast path find and deliver such an edit on four exposed RD02 batches (L1/L3/L5/L7)?

One new construction capability (tempo_aug.RECIPE_EDIT, opt-in), one package, one ledger:
  preflight (0 cost, frozen_config) -> smoke (0 fits, 0 LLM; cached L1 material) -> wiring (RD02_T1, <= 3 fits, scripted client, C_B only)
  -> pilot stage: per batch F_edit (real Fast, 4 slots / 16 calls / 24 tools) + Random_edit_B4 (same program family, frozen draws) + Menu_CA;
     after every arm of every batch is committed: the seven single-component ablations (0 LLM) in a diagnostic view
  -> whole-stage C_B -> E freeze barrier -> E -> readout (three layers: material built / delivered / selection loss) and stop.
The numerical workers, the seven-tool Fast, the physical cache, the label workers, the resume boundary and the ledger are the parent
module's (batch_research_tempo_aug_workflow_skill, imported as W). New here: the 29-program local table, the edit DSL / adapter / action table /
contracts / prompt, the edit material builder (parent stream of program 52), the Random_edit_B4 supply, the ablation view and the readout.

  --preflight | --smoke | --wiring | --run | --resume-stage pilot [--accept-unknown-usage] | --result
  workers (torch allowed, subprocess only): --stage-worker CFG | --worker-material ROOT JOB PHYS JOB_INDEX | --worker-smoke OUT
The controller never imports torch.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from methods.ttha import batch_research as br
from methods.ttha.batch_base import budget, context, spec, tempo_aug as ta
from methods.ttha.batch_base import readiness as rd
from evaluation.main_protocol_p4 import batch_research_domain_skill as dsks
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import run_batch_research_v1 as base
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_skill as W

REPO = W.REPO
ROOT = REPO / '_scratch' / 'dev_tempo_aug_recipe_edit_pilot'
MODULE = 'evaluation.main_protocol_p4.batch_research_tempo_aug_recipe_edit_pilot'
TASK = 'docs/DEV_TEMPO_AUG_RECIPE_EDIT_PILOT_TASK_2026-09-19.md'
PACKAGE = 'DEV-TEMPO-AUG-RECIPE-EDIT-PILOT'
EXPOSURE = 'EXPOSED_DEVELOPMENT_REPLAY'
DATASET = W.DATASET
SEEDS = W.SEEDS
PUBLIC, PUBLIC_STEPS = W.PUBLIC, W.PUBLIC_STEPS
JOBS = ('RD02_L1', 'RD02_L3', 'RD02_L5', 'RD02_L7')
JOB_INDEX = {'RD02_L1': 7, 'RD02_L3': 9, 'RD02_L5': 11, 'RD02_L7': 13}
JOB_T = {'RD02_L1': 15360, 'RD02_L3': 18960, 'RD02_L5': 21360, 'RD02_L7': 24480}
ORDER = {'RD02_L1': ('f_edit', 'random_edit', 'menu_ca'), 'RD02_L3': ('random_edit', 'f_edit', 'menu_ca'),
         'RD02_L5': ('f_edit', 'random_edit', 'menu_ca'), 'RD02_L7': ('random_edit', 'f_edit', 'menu_ca')}
ARMS = ('f_edit', 'random_edit', 'menu_ca')
ARM_LABEL = {'f_edit': 'F_edit', 'random_edit': 'Random_edit_B4', 'menu_ca': 'Menu_CA'}
DL = REPO / '_scratch' / 'dev_tempo_aug_decision_learning'
CACHE_FROM = {'RD02_L1': W.ROOT / 'target' / 'RD02_L1_common' / 'RD02_L1', 'RD02_L3': W.ROOT / 'target' / 'RD02_L3_common' / 'RD02_L3',
              'RD02_L5': DL / 'test' / 'RD02_L5_common' / 'RD02_L5', 'RD02_L7': DL / 'test' / 'RD02_L7_common' / 'RD02_L7'}
WIRING_JOB, WIRING_SEEDS, WIRING_PUBLIC = W.WIRING_JOB, W.WIRING_SEEDS, W.WIRING_PUBLIC
WIRING_CACHE = W.ROOT / 'wiring' / 'RD02_T1_common' / 'RD02_T1'
WIRING_JOB_INDEX = W.JOB_INDEX[WIRING_JOB]
PARENT_PROGRAM_INDEX = W.program_index([{'op': ta.RECIPE}])            # 52: the preset's per-(job, entity) stream is the parent stream of every edit
EDIT_PROFILE = 'recipe_edit_v1'
LIMITS, FAST_TOKEN_CAP, MAX_OUTPUT_TOKENS, MAX_TOOL_CORRECTIONS, EVIDENCE_ROUNDTRIP = W.LIMITS, W.FAST_TOKEN_CAP, W.MAX_OUTPUT_TOKENS, W.MAX_TOOL_CORRECTIONS, W.EVIDENCE_ROUNDTRIP
TOTAL = {'max_fit_attempts': 233, 'max_llm_requests': 64, 'max_llm_tokens': 1_600_000, 'max_wall_s': 4 * 3600, 'max_retries': 2}
HTTP_CAP = 66
ALLOC = {'wiring': {'fits': 3, 'requests': 0, 'tokens': 0}, 'pilot': {'fits': 228, 'requests': 64, 'tokens': 1_600_000}}
NUMERIC_WALL_S = 4 * 3600.
RANDOM_SEED_ROOT, RANDOM_QUANTILES, RANDOM_MAX_TRIES = W.RANDOM_SEED_ROOT, W.RANDOM_QUANTILES, W.RANDOM_MAX_TRIES
MODEL = W.MODEL
T975_DF2 = W.T975_DF2
TIE_ABS = 1e-12
SEALED_FROM_ROW = 24864


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def paths(root: Path) -> dict:
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'configs': root / 'stage_configs', 'logs': root / 'logs',
            'wiring': root / 'wiring', 'pilot': root / 'pilot', 'smoke': root / 'smoke'}


# ============================================================================= the 29 local programs (frozen table; enumeration / cache only)
def edit_program_table() -> list:
    """0 = the preset; 1..7 = one component disabled, in PRIMITIVES order; 8..28 = two disabled, in combinations order."""
    out = [[{'op': ta.RECIPE}]]
    for n in ta.PRIMITIVES:
        out.append([ta.edit_step([n])])
    for a, b in itertools.combinations(ta.PRIMITIVES, 2):
        out.append([ta.edit_step([a, b])])
    if len(out) != 29:
        raise RuntimeError('the local program table must hold 29 programs')
    return out


EDIT_PROGRAMS = edit_program_table()
EDIT_INDEX = {W._PKEY(p): i for i, p in enumerate(EDIT_PROGRAMS)}
ABLATION = tuple(ta.PRIMITIVES)                                         # the seven single-component ablations, fixed before anything runs


def edit_program_index(steps: list) -> int:
    return EDIT_INDEX[W._PKEY(validate_steps_edit(steps))]


# ============================================================================= the edit DSL (this package's program space; the repository DSL is untouched)
def validate_steps_edit(steps) -> list:
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict) or 'op' not in steps[0]:
        raise rd.PolicyError("in this package every program is exactly one step: {'op': '%s', 'disabled_ops': [...]} (0-2 component names) or "
                             "{'op': '%s'} (the preset); empty steps and explicit primitive compositions are not offered here" % (ta.RECIPE_EDIT, ta.RECIPE))
    st = steps[0]
    if st['op'] == ta.RECIPE:
        if set(st) - {'op'}:
            raise rd.PolicyError('%s takes no parameters' % ta.RECIPE)
        return [{'op': ta.RECIPE}]
    if st['op'] != ta.RECIPE_EDIT:
        raise rd.PolicyError("unknown op %r; legal: %s with disabled_ops, or %s" % (st['op'], ta.RECIPE_EDIT, ta.RECIPE))
    if set(st) - {'op', 'disabled_ops'}:
        raise rd.PolicyError('%s takes only disabled_ops; strengths, category weights, draw counts, probabilities and the execution order are not editable' % ta.RECIPE_EDIT)
    return [ta.edit_step(st.get('disabled_ops', []))]


def validate_plan_edit(pol) -> dict:
    if not isinstance(pol, dict) or 'default' not in pol:
        raise rd.PolicyError("policy must be an object with a 'default'")
    if set(pol) - {'default', 'rules', 'rationale', 'observation_fields_used'}:
        raise rd.PolicyError('unknown policy keys %s' % sorted(set(pol) - {'default', 'rules', 'rationale', 'observation_fields_used'}))
    if not isinstance(pol['default'], dict) or set(pol['default']) != {'steps'}:
        raise rd.PolicyError("default must be {'steps': [...]}")
    clean = {'default': {'steps': validate_steps_edit(pol['default']['steps'])}, 'rules': []}
    rules = pol.get('rules', []) or []
    if not isinstance(rules, list) or len(rules) > ta.MAX_RULES:
        raise rd.PolicyError('rules must be a list of at most %d rules' % ta.MAX_RULES)
    for r in rules:
        if not isinstance(r, dict) or set(r) != {'when', 'steps'}:
            raise rd.PolicyError("each rule must be {'when': predicate, 'steps': [...]}")
        rd.validate_predicate(r['when'])
        clean['rules'].append({'when': r['when'], 'steps': validate_steps_edit(r['steps'])})
    rationale = str(pol.get('rationale', ''))[:500]
    rd.validate_text(rationale, 'rationale')
    clean['rationale'] = rationale
    clean['observation_fields_used'] = [f for f in (pol.get('observation_fields_used') or []) if f in rd.FIELDS]
    return clean


def compile_plan_edit(pol: dict, table: list) -> dict:
    from methods.ttha.batch_base import policy as _policy
    clean = validate_plan_edit(pol)
    resolved, assignment, rule_index, n_unknown = {}, [], [], 0
    for row in table:
        chosen, ri = clean['default']['steps'], -1
        for i, r in enumerate(clean['rules']):
            v = _policy._eval(r['when'], row, table, resolved)       # the existing three-valued evaluator; UNKNOWN never fires
            if v is None:
                n_unknown += 1
            if v is True:
                chosen, ri = r['steps'], i
                break
        assignment.append([dict(s) for s in chosen])
        rule_index.append(ri)
    return {'policy': clean, 'assignment': assignment, 'rule_index': rule_index, 'resolved_thresholds': resolved, 'n_unknown': n_unknown}


def uniform_edit_policy(disabled: list, rationale: str) -> dict:
    return {'default': {'steps': [ta.edit_step(disabled)]}, 'rules': [], 'rationale': rationale, 'observation_fields_used': []}


def edit_action_table() -> dict:
    t = ta.action_table()
    return {'recipe_components': t['primitives'],
            'preset': {ta.RECIPE: ta.RECIPE_TEXT},
            'recipe_edit': {'op': ta.RECIPE_EDIT, 'arguments': {'disabled_ops': '0-2 names from recipe_components (order irrelevant, duplicates ignored)'},
                            'semantics': ta.RECIPE_EDIT_TEXT,
                            'randomness': 'Every edit shares the preset\'s per-(batch, entity) random stream: the categories, primitives and conv draw of each '
                                          'window are those of the preset; the retained primitives use the draws they would have used; disabling a component '
                                          'never re-draws anything. The same switch on the same entity is the same material in every plan.',
                            'joint_transform': 'Each window\'s 192 inputs AND its 48 child training targets are transformed together (the child\'s training '
                                               'targets change); scoring targets and serving inputs never change.',
                            'not_editable': 'component strengths, category weights (0.6/0.5/0.3/0.6), the number of categories drawn (2), the within-category '
                                            'choice probabilities, the random_conv probability (0.3) and the execution order.',
                            'alias': '%s with an empty disabled_ops is %s itself (the public reference; no new material)' % (ta.RECIPE_EDIT, ta.RECIPE)},
            'program_space_of_this_package': 'Every entity program is either the preset or ONE %s step; the default and up to %d observation-predicate rules '
                                             'may give different switches to different entities. Explicit primitive compositions (the repository\'s general '
                                             'DSL) are not offered in this package.' % (ta.RECIPE_EDIT, ta.MAX_RULES),
            'window': t['window'], 'not_available': t['not_available']}


CONTRACTS_EDIT = copy.deepcopy(W.CONTRACTS)
CONTRACTS_EDIT['build_material'] = {
    'arguments': {'plan_id': 'new short alphanumeric/underscore id',
                  'policy': {'default': {'steps': [{'op': ta.RECIPE_EDIT, 'disabled_ops': ['<component name>']}]},
                             'rules': [{'when': {'feature': '<overview field>', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': [{'op': ta.RECIPE_EDIT, 'disabled_ops': []}]}],
                             'rationale': '<component(s) to disable, the observation behind it, the expected material change; no data source or job name>',
                             'observation_fields_used': []}},
    'meaning': 'Construct one complete whole-batch plan as a local edit of the preset %s and freeze its child view; does not train. Each program is one %s step '
               'with 0-2 disabled components (0 = the preset = public reference, no new material). Up to %d rules; first matching rule wins, a null field never '
               'fires a rule; thresholds are numbers or a current-batch quantile. The preset\'s per-window draws are kept and only the disabled components are '
               'skipped (see overview.actions.recipe_edit). No strength, weight, probability or order argument exists. Plans with identical complete assignments '
               'are the same material and are rejected as duplicates of the existing id (no new material, fit or slot).' % (ta.RECIPE, ta.RECIPE_EDIT, ta.MAX_RULES)}
CONTRACTS_EDIT['inspect_material']['meaning'] = ('What the plan changed: retained steps per window, which drawn components were skipped and how often, identity '
                                                 '(no-op) counts, change fraction and RMS for inputs and child targets separately. Not predictive utility.')
CONTRACTS_EDIT['evaluate']['meaning'] = ('Train the complete plan (parent 0.5 + child 0.5, shared MLP, 2000 updates) under the 3 frozen paired seeds and return '
                                        'C_A only (missing-aware loss on observed future cells, Baseline-Linear serving inputs). At most 4 new complete plans per '
                                        'job; the four public references are already evaluated.')

PERMISSION_SENTENCE = ('允许按合法 T 观察构造统一或条件化方案；单实体预测变化不能视作该实体材料的独立因果贡献，条件化完整方案仍可由共享训练比较。'
                       ' (Uniform or entity-conditional complete plans may be constructed from legal T observations; a change in one entity\'s prediction is '
                       'not the independent causal contribution of that entity\'s material, yet conditional complete plans can still be compared through the '
                       'shared training.)')
PACKAGE_PARAGRAPH = ('''本包的研究对象：以公共预设 P_NoMixRecipe 为真实起点的有限局部编辑。build_material 中每个程序只能是 {"op":"P_NoMixRecipe_Edit","disabled_ops":[...]}（关闭 0–2 个组件；0 个 = 预设本身，即公共参照 P_NoMixRecipe，不产生新材料），default 与最多 8 条 T 谓词规则可对不同实体给出不同开关；显式原语组合在本包不提供。语义：每个窗口先按预设抽取类别、原语与卷积，再跳过被关闭的组件——不补抽、不重新归一化权重，窗口可能只剩一步或保持不变；保留组件使用它们原本的随机抽取，但依赖输入的量按新输入重算。窗口内 192 点输入与 48 点子训练目标联合变换（子训练目标也改变），评分真值与预测输入不变。目标是改进整批训练材料在后续预测上的效用；P_NoMixRecipe 是真实起点，不保证总是最优，也没有任何组件被预先认定有害或有益。请用当前合法观察提出"为什么关闭某个组件可能有帮助"的可检验假设：单组件修改便于解释，但不强制某个组件、不强制统一或条件化、不强制用满 4 个新方案、不强制偏离预设。可以依据第一份实验的结果修改下一份。C_A 是当前唯一可用反馈：有限时间块上的优劣不能当作未来效果的保证；没有 C_B/E。每个新方案的 rationale 简要注明拟关闭的组件、观察依据和预期的材料变化；材料变化不是效果证明。可以提交任一自身已评估方案或公共参照；Runner 不以 C_A argmin 覆盖你的提交。
(This package studies bounded local edits of the public preset P_NoMixRecipe. In build_material every program is one {"op":"P_NoMixRecipe_Edit","disabled_ops":[...]} step (disable 0-2 components; 0 = the preset = the public reference, no new material); the default and up to 8 observation-predicate rules may give different switches to different entities; explicit primitive compositions are not offered here. Semantics: each window first draws its categories, primitives and conv exactly as the preset, then the disabled components are skipped - nothing is redrawn, weights are not renormalized, a window may keep one step or stay unchanged; retained components use their original draws while input-dependent quantities are recomputed from the new input. The 192 inputs and the 48 child training targets of a window are transformed jointly (the child's training targets change too); scoring truth and serving inputs never change. The goal is better utility of the whole batch's training material on later predictions; the preset is the real starting point, not guaranteed optimal, and no component is presumed harmful or helpful. Form testable hypotheses from the current legal observations about why disabling a component might help: single-component edits are easy to interpret, but no component, no uniform/conditional form, no use of all 4 slots and no departure from the preset is required. You may revise the next plan after the first result. C_A is the only feedback: ranking on one limited time block is no guarantee of later effect; there is no C_B/E. State in each rationale the component(s) to disable, the observation behind it and the expected material change; a material change is not evidence of effect. You may commit any plan you evaluated or any public reference; the Runner never overrides your commit with a C_A argmin.)''')
FAST_SYSTEM_EDIT = W.FAST_SYSTEM + '\n' + PERMISSION_SENTENCE + '\n' + PACKAGE_PARAGRAPH


# ============================================================================= edit material builder (torch worker) and its subprocess entry
def build_material_edit(ctx, assignment: list, phys_id: str, *, job_index: int) -> dict:
    """One physical material of this package's program space: preset entities run the unchanged v2 preset path (program 52 stream);
    edit entities run the shadow-draw + per-step replay on the SAME program-52 stream of that entity (the switch never enters the seed).
    Explicit compositions are kept only for completeness of the cache (never requested by this package's DSL)."""
    reg = W.load_registry(ctx.job_dir)
    if phys_id in reg:
        raise ValueError('physical material id exists')
    key = W.material_key(assignment)
    if any(r['key'] == key for r in reg.values()):
        raise ValueError('a physical material with this assignment already exists')
    if ta.is_identity(assignment):
        raise ValueError('the all-empty plan is None; no physical material')
    t0 = time.time()
    Z, ent, k = ta.joint_parents(ctx)
    starts = ta.window_starts(ctx)
    C = np.empty(Z.shape, dtype=np.float32)
    per_step = {n: {'windows': 0, 'identity': 0} for n in ta.PRIMITIVES}
    shadow = {n: {'drawn': 0, 'skipped': 0} for n in ta.PRIMITIVES}
    retained_sets, shadow_sets, window_identity, no_step, seeds, ent_dis = {}, {}, 0, 0, {}, {}
    for e in range(len(ctx.job.roster)):
        rows = np.flatnonzero(ent == e)
        prog = assignment[e]
        if not prog:
            C[rows] = Z[rows].astype(np.float32)
            continue
        edit = ta.is_edit_program(prog)
        if edit or prog == [{'op': ta.RECIPE}]:
            pi, disabled = PARENT_PROGRAM_INDEX, (ta.canonical_disabled(prog[0]['disabled_ops']) if edit else [])
        else:
            pi, disabled = W.program_index(prog), None
        seed = W.entity_program_seed(job_index, pi, e)
        seeds['entity_%d' % e] = {'program_index': pi, 'seed': seed, 'disabled_ops': disabled}
        ent_dis['entity_%d' % e] = disabled
        streams = ta.Streams(seed)
        for p in rows:
            if edit:
                out, executed, ident, drawn = ta.run_program_edit(Z[p], prog, streams, starts[p])
            else:
                out, executed, ident = ta.run_program(Z[p], prog, streams, starts[p])
                drawn = executed
            C[p] = out
            if edit or prog == [{'op': ta.RECIPE}]:
                retained_sets['+'.join(executed) or '(none)'] = retained_sets.get('+'.join(executed) or '(none)', 0) + 1
                shadow_sets['+'.join(drawn)] = shadow_sets.get('+'.join(drawn), 0) + 1
                for n in drawn:
                    shadow[n]['drawn'] += 1
                    shadow[n]['skipped'] += int(n not in executed)
                no_step += int(not executed)
            for n, i in zip(executed, ident):
                per_step[n]['windows'] += 1
                per_step[n]['identity'] += int(i)
            window_identity += int(np.array_equal(out, Z[p].astype(np.float32)))
    secs = time.time() - t0
    d = C.astype(np.float64) - Z
    mdir = W.aug_dir(ctx.job_dir)
    npz = mdir / (phys_id + '.npz')
    np.savez(npz, Xc=C[:, :spec.L], yc=C[:, spec.L:])
    ents = []
    for e in range(len(ctx.job.roster)):
        rows = np.flatnonzero(ent == e)
        de = d[rows]
        ents.append({'steps': assignment[e], 'windows': int(rows.size),
                     'rms_change_X': float(np.sqrt((de[:, :spec.L] ** 2).mean())), 'rms_change_y': float(np.sqrt((de[:, spec.L:] ** 2).mean())),
                     'fraction_points_changed_X': float((np.abs(de[:, :spec.L]) > 0).mean()), 'fraction_points_changed_y': float((np.abs(de[:, spec.L:]) > 0).mean())})
    any_edit = any(ta.is_edit_program(p) for p in assignment)
    summary = {'material_id': phys_id, 'job_id': ctx.job.job_id, 'profile': W.PROFILE_VERSION, 'edit_profile': EDIT_PROFILE if any_edit else None, 'job_index': job_index,
               'entity_program_seeds': seeds, 'seed_rule': 'SeedSequence([%d, job_index, program_index, entity_index]).generate_state(1, uint32)[0]; every edit uses program_index %d (the preset\'s stream)' % (W.SEED_ROOT, PARENT_PROGRAM_INDEX),
               'legal_parents': int(Z.shape[0]), 'build_seconds': secs,
               'rms_change_X': float(np.sqrt((d[:, :spec.L] ** 2).mean())), 'rms_change_y': float(np.sqrt((d[:, spec.L:] ** 2).mean())),
               'fraction_points_changed_X': float((np.abs(d[:, :spec.L]) > 0).mean()), 'fraction_points_changed_y': float((np.abs(d[:, spec.L:]) > 0).mean()),
               'windows_bitwise_unchanged': window_identity, 'per_step': per_step, 'recipe_step_sets': retained_sets,
               'edit': {'parent_program_index': PARENT_PROGRAM_INDEX, 'entity_disabled_ops': ent_dis, 'shadow_per_step': shadow, 'shadow_step_sets': shadow_sets,
                        'retained_step_sets': retained_sets, 'windows_no_retained_step': no_step,
                        'semantics': 'shadow = what the preset drew on that window; skipped = drawn but disabled for that entity; retained steps replayed from the recorded RNG states'},
               'units': 'normalized by the frozen per-entity T scaler (child - parent)', 'entities': ents}
    sp = mdir / (phys_id + '__summary.json')
    context.write_json(sp, summary)
    rec = {'material_id': phys_id, 'job_id': ctx.job.job_id, 'key': key, 'assignment': assignment, 'profile': W.PROFILE_VERSION, 'alias_of': None,
           'path': str(npz), 'summary_path': str(sp), 'seed': None, 'material_index': int(phys_id[2:]) if phys_id.startswith('TA') else None,
           'kind': 'tempo_v2_recipe_edit' if any_edit else 'tempo_v2', 'edit_profile': EDIT_PROFILE if any_edit else None, 'built_local': now()}
    reg = W.load_registry(ctx.job_dir)
    reg[phys_id] = rec
    W.save_registry(ctx.job_dir, reg)
    return rec


def worker_material(root: str, job: str, phys: str, job_index: int) -> None:
    rd._init_openmp_before_torch()
    ctx = rd.open_job(DATASET, job, Path(root), stage='material')
    req = context.read_json(W.aug_dir(ctx.job_dir) / (phys + '__request.json'))
    try:
        rec = build_material_edit(ctx, req['assignment'], phys, job_index=int(job_index))
    except rd.ProgramExecutionError as exc:
        context.write_json(W.aug_dir(ctx.job_dir) / (phys + '__error.json'), {'kind': 'PROGRAM_EXECUTION_REJECTED', 'message': str(exc)[:300]})
        raise
    print('MATERIAL_OK', job, phys, 'build_s %.1f' % context.read_json(rec['summary_path'])['build_seconds'], flush=True)


def run_sub(args: list, log: Path, timeout: float) -> tuple:
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with log.open('w', encoding='utf-8') as fh:
            p = subprocess.run([sys.executable, '-B', '-m', MODULE] + [str(a) for a in args], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, env=W.worker_env())
        return p.returncode, time.time() - t0
    except subprocess.TimeoutExpired:
        return -999, time.time() - t0


def ensure_material_edit(root: Path, job: str, assignment: list, *, job_index: int) -> dict:
    """The physical record of an assignment in the job cache: existing by key (any package's physical cache copy) or built now by this
    module's worker (serial). Same id rule as the parent (next free TA index)."""
    reg = W.load_registry(Path(root) / job)
    key = W.material_key(assignment)
    for r in reg.values():
        if r['key'] == key:
            return r
    used = [int(r['material_index']) for r in reg.values() if r.get('material_index')]
    phys = 'TA%03d' % (max(used + [0]) + 1)
    context.write_json(W.aug_dir(Path(root) / job) / (phys + '__request.json'), {'assignment': assignment, 'requested_local': now(), 'package': PACKAGE})
    rc, secs = run_sub(['--worker-material', root, job, phys, job_index], Path(root) / job / 'fit_logs' / ('material_%s.log' % phys), 900)
    err = W.aug_dir(Path(root) / job) / (phys + '__error.json')
    if rc != 0:
        if err.exists():
            raise br.ToolInputError('plan rejected at execution (no fit, no material): %s' % context.read_json(err)['message'], execution_status='PROGRAM_EXECUTION_REJECTED')
        raise RuntimeError('material worker failed for %s %s' % (job, phys))
    return W.load_registry(Path(root) / job)[phys]


# ============================================================================= adapter (the parent's seven tools; build restricted to the edit DSL)
class EditAdapter(W.WorkflowAdapter):
    def overview(self):
        ov = super().overview()
        ov['actions'] = edit_action_table()
        ov['actions']['public_references'] = {'None': 'no augmentation (parent view only)', 'FixedMixup': W.FIXED_MIXUP['definition'] + '; not a TempoPFN primitive',
                                              'P_AmpResample': 'uniform tp_amplitude -> tp_resample (explicit composition; historical public reference)', 'P_NoMixRecipe': 'uniform ' + ta.RECIPE + ' (the preset; the starting point of every edit)'}
        return ov

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
            compiled = compile_plan_edit(arguments['policy'], self.ctx.overview()['entities'])
        except rd.PolicyError as exc:
            self._reject(str(exc), legal_program_step={'op': ta.RECIPE_EDIT, 'disabled_ops': ['<0-2 of: %s>' % ', '.join(ta.PRIMITIVES)]}, preset_alias={'op': ta.RECIPE},
                         valid_observation_fields=list(rd.FIELDS))
        key = W.material_key(compiled['assignment'])
        same = next((m for m, r in reg.items() if r.get('key') == key), None)
        if same is not None:
            self._reject('identical complete assignment to existing plan %s (same material): reuse that id; no new material, fit or slot' % same, alias_of=same,
                         execution_status='DUPLICATE_ASSIGNMENT')
        phys = ensure_material_edit(self.common, self.job, compiled['assignment'], job_index=self.job_index)
        rec = {**phys, 'material_id': mid, 'phys_id': phys['material_id'], 'compiled': compiled}
        reg = self._reg()
        reg[mid] = rec
        W.save_registry(self.ctx.job_dir, reg)
        programs, idx = [], []
        for steps in compiled['assignment']:
            if steps not in programs:
                programs.append(steps)
            idx.append(programs.index(steps))
        ms = {'profile': W.PROFILE_VERSION, 'edit_profile': EDIT_PROFILE, 'policy': compiled['policy'], 'programs': programs, 'entity_program': idx,
              'disabled_ops_per_program': [p[0].get('disabled_ops', []) for p in programs], 'local_program_index': [edit_program_index(p) for p in programs],
              'rule_index': compiled['rule_index'], 'resolved_thresholds': compiled['resolved_thresholds'], 'n_unknown': compiled['n_unknown'], 'alias_of': None,
              'execution_order': 'fixed: ' + ' -> '.join(ta.PRIMITIVES)}
        self.specs[mid] = ms
        context.write_json(W.aug_dir(self.ctx.job_dir) / (mid + '__compiled.json'), {'material_spec': ms, 'assignment': compiled['assignment'], 'key': key, 'phys_id': phys['material_id']})
        return br.Candidate(mid, ms, self.job)

    def inspect_material(self, plan_id, arguments, *, remaining_seconds):
        out = super().inspect_material(plan_id, arguments, remaining_seconds=remaining_seconds)
        rec = self._reg()[plan_id]
        if rec.get('summary_path') and rec.get('kind') in ('tempo_v2', 'tempo_v2_recipe_edit'):
            s = context.read_json(rec['summary_path'])
            ed = s.get('edit')
            if ed:
                legal = s['legal_parents']
                out['recipe_edit'] = {'entity_disabled_ops': ed['entity_disabled_ops'],
                                      'drawn_windows_per_component': {n: v['drawn'] for n, v in ed['shadow_per_step'].items() if v['drawn']},
                                      'skipped_windows_per_component': {n: v['skipped'] for n, v in ed['shadow_per_step'].items() if v['skipped']},
                                      'skip_fraction_of_legal_windows': {n: round(v['skipped'] / legal, 4) for n, v in ed['shadow_per_step'].items() if v['skipped']},
                                      'retained_step_sets': ed['retained_step_sets'], 'windows_no_retained_step': ed['windows_no_retained_step'],
                                      'note': 'drawn = the preset drew that component on the window; skipped = drawn but disabled for that entity; the other windows are identical to the preset'}
        return out


def adapter_factory_edit(common: Path, job_index: int, public_ids, seeds):
    def make(root, job, led, repo, *, tool_error_feedback=True, roster=None, seeds=seeds, dataset=DATASET):
        return EditAdapter(root, job, led, repo, common=common, job_index=job_index, public_ids=public_ids, tool_error_feedback=tool_error_feedback, roster=roster, seeds=seeds, dataset=dataset)
    return make


def fast_branch_edit(path, job, knowledge, led, client, common, *, job_index, public_ids, seeds):
    return base.branch(path, job, knowledge, led, client, None, evidence_roundtrip=EVIDENCE_ROUNDTRIP, max_tool_corrections=MAX_TOOL_CORRECTIONS, tool_contracts=CONTRACTS_EDIT,
                       seeds=seeds, adapter_factory=adapter_factory_edit(common, job_index, public_ids, seeds), limits=LIMITS, dataset=DATASET, commit_fn=W.commit_branch,
                       allowed_features=W.ALLOWED_FEATURES, entity_count=W.n_of(job))


def resume_fast_branch_edit(path, job, knowledge, led, client, common, *, job_index, public_ids, seeds):
    return base.resume_branch(path, job, knowledge, led, client, max_tool_corrections=MAX_TOOL_CORRECTIONS, tool_contracts=CONTRACTS_EDIT, seeds=seeds,
                              adapter_factory=adapter_factory_edit(common, job_index, public_ids, seeds), evidence_roundtrip=EVIDENCE_ROUNDTRIP, limits=LIMITS, dataset=DATASET,
                              commit_fn=W.commit_branch, allowed_features=W.ALLOWED_FEATURES, entity_count=W.n_of(job))


# ============================================================================= Random_edit_B4 (task §5.2): the parent's draw structure on the 29 local programs
def draw_random_supply_edit(job: str, job_index: int, table: list) -> dict:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([RANDOM_SEED_ROOT, int(job_index)])))
    n = len(table)
    public_keys = {W.material_key(ta.compile_plan(W.public_policy(m), table)['assignment']) for m in PUBLIC_STEPS}
    taken_keys, drawn_uniform, plans = set(public_keys), set(), []
    eligible = [i for i in range(1, len(EDIT_PROGRAMS))]                   # the 28 non-preset local programs (uniform slots)

    def draw_uniform():
        pool = [i for i in eligible if i not in drawn_uniform]
        i = int(pool[int(rng.integers(len(pool)))])
        drawn_uniform.add(i)
        return i

    for k_ in (1, 2):
        i = draw_uniform()
        pol = {'default': {'steps': EDIT_PROGRAMS[i]}, 'rules': [], 'rationale': 'random uniform local program %d (control arm)' % i, 'observation_fields_used': []}
        comp = compile_plan_edit(pol, table)
        taken_keys.add(W.material_key(comp['assignment']))
        plans.append({'slot': 'R%d' % k_, 'kind': 'uniform', 'program_index': i, 'disabled_ops': EDIT_PROGRAMS[i][0].get('disabled_ops', []), 'policy': pol, 'fallback': False, 'tries': 1})
    for k_ in (3, 4):
        found = None
        for t in range(1, RANDOM_MAX_TRIES + 1):
            f = rd.FIELDS[int(rng.integers(len(rd.FIELDS)))]
            q = RANDOM_QUANTILES[int(rng.integers(len(RANDOM_QUANTILES)))]
            a, b = [int(x) for x in rng.choice(len(EDIT_PROGRAMS), size=2, replace=False)]
            pol = {'default': {'steps': EDIT_PROGRAMS[a]}, 'rules': [{'when': {'feature': f, 'op': '>=', 'value': {'quantile': q}}, 'steps': EDIT_PROGRAMS[b]}],
                   'rationale': 'random conditional local plan (control arm)', 'observation_fields_used': [f]}
            comp = compile_plan_edit(pol, table)
            key = W.material_key(comp['assignment'])
            if set(comp['rule_index']) == {-1, 0} and key not in taken_keys:
                found = {'slot': 'R%d' % k_, 'kind': 'conditional', 'field': f, 'quantile': q, 'default_program_index': a, 'hit_program_index': b,
                         'default_disabled_ops': EDIT_PROGRAMS[a][0].get('disabled_ops', []), 'hit_disabled_ops': EDIT_PROGRAMS[b][0].get('disabled_ops', []), 'policy': pol,
                         'fallback': False, 'tries': t, 'entities_hit': int(sum(x == 0 for x in comp['rule_index'])), 'n_unknown': comp['n_unknown']}
                taken_keys.add(key)
                break
        if found is None:
            i = draw_uniform()
            pol = {'default': {'steps': EDIT_PROGRAMS[i]}, 'rules': [], 'rationale': 'random uniform local program %d (conditional slot fallback)' % i, 'observation_fields_used': []}
            taken_keys.add(W.material_key(compile_plan_edit(pol, table)['assignment']))
            found = {'slot': 'R%d' % k_, 'kind': 'uniform', 'program_index': i, 'disabled_ops': EDIT_PROGRAMS[i][0].get('disabled_ops', []), 'policy': pol, 'fallback': True, 'tries': RANDOM_MAX_TRIES}
        plans.append(found)
    return {'job': job, 'job_index': job_index, 'rng': 'Generator(PCG64(SeedSequence([%d, %d])))' % (RANDOM_SEED_ROOT, job_index), 'program_table': '29 local programs (preset, 7 single, 21 double)',
            'drawn_local': now(), 'plans': plans, 'hidden_from_fast': True}


def random_branch_edit(root: Path, job: str, led, common: Path, supply: dict, *, job_index, public_ids, seeds) -> br.RunResult:
    root.mkdir(parents=True, exist_ok=False)
    ad = EditAdapter(root, job, led, REPO, common=common, job_index=job_index, public_ids=public_ids, seeds=seeds)
    n = ad.n
    trace = []

    def emit(event, **kw):
        row = br.json_copy({'event_id': '%s:%d' % (job, len(trace)), 'event': event, **kw})
        trace.append(row)
        base.event_sink(root / 'trace.jsonl')(row)
    context.write_json(root / 'knowledge.json', asdict(br.Knowledge()))
    cands = list(ad.baselines())
    emit('job_started', mode='random_edit_b4', baseline_ids=[c.plan_id for c in cands], seeds=list(seeds), supply_rng=supply['rng'])
    for pl in supply['plans']:
        c = ad.build_material({'plan_id': pl['slot'], 'policy': pl['policy']}, remaining_seconds=led.remaining())
        emit('random_material', plan_id=c.plan_id, material_spec=c.material_spec, draw={k: v for k, v in pl.items() if k != 'policy'})
        c = ad.evaluate(c, seeds, feedback=True, remaining_seconds=led.remaining())
        cands.append(c)
        emit('random_evaluated', plan_id=c.plan_id, feedback=c.feedback(seeds, 2, n))
    order = [c.plan_id for c in cands]
    fb = {c.plan_id: c.feedback(seeds, 2, n)['mean_loss'] for c in cands}
    lo = min(fb.values())
    chosen = next(c for c in cands if fb[c.plan_id] - lo <= TIE_ABS)
    reason = 'Random_edit_B4: minimum three-seed C_A mean among %s; ties (<= 1e-12) in that order.' % ', '.join(order)
    W.commit_branch(root, DATASET, job, chosen.plan_id, reason, seeds[0])
    emit('committed', plan_id=chosen.plan_id, delivery_model_ref=chosen.model_refs[0], reason=reason)
    result = br.RunResult('COMPLETE', job, 0, chosen.plan_id, chosen.model_refs[0], 0, 0, len(supply['plans']), trace)
    context.write_json(root / 'branch_result.json', asdict(result))
    print('RANDOM_EDIT', job, 'commit', chosen.plan_id, flush=True)
    return result


# ============================================================================= stage: common cache, arms, ablation view, labels
def prepare_common_edit(root: Path, job: str, led, cfg: dict) -> Path:
    common = root / (job + '_common')
    ji, seeds, public_ids = cfg['job_index'][job], tuple(cfg['seeds']), tuple(cfg['public_ids'])
    src = (cfg.get('cache_from') or {}).get(job)
    if src and not (common / job / 'aug_materials' / 'index.json').exists():
        W.copy_physical_cache(Path(src), common, job, seeds=seeds)
    if not all(m in W.load_registry(common / job) for m in public_ids):
        t0 = time.time()
        rc, _ = W.run_sub(['--worker-build', common, job, ji], root / 'logs' / ('build_%s.log' % job), 1800)
        led.s['material_wall_seconds'] = led.s.get('material_wall_seconds', 0.0) + (time.time() - t0)
        led._save()
        if rc != 0:
            raise RuntimeError('common build failed for %s' % job)
    if cfg.get('random') and not (common / 'random_supply.json').exists():
        table = context.read_json(common / job / 'overview.json')['entities']
        dsks.write_once(common / 'random_supply.json', draw_random_supply_edit(job, ji, table))
    for m in public_ids:
        for s in seeds:
            rec = W.fit_physical(led, common, job, m, s)
            if rec['scores']['c_a']['status'] != 'SCORABLE':
                raise RuntimeError('C_A_NOT_SCORABLE: %s' % rec['cell_id'])
    return common


def _guard(root: Path, led, client, arm: str) -> None:
    led.check_wall()
    if client is not None:
        led.check_llm()
        if getattr(client, 'fatal', False) or rt.unknown_usage_blocks(led):
            raise RuntimeError('backend fatal or unknown usage')
        if arm == 'f_edit' and not W.proxy_reachable():
            raise RuntimeError('LLM proxy unreachable; stage stopped with labels withheld (resume when the proxy is back)')
    if W.numeric_seconds(root.parent) > NUMERIC_WALL_S:
        raise RuntimeError('numeric wall clock (4 h) exhausted')


def _run_arms_edit(root: Path, cfg: dict, led, client, branches, failures, *, resumed: bool) -> None:
    fc = W.FastClient(client, root, cfg['fast_system']) if client is not None else None
    for job in cfg['jobs']:
        if (root / (job + '_not_scorable.json')).exists():
            continue
        ji, seeds, public_ids = cfg['job_index'][job], tuple(cfg['seeds']), tuple(cfg['public_ids'])
        try:
            common = prepare_common_edit(root, job, led, cfg)
        except RuntimeError as exc:
            if 'C_A_NOT_SCORABLE' not in str(exc):
                raise
            context.write_json(root / (job + '_not_scorable.json'), {'job': job, 'status': 'C_A_NOT_SCORABLE'})
            failures.append({'job': job, 'kind': 'C_A_NOT_SCORABLE'})
            continue
        for arm in cfg['order'][job]:
            path = root / ('%s_%s' % (job, arm))
            branches.append((path, job))
            try:
                if path.exists() and (path / 'branch_result.json').exists():
                    prior = context.read_json(path / 'branch_result.json')
                    if prior['status'] == 'COMPLETE':
                        result = br.RunResult(**{k: v for k, v in prior.items() if k != 'trace'})
                    elif arm != 'f_edit':
                        raise RuntimeError('control branch cannot be resumed')
                    else:
                        result = resume_fast_branch_edit(path, job, br.Knowledge(), led, fc, common, job_index=ji, public_ids=public_ids, seeds=seeds)
                elif path.exists():
                    raise RuntimeError('branch directory without a result; inspect before any replay')
                elif arm == 'random_edit':
                    result = random_branch_edit(path, job, led, common, context.read_json(common / 'random_supply.json'), job_index=ji, public_ids=public_ids, seeds=seeds)
                elif arm == 'menu_ca':
                    result = W.menu_branch(path, job, led, common, job_index=ji, public_ids=public_ids, seeds=seeds)
                elif arm == 'f_edit':
                    result = fast_branch_edit(path, job, br.Knowledge(), led, fc, common, job_index=ji, public_ids=public_ids, seeds=seeds)
                else:
                    raise ValueError(arm)
                if result.status != 'COMPLETE':
                    failures.append({'branch': path.name, 'kind': result.failure_kind, 'reason': result.reason})
            except Exception as exc:  # noqa: BLE001
                failures.append({'branch': path.name, 'exception_type': type(exc).__name__, 'message': W._safe_message(exc)})
                print('BRANCH_FAILED', path.name, type(exc).__name__, flush=True)
            _guard(root, led, client, arm)


def ablation_stage(root: Path, cfg: dict, led, branches, failures) -> None:
    """After every planned arm of every batch was attempted: the seven frozen single-component ablations of each batch, built and fitted in
    the common cache (physical hits when an arm already built one), collected in a 0-LLM diagnostic view whose reference delivery is the
    preset. Scores never reach Fast / Random (they are all committed by now)."""
    for job in cfg['jobs']:
        if (root / (job + '_not_scorable.json')).exists():
            continue
        ji, seeds, public_ids = cfg['job_index'][job], tuple(cfg['seeds']), tuple(cfg['public_ids'])
        common = root / (job + '_common')
        path = root / (job + '_ablation')
        try:
            if (path / job / 'commit.json').exists():
                branches.append((path, job))
                continue
            if path.exists():
                shutil.rmtree(path)                    # the view holds only copies of cached cells; rebuilt from the cache
            ad = EditAdapter(path, job, led, REPO, common=common, job_index=ji, public_ids=public_ids, seeds=seeds)
            ad.baselines()
            rows = {}
            for n in cfg['ablation']:
                mid = 'ABL_' + n[3:]
                c = ad.build_material({'plan_id': mid, 'policy': uniform_edit_policy([n], 'component ablation: the preset with %s disabled on every entity (0-LLM diagnostic)' % n)},
                                      remaining_seconds=led.remaining())
                before = led.s['fit_attempts']
                c = ad.evaluate(c, seeds, feedback=True, remaining_seconds=led.remaining())
                rows[mid] = {'disabled': [n], 'phys_id': ad._reg()[mid]['phys_id'], 'fits_charged': led.s['fit_attempts'] - before, 'c_a_mean': c.feedback(seeds, 2, ad.n)['mean_loss']}
            W.commit_branch(path, DATASET, job, ta.RECIPE, 'ablation diagnostic view: the preset recorded as its reference delivery; no research decision, 0 LLM', seeds[0],
                            extra={'diagnostic_view': True, 'ablations': rows})
            context.write_json(path / 'ablation_view.json', {'job': job, 'materials': rows, 'built_local': now(), 'after_all_arm_commits': True})
            branches.append((path, job))
            print('ABLATION', job, {m: round(r['c_a_mean'], 4) for m, r in rows.items()}, flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({'branch': path.name, 'exception_type': type(exc).__name__, 'message': W._safe_message(exc)})
            print('ABLATION_FAILED', job, type(exc).__name__, flush=True)
        _guard(root, led, None, 'ablation')


def stage_worker(cfg_path: Path) -> None:
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
        client = rt.MeteredClient(led, root / 'raw_responses', http_cap=int(cfg['http_cap'])) if cfg.get('llm') else None
        _run_arms_edit(root, cfg, led, client, branches, failures, resumed=False)
        context.write_json(root / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches]})
        if cfg.get('ablation'):
            ablation_stage(root, cfg, led, branches, failures)
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': W._safe_message(exc)})
        stopped = type(exc).__name__
        print('EXECUTION_STOP', stopped, flush=True)
    finally:
        context.write_json(root / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures})
    W.open_labels(root, cfg, branches, failures, led, withhold_reason=stopped and 'execution stopped before every planned arm was attempted (%s)' % stopped)


# ============================================================================= budget, config, stage launch, resume
def stage_caps(root: Path, stage: str) -> dict:
    P_ = paths(root)
    rec = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    if stage in rec:
        return rec[stage]
    led = rt.RuntimeLedger(P_['ledger'], **TOTAL)
    s, alloc = led.s, ALLOC[stage]
    snap = {k: s.get(k, 0) for k in ('fit_attempts', 'fits_ok', 'fits_failed', 'cache_hits', 'retries_used', 'llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'fit_wall_seconds')}
    snap['elapsed_s'] = time.time() - s['started_epoch']
    retries = min(TOTAL['max_retries'], TOTAL['max_retries'] - s['retries_used']) if stage == 'pilot' else 0
    caps = {'max_fit_attempts': min(TOTAL['max_fit_attempts'], s['fit_attempts'] + alloc['fits'] + retries),
            'max_llm_requests': min(TOTAL['max_llm_requests'], s['llm_requests'] + alloc['requests']),
            'max_llm_tokens': min(TOTAL['max_llm_tokens'], s['llm_tokens_in'] + s['llm_tokens_out'] + alloc['tokens']),
            'max_wall_s': TOTAL['max_wall_s'], 'max_retries': min(TOTAL['max_retries'], s['retries_used'] + retries)}
    rec[stage] = {'epoch_start': time.time(), 'local': now(), 'allocation': alloc, 'snapshot_at_start': snap, 'caps': caps, 'http_cap': min(HTTP_CAP, s['llm_http_attempts'] + alloc['requests'] + 2)}
    context.write_json(P_['stages'], rec)
    return rec[stage]


def stage_config(root: Path) -> Path:
    P_ = paths(root)
    path = P_['configs'] / 'pilot.json'
    if path.exists():
        return path
    sc = stage_caps(root, 'pilot')
    cfg = {'study': 'tempo_aug_recipe_edit_pilot', 'status': 'FROZEN', 'stage': 'pilot', 'output': str(P_['pilot']), 'jobs': list(JOBS), 'order': {j: list(ORDER[j]) for j in JOBS},
           'knowledge': {}, 'labels_e': True, 'llm': True, 'random': True, 'ablation': list(ABLATION), 'seeds': list(SEEDS), 'job_index': {j: JOB_INDEX[j] for j in JOBS},
           'public_ids': list(PUBLIC), 'limits': LIMITS, 'max_tool_corrections': MAX_TOOL_CORRECTIONS, 'evidence_roundtrip': EVIDENCE_ROUNDTRIP, 'fast_token_cap': FAST_TOKEN_CAP,
           'max_output_tokens': MAX_OUTPUT_TOKENS, 'fast_system': FAST_SYSTEM_EDIT, 'contracts': CONTRACTS_EDIT, 'caps': sc['caps'], 'http_cap': sc['http_cap'],
           'ledger_path': str(P_['ledger']), 'fit_retry': True, 'fit_attempts_at_stage_start': context.read_json(P_['ledger'])['fit_attempts'],
           'cache_from': {j: str(p) for j, p in CACHE_FROM.items()}, 'model': MODEL}
    dsks.write_once(path, cfg)
    return path


def run_stage(root: Path, cfg_path: Path) -> dict:
    P_ = paths(root)
    sroot = P_['pilot']
    if not (sroot / 'experiment_started.json').exists():
        led = context.read_json(P_['ledger'])
        remaining = TOTAL['max_wall_s'] - (time.time() - led['started_epoch'])
        if remaining <= 0:
            raise RuntimeError('package wall exhausted')
        P_['logs'].mkdir(parents=True, exist_ok=True)
        print('STAGE_START pilot', now(), flush=True)
        with (P_['logs'] / 'pilot.log').open('a', encoding='utf-8') as log:
            p = subprocess.Popen([sys.executable, '-B', '-m', MODULE, '--stage-worker', str(cfg_path)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=W.worker_env())
            try:
                rc = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)
                context.write_json(root / 'pilot_supervisor_timeout.json', {'epoch': time.time(), 'pid': p.pid})
                rc = 124
        print('STAGE_EXIT pilot', rc, now(), flush=True)
    return dsks.stage_status(sroot)


def resume_stage(root: Path, *, accept_unknown_usage: bool = False, restart=()) -> None:
    """The parent's boundary: labels still withheld, one continuation, completed branches kept, a never-sent Fast call continued, missing
    arms then the ablation view, then labels."""
    from evaluation.main_protocol_p4 import run_batch_research_roundtrip as rtp
    sroot = paths(root)['pilot']
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
    for name in restart:
        b = sroot / name
        prior = context.read_json(b / 'branch_result.json') if (b / 'branch_result.json').exists() else None
        if not b.is_dir() or (prior and (prior['status'] == 'COMPLETE' or prior['new_evaluations'])) or any(b.rglob('commit.json')):
            raise RuntimeError('only an interrupted branch without commit or new evaluation can be restarted: %s' % name)
        dest = sroot / ('%s__interrupted_1' % name)
        shutil.move(b, dest)
        moved.append({'branch': name, 'kept_as': dest.name, 'prior_status': prior and prior['status']})
    context.write_json(sroot / 'resume.json', {'epoch': time.time(), 'accepted_unknown_usage': bool(accept_unknown_usage), 'restarted_branches': moved})
    rt.FIT_RETRY = bool(cfg.get('fit_retry'))
    branches, failures, stopped = [], [], None
    try:
        client = rt.MeteredClient(led, sroot / 'raw_responses', http_cap=int(cfg['http_cap']))
        _run_arms_edit(sroot, cfg, led, client, branches, failures, resumed=True)
        context.write_json(sroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches], 'resumed': True})
        ablation_stage(sroot, cfg, led, branches, failures)
    except Exception as exc:  # noqa: BLE001
        failures.append({'stage': 'execution', 'exception_type': type(exc).__name__, 'message': W._safe_message(exc)})
        stopped = type(exc).__name__
    finally:
        context.write_json(sroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures, 'resumed': True})
    W.open_labels(sroot, cfg, branches, failures, led, resumed=True, withhold_reason=stopped and 'resumed stage stopped again (%s)' % stopped)


# ============================================================================= preflight (0 cost) and frozen config
def environment() -> dict:
    return {'python': sys.version.split()[0], 'executable': sys.executable, 'numpy': np.__version__, 'KMP_DUPLICATE_LIB_OK_at_start': os.environ.get('KMP_DUPLICATE_LIB_OK'),
            'workers_env_kmp_removed': True, 'controller_imports_torch': 'torch' in sys.modules, 'proxy_reachable_tcp_precheck': W.proxy_reachable(),
            'api_key_present': bool(next((os.environ.get(n, '').strip() for n in ('CPA_API_KEY', 'OPENAI_API_KEY') if os.environ.get(n, '').strip()), ''))}


def cache_binding(job: str, src: Path, seeds=SEEDS) -> dict:
    js = rd.resolve_job(DATASET, job)
    out = {'source': str(src), 'exists': (src / 'aug_materials' / 'index.json').exists()}
    if not out['exists']:
        return out
    with np.load(src / 'scaler.npz') as z:
        out['binding'] = {'dataset': str(z['dataset']) == DATASET, 't': int(z['t']) == js.t, 'roster': [str(x) for x in z['roster']] == js.roster}
    reg = W.load_registry(src)
    out['public_materials'] = {m: m in reg for m in PUBLIC}
    out['materials_total'] = len(reg)
    cells = {}
    for p in (src / 'cells').glob('*.json'):
        r = context.read_json(p)
        if r.get('status') == 'OK' and not r.get('tag') and r['material_id'] in PUBLIC and r['model_seed'] in seeds:
            cells[(r['material_id'], r['model_seed'])] = r
    out['public_cells'] = len(cells)
    out['public_models_present'] = sum(Path(r['model_path']).exists() for r in cells.values())
    out['cells_hold_c_a_only'] = all(set(r.get('scores', {})) == {'c_a'} for r in cells.values())
    out['forbidden_label_files'] = [p.name for p in src.rglob('*') if p.name in ('c_b_scores.json', 'e_scores.json', 'e_frozen.json', 'commit.json') or p.name.startswith('predictions_e')][:3]
    out['e_target_rows'] = list(js.e_target_rows)
    out['ok'] = all(out['binding'].values()) and all(out['public_materials'].values()) and out['public_cells'] == 4 * len(seeds) and out['public_models_present'] == 4 * len(seeds) \
        and out['cells_hold_c_a_only'] and not out['forbidden_label_files'] and js.e_target_rows[1] <= SEALED_FROM_ROW
    return out


def preflight(root: Path = ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    fc = root / 'frozen_config.json'
    if fc.exists():
        return context.read_json(fc)
    caches = {j: cache_binding(j, CACHE_FROM[j]) for j in JOBS}
    wiring_cache = {'source': str(WIRING_CACHE), 'exists': (WIRING_CACHE / 'aug_materials' / 'index.json').exists(),
                    'public_cells': sum((WIRING_CACHE / 'cells' / (W.phys_cell_id(WIRING_JOB, m, s) + '.json')).exists() for m in WIRING_PUBLIC for s in WIRING_SEEDS),
                    'preset_material': ta.RECIPE in W.load_registry(WIRING_CACHE)}
    cfg = {'package': PACKAGE, 'task': TASK, 'frozen_local': now(), 'exposure': EXPOSURE, 'dataset': DATASET, 'roster': list(rd.DATASETS[DATASET]['roster']),
           'jobs': {j: {'t': JOB_T[j], 'job_index': JOB_INDEX[j], 'order': list(ORDER[j]), 'e_target_rows': list(rd.resolve_job(DATASET, j).e_target_rows)} for j in JOBS},
           'seeds': list(SEEDS), 'public_ids': list(PUBLIC), 'consumer': W.CONSUMER,
           'program_table': {'size': len(EDIT_PROGRAMS), 'order': 'preset, then single disables in PRIMITIVES order, then double disables in combinations order',
                             'programs': [{'index': i, 'disabled_ops': p[0].get('disabled_ops', [])} for i, p in enumerate(EDIT_PROGRAMS)]},
           'edit_semantics': {'op': ta.RECIPE_EDIT, 'text': ta.RECIPE_EDIT_TEXT, 'parent_program_index': PARENT_PROGRAM_INDEX, 'seed_rule': 'entity_program_seed(job_index, 52, entity); the switch never enters the seed',
                              'rng_isolation': 'shadow preset run on a window copy advances the entity stream and records the three RNG states at every step entry; retained steps are replayed on the real window from detached copies of those states; disabled steps are skipped without redraw'},
           'ablation_catalog': {'materials': ['ABL_' + n[3:] for n in ABLATION], 'disabled': [[n] for n in ABLATION], 'fixed_before_any_run': True, 'scores_reach_fast_or_random': False,
                                'when': 'after every planned arm of every batch was attempted (all commits frozen)'},
           'arms': {'F_edit': {'limits': LIMITS, 'token_cap': FAST_TOKEN_CAP, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'model': MODEL, 'knowledge': 'none (no Skill, no Generic, no cross-job trajectory)'},
                    'Random_edit_B4': {'rng': 'Generator(PCG64(SeedSequence([%d, job_index])))' % RANDOM_SEED_ROOT, 'slots': 'R1 R2 uniform from the 28 non-preset programs without replacement; R3 R4 conditional (field ~ U(FIELDS), quantile ~ U{.25,.5,.75}, >=, two distinct of the 29), both groups non-empty, dedup, 256 tries then uniform fallback',
                                       'delivery': 'argmin three-seed C_A mean over the four public references + R1..R4; ties <= 1e-12 in public then draw order', 'frozen': 'before the job\'s first new fit; hidden from Fast'},
                    'Menu_CA': 'argmin three-seed C_A mean over the four public references; ties in public order; 0 fits, 0 LLM'},
           'fast_system': FAST_SYSTEM_EDIT, 'contracts': CONTRACTS_EDIT, 'action_table': edit_action_table(),
           'budget': {**TOTAL, 'http_cap': HTTP_CAP, 'alloc': ALLOC, 'numeric_wall_s': NUMERIC_WALL_S, 'slow_requests': 0, 'new_sha': 0, 'git_commit': 0, 'single_fit_worker': True},
           'metrics': {'delta_pp': 'Delta_j(A,B) = 100 (mean_s E_j(A) - mean_s E_j(B)) / mean_s E_j(None); > 0 = B better; paired seed vector with the same denominator; SE and df=2 t95 (t=%.5f) on that vector' % T975_DF2,
                       'decomposition': 'pool_gain = E(NoMix) - min_{own pool} E; selection_loss = E(commit) - min_{own pool} E; net = pool_gain - selection_loss = E(NoMix) - E(commit); public_best - pool_best = opportunity added by the arm\'s own edits (0 when it built none); all in pp of None',
                       'shadow_selection': 'argmin three-seed C_A mean over the F_edit pool (public + own evaluated); its E reported next to the real commit; never written back',
                       'unit': 'job (4 batches); equal-weight mean, per-origin and leave-one-job-out ranges; 3 seeds = training randomness only; no re-drawn augmentation seed'},
           'caches': caches, 'wiring_cache': wiring_cache, 'environment': environment(), 'sealed_from_row': SEALED_FROM_ROW}
    if not all(c['ok'] for c in caches.values()):
        context.write_json(root / 'preflight_failed.json', cfg)
        raise RuntimeError('cache preflight failed: %s' % {j: c.get('ok') for j, c in caches.items()})
    dsks.write_once(fc, cfg)
    print('PREFLIGHT_OK', {j: (c['public_cells'], c['materials_total']) for j, c in caches.items()}, 'wiring_cache', wiring_cache['public_cells'])
    return cfg


# ============================================================================= smoke (0 fits, 0 LLM): controller checks + one torch worker on the cached L1 material
def smoke_scratch(out: Path, job: str = 'RD02_L1') -> Path:
    """A scratch physical cache of one job: static T objects and the public materials copied from the frozen cache (read-only source)."""
    src = CACHE_FROM[job]
    dst = out / (job + '_common') / job
    if not (dst / 'aug_materials' / 'index.json').exists():
        dst.mkdir(parents=True, exist_ok=True)
        for n in ('scaler.npz', 'overview.json'):
            shutil.copy2(src / n, dst / n)
        shutil.copytree(src / 'materials', dst / 'materials', dirs_exist_ok=True)
        reg = W.load_registry(src)
        keep = {}
        (dst / 'aug_materials').mkdir(exist_ok=True)
        for m in PUBLIC:
            r = dict(reg[m])
            for k_ in ('path', 'summary_path'):
                if r.get(k_):
                    shutil.copy2(r[k_], dst / 'aug_materials' / Path(r[k_]).name)
                    r[k_] = str(dst / 'aug_materials' / Path(r[k_]).name)
            keep[m] = r
        W.save_registry(dst, keep)
    return out / (job + '_common')


def worker_smoke(out: str) -> None:
    rd._init_openmp_before_torch()
    import torch
    out = Path(out)
    job, ji = 'RD02_L1', JOB_INDEX['RD02_L1']
    common = smoke_scratch(out, job)
    ctx = rd.open_job(DATASET, job, common, 'material')
    Z, ent, k = ta.joint_parents(ctx)
    starts = ta.window_starts(ctx)
    with np.load(W.load_registry(common / job)[ta.RECIPE]['path']) as z:
        Cref = np.concatenate([z['Xc'], z['yc']], axis=1)
    rec = {'job': job, 'local': now(), 'checks': {}}
    t_before, n_before = torch.get_rng_state().clone(), np.random.get_state()
    # 1) empty switch through the replay path == the cached preset material, bitwise, on every window of every entity; sample coverage
    same = calendar = censor_noop = conv = 0
    per_window_shadow = {}
    for e in range(len(ctx.job.roster)):
        rows = np.flatnonzero(ent == e)
        s = ta.Streams(W.entity_program_seed(ji, PARENT_PROGRAM_INDEX, e))
        for p in rows:
            o, ex, ident, shadow = ta.run_program_edit(Z[p], [{'op': ta.RECIPE_EDIT, 'disabled_ops': []}], s, starts[p])
            same += int(np.array_equal(o, Cref[p]))
            per_window_shadow[int(p)] = shadow
            calendar += int('tp_calendar' in shadow)
            censor_noop += int(any(n == 'tp_censor' and i for n, i in zip(ex, ident)))
            conv += int('tp_random_conv' in shadow)
    rec['checks']['empty_switch_bitwise_equals_cached_preset'] = {'equal_windows': same, 'of': int(Z.shape[0]), 'pass': same == Z.shape[0]}
    rec['checks']['sample_coverage'] = {'calendar_windows': calendar, 'censor_noop_windows': censor_noop, 'conv_windows': conv, 'pass': min(calendar, censor_noop, conv) > 0}
    # 2) disabling does not move the other windows' draws; unaffected windows stay bitwise equal; training / global RNG untouched
    for disabled in (['tp_random_conv'], ['tp_shock', 'tp_censor']):
        shadow_same = unaffected_equal = unaffected = affected = affected_changed = 0
        for e in range(len(ctx.job.roster)):
            rows = np.flatnonzero(ent == e)
            s = ta.Streams(W.entity_program_seed(ji, PARENT_PROGRAM_INDEX, e))
            for p in rows:
                o, ex, ident, shadow = ta.run_program_edit(Z[p], [{'op': ta.RECIPE_EDIT, 'disabled_ops': disabled}], s, starts[p])
                shadow_same += int(shadow == per_window_shadow[int(p)])
                hit = any(n in disabled for n in shadow)
                if hit:
                    affected += 1
                    affected_changed += int(not np.array_equal(o, Cref[p]))
                else:
                    unaffected += 1
                    unaffected_equal += int(np.array_equal(o, Cref[p]))
        rec['checks']['disable_%s' % '+'.join(disabled)] = {'shadow_steps_unchanged': shadow_same, 'of': int(Z.shape[0]), 'unaffected_windows_equal': unaffected_equal, 'unaffected_windows': unaffected,
                                                           'affected_windows': affected, 'affected_windows_changed': affected_changed,
                                                           'pass': shadow_same == Z.shape[0] and unaffected_equal == unaffected and affected > 0}
    rec['checks']['global_rng_untouched'] = {'torch': bool(torch.equal(t_before, torch.get_rng_state())), 'numpy_legacy': bool(np.array_equal(n_before[1], np.random.get_state()[1])),
                                             'pass': bool(torch.equal(t_before, torch.get_rng_state())) and bool(np.array_equal(n_before[1], np.random.get_state()[1]))}
    # 3) conditional plan: unmodified entities keep the preset material; modified entities equal the uniform switch material; order independence
    table = ctx.overview()['entities']
    uni = compile_plan_edit(uniform_edit_policy(['tp_random_conv'], 'smoke'), table)['assignment']
    cond = [([{'op': ta.RECIPE}] if e < 6 else [ta.edit_step(['tp_random_conv'])]) for e in range(len(ctx.job.roster))]
    ru = build_material_edit(ctx, uni, 'TA001', job_index=ji)                 # cache A: uniform first, then conditional
    rc_ = build_material_edit(ctx, cond, 'TA002', job_index=ji)
    common_b = smoke_scratch(out / 'order_b', job)
    ctx_b = rd.open_job(DATASET, job, common_b, 'material')
    rc_b = build_material_edit(ctx_b, cond, 'TA001', job_index=ji)            # cache B: conditional first
    with np.load(ru['path']) as a, np.load(rc_['path']) as b, np.load(rc_b['path']) as c:
        U, Cc, Cb = np.concatenate([a['Xc'], a['yc']], axis=1), np.concatenate([b['Xc'], b['yc']], axis=1), np.concatenate([c['Xc'], c['yc']], axis=1)
    unmod = np.flatnonzero(ent < 6)
    mod = np.flatnonzero(ent >= 6)
    rec['checks']['conditional_plan'] = {'unmodified_entities_equal_preset': bool(np.array_equal(Cc[unmod], Cref[unmod])), 'modified_entities_equal_uniform_switch': bool(np.array_equal(Cc[mod], U[mod])),
                                         'other_build_order_bitwise_identical': bool(np.array_equal(Cc, Cb)), 'uniform_differs_from_preset': bool(not np.array_equal(U, Cref))}
    rec['checks']['conditional_plan']['pass'] = all(v for k_, v in rec['checks']['conditional_plan'].items())
    su = context.read_json(ru['summary_path'])
    pre = context.read_json(W.load_registry(common / job)[ta.RECIPE]['summary_path'])
    rec['checks']['shadow_counts_equal_preset_per_step'] = {'pass': all(su['edit']['shadow_per_step'][n]['drawn'] == pre['per_step'][n]['windows'] for n in ta.PRIMITIVES),
                                                            'skipped_conv': su['edit']['shadow_per_step']['tp_random_conv']['skipped'], 'drawn_conv': su['edit']['shadow_per_step']['tp_random_conv']['drawn']}
    rec['checks']['shadow_counts_equal_preset_per_step']['pass'] &= su['edit']['shadow_per_step']['tp_random_conv']['skipped'] == su['edit']['shadow_per_step']['tp_random_conv']['drawn']
    # 4) cache / key semantics and branch isolation
    keys = {i: W.material_key([p] * len(ctx.job.roster)) for i, p in enumerate(EDIT_PROGRAMS)}
    pub_key = W.load_registry(common / job)[ta.RECIPE]['key']
    rec['checks']['keys'] = {'29_distinct': len(set(keys.values())) == 29, 'preset_program_key_equals_public_preset': keys[0] == pub_key,
                             'edit_keys_differ_from_public': all(keys[i] != pub_key for i in range(1, 29)), 'empty_disable_canonicalizes_to_preset': ta.edit_step([]) == {'op': ta.RECIPE}}
    rec['checks']['keys']['pass'] = all(rec['checks']['keys'].values())
    led = rt.RuntimeLedger(out / 'smoke_ledger.json', max_fit_attempts=0, max_llm_requests=0, max_llm_tokens=0, max_wall_s=3600, max_retries=0)
    view_a = out / 'view_a'
    ad = EditAdapter(view_a, job, led, REPO, common=common, job_index=ji, public_ids=PUBLIC, seeds=SEEDS)
    errors = {}
    for name, pol in (('unknown_component', uniform_edit_policy([], 'x') | {'default': {'steps': [{'op': ta.RECIPE_EDIT, 'disabled_ops': ['tp_mixup']}]}}),
                      ('three_disabled', {'default': {'steps': [{'op': ta.RECIPE_EDIT, 'disabled_ops': ['tp_regime', 'tp_shock', 'tp_censor']}]}, 'rules': [], 'rationale': 'x', 'observation_fields_used': []}),
                      ('strength_parameter', {'default': {'steps': [{'op': ta.RECIPE_EDIT, 'disabled_ops': ['tp_shock'], 'strength': 0.5}]}, 'rules': [], 'rationale': 'x', 'observation_fields_used': []}),
                      ('explicit_composition', {'default': {'steps': [{'op': 'tp_shock'}, {'op': 'tp_censor'}]}, 'rules': [], 'rationale': 'x', 'observation_fields_used': []}),
                      ('empty_steps', {'default': {'steps': []}, 'rules': [], 'rationale': 'x', 'observation_fields_used': []}),
                      ('preset_duplicate', uniform_edit_policy([], 'preset again'))):
        try:
            ad.build_material({'plan_id': 'X_' + name, 'policy': pol}, remaining_seconds=100.0)
            errors[name] = 'ACCEPTED'
        except br.ToolInputError as exc:
            errors[name] = {'message': str(exc)[:160], 'details': {k_: v for k_, v in exc.details.items() if k_ in ('execution_status', 'alias_of')}}
    rec['checks']['invalid_arguments_return_tool_errors'] = {'errors': errors, 'pass': all(v != 'ACCEPTED' for v in errors.values()) and errors['preset_duplicate']['details'].get('alias_of') == ta.RECIPE}
    c1 = ad.build_material({'plan_id': 'E_noconv', 'policy': uniform_edit_policy(['tp_random_conv'], 'smoke: disable conv')}, remaining_seconds=100.0)
    insp = ad.inspect_material('E_noconv', {'plan_id': 'E_noconv', 'entity_index': 0}, remaining_seconds=100.0)
    view_b = out / 'view_b'
    ad_b = EditAdapter(view_b, job, led, REPO, common=common, job_index=ji, public_ids=PUBLIC, seeds=SEEDS)
    rec['checks']['branch_isolation'] = {'view_a_plans': sorted(ad._reg()), 'view_b_plans': sorted(ad_b._reg()), 'physical_reused': ad._reg()['E_noconv']['phys_id'] == 'TA001',
                                         'pass': sorted(ad_b._reg()) == sorted(PUBLIC) and 'E_noconv' in ad._reg() and ad._reg()['E_noconv']['phys_id'] == 'TA001'}
    rec['checks']['inspect_material_reports_edit'] = {'recipe_edit_present': 'recipe_edit' in insp, 'skipped_conv': (insp.get('recipe_edit') or {}).get('skipped_windows_per_component'),
                                                      'window_present': 'window' in insp, 'pass': 'recipe_edit' in insp and 'window' in insp and c1.material_spec['disabled_ops_per_program'] == [['tp_random_conv']]}
    rec['checks']['no_fit_in_smoke'] = {'fit_attempts': led.s['fit_attempts'], 'pass': led.s['fit_attempts'] == 0}
    rec['all_pass'] = all(v['pass'] for v in rec['checks'].values())
    context.write_json(out / 'smoke_worker.json', rec)
    print('SMOKE_WORKER', rec['all_pass'], json.dumps({k_: v.get('pass') for k_, v in rec['checks'].items()}), flush=True)


def smoke(root: Path = ROOT) -> dict:
    sdir = paths(root)['smoke']
    if (sdir / 'smoke.json').exists():
        return context.read_json(sdir / 'smoke.json')
    sdir.mkdir(parents=True, exist_ok=True)
    rec = {'local': now(), 'controller': {}}
    # controller-side (0 torch): program table, DSL, random supply shape, contracts, prompt facts
    table = context.read_json(CACHE_FROM['RD02_L1'] / 'overview.json')['entities']
    sup = draw_random_supply_edit('RD02_L1', JOB_INDEX['RD02_L1'], table)
    kinds = [p['kind'] for p in sup['plans']]
    rec['controller']['random_supply'] = {'slots': [p['slot'] for p in sup['plans']], 'kinds': kinds, 'fallbacks': [p['fallback'] for p in sup['plans']],
                                          'distinct_keys': len({W.material_key(compile_plan_edit(p['policy'], table)['assignment']) for p in sup['plans']}) == 4,
                                          'reproducible': draw_random_supply_edit('RD02_L1', JOB_INDEX['RD02_L1'], table)['plans'] == sup['plans'],
                                          'pass': kinds[:2] == ['uniform', 'uniform'] and len(sup['plans']) == 4}
    rec['controller']['random_supply']['pass'] &= rec['controller']['random_supply']['distinct_keys'] and rec['controller']['random_supply']['reproducible']
    rec['controller']['program_table'] = {'size': len(EDIT_PROGRAMS), 'single': sum(len(p[0].get('disabled_ops', [])) == 1 for p in EDIT_PROGRAMS), 'double': sum(len(p[0].get('disabled_ops', [])) == 2 for p in EDIT_PROGRAMS),
                                          'pass': len(EDIT_PROGRAMS) == 29 and EDIT_PROGRAMS[0] == [{'op': ta.RECIPE}]}
    rec['controller']['contracts'] = {'tool_names_unchanged': set(CONTRACTS_EDIT) == set(W.CONTRACTS), 'build_names_edit_op': ta.RECIPE_EDIT in json.dumps(CONTRACTS_EDIT['build_material']),
                                      'prompt_states_joint_targets': '子训练目标也改变' in FAST_SYSTEM_EDIT and 'child training targets' in FAST_SYSTEM_EDIT,
                                      'prompt_no_component_verdict': not any(w in FAST_SYSTEM_EDIT for w in ('肯定有害', 'certainly harmful', 'is harmful', '总是更好')),
                                      'no_job_or_source_names_in_prompt': not any(w in FAST_SYSTEM_EDIT for w in ('RD02', 'PRSA', 'Beijing', 'L1', 'L3', 'L5', 'L7'))}
    rec['controller']['contracts']['pass'] = all(rec['controller']['contracts'].values())
    rec['controller']['controller_imports_torch'] = 'torch' in sys.modules
    rc, secs = run_sub(['--worker-smoke', str(sdir / 'worker')], sdir / 'worker.log', 1800)
    w = context.read_json(sdir / 'worker' / 'smoke_worker.json') if rc == 0 and (sdir / 'worker' / 'smoke_worker.json').exists() else {'all_pass': False, 'rc': rc}
    rec['worker'] = w
    rec['worker_seconds'] = secs
    rec['all_pass'] = all(v['pass'] for k_, v in rec['controller'].items() if isinstance(v, dict)) and not rec['controller']['controller_imports_torch'] and bool(w.get('all_pass'))
    context.write_json(sdir / 'smoke.json', rec)
    print('SMOKE', rec['all_pass'], json.dumps({k_: v.get('pass') for k_, v in rec['controller'].items() if isinstance(v, dict)}), 'worker', w.get('all_pass'), flush=True)
    if not rec['all_pass']:
        raise RuntimeError('smoke failed')
    return rec


# ============================================================================= wiring (RD02_T1, <= 3 fits, scripted client, C_B only; E never read)
def wiring(root: Path = ROOT) -> dict:
    preflight(root)
    P_ = paths(root)
    wroot = P_['wiring']
    if (wroot / 'wiring.json').exists():
        return context.read_json(wroot / 'wiring.json')
    sc = stage_caps(root, 'wiring')
    led = rt.RuntimeLedger(P_['ledger'], **sc['caps'])
    rt.FIT_RETRY = False
    wroot.mkdir(parents=True, exist_ok=True)
    (wroot / 'logs').mkdir(exist_ok=True)
    job, ji, seeds = WIRING_JOB, WIRING_JOB_INDEX, WIRING_SEEDS
    cfg = {'stage': 'wiring', 'output': str(wroot), 'jobs': [job], 'seeds': list(seeds), 'job_index': {job: ji}, 'public_ids': list(WIRING_PUBLIC), 'labels_e': False, 'random': False, 'llm': False}
    context.write_json(wroot / 'config.json', cfg)
    context.write_json(wroot / 'experiment_started.json', {'epoch': time.time(), 'pid': os.getpid(), 'stage': 'wiring'})
    common = wroot / (job + '_common')
    copied = W.copy_physical_cache(WIRING_CACHE, common, job, seeds=seeds)
    out = {'label': 'WIRING_ACCEPTANCE (scripted client; no LLM; not an agent result)', 'job': job, 'seeds': list(seeds), 'cache': copied, 'fits_before': led.s['fit_attempts']}
    plan = uniform_edit_policy(['tp_random_conv'], 'wiring: the preset with random_conv disabled on every entity')
    scripts = {job + '_W': [[{'tool': 'overview', 'arguments': {}}],
                            [{'tool': 'build_material', 'arguments': {'plan_id': 'E_noconv', 'policy': plan}}],
                            [{'tool': 'inspect_material', 'arguments': {'plan_id': 'E_noconv', 'entity_index': 0}}],
                            [{'tool': 'evaluate', 'arguments': {'plan_id': 'E_noconv'}}],
                            [{'tool': 'compare', 'arguments': {'a': 'None', 'b': 'E_noconv'}}],
                            [{'tool': 'commit', 'arguments': {'plan_id': 'E_noconv', 'reason': 'wiring acceptance: scripted commit of the trained edit'}}]]}
    client = W.ScriptedClient(scripts)
    path = wroot / (job + '_W')
    before = led.s['fit_attempts']
    res = fast_branch_edit(path, job, br.Knowledge(), led, client, common, job_index=ji, public_ids=WIRING_PUBLIC, seeds=seeds)
    out['branch'] = {'status': res.status, 'failure_kind': res.failure_kind, 'reason': res.reason, 'committed': res.committed_plan_id, 'calls': res.calls, 'tool_calls': res.tool_calls,
                     'new_evaluations': res.new_evaluations, 'fits_charged': led.s['fit_attempts'] - before,
                     'events': [{k_: v for k_, v in e.items() if k_ in ('event', 'tool', 'error', 'plan_id')} for e in res.trace]}
    reg = W.load_registry(path / job)
    out['material'] = {k_: reg['E_noconv'].get(k_) for k_ in ('phys_id', 'kind', 'edit_profile')} if 'E_noconv' in reg else None
    branches, failures = [(path, job)], ([] if res.status == 'COMPLETE' else [{'branch': path.name, 'kind': res.failure_kind}])
    context.write_json(wroot / 'decisions_frozen.json', {'epoch': time.time(), 'branches': [str(p) for p, j in branches]})
    context.write_json(wroot / 'execution_finished.json', {'epoch': time.time(), 'branches': [(str(p), j) for p, j in branches], 'failures': failures})
    W.open_labels(wroot, cfg, branches, failures, led)
    cb = context.read_json(path / job / 'c_b_scores.json')['cells'] if (path / job / 'c_b_scores.json').exists() else {}
    out['c_b_cells_of_new_material'] = sorted(c for c in cb if '__E_noconv__' in c)
    out['e_files_absent'] = not (path / job / 'e_scores.json').exists() and not (path / job / 'e_frozen.json').exists()
    out['fits_charged_total'] = led.s['fit_attempts'] - out['fits_before']
    out['checks'] = {'branch_complete_with_edit_commit': res.status == 'COMPLETE' and res.committed_plan_id == 'E_noconv' and out['branch']['fits_charged'] == 3,
                     'material_is_recipe_edit': bool(out['material'] and out['material']['kind'] == 'tempo_v2_recipe_edit'),
                     'scorer_collected_new_material_c_b': len(out['c_b_cells_of_new_material']) == 3, 'e_never_read': out['e_files_absent'], 'fits_at_most_3': out['fits_charged_total'] <= 3}
    out['status'] = 'PASS' if all(out['checks'].values()) else 'FAIL'
    out['written_local'] = now()
    context.write_json(wroot / 'wiring.json', out)
    print('WIRING', out['status'], out['checks'], flush=True)
    return out


# ============================================================================= readout (E only after the whole stage; three layers)
def _cells_of(bdir: Path, job: str) -> dict:
    """{material_id: {seed: {'c_a', 'c_b', 'e'}}} of one branch view from its cells + label files."""
    out = {}
    for p in sorted((bdir / job / 'cells').glob('*.json')):
        r = context.read_json(p)
        if r.get('status') == 'OK' and not r.get('tag'):
            out.setdefault(r['material_id'], {})[r['model_seed']] = {'c_a': r['scores']['c_a']['normalized_mse_macro'], 'c_b': None, 'e': None, 'phys': r.get('physical_material', r['material_id']),
                                                                    'e_per_origin': None}
    for blk in ('c_b', 'e'):
        f = bdir / job / ('%s_scores.json' % blk)
        if f.exists():
            for v in context.read_json(f)['cells'].values():
                if v['material_id'] in out and v['model_seed'] in out[v['material_id']]:
                    out[v['material_id']][v['model_seed']][blk] = v[blk]['normalized_mse_macro'] if v[blk]['status'] == 'SCORABLE' else None
                    if blk == 'e' and v[blk]['status'] == 'SCORABLE':
                        out[v['material_id']][v['model_seed']]['e_per_origin'] = [statistics.fmean(row) for row in v[blk]['per_origin_entity_normalized_mse']]
    return out


def _vec(cells: dict, m: str, blk: str):
    v = [cells[m][s][blk] for s in SEEDS] if m in cells and all(s in cells[m] for s in SEEDS) else None
    return None if v is None or any(x is None for x in v) else v


def _paired(a: list, b: list, den: float) -> dict:
    d = [100 * (x - y) / den for x, y in zip(a, b)]
    se = statistics.stdev(d) / math.sqrt(len(d))
    return {'delta_pp': statistics.fmean(d), 'per_seed_pp': d, 'se_pp': se, 't95_pp': [statistics.fmean(d) - T975_DF2 * se, statistics.fmean(d) + T975_DF2 * se]}


def _argmin_ca(cells: dict, order: list) -> str:
    means = {m: statistics.fmean(_vec(cells, m, 'c_a')) for m in order if _vec(cells, m, 'c_a')}
    lo = min(means.values())
    return next(m for m in order if m in means and means[m] - lo <= TIE_ABS)


def readout(root: Path = ROOT) -> dict:
    sroot = paths(root)['pilot']
    res = {'package': PACKAGE, 'task': TASK, 'exposure': EXPOSURE, 'finished_local': now(), 'jobs': {}, 'stage_status': dsks.stage_status(sroot)}
    for job in JOBS:
        J = {'t': JOB_T[job], 'arms': {}, 'ablation': None, 'public': None}
        menu = _cells_of(sroot / (job + '_menu_ca'), job) if (sroot / (job + '_menu_ca') / job / 'e_scores.json').exists() else {}
        pub = {m: {'c_a': _vec(menu, m, 'c_a'), 'c_b': _vec(menu, m, 'c_b'), 'e': _vec(menu, m, 'e')} for m in PUBLIC} if menu else None
        if not pub or any(pub[m]['e'] is None for m in PUBLIC):
            J['status'] = 'INCOMPLETE_PUBLIC_E'
            res['jobs'][job] = J
            continue
        none_e = statistics.fmean(pub['None']['e'])
        nomix_e = pub[ta.RECIPE]['e']
        J['public'] = {m: {**pub[m], 'e_mean': statistics.fmean(pub[m]['e']), 'delta_vs_none_pp': 100 * (none_e - statistics.fmean(pub[m]['e'])) / none_e,
                           'delta_vs_nomix_pp': _paired(nomix_e, pub[m]['e'], none_e)['delta_pp'],
                           'e_per_origin': [statistics.fmean(x) for x in zip(*[menu[m][s]['e_per_origin'] for s in SEEDS])]} for m in PUBLIC}
        J['e_none_mean'] = none_e
        pub_best = min(statistics.fmean(pub[m]['e']) for m in PUBLIC)
        for arm in ARMS:
            bdir = sroot / ('%s_%s' % (job, arm))
            A = {'label': ARM_LABEL[arm], 'status': 'MISSING'}
            if not (bdir / 'branch_result.json').exists():
                J['arms'][arm] = A
                continue
            br_ = context.read_json(bdir / 'branch_result.json')
            cells = _cells_of(bdir, job)
            commit = context.read_json(bdir / job / 'commit.json') if (bdir / job / 'commit.json').exists() else None
            A.update({'status': br_['status'], 'failure_kind': br_.get('failure_kind'), 'calls': br_['calls'], 'tool_calls': br_['tool_calls'], 'new_evaluations': br_['new_evaluations'],
                      'commit': commit and commit['material_id'], 'commit_reason': commit and commit['reason'][:400], 'commit_is_public': bool(commit and commit['material_id'] in PUBLIC)})
            reg = W.load_registry(bdir / job)
            own = [m for m in cells if m not in PUBLIC]
            A['own_materials'] = {m: {'phys': cells[m][SEEDS[0]]['phys'], 'disabled_ops_per_program': (reg.get(m, {}).get('compiled') or {}).get('policy', {}).get('default', {}).get('steps'),
                                      'assignment_switches': sorted({json.dumps(s[0].get('disabled_ops', [])) if s and s[0]['op'] == ta.RECIPE_EDIT else ('preset' if s else 'none') for s in (reg.get(m, {}).get('assignment') or [])}),
                                      'n_rules': len((reg.get(m, {}).get('compiled') or {}).get('policy', {}).get('rules', [])),
                                      'c_a': _vec(cells, m, 'c_a'), 'c_b': _vec(cells, m, 'c_b'), 'e': _vec(cells, m, 'e'),
                                      'e_mean': statistics.fmean(_vec(cells, m, 'e')) if _vec(cells, m, 'e') else None,
                                      'delta_vs_nomix_pp': _paired(nomix_e, _vec(cells, m, 'e'), none_e)['delta_pp'] if _vec(cells, m, 'e') else None} for m in own}
            pool = [m for m in list(PUBLIC) + own if _vec(cells, m, 'e')]
            if commit and _vec(cells, commit['material_id'], 'e'):
                ce = _vec(cells, commit['material_id'], 'e')
                A['commit_e'] = ce
                A['commit_e_mean'] = statistics.fmean(ce)
                A['commit_e_per_origin'] = [statistics.fmean(x) for x in zip(*[cells[commit['material_id']][s]['e_per_origin'] for s in SEEDS])]
                A['commit_c_a'] = _vec(cells, commit['material_id'], 'c_a')
                A['commit_c_b'] = _vec(cells, commit['material_id'], 'c_b')
                A['vs'] = {m: _paired(pub[m]['e'], ce, none_e) for m in PUBLIC}
                pool_means = {m: statistics.fmean(_vec(cells, m, 'e')) for m in pool}
                best = min(pool, key=lambda m: (pool_means[m], pool.index(m)))
                A['pool'] = {'materials': pool, 'e_means': pool_means, 'best': best, 'best_is_public': best in PUBLIC,
                             'pool_gain_pp': 100 * (statistics.fmean(nomix_e) - pool_means[best]) / none_e,
                             'selection_loss_pp': 100 * (A['commit_e_mean'] - pool_means[best]) / none_e,
                             'net_pp': 100 * (statistics.fmean(nomix_e) - A['commit_e_mean']) / none_e,
                             'public_best_minus_pool_best_pp': 100 * (pub_best - pool_means[best]) / none_e,
                             'commit_rank_in_pool': sorted(pool, key=lambda m: (pool_means[m], pool.index(m))).index(commit['material_id']) + 1, 'pool_size': len(pool)}
                sh = _argmin_ca(cells, pool)
                A['shadow_ca_argmin'] = {'material': sh, 'e_mean': pool_means[sh], 'same_as_commit': sh == commit['material_id'], 'delta_commit_vs_shadow_pp': 100 * (statistics.fmean(_vec(cells, sh, 'e')) - A['commit_e_mean']) / none_e}
            J['arms'][arm] = A
        abl = sroot / (job + '_ablation')
        if (abl / job / 'e_scores.json').exists():
            cells = _cells_of(abl, job)
            reg = W.load_registry(abl / job)
            rows = {}
            for n in ABLATION:
                mid = 'ABL_' + n[3:]
                if mid not in cells:
                    continue
                s = context.read_json(reg[mid]['summary_path'])
                ed = s['edit']
                rows[n] = {'material': mid, 'phys': reg[mid]['phys_id'], 'c_a': _vec(cells, mid, 'c_a'), 'c_b': _vec(cells, mid, 'c_b'), 'e': _vec(cells, mid, 'e'),
                           'vs_nomix': {blk: _paired(pub[ta.RECIPE][blk], _vec(cells, mid, blk), statistics.fmean(pub['None'][blk])) for blk in ('c_a', 'c_b', 'e') if _vec(cells, mid, blk) and pub[ta.RECIPE][blk] and pub['None'][blk]},
                           'diag': {'rms_change_X': s['rms_change_X'], 'rms_change_y': s['rms_change_y'], 'fraction_changed_X': s['fraction_points_changed_X'], 'fraction_changed_y': s['fraction_points_changed_y'],
                                    'drawn_windows': ed['shadow_per_step'][n]['drawn'], 'skipped_windows': ed['shadow_per_step'][n]['skipped'], 'legal_windows': s['legal_parents'],
                                    'trigger_fraction': ed['shadow_per_step'][n]['skipped'] / s['legal_parents'], 'windows_no_retained_step': ed['windows_no_retained_step'],
                                    'windows_bitwise_unchanged': s['windows_bitwise_unchanged']}}
            J['ablation'] = rows
            emeans = {n: statistics.fmean(r['e']) for n, r in rows.items() if r['e']}
            J['ablation_best_e'] = min(emeans, key=lambda n: (emeans[n], ABLATION.index(n))) if emeans else None
        J['status'] = 'OK'
        res['jobs'][job] = J
    # aggregates (job = unit)
    ok = [j for j in JOBS if res['jobs'][j].get('status') == 'OK']
    agg = {'jobs': ok, 'n_jobs': len(ok), 'arms': {}, 'ablation': {}}
    for arm in ARMS:
        rows = [(j, res['jobs'][j]['arms'].get(arm)) for j in ok]
        rows = [(j, a) for j, a in rows if a and a.get('commit_e_mean') is not None]
        rec = {'n': len(rows), 'commits': {j: a['commit'] for j, a in rows}, 'commit_is_public': {j: a['commit_is_public'] for j, a in rows},
               'new_evaluations': {j: a['new_evaluations'] for j, a in rows}, 'calls': {j: a['calls'] for j, a in rows}, 'vs': {}}
        for m in PUBLIC:
            d = [a['vs'][m]['delta_pp'] for _, a in rows]
            if d:
                loo = {j: statistics.fmean(x for jj, x in zip([r for r, _ in rows], d) if jj != j) for j, _ in rows} if len(d) > 1 else {}
                rec['vs'][m] = {'mean_pp': statistics.fmean(d), 'per_job_pp': dict(zip([j for j, _ in rows], d)), 'wins': sum(x > 0 for x in d), 'ties': sum(x == 0 for x in d), 'losses': sum(x < 0 for x in d),
                                'max_harm_pp': min(d), 'loo_min_pp': min(loo.values()) if loo else None, 'loo_max_pp': max(loo.values()) if loo else None}
        if rows and all('pool' in a for _, a in rows):
            rec['decomposition_mean_pp'] = {k_: statistics.fmean(a['pool'][k_] for _, a in rows) for k_ in ('pool_gain_pp', 'selection_loss_pp', 'net_pp', 'public_best_minus_pool_best_pp')}
            rec['shadow_same_as_commit'] = {j: a['shadow_ca_argmin']['same_as_commit'] for j, a in rows}
            rec['commit_rank_in_pool'] = {j: '%d/%d' % (a['pool']['commit_rank_in_pool'], a['pool']['pool_size']) for j, a in rows}
        agg['arms'][arm] = rec
    for a, b in (('random_edit', 'f_edit'), ('menu_ca', 'f_edit'), ('menu_ca', 'random_edit')):
        rows = [j for j in ok if res['jobs'][j]['arms'].get(a, {}).get('commit_e') and res['jobs'][j]['arms'].get(b, {}).get('commit_e')]
        d = {j: _paired(res['jobs'][j]['arms'][a]['commit_e'], res['jobs'][j]['arms'][b]['commit_e'], res['jobs'][j]['e_none_mean']) for j in rows}
        vals = [x['delta_pp'] for x in d.values()]
        agg['%s->%s' % (a, b)] = {'n': len(rows), 'mean_pp': statistics.fmean(vals) if vals else None, 'per_job': {j: x['delta_pp'] for j, x in d.items()}, 'wins_B': sum(x > 0 for x in vals),
                                  'ties': sum(x == 0 for x in vals), 'losses_B': sum(x < 0 for x in vals), 'max_harm_to_B_pp': min(vals) if vals else None,
                                  'same_delivery': {j: res['jobs'][j]['arms'][a]['commit'] == res['jobs'][j]['arms'][b]['commit'] for j in rows}}
    for n in ABLATION:
        rows = [(j, res['jobs'][j]['ablation'][n]) for j in ok if res['jobs'][j].get('ablation') and n in res['jobs'][j]['ablation']]
        rec = {'n': len(rows)}
        for blk in ('c_a', 'c_b', 'e'):
            d = [r['vs_nomix'][blk]['delta_pp'] for _, r in rows if blk in r['vs_nomix']]
            if d:
                rec[blk] = {'mean_pp': statistics.fmean(d), 'per_job_pp': {j: r['vs_nomix'][blk]['delta_pp'] for j, r in rows if blk in r['vs_nomix']}, 'wins': sum(x > 0 for x in d), 'losses': sum(x < 0 for x in d),
                            'max_harm_pp': min(d), 'all_same_sign': all(x > 0 for x in d) or all(x < 0 for x in d)}
        rec['trigger_fraction_mean'] = statistics.fmean(r['diag']['trigger_fraction'] for _, r in rows) if rows else None
        rec['rms_change_X_mean'] = statistics.fmean(r['diag']['rms_change_X'] for _, r in rows) if rows else None
        rec['rms_change_y_mean'] = statistics.fmean(r['diag']['rms_change_y'] for _, r in rows) if rows else None
        agg['ablation'][n] = rec
    res['aggregate'] = agg
    res['cost'] = cost_readout(root)
    context.write_json(root / 'result.json', res)
    (root / 'tables.md').write_text(tables(res), encoding='utf-8')
    write_report(root, res)
    write_method(root)
    return res


def cost_readout(root: Path) -> dict:
    led = context.read_json(paths(root)['ledger'])
    sroot = paths(root)['pilot']
    units = dsks._unit_tokens(sroot) if (sroot / 'raw_responses').exists() else {}
    return {'fit_attempts': led['fit_attempts'], 'fits_ok': led['fits_ok'], 'fits_failed': led['fits_failed'], 'cache_hits': led['cache_hits'], 'retries_used': led['retries_used'],
            'fit_wall_seconds': led['fit_wall_seconds'], 'label_wall_seconds': led.get('label_wall_seconds', 0.0), 'material_wall_seconds': led.get('material_wall_seconds', 0.0),
            'llm_requests': led['llm_requests'], 'llm_http_attempts': led['llm_http_attempts'], 'llm_tokens_in': led['llm_tokens_in'], 'llm_tokens_out': led['llm_tokens_out'],
            'llm_tokens_unknown': led['llm_tokens_unknown'], 'llm_failed_attempts': led.get('llm_failed_attempts', 0), 'unknown_usage_accepted': led.get('unknown_usage_accepted', 0),
            'elapsed_s_since_ledger': time.time() - led['started_epoch'], 'caps': led['caps'], 'per_unit_tokens': units, 'slow_requests': 0, 'new_sha': 0, 'git_commits': 0}


def _f(x, nd=4):
    return '—' if x is None else ('%.*f' % (nd, x))


def _p(x):
    return '—' if x is None else ('%+.2f' % x)


def tables(res: dict) -> str:
    L = ['# %s — tables' % PACKAGE, '', 'E = missing-aware normalized MSE macro (raw, 3-seed mean); Δ in pp of the job\'s None mean E (> 0 = second better). Job = unit.', '']
    L += ['## A. Public references per job (E mean; Δ vs None; Δ vs NoMix)', '', '| Job | E(None) | ' + ' | '.join(PUBLIC[1:]) + ' |', '|---|---:|' + '---:|' * 3]
    for j, J in res['jobs'].items():
        if J.get('status') != 'OK':
            L.append('| %s | %s |' % (j, J.get('status')))
            continue
        L.append('| %s | %s | %s |' % (j.replace('RD02_', ''), _f(J['e_none_mean']), ' | '.join('%s (%s vs None, %s vs NoMix)' % (_f(J['public'][m]['e_mean']), _p(J['public'][m]['delta_vs_none_pp']), _p(J['public'][m]['delta_vs_nomix_pp'])) for m in PUBLIC[1:])))
    L += ['', '## B. Arms per job: commit, E, Δ vs references, pool decomposition, shadow C_A argmin', '',
          '| Job | arm | status | calls / tools / new evals | commit | public? | E mean | Δ vs NoMix | Δ vs None | Δ vs FixedMixup | Δ vs AmpResample | pool best | pool gain | selection loss | net | public best − pool best | shadow C_A argmin (E) |',
          '|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|']
    for j, J in res['jobs'].items():
        if J.get('status') != 'OK':
            continue
        for arm in ARMS:
            A = J['arms'].get(arm) or {}
            if A.get('commit_e_mean') is None:
                L.append('| %s | %s | %s | | %s | | | | | | | | | | | | |' % (j.replace('RD02_', ''), ARM_LABEL[arm], A.get('status'), A.get('commit')))
                continue
            P_ = A.get('pool', {})
            sh = A.get('shadow_ca_argmin', {})
            L.append('| %s | %s | %s | %s / %s / %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s (%s) |' % (
                j.replace('RD02_', ''), ARM_LABEL[arm], A['status'], A['calls'], A['tool_calls'], A['new_evaluations'], A['commit'], 'yes' if A['commit_is_public'] else 'NEW EDIT', _f(A['commit_e_mean']),
                _p(A['vs'][ta.RECIPE]['delta_pp']), _p(A['vs']['None']['delta_pp']), _p(A['vs']['FixedMixup']['delta_pp']), _p(A['vs']['P_AmpResample']['delta_pp']),
                P_.get('best'), _p(P_.get('pool_gain_pp')), _p(P_.get('selection_loss_pp')), _p(P_.get('net_pp')), _p(P_.get('public_best_minus_pool_best_pp')), sh.get('material'), _f(sh.get('e_mean'))))
    L += ['', '## C. Own materials of the research arms (what was built; E and Δ vs NoMix)', '', '| Job | arm | plan | physical | switches in assignment | rules | C_A mean | C_B mean | E mean | Δ vs NoMix |', '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for j, J in res['jobs'].items():
        if J.get('status') != 'OK':
            continue
        for arm in ('f_edit', 'random_edit'):
            for m, r in (J['arms'].get(arm) or {}).get('own_materials', {}).items():
                L.append('| %s | %s | %s | %s | %s | %d | %s | %s | %s | %s |' % (j.replace('RD02_', ''), ARM_LABEL[arm], m, r['phys'], ', '.join(r['assignment_switches']), r['n_rules'],
                                                                                 _f(statistics.fmean(r['c_a']) if r['c_a'] else None), _f(statistics.fmean(r['c_b']) if r['c_b'] else None), _f(r['e_mean']), _p(r['delta_vs_nomix_pp'])))
    L += ['', '## D. Seven single-component ablations vs the preset (Δ pp of None; > 0 = ablation better), material change and trigger rate', '',
          '| Job | component disabled | C_A Δ | C_B Δ | E Δ [t95] | E mean | rms X | rms y | drawn / skipped / legal | no-step windows |', '|---|---|---:|---:|---|---:|---:|---:|---|---:|']
    for j, J in res['jobs'].items():
        if J.get('status') != 'OK' or not J.get('ablation'):
            continue
        for n, r in J['ablation'].items():
            v = r['vs_nomix']
            L.append('| %s | %s | %s | %s | %s [%s, %s] | %s | %s | %s | %d / %d / %d | %d |' % (j.replace('RD02_', ''), n, _p(v.get('c_a', {}).get('delta_pp')), _p(v.get('c_b', {}).get('delta_pp')), _p(v.get('e', {}).get('delta_pp')),
                                                                                              _p((v.get('e') or {}).get('t95_pp', [None, None])[0]), _p((v.get('e') or {}).get('t95_pp', [None, None])[1]), _f(statistics.fmean(r['e']) if r['e'] else None),
                                                                                              _f(r['diag']['rms_change_X'], 3), _f(r['diag']['rms_change_y'], 3), r['diag']['drawn_windows'], r['diag']['skipped_windows'], r['diag']['legal_windows'], r['diag']['windows_no_retained_step']))
    L += ['', '## E. Aggregates (job = unit)', '']
    for arm, rec in res['aggregate']['arms'].items():
        L.append('- **%s** (n=%d): commits %s; ' % (ARM_LABEL[arm], rec['n'], rec['commits']) + '; '.join('vs %s %s pp (W/T/L %d/%d/%d, max harm %s, LOO [%s, %s])' % (m, _p(v['mean_pp']), v['wins'], v['ties'], v['losses'], _p(v['max_harm_pp']), _p(v['loo_min_pp']), _p(v['loo_max_pp'])) for m, v in rec['vs'].items())
                 + ('; decomposition mean %s' % {k_: round(v, 2) for k_, v in rec['decomposition_mean_pp'].items()} if rec.get('decomposition_mean_pp') else ''))
    for k_ in ('random_edit->f_edit', 'menu_ca->f_edit', 'menu_ca->random_edit'):
        v = res['aggregate'].get(k_)
        if v:
            L.append('- **%s**: mean %s pp, W/T/L (B) %d/%d/%d, max harm %s, same delivery %s' % (k_, _p(v['mean_pp']), v['wins_B'], v['ties'], v['losses_B'], _p(v['max_harm_to_B_pp']), v['same_delivery']))
    L += ['', '| ablation | n | C_A mean Δ | C_B mean Δ | E mean Δ | E W/L | E max harm | same sign | trigger fraction |', '|---|---:|---:|---:|---:|---|---:|---|---:|']
    for n, rec in res['aggregate']['ablation'].items():
        e = rec.get('e', {})
        L.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s |' % (n, rec['n'], _p(rec.get('c_a', {}).get('mean_pp')), _p(rec.get('c_b', {}).get('mean_pp')), _p(e.get('mean_pp')), '%s/%s' % (e.get('wins', '—'), e.get('losses', '—')), _p(e.get('max_harm_pp')), e.get('all_same_sign'), _f(rec['trigger_fraction_mean'], 3)))
    L += ['', '## F. Per-origin E of each arm\'s commit (3-seed mean, macro over entities)', '', '| Job | origins | None | NoMix | ' + ' | '.join(ARM_LABEL[a] for a in ARMS) + ' |', '|---|---|---|---|' + '---|' * len(ARMS)]
    for j, J in res['jobs'].items():
        if J.get('status') != 'OK':
            continue
        L.append('| %s | %s | %s | %s | %s |' % (j.replace('RD02_', ''), '/'.join(str(o) for o in rd.resolve_job(DATASET, j).e), '/'.join(_f(x, 3) for x in J['public']['None']['e_per_origin']), '/'.join(_f(x, 3) for x in J['public'][ta.RECIPE]['e_per_origin']),
                                                 ' | '.join('/'.join(_f(x, 3) for x in (J['arms'].get(a) or {}).get('commit_e_per_origin') or []) or '—' for a in ARMS)))
    return '\n'.join(L) + '\n'


def write_report(root: Path, res: dict) -> None:
    A, C = res['aggregate'], res['cost']
    fe, re_, mc = A['arms'].get('f_edit', {}), A['arms'].get('random_edit', {}), A['arms'].get('menu_ca', {})
    ok = A['jobs']
    abl = A['ablation']
    best_abl = sorted(((n, r['e']['mean_pp']) for n, r in abl.items() if r.get('e')), key=lambda x: -x[1])
    q1 = ('七份单组件消融（%d 批）对原 NoMixRecipe 的 E 配对差（pp of None，>0 = 移除该组件更好）：%s。逐批同号的组件：%s。C_A/C_B 同向情况见 tables §D/§E。'
          % (A['n_jobs'], '; '.join('%s %s (W/L %d/%d, 最大伤害 %s)' % (n, _p(v), abl[n]['e']['wins'], abl[n]['e']['losses'], _p(abl[n]['e']['max_harm_pp'])) for n, v in best_abl),
             ', '.join(n for n, r in abl.items() if r.get('e') and r['e']['all_same_sign']) or '无'))
    new_commits = {j: m for j, m in fe.get('commits', {}).items() if not fe['commit_is_public'][j]}
    q2 = ('F_edit 在 %d 批的交付：%s（其中新编辑 %d 批：%s；其余为公共参照）。对原 NoMix %s pp（W/T/L %d/%d/%d，最大伤害 %s，LOO [%s, %s]）；对 Random_edit_B4 %s pp（W/T/L %d/%d/%d）；对 Menu_CA %s pp（W/T/L %d/%d/%d）；对 None %s pp。Random_edit_B4 对 NoMix %s pp（交付 %s）；Menu_CA 对 NoMix %s pp（交付 %s）。'
          % (fe.get('n', 0), fe.get('commits'), len(new_commits), new_commits or '无',
             _p(fe.get('vs', {}).get(ta.RECIPE, {}).get('mean_pp')), *(fe.get('vs', {}).get(ta.RECIPE, {}).get(k_, 0) for k_ in ('wins', 'ties', 'losses')), _p(fe.get('vs', {}).get(ta.RECIPE, {}).get('max_harm_pp')),
             _p(fe.get('vs', {}).get(ta.RECIPE, {}).get('loo_min_pp')), _p(fe.get('vs', {}).get(ta.RECIPE, {}).get('loo_max_pp')),
             _p(A.get('random_edit->f_edit', {}).get('mean_pp')), A.get('random_edit->f_edit', {}).get('wins_B', 0), A.get('random_edit->f_edit', {}).get('ties', 0), A.get('random_edit->f_edit', {}).get('losses_B', 0),
             _p(A.get('menu_ca->f_edit', {}).get('mean_pp')), A.get('menu_ca->f_edit', {}).get('wins_B', 0), A.get('menu_ca->f_edit', {}).get('ties', 0), A.get('menu_ca->f_edit', {}).get('losses_B', 0),
             _p(fe.get('vs', {}).get('None', {}).get('mean_pp')), _p(re_.get('vs', {}).get(ta.RECIPE, {}).get('mean_pp')), re_.get('commits'), _p(mc.get('vs', {}).get(ta.RECIPE, {}).get('mean_pp')), mc.get('commits')))
    dec = fe.get('decomposition_mean_pp') or {}
    q3 = ('F_edit 自身候选池（公共参照 + 本臂真实评估项）：池内可用收益均值 %s pp、选择损失均值 %s pp、实际净收益 %s pp（= NoMix − commit）；公共池最优 − 加入本臂新编辑后的池最优 = %s pp（0 = 新编辑没有增加机会）。commit 在池内 E 排名：%s。影子 C_A argmin 与真实 commit 相同：%s。Random_edit_B4：池内可用收益 %s、选择损失 %s、净 %s pp。'
          % (_p(dec.get('pool_gain_pp')), _p(dec.get('selection_loss_pp')), _p(dec.get('net_pp')), _p(dec.get('public_best_minus_pool_best_pp')), fe.get('commit_rank_in_pool'), fe.get('shadow_same_as_commit'),
             _p((re_.get('decomposition_mean_pp') or {}).get('pool_gain_pp')), _p((re_.get('decomposition_mean_pp') or {}).get('selection_loss_pp')), _p((re_.get('decomposition_mean_pp') or {}).get('net_pp'))))
    L = ['# %s — REPORT' % PACKAGE, '', '任务书：`%s`。身份：`%s`（四批的公共参照 E 已被研究者看过；新材料此前未评估）。完成：%s。阶段状态：%s。' % (TASK, EXPOSURE, res['finished_local'], res['stage_status'].get('status')), '',
         '## 0. §1 三问', '', '1. **原配方是否存在可改进的组件？** ' + q1, '', '2. **Fast 构造并交付的局部修改是否胜过原配方和同预算随机局部修改？** ' + q2, '', '3. **构造收益与选择损失** ' + q3, '',
         '## 1. 费用与合规', '',
         '- 物理拟合尝试 %d（成功 %d / 失败 %d / 重试 %d），缓存命中 %d；拟合墙钟 %.0f s，标签 %.0f s，材料 %.0f s。' % (C['fit_attempts'], C['fits_ok'], C['fits_failed'], C['retries_used'], C['cache_hits'], C['fit_wall_seconds'], C['label_wall_seconds'], C['material_wall_seconds']),
         '- Fast 请求 %d（HTTP 尝试 %d，失败 %d），token in/out %d/%d，未知用量 %d（已接受 %d）；Slow 0；新增 SHA 0；git commit 0。' % (C['llm_requests'], C['llm_http_attempts'], C['llm_failed_attempts'], C['llm_tokens_in'], C['llm_tokens_out'], C['llm_tokens_unknown'], C['unknown_usage_accepted']),
         '- 帽：拟合 %d、请求 %d、token %d、墙钟 %d s；账本自建立以来 %.0f s。' % (C['caps']['max_fit_attempts'], C['caps']['max_llm_requests'], C['caps']['max_llm_tokens'], C['caps']['max_wall_s'], C['elapsed_s_since_ledger']),
         '- 阶段失败记录：%s。' % (res['stage_status'].get('failures') or '无'), '',
         '## 2. 交付与池（逐批）', '', '| Job | F_edit commit | 新编辑? | Δ vs NoMix | Δ vs Random | Δ vs Menu | Random commit | Menu commit | F 池最优 | F 选择损失 | 影子 C_A argmin |', '|---|---|---|---:|---:|---:|---|---|---|---:|---|']
    for j in ok:
        J = res['jobs'][j]
        f_, r_, m_ = J['arms'].get('f_edit', {}), J['arms'].get('random_edit', {}), J['arms'].get('menu_ca', {})
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (j.replace('RD02_', ''), f_.get('commit'), 'NEW' if f_.get('commit') and not f_.get('commit_is_public') else 'public', _p((f_.get('vs') or {}).get(ta.RECIPE, {}).get('delta_pp')),
                                                                          _p(A.get('random_edit->f_edit', {}).get('per_job', {}).get(j)), _p(A.get('menu_ca->f_edit', {}).get('per_job', {}).get(j)), r_.get('commit'), m_.get('commit'),
                                                                          (f_.get('pool') or {}).get('best'), _p((f_.get('pool') or {}).get('selection_loss_pp')), (f_.get('shadow_ca_argmin') or {}).get('material')))
    L += ['', '## 3. 七消融汇总（Δ pp of None vs NoMix，> 0 = 移除更好）', '', '| component | C_A | C_B | E | E W/L | max harm | same sign | trigger |', '|---|---:|---:|---:|---|---:|---|---:|']
    for n, rec in abl.items():
        e = rec.get('e', {})
        L.append('| %s | %s | %s | %s | %s/%s | %s | %s | %s |' % (n, _p(rec.get('c_a', {}).get('mean_pp')), _p(rec.get('c_b', {}).get('mean_pp')), _p(e.get('mean_pp')), e.get('wins', '—'), e.get('losses', '—'), _p(e.get('max_harm_pp')), e.get('all_same_sign'), _f(rec['trigger_fraction_mean'], 3)))
    L += ['', '## 4. 边界', '',
          '- 所有结论限于当前固定增强随机实现（每实体一条预设父流）与 3 个 Consumer seed；未重抽增强随机种子。四批是时间重复单位，不是 12 个独立任务。',
          '- 消融差是"该组件在完整配方背景下的移除效应"，组件之间有交互，不可加总；七项最优者不以单项区间宣称显著。',
          '- oracle / 池最优带事后选择偏差，只作诊断；F_edit 的池只含公共参照与该臂真实评估项，不跨臂合并。',
          '- 本包不启动 Slow、不形成 Skill、不改概率/强度、不追加 seed、不解封新数据。',
          '', '详细表：`tables.md`；方法：`METHOD.md`；机器可读：`result.json`、`frozen_config.json`、`budget.json`、`smoke/`、`wiring/`、`pilot/`。']
    (root / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def write_method(root: Path) -> None:
    L = ['# %s — METHOD' % PACKAGE, '',
         '## 1. 新增构造能力：配方组件开关 `%s`' % ta.RECIPE_EDIT, '',
         '- 程序形式：`{"op": "%s", "disabled_ops": [...]}`，`disabled_ops` 只允许七个 `tp_*` 原语名，去重后按原 PRIMITIVES 顺序规范化，长度 0/1/2；长度 0 规范化为 `{"op": "%s"}`（语义别名，无新材料）。' % (ta.RECIPE_EDIT, ta.RECIPE),
         '- 语义：先按原配方抽取（类别 2/4、类别内二选一、卷积 p=0.3），再跳过被关闭的原语；不补抽、不重新归一化权重、不加最小改动过滤；窗口可能只剩一步或成为 identity（如实记录为 `windows_no_retained_step` / `windows_bitwise_unchanged`）。',
         '- 参数强度、类别权重、抽取数量、内部二选一概率、卷积概率、执行顺序均不可编辑（DSL 拒绝任何其他键）。',
         '- 29 个局部程序：原版 1 + 单关闭 7（PRIMITIVES 顺序）+ 双关闭 21（combinations 顺序），只用于枚举/缓存；旧 53 程序编号不变。',
         '- 完整计划仍为 default + 最多 8 条 T 谓词规则；本包 Fast/Random 的新材料限于此空间，公共参照可直接提交；显式原语组合接口在仓库保留，本包不提供。', '',
         '## 2. 随机耦合：父流与被编辑材料的隔离', '',
         '- 父流：每个实体的 `entity_program_seed(job_index, 52, entity)`（与现役 v2 原配方材料相同）；开关不参与 seed。训练 seed 与材料随机性分离（`Streams` 在每次调用前后换入/换出全局 torch/numpy 状态）。',
         '- shadow 路径：`recipe_plan_and_apply` 以 observer 逐窗口在原窗口副本上运行原配方，推进实体流，并在每个操作入口记录三种 RNG 状态（numpy Generator / 全局 torch / 全局 legacy numpy）。',
         '- 编辑路径：按原抽样顺序，对保留操作从记录的状态构造分离的 `Streams.from_snapshot` 重放到真实窗口；被关闭操作直接跳过；重放只消耗分离副本，不改变 shadow 的后续状态。',
         '- 保留操作的随机抽取与原配方对齐，但依赖输入的量（std、均值、min/max 缩放等）按新输入由源函数重算；不复制中间张量。',
         '- 不持久化 RNG 快照，逐窗口临时释放；材料记录只保存开关、父 seed、构造时间与诊断计数。', '',
         '## 3. 核对（smoke / wiring）', '',
         '- 空开关经重放路径与缓存的 v2 原配方材料逐位一致（全部合法窗口）；样例含 calendar、censor no-op、conv 命中。',
         '- 关闭组件不挪动其余窗口/操作的抽样（shadow 步序列逐窗口相同）；未命中窗口逐位等于原配方；全局 torch/numpy RNG 构造前后相同。',
         '- 条件化计划的未修改实体等于原材料、修改实体等于统一开关材料；不同构造顺序的两份缓存逐位相同。',
         '- 键语义：29 程序键互异，原版键等于公共 P_NoMixRecipe 键；分支视图互不可见；非法参数（未知组件、3 个开关、强度参数、显式组合、空 steps、重复预设）返回可纠错的工具错误。',
         '- 接线：RD02_T1 上脚本客户端 构造→评估（3 拟合）→提交，C_B scorer 收集新材料的 3 个 cell；不读取 E。', '',
         '## 4. 对照与评价', '',
         '- F_edit：父包七工具 + run_job，4 新方案 / 16 请求 / 24 工具 / 400k token，模型 cpa-grok-4.6（返回 grok-4.6-build），temperature 0；无 Skill/Generic/跨 Job 输入。',
         '- Random_edit_B4：父包 draw_random_supply 结构，PROGRAMS 换为 29 局部程序（R1/R2 从 28 个非原版均匀不放回；R3/R4 条件化，两组非空，256 次尝试后回退），交付 = 四公共 + 四私有中 C_A 均值最低。',
         '- Menu_CA：四公共参照中按 C_A 均值选一份。七消融：所有臂 commit 冻结后构造/拟合，分数不进入任何臂。',
         '- 全部输出冻结 → 统一 C_B → 冻结 E 预测 → 屏障 → 统一 E。Δ_j(A,B) = 100 (E(A) − E(B)) / E(None)；配对 seed 向量同分母；SE 与 t95(df=2) 同向量。',
         '- 三层读数：材料是否更好（消融、各臂自身池）、是否交付（commit）、选择损失（池最优 − commit）；影子 C_A argmin 只报不写回。']
    (root / 'METHOD.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


# ============================================================================= package driver
def package_run(root: Path = ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    status = context.read_json(root / 'package_status.json') if (root / 'package_status.json').exists() else {}

    def stop(where, st):
        status.update({'stopped_at': where, 'stage_status': st, 'epoch': time.time(), 'local': now()})
        context.write_json(root / 'package_status.json', status)
        print('PACKAGE_STOP', where, st.get('status') if isinstance(st, dict) else st, flush=True)
        return status
    preflight(root)
    sm = smoke(root)
    status['smoke'] = sm['all_pass']
    w = wiring(root)
    status['wiring'] = w['status']
    if w['status'] != 'PASS':
        return stop('wiring', w)
    cfgp = stage_config(root)
    st = run_stage(root, cfgp)
    status['pilot'] = st
    if st['status'] != 'FINISHED':
        return stop('pilot', st)
    status['finished_epoch'] = time.time()
    status['finished_local'] = now()
    context.write_json(root / 'package_status.json', status)
    readout(root)
    print('PACKAGE_FINISHED', flush=True)
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--resume-stage', metavar='STAGE')
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--restart', default='')
    ap.add_argument('--stage-worker', metavar='CFG')
    ap.add_argument('--worker-material', nargs=4, metavar=('ROOT', 'JOB', 'PHYS', 'JOB_INDEX'))
    ap.add_argument('--worker-smoke', metavar='OUT')
    a = ap.parse_args()
    if a.worker_material:
        worker_material(*a.worker_material)
        return
    if a.worker_smoke:
        worker_smoke(a.worker_smoke)
        return
    if a.stage_worker:
        stage_worker(Path(a.stage_worker))
        return
    assert 'torch' not in sys.modules, 'controller must not import torch'
    if a.preflight:
        preflight()
    elif a.smoke:
        preflight()
        smoke()
    elif a.wiring:
        wiring()
    elif a.resume_stage:
        if a.resume_stage != 'pilot':
            raise SystemExit('only the pilot stage can be resumed')
        resume_stage(ROOT, accept_unknown_usage=a.accept_unknown_usage, restart=tuple(x for x in a.restart.split(',') if x))
        if dsks.stage_status(paths(ROOT)['pilot'])['status'] == 'FINISHED':
            readout(ROOT)
    elif a.run:
        package_run()
    elif a.result:
        readout()
    assert 'torch' not in sys.modules, 'controller imported torch'


if __name__ == '__main__':
    main()
