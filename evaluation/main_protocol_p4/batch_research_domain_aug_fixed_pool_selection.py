"""DEV-DOMAIN-AUG-FIXED-POOL-SELECTION (docs/DEV_DOMAIN_AUG_FIXED_POOL_SELECTION_TASK_2026-09-21.md): with the candidate pool of every
case FIXED to what its original F0 trajectory actually evaluated (four public references + its own evaluated plans), learn a per-domain
SUBMISSION decision Skill from the eight Learn cases (S01-S08 of the entity-split package), pick one card per domain on the four Select
cases (Q01/Q02 of the entity-split package), then replay the eight Q03-Q06 cases of the decision-priority package with S0 (no card),
S_W* (the frozen card), R_CA (C_A argmin), Fixed_dev (the public program chosen on Select) and R_uniform (the exact expectation of a
uniform pick) - all on the identical pool and the identical legal feedback view. 0 fits, 0 predictions, 0 labels, 0 hashes, 0 commits.

Reused unchanged (read-only imports): the decision-priority package (DP) for PackageLock / SafeLedger / Pools / MeteredClient / receipts /
text checks; the entity-split package (ES) for the model identity, field definitions, action table, score helpers; entity_case (ec) for
registries, labels and keys; domain_skill (dsk) for the Skill carrier; batch_research (br) for the guidance rendering (the same
`guidance` block a Fast request carries).

  --preflight | --smoke | --wiring | --run [--accept-unknown-usage] | --result | --monitor [--interval S] [--once]
  subprocess entry: --lock-probe ROOT
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from pathlib import Path

from methods.ttha import batch_research as br
from methods.ttha import domain_skill as dsk
from methods.ttha.batch_base import budget, context, spec, tempo_aug as ta
from methods.ttha.batch_base import entity_case as ec
from evaluation.main_protocol_p4 import batch_research_domain_skill as dsks
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_skill as W
from evaluation.main_protocol_p4 import batch_research_tempo_aug_workflow_learning_loop as LL
from evaluation.main_protocol_p4 import batch_research_domain_aug_entity_split as ES
from evaluation.main_protocol_p4 import batch_research_domain_aug_decision_priority as DP

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch' / 'dev_domain_aug_fixed_pool_selection'
MODULE = 'evaluation.main_protocol_p4.batch_research_domain_aug_fixed_pool_selection'
TASK = 'docs/DEV_DOMAIN_AUG_FIXED_POOL_SELECTION_TASK_2026-09-21.md'
PACKAGE = 'DEV-DOMAIN-AUG-FIXED-POOL-SELECTION'
EXPOSURE = 'EXPOSED_DEVELOPMENT_REPLAY'
DOMAINS = ES.DOMAINS
SEEDS = ES.SEEDS
PUBLIC = ec.PUBLIC
PUBLIC_STEPS = ec.PUBLIC_STEPS
MODEL = ES.MODEL
MAX_OUTPUT_TOKENS = 8_000                                   # task §9: per-request output cap
BODY_LIMIT = ES.BODY_LIMIT                                  # 6000 characters per card
TOL = 1e-12
T975_DF2 = ES.T975_DF2
PROPOSALS = ('N1', 'N2', 'N3')
CARD_ARMS = ('n1', 'n2', 'n3')
ROLES = {'learn': (ES.ROOT / 'source', tuple('S%02d' % i for i in range(1, 9)), 'docs/DOMAIN_AUG_ENTITY_SPLIT_V1.json'),
         'select': (ES.ROOT / 'metatest', ('Q01', 'Q02'), 'docs/DOMAIN_AUG_ENTITY_SPLIT_V1.json'),
         'replay': (DP.ROOT / 'metatest', ('Q03', 'Q04', 'Q05', 'Q06'), 'docs/DOMAIN_AUG_DECISION_PRIORITY_V1.json')}
STAGE_OF_ROLE = {'select': 'select', 'replay': 'replay'}
TOTAL = {'max_fit_attempts': 0, 'max_llm_requests': 76, 'max_llm_tokens': 60_000_000, 'max_wall_s': 48 * 3600, 'max_retries': 0}   # tokens / wall: warnings only (task §9)
HTTP_CAP = 80
TRANSPORT_RETRIES = 4                                       # package-wide, at most one per logical request (inside DP.MeteredClient)
ALLOC = {'slow': {'requests': 12}, 'select': {'requests': 32}, 'replay': {'requests': 32}}      # main + one format correction each
CONCURRENCY = {'http': 4, 'numeric': 1}
TOKEN_WARNINGS = (2_000_000, 4_000_000)
PAID_WALL_WARNINGS_S = (3600, 7200)
SLOW_INPUT_TOKEN_TARGET = 60_000
SLOW_BYTES_PER_TOKEN = 2.6                                  # the parent's measured envelope (495 KB -> 191k tokens)
SLOW_REQUEST_TIMEOUT_S = 600.                               # transport parameter only (the parent's 300 s timed out once with four 200k-token requests in flight)
DECIDE_REQUEST_TIMEOUT_S = 300.
WIRING_CASES = ('D01_S01', 'D02_S01')
FIXED_ORDER = ('None', 'FixedMixup', 'P_AmpResample', 'P_NoMixRecipe')
H_ORDER = ('fixed_dev', 'r_ca', 's0', 's_w')
FIXED_MIXUP_KEY = 'FixedMixup:R:w0.25:aug_seed'
CID_TOKEN = re.compile(r'(?<![A-Za-z0-9])C\d{2}(?![A-Za-z0-9])')
SOURCE_NAMES = tuple(spec.DATASETS)
FORBIDDEN_PACKET_KEYS = ('c_b', 'e', 'e_scores', 'c_b_scores', 'commit', 'rationale', 'reason', 'committed', 'phys_id', 'physical_material', 'material_key', 'path', 'model_path',
                         'summary_path', 'plan_id', 'job_id', 'case_id', 'roster', 'arm', 'branch', 'delivery_seed', 'evaluated_by')
DISPLAY_SIG = 8


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def paths(root: Path) -> dict:
    root = Path(root)
    return {'ledger': root / 'budget.json', 'stages': root / 'stages.json', 'logs': root / 'logs', 'clean': root / 'clean_inputs', 'evaluator': root / 'evaluator',
            'evidence': root / 'evidence', 'formation': root / 'formation', 'select': root / 'select', 'replay': root / 'replay', 'freeze': root / 'freeze',
            'warnings': root / 'warnings', 'wiring': root / 'wiring', 'smoke': root / 'smoke', 'lock': root / 'package.lock'}


def sig(x, n: int = DISPLAY_SIG):
    """Round floats to n significant digits for the model-facing display copy (the evaluator keeps full precision)."""
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        return x if not math.isfinite(x) else float('%.*g' % (n, x))
    if isinstance(x, dict):
        return {k: sig(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sig(v, n) for v in x]
    return x


def fmean(v):
    return statistics.fmean(v)


def se3(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None


def stats(v: list) -> dict:
    return ES._stats(list(v))


# ============================================================================= A. pool binding (0 LLM)
def case_dirs(role: str, case: str) -> tuple:
    root, _, _ = ROLES[role]
    return root / ('%s_f0' % case) / case, root / ('%s_common' % case) / case, root / ('%s_f0' % case)


def all_cases(role: str, domain: str | None = None) -> list:
    _, tails, _ = ROLES[role]
    return ['%s_%s' % (d, t) for d in DOMAINS for t in tails if domain is None or d == domain]


def semantic_key(mid: str, rec: dict) -> str:
    if mid == 'FixedMixup':
        return FIXED_MIXUP_KEY
    if mid == 'None':
        return ec.material_key([[] for _ in range(ec.COHORT_SIZE)])
    if mid in PUBLIC_STEPS:
        return ec.material_key([PUBLIC_STEPS[mid]] * ec.COHORT_SIZE)
    return rec['key']


def _entity_matrix(cell: dict, block: str) -> list:
    return [list(map(float, row)) for row in cell['scores'][block]['per_origin_entity_normalized_mse']]


def _scores_from(src: dict, mid: str, block: str) -> dict | None:
    vals = {}
    for c, v in src.items():
        if v['material_id'] == mid and block in v:
            vals[v['model_seed']] = v[block]
    if set(vals) < set(SEEDS) or any(vals[s]['status'] != 'SCORABLE' for s in SEEDS):
        return None
    mats = [[list(map(float, row)) for row in vals[s]['per_origin_entity_normalized_mse']] for s in SEEDS]
    return _block_summary(mats)


def _block_summary(mats: list) -> dict:
    """mats: per seed, per origin, per entity losses -> macro by seed, by seed x origin, by origin, per-entity mean (over seeds and origins)."""
    n_o, n_e = len(mats[0]), len(mats[0][0])
    by_seed_origin = [[fmean(row) for row in m] for m in mats]
    by_seed = [fmean(x) for x in by_seed_origin]
    by_origin = [fmean(by_seed_origin[s][o] for s in range(len(mats))) for o in range(n_o)]
    per_entity = [fmean(mats[s][o][e] for s in range(len(mats)) for o in range(n_o)) for e in range(n_e)]
    per_entity_by_origin = [[fmean(mats[s][o][e] for s in range(len(mats))) for e in range(n_e)] for o in range(n_o)]
    return {'by_seed': by_seed, 'by_seed_origin': by_seed_origin, 'by_origin': by_origin, 'mean': fmean(by_seed), 'per_entity_mean': per_entity,
            'per_entity_by_origin': per_entity_by_origin, 'n_origins': n_o, 'n_entities': n_e}


def _program_view(mid: str, rec: dict, bdir: Path) -> dict:
    """The complete processing program without the constructor's rationale, plan id or any judgement word."""
    if mid == 'FixedMixup':
        return {'kind': 'public_reference', 'label': 'FixedMixup', 'definition': ec.FIXED_MIXUP['definition'], 'uniform': True}
    if mid in PUBLIC_STEPS:
        steps = PUBLIC_STEPS[mid]
        return {'kind': 'public_reference', 'label': mid, 'uniform': True, 'policy': {'default': {'steps': steps}, 'rules': []},
                'programs': [steps], 'entity_program': [0] * ec.COHORT_SIZE, 'program_label': ec.program_label(steps), 'execution_order': 'fixed: ' + ' -> '.join(ta.PRIMITIVES)}
    p = bdir / 'aug_materials' / (mid + '__compiled.json')
    if not p.exists():
        raise RuntimeError('compiled program missing for an evaluated plan: %s' % p)
    ms = context.read_json(p)['material_spec']
    pol = ms['policy']
    return {'kind': 'constructed_plan', 'label': ms['label'], 'uniform': ec.uniform_steps(context.read_json(p)['assignment']) is not None,
            'policy': {'default': {'steps': pol['default']['steps']}, 'rules': [{'when': r['when'], 'steps': r['steps']} for r in pol.get('rules', []) or []],
                       'observation_fields_used': pol.get('observation_fields_used', [])},
            'resolved_thresholds': ms.get('resolved_thresholds', {}), 'programs': ms['programs'], 'entity_program': ms['entity_program'], 'rule_index': ms.get('rule_index'),
            'program_labels': [ec.program_label(s) for s in ms['programs']], 'n_entities_with_null_field': ms.get('n_unknown', 0), 'execution_order': ms['execution_order']}


def _diagnostics_view(mid: str, phys: str, common: Path) -> dict | str:
    """Cached material diagnostics of the physical material (the common cache's build summary): what changed, never utility."""
    if mid == 'None':
        return 'not_applicable (no augmentation: the child view equals the parent view)'
    p = common / 'aug_materials' / (phys + '__summary.json')
    if not p.exists():
        return 'unavailable'
    s = context.read_json(p)
    out = {'units': s.get('units', 'unavailable'), 'legal_windows': s.get('legal_parents', 'unavailable')}
    for k in ('rms_change_X', 'rms_change_y', 'fraction_points_changed_X', 'fraction_points_changed_y', 'windows_bitwise_unchanged', 'per_step', 'recipe_step_sets'):
        out[k] = s.get(k, 'unavailable')
    ents = s.get('entities')
    if isinstance(ents, list) and ents:
        out['per_entity_rms_change_X'] = [e.get('rms_change_X', 'unavailable') for e in ents]
        out['per_entity_rms_change_y'] = [e.get('rms_change_y', 'unavailable') for e in ents]
    else:
        out['per_entity_rms_change_X'] = out['per_entity_rms_change_y'] = 'unavailable'
    ed = s.get('edit')
    if isinstance(ed, dict):
        dis = ed.get('entity_disabled_ops') or {}
        out['edit_disabled_ops_by_entity'] = [dis.get('entity_%d' % i, 'unavailable') for i in range(ec.COHORT_SIZE)]
    out['interpretation'] = 'child - parent change of the training material; confirms what the program did, not downstream utility in either direction'
    return out


def bind_pool(role: str, case: str) -> dict:
    """Task §3.1: P = four public references + the plans the original F0 branch actually evaluated (three seeds, SCORABLE C_A, cached C_B and E),
    de-duplicated by physical material; canonical order = lexicographic semantic key (never a score) -> C00, C01, ...; the original ids stay
    on the evaluator side. Raises on any binding conflict (task: locate the artefact, never substitute)."""
    bdir, common, branch = case_dirs(role, case)
    cs = context.read_json(bdir / 'case_spec.json')
    if cs['case_id'] != case or cs['domain'] != case[:3]:
        raise RuntimeError('case_spec mismatch for %s' % case)
    reg = ec.load_registry(bdir)
    cells = ec.branch_cells(bdir)
    cb = context.read_json(bdir / 'c_b_scores.json')['cells'] if (bdir / 'c_b_scores.json').exists() else {}
    es = context.read_json(bdir / 'e_scores.json')['cells'] if (bdir / 'e_scores.json').exists() else {}
    ca_src = {c: {'material_id': r['material_id'], 'model_seed': r['model_seed'], 'c_a': r['scores']['c_a']} for c, r in cells.items()}
    by_phys, problems = {}, []
    for mid, rec in reg.items():
        mine = {r['model_seed']: r for r in cells.values() if r['material_id'] == mid}
        if not mine:
            continue                                              # built, never evaluated: not a candidate (task §3.1)
        if set(mine) != set(SEEDS):
            problems.append({'plan': mid, 'problem': 'evaluated on %d/3 seeds' % len(mine)})
            continue
        phys = rec.get('phys_id', mid)
        for s, r in mine.items():
            if r['job_id'] != case or (mid not in ('FixedMixup',) and r.get('material_key') != rec['key']) or list(r.get('roster') or []) != list(cs['roster']):
                problems.append({'plan': mid, 'seed': s, 'problem': 'cell binding conflict (job / key / roster)'})
        ca = _scores_from(ca_src, mid, 'c_a')
        cbv = _scores_from(cb, mid, 'c_b')
        ev = _scores_from(es, mid, 'e')
        if ca is None or cbv is None or ev is None:
            problems.append({'plan': mid, 'problem': 'cache missing: c_a=%s c_b=%s e=%s' % (ca is not None, cbv is not None, ev is not None)})
            continue
        row = by_phys.setdefault(phys, {'phys': phys, 'plan_ids': [], 'public_id': mid if mid in PUBLIC else None, 'key': semantic_key(mid, rec),
                                        'label': ec.assignment_label('FixedMixup' if mid == 'FixedMixup' else (rec.get('assignment') if mid not in PUBLIC_STEPS else [PUBLIC_STEPS[mid]] * ec.COHORT_SIZE),
                                                                     mid if mid in PUBLIC else None),
                                        'c_a': ca, 'c_b': cbv, 'e': ev, 'program_source_plan': mid})
        row['plan_ids'].append(mid)
        if mid in PUBLIC:
            row['public_id'] = mid
    if problems:
        raise RuntimeError('pool binding problems in %s: %s' % (case, json.dumps(problems, ensure_ascii=False)[:800]))
    missing_public = [p for p in PUBLIC if not any(r['public_id'] == p for r in by_phys.values())]
    if missing_public:
        raise RuntimeError('public references missing from the evaluated pool of %s: %s' % (case, missing_public))
    order = sorted(by_phys.values(), key=lambda r: r['key'])
    pool = {}
    for i, r in enumerate(order):
        cid = 'C%02d' % i
        src_plan = r['public_id'] or r['program_source_plan']
        pool[cid] = {**r, 'display_id': cid, 'program': _program_view(src_plan, reg[src_plan], bdir), 'diagnostics': _diagnostics_view(src_plan, r['phys'], common)}
    commit = context.read_json(bdir / 'commit.json') if (bdir / 'commit.json').exists() else None
    f0_commit = None
    if commit:
        f0_commit = next((cid for cid, r in pool.items() if commit['material_id'] in r['plan_ids']), None)
    ov = context.read_json(bdir / 'overview.json')
    return {'role': role, 'case': case, 'domain': case[:3], 'branch_dir': str(branch), 'common_dir': str(common), 'pool': pool, 'pool_size': len(pool),
            'f0_commit_display_id': f0_commit, 'f0_commit_plan_id': commit['material_id'] if commit else None, 'built_never_evaluated': [m for m in reg if not any(r['material_id'] == m for r in cells.values())],
            'overview': ov, 'order_rule': 'lexicographic on the semantic material key (resolved assignment JSON; FixedMixup = its defined key); independent of any score'}


def _pair(pool: dict, cid: str, ref_cid: str, block: str) -> dict:
    """Paired difference ref - cand per seed (positive = candidate lower loss), per origin means and signs."""
    a, b = pool[cid][block], pool[ref_cid][block]
    d = [b['by_seed'][i] - a['by_seed'][i] for i in range(len(SEEDS))]
    d_o = [b['by_origin'][o] - a['by_origin'][o] for o in range(a['n_origins'])]
    return {'by_seed_delta': d, 'mean_delta': fmean(d), 'seed_se': se3(d), 'signs_by_seed': [int(x > 0) - int(x < 0) for x in d],
            'by_origin_mean_delta': d_o, 'signs_by_origin': [int(x > 0) - int(x < 0) for x in d_o],
            'note': 'positive = the candidate has lower %s loss than the reference; the two origins are one description, not six independent samples' % block.upper()}


def action_semantics() -> dict:
    """The parent's action table (primitive semantics, preset, edit, composition rules) without the code-provenance 'source' fields (paths are not packet content)."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != 'source'}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    return strip(ES.action_table())


def _overview_view(ov: dict) -> dict:
    cols = W._columnar(ov['entities'])
    return {'n_entities': ov['n_entities'], 'field_definitions': ES.FIELD_DEFINITIONS, 'entity_index_note': 'entities are indexed 0..N-1 in this case only; the same index is used by every per-entity array',
            'summary': ov['summary'], 'entities': cols, 'geometry': ov.get('geometry'), 'consumer': ov.get('consumer'),
            'observation_source': 'T-only observations of the training window; no later row was read'}


def clean_packet(binding: dict) -> dict:
    """Task §3.2: the whitelist view every submission agent receives (no C_B / E, no ids, no paths, no commit / rationale / arm)."""
    pool = binding['pool']
    cands = {}
    for cid, r in pool.items():
        cands[cid] = {'public_reference': r['public_id'], 'label': r['label'], 'program': r['program'], 'diagnostics': r['diagnostics'],
                      'ca': {k: r['c_a'][k] for k in ('by_seed', 'by_seed_origin', 'by_origin', 'mean', 'per_entity_mean', 'per_entity_by_origin')}}
    none_id = next(c for c, r in pool.items() if r['public_id'] == 'None')
    nomix_id = next(c for c, r in pool.items() if r['public_id'] == 'P_NoMixRecipe')
    paired = {'vs_None': {cid: _pair(pool, cid, none_id, 'c_a') for cid in pool if cid != none_id},
              'vs_P_NoMixRecipe': {cid: _pair(pool, cid, nomix_id, 'c_a') for cid in pool if cid != nomix_id}}
    packet = {'domain_id': binding['domain'], 'exposure_note': 'development case of a known domain; new entity group',
              'overview': _overview_view(binding['overview']), 'action_semantics': action_semantics(),
              'candidates': cands, 'candidate_ids': sorted(pool), 'reference_ids': {'None': none_id, 'P_NoMixRecipe': nomix_id,
                                                                                   'FixedMixup': next(c for c, r in pool.items() if r['public_id'] == 'FixedMixup'),
                                                                                   'P_AmpResample': next(c for c, r in pool.items() if r['public_id'] == 'P_AmpResample')},
              'paired_ca': paired,
              'feedback_semantics': {'ca': 'C_A = normalized MSE (48-step, entity macro mean) of the shared Consumer trained on the candidate material, scored on the two support origins '
                                           '(t, t+48) right after the training window, under three paired training seeds; by_seed_origin = [seed][origin]; per_entity_mean averages seeds and origins. '
                                           'It is support evidence for the later blocks, not a guarantee of their ranking.',
                                     'seeds': list(SEEDS), 'origins': 'index 0 = origin t, index 1 = origin t+48',
                                     'display_precision': '%d significant digits (scoring uses the cached full precision)' % DISPLAY_SIG},
              'task': ('Goal: the prediction utility of the shared Consumer on the later blocks of this same entity group (not visible to you). C_A is support evidence. You may select ANY '
                       'candidate in the pool; no rule forces the C_A argmin, a 3/3-seed gate, the preset as default, or a deviation from it. The pool is fixed: no new construction or '
                       'training in this step.')}
    return packet


PACKET_TOP_KEYS = ('domain_id', 'exposure_note', 'overview', 'action_semantics', 'candidates', 'candidate_ids', 'reference_ids', 'paired_ca', 'feedback_semantics', 'task')
CASE_ID_RX = re.compile(r'(?<![A-Za-z0-9])D0\d_[A-Za-z0-9]+(?![A-Za-z0-9])|(?<![A-Za-z0-9])[SVQ]0\d(?![A-Za-z0-9])')


def packet_label_check(packet: dict, binding: dict) -> None:
    """Fail-closed: the serialized packet carries no forbidden key, no C_B / E number, no case id, no path, no source name, no original plan id / arm / commit text."""
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in FORBIDDEN_PACKET_KEYS:
                    raise PermissionError('forbidden key %r at %s' % (k, path))
                walk(v, path + '.' + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, '%s[%d]' % (path, i))
    walk(packet, '$')
    s = json.dumps(packet, ensure_ascii=False)
    low = s.lower()
    for n in SOURCE_NAMES:
        if n in low:
            raise PermissionError('packet names a data source')
    if CASE_ID_RX.search(s):
        raise PermissionError('packet carries a case id token: %r' % CASE_ID_RX.search(s).group(0))
    if dsk._PATH_TOKEN.search(s.replace('\\n', ' ').replace('\\"', ' ')):
        m = dsk._PATH_TOKEN.search(s.replace('\\n', ' ').replace('\\"', ' '))
        raise PermissionError('packet carries a path token: %r' % m.group(0))
    for cid, r in binding['pool'].items():
        for pid in r['plan_ids']:
            if pid not in PUBLIC and re.search(r'(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])' % re.escape(pid), s):
                raise PermissionError('packet carries an original plan id %r' % pid)
        if r['phys'] not in PUBLIC and r['phys'] in s:
            raise PermissionError('packet carries a physical id')
        for blk in ('c_b', 'e'):
            for x in r[blk]['by_seed']:
                if ('%.12g' % x) in s or ('%.8g' % x) in s:
                    raise PermissionError('packet carries a %s value' % blk)
    for word in ('f_domain', 'f_generic', 'f_old', 'f_new', 'random_b4', 'fixed_dev', 'w_old', 'commit_reason'):
        if word in low:
            raise PermissionError('packet carries an arm / commit token %r' % word)


def build_clean_inputs(root: Path) -> dict:
    """A: bind every case of the three roles, write evaluator/<role>/<case>.json (full precision, ids) and clean_inputs/<role>/<case>.json (display copy)."""
    P_ = paths(root)
    idx_p = P_['clean'] / 'index.json'
    if idx_p.exists():
        return context.read_json(idx_p)
    index = {'written_local': now(), 'roles': {}, 'display_precision_sig': DISPLAY_SIG}
    for role in ('learn', 'select', 'replay'):
        index['roles'][role] = {}
        for case in all_cases(role):
            b = bind_pool(role, case)
            packet = clean_packet(b)
            packet_label_check(packet, b)
            disp = sig(packet)
            (P_['evaluator'] / role).mkdir(parents=True, exist_ok=True)
            (P_['clean'] / role).mkdir(parents=True, exist_ok=True)
            ev = {k: v for k, v in b.items() if k != 'overview'}
            ev['display_map'] = {cid: {'phys': r['phys'], 'plan_ids': r['plan_ids'], 'public_id': r['public_id'], 'label': r['label']} for cid, r in b['pool'].items()}
            ev['packet_full_precision'] = packet
            dsks.write_once(P_['evaluator'] / role / ('%s.json' % case), ev)
            dsks.write_once(P_['clean'] / role / ('%s.json' % case), disp)
            index['roles'][role][case] = {'pool_size': b['pool_size'], 'candidate_ids': sorted(b['pool']), 'f0_commit_display_id': b['f0_commit_display_id'],
                                          'built_never_evaluated_count': len(b['built_never_evaluated']), 'packet_bytes': len(json.dumps(disp, ensure_ascii=False).encode('utf-8'))}
    dsks.write_once(idx_p, index)
    return index


def load_binding(root: Path, role: str, case: str) -> dict:
    return context.read_json(paths(root)['evaluator'] / role / ('%s.json' % case))


def load_packet(root: Path, role: str, case: str) -> dict:
    return context.read_json(paths(root)['clean'] / role / ('%s.json' % case))


# ============================================================================= A. Learn evidence table (0 LLM)
def G(a_e, b_e, den) -> dict | None:
    """G(A over B) = 100 x [E(B) - E(A)] / E(None) per seed; positive = A better; den = the case three-seed None mean."""
    return DP.G(a_e, b_e, den)


def _argmin(pool: dict, block: str) -> str:
    """Lowest three-seed mean of the block; ties within TOL by candidate id."""
    best = None
    for cid in sorted(pool):
        m = pool[cid][block]['mean']
        if best is None or m < best[1] - TOL:
            best = (cid, m)
    return best[0]


def learn_case_evidence(b: dict) -> dict:
    """One Learn case for Slow: the candidate fields the submission agent will see (compact) + the late supervision (C_B / E), the original F0 submission,
    the C_A argmin, the post-hoc E best and the conflicts (task §4). Shared notes live once in the table's definitions."""
    pool = b['pool']
    none_id = next(c for c, r in pool.items() if r['public_id'] == 'None')
    nomix_id = next(c for c, r in pool.items() if r['public_id'] == 'P_NoMixRecipe')
    den = pool[none_id]['e']['mean']
    cands = {}
    for cid, r in pool.items():
        diag = r['diagnostics']
        row = {'public_reference': r['public_id'], 'label': r['label'],
               'program': {k: r['program'].get(k) for k in ('kind', 'uniform', 'policy', 'resolved_thresholds', 'program_labels', 'entity_program', 'definition') if r['program'].get(k) is not None},
               'diagnostics': (diag if isinstance(diag, str) else {k: (sig(diag.get(k), 4) if k == 'per_entity_rms_change_X' else diag.get(k)) for k in ('rms_change_X', 'rms_change_y', 'windows_bitwise_unchanged', 'per_entity_rms_change_X') if k in diag}),
               'ca': {k: (sig(r['c_a'][k], 4) if k == 'per_entity_mean' else r['c_a'][k]) for k in ('by_seed', 'by_seed_origin', 'by_origin', 'mean', 'per_entity_mean')},
               'late': {'cb_by_seed': r['c_b']['by_seed'], 'cb_mean': r['c_b']['mean'], 'e_by_seed': r['e']['by_seed'], 'e_by_origin': r['e']['by_origin'], 'e_mean': r['e']['mean']}}
        for ref_name, ref_id in (('vs_None', none_id), ('vs_P_NoMixRecipe', nomix_id)):
            if cid == ref_id:
                continue
            pa = _pair(pool, cid, ref_id, 'c_a')
            pe = _pair(pool, cid, ref_id, 'e')
            row.setdefault('paired', {})[ref_name] = {'ca_by_seed_delta': pa['by_seed_delta'], 'ca_mean_delta': pa['mean_delta'], 'ca_seed_se': pa['seed_se'], 'ca_by_origin_mean_delta': pa['by_origin_mean_delta'],
                                                     'cb_mean_delta': pool[ref_id]['c_b']['mean'] - r['c_b']['mean'], 'e_mean_delta': pe['mean_delta'], 'e_seed_se': pe['seed_se'],
                                                     'g_e_pp_of_none': G(r['e']['by_seed'], pool[ref_id]['e']['by_seed'], den)['mean']}
        if cid != none_id:
            ca_d = [pool[none_id]['c_a']['by_origin'][o] - r['c_a']['by_origin'][o] for o in range(2)]
            e_d = pool[none_id]['e']['mean'] - r['e']['mean']
            cb_d = pool[none_id]['c_b']['mean'] - r['c_b']['mean']
            seed_d = [pool[none_id]['c_a']['by_seed'][i] - r['c_a']['by_seed'][i] for i in range(3)]
            row['conflicts_vs_None'] = {'ca_origins_disagree': (ca_d[0] > 0) != (ca_d[1] > 0), 'ca_seeds_disagree': len({x > 0 for x in seed_d}) > 1,
                                        'ca_vs_e_disagree': (fmean(seed_d) > 0) != (e_d > 0), 'cb_vs_e_disagree': (cb_d > 0) != (e_d > 0)}
        cands[cid] = row
    ca_arg, cb_arg, e_arg = _argmin(pool, 'c_a'), _argmin(pool, 'c_b'), _argmin(pool, 'e')
    f0 = b['f0_commit_display_id']
    summ = {'pool_size': len(pool), 'reference_ids': {'None': none_id, 'P_NoMixRecipe': nomix_id}, 'ca_argmin': ca_arg, 'cb_argmin': cb_arg, 'e_argmin_post_hoc': e_arg,
            'original_f0_submission': f0,
            'delivery_pp_of_none': {'ca_argmin_over_None': G(pool[ca_arg]['e']['by_seed'], pool[none_id]['e']['by_seed'], den)['mean'],
                                    'f0_over_None': G(pool[f0]['e']['by_seed'], pool[none_id]['e']['by_seed'], den)['mean'] if f0 else None,
                                    'e_best_over_None': G(pool[e_arg]['e']['by_seed'], pool[none_id]['e']['by_seed'], den)['mean'],
                                    'nomix_over_None': G(pool[nomix_id]['e']['by_seed'], pool[none_id]['e']['by_seed'], den)['mean'],
                                    'candidate_opportunity_e_best_over_nomix': G(pool[e_arg]['e']['by_seed'], pool[nomix_id]['e']['by_seed'], den)['mean'],
                                    'regret_of_ca_argmin_vs_e_best': G(pool[e_arg]['e']['by_seed'], pool[ca_arg]['e']['by_seed'], den)['mean'],
                                    'regret_of_f0_vs_e_best': G(pool[e_arg]['e']['by_seed'], pool[f0]['e']['by_seed'], den)['mean'] if f0 else None},
            'ca_argmin_equals_e_argmin': ca_arg == e_arg, 'f0_equals_ca_argmin': f0 == ca_arg,
            'e_ranking_by_mean': sorted(pool, key=lambda c: pool[c]['e']['mean']), 'ca_ranking_by_mean': sorted(pool, key=lambda c: pool[c]['c_a']['mean'])}
    ov = b['packet_full_precision']['overview']
    cols = ov['entities']['columns']
    const = {k: v[0] for k, v in cols.items() if len({json.dumps(x) for x in v}) == 1}
    ents = {'n': ov['entities']['n'], 'columns': sig({k: v for k, v in cols.items() if k not in const}, 4), 'constant_columns': const}
    return {'overview': {'summary': {k: v for k, v in ov['summary'].items() if k not in const}, 'entities': ents, 'n_entities': ov['n_entities']},
            'candidates': cands, 'case_summary': summ}


LEARN_NOTES = {
    'late': 'cb = the delayed check block right after C_A (origins t+96, t+144); e = the late block (four origins t+192..t+336): the learning criterion. Neither was visible at submission time; the submission agent never sees them.',
    'paired': 'deltas are reference minus candidate (positive = the candidate has the lower loss); ca_by_origin_mean_delta indexes [origin t, origin t+48]; g_e_pp_of_none = 100 x (E(ref) - E(cand)) / E(None) of the case; per-seed E differences follow from late.e_by_seed',
    'overview': 'entities.columns hold the per-entity T-only observations (4 significant digits) of the fields that vary in the case; constant_columns list the fields equal for every entity; summary holds the batch min/p25/median/p75/max',
    'conflicts_vs_None': 'ca_origins_disagree = the C_A delta to None changes sign between the two origins (temporal variation inside the support block); ca_seeds_disagree = training repeatability; ca_vs_e_disagree / cb_vs_e_disagree = the earlier block did not carry to the late block',
    'case_summary': 'original_f0_submission = the plan the original research trajectory submitted (a learning fact for you; the new submission agent does not see it); e_argmin_post_hoc measures candidate opportunity in this pool, a description, not a deployable upper bound; ca_argmin = the lowest three-seed C_A mean',
    'diagnostics': 'child - parent change of the training material in T-scaler units; per_entity_rms_change_X indexes entities 0..N-1; confirms what the program did, never utility',
    'program': 'policy = default steps + observation rules (first match wins; thresholds resolved on this case); entity_program indexes program_labels per entity; uniform = every entity has the same program',
    'ca': 'C_A normalized MSE, entity macro; by_seed_origin = [seed][origin]; per_entity_mean averages seeds and origins (4 significant digits here); the submission agent sees exactly these fields at 8 significant digits'}


def learn_evidence(root: Path, domain: str) -> dict:
    """The frozen per-domain Learn table (evidence/<D>/learn_table.json) with legal refs."""
    P_ = paths(root)
    p = P_['evidence'] / domain / 'learn_table.json'
    if p.exists():
        return context.read_json(p)
    cases, refs = {}, []
    fixed = {pid: {'g_over_None_by_case': {}, 'g_over_P_NoMixRecipe_by_case': {}} for pid in PUBLIC}
    for case in all_cases('learn', domain):
        b = load_binding(root, 'learn', case)
        tail = case.split('_')[1]
        cases[tail] = learn_case_evidence(b)
        refs += ['learn/%s/overview' % tail, 'learn/%s/case_summary' % tail] + ['learn/%s/%s' % (tail, cid) for cid in sorted(b['pool'])]
        pool = b['pool']
        none_id = next(c for c, r in pool.items() if r['public_id'] == 'None')
        nomix_id = next(c for c, r in pool.items() if r['public_id'] == 'P_NoMixRecipe')
        den = pool[none_id]['e']['mean']
        for pid in PUBLIC:
            cid = next(c for c, r in pool.items() if r['public_id'] == pid)
            fixed[pid]['g_over_None_by_case'][tail] = G(pool[cid]['e']['by_seed'], pool[none_id]['e']['by_seed'], den)['mean']
            fixed[pid]['g_over_P_NoMixRecipe_by_case'][tail] = G(pool[cid]['e']['by_seed'], pool[nomix_id]['e']['by_seed'], den)['mean']
    for pid in PUBLIC:
        for k in ('g_over_None_by_case', 'g_over_P_NoMixRecipe_by_case'):
            v = list(fixed[pid][k].values())
            fixed[pid][k.replace('_by_case', '_equal_weight_mean')] = fmean(v)
            fixed[pid][k.replace('_by_case', '_cases_negative')] = [c for c, x in fixed[pid][k].items() if x < 0]
        refs.append('learn/fixed_public/%s' % pid)
    fixed['note'] = 'equal-weight over the eight Learn cases; pp of the case None mean; a fixed preference keeps its counterexamples (cases_negative)'
    out = {'domain_id': domain, 'frozen_local': now(), 'n_cases': len(cases), 'cases': cases, 'fixed_public_readout': fixed, 'legal_evidence_refs': sorted(set(refs)),
           'definitions': {'field_definitions': ES.FIELD_DEFINITIONS, 'action_semantics': action_semantics(), 'notes': LEARN_NOTES,
                           'seeds': list(SEEDS), 'g_pp_of_none': '100 x [E(B) - E(A)] / E(None), three-seed means; positive = A better',
                           'display_precision': 'values rounded to 5 significant digits in this table'}}
    out = sig(out, 5)
    dsks.write_once(p, out)
    return out


# ============================================================================= B. Slow: three independent submission-card proposals per domain
DECIDE_SYSTEM = ('You are the SUBMISSION step of the Fast path of a batch training-data augmentation research Harness. One case = one entity group of N series and one shared MLP '
                 'Consumer trained on the parent view (unchanged training windows) plus the child view of ONE augmentation candidate (0.5/0.5). The candidate pool of this case is '
                 'already constructed and evaluated; your only action is to select which evaluated candidate is submitted for the later blocks.\n'
                 '阅读本案合法观察、候选程序、材料诊断和 C_A，提交一个已评估候选，以改善随后时间块的整批预测。C_A 与历史指导均可能出错；说明哪条当前证据使你采用或拒绝一个偏好。'
                 '材料更平、更强或未在历史中尝试，本身不是有害证据。可以保持公共方案，也可以选择组合或条件方案，无需为了表现变化而偏离。你看不到本案 C_B/E。\n'
                 '(Read this case\'s legal observations, candidate programs, material diagnostics and C_A, and submit one evaluated candidate to improve the whole-batch prediction '
                 'of the following time blocks. C_A and historical guidance can both be wrong; say which current evidence makes you adopt or reject a preference. A material that '
                 'is smoother, stronger, or never tried in history is not by itself evidence of harm. You may keep a public reference, or choose a composed or conditional plan; '
                 'you need not deviate for the sake of a behaviour change. You cannot see this case\'s C_B / E.)\n'
                 'If `guidance.loaded` holds a domain card, it is advice learned from earlier cases of this domain: apply it where the current evidence supports it, and say '
                 'when the current evidence makes you depart from it. Without a card, decide from the packet alone. Cite only fields that exist in the request.\n'
                 'Return exactly one JSON object, no markdown: {"selected_candidate_id":"<one id from candidate_ids>","evidence_refs":["<dotted paths into this request, e.g. '
                 'candidates.C03.ca.by_seed_origin, paired_ca.vs_None.C03.by_origin_mean_delta, overview.summary.lag24_corr>"],"reason":"<why this candidate; only evidence in '
                 'the request>","uncertainty":"<which comparisons remain undecided or risky>","guidance_applied":"<the guidance sentence you applied, quoted or closely '
                 'paraphrased, or null when no guidance is loaded or none applied>"}')

FEEDBACK_ROLES_FP = ('Feedback roles, by name: C_A = the immediate support block the submission agent legally sees (origins t, t+48; three paired training seeds; per origin and '
                     'per entity). C_B = the delayed check block (origins t+96, t+144). E = the late block (origins t+192..t+336): the criterion this study cares about and the '
                     'supervision you learn from. C_B and E are shown to you as late supervision; the submission agent never sees them. When C_A and E disagree, keep the '
                     'conflict: a C_A lead is not a later gain, and no rule may a priori ignore all C_A. The eight cases are different entity groups of ONE domain at two cut '
                     'positions; the cases this card will serve are NEW entity groups of the same domain with pools of the same kind (the four public references plus what one '
                     'research trajectory evaluated). A per-entity loss change is not the independent contribution of that entity\'s material. Material diagnostics (smoother, '
                     'flatter, larger change) are not downstream utility in either direction. Never-tried is not harmful. The post-hoc E best of a pool is a description of '
                     'candidate opportunity, not a known answer.')
GOAL_FP = ('Learning goal (task §5): 从多个同域案例中提出可执行的比较与提交方法，使未见实体案例的后期表现改善。后期 C_B/E 是学习监督，不能当作可忽略旁注；当前 C_A 是部署可用证据，不保证等于最终目标。'
           '学习当前观察怎样改变候选比较、什么证据支持偏离某个偏好、何时保留不确定性。不得预设永远保留或永远拒绝 NoMix，也不得凭未尝试认定条件化/三步方案有害。 '
           '(From several cases of one domain, propose an executable comparison-and-submission method that improves the late performance of unseen entity groups. The late C_B / E '
           'are your supervision, not a footnote; the current C_A is the deployable evidence and is not guaranteed to equal the final target. Learn how the current observations '
           'change the candidate comparison, which evidence supports departing from a preference, and when to keep uncertainty. Do not preset "always keep" or "always reject" '
           'the preset P_NoMixRecipe; do not call conditional or three-step plans harmful because they were untried.) The card may form a fixed preference; then say so '
           '(preference_nature). Do not invent conditional branches the evidence does not support. NO_SUPPORTED_RULE is a valid scientific output when the evidence supports no '
           'rule; it is not resampled. Every main rule (including a kept default) needs its support / counter refs from the supplied legal refs and a status supported | '
           'hypothesis | unresolved.')
NO_IDS_FP = ('Deployable text (workflow, principles, applicability_summary, research_mode) must not contain case labels (S01, Q3 ...), candidate display ids (C03 ...), entity or '
             'column numbers, cut positions or row numbers, data source names, historical assignments or answer tables. Refer to candidates by their program semantics and by '
             'the packet fields the submission agent receives (e.g. paired_ca.vs_None.<id>.by_origin_mean_delta, candidates.<id>.diagnostics.rms_change_X, '
             'overview.summary.lag24_corr). Do not change model, scoring, tools, permissions, budgets, randomness, targets or legal windows. observable_applicability MUST be exactly '
             '{"const":true}.')
FORMAT_FP = ('Output exact JSON, no markdown: {"decision":"NO_SUPPORTED_RULE","rationale":"...","evidence_review":{"supports":[...],"contradicts_or_limits":[...],"rule_changes":[...],'
             '"uncertain_and_expected_behavior_change":"..."}} OR {"decision":"PROPOSE","candidate":{"research_mode":"<short label>","workflow":"<executable submission Workflow: '
             'how to read the packet, which comparisons to make in which order, when to keep or depart from a preference, and the commit rule>","principles":"<conditions, actions, '
             'reasons, exceptions>" or null,"observable_applicability":{"const":true},"applicability_summary":"<=400 characters","evidence_refs":["exact refs from legal_evidence_refs"],'
             '"rationale":"...","evidence_review":{"supports":["observation -> which advice it supports"],"contradicts_or_limits":["which late result contradicts or limits which judgement"],'
             '"rule_changes":["which rule is kept / dropped / rewritten because of it"],"uncertain_and_expected_behavior_change":"..."},"rule_status":[{"rule":"<one main rule>",'
             '"status":"supported|hypothesis|unresolved","support":["legal refs or short facts; at least one legal ref per rule across support+counter"],"counter":["..."]}],'
             '"commit_policy":"<one paragraph restating the submission rule>","preference_nature":"fixed|conditional|mixed",'
             '"expected_decision_example":{"ref":"<one legal candidate ref learn/<case>/<id>>","expected_choice_and_why":"..."},'
             '"counterexample_expected":{"ref":"<one legal ref>","should_hold_or_may_fail":"..."}}}. Workflow plus Principles render to at most %d characters.')
SLOW_FP = ('You are the offline Slow of a batch training-data augmentation research Harness. You receive a deterministic evidence table of EIGHT completed research cases of ONE '
           'neutral domain (entity groups of 16 series at two cut positions). For every case: the T-only overview the submission agent sees, every evaluated candidate of that '
           'case\'s pool exactly as the submission agent will see it (program, cached material diagnostics, C_A by seed / origin / entity, paired C_A differences to None and to '
           'the preset), and - separately, as supervision - the late C_B / E of every candidate, the C_A argmin, the original submission and the post-hoc E best. Also the '
           'equal-weight readout of the four public references over the eight cases with their counterexamples. You are one of three independent proposals formed from the '
           'identical table; you do not see the other two, and no earlier card of any generation is given to you. ' + FEEDBACK_ROLES_FP + ' ' + GOAL_FP + ' ' + NO_IDS_FP + ' ' +
           FORMAT_FP % BODY_LIMIT)
CAND_KEYS_FP = {'research_mode', 'workflow', 'principles', 'observable_applicability', 'applicability_summary', 'evidence_refs', 'rationale', 'evidence_review', 'rule_status',
                'commit_policy', 'preference_nature', 'expected_decision_example', 'counterexample_expected'}
SUBMIT_INTERFACE = {'packet_top_level_keys': list(PACKET_TOP_KEYS),
                    'candidate_fields': ['public_reference', 'label', 'program', 'diagnostics', 'ca.by_seed', 'ca.by_seed_origin', 'ca.by_origin', 'ca.mean', 'ca.per_entity_mean', 'ca.per_entity_by_origin'],
                    'paired_ca_fields': ['paired_ca.vs_None.<id>.{by_seed_delta, mean_delta, seed_se, signs_by_seed, by_origin_mean_delta, signs_by_origin}', 'paired_ca.vs_P_NoMixRecipe.<id>.{same}'],
                    'not_in_packet': ['C_B', 'E', 'the original submission', 'plan names', 'any other case'],
                    'guidance': 'your card is delivered as guidance.loaded[0].body (experiment_guidance) in the same request; the no-card arm gets an empty guidance block and an otherwise identical request',
                    'output_contract': {'selected_candidate_id': 'one of candidate_ids', 'evidence_refs': 'dotted paths into the request', 'reason': 'text', 'uncertainty': 'text', 'guidance_applied': 'text or null'},
                    'system_prompt_of_the_submission_agent': DECIDE_SYSTEM}


def fp_text_check(text: str, where: str) -> None:
    DP.dp_text_check(text, where)
    m = CID_TOKEN.search(str(text or ''))
    if m:
        raise ValueError('%s carries a candidate display id %r (deployable text must not index candidates)' % (where, m.group(0)))


def make_card_fp(*, domain, skill_id, workflow, principles, summary, refs, legal_refs, mode) -> dsk.Skill:
    if not isinstance(mode, str) or not mode.strip() or len(mode) > 80:
        raise ValueError('research_mode must be short nonempty text')
    dsk.check_reusable_text(mode, 'research_mode', fp_text_check)
    return dsk.make_skill(skill_id=skill_id, domain_id=domain, revision=1, workflow=workflow, principles=principles, applicability_summary=summary,
                          compatibility_note='Formed for this study\'s fixed task, Consumer, L/H, hourly sampling, T length, 16-entity cases and fixed evaluated pools.',
                          observable_applicability={'const': True}, evidence_refs=refs, legal_evidence_refs=legal_refs, source_stage='propose', status='CANDIDATE_TEST_ONLY',
                          allowed_features=ES.ALLOWED_FEATURES, text_check=fp_text_check, body_limit=BODY_LIMIT)


def _legal_ref(x, legal: set, where: str) -> str:
    return DP._legal_ref(x, legal, where)


def check_rule_status_fp(rs, legal: set) -> list:
    out = ES.check_rule_status(rs)
    for r in out:
        if not any(x in legal for x in r['support'] + r['counter']):
            raise ValueError('rule %r cites no legal evidence ref in support or counter' % r['rule'][:60])
    return out


def parse_proposal(resp, *, domain: str, slot: int, legal_refs) -> dict:
    legal = set(legal_refs)
    if not isinstance(resp, dict) or resp.get('decision') not in ('NO_SUPPORTED_RULE', 'PROPOSE'):
        raise ValueError('decision must be NO_SUPPORTED_RULE or PROPOSE')
    if resp['decision'] == 'NO_SUPPORTED_RULE':
        if set(resp) - {'decision', 'rationale', 'evidence_review'}:
            raise ValueError('NO_SUPPORTED_RULE carries only rationale and evidence_review')
        return {'decision': 'NO_SUPPORTED_RULE', 'skill': None,
                'meta': {'rationale': str(resp.get('rationale', ''))[:3000], 'evidence_review': LL.check_review(resp['evidence_review']) if 'evidence_review' in resp else None}}
    if set(resp) != {'decision', 'candidate'} or not isinstance(resp['candidate'], dict):
        raise ValueError('PROPOSE carries exactly decision and one candidate object')
    c = resp['candidate']
    if set(c) != CAND_KEYS_FP:
        raise ValueError('candidate keys must be exactly %s' % sorted(CAND_KEYS_FP))
    if c['observable_applicability'] != {'const': True}:
        raise ValueError('observable_applicability must be exactly {"const": true}')
    if not isinstance(c['commit_policy'], str) or not c['commit_policy'].strip():
        raise ValueError('commit_policy must be nonempty text')
    if c['preference_nature'] not in ('fixed', 'conditional', 'mixed'):
        raise ValueError('preference_nature must be fixed | conditional | mixed')
    ex_ref = _legal_ref(c['expected_decision_example'], legal, 'expected_decision_example')
    counter_ref = _legal_ref(c['counterexample_expected'], legal, 'counterexample_expected')
    review = LL.check_review(c['evidence_review'])
    rs = check_rule_status_fp(c['rule_status'], legal)
    s = make_card_fp(domain=domain, skill_id='%s-FP-N%d-r1' % (domain, slot), workflow=c['workflow'], principles=c['principles'], summary=c['applicability_summary'],
                     refs=c['evidence_refs'], legal_refs=legal_refs, mode=c['research_mode'])
    meta = {'research_mode': c['research_mode'], 'rationale': str(c['rationale'])[:3000], 'evidence_review': review, 'rule_status': rs, 'commit_policy': c['commit_policy'][:2000],
            'preference_nature': c['preference_nature'], 'expected_decision_example': {**c['expected_decision_example'], 'ref': ex_ref},
            'counterexample_expected': {**c['counterexample_expected'], 'ref': counter_ref}}
    return {'decision': 'PROPOSE', 'skill': s, 'meta': meta}


def slow_payload(table: dict, domain: str, slot: int) -> dict:
    body = {k: v for k, v in table.items() if k != 'legal_evidence_refs'}
    return {'domain_id': domain, 'proposal_slot': slot, 'proposal_note': 'proposal %d of three independent proposals formed from the identical table; the others are not visible to you' % slot,
            'evidence': body, 'submission_agent_interface': SUBMIT_INTERFACE, 'body_limit_characters': BODY_LIMIT, 'legal_evidence_refs': table['legal_evidence_refs'],
            'legal_evidence_refs_format': 'evidence_refs, rule_status support/counter refs and the two ref fields must be exact strings from legal_evidence_refs',
            'required': {'observable_applicability': {'const': True}}, 'status': 'CANDIDATE_TEST_ONLY'}


def payload_bytes(system: str, payload: dict) -> int:
    return DP.payload_bytes(system, payload)


def propose_one(payload: dict, call, *, domain: str, slot: int, legal_refs) -> dict:
    """One scientific proposal call; one contract correction only; transport / account / budget faults end as PROPOSE_CALL_FAILED (never resampled)."""
    low = json.dumps(payload, ensure_ascii=False).lower()
    if any(n in low for n in SOURCE_NAMES):
        raise PermissionError('Slow payload names a data source')
    attempts, prior, receipts = [], None, []
    for i in range(2):
        try:
            raw, rec = call(payload)
            receipts.append(rec)
        except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
            attempts.append({'attempt': i, 'fault_kind': type(exc).__name__, 'detail': str(exc)[:200]})
            return {'status': 'PROPOSE_CALL_FAILED', 'attempts': attempts, 'receipts': receipts, 'skill': None, 'meta': {}}
        except ValueError as exc:
            attempts.append({'attempt': i, 'error': 'response not JSON: %s' % str(exc)[:200]})
            prior = None
        else:
            try:
                out = parse_proposal(raw, domain=domain, slot=slot, legal_refs=legal_refs)
                attempts.append({'attempt': i, 'ok': True})
                return {'status': 'NO_SUPPORTED_RULE' if out['decision'] == 'NO_SUPPORTED_RULE' else 'PROPOSED', 'attempts': attempts, 'receipts': receipts, 'raw': raw, **out}
            except ValueError as exc:
                attempts.append({'attempt': i, 'error': str(exc)[:300]})
                prior = raw if dsk._jsonable(raw) else None
        if i == 0:
            payload = {**payload, 'correction': {'previous_output': prior, 'error': attempts[-1].get('error'), 'instruction': 'Correct this contract error only. NO_SUPPORTED_RULE is allowed. Do not seek a different outcome.'}}
    return {'status': 'PROPOSE_PARSE_OR_VALIDATION_FAILED', 'attempts': attempts, 'receipts': receipts, 'skill': None, 'meta': {}}


# ============================================================================= ledger / clock / warnings (one writer at a time)
def stage_caps(root: Path, stage: str) -> dict:
    P_ = paths(root)
    rec = context.read_json(P_['stages']) if P_['stages'].exists() else {}
    if stage in rec:
        return rec[stage]
    led = DP.SafeLedger(P_['ledger'], **TOTAL)
    s, alloc = led.s, ALLOC[stage]
    snap = {k: s.get(k, 0) for k in ('llm_requests', 'llm_http_attempts', 'llm_tokens_in', 'llm_tokens_out', 'llm_tokens_unknown', 'llm_failed_attempts')}
    caps = {'max_fit_attempts': 0, 'max_llm_requests': min(TOTAL['max_llm_requests'], s['llm_requests'] + alloc['requests']), 'max_llm_tokens': TOTAL['max_llm_tokens'],
            'max_wall_s': TOTAL['max_wall_s'], 'max_retries': 0}
    rec[stage] = {'epoch_start': time.time(), 'local_start': now(), 'allocation': alloc, 'snapshot_at_start': snap, 'caps': caps,
                  'http_cap': min(HTTP_CAP, s['llm_http_attempts'] + alloc['requests'] + max(0, TRANSPORT_RETRIES - s.get('llm_failed_attempts', 0)))}
    context.write_json(P_['stages'], rec)
    return rec[stage]


def start_paid_clock(root: Path) -> None:
    P_ = paths(root)
    led = DP.SafeLedger(P_['ledger'], **TOTAL)
    if led.s.get('paid_clock_started_epoch') is None:
        if led.s['llm_requests']:
            raise RuntimeError('paid requests exist before the paid clock start')
        led.s['wiring_elapsed_before_paid_clock_s'] = time.time() - led.s['started_epoch']
        led.s['paid_clock_started_epoch'] = time.time()
        led.s['paid_clock_started_local'] = now()
        led._save()


def budget_warnings(root: Path, where: str) -> list:
    P_ = paths(root)
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    tok = led.get('llm_tokens_in', 0) + led.get('llm_tokens_out', 0)
    paid = (time.time() - led['paid_clock_started_epoch']) if led.get('paid_clock_started_epoch') else 0.0
    P_['warnings'].mkdir(parents=True, exist_ok=True)
    out = []
    for th in TOKEN_WARNINGS:
        p = P_['warnings'] / ('tokens_%d.json' % th)
        if tok >= th and not p.exists():
            context.write_json(p, {'threshold': th, 'tokens': tok, 'where': where, 'local': now(), 'action': 'report used / remaining; no data, arm or request change'})
            out.append(str(p))
    for th in PAID_WALL_WARNINGS_S:
        p = P_['warnings'] / ('paid_wall_%dh.json' % (th // 3600))
        if paid >= th and not p.exists():
            context.write_json(p, {'threshold_s': th, 'paid_elapsed_s': paid, 'where': where, 'local': now(), 'action': 'report used / remaining; explain if abnormal'})
            out.append(str(p))
    for w in out:
        print('BUDGET_WARNING', w, flush=True)
    return out


def accept_unknown_usage(root: Path, where: str) -> None:
    """Operator decision recorded explicitly; never called by the driver on its own."""
    P_ = paths(root)
    led = DP.SafeLedger(P_['ledger'], **TOTAL)
    if rt.unknown_usage_blocks(led):
        led.s['unknown_usage_accepted'] = led.s['llm_tokens_unknown']
        led.event(kind='unknown_usage_accepted', accepted=led.s['llm_tokens_unknown'], where=where, operator='explicit --accept-unknown-usage')
        print('UNKNOWN_USAGE_ACCEPTED', led.s['llm_tokens_unknown'], where, flush=True)


DISPATCH_GATE: threading.BoundedSemaphore | None = None      # at most CONCURRENCY['http'] logical requests inside client.call at once, so a request queued behind
                                                            # a failure is still a NEW dispatch (blocked by the unknown-usage rule) rather than an already-numbered one


def gated(call):
    global DISPATCH_GATE
    if DISPATCH_GATE is None:
        DISPATCH_GATE = threading.BoundedSemaphore(CONCURRENCY['http'])

    def run(payload):
        with DISPATCH_GATE:
            return call(payload)
    return run


def open_stage_client(root: Path, stage: str, *, client_factory=None):
    """One ledger writer per stage; the paid clock starts before the stage ledger instance exists."""
    P_ = paths(root)
    sc = stage_caps(root, stage)
    if client_factory is None and not DP.proxy_reachable():
        raise RuntimeError('LLM proxy not reachable; stage %s refused before any paid call' % stage)
    start_paid_clock(root)
    led = DP.SafeLedger(P_['ledger'], **sc['caps'])
    if rt.unknown_usage_blocks(led):
        raise RuntimeError('package ledger holds unknown usage that no operator accepted; stage %s refused' % stage)
    out = P_[stage if stage != 'slow' else 'formation'] / 'raw_responses'
    factory = client_factory or (lambda led_, out_, http_cap, stage_: DP.MeteredClient(led_, out_, http_cap=http_cap, stage=stage_))
    client = factory(led, out, sc['http_cap'], stage)
    return led, client


def slow_stage(root: Path, *, client_factory=None) -> dict:
    """B: six independent proposal calls through the HTTP pool; one contract correction each; NO_SUPPORTED_RULE / byte-identical body = alias of s0."""
    P_ = paths(root)
    P_['formation'].mkdir(parents=True, exist_ok=True)
    summary_p = P_['formation'] / 'proposals.json'
    if summary_p.exists():
        return context.read_json(summary_p)
    tables = {d: learn_evidence(root, d) for d in DOMAINS}
    payloads = {(d, k): slow_payload(tables[d], d, k) for d in DOMAINS for k in (1, 2, 3)}
    res_p = P_['formation'] / 'reservation_check.json'
    if not res_p.exists():
        rows = {}
        for (d, k), p in payloads.items():
            nb = payload_bytes(SLOW_FP, p)
            rows['%s_N%d' % (d, k)] = {'bytes': nb, 'estimated_prompt_tokens': int(nb / SLOW_BYTES_PER_TOKEN), 'reserved_tokens_upper_2x': 2 * (nb + 2048 + MAX_OUTPUT_TOKENS),
                                       'within_target': int(nb / SLOW_BYTES_PER_TOKEN) <= SLOW_INPUT_TOKEN_TARGET * 1.15}
        dsks.write_once(res_p, {'checked_local': now(), 'payloads': rows, 'input_token_target': SLOW_INPUT_TOKEN_TARGET, 'bytes_per_token_assumed': SLOW_BYTES_PER_TOKEN,
                                'note': 'no call sent by this check; sizes of the final serialized payloads'})
    led, client = open_stage_client(root, 'slow', client_factory=client_factory)
    results, lock = {}, threading.Lock()

    def one(d, k):
        outp = P_['formation'] / d / ('proposal_N%d.json' % k)
        if outp.exists():
            prior = context.read_json(outp)
            if prior['status'] in ('PROPOSED', 'NO_SUPPORTED_RULE', 'PROPOSE_PARSE_OR_VALIDATION_FAILED'):
                with lock:
                    results[(d, k)] = prior
                return
            n_prev = len(list((P_['formation'] / d).glob('proposal_N%d_failed_*.json' % k)))
            outp.rename(P_['formation'] / d / ('proposal_N%d_failed_%d.json' % (k, n_prev + 1)))   # technical failure: the same frozen payload is called again after the operator decision
        legal = tables[d]['legal_evidence_refs']
        res = propose_one(payloads[(d, k)], gated(lambda p: client.call('slow_propose', '%s_N%d' % (d, k), p, SLOW_FP, max_tokens=MAX_OUTPUT_TOKENS, meta={'domain': d, 'slot': k},
                                                                           request_timeout=SLOW_REQUEST_TIMEOUT_S)), domain=d, slot=k, legal_refs=legal)
        rec = {**{kk: v for kk, v in res.items() if kk != 'skill'}, 'skill': res['skill'].to_json() if res.get('skill') else None, 'domain_id': d, 'slot': k, 'written_local': now()}
        outp.parent.mkdir(parents=True, exist_ok=True)
        dsks.write_once(outp, rec)
        with lock:
            results[(d, k)] = rec
    errs = DP._parallel([(d, k) for d in DOMAINS for k in (1, 2, 3)], one)
    budget_warnings(root, 'slow')
    if errs:
        raise RuntimeError('slow proposal thread failed: %s' % errs[0][1])
    failed = [(d, k) for (d, k), r in results.items() if r['status'] == 'PROPOSE_CALL_FAILED']
    if client.fatal or rt.unknown_usage_blocks(led) or failed:
        raise RuntimeError('Slow stopped: fatal=%s unknown_usage_blocks=%s call_failed=%s; finished proposals are kept; rerun --run (with --accept-unknown-usage only after the user\'s decision) to call the unfinished slots only'
                           % (client.fatal, rt.unknown_usage_blocks(led), ['%s_N%d' % x for x in failed]))
    summary = {'written_local': now(), 'domains': {}}
    for d in DOMAINS:
        rows, bodies = [], {}
        for k in (1, 2, 3):
            r = results[(d, k)]
            row = {'slot': k, 'arm': 'n%d' % k, 'status': r['status'], 'attempts': r['attempts'], 'requests': [x.get('request') for x in r.get('receipts', [])],
                   'tokens': sum((x.get('prompt_tokens') or 0) + (x.get('completion_tokens') or 0) for x in r.get('receipts', [])), 'alias_of': None}
            if r['status'] == 'NO_SUPPORTED_RULE':
                row['alias_of'] = 's0'
                row['alias_reason'] = 'NO_SUPPORTED_RULE: a valid scientific output; the arm is the no-card alias'
            elif r['status'] == 'PROPOSED':
                body = r['skill']['rendered_body']
                same = next((name for name, b in bodies.items() if b == body), None)
                if same is not None:
                    row['alias_of'] = same
                    row['alias_reason'] = 'byte-identical rendered body to %s' % same
                else:
                    bodies['n%d' % k] = body
                row['research_mode'] = r['meta'].get('research_mode')
                row['preference_nature'] = r['meta'].get('preference_nature')
                row['body_characters'] = len(body)
            else:
                row['alias_reason'] = 'technical failure (format after one correction): not NO_SUPPORTED_RULE, not an alias; no card for this slot'
            rows.append(row)
        summary['domains'][d] = {'proposals': rows, 'n_distinct_new_cards': sum(1 for r in rows if r['status'] == 'PROPOSED' and r['alias_of'] is None)}
    dsks.write_once(summary_p, summary)
    return summary


def cards_of(root: Path, domain: str) -> dict:
    """{arm: Skill} of the domain's distinct cards (aliases and failures excluded)."""
    P_ = paths(root)
    summ = context.read_json(P_['formation'] / 'proposals.json')
    out = {}
    for row in summ['domains'][domain]['proposals']:
        if row['status'] == 'PROPOSED' and row['alias_of'] is None:
            rec = context.read_json(P_['formation'] / domain / ('proposal_N%d.json' % row['slot']))
            out[row['arm']] = dsk.skill_from_json(rec['skill'])
    return out


def aliases_of(root: Path, domain: str) -> dict:
    summ = context.read_json(paths(root)['formation'] / 'proposals.json')
    return {row['arm']: row['alias_of'] for row in summ['domains'][domain]['proposals'] if row['alias_of']}


# ============================================================================= C / D. the submission decision (one LLM request per case x arm)
def guidance_block(skill: dsk.Skill | None, packet: dict) -> dict:
    """The same `guidance` block a Fast request carries (br.Knowledge.render): loaded card or empty."""
    kn = dsk.skill_knowledge(skill)
    feats = {'batch_median_' + k: v['median'] for k, v in packet['overview']['summary'].items() if isinstance(v, dict) and v.get('median') is not None}
    return kn.render(feats, frozenset(ES.ALLOWED_FEATURES))


def decision_request(packet: dict, skill: dsk.Skill | None) -> dict:
    req = {k: packet[k] for k in PACKET_TOP_KEYS}
    req['guidance'] = guidance_block(skill, packet)
    req['output_contract'] = {'selected_candidate_id': 'one of candidate_ids', 'evidence_refs': 'non-empty list of dotted paths into this request', 'reason': 'text', 'uncertainty': 'text',
                              'guidance_applied': 'text or null', 'rules': 'exactly one JSON object; only fields present in this request may be cited'}
    return req


def _resolve(obj, path: str) -> bool:
    cur = obj
    for seg in str(path).split('.'):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        elif isinstance(cur, list) and seg.isdigit() and int(seg) < len(cur):
            cur = cur[int(seg)]
        else:
            return False
    return True


def parse_decision(resp, request: dict) -> dict:
    if not isinstance(resp, dict) or set(resp) != {'selected_candidate_id', 'evidence_refs', 'reason', 'uncertainty', 'guidance_applied'}:
        raise ValueError('output must be exactly {selected_candidate_id, evidence_refs, reason, uncertainty, guidance_applied}')
    cid = resp['selected_candidate_id']
    if cid not in request['candidate_ids']:
        raise ValueError('selected_candidate_id must be one of candidate_ids')
    refs = resp['evidence_refs']
    if not isinstance(refs, list) or not refs or any(not isinstance(r, str) for r in refs):
        raise ValueError('evidence_refs must be a non-empty list of strings')
    bad = [r for r in refs if not _resolve(request, r)]
    if bad:
        raise ValueError('evidence_refs not present in the request: %s' % bad[:5])
    for k in ('reason', 'uncertainty'):
        if not isinstance(resp[k], str) or not resp[k].strip():
            raise ValueError('%s must be nonempty text' % k)
    ga = resp['guidance_applied']
    if ga is not None and not isinstance(ga, str):
        raise ValueError('guidance_applied must be text or null')
    if not request['guidance']['loaded'] and ga:
        raise ValueError('guidance_applied must be null when no guidance is loaded')
    return {'selected_candidate_id': cid, 'evidence_refs': refs[:20], 'reason': resp['reason'][:3000], 'uncertainty': resp['uncertainty'][:2000], 'guidance_applied': (ga[:1500] if ga else None)}


def decide_one(request: dict, call) -> dict:
    """One submission decision; one format correction only; transport / budget faults -> DECISION_CALL_FAILED (resumable, never substituted)."""
    attempts, prior, receipts, payload = [], None, [], request
    for i in range(2):
        try:
            raw, rec = call(payload)
            receipts.append(rec)
        except (rt.llm.AccountFault, rt.llm.TransportFault, budget.BudgetExhausted, br.BudgetExhausted) as exc:
            attempts.append({'attempt': i, 'fault_kind': type(exc).__name__, 'detail': str(exc)[:200]})
            return {'status': 'DECISION_CALL_FAILED', 'attempts': attempts, 'receipts': receipts, 'decision': None}
        except ValueError as exc:
            attempts.append({'attempt': i, 'error': 'response not JSON: %s' % str(exc)[:200]})
            prior = None
        else:
            try:
                dec = parse_decision(raw, request)
                attempts.append({'attempt': i, 'ok': True})
                return {'status': 'DONE', 'attempts': attempts, 'receipts': receipts, 'decision': dec, 'raw': raw}
            except ValueError as exc:
                attempts.append({'attempt': i, 'error': str(exc)[:300]})
                prior = raw if dsk._jsonable(raw) else None
        if i == 0:
            payload = {**request, 'correction': {'previous_output': prior, 'error': attempts[-1].get('error'), 'instruction': 'Correct this format error only; keep your decision unless it named an id outside candidate_ids.'}}
    return {'status': 'DECISION_FORMAT_FAILED', 'attempts': attempts, 'receipts': receipts, 'decision': None}


def decision_stage(root: Path, stage: str, jobs: list, *, client_factory=None) -> dict:
    """jobs: [(case, arm, Skill|None)]; one request per (case, arm) in parallel through the HTTP pool; write-once decision files; resume reuses DONE / FORMAT_FAILED."""
    P_ = paths(root)
    role = 'select' if stage == 'select' else 'replay'
    sroot = P_[stage]
    sroot.mkdir(parents=True, exist_ok=True)
    pending = []
    for case, arm, skill in jobs:
        outp = sroot / case / ('%s.json' % arm)
        if outp.exists():
            prior = context.read_json(outp)
            if prior['status'] in ('DONE', 'DECISION_FORMAT_FAILED'):
                continue
            n_prev = len(list((sroot / case).glob('%s_failed_*.json' % arm)))
            outp.rename(sroot / case / ('%s_failed_%d.json' % (arm, n_prev + 1)))
        pending.append((case, arm, skill))
    if not pending:
        return stage_status(root, stage, jobs)
    led, client = open_stage_client(root, stage, client_factory=client_factory)
    lock = threading.Lock()
    progress = {'stage': stage, 'started_local': now(), 'pending': len(pending), 'done': 0, 'failed': 0}
    context.write_json(sroot / 'progress.json', progress)

    for case in sorted({c for c, _, _ in pending}):
        if not (sroot / case / 'request_common_bytes.json').exists():
            common = {k: load_packet(root, role, case)[k] for k in PACKET_TOP_KEYS}
            (sroot / case).mkdir(parents=True, exist_ok=True)
            dsks.write_once(sroot / case / 'request_common_bytes.json', {'bytes': len(json.dumps(common, ensure_ascii=False).encode('utf-8')), 'keys': list(PACKET_TOP_KEYS),
                                                                         'note': 'the common task part is byte-identical across arms of this case; only guidance differs'})

    def one(case, arm, skill):
        packet = load_packet(root, role, case)
        req = decision_request(packet, skill)
        res = decide_one(req, gated(lambda p: client.call('fast_submit', '%s_%s' % (case, arm), p, DECIDE_SYSTEM, max_tokens=MAX_OUTPUT_TOKENS,
                                                           meta={'case': case, 'arm': arm, 'call': 1}, request_timeout=DECIDE_REQUEST_TIMEOUT_S)))
        rec = {**res, 'case': case, 'arm': arm, 'skill_id': skill.skill_id if skill else None, 'guidance_loaded': bool(req['guidance']['loaded']),
               'common_bytes': len(json.dumps({k: req[k] for k in PACKET_TOP_KEYS}, ensure_ascii=False).encode('utf-8')), 'written_local': now()}
        if res['status'] == 'DECISION_CALL_FAILED':
            fk = res['attempts'][0].get('fault_kind')
            rec['note'] = ('never sent (blocked before dispatch by the unknown-usage / cap rule): resumable without any paid call' if fk == 'BudgetExhausted'
                           else 'sent and failed (%s): the failed attempts are unknown usage; the logical request is resumable after the operator decision' % fk)
        (sroot / case).mkdir(parents=True, exist_ok=True)
        dsks.write_once(sroot / case / ('%s.json' % arm), rec)
        with lock:
            progress['done' if res['status'] == 'DONE' else 'failed'] += 1
            progress['updated_local'] = now()
            progress['pools'] = DP.pools().snapshot()
            context.write_json(sroot / 'progress.json', progress)
        print('DECISION', stage, case, arm, res['status'], (res.get('decision') or {}).get('selected_candidate_id'), flush=True)
    errs = DP._parallel(pending, one)
    budget_warnings(root, stage)
    if errs:
        raise RuntimeError('decision thread failed: %s' % errs[0][1])
    st = stage_status(root, stage, jobs)
    if client.fatal or rt.unknown_usage_blocks(led):
        st['stop'] = {'fatal': client.fatal, 'unknown_usage_blocks': rt.unknown_usage_blocks(led)}
    return st


def stage_status(root: Path, stage: str, jobs: list) -> dict:
    sroot = paths(root)[stage]
    rows, missing = {}, []
    for case, arm, _ in jobs:
        p = sroot / case / ('%s.json' % arm)
        if not p.exists():
            missing.append('%s/%s' % (case, arm))
            continue
        r = context.read_json(p)
        rows['%s/%s' % (case, arm)] = r['status']
        if r['status'] == 'DECISION_CALL_FAILED':
            missing.append('%s/%s' % (case, arm))
    return {'stage': stage, 'status': 'FINISHED' if not missing else 'INCOMPLETE', 'decisions': rows, 'unfinished': missing}


# ============================================================================= C. Select: J, W*, Fixed_dev, H_select; freeze
def select_jobs(root: Path) -> list:
    jobs = []
    for d in DOMAINS:
        cards = cards_of(root, d)
        for case in all_cases('select', d):
            jobs.append((case, 's0', None))
            for arm in CARD_ARMS:
                if arm in cards:
                    jobs.append((case, arm, cards[arm]))
    return jobs


def e_of(binding: dict, cid: str) -> list:
    return list(binding['pool'][cid]['e']['by_seed'])


def decision_of(root: Path, stage: str, case: str, arm: str, aliases: dict) -> tuple:
    """(display id or None, resolved arm) following alias chains (NO_SUPPORTED_RULE / identical body -> s0 or an earlier card)."""
    seen, a = set(), arm
    while a in aliases and a not in seen:
        seen.add(a)
        a = aliases[a]
    p = paths(root)[stage] / case / ('%s.json' % a)
    if not p.exists():
        return None, a
    r = context.read_json(p)
    return ((r['decision'] or {}).get('selected_candidate_id') if r['status'] == 'DONE' else None), a


def r_ca_of(binding: dict) -> str:
    return _argmin(binding['pool'], 'c_a')


def _unit_tokens(root: Path, stage: str) -> dict:
    led = context.read_json(paths(root)['ledger'])
    out = {}
    for e in led.get('events', []):
        if e.get('kind') == 'llm_finished' and e.get('stage') == stage:
            u = out.setdefault(e['unit'], 0)
            out[e['unit']] = u + (e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0)
    return out


def select_domain(root: Path, domain: str) -> dict:
    """J(selector) = mean over the two Select cases of mean_seed E(selected) / mean_seed E(None); lower is better (task §6)."""
    P_ = paths(root)
    cases = all_cases('select', domain)
    bind = {c: load_binding(root, 'select', c) for c in cases}
    al = aliases_of(root, domain)
    cards = cards_of(root, domain)
    tokens = _unit_tokens(root, 'select')

    def J_of(choice_fn):
        ratios, picks = [], {}
        for c in cases:
            cid = choice_fn(c)
            if cid is None:
                return None, picks
            picks[c] = {'display_id': cid, 'label': bind[c]['pool'][cid]['label'], 'e_mean': bind[c]['pool'][cid]['e']['mean']}
            none_id = next(k for k, r in bind[c]['pool'].items() if r['public_id'] == 'None')
            ratios.append(bind[c]['pool'][cid]['e']['mean'] / bind[c]['pool'][none_id]['e']['mean'])
        return fmean(ratios), picks
    arms = {}
    for arm in ('s0',) + CARD_ARMS:
        if arm != 's0' and arm not in cards and arm not in al:
            arms[arm] = {'status': 'NO_CARD (technical failure)', 'J': None}
            continue
        j, picks = J_of(lambda c: decision_of(root, 'select', c, arm, al)[0])
        arms[arm] = {'status': 'alias_of:%s' % al[arm] if arm in al else ('COMPLETE' if j is not None else 'INCOMPLETE'), 'J': j, 'picks': picks,
                     'tokens': sum(tokens.get('%s_%s' % (c, arm), 0) for c in cases) if arm not in al else 0, 'alias_of': al.get(arm)}
    fixed = {}
    for pid in FIXED_ORDER:
        j, picks = J_of(lambda c: next(k for k, r in bind[c]['pool'].items() if r['public_id'] == pid))
        fixed[pid] = {'J': j, 'picks': picks}
    j_rca, picks_rca = J_of(lambda c: r_ca_of(bind[c]))
    # W*: lowest J among distinct cards; ties <= TOL: fewer Select tokens, then smaller slot; all aliases / failures -> s0 alias
    cand = [(a, arms[a]) for a in CARD_ARMS if arms[a].get('status') == 'COMPLETE' and arms[a]['J'] is not None]
    w_star = None
    if cand:
        best = min(x[1]['J'] for x in cand)
        tied = [x for x in cand if x[1]['J'] <= best + TOL]
        tied.sort(key=lambda x: (x[1]['tokens'], CARD_ARMS.index(x[0])))
        w_star = {'arm': tied[0][0], 'J': tied[0][1]['J'], 'tie_group': [x[0] for x in tied], 'tie_break': 'fewer Select tokens, then smaller slot' if len(tied) > 1 else None,
                  'skill_id': cards[tied[0][0]].skill_id}
    else:
        w_star = {'arm': 's0', 'J': arms['s0']['J'], 'tie_group': [], 'tie_break': None, 'skill_id': None, 'note': 'no distinct card (all NO_SUPPORTED_RULE / aliases / failures): S_W* is the no-card alias'}
    fbest = min(v['J'] for v in fixed.values())
    fixed_dev = next(pid for pid in FIXED_ORDER if fixed[pid]['J'] <= fbest + TOL)
    hs_cands = {'fixed_dev': fixed[fixed_dev]['J'], 'r_ca': j_rca, 's0': arms['s0']['J'], 's_w': w_star['J']}
    hbest = min(v for v in hs_cands.values() if v is not None)
    h_select = next(k for k in H_ORDER if hs_cands[k] is not None and hs_cands[k] <= hbest + TOL)
    return {'domain_id': domain, 'cases': cases, 'arms': arms, 'fixed_public': fixed, 'r_ca': {'J': j_rca, 'picks': picks_rca}, 'w_star': w_star,
            'fixed_dev': {'public_id': fixed_dev, 'J': fixed[fixed_dev]['J'], 'tie_order': list(FIXED_ORDER)},
            'h_select': {'choice': h_select, 'J_by_strategy': hs_cands, 'tie_order': list(H_ORDER), 'meaning': 'a system-level adoption reading only; Replay reuses the chosen strategy\'s output as an alias'},
            'J_definition': 'mean over the two Select cases of mean_seed E(selected) / mean_seed E(None); lower is better; ties <= 1e-12'}


def freeze_domain(root: Path, domain: str) -> dict:
    P_ = paths(root)
    p = P_['freeze'] / ('%s.json' % domain)
    if p.exists():
        return context.read_json(p)
    sel = select_domain(root, domain)
    if sel['arms']['s0']['status'] != 'COMPLETE':
        raise RuntimeError('Select of %s incomplete for s0' % domain)
    cards = cards_of(root, domain)
    w = sel['w_star']
    rec = {**sel, 'frozen_local': now(), 'w_star_card': cards[w['arm']].to_json() if w['arm'] in cards else None, 'select_labels_returned_to_slow': False,
           'decide_system_prompt': DECIDE_SYSTEM, 'packet_top_level_keys': list(PACKET_TOP_KEYS)}
    dsks.write_once(p, rec)
    return rec


def replay_jobs(root: Path) -> list:
    jobs = []
    for d in DOMAINS:
        fr = context.read_json(paths(root)['freeze'] / ('%s.json' % d))
        skill = dsk.skill_from_json(fr['w_star_card']) if fr.get('w_star_card') else None
        for case in all_cases('replay', d):
            jobs.append((case, 's0', None))
            if skill is not None:
                jobs.append((case, 's_w', skill))
    return jobs


# ============================================================================= readout (task §10)
def _e_uniform(binding: dict) -> list:
    pool = binding['pool']
    return [fmean(pool[c]['e']['by_seed'][i] for c in pool) for i in range(len(SEEDS))]


def replay_case_record(root: Path, case: str, fr: dict) -> dict:
    b = load_binding(root, 'replay', case)
    pool = b['pool']
    none_id = next(c for c, r in pool.items() if r['public_id'] == 'None')
    nomix_id = next(c for c, r in pool.items() if r['public_id'] == 'P_NoMixRecipe')
    den = pool[none_id]['e']['mean']
    fixed_id = next(c for c, r in pool.items() if r['public_id'] == fr['fixed_dev']['public_id'])
    rca = r_ca_of(b)
    s0, _ = decision_of(root, 'replay', case, 's0', {})
    w_arm = fr['w_star']['arm']
    if w_arm == 's0':
        sw, sw_alias = s0, 's0'
    else:
        sw, sw_alias = decision_of(root, 'replay', case, 's_w', {})[0], None
    h = fr['h_select']['choice']
    h_id = {'fixed_dev': fixed_id, 'r_ca': rca, 's0': s0, 's_w': sw}[h]
    strategies = {'s_w': sw, 's0': s0, 'r_ca': rca, 'fixed_dev': fixed_id, 'none': none_id, 'nomix': nomix_id, 'h_select': h_id}
    E = {k: (e_of(b, v) if v else None) for k, v in strategies.items()}
    E['r_uniform'] = _e_uniform(b)
    ranks = sorted(pool, key=lambda c: pool[c]['e']['mean'])
    e_best = ranks[0]
    E_best = e_of(b, e_best)
    rec = {'case': case, 'domain': case[:3], 'pool_size': len(pool), 'strategies': strategies, 's_w_alias_of': sw_alias, 'e_by_seed': E, 'den_none': den,
           'labels': {k: (pool[v]['label'] if v else None) for k, v in strategies.items()},
           'g': {'s_w_over_' + k: G(E['s_w'], E[k], den) for k in ('s0', 'r_ca', 'fixed_dev', 'r_uniform', 'none', 'nomix')},
           'g_s0': {'s0_over_' + k: G(E['s0'], E[k], den) for k in ('r_ca', 'fixed_dev', 'r_uniform', 'none', 'nomix')},
           'g_h': {'h_select_over_' + k: G(E['h_select'], E[k], den) for k in ('s0', 's_w', 'r_ca', 'fixed_dev', 'none', 'nomix')},
           'g_fixed_over_none': {k: G(E[k], E['none'], den)['mean'] for k in ('r_ca', 'fixed_dev', 'nomix', 'r_uniform', 's0', 's_w', 'h_select') if E[k]},
           'same_delivery': {'s_w_vs_s0': sw == s0, 's_w_vs_r_ca': sw == rca, 's_w_vs_fixed_dev': sw == fixed_id, 's0_vs_r_ca': s0 == rca, 's0_vs_fixed_dev': s0 == fixed_id},
           'opportunity': {'pool_e_best': e_best, 'pool_e_best_label': pool[e_best]['label'], 'pool_e_best_over_none_pp': G(E_best, E['none'], den)['mean'],
                           'e_rank_in_pool': {k: (ranks.index(v) + 1 if v else None) for k, v in strategies.items()},
                           'regret_pp': {k: (G(E_best, E[k], den)['mean'] if E[k] else None) for k in ('s_w', 's0', 'r_ca', 'fixed_dev', 'r_uniform', 'h_select')},
                           'net_vs_nomix_pp': {k: (G(E[k], E['nomix'], den)['mean'] if E[k] else None) for k in ('s_w', 's0', 'r_ca', 'fixed_dev', 'r_uniform', 'h_select')},
                           'candidate_opportunity_over_nomix_pp': G(E_best, E['nomix'], den)['mean']}}
    for arm in ('s0', 's_w'):
        p = paths(root)['replay'] / case / ('%s.json' % arm)
        if p.exists():
            r = context.read_json(p)
            rec.setdefault('decisions', {})[arm] = {'status': r['status'], 'selected': (r['decision'] or {}).get('selected_candidate_id'), 'label': pool[(r['decision'] or {}).get('selected_candidate_id')]['label'] if r['status'] == 'DONE' else None,
                                                    'evidence_refs': (r['decision'] or {}).get('evidence_refs'), 'reason': (r['decision'] or {}).get('reason'), 'uncertainty': (r['decision'] or {}).get('uncertainty'),
                                                    'guidance_applied': (r['decision'] or {}).get('guidance_applied'), 'attempts': r['attempts'],
                                                    'tokens': sum((x.get('prompt_tokens') or 0) + (x.get('completion_tokens') or 0) for x in r.get('receipts', []))}
    return rec


def _agg(records: list, key_path: tuple) -> dict:
    """Equal-weight over cases: mean of per-case means; seed-paired SE from per-seed case means; win/tie/loss; LOO range."""
    vals = []
    for r in records:
        g = r
        for k in key_path:
            g = g.get(k) if isinstance(g, dict) else None
            if g is None:
                break
        if g is not None:
            vals.append(g)
    if not vals:
        return {'n': 0}
    per_seed = [fmean(v['per_seed'][i] for v in vals) for i in range(len(SEEDS))]
    means = [v['mean'] for v in vals]
    loo = [fmean(means[:i] + means[i + 1:]) for i in range(len(means))] if len(means) > 1 else means
    return {'n': len(vals), 'mean': fmean(means), 'seed_se_of_case_mean': se3(per_seed), 'per_seed_case_mean': per_seed, 'case_means': means,
            'wins': sum(1 for m in means if m > TOL), 'ties': sum(1 for m in means if abs(m) <= TOL), 'losses': sum(1 for m in means if m < -TOL),
            'max_harm': min(means), 'max_gain': max(means), 'loo_range': [min(loo), max(loo)], 'case_sd': (statistics.stdev(means) if len(means) > 1 else None)}


def readout(root: Path = ROOT) -> dict:
    P_ = paths(root)
    out = {'package': PACKAGE, 'exposure': EXPOSURE, 'written_local': now(), 'task': TASK, 'frozen': {}, 'select': {}, 'replay': {}, 'summary': {}, 'cost': cost_readout(root)}
    out['slow'] = context.read_json(P_['formation'] / 'proposals.json') if (P_['formation'] / 'proposals.json').exists() else None
    for d in DOMAINS:
        fp = P_['freeze'] / ('%s.json' % d)
        if not fp.exists():
            out['frozen'][d] = {'status': 'NOT_FROZEN'}
            continue
        fr = context.read_json(fp)
        out['frozen'][d] = {k: fr[k] for k in ('w_star', 'fixed_dev', 'h_select', 'arms', 'fixed_public', 'r_ca')}
        out['select'][d] = {'cases': {}}
        for c in fr['cases']:
            b = load_binding(root, 'select', c)
            row = {'pool_size': b['pool_size'], 'picks': {}}
            for arm in ('s0',) + CARD_ARMS:
                if arm in fr['arms'] and fr['arms'][arm].get('picks', {}).get(c):
                    row['picks'][arm] = fr['arms'][arm]['picks'][c]
            row['r_ca'] = fr['r_ca']['picks'].get(c)
            row['e_ranking'] = sorted(b['pool'], key=lambda k: b['pool'][k]['e']['mean'])
            row['labels'] = {k: r['label'] for k, r in b['pool'].items()}
            out['select'][d]['cases'][c] = row
        recs = []
        replay_done = all((P_['replay'] / c / 's0.json').exists() for c in all_cases('replay', d))
        if replay_done:
            for c in all_cases('replay', d):
                recs.append(replay_case_record(root, c, fr))
        out['replay'][d] = {'cases': {r['case']: r for r in recs}, 'complete': replay_done and all(r['e_by_seed']['s0'] and r['e_by_seed']['s_w'] for r in recs)}
        if recs:
            out['replay'][d]['aggregate'] = {k: _agg(recs, ('g', k)) for k in ('s_w_over_s0', 's_w_over_r_ca', 's_w_over_fixed_dev', 's_w_over_r_uniform', 's_w_over_none', 's_w_over_nomix')}
            out['replay'][d]['aggregate_s0'] = {k: _agg(recs, ('g_s0', k)) for k in ('s0_over_r_ca', 's0_over_fixed_dev', 's0_over_r_uniform', 's0_over_none', 's0_over_nomix')}
            out['replay'][d]['aggregate_h'] = {k: _agg(recs, ('g_h', k)) for k in ('h_select_over_s0', 'h_select_over_s_w', 'h_select_over_r_ca', 'h_select_over_fixed_dev', 'h_select_over_none', 'h_select_over_nomix')}
            out['replay'][d]['same_delivery_counts'] = {k: sum(1 for r in recs if r['same_delivery'][k]) for k in recs[0]['same_delivery']}
            out['replay'][d]['over_none_by_strategy'] = {k: fmean(r['g_fixed_over_none'][k] for r in recs if k in r['g_fixed_over_none']) for k in ('s_w', 's0', 'r_ca', 'fixed_dev', 'r_uniform', 'nomix', 'h_select')}
    both = [d for d in DOMAINS if out['replay'].get(d, {}).get('aggregate') and all(v.get('n') for sub in ('aggregate', 'aggregate_s0', 'aggregate_h') for v in out['replay'][d][sub].values())]
    if both:
        def two(key, sub='aggregate'):
            per = {d: out['replay'][d][sub][key] for d in both}
            return {'mean': fmean(v['mean'] for v in per.values()), 'by_domain': {d: v['mean'] for d, v in per.items()},
                    'wins_ties_losses': [sum(v[k] for v in per.values()) for k in ('wins', 'ties', 'losses')], 'max_harm': min(v['max_harm'] for v in per.values()),
                    'seed_se_by_domain': {d: v['seed_se_of_case_mean'] for d, v in per.items()}}
        out['summary'] = {'s_w_over_s0': two('s_w_over_s0'), 's_w_over_r_ca': two('s_w_over_r_ca'), 's_w_over_fixed_dev': two('s_w_over_fixed_dev'), 's_w_over_r_uniform': two('s_w_over_r_uniform'),
                          's_w_over_none': two('s_w_over_none'), 's_w_over_nomix': two('s_w_over_nomix'),
                          's0_over_r_ca': two('s0_over_r_ca', 'aggregate_s0'), 's0_over_fixed_dev': two('s0_over_fixed_dev', 'aggregate_s0'), 's0_over_r_uniform': two('s0_over_r_uniform', 'aggregate_s0'),
                          's0_over_none': two('s0_over_none', 'aggregate_s0'), 's0_over_nomix': two('s0_over_nomix', 'aggregate_s0'),
                          'h_select_over_s0': two('h_select_over_s0', 'aggregate_h'), 'h_select_over_s_w': two('h_select_over_s_w', 'aggregate_h'), 'h_select_over_fixed_dev': two('h_select_over_fixed_dev', 'aggregate_h'),
                          'over_none_by_strategy': {k: fmean(out['replay'][d]['over_none_by_strategy'][k] for d in both) for k in ('s_w', 's0', 'r_ca', 'fixed_dev', 'r_uniform', 'nomix', 'h_select')},
                          'units': 'pp of the case None mean; positive = the first strategy is better; domain = equal weight over its four Replay cases; overall = equal weight over the two domains',
                          'complete': all(out['replay'][d]['complete'] for d in DOMAINS)}
        out['summary']['verdict'] = verdict(out)
    context.write_json(root / 'result.json', out)
    (root / 'tables.md').write_text(tables(out), encoding='utf-8')
    return out


def verdict(res: dict) -> dict:
    s = res['summary']
    a, b, c = s['s_w_over_s0'], s['s_w_over_r_ca'], s['s_w_over_fixed_dev']
    wtl = a['wins_ties_losses']
    if wtl[0] >= 2 and a['mean'] > 0 and wtl[2] == 0:
        kind = 'CARD_REALIZED_BETTER_SELECTION_ON_SEVERAL_CASES'
    elif a['mean'] > 0 and wtl[0] > wtl[2]:
        kind = 'WEAK_POSITIVE_MIXED'
    elif abs(a['mean']) <= 0.05 and wtl[0] == 0 and wtl[2] == 0:
        kind = 'CARD_EQUALS_NO_CARD_DELIVERY'
    elif a['mean'] <= 0:
        kind = 'NO_INCREMENT_OR_HARM'
    else:
        kind = 'INCONCLUSIVE'
    return {'card_vs_no_card': kind, 'card_vs_ca_argmin_mean_pp': b['mean'], 'card_vs_fixed_dev_mean_pp': c['mean'], 'no_card_vs_ca_argmin_mean_pp': s['s0_over_r_ca']['mean'],
            'note': 'mechanical label from the equal-weight means and win/tie/loss counts; the report reads the per-case tables, not this label alone'}


def cost_readout(root: Path) -> dict:
    P_ = paths(root)
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    ev = led.get('events', [])
    by_stage = {}
    for e in ev:
        if e.get('kind') == 'llm_finished':
            r = by_stage.setdefault(e.get('stage'), {'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0})
            r['requests'] += 1
            r['prompt_tokens'] += e.get('prompt_tokens') or 0
            r['completion_tokens'] += e.get('completion_tokens') or 0
    hist = {}
    for name, r in (('entity_split(parent: Learn/Select pools)', ES.ROOT), ('decision_priority(parent: Replay pools)', DP.ROOT)):
        p = r / 'budget.json'
        if p.exists():
            l2 = context.read_json(p)
            hist[name] = {'fit_attempts': l2.get('fit_attempts'), 'fits_ok': l2.get('fits_ok'), 'cache_hits': l2.get('cache_hits'), 'llm_requests': l2.get('llm_requests'),
                          'tokens': l2.get('llm_tokens_in', 0) + l2.get('llm_tokens_out', 0), 'note': 'whole parent package (all arms / stages), read from its ledger; the pools reused here are a subset'}
    return {'this_package': {'llm_requests': led.get('llm_requests', 0), 'http_attempts': led.get('llm_http_attempts', 0), 'tokens_in': led.get('llm_tokens_in', 0), 'tokens_out': led.get('llm_tokens_out', 0),
                             'unknown_usage': led.get('llm_tokens_unknown', 0), 'unknown_accepted': led.get('unknown_usage_accepted', 0), 'failed_attempts': led.get('llm_failed_attempts', 0),
                             'fit_attempts': led.get('fit_attempts', 0), 'by_stage': by_stage,
                             'paid_wall_s': (time.time() - led['paid_clock_started_epoch']) if led.get('paid_clock_started_epoch') and not led.get('paid_clock_stopped_epoch') else led.get('paid_wall_s'),
                             'paid_clock_started_local': led.get('paid_clock_started_local')},
            'historical_pool_generation': hist, 'note': 'new fits 0; the fixed strategies R_CA / Fixed_dev / R_uniform cost 0 LLM here; the common pool cost belongs to no arm'}


def _f(x, nd=2):
    return '-' if x is None else ('%.*f' % (nd, x))


def tables(res: dict) -> str:
    L = ['# %s — tables (%s)' % (PACKAGE, res['written_local']), '', 'Units: pp of the case None mean; positive = the first strategy is better; domain = equal weight over its four Replay cases; overall = equal weight over the two domains.', '']
    if res.get('summary'):
        s = res['summary']
        L += ['## Main table (Replay, S_W* against each strategy)', '', '| comparison | overall | D01 | D02 | wins/ties/losses | max harm (case) |', '|---|---:|---:|---:|---|---:|']
        for k in ('s_w_over_s0', 's_w_over_r_ca', 's_w_over_fixed_dev', 's_w_over_r_uniform', 's_w_over_none', 's_w_over_nomix'):
            v = s[k]
            L.append('| %s | %s | %s | %s | %s | %s |' % (k, _f(v['mean']), _f(v['by_domain'].get('D01')), _f(v['by_domain'].get('D02')), '/'.join(map(str, v['wins_ties_losses'])), _f(v['max_harm'])))
        L += ['', '## No-card and H_select', '', '| comparison | overall | D01 | D02 | wins/ties/losses |', '|---|---:|---:|---:|---|']
        for k in ('s0_over_r_ca', 's0_over_fixed_dev', 's0_over_r_uniform', 's0_over_none', 's0_over_nomix', 'h_select_over_s0', 'h_select_over_s_w', 'h_select_over_fixed_dev'):
            v = s[k]
            L.append('| %s | %s | %s | %s | %s |' % (k, _f(v['mean']), _f(v['by_domain'].get('D01')), _f(v['by_domain'].get('D02')), '/'.join(map(str, v['wins_ties_losses']))))
        L += ['', '## Each strategy over None (pp)', '', '| strategy | overall | D01 | D02 |', '|---|---:|---:|---:|']
        for k, v in s['over_none_by_strategy'].items():
            L.append('| %s | %s | %s | %s |' % (k, _f(v), _f(res['replay']['D01']['over_none_by_strategy'].get(k)), _f(res['replay']['D02']['over_none_by_strategy'].get(k))))
    for d in DOMAINS:
        fr = res['frozen'].get(d, {})
        if fr.get('w_star'):
            L += ['', '## %s frozen' % d, '', '- W* = %s (J %s; tie group %s; %s)' % (fr['w_star']['arm'], _f(fr['w_star']['J'], 4), fr['w_star'].get('tie_group'), fr['w_star'].get('note') or fr['w_star'].get('tie_break') or 'unique'),
                  '- Fixed_dev = %s (J %s); R_CA J %s; S0 J %s' % (fr['fixed_dev']['public_id'], _f(fr['fixed_dev']['J'], 4), _f(fr['r_ca']['J'], 4), _f(fr['arms']['s0']['J'], 4)),
                  '- H_select = %s (J by strategy: %s)' % (fr['h_select']['choice'], {k: _f(v, 4) for k, v in fr['h_select']['J_by_strategy'].items()}),
                  '- Select arms: ' + '; '.join('%s: %s J %s tokens %s' % (a, v.get('status'), _f(v.get('J'), 4), v.get('tokens')) for a, v in fr['arms'].items())]
        rp = res['replay'].get(d, {})
        if rp.get('cases'):
            L += ['', '## %s Replay: opportunity and realization' % d, '', '| case | pool | E best (label) | best over None | S_W* pick (rank, regret) | S0 pick (rank, regret) | R_CA (rank) | Fixed_dev (rank) | R_uniform regret | S_W* over S0 | net S_W* vs NoMix |',
                  '|---|---:|---|---:|---|---|---|---|---:|---:|---:|']
            for c, r in rp['cases'].items():
                o = r['opportunity']
                L.append('| %s | %d | %s (%s) | %s | %s (%s, %s) | %s (%s, %s) | %s (%s) | %s (%s) | %s | %s | %s |' % (
                    c, r['pool_size'], o['pool_e_best'], o['pool_e_best_label'], _f(o['pool_e_best_over_none_pp']),
                    r['labels']['s_w'], o['e_rank_in_pool']['s_w'], _f(o['regret_pp']['s_w']), r['labels']['s0'], o['e_rank_in_pool']['s0'], _f(o['regret_pp']['s0']),
                    r['labels']['r_ca'], o['e_rank_in_pool']['r_ca'], r['labels']['fixed_dev'], o['e_rank_in_pool']['fixed_dev'], _f(o['regret_pp']['r_uniform']),
                    _f(r['g']['s_w_over_s0']['mean']) if r['g']['s_w_over_s0'] else '-', _f(o['net_vs_nomix_pp']['s_w'])))
            L += ['', '## %s Replay: rule -> evidence -> delivery' % d, '', '| case | S0 pick | S_W* pick | different | S_W* over S0 (per seed) | guidance applied (S_W*) | S_W* evidence refs | S0 evidence refs |', '|---|---|---|---|---|---|---|---|']
            for c, r in rp['cases'].items():
                dd = r.get('decisions', {})
                sw, s0 = dd.get('s_w', {}), dd.get('s0', {})
                g = r['g']['s_w_over_s0']
                L.append('| %s | %s | %s | %s | %s (%s) | %s | %s | %s |' % (c, s0.get('label'), sw.get('label') or (r['labels']['s_w'] + ' (alias s0)' if r['s_w_alias_of'] else None),
                                                                        'no' if r['same_delivery']['s_w_vs_s0'] else 'YES', _f(g['mean']) if g else '-', ','.join(_f(x) for x in g['per_seed']) if g else '-',
                                                                        (sw.get('guidance_applied') or '-').replace('|', '/')[:160], '; '.join(sw.get('evidence_refs') or [])[:160], '; '.join(s0.get('evidence_refs') or [])[:160]))
    c = res.get('cost', {}).get('this_package', {})
    L += ['', '## Cost', '', '- this package: %s logical requests, %s HTTP attempts, %s tokens in / %s out, unknown usage %s (accepted %s), fits %s, paid wall %s' % (
        c.get('llm_requests'), c.get('http_attempts'), c.get('tokens_in'), c.get('tokens_out'), c.get('unknown_usage'), c.get('unknown_accepted'), c.get('fit_attempts'),
        ('%.1f min' % (c['paid_wall_s'] / 60)) if c.get('paid_wall_s') else '-'), '- by stage: %s' % json.dumps(c.get('by_stage'), ensure_ascii=False),
          '- historical pool generation (whole parent packages): %s' % json.dumps(res.get('cost', {}).get('historical_pool_generation'), ensure_ascii=False)]
    return '\n'.join(L) + '\n'


# ============================================================================= driver
def preflight(root: Path = ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    p = root / 'frozen_config.json'
    if p.exists():
        return context.read_json(p)
    idx = build_clean_inputs(root)
    tables_ = {d: learn_evidence(root, d) for d in DOMAINS}
    sizes = {}
    for d in DOMAINS:
        nb = payload_bytes(SLOW_FP, slow_payload(tables_[d], d, 1))
        sizes[d] = {'bytes': nb, 'estimated_prompt_tokens': int(nb / SLOW_BYTES_PER_TOKEN), 'n_legal_refs': len(tables_[d]['legal_evidence_refs'])}
    pk = {}
    for role in ('select', 'replay'):
        for c in all_cases(role):
            req = decision_request(load_packet(root, role, c), None)
            nb = payload_bytes(DECIDE_SYSTEM, req)
            pk['%s/%s' % (role, c)] = {'bytes': nb, 'estimated_prompt_tokens': int(nb / SLOW_BYTES_PER_TOKEN)}
    cfg = {'package': PACKAGE, 'task': TASK, 'exposure': EXPOSURE, 'frozen_local': now(), 'model': MODEL, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'body_limit': BODY_LIMIT,
           'roles': {r: {'root': str(v[0]), 'tails': list(v[1]), 'roster_doc': v[2]} for r, v in ROLES.items()}, 'seeds': list(SEEDS), 'public': list(PUBLIC),
           'total_caps': TOTAL, 'http_cap': HTTP_CAP, 'transport_retries': TRANSPORT_RETRIES, 'alloc': ALLOC, 'concurrency': CONCURRENCY, 'token_warnings': TOKEN_WARNINGS,
           'paid_wall_warnings_s': PAID_WALL_WARNINGS_S, 'planned_main_requests': {'slow': 6, 'select': 16, 'replay': 16, 'total': 38}, 'max_format_corrections': 38,
           'slow_payloads': sizes, 'slow_input_token_target': SLOW_INPUT_TOKEN_TARGET, 'decision_payloads': pk, 'clean_inputs': idx, 'decide_system': DECIDE_SYSTEM, 'slow_system': SLOW_FP,
           'j_definition': 'mean over the two Select cases of mean_seed E(selected) / mean_seed E(None); lower is better; ties <= 1e-12 then fewer Select tokens then smaller slot',
           'fixed_dev_rule': 'lowest J among the four public references on the two Select cases; ties in the order None, FixedMixup, P_AmpResample, P_NoMixRecipe',
           'h_select_rule': 'lowest Select J among Fixed_dev, R_CA, S0, S_W*; ties in that order; Replay alias only',
           'g_definition': '100 x [mean_seed E(B) - mean_seed E(A)] / mean_seed E(None); domain equal weight over four Replay cases; overall equal weight over two domains',
           'environment': {'python': sys.executable, 'platform': sys.platform}, 'no_new': ['fits', 'predictions', 'labels', 'hashes', 'git commits']}
    dsks.write_once(p, cfg)
    return cfg


def package_run(root: Path = ROOT, *, accept_unknown: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    lock = DP.PackageLock(root, 'driver').acquire()
    context.write_json(root / 'driver_launch.json', {'pid': os.getpid(), 'epoch': time.time(), 'local': now()})
    DP.set_pools(CONCURRENCY['numeric'], CONCURRENCY['http'])
    try:
        if accept_unknown:
            accept_unknown_usage(root, 'driver_start')
        status = context.read_json(root / 'package_status.json') if (root / 'package_status.json').exists() else {}

        def stop(where, st):
            status.update({'stopped_at': where, 'stage_status': st, 'epoch': time.time(), 'local': now()})
            context.write_json(root / 'package_status.json', status)
            print('PACKAGE_STOP', where, json.dumps(st, ensure_ascii=False)[:400], flush=True)
            return status
        preflight(root)
        w = wiring(root)
        status['wiring'] = w['status']
        if w['status'] != 'PASS':
            return stop('wiring', w)
        try:
            sl = slow_stage(root)
        except RuntimeError as exc:
            return stop('slow', {'status': 'STOPPED', 'error': str(exc)[:600]})
        status['slow'] = {d: [(r['arm'], r['status'], r['alias_of']) for r in sl['domains'][d]['proposals']] for d in DOMAINS}
        context.write_json(root / 'package_status.json', status)
        try:
            st = decision_stage(root, 'select', select_jobs(root))
        except RuntimeError as exc:
            return stop('select', {'status': 'STOPPED', 'error': str(exc)[:600]})
        status['select'] = st
        if st['status'] != 'FINISHED':
            return stop('select', st)
        for d in DOMAINS:
            fr = freeze_domain(root, d)
            status['frozen_%s' % d] = {'w_star': fr['w_star']['arm'], 'fixed_dev': fr['fixed_dev']['public_id'], 'h_select': fr['h_select']['choice']}
        context.write_json(root / 'package_status.json', status)
        try:
            st = decision_stage(root, 'replay', replay_jobs(root))
        except RuntimeError as exc:
            return stop('replay', {'status': 'STOPPED', 'error': str(exc)[:600]})
        status['replay'] = st
        if st['status'] != 'FINISHED':
            return stop('replay', st)
        led = DP.SafeLedger(paths(root)['ledger'], **TOTAL)
        if led.s.get('paid_clock_started_epoch') and not led.s.get('paid_clock_stopped_epoch'):
            led.s['paid_clock_stopped_epoch'] = time.time()
            led.s['paid_wall_s'] = led.s['paid_clock_stopped_epoch'] - led.s['paid_clock_started_epoch']
            led._save()
        status['finished_epoch'] = time.time()
        status['finished_local'] = now()
        context.write_json(root / 'package_status.json', status)
        readout(root)
        print('PACKAGE_FINISHED', flush=True)
        return status
    finally:
        lock.release()


# ============================================================================= wiring (0 LLM): the cached numbers reproduce the parent's readout
def wiring(root: Path = ROOT) -> dict:
    P_ = paths(root)
    p = P_['wiring'] / 'wiring.json'
    if p.exists():
        return context.read_json(p)
    build_clean_inputs(root)
    parent = context.read_json(ES.ROOT / 'result.json')['source']
    rows, ok = {}, True
    for case in WIRING_CASES:
        b = load_binding(root, 'learn', case)
        pr = parent[case]
        rca = r_ca_of(b)
        rca_plan = b['pool'][rca]['plan_ids']
        e_ratio = {}
        for cid, r in b['pool'].items():
            none = next(k for k, v in b['pool'].items() if v['public_id'] == 'None')
            e_ratio[r['public_id'] or r['plan_ids'][0]] = r['e']['mean'] / b['pool'][none]['e']['mean']
        same_ratio = all(abs(e_ratio[k] - pr['pool_e_ratio_to_none'][k]) <= 1e-9 for k in pr['pool_e_ratio_to_none'])
        same_arg = pr['c_a_argmin'] in rca_plan
        packet = load_packet(root, 'learn', case)
        ca_disp = packet['candidates'][rca]['ca']['mean']
        rows[case] = {'r_ca_display': rca, 'r_ca_plan_ids': rca_plan, 'parent_c_a_argmin': pr['c_a_argmin'], 'r_ca_matches_parent_shadow_argmin': same_arg,
                      'pool_e_ratio_matches_parent': same_ratio, 'pool_size': b['pool_size'], 'parent_pool_size': len(pr['pool_e_ratio_to_none']),
                      'display_ca_mean_matches_cache_8sig': abs(ca_disp - b['pool'][rca]['c_a']['mean']) <= 1e-7 * max(1.0, abs(ca_disp)),
                      'f0_commit_display': b['f0_commit_display_id'], 'parent_committed': pr['committed'], 'f0_commit_matches': pr['committed'] in b['pool'][b['f0_commit_display_id']]['plan_ids']}
        ok = ok and same_arg and same_ratio and rows[case]['display_ca_mean_matches_cache_8sig'] and rows[case]['f0_commit_matches'] and b['pool_size'] == len(pr['pool_e_ratio_to_none'])
    led = context.read_json(P_['ledger']) if P_['ledger'].exists() else {}
    out = {'status': 'PASS' if ok else 'FAIL', 'cases': rows, 'fits': led.get('fit_attempts', 0), 'llm_requests': led.get('llm_requests', 0), 'checked_local': now(),
           'note': 'R_CA equals the parent\'s shadow C_A argmin; pool E ratios equal the parent readout; 0 fits, 0 predictions, 0 LLM'}
    dsks.write_once(p, out)
    return out


# ============================================================================= smoke (fake client only; never the production client)
class FakeSubmitClient(DP.FakeMeteredClient):
    """Deterministic transport: Slow -> a valid PROPOSE (or NO_SUPPORTED_RULE for slot 3 of D02), submit -> the candidate with the lowest ca.mean unless a card is loaded (then the preset)."""

    def __init__(self, ledger, out, *, http_cap, stage, latency=0.02, fail_requests=()):
        super().__init__(ledger, out, http_cap=http_cap, stage=stage, latency=latency, fail_requests=fail_requests)

    def _transport(self, messages, max_tokens, timeout):
        time.sleep(self.latency)
        n = len(self.sent) + 1
        self.sent.append(n)
        if n in self.fail_requests:
            raise ConnectionError('smoke transient 502 bad gateway')
        payload = json.loads(messages[1]['content'])
        pt = len(messages[1]['content']) // 4
        if 'proposal_slot' in payload:
            legal = payload['legal_evidence_refs']
            cref = next(r for r in legal if r.count('/') == 2 and r.split('/')[2].startswith('C'))
            if payload['domain_id'] == 'D02' and payload['proposal_slot'] == 3:
                text = json.dumps({'decision': 'NO_SUPPORTED_RULE', 'rationale': 'smoke', 'evidence_review': {'supports': [], 'contradicts_or_limits': [], 'rule_changes': [], 'uncertain_and_expected_behavior_change': 'smoke'}})
            else:
                wf = 'Read the paired C_A differences to the preset and to no augmentation. Keep the preset when its paired mean delta to no augmentation is positive on both origins; otherwise submit the candidate with the lowest C_A mean. (smoke slot %d)' % payload['proposal_slot']
                text = json.dumps({'decision': 'PROPOSE', 'candidate': {'research_mode': 'smoke', 'workflow': wf, 'principles': None, 'observable_applicability': {'const': True},
                                                                        'applicability_summary': 'smoke', 'evidence_refs': [cref], 'rationale': 'smoke',
                                                                        'evidence_review': {'supports': ['x'], 'contradicts_or_limits': [], 'rule_changes': [], 'uncertain_and_expected_behavior_change': 'x'},
                                                                        'rule_status': [{'rule': 'keep preset when positive', 'status': 'hypothesis', 'support': [cref], 'counter': []}],
                                                                        'commit_policy': 'as in workflow', 'preference_nature': 'conditional',
                                                                        'expected_decision_example': {'ref': cref, 'expected_choice_and_why': 'x'}, 'counterexample_expected': {'ref': cref, 'should_hold_or_may_fail': 'x'}}})
            return DP._FakeResponse(text, pt, 50)
        cands = payload['candidates']
        if payload['guidance']['loaded']:
            pick = payload['reference_ids']['P_NoMixRecipe']
        else:
            pick = min(cands, key=lambda c: (cands[c]['ca']['mean'], c))
        text = json.dumps({'selected_candidate_id': pick, 'evidence_refs': ['candidates.%s.ca.by_seed_origin' % pick, 'overview.summary.lag24_corr'], 'reason': 'smoke', 'uncertainty': 'smoke',
                           'guidance_applied': ('smoke rule' if payload['guidance']['loaded'] else None)})
        return DP._FakeResponse(text, pt, 20)


class FakeBadFormatClient(FakeSubmitClient):
    """First answer names an illegal id; the correction answers legally (tests the single format correction)."""

    def _transport(self, messages, max_tokens, timeout):
        payload = json.loads(messages[1]['content'])
        if 'proposal_slot' not in payload and 'correction' not in payload:
            time.sleep(self.latency)
            self.sent.append(len(self.sent) + 1)
            return DP._FakeResponse(json.dumps({'selected_candidate_id': 'C99', 'evidence_refs': ['nope'], 'reason': 'x', 'uncertainty': 'x', 'guidance_applied': None}), 10, 5)
        return super()._transport(messages, max_tokens, timeout)


def smoke(root: Path = ROOT) -> dict:
    """Task §8 minimum smoke on a scratch root (real read-only caches, fake client): pool eligibility / dedupe; no C_B / E pass-through; Learn whitelist; same input /
    different card; legal id + cached scoring; receipts / duplicate start / resume; unfinished decisions not connected."""
    import shutil
    import subprocess
    sroot = paths(root)['smoke'] / 'run'
    if sroot.exists():
        shutil.rmtree(sroot)
    sroot.mkdir(parents=True)
    checks = {}
    DP.set_pools(1, 4)
    # 1. pool eligibility and dedupe on real caches
    idx = build_clean_inputs(sroot)
    b = load_binding(sroot, 'learn', 'D01_S01')
    checks['pool_excludes_built_never_evaluated'] = all(m not in sum((r['plan_ids'] for r in b['pool'].values()), []) for m in b['built_never_evaluated']) and bool(b['built_never_evaluated'])
    checks['pool_has_four_public'] = sorted(r['public_id'] for r in b['pool'].values() if r['public_id']) == sorted(PUBLIC)
    checks['pool_dedupe_by_phys'] = len({r['phys'] for r in b['pool'].values()}) == len(b['pool'])
    checks['pool_sizes_in_range'] = all(6 <= v['pool_size'] <= 8 for role in idx['roles'].values() for v in role.values())
    # 2. no C_B / E pass-through, label check catches an injected leak
    pk = load_packet(sroot, 'learn', 'D01_S01')
    packet_label_check(pk, b)
    leaked = copy.deepcopy(pk)
    leaked['candidates']['C00']['e_late'] = b['pool']['C00']['e']['by_seed']
    try:
        packet_label_check(leaked, b)
        checks['label_check_catches_e_leak'] = False
    except PermissionError:
        checks['label_check_catches_e_leak'] = True
    leaked2 = copy.deepcopy(pk)
    leaked2['task'] += ' D01_S01'
    try:
        packet_label_check(leaked2, b)
        checks['label_check_catches_case_id'] = False
    except PermissionError:
        checks['label_check_catches_case_id'] = True
    # 3. Learn whitelist: Slow tables only from S cases; the Slow payload never names a Q case or a card
    tb = learn_evidence(sroot, 'D01')
    s = json.dumps(slow_payload(tb, 'D01', 1), ensure_ascii=False)
    checks['slow_payload_only_learn_cases'] = ('Q0' not in s) and all('S%02d' % i in s for i in range(1, 9)) and ('W_old' not in s) and ('_scratch' not in s)
    checks['slow_payload_has_late_supervision'] = '"late"' in s and 'original_f0_submission' in s and 'e_by_seed' in s
    # 4. Slow (fake) -> cards -> Select: same input / different card; receipts; resume; duplicate start
    fake = lambda led, out, http_cap, stage: FakeSubmitClient(led, out, http_cap=http_cap, stage=stage)
    sl = slow_stage(sroot, client_factory=fake)
    checks['slow_alias_no_supported_rule'] = any(r['alias_of'] == 's0' for r in sl['domains']['D02']['proposals'])
    checks['slow_distinct_cards_d01'] = sl['domains']['D01']['n_distinct_new_cards'] == 3
    jobs = select_jobs(sroot)
    st = decision_stage(sroot, 'select', jobs, client_factory=fake)
    checks['select_finished'] = st['status'] == 'FINISHED'
    reqs = sorted((paths(sroot)['select'] / 'raw_responses').glob('*_request.json'))
    by_case = {}
    for q in reqs:
        d = context.read_json(q)
        if d['role'] != 'fast_submit':
            continue
        body = json.loads(d['messages'][1]['content'])
        common = json.dumps({k: body[k] for k in PACKET_TOP_KEYS}, ensure_ascii=False)
        by_case.setdefault(d['case'], []).append((d['arm'], common, json.dumps(body['guidance'])))
    checks['same_common_input_across_arms'] = all(len({c for _, c, _ in v}) == 1 for v in by_case.values())
    checks['guidance_differs_by_arm'] = all(len({g for _, _, g in v}) == len(v) for v in by_case.values() if len(v) > 1)
    checks['request_receipts_written'] = len(reqs) == len([q for q in reqs if q.with_name(q.name.replace('_request.json', '_response.json')).exists()]) and len(reqs) == len(jobs) and len(list((paths(sroot)['formation'] / 'raw_responses').glob('*_request.json'))) == 6
    for d in DOMAINS:
        freeze_domain(sroot, d)
    fr = context.read_json(paths(sroot)['freeze'] / 'D01.json')
    checks['freeze_has_w_star_fixed_dev_h'] = fr['w_star']['arm'] in CARD_ARMS and fr['fixed_dev']['public_id'] in PUBLIC and fr['h_select']['choice'] in H_ORDER
    # cached scoring: a decision's E equals the cache of the selected physical material
    dec = context.read_json(paths(sroot)['select'] / 'D01_Q01' / 's0.json')
    bb = load_binding(sroot, 'select', 'D01_Q01')
    checks['legal_id_and_cached_e'] = dec['decision']['selected_candidate_id'] in bb['pool'] and fr['arms']['s0']['picks']['D01_Q01']['e_mean'] == bb['pool'][dec['decision']['selected_candidate_id']]['e']['mean']
    # 5. replay with an unfinished decision: fail one request, verify the stage is INCOMPLETE and the readout does not connect the missing arm; resume finishes without re-sending done ones
    rj = replay_jobs(sroot)
    fail_fake = lambda led, out, http_cap, stage: FakeSubmitClient(led, out, http_cap=http_cap, stage=stage, fail_requests=(3, 4))     # attempt 0 transient on 2 logical requests -> unknown usage blocks the rest
    st2 = decision_stage(sroot, 'replay', rj, client_factory=fail_fake)
    checks['replay_incomplete_after_unknown_usage'] = st2['status'] == 'INCOMPLETE' and bool(st2['unfinished'])
    led = context.read_json(paths(sroot)['ledger'])
    checks['unknown_usage_recorded'] = led['llm_tokens_unknown'] >= 1 and led['llm_tokens_unknown'] > led.get('unknown_usage_accepted', 0)
    try:
        decision_stage(sroot, 'replay', rj, client_factory=fake)
        checks['resume_refused_without_acceptance'] = False
    except RuntimeError:
        checks['resume_refused_without_acceptance'] = True
    r0 = readout(sroot)
    checks['readout_not_connected_when_incomplete'] = not r0['replay']['D01'].get('complete', True) and not r0.get('summary', {}).get('complete', False)
    n_before = context.read_json(paths(sroot)['ledger'])['llm_requests']
    done_before = sum(1 for c, a, _ in rj if (paths(sroot)['replay'] / c / ('%s.json' % a)).exists() and context.read_json(paths(sroot)['replay'] / c / ('%s.json' % a))['status'] == 'DONE')
    accept_unknown_usage(sroot, 'smoke_operator')
    st3 = decision_stage(sroot, 'replay', rj, client_factory=fake)
    n_after = context.read_json(paths(sroot)['ledger'])['llm_requests']
    checks['resume_sends_only_unfinished'] = st3['status'] == 'FINISHED' and (n_after - n_before) == len(rj) - done_before
    r1 = readout(sroot)
    checks['readout_complete_after_resume'] = r1['summary']['complete']
    checks['same_material_zero_difference'] = all(abs(r['g']['s_w_over_s0']['mean']) < 1e-12 for d in DOMAINS for r in r1['replay'][d]['cases'].values() if r['same_delivery']['s_w_vs_s0'])
    # 6. one format correction
    croot = paths(root)['smoke'] / 'corr'
    if croot.exists():
        shutil.rmtree(croot)
    croot.mkdir(parents=True)
    build_clean_inputs(croot)
    bad = lambda led, out, http_cap, stage: FakeBadFormatClient(led, out, http_cap=http_cap, stage=stage)
    stc = decision_stage(croot, 'select', [('D01_Q01', 's0', None)], client_factory=bad)
    dc = context.read_json(paths(croot)['select'] / 'D01_Q01' / 's0.json')
    checks['single_format_correction'] = stc['status'] == 'FINISHED' and dc['status'] == 'DONE' and len(dc['attempts']) == 2 and 'error' in dc['attempts'][0]
    # 7. duplicate controller refused (OS lock) via the parent's lock-probe subprocess
    lk = DP.PackageLock(sroot, 'driver').acquire()
    try:
        rc = subprocess.run([sys.executable, '-B', '-m', DP.MODULE, '--lock-probe', str(sroot)], cwd=str(REPO), capture_output=True, timeout=120).returncode
    finally:
        lk.release()
    checks['duplicate_controller_refused'] = rc == 3
    rc2 = subprocess.run([sys.executable, '-B', '-m', DP.MODULE, '--lock-probe', str(sroot)], cwd=str(REPO), capture_output=True, timeout=120).returncode
    checks['lock_free_after_release'] = rc2 == 0
    checks['no_fit_no_production_client'] = context.read_json(paths(sroot)['ledger'])['fit_attempts'] == 0
    out = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks, 'local': now(), 'root': str(sroot)}
    context.write_json(paths(root)['smoke'] / 'smoke.json', out)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


# ============================================================================= monitor (read-only)
def monitor(root: Path = ROOT, interval: float = 60.0, once: bool = False) -> None:
    import psutil
    P_ = paths(root)
    while True:
        line = ['MONITOR', now()]
        q = root / 'driver_launch.json'
        if q.exists():
            try:
                pid = int(context.read_json(q)['pid'])
                line.append('driver=%d:%s' % (pid, 'alive' if psutil.pid_exists(pid) else 'exited'))
            except Exception:  # noqa: BLE001
                pass
        held = None
        if P_['lock'].exists():
            try:
                with open(P_['lock'], 'rb') as fh:
                    fh.read(1)
                held = False
            except OSError:
                held = True
        line.append('lock=%s' % ({None: 'absent', True: 'held', False: 'free'}[held]))
        if P_['ledger'].exists():
            led = context.read_json(P_['ledger'])
            ev = led.get('events', [])
            last_llm = max((e['epoch'] for e in ev if e.get('kind') == 'llm_finished'), default=None)
            line.append('req=%d http=%d tok=%.2fM unknown=%d accepted=%d failed=%d' % (led['llm_requests'], led['llm_http_attempts'], (led['llm_tokens_in'] + led['llm_tokens_out']) / 1e6,
                                                                                 led['llm_tokens_unknown'], led.get('unknown_usage_accepted', 0), led.get('llm_failed_attempts', 0)))
            line.append('last_llm=%s' % ('%.0fs ago' % (time.time() - last_llm) if last_llm else '-'))
            if led.get('paid_clock_started_epoch'):
                line.append('paid=%.1fmin' % ((time.time() - led['paid_clock_started_epoch']) / 60))
        for stage in ('select', 'replay'):
            pp = P_[stage] / 'progress.json'
            if pp.exists():
                pr = context.read_json(pp)
                line.append('%s: pending %s done %s failed %s' % (stage, pr.get('pending'), pr.get('done'), pr.get('failed')))
        for name in ('driver.log', 'driver.err.log'):
            lg = P_['logs'] / name
            if lg.exists():
                tail = [t for t in lg.read_text(encoding='utf-8', errors='replace').splitlines() if t.strip()][-2:]
                if tail:
                    line.append('%s: %s' % (name, ' | '.join(t[:120] for t in tail)))
        print(' '.join(line), flush=True)
        if once:
            return
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--wiring', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--accept-unknown-usage', action='store_true')
    ap.add_argument('--result', action='store_true')
    ap.add_argument('--monitor', action='store_true')
    ap.add_argument('--interval', type=float, default=60.0)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--lock-probe')
    a = ap.parse_args()
    root = Path(a.root)
    if a.lock_probe:
        sys.exit(DP.lock_probe(Path(a.lock_probe)))
    elif a.preflight:
        cfg = preflight(root)
        print(json.dumps({k: cfg[k] for k in ('package', 'slow_payloads', 'decision_payloads', 'planned_main_requests', 'total_caps', 'concurrency')}, indent=1, ensure_ascii=False))
    elif a.smoke:
        smoke(root)
    elif a.wiring:
        print(json.dumps(wiring(root), indent=1, ensure_ascii=False, default=str))
    elif a.run:
        package_run(root, accept_unknown=a.accept_unknown_usage)
    elif a.result:
        readout(root)
        print((root / 'tables.md').read_text(encoding='utf-8'))
    elif a.monitor:
        monitor(root, a.interval, a.once)


if __name__ == '__main__':
    main()
