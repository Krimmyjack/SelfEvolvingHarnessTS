"""Bounded M2 experiment: same numerical base, old/roundtrip/Random on a75/a85.
Also hosts the bounded fast_baselines follow-up via --study fast_baselines, the
Source-process study (--study source_process: one Slow formation from whitelisted history,
F0/F_generic/F_source/F_program/Random on a89/a93) via batch_research_source_process,
the skill_revision study (batch_research_skill_revision) and the draft_construction study
(--study draft_construction: F_free/F_select/F_edit/Random/Menu with frozen unscored drafts on a65_g3/a85_g3)
via batch_research_draft_construction, and the exploration_quota study (--study exploration_quota:
F_auto/F_text/F_quota/Random/Menu with a frozen random probe Q on a65_g4/a85_g4) via batch_research_exploration_quota,
and the paired_refinement study (--study paired_refinement: F_refine/F_free with the derived single-leaf build form, three slots and
evidence round trip, Random-global/Random-local/Menu on a65_g5/a85_g5) via batch_research_paired_refinement, and the measured_start
study (--study measured_start: trained random initialization R1/R2 given to F_start/Random-global/Random-local, F_free with four free
slots, Menu, Start-only and R0/RB shadow selections on a65_g7/a85_g7) via batch_research_measured_start.
No outcome-dependent expansion, no credential persistence.
"""
from __future__ import annotations
import argparse
import re
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from methods.ttha import batch_research as br
from methods.ttha.batch_base import context, commit, policy, spec
from evaluation.main_protocol_p4 import batch_research_runtime as rt
from evaluation.main_protocol_p4 import run_batch_research_v1 as base
from evaluation.main_protocol_p4 import batch_research_source_process as sp
from evaluation.main_protocol_p4 import batch_research_skill_revision as sr
from evaluation.main_protocol_p4 import batch_research_draft_construction as dc
from evaluation.main_protocol_p4 import batch_research_exploration_quota as eq
from evaluation.main_protocol_p4 import batch_research_paired_refinement as pr
from evaluation.main_protocol_p4 import batch_research_measured_start as ms
from evaluation.main_protocol_p4 import batch_research_domain_skill as dsk

# Studies that share the draft-pool worker path (prepare -> common baselines -> arms via the module's run_arm/resume_arm).
STUDY_MOD={'draft_construction':dc,'exploration_quota':eq,'paired_refinement':pr,'measured_start':ms,'domain_skill':dsk}
RESUMABLE=('source_process','skill_revision','draft_construction','exploration_quota','paired_refinement','measured_start','domain_skill')

REPO=Path(__file__).resolve().parents[2]
DEFAULT=REPO/'_scratch/dev_batch_research_roundtrip'
JOBS=('a75','a85')
RANDOM_SEEDS={'a75':(910751,910752),'a85':(910851,910852)}
ORDER={'a75':('old','roundtrip','random'),'a85':('roundtrip','old','random')}
CAPS=dict(max_fit_attempts=50,max_llm_requests=64,max_llm_tokens=2000000,max_wall_s=7200,max_retries=2)

STUDY='roundtrip'
JOB_DATASETS={}

def configure_study(study, study_config=None):
    """Two explicit frozen studies, selected once per process; no outcome-driven grid.
    domain_skill takes its jobs/domains/arms/budgets from an explicit JSON (--study-config); there is no built-in run."""
    global STUDY, DEFAULT, JOBS, RANDOM_SEEDS, ORDER, CAPS, ROSTERS, STUDY_SEEDS, JOB_DATASETS
    STUDY=study; ROSTERS={}; STUDY_SEEDS=rt.SEEDS; JOB_DATASETS={}; rt.FIT_RETRY=False   # only a study config that authorizes retries turns it on
    if study=='roundtrip':
        DEFAULT=REPO/'_scratch/dev_batch_research_roundtrip'
        JOBS=('a75','a85')
        RANDOM_SEEDS={'a75':(910751,910752),'a85':(910851,910852)}
        ORDER={'a75':('old','roundtrip','random'),'a85':('roundtrip','old','random')}
        CAPS=dict(max_fit_attempts=50,max_llm_requests=64,max_llm_tokens=2000000,max_wall_s=7200,max_retries=2)
    elif study=='fast_baselines':
        DEFAULT=REPO/'_scratch/dev_batch_research_fast_baselines'
        JOBS=('a90','a95')
        RANDOM_SEEDS={'a90':(910901,910902),'a95':(910951,910952)}
        ORDER={'a90':('fast','random'),'a95':('random','fast')}
        CAPS=dict(max_fit_attempts=38,max_llm_requests=32,max_llm_tokens=1000000,max_wall_s=5400,max_retries=2)
    elif study=='source_process':
        DEFAULT=REPO/'_scratch/dev_batch_research_source_process'
        JOBS=('a89','a93')
        RANDOM_SEEDS=dict(sp.RANDOM_SEEDS)
        ORDER=dict(sp.ORDER)
        CAPS=dict(max_fit_attempts=74,max_llm_requests=130,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2)
    elif study=='skill_revision':
        DEFAULT=REPO/'_scratch/dev_batch_research_skill_revision'
        JOBS=tuple(sr.JOBS)
        RANDOM_SEEDS={}
        ORDER=dict(sr.ORDER)
        ROSTERS=dict(sr.ROSTERS)
        STUDY_SEEDS=tuple(sr.SEEDS)
        CAPS=dict(max_fit_attempts=74,max_llm_requests=130,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2)
    elif study=='draft_construction':
        DEFAULT=REPO/'_scratch/dev_batch_research_draft_construction_g3'   # task §13: G3 run; the G2 eligibility-failed record stays read-only in the un-suffixed directory
        JOBS=tuple(dc.JOBS)
        RANDOM_SEEDS={}
        ORDER=dict(dc.ORDER)
        ROSTERS=dict(dc.ROSTERS)
        STUDY_SEEDS=tuple(dc.SEEDS)
        CAPS=dict(max_fit_attempts=74,max_llm_requests=96,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2)
    elif study=='exploration_quota':
        DEFAULT=REPO/'_scratch/dev_batch_research_exploration_quota'
        JOBS=tuple(eq.JOBS)
        RANDOM_SEEDS={}
        ORDER=dict(eq.ORDER)
        ROSTERS=dict(eq.ROSTERS)
        STUDY_SEEDS=tuple(eq.SEEDS)
        CAPS=dict(max_fit_attempts=74,max_llm_requests=96,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2)
    elif study=='paired_refinement':
        DEFAULT=REPO/'_scratch/dev_batch_research_paired_refinement'
        JOBS=tuple(pr.JOBS)
        RANDOM_SEEDS={}
        ORDER=dict(pr.ORDER)
        ROSTERS=dict(pr.ROSTERS)
        STUDY_SEEDS=tuple(pr.SEEDS)
        CAPS=dict(max_fit_attempts=98,max_llm_requests=64,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2)   # task §8: 96 planned + 2 retries; 64 logical / 128 HTTP
    elif study=='measured_start':
        DEFAULT=REPO/'_scratch/dev_batch_research_measured_start'
        JOBS=tuple(ms.JOBS)
        RANDOM_SEEDS={}
        ORDER=dict(ms.ORDER)
        ROSTERS=dict(ms.ROSTERS)
        STUDY_SEEDS=tuple(ms.SEEDS)
        CAPS=dict(max_fit_attempts=98,max_llm_requests=64,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2)   # task §8: 96 planned (6+6+6+12+6+6+6 per job) + 2 retries
    elif study=='domain_skill':
        if study_config is None: raise ValueError('domain_skill needs --study-config')
        dsk.configure(dsk.load_study_config(study_config))
        DEFAULT=Path(dsk.CFG['output'])
        JOBS=tuple(dsk.JOBS)
        RANDOM_SEEDS={}
        ORDER=dict(dsk.ORDER)
        ROSTERS=dict(dsk.ROSTERS)
        STUDY_SEEDS=tuple(dsk.SEEDS)
        JOB_DATASETS=dict(dsk.JOB_DATASETS)
        CAPS=dict(dsk.CFG['caps'])
    else:
        raise ValueError('unknown study')

def study_tool_contracts():
    return sp.CONTRACTS if STUDY in ('source_process','skill_revision','draft_construction','exploration_quota','paired_refinement','measured_start') else rt.CONTRACTS

def study_seeds():
    return tuple(STUDY_SEEDS)

def study_roster(job):
    return ROSTERS.get(job)

def study_dataset(job):
    return JOB_DATASETS.get(job,'electricity')

def study_http_cap():
    """HTTP attempt cap handed to the metered client; domain_skill stages carry a cumulative package cap in their config."""
    if STUDY=='domain_skill' and dsk.CFG.get('http_cap'): return int(dsk.CFG['http_cap'])
    return 2*CAPS['max_llm_requests']

def study_ledger_path(root):
    """domain_skill may bind every stage of one package to a single package ledger; other studies keep root/budget.json."""
    if STUDY=='domain_skill' and dsk.CFG.get('ledger_path'): return Path(dsk.CFG['ledger_path'])
    return root/'budget.json'

def study_tool_corrections():
    if STUDY=='domain_skill': return int(dsk.CFG['max_tool_corrections'])
    return 0 if STUDY=='roundtrip' else 2

def configuration():
    jobs={j:asdict(context.resolve_job(study_dataset(j),j,roster=study_roster(j))) for j in JOBS}
    expected_t={'roundtrip':[19728,22344],'fast_baselines':[23664,24984],'source_process':[23400,24456],'skill_revision':[25512,25512],'draft_construction':[17088,22344],'exploration_quota':[17088,22344],'paired_refinement':[17088,22344],'measured_start':[17088,22344]}.get(STUDY)
    assert expected_t is not None or STUDY=='domain_skill', 'unknown study'   # domain_skill: t comes from its explicit config and the shared job table
    assert expected_t is None or [jobs[j]['t'] for j in JOBS]==expected_t
    cfg={'study':STUDY,'jobs':jobs,'seeds':list(study_seeds()),'random_seeds':RANDOM_SEEDS,'order':ORDER,
            'caps':CAPS,'planned_fits':{'roundtrip':48,'fast_baselines':36,'source_process':72,'skill_revision':72,'draft_construction':72,'exploration_quota':72,'paired_refinement':96,'measured_start':96}.get(STUDY,getattr(STUDY_MOD.get(STUDY),'PLANNED_FITS',None)),'http_cap':2*CAPS['max_llm_requests'],'exposure':'EXPOSED_DEVELOPMENT',
            'knowledge':'H0 empty, no Slow','treatment':'evidence_roundtrip only' if STUDY=='roundtrip' else 'batch Fast vs Fixed/Random; common bounded input-error feedback',
            'max_tool_corrections':study_tool_corrections(),
            'evidence_roundtrip':STUDY in ('roundtrip','paired_refinement','measured_start'),
            'consumer':spec.CONSUMER,'tool_contracts':study_tool_contracts(),
            'utility_rule':'Report paired effects without a new post-hoc significance threshold; single method trajectory per arm/job cannot establish reliable net utility.'}
    if STUDY=='source_process':
        cfg.update({'knowledge':'per-arm frozen: f0 empty H0; f_generic fixed generic experiment_guidance; f_source one Slow-formed experiment_guidance from whitelisted Source; f_program construction_guidance carrying only the argmin-C_A policy of the last Source job',
                    'treatment':'Source-derived experiment_guidance (F_source) against F0, F_generic, F_program, Random, Fixed and None; public tool semantics shared by every arm',
                    'arms':list(sp.ARMS),'source_whitelist':[list(x) for x in sp.SOURCE_BRANCHES],
                    'source_boundary':{'last_source_job':sp.LAST_SOURCE_JOB,'workload_end_row_exclusive':sp.SOURCE_WORKLOAD_END_ROW},
                    'slow':{'formations':1,'contract_corrections':1,'editable_hooks':['experiment_guidance'],'reads':'Source T-only episodes, C_A, post-commit C_B; no E, no Target'},
                    'per_fast_job_limits':{**asdict(br.Limits()),'wall_seconds':'remaining package wall'},
                    'generic_body':sp.GENERIC_TEXT,'tool_semantics':rt.TOOL_SEMANTICS,
                    'fit_plan':'2 jobs x (6 common + 4 Fast arms x 6 + Random 6) = 72; cap 74 reserves 2 same-configuration non-scientific retries'})
    if STUDY=='skill_revision':
        cfg.update({k:v for k,v in sr.frozen_config(CAPS).items() if k not in ('caps','jobs')})
        cfg['rosters']={j:list(r) for j,r in ROSTERS.items()}
        cfg['fit_plan']='2 jobs x (6 common + 4 Fast arms x 6 + Menu 6) = 72; cap 74 reserves 2 same-configuration non-scientific retries'
    if STUDY in STUDY_MOD:
        cfg.update({k:v for k,v in STUDY_MOD[STUDY].frozen_config(CAPS).items() if k not in ('caps','jobs')})
        cfg['rosters']={j:list(r) for j,r in ROSTERS.items()}
        cfg['fit_plan']=getattr(STUDY_MOD[STUDY],'FIT_PLAN','2 jobs x (6 common + 3 Fast arms x at most 6 + Random 6 + Menu 6) = at most 72; cap 74 reserves 2 same-configuration non-scientific retries')
    return cfg

def random_branch(root,job,led,shared,*,seeds=None,roster=None,dataset='electricity',policy_seeds=None):
    # dataset / policy_seeds: explicit for studies outside this runner's frozen tables; defaults keep the historical path.
    seeds=tuple(seeds or rt.SEEDS)
    root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(shared/job,root/job)
    adapter=rt.Adapter(root,job,led,REPO,roster=roster,seeds=seeds,dataset=dataset)
    candidates=list(adapter.baselines()); trace=[]
    def emit(event,**kw):
        row={'event_id':f'{job}:{len(trace)}','event':event,**kw};trace.append(row)
        base.event_sink(root/'trace.jsonl')(row)
    for i,seed in enumerate(RANDOM_SEEDS[job] if policy_seeds is None else policy_seeds,1):
        c=adapter.build_material({'plan_id':f'p{i}','policy':policy.random_policy(seed)},remaining_seconds=led.remaining())
        emit('random_material',sampling_seed=seed,material_spec=c.material_spec)
        c=adapter.evaluate(c,seeds,feedback=True,remaining_seconds=led.remaining())
        candidates.append(c);emit('random_evaluated',plan_id=c.plan_id,feedback=c.feedback(seeds,2,32))
    chosen=min(candidates,key=lambda c:c.feedback(seeds,2,32)['mean_loss'])
    reason='Random baseline: minimum C_A mean among None, FixedMixup, p1, p2; frozen tie order.'
    commit.commit(root,dataset,job,chosen.plan_id,reason,seeds[0])
    emit('committed',plan_id=chosen.plan_id,delivery_model_ref=chosen.model_refs[0])
    result=br.RunResult('COMPLETE',job,0,chosen.plan_id,chosen.model_refs[0],0,0,2,trace)
    context.write_json(root/'branch_result.json',asdict(result))
    print('RANDOM',job,'commit',chosen.plan_id,flush=True)
    return result

def menu_branch(root,job,led,shared,*,seeds=None,roster=None):
    """Fixed two-plan menu, 0 LLM (skill_revision §6): build and co-train MenuU and MenuFM, then commit the argmin
    three-seed C_A mean among FixedMixup, None, MenuU, MenuFM with that frozen tie order. No history, no guidance."""
    seeds=tuple(seeds or rt.SEEDS)
    root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(shared/job,root/job)
    adapter=rt.Adapter(root,job,led,REPO,roster=roster,seeds=seeds)
    by_id={c.plan_id:c for c in adapter.baselines()}; trace=[]
    def emit(event,**kw):
        row={'event_id':f'{job}:{len(trace)}','event':event,**kw};trace.append(row)
        base.event_sink(root/'trace.jsonl')(row)
    for mid,pol in sr.MENU:
        c=adapter.build_material({'plan_id':mid,'policy':pol},remaining_seconds=led.remaining())
        emit('menu_material',plan_id=mid,material_spec=c.material_spec)
        c=adapter.evaluate(c,seeds,feedback=True,remaining_seconds=led.remaining())
        by_id[mid]=c;emit('menu_evaluated',plan_id=mid,feedback=c.feedback(seeds,2,32))
    order=list(sr.MENU_TIE_ORDER)
    chosen=min((by_id[m] for m in order),key=lambda c:(c.feedback(seeds,2,32)['mean_loss'],order.index(c.plan_id)))
    reason='Fixed menu: minimum C_A three-seed mean among FixedMixup, None, MenuU, MenuFM; frozen tie order; no history read.'
    commit.commit(root,'electricity',job,chosen.plan_id,reason,seeds[0])
    emit('committed',plan_id=chosen.plan_id,delivery_model_ref=chosen.model_refs[0])
    result=br.RunResult('COMPLETE',job,0,chosen.plan_id,chosen.model_refs[0],0,0,2,trace)
    context.write_json(root/'branch_result.json',asdict(result))
    print('MENU',job,'commit',chosen.plan_id,flush=True)
    return result

def control_branch(arm,path,job,led,shared):
    return (menu_branch if arm=='menu' else random_branch)(path,job,led,shared,seeds=study_seeds(),roster=study_roster(job))

def worker(root):
    if STUDY=='domain_skill': dsk.preflight(dsk.CFG)   # refuse before any file, client, fit or label read
    if (root/'experiment_started.json').exists():raise RuntimeError('already started; no paid replay')
    root.mkdir(parents=True,exist_ok=True)
    context.write_json(root/'config.json',configuration())
    led=rt.RuntimeLedger(study_ledger_path(root),**CAPS)
    context.write_json(root/'experiment_started.json',{'epoch':time.time(),'pid':os.getpid(),'repo':str(REPO)})
    branches=[];failures=[];knowledge={};stopped_early=None;early_stop=None
    try:
        client=rt.MeteredClient(led,root/'raw_responses',http_cap=study_http_cap())
        if STUDY=='source_process':
            # Source census, P*, Generic and the single Slow output are frozen before any Target adapter opens.
            knowledge=sp.prepare(root,client)
            context.write_json(root/'source_frozen.json',{'epoch':time.time(),'f_source_available':knowledge.get('f_source') is not None})
        if STUDY=='skill_revision':
            # H_old, Generic, usage census and the one revision are frozen before any Target T is read; KEEP ends the run here.
            knowledge=sr.prepare(root,client,CAPS)
            context.write_json(root/'revision_frozen.json',{'epoch':time.time(),'f_new_available':knowledge is not None})
            if knowledge is None:
                raise EarlyStop('no revision treatment')
        commons={};pools={}
        if STUDY=='skill_revision':
            # Step 3 of the task: open both Targets' T first (population eligibility, roster binding, scope states), 0 fits.
            for job in JOBS:
                shared=root/(job+'_common');shared.mkdir()
                commons[job]=rt.Adapter(shared,job,led,REPO,roster=study_roster(job),seeds=study_seeds())
                sr.target_checks(root,job,commons[job],knowledge)
            context.write_json(root/'targets_opened.json',{'epoch':time.time(),'jobs':list(JOBS),'fits':0})
        if STUDY in STUDY_MOD:
            # Stage 2 of the task: both Targets' T only (eligibility, roster binding), both draft pools (and probes) generated and frozen; 0 fits.
            if getattr(STUDY_MOD[STUDY],'PREPARE_WITH_CLIENT',False): commons,pools=STUDY_MOD[STUDY].prepare(root,led,CAPS,client=client)   # e.g. domain_skill: routes frozen before any fit
            else: commons,pools=STUDY_MOD[STUDY].prepare(root,led,CAPS)
        for job in JOBS:
            shared=root/(job+'_common')
            if STUDY in STUDY_MOD and job not in commons: continue   # ELIGIBILITY_FAILED recorded by prepare; no substitute
            if job in commons: common=commons[job]
            else:
                shared.mkdir();common=rt.Adapter(shared,job,led,REPO,roster=study_roster(job),seeds=study_seeds())
            common.baselines()
            after=getattr(STUDY_MOD.get(STUDY),'after_baselines',None)   # e.g. measured_start: train the R1/R2 initialization once per job, before any arm
            if after is not None: after(root,job,led,shared,pools.get(job))
            for arm in ORDER[job]:
                path=root/(job+'_'+arm)
                if STUDY in STUDY_MOD and arm in STUDY_MOD[STUDY].POOL_ARMS and pools[job]['status']!='COMPLETE':
                    context.write_json(root/(job+'_'+arm+'_skipped.json'),{'status':pools[job]['status'],'note':'pool arm not run on an incomplete %s; no seed change, no substitute'%getattr(STUDY_MOD[STUDY],'POOL_NAME','draft pool')})
                    continue
                if STUDY=='source_process' and arm=='f_source' and knowledge.get('f_source') is None:
                    slow_status=context.read_json(root/'slow_result.json').get('status') if (root/'slow_result.json').exists() else 'NOT_RUN'
                    context.write_json(root/(job+'_f_source_treatment.json'),{'status':'NO_TREATMENT' if slow_status=='KEEP' else 'FORMATION_FAILED','slow_status':slow_status,
                                       'note':'F_source not run; no H0 substitute and no budget transfer'})
                    continue
                branches.append((path,job))
                try:
                    if STUDY in STUDY_MOD:result=STUDY_MOD[STUDY].run_arm(arm,path,job,led,client,shared,pools[job],seeds=study_seeds(),roster=study_roster(job),max_tool_corrections=study_tool_corrections())
                    elif arm in ('random','menu'):result=control_branch(arm,path,job,led,shared)
                    else:result=base.branch(path,job,knowledge.get(arm,br.Knowledge()) if STUDY in ('source_process','skill_revision') else br.Knowledge(),led,client,shared,
                                            evidence_roundtrip=arm=='roundtrip',max_tool_corrections=study_tool_corrections(),tool_contracts=study_tool_contracts(),
                                            seeds=study_seeds(),roster=study_roster(job))
                    if result.status!='COMPLETE':failures.append({'branch':path.name,'kind':result.failure_kind})
                except Exception as exc:
                    failures.append({'branch':path.name,'exception_type':type(exc).__name__})
                    print('BRANCH_FAILED',path.name,type(exc).__name__,flush=True)
                led.check_wall();led.check_llm()
                if client.fatal or rt.unknown_usage_blocks(led):raise RuntimeError('backend fatal or unknown usage')
        context.write_json(root/'decisions_frozen.json',{'epoch':time.time(),'branches':[str(p) for p,j in branches]})
    except EarlyStop as exc:
        early_stop=str(exc)
        print('EARLY_STOP',early_stop,flush=True)
    except Exception as exc:
        failures.append({'stage':'execution','exception_type':type(exc).__name__})
        stopped_early=type(exc).__name__
        print('EXECUTION_STOP',type(exc).__name__,flush=True)
    finally:
        context.write_json(root/'execution_finished.json',{'epoch':time.time(),'branches':[(str(p),j) for p,j in branches],'failures':failures})
    if early_stop is None:   # a legal early stop opened no Target and has no labels to open
        open_labels(root,branches,failures,led,withhold_reason=stopped_early and 'execution stopped before every planned arm was attempted (%s)'%stopped_early)
    report(root)

class EarlyStop(Exception):
    """Legal end of a run before any Target branch (e.g. Slow KEEP): not a failure, labels have nothing to open."""

def open_labels(root,branches,failures,led,*,withhold_reason=None,resumed=False):
    """Label stages run only when method work has ended for every planned arm. An early stop withholds
    them (no C_B/E opened) so the run can be resumed or explicitly finalized by an operator."""
    if withhold_reason:
        context.write_json(root/'labels_withheld.json',{'epoch':time.time(),'reason':withhold_reason,'branches':[(str(p),j) for p,j in branches],
            'unknown_usage':led.s['llm_tokens_unknown'],'next':'--resume (add --accept-unknown-usage if the ledger holds unknown usage) or --finalize to end the run without resuming'})
        print('LABELS_WITHHELD',withhold_reason,flush=True)
        return
    before=getattr(STUDY_MOD.get(STUDY),'before_labels',None)   # e.g. measured_start: freeze the R0/RB shadow selections (0 fits) before any label opens
    if before is not None: before(root,branches)
    try:
        eligible=[(p,j) for p,j in branches if (p/j/'commit.json').exists()]
        for p,j in eligible:
            if not (p/j/'c_b_scores.json').exists():base.stage(p,j,'c_b',led,roster=study_roster(j),dataset=study_dataset(j))
        if not getattr(STUDY_MOD.get(STUDY),'labels_e_allowed',lambda:True)():
            # a formation-stage run (e.g. domain_skill Source/Select) is authorized up to C_B only; no E prediction or score exists
            context.write_json(root/'labels_c_b_only.json',{'epoch':time.time(),'branches':[str(p) for p,j in eligible],'resumed':resumed})
            context.write_json(root/'execution_finished.json',{'epoch':time.time(),'branches':[(str(p),j) for p,j in branches],'failures':failures,'resumed':resumed})
            return
        for p,j in eligible:
            if not (p/j/'e_frozen.json').exists():base.stage(p,j,'freeze_e',led,roster=study_roster(j),dataset=study_dataset(j))
        context.write_json(root/'all_e_predictions_frozen.json',{'epoch':time.time(),'branches':[str(p) for p,j in eligible],'resumed':resumed})
        for p,j in eligible:
            if not (p/j/'e_scores.json').exists():base.stage(p,j,'score_e',led,roster=study_roster(j),dataset=study_dataset(j))
    except Exception as exc:
        failures.append({'stage':'external','exception_type':type(exc).__name__})
        print('EXTERNAL_STOP',type(exc).__name__,flush=True)
    context.write_json(root/'execution_finished.json',{'epoch':time.time(),'branches':[(str(p),j) for p,j in branches],'failures':failures,'resumed':resumed})

def labels_boundary_violations(root):
    """A run may be resumed only while its labels are still withheld: not finalized, no E barrier, and no
    C_B / E file opened in any branch. Read-only; returns the list of violations (empty = resumable)."""
    out=[]
    if not (root/'labels_withheld.json').exists(): out.append('labels_withheld.json missing')
    if (root/'finalized.json').exists() or (root/'labels_withheld_finalized.json').exists(): out.append('run was finalized')
    if (root/'all_e_predictions_frozen.json').exists(): out.append('E barrier already written')
    for branch in sorted(x for x in root.iterdir() if x.is_dir()):
        for jdir in sorted(x for x in branch.iterdir() if x.is_dir()):
            for name in ('c_b_scores.json','e_frozen.json','e_scores.json'):
                if (jdir/name).exists(): out.append('%s/%s/%s already opened'%(branch.name,jdir.name,name))
    return out

def finalize_worker(root):
    """Operator-declared termination of a withheld run: open labels for what was committed; nothing is resumed."""
    if not (root/'labels_withheld.json').exists(): raise RuntimeError('labels are not withheld; nothing to finalize')
    led=rt.RuntimeLedger(study_ledger_path(root),**CAPS)
    fin=context.read_json(root/'execution_finished.json')
    branches=[(Path(p),j) for p,j in fin['branches']]
    context.write_json(root/'finalized.json',{'epoch':time.time(),'reason':'operator terminated the run without resuming; unattempted arms stay unattempted','withheld':context.read_json(root/'labels_withheld.json')})
    shutil.move(root/'labels_withheld.json',root/'labels_withheld_finalized.json')
    open_labels(root,branches,list(fin['failures']),led)
    report(root)

def resume_worker(root,accept_unknown_usage=False):
    """One-time continuation after an early stop: same ledger (wall keeps counting), frozen knowledge
    reloaded from disk (Slow is never re-called), completed branches kept, a paused branch continued
    from its never-sent call, remaining arms run as planned, then the label stages for what is missing.
    Unknown usage keeps the frozen rule: paid calls stay refused unless the operator passes
    --accept-unknown-usage, which is recorded; nothing is reclassified and no cost is asserted as known."""
    if STUDY not in RESUMABLE: raise RuntimeError('resume is defined for the %s studies only'%', '.join(RESUMABLE))
    if STUDY=='domain_skill': dsk.preflight(dsk.CFG)
    if (root/'resume.json').exists(): raise RuntimeError('already resumed once; no second replay')
    fin=context.read_json(root/'execution_finished.json')
    if not any(f.get('stage')=='execution' for f in fin['failures']): raise RuntimeError('nothing to resume')
    # Label boundary first: before the ledger is touched, before any file moves, before any paid call.
    # --accept-unknown-usage only concerns the ledger and never lifts this boundary.
    blocked=labels_boundary_violations(root)
    if blocked: raise RuntimeError('resume refused; labels are no longer withheld: '+'; '.join(blocked))
    led=rt.RuntimeLedger(study_ledger_path(root),**CAPS)
    if rt.unknown_usage_blocks(led):
        if not accept_unknown_usage: raise RuntimeError('ledger holds unknown usage; the frozen rule stops paid calls. Pass --accept-unknown-usage to record an explicit operator decision, or --finalize')
        led.s['unknown_usage_accepted']=led.s['llm_tokens_unknown'];led._save()
    for name in ('execution_finished.json','all_e_predictions_frozen.json','labels_withheld.json'):
        if (root/name).exists(): shutil.move(root/name,root/name.replace('.json','_before_resume.json'))
    context.write_json(root/'resume.json',{'epoch':time.time(),'prior_failures':fin['failures'],'unknown_usage_at_resume':led.s['llm_tokens_unknown'],
        'operator_accepted_unknown_usage':bool(accept_unknown_usage and led.s.get('unknown_usage_accepted')),
        'failed_attempt_cost':'UNKNOWN; token_reserved_failed_upper=%d is an accounting estimate, not a guaranteed bound'%led.s.get('token_reserved_failed_upper',0),
        'slow_recalled':False,'e_opened_before_resume':[str(p) for p,j in fin['branches'] if (Path(p)/j/'e_scores.json').exists()]})
    frozen=context.read_json(root/'frozen_knowledge.json') if (root/'frozen_knowledge.json').exists() else {}
    knowledge={arm:(None if k is None else br.Knowledge(k['version'],tuple(br.Guidance(e['hook'],e['body'],e['applicability'],tuple(e['evidence_refs'])) for e in k['entries']))) for arm,k in frozen.items()}
    if STUDY=='skill_revision' and knowledge.get('f_new') is None: raise RuntimeError('no revision treatment was frozen; nothing to resume')
    pools=STUDY_MOD[STUDY].load_pools(root) if STUDY in STUDY_MOD else {}
    if STUDY in STUDY_MOD and not (root/getattr(STUDY_MOD[STUDY],'FROZEN_MARKER','drafts_frozen.json')).exists(): raise RuntimeError('draft pools / random supply were never frozen; nothing to resume')
    branches=[];failures=[];stopped_early=None
    try:
        client=rt.MeteredClient(led,root/'raw_responses',http_cap=study_http_cap())
        for job in JOBS:
            shared=root/(job+'_common')
            if STUDY in STUDY_MOD and (root/(job+'_eligibility.json')).exists(): continue   # ELIGIBILITY_FAILED before the stop
            if not shared.exists():
                if STUDY in STUDY_MOD: raise RuntimeError('common T of %s missing although pools were frozen'%job)
                shared.mkdir();common=rt.Adapter(shared,job,led,REPO,roster=study_roster(job),seeds=study_seeds())
                if STUDY=='skill_revision': sr.target_checks(root,job,common,knowledge)
                common.baselines()
            after=getattr(STUDY_MOD.get(STUDY),'after_baselines',None)
            if after is not None: after(root,job,led,shared,pools.get(job))
            for arm in ORDER[job]:
                path=root/(job+'_'+arm)
                if STUDY in STUDY_MOD and arm in STUDY_MOD[STUDY].POOL_ARMS and (pools[job] or {}).get('status')!='COMPLETE':
                    if not (root/(job+'_'+arm+'_skipped.json')).exists():
                        context.write_json(root/(job+'_'+arm+'_skipped.json'),{'status':(pools[job] or {}).get('status','MISSING'),'note':'pool arm not run on an incomplete %s; no seed change, no substitute'%getattr(STUDY_MOD[STUDY],'POOL_NAME','draft pool')})
                    continue
                if STUDY=='source_process' and arm=='f_source' and knowledge.get('f_source') is None:
                    if not (root/(job+'_f_source_treatment.json')).exists():
                        context.write_json(root/(job+'_f_source_treatment.json'),{'status':'NO_TREATMENT','note':'F_source not run; no H0 substitute and no budget transfer'})
                    continue
                branches.append((path,job))
                try:
                    if path.exists():
                        prior=context.read_json(path/'branch_result.json')
                        if prior['status']=='COMPLETE': result=br.RunResult(**{k:v for k,v in prior.items() if k!='trace'})
                        elif STUDY in STUDY_MOD: result=STUDY_MOD[STUDY].resume_arm(arm,path,job,led,client,pools[job],seeds=study_seeds(),roster=study_roster(job),max_tool_corrections=study_tool_corrections())
                        elif arm not in ('random','menu'): result=base.resume_branch(path,job,knowledge[arm],led,client,max_tool_corrections=study_tool_corrections(),tool_contracts=study_tool_contracts(),seeds=study_seeds(),roster=study_roster(job))
                        else: raise RuntimeError('control branch cannot be resumed')
                    elif STUDY in STUDY_MOD: result=STUDY_MOD[STUDY].run_arm(arm,path,job,led,client,shared,pools[job],seeds=study_seeds(),roster=study_roster(job),max_tool_corrections=study_tool_corrections())
                    elif arm in ('random','menu'): result=control_branch(arm,path,job,led,shared)
                    else: result=base.branch(path,job,knowledge[arm],led,client,shared,evidence_roundtrip=False,max_tool_corrections=study_tool_corrections(),tool_contracts=study_tool_contracts(),seeds=study_seeds(),roster=study_roster(job))
                    if result.status!='COMPLETE':failures.append({'branch':path.name,'kind':result.failure_kind})
                except Exception as exc:
                    failures.append({'branch':path.name,'exception_type':type(exc).__name__})
                    print('BRANCH_FAILED',path.name,type(exc).__name__,flush=True)
                led.check_wall();led.check_llm()
                if client.fatal or rt.unknown_usage_blocks(led):raise RuntimeError('backend fatal or unknown usage')
        context.write_json(root/'decisions_frozen.json',{'epoch':time.time(),'branches':[str(p) for p,j in branches],'resumed':True})
    except Exception as exc:
        failures.append({'stage':'execution','exception_type':type(exc).__name__})
        stopped_early=type(exc).__name__
        print('EXECUTION_STOP',type(exc).__name__,flush=True)
    finally:
        context.write_json(root/'execution_finished.json',{'epoch':time.time(),'branches':[(str(p),j) for p,j in branches],'failures':failures,'resumed':True})
    open_labels(root,branches,failures,led,withhold_reason=stopped_early and 'resumed execution stopped again before every planned arm was attempted (%s)'%stopped_early,resumed=True)
    report(root)

def compare_commits(a,b,block):
    if not a or not b:return None
    ca=a['plans'].get(a['controller']['committed_plan_id'],[])
    cb=b['plans'].get(b['controller']['committed_plan_id'],[])
    aa={c['seed']:c for c in ca}
    ds=[aa[c['seed']][block]['normalized_mse_macro']-c[block]['normalized_mse_macro'] for c in cb
        if c['seed'] in aa and c.get(block) and aa[c['seed']].get(block)]
    if len(ds)!=3:return None
    return {'delta_by_seed':ds,'mean_delta':statistics.mean(ds),'seed_se':statistics.stdev(ds)/3**.5,
            'positive_means':'second_arm_improves','scope':'fixed materials/workload; training seeds are not independent Agent trajectories'}

def report(root):
    keep=None  # a hand-written study REPORT.md must survive re-summarisation
    if STUDY in RESUMABLE and (root/'REPORT.md').exists():
        txt=(root/'REPORT.md').read_text(encoding='utf-8')
        if not txt.startswith('# W 批级研究 Workflow 首包'): keep=txt
    base.report(root,seeds=study_seeds())  # same plan/model/score collection; replace its V1-only status below
    r=context.read_json(root/'result.json');r.pop('slow',None)
    r['status']='COMPLETE' if len(r['branches'])==sum(len(ORDER[j]) for j in JOBS) and not r['failures'] and all(
        b['controller']['status']=='COMPLETE' and b['delivery_e'] is not None for b in r['branches'].values()) else 'PARTIAL'
    r['study']=STUDY
    r['comparisons']={};r['treatment_present']=False
    for name,b in r['branches'].items():
        trace=[json.loads(x) for x in (root/name/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
        b['deferred']=[x for x in trace if x['event']=='action_batch_deferred']
        b['tool_rejections']=[x for x in trace if x['event']=='tool_rejected']
        b['max_tool_corrections']=next((x.get('max_tool_corrections',0) for x in trace if x['event']=='job_started'),0)
        # Allowed correction and actual next request are different observations.
        b['error_feedback_deliveries']=sum(
            bool(x.get('correction_allowed')) and any(y['event']=='fast_request' for y in trace[i+1:])
            for i,x in enumerate(trace) if x['event']=='tool_rejected')
        b['mode_enabled']=any(x.get('event')=='job_started' and x.get('evidence_roundtrip') for x in trace)
        b['runtime_deferral_triggered']=bool(b['deferred'])
        b['treatment_present']=b['mode_enabled'] and b['controller']['calls']>0
        r['treatment_present']|=b['treatment_present']
        if STUDY=='fast_baselines':
            b['evidence_roundtrip_enabled']=b.pop('mode_enabled')
            b.pop('treatment_present')
            b['fast_executed']=name.endswith('_fast') and b['controller']['calls']>0
        available={p:statistics.mean(c['external']['normalized_mse_macro'] for c in cells)
                   for p,cells in b['plans'].items() if len(cells)==3 and all(c['external'] for c in cells)}
        b['observed_candidate_oracle_e']=min(available.values()) if available else None
        b['commit_regret_within_observed_candidates']=b['paired_mean_e']-min(available.values()) if available and b['paired_mean_e'] is not None else None
    if STUDY=='source_process':
        sp.finish_report(root,r,JOBS,ORDER)
        if keep is not None: (root/'REPORT.md').write_text(keep,encoding='utf-8')
        return
    if STUDY=='skill_revision':
        sr.finish_report(root,r,JOBS,ORDER,study_seeds())
        if keep is not None: (root/'REPORT.md').write_text(keep,encoding='utf-8')
        return
    if STUDY in STUDY_MOD:
        STUDY_MOD[STUDY].finish_report(root,r,JOBS,ORDER,study_seeds())
        if keep is not None: (root/'REPORT.md').write_text(keep,encoding='utf-8')
        return
    for job in JOBS:
        if STUDY=='roundtrip':
            old=r['branches'].get(job+'_old');new=r['branches'].get(job+'_roundtrip');rand=r['branches'].get(job+'_random')
            r['comparisons'][job]={'old_minus_roundtrip':{block:compare_commits(old,new,block) for block in ('c_a','c_b','external')},
                                   'random_minus_roundtrip':compare_commits(rand,new,'external')}
        else:
            fast=r['branches'].get(job+'_fast');rand=r['branches'].get(job+'_random')
            r['comparisons'][job]={'random_minus_fast':{block:compare_commits(rand,fast,block) for block in ('c_a','c_b','external')}}
    if STUDY=='fast_baselines':
        r.pop('treatment_present')  # This field described the old roundtrip intervention.
        r['fast_executed_jobs']=sum(b['fast_executed'] for b in r['branches'].values())
    r['utility_supported']=None
    r['utility_scope']='Single Agent trajectory per arm/job; report effects and uncertainty, not automatic Skill promotion.'
    context.write_json(root/'result.json',r)
    lines=['# W M2：'+('先读工具证据再决策' if STUDY=='roundtrip' else '批级Fast对固定与随机基线'),'', '状态：'+r['status'],'',
           '|分支|状态|commit|第一seed E|三seed E均值|deferred|候选集合内E遗憾|',
           '|---|---|---|---:|---:|---:|---:|']
    for name,b in r['branches'].items():
        lines.append(f"|{name}|{b['controller']['status']}|{b['controller']['committed_plan_id']}|{b['delivery_e']}|{b['paired_mean_e']}|{len(b['deferred'])}|{b['commit_regret_within_observed_candidates']}|")
    lines+=['','各作业配对比较；正值为第二个方法更好：']
    for j,x in r['comparisons'].items():lines.append(j+': '+json.dumps(x,ensure_ascii=False))
    lines+=['','预算：'+json.dumps(r.get('budget'),ensure_ascii=False),'',
            '完整候选及基线、逐seed/时间块读数见result.json。主指标不改，逐实体差不作因果归因；候选oracle只含实际共同拟合的完整方案。',
            'Source/Slow调用为0；没有知识积累结论。未执行的延期动作不能补成模型成绩。',
            'utility_supported=null：效果读数不自动等同可泛化的净效用证明。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('ROUNDTRIP_REPORT',r['status'],flush=True)

def smoke_source_process(cfg):
    """Zero-fit, zero-LLM checks of the study wiring (task §9): frozen inputs, information walls, dispatch."""
    from unittest.mock import patch
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    assert cfg['planned_fits']==72 and cfg['caps']==dict(max_fit_attempts=74,max_llm_requests=130,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2)
    assert cfg['http_cap']==260 and cfg['max_tool_corrections']==2 and cfg['evidence_roundtrip'] is False
    assert cfg['order']=={'a89':('f0','f_generic','f_source','f_program','random'),'a93':('random','f_program','f_source','f_generic','f0')}
    j89,j93=cfg['jobs']['a89'],cfg['jobs']['a93']
    assert (j89['t'],tuple(j89['c_a']),tuple(j89['c_b']),tuple(j89['e']))==(23400,(23400,23448),(23496,23544),(23592,23640,23688,23736))
    assert (j93['t'],tuple(j93['c_a']),tuple(j93['c_b']),tuple(j93['e']))==(24456,(24456,24504),(24552,24600),(24648,24696,24744,24792))
    assert j89['train_range'][0]==sp.SOURCE_WORKLOAD_END_ROW and max(j89['e'])+spec.H==j93['train_range'][0]
    assert cfg['random_seeds']=={'a89':(910891,910892),'a93':(910931,910932)}
    # Public semantics reach every arm through the same contracts; historical CONTRACTS stay bare.
    tc=cfg['tool_contracts']
    assert set(tc)==br.TOOLS and 'semantics' in tc['build_material'] and 'same entity' in tc['build_material']['semantics']['donor_rule_R']
    assert 'derived (augmented) pairs are never donors' in tc['build_material']['semantics']['donor_scope']
    assert 'semantics' not in rt.CONTRACTS['build_material']
    # Source census: deterministic, private-channel free, whitelisted branches only, boundary respected.
    ev=sp.build_source_evidence(); ev2=sp.build_source_evidence()
    assert json.dumps(ev,sort_keys=True)==json.dumps(ev2,sort_keys=True)
    assert [b['branch'] for b in ev['branches']]==[x[1] for x in sp.SOURCE_BRANCHES] and len(ev['branches'])==9
    assert all(j['overview']['workload_end_row_exclusive']<=sp.SOURCE_WORKLOAD_END_ROW for j in ev['jobs'].values())
    text=json.dumps(ev,ensure_ascii=False)
    for bad in ('"e":','"external"','e_scores','a90','a95','a94','a89','a93'):
        assert bad not in text,bad
    failed=[b for b in ev['branches'] if b['status']!='COMPLETE']
    assert [b['branch'] for b in failed]==['a85_old'] and failed[0]['failure']['mechanical_note']['predicate_features_outside_public_vocabulary']==['last168_std_over_full']
    assert all(b['delayed_c_b'] is not None for b in ev['branches'] if b['status']=='COMPLETE')
    assert ev['branches'][2]['initial_knowledge']['version']==1 and ev['branches'][0]['initial_knowledge']['version']==0
    # P*: argmin C_A of the last Source job, normalized policy only, text contract respected.
    seed=sp.select_program_seed()
    assert seed['selected']['material_id']=='p2' and seed['selected']['branch']=='a85_random' and len(seed['guidance_body'])<=1200
    assert set(seed['policy'])=={'default','rules'} and 'a85' not in seed['guidance_body'] and 'rationale' not in seed['guidance_body']
    policy.validate_text(seed['guidance_body'],allow_entity_ids=False); policy.validate_text(sp.GENERIC_TEXT,allow_entity_ids=False)
    assert cfg['generic_body']==sp.GENERIC_TEXT
    # Slow contract: only experiment_guidance may be proposed; KEEP and const:true are legal; refs must be legal.
    refs=frozenset(ev['legal_evidence_refs'])
    ok=br.apply_update(br.Knowledge(),{'decision':'PROPOSE','hook':'experiment_guidance','body':'compare one uniform and one conditional plan before commit','observable_applicability':{'const':True},'evidence_refs':[sorted(refs)[0]],'rationale':'x'},allowed_features=rt.BATCH_FIELDS,legal_evidence_refs=refs)
    assert ok.version==1 and ok.render({},rt.BATCH_FIELDS)['loaded'][0]['hook']=='experiment_guidance'
    assert br.apply_update(br.Knowledge(),{'decision':'KEEP','rationale':'nothing reusable'},allowed_features=rt.BATCH_FIELDS,legal_evidence_refs=refs).version==0
    for bad_prop in ({'decision':'PROPOSE','hook':'experiment_guidance','body':'x','observable_applicability':{'const':True},'evidence_refs':['not-a-ref'],'rationale':'x'},
                     {'decision':'PROPOSE','hook':'experiment_guidance','body':'x','observable_applicability':None,'evidence_refs':[sorted(refs)[0]],'rationale':'x'}):
        try: br.apply_update(br.Knowledge(),bad_prop,allowed_features=rt.BATCH_FIELDS,legal_evidence_refs=refs); raise AssertionError('accepted illegal proposal')
        except ValueError: pass
    class Scripted:
        last_text='{"decision":"PROPOSE","hook":"decision_guidance"}'
        def __init__(self,outs): self.outs=list(outs); self.calls=[]
        def call(self,role,unit,payload,system,max_tokens=4096,request_timeout=300.):
            self.calls.append((role,unit,sorted(payload.keys())))
            assert 'correction' in payload if role=='slow_contract_correction' else 'correction' not in payload
            return self.outs.pop(0)
    with TemporaryDirectory() as td:
        c=Scripted([{'decision':'PROPOSE','hook':'decision_guidance','body':'x','observable_applicability':{'const':True},'evidence_refs':[sorted(refs)[0]],'rationale':'x'},{'decision':'KEEP','rationale':'fine'}])
        h=sp.form_source_guidance(Path(td),c,ev)
        assert h is not None and h.version==0 and [x[0] for x in c.calls]==['slow','slow_contract_correction']
        assert context.read_json(Path(td)/'slow_result.json')['status']=='KEEP' and (Path(td)/'slow_contract_error_0.json').exists()
        sent=context.read_json(Path(td)/'slow_evidence.json'); assert sent['editable_hooks']==['experiment_guidance'] and '"external"' not in json.dumps(sent)
        c=Scripted([{'decision':'PROPOSE','hook':'decision_guidance','body':'x','observable_applicability':{'const':True},'evidence_refs':[sorted(refs)[0]],'rationale':'x'}]*2)
        (Path(td)/'again').mkdir(); assert sp.form_source_guidance(Path(td)/'again',c,ev) is None and context.read_json(Path(td)/'again'/'slow_result.json')['status']=='PARSE_OR_VALIDATION_FAILED'
    # Worker dispatch: per-arm frozen knowledge, semantic contracts, corrections=2, F_source skipped only when absent.
    for available in (False,True):
        with TemporaryDirectory() as td:
            root=Path(td)/'run'
            k={'f0':br.Knowledge(),'f_generic':br.Knowledge(1,(br.Guidance('experiment_guidance',sp.GENERIC_TEXT,{'const':True},('g',)),)),
               'f_program':br.Knowledge(1,(br.Guidance('construction_guidance',seed['guidance_body'],{'const':True},('p',)),)),
               'f_source':br.Knowledge(1,(br.Guidance('experiment_guidance','scripted source card',{'const':True},('s',)),)) if available else None}
            def fake_prepare(r,client):
                context.write_json(r/'slow_result.json',{'status':'CANDIDATE_TEST_ONLY' if available else 'KEEP'}); return k
            done=br.RunResult('COMPLETE','a89',0,'None','ref',1,1,0)
            with patch.object(rt,'MeteredClient') as mc, patch.object(sp,'prepare',side_effect=fake_prepare), patch.object(rt,'Adapter') as ad, \
                 patch.object(base,'branch',return_value=done) as brn, patch.object(sys.modules[__name__],'random_branch',return_value=done) as rb, \
                 patch.object(base,'stage'), patch.object(sys.modules[__name__],'report'):
                mc.return_value.fatal=False; ad.return_value.baselines.return_value=()
                worker(root)
            arms=[c.args[0].name for c in brn.call_args_list]
            expect=[j+'_'+a for j in JOBS for a in ORDER[j] if a!='random' and (a!='f_source' or available)]
            assert arms==expect,arms
            assert rb.call_count==2
            for c in brn.call_args_list:
                arm=c.args[0].name.split('_',1)[1]
                assert c.args[2] is k[arm] and c.kwargs['tool_contracts'] is sp.CONTRACTS and c.kwargs['max_tool_corrections']==2 and c.kwargs['evidence_roundtrip'] is False
            assert (root/'a89_f_source_treatment.json').exists()==(not available) and (root/'source_frozen.json').exists()
            fin=context.read_json(root/'execution_finished.json'); assert len(fin['branches'])==(10 if available else 8) and not fin['failures']
    print('SOURCE_PROCESS_SMOKE_OK: config, semantics, census, P*, generic, slow contract, dispatch')

def smoke_fault_paths():
    """Simulated faults only (0 fits, 0 LLM): frozen unknown-usage rule, withheld labels on early stop,
    explicit resume/finalize. Exercises the real worker/resume/finalize entries with patched branches."""
    from unittest.mock import patch
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    # (i) transport: one 5xx then success -> usage unknown for the failed attempt -> later paid calls refused until accepted.
    class Boom(Exception): pass
    Boom.__name__='InternalServerError'
    class FakeAPI:
        def __init__(self): self.n=0; self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self,**kw):
            self.n+=1
            if self.n==1: raise Boom('simulated 5xx')
            return SimpleNamespace(model='grok-4.6-build',usage=SimpleNamespace(prompt_tokens=10,completion_tokens=5),
                                   choices=[SimpleNamespace(message=SimpleNamespace(content='{"actions":[]}'))],model_dump=lambda mode='json':{'model':'grok-4.6-build','usage':{'prompt_tokens':10,'completion_tokens':5}})
    with TemporaryDirectory() as td:
        led=rt.RuntimeLedger(Path(td)/'budget.json',**CAPS)
        c=rt.MeteredClient.__new__(rt.MeteredClient); c.api=FakeAPI(); c.ledger=led; c.out=Path(td)/'raw'; c.out.mkdir(); c.fatal=False; c.http_cap=10; c.last_text=None
        assert c.call('fast','u',{'x':1},'sys')=={'actions':[]}
        assert led.s['llm_tokens_unknown']==1 and led.s['llm_failed_attempts']==1 and led.s['llm_http_attempts']==2 and led.s['llm_tokens_in']==10
        assert rt.unknown_usage_blocks(led)
        try: c.call('fast','u',{'x':2},'sys'); raise AssertionError('frozen rule not applied')
        except rt.budget.BudgetExhausted: pass
        led.s['unknown_usage_accepted']=1; led._save()
        assert not rt.unknown_usage_blocks(led) and c.call('fast','u',{'x':3},'sys')=={'actions':[]}
    # (ii)-(iv) worker early stop -> labels withheld; resume needs explicit acceptance; finalize opens labels without resuming.
    first=[JOBS[0]+'_'+a for a in ORDER[JOBS[0]][:2]]
    def run_worker_with_stop(root):
        k={a:br.Knowledge() for a in ('f0','f_generic','f_program','f_source','f_old','f_new')}
        def fake_prepare(r,client,*rest):
            context.write_json(r/'slow_result.json',{'status':'CANDIDATE_TEST_ONLY'}); context.write_json(r/'frozen_knowledge.json',{a:asdict(h) for a,h in k.items()}); return k
        def fake_dc_prepare(r,led,caps):
            pool={'job':None,'status':'COMPLETE','drafts':[{'draft_id':'D%d'%i,'policy':{'default':{'steps':[]},'rules':[]},'assignment':[[]]*32} for i in (1,2,3,4)],'excluded_keys':{},'probe':{'job':None,'q':'D1','pi':['D1','D2','D3','D4'],'q_key':'[]'}}
            for j in JOBS: context.write_json(r/(j+'_draft_pool.json'),{**pool,'job':j}); context.write_json(r/(j+'_probe.json'),{**pool['probe'],'job':j}); (r/(j+'_common')).mkdir(exist_ok=True)
            context.write_json(r/'drafts_frozen.json',{'epoch':time.time()}); return {j:rt.Adapter.return_value for j in JOBS},{j:{**pool,'job':j} for j in JOBS}
        calls=[]
        def fake_branch(path,job,h,led,client,shared=None,**kw):
            calls.append(path.name)
            if len(calls)==1:
                (path/job).mkdir(parents=True); context.write_json(path/job/'commit.json',{'material_id':'None'})
                done=br.RunResult('COMPLETE',job,0,'None','ref',1,1,0); context.write_json(path/'branch_result.json',asdict(done)); return done
            led.s['llm_tokens_unknown']+=1; led._save()   # simulated failed attempt; its usage is unknown
            return br.RunResult('INCOMPLETE',job,0,None,None,1,0,0,failure_kind='AGENT_CALL_FAILED',reason='BudgetExhausted')
        stages=[]
        with patch.object(rt,'MeteredClient') as mc, patch.object(sp,'prepare',side_effect=fake_prepare), patch.object(sr,'prepare',side_effect=fake_prepare), patch.object(sr,'target_checks'), patch.object(rt,'Adapter') as ad, \
             patch.object(base,'branch',side_effect=fake_branch), patch.object(sys.modules[__name__],'random_branch'), patch.object(sys.modules[__name__],'menu_branch'), \
             patch.object(dc,'prepare',side_effect=fake_dc_prepare), patch.object(eq,'prepare',side_effect=fake_dc_prepare), patch.object(dc,'fixed_candidates_branch'), \
             patch.object(base,'stage',side_effect=lambda p,j,name,led,**kw: stages.append((p.name,name))), patch.object(sys.modules[__name__],'report'):
            mc.return_value.fatal=False; ad.return_value.baselines.return_value=()
            worker(root)
        return calls,stages
    with TemporaryDirectory() as td:
        root=Path(td)/'run'
        calls,stages=run_worker_with_stop(root)
        assert calls==first and stages==[], (calls,stages)
        assert (root/'labels_withheld.json').exists() and not (root/'all_e_predictions_frozen.json').exists()
        fin=context.read_json(root/'execution_finished.json'); assert any(f.get('stage')=='execution' for f in fin['failures'])
        assert context.read_json(root/'budget.json')['llm_tokens_unknown']==1
        # resume without acceptance is refused by the frozen rule
        try: resume_worker(root); raise AssertionError('resume ignored unknown usage')
        except RuntimeError as exc: assert 'unknown usage' in str(exc)
        assert not (root/'resume.json').exists()
        # resume with explicit acceptance: continues, records the decision, opens labels only at the end
        stages=[]
        ctrl=br.RunResult('COMPLETE',JOBS[1],0,'None','ref',0,0,2)
        with patch.object(rt,'MeteredClient') as mc, patch.object(rt,'Adapter') as ad, patch.object(sr,'target_checks'), \
             patch.object(base,'branch',return_value=br.RunResult('COMPLETE',JOBS[1],0,'None','ref',1,1,0)), patch.object(sys.modules[__name__],'random_branch',return_value=ctrl), patch.object(sys.modules[__name__],'menu_branch',return_value=ctrl), \
             patch.object(dc,'fixed_candidates_branch',return_value=ctrl), \
             patch.object(base,'stage',side_effect=lambda p,j,name,led,**kw: stages.append((p.name,name))), patch.object(sys.modules[__name__],'report'):
            mc.return_value.fatal=False; ad.return_value.baselines.return_value=()
            resume_worker(root,accept_unknown_usage=True)
        rj=context.read_json(root/'resume.json'); assert rj['operator_accepted_unknown_usage'] and rj['unknown_usage_at_resume']==1 and 'UNKNOWN' in rj['failed_attempt_cost']
        assert context.read_json(root/'budget.json')['llm_tokens_unknown']==1  # never reclassified
        assert (root/'labels_withheld_before_resume.json').exists() and (root/'all_e_predictions_frozen.json').exists()
        assert context.read_json(root/'execution_finished.json')['failures']==[]
        assert [s for s in stages if s[0]==first[0]]==[(first[0],'c_b'),(first[0],'freeze_e'),(first[0],'score_e')]
    with TemporaryDirectory() as td:
        root=Path(td)/'run'
        run_worker_with_stop(root)
        stages=[]
        with patch.object(base,'stage',side_effect=lambda p,j,name,led,**kw: stages.append((p.name,name))), patch.object(sys.modules[__name__],'report'):
            finalize_worker(root)
        assert (root/'finalized.json').exists() and (root/'labels_withheld_finalized.json').exists() and not (root/'labels_withheld.json').exists()
        assert stages==[(first[0],'c_b'),(first[0],'freeze_e'),(first[0],'score_e')], stages
        assert (root/'all_e_predictions_frozen.json').exists()
        # (v) a finalized run can never be resumed, even with --accept-unknown-usage; nothing is touched by the refusal.
        before={f.name:f.read_bytes() for f in root.glob('*.json')}
        try: resume_worker(root,accept_unknown_usage=True); raise AssertionError('resumed a finalized run')
        except RuntimeError as exc: assert 'labels are no longer withheld' in str(exc) and 'finalized' in str(exc)
        assert {f.name:f.read_bytes() for f in root.glob('*.json')}==before and not (root/'resume.json').exists()
    # (vi) labels withheld on paper but a C_B/E file already exists in a branch: resume refused before ledger/file/paid-call changes.
    with TemporaryDirectory() as td:
        root=Path(td)/'run'
        run_worker_with_stop(root)
        assert labels_boundary_violations(root)==[]
        context.write_json(root/first[0]/JOBS[0]/'c_b_scores.json',{'job_id':JOBS[0],'cells':{}})
        before={f.name:f.read_bytes() for f in root.glob('*.json')}
        with patch.object(rt,'MeteredClient') as mc:
            try: resume_worker(root,accept_unknown_usage=True); raise AssertionError('resumed after a label was opened')
            except RuntimeError as exc: assert 'c_b_scores.json already opened' in str(exc)
            assert mc.call_count==0
        assert {f.name:f.read_bytes() for f in root.glob('*.json')}==before and (root/'labels_withheld.json').exists() and not (root/'resume.json').exists()
        assert 'unknown_usage_accepted' not in context.read_json(root/'budget.json')
    print('FAULT_PATH_SMOKE_OK: frozen unknown-usage rule, labels withheld on early stop, explicit resume/finalize, resume refused once labels open')

def smoke_skill_revision(cfg):
    """Zero-fit, zero-LLM checks of the skill_revision differences (task §9): explicit population threading,
    frozen inputs, revision contract and early stop, arm dispatch and rendering. Reads only an old exposed T (a85)."""
    from unittest.mock import patch
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    import numpy as np
    from methods.ttha.batch_base import materials, train, data
    assert cfg['planned_fits']==72 and cfg['caps']==dict(max_fit_attempts=74,max_llm_requests=130,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2) and cfg['http_cap']==260
    assert {j:tuple(v) for j,v in cfg['order'].items()}=={'a97_g0':('f0','f_generic','f_old','f_new','menu'),'a97_g1':('menu','f_new','f_old','f_generic','f0')} and cfg['seeds']==[20260925,20260926,20260927]
    for j in ('a97_g0','a97_g1'):
        job=cfg['jobs'][j]
        assert (job['t'],tuple(job['train_range']),tuple(job['c_a']),tuple(job['c_b']),tuple(job['e']))==(25512,(24840,25512),(25512,25560),(25608,25656),(25704,25752,25800,25848)),job
        assert tuple(job['roster_override'])==sr.ROSTERS[j] and len(set(job['roster_override']))==32
    assert set(sr.G0).isdisjoint(sr.G1) and list(sr.G0)==spec.DATASETS['electricity']['roster']
    assert [m for m,_ in sr.MENU]==['MenuU','MenuFM'] and sr.MENU_TIE_ORDER==('FixedMixup','None','MenuU','MenuFM') and cfg['max_tool_corrections']==2 and cfg['evidence_roundtrip'] is False
    assert 'semantics' in cfg['tool_contracts']['build_material']
    # (1) explicit G0 == old default on the same old legal T; G1 reads its own columns; the frozen population cannot be swapped inside a run.
    with TemporaryDirectory() as td:
        a=context.open_job('electricity','a85',Path(td)/'default')
        b=context.open_job('electricity','a85',Path(td)/'g0',roster=sr.G0)
        assert np.array_equal(a.scaler.mean,b.scaler.mean) and np.array_equal(a.scaler.std,b.scaler.std) and np.array_equal(a.parents.X_norm,b.parents.X_norm) and np.array_equal(a.parents.y_norm,b.parents.y_norm) and np.array_equal(a.parents.hours,b.parents.hours)
        assert a.parents.entity_names==b.parents.entity_names==list(sr.G0) and json.dumps(a.overview()['entities'])==json.dumps(b.overview()['entities'])
        assign=[[{'op':'freqmask','mu':0.1}]]+[[] for _ in range(31)]
        ra=materials.build(a,assign,'m');rb=materials.build(b,assign,'m')
        with np.load(ra.path) as za, np.load(rb.path) as zb: assert np.array_equal(za['Xc'],zb['Xc']) and np.array_equal(za['Yc'],zb['Yc'])
        c=context.open_job('electricity','a85',Path(td)/'g1',roster=sr.G1)
        assert c.parents.entity_names==list(sr.G1) and c.slice.rows(c.job.train_range[0],c.job.train_range[1]).shape==(672,32) and not np.array_equal(a.scaler.mean,c.scaler.mean)
        assert [str(x) for x in np.load(Path(td)/'g1'/'a85'/'scaler.npz')['roster']]==list(sr.G1)
        try: context.open_job('electricity','a85',Path(td)/'g1',roster=sr.G0); raise AssertionError('population swap accepted')
        except RuntimeError as exc: assert 'roster drift' in str(exc)
        try: context.open_job('electricity','a85',Path(td)/'default',roster=sr.G1); raise AssertionError('implicit->explicit swap accepted')
        except RuntimeError as exc: assert 'roster' in str(exc)
        try: data.load_slice('electricity',0,24,roster=list(sr.G1[:31])+['999999']); raise AssertionError('unknown column accepted')
        except RuntimeError: pass
        assert train.cell_id('a97_g0','None',1)!=train.cell_id('a97_g1','None',1)
        assert context.resolve_job('electricity','a97_g1').job_id=='a97_g1' and context.resolve_job('electricity','a97_g1').t==25512
    # (2) population and job configuration round-trip into the fit and label subprocesses.
    js=context.resolve_job('electricity','a97_g1',roster=sr.G1)
    with TemporaryDirectory() as td:
        ctx=SimpleNamespace(job=js,job_dir=Path(td),run_dir=Path(td))
        (Path(td)/'cells').mkdir()
        seen={}
        def fake_run(cmd,**kw): seen['cmd']=cmd; return SimpleNamespace(returncode=1)
        led=rt.RuntimeLedger(Path(td)/'budget.json',**CAPS)
        with patch.object(rt.subprocess,'run',side_effect=fake_run):
            try: rt.fit(ctx,SimpleNamespace(material_id='m',alias_of=None),(20260925,),led,REPO)
            except RuntimeError: pass
        i=seen['cmd'].index('--roster'); assert seen['cmd'][i+1]==','.join(sr.G1) and '--job' in seen['cmd'] and seen['cmd'][seen['cmd'].index('--job')+1]=='a97_g1'
        got={}
        with patch.object(train,'fit_one',side_effect=lambda *a,**k: got.update(k) or {'cell_id':'x','scores':{},'train':{'seconds':0}}), patch.object(sys,'argv',['train','--dataset','electricity','--job','a97_g1','--run-dir',td,'--material','m','--seed','1','--roster',','.join(sr.G1)]):
            train.main()
        assert got['roster']==list(sr.G1)
        with patch.object(base.subprocess,'run',side_effect=lambda cmd,**kw: seen.update(stage=cmd) or SimpleNamespace(returncode=0)):
            base.stage(Path(td),'a97_g1','c_b',SimpleNamespace(remaining=lambda:5),roster=sr.G1)
        assert seen['stage'][seen['stage'].index('--roster')+1]==','.join(sr.G1)
        with patch.object(base.commit,'open_c_b',side_effect=lambda *a,**k: got.update(stage_roster=k.get('roster'))), patch.object(sys,'argv',['v1','--stage','c_b','--output',td,'--job','a97_g1','--roster',','.join(sr.G1)]):
            base.main()
        assert got['stage_roster']==list(sr.G1)
    # (3) Slow: H_old and legal usage evidence only; single editable hook; KEEP / no-edit / real edit outcomes.
    h_old=sr.load_h_old(); old_body=h_old.entries[0].body
    ev=sr.build_usage_evidence(); assert len(ev['usage_branches'])==10 and ev['boundary']['e_present'] is False and ev['historical_source_census']['branches']
    refs=frozenset(ev['legal_evidence_refs']); some=sorted(refs)[0]
    class Scripted:
        last_text='{}'
        def __init__(self,outs): self.outs=list(outs); self.calls=[]
        def call(self,role,unit,payload,system,max_tokens=4096,request_timeout=300.):
            self.calls.append(role); text=json.dumps(payload,ensure_ascii=False)
            assert br.json_copy(payload['parent_knowledge'])==br.json_copy(asdict(h_old)) and payload['editable_hooks']==['experiment_guidance'] and unit=='usage_boundary'
            assert not re.search(r'"e":|"external"|e_scores|\ba97|\bg0\b|\bg1\b|25512',text)
            return self.outs.pop(0)
    with TemporaryDirectory() as td:
        st,h=sr.revise(Path(td),Scripted([{'decision':'KEEP','rationale':'no change'}]),h_old,ev); assert (st,h)==('NO_REVISION_PROPOSED',None)
        same={'decision':'PROPOSE','hook':'experiment_guidance','body':old_body,'observable_applicability':{'const':True},'evidence_refs':[some],'rationale':'x'}
        st,h=sr.revise(Path(td),Scripted([same]),h_old,ev); assert (st,h)==('NO_ACTIONABLE_EDIT',None)
        c=Scripted([{**same,'hook':'decision_guidance'},{'decision':'KEEP','rationale':'ok'}]); st,h=sr.revise(Path(td),c,h_old,ev)
        assert st=='NO_REVISION_PROPOSED' and c.calls==['slow','slow_contract_correction'] and context.read_json(Path(td)/'slow_result.json')['contract_corrections_used']==1
        st,h=sr.revise(Path(td),Scripted([{**same,'body':'compare one uniform plan against the incumbent before committing'}]),h_old,ev)
        assert st=='CANDIDATE_TEST_ONLY' and h.version==h_old.version+1 and context.read_json(Path(td)/'slow_result.json')['diff']['body_changed'] is True
        assert h.render({},rt.BATCH_FIELDS)['loaded'][0]['body'].startswith('compare one uniform')
    # (4) worker: KEEP ends the run before any Target is opened; a real edit dispatches all arms with per-arm knowledge, seeds, rosters, contracts.
    def fake_prepare_keep(r,client,caps): context.write_json(r/'early_stop.json',{'status':'NO_REVISION_PROPOSED'}); return None
    with TemporaryDirectory() as td:
        root=Path(td)/'run'
        with patch.object(rt,'MeteredClient') as mc, patch.object(sr,'prepare',side_effect=fake_prepare_keep), patch.object(rt,'Adapter') as ad, patch.object(base,'branch') as brn, patch.object(sys.modules[__name__],'report'):
            mc.return_value.fatal=False
            worker(root)
        assert ad.call_count==0 and brn.call_count==0 and not (root/'labels_withheld.json').exists() and not (root/'all_e_predictions_frozen.json').exists()
        fin=context.read_json(root/'execution_finished.json'); assert fin['branches']==[] and fin['failures']==[]
    k={'f0':br.Knowledge(),'f_generic':sr.load_generic(),'f_old':h_old,'f_new':br.Knowledge(2,(br.Guidance('experiment_guidance','revised body','{"const":true}' and {'const':True},('r',)),))}
    def fake_prepare_edit(r,client,caps):
        context.write_json(r/'frozen_knowledge.json',{a:(asdict(h) if h else None) for a,h in k.items()}); return k
    with TemporaryDirectory() as td:
        root=Path(td)/'run'
        done=br.RunResult('COMPLETE','a97_g0',0,'None','ref',1,1,0)
        with patch.object(rt,'MeteredClient') as mc, patch.object(sr,'prepare',side_effect=fake_prepare_edit), patch.object(rt,'Adapter') as ad, patch.object(sr,'target_checks') as tc, \
             patch.object(base,'branch',return_value=done) as brn, patch.object(sys.modules[__name__],'menu_branch',return_value=done) as mb, patch.object(base,'stage'), patch.object(sys.modules[__name__],'report'):
            mc.return_value.fatal=False; ad.return_value.baselines.return_value=()
            worker(root)
        assert [c.args[0].name for c in brn.call_args_list]==['a97_g0_f0','a97_g0_f_generic','a97_g0_f_old','a97_g0_f_new','a97_g1_f_new','a97_g1_f_old','a97_g1_f_generic','a97_g1_f0']
        assert [c.args[0].name for c in mb.call_args_list]==['a97_g0_menu','a97_g1_menu'] and all(c.kwargs['seeds']==(20260925,20260926,20260927) for c in mb.call_args_list)
        assert mb.call_args_list[0].kwargs['roster']==sr.G0 and mb.call_args_list[1].kwargs['roster']==sr.G1
        for c in brn.call_args_list:
            job,arm=c.args[0].name.split('_',2)[0]+'_'+c.args[0].name.split('_',2)[1],c.args[0].name.split('_',2)[2]
            assert c.args[2] is k[arm] and c.kwargs['roster']==sr.ROSTERS[job] and c.kwargs['seeds']==(20260925,20260926,20260927) and c.kwargs['tool_contracts'] is sp.CONTRACTS and c.kwargs['max_tool_corrections']==2
        assert [c.args[1] for c in tc.call_args_list]==['a97_g0','a97_g1'] and all(c.kwargs['roster']==sr.ROSTERS[c.args[1]] and c.kwargs['seeds']==(20260925,20260926,20260927) for c in ad.call_args_list)
        fin=context.read_json(root/'execution_finished.json'); assert len(fin['branches'])==10 and not fin['failures']
    # (5) rendering: each arm carries only its own knowledge; Menu takes no knowledge at all.
    feats={f:0.5 for f in rt.BATCH_FIELDS}
    assert k['f0'].render(feats,rt.BATCH_FIELDS)['loaded']==[]
    assert [x['body'] for x in k['f_generic'].render(feats,rt.BATCH_FIELDS)['loaded']]==[sp.GENERIC_TEXT]
    assert [x['body'] for x in k['f_old'].render(feats,rt.BATCH_FIELDS)['loaded']]==[old_body]
    assert [x['body'] for x in k['f_new'].render(feats,rt.BATCH_FIELDS)['loaded']]==['revised body']
    import inspect; assert 'knowledge' not in inspect.signature(menu_branch).parameters
    print('SKILL_REVISION_SMOKE_OK: config, population threading, subprocess round-trip, revision contract, early stop, dispatch, rendering')

def smoke_draft_construction(cfg):
    """Zero-fit, zero-LLM checks of the draft_construction additions (task §8): frozen draft flow, per-arm supply and
    contracts, F_select admission before any material/fit, unscored drafts never commit or feed back, preparation and
    dispatch. Reads only an old exposed T (a85, default roster) and synthetic overviews; never this package's Targets."""
    from unittest.mock import patch
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    import csv
    import numpy as np
    assert cfg['planned_fits']==72 and cfg['caps']==dict(max_fit_attempts=74,max_llm_requests=96,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2) and cfg['http_cap']==192
    assert {j:tuple(v) for j,v in cfg['order'].items()}=={'a65_g3':('f_free','f_select','f_edit','random','menu'),'a85_g3':('menu','random','f_edit','f_select','f_free')} and cfg['seeds']==[20260928,20260929,20260930]
    exp={'a65_g3':(17088,(16416,17088),(17088,17136),(17184,17232),(17280,17328,17376,17424)),'a85_g3':(22344,(21672,22344),(22344,22392),(22440,22488),(22536,22584,22632,22680))}
    for j,(t,tr,ca,cb,e) in exp.items():
        job=cfg['jobs'][j]
        assert (job['t'],tuple(job['train_range']),tuple(job['c_a']),tuple(job['c_b']),tuple(job['e']))==(t,tr,ca,cb,e),job
        assert tuple(job['roster_override'])==dc.G3 and len(set(job['roster_override']))==32
    assert max(exp['a65_g3'][4])+spec.H<=exp['a85_g3'][1][0]   # disjoint complete workloads
    with open(spec.DATASETS['electricity']['path'],encoding='utf-8') as f: hdr=next(csv.reader(f))
    cols=sorted(c for c in hdr if c!='date')
    assert tuple(cols[96:128])==dc.G3 and tuple(cols[64:96])==dc.G2 and '182' in dc.G2 and '182' not in dc.G3
    assert set(dc.G3).isdisjoint(sr.G0) and set(dc.G3).isdisjoint(sr.G1) and set(dc.G3).isdisjoint(dc.G2)
    assert cfg['population_revision']['zero_scale_rule_unchanged'] is True and str(DEFAULT).endswith('_g3') and dc.G2_RUN.resolve()!=DEFAULT.resolve()
    assert cfg['max_tool_corrections']==2 and cfg['evidence_roundtrip'] is False and cfg['environment']['torch'] and cfg['knowledge'].startswith('H0 empty')
    # (1) draft pool: deterministic frozen flow on a synthetic overview; exclusion by compiled assignment only; 64 cap.
    rng=np.random.RandomState(7)
    table=[{f:(0 if f=='missing_count' else spec.N_PARENTS if f=='n_parent_pairs' else float(rng.rand())) for f in spec.OBS_FIELDS} for _ in range(32)]
    pool=dc.generate_pool('a65_g3',table); pool2=dc.generate_pool('a65_g3',table)
    assert json.dumps(pool,sort_keys=True)==json.dumps(pool2,sort_keys=True) and pool['status']=='COMPLETE' and [d['draft_id'] for d in pool['drafts']]==['D1','D2','D3','D4']
    keys=[policy.assignment_key(d['assignment']) for d in pool['drafts']]
    assert len(set(keys))==4 and not set(keys)&set(pool['excluded_keys'].values()) and pool['fits']==0 and pool['materials_built']==0
    assert all(d['proposal_seed']==2026091400+d['j'] for d in pool['drafts']) and pool['n_proposals_tried']<=64
    assert [a['status'] for a in pool['attempts']].count('kept')==4 and pool['attempts'][-1]['status']=='kept'
    for d in pool['drafts']:
        c=policy.compile_policy(d['policy'],table); assert c['assignment']==d['assignment'] and c['rule_index']==d['rule_index']
        assert d['policy']['rationale']==dc.DRAFT_RATIONALE and set(d['policy'])=={'default','rules','rationale','observation_fields_used'}
        assert sum(d['entities_by_rule'].values())==32 and d['n_identity']+d['n_entities_changed']==32
    assert dc.generate_pool('a85_g3',table)['drafts'][0]['proposal_seed']>=2026092400
    with patch.object(policy,'random_policy',side_effect=lambda seed: policy.identity_policy()):
        bad=dc.generate_pool('a65_g3',table)
    assert bad['status']=='DRAFT_POOL_INCOMPLETE' and bad['n_proposals_tried']==64 and bad['drafts']==[] and all(a['status']=='duplicate_of_None' for a in bad['attempts'])
    def dict_keys(x,acc):
        if isinstance(x,dict):
            for k,v in x.items(): acc.add(k); dict_keys(v,acc)
        elif isinstance(x,list):
            for v in x: dict_keys(v,acc)
        return acc
    # (2)-(3) on a real old T (a85, default roster): supply per arm, contracts, admission before any material/fit.
    with TemporaryDirectory() as td:
        led=rt.RuntimeLedger(Path(td)/'budget.json',**CAPS)
        plain=rt.Adapter(Path(td)/'plain','a85',led,REPO)
        assert plain.ctx.stage=='material' and plain.ctx.slice.rows(*plain.ctx.job.material_rows).shape==(672,32)
        rp=dc.generate_pool('a65_g3',plain.overview()['entities'])   # the job label only fixes the seed stream; the T here is the old exposed a85
        rp['job']='a85'
        assert 'unscored_drafts' not in plain.overview()
        sel=dc.DraftAdapter(Path(td)/'sel','a85',led,REPO,pool=rp,select_only=True,tool_error_feedback=True)
        edit=dc.DraftAdapter(Path(td)/'edit','a85',led,REPO,pool=rp,select_only=False,tool_error_feedback=True)
        for a in (sel,edit):
            ov=a.overview(); v=ov['unscored_drafts']
            assert [d['draft_id'] for d in v['drafts']]==['D1','D2','D3','D4'] and 'assignment' not in v['drafts'][0] and 'no quality order' in v['note']
            assert not dict_keys(v['drafts'],set())&{'c_a','c_b','e','external','feedback','model_refs','model_path','model_seeds','fit_id','scores','loss','cells','ca_losses'} and br.public_copy(ov)
            assert json.dumps(ov['entities'])==json.dumps(plain.overview()['entities'])
        assert sel.overview()['unscored_drafts']==edit.overview()['unscored_drafts']
        try: dc.DraftAdapter(Path(td)/'other','a85',led,REPO,pool={**rp,'job':'a65_g3'}); raise AssertionError('foreign pool accepted')
        except ValueError: pass
        try: dc.DraftAdapter(Path(td)/'other2','a85',led,REPO,pool={**rp,'status':'DRAFT_POOL_INCOMPLETE'}); raise AssertionError('incomplete pool accepted')
        except ValueError: pass
        cf,cs,ce=dc.arm_contracts('f_free'),dc.arm_contracts('f_select'),dc.arm_contracts('f_edit')
        assert cf==dc.CONTRACTS and 'draft_rule' not in json.dumps(cf) and set(cs)==set(ce)==br.TOOLS
        assert 'draft_rule' in cs['build_material'] and 'draft_rule' in ce['build_material'] and 'only' in cs['build_material']['draft_rule'] and 'any legal policy' in ce['build_material']['draft_rule']
        for t in br.TOOLS:
            assert {k:v for k,v in cs[t].items() if k!='draft_rule'}=={k:v for k,v in ce[t].items() if k!='draft_rule'}==cf[t]
        assert not re.search(r'histor|bold|better|recommend|winner|effect',cs['build_material']['draft_rule']+ce['build_material']['draft_rule'],re.I)
        d1=rp['drafts'][0]['policy']
        with patch.object(rt,'fit',side_effect=AssertionError('no fit may happen here')):
            c=sel.build_material({'plan_id':'mine','policy':d1},remaining_seconds=10.); assert c.material_spec['assignment']==rp['drafts'][0]['assignment']
            c=sel.build_material({'plan_id':'same_text_differs','policy':{**d1,'rationale':'my own words'}},remaining_seconds=10.); assert sel.refs['same_text_differs'].alias_of=='mine'
            c=sel.build_material({'plan_id':'ident','policy':policy.identity_policy()},remaining_seconds=10.); assert c.material_spec['assignment']==[[]]*32
            changed=br.json_copy(d1); changed['default']['steps']=[{'op':'freqmask','mu':0.2}]+changed['default']['steps'][:1]
            if policy.compile_policy(changed,plain.overview()['entities'])['key'] in sel.admitted: changed['rules']=[]
            before=json.dumps(context.read_json(sel.ctx.job_dir/'materials'/'index.json'),sort_keys=True)
            try: sel.build_material({'plan_id':'edited','policy':changed},remaining_seconds=10.); raise AssertionError('out-of-pool build admitted')
            except br.ToolInputError as exc: assert 'D1..D4' in str(exc) and exc.details['admitted_assignments']==['D1','D2','D3','D4','FixedMixup','None']
            assert 'edited' not in sel.refs and json.dumps(context.read_json(sel.ctx.job_dir/'materials'/'index.json'),sort_keys=True)==before
            assert led.s['fit_attempts']==0
            c=edit.build_material({'plan_id':'edited','policy':changed},remaining_seconds=10.); assert 'edited' in edit.refs and c.material_spec['assignment']!=rp['drafts'][0]['assignment']
            c=plain.build_material({'plan_id':'edited','policy':changed},remaining_seconds=10.); assert 'edited' in plain.refs
            strict=dc.DraftAdapter(Path(td)/'strict','a85',led,REPO,pool=rp,select_only=True,tool_error_feedback=False)
            try: strict.build_material({'plan_id':'edited','policy':changed},remaining_seconds=10.); raise AssertionError('admitted')
            except ValueError as exc: assert not isinstance(exc,br.ToolInputError)
        assert led.s['fit_attempts']==0
        # (4) run_job with a scripted client: an unscored draft cannot be committed or fed back; two new-evaluation slots count as live; admission rejections are ordinary corrections.
        fits=[]
        def fake_fit(ctx,ref,seeds,ledger,repo,feedback=True):
            fits.append(ref.material_id); return [{'model_path':'m_%s_%d'%(ref.material_id,s),'scores':{'c_a':{'per_origin_entity_normalized_mse':[[0.1]*32]*2,'n_nonfinite_predictions':0}}} for s in seeds]
        class Scripted:
            def __init__(self,outs): self.outs=list(outs); self.seen=[]
            def __call__(self,req): self.seen.append(req); return self.outs.pop(0)
        with patch.object(rt,'fit',side_effect=fake_fit):
            a=dc.DraftAdapter(Path(td)/'job','a85',led,REPO,pool=rp,select_only=True,tool_error_feedback=True)
            bl=a.baselines(); d=[x['policy'] for x in rp['drafts']]
            client=Scripted([{'actions':[{'tool':'commit','arguments':{'plan_id':'D1','reason':'draft'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'D1','policy':d[0]}},{'tool':'evaluate','arguments':{'plan_id':'D1'}},
                                         {'tool':'build_material','arguments':{'plan_id':'D2','policy':d[1]}},{'tool':'evaluate','arguments':{'plan_id':'D2'}},
                                         {'tool':'build_material','arguments':{'plan_id':'D3','policy':d[2]}},{'tool':'evaluate','arguments':{'plan_id':'D3'}}]}])
            res=br.run_job(job_id='a85',knowledge=br.Knowledge(),adapter=a,client=client,seeds=rt.SEEDS,baselines=bl,allowed_features=rt.BATCH_FIELDS,tool_contracts=dc.arm_contracts('f_select'),max_tool_corrections=2)
            ev=[x for x in res.trace if x['event']=='tool_rejected']
            assert res.status=='INCOMPLETE' and res.failure_kind=='BUDGET_EXHAUSTED' and 'new candidate budget' in res.reason and res.new_evaluations==2
            assert len(ev)==1 and ev[0]['tool']=='commit' and 'unknown plan' in ev[0]['error']['message'] and fits==['None','FixedMixup','D1','D2']
            assert client.seen[0]['overview']['unscored_drafts']['drafts'][0]['draft_id']=='D1' and all(c['plan_id'] in ('None','FixedMixup') for c in client.seen[0]['candidates'])
            assert client.seen[0]['tool_contracts']['build_material']['draft_rule']==dc.arm_contracts('f_select')['build_material']['draft_rule']
            fits.clear()
            a=dc.DraftAdapter(Path(td)/'job2','a85',led,REPO,pool=rp,select_only=True,tool_error_feedback=True); bl=a.baselines()
            client=Scripted([{'actions':[{'tool':'build_material','arguments':{'plan_id':'x','policy':changed}},{'tool':'evaluate','arguments':{'plan_id':'x'}}]},
                             {'actions':[{'tool':'commit','arguments':{'plan_id':'None','reason':'baseline'}}]}])
            res=br.run_job(job_id='a85',knowledge=br.Knowledge(),adapter=a,client=client,seeds=rt.SEEDS,baselines=bl,allowed_features=rt.BATCH_FIELDS,tool_contracts=dc.arm_contracts('f_select'),max_tool_corrections=2)
            ev=[x for x in res.trace if x['event']=='tool_rejected']
            assert res.status=='COMPLETE' and res.committed_plan_id=='None' and res.new_evaluations==0 and len(ev)==1 and ev[0]['error']['admitted_assignments'][0]=='D1' and ev[0]['unexecuted_actions'][0]['tool']=='evaluate'
            assert fits==['None','FixedMixup'] and 'x' not in a.refs
        assert led.s['fit_attempts']==0
    # (5) preparation and dispatch (patched adapters/branches): pools frozen before any baseline fit; per-arm factories, contracts, controls, skips.
    synthetic={'entities':table,'n_entities':32,'train_rows':[0,672],'batch_features':{}}
    with TemporaryDirectory() as td:
        root=Path(td)/'run'; root.mkdir()
        log=[]
        def fake_adapter(run_dir,job,led,repo,**kw):
            if job=='a85_g3': raise RuntimeError('ELIGIBILITY_FAIL zero-scale entities: 1')
            m=SimpleNamespace(ctx=SimpleNamespace(job=SimpleNamespace(roster=list(dc.G3),roster_override=dc.G3),stage='material'),overview=lambda: synthetic,baselines=lambda: log.append('baselines') or ())
            log.append('open:'+job); return m
        led=rt.RuntimeLedger(root/'budget.json',**CAPS)
        with patch.object(rt,'Adapter',side_effect=fake_adapter):
            commons,pools=dc.prepare(root,led,CAPS)
        assert list(commons)==['a65_g3'] and pools['a85_g3'] is None and pools['a65_g3']['status']=='COMPLETE' and (root/'a85_g3_eligibility.json').exists() and (root/'a65_g3_draft_pool.json').exists()
        assert context.read_json(root/'drafts_frozen.json')['jobs']=={'a65_g3':'COMPLETE','a85_g3':'ELIGIBILITY_FAILED'} and context.read_json(root/'a65_g3_target_checks.json')['roster']==list(dc.G3) and log==['open:a65_g3']
        assert dc.load_pools(root)['a65_g3']==pools['a65_g3'] and context.read_json(root/'frozen_config.json')['arm_contracts']['f_select']==dc.arm_contracts('f_select')
    with TemporaryDirectory() as td:
        root=Path(td)/'run'
        pool_ok={**dc.generate_pool('a65_g3',table),'job':'a65_g3'}; pool_bad={**dc.generate_pool('a85_g3',table),'job':'a85_g3','status':'DRAFT_POOL_INCOMPLETE'}
        log=[]
        def fake_prepare(r,led,caps):
            log.append('prepare'); m=SimpleNamespace(baselines=lambda: log.append('baselines') or ())
            for j,p in (('a65_g3',pool_ok),('a85_g3',pool_bad)): context.write_json(r/(j+'_draft_pool.json'),p)
            context.write_json(r/'drafts_frozen.json',{'epoch':0}); return {'a65_g3':m,'a85_g3':m},{'a65_g3':pool_ok,'a85_g3':pool_bad}
        done=br.RunResult('COMPLETE','a65_g3',0,'None','ref',1,1,0)
        with patch.object(rt,'MeteredClient') as mc, patch.object(dc,'prepare',side_effect=fake_prepare), patch.object(base,'branch',return_value=done) as brn, \
             patch.object(dc,'fixed_candidates_branch',return_value=done) as fcb, patch.object(base,'stage'), patch.object(sys.modules[__name__],'report'):
            mc.return_value.fatal=False
            worker(root)
        assert log[0]=='prepare' and log.count('baselines')==2
        assert [c.args[0].name for c in brn.call_args_list]==['a65_g3_f_free','a65_g3_f_select','a65_g3_f_edit','a85_g3_f_free']
        for c in brn.call_args_list:
            arm=c.args[0].name.split('_',2)[2]; f=c.kwargs['adapter_factory']
            assert c.args[2]==br.Knowledge() and c.kwargs['tool_contracts']==dc.arm_contracts(arm) and c.kwargs['max_tool_corrections']==2 and c.kwargs['evidence_roundtrip'] is False and c.kwargs['roster']==dc.G3 and c.kwargs['seeds']==dc.SEEDS
            if arm=='f_free': assert f is None
            else: assert f.func is dc.DraftAdapter and f.keywords=={'pool':pool_ok,'select_only':arm=='f_select'}
        assert [(c.args[0].name,c.kwargs['tag']) for c in fcb.call_args_list]==[('a65_g3_random','random'),('a65_g3_menu','menu'),('a85_g3_menu','menu')]
        rnd=fcb.call_args_list[0].kwargs; assert [m for m,_ in rnd['plans']]==['D1','D2'] and rnd['plans'][0][1]==pool_ok['drafts'][0]['policy'] and rnd['tie_order']==dc.RANDOM_TIE_ORDER and rnd['expected_assignments']['D2']==pool_ok['drafts'][1]['assignment']
        menu=fcb.call_args_list[1].kwargs; assert menu['plans']==list(dc.MENU) and menu['tie_order']==dc.MENU_TIE_ORDER and 'expected_assignments' not in menu
        assert sorted(p.name for p in root.glob('*_skipped.json'))==['a85_g3_f_edit_skipped.json','a85_g3_f_select_skipped.json','a85_g3_random_skipped.json']
        fin=context.read_json(root/'execution_finished.json'); assert len(fin['branches'])==7 and not fin['failures']
    print('DRAFT_CONSTRUCTION_SMOKE_OK: config, frozen draft flow, per-arm supply and contracts, F_select admission before fit, unscored drafts never commit, preparation, dispatch')

def smoke_exploration_quota(cfg):
    """Zero-fit, zero-LLM checks of the exploration_quota additions (task §8): frozen pool + probe streams, identical Q for
    every Fast arm, F_quota execution semantics inside the real run_job (order of slots, rejection before slot/fit,
    commit gate, rename/baseline/partial identity, restore), the same actions legal in F_auto/F_text and old studies,
    preparation and dispatch. Reads only an old exposed T (a85, default roster) and synthetic overviews."""
    from unittest.mock import patch
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    import csv
    import numpy as np
    assert cfg['planned_fits']==72 and cfg['caps']==dict(max_fit_attempts=74,max_llm_requests=96,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2) and cfg['http_cap']==192
    assert {j:tuple(v) for j,v in cfg['order'].items()}=={'a65_g4':('f_auto','f_text','f_quota','random','menu'),'a85_g4':('menu','random','f_quota','f_text','f_auto')} and cfg['seeds']==[20261001,20261002,20261003]
    exp={'a65_g4':(17088,(16416,17088),(17088,17136),(17184,17232),(17280,17328,17376,17424)),'a85_g4':(22344,(21672,22344),(22344,22392),(22440,22488),(22536,22584,22632,22680))}
    for j,(t,tr,ca,cb,e) in exp.items():
        job=cfg['jobs'][j]
        assert (job['t'],tuple(job['train_range']),tuple(job['c_a']),tuple(job['c_b']),tuple(job['e']))==(t,tr,ca,cb,e),job
        assert tuple(job['roster_override'])==eq.G4 and len(set(job['roster_override']))==32
    with open(spec.DATASETS['electricity']['path'],encoding='utf-8') as f: hdr=next(csv.reader(f))
    cols=sorted(c for c in hdr if c!='date')
    assert tuple(cols[128:160])==eq.G4 and set(eq.G4).isdisjoint(sr.G0|set()) if False else tuple(cols[128:160])==eq.G4
    assert all(set(eq.G4).isdisjoint(g) for g in (sr.G0,sr.G1,dc.G2,dc.G3))
    assert cfg['quota']['applies_to']=='f_quota only' and cfg['probe']['seed_base']==2026091600 and cfg['draft_pool']['seed_base']==2026091500 and cfg['environment']['torch']
    assert cfg['max_tool_corrections']==2 and cfg['evidence_roundtrip'] is False and str(DEFAULT).endswith('dev_batch_research_exploration_quota')
    # (1) frozen streams on a synthetic overview: same T/seed -> same drafts and pi/Q; independent seeds per job; no quality fields.
    rng=np.random.RandomState(11)
    table=[{f:(0 if f=='missing_count' else spec.N_PARENTS if f=='n_parent_pairs' else float(rng.rand())) for f in spec.OBS_FIELDS} for _ in range(32)]
    pool=eq.generate_pool('a65_g4',table); pool2=eq.generate_pool('a65_g4',table)
    assert json.dumps(pool,sort_keys=True)==json.dumps(pool2,sort_keys=True) and pool['status']=='COMPLETE' and all(d['proposal_seed']==2026091500+d['j'] for d in pool['drafts'])
    assert '2026091500' in pool['rule'] and eq.generate_pool('a85_g4',table)['drafts'][0]['proposal_seed']>=2026092500
    assert json.dumps(dc.generate_pool('a65_g3',table),sort_keys=True)!=json.dumps(pool,sort_keys=True) and '2026091400' in dc.generate_pool('a65_g3',table)['rule']   # the G3 stream is untouched
    pr=eq.draw_probe('a65_g4',pool); pr2=eq.draw_probe('a65_g4',pool)
    assert pr==pr2 and sorted(pr['pi'])==['D1','D2','D3','D4'] and pr['q']==pr['pi'][0] and pr['seed']==2026091600 and eq.draw_probe('a85_g4',eq.generate_pool('a85_g4',table))['seed']==2026092600
    assert pr['q_key']==policy.assignment_key(next(d for d in pool['drafts'] if d['draft_id']==pr['q'])['assignment']) and pr['q_key'] not in pool['excluded_keys'].values()
    assert not re.search(r'scores?|loss|better|confiden|distance',json.dumps({k:v for k,v in pr.items() if k!='meaning'}),re.I)
    with patch.object(policy,'random_policy',side_effect=lambda seed: policy.identity_policy()):
        bad=eq.generate_pool('a65_g4',table)
    assert bad['status']=='DRAFT_POOL_INCOMPLETE' and eq.draw_probe('a65_g4',bad) is None
    # (2)-(4) on a real old T (a85, default roster).
    fits=[]
    def fake_fit(ctx,ref,seeds,ledger,repo,feedback=True):
        fits.append(ref.material_id); return [{'model_path':'m_%s_%d'%(ref.material_id,s),'scores':{'c_a':{'per_origin_entity_normalized_mse':[[0.1]*32]*2,'n_nonfinite_predictions':0}}} for s in seeds]
    class Scripted:
        def __init__(self,outs): self.outs=list(outs); self.seen=[]
        def __call__(self,req): self.seen.append(req); return self.outs.pop(0)
    with TemporaryDirectory() as td:
        led=rt.RuntimeLedger(Path(td)/'budget.json',**CAPS)
        plain=rt.Adapter(Path(td)/'plain','a85',led,REPO)
        rp=eq.generate_pool('a65_g4',plain.overview()['entities']); rp['job']='a85'; rp['probe']={**eq.draw_probe('a65_g4',rp),'job':'a85'}
        Q=rp['probe']['q']; pi=rp['probe']['pi']; pol={d['draft_id']:d['policy'] for d in rp['drafts']}
        nonq=[d for d in pi if d!=Q]
        arms={a:eq.QuotaAdapter(Path(td)/a,'a85',led,REPO,pool=rp,arm=a,tool_error_feedback=True) for a in eq.FAST_ARMS}
        ovs={a:x.overview() for a,x in arms.items()}
        assert all(o['random_probe_id']==Q and o['random_probe_note']==eq.PROBE_NOTE and o['unscored_drafts']==ovs['f_auto']['unscored_drafts'] for o in ovs.values())
        assert json.dumps(ovs['f_auto'],sort_keys=True)==json.dumps(ovs['f_quota'],sort_keys=True) and br.public_copy(ovs['f_quota'])
        assert 'random_probe_id' not in dc.DraftAdapter(Path(td)/'old','a85',led,REPO,pool={k:v for k,v in rp.items() if k!='probe'}).overview()
        assert rt.Adapter.check_evaluate(plain,'x',None) is None and rt.Adapter.check_commit(plain,'x',None) is None and plain.check_compiled('x',{}) is None
        try: eq.QuotaAdapter(Path(td)/'bad','a85',led,REPO,pool=rp,arm='f_edit'); raise AssertionError('foreign arm accepted')
        except ValueError: pass
        try: eq.QuotaAdapter(Path(td)/'bad2','a85',led,REPO,pool={k:v for k,v in rp.items() if k!='probe'},arm='f_quota'); raise AssertionError('missing probe accepted')
        except ValueError: pass
        ca,ct,cq=eq.arm_contracts('f_auto'),eq.arm_contracts('f_text'),eq.arm_contracts('f_quota')
        for c in (ca,ct,cq):
            assert set(c)==br.TOOLS and c['build_material']['draft_rule']==dc.arm_contracts('f_select')['build_material']['draft_rule'] and 'evaluation_requirement' in c['commit']
        assert ct['evaluate']['arm_rule'].startswith('SUGGESTION') and cq['evaluate']['arm_rule'].startswith('CONTRACT') and eq.CORE_RULE in ct['evaluate']['arm_rule'] and eq.CORE_RULE in cq['evaluate']['arm_rule']
        assert 'no evaluation suggestion and no gate' in ca['evaluate']['arm_rule'] and 'none for this arm' in ca['commit']['evaluation_requirement']
        assert not re.search(r'histor|bold|risk|winner|aggressive|mild',eq.CORE_RULE+eq.PROBE_NOTE+ca['evaluate']['arm_rule']+ct['evaluate']['arm_rule']+cq['evaluate']['arm_rule'],re.I)
        assert 'evaluate' in ca and ca['evaluate']['arm_rule']!=ct['evaluate']['arm_rule']!=cq['evaluate']['arm_rule']
        rj=lambda a,client: br.run_job(job_id='a85',knowledge=br.Knowledge(),adapter=a,client=client,seeds=rt.SEEDS,baselines=a.baselines(),allowed_features=rt.BATCH_FIELDS,tool_contracts=eq.arm_contracts(a.arm),max_tool_corrections=2)
        with patch.object(rt,'fit',side_effect=fake_fit):
            # (a) f_quota: non-Q first is allowed; a second non-Q before Q is rejected before slot/fit; Q under another id is recognised; then None may be committed.
            a=eq.QuotaAdapter(Path(td)/'qa','a85',led,REPO,pool=rp,arm='f_quota',tool_error_feedback=True)
            client=Scripted([{'actions':[{'tool':'build_material','arguments':{'plan_id':'first','policy':pol[nonq[0]]}},{'tool':'evaluate','arguments':{'plan_id':'first'}},
                                         {'tool':'build_material','arguments':{'plan_id':'second','policy':pol[nonq[1]]}},{'tool':'evaluate','arguments':{'plan_id':'second'}},{'tool':'commit','arguments':{'plan_id':'first','reason':'x'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'myq','policy':{**pol[Q],'rationale':'renamed probe'}}},{'tool':'evaluate','arguments':{'plan_id':'myq'}},{'tool':'commit','arguments':{'plan_id':'None','reason':'baseline'}}]}])
            res=rj(a,client); ev=[x for x in res.trace if x['event']=='tool_rejected']
            assert res.status=='COMPLETE' and res.committed_plan_id=='None' and res.new_evaluations==2 and len(ev)==1 and ev[0]['tool']=='evaluate' and ev[0]['arguments']['plan_id']=='second'
            assert ev[0]['error']['message'].startswith('quota:') and ev[0]['error']['random_probe_id']==Q and ev[0]['error']['q_evaluated'] is False and ev[0]['error']['non_q_new_candidates']==['first'] and [u['tool'] for u in ev[0]['unexecuted_actions']]==['commit']
            assert fits==['None','FixedMixup','first','myq'] and 'second' in a.refs and 'second' not in a.fitted and a.q_complete()
            assert client.seen[0]['overview']['random_probe_id']==Q and client.seen[0]['tool_contracts']['evaluate']['arm_rule'].startswith('CONTRACT') and client.seen[1]['remaining']['tool_corrections']==1
            # (b) f_quota: commit (None) before Q is refused before anything is written; Q first then commit None works; a third quota rejection ends the branch.
            fits.clear(); a=eq.QuotaAdapter(Path(td)/'qb','a85',led,REPO,pool=rp,arm='f_quota',tool_error_feedback=True)
            client=Scripted([{'actions':[{'tool':'commit','arguments':{'plan_id':'None','reason':'early'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':Q,'policy':pol[Q]}},{'tool':'evaluate','arguments':{'plan_id':Q}},{'tool':'commit','arguments':{'plan_id':'FixedMixup','reason':'after Q'}}]}])
            res=rj(a,client); ev=[x for x in res.trace if x['event']=='tool_rejected']
            assert res.status=='COMPLETE' and res.committed_plan_id=='FixedMixup' and len(ev)==1 and ev[0]['tool']=='commit' and 'refused until the random probe' in ev[0]['error']['message'] and fits==['None','FixedMixup',Q]
            assert not (Path(td)/'qb'/'a85'/'commit.json').exists()
            fits.clear(); a=eq.QuotaAdapter(Path(td)/'qc','a85',led,REPO,pool=rp,arm='f_quota',tool_error_feedback=True)
            client=Scripted([{'actions':[{'tool':'commit','arguments':{'plan_id':'None','reason':'1'}}]}]*3)
            res=rj(a,client); assert res.status=='INCOMPLETE' and res.failure_kind=='BUDGET_EXHAUSTED' and 'correction budget' in res.reason and res.committed_plan_id is None and sum(x['event']=='tool_rejected' for x in res.trace)==3 and fits==['None','FixedMixup']
            # (c) f_quota: Q first, then a non-Q, commit the non-Q (Q completion does not force delivery of Q).
            fits.clear(); a=eq.QuotaAdapter(Path(td)/'qd','a85',led,REPO,pool=rp,arm='f_quota',tool_error_feedback=True)
            client=Scripted([{'actions':[{'tool':'build_material','arguments':{'plan_id':Q,'policy':pol[Q]}},{'tool':'evaluate','arguments':{'plan_id':Q}},
                                         {'tool':'build_material','arguments':{'plan_id':'other','policy':pol[nonq[0]]}},{'tool':'evaluate','arguments':{'plan_id':'other'}},{'tool':'commit','arguments':{'plan_id':'other','reason':'x'}}]}])
            res=rj(a,client); assert res.status=='COMPLETE' and res.committed_plan_id=='other' and res.new_evaluations==2 and not [x for x in res.trace if x['event']=='tool_rejected']
            # (d) identity: baselines, a built-but-unfitted Q and a partially fitted Q never count as Q complete; restore recovers Q from real candidate state.
            a=eq.QuotaAdapter(Path(td)/'qe','a85',led,REPO,pool=rp,arm='f_quota',tool_error_feedback=True); a.baselines(); assert not a.q_complete() and a.non_q_new()==[]
            c=a.build_material({'plan_id':'qq','policy':pol[Q]},remaining_seconds=10.); assert not a.q_complete()
            a.fitted['qq']=br.Candidate('qq',c.material_spec,'a85',model_seeds=rt.SEEDS[:2],model_refs=('m1','m2'),ca_losses=()); assert not a.q_complete()
            del a.fitted['qq']; a.evaluate(c,rt.SEEDS,feedback=True,remaining_seconds=10.); assert a.q_complete()
            b=eq.QuotaAdapter(Path(td)/'qe','a85',led,REPO,pool=rp,arm='f_quota',tool_error_feedback=True); b.baselines(); assert not b.q_complete()
            b.restore(['qq']); assert b.q_complete() and led.s['fit_attempts']==0
            # (e) the same commit-before-Q is legal in F_auto / F_text and in the old draft study's adapter.
            for arm in ('f_auto','f_text'):
                fits.clear(); a=eq.QuotaAdapter(Path(td)/('legal_'+arm),'a85',led,REPO,pool=rp,arm=arm,tool_error_feedback=True)
                res=rj(a,Scripted([{'actions':[{'tool':'commit','arguments':{'plan_id':'None','reason':'no new fits'}}]}]))
                assert res.status=='COMPLETE' and res.committed_plan_id=='None' and not [x for x in res.trace if x['event']=='tool_rejected'] and fits==['None','FixedMixup']
            old=dc.DraftAdapter(Path(td)/'legacy','a85',led,REPO,pool={k:v for k,v in rp.items() if k!='probe'},select_only=True,tool_error_feedback=True)
            res=br.run_job(job_id='a85',knowledge=br.Knowledge(),adapter=old,client=Scripted([{'actions':[{'tool':'commit','arguments':{'plan_id':'None','reason':'x'}}]}]),seeds=rt.SEEDS,baselines=old.baselines(),allowed_features=rt.BATCH_FIELDS,tool_contracts=dc.arm_contracts('f_select'),max_tool_corrections=2)
            assert res.status=='COMPLETE' and not [x for x in res.trace if x['event']=='tool_rejected']
        assert led.s['fit_attempts']==0
        # pool + probe files round-trip through load_pools with the same Q
        (Path(td)/'lp').mkdir(); context.write_json(Path(td)/'lp'/'a65_g4_draft_pool.json',{k:v for k,v in rp.items() if k!='probe'}); context.write_json(Path(td)/'lp'/'a65_g4_probe.json',rp['probe'])
        lp=eq.load_pools(Path(td)/'lp'); assert lp['a65_g4']['probe']['q']==Q and lp['a65_g4']['probe']['pi']==pi and lp['a85_g4'] is None
    # (5) preparation with patched adapters; dispatch with patched branches.
    synthetic={'entities':table,'n_entities':32,'train_rows':[0,672],'batch_features':{}}
    with TemporaryDirectory() as td:
        root=Path(td)/'run'; root.mkdir(); log=[]
        def fake_adapter(run_dir,job,led,repo,**kw):
            m=SimpleNamespace(ctx=SimpleNamespace(job=SimpleNamespace(roster=list(eq.G4),roster_override=eq.G4),stage='material'),overview=lambda: synthetic,baselines=lambda: ())
            log.append(job); return m
        led=rt.RuntimeLedger(root/'budget.json',**CAPS)
        with patch.object(rt,'Adapter',side_effect=fake_adapter):
            commons,pools=eq.prepare(root,led,CAPS)
        assert log==['a65_g4','a85_g4'] and all(pools[j]['status']=='COMPLETE' and pools[j]['probe']['q']==pools[j]['probe']['pi'][0] for j in eq.JOBS)
        fz=context.read_json(root/'drafts_frozen.json'); assert fz['probes']=={j:pools[j]['probe']['q'] for j in eq.JOBS} and fz['fit_attempts_at_freeze']==0
        assert eq.load_pools(root)=={j:{**context.read_json(root/(j+'_draft_pool.json')),'probe':context.read_json(root/(j+'_probe.json'))} for j in eq.JOBS}
        assert context.read_json(root/'frozen_config.json')['arm_contracts']['f_quota']==eq.arm_contracts('f_quota') and pools['a65_g4']['probe']['seed']!=pools['a85_g4']['probe']['seed']
    with TemporaryDirectory() as td:
        root=Path(td)/'run'
        pools={j:{**eq.generate_pool(j,table),'job':j} for j in eq.JOBS}
        for j in eq.JOBS: pools[j]['probe']=eq.draw_probe(j,pools[j])
        def fake_prepare(r,led,caps):
            for j,p in pools.items(): context.write_json(r/(j+'_draft_pool.json'),{k:v for k,v in p.items() if k!='probe'}); context.write_json(r/(j+'_probe.json'),p['probe'])
            context.write_json(r/'drafts_frozen.json',{'epoch':0}); return {j:SimpleNamespace(baselines=lambda: ()) for j in eq.JOBS},pools
        done=br.RunResult('COMPLETE','a65_g4',0,'None','ref',1,1,0)
        with patch.object(rt,'MeteredClient') as mc, patch.object(eq,'prepare',side_effect=fake_prepare), patch.object(base,'branch',return_value=done) as brn, \
             patch.object(dc,'fixed_candidates_branch',return_value=done) as fcb, patch.object(base,'stage'), patch.object(sys.modules[__name__],'report'):
            mc.return_value.fatal=False
            worker(root)
        assert [c.args[0].name for c in brn.call_args_list]==['a65_g4_f_auto','a65_g4_f_text','a65_g4_f_quota','a85_g4_f_quota','a85_g4_f_text','a85_g4_f_auto']
        for c in brn.call_args_list:
            job,arm=c.args[0].name[:6],c.args[0].name[7:]; f=c.kwargs['adapter_factory']
            assert c.args[2]==br.Knowledge() and c.kwargs['tool_contracts']==eq.arm_contracts(arm) and c.kwargs['max_tool_corrections']==2 and c.kwargs['roster']==eq.G4 and c.kwargs['seeds']==eq.SEEDS
            assert f.func is eq.QuotaAdapter and f.keywords=={'pool':pools[job],'arm':arm}
        assert [(c.args[0].name,c.kwargs['tag']) for c in fcb.call_args_list]==[('a65_g4_random','random'),('a65_g4_menu','menu'),('a85_g4_menu','menu'),('a85_g4_random','random')]
        for c in fcb.call_args_list:
            job=c.args[0].name[:6]; pi=pools[job]['probe']['pi']
            if c.kwargs['tag']=='random': assert [m for m,_ in c.kwargs['plans']]==pi[:2] and c.kwargs['tie_order']==('FixedMixup','None',pi[0],pi[1]) and set(c.kwargs['expected_assignments'])==set(pi[:2])
            else: assert c.kwargs['plans']==list(eq.MENU) and c.kwargs['tie_order']==eq.MENU_TIE_ORDER
        fin=context.read_json(root/'execution_finished.json'); assert len(fin['branches'])==10 and not fin['failures']
    print('EXPLORATION_QUOTA_SMOKE_OK: config, frozen pool+probe streams, identical Q supply, quota execution inside run_job, identity/restore, legal elsewhere, preparation, dispatch')

def smoke_paired_refinement(cfg):
    """Zero-fit, zero-LLM checks of the paired_refinement additions (task §9): single-leaf derivation and its rejections, actual
    array change confined to the edited branch, the same helper in both Fast arms with only F_refine's second slot constrained,
    0/1-slot commits, cache/rename never bypassing the slot rule, first feedback returning before the second build, resume with
    the three-slot Limits / round trip / parent-child state, Random-global/local determinism and non-leakage, G5 threading, dispatch.
    Reads only an old exposed T (a85, default roster) and synthetic overviews; never this package's Targets."""
    from unittest.mock import patch
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    import csv
    import numpy as np
    assert cfg['planned_fits']==96 and cfg['caps']==dict(max_fit_attempts=98,max_llm_requests=64,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2) and cfg['http_cap']==128
    assert {j:tuple(v) for j,v in cfg['order'].items()}=={'a65_g5':('f_refine','f_free','random_global','random_local','menu'),'a85_g5':('menu','random_local','random_global','f_free','f_refine')} and cfg['seeds']==[20261004,20261005,20261006]
    exp={'a65_g5':(17088,(16416,17088),(17088,17136),(17184,17232),(17280,17328,17376,17424)),'a85_g5':(22344,(21672,22344),(22344,22392),(22440,22488),(22536,22584,22632,22680))}
    for j,(t,tr,ca,cb,e) in exp.items():
        job=cfg['jobs'][j]
        assert (job['t'],tuple(job['train_range']),tuple(job['c_a']),tuple(job['c_b']),tuple(job['e']))==(t,tr,ca,cb,e),job
        assert tuple(job['roster_override'])==pr.G5 and len(set(job['roster_override']))==32
    assert max(exp['a65_g5'][4])+spec.H<=exp['a85_g5'][1][0]
    with open(spec.DATASETS['electricity']['path'],encoding='utf-8') as f: hdr=next(csv.reader(f))
    cols=sorted(c for c in hdr if c!='date')
    assert len(cols)==321 and tuple(cols[160:192])==pr.G5 and all(set(pr.G5).isdisjoint(g) for g in (sr.G0,sr.G1,dc.G2,dc.G3,eq.G4))
    assert cfg['max_tool_corrections']==2 and cfg['evidence_roundtrip'] is True and cfg['environment']['torch'] and cfg['knowledge'].startswith('H0 empty') and str(DEFAULT).endswith('dev_batch_research_paired_refinement')
    assert cfg['per_fast_job_limits']=={'max_calls':16,'max_tools':32,'max_new_evaluations':3,'wall_seconds':'remaining package wall'} and cfg['fit_plan']==pr.FIT_PLAN and cfg['random_supply']['visible_to_fast'] is False
    assert cfg['random_supply']['seed_base']==2026091700 and '2026091800' in cfg['random_local']['slot2'] and cfg['second_slot_contract']['applies_to']=='f_refine only'
    # contracts: same derived form / identity rule / process note in both arms; only the slot contract differs; no effect hints.
    cr,cf=pr.arm_contracts('f_refine'),pr.arm_contracts('f_free')
    for c in (cr,cf):
        assert set(c)==br.TOOLS and c['build_material']['derived_form']==pr.DERIVED_FORM and c['build_material']['identity_rule']==pr.IDENTITY_RULE and c['build_material']['process_note']==pr.PROCESS_NOTE
        assert 'At most 3 new complete plans' in c['evaluate']['meaning'] and c['build_material']['semantics']==sp.CONTRACTS['build_material']['semantics']
    assert cr['evaluate']['slot_contract'].startswith('CONTRACT') and 'no requirement' in cf['evaluate']['slot_contract'] and {k:v for k,v in cr.items() if k!='evaluate'}=={k:v for k,v in cf.items() if k!='evaluate'}
    assert not re.search(r'bold|stronger is|prefer frequency|protect|histor|winner|D1',json.dumps([pr.DERIVED_FORM,pr.IDENTITY_RULE,cr['evaluate']['slot_contract'],cf['evaluate']['slot_contract']]),re.I)
    assert 'Stronger, weaker, uniform and conditional constructions are all hypotheses, not defaults' in pr.PROCESS_NOTE
    # (1) derivation on a synthetic table: exactly one leaf changes, the parent is untouched, every illegal edit is rejected.
    rng=np.random.RandomState(7)
    table=[{f:(0 if f=='missing_count' else spec.N_PARENTS if f=='n_parent_pairs' else float(rng.rand())) for f in spec.OBS_FIELDS} for _ in range(32)]
    parent=policy.validate_policy({'default':{'steps':[{'op':'timemixup','donor_rule':'R','w':0.25}]},
                                   'rules':[{'when':{'feature':'std','op':'>=','value':{'quantile':0.5}},'steps':[{'op':'freqmask','mu':0.10},{'op':'freqmix','donor_rule':'U','mu':0.05}]},
                                            {'when':{'const':False},'steps':[{'op':'freqmix','donor_rule':'U','mu':0.05}]}],'rationale':'x'})
    frozen=json.dumps(parent,sort_keys=True)
    child,rec=pr.derive_policy(parent,{'location':'default','rule_index':None,'step_index':0,'parameter':'w','new_value':0.5})
    assert child['default']['steps'][0]['w']==0.5 and child['rules']==parent['rules'] and rec=={'location':'default','rule_index':None,'step_index':0,'op':'timemixup','parameter':'w','old_value':0.25,'new_value':0.5}
    child,rec=pr.derive_policy(parent,{'location':'rule','rule_index':0,'step_index':1,'parameter':'mu','new_value':0.2})
    assert child['rules'][0]['steps'][1]['mu']==0.2 and child['rules'][0]['steps'][0]==parent['rules'][0]['steps'][0] and child['default']==parent['default'] and child['rules'][1]==parent['rules'][1]
    assert json.dumps(parent,sort_keys=True)==frozen
    for bad in ({'location':'default','rule_index':0,'step_index':0,'parameter':'w','new_value':0.5},{'location':'rule','rule_index':2,'step_index':0,'parameter':'mu','new_value':0.2},
                {'location':'rule','rule_index':0,'step_index':2,'parameter':'mu','new_value':0.2},{'location':'default','rule_index':None,'step_index':0,'parameter':'mu','new_value':0.2},
                {'location':'default','rule_index':None,'step_index':0,'parameter':'donor_rule','new_value':'U'},{'location':'default','rule_index':None,'step_index':0,'parameter':'w','new_value':0.25},
                {'location':'default','rule_index':None,'step_index':0,'parameter':'w','new_value':0.3},{'location':'default','rule_index':None,'step_index':0,'parameter':'w','new_value':True},
                {'location':'elsewhere','rule_index':None,'step_index':0,'parameter':'w','new_value':0.5},{'location':'default','step_index':0,'parameter':'w','new_value':0.5}):
        try: pr.derive_policy(parent,bad); raise AssertionError('accepted %r'%bad)
        except pr.EditError: pass
    assert json.dumps(parent,sort_keys=True)==frozen
    pk=policy.compile_policy(parent,table)['key']
    nb=pr.enumerate_neighbours(parent,table,{policy.compile_policy(policy.identity_policy(),table)['key'],policy.compile_policy(policy.fixed_mixup_policy(),table)['key']},pk)
    assert len(nb)==8 and [n['status'] for n in nb if n['location']=='rule' and n['rule_index']==1]==['inactive_branch','inactive_branch'] and sum(n['status']=='kept' for n in nb)==6
    assert [(n['location'],n['rule_index'],n['step_index'],n['new_value']) for n in nb[:2]]==[('default',None,0,0.1),('default',None,0,0.5)] and nb==pr.enumerate_neighbours(parent,table,{policy.compile_policy(policy.identity_policy(),table)['key'],policy.compile_policy(policy.fixed_mixup_policy(),table)['key']},pk)
    assert pr.enumerate_neighbours(policy.validate_policy(policy.identity_policy()),table,set(),'k')==[] and len(pr.enumerate_neighbours(policy.validate_policy(policy.fixed_mixup_policy()),table,set(),'k'))==2
    # frozen supply: deterministic, R-ids, this study's seed base; the G3/G4 streams untouched; SUPPLY_INCOMPLETE path.
    sup=pr.generate_supply('a65_g5',table); sup2=pr.generate_supply('a65_g5',table)
    assert json.dumps(sup,sort_keys=True)==json.dumps(sup2,sort_keys=True) and [d['draft_id'] for d in sup['drafts']]==['R1','R2','R3','R4'] and all(d['proposal_seed']==2026091700+d['j'] for d in sup['drafts'])
    assert sup['status']=='COMPLETE' and sup['visible_to_fast'] is False and pr.generate_supply('a85_g5',table)['drafts'][0]['proposal_seed']>=2026092700 and '2026091400' in dc.generate_pool('a65_g3',table)['rule']
    with patch.object(policy,'random_policy',side_effect=lambda seed: policy.identity_policy()):
        assert pr.generate_supply('a65_g5',table)['status']=='SUPPLY_INCOMPLETE'
    # (2)-(5) on a real old T (a85, default roster); fits are faked, materials are real.
    fits=[]
    def fake_fit(ctx,ref,seeds,ledger,repo,feedback=True):
        fits.append(ref.material_id); return [{'model_path':'m_%s_%d'%(ref.material_id,s),'scores':{'c_a':{'per_origin_entity_normalized_mse':[[0.1+0.01*len(fits)]*32]*2,'n_nonfinite_predictions':0}}} for s in seeds]
    class Scripted:
        def __init__(self,outs): self.outs=list(outs); self.seen=[]
        def __call__(self,req):
            self.seen.append(req); out=self.outs.pop(0)
            if isinstance(out,Exception): raise out
            return out
    U=lambda steps,rat='hypothesis: uniform test': policy.uniform_policy(steps,rat)
    FM=lambda mu: U([{'op':'freqmask','mu':mu}],'hypothesis: uniform freqmask %s'%mu)
    COND={'default':{'steps':[]},'rules':[{'when':{'feature':'std','op':'>=','value':{'quantile':0.5}},'steps':[{'op':'freqmask','mu':0.10}]}],'rationale':'hypothesis: mask high-std half','observation_fields_used':['std']}
    DER=lambda pid,parent,loc,ri,si,par,v: {'plan_id':pid,'parent_plan_id':parent,'parameter_edit':{'location':loc,'rule_index':ri,'step_index':si,'parameter':par,'new_value':v},'rationale':'hypothesis: %s -> %s'%(par,v)}
    rj=lambda a,client: br.run_job(job_id='a85',knowledge=br.Knowledge(),adapter=a,client=client,seeds=rt.SEEDS,baselines=a.baselines(),allowed_features=rt.BATCH_FIELDS,tool_contracts=pr.arm_contracts(a.arm),
                                   max_tool_corrections=2,evidence_roundtrip=True,limits=br.Limits(**{**pr.LIMITS,'wall_seconds':600.}))
    with TemporaryDirectory() as td:
        led=rt.RuntimeLedger(Path(td)/'budget.json',**CAPS)
        try: pr.RefineAdapter(Path(td)/'bad','a85',led,REPO,arm='random_local'); raise AssertionError('control arm accepted')
        except ValueError: pass
        with patch.object(rt,'fit',side_effect=fake_fit):
            # (2) actual array change: derived from FixedMixup (uniform) touches every entity; derived from a conditional parent touches only its rule's entities.
            a=pr.RefineAdapter(Path(td)/'arr','a85',led,REPO,arm='f_free',tool_error_feedback=True); a.baselines()
            c=a.build_material(DER('W5','FixedMixup','default',None,0,'w',0.5),remaining_seconds=60.)
            d=c.material_spec['derivation']; assert c.material_spec['parent_plan_id']=='FixedMixup' and c.material_spec['parameter_edit']['old_value']==0.25 and d['entities_changed_arrays']==list(range(32)) and d['unchanged_entities_bit_identical'] and d['branch_entities']==list(range(32))
            assert c.material_spec['policy']['default']['steps']==[{'op':'timemixup','donor_rule':'R','w':0.5}] and c.material_spec['policy']['rules']==[] and context.read_json(Path(td)/'arr'/'a85'/'materials'/'W5__compiled.json')['parent_plan_id']=='FixedMixup'
            assert not re.search(r'loss|score|better|improve|expect|predict',json.dumps({k:v for k,v in d.items() if k!='note'}),re.I) and 'not a prediction' in d['note']
            assert a.specs['FixedMixup']['policy']['default']['steps'][0]['w']==0.25   # parent untouched
            c1=a.build_material({'plan_id':'C1','policy':COND},remaining_seconds=60.); a.evaluate(c1,rt.SEEDS,feedback=True,remaining_seconds=60.)
            c2=a.build_material(DER('C2','C1','rule',0,0,'mu',0.2),remaining_seconds=60.); d=c2.material_spec['derivation']
            ents=[i for i,r in enumerate(a.specs['C1']['rule_index']) if r==0]
            assert 0<len(ents)<32 and d['entities_changed_arrays']==ents and d['branch_entities']==ents and d['unchanged_entities_bit_identical'] and set(d['max_abs_change_by_entity'])=={str(e) for e in ents}
            assert c2.material_spec['policy']['rules'][0]['steps'][0]['mu']==0.2 and c2.material_spec['policy']['default']==a.specs['C1']['policy']['default'] and c2.material_spec['assignment']!=a.specs['C1']['assignment']
            with np.load(a.refs['C1'].path) as p1, np.load(a.refs['C2'].path) as p2:
                assert all(np.array_equal(p1['Xc'][e],p2['Xc'][e]) for e in range(32) if e not in ents) and all(not np.array_equal(p1['Xc'][e],p2['Xc'][e]) for e in ents)
            # rejections at build: unknown / untrained / foreign parent, inactive branch, identity with a registered plan (before material/fit)
            n_mat=len(list((Path(td)/'arr'/'a85'/'materials').glob('*.npz')))
            for args,frag in ((DER('X','nope','default',None,0,'w',0.5),'plan of this run'),(DER('X','C2','rule',0,0,'mu',0.05),'not fully evaluated'),
                              (DER('X','None','default',None,0,'w',0.5),'this branch has 0 steps'),(DER('X','FixedMixup','default',None,0,'w',0.25),'equals the current value'),
                              ({'plan_id':'X','parent_plan_id':'FixedMixup','parameter_edit':{},'rationale':'r'},'exactly the keys'),({**DER('X','FixedMixup','default',None,0,'w',0.5),'rationale':' '},'nonempty'),
                              ({'plan_id':'X','policy':policy.fixed_mixup_policy()},"identical to the registered plan 'FixedMixup'"),({'plan_id':'X','policy':U([{'op':'timemixup','donor_rule':'R','w':0.5}],'same as W5')},"identical to the registered plan 'W5'"),
                              ({'plan_id':'X','parent_plan_id':'FixedMixup','rationale':'r'},'derived form')):
                try: a.build_material(args,remaining_seconds=60.); raise AssertionError('accepted %r'%args)
                except br.ToolInputError as exc: assert frag in str(exc),(frag,str(exc))
            assert 'X' not in a.refs and len(list((Path(td)/'arr'/'a85'/'materials').glob('*.npz')))==n_mat
            c3=a.build_material({'plan_id':'C3','policy':{**COND,'rules':[{**COND['rules'][0],'steps':[{'op':'freqmask','mu':0.05}]},{'when':{'const':False},'steps':[{'op':'freqmix','donor_rule':'U','mu':0.05}]}]}},remaining_seconds=60.); a.evaluate(c3,rt.SEEDS,feedback=True,remaining_seconds=60.)
            try: a.build_material(DER('X','C3','rule',1,0,'mu',0.2),remaining_seconds=60.); raise AssertionError('inactive branch accepted')
            except br.ToolInputError as exc: assert 'applies to no current entity' in str(exc) and exc.details['entities_by_branch']['1']==0
            # (3a) F_refine: first feedback returns before the derived second build; deferral is real; commit after the second evaluation is deferred once more.
            fits.clear(); a=pr.RefineAdapter(Path(td)/'ra','a85',led,REPO,arm='f_refine',tool_error_feedback=True)
            client=Scripted([{'actions':[{'tool':'build_material','arguments':{'plan_id':'P1','policy':FM(0.1)}},{'tool':'evaluate','arguments':{'plan_id':'P1'}},{'tool':'build_material','arguments':DER('P2','P1','default',None,0,'mu',0.2)},{'tool':'evaluate','arguments':{'plan_id':'P2'}}]},
                             {'actions':[{'tool':'build_material','arguments':DER('P2','P1','default',None,0,'mu',0.2)},{'tool':'evaluate','arguments':{'plan_id':'P2'}},{'tool':'compare','arguments':{'a':'P1','b':'P2'}},{'tool':'commit','arguments':{'plan_id':'P2','reason':'x'}}]},
                             {'actions':[{'tool':'commit','arguments':{'plan_id':'P2','reason':'child better in C_A'}}]}])
            res=rj(a,client); dfr=[x for x in res.trace if x['event']=='action_batch_deferred']
            assert res.status=='COMPLETE' and res.committed_plan_id=='P2' and res.new_evaluations==2 and res.calls==3 and fits==['None','FixedMixup','P1','P2'] and not [x for x in res.trace if x['event']=='tool_rejected']
            assert len(dfr)==2 and [t['tool'] for t in dfr[0]['actions']]==['build_material','evaluate'] and [t['tool'] for t in dfr[1]['actions']]==['commit']
            p1=next(c for c in client.seen[1]['candidates'] if c['plan_id']=='P1'); assert p1['trained'] and p1['feedback']['mean_loss']>0 and 'P2' not in [c['plan_id'] for c in client.seen[1]['candidates']]
            p2=next(c for c in client.seen[2]['candidates'] if c['plan_id']=='P2'); assert p2['trained'] and p2['material_spec']['parent_plan_id']=='P1' and p2['material_spec']['parameter_edit']['new_value']==0.2
            assert client.seen[0]['remaining']['new_evaluations']==3 and client.seen[0]['remaining']['tools']==32 and client.seen[0]['tool_contracts']['evaluate']['slot_contract'].startswith('CONTRACT') and 'evidence_roundtrip' in client.seen[0]['contract']
            assert 'unscored_drafts' not in client.seen[0]['overview'] and 'R1' not in json.dumps(client.seen)   # supply never reaches Fast
            # (3b) F_refine: derived before slot 1 -> rejected; free second -> rejected at evaluate before slot/fit, kept for the third slot; third free.
            fits.clear(); a=pr.RefineAdapter(Path(td)/'rb','a85',led,REPO,arm='f_refine',tool_error_feedback=True)
            client=Scripted([{'actions':[{'tool':'build_material','arguments':DER('D0','FixedMixup','default',None,0,'w',0.5)},{'tool':'evaluate','arguments':{'plan_id':'D0'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'P1','policy':FM(0.1)}},{'tool':'evaluate','arguments':{'plan_id':'P1'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'P2','policy':FM(0.2)}},{'tool':'evaluate','arguments':{'plan_id':'P2'}},{'tool':'commit','arguments':{'plan_id':'P2','reason':'x'}}]},
                             {'actions':[{'tool':'build_material','arguments':DER('P3','P1','default',None,0,'mu',0.05)},{'tool':'evaluate','arguments':{'plan_id':'P3'}}]},
                             {'actions':[{'tool':'evaluate','arguments':{'plan_id':'P2'}}]},
                             {'actions':[{'tool':'commit','arguments':{'plan_id':'P2','reason':'third slot'}}]}])
            res=rj(a,client); rej=[x for x in res.trace if x['event']=='tool_rejected']
            assert res.status=='COMPLETE' and res.committed_plan_id=='P2' and res.new_evaluations==3 and fits==['None','FixedMixup','P1','P3','P2'] and len(rej)==2
            assert rej[0]['tool']=='build_material' and 'derived form is not available' in rej[0]['error']['message'] and [u['tool'] for u in rej[0]['unexecuted_actions']]==['evaluate'] and 'D0' not in a.refs
            assert rej[1]['tool']=='evaluate' and rej[1]['error']['message'].startswith('second-slot') and rej[1]['error']['legal_parents']==['None','FixedMixup','P1'] and [u['tool'] for u in rej[1]['unexecuted_actions']]==['commit'] and 'P2' in a.refs
            assert client.seen[3]['remaining']['tool_corrections']==0 and client.seen[3]['remaining']['new_evaluations']==2
            # (3c) rename / identical derived child never bypasses: rejected before material; evaluate on the existing id is a cached read (no slot, no fit).
            fits.clear(); a=pr.RefineAdapter(Path(td)/'rc','a85',led,REPO,arm='f_refine',tool_error_feedback=True)
            client=Scripted([{'actions':[{'tool':'build_material','arguments':{'plan_id':'P1','policy':U([{'op':'timemixup','donor_rule':'R','w':0.5}],'w .5')}},{'tool':'evaluate','arguments':{'plan_id':'P1'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'P1b','policy':U([{'op':'timemixup','donor_rule':'R','w':0.5}],'renamed')}},{'tool':'evaluate','arguments':{'plan_id':'P1b'}}]},
                             {'actions':[{'tool':'build_material','arguments':DER('P2','P1','default',None,0,'w',0.25)},{'tool':'evaluate','arguments':{'plan_id':'P2'}}]},
                             {'actions':[{'tool':'evaluate','arguments':{'plan_id':'P1'}},{'tool':'commit','arguments':{'plan_id':'None','reason':'x'}}]},
                             {'actions':[{'tool':'commit','arguments':{'plan_id':'None','reason':'baseline'}}]}])
            res=rj(a,client); rej=[x for x in res.trace if x['event']=='tool_rejected']
            assert res.status=='COMPLETE' and res.committed_plan_id=='None' and res.new_evaluations==1 and fits==['None','FixedMixup','P1'] and len(rej)==2 and 'P1b' not in a.refs and 'P2' not in a.refs
            assert rej[0]['error']['existing_plan_id']=='P1' and rej[1]['error']['existing_plan_id']=='FixedMixup' and sum(x['event']=='tool_completed' and x['tool']=='evaluate' for x in res.trace)==2
            # (3d) 0-slot and 1-slot commits are legal in F_refine; F_free may derive first (parent FixedMixup) and never meets the slot rule.
            for outs,cid,nfit in (([{'actions':[{'tool':'commit','arguments':{'plan_id':'None','reason':'stop'}}]}],'None',2),
                                  ([{'actions':[{'tool':'build_material','arguments':{'plan_id':'P1','policy':FM(0.2)}},{'tool':'evaluate','arguments':{'plan_id':'P1'}}]},{'actions':[{'tool':'commit','arguments':{'plan_id':'P1','reason':'one slot'}}]}],'P1',3)):
                fits.clear(); a=pr.RefineAdapter(Path(td)/('rd%d'%nfit),'a85',led,REPO,arm='f_refine',tool_error_feedback=True)
                res=rj(a,Scripted(outs)); assert res.status=='COMPLETE' and res.committed_plan_id==cid and len(fits)==nfit and not [x for x in res.trace if x['event']=='tool_rejected']
            fits.clear(); a=pr.RefineAdapter(Path(td)/'ff','a85',led,REPO,arm='f_free',tool_error_feedback=True)
            client=Scripted([{'actions':[{'tool':'build_material','arguments':DER('P1','FixedMixup','default',None,0,'w',0.1)},{'tool':'evaluate','arguments':{'plan_id':'P1'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'P2','policy':FM(0.05)}},{'tool':'evaluate','arguments':{'plan_id':'P2'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'P3','policy':COND}},{'tool':'evaluate','arguments':{'plan_id':'P3'}}]},
                             {'actions':[{'tool':'commit','arguments':{'plan_id':'P3','reason':'x'}}]}])
            res=rj(a,client); assert res.status=='COMPLETE' and res.new_evaluations==3 and fits==['None','FixedMixup','P1','P2','P3'] and not [x for x in res.trace if x['event']=='tool_rejected'] and a.specs['P1']['parent_plan_id']=='FixedMixup'
            assert client.seen[0]['tool_contracts']['evaluate']['slot_contract']==cf['evaluate']['slot_contract'] and client.seen[0]['tool_contracts']['build_material']==cf['build_material']
            # (4) resume keeps the three-slot Limits, the round trip and the parent-child state; a rejected evaluate does not count as a consumed slot.
            fits.clear(); shared=Path(td)/'shared'; rt.Adapter(shared,'a85',led,REPO).baselines()
            client=Scripted([{'actions':[{'tool':'build_material','arguments':{'plan_id':'P1','policy':FM(0.1)}},{'tool':'evaluate','arguments':{'plan_id':'P1'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'P2','policy':FM(0.2)}},{'tool':'evaluate','arguments':{'plan_id':'P2'}}]},
                             {'actions':[{'tool':'build_material','arguments':DER('P3','P1','default',None,0,'mu',0.05)},{'tool':'evaluate','arguments':{'plan_id':'P3'}}]},
                             RuntimeError('transport')])
            fake_client=SimpleNamespace(fast=lambda unit: client)
            res=pr.run_arm('f_refine',Path(td)/'res','a85',led,fake_client,shared,None,seeds=rt.SEEDS,roster=None,max_tool_corrections=2)
            assert res.status=='INCOMPLETE' and res.failure_kind=='AGENT_CALL_FAILED' and res.new_evaluations==2 and res.calls==4 and sum(x['event']=='tool_rejected' for x in res.trace)==1
            with patch.object(base.br,'run_job',return_value=br.RunResult('INCOMPLETE','a85',0,None,None,4,6,2)) as run:
                pr.resume_arm('f_refine',Path(td)/'res','a85',led,fake_client,None,seeds=rt.SEEDS,roster=None,max_tool_corrections=2)
            kw=run.call_args.kwargs; lim=kw['limits']
            assert (lim.max_calls,lim.max_tools,lim.max_new_evaluations)==(16,32,3) and kw['evidence_roundtrip'] is True and kw['max_tool_corrections']==2 and kw['tool_contracts']==pr.arm_contracts('f_refine')
            assert kw['resume']['new_evaluations']==2 and kw['resume']['evaluated_new']==['P1','P3'] and kw['resume']['corrections_used']==1 and kw['resume']['calls']==3
            restored={c.plan_id:c for c in kw['baselines']}; assert set(restored)=={'None','FixedMixup','P1','P2','P3'} and restored['P3'].material_spec['parent_plan_id']=='P1' and len(restored['P3'].model_refs)==3 and restored['P2'].model_refs==()
            assert isinstance(kw['adapter'],pr.RefineAdapter) and kw['adapter'].arm=='f_refine' and kw['adapter'].n_new_complete()==2 and kw['adapter'].is_derived('P3') and context.read_json(Path(td)/'res'/'resume.json')['new_fits_for_restore']==0
            # (5) Random-local: deterministic start, neighbour census, one draw, dedup; Random-global through the shared control builder.
            fits.clear(); table85=rt.Adapter(Path(td)/'plain','a85',led,REPO).overview()['entities']; sup=pr.generate_supply('a65_g5',table85); sup['job']='a85'
            commits=[]
            with patch.object(pr.commit,'commit',side_effect=lambda root,ds,job,mid,reason,seed: commits.append((mid,reason))):
                res=pr.random_local_branch(Path(td)/'rl','a85',led,shared,sup,seeds=rt.SEEDS,roster=None)
            rec=context.read_json(Path(td)/'rl'/'local_search.json')
            assert res.status=='COMPLETE' and rec['status']=='COMPLETE' and rec['seed']==2026091800 and rec['parent'] in ('FixedMixup','R1') and rec['parents']['None']['n_kept']==0 and rec['n_kept']>=1 and 0<=rec['drawn_index']<rec['n_kept']
            assert fits[:3]==['None','FixedMixup','R1'] and fits[3]=='L2' and fits[4]==rec['third'] and rec['third'] in ('R2','R3','R4') and len(fits)==5 and commits[0][0]==res.committed_plan_id
            l2=context.read_json(Path(td)/'rl'/'a85'/'materials'/'L2__compiled.json'); assert l2['parent_plan_id']==rec['parent'] and l2['parameter_edit']==rec['edit'] and l2['derivation']['unchanged_entities_bit_identical']
            exp_idx=int(np.random.RandomState(2026091800).randint(rec['n_kept'])); assert rec['drawn_index']==exp_idx
            # the C_A-best parent wins (fake C_A grows with fit order, so FixedMixup < R1 here); L2 equals the drawn neighbour recompiled from the parent policy
            assert rec['parent']=='FixedMixup' and rec['parents']['FixedMixup']['c_a_mean']<rec['parents']['R1']['c_a_mean']
            kept=[n for n in pr.enumerate_neighbours(policy.validate_policy(policy.fixed_mixup_policy()),table85,set(),'k') if n['status']=='kept']
            assert l2['policy']['default']['steps'][0]['w']==kept[exp_idx]['new_value']
            fits.clear()
            with patch.object(pr.commit,'commit',side_effect=lambda *a,**k: None):
                res=pr.run_arm('random_global',Path(td)/'rg','a85',led,None,shared,sup,seeds=rt.SEEDS,roster=None,max_tool_corrections=2)
            assert res.status=='COMPLETE' and fits==['None','FixedMixup','R1','R2','R3'] and res.new_evaluations==3
        assert led.s['fit_attempts']==0
    # (6) preparation with patched adapters (supply written, marker written, 0 fits); dispatch order / kwargs / G5 / limits with patched branches.
    synthetic={'entities':table,'n_entities':32,'train_rows':[0,672],'batch_features':{}}
    with TemporaryDirectory() as td:
        root=Path(td)/'run'; root.mkdir(); log=[]
        def fake_adapter(run_dir,job,led,repo,**kw):
            log.append((job,kw.get('roster'))); return SimpleNamespace(ctx=SimpleNamespace(job=SimpleNamespace(roster=list(pr.G5),roster_override=pr.G5),stage='material'),overview=lambda: synthetic,baselines=lambda: ())
        led=rt.RuntimeLedger(root/'budget.json',**CAPS)
        with patch.object(rt,'Adapter',side_effect=fake_adapter):
            commons,pools=pr.prepare(root,led,CAPS)
        assert log==[('a65_g5',pr.G5),('a85_g5',pr.G5)] and all(pools[j]['status']=='COMPLETE' and [d['draft_id'] for d in pools[j]['drafts']]==['R1','R2','R3','R4'] for j in pr.JOBS)
        assert (root/pr.FROZEN_MARKER).exists() and context.read_json(root/pr.FROZEN_MARKER)['fit_attempts_at_freeze']==0 and pr.load_pools(root)=={j:context.read_json(root/(j+'_random_pool.json')) for j in pr.JOBS}
        assert context.read_json(root/'frozen_config.json')['arm_contracts']['f_refine']==pr.arm_contracts('f_refine') and pools['a65_g5']['drafts'][0]['proposal_seed']!=pools['a85_g5']['drafts'][0]['proposal_seed']
    with TemporaryDirectory() as td:
        root=Path(td)/'run'
        pools={j:{**pr.generate_supply(j,table),'job':j} for j in pr.JOBS}
        def fake_prepare(r,led,caps):
            for j,p in pools.items(): context.write_json(r/(j+'_random_pool.json'),p)
            context.write_json(r/pr.FROZEN_MARKER,{'epoch':0}); return {j:SimpleNamespace(baselines=lambda: ()) for j in pr.JOBS},pools
        done=br.RunResult('COMPLETE','a65_g5',0,'None','ref',1,1,0)
        with patch.object(rt,'MeteredClient') as mc, patch.object(pr,'prepare',side_effect=fake_prepare), patch.object(base,'branch',return_value=done) as brn, \
             patch.object(dc,'fixed_candidates_branch',return_value=done) as fcb, patch.object(pr,'random_local_branch',return_value=done) as rlb, patch.object(base,'stage') as stg, patch.object(sys.modules[__name__],'report'):
            mc.return_value.fatal=False
            worker(root)
        assert [c.args[0].name for c in brn.call_args_list]==['a65_g5_f_refine','a65_g5_f_free','a85_g5_f_free','a85_g5_f_refine']
        for c in brn.call_args_list:
            job,arm=c.args[0].name[:6],c.args[0].name[7:]; f=c.kwargs['adapter_factory']
            assert c.args[2]==br.Knowledge() and c.kwargs['tool_contracts']==pr.arm_contracts(arm) and c.kwargs['max_tool_corrections']==2 and c.kwargs['roster']==pr.G5 and c.kwargs['seeds']==pr.SEEDS
            assert c.kwargs['evidence_roundtrip'] is True and c.kwargs['limits']==pr.LIMITS and f.func is pr.RefineAdapter and f.keywords=={'arm':arm}
        assert [(c.args[0].name,c.kwargs['tag']) for c in fcb.call_args_list]==[('a65_g5_random_global','random'),('a65_g5_menu','menu'),('a85_g5_menu','menu'),('a85_g5_random_global','random')]
        for c in fcb.call_args_list:
            if c.kwargs['tag']=='random': assert [m for m,_ in c.kwargs['plans']]==['R1','R2','R3'] and c.kwargs['tie_order']==pr.GLOBAL_TIE_ORDER and set(c.kwargs['expected_assignments'])=={'R1','R2','R3'}
            else: assert c.kwargs['plans']==list(pr.MENU) and c.kwargs['tie_order']==pr.MENU_TIE_ORDER
        assert [c.args[0].name for c in rlb.call_args_list]==['a65_g5_random_local','a85_g5_random_local'] and all(c.args[4]['job']==c.args[0].name[:6] and c.kwargs['roster']==pr.G5 for c in rlb.call_args_list)
        assert all(c.kwargs.get('roster')==pr.G5 for c in stg.call_args_list) and study_roster('a65_g5')==pr.G5 and study_roster('a85_g5')==pr.G5   # label stages receive the explicit G5 (roster pass-through itself is covered by smoke_fault_paths)
        fin=context.read_json(root/'execution_finished.json'); assert len(fin['branches'])==10 and not fin['failures']
    print('PAIRED_REFINEMENT_SMOKE_OK: config/G5, derivation and rejections, array change confined to the edited branch, same helper in both arms, F_refine slot rule only, 0/1-slot commits, no rename bypass, feedback before second build, resume state, random-local determinism, no supply leak, dispatch')

def smoke_measured_start(cfg):
    """Zero-fit, zero-LLM checks of the measured_start additions (task §10): frozen supply and G7, real R1/R2 candidates with tool
    references and C_A visible to F_start only, initialization = 2 slots (F_start 2 more, F_free 4, cached evaluate never a slot),
    optional derived form with R1/R2 as parents and no second-slot contract, resume keeping the start point / slots / limits,
    R0/RB shadow frozen before labels without touching the commit, INIT_INCOMPLETE path, dispatch order with the two hooks.
    Reads only an old exposed T (a85, default roster) and synthetic overviews; never this package's Targets."""
    from unittest.mock import patch
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    import csv
    import numpy as np
    assert cfg['planned_fits']==96 and cfg['caps']==dict(max_fit_attempts=98,max_llm_requests=64,max_llm_tokens=3000000,max_wall_s=10800,max_retries=2) and cfg['http_cap']==128
    assert {j:tuple(v) for j,v in cfg['order'].items()}=={'a65_g7':('f_start','f_free','random_global','random_local','menu'),'a85_g7':('menu','random_local','random_global','f_free','f_start')} and cfg['seeds']==[20261010,20261011,20261012]
    exp={'a65_g7':(17088,(16416,17088),(17088,17136),(17184,17232),(17280,17328,17376,17424)),'a85_g7':(22344,(21672,22344),(22344,22392),(22440,22488),(22536,22584,22632,22680))}
    for j,(t,tr,ca,cb,e) in exp.items():
        job=cfg['jobs'][j]
        assert (job['t'],tuple(job['train_range']),tuple(job['c_a']),tuple(job['c_b']),tuple(job['e']))==(t,tr,ca,cb,e),job
        assert tuple(job['roster_override'])==ms.G7 and len(set(job['roster_override']))==32
    with open(spec.DATASETS['electricity']['path'],encoding='utf-8') as f: hdr=next(csv.reader(f))
    cols=sorted(c for c in hdr if c!='date')
    assert len(cols)==321 and tuple(cols[224:256])==ms.G7 and '298' in cols[192:224] and all(set(ms.G7).isdisjoint(g) for g in (sr.G0,sr.G1,dc.G2,dc.G3,eq.G4,pr.G5))
    assert cfg['max_tool_corrections']==2 and cfg['evidence_roundtrip'] is True and cfg['environment']['torch'] and cfg['knowledge'].startswith('H0 empty') and str(DEFAULT).endswith('dev_batch_research_measured_start')
    assert cfg['per_fast_job_limits']=={'f_start':{'max_calls':16,'max_tools':40,'max_new_evaluations':2,'wall_seconds':'remaining package wall'},'f_free':{'max_calls':16,'max_tools':40,'max_new_evaluations':4,'wall_seconds':'remaining package wall'}}
    assert cfg['fit_plan']==ms.FIT_PLAN and cfg['random_supply']['seed_base']==2026092300 and '2026092400' in cfg['random_local']['rule'] and cfg['initialization']['plans']==['R1','R2'] and cfg['shadow_selection']['frozen'].startswith('per arm')
    cs,cf=ms.arm_contracts('f_start'),ms.arm_contracts('f_free')
    for c in (cs,cf):
        assert set(c)==br.TOOLS and c['build_material']['derived_form']==pr.DERIVED_FORM and c['build_material']['identity_rule']==pr.IDENTITY_RULE and c['build_material']['process_note']==pr.PROCESS_NOTE
    assert 'At most 2 more' in cs['evaluate']['meaning'] and 'At most 4 new' in cf['evaluate']['meaning'] and 'R1 or R2' in cs['commit']['meaning'] and 'R1' not in json.dumps(cf)
    assert not re.search(r'bold|stronger is|prefer frequency|protect|historical best|winner|D1|must modify|should modify|prefer',json.dumps([cs['evaluate']['slot_contract'],cf['evaluate']['slot_contract'],ms.INIT_NOTE]),re.I)   # 'no history' in INIT_NOTE is a neutral fact
    # (1) frozen supply: deterministic, R-ids, this study's seed base; older streams untouched; SUPPLY_INCOMPLETE.
    rng=np.random.RandomState(5)
    table=[{f:(0 if f=='missing_count' else spec.N_PARENTS if f=='n_parent_pairs' else float(rng.rand())) for f in spec.OBS_FIELDS} for _ in range(32)]
    sup=ms.generate_supply('a65_g7',table); sup2=ms.generate_supply('a65_g7',table)
    assert json.dumps(sup,sort_keys=True)==json.dumps(sup2,sort_keys=True) and [d['draft_id'] for d in sup['drafts']]==['R1','R2','R3','R4'] and all(d['proposal_seed']==2026092300+d['j'] for d in sup['drafts'])
    assert sup['status']=='COMPLETE' and sup['initialization']==['R1','R2'] and ms.generate_supply('a85_g7',table)['drafts'][0]['proposal_seed']>=2026093300 and '2026091700' in pr.generate_supply('a65_g5',table)['rule']
    with patch.object(policy,'random_policy',side_effect=lambda seed: policy.identity_policy()):
        assert ms.generate_supply('a65_g7',table)['status']=='SUPPLY_INCOMPLETE'
    # (2)-(6) on a real old T (a85, default roster); fits are faked, materials are real.
    fits=[]
    def fake_fit(ctx,ref,seeds,ledger,repo,feedback=True):
        fits.append(ref.material_id); return [{'model_path':'m_%s_%d'%(ref.material_id,s),'scores':{'c_a':{'per_origin_entity_normalized_mse':[[0.1+0.01*len(fits)]*32,[0.1+0.01*len(fits)+0.005]*32],'per_origin_normalized_mse_mean':[0.1+0.01*len(fits),0.1+0.01*len(fits)+0.005],'n_nonfinite_predictions':0,'normalized_mse_macro':0.1+0.01*len(fits)+0.0025}}} for s in seeds]
    class Scripted:
        def __init__(self,outs): self.outs=list(outs); self.seen=[]
        def __call__(self,req):
            self.seen.append(req); out=self.outs.pop(0)
            if isinstance(out,Exception): raise out
            return out
    U=lambda steps,rat='hypothesis: uniform test': policy.uniform_policy(steps,rat)
    FM=lambda mu: U([{'op':'freqmask','mu':mu}],'hypothesis: uniform freqmask %s'%mu)
    DER=lambda pid,parent,loc,ri,si,par,v: {'plan_id':pid,'parent_plan_id':parent,'parameter_edit':{'location':loc,'rule_index':ri,'step_index':si,'parameter':par,'new_value':v},'rationale':'hypothesis: %s -> %s'%(par,v)}
    rj=lambda a,client,arm: br.run_job(job_id='a85',knowledge=br.Knowledge(),adapter=a,client=client,seeds=ms.SEEDS,baselines=a.baselines(),allowed_features=rt.BATCH_FIELDS,tool_contracts=ms.arm_contracts(arm),
                                       max_tool_corrections=2,evidence_roundtrip=True,limits=br.Limits(**{**ms.LIMITS[arm],'wall_seconds':600.}))
    with TemporaryDirectory() as td:
        led=rt.RuntimeLedger(Path(td)/'budget.json',**CAPS)
        root=Path(td)/'run'; root.mkdir()
        with patch.object(rt,'fit',side_effect=fake_fit):
            shared=root/'a85_common'; rt.Adapter(shared,'a85',led,REPO,seeds=ms.SEEDS).baselines()
            pool=ms.generate_supply('a65_g7',rt.Adapter(shared,'a85',led,REPO,seeds=ms.SEEDS).overview()['entities']); pool['job']='a85'
            # (2) initialization: R1/R2 really "trained" once in the init copy; record + Start-only frozen; idempotent; 0 ledger fits under the fake.
            rec=ms.initialize(root,'a85',led,shared,pool); fits_init=[f for f in fits if f.startswith('R')]
            assert rec['status']=='COMPLETE' and fits_init==['R1','R2'] and set(rec['plans'])=={'R1','R2'} and rec['role']==ms.INIT_ROLE and rec['start_only']['selected'] in ('FixedMixup','None','R1','R2')
            assert context.read_json(root/'a85_init'/'a85'/'materials'/'R1__compiled.json')['role']==ms.INIT_ROLE and rec['start_only']['selected']==min(('FixedMixup','None','R1','R2'),key=lambda m:(rec['start_only']['c_a_means'][m],('FixedMixup','None','R1','R2').index(m)))
            assert ms.initialize(root,'a85',led,shared,pool)==rec and [f for f in fits if f.startswith('R')]==['R1','R2'] and ms.initialize(root,'a85',led,shared,{**pool,'status':'SUPPLY_INCOMPLETE'})==rec
            assert '"e"' not in json.dumps(rec['plans']['R1']['feedback']) and set(rec['plans']['R1']['feedback'])<={'block','seeds','loss_by_seed','loss_by_seed_origin','loss_by_seed_entity','mean_loss'}
            # (3) F_start sees R1/R2 as real evaluated candidates (restored from cache), F_free does not.
            fits.clear(); bs=root/'a85_f_start'; bs.mkdir(); shutil.copytree(root/'a85_init'/'a85',bs/'a85')
            a=ms.MeasuredAdapter(bs,'a85',led,REPO,arm='f_start',tool_error_feedback=True,seeds=ms.SEEDS); base_c=a.baselines()
            assert [c.plan_id for c in base_c]==['None','FixedMixup','R1','R2'] and all(len(c.model_refs)==3 and len(c.ca_losses)==3 for c in base_c) and led.s['fit_attempts']==0
            ov=a.overview(); assert ov['initialization']['plans']==['R1','R2'] and ov['initialization']['search_slots_remaining_for_you']==2 and br.public_copy(ov)
            bf=root/'a85_f_free'; bf.mkdir(); shutil.copytree(shared/'a85',bf/'a85')
            b=ms.MeasuredAdapter(bf,'a85',led,REPO,arm='f_free',tool_error_feedback=True,seeds=ms.SEEDS); assert [c.plan_id for c in b.baselines()]==['None','FixedMixup'] and 'initialization' not in b.overview()
            try: ms.MeasuredAdapter(Path(td)/'bad','a85',led,REPO,arm='f_refine'); raise AssertionError('foreign arm accepted')
            except ValueError: pass
            # (under the fake fit R1/R2 reappear in `fits` at every restore because no cell files exist; the ledger stays at 0 and real runs hit the cache)
            # (4a) F_start run: cached evaluate of R1 takes no slot; derived from R1 legal at once (no second-slot contract); R2 committable; cap = 2.
            fits.clear(); shutil.copytree(root/'a85_init'/'a85',root/'fs_a'/'a85'); a=ms.MeasuredAdapter(root/'fs_a','a85',led,REPO,arm='f_start',tool_error_feedback=True,seeds=ms.SEEDS)
            r1pol=pool['drafts'][0]['policy']
            def leaf_edit(pid,parent,pol):
                st=pol['default']['steps']
                if not st: return DER(pid,'FixedMixup','default',None,0,'w',0.5)
                par=pr.PARAM_OF[st[0]['op']]; v=[x for x in spec.ACTIONS[st[0]['op']][par] if abs(x-float(st[0][par]))>1e-9][0]
                return DER(pid,parent,'default',None,0,par,v)
            p2=leaf_edit('P2','R1',r1pol)
            client=Scripted([{'actions':[{'tool':'evaluate','arguments':{'plan_id':'R1'}},{'tool':'compare','arguments':{'a':'R1','b':'FixedMixup'}},{'tool':'build_material','arguments':{'plan_id':'P1','policy':FM(0.05)}},{'tool':'evaluate','arguments':{'plan_id':'P1'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'P1','policy':FM(0.05)}},{'tool':'evaluate','arguments':{'plan_id':'P1'}}]},
                             {'actions':[{'tool':'build_material','arguments':p2},{'tool':'evaluate','arguments':{'plan_id':'P2'}}]},
                             {'actions':[{'tool':'commit','arguments':{'plan_id':'R2','reason':'initialization plan delivered'}}]}])
            res=rj(a,client,'f_start'); dfr=[x for x in res.trace if x['event']=='action_batch_deferred']
            assert res.status=='COMPLETE' and res.committed_plan_id=='R2' and res.new_evaluations==2 and [f for f in fits if f not in ('None','FixedMixup','R1','R2')]==['P1','P2'] and not [x for x in res.trace if x['event']=='tool_rejected']
            assert len(dfr)==1 and [t['tool'] for t in dfr[0]['actions']]==['build_material','evaluate'] and sum(x['event']=='tool_completed' and x['tool']=='evaluate' for x in res.trace)==3
            q=client.seen[0]; assert q['remaining']['new_evaluations']==2 and q['remaining']['tools']==40 and q['overview']['initialization']['role']==ms.INIT_ROLE and {c['plan_id'] for c in q['candidates']}=={'None','FixedMixup','R1','R2'}
            assert all(c['trained'] and c['feedback']['mean_loss']>0 for c in q['candidates']) and all(c['material_spec'].get('role')==ms.INIT_ROLE for c in q['candidates'] if c['plan_id'] in ('R1','R2')) and 'evidence_roundtrip' in q['contract']
            assert client.seen[3]['remaining']['new_evaluations']==0 and a.specs['P2']['parent_plan_id']==p2['parent_plan_id'] and next(c for c in client.seen[3]['candidates'] if c['plan_id']=='P2')['material_spec']['parent_plan_id']==p2['parent_plan_id']
            # (4b) a third new evaluation in F_start is refused by the controller budget; identity with R1 is rejected before material.
            fits.clear(); shutil.copytree(root/'a85_init'/'a85',root/'fs_b'/'a85'); a=ms.MeasuredAdapter(root/'fs_b','a85',led,REPO,arm='f_start',tool_error_feedback=True,seeds=ms.SEEDS)
            client=Scripted([{'actions':[{'tool':'build_material','arguments':{'plan_id':'X1','policy':{**r1pol,'rationale':'same as R1'}}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'A','policy':FM(0.05)}},{'tool':'evaluate','arguments':{'plan_id':'A'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'B','policy':FM(0.2)}},{'tool':'evaluate','arguments':{'plan_id':'B'}}]},
                             {'actions':[{'tool':'build_material','arguments':{'plan_id':'C','policy':U([{'op':'timemixup','donor_rule':'U','w':0.1}])}},{'tool':'evaluate','arguments':{'plan_id':'C'}}]}])
            res=rj(a,client,'f_start'); rej=[x for x in res.trace if x['event']=='tool_rejected']
            assert res.status=='INCOMPLETE' and res.failure_kind=='BUDGET_EXHAUSTED' and 'new candidate budget' in res.reason and res.new_evaluations==2 and [f for f in fits if f not in ('None','FixedMixup','R1','R2')]==['A','B']
            assert len(rej)==1 and rej[0]['error']['existing_plan_id']=='R1' and 'X1' not in a.refs and 'C' in a.refs
            # (4c) F_free: four new evaluations, then the fifth is refused; never sees R1/R2.
            fits.clear(); shutil.copytree(shared/'a85',root/'ff_a'/'a85'); b=ms.MeasuredAdapter(root/'ff_a','a85',led,REPO,arm='f_free',tool_error_feedback=True,seeds=ms.SEEDS)
            outs=[{'actions':[{'tool':'build_material','arguments':{'plan_id':'Q%d'%i,'policy':FM(mu) if mu is not None else U([{'op':'timemixup','donor_rule':'U','w':w}])}},{'tool':'evaluate','arguments':{'plan_id':'Q%d'%i}}]} for i,(mu,w) in enumerate([(0.05,None),(0.1,None),(0.2,None),(None,0.1)])]
            outs.append({'actions':[{'tool':'build_material','arguments':{'plan_id':'Q4','policy':U([{'op':'timemixup','donor_rule':'U','w':0.5}])}},{'tool':'evaluate','arguments':{'plan_id':'Q4'}}]})
            res=rj(b,Scripted(outs),'f_free')
            assert res.status=='INCOMPLETE' and res.failure_kind=='BUDGET_EXHAUSTED' and res.new_evaluations==4 and [f for f in fits if f not in ('None','FixedMixup')]==['Q0','Q1','Q2','Q3']
            fits.clear(); shutil.copytree(shared/'a85',root/'ff_b'/'a85'); b=ms.MeasuredAdapter(root/'ff_b','a85',led,REPO,arm='f_free',tool_error_feedback=True,seeds=ms.SEEDS)
            client=Scripted([{'actions':[{'tool':'build_material','arguments':{'plan_id':'S','policy':{**r1pol,'rationale':'own construction'}}},{'tool':'evaluate','arguments':{'plan_id':'S'}}]},{'actions':[{'tool':'commit','arguments':{'plan_id':'S','reason':'x'}}]}])
            res=rj(b,client,'f_free'); assert res.status=='COMPLETE' and res.committed_plan_id=='S' and res.new_evaluations==1 and [f for f in fits if f not in ('None','FixedMixup')]==['S'] and led.s['fit_attempts']==0
            assert 'R1' not in json.dumps(client.seen) and 'initialization' not in client.seen[0]['overview'] and client.seen[0]['remaining']['new_evaluations']==4 and {c['plan_id'] for c in client.seen[0]['candidates']}=={'None','FixedMixup'}
            # (5) resume keeps the start point, the two-slot limit, the round trip and the consumed slot; R1/R2 restored from cache, never refitted.
            fits.clear()
            client=Scripted([{'actions':[{'tool':'build_material','arguments':DER('K1','FixedMixup','default',None,0,'w',0.5)},{'tool':'evaluate','arguments':{'plan_id':'K1'}}]},
                             RuntimeError('transport')])
            fake_client=SimpleNamespace(fast=lambda unit: client)
            res=ms.run_arm('f_start',root/'a85_res','a85',led,fake_client,shared,pool,seeds=ms.SEEDS,roster=None,max_tool_corrections=2)
            assert res.status=='INCOMPLETE' and res.failure_kind=='AGENT_CALL_FAILED' and res.new_evaluations==1 and res.calls==2 and [f for f in fits if f not in ('None','FixedMixup','R1','R2')]==['K1']
            with patch.object(base.br,'run_job',return_value=br.RunResult('INCOMPLETE','a85',0,None,None,2,2,1)) as run:
                ms.resume_arm('f_start',root/'a85_res','a85',led,fake_client,pool,seeds=ms.SEEDS,roster=None,max_tool_corrections=2)
            kw=run.call_args.kwargs; lim=kw['limits']; restored={c.plan_id:c for c in kw['baselines']}
            assert (lim.max_calls,lim.max_tools,lim.max_new_evaluations)==(16,40,2) and kw['evidence_roundtrip'] is True and kw['resume']['new_evaluations']==1 and kw['resume']['evaluated_new']==['K1']
            assert set(restored)=={'None','FixedMixup','R1','R2','K1'} and all(len(restored[m].model_refs)==3 for m in ('R1','R2','K1')) and restored['K1'].material_spec['parent_plan_id']=='FixedMixup'
            assert isinstance(kw['adapter'],ms.MeasuredAdapter) and kw['adapter'].arm=='f_start' and led.s['fit_attempts']==0 and context.read_json(root/'a85_res'/'resume.json')['new_fits_for_restore']==0
            # (6) Random-global / Random-local from the same initialization; INIT_INCOMPLETE path.
            fits.clear(); commits=[]
            with patch.object(ms.commit,'commit',side_effect=lambda root_,ds,job,mid,reason,seed: commits.append(mid)):
                res=ms.random_global_branch(root/'a85_rg','a85',led,root/'a85_init',pool)
                assert res.status=='COMPLETE' and [f for f in fits if f not in ('None','FixedMixup','R1','R2')]==['R3','R4'] and res.new_evaluations==2 and commits[-1]==res.committed_plan_id
                rows=[json.loads(l) for l in (root/'a85_rg'/'trace.jsonl').read_text(encoding='utf-8').splitlines()]; assert rows[0]['event']=='initialization_restored' and rows[0]['plans']==['R1','R2']
                fits.clear(); res=ms.random_local_branch(root/'a85_rl','a85',led,root/'a85_init',pool)
                ls=context.read_json(root/'a85_rl'/'local_search.json'); assert res.status=='COMPLETE' and ls['status']=='COMPLETE' and [s['seed'] for s in ls['steps']]==[2026092400,2026092401] and [f for f in fits if f not in ('None','FixedMixup','R1','R2')]==['L1','L2']
                for k,s in enumerate(ls['steps']):
                    assert s['parent'] in s['eligible'] and s['drawn_index']==int(np.random.RandomState(s['seed']).randint(s['n_kept'])) and context.read_json(root/'a85_rl'/'a85'/'materials'/('L%d__compiled.json'%(k+1)))['parent_plan_id']==s['parent']
                assert ls['steps'][1]['candidates'].keys()>={'FixedMixup','None','R1','R2','L1'} and res.new_evaluations==2
            assert led.s['fit_attempts']==0
            bad=root/'bad_init'; bad.mkdir(); (bad/'a85_init').mkdir(); context.write_json(bad/'a85_init'/'initialization.json',{'status':'INIT_FAILED'}); (bad/'a85_common').mkdir()
            res=ms.run_arm('random_global',bad/'a85_random_global','a85',led,None,bad/'a85_common',pool,seeds=ms.SEEDS,roster=None,max_tool_corrections=2)
            assert res.status=='INCOMPLETE' and res.failure_kind=='INIT_INCOMPLETE' and context.read_json(bad/'a85_random_global'/'branch_result.json')['failure_kind']=='INIT_INCOMPLETE'
        # (7) shadow selection on synthetic cells: kinds, tie order, commit untouched, idempotent freeze.
        def cellset(root_,job,arm,plans,commit_id,eval_order):
            d=root_/(job+'_'+arm); (d/job/'cells').mkdir(parents=True)
            for m,(o1,o2) in plans.items():
                for i,s in enumerate(ms.SEEDS):
                    context.write_json(d/job/'cells'/('%s__%s__s%d.json'%(job,m,s)),{'status':'OK','material_id':m,'model_seed':s,'scores':{'c_a':{'normalized_mse_macro':(o1[i]+o2[i])/2,'per_origin_normalized_mse_mean':[o1[i],o2[i]]}}})
            (d/'trace.jsonl').write_text('\n'.join(json.dumps({'event':'random_evaluated','plan_id':m}) for m in eval_order)+'\n',encoding='utf-8')
            context.write_json(d/job/'commit.json',{'material_id':commit_id}); return d
        sroot=Path(td)/'shadow'; sroot.mkdir()
        d1=cellset(sroot,'a85','random_global',{'None':([.30,.30,.30],[.30,.30,.30]),'FixedMixup':([.20,.21,.22],[.26,.27,.28]),'R1':([.25,.26,.27],[.22,.23,.24])},'FixedMixup',['R1'])
        sh=ms.shadow_selection(d1,'a85','random_global'); assert sh['R0']=='FixedMixup' and sh['RB']=='R1' and sh['kind']=='CONSISTENT_CONFLICT' and sh['actual_commit']=='FixedMixup' and sh['evaluated_set'][:3]==['FixedMixup','None','R1']
        d2=cellset(sroot,'a85','menu',{'None':([.3]*3,[.3]*3),'FixedMixup':([.2]*3,[.2]*3),'MenuU':([.2]*3,[.2]*3),'MenuFM':([.35,.15,.35],[.15]*3)},'MenuFM',['MenuU','MenuFM'])
        sh=ms.shadow_selection(d2,'a85','menu'); assert sh['R0']=='FixedMixup' and sh['RB']=='MenuFM' and sh['kind']=='MIXED' and sh['actual_commit']=='MenuFM' and sh['evaluated_set']==['FixedMixup','None','MenuU','MenuFM']
        d3=cellset(sroot,'a85','f_free',{'None':([.3]*3,[.3]*3),'FixedMixup':([.2]*3,[.2]*3),'Z':([.2]*3,[.2]*3)},'Z',['Z'])
        sh=ms.shadow_selection(d3,'a85','f_free'); assert sh['R0']=='FixedMixup' and sh['RB']=='FixedMixup' and sh['kind']=='SAME_SELECTION' and sh['actual_commit']=='Z' and not sh['actual_commit_is_R0']
        ms.before_labels(sroot,[(d1,'a85'),(d2,'a85'),(d3,'a85')]); m1=context.read_json(d1/'shadow_selection.json')
        assert all((d/'shadow_selection.json').exists() for d in (d1,d2,d3)) and context.read_json(sroot/'shadow_selections_frozen.json')['branches']==[str(d) for d in (d1,d2,d3)]
        ms.before_labels(sroot,[(d1,'a85')]); assert context.read_json(d1/'shadow_selection.json')==m1 and context.read_json(d1/'a85'/'commit.json')['material_id']=='FixedMixup'
    # (8) preparation with patched adapters; dispatch with patched branches: initialization after the common baselines and before every arm, shadow hook before labels.
    synthetic={'entities':table,'n_entities':32,'train_rows':[0,672],'batch_features':{}}
    with TemporaryDirectory() as td:
        root=Path(td)/'run'; root.mkdir(); log=[]
        def fake_adapter(run_dir,job,led,repo,**kw):
            log.append((job,kw.get('roster'))); return SimpleNamespace(ctx=SimpleNamespace(job=SimpleNamespace(roster=list(ms.G7),roster_override=ms.G7),stage='material'),overview=lambda: synthetic,baselines=lambda: ())
        led=rt.RuntimeLedger(root/'budget.json',**CAPS)
        with patch.object(rt,'Adapter',side_effect=fake_adapter):
            commons,pools=ms.prepare(root,led,CAPS)
        assert log==[('a65_g7',ms.G7),('a85_g7',ms.G7)] and all(pools[j]['status']=='COMPLETE' for j in ms.JOBS) and (root/ms.FROZEN_MARKER).exists() and ms.load_pools(root)=={j:context.read_json(root/(j+'_random_pool.json')) for j in ms.JOBS}
    with TemporaryDirectory() as td:
        root=Path(td)/'run'
        pools={j:{**ms.generate_supply(j,table),'job':j} for j in ms.JOBS}
        def fake_prepare(r,led,caps):
            for j,p in pools.items(): context.write_json(r/(j+'_random_pool.json'),p)
            context.write_json(r/ms.FROZEN_MARKER,{'epoch':0}); return {j:SimpleNamespace(baselines=lambda j=j: calls.append(('baselines',j))) for j in ms.JOBS},pools
        calls=[]
        done=br.RunResult('COMPLETE','a65_g7',0,'None','ref',1,1,0)
        def fake_init(r,job,led,shared,pool): calls.append(('init',job)); return {'status':'COMPLETE'}
        def fake_branch(path,*a,**k): calls.append(('arm',path.name)); return done
        with patch.object(rt,'MeteredClient') as mc, patch.object(ms,'prepare',side_effect=fake_prepare), patch.object(ms,'initialize',side_effect=fake_init), patch.object(base,'branch',side_effect=fake_branch) as brn, \
             patch.object(ms,'random_global_branch',side_effect=fake_branch) as rg, patch.object(ms,'random_local_branch',side_effect=fake_branch) as rl, patch.object(dc,'fixed_candidates_branch',side_effect=fake_branch) as fcb, \
             patch.object(ms,'before_labels',side_effect=lambda r,b: calls.append(('shadow',len(b)))) as bl, patch.object(base,'stage'), patch.object(sys.modules[__name__],'report'):
            mc.return_value.fatal=False
            (root).mkdir();
            for j in ms.JOBS: (root/(j+'_init')).mkdir(parents=True); context.write_json(root/(j+'_init')/'initialization.json',{'status':'COMPLETE'})
            worker(root)
        assert calls==[('baselines','a65_g7'),('init','a65_g7'),('arm','a65_g7_f_start'),('arm','a65_g7_f_free'),('arm','a65_g7_random_global'),('arm','a65_g7_random_local'),('arm','a65_g7_menu'),
                       ('baselines','a85_g7'),('init','a85_g7'),('arm','a85_g7_menu'),('arm','a85_g7_random_local'),('arm','a85_g7_random_global'),('arm','a85_g7_f_free'),('arm','a85_g7_f_start'),('shadow',10)],calls
        for c in brn.call_args_list:
            job,arm=c.args[0].name[:6],c.args[0].name[7:]; f=c.kwargs['adapter_factory']
            assert c.args[5].name==(job+'_init' if arm=='f_start' else job+'_common') and c.kwargs['limits']==ms.LIMITS[arm] and c.kwargs['evidence_roundtrip'] is True and c.kwargs['tool_contracts']==ms.arm_contracts(arm) and c.kwargs['roster']==ms.G7 and c.kwargs['seeds']==ms.SEEDS
            assert f.func is ms.MeasuredAdapter and f.keywords=={'arm':arm} and c.kwargs['max_tool_corrections']==2
        assert [c.args[3].name for c in rg.call_args_list]==['a65_g7_init','a85_g7_init'] and [c.args[3].name for c in rl.call_args_list]==['a65_g7_init','a85_g7_init'] and all(c.kwargs['plans']==list(ms.MENU) for c in fcb.call_args_list)
        fin=context.read_json(root/'execution_finished.json'); assert len(fin['branches'])==10 and not fin['failures']
    print('MEASURED_START_SMOKE_OK: config/G7, frozen supply, real R1/R2 initialization + Start-only, F_start supply/limits/cached evaluate/derived-from-init/no slot contract, F_free 4 slots and no leak, resume state, random-global/local from the same start, INIT_INCOMPLETE, R0/RB shadow frozen without touching commits, dispatch hooks')

def smoke():
    from unittest.mock import patch
    if STUDY=='domain_skill':
        raise RuntimeError('the domain_skill integration smoke is: python -m evaluation.main_protocol_p4.batch_research_domain_skill --smoke')
    cfg=configuration()
    if STUDY=='measured_start':
        smoke_measured_start(cfg)
        smoke_fault_paths()
        return
    if STUDY=='paired_refinement':
        smoke_paired_refinement(cfg)
        smoke_fault_paths()
        return
    if STUDY=='exploration_quota':
        smoke_exploration_quota(cfg)
        smoke_fault_paths()
        return
    if STUDY=='draft_construction':
        smoke_draft_construction(cfg)
        smoke_fault_paths()
        return
    if STUDY=='skill_revision':
        smoke_skill_revision(cfg)
        smoke_fault_paths()
        return
    if STUDY=='source_process':
        smoke_source_process(cfg)
        smoke_fault_paths()
    elif STUDY=='roundtrip':
        assert cfg['planned_fits']==48 and cfg['caps']['max_fit_attempts']==50
        assert cfg['order']['a85'][0]=='roundtrip' and cfg['order']['a75'][0]=='old'
    else:
        assert cfg['planned_fits']==36 and cfg['caps']['max_fit_attempts']==38
        assert cfg['max_tool_corrections']==2 and cfg['evidence_roundtrip'] is False
        assert cfg['http_cap']==64 and cfg['caps']['max_wall_s']==5400
        assert cfg['order']['a90'][0]=='fast' and cfg['order']['a95'][0]=='random'
        # Check the actual shared branch entry passes the option to both sides.
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        with TemporaryDirectory() as td:
            with patch.object(base.rt,'Adapter') as adapter, patch.object(base.br,'run_job',return_value=br.RunResult('INCOMPLETE','a90',0,None,None,0,0,0)) as run:
                adapter.return_value.baselines.return_value=()
                base.branch(Path(td)/'branch','a90',br.Knowledge(),SimpleNamespace(remaining=lambda:10,check_wall=lambda:None),SimpleNamespace(fast=lambda unit:None),max_tool_corrections=2)
                assert adapter.call_args.kwargs['tool_error_feedback'] is True
                assert run.call_args.kwargs['max_tool_corrections']==2
                assert run.call_args.kwargs['evidence_roundtrip'] is False
    calls=[]
    class FakeLedger:
        def remaining(self):return 4
    with patch.object(base.subprocess,'run',side_effect=lambda cmd,**kw: calls.append((cmd,kw)) or type('R',(),{'returncode':0})()):
        base.stage(Path('branch'),'a75','score_e',FakeLedger())
    assert '--stage' in calls[0][0] and 'score_e' in calls[0][0] and calls[0][1]['timeout']==4
    assert compare_commits(None,None,'external') is None
    print('ROUNDTRIP_SMOKE_OK: config, shared stage, missing evidence')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--worker',action='store_true');ap.add_argument('--report',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--resume-worker',action='store_true');ap.add_argument('--accept-unknown-usage',action='store_true');ap.add_argument('--finalize',action='store_true');ap.add_argument('--output',type=Path);ap.add_argument('--study',choices=['roundtrip','fast_baselines','source_process','skill_revision','draft_construction','exploration_quota','paired_refinement','measured_start','domain_skill'],default='roundtrip')
    ap.add_argument('--study-config',type=Path,default=None,help='domain_skill only: explicit JSON configuration (jobs, domains, arms, seeds, caps)')
    a=ap.parse_args();configure_study(a.study,a.study_config);root=(a.output or DEFAULT).resolve()
    extra=['--study-config',str(a.study_config.resolve())] if STUDY=='domain_skill' else []
    if STUDY=='domain_skill' and (a.run or a.worker or a.resume or a.resume_worker): dsk.preflight(dsk.CFG)   # no file, client, fit or label before this
    if a.smoke:smoke();return
    if (a.report or a.worker) and (root/'config.json').exists():
        stored=context.read_json(root/'config.json').get('study','roundtrip')
        if stored!=STUDY:raise RuntimeError('study does not match stored config; do not relabel an existing run')
    if a.report:report(root);return
    if a.worker:worker(root);return
    if a.resume_worker:resume_worker(root,accept_unknown_usage=a.accept_unknown_usage);return
    if a.finalize:finalize_worker(root);return
    if a.resume:
        led=context.read_json(study_ledger_path(root));remaining=int(CAPS['max_wall_s']-(time.time()-led['started_epoch']))
        if remaining<=0:raise RuntimeError('package wall already exhausted')
        with (root/'driver.log').open('a',encoding='utf-8') as f:
            p=subprocess.Popen([sys.executable,'-B','-m','evaluation.main_protocol_p4.run_batch_research_roundtrip','--resume-worker','--study',STUDY,'--output',str(root)]+extra+(['--accept-unknown-usage'] if a.accept_unknown_usage else []),cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
            context.write_json(root/'launch_resume.json',{'epoch':time.time(),'pid':p.pid,'supervisor_pid':os.getpid(),'deadline_seconds':remaining,'study':STUDY})
            try:rc=p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
                context.write_json(root/'supervisor_timeout.json',{'epoch':time.time(),'pid':p.pid});rc=124
        print('RESUME_EXIT',rc,flush=True);return
    if not a.run:ap.error('choose --run, --report or --smoke')
    if root.exists():raise RuntimeError('output exists; no restart/overwrite')
    root.mkdir(parents=True)
    context.write_json(root/'config.json',configuration())
    with (root/'driver.log').open('w',encoding='utf-8') as f:
        p=subprocess.Popen([sys.executable,'-B','-m','evaluation.main_protocol_p4.run_batch_research_roundtrip','--worker','--study',STUDY,'--output',str(root)]+extra,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        context.write_json(root/'launch.json',{'epoch':time.time(),'pid':p.pid,'supervisor_pid':os.getpid(),'deadline_seconds':CAPS['max_wall_s'],'study':STUDY})
        try:rc=p.wait(timeout=CAPS['max_wall_s'])
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
            context.write_json(root/'supervisor_timeout.json',{'epoch':time.time(),'pid':p.pid});rc=124
    print('ROUNDTRIP_EXIT',rc,flush=True)

if __name__=='__main__':main()
