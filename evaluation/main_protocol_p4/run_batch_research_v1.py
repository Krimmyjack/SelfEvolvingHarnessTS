"""One bounded W runner: live Fast -> commit -> delayed Slow -> later-job replay.
Run with the existing Windows project Python: -m evaluation.main_protocol_p4.run_batch_research_v1 --run
The supervisor enforces a 7200-second experiment deadline and kills only its own process tree.
"""
from __future__ import annotations
import argparse
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
from methods.ttha.batch_base import context, commit, spec
from evaluation.main_protocol_p4 import batch_research_runtime as rt

REPO=Path(__file__).resolve().parents[2]
DEFAULT=REPO/'_scratch/dev_batch_research_workflow_v1'

def ledger(root):
    return rt.RuntimeLedger(root/'budget.json',max_fit_attempts=33,max_llm_requests=50,max_llm_tokens=2000000,max_wall_s=7200,max_retries=2)

def event_sink(path):
    def emit(row):
        with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    return emit

def branch(root, job, h, led, client, shared=None, *, evidence_roundtrip=False, max_tool_corrections=0, tool_contracts=None, seeds=None, roster=None, adapter_factory=None, limits=None, dataset='electricity',
           commit_fn=None, allowed_features=None, entity_count=None, request_trace_dedupe=False):
    # request_trace_dedupe: opt-in request-side overview deduplication (br.dedupe_trace_for_request); False keeps every historical request byte-identical.
    # commit_fn / allowed_features / entity_count: a profile's own commit writer, batch-feature vocabulary and population size; None keeps the augmentation path.
    # dataset: spec.DATASETS key bound once here for the adapter (T, materials, fits) and the commit; default = historical electricity.
    seeds=tuple(seeds or rt.SEEDS)
    # limits: a study's explicit Limits fields (calls/tools/new evaluations); None keeps the historical two-slot default. Wall is always the remaining package wall.
    lim=br.Limits(**{**(limits or {}),'wall_seconds':led.remaining()})
    root.mkdir(parents=True,exist_ok=False)
    if shared: shutil.copytree(shared/job,root/job)
    # adapter_factory: a study's rt.Adapter subclass (same constructor); None keeps the live adapter.
    adapter=(adapter_factory or rt.Adapter)(root,job,led,REPO,tool_error_feedback=max_tool_corrections>0,roster=roster,seeds=seeds,dataset=dataset)
    baselines=adapter.baselines()
    context.write_json(root/'knowledge.json',asdict(h))
    result=br.run_job(job_id=job,knowledge=h,adapter=adapter,client=client.fast(root.name),seeds=seeds,baselines=baselines,
        allowed_features=rt.BATCH_FIELDS if allowed_features is None else allowed_features,tool_contracts=rt.CONTRACTS if tool_contracts is None else tool_contracts,limits=lim,
        on_event=event_sink(root/'trace.jsonl'),guard=led.check_wall,evidence_roundtrip=evidence_roundtrip,max_tool_corrections=max_tool_corrections,
        entity_count=32 if entity_count is None else entity_count,request_trace_dedupe=request_trace_dedupe)
    context.write_json(root/'branch_result.json',asdict(result))
    if result.status=='COMPLETE':
        (commit_fn or commit.commit)(root,dataset,job,result.committed_plan_id,result.trace[-2]['output']['reason'],seeds[0])
    print('BRANCH',root.name,result.status,'commit',result.committed_plan_id,flush=True)
    return result

def resume_branch(root, job, h, led, client, *, max_tool_corrections=0, tool_contracts=None, seeds=None, roster=None, adapter_factory=None, evidence_roundtrip=False, limits=None, dataset='electricity',
                  commit_fn=None, allowed_features=None, entity_count=None, request_trace_dedupe=False, allow_multiple_resumes=False):
    # allow_multiple_resumes: opt-in for a study whose stage may be stopped more than once (unknown-usage stops); a second resume keeps the earlier
    # *_before_resume files under numbered names (trace_before_resume2.jsonl, branch_result_before_resume2.json, resume2.json). False = historical behaviour.
    seeds=tuple(seeds or rt.SEEDS)
    lim=br.Limits(**{**(limits or {}),'wall_seconds':led.remaining()})
    """Continue a branch whose next Fast call was never sent (AGENT_CALL_FAILED/BudgetExhausted before transport).
    The request that will now be sent is the one the controller had built: same trace prefix, same
    candidates and feedback, same remaining budgets. Originals are kept as *_before_resume."""
    prior=context.read_json(root/'branch_result.json')
    if prior['status']!='INCOMPLETE' or prior['failure_kind']!='AGENT_CALL_FAILED' or (root/job/'commit.json').exists() or ((root/'resume.json').exists() and not allow_multiple_resumes):
        raise RuntimeError('branch is not in a resumable state')
    n_res=1+sum(1 for x in root.glob('resume*.json')) if allow_multiple_resumes else 1
    sfx='' if n_res==1 else str(n_res)
    if context.read_json(root/'knowledge.json')!=br.json_copy(asdict(h)): raise RuntimeError('frozen knowledge differs')
    rows=[json.loads(x) for x in (root/'trace.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    pending=max(i for i,r in enumerate(rows) if r['event']=='fast_request')
    if any(r['event']=='fast_response' for r in rows[pending:]): raise RuntimeError('last request was answered; not a never-sent call')
    prefix=rows[:pending]
    calls=sum(r['event']=='fast_response' for r in prefix); tools=sum(r['event']=='tool_started' for r in prefix)
    corrections=sum(r['event']=='tool_rejected' for r in prefix)
    if (calls+1,tools)!=(prior['calls'],prior['tool_calls']): raise RuntimeError('trace prefix and recorded counters disagree')
    baseline_ids=next(r['baseline_ids'] for r in prefix if r['event']=='job_started')
    built=[r['arguments']['plan_id'] for r in prefix if r['event']=='tool_started' and r['tool']=='build_material' and any(
        c['event']=='tool_completed' and c['tool']=='build_material' and c['output']['plan_id']==r['arguments']['plan_id'] for c in prefix)]
    evaluated=[r['output']['plan_id'] for r in prefix if r['event']=='tool_completed' and r['tool'] in ('evaluate','commit') and r['output'].get('plan_id') not in baseline_ids]
    evaluated_new=list(dict.fromkeys(evaluated))
    if len(evaluated_new)!=prior['new_evaluations']: raise RuntimeError('new-evaluation prefix disagrees with the record')
    features={}
    for r in prefix:
        if r['event']=='tool_completed' and r['tool']=='inspect_data': features.update(r['output'].get('batch_features',{}))
    adapter=(adapter_factory or rt.Adapter)(root,job,led,REPO,tool_error_feedback=max_tool_corrections>0,roster=roster,seeds=seeds,dataset=dataset)
    base_c=adapter.baselines()
    if [c.plan_id for c in base_c]!=baseline_ids: raise RuntimeError('baseline ids differ')
    fitted_ids=[m for m in built if m in evaluated_new]
    unfitted=[m for m in built if m not in evaluated_new]
    restored=adapter.restore(fitted_ids)
    for m in unfitted:  # built but never trained: restore the material only
        if hasattr(adapter,'restore_built'):   # a profile with its own material index
            restored.append(adapter.restore_built(m));continue
        idx=context.read_json(root/job/'materials'/'index.json');adapter.refs[m]=rt.materials.MaterialRef(**idx[m]);adapter.specs[m]=context.read_json(root/job/'materials'/(m+'__compiled.json'))
        restored.append(br.Candidate(m,adapter.specs[m],job))
    order={m:i for i,m in enumerate(built)}
    restored.sort(key=lambda c:order[c.plan_id])
    shutil.move(root/'trace.jsonl',root/('trace_before_resume%s.jsonl'%sfx));shutil.move(root/'branch_result.json',root/('branch_result_before_resume%s.json'%sfx))
    context.write_json(root/('resume%s.json'%sfx),{'epoch':time.time(),'reason':'next Fast call was never sent (client refused after a successfully retried transport fault); trajectory continued from the identical built request',
        'prefix_events':len(prefix),'calls_before':calls,'tools_before':tools,'new_evaluations_before':len(evaluated_new),'corrections_before':corrections,'restored_plans':[c.plan_id for c in restored],'new_fits_for_restore':0,
        'resume_number':n_res})
    sink=event_sink(root/'trace.jsonl')
    for r in prefix: sink(r)
    sink({'event_id':job+':resume','event':'job_resumed','calls':calls,'tools_used':tools,'new_evaluations':len(evaluated_new),'corrections_used':corrections,'restored':[c.plan_id for c in restored],'epoch':time.time()})
    # restored plans (evaluated or only built) go in build order through resume['built'], never as evaluated baselines
    resume={'trace':prefix,'calls':calls,'tools_used':tools,'new_evaluations':len(evaluated_new),'corrections_used':corrections,'evaluated_new':evaluated_new,'features':features,'built':tuple(restored)}
    result=br.run_job(job_id=job,knowledge=h,adapter=adapter,client=client.fast(root.name),seeds=seeds,baselines=tuple(base_c),
        allowed_features=rt.BATCH_FIELDS if allowed_features is None else allowed_features,tool_contracts=rt.CONTRACTS if tool_contracts is None else tool_contracts,limits=lim,
        on_event=sink,guard=led.check_wall,evidence_roundtrip=evidence_roundtrip,max_tool_corrections=max_tool_corrections,resume=resume,
        entity_count=32 if entity_count is None else entity_count,request_trace_dedupe=request_trace_dedupe)
    context.write_json(root/'branch_result.json',asdict(result))
    if result.status=='COMPLETE':
        (commit_fn or commit.commit)(root,dataset,job,result.committed_plan_id,result.trace[-2]['output']['reason'],seeds[0])
    print('BRANCH_RESUMED',root.name,result.status,'commit',result.committed_plan_id,flush=True)
    return result

def stage(root, job, name, led, roster=None, dataset='electricity'):
    cmd=[sys.executable,'-B','-m','evaluation.main_protocol_p4.run_batch_research_v1','--stage',name,'--output',str(root),'--job',job,'--dataset',dataset]
    if roster: cmd+=['--roster',','.join(roster)]
    pr=subprocess.run(cmd,cwd=REPO,timeout=led.remaining())
    if pr.returncode: raise RuntimeError('stage failed: '+name)

def alignment(root,led):
    import numpy as np
    import torch
    previous=REPO/'_scratch/dev_batch_research_workflow_v1_alignment'
    if root.resolve()!=previous.resolve() and (previous/'completed.json').exists():
        receipt=context.read_json(previous/'completed.json')
        check=context.read_json(previous/'alignment.json')
        if receipt['fit_attempts']!=1 or receipt['api_calls']!=0 or check['weights_max_abs_diff']!=0 or check['c_a_diff']!=0 or not check['materials_equal']:
            raise RuntimeError('prior alignment not qualified; do not consume a second original-seed fit')
        oldroot=previous/'alignment/a55'
        model=oldroot/'runs/a55__R__s20260918.pt'
        weights=torch.load(model,map_location='cpu',weights_only=True)
        expected=torch.load(REPO/'_scratch/ts_aug_donor_pattern_probe/runs/a55_R_s20260918.pt',map_location='cpu',weights_only=True)
        if any(not torch.equal(weights[k],expected[k]) for k in weights): raise RuntimeError('stored alignment weights changed')
        spent=context.read_json(previous/'budget.json')
        led.s['fit_attempts']+=spent['fit_attempts'];led.s['fits_ok']+=spent['fits_ok'];led.s['fits_failed']+=spent['fits_failed']
        led.s['fit_wall_seconds']+=spent['fit_wall_seconds']
        led.s['started_epoch']-=receipt['seconds']
        led.s['events'].append({'kind':'prior_local_alignment','source':str(previous),'fit_attempts':1,'seconds':receipt['seconds']})
        led._save();led.check_wall()
        context.write_json(root/'alignment.json',{**check,'reused_from':str(previous),'new_fits':0})
        print('ALIGNMENT reused local exact reproduction; 1 prior fit counted in total33',flush=True)
        return
    a=rt.Adapter(root/'alignment','a55',led,REPO)
    c=a.build_material({'plan_id':'R','policy':rt.policy.fixed_mixup_policy()},remaining_seconds=led.remaining())
    probe=REPO/'_scratch/ts_aug_donor_pattern_probe'
    with np.load(probe/'materials/a55__children.npz') as old, np.load(a.refs['R'].path) as new:
        if not (np.array_equal(old['RX'],new['Xc']) and np.array_equal(old['RY'],new['Yc'])): raise RuntimeError('material alignment failed')
    recs=rt.fit(a.ctx,a.refs['R'],(20260918,),led,REPO)
    mine=torch.load(recs[0]['model_path'],map_location='cpu',weights_only=True)
    old=torch.load(probe/'runs/a55_R_s20260918.pt',map_location='cpu',weights_only=True)
    diff=max(float((mine[k]-old[k]).abs().max()) for k in mine)
    pc=context.read_json(probe/'cells/a55_R_s20260918.json')
    delta=recs[0]['scores']['c_a']['normalized_mse_macro']-pc['scores']['c_a']['normalized_mse_macro']
    context.write_json(root/'alignment.json',{'weights_max_abs_diff':diff,'c_a_diff':delta,'materials_equal':True,'fit_attempts':1})
    if diff!=0 or delta!=0: raise RuntimeError('original seed alignment failed')
    print('ALIGNMENT weights and C_A exact',flush=True)

def update(root,source,client,led):
    result=context.read_json(source/'branch_result.json')
    cb=context.read_json(source/'a55/c_b_scores.json')
    refs=frozenset(row['event_id'] for row in result['trace'])|{'a55:delayed'}
    payload={'parent_knowledge':asdict(br.Knowledge()),'source_episode':result,'delayed_evidence':{'reference':'a55:delayed','results':cb},
             'legal_evidence_refs':sorted(refs),'allowed_batch_features':sorted(rt.BATCH_FIELDS),
             'batch_field_semantics':'Each batch_median_FIELD is median of current 32 T-only entity FIELD values. Material predicates use per-entity names separately.',
             'status':'CANDIDATE_TEST_ONLY'}
    context.write_json(root/'slow_evidence.json',payload)
    prior=None
    for i in range(2):
        # Only contract/JSON errors permit one repair. Backend errors are not semantic retries.
        try:
            proposal=client.call('slow' if i==0 else 'slow_contract_correction','a55_boundary',payload,rt.SLOW_SYSTEM)
            prior=proposal
            h=br.apply_update(br.Knowledge(),proposal,allowed_features=rt.BATCH_FIELDS,legal_evidence_refs=refs)
            for item in h.entries: rt.policy.validate_text(item.body,allow_entity_ids=False)
            context.write_json(root/'slow_result.json',{'status':'KEEP' if h.version==0 else 'CANDIDATE_TEST_ONLY','proposal':proposal,'knowledge':asdict(h)})
            return h
        except (ValueError,rt.policy.PolicyError) as exc:
            context.write_json(root/('slow_contract_error_%d.json'%i),{'type':type(exc).__name__,'message':str(exc),'proposal':prior})
            if i==0:
                payload={**payload,'correction':{'previous_output':prior if prior is not None else client.last_text,'error':str(exc),'instruction':'Correct this contract error only. KEEP is allowed. Do not seek a different outcome.'}}
                continue
    context.write_json(root/'slow_result.json',{'status':'PARSE_OR_VALIDATION_FAILED','knowledge':asdict(br.Knowledge())})
    return br.Knowledge()

def run_worker(root):
    if (root/'experiment_started.json').exists(): raise RuntimeError('run already started; do not replay paid calls')
    root.mkdir(parents=True,exist_ok=True)
    led=ledger(root)
    context.write_json(root/'experiment_started.json',{'epoch':time.time(),'pid':os.getpid(),'repo':str(REPO),'seeds':rt.SEEDS,'tool_contracts':rt.CONTRACTS,'exposure':'EXPOSED_DEVELOPMENT'})
    branches=[];failures=[];h=br.Knowledge()
    try:
        client=rt.MeteredClient(led,root/'raw_responses')  # config only; no unmetered ping
        alignment(root,led)
        src=root/'a55_H0'
        result=branch(src,'a55',h,led,client);branches.append((src,'a55'))
        if client.fatal: raise RuntimeError('fatal backend fault')
        if result.status=='COMPLETE':
            stage(src,'a55','c_b',led)
            try: h=update(root,src,client,led)
            except rt.llm.AccountFault: raise
            except Exception as exc:
                failures.append({'stage':'slow','exception_type':type(exc).__name__})
                context.write_json(root/'slow_result.json',{'status':'FORMATION_FAILED','exception_type':type(exc).__name__})
        common=root/'a65_common';common.mkdir()
        rt.Adapter(common,'a65',led,REPO).baselines()
        par=root/'a65_parent'
        result=branch(par,'a65',br.Knowledge(),led,client,common);branches.append((par,'a65'))
        if client.fatal: raise RuntimeError('fatal backend fault')
        if h.version:
            child=root/'a65_candidate'
            result=branch(child,'a65',h,led,client,common);branches.append((child,'a65'))
            if client.fatal: raise RuntimeError('fatal backend fault')
        else: context.write_json(root/'treatment.json',{'status':'NO_TREATMENT','reason':'No legal nonempty boundary update; no artificial candidate arm'})
        # All planned branches frozen before ANY a65 C_B. Slow never sees a65 data.
        context.write_json(root/'decisions_frozen.json',{'epoch':time.time(),'branches':[str(p) for p,j in branches]})
        for p,j in branches:
            if (p/j/'commit.json').exists() and not (p/j/'c_b_scores.json').exists(): stage(p,j,'c_b',led)
    except Exception as exc:
        failures.append({'stage':'worker','exception_type':type(exc).__name__})
        print('WORKER_STOP',type(exc).__name__,flush=True)
    finally:
        context.write_json(root/'execution_finished.json',{'epoch':time.time(),'branches':[(str(p),j) for p,j in branches],'failures':failures})
    # No new Agent work from here; freeze every E prediction before any E target read.
    try:
        eligible=[(p,j) for p,j in branches if (p/j/'c_b_scores.json').exists()]
        for p,j in eligible: stage(p,j,'freeze_e',led)
        context.write_json(root/'all_e_predictions_frozen.json',{'epoch':time.time(),'branches':[str(p) for p,j in eligible]})
        for p,j in eligible: stage(p,j,'score_e',led)
    except Exception as exc: failures.append({'stage':'external_evaluation','exception_type':type(exc).__name__})
    context.write_json(root/'execution_finished.json',{'epoch':time.time(),'branches':[(str(p),j) for p,j in branches],'failures':failures})
    report(root)

def report(root, seeds=None):
    delivery_seed=tuple(seeds or rt.SEEDS)[0]
    summary={'status':'PARTIAL','branches':{},'exposure':'EXPOSED_DEVELOPMENT','failures':[]}
    receipt=root/'execution_finished.json'
    if receipt.exists(): summary['failures']=context.read_json(receipt)['failures']
    lines=['# W 批级研究 Workflow 首包','', '开发结果；不代表完整 A3/A5 或独立泛化。','', '|分支|状态|Fast commit|交付 seed E|三 seed E 均值|','|---|---|---|---:|---:|']
    for p in sorted(root.glob('*/branch_result.json')):
        r=context.read_json(p);job=r['job_id'];entry={'controller':{k:v for k,v in r.items() if k!='trace'},'plans':{}}
        cb=p.parent/job/'c_b_scores.json';ee=p.parent/job/'e_scores.json'
        es=context.read_json(ee)['cells'] if ee.exists() else {}
        cs=context.read_json(cb)['cells'] if cb.exists() else {}
        for cell in sorted((p.parent/job/'cells').glob('*.json')):
            c=context.read_json(cell)
            if c.get('status')!='OK': continue
            mid=c['material_id'];item=entry['plans'].setdefault(mid,[])
            item.append({'seed':c['model_seed'],'c_a':c['scores'].get('c_a'),'c_b':cs.get(c['cell_id'],{}).get('c_b'),'external':es.get(c['cell_id'],{}).get('e'),'model_path':c['model_path']})
        chosen=entry['plans'].get(r['committed_plan_id'],[])
        ev=[c['external']['normalized_mse_macro'] for c in chosen if c['external']]
        delivery=next((c['external']['normalized_mse_macro'] for c in chosen if c['seed']==delivery_seed and c['external']),None)
        mean=sum(ev)/len(ev) if len(ev)==3 else None
        entry['delivery_e']=delivery;entry['paired_mean_e']=mean
        entry['treatment_present']=any(row.get('event')=='fast_request' and row.get('guidance',{}).get('loaded') for row in r['trace'])
        entry['tool_sequence']=[row['tool'] for row in r['trace'] if row.get('event')=='tool_completed']
        entry['vs_baselines']={}
        for baseline in ('None','FixedMixup'):
            base={c['seed']:c for c in entry['plans'].get(baseline,[])}
            block_results={}
            for block in ('c_a','c_b','external'):
                differences=[base[c['seed']][block]['normalized_mse_macro']-c[block]['normalized_mse_macro'] for c in chosen if c['seed'] in base and c[block] is not None and base[c['seed']][block] is not None]
                if len(differences)==3:
                    block_results[block]={'delta_by_seed':differences,'mean_delta':statistics.mean(differences),'seed_se':statistics.stdev(differences)/3**.5,'positive_means':'committed_plan_improves'}
            entry['vs_baselines'][baseline]=block_results
        summary['branches'][p.parent.name]=entry
        lines.append('|%s|%s|%s|%s|%s|'%(p.parent.name,r['status'],r['committed_plan_id'],delivery,mean))
    parent=summary['branches'].get('a65_parent');child=summary['branches'].get('a65_candidate')
    if parent and child:
        aa=parent['plans'].get(parent['controller']['committed_plan_id'],[])
        bb=child['plans'].get(child['controller']['committed_plan_id'],[])
        summary['new_guidance_vs_parent']={}
        for block in ('c_a','c_b','external'):
            by_seed={c['seed']:c for c in aa}
            ds=[by_seed[c['seed']][block]['normalized_mse_macro']-c[block]['normalized_mse_macro'] for c in bb if c['seed'] in by_seed and c[block] is not None and by_seed[c['seed']][block] is not None]
            if len(ds)==3: summary['new_guidance_vs_parent'][block]={'delta_by_seed':ds,'mean_delta':statistics.mean(ds),'seed_se':statistics.stdev(ds)/3**.5,'positive_means':'candidate_guidance_improves','scope':'Single exposed development job, training seeds not independent method trajectories'}
    slow=root/'slow_result.json';summary['slow']=context.read_json(slow) if slow.exists() else None
    needed=3 if summary['slow'] and summary['slow'].get('status')=='CANDIDATE_TEST_ONLY' else 2
    if len(summary['branches'])==needed and all(x['controller']['status']=='COMPLETE' and x['delivery_e'] is not None for x in summary['branches'].values()) and not summary['failures']:
        summary['status']='COMPLETE'
    if (root/'budget.json').exists():summary['budget']={k:v for k,v in context.read_json(root/'budget.json').items() if k!='events'}
    context.write_json(root/'result.json',summary)
    lines+=['','状态：'+summary['status'],'','Slow：'+str((summary['slow'] or {}).get('status','NOT_RUN')),'','预算与完整逐 seed / origin / entity 分数见 result.json；真实请求和返回见 raw_responses/。','实体预测差是共同训练结果的贡献分解，不是单实体训练动作的因果价值。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT',summary['status'],str(root/'REPORT.md'),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--worker',action='store_true');ap.add_argument('--report',action='store_true');ap.add_argument('--stage',choices=['c_b','freeze_e','score_e']);ap.add_argument('--job');ap.add_argument('--output',type=Path,default=DEFAULT);ap.add_argument('--roster',default=None);ap.add_argument('--dataset',choices=sorted(spec.DATASETS),default='electricity')
    a=ap.parse_args();root=a.output.resolve()
    if a.stage:
        roster=None if a.roster is None else a.roster.split(',')
        if a.stage == 'score_e':
            commit.score_e(root,a.dataset,a.job,cohort_root=root.parent,roster=roster)
        else:
            getattr(commit,'open_c_b' if a.stage=='c_b' else a.stage)(root,a.dataset,a.job,roster=roster)
        return
    if a.report: report(root);return
    if a.worker: run_worker(root);return
    if not a.run: ap.error('choose --run or --report')
    if root.exists(): raise RuntimeError('output exists; inspect rather than overwrite/restart')
    root.mkdir(parents=True)
    with (root/'driver.log').open('w',encoding='utf-8') as f:
        p=subprocess.Popen([sys.executable,'-B','-m','evaluation.main_protocol_p4.run_batch_research_v1','--worker','--output',str(root)],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        context.write_json(root/'launch.json',{'epoch':time.time(),'pid':p.pid,'supervisor_pid':os.getpid(),'deadline_seconds':7200,'repo':str(REPO)})
        try: rc=p.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
            context.write_json(root/'supervisor_timeout.json',{'epoch':time.time(),'pid':p.pid})
            rc=124
    print('RUN_EXIT',rc,flush=True)
    if rc: report(root)

if __name__=='__main__': main()
