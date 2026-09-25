"""W's thin live adapter over the frozen P numerical base. No alternate trainer."""
from __future__ import annotations
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from methods.ttha import batch_research as br
from methods.ttha.batch_base import context, materials, policy, spec, train, budget, llm

SEEDS = (20260922, 20260923, 20260924)
FIT_RETRY = False   # set by a study that authorizes same-configuration non-scientific fit retries (ledger max_retries caps them)
BATCH_FIELDS = frozenset('batch_median_' + f for f in spec.OBS_FIELDS)
CONTRACTS = {
    'overview': {'arguments': {}, 'meaning': 'All 32 T-only rows, batch summaries, Consumer and legal actions.'},
    'inspect_data': {'arguments': {'entity_indices': '[0..31 integers]', 'kind': 'hour_profile|daily_means|segment', 'sub_range': 'optional [absolute_start,absolute_end) inside T'}, 'meaning': 'Inspect current T only.'},
    'build_material': {'arguments': {'plan_id': 'new short alphanumeric/underscore id', 'policy': {'default': {'steps': []}, 'rules': [{'when': {'feature': '<overview field>', 'op': '>=', 'value': {'quantile': 0.5}}, 'steps': []}], 'rationale': '<current hypothesis, no dataset/job name>', 'observation_fields_used': []}}, 'meaning': 'Provide actual values, not placeholders. First matching rule wins; no rule means default. Empty steps means identity. Each step uses op timemixup/freqmask/freqmix and all parameters from overview.actions. Rules optional, up to 8; at most 2 steps. Predicate all/any/not/const supported; threshold numeric or current-batch quantile. This tool constructs materials but does not train.'},
    'inspect_material': {'arguments': {'plan_id': '<built id>'}, 'meaning': 'T-only material change and donor diagnostics; not predictive utility.'},
    'evaluate': {'arguments': {'plan_id': '<built id>'}, 'meaning': 'Train the complete shared plan under 3 frozen paired seeds; return C_A only. At most 2 new complete plans.'},
    'compare': {'arguments': {'a': '<evaluated id>', 'b': '<evaluated id>'}, 'meaning': 'Positive paired delta means b has lower C_A loss; seed uncertainty, not future certainty.'},
    'commit': {'arguments': {'plan_id': '<evaluated id>', 'reason': '<brief evidence-based decision>'}, 'meaning': 'Final action. Pick any evaluated plan including None or Fixed-Mixup. No automatic argmin, no later selection override.'},
}

# Public tool facts (DEV-BATCH-RESEARCH-SOURCE-PROCESS §3): semantics of the live donor rules and apply path,
# stated for every arm that opts in; no efficacy advice. Historical entry points keep the bare CONTRACTS.
TOOL_SEMANTICS = {
    'donor_rule_R': "Within ONE entity: a random derangement over that entity's own 433 original parent pairs (no self-pairing, one-to-one). Donor and recipient belong to the same entity.",
    'donor_rule_U': 'Within ONE entity: the original parent pairs are grouped by the clock hour at which the parent window starts (24 groups); one derangement is drawn inside each hour group, so every donor starts at the same clock hour as its recipient.',
    'donor_scope': 'Neither rule is a cross-entity donor. Donors are always ORIGINAL parent pairs of the same entity; derived (augmented) pairs are never donors, also in a second step.',
    'construction': 'TimeMixup and FreqMix act jointly on the legal historical window [X;y] (192 input + 48 target rows of one parent pair); FreqMask zeroes random rFFT bins of that joint window. Each (entity, step) has its own frozen random stream. The per-entity scaler is frozen from real T only; derived pairs are not re-normalized.',
    'training': 'One complete policy is expanded to all 32 entities (first matching rule, else default; empty steps = identity). All original parent pairs and all derived pairs of that plan enter ONE shared model training (parent view 0.5 + child view 0.5; identity entities supply their parent view in the child slot). Plans with an identical expanded assignment are the same material (alias) and are not fitted again.',
    'interpretation': "A change in one entity's prediction loss between two complete plans is a contribution readout of the shared model, not the independent causal value of that entity's training material.",
}


def contracts_with_semantics():
    """CONTRACTS plus the public tool facts above; the mapping keys stay exactly the tool names."""
    out = br.json_copy(CONTRACTS)
    facts = dict(TOOL_SEMANTICS)
    out['build_material']['semantics'] = {k: facts[k] for k in ('donor_rule_R', 'donor_rule_U', 'donor_scope', 'construction', 'training')}
    out['inspect_material']['semantics'] = {k: facts[k] for k in ('donor_scope', 'interpretation')}
    out['evaluate']['semantics'] = {k: facts[k] for k in ('training', 'interpretation')}
    out['compare']['semantics'] = {'interpretation': facts['interpretation']}
    return out


class RuntimeLedger(budget.Ledger):
    def remaining(self):
        self.check_wall()
        return max(0., self.s['caps']['max_wall_s'] - (time.time()-self.s['started_epoch']))

    def reserve_fit(self, cid):
        self.check_fit()
        self.s['fit_attempts'] += 1
        self.s['events'].append({'kind': 'fit_started', 'cell': cid, 'epoch': time.time()})
        self._save()

    def finish_fit(self, cid, ok, seconds, reason):
        self.s['fits_ok' if ok else 'fits_failed'] += 1
        self.s['fit_wall_seconds'] += seconds
        self.s['events'].append({'kind': 'fit_finished', 'cell': cid, 'ok': ok, 'seconds': seconds, 'reason': reason, 'epoch': time.time()})
        self._save()


def fit(ctx, ref, seeds, ledger, repo, feedback=True):
    """Serial, reserved-before-launch workers; strict remaining-time deadline."""
    out=[]
    for seed in seeds:
        cid=train.cell_id(ctx.job.job_id, ref.material_id, seed)
        cp=ctx.job_dir/'cells'/(cid+'.json')
        cp.parent.mkdir(exist_ok=True)
        src=cp
        if not src.exists() and ref.alias_of:
            src=cp.parent/(train.cell_id(ctx.job.job_id, ref.alias_of, seed)+'.json')
        if src.exists():
            rec=context.read_json(src)
            if rec.get('status')!='OK' or rec['model_seed']!=seed or rec['job_id']!=ctx.job.job_id:
                raise RuntimeError('invalid cached fit')
            # Semantic cache binding (existing record fields, no hash): source, population, training slice, candidate.
            if rec.get('dataset','electricity')!=ctx.job.dataset: raise RuntimeError('cached fit belongs to another dataset')
            if (rec.get('roster') is None and ctx.job.roster_override is not None) or (rec.get('roster') is not None and list(rec['roster'])!=ctx.job.roster):
                raise RuntimeError('cached fit belongs to another population')
            if feedback and rec.get('rows_read') is not None and list(rec['rows_read'])!=list(ctx.job.evaluate_rows): raise RuntimeError('cached fit used another training/evaluation slice')
            if rec.get('material_id') not in (ref.material_id,ref.alias_of): raise RuntimeError('cached fit belongs to another candidate')
            if feedback and not rec.get('scores',{}).get('c_a'):
                raise RuntimeError('calibration-free model cannot masquerade as evaluated cache')
            if not Path(rec['model_path']).exists(): raise RuntimeError('cached model missing')
            if src!=cp:
                rec={**rec,'cell_id':cid,'material_id':ref.material_id,'material_alias_of':ref.alias_of,'source_cell':rec['cell_id']}
                context.write_json(cp,rec)
            ledger.note_cache(cid);out.append(rec);continue
        for attempt in (0,1):
            ledger.reserve_fit(cid)
            timeout=min(300.,ledger.remaining())
            # fit_module: a context may name another worker with the same CLI (e.g. the data-readiness profile); default = the augmentation worker.
            cmd=[sys.executable,'-B','-m',getattr(ctx,'fit_module','methods.ttha.batch_base.train'),'--dataset',ctx.job.dataset,'--job',ctx.job.job_id,'--run-dir',str(ctx.run_dir),'--material',ref.material_id,'--seed',str(seed),'--train-deadline',str(max(1.,timeout-15))]
            if not feedback: cmd.append('--no-feedback')
            if getattr(ctx.job,'roster_override',None) is not None: cmd+=['--roster',','.join(ctx.job.roster)]
            start=time.time(); ok=False;reason=''
            logs=ctx.job_dir/'fit_logs';logs.mkdir(exist_ok=True)
            try:
                with (logs/(cid+('.log' if attempt==0 else '.retry.log'))).open('w',encoding='utf-8') as log:
                    p=subprocess.run(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                ok=p.returncode==0 and cp.exists() and context.read_json(cp).get('status')=='OK'
                reason='' if ok else 'worker_failed'
            except subprocess.TimeoutExpired: reason='worker_timeout'
            finally: ledger.finish_fit(cid,ok,time.time()-start,reason)
            print('FIT',cid,'OK' if ok else reason,round(time.time()-start,1),flush=True)
            if ok: break
            # Opt-in (FIT_RETRY): one same-configuration re-launch of this exact cell, charged as a physical attempt and
            # drawn from the ledger's run-wide retry cap; never a new seed or candidate. Default off = historical behaviour.
            if attempt==0 and FIT_RETRY and ledger.can_retry():
                ledger.s['retries_used']+=1
                ledger.s['events'].append({'kind':'fit_retry','cell':cid,'reason':reason,'epoch':time.time()});ledger._save()
                continue
            raise RuntimeError(reason)
        rec=context.read_json(cp)
        if feedback and rec['scores']['c_a']['n_nonfinite_predictions']:
            raise RuntimeError('nonfinite predictions')
        out.append(rec)
    return out

class Adapter:
    def __init__(self, run_dir, job, ledger, repo, *, tool_error_feedback=False, roster=None, seeds=SEEDS, dataset='electricity'):
        # roster: explicit 32-entity population (None = dataset default); seeds: this study's frozen paired training seeds.
        # dataset: spec.DATASETS key; the default keeps every historical electricity call unchanged.
        self.ctx=context.open_job(dataset,job,run_dir,'material',roster=roster)
        self.seeds=tuple(seeds)
        self.ledger,self.repo=ledger,repo
        self.refs={};self.specs={};self.fitted={}
        self.tool_error_feedback=tool_error_feedback

    def overview(self):
        ov=br.json_copy(self.ctx.overview())
        ov['batch_features']={'batch_median_'+k:v['median'] for k,v in ov['summary'].items() if v is not None}
        ov['batch_feature_definition']='Median across all 32 T-only entity observations; never a first-entity proxy.'
        return ov

    def _reject_input(self, message):
        if self.tool_error_feedback:
            raise br.ToolInputError(message)
        raise ValueError(message)

    def inspect_data(self, arguments, *, remaining_seconds):
        if set(arguments)-{'entity_indices','kind','sub_range'}: self._reject_input('unknown inspection argument')
        indices=arguments.get('entity_indices')
        if not isinstance(indices,list) or not indices or any(type(i)!=int or not 0<=i<32 for i in indices): self._reject_input('entity indices must be 0..31')
        if self.tool_error_feedback:
            if arguments.get('kind','hour_profile') not in ('hour_profile','daily_means','segment'):
                self._reject_input('kind must be hour_profile, daily_means or segment')
            span=arguments.get('sub_range')
            if span is not None and (not isinstance(span,list) or len(span)!=2 or any(type(x)!=int for x in span) or span[0]>=span[1]):
                self._reject_input('sub_range must be two increasing integer boundaries')
        # Actual permission/region checks are not converted into recoverable input errors.
        return self.ctx.inspect_data(indices,arguments.get('kind','hour_profile'),arguments.get('sub_range'))

    def build_material(self, arguments, *, remaining_seconds):
        if set(arguments)!={'plan_id','policy'}: self._reject_input('build needs plan_id and policy')
        mid=arguments['plan_id']
        if not isinstance(mid,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}',mid): self._reject_input('unsafe plan id')
        if mid in self.refs: self._reject_input('plan id already exists; use existing plan or a new id')
        try:
            compiled=policy.compile_policy(arguments['policy'],self.overview()['entities'])
        except policy.PolicyError as exc:
            if not self.tool_error_feedback:
                raise
            # Only grammar compilation is wrapped; materials.build, writes and fits are not.
            raise br.ToolInputError(str(exc), valid_observation_fields=list(spec.OBS_FIELDS)) from None
        self.check_compiled(mid,compiled)   # study-specific post-compilation admission (before any material or fit); no-op here
        ref=materials.build(self.ctx,compiled['assignment'],mid)
        ms={k:compiled[k] for k in ('policy','assignment','rule_index','resolved_thresholds','n_unknown')}
        self.refs[mid]=ref;self.specs[mid]=ms
        context.write_json(self.ctx.job_dir/'materials'/(mid+'__compiled.json'),ms)
        return br.Candidate(mid,ms,self.ctx.job.job_id)

    def check_compiled(self, plan_id, compiled):
        """Hook for an arm-level admission rule on the compiled assignment; the base adapter admits every legal policy."""

    def check_evaluate(self, plan_id, candidate):
        """Hook run by the controller before a new evaluation is counted or any fit reserved; no-op by default."""

    def check_commit(self, plan_id, candidate):
        """Hook run by the controller before a delivery is bound or written; no-op by default."""

    def inspect_material(self, plan_id, arguments, *, remaining_seconds):
        if set(arguments)!={'plan_id'}: self._reject_input('inspect_material only takes plan_id')
        return materials.inspect_material(self.ctx,self.refs[plan_id])

    def evaluate(self, candidate, seeds, *, feedback, remaining_seconds):
        recs=fit(self.ctx,self.refs[candidate.plan_id],seeds,self.ledger,self.repo,feedback)
        losses=tuple(tuple(tuple(row) for row in r['scores']['c_a']['per_origin_entity_normalized_mse']) for r in recs) if feedback else ()
        fitted=replace(candidate,model_seeds=tuple(seeds),model_refs=tuple(r['model_path'] for r in recs),ca_losses=losses)
        self.fitted[candidate.plan_id]=fitted
        return fitted

    def restore(self, plan_ids):
        """Rebuild already built (and fitted) private plans of this branch from disk; fits are cache hits only."""
        index=context.read_json(self.ctx.job_dir/'materials'/'index.json')
        out=[]
        for mid in plan_ids:
            if mid in self.refs: raise ValueError('plan already present')
            self.refs[mid]=materials.MaterialRef(**index[mid])
            self.specs[mid]=context.read_json(self.ctx.job_dir/'materials'/(mid+'__compiled.json'))
            before=self.ledger.s['fit_attempts']
            fitted=self.evaluate(br.Candidate(mid,self.specs[mid],self.ctx.job.job_id),self.seeds,feedback=True,remaining_seconds=self.ledger.remaining())
            if self.ledger.s['fit_attempts']!=before: raise RuntimeError('restore must not fit')
            out.append(fitted)
        return out

    def baselines(self):
        result=[]
        for mid,p in [('None',policy.identity_policy()),('FixedMixup',policy.fixed_mixup_policy())]:
            c=self.build_material({'plan_id':mid,'policy':p},remaining_seconds=self.ledger.remaining())
            result.append(self.evaluate(c,self.seeds,feedback=True,remaining_seconds=self.ledger.remaining()))
        return tuple(result)

FAST_SYSTEM='''You are the Fast path of a batch training-data research Harness. Your unit is one entire 32-entity job and one shared MLP, not separate per-entity models. Use only current T observations, tool results and C_A. Investigate, construct, evaluate, compare or stop as useful within budget. Uniform and heterogeneous plans are both legal; diversity is not a goal. FixedMixup and None are fully evaluated common baselines. You may commit either baseline without spending more fits. No C_B or E is available; do not invent future results. The whole shared-plan comparison is the value signal; individual prediction changes are not causal labels for those entities' training data. Skill advice does not prohibit public tools. Return exactly one JSON object {"actions":[{"tool":"listed_name","arguments":{...}}]}, no markdown. Batched actions execute in order and must use valid IDs. Commit must be last. State brief hypotheses in material rationale or commit reason, not hidden chain-of-thought. Do not include dataset/job identifiers in reusable rationale. Do not change seeds, model, training budget, scaler, scoring or population.'''

SLOW_SYSTEM='''You are the boundary Slow of the same batch research Harness. You see only the completed source job's legal episode, C_A and post-commit delayed C_B. E and next-job data are absent. Propose at most ONE reusable behavioral hook change or KEEP. Hooks: observation_guidance, construction_guidance, experiment_guidance, decision_guidance. Target one mechanism; do not change model, scoring, tool implementation, permissions or budgets. Distinguish observed evidence, hypotheses and untested actions; don't turn missing evidence into prohibitions. Shared-plan loss differences do not identify causal entity-level training value. Do not cite dataset names, job labels, entity IDs or memorize a current assignment. Guidance should tell future Fast how to observe, construct, compare or decide, with evidence limitations. A proposal is CANDIDATE_TEST_ONLY, not promoted knowledge. Output exact JSON: {"decision":"KEEP","rationale":"..."} OR {"decision":"PROPOSE","hook":"one hook","body":"<=1200 characters","observable_applicability":{"const":true},"evidence_refs":["supplied reference"],"rationale":"..."}. const:true is explicit unconditional; null/missing is incomplete, never unconditional. Conditional scopes use only supplied batch feature names with numeric {feature,op,value} leaves (>,>=,<,<=,==), all/any/not/const. No implicit field renaming. KEEP is a valid outcome and will not be resampled.'''

def unknown_usage_blocks(led):
    """True while the ledger holds unknown usage that no operator decision has accepted."""
    return led.s['llm_tokens_unknown']>led.s.get('unknown_usage_accepted',0)


class MeteredClient:
    """Shared transport classification; reserve attempts and conservative token bounds before sends."""
    def __init__(self, ledger, out, *, http_cap=100):
        import openai
        key=next((os.environ.get(n,'').strip() for n in ('CPA_API_KEY','OPENAI_API_KEY') if os.environ.get(n,'').strip()),'')
        if not key: raise llm.AccountFault('API key unavailable')
        self.api=openai.OpenAI(api_key=key,base_url='http://127.0.0.1:8318/v1',max_retries=0,timeout=300)
        self.ledger,self.out=ledger,Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.fatal=False
        self.http_cap=http_cap
        self.last_text=None

    def call(self, role, unit, payload, system, max_tokens=4096, request_timeout=300.):
        if self.fatal: raise llm.AccountFault('backend identity/authentication previously failed')
        led=self.ledger;led.check_llm()
        if unknown_usage_blocks(led): raise budget.BudgetExhausted('unknown usage prevents further budgeted calls')
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False,allow_nan=False)}]
        upper=len(json.dumps(messages,ensure_ascii=False).encode('utf-8'))+2048+max_tokens
        used=led.s['llm_tokens_in']+led.s['llm_tokens_out']+led.s.get('token_reserved_failed_upper',0)
        if used+2*upper>led.s['caps']['max_llm_tokens']: raise budget.BudgetExhausted('insufficient conservative remaining token budget')
        led.s['llm_requests']+=1;number=led.s['llm_requests'];led._save()
        prefix=self.out/('%03d_%s'%(number,role))
        context.write_json(str(prefix)+'_request.json',{'role':role,'unit':unit,'messages':messages,'requested_model':'cpa-grok-4.6','temperature':0,'max_tokens':max_tokens})
        for attempt in range(2):
            led.check_wall()
            if led.s['llm_http_attempts']>=getattr(self,'http_cap',100): raise budget.BudgetExhausted('HTTP cap')
            led.s['llm_http_attempts']+=1;led._save()
            try:
                response=self.api.chat.completions.create(model='cpa-grok-4.6',messages=messages,temperature=0,max_tokens=max_tokens,timeout=min(float(request_timeout),led.remaining()))
            except Exception as exc:
                kind=llm.classify_fault(exc)
                # Never log the SDK exception body (it can contain connection credentials).
                context.write_json(str(prefix)+'_attempt%d.json'%attempt,{'failure_kind':kind,'exception_type':type(exc).__name__,'epoch':time.time()})
                # Frozen rule: a failed attempt has unknown usage; it stops every later paid call unless an
                # operator explicitly accepts it (resume --accept-unknown-usage). The reserved amount is an
                # accounting estimate that only tightens the token cap; it is not a guaranteed bound.
                led.s['llm_tokens_unknown']+=1
                led.s.setdefault('llm_failed_attempts',0);led.s['llm_failed_attempts']+=1
                led.s.setdefault('token_reserved_failed_upper',0);led.s['token_reserved_failed_upper']+=upper
                led._save()
                if kind=='ACCOUNT_OR_PERMISSION_FAULT': self.fatal=True;raise llm.AccountFault(kind) from None
                if kind=='TRANSPORT_TRANSIENT_FAULT' and attempt==0: continue
                raise llm.TransportFault(kind) from None
            context.write_json(str(prefix)+'_response.json',response.model_dump(mode='json'))
            usage=response.usage
            pt=getattr(usage,'prompt_tokens',None);ct=getattr(usage,'completion_tokens',None)
            if pt is None or ct is None: led.s['llm_tokens_unknown']+=1
            else: led.s['llm_tokens_in']+=pt;led.s['llm_tokens_out']+=ct
            led.s['events'].append({'kind':'llm_finished','role':role,'unit':unit,'request':number,'returned_model':response.model,'epoch':time.time()});led._save()
            if response.model!='grok-4.6-build': self.fatal=True;raise llm.AccountFault('returned model mismatch')
            text=response.choices[0].message.content if response.choices else ''
            print('LLM',role,unit,'request',number,'usage',pt,ct,flush=True)
            if not isinstance(text,str): raise ValueError('no response text')
            self.last_text=text
            return json.loads(text)
        raise AssertionError('unreachable')

    def fast(self, unit):
        return lambda payload:self.call('fast',unit,payload,FAST_SYSTEM)
