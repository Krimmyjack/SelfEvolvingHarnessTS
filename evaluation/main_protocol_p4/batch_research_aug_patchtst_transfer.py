"""Frozen augmentation-program transfer from the offline-skill MLP study to official PatchTST.
No experiment LLM. Parent packages and reference source are read-only. All writes are opt-in here.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / '_scratch/dev_aug_patchtst_transfer'
PARENT = REPO / '_scratch/dev_aug_offline_skill'
SCREEN = REPO / '_scratch/dev_aug_task_family_screen'
REF = ROOT / 'ref/PatchTST/PatchTST_supervised'
MODULE = 'evaluation.main_protocol_p4.batch_research_aug_patchtst_transfer'
SEEDS = [20269181, 20269182, 20269183]
DOMAINS = ['D01', 'D02', 'D03']
ARMS = ['None', 'NoMix', 'F0', 'F_naive', 'F_shared', 'F_domain']
ARCH = dict(enc_in=1, seq_len=192, pred_len=48, e_layers=3, n_heads=16,
            d_model=128, d_ff=256, dropout=0.2, fc_dropout=0.2, head_dropout=0.,
            individual=False, patch_len=16, stride=8, padding_patch='end', revin=1,
            affine=0, subtract_last=0, decomposition=0, kernel_size=25)
MARKS = [250, 500, 1000, 2000]
LRS = [0.0001, 0.001]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp_%d_%d' % (os.getpid(), threading.get_ident()))
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, p)


def concurrency():
    p=ROOT/'resources.json'
    return int(read(p)['numeric_workers']) if p.exists() else 1


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def make_plan():
    fp = read(PARENT / 'frozen_config.json')
    ev = read(PARENT / 'test/evaluation.json')
    nv = read(PARENT / 'naive/evaluation.json')
    test = sorted(c for d in DOMAINS for c in fp['cases']['test'][d])
    source = [f'{d}_SCR_A{a}_G1' for d in DOMAINS for a in (1, 2)]
    choices, bindings = {}, {}
    for c in test:
        choices[c] = {'None': 'None', 'NoMix': 'P_NoMixRecipe'}
        reg = read(PARENT / 'common' / c / 'aug_materials/index.json')
        bindings[c] = {}
        for arm in ARMS[2:]:
            key = {'F0': 'f0', 'F_naive': 'f_naive', 'F_shared': 'f_shared', 'F_domain': 'f_domain'}[arm]
            cm = (nv if arm == 'F_naive' else ev)['commits'][c + '__' + key]
            choices[c][arm] = cm['physical_material'] if cm else 'None'
            if cm and reg[choices[c][arm]]['key'] != cm['material_key']:
                raise RuntimeError('parent commit/material binding mismatch: ' + c + arm)
        for mat in set(choices[c].values()):
            r = reg[mat]
            if r.get('alias_of'):
                raise RuntimeError('expected physical material: ' + mat)
            if mat != 'None' and not (PARENT / 'common' / c / 'aug_materials' / (mat + '.npz')).exists():
                raise RuntimeError('missing parent material')
            bindings[c][mat] = {'key': r['key'], 'kind': r['kind']}
    jobs = [dict(case=c, material=m, seed=s) for c in test for m in sorted(set(choices[c].values())) for s in SEEDS]
    return dict(package='DEV-AUG-PATCHTST-TRANSFER', created_local=stamp(),
                identity='FROZEN_MLP_DERIVED_PROGRAM_TRANSFER_ON_EXPOSED_DEVELOPMENT_TEST',
                architecture=ARCH, seeds=SEEDS, source_cases=source, test_cases=test, arms=ARMS,
                choices=choices, material_bindings=bindings, test_jobs=jobs,
                calibration=dict(material='None', lrs=LRS, checkpoints=MARKS, seed=SEEDS[0],
                                 objective='equal-domain mean C_A+C_B nMSE on six Source cases; exact tie: fewer updates then lower lr'),
                training=dict(optimizer='AdamW', weight_decay=0.0001, parent_batch=64, parent_child_loss=[0.5, 0.5],
                              dtype='float32', tf32=False, deterministic=True,
                              batch_stream='unchanged train.batch_indices prefix', inference='raw serving input, official RevIN retained'),
                budget=dict(test_fits=len(jobs), calibration_fits=12, wiring_fits=3, fit_cap=len(jobs)+15,
                            llm=0, gpu_concurrency=2, per_fit_timeout_s=1800, wall_cap_hours=16),
                reference=dict(repo='https://github.com/yuqinie98/PatchTST', commit='204c21e',
                               config_source='PatchTST_supervised/scripts/PatchTST/electricity.sh', license='Apache-2.0'),
                limitations=['No new Fast/Slow decision or Consumer-aware Skill learning.',
                             'Controlled task adaptation, not official long-horizon benchmark reproduction.',
                             'Naive parent incomplete deliveries remain None, matching the main table.'])


def setup():
    os.environ.pop('KMP_DUPLICATE_LIB_OK', None)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from methods.ttha.batch_base import readiness as rd
    rd._init_openmp_before_torch()
    os.environ.pop('KMP_DUPLICATE_LIB_OK', None)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('authorized local CUDA device unavailable')
    return torch


def model_new(torch, seed):
    import importlib.util
    if str(REF) not in sys.path:
        sys.path.insert(0, str(REF))
    sp = importlib.util.spec_from_file_location('official_patchtst_model', REF / 'models/PatchTST.py')
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    torch.manual_seed(seed)
    return mod.Model(argparse.Namespace(**ARCH)).to('cuda')


def cases():
    from methods.ttha.batch_base import entity_case as ec
    return ec.cases_from_split(read(PARENT / 'split_copy.json'))


def parent_dir(c):
    p = PARENT / 'common' / c
    return p if (p / 'scaler.npz').exists() else SCREEN / 'cases' / c


def prepare_case(c):
    import numpy as np
    from methods.ttha.batch_base import entity_case as ec
    cs = cases()[c]
    ctx = ec.open_case(cs, ROOT / 'cases', 'material')
    with np.load(parent_dir(c) / 'scaler.npz') as sc:
        if not all(np.array_equal(getattr(ctx.scaler, k), sc[k]) for k in ['mean', 'std', 'scale']):
            raise RuntimeError('parent scaler changed')
        if list(sc['roster']) != list(cs.roster) or int(sc['t']) != cs.t:
            raise RuntimeError('parent roster/cut changed')
    np.savez(ROOT / 'cases' / c / 'parents.npz', Xp=ctx.parents.X_norm.reshape(-1, 192).astype('float32'),
             yp=ctx.parents.y_norm.reshape(-1, 48).astype('float32'))
    write(ROOT / 'cases' / c / 'prepared.json', dict(case=c, status='OK', rows=list(cs.material_rows),
          pool=ctx.parents.X_norm.shape[0]*ctx.parents.X_norm.shape[1], parent_scaler_equal=True))
    print('PREPARED', c, flush=True)


def val_inputs(c, block):
    import numpy as np
    from methods.ttha.batch_base import entity_case as ec, data
    cs = cases()[c]
    if block == 'source_validation':
        if cs.role != 'source':
            raise RuntimeError('validation selection may only read Source')
        origins = cs.c_a + cs.c_b
    elif block == 'e':
        if not (ROOT / 'all_test_models_frozen.json').exists():
            raise RuntimeError('global model freeze missing before E scoring')
        origins = cs.e
    else:
        raise ValueError(block)
    with np.load(ROOT / 'cases' / c / 'scaler.npz') as z:
        sc = data.Scaler(mean=z['mean'].copy(), std=z['std'].copy(), scale=z['scale'].copy(), floor_hits=0)
        mase = z['mase'].copy()
    sl = ec.load_case_slice(cs, min(origins)-192, max(origins)+48)
    wins = data.build_windows(sl, list(origins), sc, with_truth=True)
    return (np.stack([w.X_norm for w in wins], axis=1).astype('float32'),
            np.stack([w.y_true_raw for w in wins], axis=1), sc, mase, list(origins))


def predict(torch, model, x):
    import numpy as np
    model.eval()
    shape = x.shape[:-1]
    z = torch.as_tensor(x.reshape(-1, 192), device='cuda')
    with torch.no_grad():
        parts = [model(z[k:k+64, :, None]).squeeze(-1).cpu().numpy() for k in range(0, len(z), 64)]
    return np.concatenate(parts).reshape(*shape, 48)


def fit_job(j):
    import numpy as np
    torch = setup()
    from methods.ttha.batch_base import train
    c, mat, seed, lr, steps = j['case'], j['material'], j['seed'], j['lr'], j['steps']
    out = ROOT / j['output']; out.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / 'cases' / c / 'parents.npz') as z:
        xp = torch.as_tensor(z['Xp'].copy(), device='cuda'); yp = torch.as_tensor(z['yp'].copy(), device='cuda')
    child = None
    if mat != 'None':
        mp = parent_dir(c) / 'aug_materials' / (mat + '.npz')
        if not mp.exists() and c in read(ROOT/'plan.json')['source_cases']:
            mp = SCREEN / 'cases' / c / 'aug_materials' / (mat + '.npz')
        with np.load(mp) as z:
            if z['Xc'].shape != tuple(xp.shape) or z['yc'].shape != tuple(yp.shape):
                raise RuntimeError('child-parent shape mismatch')
            if not np.isfinite(z['Xc']).all() or not np.isfinite(z['yc']).all():
                raise RuntimeError('nonfinite frozen material')
            child = (torch.as_tensor(z['Xc'].copy(), device='cuda'), torch.as_tensor(z['yc'].copy(), device='cuda'))
    idxs = torch.as_tensor(train.batch_indices(seed, n_pool=len(xp), n_updates=steps), device='cuda')
    model = model_new(torch, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    validation = val_inputs(c, 'source_validation') if j.get('calibration') else None
    curve, scores = [], {}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start = time.time()
    for step in range(1, steps+1):
        model.train(); idx = idxs[step-1]; opt.zero_grad(set_to_none=True)
        loss_p = ((model(xp[idx, :, None]).squeeze(-1)-yp[idx])**2).mean()
        loss = loss_p if child is None else 0.5*loss_p + 0.5*((model(child[0][idx, :, None]).squeeze(-1)-child[1][idx])**2).mean()
        loss.backward(); opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            value = float(loss.detach())
            if not np.isfinite(value):
                raise RuntimeError('nonfinite training loss')
            curve.append([step, value])
            write(out/'progress.json', dict(case=c, material=mat, seed=seed, step=step, steps=steps, elapsed_s=time.time()-start))
        if validation is not None and step in MARKS:
            x,y,sc,mase,origins = validation
            pn = predict(torch, model, x)
            v = ((pn - ((y-sc.mean[:,None,None])/sc.scale[:,None,None]))**2).mean(axis=(0,2))
            scores[str(step)] = dict(nmse=float(v.mean()), per_origin=v.tolist(), origins=origins)
    torch.cuda.synchronize()
    elapsed = time.time()-start
    # Save via temp replace so a partial write is never a completed cell.
    tmp = out/'model.tmp.pt'; torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, tmp)
    os.replace(tmp, out/'model.pt')
    rec = dict(status='OK', job=j, parameters=sum(p.numel() for p in model.parameters()), train_seconds=elapsed,
               peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20, loss_curve=curve, validation=scores,
               torch=str(torch.__version__), device=torch.cuda.get_device_name(0), completed_local=stamp())
    write(out/'fit.json', rec)
    print('FIT_OK', j['output'], 'seconds', round(elapsed,1), 'peak_MB', round(rec['peak_cuda_mb']), flush=True)


def score_case(c):
    import numpy as np
    torch = setup()
    from methods.ttha.batch_base import entity_case as ec
    plan = read(ROOT/'plan.json')
    x,y,sc,mase,origins = val_inputs(c, 'e')
    out = {}
    for m in sorted(set(plan['choices'][c].values())):
        out[m] = {}
        for seed in SEEDS:
            p = ROOT/'test'/c/m/str(seed)
            rec = read(p/'fit.json')
            if rec['status'] != 'OK':
                raise RuntimeError('incomplete fit')
            model = model_new(torch, seed)
            model.load_state_dict(torch.load(p/'model.pt', map_location='cuda', weights_only=True))
            pr = predict(torch, model, x)*sc.scale[:,None,None] + sc.mean[:,None,None]
            s = ec.score_with_status(pr, y, sc, mase)
            if s['status'] != 'SCORABLE':
                raise RuntimeError('non-scorable result')
            out[m][str(seed)] = dict(nmse=s['normalized_mse_macro'], mae=s['mae_raw_macro'],
                                    per_entity=s['per_entity_normalized_mse'],
                                    per_origin=(((pr-y)/sc.scale[:,None,None])**2).mean(axis=(0,2)).tolist())
            np.savez(p/'pred_e.npz', pred_raw=pr, origins=origins)
            del model
    write(ROOT/'scores'/(c+'.json'), dict(case=c, scores=out, origins=origins, scored_local=stamp()))
    print('SCORE_OK', c, flush=True)


def smoke():
    import numpy as np
    torch = setup()
    model = model_new(torch, SEEDS[0]); model.eval()
    x = torch.arange(192, device='cuda', dtype=torch.float32)[None,:,None].repeat(2,1,1)/192
    with torch.no_grad():
        a = model(x); b = model(x)
    assert tuple(a.shape) == (2,48,1) and torch.isfinite(a).all() and torch.equal(a,b)
    torch.manual_seed(SEEDS[0]); model.train(); loss=model(x).square().mean(); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model2=model_new(torch,SEEDS[0]); model2.eval()
    with torch.no_grad(): assert torch.equal(a,model2(x))
    from methods.ttha.batch_base import train
    assert np.array_equal(train.batch_indices(SEEDS[0], 6928,250),train.batch_indices(SEEDS[0],6928,2000)[:250])
    assert not (ROOT/'all_test_models_frozen.json').exists()
    try:
        val_inputs(read(ROOT/'plan.json')['test_cases'][0], 'e')
    except RuntimeError as e:
        assert 'global model freeze' in str(e)
    else: raise AssertionError('E barrier did not reject')
    write(ROOT/'smoke.json', dict(status='PASS', checks=['official shape/finiteness','eval repeat','finite gradients',
          'seeded initialization','batch prefix','E before barrier rejected'], torch=str(torch.__version__),
          parameters=sum(p.numel() for p in model.parameters()), cuda_free_mb=torch.cuda.mem_get_info()[0]/2**20))
    print('SMOKE_PASS', flush=True)


def summarize():
    import numpy as np
    from evaluation.main_protocol_p4.batch_research_aug_offline_skill import boot_ci
    plan=read(ROOT/'plan.json'); pc={}
    for c in plan['test_cases']:
        r=read(ROOT/'scores'/(c+'.json'))['scores']
        nm={a:[r[plan['choices'][c][a]][str(s)]['nmse'] for s in SEEDS] for a in ARMS}
        base=statistics.fmean(nm['None'])
        pc[c]=dict(domain=c[:3], choices=plan['choices'][c], nmse=nm,
                   gain={a:100*(base-statistics.fmean(nm[a]))/base for a in ARMS})
        for a in ARMS:
            pc[c]['domain_vs_'+a]=100*(statistics.fmean(nm[a])-statistics.fmean(nm['F_domain']))/base
    table={}
    for a in ARMS:
        ds={d:statistics.fmean(pc[c]['gain'][a] for c in pc if c.startswith(d)) for d in DOMAINS}
        mse=statistics.fmean(statistics.fmean(statistics.fmean(pc[c]['nmse'][a]) for c in pc if c.startswith(d)) for d in DOMAINS)
        table[a]=dict(gain_pp=statistics.fmean(ds.values()), domains=ds, nmse=mse)
    comparisons={}
    for a in ARMS:
        if a=='F_domain':continue
        key='domain_vs_'+a
        ds={d:statistics.fmean(pc[c][key] for c in pc if c.startswith(d)) for d in DOMAINS}
        comparisons[a]=dict(gain_pp=statistics.fmean(ds.values()),domains=ds,ci95=boot_ci(pc,key),
              win_same_loss=[sum(pc[c][key]>1e-10 for c in pc),sum(abs(pc[c][key])<=1e-10 for c in pc),sum(pc[c][key]<-1e-10 for c in pc)])
    result=dict(status='COMPLETE',identity=plan['identity'],consumer=read(ROOT/'consumer_frozen.json'),
                table=table, comparisons_F_domain=comparisons, per_case=pc)
    write(ROOT/'result.json',result)
    lines=['# PatchTST：冻结增强方案跨 Consumer 迁移','',
           '同三域、40 案例、三 seed；不重新调用 Fast/Slow。按 Source 校准一次 Consumer，所有臂共用；MLP 派生方案保持不变。',
           '本结果检验方案跨模型迁移，不代表 Skill 针对 PatchTST 重新学习，也不是标准长期预测榜复现。','',
           '| 方法 | nMSE（三域等权） | pp of PatchTST None | 电力 | 交通 | 太阳能 |','|---|---:|---:|---:|---:|---:|']
    for a,r in table.items():lines.append('| %s | %.4f | %+.2f | %+.2f | %+.2f | %+.2f |'%(a,r['nmse'],r['gain_pp'],*[r['domains'][d] for d in DOMAINS]))
    lines+=['','域卡方案相对各对照（pp；与父包相同的时期×实体组聚类区间）：','']
    for a,r in comparisons.items():lines.append('- 对 %s：%+.2f，95%% [%+.2f, %+.2f]；胜/同/负 %s。'%(a,r['gain_pp'],r['ci95']['lo95'],r['ci95']['hi95'],r['win_same_loss']))
    lines+=['','训练步数/学习率：见 consumer_frozen.json；成本与全部启动/完成事件见 ledger.json。',
            '没有新增 LLM 学习/部署调用，因此不能将历史 token 节省重复作为本轮实测成本。',
            'NoMix 和朴素卡均保留，不按结果移除对照。所有未成功单元必须保留，不按部分案例生成完成结论。']
    led=read(ROOT/'ledger.json')
    lines+=['', '运行：%d 次成功拟合 / %d 次拟合尝试；0 实验 LLM。'%(led['fits_ok'],led['fit_attempts']),
            'Source 校准：lr=%g，updates=%d；所有 Test 臂共用。'%(result['consumer']['lr'],result['consumer']['steps'])]
    if (ROOT/'instrument_recovery_1.json').exists():
        lines+=['Source 进程两次中断，第二次退出码0xC000013A（控制台中断，发起者未知）；修复子进程无窗口启动后原配置恢复，失败工件与两次各1次额度修订保留。测试未用于修订。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'tables.md').write_text('\n'.join(lines[5:])+'\n',encoding='utf-8')


def run_sub(args, log):
    env=dict(os.environ);env.pop('KMP_DUPLICATE_LIB_OK',None)
    env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with Path(log).open('w',encoding='utf-8') as f:
        p=subprocess.run([sys.executable,'-B','-u','-m',MODULE]+list(args),cwd=REPO,env=env,
                         stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=1800)
    write(Path(log).with_suffix('.exit.json'),dict(returncode=p.returncode,finished_local=stamp(),args=list(args)))
    if p.returncode:raise RuntimeError('worker failed rc=%s: %s'%(p.returncode,log))


class Run:
    def __init__(self):
        self.lock=threading.Lock();self.start=time.time()
        self.ledger=read(ROOT/'ledger.json') if (ROOT/'ledger.json').exists() else dict(events=[],fit_attempts=0,fits_ok=0,started_local=stamp(),started_epoch=self.start)
    def event(self,kind,**kw):
        with self.lock:
            if kind=='fit_start':
                extra=read(ROOT/'instrument_recovery_1.json')['additional_fit_attempts'] if (ROOT/'instrument_recovery_1.json').exists() else 0
                extra+=read(ROOT/'instrument_recovery_2.json')['additional_fit_attempts'] if (ROOT/'instrument_recovery_2.json').exists() else 0
                if self.ledger['fit_attempts']>=read(ROOT/'plan.json')['budget']['fit_cap']+extra:raise RuntimeError('fit cap reached')
                self.ledger['fit_attempts']+=1
            if kind=='fit_ok':self.ledger['fits_ok']+=1
            self.ledger['events'].append(dict(kind=kind,time=stamp(),**kw));write(ROOT/'ledger.json',self.ledger)
    def fit(self,j):
        path=ROOT/j['output']/'fit.json'
        if path.exists():
            old=read(path)
            if old['status']!='OK' or old['job']!=j:raise RuntimeError('cached fit binding mismatch')
            return old
        if time.time()-self.ledger['started_epoch']>16*3600:raise RuntimeError('16h wall cap')
        self.event('fit_start',job=j)
        jp=ROOT/j['output']/'job.json';write(jp,j)
        run_sub(['--fit',str(jp)],ROOT/j['output']/'worker.log')
        r=read(path);self.event('fit_ok',output=j['output'],seconds=r['train_seconds'])
        print('PROGRESS',self.ledger['fits_ok'],'fits',j['output'],flush=True)
        return r
    def fits(self,jobs,n=None):
        n=concurrency() if n is None else n
        # At most n submitted workers; a failure stops further dispatch after in-flight work exits.
        out=[]
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            it=iter(jobs); pending={ex.submit(self.fit,j):j for j in list(next(it,None) for _ in range(n)) if j is not None}
            while pending:
                done,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
                for f in done:
                    pending.pop(f);out.append(f.result())
                for _ in done:
                    j=next(it,None)
                    if j is not None:pending[ex.submit(self.fit,j)]=j
        return out


def run_all():
    import numpy as np
    import msvcrt
    ROOT.mkdir(parents=True,exist_ok=True)
    lockfile=(ROOT/'controller.lock').open('a+b');lockfile.seek(0)
    try:msvcrt.locking(lockfile.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('another controller owns the package lock')
    run=Run()
    try:
        if not (ROOT/'plan.json').exists():write(ROOT/'plan.json',make_plan())
        plan=read(ROOT/'plan.json')
        write(ROOT/'status.json',dict(status='RUNNING',phase='smoke',pid=os.getpid(),updated=stamp()))
        if not (ROOT/'smoke.json').exists():run_sub(['--smoke'],ROOT/'logs/smoke.log')
        with cf.ThreadPoolExecutor(max_workers=concurrency()) as ex:
            fs=[ex.submit(run_sub,['--prepare',c],ROOT/'logs'/('prepare_'+c+'.log')) for c in plan['source_cases']+plan['test_cases'] if not (ROOT/'cases'/c/'prepared.json').exists()]
            for f in fs:f.result()
        # Real deterministic rerun, including a child material; only Source rows, no outcomes.
        if not (ROOT/'wiring.json').exists():
            cs=plan['source_cases'][0]
            wire=[dict(case=cs,material=m,seed=SEEDS[0],lr=0.0001,steps=100,calibration=False,output='wiring/'+n)
                  for m,n in [('None','none'),('P_NoMixRecipe','child_a'),('P_NoMixRecipe','child_b')]]
            recs=run.fits(wire,n=1)
            run_sub(['--check-wiring'],ROOT/'logs/check_wiring.log')
        if not (ROOT/'consumer_frozen.json').exists():
            write(ROOT/'status.json',dict(status='RUNNING',phase='source_calibration',pid=os.getpid(),updated=stamp()))
            jobs=[dict(case=c,material='None',seed=SEEDS[0],lr=lr,steps=2000,calibration=True,output=f'calibration/{c}/lr{lr}') for c in plan['source_cases'] for lr in LRS]
            recs=run.fits(jobs)
            values=[]
            for lr in LRS:
                for step in MARKS:
                    vs={d:statistics.fmean(r['validation'][str(step)]['nmse'] for r in recs if r['job']['case'].startswith(d) and r['job']['lr']==lr) for d in DOMAINS}
                    values.append(dict(lr=lr,steps=step,objective=statistics.fmean(vs.values()),domains=vs))
            chosen=min(values,key=lambda r:(r['objective'],r['steps'],r['lr']))
            write(ROOT/'consumer_frozen.json',dict(**chosen,architecture=ARCH,all_source_options=values,frozen_local=stamp(),
                  note='Calibration uses None only, Source C_A+C_B only, never Test performance.'))
        frozen=read(ROOT/'consumer_frozen.json')
        timing=[read(q)['train_seconds'] for q in (ROOT/'calibration').glob('*/lr*/fit.json')]
        estimate=len(plan['test_jobs'])*statistics.fmean(timing)*(frozen['steps']/2000)/concurrency()/3600
        write(ROOT/'cost_estimate.json',dict(test_hours_at_current_concurrency=estimate,numeric_workers=concurrency(),source_measured_fit_seconds=timing,selected_steps=frozen['steps'],note='Source None time; child paths have two forwards and can cost up to roughly twice this estimate.'))
        jobs=[dict(**j,lr=frozen['lr'],steps=frozen['steps'],calibration=False,output=f"test/{j['case']}/{j['material']}/{j['seed']}") for j in plan['test_jobs']]
        write(ROOT/'status.json',dict(status='RUNNING',phase='test_training',pid=os.getpid(),updated=stamp(),chosen=frozen,physical_test_fits=len(jobs)))
        run.fits(jobs)
        write(ROOT/'all_test_models_frozen.json',dict(frozen_local=stamp(),models=len(jobs),consumer=frozen,choices_file='plan.json'))
        write(ROOT/'status.json',dict(status='RUNNING',phase='external_E_scoring',pid=os.getpid(),updated=stamp()))
        with cf.ThreadPoolExecutor(max_workers=concurrency()) as ex:
            futures=[ex.submit(run_sub,['--score',c],ROOT/'logs'/('score_'+c+'.log')) for c in plan['test_cases'] if not (ROOT/'scores'/(c+'.json')).exists()]
            for f in futures:f.result()
        summarize()
        write(ROOT/'status.json',dict(status='COMPLETE',finished_local=stamp(),pid=os.getpid(),fits=run.ledger['fits_ok'],wall_seconds=time.time()-run.ledger['started_epoch']))
        print('COMPLETE',flush=True)
    except BaseException as exc:
        write(ROOT/'status.json',dict(status='STOPPED_ERROR',error=str(exc),traceback=traceback.format_exc(),pid=os.getpid(),updated=stamp()))
        raise
    finally:
        lockfile.seek(0);msvcrt.locking(lockfile.fileno(),msvcrt.LK_UNLCK,1);lockfile.close()


def check_wiring():
    torch=setup()
    a=torch.load(ROOT/'wiring/child_a/model.pt',weights_only=True,map_location='cpu')
    b=torch.load(ROOT/'wiring/child_b/model.pt',weights_only=True,map_location='cpu')
    assert all(torch.equal(a[k],b[k]) for k in a), 'real child rerun must be bitwise equal'
    n=torch.load(ROOT/'wiring/none/model.pt',weights_only=True,map_location='cpu')
    param_names=set(dict(model_new(torch,SEEDS[0]).named_parameters()))
    delta=sum(float((a[k]-n[k]).float().square().sum()) for k in param_names)
    assert delta>0,'child training path did not change the model'
    rs=[read(ROOT/'wiring'/x/'fit.json') for x in ['none','child_a','child_b']]
    write(ROOT/'wiring.json',dict(status='PASS',real_child_rerun_bitwise_equal=True,child_vs_none_weight_sq_difference=delta,
          seconds=[r['train_seconds'] for r in rs],peak_cuda_mb=max(r['peak_cuda_mb'] for r in rs),
          estimated_2000_step_567_fits_hours=statistics.fmean(r['train_seconds'] for r in rs)*20*567/3600,
          resource_note='single-worker extrapolation; current concurrency and selected steps reported separately'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',action='store_true');ap.add_argument('--run',action='store_true')
    ap.add_argument('--prepare');ap.add_argument('--fit');ap.add_argument('--score');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--check-wiring',action='store_true');ap.add_argument('--tables',action='store_true')
    a=ap.parse_args()
    if a.plan:
        if (ROOT/'plan.json').exists():raise RuntimeError('plan already frozen')
        write(ROOT/'plan.json',make_plan()); print('PLAN',len(read(ROOT/'plan.json')['test_jobs']))
    elif a.run:run_all()
    elif a.prepare:prepare_case(a.prepare)
    elif a.fit:fit_job(read(a.fit))
    elif a.score:score_case(a.score)
    elif a.smoke:smoke()
    elif a.check_wiring:check_wiring()
    elif a.tables:summarize()
