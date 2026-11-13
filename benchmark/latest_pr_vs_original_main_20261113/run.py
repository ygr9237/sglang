"""Run isolated latest-PR versus original-main benchmarks on host 1."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

TOOLS=Path('/tmp/codex-sglang-pr5-20261112-run')
ROOT=TOOLS/'pr-final-main-20261113'
RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
state={'running':True,'comparison':'original upstream main vs latest PR head only','steps':[]}
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SGLANG_VIT_ENABLE_CUDA_GRAPH='0')
env.pop('COVERAGE_PROCESS_START',None)
python=str(TOOLS/'venv/bin/python')
def save():(RESULTS/'status.json').write_text(json.dumps(state,indent=2)+'\n')
def run(name,command,source,devices,timeout=900):
    step=dict(name=name,command=command,devices=devices,started=time.time());state['steps'].append(step);save()
    run_env=dict(env,PYTHONPATH=str(source/'python'),CUDA_VISIBLE_DEVICES=devices,SGLANG_CACHE_DIR=str(ROOT/name/'cache'),TORCH_EXTENSIONS_DIR=str(ROOT/name/'extensions'))
    with (RESULTS/(name+'.log')).open('w') as out:r=subprocess.run(command,cwd=source,env=run_env,stdout=out,stderr=subprocess.STDOUT,timeout=timeout)
    step.update(exit_code=r.returncode,elapsed_s=time.time()-step['started']);save();assert r.returncode==0,name
try:
    save()
    vision=TOOLS/'pr49342-20261113/source';pp=TOOLS/'pr49116-20261113/source'
    state['latest_source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [vision/'python/sglang/srt/managers/mm_schedule.py',vision/'python/sglang/srt/models/kimi_k3.py',vision/'python/sglang/srt/multimodal/mm_utils.py',pp/'python/sglang/srt/managers/scheduler_pp_mixin.py']}
    def vision_job():run('vision',[python,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(ROOT/'vision_main.py'),'latest-vs-main'],vision,'0,1')
    def pp_job():
        run('pp-sender',[python,str(ROOT/'pp_main.py'),'sender'],pp,'6')
        run('pp-transport',[python,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(ROOT/'pp_main.py'),'transport'],pp,'6,7')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(vision_job),pool.submit(pp_job)]
        for job in jobs:job.result()
    rows=[json.loads((RESULTS/f'vision-rank{r}.json').read_text()) for r in range(2)]
    assert all(len(r)==72 for r in rows)
    comparisons=[]
    for n in [1,10,20]:
        for cache in ['cold','hot']:
            selected=[r for r in rows[0] if r['images']==n and r['cache']==cache]
            assert len({r['output_sha256'] for rr in rows for r in rr if r['images']==n and r['cache']==cache})==1
            times={v:statistics.median([x for r in selected if r['version']==v for x in r['samples_ms']]) for v in ['main','current']}
            comparisons.append(dict(images=n,cache=cache,original_main_ms=times['main'],latest_pr_ms=times['current'],latency_change_percent=(times['current']/times['main']-1)*100,bit_identical=True))
    for name in ['sender','transport']:
        p=ROOT/f'main-{name}-results.json';assert json.loads(p.read_text())['complete']
        (RESULTS/p.name).write_bytes(p.read_bytes())
    state.update(successful=True,vision=comparisons)
    assert state['latest_source_hashes']=={str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in state['latest_source_hashes']}
except BaseException as error:state.update(successful=False,error=str(error))
finally:state['running']=False;save()
