"""Repeat GPU and sender measurements sequentially with fixed CPU placement."""
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

TOOLS=Path('/tmp/codex-sglang-pr5-20261112-run');ROOT=TOOLS/'pr-final-main-20261113';RESULTS=ROOT/'results'
state=json.loads((RESULTS/'status.json').read_text());assert state['successful'] and not state['running']
shutil.copytree(RESULTS,ROOT/'concurrent-results',dirs_exist_ok=True)
state.update(running=True,successful=False,primary_vision_run='vision-clean',primary_pp_sender_run='pp-sender-clean',pp_transport_run='pp-transport')
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SGLANG_VIT_ENABLE_CUDA_GRAPH='0');env.pop('COVERAGE_PROCESS_START',None)
python=str(TOOLS/'venv/bin/python')
def save():(RESULTS/'status.json').write_text(json.dumps(state,indent=2)+'\n')
def run(name,command,source,devices):
    step=dict(name=name,command=command,devices=devices,started=time.time());state['steps'].append(step);save()
    run_env=dict(env,PYTHONPATH=str(source/'python'),CUDA_VISIBLE_DEVICES=devices,SGLANG_CACHE_DIR=str(ROOT/name/'cache'),TORCH_EXTENSIONS_DIR=str(ROOT/name/'extensions'))
    with (RESULTS/(name+'.log')).open('w') as out:r=subprocess.run(command,cwd=source,env=run_env,stdout=out,stderr=subprocess.STDOUT,timeout=600)
    step.update(exit_code=r.returncode,elapsed_s=time.time()-step['started']);save();assert r.returncode==0,name
try:
    save()
    run('vision-clean',[python,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(ROOT/'vision_main.py'),'latest-vs-main'],TOOLS/'pr49342-20261113/source','0,1')
    run('pp-sender-clean',[python,str(ROOT/'pp_main.py'),'sender'],TOOLS/'pr49116-20261113/source','6')
    (RESULTS/'main-sender-results.json').write_bytes((ROOT/'main-sender-results.json').read_bytes())
    rows=[json.loads((RESULTS/f'vision-rank{r}.json').read_text()) for r in range(2)]
    assert all(len(r)==72 for r in rows)
    comparisons=[]
    for n in [1,10,20]:
        for cache in ['cold','hot']:
            selected=[r for r in rows[0] if r['images']==n and r['cache']==cache]
            assert len({r['output_sha256'] for rr in rows for r in rr if r['images']==n and r['cache']==cache})==1
            times={v:statistics.median([x for r in selected if r['version']==v for x in r['samples_ms']]) for v in ['main','current']}
            comparisons.append(dict(images=n,cache=cache,original_main_ms=times['main'],latest_pr_ms=times['current'],latency_change_percent=(times['current']/times['main']-1)*100,bit_identical=True))
    state.update(successful=True,vision=comparisons)
except BaseException as error:state.update(successful=False,error=str(error))
finally:state['running']=False;save()
