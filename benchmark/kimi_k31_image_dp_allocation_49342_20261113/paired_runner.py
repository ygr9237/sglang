"""Run and summarize the supplementary balanced GPU benchmark."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

TOOLS=Path('/tmp/codex-sglang-pr5-20261112-run')
ROOT=TOOLS/'pr49342-20261113'
SOURCE=ROOT/'source'
RESULTS=ROOT/'results'
env=os.environ.copy()
env.update(PYTHONPATH=str(SOURCE/'python'),CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SGLANG_CACHE_DIR=str(ROOT/'cache'),TORCH_EXTENSIONS_DIR=str(ROOT/'torch_extensions'))
env.pop('COVERAGE_PROCESS_START',None)
command=[str(TOOLS/'venv/bin/python'),'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(ROOT/'paired_bench.py'),'paired']
record=dict(running=True,command=command,coverage_instrumentation=False,gc_enabled=False,repeats=6,warmups=6,samples_per_repeat=12,started=time.time())
path=RESULTS/'paired-status.json'
path.write_text(json.dumps(record,indent=2)+'\n')
try:
    with (RESULTS/'paired-gpu.log').open('w') as stream:
        result=subprocess.run(command,cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
    record.update(exit_code=result.returncode,elapsed_s=time.time()-record['started'])
    assert result.returncode==0
    rows=[json.loads((RESULTS/f'paired-gpu-rank{rank}.json').read_text()) for rank in range(2)]
    assert all(len(r)==72 for r in rows)
    summary=[]
    for images in [1,10,20]:
        for cache in ['cold','hot']:
            subset=[r for r in rows[0] if r['images']==images and r['cache']==cache]
            assert len({r['output_sha256'] for rank in rows for r in rank if r['images']==images and r['cache']==cache})==1
            values={v:statistics.median([s for r in subset if r['version']==v for s in r['samples_ms']]) for v in ['before','after']}
            medians={v:[r['median_ms'] for r in subset if r['version']==v] for v in ['before','after']}
            summary.append(dict(images=images,cache=cache,before_ms=values['before'],after_ms=values['after'],change_percent=(values['after']/values['before']-1)*100,round_medians=medians,bit_identical=True))
    record.update(successful=True,comparisons=summary)
except BaseException as error:
    record.update(successful=False,error=str(error))
finally:
    record['running']=False
    path.write_text(json.dumps(record,indent=2)+'\n')
