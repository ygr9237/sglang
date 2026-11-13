"""Collect uninstrumented GPU timing and compare every before/after digest."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

TOOLS=Path('/tmp/codex-sglang-pr5-20261112-run')
ROOT=TOOLS/'pr49342-20261113'
RESULTS=ROOT/'results'
SOURCE=ROOT/'source'
state=json.loads((RESULTS/'status.json').read_text())
assert not state['running'] and state['successful'], state
for name,digest in state['sha256'].items():
    assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest
for rank in range(2):
    shutil.copyfile(RESULTS/f'after-gpu-rank{rank}.json',RESULTS/f'coverage-after-gpu-rank{rank}.json')
env=os.environ.copy()
env.update(PYTHONPATH=str(SOURCE/'python'),CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SGLANG_CACHE_DIR=str(ROOT/'cache'),TORCH_EXTENSIONS_DIR=str(ROOT/'torch_extensions'))
env.pop('COVERAGE_PROCESS_START',None)
command=[str(TOOLS/'venv/bin/python'),'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(ROOT/'gpu_bench.py'),'after']
record=dict(running=True,command=command,devices='0,1',coverage_instrumentation=False,started=time.time())
path=RESULTS/'gpu-uninstrumented-status.json'
path.write_text(json.dumps(record,indent=2)+'\n')
try:
    with (RESULTS/'gpu-after-uninstrumented.log').open('w') as stream:
        result=subprocess.run(command,cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
    record.update(exit_code=result.returncode,elapsed_s=time.time()-record['started'])
    assert result.returncode==0
    comparisons=[]
    for rank in range(2):
        before=json.loads((RESULTS/f'before-gpu-rank{rank}.json').read_text())
        after=json.loads((RESULTS/f'after-gpu-rank{rank}.json').read_text())
        for old,new in zip(before,after):
            assert old['images']==new['images'] and old['cache']==new['cache']
            assert old['output_sha256']==new['output_sha256']
            comparisons.append(dict(rank=rank,images=old['images'],cache=old['cache'],before_ms=old['median_ms'],after_ms=new['median_ms'],change_percent=(new['median_ms']/old['median_ms']-1)*100,bit_identical=True,output_sha256=new['output_sha256']))
        failure=after[-1]
        assert failure['coordinated_error'] and failure['next_encoding_ok'] and failure['bit_identical_recovery']
    record.update(successful=True,all_outputs_bit_identical=True,comparisons=comparisons)
except BaseException as error:
    record.update(successful=False,error=str(error))
finally:
    record['running']=False
    path.write_text(json.dumps(record,indent=2)+'\n')
