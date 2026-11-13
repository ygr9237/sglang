import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/pr49518_fix_20261113');SOURCE=ROOT/'source'
ENV=os.environ.copy();ENV.update(PYTHONPATH=str(SOURCE/'python'),SGLANG_CACHE_DIR=str(ROOT/'cache'),SGLANG_RUST_BUILD_MODE='auto',RUSTUP_TOOLCHAIN='1.90',CUDA_VISIBLE_DEVICES='99',OMP_NUM_THREADS='1',CARGO_BUILD_JOBS='4')
status={'running':True,'steps':[]}
files=json.loads((ROOT/'changed-files.json').read_text())
status['sha256']={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in files}
def run(name,cmd,devices='99',timeout=900):
 t=time.monotonic()
 with (ROOT/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=SOURCE,env=dict(ENV,CUDA_VISIBLE_DEVICES=devices),stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 status['steps'].append(dict(name=name,rc=r.returncode,seconds=round(time.monotonic()-t,2)));(ROOT/'final-status.json').write_text(json.dumps(status,indent=2));assert r.returncode==0,name
try:
 (ROOT/'final-status.json').write_text(json.dumps(status,indent=2))
 run('new-tests',['python','test/registered/unit/mem_cache/test_mamba_l3_snapshot_interval.py'])
 run('regressions',['python','-m','pytest','-q','test/registered/unit/mem_cache/test_rust_tree_core_integration.py','test/registered/unit/mem_cache/test_tree_core_registry.py','test/registered/unit/mem_cache/test_mamba_path_state_cap.py','-k','mamba or storage_backup or backup_storage or hicache_write_through or registry or default_backend'])
 run('plan-bench',['python',str(ROOT/'plan_bench.py')])
 run('l3-roundtrip',['python',str(ROOT/'l3_roundtrip.py')],devices='0',timeout=900)
except Exception as exc:status['error']=str(exc)
finally:
 status['running']=False;(ROOT/'final-status.json').write_text(json.dumps(status,indent=2))
