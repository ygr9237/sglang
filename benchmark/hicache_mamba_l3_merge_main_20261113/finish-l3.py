import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
R=Path('/tmp/codex-sglang-pr5-20261112-run/pr49518-conflict-20261113');S=R/'source';T=R.parent;O=R/'results';O.mkdir(exist_ok=True)
env=os.environ.copy();env.pop('SKIP',None);env.update(PYTHONPATH=str(S/'python'),CUDA_VISIBLE_DEVICES='99',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SGLANG_CACHE_DIR=str(R/'cache'),TORCH_EXTENSIONS_DIR=str(R/'torch_extensions'),SGLANG_RUST_BUILD_MODE='auto',CARGO_HOME=str(T/'cargo-home'),RUSTUP_HOME=str(T/'rustup'),RUSTUP_TOOLCHAIN='1.92',CARGO_TARGET_DIR=str(T/'cargo-target'),CARGO_BUILD_JOBS='4',CARGO_NET_OFFLINE='true',PRE_COMMIT_HOME=str(T/'precommit-cache'))
env['PATH']=os.pathsep.join([str(T/'venv/bin'),str(T/'cargo-home/bin'),str(T/'node/bin'),env['PATH']]);PY=str(T/'venv/bin/python');files=json.loads((R/'changed-files.json').read_text());state=json.loads((O/'status-before-storage-stop.json').read_text());state.update(running=True,successful=False);state.pop('error',None);state['steps']=[x for x in state['steps'] if x['rc']==0]
def save():(O/'status.json').write_text(json.dumps(state,indent=2))
def run(n,cmd,gpu='99',expected=0):
 t=time.monotonic()
 with (O/(n+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=S,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 state['steps'].append(dict(name=n,rc=p.returncode,seconds=round(time.monotonic()-t,2)));save()
 if expected is not None:assert p.returncode==expected,(n,p.returncode)
 return p.returncode
try:
 save()
 run('l3-roundtrip',[PY,str(R/'l3_roundtrip.py')],gpu='0');run('plan-bench',[PY,str(R/'plan_bench.py')])
 run('diff-check',['git','diff','--check'])
 assert state['sha256']=={n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in files}
 with tarfile.open(R/'tested-source.tar.gz','w:gz') as t:
  for n in files:t.add(S/n,arcname=n)
 state['successful']=True
except Exception as exc:state.update(successful=False,error=str(exc))
finally:state['running']=False;save()
