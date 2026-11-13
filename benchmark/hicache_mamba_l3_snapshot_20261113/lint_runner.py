import json,os,subprocess,time,hashlib
from pathlib import Path
ROOT=Path('/tmp/codex-sglang-pr5-20261112-run/pr49518-20261113')
SOURCE=ROOT/'source'; TOOLS=ROOT.parent; RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
env=os.environ.copy();env.pop('SKIP',None);env.update(PRE_COMMIT_HOME=str(TOOLS/'precommit-cache'),RUSTUP_HOME=str(TOOLS/'rustup'),CARGO_HOME=str(TOOLS/'cargo-home'),CARGO_TARGET_DIR=str(TOOLS/'cargo-target'),CARGO_BUILD_JOBS='4',RUSTUP_TOOLCHAIN='1.92')
env['PATH']=os.pathsep.join([str(TOOLS/'venv/bin'),str(TOOLS/'cargo-home/bin'),str(TOOLS/'node/bin'),'/root/.cargo/bin',env['PATH']])
files=json.loads((ROOT/'changed-files.json').read_text()); status={'steps':[],'running':True}
def run(name,cmd):
 t=time.monotonic()
 with (RESULTS/(name+'.log')).open('w') as f:rc=subprocess.run(cmd,cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
 status['steps'].append(dict(name=name,rc=rc,seconds=round(time.monotonic()-t,2)));(RESULTS/'status.json').write_text(json.dumps(status,indent=2));return rc
try:
 run('git-init',['git','init','-q']); run('git-config',['git','config','core.autocrlf','false']);run('git-branch',['git','checkout','-b','codex/pr49518-validation']);run('git-add',['git','add','.'])
 rc=run('precommit-initial',[str(TOOLS/'venv/bin/pre-commit'),'run','--files',*files,'--show-diff-on-failure'])
 if rc not in (0,1):raise RuntimeError('pre-commit setup failed')
 rc=run('precommit-final',[str(TOOLS/'venv/bin/pre-commit'),'run','--files',*files,'--show-diff-on-failure']);assert rc==0
 run('rustfmt-check',['cargo','fmt','--manifest-path','rust/sglang-radix-tree/Cargo.toml','--','--check'])
 status['sha256']={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in files}
 with __import__('tarfile').open(ROOT/'formatted-source.tar.gz','w:gz') as t:
  for n in files:t.add(SOURCE/n,arcname=n)
except Exception as exc:status['error']=str(exc)
finally:
 status['running']=False;(RESULTS/'status.json').write_text(json.dumps(status,indent=2))
