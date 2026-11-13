"""Validate the receive-allocation fix in an isolated PR checkout on host 1."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

TOOLS = Path('/tmp/codex-sglang-pr5-20261112-run')
ROOT = TOOLS / 'pr49342-20261113'
SOURCE = ROOT / 'source'
RESULTS = ROOT / 'results'
BEFORE = '5a6684f877e3a3759e9049342bf9af5bf46a2638'
RUNTIME = 'python/sglang/srt/multimodal/mm_utils.py'
OWNERSHIP = 'test/registered/unit/multimodal/test_kimi_k3_pcp_vision_dp.py'
CONSENSUS = 'test/registered/unit/multimodal/test_kimi_k3_image_dp_consensus.py'
CHANGED = [RUNTIME, OWNERSHIP, CONSENSUS]
PYTHON = str(TOOLS / 'venv/bin/python')
RESULTS.mkdir(exist_ok=True)
state = {'running': True, 'before': BEFORE, 'steps': []}
env = os.environ.copy()
env.pop('SKIP', None)
env.update(PYTHONPATH=str(SOURCE / 'python'), CUDA_VISIBLE_DEVICES='99', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', SGLANG_TEST_MAX_RETRY='0', SGLANG_CACHE_DIR=str(ROOT / 'cache'), TORCH_EXTENSIONS_DIR=str(ROOT / 'torch_extensions'), PRE_COMMIT_HOME=str(TOOLS / 'precommit-cache'), RUSTUP_HOME=str(TOOLS / 'rustup'), CARGO_HOME=str(TOOLS / 'cargo-home'), CARGO_TARGET_DIR=str(TOOLS / 'cargo-target'), CARGO_BUILD_JOBS='4', COVERAGE_FILE=str(RESULTS / '.coverage'))
env['PATH'] = os.pathsep.join([str(TOOLS / 'venv/bin'), str(TOOLS / 'cargo-home/bin'), str(TOOLS / 'node/bin'), '/root/.cargo/bin', env['PATH']])

def save():
    (RESULTS / 'status.json').write_text(json.dumps(state, indent=2) + '\n')

def run(name, command, *, cwd=SOURCE, devices='99', timeout=900, expected=0):
    step = {'name': name, 'command': command, 'devices': devices, 'started': time.time()}
    state['steps'].append(step)
    save()
    with (RESULTS / (name + '.log')).open('w') as stream:
        result = subprocess.run(command, cwd=cwd, env=dict(env, CUDA_VISIBLE_DEVICES=devices), stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    step.update(exit_code=result.returncode, elapsed_s=round(time.time()-step['started'], 3))
    save()
    if result.returncode != expected:
        raise RuntimeError(f'{name}: expected {expected}, got {result.returncode}')

try:
    save()
    if not SOURCE.exists():
        run('clone', ['git','clone','--no-local','--no-checkout',str(ROOT/'seed.git'),str(SOURCE)], cwd=ROOT)
        run('checkout', ['git','checkout','-b','codex/pr49342-validation',BEFORE])
        run('apply-check', ['git','apply','--check',str(ROOT/'candidate.patch')])
        run('apply', ['git','apply',str(ROOT/'candidate.patch')])
    state['sha256']={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in CHANGED}
    run('environment', [PYTHON,'-c',"import torch,sys,transformers,json; print(json.dumps({'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__}))"])
    candidate=(SOURCE/RUNTIME).read_bytes()
    original=subprocess.check_output(['git','show',BEFORE+':'+RUNTIME],cwd=SOURCE)
    try:
        (SOURCE/RUNTIME).write_bytes(original)
        run('negative-control', [PYTHON,CONSENSUS,'-v'], timeout=120, expected=1)
        assert 'receive_allocation' in (RESULTS/'negative-control.log').read_text(), 'Negative control failed outside the receive-allocation case'
    finally:
        (SOURCE/RUNTIME).write_bytes(candidate)
    run('ownership', [PYTHON,OWNERSHIP,'-v'])
    run('consensus', [PYTHON,CONSENSUS,'-v'])
    related=[OWNERSHIP,CONSENSUS,'test/registered/unit/multimodal/test_mrope_encoder_utils.py','test/registered/unit/models/test_kimi_k25.py','test/registered/unit/models/test_kimi_vl.py','test/registered/unit/models/test_kimi_k3_vision.py']
    run('cpu-coverage', [PYTHON,'-m','coverage','run','--rcfile='+str(ROOT/'coverage.ini'),'-m','pytest',*related,'-v'])
    try:
        (SOURCE/RUNTIME).write_bytes(original)
        run('gpu-before', [PYTHON,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(ROOT/'gpu_bench.py'),'before'], devices='0,1', timeout=600)
    finally:
        (SOURCE/RUNTIME).write_bytes(candidate)
    run('gpu-after', [PYTHON,'-m','torch.distributed.run','--standalone','--nproc-per-node=2','--module','coverage','run','--rcfile='+str(ROOT/'coverage.ini'),'-p',str(ROOT/'gpu_bench.py'),'after'], devices='0,1', timeout=600)
    run('coverage-combine', [PYTHON,'-m','coverage','combine','--rcfile='+str(ROOT/'coverage.ini')])
    run('coverage-xml', [PYTHON,'-m','coverage','xml','--rcfile='+str(ROOT/'coverage.ini'),'-o',str(RESULTS/'coverage.xml')])
    run('fix-diff-coverage', [str(TOOLS/'venv/bin/diff-cover'),str(RESULTS/'coverage.xml'),'--compare-branch=HEAD','--fail-under=60','--json-report='+str(RESULTS/'fix-diff-coverage.json')])
    run('pr-diff-coverage', [str(TOOLS/'venv/bin/diff-cover'),str(RESULTS/'coverage.xml'),'--compare-branch=704681229f1cdda291dbd2261dabb2ae80a4a49d','--fail-under=60','--json-report='+str(RESULTS/'pr-diff-coverage.json')])
    run('precommit-all-files', [str(TOOLS/'venv/bin/pre-commit'),'run','--all-files','--show-diff-on-failure'])
    run('diff-check', ['git','diff','--check','HEAD'])
    run('final-status', ['git','status','--short'])
    assert state['sha256']=={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in CHANGED}, 'Checks modified candidate source'
    state['successful']=True
except BaseException as exc:
    state.update(successful=False,error=str(exc))
finally:
    state['running']=False
    save()
