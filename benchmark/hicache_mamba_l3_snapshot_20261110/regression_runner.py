import json
import pathlib
import subprocess

p = pathlib.Path(__file__).parent
commands = [
    ('new-tests', ['python', 'test/registered/unit/mem_cache/test_mamba_l3_snapshot_interval.py']),
    ('regressions', ['python', '-m', 'pytest', '-q',
        'test/registered/unit/mem_cache/test_rust_tree_core_integration.py',
        'test/registered/unit/mem_cache/test_tree_core_registry.py',
        'test/registered/unit/mem_cache/test_mamba_path_state_cap.py',
        '-k', 'mamba or storage_backup or backup_storage or hicache_write_through or registry or default_backend']),
]
status = []
for name, cmd in commands:
    with (p/(name+'.log')).open('w') as f:
        r = subprocess.run(cmd, cwd=p/'source', stdout=f, stderr=subprocess.STDOUT, timeout=1800)
    status.append(dict(name=name, rc=r.returncode))
    (p/'regression-status.json').write_text(json.dumps(status))
    if r.returncode:
        break
