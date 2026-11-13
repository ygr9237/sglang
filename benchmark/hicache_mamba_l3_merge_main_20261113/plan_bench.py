"""CPU cost of storage-plan generation, not a serving or storage-I/O benchmark."""
import gc
import json
import statistics
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).parent / 'source'
sys.path.insert(0, str(SOURCE/'test/registered/unit/mem_cache'))
from test_mamba_l3_snapshot_interval import _backup, _cache, _insert

rows = []
for backend in ['python', 'rust']:
    for depth in [1, 64, 256]:
        cache = _cache(backend, interval=32768)
        for i in range(depth):
            node = _insert(cache, (i+1)*128, slot=i+1)
            _backup(cache, node)
        core = cache.tree_core
        timings = {0: [], 32768: []}
        for interval in timings:
            for _ in range(16):
                core.build_storage_backup_spec(node, False, interval)
        gc.disable()
        try:
            for round_id in range(6):
                for interval in ([0,32768] if round_id%2==0 else [32768,0]):
                    start = time.perf_counter_ns()
                    for _ in range(200):
                        core.build_storage_backup_spec(node, False, interval)
                    timings[interval].append((time.perf_counter_ns()-start)/200/1000)
        finally:
            gc.enable()
        row = dict(backend=backend, depth=depth, node_end_tokens=depth*128,
                   median_us={str(k):statistics.median(v) for k,v in timings.items()},
                   rounds_us={str(k):v for k,v in timings.items()},
                   core=type(core).__name__)
        rows.append(row)
        print(json.dumps(row),flush=True)
(Path(__file__).parent/'plan-bench.json').write_text(json.dumps(rows,indent=2))
