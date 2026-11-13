"""Original upstream main versus current PR; no earlier PR revision is loaded."""
import gc
import json
import os
from pathlib import Path
import sys

import torch
import bench_pp_output_metadata as bench

ROOT=Path(__file__).resolve().parent
mode=sys.argv[1]
rank=int(os.environ.get('LOCAL_RANK',0))
desired=set(range(4)) if rank==0 else set(range(96,100))
if desired<=os.sched_getaffinity(0):os.sched_setaffinity(0,desired)
torch.set_num_threads(1);torch.set_grad_enabled(False);torch.manual_seed(0);torch.cuda.set_device(rank)
functions={arm:bench.load_prepare(arm) for arm in ['base','fixed']}
selected={'no_logprob_b25','python_top20_b25','python_prompt_top20_32k','cpu_rows_b25','cpu_view32k_b1','cpu_view32k_b25','cuda_rows_b25','cuda_view32k_b1'}
if mode=='transport':selected={'no_logprob_b25','python_top20_b25','cpu_rows_b25','cpu_view32k_b1','cpu_view32k_b25'}
all_cases=bench.cases()
bench.cases=lambda:[c for c in all_cases if c['name'] in selected]
original_save=bench.save
bench.save=lambda name,value:original_save('main-'+name,value)
original_summary=bench.summary
bench.summary=lambda values:original_summary(values)|{'samples_us':values}
gc.collect();gc.disable()
if mode=='sender':
    bench.sender(functions)
    probes=[]
    for case in [c for c in all_cases if c['name']=='cuda_view32k_b1']:
        result,batch=bench.make_payload(case)
        for arm,prepare in functions.items():
            wire=prepare(result,batch)
            metadata,_=bench._split_tensor_dict(wire)
            def leaves(value):
                if isinstance(value,torch.Tensor):yield value
                elif isinstance(value,(list,tuple)):
                    for item in value:yield from leaves(item)
                elif isinstance(value,dict):
                    for item in value.values():yield from leaves(item)
            tensors=list(leaves(metadata))
            probes.append(dict(arm=arm,case=case['name'],metadata_devices=[str(t.device) for t in tensors],metadata_storage_bytes=[t.untyped_storage().nbytes() for t in tensors],cpu_ready=all(t.device.type=='cpu' for t in tensors)))
    assert not probes[0]['cpu_ready'] and probes[1]['cpu_ready']
    (ROOT/'results/pp-readiness.json').write_text(json.dumps(probes,indent=2)+'\n')
elif mode=='transport':
    bench.transport(functions)
