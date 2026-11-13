"""Balanced, same-process old/new encoder timing on the actual BF16 checkpoint."""
from pathlib import Path
import ast
import gc
import subprocess

ROOT = Path(__file__).resolve().parent
setup = (ROOT/'gpu_bench.py').read_text().split('rows = []')[0]
exec(compile(setup,str(ROOT/'gpu_bench.py'),'exec'),globals())
import sglang.srt.multimodal.mm_utils as helpers

source=subprocess.check_output(['git','show','5a6684f877e3a3759e9049342bf9af5bf46a2638:python/sglang/srt/multimodal/mm_utils.py'],text=True)
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run_dp_sharded_mrope_vision_model')
namespace={}
exec(compile(ast.Module(body=[node],type_ignores=[]),'published-mm_utils.py','exec'),vars(helpers),namespace)
functions={'before':namespace['run_dp_sharded_mrope_vision_model'],'after':helpers.run_dp_sharded_mrope_vision_model}
rows=[]
gc.disable()
try:
    with torch.inference_mode(),get_parallel().override(attn_tp_size=1,attn_tp_rank=0,attn_cp_size=2,attn_cp_rank=RANK,attn_dp_size=1,tp_group=group):
        for n in [1,10,20]:
            items=items_for(n)
            for cache in ['cold','hot']:
                for repeat in range(6):
                    order=['before','after'] if repeat%2==0 else ['after','before']
                    for version in order:
                        helpers.run_dp_sharded_mrope_vision_model=functions[version]
                        mm_schedule.embedding_cache=MultiModalStaticCache(0 if cache=='cold' else 4*1024**3)
                        output=call(items)
                        digest=hashlib.sha256(output.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                        del output
                        for _ in range(6):call(items)
                        samples=[]
                        for _ in range(12):
                            dist.barrier()
                            torch.cuda.synchronize()
                            begin=time.perf_counter()
                            output=call(items)
                            torch.cuda.synchronize()
                            value=torch.tensor((time.perf_counter()-begin)*1000,device='cuda')
                            dist.all_reduce(value,op=dist.ReduceOp.MAX)
                            samples.append(value.item())
                            del output
                        row=dict(rank=RANK,images=n,cache=cache,repeat=repeat,order=order,version=version,median_ms=statistics.median(samples),samples_ms=samples,output_sha256=digest)
                        rows.append(row)
                        print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
                digests={r['output_sha256'] for r in rows if r['images']==n and r['cache']==cache}
                assert len(digests)==1
            del items
            mm_schedule.embedding_cache.clear()
            torch.cuda.empty_cache()
finally:
    helpers.run_dp_sharded_mrope_vision_model=functions['after']
    gc.enable()
(ROOT/'results'/f'paired-gpu-rank{RANK}.json').write_text(json.dumps(rows,indent=2)+'\n')
group.destroy()
dist.destroy_process_group()
