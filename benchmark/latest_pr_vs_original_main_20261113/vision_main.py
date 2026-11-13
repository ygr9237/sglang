"""Original upstream main versus the current PR, actual weights and native groups."""
from pathlib import Path
import __future__
import ast
import gc
import importlib.util
import sys
import os

rank=int(os.environ["LOCAL_RANK"])
desired=set(range(8,12)) if rank==0 else set(range(104,108))
if desired<=os.sched_getaffinity(0):os.sched_setaffinity(0,desired)

ROOT=Path(__file__).resolve().parent
exec(compile((ROOT/'gpu_setup.py').read_text().split('rows = []')[0],str(ROOT/'gpu_setup.py'),'exec'),globals())
import sglang.srt.models.kimi_k3 as kimi
import sglang.srt.multimodal.mm_utils as helpers

def function(path,name,scope,classname=None):
    nodes=ast.parse(path.read_text()).body
    if classname is not None:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==classname).body
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),scope,namespace)
    return namespace[name]

spec=importlib.util.spec_from_file_location('original_main_mm_schedule',ROOT/'vision-original/mm_schedule.py')
original_schedule=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=original_schedule
spec.loader.exec_module(original_schedule)
original_method=function(ROOT/'vision-original/kimi_k3.py','get_image_feature',vars(kimi),'KimiK3ForConditionalGeneration')
class OriginalImageModel(torch.nn.Module):
    get_image_feature=original_method
original_model=OriginalImageModel()
original_model.use_data_parallel=True
original_model.vision_tower=tower
original_model.mm_projector=projector
original_dp=function(ROOT/'vision-original/mm_utils.py','run_dp_sharded_mrope_vision_model',vars(helpers))
variants={'main':(original_model,original_schedule,original_dp),'current':(model,mm_schedule,helpers.run_dp_sharded_mrope_vision_model)}
assert not hasattr(original_model,'get_image_dp_group')
rows=[]
gc.disable()
try:
    with torch.inference_mode(),get_parallel().override(attn_tp_size=1,attn_tp_rank=0,attn_cp_size=2,attn_cp_rank=RANK,attn_dp_size=1,tp_group=group):
        for n in [1,10,20]:
            items=items_for(n)
            for cache in ['cold','hot']:
                for repeat in range(6):
                    order=['main','current'] if repeat%2==0 else ['current','main']
                    for version in order:
                        model,mm_schedule,helpers.run_dp_sharded_mrope_vision_model=variants[version]
                        mm_schedule.embedding_cache=MultiModalStaticCache(0 if cache=='cold' else 4*1024**3)
                        output=call(items)
                        digest=hashlib.sha256(output.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                        del output
                        for _ in range(6):call(items)
                        samples=[]
                        for _ in range(12):
                            dist.barrier();torch.cuda.synchronize();begin=time.perf_counter()
                            output=call(items);torch.cuda.synchronize()
                            value=torch.tensor((time.perf_counter()-begin)*1000,device='cuda')
                            dist.all_reduce(value,op=dist.ReduceOp.MAX)
                            samples.append(value.item());del output
                        row=dict(rank=RANK,images=n,cache=cache,repeat=repeat,order=order,version=version,median_ms=statistics.median(samples),samples_ms=samples,output_sha256=digest)
                        rows.append(row)
                        print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
                assert len({r['output_sha256'] for r in rows if r['images']==n and r['cache']==cache})==1
            del items
            for _,schedule,_ in variants.values():schedule.embedding_cache.clear()
            torch.cuda.empty_cache()
finally:
    helpers.run_dp_sharded_mrope_vision_model=variants['current'][2]
    gc.enable()
(ROOT/'results'/f'vision-rank{RANK}.json').write_text(json.dumps(rows,indent=2)+'\n')
group.destroy();dist.destroy_process_group()
