import os,json,datetime,time,statistics,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from safetensors import safe_open
from sglang.srt.managers import mm_schedule
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.models.kimi_k3_vl import KimiK3MultiModalProjector,KimiK3VisionTower
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_parallel
RANK=int(os.environ['LOCAL_RANK']);MODE=os.environ['CHECK_MODE'];torch.cuda.set_device(RANK)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120))
cpu_group=dist.new_group(backend='gloo')
class Group:
 world_size=2
 rank_in_group=RANK
 def __init__(self):self.cpu_group=cpu_group
 def broadcast(self,tensor,src):dist.broadcast(tensor,src)
 def all_gather(self,tensor,dim):
  outputs=[torch.empty_like(tensor) for _ in range(2)]
  dist.all_gather(outputs,tensor.contiguous());return torch.cat(outputs,dim)
model_path = Path('/modules/Kimi-K3.1-Pro')
cfg = json.loads((model_path / 'config.json').read_text())['vision_config']
cfg['hidden_size'] = cfg['vt_hidden_size']
cfg = SimpleNamespace(**cfg)
with torch.device('cuda'):
    tower = KimiK3VisionTower(cfg).to(device=torch.device('cuda', RANK), dtype=torch.bfloat16).eval()
    projector = KimiK3MultiModalProjector(cfg).to(device=torch.device('cuda', RANK), dtype=torch.bfloat16).eval()
mapping = json.loads((model_path / 'model.safetensors.index.json').read_text())['weight_map']
for prefix, module in [('vision_tower.', tower), ('mm_projector.', projector)]:
    state = {}
    for shard in sorted({v for k, v in mapping.items() if k.startswith(prefix)}):
        with safe_open(model_path / shard, framework='pt', device='cpu') as reader:
            for key in reader.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = reader.get_tensor(key)
    module.load_state_dict(state, strict=True)
del state
model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
torch.nn.Module.__init__(model)
model.use_data_parallel = True
model.vision_tower = tower
model.mm_projector = projector

rows=[]
def call(items):
 req=mm_schedule.PerImageRequestInfo(0,items,[(i*1369,(i+1)*1369-1) for i in range(len(items))],0,len(items)*1369)
 result=mm_schedule._batch_encode_per_image_misses(model.get_image_feature,[req],torch.device('cuda',RANK))
 return torch.cat([result[(getattr(item,"cache_key",item.hash),1369)] for item in items])
with torch.inference_mode(),get_parallel().override(attn_tp_size=1,attn_tp_rank=0,attn_cp_size=2,attn_cp_rank=RANK,attn_dp_size=1,tp_group=Group()):
 for n in [1,10,20]:
  gen=torch.Generator(device='cuda').manual_seed(20261110)
  items=[mm_schedule.MultimodalDataItem(modality=Modality.IMAGE,hash=i+1,offsets=[(i*1369,(i+1)*1369-1)],feature=torch.randn((5476,3,14,14),device='cuda',dtype=torch.bfloat16,generator=gen),model_specific_data={'grid_thws':torch.tensor([[1,74,74]])}) for i in range(n)]
  for state in ['cold','hot']:
   mm_schedule.embedding_cache=MultiModalStaticCache(0 if state=='cold' else 4*1024**3)
   output=call(items)
   peer=output.clone();dist.broadcast(peer,0);assert torch.equal(output,peer)
   digest=hashlib.sha256(output.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
   del output,peer
   for _ in range(6):call(items)
   samples=[]
   for _ in range(12):
    dist.barrier();torch.cuda.synchronize();begin=time.perf_counter()
    output=call(items);torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)*1000
    sample=torch.tensor(elapsed,device='cuda');dist.all_reduce(sample,op=dist.ReduceOp.MAX);samples.append(sample.item());del output
   row=dict(mode=MODE,images=n,cache=state,rank=RANK,median_ms=statistics.median(samples),samples_ms=samples,output_sha256=digest)
   rows.append(row);print(json.dumps(row),flush=True)
  del items;mm_schedule.embedding_cache.clear();torch.cuda.empty_cache()
 if MODE=='after':
  from sglang.srt.multimodal.image_dp_consensus import ImageDPCoordinationError
  gen=torch.Generator(device='cuda').manual_seed(17)
  items=[mm_schedule.MultimodalDataItem(modality=Modality.IMAGE,hash=100+i,offsets=[(i*1369,(i+1)*1369-1)],feature=torch.randn((5476,3,14,14),device='cuda',dtype=torch.bfloat16,generator=gen),model_specific_data={'grid_thws':torch.tensor([[1,74,74]])}) for i in range(2)]
  mm_schedule.embedding_cache=MultiModalStaticCache(0)
  def fail(_module,_args):
   if RANK==0:raise MemoryError('injected GPU-encoder Python failure')
  hook=tower.register_forward_pre_hook(fail);caught=False
  try:call(items)
  except ImageDPCoordinationError:caught=True
  finally:hook.remove()
  assert caught;torch.cuda.synchronize();recovered=call(items)
  peer=recovered.clone();dist.broadcast(peer,0);assert torch.equal(recovered,peer)
  rows.append(dict(case='GPU injected owner error',coordinated_error=caught,next_encoding_ok=True,rank=RANK))
Path('/audit_results',MODE+'-gpu-rank'+str(RANK)+'.json').write_text(json.dumps(rows,indent=2))
dist.destroy_process_group()
