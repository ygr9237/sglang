"""Actual BF16 weights, native GroupCoordinator, TP2/CP2, before/after fix."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors import safe_open

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers import mm_schedule
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.models.kimi_k3_vl import KimiK3MultiModalProjector, KimiK3VisionTower
from sglang.srt.multimodal.image_dp_consensus import ImageDPCoordinationError
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_parallel

ROOT = Path(__file__).resolve().parent
MODE = sys.argv[1]
RANK = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(RANK)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=120))
group = GroupCoordinator(
    group_ranks=[[0, 1]], local_rank=RANK, torch_distributed_backend='nccl',
    use_pynccl=True, use_mscclpp=False, use_custom_allreduce=False,
    use_torch_symm_mem_all_reduce=False, use_hpu_communicator=False,
    use_xpu_communicator=False, group_name='image-dp-validation',
    gloo_timeout=datetime.timedelta(seconds=120),
)
model_path = Path('/modules/Kimi-K3.1-Pro')
config = json.loads((model_path/'config.json').read_text())['vision_config']
config['hidden_size'] = config['vt_hidden_size']
config = SimpleNamespace(**config)
with torch.device('cuda'):
    tower = KimiK3VisionTower(config).to(device=torch.device('cuda', RANK), dtype=torch.bfloat16).eval()
    projector = KimiK3MultiModalProjector(config).to(device=torch.device('cuda', RANK), dtype=torch.bfloat16).eval()
mapping = json.loads((model_path/'model.safetensors.index.json').read_text())['weight_map']
for prefix, module in [('vision_tower.', tower), ('mm_projector.', projector)]:
    state = {}
    for shard in sorted({v for k, v in mapping.items() if k.startswith(prefix)}):
        with safe_open(model_path/shard, framework='pt', device='cpu') as reader:
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

def items_for(n, seed=20261110):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    return [mm_schedule.MultimodalDataItem(
        modality=Modality.IMAGE, hash=i+1, offsets=[(i*1369,(i+1)*1369-1)],
        feature=torch.randn((5476,3,14,14),device='cuda',dtype=torch.bfloat16,generator=gen),
        model_specific_data={'grid_thws':torch.tensor([[1,74,74]])},
    ) for i in range(n)]

def call(items):
    req = mm_schedule.PerImageRequestInfo(0,items,[(i*1369,(i+1)*1369-1) for i in range(len(items))],0,len(items)*1369)
    result = mm_schedule._batch_encode_per_image_misses(model.get_image_feature,[req],torch.device('cuda',RANK))
    return torch.cat([result[(item.cache_key,1369)] for item in items])

rows = []
with torch.inference_mode(), get_parallel().override(attn_tp_size=1,attn_tp_rank=0,attn_cp_size=2,attn_cp_rank=RANK,attn_dp_size=1,tp_group=group):
    for n in [1,10,20]:
        items = items_for(n)
        for cache in ['cold','hot']:
            mm_schedule.embedding_cache = MultiModalStaticCache(0 if cache=='cold' else 4*1024**3)
            output = call(items)
            peer = output.clone()
            dist.broadcast(peer,0)
            assert torch.equal(output,peer)
            digest = hashlib.sha256(output.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
            del output,peer
            for _ in range(6):
                call(items)
            samples=[]
            for _ in range(12):
                dist.barrier()
                torch.cuda.synchronize()
                started=time.perf_counter()
                output=call(items)
                torch.cuda.synchronize()
                elapsed=(time.perf_counter()-started)*1000
                sample=torch.tensor(elapsed,device='cuda')
                dist.all_reduce(sample,op=dist.ReduceOp.MAX)
                samples.append(sample.item())
                del output
            row=dict(mode=MODE,images=n,cache=cache,rank=RANK,median_ms=statistics.median(samples),samples_ms=samples,output_sha256=digest)
            rows.append(row)
            print(json.dumps(row),flush=True)
        del items
        mm_schedule.embedding_cache.clear()
        torch.cuda.empty_cache()
    if MODE=='after':
        items=items_for(2,seed=17)
        mm_schedule.embedding_cache=MultiModalStaticCache(0)
        reference=call(items)
        original_empty=torch.empty
        injected=False
        def fail_receive(*size,**kwargs):
            global injected
            if RANK==0 and len(size)==1 and isinstance(size[0],tuple) and len(size[0])==3 and size[0][0]==2738:
                injected=True
                raise torch.OutOfMemoryError('injected GPU receive-buffer allocation failure')
            return original_empty(*size,**kwargs)
        caught=False
        started=time.perf_counter()
        with patch('torch.empty',side_effect=fail_receive):
            try:
                call(items)
            except ImageDPCoordinationError as exc:
                caught=True
                if RANK==0:
                    assert isinstance(exc.__cause__,torch.OutOfMemoryError)
        assert caught and (injected or RANK!=0)
        torch.cuda.synchronize()
        recovered=call(items)
        assert torch.equal(reference,recovered)
        peer=recovered.clone()
        dist.broadcast(peer,0)
        assert torch.equal(recovered,peer)
        rows.append(dict(case='GPU receive allocation failure',rank=RANK,injected=injected,coordinated_error=caught,next_encoding_ok=True,bit_identical_recovery=True,elapsed_s=time.perf_counter()-started))
(ROOT/'results'/f'{MODE}-gpu-rank{RANK}.json').write_text(json.dumps(rows,indent=2)+'\n')
group.destroy()
dist.destroy_process_group()
