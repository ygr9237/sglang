"""Real-weight image-DP check. No model-server or production-source changes."""
import datetime
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors import safe_open

import sglang
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.models.kimi_k3_vl import KimiK3MultiModalProjector, KimiK3VisionTower
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_parallel

STAGE = Path(__file__).parent
MODE = os.environ['CHECK_MODE']
RANK = int(os.environ['LOCAL_RANK'])
assert str(STAGE / MODE / 'python') in sglang.__file__, sglang.__file__
torch.cuda.set_device(RANK)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=300))


class Group:
    world_size = 2
    rank_in_group = RANK

    def broadcast(self, tensor, src):
        dist.broadcast(tensor, src)

    def all_gather(self, tensor, dim):
        shards = [torch.empty_like(tensor) for _ in range(2)]
        dist.all_gather(shards, tensor.contiguous())
        return torch.cat(shards, dim)


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
calls = []


def record(_module, args, kwargs):
    grids = kwargs.get('grid_thw_list')
    if grids is None:
        grids = (args[1] if len(args) > 1 else kwargs['grid_hw']).cpu().tolist()
    calls.extend(grids)


hook = tower.register_forward_pre_hook(record, with_kwargs=True)
cases = {
    'single_1k': [(1, 74, 74)],
    'two_1k': [(1, 74, 74)] * 2,
    'uneven_three': [(1, 146, 146), (1, 74, 74), (1, 56, 56)],
    'ten_1k': [(1, 74, 74)] * 10,
    'twenty_1k': [(1, 74, 74)] * 20,
    'temporal_pool': [(2, 20, 20), (1, 22, 22), (2, 16, 16)],
}
rows = []
with torch.inference_mode(), get_parallel().override(
    attn_tp_size=1, attn_tp_rank=0, attn_cp_size=2, attn_cp_rank=RANK,
    attn_dp_size=1, tp_group=Group(),
):
    for name, grids in cases.items():
        generator = torch.Generator(device='cuda').manual_seed(20261109)
        items = [MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.randn((t*h*w, 3, 14, 14), device='cuda', dtype=torch.bfloat16, generator=generator),
            model_specific_data={'grid_thws': torch.tensor([[t,h,w]])},
        ) for t,h,w in grids]
        calls.clear()
        output = model.get_image_feature(items)
        owners = [None, None]
        dist.all_gather_object(owners, list(calls))
        count = sum(map(len, owners))
        assert count == len(grids) * (1 if MODE == 'pr2-main' else 2), (name, owners)
        peer = output.clone()
        dist.broadcast(peer, 0)
        torch.testing.assert_close(output, peer, atol=0, rtol=0)
        if RANK == 0:
            torch.save(output.cpu(), STAGE / f'{MODE}-{name}.pt')
        for _ in range(4):
            model.get_image_feature(items)
        samples = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(12):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = model.get_image_feature(items)
            torch.cuda.synchronize()
            elapsed = torch.tensor((time.perf_counter()-start)*1000, device='cuda')
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples.append(elapsed.item())
            del result
        row = dict(case=name, mode=MODE, rank=RANK, shape=list(output.shape), owners=owners,
                   median_ms=statistics.median(samples), samples_ms=samples,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   output_sha256=hashlib.sha256(output.cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
        rows.append(row)
        print(json.dumps(row), flush=True)
        del output, peer, items
        torch.cuda.empty_cache()
(STAGE / f'{MODE}-gpu-rank{RANK}.json').write_text(json.dumps(rows, indent=2))
hook.remove()
dist.barrier()
dist.destroy_process_group()
