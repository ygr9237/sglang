"""Two real ranks: cache miss lists and chunk boundaries must agree."""
import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_parallel

rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=30))


class Group:
    world_size = 2
    rank_in_group = rank
    collectives = 0

    def broadcast(self, tensor, src):
        self.collectives += 1
        dist.broadcast(tensor, src)

    def all_gather(self, tensor, dim):
        self.collectives += 1
        outputs = [torch.empty_like(tensor) for _ in range(2)]
        dist.all_gather(outputs, tensor)
        return torch.cat(outputs, dim)


class Tower:
    device = torch.device('cpu')
    merge_kernel_size = (1, 1)
    config = SimpleNamespace(hidden_size=1)
    patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=torch.empty(1)))

    def __call__(self, pixels, **_kwargs):
        return pixels.reshape(-1, 1, 1)


group = Group()
items = [MultimodalDataItem(
    modality=Modality.IMAGE, hash=i+1, offsets=[(4*i,4*i+3)],
    feature=torch.full((4,1),float(i+1)),
) for i in range(3)]
cache = MultiModalStaticCache(4096)
mm_schedule.embedding_cache = cache
request = mm_schedule.PerImageRequestInfo(0, items, [(0,3),(4,7),(8,11)], 0, 12)
miss_lists = []


def encode(misses):
    miss_lists.append([item.hash for item in misses])
    return run_dp_sharded_mrope_vision_model(
        Tower(), torch.cat([item.feature for item in misses]),
        [[1,1,4]]*len(misses), rope_type='rope_2d', parallel_group=group,
    )


def run():
    return mm_schedule._batch_encode_per_image_misses(encode,[request],torch.device('cpu'))


with get_parallel().override(attn_tp_size=1,attn_tp_rank=0):
    cold = run()
    hot = run()
    assert miss_lists == [[1,2,3]]
    for key in cold:
        torch.testing.assert_close(cold[key],hot[key],atol=0,rtol=0)
    cache.free(items[1].cache_key,None)
    partial = run()
    assert miss_lists == [[1,2,3],[2]]
    for key in cold:
        torch.testing.assert_close(cold[key],partial[key],atol=0,rtol=0)
    request.extend_prefix_len=2
    request.extend_seq_len=4
    crossing = run()
    assert len(crossing)==2 and len(miss_lists)==2
    request.extend_prefix_len=4
    request.extend_seq_len=4
    boundary = run()
    assert len(boundary)==1 and len(miss_lists)==2
    request.extend_seq_len=0
    assert run()=={} and len(miss_lists)==2
    cache.free(items[0].cache_key,None)
    request.extend_prefix_len=0
    request.extend_seq_len=2
    assert len(run())==1 and miss_lists==[[1,2,3],[2],[1]]
    assert group.collectives==3
states=[None,None]
row=dict(rank=rank,miss_lists=miss_lists,collectives=group.collectives,checks=7)
dist.all_gather_object(states,row)
assert states[0]['miss_lists']==states[1]['miss_lists']
(Path(__file__).parent/f'gloo-cache-rank{rank}.json').write_text(json.dumps(row,indent=2))
print(json.dumps(row),flush=True)
dist.destroy_process_group()
