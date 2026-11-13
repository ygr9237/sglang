import dataclasses,json
from types import SimpleNamespace as NS
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.runtime_context import get_context
c=MambaComponent.__new__(MambaComponent)
c.cache=NS(hicache_mamba_l3_snapshot_interval=16,enable_mamba_extra_buffer=True,is_swa_enabled=False)
c.checkpoints_level=2
req=NS(origin_input_ids=list(range(8)),mamba_branching_seqlen=None,kv=NS(mamba_last_track_seqlen=4))
p=InsertParams()
with get_context().override_server_args(disaggregation_mode='null'):
 length=c.prepare_for_caching_req(req,p,8,False)
assert length==0
encoded=dataclasses.asdict(p)
assert encoded['mamba_storage_prompt_boundary'] and encoded['mamba_checkpoint_deferred']
print(json.dumps({'checkpoint_deferred':encoded['mamba_checkpoint_deferred'],'storage_prompt_boundary':encoded['mamba_storage_prompt_boundary'],'checkpoint_length':length,'serialization_retains_both_flags':True}))
