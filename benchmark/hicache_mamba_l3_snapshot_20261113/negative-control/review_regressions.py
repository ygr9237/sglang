from pathlib import Path
import hashlib,json,sys
import pytest

SOURCE=Path('/data/k31_pr4_l3_completion_20261110/source')
sys.path.insert(0,str(SOURCE/'test/registered/unit/mem_cache'))
from test_mamba_l3_snapshot_interval import _cache,_insert,_backup,_has_mamba,_key,_write_and_ack,MatchPrefixParams,PoolName

@pytest.mark.parametrize('backend',['python','rust'])
def test_pending_periodic_checkpoint_survives_kv_split_and_real_ack(backend):
    cache=_cache(backend,interval=16,page_size=2)
    node=_insert(cache,20)
    _backup(cache,node,pending=True)
    cache.ongoing_write_through={91:(node,None,[node])}
    assert _has_mamba(cache.tree_core.build_storage_backup_spec(node,False,16))
    match=cache.tree_core.match_prefix(MatchPrefixParams(key=_key(list(range(18)))))
    # Apply the real cache's split bookkeeping, then its actual L2 ACK handler.
    for action in match.cache_actions:
        if getattr(action,'ack_id',None)==91:
            cache._replace_pending_write_through_node(
                action.ack_id,action.old_node_id,[action.new_node_id,action.new_child_node_id])
    cache._finish_write_through_ack(91)
    calls=cache.cache_controller.write_storage.call_args_list
    assert sum(len(call.args[0]) for call in calls)==20
    kept=sum(any(x.name==PoolName.MAMBA for x in call.kwargs['extra_pools'] or ()) for call in calls)
    print(json.dumps({'backend':backend,'case':'split_pending','kv_write_lengths':[len(c.args[0]) for c in calls],'state_writes':kept}),flush=True)
    assert kept==1,'KV-only split moved the crossing onto a node with no Mamba state'

@pytest.mark.parametrize('backend',['python','rust'])
def test_promotion_of_already_selected_periodic_node_does_not_rewrite(backend):
    cache=_cache(backend,interval=16,page_size=2)
    node=_insert(cache,16)
    _backup(cache,node)
    first=_write_and_ack(cache,node)
    assert any(x.name==PoolName.MAMBA for x in first.kwargs['extra_pools'])
    assert cache.cache_controller.write_storage.call_count==1
    _insert(cache,16,slot=2,endpoint=True)
    calls=cache.cache_controller.write_storage.call_args_list
    print(json.dumps({'backend':backend,'case':'periodic_promotion','kv_write_lengths':[len(c.args[0]) for c in calls],'write_calls':len(calls)}),flush=True)
    assert len(calls)==1,'Periodic checkpoint was already included in the first L3 write'

if __name__=='__main__':
    expected=json.loads(Path('/data/pr49518_review_20261113/head-file-hashes.json').read_text())
    for rel,row in expected.items():
        assert hashlib.sha256((SOURCE/rel).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==row['sha256'],rel
    print(json.dumps({'verified_pr_head_files':len(expected),'head':'dbf1e025fdd52da542c24ec16cfc127998016c0b'}),flush=True)
    raise SystemExit(pytest.main(['--noconftest','-o','addopts=','-q','-s',str(SOURCE/'test/registered/unit/mem_cache/test_mamba_l3_snapshot_interval.py'),__file__]))
