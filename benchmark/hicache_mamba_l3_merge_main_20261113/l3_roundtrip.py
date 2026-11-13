"""Isolated Mooncake I/O + real host pools; no model server or production store.

The tree/index allocator fixture is the candidate's registered CPU fixture.
Storage backup, sparse hit query, physical buffers and GPU transfers are real.
This is a component round trip, not a full-model/HTTP accuracy benchmark.
"""
import dataclasses
import argparse
import importlib.util
import importlib.metadata
import json
import os
import pathlib
import queue
import socket
import subprocess
import sys
import time
import threading
import uuid

import torch

from sglang.kernels.ops.attention.fla.kda import fused_recurrent_kda
from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo, MLABufferInfo
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig, PoolName, PoolTransfer, PoolHitPolicy, SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

ROOT = pathlib.Path(__file__).parent
SOURCE = ROOT/'source'
module_spec = importlib.util.spec_from_file_location('snapshot_tests',
    SOURCE/'test/registered/unit/mem_cache/test_mamba_l3_snapshot_interval.py')
fixture = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(fixture)
PAGE = 64
END = 73728
STEP = 8192
CAP = END+2048
STATE_CAP = 32
SHIFT = 1024
SS = 16


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def mla_pool(width=16, page_size=PAGE, capacity=CAP):
    rows = (torch.randn(capacity, 1, width, dtype=torch.bfloat16, device='cuda'),)
    info = MLABufferInfo(buffers=rows, data_ptrs=torch.tensor([x.data_ptr() for x in rows],
        dtype=torch.uint64, device='cuda'), layer_indices=(0,), device_layer_num=1,
        page_size=page_size, kv_lora_rank=width, qk_rope_head_dim=0, kv_cache_dim=width,
        dtype=torch.bfloat16, device=rows[0].device)
    host = MLATokenToKVPoolHost(info, 1.1, 0, page_size, 'page_first',
        device_capacity_tokens=capacity, start_layer=0, end_layer=1, target_layer_num=1)
    return rows, host


def wipe(tensors):
    for x in tensors:
        x.zero_()


def run_case(backend, interval, master_port, results):
    torch.manual_seed(20261110)
    torch.cuda.manual_seed_all(20261110)
    kv, kh = mla_pool()
    draft, dh = mla_pool(8)
    # A real registered state-derived sidecar, independent from draft KV pages.
    side, sh = mla_pool(16, page_size=1, capacity=64)
    temporal = torch.randn(2, STATE_CAP, 2, 128, 128, dtype=torch.bfloat16, device='cuda')
    conv = tuple(torch.randn(2, STATE_CAP, 3, 256, dtype=torch.bfloat16, device='cuda') for _ in range(3))
    mh = MambaPoolHost(MambaStateBufferInfo(temporal=temporal, conv=conv), 2, 0,
        device_capacity=STATE_CAP, layout='page_first')
    pools = {PoolName.KV: kh, PoolName.MAMBA: mh, PoolName.DRAFT: dh,
        PoolName.DRAFT_INDEXER: sh}
    gpu = [*kv, temporal, *conv, *draft, *side]
    host = [kh.kv_buffer, mh.temporal_buffer, *mh.conv_buffer, dh.kv_buffer, sh.kv_buffer]
    expected = [x.clone() for x in gpu]
    cfg = HiCacheStorageConfig(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        attn_cp_rank=0, attn_cp_size=1, is_mla_model=True,
        enable_storage_metrics=False, is_page_first_layout=True,
        model_name='snapshot-roundtrip', extra_config=dict(
            local_hostname='127.0.0.1', metadata_server='P2PHANDSHAKE',
            protocol='tcp', device_name='', master_server_address=f'127.0.0.1:{master_port}',
            global_segment_size=128*1024**2, check_server=False,
            standalone_storage=False, extra_backend_tag=uuid.uuid4().hex))
    store = MooncakeStore(cfg, kh)
    try:
        store.register_mem_pool_host(kh)
        store.registered_pools[PoolName.KV] = kh
        for name, pool in pools.items():
            if name != PoolName.KV:
                store.register_mem_host_pool_v2(pool, name)
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.storage_stop_event = threading.Event()
        cc.storage_config = cfg
        cc.storage_backend = store
        cc.storage_backend_type = 'mooncake'
        cc.page_size = PAGE
        cc.storage_host_pool = kh
        cc.mem_pool_host = HostPoolGroup([
            PoolEntry(name, pool, None, lambda layer: layer,
                is_primary_index_anchor=(name == PoolName.KV))
            for name, pool in pools.items()])
        cc.backup_queue = queue.Queue()
        cc.page_set_func = cc._page_set_zero_copy
        cache = fixture._cache(backend, interval=interval, page_size=PAGE)
        cache.cache_controller = cc
        cache.sidecar_pool_specs = [SidecarPoolSpec(PoolName.DRAFT, PoolName.KV),
            SidecarPoolSpec(PoolName.DRAFT_INDEXER, PoolName.MAMBA, PoolHitPolicy.TRAILING_PAGES)]
        all_hashes = []
        selected = []
        object_count = 0
        for slot, end in enumerate(range(STEP, END+1, STEP), 1):
            node = fixture._insert(cache, end, slot=slot, endpoint=end == END)
            value, groups = cache.tree_core.build_backup_spec(node)
            for transfers in groups.values():
                for t in transfers:
                    t.host_indices = t.device_indices + SS
            cache.tree_core.commit_backup(node, value+SHIFT, groups)
            # Actual D2H via production pool methods, before the tree writes L3.
            hidx, didx = value+SHIFT, value.to('cuda')
            kh.backup_from_device_all_layer(None, hidx, didx, 'kernel')
            dh.backup_from_device_all_layer(None, hidx, didx, 'kernel')
            mh.backup_from_device_all_layer(None, torch.tensor([slot+SS]),
                torch.tensor([slot], device='cuda'), 'kernel')
            sh.backup_from_device_all_layer(None, torch.tensor([slot+SS]),
                torch.tensor([slot], device='cuda'), 'kernel')
            torch.cuda.synchronize()
            cache.write_backup_storage(node)
            op = cc.backup_queue.get_nowait()
            cc._page_backup(op)
            assert op.completed_tokens == STEP, (op.completed_tokens, STEP)
            all_hashes.extend(op.hash_value)
            state_ops = [t for t in op.pool_transfers or () if t.name == PoolName.MAMBA]
            if state_ops:
                selected.append(end)
                assert all(op.pool_storage_result.extra_pool_hit_pages.get(t.name, 0)
                    == len(t.keys) for t in state_ops)
            side_ops = [t for t in op.pool_transfers or () if t.name == PoolName.DRAFT_INDEXER]
            assert len(side_ops) == len(state_ops)
            if side_ops:
                assert torch.equal(side_ops[0].host_indices, state_ops[0].host_indices)
                assert side_ops[0].keys == state_ops[0].keys
                assert op.pool_storage_result.extra_pool_hit_pages[PoolName.DRAFT_INDEXER] == 1
            object_count += len(op.hash_value)*2 + len(state_ops)*5
            cache._finish_backup_ack(
                op, requested_tokens=STEP, agreed_tokens=STEP, log_metrics=False
            )
        assert len(all_hashes)*PAGE == END
        assert selected == (list(range(STEP, END+1, STEP)) if interval == 0 else [32768,65536,73728]), selected
        # Forget the tree and physically zero both levels. Only L3 is retained.
        cache.tree_core.reset()
        wipe(gpu); wipe(host)
        torch.cuda.synchronize()
        assert all(torch.count_nonzero(x).item() == 0 for x in gpu+host)
        record = dict(backend=backend, interval=interval, selected_tokens=selected,
            kv_pages=len(all_hashes), physical_objects_written=object_count, checks=[])
        required = [PoolTransfer(PoolName.MAMBA, hit_policy=PoolHitPolicy.TRAILING_PAGES),
            PoolTransfer(PoolName.DRAFT),
            PoolTransfer(PoolName.DRAFT_INDEXER, hit_policy=PoolHitPolicy.TRAILING_PAGES)]
        for target in [END, 57344, 24576]:
            keys = all_hashes[:target//PAGE]
            hit = store.batch_exists_v2(keys, required)
            expected_end = max([x for x in selected if x <= target], default=0)
            assert hit.kv_hit_pages*PAGE == expected_end, dataclasses.asdict(hit)
            check = dict(target=target, hit_tokens=expected_end)
            if not expected_end:
                record['checks'].append(check)
                continue
            wipe(gpu); wipe(host)
            n = expected_end//PAGE
            hi = torch.arange(SHIFT, SHIFT+expected_end, dtype=torch.int64)
            state_host = torch.tensor([SS+1], dtype=torch.int64)
            transfers = [PoolTransfer(PoolName.KV, host_indices=hi, keys=all_hashes[:n]),
                PoolTransfer(PoolName.DRAFT, host_indices=hi, keys=all_hashes[:n]),
                PoolTransfer(PoolName.MAMBA, host_indices=state_host, keys=all_hashes[n-1:n]),
                PoolTransfer(PoolName.DRAFT_INDEXER, host_indices=state_host, keys=all_hashes[n-1:n])]
            loaded = store.batch_get_v2(transfers)
            assert all(all(xs) for xs in loaded.values()), loaded
            di = torch.arange(expected_end, device='cuda', dtype=torch.int64)
            kh.load_to_device_per_layer(None, hi.to('cuda'), di, 0, 'kernel')
            dh.load_to_device_per_layer(None, hi.to('cuda'), di, 0, 'kernel')
            for layer in range(2):
                mh.load_to_device_per_layer(None, state_host.to('cuda'),
                    torch.tensor([1], device='cuda'), layer, 'kernel')
            sh.load_to_device_per_layer(None, state_host.to('cuda'),
                torch.tensor([1], device='cuda'), 0, 'kernel')
            torch.cuda.synchronize()
            source_slot = expected_end//STEP
            assert torch.equal(kv[0][:expected_end], expected[0][:expected_end])
            assert torch.equal(temporal[:,1], expected[1][:,source_slot])
            for i, x in enumerate(conv):
                assert torch.equal(x[:,1], expected[2+i][:,source_slot])
            assert torch.equal(draft[0][:expected_end], expected[-2][:expected_end])
            assert torch.equal(side[0][1], expected[-1][source_slot])
            # Actual KDA decode kernel: compare output/final state before vs restore.
            q,k,v = [torch.randn(1,1,2,128,dtype=torch.bfloat16,device='cuda') for _ in range(3)]
            g = -torch.rand(1,1,2,128,device='cuda')
            beta = torch.rand(1,1,2,device='cuda')
            original = expected[1][0,source_slot].unsqueeze(0).contiguous()
            restored = temporal[0,1].unsqueeze(0).contiguous()
            a = fused_recurrent_kda(q,k,v,g,beta,initial_state=original.clone(),inplace_final_state=False)
            b = fused_recurrent_kda(q,k,v,g,beta,initial_state=restored.clone(),inplace_final_state=False)
            assert all(torch.equal(x,y) for x,y in zip(a,b,strict=True))
            check.update(tensors_bitwise_equal=True, kda_output_and_state_bitwise_equal=True)
            record['checks'].append(check)
        # Remove one physical state component: must fall back to a complete checkpoint.
        state_key = store._tag_keys(store._get_hybrid_page_component_keys(
            [all_hashes[-1]], PoolTransfer(PoolName.MAMBA))[0])[0]
        assert store.store.remove(state_key, force=True) == 0
        hit = store.batch_exists_v2(all_hashes, required)
        assert hit.kv_hit_pages*PAGE == selected[-2]
        record['missing_state_fallback_tokens'] = hit.kv_hit_pages*PAGE
        side_key = store._tag_keys(store._get_hybrid_page_component_keys(
            [all_hashes[selected[-2]//PAGE-1]], PoolTransfer(PoolName.DRAFT_INDEXER))[0])[0]
        assert store.store.remove(side_key, force=True) == 0
        hit = store.batch_exists_v2(all_hashes, required)
        assert hit.kv_hit_pages*PAGE == selected[-3]
        record['missing_state_sidecar_fallback_tokens'] = hit.kv_hit_pages*PAGE
        # Remove draft page at the head: whole-prefix draft requirements reject it.
        draft_key = store._tag_keys(store._get_hybrid_page_component_keys(
            [all_hashes[0]], PoolTransfer(PoolName.DRAFT))[0])[0]
        assert store.store.remove(draft_key, force=True) == 0
        assert store.batch_exists_v2(all_hashes, required).kv_hit_pages == 0
        record['missing_draft_rejected'] = True
        results.append(record)
        (ROOT/'l3-results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(record), flush=True)
    finally:
        store.close()
        store.store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', nargs=3, metavar=('BACKEND','INTERVAL','MASTER_PORT'))
    args = parser.parse_args()
    from sglang.srt.mem_cache.rust_tree_core.extension import bindings
    import mooncake
    try:
        mooncake_version = importlib.metadata.version('mooncake-transfer-engine')
    except importlib.metadata.PackageNotFoundError:
        mooncake_version = None
    (ROOT/'l3-environment.json').write_text(json.dumps(dict(
        torch=torch.__version__, mooncake_version=mooncake_version, mooncake_module=mooncake.__file__,
        gpu=torch.cuda.get_device_name(), dtype='bfloat16',
        rust_extension=bindings.__file__, rustc=subprocess.check_output(
            ['rustc','--version'],text=True).strip()), indent=2))
    if args.case:
        backend, interval, master_port = args.case
        records = []
        run_case(backend, int(interval), int(master_port), records)
        (ROOT/f'l3-case-{backend}-{interval}.json').write_text(json.dumps(records[0], indent=2))
        sys.exit(0)
    ports = [free_port() for _ in range(3)]
    assert len(set(ports)) == 3
    results = []
    with (ROOT/'mooncake-master.log').open('w') as log:
        master = subprocess.Popen(['mooncake_master', f'--rpc_port={ports[0]}',
            f'--metrics_port={ports[1]}', f'--http_metadata_server_port={ports[2]}'],
            stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if master.poll() is not None:
                    raise RuntimeError('isolated Mooncake master exited')
                try:
                    with socket.create_connection(('127.0.0.1', ports[0]), timeout=.2):
                        break
                except OSError:
                    time.sleep(.1)
            for backend in ['python','rust']:
                for interval in [0,32768]:
                    # Each worker owns its CUDA host registrations and store client.
                    subprocess.run([sys.executable,__file__,'--case',backend,str(interval),str(ports[0])],
                        check=True,timeout=600)
                    results.append(json.loads((ROOT/f'l3-case-{backend}-{interval}.json').read_text()))
            assert len(results) == 4
            (ROOT/'l3-results.json').write_text(json.dumps(results, indent=2))
            print('ALL FOUR L3 ROUND TRIPS PASSED', flush=True)
        finally:
            master.terminate()
            try:
                master.wait(timeout=15)
            except subprocess.TimeoutExpired:
                master.kill(); master.wait()
