# Kimi-K3.1 CP image ownership with cache/error coordination

Base: `704681229f1cdda291dbd2261dabb2ae80a4a49d`. The primary comparison is unmodified pinned main versus the complete proposed change, including ownership and its required coordination. Two independent B300 hosts, two ranks per host, PyTorch 2.14.1+cu130; actual Kimi-K3.1-Pro BF16 ViT and projector weights. Host A runs original then final, host B runs final then original.

`gpu_check.py` is the actual harness used. Fixed random preprocessed patch tensors, grid=(1,74,74), 5476 raw patches and 1369 output tokens per image; 1/10/20 images, cold/hot embedding caches. Six warmups and twelve samples, maximum wall time across ranks, median reported. ViT CUDA graph disabled. These are encoder/cache measurements, not JPEG preprocessing, HTTP TTFT, full-language-model or concurrent service tests. Every before/after and rank output hash agrees.

Cold 10-image latency improves about 48%, cold 20-image latency about 49%. Single-image encoding adds about 0.31–0.38 ms; hot-cache processing adds about 0.28–0.38 ms for the required metadata collective. Raw samples are in each host directory.

CPU logs rerun the final registered entry points with GPUs hidden: the two-rank Gloo coordination test exercises 18 scenarios and the existing ownership file passes 11 tests on each host. The coordinator unittest uses the community CustomTestCase base; this test-only adjustment occurred after GPU measurements, with runtime file hashes unchanged. `precommit.log` records the final changed-file hooks. Apply/reverse and final-file comparisons passed for the complete and incremental patches.

## Reproduce

Create two source trees at the base SHA and apply `complete-main.patch` only to the final tree. Use the same SGLang dependency image, mount the checkpoint read-only at `/modules/Kimi-K3.1-Pro`, and mount an empty writable results directory at `/audit_results`. For each tree set `PYTHONPATH` to its `python` directory and `CHECK_MODE=before` or `after`, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 SGLANG_VIT_ENABLE_CUDA_GRAPH=0 \
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 \
  --master_port=29501 /path/to/gpu_check.py
```

Run the final tests with `CUDA_VISIBLE_DEVICES=99` and the final tree on `PYTHONPATH`:

```bash
python test/registered/unit/multimodal/test_kimi_k3_pcp_vision_dp.py
python test/registered/unit/multimodal/test_kimi_k3_image_dp_consensus.py
```

The GPU error injection raises Python MemoryError before a real tower call. The test caller catches the coordinated error and successfully encodes again; it does not establish full-server recovery from fatal CUDA OOM. The scheduler may still exit on a propagated encoding exception. Other CP widths, production replay and long-running full-service behavior remain unqualified.
