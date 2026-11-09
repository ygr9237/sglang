# Kimi-K3.1 prefill-CP image ownership: evidence

Base: `704681229f1cdda291dbd2261dabb2ae80a4a49d`. Two B300 hosts; PyTorch 2.14.1+cu130, real Kimi-K3.1-Pro BF16 ViT and projector weights. Tested attention-TP1, CP2, attention-DP1. Hosts used opposite A/B orders. Full service was not started for these measurements.

`gpu_check.py` calls the actual `get_image_feature`, checks owner counts and rank agreement, and saves feature tensors for comparison. Fixed random preprocessed patch inputs: a 1K grid is 74×74 patches. Six cases: one/two/ten/twenty 1K images, uneven sizes, and a small temporal-pooling check. Not a JPEG/PNG, video service, HTTP or full-LLM benchmark. Four warmups and twelve samples per case; sample time is the maximum across ranks and the report uses the median. ViT graphs are disabled.

The per-rank JSON records raw timings, ownership, tensor content hashes and allocator peaks. `gpu-comparison.json` reports exact equality of baseline and patched saved tensors. Allocator peaks describe only this encoder benchmark. The CPU suites report 109 tests and 4 subtests per host; `cpu-no-gpu-final.log` also runs with GPUs hidden. The final registered-test entry point adds no runtime changes and passes 11 tests directly. The Gloo harness checks cold/hot/partial caches and chunk boundaries with real collectives and a synthetic tower.

## Reproduce

From a clone containing this evidence directory, with the checkpoint mounted at `/modules/Kimi-K3.1-Pro` and SGLang dependencies installed:

```bash
EVIDENCE="$PWD/benchmark/kimi_k31_cp_vision_0014_20261110"
RUN_ROOT="$(mktemp -d)"
git worktree add --detach "$RUN_ROOT/upstream" 704681229f1cdda291dbd2261dabb2ae80a4a49d
git worktree add --detach "$RUN_ROOT/pr2-main" 704681229f1cdda291dbd2261dabb2ae80a4a49d
git -C "$RUN_ROOT/pr2-main" apply "$EVIDENCE/0014-main.patch"
cp "$EVIDENCE/gpu_check.py" "$EVIDENCE/gloo_cache_check.py" "$RUN_ROOT/"
for mode in upstream pr2-main; do
  PYTHONPATH="$RUN_ROOT/$mode/python" CHECK_MODE="$mode" CUDA_VISIBLE_DEVICES=0,1 \
  OMP_NUM_THREADS=4 SGLANG_VIT_ENABLE_CUDA_GRAPH=0 \
    torchrun --standalone --nproc_per_node=2 "$RUN_ROOT/gpu_check.py"
done
PYTHONPATH="$RUN_ROOT/pr2-main/python" CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=4 \
  python "$RUN_ROOT/pr2-main/test/registered/unit/multimodal/test_kimi_k3_pcp_vision_dp.py"
PYTHONPATH="$RUN_ROOT/pr2-main/python" CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=4 \
  torchrun --standalone --nproc_per_node=2 "$RUN_ROOT/gloo_cache_check.py"
```

Reverse the first loop order for a second-host check. For a different checkpoint mount, update `model_path` in the harness. CPU CI uses an out-of-range CUDA index because the upstream test utilities do not accept an empty `CUDA_VISIBLE_DEVICES` string.

Current-main full-service TTFT, other CP widths, external encoders and production replay were not measured. Historical downstream service checks are separately disclosed in the PR; the microbenchmark does not establish an end-to-end speedup.
