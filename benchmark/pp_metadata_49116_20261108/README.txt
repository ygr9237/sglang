PP output metadata performance evidence for sgl-project/sglang#49116

Base: cbbc6b67f48008a8b739f4667d7b14abb89088cf
Fix: a53050cb86f7bd51ed44b0ffda5add350d0a7ff7

The benchmark imports the unmodified runtime modules from a checkout containing the fix,
and selects the base/fixed SchedulerPPMixin module for the preparation function.
These are component measurements; no model weights are loaded.

Reproduction from an SGLang development checkout with the same dependencies:
1. Put these benchmark files in a writable directory, and cd there.
2. Create base/ and fixed/; populate their scheduler_pp_mixin.py files with:
   git -C /path/to/sglang show cbbc6b67f48008a8b739f4667d7b14abb89088cf:python/sglang/srt/managers/scheduler_pp_mixin.py > base/scheduler_pp_mixin.py
   git -C /path/to/sglang show a53050cb86f7bd51ed44b0ffda5add350d0a7ff7:python/sglang/srt/managers/scheduler_pp_mixin.py > fixed/scheduler_pp_mixin.py
3. Export PYTHONPATH=/path/to/sglang/python and OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.
4. CUDA_VISIBLE_DEVICES=0 python bench_pp_output_metadata.py sender
5. CUDA_VISIBLE_DEVICES=0 python bench_pp_output_metadata.py profile
6. CUDA_VISIBLE_DEVICES=0,6 python -m torch.distributed.run --standalone --nproc-per-node=2 bench_pp_output_metadata.py transport
7. CUDA_VISIBLE_DEVICES=0,6 python -m torch.distributed.run --standalone --nproc-per-node=2 bench_pp_metadata_controls.py

The script pins the sender to CPUs 0-3 and receiver to 96-99 when those sets are allowed.
Change that policy for a different machine topology. GPU0/GPU6 communicate via local NVLink;
metadata uses a Gloo CPU process group. The ACK is sent only after receive/deserialization.
The harness initializes Torch process groups and the GroupCoordinator fields used by its
real send/recv methods; it does not construct a serving scheduler or run a model forward.

All timed cases verify payload values. Sender/Profile CUDA baseline inputs are complete
before serialization; the known-unsafe CUDA baseline is omitted from real comm-stream
transport. Controls repeat the small cases with cyclic GC disabled. Raw JSON preserves
both ordinary-GC transport and control results. Do not substitute component speedups
for model TPM, TTFT, cross-node performance or full distributed-hang validation.
