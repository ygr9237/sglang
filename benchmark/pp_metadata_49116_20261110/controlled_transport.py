"""Compare the published fix with the candidate; both CUDA paths are safe."""
import gc
import os

import torch

import bench_pp_output_metadata as bench

if __name__ == "__main__":
    rank = int(os.environ["LOCAL_RANK"])
    desired = set(range(4)) if rank == 0 else set(range(96, 100))
    if desired <= os.sched_getaffinity(0):
        os.sched_setaffinity(0, desired)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    original_summary = bench.summary
    bench.summary = lambda values: original_summary(values) | {"samples_us": values}
    gc.collect()
    gc.disable()
    bench.transport({arm: bench.load_prepare(arm) for arm in ["base", "fixed"]})
