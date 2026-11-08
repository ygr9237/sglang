"""Repeat warmed controls with cyclic GC disabled; production methods are unchanged."""

import gc
import os

import torch

import bench_pp_output_metadata as bench

if __name__ == "__main__":
    rank = int(os.environ["LOCAL_RANK"])
    desired = set(range(0, 4)) if rank == 0 else set(range(96, 100))
    if desired <= os.sched_getaffinity(0):
        os.sched_setaffinity(0, desired)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    selected = [
        case
        for case in bench.cases()
        if case["name"] in {"no_logprob_b25", "python_top20_b25"}
    ]
    bench.cases = lambda: selected * 5
    original_summary, original_save = bench.summary, bench.save
    bench.summary = lambda values: original_summary(values) | {"samples_us": values}
    bench.save = lambda name, value: original_save("control-" + name, value)
    gc.collect()
    gc.disable()
    bench.transport({arm: bench.load_prepare(arm) for arm in ["base", "fixed"]})
