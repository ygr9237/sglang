"""Original unpatched numeric-only path versus candidate, using the same payloads."""
import os

import torch

import bench_pp_output_metadata as bench

if __name__ == "__main__":
    if set(range(4)) <= os.sched_getaffinity(0):
        os.sched_setaffinity(0, set(range(4)))
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    selected = [c for c in bench.cases() if c["name"] == "python_prompt_top20_32k"]
    bench.cases = lambda: selected
    original_save = bench.save
    bench.save = lambda name, value: original_save("reference-" + name, value)
    bench.sender({"base": bench.load_prepare("unpatched"), "fixed": bench.load_prepare("fixed")})
