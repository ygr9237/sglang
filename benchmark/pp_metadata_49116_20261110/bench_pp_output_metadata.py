"""Benchmark PP metadata preparation, pickle, and the actual output transport.

Run sender measurements on one GPU. Run transport with torchrun --nproc-per-node=2.
This measures the output leg only, not model forward or serving throughput.
"""

import argparse
import gc
import importlib.util
import json
import os
import pickle
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import GroupCoordinator, _split_tensor_dict
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult

ROOT = Path(__file__).resolve().parent


def load_prepare(arm):
    name = "pp_metadata_benchmark_" + arm
    spec = importlib.util.spec_from_file_location(
        name, ROOT / arm / "scheduler_pp_mixin.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.SchedulerPPMixin()._pp_prepare_tensor_dict


def cases():
    return [
        {"name": "no_logprob_b1", "kind": "none", "batch": 1},
        {"name": "no_logprob_b25", "kind": "none", "batch": 25},
        {"name": "no_logprob_b50", "kind": "none", "batch": 50},
        {"name": "top_level_tensor_b25", "kind": "top", "batch": 25},
        {"name": "python_top20_b25", "kind": "python", "batch": 25},
        {"name": "python_top20_b50", "kind": "python", "batch": 50},
        {"name": "cpu_rows_b1", "kind": "cpu", "batch": 1},
        {"name": "cpu_rows_b25", "kind": "cpu", "batch": 25},
        {"name": "cpu_rows_b50", "kind": "cpu", "batch": 50},
        {"name": "cpu_view32k_b1", "kind": "cpu", "batch": 1, "storage_rows": 32768},
        {"name": "cpu_view32k_b25", "kind": "cpu", "batch": 25, "storage_rows": 32768},
        {"name": "cpu_view128k_b1", "kind": "cpu", "batch": 1, "storage_rows": 131072},
        {"name": "cuda_rows_b1", "kind": "cuda", "batch": 1},
        {"name": "cuda_rows_b25", "kind": "cuda", "batch": 25},
        {"name": "cuda_rows_b50", "kind": "cuda", "batch": 50},
        {"name": "cuda_view32k_b1", "kind": "cuda", "batch": 1, "storage_rows": 32768},
        {
            "name": "cuda_view32k_b25",
            "kind": "cuda",
            "batch": 25,
            "storage_rows": 32768,
        },
        {
            "name": "python_prompt_top20_32k",
            "kind": "python_prompt",
            "batch": 1,
            "storage_rows": 32768,
        },
    ]


def make_payload(case):
    b = case["batch"]
    logits = LogitsProcessorOutput(next_token_logits=None)
    kind = case["kind"]
    if kind == "top":
        logits.next_token_logprobs = torch.full((b,), -1.0, device="cuda")
    elif kind != "none":
        rows = case.get("storage_rows", b)
        device = "cuda" if kind == "cuda" else "cpu"
        values = torch.arange(rows * 20, device=device, dtype=torch.float32).view(
            rows, 20
        )
        indices = torch.arange(rows * 20, device=device, dtype=torch.int64).view(
            rows, 20
        )
        if kind == "python_prompt":
            logits.input_top_logprobs_val = [values.tolist()]
            logits.input_top_logprobs_idx = [indices.tolist()]
        else:
            vals, ids = list(values[-b:].unbind()), list(indices[-b:].unbind())
            if kind == "python":
                vals, ids = [v.tolist() for v in vals], [v.tolist() for v in ids]
            logits.next_token_top_logprobs_val = vals
            logits.next_token_top_logprobs_idx = ids
    result = GenerationBatchResult(
        next_token_ids=torch.arange(b, device="cuda"), logits_output=logits
    )
    batch = SimpleNamespace(return_logprob=kind != "none")
    torch.cuda.synchronize()
    return result, batch


def percentile(values, q):
    a = sorted(values)
    position = (len(a) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (position - lo)


def summary(values):
    return {
        "p50_us": statistics.median(values),
        "p95_us": percentile(values, 0.95),
        "mean_us": statistics.mean(values),
        "samples": len(values),
    }


def one_sender(prepare, result, batch):
    t0 = time.perf_counter_ns()
    wire = prepare(result, batch)
    t1 = time.perf_counter_ns()
    metadata, tensors = _split_tensor_dict(wire)
    t2 = time.perf_counter_ns()
    encoded = pickle.dumps(metadata)
    t3 = time.perf_counter_ns()
    row = {
        "prepare_us": (t1 - t0) / 1000,
        "split_us": (t2 - t1) / 1000,
        "pickle_us": (t3 - t2) / 1000,
        "total_us": (t3 - t0) / 1000,
        "pickle_bytes": len(encoded),
    }
    return row


def save(name, data):
    p = ROOT / (name + ".part")
    p.write_text(json.dumps(data, indent=2))
    p.replace(ROOT / (name + ".json"))


def validate(result, batch, functions):
    base = functions["base"](result, batch)
    fixed = functions["fixed"](result, batch)

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and a.shape == b.shape
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
        elif isinstance(a, (list, tuple)):
            assert type(a) is type(b) and len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        else:
            assert a == b

    equal(base, fixed)
    assert base["next_token_ids"] is fixed["next_token_ids"] is result.next_token_ids


def sender(functions):
    out = {
        "kind": "sender_prepare_split_pickle",
        "rounds": 6,
        "warmups_per_arm": 3,
        "cases": [],
    }
    for case in cases():
        result, batch = make_payload(case)
        validate(result, batch, functions)
        timings = {arm: [] for arm in functions}
        trial = {}
        for arm, fn in functions.items():
            for _ in range(3):
                one_sender(fn, result, batch)
            trial[arm] = one_sender(fn, result, batch)["total_us"]
        count = min(200, max(3, int(40000 / max(trial.values()))))
        rounds = {arm: [] for arm in functions}
        gc.collect()
        gc.disable()
        for r in range(6):
            for arm in ["base", "fixed"] if r % 2 == 0 else ["fixed", "base"]:
                rows = [one_sender(functions[arm], result, batch) for _ in range(count)]
                timings[arm].extend(rows)
                rounds[arm].append(statistics.median(x["total_us"] for x in rows))
        gc.enable()
        item = {"case": case, "iterations_per_round": count, "arms": {}}
        for arm, rows in timings.items():
            item["arms"][arm] = {
                key: summary([x[key] for x in rows])
                for key in ["prepare_us", "split_us", "pickle_us", "total_us"]
            }
            item["arms"][arm]["pickle_bytes"] = rows[-1]["pickle_bytes"]
            item["arms"][arm]["round_median_total_us"] = rounds[arm]
        out["cases"].append(item)
        save("sender-results", out)
        print(
            json.dumps(
                {
                    "case": case["name"],
                    "base_us": item["arms"]["base"]["total_us"]["p50_us"],
                    "fixed_us": item["arms"]["fixed"]["total_us"]["p50_us"],
                    "samples": count * 6,
                }
            ),
            flush=True,
        )
        del result, batch
    out["complete"] = True
    save("sender-results", out)


def profile(functions):
    chosen = {
        "no_logprob_b25",
        "cpu_view32k_b25",
        "cuda_rows_b25",
        "cuda_view32k_b25",
        "python_prompt_top20_32k",
    }
    output = []
    for case in cases():
        if case["name"] not in chosen:
            continue
        result, batch = make_payload(case)
        for arm, fn in functions.items():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                with torch.profiler.record_function("pp.prepare_metadata"):
                    wire = fn(result, batch)
                with torch.profiler.record_function("pp.split_top_level"):
                    metadata, _ = _split_tensor_dict(wire)
                with torch.profiler.record_function("pp.pickle_metadata"):
                    encoded = pickle.dumps(metadata)
                torch.cuda.synchronize()
            keys = []
            for event in prof.key_averages():
                if any(
                    x in event.key.lower()
                    for x in ["pp.", "clone", "copy", "memcpy", "synchron", "aten::to"]
                ):
                    keys.append(
                        {
                            "key": event.key,
                            "count": event.count,
                            "cpu_total_us": event.cpu_time_total,
                            "self_cpu_us": event.self_cpu_time_total,
                            "device_total_us": event.device_time_total,
                            "cpu_memory_bytes": event.cpu_memory_usage,
                        }
                    )
            trace = ROOT / ("profile-" + case["name"] + "-" + arm + ".json")
            prof.export_chrome_trace(str(trace))
            output.append(
                {
                    "case": case["name"],
                    "arm": arm,
                    "pickle_bytes": len(encoded),
                    "events": keys,
                }
            )
            del wire, metadata, encoded
        del result, batch
    save("profiles", output)


def transport(functions):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    cpu = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
    group = object.__new__(GroupCoordinator)
    group.world_size, group.rank_in_group, group.ranks = 2, rank, [0, 1]
    group.device_group, group.cpu_group = dist.group.WORLD, cpu
    forward, comm = torch.cuda.Stream(), torch.cuda.Stream()
    ack = torch.empty(1, dtype=torch.int32)
    selected = {
        "no_logprob_b25",
        "python_top20_b25",
        "cpu_rows_b25",
        "cpu_view32k_b1",
        "cpu_view32k_b25",
        "cuda_rows_b25",
        "cuda_view32k_b1",
    }
    out = {
        "kind": "actual_GroupCoordinator_output_leg_with_ack",
        "rounds": 6,
        "cases": [],
    }
    for case in cases():
        if case["name"] not in selected:
            continue
        result, batch = make_payload(case) if rank == 0 else (None, None)
        arms = ["base", "fixed"]
        count = 20 if case.get("storage_rows", 0) else 250
        timings = {arm: [] for arm in arms}
        round_medians = {arm: [] for arm in arms}
        for r in range(6):
            order = arms if r % 2 == 0 else list(reversed(arms))
            for arm in order:
                dist.barrier(group=cpu)
                samples = []
                for i in range(count + 3):
                    if rank == 0:
                        t0 = time.perf_counter_ns()
                        with torch.cuda.stream(forward):
                            wire = functions[arm](result, batch)
                            ready = forward.record_event()
                        with torch.cuda.stream(comm):
                            comm.wait_event(ready)
                            work = group.send_tensor_dict(wire, dst=1, async_send=True)
                        for w in work:
                            if w.work is not None:
                                w.work.wait()
                        dist.recv(ack, src=1, group=cpu)
                        elapsed = (time.perf_counter_ns() - t0) / 1000
                        if i >= 3:
                            samples.append(elapsed)
                        del wire, work, ready
                    else:
                        with torch.cuda.stream(comm):
                            wire = group.recv_tensor_dict(src=0)
                        comm.synchronize()
                        if i == 0:
                            assert wire["next_token_ids"].tolist() == list(
                                range(case["batch"])
                            )
                            if case["kind"] != "none":
                                leaf = wire["next_token_top_logprobs_val"][0]
                                if isinstance(leaf, torch.Tensor):
                                    assert leaf.device.type == "cpu"
                                values = (
                                    leaf.tolist()
                                    if isinstance(leaf, torch.Tensor)
                                    else leaf
                                )
                                start = (
                                    case.get("storage_rows", case["batch"])
                                    - case["batch"]
                                ) * 20
                                assert values == list(range(start, start + 20))
                        del wire
                        dist.send(ack, dst=0, group=cpu)
                if rank == 0:
                    timings[arm].extend(samples)
                    round_medians[arm].append(statistics.median(samples))
        if rank == 0:
            item = {
                "case": case,
                "iterations_per_round": count,
                "arms": {
                    arm: {
                        "total_us": summary(values),
                        "round_median_total_us": round_medians[arm],
                    }
                    for arm, values in timings.items()
                },
            }
            out["cases"].append(item)
            save("transport-results", out)
            print(
                json.dumps(
                    {
                        "transport_case": case["name"],
                        "arms": {
                            a: item["arms"][a]["total_us"]["p50_us"] for a in arms
                        },
                    }
                ),
                flush=True,
            )
            del result, batch
    if rank == 0:
        out["complete"] = True
        save("transport-results", out)
    dist.destroy_process_group(cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["sender", "profile", "transport"])
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    desired = set(range(0, 4)) if rank == 0 else set(range(96, 100))
    if hasattr(os, "sched_getaffinity") and desired <= os.sched_getaffinity(0):
        os.sched_setaffinity(0, desired)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    functions = {arm: load_prepare(arm) for arm in ["base", "fixed"]}
    if args.mode == "sender":
        sender(functions)
    elif args.mode == "profile":
        profile(functions)
    else:
        transport(functions)
