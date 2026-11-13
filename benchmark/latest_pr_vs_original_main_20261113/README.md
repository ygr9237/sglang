# Latest PR heads versus original upstream main

This dataset compares only unpatched main with the current PR code. No earlier PR revision is used as a performance baseline.

| PR | Unpatched main | Latest PR head |
|---|---|---|
| #49116 | `d1051b47bf77bf4530e352cc2543b9a9dedb86a2` | `db70c52999cf1d31d971008cf79939af47d420c9` |
| #49342 | `704681229f1cdda291dbd2261dabb2ae80a4a49d` | `b7cc46c583d2236c334fd281633218d2e858b98d` |

2026-11-13, host 1, Xeon 6776P / B300, Python 3.12.16, Torch 2.14.1+cu130. Cyclic GC and coverage disabled; six alternating A/B rounds. Source snapshots and latest source hashes were checked against these Git commits; deployment code was unchanged.

## #49342: image ownership, cache and error coordination

Actual BF16 ViT/projector weights, TP2/CP2, attention-TP=1, grid=(1,74,74), native GroupCoordinator/PyNccl, ViT CUDA graph disabled. The original main cache scheduler and vision functions are loaded from the pinned source snapshots. The latest arm uses the current PR. Weights, inputs and remaining modules are shared. Six rounds, 6 warmups + 12 samples per arm/round, 72 samples per arm/case; timing takes the slower rank. GPU workers use fixed CPU affinity and this measurement runs independently of the PP sender repeat.

Cold cache:

| Images | Original main ms | Latest PR ms | Latency change |
|---|---:|---:|---:|
| 1 | 10.994 | 12.692 | +15.44% |
| 10 | 91.579 | 47.875 | -47.72% |
| 20 | 184.022 | 93.457 | -49.21% |


Hot cache:

| Images | Original main ms | Latest PR ms | Added ms |
|---|---:|---:|---:|
| 1 | 0.04459 | 0.49356 | +0.44897 |
| 10 | 0.11513 | 0.55890 | +0.44377 |
| 20 | 0.19673 | 0.62407 | +0.42734 |


Outputs are bit-identical across main/current, cold/hot, repeats and ranks. The current PR improves multi-image encoding and adds measurable single-image/cache-hit coordination cost. These are encoder/cache microbenchmarks, not service TTFT/throughput.

## #49116: compact, communication-ready output metadata

Sender preparation + split + pickle, P50. Bytes refer to serialized object metadata; top-level tensor transport is separate. Input creation is outside timing. Sampling is adaptive (see each case), six balanced rounds. CUDA sender timing executes on the forward stream.

| Payload | Main ms | Latest PR ms | Main bytes | Latest PR bytes |
|---|---:|---:|---:|---:|
| No logprobs, B25 | 0.00395 | 0.00405 | 147 | 147 |
| Python top-20 rows, B25 | 0.01503 | 0.02226 | 6,679 | 6,679 |
| CPU tensor rows, B25 | 0.98457 | 1.07526 | 164,701 | 20,609 |
| CPU 32K-storage views, B1 | 2.96551 | 0.06549 | 7,865,733 | 1,623 |
| CPU 32K-storage views, B25 | 205.18596 | 1.11230 | 196,623,359 | 20,609 |
| CUDA tensor rows, B25 | 2.18557 | 2.01020 | 164,851 | 20,609 |
| CUDA 32K-storage views, B1 | 4.61322 | 0.11578 | 7,865,739 | 1,623 |
| 32K prompt top-20 Python lists | 13.20526 | 17.57513 | 9,308,003 | 9,308,003 |


Actual two-rank GroupCoordinator output leg, including preparation, pickle, NCCL/Gloo transfer and ACK; six balanced rounds, 1,500 samples/arm for ordinary cases and 120/arm for storage-view cases:

| Output leg | Main P50 / P95 ms | Latest PR P50 / P95 ms |
|---|---:|---:|
| No logprobs, B25 | 0.338 / 0.374 | 0.338 / 0.376 |
| Python top-20 rows, B25 | 0.399 / 0.436 | 0.408 / 0.432 |
| CPU tensor rows, B25 | 4.842 / 5.940 | 5.584 / 6.652 |
| CPU 32K-storage views, B1 | 13.276 / 23.145 | 0.847 / 1.000 |
| CPU 32K-storage views, B25 | 639.340 / 800.049 | 4.606 / 5.942 |


The main CUDA probe leaves CUDA tensors in object metadata; the current PR produces independent CPU storage (80 and 160 bytes for the two 20-element rows, versus 2,621,440 and 5,242,880 bytes backing storage). The transport comparison uses CPU metadata. This does not time the unsafe mixed-stream CUDA baseline. Numeric-only row snapshots and some small tensor rows add overhead; this fix is evaluated for correctness and compact metadata, not a universal speedup.

## Files

`summary.json` and `results/status.json` identify the exact comparison and primary runs. `vision_main.py` / `pp_main.py` and `refine.py` reproduce the measurements using the included original and current PP source snapshots and pinned PR checkouts. Initial concurrent measurements and the selected sequential GPU/sender measurements are retained in separate logs; the tables use the runs named in status.json. Full-service stress and model-serving accuracy are outside this measurement.
