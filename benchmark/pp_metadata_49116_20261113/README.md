# PR #49116 metadata validation, 2026-11-13

Candidate `db70c52999cf1d31d971008cf79939af47d420c9` is based on official main `d1051b47bf77bf4530e352cc2543b9a9dedb86a2`. The `base` arm is the previously published PR head `11aa5e5b12b8160293b472eac12a8f1a28a57310`; `fixed` is this candidate; `unpatched` is the official main reference, used only with numeric payloads.

Hardware: Xeon 6776P / B300; Python 3.12.16, Torch 2.14.1+cu130, Transformers 5.19.0. These are configuration and component checks, not model-serving throughput or TTFT measurements.

## Results

- Native registered tests: 8 CPU tests and 1 combined CUDA test passed. Pytest: 9 tests and 13 subtests passed.
- Changed executable runtime lines: 100% (19/19), exceeding the 60% gate.
- Actual producer layouts: eight prefill/decode × top-k/requested-token × Python/tensor layouts passed, plus routed-expert output identity.
- Full-repository default-stage pre-commit: 36 hooks passed; 2 had no matching files; no SKIP exclusions. Validation left source unchanged.
- 32K prompt top-20 sender preparation + pickle: 71.149 -> 17.547 ms (75.3% lower). A separate same-input paired reference measured unpatched main 12.091 ms versus candidate 16.902 ms. Independent row snapshots still have a cost.
- Pure Python output rows B25 sender: 0.0624 -> 0.0225 ms. Tensor copy counts, compactness and payload sizes remain covered.
- Real two-rank Python-row output leg B25 P50/P95: 0.545/0.835 -> 0.467/0.512 ms.

Tensor-row transport has bimodal jitter: CPU B25 aggregate P50/P95 was 4.927/5.706 -> 5.560/5.695 ms; five of six paired round medians differ by less than 1%. Sender CPU-row preparation + pickle was 1.072 -> 1.068 ms. Raw per-round and per-sample data are retained; no general tensor-transport speedup is claimed.

## Reproduction

Use a complete SGLang development environment. This evidence branch contains the official-main source and the three standalone arm modules.

```bash
# At the repository root of this evidence branch:
git apply benchmark/pp_metadata_49116_20261113/candidate.patch
PYTHONPATH=python CUDA_VISIBLE_DEVICES=99 python test/registered/unit/managers/test_pp_output_exchange_order.py -v
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python test/registered/e2e/parallelism/test_pp_output_metadata.py -v
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python benchmark/pp_metadata_49116_20261113/bench_pp_output_metadata.py sender
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python benchmark/pp_metadata_49116_20261113/numeric_reference.py
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python benchmark/pp_metadata_49116_20261113/bench_pp_output_metadata.py profile
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 benchmark/pp_metadata_49116_20261113/controlled_transport.py
PYTHONPATH=python CUDA_VISIBLE_DEVICES=99 python benchmark/pp_metadata_49116_20261113/producer_probe.py
# Repository lint uses the clean candidate tree, excluding evidence files:
git worktree add ../pp49116-lint db70c52999cf1d31d971008cf79939af47d420c9
cd ../pp49116-lint
pre-commit run --all-files --show-diff-on-failure
```

Sender and transport scripts alternate the arms across six rounds; input creation is outside timing. Two-rank transport includes preparation, pickle, NCCL/Gloo transfer and completion ACK, with cyclic GC disabled. There are 1,500 samples per arm for small transport messages and 120 for large-view messages. The timing JSON files include percentile, per-round and raw sample data where collected.
