# Mamba/KDA L3 snapshot interval validation

Candidate: [dbf1e025fdd52da542c24ec16cfc127998016c0b](https://github.com/ygr9237/sglang/commit/dbf1e025fdd52da542c24ec16cfc127998016c0b). PR base: `3830fb0713d2451d4e7a993ce7bef8c0490a15ed`.

- Registered tests: 30 new, 94 relevant regressions and 45 subtests passed; 5 skipped.
- Four real Mooncake component round trips: Python/Rust, interval 0/32768, one B300, TCP loopback.
- After resetting the tree and zeroing device/host buffers, restored KV, BF16 temporal/conv, draft KV and a registered state-derived sidecar were bitwise equal. The subsequent KDA decode output and final state matched the saved endpoint bitwise (10 endpoint comparisons).
- Missing state/sidecar objects selected a complete earlier checkpoint. Missing required draft KV rejected the prefix.

The component runs used base `0c44fbff44da7d1a11f0e8b4299e4b8a702ac414`. All 14 candidate files match the publication candidate; CPU regressions were rerun after rebasing. Check `summary.json` for LF-normalized source hashes.

## Scope

Real host pools, D2H/H2D copies, production backup/storage methods and an isolated Mooncake master/client were used. Tree/index allocation used the registered test fixture. Each case ran in a separate process to isolate CUDA host-memory registrations.

This is not full-model DFLASH/PP validation, an allocator-eviction test, a recomputation-logits comparison, or an HTTP/RDMA performance benchmark. Log messages about unsupported small-row JIT layouts select the existing kernel fallback. The initial local segment lookup message is emitted during TCP engine setup; warmup and all asserted operations succeeded.

## Reproduce

Requires the SGLang Python/CUDA dependencies, Mooncake SDK/master binary, and Rust toolchain. The validation image used Python 3.12.16, PyTorch 2.14.1+cu130, Rust 1.90.0 and B300. Its Mooncake SDK had no pip distribution metadata; `l3/l3-environment.json` records this explicitly.

Place the harness files in `$RUN_ROOT`, and put the candidate checkout in `$RUN_ROOT/source`:

```bash
git clone --depth 1 --branch codex/hicache-mamba-l3-snapshot-interval \
  https://github.com/ygr9237/sglang.git "$RUN_ROOT/source"
git -C "$RUN_ROOT/source" fetch origin dbf1e025fdd52da542c24ec16cfc127998016c0b
git -C "$RUN_ROOT/source" checkout --detach dbf1e025fdd52da542c24ec16cfc127998016c0b

export PATH="$HOME/.cargo/bin:$PATH"
export PYTHONPATH="$RUN_ROOT/source/python"
export RUSTUP_TOOLCHAIN=1.90 SGLANG_RUST_BUILD_MODE=auto
export SGLANG_CACHE_DIR="$RUN_ROOT/cache"
export CARGO_BUILD_JOBS=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=99 python "$RUN_ROOT/regression_runner.py"
CUDA_VISIBLE_DEVICES=0 python "$RUN_ROOT/l3_roundtrip.py"
CUDA_VISIBLE_DEVICES=99 taskset -c 0 python "$RUN_ROOT/plan_bench.py"
```

The interval is opt-in. In the 72K prefix / 8K-node fixture, state checkpoints decreased from 9 to 3 while all 1152 KV pages were retained. A 56K partial prefix restored only 32K with the sparse policy; the remaining tokens require recomputation. KDA comparisons check the restored endpoint, not the final logits after that recomputation.
