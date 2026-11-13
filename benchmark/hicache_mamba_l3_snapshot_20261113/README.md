# Mamba/KDA sparse L3 interval validation — 2026-11-13

Revised candidate for sgl-project/sglang#49518, on publication base `3830fb0713d2451d4e7a993ce7bef8c0490a15ed`. The complete patch, 14 tested source files and SHA256 records are included here. Runtime services were not changed.

- 36 policy tests passed with Python and the actual compiled Rust core. Pending writes use `UnifiedRadixCache._finish_write_through_ack`, including a KV-only split at 18 before the legal state at 20 is acknowledged. Periodic-node promotion does not submit an extra storage task.
- 94 related regressions and 45 subtests passed; 5 skipped.
- Additional radix-tree Clippy passed with `--all-targets --features python-extension,inspection,tch/doc-only -- -D warnings` (Rust 1.90).
- Every applicable pre-commit hook on all 14 PR files passed, including Rust workspace Clippy and Rustfmt. Hooks with no matching files are reported as skipped by pre-commit.
- Four real Mooncake round trips passed on B300 (Python/Rust × default/32K): restored tensors and subsequent KDA output/state were bitwise equal; partial hits and missing-object fallbacks passed.

## Latest candidate results

A 73,728-token prefix with nine 8,192-token nodes and 64-token KV pages writes nine state checkpoints with interval 0 and three with interval 32768. Both retain all 1,152 KV pages: 66.7% fewer state checkpoints. The final 72K prompt endpoint is preserved. These results compare two settings of the revised candidate.

Six alternating CPU benchmark rounds, 200 calls each, at depth 256: Python 1.39 → 21.55 µs and Rust 9.13 → 9.98 µs (default → 32K). These are component planning measurements, with no serving throughput claim.

## Reproduction

Apply `complete-main.patch` to the publication base, then use that checkout as `source/` next to the runners under `tests/`. Use an environment with the SGLang test dependencies and a native Rust toolchain. `tests/final_runner.py` records the exact CPU, regression, planning and L3 commands. Its paths are those of the isolated validation directory and must be adjusted for another checkout. `lint_runner.py` records the full pre-commit command and installed Rust toolchain (1.92); the real native backend was built using Rust 1.90.

The L3 runner requires a B300-compatible CUDA environment and `mooncake_master`. It creates a separate local Mooncake instance; it is a component test, without a full model or cross-host RDMA. Original-head negative controls under `negative-control/` reproduce four failures (two defects × two backends); their source path points to the original-head snapshot, not this revised candidate.
