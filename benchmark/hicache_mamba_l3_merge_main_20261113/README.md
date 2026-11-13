# Snapshot interval after merging main — 2026-11-13

Validated against main 5792cf6e5e11aebc62d41cf19828f615a8fa7e0a, after a 12,494-blob source verification. Preserve both deferred-checkpoint and L3 prompt-boundary fields; keep checkpoint level 3 in the controlled fixture and initialize the upstream backup-token ledger.

36 policy tests, 94 related regressions and 45 subtests passed; 5 skipped. All applicable pre-commit hooks, workspace Rustfmt/Clippy and extra native Python-binding/inspection Clippy passed. The compatibility probe confirms both insertion flags survive serialization.

Four real Mooncake/B300 component round trips (Python/Rust × L3 interval 0/32768) passed, including zeroed device/host buffers, bitwise KDA output/state checks, partial hits and missing-object fallback. The latest controlled trace uses checkpoint level 3: nine 8K checkpoints over 72K tokens, 64-token KV pages. Sparse L3 writes retain 32K, 64K and the 72K endpoint (9→3), preserving 1152 KV pages. These compare settings of the current candidate.

Apply complete-main.patch to the stated main revision. The source directory contains the 14 tested files. Use PYTHONPATH=<checkout>/python and execute test/registered/unit/mem_cache/test_mamba_l3_snapshot_interval.py. Run the three regression files and selection recorded in validate-resume.py; its tool/environment paths must be adjusted for another machine. Put the checkout at source/ next to l3_roundtrip.py and plan_bench.py to reproduce component measurements; Mooncake/CUDA dependencies and mooncake_master are required. Full serving and cross-host RDMA were not measured.

The prior MLU gate failed before test execution because run-ci was missing; see gate-before-merge.log. No CI workflow or permission gate was changed.
