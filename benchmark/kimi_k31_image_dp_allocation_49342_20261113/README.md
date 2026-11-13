# PR #49342 receive-buffer coordination validation

Source: `b7cc46c583d2236c334fd281633218d2e858b98d`; previous revision: `5a6684f877e3a3759e9049342bf9af5bf46a2638`.

Host 1, two B300 GPUs, actual BF16 checkpoint, native GroupCoordinator/PyNccl. The new receive-allocation regression fails on the previous runtime with a peer entering payload communication. With the fix, both ranks stop before payload communication and the next encoding succeeds. The GPU path also catches an injected OutOfMemoryError and produces bit-identical features after recovery.

- 95 CPU tests and 11 ownership subtests passed; 19 real two-rank Gloo scenarios.
- Fix changed-line coverage: 8/8 (100%). Full PR: 199/213 (93.43%).
- Full default-stage pre-commit: 36 hooks passed, 2 had no matching files.
- Actual-weight 1/10/20-image cold/hot outputs match across revisions and ranks.
- Six balanced same-process A/B rounds without coverage: cold latency changes -0.43% / -0.22% / +0.06%. See raw samples and `results/paired-status.json`; hot-cache noise is not presented as a reliable speedup.

`validate.py` runs correctness, coverage and full pre-commit. `post_gpu.py` removes coverage instrumentation from the supplementary timing run. `paired_runner.py` / `paired_bench.py` provide the balanced A/B measurements. A one-pass single-image timing outlier and the instrumented measurements are retained separately.

`summary.json`, `manifest.json`, source snapshots, patches and raw results are included. `server-evidence.tar.gz` contains the original server export. This tests capturable allocation exceptions, not physical GPU-memory exhaustion, fatal asynchronous CUDA failures or request-level scheduler recovery. CP4/CP8 and full-service stress are unqualified.
