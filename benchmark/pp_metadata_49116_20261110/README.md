# PP metadata #49116: latest validation, 2026-11-10

Source: `11aa5e5b12b8160293b472eac12a8f1a28a57310`. Before fast paths: `a53050cb86f7bd51ed44b0ffda5add350d0a7ff7`. Numeric-only unpatched reference: `cbbc6b67f48008a8b739f4667d7b14abb89088cf`.

Hardware: Xeon 6776P, NVIDIA B300; Python 3.12.16, Torch 2.14.1+cu130, CUDA 13.0, NCCL 2.30.7. Isolated component tests; no model weights or serving benchmark.

- 7 CPU and 2 CUDA regressions passed, zero skips; changed-file pre-commit passed.
- Full 32K prompt top-20 Python lists, sender preparation + pickle P50: 173.981 -> 69.930 ms. The fresh unpatched numeric-only reference is 12.107 ms: traversal overhead is reduced, not eliminated. Payload remains 9,308,003 bytes.
- Actual B25 Python-list output leg P50/P95: 0.713/0.771 -> 0.565/0.593 ms. CPU tensor transport had round jitter and no consistent improvement; see all cases in summary.json.
- CUDA 32K view/B25: extra CPU clone count 50 -> 0; D2H count stays 50; serialized size stays 20,609 bytes.

Six alternating A/B rounds. Output transport uses real two-rank GroupCoordinator NCCL/Gloo plus completion ACK, with cyclic GC disabled: 1,500 samples/arm for small messages and 120 for large views. Sender timing excludes input creation and uses 3-200 adaptive samples per round. All values/dtypes/shapes are checked.

## Reproduce

In a compatible SGLang dependency environment, place this directory outside the source checkout, then:

```bash
git clone https://github.com/ygr9237/sglang.git source
git -C source checkout 11aa5e5b12b8160293b472eac12a8f1a28a57310
python runner.py
```

The runner uses GPU 0 for sender/CUDA tests and GPUs 0,6 for transport. It runs the two standard test entry points, sender A/B, numeric reference, profiling and transport checks. New outputs are written beside the scripts; recorded outputs are under results/. The three module arms are loaded from base/, fixed/ and unpatched/. precommit.log records the changed-file checks. source-manifest.json maps tested raw bytes and normalized Git files. Source paths in logs/traces are redacted to $RUN_ROOT; measurements are unchanged.
