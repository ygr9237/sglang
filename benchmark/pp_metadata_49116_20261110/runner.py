"""Bounded isolated regression and A/B component checks; no model server."""
import json
import os
import signal
from pathlib import Path
import subprocess
import time

p = Path(__file__).parent
source = p / "source"
env = dict(os.environ, PYTHONPATH=str(source / "python"), OMP_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CARGO_BUILD_JOBS="4",
           MAX_JOBS="2", PATH="/root/.cargo/bin:" + os.environ["PATH"],
           RUSTUP_TOOLCHAIN="1.90", SGLANG_RUST_BUILD_MODE="auto",
           SGLANG_CACHE_DIR=str(p / "cache"), TORCH_EXTENSIONS_DIR=str(p / "torch_extensions"))
commands = [
    ("cpu-tests", "99", ["python", str(source / "test/registered/unit/managers/test_pp_output_exchange_order.py"), "-v"]),
    ("cuda-tests", "0", ["python", str(source / "test/registered/e2e/parallelism/test_pp_output_metadata.py"), "-v"]),
    ("sender", "0", ["python", str(p / "bench_pp_output_metadata.py"), "sender"]),
    ("reference", "0", ["python", str(p / "numeric_reference.py")]),
    ("profile", "0", ["python", str(p / "bench_pp_output_metadata.py"), "profile"]),
    ("profile-check", "99", ["python", str(p / "profile_check.py")]),
    ("transport", "0,6", ["python", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", str(p / "controlled_transport.py")]),
]
statuses = []
for name, devices, cmd in commands:
    env["CUDA_VISIBLE_DEVICES"] = devices
    started = time.monotonic()
    with (p / (name + ".log")).open("w") as log:
        job = subprocess.Popen(cmd, cwd=p, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
        (p / "active-job.json").write_text(json.dumps({"name": name, "pid": job.pid, "active": True}))
        try:
            rc = job.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(job.pid, signal.SIGTERM)
            try:
                job.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(job.pid, signal.SIGKILL)
                job.wait()
            rc = 124
    (p / "active-job.json").write_text(json.dumps({"name": name, "pid": job.pid, "active": False}))
    statuses.append({"name": name, "rc": rc, "elapsed_s": time.monotonic() - started})
    (p / "status.json").write_text(json.dumps(statuses, indent=2))
    if rc:
        break
