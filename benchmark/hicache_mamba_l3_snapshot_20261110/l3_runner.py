import json
import os
import pathlib
import signal
import subprocess

p = pathlib.Path(__file__).parent
with (p/'l3-roundtrip.log').open('w') as log:
    job = subprocess.Popen(['python', str(p/'l3_roundtrip.py')], stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
    try:
        rc = job.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        os.killpg(job.pid, signal.SIGINT)
        try:
            job.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(job.pid, signal.SIGKILL)
            job.wait()
        rc = 124
(p/'l3-status.json').write_text(json.dumps({'rc': rc}))
