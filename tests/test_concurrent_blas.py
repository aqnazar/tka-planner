"""Two request threads doing large matrix products at once must not kill the server.

numpy's bundled OpenBLAS (threaded build) crashed the process with an access violation,
or hung it, when two Python threads entered a multithreaded BLAS routine together; the
planner server runs every request on its own thread (2026-11-08). Importing tka_planner
pins BLAS to one thread before numpy loads.
"""

import subprocess
import sys
import textwrap

CHILD = textwrap.dedent("""
    import threading, time
    import tka_planner  # noqa: F401  -- must configure BLAS before numpy loads
    import numpy as np

    def work(seed):
        rng = np.random.default_rng(seed)
        points = rng.normal(size=(1_000_000, 3))
        end = time.time() + 4.0
        while time.time() < end:
            points @ np.linalg.qr(rng.normal(size=(3, 3)))[0].T

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("survived")
""")


def test_concurrent_matmuls_do_not_kill_the_process():
    r = subprocess.run([sys.executable, "-X", "faulthandler", "-c", CHILD],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "survived" in r.stdout, f"exit {r.returncode}\n{r.stderr[-2000:]}"


def test_importing_the_package_pins_blas_to_one_thread():
    code = ("import os; os.environ.pop('OPENBLAS_NUM_THREADS', None); "
            "import tka_planner; print(os.environ.get('OPENBLAS_NUM_THREADS'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "1"
