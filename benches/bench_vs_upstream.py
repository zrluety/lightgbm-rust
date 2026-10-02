"""Wall-clock and memory benchmark: lightgbm-rust vs upstream LightGBM 4.7.0 (pip wheel).

Synthetic tasks (default: HIGGS-like binary, 1,000,000 rows x 28 float64
features; the ``sparse`` tasks pass a scipy CSR matrix). For every (task,
thread count) both engines run ``--repeats`` times, interleaved (upstream,
rust, upstream, rust, ...) so that host-load drift affects both alike; the
best time of each phase is kept. Each run is a fresh process, whose peak
resident set size (``ru_maxrss``) is recorded together with the RSS after the
input data was generated, so ``peak_rss_mb - data_rss_mb`` is the memory the
engine added. Both engines get identical parameters, including
``force_row_wise=True`` (upstream otherwise picks row- or col-wise by a timing
test). Writes ``tests/report/bench.json``.

    uv run --no-sync python benches/bench_vs_upstream.py [--rows N] [--iters N] [--repeats N]
        [--threads 1,4,8,16] [--tasks binary,regression,binary_255,binary_wide,sparse,wide_sparse]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    # name: (objective, extra params, cols, density (None: dense ndarray), rows divisor)
    "binary": ("binary", {}, 28, None, 1),
    "regression": ("regression", {}, 28, None, 1),
    "binary_255": ("binary", {"num_leaves": 255, "min_data_in_leaf": 100}, 28, None, 1),
    "binary_wide": ("binary", {}, 200, None, 1),
    "sparse": ("binary", {}, 100, 0.05, 1),
    "wide_sparse": ("binary", {}, 5000, 0.002, 4),
}
ENGINES = {"upstream lightgbm 4.7.0": "lightgbm", "lightgbm-rust": "lightgbm_rust"}


def make_data(rows: int, cols: int, objective: str = "binary", density: float | None = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    if density is None:
        X = rng.standard_normal((rows, cols))
        X[:, : cols // 4] = np.round(X[:, : cols // 4] * 3)  # some low-cardinality columns
        w = rng.standard_normal(cols)
        logit = X @ w * 0.3 + 0.5 * np.sin(X[:, 0] * X[:, 1]) + rng.standard_normal(rows) * 0.5
    else:
        import scipy.sparse as sp

        nnz = int(rows * cols * density)
        r = rng.integers(0, rows, size=nnz)
        c = np.minimum(rng.geometric(1.5 / cols, size=nnz) - 1, cols - 1)  # skewed column popularity
        v = np.round(rng.standard_normal(nnz) * 2, 1) + 0.05
        X = sp.csr_matrix((v, (r, c)), shape=(rows, cols))
        X.sum_duplicates()
        w = rng.standard_normal(cols)
        logit = X @ w * 0.5 + rng.standard_normal(rows) * 0.5
    y = (logit > 0).astype(np.float64) if objective == "binary" else logit
    return X, y


def worker(args) -> None:
    """One timed run in this process; prints a JSON line and saves predictions."""
    objective, extra, cols, density, div = TASKS[args.task]
    X, y = make_data(args.rows // div, cols, objective, density)
    data_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mod = __import__(ENGINES[args.engine])
    params = {"objective": objective, "num_threads": args.threads_one, "verbosity": -1, "force_row_wise": True,
              "seed": 1, **extra}
    t0 = time.perf_counter()
    ds = mod.Dataset(X, label=y, params=params, free_raw_data=False).construct()
    t1 = time.perf_counter()
    bst = mod.train(params, ds, num_boost_round=args.iters)
    t2 = time.perf_counter()
    pred = bst.predict(X)
    t3 = time.perf_counter()
    np.save(args.pred_out, pred)
    print(json.dumps({"times": [t1 - t0, t2 - t1, t3 - t2], "data_rss_kb": data_rss}), flush=True)


def run_once(engine: str, task: str, threads: int, rows: int, iters: int, pred_out: str):
    cmd = [sys.executable, __file__, "--worker", "--engine", engine, "--task", task, "--threads-one", str(threads),
           "--rows", str(rows), "--iters", str(iters), "--pred-out", pred_out]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    out = proc.stdout.read()
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        raise RuntimeError(f"{engine} {task} failed with {proc.returncode}")
    res = json.loads(out.strip().splitlines()[-1])
    return res["times"], res["data_rss_kb"] / 1024, usage.ru_maxrss / 1024, np.load(pred_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", default="")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--engine", help=argparse.SUPPRESS)
    ap.add_argument("--task", help=argparse.SUPPRESS)
    ap.add_argument("--threads-one", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--pred-out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return

    import lightgbm

    assert lightgbm.__version__ == "4.7.0", lightgbm.__version__
    ncpu = os.cpu_count() or 1
    threads = sorted({int(t) for t in args.threads.split(",") if t} or {1, 4, 8, ncpu})
    results = []
    tmp = tempfile.mkdtemp()
    for task in args.tasks.split(","):
        objective, extra, cols, density, div = TASKS[task]
        for t in threads:
            best = {name: [float("inf")] * 3 for name in ENGINES}
            peak = {name: 0.0 for name in ENGINES}
            data_rss = {name: 0.0 for name in ENGINES}
            preds = {}
            for _ in range(args.repeats):
                for name in ENGINES:
                    out = os.path.join(tmp, "pred.npy")
                    times, data_rss[name], rss, preds[name] = run_once(name, task, t, args.rows, args.iters, out)
                    best[name] = [min(b, x) for b, x in zip(best[name], times)]
                    peak[name] = max(peak[name], rss)
            names = list(ENGINES)
            diff = float(np.max(np.abs(preds[names[0]] - preds[names[1]])))
            for name in ENGINES:
                results.append({"task": task, "rows": args.rows // div, "cols": cols, "density": density,
                                "engine": name, "threads": t, "construct_s": best[name][0],
                                "train_s": best[name][1], "predict_s": best[name][2],
                                "data_rss_mb": round(data_rss[name], 1), "peak_rss_mb": round(peak[name], 1),
                                "max_abs_prediction_diff": diff})
                print(json.dumps(results[-1]), flush=True)

    out = {
        "rows": args.rows,
        "iterations": args.iters,
        "repeats": args.repeats,
        "tasks": {k: {"objective": TASKS[k][0], "params": TASKS[k][1], "cols": TASKS[k][2],
                      "density": TASKS[k][3], "rows": args.rows // TASKS[k][4]}
                  for k in args.tasks.split(",")},
        "host": f"{platform.platform()}, {ncpu} logical CPUs, Python {platform.python_version()}",
        "results": results,
        "note": (
            "Wall-clock seconds, best of the repeats with the engines interleaved, each run in a fresh "
            "process. peak_rss_mb is the process's peak resident set size (max over the repeats) and "
            "data_rss_mb its RSS after generating the input. Both engines use force_row_wise=True and "
            "identical parameters. Indicative only: a single host, no isolation from other load."
        ),
    }
    path = ROOT / "tests" / "report" / "bench.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
