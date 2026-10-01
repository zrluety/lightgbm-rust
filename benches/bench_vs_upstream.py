"""Wall-clock benchmark: lightgbm-rust vs upstream LightGBM 4.7.0 (pip wheel).

Synthetic dense tasks (default: HIGGS-like binary, 1,000,000 rows x 28
float64 features). For every (task, thread count) both engines run
``--repeats`` times, interleaved (upstream, rust, upstream, rust, ...) so that
host-load drift affects both alike; the best time of each phase is kept.
Both engines get identical parameters, including ``force_row_wise=True``
(upstream otherwise picks row- or col-wise by a timing test). Writes
``tests/report/bench.json``.

    uv run --no-sync python benches/bench_vs_upstream.py [--rows N] [--iters N] [--repeats N]
        [--threads 1,4,8,16] [--tasks binary,regression,binary_255,binary_wide]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import lightgbm
import numpy as np

import lightgbm_rust

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    # name: (objective, extra params, cols)
    "binary": ("binary", {}, 28),
    "regression": ("regression", {}, 28),
    "binary_255": ("binary", {"num_leaves": 255, "min_data_in_leaf": 100}, 28),
    "binary_wide": ("binary", {}, 200),
}


def make_data(rows: int, cols: int, objective: str = "binary", seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols))
    X[:, : cols // 4] = np.round(X[:, : cols // 4] * 3)  # some low-cardinality columns
    w = rng.standard_normal(cols)
    logit = X @ w * 0.3 + 0.5 * np.sin(X[:, 0] * X[:, 1]) + rng.standard_normal(rows) * 0.5
    y = (logit > 0).astype(np.float64) if objective == "binary" else logit
    return X, y


def run_once(mod, X, y, params, iters):
    t0 = time.perf_counter()
    ds = mod.Dataset(X, label=y, params=params, free_raw_data=False).construct()
    t1 = time.perf_counter()
    bst = mod.train(params, ds, num_boost_round=iters)
    t2 = time.perf_counter()
    pred = bst.predict(X)
    t3 = time.perf_counter()
    return [t1 - t0, t2 - t1, t3 - t2], pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", default="")
    ap.add_argument("--tasks", default=",".join(TASKS))
    args = ap.parse_args()

    assert lightgbm.__version__ == "4.7.0", lightgbm.__version__
    assert lightgbm is not lightgbm_rust
    ncpu = os.cpu_count() or 1
    threads = sorted({int(t) for t in args.threads.split(",") if t} or {1, 4, 8, ncpu})
    engines = (("upstream lightgbm 4.7.0", lightgbm), ("lightgbm-rust", lightgbm_rust))
    results = []
    for task in args.tasks.split(","):
        objective, extra, cols = TASKS[task]
        X, y = make_data(args.rows, cols, objective)
        for t in threads:
            params = {"objective": objective, "num_threads": t, "verbosity": -1, "force_row_wise": True,
                      "seed": 1, **extra}
            best = {name: [float("inf")] * 3 for name, _ in engines}
            preds = {}
            for _ in range(args.repeats):
                for name, mod in engines:
                    times, preds[name] = run_once(mod, X, y, params, args.iters)
                    best[name] = [min(b, x) for b, x in zip(best[name], times)]
            diff = float(np.max(np.abs(preds[engines[0][0]] - preds[engines[1][0]])))
            for name, _ in engines:
                results.append({"task": task, "cols": cols, "engine": name, "threads": t,
                                "construct_s": best[name][0], "train_s": best[name][1],
                                "predict_s": best[name][2], "max_abs_prediction_diff": diff})
                print(json.dumps(results[-1]), flush=True)

    out = {
        "rows": args.rows,
        "iterations": args.iters,
        "repeats": args.repeats,
        "tasks": {k: {"objective": TASKS[k][0], "params": TASKS[k][1], "cols": TASKS[k][2]}
                  for k in args.tasks.split(",")},
        "host": f"{platform.platform()}, {ncpu} logical CPUs, Python {platform.python_version()}",
        "results": results,
        "note": (
            "Wall-clock seconds, best of the repeats with the engines interleaved. Both engines use "
            "force_row_wise=True and identical parameters. Indicative only: a single host, no isolation "
            "from other load."
        ),
    }
    path = ROOT / "tests" / "report" / "bench.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
